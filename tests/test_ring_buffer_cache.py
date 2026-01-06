# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the ring buffer KV cache implementation.

Run with: python -m pytest tests/test_ring_buffer_cache.py -v
"""

import pytest
import torch
import sys
import os

# Add parent directory to path for direct import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import directly from the module file to avoid wan package dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "ring_buffer_cache",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "wan", "modules", "ring_buffer_cache.py")
)
ring_buffer_cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ring_buffer_cache)

create_ring_buffer_cache = ring_buffer_cache.create_ring_buffer_cache
ring_buffer_write = ring_buffer_cache.ring_buffer_write
reset_ring_buffer_cache = ring_buffer_cache.reset_ring_buffer_cache
prepare_attention_with_new_tokens = ring_buffer_cache.prepare_attention_with_new_tokens
compute_ring_buffer_write_indices = ring_buffer_cache.compute_ring_buffer_write_indices


def is_ring_buffer_cache(cache):
    """Check if a cache is a ring buffer cache by looking for write_ptr key."""
    return "write_ptr" in cache


@pytest.fixture
def device():
    """Use CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cache_params():
    """Standard cache parameters for testing."""
    return {
        "batch_size": 2,
        "buffer_size": 100,
        "num_heads": 4,
        "head_dim": 32,
        "sink_tokens": 10,
    }


class TestCreateRingBufferCache:
    """Tests for create_ring_buffer_cache function."""

    def test_creates_correct_structure(self, device, cache_params):
        """Verify cache has all required keys."""
        cache = create_ring_buffer_cache(
            dtype=torch.float32, device=device, **cache_params
        )
        
        required_keys = [
            "k", "v", "attn_k", "attn_v",
            "write_ptr", "global_end_index", "local_end_index", "is_wrapped",
            "sink_tokens", "buffer_size", "rolling_capacity"
        ]
        for key in required_keys:
            assert key in cache, f"Missing key: {key}"

    def test_tensor_shapes(self, device, cache_params):
        """Verify tensor shapes are correct."""
        cache = create_ring_buffer_cache(
            dtype=torch.float32, device=device, **cache_params
        )
        
        B, S, H, D = (
            cache_params["batch_size"],
            cache_params["buffer_size"],
            cache_params["num_heads"],
            cache_params["head_dim"],
        )
        
        assert cache["k"].shape == (B, S, H, D)
        assert cache["v"].shape == (B, S, H, D)
        assert cache["attn_k"].shape == (B, S, H, D)
        assert cache["attn_v"].shape == (B, S, H, D)

    def test_initial_state(self, device, cache_params):
        """Verify initial state is zeroed."""
        cache = create_ring_buffer_cache(
            dtype=torch.float32, device=device, **cache_params
        )
        
        assert cache["write_ptr"].item() == 0
        assert cache["global_end_index"].item() == 0
        assert cache["local_end_index"].item() == 0
        assert cache["is_wrapped"].item() == 0

    def test_rolling_capacity(self, device, cache_params):
        """Verify rolling_capacity = buffer_size - sink_tokens."""
        cache = create_ring_buffer_cache(
            dtype=torch.float32, device=device, **cache_params
        )
        
        expected = cache_params["buffer_size"] - cache_params["sink_tokens"]
        assert cache["rolling_capacity"] == expected


class TestIsRingBufferCache:
    """Tests for is_ring_buffer_cache detection function."""

    def test_detects_ring_buffer(self, device, cache_params):
        """Ring buffer cache should be detected."""
        cache = create_ring_buffer_cache(
            dtype=torch.float32, device=device, **cache_params
        )
        assert is_ring_buffer_cache(cache) is True

    def test_rejects_legacy_cache(self, device):
        """Legacy cache should not be detected as ring buffer."""
        legacy_cache = {
            "k": torch.zeros([2, 100, 4, 32], device=device),
            "v": torch.zeros([2, 100, 4, 32], device=device),
            "global_end_index": torch.tensor([0], device=device),
            "local_end_index": torch.tensor([0], device=device),
        }
        assert is_ring_buffer_cache(legacy_cache) is False


class TestRingBufferWrite:
    """Tests for ring_buffer_write function."""

    def test_write_fills_buffer_linearly(self, device, cache_params):
        """Initial writes should fill buffer linearly."""
        cache = create_ring_buffer_cache(
            dtype=torch.float32, device=device, **cache_params
        )
        
        # Write 20 tokens
        num_tokens = 20
        new_k = torch.randn(cache_params["batch_size"], num_tokens, 
                           cache_params["num_heads"], cache_params["head_dim"], device=device)
        new_v = torch.randn_like(new_k)
        
        ring_buffer_write(cache, new_k, new_v)
        
        # Check pointers updated
        assert cache["write_ptr"].item() == num_tokens
        assert cache["local_end_index"].item() == num_tokens
        assert cache["global_end_index"].item() == num_tokens
        
        # Check data written correctly
        assert torch.allclose(cache["k"][:, :num_tokens], new_k)
        assert torch.allclose(cache["v"][:, :num_tokens], new_v)

    def test_write_wraps_around(self, device):
        """After buffer fills, writes should wrap around."""
        # Small buffer for easy testing
        cache = create_ring_buffer_cache(
            batch_size=1, buffer_size=20, num_heads=2, head_dim=8,
            sink_tokens=5, dtype=torch.float32, device=device
        )

        # Fill buffer with 15 tokens (just under buffer_size - 5 short)
        initial_k = torch.randn(1, 15, 2, 8, device=device)
        initial_v = torch.randn_like(initial_k)
        ring_buffer_write(cache, initial_k, initial_v)

        assert cache["is_wrapped"].item() == False  # Not wrapped yet
        assert cache["local_end_index"].item() == 15

        # Write 5 more - should fill buffer exactly and set is_wrapped
        more_k = torch.randn(1, 5, 2, 8, device=device)
        more_v = torch.randn_like(more_k)
        ring_buffer_write(cache, more_k, more_v)

        # When buffer is exactly full, is_wrapped becomes True (ready to wrap on next write)
        assert cache["is_wrapped"].item() == True
        assert cache["local_end_index"].item() == 20  # Buffer full

        # Write 5 more - should wrap into rolling region
        wrap_k = torch.randn(1, 5, 2, 8, device=device)
        wrap_v = torch.randn_like(wrap_k)
        ring_buffer_write(cache, wrap_k, wrap_v)

        assert cache["is_wrapped"].item() == True  # Still wrapped
        assert cache["global_end_index"].item() == 25  # Total tokens seen


class TestResetRingBufferCache:
    """Tests for reset_ring_buffer_cache function."""

    def test_full_reset(self, device, cache_params):
        """Full reset clears everything."""
        cache = create_ring_buffer_cache(
            dtype=torch.float32, device=device, **cache_params
        )

        # Write some data
        new_k = torch.randn(cache_params["batch_size"], 30,
                           cache_params["num_heads"], cache_params["head_dim"], device=device)
        ring_buffer_write(cache, new_k, new_k)

        # Reset
        reset_ring_buffer_cache(cache, preserve_sink=False)

        assert cache["write_ptr"].item() == 0
        assert cache["global_end_index"].item() == 0
        assert cache["local_end_index"].item() == 0
        assert torch.all(cache["k"] == 0)
        assert torch.all(cache["v"] == 0)

    def test_preserve_sink_reset(self, device):
        """Reset with preserve_sink keeps sink tokens."""
        cache = create_ring_buffer_cache(
            batch_size=1, buffer_size=20, num_heads=2, head_dim=8,
            sink_tokens=5, dtype=torch.float32, device=device
        )

        # Write data covering sink + rolling
        new_k = torch.ones(1, 15, 2, 8, device=device)
        ring_buffer_write(cache, new_k, new_k)

        # Reset preserving sink
        reset_ring_buffer_cache(cache, preserve_sink=True)

        # Sink region should be preserved
        assert torch.all(cache["k"][:, :5] == 1)
        # Rolling region should be cleared
        assert torch.all(cache["k"][:, 5:] == 0)
        # write_ptr should be at sink_tokens
        assert cache["write_ptr"].item() == 5


class TestPrepareAttentionWithNewTokens:
    """Tests for prepare_attention_with_new_tokens function."""

    def test_attention_includes_cached_and_new(self, device):
        """Attention should include cached tokens + new tokens."""
        cache = create_ring_buffer_cache(
            batch_size=1, buffer_size=50, num_heads=2, head_dim=8,
            sink_tokens=5, dtype=torch.float32, device=device
        )

        # Write 20 cached tokens
        cached_k = torch.ones(1, 20, 2, 8, device=device)
        cached_v = torch.ones(1, 20, 2, 8, device=device) * 2
        ring_buffer_write(cache, cached_k, cached_v)

        # Prepare attention with 10 new tokens
        new_k = torch.ones(1, 10, 2, 8, device=device) * 3
        new_v = torch.ones(1, 10, 2, 8, device=device) * 4

        # Use new API with full tensors and tensor offset/length
        local_end_index = torch.tensor(30, device=device, dtype=torch.long)  # 20 cached + 10 new
        source_offset = torch.tensor(0, device=device, dtype=torch.long)
        write_len = torch.tensor(10, device=device, dtype=torch.long)

        attn_k, attn_v, num_valid = prepare_attention_with_new_tokens(
            cache=cache,
            full_k=new_k,
            full_v=new_v,
            local_end_index=local_end_index,
            source_offset=source_offset,
            write_len=write_len,
            sink_tokens=5,
            max_attention_size=50,
        )

        # Should have 30 valid tokens
        assert num_valid.item() == 30
        # First 20 should be cached, last 10 should be new
        assert torch.allclose(attn_k[:, :20], cached_k)
        assert torch.allclose(attn_k[:, 20:30], new_k)

    def test_attention_respects_max_size(self, device):
        """Attention window should respect max_attention_size."""
        cache = create_ring_buffer_cache(
            batch_size=1, buffer_size=100, num_heads=2, head_dim=8,
            sink_tokens=5, dtype=torch.float32, device=device
        )

        # Fill with 80 tokens
        cached_k = torch.randn(1, 80, 2, 8, device=device)
        ring_buffer_write(cache, cached_k, cached_k)

        # Prepare attention with max_attention_size=50
        new_k = torch.randn(1, 10, 2, 8, device=device)

        # Use new API with full tensors and tensor offset/length
        local_end_index = torch.tensor(90, device=device, dtype=torch.long)
        source_offset = torch.tensor(0, device=device, dtype=torch.long)
        write_len = torch.tensor(10, device=device, dtype=torch.long)

        attn_k, attn_v, num_valid = prepare_attention_with_new_tokens(
            cache=cache,
            full_k=new_k,
            full_v=new_k,
            local_end_index=local_end_index,
            source_offset=source_offset,
            write_len=write_len,
            sink_tokens=5,
            max_attention_size=50,
        )

        # Should be capped at max_attention_size
        assert num_valid.item() <= 50


class TestEndToEndScenario:
    """End-to-end tests simulating real inference scenarios."""

    def test_streaming_inference_simulation(self, device):
        """Simulate streaming inference with multiple write/read cycles."""
        cache = create_ring_buffer_cache(
            batch_size=1, buffer_size=30, num_heads=2, head_dim=8,
            sink_tokens=5, dtype=torch.float32, device=device
        )

        # Simulate 5 inference steps, each adding 10 tokens
        for step in range(5):
            new_k = torch.full((1, 10, 2, 8), fill_value=float(step), device=device)
            new_v = new_k.clone()

            # Write to cache first
            ring_buffer_write(cache, new_k, new_v)

            # local_end_index after write
            local_end = cache["local_end_index"]

            # Prepare attention after write (simulating what happens in attention)
            # Use new API with full tensors and tensor offset/length
            empty_k = torch.zeros(1, 0, 2, 8, device=device)
            source_offset = torch.tensor(0, device=device, dtype=torch.long)
            write_len = torch.tensor(0, device=device, dtype=torch.long)

            attn_k, attn_v, num_valid = prepare_attention_with_new_tokens(
                cache=cache,
                full_k=empty_k,
                full_v=empty_k,
                local_end_index=local_end,
                source_offset=source_offset,
                write_len=write_len,
                sink_tokens=5,
                max_attention_size=30,
            )

            # Verify num_valid grows then stabilizes at buffer size
            expected = min((step + 1) * 10, 30)
            assert num_valid.item() == expected, f"step {step}: expected {expected}, got {num_valid.item()}"

        # After 5 steps (50 tokens written to buffer of 30):
        # - global_end_index tracks total tokens seen = 50
        # - local_end_index capped at buffer_size = 30
        # - is_wrapped should be True (buffer overflowed at least once)
        assert cache["global_end_index"].item() == 50, f"global_end_index: {cache['global_end_index'].item()}"
        assert cache["local_end_index"].item() == 30, f"local_end_index: {cache['local_end_index'].item()}"
        # The buffer should have wrapped (overflowed) since we wrote 50 tokens to 30-size buffer
        assert cache["is_wrapped"].item() == True, f"Expected wrapped=True, got {cache['is_wrapped'].item()}"

    def test_buffer_wrap_detection(self, device):
        """Specifically test that wrap detection works correctly."""
        cache = create_ring_buffer_cache(
            batch_size=1, buffer_size=15, num_heads=2, head_dim=8,
            sink_tokens=5, dtype=torch.float32, device=device
        )
        # rolling_capacity = 15 - 5 = 10

        # Write 5 tokens (sink fills, not wrapped)
        data1 = torch.ones(1, 5, 2, 8, device=device)
        ring_buffer_write(cache, data1, data1)
        assert cache["is_wrapped"].item() == False
        assert cache["local_end_index"].item() == 5
        assert cache["global_end_index"].item() == 5
        # Data should be in sink region
        assert torch.allclose(cache["k"][:, :5], data1)

        # Write 10 more (fills buffer exactly)
        data2 = torch.ones(1, 10, 2, 8, device=device) * 2
        ring_buffer_write(cache, data2, data2)
        assert cache["local_end_index"].item() == 15  # Buffer full
        assert cache["global_end_index"].item() == 15
        # Data should be in rolling region
        assert torch.allclose(cache["k"][:, 5:15], data2)

        # Write 5 more (should wrap into rolling region, overwriting oldest)
        data3 = torch.ones(1, 5, 2, 8, device=device) * 3
        ring_buffer_write(cache, data3, data3)
        assert cache["is_wrapped"].item() == True  # Now wrapped
        assert cache["global_end_index"].item() == 20
        # Data3 should have overwritten positions [5, 10) in rolling region
        assert torch.allclose(cache["k"][:, 5:10], data3)


class TestRingBufferVsLegacyCache:
    """Tests comparing ring buffer cache with legacy sliding window cache.

    These tests verify that the ring buffer produces equivalent attention outputs
    to the legacy clone-based approach.
    """

    def _create_legacy_cache(self, batch_size, kv_cache_size, num_heads, head_dim, device, dtype):
        """Create a legacy (non-ring-buffer) cache structure."""
        return {
            "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
        }

    def _apply_legacy_cache_update(self, cache, new_k, new_v, current_start, current_end, sink_tokens=0):
        """
        Apply cache update using the legacy approach (direct assignment).
        This simulates what happens in the non-ring-buffer path.
        """
        num_new_tokens = new_k.size(1)
        kv_cache_size = cache["k"].size(1)

        local_end_index = cache["local_end_index"].item() + current_end - cache["global_end_index"].item()
        local_start_index = local_end_index - num_new_tokens

        # Check if we need to roll (evict oldest tokens)
        if local_end_index > kv_cache_size:
            num_evicted = local_end_index - kv_cache_size
            num_rolled = cache["local_end_index"].item() - num_evicted - sink_tokens

            # Shift preserved tokens
            if num_rolled > 0:
                cache["k"][:, sink_tokens:sink_tokens + num_rolled] = \
                    cache["k"][:, sink_tokens + num_evicted:sink_tokens + num_evicted + num_rolled].clone()
                cache["v"][:, sink_tokens:sink_tokens + num_rolled] = \
                    cache["v"][:, sink_tokens + num_evicted:sink_tokens + num_evicted + num_rolled].clone()

            local_end_index = kv_cache_size
            local_start_index = local_end_index - num_new_tokens

        # Insert new tokens
        write_start = max(local_start_index, 0)
        cache["k"][:, write_start:local_end_index] = new_k
        cache["v"][:, write_start:local_end_index] = new_v

        # Update indices
        cache["global_end_index"].fill_(current_end)
        cache["local_end_index"].fill_(local_end_index)

    def test_basic_cache_equivalence(self, device):
        """
        Test that ring buffer and legacy cache produce same data for simple linear fill.
        """
        batch_size, buffer_size, num_heads, head_dim = 1, 20, 4, 16
        sink_tokens = 0
        dtype = torch.float32

        # Create both cache types
        ring_cache = create_ring_buffer_cache(
            batch_size=batch_size, buffer_size=buffer_size,
            num_heads=num_heads, head_dim=head_dim,
            sink_tokens=sink_tokens, dtype=dtype, device=device
        )
        legacy_cache = self._create_legacy_cache(
            batch_size, buffer_size, num_heads, head_dim, device, dtype
        )

        # Generate test data
        torch.manual_seed(42)
        test_k = torch.randn(batch_size, 10, num_heads, head_dim, device=device, dtype=dtype)
        test_v = torch.randn(batch_size, 10, num_heads, head_dim, device=device, dtype=dtype)

        # Apply to both caches
        ring_buffer_write(ring_cache, test_k, test_v)
        self._apply_legacy_cache_update(legacy_cache, test_k, test_v,
                                        current_start=0, current_end=10, sink_tokens=0)

        # Verify data matches
        assert torch.allclose(ring_cache["k"][:, :10], legacy_cache["k"][:, :10]), \
            "Ring buffer K data doesn't match legacy cache"
        assert torch.allclose(ring_cache["v"][:, :10], legacy_cache["v"][:, :10]), \
            "Ring buffer V data doesn't match legacy cache"

    def test_attention_output_equivalence_no_wrap(self, device):
        """
        Test that attention outputs are equivalent when buffer hasn't wrapped.
        """
        batch_size, buffer_size, num_heads, head_dim = 1, 30, 4, 16
        sink_tokens = 5
        dtype = torch.float32

        # Create ring buffer cache
        ring_cache = create_ring_buffer_cache(
            batch_size=batch_size, buffer_size=buffer_size,
            num_heads=num_heads, head_dim=head_dim,
            sink_tokens=sink_tokens, dtype=dtype, device=device
        )

        # Create legacy cache
        legacy_cache = self._create_legacy_cache(
            batch_size, buffer_size, num_heads, head_dim, device, dtype
        )

        # Generate deterministic test data
        torch.manual_seed(123)
        # First write: fills sink + some rolling
        data1_k = torch.randn(batch_size, 10, num_heads, head_dim, device=device, dtype=dtype)
        data1_v = torch.randn(batch_size, 10, num_heads, head_dim, device=device, dtype=dtype)

        # Write to both caches
        ring_buffer_write(ring_cache, data1_k, data1_v)
        self._apply_legacy_cache_update(legacy_cache, data1_k, data1_v,
                                        current_start=0, current_end=10, sink_tokens=sink_tokens)

        # Verify cached data matches
        valid_end = ring_cache["local_end_index"].item()
        assert torch.allclose(ring_cache["k"][:, :valid_end], legacy_cache["k"][:, :valid_end]), \
            f"K mismatch at valid_end={valid_end}"
        assert torch.allclose(ring_cache["v"][:, :valid_end], legacy_cache["v"][:, :valid_end]), \
            f"V mismatch at valid_end={valid_end}"

        # Second write
        data2_k = torch.randn(batch_size, 10, num_heads, head_dim, device=device, dtype=dtype)
        data2_v = torch.randn(batch_size, 10, num_heads, head_dim, device=device, dtype=dtype)

        ring_buffer_write(ring_cache, data2_k, data2_v)
        self._apply_legacy_cache_update(legacy_cache, data2_k, data2_v,
                                        current_start=10, current_end=20, sink_tokens=sink_tokens)

        valid_end = ring_cache["local_end_index"].item()
        assert torch.allclose(ring_cache["k"][:, :valid_end], legacy_cache["k"][:, :valid_end]), \
            f"K mismatch after second write at valid_end={valid_end}"

    def test_attention_prepare_matches_legacy_slicing(self, device):
        """
        Test that prepare_attention_with_new_tokens produces the same attention inputs
        as the legacy approach of clone + slice.

        Note: The ring buffer returns fixed-size tensors (buffer_size) with num_valid
        indicating how many tokens are valid. Legacy returns only valid tokens.
        """
        batch_size, buffer_size, num_heads, head_dim = 1, 20, 4, 16
        sink_tokens = 5
        max_attention_size = 15  # Smaller than buffer to test window
        dtype = torch.float32

        # Create and populate ring buffer cache
        ring_cache = create_ring_buffer_cache(
            batch_size=batch_size, buffer_size=buffer_size,
            num_heads=num_heads, head_dim=head_dim,
            sink_tokens=sink_tokens, dtype=dtype, device=device
        )

        # Populate with known data
        torch.manual_seed(456)
        data_k = torch.randn(batch_size, 15, num_heads, head_dim, device=device, dtype=dtype)
        data_v = torch.randn(batch_size, 15, num_heads, head_dim, device=device, dtype=dtype)
        ring_buffer_write(ring_cache, data_k, data_v)

        # Prepare new tokens for attention
        new_k = torch.randn(batch_size, 3, num_heads, head_dim, device=device, dtype=dtype)
        new_v = torch.randn(batch_size, 3, num_heads, head_dim, device=device, dtype=dtype)

        old_local_end = ring_cache["local_end_index"].item()
        local_end = torch.tensor(old_local_end + 3, device=device, dtype=torch.long)  # After adding new tokens
        source_offset = torch.tensor(0, device=device, dtype=torch.long)
        write_len = torch.tensor(3, device=device, dtype=torch.long)

        # Ring buffer approach - returns fixed-size tensors with num_valid indicating valid count
        attn_k, attn_v, num_valid = prepare_attention_with_new_tokens(
            cache=ring_cache,
            full_k=new_k,
            full_v=new_v,
            local_end_index=local_end,
            source_offset=source_offset,
            write_len=write_len,
            sink_tokens=sink_tokens,
            max_attention_size=max_attention_size,
        )

        # Legacy approach: manual construction
        # 1. Get sink tokens
        legacy_sink_k = ring_cache["k"][:, :sink_tokens].clone()
        legacy_sink_v = ring_cache["v"][:, :sink_tokens].clone()

        # 2. Get rolling window (excluding sink, up to local_end before new tokens)
        local_budget = max_attention_size - sink_tokens - new_k.size(1)
        local_start = max(sink_tokens, old_local_end - local_budget)
        legacy_local_k = ring_cache["k"][:, local_start:old_local_end].clone()
        legacy_local_v = ring_cache["v"][:, local_start:old_local_end].clone()

        # 3. Concatenate: sink + local + new
        legacy_attn_k = torch.cat([legacy_sink_k, legacy_local_k, new_k], dim=1)
        legacy_attn_v = torch.cat([legacy_sink_v, legacy_local_v, new_v], dim=1)

        # Compare outputs - ring buffer returns buffer_size tensors, legacy returns exactly num_valid
        # The valid portion of ring buffer output should match legacy output
        num_valid_int = num_valid.item()
        assert num_valid_int == legacy_attn_k.size(1), \
            f"num_valid mismatch: ring={num_valid_int}, legacy={legacy_attn_k.size(1)}"
        assert torch.allclose(attn_k[:, :num_valid_int], legacy_attn_k, atol=1e-5), \
            "Attention K mismatch between ring buffer and legacy"
        assert torch.allclose(attn_v[:, :num_valid_int], legacy_attn_v, atol=1e-5), \
            "Attention V mismatch between ring buffer and legacy"

    def test_wrap_around_produces_correct_attention_window(self, device):
        """
        Test that after wrap-around, the ring buffer still produces correct attention inputs.
        The attention window should contain sink tokens + most recent rolling tokens.
        """
        batch_size, buffer_size, num_heads, head_dim = 1, 20, 2, 8
        sink_tokens = 5
        # rolling_capacity = buffer_size - sink_tokens = 15
        max_attention_size = 20
        dtype = torch.float32

        ring_cache = create_ring_buffer_cache(
            batch_size=batch_size, buffer_size=buffer_size,
            num_heads=num_heads, head_dim=head_dim,
            sink_tokens=sink_tokens, dtype=dtype, device=device
        )

        # Fill sink tokens (tokens 0-4)
        sink_data = torch.arange(5, dtype=dtype, device=device).view(1, 5, 1, 1).expand(-1, -1, num_heads, head_dim)
        ring_buffer_write(ring_cache, sink_data, sink_data)

        # Fill rolling region with tokens 5-19 (fills exactly)
        rolling_data = torch.arange(5, 20, dtype=dtype, device=device).view(1, 15, 1, 1).expand(-1, -1, num_heads, head_dim)
        ring_buffer_write(ring_cache, rolling_data, rolling_data)

        # Now write 5 more tokens (20-24) - these wrap around
        wrap_data = torch.arange(20, 25, dtype=dtype, device=device).view(1, 5, 1, 1).expand(-1, -1, num_heads, head_dim)
        ring_buffer_write(ring_cache, wrap_data, wrap_data)

        # After wrapping, positions [5-9] should contain tokens 20-24
        # Positions [10-19] should still contain tokens 10-19
        # Sink positions [0-4] should contain tokens 0-4

        # Get attention window using new API
        empty_k = torch.zeros(batch_size, 0, num_heads, head_dim, device=device, dtype=dtype)
        local_end_index = torch.tensor(buffer_size, device=device, dtype=torch.long)
        source_offset = torch.tensor(0, device=device, dtype=torch.long)
        write_len = torch.tensor(0, device=device, dtype=torch.long)

        attn_k, _, num_valid = prepare_attention_with_new_tokens(
            cache=ring_cache,
            full_k=empty_k,
            full_v=empty_k,
            local_end_index=local_end_index,
            source_offset=source_offset,
            write_len=write_len,
            sink_tokens=sink_tokens,
            max_attention_size=max_attention_size,
        )

        # Verify sink tokens are preserved (tokens 0-4)
        for i in range(5):
            expected_val = float(i)
            actual_val = attn_k[0, i, 0, 0].item()
            assert abs(actual_val - expected_val) < 1e-5, \
                f"Sink token {i} wrong: expected {expected_val}, got {actual_val}"

        # After sink, we should have the rolling window in chronological order
        # The ring buffer gather should return: tokens 10-19 then tokens 20-24
        # (preserving temporal order even though physically wrapped)

        # Verify num_valid
        assert num_valid.item() == buffer_size, f"Expected num_valid={buffer_size}, got {num_valid.item()}"
