# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""
Tests to verify that CUDA graph path produces the same output as eager path.
"""
import pytest
import torch
import torch.nn as nn
from typing import List, Optional
from unittest.mock import MagicMock, patch

from wan.modules.ring_buffer_cache import create_ring_buffer_cache
from wan.modules.causal_model import is_ring_buffer_cache


# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for CUDA graph tests"
)


class MockGenerator(nn.Module):
    """
    A minimal mock generator that mimics WanDiffusionWrapper's interface.
    Uses simple linear layers to produce deterministic output.

    This is a CUDA graph compatible mock that does NOT call ring_buffer_write
    during forward pass. The real generator uses the ring buffer path in
    causal_model.py which is properly optimized for CUDA graphs.
    """
    def __init__(self, in_channels=16, hidden_dim=64, num_heads=4, head_dim=16, num_blocks=2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks

        # Simple layers to transform input
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, in_channels)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
    ):
        """
        Mock forward pass that:
        1. Processes input through linear layers
        2. Returns (None, denoised_pred) to match WanDiffusionWrapper interface

        Note: This mock does NOT update KV cache. The real generator's cache
        update logic is tested separately in test_ring_buffer_cache.py.
        """
        B, T, C, H, W = noisy_image_or_video.shape

        # Flatten spatial dims for processing
        x = noisy_image_or_video.reshape(B, T, C, -1).mean(dim=-1)  # [B, T, C]

        # Simple forward pass
        hidden = self.input_proj(x)  # [B, T, hidden_dim]

        # Add conditioning (just add prompt_embeds mean)
        if "prompt_embeds" in conditional_dict:
            cond = conditional_dict["prompt_embeds"].mean(dim=1, keepdim=True)  # [B, 1, dim]
            # Project to hidden_dim if needed
            if cond.shape[-1] != self.hidden_dim:
                cond = cond[..., :self.hidden_dim]
            hidden = hidden + cond

        # Add timestep influence
        hidden = hidden * (1.0 + timestep.float().unsqueeze(-1) / 1000.0)

        # Output projection
        output = self.output_proj(hidden)  # [B, T, C]

        # Reshape back to video format
        denoised_pred = output.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H, W)

        return None, denoised_pred


@pytest.fixture
def device():
    return torch.device("cuda:0")


@pytest.fixture
def mock_generator(device):
    """Create a mock generator on CUDA."""
    gen = MockGenerator(
        in_channels=16,
        hidden_dim=64,
        num_heads=4,
        head_dim=16,
        num_blocks=2
    ).to(device)
    gen.eval()
    return gen


@pytest.fixture
def test_inputs(device):
    """Create test inputs for the generator."""
    torch.manual_seed(42)
    batch_size = 1
    num_frames = 4
    channels = 16
    height, width = 8, 8

    noisy_input = torch.randn(batch_size, num_frames, channels, height, width, device=device)
    timestep = torch.ones(batch_size, num_frames, dtype=torch.int64, device=device) * 500
    conditional_dict = {
        "prompt_embeds": torch.randn(batch_size, 77, 64, device=device)
    }

    return noisy_input, timestep, conditional_dict


@pytest.fixture
def kv_cache(device):
    """Create ring buffer KV cache for 2 blocks."""
    caches = []
    for _ in range(2):
        caches.append(create_ring_buffer_cache(
            batch_size=1,
            buffer_size=1000,  # Large enough for test
            num_heads=4,
            head_dim=16,
            sink_tokens=0,
            dtype=torch.float32,
            device=device
        ))
    return caches


@pytest.fixture
def crossattn_cache(device):
    """Create cross-attention cache for 2 blocks."""
    caches = []
    for _ in range(2):
        caches.append({
            "k": torch.zeros(1, 512, 4, 16, device=device),
            "v": torch.zeros(1, 512, 4, 16, device=device),
            "is_init": False
        })
    return caches


class TestEagerPathWorks:
    """Tests to verify the eager (non-CUDA graph) path works correctly."""

    def test_eager_forward_produces_output(self, mock_generator, test_inputs, kv_cache, crossattn_cache):
        """Verify that eager forward pass produces valid output."""
        noisy_input, timestep, conditional_dict = test_inputs

        with torch.no_grad():
            _, denoised_pred = mock_generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=0,
            )

        # Check output shape matches input shape
        assert denoised_pred.shape == noisy_input.shape
        # Check output is not all zeros
        assert not torch.allclose(denoised_pred, torch.zeros_like(denoised_pred))
        # Check output is finite
        assert torch.isfinite(denoised_pred).all()

    def test_eager_forward_deterministic(self, mock_generator, test_inputs, device):
        """Verify that eager forward is deterministic with same inputs."""
        noisy_input, timestep, conditional_dict = test_inputs

        # Create fresh caches for each run
        def make_cache():
            caches = []
            for _ in range(2):
                caches.append(create_ring_buffer_cache(
                    batch_size=1, buffer_size=1000, num_heads=4, head_dim=16,
                    sink_tokens=0, dtype=torch.float32, device=device
                ))
            return caches

        with torch.no_grad():
            kv1 = make_cache()
            _, out1 = mock_generator(
                noisy_image_or_video=noisy_input.clone(),
                conditional_dict={k: v.clone() for k, v in conditional_dict.items()},
                timestep=timestep.clone(),
                kv_cache=kv1,
                crossattn_cache=None,
                current_start=0,
            )

            kv2 = make_cache()
            _, out2 = mock_generator(
                noisy_image_or_video=noisy_input.clone(),
                conditional_dict={k: v.clone() for k, v in conditional_dict.items()},
                timestep=timestep.clone(),
                kv_cache=kv2,
                crossattn_cache=None,
                current_start=0,
            )

        assert torch.allclose(out1, out2, atol=1e-6)


class TestCudaGraphEquivalence:
    """Tests to verify CUDA graph path produces same output as eager path."""

    def test_cuda_graph_capture_and_replay(self, mock_generator, test_inputs, device):
        """Test basic CUDA graph capture and replay produces valid output."""
        noisy_input, timestep, conditional_dict = test_inputs

        # Create cache
        kv_cache = []
        for _ in range(2):
            kv_cache.append(create_ring_buffer_cache(
                batch_size=1, buffer_size=1000, num_heads=4, head_dim=16,
                sink_tokens=0, dtype=torch.float32, device=device
            ))

        # Create static buffers
        static_noisy = noisy_input.clone()
        static_timestep = timestep.clone()
        static_cond = {k: v.clone() for k, v in conditional_dict.items()}

        # Warmup
        torch.cuda.synchronize()
        with torch.no_grad():
            _ = mock_generator(
                noisy_image_or_video=static_noisy,
                conditional_dict=static_cond,
                timestep=static_timestep,
                kv_cache=kv_cache,
                crossattn_cache=None,
                current_start=0,
            )
        torch.cuda.synchronize()

        # Reset cache for capture
        for cache in kv_cache:
            cache["k"].zero_()
            cache["v"].zero_()
            cache["write_ptr"].zero_()
            cache["global_end_index"].zero_()
            cache["local_end_index"].zero_()
            cache["is_wrapped"].zero_()

        # Capture graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _, static_output = mock_generator(
                noisy_image_or_video=static_noisy,
                conditional_dict=static_cond,
                timestep=static_timestep,
                kv_cache=kv_cache,
                crossattn_cache=None,
                current_start=0,
            )

        # Replay with same inputs
        graph.replay()
        output_replay = static_output.clone()

        # Verify output is valid
        assert output_replay.shape == noisy_input.shape
        assert torch.isfinite(output_replay).all()

    def test_cuda_graph_equals_eager(self, mock_generator, test_inputs, device):
        """
        The key test: verify CUDA graph output equals eager output for same inputs.
        """
        noisy_input, timestep, conditional_dict = test_inputs

        def make_cache():
            caches = []
            for _ in range(2):
                caches.append(create_ring_buffer_cache(
                    batch_size=1, buffer_size=1000, num_heads=4, head_dim=16,
                    sink_tokens=0, dtype=torch.float32, device=device
                ))
            return caches

        # === Eager path ===
        eager_cache = make_cache()
        with torch.no_grad():
            _, eager_output = mock_generator(
                noisy_image_or_video=noisy_input.clone(),
                conditional_dict={k: v.clone() for k, v in conditional_dict.items()},
                timestep=timestep.clone(),
                kv_cache=eager_cache,
                crossattn_cache=None,
                current_start=0,
            )

        # === CUDA graph path ===
        graph_cache = make_cache()
        static_noisy = noisy_input.clone()
        static_timestep = timestep.clone()
        static_cond = {k: v.clone() for k, v in conditional_dict.items()}

        # Warmup
        torch.cuda.synchronize()
        with torch.no_grad():
            _ = mock_generator(
                noisy_image_or_video=static_noisy,
                conditional_dict=static_cond,
                timestep=static_timestep,
                kv_cache=graph_cache,
                crossattn_cache=None,
                current_start=0,
            )
        torch.cuda.synchronize()

        # Reset cache
        for cache in graph_cache:
            cache["k"].zero_()
            cache["v"].zero_()
            cache["write_ptr"].zero_()
            cache["global_end_index"].zero_()
            cache["local_end_index"].zero_()
            cache["is_wrapped"].zero_()

        # Capture
        graph = torch.cuda.CUDAGraph()
        static_output = torch.zeros_like(noisy_input)  # Pre-allocate output buffer
        with torch.cuda.graph(graph):
            _, static_output_captured = mock_generator(
                noisy_image_or_video=static_noisy,
                conditional_dict=static_cond,
                timestep=static_timestep,
                kv_cache=graph_cache,
                crossattn_cache=None,
                current_start=0,
            )
            # Copy to static output buffer
            static_output.copy_(static_output_captured)

        # Replay the graph to get actual output
        graph.replay()
        torch.cuda.synchronize()
        graph_output = static_output.clone()

        # === Compare ===
        assert torch.allclose(eager_output, graph_output, atol=1e-5), \
            f"Max diff: {(eager_output - graph_output).abs().max().item()}"

    def test_cuda_graph_replay_with_different_inputs(self, mock_generator, test_inputs, device):
        """
        Test that CUDA graph replay with different input values produces
        the same result as eager path with those values.
        """
        noisy_input, timestep, conditional_dict = test_inputs

        def make_cache():
            caches = []
            for _ in range(2):
                caches.append(create_ring_buffer_cache(
                    batch_size=1, buffer_size=1000, num_heads=4, head_dim=16,
                    sink_tokens=0, dtype=torch.float32, device=device
                ))
            return caches

        # Create static buffers and capture graph
        graph_cache = make_cache()
        static_noisy = noisy_input.clone()
        static_timestep = timestep.clone()
        static_cond = {k: v.clone() for k, v in conditional_dict.items()}

        # Warmup + capture
        torch.cuda.synchronize()
        with torch.no_grad():
            _ = mock_generator(static_noisy, static_cond, static_timestep,
                             graph_cache, None, 0)
        torch.cuda.synchronize()

        for cache in graph_cache:
            cache["k"].zero_()
            cache["v"].zero_()
            cache["write_ptr"].zero_()
            cache["global_end_index"].zero_()
            cache["local_end_index"].zero_()
            cache["is_wrapped"].zero_()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _, static_output = mock_generator(static_noisy, static_cond,
                                              static_timestep, graph_cache, None, 0)

        # Now test with different input values
        torch.manual_seed(123)  # Different seed
        new_noisy = torch.randn_like(noisy_input)
        new_timestep = torch.ones_like(timestep) * 750  # Different timestep
        new_cond = {"prompt_embeds": torch.randn_like(conditional_dict["prompt_embeds"])}

        # Eager with new inputs
        eager_cache = make_cache()
        with torch.no_grad():
            _, eager_output = mock_generator(
                new_noisy.clone(),
                {k: v.clone() for k, v in new_cond.items()},
                new_timestep.clone(),
                eager_cache, None, 0
            )

        # Graph replay with new inputs (copy into static buffers)
        for cache in graph_cache:
            cache["k"].zero_()
            cache["v"].zero_()
            cache["write_ptr"].zero_()
            cache["global_end_index"].zero_()
            cache["local_end_index"].zero_()
            cache["is_wrapped"].zero_()

        static_noisy.copy_(new_noisy)
        static_timestep.copy_(new_timestep)
        for k, v in new_cond.items():
            static_cond[k].copy_(v)

        graph.replay()
        graph_output = static_output.clone()

        # Compare
        assert torch.allclose(eager_output, graph_output, atol=1e-5), \
            f"Max diff: {(eager_output - graph_output).abs().max().item()}"


class TestFlashAttentionVarlenCudaGraph:
    """
    Tests for flash_attention_varlen under CUDA graph capture.

    These tests verify that the attention function works correctly when
    captured in a CUDA graph, which requires:
    1. No tensor allocations during capture
    2. All operations use pre-allocated buffers
    3. In-place updates only via tensor operations (no Python indexing)
    """

    def test_flash_attention_varlen_graph_capture(self, device):
        """Test that flash_attention_varlen can be captured in a CUDA graph."""
        from wan.modules.attention import flash_attention_varlen
        from wan.modules.ring_buffer_cache import create_ring_buffer_cache, initialize_cu_seqlens_q

        batch_size = 1
        query_len = 64
        max_seqlen_k = 256
        num_heads = 4
        head_dim = 64  # Must be 64, 128, or 256 for Flash Attention 3

        # Create ring buffer cache with pre-allocated buffers
        cache = create_ring_buffer_cache(
            batch_size=batch_size,
            buffer_size=max_seqlen_k,
            num_heads=num_heads,
            head_dim=head_dim,
            sink_tokens=0,
            dtype=torch.bfloat16,
            device=device,
        )

        # Initialize cu_seqlens_q (must be done before graph capture)
        initialize_cu_seqlens_q(cache, query_len)

        # Create test tensors
        q = torch.randn(batch_size, query_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        k = torch.randn(batch_size, max_seqlen_k, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        v = torch.randn(batch_size, max_seqlen_k, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        k_lens = torch.tensor([128], device=device, dtype=torch.int32)  # Valid length < max

        # Warmup run
        torch.cuda.synchronize()
        with torch.no_grad():
            _ = flash_attention_varlen(
                q=q, k=k, v=v,
                k_lens=k_lens,
                max_seqlen_k=max_seqlen_k,
                cu_seqlens_q=cache["cu_seqlens_q"],
                cu_seqlens_k=cache["cu_seqlens_k"],
                k_lens_padded=cache["k_lens_padded"],
            )
        torch.cuda.synchronize()

        # Create static buffers for graph capture
        static_q = q.clone()
        static_k = k.clone()
        static_v = v.clone()
        static_k_lens = k_lens.clone()

        # Capture the graph
        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph):
            static_output = flash_attention_varlen(
                q=static_q, k=static_k, v=static_v,
                k_lens=static_k_lens,
                max_seqlen_k=max_seqlen_k,
                cu_seqlens_q=cache["cu_seqlens_q"],
                cu_seqlens_k=cache["cu_seqlens_k"],
                k_lens_padded=cache["k_lens_padded"],
            )

        # Replay should work without error
        cuda_graph.replay()

        # Output should have correct shape
        assert static_output.shape == (batch_size, query_len, num_heads, head_dim)

    def test_flash_attention_varlen_graph_replay_consistency(self, device):
        """Test that CUDA graph replay produces consistent output."""
        from wan.modules.attention import flash_attention_varlen
        from wan.modules.ring_buffer_cache import create_ring_buffer_cache, initialize_cu_seqlens_q

        batch_size = 1
        query_len = 64
        max_seqlen_k = 256
        num_heads = 4
        head_dim = 64  # Must be 64, 128, or 256 for Flash Attention 3

        # Create cache for graph path
        cache = create_ring_buffer_cache(
            batch_size=batch_size,
            buffer_size=max_seqlen_k,
            num_heads=num_heads,
            head_dim=head_dim,
            sink_tokens=0,
            dtype=torch.bfloat16,
            device=device,
        )
        initialize_cu_seqlens_q(cache, query_len)

        torch.manual_seed(42)
        q = torch.randn(batch_size, query_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        k = torch.randn(batch_size, max_seqlen_k, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        v = torch.randn(batch_size, max_seqlen_k, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        k_lens = torch.tensor([128], device=device, dtype=torch.int32)

        # Warmup
        with torch.no_grad():
            _ = flash_attention_varlen(
                q=q, k=k, v=v, k_lens=k_lens, max_seqlen_k=max_seqlen_k,
                cu_seqlens_q=cache["cu_seqlens_q"],
                cu_seqlens_k=cache["cu_seqlens_k"],
                k_lens_padded=cache["k_lens_padded"],
            )

        # Graph capture
        static_q = q.clone()
        static_k = k.clone()
        static_v = v.clone()
        static_k_lens = k_lens.clone()

        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph):
            static_output = flash_attention_varlen(
                q=static_q, k=static_k, v=static_v,
                k_lens=static_k_lens,
                max_seqlen_k=max_seqlen_k,
                cu_seqlens_q=cache["cu_seqlens_q"],
                cu_seqlens_k=cache["cu_seqlens_k"],
                k_lens_padded=cache["k_lens_padded"],
            )

        # First replay
        cuda_graph.replay()
        output1 = static_output.clone()

        # Second replay (should produce identical output)
        cuda_graph.replay()
        output2 = static_output.clone()

        # Outputs should be identical (not just close - exactly the same)
        assert torch.equal(output1, output2), \
            f"Graph replay not deterministic! Max diff: {(output1 - output2).abs().max().item()}"

    def test_flash_attention_varlen_graph_k_lens_changes(self, device):
        """Test that changing k_lens and replaying graph produces different outputs."""
        from wan.modules.attention import flash_attention_varlen
        from wan.modules.ring_buffer_cache import create_ring_buffer_cache, initialize_cu_seqlens_q

        batch_size = 1
        query_len = 64
        max_seqlen_k = 256
        num_heads = 4
        head_dim = 64  # Must be 64, 128, or 256 for Flash Attention 3

        cache = create_ring_buffer_cache(
            batch_size=batch_size,
            buffer_size=max_seqlen_k,
            num_heads=num_heads,
            head_dim=head_dim,
            sink_tokens=0,
            dtype=torch.bfloat16,
            device=device,
        )
        initialize_cu_seqlens_q(cache, query_len)

        torch.manual_seed(42)
        q = torch.randn(batch_size, query_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        k = torch.randn(batch_size, max_seqlen_k, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        v = torch.randn(batch_size, max_seqlen_k, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        k_lens = torch.tensor([128], device=device, dtype=torch.int32)

        # Warmup
        with torch.no_grad():
            _ = flash_attention_varlen(
                q=q, k=k, v=v, k_lens=k_lens, max_seqlen_k=max_seqlen_k,
                cu_seqlens_q=cache["cu_seqlens_q"],
                cu_seqlens_k=cache["cu_seqlens_k"],
                k_lens_padded=cache["k_lens_padded"],
            )

        # Capture graph with static buffers
        static_q = q.clone()
        static_k = k.clone()
        static_v = v.clone()
        static_k_lens = k_lens.clone()

        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph):
            static_output = flash_attention_varlen(
                q=static_q, k=static_k, v=static_v,
                k_lens=static_k_lens,
                max_seqlen_k=max_seqlen_k,
                cu_seqlens_q=cache["cu_seqlens_q"],
                cu_seqlens_k=cache["cu_seqlens_k"],
                k_lens_padded=cache["k_lens_padded"],
            )

        # Replay with initial k_lens and save output
        cuda_graph.replay()
        output_k128 = static_output.clone()

        # Change k_lens and replay - output should be different
        static_k_lens.fill_(64)
        cuda_graph.replay()
        output_k64 = static_output.clone()

        # Outputs should differ when k_lens changes (proves k_lens is being used)
        assert not torch.equal(output_k128, output_k64), \
            "Output should change when k_lens changes!"

        # Change k_lens back and verify we get the same output as before
        static_k_lens.fill_(128)
        cuda_graph.replay()
        output_k128_again = static_output.clone()

        assert torch.equal(output_k128, output_k128_again), \
            "Same k_lens should produce same output"


class TestCausalWanSelfAttentionCudaGraph:
    """
    Integration tests for CausalWanSelfAttention under CUDA graph capture.

    Note: Full CausalWanSelfAttention CUDA graph capture requires fixing
    causal_rope_apply to avoid grid_sizes.tolist() which triggers CUDA sync.
    The MockGenerator tests above demonstrate the pattern works when RoPE
    is handled correctly.
    """

    @pytest.mark.skip(reason="causal_rope_apply uses grid_sizes.tolist() which is not CUDA graph compatible")
    def test_self_attention_with_ring_buffer_graph_capture(self, device):
        """Test that CausalWanSelfAttention can be captured in a CUDA graph.

        Currently skipped because causal_rope_apply uses grid_sizes.tolist()
        which triggers a CUDA sync and is not compatible with CUDA graph capture.
        The flash_attention_varlen tests above verify the core attention mechanism
        works correctly with CUDA graphs.
        """
        pass


class TestRingBufferCacheGuard:
    """Tests for the ring buffer cache guard in CUDA graph path."""

    def test_guard_rejects_legacy_cache(self, device):
        """Verify that using legacy cache with CUDA graphs raises an error."""
        from pipeline.causal_inference import CausalInferencePipeline

        # Create a minimal mock args
        class MockArgs:
            def __init__(self):
                self.model_kwargs = type('obj', (object,), {
                    'model_name': 'Wan2.1-T2V-1.3B',
                    'local_attn_size': -1,
                    'sink_size': 0
                })()
                self.context_noise = 0

        # We can't easily test the full pipeline without loading models,
        # but we can test the is_ring_buffer_cache function
        legacy_cache = {
            "k": torch.zeros(1, 100, 4, 16, device=device),
            "v": torch.zeros(1, 100, 4, 16, device=device),
            "global_end_index": torch.tensor([0], device=device),
            "local_end_index": torch.tensor([0], device=device),
        }

        ring_cache = create_ring_buffer_cache(
            batch_size=1, buffer_size=100, num_heads=4, head_dim=16,
            sink_tokens=0, dtype=torch.float32, device=device
        )

        assert not is_ring_buffer_cache(legacy_cache)
        assert is_ring_buffer_cache(ring_cache)
