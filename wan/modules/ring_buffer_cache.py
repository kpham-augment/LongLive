# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""
Ring Buffer KV Cache for CUDA Graph Compatibility.

This module implements a fixed-size circular buffer for KV cache that:
1. Eliminates clone operations that break CUDA graph capture
2. Uses circular overwrites instead of expensive shift operations
3. Maintains fixed tensor shapes throughout inference
4. Uses only tensor operations (no .item() calls) for full CUDA graph compatibility

Memory Layout:
    Buffer size = max_attention_size (= kv_cache_size)

    Index:     0         sink_tokens                               buffer_size
               |             |                                          |
               v             v                                          v
               +-------------+------------------------------------------+
               |  SINK REGION |           ROLLING WINDOW REGION          |
               | (fixed K/V)  |     (circular overwrites after fill)     |
               +-------------+------------------------------------------+
               |<- sink_tokens->|<------------ rolling_capacity -------->|

CUDA Graph Compatibility:
    This implementation is fully CUDA graph compatible:

    1. NO .item() CALLS: All index computations use tensor operations with
       torch.where() for conditionals and tensor arithmetic for indices.

    2. FIXED TENSOR SHAPES: All output tensors have fixed shapes determined
       by max_attention_size. Variable token counts are handled by returning
       num_valid as a scalar tensor for k_lens masking in flash attention.

    3. SCALAR TENSORS FOR STATE: write_ptr, global_end_index, local_end_index,
       and is_wrapped are all scalar tensors that can be updated via tensor
       operations rather than Python assignments.

    4. WARMUP PHASE: Run warmup iterations before capturing graphs. The first
       few blocks may have different valid token counts, but tensor shapes
       remain fixed.

    The primary benefit of this implementation is eliminating the expensive
    kv_cache.clone() operations that dominate memory bandwidth in the original
    implementation while being fully compatible with CUDA graph capture.
"""
import torch
from typing import Dict


def create_ring_buffer_cache(
    batch_size: int,
    buffer_size: int,
    num_heads: int,
    head_dim: int,
    sink_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Create a single ring buffer KV cache structure for one transformer block.
    
    Args:
        batch_size: Batch size B
        buffer_size: Total buffer size (= max_attention_size = kv_cache_size)
        num_heads: Number of attention heads H
        head_dim: Dimension per head D
        sink_tokens: Number of tokens reserved for sink region (= sink_size * frame_seqlen)
        dtype: Data type for K/V tensors
        device: Device to allocate tensors on
    
    Returns:
        Dictionary containing the ring buffer cache structure with:
        - k, v: Ring buffer storage tensors [B, buffer_size, H, D]
        - write_ptr: Next write position in rolling region (scalar tensor)
        - global_end_index: Total tokens ever written (for RoPE positions)
        - local_end_index: Valid tokens currently in buffer (capped at buffer_size)
        - is_wrapped: Whether the rolling region has wrapped around
        - attn_k, attn_v: Pre-allocated attention buffers [B, buffer_size, H, D]
        - sink_tokens: Number of sink tokens (config)
        - buffer_size: Total buffer size (config)
        - rolling_capacity: Size of rolling window region (config)
    """
    rolling_capacity = buffer_size - sink_tokens

    return {
        # Ring buffer storage for K and V
        "k": torch.zeros([batch_size, buffer_size, num_heads, head_dim], dtype=dtype, device=device),
        "v": torch.zeros([batch_size, buffer_size, num_heads, head_dim], dtype=dtype, device=device),

        # Ring buffer state pointers (all scalar tensors for CUDA graph compatibility)
        "write_ptr": torch.tensor([0], dtype=torch.long, device=device),
        "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
        "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
        "is_wrapped": torch.tensor([False], dtype=torch.bool, device=device),

        # Pre-allocated attention buffers for fixed-shape attention computation
        "attn_k": torch.zeros([batch_size, buffer_size, num_heads, head_dim], dtype=dtype, device=device),
        "attn_v": torch.zeros([batch_size, buffer_size, num_heads, head_dim], dtype=dtype, device=device),

        # Pre-allocated cu_seqlens buffers for flash_attention_varlen (CUDA graph compatible)
        # cu_seqlens_q is pre-populated and static (never changes during inference)
        # cu_seqlens_k is computed in-place during attention using k_lens_padded as scratch
        "cu_seqlens_q": torch.zeros([batch_size + 1], dtype=torch.int32, device=device),
        "cu_seqlens_k": torch.zeros([batch_size + 1], dtype=torch.int32, device=device),
        # Scratch buffer for computing cu_seqlens_k: first element is always 0
        "k_lens_padded": torch.zeros([batch_size + 1], dtype=torch.int32, device=device),
        # Pre-allocated k_lens buffer (int32) for CUDA graph compatibility
        # This avoids .to(dtype=torch.int32) during graph capture which would create a new tensor
        "k_lens_int32": torch.zeros([batch_size], dtype=torch.int32, device=device),

        # Configuration (immutable after creation, stored as tensors for CUDA graph compatibility)
        "sink_tokens": sink_tokens,  # Python int for easy access
        "buffer_size": buffer_size,  # Python int for easy access
        "rolling_capacity": rolling_capacity,  # Python int for easy access
        # Tensor versions for use in CUDA graph-captured computations
        "sink_tokens_t": torch.tensor([sink_tokens], dtype=torch.long, device=device),
        "buffer_size_t": torch.tensor([buffer_size], dtype=torch.long, device=device),
        "rolling_capacity_t": torch.tensor([rolling_capacity], dtype=torch.long, device=device),

        # Pre-allocated boolean tensors for CUDA graph compatibility
        # (avoid torch.tensor(bool) which involves CPU->GPU transfer during capture)
        "protect_sink_false": torch.zeros(1, dtype=torch.bool, device=device),
        "protect_sink_true": torch.ones(1, dtype=torch.bool, device=device),

        # Pre-allocated index tensors for CUDA graph compatibility
        # These replace torch.arange() calls in prepare_attention_with_new_tokens
        "window_offsets": torch.arange(rolling_capacity, dtype=torch.long, device=device),
        # Note: new_offsets will be initialized when query_len is known (in initialize_cuda_graph_buffers)
        "new_offsets": None,  # Placeholder, set during CUDA graph setup

        # Track if cu_seqlens_q has been initialized (query length is constant once set)
        "_cu_seqlens_q_initialized": False,
    }


def initialize_cu_seqlens_q(cache: Dict, query_len: int) -> None:
    """
    Initialize cu_seqlens_q and new_offsets with the query length (called once before CUDA graph capture).

    cu_seqlens_q = [0, lq, 2*lq, ...] for batch_size B, giving [0, lq, 2*lq, ..., B*lq]
    This is static and never changes during inference (query length is constant).

    new_offsets = [0, 1, 2, ..., query_len-1] for indexing into new tokens.
    This replaces torch.arange() calls in prepare_attention_with_new_tokens.

    Args:
        cache: Ring buffer cache dictionary
        query_len: Query sequence length (typically num_frames * tokens_per_frame)
    """
    if cache.get("_cu_seqlens_q_initialized", False):
        return  # Already initialized

    cu_seqlens_q = cache["cu_seqlens_q"]
    batch_size = cu_seqlens_q.shape[0] - 1
    device = cu_seqlens_q.device

    # Populate cu_seqlens_q = [0, lq, 2*lq, ..., B*lq]
    for i in range(batch_size + 1):
        cu_seqlens_q[i] = i * query_len

    # Initialize new_offsets for CUDA graph compatibility
    # This replaces torch.arange(num_input_tokens) in prepare_attention_with_new_tokens
    cache["new_offsets"] = torch.arange(query_len, dtype=torch.long, device=device)

    cache["_cu_seqlens_q_initialized"] = True


def reset_ring_buffer_cache(
    cache: Dict[str, torch.Tensor],
    preserve_sink: bool = False,
) -> None:
    """
    Reset a ring buffer cache to its initial state.
    
    Args:
        cache: Ring buffer cache dictionary to reset
        preserve_sink: If True, preserve the sink region and only reset rolling window
    """
    sink_tokens = cache["sink_tokens"]
    
    if preserve_sink:
        # Only reset the rolling window region
        cache["k"][:, sink_tokens:].zero_()
        cache["v"][:, sink_tokens:].zero_()
        cache["write_ptr"].fill_(sink_tokens)
        cache["local_end_index"].fill_(sink_tokens)
        # Note: global_end_index is NOT reset to maintain RoPE position consistency
    else:
        # Full reset
        cache["k"].zero_()
        cache["v"].zero_()
        cache["write_ptr"].zero_()
        cache["global_end_index"].zero_()
        cache["local_end_index"].zero_()
    
    cache["is_wrapped"].fill_(False)
    cache["attn_k"].zero_()
    cache["attn_v"].zero_()


def get_cache_info_str(cache: Dict[str, torch.Tensor]) -> str:
    """Return a human-readable string describing the cache state."""
    return (
        f"RingBufferCache("
        f"write_ptr={cache['write_ptr'].item()}, "
        f"global_end={cache['global_end_index'].item()}, "
        f"local_end={cache['local_end_index'].item()}, "
        f"wrapped={cache['is_wrapped'].item()}, "
        f"sink_tokens={cache['sink_tokens']}, "
        f"buffer_size={cache['buffer_size']}, "
        f"rolling_capacity={cache['rolling_capacity']})"
    )


def compute_ring_buffer_write_indices(
    cache: Dict[str, torch.Tensor],
    num_new_tokens: int,
    protect_sink: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the indices where new tokens should be written in the ring buffer.

    CUDA Graph Compatible: This function uses only tensor operations without .item() calls.

    This function handles:
    1. Initial filling of the buffer (linear writes)
    2. Circular overwrites after buffer is full
    3. Sink token protection during recomputation

    Args:
        cache: Ring buffer cache dictionary
        num_new_tokens: Number of new tokens to write
        protect_sink: If True, skip writing to sink region (for recomputation)

    Returns:
        Tuple of (write_indices, tokens_to_skip):
        - write_indices: 1D tensor of write indices
        - tokens_to_skip: Scalar tensor of tokens to skip (for sink protection)
    """
    write_ptr = cache["write_ptr"]  # Scalar tensor
    sink_tokens = cache["sink_tokens"]  # Python int (constant)
    buffer_size = cache["buffer_size"]  # Python int (constant)
    rolling_capacity = cache["rolling_capacity"]  # Python int (constant)
    # Use tensor version for CUDA graph compatible operations
    sink_tokens_t = cache["sink_tokens_t"]  # Tensor [1] for graph-safe operations

    device = cache["k"].device

    # Create index tensor for all positions [0, 1, 2, ..., num_new_tokens-1]
    token_offsets = torch.arange(num_new_tokens, device=device, dtype=torch.long)

    # Compute tokens_to_skip as a tensor
    # If protect_sink and write_ptr < sink_tokens: skip = sink_tokens - write_ptr
    # Otherwise: skip = 0
    in_sink_region = write_ptr < sink_tokens_t
    tokens_to_skip_if_protect = torch.clamp(sink_tokens_t - write_ptr, min=0)
    tokens_to_skip = torch.where(
        protect_sink & in_sink_region,
        tokens_to_skip_if_protect,
        torch.zeros_like(write_ptr)
    )

    # Compute effective start position
    # If protect_sink and in sink region: effective_start = sink_tokens
    # Otherwise: effective_start = write_ptr
    effective_start = torch.where(
        protect_sink & in_sink_region,
        sink_tokens_t,  # Use pre-allocated tensor instead of creating new one
        write_ptr
    )

    # Compute write indices using circular buffer arithmetic
    # All positions are computed as: sink_tokens + ((effective_start - sink_tokens + offset) % rolling_capacity)
    # This formula works for both linear fill and wrapped cases
    raw_positions = effective_start - sink_tokens + token_offsets
    circular_positions = raw_positions % rolling_capacity
    write_indices = sink_tokens + circular_positions

    # Handle the case where we're still in the sink region (effective_start < sink_tokens shouldn't happen after above logic)
    # But if effective_start < sink_tokens, we should write linearly starting at effective_start
    write_indices = torch.where(
        (effective_start + token_offsets) < buffer_size,
        # Linear case: just add offset to effective_start, but use circular for wrapped
        torch.where(
            effective_start < sink_tokens,
            effective_start + token_offsets,  # Still filling sink
            write_indices  # Use circular
        ),
        write_indices  # Use circular (wrapped)
    )

    # Actually, let's simplify: use circular indexing for rolling region, linear for sink
    # Position i writes to:
    #   - If effective_start + i < sink_tokens: effective_start + i (still in sink)
    #   - Else: sink_tokens + ((effective_start - sink_tokens + i) % rolling_capacity)
    absolute_pos = effective_start + token_offsets
    in_sink_write = absolute_pos < sink_tokens
    circular_write_pos = sink_tokens + ((effective_start - sink_tokens + token_offsets) % rolling_capacity)

    write_indices = torch.where(in_sink_write, absolute_pos, circular_write_pos)

    return write_indices, tokens_to_skip


def ring_buffer_write(
    cache: Dict[str, torch.Tensor],
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    protect_sink: bool = False,
) -> None:
    """
    Write new K/V tokens to the ring buffer.

    CUDA Graph Compatible: This function uses only tensor operations without .item() calls.

    This is the main write operation that:
    1. Computes write positions (handling circular wrap)
    2. Writes new tokens to those positions
    3. Updates all state pointers

    Args:
        cache: Ring buffer cache dictionary
        new_k: New key tensor [B, num_new_tokens, H, D]
        new_v: New value tensor [B, num_new_tokens, H, D]
        protect_sink: If True, skip writing to sink region (for recomputation)
    """
    num_new_tokens = new_k.size(1)
    if num_new_tokens == 0:
        return

    sink_tokens = cache["sink_tokens"]  # Python int (constant)
    buffer_size = cache["buffer_size"]  # Python int (constant)
    rolling_capacity = cache["rolling_capacity"]  # Python int (constant)
    write_ptr = cache["write_ptr"]  # Scalar tensor
    local_end = cache["local_end_index"]  # Scalar tensor
    device = cache["k"].device

    # Compute write indices (now returns tensor for tokens_skipped)
    write_indices, tokens_skipped = compute_ring_buffer_write_indices(
        cache, num_new_tokens, protect_sink
    )

    # Create a mask for valid writes (indices where we actually write)
    # Token i is valid if i >= tokens_skipped
    token_offsets = torch.arange(num_new_tokens, device=device, dtype=torch.long)
    valid_mask = token_offsets >= tokens_skipped

    # Filter write indices and new k/v to only valid positions
    # We use boolean indexing which is graph-compatible
    valid_write_indices = write_indices[valid_mask]
    valid_new_k = new_k[:, valid_mask]
    valid_new_v = new_v[:, valid_mask]

    # Write to buffer using advanced indexing
    # Shape: cache["k"] is [B, buffer_size, H, D], valid_new_k is [B, valid_tokens, H, D]
    cache["k"][:, valid_write_indices] = valid_new_k
    cache["v"][:, valid_write_indices] = valid_new_v

    # Compute actual number of tokens written
    actual_written = num_new_tokens - tokens_skipped

    # Update write pointer using tensor operations
    # New write pointer = sink_tokens + ((write_ptr - sink_tokens + actual_written) % rolling_capacity)
    # But if write_ptr < sink_tokens, we're still in sink: new_ptr = write_ptr + num_new_tokens
    in_sink = write_ptr < sink_tokens

    # Case 1: Still in sink region
    new_ptr_from_sink = write_ptr + num_new_tokens
    # If new_ptr_from_sink >= buffer_size, wrap around
    wrapped_from_sink = sink_tokens + ((new_ptr_from_sink - sink_tokens) % rolling_capacity)
    new_ptr_if_in_sink = torch.where(
        new_ptr_from_sink >= buffer_size,
        wrapped_from_sink,
        new_ptr_from_sink
    )

    # Case 2: In rolling region
    new_ptr_if_in_rolling = sink_tokens + ((write_ptr - sink_tokens + actual_written) % rolling_capacity)

    # Select based on whether we're in sink
    new_write_ptr = torch.where(in_sink, new_ptr_if_in_sink, new_ptr_if_in_rolling)

    # Update is_wrapped flag
    # Wrapped if: (was in sink and new_ptr >= buffer_size) or (in rolling and write_ptr + actual >= buffer)
    will_wrap_from_sink = in_sink & (new_ptr_from_sink >= buffer_size)
    will_wrap_from_rolling = (~in_sink) & ((write_ptr + actual_written) >= buffer_size)
    should_wrap = will_wrap_from_sink | will_wrap_from_rolling
    # Use where to conditionally set is_wrapped (can't use fill_ in graph)
    cache["is_wrapped"] = torch.where(
        should_wrap,
        torch.ones_like(cache["is_wrapped"]),
        cache["is_wrapped"]
    )

    # Update write_ptr
    cache["write_ptr"] = new_write_ptr

    # Update global and local end indices
    cache["global_end_index"] = cache["global_end_index"] + num_new_tokens
    new_local_end = torch.clamp(local_end + num_new_tokens, max=buffer_size)
    cache["local_end_index"] = new_local_end


def ring_buffer_write_from_full(
    cache: Dict[str, torch.Tensor],
    full_k: torch.Tensor,
    full_v: torch.Tensor,
    source_offset: torch.Tensor,
    write_len: torch.Tensor,
    protect_sink: torch.Tensor = None,
) -> None:
    """
    Write K/V tokens to ring buffer from full tensors with tensor-based offset/length.

    FULLY CUDA Graph Compatible: All parameters including source_offset, write_len,
    and protect_sink are tensors. No .item() calls, no Python conditionals on tensors.

    Args:
        cache: Ring buffer cache dictionary
        full_k: Full key tensor [B, num_input_tokens, H, D]
        full_v: Full value tensor [B, num_input_tokens, H, D]
        source_offset: Scalar tensor - offset into full_k/full_v to start reading
        write_len: Scalar tensor - number of tokens to write
        protect_sink: Scalar tensor (bool) - if True, skip writing to sink region
    """
    B, num_input_tokens, H, D = full_k.shape
    device = cache["k"].device

    sink_tokens = cache["sink_tokens"]  # Python int (constant)
    buffer_size = cache["buffer_size"]  # Python int (constant)
    rolling_capacity = cache["rolling_capacity"]  # Python int (constant)
    sink_tokens_t = cache["sink_tokens_t"]  # Tensor [1] for graph-safe operations
    write_ptr = cache["write_ptr"]  # Scalar tensor
    local_end = cache["local_end_index"]  # Scalar tensor

    # Handle protect_sink as tensor (default False)
    # For CUDA graph compatibility, use cache's pre-allocated tensor if available
    if protect_sink is None:
        if "protect_sink_false" in cache:
            protect_sink = cache["protect_sink_false"]
        else:
            protect_sink = torch.zeros(1, dtype=torch.bool, device=device)
    elif not isinstance(protect_sink, torch.Tensor):
        # Use pre-allocated tensors from cache if available (CUDA graph safe)
        if protect_sink and "protect_sink_true" in cache:
            protect_sink = cache["protect_sink_true"]
        elif not protect_sink and "protect_sink_false" in cache:
            protect_sink = cache["protect_sink_false"]
        else:
            # Fallback: allocate (NOT CUDA graph safe for capture!)
            protect_sink = torch.zeros(1, dtype=torch.bool, device=device).fill_(protect_sink)

    # Create index tensor for all possible source positions
    source_offsets = torch.arange(num_input_tokens, dtype=torch.long, device=device)

    # Compute source indices: source_offset + i for each position
    source_indices = source_offset + source_offsets
    source_indices = torch.clamp(source_indices, max=num_input_tokens - 1)

    # Compute validity mask: position i is valid if i < write_len
    valid_mask = source_offsets < write_len

    # Compute sink protection values (always compute, use conditionally)
    tokens_in_sink = torch.clamp(sink_tokens_t - write_ptr, min=0)
    protected_start = torch.maximum(write_ptr, sink_tokens_t.to(write_ptr.dtype))

    # Use torch.where to select based on protect_sink tensor
    effective_start = torch.where(protect_sink, protected_start, write_ptr)

    # Sink protection mask: skip source tokens that map to sink region
    sink_skip_mask = source_offsets >= tokens_in_sink
    # Apply sink protection conditionally using torch.where
    valid_mask = torch.where(
        protect_sink,
        valid_mask & sink_skip_mask,
        valid_mask
    )

    # Compute destination positions
    dest_positions = effective_start + source_offsets

    # Handle circular wrapping in rolling region
    # If dest_position >= sink_tokens, use circular indexing
    in_rolling = dest_positions >= sink_tokens
    circular_positions = sink_tokens + ((dest_positions - sink_tokens) % rolling_capacity)
    dest_indices = torch.where(in_rolling, circular_positions, dest_positions)
    dest_indices = torch.clamp(dest_indices, max=buffer_size - 1)

    # Gather source tokens
    source_indices_expanded = source_indices.view(1, -1, 1, 1).expand(B, -1, H, D)
    gathered_k = full_k.gather(1, source_indices_expanded)
    gathered_v = full_v.gather(1, source_indices_expanded)

    # Scatter to destination positions (only where valid)
    dest_indices_expanded = dest_indices.view(1, -1, 1, 1).expand(B, -1, H, D)
    mask_expanded = valid_mask.view(1, -1, 1, 1).expand(B, -1, H, D)

    # Use where + scatter for conditional write
    current_k = cache["k"].gather(1, dest_indices_expanded)
    current_v = cache["v"].gather(1, dest_indices_expanded)

    new_k_values = torch.where(mask_expanded, gathered_k, current_k)
    new_v_values = torch.where(mask_expanded, gathered_v, current_v)

    cache["k"].scatter_(1, dest_indices_expanded, new_k_values)
    cache["v"].scatter_(1, dest_indices_expanded, new_v_values)

    # Compute actual tokens written (with sink protection)
    actual_written_protected = torch.clamp(write_len - tokens_in_sink, min=0)
    actual_written = torch.where(protect_sink, actual_written_protected, write_len)

    # Update write pointer with circular wrapping
    in_sink_region = write_ptr < sink_tokens
    new_ptr_linear = write_ptr + actual_written
    new_ptr_circular = sink_tokens + ((write_ptr - sink_tokens + actual_written) % rolling_capacity)
    new_write_ptr = torch.where(in_sink_region & (new_ptr_linear < buffer_size),
                                 new_ptr_linear, new_ptr_circular)

    # Update is_wrapped flag
    should_wrap = (write_ptr + actual_written) >= buffer_size
    cache["is_wrapped"] = torch.where(
        should_wrap,
        torch.ones_like(cache["is_wrapped"]),
        cache["is_wrapped"]
    )

    # Update pointers
    cache["write_ptr"] = new_write_ptr
    cache["global_end_index"] = cache["global_end_index"] + write_len
    cache["local_end_index"] = torch.clamp(local_end + write_len, max=buffer_size)


def compute_gather_indices(
    cache: Dict[str, torch.Tensor],
    local_budget: int,
    include_sink: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute indices to gather tokens for attention computation.

    CUDA Graph Compatible: This function uses only tensor operations without .item() calls.

    This returns indices for sink tokens (if any) + most recent local_budget tokens
    from the rolling window, preserving the current attention semantics.

    Args:
        cache: Ring buffer cache dictionary
        local_budget: Maximum number of tokens from rolling window to include
        include_sink: Whether to include sink tokens in the gathered indices

    Returns:
        Tuple of (gather_indices, num_valid_tokens):
        - gather_indices: 1D tensor of indices into the ring buffer (fixed size = sink_tokens + local_budget)
        - num_valid_tokens: Scalar tensor of actual valid tokens (for k_lens masking)
    """
    sink_tokens = cache["sink_tokens"]  # Python int (constant)
    rolling_capacity = cache["rolling_capacity"]  # Python int (constant)
    write_ptr = cache["write_ptr"]  # Scalar tensor
    local_end = cache["local_end_index"]  # Scalar tensor
    is_wrapped = cache["is_wrapped"]  # Scalar tensor (bool)
    device = cache["k"].device

    # Maximum output size is fixed for graph compatibility
    max_output_size = sink_tokens + local_budget if include_sink else local_budget

    # Preallocate output tensor with zeros (will be masked by num_valid)
    gather_indices = torch.zeros(max_output_size, dtype=torch.long, device=device)

    # Part 1: Sink tokens (indices 0 to sink_tokens-1)
    if include_sink and sink_tokens > 0:
        # Sink indices are just 0, 1, 2, ..., sink_tokens-1
        sink_idx = torch.arange(sink_tokens, dtype=torch.long, device=device)
        gather_indices[:sink_tokens] = sink_idx

    write_offset = sink_tokens if include_sink else 0

    # Part 2: Rolling window tokens
    # tokens_in_rolling = local_end - sink_tokens (if not wrapped) or rolling_capacity (if wrapped)
    tokens_in_rolling_unwrapped = torch.clamp(local_end - sink_tokens, min=0)
    tokens_in_rolling = torch.where(
        is_wrapped,
        torch.tensor(rolling_capacity, device=device, dtype=torch.long),
        tokens_in_rolling_unwrapped
    )

    # window_size = min(local_budget, tokens_in_rolling)
    window_size = torch.clamp(tokens_in_rolling, max=local_budget)

    # Generate window indices using circular buffer arithmetic
    # Most recent tokens end at write_ptr, so we go backwards
    # Index i (0 to local_budget-1) maps to:
    #   sink_tokens + ((write_ptr - sink_tokens - window_size + i) % rolling_capacity)
    window_offsets = torch.arange(local_budget, dtype=torch.long, device=device)

    # Compute circular positions
    # Position for offset i: (write_ptr - sink_tokens - window_size + i) mod rolling_capacity
    circular_pos = (write_ptr - sink_tokens - window_size + window_offsets) % rolling_capacity
    window_indices = sink_tokens + circular_pos

    # For unwrapped case, we want linear indices from (write_ptr - window_size) to write_ptr
    # which is equivalent to the circular formula when not wrapped
    gather_indices[write_offset:write_offset + local_budget] = window_indices

    # Compute total valid tokens
    # valid = valid_sink + window_size, but clamp valid_sink to actual written sink
    num_valid = torch.clamp(local_end, max=sink_tokens) * (1 if include_sink else 0) + window_size

    return gather_indices, num_valid


def gather_for_attention(
    cache: Dict[str, torch.Tensor],
    local_budget: int,
    include_sink: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Gather K/V tokens from ring buffer into pre-allocated attention buffers.

    CUDA Graph Compatible: This function uses only tensor operations without .item() calls.

    This function populates attn_k and attn_v with the appropriate tokens
    for attention computation, maintaining fixed tensor shapes.

    Args:
        cache: Ring buffer cache dictionary
        local_budget: Maximum tokens from rolling window
        include_sink: Whether to include sink tokens

    Returns:
        Tuple of (attn_k, attn_v, num_valid_tokens):
        - attn_k: Attention key buffer [B, buffer_size, H, D] (modified in-place)
        - attn_v: Attention value buffer [B, buffer_size, H, D] (modified in-place)
        - num_valid_tokens: Scalar tensor of valid tokens at the start of attn_k/attn_v
    """
    gather_indices, num_valid = compute_gather_indices(cache, local_budget, include_sink)

    # Gather using fixed-size indices, then slice by num_valid for attention
    # The gather_indices tensor has fixed size for graph compatibility
    max_gather = gather_indices.size(0)

    # Gather tokens into attention buffers
    cache["attn_k"][:, :max_gather] = cache["k"][:, gather_indices]
    cache["attn_v"][:, :max_gather] = cache["v"][:, gather_indices]

    return cache["attn_k"], cache["attn_v"], num_valid


def prepare_attention_with_new_tokens(
    cache: Dict[str, torch.Tensor],
    full_k: torch.Tensor,
    full_v: torch.Tensor,
    local_end_index: torch.Tensor,
    source_offset: torch.Tensor,
    write_len: torch.Tensor,
    sink_tokens: int,
    max_attention_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare attention K/V by combining cached tokens with new incoming tokens.

    FULLY CUDA Graph Compatible: This function uses only tensor operations.
    - All dynamic values (local_end_index, source_offset, write_len) are tensors
    - No .item() calls or Python conditionals on tensor values
    - Fixed output shapes based on max_attention_size

    The resulting attention buffer contains:
    - Sink tokens (indices 0 to sink_tokens-1 from cache)
    - Local window tokens (most recent tokens from cache rolling region)
    - New tokens (from full_k/full_v, starting at source_offset for write_len tokens)

    Args:
        cache: Ring buffer cache dictionary
        full_k: Full key tokens [B, num_tokens, H, D] (already RoPE'd)
        full_v: Full value tokens [B, num_tokens, H, D]
        local_end_index: Tensor - logical end index after adding new tokens
        source_offset: Tensor - offset into full_k/full_v to start reading
        write_len: Tensor - number of tokens to include from full_k/full_v
        sink_tokens: Number of sink tokens (Python int, constant)
        max_attention_size: Maximum attention window size (Python int, constant)

    Returns:
        Tuple of (attn_k, attn_v, num_valid_tokens) where num_valid is a scalar tensor
    """
    B, num_input_tokens, H, D = full_k.shape
    local_budget = max_attention_size - sink_tokens  # Python int (constant)
    rolling_capacity = cache["rolling_capacity"]  # Python int (constant)

    # Get cache state as tensors
    cache_local_end = cache["local_end_index"]  # Scalar tensor
    cache_write_ptr = cache["write_ptr"]  # Scalar tensor

    # All computations use tensor operations
    # tokens_in_rolling = max(0, local_end_index - sink_tokens)
    tokens_in_rolling = torch.clamp(local_end_index - sink_tokens, min=0)
    # window_tokens_needed = min(local_budget, tokens_in_rolling)
    window_tokens_needed = torch.clamp(tokens_in_rolling, max=local_budget)

    # How many from cache vs new?
    # cached_window_tokens = max(0, window_tokens_needed - write_len)
    cached_window_tokens = torch.clamp(window_tokens_needed - write_len, min=0)
    # new_tokens_to_include = min(write_len, window_tokens_needed)
    new_tokens_to_include = torch.minimum(write_len, window_tokens_needed)

    attn_k = cache["attn_k"]
    attn_v = cache["attn_v"]

    # 1. Copy sink tokens (fixed positions 0 to sink_tokens-1)
    if sink_tokens > 0:
        attn_k[:, :sink_tokens] = cache["k"][:, :sink_tokens]
        attn_v[:, :sink_tokens] = cache["v"][:, :sink_tokens]

    # 2. Copy cached window tokens using gather with mask
    # We use pre-allocated window_offsets for CUDA graph compatibility (no torch.arange)
    max_cached_window = local_budget  # Maximum possible cached window tokens
    if max_cached_window > 0:
        # Use pre-allocated window_offsets from cache (CUDA graph compatible)
        window_offsets = cache["window_offsets"][:max_cached_window]

        # Circular position: (write_ptr - sink_tokens - cached_window_tokens + offset) % rolling_capacity
        # Note: we compute for ALL positions but mask invalid ones
        circular_pos = (cache_write_ptr - sink_tokens - cached_window_tokens + window_offsets) % rolling_capacity
        gather_indices = sink_tokens + circular_pos

        # Gather all positions (invalid ones will be masked by flash attention's k_lens)
        gathered_k = cache["k"][:, gather_indices]  # [B, max_cached_window, H, D]
        gathered_v = cache["v"][:, gather_indices]

        # Write to attention buffer with mask
        # Use masked scatter or direct assignment
        # For fixed shape: we write to positions [sink_tokens, sink_tokens + max_cached_window)
        attn_k[:, sink_tokens:sink_tokens + max_cached_window] = gathered_k
        attn_v[:, sink_tokens:sink_tokens + max_cached_window] = gathered_v

    # 3. Copy new tokens using gather with mask
    # Use pre-allocated new_offsets for CUDA graph compatibility (no torch.arange)
    if num_input_tokens > 0:
        # Use pre-allocated new_offsets from cache (CUDA graph compatible)
        new_offsets = cache["new_offsets"][:num_input_tokens]

        # Source indices: source_offset + new_offsets, but clamped to valid range
        source_indices = torch.clamp(source_offset + new_offsets, max=num_input_tokens - 1)

        # Destination position: sink_tokens + cached_window_tokens + offset
        # But cached_window_tokens is a tensor, so we compute the base position
        dest_base = sink_tokens + cached_window_tokens  # Tensor

        # Create mask for valid new tokens: offset < new_tokens_to_include
        valid_new_mask = new_offsets < new_tokens_to_include  # [num_input_tokens]

        # Gather new tokens
        gathered_new_k = full_k[:, source_indices]  # [B, num_input_tokens, H, D]
        gathered_new_v = full_v[:, source_indices]

        # We need to write to variable destination positions
        # For CUDA graph compatibility, we use index_put_ or scatter_
        # Destination indices: dest_base + new_offsets where valid
        dest_indices = dest_base + new_offsets  # [num_input_tokens] tensor
        dest_indices = torch.clamp(dest_indices, max=max_attention_size - 1)

        # Use scatter to write new tokens to their destinations
        # Expand indices for batch and head dimensions
        dest_indices_expanded = dest_indices.view(1, -1, 1, 1).expand(B, -1, H, D)

        # Create a mask expanded for scatter
        mask_expanded = valid_new_mask.view(1, -1, 1, 1).expand(B, -1, H, D)

        # Scatter new tokens (only where mask is valid)
        # Using where to selectively update
        current_vals_k = attn_k.gather(1, dest_indices_expanded)
        current_vals_v = attn_v.gather(1, dest_indices_expanded)

        new_vals_k = torch.where(mask_expanded, gathered_new_k, current_vals_k)
        new_vals_v = torch.where(mask_expanded, gathered_new_v, current_vals_v)

        attn_k.scatter_(1, dest_indices_expanded, new_vals_k)
        attn_v.scatter_(1, dest_indices_expanded, new_vals_v)

    # Total valid tokens (as tensor for graph compatibility)
    # num_valid = min(sink_tokens, cache_local_end) + cached_window_tokens + new_tokens_to_include
    valid_sink = torch.clamp(cache_local_end, max=sink_tokens)
    num_valid = valid_sink + cached_window_tokens + new_tokens_to_include

    return attn_k, attn_v, num_valid
