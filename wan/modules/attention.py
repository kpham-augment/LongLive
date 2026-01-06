# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface

    def is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "hopper" in device_name
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
    if FLASH_ATTN_3_AVAILABLE:
        print("Flash Attention 3 is available and will be used")
    else:
        print("Flash Attention 3 interface found but not using Hopper GPU")
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False
    print("Flash Attention 3 is not available (module not found)")

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
    if not FLASH_ATTN_3_AVAILABLE:
        print("Flash Attention 2 is available and will be used")
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False
    print("Flash Attention 2 is not available (module not found)")

if not FLASH_ATTN_2_AVAILABLE and not FLASH_ATTN_3_AVAILABLE:
    print("No Flash Attention available, will fall back to torch.nn.functional.scaled_dot_product_attention")

# FLASH_ATTN_3_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
    'flash_attention_varlen',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        # FA3 returns the output tensor directly (not wrapped in a tuple like FA2)
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic).unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out


def flash_attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_lens: torch.Tensor,
    max_seqlen_k: int,
    dropout_p: float = 0.,
    softmax_scale: float = None,
    causal: bool = False,
    window_size: tuple = (-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    cu_seqlens_q: torch.Tensor = None,
    cu_seqlens_k: torch.Tensor = None,
    k_lens_padded: torch.Tensor = None,
) -> torch.Tensor:
    """
    CUDA Graph Compatible flash attention with variable-length key/value sequences.

    Unlike flash_attention(), this function:
    1. Does NOT slice k/v tensors based on k_lens (avoids .item() calls)
    2. Expects k/v to be pre-padded to max_seqlen_k
    3. Uses k_lens tensor directly for cu_seqlens_k computation

    This makes it compatible with CUDA graph capture where tensor values can change
    but tensor shapes and operations must remain fixed.

    For CUDA graph compatibility, pass pre-allocated cu_seqlens_q and cu_seqlens_k
    tensors. cu_seqlens_q should already be populated (it's static for fixed lq).
    cu_seqlens_k will be computed from k_lens using in-place operations.

    Args:
        q: Query tensor [B, Lq, Nq, C1]
        k: Key tensor [B, max_seqlen_k, Nk, C1] - may contain padding beyond k_lens
        v: Value tensor [B, max_seqlen_k, Nk, C2] - may contain padding beyond k_lens
        k_lens: Tensor [B] with actual valid lengths for each batch element
        max_seqlen_k: Maximum key sequence length (fixed for graph compatibility)
        dropout_p: Dropout probability
        softmax_scale: Scaling factor for QK^T
        causal: Whether to apply causal mask
        window_size: Sliding window attention parameters
        deterministic: Whether to use deterministic algorithm
        dtype: Data type for computation
        cu_seqlens_q: Pre-allocated cumsum buffer [B+1] for q, already populated
        cu_seqlens_k: Pre-allocated cumsum buffer [B+1] for k (will be computed)
        k_lens_padded: Pre-allocated scratch buffer [B+1] for computing cu_seqlens_k

    Returns:
        Attention output tensor [B, Lq, Nq, C2]
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    b, lq, out_dtype = q.size(0), q.size(1), q.dtype

    # For CUDA graph compatibility, avoid .to() calls that create new tensors
    # Only convert if dtype doesn't match (caller should ensure matching dtypes during graph capture)
    def to_half(x):
        if x.dtype in half_dtypes:
            return x
        return x.to(dtype)

    # Flatten q without slicing (full tensor)
    # Note: flatten returns a view for contiguous tensors, which is graph-safe
    q = to_half(q).flatten(0, 1)

    # Flatten k/v without slicing - k_lens handles the masking
    k = to_half(k).flatten(0, 1)
    v = to_half(v).flatten(0, 1)

    # Ensure k_lens is int32 on correct device
    # For CUDA graph compatibility, k_lens should already be int32 from caller
    # Only convert if necessary (and caller should ensure this doesn't happen during graph capture)
    if k_lens.dtype == torch.int32 and k_lens.device == k.device:
        k_lens_int32 = k_lens  # No conversion needed - use same tensor
    else:
        k_lens_int32 = k_lens.to(dtype=torch.int32, device=k.device)

    # Match q,k dtype to v - should be no-op if they're already the same
    if q.dtype != v.dtype:
        q = q.to(v.dtype)
    if k.dtype != v.dtype:
        k = k.to(v.dtype)

    # Compute cumulative sequence lengths for varlen attention
    # For CUDA graph compatibility, use pre-allocated buffers if provided
    if cu_seqlens_q is None:
        # Non-graph path: allocate as needed
        q_lens = torch.full((b,), lq, dtype=torch.int32, device=q.device)
        cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32)
    # else: cu_seqlens_q is already pre-populated with [0, lq, 2*lq, ...] - static, no update needed

    if cu_seqlens_k is None:
        # Non-graph path: allocate as needed
        cu_seqlens_k = torch.cat([k_lens_int32.new_zeros([1]), k_lens_int32]).cumsum(0, dtype=torch.int32)
    else:
        # CUDA graph compatible path: use in-place tensor operations only
        # k_lens_padded is pre-allocated with first element = 0
        # Copy k_lens into positions [1:] using view + copy_ (both are captured ops)
        k_lens_padded[1:].copy_(k_lens_int32)
        # Compute cumsum in-place into cu_seqlens_k
        torch.cumsum(k_lens_padded, dim=0, out=cu_seqlens_k)

    # Apply attention using varlen interface
    if FLASH_ATTN_3_AVAILABLE:
        result = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)
        # FA3 may return a tuple (output, softmax_lse) or just output depending on version
        x = result[0] if isinstance(result, tuple) else result
        x = x.unflatten(0, (b, lq))
    elif FLASH_ATTN_2_AVAILABLE:
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))
    else:
        raise RuntimeError("flash_attention_varlen requires Flash Attention 2 or 3")

    # For CUDA graph compatibility, only convert dtype if necessary
    if x.dtype != out_dtype:
        return x.type(out_dtype)
    return x
