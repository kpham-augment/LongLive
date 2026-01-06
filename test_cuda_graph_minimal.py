#!/usr/bin/env python3
"""Minimal CUDA graph test to isolate the issue."""
import torch
import torch.nn as nn

print("Testing minimal CUDA graph capture and replay...")

# Test 1: Simple linear layer
print("\n=== Test 1: Simple Linear ===")
model = nn.Linear(64, 64).cuda()
x = torch.randn(1, 64, device='cuda')

# Warmup
with torch.no_grad():
    _ = model(x)
torch.cuda.synchronize()

# Capture
static_x = x.clone()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    static_out = model(static_x)

# Replay
graph.replay()
print(f"Test 1 PASSED: output shape = {static_out.shape}")


# Test 2: Flash Attention (simulating the core operation)
print("\n=== Test 2: Flash Attention Varlen ===")
try:
    from wan.modules.attention import flash_attention_varlen
    from wan.modules.ring_buffer_cache import create_ring_buffer_cache, initialize_cu_seqlens_q
    
    batch_size = 1
    query_len = 64
    max_seqlen_k = 256
    num_heads = 4
    head_dim = 64
    
    cache = create_ring_buffer_cache(
        batch_size=batch_size,
        buffer_size=max_seqlen_k,
        num_heads=num_heads,
        head_dim=head_dim,
        sink_tokens=0,
        dtype=torch.bfloat16,
        device='cuda',
    )
    initialize_cu_seqlens_q(cache, query_len)
    
    q = torch.randn(batch_size, query_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(batch_size, max_seqlen_k, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(batch_size, max_seqlen_k, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    k_lens = torch.tensor([128], device='cuda', dtype=torch.int32)
    
    # Warmup
    with torch.no_grad():
        _ = flash_attention_varlen(
            q=q, k=k, v=v, k_lens=k_lens, max_seqlen_k=max_seqlen_k,
            cu_seqlens_q=cache["cu_seqlens_q"],
            cu_seqlens_k=cache["cu_seqlens_k"],
            k_lens_padded=cache["k_lens_padded"],
        )
    torch.cuda.synchronize()
    
    # Capture
    static_q = q.clone()
    static_k = k.clone()
    static_v = v.clone()
    static_k_lens = k_lens.clone()
    
    graph2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph2):
        static_attn_out = flash_attention_varlen(
            q=static_q, k=static_k, v=static_v, k_lens=static_k_lens, max_seqlen_k=max_seqlen_k,
            cu_seqlens_q=cache["cu_seqlens_q"],
            cu_seqlens_k=cache["cu_seqlens_k"],
            k_lens_padded=cache["k_lens_padded"],
        )
    
    # Replay
    graph2.replay()
    print(f"Test 2 PASSED: output shape = {static_attn_out.shape}")
except Exception as e:
    print(f"Test 2 FAILED: {e}")


# Test 3: RoPE + Attention (simulating the forward path)
print("\n=== Test 3: causal_rope_apply with CUDA Graph ===")
try:
    from wan.modules.causal_model import causal_rope_apply
    from wan.modules.model import sinusoidal_embedding_1d

    batch_size = 1
    seq_len = 100
    num_heads = 16
    head_dim = 128
    freq_dim = head_dim // 2

    # Create inputs
    x = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float32)
    grid_sizes = torch.tensor([[5, 10, 2]], device='cuda', dtype=torch.long)  # F=5, H=10, W=2 -> 100 tokens

    # Create freqs (using the model's sinusoidal embedding)
    freqs = torch.randn(1024, freq_dim, device='cuda', dtype=torch.float32)  # Simplified freqs

    # Warmup
    with torch.no_grad():
        _ = causal_rope_apply(x, grid_sizes, freqs, start_frame=0, cached_grid_dims=(5, 10, 2))
    torch.cuda.synchronize()
    print("Warmup done")

    # Capture
    static_x = x.clone()
    graph3 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph3):
        static_out = causal_rope_apply(static_x, grid_sizes, freqs, start_frame=0, cached_grid_dims=(5, 10, 2))
    print("Capture done")

    # Replay
    graph3.replay()
    print(f"Test 3 PASSED: output shape = {static_out.shape}")
except Exception as e:
    import traceback
    print(f"Test 3 FAILED: {e}")
    traceback.print_exc()


# Test 4: Ring buffer write (the cache update operation)
print("\n=== Test 4: Ring Buffer Write with CUDA Graph ===")
try:
    from wan.modules.ring_buffer_cache import create_ring_buffer_cache, ring_buffer_write_from_full

    cache = create_ring_buffer_cache(
        batch_size=1, buffer_size=1000, num_heads=4, head_dim=64,
        sink_tokens=0, dtype=torch.bfloat16, device='cuda'
    )

    full_k = torch.randn(1, 64, 4, 64, device='cuda', dtype=torch.bfloat16)
    full_v = torch.randn(1, 64, 4, 64, device='cuda', dtype=torch.bfloat16)
    source_offset = torch.tensor(0, device='cuda', dtype=torch.int64)
    write_len = torch.tensor(64, device='cuda', dtype=torch.int64)

    # Warmup
    ring_buffer_write_from_full(cache, full_k, full_v, source_offset, write_len, protect_sink=False)
    torch.cuda.synchronize()
    print("Warmup done")

    # Reset cache
    cache["k"].zero_()
    cache["v"].zero_()
    cache["write_ptr"].zero_()

    # Capture
    static_full_k = full_k.clone()
    static_full_v = full_v.clone()
    graph4 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph4):
        ring_buffer_write_from_full(cache, static_full_k, static_full_v, source_offset, write_len, protect_sink=False)
    print("Capture done")

    # Reset cache before replay
    cache["k"].zero_()
    cache["v"].zero_()
    cache["write_ptr"].zero_()

    # Replay
    graph4.replay()
    print(f"Test 4 PASSED: write_ptr = {cache['write_ptr'].item()}")
except Exception as e:
    import traceback
    print(f"Test 4 FAILED: {e}")
    traceback.print_exc()

print("\n=== All tests complete ===")

