# Quick summary of hardware testbed
- H100 SXM 80GB HBM3
- FlashAttention 3 

# Latency Analysis
**Inference pipeline overview**         
From a high level, blocks of frames are generated sequentially in an AR manner (what is referred to as `temporal_block`). To generate each block, a denoise loop (with multiple denoising steps) is run. Each denoise step consists of a generator forward pass (and a negligible add noise step). When a prompt switch occurs, there is an additional KV/cross-attention cache management path before continuing generation: caches are reset and a short “recache” forward pass replays the most recent frames (up to the local attention window) under the new conditioning to rebuild a consistent KV state. This extra recache work is the primary reason the first block after a switch is slower than steady-state.
After all blocks are generated in latent space, they are decoded into images by VAE decoder and then stitched together to form a video.

**Definitions**             
Steady-state inter-frame is calculated as block generation time divided by `num_frame_per_block`. Note here that if we were to stream the output video, the actual perceived inter-frame latency from end-user perspective might be higher (because we would need a VAE decoder after every block generation)  
Prompt switch latency is calculated as the time taken to generate the first block after the switch minus the time taken to generate the first block before the switch.     
- For example, if Block 53 generation time is 519.30 ms and Block 54 generation time is 881.24 ms then the prompt switch latency is 881.24 - 519.30 = 361.94 ms.      

**Measurement**  
For BF16 weights
- For `num_sample=1` (when batch size is 1), the Steady-state inter-frame latency is 172.97 ms/frame and Prompt switch latency is 363.88 ms on average. There are no effect on when the prompt switch occurs because we only recache up to `local_attn_size` frames worth of tokens.     
- [TODO] what is the behaviour when `num_sample > 1` (when batch size > 1)?

While the paper mentions support for FP8/INT8, the codebase does not seem to have support for it.

## Break down of latency
- Each denoising step takes 110-120ms. For a single transformer layer, the timings are:
    - Self Attention: ~800 us for first layer self-attn, and ~220 us for subsequent layers
    - Cross Attention: ~41 us
    - Self Attention KV cache temporary clone: ~100us
    - Self Attention KV cache roll and insert for local window: ~170us
    - Self Attention KV cache direct insert for sink tokens: ~180us
    - A more complete kernel break down is reported [below][kernel]  
    - Cross Attention KV cache ops are append and reset when the condition changes. 
- After the forward pass of the last transformer layer, apply KV cache update for newly generated tokens: ~860 us
- During prompt switching, KV recache takes ~360ms
- VAE decode time 22s
- Total kernel launch overhead (i.e the gaps between kernel executions) is in the order of O(100us) per transformer layer. (Hard to measure this programmatically but can tell by looking at nsys report)
- No quant/dequant for FP8/INT8 case as it is curerntly not supported


# Prioritized optimization plan
- Smaller local attention window and/or less sink tokens
These are ranked in roughly the order of ratio of estimated-effort-to-potential-gain.   
- CUDA graphs and replay. Right now, there are very substantial gaps between kernel execution. Specifically, these gaps are
    - Misc gaps (and perhaps unnecessary MemCpy DtoD) during KV cache ops ~ 10us (small but will add up)
    - KV ops and attention kernel is ~100us!! (very substantial)
    - The challenge here involves reimplement how KV caching is done currently to make it graph-able
- Compiler fusion (torch.compile) or specialized kernels to speedup the KV cache operations
- As discussed above, the repo does not have quantization support so we can add support for INT8/FP8. By default the weights are loaded in BF16. We should try INT8 and FP8 for a subset (or all) of the model weights, e.g keep VAE weights in BF16 and the transformer weights in INT8.


# Optimization Log
| Changes | Config | Latency (Steady-state inter-frame) | Quality drop | Keep/Drop |
| -------- | ------- | ------- | ------------ | --------- |
| Smaller local attention window and/or less sink tokens | local_attn_size=10, sink_size=1 | 167ms | Yes | Drop |
| Smaller local attention window and/or less sink tokens | local_attn_size=9, sink_size=3 | 163ms | No | Keep |
| CUDA graphs | | | No changes to quality | Keep |
| Compiler fusion and/or specialized KV ops kernels | | | No changes to quality  | Keep |

**Sweet spot recommendation** Keeping the local attention window at 9 and sink tokens at 3 seems to be a good balance between latency and quality.
# Open question
One idea is to use a vision encoder to encode the last generated frames and use it as a condition for the next generation. This way we do not have to do kv recache and can start fresh (given some frames of context). Still use kv caching during stable generation.
# Running instruction
- To replicate the profile results. Do `nsys profile --output="longlive" --capture-range=cudaProfilerApi  --force-overwrite=true   bash interactive_inference.sh` on `main`. Modify `configs/longlive_interactive_inference.yaml` to change the the attention window and sink token config.
- 
# Appendix
Kernel break down for a denoising step
[TODO]:
[kernel]: 