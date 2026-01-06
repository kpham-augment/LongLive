#!/usr/bin/env python3
"""
Integration test for CUDA graph capture and replay with the full model.
This test loads the actual model and tests CUDA graph functionality.
"""
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omegaconf import OmegaConf

print("Loading config...")
config = OmegaConf.load("configs/longlive_inference.yaml")
config.use_cuda_graphs = True  # Enable CUDA graphs for testing
config.num_output_frames = 6  # Minimal frames for testing
config.num_samples = 1

device = torch.device("cuda")
torch.set_grad_enabled(False)

print("Initializing pipeline...")
from pipeline import CausalInferencePipeline
pipeline = CausalInferencePipeline(config, device=device)

# Load generator checkpoint
if config.generator_ckpt:
    print(f"Loading generator checkpoint from {config.generator_ckpt}...")
    state_dict = torch.load(config.generator_ckpt, map_location="cpu")
    if "generator" in state_dict or "generator_ema" in state_dict:
        raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
    elif "model" in state_dict:
        raw_gen_state_dict = state_dict["model"]
    else:
        raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")
    
    gen_state_dict = {}
    for k, v in raw_gen_state_dict.items():
        if k.startswith("module."):
            gen_state_dict[k[7:]] = v
        else:
            gen_state_dict[k] = v
    pipeline.generator.load_state_dict(gen_state_dict, strict=False)
    print("Generator checkpoint loaded")

# Skip LoRA loading for this test
print("Skipping LoRA loading for CUDA graph test")

print("\n=== Testing CUDA Graph Integration ===")

# Create test inputs
batch_size = 1
num_frames = config.num_output_frames
height, width = 480, 832  # Standard resolution
latent_h, latent_w = height // 8, width // 8
latent_channels = 16

# Convert entire pipeline to bfloat16 and move to CUDA for inference
pipeline = pipeline.to(dtype=torch.bfloat16)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)

noise = torch.randn(
    batch_size, num_frames, latent_channels, latent_h, latent_w,
    device=device, dtype=torch.bfloat16
)

prompts = ["A beautiful sunset over the ocean with waves crashing on the shore."]

print(f"Input shape: {noise.shape}")
print(f"Prompts: {prompts}")

# Run inference with CUDA graphs
print("\nRunning inference with CUDA graphs...")
try:
    video, latents = pipeline.inference(
        noise=noise,
        text_prompts=prompts,
        return_latents=True,
        low_memory=False,
        use_cuda_graph=True,
    )
    print(f"Output video shape: {video.shape}")
    print(f"Output latents shape: {latents.shape}")
    print("\n=== CUDA Graph Integration Test PASSED ===")
except Exception as e:
    print(f"\n=== CUDA Graph Integration Test FAILED ===")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

