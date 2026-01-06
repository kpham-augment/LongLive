# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from typing import List
import torch
import torch.distributed as dist

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
from utils.debug_option import DEBUG
from wan.modules.causal_model import is_ring_buffer_cache

class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        if DEBUG:
            print(f"args.model_kwargs: {args.model_kwargs}")
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        # hard code for Wan2.1-T2V-1.3B
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size = args.model_kwargs.local_attn_size

        # CUDA Graph support - stores graphs keyed by (batch_size, num_frames, current_start)
        self._cuda_graphs: dict = {}
        self._cuda_graph_static_inputs: dict = {}
        self._cuda_graph_static_outputs: dict = {}

        # Normalize to list if sequence-like (e.g., OmegaConf ListConfig)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def _get_cuda_graph_key(self, batch_size: int, num_frames: int, current_start: int) -> tuple:
        """Generate a key for CUDA graph caching."""
        return (batch_size, num_frames, current_start)

    def clear_cuda_graphs(self) -> None:
        """Clear all cached CUDA graphs to free memory."""
        self._cuda_graphs.clear()
        self._cuda_graph_static_inputs.clear()
        self._cuda_graph_static_outputs.clear()

    def _run_generator_with_cuda_graph(
        self,
        noisy_input: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        current_start: int,
        warmup: bool = False,
    ) -> torch.Tensor:
        """
        Run the generator forward pass with CUDA graph capture/replay.

        Args:
            noisy_input: Input tensor of shape (batch_size, num_frames, C, H, W)
            conditional_dict: Conditioning dictionary
            timestep: Timestep tensor
            current_start: Current start position in tokens
            warmup: If True, run warmup iterations before capture

        Returns:
            denoised_pred: Denoised prediction tensor

        Raises:
            RuntimeError: If KV cache is not a ring buffer cache. CUDA graphs require
                ring buffer cache to avoid .item() calls and variable tensor slicing.
        """
        # Validate that ring buffer cache is used - required for CUDA graph compatibility
        if self.kv_cache1 is not None:
            for block_idx, cache in enumerate(self.kv_cache1):
                if not is_ring_buffer_cache(cache):
                    raise RuntimeError(
                        f"CUDA graphs require ring buffer cache, but block {block_idx} has legacy cache. "
                        f"Use use_ring_buffer=True in _initialize_kv_cache() or disable CUDA graphs."
                    )

        batch_size, num_frames = noisy_input.shape[:2]
        graph_key = self._get_cuda_graph_key(batch_size, num_frames, current_start)

        import time
        if graph_key not in self._cuda_graphs:
            # First time seeing this configuration - capture the graph
            print(f"[CUDA Graph] New graph_key={graph_key}, will capture...")

            # Initialize cu_seqlens_q for all KV cache blocks (once, before graph capture)
            # Query length = num_frames * frame_seq_length
            query_len = num_frames * self.frame_seq_length
            from wan.modules.ring_buffer_cache import initialize_cu_seqlens_q
            for cache in self.kv_cache1:
                initialize_cu_seqlens_q(cache, query_len)

            # Setup CUDA graph static buffers in the model to avoid dynamic tensor creation
            # noisy_input shape: [batch, num_frames, C, H, W]
            _, _, c, h, w = noisy_input.shape
            self.generator.model.setup_cuda_graph_buffers(
                batch_size=batch_size,
                num_frames=num_frames,
                height=h,
                width=w,
                device=noisy_input.device
            )

            if warmup:
                # Warmup run to ensure CUDA is ready
                # NOTE: First warmup triggers torch.compile for FlexAttention which can take MINUTES
                # IMPORTANT: Warmup must occur on a side stream per PyTorch CUDA graph docs
                print(f"[CUDA Graph] Running warmup (first run triggers torch.compile, may take minutes)...")
                t0 = time.time()

                # Create a side stream for warmup
                warmup_stream = torch.cuda.Stream()
                warmup_stream.wait_stream(torch.cuda.current_stream())

                with torch.cuda.stream(warmup_stream):
                    with torch.no_grad():
                        _ = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start,
                        )

                # Wait for warmup to complete before capture
                torch.cuda.current_stream().wait_stream(warmup_stream)
                torch.cuda.synchronize()
                print(f"[CUDA Graph] Warmup done in {time.time() - t0:.2f}s")

                # Reset caches after warmup but before capture to ensure consistent state
                # The graph will capture fresh cache writes starting from current_start
                from wan.modules.ring_buffer_cache import reset_ring_buffer_cache
                for cache in self.kv_cache1:
                    reset_ring_buffer_cache(cache, preserve_sink=False)
                for cache in self.crossattn_cache:
                    if isinstance(cache, dict) and "is_init" in cache:
                        cache["is_init"] = False
                print(f"[CUDA Graph] Caches reset for capture")

            # Create static input buffers
            static_noisy_input = noisy_input.clone()
            static_timestep = timestep.clone()
            static_conditional_dict = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in conditional_dict.items()
            }

            # Capture the graph using a private memory pool to avoid memory conflicts
            # This is recommended by PyTorch for CUDA graph capture
            print(f"[CUDA Graph] Capturing graph...")
            t0 = time.time()
            cuda_graph = torch.cuda.CUDAGraph()

            # Use a private memory pool for the graph to avoid memory conflicts
            # The pool_id ensures the graph's memory allocations don't interfere with other operations
            mempool = torch.cuda.graph_pool_handle()

            with torch.cuda.graph(cuda_graph, pool=mempool):
                _, static_denoised_pred = self.generator(
                    noisy_image_or_video=static_noisy_input,
                    conditional_dict=static_conditional_dict,
                    timestep=static_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start,
                )
            print(f"[CUDA Graph] Capture complete for graph_key={graph_key} in {time.time() - t0:.2f}s")

            # Store the graph and static buffers
            self._cuda_graphs[graph_key] = cuda_graph
            self._cuda_graph_static_inputs[graph_key] = {
                "noisy_input": static_noisy_input,
                "timestep": static_timestep,
                "conditional_dict": static_conditional_dict,
            }
            self._cuda_graph_static_outputs[graph_key] = static_denoised_pred

            # Debug: Store original cache tensor addresses for verification during replay
            self._debug_cache_ptrs = {}
            for i, cache in enumerate(self.kv_cache1[:1]):  # Check first cache only
                for key in ['k', 'v', 'k_lens_padded', 'cu_seqlens_k', 'cu_seqlens_q']:
                    if key in cache:
                        self._debug_cache_ptrs[f"{i}_{key}"] = cache[key].data_ptr()
                        print(f"[CUDA Graph] Captured cache[{i}][{key}] at ptr={cache[key].data_ptr()}")

            return static_denoised_pred.clone()
        else:
            # Replay the captured graph
            print(f"[CUDA Graph] Replaying graph_key={graph_key}")
            static_inputs = self._cuda_graph_static_inputs[graph_key]

            # Debug: Check if cache tensor addresses are stable
            if hasattr(self, '_debug_cache_ptrs'):
                for i, cache in enumerate(self.kv_cache1[:1]):  # Check first cache only
                    for key in ['k', 'v', 'k_lens_padded', 'cu_seqlens_k', 'cu_seqlens_q']:
                        if key in cache:
                            current_ptr = cache[key].data_ptr()
                            orig_ptr = self._debug_cache_ptrs.get(f"{i}_{key}")
                            if orig_ptr and current_ptr != orig_ptr:
                                print(f"[CUDA Graph] WARNING: Cache tensor {key} address changed! {orig_ptr} -> {current_ptr}")

            # Reset caches to match state at capture time
            from wan.modules.ring_buffer_cache import reset_ring_buffer_cache
            for cache in self.kv_cache1:
                reset_ring_buffer_cache(cache, preserve_sink=False)
            for cache in self.crossattn_cache:
                if isinstance(cache, dict) and "is_init" in cache:
                    cache["is_init"] = False

            # Copy new data into static buffers
            static_inputs["noisy_input"].copy_(noisy_input)
            static_inputs["timestep"].copy_(timestep)
            for k, v in conditional_dict.items():
                if isinstance(v, torch.Tensor):
                    static_inputs["conditional_dict"][k].copy_(v)

            # Replay the graph
            print(f"[CUDA Graph] About to replay graph...")
            torch.cuda.synchronize()
            print(f"[CUDA Graph] Synchronized before replay")
            try:
                self._cuda_graphs[graph_key].replay()
                print(f"[CUDA Graph] Replay completed successfully")
            except Exception as e:
                print(f"[CUDA Graph] Replay failed with exception: {e}")
                raise
            torch.cuda.synchronize()
            print(f"[CUDA Graph] Synchronized after replay")

            return self._cuda_graph_static_outputs[graph_key].clone()

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        use_cuda_graph: bool = False,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            return_latents (bool): Whether to return the latents.
            use_cuda_graph (bool): If True, use CUDA graphs to reduce CPU overhead.
                Requires static shapes and ring buffer cache. Default: False.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        # Decide the device for output based on low_memory (CPU for low-memory mode; otherwise GPU)
        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        kv_policy = ""
        if local_attn_cfg != -1:
            # local attention
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            # global attention
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")

        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size,
            use_ring_buffer=True,  # Use ring buffer for CUDA graph compatibility
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 2: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks

        for block_idx, current_num_frames in enumerate(all_num_frames):
            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame:current_start_frame + current_num_frames]
            current_start_tokens = current_start_frame * self.frame_seq_length

            # Step 2.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    if use_cuda_graph:
                        # Use CUDA graph for the generator forward pass
                        denoised_pred = self._run_generator_with_cuda_graph(
                            noisy_input=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            current_start=current_start_tokens,
                            warmup=(block_idx == 0 and index == 0),
                        )
                    else:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_tokens
                        )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    if use_cuda_graph:
                        denoised_pred = self._run_generator_with_cuda_graph(
                            noisy_input=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            current_start=current_start_tokens,
                            warmup=False,
                        )
                    else:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_tokens
                        )
            # Step 2.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred.to(output.device)
            # Step 2.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            if use_cuda_graph:
                # Use CUDA graph for the context update pass
                _ = self._run_generator_with_cuda_graph(
                    noisy_input=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    current_start=current_start_tokens,
                    warmup=False,
                )
            else:
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_tokens,
                )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 3: Decode the output
        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            # Calculate inter-frame latency (steady-state)
            # Use blocks after the first one to avoid initialization overhead
            if len(block_times) > 1:
                steady_state_blocks = block_times[1:]  # Skip first block
                avg_block_time = sum(steady_state_blocks) / len(steady_state_blocks)
                inter_frame_latency = avg_block_time / self.num_frame_per_block
            else:
                avg_block_time = block_times[0] if block_times else 0
                inter_frame_latency = avg_block_time / self.num_frame_per_block

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")
            print(f"\n  Performance Metrics:")
            print(f"  - Steady-state inter-frame latency: {inter_frame_latency:.2f} ms/frame")
            print(f"    (avg block time: {avg_block_time:.2f} ms for {self.num_frame_per_block} frames)")

        if return_latents:
            return video, output.to(noise.device)
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device, kv_cache_size_override: int | None = None, use_ring_buffer: bool = True):
        """
        Initialize a Per-GPU KV cache for the Wan model.

        Args:
            batch_size: Batch size for the cache
            dtype: Data type for K/V tensors
            device: Device to allocate on
            kv_cache_size_override: Override for cache size (if None, computed from local_attn_size)
            use_ring_buffer: If True (default), use ring buffer cache for CUDA graph compatibility.
                            If False, use legacy clone-based cache.
        """
        # Determine cache size
        if kv_cache_size_override is not None:
            kv_cache_size = kv_cache_size_override
        else:
            if self.local_attn_size != -1:
                # Local attention: cache only needs to store the window
                kv_cache_size = self.local_attn_size * self.frame_seq_length
            else:
                # Global attention: default cache for 21 frames (backward compatibility)
                kv_cache_size = 32760

        # Get sink_size from model (defaults to 0 if not set)
        sink_size = getattr(self.generator.model, 'sink_size', 0)
        if hasattr(self.generator.model, 'config') and hasattr(self.generator.model.config, 'sink_size'):
            sink_size = self.generator.model.config.sink_size
        sink_tokens = sink_size * self.frame_seq_length

        kv_cache1 = []

        if use_ring_buffer:
            # Use new ring buffer cache for CUDA graph compatibility
            from wan.modules.ring_buffer_cache import create_ring_buffer_cache

            # Wan 1.3B model: 12 heads, 128 dim per head
            num_heads = 12
            head_dim = 128

            for _ in range(self.num_transformer_blocks):
                kv_cache1.append(create_ring_buffer_cache(
                    batch_size=batch_size,
                    buffer_size=kv_cache_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    sink_tokens=sink_tokens,
                    dtype=dtype,
                    device=device,
                ))
        else:
            # Legacy cache structure (backward compatible)
            for _ in range(self.num_transformer_blocks):
                kv_cache1.append({
                    "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
                })

        self.kv_cache1 = kv_cache1  # always store the clean cache
        self._use_ring_buffer_cache = use_ring_buffer

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

    def _set_all_modules_max_attention_size(self, local_attn_size_value: int):
        """
        Set max_attention_size on all submodules that define it.
        If local_attn_size_value == -1, use the model's global default (32760 for Wan, 28160 for 5B).
        Otherwise, set to local_attn_size_value * frame_seq_length.
        """
        if local_attn_size_value == -1:
            target_size = 32760
            policy = "global"
        else:
            target_size = int(local_attn_size_value) * self.frame_seq_length
            policy = "local"

        updated_modules = []
        # Update root model if applicable
        if hasattr(self.generator.model, "max_attention_size"):
            try:
                prev = getattr(self.generator.model, "max_attention_size")
            except Exception:
                prev = None
            setattr(self.generator.model, "max_attention_size", target_size)
            updated_modules.append("<root_model>")

        # Update all child modules
        for name, module in self.generator.model.named_modules():
            if hasattr(module, "max_attention_size"):
                try:
                    prev = getattr(module, "max_attention_size")
                except Exception:
                    prev = None
                try:
                    setattr(module, "max_attention_size", target_size)
                    updated_modules.append(name if name else module.__class__.__name__)
                except Exception:
                    pass