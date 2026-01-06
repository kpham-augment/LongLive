# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
from typing import List
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
from pipeline.causal_inference import CausalInferencePipeline
from utils.debug_option import DEBUG


class InteractiveCausalInferencePipeline(CausalInferencePipeline):
    def __init__(
        self,
        args,
        device,
        *,
        generator: WanDiffusionWrapper | None = None,
        text_encoder: WanTextEncoder | None = None,
        vae: WanVAEWrapper | None = None,
    ):
        super().__init__(args, device, generator=generator, text_encoder=text_encoder, vae=vae)
        self.global_sink = getattr(args, "global_sink", False)

    # Internal helpers
    def _recache_after_switch(self, output, current_start_frame, new_conditional_dict):
        torch.cuda.nvtx.range_push("recache_after_switch")
        if not self.global_sink:
            # reset kv cache
            torch.cuda.nvtx.range_push("kv_cache_reset")
            for block_idx in range(self.num_transformer_blocks):
                cache = self.kv_cache1[block_idx]
                # Check if using ring buffer cache
                if "write_ptr" in cache:
                    from wan.modules.ring_buffer_cache import reset_ring_buffer_cache
                    reset_ring_buffer_cache(cache, preserve_sink=False)
                else:
                    cache["k"].zero_()
                    cache["v"].zero_()
                    # cache["global_end_index"].zero_()
                    # cache["local_end_index"].zero_()
            torch.cuda.nvtx.range_pop()
        else:
            # global_sink=True: preserve sink tokens, reset only rolling window
            torch.cuda.nvtx.range_push("kv_cache_reset_preserve_sink")
            for block_idx in range(self.num_transformer_blocks):
                cache = self.kv_cache1[block_idx]
                if "write_ptr" in cache:
                    from wan.modules.ring_buffer_cache import reset_ring_buffer_cache
                    reset_ring_buffer_cache(cache, preserve_sink=True)
            torch.cuda.nvtx.range_pop()

        # reset cross-attention cache
        torch.cuda.nvtx.range_push("crossattn_cache_reset")
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False
        torch.cuda.nvtx.range_pop()

        # recache
        if current_start_frame == 0:
            torch.cuda.nvtx.range_pop()
            return
        
        num_recache_frames = current_start_frame if self.local_attn_size == -1 else min(self.local_attn_size, current_start_frame)
        recache_start_frame = current_start_frame - num_recache_frames
        
        frames_to_recache = output[:, recache_start_frame:current_start_frame]
        # move to gpu
        if frames_to_recache.device.type == 'cpu':
            target_device = next(self.generator.parameters()).device
            frames_to_recache = frames_to_recache.to(target_device)
        batch_size = frames_to_recache.shape[0]
        print(f"num_recache_frames: {num_recache_frames}, recache_start_frame: {recache_start_frame}, current_start_frame: {current_start_frame}")
        
        # prepare blockwise causal mask
        device = frames_to_recache.device
        block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
            device=device,
            num_frames=num_recache_frames,
            frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block,
            local_attn_size=self.local_attn_size
        )
        
        context_timestep = torch.ones([batch_size, num_recache_frames], 
                                    device=device, dtype=torch.int64) * self.args.context_noise
        
        self.generator.model.block_mask = block_mask
        
        # recache
        with torch.no_grad():
            self.generator(
                noisy_image_or_video=frames_to_recache,
                conditional_dict=new_conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=recache_start_frame * self.frame_seq_length,
                sink_recache_after_switch=not self.global_sink,
            )

        # reset cross-attention cache
        torch.cuda.nvtx.range_push("crossattn_cache_reset")
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()

    def inference(
        self,
        noise: torch.Tensor,
        *,
        text_prompts_list: List[List[str]],
        switch_frame_indices: List[int],
        return_latents: bool = False,
        low_memory: bool = False,
        profile: bool = False,
        use_cuda_graph: bool = False,
    ):
        """Generate a video and switch prompts at specified frame indices.

        Args:
            noise: Noise tensor, shape = (B, T_out, C, H, W).
            text_prompts_list: List[List[str]], length = N_seg. Prompt list used for segment i (aligned with batch).
            switch_frame_indices: List[int], length = N_seg - 1. The i-th value indicates that when generation reaches this frame (inclusive)
                we start using the prompts for segment i+1.
            return_latents: Whether to also return the latent tensor.
            low_memory: Enable low-memory mode.
            profile: Whether to enable profiling and print timing information.
            use_cuda_graph: If True, use CUDA graphs to reduce CPU overhead.
                Requires static shapes and ring buffer cache. Default: False.
        """
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert len(text_prompts_list) >= 1, "text_prompts_list must not be empty"
        assert len(switch_frame_indices) == len(text_prompts_list) - 1, (
            "length of switch_frame_indices should be one less than text_prompts_list"
        )
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

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
            # Track prompt switches
            switch_times = []  # List of (switch_frame_idx, recache_time_ms, total_switch_time_ms)
            switch_start = torch.cuda.Event(enable_timing=True)
            recache_end = torch.cuda.Event(enable_timing=True)
            switch_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # encode all prompts
        print(text_prompts_list)
        torch.cuda.nvtx.range_push("text_encoder")
        cond_list = [self.text_encoder(text_prompts=p) for p in text_prompts_list]
        torch.cuda.nvtx.range_pop()

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder,
                target_device=gpu,
                preserved_memory_gb=gpu_memory_preservation,
            )

        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )

        # initialize caches
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

        torch.cuda.nvtx.range_push("initialize_kv_cache")
        self._initialize_kv_cache(
            batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size
        )
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("initialize_crossattn_cache")
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )
        torch.cuda.nvtx.range_pop()

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # temporal denoising by blocks
        all_num_frames = [self.num_frame_per_block] * num_blocks
        segment_idx = 0  # current segment index
        next_switch_pos = (
            switch_frame_indices[segment_idx]
            if segment_idx < len(switch_frame_indices)
            else None
        )

        if DEBUG:
            print("[MultipleSwitch] all_num_frames", all_num_frames)
            print("[MultipleSwitch] switch_frame_indices", switch_frame_indices)

        block_idx = 0
        for current_num_frames in all_num_frames:
            torch.cuda.nvtx.range_push(f"temporal_block_{block_idx}")
            if profile:
                block_start.record()

            # Track if we're switching prompts in this block
            switched_in_this_block = False
            if next_switch_pos is not None and current_start_frame >= next_switch_pos:
                # Start timing the prompt switch
                if profile:
                    switch_start.record()

                switched_in_this_block = True
                segment_idx += 1
                torch.cuda.nvtx.range_push(f"recache_after_switch_segment_{segment_idx}")
                self._recache_after_switch(output, current_start_frame, cond_list[segment_idx])
                torch.cuda.nvtx.range_pop()

                # Record end of recaching
                if profile:
                    recache_end.record()
                    torch.cuda.synchronize()

                if DEBUG:
                    print(
                        f"[MultipleSwitch] Switch to segment {segment_idx} at frame {current_start_frame}"
                    )
                next_switch_pos = (
                    switch_frame_indices[segment_idx]
                    if segment_idx < len(switch_frame_indices)
                    else None
                )
                print(f"segment_idx: {segment_idx}")
                print(f"text_prompts_list[segment_idx]: {text_prompts_list[segment_idx]}")
            cond_in_use = cond_list[segment_idx]

            noisy_input = noise[
                :, current_start_frame : current_start_frame + current_num_frames
            ]
            current_start_tokens = current_start_frame * self.frame_seq_length

            # ---------------- Spatial denoising loop ----------------
            torch.cuda.nvtx.range_push("spatial_denoising_loop")
            for index, current_timestep in enumerate(self.denoising_step_list):
                torch.cuda.nvtx.range_push(f"denoising_step_{index}")
                timestep = (
                    torch.ones([batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64)
                    * current_timestep
                )

                if index < len(self.denoising_step_list) - 1:
                    torch.cuda.nvtx.range_push("generator_forward")
                    if use_cuda_graph:
                        denoised_pred = self._run_generator_with_cuda_graph(
                            noisy_input=noisy_input,
                            conditional_dict=cond_in_use,
                            timestep=timestep,
                            current_start=current_start_tokens,
                            warmup=(block_idx == 0 and index == 0),
                        )
                    else:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=cond_in_use,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_tokens,
                        )
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.nvtx.range_push("add_noise")
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep
                        * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])
                    torch.cuda.nvtx.range_pop()
                else:
                    torch.cuda.nvtx.range_push("generator_forward")
                    if use_cuda_graph:
                        denoised_pred = self._run_generator_with_cuda_graph(
                            noisy_input=noisy_input,
                            conditional_dict=cond_in_use,
                            timestep=timestep,
                            current_start=current_start_tokens,
                            warmup=False,
                        )
                    else:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=cond_in_use,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_tokens,
                        )
                    torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_pop()  # denoising_step
            torch.cuda.nvtx.range_pop()  # spatial_denoising_loop

            # Record output
            output[:, current_start_frame : current_start_frame + current_num_frames] = denoised_pred.to(output.device)

            # rerun with clean context to update cache
            torch.cuda.nvtx.range_push("clean_context_cache_update")
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            if use_cuda_graph:
                _ = self._run_generator_with_cuda_graph(
                    noisy_input=denoised_pred,
                    conditional_dict=cond_in_use,
                    timestep=context_timestep,
                    current_start=current_start_tokens,
                    warmup=False,
                )
            else:
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=cond_in_use,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_tokens,
                )
            torch.cuda.nvtx.range_pop()

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

                # If we switched in this block, record both recache and total switch latency
                if switched_in_this_block:
                    switch_end.record()
                    torch.cuda.synchronize()
                    recache_time = switch_start.elapsed_time(recache_end)
                    total_switch_time = switch_start.elapsed_time(switch_end)
                    switch_times.append((current_start_frame, recache_time, total_switch_time))

            # Update frame pointer
            current_start_frame += current_num_frames
            block_idx += 1
            torch.cuda.nvtx.range_pop()  # temporal_block

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Standard decoding
        torch.cuda.nvtx.range_push("vae_decode")
        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        torch.cuda.nvtx.range_pop()

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            # Calculate inter-frame latency (steady-state)
            # Exclude blocks where prompt switching occurred
            switch_block_indices = set()
            for switch_frame, _, _ in switch_times:
                # Find which block this switch occurred in
                block_idx = switch_frame // self.num_frame_per_block
                switch_block_indices.add(block_idx)

            # Get steady-state blocks (skip first block and switch blocks)
            steady_state_blocks = [
                bt for i, bt in enumerate(block_times[1:], start=1)
                if i not in switch_block_indices
            ]

            if steady_state_blocks:
                avg_block_time = sum(steady_state_blocks) / len(steady_state_blocks)
                inter_frame_latency = avg_block_time / self.num_frame_per_block
            elif len(block_times) > 1:
                # Fallback: use all blocks except first
                avg_block_time = sum(block_times[1:]) / len(block_times[1:])
                inter_frame_latency = avg_block_time / self.num_frame_per_block
            else:
                avg_block_time = block_times[0] if block_times else 0
                inter_frame_latency = avg_block_time / self.num_frame_per_block

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                switch_marker = " [PROMPT SWITCH]" if i in switch_block_indices else ""
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion){switch_marker}")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

            print(f"\n  Performance Metrics:")
            print(f"  - Steady-state inter-frame latency: {inter_frame_latency:.2f} ms/frame")
            print(f"    (avg block time: {avg_block_time:.2f} ms for {self.num_frame_per_block} frames, {len(steady_state_blocks)} steady-state blocks)")

            if switch_times:
                print(f"\n  Prompt Switch Latencies:")
                for switch_frame, recache_time, total_time in switch_times:
                    generation_time = total_time - recache_time
                    print(f"    - Switch at frame {switch_frame}:")
                    print(f"        Recache overhead: {recache_time:.2f} ms")
                    print(f"        First block generation: {generation_time:.2f} ms")
                    print(f"        Total switch latency: {total_time:.2f} ms")

                avg_recache = sum(r for _, r, _ in switch_times) / len(switch_times)
                avg_total = sum(t for _, _, t in switch_times) / len(switch_times)
                avg_generation = avg_total - avg_recache

                print(f"\n    Average Prompt-Switch Metrics:")
                print(f"      - Recache overhead: {avg_recache:.2f} ms")
                print(f"      - First block generation: {avg_generation:.2f} ms")
                print(f"      - Total switch latency: {avg_total:.2f} ms")
                print(f"      - Overhead vs steady-state block: {avg_total - avg_block_time:.2f} ms ({100 * (avg_total - avg_block_time) / avg_block_time:.1f}% slower)")

        if return_latents:
            return video, output
        return video