# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional
import time
import torch
import numpy as np

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
from pipeline.causal_inference import CausalInferencePipeline
import torch.distributed as dist
from utils.debug_option import DEBUG
from utils.profiling import create_profiling_state, append_timing, append_labeled_timing


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
        profiling_state = getattr(self, "_profiling_state", None)
        if not self.global_sink:
            # reset kv cache
            for block_idx in range(self.num_transformer_blocks):
                cache = self.kv_cache1[block_idx]
                cache["k"].zero_()
                cache["v"].zero_()
                # cache["global_end_index"].zero_()
                # cache["local_end_index"].zero_()
            
        # reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

        # recache
        if current_start_frame == 0:
            return
        
        num_recache_frames = current_start_frame if self.local_attn_size == -1 else min(self.local_attn_size, current_start_frame)
        recache_start_frame = current_start_frame - num_recache_frames
        
        frames_to_recache = output[:, recache_start_frame:current_start_frame]
        # move to gpu
        if frames_to_recache.device.type == 'cpu':
            target_device = next(self.generator.parameters()).device
            transfer_start = time.perf_counter() if profiling_state is not None else None
            frames_to_recache = frames_to_recache.to(target_device)
            if transfer_start is not None:
                append_labeled_timing(
                    profiling_state["device_transfer_ms"],
                    "recache_cpu_to_gpu",
                    (time.perf_counter() - transfer_start) * 1000.0
                )
        batch_size = frames_to_recache.shape[0]
        print(f"num_recache_frames: {num_recache_frames}, recache_start_frame: {recache_start_frame}, current_start_frame: {current_start_frame}")
        
        # prepare blockwise causal mask
        device = frames_to_recache.device
        mask_start = time.perf_counter() if profiling_state is not None else None
        block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
            device=device,
            num_frames=num_recache_frames,
            frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block,
            local_attn_size=self.local_attn_size
        )
        if mask_start is not None:
            append_timing(
                profiling_state["kv_recache_ms"]["cpu"],
                (time.perf_counter() - mask_start) * 1000.0
            )
        
        context_timestep = torch.ones([batch_size, num_recache_frames], 
                                    device=device, dtype=torch.int64) * self.args.context_noise
        
        self.generator.model.block_mask = block_mask
        
        # recache
        recache_gpu_start = torch.cuda.Event(enable_timing=True) if profiling_state is not None else None
        recache_gpu_end = torch.cuda.Event(enable_timing=True) if profiling_state is not None else None
        with torch.no_grad():
            if recache_gpu_start is not None:
                recache_gpu_start.record()
            self.generator(
                noisy_image_or_video=frames_to_recache,
                conditional_dict=new_conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=recache_start_frame * self.frame_seq_length,
            )
            if recache_gpu_end is not None:
                recache_gpu_end.record()
                sync_start = time.perf_counter()
                torch.cuda.synchronize()
                append_labeled_timing(
                    profiling_state["host_sync_ms"],
                    "recache_gpu",
                    (time.perf_counter() - sync_start) * 1000.0
                )
                append_timing(
                    profiling_state["kv_recache_ms"]["gpu"],
                    recache_gpu_start.elapsed_time(recache_gpu_end)
                )
        
        # reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

    def inference(
        self,
        noise: torch.Tensor,
        *,
        text_prompts_list: List[List[str]],
        switch_frame_indices: List[int],
        return_latents: bool = False,
        low_memory: bool = False,
        profile: bool = False,
        profile_output_dir: Optional[str] = None,
    ):
        """Generate a video and switch prompts at specified frame indices.

        Args:
            noise: Noise tensor, shape = (B, T_out, C, H, W).
            text_prompts_list: List[List[str]], length = N_seg. Prompt list used for segment i (aligned with batch).
            switch_frame_indices: List[int], length = N_seg - 1. The i-th value indicates that when generation reaches this frame (inclusive)
                we start using the prompts for segment i+1.
            return_latents: Whether to also return the latent tensor.
            low_memory: Enable low-memory mode.
        """
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert len(text_prompts_list) >= 1, "text_prompts_list must not be empty"
        assert len(switch_frame_indices) == len(text_prompts_list) - 1, (
            "length of switch_frame_indices should be one less than text_prompts_list"
        )
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        
        # encode all prompts
        print(text_prompts_list)
        cond_list = [self.text_encoder(text_prompts=p) for p in text_prompts_list]

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
        self._profile_cached_quantized_video = None

        # Set up profiling if requested
        profiling_summary = None
        self.last_profiling_summary = None
        profiling_state = None
        if profile:
            profiling_state = create_profiling_state()
            self._set_module_profiling_state(profiling_state)
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            recache_times = []
            recache_start = torch.cuda.Event(enable_timing=True)
            recache_end = torch.cuda.Event(enable_timing=True)
            denoise_step_start = torch.cuda.Event(enable_timing=True)
            denoise_step_end = torch.cuda.Event(enable_timing=True)

            frame_latencies_ms = []         # steady-state inter-frame
            prompt_switch_latencies_ms = [] # per prompt switch
            measuring_prompt_switch = False
            prompt_switch_start = torch.cuda.Event(enable_timing=True)
            init_start.record()
        else:
            if self._profiling_state is not None:
                self._set_module_profiling_state(None)

        def record_cuda_sync(label: str):
            if profile and profiling_state is not None:
                start = time.perf_counter()
                torch.cuda.synchronize()
                append_labeled_timing(
                    profiling_state["host_sync_ms"], label, (time.perf_counter() - start) * 1000.0
                )
            else:
                torch.cuda.synchronize()

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

        self._initialize_kv_cache(
            batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size
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
            record_cuda_sync("init_phase")
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

        for current_num_frames in all_num_frames:
            if next_switch_pos is not None and current_start_frame >= next_switch_pos:
                if profile:
                    prompt_switch_start.record()
                    recache_start.record()
                segment_idx += 1
                self._recache_after_switch(output, current_start_frame, cond_list[segment_idx])
                if DEBUG:
                    print(
                        f"[MultipleSwitch] Switch to segment {segment_idx} at frame {current_start_frame}"
                    )
                next_switch_pos = (
                    switch_frame_indices[segment_idx]
                    if segment_idx < len(switch_frame_indices)
                    else None
                )

                if profile:
                    recache_end.record()
                    record_cuda_sync("recache_switch")
                    recache_elapsed = recache_start.elapsed_time(recache_end)
                    recache_times.append(recache_elapsed)
                    measuring_prompt_switch = True

                print(f"segment_idx: {segment_idx}")
                print(f"text_prompts_list[segment_idx]: {text_prompts_list[segment_idx]}")
            cond_in_use = cond_list[segment_idx]

            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame : current_start_frame + current_num_frames
            ]

            # ---------------- Spatial denoising loop ----------------
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = (
                    torch.ones([batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64)
                    * current_timestep
                )

                if index < len(self.denoising_step_list) - 1:
                    if profile:
                        denoise_step_start.record()
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )
                    if profile:
                        denoise_step_end.record()
                        denoise_step_end.synchronize()
                        append_timing(
                            profiling_state["denoising_step_ms"][int(index)],
                            denoise_step_start.elapsed_time(denoise_step_end)
                        )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep
                        * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    if profile:
                        denoise_step_start.record()
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )
                    if profile:
                        denoise_step_end.record()
                        denoise_step_end.synchronize()
                        append_timing(
                            profiling_state["denoising_step_ms"][int(index)],
                            denoise_step_start.elapsed_time(denoise_step_end)
                        )

            # Record output
            transfer_start = None
            if (profile and profiling_state is not None and output.device.type == "cpu"
                    and denoised_pred.device != output.device):
                transfer_start = time.perf_counter()
            output[:, current_start_frame : current_start_frame + current_num_frames] = denoised_pred.to(output.device)
            if transfer_start is not None:
                append_labeled_timing(
                    profiling_state["device_transfer_ms"],
                    "latents_gpu_to_cpu",
                    (time.perf_counter() - transfer_start) * 1000.0
                )

            # rerun with clean context to update cache
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=cond_in_use,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            if profile:
                block_end.record()
                record_cuda_sync("block_end")
                block_elapsed = block_start.elapsed_time(block_end)
                block_times.append(block_elapsed)
                per_frame_latency_ms = block_elapsed / float(current_num_frames)
                frame_latencies_ms.append(per_frame_latency_ms)
                if measuring_prompt_switch:
                    # Time from prompt_switch_start to the end of the first block
                    # that uses the new prompt.
                    switch_elapsed_ms = prompt_switch_start.elapsed_time(block_end)
                    prompt_switch_latencies_ms.append(switch_elapsed_ms)
                    measuring_prompt_switch = False

            # Update frame pointer
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            record_cuda_sync("diffusion_total")
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Standard decoding
        decode_transfer_start = None
        if (profile and profiling_state is not None and output.device != noise.device):
            decode_transfer_start = time.perf_counter()
        decode_input = output.to(noise.device)
        if decode_transfer_start is not None:
            append_labeled_timing(
                profiling_state["device_transfer_ms"],
                "latents_cpu_to_gpu_decode",
                (time.perf_counter() - decode_transfer_start) * 1000.0
            )
        video = self.vae.decode_to_pixel(decode_input, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        if profile and profiling_state is not None:
            quant_start = time.perf_counter()
            quantized_video = (video * 255.0).round().clamp_(0, 255).to(torch.uint8)
            append_timing(
                profiling_state["quantize_ms"],
                (time.perf_counter() - quant_start) * 1000.0
            )
            dequant_start = time.perf_counter()
            _ = quantized_video.to(video.dtype) / 255.0
            append_timing(
                profiling_state["dequantize_ms"],
                (time.perf_counter() - dequant_start) * 1000.0
            )
            self._profile_cached_quantized_video = quantized_video

        if profile:
            vae_end.record()
            record_cuda_sync("vae_decode")
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time
            init_pct = 100 * init_time / total_time if total_time else 0.0
            diffusion_pct = 100 * diffusion_time / total_time if total_time else 0.0
            vae_pct = 100 * vae_time / total_time if total_time else 0.0
            block_stats = []
            for i, block_time in enumerate(block_times):
                percent = 100 * block_time / diffusion_time if diffusion_time else 0.0
                block_stats.append({
                    "block_index": i,
                    "time_ms": block_time,
                    "percent_of_diffusion": percent
                })
            recache_stats = []
            for i, recache_time in enumerate(recache_times):
                percent = 100 * recache_time / diffusion_time if diffusion_time else 0.0
                recache_stats.append({
                    "recache_index": i,
                    "time_ms": recache_time,
                    "percent_of_diffusion": percent
                })
            # Steady-state inter-frame latency
            inter_frame_mean = float(np.mean(frame_latencies_ms)) if frame_latencies_ms else 0.0
            inter_frame_p95  = float(np.percentile(frame_latencies_ms, 95)) if frame_latencies_ms else 0.0
            inter_frame_max  = float(np.max(frame_latencies_ms)) if frame_latencies_ms else 0.0
            # Prompt-switch latency
            prompt_switch_mean = float(np.mean(prompt_switch_latencies_ms)) if prompt_switch_latencies_ms else 0.0
            prompt_switch_max  = float(np.max(prompt_switch_latencies_ms)) if prompt_switch_latencies_ms else 0.0

            def _basic_stats(samples):
                if not samples:
                    return {
                        "count": 0,
                        "total_ms": 0.0,
                        "mean_ms": 0.0,
                        "p95_ms": 0.0,
                        "max_ms": 0.0,
                    }
                arr = np.array(samples, dtype=np.float64)
                return {
                    "count": int(arr.size),
                    "total_ms": float(arr.sum()),
                    "mean_ms": float(arr.mean()),
                    "p95_ms": float(np.percentile(arr, 95)),
                    "max_ms": float(arr.max()),
                }

            def _dict_stats(bucket):
                return {label: _basic_stats(vals) for label, vals in bucket.items()}

            denoising_step_stats = []
            if profiling_state is not None:
                for step_idx in sorted(profiling_state["denoising_step_ms"].keys()):
                    stats = _basic_stats(profiling_state["denoising_step_ms"][step_idx])
                    stats["step_index"] = int(step_idx)
                    denoising_step_stats.append(stats)

            attention_kernel_stats = {}
            if profiling_state is not None:
                attention_kernel_stats = {
                    kind: _basic_stats(samples)
                    for kind, samples in profiling_state["attention_kernel_ms"].items()
                }
                attention_kernel_stats["counts"] = dict(profiling_state["attention_kernel_counts"])

            kv_operation_stats = {}
            if profiling_state is not None:
                kv_operation_stats = {
                    "prepare": _basic_stats(profiling_state["kv_prepare_ms"]),
                    "apply": _basic_stats(profiling_state["kv_apply_ms"]),
                    "sink_concat": _basic_stats(profiling_state["kv_sink_concat_ms"]),
                    "counts": dict(profiling_state["kv_cache_op_counts"]),
                }

            kv_recache_summary = {}
            if profiling_state is not None:
                kv_recache_summary = {
                    label: _basic_stats(entries)
                    for label, entries in profiling_state["kv_recache_ms"].items()
                }

            quantization_stats = {}
            if profiling_state is not None:
                quantization_stats = {
                    "quantize": _basic_stats(profiling_state["quantize_ms"]),
                    "dequantize": _basic_stats(profiling_state["dequantize_ms"]),
                }

            vae_kernel_stats = {}
            if profiling_state is not None:
                vae_kernel_stats = {
                    "decode": _basic_stats(profiling_state["vae_decode_ms"]),
                    "encode": _basic_stats(profiling_state["vae_encode_ms"]),
                }

            device_transfer_stats = _dict_stats(profiling_state["device_transfer_ms"]) if profiling_state is not None else {}
            host_sync_stats = _dict_stats(profiling_state["host_sync_ms"]) if profiling_state is not None else {}

            # Prepare profiling summary
            profiling_summary = {
                "init_time_ms": init_time,
                "init_percentage": init_pct,
                "diffusion_time_ms": diffusion_time,
                "diffusion_percentage": diffusion_pct,
                "vae_time_ms": vae_time,
                "vae_percentage": vae_pct,
                "total_time_ms": total_time,
                "block_stats": block_stats,
                "recache_stats": recache_stats,
                "batch_size": batch_size,                
                "num_blocks": len(block_stats),
                "num_output_frames": num_output_frames,
                "num_frame_per_block": self.num_frame_per_block,
                "local_attn_size": self.local_attn_size,
                "kv_cache_size": kv_cache_size,
                "device": str(noise.device),

                "inter_frame_latency_ms_mean": inter_frame_mean,
                "inter_frame_latency_ms_p95": inter_frame_p95,
                "inter_frame_latency_ms_max": inter_frame_max,

                "prompt_switch_latencies_ms": prompt_switch_latencies_ms,
                "prompt_switch_latency_ms_mean": prompt_switch_mean,
                "prompt_switch_latency_ms_max": prompt_switch_max,
                "denoising_step_stats": denoising_step_stats,
                "attention_kernel_stats": attention_kernel_stats,
                "kv_operation_stats": kv_operation_stats,
                "kv_recache_breakdown_ms": kv_recache_summary,
                "quantization_stats": quantization_stats,
                "vae_kernel_stats": vae_kernel_stats,
                "device_transfer_stats": device_transfer_stats,
                "host_sync_stats": host_sync_stats,
            }
            self.last_profiling_summary = profiling_summary

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({init_pct:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({diffusion_pct:.2f}%)")
            for i, block_time in enumerate(block_times):
                percent = 100 * block_time / diffusion_time if diffusion_time else 0.0
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({percent:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({vae_pct:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

            print(f"  - Inter-frame latency (mean): {inter_frame_mean:.2f} ms")
            print(f"  - Inter-frame latency (p95):  {inter_frame_p95:.2f} ms")
            print(f"  - Inter-frame latency (max):  {inter_frame_max:.2f} ms")
            if prompt_switch_latencies_ms:
                print(f"  - Prompt-switch latency (mean): {prompt_switch_mean:.2f} ms")
                print(f"  - Prompt-switch latency (max):  {prompt_switch_max:.2f} ms")
            if profile_output_dir and profiling_summary is not None:
                self._write_profiling_results(profile_output_dir, profiling_summary, interactive=True)

        if self._profiling_state is not None:
            self._set_module_profiling_state(None)

        if return_latents:
            return video, output
        return video 
