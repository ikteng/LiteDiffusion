"""Blackwell-native MiniMax-H3 transformer for the pruned ComfyUI NVFP4 checkpoint.

The public diffusers checkpoint spends 13.04B of its 33.12B parameters on per-block
AdaLN projections.  ComfyUI's pruned checkpoint replaces those projections with an
interpolated 1025-point timestep curve, fuses Q/K/V, and stores the four large linear
layers in every block as NVFP4.  This adapter keeps diffusers' packed-sequence contract
so the rest of the split Space (schedulers, VAEs and remote conditioner) stays unchanged.

The kernel/layout conventions follow ComfyUI's Apache-2.0 implementation:
https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/model.py
"""

from __future__ import annotations

import json
import math
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import comfy_kitchen as kitchen
from comfy_kitchen.tensor import QuantizedTensor, TensorCoreNVFP4Layout
from diffusers.models.attention_dispatch import dispatch_attention_fn

try:
    import triton
    import triton.language as tl
except ImportError:  # PyTorch CUDA wheels include Triton; retain a portable fallback for source inspection/tests.
    triton = None
    tl = None


NVFP4_REPO = os.environ.get("H3_NVFP4_REPO", "lilcheaty/MiniMax-H3-NVFP4")
NVFP4_FILE = os.environ.get("H3_NVFP4_FILE", "minimax_h3_fl2va_pruned_nvfp4.safetensors")

HIDDEN = 5376
HEADS = 56
HEAD_DIM = 128
FFN = 14336
TEXT_DIM = 5120
TIME_DIM = 8
VIDEO_DIM = 24 * 1 * 2 * 2
AUDIO_DIM = 32
LAYERS = 50
REFINER_LAYERS = 2
EPS = 1e-5

# Cache-DiT's DBCache/TaylorSeer design is adapted directly into this custom in-place block loop. The upstream
# Diffusers hook cannot wrap this lean NVFP4 transformer because it is intentionally not a ModelMixin and its blocks
# have a fused packed video/audio signature. Balanced uses F1B0 DBCache with a first-order Taylor forecast of the
# cached tail. Ultra Fast retains the more aggressive full-output forecast.
# https://github.com/vipshop/cache-dit
EASYCACHE_THRESHOLD = max(0.0, float(os.environ.get("H3_EASYCACHE_THRESHOLD", "0.10")))
EASYCACHE_START = min(1.0, max(0.0, float(os.environ.get("H3_EASYCACHE_START", "0.15"))))
EASYCACHE_END = min(1.0, max(EASYCACHE_START, float(os.environ.get("H3_EASYCACHE_END", "0.95"))))
EASYCACHE_SUBSAMPLE = max(1, int(os.environ.get("H3_EASYCACHE_SUBSAMPLE", "8")))
FIRST_BLOCK_THRESHOLD = max(0.0, float(os.environ.get("H3_FIRST_BLOCK_THRESHOLD", "0.08")))
FIRST_BLOCK_DENSE_START = max(1, int(os.environ.get("H3_FIRST_BLOCK_DENSE_START", "3")))
FIRST_BLOCK_DENSE_END = max(1, int(os.environ.get("H3_FIRST_BLOCK_DENSE_END", "2")))
FIRST_BLOCK_SUBSAMPLE = max(1, int(os.environ.get("H3_FIRST_BLOCK_SUBSAMPLE", "8")))
FIRST_BLOCK_MAX_CACHED = max(1, int(os.environ.get("H3_FIRST_BLOCK_MAX_CACHED", "2")))
FORECAST_BLEND = min(1.0, max(0.0, float(os.environ.get("H3_FORECAST_BLEND", "0.65"))))
FUSED_ADALN = os.environ.get("H3_FUSED_ADALN", "0") == "1" and triton is not None
SOL_ATTN = os.environ.get("H3_SOL_ATTN", "1") == "1"
SOL_ATTN_BACKEND = os.environ.get("H3_SOL_ATTN_BACKEND", "triton").lower()
SOL_ATTN_TAU = float(os.environ.get("H3_SOL_ATTN_TAU", "1.0"))
SOL_ATTN_DENSE_STEPS = max(0, int(os.environ.get("H3_SOL_ATTN_DENSE_STEPS", "10")))
SOL_ATTN_DENSE_LAYERS = max(0, int(os.environ.get("H3_SOL_ATTN_DENSE_LAYERS", "2")))
SOL_ATTN_MIN_TOKENS = max(0, int(os.environ.get("H3_SOL_ATTN_MIN_TOKENS", "24576")))


if triton is not None:

    @triton.jit
    def _adaln_modulate_kernel(
        x, shift, scale, row_ids, elements: tl.constexpr, hidden: tl.constexpr, modulation_stride: tl.constexpr
    ):
        offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
        mask = offsets < elements
        columns = offsets % hidden
        rows = offsets // hidden
        modulation_rows = tl.load(row_ids + rows, mask=mask, other=0)
        modulation_offsets = modulation_rows * modulation_stride + columns
        values = tl.load(x + offsets, mask=mask)
        shifts = tl.load(shift + modulation_offsets, mask=mask)
        scales = tl.load(scale + modulation_offsets, mask=mask)
        tl.store(x + offsets, values * (1.0 + scales) + shifts, mask=mask)

    @triton.jit
    def _adaln_gate_kernel(
        x, update, gate, row_ids, elements: tl.constexpr, hidden: tl.constexpr, modulation_stride: tl.constexpr
    ):
        offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
        mask = offsets < elements
        columns = offsets % hidden
        rows = offsets // hidden
        modulation_rows = tl.load(row_ids + rows, mask=mask, other=0)
        gates = tl.load(gate + modulation_rows * modulation_stride + columns, mask=mask)
        values = tl.load(x + offsets, mask=mask)
        updates = tl.load(update + offsets, mask=mask)
        tl.store(x + offsets, values + updates * gates, mask=mask)


class H3StepCache:
    """ComfyUI EasyCache-style adaptive reuse of a complete H3 denoising result.

    This caches the model residual, not the generated video.  A request with a new prompt, seed, canvas or keyframe
    starts from an empty cache.  Decisions use a sparse sample of generated video latent rows, while the reused
    residual contains every video and audio row so their joint denoising trajectory stays coupled.
    """

    def __init__(self):
        self.total_steps = 0
        self.step = 0
        self.skipped = 0
        self.profile = "balanced"
        self.consecutive_skips = 0
        self.last_actual_step = None
        self.previous_input = None
        self.previous_output = None
        self.previous_output_norm = None
        self.relative_rate = None
        self.accumulated_change = None
        self.video_residual = None
        self.audio_residual = None
        self.video_residual_slope = None
        self.audio_residual_slope = None
        self.pending_input = None
        self.pending_input_change = None
        self.pending_track = False
        self.head_residual_probe = None
        self.tail_residual = None
        self.tail_residual_slope = None
        self.first_block_output = None
        self.last_tail_step = None

    def begin(self, total_steps: int | None, profile: str = "balanced") -> None:
        self.__init__()
        self.total_steps = max(0, int(total_steps or 0))
        self.profile = str(profile or "balanced").lower()

    @property
    def enabled(self) -> bool:
        return self.profile != "exact" and self.total_steps > 2

    def _forecast(self, video_input, audio_input):
        distance = max(1, self.step - int(self.last_actual_step or 0))
        video_residual = self.video_residual
        audio_residual = self.audio_residual
        if self.video_residual_slope is not None:
            video_residual = video_residual + self.video_residual_slope * (distance * FORECAST_BLEND)
            audio_residual = audio_residual + self.audio_residual_slope * (distance * FORECAST_BLEND)
        self.skipped += 1
        self.consecutive_skips += 1
        self.step += 1
        return video_input + video_residual, audio_input + audio_residual

    def try_reuse(self, video_input, audio_input, condition_rows: int):
        self.pending_input = None
        self.pending_input_change = None
        self.pending_track = False
        if not self.enabled:
            return None

        # Balanced uses NVIDIA's H3 FirstBlockCache below, after block 0 has produced a high-signal residual.
        # Only the deliberately aggressive Ultra profile forecasts a whole transformer call before block 0.
        if not self.profile.startswith("ultra"):
            return None

        # Ultra Fast is deliberately bounded: no more than three forecasts can separate exact transformer calls, and
        # the high-noise warmup plus low-noise tail remain exact.  At the default 16 steps this executes 7 full DiT
        # evaluations instead of 16 while still sampling the original 16-step scheduler trajectory.
        if self.profile.startswith("ultra"):
            can_forecast = (
                self.step >= 3
                and self.step < self.total_steps - 2
                and self.consecutive_skips < 3
                and self.last_actual_step is not None
                and self.video_residual is not None
                and self.audio_residual is not None
                and self.video_residual.shape == video_input.shape
                and self.audio_residual.shape == audio_input.shape
            )
            if can_forecast:
                return self._forecast(video_input, audio_input)
            return None

        if EASYCACHE_THRESHOLD <= 0.0:
            return None

        end_step = math.floor(self.total_steps * EASYCACHE_END)
        if self.step >= end_step:
            return None

        # Condition latents are static. Excluding them makes the change estimate reflect the generated trajectory.
        sampled_input = video_input[0, condition_rows::EASYCACHE_SUBSAMPLE].detach().float()
        self.pending_input = sampled_input
        self.pending_track = True
        if self.previous_input is not None:
            self.pending_input_change = (sampled_input - self.previous_input).abs().mean()

        start_step = math.ceil(self.total_steps * EASYCACHE_START)
        can_reuse = (
            self.step >= start_step
            and self.pending_input_change is not None
            and self.relative_rate is not None
            and self.previous_output_norm is not None
            and self.video_residual is not None
            and self.audio_residual is not None
            and self.video_residual.shape == video_input.shape
            and self.audio_residual.shape == audio_input.shape
        )
        if not can_reuse:
            return None

        estimated_change = self.relative_rate * self.pending_input_change
        estimated_change = estimated_change / self.previous_output_norm.clamp_min(1e-6)
        accumulated = estimated_change if self.accumulated_change is None else self.accumulated_change + estimated_change
        if bool((accumulated < EASYCACHE_THRESHOLD).item()):
            self.accumulated_change = accumulated
            self.skipped += 1
            self.step += 1
            return video_input + self.video_residual, audio_input + self.audio_residual
        return None

    def first_block_decision(self, block_input_probe: torch.Tensor, block_output: torch.Tensor) -> bool:
        """Return True when blocks 1..49 can reuse their previous joint residual.

        This is Cache-DiT DBCache F1B0, specialized for H3's fused packed sequence. The first block is always
        evaluated. A strided probe supplies the normalized residual-difference signal, avoiding two reductions over
        the full multi-hundred-MB activation while the cached tail still covers every text/video/audio row.
        """
        if not self.enabled or self.profile.startswith("ultra"):
            return False
        if FIRST_BLOCK_THRESHOLD <= 0.0:
            # Keep the Balanced step clock healthy when caching is disabled through the environment.
            self.first_block_output = block_output.detach().clone()
            return False

        keep_dense = self.step < FIRST_BLOCK_DENSE_START or self.step >= self.total_steps - FIRST_BLOCK_DENSE_END
        residual_probe = block_output[::FIRST_BLOCK_SUBSAMPLE] - block_input_probe
        reusable = (
            not keep_dense
            and self.consecutive_skips < FIRST_BLOCK_MAX_CACHED
            and self.head_residual_probe is not None
            and self.tail_residual is not None
            and self.tail_residual.shape == block_output.shape
        )
        should_reuse = False
        if reusable:
            difference = (residual_probe - self.head_residual_probe).abs().mean()
            reference = self.head_residual_probe.abs().mean().clamp_min(1e-8)
            should_reuse = bool(((difference / reference) <= FIRST_BLOCK_THRESHOLD).item())

        if should_reuse:
            self.skipped += 1
            self.consecutive_skips += 1
            self.step += 1
            return True

        self.head_residual_probe = residual_probe.detach()
        # Only dense steps need the full head anchor. Previously both the input and output were cloned every step;
        # probing the input before block 0 removes one full-sequence copy from cache-hit steps entirely.
        self.first_block_output = block_output.detach().clone()
        return False

    def apply_cached_tail(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply Cache-DiT's cached residual and first-order TaylorSeer forecast without full-size temporaries."""
        hidden_states.add_(self.tail_residual)
        if self.tail_residual_slope is not None and self.last_tail_step is not None:
            hidden_states.add_(self.tail_residual_slope, alpha=self.step - self.last_tail_step)
        return hidden_states

    def update_first_block_tail(self, final_block_output: torch.Tensor) -> None:
        if self.first_block_output is None:
            return
        new_tail = (final_block_output - self.first_block_output).detach()
        if self.tail_residual is not None and self.last_tail_step is not None:
            gap = max(1, self.step - self.last_tail_step)
            self.tail_residual_slope = (new_tail - self.tail_residual) / gap
        self.tail_residual = new_tail
        self.last_tail_step = self.step
        self.last_actual_step = self.step
        self.consecutive_skips = 0
        self.step += 1
        self.first_block_output = None

    def update(self, video_input, audio_input, video_output, audio_output, condition_rows: int) -> None:
        # Balanced's clock and state are updated at the block-stack boundary by FirstBlockCache.
        if self.enabled and not self.profile.startswith("ultra"):
            return
        if self.pending_track:
            sampled_output = video_output[0, condition_rows::EASYCACHE_SUBSAMPLE].detach().float()
            if self.previous_output is not None and self.pending_input_change is not None:
                output_change = (sampled_output - self.previous_output).abs().mean()
                self.relative_rate = output_change / self.pending_input_change.clamp_min(1e-6)
            self.previous_input = self.pending_input.clone()
            self.previous_output = sampled_output.clone()
            self.previous_output_norm = sampled_output.abs().mean()
            if not self.profile.startswith("ultra"):
                self.video_residual = (video_output - video_input).detach()
                self.audio_residual = (audio_output - audio_input).detach()
            self.accumulated_change = None
        if self.profile.startswith("ultra"):
            new_video_residual = (video_output - video_input).detach()
            new_audio_residual = (audio_output - audio_input).detach()
            if self.video_residual is not None and self.last_actual_step is not None:
                gap = max(1, self.step - self.last_actual_step)
                self.video_residual_slope = (new_video_residual - self.video_residual) / gap
                self.audio_residual_slope = (new_audio_residual - self.audio_residual) / gap
            self.video_residual = new_video_residual
            self.audio_residual = new_audio_residual
        self.last_actual_step = self.step
        self.consecutive_skips = 0
        self.step += 1
        self.pending_input = None
        self.pending_input_change = None
        self.pending_track = False

    def finish(self) -> dict:
        stats = {
            "steps": self.step,
            "computed": max(0, self.step - self.skipped),
            "forecasted": self.skipped,
            "profile": self.profile,
        }
        if self.enabled and self.step:
            computed = max(1, self.step - self.skipped)
            print(
                f"[h3-nvfp4] Cache-DiT reused {self.skipped}/{self.step} block-stack evaluations "
                f"({self.step / computed:.2f}x denoiser-work reduction)",
                flush=True,
            )
        self.begin(None)
        return stats


class H3SolAttention:
    """NVIDIA Sol-Attn policy adapted to H3's single-GPU packed attention.

    The packed prefix (text, conditioning video and generated audio) remains an exact KV sink and its query rows are
    recomputed densely. Only target-video query/key interactions become sparse, after ten dense denoising steps and
    outside the first two transformer blocks. Any unavailable/JIT-failing backend falls back to cuDNN for the request.
    """

    def __init__(self):
        self.enabled = SOL_ATTN
        self.step = 0
        self.video_start = 0
        self.sparse_calls = 0
        self.dense_calls = 0
        self.failure = None

    def begin(self):
        self.step = 0
        self.video_start = 0
        self.sparse_calls = 0
        self.dense_calls = 0
        self.failure = None

    def observe(self, video_indices: torch.Tensor, sequence: int, step: int) -> None:
        self.step = int(step)
        if not self.video_start:
            deltas = video_indices[1:] - video_indices[:-1]
            breaks = (deltas != 1).nonzero().flatten()
            start = int(breaks[-1]) + 1 if len(breaks) else 0
            self.video_start = int(video_indices[start]) if video_indices.numel() else sequence

    def __call__(self, query, key, value, layer: int):
        tokens = int(query.shape[1])
        if (
            not self.enabled
            or self.failure is not None
            or self.step < SOL_ATTN_DENSE_STEPS
            or layer < SOL_ATTN_DENSE_LAYERS
            or tokens < SOL_ATTN_MIN_TOKENS
            or not 0 < self.video_start < tokens
        ):
            self.dense_calls += 1
            return None
        try:
            if SOL_ATTN_BACKEND == "triton":
                from sol_attn.triton_ref import sol_attn
            else:
                from sol_attn import sol_attn

            q, k, v = (tensor.contiguous() for tensor in (query, key, value))
            kwargs = {
                "tau": SOL_ATTN_TAU,
                "thresh_type": "diag",
                "sink_start": 0,
                "sink_tokens": self.video_start,
            }
            if SOL_ATTN_BACKEND != "triton":
                kwargs["kv_splits"] = 1
            attended = sol_attn(q, k, v, **kwargs)
            # An exact KV sink does not make the prefix's own queries dense. H3 jointly generates audio in that
            # prefix, so reproduce those rows with exact attention as NVIDIA's H3 integration does.
            prefix = self.video_start
            dense_prefix = F.scaled_dot_product_attention(
                q[:, :prefix].transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)
            attended[:, :prefix] = dense_prefix
            self.sparse_calls += 1
            return attended
        except Exception as error:
            self.failure = f"{type(error).__name__}: {error}"
            print(f"[h3-sol-attn] falling back to dense attention: {self.failure}", flush=True)
            self.dense_calls += 1
            return None


def _quant_config(handle, prefix: str) -> dict | None:
    key = f"{prefix}.comfy_quant"
    if key not in handle.keys():
        return None
    return json.loads(handle.get_tensor(key).numpy().tobytes())


class H3Linear(nn.Module):
    """A plain or comfy-kitchen NVFP4 linear, selected by checkpoint metadata."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        self.register_buffer("input_scale", None)
        self.register_buffer("pre_quant_scale", None)
        self.quantized = False
        self.full_precision_mm = False
        # LoRAs stay as an activation-space BF16 branch. Merging into NVFP4 would requantize most of the small update
        # away; these are deliberately plain request-local tensors so Space packing never duplicates optional LoRAs.
        self.lora_a = None
        self.lora_b = None
        self.lora_parts = None
        self.lora_scale = 0.0

    def set_lora(self, a: torch.Tensor | None, b: torch.Tensor | None, scale: float = 1.0) -> None:
        self.lora_a, self.lora_b, self.lora_scale = a, b, float(scale)
        self.lora_parts = None

    def set_lora_parts(self, parts: list[tuple[torch.Tensor, torch.Tensor, int, int]], scale: float = 1.0) -> None:
        """Attach separate low-rank updates to slices of one fused output matrix (Diffusers Q/K/V -> packed QKV)."""
        self.lora_a = self.lora_b = None
        # Concatenate A once at activation time. Doing this inside every transformer forward would copy hundreds of MB
        # over a four-step request and erase much of the Turbo speedup.
        self.lora_parts = (
            torch.cat([a for a, _, _, _ in parts]),
            [(b, start, end, a.shape[0]) for a, b, start, end in parts],
        )
        self.lora_scale = float(scale)

    def _apply_lora(self, source: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        if self.lora_parts and self.lora_scale != 0.0:
            # Fuse the three input projections into one GEMM, then retain separate B matrices so the block-diagonal
            # output projection does not waste 3x the arithmetic on zeros.
            lora_a, outputs = self.lora_parts
            low_rank = F.linear(source.to(lora_a.dtype), lora_a)
            offset = 0
            for b, start, end, rank in outputs:
                delta = F.linear(low_rank[..., offset : offset + rank], b)
                output[..., start:end].add_(delta.to(output.dtype), alpha=self.lora_scale)
                offset += rank
            return output
        if self.lora_a is None or self.lora_b is None or self.lora_scale == 0.0:
            return output
        delta = F.linear(F.linear(source.to(self.lora_a.dtype), self.lora_a), self.lora_b)
        return output.add_(delta.to(output.dtype), alpha=self.lora_scale)

    def load(self, handle, prefix: str) -> None:
        config = _quant_config(handle, prefix)
        weight = handle.get_tensor(f"{prefix}.weight")

        if config is None:
            self.weight = nn.Parameter(
                weight if self.compute_dtype is None else weight.to(self.compute_dtype), requires_grad=False
            )
        elif config.get("format") == "nvfp4":
            block_scale = handle.get_tensor(f"{prefix}.weight_scale")
            if block_scale.dtype == torch.uint8:
                block_scale = block_scale.view(torch.float8_e4m3fn)
            tensor_scale = handle.get_tensor(f"{prefix}.weight_scale_2").float()
            params = TensorCoreNVFP4Layout.Params(
                scale=tensor_scale,
                block_scale=block_scale,
                orig_dtype=torch.bfloat16,
                orig_shape=(self.out_features, self.in_features),
            )
            quantized = QuantizedTensor(weight.to(torch.uint8), "TensorCoreNVFP4Layout", params)
            self.weight = nn.Parameter(quantized, requires_grad=False)
            self.quantized = True
            self.full_precision_mm = bool(config.get("full_precision_matrix_mult", False))
            for name in ("input_scale", "pre_quant_scale"):
                key = f"{prefix}.{name}"
                if key in handle.keys():
                    setattr(self, name, handle.get_tensor(key))
        else:
            raise ValueError(f"Unsupported quantization on {prefix}: {config}")

        bias_key = f"{prefix}.bias"
        if bias_key in handle.keys():
            bias = handle.get_tensor(bias_key)
            self.bias = nn.Parameter(
                bias if self.compute_dtype is None else bias.to(self.compute_dtype), requires_grad=False
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        lora_input = hidden_states
        if self.pre_quant_scale is not None:
            hidden_states = hidden_states * self.pre_quant_scale.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
        if not self.quantized:
            hidden_states = hidden_states.to(self.weight.dtype)
            output = F.linear(
                hidden_states,
                self.weight,
                self.bias,
            )
            return self._apply_lora(lora_input, output)

        if self.full_precision_mm:
            # Some AWQ checkpoints use NVFP4 as a compact weight format but deliberately retain BF16 activations and
            # GEMMs. Dequantization is layer-local, so residency stays compact without adding activation error.
            weight = self.weight.dequantize().to(hidden_states.dtype)
            output = F.linear(hidden_states, weight, None if self.bias is None else self.bias.to(hidden_states.dtype))
            return self._apply_lora(lora_input, output)

        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        scale = None if self.input_scale is None else self.input_scale.to(flat.device)
        quantized_input = QuantizedTensor.from_float(flat, "TensorCoreNVFP4Layout", scale=scale)
        output = F.linear(
            quantized_input,
            self.weight,
            None if self.bias is None else self.bias.to(hidden_states.dtype),
        )
        output = output.reshape(*shape[:-1], self.out_features)
        return self._apply_lora(lora_input, output)


class H3RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = EPS):
        super().__init__()
        self.width = width
        self.eps = eps
        self.register_parameter("weight", None)

    def load(self, handle, prefix: str) -> None:
        self.weight = nn.Parameter(handle.get_tensor(f"{prefix}.weight"), requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            hidden_states,
            (self.width,),
            self.weight,
            self.eps,
        )


class H3Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = H3Linear(HIDDEN, 3 * HEADS * HEAD_DIM)
        self.q_norm = H3RMSNorm(HEAD_DIM)
        self.k_norm = H3RMSNorm(HEAD_DIM)
        self.out_proj = H3Linear(HEADS * HEAD_DIM, HIDDEN)

    def load(self, handle, prefix: str) -> None:
        self.qkv_proj.load(handle, f"{prefix}.qkv_proj")
        self.q_norm.load(handle, f"{prefix}.q_norm")
        self.k_norm.load(handle, f"{prefix}.k_norm")
        self.out_proj.load(handle, f"{prefix}.out_proj")

    def forward(self, hidden_states, rope_table, backend: str, sparse=None, layer: int = -1):
        sequence = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)
        query, key, value = qkv.split(HEADS * HEAD_DIM, dim=-1)
        query = query.view(1, sequence, HEADS, HEAD_DIM)
        key = key.view(1, sequence, HEADS, HEAD_DIM)
        value = value.view(1, sequence, HEADS, HEAD_DIM)

        # One in-place kernel replaces Q RMSNorm, K RMSNorm and both partial RoPE applications.
        kitchen.rms_rope_split_half_(
            query,
            key,
            rope_table,
            self.q_norm.weight,
            self.k_norm.weight,
            epsilon=self.q_norm.eps,
            rot_dim=rope_table.shape[-3] * 2,
        )
        attended = sparse(query, key, value, layer) if sparse is not None else None
        if attended is None:
            attended = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=backend,
            )
        return self.out_proj(attended.reshape(sequence, HEADS * HEAD_DIM))


class H3MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = H3Linear(HIDDEN, 2 * FFN)
        self.fc2 = H3Linear(FFN, HIDDEN)

    def load(self, handle, prefix: str) -> None:
        self.fc1.load(handle, f"{prefix}.fc1")
        self.fc2.load(handle, f"{prefix}.fc2")

    def forward(self, hidden_states):
        gate, up = self.fc1(hidden_states).chunk(2, dim=-1)
        return self.fc2(F.silu(gate).mul_(up))


class H3RefinerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = H3RMSNorm(HIDDEN)
        self.attn = H3Attention()
        self.norm2 = H3RMSNorm(HIDDEN)
        self.mlp = H3MLP()

    def load(self, handle, prefix: str) -> None:
        self.norm1.load(handle, f"{prefix}.norm1")
        self.attn.load(handle, f"{prefix}.attn")
        self.norm2.load(handle, f"{prefix}.norm2")
        self.mlp.load(handle, f"{prefix}.mlp")


class H3AdaLN(nn.Module):
    def __init__(self, expand: int, modalities: int):
        super().__init__()
        self.expand = expand
        self.modalities = modalities
        # Curve checkpoints deliberately evaluate interpolation and modulation projection in FP32. Expanding the
        # checkpoint's tiny FP16 [*, 8] matrices once at load avoids 51 request-step casts.
        self.linear = H3Linear(
            TIME_DIM, expand * HIDDEN * modalities, bias=True, compute_dtype=torch.float32
        )
        self.lora_a = None
        self.lora_b = None
        self.lora_scale = 0.0

    def set_lora(self, a: torch.Tensor | None, b: torch.Tensor | None, scale: float = 1.0) -> None:
        self.lora_a, self.lora_b, self.lora_scale = a, b, float(scale)

    def load(self, handle, prefix: str) -> None:
        self.linear.load(handle, f"{prefix}.linear")

    def forward(self, time_embedding, output_dtype=None, lora_time_embedding=None):
        projected = self.linear(time_embedding)
        if self.lora_a is not None and lora_time_embedding is not None and self.lora_scale != 0.0:
            delta = F.linear(F.linear(lora_time_embedding.to(self.lora_a.dtype), self.lora_a), self.lora_b)
            projected = projected.add_(delta.to(projected.dtype), alpha=self.lora_scale)
        if output_dtype is not None:
            # One contiguous conversion is numerically identical to converting the six chunk views independently,
            # and removes five CUDA launches from every one of the 50 blocks.
            projected = projected.to(output_dtype)
        projected = projected.view(-1, self.expand * HIDDEN)
        return projected.chunk(self.expand, dim=-1)


class H3Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = H3RMSNorm(HIDDEN)
        self.attn = H3Attention()
        self.norm2 = H3RMSNorm(HIDDEN)
        self.mlp = H3MLP()
        self.adaln_proj = H3AdaLN(6, 3)

    def load(self, handle, prefix: str) -> None:
        self.norm1.load(handle, f"{prefix}.norm1")
        self.attn.load(handle, f"{prefix}.attn")
        self.norm2.load(handle, f"{prefix}.norm2")
        self.mlp.load(handle, f"{prefix}.mlp")
        self.adaln_proj.load(handle, f"{prefix}.adaln_proj")


class H3FinalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = H3RMSNorm(HIDDEN)
        self.adaln_proj = H3AdaLN(2, 1)
        self.video_out = H3Linear(HIDDEN, VIDEO_DIM, bias=True, compute_dtype=torch.float32)
        self.audio_out = H3Linear(HIDDEN, AUDIO_DIM, bias=True, compute_dtype=torch.float32)

    def load(self, handle, prefix: str) -> None:
        self.norm.load(handle, f"{prefix}.norm")
        self.adaln_proj.load(handle, f"{prefix}.adaln_proj")
        self.video_out.load(handle, f"{prefix}.video_out")
        self.audio_out.load(handle, f"{prefix}.audio_out")


class H3NVFP4Transformer(nn.Module):
    """Diffusers-compatible H3 transformer backed by fused comfy-kitchen NVFP4 kernels."""

    def __init__(self):
        super().__init__()
        # The modular pipeline reads these values through the diffusers component config rather than inspecting the
        # module itself. Keep the public transformer contract even though this lean adapter is not a ConfigMixin.
        self.config = SimpleNamespace(
            patch_size=(1, 2, 2),
            in_channels=24,
            audio_in_channels=AUDIO_DIM,
            text_dim=TEXT_DIM,
        )
        self.video_patch_proj = H3Linear(VIDEO_DIM, HIDDEN, bias=True, compute_dtype=torch.float32)
        self.audio_patch_proj = H3Linear(AUDIO_DIM, HIDDEN, bias=True, compute_dtype=torch.float32)
        self.condition_proj = H3Linear(TEXT_DIM, HIDDEN, bias=True)
        self.token_refiner = nn.ModuleList([H3RefinerBlock() for _ in range(REFINER_LAYERS)])
        self.token_refiner_norm = H3RMSNorm(HIDDEN)
        self.blocks = nn.ModuleList([H3Block() for _ in range(LAYERS)])
        self.final_layer = H3FinalLayer()
        self.register_buffer("adaln_t_table", None)
        self.register_buffer("rope_inv_freq", None)
        self.attention_backend = "_native_cudnn"
        self._text_cache = None
        self._rope_cache = None
        self._segment_cache = None
        self._condition_video_rows = None
        self._condition_video_embedding = None
        self._output_indices = None
        self._generated_rows = None
        self._lora_egrid = None
        self._active_lora = None
        self._active_lora_modules = 0
        self._step_cache = H3StepCache()
        self._sol_attention = H3SolAttention()

    @property
    def dtype(self) -> torch.dtype:
        """Match ModelMixin's placement contract used by ModularPipeline.to()."""
        return self.condition_proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.adaln_t_table.device

    def load(self, path: str) -> None:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as handle:
            self.video_patch_proj.load(handle, "video_patch_proj")
            self.audio_patch_proj.load(handle, "audio_patch_proj")
            self.condition_proj.load(handle, "condition_proj")
            for index, block in enumerate(self.token_refiner):
                block.load(handle, f"token_refiner.blocks.{index}")
            self.token_refiner_norm.load(handle, "token_refiner.final_norm")
            for index, block in enumerate(self.blocks):
                block.load(handle, f"blocks.{index}")
            self.final_layer.load(handle, "final_layer")
            self.adaln_t_table = handle.get_tensor("adaln_t_table")
            self.rope_inv_freq = handle.get_tensor("rope.inv_freq")
        # Every loaded tensor is already a frozen Parameter (or a buffer). Avoid mutating the quantized tensor
        # subclass through a redundant requires_grad_ dispatch.
        self.eval()

    def set_attention_backend(self, backend: str) -> None:
        self.attention_backend = backend

    @staticmethod
    def _lora_module_name(name: str) -> str:
        name = name.removeprefix("diffusion_model.")
        # Released adapters use several names for the Diffusers containers; this engine keeps both as bare ModuleLists.
        return (
            name.replace("token_refiner.refiner_blocks.", "token_refiner.")
            .replace("token_refiner.blocks.", "token_refiner.")
            .replace("transformer_blocks.", "blocks.")
            .replace(".attn.to_out.0", ".attn.out_proj")
            .replace(".ff.net.0.proj", ".mlp.fc1")
            .replace(".ff.net.2", ".mlp.fc2")
        )

    def clear_lora(self) -> None:
        for module in self.modules():
            if isinstance(module, (H3Linear, H3AdaLN)):
                module.set_lora(None, None, 0.0)
        self._lora_egrid = None
        self._active_lora = None
        self._active_lora_modules = 0

    def activate_lora(self, path: str | None, egrid_path: str | None, strength: float = 1.0) -> int:
        """Attach one standard H3 A/B safetensors LoRA without dequantizing the NVFP4 base."""
        self.clear_lora()
        if not path:
            return 0
        from safetensors import safe_open

        staged = []
        staged_parts = []
        needs_egrid = False
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            pairs = []
            for a_suffix, b_suffix in (
                (".lora_A.weight", ".lora_B.weight"),
                (".lora_A.default.weight", ".lora_B.default.weight"),
            ):
                pairs.extend(
                    (key[: -len(a_suffix)], key, key[: -len(a_suffix)] + b_suffix)
                    for key in keys
                    if key.endswith(a_suffix)
                )
            tensors = {}
            for raw_name, a_key, b_key in sorted(pairs):
                if b_key not in keys:
                    raise ValueError(f"LoRA is missing {b_key}")
                a, b = handle.get_tensor(a_key), handle.get_tensor(b_key)
                if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1] or a.shape[0] > 256:
                    raise ValueError(f"Invalid H3 LoRA matrices for {raw_name}: A{tuple(a.shape)} B{tuple(b.shape)}")
                alpha_key = raw_name + ".alpha"
                rank_scale = float(handle.get_tensor(alpha_key).item()) / a.shape[0] if alpha_key in keys else 1.0
                tensors[raw_name] = (a, b, rank_scale)

            consumed = set()
            for raw_name in sorted(tensors):
                if raw_name in consumed:
                    continue
                if raw_name.endswith(".attn.to_q"):
                    base = raw_name.removesuffix(".attn.to_q")
                    qkv_names = [f"{base}.attn.to_{part}" for part in "qkv"]
                    if not all(name in tensors for name in qkv_names):
                        raise ValueError(f"Fused QKV LoRA is incomplete for {base}")
                    name = self._lora_module_name(f"{base}.attn.qkv_proj")
                    module = self.get_submodule(name)
                    if not isinstance(module, H3Linear) or module.out_features % 3:
                        raise ValueError(f"Unsupported fused H3 QKV target: {base}")
                    output_width = module.out_features // 3
                    parts = []
                    scales = set()
                    for index, qkv_name in enumerate(qkv_names):
                        a, b, rank_scale = tensors[qkv_name]
                        if a.shape[1] != module.in_features or b.shape[0] != output_width:
                            raise ValueError(
                                f"LoRA shape mismatch for {qkv_name}: A{tuple(a.shape)} B{tuple(b.shape)}, "
                                f"expected [rank,{module.in_features}] and [{output_width},rank]"
                            )
                        parts.append(
                            (
                                a.to(self.device, torch.bfloat16),
                                b.to(self.device, torch.bfloat16),
                                index * output_width,
                                (index + 1) * output_width,
                            )
                        )
                        scales.add(rank_scale)
                    if len(scales) != 1:
                        raise ValueError(f"Q/K/V LoRA scales differ for {base}")
                    staged_parts.append((module, parts, strength * scales.pop()))
                    consumed.update(qkv_names)
                    continue
                if raw_name.endswith((".attn.to_k", ".attn.to_v")):
                    # Alphabetically K and V can appear before Q. The Q branch above validates and consumes the whole
                    # group when it is reached; defer these names instead of rejecting a correctly ordered state dict.
                    continue

                a, b, rank_scale = tensors[raw_name]
                name = self._lora_module_name(raw_name)
                if name.endswith(".adaln_proj.linear"):
                    module = self.get_submodule(name.removesuffix(".linear"))
                    expected_in = 2688
                    expected_out = module.expand * HIDDEN * module.modalities
                    needs_egrid = True
                else:
                    module = self.get_submodule(name)
                    if not isinstance(module, H3Linear):
                        raise ValueError(f"Unsupported H3 LoRA target: {raw_name}")
                    expected_in, expected_out = module.in_features, module.out_features
                if a.ndim != 2 or b.ndim != 2 or a.shape[1] != expected_in or b.shape[0] != expected_out:
                    raise ValueError(
                        f"LoRA shape mismatch for {raw_name}: A{tuple(a.shape)} B{tuple(b.shape)}, "
                        f"expected [rank,{expected_in}] and [{expected_out},rank]"
                    )
                staged.append(
                    (module, a.to(self.device, torch.bfloat16), b.to(self.device, torch.bfloat16), strength * rank_scale)
                )

            dangling_qkv = sorted(
                name
                for name in tensors
                if name.endswith((".attn.to_k", ".attn.to_v")) and name not in consumed
            )
            if dangling_qkv:
                raise ValueError(f"Fused QKV LoRA has no matching query tensor for {dangling_qkv[0]}")

        if not staged and not staged_parts:
            raise ValueError("No supported H3 LoRA A/B tensor pairs were found.")
        if needs_egrid:
            if not egrid_path:
                raise ValueError("This pruned-base LoRA needs the H3 timestep embedding grid.")
            with safe_open(egrid_path, framework="pt", device="cpu") as handle:
                grid = handle.get_tensor("silu_t_emb_grid")
            if grid.ndim != 2 or grid.shape[1] != 2688:
                raise ValueError(f"Invalid H3 timestep grid shape: {tuple(grid.shape)}")
            self._lora_egrid = grid.to(self.device, torch.bfloat16)
        for module, a, b, scale in staged:
            module.set_lora(a, b, scale)
        for module, parts, scale in staged_parts:
            module.set_lora_parts(parts, scale)
        self._active_lora = os.path.basename(path)
        self._active_lora_modules = len(staged) + sum(len(parts) for _, parts, _ in staged_parts)
        print(
            f"[h3-lora] active {self._active_lora}: {self._active_lora_modules} targets, "
            f"strength {float(strength):g}",
            flush=True,
        )
        return self._active_lora_modules

    def _lora_time_embedding(self, timestep: torch.Tensor) -> torch.Tensor | None:
        if self._lora_egrid is None:
            return None
        position = timestep.float().clamp(0.0, 1.0) * (self._lora_egrid.shape[0] - 1)
        lower = position.floor().long().clamp(max=self._lora_egrid.shape[0] - 2)
        return torch.lerp(
            self._lora_egrid[lower],
            self._lora_egrid[lower + 1],
            (position - lower).to(self._lora_egrid.dtype).unsqueeze(1),
        )

    def begin_request(self, total_steps: int | None = None, profile: str = "balanced") -> None:
        self._text_cache = None
        self._rope_cache = None
        self._segment_cache = None
        self._condition_video_rows = None
        self._condition_video_embedding = None
        self._output_indices = None
        self._generated_rows = None
        self._step_cache.begin(total_steps, profile)
        self._sol_attention.begin()

    def end_request(self) -> dict:
        stats = self._step_cache.finish()
        stats["sol_sparse_calls"] = self._sol_attention.sparse_calls
        stats["sol_dense_calls"] = self._sol_attention.dense_calls
        stats["sol_failure"] = self._sol_attention.failure
        stats["lora"] = self._active_lora
        stats["lora_modules"] = self._active_lora_modules
        self._text_cache = None
        self._rope_cache = None
        self._segment_cache = None
        self._condition_video_rows = None
        self._condition_video_embedding = None
        self._output_indices = None
        self._generated_rows = None
        return stats

    def _refine_text(self, text_states: torch.Tensor) -> torch.Tensor:
        key = (text_states.data_ptr(), tuple(text_states.shape), text_states.device)
        if self._text_cache is not None and self._text_cache[0] == key:
            return self._text_cache[1]
        hidden = self.condition_proj(text_states)
        # Text is tiny compared with the video sequence; use the same fused QKV path with an identity RoPE omitted.
        for block in self.token_refiner:
            residual = hidden
            normalized = block.norm1(hidden)
            qkv = block.attn.qkv_proj(normalized)
            query, key_states, value = qkv.split(HEADS * HEAD_DIM, dim=-1)
            query = block.attn.q_norm(query.view(1, -1, HEADS, HEAD_DIM))
            key_states = block.attn.k_norm(key_states.view(1, -1, HEADS, HEAD_DIM))
            value = value.view(1, -1, HEADS, HEAD_DIM)
            attended = dispatch_attention_fn(
                query,
                key_states,
                value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self.attention_backend,
            ).reshape(-1, HEADS * HEAD_DIM)
            hidden = residual + block.attn.out_proj(attended)
            hidden = hidden + block.mlp(block.norm2(hidden))
        hidden = self.token_refiner_norm(hidden)
        self._text_cache = (key, hidden)
        return hidden

    def _rope(self, position_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        key = (position_ids.data_ptr(), tuple(position_ids.shape), position_ids.device, dtype)
        if self._rope_cache is not None and self._rope_cache[0] == key:
            return self._rope_cache[1]
        positions = position_ids.to(torch.float32)
        frequencies = positions.unsqueeze(-1) * self.rope_inv_freq.to(position_ids.device).view(1, 1, -1)
        temporal, height, width = frequencies.unbind(dim=1)
        angles = torch.cat((temporal, height, width), dim=-1)
        cosine, sine = angles.cos(), angles.sin()
        table = torch.stack((cosine, -sine, sine, cosine), dim=-1)
        table = table.reshape(1, position_ids.shape[0], 1, angles.shape[-1], 2, 2).to(dtype)
        self._rope_cache = (key, table)
        return table

    def _time_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        table = self.adaln_t_table.to(timestep.device)
        position = timestep.float().clamp(0.0, 1.0) * (table.shape[0] - 1)
        lower = position.floor().long().clamp(max=table.shape[0] - 2)
        return torch.lerp(
            table[lower],
            table[lower + 1],
            (position - lower).to(table.dtype).unsqueeze(1),
        )

    def _segments(self, indices: torch.Tensor):
        if self._segment_cache is None:
            host = indices.detach().cpu()
            changes = (host[1:] != host[:-1]).nonzero().flatten().add(1).tolist()
            bounds = [0, *changes, len(host)]
            # Python row ids avoid indexing modulation tensors with CUDA scalar tensors in every block.
            self._segment_cache = [
                (start, stop, int(host[start])) for start, stop in zip(bounds[:-1], bounds[1:])
            ]
        return self._segment_cache

    def _video_layout(self, video_indices: torch.Tensor) -> int:
        """Number of leading, static keyframe-patch rows in the video latent tensor."""
        if self._condition_video_rows is None:
            host = video_indices.detach().cpu()
            discontinuities = (host[1:] - host[:-1] != 1).nonzero().flatten()
            self._condition_video_rows = int(discontinuities[0]) + 1 if len(discontinuities) else 0
        return self._condition_video_rows

    def _project_video(self, hidden_states: torch.Tensor, condition_rows: int, dtype: torch.dtype) -> torch.Tensor:
        source = hidden_states[0]
        if condition_rows == 0:
            return self.video_patch_proj(source.float()).to(dtype)
        if self._condition_video_embedding is None:
            self._condition_video_embedding = self.video_patch_proj(source[:condition_rows].float()).to(dtype)
        generated = self.video_patch_proj(source[condition_rows:].float()).to(dtype)
        return torch.cat((self._condition_video_embedding, generated), dim=0)

    @staticmethod
    def _modulate(hidden, shift, scale, row_ids, segments):
        if FUSED_ADALN and hidden.is_cuda and hidden.is_contiguous():
            _adaln_modulate_kernel[(triton.cdiv(hidden.numel(), 256),)](
                hidden, shift, scale, row_ids, hidden.numel(), HIDDEN, shift.stride(0), num_warps=4
            )
            return hidden
        for start, stop, row in segments:
            hidden[start:stop].mul_(1.0 + scale[row]).add_(shift[row])
        return hidden

    @staticmethod
    def _gate(hidden, update, gate, row_ids, segments):
        if FUSED_ADALN and hidden.is_cuda and hidden.is_contiguous() and update.is_contiguous():
            _adaln_gate_kernel[(triton.cdiv(hidden.numel(), 256),)](
                hidden, update, gate, row_ids, hidden.numel(), HIDDEN, gate.stride(0), num_warps=4
            )
            return hidden
        for start, stop, row in segments:
            hidden[start:stop].addcmul_(update[start:stop], gate[row])
        return hidden

    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        timestep,
        timestep_indices,
        token_tags,
        position_ids,
        video_indices,
        audio_indices,
        text_indices,
        attention_kwargs=None,
        return_dict=True,
    ):
        from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerOutput

        if hidden_states.shape[0] != 1:
            raise ValueError("The NVFP4 MiniMax-H3 engine supports batch size 1.")

        condition_rows = self._video_layout(video_indices)
        reused = self._step_cache.try_reuse(hidden_states, audio_hidden_states, condition_rows)
        if reused is not None:
            video_output, audio_output = reused
            if not return_dict:
                return video_output, audio_output
            return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)

        text = self._refine_text(encoder_hidden_states[0].to(torch.bfloat16))
        video = self._project_video(hidden_states, condition_rows, text.dtype)
        audio = self.audio_patch_proj(audio_hidden_states[0].float()).to(text.dtype)
        # Text, video and audio indices partition the packed sequence, so initialization would only add a full HBM
        # write before the three index copies overwrite every row.
        packed = text.new_empty((position_ids.shape[0], HIDDEN))
        packed.index_copy_(0, text_indices, text)
        packed.index_copy_(0, video_indices, video)
        packed.index_copy_(0, audio_indices, audio)

        time_embedding = self._time_embedding(timestep)
        lora_time_embedding = self._lora_time_embedding(timestep)
        adaln_indices = timestep_indices * 3 + token_tags.clamp(min=0)
        segments = self._segments(adaln_indices)
        rope = self._rope(position_ids, packed.dtype)
        use_sol_attention = self._step_cache.profile != "exact" and self._sol_attention.enabled
        self._sol_attention.observe(video_indices, packed.shape[0], self._step_cache.step)

        reused_tail = False
        for layer, block in enumerate(self.blocks):
            if layer == 0:
                # Block 0 updates the packed tensor in place. Cache-DiT only needs a strided input probe for its
                # decision; a full anchor is created later only when the remaining blocks really run.
                block_input_probe = packed[::FIRST_BLOCK_SUBSAMPLE].detach().clone()
            # One conversion per small modulation table, rather than one conversion per sequence segment.
            modulations = block.adaln_proj(time_embedding, packed.dtype, lora_time_embedding)
            shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = modulations
            normalized = self._modulate(block.norm1(packed), shift_attn, scale_attn, adaln_indices, segments)
            packed = self._gate(
                packed,
                block.attn(
                    normalized,
                    rope,
                    self.attention_backend,
                    self._sol_attention if use_sol_attention else None,
                    layer,
                ),
                gate_attn,
                adaln_indices,
                segments,
            )
            normalized = self._modulate(block.norm2(packed), shift_mlp, scale_mlp, adaln_indices, segments)
            packed = self._gate(packed, block.mlp(normalized), gate_mlp, adaln_indices, segments)

            if layer == 0:
                if self._step_cache.first_block_decision(block_input_probe, packed):
                    packed = self._step_cache.apply_cached_tail(packed)
                    reused_tail = True
                    break
            if layer == len(self.blocks) - 1 and not reused_tail:
                self._step_cache.update_first_block_tail(packed)

        shift, scale = self.final_layer.adaln_proj(time_embedding, lora_time_embedding=lora_time_embedding)

        # Keyframe output rows are discarded by the scheduler. Avoid their FP32 output projection and put zeros in
        # those unused slots to retain the pipeline's expected tensor shape.
        generated_video_indices = video_indices[condition_rows:]
        if self._output_indices is None:
            self._generated_rows = generated_video_indices.shape[0]
            self._output_indices = torch.cat((generated_video_indices, audio_indices))
        generated_rows = self._generated_rows
        normalized_output = self.final_layer.norm(packed.index_select(0, self._output_indices))
        video_times = timestep_indices.index_select(0, generated_video_indices)
        video_hidden = normalized_output[:generated_rows]
        video_hidden = video_hidden * (1.0 + scale.index_select(0, video_times)) + shift.index_select(0, video_times)
        generated_video_output = self.final_layer.video_out(video_hidden.float())
        if condition_rows:
            video_output = generated_video_output.new_zeros((1, hidden_states.shape[1], VIDEO_DIM))
            video_output[0, condition_rows:] = generated_video_output
        else:
            video_output = generated_video_output.unsqueeze(0)

        audio_times = timestep_indices.index_select(0, audio_indices)
        audio_hidden = normalized_output[generated_rows:]
        audio_hidden = audio_hidden * (1.0 + scale.index_select(0, audio_times)) + shift.index_select(0, audio_times)
        audio_output = self.final_layer.audio_out(audio_hidden.float()).unsqueeze(0)

        self._step_cache.update(
            hidden_states,
            audio_hidden_states,
            video_output,
            audio_output,
            condition_rows,
        )

        if not return_dict:
            return video_output, audio_output
        return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)


def load_transformer() -> H3NVFP4Transformer:
    if torch.version.cuda is None or int(torch.version.cuda.split(".")[0]) < 13:
        raise RuntimeError("NVFP4 requires the CUDA 13 PyTorch build.")
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=NVFP4_REPO, filename=NVFP4_FILE)
    transformer = H3NVFP4Transformer()
    transformer.load(path)
    print(f"[h3-nvfp4] loaded {NVFP4_REPO}/{NVFP4_FILE}", flush=True)
    return transformer


def status() -> str:
    return (
        f"NVFP4 · Cache-DiT F1B0+TaylorSeer {FIRST_BLOCK_THRESHOLD:g} / forecast {FORECAST_BLEND:g} · "
        f"pruned AdaLN curve · fused QKV/QK-norm/RoPE · `{NVFP4_REPO}`"
    )
