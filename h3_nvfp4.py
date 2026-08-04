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
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def load(self, handle, prefix: str) -> None:
        config = _quant_config(handle, prefix)
        weight = handle.get_tensor(f"{prefix}.weight")

        if config is None:
            self.weight = nn.Parameter(
                weight if self.compute_dtype is None else weight.to(self.compute_dtype), requires_grad=False
            )
        elif config.get("format") == "nvfp4":
            from comfy_kitchen.tensor import QuantizedTensor, TensorCoreNVFP4Layout

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
        if self.pre_quant_scale is not None:
            hidden_states = hidden_states * self.pre_quant_scale.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
        if not self.quantized:
            hidden_states = hidden_states.to(self.weight.dtype)
            return F.linear(
                hidden_states,
                self.weight,
                self.bias,
            )

        from comfy_kitchen.tensor import QuantizedTensor

        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        scale = None if self.input_scale is None else self.input_scale.to(flat.device)
        quantized_input = QuantizedTensor.from_float(flat, "TensorCoreNVFP4Layout", scale=scale)
        output = F.linear(
            quantized_input,
            self.weight,
            None if self.bias is None else self.bias.to(hidden_states.dtype),
        )
        return output.reshape(*shape[:-1], self.out_features)


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
            self.weight.to(device=hidden_states.device, dtype=hidden_states.dtype),
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

    def forward(self, hidden_states, rope_table, backend: str):
        import comfy_kitchen as kitchen
        from diffusers.models.attention_dispatch import dispatch_attention_fn

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
            self.q_norm.weight.to(query.device),
            self.k_norm.weight.to(key.device),
            epsilon=self.q_norm.eps,
            rot_dim=rope_table.shape[-3] * 2,
        )
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

    def load(self, handle, prefix: str) -> None:
        self.linear.load(handle, f"{prefix}.linear")

    def forward(self, time_embedding):
        projected = self.linear(time_embedding)
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
        self._segment_boundaries = None

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

    def begin_request(self) -> None:
        self._text_cache = None
        self._rope_cache = None
        self._segment_boundaries = None

    def end_request(self) -> None:
        self.begin_request()

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
            from diffusers.models.attention_dispatch import dispatch_attention_fn

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
        return torch.lerp(table[lower], table[lower + 1], (position - lower).unsqueeze(1))

    def _segments(self, indices: torch.Tensor):
        if self._segment_boundaries is None:
            host = indices.detach().cpu()
            changes = (host[1:] != host[:-1]).nonzero().flatten().add(1).tolist()
            self._segment_boundaries = [0, *changes, len(host)]
        bounds = self._segment_boundaries
        return [(a, b, indices[a]) for a, b in zip(bounds[:-1], bounds[1:])]

    @staticmethod
    def _modulate(hidden, shift, scale, segments):
        for start, stop, row in segments:
            hidden[start:stop].mul_(1.0 + scale[row].to(hidden.dtype)).add_(shift[row].to(hidden.dtype))
        return hidden

    @staticmethod
    def _gate(hidden, update, gate, segments):
        for start, stop, row in segments:
            hidden[start:stop].addcmul_(update[start:stop], gate[row].to(hidden.dtype))
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

        text = self._refine_text(encoder_hidden_states[0].to(torch.bfloat16))
        video = self.video_patch_proj(hidden_states[0].float()).to(text.dtype)
        audio = self.audio_patch_proj(audio_hidden_states[0].float()).to(text.dtype)
        packed = text.new_zeros((position_ids.shape[0], HIDDEN))
        packed.index_copy_(0, text_indices, text)
        packed.index_copy_(0, video_indices, video)
        packed.index_copy_(0, audio_indices, audio)

        time_embedding = self._time_embedding(timestep)
        adaln_indices = timestep_indices * 3 + token_tags.clamp(min=0)
        segments = self._segments(adaln_indices)
        rope = self._rope(position_ids, packed.dtype)

        for block in self.blocks:
            shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(time_embedding)
            normalized = self._modulate(block.norm1(packed), shift_attn, scale_attn, segments)
            packed = self._gate(
                packed,
                block.attn(normalized, rope, self.attention_backend),
                gate_attn,
                segments,
            )
            normalized = self._modulate(block.norm2(packed), shift_mlp, scale_mlp, segments)
            packed = self._gate(packed, block.mlp(normalized), gate_mlp, segments)

        normalized = self.final_layer.norm(packed)
        shift, scale = self.final_layer.adaln_proj(time_embedding)

        video_times = timestep_indices.index_select(0, video_indices)
        video_hidden = normalized.index_select(0, video_indices)
        video_hidden = video_hidden * (1.0 + scale.index_select(0, video_times)) + shift.index_select(0, video_times)
        video_output = self.final_layer.video_out(video_hidden.float()).unsqueeze(0)

        audio_times = timestep_indices.index_select(0, audio_indices)
        audio_hidden = normalized.index_select(0, audio_indices)
        audio_hidden = audio_hidden * (1.0 + scale.index_select(0, audio_times)) + shift.index_select(0, audio_times)
        audio_output = self.final_layer.audio_out(audio_hidden.float()).unsqueeze(0)

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
    return f"NVFP4 · pruned AdaLN curve · fused QKV/QK-norm/RoPE · `{NVFP4_REPO}`"
