"""Local, truncated Qwen3-VL conditioner for MiniMax-H3.

The canonical diffusers checkpoint stores all 64 language layers plus the LM head in BF16 (66.7 GB), although H3
only reads the unnormalized state after layer 50.  ComfyUI's Apache-2.0 conversion removes the unused tail and head,
keeps the vision tower in BF16, and stores the 50 language layers as NVFP4-AWQ.  This adapter loads that single
15.7 GB file directly into Transformers' Qwen3-VL architecture and exposes the tiny contract used by diffusers.

No ComfyUI application or server is launched. Preprocessing remains Transformers' canonical Qwen3-VL processor.
By default the checkpoint's quality-oriented weight-only policy is honored: compact NVFP4-AWQ weights are
dequantized one layer at a time for BF16 GEMMs. Native W4A4 is available as an aggressive opt-in.
"""

from __future__ import annotations

import copy
import os
from types import SimpleNamespace

import torch
import torch.nn as nn

from h3_nvfp4 import H3Linear


CONDITIONER_REPO = os.environ.get("H3_LOCAL_CONDITIONER_REPO", "Comfy-Org/MiniMax-H3")
CONDITIONER_FILE = os.environ.get(
    "H3_LOCAL_CONDITIONER_FILE",
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
)
SOURCE_REPO = os.environ.get("H3_MODEL_REPO", "MiniMaxAI/MiniMax-H3")
LAYERS = 50
NATIVE_NVFP4 = os.environ.get("H3_CONDITIONER_NATIVE_NVFP4", "0") == "1"


class QuantizedEmbedding(nn.Module):
    """Row-wise INT8 token lookup without dequantizing the 1.56 GB BF16 vocabulary table."""

    def __init__(self, handle, prefix: str):
        super().__init__()
        self.register_buffer("weight", handle.get_tensor(f"{prefix}.weight"))
        self.register_buffer("scale", handle.get_tensor(f"{prefix}.weight_scale").float())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        flat = input_ids.reshape(-1)
        values = self.weight.index_select(0, flat).reshape(*input_ids.shape, self.weight.shape[1])
        scales = self.scale.index_select(0, flat).reshape(*input_ids.shape, 1)
        return values.to(torch.bfloat16).mul_(scales.to(torch.bfloat16))


class Layer50Backbone(nn.Module):
    """Avoid retaining 50 intermediate tensors merely to satisfy diffusers' hidden-state indexing API."""

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, *args, **kwargs):
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("return_dict", None)
        kwargs["use_cache"] = False
        output = self.core(*args, **kwargs)
        # get_qwen3vl_prompt_embeds asks for hidden_states[50]. The first 50 entries need not be materialized.
        return SimpleNamespace(hidden_states=(None,) * LAYERS + (output.last_hidden_state,))


class LocalH3Conditioner(nn.Module):
    """The subset of Qwen3VLForConditionalGeneration that MiniMax-H3 actually calls."""

    def __init__(self, core: nn.Module, source_config):
        super().__init__()
        public_config = copy.deepcopy(source_config)
        # Diffusers rejects a nominally 50-layer model because a normal last_hidden_state is post-norm. This adapter
        # removes the final norm and returns the raw 50th-layer state, so advertise index 50 as available explicitly.
        public_config.text_config.num_hidden_layers = LAYERS + 1
        self.config = public_config
        self.model = Layer50Backbone(core)

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16

    @property
    def device(self) -> torch.device:
        return self.model.core.visual.patch_embed.proj.weight.device


def _target_name(checkpoint_name: str) -> str:
    if checkpoint_name.startswith("model.layers."):
        return "language_model.layers." + checkpoint_name.removeprefix("model.layers.")
    if checkpoint_name.startswith("visual."):
        return checkpoint_name
    raise KeyError(checkpoint_name)


def _build_core(handle):
    from accelerate import init_empty_weights
    from transformers import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    config = Qwen3VLConfig.from_pretrained(SOURCE_REPO, subfolder="text_encoder")
    config.text_config.num_hidden_layers = LAYERS
    config.text_config.use_cache = False
    config.text_config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation = "sdpa"

    with init_empty_weights(include_buffers=False):
        core = Qwen3VLModel(config)

    keys = set(handle.keys())
    embedding_prefix = "model.embed_tokens"
    core.language_model.embed_tokens = QuantizedEmbedding(handle, embedding_prefix)
    consumed = {
        key for key in keys if key == f"{embedding_prefix}.comfy_quant" or key.startswith(f"{embedding_prefix}.weight")
    }

    quantized_prefixes = sorted(
        key.removesuffix(".comfy_quant")
        for key in keys
        if key.startswith("model.layers.") and key.endswith(".comfy_quant")
    )
    if len(quantized_prefixes) != LAYERS * 7:
        raise RuntimeError(f"Expected {LAYERS * 7} quantized language linears, found {len(quantized_prefixes)}.")

    for source_prefix in quantized_prefixes:
        target_prefix = _target_name(source_prefix)
        parent_name, child_name = target_prefix.rsplit(".", 1)
        parent = core.get_submodule(parent_name)
        original = getattr(parent, child_name)
        linear = H3Linear(original.in_features, original.out_features, bias=original.bias is not None)
        linear.load(handle, source_prefix)
        if NATIVE_NVFP4:
            linear.full_precision_mm = False
        setattr(parent, child_name, linear)
        consumed.update(key for key in keys if key.startswith(f"{source_prefix}."))

    # MiniMax-H3 consumes the raw output of layer 49. The released Comfy checkpoint intentionally has no final norm.
    core.language_model.norm = nn.Identity()

    plain_state = {}
    for source_name in sorted(keys - consumed):
        if source_name.startswith("visual.") or source_name.startswith("model.layers."):
            plain_state[_target_name(source_name)] = handle.get_tensor(source_name)
            consumed.add(source_name)

    unknown = keys - consumed
    if unknown:
        raise RuntimeError(f"Unhandled local-conditioner tensors: {sorted(unknown)[:12]}")

    core.load_state_dict(plain_state, strict=False, assign=True)
    meta = [name for name, value in core.named_parameters() if value.is_meta]
    if meta:
        raise RuntimeError(f"Local conditioner still has uninitialized parameters: {meta[:12]}")
    core.eval()
    return core, config


def load_local_conditioner():
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import Qwen3VLProcessor

    path = hf_hub_download(CONDITIONER_REPO, CONDITIONER_FILE)
    with safe_open(path, framework="pt", device="cpu") as handle:
        core, config = _build_core(handle)

    processor = Qwen3VLProcessor.from_pretrained(SOURCE_REPO, subfolder="text_encoder")
    model = LocalH3Conditioner(core, config).eval()
    print(f"[h3-cond] loaded local layer-50 conditioner {CONDITIONER_REPO}/{CONDITIONER_FILE}", flush=True)
    return model, processor.tokenizer, processor


def status() -> str:
    compute = "native W4A4" if NATIVE_NVFP4 else "BF16 GEMM"
    return f"local layer-50 Qwen3-VL NVFP4-AWQ weights / {compute} · `{CONDITIONER_REPO}`"
