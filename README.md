---
title: MiniMax-H3 Ultra Fast
emoji: ⚡
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: true
short_description: Ultra-fast local NVFP4 video + synchronized audio generation
suggested_hardware: zero-a10g
---

# MiniMax-H3 Ultra Fast — local conditioner + pruned NVFP4 on Blackwell

Joint video and synchronized sound from MiniMax-H3, with the repeatedly executed transformer rebuilt around the
Blackwell-native ComfyUI optimization path:

- 12.5 GB pruned NVFP4 transformer instead of the 61.7 GiB BF16 inference transformer.
- 20.1B effective inference parameters instead of 33.1B; the redundant 13.04B AdaLN projection weights become a
  compact sampled timestep curve.
- One fused QKV projection per attention layer.
- One fused in-place Q/K RMSNorm + partial split-half RoPE kernel.
- Native CUDA 13 NVFP4 tensor-core GEMMs through `comfy-kitchen`.
- A local 15.7 GB Qwen3-VL NVFP4-AWQ conditioner replaces the normal cross-Space API call. It contains exactly the
  first 50 language layers H3 reads, with the unused 14-layer tail and vocabulary head removed.
- The default 28-step Balanced schedule uses NVIDIA Sol-Engine's H3 FirstBlockCache: block 1 is evaluated as a
  high-signal change probe before deciding whether blocks 2–50 can reuse their previous joint residual. An optional
  Ultra Fast mode forecasts between exact anchors, while Exact evaluates every requested step. This is not
  prompt-to-video output caching.
- NVIDIA Sol-Attn's native SM120 kernel sparsifies target-video attention after a dense warmup while retaining text,
  conditioning-video and generated-audio rows as an exact KV sink with dense prefix queries.
- Segment-wise in-place AdaLN modulation and gated residual accumulation.
- Each block converts its complete AdaLN table in one launch instead of six, and the final RMSNorm runs only on
  generated video/audio rows whose outputs are retained.
- Optional exact Triton AdaLN modulation/gating kernels can replace hundreds of tiny per-segment launches on
  persistent workers; they remain off on ZeroGPU because their cold compilation cost is too high.
- Video and audio output heads run only on their own rows, not the full packed sequence.
- Prompt refinement and the rotary table are cached for the request instead of recomputed at every denoising step.
- Static keyframe patch projections are computed once, packed buffers avoid a redundant full zero-fill, and cached
  segment metadata uses CPU scalar row ids rather than repeated CUDA-scalar indexing.
- The video VAE and audio VAE remain full precision.

The source model is [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3). The pruned NVFP4
checkpoint is
[`lilcheaty/MiniMax-H3-NVFP4`](https://huggingface.co/lilcheaty/MiniMax-H3-NVFP4), derived from ComfyUI's
[`MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) repackage.

## Why the pruned transformer matters

MiniMax-H3's published model card notes that about 13B parameters live in AdaLN-related branches and that their
outputs can be precomputed for inference. The pruned checkpoint makes that concrete: it samples the shared timestep
embedding curve at 1025 points and linearly interpolates an 8-value coordinate for each requested timestep. Every
block's modulation projection consequently becomes `[96768, 8]` instead of `[96768, 2688]`.

| parameter group | original BF16 architecture | pruned architecture |
|---|---:|---:|
| AdaLN projections | 13.04B | 0.04B |
| MLP | 12.02B | 11.56B |
| attention | 8.02B | 7.71B |
| refiner, norms and embeddings | 0.05B | 0.80B |
| **total** | **33.12B** | **20.11B** |

The four large matrices in each of the 50 blocks—fused QKV, attention output, MLP up/gate and MLP down—are NVFP4.
The modulation curve, norms, embeddings, biases and final heads stay at higher precision.

## Why this is faster than the old 4-bit Space

Quantization alone does not guarantee speed. The older 4-bit comparison paid for CPU offload traffic on every layer
because its runtime did not keep the transformer resident. This engine is about 12 GB, so the transformer, both VAEs,
working activations and decoder workspace fit together on the 95 GiB `xlarge` ZeroGPU worker. There is no layerwise
host-device weight traffic in the denoising loop.

ComfyUI reports about a 2× NVFP4 uplift over FP8/BF16 on Blackwell in supported workloads. The H3 checkpoint author
measured 1.90 s/iteration for pruned NVFP4 versus 2.17 s/iteration for pruned INT8 ConvRot on an RTX PRO 6000
Blackwell at 864×480, 39 frames. Those numbers are useful implementation evidence, not a promise for every canvas:
H3 attention grows quadratically with packed sequence length, so resolution, duration and keyframe vision tokens
still dominate large requests.

## One-Space deployment

The original BF16 deployment had to be split because its 62.14 GiB conditioner plus 61.7 GiB transformer could not
fit comfortably under one Space's storage and runtime limits. The pruned formats change that calculation:

| component | where it runs | precision / format |
|---|---|---|
| Qwen3-VL layer-50 conditioner | this Space | truncated NVFP4-AWQ weights / BF16 GEMMs + BF16 vision tower |
| H3 transformer | this Space | pruned NVFP4 + higher-precision islands |
| video VAE | this Space | full precision checkpoint policy |
| audio VAE | this Space | FP32 |

The embeddings stay on the same GPU worker and flow directly into H3: there is no Gradio round trip, second queue,
safetensors serialization, or user-quota handoff during normal generation. Prompt upsampling still uses the remote
BF16 conditioner because rewriting requires the language-model head and the 14 decoder layers deliberately removed
from the local inference-only checkpoint. The remote service also remains an automatic fallback if local loading
fails.

## Kernel path

`h3_nvfp4.py` adapts the public ComfyUI H3 implementation to diffusers' packed transformer signature. It deliberately
does not install or launch the ComfyUI application. The small adapter uses only `comfy-kitchen` for:

1. Dynamic NVFP4 activation quantization and native FP4 matrix multiplication.
2. Fused in-place Q/K RMSNorm and three-axis split-half rotary embedding.

Attention uses NVIDIA Sol-Attn's SM120 CuTe kernel on eligible target-video calls and keeps diffusers'
`_native_cudnn` backend for the first ten denoising steps, the first two blocks, Exact mode, short sequences and any
safe fallback. The MLP uses one fused QKV-style gate/up matrix, in-place SiLU×up, and the NVFP4 down projection.

Above that kernel path, Ultra Fast preserves the scheduler's 16-step trajectory but evaluates the full DiT only at
7 anchor steps. Between anchors it extrapolates the joint video/audio residual from the last two exact evaluations.
It keeps the first three and last two evaluations exact and never forecasts more than three consecutive steps.
Balanced uses the official H3 FirstBlockCache signal at `0.08`, with three dense warmup and two dense tail steps;
Exact disables all block reuse and sparse attention.
The result panel and Space log report actual DiT evaluations and forecasts for every request. Different prompts and
seeds never share this state.

The old BF16/AoTI engine remains available with `H3_ENGINE=bf16`. It is useful as a quality/debug reference, but it is
not the default.

## Quality trade-off

The transformer, local conditioner, and accelerated step modes are approximate. The checkpoint author reports that
4-bit weights can show more
mid-motion artifacts and weaker shape retention than the larger INT8 ConvRot checkpoint on difficult 15-second
clips. The comparison was not fully controlled, so treat it as a real caution rather than a quantified quality
score. Use Balanced for difficult fast motion, eyes, fingers, or shape retention, and Exact when every requested
transformer evaluation matters. Increasing the step slider does not change Ultra Fast's maximum three-step forecast
span; it adds more exact anchor evaluations as well as scheduler steps.

The Space keeps both VAEs full precision and leaves AdaLN, norms, embeddings and output heads out of NVFP4. For the
exact original denoiser, set `H3_ENGINE=bf16`; this restores the 61.7 GiB unquantized transformer and its AoTI option.

## Space variables

| variable | default | meaning |
|---|---|---|
| `H3_ENGINE` | `nvfp4` | `nvfp4` ultra engine or `bf16` reference engine. |
| `H3_NVFP4_REPO` | `lilcheaty/MiniMax-H3-NVFP4` | Repository containing the pruned Comfy-format transformer. |
| `H3_NVFP4_FILE` | `minimax_h3_fl2va_pruned_nvfp4.safetensors` | FL2VA/T2VA transformer file. |
| `H3_MODEL_REPO` | `MiniMaxAI/MiniMax-H3` | Canonical schedulers and VAE checkpoint. |
| `H3_CONDITIONER_MODE` | `local` | Use the local truncated conditioner; `remote` restores the split deployment. |
| `H3_LOCAL_CONDITIONER_REPO` | `Comfy-Org/MiniMax-H3` | Repository containing the truncated conditioner. |
| `H3_LOCAL_CONDITIONER_FILE` | `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | Local layer-50 conditioner checkpoint. |
| `H3_CONDITIONER_NATIVE_NVFP4` | `0` | Use native W4A4 conditioner GEMMs for maximum speed; default BF16 GEMMs add less activation error. |
| `H3_CONDITIONER` | `multimodalart/qwen3vl-conditioner` | Prompt-upsampling and local-load fallback service. |
| `H3_PLACEMENT` | `lazy` (`nvfp4`) | Move the compact transformer and VAEs on the first GPU call, then keep them resident. |
| `H3_ATTENTION` | `_native_cudnn` | Attention backend for both the main stack and text refiner. |
| `H3_EASYCACHE_THRESHOLD` | `0.10` | Balanced-mode maximum accumulated estimated change before a full denoiser evaluation. |
| `H3_EASYCACHE_START` | `0.15` | Fraction of the schedule before which every step is evaluated. |
| `H3_EASYCACHE_END` | `0.95` | Fraction of the schedule after which every step is evaluated. |
| `H3_EASYCACHE_SUBSAMPLE` | `8` | Generated-video row stride used by the inexpensive change estimator. |
| `H3_FIRST_BLOCK_THRESHOLD` | `0.08` | Balanced-mode normalized first-block residual threshold, matching NVIDIA's H3 preset. |
| `H3_FIRST_BLOCK_DENSE_START` | `3` | Dense warmup steps retained before FirstBlockCache may reuse blocks 2–50. |
| `H3_FIRST_BLOCK_DENSE_END` | `2` | Dense tail steps retained after FirstBlockCache reuse is disabled. |
| `H3_FORECAST_BLEND` | `0.65` | Ultra Fast linear-trend strength; lower values stay closer to last-residual reuse. |
| `H3_FUSED_ADALN` | `0` | Opt in to fused AdaLN kernels; compilation is expensive on fresh ZeroGPU workers. |
| `H3_SOL_ATTN` | `1` | Use NVIDIA Sol-Attn outside Exact mode, with automatic dense fallback. |
| `H3_SOL_ATTN_TAU` | `1.0` | Official H3 sparse-routing threshold. |
| `H3_SOL_ATTN_DENSE_STEPS` | `10` | Initial denoising steps kept on exact dense attention. |
| `H3_SOL_ATTN_DENSE_LAYERS` | `2` | Initial transformer blocks kept on exact dense attention. |
| `H3_SOL_ATTN_MIN_TOKENS` | `8192` | Skip sparse-kernel overhead on shorter packed sequences. |
| `H3_GPU_SIZE` | `xlarge` | 95 GiB Blackwell ZeroGPU allocation. |
| `H3_AOTI` | `0` | BF16 engine only: load the optional repeated-block AoTI package. |

## Runtime requirements

- PyTorch 2.11 with CUDA 13.0.
- A Blackwell GPU (`sm120` for this Space). NVFP4 on older architectures is emulated and can be slower than BF16.
- `comfy-kitchen==0.2.26` for the native layouts and fused Q/K kernel.
- The pinned MiniMax-H3 diffusers pull request for the modular schedulers, packing and VAE decode path.

No secret is required. All model artifacts are public. Normal requests use one local GPU booking; a prompt-upsampling
request forwards the requesting user's ZeroGPU identity to the remote rewriter/conditioner.

## Attribution

This is an optimized derivative of the original
[`multimodalart/minimax-h3`](https://huggingface.co/spaces/multimodalart/minimax-h3) Space.

The fused/pruned model structure follows
[`comfy/ldm/minimax/model.py`](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/model.py) from
ComfyUI (Apache-2.0). FirstBlockCache and Sol-Attn are adapted from NVIDIA's Apache-2.0
[`MiniMax-H3 Sol-Engine`](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3) release. The quantized checkpoint and its conversion notes are from
[`lilcheaty/MiniMax-H3-NVFP4`](https://huggingface.co/lilcheaty/MiniMax-H3-NVFP4). MiniMax-H3 weights remain governed
by the MiniMax-H3 Community License Agreement.
