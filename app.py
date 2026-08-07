"""MiniMax-H3 Ultra Fast: local layer-50 conditioning plus `t2va` / `fl2va` generation."""

from __future__ import annotations

import hashlib
import ipaddress
import os
import re
import shutil
import socket
import tempfile
import time
import traceback
from collections import OrderedDict
from functools import cache
from urllib.parse import urljoin, urlsplit

# Before anything that could initialize CUDA: `import spaces` patches `torch.cuda` so model loading can happen at
# startup rather than on GPU time.
import spaces
import gradio as gr
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from gradio import Request, Server
from gradio.data_classes import FileData

MODEL_REPO = os.environ.get("H3_MODEL_REPO", "MiniMaxAI/MiniMax-H3")
CONDITIONER_SPACE = os.environ.get("H3_CONDITIONER", "multimodalart/qwen3vl-conditioner")
CONDITIONER_MODE = os.environ.get("H3_CONDITIONER_MODE", "local").lower()
# `nvfp4` is the Blackwell-native ultra path; `bf16` preserves the original 33B diffusers transformer as a fallback.
ENGINE = os.environ.get("H3_ENGINE", "nvfp4").lower()
# `pack` places the transformer at startup, `lazy` moves everything on the first GPU call, `offload` hands placement to
# `ComponentsManager.enable_auto_cpu_offload`.
PLACEMENT = os.environ.get("H3_PLACEMENT", "lazy" if ENGINE == "nvfp4" else "pack").lower()
# cuDNN's fused attention is 10-20% faster than the SDPA default on this pool and needs nothing installed.
ATTENTION = os.environ.get("H3_ATTENTION", "_native_cudnn").lower()
GPU_SIZE = os.environ.get("H3_GPU_SIZE", "xlarge")
OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "h3-outputs")
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
TURBO_4_LORA_REPO = "lightx2v/Minimax-h3-Turbo"
TURBO_4_LORA_FILE = "minimax_h3_fl2v_turbo_4step_v0.1.safetensors"
# LightX2V's reference loader uses alpha=8 for this rank-128 PEFT adapter. The custom NVFP4 branch applies the final
# activation-space multiplier directly, so preserve PEFT's alpha/rank scale here instead of applying it 16x too hard.
TURBO_4_LORA_SCALE = 8 / 128
TURBO_8_LORA_REPO = "larryvrh/MiniMax-H3-Turbo-Lora"
TURBO_8_LORA_FILE = "minimax_h3_turbo_4step_ema_ckpt850.safetensors"
LORA_MAX_BYTES = 2 * 1024**3
EGRID_COMMIT = "a7624b4c00626a8ae7e78860769389d706565190"
EGRID_SHA256 = "30eb3c2cc7fb6b470d9717ff840d359313ac27cd64b705e32da1baa10f72d6a8"
EGRID_URL = (
    f"https://raw.githubusercontent.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/{EGRID_COMMIT}/"
    "h3_silu_temb_grid.safetensors"
)

GENERATION_PRESETS = {
    "Balanced — best overall (recommended)": (28, "Balanced", "None"),
    "Turbo 8-step — faster, cleaner": (8, "Exact", "Turbo · 8 steps"),
    "Turbo 4-step — fastest, more artifacts": (4, "Exact", "Turbo · 4 steps"),
    "Exact 28-step — maximum fidelity": (28, "Exact", "None"),
    "Ultra cache — experimental speed": (28, "Ultra Fast", "None"),
}
DEFAULT_PRESET = next(iter(GENERATION_PRESETS))
CUSTOM_PRESET = "Custom — manual controls"

# Must stay identical to the conditioner's table: the *label* goes over the wire, so a canvas that half does not know
# is rejected there and surfaces as a failure here.
CANVASES = {
    # 16:9
    "960x544 · 16:9 fast": (544, 960),
    "1024x576 · 16:9 fast": (576, 1024),
    "1152x640 · 16:9": (640, 1152),
    "1280x704 · 16:9": (704, 1280),
    "1344x768 · 16:9 full": (768, 1344),
    # 9:16
    "544x960 · 9:16 fast": (960, 544),
    "640x1152 · 9:16": (1152, 640),
    "768x1344 · 9:16 full": (1344, 768),
    # 1:1
    "544x544 · 1:1 fast": (544, 544),
    "768x768 · 1:1 full": (768, 768),
    # 4:3 / 3:4
    "768x576 · 4:3 fast": (576, 768),
    "1024x768 · 4:3 full": (768, 1024),
    "576x768 · 3:4 fast": (768, 576),
    "768x1024 · 3:4 full": (1024, 768),
    # 21:9
    "1152x512 · 21:9 fast": (512, 1152),
    "1536x672 · 21:9 full": (672, 1536),
}
DEFAULT_CANVAS = "960x544 · 16:9 fast"
FPS, FRAMES_PER_CHUNK, LATENTS_PER_CHUNK = 24, 17, 5
# It is the *snapped* frame count the ceiling has to hold for: 15 s is 360 frames, which rounds up to 362, i.e.
# 15.083 s, and is refused.
MIN_UI_DURATION, MAX_UI_DURATION = 2, 14


def resolve_canvas(value: str) -> str:
    """Return a canonical canvas, including for cached clients that submit an old WIDTHxHEIGHT label."""
    canvas = str(value).strip()
    if canvas in CANVASES:
        return canvas

    match = re.match(r"^(\d{3,4})\s*[x×]\s*(\d{3,4})(?:\s*·.*)?$", canvas)
    if match:
        requested_width, requested_height = (int(part) for part in match.groups())
        if 256 <= requested_width <= 2048 and 256 <= requested_height <= 2048:
            requested_ratio = requested_width / requested_height
            requested_area = requested_width * requested_height

            def distance(item):
                _, (height, width) = item
                ratio_error = abs(width / height - requested_ratio) / requested_ratio
                area_error = abs(width * height - requested_area) / requested_area
                return ratio_error * 4 + area_error

            nearest, (height, width) = min(CANVASES.items(), key=distance)
            ratio_error = abs(width / height - requested_ratio) / requested_ratio
            if ratio_error <= 0.12:
                print(f"[canvas] translated legacy {canvas!r} to {nearest!r}", flush=True)
                return nearest

    raise ValueError(
        f"Unknown canvas {value!r}. Refresh the Space and choose a supported format: {', '.join(CANVASES)}"
    )


def snap_frames(seconds: float) -> int:
    """The frame count MiniMax-H3's video VAE can decode: the next `17 * n + 5` at 24 fps."""
    frames = max(1, round(float(seconds) * FPS))
    while frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        frames += 1
    return frames


def lower_duration_floor(seconds: float = MIN_UI_DURATION) -> None:
    """Let the pipeline generate below its 5 s floor. 56 frames (2.33 s) is fine on the released checkpoint."""
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline

    MiniMaxH3ModularPipeline.min_duration = property(lambda self: float(seconds))


PIPE = None
MANAGER = None
COND_PIPE = None
COND_ERROR: str | None = None
LOAD_ERROR: str | None = None
LOADED_IN: float | None = None
# A prompt + its two anchors typically occupies only a few MiB after conditioning. Keeping the newest four on CPU
# makes Draft -> Final skip the 50-layer Qwen pass whenever ZeroGPU retains this warm worker between requests.
CONDITION_CACHE = OrderedDict()
CONDITION_CACHE_SIZE = 4


def status() -> str:
    if LOAD_ERROR:
        return LOAD_ERROR
    if PIPE is None:
        payload = (
            "pruned NVFP4 transformer + local NVFP4 conditioner + full-precision VAEs (~44 GB)"
            if ENGINE == "nvfp4"
            else "BF16 transformer + VAEs (77.3 GB)"
        )
        return f"Loading {payload}. Watch the Space logs."
    if ENGINE == "nvfp4":
        import h3_nvfp4

        engine_status = h3_nvfp4.status()
    else:
        import h3_aoti

        engine_status = f"BF16, unquantized · {h3_aoti.status()}"
    if COND_PIPE is not None:
        import h3_local_conditioner

        conditioner_status = h3_local_conditioner.status()
    else:
        conditioner_status = f"remote `{CONDITIONER_SPACE}`" + (" (local fallback)" if COND_ERROR else "")
    return (
        f"Ready · **{engine_status}** · VAEs full precision · placement `{PLACEMENT}` · attention `{ATTENTION}` · "
        f"loaded in {LOADED_IN:.0f}s · conditioner {conditioner_status}"
    )


def load_models() -> str | None:
    """Load the compact generator and, by default, its local truncated conditioner at startup.

    `MiniMaxH3GeneratorBlocks` declares `transformer`, `vae`, `audio_vae`, the two schedulers and `video_processor`,
    so `load_components` fetches exactly those subfolders — `text_encoder/` and `transformer_ref/` are never touched.
    Both autoencoders carry `_keep_in_fp32_modules` over every module and stay float32: a bfloat16 audio VAE decodes
    the soundtrack roughly 20 dB too quiet.
    """
    global PIPE, MANAGER, COND_PIPE, COND_ERROR, LOAD_ERROR, LOADED_IN

    if PIPE is not None or LOAD_ERROR is not None:
        return LOAD_ERROR

    started = time.time()
    try:
        import torch
        from diffusers import ComponentsManager

        from h3_split_blocks import MiniMaxH3GeneratorBlocks

        lower_duration_floor()
        manager = ComponentsManager()
        blocks = MiniMaxH3GeneratorBlocks()
        print(f"[gen] loading {[c.name for c in blocks.expected_components]} from {MODEL_REPO} ...", flush=True)
        pipe = blocks.init_pipeline(MODEL_REPO, components_manager=manager, collection="h3")
        # This is consumed by the endpoint's Gradio Progress tracker and forwarded across the ZeroGPU worker RPC,
        # producing one real queue update per denoising step.
        pipe.set_progress_bar_config(desc="Denoising")
        if ENGINE == "nvfp4":
            # Do not download the 61.7 GiB BF16 transformer. The schedulers and full-precision VAEs stay canonical;
            # only the repeatedly executed DiT is replaced with the pruned Blackwell-native checkpoint.
            pipe.load_components(
                names=["vae", "audio_vae", "scheduler", "audio_scheduler", "video_processor"],
                dtype=torch.bfloat16,
            )
            from h3_nvfp4 import load_transformer

            pipe.update_components(transformer=load_transformer())
        elif ENGINE == "bf16":
            pipe.load_components(dtype=torch.bfloat16)
        else:
            raise ValueError(f"H3_ENGINE must be `nvfp4` or `bf16`, got {ENGINE!r}")
        pipe.transformer.set_attention_backend(ATTENTION)

        # Still startup, still free: an AoTI package carries no weights and opens its archive lazily inside the GPU
        # worker. Off unless `H3_AOTI=1`.
        if ENGINE == "bf16":
            import h3_aoti

            h3_aoti.maybe_load(pipe.transformer)

        if PLACEMENT == "pack":
            # Scoped to the transformer. `spaces` packs every startup-resident CUDA tensor into a second on-disk copy,
            # and packing all 77.3 GB busts the 150 GB storage quota; the 61.7 GB transformer alone fits. The ~10 GB of
            # fp32 VAEs move on the first GPU call instead.
            pipe.transformer.to("cuda")

        if PLACEMENT == "offload":
            manager.enable_auto_cpu_offload(device="cuda")
            _arm_decode_hooks(pipe)

        cond_pipe = None
        if CONDITIONER_MODE == "local":
            try:
                from h3_local_conditioner import load_local_conditioner
                from h3_split_blocks import MiniMaxH3ConditionerBlocks

                print("[cond] loading the local truncated NVFP4-AWQ conditioner ...", flush=True)
                text_encoder, tokenizer, processor = load_local_conditioner()
                cond_pipe = MiniMaxH3ConditionerBlocks().init_pipeline(MODEL_REPO)
                cond_pipe.update_components(
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    processor=processor,
                )
            except Exception as error:
                traceback.print_exc()
                COND_ERROR = f"{type(error).__name__}: {error}"
                print(f"[cond] local load failed ({COND_ERROR}); retaining the remote fallback", flush=True)
        elif CONDITIONER_MODE != "remote":
            raise ValueError(f"H3_CONDITIONER_MODE must be `local` or `remote`, got {CONDITIONER_MODE!r}")

        PIPE, MANAGER, COND_PIPE = pipe, manager, cond_pipe
        LOADED_IN = time.time() - started
        print(f"[gen] ready in {LOADED_IN:.0f}s", flush=True)
    except Exception as error:
        traceback.print_exc()
        LOAD_ERROR = f"**Loading `{MODEL_REPO}` failed** after {time.time() - started:.0f}s: `{type(error).__name__}: {error}`"
    return LOAD_ERROR


def _arm_decode_hooks(pipe):
    """Make the offload hooks fire for the two VAEs.

    `enable_auto_cpu_offload` wraps `forward`, and the decode blocks call `vae.decode(...)` directly, so the hook
    never runs and the VAE is still on the host when the latents arrive on the card.
    """
    for name in ("vae", "audio_vae"):
        module = getattr(pipe, name)
        inner = module.decode

        def armed(*args, _module=module, _decode=inner, **kwargs):
            hook = getattr(_module, "_hf_hook", None)
            if hook is not None:
                hook.pre_forward(_module)
            return _decode(*args, **kwargs)

        module.decode = armed


@cache
def conditioner():
    """The other half, over the gradio API. `gradio_client` attaches the caller's own ZeroGPU token per call, so the
    conditioner's booking is billed to whoever asked for the video."""
    from gradio_client import Client

    return Client(CONDITIONER_SPACE)


def conditioner_client(ip_token: str | None):
    """Create a caller-attributed client in Server mode; fall back to the cached anonymous client when unavailable."""
    if not ip_token:
        return conditioner()
    from gradio_client import Client

    return Client(CONDITIONER_SPACE, headers={"x-ip-token": ip_token})


def encode_remote(prompt, image_path, last_image_path, canvas, num_frames, rewrite_prompt=False, ip_token=None):
    """`/encode` on the conditioner Space: a safetensors file holding `prompt_embeds` + `text_token_tags`, with the
    resolved `height` / `width` / `num_frames` in its metadata, plus the plan. `canvas` is the label."""
    from gradio_client import handle_file
    from safetensors import safe_open

    path, plan = conditioner_client(ip_token).predict(
        prompt=prompt,
        image_path=handle_file(image_path) if image_path else None,
        last_image_path=handle_file(last_image_path) if last_image_path else None,
        canvas=canvas,
        num_frames=num_frames,
        rewrite_prompt=bool(rewrite_prompt),
        api_name="/encode",
    )
    with safe_open(path, framework="pt") as handle:
        metadata = handle.metadata()
        return handle.get_tensor("prompt_embeds"), handle.get_tensor("text_token_tags"), metadata, plan


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_egrid() -> str:
    """Fetch the Turbo author's pruned-base timestep grid once, pinned and checksum-verified."""
    import requests

    directory = os.path.join(tempfile.gettempdir(), "h3-lora-assets")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"h3-silu-temb-{EGRID_SHA256[:12]}.safetensors")
    if os.path.isfile(path):
        if _sha256(path) == EGRID_SHA256:
            return path
    temporary = path + ".download"
    with requests.get(EGRID_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(temporary, "wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                output.write(chunk)
    digest = _sha256(temporary)
    if digest != EGRID_SHA256:
        os.unlink(temporary)
        raise ValueError("The pinned H3 timestep grid failed its SHA-256 check.")
    os.replace(temporary, path)
    return path


def _lora_cache_directory(source: str) -> str:
    """Return one bounded cache directory and evict all but the newest other adapter."""
    root = os.path.join(tempfile.gettempdir(), "h3-lora-downloads")
    cache_key = hashlib.sha256(source.encode()).hexdigest()[:20]
    local_dir = os.path.join(root, cache_key)
    os.makedirs(local_dir, exist_ok=True)
    others = sorted(
        (entry for entry in os.scandir(root) if entry.is_dir() and entry.path != local_dir),
        key=lambda entry: entry.stat().st_mtime,
        reverse=True,
    )
    for stale in others[1:]:
        shutil.rmtree(stale.path, ignore_errors=True)
    return local_dir


def _validate_public_lora_url(url: str) -> str:
    """Allow public HTTPS downloads while refusing credentials, unusual ports and private-network destinations."""
    if len(url) > 2048:
        raise ValueError("Direct LoRA URL is too long.")
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError("Direct LoRA URLs must use public HTTPS.")
    if parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ValueError("Direct LoRA URLs cannot contain credentials or a non-standard port.")
    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
        }
    except socket.gaierror as error:
        raise ValueError("Direct LoRA URL hostname could not be resolved.") from error
    if not addresses:
        raise ValueError("Direct LoRA URL hostname did not resolve.")
    for raw_address in addresses:
        address = ipaddress.ip_address(raw_address)
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
            address = address.ipv4_mapped
        if not address.is_global:
            raise ValueError("Direct LoRA URLs cannot access private or local networks.")
    return url


def _download_lora_url(url: str) -> tuple[str, str]:
    """Stream one public safetensors URL into the bounded adapter cache, validating every redirect and byte."""
    import requests
    from safetensors import safe_open

    original = _validate_public_lora_url(url)
    local_dir = _lora_cache_directory(original)
    path = os.path.join(local_dir, "adapter.safetensors")
    if os.path.isfile(path) and 0 < os.path.getsize(path) <= LORA_MAX_BYTES:
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                if handle.keys():
                    os.utime(local_dir, None)
                    parsed = urlsplit(original)
                    return path, f"{parsed.hostname}{parsed.path}"[:240]
        except Exception:
            os.unlink(path)

    temporary = path + ".download"
    current = original
    try:
        for _ in range(6):
            current = _validate_public_lora_url(current)
            with requests.get(
                current,
                stream=True,
                allow_redirects=False,
                timeout=(10, 120),
                headers={"User-Agent": "MiniMax-H3-Space/1.0"},
            ) as response:
                if response.is_redirect or response.is_permanent_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("Direct LoRA URL returned an empty redirect.")
                    current = urljoin(current, location)
                    continue
                response.raise_for_status()
                declared = response.headers.get("content-length")
                if declared is not None and int(declared) > LORA_MAX_BYTES:
                    raise ValueError("Direct LoRA exceeds the 2 GiB safety limit.")
                total = 0
                with open(temporary, "wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > LORA_MAX_BYTES:
                            raise ValueError("Direct LoRA exceeds the 2 GiB safety limit.")
                        output.write(chunk)
                if total == 0:
                    raise ValueError("Direct LoRA URL returned an empty file.")
                with safe_open(temporary, framework="pt", device="cpu") as handle:
                    if not handle.keys():
                        raise ValueError("Direct LoRA contains no safetensors tensors.")
                os.replace(temporary, path)
                os.utime(local_dir, None)
                parsed = urlsplit(original)
                return path, f"{parsed.hostname}{parsed.path}"[:240]
        raise ValueError("Direct LoRA URL redirected too many times.")
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def resolve_lora(preset: str, repo_id: str, filename: str) -> tuple[str | None, str | None, str, float]:
    """Resolve one public safetensors LoRA before ZeroGPU is booked; arbitrary code is never downloaded or run."""
    preset = str(preset or "None")
    if preset == "None":
        return None, None, "None", 1.0
    adapter_scale = 1.0
    if preset == "Turbo · 4 steps":
        repo_id, filename = TURBO_4_LORA_REPO, TURBO_4_LORA_FILE
        adapter_scale = TURBO_4_LORA_SCALE
    elif preset == "Turbo · 8 steps":
        repo_id, filename = TURBO_8_LORA_REPO, TURBO_8_LORA_FILE
    else:
        repo_id, filename = str(repo_id or "").strip(), str(filename or "").strip()
        if repo_id.startswith("https://"):
            path, label = _download_lora_url(repo_id)
            return path, _download_egrid(), label, adapter_scale
        if "://" in repo_id:
            raise ValueError("Direct LoRA URLs must use public HTTPS.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id):
        raise ValueError("Custom LoRA source must be `owner/repository` or a public HTTPS URL.")
    if not filename.endswith(".safetensors") or filename.startswith(("/", ".")) or ".." in filename.split("/"):
        raise ValueError("Custom LoRA must be a safe `.safetensors` filename inside that repository.")

    from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url

    metadata = get_hf_file_metadata(hf_hub_url(repo_id, filename), token=False)
    if metadata.size is None or metadata.size > LORA_MAX_BYTES:
        raise ValueError("LoRA file is missing a declared size or exceeds the 2 GiB safety limit.")
    # Keep arbitrary public adapters in a bounded temp cache instead of letting user-selected repos fill Space disk.
    local_dir = _lora_cache_directory(f"hf://{repo_id}/{filename}")
    path = hf_hub_download(repo_id=repo_id, filename=filename, token=False, local_dir=local_dir)
    os.utime(local_dir, None)
    # Only the older pruned-base adapter contains AdaLN targets and needs its external timestep lookup grid. Custom
    # adapters keep the grid available for backwards compatibility; loaders that do not target AdaLN simply ignore it.
    egrid_path = None if preset == "Turbo · 4 steps" else _download_egrid()
    return path, egrid_path, f"{repo_id}/{filename}", adapter_scale


# Seconds of GPU one request needs, from the packed video rows it is about to denoise: linear in the rows for the
# matmuls, quadratic for the attention, against the AoTI block package this Space runs.
_DUR_B, _DUR_C = 1.1745e-4, 3.8396e-9
# The two resident decoders and the mux, which scale with the output rather than with the step count.
_DECODE_BASE, _DECODE_PER_DEFAULT_CANVAS, _DEFAULT_CANVAS_PIXELS = 15, 15, 960 * 544 * 124
# `pack` mode: only the ~10 GB of VAEs move on a cold worker.
_PLACEMENT_ALLOWANCE, _PAD = 12, 10
_RETRYABLE_GPU_ERRORS = ("uncorrectable ECC error", "cudaErrorECCUncorrectable")


def get_duration(prompt, prompt_embeds, text_token_tags, image, last_image, height, width, num_frames, steps, seed, *a, **k):
    height, width, num_frames, steps = int(height), int(width), int(num_frames), int(steps)
    latent_frames = (num_frames - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK * LATENTS_PER_CHUNK + 2
    patches = (height // 32) * (width // 32)
    rows = latent_frames * patches + (int(image is not None) + int(last_image is not None)) * patches
    denoise = steps * (_DUR_B * rows + _DUR_C * rows**2)
    decode = _DECODE_BASE + _DECODE_PER_DEFAULT_CANVAS * (height * width * num_frames) / _DEFAULT_CANVAS_PIXELS
    local_conditioning = 20 if prompt_embeds is None else 0
    return max(60, int(denoise + decode) + local_conditioning + _PLACEMENT_ALLOWANCE + _PAD)


@spaces.GPU(duration=get_duration, size=GPU_SIZE)
def _generate(
    prompt, prompt_embeds, text_token_tags, image, last_image, height, width, num_frames, steps, seed, acceleration,
    lora_path, egrid_path, lora_strength, conditioning_cache_key,
):
    """The only thing on GPU time: local conditioning, packed denoising and the two decoders.

    Only generated outputs and two timing scalars come back—a `@spaces.GPU` return crosses a process boundary by
    pickling, and the full `PipelineState` still holds packed latents, the rotary grid and row indices on the card.
    """
    import torch

    if COND_PIPE is not None and prompt_embeds is None:
        COND_PIPE.text_encoder.to("cuda")
    if PLACEMENT == "lazy":
        PIPE.to("cuda")
    elif PLACEMENT == "pack":
        PIPE.vae.to("cuda")
        PIPE.audio_vae.to("cuda")

    condition_seconds = None
    num_text_tokens = None
    condition_cache_hit = False
    if prompt_embeds is None:
        if COND_PIPE is None:
            raise RuntimeError(f"The local conditioner is unavailable: {COND_ERROR or 'disabled'}")
        cached = CONDITION_CACHE.pop(conditioning_cache_key, None) if conditioning_cache_key else None
        if cached is not None:
            prompt_embeds, text_token_tags = cached
            CONDITION_CACHE[conditioning_cache_key] = cached
            condition_seconds = 0.0
            num_text_tokens = int(prompt_embeds.shape[1])
            condition_cache_hit = True
            print(f"[cond] reused {num_text_tokens}-token embedding", flush=True)
        else:
            conditioned = time.time()
            condition_state = COND_PIPE(
                prompt=prompt,
                image=image,
                last_image=last_image,
                height=int(height),
                width=int(width),
            )
            prompt_embeds = condition_state.get("prompt_embeds")
            text_token_tags = condition_state.get("text_token_tags")
            condition_seconds = time.time() - conditioned
            num_text_tokens = int(prompt_embeds.shape[1])
            if conditioning_cache_key:
                cpu_embeds = prompt_embeds.detach().to("cpu")
                cpu_tags = text_token_tags.detach().to("cpu") if hasattr(text_token_tags, "detach") else text_token_tags
                CONDITION_CACHE[conditioning_cache_key] = (cpu_embeds, cpu_tags)
                while len(CONDITION_CACHE) > CONDITION_CACHE_SIZE:
                    CONDITION_CACHE.popitem(last=False)

    activate_lora = getattr(PIPE.transformer, "activate_lora", None)
    clear_lora = getattr(PIPE.transformer, "clear_lora", None)
    if lora_path and activate_lora is None:
        raise RuntimeError("LoRAs are supported only by the NVFP4 engine.")
    if activate_lora is not None:
        activate_lora(lora_path, egrid_path, float(lora_strength))
    begin_request = getattr(PIPE.transformer, "begin_request", None)
    end_request = getattr(PIPE.transformer, "end_request", None)
    if begin_request is not None:
        # MiniMaxH3Scheduler includes terminal sigma=0 in `num_inference_steps`, so N points execute N-1 forwards.
        begin_request(max(1, int(steps) - 1), acceleration)
    cache_stats = None
    try:
        with torch.inference_mode():
            state = PIPE(
                prompt_embeds=prompt_embeds.to("cuda", non_blocking=True),
                text_token_tags=text_token_tags,
                image=image,
                last_image=last_image,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=int(steps),
                generator=torch.Generator("cpu").manual_seed(int(seed)),
            )
    finally:
        if end_request is not None:
            cache_stats = end_request()
        if clear_lora is not None:
            clear_lora()
    return (
        state.get("videos")[0],
        state.get("audio")[0].cpu(),
        state.get("sampling_rate"),
        condition_seconds,
        num_text_tokens,
        cache_stats,
        condition_cache_hit,
    )


def _generate_with_hardware_retry(*args):
    """Resubmit once when ZeroGPU assigns a worker with a fatal ECC fault.

    The exception is raised by ``spaces`` before ``_generate`` begins, so CUDA cleanup inside the worker cannot repair
    it. A fresh decorated call lets the scheduler select another GPU. All application errors propagate immediately.
    """
    try:
        return _generate(*args)
    except Exception as error:
        if not any(marker in str(error) for marker in _RETRYABLE_GPU_ERRORS):
            raise
        print(f"[gpu] unhealthy ZeroGPU worker ({error}); resubmitting once", flush=True)
        return _generate(*args)


def generate(
    prompt, image_path=None, last_image_path=None, canvas=DEFAULT_CANVAS, duration=5, steps=28, seed=42,
    upsample=False, acceleration="Balanced", lora_preset="None", lora_repo="", lora_filename="",
    lora_strength=1.0, generation_preset=CUSTOM_PRESET, ip_token=None, progress=gr.Progress(track_tqdm=True),
):
    """One request. The appended UI preset leaves older positional API parameters intact."""
    if LOAD_ERROR:
        raise RuntimeError(LOAD_ERROR)
    if PIPE is None:
        raise RuntimeError("The denoiser is still loading.")
    if not prompt or not prompt.strip():
        raise ValueError("MiniMax-H3 always takes a prompt, keyframes or not.")
    canvas = resolve_canvas(canvas)
    print(f"[prompt] {prompt!r}", flush=True)

    generation_preset = str(generation_preset or CUSTOM_PRESET)
    if generation_preset != CUSTOM_PRESET:
        try:
            steps, acceleration, lora_preset = GENERATION_PRESETS[generation_preset]
        except KeyError as error:
            raise ValueError("Unknown speed and quality preset.") from error

    def input_path(value):
        if value is None:
            return None
        if isinstance(value, dict):
            return value.get("path")
        return getattr(value, "path", value)

    image_path, last_image_path = input_path(image_path), input_path(last_image_path)
    if image_path:
        image_path, canvas = _fit_keyframe(image_path, canvas)
    if last_image_path:
        last_image_path, canvas = _fit_keyframe(last_image_path, canvas)

    requested_steps = int(steps)
    displayed_steps = requested_steps
    if lora_preset == "Turbo · 4 steps":
        # The native scheduler includes its terminal zero in this count; 5 points produce 4 exact Euler evaluations.
        steps, displayed_steps = 5, 4
    elif lora_preset == "Turbo · 8 steps":
        steps, displayed_steps = 9, 8
    else:
        steps = requested_steps
    lora_path, egrid_path, lora_label, adapter_scale = resolve_lora(lora_preset, lora_repo, lora_filename)
    # Few-step distilled trajectories are too short to benefit safely from block reuse. Their speed comes from the
    # LoRA; keep every requested Turbo denoiser evaluation exact.
    run_acceleration = "Exact" if lora_preset.startswith("Turbo") else acceleration
    acceleration_label = (
        generation_preset
        if generation_preset != CUSTOM_PRESET
        else (f"{lora_preset}, exact denoiser" if lora_preset.startswith("Turbo") else acceleration)
    )

    from PIL import Image, ImageOps

    from diffusers.utils import encode_video

    num_frames = snap_frames(duration)
    height, width = CANVASES[canvas]

    def keyframe(path):
        # Both local conditioner and denoiser receive the same upright RGB source; their resize blocks then apply the
        # same target canvas independently.
        return ImageOps.exif_transpose(Image.open(path)).convert("RGB") if path else None

    first_frame, final_frame = keyframe(image_path), keyframe(last_image_path)
    prompt_embeds = text_token_tags = None
    condition_seconds = None
    num_text_tokens = None
    refined = ""

    # Prompt rewriting needs the discarded LM head and decoder tail, so it intentionally retains the remote path.
    # Normal generation—the default—keeps embeddings on this worker and never serializes them through another API.
    if upsample or COND_PIPE is None:
        progress(
            0.0,
            desc=f"Upsampling and conditioning on {CONDITIONER_SPACE} ..."
            if upsample
            else f"Local conditioner unavailable; using {CONDITIONER_SPACE} ...",
        )
        conditioned = time.time()
        prompt_embeds, text_token_tags, metadata, plan = encode_remote(
            prompt, image_path, last_image_path, canvas, num_frames, rewrite_prompt=upsample, ip_token=ip_token
        )
        condition_seconds = time.time() - conditioned
        height, width, num_frames = (int(metadata[key]) for key in ("height", "width", "num_frames"))
        num_text_tokens = int(plan["num_text_tokens"])
        refined = plan.get("refined_prompt") or ""
        if refined:
            print(f"[prompt:upsampled] {refined!r}", flush=True)

    progress(
        0.1,
        desc=("Local conditioning + " if prompt_embeds is None else "")
        + f"denoising {displayed_steps} steps at {width}x{height}, {num_frames} frames ...",
    )
    started = time.time()
    # gr.Server does not inject Progress parameters like Blocks does. Bind this request's tracker explicitly so the
    # spaces.GPU wrapper copies it into the forked worker and forwards Diffusers' tqdm updates back to this event.
    from gradio.context import LocalContext

    progress_token = LocalContext.progress.set(progress)
    try:
        conditioning_key = hashlib.sha256(
            "\0".join(
                [
                    prompt,
                    str(width),
                    str(height),
                    _sha256(image_path) if image_path else "",
                    _sha256(last_image_path) if last_image_path else "",
                ]
            ).encode()
        ).hexdigest()
        frames, audio, sampling_rate, local_condition_seconds, local_num_text_tokens, cache_stats, condition_cache_hit = _generate_with_hardware_retry(
            prompt,
            prompt_embeds,
            text_token_tags,
            first_frame,
            final_frame,
            height,
            width,
            num_frames,
            steps,
            seed,
            run_acceleration,
            lora_path,
            egrid_path,
            float(lora_strength) * adapter_scale,
            conditioning_key,
        )
    finally:
        LocalContext.progress.reset(progress_token)
    generate_seconds = time.time() - started
    cache_stats = cache_stats or {"computed": max(1, int(steps) - 1), "forecasted": 0}
    actual_evaluations = int(cache_stats.get("steps", cache_stats["computed"] + cache_stats["forecasted"]))
    if local_condition_seconds is not None:
        condition_seconds = local_condition_seconds
        num_text_tokens = local_num_text_tokens
    denoise_seconds = generate_seconds - (local_condition_seconds or 0.0)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"h3-{int(time.time() * 1000)}.mp4")
    encode_video(frames, fps=FPS, output_path=path, audio=audio, audio_sample_rate=sampling_rate)

    report = (
        f"`{width}x{height}`, {num_frames} frames ({num_frames / FPS:.3f} s), "
        f"{actual_evaluations} denoiser evaluations · "
        f"{cache_stats['computed']} full DiT evaluations + {cache_stats['forecasted']} cached block-stack reuses "
        f"({acceleration_label}) · LoRA {lora_label if lora_path else 'off'}"
        f"{f' @ {float(lora_strength):g}' if lora_path else ''} · "
        f"Sol-Attn {cache_stats.get('sol_sparse_calls', 0)} sparse calls · "
        f"conditioner {condition_seconds:.0f}s ({num_text_tokens} tokens"
        f"{', cached' if condition_cache_hit else ''}"
        f"{', upsampled' if refined else ''}) · "
        f"denoise + decode {denoise_seconds:.0f}s ({denoise_seconds / max(1, actual_evaluations):.1f} s/eval) · "
        f"seed {int(seed)}"
    )
    print(f"[gen] {report}", flush=True)
    return FileData(path=path), report, refined


def _fit_keyframe(image_path, current_canvas):
    """Cover-crop an uploaded keyframe to the closest supported aspect ratio and select that ratio's smallest
    (fastest) canvas, unless the user already picked a matching ratio."""
    if not image_path:
        return None, current_canvas
    from PIL import Image as _Image

    img = _Image.open(image_path)
    aspect = img.width / img.height
    fastest = {}
    for label, (h, w) in CANVASES.items():
        r = w / h
        if r not in fastest or w * h < fastest[r][1][0] * fastest[r][1][1]:
            fastest[r] = (label, (h, w))
    ratio = min(fastest, key=lambda r: abs(r - aspect))
    label, (h, w) = fastest[ratio]

    cur_h, cur_w = CANVASES[current_canvas]
    if abs(cur_w / cur_h - aspect) <= abs(ratio - aspect):
        label = current_canvas
        h, w = cur_h, cur_w

    target = w / h
    if abs(img.width / img.height - target) <= 1e-3:
        return image_path, label
    if img.width / img.height > target:
        new_w = int(img.height * target)
        left = (img.width - new_w) // 2
        img = img.crop((left, 0, left + new_w, img.height))
    else:
        new_h = int(img.width / target)
        top = (img.height - new_h) // 2
        img = img.crop((0, top, img.width, top + new_h))
    img.save(image_path)
    return image_path, label


PRESET_DESCRIPTIONS = {
    "Balanced — best overall (recommended)": "Full quality schedule with conservative Cache-DiT acceleration.",
    "Turbo 8-step — faster, cleaner": "Distilled eight-step path with better Turbo consistency.",
    "Turbo 4-step — fastest, more artifacts": "LightX2V v0.1 four-step preview; maximum speed with some detail loss.",
    "Exact 28-step — maximum fidelity": "Dense reference path with approximate caching disabled.",
    "Ultra cache — experimental speed": "Aggressive forecasting on the full schedule; inspect results carefully.",
    CUSTOM_PRESET: "Expose schedule, cache engine, LoRA source, and strength controls.",
}

EXAMPLES = [
    {
        "title": "Snow fox",
        "prompt": "A red fox trotting through a snowy pine forest at dawn, snow crunching underfoot",
        "canvas": "960x544 · 16:9 fast",
    },
    {
        "title": "Night market",
        "prompt": "A busy night market, neon signs reflecting in puddles, sizzling street food",
        "canvas": "544x960 · 9:16 fast",
    },
    {
        "title": "Concert hall",
        "prompt": "A cellist playing a slow melody in an empty concert hall",
        "canvas": "544x544 · 1:1 fast",
    },
    {
        "title": "Coastal cinema",
        "prompt": "Waves crashing against black basalt cliffs at golden hour, gulls calling above the surf",
        "canvas": "960x544 · 16:9 fast",
    },
]


app = Server(title="MiniMax-H3 Ultra Fast")
app.mount("/studio-assets", StaticFiles(directory=FRONTEND_DIR), name="studio-assets")


@app.mcp.tool(name="generate_video")
@app.api(name="generate")
def generate_api(
    prompt: str,
    image_path: FileData | None = None,
    last_image_path: FileData | None = None,
    canvas: str = DEFAULT_CANVAS,
    duration: float = 5,
    steps: int = 28,
    seed: float = 42,
    upsample: bool = False,
    acceleration: str = "Balanced",
    lora_preset: str = "None",
    lora_repo: str = "",
    lora_filename: str = "",
    lora_strength: float = 1.0,
    generation_preset: str = DEFAULT_PRESET,
    request: Request = None,
) -> tuple[FileData, str, str]:
    """Queued generation endpoint used by both the React studio and ordinary gradio_client callers."""
    ip_token = request.headers.get("x-ip-token") if request is not None else None
    return generate(
        prompt,
        image_path,
        last_image_path,
        canvas,
        duration,
        steps,
        seed,
        upsample,
        acceleration,
        lora_preset,
        lora_repo,
        lora_filename,
        lora_strength,
        generation_preset,
        ip_token=ip_token,
    )


@app.get("/status")
def studio_status():
    return {"ready": PIPE is not None and LOAD_ERROR is None, "status": status()}


@app.get("/studio-config")
def studio_config():
    return {
        "canvases": [
            {"label": label, "height": height, "width": width, "fast": "fast" in label}
            for label, (height, width) in CANVASES.items()
        ],
        "default_canvas": DEFAULT_CANVAS,
        "duration": {"min": MIN_UI_DURATION, "max": MAX_UI_DURATION, "default": 5},
        # `steps` and `acceleration` are the schedule this preset substitutes. The studio needs them to price each
        # preset with `get_duration`'s formula before the user picks one, so a 4-step run is visibly cheaper.
        "presets": [
            {
                "value": value,
                "description": PRESET_DESCRIPTIONS[value],
                "recommended": value == DEFAULT_PRESET,
                "custom": value == CUSTOM_PRESET,
                "steps": GENERATION_PRESETS.get(value, (28, "Balanced", "None"))[0],
                "acceleration": GENERATION_PRESETS.get(value, (28, "Balanced", "None"))[1],
            }
            for value in [*GENERATION_PRESETS, CUSTOM_PRESET]
        ],
        "default_preset": DEFAULT_PRESET,
        "custom_preset": CUSTOM_PRESET,
        "examples": EXAMPLES,
    }


@app.get("/", response_class=FileResponse)
def frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


load_models()

if __name__ == "__main__":
    app.launch(show_error=True, allowed_paths=[OUTPUT_DIR], mcp_server=True)
