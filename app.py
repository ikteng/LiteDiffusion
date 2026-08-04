"""MiniMax-H3 Ultra Fast: local layer-50 conditioning plus `t2va` / `fl2va` generation."""

from __future__ import annotations

import os
import tempfile
import time
import traceback
from functools import cache

# Before anything that could initialize CUDA: `import spaces` patches `torch.cuda` so model loading can happen at
# startup rather than on GPU time.
import spaces
import gradio as gr

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


def encode_remote(prompt, image_path, last_image_path, canvas, num_frames, rewrite_prompt=False):
    """`/encode` on the conditioner Space: a safetensors file holding `prompt_embeds` + `text_token_tags`, with the
    resolved `height` / `width` / `num_frames` in its metadata, plus the plan. `canvas` is the label."""
    from gradio_client import handle_file
    from safetensors import safe_open

    path, plan = conditioner().predict(
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
def _generate(prompt, prompt_embeds, text_token_tags, image, last_image, height, width, num_frames, steps, seed, acceleration):
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
    if prompt_embeds is None:
        if COND_PIPE is None:
            raise RuntimeError(f"The local conditioner is unavailable: {COND_ERROR or 'disabled'}")
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

    begin_request = getattr(PIPE.transformer, "begin_request", None)
    end_request = getattr(PIPE.transformer, "end_request", None)
    if begin_request is not None:
        begin_request(int(steps), acceleration)
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
    return (
        state.get("videos")[0],
        state.get("audio")[0].cpu(),
        state.get("sampling_rate"),
        condition_seconds,
        num_text_tokens,
        cache_stats,
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


def generate(prompt, image_path=None, last_image_path=None, canvas=DEFAULT_CANVAS, duration=5, steps=16, seed=42, upsample=False, acceleration="Ultra Fast", progress=gr.Progress(track_tqdm=True)):
    """One request. `upsample` is last and defaults off, so a positional API client that predates it is unaffected."""
    if LOAD_ERROR:
        raise gr.Error(LOAD_ERROR)
    if PIPE is None:
        raise gr.Error("The denoiser is still loading.")
    if not prompt or not prompt.strip():
        raise gr.Error("MiniMax-H3 always takes a prompt, keyframes or not.")

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
            prompt, image_path, last_image_path, canvas, num_frames, rewrite_prompt=upsample
        )
        condition_seconds = time.time() - conditioned
        height, width, num_frames = (int(metadata[key]) for key in ("height", "width", "num_frames"))
        num_text_tokens = int(plan["num_text_tokens"])
        refined = plan.get("refined_prompt") or ""

    progress(
        0.1,
        desc=("Local conditioning + " if prompt_embeds is None else "")
        + f"denoising {steps} steps at {width}x{height}, {num_frames} frames ...",
    )
    started = time.time()
    frames, audio, sampling_rate, local_condition_seconds, local_num_text_tokens, cache_stats = _generate_with_hardware_retry(
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
        acceleration,
    )
    generate_seconds = time.time() - started
    cache_stats = cache_stats or {"computed": int(steps), "forecasted": 0}
    if local_condition_seconds is not None:
        condition_seconds = local_condition_seconds
        num_text_tokens = local_num_text_tokens
    denoise_seconds = generate_seconds - (local_condition_seconds or 0.0)

    directory = os.path.join(tempfile.gettempdir(), "h3-outputs")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"h3-{int(time.time() * 1000)}.mp4")
    encode_video(frames, fps=FPS, output_path=path, audio=audio, audio_sample_rate=sampling_rate)

    report = (
        f"`{width}x{height}`, {num_frames} frames ({num_frames / FPS:.3f} s), {int(steps)} scheduler steps · "
        f"{cache_stats['computed']} DiT evaluations + {cache_stats['forecasted']} forecasts ({acceleration}) · "
        f"conditioner {condition_seconds:.0f}s ({num_text_tokens} tokens"
        f"{', upsampled' if refined else ''}) · "
        f"denoise + decode {denoise_seconds:.0f}s ({denoise_seconds / int(steps):.1f} s/step) · seed {int(seed)}"
    )
    print(f"[gen] {report}", flush=True)
    return path, report, refined, gr.update(visible=bool(refined))


def _fit_keyframe(image_path, current_canvas):
    """Cover-crop an uploaded keyframe to the closest supported aspect ratio and select that ratio's smallest
    (fastest) canvas, unless the user already picked a matching ratio."""
    if not image_path:
        return gr.update(), gr.update()
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
        return gr.update(), gr.update(value=label)
    if img.width / img.height > target:
        new_w = int(img.height * target)
        left = (img.width - new_w) // 2
        img = img.crop((left, 0, left + new_w, img.height))
    else:
        new_h = int(img.width / target)
        top = (img.height - new_h) // 2
        img = img.crop((0, top, img.width, top + new_h))
    img.save(image_path)
    return gr.update(value=image_path), gr.update(value=label)


load_models()

INTRO = """# MiniMax-H3 Ultra Fast

<div align="center">
  <a href="https://huggingface.co/MiniMaxAI/MiniMax-H3" target="_blank" rel="noopener"><strong>[ model ]</strong></a> &nbsp;
  <a href="https://huggingface.co/lilcheaty/MiniMax-H3-NVFP4" target="_blank" rel="noopener"><strong>[ NVFP4 ]</strong></a> &nbsp;
  <a href="https://www.minimax.io/blog/minimax-h3" target="_blank" rel="noopener"><strong>[ blog ]</strong></a> &nbsp;
  <a href="https://huggingface.co/spaces/multimodalart/minimax-h3" target="_blank" rel="noopener"><strong>[ original Space ]</strong></a>
</div>

**MiniMax-H3 Ultra Fast** runs a local truncated Qwen3-VL conditioner and the pruned Blackwell-native NVFP4
transformer with fused QKV, fused Q/K norm + RoPE, full-precision video/audio decoders, and synchronized sound. Its
default 16-step trajectory uses only **7 exact DiT evaluations** plus **9 bounded linear residual forecasts**.
It is optimized from the original [`multimodalart/minimax-h3`](https://huggingface.co/spaces/multimodalart/minimax-h3)
Space.
"""

CSS = """
.main.fillable {max-width: 1250px !important}
.dark .gradio-container { color: var(--body-text-color); }
"""

with gr.Blocks(title="MiniMax-H3 Ultra Fast") as demo:
    gr.Markdown(INTRO)

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                lines=3,
                value="A red fox trotting through a snowy pine forest at dawn, snow crunching underfoot",
            )
            upsample = gr.Checkbox(label="Upsample prompt", value=False)
            with gr.Row():
                image = gr.Image(label="First frame (optional)", type="filepath")
                last_image = gr.Image(label="Last frame (optional)", type="filepath")
            run = gr.Button("Generate", variant="primary")
            with gr.Accordion("Advanced options", open=False):
                canvas = gr.Dropdown(label="Canvas", choices=list(CANVASES), value=DEFAULT_CANVAS)
                duration = gr.Slider(label="Duration (s)", minimum=MIN_UI_DURATION, maximum=MAX_UI_DURATION, step=1, value=5)
                acceleration = gr.Radio(
                    label="Acceleration",
                    choices=["Ultra Fast", "Balanced", "Exact"],
                    value="Ultra Fast",
                    info="Ultra Fast forecasts at most 3 steps in a row; Exact evaluates every step.",
                )
                steps = gr.Slider(label="Scheduler steps", minimum=8, maximum=40, step=1, value=16)
                seed = gr.Number(label="Seed", value=42, precision=0)

        with gr.Column():
            video = gr.Video(label="Video + soundtrack")
            report = gr.Markdown(visible=False)
            # An output, so it can be revealed only for a request that asked for a rewrite.
            with gr.Accordion("Upsampled prompt", open=False, visible=False) as upsampled_panel:
                upsampled = gr.Textbox(show_label=False, lines=8, interactive=False)

    image.upload(_fit_keyframe, [image, canvas], [image, canvas])

    gr.Examples(
        examples=[
            ["A red fox trotting through a snowy pine forest at dawn, snow crunching underfoot", None, None, "1344x768 · 16:9 full"],
            ["A busy night market, neon signs reflecting in puddles, sizzling street food", None, None, "768x1344 · 9:16 full"],
            ["A cellist playing a slow melody in an empty concert hall", None, None, "768x768 · 1:1 full"],
            ["The fox looks around, then trots deeper into the forest", "examples/first.png", None, "1344x768 · 16:9 full"],
            ["A slow seamless camera move from the first view to the last", "examples/first.png", "examples/last.png", "1344x768 · 16:9 full"],
        ],
        inputs=[prompt, image, last_image, canvas],
        outputs=[video, report, upsampled, upsampled_panel],
        fn=generate,
        cache_examples=True,
        cache_mode="lazy",
    )

    run.click(
        generate,
        [prompt, image, last_image, canvas, duration, steps, seed, upsample, acceleration],
        [video, report, upsampled, upsampled_panel],
        api_name="generate",
    )

    gr.Markdown(
        '<div style="text-align:center"><a href="https://x.com/realmrfakename" target="_blank" '
        'rel="noopener">@realmrfakename</a></div>'
    )


if __name__ == "__main__":
    demo.queue().launch(show_error=True, theme=gr.themes.Citrus(), css=CSS)
