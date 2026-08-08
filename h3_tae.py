"""Tiny MiniMax-H3 latent previews streamed through Gradio progress packets.

The 2D decoder architecture follows ComfyUI's MIT-licensed TAESD blocks and Kijai's H3 checkpoint layout. It is
preview-only: final frames still come exclusively from MiniMax-H3's full video VAE.
"""

from __future__ import annotations

import os
import time
from urllib.parse import quote

import torch
import torch.nn as nn


TAE_REPO = "Kijai/MiniMax-H3-TAE"
TAE_FILE = "vae_approx/taeh3.safetensors"
PREVIEW_MAX_EDGE = 384
PREVIEW_POINTS = 4

_DECODER = None
_OUTPUT_DIR = None
_FAILED = False


def _conv(n_in: int, n_out: int, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class _Clamp(nn.Module):
    def forward(self, value):
        return torch.tanh(value / 3) * 3


class _Block(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            _conv(n_in, n_out), nn.ReLU(), _conv(n_out, n_out), nn.ReLU(), _conv(n_out, n_out)
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()

    def forward(self, value):
        return self.fuse(self.conv(value) + self.skip(value))


def _build_decoder(state):
    by_index = {}
    for key, value in state.items():
        head, _, tail = key.partition(".")
        by_index.setdefault(int(head), {})[tail] = value
    modules = []
    for index in range(max(by_index) + 1):
        entry = by_index.get(index)
        if entry is None:
            modules.append(_Clamp() if index == 0 else nn.ReLU() if index == 2 else nn.Upsample(scale_factor=2))
        elif "conv.0.weight" in entry:
            weight = entry["conv.0.weight"]
            modules.append(_Block(weight.shape[1], weight.shape[0]))
        elif "weight" in entry:
            weight = entry["weight"]
            modules.append(_conv(weight.shape[1], weight.shape[0], bias="bias" in entry))
        else:
            raise ValueError(f"Unrecognized H3 TAE module {index}: {sorted(entry)}")
    decoder = nn.Sequential(*modules)
    decoder.load_state_dict(state)
    return decoder.eval()


def load_preview_model(output_dir: str):
    """Load the 9.8 MB decoder on CPU at startup; CUDA placement happens only inside a booked request."""
    global _DECODER, _OUTPUT_DIR, _FAILED
    if _DECODER is not None or _FAILED:
        return
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        path = hf_hub_download(TAE_REPO, TAE_FILE)
        _DECODER = _build_decoder(load_file(path, device="cpu"))
        _OUTPUT_DIR = os.path.join(output_dir, "previews")
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        print(f"[tae] loaded {TAE_REPO}/{TAE_FILE}", flush=True)
    except Exception as error:
        _FAILED = True
        print(f"[tae] disabled ({type(error).__name__}: {error})", flush=True)


def _unpatchify(components, state):
    patch_t, patch_h, patch_w = components.patch_size
    channels = components.vae_latent_channels
    rows = state.latents[state.num_condition_video_rows :]
    rows = rows.reshape(
        -1,
        state.num_latent_frames // patch_t,
        state.latent_height // patch_h,
        state.latent_width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return rows.reshape(-1, channels, state.num_latent_frames, state.latent_height, state.latent_width)


@torch.inference_mode()
def maybe_emit_preview(components, state, step: int, total: int) -> None:
    """Decode three representative latent frames at four milestones and publish a tiny animated WebP."""
    if _DECODER is None or _OUTPUT_DIR is None or total < 2:
        return
    milestones = {max(0, round((total - 1) * fraction)) for fraction in (0.12, 0.38, 0.66, 0.9)}
    if step not in milestones:
        return
    try:
        from gradio.context import LocalContext
        from PIL import Image

        progress = LocalContext.progress.get()
        if progress is None:
            return
        latents = _unpatchify(components, state)
        picks = torch.linspace(0, latents.shape[2] - 1, min(3, latents.shape[2])).round().long().tolist()
        decoder = _DECODER.to(device=latents.device, dtype=torch.bfloat16)
        frames = []
        for index in picks:
            rgb = decoder(latents[:1, :, index].to(torch.bfloat16))[0].float().clamp(0, 1)
            array = rgb.mul(255).to(torch.uint8).movedim(0, -1).cpu().numpy()
            image = Image.fromarray(array)
            image.thumbnail((PREVIEW_MAX_EDGE, PREVIEW_MAX_EDGE), Image.Resampling.LANCZOS)
            frames.append(image)
        name = f"tae-{os.getpid()}-{int(time.time() * 1000)}.webp"
        path = os.path.join(_OUTPUT_DIR, name)
        frames[0].save(
            path,
            format="WEBP",
            save_all=len(frames) > 1,
            append_images=frames[1:],
            duration=420,
            loop=0,
            quality=72,
            method=3,
        )
        url = "/gradio_api/file=" + quote(path, safe="")
        progress((step + 1) / total, desc=f"TAE_PREVIEW|{url}|Preview {step + 1}/{total}")
    except Exception as error:
        print(f"[tae] preview skipped ({type(error).__name__}: {error})", flush=True)


def status() -> str:
    return "TAE live previews" if _DECODER is not None else "TAE unavailable"
