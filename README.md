---
title: LiteDiffusion
emoji: 🖼️
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.26.0
python_version: '3.12'
app_file: app.py
pinned: false
short_description: Small distilled text-to-image models, fast on ordinary CPUs
---

# LiteDiffusion

A lightweight Gradio app for generating images from text on plain CPUs — no GPU, no CUDA, no quantization
toolchain required. It trades MiniMax-H3's video+audio generation for small, distilled text-to-image models that
are actually fast without an accelerator.

## Models

| model | params | steps | notes |
|---|---:|---:|---|
| [`stabilityai/sd-turbo`](https://huggingface.co/stabilityai/sd-turbo) | ~1B | 1 | Distilled SD2.1; best quality of the three, still single-step. Default. |
| [`IDKiro/sdxs-512-0.9`](https://huggingface.co/IDKiro/sdxs-512-0.9) | ~0.6B | 1 | Smallest and fastest; built specifically for real-time generation. |
| [`segmind/tiny-sd`](https://huggingface.co/segmind/tiny-sd) | ~0.5B UNet | 20 | Smaller architecture, but needs more steps, so it isn't the fastest wall-clock option. |

All three run through `diffusers.AutoPipelineForText2Image` in FP32 on CPU. Each pipeline is loaded lazily on
first use and then kept in memory for the rest of the session.

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

The first generation with a given model downloads its weights from the Hugging Face Hub (a few hundred MB to
~2 GB depending on the model) and caches them locally.

## Why not MiniMax-H3

MiniMax-H3 is a 20B+ parameter video-and-audio diffusion transformer built around Blackwell-only NVFP4 tensor-core
kernels; there is no way to make that fast on a CPU; the parameter count and attention cost make a CPU pass take
minutes to hours per clip regardless of kernel choice. This Space instead targets a different, achievable goal:
real text-to-image generation in a few seconds on hardware everyone already has.
