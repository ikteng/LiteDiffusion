# LiteDiffusion

A local text-to-image app: a FastAPI backend around small, distilled diffusion models, with a React/TypeScript
frontend (Generate, Gallery, Settings, About tabs). Runs entirely on your own machine — no cloud billing, no
time-limited hosted quota. Text-to-image models are small (≤2.5GB) and CPU-friendly; video models are heavier
(13-28GB) and benefit from an NVIDIA GPU.

## Models

| model | type | notes |
|---|---|---|
| [`IDKiro/sdxs-512-0.9`](https://huggingface.co/IDKiro/sdxs-512-0.9) | local image | 1-step SDXL; fastest model, good for real-time. **Default.** |
| [`tianweiy/DMD2`](https://huggingface.co/tianweiy/DMD2) | local image | 4-step SDXL distilled; good quality with few steps. |
| [`latent-consistency/lcm-lora-sdxl`](https://huggingface.co/latent-consistency/lcm-lora-sdxl) | local image | 1-2 step LCM-LoRA SDXL; dreamy aesthetic, very fast. |
| [`Efficient-Large-Model/Sana_600M_512px`](https://huggingface.co/Efficient-Large-Model/Sana_600M_512px_diffusers) | local image | 0.6B param, 8-step; tiny and sharp. |
| [`PixArt-alpha/PixArt-Sigma-XL-2-512-MS`](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-512-MS) | local image | 0.6B param, 6-step; small and detailed. |
| [`stabilityai/sdxl-turbo`](https://huggingface.co/stabilityai/sdxl-turbo) | local image | 1-step SDXL; best quality at speed, ~2.3GB. |
| [`ali-vilab/text-to-video-ms-1.7b`](https://huggingface.co/ali-vilab/text-to-video-ms-1.7b) | local video | 1.7B param T2V; smallest local video download. **Default.** |
| [`Lightricks/LTX-Video`](https://huggingface.co/Lightricks/LTX-Video) | local video | LTX-Video 2B; 30-step, 512px T2V, fast on GPU. |
| [`Lightricks/LTX-Video`](https://huggingface.co/Lightricks/LTX-Video) | local video (I2V) | LTX-Video 2B image-to-video — good quality, supports first+last frame. **Recommended.** |
| [`ali-vilab/i2vgen-xl`](https://huggingface.co/ali-vilab/i2vgen-xl) | local video (I2V) | Lightweight 1.3B image-to-video model — smallest local video download. |
| [`stabilityai/sv3d`](https://huggingface.co/stabilityai/sv3d) | local video (I2V) | 2.5B image-to-3D (multi-view), CPU-friendly, small download. |
| [`stabilityai/sv3d_humin3d`](https://huggingface.co/stabilityai/sv3d_humin3d) | local video (I2V) | 3.2B image-to-3D optimized for human figures, CPU-friendly. |
| [`stabilityai/svd`](https://huggingface.co/stabilityai/svd) | local video (I2V) | ~1.2B Param I2V (Stable Video Diffusion); smallest local I2V download. **Recommended.** |
| [`stabilityai/svd_xt`](https://huggingface.co/stabilityai/svd_xt) | local video (I2V) | SVD extended — 14-frame, higher quality variant. |
| [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | online image | Free online inference via Hugging Face API — no local download. |
| [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5) | online image | Free online inference via Hugging Face API — no local download. |

Local models run through `diffusers.DiffusionPipeline` (or specialized pipelines for video). The backend auto-detects a CUDA GPU
(`torch.cuda.is_available()`) and uses it in FP16 if present, otherwise falls back to FP32 on CPU. Each pipeline
is loaded lazily on first use and kept in memory for the rest of the session. Only the default model
(`sdxs-512`) is prefetched at startup; heavier models download on first selection.

Video generation has two pages: Text-to-Video for prompt-only generation using T2V models, and
Image-to-Video for animating a reference image with I2V models (text + image). The Image-to-Video page
supports an optional last frame for first+last-frame control on LTX-Video. All video models run locally;
heavier checkpoints download on first selection.

## Running locally

```bash
pip install -r requirements.txt

cd frontend
npm install
npm run build
cd ..

python run.py
```

`python run.py` starts the FastAPI server on `http://127.0.0.1:8000` and opens it in your browser. The frontend
build step is one-time (or whenever the UI changes) — day-to-day, `python run.py` is the only command you need.

The first generation with a given model downloads its weights from the Hugging Face Hub (a few hundred MB to
~2 GB depending on the model) and caches them locally. If you have an NVIDIA GPU, installing a CUDA-enabled
build of `torch` (see [pytorch.org](https://pytorch.org/get-started/locally/)) will speed generation up
significantly; otherwise it runs fine on CPU.

Generated images and their metadata are saved to `outputs/` and persist across restarts (visible in the Gallery
tab).

## Frontend development

For UI iteration with hot reload, run the API and the Vite dev server side by side:

```bash
uvicorn backend.main:app --reload --port 8000
```

```bash
cd frontend
npm run dev
```

The dev server proxies `/api` and `/outputs` requests to the FastAPI server.

## Architecture

- `backend/` — FastAPI app: `config.py` (models/device), `pipelines.py` (generation), `jobs.py` (async job
  queue, one generation at a time), `store.py` (disk-persisted history), `routes/` (REST endpoints under `/api`).
- `frontend/` — Vite + React + TypeScript SPA, built to `frontend/dist/` and served by the backend.
- `run.py` — single entrypoint for normal use.

The job/history data model carries a `media_type` field (`image` or `video`). Image generation uses the local
models above. Video generation uses the Image-to-Video page with an I2V model that animates a reference image you provide.

## Why not MiniMax-H3

MiniMax-H3 is a 20B+ parameter video-and-audio diffusion transformer built around Blackwell-only NVFP4 tensor-core
kernels; there's no practical way to run that on a typical local machine. This project instead targets a
different, achievable goal: real text-to-image generation in a few seconds on hardware people already have.
