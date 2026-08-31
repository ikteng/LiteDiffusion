# LiteDiffusion

A local text-to-image app: a FastAPI backend around small, distilled diffusion models, with a React/TypeScript
frontend (Generate, Gallery, Settings, About tabs). Runs entirely on your own machine — no cloud billing, no
time-limited hosted quota.

## Models

| model | type | notes |
|---|---|---|
| [`IDKiro/sdxs-512-0.9`](https://huggingface.co/IDKiro/sdxs-512-0.9) | local image | Smallest and fastest; built specifically for real-time generation. **Default.** |
| [`stabilityai/sd-turbo`](https://huggingface.co/stabilityai/sd-turbo) | local image | Distilled SD2.1; best quality of the local options, still single-step. |
| [`ali-vilab/text-to-video-ms-1.7b`](https://huggingface.co/ali-vilab/text-to-video-ms-1.7b) | local video | 1.7B parameter text-to-video model. |
| [`kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-5s`](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-5s-Diffusers) | local video | 2B parameter distilled T2V, 16 steps. |
| [`kandinskylab/Kandinsky-5.0-T2V-Lite-nocfg-5s`](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-nocfg-5s-Diffusers) | local video | 2B parameter no-CFG balanced T2V. |
| [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | online image | Free online inference via Hugging Face API — no local download. |
| [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5) | online image | Free online inference via Hugging Face API — no local download. |
| [`ali-vilab/text-to-video-ms-1.7b`](https://huggingface.co/ali-vilab/text-to-video-ms-1.7b) | online video | Free online inference via Hugging Face API — no local download. |
| [`guoyww/animatediff`](https://huggingface.co/guoyww/animatediff) | online video | Free online inference via Hugging Face API — no local download. |

Local models run through `diffusers.DiffusionPipeline` (or specialized pipelines for T2V). The backend auto-detects a CUDA GPU
(`torch.cuda.is_available()`) and uses it in FP16 if present, otherwise falls back to FP32 on CPU. Each pipeline
is loaded lazily on first use and kept in memory for the rest of the session. Only the default model
(`sdxs-512`) is prefetched at startup; heavier models download on first selection.

Video generation uses the chosen image model to render a few keyframes, then ffmpeg motion-compensated interpolation
produces a smooth MP4 — fast and CPU-friendly, with no extra model download. Local T2V models generate frames directly
for better motion. Or use any remote option for online inference without downloading weights.

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

The job/history data model carries a `media_type` field (`image` or `video`). Image generation uses the
local or remote models above. Video generation can use remote online inference or local keyframe interpolation
depending on the selected model.

## Why not MiniMax-H3

MiniMax-H3 is a 20B+ parameter video-and-audio diffusion transformer built around Blackwell-only NVFP4 tensor-core
kernels; there's no practical way to run that on a typical local machine. This project instead targets a
different, achievable goal: real text-to-image generation in a few seconds on hardware people already have.
