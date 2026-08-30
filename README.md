# LiteDiffusion

A local text-to-image app: a FastAPI backend around small, distilled diffusion models, with a React/TypeScript
frontend (Generate, Gallery, Settings, About tabs). Runs entirely on your own machine — no cloud billing, no
time-limited hosted quota.

## Models

| model | params | steps | notes |
|---|---:|---:|---|
| [`IDKiro/sdxs-512-0.9`](https://huggingface.co/IDKiro/sdxs-512-0.9) | ~0.6B | 1 | Smallest and fastest; built specifically for real-time generation. **Default.** |
| [`stabilityai/sd-turbo`](https://huggingface.co/stabilityai/sd-turbo) | ~1B | 1 | Distilled SD2.1; best quality of the three, still single-step. |
| [`segmind/tiny-sd`](https://huggingface.co/segmind/tiny-sd) | ~0.5B UNet | 20 | Smaller architecture, but needs more steps, so it isn't the fastest wall-clock option. |

All three run through `diffusers.DiffusionPipeline`. The backend auto-detects a CUDA GPU
(`torch.cuda.is_available()`) and uses it in FP16 if present, otherwise falls back to FP32 on CPU. Each pipeline
is loaded lazily on first use and kept in memory for the rest of the session.

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
distilled models above; video generation (`backend/pipelines.py:generate_video`) renders a few keyframes with
the chosen image model, then interpolates between them into a short MP4 — fast and CPU-friendly, with no extra
model download. A heavier real text-to-video model (e.g. AnimateLCM) can later replace that function behind the
same `media_type: video` path.

## Why not MiniMax-H3

MiniMax-H3 is a 20B+ parameter video-and-audio diffusion transformer built around Blackwell-only NVFP4 tensor-core
kernels; there's no practical way to run that on a typical local machine. This project instead targets a
different, achievable goal: real text-to-image generation in a few seconds on hardware people already have.
