"""Lightweight CPU text-to-image Space: a small choice of distilled diffusion models, no GPU required."""

from __future__ import annotations

import os
import time

import gradio as gr
import torch
from diffusers import AutoPipelineForText2Image

torch.set_num_threads(os.cpu_count() or 4)

MODELS = {
    "SD-Turbo — best quality (Recommended)": {
        "repo": "stabilityai/sd-turbo",
        "steps": 1,
        "guidance_scale": 0.0,
        "size": 512,
    },
    "SDXS-512 — fastest": {
        "repo": "IDKiro/sdxs-512-0.9",
        "steps": 1,
        "guidance_scale": 0.0,
        "size": 512,
    },
    "Tiny SD — smaller model, more steps": {
        "repo": "segmind/tiny-sd",
        "steps": 20,
        "guidance_scale": 7.0,
        "size": 512,
    },
}
DEFAULT_MODEL = next(iter(MODELS))

_pipelines: dict[str, AutoPipelineForText2Image] = {}


def load_pipeline(model_name: str) -> AutoPipelineForText2Image:
    if model_name not in _pipelines:
        repo = MODELS[model_name]["repo"]
        pipe = AutoPipelineForText2Image.from_pretrained(repo, torch_dtype=torch.float32)
        pipe.to("cpu")
        _pipelines[model_name] = pipe
    return _pipelines[model_name]


def generate(prompt: str, model_name: str, seed: int, progress=gr.Progress()):
    if not prompt or not prompt.strip():
        raise gr.Error("Enter a prompt first.")

    progress(0, desc=f"Loading {model_name}…")
    pipe = load_pipeline(model_name)
    config = MODELS[model_name]

    generator = torch.Generator(device="cpu")
    if seed is not None and seed >= 0:
        generator = generator.manual_seed(int(seed))

    progress(0.3, desc="Generating…")
    start = time.time()
    result = pipe(
        prompt,
        num_inference_steps=config["steps"],
        guidance_scale=config["guidance_scale"],
        height=config["size"],
        width=config["size"],
        generator=generator,
    )
    elapsed = time.time() - start

    image = result.images[0]
    return image, f"Generated in {elapsed:.1f}s on CPU with {model_name.split(' — ')[0]}."


with gr.Blocks(title="Fast CPU Text-to-Image") as demo:
    gr.Markdown(
        "# Fast CPU Text-to-Image\n"
        "Small, distilled diffusion models that run on ordinary CPUs. No GPU required."
    )
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Prompt", placeholder="A watercolor fox in a snowy forest", lines=3)
            model_choice = gr.Dropdown(
                choices=list(MODELS.keys()), value=DEFAULT_MODEL, label="Model"
            )
            seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
            run_button = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            output_image = gr.Image(label="Result", type="pil")
            status = gr.Markdown()

    run_button.click(
        fn=generate,
        inputs=[prompt, model_choice, seed],
        outputs=[output_image, status],
    )
    prompt.submit(
        fn=generate,
        inputs=[prompt, model_choice, seed],
        outputs=[output_image, status],
    )

if __name__ == "__main__":
    demo.queue().launch()
