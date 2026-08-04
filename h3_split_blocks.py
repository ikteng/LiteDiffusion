"""Composable conditioner and generator halves of MiniMax-H3, for both checkpoint partitions.

The blocks cut `MiniMaxH3Blocks` at its `text_encoder` step. They can run in separate Spaces with `prompt_embeds` and
`text_token_tags` as a wire format, or sequentially in one GPU worker when a compact local conditioner fits beside
the generator.

`resize` / `setup` run on **both** sides: they own no pretrained component, and each half needs the canvas and the
prepared keyframes or normalized references. Both conditioner halves also return the resolved `height` / `width` /
`num_frames`, which the generating half pins rather than re-deriving.

Two things the blocks leave to the caller: a keyframe reaches them EXIF-transposed and in RGB, and the `t2va` / `fl2va`
frame count is aligned to `17 * n + 5` before the call, since that arithmetic lives on the denoising side of the cut.
"""

import torch

from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3AfterDenoiseStep
from diffusers.modular_pipelines.minimax_h3.encoders import (
    MiniMaxH3Ref2VAReferenceEncoderStep,
    MiniMaxH3Ref2VATextEncoderStep,
    MiniMaxH3TextEncoderStep,
)
from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
    MiniMaxH3AutoKeyframeVaeEncoderStep,
    MiniMaxH3AutoResizeStep,
    MiniMaxH3CoreDenoiseStep,
    MiniMaxH3DecodeStep,
    MiniMaxH3Ref2VACoreDenoiseStep,
    _generation_outputs,
)
from diffusers.modular_pipelines.modular_pipeline import SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import OutputParam


class MiniMaxH3ParallelDecodeStep(MiniMaxH3DecodeStep):
    """Decode independent video and audio latents concurrently without changing either decoder's precision."""

    @torch.no_grad()
    def __call__(self, components, state):
        device = components._execution_device
        if device.type != "cuda":
            return super().__call__(components, state)

        block_state = self.get_block_state(state)
        current_stream = torch.cuda.current_stream(device)
        video_stream = torch.cuda.Stream(device=device)
        audio_stream = torch.cuda.Stream(device=device)
        video_stream.wait_stream(current_stream)
        audio_stream.wait_stream(current_stream)

        with torch.cuda.stream(video_stream):
            video_mean = torch.as_tensor(components.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
            video_std = torch.as_tensor(components.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
            video_latents = block_state.latents * video_std + video_mean
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                decoded_video = components.vae.decode(video_latents, return_dict=False)[0]
            pixel_mean = torch.as_tensor(components.pixel_mean, device=device).view(1, -1, 1, 1, 1)
            pixel_std = torch.as_tensor(components.pixel_std, device=device).view(1, -1, 1, 1, 1)
            decoded_video = (decoded_video.float() * pixel_std + pixel_mean).clamp(0, 1)

        with torch.cuda.stream(audio_stream):
            audio_mean = torch.as_tensor(components.audio_vae.config.latents_mean, device=device).view(1, -1, 1)
            audio_std = torch.as_tensor(components.audio_vae.config.latents_std, device=device).view(1, -1, 1)
            audio_latents = block_state.audio_latents * audio_std + audio_mean
            decoded_audio = components.audio_vae.decode(audio_latents, return_dict=False)[0]
            decoded_audio = decoded_audio.float().permute(1, 0, 2)

        current_stream.wait_stream(video_stream)
        current_stream.wait_stream(audio_stream)
        block_state.videos = components.video_processor.postprocess_video(
            decoded_video, output_type=block_state.output_type
        )
        block_state.audio = decoded_audio
        block_state.sampling_rate = components.audio_sampling_rate
        self.set_block_state(state, block_state)
        return components, state


def _wire_outputs(num_frames: bool = True) -> list[OutputParam]:
    """The wire format of the split. `num_frames` is declared by the `ref2va` half alone, whose setup resolves one."""
    return [
        OutputParam.template("prompt_embeds"),
        OutputParam("text_token_tags", description="The per-row modality tag of every row of `prompt_embeds`."),
        OutputParam("height", type_hint=int, description="Resolved height of the generated video in pixels."),
        OutputParam("width", type_hint=int, description="Resolved width of the generated video in pixels."),
        *(
            [OutputParam("num_frames", type_hint=int, description="Resolved number of frames, of the form 17 * n + 5.")]
            if num_frames
            else []
        ),
    ]


class MiniMaxH3ConditionerBlocks(SequentialPipelineBlocks):
    """The conditioner half of a split MiniMax-H3: the keyframes on the canvas plus the Qwen3-VL read at layer 50."""

    model_name = "minimax-h3"
    block_classes = [MiniMaxH3AutoResizeStep, MiniMaxH3TextEncoderStep]
    block_names = ["resize", "text_encoder"]

    @property
    def description(self):
        return (
            "The conditioner half of a split MiniMax-H3 deployment: puts the keyframes onto the target canvas and "
            "encodes MiniMax-H3's presentation of the request into the `prompt_embeds` / `text_token_tags` pair the "
            "denoising half consumes. The frame count is the caller's to align."
        )

    @property
    def outputs(self):
        return _wire_outputs(num_frames=False)


class MiniMaxH3GeneratorBlocks(SequentialPipelineBlocks):
    """The denoising half of a split MiniMax-H3: `MiniMaxH3Blocks` with its `text_encoder` step removed."""

    model_name = "minimax-h3"
    block_classes = [
        MiniMaxH3AutoResizeStep,
        MiniMaxH3AutoKeyframeVaeEncoderStep,
        MiniMaxH3CoreDenoiseStep,
        MiniMaxH3AfterDenoiseStep,
        MiniMaxH3ParallelDecodeStep,
    ]
    block_names = ["resize", "vae_encoder", "denoise", "after_denoise", "decode"]

    @property
    def description(self):
        return (
            "The denoising half of a split MiniMax-H3 deployment: the `t2va` / `fl2va` branch of `MiniMaxH3Blocks` "
            "without its text-encoder step, so `prompt_embeds` and `text_token_tags` come in as inputs and the "
            "conditioner is supplied by the caller or by the preceding local conditioner half."
        )

    @property
    def outputs(self):
        return _generation_outputs()


class MiniMaxH3Ref2VAConditionerBlocks(SequentialPipelineBlocks):
    """The conditioner half of a split `ref2va`: the resolved plan plus the Qwen3-VL read at its 50th layer.

    Component for component this is `MiniMaxH3ConditionerBlocks`, so one conditioner Space serves both partitions.
    What differs is the presentation: `ref2va` prepends a label per reference and a vision block per image and per
    merged video frame pair, so the references themselves have to reach this half.
    """

    model_name = "minimax-h3"
    block_classes = [MiniMaxH3Ref2VASetupStep, MiniMaxH3Ref2VATextEncoderStep]
    block_names = ["setup", "text_encoder"]

    @property
    def description(self):
        return (
            "The conditioner half of a split MiniMax-H3 `ref2va` deployment: resolves the request plan (canvas, frame "
            "count, references normalized onto MiniMax-H3's own rates and resolutions) and encodes MiniMax-H3's "
            "presentation of it into the `prompt_embeds` / `text_token_tags` pair the denoising half consumes."
        )

    @property
    def outputs(self):
        return _wire_outputs()


class MiniMaxH3Ref2VAGeneratorBlocks(SequentialPipelineBlocks):
    """The denoising half of a split `ref2va`: the `ref2va` branch with its `text_encoder` step removed.

    `reference_encoder` stays here, next to the two autoencoders it runs: its output shapes are where every reference
    block's geometry in the packed layout comes from.
    """

    model_name = "minimax-h3"
    block_classes = [
        MiniMaxH3Ref2VASetupStep,
        MiniMaxH3Ref2VAReferenceEncoderStep,
        MiniMaxH3Ref2VACoreDenoiseStep,
        MiniMaxH3AfterDenoiseStep,
        MiniMaxH3ParallelDecodeStep,
    ]
    block_names = ["setup", "reference_encoder", "denoise", "after_denoise", "decode"]

    @property
    def description(self):
        return (
            "The denoising half of a split MiniMax-H3 `ref2va` deployment: the `ref2va` branch of `MiniMaxH3Blocks` "
            "without its text-encoder step, so `prompt_embeds` and `text_token_tags` come in as inputs and the "
            "conditioner is supplied by the caller or preceding local half. The transformer is the `transformer_ref` partition."
        )

    @property
    def outputs(self):
        return _generation_outputs()
