import type { ReactNode } from "react";
import { ExternalLink, Heart } from "lucide-react";
import { Button } from "../ui/Button";
import { Sheet } from "../ui/Sheet";

const SPACE_URL = "https://huggingface.co/spaces/mrfakename/minimax-h3-faster";

const LINKS = [
  ["Original Space", "https://huggingface.co/spaces/multimodalart/minimax-h3"],
  ["MiniMax-H3 model", "https://huggingface.co/MiniMaxAI/MiniMax-H3"],
  ["NVFP4 checkpoint", "https://huggingface.co/lilcheaty/MiniMax-H3-NVFP4"],
  ["4-step Turbo LoRA", "https://huggingface.co/lightx2v/Minimax-h3-Turbo"],
  ["Sana / Sol-Engine", "https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/optimized"],
  ["H3 TAE previews", "https://huggingface.co/Kijai/MiniMax-H3-TAE"],
];

/**
 * What this Space actually does, for someone who wants to know.
 *
 * A short technical note rather than a marketing panel: the numbers here are the ones that explain why a model whose
 * BF16 weights do not fit on the worker runs at all, and why the presets cost what they cost. Everything stated is
 * read off `app.py`, `h3_nvfp4.py` and `h3_local_conditioner.py` — if one of those changes, this changes with it.
 */
export function AboutSheet({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <Sheet
      open={open}
      onClose={onClose}
      placement="side"
      title="How this Space works"
      subtitle="MiniMax-H3 generating video and its soundtrack together, on one Blackwell ZeroGPU worker."
      footer={
        <div className="flex justify-end gap-2">
          <Button variant="ghost" onClick={onClose}>
            Close
          </Button>
          <Button variant="primary" onClick={() => window.open(SPACE_URL, "_blank")}>
            <Heart /> Like the Space
          </Button>
        </div>
      }
    >
      <div className="flex flex-col gap-7 text-[13px] leading-[1.65] text-muted">
        <Part title="The pipeline">
          <P>
            One prompt makes one pass. A local <Term>Qwen3-VL</Term> conditioner encodes your text (and any keyframes)
            into embeddings; a 50-block diffusion transformer denoises video and audio rows <em>in the same packed
            sequence</em>, so the soundtrack is generated with the picture rather than dubbed onto it afterwards; then
            two decoders run and PyAV muxes the result into an MP4.
          </P>
          <P>
            Everything happens inside one <Term>@spaces.GPU</Term> call on a single 95 GiB worker. There is no
            cross-Space round trip and no layerwise host-to-device weight traffic in the denoising loop — which is what
            the earlier 4-bit builds of this model spent most of their time on.
          </P>
        </Part>

        <Part title="Why the weights fit">
          <div className="rounded-xl bg-sunken px-3 py-0.5 ring-1 ring-inset ring-line">
            <Row label="Transformer" value="61.7 GiB → 12.5 GB" />
            <Row label="Conditioner" value="66.7 GB → 15.7 GB" />
            <Row label="Whole stack" value="77.3 GB → ~44 GB" />
          </div>
          <P>
            The transformer is both pruned and quantized. MiniMax-H3 spends ~13B of its 33.1B parameters on AdaLN
            modulation projections whose outputs depend only on the timestep, so the pruned checkpoint replaces them
            with a 1025-point lookup curve — 20.1B parameters left. Attention and MLP matrices are then stored as{" "}
            <Term>NVFP4</Term> and multiplied natively on Blackwell tensor cores, which is why this Space requires an
            sm120 card and the CUDA 13 build of PyTorch.
          </P>
          <P>
            The conditioner is truncated rather than compressed alone: H3 reads the unnormalized hidden state after
            language layer 50, so the other 14 layers and the LM head are simply absent. Both autoencoders are the one
            thing left at full precision — a bfloat16 audio VAE decodes the soundtrack about 20 dB too quiet.
          </P>
        </Part>

        <Part title="What the presets change">
          <P>
            <Term>Exact</Term> evaluates all 50 blocks on every scheduler step. Nothing is reused and sparse attention
            is off. It is the reference path.
          </P>
          {/* The measurements are bound with non-breaking spaces — "35 s → 23 s." is one figure and reads as noise
              when the line break lands inside it. Note that a JSX comment cannot go mid-paragraph: it splits the text
              node and eats the surrounding whitespace. */}
          <P>
            <Term>Balanced</Term> is Cache-DiT block reuse. Block 0 always runs; its output residual is compared against
            the last one on a strided probe, and if the relative change is under 8% the remaining 49 blocks are skipped
            and the cached tail is replayed with a first-order Taylor correction. The first three and last two steps are
            always dense, and it will never skip more than twice in a row. Measured on a warm 960×544 / 56-frame /
            28-step run: 20 full evaluations and 7 reuses instead of 27, denoise and decode 35&nbsp;s&nbsp;→&nbsp;23&nbsp;s.
          </P>
          <P>
            <Term>Ultra cache</Term> forecasts the whole transformer's residual — video and audio together, so the joint
            trajectory stays coupled — for up to three consecutive steps. Faster, and the one preset where you should
            actually look at the output before trusting it.
          </P>
          <P>
            <Term>Turbo 4- and 8-step</Term> load a distilled LoRA and cut the schedule outright. They force Exact,
            because a few-step trajectory is too short for block reuse to be safe. The 4-step preset uses LightX2V's
            preview FL2V Turbo adapter; the 8-step preset retains the earlier adapter and its longer schedule.
          </P>
          <P className="text-faint">
            All of this is inference-work caching, not result caching. Every prompt and seed starts from fresh latents.
          </P>
        </Part>

        <Part title="Creative workflows">
          <P>
            <Term>Reference studio</Term> runs MiniMax-H3’s Ref2VA checkpoint with up to 12 ordered references: nine
            images, three videos and three audio files within that total. Reference order is preserved through both
            the local Qwen conditioner and the joint video/audio denoiser. It stays on the full 28-step quality path.
          </P>
          <P>
            <Term>Storyboard</Term> renders shots in order, captures each finished clip’s last frame locally, and uses
            it to anchor the next shot. The clips are joined on CPU with a one-frame seam trim, so assembly does not
            consume another ZeroGPU allocation. Shot recipes and source references remain in browser history.
          </P>
          <P>
            <Term>TAE live preview</Term> decodes a tiny animated approximation at denoising milestones. It adds a
            little work but gives useful visual feedback before the full video and audio VAEs finish; exported frames
            always come from the original full-precision VAE.
          </P>
        </Part>

        <Part title="Things that will surprise you">
          <P>
            <Term>Lengths snap.</Term> The video VAE decodes 17n + 5 frames at 24 fps and nothing else, so a request for
            5&nbsp;s becomes 124 frames — 5.17&nbsp;s. The slider shows the length you will actually get.
          </P>
          <P>
            <Term>Times are bookings, not durations.</Term> ZeroGPU reserves a worker for a computed number of seconds
            and charges your quota against the reservation, so "books ≈2m 52s" is what the run costs you, not how long
            you will wait. A lighter cache engine finishes sooner without booking less.
          </P>
          <P>
            <Term>Enhance leaves this worker.</Term> Rewriting a prompt needs the LM head and the 14 decoder layers the
            local checkpoint drops, so prompt enhancement is handled by a separate conditioner Space and adds a queue
            hop and roughly 20 seconds.
          </P>
          <P>
            <Term>Keyframes can override your canvas.</Term> An uploaded frame is centre-cropped to the nearest
            supported aspect ratio, and if that ratio fits the image better than the one you picked, it wins.
          </P>
        </Part>

        <Part title="Sources">
          <div className="flex flex-wrap gap-1.5">
            {LINKS.map(([label, href]) => (
              <Button key={href} variant="outline" size="sm" onClick={() => window.open(href, "_blank")}>
                {label} <ExternalLink className="size-3" />
              </Button>
            ))}
          </div>
          <P className="text-[11.5px] text-faint">
            Optimized from multimodalart's original Space. Model structure follows ComfyUI's implementation; block-cache
            and sparse-attention work is adapted from NVIDIA's Sol-Engine. H/t to Blanchon for the pointer. Weights are
            governed by the MiniMax-H3 Community License. Built by{" "}
            <a
              href="https://x.com/realmrfakename"
              target="_blank"
              rel="noreferrer"
              className="text-muted underline underline-offset-2 hover:text-ink"
            >
              @realmrfakename
            </a>
            .
          </P>
        </Part>
      </div>
    </Sheet>
  );
}

function Part({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="flex flex-col gap-2.5">
      <h3 className="text-[10.5px] font-semibold uppercase tracking-[0.1em] text-faint">{title}</h3>
      {children}
    </section>
  );
}

function P({ children, className }: { children: ReactNode; className?: string }) {
  return <p className={className}>{children}</p>;
}

/** An inline term, styled once so the doc emphasises names and modes consistently rather than by eye. */
function Term({ children }: { children: ReactNode }) {
  return <strong className="font-semibold text-ink">{children}</strong>;
}

/** A before/after figure. The numbers are the argument, so they get their own column instead of sitting in prose. */
function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-line py-1.5 last:border-0">
      <span className="text-[12.5px] text-muted">{label}</span>
      <span className="tabular text-[12.5px] font-medium text-ink">{value}</span>
    </div>
  );
}
