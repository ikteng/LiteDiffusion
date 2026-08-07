import { ExternalLink, Heart } from "lucide-react";
import { Button } from "../ui/Button";
import { Sheet } from "../ui/Sheet";

const SPACE_URL = "https://huggingface.co/spaces/mrfakename/minimax-h3-faster";

const FACTS = [
  ["NVFP4", "A 12.5 GB pruned transformer replaces the 61.7 GiB BF16 one, with native Blackwell FP4 GEMMs."],
  ["One worker", "Conditioner, transformer and both decoders share a single GPU — no cross-Space round trip."],
  ["Real audio", "The soundtrack is generated jointly with the picture, not dubbed on afterwards."],
];

const LINKS = [
  ["Original Space", "https://huggingface.co/spaces/multimodalart/minimax-h3"],
  ["MiniMax-H3 model", "https://huggingface.co/MiniMaxAI/MiniMax-H3"],
  ["NVFP4 checkpoint", "https://huggingface.co/lilcheaty/MiniMax-H3-NVFP4"],
  ["Sana / Sol-Engine", "https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/optimized"],
];

export function AboutSheet({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <Sheet
      open={open}
      onClose={onClose}
      placement="center"
      title="MiniMax-H3 Ultra Fast"
      subtitle="Video and a synchronized soundtrack, generated together on one Blackwell ZeroGPU worker."
      footer={
        <div className="flex justify-end gap-2">
          <Button variant="ghost" onClick={onClose}>
            Close
          </Button>
          <Button variant="primary" onClick={() => window.open(SPACE_URL, "_blank")}>
            <Heart className="size-4" /> Like the Space
          </Button>
        </div>
      }
    >
      <div className="flex flex-col gap-4">
        <dl className="flex flex-col gap-2.5">
          {FACTS.map(([title, copy]) => (
            <div key={title} className="rounded-xl border border-line bg-surface p-3">
              <dt className="text-[12.5px] font-semibold text-accent">{title}</dt>
              <dd className="mt-1 text-[12px] leading-[1.6] text-muted">{copy}</dd>
            </div>
          ))}
        </dl>

        <div className="flex flex-wrap gap-1.5">
          {LINKS.map(([label, href]) => (
            <Button key={href} variant="outline" size="sm" onClick={() => window.open(href, "_blank")}>
              {label} <ExternalLink className="size-3" />
            </Button>
          ))}
        </div>

        <p className="text-[11.5px] leading-[1.6] text-faint">
          Optimized from multimodalart's original Space. H/t to Blanchon for pointing to Sana/Sol-Engine and Cache-DiT.
          Built by{" "}
          <a
            href="https://x.com/realmrfakename"
            target="_blank"
            rel="noreferrer"
            className="text-muted underline underline-offset-2 hover:text-ink"
          >
            @realmrfakename
          </a>
          .
        </p>
      </div>
    </Sheet>
  );
}
