import { useEffect, useRef } from "react";
import { Clapperboard, Clock3, Gauge, Hash, ImagePlus, Loader2, Ratio, Sparkles, Zap } from "lucide-react";
import { cx } from "../lib/cx";
import { findCanvas, findPreset, formatBudget, formatClock, presetName, snapFrames, FPS } from "../lib/studio";
import type { GenerationValues, StudioConfig } from "../types";
import { Button } from "../ui/Button";
import { ControlPill } from "./ControlPill";
import { budgetFor, FormatPanel, FramesPanel, LengthPanel, SeedPanel, SpeedPanel } from "./panels";

type Props = {
  config: StudioConfig;
  values: GenerationValues;
  update: <K extends keyof GenerationValues>(key: K, value: GenerationValues[K]) => void;
  onApplyExample: (prompt: string, canvas: string) => void;
  onGenerate: () => void;
  running: boolean;
  /** Non-null when generating is impossible right now — shown in place of the GPU estimate. */
  blockedReason: string | null;
};

export function Composer({ config, values, update, onApplyExample, onGenerate, running, blockedReason }: Props) {
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const canvas = findCanvas(config, values.canvas);
  const preset = findPreset(config, values.preset);
  const frames = snapFrames(values.duration);
  const keyframes = Number(values.image != null) + Number(values.lastImage != null);
  const steps = preset.custom ? values.steps : preset.steps;
  const budget = budgetFor(config, values, steps);
  const ready = values.prompt.trim().length > 0 && !running && !blockedReason;

  // Grow the prompt with its content instead of scrolling inside a fixed box, up to a point where the page takes over.
  useEffect(() => {
    const element = promptRef.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 260)}px`;
  }, [values.prompt]);

  return (
    <section aria-label="Prompt and settings" className="flex flex-col gap-2.5">
      <div className="rounded-2xl border border-line bg-surface transition-colors duration-150 focus-within:border-line-strong">
        <textarea
          ref={promptRef}
          rows={1}
          value={values.prompt}
          aria-label="Describe the scene"
          placeholder="Describe the scene — the action, the camera, the mood, the sound…"
          onChange={(event) => update("prompt", event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && (event.metaKey || event.ctrlKey) && ready) {
              event.preventDefault();
              onGenerate();
            }
          }}
          className="min-h-[74px] w-full resize-none bg-transparent px-4 pt-3.5 text-[15px] leading-[1.55] text-ink placeholder:text-faint focus:outline-none"
        />

        <div className="flex items-center gap-1.5 px-2.5 pb-2.5 pt-1">
          <ControlPill
            label="Frames"
            value={keyframes === 0 ? "None" : keyframes === 1 ? "1 image" : "2 images"}
            icon={<ImagePlus className="size-3.5" />}
            active={keyframes > 0}
            width="sm:w-[22rem]"
          >
            {() => <FramesPanel values={values} update={update} />}
          </ControlPill>

          <button
            type="button"
            aria-pressed={values.upsample}
            title="Rewrites your prompt with a larger remote model before conditioning. Adds a queue hop and about 20 seconds."
            onClick={() => update("upsample", !values.upsample)}
            className={cx(
              "flex h-9 shrink-0 items-center gap-1.5 rounded-lg border px-2.5 text-[13px] font-medium transition-colors duration-100",
              values.upsample
                ? "border-accent/50 bg-accent/12 text-accent"
                : "border-line bg-surface text-muted hover:border-line-strong hover:bg-raised hover:text-ink",
            )}
          >
            <Sparkles className="size-3.5" />
            <span className="hidden sm:inline">Enhance</span>
          </button>

          <div className="min-w-0 flex-1" />

          <span
            className={cx(
              "tabular hidden shrink-0 px-1 text-right text-[11px] leading-tight sm:block",
              blockedReason ? "text-warn" : "text-faint",
            )}
          >
            {blockedReason ?? (
              <>
                books ≈{formatBudget(budget)}
                <br />
                of GPU
              </>
            )}
          </span>

          <Button variant="primary" size="lg" disabled={!ready} onClick={onGenerate} className="shrink-0">
            {running ? <Loader2 className="size-4 animate-spin" /> : <Zap className="size-4" fill="currentColor" />}
            {running ? "Generating…" : "Generate"}
          </Button>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        <ControlPill
          label="Speed"
          value={presetName(values.preset)}
          icon={<Gauge className="size-3.5" />}
          active={values.preset !== config.default_preset}
          width="sm:w-[27rem]"
        >
          {() => <SpeedPanel config={config} values={values} update={update} />}
        </ControlPill>

        <ControlPill
          label="Format"
          value={`${canvas.width}×${canvas.height}`}
          icon={<Ratio className="size-3.5" />}
          active={values.canvas !== config.default_canvas}
          width="sm:w-[24rem]"
        >
          {() => <FormatPanel config={config} values={values} update={update} />}
        </ControlPill>

        <ControlPill
          label="Length"
          value={formatClock(frames / FPS)}
          icon={<Clock3 className="size-3.5" />}
          active={values.duration !== config.duration.default}
          width="sm:w-[22rem]"
        >
          {() => <LengthPanel config={config} values={values} update={update} />}
        </ControlPill>

        <ControlPill
          label="Seed"
          value={String(values.seed)}
          icon={<Hash className="size-3.5" />}
          align="end"
          width="sm:w-[21rem]"
        >
          {() => <SeedPanel values={values} update={update} />}
        </ControlPill>
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        <span className="inline-flex items-center gap-1.5 pr-0.5 text-[11px] uppercase tracking-[0.06em] text-faint">
          <Clapperboard className="size-3.5" /> Try
        </span>
        {config.examples.map((example) => (
          <button
            key={example.title}
            type="button"
            onClick={() => onApplyExample(example.prompt, example.canvas)}
            className="h-7 rounded-md border border-line px-2 text-[12px] text-muted transition-colors duration-100 hover:border-line-strong hover:bg-raised hover:text-ink"
          >
            {example.title}
          </button>
        ))}
      </div>
    </section>
  );
}
