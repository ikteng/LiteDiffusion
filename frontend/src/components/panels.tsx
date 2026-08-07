import { Dices } from "lucide-react";
import {
  estimateGpuSeconds,
  findCanvas,
  formatBudget,
  formatClock,
  groupCanvases,
  presetName,
  presetTagline,
  randomSeed,
  snapFrames,
  FPS,
} from "../lib/studio";
import { cx } from "../lib/cx";
import type { Acceleration, GenerationValues, LoraPreset, StudioConfig } from "../types";
import { Button } from "../ui/Button";
import { Field, Slider, TextInput } from "../ui/Controls";
import { OptionList } from "../ui/OptionList";
import { Segmented } from "../ui/Segmented";
import { KeyframeSlot } from "./KeyframeSlot";

type Update = <K extends keyof GenerationValues>(key: K, value: GenerationValues[K]) => void;

type PanelProps = {
  config: StudioConfig;
  values: GenerationValues;
  update: Update;
};

/** GPU seconds for the current request, optionally overriding the step count to price a preset the user has not picked. */
export function budgetFor(config: StudioConfig, values: GenerationValues, steps: number): number {
  const canvas = findCanvas(config, values.canvas);
  return estimateGpuSeconds({
    width: canvas.width,
    height: canvas.height,
    frames: snapFrames(values.duration),
    steps,
    keyframes: Number(values.image != null) + Number(values.lastImage != null),
    upsample: values.upsample,
  });
}

function sentence(text: string): string {
  if (!text) return "";
  return `${text[0].toUpperCase()}${text.slice(1)}${/[.!?]$/.test(text) ? "" : "."}`;
}

export function SpeedPanel({ config, values, update }: PanelProps) {
  const options = config.presets.map((preset) => ({
    value: preset.value,
    title: presetName(preset.value),
    // The tagline is the tail of an em-dashed label ("… — best overall"), so it needs a capital to open a sentence.
    description: preset.custom ? preset.description : `${sentence(presetTagline(preset.value))} ${preset.description}`,
    badge: preset.recommended ? "recommended" : undefined,
    meta: preset.custom ? undefined : formatBudget(budgetFor(config, values, preset.steps)),
  }));

  return (
    <div className="flex flex-col gap-3">
      <OptionList
        ariaLabel="Generation preset"
        value={values.preset}
        options={options}
        onChange={(value) => update("preset", value)}
        renderExtra={(value) =>
          config.presets.find((preset) => preset.value === value)?.custom ? (
            <CustomPresetControls values={values} update={update} />
          ) : null
        }
      />
      <p className="border-t border-line pt-2.5 text-[11px] leading-[1.5] text-faint">
        Times are the ZeroGPU allocation each run books, not a promise of wall-clock speed.
      </p>
    </div>
  );
}

function CustomPresetControls({ values, update }: { values: GenerationValues; update: Update }) {
  return (
    <>
      <Field label="Scheduler steps" value={`${values.steps} steps`}>
        <Slider
          ariaLabel="Scheduler steps"
          min={4}
          max={40}
          value={values.steps}
          onChange={(next) => update("steps", next)}
        />
      </Field>

      <Field label="Cache engine" hint="Balanced reuses residuals conservatively; Exact evaluates every step.">
        <Segmented<Acceleration>
          ariaLabel="Cache engine"
          value={values.acceleration}
          onChange={(next) => update("acceleration", next)}
          options={[
            { value: "Exact", label: "Exact" },
            { value: "Balanced", label: "Balanced" },
            { value: "Ultra Fast", label: "Ultra" },
          ]}
        />
      </Field>

      <Field label="Turbo LoRA">
        <Segmented<LoraPreset>
          ariaLabel="Turbo LoRA"
          value={values.loraPreset}
          onChange={(next) => update("loraPreset", next)}
          options={[
            { value: "None", label: "None" },
            { value: "Turbo · 8 steps", label: "8-step" },
            { value: "Turbo · 4 steps", label: "4-step" },
            { value: "Custom", label: "Custom" },
          ]}
        />
      </Field>

      {values.loraPreset === "Custom" && (
        <div className="flex flex-col gap-2.5">
          <Field label="LoRA repository">
            <TextInput
              placeholder="owner/repository"
              value={values.loraRepo}
              onChange={(event) => update("loraRepo", event.target.value)}
            />
          </Field>
          <Field label="LoRA file">
            <TextInput
              placeholder="adapter.safetensors"
              value={values.loraFilename}
              onChange={(event) => update("loraFilename", event.target.value)}
            />
          </Field>
        </div>
      )}

      {values.loraPreset !== "None" && (
        <Field label="LoRA strength" value={values.loraStrength.toFixed(2)}>
          <Slider
            ariaLabel="LoRA strength"
            min={0}
            max={2}
            step={0.05}
            value={values.loraStrength}
            onChange={(next) => update("loraStrength", next)}
          />
        </Field>
      )}
    </>
  );
}

export function FormatPanel({ config, values, update }: PanelProps) {
  const groups = groupCanvases(config.canvases);
  const canvas = findCanvas(config, values.canvas);
  const activeGroup = groups.find((group) => group.options.some((option) => option.label === canvas.label)) ?? groups[0];
  const tier = activeGroup.options.findIndex((option) => option.label === canvas.label);

  return (
    <div className="flex flex-col gap-3.5">
      <Field label="Aspect ratio" value={activeGroup.ratio}>
        <div role="radiogroup" aria-label="Aspect ratio" className="flex flex-wrap gap-1.5">
          {groups.map((group) => {
            const selected = group.ratio === activeGroup.ratio;
            const landscape = group.value >= 1;
            return (
              <button
                key={group.ratio}
                type="button"
                role="radio"
                aria-checked={selected}
                // Carry the size tier across ratios, so someone on "full" does not silently drop to "fast".
                onClick={() =>
                  update("canvas", group.options[Math.min(Math.max(tier, 0), group.options.length - 1)].label)
                }
                className={cx(
                  "flex min-w-[4.25rem] flex-1 flex-col items-center gap-1.5 rounded-lg border px-2 py-2 transition-colors duration-100",
                  selected
                    ? "border-accent/50 bg-accent/10 text-ink"
                    : "border-line bg-sunken text-muted hover:border-line-strong hover:text-ink",
                )}
              >
                {/* The box has to be square for the percentages below to describe the ratio they claim to. */}
                <span aria-hidden className="grid size-6 place-items-center">
                  <span
                    className={cx("block rounded-[3px] border", selected ? "border-accent" : "border-line-strong")}
                    style={
                      landscape
                        ? { width: "100%", height: `${100 / group.value}%` }
                        : { height: "100%", width: `${100 * group.value}%` }
                    }
                  />
                </span>
                <span className="tabular text-[11.5px] font-medium">{group.ratio}</span>
              </button>
            );
          })}
        </div>
      </Field>

      <Field label="Resolution" value={`${canvas.width} × ${canvas.height}`}>
        <Segmented
          ariaLabel="Resolution"
          value={canvas.label}
          onChange={(next) => update("canvas", next)}
          options={activeGroup.options.map((option) => ({
            value: option.label,
            // The short edge is the familiar name for a video size (544p, 768p) regardless of orientation.
            label: `${Math.min(option.width, option.height)}p`,
            hint: option.fast ? "fast" : undefined,
          }))}
        />
      </Field>

      <p className="text-[11px] leading-[1.5] text-faint">
        Attention cost grows with the square of the canvas, so a larger frame is much slower than a longer clip.
      </p>
    </div>
  );
}

export function LengthPanel({ config, values, update }: PanelProps) {
  const frames = snapFrames(values.duration);
  const seconds = frames / FPS;

  return (
    <div className="flex flex-col gap-3">
      <Field label="Clip length" value={`${formatClock(seconds)} · ${frames} frames`}>
        <Slider
          ariaLabel="Clip length in seconds"
          min={config.duration.min}
          max={config.duration.max}
          value={values.duration}
          onChange={(next) => update("duration", next)}
        />
      </Field>
      <p className="text-[11px] leading-[1.5] text-faint">
        The video decoder only accepts <span className="tabular">17n + 5</span> frames at {FPS} fps, so lengths snap to
        the next valid count — {values.duration} s becomes {seconds.toFixed(2)} s.
      </p>
    </div>
  );
}

export function SeedPanel({ values, update }: Omit<PanelProps, "config">) {
  return (
    <div className="flex flex-col gap-3">
      <Field label="Seed" hint="The same seed, prompt and settings reproduce the same clip.">
        <div className="flex gap-2">
          <TextInput
            type="number"
            inputMode="numeric"
            value={String(values.seed)}
            onChange={(event) => update("seed", Math.trunc(Number(event.target.value)) || 0)}
          />
          <Button variant="outline" onClick={() => update("seed", randomSeed())} className="shrink-0">
            <Dices className="size-4" /> Randomise
          </Button>
        </div>
      </Field>
    </div>
  );
}

export function FramesPanel({ values, update }: Omit<PanelProps, "config">) {
  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 gap-2">
        <KeyframeSlot
          label="First frame"
          hint="Start here"
          file={values.image}
          onChange={(file) => update("image", file)}
        />
        <KeyframeSlot
          label="Last frame"
          hint="End here"
          file={values.lastImage}
          onChange={(file) => update("lastImage", file)}
        />
      </div>
      <p className="text-[11px] leading-[1.5] text-faint">
        Optional. Give one frame to animate from it, or both to interpolate between them. Images are cover-cropped to
        the selected aspect ratio.
      </p>
    </div>
  );
}
