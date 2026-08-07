import { useRef, useState } from "react";
import { ChevronDown, Dices } from "lucide-react";
import { Collapsible } from "@base-ui/react/collapsible";
import { Radio } from "@base-ui/react/radio";
import { RadioGroup } from "@base-ui/react/radio-group";
import { AnimatePresence, motion } from "framer-motion";
import { FADE } from "../lib/motion";
import {
  estimateGpuSeconds,
  findCanvas,
  formatBudget,
  formatClock,
  groupCanvases,
  presetAxis,
  presetName,
  presetTagline,
  randomSeed,
  snapFrames,
  FPS,
} from "../lib/studio";
import { cx } from "../lib/cx";
import type { Acceleration, GenerationValues, LoraPreset, StudioConfig } from "../types";
import { Button } from "../ui/Button";
import { Field, Slider, Switch, TextInput } from "../ui/Controls";
import { NotchSlider } from "../ui/NotchSlider";
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

/**
 * The presets as one axis rather than a list of paragraphs.
 *
 * They are genuinely ordered — `presetAxis` derives that order from each preset's step count and cache engine — so a
 * slider says "a little faster than the default" in one gesture, where six radio rows made you read all six. What is
 * *not* on the axis lives under Advanced: manual control, which is an escape from the presets rather than a point
 * along them.
 */
export function SpeedPanel({ config, values, update }: PanelProps) {
  const axis = presetAxis(config.presets);
  const custom = config.presets.find((preset) => preset.custom);
  const manual = custom != null && values.preset === custom.value;
  const [advanced, setAdvanced] = useState(manual);

  // While manual controls are in effect the axis has no position, so hold the thumb at the last preset the user chose.
  const lastOnAxis = useRef(Math.max(0, axis.findIndex((preset) => preset.recommended)));
  const index = axis.findIndex((preset) => preset.value === values.preset);
  if (index >= 0) lastOnAxis.current = index;
  const shown = index >= 0 ? index : lastOnAxis.current;
  const current = axis[shown];

  const steps = manual ? values.steps : current.steps;
  const budget = formatBudget(budgetFor(config, values, steps));

  return (
    <div className="flex flex-col gap-3">
      <NotchSlider
        ariaLabel="Generation preset"
        minLabel="Faster"
        maxLabel="Smarter"
        stops={axis.map((preset) => presetName(preset.value))}
        index={shown}
        disabled={manual}
        onChange={(next) => update("preset", axis[next].value)}
      />

      {/* `layout` on the card is what keeps the popover from jolting when a longer description swaps in. */}
      <motion.div layout transition={FADE} className="overflow-hidden rounded-lg bg-sunken px-3 py-2.5">
        <p className="flex items-baseline gap-2">
          <span className="text-[13.5px] font-medium text-ink">{manual ? "Manual" : presetName(current.value)}</span>
          <AnimatePresence initial={false}>
            {!manual && current.recommended && (
              <motion.span
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={FADE}
                className="rounded bg-accent/15 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-[0.06em] text-accent"
              >
                recommended
              </motion.span>
            )}
          </AnimatePresence>
          <span className="tabular ml-auto shrink-0 text-[12px] text-muted">{budget}</span>
        </p>
        {/* In manual mode the controls themselves are two rows below, so restating them here would only add noise. */}
        {!manual && (
          <motion.p
            key={current.value}
            initial={{ opacity: 0, y: 3 }}
            animate={{ opacity: 1, y: 0 }}
            transition={FADE}
            className="mt-1 text-[12px] leading-[1.5] text-muted"
          >
            {sentence(presetTagline(current.value))} {current.description}
          </motion.p>
        )}
        <p className="tabular mt-1.5 text-[11px] text-faint">
          {steps} steps · {manual ? values.acceleration : current.acceleration} cache · books the GPU for {budget}
        </p>
      </motion.div>

      {custom && (
        // The height transition is CSS on `--collapsible-panel-height` rather than a JS `height: auto` animation:
        // Base UI has already measured the panel, and re-measuring it from JS on every frame inside a popover that is
        // simultaneously being repositioned is exactly how a disclosure ends up stuttering.
        <Collapsible.Root open={advanced} onOpenChange={setAdvanced} className="border-t border-line pt-2.5">
          <Collapsible.Trigger className="group flex w-full items-center gap-1.5 text-[12px] text-muted transition-colors duration-100 hover:text-ink">
            Advanced
            <ChevronDown className="size-3.5 transition-transform duration-200 group-data-panel-open:rotate-180" />
          </Collapsible.Trigger>

          <Collapsible.Panel
            className={cx(
              "h-[var(--collapsible-panel-height)] overflow-hidden",
              "transition-[height] duration-200 ease-out",
              "data-starting-style:h-0 data-ending-style:h-0",
            )}
          >
            <div className="mt-3 flex flex-col gap-3">
              <Switch
                checked={manual}
                onChange={(next) => update("preset", next ? custom.value : axis[lastOnAxis.current].value)}
                label="Manual controls"
                description="Set the schedule, cache engine and LoRA yourself instead of using a preset."
              />
              <AnimatePresence initial={false}>
                {manual && (
                  <motion.div
                    key="manual"
                    initial={{ opacity: 0, y: -6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -6 }}
                    transition={FADE}
                    className="flex flex-col gap-3"
                  >
                    <CustomPresetControls values={values} update={update} />
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </Collapsible.Panel>
        </Collapsible.Root>
      )}

      <p className="text-[11px] leading-[1.5] text-faint">
        Times are the ZeroGPU allocation each run books, not a promise of wall-clock speed — a lighter cache engine
        finishes sooner without booking less.
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
        <RadioGroup
          aria-label="Aspect ratio"
          value={activeGroup.ratio}
          // Carry the size tier across ratios, so someone on "full" does not silently drop to "fast".
          onValueChange={(next) => {
            const group = groups.find((candidate) => candidate.ratio === next);
            if (!group) return;
            update("canvas", group.options[Math.min(Math.max(tier, 0), group.options.length - 1)].label);
          }}
          className="flex flex-wrap gap-1.5"
        >
          {groups.map((group) => {
            const selected = group.ratio === activeGroup.ratio;
            const landscape = group.value >= 1;
            return (
              <Radio.Root
                key={group.ratio}
                value={group.ratio}
                className={cx(
                  "flex min-w-[4.25rem] flex-1 cursor-pointer flex-col items-center gap-1.5 rounded-lg border px-2 py-2",
                  "transition-colors duration-100",
                  selected
                    ? "border-accent/50 bg-accent/10 text-ink"
                    : "border-line bg-sunken text-muted hover:border-line-strong hover:text-ink",
                )}
              >
                {/* The box has to be square for the percentages below to describe the ratio they claim to. */}
                <span aria-hidden className="grid size-6 place-items-center">
                  <span
                    className={cx(
                      "block rounded-[3px] border transition-colors duration-100",
                      selected ? "border-accent" : "border-line-strong",
                    )}
                    style={
                      landscape
                        ? { width: "100%", height: `${100 / group.value}%` }
                        : { height: "100%", width: `${100 * group.value}%` }
                    }
                  />
                </span>
                <span className="tabular text-[11.5px] font-medium">{group.ratio}</span>
              </Radio.Root>
            );
          })}
        </RadioGroup>
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
          {/* Deliberately not `type="number"`: a nine-digit seed is never stepped one at a time, and the native
              spinner would take up a third of the field to offer exactly that. */}
          <TextInput
            inputMode="numeric"
            pattern="[0-9]*"
            aria-label="Seed"
            value={String(values.seed)}
            onChange={(event) => update("seed", Number(event.target.value.replace(/\D/g, "").slice(0, 10)) || 0)}
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
