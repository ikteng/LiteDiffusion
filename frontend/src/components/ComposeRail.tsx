import { useEffect, useRef } from "react";
import { Gauge, Loader2, Sparkles, Zap } from "lucide-react";
import { Toggle } from "@base-ui/react/toggle";
import { AnimatePresence, motion } from "framer-motion";
import { cx } from "../lib/cx";
import { FADE, POP } from "../lib/motion";
import { findPreset, formatBudget } from "../lib/studio";
import type { GenerationValues, RuntimeEstimate, StudioConfig } from "../types";
import { Button } from "../ui/Button";
import { Section } from "../ui/Section";
import {
  budgetFor,
  fasterOption,
  FormatSettings,
  KeyframeSettings,
  LengthSettings,
  SeedSettings,
  SpeedSettings,
} from "./settings";

type Props = {
  config: StudioConfig;
  values: GenerationValues;
  update: <K extends keyof GenerationValues>(key: K, value: GenerationValues[K]) => void;
  onApplyExample: (prompt: string, canvas: string) => void;
  onGenerate: () => void;
  running: boolean;
  /** Non-null when generating is impossible right now — shown in place of the GPU estimate. */
  blockedReason: string | null;
  runtimeEstimate: RuntimeEstimate | null;
};

/**
 * Everything that goes into a shot, in one scrollable column.
 *
 * Nothing here is behind a popover. The rail is a form: the settings in the order you would actually revisit them,
 * each showing its current value without being opened. On a wide window the rail scrolls independently of the viewer
 * so the clip you are looking at stays put while you change what comes next.
 *
 * Speed comes first and is the one accented section, because it is the only setting that decides how long you wait —
 * a four-minute booking and a one-minute booking are the same three keystrokes apart, and someone who never scrolls
 * past the prompt would never find that out. It is also the only setting that persists between visits, which is what
 * makes putting it at the top cheap rather than nagging: set it once and the rail opens on your answer.
 */
export function ComposeRail({
  config,
  values,
  update,
  onApplyExample,
  onGenerate,
  running,
  blockedReason,
  runtimeEstimate,
}: Props) {
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const preset = findPreset(config, values.preset);
  const steps = preset.custom ? values.steps : preset.steps;
  const budget = budgetFor(config, values, steps);
  const faster = fasterOption(config, values);
  const ready = values.prompt.trim().length > 0 && !running && !blockedReason;

  // Grow the prompt with its content instead of scrolling inside a fixed box, up to a point where the rail takes over.
  useEffect(() => {
    const element = promptRef.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 220)}px`;
  }, [values.prompt]);

  return (
    <div className="flex min-h-0 flex-col lg:h-full">
      <div className="scrollbar-slim divide-y divide-line lg:min-h-0 lg:flex-1 lg:overflow-y-auto lg:overscroll-contain">
        <Section
          title="Speed"
          tone="accent"
          action={
            // Only offered when it is worth taking, so it never becomes a button people learn to ignore. It names the
            // saving rather than the preset, because the saving is the thing being decided.
            <AnimatePresence initial={false} mode="popLayout">
              {faster && (
                <motion.div
                  key={faster.saves}
                  initial={{ opacity: 0, scale: 0.94 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.94 }}
                  transition={POP}
                >
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => update("preset", faster.preset)}
                    className="text-accent hover:bg-accent/12 hover:text-accent"
                  >
                    <Gauge /> Book {formatBudget(faster.saves)} less
                  </Button>
                </motion.div>
              )}
            </AnimatePresence>
          }
        >
          <SpeedSettings config={config} values={values} update={update} runtimeEstimate={runtimeEstimate} />
        </Section>

        <Section
          title="Prompt"
          action={
            <Toggle
              pressed={values.upsample}
              onPressedChange={(pressed) => update("upsample", pressed)}
              title="Rewrites your prompt with a larger remote model before conditioning. Adds a queue hop and about 20 seconds."
              className={cx(
                "flex h-7 shrink-0 items-center gap-1.5 rounded-lg px-2 text-[12px] font-medium ring-1 ring-inset",
                "transition-colors duration-100",
                "bg-sunken text-muted ring-line hover:text-ink hover:ring-line-strong",
                "data-pressed:bg-accent/12 data-pressed:text-accent data-pressed:ring-accent/45",
              )}
              render={<motion.button whileTap={{ scale: 0.96 }} transition={POP} />}
            >
              {/* The sparkle leans in when the rewrite is on — a state you would otherwise only read off the ring. */}
              <motion.span
                animate={{ rotate: values.upsample ? 0 : -12, scale: values.upsample ? 1 : 0.92 }}
                transition={POP}
                className="grid place-items-center"
              >
                <Sparkles className="size-3.5" />
              </motion.span>
              Enhance
            </Toggle>
          }
        >
          <textarea
            ref={promptRef}
            rows={3}
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
            className={cx(
              "min-h-[88px] w-full resize-none rounded-xl bg-sunken p-3 text-[13.5px] leading-[1.55] text-ink",
              "ring-1 ring-inset ring-line transition-[box-shadow,background-color] duration-100",
              "placeholder:text-faint hover:ring-line-strong focus:bg-canvas focus:ring-accent focus:outline-none",
            )}
          />

          <div className="mt-2.5 flex flex-wrap gap-1.5">
            {config.examples.map((example) => (
              <button
                key={example.title}
                type="button"
                onClick={() => onApplyExample(example.prompt, example.canvas)}
                className="h-7 rounded-lg px-2 text-[11.5px] text-muted ring-1 ring-inset ring-line transition-colors duration-100 hover:bg-raised hover:text-ink hover:ring-line-strong"
              >
                {example.title}
              </button>
            ))}
          </div>
        </Section>

        <Section title="Keyframes" defaultOpen={false}>
          <KeyframeSettings values={values} update={update} />
        </Section>

        <Section title="Format" defaultOpen={false}>
          <FormatSettings config={config} values={values} update={update} />
        </Section>

        <Section title="Length" defaultOpen={false}>
          <LengthSettings config={config} values={values} update={update} />
        </Section>

        <Section title="Seed" defaultOpen={false}>
          <SeedSettings values={values} update={update} />
        </Section>
      </div>

      {/* Sticky rather than merely last: on a phone the rail is the whole page, and Generate has to stay in reach of a
          thumb no matter how far down the settings you have scrolled. */}
      <div className="sticky bottom-0 z-20 flex items-center gap-3 border-t border-line bg-canvas/92 px-4 py-3 backdrop-blur-md">
        <div className="tabular min-w-0 flex-1 text-[11.5px] leading-tight">
          <AnimatePresence mode="popLayout" initial={false}>
            <motion.span
              key={blockedReason ?? `${runtimeEstimate?.seconds ?? "learning"}-${budget}`}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={FADE}
              className={cx("block truncate", blockedReason ? "text-warn" : "text-faint")}
            >
              {blockedReason ??
                (runtimeEstimate
                  ? `ETA ≈${formatBudget(runtimeEstimate.seconds)} from ${runtimeEstimate.samples} run${runtimeEstimate.samples === 1 ? "" : "s"} · GPU booking ${formatBudget(budget)}`
                  : `ETA learns after this preset's first run · GPU booking ${formatBudget(budget)}`)}
            </motion.span>
          </AnimatePresence>
        </div>

        <Button variant="primary" size="lg" disabled={!ready} onClick={onGenerate}>
          {running ? <Loader2 className="animate-spin" /> : <Zap fill="currentColor" />}
          {running ? "Generating…" : "Generate"}
        </Button>
      </div>
    </div>
  );
}
