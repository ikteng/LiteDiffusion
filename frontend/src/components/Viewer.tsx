import { useEffect, useState } from "react";
import { MediaPlayer, MediaProvider } from "@vidstack/react";
import { DefaultVideoLayout, defaultLayoutIcons } from "@vidstack/react/player/layouts/default";
import "@vidstack/react/player/styles/base.css";
import "@vidstack/react/player/styles/default/theme.css";
import { AlertTriangle, Check, Clock3, Copy, CornerUpLeft, Download, Film, Sparkles, Volume2, X } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import { cx } from "../lib/cx";
import { FADE, SETTLE } from "../lib/motion";
import { formatBudget, formatClock, formatEta, parseReport, presetName } from "../lib/studio";
import type { HistoryItem, RunPhase, RunProgress } from "../types";
import { Button } from "../ui/Button";
import { Chip, Progress } from "../ui/Controls";
import { Tip } from "../ui/Tip";

/**
 * The run, as five honest steps.
 *
 * `api.ts` already resolves a phase from the Gradio status stream; naming the phases here is what turns an anonymous
 * bar into "the worker is up, it is on step 14 of 28".
 */
const PHASES: { key: RunPhase; label: string; blurb: string }[] = [
  { key: "queue", label: "Queued", blurb: "Waiting for a ZeroGPU worker. Keep this tab open." },
  { key: "gpu", label: "Starting", blurb: "Bringing the worker up and moving weights onto the card." },
  { key: "conditioning", label: "Reading prompt", blurb: "The local Qwen3-VL conditioner is encoding your prompt." },
  { key: "denoising", label: "Denoising", blurb: "Generating the picture and its soundtrack together." },
  { key: "finalizing", label: "Decoding", blurb: "Running both decoders and muxing the audio onto the frames." },
];

type Props = {
  item: HistoryItem | null;
  progress: RunProgress;
  running: boolean;
  error: string | null;
  onDismissError: () => void;
  onReusePrompt: (item: HistoryItem) => void;
  className?: string;
};

/**
 * The clip, and only the clip.
 *
 * The player is a fixed frame that the video letterboxes itself into with `object-contain`, rather than a box that
 * resizes to each clip's aspect ratio. That is what stops the whole right-hand pane from resizing every time you click
 * a 9:16 clip in a history of 16:9 ones.
 */
export function Viewer({ item, progress, running, error, onDismissError, onReusePrompt, className }: Props) {
  const showing = error ? "error" : running ? "run" : item ? "result" : "empty";

  return (
    <div className={cx("flex min-h-0 flex-col", className)}>
      {/* Two different sizing rules, because the pane means two different things. On a phone it is a block in a
          scrolling page, so it takes a share of the viewport. On a desktop it is the pane itself, so it takes
          whatever the column has left after the toolbar and the history strip. */}
      <div className="relative h-[46vh] min-h-[220px] overflow-hidden rounded-2xl bg-black ring-1 ring-inset ring-line lg:h-auto lg:flex-1">
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={showing === "result" ? `result-${item!.id}` : showing}
            initial={{ opacity: 0, scale: 0.99 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.99 }}
            transition={SETTLE}
            className="absolute inset-0"
          >
            {showing === "empty" && <EmptyState />}
            {showing === "error" && <ErrorState message={error!} onDismiss={onDismissError} />}
            {showing === "run" && <RunState progress={progress} />}
            {showing === "result" && item && <Player item={item} />}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* The toolbar keeps its slot whatever the player is showing, so the pane does not resize between states. */}
      <AnimatePresence initial={false} mode="popLayout">
        {item && !running && !error && (
          <motion.div
            key={item.id}
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={FADE}
            className="shrink-0"
          >
            <ResultToolbar item={item} onReusePrompt={onReusePrompt} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function Player({ item }: { item: HistoryItem }) {
  return (
    <MediaPlayer
      key={item.url}
      src={{ src: item.url, type: "video/mp4" }}
      title={item.prompt}
      autoPlay
      playsInline
      className="size-full bg-black text-white"
    >
      <MediaProvider className="size-full [&_video]:size-full [&_video]:object-contain" />
      <DefaultVideoLayout icons={defaultLayoutIcons} />
    </MediaPlayer>
  );
}

function EmptyState() {
  return (
    <div className="grid size-full place-items-center bg-sunken px-6 text-center">
      <div>
        <Film className="mx-auto size-6 text-faint" />
        <p className="mt-3 text-[14px] font-medium text-ink">Nothing generated yet</p>
        {/* No "on the left": the composer is above this pane on a phone, and copy that points the wrong way is
            worse than copy that points nowhere. */}
        <p className="mx-auto mt-1 max-w-xs text-[12.5px] leading-[1.55] text-muted">
          Describe a scene and press Generate. The video and its soundtrack come out of the same pass.
        </p>
      </div>
    </div>
  );
}

function RunState({ progress }: { progress: RunProgress }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!progress.etaCountsDown) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1_000);
    return () => window.clearInterval(timer);
  }, [progress.etaCountsDown]);
  const activeIndex = Math.max(
    0,
    PHASES.findIndex((phase) => phase.key === (progress.phase ?? "queue")),
  );
  const active = PHASES[activeIndex];
  const steps =
    progress.index != null && progress.length != null ? `step ${progress.index} of ${progress.length}` : null;
  const remaining =
    progress.eta == null
      ? undefined
      : Math.max(
          0,
          progress.eta - (progress.etaCountsDown ? (now - (progress.etaUpdatedAt ?? now)) / 1_000 : 0),
        );
  const eta = formatEta(remaining);
  const etaLabel =
    progress.etaSource === "history" && remaining != null
      ? progress.etaCountsDown
        ? `ETA ${eta ?? "finishing…"}`
        : `Run ETA ≈${formatBudget(remaining)}`
      : eta;

  return (
    <section
      aria-label="Generation progress"
      className="flex size-full flex-col items-center justify-center gap-6 bg-sunken px-6"
    >
      <div className="w-full max-w-sm">
        {/* A finished phase fills left-to-right rather than switching colour, so the row reads as a track being
            covered — which is what the five phases actually are. */}
        <div className="flex gap-1" aria-hidden>
          {PHASES.map((phase, index) => (
            <span key={phase.key} className="relative h-1 flex-1 overflow-hidden rounded-full bg-line">
              <motion.span
                className="absolute inset-y-0 left-0 rounded-full bg-accent"
                initial={false}
                animate={{ width: index < activeIndex ? "100%" : "0%" }}
                transition={SETTLE}
              />
              {index === activeIndex && <span className="sweep block h-full w-1/2 rounded-full bg-accent" />}
            </span>
          ))}
        </div>

        {/* Phase copy is the one thing on screen that changes on its own, so it gets a proper hand-off. */}
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={active.key}
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -5 }}
            transition={FADE}
          >
            <p className="mt-3.5 text-center text-[15px] font-medium text-ink">{active.label}</p>
            <p className="mt-1 text-center text-[12.5px] leading-[1.5] text-muted">{active.blurb}</p>
          </motion.div>
        </AnimatePresence>
      </div>

      <div className="w-full max-w-sm">
        <Progress value={progress.progress} label={progress.label} />
        <div className="tabular mt-2 flex justify-between gap-3 text-[11.5px] text-faint">
          <span className="truncate">
            {steps ??
              (progress.stage === "queued" && progress.position != null
                ? `${progress.position + 1} ahead in the queue`
                : progress.label)}
          </span>
          <span className="shrink-0">
            {etaLabel ?? (progress.progress != null ? `${Math.round(progress.progress * 100)}%` : "")}
            {progress.etaSource === "history" && progress.etaSamples
              ? ` · ${progress.etaSamples} run${progress.etaSamples === 1 ? "" : "s"}`
              : ""}
          </span>
        </div>
      </div>
    </section>
  );
}

function ErrorState({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  return (
    <div role="alert" className="grid size-full place-items-center bg-sunken px-6">
      <div className="max-w-md text-center">
        <AlertTriangle className="mx-auto size-6 text-bad" />
        <p className="mt-3 text-[14px] font-medium text-ink">That run did not finish</p>
        <p className="mt-1.5 break-words text-[12.5px] leading-[1.55] text-muted">{message}</p>
        <Button variant="outline" size="sm" onClick={onDismiss} className="mt-4">
          <X /> Dismiss
        </Button>
      </div>
    </div>
  );
}

function ResultToolbar({ item, onReusePrompt }: { item: HistoryItem; onReusePrompt: (item: HistoryItem) => void }) {
  const facts = item.report ? parseReport(item.report) : [];
  const [showPrompt, setShowPrompt] = useState(false);

  return (
    <div className="pt-3">
      <div className="flex flex-wrap items-center gap-2">
        <Chip tone="ok">
          <Check className="size-3" /> {item.canvas.width}×{item.canvas.height}
        </Chip>
        <Chip>
          <Volume2 className="size-3" /> {formatClock(item.seconds)} with sound
        </Chip>
        <Chip>
          <Clock3 className="size-3" /> ran in {formatBudget(item.runtimeSeconds)}
        </Chip>
        <Chip>{presetName(item.preset)}</Chip>

        <div className="flex-1" />

        {item.refinedPrompt && (
          <Button variant="ghost" size="sm" onClick={() => setShowPrompt((current) => !current)}>
            <Sparkles /> {showPrompt ? "Hide" : "Enhanced"}
          </Button>
        )}
        <Tip label="Put this prompt and seed back in the composer">
          <Button variant="ghost" size="sm" iconOnly aria-label="Reuse prompt" onClick={() => onReusePrompt(item)}>
            <CornerUpLeft />
          </Button>
        </Tip>
        <CopyButton text={item.report ? item.report.replaceAll("`", "") : ""} />
        <Button variant="outline" size="sm" onClick={() => downloadVideo(item.url)}>
          <Download /> Download
        </Button>
      </div>

      <AnimatePresence initial={false}>
        {showPrompt && item.refinedPrompt && (
          <motion.p
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={FADE}
            className="overflow-hidden text-[12.5px] leading-[1.6] text-muted"
          >
            <span className="mt-2.5 block rounded-xl bg-surface p-3 ring-1 ring-inset ring-line">
              {item.refinedPrompt}
            </span>
          </motion.p>
        )}
      </AnimatePresence>

      {facts.length > 0 && (
        <dl className="mt-2.5 flex flex-wrap gap-1.5" aria-label="Generation details">
          {facts.map((fact) => (
            <dd key={fact} className="tabular rounded-md bg-surface px-2 py-1 text-[11px] text-faint">
              {fact}
            </dd>
          ))}
        </dl>
      )}
    </div>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  if (!text) return null;
  return (
    <Tip label="Copy the generation report">
      <Button
        variant="ghost"
        size="sm"
        iconOnly
        aria-label="Copy details"
        onClick={async () => {
          await navigator.clipboard.writeText(text);
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1600);
        }}
      >
        {/* The tick swaps in place of the clipboard rather than next to it, so the row does not reflow on every copy. */}
        <span className="grid size-4 place-items-center">
          <AnimatePresence mode="wait" initial={false}>
            <motion.span
              key={copied ? "copied" : "idle"}
              initial={{ opacity: 0, scale: 0.6 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.6 }}
              transition={FADE}
            >
              {copied ? <Check /> : <Copy />}
            </motion.span>
          </AnimatePresence>
        </span>
      </Button>
    </Tip>
  );
}

function downloadVideo(url: string) {
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "minimax-h3.mp4";
  anchor.click();
}
