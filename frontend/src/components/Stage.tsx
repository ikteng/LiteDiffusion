import { useEffect, useRef, useState } from "react";
import { AlertTriangle, Check, Copy, Download, RotateCcw, Sparkles, Volume2 } from "lucide-react";
import { cx } from "../lib/cx";
import { formatEta, parseReport } from "../lib/studio";
import type { CanvasOption, GeneratedVideo, RunPhase, RunProgress } from "../types";
import { Button } from "../ui/Button";
import { Chip, Progress } from "../ui/Controls";

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
  video: GeneratedVideo | null;
  progress: RunProgress;
  error: string | null;
  canvas: CanvasOption;
  onDismissError: () => void;
};

export function Stage({ video, progress, error, canvas, onDismissError }: Props) {
  const running = progress.stage === "connecting" || progress.stage === "queued" || progress.stage === "generating";

  if (error) return <ErrorPanel message={error} onDismiss={onDismissError} />;
  if (running) return <RunPanel progress={progress} canvas={canvas} />;
  if (video) return <ResultPanel video={video} canvas={canvas} />;
  return null;
}

/**
 * Reserve the shape of the clip that is coming, so the layout does not jump when the video arrives.
 *
 * The matching `max-width` is what keeps a 9:16 clip in its own aspect ratio: without it the height cap would bind
 * while the width stayed at 100%, and a portrait video would sit in a wide letterboxed box.
 */
function stageStyle(canvas: CanvasOption) {
  const ratio = canvas.width / canvas.height;
  return {
    aspectRatio: `${canvas.width} / ${canvas.height}`,
    maxHeight: "56vh",
    maxWidth: `calc(56vh * ${ratio})`,
  };
}

function RunPanel({ progress, canvas }: { progress: RunProgress; canvas: CanvasOption }) {
  const activeIndex = Math.max(
    0,
    PHASES.findIndex((phase) => phase.key === (progress.phase ?? "queue")),
  );
  const active = PHASES[activeIndex];
  const steps =
    progress.index != null && progress.length != null ? `step ${progress.index} of ${progress.length}` : null;
  const eta = formatEta(progress.eta);

  return (
    <section aria-label="Generation progress" className="rise flex flex-col gap-3">
      <div
        style={stageStyle(canvas)}
        className="mx-auto flex w-full max-w-full flex-col items-center justify-center gap-5 rounded-2xl border border-line bg-sunken px-6"
      >
        <div className="w-full max-w-sm">
          <div className="flex gap-1" aria-hidden>
            {PHASES.map((phase, index) => (
              <span
                key={phase.key}
                className={cx(
                  "h-1 flex-1 overflow-hidden rounded-full",
                  index < activeIndex ? "bg-accent" : "bg-line",
                )}
              >
                {index === activeIndex && <span className="sweep block h-full w-1/2 rounded-full bg-accent" />}
              </span>
            ))}
          </div>
          <p className="mt-3 text-center text-[15px] font-medium text-ink">{active.label}</p>
          <p className="mt-1 text-center text-[12.5px] leading-[1.5] text-muted">{active.blurb}</p>
        </div>

        <div className="w-full max-w-sm">
          <Progress value={progress.progress} label={progress.label} />
          <div className="tabular mt-2 flex justify-between text-[11.5px] text-faint">
            <span className="truncate">
              {steps ?? (progress.stage === "queued" && progress.position != null
                ? `${progress.position + 1} ahead in the queue`
                : progress.label)}
            </span>
            <span>{eta ?? (progress.progress != null ? `${Math.round(progress.progress * 100)}%` : "")}</span>
          </div>
        </div>
      </div>
    </section>
  );
}

function ResultPanel({ video, canvas }: { video: GeneratedVideo; canvas: CanvasOption }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const facts = video.report ? parseReport(video.report) : [];
  const [showPrompt, setShowPrompt] = useState(false);

  // A fresh result replaces the element's src; make sure it actually restarts rather than holding the last frame.
  useEffect(() => {
    videoRef.current?.load();
  }, [video.url]);

  return (
    <section aria-label="Generated video" className="rise flex flex-col gap-3">
      <video
        ref={videoRef}
        src={video.url}
        controls
        autoPlay
        playsInline
        style={stageStyle(canvas)}
        className="mx-auto w-full rounded-2xl border border-line bg-black object-contain"
      />

      <div className="flex flex-wrap items-center gap-2">
        <Chip tone="ok">
          <Check className="size-3" /> Ready
        </Chip>
        <Chip>
          <Volume2 className="size-3" /> Sound included
        </Chip>
        <div className="flex-1" />
        {video.refinedPrompt && (
          <Button variant="ghost" size="sm" onClick={() => setShowPrompt((current) => !current)}>
            <Sparkles className="size-3.5" /> {showPrompt ? "Hide" : "Enhanced"} prompt
          </Button>
        )}
        <CopyButton text={video.report ? video.report.replaceAll("`", "") : ""} />
        <Button variant="outline" size="sm" onClick={() => downloadVideo(video.url)}>
          <Download className="size-3.5" /> Download
        </Button>
      </div>

      {showPrompt && video.refinedPrompt && (
        <p className="rise rounded-xl border border-line bg-surface p-3 text-[12.5px] leading-[1.6] text-muted">
          {video.refinedPrompt}
        </p>
      )}

      {facts.length > 0 && (
        <dl className="flex flex-wrap gap-1.5" aria-label="Generation details">
          {facts.map((fact) => (
            <dd key={fact} className="tabular rounded-md bg-surface px-2 py-1 text-[11px] text-faint">
              {fact}
            </dd>
          ))}
        </dl>
      )}
    </section>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  if (!text) return null;
  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={async () => {
        await navigator.clipboard.writeText(text);
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1600);
      }}
    >
      {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
      {copied ? "Copied" : "Details"}
    </Button>
  );
}

function downloadVideo(url: string) {
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "minimax-h3.mp4";
  anchor.click();
}

function ErrorPanel({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  return (
    <section
      role="alert"
      className="rise flex items-start gap-3 rounded-2xl border border-bad/40 bg-bad/8 px-4 py-3.5"
    >
      <AlertTriangle className="mt-0.5 size-4 shrink-0 text-bad" />
      <div className="min-w-0 flex-1">
        <p className="text-[13px] font-medium text-ink">That run did not finish</p>
        <p className="mt-1 break-words text-[12.5px] leading-[1.55] text-muted">{message}</p>
      </div>
      <Button variant="ghost" size="sm" onClick={onDismiss}>
        <RotateCcw className="size-3.5" /> Dismiss
      </Button>
    </section>
  );
}
