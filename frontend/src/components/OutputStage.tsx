import { Button, Card, Chip, Disclosure, Spinner } from "@heroui/react";
import { ChevronDown, Download, Film, RotateCcw, Sparkles, Volume2 } from "lucide-react";
import type { GeneratedVideo, RunProgress } from "../types";

type Props = {
  video: GeneratedVideo | null;
  progress: RunProgress;
  error: string | null;
  onReset: () => void;
};

function formatEta(seconds?: number) {
  if (!seconds || !Number.isFinite(seconds)) return null;
  return seconds < 60 ? `~${Math.ceil(seconds)} sec` : `~${Math.ceil(seconds / 60)} min`;
}

function progressDetail(progress: RunProgress) {
  if (progress.stage === "queued") return "In queue";
  if (progress.index != null && progress.length != null) {
    return `${progress.index} / ${progress.length}${progress.unit ? ` ${progress.unit}` : ""}`;
  }
  if (progress.progress != null && progress.exact) return `${Math.round(progress.progress * 100)}%`;
  return progress.label;
}

export function OutputStage({ video, progress, error, onReset }: Props) {
  const running = ["connecting", "queued", "generating"].includes(progress.stage);

  return (
    <section className="sticky top-20" aria-label="Generation output">
      <Card className="min-h-[560px] overflow-hidden xl:min-h-[calc(100vh-128px)]" variant="secondary">
        <Card.Content className="flex min-h-[560px] p-2 xl:min-h-[calc(100vh-128px)]">
          {running ? (
            <div className="flex flex-1 flex-col items-center justify-center rounded-xl bg-background px-8 text-center">
              <div className="mb-6 grid size-16 place-items-center rounded-full border border-divider bg-default/5">
                <Spinner size="lg" color="accent" />
              </div>
              <p className="text-xs font-medium uppercase tracking-[0.14em] text-accent">Creating your video</p>
              <h2 className="mt-2 text-2xl font-semibold tracking-tight">{progress.label}</h2>
              <p className="mt-2 max-w-md text-sm leading-6 text-muted">
                {progress.stage === "queued"
                  ? "A ZeroGPU worker will pick this up shortly. Keep this tab open."
                  : "MiniMax-H3 is generating the visuals and soundtrack together."}
              </p>
              <div className="mt-7 h-1 w-full max-w-md overflow-hidden rounded-full bg-default/10">
                <span
                  role="progressbar"
                  aria-label={progress.label}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={progress.progress == null ? undefined : Math.round(progress.progress * 100)}
                  className={`block h-full rounded-full bg-accent transition-all ${progress.progress == null ? "w-1/3 animate-pulse" : ""}`}
                  style={progress.progress == null ? undefined : { width: `${Math.max(4, progress.progress * 100)}%` }}
                />
              </div>
              <div className="mt-2 flex w-full max-w-md justify-between text-xs text-muted">
                <span className="max-w-[70%] truncate" title={progressDetail(progress)}>{progressDetail(progress)}</span>
                <span>{formatEta(progress.eta) ?? (progress.progress != null ? `${Math.round(progress.progress * 100)}%` : "Live progress")}</span>
              </div>
            </div>
          ) : video ? (
            <div className="flex min-w-0 flex-1 flex-col">
              <video className="min-h-0 flex-1 rounded-xl bg-black object-contain" src={video.url} controls autoPlay playsInline />
              <div className="flex flex-wrap items-center justify-between gap-3 px-2 py-3">
                <div className="flex items-center gap-2">
                  <Chip size="sm" variant="soft" color="success">Complete</Chip>
                  <span className="flex items-center gap-1 text-xs text-muted"><Volume2 className="size-3.5" /> Native audio</span>
                </div>
                <div className="flex gap-1.5">
                  <Button variant="ghost" size="sm" onPress={onReset}><RotateCcw className="size-4" /> New</Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    onPress={() => {
                      const anchor = document.createElement("a");
                      anchor.href = video.url;
                      anchor.download = "minimax-h3.mp4";
                      anchor.click();
                    }}
                  >
                    <Download className="size-4" /> Download
                  </Button>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex flex-1 flex-col items-center justify-center rounded-xl bg-background px-8 text-center">
              <div className="mb-6 grid size-20 place-items-center rounded-2xl border border-divider bg-default/5 text-muted">
                <Film className="size-8" strokeWidth={1.4} />
              </div>
              <Chip size="sm" variant="soft">Preview</Chip>
              <h2 className={`mt-3 text-2xl font-semibold tracking-tight ${error ? "text-danger" : ""}`}>
                {error ? "Generation stopped" : "Your video will appear here"}
              </h2>
              <p className="mt-2 max-w-md text-sm leading-6 text-muted">
                {error ?? "Write a prompt, choose a mode, and generate video with a synchronized soundtrack."}
              </p>
              {error ? (
                <Button className="mt-5" variant="secondary" onPress={onReset}>Try again</Button>
              ) : (
                <div className="mt-5 flex gap-2">
                  <Chip size="sm" variant="soft"><Sparkles className="size-3" /> Fresh generation</Chip>
                  <Chip size="sm" variant="soft"><Volume2 className="size-3" /> Native audio</Chip>
                </div>
              )}
            </div>
          )}
        </Card.Content>
      </Card>

      {video?.refinedPrompt && (
        <Disclosure className="mt-2 rounded-xl border border-divider bg-surface-secondary">
          <Disclosure.Heading>
            <Disclosure.Trigger className="flex w-full items-center justify-between p-4 text-sm">
              <span className="flex items-center gap-2"><Sparkles className="size-4" /> Enhanced prompt</span>
              <Disclosure.Indicator><ChevronDown className="size-4" /></Disclosure.Indicator>
            </Disclosure.Trigger>
          </Disclosure.Heading>
          <Disclosure.Content><Disclosure.Body className="px-4 pb-4 text-sm leading-6 text-muted">{video.refinedPrompt}</Disclosure.Body></Disclosure.Content>
        </Disclosure>
      )}

      {video?.report && (
        <Disclosure className="mt-2 rounded-xl border border-divider bg-surface-secondary">
          <Disclosure.Heading>
            <Disclosure.Trigger className="flex w-full items-center justify-between p-4 text-sm">
              <span>Generation details</span><Disclosure.Indicator><ChevronDown className="size-4" /></Disclosure.Indicator>
            </Disclosure.Trigger>
          </Disclosure.Heading>
          <Disclosure.Content><Disclosure.Body className="px-4 pb-4 font-mono text-xs leading-5 text-muted">{video.report.replaceAll("`", "")}</Disclosure.Body></Disclosure.Content>
        </Disclosure>
      )}
    </section>
  );
}
