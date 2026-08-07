import { Braces, Info, Zap } from "lucide-react";
import { cx } from "../lib/cx";
import type { ModelStatus } from "../types";
import { Button } from "../ui/Button";
import { Popover } from "../ui/Popover";

type Props = {
  model: ModelStatus;
  onOpenUsage: () => void;
  onOpenAbout: () => void;
};

export function Header({ model, onOpenUsage, onOpenAbout }: Props) {
  const tone = !model.reachable ? "bad" : model.ready ? "ok" : "warn";
  const summary = !model.reachable ? "Offline" : model.ready ? "Ready" : "Warming up";

  return (
    <header className="sticky top-0 z-30 border-b border-line bg-canvas/85 backdrop-blur-md">
      <div className="mx-auto flex h-14 w-full max-w-5xl items-center gap-2 px-4">
        <span className="grid size-7 shrink-0 place-items-center rounded-lg bg-accent text-accent-ink">
          <Zap className="size-4" fill="currentColor" />
        </span>
        <span className="mr-auto truncate text-[14px] font-semibold tracking-tight">MiniMax-H3 Ultra Fast</span>

        <Popover
          panelLabel="Engine status"
          align="end"
          width="w-[22rem]"
          triggerClassName="rounded-lg"
          trigger={() => (
            <span className="flex h-9 items-center gap-2 rounded-lg px-2.5 text-[12.5px] text-muted hover:bg-raised hover:text-ink">
              <span
                className={cx(
                  "size-1.5 shrink-0 rounded-full",
                  tone === "ok" ? "bg-ok" : tone === "warn" ? "bg-warn" : "bg-bad",
                  !model.ready && model.reachable && "animate-pulse",
                )}
              />
              <span className="hidden sm:inline">{summary}</span>
            </span>
          )}
        >
          {() => (
            <div className="flex flex-col gap-2">
              <p className="text-[12px] font-medium text-ink">{summary}</p>
              <p className="break-words text-[11.5px] leading-[1.55] text-muted">{cleanStatus(model.status)}</p>
            </div>
          )}
        </Popover>

        <Button variant="ghost" size="sm" onClick={onOpenUsage}>
          <Braces className="size-4" />
          <span className="hidden sm:inline">API</span>
        </Button>
        <Button variant="ghost" size="sm" iconOnly onClick={onOpenAbout} aria-label="About this Space">
          <Info className="size-4" />
        </Button>
      </div>
    </header>
  );
}

/** The server writes its status in Markdown for the old Gradio label; strip the emphasis rather than render it. */
function cleanStatus(status: string): string {
  return status.replaceAll("**", "").replaceAll("`", "");
}
