import { ChevronDown, Trash2 } from "lucide-react";
import { Collapsible } from "@base-ui/react/collapsible";
import { AnimatePresence, motion } from "framer-motion";
import { cx } from "../lib/cx";
import { POP, SETTLE } from "../lib/motion";
import { formatClock, presetName } from "../lib/studio";
import type { HistoryItem } from "../types";
import { Tip } from "../ui/Tip";

type Props = {
  items: HistoryItem[];
  selectedId: string | null;
  onSelect: (item: HistoryItem) => void;
  onDelete: (item: HistoryItem) => void;
};

/**
 * Every clip saved in this browser, newest first.
 *
 * A compact project grid rather than a tiny filmstrip: clips are large enough to recognise and the newest projects
 * remain directly below the active player. The thumbnails stay 16:9 and crop portrait clips so the grid remains calm.
 */
export function History({ items, selectedId, onSelect, onDelete }: Props) {
  if (items.length === 0) return null;

  return (
    <Collapsible.Root defaultOpen render={<section aria-label="Saved clips" className="mt-4 shrink-0" />}>
      <div className="mb-2 flex items-baseline gap-2">
        <Collapsible.Trigger className="group flex items-center gap-1 text-[10.5px] font-semibold uppercase tracking-[0.1em] text-faint">
          <ChevronDown className="size-3 transition-transform duration-200 group-data-panel-open:rotate-180" />
          History
        </Collapsible.Trigger>
        <span className="tabular text-[11px] text-faint">
          {items.length} saved clip{items.length === 1 ? "" : "s"}
        </span>
      </div>

      <Collapsible.Panel className="h-[var(--collapsible-panel-height)] overflow-hidden transition-[height] duration-200 ease-out data-starting-style:h-0 data-ending-style:h-0">
        <div className="scrollbar-slim max-h-64 overflow-y-auto pr-1 pb-1">
          <div className="grid grid-cols-[repeat(auto-fill,minmax(11rem,1fr))] gap-2.5">
            <AnimatePresence initial={false} mode="popLayout">
              {items.map((item) => (
                <motion.div
                  key={item.id}
                  layout
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  transition={SETTLE}
                  className="group/tile relative min-w-0 overflow-hidden rounded-xl bg-surface ring-1 ring-inset ring-line"
                >
                  <button
                    type="button"
                    onClick={() => onSelect(item)}
                    aria-current={item.id === selectedId}
                    title={item.prompt}
                    className={cx(
                      "block aspect-video w-full overflow-hidden rounded-t-xl bg-black ring-1 ring-inset transition-shadow duration-150",
                      item.id === selectedId ? "ring-2 ring-accent" : "ring-transparent hover:ring-line-strong",
                    )}
                  >
                    {/* `#t=0.1` asks for a frame rather than a black poster; `preload="metadata"` keeps a long
                        history from downloading every clip again just to draw the strip. */}
                    <video
                      src={`${item.url}#t=0.1`}
                      muted
                      playsInline
                      preload="metadata"
                      tabIndex={-1}
                      className="pointer-events-none size-full object-cover"
                    />
                  </button>

                  <button type="button" onClick={() => onSelect(item)} className="block w-full px-2.5 py-2 text-left">
                    <span className="block truncate text-[11.5px] font-medium text-ink">{item.prompt}</span>
                    <span className="tabular mt-0.5 block text-[10.5px] text-faint">
                      {formatClock(item.seconds)} · {presetName(item.preset)} · seed {item.seed}
                    </span>
                  </button>

                  {/* Always visible on touch, where there is no hover to reveal it. */}
                  <Tip label="Remove from history">
                    <motion.button
                      type="button"
                      onClick={() => onDelete(item)}
                      aria-label="Remove clip from history"
                      whileTap={{ scale: 0.9 }}
                      transition={POP}
                      className="absolute right-1.5 top-1.5 grid size-7 place-items-center rounded-lg bg-black/75 text-muted backdrop-blur-sm transition-[color,opacity] duration-100 hover:text-bad focus-visible:opacity-100 sm:opacity-0 sm:group-hover/tile:opacity-100"
                    >
                      <Trash2 className="size-3" />
                    </motion.button>
                  </Tip>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>
      </Collapsible.Panel>
    </Collapsible.Root>
  );
}
