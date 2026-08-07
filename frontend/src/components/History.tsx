import { ChevronDown, Trash2 } from "lucide-react";
import { Collapsible } from "@base-ui/react/collapsible";
import { ScrollArea } from "@base-ui/react/scroll-area";
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
 * A filmstrip rather than a list: the thumbnail is the only part of a generated clip anyone recognises, and a row of
 * them fits under the player on a phone as readily as on a desktop. The tiles are a uniform 16:9 with `object-cover`
 * so a portrait clip does not make the strip twice as tall as its neighbours.
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
      <ScrollArea.Root>
        <ScrollArea.Viewport className="pb-2.5">
          <ScrollArea.Content className="flex gap-2">
            <AnimatePresence initial={false} mode="popLayout">
              {items.map((item) => (
                <motion.div
                  key={item.id}
                  layout
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  transition={SETTLE}
                  className="group/tile relative shrink-0"
                >
                  <button
                    type="button"
                    onClick={() => onSelect(item)}
                    aria-current={item.id === selectedId}
                    title={item.prompt}
                    className={cx(
                      "block h-16 w-28 overflow-hidden rounded-lg bg-black ring-1 ring-inset transition-shadow duration-150",
                      item.id === selectedId ? "ring-2 ring-accent" : "ring-line hover:ring-line-strong",
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

                  <span className="tabular pointer-events-none absolute bottom-1 left-1 rounded bg-black/75 px-1 text-[9.5px] font-medium text-ink backdrop-blur-sm">
                    {formatClock(item.seconds)} · {presetName(item.preset)}
                  </span>

                  {/* Always visible on touch, where there is no hover to reveal it. */}
                  <Tip label="Remove from history">
                    <motion.button
                      type="button"
                      onClick={() => onDelete(item)}
                      aria-label="Remove clip from history"
                      whileTap={{ scale: 0.9 }}
                      transition={POP}
                      className="absolute right-1 top-1 grid size-5 place-items-center rounded-md bg-black/75 text-muted backdrop-blur-sm transition-[color,opacity] duration-100 hover:text-bad focus-visible:opacity-100 sm:opacity-0 sm:group-hover/tile:opacity-100"
                    >
                      <Trash2 className="size-3" />
                    </motion.button>
                  </Tip>
                </motion.div>
              ))}
            </AnimatePresence>
          </ScrollArea.Content>
        </ScrollArea.Viewport>

        <ScrollArea.Scrollbar
          orientation="horizontal"
          className="flex h-1.5 touch-none rounded-full bg-line/60 opacity-0 transition-opacity duration-150 data-hovering:opacity-100 data-scrolling:opacity-100 data-scrolling:duration-0"
        >
          <ScrollArea.Thumb className="h-full rounded-full bg-line-strong" />
        </ScrollArea.Scrollbar>
      </ScrollArea.Root>
      </Collapsible.Panel>
    </Collapsible.Root>
  );
}
