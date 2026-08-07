import { useEffect, useId, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { X } from "lucide-react";
import { cx } from "../lib/cx";

type Props = {
  /** Announced on the panel, and shown as the title of the mobile sheet. */
  panelLabel: string;
  trigger: (state: { open: boolean }) => ReactNode;
  triggerClassName?: string;
  children: (close: () => void) => ReactNode;
  align?: "start" | "end";
  /** Desktop panel width utility. The mobile sheet is always full-bleed. */
  width?: string;
  disabled?: boolean;
};

/** Where the desktop panel ended up, and how tall it is allowed to be there. */
type Anchor = { side: "top" | "bottom"; maxHeight: number };

/** Breathing room at the viewport edge. */
const GUTTER = 12;
/** The gap between trigger and panel, which `sm:top-[calc(100%+8px)]` also encodes. */
const OFFSET = 8;
/** A panel shorter than this is not worth showing; scroll it and accept the overflow instead. */
const MIN_USEFUL = 200;
/** Matches the old `34rem` cap: past this a panel is tall enough that more height stops helping. */
const PREFERRED = 544;

/**
 * How much room the trigger leaves, and therefore which way the panel should open.
 *
 * A plain `max-height` is not enough: the composer sits mid-screen, so a panel that is merely *short* can still be
 * anchored low enough to run off the bottom. Measuring is the only way — CSS cannot see the trigger's position.
 * Opening downward is the default; we only flip when below is cramped *and* above is genuinely roomier, so a pill
 * near the top of the window never sends its panel off the top instead.
 */
function measure(trigger: HTMLElement): Anchor {
  const rect = trigger.getBoundingClientRect();
  const below = window.innerHeight - rect.bottom - OFFSET - GUTTER;
  const above = rect.top - OFFSET - GUTTER;
  const side = below >= PREFERRED || below >= above ? "bottom" : "top";
  return { side, maxHeight: Math.max(MIN_USEFUL, Math.min(PREFERRED, side === "top" ? above : below)) };
}

/**
 * A single disclosure primitive for every inline control.
 *
 * On phones it presents as a bottom sheet with a backdrop, because a 26rem panel anchored to a pill is unusable at
 * 380px wide. From `sm` up it is a panel anchored under its trigger. `position: fixed` is not clipped by an
 * `overflow` ancestor, so the same markup serves both — as long as no ancestor creates a containing block with
 * `transform`/`filter`.
 */
export function Popover({
  panelLabel,
  trigger,
  triggerClassName,
  children,
  align = "start",
  width = "sm:w-[24rem]",
  disabled,
}: Props) {
  const [open, setOpen] = useState(false);
  const [anchor, setAnchor] = useState<Anchor | null>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const panelId = useId();

  function close() {
    setOpen(false);
    triggerRef.current?.focus();
  }

  // Below `sm` the panel is a fixed bottom sheet, which needs no measuring — hence the media query.
  useLayoutEffect(() => {
    if (!open) return;
    function place() {
      const trigger = triggerRef.current;
      setAnchor(trigger && window.matchMedia("(min-width: 640px)").matches ? measure(trigger) : null);
    }
    place();
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => {
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;

    function onPointerDown(event: MouseEvent | TouchEvent) {
      const target = event.target as Node;
      if (panelRef.current?.contains(target) || triggerRef.current?.contains(target)) return;
      setOpen(false);
    }
    function onKeyDown(event: KeyboardEvent) {
      if (event.key !== "Escape") return;
      event.stopPropagation();
      close();
    }

    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("touchstart", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("touchstart", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  return (
    <div className="relative">
      <button
        ref={triggerRef}
        type="button"
        disabled={disabled}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls={open ? panelId : undefined}
        onClick={() => setOpen((current) => !current)}
        className={triggerClassName}
      >
        {trigger({ open })}
      </button>

      {open && (
        <>
          <div
            className="fixed inset-0 z-40 bg-black/60 backdrop-blur-[2px] sm:hidden"
            onClick={() => setOpen(false)}
            aria-hidden
          />
          <div
            ref={panelRef}
            id={panelId}
            role="dialog"
            aria-label={panelLabel}
            style={anchor ? { maxHeight: anchor.maxHeight } : undefined}
            className={cx(
              "rise scrollbar-slim fixed inset-x-0 bottom-0 z-50 max-h-[82vh] overflow-y-auto",
              "rounded-t-2xl border-t border-line bg-surface p-4 shadow-[0_-20px_60px_-15px_rgba(0,0,0,0.7)]",
              "sm:absolute sm:inset-x-auto sm:max-h-[min(70vh,34rem)]",
              anchor?.side === "top"
                ? "sm:bottom-[calc(100%+8px)] sm:top-auto"
                : "sm:bottom-auto sm:top-[calc(100%+8px)]",
              "sm:rounded-xl sm:border sm:p-3 sm:shadow-[0_24px_60px_-15px_rgba(0,0,0,0.75)]",
              align === "end" ? "sm:right-0" : "sm:left-0",
              width,
            )}
          >
            <div className="mb-3 flex items-center justify-between sm:hidden">
              <h2 className="text-sm font-semibold">{panelLabel}</h2>
              <button
                type="button"
                onClick={close}
                aria-label={`Close ${panelLabel}`}
                className="grid size-8 place-items-center rounded-lg text-muted hover:bg-raised hover:text-ink"
              >
                <X className="size-4" />
              </button>
            </div>
            {children(close)}
          </div>
        </>
      )}
    </div>
  );
}
