import { useState, type ReactNode } from "react";
import { Drawer } from "@base-ui/react/drawer";
import { Popover as Base } from "@base-ui/react/popover";
import { useMediaQuery } from "@base-ui/react/unstable-use-media-query";
import { AnimatePresence, motion } from "framer-motion";
import { cx } from "../lib/cx";
import { POP } from "../lib/motion";

type Props = {
  /** Announced on the panel, and shown as the title of the phone sheet. */
  panelLabel: string;
  trigger: (state: { open: boolean }) => ReactNode;
  triggerClassName?: string;
  children: (close: () => void) => ReactNode;
  align?: "start" | "end";
  /** Anchored-panel width. The phone sheet is always full-bleed. */
  width?: string;
  disabled?: boolean;
};

/**
 * One disclosure primitive for every inline control, in the two shapes the two form factors actually want.
 *
 * From `sm` up it is a Base UI Popover: anchored under its trigger, flipped and shifted by the positioner rather than
 * by measuring code of our own, and capped at the height the positioner reports as available.
 *
 * Below `sm` it is a Base UI Drawer, which is not merely a popover pinned to the bottom — it comes with the gesture a
 * bottom sheet is expected to have: drag it down to dismiss, with the backdrop fading in proportion to the drag. That
 * gesture is why the two branches are different components instead of one component with different classes.
 */
export function Popover({
  panelLabel,
  trigger,
  triggerClassName,
  children,
  align = "start",
  width = "w-[24rem]",
  disabled,
}: Props) {
  const anchored = useMediaQuery("(min-width: 640px)", { defaultMatches: true });
  const [open, setOpen] = useState(false);
  const close = () => setOpen(false);

  if (!anchored) {
    return (
      <Drawer.Root open={open} onOpenChange={setOpen}>
        <Drawer.Trigger className={triggerClassName} disabled={disabled}>
          {trigger({ open })}
        </Drawer.Trigger>
        <Drawer.Portal>
          <Drawer.Backdrop className={BACKDROP} />
          <Drawer.Viewport className="fixed inset-0 z-50 flex items-end justify-center">
            {/* The sheet bleeds 3rem past the bottom edge so an overscroll bounce reveals more sheet, not the page. */}
            <Drawer.Popup className={SHEET}>
              <div aria-hidden className="mx-auto mb-3 h-1 w-9 rounded-full bg-line-strong" />
              <Drawer.Content className="mx-auto w-full max-w-lg">
                <Drawer.Title className="mb-3 text-[13px] font-semibold tracking-tight text-ink">
                  {panelLabel}
                </Drawer.Title>
                {children(close)}
              </Drawer.Content>
            </Drawer.Popup>
          </Drawer.Viewport>
        </Drawer.Portal>
      </Drawer.Root>
    );
  }

  return (
    <Base.Root open={open} onOpenChange={setOpen}>
      <Base.Trigger className={triggerClassName} disabled={disabled}>
        {trigger({ open })}
      </Base.Trigger>
      {/* `keepMounted` hands the unmount to AnimatePresence, which is what lets the panel animate on the way out. */}
      <AnimatePresence>
        {open && (
          <Base.Portal keepMounted>
            {/* The extra top padding is the sticky header: a panel that flips upwards is allowed to fill the page,
                but not to sit on top of the one bar that is meant to stay visible. */}
            <Base.Positioner
              sideOffset={8}
              align={align}
              collisionPadding={{ top: 68, right: 12, bottom: 12, left: 12 }}
              className="z-50"
            >
              <Base.Popup
                aria-label={panelLabel}
                className={cx(
                  "scrollbar-slim max-h-[min(var(--available-height),34rem)] overflow-y-auto",
                  "rounded-xl border border-line bg-surface p-3 text-ink outline-none",
                  "shadow-[0_24px_60px_-15px_rgba(0,0,0,0.75)]",
                  // Scaling from the corner the panel is anchored by is why it reads as coming *out of* the pill.
                  "origin-[var(--transform-origin)]",
                  width,
                )}
                render={
                  <motion.div
                    initial={{ opacity: 0, scale: 0.96 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.96 }}
                    transition={POP}
                  />
                }
              >
                {children(close)}
              </Base.Popup>
            </Base.Positioner>
          </Base.Portal>
        )}
      </AnimatePresence>
    </Base.Root>
  );
}

/**
 * Drawer styling is CSS, not Framer Motion, and deliberately so: while you are dragging, the sheet has to track your
 * finger exactly. Base UI publishes that as `--drawer-swipe-movement-y` and `--drawer-swipe-progress`, and a JS
 * animation library laying its own transform on top would fight it.
 */
const BACKDROP = cx(
  "fixed inset-0 z-40 min-h-dvh bg-black backdrop-blur-[2px]",
  "opacity-[calc(0.65*(1-var(--drawer-swipe-progress)))]",
  "transition-opacity duration-[450ms] ease-[cubic-bezier(0.32,0.72,0,1)] data-swiping:duration-0",
  "data-starting-style:opacity-0 data-ending-style:opacity-0",
);

const SHEET = cx(
  "scrollbar-slim -mb-12 max-h-[calc(85dvh+3rem)] w-full touch-auto overflow-y-auto overscroll-contain",
  "rounded-t-2xl border-t border-line bg-surface px-4 pt-2.5 outline-none",
  "pb-[calc(1rem+env(safe-area-inset-bottom,0px)+3rem)]",
  "shadow-[0_-20px_60px_-15px_rgba(0,0,0,0.7)]",
  "[transform:translateY(var(--drawer-swipe-movement-y))]",
  "transition-transform duration-[450ms] ease-[cubic-bezier(0.32,0.72,0,1)] data-swiping:select-none",
  "data-starting-style:[transform:translateY(calc(100%-3rem+2px))]",
  "data-ending-style:[transform:translateY(calc(100%-3rem+2px))]",
  "data-ending-style:duration-[calc(var(--drawer-swipe-strength)*400ms)]",
);
