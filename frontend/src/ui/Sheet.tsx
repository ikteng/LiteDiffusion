import type { ReactNode } from "react";
import { X } from "lucide-react";
import { Dialog } from "@base-ui/react/dialog";
import { Drawer } from "@base-ui/react/drawer";
import { AnimatePresence, motion } from "framer-motion";
import { cx } from "../lib/cx";
import { FADE, POP } from "../lib/motion";

type Props = {
  open: boolean;
  onClose: () => void;
  title: string;
  subtitle?: string;
  /** `side` docks to the right edge and can be swiped away; `center` is a classic modal. */
  placement?: "side" | "center";
  children: ReactNode;
  footer?: ReactNode;
};

/**
 * The full-page surfaces: API usage and About.
 *
 * `side` is a Drawer rather than a Dialog because a right-docked panel is the one shape people expect to be able to
 * throw back off the edge of the screen. `center` is a Dialog, because a modal that can be swiped anywhere is just
 * a modal you dismiss by accident.
 */
export function Sheet({ open, onClose, title, subtitle, placement = "side", children, footer }: Props) {
  const onOpenChange = (next: boolean) => {
    if (!next) onClose();
  };

  const head = (
    <header className="flex shrink-0 items-start justify-between gap-4 border-b border-line px-5 py-4">
      <div className="min-w-0">
        <h2 className="text-[15px] font-semibold tracking-tight">{title}</h2>
        {subtitle && <p className="mt-0.5 text-[12px] leading-[1.5] text-muted">{subtitle}</p>}
      </div>
      <CloseButton title={title} placement={placement} />
    </header>
  );

  const body = <div className="scrollbar-slim min-h-0 flex-1 overflow-y-auto px-5 py-5">{children}</div>;
  const foot = footer && <footer className="shrink-0 border-t border-line px-5 py-3">{footer}</footer>;

  if (placement === "side") {
    return (
      <Drawer.Root open={open} onOpenChange={onOpenChange} swipeDirection="right">
        <Drawer.Portal>
          <Drawer.Backdrop className={BACKDROP} />
          <Drawer.Viewport className="fixed inset-0 z-[60] flex items-stretch justify-end">
            {/* Bleeding 3rem past the right edge keeps an overscroll bounce inside the panel. */}
            <Drawer.Popup className={PANEL}>
              <Drawer.Title className="sr-only">{title}</Drawer.Title>
              <Drawer.Content className="flex min-h-0 w-full flex-1 flex-col">
                {head}
                {body}
                {foot}
              </Drawer.Content>
            </Drawer.Popup>
          </Drawer.Viewport>
        </Drawer.Portal>
      </Drawer.Root>
    );
  }

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <AnimatePresence>
        {open && (
          <Dialog.Portal keepMounted>
            <Dialog.Backdrop
              className="fixed inset-0 z-[60] bg-black/65 backdrop-blur-sm"
              render={<motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={FADE} />}
            />
            <Dialog.Viewport className="fixed inset-0 z-[60] grid place-items-center p-4">
              <Dialog.Popup
                className="flex max-h-[88vh] w-full max-w-lg flex-col overflow-hidden rounded-2xl border border-line bg-canvas outline-none"
                render={
                  <motion.div
                    initial={{ opacity: 0, scale: 0.97, y: 8 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.97, y: 8 }}
                    transition={POP}
                  />
                }
              >
                <Dialog.Title className="sr-only">{title}</Dialog.Title>
                {head}
                {body}
                {foot}
              </Dialog.Popup>
            </Dialog.Viewport>
          </Dialog.Portal>
        )}
      </AnimatePresence>
    </Dialog.Root>
  );
}

/** The two placements close through different components, so the button has to know which one it is inside. */
function CloseButton({ title, placement }: { title: string; placement: "side" | "center" }) {
  const className =
    "grid size-8 shrink-0 place-items-center rounded-lg text-muted transition-colors duration-100 hover:bg-raised hover:text-ink";
  const label = `Close ${title}`;
  const icon = <X className="size-4" />;
  return placement === "side" ? (
    <Drawer.Close aria-label={label} className={className}>
      {icon}
    </Drawer.Close>
  ) : (
    <Dialog.Close aria-label={label} className={className}>
      {icon}
    </Dialog.Close>
  );
}

const BACKDROP = cx(
  "fixed inset-0 z-[60] min-h-dvh bg-black backdrop-blur-sm",
  "opacity-[calc(0.65*(1-var(--drawer-swipe-progress)))]",
  "transition-opacity duration-[450ms] ease-[cubic-bezier(0.32,0.72,0,1)] data-swiping:duration-0",
  "data-starting-style:opacity-0 data-ending-style:opacity-0",
);

const PANEL = cx(
  "-mr-12 flex h-full w-[calc(100%+3rem)] flex-col border-l border-line bg-canvas pr-12 outline-none",
  "sm:w-[calc(36rem+3rem)] sm:max-w-[calc(100vw-2rem+3rem)]",
  "shadow-[-24px_0_60px_-15px_rgba(0,0,0,0.75)]",
  "[transform:translateX(var(--drawer-swipe-movement-x))]",
  "transition-transform duration-[450ms] ease-[cubic-bezier(0.32,0.72,0,1)] data-swiping:select-none",
  "data-starting-style:[transform:translateX(calc(100%-3rem+2px))]",
  "data-ending-style:[transform:translateX(calc(100%-3rem+2px))]",
  "data-ending-style:duration-[calc(var(--drawer-swipe-strength)*400ms)]",
);
