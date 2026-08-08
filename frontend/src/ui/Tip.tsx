import type { ReactElement } from "react";
import { Tooltip } from "@base-ui/react/tooltip";

/**
 * A tooltip for controls that are only an icon.
 *
 * The transition is CSS on Base UI's `data-starting-style` / `data-ending-style` rather than Framer Motion: tooltips
 * open and close constantly as the pointer crosses a toolbar, and a spring that has to mount a motion component each
 * time is both heavier and less crisp than the two-property transition below.
 *
 * Requires `<Tooltip.Provider>` at the root — that shared provider is what makes the second tooltip in a row appear
 * instantly instead of waiting out the delay again.
 */
export function Tip({ label, children, side = "top" }: { label: string; children: ReactElement; side?: "top" | "bottom" | "left" | "right" }) {
  return (
    <Tooltip.Root>
      <Tooltip.Trigger render={children} />
      <Tooltip.Portal>
        <Tooltip.Positioner side={side} sideOffset={7} collisionPadding={8} className="z-60">
          <Tooltip.Popup className="origin-[var(--transform-origin)] rounded-lg border border-line bg-raised px-2 py-1 text-[11.5px] leading-4 text-ink shadow-[0_8px_24px_-8px_rgb(0_0_0/0.8)] transition-[opacity,transform] duration-150 ease-out data-ending-style:scale-95 data-ending-style:opacity-0 data-starting-style:scale-95 data-starting-style:opacity-0">
            {label}
          </Tooltip.Popup>
        </Tooltip.Positioner>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}
