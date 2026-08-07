import type { ReactNode } from "react";
import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { Collapsible } from "@base-ui/react/collapsible";
import { cx } from "../lib/cx";

/**
 * One labelled block of the control rail.
 *
 * Sections are separated by the rail's `divide-y` rather than by their own borders, so the rules always line up with
 * the rail edges and a section can be added or removed without leaving a double line behind.
 */
export function Section({
  title,
  action,
  tone = "plain",
  defaultOpen = true,
  children,
  className,
}: {
  title: string;
  /** Right-hand slot on the title row — a readout, or a control that belongs to the whole section. */
  action?: ReactNode;
  /**
   * `accent` marks the one section worth reading before the others. It is a deliberately scarce treatment: a second
   * accented section would make both of them ordinary again.
   */
  tone?: "plain" | "accent";
  /** Each rail block can get out of the way without hiding controls behind a separate dialog. */
  defaultOpen?: boolean;
  children: ReactNode;
  className?: string;
}) {
  const accent = tone === "accent";
  const [open, setOpen] = useState(defaultOpen);
  return (
    <Collapsible.Root open={open} onOpenChange={setOpen} render={<section />}>
      <div
        className={cx(
          "px-4 py-4",
          accent && "relative bg-accent/[0.045] before:absolute before:inset-y-0 before:left-0 before:w-0.5 before:bg-accent/60",
          className,
        )}
      >
        <div className={cx("flex min-h-5 items-center justify-between gap-3", open && "mb-3")}>
          <Collapsible.Trigger
            className={cx(
              "group flex min-w-0 items-center gap-1.5 text-[10.5px] font-semibold uppercase tracking-[0.1em]",
              accent ? "text-accent" : "text-faint",
            )}
          >
            <ChevronDown className="size-3.5 transition-transform duration-200 group-data-panel-open:rotate-180" />
            {title}
          </Collapsible.Trigger>
          {action}
        </div>
        <Collapsible.Panel
          className={cx(
            "h-[var(--collapsible-panel-height)] overflow-hidden transition-[height] duration-200 ease-out",
            "data-starting-style:h-0 data-ending-style:h-0",
          )}
        >
          {children}
        </Collapsible.Panel>
      </div>
    </Collapsible.Root>
  );
}
