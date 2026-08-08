import type { ReactNode } from "react";
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
  children,
  className,
}: {
  title: string;
  /** Right-hand slot on the title row — a readout, or a control that belongs to the whole section. */
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cx("px-4 py-4", className)}>
      <div className="mb-3 flex min-h-5 items-center justify-between gap-3">
        <h2 className="text-[10.5px] font-semibold uppercase tracking-[0.1em] text-faint">{title}</h2>
        {action}
      </div>
      {children}
    </section>
  );
}
