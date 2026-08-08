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
  tone = "plain",
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
  children: ReactNode;
  className?: string;
}) {
  const accent = tone === "accent";
  return (
    <section
      className={cx(
        "px-4 py-4",
        // A left rule rather than a filled panel: the tint has to survive being the first thing under the header
        // without turning the top of the rail into a coloured slab.
        accent && "relative bg-accent/[0.045] before:absolute before:inset-y-0 before:left-0 before:w-0.5 before:bg-accent/60",
        className,
      )}
    >
      <div className="mb-3 flex min-h-5 items-center justify-between gap-3">
        <h2
          className={cx(
            "text-[10.5px] font-semibold uppercase tracking-[0.1em]",
            accent ? "text-accent" : "text-faint",
          )}
        >
          {title}
        </h2>
        {action}
      </div>
      {children}
    </section>
  );
}
