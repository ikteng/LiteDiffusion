import { ChevronDown } from "lucide-react";
import type { ReactNode } from "react";
import { cx } from "../lib/cx";
import { Popover } from "../ui/Popover";

type Props = {
  label: string;
  value: string;
  icon?: ReactNode;
  align?: "start" | "end";
  width?: string;
  disabled?: boolean;
  /** Marks the pill as carrying a non-default setting, so a changed value is visible without opening it. */
  active?: boolean;
  children: (close: () => void) => ReactNode;
};

/**
 * The only shape a setting takes in this UI: a pill showing `label · value`, opening a panel with the real control.
 *
 * Every setting is therefore one click from the prompt, always shows its current value, and never occupies vertical
 * space when the user does not care about it.
 */
export function ControlPill({ label, value, icon, align, width, disabled, active, children }: Props) {
  return (
    <Popover
      panelLabel={label}
      align={align}
      width={width}
      disabled={disabled}
      triggerClassName="block max-w-full"
      trigger={({ open }) => (
        <span
          className={cx(
            "flex h-9 max-w-full items-center gap-2 rounded-lg border px-2.5 transition-colors duration-100",
            open
              ? "border-accent/60 bg-raised"
              : active
                ? "border-line-strong bg-surface hover:bg-raised"
                : "border-line bg-surface hover:border-line-strong hover:bg-raised",
            disabled && "cursor-not-allowed opacity-50",
          )}
        >
          {icon && <span className={cx("shrink-0", active ? "text-accent" : "text-faint")}>{icon}</span>}
          <span className="hidden text-[11px] uppercase tracking-[0.06em] text-faint sm:inline">{label}</span>
          <span className="tabular truncate text-[13px] font-medium text-ink">{value}</span>
          <ChevronDown
            className={cx("size-3.5 shrink-0 text-faint transition-transform duration-150", open && "rotate-180")}
          />
        </span>
      )}
    >
      {children}
    </Popover>
  );
}
