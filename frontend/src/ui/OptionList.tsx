import { Check } from "lucide-react";
import type { ReactNode } from "react";
import { cx } from "../lib/cx";

export type ListOption<T extends string> = {
  value: T;
  title: string;
  description?: string;
  /** Right-aligned metadata — a GPU estimate, a step count. */
  meta?: ReactNode;
  badge?: string;
};

type Props<T extends string> = {
  ariaLabel: string;
  value: T;
  options: ListOption<T>[];
  onChange: (value: T) => void;
  /** Rendered directly under the matching row, for options that carry their own controls. */
  renderExtra?: (value: T) => ReactNode;
};

/**
 * A vertical radiogroup where every option shows its own description.
 *
 * This replaces the old split where a tab bar showed three of the presets and a hidden dropdown showed all six: one
 * list, one source of truth, and the descriptions the server already sends are finally visible at the point of choice.
 */
export function OptionList<T extends string>({ ariaLabel, value, options, onChange, renderExtra }: Props<T>) {
  return (
    <div role="radiogroup" aria-label={ariaLabel} className="flex flex-col gap-1">
      {options.map((option) => {
        const selected = option.value === value;
        const extra = selected ? renderExtra?.(option.value) : null;
        return (
          <div key={option.value}>
            <button
              type="button"
              role="radio"
              aria-checked={selected}
              onClick={() => onChange(option.value)}
              className={cx(
                "flex w-full items-start gap-3 rounded-lg border px-3 py-2.5 text-left transition-colors duration-100",
                selected
                  ? "border-accent/45 bg-accent/10"
                  : "border-transparent hover:border-line hover:bg-raised/70",
              )}
            >
              <span
                aria-hidden
                className={cx(
                  "mt-0.5 grid size-4 shrink-0 place-items-center rounded-full border",
                  selected ? "border-accent bg-accent text-accent-ink" : "border-line-strong",
                )}
              >
                {selected && <Check className="size-3" strokeWidth={3.5} />}
              </span>
              <span className="min-w-0 flex-1">
                <span className="flex flex-wrap items-center gap-1.5">
                  <span className="text-[13px] font-medium text-ink">{option.title}</span>
                  {option.badge && (
                    <span className="rounded bg-accent/15 px-1.5 py-px text-[10px] font-medium uppercase tracking-wide text-accent">
                      {option.badge}
                    </span>
                  )}
                </span>
                {option.description && (
                  <span className="mt-0.5 block text-[11.5px] leading-[1.45] text-muted">{option.description}</span>
                )}
              </span>
              {option.meta && <span className="tabular shrink-0 pt-0.5 text-[11px] text-faint">{option.meta}</span>}
            </button>
            {extra && <div className="mt-2 mb-1 ml-7 flex flex-col gap-3 border-l border-line pl-3">{extra}</div>}
          </div>
        );
      })}
    </div>
  );
}
