import { useRef, type KeyboardEvent, type ReactNode } from "react";
import { cx } from "../lib/cx";

export type SegmentedOption<T extends string> = {
  value: T;
  label: ReactNode;
  hint?: string;
  disabled?: boolean;
};

type Props<T extends string> = {
  ariaLabel: string;
  value: T;
  options: SegmentedOption<T>[];
  onChange: (value: T) => void;
  /** Stack vertically instead of splitting the row into equal columns. */
  vertical?: boolean;
  className?: string;
};

/**
 * A radiogroup with roving tabindex and arrow-key movement.
 *
 * Used wherever a choice is small and worth showing in full — resolution within an aspect ratio, cache engine, LoRA
 * schedule. Anything longer than about five options belongs in a list instead.
 */
export function Segmented<T extends string>({ ariaLabel, value, options, onChange, vertical, className }: Props<T>) {
  const containerRef = useRef<HTMLDivElement>(null);

  function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const forward = vertical ? "ArrowDown" : "ArrowRight";
    const backward = vertical ? "ArrowUp" : "ArrowLeft";
    if (event.key !== forward && event.key !== backward) return;
    event.preventDefault();
    const selectable = options.filter((option) => !option.disabled);
    const index = selectable.findIndex((option) => option.value === value);
    const step = event.key === forward ? 1 : -1;
    const next = selectable[(index + step + selectable.length) % selectable.length];
    if (!next) return;
    onChange(next.value);
    const buttons = containerRef.current?.querySelectorAll<HTMLButtonElement>("[role='radio']");
    buttons?.[options.indexOf(next)]?.focus();
  }

  return (
    <div
      ref={containerRef}
      role="radiogroup"
      aria-label={ariaLabel}
      onKeyDown={onKeyDown}
      className={cx(
        "flex gap-1 rounded-xl border border-line bg-sunken p-1",
        vertical ? "flex-col" : "flex-row",
        className,
      )}
    >
      {options.map((option) => {
        const selected = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={selected}
            disabled={option.disabled}
            tabIndex={selected ? 0 : -1}
            onClick={() => onChange(option.value)}
            className={cx(
              "flex min-w-0 flex-1 flex-col items-center justify-center rounded-lg px-2 py-1.5 transition-colors duration-100",
              "disabled:cursor-not-allowed disabled:opacity-40",
              selected ? "bg-raised text-ink shadow-sm" : "text-muted hover:bg-raised/60 hover:text-ink",
            )}
          >
            <span className="truncate text-[13px] font-medium leading-5">{option.label}</span>
            {option.hint && (
              <span className={cx("tabular truncate text-[10px] leading-4", selected ? "text-muted" : "text-faint")}>
                {option.hint}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
