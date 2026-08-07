import { useId, type ReactNode } from "react";
import { Radio } from "@base-ui/react/radio";
import { RadioGroup } from "@base-ui/react/radio-group";
import { motion } from "framer-motion";
import { cx } from "../lib/cx";
import { POP } from "../lib/motion";

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
 * A radio group drawn as one connected control.
 *
 * Used wherever a choice is small and worth showing in full — resolution within an aspect ratio, cache engine, LoRA
 * schedule. Anything longer than about five options belongs in a list instead.
 *
 * The selected background is a single shared element that Framer Motion moves between the segments (`layoutId`), so
 * picking a neighbour reads as the highlight sliding across rather than one box blinking off and another on. Because
 * it is the *same* element, it never double-renders during the handover.
 */
export function Segmented<T extends string>({ ariaLabel, value, options, onChange, vertical, className }: Props<T>) {
  const highlight = useId();

  return (
    <RadioGroup
      aria-label={ariaLabel}
      value={value}
      onValueChange={(next) => onChange(next as T)}
      className={cx(
        "flex gap-1 rounded-xl border border-line bg-sunken p-1",
        vertical ? "flex-col" : "flex-row",
        className,
      )}
    >
      {options.map((option) => {
        const selected = option.value === value;
        return (
          <Radio.Root
            key={option.value}
            value={option.value}
            disabled={option.disabled}
            className={cx(
              "relative flex min-w-0 flex-1 cursor-pointer flex-col items-center justify-center rounded-lg px-2 py-1.5",
              "transition-colors duration-100 data-disabled:cursor-not-allowed data-disabled:opacity-40",
              selected ? "text-ink" : "text-muted hover:bg-raised/60 hover:text-ink",
            )}
          >
            {selected && (
              <motion.span
                aria-hidden
                layoutId={highlight}
                transition={POP}
                className="absolute inset-0 rounded-lg bg-raised shadow-sm"
              />
            )}
            {/* The label sits above the highlight, which is the only reason these two need a stacking context. */}
            <span className="relative truncate text-[13px] font-medium leading-5">{option.label}</span>
            {option.hint && (
              <span
                className={cx("tabular relative truncate text-[10px] leading-4", selected ? "text-muted" : "text-faint")}
              >
                {option.hint}
              </span>
            )}
          </Radio.Root>
        );
      })}
    </RadioGroup>
  );
}
