import { useId } from "react";
import { Radio } from "@base-ui/react/radio";
import { RadioGroup } from "@base-ui/react/radio-group";
import { motion } from "framer-motion";
import { cx } from "../lib/cx";
import { POP } from "../lib/motion";

export type SegmentedOption<T extends string> = {
  value: T;
  label: string;
  /** Second line, for the detail that distinguishes two otherwise similar options. */
  hint?: string;
  disabled?: boolean;
};

type Props<T extends string> = {
  ariaLabel: string;
  value: T;
  options: SegmentedOption<T>[];
  onChange: (value: T) => void;
  vertical?: boolean;
  className?: string;
};

/**
 * A radio group that looks like a set of tabs.
 *
 * The selected background is a single shared element moved by Framer Motion's `layoutId` rather than a class toggled on
 * each segment — one object sliding to the choice you made, instead of one fading out while another fades in. `useId`
 * scopes that shared element per instance so two segmented controls on screen never animate into each other.
 */
export function Segmented<T extends string>({ ariaLabel, value, options, onChange, vertical, className }: Props<T>) {
  const highlight = useId();

  return (
    <RadioGroup
      aria-label={ariaLabel}
      value={value}
      onValueChange={(next) => onChange(next as T)}
      className={cx(
        "flex gap-1 rounded-xl bg-sunken p-1 ring-1 ring-inset ring-line",
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
              "relative flex min-h-8 min-w-0 flex-1 cursor-pointer flex-col items-center justify-center rounded-lg px-2 py-1",
              "transition-colors duration-100 data-disabled:cursor-not-allowed data-disabled:opacity-40",
              selected ? "text-ink" : "text-muted hover:bg-raised/60 hover:text-ink",
            )}
          >
            {selected && (
              <motion.span
                aria-hidden
                layoutId={highlight}
                transition={POP}
                className="absolute inset-0 rounded-lg bg-raised shadow-[inset_0_1px_0_rgb(255_255_255/0.07),0_1px_2px_rgb(0_0_0/0.35)]"
              />
            )}
            <span className="relative truncate text-[12.5px] font-medium leading-[1.15]">{option.label}</span>
            {option.hint && (
              <span className={cx("tabular relative mt-0.5 truncate text-[10px] leading-3", selected ? "text-muted" : "text-faint")}>
                {option.hint}
              </span>
            )}
          </Radio.Root>
        );
      })}
    </RadioGroup>
  );
}
