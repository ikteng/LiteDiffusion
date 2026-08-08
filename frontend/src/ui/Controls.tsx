import type { ComponentProps, ReactNode } from "react";
import { Dices, Minus, Plus } from "lucide-react";
import { Field as BaseField } from "@base-ui/react/field";
import { Input } from "@base-ui/react/input";
import { NumberField } from "@base-ui/react/number-field";
import { Progress as BaseProgress } from "@base-ui/react/progress";
import { Switch as BaseSwitch } from "@base-ui/react/switch";
import { motion } from "framer-motion";
import { cx } from "../lib/cx";
import { POP } from "../lib/motion";
import { Tip } from "./Tip";

/**
 * Label, live readout and hint around one control.
 *
 * `Field.Root` is what wires the label and the description to whatever control sits inside — including the slider and
 * the switch, which a plain `<label>` cannot describe correctly.
 */
export function Field({
  label,
  hint,
  value,
  children,
}: {
  label: string;
  hint?: ReactNode;
  /** Right-aligned live readout, so the current setting is legible without reading the control. */
  value?: ReactNode;
  children: ReactNode;
}) {
  return (
    <BaseField.Root className="block">
      <span className="mb-2 flex items-baseline justify-between gap-3">
        <BaseField.Label className="text-[12px] font-medium text-muted">{label}</BaseField.Label>
        {value && <span className="tabular shrink-0 text-[12px] font-medium text-ink">{value}</span>}
      </span>
      {children}
      {hint && (
        <BaseField.Description className="mt-2 block text-[11px] leading-[1.5] text-faint">
          {hint}
        </BaseField.Description>
      )}
    </BaseField.Root>
  );
}

type TextInputProps = Omit<ComponentProps<typeof Input>, "className"> & { className?: string };

export function TextInput({ className, ...rest }: TextInputProps) {
  return (
    <Input
      {...rest}
      className={cx(
        "h-9 w-full rounded-lg bg-sunken px-2.5 text-[13px] text-ink ring-1 ring-inset ring-line",
        "transition-[box-shadow,background-color] duration-100 placeholder:text-faint hover:ring-line-strong",
        "focus:bg-canvas focus:ring-accent focus:outline-none",
        className,
      )}
    />
  );
}

/**
 * An integer field with steppers.
 *
 * The steppers are not decoration on a seed: the neighbours of a seed you liked are the most useful thing to try next,
 * and ±1 is a click here instead of a select-all-and-retype. Grouping is disabled because `1,234,567` is a quantity
 * and a seed is a name.
 */
export function NumberInput({
  value,
  onChange,
  min,
  max,
  ariaLabel,
  onRandomise,
}: {
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  ariaLabel: string;
  onRandomise?: () => void;
}) {
  return (
    <div className="flex gap-2">
      <NumberField.Root
        value={value}
        onValueChange={(next) => onChange(next ?? 0)}
        min={min}
        max={max}
        step={1}
        largeStep={1000}
        format={{ useGrouping: false }}
        className="min-w-0 flex-1"
      >
        <NumberField.Group className="flex h-9 w-full overflow-hidden rounded-lg bg-sunken ring-1 ring-inset ring-line transition-shadow duration-100 hover:ring-line-strong focus-within:ring-accent">
          <NumberField.Decrement className={STEPPER} aria-label={`Decrease ${ariaLabel}`}>
            <Minus className="size-3.5" />
          </NumberField.Decrement>
          <NumberField.Input
            aria-label={ariaLabel}
            className="tabular h-full min-w-0 flex-1 bg-transparent text-center text-[13px] text-ink focus:outline-none"
          />
          <NumberField.Increment className={STEPPER} aria-label={`Increase ${ariaLabel}`}>
            <Plus className="size-3.5" />
          </NumberField.Increment>
        </NumberField.Group>
      </NumberField.Root>

      {onRandomise && (
        <Tip label="Pick a random seed">
          <button
            type="button"
            onClick={onRandomise}
            aria-label="Randomise seed"
            className="grid size-9 shrink-0 place-items-center rounded-lg bg-raised text-muted shadow-[inset_0_1px_0_rgb(255_255_255/0.06)] transition-colors duration-100 hover:bg-line hover:text-ink"
          >
            <Dices className="size-4" />
          </button>
        </Tip>
      )}
    </div>
  );
}

const STEPPER = cx(
  "grid w-8 shrink-0 place-items-center text-muted transition-colors duration-100",
  "hover:bg-raised hover:text-ink data-disabled:text-faint/50 data-disabled:hover:bg-transparent",
);

/**
 * A labelled switch.
 *
 * The thumb moves by changing which side of the track it is anchored to and letting Framer Motion's `layout` spring it
 * across, so it lands the same way whatever the track ends up measuring.
 */
export function Switch({
  checked,
  onChange,
  label,
  description,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: ReactNode;
  description?: string;
}) {
  return (
    <BaseField.Root className="flex items-start gap-3">
      <BaseSwitch.Root
        checked={checked}
        onCheckedChange={onChange}
        className={cx(
          "mt-px inline-flex h-5 w-9 shrink-0 items-center rounded-full p-[2px]",
          "bg-line ring-1 ring-inset ring-white/5 transition-colors duration-150",
          "data-checked:bg-accent",
        )}
      >
        <BaseSwitch.Thumb
          className="block size-4 rounded-full bg-ink shadow-[0_1px_2px_rgb(0_0_0/0.5)] data-checked:ml-auto data-checked:bg-accent-ink"
          render={<motion.span layout transition={POP} />}
        />
      </BaseSwitch.Root>
      <span className="min-w-0">
        <BaseField.Label className="block cursor-pointer text-[13px] font-medium text-ink">{label}</BaseField.Label>
        {description && (
          <BaseField.Description className="mt-0.5 block text-[11.5px] leading-[1.45] text-muted">
            {description}
          </BaseField.Description>
        )}
      </span>
    </BaseField.Root>
  );
}

export function Chip({
  children,
  tone = "neutral",
  className,
}: {
  children: ReactNode;
  tone?: "neutral" | "accent" | "ok" | "bad";
  className?: string;
}) {
  const tones = {
    neutral: "bg-raised text-muted ring-line",
    accent: "bg-accent/12 text-accent ring-accent/35",
    ok: "bg-ok/12 text-ok ring-ok/30",
    bad: "bg-bad/12 text-bad ring-bad/35",
  } as const;
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-[11px] font-medium ring-1 ring-inset",
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

/**
 * Determinate and indeterminate progress from one component.
 *
 * A `null` value is genuinely unknown, so it gets a shuttle rather than a bar at some invented percentage — the point
 * is that it must never read as "nearly done".
 */
export function Progress({ value, label }: { value: number | null; label: string }) {
  const percent = value == null ? null : Math.round(value * 100);
  return (
    <BaseProgress.Root value={percent} className="block w-full">
      <BaseProgress.Label className="sr-only">{label}</BaseProgress.Label>
      <BaseProgress.Track className="relative h-1.5 w-full overflow-hidden rounded-full bg-line">
        {percent == null ? (
          <span className="sweep absolute inset-y-0 left-0 w-1/3 rounded-full bg-accent" />
        ) : (
          <BaseProgress.Indicator className="h-full min-w-[3px] rounded-full bg-accent transition-[width] duration-500 ease-out" />
        )}
      </BaseProgress.Track>
    </BaseProgress.Root>
  );
}
