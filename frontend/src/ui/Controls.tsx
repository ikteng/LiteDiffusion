import type { ComponentProps, ReactNode } from "react";
import { Field as BaseField } from "@base-ui/react/field";
import { Input } from "@base-ui/react/input";
import { Progress as BaseProgress } from "@base-ui/react/progress";
import { Slider as BaseSlider } from "@base-ui/react/slider";
import { Switch as BaseSwitch } from "@base-ui/react/switch";
import { motion } from "framer-motion";
import { cx } from "../lib/cx";
import { POP } from "../lib/motion";

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
  hint?: string;
  /** Right-aligned live readout, so the current setting is legible without reading the control. */
  value?: ReactNode;
  children: ReactNode;
}) {
  return (
    <BaseField.Root className="block">
      <span className="mb-1.5 flex items-baseline justify-between gap-3">
        <BaseField.Label className="text-[11px] font-medium uppercase tracking-[0.07em] text-faint">
          {label}
        </BaseField.Label>
        {value && <span className="tabular text-[12px] text-muted">{value}</span>}
      </span>
      {children}
      {hint && (
        <BaseField.Description className="mt-1.5 block text-[11px] leading-[1.45] text-faint">
          {hint}
        </BaseField.Description>
      )}
    </BaseField.Root>
  );
}

type SliderProps = {
  min: number;
  max: number;
  step?: number;
  value: number;
  onChange: (value: number) => void;
  ariaLabel: string;
};

/**
 * A continuous slider.
 *
 * `thumbAlignment="edge"` keeps the thumb inside the track at both ends instead of hanging half of it over the edge
 * of the panel. The position transition is cancelled while dragging (`data-dragging`) so the thumb tracks the pointer
 * exactly, and only animates when the value is changed by a keypress or a click on the track.
 */
export function Slider({ min, max, step = 1, value, onChange, ariaLabel }: SliderProps) {
  return (
    <BaseSlider.Root
      className="group block"
      min={min}
      max={max}
      step={step}
      value={value}
      thumbAlignment="edge"
      onValueChange={(next) => onChange(Array.isArray(next) ? next[0] : next)}
    >
      <BaseSlider.Control className="flex h-5 w-full touch-none select-none items-center">
        <BaseSlider.Track className="h-1 w-full rounded-full bg-line">
          <BaseSlider.Indicator className="rounded-full bg-accent transition-[width] duration-150 ease-out group-data-dragging:duration-0" />
          <BaseSlider.Thumb
            getAriaLabel={() => ariaLabel}
            className={cx(
              "size-4 rounded-full border-[3px] border-canvas bg-ink shadow-[0_0_0_1px_var(--color-line-strong)]",
              // See NotchSlider: the position is Base UI's to write, so the transition is CSS and steps out of the
              // way while dragging. `scale` is its own property, so it never disturbs the thumb measurement.
              "[transition:inset-inline-start_150ms_ease-out,scale_120ms_ease-out]",
              "group-data-dragging:scale-110 group-data-dragging:[transition:scale_120ms_ease-out]",
            )}
          />
        </BaseSlider.Track>
      </BaseSlider.Control>
    </BaseSlider.Root>
  );
}

type TextInputProps = Omit<ComponentProps<typeof Input>, "className"> & { className?: string };

export function TextInput({ className, ...rest }: TextInputProps) {
  return (
    <Input
      {...rest}
      className={cx(
        "h-9 w-full rounded-lg border border-line bg-sunken px-2.5 text-[13px] text-ink",
        "transition-colors duration-100 placeholder:text-faint hover:border-line-strong",
        "focus:border-accent focus:outline-none",
        className,
      )}
    />
  );
}

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
          "mt-0.5 inline-flex h-5 w-9 shrink-0 items-center rounded-full border p-[2px]",
          "border-line-strong bg-sunken transition-colors duration-150",
          "data-checked:border-accent data-checked:bg-accent",
        )}
      >
        <BaseSwitch.Thumb
          className="block size-3.5 rounded-full bg-ink data-checked:ml-auto data-checked:bg-accent-ink"
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
    neutral: "border-line bg-raised text-muted",
    accent: "border-accent/40 bg-accent/12 text-accent",
    ok: "border-ok/35 bg-ok/12 text-ok",
    bad: "border-bad/40 bg-bad/12 text-bad",
  } as const;
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11px] font-medium",
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
      <BaseProgress.Track className="relative h-1 w-full overflow-hidden rounded-full bg-line">
        {percent == null ? (
          <span className="sweep absolute inset-y-0 left-0 w-1/3 rounded-full bg-accent" />
        ) : (
          <BaseProgress.Indicator className="h-full min-w-[3px] rounded-full bg-accent transition-[width] duration-500 ease-out" />
        )}
      </BaseProgress.Track>
    </BaseProgress.Root>
  );
}
