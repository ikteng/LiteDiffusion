import { useId, type InputHTMLAttributes, type ReactNode } from "react";
import { cx } from "../lib/cx";

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
    <label className="block">
      <span className="mb-1.5 flex items-baseline justify-between gap-3">
        <span className="text-[11px] font-medium uppercase tracking-[0.07em] text-faint">{label}</span>
        {value && <span className="tabular text-[12px] text-muted">{value}</span>}
      </span>
      {children}
      {hint && <span className="mt-1.5 block text-[11px] leading-[1.45] text-faint">{hint}</span>}
    </label>
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

export function Slider({ min, max, step = 1, value, onChange, ariaLabel }: SliderProps) {
  const fill = max === min ? 0 : ((value - min) / (max - min)) * 100;
  return (
    <input
      type="range"
      className="range"
      aria-label={ariaLabel}
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(event) => onChange(Number(event.target.value))}
      style={{ ["--fill" as string]: `${fill}%` }}
    />
  );
}

export function TextInput({ className, ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...rest}
      className={cx(
        "h-9 w-full rounded-lg border border-line bg-sunken px-2.5 text-[13px] text-ink",
        "placeholder:text-faint hover:border-line-strong focus:border-accent focus:outline-none",
        className,
      )}
    />
  );
}

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
  const id = useId();
  return (
    <div className="flex items-start gap-3">
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-labelledby={id}
        onClick={() => onChange(!checked)}
        className={cx(
          "mt-0.5 inline-flex h-5 w-9 shrink-0 items-center rounded-full border transition-colors duration-150",
          checked ? "border-accent bg-accent" : "border-line-strong bg-sunken",
        )}
      >
        <span
          className={cx(
            "size-3.5 rounded-full bg-ink transition-transform duration-150",
            checked ? "translate-x-[18px] bg-accent-ink" : "translate-x-[3px]",
          )}
        />
      </button>
      <span className="min-w-0">
        <span id={id} className="block cursor-default text-[13px] font-medium text-ink">
          {label}
        </span>
        {description && <span className="mt-0.5 block text-[11.5px] leading-[1.45] text-muted">{description}</span>}
      </span>
    </div>
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

export function Progress({ value, label }: { value: number | null; label: string }) {
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={value == null ? undefined : Math.round(value * 100)}
      className="relative h-1 w-full overflow-hidden rounded-full bg-line"
    >
      {value == null ? (
        <span className="sweep absolute inset-y-0 left-0 w-1/3 rounded-full bg-accent" />
      ) : (
        <span
          className="block h-full rounded-full bg-accent transition-[width] duration-500 ease-out"
          style={{ width: `${Math.max(1.5, value * 100)}%` }}
        />
      )}
    </div>
  );
}
