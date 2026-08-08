import { Slider as Base } from "@base-ui/react/slider";
import { cx } from "../lib/cx";

/**
 * The one slider in the app.
 *
 * Continuous and stepped sliders used to be two different components that looked nothing alike; they are the same
 * control now, and `stops` is the only thing that changes — dots appear on the track and the thumb snaps between them.
 * Everything else (track height, thumb, halo, end captions) is shared, so a preset picker and a duration slider read as
 * the same instrument.
 *
 * The thumb size is a JS constant because the dot positions have to use the same inset maths Base UI uses internally:
 * with `thumbAlignment="edge"` the thumb centre travels `THUMB/2 … width - THUMB/2`, not `0 … width`.
 */
const THUMB = 18;

type Props = {
  ariaLabel: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step?: number;
  disabled?: boolean;
  /**
   * Labels for a stepped slider, one per position. The slider then runs `0 … stops.length - 1` and each label is what
   * a screen reader announces for that index.
   */
  stops?: string[];
  minLabel?: string;
  maxLabel?: string;
};

export function Slider({
  ariaLabel,
  value,
  onChange,
  min,
  max,
  step = 1,
  disabled,
  stops,
  minLabel,
  maxLabel,
}: Props) {
  const span = Math.max(1, max - min);
  const at = (position: number) => `calc(${THUMB / 2}px + (100% - ${THUMB}px) * ${(position - min) / span})`;

  return (
    <Base.Root
      className={cx("group block select-none", disabled && "opacity-40")}
      min={min}
      max={max}
      step={step}
      value={value}
      disabled={disabled}
      thumbAlignment="edge"
      onValueChange={(next) => onChange(Array.isArray(next) ? next[0] : next)}
    >
      <Base.Control className="relative flex h-5 w-full touch-none items-center">
        <Base.Track className="h-1.5 w-full rounded-full bg-line">
          <Base.Indicator className="rounded-full bg-accent transition-[width] duration-200 ease-out group-data-dragging:duration-0" />

          {/* After the indicator so the dots sit on the fill rather than under it, and never under the thumb.
              The two end stops are skipped: at `thumbAlignment="edge"` they land on the track's own rounded caps,
              where a 4px dot reads as a smudge rather than as a position. The end captions already name them. */}
          {stops && (
            <div aria-hidden className="pointer-events-none absolute inset-0">
              {stops.map((label, index) =>
                index === value || index === 0 || index === stops.length - 1 ? null : (
                  <span
                    key={label}
                    style={{ left: at(index) }}
                    className={cx(
                      "absolute top-1/2 size-1 -translate-x-1/2 -translate-y-1/2 rounded-full",
                      index < value ? "bg-accent-ink/50" : "bg-line-strong",
                    )}
                  />
                ),
              )}
            </div>
          )}

          <Base.Thumb
            style={{ width: THUMB, height: THUMB }}
            getAriaLabel={() => ariaLabel}
            getAriaValueText={stops ? (_formatted, current) => stops[current] ?? String(current) : undefined}
            className={THUMB_CLASS}
          />
        </Base.Track>
      </Base.Control>

      {(minLabel || maxLabel) && (
        <div className="mt-2 flex items-baseline justify-between text-[11px] text-faint">
          <span>{minLabel}</span>
          <span>{maxLabel}</span>
        </div>
      )}
    </Base.Root>
  );
}

/*
 * The position is Base UI's to write — it re-measures the thumb on every value change — so the position transition is
 * CSS and gets out of the way while dragging. `scale` and `box-shadow` are separate properties from the `translate`
 * Base UI uses, so growing the halo never disturbs that measurement.
 */
const THUMB_CLASS = cx(
  "rounded-full bg-ink shadow-[0_1px_3px_rgb(0_0_0/0.6)]",
  "[--halo:color-mix(in_oklch,var(--color-accent)_28%,transparent)]",
  "[transition:inset-inline-start_200ms_ease-out,box-shadow_150ms_ease-out,scale_150ms_ease-out]",
  "group-hover:shadow-[0_1px_3px_rgb(0_0_0/0.6),0_0_0_5px_var(--halo)]",
  "group-data-dragging:scale-105",
  "group-data-dragging:shadow-[0_1px_3px_rgb(0_0_0/0.6),0_0_0_7px_var(--halo)]",
  "group-data-dragging:[transition:box-shadow_150ms_ease-out,scale_150ms_ease-out]",
);
