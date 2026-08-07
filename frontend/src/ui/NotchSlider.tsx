import { cx } from "../lib/cx";

const THUMB = 24;

type Props = {
  /** One label per stop, used for the value announcement and the tooltip on each dot. */
  stops: string[];
  index: number;
  onChange: (index: number) => void;
  ariaLabel: string;
  /** Captions for the two ends of the axis. */
  minLabel: string;
  maxLabel: string;
  disabled?: boolean;
};

/**
 * A discrete slider for an ordered set of choices.
 *
 * A list of radio rows makes the reader compare six paragraphs to find "a bit faster than the default". An axis makes
 * that one gesture, because the choices really are ordered — so the axis, not the list, is the honest control.
 *
 * The dots mark every stop the thumb can land on. The active one is omitted: the thumb is already sitting there, and a
 * dot showing through the middle of it reads as a second, wrong marker.
 */
export function NotchSlider({ stops, index, onChange, ariaLabel, minLabel, maxLabel, disabled }: Props) {
  const last = Math.max(1, stops.length - 1);
  const fraction = Math.min(Math.max(index, 0), last) / last;
  // A native thumb's centre travels from `THUMB/2` to `width - THUMB/2`, so a plain percentage would drift at the ends.
  const at = (value: number) => `calc(${THUMB / 2}px + (100% - ${THUMB}px) * ${value})`;

  return (
    <div className={cx("select-none", disabled && "opacity-45")}>
      <div className="mb-1.5 flex items-baseline justify-between text-[11.5px] text-muted">
        <span>{minLabel}</span>
        <span>{maxLabel}</span>
      </div>

      <div className="relative h-[30px] w-full rounded-full bg-raised">
        <input
          type="range"
          className="notch absolute inset-0"
          aria-label={ariaLabel}
          aria-valuetext={stops[index] ?? ""}
          min={0}
          max={last}
          step={1}
          value={Math.min(Math.max(index, 0), last)}
          disabled={disabled}
          onChange={(event) => onChange(Number(event.target.value))}
          style={{ ["--fill" as string]: at(fraction) }}
        />

        {/* After the input, so the dots sit on top of the accent fill rather than under it. */}
        <div aria-hidden className="pointer-events-none absolute inset-0">
          {stops.map((stop, stopIndex) =>
            stopIndex === index ? null : (
              <span
                key={stop}
                style={{ left: at(stopIndex / last) }}
                className={cx(
                  "absolute top-1/2 size-[5px] -translate-x-1/2 -translate-y-1/2 rounded-full",
                  stopIndex < index ? "bg-accent-ink/45" : "bg-line-strong",
                )}
              />
            ),
          )}
        </div>
      </div>
    </div>
  );
}
