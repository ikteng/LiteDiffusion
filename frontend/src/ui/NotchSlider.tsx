import { Slider } from "@base-ui/react/slider";
import { cx } from "../lib/cx";

const TRACK = 30;
const THUMB = 24;

type Props = {
  /** One label per stop, used for the value announcement and to name the position. */
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
 *
 * Motion here is CSS rather than Framer Motion, and that is deliberate. Base UI positions the thumb by writing
 * `inset-inline-start` and sizes the fill by writing `width`; a JS animation loop would be overwriting those same
 * values sixty times a second while you drag. A transition that switches itself off under `data-dragging` gives the
 * better behaviour anyway — the thumb glides when you step it with a key or click a dot, and pins to your finger when
 * you actually drag it.
 */
export function NotchSlider({ stops, index, onChange, ariaLabel, minLabel, maxLabel, disabled }: Props) {
  const last = Math.max(1, stops.length - 1);
  const value = Math.min(Math.max(index, 0), last);
  // `thumbAlignment="edge"` puts the thumb centre at `THUMB/2 … width - THUMB/2`, so the dots have to use the same
  // inset travel or they would drift away from the thumb at the ends.
  const at = (stop: number) => `calc(${THUMB / 2}px + (100% - ${THUMB}px) * ${stop / last})`;

  return (
    <Slider.Root
      className={cx("group block select-none", disabled && "opacity-45")}
      min={0}
      max={last}
      step={1}
      value={value}
      disabled={disabled}
      thumbAlignment="edge"
      onValueChange={(next) => onChange(Array.isArray(next) ? next[0] : next)}
    >
      <div className="mb-1.5 flex items-baseline justify-between text-[11.5px] text-muted">
        <span>{minLabel}</span>
        <span>{maxLabel}</span>
      </div>

      <Slider.Control
        className="relative flex w-full touch-none items-center rounded-full bg-raised"
        style={{ height: TRACK }}
      >
        <Slider.Track className="h-full w-full rounded-full">
          <Slider.Indicator className="rounded-full bg-accent transition-[width] duration-200 ease-out group-data-dragging:duration-0" />

          {/* After the indicator, so the dots sit on top of the accent fill rather than under it. */}
          <div aria-hidden className="pointer-events-none absolute inset-0">
            {stops.map((stop, stopIndex) =>
              stopIndex === value ? null : (
                <span
                  key={stop}
                  style={{ left: at(stopIndex) }}
                  className={cx(
                    "absolute top-1/2 size-[5px] -translate-x-1/2 -translate-y-1/2 rounded-full",
                    stopIndex < value ? "bg-accent-ink/45" : "bg-line-strong",
                  )}
                />
              ),
            )}
          </div>

          <Slider.Thumb
            style={{ width: THUMB, height: THUMB }}
            getAriaLabel={() => ariaLabel}
            getAriaValueText={(_formatted, current) => stops[current] ?? String(current)}
            className={cx(
              "rounded-full bg-ink shadow-[0_1px_4px_rgb(0_0_0/0.55)]",
              // Base UI writes `translate` on the thumb, and Tailwind's `scale-*` is the separate `scale` property,
              // so the press feedback and the positioning never overwrite each other.
              "[transition:inset-inline-start_200ms_ease-out,scale_120ms_ease-out]",
              "group-data-dragging:scale-[1.06] group-data-dragging:[transition:scale_120ms_ease-out]",
            )}
          />
        </Slider.Track>
      </Slider.Control>
    </Slider.Root>
  );
}
