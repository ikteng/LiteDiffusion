import type { Acceleration, GenerationValues, LoraPreset } from "../types";

/**
 * The speed settings, remembered between visits.
 *
 * Speed is the one choice that is about the person rather than the shot: someone iterating wants 4-step every time,
 * someone rendering a final wants Exact every time, and neither should have to say so again on the next reload. Prompt,
 * seed, keyframes and canvas are all properties of *this* clip, so none of them are persisted — restoring the last
 * prompt on load would be a surprise, not a convenience.
 *
 * Everything read back is re-validated. The stored value can be older than the current build, hand-edited, or written
 * by a different version of the Space, so a shape that does not typecheck at runtime is dropped rather than trusted.
 */
const KEY = "minimax-h3.speed.v1";

export type SpeedValues = Pick<
  GenerationValues,
  "preset" | "steps" | "acceleration" | "loraPreset" | "loraRepo" | "loraFilename" | "loraStrength"
>;

const ACCELERATIONS: readonly string[] = ["Balanced", "Ultra Fast", "Exact"] satisfies Acceleration[];
const LORA_PRESETS: readonly string[] = [
  "None",
  "Turbo · 4 steps",
  "Turbo · 8 steps",
  "Custom",
] satisfies LoraPreset[];

/**
 * Storage can throw rather than merely be empty.
 *
 * A Space renders inside a cross-origin iframe on huggingface.co, so a browser blocking third-party storage raises a
 * SecurityError on the *first property access* — not on the read. Remembering the preset is a nicety; failing to do it
 * must never stop the studio loading, so both directions swallow everything.
 */
function storage(): Storage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

export function loadSpeed(): Partial<SpeedValues> {
  let parsed: unknown;
  try {
    const raw = storage()?.getItem(KEY);
    if (!raw) return {};
    parsed = JSON.parse(raw);
  } catch {
    return {};
  }
  if (typeof parsed !== "object" || parsed === null) return {};
  const stored = parsed as Record<string, unknown>;

  const speed: Partial<SpeedValues> = {};
  // The preset is only checked for shape here; `App` is what decides whether the server still offers it, because only
  // `/studio-config` knows that and it has not answered yet when this runs.
  if (typeof stored.preset === "string") speed.preset = stored.preset;
  if (typeof stored.steps === "number" && Number.isInteger(stored.steps) && stored.steps >= 1 && stored.steps <= 100) {
    speed.steps = stored.steps;
  }
  if (typeof stored.acceleration === "string" && ACCELERATIONS.includes(stored.acceleration)) {
    speed.acceleration = stored.acceleration as Acceleration;
  }
  if (typeof stored.loraPreset === "string" && LORA_PRESETS.includes(stored.loraPreset)) {
    speed.loraPreset = stored.loraPreset as LoraPreset;
  }
  if (typeof stored.loraRepo === "string") speed.loraRepo = stored.loraRepo.slice(0, 200);
  if (typeof stored.loraFilename === "string") speed.loraFilename = stored.loraFilename.slice(0, 200);
  if (typeof stored.loraStrength === "number" && Number.isFinite(stored.loraStrength)) {
    speed.loraStrength = Math.min(2, Math.max(0, stored.loraStrength));
  }
  return speed;
}

export function saveSpeed(values: GenerationValues): void {
  const speed: SpeedValues = {
    preset: values.preset,
    steps: values.steps,
    acceleration: values.acceleration,
    loraPreset: values.loraPreset,
    loraRepo: values.loraRepo,
    loraFilename: values.loraFilename,
    loraStrength: values.loraStrength,
  };
  try {
    storage()?.setItem(KEY, JSON.stringify(speed));
  } catch {
    // Blocked or full. Nothing to do and nothing worth telling the user about.
  }
}
