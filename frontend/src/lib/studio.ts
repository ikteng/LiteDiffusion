import type { Acceleration, CanvasOption, PresetOption, StudioConfig } from "../types";

/** Mirrors `FPS`, `FRAMES_PER_CHUNK`, `LATENTS_PER_CHUNK` in app.py. */
export const FPS = 24;
const FRAMES_PER_CHUNK = 17;
const LATENTS_PER_CHUNK = 5;

/**
 * The frame count MiniMax-H3's video VAE can actually decode: the next `17n + 5` at 24 fps.
 *
 * The old UI showed the raw slider value, so "5 seconds" silently produced 5.042 s of video. Snapping here lets the
 * composer show the length the user is really going to get.
 */
export function snapFrames(seconds: number): number {
  let frames = Math.max(1, Math.round(seconds * FPS));
  while (frames % FRAMES_PER_CHUNK !== LATENTS_PER_CHUNK) frames += 1;
  return frames;
}

/**
 * Seconds of ZeroGPU this request will book.
 *
 * Mirrors `get_duration` in app.py, including its padding — this is the quota the user spends, not a promise about
 * wall-clock time. Keep the two in sync; the constants below are copied verbatim from that function.
 */
const DUR_B = 1.1745e-4;
const DUR_C = 3.8396e-9;
const DECODE_BASE = 15;
const DECODE_PER_DEFAULT_CANVAS = 15;
const DEFAULT_CANVAS_PIXELS = 960 * 544 * 124;
const PLACEMENT_ALLOWANCE = 12;
const PAD = 10;
const LOCAL_CONDITIONING = 20;

export function estimateGpuSeconds(input: {
  width: number;
  height: number;
  frames: number;
  steps: number;
  keyframes: number;
  /** Upsampling conditions remotely, so the local conditioning allowance drops out of the booking. */
  upsample: boolean;
}): number {
  const { width, height, frames, steps, keyframes, upsample } = input;
  const latentFrames = Math.floor((frames - LATENTS_PER_CHUNK) / FRAMES_PER_CHUNK) * LATENTS_PER_CHUNK + 2;
  const patches = Math.floor(height / 32) * Math.floor(width / 32);
  const rows = latentFrames * patches + keyframes * patches;
  const denoise = steps * (DUR_B * rows + DUR_C * rows ** 2);
  const decode = DECODE_BASE + (DECODE_PER_DEFAULT_CANVAS * (height * width * frames)) / DEFAULT_CANVAS_PIXELS;
  const conditioning = upsample ? 0 : LOCAL_CONDITIONING;
  return Math.max(60, Math.floor(denoise + decode) + conditioning + PLACEMENT_ALLOWANCE + PAD);
}

/** "1m 12s" / "48s" — compact enough to sit inside a button. */
export function formatBudget(seconds: number): string {
  const whole = Math.round(seconds);
  if (whole < 60) return `${whole}s`;
  const minutes = Math.floor(whole / 60);
  const rest = whole % 60;
  return rest === 0 ? `${minutes}m` : `${minutes}m ${rest}s`;
}

/** "0:05" for clip lengths, where a minutes:seconds clock reads faster than "5.0 s". */
export function formatClock(seconds: number): string {
  const total = Math.round(seconds);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

export function formatEta(seconds?: number): string | null {
  if (seconds == null || !Number.isFinite(seconds) || seconds <= 0) return null;
  return seconds < 60 ? `${Math.ceil(seconds)}s left` : `${Math.ceil(seconds / 60)}m left`;
}

function greatestCommonDivisor(a: number, b: number): number {
  return b === 0 ? a : greatestCommonDivisor(b, a % b);
}

/**
 * The aspect ratio a canvas belongs to.
 *
 * Server labels look like `960x544 · 16:9 fast`, so the authored ratio is right there and is what the user recognises.
 * Reducing the pixel dimensions is only the fallback, because 960x544 reduces to 30:17, not 16:9.
 */
export function ratioOf(canvas: CanvasOption): string {
  const authored = canvas.label.split("·")[1]?.trim().split(/\s+/)[0];
  if (authored && /^\d+:\d+$/.test(authored)) return authored;
  const divisor = greatestCommonDivisor(canvas.width, canvas.height) || 1;
  return `${canvas.width / divisor}:${canvas.height / divisor}`;
}

export type AspectGroup = {
  ratio: string;
  /** Ratio as a number, for drawing the shape swatch. */
  value: number;
  options: CanvasOption[];
};

/**
 * Collapse the flat 16-entry canvas list into "pick a shape, then pick a size".
 *
 * Groups keep first-appearance order from the server, which is already curated (16:9 first, exotic ratios last), and
 * sizes are sorted small-to-large so the segmented control reads as a quality ramp.
 */
export function groupCanvases(canvases: CanvasOption[]): AspectGroup[] {
  const groups = new Map<string, AspectGroup>();
  for (const canvas of canvases) {
    const ratio = ratioOf(canvas);
    let group = groups.get(ratio);
    if (!group) {
      group = { ratio, value: canvas.width / canvas.height, options: [] };
      groups.set(ratio, group);
    }
    group.options.push(canvas);
  }
  for (const group of groups.values()) {
    group.options.sort((a, b) => a.width * a.height - b.width * b.height);
  }
  return [...groups.values()];
}

export function findCanvas(config: StudioConfig, label: string): CanvasOption {
  return config.canvases.find((canvas) => canvas.label === label) ?? config.canvases[0];
}

export function findPreset(config: StudioConfig, value: string): PresetOption {
  return config.presets.find((preset) => preset.value === value) ?? config.presets[0];
}

/** How much of the schedule each cache engine actually evaluates, least to most. */
const ACCELERATION_RANK: Record<Acceleration, number> = { "Ultra Fast": 0, Balanced: 1, Exact: 2 };

/**
 * The presets ordered along the one axis they vary on: how much computation the run really performs.
 *
 * More scheduler steps is smarter; at equal steps, less approximate caching is smarter. Both numbers come from
 * `/studio-config`, so a preset added on the server lands in the right place without a table to maintain here.
 * `Custom` is excluded — it is not a point on the axis but an escape from it.
 */
export function presetAxis(presets: PresetOption[]): PresetOption[] {
  return presets
    .filter((preset) => !preset.custom)
    .slice()
    .sort(
      (a, b) => a.steps - b.steps || ACCELERATION_RANK[a.acceleration] - ACCELERATION_RANK[b.acceleration],
    );
}

/** `"Balanced — best overall (recommended)"` → `"Balanced"`. The em dash tail is a tagline, not part of the name. */
export function presetName(value: string): string {
  return value.split("—")[0].trim() || value;
}

/** …and the tail, which is the most useful thing to show next to the name. */
export function presetTagline(value: string): string {
  const tail = value.split("—").slice(1).join("—").trim();
  return tail.replace(/\s*\(recommended\)\s*/i, "").trim();
}

/**
 * Break the server's single-line report into separate facts.
 *
 * Splits on the `·` separators and on commas that are not inside parentheses, so
 * `conditioner 12s (245 tokens, upsampled)` survives as one fact.
 */
export function parseReport(report: string): string[] {
  const facts: string[] = [];
  let current = "";
  let depth = 0;
  for (const character of report.replaceAll("`", "")) {
    if (character === "(") depth += 1;
    if (character === ")") depth = Math.max(0, depth - 1);
    if (depth === 0 && (character === "·" || character === ",")) {
      facts.push(current);
      current = "";
      continue;
    }
    current += character;
  }
  facts.push(current);
  return facts.map((fact) => fact.trim()).filter(Boolean);
}

export function randomSeed(): number {
  return Math.floor(Math.random() * 2 ** 31);
}
