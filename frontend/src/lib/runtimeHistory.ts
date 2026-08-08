import { findCanvas, findPreset, snapFrames } from "./studio";
import type { GenerationValues, RuntimeEstimate, StudioConfig } from "../types";

const KEY = "minimax-h3.runtime-history.v1";
const MAX_SAMPLES = 60;

export type RuntimeSample = {
  profile: string;
  width: number;
  height: number;
  frames: number;
  steps: number;
  keyframes: number;
  upsample: boolean;
  seconds: number;
  createdAt: number;
};

function storage(): Storage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

function validSample(value: unknown): value is RuntimeSample {
  if (typeof value !== "object" || value === null) return false;
  const sample = value as Record<string, unknown>;
  return (
    typeof sample.profile === "string" &&
    ["width", "height", "frames", "steps", "keyframes", "seconds", "createdAt"].every(
      (key) => typeof sample[key] === "number" && Number.isFinite(sample[key]),
    ) &&
    typeof sample.upsample === "boolean" &&
    (sample.seconds as number) >= 1 &&
    (sample.seconds as number) <= 60 * 60
  );
}

export function loadRuntimeSamples(): RuntimeSample[] {
  try {
    const parsed: unknown = JSON.parse(storage()?.getItem(KEY) ?? "[]");
    return Array.isArray(parsed) ? parsed.filter(validSample).slice(0, MAX_SAMPLES) : [];
  } catch {
    return [];
  }
}

function profileFor(config: StudioConfig, values: GenerationValues): { profile: string; steps: number } {
  const preset = findPreset(config, values.preset);
  const steps = preset.custom ? values.steps : preset.steps;
  const acceleration = preset.custom ? values.acceleration : preset.acceleration;
  const adapter = preset.custom
    ? `${values.loraPreset}:${values.loraRepo}:${values.loraFilename}:${values.loraStrength}`
    : values.preset;
  return { profile: `${values.preset}|${steps}|${acceleration}|${adapter}`, steps };
}

export function runtimeSampleFor(
  config: StudioConfig,
  values: GenerationValues,
  seconds: number,
): RuntimeSample {
  const canvas = findCanvas(config, values.canvas);
  const { profile, steps } = profileFor(config, values);
  return {
    profile,
    width: canvas.width,
    height: canvas.height,
    frames: snapFrames(values.duration),
    steps,
    keyframes: values.referenceMode === "omni"
      ? values.references.reduce((sum, reference) => sum + (reference.kind === "video" ? 5 : 1), 0)
      : Number(values.image != null) + Number(values.lastImage != null),
    upsample: values.upsample,
    seconds,
    createdAt: Date.now(),
  };
}

export function rememberRuntime(samples: RuntimeSample[], sample: RuntimeSample): RuntimeSample[] {
  const next = [sample, ...samples].slice(0, MAX_SAMPLES);
  try {
    storage()?.setItem(KEY, JSON.stringify(next));
  } catch {
    // Runtime history is a convenience; private browsing and blocked iframe storage must not affect generation.
  }
  return next;
}

/** Relative work used only to transfer an observed runtime between canvas sizes and clip lengths of one preset. */
function work(sample: Pick<RuntimeSample, "width" | "height" | "frames" | "steps" | "keyframes">): number {
  const latentFrames = Math.floor((sample.frames - 5) / 17) * 5 + 2;
  const patches = Math.floor(sample.width / 32) * Math.floor(sample.height / 32);
  const rows = latentFrames * patches + sample.keyframes * patches;
  const denoise = sample.steps * (rows + rows ** 2 / 80_000);
  const decode = (sample.width * sample.height * sample.frames) / 80;
  return denoise + decode;
}

/**
 * Wall-clock estimate learned from this browser's completed runs of the same preset.
 *
 * Samples at nearby sizes count more heavily. Each observation is scaled by relative model work before averaging, so
 * a remembered 544p run can still estimate a 768p request without confusing a ZeroGPU booking with elapsed time.
 */
export function estimateRuntime(
  samples: RuntimeSample[],
  config: StudioConfig,
  values: GenerationValues,
): RuntimeEstimate | null {
  const canvas = findCanvas(config, values.canvas);
  const { profile, steps } = profileFor(config, values);
  const target = {
    width: canvas.width,
    height: canvas.height,
    frames: snapFrames(values.duration),
    steps,
    keyframes: values.referenceMode === "omni"
      ? values.references.reduce((sum, reference) => sum + (reference.kind === "video" ? 5 : 1), 0)
      : Number(values.image != null) + Number(values.lastImage != null),
  };
  const targetWork = work(target);
  const matches = samples
    .filter((sample) => sample.profile === profile && sample.upsample === values.upsample)
    .map((sample) => {
      const ratio = targetWork / work(sample);
      const distance = Math.abs(Math.log(ratio)) + Math.abs(sample.keyframes - target.keyframes) * 0.35;
      return {
        seconds: sample.seconds * Math.max(0.25, Math.min(4, ratio)),
        weight: Math.exp(-1.5 * distance),
      };
    })
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 8);
  if (matches.length === 0) return null;
  const totalWeight = matches.reduce((sum, sample) => sum + sample.weight, 0);
  const seconds = matches.reduce((sum, sample) => sum + sample.seconds * sample.weight, 0) / totalWeight;
  return { seconds: Math.max(1, seconds), samples: matches.length };
}
