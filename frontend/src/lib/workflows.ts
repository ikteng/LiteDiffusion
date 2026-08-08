import type { GenerationRecipe, GenerationValues, HistoryItem, StudioConfig } from "../types";
import { findCanvas, randomSeed } from "./studio";

const TURBO_DRAFT = "Turbo 4-step — fastest, more artifacts";
const EXACT_FINAL = "Exact 28-step — maximum fidelity";

function fileFromBlob(blob: Blob | null | undefined, name: string): File | null {
  if (!blob) return null;
  return blob instanceof File ? blob : new File([blob], name, { type: blob.type || "image/png" });
}

/** Store every authored setting so a clip is a reusable project, not merely an MP4. */
export function recipeFrom(values: GenerationValues): GenerationRecipe {
  const { image: _image, lastImage: _lastImage, references: _references, ...recipe } = values;
  return recipe;
}

function legacyRecipe(item: HistoryItem, config: StudioConfig): GenerationRecipe {
  return {
    prompt: item.prompt,
    canvas: findCanvas(config, item.canvas.label).label,
    duration: item.seconds,
    seed: item.seed,
    upsample: false,
    preset: config.presets.some((preset) => preset.value === item.preset) ? item.preset : config.default_preset,
    steps: 28,
    acceleration: "Balanced",
    loraPreset: "None",
    loraRepo: "",
    loraFilename: "",
    loraStrength: 1,
    referenceMode: "keyframes",
  };
}

export function valuesFromHistory(item: HistoryItem, config: StudioConfig): GenerationValues {
  const recipe = item.recipe ?? legacyRecipe(item, config);
  return {
    ...recipe,
    canvas: findCanvas(config, recipe.canvas).label,
    image: fileFromBlob(item.sourceImage, "first-frame.png"),
    lastImage: fileFromBlob(item.sourceLastImage, "last-frame.png"),
    references: (item.sourceReferences ?? []).map((reference, index) => ({
      id: `${item.id}-ref-${index}`,
      file: new File([reference.blob], reference.name, { type: reference.type }),
      kind: reference.type.startsWith("video/") ? "video" : reference.type.startsWith("audio/") ? "audio" : "image",
    })),
  };
}

export function draftValues(values: GenerationValues, config: StudioConfig): GenerationValues {
  return {
    ...values,
    preset: config.presets.some((preset) => preset.value === TURBO_DRAFT) ? TURBO_DRAFT : values.preset,
  };
}

export function finalValues(item: HistoryItem, config: StudioConfig): GenerationValues {
  return {
    ...valuesFromHistory(item, config),
    preset: config.presets.some((preset) => preset.value === EXACT_FINAL) ? EXACT_FINAL : config.default_preset,
    steps: 28,
    acceleration: "Exact",
    loraPreset: "None",
    loraRepo: "",
    loraFilename: "",
    loraStrength: 1,
  };
}

export function remixValues(item: HistoryItem, config: StudioConfig): GenerationValues {
  return { ...valuesFromHistory(item, config), seed: randomSeed() };
}

/** Decode only the final frame in-browser; no upload or extra model/API call is involved. */
export async function finalFrameFile(url: string): Promise<File> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not read this clip (${response.status}).`);
  const objectUrl = URL.createObjectURL(await response.blob());
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.src = objectUrl;

  try {
    await new Promise<void>((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = () => reject(new Error("The browser could not decode this clip."));
      video.load();
    });
    const target = Math.max(0, video.duration - Math.min(0.04, 1 / 48));
    await new Promise<void>((resolve, reject) => {
      video.onseeked = () => resolve();
      video.onerror = () => reject(new Error("The browser could not seek to the final frame."));
      video.currentTime = target;
    });
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0);
    const blob = await new Promise<Blob>((resolve, reject) =>
      canvas.toBlob((result) => (result ? resolve(result) : reject(new Error("Could not capture the final frame."))), "image/png"),
    );
    return new File([blob], "continued-from-final-frame.png", { type: "image/png" });
  } finally {
    video.removeAttribute("src");
    video.load();
    URL.revokeObjectURL(objectUrl);
  }
}
