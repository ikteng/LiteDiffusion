import { Client, handle_file, type StatusMessage } from "@gradio/client";
import type { GeneratedVideo, GenerationValues, RunProgress, StudioConfig } from "./types";

type FilePayload = { url?: string; path?: string; name?: string };

let clientPromise: Promise<Client> | null = null;

function getClient() {
  clientPromise ??= Client.connect(window.location.origin);
  return clientPromise;
}

export async function fetchStudioConfig(): Promise<StudioConfig> {
  const response = await fetch("/studio-config");
  if (!response.ok) throw new Error(`Studio configuration failed (${response.status}).`);
  return response.json() as Promise<StudioConfig>;
}

export async function fetchModelStatus(): Promise<{ ready: boolean; status: string }> {
  const response = await fetch("/status");
  if (!response.ok) throw new Error(`Model status failed (${response.status}).`);
  return response.json() as Promise<{ ready: boolean; status: string }>;
}

function statusLabel(message: StatusMessage): RunProgress {
  const progressItem = message.progress_data?.at(-1);
  if (message.stage === "pending") {
    const position = message.position;
    return {
      stage: "queued",
      label: position != null ? `Queued · ${position + 1} ahead` : "Waiting for a GPU",
      progress: null,
      position,
      eta: message.eta,
    };
  }
  if (message.stage === "generating" || message.stage === "streaming") {
    return {
      stage: "generating",
      label: progressItem?.desc || "Generating video and sound",
      progress: progressItem?.progress ?? null,
      eta: message.eta,
    };
  }
  if (message.stage === "complete") {
    return { stage: "complete", label: "Generation complete", progress: 1 };
  }
  if (message.stage === "error") {
    const detail = typeof message.message === "string" ? message.message : "Generation failed.";
    return { stage: "error", label: detail, progress: null };
  }
  return { stage: "connecting", label: "Connecting to the generator", progress: null };
}

function outputUrl(file: FilePayload): string {
  if (file.url) return file.url;
  if (file.path) return `/gradio_api/file=${encodeURIComponent(file.path)}`;
  throw new Error("Generation completed without a playable video URL.");
}

export async function runGeneration(
  values: GenerationValues,
  onProgress: (progress: RunProgress) => void,
): Promise<GeneratedVideo> {
  onProgress({ stage: "connecting", label: "Connecting to the generator", progress: null });
  const client = await getClient();
  const submission = client.submit("/generate", {
    prompt: values.prompt,
    image_path: values.image ? handle_file(values.image) : null,
    last_image_path: values.lastImage ? handle_file(values.lastImage) : null,
    canvas: values.canvas,
    duration: values.duration,
    steps: values.steps,
    seed: values.seed,
    upsample: values.upsample,
    acceleration: values.acceleration,
    lora_preset: values.loraPreset,
    lora_repo: values.loraRepo,
    lora_filename: values.loraFilename,
    lora_strength: values.loraStrength,
    generation_preset: values.preset,
  });

  let result: unknown[] | null = null;
  for await (const event of submission) {
    if (event.type === "status") {
      const next = statusLabel(event);
      onProgress(next);
      if (next.stage === "error") throw new Error(next.label);
    } else if (event.type === "data") {
      result = event.data as unknown[];
    }
  }

  if (!result || result.length < 3) throw new Error("The generator returned an incomplete response.");
  return {
    url: outputUrl(result[0] as FilePayload),
    report: String(result[1] ?? ""),
    refinedPrompt: String(result[2] ?? ""),
  };
}
