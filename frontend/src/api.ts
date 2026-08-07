import { Client, handle_file, type StatusMessage } from "@gradio/client";
import type {
  Acceleration,
  GeneratedVideo,
  GenerationValues,
  PresetOption,
  RunProgress,
  RuntimeEstimate,
  StudioConfig,
} from "./types";
import { FALLBACK_CONFIG } from "./types";

type FilePayload = { url?: string; path?: string; name?: string; data?: string };

let clientPromise: Promise<Client> | null = null;

function getClient() {
  clientPromise ??= Client.connect(window.location.origin, { events: ["data", "status"] });
  return clientPromise;
}

/**
 * `steps` and `acceleration` are what let the studio price each preset before it is chosen. Older deployments of
 * `/studio-config` do not send them, so fall back to the known values rather than showing every preset the same cost.
 */
function withPresetSchedule(preset: PresetOption): PresetOption {
  if (typeof preset.steps === "number" && preset.acceleration) return preset;
  const known = FALLBACK_CONFIG.presets.find((item) => item.value === preset.value);
  return {
    ...preset,
    steps: preset.steps ?? known?.steps ?? 28,
    acceleration: preset.acceleration ?? known?.acceleration ?? ("Balanced" as Acceleration),
  };
}

export async function fetchStudioConfig(): Promise<StudioConfig> {
  const response = await fetch("/studio-config");
  if (!response.ok) throw new Error(`Studio configuration failed (${response.status}).`);
  const config = (await response.json()) as StudioConfig;
  return { ...config, presets: config.presets.map(withPresetSchedule) };
}

export async function fetchModelStatus(): Promise<{ ready: boolean; status: string }> {
  const response = await fetch("/status");
  if (!response.ok) throw new Error(`Model status failed (${response.status}).`);
  return response.json() as Promise<{ ready: boolean; status: string }>;
}

function statusLabel(message: StatusMessage, startedAt: number, previous: RunProgress): RunProgress {
  const items = message.progress_data ?? [];
  const isProgressPacket = message.progress_data !== undefined;
  const progressItem = [...items].reverse().find(
    (item) => item.desc || item.progress != null || (item.index != null && item.length != null),
  );
  if (message.stage === "pending" && !isProgressPacket) {
    const position = message.position;
    return {
      stage: "queued",
      label: position != null ? `Queued · ${position + 1} ahead` : "Waiting for a GPU",
      progress: null,
      position,
      eta: message.eta,
      etaUpdatedAt: Date.now(),
      etaCountsDown: false,
      etaSource: "queue",
      exact: false,
      phase: "queue",
    };
  }
  // Gradio 6 emits explicit gr.Progress packets with stage="pending" even after execution starts.
  if (message.stage === "generating" || message.stage === "streaming" || isProgressPacket) {
    const trackedStep = progressItem?.index != null && progressItem.length != null && progressItem.length > 0;
    const rawFraction = progressItem?.progress ?? (trackedStep ? progressItem.index! / progressItem.length! : null);
    const gpuInitialization = progressItem?.desc === "ZeroGPU init";
    const denoising = trackedStep && !gpuInitialization;
    const gpuPreflight = !trackedStep && progressItem?.desc?.toLowerCase().includes("denoising") === true;
    let phase: RunProgress["phase"] = denoising
      ? "denoising"
      : gpuInitialization || gpuPreflight
        ? "gpu"
        : progressItem
          ? "conditioning"
          : previous.phase;
    let fraction = rawFraction;
    // Keep the full-job bar monotonic while still showing exact denoising counts separately.
    if (rawFraction != null && gpuInitialization) fraction = 0.1 + rawFraction * 0.1;
    if (rawFraction != null && denoising) fraction = 0.2 + rawFraction * 0.7;
    if (!progressItem && previous.phase === "denoising") {
      fraction = 0.9;
      phase = "finalizing";
    } else if (!progressItem && previous.phase === "gpu") {
      fraction = Math.max(previous.progress ?? 0, 0.2);
      phase = "conditioning";
    }
    const exact = fraction != null;
    if (fraction == null && !progressItem && previous.stage === "generating") {
      fraction = previous.progress;
    }
    if (fraction == null && message.eta != null && message.eta > 0) {
      const elapsed = (Date.now() - startedAt) / 1000;
      fraction = elapsed / (elapsed + message.eta);
    }
    if (fraction != null) fraction = Math.max(0, Math.min(1, fraction));
    return {
      stage: "generating",
      label: denoising
        ? progressItem?.desc || "Denoising video and audio"
        : phase === "finalizing"
          ? "Finalizing video and audio"
          : phase === "gpu"
            ? "Preparing the GPU worker"
            : phase === "conditioning" && !progressItem
              ? "Conditioning the prompt"
              : progressItem?.desc || previous.label || "Generating video and sound",
      progress: fraction,
      eta: message.eta,
      index: progressItem?.index ?? undefined,
      length: progressItem?.length ?? undefined,
      unit: progressItem?.unit ?? undefined,
      exact,
      phase,
    };
  }
  if (message.stage === "complete") {
    return { stage: "complete", label: "Generation complete", progress: 1, exact: true };
  }
  if (message.stage === "error") {
    const detail = typeof message.message === "string" ? message.message : "Generation failed.";
    return { stage: "error", label: detail, progress: null, exact: false };
  }
  return { stage: "connecting", label: "Connecting to the generator", progress: null, exact: false };
}

function outputUrl(file: FilePayload | string): string {
  if (typeof file === "string") {
    if (/^(https?:|blob:|data:)/.test(file)) return file;
    return `/gradio_api/file=${encodeURIComponent(file)}`;
  }
  if (file.url) return file.url;
  if (file.data) return file.data;
  if (file.path) return `/gradio_api/file=${encodeURIComponent(file.path)}`;
  throw new Error("Generation completed without a playable video URL.");
}

function unwrapServerOutput(payload: unknown[]): unknown[] {
  let output = payload;
  while (output.length === 1 && Array.isArray(output[0])) output = output[0];
  return output;
}

export async function runGeneration(
  values: GenerationValues,
  onProgress: (progress: RunProgress) => void,
  runtimeEstimate: RuntimeEstimate | null = null,
): Promise<GeneratedVideo> {
  onProgress({
    stage: "connecting",
    label: "Connecting to the generator",
    progress: null,
    exact: false,
    eta: runtimeEstimate?.seconds,
    etaUpdatedAt: Date.now(),
    etaCountsDown: false,
    etaSamples: runtimeEstimate?.samples,
    etaSource: runtimeEstimate ? "history" : undefined,
  });
  const startedAt = Date.now();
  let runtimeStartedAt: number | null = null;
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
  let latestProgress: RunProgress = {
    stage: "connecting",
    label: "Connecting to the generator",
    progress: null,
    exact: false,
  };
  for await (const event of submission) {
    if (event.type === "status") {
      const next = statusLabel(event, startedAt, latestProgress);
      if (next.stage === "generating" && next.phase !== "queue") {
        runtimeStartedAt ??= Date.now();
        if (runtimeEstimate) {
          next.eta = Math.max(0, runtimeEstimate.seconds - (Date.now() - runtimeStartedAt) / 1000);
          next.etaUpdatedAt = Date.now();
          next.etaCountsDown = true;
          next.etaSamples = runtimeEstimate.samples;
          next.etaSource = "history";
        }
      } else if (runtimeEstimate && next.stage === "queued") {
        // Queue ETA describes worker availability. Keep the learned generation ETA separate and frozen until work starts.
        next.eta = runtimeEstimate.seconds;
        next.etaUpdatedAt = Date.now();
        next.etaCountsDown = false;
        next.etaSamples = runtimeEstimate.samples;
        next.etaSource = "history";
      }
      latestProgress = next;
      onProgress(next);
      if (next.stage === "error") throw new Error(next.label);
    } else if (event.type === "data") {
      result = Array.isArray(event.data) ? event.data : [event.data];
    }
  }

  if (!result) throw new Error("The generator finished without returning a result.");
  result = unwrapServerOutput(result);
  if (result.length < 3) throw new Error("The generator returned an incomplete response.");
  return {
    url: outputUrl(result[0] as FilePayload | string),
    report: String(result[1] ?? ""),
    refinedPrompt: String(result[2] ?? ""),
    runtimeSeconds: (Date.now() - (runtimeStartedAt ?? startedAt)) / 1000,
  };
}
