export type CanvasOption = {
  label: string;
  height: number;
  width: number;
  fast: boolean;
};

export type PresetOption = {
  value: string;
  description: string;
  recommended: boolean;
  custom: boolean;
};

export type PromptExample = {
  title: string;
  prompt: string;
  canvas: string;
};

export type StudioConfig = {
  canvases: CanvasOption[];
  default_canvas: string;
  duration: { min: number; max: number; default: number };
  presets: PresetOption[];
  default_preset: string;
  custom_preset: string;
  examples: PromptExample[];
};

export type GenerationValues = {
  prompt: string;
  image: File | null;
  lastImage: File | null;
  canvas: string;
  duration: number;
  seed: number;
  upsample: boolean;
  preset: string;
  steps: number;
  acceleration: "Balanced" | "Ultra Fast" | "Exact";
  loraPreset: "None" | "Turbo · 4 steps" | "Turbo · 8 steps" | "Custom";
  loraRepo: string;
  loraFilename: string;
  loraStrength: number;
};

export type GeneratedVideo = {
  url: string;
  report: string;
  refinedPrompt: string;
};

export type RunProgress = {
  stage: "idle" | "connecting" | "queued" | "generating" | "complete" | "error";
  label: string;
  progress: number | null;
  position?: number;
  eta?: number;
};

export const FALLBACK_CONFIG: StudioConfig = {
  canvases: [
    { label: "960x544 · 16:9 fast", width: 960, height: 544, fast: true },
    { label: "1024x576 · 16:9 fast", width: 1024, height: 576, fast: true },
    { label: "1152x640 · 16:9", width: 1152, height: 640, fast: false },
    { label: "1280x704 · 16:9", width: 1280, height: 704, fast: false },
    { label: "1344x768 · 16:9 full", width: 1344, height: 768, fast: false },
    { label: "544x960 · 9:16 fast", width: 544, height: 960, fast: true },
    { label: "640x1152 · 9:16", width: 640, height: 1152, fast: false },
    { label: "768x1344 · 9:16 full", width: 768, height: 1344, fast: false },
    { label: "544x544 · 1:1 fast", width: 544, height: 544, fast: true },
    { label: "768x768 · 1:1 full", width: 768, height: 768, fast: false },
  ],
  default_canvas: "960x544 · 16:9 fast",
  duration: { min: 2, max: 14, default: 5 },
  presets: [
    {
      value: "Balanced — best overall (recommended)",
      description: "Full quality schedule with conservative Cache-DiT acceleration.",
      recommended: true,
      custom: false,
    },
    {
      value: "Turbo 8-step — faster, cleaner",
      description: "Distilled eight-step path with better Turbo consistency.",
      recommended: false,
      custom: false,
    },
    {
      value: "Turbo 4-step — fastest, more artifacts",
      description: "Maximum speed; expect sharper textures and more motion artifacts.",
      recommended: false,
      custom: false,
    },
    {
      value: "Exact 28-step — maximum fidelity",
      description: "Dense reference path with approximate caching disabled.",
      recommended: false,
      custom: false,
    },
    {
      value: "Ultra cache — experimental speed",
      description: "Aggressive forecasting on the full schedule; inspect results carefully.",
      recommended: false,
      custom: false,
    },
    {
      value: "Custom — manual controls",
      description: "Expose schedule, cache engine, LoRA source, and strength controls.",
      recommended: false,
      custom: true,
    },
  ],
  default_preset: "Balanced — best overall (recommended)",
  custom_preset: "Custom — manual controls",
  examples: [
    {
      title: "Snow fox",
      prompt: "A red fox trotting through a snowy pine forest at dawn, snow crunching underfoot",
      canvas: "960x544 · 16:9 fast",
    },
    {
      title: "Night market",
      prompt: "A busy night market, neon signs reflecting in puddles, sizzling street food",
      canvas: "544x960 · 9:16 fast",
    },
    {
      title: "Concert hall",
      prompt: "A cellist playing a slow melody in an empty concert hall",
      canvas: "544x544 · 1:1 fast",
    },
  ],
};
