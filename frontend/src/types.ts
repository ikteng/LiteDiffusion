export type MediaType = "image" | "video";
export type JobStatus = "queued" | "running" | "succeeded" | "failed";

export interface GenerateRequest {
  prompt: string;
  model: string;
  seed?: number | null;
  media_type?: MediaType;
}

export interface JobResult {
  media_type: MediaType;
  file_url: string;
  width: number;
  height: number;
  seed: number;
  elapsed_seconds: number;
  frames?: number | null;
  fps?: number | null;
}

export interface JobResponse {
  id: string;
  status: JobStatus;
  media_type: MediaType;
  prompt: string;
  model: string;
  created_at: number;
  error?: string | null;
  result?: JobResult | null;
}

export interface HistoryItem {
  id: string;
  media_type: MediaType;
  prompt: string;
  model: string;
  seed: number;
  width: number;
  height: number;
  elapsed_seconds: number;
  created_at: number;
  file: string;
  file_url: string;
  frames?: number | null;
  fps?: number | null;
}

export interface HistoryListResponse {
  items: HistoryItem[];
  total: number;
}

export interface ModelInfo {
  key: string;
  label: string;
  repo: string;
  steps: number;
  guidance_scale: number;
  size: number;
  kind: string;
  quantized: boolean;
  pipeline: string;
  frame_arg: string;
  remote: boolean;
}

export interface ModelListResponse {
  models: ModelInfo[];
  default: string;
}

export interface SettingsResponse {
  device: string;
  dtype: string;
}
