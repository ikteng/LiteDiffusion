import type {
  GenerateRequest,
  HistoryListResponse,
  JobResponse,
  ModelListResponse,
  SettingsResponse,
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed: ${res.status}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const api = {
  generate: (body: GenerateRequest) =>
    request<JobResponse>("/api/generate", { method: "POST", body: JSON.stringify(body) }),
  getJob: (id: string) => request<JobResponse>(`/api/jobs/${id}`),
  getHistory: (limit = 50, offset = 0) =>
    request<HistoryListResponse>(`/api/history?limit=${limit}&offset=${offset}`),
  deleteHistory: (id: string) => request<void>(`/api/history/${id}`, { method: "DELETE" }),
  getModels: () => request<ModelListResponse>("/api/models"),
  getSettings: () => request<SettingsResponse>("/api/settings"),
};
