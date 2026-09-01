import { useEffect, useState } from "react";
import { Video, Sparkles, Loader2, Download, ImageOff, Dices, Clock } from "lucide-react";
import { api } from "../api";
import { useJobPolling } from "../hooks/useJobPolling";
import { useModelDownloads } from "../hooks/useModelDownloads";
import type { ModelInfo } from "../types";
import GalleryTab from "./GalleryTab";
import CustomSelect from "./CustomSelect";

const EXAMPLE_PROMPTS = [
  "A watercolor fox wandering through a snowy forest",
  "Neon-lit cyberpunk alley, rain rippling on the ground",
  "Cozy cabin in autumn, leaves drifting in warm light",
  "Astronaut floating above Mars, slow gentle drift",
];

const DURATION_OPTIONS = [
  { value: "2", label: "2 seconds" },
  { value: "4", label: "4 seconds" },
  { value: "5", label: "5 seconds" },
  { value: "8", label: "8 seconds" },
  { value: "10", label: "10 seconds" },
  { value: "12", label: "12 seconds" },
];

export default function TextToVideoPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [prompt, setPrompt] = useState("");
  const [modelKey, setModelKey] = useState("");
  const [duration, setDuration] = useState(5);
  const [seed, setSeed] = useState(0);
  const [randomSeed, setRandomSeed] = useState(true);
  const [jobId, setJobId] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [galleryRefreshKey, setGalleryRefreshKey] = useState(0);

  const job = useJobPolling(jobId);
  const isRunning = job?.status === "queued" || job?.status === "running";
  const selectedModel = models.find((m) => m.key === modelKey);
  const { downloadModel, getStatus } = useModelDownloads();

  const videoGroups = [
    {
      label: "Local",
      options: models
        .filter((m) => m.kind === "video" && !m.remote)
        .map((m) => {
          const status = getStatus(m.key) as "idle" | "downloading" | "ready";
          return {
            value: m.key,
            label: m.label,
            downloadStatus: status === "ready" ? undefined : status,
            onDownload: () => downloadModel(m.key),
          };
        }),
    },
    {
      label: "Remote",
      options: models
        .filter((m) => m.kind === "video" && m.remote)
        .map((m) => ({ value: m.key, label: m.label })),
    },
  ];

  useEffect(() => {
    api.getModels().then((res) => {
      setModels(res.models);
      const defaultModel = res.models.find((m) => m.kind === "video") || res.models[0];
      setModelKey(defaultModel?.key ?? "");
    });
  }, []);

  useEffect(() => {
    if (job?.status === "succeeded") {
      setGalleryRefreshKey((k) => k + 1);
    }
  }, [job?.status]);

  async function handleGenerate() {
    if (!prompt.trim() || isRunning) return;
    setSubmitError(null);
    try {
      const res = await api.generate({
        prompt,
        model: modelKey,
        seed: randomSeed ? -1 : seed,
        media_type: "video",
        duration,
      });
      setJobId(res.id);
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Failed to submit job");
    }
  }

  async function handleCancel() {
    if (!jobId) return;
    try {
      await api.cancelJob(jobId);
    } catch {
      setSubmitError("Failed to cancel job");
    }
  }

  return (
    <div className="flex flex-col gap-6 p-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="flex flex-col gap-4 bg-zinc-900 border border-zinc-800 rounded-2xl p-5">
          <div className="flex flex-col gap-1.5">
            <label className="text-sm font-medium text-zinc-300">Prompt</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="A watercolor fox wandering through a snowy forest"
              rows={3}
              className="bg-zinc-800 rounded-lg p-3 text-sm outline-none focus:ring-2 focus:ring-violet-600 placeholder:text-zinc-600"
            />
            <div className="flex flex-wrap gap-1.5">
              {EXAMPLE_PROMPTS.map((p) => (
                <button
                  key={p}
                  type="button"
                  onClick={() => setPrompt(p)}
                  className="text-xs text-zinc-400 bg-zinc-800/70 hover:bg-zinc-800 hover:text-zinc-200 rounded-full px-2.5 py-1 transition-colors cursor-pointer"
                >
                  {p}
                </button>
              ))}
            </div>
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-sm font-medium text-zinc-300">Model</label>
            <CustomSelect
              value={modelKey}
              onChange={setModelKey}
              groups={videoGroups}
              placeholder="Select a model"
            />
            {selectedModel && (
              <p className="text-xs text-zinc-500">
                {selectedModel.steps} step{selectedModel.steps === 1 ? "" : "s"} · {selectedModel.size}×
                {selectedModel.size}px
                {selectedModel.quantized && " · 4-bit quantized"}
                {selectedModel.remote && " · remote"}
              </p>
            )}
            {selectedModel && selectedModel.remote && (
              <p className="text-xs text-zinc-500">{selectedModel.repo}</p>
            )}
            <p className="text-xs text-zinc-500">
              {selectedModel?.kind === "video"
                ? selectedModel?.remote
                  ? "Generates via free online inference API — no local download. Results may be rate-limited."
                  : "Generates video frames directly from the prompt using a distilled T2V model. Slower on CPU, better motion."
                : "Generates a few keyframes, then uses motion-compensated interpolation for a smooth clip. Fast and CPU-friendly — no extra model download."}
            </p>

            <div className="flex flex-col gap-1.5">
              <label className="text-sm font-medium text-zinc-300 flex items-center gap-2">
                <Clock size={14} />
                Duration
              </label>
              <CustomSelect
                value={duration.toString()}
                onChange={(v) => setDuration(Number(v))}
                groups={[{ label: "Duration", options: DURATION_OPTIONS }]}
                placeholder="Select duration"
              />
              <p className="text-xs text-zinc-500">
                {duration * 12} frames at 12 fps
              </p>
            </div>
          </div>

          <details className="text-sm text-zinc-400 group">
            <summary className="cursor-pointer font-medium text-zinc-300 select-none">Advanced settings</summary>
            <div className="mt-3 flex flex-col gap-2 border-t border-zinc-800 pt-3">
              <label className="flex items-center gap-2 cursor-pointer w-fit">
                <input
                  type="checkbox"
                  checked={randomSeed}
                  onChange={(e) => setRandomSeed(e.target.checked)}
                  className="accent-violet-600"
                />
                <Dices size={14} />
                Random seed
              </label>
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min={0}
                  max={4294967295}
                  step={1}
                  value={seed}
                  disabled={randomSeed}
                  onChange={(e) => setSeed(Number(e.target.value))}
                  className="flex-1 accent-violet-600 disabled:opacity-40 cursor-pointer"
                />
                <input
                  type="number"
                  min={0}
                  max={4294967295}
                  value={seed}
                  disabled={randomSeed}
                  onChange={(e) => setSeed(Number(e.target.value))}
                  className="bg-zinc-800 rounded-lg p-2 text-sm outline-none w-28 disabled:opacity-40"
                />
              </div>
            </div>
          </details>

          <button
            onClick={handleGenerate}
            disabled={isRunning || !prompt.trim()}
            className="mt-1 flex items-center justify-center gap-2 bg-violet-600 hover:bg-violet-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg py-2.5 font-semibold text-sm cursor-pointer transition-colors"
          >
            {isRunning ? <Loader2 size={16} className="animate-spin" /> : <Video size={16} />}
            {isRunning ? "Generating…" : "Generate video"}
          </button>
          {isRunning && (
            <button
              onClick={handleCancel}
              className="mt-1 flex items-center justify-center gap-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-lg py-2.5 font-semibold text-sm cursor-pointer transition-colors border border-zinc-700"
            >
              Stop
            </button>
          )}

          {submitError && (
            <p className="text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
              {submitError}
            </p>
          )}
        </div>

        <div className="flex flex-col items-center justify-center bg-zinc-900 border border-zinc-800 rounded-2xl p-4 min-h-[320px] gap-3">
          {job?.status === "succeeded" && job.result && (
            <>
              {job.result.media_type === "video" ? (
                <video
                  src={job.result.file_url}
                  autoPlay
                  loop
                  muted
                  controls
                  className="rounded-lg max-w-full max-h-[420px]"
                />
              ) : (
                <img src={job.result.file_url} alt={job.prompt} className="rounded-lg max-w-full max-h-[420px]" />
              )}
              <div className="flex items-center gap-3 text-xs text-zinc-500">
                <span>
                  {job.result.elapsed_seconds.toFixed(1)}s · seed {job.result.seed}
                  {job.result.frames ? ` · ${job.result.frames} frames` : ""}
                </span>
                <a
                  href={job.result.file_url}
                  download
                  className="flex items-center gap-1 text-violet-400 hover:text-violet-300"
                >
                  <Download size={12} />
                  Download
                </a>
              </div>
            </>
          )}
          {job?.status === "failed" && (
            <div className="flex flex-col items-center gap-2 text-center">
              <ImageOff className="text-red-400" size={28} />
              <p className="text-sm text-red-400">{job.error}</p>
            </div>
          )}
          {isRunning && (
            <div className="flex flex-col items-center gap-2 text-zinc-400">
              <Loader2 size={28} className="animate-spin text-violet-400" />
              <p className="text-sm capitalize">{job?.status}…</p>
            </div>
          )}
          {!job && (
            <div className="flex flex-col items-center gap-2 text-zinc-600">
              <Sparkles size={28} />
              <p className="text-sm">Your clip will appear here</p>
            </div>
          )}
        </div>
      </div>

      <GalleryTab refreshKey={galleryRefreshKey} />
    </div>
  );
}
