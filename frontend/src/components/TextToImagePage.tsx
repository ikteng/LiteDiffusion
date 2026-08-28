import { useEffect, useState } from "react";
import { Sparkles, Loader2, Download, ImageOff, Dices } from "lucide-react";
import { api } from "../api";
import { useJobPolling } from "../hooks/useJobPolling";
import type { ModelInfo } from "../types";
import SettingsPanel from "./SettingsPanel";
import GalleryTab from "./GalleryTab";

const EXAMPLE_PROMPTS = [
  "A watercolor fox in a snowy forest",
  "Neon-lit cyberpunk alley in the rain",
  "Cozy cabin, autumn leaves, warm light",
  "Astronaut riding a horse on Mars",
];

export default function TextToImagePage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [prompt, setPrompt] = useState("");
  const [modelKey, setModelKey] = useState("");
  const [seed, setSeed] = useState(0);
  const [randomSeed, setRandomSeed] = useState(true);
  const [jobId, setJobId] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [galleryRefreshKey, setGalleryRefreshKey] = useState(0);

  const job = useJobPolling(jobId);
  const isRunning = job?.status === "queued" || job?.status === "running";
  const selectedModel = models.find((m) => m.key === modelKey);

  useEffect(() => {
    api.getModels().then((res) => {
      setModels(res.models);
      setModelKey(res.default);
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
      const res = await api.generate({ prompt, model: modelKey, seed: randomSeed ? -1 : seed });
      setJobId(res.id);
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Failed to submit job");
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
              placeholder="A watercolor fox in a snowy forest"
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
            <select
              value={modelKey}
              onChange={(e) => setModelKey(e.target.value)}
              className="bg-zinc-800 rounded-lg p-2.5 text-sm outline-none focus:ring-2 focus:ring-violet-600 cursor-pointer"
            >
              {models.map((m) => (
                <option key={m.key} value={m.key}>
                  {m.label}
                </option>
              ))}
            </select>
            {selectedModel && (
              <p className="text-xs text-zinc-500">
                {selectedModel.steps} step{selectedModel.steps === 1 ? "" : "s"} · {selectedModel.size}×
                {selectedModel.size}px
              </p>
            )}
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
            {isRunning ? <Loader2 size={16} className="animate-spin" /> : <Sparkles size={16} />}
            {isRunning ? "Generating…" : "Generate"}
          </button>

          {submitError && (
            <p className="text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
              {submitError}
            </p>
          )}
        </div>

        <div className="flex flex-col items-center justify-center bg-zinc-900 border border-zinc-800 rounded-2xl p-4 min-h-[320px] gap-3">
          {job?.status === "succeeded" && job.result && (
            <>
              <img src={job.result.file_url} alt={job.prompt} className="rounded-lg max-w-full max-h-[420px]" />
              <div className="flex items-center gap-3 text-xs text-zinc-500">
                <span>
                  {job.result.elapsed_seconds.toFixed(1)}s · seed {job.result.seed}
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
              <p className="text-sm">Your image will appear here</p>
            </div>
          )}
        </div>
      </div>

      <SettingsPanel />

      <GalleryTab refreshKey={galleryRefreshKey} />
    </div>
  );
}
