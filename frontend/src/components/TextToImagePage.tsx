import { useEffect, useState } from "react";
import { Sparkles, Loader2, Download, ImageOff, Dices, Square, ChevronDown, SlidersHorizontal } from "lucide-react";
import { api } from "../api";
import { useJobPolling } from "../hooks/useJobPolling";
import { useModelDownloads } from "../hooks/useModelDownloads";
import type { ModelInfo } from "../types";
import { formatModelSize, sortModelsBySize } from "../utils";
import SettingsPanel from "./SettingsPanel";
import GalleryTab from "./GalleryTab";
import CustomSelect from "./CustomSelect";

const EXAMPLE_PROMPTS = [
  "A watercolor fox in a snowy forest",
  "Neon-lit cyberpunk alley in the rain",
  "Cozy cabin, autumn leaves, warm light",
  "Astronaut riding a horse on Mars",
];

export default function TextToImagePage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [modelKey, setModelKey] = useState("");
  const [seed, setSeed] = useState(0);
  const [randomSeed, setRandomSeed] = useState(true);
  const [steps, setSteps] = useState<number | "">("");
  const [guidanceScale, setGuidanceScale] = useState<number | "">("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [galleryRefreshKey, setGalleryRefreshKey] = useState(0);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const job = useJobPolling(jobId);
  const isRunning = job?.status === "queued" || job?.status === "running";
  const selectedModel = models.find((m) => m.key === modelKey);
  const { downloadModel, getStatus } = useModelDownloads();

  const imageGroups = [
    {
      label: "",
      options: sortModelsBySize(models.filter((m) => m.kind === "image" && !m.remote))
        .map((m) => {
          const status = getStatus(m.key) as "idle" | "downloading" | "ready";
          return {
            value: m.key,
            label: m.label,
            sublabel: formatModelSize(m.approx_size_mb),
            downloadStatus: status === "ready" ? undefined : status,
            onDownload: () => downloadModel(m.key),
          };
        }),
    },
  ];

  useEffect(() => {
    api.getModels().then((res) => {
      setModels(res.models);
      const defaultModel =
        res.models.find((m) => m.key === res.default && m.kind === "image") ||
        res.models.find((m) => m.kind === "image") ||
        res.models[0];
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
        negative_prompt: negativePrompt.trim() || null,
        steps: steps === "" ? null : steps,
        guidance_scale: guidanceScale === "" ? null : guidanceScale,
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
            <CustomSelect
              value={modelKey}
              onChange={setModelKey}
              groups={imageGroups}
              placeholder="Select a model"
            />
            {selectedModel && (
              <p className="text-xs text-zinc-500">
                {selectedModel.steps} step{selectedModel.steps === 1 ? "" : "s"} · {selectedModel.size}×
                {selectedModel.size}px
                {selectedModel.approx_size_mb > 0 && ` · ${formatModelSize(selectedModel.approx_size_mb)}`}
                {selectedModel.remote && " · remote"}
              </p>
            )}
            {selectedModel && selectedModel.remote && (
              <p className="text-xs text-zinc-500">{selectedModel.repo}</p>
            )}
          </div>

          <div className="bg-zinc-800/40 border border-zinc-800 rounded-xl overflow-hidden">
            <button
              type="button"
              onClick={() => setAdvancedOpen((v) => !v)}
              className="w-full flex items-center justify-between p-3 cursor-pointer"
            >
              <span className="flex items-center gap-2 text-sm font-medium text-zinc-300">
                <SlidersHorizontal size={14} />
                Advanced settings
              </span>
              <ChevronDown
                size={16}
                className={`text-zinc-500 transition-transform ${advancedOpen ? "rotate-180" : ""}`}
              />
            </button>

            {advancedOpen && (
              <div className="flex flex-col gap-5 border-t border-zinc-800 p-4">
                <div className="flex flex-col gap-1.5">
                  <label className="text-xs font-medium text-zinc-300">Negative prompt</label>
                  <p className="text-xs text-zinc-500">Describe what you don't want to see in the image.</p>
                  <textarea
                    value={negativePrompt}
                    onChange={(e) => setNegativePrompt(e.target.value)}
                    placeholder="blurry, low quality, watermark"
                    rows={2}
                    className="bg-zinc-800 rounded-lg p-2.5 text-sm outline-none focus:ring-2 focus:ring-violet-600 placeholder:text-zinc-600"
                  />
                </div>

                <div className="flex flex-col gap-1.5">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-medium text-zinc-300">Steps</label>
                    <span className="text-xs text-zinc-500 tabular-nums">
                      {steps === "" ? `Default (${selectedModel?.steps ?? "…"})` : steps}
                    </span>
                  </div>
                  <p className="text-xs text-zinc-500">
                    How many refinement passes to run. More can sharpen detail but takes longer; fast models need
                    very few.
                  </p>
                  <div className="flex items-center gap-3">
                    <input
                      type="range"
                      min={1}
                      max={150}
                      step={1}
                      value={steps === "" ? selectedModel?.steps ?? 20 : steps}
                      onChange={(e) => setSteps(Number(e.target.value))}
                      className="flex-1 accent-violet-600 cursor-pointer"
                    />
                    <input
                      type="number"
                      min={1}
                      max={150}
                      value={steps}
                      placeholder={selectedModel ? String(selectedModel.steps) : undefined}
                      onChange={(e) => setSteps(e.target.value === "" ? "" : Number(e.target.value))}
                      className="bg-zinc-800 rounded-lg p-2 text-sm outline-none w-20 shrink-0"
                    />
                  </div>
                </div>

                <div className="flex flex-col gap-1.5">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-medium text-zinc-300">Guidance scale</label>
                    <span className="text-xs text-zinc-500 tabular-nums">
                      {guidanceScale === "" ? `Default (${selectedModel?.guidance_scale ?? "…"})` : guidanceScale}
                    </span>
                  </div>
                  <p className="text-xs text-zinc-500">
                    How closely to follow the prompt. Lower is more creative, higher is more literal (and can look
                    over-processed).
                  </p>
                  <div className="flex items-center gap-3">
                    <input
                      type="range"
                      min={0}
                      max={30}
                      step={0.5}
                      value={guidanceScale === "" ? selectedModel?.guidance_scale ?? 7 : guidanceScale}
                      onChange={(e) => setGuidanceScale(Number(e.target.value))}
                      className="flex-1 accent-violet-600 cursor-pointer"
                    />
                    <input
                      type="number"
                      min={0}
                      max={30}
                      step={0.5}
                      value={guidanceScale}
                      placeholder={selectedModel ? String(selectedModel.guidance_scale) : undefined}
                      onChange={(e) => setGuidanceScale(e.target.value === "" ? "" : Number(e.target.value))}
                      className="bg-zinc-800 rounded-lg p-2 text-sm outline-none w-20 shrink-0"
                    />
                  </div>
                </div>

                <div className="flex flex-col gap-1.5">
                  <label className="text-xs font-medium text-zinc-300">Seed</label>
                  <p className="text-xs text-zinc-500">
                    Same seed + same settings reproduces the same image. Lock it to compare tweaks side by side.
                  </p>
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
                      className="bg-zinc-800 rounded-lg p-2 text-sm outline-none w-28 shrink-0 disabled:opacity-40"
                    />
                  </div>
                  <label className="flex items-center gap-1.5 cursor-pointer text-xs text-zinc-400">
                    <input
                      type="checkbox"
                      checked={randomSeed}
                      onChange={(e) => setRandomSeed(e.target.checked)}
                      className="accent-violet-600"
                    />
                    <Dices size={13} />
                    Random
                  </label>
                </div>
              </div>
            )}
          </div>

          <div className="mt-1 flex items-stretch gap-2">
            <button
              onClick={handleGenerate}
              disabled={isRunning || !prompt.trim()}
              className="flex-1 flex items-center justify-center gap-2 bg-violet-600 hover:bg-violet-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg py-2.5 font-semibold text-sm cursor-pointer transition-colors"
            >
              {isRunning ? <Loader2 size={16} className="animate-spin" /> : <Sparkles size={16} />}
              {isRunning ? "Generating…" : "Generate"}
            </button>
            {isRunning && (
              <button
                onClick={handleCancel}
                title="Stop generation"
                aria-label="Stop generation"
                className="flex items-center justify-center w-10 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-lg cursor-pointer transition-colors border border-zinc-700"
              >
                <Square size={14} fill="currentColor" />
              </button>
            )}
          </div>

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

      <SettingsPanel kind="image" />

      <GalleryTab refreshKey={galleryRefreshKey} />
    </div>
  );
}
