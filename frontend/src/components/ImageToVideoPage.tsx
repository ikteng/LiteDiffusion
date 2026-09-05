import { useEffect, useRef, useState } from "react";
import { Film, Sparkles, Loader2, Download, ImageOff, Dices, Clock, Square, Upload, X } from "lucide-react";
import { api } from "../api";
import { useJobPolling } from "../hooks/useJobPolling";
import { useModelDownloads } from "../hooks/useModelDownloads";
import type { ModelInfo } from "../types";
import { formatModelSize, sortModelsBySize } from "../utils";
import GalleryTab from "./GalleryTab";
import CustomSelect from "./CustomSelect";
import SettingsPanel from "./SettingsPanel";

const MOTION_PROMPTS = [
  "Camera slowly pushes in, gentle breeze",
  "Leaves drift in the wind, soft lighting",
  "Slow pan across the scene, cinematic",
  "Subtle zoom out, warm sunlight",
];

const DURATION_OPTIONS = [
  { value: "2", label: "2 seconds" },
  { value: "4", label: "4 seconds" },
  { value: "5", label: "5 seconds" },
  { value: "8", label: "8 seconds" },
  { value: "10", label: "10 seconds" },
  { value: "12", label: "12 seconds" },
];

const I2V_PIPELINES = [
  "LTXImageToVideoPipeline",
  "Kandinsky5I2VPipeline",
  "WanImageToVideoPipeline",
  "I2VGenXLPipeline",
  "SV3DPipeline",
  "StableVideoDiffusionPipeline",
];

export default function ImageToVideoPage({ mode }: { mode: "single" | "first-last" }) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [prompt, setPrompt] = useState("");
  const [modelKey, setModelKey] = useState("");
  const [duration, setDuration] = useState(5);
  const [seed, setSeed] = useState(0);
  const [randomSeed, setRandomSeed] = useState(true);
  const [jobId, setJobId] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [galleryRefreshKey, setGalleryRefreshKey] = useState(0);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [endImageFile, setEndImageFile] = useState<File | null>(null);
  const [endImagePreviewUrl, setEndImagePreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const endFileInputRef = useRef<HTMLInputElement>(null);

  const job = useJobPolling(jobId);
  const isRunning = job?.status === "queued" || job?.status === "running";
  const selectedModel = models.find((m) => m.key === modelKey);
  const { downloadModel, getStatus } = useModelDownloads();
  const supportsEndFrame = mode === "first-last" && selectedModel?.pipeline === "LTXImageToVideoPipeline";
  const promptRequired = selectedModel?.requires_prompt ?? true;

  const i2vModels = models.filter(
    (m) =>
      I2V_PIPELINES.includes(m.pipeline) &&
      !m.remote &&
      (mode !== "first-last" || m.pipeline === "LTXImageToVideoPipeline")
  );
  const modelGroups = [
    {
      label: "",
      options: sortModelsBySize(i2vModels).map((m) => {
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
      updateDefaultModel(res.models);
    });
  }, []);

  useEffect(() => {
    if (models.length > 0) {
      updateDefaultModel(models);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  function updateDefaultModel(allModels: ModelInfo[]) {
    const available = allModels.filter(
      (m) =>
        I2V_PIPELINES.includes(m.pipeline) &&
        !m.remote &&
        (mode !== "first-last" || m.pipeline === "LTXImageToVideoPipeline")
    );
    const current = available.find((m) => m.key === modelKey);
    if (!current && available.length > 0) {
      const def = available.find((m) => m.label.includes("Recommended")) || available[0];
      setModelKey(def?.key ?? "");
    }
  }

  useEffect(() => {
    if (job?.status === "succeeded") {
      setGalleryRefreshKey((k) => k + 1);
    }
  }, [job?.status]);

  useEffect(() => {
    return () => {
      if (imagePreviewUrl) URL.revokeObjectURL(imagePreviewUrl);
      if (endImagePreviewUrl) URL.revokeObjectURL(endImagePreviewUrl);
    };
  }, [imagePreviewUrl, endImagePreviewUrl]);

  useEffect(() => {
    if (!supportsEndFrame) {
      handleEndFileSelect(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [supportsEndFrame]);

  function handleFileSelect(file: File | null) {
    if (imagePreviewUrl) URL.revokeObjectURL(imagePreviewUrl);
    if (!file) {
      setImageFile(null);
      setImagePreviewUrl(null);
      return;
    }
    setImageFile(file);
    setImagePreviewUrl(URL.createObjectURL(file));
  }

  function handleEndFileSelect(file: File | null) {
    setEndImagePreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return file ? URL.createObjectURL(file) : null;
    });
    setEndImageFile(file);
  }

  const canGenerate = (!promptRequired || prompt.trim()) && imageFile;

  async function handleGenerate() {
    if (!canGenerate || isRunning) return;
    setSubmitError(null);
    try {
      const formData = new FormData();
      formData.append("prompt", prompt);
      formData.append("model", modelKey);
      formData.append("seed", randomSeed ? "-1" : seed.toString());
      formData.append("media_type", "video");
      formData.append("duration", duration.toString());
      if (imageFile) formData.append("image", imageFile);
      if (supportsEndFrame && endImageFile) {
        formData.append("end_image", endImageFile);
      }
      const res = await api.generate(formData);
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
            <label className="text-sm font-medium text-zinc-300 flex items-center justify-between">
              <span>Start image</span>
              <span className="text-xs text-violet-400 font-medium">Required</span>
            </label>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => handleFileSelect(e.target.files?.[0] ?? null)}
            />
            {imagePreviewUrl ? (
              <div className="relative w-fit">
                <img
                  src={imagePreviewUrl}
                  alt="Start frame preview"
                  className="rounded-lg max-h-48 border border-zinc-800"
                />
                <button
                  type="button"
                  onClick={() => handleFileSelect(null)}
                  title="Remove image"
                  aria-label="Remove image"
                  className="absolute -top-2 -right-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-full p-1 cursor-pointer border border-zinc-700"
                >
                  <X size={13} />
                </button>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="flex flex-col items-center justify-center gap-2 border-2 border-dashed border-zinc-700 hover:border-zinc-600 rounded-lg py-8 text-zinc-500 hover:text-zinc-300 cursor-pointer transition-colors"
              >
                <Upload size={24} />
                <span className="text-sm">Click to upload an image</span>
              </button>
            )}
          </div>

          {supportsEndFrame && (
            <div className="flex flex-col gap-1.5">
              <label className="text-sm font-medium text-zinc-300">End image (optional)</label>
              <input
                ref={endFileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => handleEndFileSelect(e.target.files?.[0] ?? null)}
              />
              {endImagePreviewUrl ? (
                <div className="relative w-fit">
                  <img
                    src={endImagePreviewUrl}
                    alt="End frame preview"
                    className="rounded-lg max-h-48 border border-zinc-800"
                  />
                  <button
                    type="button"
                    onClick={() => handleEndFileSelect(null)}
                    title="Remove end image"
                    aria-label="Remove end image"
                    className="absolute -top-2 -right-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-full p-1 cursor-pointer border border-zinc-700"
                  >
                    <X size={13} />
                  </button>
                </div>
              ) : (
                <button
                  type="button"
                  onClick={() => endFileInputRef.current?.click()}
                  className="flex flex-col items-center justify-center gap-1.5 border-2 border-dashed border-zinc-700 hover:border-zinc-600 rounded-lg py-6 text-zinc-500 hover:text-zinc-300 cursor-pointer transition-colors"
                >
                  <Upload size={20} />
                  <span className="text-sm">Click to upload the end frame</span>
                </button>
              )}
              <p className="text-xs text-zinc-500">
                Only available for LTX-Video. Animates between the start and end frame.
              </p>
            </div>
          )}

          <div className="flex flex-col gap-1.5">
            <label className="text-sm font-medium text-zinc-300 flex items-center justify-between">
              <span>{promptRequired ? "Motion prompt" : "Prompt (optional)"}</span>
              {!promptRequired && <span className="text-xs text-zinc-500">Not used by this model</span>}
            </label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe the motion you want, e.g. gentle wind, slow zoom in"
              rows={3}
              className="bg-zinc-800 rounded-lg p-3 text-sm outline-none focus:ring-2 focus:ring-violet-600 placeholder:text-zinc-600"
            />
            <div className="flex flex-wrap gap-1.5">
              {MOTION_PROMPTS.map((p) => (
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
              groups={modelGroups}
              placeholder="Select a model"
            />
            {selectedModel && (
              <p className="text-xs text-zinc-500">
                {selectedModel.steps} step{selectedModel.steps === 1 ? "" : "s"} · {selectedModel.size}×
                {selectedModel.size}px
                {selectedModel.approx_size_mb > 0 && ` · ${formatModelSize(selectedModel.approx_size_mb)}`}
                {selectedModel.quantized && " · 4-bit quantized"}
                {selectedModel.remote && " · remote"}
              </p>
            )}
            {selectedModel && selectedModel.remote && (
              <p className="text-xs text-zinc-500">{selectedModel.repo}</p>
            )}
            <p className="text-xs text-zinc-500">
              {selectedModel?.remote
                ? selectedModel?.provider === "pollinations"
                  ? "Generates via Pollinations.ai — free, no API token required, no local download."
                  : "Generates via Hugging Face Inference API — requires HF token, free tier rate-limited."
                : "Animates your start image into a short video clip, guided by your motion prompt."}
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
            </div>
          </details>

          <div className="mt-1 flex items-stretch gap-2">
            <button
              onClick={handleGenerate}
              disabled={isRunning || !canGenerate}
              className="flex-1 flex items-center justify-center gap-2 bg-violet-600 hover:bg-violet-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg py-2.5 font-semibold text-sm cursor-pointer transition-colors"
            >
              {isRunning ? <Loader2 size={16} className="animate-spin" /> : <Film size={16} />}
              {isRunning ? "Generating…" : "Generate video"}
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
              <video
                src={job.result.file_url}
                autoPlay
                loop
                muted
                controls
                className="rounded-lg max-w-full max-h-[420px]"
              />
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
              <p className="text-sm">Your video will appear here</p>
            </div>
          )}
        </div>
      </div>

      <SettingsPanel kind="video" excludePipelines={I2V_PIPELINES} />

      <GalleryTab refreshKey={galleryRefreshKey} />
    </div>
  );
}
