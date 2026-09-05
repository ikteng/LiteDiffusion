import { useEffect, useState } from "react";
import { ChevronDown, Cpu, Zap, ExternalLink } from "lucide-react";
import { api } from "../api";
import type { ModelInfo, SettingsResponse } from "../types";
import { formatModelSize, sortModelsBySize } from "../utils";

export default function SettingsPanel({
  kind = "image",
  excludePipelines,
}: { kind?: "image" | "video"; excludePipelines?: string[] }) {
  const [open, setOpen] = useState(false);
  const [settings, setSettings] = useState<SettingsResponse | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);

  useEffect(() => {
    api.getSettings().then(setSettings);
    api.getModels().then((res) => {
      const filtered = res.models.filter(
        (m) => m.kind === kind && (!excludePipelines || !excludePipelines.includes(m.pipeline))
      );
      const local = sortModelsBySize(filtered.filter((m) => !m.remote));
      const remote = filtered.filter((m) => m.remote);
      setModels([...local, ...remote]);
    });
  }, [kind]);

  const isGpu = settings?.device === "cuda";

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-2xl overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between p-4 cursor-pointer"
      >
        <span className="flex items-center gap-2 text-sm font-medium text-zinc-300">
          <Cpu size={15} />
          Settings & models
        </span>
        <div className="flex items-center gap-2">
          <span
            className={`text-xs font-medium px-2 py-0.5 rounded-full ${
              isGpu ? "bg-emerald-500/15 text-emerald-400" : "bg-zinc-800 text-zinc-400"
            }`}
          >
            {settings?.device.toUpperCase() ?? "…"}
          </span>
          <ChevronDown size={16} className={`text-zinc-500 transition-transform ${open ? "rotate-180" : ""}`} />
        </div>
      </button>

      {open && (
        <div className="px-4 pb-4 flex flex-col gap-4 border-t border-zinc-800 pt-4">
          <div className="flex flex-col gap-2 text-sm">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 text-zinc-400">
                <Cpu size={14} />
                Device
              </span>
              <span className="text-zinc-200">{settings?.device.toUpperCase() ?? "…"}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 text-zinc-400">
                <Zap size={14} />
                Precision
              </span>
              <span className="text-zinc-200">{settings?.dtype ?? "…"}</span>
            </div>
            <p className="text-xs text-zinc-600">
              Detected automatically at startup — CUDA + FP16 if an NVIDIA GPU is available, otherwise CPU + FP32.
            </p>
          </div>

          <div className="flex flex-col gap-2">
            <p className="text-xs font-medium text-zinc-400 uppercase tracking-wide">Available models</p>
            {models.map((m) => (
              <a
                key={m.key}
                href={`https://huggingface.co/${m.repo}`}
                target="_blank"
                rel="noreferrer"
                className="group flex items-center justify-between bg-zinc-800/50 hover:bg-zinc-800 rounded-lg p-2.5 transition-colors"
              >
                <div>
                  <p className="text-sm text-zinc-200 flex items-center gap-1.5">
                    {m.label}
                    <ExternalLink size={11} className="text-zinc-600 group-hover:text-zinc-400" />
                  </p>
                  <p className="text-xs text-zinc-500">{m.repo}</p>
                </div>
                <div className="text-right text-xs text-zinc-500 shrink-0 pl-3">
                  <p>
                    {m.steps} step{m.steps === 1 ? "" : "s"}
                  </p>
                  <p>
                    {m.size}×{m.size}px
                  </p>
                  {m.approx_size_mb > 0 && <p>{formatModelSize(m.approx_size_mb)}</p>}
                </div>
              </a>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
