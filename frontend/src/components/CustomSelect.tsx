import { useState, useRef, useEffect } from "react";
import { ChevronDown, Check, Download, Loader2 } from "lucide-react";

interface Option {
  value: string;
  label: string;
  sublabel?: string;
  downloadStatus?: "idle" | "downloading" | "ready";
  onDownload?: () => void;
  disabled?: boolean;
}

interface OptionGroup {
  label: string;
  options: Option[];
}

interface CustomSelectProps {
  value: string;
  onChange: (value: string) => void;
  groups: OptionGroup[];
  placeholder?: string;
}

export default function CustomSelect({ value, onChange, groups, placeholder }: CustomSelectProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const selectedLabel =
    groups.find((g) => g.options.some((o) => o.value === value))?.options.find((o) => o.value === value)?.label ||
    placeholder || "";

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between gap-2 bg-zinc-800 rounded-lg p-2.5 text-sm text-left outline-none focus:ring-2 focus:ring-violet-600 cursor-pointer"
      >
        <span className={`truncate min-w-0 ${value ? "text-zinc-200" : "text-zinc-500"}`}>{selectedLabel}</span>
        <ChevronDown size={14} className="text-zinc-400 transition-transform shrink-0" />
      </button>

      {open && (
        <div className="absolute z-20 mt-1 w-full bg-zinc-800 border border-zinc-700 rounded-lg shadow-lg max-h-72 overflow-y-auto">
          {groups.filter((g) => g.options.length > 0).map((group, idx) => (
            <div key={group.label || idx} className={idx > 0 ? "border-t border-zinc-700" : ""}>
              {group.label && (
                <div className="px-2.5 py-1.5 text-xs font-medium text-zinc-500 uppercase tracking-wider">
                  {group.label}
                </div>
              )}
               {group.options.map((opt) => {
                const notDownloaded = opt.downloadStatus && opt.downloadStatus !== "ready";
                const tooltip = notDownloaded
                  ? opt.downloadStatus === "downloading"
                    ? "Downloading…"
                    : "Not downloaded. Click the download button to use this model."
                  : undefined;
                return (
                  <div
                    key={opt.value}
                    title={tooltip}
                    className={`flex items-center justify-between px-2.5 py-1.5 text-sm ${
                      opt.disabled
                        ? "opacity-40 cursor-not-allowed"
                        : "hover:bg-zinc-700/50 cursor-pointer"
                    }`}
                  >
                    <button
                      onClick={() => {
                        if (opt.disabled || notDownloaded) return;
                        onChange(opt.value);
                        setOpen(false);
                      }}
                      disabled={!!opt.disabled}
                      className={`flex items-center gap-2 text-left flex-1 min-w-0 ${notDownloaded ? "text-zinc-500" : "text-zinc-200"}`}
                    >
                      <span className="truncate min-w-0">{opt.label}</span>
                      {opt.sublabel && <span className="text-xs text-zinc-500 shrink-0">· {opt.sublabel}</span>}
                      {opt.value === value && <Check size={12} className="text-violet-400 ml-auto shrink-0" />}
                    </button>
                    {opt.downloadStatus && opt.downloadStatus !== "ready" && opt.onDownload && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          opt.onDownload!();
                        }}
                        className={`ml-2 p-1 rounded hover:bg-zinc-700 ${
                          opt.downloadStatus === "downloading" ? "text-zinc-300" : "text-violet-400"
                        }`}
                        title="Download model"
                      >
                        {opt.downloadStatus === "downloading" ? (
                          <Loader2 size={12} className="animate-spin" />
                        ) : (
                          <Download size={12} />
                        )}
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
