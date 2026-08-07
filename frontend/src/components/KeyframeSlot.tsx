import { ImagePlus, X } from "lucide-react";
import { useEffect, useId, useState } from "react";

type Props = {
  label: string;
  hint: string;
  file: File | null;
  onChange: (file: File | null) => void;
};

export function KeyframeSlot({ label, hint, file, onChange }: Props) {
  const inputId = useId();
  const [preview, setPreview] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  return (
    <div className="relative overflow-hidden rounded-xl border border-dashed border-line bg-sunken">
      <input
        id={inputId}
        type="file"
        accept="image/*"
        className="sr-only"
        onChange={(event) => onChange(event.target.files?.[0] ?? null)}
      />
      {preview ? (
        <>
          <img src={preview} alt={`${label} preview`} className="h-28 w-full object-cover" />
          <button
            type="button"
            onClick={() => onChange(null)}
            aria-label={`Remove ${label}`}
            className="absolute right-1.5 top-1.5 grid size-7 place-items-center rounded-lg bg-black/70 text-ink backdrop-blur hover:bg-black/85"
          >
            <X className="size-3.5" />
          </button>
          <span className="absolute bottom-1.5 left-1.5 rounded bg-black/70 px-1.5 py-0.5 text-[10px] font-medium backdrop-blur">
            {label}
          </span>
        </>
      ) : (
        <label
          htmlFor={inputId}
          className="flex h-28 cursor-pointer flex-col items-center justify-center gap-1 px-2 text-center hover:bg-raised/50"
        >
          <ImagePlus className="size-4 text-faint" />
          <span className="text-[12px] font-medium text-ink">{label}</span>
          <span className="text-[10.5px] text-faint">{hint}</span>
        </label>
      )}
    </div>
  );
}
