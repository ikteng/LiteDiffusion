import { Button } from "@heroui/react";
import { ImagePlus, X } from "lucide-react";
import { useEffect, useId, useState } from "react";

type Props = {
  label: string;
  hint: string;
  file: File | null;
  onChange: (file: File | null) => void;
};

export function MediaDropzone({ label, hint, file, onChange }: Props) {
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
    <div className="relative min-h-28 overflow-hidden rounded-xl border border-dashed border-divider bg-background">
      <input
        className="sr-only"
        id={inputId}
        type="file"
        accept="image/*"
        onChange={(event) => onChange(event.target.files?.[0] ?? null)}
      />
      {preview ? (
        <>
          <img className="h-32 w-full object-cover" src={preview} alt={`${label} preview`} />
          <Button className="absolute right-2 top-2" size="sm" variant="secondary" isIconOnly aria-label={`Remove ${label}`} onPress={() => onChange(null)}>
            <X className="size-3.5" />
          </Button>
          <label className="absolute bottom-2 left-2 cursor-pointer rounded-md bg-black/70 px-2 py-1 text-[10px] text-white" htmlFor={inputId}>{label}</label>
        </>
      ) : (
        <label className="flex min-h-28 cursor-pointer flex-col items-center justify-center gap-1.5 p-3 text-center" htmlFor={inputId}>
          <span className="grid size-8 place-items-center rounded-lg bg-default/8 text-muted"><ImagePlus className="size-4" /></span>
          <strong className="text-xs font-medium">{label}</strong>
          <small className="text-[11px] text-muted">{hint}</small>
        </label>
      )}
    </div>
  );
}
