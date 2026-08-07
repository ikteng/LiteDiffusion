import { ImagePlus, X } from "lucide-react";
import { useEffect, useId, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { cx } from "../lib/cx";
import { FADE, POP } from "../lib/motion";

type Props = {
  label: string;
  hint: string;
  file: File | null;
  onChange: (file: File | null) => void;
};

export function KeyframeSlot({ label, hint, file, onChange }: Props) {
  const inputId = useId();
  const [preview, setPreview] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

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
    <div
      onDragEnter={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        const dropped = [...event.dataTransfer.files].find((candidate) => candidate.type.startsWith("image/"));
        if (dropped) onChange(dropped);
      }}
      className={cx(
        "relative h-36 overflow-hidden rounded-xl bg-sunken transition-colors duration-150",
        // Solid once it holds an image, dashed while it is still an invitation to drop one in.
        preview ? "border border-line" : "border border-dashed border-line",
        dragging && "border-accent bg-accent/8",
      )}
    >
      <input
        id={inputId}
        type="file"
        accept="image/*"
        className="sr-only"
        onChange={(event) => onChange(event.target.files?.[0] ?? null)}
      />
      <AnimatePresence initial={false} mode="popLayout">
        {preview ? (
          <motion.div
            key="preview"
            initial={{ opacity: 0, scale: 1.04 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            transition={FADE}
            className="absolute inset-0"
          >
            <img src={preview} alt={`${label} preview`} className="size-full object-cover" />
            <motion.button
              type="button"
              onClick={() => onChange(null)}
              aria-label={`Remove ${label}`}
              whileTap={{ scale: 0.92 }}
              transition={POP}
              className="absolute right-1.5 top-1.5 grid size-7 place-items-center rounded-lg bg-black/70 text-ink backdrop-blur hover:bg-black/85"
            >
              <X className="size-3.5" />
            </motion.button>
            <span className="absolute bottom-1.5 left-1.5 rounded bg-black/70 px-1.5 py-0.5 text-[10px] font-medium backdrop-blur">
              {label}
            </span>
          </motion.div>
        ) : (
          <motion.label
            key="empty"
            htmlFor={inputId}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={FADE}
            className="absolute inset-0 flex cursor-pointer flex-col items-center justify-center gap-1 px-2 text-center transition-colors duration-100 hover:bg-raised/50"
          >
            <ImagePlus className="size-4 text-faint" />
            <span className="text-[12px] font-medium text-ink">{label}</span>
            <span className="text-[10.5px] text-faint">{hint}</span>
          </motion.label>
        )}
      </AnimatePresence>
    </div>
  );
}
