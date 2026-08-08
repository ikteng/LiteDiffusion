import { useEffect, useMemo, useRef } from "react";
import { ArrowDown, ArrowUp, FileAudio, Film, ImageIcon, Plus, Trash2 } from "lucide-react";
import type { GenerationValues, ReferenceAsset, ReferenceKind, StudioConfig } from "../types";
import { Button } from "../ui/Button";
import { Segmented } from "../ui/Segmented";
import { KeyframeSettings } from "./settings";

type Props = {
  config: StudioConfig;
  values: GenerationValues;
  update: <K extends keyof GenerationValues>(key: K, value: GenerationValues[K]) => void;
};

function kindOf(file: File): ReferenceKind | null {
  if (file.type.startsWith("image/")) return "image";
  if (file.type.startsWith("video/")) return "video";
  if (file.type.startsWith("audio/")) return "audio";
  return null;
}

function icon(kind: ReferenceKind) {
  return kind === "image" ? <ImageIcon /> : kind === "video" ? <Film /> : <FileAudio />;
}

function ReferenceThumb({ asset }: { asset: ReferenceAsset }) {
  const url = useMemo(() => URL.createObjectURL(asset.file), [asset.file]);
  useEffect(() => () => URL.revokeObjectURL(url), [url]);
  if (asset.kind === "image") return <img src={url} alt="" className="size-full object-cover" />;
  if (asset.kind === "video") return <video src={`${url}#t=.1`} muted preload="metadata" className="size-full object-cover" />;
  return icon(asset.kind);
}

export function ReferenceLibrary({ config, values, update }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const counts = useMemo(
    () => Object.fromEntries(["image", "video", "audio"].map((kind) => [kind, values.references.filter((r) => r.kind === kind).length])),
    [values.references],
  );
  const limits = config.ref2va ?? { enabled: true, max_total: 12, max_images: 9, max_videos: 3, max_audio: 3, minimum_duration: 5 };

  function add(files: FileList | null) {
    const next = [...values.references];
    for (const file of Array.from(files ?? [])) {
      const kind = kindOf(file);
      if (!kind || next.length >= limits.max_total) continue;
      const limit = kind === "image" ? limits.max_images : kind === "video" ? limits.max_videos : limits.max_audio;
      if (next.filter((asset) => asset.kind === kind).length >= limit) continue;
      next.push({ id: crypto.randomUUID(), file, kind });
    }
    update("references", next);
    if (values.duration < limits.minimum_duration) update("duration", limits.minimum_duration);
  }

  function move(index: number, delta: number) {
    const target = index + delta;
    if (target < 0 || target >= values.references.length) return;
    const next = [...values.references];
    [next[index], next[target]] = [next[target], next[index]];
    update("references", next);
  }

  return (
    <div>
      <Segmented
        ariaLabel="Reference workflow"
        value={values.referenceMode}
        onChange={(value) => update("referenceMode", value as GenerationValues["referenceMode"])}
        options={[
          { value: "keyframes", label: "Keyframes" },
          { value: "omni", label: "Reference studio" },
        ]}
      />
      {values.referenceMode === "keyframes" ? (
        <div className="mt-3"><KeyframeSettings values={values} update={update} /></div>
      ) : (
        <div className="mt-3 space-y-3">
          <p className="text-[11.5px] leading-relaxed text-muted">
            Ordered image, video and audio references guide identity, motion, look and voice. Ref2VA uses the full 28-step quality path.
          </p>
          <div className="flex flex-wrap gap-1.5 text-[10.5px] text-faint">
            <span>{counts.image}/{limits.max_images} images</span><span>·</span>
            <span>{counts.video}/{limits.max_videos} videos</span><span>·</span>
            <span>{counts.audio}/{limits.max_audio} audio</span>
          </div>
          <div className="space-y-1.5">
            {values.references.map((asset, index) => (
              <div key={asset.id} className="flex items-center gap-2 rounded-xl bg-sunken p-2 ring-1 ring-inset ring-line">
                <span className="grid size-9 shrink-0 place-items-center overflow-hidden rounded-lg bg-raised text-accent [&>svg]:size-4"><ReferenceThumb asset={asset} /></span>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-[12px] font-medium text-ink">{index + 1}. {asset.file.name}</p>
                  <p className="text-[10.5px] capitalize text-faint">{asset.kind} reference</p>
                </div>
                <button aria-label="Move up" disabled={!index} onClick={() => move(index, -1)} className="text-muted disabled:opacity-25"><ArrowUp className="size-3.5" /></button>
                <button aria-label="Move down" disabled={index === values.references.length - 1} onClick={() => move(index, 1)} className="text-muted disabled:opacity-25"><ArrowDown className="size-3.5" /></button>
                <button aria-label="Remove" onClick={() => update("references", values.references.filter((item) => item.id !== asset.id))} className="text-muted hover:text-bad"><Trash2 className="size-3.5" /></button>
              </div>
            ))}
          </div>
          <input ref={input} hidden multiple type="file" accept="image/*,video/*,audio/*" onChange={(event) => add(event.target.files)} />
          <Button variant="outline" size="sm" onClick={() => input.current?.click()} disabled={values.references.length >= limits.max_total} className="w-full">
            <Plus /> Add references
          </Button>
          {counts.audio > 0 && counts.image + counts.video === 0 && <p className="text-[11px] text-warn">Add an image or video alongside audio.</p>}
        </div>
      )}
    </div>
  );
}
