import { useEffect, useState } from "react";
import { Trash2, Images, Loader2, ChevronDown, Video, Download } from "lucide-react";
import { api } from "../api";
import type { HistoryItem } from "../types";

export default function GalleryTab({ refreshKey }: { refreshKey?: number } = {}) {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [open, setOpen] = useState(true);

  async function refresh() {
    setLoading(true);
    const res = await api.getHistory();
    setItems(res.items);
    setLoading(false);
  }

  useEffect(() => {
    refresh();
  }, [refreshKey]);

  async function handleDelete(id: string) {
    await api.deleteHistory(id);
    setItems((prev) => prev.filter((item) => item.id !== id));
  }

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-2xl overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between p-4 cursor-pointer"
      >
        <span className="flex items-center gap-2 text-sm font-medium text-zinc-300">
          <Images size={15} />
          Gallery
        </span>
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-400">
            {loading ? "…" : items.length}
          </span>
          <ChevronDown size={16} className={`text-zinc-500 transition-transform ${open ? "rotate-180" : ""}`} />
        </div>
      </button>

      {open && (
        <div className="border-t border-zinc-800">
          {loading && (
            <div className="flex flex-col items-center gap-2 text-zinc-500 p-10">
              <Loader2 size={24} className="animate-spin" />
              <p className="text-sm">Loading gallery…</p>
            </div>
          )}

          {!loading && items.length === 0 && (
            <div className="flex flex-col items-center gap-2 text-zinc-600 p-10">
              <Images size={28} />
              <p className="text-sm">No generations yet — generate an image or video above to get started.</p>
            </div>
          )}

          {!loading && items.length > 0 && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4">
              {items.map((item) => (
                <div
                  key={item.id}
                  className="group relative bg-zinc-800/60 border border-zinc-800 rounded-xl overflow-hidden hover:border-zinc-700 transition-colors"
                >
                  {item.media_type === "video" ? (
                    <video
                      src={item.file_url}
                      muted
                      loop
                      playsInline
                      className="w-full aspect-square object-cover bg-black"
                    />
                  ) : (
                    <img src={item.file_url} alt={item.prompt} className="w-full aspect-square object-cover" />
                  )}
                  <span className="absolute top-2 left-2 bg-black/60 rounded-full p-1.5 text-violet-300">
                    {item.media_type === "video" ? <Video size={14} /> : <Images size={14} />}
                  </span>
                  <a
                    href={item.file_url}
                    download
                    title="Download"
                    className="absolute top-2 right-2 bg-black/60 hover:bg-violet-600 rounded-full p-1.5 opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer no-underline"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <Download size={14} />
                  </a>
                  <button
                    onClick={() => handleDelete(item.id)}
                    title="Delete"
                    className="absolute top-9 right-2 bg-black/60 hover:bg-red-600 rounded-full p-1.5 opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
                  >
                    <Trash2 size={14} />
                  </button>
                  <div className="p-2.5">
                    <p className="text-xs text-zinc-300 line-clamp-2">{item.prompt}</p>
                    <p className="text-[11px] text-zinc-500 mt-1">
                      {item.model} · seed {item.seed}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
