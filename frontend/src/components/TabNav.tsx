import { ImageIcon, Video } from "lucide-react";
import type { ComponentType } from "react";

export type TabKey = "text-to-image" | "text-to-video";

const TABS: { key: TabKey; label: string; icon: ComponentType<{ size?: number }> }[] = [
  { key: "text-to-image", label: "Text-to-Image", icon: ImageIcon },
  { key: "text-to-video", label: "Text-to-Video", icon: Video },
];

export default function TabNav({
  active,
  onChange,
}: {
  active: TabKey;
  onChange: (tab: TabKey) => void;
}) {
  return (
    <nav className="border-b border-zinc-800">
      <div className="max-w-5xl mx-auto flex gap-1 px-6">
        {TABS.map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => onChange(key)}
            className={`flex items-center gap-1.5 px-3.5 py-2.5 text-sm font-medium rounded-t-lg border-b-2 transition-colors ${
              active === key
                ? "border-violet-500 text-white cursor-pointer"
                : "border-transparent text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900/50 cursor-pointer"
            }`}
          >
            <Icon size={15} />
            {label}
          </button>
        ))}
      </div>
    </nav>
  );
}
