import { ImageIcon, Video, Film, ChevronDown, ChevronLeft, ChevronRight, Images, GitCommitHorizontal, PanelsTopLeft } from "lucide-react";
import type { ComponentType } from "react";

export type TabKey = "text-to-image" | "text-to-video" | "image-to-video";
export type ImageToVideoMode = "single" | "first-last";

export interface NavState {
  tab: TabKey;
  i2vMode: ImageToVideoMode;
}

const TOP_LEVEL: { key: TabKey; label: string; icon: ComponentType<{ size?: number }> }[] = [
  { key: "text-to-image", label: "Text-to-Image", icon: ImageIcon },
  { key: "text-to-video", label: "Text-to-Video", icon: Video },
  { key: "image-to-video", label: "Image-to-Video", icon: Film },
];

const I2V_MODES: { key: ImageToVideoMode; label: string; icon: ComponentType<{ size?: number }> }[] = [
  { key: "single", label: "Single reference", icon: Images },
  { key: "first-last", label: "First + last frame", icon: GitCommitHorizontal },
];

export default function Sidebar({
  nav,
  onChange,
  collapsed,
  onToggleCollapsed,
}: {
  nav: NavState;
  onChange: (nav: NavState) => void;
  collapsed: boolean;
  onToggleCollapsed: () => void;
}) {
  return (
    <aside
      className={`self-stretch shrink-0 border-r border-zinc-800 bg-zinc-950 flex flex-col min-h-0 transition-all ${
        collapsed ? "w-14" : "w-56"
      }`}
    >
      <div className="flex items-center justify-between h-12 px-2 shrink-0">
        {!collapsed && (
          <span className="flex items-center gap-2 pl-1.5 text-[11px] font-semibold uppercase tracking-wider text-zinc-500 truncate select-none">
            <PanelsTopLeft size={14} className="text-zinc-600 shrink-0" />
            Menu
          </span>
        )}
        <button
          onClick={onToggleCollapsed}
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          className="flex items-center justify-center w-8 h-8 rounded-lg text-zinc-500 hover:text-zinc-200 hover:bg-zinc-900 cursor-pointer transition-colors shrink-0"
        >
          {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </div>
      {!collapsed && <div className="mx-3 border-b border-zinc-800" />}

      <nav className="flex-1 min-h-0 overflow-y-auto flex flex-col gap-1 p-2">
        {TOP_LEVEL.map(({ key, label, icon: Icon }) => {
          const active = nav.tab === key;
          const isI2V = key === "image-to-video";
          return (
            <div key={key} className="flex flex-col">
              <button
                onClick={() => onChange({ tab: key, i2vMode: nav.i2vMode })}
                title={collapsed ? label : undefined}
                className={`flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-sm font-medium transition-colors cursor-pointer ${
                  active
                    ? "bg-violet-600/15 text-violet-300"
                    : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900"
                }`}
              >
                <Icon size={16} />
                {!collapsed && <span className="flex-1 text-left truncate">{label}</span>}
                {!collapsed && isI2V && (
                  <ChevronDown size={14} className={`text-zinc-500 transition-transform ${active ? "rotate-180" : ""}`} />
                )}
              </button>

              {!collapsed && isI2V && active && (
                <div className="ml-4 mt-1 flex flex-col gap-0.5 border-l border-zinc-800 pl-2.5">
                  {I2V_MODES.map(({ key: modeKey, label: modeLabel, icon: ModeIcon }) => {
                    const modeActive = nav.i2vMode === modeKey;
                    return (
                      <button
                        key={modeKey}
                        onClick={() => onChange({ tab: "image-to-video", i2vMode: modeKey })}
                        className={`flex items-center gap-2 px-2 py-1.5 rounded-md text-xs font-medium transition-colors cursor-pointer ${
                          modeActive
                            ? "bg-violet-600/10 text-violet-300"
                            : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900"
                        }`}
                      >
                        <ModeIcon size={13} />
                        <span className="truncate">{modeLabel}</span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </nav>
    </aside>
  );
}
