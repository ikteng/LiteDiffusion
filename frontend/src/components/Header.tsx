import { Sparkles } from "lucide-react";

export default function Header() {
  return (
    <header className="border-b border-zinc-800 bg-zinc-950/80 backdrop-blur sticky top-0 z-10">
      <div className="max-w-5xl mx-auto flex items-center gap-3 px-6 py-4">
        <div className="flex items-center justify-center w-9 h-9 rounded-xl bg-violet-600/15 text-violet-400">
          <Sparkles size={18} />
        </div>
        <div>
          <h1 className="text-base font-semibold tracking-tight leading-tight">LiteDiffusion</h1>
          <p className="text-xs text-zinc-500 leading-tight">Local image &amp; video generation on your hardware</p>
        </div>
      </div>
    </header>
  );
}
