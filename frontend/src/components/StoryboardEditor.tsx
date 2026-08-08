import { ArrowDown, ArrowUp, Copy, Plus, Trash2 } from "lucide-react";
import type { StoryboardShot } from "../types";
import { Button } from "../ui/Button";

export function StoryboardEditor({ shots, onChange }: { shots: StoryboardShot[]; onChange: (shots: StoryboardShot[]) => void }) {
  const patch = (id: string, prompt: string) => onChange(shots.map((shot) => shot.id === id ? { ...shot, prompt } : shot));
  const move = (index: number, delta: number) => {
    const target = index + delta;
    if (target < 0 || target >= shots.length) return;
    const next = [...shots]; [next[index], next[target]] = [next[target], next[index]]; onChange(next);
  };
  return (
    <div className="space-y-2.5">
      {shots.map((shot, index) => (
        <article key={shot.id} className="rounded-xl bg-sunken p-2.5 ring-1 ring-inset ring-line">
          <div className="mb-2 flex items-center gap-2">
            <span className="grid size-5 place-items-center rounded-md bg-accent/15 text-[10px] font-semibold text-accent">{index + 1}</span>
            <span className="text-[11px] font-medium text-muted">Shot {index + 1}</span><span className="flex-1" />
            <button disabled={!index} onClick={() => move(index, -1)} className="text-muted disabled:opacity-20"><ArrowUp className="size-3.5" /></button>
            <button disabled={index === shots.length - 1} onClick={() => move(index, 1)} className="text-muted disabled:opacity-20"><ArrowDown className="size-3.5" /></button>
            <button onClick={() => onChange([...shots.slice(0, index + 1), { ...shot, id: crypto.randomUUID() }, ...shots.slice(index + 1)])} className="text-muted hover:text-ink"><Copy className="size-3.5" /></button>
            <button disabled={shots.length <= 2} onClick={() => onChange(shots.filter((item) => item.id !== shot.id))} className="text-muted hover:text-bad disabled:opacity-20"><Trash2 className="size-3.5" /></button>
          </div>
          <textarea value={shot.prompt} onChange={(event) => patch(shot.id, event.target.value)} rows={3} placeholder="Action, camera, dialogue and sound for this shot…" className="w-full resize-none rounded-lg bg-canvas p-2.5 text-[12.5px] leading-relaxed text-ink ring-1 ring-inset ring-line placeholder:text-faint focus:ring-accent focus:outline-none" />
        </article>
      ))}
      <Button variant="outline" size="sm" onClick={() => onChange([...shots, { id: crypto.randomUUID(), prompt: "" }])} disabled={shots.length >= 8} className="w-full"><Plus /> Add shot</Button>
      <p className="text-[10.5px] leading-relaxed text-faint">Each shot continues from the previous final frame, then the editor joins them into one film.</p>
    </div>
  );
}
