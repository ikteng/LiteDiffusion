import { Video } from "lucide-react";

export default function TextToVideoPage() {
  return (
    <div className="flex flex-col items-center justify-center gap-3 text-zinc-600 p-24">
      <Video size={32} />
      <p className="text-sm text-zinc-400 font-medium">Text-to-video is coming soon</p>
      <p className="text-xs text-zinc-600 max-w-sm text-center">
        The backend already reserves a spot for it in the job model — this tab will light up once a local
        text-to-video pipeline lands.
      </p>
    </div>
  );
}
