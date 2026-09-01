import { useState } from "react";
import Header from "./components/Header";
import Sidebar, { type NavState } from "./components/Sidebar";
import TextToImagePage from "./components/TextToImagePage";
import TextToVideoPage from "./components/TextToVideoPage";
import ImageToVideoPage from "./components/ImageToVideoPage";

export default function App() {
  const [nav, setNav] = useState<NavState>({ tab: "text-to-image", i2vMode: "single" });
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col">
      <Header />
      <div className="flex flex-1 min-h-0">
        <Sidebar nav={nav} onChange={setNav} collapsed={collapsed} onToggleCollapsed={() => setCollapsed((v) => !v)} />
        <main className="flex-1 max-w-5xl mx-auto w-full">
          {nav.tab === "text-to-image" && <TextToImagePage />}
          {nav.tab === "text-to-video" && <TextToVideoPage />}
          {nav.tab === "image-to-video" && <ImageToVideoPage mode={nav.i2vMode} />}
        </main>
      </div>
    </div>
  );
}
