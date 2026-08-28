import { useState } from "react";
import Header from "./components/Header";
import TabNav, { type TabKey } from "./components/TabNav";
import TextToImagePage from "./components/TextToImagePage";
import TextToVideoPage from "./components/TextToVideoPage";

export default function App() {
  const [tab, setTab] = useState<TabKey>("text-to-image");

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <Header />
      <TabNav active={tab} onChange={setTab} />
      <main className="max-w-5xl mx-auto">
        {tab === "text-to-image" && <TextToImagePage />}
        {tab === "text-to-video" && <TextToVideoPage />}
      </main>
    </div>
  );
}
