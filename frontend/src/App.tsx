import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { fetchModelStatus, fetchStudioConfig, runGeneration } from "./api";
import { AboutSheet } from "./components/AboutSheet";
import { Composer } from "./components/Composer";
import { Header } from "./components/Header";
import { Stage } from "./components/Stage";
import { UsageSheet } from "./components/UsageSheet";
import { cx } from "./lib/cx";
import { SETTLE } from "./lib/motion";
import { findCanvas } from "./lib/studio";
import type { GeneratedVideo, GenerationValues, ModelStatus, RunProgress, StudioConfig } from "./types";
import { FALLBACK_CONFIG } from "./types";

const IDLE: RunProgress = { stage: "idle", label: "Ready", progress: null };
const STATUS_POLL_MS = 15_000;

function initialValues(config: StudioConfig): GenerationValues {
  return {
    prompt: "",
    image: null,
    lastImage: null,
    canvas: config.default_canvas,
    duration: config.duration.default,
    seed: 42,
    upsample: false,
    preset: config.default_preset,
    steps: 28,
    acceleration: "Balanced",
    loraPreset: "None",
    loraRepo: "",
    loraFilename: "",
    loraStrength: 1,
  };
}

export default function App() {
  const [config, setConfig] = useState(FALLBACK_CONFIG);
  const [values, setValues] = useState(() => initialValues(FALLBACK_CONFIG));
  const [model, setModel] = useState<ModelStatus>({ ready: false, status: "Contacting the Space…", reachable: true });
  const [progress, setProgress] = useState<RunProgress>(IDLE);
  const [video, setVideo] = useState<GeneratedVideo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [usageOpen, setUsageOpen] = useState(false);
  const [aboutOpen, setAboutOpen] = useState(false);
  const stageRef = useRef<HTMLDivElement>(null);

  const running =
    progress.stage === "connecting" || progress.stage === "queued" || progress.stage === "generating";
  const hasStage = running || video != null || error != null;

  useEffect(() => {
    fetchStudioConfig()
      .then((next) => {
        setConfig(next);
        // Keep whatever the user already selected if the server still offers it; otherwise adopt the server default.
        setValues((current) => ({
          ...current,
          canvas: next.canvases.some((item) => item.label === current.canvas) ? current.canvas : next.default_canvas,
          preset: next.presets.some((item) => item.value === current.preset) ? current.preset : next.default_preset,
          duration: current.duration === FALLBACK_CONFIG.duration.default ? next.duration.default : current.duration,
        }));
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    let cancelled = false;
    function refresh() {
      fetchModelStatus()
        .then((next) => !cancelled && setModel({ ...next, reachable: true }))
        .catch(() => !cancelled && setModel({ ready: false, status: "The Space is not responding.", reachable: false }));
    }
    refresh();
    const timer = window.setInterval(refresh, STATUS_POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  const update = useCallback(
    <K extends keyof GenerationValues>(key: K, value: GenerationValues[K]) =>
      setValues((current) => ({ ...current, [key]: value })),
    [],
  );

  const applyExample = useCallback((prompt: string, canvas: string) => {
    setValues((current) => ({ ...current, prompt, canvas }));
  }, []);

  async function generate() {
    if (!values.prompt.trim() || running) return;
    setError(null);
    setVideo(null);
    try {
      const result = await runGeneration(values, setProgress);
      setVideo(result);
      setProgress({ stage: "complete", label: "Complete", progress: 1, exact: true });
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Generation failed. Please try again.";
      setError(message);
      setProgress({ stage: "error", label: message, progress: null });
    }
  }

  // On a short window the result lands below the fold; bring it into view rather than leaving the user to find it.
  useEffect(() => {
    if (video) stageRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [video]);

  const blockedReason = model.reachable && !model.ready ? "Engine still loading" : null;

  return (
    <div className="flex min-h-screen flex-col">
      <Header model={model} onOpenUsage={() => setUsageOpen(true)} onOpenAbout={() => setAboutOpen(true)} />

      <main className="flex-1">
        <div
          className={cx(
            "mx-auto flex min-h-[calc(100vh-3.5rem)] w-full max-w-3xl flex-col gap-5 px-4 pb-10",
            hasStage ? "justify-start pt-6" : "justify-center pt-0",
          )}
        >
          {/* The hero is the page until there is something better to look at, then it gets out of the way and the
              composer takes its place. `layout` on the composer is what makes that a move rather than a jump. */}
          <AnimatePresence initial={false}>
            {!hasStage && (
              <motion.div
                key="hero"
                layout
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -12, height: 0, marginBottom: -20 }}
                transition={SETTLE}
                className="overflow-hidden text-center"
              >
                <h1 className="text-balance text-[26px] font-semibold leading-tight tracking-[-0.03em] sm:text-[32px]">
                  Make a scene from a sentence.
                </h1>
                <p className="mt-2 text-[13.5px] text-muted">
                  Video and its soundtrack, generated together in one pass.
                </p>
              </motion.div>
            )}
          </AnimatePresence>

          {/* `layout="position"` and not plain `layout`: the composer must slide up when the hero leaves, but the
              textarea grows as you type, and springing its *height* would leave the caret trailing the cursor. */}
          <motion.div layout="position" transition={SETTLE}>
            <Composer
              config={config}
              values={values}
              update={update}
              onApplyExample={applyExample}
              onGenerate={generate}
              running={running}
              blockedReason={blockedReason}
            />
          </motion.div>

          <div ref={stageRef}>
            <Stage
              video={video}
              progress={progress}
              error={error}
              canvas={findCanvas(config, values.canvas)}
              onDismissError={() => {
                setError(null);
                setProgress(IDLE);
              }}
            />
          </div>
        </div>
      </main>

      <UsageSheet
        open={usageOpen}
        onClose={() => setUsageOpen(false)}
        defaultCanvas={config.default_canvas}
        defaultPreset={config.default_preset}
      />
      <AboutSheet open={aboutOpen} onClose={() => setAboutOpen(false)} />
    </div>
  );
}
