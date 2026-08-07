import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchModelStatus, fetchStudioConfig, runGeneration } from "./api";
import { AboutSheet } from "./components/AboutSheet";
import { ComposeRail } from "./components/ComposeRail";
import { Header } from "./components/Header";
import { History } from "./components/History";
import { UsageSheet } from "./components/UsageSheet";
import { Viewer } from "./components/Viewer";
import { loadSpeed, saveSpeed } from "./lib/storage";
import { deleteHistoryItem, restoreHistory, saveHistoryItem } from "./lib/history";
import { estimateRuntime, loadRuntimeSamples, rememberRuntime, runtimeSampleFor } from "./lib/runtimeHistory";
import { findCanvas, snapFrames, FPS } from "./lib/studio";
import { draftValues, finalFrameFile, finalValues, recipeFrom, remixValues, valuesFromHistory } from "./lib/workflows";
import type { GenerationValues, HistoryItem, ModelStatus, RunProgress, StudioConfig } from "./types";
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
    // Whatever speed this browser last used wins over the defaults. `loadSpeed` has already validated the shape; the
    // effect below is what checks the remembered preset against the presets the server actually offers today.
    ...loadSpeed(),
  };
}

export default function App() {
  const [config, setConfig] = useState(FALLBACK_CONFIG);
  const [values, setValues] = useState(() => initialValues(FALLBACK_CONFIG));
  const [model, setModel] = useState<ModelStatus>({ ready: false, status: "Contacting the Space…", reachable: true });
  const [progress, setProgress] = useState<RunProgress>(IDLE);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [runtimeSamples, setRuntimeSamples] = useState(loadRuntimeSamples);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [usageOpen, setUsageOpen] = useState(false);
  const [aboutOpen, setAboutOpen] = useState(false);
  const viewerRef = useRef<HTMLDivElement>(null);

  const running =
    progress.stage === "connecting" || progress.stage === "queued" || progress.stage === "generating";
  const selected = history.find((item) => item.id === selectedId) ?? null;
  const runtimeEstimate = useMemo(
    () => estimateRuntime(runtimeSamples, config, values),
    [runtimeSamples, config, values],
  );

  useEffect(() => {
    let cancelled = false;
    restoreHistory().then((restored) => {
      if (cancelled) {
        for (const item of restored) URL.revokeObjectURL(item.url);
        return;
      }
      setHistory((current) => {
        const ids = new Set(current.map((item) => item.id));
        return [...current, ...restored.filter((item) => !ids.has(item.id))].sort((a, b) => b.createdAt - a.createdAt);
      });
      setSelectedId((current) => current ?? restored[0]?.id ?? null);
    });
    return () => {
      cancelled = true;
    };
  }, []);

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

  // Persisted on change rather than on generate, so a preset you picked and then walked away from is still there when
  // you come back — the point is that you never have to set it twice.
  useEffect(() => {
    saveSpeed(values);
  }, [
    values.preset,
    values.steps,
    values.acceleration,
    values.loraPreset,
    values.loraRepo,
    values.loraFilename,
    values.loraStrength,
  ]);

  const update = useCallback(
    <K extends keyof GenerationValues>(key: K, value: GenerationValues[K]) =>
      setValues((current) => ({ ...current, [key]: value })),
    [],
  );

  const applyExample = useCallback((prompt: string, canvas: string) => {
    setValues((current) => ({ ...current, prompt, canvas }));
  }, []);

  async function generate(requestedValues: GenerationValues = values) {
    if (!requestedValues.prompt.trim() || running) return;
    setError(null);
    // On a phone the viewer is below the rail; bring it up rather than leaving the run happening off-screen.
    viewerRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    try {
      // Canonicalize against the live config so a stale embedded bundle cannot submit a display-only resolution.
      const requestValues = { ...requestedValues, canvas: findCanvas(config, requestedValues.canvas).label };
      const requestEstimate = estimateRuntime(runtimeSamples, config, requestValues);
      const result = await runGeneration(requestValues, setProgress, requestEstimate);
      const item: HistoryItem = {
        ...result,
        id: globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`,
        createdAt: Date.now(),
        prompt: requestValues.prompt,
        canvas: findCanvas(config, requestValues.canvas),
        seconds: snapFrames(requestValues.duration) / FPS,
        seed: requestValues.seed,
        preset: requestValues.preset,
        recipe: recipeFrom(requestValues),
        sourceImage: requestValues.image,
        sourceLastImage: requestValues.lastImage,
      };
      setHistory((current) => [item, ...current]);
      // The UI can show the result immediately; durable browser storage continues without delaying completion.
      void saveHistoryItem(item).catch((reason) => console.warn("Could not persist generated clip.", reason));
      setRuntimeSamples((current) => rememberRuntime(current, runtimeSampleFor(config, requestValues, result.runtimeSeconds)));
      setSelectedId(item.id);
      setProgress({ stage: "complete", label: "Complete", progress: 1, exact: true });
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Generation failed. Please try again.";
      setError(message);
      setProgress({ stage: "error", label: message, progress: null });
    }
  }

  function remix(item: HistoryItem) {
    setValues(remixValues(item, config));
    setError(null);
  }

  async function continueFrom(item: HistoryItem) {
    if (running) return;
    setError(null);
    try {
      const frame = await finalFrameFile(item.url);
      setValues({
        ...valuesFromHistory(item, config),
        image: frame,
        lastImage: null,
        seed: Math.floor(Math.random() * 2 ** 31),
      });
      viewerRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not capture the final frame.");
    }
  }

  function removeFromHistory(item: HistoryItem) {
    if (item.url.startsWith("blob:")) URL.revokeObjectURL(item.url);
    void deleteHistoryItem(item.id);
    setHistory((current) => {
      const next = current.filter((candidate) => candidate.id !== item.id);
      // Deleting what you were watching should land on the neighbour, not on the empty state.
      if (item.id === selectedId) setSelectedId(next[0]?.id ?? null);
      return next;
    });
  }

  const blockedReason = model.reachable && !model.ready ? "Engine still loading" : null;

  return (
    // `lg:h-dvh lg:overflow-hidden` makes the two panes scroll independently on a desktop; below `lg` the whole thing
    // is one ordinary scrolling document, which is the only layout that behaves on a phone.
    <div className="flex min-h-dvh flex-col lg:h-dvh lg:overflow-hidden">
      <Header model={model} onOpenUsage={() => setUsageOpen(true)} onOpenAbout={() => setAboutOpen(true)} />

      <main className="grid min-h-0 flex-1 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)] xl:grid-cols-[minmax(0,24rem)_minmax(0,1fr)]">
        <div className="min-h-0 border-line lg:border-r">
          <ComposeRail
            config={config}
            values={values}
            update={update}
            onApplyExample={applyExample}
            onGenerate={() => void generate()}
            onGenerateDraft={() => void generate(draftValues(values, config))}
            running={running}
            blockedReason={blockedReason}
            runtimeEstimate={runtimeEstimate}
          />
        </div>

        {/* On a desktop this column is a fixed frame: the player takes what is left after the history strip, and
            nothing here scrolls. On a phone it is just the bottom half of the page. */}
        <div ref={viewerRef} className="flex min-h-0 flex-col p-4 lg:overflow-hidden lg:p-5">
          <Viewer
            className="lg:min-h-0 lg:flex-1"
            item={selected}
            progress={progress}
            running={running}
            error={error}
            onDismissError={() => {
              setError(null);
              setProgress(IDLE);
            }}
            onReusePrompt={(item) => setValues(valuesFromHistory(item, config))}
            onRenderFinal={(item) => void generate(finalValues(item, config))}
            onContinue={(item) => void continueFrom(item)}
            onRemix={remix}
          />
          <History
            items={history}
            selectedId={selectedId}
            onSelect={(item) => {
              setSelectedId(item.id);
              setError(null);
            }}
            onDelete={removeFromHistory}
          />
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
