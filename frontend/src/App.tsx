import {
  Button,
  Card,
  Chip,
  Disclosure,
  Input,
  ListBox,
  Modal,
  Select,
  Slider,
  Switch,
  Tabs,
  TextArea,
} from "@heroui/react";
import {
  ChevronDown,
  Clock3,
  Code2,
  Dices,
  ExternalLink,
  ImagePlus,
  Info,
  Play,
  Settings2,
  Sparkles,
  WandSparkles,
  Zap,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { fetchModelStatus, fetchStudioConfig, runGeneration } from "./api";
import { MediaDropzone } from "./components/MediaDropzone";
import { OutputStage } from "./components/OutputStage";
import { UsageDrawer } from "./components/UsageDrawer";
import type { GeneratedVideo, GenerationValues, RunProgress, StudioConfig } from "./types";
import { FALLBACK_CONFIG } from "./types";

const INITIAL_PROMPT = "A red fox trotting through a snowy pine forest at dawn, snow crunching underfoot";
const IDLE: RunProgress = { stage: "idle", label: "Ready", progress: null };

function initialValues(config: StudioConfig): GenerationValues {
  return {
    prompt: INITIAL_PROMPT,
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

type StudioSelectProps = {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
};

function StudioSelect({ label, value, options, onChange }: StudioSelectProps) {
  return (
    <Select
      aria-label={label}
      selectedKey={value}
      onSelectionChange={(key) => key != null && onChange(String(key))}
      fullWidth
      variant="secondary"
    >
      <Select.Trigger>
        <Select.Value />
        <Select.Indicator><ChevronDown className="size-4" /></Select.Indicator>
      </Select.Trigger>
      <Select.Popover>
        <ListBox>
          {options.map((option) => <ListBox.Item id={option} key={option}>{option}</ListBox.Item>)}
        </ListBox>
      </Select.Popover>
    </Select>
  );
}

function SectionTitle({ icon, title, description }: { icon: React.ReactNode; title: string; description?: string }) {
  return (
    <div className="flex items-start gap-3">
      <span className="mt-0.5 grid size-8 shrink-0 place-items-center rounded-lg bg-default/8 text-muted">{icon}</span>
      <div>
        <h2 className="text-sm font-semibold tracking-tight text-foreground">{title}</h2>
        {description && <p className="mt-0.5 text-xs leading-5 text-muted">{description}</p>}
      </div>
    </div>
  );
}

export default function App() {
  const [config, setConfig] = useState(FALLBACK_CONFIG);
  const [values, setValues] = useState(() => initialValues(FALLBACK_CONFIG));
  const [model, setModel] = useState({ ready: false, status: "Starting MiniMax-H3…", loading: true });
  const [progress, setProgress] = useState<RunProgress>(IDLE);
  const [video, setVideo] = useState<GeneratedVideo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [aboutOpen, setAboutOpen] = useState(false);
  const [usageOpen, setUsageOpen] = useState(false);
  const isRunning = ["connecting", "queued", "generating"].includes(progress.stage);
  const isCustom = values.preset === config.custom_preset;
  const quickPresets = config.presets.filter(
    (preset) => !preset.custom && !preset.value.startsWith("Exact") && !preset.value.startsWith("Ultra"),
  );
  const selectedCanvas = useMemo(
    () => config.canvases.find((item) => item.label === values.canvas) ?? config.canvases[0],
    [config.canvases, values.canvas],
  );

  useEffect(() => {
    fetchStudioConfig()
      .then((next) => {
        setConfig(next);
        setValues((current) => ({
          ...current,
          canvas: next.canvases.some((item) => item.label === current.canvas) ? current.canvas : next.default_canvas,
          preset: next.presets.some((item) => item.value === current.preset) ? current.preset : next.default_preset,
        }));
      })
      .catch(() => undefined);

    const refresh = () => {
      fetchModelStatus()
        .then((next) => setModel({ ...next, loading: !next.ready }))
        .catch(() => setModel({ ready: false, status: "Generator is reconnecting…", loading: false }));
    };
    refresh();
    const timer = window.setInterval(refresh, 15_000);
    return () => window.clearInterval(timer);
  }, []);

  function update<K extends keyof GenerationValues>(key: K, value: GenerationValues[K]) {
    setValues((current) => ({ ...current, [key]: value }));
  }

  async function generate() {
    if (!values.prompt.trim() || isRunning) return;
    setError(null);
    setVideo(null);
    try {
      const result = await runGeneration(values, setProgress);
      setVideo(result);
      setProgress({ stage: "complete", label: "Generation complete", progress: 1 });
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Generation failed. Please try again.";
      setError(message);
      setProgress({ stage: "error", label: message, progress: null });
    }
  }

  function resetOutput() {
    setVideo(null);
    setError(null);
    setProgress(IDLE);
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <main className="mx-auto w-full max-w-[1600px] px-4 py-6 sm:px-6 lg:px-8">
        <div className="mb-5 flex flex-col justify-between gap-4 md:flex-row md:items-end">
          <div>
            <div className="mb-2 flex items-center gap-2 text-xs font-medium text-accent">
              <Zap className="size-3.5" fill="currentColor" /> NVFP4 video + native audio
            </div>
            <h1 className="text-3xl font-semibold tracking-[-0.04em] sm:text-4xl">Make a scene from a sentence.</h1>
            <p className="mt-2 text-sm text-muted">Video and synchronized sound, generated together.</p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Chip variant="soft" color={model.ready ? "success" : "warning"}>
              <span className={`size-1.5 rounded-full ${model.ready ? "bg-success" : "bg-warning"}`} />
              {model.ready ? "Model ready" : "First run may take longer"}
            </Chip>
            <Button variant="secondary" size="sm" onPress={() => setUsageOpen(true)}>
              <Code2 className="size-4" /> API & MCP
            </Button>
            <Button variant="ghost" size="sm" isIconOnly aria-label="About this Space" onPress={() => setAboutOpen(true)}>
              <Info className="size-4" />
            </Button>
          </div>
        </div>

        <div className="grid items-start gap-4 xl:grid-cols-[430px_minmax(0,1fr)]">
          <Card className="overflow-hidden" variant="secondary">
            <Card.Content className="divide-y divide-divider p-0">
              <section className="space-y-4 p-5">
                <SectionTitle icon={<WandSparkles className="size-4" />} title="What should happen?" description="Mention the action, camera, mood, and sound." />
                <TextArea
                  aria-label="Video prompt"
                  value={values.prompt}
                  onChange={(event) => update("prompt", event.target.value)}
                  rows={6}
                  fullWidth
                  variant="secondary"
                  placeholder="A cinematic close-up of…"
                />
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="mr-1 text-[11px] text-muted">Try</span>
                  {config.examples.map((example) => (
                    <Button
                      key={example.title}
                      size="sm"
                      variant="tertiary"
                      onPress={() => setValues((current) => ({ ...current, prompt: example.prompt, canvas: example.canvas }))}
                    >
                      {example.title}
                    </Button>
                  ))}
                </div>
                <Switch isSelected={values.upsample} onChange={(selected) => update("upsample", selected)}>
                  <Switch.Control><Switch.Thumb /></Switch.Control>
                  <Switch.Content>
                    <span className="text-sm">Enhance prompt</span>
                    <span className="text-xs text-muted">Adds creative detail; slightly slower.</span>
                  </Switch.Content>
                </Switch>
              </section>

              <section className="space-y-4 p-5">
                <SectionTitle icon={<Sparkles className="size-4" />} title="Choose a mode" description="Balanced keeps the full 28-step quality schedule." />
                <Tabs
                  aria-label="Generation mode"
                  selectedKey={values.preset}
                  onSelectionChange={(key) => update("preset", String(key))}
                  variant="secondary"
                >
                  <Tabs.ListContainer>
                    <Tabs.List>
                      {quickPresets.map((preset) => (
                        <Tabs.Tab id={preset.value} key={preset.value}>
                          {preset.value.split(" — ")[0].replace("Turbo ", "")}
                        </Tabs.Tab>
                      ))}
                    </Tabs.List>
                  </Tabs.ListContainer>
                </Tabs>
                <p className="text-xs leading-5 text-muted">
                  {config.presets.find((preset) => preset.value === values.preset)?.description}
                </p>
              </section>

              <section className="space-y-4 p-5">
                <div className="flex items-center justify-between">
                  <SectionTitle icon={<Clock3 className="size-4" />} title="Format" />
                  <Chip size="sm" variant="soft">{selectedCanvas.width} × {selectedCanvas.height}</Chip>
                </div>
                <StudioSelect
                  label="Canvas"
                  value={values.canvas}
                  options={config.canvases.map((canvas) => canvas.label)}
                  onChange={(value) => update("canvas", value)}
                />
                <div>
                  <div className="mb-2 flex items-center justify-between text-xs">
                    <span className="text-muted">Duration</span><strong>{values.duration} seconds</strong>
                  </div>
                  <Slider
                    aria-label="Duration in seconds"
                    minValue={config.duration.min}
                    maxValue={config.duration.max}
                    step={1}
                    value={values.duration}
                    onChange={(next) => update("duration", Number(next))}
                  >
                    <Slider.Track><Slider.Fill /><Slider.Thumb /></Slider.Track>
                  </Slider>
                </div>
              </section>

              <Disclosure>
                <Disclosure.Heading>
                  <Disclosure.Trigger className="flex w-full items-center justify-between p-5 text-sm">
                    <span className="flex items-center gap-2"><ImagePlus className="size-4" /> Guide with images <span className="text-xs text-muted">Optional</span></span>
                    <Disclosure.Indicator><ChevronDown className="size-4" /></Disclosure.Indicator>
                  </Disclosure.Trigger>
                </Disclosure.Heading>
                <Disclosure.Content>
                  <Disclosure.Body className="px-5 pb-5">
                    <div className="grid grid-cols-2 gap-2">
                      <MediaDropzone label="Start frame" hint="Upload image" file={values.image} onChange={(file) => update("image", file)} />
                      <MediaDropzone label="End frame" hint="Upload image" file={values.lastImage} onChange={(file) => update("lastImage", file)} />
                    </div>
                  </Disclosure.Body>
                </Disclosure.Content>
              </Disclosure>

              <Disclosure>
                <Disclosure.Heading>
                  <Disclosure.Trigger className="flex w-full items-center justify-between p-5 text-sm">
                    <span className="flex items-center gap-2"><Settings2 className="size-4" /> Advanced</span>
                    <span className="flex items-center gap-2 text-xs text-muted">Seed {values.seed}<Disclosure.Indicator><ChevronDown className="size-4" /></Disclosure.Indicator></span>
                  </Disclosure.Trigger>
                </Disclosure.Heading>
                <Disclosure.Content>
                  <Disclosure.Body className="space-y-4 px-5 pb-5">
                    <StudioSelect
                      label="Generation preset"
                      value={values.preset}
                      options={config.presets.map((preset) => preset.value)}
                      onChange={(value) => update("preset", value)}
                    />
                    <label className="block space-y-1.5 text-xs text-muted">
                      <span className="flex items-center gap-1.5"><Dices className="size-3.5" /> Seed</span>
                      <Input type="number" value={String(values.seed)} onChange={(event) => update("seed", Number(event.target.value))} fullWidth />
                    </label>
                    {isCustom && (
                      <div className="grid gap-3 sm:grid-cols-2">
                        <StudioSelect label="Cache mode" value={values.acceleration} options={["Balanced", "Ultra Fast", "Exact"]} onChange={(value) => update("acceleration", value as GenerationValues["acceleration"])} />
                        <Input aria-label="Schedule points" type="number" min={4} max={40} value={String(values.steps)} onChange={(event) => update("steps", Number(event.target.value))} />
                        <StudioSelect label="LoRA mode" value={values.loraPreset} options={["None", "Turbo · 4 steps", "Turbo · 8 steps", "Custom"]} onChange={(value) => update("loraPreset", value as GenerationValues["loraPreset"])} />
                        {values.loraPreset === "Custom" && (
                          <>
                            <Input aria-label="Public LoRA repository" placeholder="owner/repository" value={values.loraRepo} onChange={(event) => update("loraRepo", event.target.value)} />
                            <Input aria-label="LoRA filename" placeholder="adapter.safetensors" value={values.loraFilename} onChange={(event) => update("loraFilename", event.target.value)} />
                          </>
                        )}
                      </div>
                    )}
                  </Disclosure.Body>
                </Disclosure.Content>
              </Disclosure>
            </Card.Content>

            <Card.Footer className="flex-col gap-3 p-5">
              <Button
                variant="primary"
                size="lg"
                fullWidth
                isDisabled={!values.prompt.trim() || isRunning}
                onPress={generate}
              >
                {isRunning ? <>{progress.stage === "queued" ? "Waiting for GPU" : "Generating…"}</> : <><Play className="size-4" fill="currentColor" /> Generate video</>}
              </Button>
              <p className="text-center text-[11px] text-muted">Every click creates a fresh generation.</p>
            </Card.Footer>
          </Card>

          <OutputStage video={video} progress={progress} error={error} onReset={resetOutput} />
        </div>
      </main>

      <footer className="mx-auto flex max-w-[1600px] justify-between px-4 py-6 text-xs text-muted sm:px-6 lg:px-8">
        <span>MiniMax-H3 Ultra Fast</span>
        <a href="https://x.com/realmrfakename" target="_blank" rel="noreferrer">@realmrfakename</a>
      </footer>

      <UsageDrawer isOpen={usageOpen} onOpenChange={setUsageOpen} />

      <Modal isOpen={aboutOpen} onOpenChange={setAboutOpen}>
        <Modal.Backdrop>
          <Modal.Container size="lg">
            <Modal.Dialog>
              <Modal.CloseTrigger />
              <Modal.Header><Modal.Icon><Zap /></Modal.Icon><Modal.Heading>MiniMax-H3 Ultra Fast</Modal.Heading></Modal.Header>
              <Modal.Body className="space-y-4">
                <p className="text-sm leading-6 text-muted">Video and synchronized sound on one Blackwell ZeroGPU worker, with a pruned NVFP4 transformer, local conditioner, Sol-Attn, fused kernels, and Cache-DiT.</p>
                <div className="grid grid-cols-3 gap-2">
                  {[['NVFP4', 'Native weights'], ['28 steps', 'Balanced mode'], ['Local', 'Conditioner + VAEs']].map(([title, copy]) => (
                    <Card variant="secondary" key={title}><Card.Content className="p-3"><strong className="text-sm text-accent">{title}</strong><p className="mt-1 text-xs text-muted">{copy}</p></Card.Content></Card>
                  ))}
                </div>
                <div className="flex flex-wrap gap-2">
                  <Button variant="tertiary" size="sm" onPress={() => window.open("https://huggingface.co/spaces/multimodalart/minimax-h3", "_blank")}>Original Space <ExternalLink className="size-3" /></Button>
                  <Button variant="tertiary" size="sm" onPress={() => window.open("https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/optimized", "_blank")}>Sana / Sol-Engine <ExternalLink className="size-3" /></Button>
                </div>
                <p className="text-xs text-muted">H/t to Blanchon for pointing to Sana/Sol-Engine and Cache-DiT. If this helps, please like the Space &lt;3</p>
              </Modal.Body>
              <Modal.Footer>
                <Button variant="ghost" onPress={() => setAboutOpen(false)}>Close</Button>
                <Button variant="primary" onPress={() => window.open("https://huggingface.co/spaces/mrfakename/minimax-h3-faster", "_blank")}>Like the Space</Button>
              </Modal.Footer>
            </Modal.Dialog>
          </Modal.Container>
        </Modal.Backdrop>
      </Modal>
    </div>
  );
}
