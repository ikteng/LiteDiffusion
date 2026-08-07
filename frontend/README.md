# MiniMax-H3 React studio

The Space UI is a Vite + React 19 client styled with Tailwind 4 and a small set of local primitives — there is no
component library. Gradio remains the queued Python API and ZeroGPU boundary.

```bash
npm install
npm run dev
npm run build
```

The local Vite dev server is served at `http://127.0.0.1:5173/studio-assets/` and proxies `/status`, `/studio-config`
and `/gradio_api` to `127.0.0.1:7860`. With no Python server running it falls back to `FALLBACK_CONFIG` in
`src/types.ts`, so the interface can be worked on without loading the model.

## Shape of the UI

One column, prompt first. The composer holds the prompt, the per-shot inputs (keyframes, prompt enhancement) and the
Generate button; the settings that persist between shots — speed, format, length, seed — sit beneath it as pills. Every
pill opens the same `Popover`: an anchored panel from `sm` up, a bottom sheet below it. The stage under the composer
reserves the aspect ratio of the clip that is coming, so nothing moves when the video arrives.

Speed is a single *faster → smarter* axis rather than a list of presets, because the presets really are ordered:
`presetAxis` sorts them by scheduler steps and then by how much of the schedule the cache engine skips, both of which
come from `/studio-config`. A preset added on the server therefore lands in the right place with no table to update
here. Manual control is not a point on that axis, so it lives under Advanced and disables the slider while it is on.

## Handoff map

- `src/App.tsx` — page composition, generation state, and `/status` polling
- `src/components/Composer.tsx` — prompt box, inline toolbar, control pills, examples
- `src/components/panels.tsx` — the body of each control pill's popover
- `src/components/ControlPill.tsx` — the shape every setting takes
- `src/components/Stage.tsx` — run phases, result, and error states
- `src/components/Header.tsx`, `UsageSheet.tsx`, `AboutSheet.tsx`, `KeyframeSlot.tsx`
- `src/ui/` — the primitives: `Button`, `Popover`, `Sheet`, `Segmented`, `NotchSlider`, `Controls`
- `src/lib/studio.ts` — derived values: frame snapping, GPU-second estimates, canvas grouping, formatting
- `src/styles.css` — the `@theme` token block, the slider, and two keyframes
- `src/api.ts` — isolated Gradio client adapter and progress-event mapping

## Two things that must stay in sync with `app.py`

- `estimateGpuSeconds` in `src/lib/studio.ts` mirrors `get_duration`. It is what the composer's "books ≈…" readout and
  the per-preset costs in the Speed panel report, so a change to the Python timing constants has to be copied here.
- `snapFrames` encodes the video VAE's `17n + 5` rule at 24 fps. It is why a requested 5 s clip is announced as 5.04 s.

Run `npm run build` after visual changes. The checked-in `dist/` bundle is what the Gradio Space serves.
