# MiniMax-H3 React studio

The Space UI is a Vite + React 19 client: [Base UI](https://base-ui.com) (`@base-ui/react`) for the unstyled,
accessible behaviour, Tailwind 4 for every pixel of the styling, and Framer Motion for the movement. Gradio remains the
queued Python API and ZeroGPU boundary.

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
pill opens the same `Popover`: a Base UI Popover anchored under its trigger from `sm` up, a Base UI Drawer bottom sheet
below it. The stage under the composer reserves the aspect ratio of the clip that is coming, so nothing moves when the
video arrives.

Speed is a single *faster → smarter* axis rather than a list of presets, because the presets really are ordered:
`presetAxis` sorts them by scheduler steps and then by how much of the schedule the cache engine skips, both of which
come from `/studio-config`. A preset added on the server therefore lands in the right place with no table to update
here. Manual control is not a point on that axis, so it lives under Advanced and disables the slider while it is on.

## The motion rule

`src/lib/motion.ts` holds the only three transitions in the app — `POP` (presses, popovers, the segmented highlight),
`SETTLE` (layout and stage changes) and `FADE` (crossfades). Reaching for a fourth is usually a sign something wants a
different gesture rather than a different curve. `main.tsx` wraps the tree in `<MotionConfig reducedMotion="user">`, so
every JS animation honours the OS setting without a single check at the call site.

Which library animates a given thing is not a matter of taste:

- **Framer Motion** owns enter, exit and layout — anything that happens *because React re-rendered*. That means
  `AnimatePresence` around the popover, the stage's keyed states and the hero; `layoutId` for the segmented highlight;
  `layout="position"` on the composer so it slides up without springing the growing textarea's height.
- **CSS** owns anything Base UI writes on every frame. The slider thumb's `inset-inline-start`, the drawer's
  `--drawer-swipe-movement-y`, the collapsible's `--collapsible-panel-height`: Base UI is already the author of those
  values, and a JS animation laying a second transform on top fights it. The slider is the sharp case — Base UI
  re-measures the thumb with `getBoundingClientRect()` on every value change, so a Framer `whileTap={{ scale }}` (a
  `transform`) would inflate that measurement and drift the thumb mid-drag. The CSS `scale` property is separate from
  the `translate` Base UI writes, so it is safe; `group-data-dragging:` cancels the position transition so the thumb
  tracks the pointer exactly.

Base UI's `[data-starting-style]` / `[data-ending-style]` / `[data-open]` / `[data-disabled]` hooks do the rest, and
`keepMounted` on a portal is what hands the unmount to `AnimatePresence` so panels get an exit animation at all.

## Handoff map

- `src/App.tsx` — page composition, generation state, and `/status` polling
- `src/components/Composer.tsx` — prompt box, inline toolbar, control pills, examples
- `src/components/panels.tsx` — the body of each control pill's popover
- `src/components/ControlPill.tsx` — the shape every setting takes
- `src/components/Stage.tsx` — run phases, result, and error states
- `src/components/Header.tsx`, `UsageSheet.tsx`, `AboutSheet.tsx`, `KeyframeSlot.tsx`
- `src/ui/` — the primitives over Base UI: `Button`, `Popover`, `Sheet`, `Segmented`, `NotchSlider`, `Controls`
- `src/lib/motion.ts` — the three shared transitions
- `src/lib/studio.ts` — derived values: frame snapping, GPU-second estimates, canvas grouping, formatting
- `src/styles.css` — the `@theme` token block, the focus ring, and the indeterminate sweep
- `src/api.ts` — isolated Gradio client adapter and progress-event mapping

## Two things that must stay in sync with `app.py`

- `estimateGpuSeconds` in `src/lib/studio.ts` mirrors `get_duration`. It is what the composer's "books ≈…" readout and
  the per-preset costs in the Speed panel report, so a change to the Python timing constants has to be copied here.
- `snapFrames` encodes the video VAE's `17n + 5` rule at 24 fps. It is why a requested 5 s clip is announced as 5.04 s.

Run `npm run build` after visual changes. The checked-in `dist/` bundle is what the Gradio Space serves.
