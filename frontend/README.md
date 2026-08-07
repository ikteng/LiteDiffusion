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

Two panes: **compose on the left, watch on the right.** The left rail is the whole request — speed, prompt, keyframes,
format, length, seed — as one scrolling stack of `Section`s with a sticky action bar pinned to its foot showing what the
run will cost and the Generate button. The right pane is the player, its result toolbar, and the browser's saved history
filmstrip. Finished MP4s and their metadata are restored from bounded IndexedDB storage, and playback uses Vidstack's
React player. Rail sections and the history strip are collapsible; manual scheduler and LoRA control remain nested under
Advanced because they are an escape from the presets rather than another ordinary setting.

Below `lg` the grid collapses to one column and the two panes stack: rail on top, viewer beneath. That is the only
structural change — no separate mobile components, no drawer. `App.tsx` is `lg:h-dvh lg:overflow-hidden` so on a desktop
each pane scrolls independently inside a fixed shell, and below `lg` the whole thing is one ordinary scrolling document,
which is the only layout that behaves on a phone. Pressing Generate scrolls the viewer into view, since on a phone it
is off-screen below the rail.

The player is a fixed frame that the video letterboxes into with `object-contain`, not a box sized to each clip's
aspect ratio. That is what stops the pane from resizing every time you click a 9:16 clip in a history of 16:9 ones.

Speed is a single *faster → smarter* axis rather than a list of presets, because the presets really are ordered:
`presetAxis` sorts them by scheduler steps and then by how much of the schedule the cache engine skips, both of which
come from `/studio-config`. A preset added on the server therefore lands in the right place with no table to update
here. Manual control is not a point on that axis, so it lives under Advanced and disables the slider while it is on.

Speed sits at the top of the rail and is the one section with `tone="accent"`, because it is the only setting that
decides how long you wait — and the only one someone who never scrolls past the prompt would otherwise never find. The
treatment is deliberately scarce: a second accented section would make both of them ordinary. Its title row carries a
one-press shortcut to the fastest preset, computed by `fasterOption` in `settings.tsx`, which returns `null` when the
user is already on the fastest preset, when manual controls are in effect, or when the saving is under 30 s — so the
button can never decay into one people learn to ignore.

`src/lib/storage.ts` persists the speed group (`preset`, `steps`, `acceleration` and the four LoRA fields) to
`localStorage` under `minimax-h3.speed.v1`, written on change rather than on generate. Speed is the only thing
persisted, because it describes the person rather than the shot; prompt, seed, keyframes and canvas are properties of
*this* clip. Two hazards it handles: a Space renders in a cross-origin iframe, so `window.localStorage` can throw a
`SecurityError` on the *property access itself* when third-party storage is blocked — hence the accessor is wrapped, not
just the read — and every field is shape-checked on load, so a corrupt or hand-edited blob degrades to the defaults
rather than taking the app down. A remembered preset that the server no longer offers is dropped by an effect in
`App.tsx`, which is the only place that knows today's preset list.

`src/lib/runtimeHistory.ts` retains up to 60 completed post-queue runtimes in the same browser. ETA uses only matching
presets, weights nearby workloads more heavily, and scales observations for canvas, frame count and keyframes. It never
uses the ZeroGPU reservation as elapsed time; the quota booking remains a separate, explicitly labelled figure.

There is exactly one slider component, `ui/Slider.tsx`. Duration, preset, scheduler steps and LoRA strength are all the
same instrument; passing `stops` is the only difference, and it adds dots on the track and screen-reader names for each
position. Anything that makes one slider look unlike another belongs in that file or nowhere.

## The motion rule

`src/lib/motion.ts` holds the only three transitions in the app — `POP` (presses, popovers, the segmented highlight),
`SETTLE` (layout and stage changes) and `FADE` (crossfades). Reaching for a fourth is usually a sign something wants a
different gesture rather than a different curve. `main.tsx` wraps the tree in `<MotionConfig reducedMotion="user">`, so
every JS animation honours the OS setting without a single check at the call site.

Which library animates a given thing is not a matter of taste:

- **Framer Motion** owns enter, exit and layout — anything that happens *because React re-rendered*. That means
  `AnimatePresence` around the viewer's keyed states, the result toolbar and the history tiles; `layoutId` for the
  segmented highlight; `layout` on the preset card so the rail does not jolt when a longer description swaps in.
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

- `src/App.tsx` — the two-pane shell, generation state, restored browser history, and `/status` polling
- `src/components/ComposeRail.tsx` — the left pane: prompt, examples, every settings section, the sticky action bar
- `src/components/settings.tsx` — the body of each section, plus `budgetFor` and `fasterOption`
- `src/components/Viewer.tsx` — the player and its empty, running, error and result states
- `src/components/History.tsx` — the persistent browser filmstrip
- `src/components/Header.tsx`, `UsageSheet.tsx`, `KeyframeSlot.tsx`
- `src/components/AboutSheet.tsx` — the technical note behind the ⓘ button: the pipeline, the quantization figures, what
  each preset actually changes, the four behaviours that surprise people. Every number in it is read off `app.py`,
  `h3_nvfp4.py` and `h3_local_conditioner.py`; if those change, this changes with them
- `src/ui/` — the primitives over Base UI: `Button`, `Slider`, `Segmented`, `Controls`, `Section`, `Popover`, `Sheet`,
  `Tip`
- `src/lib/storage.ts` — the remembered speed group and its validated `localStorage` access
- `src/lib/runtimeHistory.ts` — validated wall-clock samples and the per-preset ETA estimator
- `src/lib/history.ts` — bounded IndexedDB persistence for generated MP4s and their metadata
- `src/lib/motion.ts` — the three shared transitions
- `src/lib/studio.ts` — derived values: frame snapping, GPU-second estimates, canvas grouping, formatting
- `src/styles.css` — the `@theme` token block, the focus ring, and the indeterminate sweep
- `src/api.ts` — isolated Gradio client adapter and progress-event mapping

## Two things that must stay in sync with `app.py`

- `estimateGpuSeconds` in `src/lib/studio.ts` mirrors `get_duration`. It feeds the separately labelled GPU-booking
  readouts and the saving named on the faster-preset shortcut, so a change to the Python timing constants has to be
  copied here or those quota figures will not match what the server honours.
- `snapFrames` encodes the video VAE's `17n + 5` rule at 24 fps. It is why a requested 5 s clip is announced as 5.17 s.

Run `npm run build` after visual changes. The checked-in `dist/` bundle is what the Gradio Space serves.
