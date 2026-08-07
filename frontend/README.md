# MiniMax-H3 React studio

The Space UI is a Vite + React 19 client built from HeroUI v3 compound components. Gradio remains the queued Python API and ZeroGPU boundary.

```bash
npm install
npm run dev
npm run build
```

The local Vite preview is served at `http://127.0.0.1:5173/studio-assets/`. It falls back to local configuration when the Python server is unavailable, so designers can work on the interface without loading the model.

## Handoff map

- `src/App.tsx` — studio composition, form state, and HeroUI controls
- `src/components/OutputStage.tsx` — empty, queued, generating, error, and result states
- `src/components/UsageDrawer.tsx` — API and MCP integration instructions
- `src/components/MediaDropzone.tsx` — optional keyframe inputs
- `src/styles.css` — intentionally minimal page canvas and product accent
- `src/api.ts` — isolated Gradio client adapter

Run `npm run build` after visual changes. The checked-in `dist/` bundle is what the Gradio Space serves.
