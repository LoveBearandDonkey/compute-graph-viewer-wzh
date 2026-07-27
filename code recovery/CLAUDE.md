# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A static, single-page **Ascend Tiling Visualization Workbench** — a teaching/debug tool that replays Ascend C operator execution traces and highlights, step by step, which slice of a logical tensor each block/tile/loop touches, how data moves through the memory hierarchy, and the execution timeline. Conceptually it is a Triton-Viz-style trace replayer adapted to Huawei Ascend NPU (`dav-2201` / Ascend 950) semantics. See [docs/ascend-tiling-visualization-knowledge.md](docs/ascend-tiling-visualization-knowledge.md) for the conceptual model behind the three panes.

## Running

There is **no build system** — pure vanilla JS/CSS/HTML, no `package.json`, no bundler, no tests.

The app loads fixtures via `fetch()` and depends on `../vendor/pto-design-system/`, so it must be served over HTTP from the **parent directory** (`/Users/yin/pto/`), not opened as `file://` and not served from `tiling/` itself:

```sh
cd /Users/yin/pto && python3 -m http.server 8000
# then open http://localhost:8000/tiling/index.html
```

Always verify visual changes in the browser before considering a task complete.

### Cache busting

`index.html` references `src/app.js` and `src/styles.css` with a `?v=YYYYMMDD-...` query string. When you change either file, bump that version token in [index.html](index.html) or the browser may serve a stale cached copy.

## Architecture

Three layers: **trace fixtures (data)** → **app.js (single IIFE driver)** → **PTO design-system patterns (vendored UI)**.

### Trace fixtures are the source of truth

Everything the UI shows is data-driven from a trace JSON in `data/fixtures/`, validated against [data/schemas/trace.schema.json](data/schemas/trace.schema.json). A trace bundles: `operator`/`arch`/`launch`/`tiling` metadata, the `source` listing, the `memory` tier model, `stages` (hardware pipeline phases), `steps` (the replayable timeline; each step links to `sourceLines`, `memoryRegions`, `metrics`, and a `visualState`), and `puzzles`. The `.asc` files in `data/sources/` are the original Ascend C kernels the fixtures were derived from.

The three fixtures correspond to the three operator `kind`s the app special-cases:
- `vector` — `add_tpipe_tque.trace.json` (block partitioning, TQue/UB buffers, DataCopy/Add/CopyOut)
- `cube` — `matmul.trace.json` (Cube matmul, L1/L0A/L0B/CO1 staging, Mmad accumulation, Fixpipe)
- `fusion` — `matmul_leakyrelu_fusion.trace.json` (AIC→AIV producer/consumer with cross-core sync)

To add an operator, add a fixture (register it in the `FIXTURES` array at the top of [src/app.js](src/app.js)) plus the kind-specific derivation branches below.

### app.js (single ~1460-line IIFE)

One file, no modules. Key structure:
- `state` — the single mutable store (current sample, `stepIndex`, viewport pan/zoom, architecture overlay, playback). `render()` re-derives the DOM from `state` + the current trace.
- **Kind dispatch**: the visual highlighting is computed per step by `deriveVisualState` → `deriveVectorVisualState` / `deriveCubeVisualState` / `deriveFusionVisualState`, with matching `*BufferBlocks` / `*Selectors` / `*Routes` helpers. When changing how a kind is visualized, edit its branch in all three families.
- **Three panes** rendered together: Source listing (with `highlightAscendC` tokenizer), the Trace Visual pane (a `<canvas>` 3D logical-tensor viewport + an on-chip tile lens + the execution timeline canvas), and the Memory Architecture pane (driven by the vendored `memory-architecture` / `hardware-architecture-viewport` patterns).
- Playback (auto-advance through steps) is mounted via the `floating-playback-control` pattern; element IDs it owns are listed in `state.playbackIds`.
- UI strings are bilingual: English lives in the fixtures, Chinese overrides come from the `TEXT_ZH` map. Add a `TEXT_ZH` entry when introducing new English copy.

### Vendored PTO design system (`../vendor/pto-design-system/`)

The shell, IDE frame, panes, timeline, playback, and memory-architecture visuals all come from vendored CSS/JS patterns loaded in [index.html](index.html). **Design-system rule (enforced):** any UI here must consume the shared PTO tokens, components, and patterns first. Do not introduce a private visual style system or new visual tokens without a preview and explicit approval — prefer extending the vendored patterns. The `pto-design-system` skill encodes this workflow.

## Conventions

- Dark UI: use neutral grays (e.g. `#292929`), never blue-tinted darks.
- Do not remove existing behavior (e.g. auto-loading the sample fixtures) unless explicitly asked.
- Get plan approval before starting visual/UI changes.
