# model-graphviz (vendored)

pto-design-system **model-graphviz** pattern — the hierarchical whole-network
(整网图) compute-graph renderer. Vendored verbatim (same convention as
`../memory-architecture`, `../aiv-core-object`) and registered globally as
`window.PtoModelGraphvizPattern`.

| file | what it is |
|---|---|
| `pattern.js` | SVG renderer. Public API: `render(container, graph, opts)` / `renderController(container, graph, opts)`. Verbatim. |
| `pattern.css` | node / edge / cluster / hover styles. Verbatim. |
| `openpangu-graph.json` | a **pre-positioned** graph (nodes/edges/clusters with `x/y`) for openPangu-2.0-Flash — 55 nodes · 61 edges · 5 clusters (Model → DecoderLayer → Attention / MoE → QKV / Expert → Operator). Captured from the reference deployment's own layout engine (`svg.ptoModelGraphviz.graph`), so it reproduces the reference 整网图 exactly without shipping the 87 KB layout engine. |
| `graph-meta.ts` | typed wrappers + `NODE_DIM` (operator → which parallel cut TP/PP/DP/EP partitions it) that bridges the graph to our 立方重组 / 整网切分 cube. |

`renderController` returns a handle with `selectNode(id)`, `clearSelection()`,
`setFocus`, `fit()`, `setTransform/getTransform`, `destroy()` and fires
`opts.onSelect({ nodeId, relatedNodeIds, relatedClusterIds, source })` on click.

Rendering is theme-agnostic in CSS (node fills come from JS via `opts.theme`),
so pass `theme: 'light' | 'dark'` explicitly — the app sets `data-theme` on React
roots, not on `documentElement`.

Provenance: openPangu-2.0-Flash model architecture, extracted from `config.json`
+ published source; layout is schematic. Consumed by `src/view/WholeNetGraph.tsx`.
