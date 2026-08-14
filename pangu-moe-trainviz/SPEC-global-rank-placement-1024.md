# Pangu MoE TrainViz · 1024 Rank Global Placement Spec

> Status: draft
> Target page: `pangu-moe-trainviz/op-rank-time.html`
> Scope: add a global 32/1024 placement switch and reconstruct the 1024-rank parallel allocation view from the training parallel configuration.

## 1. Goal

Current main view renders a 32-rank diagnostic slice. The target is to add a top-level global switch:

- `32 slice`: keep the existing demo/runtime trace behavior.
- `1024 global`: project `DP32 x PP8 x TP4 = 1024 ranks` onto a physical Huawei-style supernode/rack placement model.

The 1024 mode should help users answer:

- Where is a global rank physically located?
- Which DP/PP/TP group does it belong to?
- Which PP stage owns which model layers?
- Which physical racks/supernodes are involved when an object says `tp`, `pp`, `dp`, `ep`, `cp`, or `sp`?
- How does the 32-rank diagnostic slice relate to the full 1024-rank allocation?

## 2. Non-Goals

- Do not fabricate a real 1024-rank profiler trace.
- Do not replace the existing 32-rank 1F1B simulation.
- Do not make the 1024 view a flat grid of 1024 text labels.
- Do not introduce a new component visual language outside PTO design-system controls.

The 1024 view is a placement projection, not measured runtime evidence.

## 3. Parallel Presets

Introduce a placement preset layer:

```js
const PLACEMENT_PRESETS = {
  slice32: {
    id: 'slice32',
    label: '32 slice',
    mode: 'trace-slice',
    parallel: { dp: 2, pp: 4, tp: 2, ep: 2, cp: 1, sp: 1 },
    rankFormula: 'rank = (((dp * PP + pp) * TP + tp) * EP + ep)',
  },
  global1024: {
    id: 'global1024',
    label: '1024 global',
    mode: 'placement-projection',
    parallel: { dp: 32, pp: 8, tp: 4, ep: 1, cp: 1, sp: 1 },
    rankFormula: 'rank = (dp * PP + pp) * TP + tp',
  },
};
```

Notes:

- `slice32` keeps the existing EP dimension because the current demo is `DP2 x PP4 x TP2 x EP2`.
- `global1024` follows the archived product target `DP32 x PP8 x TP4 = 1024`.
- If a future training config provides EP/CP/SP in the global world, the preset should accept those fields without forcing the UI rewrite.

## 4. Physical Topology Assumption

Use a replaceable physical topology preset, initially modeled after Huawei CloudMatrix-style supernode/rack organization.

Draft preset:

```js
const PHYSICAL_TOPOLOGY_PRESETS = {
  cloudmatrix384Projection: {
    id: 'cloudmatrix384Projection',
    label: 'CloudMatrix384 projection',
    npuPerNode: 8,
    npuPerComputeRack: 32,
    nodesPerComputeRack: 4,
    npuPerSuperNode: 384,
    computeRacksPerFullSuperNode: 12,
    communicationRacksPerFullSuperNode: 4,
  },
};
```

For 1024 ranks:

- SuperNode 0: ranks `0-383`
- SuperNode 1: ranks `384-767`
- SuperNode 2: ranks `768-1023` partial supernode
- Total compute racks: `1024 / 32 = 32`
- Total 8-NPU nodes: `1024 / 8 = 128`

Important: this is a physical placement projection. If an internal rack map is later available, replace this preset data and keep the rendering API.

## 5. Rank Mapping

Logical rank coordinates:

```js
function globalRankOf({ dp, pp, tp }) {
  return (dp * GLOBAL.pp + pp) * GLOBAL.tp + tp;
}

function globalCoordsOf(rank) {
  const tp = rank % GLOBAL.tp;
  const pp = Math.floor(rank / GLOBAL.tp) % GLOBAL.pp;
  const dp = Math.floor(rank / (GLOBAL.tp * GLOBAL.pp));
  return { dp, pp, tp };
}
```

Physical coordinates:

```js
function physicalCoordsOf(rank) {
  const superNode = Math.floor(rank / 384);
  const rankInSuperNode = rank % 384;
  const computeRack = Math.floor(rank / 32);
  const rackInSuperNode = Math.floor(rankInSuperNode / 32);
  const nodeInRack = Math.floor((rank % 32) / 8);
  const npuInNode = rank % 8;
  return { superNode, computeRack, rackInSuperNode, nodeInRack, npuInNode };
}
```

## 6. UI Changes

Add a compact top control in the stage toolbar:

- Label: `Global`
- Segmented options: `32` and `1024`
- Default: `32`
- Persist to localStorage key: `pangu-moe-trainviz-placement-mode`

Behavior:

- `32`: current 3D hardware plane, current swimlane, current Card Load.
- `1024`: global physical placement layer, grouped by supernode/rack/node/NPU.

Use PTO segmented-control tokens/classes or the existing page button style. Do not create a new private button language.

## 7. Main 3D View

### 32 Slice

Keep current layout:

- `NODE_DEFS = DP x PP`
- cards inside each node are `TP x EP`
- 1F1B trace drives active card colors and communication arcs

### 1024 Global

Add a new hardware layer:

- Supernode bands on the ground plane.
- Rack columns within each supernode.
- 8-NPU node grouping inside each rack.
- 1024 NPU instances rendered with `THREE.InstancedMesh`.

Default labels should be aggregate labels only:

- `SN0`
- `Rack 00-11`
- `DP0-31`
- `PP0-7`
- `TP0-3`

Show individual rank labels only on hover/selection.

Hover tooltip should include:

- global rank
- `DP / PP / TP`
- `SuperNode / Rack / Node / NPU`
- PP layer range
- object projection match, if selected

## 8. Model Layer / PP Mapping

For `openPangu-R-72B` current page model:

- total layers: 50
- dense layers: `L0-L3`
- MoE layers: `L4-L49`
- global PP: 8

Use an even layer partition helper:

```js
function layerRangeForPp(pp, ppCount = 8, totalLayers = 50) {
  const base = Math.floor(totalLayers / ppCount);
  const rem = totalLayers % ppCount;
  const count = base + (pp < rem ? 1 : 0);
  let start = 0;
  for (let i = 0; i < pp; i += 1) start += base + (i < rem ? 1 : 0);
  return [start, start + count - 1];
}
```

This yields PP ranges for visual grouping. If a real training config provides stage ranges, it should override the helper.

## 9. Bottom Pane Behavior

### Timeline

- Keep the existing 32-rank timeline as the measured/simulated diagnostic slice.
- In 1024 mode, show meta copy: `1024 global placement · timeline remains 32-rank slice evidence`.
- Do not render 1024 swimlane rows by default.

### Card Load

Add aggregate rendering in 1024 mode:

- default: rack-level heatmap
- click rack: node-level detail
- click node: 8 NPU cards

Metrics can be projected from current rank model:

- util = deterministic placement projection or selected slice evidence
- comm = projected by logical group, not measured profiler
- state = ok/warn/alert based on projected value

### Rank Load

If reintroduced as a separate view, render:

- `DP x PP` matrix
- each cell has four TP stripes
- hover gives rank range

## 10. Object Mapping

Object tags from `training-object-registry.js` should map to predicates:

- `tp_col:2/4` -> all ranks where `tp === 2`
- `pp:3/8` -> all ranks where `pp === 3`
- `dp:7/32` -> all ranks where `dp === 7`
- `ep:*` -> if global preset has no EP dimension, show as MoE logical shard only
- `cp:*` / `sp:*` -> show as logical overlay, not physical rank split unless config supplies dimensions

Inspector copy must distinguish:

- `slice evidence`: proven by current 32-rank demo/trace
- `global projection`: inferred from parallel placement config

## 11. Interaction Rules

- Clicking `1024` switches physical layer and bottom Card Load aggregation.
- Clicking an object in the left list highlights matching global ranks in 1024 mode.
- Clicking a rank/rack in 1024 mode opens the Focus inspector with both logical and physical coordinates.
- Switching back to `32` restores current focus/diagnostic behavior.
- View presets `axis / front / side` must work in both modes.

## 12. Performance Requirements

- 1024 individual ranks must be instanced, not separate DOM labels.
- Label count should be capped by LOD:
  - default: supernode/rack labels
  - hover: one rank tooltip
  - selected: selected rank/rack labels
- No more than one full 1024 placement rebuild per mode switch.
- Theme switching should update material colors without rebuilding rank coordinates.

## 13. Validation Checklist

- `global1024` produces exactly 1024 unique ranks.
- `globalCoordsOf(globalRankOf(coords))` round-trips for all ranks.
- Physical placement count:
  - 3 supernode groups
  - 32 compute racks
  - 128 8-NPU nodes
  - 1024 NPU/rank instances
- `32` mode still parses and renders existing trace.
- `1024` mode does not change training step playback semantics.
- Left object selection highlights expected rank predicate.
- Light/dark mode keeps labels readable.
- Front/side/isometric camera presets remain nonblank and framed.

## 14. Risks / Open Questions

- Huawei physical topology data may differ from this CloudMatrix384-style projection. Keep topology replaceable.
- Current `train.txt` is UX/spec narrative, not a machine-readable training config. If a real config appears, parse that first.
- Global EP/CP/SP dimensions are not confirmed for the `DP32 x PP8 x TP4 = 1024` target. Treat them as logical overlays until sourced.
- Existing 32-rank trace uses `DP2 x PP4 x TP2 x EP2`, so 1024 projection must not imply the trace itself scales to 1024 measured rows.

