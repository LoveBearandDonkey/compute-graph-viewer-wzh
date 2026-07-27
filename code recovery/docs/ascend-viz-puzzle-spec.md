# Ascend Tiling Viz Workbench Spec

Status: draft  
Owner workspace: `/Users/yin/pto/tiling`  
Reference repo: `/Users/yin/gitcode/asc-devkit-master`  
Real developer scenario: `/Users/yin/gitcode/deepseekv3.2源码/lightning_indexer_prolog_quant.py`

## 1. Goal

Build an Ascend C / PyPTO version of the triton-viz visual debugging experience, with optional guided challenges derived from Triton-Puzzles.

The tool must not be a static explainer. It should let an operator developer see, step through, and reason about tiling, memory movement, block mapping, vector/cube execution, fusion synchronization, and profiling evidence while writing or tuning an actual Ascend operator.

The MVP has one primary workbench mode:

1. **Visualizer mode**
   - Show source code, trace steps, tensor regions, memory tiers, queue state, sync events, and profiling/sanitizer evidence.
   - Comparable to triton-viz's kernel visualizer/profiler/sanitizer flow, but with Ascend C/PyPTO semantics.

Deferred work:

- **Guided challenge / puzzle tasks** are not a top-level MVP mode. They are optional teaching prompts attached to a trace step or selected object.
- **Developer workflow** is not a top-level MVP mode until the tool can ingest real Ascend C/PyPTO source, expose tiling/config edits, regenerate a trace, and map results back to code.

## 2. Non-Goals

- Do not directly run Triton kernels.
- Do not reuse triton-viz runtime as an engine. Its value here is interaction pattern and mental model, not execution semantics.
- Do not build a private visual style system inside `/Users/yin/pto/tiling`.
- Do not require automatic parsing of every Ascend C dialect feature in MVP.
- Do not require NPU hardware for the first UI milestone. Runtime evidence can be attached later from profiling/sanitizer outputs.
- Do not attempt full CANN compiler simulation. The tool should show the operator developer's tiling and execution plan, then overlay real evidence when available.

## 3. Open Source / Official References

### 3.1 UX References

- `triton-viz`
  - Reference role: visualizing tile-based kernel programming, memory load/store, matmul, trace save/load, profiler, sanitizer.
  - Adaptation: `tl.load`, `tl.store`, `tl.dot` become Ascend operations such as `DataCopy`, `LoadData`, `Mmad`, `Fixpipe`, and PyPTO `matmul`.

- `Triton-Puzzles`
  - Reference role: progressive interactive puzzles that teach tile/block programming without making the first experience depend on production hardware.
  - Adaptation: puzzle tasks teach Ascend C/PyPTO tiling concepts instead of Triton program IDs and pointer expressions.

### 3.2 Ascend Official Reference

Primary source: `/Users/yin/gitcode/asc-devkit-master`

Key official concepts:

- API levels:
  - `TPipe/TQue` framework API
  - basic C++ tensor API
  - SIMD/SIMT C API
  - high-level APIs
  - operator templates
- Execution and memory concepts:
  - `GlobalTensor`
  - `LocalTensor`
  - GM, UB, L1, A1, B1, A2, B2, CO1
  - `GetBlockIdx`
  - `DataCopy`
  - `DataCopyPad`
  - `LoadData`
  - `Mmad`
  - `Fixpipe`
  - `TPipe`
  - `TQue`
  - `SetFlag` / `WaitFlag`
  - `CrossCoreSetFlag` / `CrossCoreWaitFlag`
  - `PipeBarrier`
  - `__vector__`, `__cube__`, `__mix__(1, 2)`
- Tool evidence:
  - printf
  - assert
  - dump
  - CPU debug
  - profiling
  - sanitizer
  - clock
  - simulator

### 3.3 UI Reference Bundle

Canonical PTO UI reference:

`/Users/yin/pto-design-system/SKILL.md`

This spec follows the PTO design-system skill as the UI governance source. The tiling workbench is a layout-heavy, visualization-heavy PTO page, so implementation must use **Workflow C: Pattern-first PTO page or block**.

Required UI baseline before implementation, in this order:

1. `/Users/yin/pto-design-system/references/DESIGN.md`
   - canonical system spec for theme, surfaces, typography, spacing, components, and governance.
2. `/Users/yin/pto-design-system/references/quick-reference.md`
   - token and class cheat sheet for implementation.
3. `/Users/yin/pto-design-system/references/retrofit-container-audit.md`
   - required when migrating or borrowing existing demo layouts.
4. `/Users/yin/pto-design-system/patterns/patterns.json`
   - shared pattern registry.
5. Matched `pattern.json` files under `/Users/yin/pto-design-system/patterns/<pattern-id>/pattern.json`
   - canonical reuse contracts for each pattern, including required APIs, allowed overrides, and forbidden overrides.
6. `/Users/yin/pto-design-system/tokens/foundation.css`
7. `/Users/yin/pto-design-system/tokens/semantic.css`
8. `/Users/yin/pto-design-system/tokens/components.css`
9. `/Users/yin/pto-design-system/css/style.css`
10. `/Users/yin/pto-design-system/design-system-preview.html`
    - helper for base components only. It is not canonical for full IDE, graph, timeline, architecture, or playback pages.

Runtime pages under `/Users/yin/pto/tiling` should consume the checked-in PTO module asset path used by the local PTO workspace. Existing PTO pages commonly load `../vendor/pto-design-system/...`; implementation must verify the concrete link path before writing HTML.

UI rule:

Do not invent a new button, toggle, badge, card, panel, spacing scale, color language, IDE shell, split-pane kernel, swimlane task bar, memory architecture block, AIC/AIV object, or playback chrome inside this module. Use the shared pattern contracts first. If a needed transfer/pipeline visual is not covered, create `/Users/yin/pto/tiling/component-preview.html` and wait for approval before using it in the real workbench.

### 3.4 Teaching Content Reference

The conceptual teaching content for the first product page should be sourced from:

`/Users/yin/pto/tiling/docs/ascend-tiling-visualization-knowledge.md`

That document explains the core mental model this spec relies on:

- GM is a flat address space, not a physical 3D block.
- Tensor shape, layout, and stride create a logical tensor view over flat GM buffers.
- Tiling maps the logical tensor view to `blockIdx`, loop progress, tile ranges, and memory offsets.
- CopyIn / Compute / CopyOut describe the lifecycle of one logical tile through GM and on-chip buffers.
- The 3D viewport answers "which part of the logical tensor is this step touching?"
- The memory architecture diagram answers "which hardware objects and routes does this tile pass through?"

## 4. Core Visualization Model

The workbench must be **3D tensor-first**, not timeline-first.

The reference UX is Triton Viz: a large 3D tensor/program-space viewport with side controls and code linkage. The Ascend version must preserve that core effect while changing the semantics from Triton `tl.load` / `tl.store` / `tl.dot` to Ascend C / PyPTO tiling, memory, and compute concepts.

### 4.1 Full Tensor Space Is Logical, Not Physical

The 3D viewport must never imply that GM physically stores a 3D tensor.

Required interpretation:

- GM is flat memory.
- `GlobalTensor`, tensor shape, layout, stride, and offset define logical tensor views over GM.
- The 3D viewport renders this logical access space.
- Highlighted tiles represent current logical ranges that map back to flat GM offsets.

For a 1D vector sample, the viewport may fold execution structure into:

```text
X axis: element inside tile
Y axis: tileIdx / progress
Z axis: blockIdx / core
```

For matmul, the viewport may use:

```text
X axis: N tile
Y axis: M tile
Z axis: K tile / reduction step
```

The `Z` axis in matmul is not output tensor depth. It is a visualization of K-axis accumulation.

### 4.2 Required Visual Layers

Every trace step must drive four synchronized visual layers:

1. **3D full tensor viewport**
   - primary center-stage visualization.
   - shows the full logical tensor or tensor set.
   - highlights current block/tile/slice.
   - shows load, store, mask, padding, reduction, and output commit states.

2. **On-chip tile lens**
   - compact overlay cards inside the 3D viewport for the current selected tile.
   - shows local tile residency in UB, L1, L0A, L0B, L0C, CO1, or Vector/Cube staging objects.
   - does not replace the full tensor view.

3. **Memory architecture path focus**
   - uses `memory-architecture-layout`.
   - shows hardware route focus through GM/L2/UB/L1/L0A/L0B/L0C/Cube/Vector/C-V lanes.
   - highlights active hardware nodes and routes.
   - may show current data block residency/occupancy inside AIC/AIV buffer grids when the trace provides local buffer-block state.
   - does not replace the full logical tensor tile grid.

4. **Source / inspector / playback**
   - source lines, formulas, offsets, queue events, sync events, and evidence update with the same step.

### 4.3 3D Tensor Viewport Requirements

The 3D tensor viewport must be implemented as a real 3D scene, preferably using Three.js and OrbitControls. Do not approximate the main Triton Viz effect with flat cards, static SVG, or timeline bars.

Functional requirements:

- pan/orbit/zoom around the tensor block.
- render full tensor bounds.
- render grid/cell/voxel or instanced block representation.
- highlight selected tile range.
- support drag pan plus explicit zoom buttons and fit/reset.
- do not zoom the 3D tensor viewport on ordinary mouse wheel.
- support multiple tensor surfaces when needed, such as `x/y/z`, `A/B/C`, or `input/output/workspace`.
- keep short axis names visible in the 3D scene so the user can orient the logical space.
- keep operation names, detailed axis explanations, and the "GM is flat" explanation out of always-visible chrome. They may appear in hover tips, inspector content, or selected-object detail.
- support selected program/core/tile controls similar to Triton Viz Program IDs, but named for Ascend semantics:
  - `blockIdx`
  - `progress`
  - `mIndex`
  - `nIndex`
  - `kIndex`
  - AIV sub-block when relevant
- expose a no-WebGL fallback that still shows a 2D tensor/tile map and a clear warning.

Implementation note:

Triton Viz front-end code is MIT-licensed and useful as a reference, but it cannot be dropped in directly. Its `OpRecord`, program IDs, load/store descriptors, pointer offsets, masks, and backend endpoints are Triton-specific. Ascend Viz should reuse the visual pattern and selected rendering ideas, not the runtime data contract.

If code is copied or adapted from Triton Viz:

- preserve license and attribution.
- vendor dependencies locally when possible.
- replace CDN imports with project-managed assets if offline preview is required.
- wrap the viewport in PTO shell/chrome and PTO controls.
- keep non-scene UI token-derived.

### 4.4 Memory Architecture Path Focus Requirements

The `hardware-architecture-viewport` pattern must host the memory architecture stage, and the `memory-architecture-layout` pattern must render the hardware path content inside it.

Required interaction contract:

- reuse `window.PtoHardwareArchitectureViewport.mount(...)` for pan, detail toggle, zoom controls, readout, and stage background.
- ordinary wheel must not zoom the architecture map.
- macOS `Command + wheel` (`event.metaKey`) zooms around the pointer, matching the accurate hardware-map reference page.
- drag pans the architecture canvas.
- product code may pass focus/block state into the shared memory architecture APIs, but must not fork viewport pan/zoom behavior.

Use it to answer:

```text
This tile is moving through which hardware objects?
Which route is active?
Which hardware objects are unrelated for this step?
Which intermediate route is avoided by fusion?
Which on-chip buffer currently hosts this data block?
Which local buffer segment/cells are occupied by the selected data block, if buffer cell state is available?
```

Do not use it to answer:

```text
Which cells of the full logical tensor are touched?
Which M/N/K tile is selected?
Which output elements are committed?
```

Those questions belong to the 3D tensor viewport.

Buffer grid role:

- Existing AIC/AIV/memory architecture patterns already render capacity-like buffer grids.
- Those grids are appropriate for **local on-chip data block residency**, such as `xLocal` in UB, `A[mTile,kTile]` in L0A, `B[kTile,nTile]` in L0B, or partial `C[mTile,nTile]` in L0C/CO1.
- Those grids are not appropriate for rendering the entire logical tensor space, such as full `x[0:N]`, full `A[M,K]`, or full `C[M,N]`.
- Product pages must not hard-code grid cell colors by reaching into generated DOM. Dynamic cell state must be added through shared pattern APIs first.

Path focus mappings required by MVP:

| Step type | Architecture focus |
|---|---|
| Vector CopyIn | L2/GM rail -> AIV UB route, plus source input tensor node metadata |
| Vector Compute | AIV UB + Vector execution object |
| Vector CopyOut | AIV UB -> L2/GM route |
| Cube CopyIn/LoadData | L2/GM rail -> AIC L1 -> L0A/L0B |
| Cube Mmad | L0A + L0B + Cube + L0C |
| Cube Fixpipe/CopyOut | L0C/CO1 -> GM C route or output tier |
| Fusion sync | AIC result producer + AIV consumer lanes + CrossCore flag state |
| Fusion direct C-V | L0C -> UB / UB -> L1 route, with GM workspace marked avoided when applicable |

Existing `memory-architecture-layout` already supports `setPathFocus(container, preset, { selectors, routes, errorSelectors })`. Product pages must call this API rather than reimplementing route geometry or focus classes locally.

### 4.5 Shared Pattern API: Buffer Grid Cell State

`aic-core-object`, `aiv-core-object`, and `memory-architecture-layout` patterns render buffer grids. Their public contracts must expose dynamic local data block occupancy through a shared API, so product pages never color generated cells directly.

Required API:

```js
window.PtoAicCorePattern.setBufferBlocks(root, blocks)
window.PtoAivCorePattern.setBufferBlocks(root, blocks)
window.PtoMemoryArchitecturePattern.setBufferBlocks(root, blocks)
window.PtoMemoryArchitecturePattern.clearBufferBlocks(root)
```

Proposed block payload:

```json
{
  "core": "mem950-aiv1",
  "buffer": "UB",
  "label": "xLocal",
  "sourceTile": "x[block0,progress0,:]",
  "state": "enqueued",
  "cellRange": [0, 15],
  "tone": "input"
}
```

Required behavior:

- cell state must be applied by the shared pattern, not by page-local CSS selectors.
- cells should support states such as `loaded`, `allocated`, `enqueued`, `dequeued`, `accumulating`, `committed`, and `avoided`.
- each block should expose tooltip metadata linking back to source tile, GM range, queue, or operation.
- buffer block coloration must remain token-derived or pattern-defined.
- if a downstream page is wired to an older pattern copy without this API, it must fall back to the separate on-chip tile lens/component preview instead of mutating shared pattern internals.

### 4.6 Step State Contract

Each visualized step must carry enough state to update all visual layers.

Minimum step state:

- source refs.
- current block/core identifiers.
- loop identifiers.
- logical tensor view identifiers.
- selected logical tile/range.
- flat GM offset expressions and resolved ranges when known.
- on-chip residency state.
- memory architecture focus payload.
- queue/sync state.
- optional evidence pointers.

The same step should be able to answer:

```text
Where is this tile in the full logical tensor?
What flat GM range does it map to?
Which on-chip buffer currently holds it?
Which hardware route is active?
Which source lines created this state?
```

## 5. MVP Samples

MVP must include exactly three official sample traces first. These should be deterministic fixtures before automatic parser work begins.

### 5.1 Sample A: Vector Add with TPipe/TQue

Source:

`/Users/yin/gitcode/asc-devkit-master/examples/01_simd_cpp_api/00_introduction/01_vector/add_tpipe_tque/add.asc`

Purpose:

- Teach basic vector tiling and queue-driven memory flow.
- Show the simplest complete shape-to-block-to-tile-to-copy-to-compute-to-copyout loop.

Key source concepts:

- `AddCustomTilingData`
  - `totalLength`
  - `tileNum`
- `KernelAdd::Init`
  - `blockLength = totalLength / GetBlockNum()`
  - `tileLength = blockLength / tileNum / BUFFER_NUM`
  - `xGm/yGm/zGm.SetGlobalBuffer(... + blockLength * GetBlockIdx())`
  - `pipe.InitBuffer(...)`
- `KernelAdd::Process`
  - `loopCount = tileNum * BUFFER_NUM`
  - loop of `CopyIn(i)`, `Compute(i)`, `CopyOut(i)`
- `CopyIn`
  - `AllocTensor`
  - `DataCopy(GM -> VECIN)`
  - `EnQue`
- `Compute`
  - `DeQue`
  - `AllocTensor`
  - `Add`
  - `EnQue`
  - `FreeTensor`
- `CopyOut`
  - `DeQue`
  - `DataCopy(VECOUT -> GM)`
  - `FreeTensor`

Required visualization:

- 3D full tensor viewport:
  - primary view shows the full logical vector space, not a physical GM shape.
  - fold the 1D GM buffers into `[blockIdx, progress, element]` for teaching and navigation.
  - render `x`, `y`, and `z` as related tensor surfaces or selectable tensor layers.
  - selected `blockIdx` and `progress` highlight the active tile range.
  - CopyIn highlights source `x/y` GM slices.
  - Compute keeps the selected logical tile outline active while showing local `xLocal + yLocal -> zLocal`.
  - CopyOut fills or commits the corresponding `z` slice.
- On-chip tile lens:
  - show `xLocal`, `yLocal`, `zLocal` tile slots.
  - show `BUFFER_NUM` queue slots for double buffer context.
  - local tile lens must be small and explanatory; it must not replace the full tensor view.
- Memory architecture path focus:
  - CopyIn calls `memory-architecture-layout` focus for L2/GM -> AIV UB.
  - Compute focuses AIV UB + Vector.
  - CopyOut focuses AIV UB -> L2/GM.
  - This layer explains the hardware route only; tensor tile coloration remains in the 3D viewport.
- Host launch strip:
  - input files
  - H2D copy
  - `add_custom<<<numBlocks, nullptr, stream>>>`
  - stream sync
  - D2H copy
- Block map:
  - eight blocks over a contiguous vector.
  - selected block highlights GM range.
- Tile timeline:
  - each progress index maps to `progress * tileLength`.
  - selected step shows `xGm`, `yGm`, `zGm` read/write regions.
- Queue panel:
  - `inQueueX`, `inQueueY`, `outQueueZ`.
  - queue slots show `empty`, `allocated`, `enqueued`, `dequeued`, `freed`.
- Memory panel:
  - GM and UB/VECIN/VECOUT regions.
- Puzzle tasks:
  - Given `totalLength`, `numBlocks`, `tileNum`, `BUFFER_NUM`, compute `blockLength` and `tileLength`.
  - Given `blockIdx` and `progress`, identify x/y read range and z write range.
  - Predict queue state after `CopyIn`, `Compute`, and `CopyOut`.

### 5.2 Sample B: Cube Matmul

Source:

`/Users/yin/gitcode/asc-devkit-master/examples/01_simd_cpp_api/00_introduction/02_matrix/matmul/matmul.asc`

Purpose:

- Teach Cube path and matrix tile movement.
- Show the core `GM -> L1/A1/B1 -> L0/A2/B2 -> CO1 -> GM` pipeline.

Key source concepts:

- template parameters:
  - `M`, `K`, `N`
  - `singleCoreM`, `singleCoreK`, `singleCoreN`
  - `baseM`, `baseK`, `baseN`
- `InitGMOffsets`
  - `mIter = M / singleCoreM`
  - `mIterIdx = GetBlockIdx() % mIter`
  - `nIterIdx = GetBlockIdx() / mIter`
  - `gmOffsetA`, `gmOffsetB`, `gmOffsetC`
- local tensors:
  - `A1`, `A2`, `B1`, `B2`, `CO1`
- nested loops:
  - `mIndex`
  - `nIndex`
  - `kIndex`
- movement and compute:
  - `CopyInA`: GM A ND to A1 Nz.
  - `CopyInB`: GM B ND to B1 Nz.
  - `DataLoadA`: A1 to A2.
  - `DataLoadB`: B1 to B2, with transpose semantics.
  - `Mmad`: A2/B2 to CO1.
  - `Fixpipe`: CO1 to GM C, format and dtype conversion.
- architecture branches:
  - `__NPU_ARCH__ == 2201`
  - `__NPU_ARCH__ == 3510`

Required visualization:

- 3D full tensor viewport:
  - primary view shows logical `A[M,K]`, `B[K,N]`, and `C[M,N]` tensor spaces.
  - selected `blockIdx` highlights the C output tile determined by `mIterIdx` and `nIterIdx`.
  - `kIndex` highlights the current A/B reduction tiles.
  - the Z/depth dimension may represent K-axis accumulation, not output tensor physical depth.
  - first `kIndex` step visually marks C/L0C initialization; later `kIndex` steps visually mark accumulation.
- On-chip tile lens:
  - show current A tile in L1/A1 and L0A/A2.
  - show current B tile in L1/B1 and L0B/B2.
  - show C partial result in CO1/L0C.
  - distinguish load, compute, accumulate, and fixpipe states.
- Memory architecture path focus:
  - CopyIn / LoadData focuses L2/GM -> AIC L1 -> L0A/L0B.
  - Mmad focuses L0A + L0B + Cube + L0C.
  - Fixpipe / output focuses L0C/CO1 -> output GM C route.
- Matrix block map:
  - C matrix divided by `singleCoreM` and `singleCoreN`.
  - selected `blockIdx` maps to C tile.
- K accumulation timeline:
  - `kIndex` shows repeated A/B tile movement and accumulation into CO1.
  - first `kIndex` has `cmatrixInitVal = true`; later steps accumulate.
- Memory tiers:
  - GM A/B/C.
  - A1/B1 in L1.
  - A2/B2 in L0A/L0B.
  - CO1/L0C.
- Data movement arrows:
  - `DataCopy`
  - `LoadData`
  - `Mmad`
  - `Fixpipe`
- Architecture toggle:
  - `dav-2201`
  - `dav-3510`
  - show different `LoadData2DParams` / `LoadData2DParamsV2` parameter panels.
- Puzzle tasks:
  - Given `blockIdx`, identify `mIterIdx`, `nIterIdx`, and C tile.
  - Given `mIndex/nIndex/kIndex`, identify GM offsets.
  - Predict which memory tier contains A/B/C at each step.
  - Explain why B uses transpose in `LoadDataB`.

### 5.3 Sample C: Fusion Matmul + LeakyRelu

Source:

`/Users/yin/gitcode/asc-devkit-master/examples/01_simd_cpp_api/00_introduction/03_fusion_operation/matmul_leakyrelu_basic_api/matmul_leakyrelu_basic_api.asc`

Purpose:

- Teach fusion as more than "two ops in a row".
- Show Cube/Vector collaboration, cross-core synchronization, and split ownership of one output tile.

Why this sample is required:

- It covers the MVP's fusion requirement better than a single-core fused API sample.
- It has both Cube and Vector roles.
- It has explicit `__mix__(1, 2)` and `CrossCoreSetFlag` / `CrossCoreWaitFlag`.
- It shows one Cube result consumed by two Vector cores.

Key source concepts:

- `__global__ __mix__(1, 2) void mmad_vec_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c)`
  - Kernel uses Cube:Vector ratio 1:2.
- `ASCEND_IS_AIC`
  - Cube path computes Matmul.
- `ASCEND_IS_AIV`
  - Vector path waits, then applies LeakyRelu.
- `InitGMOffsetsMatrix`
  - Cube block maps to C tile.
- `InitGMOffsetsVector`
  - Vector block maps to half of the Cube result.
  - `GetBlockIdx() / 2` maps AIV pairs to AIC tile.
  - `GetBlockIdx() % 2` selects top or bottom half.
- Cube flow:
  - `CopyInA`
  - `CopyInB`
  - `DataLoadA`
  - `DataLoadB`
  - `Mmad`
  - `Fixpipe`
  - `CrossCoreSetFlag`
- Vector flow:
  - `CrossCoreWaitFlag`
  - `DataCopyPad GM -> VECCALC`
  - `LeakyRelu`
  - `DataCopyPad VECCALC -> GM`

Required visualization:

- 3D full tensor viewport:
  - primary view shows `A`, `B`, and final `C` / output logical tensor spaces.
  - selected AIC block highlights the full C tile produced by Cube.
  - selected AIV block highlights the top or bottom half of that C tile according to `GetBlockIdx() % 2`.
  - if the sample path uses GM as an intermediate, show that GM intermediate explicitly.
  - if a direct C-V path is being taught, mark the GM workspace as avoided rather than pretending the path did not exist.
- On-chip tile lens:
  - show AIC-side L0C/CO1 producer tile.
  - show AIV-side UB tile for LeakyRelu.
  - show split ownership between the two AIV consumers.
- Memory architecture path focus:
  - AIC Matmul focuses L2/GM -> AIC L1/L0A/L0B -> Cube -> L0C.
  - CrossCoreSetFlag / CrossCoreWaitFlag focuses producer/consumer lane synchronization.
  - Vector epilogue focuses AIV UB + Vector + output route.
  - Direct C-V teaching focuses L0C -> UB / UB -> L1 routes through `memory-architecture-layout`.
- Cross-core swimlane:
  - one AIC lane.
  - two AIV lanes.
  - AIC produces `baseM * baseN`.
  - AIV0 handles first `baseM / 2 * baseN`.
  - AIV1 handles second `baseM / 2 * baseN`.
- Sync event overlay:
  - `CrossCoreSetFlag` on AIC lane.
  - `CrossCoreWaitFlag` gates both AIV lanes.
- C tile ownership:
  - C tile split horizontally into two vector subtiles.
  - selected AIV block shows `GetBlockIdx() % 2` region.
- Memory flow:
  - AIC: GM/L1/L0/CO1/GM.
  - AIV: GM/UB/GM.
- Puzzle tasks:
  - Given Cube `blockIdx`, identify the paired Vector `blockIdx` values.
  - Given Vector `blockIdx`, identify which half of the C tile it owns.
  - Predict when AIV can start.
  - Identify why direct fusion still needs synchronization.

## 6. Real Developer Scenario: Lightning Indexer Tiling

The MVP must demonstrate how the same tool applies to real code, using the local PyPTO operator:

`/Users/yin/gitcode/deepseekv3.2源码/lightning_indexer_prolog_quant.py`

The user may call it "lighting indexer"; the source file uses `lightning_indexer_prolog_quant`.

### 6.1 Why This Scenario Matters

This file already contains the right hooks for visual tiling:

- semantic labels:
  - `Key-LayerNorm`
  - `Key-Rope2D`
  - `Prolog-Quant`
  - `Query-Linear`
  - `Query-Dequant`
  - `Query-Hadamard`
  - `Query-Quant`
  - `Key-Linear`
  - `Key-Hadamard`
  - `Key-Quant`
  - `Weight-Linear`
- cube tile controls:
  - `pypto.set_cube_tile_shapes(...)`
  - `q_linear`
  - `q_hd`
  - `k_linear`
  - `w_linear`
- vector tile controls:
  - `pypto.set_vec_tile_shapes(...)`
- dynamic and loop controls:
  - `pypto.mark_dynamic(...)`
  - `pypto.loop(...)`
  - `pypto.loop_unroll(...)`
  - `unroll_list`
- pass options:
  - `nbuffer_merge_mode`
  - `l1_reuse_map`
  - `copyin_threshold`
  - `cycle_upper_bound`
- memory/scatter semantics:
  - `pypto.view`
  - `pypto.reshape`
  - `pypto.matmul`
  - `pypto.assemble`
  - `pypto.scatter_update`

### 6.2 Developer Workflow

Target workflow for a developer writing or tuning `lightning_indexer_prolog_quant`:

1. Developer opens the operator in the workbench.
2. Tool extracts semantic stages from `pypto.set_semantic_label`.
3. Tool extracts tile shapes from `set_cube_tile_shapes` and `set_vec_tile_shapes`.
4. Tool extracts loop structure from `loop`, `loop_unroll`, `tIdx`, and `unrollLength`.
5. Tool builds an execution plan:
   - dynamic axis: `t`
   - unrolled slices: `tIdx`, `t_tile`
   - Query path
   - Key path
   - Weight path
   - output assembly/scatter
6. UI renders the plan.
7. Developer changes configs:
   - `q_linear`
   - `q_hd`
   - `k_linear`
   - `w_linear`
   - `unroll_list`
   - `l1_reuse_param`
   - `copy_in_threshold`
   - `cycle_upper_bound`
8. Tool re-renders affected stages without requiring a full compiler simulation.
9. If profiling/sanitizer evidence exists, tool overlays it:
   - slow semantic stage
   - memory hot spot
   - exceeded copy threshold
   - invalid/scatter risk
   - unbalanced vector/cube stage
10. Developer decides whether to change tiling, split a stage, increase/decrease unroll, or adjust reuse.

### 6.3 Lightning Indexer Stage View

The workbench should show three stage columns or tabs:

1. Query path
   - `Query-Linear`
   - `Query-Dequant`
   - RoPE split/recompose
   - `Query-Hadamard`
   - `Query-Quant`
   - `assemble q_int8/q_scale`

2. Key path
   - `Key-Linear`
   - `Key-LayerNorm`
   - `Key-Rope2D`
   - `Key-Hadamard`
   - `Key-Quant`
   - `scatter_update k_cache/k_cache_scale`

3. Weight path
   - `Weight-Linear`
   - scaling by `sqrt(head_num) * sqrt(head_dim)`
   - output `weights`

For each stage, inspector shows:

- source label
- source line
- input tensors and shapes
- output tensors and shapes
- cube/vector tile shape
- loop index and valid shape
- memory movement summary
- likely memory tier
- related config fields
- profiling evidence if available

### 6.4 Lightning Indexer Guided Challenge Tasks

Guided challenges should include real-code tasks when the developer workflow is available:

- Given `t`, `unroll_list`, and selected `tIdx`, identify `t_tile`.
- Given `q_linear`, identify the cube tile dimensions used by `Query-Linear`.
- Given `rope_head_dim`, identify `q_rope` and `q_nope` views.
- Given `k_cache_index`, predict scatter destination for `k_cache`.
- Given `l1_reuse_map`, explain which stages can reuse L1 and which cannot.
- Given `copyin_threshold`, identify a stage likely to trigger copy-in pressure.

## 7. Trace Model

The core artifact is an Ascend visualization trace. Working name: `.avz`.

MVP may use plain JSON files. Later `.avz` can be a zip container.

### 7.1 AVZ Container Layout

```text
operator.avz
├── manifest.json
├── source/
│   ├── operator.asc
│   └── operator.py
├── trace.json
├── tensors.json
├── evidence/
│   ├── profiling.json
│   ├── sanitizer.json
│   ├── dump.json
│   └── clock.json
└── assets/
    └── screenshots-or-notes
```

### 7.2 Trace JSON Top Level

```json
{
  "schemaVersion": "0.1",
  "operator": {
    "id": "sample.add_tpipe_tque",
    "name": "add_custom",
    "kind": "vector",
    "sourceLanguage": "ascendc",
    "apiLevel": "tpipe_tque",
    "sourcePath": "/Users/yin/gitcode/asc-devkit-master/examples/01_simd_cpp_api/00_introduction/01_vector/add_tpipe_tque/add.asc"
  },
  "arch": {
    "npuArch": "dav-2201",
    "soc": "Atlas A2/A3",
    "kernelTag": "__vector__"
  },
  "launch": {},
  "tiling": {},
  "tensorViews": [],
  "memory": {},
  "stages": [],
  "steps": [],
  "puzzles": [],
  "evidence": {}
}
```

### 7.3 Stage Object

```json
{
  "id": "copy_in_x_y",
  "label": "CopyIn",
  "semanticLabel": "Vector-CopyIn",
  "sourceRef": {
    "path": "add.asc",
    "line": 56,
    "symbol": "KernelAdd::CopyIn"
  },
  "unit": "vector",
  "operations": ["AllocTensor", "DataCopy", "EnQue"],
  "inputs": ["xGm", "yGm"],
  "outputs": ["xLocal", "yLocal"],
  "memoryTransfers": [
    {
      "from": "GM:x",
      "to": "UB:VECIN:xLocal",
      "lengthExpr": "tileLength",
      "offsetExpr": "progress * tileLength"
    }
  ],
  "queueEvents": [
    {"queue": "inQueueX", "event": "AllocTensor"},
    {"queue": "inQueueX", "event": "EnQue"}
  ]
}
```

### 7.4 Step Object

```json
{
  "id": "block0.progress0.copy_in",
  "stageId": "copy_in_x_y",
  "blockIdx": 0,
  "loop": {
    "progress": 0
  },
  "sourceRefs": [
    {"path": "add.asc", "line": 60},
    {"path": "add.asc", "line": 61}
  ],
  "highlights": {
    "sourceLines": [60, 61],
    "memoryRegions": ["GM:x[0:128]", "GM:y[0:128]", "UB:xLocal", "UB:yLocal"],
    "queues": ["inQueueX", "inQueueY"]
  },
  "visualState": {
    "tensorViewport": {
      "activeViews": ["x", "y", "z"],
      "selectedTiles": [
        {
          "viewId": "x",
          "role": "load",
          "logicalRange": {
            "blockIdx": 0,
            "progress": 0,
            "elementRange": [0, 128]
          },
          "gmRange": {
            "offsetExpr": "blockIdx * blockLength + progress * tileLength",
            "offset": 0,
            "lengthExpr": "tileLength",
            "length": 128
          },
          "state": "read"
        }
      ],
      "cameraPreset": "vector-block-progress-element"
    },
    "onChipLens": {
      "buffers": [
        {"id": "xLocal", "tier": "UB", "queue": "inQueueX", "state": "allocated"},
        {"id": "yLocal", "tier": "UB", "queue": "inQueueY", "state": "allocated"}
      ]
    },
    "architectureFocus": {
      "preset": "ascend950b",
      "selectors": [
        "[data-mem950-node=\"rail:L2\"]",
        "#mem950-aiv1 [data-aiv-node=\"buffer:UB\"]"
      ],
      "routes": ["l2-to-aiv1"],
      "bufferBlocks": [
        {
          "core": "mem950-aiv1",
          "buffer": "UB",
          "label": "xLocal",
          "sourceTile": "x[block0,progress0,:]",
          "state": "loaded",
          "cellRange": [0, 15],
          "tone": "input"
        }
      ]
    }
  },
  "explanation": "Copy x/y tile from GM into local VECIN queue buffers."
}
```

### 7.5 Tensor View Object

`tensorViews` describes how flat memory and logical tensor dimensions are rendered in the 3D viewport.

Vector Add example:

```json
{
  "id": "x",
  "label": "GM:x",
  "storage": "GM",
  "dtype": "fp32",
  "physicalAddressing": {
    "base": "xGm",
    "layout": "flat",
    "length": 16384
  },
  "logicalShape": [8, 16, 128],
  "logicalAxes": [
    {"name": "blockIdx", "meaning": "core/block assignment"},
    {"name": "progress", "meaning": "tile loop progress"},
    {"name": "element", "meaning": "element inside tile"}
  ],
  "visualRole": "input",
  "render": {
    "kind": "voxelGrid",
    "collapseThreshold": 4096,
    "axisPreset": "block-progress-element"
  }
}
```

Matmul example:

```json
{
  "id": "C",
  "label": "GM:C",
  "storage": "GM",
  "dtype": "fp16",
  "physicalAddressing": {
    "base": "cGm",
    "layout": "ND",
    "shape": ["M", "N"]
  },
  "logicalShape": ["mTiles", "nTiles", "kTiles"],
  "logicalAxes": [
    {"name": "mTile", "meaning": "output M tile"},
    {"name": "nTile", "meaning": "output N tile"},
    {"name": "kTile", "meaning": "reduction progress, not output depth"}
  ],
  "visualRole": "output-accumulation",
  "render": {
    "kind": "matmulTiles",
    "axisPreset": "m-n-k"
  }
}
```

Rules:

- `physicalAddressing` explains how the logical view maps to flat or laid-out GM.
- `logicalShape` is allowed to be a visualization shape, as long as `logicalAxes` explains each axis.
- 1D vector buffers may be rendered as 3D `[blockIdx, progress, element]`.
- 2D output matrices may use a K-depth visualization only for accumulation; this must be labeled as reduction progress.
- The 3D viewport should use this object, not infer shape from labels alone.

### 7.6 Architecture Focus Object

`visualState.architectureFocus` is a direct input to `memory-architecture-layout`.

Example:

```json
{
  "preset": "ascend950b",
  "selectors": [
    "#mem950-aic [data-aic-node=\"buffer:L0C\"]",
    "#mem950-aiv1 [data-aiv-node=\"buffer:UB\"]"
  ],
  "routes": ["aic-to-aiv1"],
  "bufferBlocks": [
    {
      "core": "mem950-aic",
      "buffer": "L0C",
      "label": "C partial",
      "sourceTile": "C[m0,n0]",
      "state": "accumulating",
      "cellRange": [20, 39],
      "tone": "output"
    },
    {
      "core": "mem950-aiv1",
      "buffer": "UB",
      "label": "epilogue tile",
      "sourceTile": "C[m0,n0].topHalf",
      "state": "loaded",
      "cellRange": [0, 19],
      "tone": "fusion"
    }
  ],
  "errorSelectors": [],
  "note": "Fusion path uses L0C->UB direct lane; GM workspace is avoided."
}
```

Rules:

- Use selectors and route ids supported by `memory-architecture-layout`.
- Use `bufferBlocks` only for local on-chip buffer occupancy, not full logical tensor grids.
- Do not encode full tensor tile ranges in this object.
- For avoided intermediates, use inspector metadata or diagnostic selectors; do not fake inactive routes as active.

### 7.7 On-Chip Lens Object

`visualState.onChipLens` describes the current local tile state.

Example:

```json
{
  "buffers": [
    {"id": "A1", "tier": "L1", "state": "loaded", "sourceTile": "A[m0,k0]"},
    {"id": "A2", "tier": "L0A", "state": "loaded", "sourceTile": "A[m0,k0]"},
    {"id": "CO1", "tier": "L0C", "state": "accumulating", "targetTile": "C[m0,n0]"}
  ],
  "queues": [
    {"id": "inQueueX", "slot": 0, "state": "enqueued"}
  ],
  "operation": {
    "name": "Mmad",
    "inputs": ["A2", "B2"],
    "outputs": ["CO1"]
  }
}
```

Rules:

- The lens should show current tile residency and operation state.
- It should not draw another full tensor.
- It must remain synchronized with the selected 3D tensor tile and architecture focus.

### 7.8 Tiling Object

For Ascend C:

```json
{
  "type": "static_or_host_tiling",
  "params": {
    "totalLength": 16384,
    "numBlocks": 8,
    "tileNum": 8,
    "bufferNum": 2,
    "blockLengthExpr": "totalLength / numBlocks",
    "tileLengthExpr": "blockLength / tileNum / bufferNum"
  }
}
```

For PyPTO:

```json
{
  "type": "pypto_tiling",
  "dynamicAxes": ["t"],
  "configs": {
    "q_linear": ["L0M", "L1M", "L0K", "L1K", "L0N", "L1N"],
    "q_hd": ["L0M", "L1M", "L0K", "L1K", "L0N", "L1N"],
    "k_linear": ["L0M", "L1M", "L0K", "L1K", "L0N", "L1N"],
    "w_linear": ["L0M", "L1M", "L0K", "L1K", "L0N", "L1N"],
    "unroll_list": []
  },
  "passOptions": {
    "l1_reuse_map": {},
    "copyin_threshold": null,
    "cycle_upper_bound": null
  }
}
```

## 8. Source Extraction Strategy

MVP should use three extraction levels.

### 8.1 Level 0: Handwritten Fixture

Used for the three official samples.

Pros:

- Fast.
- Stable.
- Allows UI and schema to converge before parser work.

Deliverables:

- `data/fixtures/add_tpipe_tque.trace.json`
- `data/fixtures/matmul.trace.json`
- `data/fixtures/matmul_leakyrelu_fusion.trace.json`

### 8.2 Level 1: Lightweight Source Extractor

Used for known Ascend C and PyPTO patterns.

Ascend C extraction:

- class name
- global kernel declaration
- `__vector__`, `__cube__`, `__mix__`
- `Init`
- `Process`
- helper functions named `CopyIn`, `Compute`, `CopyOut`, `DataLoad`, `LeaklyGeluCompute`
- `GlobalTensor` and `LocalTensor`
- `DataCopy`, `DataCopyPad`, `LoadData`, `Mmad`, `Fixpipe`
- `SetFlag`, `WaitFlag`, `CrossCoreSetFlag`, `CrossCoreWaitFlag`

PyPTO extraction:

- `set_semantic_label`
- `set_cube_tile_shapes`
- `set_vec_tile_shapes`
- `mark_dynamic`
- `loop`
- `loop_unroll`
- `view`
- `reshape`
- `matmul`
- `assemble`
- `scatter_update`
- config fields in dataclasses

### 8.3 Level 2: Runtime Evidence Attachment

Used after developer runs profiling/sanitizer/dump.

Inputs:

- profiling result
- sanitizer result
- dump result
- clock result
- generated code metadata if available

The visualizer should attach evidence to stages by:

- kernel name
- source ref
- semantic label
- op name
- tensor name
- org index or profiler row id if available

## 9. UI Specification

The UI must consume PTO design system assets first, using `/Users/yin/pto-design-system/SKILL.md` as the governing UI reference. This workbench is an IDE/workbench visualization surface, so use the design-system **Workflow C: Pattern-first PTO page or block**.

Do not use `design-system-preview.html` as the page-level source of truth. It is only a helper for base components. Full-page behavior must come from `patterns/patterns.json` and each matched `patterns/<pattern-id>/pattern.json`.

### 9.1 Required PTO Patterns

Read these pattern contracts before implementing the workbench:

| Workbench need | PTO pattern id | Contract path | Usage requirement |
|---|---|---|---|
| IDE-like page shell, activity rail, panes, pane headers, inspector docks | `ide-frame` | `/Users/yin/pto-design-system/patterns/ide-frame/pattern.json` | Use as the upper-level page shell. Do not rebuild IDE chrome with generic divs. |
| Draggable split panes | `workbench-shell` | `/Users/yin/pto-design-system/patterns/workbench-shell/pattern.json` | Use only as the resize kernel. Do not override `.pto-workbench-shell__*` internals. |
| Playback, step, pause, and optional scrubber chrome | `floating-playback-control` | `/Users/yin/pto-design-system/patterns/floating-playback-control/pattern.json` | Use the exported playback API. In this workbench, the execution timeline owns step selection and replaces the standalone scrubber. Do not recreate playback buttons locally. |
| Execution trace lanes, timeline task bars, per-stage hover tooltip | `swimlane-task-bar` | `/Users/yin/pto-design-system/patterns/swimlane-task/pattern.json` | Use canvas API `window.PtoSwimlaneTaskPattern.*`; do not rebuild task bars with DOM/CSS. |
| 3D logical tensor/program-space viewport | no approved PTO pattern yet | `/Users/yin/pto/tiling/component-preview.html` required before final UI | Use Three.js/OrbitControls or equivalent real 3D renderer. This is a data-viz pattern candidate, not a private button/card style. |
| GM/UB/L1/L0 hardware route diagrams and local buffer occupancy | `memory-architecture-layout` | `/Users/yin/pto-design-system/patterns/memory-architecture/pattern.json` | Use for architecture path focus via `setPathFocus` / `clearPathFocus`; after shared cell-state API exists, use buffer grids for local data block occupancy. Do not use it as the full logical tensor tile grid. |
| AIC object shell for Cube-side visualization | `aic-core-object` | `/Users/yin/pto-design-system/patterns/aic-core-object/pattern.json` | Use for Cube/AIC role in fusion and matmul panels. |
| AIV object shell for Vector-side visualization | `aiv-core-object` | `/Users/yin/pto-design-system/patterns/aiv-core-object/pattern.json` | Use for Vector/AIV role in add and fusion panels. |
| Optional model/stage graph, when rendering semantic stage graph | `model-graphviz` | `/Users/yin/pto-design-system/patterns/model-graphviz/pattern.json` | Use for graph-like source/stage overlays when the page owns graph data. |
| Optional pass/operator node cards, when showing source/operator graph nodes | `pass-ir-graph-node` | `/Users/yin/pto-design-system/patterns/pass-ir-graph-node/pattern.json` | Use for node cards; do not create private graph node styling. |

Use existing PTO base classes and tokens for controls:

- buttons: `.btn`, `.btn-solid`, `.btn-ghost`, `.btn-icon`, size modifiers, `.is-selected`
- tabs and segmented controls: `.tab-control`, `.tab-control-item`, `.segmented-control`, `.toolbar-control`, `.toolbar-readout`
- panels and cards: `.panel-shell`, `.panel-shell-quiet`, `.workbench-pane`, `.workbench-pane-body-fill`
- inspector: `.inspector-rail`, `.inspector-section`, `.inspector-soft-card`
- badges/status: `.badge`, status color variables, tag/chip tokens

Use semantic CSS variables from:

- `/Users/yin/pto-design-system/tokens/foundation.css`
- `/Users/yin/pto-design-system/tokens/semantic.css`
- `/Users/yin/pto-design-system/tokens/components.css`

Implementation must load the local PTO runtime asset path that exists for `/Users/yin/pto/tiling`; existing modules commonly use `../vendor/pto-design-system/...`, but this must be verified before writing HTML.

### 9.1.1 Pattern Contract Check

Before consuming any matched pattern:

1. Read the matched `pattern.json`.
2. Confirm required APIs are exported by `pattern.js`.
3. Confirm the preview `pattern.html` exercises those APIs.
4. Confirm any capability claimed in UI copy is implemented by the pattern contract.
5. If preview, contract, and implementation disagree, fix the shared pattern or avoid claiming that behavior.

If the memory transfer/pipeline visual needs a new pattern, create:

`/Users/yin/pto/tiling/component-preview.html`

before shipping the real workbench. The preview must show:

- closest existing PTO pattern
- proposed new pattern
- token usage
- intended usage
- state coverage: normal, hover, active, selected, disabled where applicable
- why the current system is insufficient
- whether it belongs in shared system or is data-viz-only

Do not ship the real workbench with unapproved new visuals.

The 3D tensor viewport currently exceeds the existing PTO pattern library. Therefore Phase 1 implementation must either:

1. create `component-preview.html` for the Three.js full tensor viewport and obtain approval before placing it in the product page, or
2. keep the product page in a clearly marked prototype state until that viewport pattern is reviewed.

The approval gate applies to the viewport scene and its data-viz states. It does not allow creating new private styles for buttons, toggles, panes, inspector blocks, playback controls, or cards.

### 9.2 Main Layout

Workbench layout:

- The primary workbench uses three equal-width columns by default:
  - left source/code pane.
  - center logical tensor viewport.
  - right memory architecture pane.
- Split-pane drag handles should be narrow and low-padding. The visible divider is a thin center line, not a wide padded gutter.
- Top bar:
  - owned by `ide-frame` shell in standalone mode.
  - no Visualizer/Puzzle/Developer mode tabs in MVP.
  - arch selector: `dav-2201`, `dav-3510`
  - evidence toggles: profiling, sanitizer, dump, clock
- Left pane:
  - sample/operator selector
  - source code
  - highlighted current lines
  - semantic label badges
  - source code line styling must match `/Users/yin/pto/ascend-950-workbench-demo/index.html` left source pane: line-number gutter, mono code text, hover/selected row state, and no separate nested code card shell.
- Center pane:
  - primary 3D full tensor viewport.
  - supports drag pan plus explicit zoom buttons and fit/reset; ordinary wheel must not zoom this viewport.
  - tensor/program-space controls for `blockIdx`, `progress`, `mIndex`, `nIndex`, `kIndex`, and AIV sub-block where applicable.
  - on-chip tile lens appears as compact overlay cards on top of the viewport, limited to current local buffer blocks.
  - short axis labels remain visible in the scene; operation labels, detailed axis semantics, and flat-GM explanation are hover/inspector content, not persistent viewport badges.
  - bottom execution timeline is paired with playback controls.
  - changes by sample type:
    - vector sample: full logical vector space folded by block/progress/element + UB queue lens.
    - cube sample: A/B/C logical matrix spaces + M/N/K reduction lens.
    - fusion sample: C tile producer/consumer split + AIC/AIV sync lens.
    - lightning indexer: semantic stage graph + 3D tile preview for selected stage.
- Right pane:
  - memory architecture focus is open by default.
  - must reuse `hardware-architecture-viewport` around `memory-architecture-layout`.
  - default architecture zoom is 50 percent.
  - drag pans the diagram; Command + wheel zooms around the pointer; ordinary wheel does not zoom.
  - shows hardware routes and local buffer block occupancy only.
  - does not show the full logical tensor tile grid.
- Inspector:
  - collapsed by default.
  - opens as a fixed right-side drawer attached to the whole workbench window, not inside the center viewport pane.
  - drawer height matches the page content area below the topbar.
  - opens only after the user selects an object, such as a source line, tensor viewport, local buffer card, architecture buffer block, route, or timeline step.
  - contains current step details, variables, tile formulas, memory offsets, queue/sync state, evidence, and hover-tip explanations.
- Bottom timeline and playback:
  - provided by `floating-playback-control`
  - play/pause
  - step previous/next
  - speed
  - the execution timeline replaces the standalone playback scrubber for step selection.
  - selected block/tile/stage label.
  - playback chrome must not be rebuilt as a page-local footer bar.

### 9.3 Interaction Rules

- Selecting source line highlights the corresponding visual stage.
- Selecting a 3D tensor tile highlights corresponding source operations, formulas, and hardware path focus.
- Selecting any object opens the inspector drawer; normal step playback leaves the inspector closed unless it was already open.
- Selecting a step updates:
  - source highlight
  - 3D tensor tile / operation state
  - on-chip tile lens
  - memory architecture focus
  - inspector content only when the inspector drawer is open
  - playback label
- Changing arch updates:
  - architecture-specific branch
  - LoadData parameter panel
  - memory tier labels when applicable
- Changing tiling config in the deferred developer workflow recomputes the execution plan and marks affected stages.
- Hovering or selecting a hardware node in the architecture diagram should explain routes and memory tiers, but must not pretend to select tensor elements unless the trace has an explicit tensor mapping.

## 10. Guided Challenge Specification

Guided challenges should be built on the same trace data as the visualizer, but they are not a top-level MVP mode.

Each puzzle has:

- prompt
- context variables
- expected answer
- visual hint
- source hint
- validation
- explanation

Example puzzle object:

```json
{
  "id": "add.block-range.1",
  "sampleId": "sample.add_tpipe_tque",
  "kind": "range_prediction",
  "prompt": "For blockIdx=3 and progress=2, which xGm range is copied?",
  "context": {
    "totalLength": 16384,
    "numBlocks": 8,
    "tileNum": 8,
    "bufferNum": 2
  },
  "answer": {
    "expr": "blockIdx * blockLength + progress * tileLength",
    "range": [6400, 6528]
  },
  "sourceHint": {"path": "add.asc", "line": 60},
  "explanation": "blockLength=2048 and tileLength=128, so the base is 3*2048 and progress offset is 2*128."
}
```

Challenge categories:

- block mapping
- tile offset
- memory tier identification
- queue state prediction
- matrix tile ownership
- K-axis accumulation
- architecture branch selection
- cross-core synchronization
- PyPTO semantic stage and tile config mapping

## 11. Developer Mode Specification

Developer mode turns a real operator into an editable tiling visualization.

### 11.1 Inputs

Required:

- source file path
- operator name or entry function
- shape/config values

Optional:

- generated trace
- profiling output
- sanitizer output
- dump output
- known PTO/PyPTO metadata

### 11.2 Example Developer Session

For `lightning_indexer_prolog_quant.py`:

1. User selects the source file.
2. Tool detects `@pypto.jit`.
3. Tool detects `lightning_indexer_prolog_quant_compute`.
4. Tool detects dynamic axis `t`.
5. Tool detects `IndexerPrologQuantConfigs`.
6. Tool lists editable configs:
   - `q_linear`
   - `q_hd`
   - `k_linear`
   - `w_linear`
   - `unroll_list`
   - `l1_reuse_param`
   - `copy_in_threshold`
   - `cycle_upper_bound`
   - `block_size`
7. User selects a stage, for example `Query-Linear`.
8. Tool shows cube tile shapes and affected tensors.
9. User edits `q_linear`.
10. Tool recomputes visualization:
    - tile dimensions
    - loop count
    - memory movement estimates
    - related downstream stages
11. User attaches profiling.
12. Tool overlays slow stages and warns when tile choices likely increase copy pressure or reduce reuse.

### 11.3 Developer Mode Outputs

- `trace.json`
- `tiling-report.md`
- optional `.avz`
- optional UI snapshot

Report should include:

- semantic stage list
- current tiling config
- memory movement summary
- potential bottlenecks
- evidence mapping
- recommended next experiments

## 12. Implementation Phases

### Phase 0: Spec and Data Contract

Deliverables:

- this spec file
- `/Users/yin/pto/tiling/docs/ascend-tiling-visualization-knowledge.md`
- trace schema draft
- fixture directory plan

Acceptance:

- spec covers three MVP samples.
- spec covers real developer workflow.
- spec describes UI, trace, puzzle, evidence, 3D tensor viewport, on-chip tile lens, and memory-architecture path-focus layers.

### Phase 1: Fixture-Driven Visualizer

Deliverables:

- three trace fixtures:
  - Add
  - Matmul
  - Matmul + LeakyRelu fusion
- `component-preview.html` for the Three.js 3D full tensor viewport if it is not yet approved as a shared PTO pattern.
- static workbench using fixture data
- source pane and playback
- 3D full tensor viewport for all three samples
- on-chip tile lens for current selected tile
- memory-architecture path focus for current selected step

Acceptance:

- user can select each sample.
- user can step through every sample.
- source highlight, 3D tensor tile highlight, on-chip lens, memory architecture focus, playback, and inspector update together.
- no dependency on NPU runtime.
- vector sample clearly shows full logical vector space folded by block/progress/element.
- matmul sample clearly shows A/B/C logical tensor spaces and K-axis accumulation.
- fusion sample clearly shows AIC C tile ownership, AIV half-tile ownership, sync, and direct/avoided intermediate path.

### Phase 2: Guided Challenges

Deliverables:

- puzzle definitions for all three samples.
- validation UI, attached to selected steps or objects rather than a top-level mode.
- explanation panel.

Acceptance:

- each sample has at least five puzzles.
- puzzle answers are computed from trace/config data.
- incorrect answer shows source and visual hints.

### Phase 3: Developer Mode for Lightning Indexer

Deliverables:

- PyPTO lightweight extractor for `lightning_indexer_prolog_quant.py`.
- semantic stage graph.
- editable tiling config panel.
- regenerated trace preview.

Acceptance:

- tool detects semantic labels.
- tool detects cube/vector tile shape calls.
- tool detects loop/unroll structure.
- user can change config values and see affected stages update.

### Phase 4: Evidence Overlay

Deliverables:

- profiling evidence attachment.
- sanitizer evidence attachment.
- dump/clock placeholder support.

Acceptance:

- evidence can be loaded from JSON.
- evidence maps to stage/source/trace step.
- UI can toggle evidence layers.

### Phase 5: Packaging and Authoring Workflow

Deliverables:

- `.avz` save/load.
- import/export commands.
- docs for authoring new sample traces.

Acceptance:

- a developer can add a new operator fixture without changing UI code.
- `.avz` can be reopened and replayed.

## 13. Suggested File Layout

```text
/Users/yin/pto/tiling
├── README.md
├── docs/
│   ├── ascend-viz-puzzle-spec.md
│   └── ascend-tiling-visualization-knowledge.md
├── component-preview.html
├── data/
│   ├── fixtures/
│   │   ├── add_tpipe_tque.trace.json
│   │   ├── matmul.trace.json
│   │   └── matmul_leakyrelu_fusion.trace.json
│   └── schemas/
│       └── trace.schema.json
├── src/
│   ├── index.html
│   ├── app.js
│   ├── extractors/
│   │   ├── ascendc-light.js
│   │   └── pypto-light.js
│   ├── renderers/
│   │   ├── source-pane.js
│   │   ├── tensor-viewport-3d.js
│   │   ├── tensor-view-adapter.js
│   │   ├── on-chip-tile-lens.js
│   │   ├── architecture-focus.js
│   │   ├── memory-map.js
│   │   ├── matrix-tiles.js
│   │   ├── queue-panel.js
│   │   ├── cross-core-swimlane.js
│   │   └── inspector.js
│   └── puzzles/
│       ├── engine.js
│       └── validators.js
└── assets/
```

## 14. Risks and Mitigations

Risk: automatic source parsing becomes too broad.

Mitigation: MVP uses fixtures; extractor supports only known Ascend C/PyPTO patterns first.

Risk: UI becomes an animation without engineering value.

Mitigation: every visual step must link to source refs, formulas, tensor ranges, and evidence placeholders.

Risk: 3D tensor view misleads users into thinking GM is physically 3D.

Mitigation: every tensor view must label logical axes, physical addressing, GM offset mapping, and whether a visual axis is execution-derived, such as `blockIdx`, `progress`, or K reduction.

Risk: architecture diagram is misused as the full tensor tile grid.

Mitigation: `memory-architecture-layout` is for hardware node focus, route focus, and local on-chip buffer occupancy. Full logical tensor tile selection belongs to the 3D viewport. Local buffer occupancy is applied only through shared `setBufferBlocks` APIs and mirrored by compact viewport overlay cards.

Risk: fusion visualization becomes too complex.

Mitigation: start with one AIC and two AIV lanes for selected tile only; do not attempt whole-chip scheduling first.

Risk: PyPTO generated code differs from high-level semantic labels.

Mitigation: the deferred developer workflow separates high-level semantic plan from runtime evidence. Evidence overlay can later map generated kernels back to labels when metadata exists.

Risk: new visual style leaks into PTO module.

Mitigation: use existing PTO patterns; create component preview and approval gate for new pipeline/transfer visuals.

## 15. Acceptance Criteria for MVP

MVP is done when:

- Three samples are present:
  - Vector Add
  - Cube Matmul
  - Fusion Matmul + LeakyRelu
- Each sample has:
  - source pane
  - trace playback
  - 3D full tensor viewport
  - on-chip tile lens
  - memory-architecture path focus
  - inspector
  - at least five puzzle tasks
- Visualizer mode can explain:
  - block mapping
  - tile size and offsets
  - memory movement
  - queue events for Add
  - Cube memory hierarchy for Matmul
  - cross-core sync for Fusion
  - why the 3D tensor is a logical access space over flat GM, not physical memory shape
  - which hardware route is active for the selected tile
- Developer Mode can ingest `lightning_indexer_prolog_quant.py` at lightweight level and show:
  - semantic stages
  - cube/vector tile shape settings
  - loop/unroll structure
  - editable config values
- PTO design system is reused according to `/Users/yin/pto-design-system/SKILL.md`, not by ad hoc visual imitation.
- Matched pattern contracts are read before implementation: `ide-frame`, `workbench-shell`, `floating-playback-control`, `swimlane-task-bar`, `memory-architecture-layout`, `aic-core-object`, and `aiv-core-object` at minimum.
- The 3D tensor viewport has a component preview or an approved shared PTO pattern contract before final UI use.
- Local data block occupancy in AIC/AIV/memory buffer grids is implemented only through approved shared cell-state APIs, or remains in the on-chip tile lens preview until those APIs exist.
- Any missing data-viz pattern has a preview page before entering final UI.

## 16. Open Questions

- Should the artifact extension be `.avz`, or should it remain `trace.json` until save/load is mature?
- Should Developer Mode edit source files directly, or produce a suggested config patch only?
- Which profiling output format should be normalized first?
- Should `lightning_indexer_prolog_quant.py` be treated as PyPTO source only, or should the tool also consume generated Ascend C when available?
- Should guided challenges stay as step-level inspector prompts, or later become a separate route?

## 17. Immediate Next Steps

1. Done: create `data/schemas/trace.schema.json`.
2. Done: create the three MVP fixtures by hand.
3. Done: add shared buffer block APIs for `PtoAicCorePattern.setBufferBlocks`, `PtoAivCorePattern.setBufferBlocks`, `PtoMemoryArchitecturePattern.setBufferBlocks`, and `PtoMemoryArchitecturePattern.clearBufferBlocks`.
4. Done: extend fixtures with `visualState.tensorViewport`, `visualState.onChipLens`, and `visualState.architectureFocus.bufferBlocks`.
5. Done for MVP: build a fixture-driven static workbench with tensor-first visualization, on-chip tile lens, memory architecture route focus, and buffer block occupancy.
6. Still open: build `component-preview.html` or a shared PTO pattern contract for the full 3D tensor viewport and transfer/tile-lens visual states before promoting the local MVP renderer to a reusable pattern.
7. Done for MVP UI: remove top-level Visualizer/Puzzle/Developer tabs, make Memory Architecture the right pane, collapse Inspector by default, move On-Chip Tile Lens into the viewport, and combine execution timeline with playback controls.
8. Still open: add guided challenge interactions on top of selected trace steps or objects.
9. Still open: add PyPTO lightweight extraction for `lightning_indexer_prolog_quant.py`.
