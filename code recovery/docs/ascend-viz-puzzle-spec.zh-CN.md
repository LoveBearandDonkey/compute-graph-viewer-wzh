# Ascend Viz / Puzzle Tiling Workbench 规格说明

状态：草稿  
Owner workspace：`/Users/yin/pto/tiling`  
参考仓库：`/Users/yin/gitcode/asc-devkit-master`  
真实开发者场景：`/Users/yin/gitcode/deepseekv3.2源码/lightning_indexer_prolog_quant.py`

## 1. 目标

构建一个面向 Ascend C / PyPTO 的 `triton-viz` 与 `Triton-Puzzles` 体验版本。

这个工具不能只是静态解释器。它应该让算子开发者在编写或调优真实 Ascend 算子时，能够看到、逐步执行并推理 tiling、内存搬运、block 映射、Vector/Cube 执行、融合同步以及 profiling 证据。

MVP 包含三种模式：

1. **Visualizer mode**
   - 展示源码、trace step、tensor region、内存层级、queue 状态、sync 事件，以及 profiling/sanitizer 证据。
   - 体验上类似 `triton-viz` 的 kernel visualizer / profiler / sanitizer 流程，但语义改为 Ascend C/PyPTO。

2. **Puzzle mode**
   - 交互式练习：用户在看到答案前先预测 block range、tile region、queue transition、memory movement 或 cross-core handoff。
   - 类似 Triton-Puzzles，但概念换成 `GetBlockIdx`、`TPipe/TQue`、`DataCopy`、`LoadData`、`Mmad`、`Fixpipe`、`CrossCoreSetFlag`、PyPTO tile shape，以及真实 Ascend memory tier。

3. **Developer mode**
   - 把真实 Ascend C 或 PyPTO 源码转换为可编辑的 tiling 可视化。
   - 允许开发者修改 tiling/config 值，然后看到受影响的逻辑 tensor tile、片上驻留状态、架构路径 focus、源码解释和 inspector 说明同步更新。

## 2. 非目标

- 不直接运行 Triton kernel。
- 不把 `triton-viz` runtime 复用为引擎。这里复用的是交互模式和心智模型，不是执行语义。
- 不在 `/Users/yin/pto/tiling` 内构建私有视觉风格系统。
- MVP 不要求自动解析所有 Ascend C 方言特性。
- 第一个 UI milestone 不要求 NPU 硬件。runtime 证据可以后续从 profiling/sanitizer 输出接入。
- 不尝试完整模拟 CANN compiler。工具应该展示算子开发者的 tiling 与执行计划，并在有真实证据时叠加证据。

## 3. 开源 / 官方参考

### 3.1 UX 参考

- `triton-viz`
  - 参考作用：可视化 tile-based kernel programming、memory load/store、matmul、trace save/load、profiler、sanitizer。
  - 适配方式：`tl.load`、`tl.store`、`tl.dot` 映射为 Ascend 操作，例如 `DataCopy`、`LoadData`、`Mmad`、`Fixpipe`，以及 PyPTO `matmul`。

- `Triton-Puzzles`
  - 参考作用：用渐进式交互 puzzle 教 tile/block programming，且初始体验不依赖生产硬件。
  - 适配方式：puzzle 任务教授 Ascend C/PyPTO tiling 概念，而不是 Triton program ID 和 pointer expression。

### 3.2 Ascend 官方参考

主来源：`/Users/yin/gitcode/asc-devkit-master`

关键官方概念：

- API 层级：
  - `TPipe/TQue` framework API
  - 基础 C++ tensor API
  - SIMD/SIMT C API
  - high-level API
  - 算子模板
- 执行与内存概念：
  - `GlobalTensor`
  - `LocalTensor`
  - GM、UB、L1、A1、B1、A2、B2、CO1
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
  - `__vector__`、`__cube__`、`__mix__(1, 2)`
- 工具证据：
  - printf
  - assert
  - dump
  - CPU debug
  - profiling
  - sanitizer
  - clock
  - simulator

### 3.3 UI 参考包

PTO 标准 UI 参考：

`/Users/yin/pto-design-system/SKILL.md`

本规格以 PTO design-system skill 作为 UI 治理来源。tiling workbench 是布局密集、可视化密集的 PTO 页面，因此实现必须使用 **Workflow C: Pattern-first PTO page or block**。

实现前必须按以下顺序读取 UI baseline：

1. `/Users/yin/pto-design-system/references/DESIGN.md`
   - theme、surface、typography、spacing、component 与 governance 的标准系统规格。
2. `/Users/yin/pto-design-system/references/quick-reference.md`
   - token 与 class 的实现速查表。
3. `/Users/yin/pto-design-system/references/retrofit-container-audit.md`
   - 迁移或借用已有 demo layout 时必须读取。
4. `/Users/yin/pto-design-system/patterns/patterns.json`
   - shared pattern registry。
5. `/Users/yin/pto-design-system/patterns/<pattern-id>/pattern.json` 下匹配的 `pattern.json`
   - 每个 pattern 的标准复用契约，包括 required API、allowed override 与 forbidden override。
6. `/Users/yin/pto-design-system/tokens/foundation.css`
7. `/Users/yin/pto-design-system/tokens/semantic.css`
8. `/Users/yin/pto-design-system/tokens/components.css`
9. `/Users/yin/pto-design-system/css/style.css`
10. `/Users/yin/pto-design-system/design-system-preview.html`
    - 只作为基础组件 helper。它不是完整 IDE、graph、timeline、architecture 或 playback 页面的标准来源。

`/Users/yin/pto/tiling` 下的 runtime 页面应消费本地 PTO workspace 使用的已签入模块资产路径。现有 PTO 页面通常加载 `../vendor/pto-design-system/...`；实现 HTML 前必须确认具体 link path。

UI 规则：

不要在本模块内发明新的 button、toggle、badge、card、panel、spacing scale、color language、IDE shell、split-pane kernel、swimlane task bar、memory architecture block、AIC/AIV object 或 playback chrome。优先使用 shared pattern contract。如果所需 transfer/pipeline visual 未被覆盖，先创建 `/Users/yin/pto/tiling/component-preview.html`，等待批准后再用于真实 workbench。

### 3.4 教学内容参考

首个产品页面的概念教学内容应来自：

`/Users/yin/pto/tiling/docs/ascend-tiling-visualization-knowledge.md`

该文档解释了本规格依赖的核心心智模型：

- GM 是扁平地址空间，不是物理 3D block。
- Tensor shape、layout 与 stride 在扁平 GM buffer 上创建逻辑 tensor view。
- Tiling 把逻辑 tensor view 映射到 `blockIdx`、loop progress、tile range 与 memory offset。
- CopyIn / Compute / CopyOut 描述一个逻辑 tile 穿过 GM 与片上 buffer 的生命周期。
- 3D viewport 回答：“这个 step 正在触碰逻辑 tensor 的哪一部分？”
- memory architecture diagram 回答：“这个 tile 会经过哪些硬件对象和路线？”

## 4. 核心可视化模型

Workbench 必须是 **3D tensor-first**，而不是 timeline-first。

参考 UX 是 Triton Viz：中心是大型 3D tensor / program-space viewport，旁边有控制与源码联动。Ascend 版本必须保留这个核心效果，同时把语义从 Triton `tl.load` / `tl.store` / `tl.dot` 改为 Ascend C / PyPTO 的 tiling、memory 与 compute 概念。

### 4.1 完整 Tensor Space 是逻辑空间，不是物理空间

3D viewport 绝不能暗示 GM 物理上存储了一个 3D tensor。

必要解释：

- GM 是扁平内存。
- `GlobalTensor`、tensor shape、layout、stride 与 offset 定义了 GM 上的逻辑 tensor view。
- 3D viewport 渲染的是这个逻辑访问空间。
- 高亮 tile 表示当前逻辑 range，并能映射回扁平 GM offset。

对于 1D vector sample，viewport 可以把执行结构折叠成：

```text
X axis: element inside tile
Y axis: tileIdx / progress
Z axis: blockIdx / core
```

对于 matmul，viewport 可以使用：

```text
X axis: N tile
Y axis: M tile
Z axis: K tile / reduction step
```

matmul 里的 `Z` axis 不是 output tensor depth，而是 K-axis accumulation 的可视化。

### 4.2 必需视觉层

每个 trace step 必须驱动四个同步视觉层：

1. **3D full tensor viewport**
   - 中心主可视化。
   - 展示完整逻辑 tensor 或 tensor set。
   - 高亮当前 block/tile/slice。
   - 展示 load、store、mask、padding、reduction 与 output commit 状态。

2. **On-chip tile lens**
   - 当前选中 tile 的小型辅助视图。
   - 展示该 local tile 在 UB、L1、L0A、L0B、L0C、CO1 或 Vector/Cube staging object 中的驻留状态。
   - 不替代 full tensor view。

3. **Memory architecture path focus**
   - 使用 `memory-architecture-layout`。
   - 展示穿过 GM/L2/UB/L1/L0A/L0B/L0C/Cube/Vector/C-V lanes 的硬件 route focus。
   - 高亮活跃硬件节点与路线。
   - 当 trace 提供 local buffer-block state 时，可以展示当前 data block 在 AIC/AIV buffer grid 内的驻留/占用。
   - 不替代完整逻辑 tensor tile grid。

4. **Source / inspector / playback**
   - source line、formula、offset、queue event、sync event 与 evidence 随同一个 step 更新。

### 4.3 3D Tensor Viewport 要求

3D tensor viewport 必须实现为真实 3D scene，优先使用 Three.js 与 OrbitControls。不要用 flat card、静态 SVG 或 timeline bar 近似主 Triton Viz 效果。

功能要求：

- 能围绕 tensor block pan/orbit/zoom。
- 渲染完整 tensor bounds。
- 渲染 grid/cell/voxel 或 instanced block 表示。
- 高亮选中的 tile range。
- 在需要时支持多个 tensor surface，例如 `x/y/z`、`A/B/C` 或 `input/output/workspace`。
- 支持 viewport 上方或内部的 operation chip，例如 `CopyIn`、`DataCopy`、`LoadData`、`Mmad`、`Fixpipe`、`LeakyRelu`、`CopyOut`。
- 支持类似 Triton Viz Program IDs 的 selected program/core/tile 控件，但使用 Ascend 语义命名：
  - `blockIdx`
  - `progress`
  - `mIndex`
  - `nIndex`
  - `kIndex`
  - 相关时包含 AIV sub-block
- 暴露 no-WebGL fallback：仍然显示 2D tensor/tile map，并给出明确警告。

实现说明：

Triton Viz front-end 代码是 MIT 许可，适合作参考，但不能直接整套放入。它的 `OpRecord`、program IDs、load/store descriptor、pointer offset、mask 与 backend endpoint 都是 Triton-specific。Ascend Viz 应复用视觉模式和选定的 rendering idea，而不是 runtime data contract。

如果复制或改写 Triton Viz 代码：

- 保留 license 与 attribution。
- 尽可能本地 vendor 依赖。
- 如果需要离线预览，用项目管理资产替换 CDN import。
- 用 PTO shell/chrome 与 PTO controls 包住 viewport。
- 非 scene UI 必须 token-derived。

### 4.4 Memory Architecture Path Focus 要求

`memory-architecture-layout` pattern 是解释硬件路径的必需层。

用它回答：

```text
这个 tile 正在经过哪些硬件对象？
哪条 route 当前处于 active？
哪些硬件对象与当前 step 无关？
fusion 避免了哪条 intermediate route？
哪个 on-chip buffer 当前承载这个 data block？
如果有 buffer cell state，选中 data block 占用了哪些 local buffer segment/cell？
```

不要用它回答：

```text
完整逻辑 tensor 的哪些 cell 被触碰？
选中了哪个 M/N/K tile？
哪些 output element 已经 committed？
```

这些问题属于 3D tensor viewport。

Buffer grid 的作用：

- 现有 AIC/AIV/memory architecture pattern 已经渲染类似容量的 buffer grid。
- 这些 grid 适合表示 **local on-chip data block residency**，例如 UB 里的 `xLocal`、L0A 里的 `A[mTile,kTile]`、L0B 里的 `B[kTile,nTile]`，或 L0C/CO1 里的 partial `C[mTile,nTile]`。
- 这些 grid 不适合渲染完整逻辑 tensor space，例如完整 `x[0:N]`、完整 `A[M,K]` 或完整 `C[M,N]`。
- 产品页面不能通过深入 generated DOM 硬编码 grid cell color。Dynamic cell state 必须先通过 shared pattern API 添加。

MVP 必须支持的 path focus 映射：

| Step type | Architecture focus |
|---|---|
| Vector CopyIn | L2/GM rail -> AIV UB route，加上 source input tensor node metadata |
| Vector Compute | AIV UB + Vector execution object |
| Vector CopyOut | AIV UB -> L2/GM route |
| Cube CopyIn/LoadData | L2/GM rail -> AIC L1 -> L0A/L0B |
| Cube Mmad | L0A + L0B + Cube + L0C |
| Cube Fixpipe/CopyOut | L0C/CO1 -> GM C route 或 output tier |
| Fusion sync | AIC result producer + AIV consumer lanes + CrossCore flag state |
| Fusion direct C-V | L0C -> UB / UB -> L1 route，并在适用时标记 GM workspace avoided |

现有 `memory-architecture-layout` 已经支持 `setPathFocus(container, preset, { selectors, routes, errorSelectors })`。产品页面必须调用该 API，不要在本地重新实现 route geometry 或 focus class。

### 4.5 待补 Shared Pattern API：Buffer Grid Cell State

当前 `aic-core-object`、`aiv-core-object` 与 `memory-architecture-layout` pattern 会渲染 buffer grid，但它们的 public contract 只覆盖静态 grid geometry 和 highlighted band columns。它们尚未暴露稳定 API 来表达动态 data block occupancy。

在最终产品集成前，将以下内容加入 design-system TODO：

```js
window.PtoAicCorePattern.setBufferBlocks(root, blocks)
window.PtoAivCorePattern.setBufferBlocks(root, blocks)
window.PtoMemoryArchitecturePattern.setBufferBlocks(root, blocks)
window.PtoMemoryArchitecturePattern.clearBufferBlocks(root)
```

建议的 block payload：

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

必要行为：

- cell state 必须由 shared pattern 应用，而不是通过 page-local CSS selector 实现。
- cell 应支持 `loaded`、`allocated`、`enqueued`、`dequeued`、`accumulating`、`committed`、`avoided` 等状态。
- 每个 block 应暴露 tooltip metadata，能链接回 source tile、GM range、queue 或 operation。
- buffer block 色彩必须来自 token 或 pattern 定义。
- 如果该 API 尚未准备好，workbench 只能在单独的 on-chip tile lens / component preview 中展示 buffer block occupancy，不能 mutation shared pattern internals。

### 4.6 Step State Contract

每个可视化 step 必须携带足够状态，以更新全部视觉层。

最小 step state：

- source refs。
- 当前 block/core identifier。
- loop identifier。
- logical tensor view identifier。
- 选中的 logical tile/range。
- flat GM offset expression，以及已知时的 resolved range。
- on-chip residency state。
- memory architecture focus payload。
- queue/sync state。
- 可选 evidence pointer。

同一个 step 应能回答：

```text
这个 tile 在完整逻辑 tensor 中的位置在哪里？
它映射到哪段 flat GM range？
哪个 on-chip buffer 当前持有它？
哪条 hardware route 当前 active？
哪些 source lines 创建了这个状态？
```

## 5. MVP Samples

MVP 必须首先包含三个官方 sample trace。这些 trace 应在自动 parser 工作开始前作为 deterministic fixtures。

### 5.1 Sample A：Vector Add with TPipe/TQue

Source：

`/Users/yin/gitcode/asc-devkit-master/examples/01_simd_cpp_api/00_introduction/01_vector/add_tpipe_tque/add.asc`

目的：

- 教授基础 vector tiling 和 queue-driven memory flow。
- 展示最简单完整的 shape -> block -> tile -> copy -> compute -> copyout loop。

关键源码概念：

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
  - `CopyIn(i)`、`Compute(i)`、`CopyOut(i)` loop
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

必要可视化：

- 3D full tensor viewport：
  - 主视图展示完整逻辑 vector space，而不是物理 GM shape。
  - 为教学与导航，把 1D GM buffer 折叠为 `[blockIdx, progress, element]`。
  - 将 `x`、`y`、`z` 渲染为相关 tensor surface 或可选 tensor layer。
  - 选中的 `blockIdx` 与 `progress` 高亮 active tile range。
  - CopyIn 高亮 source `x/y` GM slice。
  - Compute 保持选中逻辑 tile outline active，并显示本地 `xLocal + yLocal -> zLocal`。
  - CopyOut 填充或 commit 对应 `z` slice。
- On-chip tile lens：
  - 显示 `xLocal`、`yLocal`、`zLocal` tile slot。
  - 显示 `BUFFER_NUM` queue slot，用于 double buffer context。
  - local tile lens 必须小而解释性强；不能替代 full tensor view。
- Memory architecture path focus：
  - CopyIn 调用 `memory-architecture-layout` focus，路径为 L2/GM -> AIV UB。
  - Compute focus AIV UB + Vector。
  - CopyOut focus AIV UB -> L2/GM。
  - 该层只解释 hardware route；tensor tile coloration 仍留在 3D viewport。
- Host launch strip：
  - input files
  - H2D copy
  - `add_custom<<<numBlocks, nullptr, stream>>>`
  - stream sync
  - D2H copy
- Block map：
  - 连续 vector 上的 8 个 block。
  - 选中 block 高亮 GM range。
- Tile timeline：
  - 每个 progress index 映射为 `progress * tileLength`。
  - 选中 step 展示 `xGm`、`yGm`、`zGm` read/write region。
- Queue panel：
  - `inQueueX`、`inQueueY`、`outQueueZ`。
  - queue slot 显示 `empty`、`allocated`、`enqueued`、`dequeued`、`freed`。
- Memory panel：
  - GM 和 UB/VECIN/VECOUT region。
- Puzzle tasks：
  - 给定 `totalLength`、`numBlocks`、`tileNum`、`BUFFER_NUM`，计算 `blockLength` 和 `tileLength`。
  - 给定 `blockIdx` 和 `progress`，识别 x/y read range 与 z write range。
  - 预测 `CopyIn`、`Compute`、`CopyOut` 后的 queue state。

### 5.2 Sample B：Cube Matmul

Source：

`/Users/yin/gitcode/asc-devkit-master/examples/01_simd_cpp_api/00_introduction/02_matrix/matmul/matmul.asc`

目的：

- 教授 Cube path 和 matrix tile movement。
- 展示核心 `GM -> L1/A1/B1 -> L0/A2/B2 -> CO1 -> GM` pipeline。

关键源码概念：

- template parameters：
  - `M`、`K`、`N`
  - `singleCoreM`、`singleCoreK`、`singleCoreN`
  - `baseM`、`baseK`、`baseN`
- `InitGMOffsets`
  - `mIter = M / singleCoreM`
  - `mIterIdx = GetBlockIdx() % mIter`
  - `nIterIdx = GetBlockIdx() / mIter`
  - `gmOffsetA`、`gmOffsetB`、`gmOffsetC`
- local tensors：
  - `A1`、`A2`、`B1`、`B2`、`CO1`
- nested loops：
  - `mIndex`
  - `nIndex`
  - `kIndex`
- movement and compute：
  - `CopyInA`：GM A ND to A1 Nz。
  - `CopyInB`：GM B ND to B1 Nz。
  - `DataLoadA`：A1 to A2。
  - `DataLoadB`：B1 to B2，包含 transpose semantics。
  - `Mmad`：A2/B2 to CO1。
  - `Fixpipe`：CO1 to GM C，format 与 dtype conversion。
- architecture branches：
  - `__NPU_ARCH__ == 2201`
  - `__NPU_ARCH__ == 3510`

必要可视化：

- 3D full tensor viewport：
  - 主视图展示逻辑 `A[M,K]`、`B[K,N]` 与 `C[M,N]` tensor space。
  - 选中 `blockIdx` 高亮由 `mIterIdx` 和 `nIterIdx` 决定的 C output tile。
  - `kIndex` 高亮当前 A/B reduction tile。
  - Z/depth 维度可表示 K-axis accumulation，不是 output tensor physical depth。
  - 第一个 `kIndex` step 视觉标记 C/L0C initialization；后续 `kIndex` step 视觉标记 accumulation。
- On-chip tile lens：
  - 显示 L1/A1 和 L0A/A2 中的当前 A tile。
  - 显示 L1/B1 和 L0B/B2 中的当前 B tile。
  - 显示 CO1/L0C 中的 C partial result。
  - 区分 load、compute、accumulate 与 fixpipe state。
- Memory architecture path focus：
  - CopyIn / LoadData focus L2/GM -> AIC L1 -> L0A/L0B。
  - Mmad focus L0A + L0B + Cube + L0C。
  - Fixpipe / output focus L0C/CO1 -> output GM C route。
- Matrix block map：
  - C matrix 按 `singleCoreM` 和 `singleCoreN` 划分。
  - 选中 `blockIdx` 映射到 C tile。
- K accumulation timeline：
  - `kIndex` 展示 A/B tile 的重复搬运和向 CO1 accumulation。
  - 第一个 `kIndex` 有 `cmatrixInitVal = true`；后续 step 执行 accumulate。
- Memory tiers：
  - GM A/B/C。
  - L1 中的 A1/B1。
  - L0A/L0B 中的 A2/B2。
  - CO1/L0C。
- Data movement arrows：
  - `DataCopy`
  - `LoadData`
  - `Mmad`
  - `Fixpipe`
- Architecture toggle：
  - `dav-2201`
  - `dav-3510`
  - 展示不同 `LoadData2DParams` / `LoadData2DParamsV2` parameter panel。
- Puzzle tasks：
  - 给定 `blockIdx`，识别 `mIterIdx`、`nIterIdx` 与 C tile。
  - 给定 `mIndex/nIndex/kIndex`，识别 GM offsets。
  - 预测每个 step 中 A/B/C 位于哪个 memory tier。
  - 解释为什么 B 在 `LoadDataB` 中使用 transpose。

### 5.3 Sample C：Fusion Matmul + LeakyRelu

Source：

`/Users/yin/gitcode/asc-devkit-master/examples/01_simd_cpp_api/00_introduction/03_fusion_operation/matmul_leakyrelu_basic_api/matmul_leakyrelu_basic_api.asc`

目的：

- 教授 fusion 不只是“两个 op 串起来”。
- 展示 Cube/Vector 协作、cross-core synchronization，以及同一个 output tile 的 split ownership。

为什么该 sample 是必需的：

- 它比 single-core fused API sample 更好覆盖 MVP 的 fusion requirement。
- 同时包含 Cube 和 Vector role。
- 显式包含 `__mix__(1, 2)` 和 `CrossCoreSetFlag` / `CrossCoreWaitFlag`。
- 展示一个 Cube result 被两个 Vector core 消费。

关键源码概念：

- `__global__ __mix__(1, 2) void mmad_vec_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c)`
  - Kernel 使用 Cube:Vector = 1:2。
- `ASCEND_IS_AIC`
  - Cube path 计算 Matmul。
- `ASCEND_IS_AIV`
  - Vector path 等待，然后应用 LeakyRelu。
- `InitGMOffsetsMatrix`
  - Cube block 映射到 C tile。
- `InitGMOffsetsVector`
  - Vector block 映射到 Cube result 的一半。
  - `GetBlockIdx() / 2` 将 AIV pair 映射到 AIC tile。
  - `GetBlockIdx() % 2` 选择 top 或 bottom half。
- Cube flow：
  - `CopyInA`
  - `CopyInB`
  - `DataLoadA`
  - `DataLoadB`
  - `Mmad`
  - `Fixpipe`
  - `CrossCoreSetFlag`
- Vector flow：
  - `CrossCoreWaitFlag`
  - `DataCopyPad GM -> VECCALC`
  - `LeakyRelu`
  - `DataCopyPad VECCALC -> GM`

必要可视化：

- 3D full tensor viewport：
  - 主视图展示 `A`、`B` 和最终 `C` / output 的逻辑 tensor space。
  - 选中 AIC block 高亮 Cube 产生的完整 C tile。
  - 选中 AIV block 根据 `GetBlockIdx() % 2` 高亮该 C tile 的 top 或 bottom half。
  - 如果 sample path 使用 GM 作为 intermediate，明确展示该 GM intermediate。
  - 如果正在教学 direct C-V path，则将 GM workspace 标为 avoided，而不是假装路径不存在。
- On-chip tile lens：
  - 显示 AIC 侧 L0C/CO1 producer tile。
  - 显示 AIV 侧 LeakyRelu 的 UB tile。
  - 显示两个 AIV consumer 的 split ownership。
- Memory architecture path focus：
  - AIC Matmul focus L2/GM -> AIC L1/L0A/L0B -> Cube -> L0C。
  - CrossCoreSetFlag / CrossCoreWaitFlag focus producer/consumer lane synchronization。
  - Vector epilogue focus AIV UB + Vector + output route。
  - Direct C-V teaching focus 通过 `memory-architecture-layout` 的 L0C -> UB / UB -> L1 routes。
- Cross-core swimlane：
  - 一条 AIC lane。
  - 两条 AIV lane。
  - AIC 产生 `baseM * baseN`。
  - AIV0 处理前半 `baseM / 2 * baseN`。
  - AIV1 处理后半 `baseM / 2 * baseN`。
- Sync event overlay：
  - AIC lane 上的 `CrossCoreSetFlag`。
  - `CrossCoreWaitFlag` gate 两条 AIV lane。
- C tile ownership：
  - C tile 横向切成两个 vector subtile。
  - 选中 AIV block 展示 `GetBlockIdx() % 2` region。
- Memory flow：
  - AIC：GM/L1/L0/CO1/GM。
  - AIV：GM/UB/GM。
- Puzzle tasks：
  - 给定 Cube `blockIdx`，识别成对的 Vector `blockIdx` 值。
  - 给定 Vector `blockIdx`，识别它拥有 C tile 的哪一半。
  - 预测 AIV 什么时候可以开始。
  - 识别为什么 direct fusion 仍需要 synchronization。

## 6. 真实开发者场景：Lightning Indexer Tiling

MVP 必须展示同一个工具如何应用到真实代码，使用本地 PyPTO 算子：

`/Users/yin/gitcode/deepseekv3.2源码/lightning_indexer_prolog_quant.py`

用户可能称它为 “lighting indexer”；源文件名使用 `lightning_indexer_prolog_quant`。

### 6.1 为什么这个场景重要

该文件已经包含适合 visual tiling 的 hook：

- semantic labels：
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
- cube tile controls：
  - `pypto.set_cube_tile_shapes(...)`
  - `q_linear`
  - `q_hd`
  - `k_linear`
  - `w_linear`
- vector tile controls：
  - `pypto.set_vec_tile_shapes(...)`
- dynamic and loop controls：
  - `pypto.mark_dynamic(...)`
  - `pypto.loop(...)`
  - `pypto.loop_unroll(...)`
  - `unroll_list`
- pass options：
  - `nbuffer_merge_mode`
  - `l1_reuse_map`
  - `copyin_threshold`
  - `cycle_upper_bound`
- memory/scatter semantics：
  - `pypto.view`
  - `pypto.reshape`
  - `pypto.matmul`
  - `pypto.assemble`
  - `pypto.scatter_update`

### 6.2 Developer Workflow

开发者编写或调优 `lightning_indexer_prolog_quant` 时的目标 workflow：

1. 开发者在 workbench 中打开 operator。
2. 工具从 `pypto.set_semantic_label` 提取 semantic stage。
3. 工具从 `set_cube_tile_shapes` 和 `set_vec_tile_shapes` 提取 tile shape。
4. 工具从 `loop`、`loop_unroll`、`tIdx` 和 `unrollLength` 提取 loop structure。
5. 工具构建 execution plan：
   - dynamic axis：`t`
   - unrolled slices：`tIdx`、`t_tile`
   - Query path
   - Key path
   - Weight path
   - output assembly/scatter
6. UI 渲染该 plan。
7. 开发者修改 configs：
   - `q_linear`
   - `q_hd`
   - `k_linear`
   - `w_linear`
   - `unroll_list`
   - `l1_reuse_param`
   - `copy_in_threshold`
   - `cycle_upper_bound`
8. 工具重新渲染受影响 stage，不要求完整 compiler simulation。
9. 如果存在 profiling/sanitizer evidence，工具进行 overlay：
   - slow semantic stage
   - memory hot spot
   - exceeded copy threshold
   - invalid/scatter risk
   - unbalanced vector/cube stage
10. 开发者决定是否修改 tiling、拆分 stage、增加/减少 unroll，或调整 reuse。

### 6.3 Lightning Indexer Stage View

Workbench 应展示三列 stage 或三个 tab：

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

每个 stage 的 inspector 显示：

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

### 6.4 Lightning Indexer Puzzle Tasks

Puzzle mode 应包含真实代码任务：

- 给定 `t`、`unroll_list` 和选中的 `tIdx`，识别 `t_tile`。
- 给定 `q_linear`，识别 `Query-Linear` 使用的 cube tile dimensions。
- 给定 `rope_head_dim`，识别 `q_rope` 与 `q_nope` view。
- 给定 `k_cache_index`，预测 `k_cache` 的 scatter destination。
- 给定 `l1_reuse_map`，解释哪些 stage 可以复用 L1，哪些不能。
- 给定 `copyin_threshold`，识别可能触发 copy-in pressure 的 stage。

## 7. Trace Model

核心 artifact 是 Ascend visualization trace。工作名：`.avz`。

MVP 可以使用普通 JSON 文件。后续 `.avz` 可以成为 zip container。

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
  "explanation": "将 x/y tile 从 GM 复制到本地 VECIN queue buffer。"
}
```

### 7.5 Tensor View Object

`tensorViews` 描述 flat memory 与 logical tensor dimensions 如何在 3D viewport 中渲染。

Vector Add example：

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

Matmul example：

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

规则：

- `physicalAddressing` 解释逻辑 view 如何映射到 flat 或 laid-out GM。
- `logicalShape` 可以是 visualization shape，只要 `logicalAxes` 解释每个 axis。
- 1D vector buffer 可以渲染为 3D `[blockIdx, progress, element]`。
- 2D output matrix 只有在表示 accumulation 时才可以使用 K-depth visualization；必须标注为 reduction progress。
- 3D viewport 应使用该 object，不要只根据 label 推断 shape。

### 7.6 Architecture Focus Object

`visualState.architectureFocus` 是 `memory-architecture-layout` 的直接输入。

Example：

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

规则：

- 使用 `memory-architecture-layout` 支持的 selector 与 route id。
- `bufferBlocks` 只用于 local on-chip buffer occupancy，不用于完整逻辑 tensor grid。
- 不要在该 object 中编码完整 tensor tile range。
- 对于 avoided intermediate，使用 inspector metadata 或 diagnostic selectors；不要把 inactive route 伪装成 active。

### 7.7 On-Chip Lens Object

`visualState.onChipLens` 描述当前 local tile state。

Example：

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

规则：

- lens 应展示当前 tile residency 与 operation state。
- 它不应绘制另一个 full tensor。
- 必须与选中的 3D tensor tile 和 architecture focus 保持同步。

### 7.8 Tiling Object

For Ascend C：

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

For PyPTO：

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

MVP 应使用三层 extraction。

### 8.1 Level 0：Handwritten Fixture

用于三个官方 sample。

优点：

- 快。
- 稳定。
- 允许 UI 与 schema 在 parser 工作开始前先收敛。

交付：

- `data/fixtures/add_tpipe_tque.trace.json`
- `data/fixtures/matmul.trace.json`
- `data/fixtures/matmul_leakyrelu_fusion.trace.json`

### 8.2 Level 1：Lightweight Source Extractor

用于已知 Ascend C 与 PyPTO pattern。

Ascend C extraction：

- class name
- global kernel declaration
- `__vector__`、`__cube__`、`__mix__`
- `Init`
- `Process`
- 名为 `CopyIn`、`Compute`、`CopyOut`、`DataLoad`、`LeaklyGeluCompute` 的 helper functions
- `GlobalTensor` 与 `LocalTensor`
- `DataCopy`、`DataCopyPad`、`LoadData`、`Mmad`、`Fixpipe`
- `SetFlag`、`WaitFlag`、`CrossCoreSetFlag`、`CrossCoreWaitFlag`

PyPTO extraction：

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
- dataclass 中的 config fields

### 8.3 Level 2：Runtime Evidence Attachment

在开发者运行 profiling/sanitizer/dump 后使用。

Inputs：

- profiling result
- sanitizer result
- dump result
- clock result
- generated code metadata if available

Visualizer 应通过以下字段将 evidence 关联到 stages：

- kernel name
- source ref
- semantic label
- op name
- tensor name
- org index 或 profiler row id（如可用）

## 9. UI Specification

UI 必须优先消费 PTO design system assets，并以 `/Users/yin/pto-design-system/SKILL.md` 作为 UI 治理参考。该 workbench 是 IDE/workbench visualization surface，因此使用 design-system **Workflow C: Pattern-first PTO page or block**。

不要把 `design-system-preview.html` 当成 page-level source of truth。它只是基础组件 helper。完整页面行为必须来自 `patterns/patterns.json` 和每个匹配的 `patterns/<pattern-id>/pattern.json`。

### 9.1 Required PTO Patterns

实现 workbench 前读取这些 pattern contract：

| Workbench need | PTO pattern id | Contract path | Usage requirement |
|---|---|---|---|
| IDE-like page shell、activity rail、pane、pane header、inspector dock | `ide-frame` | `/Users/yin/pto-design-system/patterns/ide-frame/pattern.json` | 作为上层 page shell 使用。不要用 generic div 重建 IDE chrome。 |
| 可拖拽 split pane | `workbench-shell` | `/Users/yin/pto-design-system/patterns/workbench-shell/pattern.json` | 只作为 resize kernel 使用。不要 override `.pto-workbench-shell__*` internals。 |
| Playback、step、pause、scrubber、collapsed chrome | `floating-playback-control` | `/Users/yin/pto-design-system/patterns/floating-playback-control/pattern.json` | 使用导出的 playback API。不要本地重建 playback footer 或 scrubber。 |
| Execution trace lane、timeline task bar、per-stage hover tooltip | `swimlane-task-bar` | `/Users/yin/pto-design-system/patterns/swimlane-task/pattern.json` | 使用 canvas API `window.PtoSwimlaneTaskPattern.*`；不要用 DOM/CSS 重建 task bar。 |
| 3D logical tensor/program-space viewport | 尚无已批准 PTO pattern | 最终 UI 前必须创建 `/Users/yin/pto/tiling/component-preview.html` | 使用 Three.js/OrbitControls 或等价真实 3D renderer。这是 data-viz pattern candidate，不是私有 button/card style。 |
| GM/UB/L1/L0 硬件 route diagram 与 local buffer occupancy | `memory-architecture-layout` | `/Users/yin/pto-design-system/patterns/memory-architecture/pattern.json` | 通过 `setPathFocus` / `clearPathFocus` 做 architecture path focus；shared cell-state API 存在后，用 buffer grid 表达 local data block occupancy。不要把它用作完整 logical tensor tile grid。 |
| Cube 侧可视化的 AIC object shell | `aic-core-object` | `/Users/yin/pto-design-system/patterns/aic-core-object/pattern.json` | 用于 fusion 与 matmul panel 中的 Cube/AIC role。 |
| Vector 侧可视化的 AIV object shell | `aiv-core-object` | `/Users/yin/pto-design-system/patterns/aiv-core-object/pattern.json` | 用于 add 与 fusion panel 中的 Vector/AIV role。 |
| 可选 model/stage graph，用于渲染 semantic stage graph | `model-graphviz` | `/Users/yin/pto-design-system/patterns/model-graphviz/pattern.json` | 当页面拥有 graph data 时用于 graph-like source/stage overlay。 |
| 可选 pass/operator node card，用于展示 source/operator graph node | `pass-ir-graph-node` | `/Users/yin/pto-design-system/patterns/pass-ir-graph-node/pattern.json` | 用于 node card；不要创建私有 graph node styling。 |

Controls 使用现有 PTO base class 与 token：

- buttons：`.btn`、`.btn-solid`、`.btn-ghost`、`.btn-icon`、size modifiers、`.is-selected`
- tabs and segmented controls：`.tab-control`、`.tab-control-item`、`.segmented-control`、`.toolbar-control`、`.toolbar-readout`
- panels and cards：`.panel-shell`、`.panel-shell-quiet`、`.workbench-pane`、`.workbench-pane-body-fill`
- inspector：`.inspector-rail`、`.inspector-section`、`.inspector-soft-card`
- badges/status：`.badge`、status color variables、tag/chip tokens

使用以下文件中的 semantic CSS variables：

- `/Users/yin/pto-design-system/tokens/foundation.css`
- `/Users/yin/pto-design-system/tokens/semantic.css`
- `/Users/yin/pto-design-system/tokens/components.css`

实现必须加载适用于 `/Users/yin/pto/tiling` 的本地 PTO runtime asset path。现有模块通常使用 `../vendor/pto-design-system/...`，但写 HTML 前必须先验证。

### 9.1.1 Pattern Contract Check

消费任何匹配 pattern 之前：

1. 阅读匹配的 `pattern.json`。
2. 确认 required API 已由 `pattern.js` 导出。
3. 确认 preview `pattern.html` 覆盖这些 API。
4. 确认 UI copy 声称的任何能力都由 pattern contract 实现。
5. 如果 preview、contract 与 implementation 不一致，先修 shared pattern，或避免声称该行为。

如果 memory transfer/pipeline visual 需要新 pattern，先创建：

`/Users/yin/pto/tiling/component-preview.html`

再交付真实 workbench。preview 必须展示：

- 最接近的现有 PTO pattern
- proposed new pattern
- token usage
- intended usage
- state coverage：normal、hover、active、selected、disabled（如适用）
- 为什么当前系统不足
- 它应该进入 shared system 还是保持 data-viz-only

不要把未批准的新 visual 放进真实 workbench。

3D tensor viewport 当前超出现有 PTO pattern library。因此 Phase 1 实现必须二选一：

1. 为 Three.js full tensor viewport 创建 `component-preview.html`，并在放入产品页面前获得 approval；或
2. 在该 viewport pattern 完成 review 前，让产品页面保持明确标记的 prototype 状态。

Approval gate 适用于 viewport scene 及其 data-viz states。它不允许为 button、toggle、pane、inspector block、playback control 或 card 创建新的私有样式。

### 9.2 Main Layout

Workbench layout：

- Top bar：
  - standalone mode 下由 `ide-frame` shell 拥有。
  - sample/operator selector
  - mode selector：Visualizer / Puzzle / Developer
  - arch selector：`dav-2201`、`dav-3510`
  - evidence toggles：profiling、sanitizer、dump、clock
- Left pane：
  - source code
  - highlighted current lines
  - semantic label badges
- Center pane：
  - primary 3D full tensor viewport。
  - `blockIdx`、`progress`、`mIndex`、`nIndex`、`kIndex` 以及相关时的 AIV sub-block 的 tensor/program-space controls。
  - current load/store/compute/fusion stage 的 operation chips。
  - UB/L1/L0 state 的 compact on-chip tile lens。
  - 3D viewport 下方可选 swimlane/timeline，但不能替代它。
  - 随 sample type 变化：
    - vector sample：按 block/progress/element 折叠的完整 logical vector space + UB queue lens。
    - cube sample：A/B/C logical matrix spaces + M/N/K reduction lens。
    - fusion sample：C tile producer/consumer split + AIC/AIV sync lens。
    - lightning indexer：semantic stage graph + 选中 stage 的 3D tile preview。
- Right pane：
  - inspector
  - current step details
  - variables
  - tile formulas
  - memory offsets
  - memory-architecture path focus embed 或 inspector section
  - queue/sync state
  - profiling/sanitizer evidence
- Floating playback mount：
  - 由 `floating-playback-control` 提供
  - play/pause
  - step previous/next
  - speed
  - scrubber
  - selected block/tile/stage
  - 不能作为 page-local footer bar 重建

### 9.3 Interaction Rules

- 选择 source line 会高亮对应 visual stage。
- 选择 3D tensor tile 会高亮对应 source operation、formula 与 hardware path focus。
- 选择一个 step 会更新：
  - source highlight
  - 3D tensor tile / operation state
  - on-chip tile lens
  - memory architecture focus
  - inspector
  - playback label
  - Puzzle mode 下的 puzzle prompt
- 切换 arch 会更新：
  - architecture-specific branch
  - LoadData parameter panel
  - 适用时更新 memory tier label
- 在 Developer mode 修改 tiling config 会 recompute execution plan，并标记受影响 stage。
- hover 或选择 architecture diagram 中的 hardware node 应解释 route 与 memory tier，但不能假装选择了 tensor element，除非 trace 有明确 tensor mapping。

## 10. Puzzle Mode Specification

Puzzle mode 应建立在与 Visualizer mode 相同的 trace data 之上。

每个 puzzle 包含：

- prompt
- context variables
- expected answer
- visual hint
- source hint
- validation
- explanation

Example puzzle object：

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

Puzzle categories：

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

Developer mode 把真实 operator 转换为可编辑的 tiling visualization。

### 11.1 Inputs

Required：

- source file path
- operator name or entry function
- shape/config values

Optional：

- generated trace
- profiling output
- sanitizer output
- dump output
- known PTO/PyPTO metadata

### 11.2 Example Developer Session

对于 `lightning_indexer_prolog_quant.py`：

1. 用户选择 source file。
2. 工具检测 `@pypto.jit`。
3. 工具检测 `lightning_indexer_prolog_quant_compute`。
4. 工具检测 dynamic axis `t`。
5. 工具检测 `IndexerPrologQuantConfigs`。
6. 工具列出可编辑 configs：
   - `q_linear`
   - `q_hd`
   - `k_linear`
   - `w_linear`
   - `unroll_list`
   - `l1_reuse_param`
   - `copy_in_threshold`
   - `cycle_upper_bound`
   - `block_size`
7. 用户选择一个 stage，例如 `Query-Linear`。
8. 工具展示 cube tile shapes 与受影响 tensor。
9. 用户编辑 `q_linear`。
10. 工具 recompute visualization：
    - tile dimensions
    - loop count
    - memory movement estimates
    - related downstream stages
11. 用户 attach profiling。
12. 工具 overlay slow stages，并在 tile choices 可能增加 copy pressure 或降低 reuse 时给出 warning。

### 11.3 Developer Mode Outputs

- `trace.json`
- `tiling-report.md`
- optional `.avz`
- optional UI snapshot

Report 应包含：

- semantic stage list
- current tiling config
- memory movement summary
- potential bottlenecks
- evidence mapping
- recommended next experiments

## 12. Implementation Phases

### Phase 0：Spec and Data Contract

Deliverables：

- 本 spec file
- `/Users/yin/pto/tiling/docs/ascend-tiling-visualization-knowledge.md`
- trace schema draft
- fixture directory plan

Acceptance：

- spec 覆盖三个 MVP samples。
- spec 覆盖真实 developer workflow。
- spec 描述 UI、trace、puzzle、evidence、3D tensor viewport、on-chip tile lens 与 memory-architecture path-focus layers。

### Phase 1：Fixture-Driven Visualizer

Deliverables：

- 三个 trace fixtures：
  - Add
  - Matmul
  - Matmul + LeakyRelu fusion
- 如果 Three.js 3D full tensor viewport 尚未作为 shared PTO pattern 获批，则创建对应 `component-preview.html`。
- 使用 fixture data 的 static workbench
- source pane and playback
- 三个 samples 的 3D full tensor viewport
- 当前 selected tile 的 on-chip tile lens
- 当前 selected step 的 memory-architecture path focus

Acceptance：

- 用户可以选择每个 sample。
- 用户可以 step through 每个 sample。
- source highlight、3D tensor tile highlight、on-chip lens、memory architecture focus、playback 与 inspector 一起更新。
- 不依赖 NPU runtime。
- vector sample 清楚展示按 block/progress/element 折叠的 full logical vector space。
- matmul sample 清楚展示 A/B/C logical tensor spaces 与 K-axis accumulation。
- fusion sample 清楚展示 AIC C tile ownership、AIV half-tile ownership、sync 与 direct/avoided intermediate path。

### Phase 2：Puzzle Mode

Deliverables：

- 三个 samples 的 puzzle definitions。
- validation UI。
- explanation panel。

Acceptance：

- 每个 sample 至少有五个 puzzles。
- puzzle answer 从 trace/config data 计算得出。
- incorrect answer 展示 source 与 visual hints。

### Phase 3：Developer Mode for Lightning Indexer

Deliverables：

- 面向 `lightning_indexer_prolog_quant.py` 的 PyPTO lightweight extractor。
- semantic stage graph。
- editable tiling config panel。
- regenerated trace preview。

Acceptance：

- 工具检测 semantic labels。
- 工具检测 cube/vector tile shape calls。
- 工具检测 loop/unroll structure。
- 用户可修改 config values 并看到受影响 stages 更新。

### Phase 4：Evidence Overlay

Deliverables：

- profiling evidence attachment。
- sanitizer evidence attachment。
- dump/clock placeholder support。

Acceptance：

- evidence 可以从 JSON 加载。
- evidence 映射到 stage/source/trace step。
- UI 可以 toggle evidence layers。

### Phase 5：Packaging and Authoring Workflow

Deliverables：

- `.avz` save/load。
- import/export commands。
- 编写新 sample traces 的 docs。

Acceptance：

- 开发者可以添加新 operator fixture 而不改 UI code。
- `.avz` 可以重新打开并 replay。

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

Risk：automatic source parsing 变得过宽。

Mitigation：MVP 使用 fixtures；extractor 先只支持已知 Ascend C/PyPTO patterns。

Risk：UI 变成没有工程价值的 animation。

Mitigation：每个 visual step 必须链接到 source refs、formulas、tensor ranges 与 evidence placeholders。

Risk：3D tensor view 误导用户以为 GM 是物理 3D。

Mitigation：每个 tensor view 必须标注 logical axes、physical addressing、GM offset mapping，以及 visual axis 是否来自执行结构，例如 `blockIdx`、`progress` 或 K reduction。

Risk：architecture diagram 被误用为完整 tensor tile grid。

Mitigation：`memory-architecture-layout` 用于 hardware node focus、route focus 与 local on-chip buffer occupancy。完整 logical tensor tile selection 属于 3D viewport。在 shared cell-state API 存在前，local buffer occupancy 属于 on-chip tile lens 或 component preview。

Risk：fusion visualization 变得过于复杂。

Mitigation：先从 selected tile 的一条 AIC 与两条 AIV lane 开始；不要一开始尝试 whole-chip scheduling。

Risk：PyPTO generated code 与 high-level semantic label 不一致。

Mitigation：Developer Mode 区分 high-level semantic plan 与 runtime evidence。后续有 metadata 时，evidence overlay 可以把 generated kernels 映射回 labels。

Risk：新的 visual style 泄漏进 PTO module。

Mitigation：使用现有 PTO patterns；为新的 pipeline/transfer visuals 创建 component preview 与 approval gate。

## 15. MVP Acceptance Criteria

MVP 完成条件：

- 存在三个 samples：
  - Vector Add
  - Cube Matmul
  - Fusion Matmul + LeakyRelu
- 每个 sample 包含：
  - source pane
  - trace playback
  - 3D full tensor viewport
  - on-chip tile lens
  - memory-architecture path focus
  - inspector
  - 至少五个 puzzle tasks
- Visualizer mode 能解释：
  - block mapping
  - tile size and offsets
  - memory movement
  - Add 的 queue events
  - Matmul 的 Cube memory hierarchy
  - Fusion 的 cross-core sync
  - 为什么 3D tensor 是 flat GM 上的 logical access space，而不是 physical memory shape
  - selected tile 当前 active 的 hardware route
- Developer Mode 可以 lightweight ingest `lightning_indexer_prolog_quant.py` 并展示：
  - semantic stages
  - cube/vector tile shape settings
  - loop/unroll structure
  - editable config values
- 按 `/Users/yin/pto-design-system/SKILL.md` 复用 PTO design system，而不是 ad hoc visual imitation。
- 实现前必须读取匹配 pattern contracts：至少包括 `ide-frame`、`workbench-shell`、`floating-playback-control`、`swimlane-task-bar`、`memory-architecture-layout`、`aic-core-object` 与 `aiv-core-object`。
- 3D tensor viewport 在最终 UI 使用前必须拥有 component preview 或已批准的 shared PTO pattern contract。
- AIC/AIV/memory buffer grids 中的 local data block occupancy 只能通过已批准 shared cell-state APIs 实现；在这些 API 存在前，只能保留在 on-chip tile lens preview。
- 任何缺失的 data-viz pattern 在进入 final UI 前都必须有 preview page。

## 16. Open Questions

- artifact extension 应该是 `.avz`，还是在 save/load 成熟前继续保持 `trace.json`？
- Developer Mode 应该直接编辑 source files，还是只产出 suggested config patch？
- 应优先 normalize 哪种 profiling output format？
- `lightning_indexer_prolog_quant.py` 应只作为 PyPTO source 处理，还是在 generated Ascend C 可用时也消费它？
- Puzzle Mode 应属于同一页面，还是独立 route？

## 17. Immediate Next Steps

1. 创建 `data/schemas/trace.schema.json`。
2. 手写创建三个 MVP fixtures。
3. 为 3D full tensor viewport 以及任何新的 transfer/tile-lens visual states 构建 `component-preview.html`。
4. 为 `PtoAicCorePattern.setBufferBlocks`、`PtoAivCorePattern.setBufferBlocks` 和 `PtoMemoryArchitecturePattern.setBufferBlocks` 添加 shared pattern TODO/proposal。
5. 用 `tensorViews`、`visualState.tensorViewport`、`visualState.onChipLens` 与 `visualState.architectureFocus.bufferBlocks` 扩展 fixtures。
6. 构建一个 fixture-driven static workbench，并使用真实 3D tensor primary view。
7. 在同一 trace model 上添加 Puzzle Mode。
8. 为 `lightning_indexer_prolog_quant.py` 添加 PyPTO lightweight extraction。
