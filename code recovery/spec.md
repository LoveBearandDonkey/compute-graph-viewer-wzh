# Conv2D + Bias + ReLU Code Recovery Spec

## 1. 文档状态

- **目标模块**：`pto_compute-graph-viewer/code recovery/`
- **产品形态**：Ascend C 静态 Code Recovery Workbench
- **首版案例**：`Conv2D + Bias + ReLU`
- **目标架构**：Ascend 910B
- **参考 API 风格**：CANN Community `9.1.0-beta.2` Basic API
- **实现状态**：待开发
- **数据可信度**：源码确认、静态推断、未知；不包含真实运行观测

本 Spec 用于把现有 Tiling Visualization Workbench 的复制版本收敛为一个 Conv 专题 Code Recovery 原型。首版只恢复代码能够支持的实现模型，不将静态控制流、演示参数或视觉动画描述为真实硬件时间线。

---

## 2. 背景

当前页面以 Add、MatMul 和 MatMul + LeakyReLU 三个 trace fixture 为主，通过 Source、Trace Visual、Memory Architecture 三栏展示源码、逻辑 tile、执行步骤和硬件数据路径。

新的主案例来自 `conv_bias_relu_reference.asc`。该文件描述了以下目标执行链：

```text
Feature GM ──→ A1/L1 ──LoadData3D──→ A2/L0A ─┐
                                               │
Filter GM ───→ B1/L1 ──LoadData────→ B2/L0B ──┼─Mmad──→ CO1/L0C
                                               │
Bias GM ─────→ C1 ────────?────────→ C2 ──────┘

CO1/L0C ──DataCopy + ReLU + FP32→FP16──→ C1 ──?──→ Output GM
```

但该文件明确是 execution reference，不是完整、可编译、已上板验证的自定义算子工程。它缺少真实 host tiling、部分函数体、真实 buffer allocation、block 映射和 profiling 数据。

Code Recovery 的价值不是补全这些未知，而是：

1. 从代码中恢复已经存在的对象、阶段、依赖和数据路径。
2. 明确区分代码确认、静态推断和未知。
3. 让同一个 output tile、buffer、stage 和源码位置在不同视图中保持同一身份。
4. 暴露缺失的实现关系，并告诉用户下一步需要什么证据。

---

## 3. 产品目标

### 3.1 一句话目标

让 Ascend C 开发者从一份不完整的 Conv kernel reference 中，看清“代码已经表达了什么、工具推断了什么、哪些关键工程事实仍然缺失”。

### 3.2 目标用户

- 正在学习 Cube Conv、LoadData3D 和 implicit GEMM 的 Ascend C 开发者。
- 需要审查 host、tiling、kernel 和 memory path 是否闭合的算子开发者。
- 负责 Ascend C 开发工具、诊断工具和学习体验的产品与设计人员。
- 后续需要把静态实现模型与 Recommendation Trace、correctness 和 profiling 对齐的工程团队。

### 3.3 核心任务

用户应能完成以下任务：

1. 从源码定位 Conv 的 shape、tiling、buffer、K loop 和 epilogue。
2. 理解语义卷积如何映射到 Cube execution domain。
3. 查看当前 output tile 对应的 feature window、channel range 和 K slice。
4. 沿逻辑执行顺序查看数据在 GM、L1、L0A、L0B、L0C 和 C1/C2 之间的变化。
5. 点击任一代码行、步骤、tile 或 buffer，查看其证据和未解析项。
6. 识别当前 reference 中的实现缺口，而不是被完整动画误导。

### 3.4 成功标准

- 页面不再把 `M×N×K` 描述为三维 tensor。
- 用户能区分语义 tensor、Cube execution domain 和物理 buffer layout。
- 每个可见事实都有证据等级和源码链接。
- Bias C1→C2、L1 allocation、block mapping 和未实现 helper 等缺口可见且可追溯。
- 没有 runtime evidence 时，页面不显示伪造 duration、overlap、stall 或 profiling 结论。

---

## 4. 非目标

首版不包含：

- 通用 Ascend C parser 或任意源码导入。
- 自动生成 host tiling。
- 自动修复 `conv_bias_relu_reference.asc`。
- CANN 编译、上板 correctness 或 profiling。
- 真实硬件 duration、lane overlap、pipeline bubble 或性能诊断。
- Recommendation Trace 候选生成。
- 多算子 sample browser。
- AIC→AIV Vector epilogue 或 Cube–Vector 融合路径。

旧 Add、MatMul 和 Fusion fixture 文件可以保留为参考，但不注册、不加载，也不出现在首版 UI 中。

---

## 5. 可信度与证据模型

### 5.1 证据等级

所有可见对象、关系和结论必须使用以下一种状态：

| 状态 | 含义 | UI 表达 |
| --- | --- | --- |
| `confirmed` | 源码或明确输入直接确认 | 默认实体状态 +「代码确认」或「演示输入」 |
| `inferred` | 根据控制流、命名、注释或 API 语义进行静态推断 | 虚线/弱化连接 +「静态推断」 |
| `unknown` | 当前材料不足以确认 | 中性占位/断开连接 +「待补全」 |

首版不使用 `estimated` 和 `observed`。

### 5.2 演示输入与源码恢复的区别

参考源码中的 `tiling` 被初始化为空值，无法直接驱动可视化。首版提供固定演示 context，但必须标记为：

> Demo context · not recovered from host tiling

演示输入属于页面输入，不属于源码恢复结论。由演示输入计算出的 `Ho/Wo/M/K/N` 和 tile 数必须保留其 input provenance。

### 5.3 必须显示的未知

以下内容不得被 mock 数据覆盖：

1. `FmapA1Elements()` 返回 0，真实 feature L1 staged window 大小未知。
2. `WeightB1Elements()` 返回 0，真实 filter L1 staged size 未知。
3. `outputTileIndex` 没有与 block/core 建立实际映射。
4. `CopyFeatureTileGmToA1` 没有函数体。
5. `CopyFilterTileGmToB1` 没有函数体。
6. `CopyBiasGmToC1` 没有函数体。
7. `CopyOutputTileC1ToGm` 没有函数体。
8. `LoadFilterToL0B` 没有函数体。
9. LoadData3D 的 pad/start/extension 参数仍需目标 CANN 版本确认。
10. Bias 被搬入 C1，但 Mmad 读取 `biasC2`；源码中缺少明确的 C1→C2 路径。
11. epilogue API 字段和 C1→GM 路径仍标有 `VERIFY-ON-TARGET`。
12. 没有 runtime trace，因此所有真实 timing、overlap 和 stall 均未知。

---

## 6. 演示执行上下文

首版使用以下固定 context，以获得可读的空间窗口、M/N tile 和 K loop：

| 字段 | 值 |
| --- | ---: |
| Batch | 1 |
| Ci | 16 |
| Hi | 8 |
| Wi | 8 |
| Co | 32 |
| Kh | 3 |
| Kw | 3 |
| Stride H/W | 1 / 1 |
| Pad T/R/B/L | 1 / 1 / 1 / 1 |
| Dilation H/W | 1 / 1 |
| Groups | 1 |
| Feature / Filter / Output | FP16 |
| Accumulator / Bias | FP32 |

派生值：

```text
Ho = 8
Wo = 8

Cube M = Ho × Wo = 64
Cube K = Ci × Kh × Kw = 144
Cube N = Co = 32
```

演示 tile：

```text
tileM = 32
tileK = 48
tileN = 16

M tile count = 2
N tile count = 2
K loop count = 3
```

这些 tile 值只服务于交互演示，不表示 CANN 自动 tiling、推荐结果或已验证的最优配置。

---

## 7. 统一对象身份

所有视图必须使用稳定 ID，不得在 Source、Tiling、Timeline 和 Memory Architecture 中分别维护同一对象的不同名称。

### 7.1 Tensor

| ID | 名称 | 语义 |
| --- | --- | --- |
| `tensor:feature` | Feature X | `X[N,Ci,Hi,Wi]` |
| `tensor:filter` | Filter W | `W[Co,Ci,Kh,Kw]` |
| `tensor:bias` | Bias | `Bias[Co]` |
| `tensor:output` | Output Y | `Y[N,Co,Ho,Wo]` |

### 7.2 Buffer

| ID | 名称 | 位置 |
| --- | --- | --- |
| `buffer:feature:a1` | fmapA1 | A1 / logical L1 |
| `buffer:filter:b1` | weightB1 | B1 / logical L1 |
| `buffer:bias:c1` | biasC1 | C1 |
| `buffer:bias:c2` | biasC2 | C2 |
| `buffer:feature:a2` | fmapA2 | A2 / L0A |
| `buffer:filter:b2` | weightB2 | B2 / L0B |
| `buffer:accum:co1` | accumCo1 | CO1 / L0C |
| `buffer:output:c1` | outC1 | C1 |

### 7.3 Stage

| ID | 名称 |
| --- | --- |
| `stage:init` | Init GM Views |
| `stage:allocate` | Allocate Local Tensors |
| `stage:copy-feature` | Feature GM → A1 |
| `stage:copy-filter` | Filter GM → B1 |
| `stage:copy-bias` | Bias GM → C1 |
| `stage:sync-mte2-mte1` | MTE2 → MTE1 |
| `stage:load-feature` | LoadData3D A1 → A2 |
| `stage:load-filter` | LoadData B1 → B2 |
| `stage:sync-mte1-m` | MTE1 → M |
| `stage:mmad-bias-init` | Mmad + Bias Init |
| `stage:mmad-accumulate` | Mmad K Accumulate |
| `stage:sync-m-fix` | M → FIX |
| `stage:epilogue` | CO1 → C1 + ReLU + Cast |
| `stage:copy-output` | C1 → Output GM |

### 7.4 Tile domain

| ID | 含义 |
| --- | --- |
| `tile:output:m-n` | 当前 `tileM × tileN` 输出 tile |
| `tile:feature-window` | 当前 output positions 对应的 feature receptive fields |
| `tile:reduction:k` | 当前 `tileK` reduction slice |
| `tile:a2` | `tileM × currentK` |
| `tile:b2` | `currentK × tileN` |
| `tile:co1` | `tileM × tileN` |

---

## 8. 页面结构

### 8.1 PTO Shell

页面必须继续使用共享 PTO pattern：

- `ide-frame`
- `workbench-shell`
- `floating-playback-control`
- `swimlane-task`
- `hardware-architecture-viewport`
- `memory-architecture-layout`
- `aic-core-object`

页面不创建私有 button、toggle、badge、card、panel 或 architecture visual grammar。

默认设置：

- Theme：dark
- Host：standalone
- Activity rail：保留 Explorer、Search、Source Control、Terminal
- 三栏可拖拽
- Inspector 默认关闭
- Playback 保留
- 页面专属 split/localStorage key

页面标题：

```text
Ascend Code Recovery
Conv2D · Static Implementation Model
```

状态栏常驻：

```text
Logical order · no runtime timing
Conv2D + Bias + ReLU
Step x/12
Ascend 910B
```

### 8.2 左栏：Source & Evidence

标题：`Source`

内容：

- 完整显示 `conv_bias_relu_reference.asc`。
- 保留真实行号。
- 当前 step 高亮一个或多个相关源码行。
- 行级证据标记区分 `confirmed/inferred/unknown`。
- Source pane 内不再显示旧 sample tabs。

用户点击代码行时：

1. 选择对应 stable object。
2. 更新中栏相关 tile/stage。
3. 更新右栏 hardware focus。
4. 打开 Inspector。

Inspector 按以下结构展示：

1. 正在发生什么
2. 所在链路层级
3. 代码证据
4. 静态推断
5. 未知与缺失证据
6. 下一步验证

### 8.3 中栏：Conv Execution Model

原 `Tensor 3D Viewport` 替换为二维联动视图，禁止将 K 画成输出 tensor 的第三维。

#### A. Feature Map / Load3D

- 显示 `Hi×Wi` 的二维空间网格。
- 高亮当前 output position 对应的 `Kh×Kw` receptive field。
- padding 位置显示为空值/填充值区域。
- channel 使用 `Ci range` 和 `C0 group` 标签，不使用空间深度轴。
- 当前 K slice 映射到 `Ci×Kh×Kw` 中的具体范围。

#### B. Output Tile

- 显示 `Ho×Wo` 空间网格。
- 高亮当前 `tileM` 对应的 flatten positions。
- 单独显示当前 `Co` channel range，即 Cube N tile。
- 明确显示：

```text
M = flattened Ho×Wo positions
N = output channels
K = Ci×Kh×Kw reduction
```

#### C. Cube Tile Lens

同时展示：

- A2：`tileM × currentK`
- B2：`currentK × tileN`
- CO1：`tileM × tileN`
- Bias：当前 `tileN` channel slice

状态：

- `allocated`
- `loaded`
- `initializing`
- `accumulating`
- `committed`
- `unknown`

K0 使用 Bias 初始化 CO1；K1/K2 继续累加。Bias C1→C2 必须以断开的 unknown connection 显示。

#### D. Logical Execution Sequence

标题：

```text
Logical Execution Sequence
Order only · duration unavailable
```

要求：

- 使用等宽离散 step。
- 不显示时间刻度。
- 不显示毫秒、微秒或 cycle。
- 宽度不表示 duration。
- 使用共享 `PtoSwimlaneTaskPattern.drawTaskBar` 单段模式。
- 不传入虚构的 input/output timing arrays。
- 点击 step 更新所有视图。

### 8.4 右栏：Memory Architecture

标题：`Memory Architecture`

默认 preset：`ascend910b`

显示以下粗粒度路径：

1. Feature GM → L1/A1 → L0A/A2
2. Filter GM → L1/B1 → L0B/B2
3. Bias GM → C1 → C2
4. L0A + L0B + Bias → Cube → L0C
5. L0C → C1 → Output GM

约束：

- 使用共享 path focus 和 buffer-block API。
- route geometry 不写在产品 CSS 中。
- focus 不改变 route stroke width。
- C1/C2 若不存在精确共享硬件节点，只在 tile lens 表达；架构图只聚焦可确认的上层区域。
- C1→C2 和 C1→GM 未确认部分用 unknown，不绘制为完整实线硬件路径。
- 默认不选中业务节点，不 dim 全图。

### 8.5 Inspector

Inspector 默认关闭，只在用户点击对象时打开。

内容至少包含：

- 对象 stable ID
- 当前 stage
- 当前 M/N/K range
- 相关 tensor/buffer
- source refs
- evidence kind
- sync dependency
- API 参数
- missing evidence
- applicability

对 unknown 项提供明确的下一步，例如：

- 需要 host tiling 实现
- 需要补齐 helper 函数
- 需要安装版本的 CANN header
- 需要 correctness run
- 需要 profiling trace

---

## 9. 逻辑回放

首版采用 12 个逻辑步骤：

| Step | Stage | 主要变化 | 证据 |
| ---: | --- | --- | --- |
| 1 | Init GM Views | 建立 feature/filter/bias/output GlobalTensor | confirmed |
| 2 | Allocate Local Tensors | 建立 A1/B1/C1/C2/A2/B2/CO1/outC1 | confirmed；A1/B1 size unknown |
| 3 | Copy Inputs | 调用 Feature/Filter/Bias GM→local helper | 调用 confirmed；具体范围 inferred/unknown |
| 4 | MTE2→MTE1 | 等待输入搬运完成 | confirmed |
| 5 | K0 Load | LoadData3D A1→A2；Filter B1→B2 | feature confirmed；filter helper unknown |
| 6 | MTE1→M | 等待 L0 输入可供 Cube 使用 | confirmed |
| 7 | K0 Mmad + Bias | `cmatrixInitVal=true`，使用 biasC2 初始化 CO1 | confirmed；C1→C2 unknown |
| 8 | K1 Mmad | CO1 累加第二个 K slice | confirmed |
| 9 | K2 Mmad | CO1 累加最后一个 K slice | confirmed |
| 10 | M→FIX | 等待 Cube 结果进入输出阶段 | confirmed |
| 11 | Epilogue | CO1→C1，FP32→FP16，ReLU requested | code confirmed；target behavior inferred |
| 12 | Copy Output | C1→Output GM helper | 调用 confirmed；实现 unknown |

播放行为：

- Back/Forward：切换一步。
- Play：按固定 UI 节奏自动切换逻辑 step。
- Scrubber：选择 step index。
- UI 播放速度只表示演示速度，不表示硬件速度。
- 播放不会自动打开 Inspector。

---

## 10. Trace 数据接口

现有 trace schema 做最小扩展。

### 10.1 Root

```json
{
  "schemaVersion": "0.2",
  "execution": {
    "mode": "logical-order",
    "durationStatus": "unknown"
  }
}
```

### 10.2 Context

```json
{
  "context": {
    "source": "demo-input",
    "targetSoC": "Ascend 910B",
    "cannVersion": "9.1.0-beta.2 style; unverified",
    "shape": {},
    "dtype": {},
    "layout": {
      "featureGm": "unknown",
      "featureL1": "NC1HWC0/NZ-derived; inferred",
      "filter": "unknown",
      "output": "unknown"
    }
  }
}
```

### 10.3 Evidence

stage、step 和 memory object 增加：

```json
{
  "objectId": "stage:load-feature",
  "evidenceKind": "confirmed",
  "sourceRefs": [
    {
      "line": 132,
      "symbol": "LoadFmapWindow3D"
    }
  ],
  "applicability": "conv_bias_relu_reference.asc demo context",
  "missingEvidence": []
}
```

### 10.4 Conv viewport

继续使用现有 `visualState.tensorViewport` 容器：

```json
{
  "kind": "conv2d-cube",
  "featureWindow": {
    "outputPositionRange": [0, 32],
    "spatialCells": [],
    "paddingCells": []
  },
  "outputTile": {
    "mRange": [0, 32],
    "coRange": [0, 16]
  },
  "reduction": {
    "kRange": [0, 48],
    "step": 0,
    "total": 3
  },
  "cubeTiles": {
    "a2": {},
    "b2": {},
    "co1": {},
    "bias": {}
  }
}
```

### 10.5 Architecture focus

继续使用现有接口：

```json
{
  "architectureFocus": {
    "selectors": [],
    "routes": [],
    "bufferBlocks": []
  }
}
```

产品页不得直接修改共享 architecture renderer 生成的 DOM。

---

## 11. 文件与实现组织

主要修改：

```text
code recovery/
├── spec.md
├── index.html
├── README.md
├── CLAUDE.md
├── data/
│   ├── schemas/trace.schema.json
│   ├── sources/conv_bias_relu_reference.asc
│   └── fixtures/conv_bias_relu.trace.json
└── src/
    ├── app.js
    └── styles.css
```

实现原则：

- 复用现有 vanilla JS 架构，不引入 bundler。
- 首版可以在现有 IIFE 中新增 Conv derivation，但应把 Conv 数学派生、visual-state derivation 和 renderer 分成清晰函数组。
- fixture 是展示事实的 source of truth。
- `app.js` 只负责状态、派生、联动和渲染。
- `styles.css` 只负责模块布局和 data-viz 状态，不创建新视觉系统。
- 修改 JS/CSS 后更新 `index.html` cache-busting query。

---

## 12. 实施阶段

### Phase 1：数据基线

1. 复制 reference source。
2. 创建 Conv fixture。
3. 扩展 schema。
4. 建立 stable object IDs。
5. 为每个对象补 evidenceKind/sourceRefs/missingEvidence。
6. 停止注册旧 samples。

完成条件：

- Fixture 可加载。
- Schema 校验通过。
- 12 个 step 均能关联到 source 和 stage。

### Phase 2：静态恢复模型

1. 实现 Conv shape 和 tile 派生。
2. 实现 K slice 与 feature window 的映射。
3. 实现 source line→object 和 object→step 映射。
4. 实现 unknown relation。
5. 把 Bias C1→C2 缺口建模为首个 unresolved dependency。

完成条件：

- 所有 M/N/K range 可解释。
- 所有未知均保留，不被默认值覆盖。

### Phase 3：中心视图

1. 删除 3D tensor 叙事。
2. 实现 Feature Map / Load3D 视图。
3. 实现 Output Tile 视图。
4. 实现 Cube Tile Lens。
5. 实现 logical-order timeline。
6. 接入 playback。

完成条件：

- 用户能从 output tile 回到 feature window、K slice 和 Cube tile。
- 页面没有 rank/axis 误导。

### Phase 4：Memory Architecture 与 Inspector

1. 切换为 910B preset。
2. 添加 Conv path focus payload。
3. 映射 local buffer occupancy。
4. 实现 evidence inspector。
5. 实现源码、timeline、tile、buffer 的双向选择。

完成条件：

- 选择一个 stable object 后，三个 pane 聚焦同一对象。
- unknown 硬件路径不显示为 confirmed。

### Phase 5：收尾与验证

1. 更新 README/CLAUDE 的真实目录和运行方式。
2. 更新 cache key。
3. 检查 PTO pattern contract。
4. 检查 legacy container decoration。
5. 运行本地 HTTP 服务完成视觉验证。

---

## 13. 验收测试

### 13.1 数据测试

- JSON 可解析并通过 schema。
- 所有 source line 均存在于复制的 `.asc` 文件。
- 所有 stageId/objectId 有定义。
- 每个可见对象包含 evidenceKind。
- `Ho/Wo/M/K/N` 计算正确。
- tile count 和 K loop count 正确。
- K 最后一段范围不会越过 144。

### 13.2 交互测试

- 首次打开只显示 Conv。
- 12 个 step 可前进、后退、播放和 scrub。
- Step 更新 Source、Conv View、Tile Lens、Architecture 和 Status。
- 点击源码行能选中对应 stage/object。
- 点击 tile、timeline 或 buffer 能打开 Inspector。
- Explorer 和 Inspector 可重复开关。
- Fit、zoom、pan 和 pane resize 可重复使用。
- 页面刷新后 split 状态不会污染原 tiling 页面。

### 13.3 语义测试

- 不存在 `Tensor 3D Viewport` 文案。
- 不把 K 称为 C/Y 的第三维。
- 正确展示：
  - `X[N,Ci,Hi,Wi]`
  - `W[Co,Ci,Kh,Kw]`
  - `Y[N,Co,Ho,Wo]`
- 正确解释：
  - `M=Ho×Wo`
  - `K=Ci×Kh×Kw`
  - `N=Co`
- 演示 tiling 明确标为 input context。
- timeline 明确标为 logical order。
- Bias C1→C2 显示为 unknown。
- 未实现 helper 不显示为完整数据搬运事实。

### 13.4 视觉与可访问性测试

- 验证 1440×900、1920×1080 和高分辨率窗口。
- 中栏变窄时仍能读取 Feature、Output 和 Cube Lens。
- timeline step 可点击并有明确 selected state。
- Canvas 提供有意义的 aria-label 和无 Canvas 时的文字 fallback。
- 键盘 focus 可见。
- 架构 route focus 不改变线宽。
- 默认状态不选中业务对象、不 dim 架构图。

### 13.5 PTO 设计系统检查

- 页面仍以 `ide-frame` 为 shell。
- 使用共享 split、playback、timeline 和 architecture API。
- 没有私有 button/toggle/badge/card 系统。
- 没有重复 pane chrome。
- 检查以下 legacy decoration：
  - `border-left`
  - `border-inline-start`
  - `box-shadow: inset`
  - generic container `::before/::after`
  - `linear-gradient(90deg)`
  - `linear-gradient(to right)`
- 每个残留项标记为 PTO-owned、data-viz-exempt 或待处理。

---

## 14. 运行与验证

该页面没有构建系统，必须通过 HTTP 服务访问，不能直接使用 `file://`。

建议从 `pto_compute-graph-viewer/` 启动本地服务，然后打开：

```text
/code%20recovery/index.html
```

最终交付前至少验证：

1. fixture fetch 成功。
2. vendor PTO pattern 加载成功。
3. 910B architecture iframe ready。
4. 12 个 step 均能播放。
5. 浏览器控制台无未处理异常。

---

## 15. 工作量

静态恢复演示首版预计：

| 工作项 | 估算 |
| --- | ---: |
| Fixture、schema、证据映射 | 0.5–1 天 |
| Conv 二维 tiling / Load3D 视图 | 1–1.5 天 |
| Cube lens 与 logical timeline | 0.5–1 天 |
| 910B architecture 联动 | 0.5 天 |
| Inspector、测试与收尾 | 0.5–1 天 |
| **合计** | **3–5 个工作日** |

该估算不包含：

- host tiling 工程；
- kernel 补全；
- CANN 编译；
- correctness；
- 910B 上板；
- profiling；
- target/implemented/observed compare。

---

## 16. 后续演进

首版完成后按以下顺序扩展：

1. 补齐 host tiling 和真实 block→output tile mapping。
2. 补齐 reference 中未实现的 helper 和 Bias C1→C2 路径。
3. 在目标 CANN/910B 环境完成编译与 correctness。
4. 将已验证实现升级为 GM-003。
5. 采集 profiling，并以 observation overlay 附加到现有对象。
6. 比较 Recommendation target、Code Recovery implemented 和 runtime observed。

任何 runtime 数据只能覆盖其实际运行的 shape、tiling、SoC 和 CANN context，不得泛化为整个算子。
