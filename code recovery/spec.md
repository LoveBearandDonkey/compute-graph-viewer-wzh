# Conv2D + Bias + ReLU Code Recovery Spec

## 1. 文档状态

- 产品：Ascend Code Recovery Workbench
- 案例：固定参数 Conv2D + Bias + ReLU
- Spec 版本：0.5
- 更新日期：2026-08-05
- 当前里程碑：阶段 2/5 完成；阶段 3 待开始
- 模型性质：Static Code-Recovery Model
- 非运行事实：未编译、未上板、未做 correctness/profiling

本文档定义产品行为与验收边界；实施进度详见 `Conv + Bias + ReLU Code Recovery 详细实施计划.md`。

本 Spec 服务于 UX 概念验证和体验 Pattern 提炼，不代表真实 Ascend C 工程建议或产品规划。

## 2. 问题定义

Ascend C 开发者需要同时连接 Host tiling 与 Kernel 执行，但常见工具把两者分开：

- Host 决定 shape、M/K/N、tile 数、blockDim 和分配规模；
- Kernel 决定 block 映射、搬运、LocalTensor、同步、Mmad 和 Fixpipe；
- 单看任一文件都难以回答“当前代码为什么处理这一块数据、由哪个硬件单元执行”。

本工具通过统一执行上下文将两份源码、Tensor 变化、硬件参与、Event 和执行顺序连接起来。执行上下文可以是逻辑步骤、instruction、event dependency 或 Loop Group。

## 3. 产品目标

### 3.1 一句话目标

让用户从 Host/Kernel 任一代码位置进入，都能在同一执行上下文中理解数据形态、搬运路径、硬件参与、存储用量和同步依赖。

### 3.2 用户

- 学习 Ascend C Conv/Cube 编程模型的开发者；
- 调试 tiling、format、memory 和 Event 的算子开发者；
- 设计 Ascend C IDE、trace viewer 或诊断工具的产品/UX 团队。

### 3.3 成功标准

- Host 和 Kernel 不是两个孤立代码阅读器；
- 每个可见步骤可追溯到一个或多个文件行号；
- shape/layout/dtype/bytes/地址和 Event 有明确证据等级；
- 逻辑顺序不冒充真实时间；
- 静态源码结论不冒充已编译或设备观测。

## 4. 非目标

- 解析任意 Ascend C 工程；
- 替代编译器、Tiling API 或 profiler；
- 声称当前固定 tiling 是自动生成或最优；
- 提供真实 910B 总容量结论；
- 生成未由源码或 fixture 支撑的 duration；
- 恢复 Inspector；
- 把物理 Output `[64,32]` 误写为 NCHW。

## 5. 权威输入

### 5.1 源码

| fileId | 文件 |
| --- | --- |
| `host` | `src/conv_bias_relu_complete_demo/op_host/conv_bias_relu_tiling.cpp` |
| `kernel` | `src/conv_bias_relu_complete_demo/op_kernel/conv_bias_relu_reference_complete.asc` |

### 5.2 辅助解释

`Conv2D + Bias + ReLU 执行过程.md` 用于组织语义，但若其描述与源码冲突，以源码事实为准，并将冲突记录为待验证项。

## 6. 执行上下文

### 6.1 Tensor 合约

| Tensor | 逻辑 Shape | GM 物理 Shape | 物理 Format | Dtype | Bytes |
| --- | --- | --- | --- | --- | ---: |
| Feature X0 | `[1,16,8,8]` | `[1,1,8,8,16]` | NC1HWC0 | FP16 | 2048 |
| Weight W0 | `[32,16,3,3]` | `[144,32]` | ND | FP16 | 9216 |
| Bias D0 | `[32]` | `[32]` | ND | FP32 | 128 |
| Output Y0 | `[1,32,8,8]` | `[64,32]` | ND | FP16 | 4096 |

Output `[64,32]` 等价 NHWC `[1,8,8,32]`。若接口需要 NCHW，必须新增独立 layout conversion，不得在当前 Fixpipe 步骤中隐含转换。

### 6.2 Cube 与 Tiling

```text
A[64,144] × B[144,32] + Bias[32] → C[64,32]
```

```text
tileM = 16
tileK = 16
tileN = 16

mTiles = 4
kTiles = 9
nTiles = 2
outputTileCount = 8
blockDim = 8
```

### 6.3 Core 映射

```text
OT = Mi × 2 + Nj
Mi = blockIdx / 2
Nj = blockIdx % 2
mStart = Mi × 16
nStart = Nj × 16
```

OT0～OT7 各负责一个 `[16,16]` 输出 Tile。

## 7. 证据模型

| 状态 | 含义 | 示例 |
| --- | --- | --- |
| `confirmed` | 当前源码直接表达 | tile=16/16/16、C1→C2、reluEn=true |
| `derived` | 从 confirmed 参数确定性计算 | Ho=8、K=144、outputOffset |
| `inferred` | 依据 API 语义解释 | 某硬件别名或物理参与关系 |
| `unverified` | 源码表达目标行为，但目标环境未验证 | CANN API 字段/重载实际可编译性 |
| `unknown` | 当前材料不足 | 未来 profiling duration |

界面必须允许同一对象同时表达：

- 代码事实：confirmed；
- 目标环境状态：unverified。

## 8. 页面信息架构

```text
┌────────────────────────────────────────────────────────────┐
│ Source              │ Tensor State        │ Hardware       │
│ Host | Kernel tabs  │ & Transformation    │ Participation  │
├────────────────────────────────────────────────────────────┤
│ Instructions | Timeline                                  │
│ Playback                                               │
└────────────────────────────────────────────────────────────┘
```

### 8.1 Source

- 使用现有 PTO Pane 与 Tab Control；
- 两个 Tab 均加载完整文件；
- 行号、C++/Ascend C 语法高亮；
- 当前步骤高亮一行或多行；
- 可执行/可解释行显示语义 Tag；
- HardEvent Tag 显示事件类型；
- 不引入 Monaco；
- 不出现 Inspector。

### 8.2 Tensor State & Transformation

当前步骤至少显示：

- Tensor/Buffer 名称与角色；
- logical/physical shape；
- logical/physical layout；
- dtype；
- valid/allocated elements 与 bytes；
- padding；
- location；
- transformation；
- confidence/source/verification note。

### 8.2.1 Tensor Data Dump

用户需要从抽象 shape 下钻到具体 Tile 的数值，例如一个 `16×16` 的 Output Tile，看到每个坐标分别是什么数据。

- Data Dump 在 Tensor 视图中展示具体 Tile 的实际矩阵数据；
- 数据必须关联具体运行、Instruction、Tile、dtype、layout 和存储位置；
- 实际数值只能来自 Data Dump 或其他明确的运行数据；
- 没有 dump 数据时只显示 shape 与元数据，不生成或猜测矩阵内容；
- 使用坐标网格表达 Tile，每个单元格显示对应数值；
- 数值格式必须匹配 dtype（FP16 / FP32 等）与 layout；
- 与 Instruction、Tensor State 和 Hardware Participation 保持同一执行上下文；
- 无数据状态显示 `No dump data` 或等价表述，不伪造数值。

### 8.3 Hardware Participation

存储节点显示 used bytes / demo capacity；执行单元只表达当前逻辑状态：`Participating`、`Waiting on dependency`、`Not involved`。不得用 `Idle` 暗示真实硬件空闲，也不得给 MTE/Cube/Fixpipe 绘制“空间占用”。

当前阶段使用简化 910B 架构图。硬件总容量属于 Demo 配置；精确参与路径在阶段 3 完成。

### 8.4 Execution Dock

Instructions 表达逻辑顺序和依赖，不使用时间比例。

当前范围内 Timeline 始终显示 unavailable：

```text
Estimated Timeline unavailable
Not Profiling Data
```

只有未来存在明确来源和适用范围的估算或 profiling 数据时才重新评估。仅出现 fixture `startTime/duration` 字段不足以启用 Timeline。

Playback 只推进离散步骤。

## 9. Source 双 Tab 交互

### 9.1 状态

```text
selectedExecutionContext
activeSourceFileId
executionView
playing
```

`selectedExecutionContext` 是跨区域共享的执行上下文，可以聚焦 step、instruction、event dependency 或 Loop Group。`activeSourceFileId` 只控制左栏显示哪个文件，不创建第二套执行上下文。

当前阶段的实现可以继续使用 `selectedStepId` 作为兼容锚点；阶段 4 的体验验收以共享执行上下文为准，不要求对外暴露内部状态结构。

### 9.2 规则

1. 切换 Source Tab 不改变当前执行上下文；
2. 点击代码行选择与 `fileId + line` 匹配的步骤；
3. 点击代码行后保留当前 Tab；
4. 播放或聚焦执行对象时，自动切到当前上下文的 primary `sourceRef.fileId`；
5. 一个步骤可关联两个文件；
6. 当前 Tab 无该步骤引用时不显示虚假高亮；
7. 代码滚动只定位当前文件中的首个 active line。

## 10. Trace 接口

### 10.1 多文件

```ts
type SourceFile = {
  id: "host" | "kernel" | string;
  label: string;
  path: string;
  projectPath: string;
  language?: "cpp" | "ascendc" | string;
  lines?: SourceLine[];
};
```

### 10.2 文件感知引用

```ts
type SourceReference = {
  fileId: string;
  lines: number[];
  symbol?: string;
  note?: string;
};
```

### 10.3 Step

```ts
type TraceStep = {
  id: string;
  stageId: string;
  label: string;
  summary: string;
  sourceRefs: SourceReference[];
  sourceLines: number[]; // legacy Kernel compatibility
  loop?: {
    kIndex?: number;
    kRange?: [number, number];
    iterationRange?: [number, number];
  };
  tensorSnapshots?: TensorSnapshot[];
  dataFlows?: DataFlow[];
  eventDependencies?: string[];
  evidenceKind: Confidence;
};
```

### 10.4 Root

```ts
type TraceModel = {
  schemaVersion: "0.3";
  operator: Operator;
  launch: Launch;
  tiling: Tiling;
  sources: SourceFile[];
  source: LegacySource;
  tensors: TensorDefinition[];
  buffers: BufferDefinition[];
  events: EventDependency[];
  stages: Stage[];
  steps: TraceStep[];
  memory: MemoryModel;
  evidence: Evidence;
};
```

## 11. 逻辑步骤

| # | stageId | 说明 | source |
| ---: | --- | --- | --- |
| 1 | `host-shape` | 参数和 Ho/Wo | host |
| 2 | `host-tiling` | M/K/N、tile 和 tile count | host |
| 3 | `host-memory` | GM/Local 元素数 | host |
| 4 | `host-launch` | blockDim 与 block→OT | host + kernel |
| 5 | `allocate` | LocalTensor 地址与容量 | kernel |
| 6 | `copy-inputs` | Feature/Weight/Bias GM→L1 | kernel |
| 7 | `sync-mte2-mte1` | L1 ready | kernel |
| 8 | `bias-c1-c2` | Bias Table 搬运 | kernel |
| 9 | `load-k` | K0 Load3D + Load2D | kernel |
| 10 | `sync-mte1-m` | A2/B2 ready | kernel |
| 11 | `mmad-init` | I0 + Bias | kernel |
| 12 | `sync-m-mte1` | I1～I8 Buffer reuse | kernel |
| 13 | `mmad-loop` | I1～I8 Loop Group，包含 I8 final | kernel |
| 14 | `sync-m-fix` | Acc8 ready | kernel |
| 15 | `fixpipe-output` | ReLU/cast/layout/write GM | kernel |

## 12. 数据变化规范

### 12.1 CopyIn

```text
X0 GM NC1HWC0 [1,1,8,8,16] → A1 same format, 2048 B
W[Nj] GM ND [144,16] → B1 NZ, 4608 B
D[Nj] GM ND [16] → C1 linear, 64 B
```

Tensor State 使用 Feature X、Weight W、Bias 三个页签逐一表达 source slice、目标 Buffer、layout、dtype、bytes、L1 地址与对齐信息。切换页签不得改变当前执行上下文、源码高亮、Timeline 或 Hardware Participation。

- Feature 的 GM source 与 A1 destination 复用 PTO `tensor-volume-canvas`，以 `neutral + base` 的 `W × H × C0` 物理体积对照 NC1HWC0 搬运，不对 source 或 destination 添加语义色。
- Weight 的 GM ND slice 与 B1 NZ destination 复用 PTO `matrix-canvas`，保持 `[K=144,N tile=16]` 源坐标范围，由中间 transformation 明确表达 `ND → NZ`。
- Bias 的 GM slice 与 C1 staging destination 复用 PTO `matrix-canvas`，按 `1 × N tile` 表达 16 个 FP32 值。
- Pattern 只承担 tensor 几何、轴与当前/已写入状态；MTE2、bytes、地址、对齐与同步状态由 Tensor State 页面层提供。

CopyIn 完成只表示三路数据已由 MTE2 写入 A1/B1/C1：

```text
Copied by MTE2 → Awaiting MTE2_MTE1 → MTE1 blocked
```

下一步 `sync-mte2-mte1` 的 `SetFlag/WaitFlag` 完成后，A1/B1/C1 才进入 `Readable by MTE1`。Event 只改变可消费状态，不再次搬运 Tensor 数据。Bias C1→C2 属于后续独立步骤，不得合并进 CopyIn。

### 12.2 Bias

```text
C1 [16] FP32 → C2 / Bias Table [16] FP32
64 B
```

### 12.3 每次 K Load

```text
A1 NC1HWC0 → LoadData3D → A2 [16,16] ZZ, 512 B
B1 NZ → LoadData2D + transpose → B2 [16,16] ZN, 512 B
```

LoadData3D 需要表达 Padding、滑窗取数和 Image-to-Column，不得画成三维矩阵乘法。

### 12.4 Mmad

I0：

```text
A2[16,16] FP16 ZZ × B2[16,16] FP16 ZN + C2[16] FP32
→ CO1[16,16] FP32 NZ
```

I1～I8：

```text
CO1 + A2 × B2 → CO1
```

### 12.5 Fixpipe

```text
CO1 [16,16] FP32 NZ
→ ReLU
→ FP32 to FP16
→ ND [16,16]
→ GM outputOffset = mStart × 32 + nStart
```

每核写出 512 B。

## 13. Event 规范

| Event | Producer | Consumer | 作用 |
| --- | --- | --- | --- |
| `MTE2_MTE1` | MTE2 | MTE1 | 防止 MTE1 过早读 L1 |
| `M_MTE1` | Cube | MTE1 | 防止单 Buffer A2/B2 被过早覆盖 |
| `MTE1_M` | MTE1 | Cube | 防止 Cube 过早读 A2/B2 |
| `M_FIX` | Cube | Fixpipe | 防止 Fixpipe 过早读 CO1 |

Event 是执行依赖，不是 Tensor 搬运；视觉上必须与 DataFlow Edge 区分。

Event 在 Code Recovery 中的主载体是 Execution Dock；Source 回链同步代码，Tensor State 表达 waiting/ready/reusable 等状态影响，Hardware Participation 只表达执行单元和依赖方向。当前阶段只要求位置、类型、方向和跨视图联动；完整 producer/consumer 因果解释属于 Future Work。

## 14. Local Memory 规范

| Buffer | Bytes | 位置 | 证据 |
| --- | ---: | --- | --- |
| A1 | 2048 | L1 @0 | confirmed |
| B1 | 4608 | L1 @2048 | confirmed |
| C1 | 64 | L1 @6656 | confirmed |
| C2 | 64 | Bias Table @0 | confirmed |
| A2 | 512 | L0A @0 | confirmed |
| B2 | 512 | L0B @0 | confirmed |
| CO1 | 1024 | L0C @0 | confirmed |

A1/B1/C1 共享 L1，地址不得重叠。A2/B2/C2/CO1 位于不同物理 Buffer，地址 0 不能被误读为相互覆盖。

## 15. 阶段与进度

这是唯一实施顺序：

| 阶段 | 状态 | 完成日期 |
| --- | --- | --- |
| 阶段 1：Workbench MVP | 已完成 | 2026-07-27 |
| 阶段 2：Tensor Code Recovery | 已完成；已按完整双源码刷新 | 2026-07-28 |
| 阶段 3：Hardware Participation | 下一阶段 | — |
| 阶段 4：Execution UX Patterns | 未开始；包含 Tensor Data Dump | — |
| 阶段 5：验证与交付 | 未开始 | — |

当前进度：**2/5**。

阶段 2 包含：

- Source 双 Tab；
- `sources/sourceRefs` schema；
- 两个文件的步骤标记；
- 新固定 tiling 和 block mapping；
- 精确 Buffer/bytes；
- 9 次 K 语义；
- Bias C1→C2；
- Fixpipe 直写 GM。

阶段 3 完成硬件参与路径和 Demo occupancy；阶段 4 完成通用 Instruction Visualization、Event 主载体、实现态 Tensor Lifecycle、Loop Group 和 Tensor Data Dump；阶段 5 完成交付回归。

## 16. 验收测试

### 16.1 数据

- JSON 可解析并符合 schema 0.3；
- 所有 `sourceRefs.fileId` 存在；
- 所有行号不超过对应源码总行数；
- `M/K/N=64/144/32`；
- `tile=16/16/16`；
- tile count=4/9/2；
- blockDim=8；
- K loop=9；
- Buffer bytes 与源码一致。

### 16.2 Source

- 两个 Tab 显示完整文件；
- Tab 标签不会挤压代码区；
- Host 和 Kernel 均存在可点击 Tag；
- Host `SetBlockDim` 与 Kernel `GetBlockIdx` 联动同一步；
- 4 种 HardEvent 均可从代码进入；
- 选中步骤时滚动到当前文件首个引用行。

### 16.3 Tensor

- Load3D 显示 feature window，不显示三维 Matmul；
- A2/B2 是二维 `[16,16]`；
- I0 有 Bias，I1～I8 无 Bias；
- I8 后 K progress=9/9；
- Fixpipe 输出 ND `[M,Co]`，不显示为 NCHW 物理布局；
- 用户能进入具体 Tensor / Tile，看到每个坐标的数值或明确的 `No dump data` 状态；
- 有 dump 数据时，数值与当前 instruction / 迭代 / buffer 上下文一致；
- 无 dump 数据时只显示 shape 与元数据，不生成或猜测矩阵内容；
- 数值格式与 dtype / layout 一致。

### 16.4 状态

- Source、Tensor、Hardware、Instructions、Playback 使用同一执行上下文；
- instruction 与 event dependency 均可成为执行上下文；
- Source Tab 是独立显示状态，不创建第二套执行上下文；
- Event 在 Dock、Tensor 和 Hardware 中职责一致；
- Iter 1～8 折叠或展开不丢失上下文；
- Timeline 保持 duration unavailable；
- Inspector 不存在。

### 16.5 可信度

- 不显示旧版“helper missing”“Bias path missing”“offset unknown”；
- 显示“未编译/未上板/无 profiling”；
- 不把 Demo capacity 称为真实 910B capacity；
- 不把固定 tiling 称为自动或最优 tiling。

## 17. 运行

必须通过本地 HTTP 服务访问：

```text
http://127.0.0.1:4180/pto_compute-graph-viewer/code%20recovery/index.html
```

直接打开 `file://.../index.html` 会导致 JSON 与源码 `fetch()` 失败。

## 18. 后续演进

### 当前五阶段计划

1. 阶段 3：将精确 Buffer bytes、执行单元和 Event 方向映射到 Hardware Participation；
2. 阶段 4：建立通用 Instruction Visualization、Event 主载体、实现态 Tensor Lifecycle、Tensor Data Dump，并展开 Iter 1～8 Loop Group；
3. 阶段 5：完成跨视图上下文、内容边界、响应式、可访问性和文档回归。

### Future Work

1. **Dependency-as-causality 深化**：点击 Event 后解释 producer、consumer、保护的 Tensor/Buffer、阻止的提前执行、潜在错配和串行化原因；
2. **Evidence-aware visualization 深化**：完整区分设计意图、代码确认、静态推断、估计、观测和 Unknown；
3. 接入目标 CANN 编译结果、设备 correctness/dump 或 profiling observation；
4. 只有获得明确时间证据后才生成 Estimated 或 Observed Timeline；
5. 增加 NHWC→NCHW 可选后处理步骤，而不是修改当前 Kernel 事实；
6. **Tensor Data Dump 数据源接入**：阶段 4 已定义具体 Tile 数值查看与无数据降级；接入真实 dump / correctness 数据后，再验证数值到 instruction、迭代、dtype、layout 和存储位置的完整映射。
