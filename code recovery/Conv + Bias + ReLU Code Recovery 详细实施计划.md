# Conv + Bias + ReLU Code Recovery 实施计划

> 当前版本：2026-08-05  
> 当前进度：阶段 2/5 已完成，阶段 3 待开始

## 1. 文档定位

本文档是 Code Recovery HTML Demo 的唯一实施顺序和进度记录。它服务于 UX 概念验证与体验 Pattern 提炼，不代表真实 Ascend C 工程建议或产品规划。

本 Demo 的目标不是证明可以自动理解所有 kernel，而是验证：通过 KEM 约束下的一组稳定体验范式，能否帮助用户从源码建立正确的 kernel execution mental model。

配套材料：

- 产品规范：`spec.md`
- 执行语义：`Conv2D + Bias + ReLU 执行过程.md`
- 跨产品设计框架：`../../Kernel Execution Model v0.1.md`

## 2. 体验目标与统一原则

用户应能沿着同一条理解路径完成判断：

```text
定位源码
→ 理解当前 instruction / event
→ 查看 Tensor 与 Memory 状态
→ 查看硬件参与
→ 理解前后逻辑关系
→ 回到源码继续阅读
```

### 2.1 统一执行上下文

Source、Tensor State、Hardware Participation 和 Execution Dock 共同解释同一个执行上下文。上下文可以是：

- 一个 instruction；
- 一个 event dependency；
- 一个循环或逻辑阶段。

切换任一区域时，其他区域保持同一上下文，不各自形成独立叙事。

### 2.2 当前采用的 KEM UX Pattern

- **Source-anchored explanation**：所有执行解释可回到源码；
- **Context-preserving synchronized views**：跨视图保持同一上下文；
- **Tensor lifecycle visualization**：区分地址存在、数据有效、消费与复用；
- **Loop compression with drill-down**：保留循环骨架和次数，按需展开；
- **Dependency-as-causality**：当前只实现 Event 位置、方向与联动，深度因果解释暂缓；
- **Evidence-aware visualization**：当前只保留基本边界声明，完整证据体系暂缓。

### 2.3 静态 Demo 边界

当前内容是固定参数的静态执行解释：

- 不代表已通过目标 CANN 编译；
- 不代表已在 910B/Atlas A2/A3 上运行；
- 不提供真实性能、duration、stall、overlap 或利用率；
- Timeline 当前保持 unavailable；
- Hardware capacity 和 occupancy 均为 Demo context；
- 不确定内容不得表现为真实运行结论。

## 3. Demo 基线

### 3.1 权威输入

1. Host Tiling：`src/conv_bias_relu_complete_demo/op_host/conv_bias_relu_tiling.cpp`
2. Device Kernel：`src/conv_bias_relu_complete_demo/op_kernel/conv_bias_relu_reference_complete.asc`
3. 执行解读：`Conv2D + Bias + ReLU 执行过程.md`

### 3.2 固定执行上下文

| 对象 | 当前 Demo |
| --- | --- |
| Feature | 逻辑 `[1,16,8,8]`；GM `NC1HWC0 [1,1,8,8,16]`；FP16 |
| Weight | 逻辑 `[32,16,3,3]`；GM `ND [144,32]`；FP16 |
| Bias | `ND [32]`；FP32 |
| Output | GM `ND [64,32]`；FP16；语义等价于 NHWC `[1,8,8,32]` |
| Cube | `A[64,144] × B[144,32] + Bias[32] → C[64,32]` |
| Tiling | `tileM/tileK/tileN = 16/16/16`；Tile 数 `4/9/2` |
| Core mapping | `OT=Mi×2+Nj`；`blockDim=8`；每核一个 `[16,16]` Output Tile |
| K loop | Iter 0～8，共 9 次；Iter 0 加 Bias，Iter 1～8 只累加 |
| Output | Fixpipe 从 CO1 直写 GM，完成 NZ→ND、FP32→FP16 和 ReLU |

这些值是当前参考源码和确定性计算所确认的 Demo 事实，不代表自动 tiling 或性能最优方案。

## 4. 已敲定的体验基础

本节只记录结论；已完成的页面细节不在计划中重复展开。

### 4.1 页面结构

```text
Source | Tensor State & Transformation | Hardware Participation
---------------------------------------------------------------
Instructions / Timeline(unavailable) / Playback
```

- Source 包含 Host / Kernel 双文件 Tab；
- Execution Dock 横跨页面底部；
- Terminal 与 Visualization Dock 互斥；
- Inspector 已移出范围；
- 页面结构继续复用 PTO IDE Frame。

### 4.2 Source 与执行联动

- Host 与 Kernel 均使用完整源码，不使用关键行摘录；
- 源码标记、播放和执行图共享同一执行上下文；
- 切换文件只改变可见源码，不改变当前上下文；
- 当前文件没有对应引用时不伪造高亮。

### 4.3 顶层执行叙事

| Stage | 内容 | 性质 |
| ---: | --- | --- |
| 1 | Input Shape | Host 配置 |
| 2 | Host Tiling | M/K/N、Tile 和 Tile 数 |
| 3 | Host 执行配置 | Buffer 元素数、blockDim、OT→Mi/Nj |
| 4 | Allocate Memory | 建立 GM 句柄和 LocalTensor 地址视图 |
| 5 | Copy Inputs | MTE2：GM→L1 |
| 6 | MTE2_MTE1 Sync | 发布 L1 输入 ready |
| 7 | Copy Data C2 | MTE1：Bias C1→C2 |
| 8 | K Loop · Iter 0～8 | Load、同步、Mmad 初始化与累加 |
| 9 | M_FIX Sync | 发布最终 Acc8 ready |
| 10 | Fixpipe Output | CO1→GM，格式/类型转换与 ReLU |

Host Shape、Tiling、Buffer 配置和 blockDim 不是 AI Core 数据搬运，不得画成 GM 数据流。

### 4.4 K Loop

```text
Iter 0 · Initialize
  Load A2/B2 → MTE1_M → Mmad + Bias → Acc0

Iter 1–8 · Accumulate ×8
  M_MTE1 → Load A2/B2 → MTE1_M → Mmad → Acc1…Acc8
```

规则：

- Iter 0 没有 `M_MTE1`；
- Iter 1～8 每轮先等待 A2/B2 可复用，再覆盖为下一 K Tile；
- Bias 只在 Iter 0 加入；
- 默认压缩显示 `×8`，展开后可定位具体 Iter；
- 每核共 9 次 Mmad，全 8 核共 72 次。

### 4.5 Allocate Memory

Allocate Memory 只回答 Tensor 位于哪里、地址范围和 Buffer 归属，不表达数据已经搬入。

| Buffer | Tensor 与真实地址 |
| --- | --- |
| L1 | `fmapA1 [0,2048)`、`weightB1 [2048,6656)`、`biasC1 [6656,6720)` |
| L0A | `fmapA2 [0,512)` |
| L0B | `weightB2 [0,512)` |
| Bias Table | `biasC2 [0,64)` |
| L0C | `accumCo1 [0,1024)` |

L1 中三个 Tensor 共用连续地址空间；L0A、L0B、Bias Table、L0C 相互独立，因此都可以从地址 0 开始。块宽使用统一可读比例，地址和字节数保持真实；Tensor 保留 Hover，不打开固定点击详情。

## 5. 阶段与进度

| 阶段 | 交付 | 状态 |
| --- | --- | --- |
| 1. Workbench MVP | 三栏工作台、全宽 Dock、播放、Terminal 互斥、移除 Inspector | **已完成 · 2026-07-27** |
| 2. Tensor Code Recovery | 双源码、新固定 tiling、精确 Local Memory、9 次 K 语义、Bias C1→C2、Fixpipe 直写 GM | **已完成 · 2026-07-28** |
| 3. Hardware Participation | 当前步骤的执行单元、存储节点、Demo occupancy 与 Event 方向 | **下一阶段** |
| 4. Execution UX Patterns | 通用 Instruction 图、Event 主载体、Tensor Lifecycle、Loop 展开、Tensor Data Dump（具体 Tile 数值查看） | **未开始** |
| 5. 验证与交付 | 跨视图一致性、状态边界、响应式、可访问性与文档回归 | **未开始** |

阶段 2 的“完成”只表示 Demo 的静态体验基线完成，不表示源码已经编译、运行或 profiling。

## 6. 后续实施

### 6.1 阶段 3：Hardware Participation

#### 体验目标

帮助用户把当前源码操作映射到参与的执行单元和存储位置，而不是展示一张与当前任务无关的完整芯片架构图。

#### 待办

1. 将当前 instruction 映射到 MTE2、MTE1、Cube、Fixpipe 和相关存储节点；
2. 存储节点显示当前使用字节数和 Demo capacity；
3. 执行单元只表达当前逻辑状态：
   - `Participating`
   - `Waiting on dependency`
   - `Not involved`
4. 不用 `Idle` 暗示真实硬件空闲，不给执行单元显示空间占用；
5. `MTE2_MTE1`、`M_MTE1`、`MTE1_M`、`M_FIX` 可在 Hardware 中显示依赖方向，但 Event 的主叙事仍在 Execution Dock；
6. Demo capacity 与真实芯片容量概念分层；
7. 未被当前代码使用或无法确认的硬件连接不高亮。

#### 完成标准

- 用户能看出当前 instruction 由哪个单元参与、访问哪个存储；
- Buffer occupancy 与执行单元状态不会混淆；
- Hardware 中的 Event 方向与 Dock 保持一致；
- 页面不会暗示真实利用率、stall 或硬件时序。

### 6.2 阶段 4：Execution UX Patterns

阶段 4 同时验证三个可复用 Pattern：Event execution context、Tensor Lifecycle 和 Instruction Visualization。它们共享执行上下文，但分别回答不同问题。

#### 6.2.1 Event 主载体与跨视图联动

##### 产品分层

```text
Recommendation Trace / Pipeline
  候选方案应该具有怎样的 Event dependency

Code Recovery / Execution Dock
  代码中的同步 instruction 位于什么逻辑位置

Observation
  未来回答是否真实等待、等待多久
```

##### Code Recovery 区域分工

| 区域 | Event 表达 |
| --- | --- |
| Source | 高亮同步代码 |
| Execution Dock | Event 主要载体；使用 Event lane 或位于相关 instruction 之间的 Event block |
| Tensor State | 显示 waiting、ready、reusable 等状态影响 |
| Hardware Participation | 显示涉及的执行单元和依赖方向 |

当前范围只实现 Event 的位置、类型、方向与跨视图联动，不展开完整因果诊断。

#### 6.2.2 实现执行层 Tensor Lifecycle

##### 与 Memory Map 的边界

| 视图 | 回答 |
| --- | --- |
| Memory Map | 放在哪里、地址是多少、哪些 Tensor 共用地址空间 |
| Tensor Lifecycle | 何时具有有效数据、何时被读取、何时可以复用、下一次承载哪个 Tile |

实现态生命周期使用“物理 Buffer 泳道 × 逻辑执行位置”：

```text
                 Allocate  Copy In  Load K0  Mmad K0  Load K1  Mmad K1  Fixpipe
L1  fmapA1       view ───── ready ─────────── consumed ────────────────────────
L0A fmapA2                          K0 ready ─ consume │ K1 overwrite ─ consume
L0B weightB2                        K0 ready ─ consume │ K1 overwrite ─ consume
L0C accumCo1                                  Acc0 ───── Acc1 ─────── Acc8 ─ consume
```

视觉规则：

1. 横轴命名为 `Logical execution order`，不叫 Timeline；
2. 逻辑位置等宽，不用长度表达真实 duration；
3. 纵向按物理 Buffer 分泳道；
4. 生命周期块表达逻辑有效区间，不画实际 Tensor 数据；
5. 同一地址被后续 Tile 复用时，显示连续版本块；
6. 复用关系只连接相同地址范围的前后对象；
7. 当前 instruction 使用垂直焦点线；
8. 地址范围和字节数保持真实；
9. Allocate Memory 只显示 `View created`，不提前显示 ready、K0 loaded 或 Acc0；
10. 无运行数据时不显示 duration、stall 或 overlap。

Memory Reuse Viewer 可作为视觉结构参考，但层次由数据来源决定：Demo tick 是概念演示，源码或编译材料对应静态实现恢复，只有仿真或 profiling 数据才能标为观测态。

#### 6.2.3 通用 Instruction Visualization Pattern

##### 目标与边界

对能够识别语义角色的 instruction，使用一套与算子类型无关的视觉语法生成执行图。无法识别的内容保留为 `Unknown / Source-only`，不隐藏、不猜测。

##### 通用语义

| 维度 | 示例 |
| --- | --- |
| Instruction | `DataCopy`、`LoadData3D`、`Mmad`、`Fixpipe` |
| Execution role | Host、Scalar、MTE2、MTE1、Cube、Vector、Fixpipe、Event |
| Action type | Configure、Allocate、Move、Transform、Compute、Sync、Store |
| Operands | `fmapA1 → fmapA2`、`A2/B2 → CO1` |
| Memory | GM、L1、L0A、L0B、Bias Table、L0C |
| Scope | Once、Per core、Per tile、Per iteration、Conditional |
| Structure | Sequence、Loop、Branch |
| Dependency | Data、Event、Control |
| Recognition | Recognized、Inferred、Unknown |
| Evidence | Source、Compiler/IR、API Registry、Runtime |

这些语义是 Demo 的设计输入，不代表真实解析器或工程数据结构。

Pattern 首期只要求 Instruction、Execution role、Action type、Structure、Dependency、Recognition，以及 Source 与 Hardware 联动。Tensor、Memory 语义在后续 Pattern 中下钻，不阻塞基本 Instruction Flow。

##### 视觉语法

1. **泳道**：按执行职责分组；未使用的空泳道隐藏；
2. **逻辑轴**：只表示顺序，节点宽度不表示耗时；
3. **Instruction block**：默认显示 instruction 和主要输入→输出，补充信息通过 Hover 或渐进披露；
4. **硬件联动**：选中 instruction 时高亮参与的执行单元；
5. **关系**：区分 Data、Event、Control，不共用同一种视觉编码；
6. **循环**：统一表达 Initialize、Repeated body ×N、Tail/Finalize；
7. **分支**：显示共同条件和分支路径，不冒充真实已执行路径；
8. **Event**：不画成普通计算；紧凑模式使用依赖线，展开模式可显示 signal/wait block；
9. **Unknown**：保留源码位置和 unresolved 状态，不推断硬件或数据流。

##### 自动组织规则

```text
区分 Host / Kernel
→ 识别 Loop / Branch
→ 保持结构内逻辑顺序
→ 按 execution role 放入泳道
→ 连接明确数据关系
→ 添加 Event dependency
→ 压缩重复循环
→ 保留 Unknown
→ 默认聚焦主执行路径
```

通用性来自稳定的视觉语法、信息层级和 Unknown 降级规则，不要求不同算子生成相同形状的图。

##### Source 与 Hardware 联动

- 点击源码聚焦对应 instruction，点击 instruction 返回对应源码；
- 一个 instruction 可以关联多处源码；
- 切换 Host / Kernel 不丢失当前执行上下文；
- 当前文件没有对应引用时不伪造高亮；
- Loop 折叠或展开后保持 Source 与 Hardware 上下文；
- 无法确认执行角色时不高亮硬件单元。

##### 识别与 AI 边界

确定性识别优先使用 AST / CFG / Compiler IR、API 语义注册表、类型、符号和定义—使用关系。大模型只辅助解释陌生封装、复杂函数和未登记 API，其结果必须标为 `Inferred`，不能覆盖确定性事实。

| 状态 | 含义 |
| --- | --- |
| Recognized | 有源码、编译材料或已登记 API 语义支持 |
| Inferred | 根据上下文形成的候选解释，并保留依据 |
| Unknown | 当前证据不足，只显示源码位置 |

##### Pattern 首期验收

- 能从源码进入对应 instruction，并从 instruction 返回源码；
- 能看出主要顺序、循环和依赖；
- 能识别当前参与的硬件单元；
- Unknown 不被隐藏或包装成事实；
- 逻辑轴不暗示真实耗时；
- 不依赖 Tensor、Memory 视图也能完成基本执行理解。

#### 6.2.4 Loop Group

- Iter 0 与 Iter 1～8 始终分开；
- Iter 1～8 默认显示 `×8`；
- 展开后复用同一 Instruction Visualization Pattern；
- 展开或折叠不改变当前 Source、Tensor、Hardware 上下文；
- 不用横向长度暗示各 Iter 的真实耗时。

#### 6.2.5 Timeline

当前范围内 Timeline 始终显示 unavailable。只有未来存在明确来源和适用范围的估算或 profiling 数据时才重新评估：

- 估算必须标记 `Estimated Timeline · Not Profiling Data`；
- profiling 才能使用真实时间尺度；
- 仅出现 `startTime/duration` fixture 字段不足以启用 Timeline。

#### 6.2.6 Tensor Data Dump

##### 用户需求

用户需要从抽象 shape 下钻到具体数据：例如一个 `16×16` 的 Output Tile，能看到每个坐标分别是什么数值，而不只是看到 Tensor 的存在、地址或生命周期。

##### 定义与边界

- Data Dump 在 Tensor 视图中展示具体 Tile 的实际矩阵数据；
- 数据必须关联具体运行、Instruction、Tile、dtype、layout 和存储位置；
- 实际数值只能来自 Data Dump 或其他明确的运行数据；
- 没有 dump 数据时只显示 shape 与元数据，不生成或猜测矩阵内容；
- 展示内容随当前执行上下文联动，例如当前 Iter 的 A2/B2、当前 Acc 或 Fixpipe 输出 Tile。

##### 体验要求

- 使用坐标网格表达 Tile（如 `16×16`），每个单元格显示对应数值；
- 数值格式必须匹配 dtype（FP16 / FP32 等）与 layout；
- 与 Instruction、Tensor State 和 Hardware Participation 保持同一执行上下文；
- 无数据状态明确显示 `No dump data` 或等价表述，不伪造数值。

##### 阶段 4 验收

- 用户能进入具体 Tensor / Tile，看到每个坐标的数值或明确的“无 dump 数据”状态；
- 数值与当前 instruction / 迭代 / buffer 上下文一致；
- 无 dump 数据时页面不生成或猜测矩阵内容；
- 数值展示不暗示真实性能或运行时长。

### 6.3 阶段 5：验证与交付

#### 功能与一致性

- Host / Kernel 源码引用有效；
- Source、Tensor、Hardware、Dock 保持同一执行上下文；
- instruction 与 event dependency 均可成为上下文；
- Iter 1～8 折叠和展开不丢失上下文；
- Event 在 Pipeline、Dock、Tensor 和 Hardware 中职责一致；
- Memory Map 不显示数据已经 ready；
- Tensor Lifecycle 不暗示真实时间；
- Timeline 保持 unavailable。

#### 内容正确性

- `M/K/N=64/144/32`，Tile 为 `16/16/16`，Tile 数为 `4/9/2`；
- Iter 0 有 Bias，Iter 1～8 无 Bias；
- 每核 9 次、全 8 核 72 次 Mmad；
- A2/B2 的 K Tile 只在相应 Load 后更新；
- Fixpipe 显示 NZ→ND、FP32→FP16、ReLU 和 GM 写回；
- Local Memory 地址统一使用 `[start,end)`。

#### 体验与视觉

- 窄屏不出现整页水平溢出；
- 长文件名、滚动定位、Hover 和键盘焦点可用；
- 小 Tensor 块保持可见；
- 逻辑依赖、数据搬运和控制关系不会使用同一种视觉编码；
- 使用 PTO 共享 token、component 和 pattern，不新增私有视觉体系；
- 错误、空状态与本地 HTTP 启动提示完整。

## 7. Future Work

以下内容已记录，但不阻塞当前五阶段计划：

1. **Dependency-as-causality 深化**  
   点击 Event 后解释 producer、consumer、保护的 Tensor/Buffer、阻止的提前执行、潜在错配和串行化原因。

2. **Evidence-aware visualization 深化**  
   系统区分代码确认、静态推断、设计估计、运行观测和 Unknown，并支持 Recommendation / Recovery / Observation 对照。

3. **Observation overlay**  
   接入仿真或 profiling 后，再表达真实 duration、等待、stall、overlap 和运行时 Tensor residency。

4. **Tensor Data Dump 数据源接入**  
   阶段 4 已定义具体 Tile 数值查看与无数据降级；接入真实 dump / correctness 数据后，再验证数值到 instruction、迭代、dtype、layout 和存储位置的完整映射，以及格式 / layout 转换后的数值一致性。

## 8. 启动

页面通过 `fetch()` 加载 JSON 和源码，必须使用本地 HTTP 服务，不能直接以 `file://` 打开。

从工作区根目录启动服务后访问：

```text
http://127.0.0.1:4180/pto_compute-graph-viewer/code%20recovery/index.html
```
