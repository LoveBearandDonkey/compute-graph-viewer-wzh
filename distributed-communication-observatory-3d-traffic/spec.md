# Distributed Communication Observatory 产品与体验规格

> 状态：Concept Spec / 尚未进入页面实现  
> 版本：v0.2 · 2026-08-03  
> 项目目录：`Design/distributed-communication-observatory/`  
> 核心原则：建立面向模型训推任务的统一通信证据工作台；不同并行策略使用不同诊断 Lens，不能被强行塞进同一种图，也不能让通用平台 Chrome 淹没当前任务主体。

---

## 1. 产品判断

### 1.1 一句话定位

> 当模型训推任务在多设备扩展后出现性能、容量或稳定性异常时，系统把模型阶段、并行组、Collective、Rank、执行等待和物理拓扑串成可追溯证据，并通过与当前并行策略匹配的诊断 Lens，帮助用户在 3–5 分钟内判断通信是否进入关键路径、为什么发生以及下一步如何验证。

### 1.2 为什么不能做成 MoE 专用产品

通信可视化不是为了展示“数据在流动”，而是为了回答：模型扩到更多设备后为什么没有按预期变快，以及应该调整并行策略、数据拆分、Rank Placement、Collective、Overlap、Buffer 还是 Runtime。

MoE 只是首个最适合验证 Sankey 和容量诊断的 Lens，因为其 Dispatch / Combine 同时具备：

- 输入相关、随 Batch / Request 变化的动态路由；
- 多 Source 到多 Expert 的流量守恒关系；
- Expert 容量、Token Drop、负载倾斜和 Tail Latency 风险；
- EP AllToAll 与跨 Node 拓扑的直接关系；
- 可以被 Sankey 清楚表达的“分流—汇流”结构。

DP、TP、PP、CP 的通信同样重要。产品必须共享证据底座，但按任务提供不同 Lens，而不是强行统一成一张 Sankey：

| 并行方式 | 首要问题 | 更适合的主视图 | Observatory Lens |
|---|---|---|---|
| EP / MoE | Token 如何分配、哪个 Expert / Rank 过热 | 分层 Sankey + Expert Capacity | `MoE Route & Capacity` · 首个交付 |
| DP / FSDP | 梯度 / 参数通信是否被计算覆盖、哪个 Rank 最晚 | Timeline + Rank Matrix | `Gradient Sync` |
| TP / SP | 每层 Collective 是否延迟敏感、TP Group 是否跨慢链路 | Layer Timeline + Topology | `Layer Collective` |
| PP | Stage Bubble、Microbatch 和负载平衡 | Pipeline Swimlane | `Pipeline Transfer` |
| CP | 长序列 Activation / KV 分片及通信 | Timeline + Topology | `Context Transfer` |
| 跨策略 Runtime | Collective 配对、Wait Chain、拓扑与 Hang | Timeline + Dependency + Topology | `Runtime & Topology` · 横向证据 Lens |

成熟系统同样把 DP、TP、PP、CP、EP 视为不同但可组合的并行维度，而不是同一种通信问题：[Megatron Core Parallelism Strategies](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)。MoE Router、Load Balancing 与 AllToAll Token Dispatch 的工程关系见 [Megatron Core MoE Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html) 和 [DeepSpeed-MoE](https://arxiv.org/abs/2201.05596)。

### 1.3 可扩展产品架构

```text
Distributed Communication Observatory
├── Shared Context
│   └── Run / Step / Request / Phase / Baseline
├── Shared Communication Evidence Plane
│   └── ParallelGroup / Rank / Collective / Event / Wait / Topology / Metric
├── Shared Interaction State
│   └── Window / Selection / Filter / Compare / Evidence completeness
└── Task-specific Lenses
    ├── MoE Route & Capacity
    ├── Gradient Sync
    ├── Layer Collective
    ├── Pipeline Transfer
    ├── Context Transfer
    └── Runtime & Topology
```

每个 Lens 必须提供相同的产品契约，但可以使用完全不同的主体可视化：

```text
LensContract
├── trigger                  用户带着什么症状进入
├── primaryQuestion          当前只回答哪个核心问题
├── requiredArtifacts        没有这些证据就不能给出什么结论
├── primaryVisualization     Sankey / Matrix / Swimlane / Topology
├── diagnostics[]            事实、推导、备选解释
├── actions[]                最小候选动作
└── validationGates[]        怎样证明改善且没有破坏正确性
```

共享底座不决定页面长相。用户进入某个 Lens 后，只挂载该任务需要的主视觉和证据；其他 Lens 不以 Tab 墙、缩略图或常驻面板占据界面。

### 1.4 产品价值阶梯

任何 Lens 的可视化都只能先完成描述，产品必须继续走到诊断、决策与验证。以下以首个 MoE Lens 为例：

```text
描述：发生了什么？
  R2 接收了 44% 的 Payload Bytes
        ↓
诊断：为什么值得关注？
  R2 上的 E32/E33 负载为均值 2.1×，造成 4.8 ms 暴露等待
        ↓
决策：应该做什么？
  比较 Router Balance、Expert Placement 与 EP Group 映射
        ↓
验证：改动是否有效？
  Exposed Comm、Max/Mean、Buffer Headroom 和 Step / TPOT 门禁通过
```

如果页面只能让用户复述 Ribbon 大小，不能判断原因或选择动作，本产品假设失败。

---

## 2. 目标用户、触发场景与核心决策

### 2.1 用户层级

| 用户 | 首要问题 | 默认可见层级 |
|---|---|---|
| 分布式 / 性能工程师（Primary） | 当前并行策略的通信是否进入关键路径，瓶颈来自哪里 | L1 结论 + 当前 Lens 的 L2 主视图 / Timeline |
| 模型 / 算法工程师 | 数据拆分、Router、Expert 或长上下文策略是否造成负载与质量风险 | 模型语义 → Parallel Group → Rank |
| Runtime / DFX 工程师 | Collective、Barrier、Stream、拓扑为何等待或 Hang | Runtime Event、Wait Chain、Fabric Path |
| Kernel / Codegen 工程师 | Collective 前后处理或设备通信任务运行在哪类任务与引擎 | Task / Stream → AIC / AIV / Core |

门外汉可以从 L1 理解“发生了什么、影响什么、下一步是什么”，但界面不通过简化术语隐藏工程事实。专家可以固定链接直达 L2 / L3。

### 2.2 触发事件

- DP Gradient Sync、TP Layer Collective、PP Transfer、CP Context Transfer 或 EP Dispatch / Combine 相对 Baseline 明显回退；
- Step Time、TTFT、TPOT 或 Throughput 未达到并行扩展预期；
- 某 Rank / Expert 的 Token 数、Bytes 或计算时间显著高于均值；
- 某个 Pipeline Stage 出现 Bubble，或某个 Collective 长期等候最慢 Rank；
- Receive Buffer 接近或超过安全容量；
- Collective 超时、Barrier 长期等待或运行 Hang；
- Parallel Group 或 Rank Placement 从片内高速域跨到 RoCE；
- 配置、模型版本或输入变化后出现通信结构回归。

### 2.3 核心决策

用户最终要判断：

1. 通信是否真的位于当前 Step / Request 的关键路径；
2. 异常首先属于 DP、TP/SP、PP、CP、EP/MoE 还是跨策略 Runtime；
3. 在对应 Lens 中，问题主要属于哪一类：
   - Router / Expert Load；
   - Gradient / Parameter Sync；
   - Layer Collective 粒度与频率；
   - Pipeline Stage / Microbatch；
   - Context / KV Transfer；
   - Expert Placement / Rank Mapping；
   - Collective 算法、粒度或精度；
   - Communication–Compute Overlap；
   - Buffer Capacity；
   - Runtime / Barrier / Fabric；
   - 设备侧 Pack / Unpack / Dispatch / Combine；
4. 最小候选动作是什么；
5. 应运行什么验证，怎样算通过。

### 2.4 成功指标

- Time to first conclusion：≤ 30 秒；
- Time to decisive evidence：≤ 3 分钟；
- 3–5 分钟内完成“结论 → 证据 → 动作 → 验证计划”；
- 用户能区分并行策略、逻辑通信量、物理链路流量和关键路径等待；
- 不把 Bytes 大自动解释为性能根因；
- 不把缺失的 Task / Core 关联猜测为 AIC/AIV 归因。

---

## 3. 问题证据与仓库依据

| 真实问题 | 用户影响 | 仓库证据 | 对产品的约束 |
|---|---|---|---|
| DeepSeek-V3.2 使用 DP2+TP8+EP16，EP 跨 16 卡并涉及 HCCS / RoCE | 多种并行组与物理拓扑叠加，单看 Rank ID 无法判断代价 | [pypto-lib #156](../../github_issues/pypto-lib/pypto-lib_issues.md) | 必须显示 Parallel Group、Node 与 Fabric 归属 |
| 平均负载公式不能保证 Receive Buffer 上界；倾斜时单 Expert 可能接收所有 Token | 可能导致容量浪费、溢出或错误安全感 | [pypto-lib #456](../../github_issues/pypto-lib/pypto-lib_issues.md) | Expert 节点必须显示 Capacity、Headroom 与最坏情况边界 |
| 数据相关 Expert Load Drift 与特定 Die / Card 拓扑组合触发 Hang | 均匀流量测试正常也不能排除真实运行问题 | [pypto-lib #502](../../github_issues/pypto-lib/pypto-lib_issues.md) | Flow、Timeline、Wait Chain 与 Topology 必须联动，不能只看流量 |
| 8 Rank 路由超时缺少通信上下文，外层 Harness 只能看到整体卡住 | 无法区分 Context、Lowering、Kernel 与 Runtime | [PTOAS #254](../../github_issues/PTOAS/PTOAS_issues.md) | L3 必须保留 Collective / Comm Context / Runtime Event 证据 |
| 产品规划要求 TP/EP/MoE 通信、跨 Rank 校验与通信—计算重叠可视化 | 方向属于 PyPTO 3.0 的明确产品边界 | [产品规划 §5.7](../../Product_Planning/PyPTO3.0_Toolkit_产品功能规划.md) | 功能以验证和归因为目标，不做纯流量展示器 |

当前证据可证明问题真实存在，但不能证明仓库已经拥有完整的 Token→Expert→Task→Core 统一 Artifact。实现阶段必须先审计真实产物；缺失字段按 `missing` 处理。

当前仓库证据对 EP/MoE 最完整，因此首个 Lens 从 MoE 开始。Gradient Sync、Layer Collective、Pipeline Transfer 和 Context Transfer 在进入实现前，必须分别补齐 Issue、Artifact、目标决策与度量结果；不能只因为平台名称变大就预造空功能。

---

## 4. 五层领域模型

界面中的每个对象必须属于以下层级之一。层级可以联动，但不能用一条 Ribbon 混写不同关系。

### 4.1 模型任务层：用户为什么看到这次通信

```text
Training Step / Inference Request
└── Prefill / Decode / Forward / Backward
    └── Transformer Layer
        └── MoE Layer
            ├── Router
            ├── Dispatch
            ├── Expert Compute
            └── Combine
```

回答：这是哪个 Run、Step / Request、模型阶段、Layer 和 Phase？

### 4.2 并行与路由层：逻辑工作如何被拆分

```text
Parallel Plan
├── DP Group
├── TP / SP Group
├── PP Stage
├── CP Group
└── EP Group
    └── Rank
        └── Expert / Tensor Shard
            └── Token Group
```

回答：谁把哪些 Token 发给哪个 Expert，Expert 被放在哪个 Rank / Node？

### 4.3 Collective 与 Runtime 层：通信怎样执行

```text
Collective
├── Dispatch AllToAll(v)
├── Combine AllToAll(v)
├── Send / Recv Event
├── Barrier / Signal / Fence
└── Stream / Runtime Task
```

回答：哪次 Collective、哪条 Event、哪一侧先到或在等待什么？

### 4.4 物理拓扑与执行层：通信落到哪里

```text
Host / Cluster
└── Node
    └── Device / Die
        ├── Fabric Path       LOCAL / HCCS / RoCE
        ├── Communication / DMA path
        └── Compute Task
            ├── AIC
            ├── AIV
            └── Core
```

回答：Rank Placement 是否匹配高速域，Pack / Dispatch / Combine 任务运行在哪类引擎？

`Rank → AIC/AIV` 是任务归因和容器下钻，不是数据真的先从 Rank 传输到 AIC/AIV。真实网络搬运还可能涉及 Runtime、通信 Kernel、DMA 或其他通信路径；没有 Artifact 时不得推断。

### 4.5 时间与证据层：结论发生在什么时候、依据是什么

```text
Profile Window
├── Event timestamp / duration
├── Step / Microbatch / Request
├── Baseline / Current / Candidate
├── Metric / threshold / noise
└── Evidence Reference
    ├── real
    ├── derived
    ├── mock
    └── missing
```

回答：结论使用哪个窗口、怎样聚合、与什么比较、是否完整可信？

### 4.6 推荐下钻顺序

```text
Run / Step / Request
  → MoE Layer / Phase
  → EP Group / Rank
  → Expert / Token Group
  → Collective / Event
  → Task / Stream
  → Fabric 或 AIC / AIV / Core
```

不默认从 Rank 直接跳到 Core，因为这会跳过模型开发者最需要的 Layer、Expert 和 Collective 语义。

---

## 5. 四种“流量”口径

一条 Ribbon 在任一时刻只能编码一种宽度口径：

| 口径 | 含义 | 主要决策 | 数据要求 |
|---|---|---|---|
| Token Count | Router 分配给 Expert 的 Token 数 | 判断路由和 Expert Balance | Router / Dispatch metadata |
| Logical Payload Bytes | Source Rank → Destination Rank 的逻辑 Payload | 判断通信规模和容量 | Comm event bytes |
| Physical Link Bytes | Collective 算法在 HCCS / RoCE 等链路产生的真实流量 | 判断链路压力和拓扑代价 | Fabric counters / algorithm trace |
| Exposed Communication Time | 未被计算覆盖、实际进入关键路径的通信时间 | 判断性能根因和优化优先级 | Timeline、依赖、overlap / wait |

默认口径由当前 Lens 决定，而不是全产品固定：MoE Route & Capacity 默认 `Token Count`；Gradient Sync / Layer Collective 更可能默认 `Logical Payload Bytes` 或 `Exposed Communication Time`；Pipeline Transfer 默认以 Stage Event / Bubble 为主，不强制使用 Ribbon。用户只有在存在真实 Artifact 时才能切换到 Physical Bytes。

界面必须持续显示当前口径，不能用 Tooltip 才说明。禁止把 Token、Logical Bytes、Physical Bytes 与 Duration 混成一个“Traffic Weight”。

### 5.1 关键派生指标

```text
expert_load_ratio      = expert_tokens / mean_expert_tokens
rank_receive_ratio     = rank_received_payload / mean_rank_received_payload
buffer_headroom        = capacity_rows - received_rows
exposed_comm_time      = critical_path_comm_time - hidden_overlap_time
overlap_ratio          = overlapped_comm_time / total_comm_time
cross_node_ratio       = roce_payload / total_payload
```

派生指标必须显示公式、窗口和 Artifact；没有依赖图时不能声称精确的 `exposed_comm_time`。

---

## 6. 共享平台能力与首个 Lens

### 6.1 Observatory 共享能力

共享层只提供所有通信诊断都会使用的能力，不提供一个“万能通信大盘”：

| 共享能力 | 产品职责 | 不负责 |
|---|---|---|
| Context Resolver | 统一 Run、Step / Request、Phase、并行配置与 Baseline | 不决定哪个图是主体 |
| Communication Evidence Graph | 关联 Parallel Group、Rank、Collective、Event、Wait、Topology 和 Artifact | 不用猜测补齐缺失关系 |
| Critical Path & Overlap | 计算或解释暴露通信、等待与重叠 | 不把 Duration 自动当成性能损失 |
| Baseline Contract | 检查环境、输入、规模、硬件和采集条件是否可比 | 条件不一致时不输出伪精确百分比 |
| Shared Timeline Window | 为所有 Lens 提供真实时间范围与事件选择 | 不要求所有 Lens 都显示 Playback |
| Evidence & Provenance | 统一 real / derived / mock / missing 和证据完整性 | 不把建议包装成事实 |
| Lens Router | 根据异常 Collective、模型阶段或用户症状进入合适 Lens | 不同时铺开所有 Lens |

Lens Router 的默认行为：

```text
AllToAll(v) + MoE Layer       → MoE Route & Capacity
AllReduce / ReduceScatter     → Gradient Sync 或 Layer Collective
AllGather + Layer Activation  → Layer Collective
P2P + Pipeline Stage          → Pipeline Transfer
Context / KV communication    → Context Transfer
Barrier / Timeout / Hang      → Runtime & Topology
```

只有已经实现并拥有足够 Artifact 的 Lens 才出现在产品中。首个版本只有 MoE Lens 时，不显示一个只有单选项的 Lens 切换器。

### 6.2 首个交付：MoE Route & Capacity Lens

选择 MoE 作为首个交付是证据和视觉任务的优先级，不是产品命名或数据模型的边界。

### 6.3 MoE Lens 功能优先级

| 优先级 | 功能 | 回答的问题 | MVP 判断 |
|---|---|---|---|
| P0 | Run / Step / Request 与 Baseline 上下文 | 当前分析对象和比较条件是什么 | 必须 |
| P0 | L1 Communication Diagnosis | 发生了什么、影响什么、先看哪里 | 必须 |
| P0 | MoE Phase Navigator | Router / Dispatch / Compute / Combine 哪个阶段异常 | 必须 |
| P0 | Hierarchical Route Sankey | Rank 间 Token 流量如何归因到 AIC/AIV/Unattributed 任务端点 | 必须，新 Pattern Preview |
| P0 | Expert Capacity Evidence | 哪些 Expert 接近容量、溢出或导致 Padding 浪费 | 必须，由 Inspector 与 Timeline 承载，不替换 Sankey 的引擎层级 |
| P0 | Profile Window + Event Timeline | 异常发生在哪段时间，底层事件是什么 | 必须 |
| P0 | Contextual Inspector | 为什么这样判断，证据完整吗 | 必须；默认承载 L1，选择后原位切换证据 |
| P0 | Unattributed / Missing 边界 | 哪些 Bytes 无法关联到 Expert、Task 或 Core | 必须 |
| P1 | Current / Baseline Difference Mode | 流量结构和关键路径怎样变化 | 下一轮 |
| P1 | Topology Evidence | 是否因 Rank Placement 跨越慢链路或异常 Die / Card | 下一轮，L3 按需打开 |
| P1 | Candidate Experiment | 调整 Capacity、Placement、EP Size 或 Overlap 后验证 | 下一轮 |
| P2 | Task → AIC/AIV/Core Attribution | 设备侧前后处理为何慢 | 专家下钻，不进入默认主视图 |

### 6.4 L1 诊断结构

Inspector 首屏只显示 1 条首要结论和必要证据，不展示 KPI 卡片墙。

```text
[HIGH] MoE Layer 27 Dispatch 出现 Expert Load Skew
影响：R2 上的 E32/E33 为均值 2.1×，造成 4.8 ms 暴露等待；Buffer 余量 12%
依据：Router counts + Dispatch trace + Rank wait chain（3/3 完整）
下一步：查看 E32/E33 的 Token 来源与 Rank Placement
```

结论必须分别标注：

- 事实：Token Count、Bytes、Duration、Wait；
- 推导：Skew、Exposed Time、容量风险；
- 建议：候选动作；
- 缺失：无法取得的 Router、Fabric 或 Core 证据。

### 6.5 Hierarchical Route Sankey

#### 默认结构

Dispatch 模式先保持 Rank 间通信结构，展开后把端点归因到执行引擎：

```text
Source Node / Rank / AIC-AIV-Unattributed
                ───────────────→ Destination Node / Rank / AIC-AIV-Unattributed
```

Rank、Node 是复合容器，不通过额外 Ribbon 表示包含关系。折叠时 Ribbon 连接 Rank；展开时端点重新挂到 AIC、AIV 或 Unattributed。Expert ID、Capacity 与 Token 来源作为同一 Event 的模型语义字段进入 Inspector 和 Timeline，不占用 Rank 的结构子节点。

Combine 模式反向表达：

```text
Destination Rank / AIC-AIV-Unattributed
                ───────────────→ Original Rank / AIC-AIV-Unattributed
```

#### 默认展开策略

- 首屏只展开系统诊断出的 Hot Rank，并显示 AIC/AIV/Unattributed 归因；
- 其他 Node / Rank 聚合显示，避免让用户先处理规模；
- 展开路径固定显示在图上方，例如 `EP16 / Node B / Rank 2 / AIC–AIV`；
- 高规模时优先按 Node → Rank → Engine 聚合，不抽样隐藏异常；
- 聚合或裁剪必须显示对象数量与统计范围。

#### 视觉编码

| 属性 | 编码 | 约束 |
|---|---|---|
| 当前流量口径 | Ribbon 宽度 | 同屏只允许一种口径 |
| Source Group | Ribbon 色相 | 使用设计系统 Viz Highlight Ramp，不使用状态色 |
| Hot / Overflow | 节点状态 + 文本 / 图标 | `danger` 只表达异常，不作为普通系列色 |
| Expert Capacity | Inspector 内的 Load / Capacity 证据 | 不混入 AIC/AIV 结构节点，不增加独立仪表盘 |
| Cross-node | Ribbon 线型 / Fabric 标签 | 不能只靠橙色区分 |
| Unattributed | 灰阶 + 虚线 + 明确文字 | 不用透明到无法点击 |
| Baseline Diff | P1 独立 Difference Mode | 默认视图不叠加两套 Ribbon 造成噪声 |

Hover 只预览：Source、Target、Token / Bytes、Events、Fabric、Window、Provenance。Click 才形成持久选择并联动 Timeline 与 Inspector。

### 6.6 Expert Capacity

Capacity 不做独立卡片墙，直接成为 Expert 节点的一部分：

```text
E32   842 / 896 rows   94% capacity
      ├──────────────┤
      load         limit
```

必须区分：

- 当前接收量；
- Runtime 实际容量；
- Router / 配置的 Capacity Factor；
- 理论最坏情况；
- Token Drop / Pad / Packed Layout；
- `missing capacity`，不能用 0 代替。

### 6.7 Timeline 与窗口

Timeline 负责时间、顺序与等待；Sankey 负责窗口聚合，两者不互相替代。Timeline 使用通用 `hierarchical-timeline` Pattern，层级由 Lens 数据配置，不继承任何特定训练任务的信息架构。

MoE Lens 推荐层级：

```text
Layer
├── Router
├── Dispatch
│   └── Collective
│       └── Rank
│           └── Task
├── Expert Compute
│   └── Expert
├── Rank Wait / Barrier
└── Combine
    └── Collective
        └── Rank
            └── Task
```

- Pattern 必须支持固定行头、层级展开、Fit 与 1×–64× 横向缩放；缩放只改变时间尺度，不改变事件语义；
- 底部 Dock 初始保持紧凑，但高度由 `workbench-shell` 拖拽调整；Pattern 必须在任意合法高度内滚动和重新布局；
- Profile Window Handle 改变聚合窗口，Sankey 重新聚合；
- 点击 Ribbon，Timeline 只强调组成该 Ribbon 的底层事件；
- 点击 Wait，Sankey 强调造成该等待的 Rank / Expert 路径；
- Sankey 与 Timeline 通过 `transactionId / collectiveId / eventId / taskId / rankId / expertId` 建立证据关联；
- 默认不显示 Playback；只有用户需要理解跨 Step / Microbatch 状态传播时，才接入共享 `floating-playback-control`；
- 禁止沿 Ribbon 播放粒子模拟数据包。

### 6.8 Contextual Inspector

Inspector 默认显示当前 Phase 的 L1 Diagnosis。选中 Expert、Ribbon、Collective 或 Wait 后，Inspector 原位切换为该对象的证据；用户可以从右上角关闭 Inspector，让 Sankey 占满主体。

内容固定为与当前选择有关的四段：

1. `Why it matters`：对 Step / TTFT / TPOT / Buffer / Stability 的影响；
2. `Evidence`：事实、推导公式、Artifact、完整性；
3. `Alternative explanations`：仍可能的混杂因素；
4. `Next action`：最小实验与验证门禁。

不显示与选择无关的全部 JSON、环境字段、通用帮助或操作历史。原始数据通过 L3 链接打开。

### 6.9 Candidate Experiment（P1）

支持的候选动作：

- Router Load Balance / Capacity Factor；
- Expert Placement / Rank Mapping；
- EP Group Size 或 Parallel Folding；
- Dispatch / Combine 精度与实现；
- Communication–Compute Overlap；
- Runtime Buffer Sizing / Packed Layout；
- 拓扑相关 Hang 的安全 Placement Workaround。

候选只生成实验配置和验证计划，不直接宣称结果。验证至少比较：

- Expert Max/Mean；
- Buffer Headroom / Drop；
- Logical / Physical Bytes；
- Exposed Communication Time；
- Step Time 或 TTFT / TPOT；
- Correctness / Hang Gate；
- Baseline 与 Candidate 的环境可比性。

---

## 7. 3–5 分钟黄金任务流

| 步骤 | 用户问题 | 界面主体 | 用户动作 | 完成信号 |
|---|---|---|---|---|
| 1. 获得结论 | 当前 MoE 通信是否真的影响性能或稳定性 | L1 Diagnosis | 接受首要异常，确认 Run / Window / Evidence | 能说出异常、影响和证据完整性 |
| 2. 定位模型阶段 | 哪个 Layer / Phase 发生问题 | Phase Navigator + Timeline | 选择 Layer 27 / Dispatch | 主体只展示相关事件和路由 |
| 3. 识别倾斜对象 | 哪个 Rank / 引擎端点承载集中，关联到哪个 Expert | Hierarchical Sankey + Inspector | 展开 R2，选择 AIC/AIV 端点，再核对 E32/E33 证据 | 能区分通信结构、引擎归因和 Expert 容量三种语义 |
| 4. 核对因果证据 | 大流量是否进入关键路径，是否还有其他解释 | Timeline + Evidence Dock | 查看 Wait、Overlap、Topology / Runtime 证据 | 区分 Router、Placement、Overlap 或 Runtime |
| 5. 形成验证计划 | 最小动作是什么，怎样证明有效 | Next Action | 创建候选实验或打开验证命令 | 得到可执行动作和明确通过门槛 |

主流程不要求用户先浏览所有 Rank、打开硬件架构图或理解 Core 级细节；AIC/AIV 只作为 Rank 的一级任务归因标签，并提供 Inspector 解释。

---

## 8. 整体界面规格

### 8.1 1440 × 900 主布局

```text
┌──────────────────────────────────────────────────────────────────────┐
│ IDE Top Bar：Product / Theme / Language / Run Info / Dock Controls  │
├──────────────────────────────────────────────────────────────────────┤
│ Hierarchical Route Attribution                                      │
│             Router · Dispatch · Expert Compute · Combine             │
│                                                                      │
│              MoE Hierarchical Route Sankey       │ Inspector         │
│                                                   │ L1 Diagnosis      │
│                                                   │ Evidence / Action │
├──────────────────────────────────────────────────────────────────────┤
│ PROFILE TIMELINE：Event / Wait + Window Brush                        │
└──────────────────────────────────────────────────────────────────────┘
```

比例约束：

- 当前 Lens 的主视觉在 Inspector 关闭时占内容区 100% 宽度；
- Run Context 通过右上角 Info 弹层按需查看，不形成常驻侧栏；
- Dock 打开后建议主 / 辅约为 72 / 28，可拖拽但不能小于主图可读阈值；
- Timeline 初始高度约占工作区 18–24%，但必须允许向上拖拽扩展并支持收起；缩放和定位属于 Timeline Pattern，不以初始 Dock 高度为由删除；
- L1 Diagnosis 位于 Inspector 顶部，仅容纳 1 条主结论，不演变为 Dashboard Header；
- 页面任一时刻只有一个主焦点：Diagnosis、选中 Expert / Ribbon 或 Wait。

### 8.2 IDE 顶栏与 Run Info

只保留会改变诊断语义的上下文：

- Active Task / Lens：只有两个以上已实现 Lens 时才显示切换；
- Run；
- Training Step / Inference Request；
- Prefill / Decode / Forward / Backward；
- Current / Baseline 可比状态；
- Evidence completeness。

以上诊断上下文进入右上角 Info 图标触发的轻量弹层，不常驻占用工作区，也不横向堆叠在 IDE 顶栏。顶栏只保留产品身份，以及深浅模式、中英文、Run Info、Bottom Timeline 和 Inspector 开关。首版只有 MoE Lens 时不显示单选 Lens 控件。不放置：全局搜索、Terminal、Git、设置、账户头像、通知、帮助中心、分享、导出和通用 Dashboard 导航。

### 8.3 明确排除的无关 UI

- 永久三栏或四栏；
- 与任务无关的 Activity Rail 命令、文件 Explorer、代码编辑器和假的 IDE 工具；Activity Rail 仅保留 Traffic、Evidence、Timeline 三个任务入口；
- KPI 卡片墙、装饰性环形图和仪表盘；
- 与当前诊断无关的模型总览、硬件架构背景图；
- 粒子流、发光路径和没有数据语义的动画；
- 同时出现 Sankey、Heatmap、拓扑图、模型图和表格争夺主焦点；
- 用顶层 Tab 墙同时陈列所有通信 Lens；
- 没有真实 Step / Event 语义的 Playback；
- 为正常状态创建大量绿色 Badge；
- 每个节点都展示完整 metadata；
- 只为“显得专业”而增加的 Toolbar、筛选器和 Legend。

如果一个控件不能改变核心决策、定位证据、执行动作或恢复状态，应从默认界面删除。

---

## 9. L1 / L2 / L3 信息密度

| 层级 | 回答的问题 | 内容 | 显示方式 |
|---|---|---|---|
| L1 结论 | 发生了什么、影响什么、下一步是什么 | 一条诊断、影响、证据完整性、唯一下一步 | Inspector 顶部；选择证据对象后原位切换 |
| L2 工程证据 | 问题在哪、为什么 | 当前 Lens 的主视觉 + Timeline / Baseline Diff；MoE Lens 为 Sankey + Capacity | 默认工作区 |
| L3 专家证据 | 原始事实、Runtime 与硬件到底是什么 | Collective、Event、Wait Chain、Topology、Task / Stream、AIC/AIV/Core、Raw Trace | Contextual Dock 或独立证据页，按需打开 |

从 L3 返回必须保留 Run、Layer、Phase、Window、Selection、Expansion 和 Compare 状态。

---

## 10. PTO Design System 映射

### 10.1 现有 Pattern

| 页面职责 | 使用 Pattern | 使用边界 |
|---|---|---|
| Standalone 分析工作台 | `ide-frame` | 使用共享 IDE Skin、Activity Rail、Inspector、Top Bar 与 Status Strip；不伪造文件 Explorer / Terminal / Editor |
| Main / Inspector 与 Bottom Timeline 分栏 | `workbench-shell` | 只使用共享 Resize Kernel；Pane 数量由任务状态决定 |
| Hierarchical Profiling Timeline | `hierarchical-timeline` | Pattern 管理层级、固定行头、窗口、缩放与选择；业务仅配置行与事件，不复制 MB Lifecycle 页面结构 |
| Communication / Wait Event Bar | `swimlane-task` | 作为 `hierarchical-timeline` 的绘制原语；无原始 IN/OUT 数组时使用单段 Task Bar，不伪造三段样式 |
| Hierarchical Route Attribution | `communication-traffic-sankey` | Pattern 管理层级端点、Ribbon、时间窗聚合与选择；业务只注入 Rank / Engine Attribution / Event / Window 数据，Expert / Capacity 留在诊断证据层 |
| Expert Skew / Exposed Time 趋势 | `training-metrics-chart` | 只在用户查看跨 Step 趋势时按需打开；不作为首屏装饰 |
| 模型 Layer 上下文 | `model-training-graphviz` | 仅作为 L3 定位入口，不与 Sankey 同时默认展示 |
| 全局 Rank 上下文 | `model-parallel-rank-deck` | 先显示完整 Rank Inventory，再聚焦一个通信组；本产品使用无并行叠层的嵌入态，不取代组内 Sankey |
| 硬件 / Fabric 证据 | `memory-architecture` / `hardware-architecture-viewport` | 只有诊断指向 Placement / Fabric 时打开；不叠在 Sankey 背后 |
| Task → AIC/AIV 证据 | `aic-core-object` / `aiv-core-object` | 仅在真实 Task / Core 映射存在时使用 |
| 跨 Step 回放 | `floating-playback-control` | MVP 默认不用；只有回放会改变诊断状态时挂载 |

所有颜色、字体、间距、边框、状态、按钮和 Pane Chrome 使用共享 Token、基础组件和 Pattern。业务样式只负责当前 Lens 主视觉的挂载尺寸、语义布局与必要适配。

### 10.2 新 Pattern Preview Gate

以下能力已经通过 Preview Gate，以 `communication-traffic-sankey` 登记到 Design System：

#### `communication-traffic-sankey`

- Node → Rank → AIC / AIV / Unattributed 复合节点；
- 折叠 / 展开后 Ribbon 端点守恒重挂；
- Dispatch / Combine 双向语义；
- Token / Logical Bytes / Physical Bytes / Exposed Time 单口径切换；
- 引擎端点流量标注；Expert Capacity 不占用该 Pattern 的层级节点；
- Unattributed、Overflow、Cross-node 与 Baseline Diff 状态；
- 与 Timeline Event ID 的双向选择契约；
- Canvas / SVG 的键盘和数据表替代表达。

实现前必须创建独立 Component Preview，至少评审：

- normal；
- hover；
- selected；
- rank-expanded；
- expert-hot；
- capacity-warning / overflow；
- unattributed；
- baseline-diff；
- empty / missing data；
- high-scale aggregated。

该 Pattern 已由用户批准并先进入 `vendor/pto-design-system/patterns/` 注册，再由本项目消费。后续扩展仍不得在业务页面沉淀私有 Sankey 样式。

### 10.3 不新增的组件

- L1 Diagnosis 使用 Inspector 内的现有 Typography、Badge、Button、Panel 组合，不创建“诊断卡片系统”；
- Phase Navigator 使用现有 Tabs / Segmented Control 语义，并放在 Sankey Pane Header 中央，不创建特殊导航；
- Capacity 不创建独立 Gauge Pattern，先作为 `moe-route-sankey` 的 Expert Node 状态；
- 证据面使用 `ide-frame` 的 Inspector Dock，不创建另一套侧栏皮肤。

### 10.4 Pangu 全局 Rank Deck

`model-parallel-rank-deck` 复用设计系统内的 openPangu 模型配置、Rank Manifest 和 Three.js 实例化渲染。当前产品首先建立 32-Rank 全局范围，并以 8×4 网格改善总览可读性；网格位置只表达 Rank Inventory，不编码拓扑距离、带宽或延迟。

该视图与组内通信证据严格分工：

```text
32 Rank Global Rank Deck
  └─ 选择 EP group · PP2 / replica0
       ├─ 聚焦：R2 / R6 / R10 / R14
       ├─ 下一步 Sankey：同一 Collective 内的 4-Rank 流量守恒
       ├─ Timeline：Phase → Collective → Rank → Task
       └─ Inspector：范围、来源等级、限制与下一步
```

本轮不叠加 PP / TP / EP 分组框，也不显示 Rank 内 Layer Payload。通信组选择使用 `focusCommunicationGroup()`：只降低非成员 Rank 的视觉权重并显示组内连接，保留 32-Rank 全局位置。32-Rank 放置是明确标注的 Demo Topology，不宣称为 openPangu 官方训练部署；模型结构沿用 source-checked Pattern Preset。

---

## 11. 共享状态与多视图联动

```text
AnalysisState
├── lensId
├── runId
├── baselineRunId
├── stepOrRequestId
├── parallelGroupId
├── collectiveId
├── layerId
├── phase                    lens-defined model / execution phase
├── profileWindow            start / end / aggregation
├── primaryMeasure           lens-defined; tokens | logicalBytes | physicalBytes | exposedTime | bubble
├── selectedObjectId
├── selectedObjectType
├── expandedHierarchyIds[]
├── compareMode
└── evidenceCompleteness
```

| 触发 | 共享状态 | Sankey | Timeline | Inspector | 保持不变 |
|---|---|---|---|---|---|
| 选择 Phase | `phase` | 切换 Dispatch / Combine 语义 | 聚焦对应 Lane | 更新 Phase 证据 | Window、Layer |
| Brush Timeline | `profileWindow` | 重新聚合 | 显示窗口 | 更新指标与口径 | Selection 若仍有效 |
| 点击 AIC/AIV 端点 | `selectedObjectId` | 强调相关 Rank 流量 | 强调关联 Task / Wait | 显示引擎归因，并关联 Expert / Capacity 证据 | 缩放、Window |
| 点击 Ribbon | `selectedObjectId` | 持久选择路径 | 强调组成 Event | 显示 Bytes、Fabric、Overlap | 其他折叠状态 |
| 点击 Wait | `selectedObjectId` | 强调可能阻塞路径 | 选择 Wait Chain | 显示依赖与备选解释 | Layer、Phase |
| 展开 Rank | `expandedHierarchyIds` | 端点重挂到 AIC/AIV/Unattributed | 不改变时间过滤 | 更新路径面包屑 | Selection、Window |

交互状态严格区分：

- Hover：临时预览，不过滤、不自动缩放；
- Selection：持久焦点，可由 `Esc` 清除；
- Filter：明确改变数据范围，持续显示；
- Window：只改变聚合时间；
- Expansion：只改变层级显示，不改变数据；
- Playhead：只有真实回放存在时使用。

---

## 12. 数据对象与 Artifact 契约

### 12.1 核心对象

```text
Run
├── ModelContext
│   └── Step / Request → Layer → Phase
├── ParallelPlan
│   └── Group → Rank → Expert placement
├── RouterDecision
│   └── tokenId → expertId / score / topK
├── TokenRoute
│   └── sourceRank → destinationRank / expertId
├── Collective
│   └── commContext / algorithm / participants
├── CommEvent
│   └── timestamp / duration / payload / src / dst / transport
├── RuntimeTask
│   └── stream / wait / signal / task / engine
├── PhysicalTopology
│   └── host / node / device / die / fabric
├── Capacity
│   └── expert / current / limit / drop / padding
├── Metric
└── EvidenceRef
```

### 12.2 稳定 ID 与关系

每个对象至少包含：

```text
id / type / runId / label
source             real | derived | mock | missing
sourceRef
timestamp / step   when applicable
relations[]
```

必要关系：

```text
belongs-to-layer
belongs-to-phase
member-of-parallel-group
hosted-on-rank
routed-to-expert
implemented-by-collective
materialized-as-event
waits-on
runs-as-task
mapped-to-engine
traverses-fabric
compared-with
supports-diagnostic
```

### 12.3 缺失数据降级

| 缺失数据 | 允许展示 | 禁止结论 |
|---|---|---|
| 无 Router / Token→Expert | Rank-to-Rank Logical Payload | 不能诊断 Expert Load 或 Router Skew |
| 无 Physical Counter | Logical Bytes + 推定 Transport | 不能称为 Physical Link Bytes 或链路饱和 |
| 无依赖 / Overlap | Event Duration | 不能计算精确 Exposed Time |
| 无 Task / Stream | Collective Event | 不能下钻 AIC/AIV/Core |
| 无 Capacity | Expert Receive Count | 不能显示 Headroom、Overflow 或安全结论 |
| Baseline 条件不一致 | 分别展示两次 Run | 不计算精确回退百分比 |

所有降级必须在 L1 Evidence Completeness 和对应对象附近可见。

---

## 13. 诊断规则与动作映射

| 诊断类别 | 决定性证据 | 不能只凭 | 候选动作 |
|---|---|---|---|
| Router / Expert Skew | Token→Expert counts、跨 Step 稳定性、Capacity | Rank Bytes | Router Balance、Capacity Factor、Placement |
| Placement / Topology | Expert placement、Rank group、Fabric path、Baseline mapping | Rank ID 相邻 | 调整 Rank / Expert Placement、Group Mapping |
| Overlap 不足 | Comm 与 Compute 依赖、暴露窗口、Wait | Collective Duration | Overlap、Bucket / Microbatch、调度顺序 |
| Collective / Runtime | Participants、event pairing、wait chain、barrier、timeout | 流量大小 | 修复 Context、配对、算法或 Runtime |
| Buffer Risk | Receive Count、Runtime Capacity、Drop / Padding | 平均负载 | Runtime sizing、Packed Layout、安全容量 |
| Device-side overhead | Pack / Dispatch / Combine Task、Stream、AIC/AIV/Core timeline | Rank-to-Rank Bytes | Kernel fusion、任务调度、Core 分配 |

系统必须显示 Alternative Explanations，避免把相关性写成根因。例如“R2 流量大”可能来自 Router 偏斜、热门 Expert 共置、输入 Batch 特性或 Dispatch 复制；只有补齐对应证据后才能收敛。

---

## 14. 场景矩阵

| 场景 | 目的 | 必需表现 |
|---|---|---|
| 正常均衡 | 建立路由、容量和等待参照 | Max/Mean 接近 1、无暴露等待、容量安全 |
| Router Skew | 验证 Token→Expert 分流和热点识别 | Hot Expert、来源 Token、跨 Step 趋势 |
| Placement 问题 | 验证模型语义到物理拓扑 | 相同路由下跨 Node 比例与等待上升 |
| Buffer Risk | 验证最坏情况与容量边界 | Headroom、Overflow / Drop / Padding 清楚 |
| Overlap 回退 | 验证 Bytes 不变但关键路径变差 | Sankey 相似，Timeline 暴露时间增加 |
| Runtime Hang | 验证 Wait Chain 与拓扑组合 | 最早长期等待、参与 Rank、Barrier / Signal |
| Baseline / Current | 验证可比实验与归因 | 条件一致性、绝对值、差值、混杂因素 |
| Partial Attribution | 验证缺失不会伪装成正常 | Unattributed 与禁止结论 |
| Empty Window | 验证时间范围为空 | 原因、最近 Event、恢复动作 |
| High Scale | 验证 16+ Rank / 256 Expert | 分层聚合、Hot Path 自动展开、无毛线图 |

---

## 15. 可访问性与高密度可视化要求

- Sankey 提供可键盘遍历的 Rank / Expert / Ribbon 对象列表；
- 图形同时提供数据表或层级树替代表达；
- 展开 / 收起、Brush、Resize 均有键盘和数值输入路径；
- 颜色不是唯一编码，Hot、Cross-node、Unattributed、Overflow 同时使用文字、图标或线型；
- Focus 与 Selection 分离；
- Tooltip 不包含只能 Hover 才能获得的决定性信息；
- 200% 浏览器缩放下仍可完成黄金流；
- `prefers-reduced-motion` 下无粒子或不必要过渡；
- 大规模聚合说明对象数、范围与是否采样；
- Canvas / SVG 节点具备语义名称，关键变化通过 `aria-live` 摘要。

---

## 16. 实现阶段与 Gate

### 阶段 A：共享证据面与首个 Lens 数据可行性

1. 先建立 Parallel Group、Rank、Collective、Event、Wait、Topology 与 Evidence 的共享对象合同；
2. 为首个 MoE Lens 审计 Router、Dispatch、Expert、Capacity、Timeline 与 Task Artifact；
3. 确认稳定 ID 能否贯通 Token→Expert→Rank→Event→Task；
4. 输出字段可用性和 `real / derived / mock / missing` 表；
5. 无法支撑 Expert / Time / Capacity 中至少两项时，不进入 MoE Lens 高保真实现。

### 阶段 B：新 Pattern Preview

1. 只实现 `moe-route-sankey` Preview；
2. 用 4 Rank 验证层级和守恒，用 16 Rank 验证聚合；
3. 评审 Capacity、Unattributed、Selection 与 Timeline 联动；
4. 用户批准后先回流 PTO 设计系统。

### 阶段 C：产品骨架

1. 使用 `ide-frame` 标准骨架，Activity Rail 只挂载 Traffic、Evidence、Timeline；Run 上下文进入右上角 Info 弹层；
2. 只实现 Inspector 内的 L1、Sankey 标题栏中的 Phase、Sankey、Timeline 与按需证据；
3. 完成异常、正常、partial、empty 四种基础场景；
4. 不实现 Candidate、Topology 全景或 AIC/AIV 深层页。

### 阶段 D：证据闭环

1. 接入 Baseline / Current；
2. 增加 Topology、Wait Chain 与 Candidate Experiment；
3. 完成“行动 → 重跑 → 门禁”闭环；
4. 专家、产品和 UX 联合验证后再登记 Launch。

### 阶段 E：新增并行 Lens

每增加一个 Lens，都必须重新经过机会卡、证据表、黄金任务流和主视觉选择，不能复制 MoE 页面：

1. `Gradient Sync`：先验证 DP / FSDP 的慢 Rank、Bucket 与 Overlap 问题，主视觉优先 Rank Matrix + Timeline；
2. `Layer Collective`：验证 TP / SP 的层内 Collective 与拓扑敏感性，主视觉优先 Layer Timeline；
3. `Pipeline Transfer`：验证 PP Bubble 和 Stage Balance，主视觉使用 Pipeline Swimlane；
4. `Context Transfer`：验证 CP / KV / Activation 搬运与长序列代价，按 Artifact 选择 Timeline 或 Topology；
5. 复用 Shared Context / Evidence / State，不复用不匹配的 MoE Sankey。

---

## 17. 验收门槛

### 产品

- 用户能判断“通信是否真正影响 Step / SLO”，而不是只发现大流量；
- 产品数据模型和工作台 Shell 不把通信等同于 MoE；首个 MoE Lens 也不被未验证的其他 Lens 功能稀释；
- 每条诊断都有最小动作与验证门禁；
- 首屏只有一个主问题和一个主视觉。

### 专家可信度

- Token、Logical Bytes、Physical Bytes、Duration、Exposed Time 不混用；
- Rank、Expert、Collective、Task、AIC/AIV 与 Fabric 关系准确；
- 缺失数据不会被推断填补；
- Bytes 大不会自动被标记为根因；
- Router Skew、Placement、Overlap、Buffer 与 Runtime 证据可区分。

### 交互

- 3–5 分钟完成黄金流；
- Window、Selection、Filter、Expansion、Hover 语义不混淆；
- Sankey 与 Timeline 双向定位；
- Inspector 关闭时黄金流仍成立；
- 返回 L2 后上下文完整保留。

### 设计系统

- 复用 `ide-frame`、`workbench-shell`、`swimlane-task` 等现有 Pattern；
- 新 Sankey 先 Preview、获批、回流，再由业务页消费；
- 没有业务私有按钮、Badge、Pane Chrome、播放条或主题；深浅模式直接消费设计系统主题；
- 没有与核心决策无关的 UI 元素。

### 工程与可访问性

- 标准视口 1440 × 900 可完成黄金流；
- 关键操作可键盘完成；
- 图形有数据表 / 树替代表达；
- HTTP 打开无控制台错误；
- 高规模场景不阻塞交互，聚合规则可解释。

---

## 18. 待验证问题

1. 仓库真实 Artifact 是否能稳定关联 `tokenId → expertId → rank → collectiveEvent`？
2. AIC/AIV 归因对应的是 Pack / Dispatch / Combine Task，还是通信路径本身；哪些字段可以证明？
3. Physical Bytes 是否有链路 Counter，还是只能从 Collective 算法推导？
4. Exposed Communication Time 是否有完整依赖图支撑？
5. 用户最常从 Step Time、Profiler Collective、Buffer Risk 还是 Hang 进入？
6. Expert Capacity 的 Runtime 合同在 Prefill、Decode 和训练场景中是否一致？
7. 4 Rank Preview 的层级交互能否自然扩展到 EP16 / 256 Experts？
8. 第一版应以训练 Step 还是分布式推理 Prefill / Decode 为默认场景？

这些问题决定首个 Lens 的数据与默认场景，但不改变产品核心：让用户从任一种模型训推通信异常进入匹配的诊断 Lens，并走到可信、可执行、可验证的工程决策。
