# SPEC · openPangu-R-72B-MoE-V1.3 · 算子 × 硬件 × 时间

> 目标页面: `pangu-moe-trainviz/op-rank-time.html`
> 当前状态: V1.3 训练透视 workbench 规格, 2026-06-29 更新
> 核心契约: **真实架构数据只读; 1F1B schedule / Loss / MoE 负载 / 诊断均为确定性仿真或推断, 不能写成 profiler 实测。**

---

## 0. 页面定位

本页已经不是单纯的"算子-硬件前向播放页",而是一个训练透视 workbench:

1. 上方 3D 舞台:完整模型架构 + 32 rank UB-fabric 卡阵 + 当前 1F1B tick 的算子/通信高亮。
2. 右侧 Inspector:诊断视图与聚焦 rank 视图。
3. 底部 Analysis Dock:Loss、Timeline/Swimlane、MoE Load、Card Load 四个分析视图。
4. 两套时间轴:
   - 外层 `step`:训练指标主时间,来自 `data/timeseries.js`,驱动 Loss、MoE 负载、Card Load、异常诊断。
   - 内层 `traceIndex`:一次迭代内 1F1B tick,来自 `data/strict-1f1b-trace-sim.json`,驱动 3D 高亮和 Swimlane 播放头。

页面回答三个问题:

- 当前 1F1B tick 中,哪个 PP stage / micro-batch / layer / op 正在映射到哪些 rank。
- 训练 step 上的 loss / load-balance 异常如何映射到 MoE 专家负载、EP All-to-All 和 rank 等待。
- collective 通信高亮与单卡故障如何区分,避免把所有参与 rank 都解释成坏卡。

---

## 1. 不可破坏约束

1. **架构只读**:算子拓扑仍复用 `buildUltraMoE718BGraph({moeExpanded:true, expertExpanded:true})` 作为只读模板。页面可以在渲染层按 72B config 复制成 L0-L49,但不能改 `data/graph-ultramoe-718b.js` 的节点、边、标签和语义。
2. **完整层口径**:`Embedding/Input` 只出现一次;`L0-L3` 是 Dense decode layers;`L4-L49` 是 MoE decode layers;`Final RMSNorm / LM Head / Logits All-Gather / Logits` 只出现一次。openPangu-R config 声明 `num_mtp_layers=1`,但当前主图聚焦 causal LM forward,不画 MTP head。
3. **MoE 结构诚实**:每个 MoE layer 是 `MLA -> Pre-RMSNorm -> Router -> A2A Dispatch -> Routed Expert Pool + Shared Expert -> A2A Combine -> Post-MLP RMSNorm`。`Expert Pool` 表示本层 80 routed experts,不是 2 个层,也不是 2 张卡。
4. **时间诚实**:`strict-1f1b-trace-sim.json` 是 deterministic schedule simulation,不是 profiler kernel trace。3D 上的 `opFocus` 是代表算子,不能写成完整 kernel 序列。
5. **气泡诚实**:steady tick 可以展示 32 卡满载;warmup/drain tick 必须允许 bubble,并在 tooltip / Inspector 中明确为等待依赖或流水线排空,不能伪装成计算。
6. **collective 语义诚实**:EP A2A / TP AR 高亮所有参与 rank 只表示通信参与面;不能暗示每张卡独立故障。
7. **单一状态源**:外层训练时间以 `focusContext.step` 为准;内层 1F1B 时间以 `traceIndex` 为准。不要再引入独立 PP 游标或组件私有 step。
8. **设计系统优先**:页面外壳、播放条、swimlane、tokens 必须继续复用 PTO design system / patterns;新增 UI 先考虑现有 pattern。

---

## 2. 数据来源与保真边界

### 2.1 算子拓扑模板: `data/graph-ultramoe-718b.js`

页面调用:

```js
window.buildUltraMoE718BGraph({ moeExpanded: true, expertExpanded: true })
```

关键节点:

- 输入: `token_ids`, `position_ids`, `attn_mask`
- 主干: `embedding`, `embedding_weight`, `dense_block`
- MoE: `mla`, `moe_prenorm`, `gate`, `w_gate`, `a2a_dispatch`, `expert_group_00..31`, `shared_expert`, `w_expert`, `a2a_combine`, `moe_residual`
- 输出: `final_norm`, `lm_head`, `lm_head_weight`, `logits_allgather`, `logits`

`logits_allgather` 是当前 V1.3 输出路径的一部分;MTP head 不在当前 graph 中。这个文件名和 builder 名称保留历史命名,当前页面的层数、Dense/MoE 切分、专家数和并行配置由 `MODEL_CONFIG` 覆盖。

### 2.2 1F1B tick trace: `data/strict-1f1b-trace-sim.json`

- schema: `pto.strict-1f1b-trace-sim.v2`
- fidelity: `schedule-simulated-not-profiler-measured`
- 配置: `DP2 x PP4 x TP2 x EP2`,8 micro-batches,non-interleaved 1F1B,含 warmup / steady / drain。
- stage range: `PP0 = L0-L12`, `PP1 = L13-L25`, `PP2 = L26-L37`, `PP3 = L38-L49`。
- 每个 tick 的 stage event 字段:

```ts
{
  phase: 'F' | 'B' | 'bubble',
  micro?: number,
  layerRange?: [number, number],
  opFocus?: { layer: number, step: string },
  stepIndex?: number,
  stepTotal?: number,
  label: string,
  comm?: 'tp' | 'ep' | 'pp' | 'dp',
  dependsOn: string[],
  produces: string[],
  explain: string
}
```

3D 舞台只消费该事件的代表 `opFocus`;Timeline/Swimlane 的 wall-clock task 来自 `buildSimulated1F1BRuntime()`。

### 2.3 训练指标主时间: `data/timeseries.js`

`window.TS_DATA` 生成 step `1900..2100` 的合成指标:

- `train_loss`
- `val_loss`
- `eval_mmlu`
- `grad_norm`
- `load_balance_loss`

关键事件:

- `faultStep = 1997`:混合精度权重更新 stride 算错 -> 写越界,节点指向 `gate`。
- `collapseStep = 1998`:路由坍缩起点。
- `defaultStep = 2000`:页面默认打开 step。

`regimeOfStep(step)`:

- `< 1997`:健康区
- `1997..1999`:根因 / 故障区
- `>= 2000`:路由坍缩区

### 2.4 分析模型: `js/analysis-data.js`

- `buildMoeRuntimeMetrics()`:生成 L4-L49 × 80 experts 的负载、drop、reroute、A2A send/recv 等矩阵。
- `buildMoeLoadViewModel()`:MoE heatmap 视图模型。
- `buildSimulated1F1BRuntime()`:生成 32 rank 的 F/B/TP/EP/PP/bubble swimlane task。
- `buildRankLoadViewModel()`:聚合 rank compute / comm / bubble 比例。
- `buildCardLoadViewModel()`:把 rank 负载聚合成每 step 的 32 卡占用快照。

`collapseIntensity(step)` 从 `load_balance_loss` 派生,用于让 MoE 负载和 Card Load 随外层 step 进入坍缩状态。该逻辑是演示推断,不是实测。

### 2.5 openPangu-R-72B 模型与并行口径

本 demo 的模型结构口径来自 `openPangu-R-72B-2512/config.json` 和 OmniInfer `pangu_moe_v2.py`:

- `num_hidden_layers = 50`
- `mlp_only_layers = [0,1,2,3]`,因此 L0-L3 为 Dense decoder layer。
- L4-L49 为 MoE decoder layer,共 46 层。
- `num_experts = 80`, `num_experts_per_tok = 8`。
- `hidden_size = 4608`, `num_attention_heads = 64`, `num_key_value_heads = 4`。
- `moe_intermediate_size = 1280`, `shared_expert_intermediate_size = 2560`。

训练并行口径参考用户提供的 Pangu 72B 论文 `arXiv:2505.21411v1`;论文明确披露:

- 训练并行配置: `TP=8`, `EP=2`, `PP=5`, `VPP=5`, `CP=1`。

DP 保真边界:

- `DP` 未作为论文原文配置项披露。
- 若仅按 `4K Ascend NPUs` 近似为 `4000` 张卡,且把训练 model-parallel replica 物理占卡数按 `TP x EP x PP x CP = 8 x 2 x 5 x 1 = 80` 计算,可推导 `DP ≈ 4000 / 80 = 50`。
- 这个 `DP≈50` 必须标注为 inferred/derived,不能写成论文原文配置。
- `VPP=5` 是虚拟流水切分/调度维度,不乘入物理 rank 数。
- 页面为了让关键通信可见,使用 `DP2 x PP4 x TP2 x EP2 = 32 ranks` 作为缩小 demo,不是论文原始训练规模。
- 若有人把 `4K` 解释为 `4096`,则 `4096 / 80 = 51.2` 不是整数 DP;因此页面/spec 中应写 `4K 近似推导 DP≈50`,不要写精确 `DP=51.2` 或 `DP=50` 官方值。

---

## 3. 架构渲染契约

### 3.1 坐标与层复制

页面采用 Z-up 读法:

- 显示 `X`:地面左右 / graph local X。
- 显示 `Y`:地面深度 / decode layer 复制方向。
- 显示 `Z`:高度 / graph local Y。

实现内部 Three.js 仍使用 `Vector3(x, y, z)`,其中内部 `y` 是显示 `Z`,内部 `z` 是显示 `Y`。spec 和 UI 文案以页面显示坐标为准。

复制规则:

- `input` 层:只放输入、Embedding、Embedding Weight。
- `L0-L3`:Dense layer block,每层一块,没有 Router / A2A / Expert Pool / Shared Expert。
- `L4-L49`:MoE layer,每层复制完整 MoE 代表层。
- `output` 层:只放 Final RMSNorm、LM Head、Logits All-Gather、Logits。
- Hidden-state 主链路为 `Embedding -> L0 -> L1 -> L2 -> L3 -> L4 ... L49 -> Final/Head`。

### 3.2 视觉取舍

- L0 和 L49 为 solid 重点层,写深度缓冲,保证遮挡正确。
- MoE 主体每 10 层显示明显 label / marker;其它层保留低透明度,不能完全消失。
- Expert Pool 在普通态是单个 block;在 hover、active、`op:experts`、`comm:ep`、`data:dispatch` 等焦点下显示网格:
  - 细网格:80 experts。
  - 粗网格:2 EP buckets,每 bucket 约 40 routed experts。
- 侧视图必须保留 hidden-state rail,用于解释层深度和 PP range。
- Front/Side flat views 的文字尽量贴在真实 3D 对象上,不要新增远离对象的大 billboard。

---

## 4. 硬件映射契约

### 4.1 代表硬件配置

页面代表配置为:

```txt
DP2 x PP4 x TP2 x EP2 = 32 ranks
```

72B 论文训练配置参考:

```txt
official paper: TP8 / EP2 / PP5 / VPP5 / CP1
derived: model-parallel replica = TP8 x EP2 x PP5 x CP1 = 80 ranks
derived: DP ≈ 50 when 4K Ascend NPUs are approximated as 4000 ranks
```

72B 训练配置的展示规则:

- `VPP5` 显示为每个 physical PP stage 内的 virtual pipeline chunks,不增加物理卡片数量。
- `EP2` 是论文披露训练配置,也被当前 32-rank demo 保留。
- `DP≈50` 是由 4K NPU 和 80-rank model-parallel replica 推导,Inspector/tooltip 必须标注 `derived`。
- demo 将论文物理模型并行缩小为 `PP4 x TP2 x EP2`;目的不是复刻吞吐,而是让 PP/TP/EP/DP 四类通信在 32 rank 内同时可见。

rank 公式:

```txt
rank = (((dp * PP + pp_stage) * TP + tp) * EP + ep)
```

当前 8 个 node/tray:

| node | DP | PP | rank range |
|---|---:|---:|---|
| D0.PP0 | 0 | 0 | 0-3 |
| D0.PP1 | 0 | 1 | 4-7 |
| D0.PP2 | 0 | 2 | 8-11 |
| D0.PP3 | 0 | 3 | 12-15 |
| D1.PP0 | 1 | 0 | 16-19 |
| D1.PP1 | 1 | 1 | 20-23 |
| D1.PP2 | 1 | 2 | 24-27 |
| D1.PP3 | 1 | 3 | 28-31 |

每个 node 4 张卡,本地坐标为 `TP0/1 x EP0/1`;每张卡显示 Ascend logo plate、board edge、package、tray outline。UB-fabric 卡片尺寸沿用当前实现:

- board box: `0.42 x 0.04 x 0.3` 乘 `SC`
- package box: `0.2 x 0.02 x 0.16` 乘 `SC`

### 4.2 算子到硬件

| step | label | nodes | comm | rank 角色 |
|---|---|---|---|---|
| `embedding` | Parallel Embedding | `embedding`, `embedding_weight` | TP | 同 DP/PP/EP 下 2 个 TP rank 做词表/hidden shard 通信 |
| `dense` | Dense 解码层 ×4 | `dense_block` | TP | 同 DP/PP/EP 下 2 个 TP rank 做 dense shard 通信 |
| `mla` | MLA 注意力 | `mla` | TP | 同 DP/PP/EP 下 2 个 TP rank 做 attention shard 通信 |
| `moe_prenorm` | Pre-RMSNorm | `moe_prenorm` | local | per-rank local compute |
| `gate` | Router Top-8 | `gate`, `w_gate` | local | per-rank routing |
| `a2a_dispatch` | All-to-All 分发 | `a2a_dispatch` | EP | 同 DP/PP/TP 下 2 个 EP rank 做 token dispatch collective |
| `experts` | Expert Pool + Shared Expert | `expert_group_00..31`, `shared_expert`, `w_expert` | local | EP shard / expert bucket compute |
| `a2a_combine` | All-to-All 汇聚 | `a2a_combine` | EP | 同 DP/PP/TP 下 2 个 EP rank 做 token combine collective |
| `moe_residual` | Post-MLP RMSNorm | `moe_residual` | local | residual + RMSNorm |
| `final_norm` | Final RMSNorm | `final_norm` | local | output local normalize |
| `lm_head` | LM Head | `lm_head`, `lm_head_weight` | TP | vocab projection shard + All-Reduce |
| `dp_sync` | DP Gradient Sync | swimlane task / 3D DP arcs | DP | 同 PP/TP/EP 坐标跨 D0/D1 replica 做梯度同步 |

额外静态对象:

- `logits_allgather`:输出路径上的 gather / no-op 语义对象,支持 `comm:logits` 对象聚焦。
- `pipeline bubble`:不是算子,是等待依赖或 drain,卡片使用 bubble 色和 tooltip 说明。

### 4.3 通信线

- TP:同 node 内固定 EP index 的 2 张卡连线 + group halo。
- EP:同 node 内固定 TP index 的 2 张卡连线 + group halo。
- PP:同 DP 的相邻 PP stage 弧线,仅在当前 tick 存在 hidden/grad handoff 或 `comm:pp` 聚焦时显示。
- DP gradient:同 PP stage 跨 D0/D1 副本的弧线;对象聚焦 `comm:dp` 时变亮。

---

## 5. 双时间轴与状态模型

### 5.1 外层训练 step

外层 step 由 `focusContext.step` 管理,是 Loss / MoE / Card / Inspector 的主时间。

控制入口:

- 底部 floating playback 的 back / play / forward / replay / scrubber。
- Loss chart hover / brush。
- 诊断 action 或 anomaly selection。

播放语义:

- `STEP_PLAY_INTERVAL = 110ms`
- 播放条 counter: `step current / STEP_MAX`
- 播放条 title: `step current · regime`

外层 step 不应直接覆盖当前 `traceIndex`;内层 tick 是一次迭代内的独立播放头。

### 5.2 内层 1F1B tick

内层 tick 由 `traceIndex` 管理,是 3D scene / current tick / swimlane playhead 的主时间。

控制入口:

- 3D 舞台右上 `1F1B` 上一帧 / 播放 / 下一帧。
- Timeline/Swimlane 点击 wall-clock 位置 seek。

播放语义:

- `INNER_PLAY_INTERVAL = 800ms`
- label: `t{traceIndex}/{lastTick}`
- `statusLeft`: `迭代内 1F1B · tX/Y · region · comm/phase/layer`

`traceIndex` 变化必须触发 `redraw()` 和 Timeline 重绘。

### 5.3 focusContext

`focusContext` 是跨视图共享上下文:

```ts
{
  source: string,
  step: number,
  window: [number, number] | null,
  layer: number | null,
  op: string | null,
  rank: number | null,
  dp: number | null,
  stage: number | null,
  tp: number | null,
  metric: string | null,
  severity: number,
  confidence: 'inferred' | string,
  regime: 'healthy' | 'fault' | 'collapse'
}
```

订阅者必须通过 `onContext()` 响应变化;不要让 Loss、MoE、Card、Inspector 各自保存互相冲突的状态。

---

## 6. Analysis Dock

底部 dock 使用 `createAnalysisDock()` 管理 tab、标题、meta、active view 和 localStorage。

### 6.1 Loss

标题: `训练效果趋势`

数据:

- `train_loss`
- `val_loss`
- `eval_mmlu`
- `grad_norm`
- `load_balance_loss`

交互:

- cursor hover -> `dispatch({source:'loss', step})`
- brush -> `dispatch({source:'loss', window})`
- 非健康区 regime badge 可点击,选中 `loss-collapse`

### 6.2 Timeline / Swimlane

标题: `Timeline / Swimlane`

内容:

- 32 rank 行。
- F/B compute task 以 micro-batch 色显示。
- `tp`, `ep`, `pp` 通信 task 可用细条子轨或"通信双轨"显示。
- bubble 以斜纹 wait block 显示。
- 当前 `traceIndex` 映射为竖直 playhead。

交互:

- 点击 timeline 按 wall-clock seek 内层 tick。
- hover tooltip 必须显示 rank、kind、category、start、duration、status 和语义解释。
- `commSplit = true` 时行高翻倍,上半计算、下半通信。

下钻高亮:

- `moe-a2a`:高亮所有 EP communication task,强制打开通信双轨,3D 同步聚焦 `comm:ep`。
- `rank-wait`:高亮目标 rank 的 bubble / pp / ep / tp task,3D 同步聚焦该 rank 所在 node。

### 6.3 MoE Load

标题: `MoE Load`

矩阵:

```txt
L4-L49 = 46 rows
experts 0-79 = 80 columns
```

可选指标:

- `loadRatio`
- `droppedTokens`
- `reroutedTokens`
- `allToAllSendTokens`

交互:

- hover 显示 layer/expert/metric 和 layer avg/p95。
- click expert -> `dispatch({source:'moe', layer, op:'experts', metric})`,设置对象聚焦 `op:experts`,状态条显示 expert 读数。

### 6.4 Card Load

标题: `Card Load`

布局:

- 32 个正方形卡片。
- 按 DP/PP 组排序,组内按 TP、EP 排序。
- 当前实现 `--card-cols = ceil(cards.length / (TP x EP))`,在 `TP2 x EP2` 下形成 8 组、每组 4 卡的紧凑网格。

指标:

- 底色浓淡: `utilRatio`
- 下方 meter: `commRatio`
- 状态:
  - `ok`:正常
  - `warn`:util > 95% 或 comm > 50%
  - `alert`:util < 30%

交互:

- info popover 解释 `util = sum(compute_us) / iter_wall_us`。
- click card -> 硬件聚焦,设置 `focusNode`,对象聚焦 `hw:cards`。

---

## 7. Inspector 与异常诊断

右侧 Inspector 有两个 tab:

- `诊断`:总诊断、finding list、诊断卡。
- `聚焦`:当前 focused rank / current event / comm tag。

### 7.1 anomaly 类型

`computeAnomalies()` 当前派生:

1. `loss-collapse`:Loss 域异常,锚点在 step,不会生成 3D badge。
2. `moe-zone`:MoE 路由坍缩层带,取当前 MoE metrics 里 max load 最高的两层聚合成一个 zone。
3. `rank-{id}`:bubble ratio 最高的 rank,锚点在对应硬件 node。

所有异常都必须带:

- severity:1/2/3
- diagnosis
- evidence
- cause
- confidence:`推断`
- swimlaneNote
- actions

### 7.2 Inspector findings

除 anomaly 映射外,Inspector 还生成建议类 finding:

- `finding-pp-size`:PP stage 空闲与切分建议。
- `finding-tp-size`:TP All-Reduce 通信占比建议。

finding 只提供推断和下一步入口,不能写成自动根因判定。

### 7.3 诊断 action

当前 action 类型:

- 切换到底部 view:Loss / Timeline / MoE Load / Card Load。
- 设置对象聚焦:`op:experts`, `comm:ep`, `comm:tp`, `hw:cards`。
- 下钻 Swimlane / 3D:
  - `drillSwimlane({type:'moe-a2a', ...})`
  - `drillSwimlane({type:'rank-wait', ...})`
- 复制 profiler 命令到 clipboard。

复制命令是采集建议,不是页面已执行的操作。

### 7.4 3D anomaly 标记

- DOM badge 只贴 `zone` / `card` 锚点。
- `moe-zone` 用 3D dashed box 包住 Expert Pool 的层带,外加四角包角线。
- 选中状态:
  - 选中 badge 加白 outline。
  - 其它 3D badge dim。
  - `loss-collapse` 或 `moe-zone` 选中时 anomaly box 加亮。
- `Escape` 清除 anomaly selection、three drill target 和 swimlane highlight。

---

## 8. 3D 舞台交互

### 8.1 顶部工具

舞台右上工具:

- Zoom:缩小 / slider / 放大 / reset / 百分比输出。
- View preset:`轴测`, `正视`, `侧视`。
- Object filter panel。
- Inner 1F1B tick playback。
- Axis overlay。

相机:

- 默认 perspective,长焦 FOV 12。
- 预设视图切换到 ortho。
- 拖拽旋转。
- `Command + 拖拽` 平移。
- 滚轮缩放。
- zoom ratio 范围约 `0.7x .. 3.0x`。

### 8.2 对象聚焦

object focus 值:

- `all`
- `layer:dense`, `layer:moe`
- `op:mla`, `op:router`, `op:experts`, `op:norm`
- `data:hidden`, `data:dispatch`, `data:logits`
- `comm:tp`, `comm:ep`, `comm:pp`, `comm:logits`
- `hw:cards`

规则:

- 聚焦时匹配对象增强,不匹配对象降低 opacity。
- `comm:ep` 和 `data:dispatch` 必须同时让 A2A 链路、Expert Pool 和相关 rank 参与面可读。
- `hw:cards` 聚焦时通信/架构动态线退到背景,硬件卡保持主视觉。

### 8.3 Hover / click

- hover card:显示 node label、TP/EP 坐标、global rank、当前 event、role、通信解释和 drill 说明。
- hover architecture object:显示 layer、对象类型、desc、上下游 neighbor 高亮。
- hover side-view hidden rail:解释 hidden state main path。
- click card:设置 `focusNode`。

Tooltip 必须继续强调:

- Expert Pool 是本层 80 routed experts,不是层也不是卡。
- bubble 是等待依赖,不是计算。
- collective 高亮不是单卡独立故障。

---

## 9. UI / Design System 依赖

CSS:

- `../vendor/pto-design-system/tokens/foundation.css`
- `../vendor/pto-design-system/tokens/semantic.css`
- `../vendor/pto-design-system/tokens/components.css`
- `../vendor/pto-design-system/css/style.css`
- `../vendor/pto-design-system/patterns/workbench-shell/pattern.css`
- `../vendor/pto-design-system/patterns/floating-playback-control/pattern.css`
- `../vendor/pto-design-system/patterns/ide-frame/pattern.css`
- `./vendor/swimlane-task/pattern.css`

JS globals:

- `workbench-shell/pattern.js`
- `floating-playback-control/pattern.js`
- `swimlane-task/pattern.js`
- `ide-frame/pattern.js`
- `../js/colormap.js`
- `./pangu-palette.js`

Three.js importmap:

```json
{ "imports": { "three": "./vendor/three/build/three.module.min.js" } }
```

布局:

- `pto-ide-frame` 占满 `100vw x 100vh`,不是 4:3 framed preview。
- 主 split:上方 stage row / 底部 analysis dock,默认 `60/40`。
- stage row split:3D preview / inspector,默认 `72/28`。
- Analysis / Inspector toggle 状态写入 localStorage。
- active analysis view 写入 `op-rank-time-analysis-view`。
- theme 写入 `op-rank-time-theme`。

---

## 10. 视觉规范

- 页面主题默认 dark,支持 light。
- 当前 palette 固定:
  - `SELECTED_PALETTE_ID = 'balanced'`
  - `SELECTED_LIGHT_VARIANT_ID = 'clear'`
- 图例颜色必须从 `SEM_COLOR` / `LINE` 同源生成,保证图例、3D、swimlane 一致。
- Light mode 中 3D label / tooltip / UI 文本必须使用深色可读文字。
- 3D label 不使用大面积描边;字号自动缩放以适配节点。
- 不新增装饰性渐变/orb/bokeh。
- 不把 explanation 散成大型 3D billboard;说明放 info popup、tooltip、Inspector 或 status strip。
- 卡片、控件 radius 遵循 design system;不要在 card 内再套 card。

---

## 11. 非目标

- 不做完整 4K/4096 规模 rank 渲染;页面只能展示代表切片或聚合视图。
- 不把 `DP≈50` 写成 Pangu Pro MoE 72B 论文原文配置;它只是由 `4K Ascend NPUs` 和 `TP8 x EP2 x PP5 x CP1` 推导。
- 不把 32-rank demo 写成论文原始训练配置;它是为了展示 DP/PP/TP/EP 关键通信的缩小切片。
- 不声明 profiler measured kernel duration。
- 不把仿真诊断写成确定根因;所有当前诊断 confidence 为推断。
- 不把 EP bucket 解释成 layer 或 card。
- 不新增/恢复 MTP head,除非 topology source 明确提供节点且页面有对应模型口径。
- 不修改 `graph-ultramoe-718b.js` 来适配页面高亮;页面高亮必须从真实节点映射表推导。
- 不把 warmup/drain bubble 隐藏成"全卡满载"。

---

## 12. 验收清单

1. 页面打开后 topbar `traceMeta` 显示 `schedule-simulated-not-profiler-measured`、schedule 名称和 tick 数。
2. 3D 主图显示 `Embedding/Input` 一次、`L0-L3 Dense`、`L4-L49 MoE`、`Final/Head/Logits` 一次;info popup 明确当前不画 MTP。
3. `traceIndex = 4` 一类 steady tick 下,PP0/PP1/PP2/PP3 显示不同 micro-batch / phase / op,card tooltip 能说清 rank 当前 role。
4. `traceIndex` 切到 warmup/drain bubble tick 时,卡片和 Inspector 明确显示 pipeline bubble。
5. floating playback 改变外层 step 时,Loss cursor、MoE Load、Card Load、Inspector findings 同步更新。
6. 内层 1F1B 控件只改变 3D 高亮和 Swimlane playhead,不改变 Loss step。
7. Timeline 通信双轨可切换;A2A 下钻必须强制打开通信双轨并解释 collective 语义。
8. MoE Load 点击 expert 后,对象聚焦为 `op:experts`,状态条显示 layer/expert/metric。
9. Card Load 点击卡后,3D 聚焦对应 node,对象聚焦为 `hw:cards`。
10. 选择 `loss-collapse` / `moe-zone` / rank finding 后,Inspector 诊断卡显示 evidence、cause、confidence、actions。
11. `moe-zone` 选中时 3D dashed anomaly box 包住对应 Expert Pool 层带。
12. Theme 切换后 3D scene、labels、tooltip、legend、Swimlane、Analysis Dock 均可读。
13. `轴测/正视/侧视`、拖拽旋转、`Command + 拖拽` 平移、滚轮/slider 缩放都正常。
14. Object filter 中 `comm:ep`, `comm:dp`, `data:hidden`, `hw:cards` 等焦点不会让关键对象完全不可见。
15. 页面文字不把仿真 trace、合成指标、推断诊断说成实测。

---

## 13. 后续改动规则

任何后续页面改动必须同时更新本 spec:

- 新增数据文件或字段:更新 §2。
- 修改 architecture / hardware / step 映射:更新 §3 / §4。
- 修改播放语义:更新 §5。
- 新增 Analysis Dock view:更新 §6。
- 新增 anomaly/finding/action:更新 §7。
- 新增 UI pattern 或交互:更新 §8 / §9。
- 改变保真边界或非目标:更新 §1 / §11。
