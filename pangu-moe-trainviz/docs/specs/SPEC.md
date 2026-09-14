# 训练透视 · 工程 Spec（实时监护版）

> **工作代号**：TrainScope · 训练透视
> **一句话**：训练**正在跑**时，沿「计算逻辑架构 × Ascend 910B 物理运行时」两轴，用统一 step/phase 和 **hover-first** 联动把 loss、架构节点、rank、通信流对上号，实时排出偏差最大的节点、直出根因、给出调整建议。
> **版本**：v2（2026-06-16）　**状态**：开工基线
> **关系**：本 SPEC 是 `../product/PRD-训练透视.md` 的工程落地；本地归档的旧 SPEC（七步事故剧本）降级为 §7「事故回放模式」的内容载体，资产零浪费。
> **接入对象**：openPangu-Ultra-MoE-718B（真实 config，详见附录）；模型可经 adapter 替换。

---

## 0. 本版相对 archive 的工程差异

| 维度 | archive（事故复盘剧本） | 本版（实时监护） |
|---|---|---|
| 主模式 | 单一已烘焙事故的七步回放 | **实时偏差监护**为主，回放为辅 |
| 架构图对象 | Pangu Pro MoE（72B）/ 实现里曾是 7B Dense | **openPangu-Ultra-MoE-718B**，整体重建图数据 |
| 中心交互 | 顺七步点一遍 | **hover 对象 → 双轴点亮 → 内嵌根因 tag/tip → 建议** |
| 数据 | 全合成 mock | 逻辑轴接真实开源遥测（OLMoE/BLOOM），物理轴自采/合成 |
| 新增构件 | — | 偏差雷达 C1、内嵌根因 tag/tip C3、调整建议 C4、数据接入层 |
| 布局 | 三栏 + 底 dock | **A·并置双轴**：左右等宽可拖（50/50），底部实时 loss + step phase dock |

**不变的地基（直接复用，不重写）**：PtoWorkbenchShell 分栏外壳、model-training-graphviz 渲染引擎与联动高亮、model-graphviz mesh、training-metrics-chart、事件总线四类广播、PTO token 视觉规范。

---

## 1. 范围与对象：双轴 × 统一 step/phase

| 轴 | 承载 | 视图 | 回答 |
|---|---|---|---|
| **计算轴**（逻辑架构） | 层 / 算子 / 专家桶 / 权重 / 梯度 / loss contribution | 左舞台 · 718B MoE 架构图 + step phase overlay | 当前 step 的计算逻辑里**哪个环节**先偏了？ |
| **物理轴**（910B 运行时） | Ascend 910B 卡 / device / rank / TP·PP·CP·DP·EP 坐标 / HCCL 通信 | 右舞台 · 910B physical base + runtime overlay | **哪张卡 / 哪个 rank / 哪条通信流**表现异常？ |
| **关系层**（核心价值） | model object ↔ rank placement ↔ device ↔ weight shard ↔ comm flow | hover-first 联动 + 统一 step/phase + 底部 loss cursor | 同一异常的多个投影**对上号** = 可解释 |

**边界**：训练**正确性**排障，不是性能 profiler。不接 Chrome Trace（CTF）；通信是按 step/phase 的轻量流量快照，非 kernel trace。`rank` 是分布式进程/通信成员，常见 demo 可一 rank 绑定一张 910B 卡，但产品语义不能写成“rank 就是卡”。

---

## 2. 信息架构（A·并置双轴）

```
┌─ 顶部：run 标识 + 实时状态条（当前 step · 偏差雷达 TOP-N 告警）───────────────────┐
├──────────────────────────────────┬──────────────────────────────────────────┤
│  计算轴 · 718B MoE 架构图 [主舞台]   ┃  物理轴 · Ascend 910B Runtime Placement │
│  Step phase rail: Batch/Fwd/Loss/Bwd/Sync/Update ┃  910B 卡底图 + rank placement     │
│  Gate→A2A→256 routed experts【32 runtime buckets】→Combine ┃  TP/PP/CP/DP/EP lens 切换 │
│  节点叠加 execution / metric / parameter 状态   ┃  通信 overlay：All2All/AllReduce/P2P │
│            ⟵ 中缝可拖（默认 50/50）⟶  ┃                                          │
├──────────────────────────────────┴──────────────────────────────────────────┤
│  ⟱ 高度可拖 ⟱   底部 realtime loss console（图表即 step cursor，无独立播放条）       │
│  主图：global train/val loss · expected/previous/threshold · 异常点 · 兴趣窗口   │
│  辅助：grad norm / load balance loss / router z-loss / expert load var           │
│  交互：hover loss/phase/rank/node → step + phase → 架构路径 + 910B 通信 overlay  │
└──────────────────────────────────────────────────────────────────────────────┘
```

**workbench-shell 落地**：
- 外层 `.workbench-frame`（column）：header = run 标识 + 实时状态条；其下 **vertical split**（`direction:'vertical'`，panes=`[双轴主区, 底部 dock]`，`sizes:[64,36]`，`minSize:[320,220]`，gutter 上下拖 → dock 高度可调）。
- 主区内层 = **horizontal split**（`panes:['#axis-compute','#axis-physical']`，`sizes:[50,50]`，`minSize:[420,420]`，中缝左右拖）。用 `initNestedResizablePanes`。
- pane 视觉由产品侧给：`--surface-2` + `1px solid --border-subtle` + `--radius-md`；内部内容块 recessed `--surface-1`。gutter 把手用 pattern `::before`，不自造。
- `storageKey` 持久化比例。**默认左右等宽**（中缝可拖）。

---

## 3. 计算轴：718B MoE 架构图（本版重头戏 · 整体重建）

> 复用 `model-training-graphviz` 渲染引擎 + 交互基建（`controller.selectNode`/`data-node-id`/折叠展开/证据面板/provenance/主题）。**只重写图数据 JSON**，从旧拓扑整体换成 718B MoE。
> 新文件：`patterns/.../openpangu_ultramoe_718b_architecture_graphviz.html`（不覆盖现有 7B-Dense 文件，保留为对照样例）。

### 3.1 拓扑骨架

```
输入(Token IDs / Embeds / Position / Mask)
  → Embedding → RoPE
  ├─[Dense 解码层 ×3]              ← first_k_dense_replace=3，前 3 层无路由
  │     MLA 注意力 → Dense MLP(SwiGLU)
  ├─[MoE 解码层 ×58  ▼默认折叠]    ← 出事/选中时展开内部
  │     MLA 注意力(q_lora 1536 / kv_lora 512)
  │     → Pre-RMSNorm
  │     → Gate 路由器(top-8, routed_scaling 2.5, norm_topk_prob)
  │     → All-to-All 分发（通信事件，投影到物理轴 overlay）
  │     → 256 路由专家【聚合成 32 个 runtime EP buckets，每桶 8 专家】
  │       ＋ 共享专家(单独画 · 恒亮 · 永不黑)
  │     → All-to-All 汇聚（通信事件，投影到物理轴 overlay）
  │     → 残差(sandwich_norm)
  └─ Final RMSNorm → LM Head → Logits
       ＋ MTP 头 ×1(num_nextn_predict_layers=1，尾部多 token 预测)
```

### 3.2 cluster / node / edge 清单（图数据规格）

**clusters**（带 `repeat:"× N"` 折叠标签）：
| id | label | repeat | colorKey |
|---|---|---|---|
| `wrapper` | PanguUltraMoEForCausalLM | — | neutral |
| `dense_layer` | Dense 解码层 | × 3 | layer-dense |
| `moe_layer` | MoE 解码层 | × 58 | layer-moe |
| `mla` | MLA 注意力 | — | attention |
| `moe_block` | MoE 块（Gate→A2A→专家→Combine） | — | moe |
| `mtp_head` | MTP 头 | × 1 | head |

**nodes**（关键，`kind:op|tensor`）：
- 输入/嵌入：`token_ids` `input_embeds` `position_ids` `attn_mask` `embedding` `rope`
- MLA：`q_lora_down`(1536) `q_up` `kv_lora_down`(512) `kv_up` `rope_apply` `attn_scores` `softmax` `attn_ctx` `attn_out`
- MoE 块：`moe_prenorm` `gate`(top-8) `routed_scaling` `a2a_dispatch` `expert_group_00…expert_group_31`（32 个聚合组）`shared_expert`（单独·恒亮）`a2a_combine` `moe_residual`
- 尾部：`final_norm` `lm_head` `logits` `mtp_norm` `mtp_head` `mtp_logits`

**edges**（`edgeType` 决定颜色 —— 这是桥的视觉关键）：
| edgeType | 用途 | 视觉 |
|---|---|---|
| `activation` | 前向数据流 | 默认细灰 |
| `parameter` | 权重边（W_gate / W_expert） | 紫 |
| `gradient` | 梯度回溯边 | 橙 |
| `communication` | **All-to-All 分发/汇聚** | **专门通信色**（只在计算轴内部表达通信语义；不画跨轴连线） |

### 3.3 三个表达决策（已定）

1. **256 routed experts 聚合成 32 个 runtime EP buckets**（demo 中每桶 8 专家）→ 这是运行时放置桶，不等同于模型架构里只有 32 个专家，也不天然等同于 32 张卡。
2. **共享专家单独画、恒亮**——作对照组，一眼看出坏的只是 256 路由专家。
3. **通信边专色**——All-to-All 边在计算轴内部单独配色，区别于权重边/梯度边；跨到物理轴时使用 hover-first 高亮和通信 overlay，select 仅作锁定兜底，不使用 gutter 连线。

### 3.4 关系钩子（archive 缺失，本版新增）

每个 `expert_group_NN` 节点写 `data-ep-bucket="NN"`。选中专家桶 → 物理轴点亮当前 placement 中承载该 bucket 的 rank/device；选中 910B 卡或 rank → 计算轴点亮它承载的 expert bucket / weight shard / communication event。引擎在 `pattern.js` 渲染时已把 `node.id` 写进 `el.dataset.nodeId`，关系映射另建 `placementMap.byNode/byRank/byDevice` 索引（§8/§9）。

### 3.5 折叠/展开行为
- 默认：61 层折叠为 `Dense×3` + `MoE×58` + `MTP×1` 三个折叠簇，主干清爽。
- 偏差雷达告警某 MoE 层 / 用户选中 → 该 MoE 层 cluster 展开，露出 Gate→A2A→专家→Combine 内部；其余仍折叠。
- `controller.selectNode(id,{relatedNodeIds})` 只改强调，**绝不移动视口**。

### 3.6 trainingEvidence（挂在关键节点）
`gate` / `a2a_dispatch` / `expert_group_NN` 上挂证据：`{dimension, metric, what, evidence[], action, relatedNodeIds[], sources[]}`，接路由坍缩故事（负载均衡损失骤降 → 部分专家被饿死 → 对应卡变黑洞）。`provenance` 溯源到真实 openPangu `config.json`。

---

## 4. 物理轴：Ascend 910B Runtime Placement

物理轴以 **Ascend 910B 卡**为稳定底图，rank 和并行策略是运行时叠加层：

```text
910B Card = 物理实体
device_id / host / slot = 设备坐标
rank = 分布式进程 / 通信成员
TP / PP / CP / DP / EP = 该 rank 在训练策略里的坐标
```

### 4.1 三层渲染

1. **Hardware Base Layer**：910B card grid，显示 host、slot、device_id、HBM、util。demo 可先画 32 张 910B 卡作为当前诊断窗口，不声称这是完整物理全集。
2. **Rank Placement Layer**：每张卡贴 `global_rank / local_rank / TP / PP / CP / DP / EP`；默认 demo 可一 rank 一卡，但 schema 支持一机多卡、多 rank 复用或 rank 迁移。
3. **Comm Overlay Layer**：按当前 `step + phase + lens` 叠加通信流；只画在物理轴内，不画计算轴到物理轴的线。

### 4.2 TP / PP / CP / DP / EP 是 lens，不是五张同构网格

- **DP lens**：高亮 replica group，显示 global loss 聚合、gradient all-reduce、慢 replica。
- **TP lens**：高亮同一层 tensor shard group，显示 matmul shard 汇合、reduce-scatter/all-reduce。
- **PP lens**：按模型深度 stage 分区，显示 microbatch 气泡、send/recv stage boundary。
- **CP lens**：按 sequence/context shard 分段，显示 context exchange / all-gather。
- **EP lens**：显示 expert placement、token dispatch、All-to-All inflow/outflow、专家桶黑洞。

### 4.3 通信 overlay 规则

- `all2all`：EP/MoE token dispatch，默认只画聚合流与 TOP-K 异常边；hover 才展开源/目标 rank 细边。
- `allreduce`：DP/TP 梯度或 shard 聚合，用 group 边界 + pulse 表达同步。
- `p2p`：PP stage 间 send/recv，用相邻 stage 单向箭头 + microbatch bubble 表达。
- `allgather/reducescatter`：CP/TP shard 交换，用 shard band 表达。

**节点色** = 当前 lens 的主指标（util / local loss contribution / HBM / token inflow）；**overlay 边宽 + 透明度** = bytes 或 tokens；**动画/脉冲**只用于当前 phase，不常驻。

---

## 5. Step / Phase / 桥机制：让多个投影对上号

**两根支柱**（一切联动从此派生）：
1. **统一 step 轴**（时间对齐）：底部 realtime loss 的 cursor 是全局时间锚点，框选一次 = 所有视图聚焦同一窗口。
2. **Step Execution Rail**（计算相位对齐）：每个 step 拆成 `Batch → Forward → Loss → Backward → Grad Sync → Optimizer Update`；**按这个顺序执行完一次就是一个训练 step**。hover 某 phase 时，计算轴高亮对应执行路径，物理轴切换对应通信 overlay。
3. **hover 即双向点亮**（对象对齐）：hover 任一对象，两轴其余部分自动点亮关联；click 只作为可选锁定/键盘兜底，不作为默认理解路径。

**映射表**：`placementMap.byNode`（nodeId ↔ expert bucket ↔ rankIds ↔ weightShard）+ `placementMap.byRank`（rankId ↔ deviceId ↔ TP/PP/CP/DP/EP ↔ nodeIds）+ `placementMap.byDevice`（910B card ↔ rankIds）。不再假设 expert bucket ↔ rank ↔ card 永远 1:1，只把 demo 当前 placement 画出来。

**一个故障，四种说法（产品要自动对上号）**：loss=曲线异常点　逻辑=路由坍缩/专家饿死　通信=All-to-All 入流黑洞　物理=某 910B card/rank util 与 token inflow 同步下降。

---

## 6. 实时监护能力（新增构件）

### C0 Step Execution Rail（计算逻辑解释层）
- 把 `step` 从一个数字拆成可播放的训练相位：`Batch / Forward / Loss / Backward / Grad Sync / Optimizer Update`。
- 每个 phase 必须有 hover tip，明确说明：`Batch → Forward → Loss → Backward → Grad Sync → Optimizer Update` 按顺序执行一次 = 一个 step。
- 每个 phase 绑定 `nodePath[]`：Forward 高亮 activation 路径，Backward 高亮 gradient 路径，Update 高亮 parameter/update 节点。
- loss 异常点 hover 后自动定位到最可能 phase，例如 MoE 路由异常定位到 `Forward · Gate → A2A Dispatch → Expert Bucket`。

### C1 Realtime Loss Console（底部 dock 主体）
- 主图复用 `training-metrics-chart`：`train_loss / val_loss / expected_loss / previous_run_loss / spike_threshold`，支持 cursor、异常点、brush 兴趣窗口。
- 辅助前导指标：`eval_mmlu / grad_norm / load_balance_loss / router_z_loss / expert_load_var`。eval 不再混在主 loss 图里。
- per-rank 不铺 32 张小图；默认显示 rank loss contribution 分布带，hover 某 rank 时叠加该 rank local loss overlay。

### C2 偏差雷达
- 把当前 step「偏差最大」的对象（层/卡/链路/专家组）自动排序、置顶、告警。
- **「偏差大」度量（可解释，不上黑盒）**：候选信号 = 指标对滚动 EMA/方差带的 z-score、梯度范数突变、负载均衡损失骤降、某 rank 流量/利用率对同组均值偏离。MVP 用阈值/基线规则。
- **前导指标**：grad norm、负载均衡损失、router z-loss、专家负载方差——常先于 loss 崩动；雷达盯"谁先偏、偏得最大"。

### C3 内嵌根因 tag / hover tip
- 不再做右下角独立根因模块；一句自然语言根因 + 证据链融入对象 tag、phase tip、loss tooltip。
- 监控→诊断交接：hover loss 越带 → 哪个前导指标先偏 → 指向轴（负载均衡损失骤降=逻辑轴 Gate；某 PP stage grad 尖峰=物理轴）。

### C4 调整建议（对象 hover tip 内 · 就地展开）
- 给可操作下一步：隔离某 rank / 跳过某 step / 调 lr / 重平衡专家。MVP 规则化，二期智能化，不占用底部 loss console。

---

## 7. 事故回放模式（辅 · 沙盒隔离）

> **硬隔离规则（最高优先）**：本节是一个**插件式的可选样本**，**不得反向约束 §1–§6 的实时监护设计**。两轴布局、718B MoE 架构图、偏差雷达、根因/建议、桥——这些的形态由"实时监护"独立决定，**绝不为了兼容旧七步而妥协**。旧七步的"预置答案/展览馆"性质已被否决，只有它的**零件**（视图构件、联动基建）和**回放样本价值**被收编。

回放模式与实时模式**共用同一套已建好的双轴视图与 step 轴**，仅切换数据源（历史快照 vs 实时流）。它作为一个独立数据包载入本地归档旧 SPEC 的七步因果链——是这套视图的一个"可重复演示样本"，而非另一套界面。**实现上排在最后（§11 step 6），且砍掉它不影响 step 1–5 的任何交付。** 七步脚本见本地 archive §4。

---

## 8. 数据模型（schema · 在 archive 基础上扩展）

```jsonc
// 1) model graph —— model-training-graphviz 运行时 schema（718B MoE 拓扑，§3）
{ "width","height",
  "clusters":[{ "id","label","x","y","width","height","colorKey","repeat":"× N" }],
  "nodes":   [{ "id","label","typeLabel","kind","x","y","width","height","colorKey",
                "epBucket":"00..31"  /* 仅 expert_group_NN，关系钩子 */ }],
  "edges":   [{ "source","target","tag","edgeType":"activation|parameter|gradient|communication" }],
  "trainingEvidence": { "<nodeId>": { "dimension","metric","what","evidence":[],"action","relatedNodeIds":[],"sources":[] } },
  "provenance": { "config":"openPangu config.json", "values":{...} } }

// 2) stepTrace —— step 数字到计算逻辑的可解释关系
{ "byStep":{ "<step>":{ "activePhase":"forward|loss|backward|sync|update",
    "phases":[{ "id":"forward", "label":"Forward",
      "nodePath":["embedding","moe_prenorm","gate","a2a_dispatch","expert_group_19","a2a_combine","lm_head"],
      "commPrimitive":"all2all", "physicalLens":"ep",
      "metrics":{ "gate":{"router_z_loss":3.2}, "expert_group_19":{"token_inflow":0.12} } },
    { "id":"loss", "nodePath":["logits","loss"], "metrics":{"global_loss":4.83} },
    { "id":"backward", "nodePath":["lm_head","a2a_combine","expert_group_19","gate"], "metrics":{"grad_norm":2.7} },
    { "id":"sync", "nodePath":["a2a_combine"], "commPrimitive":"allreduce", "physicalLens":"dp" },
    { "id":"update", "nodePath":["gate_weight","expert_weight_19"], "metrics":{"update_ratio":0.018} }] } } }

// 3) timeseries —— realtime loss + 前导指标 + per-rank overlay
{ "steps":[...],
  "series":{ "train_loss":[],"val_loss":[],"eval_mmlu":[],
    "grad_norm":[],"load_balance_loss":[],"router_z_loss":[],"expert_load_var":[] },
  "rankSeries":{ "<rankId>":{ "local_loss":[],"loss_contribution":[] } },
  "baseline":{ "<series>":{ "ema":[],"band":[lo,hi] } },
  "anomalies":[{ "step", "seriesId", "phaseId", "nodeIds":[], "rankIds":[] }] }

// 4) physicalTopology + rankPlacement —— 910B 物理底图与运行时放置
{ "cards":[{ "cardId":"card_019","kind":"Ascend 910B","hostId","slot","deviceId",
    "hbmGb":64,"rankIds":["rank_19"] }],
  "ranks":[{ "rankId":"rank_19","globalRank":19,"localRank":3,"cardId":"card_019",
    "coords":{"tp":3,"pp":1,"cp":0,"dp":2,"ep":19} }] }

// 5) commSnapshots —— 每 step/phase 的通信 overlay
{ "byStep":{ "<step>":{ "<phaseId>":{ "primitive":"all2all|allreduce|p2p|allgather|reducescatter",
    "parallelAxis":"ep|dp|tp|pp|cp",
    "affectedNodeId":"a2a_dispatch",
    "flows":[{ "srcRank","dstRank","bytes","tokens","latencyMs","anomaly":"blackhole|slow|none" }],
    "rankStats":{ "<rankId>":{ "util","hbm","inflowBytes","outflowBytes","localLoss" } } } } } }

// 6) weightDetail —— 每节点权重/shape 详情
{ "<nodeId>":{ "normal":{ "shape":[...],"hist":[...] },
   "anomaly":{ "step","shape":[...],"hist":[...],"note" },
   "routingHeatmap":{ "rows","cols","matrix":[[...]] } } }

// 7) deviationRadar —— 当前 step TOP-N 偏差对象（派生，可预算或实时算）
[ { "objectType":"expert_group|rank|edge|layer","id","epRank","zScore","signal","leadIndicator" } ]

// 8) placementMap —— 对象↔rank↔device↔weight，而不是跨轴视觉连线
{ "byNode":{ "<nodeId>":{ "relatedNodeIds":[],"rankIds":[],"cardIds":[],"weightShards":[] } },
  "byRank":{ "<rankId>":{ "cardId","nodeIds":[],"expertBuckets":[],"coords":{} } },
  "byDevice":{ "<cardId>":{ "rankIds":[],"hostId","slot" } } }
```

**接入层（让"模型对象可换"成立）**：各源写 adapter（wandb / tensorboard / msprof / HCCL trace / ckpt dump）归一化进上述统一 schema，视图只认 schema。换模型 = 换 adapter + 换拓扑 JSON，视图层不动。逻辑轴首发推荐 **OLMoE**（路由/负载均衡真实逐步曲线）；物理轴短期自采或合成（§附录）。

---

## 9. 联动机制（事件总线）

轻量事件总线（借 workbench `GEW.bus` 思路自建，不接其 trace），七类广播：
1. `interestWindow:{stepStart,stepEnd}`——底部框选 → 各视图按 step 聚焦。
2. `hoverState:{objectType,id,relatedNodeIds,rankIds,cardIds,stepCursor,phaseId}`——任一对象 hover → 两轴点亮。关系解析复用 `relationForNode`：自身 ∪ relatedNodeIds ∪ evidence.relatedNodeIds ∪ 图上直连 ∪ **placementMap**。
3. `select:{objectType,id,...}`——仅用于用户显式锁定或键盘可访问性；不作为默认联动入口。
4. `stepCursor`——底部 loss chart hover/brush 驱动，stepTrace、计算轴、物理轴 overlay、曲线 cursor 同步。
5. `phaseCursor:{step,phaseId}`——Step Execution Rail hover 驱动，计算轴高亮 nodePath，物理轴切换 comm overlay。
6. `commHover:{flowId,srcRank,dstRank,primitive}`——物理轴通信流 hover → 高亮 rank/card/nodePath。
7. `deviation:{step,topN}`——偏差雷达广播当前 step TOP-N，置顶告警，hover 触发联动，click 可锁定。

架构图联动调用 `controller.selectNode(id,{relatedNodeIds,source})` / `controller.setPhase(...)`，**只改强调，绝不移动视口**。

---

## 10. 技术栈 & 文件结构

- 纯原生 HTML/CSS/JS，无框架、无外部 UI 库、无 CDN。分栏用 PtoWorkbenchShell，不引 split.js。
- **Token 顺序**：`foundation → semantic → components → style`。
- **Pattern 顺序**：`model-graphviz → model-training-graphviz → workbench-shell → training-metrics-chart`。
- 颜色/间距/字体/圆角一律 `var(--*)`；viz 色带仅用于图/边/热图；优先级用 fill（P0 红/P1 橙/P2 黄，P2 不用蓝）；至多一层可见边界，图节点无阴影；不卡中卡、不左侧描边条。
- 文件结构（现状 + 新增）：
  ```
  pangu-moe-trainviz/
    index.html
    css/app.css
    js/{bus,graph-view,timeline,param-rail,inspector,comm-dock,info,app}.js
    js/{deviation-radar,rootcause,placement,step-trace}.js # 新增 C1/C3/关系层
    data/{graph,timeseries,comm,weight}.js
    data/{graph-ultramoe-718b,deviation,placement,step-trace}.js # 新增 718B 拓扑 + 派生
    data/adapters/{olmoe,bloom,msprof}.js             # 新增接入层
  ```

---

## 11. MVP 构建顺序 & 验收

**构建顺序**（增量，每步可在 localhost 验收）：
1. **左·计算轴 718B MoE 架构图**（§3）——先单独做对：61 层折叠、MoE 层展开、32 runtime EP buckets、共享专家单独、通信边专色。
2. **Step Execution Rail + realtime loss console**（§5/§6）——复用 `training-metrics-chart`，hover loss 异常点能定位 step + phase + nodePath。
3. **右·Ascend 910B 物理轴**（§4）——910B card base + rank placement + TP/PP/CP/DP/EP lens。
4. **通信 overlay**（§4.3/§8）——All-to-All / AllReduce / P2P 等叠在物理轴内部，不画跨轴线。
5. **关系联动**（§5/§9）：hover 专家桶/rank/card/flow ↔ 两轴点亮，click 只作为可选锁定兜底，统一 step/phase。
6. **数据接入层**：接通 1 个真实逻辑轴源（OLMoE/BLOOM）+ 1 套物理轴（自采/合成，清晰标注 synthetic physical telemetry）。
7. **事故回放模式**：载入 archive 七步资产。

**验收标准**：
1. 训练正在跑时，偏差雷达自动排出 TOP-N 偏差对象，无需翻日志。
2. 选中任一对象，计算轴与物理轴**两投影同时点亮、能对上号**（model object ↔ rank placement ↔ 910B card ↔ comm flow）。
3. 根因以**一句自然语言 + 证据链**呈现，非堆栈。
4. 每个已识别异常给**至少一条可操作调整建议**。
5. 至少接通**一个真实逻辑轴源**（OLMoE/BLOOM），对象可经 adapter 替换。
6. 事故回放模式下，archive 七步因果链 1:1 可用。
7. 视觉 100% 走 PTO token，至多一层可见边界，图节点无阴影。

---

## 12. 风险与开放项

1. **「偏差大」度量定义**（§6 C1）：阈值/基线规则需和算法/Infra 共定，必须可解释，不上黑盒。— 最高优先。
2. **物理轴真实数据缺口**：逐 rank 通信 trace 开源拿不到，MVP 物理轴大概率自采或合成。— 需定首发口径。
3. **实时性工程代价**：真·流式接入需训练侧埋点 + 近实时管道；一期落不了可先做近实时轮询过渡。
4. **图数据规模**：61 层 × 256 专家若全展开会糊屏——默认折叠 + 32 组聚合是必须，不是可选。
5. **证据可靠性**：用研 13 条 insight 全 `unverified`，重投入前对核心支撑页人审。

---

## 附录：接入对象真实规格（openPangu-Ultra-MoE-718B）

**架构**：61 层（前 `first_k_dense_replace=3` dense + 58 MoE）+ 1 MTP 头；hidden 7680，intermediate 18432，moe_intermediate 2048。
**MoE**：256 路由专家 + 1 共享专家，top-8 激活，`routed_scaling_factor=2.5`，`norm_topk_prob=true`。共享专家恒激活 → 路由坍缩只影响 256 路由专家。
**MLA**：128 头，`q_lora_rank=1536`，`kv_lora_rank=512`，`qk_nope=128`+`qk_rope=64`，`v_head_dim=128`。
**其它**：vocab 153600，max_position 131072，`sandwich_norm=true`，`rope_theta=2.56e7`。
**规模/训练**：718B 总 / 39B 活；19T tokens；4000×昇腾 910B（64GB HBM），MFU 18.9%，0.61M tok/s。

**并行/渲染网格**：物理卡总数由训练策略与 placement 共同决定，常见表达可写成 `N = TP × PP × CP × DP`，MoE 专家并行 `EP` 通常作为 DP 维内的专家放置维度。demo 页面可渲染 32 张 Ascend 910B 作为当前诊断窗口，显示 `rank → device → TP/PP/CP/DP/EP coords → expert bucket` 的运行时映射；这不是“1 个 EP 组天然等于 32 张卡”的定义。推荐配置示例可保留为配置说明：`TP=8 × PP=16 × CP=1 × DP=32`，`EP=32`，总规模约 4096 卡，每卡承载若干层/张量分片/专家桶，具体以 `rankPlacement` 为准。

术语大白话见 `../product/PRD-训练透视.md` 附录 C（256 人会诊中心比喻）。

---

*配套：`../product/PRD-训练透视.md`（产品需求）、本地归档的旧 SPEC（事故回放七步剧本，本版 §7 内容载体）、`design-brief-昇腾CANN.md`（用研约束）、`INSIGHT-ANALYSIS.md`（现状对标）。*
