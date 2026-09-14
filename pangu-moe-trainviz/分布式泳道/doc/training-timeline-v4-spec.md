# training-timeline-v4-spec.md

> **训练时序 v4 规格说明**  
> 状态：Draft · 基于 92B Omni MoE (256 expert), stage2, seq 32K, 128 ranks  
> 更新：2026-09-14

---

## 1. 并行配置

| 维度 | 值 | 说明 |
|------|-----|------|
| world_size | 128 | 总 rank 数 |
| TP | 4 | 张量并行 |
| CP | 2 | 上下文并行（Ulysses Ring Attention） |
| PP | 1 | 流水线并行（不启用） |
| EP | 16 | 专家并行 |
| DP (dense) | 16 | Dense 路径数据并行 |
| EDP (expert) | 2 | 专家路径数据并行 |
| ETP | 4 | 专家 TP（moe-tp-extend-ep 开启时 TP×EP 共同切分） |
| SP | 开启 | 序列并行 |
| TokenDispatcher | pangu_dropless | All-to-All token 调度 |
| Expert Placement | fine-grained, threshold=0.08 | 动态专家放置 |
| MBS | 4 | micro batch size |
| Seq | 32768 | 序列长度 |

### 并行网格

两套并行网格在同一组 rank 上共存：

**Dense 路径（Attention / Dense FFN / LayerNorm）：**
```
world_size = TP × CP × PP × DP
      128  =  4  ×  2  ×  1  ×  16
```

**专家路径（MoE Expert 参数）：**
```
world_size = ETP × EP × PP × EDP
      128  =  4  × 16 ×  1  ×   2
```

**每 rank expert 数：**
```
num_experts_per_rank = total_experts ÷ (TP × EP) = 256 ÷ (4 × 16) = 4
```

---

## 2. Rank 映射

Rank 坐标公式（TP=4, CP=2, DP=16, PP=1）：

```
rank = dp_idx × (TP × CP) + cp_idx × TP + tp_idx
     = dp_idx × 8 + cp_idx × 4 + tp_idx
```

| dp_idx | cp_idx | tp_idx | rank |
|--------|--------|--------|------|
| 0 | 0 | 0..3 | 0,1,2,3 |
| 0 | 1 | 0..3 | 4,5,6,7 |
| 1 | 0 | 0..3 | 8,9,10,11 |
| 1 | 1 | 0..3 | 12,13,14,15 |
| ... | ... | ... | ... |
| 15 | 0 | 0..3 | 120,121,122,123 |
| 15 | 1 | 0..3 | 124,125,126,127 |

---

## 3. 通信域划分

### TP Group (size=4)
每 4 个连续 rank 组成一个 TP group，共 32 组。
- TP0: `[0,1,2,3]`, TP1: `[4,5,6,7]`, ..., TP31: `[124,125,126,127]`
- 通信原语：AllGather, ReduceScatter (MC2 fused)

### CP Group (size=2)
每个 DP replica 内，cp_idx=0 和 cp_idx=1 各一组。
- CP0 (all DP): `[0,1,2,3, 8,9,10,11, ..., 120,121,122,123]`
- CP1 (all DP): `[4,5,6,7, 12,13,14,15, ..., 124,125,126,127]`
- 通信原语：All2All (Ulysses)

### EP Group (size=16, inter-DP)
EP Group 跨 DP replica 组建，从每个 DP replica 的相同 tp_idx/cp_idx 取 1 个 rank。
共 16 个 EP Group，每个包含 16 个 rank（来自 16 个 DP replica）。
- 通信原语：All2Allv (pangu_dropless)

### DP Group (size=16)
相同 TP/CP 坐标的所有 rank（跨 16 个 DP replica）。
- 通信原语：ReduceScatter (Distributed Optimizer)

### EDP Group (size=2)
每 8 个 DP replica 为一组，共 2 个 EDP group。
- 通信原语：ReduceScatter

---

## 4. EP Group 成员（跨 DP 的显式映射）

EP=16 时，每个 EP Group 从每个 DP replica 取一个 rank。由于 ETP=4，EP Group 的构建方式为：

一个 EP Group 包含：
- 从 16 个 DP replica 中各取 1 个 rank
- 每个 rank 的 tp_idx 和 cp_idx 在组内固定

**EP Group 0 成员（全量示例）：**

| dp_idx | cp_idx | tp_idx | globalRank | epRank |
|--------|--------|--------|------------|--------|
| 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 8 | 1 |
| 2 | 0 | 0 | 16 | 2 |
| 3 | 0 | 0 | 24 | 3 |
| 4 | 0 | 0 | 32 | 4 |
| 5 | 0 | 0 | 40 | 5 |
| 6 | 0 | 0 | 48 | 6 |
| 7 | 0 | 0 | 56 | 7 |
| 8 |  MAR |   |   |   |
| 9 |  MAR |   |   |   |
| ... | ... | ... | ... | ... |
| 15 | 0 | 0 | 120 | 15 |

> 注：EP Group 的成员必须由配置显式提供，所有 EP Group 的完整 128 rank 坐标在 mock-profile-v1.json `epGroups` 字段中。

---

## 5. 调度时序（单 step，无 PP）

### 宏观调度

基于 1F1B 通信掩盖特性说明书的 No-PP 场景：

```
Step Start
  ├── zero_grad_buffer()
  │
  ├── [MB0_FWD]  ← forward + 隐藏 EP a2a 的 dispatching
  ├── [MB1_FWD]  ← 同时 MB0 开始 backward
  │     └── MB0_BWD 的 attention/dense 计算掩盖 MB1 的 EP a2a 通信
  ├── [MB2_FWD]
  │     └── MB1_BWD 掩盖 MB2 的 EP a2a 通信
  ├── [MB3_FWD]
  │     └── MB2_BWD 掩盖 MB3 的 EP a2a 通信
  ├── MB3_BWD ← 最后一个 microbatch 触发梯度同步
  │
  ├── finalize_model_grads_func() ← DP reduce-scatter 完成
  └── optimizer.step()
```

### 单 MB 粒度时序（一层展开）

每个 MB 内部对每个 Transformer Layer，时序分 [Dense Part | MoE Part]：

**Forward：**
```
┌────────── Dense Part ──────────┬────── MoE Part ──────┐
│ Input Norm │ Attention │ Attn  │ Router│ A2A  │ Expert │
│ (RMSNorm)  │ (MLA SWA) │ Resid │ (Gate)│Disp. │ GMM   │
└────────────────────────────────┴──────────────────────┘
```

- Dense→MoE 通信：TP all-gather (MC2 fused) 与 attention 计算重叠
- MoE All-to-All：pangu_dropless token dispatch，与下一层 dense 计算重叠

**Backward：**
```
┌────── MoE Part ──────┬────────── Dense Part ──────────┐
│ Expert │ A2A  │ Router│ Attn     │Attn   │ Input      │
│ Grad   │Unperm│ Grad  │ Grad     │Resid  │ Norm Grad  │
│ (GMM)  │      │       │ (MLA SWA)│ Grad   │ (RMSNorm)  │
└──────────────────────┴────────────────────────────────┘
```

- Backward 中 A2A permute/unpermute 通信被 backward dense 计算掩盖
- TP reduce-scatter (MC2 fused) 与 MLP backward 计算重叠

---

## 6. 通信原语总表

| 通信域 | 规模 | 参与方 | 通信原语 | 方向 |
|--------|------|--------|----------|------|
| TP Group | 4 | 每 4 个连续 rank | AllGather | FWD: Dense→MoE |
| TP Group | 4 | 每 4 个连续 rank | ReduceScatter (MC2 fused) | BWD: MoE→Dense |
| CP Group | 2 | rank_i, rank_i+2 | All2All (Ulysses) | FWD/BWD |
| EP Group | 16 | EP 内所有 rank | All2Allv (pangu_dropless) | FWD: Dispatch / BWD: Combine |
| DP Group | 16 | 相同 TP/CP 的所有 rank | ReduceScatter | BWD: 梯度同步 |
| EDP Group | 2 | 每 2 个 EDP rank | ReduceScatter | BWD: 专家梯度同步 |

---

## 7. Mock Profile 数据契约

详见 `mock-profile/profile-schema.json` 和 `mock-profile/mock-profile-v1.json`。

### 事件字段要求

每个事件必须包含：
- `id`: 唯一标识
- `rank`: global rank
- `layer`: 层号
- `microbatchId`: MB00–MB03
- `phase`: forward / backward
- `kind`: compute / comm / wait / optimizer / misc
- `commPrimitive`（通信事件）: all-to-all / all-gather / reduce-scatter / all-to-all-v
- `commPhase`（EP 通信事件）: dispatch / combine
- `startUs` / `endUs`: 微秒时间戳
- `sendTokens` / `recvTokens`（通信事件）: token 数量
- `waitDurationUs`（通信事件）: 等待耗时
- `status`: normal / warning / critical
- `diagnosisRole`（异常事件）: root-cause / symptom / affected / context
- `epGroupId`（EP 相关事件）: EP 组 ID
- `epRank`（EP 相关事件）: EP 组内坐标

### Incident 字段

```json
{
  "focus": {
    "layer": 38,
    "microbatchId": "MB03",
    "epGroupId": "ep-group/pp0/dp0",
    "eventIds": ["event/dispatch/l38/mb03/r151"],
    "rankIds": [151],
    "nodeIds": ["router_gate", "routed_expert_bank"]
  }
}
```

### Router Metrics 字段

```json
{
  "layer": 38,
  "microbatchId": "MB03",
  "expertLoads": [{ "expertId": 193, "tokenCount": 8028, "tokenShare": 0.98 }],
  "totalTokens": 8192,
  "top1TokenCount": 8028,
  "deadExpertCount": 247
}
```

---

## 8. Expert Placement

配置：`fine-grained, threshold=0.08, freq=500`

- 256 个 logical expert 分布到 16 个 EP Group
- 每个 EP Group 持有 256 ÷ 16 = 16 个 expert
- 每个 EP Group 内有 8 个 EP Rank（EDP=2 将 16 个 DP replica 分为两组，每组 8 个）
- 每 rank 存放 2 个路由专家（16 experts ÷ 8 ranks）
- 动态放置阈值 0.08：负载偏差超 8% 即触发迁移

### 事故 Expert 放置

| expertId | epGroupId | ownerGlobalRank | epRank | 说明 |
|----------|-----------|-----------------|--------|------|
| E193 | ep-group/pp0/dp0 | 14 | 14 | 热点 expert，承接 98% token |
| E000–E192, E194–E255 | 各 EP Group | 分布 | 分布 | dead experts (247/255 个) |


---

## 10. 参考文档

- 并行配置来源: `config/omni/92BOmni_MOE/PILOT/92BOMNI_MOE_stage2/`
- 1F1B 通信掩盖: `doc/1F1B通信掩盖特性说明书.md`
- v3 规格: `training-structure-time-v3-moe-incident-spec.md`
- JSON Schema: `mock-profile/profile-schema.json`
- Mock Profile: `mock-profile/mock-profile-v1.json`
