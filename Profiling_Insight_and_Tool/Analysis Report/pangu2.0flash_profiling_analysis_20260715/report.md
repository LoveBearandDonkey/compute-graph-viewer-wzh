# Pangu 2.0 flash 72B 多机多卡训练性能诊断报告（4 节点 × 8 卡 / TP1·PP4·EP4·DP2）

## 1. 结论速览

- **性能健康度**：73 / 100 (A) → 优化后预估 **86 / 100 (A)** — 计算 58% · 通信 91% · 调度 78% · 内存 N/A
- **结论**：单步 ~16.20 s，32 卡 step 时间高度一致（CV≈0.02%），但存在三重瓶颈——**PP 末级计算过载 + EP all-to-all 暴露通信 + MoGE 路由负载不均**。
- **头号瓶颈**：PP stage3（rank24–31）计算 ~12.70 s/step，是其余各级（~8.29 s）的 **1.54×**（多 ~4.41 s）；stage0–2 的 24 卡因此每步空等 ~4.26 s 流水线 bubble（占 step **~26%**）。
- **二号瓶颈**：EP=4 的 MoE all-to-all（token dispatch/combine）**完全暴露在关键路径**（Overlapped=0），44 层 MoE 累计约 **1.32 s/step** 纯通信开销，未被任何计算掩盖。
- **三号瓶颈**：MoGE 分组路由在深层（L30+）出现 **expert 负载倾斜**，高峰 expert 接收 6–8× 于低峰 expert，造成 EP 组内算力浪费与 all-to-all buffer 不均。
- **收益上限**：拉平 stage3 负载 + 开启 EP 通信重叠 + 路由均衡优化，预计节省 **~30–35% 单步耗时（~4.9–5.7 s）**。

## 2. 行动清单

> 默认按"预期收益"从高到低排序；同档收益按"修改难度"从低到高排序。

| # | 优先级 | 问题 | 预期收益 | 修改难度 |
|---|---|---|---|---|
| 1 | P0 | PP 末级（stage3, rank24–31）计算过载，制造 ~26% 单步 bubble | -20~25% 单步耗时（~3.2–4.1 s） | 中 |
| 2 | P0 | EP all-to-all 通信入关键路径且零重叠，44 层累计 ~1.32 s/step | -6~8% 单步耗时（~1.0–1.3 s） | 中 |
| 3 | P0 | 计算-通信全局零重叠（Overlapped=0），DP allreduce + PP P2P 全部暴露 | -8~12% 单步耗时（叠加 #2 后） | 中 |
| 4 | P1 | MoGE 路由负载不均——深层 expert 热点，EP 组内算力浪费 | 中（降 EP 同步等待 + 提 MFU） | 中 |
| 5 | P1 | 关键环境变量未设置（缓存/显存分配器） | 中（降 host 下发开销、缓解碎片） | 低 |
| 6 | P1 | 动态 Shape 触发算子在线编译 | 低~中 | 低 |
| 7 | P2 | stage3 重载卡 AI Core 降频（rank24–31 低至 1200–1350 MHz） | 低 | 中 |

## 3. 问题详情

### 3.1 [P0] PP 末级（stage3, rank24–31）计算过载，制造 ~26% 单步 bubble

- **证据**（`cluster_time_summary` → `ClusterTimeSummary` 表，step 4，8 个代表 rank，单位换算为 ms）：

  | rank | 节点 | PP stage | EP 组 | DP 对 | 计算(ms) | 暴露通信(ms) | 其中 stage 等待(ms) | 实际传输(ms) | Free(ms) | stepTime(ms) | 计算占比 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | 0 | node1 | s0 (L0-11) | A | (0,4) | 8150.2 | 7800.5 | 4450.3 | 3350.2 | 249.3 | 16200.0 | 50.3% |
  | 4 | node1 | s0 (L0-11) | A | (0,4) | 8100.8 | 7850.1 | 4500.1 | 3350.0 | 249.1 | 16200.0 | 50.0% |
  | 8 | node2 | s1 (L12-23) | A | (8,12) | 8300.5 | 7650.3 | 4250.3 | 3400.0 | 249.2 | 16200.0 | 51.2% |
  | 12 | node2 | s1 (L12-23) | A | (8,12) | 8350.1 | 7600.8 | 4200.7 | 3400.1 | 249.1 | 16200.0 | 51.5% |
  | 16 | node3 | s2 (L24-35) | B | (16,20) | 8400.3 | 7500.5 | 4100.4 | 3400.1 | 299.2 | 16200.0 | 51.9% |
  | 20 | node3 | s2 (L24-35) | B | (16,20) | 8450.2 | 7450.7 | 4050.6 | 3400.1 | 299.1 | 16200.0 | 52.2% |
  | **24** | node4 | **s3 (L36-47)** | C | (24,28) | **12900.4** | 3000.3 | 500.1 | 2500.2 | 299.3 | 16200.0 | **79.6%** |
  | **28** | node4 | **s3 (L36-47)** | C | (24,28) | **12700.1** | 3200.6 | 700.5 | 2500.1 | 299.3 | 16200.0 | **78.4%** |

  - step 时间在 32 卡间高度一致（16200.0 ± 5 ms，CV≈0.02%）——典型的**全局同步训练**，不存在硬件型快/慢卡（无掉队 rank）。
  - 但 stage3（rank24–31）计算均值 **~12700 ms**，stage0–2 均值 **~8292 ms**——**末级多算 ~4408 ms/step（1.54×）**。
  - 末级"多出来的活"已定位到落盘算子（`compute_op_sum`/`kernel_details.csv`）：
    - **`MatMulV2`（lm_head 投影 [4608→153600]，平均 ~20.0 ms/次 × 64 次 ≈ 1280 ms）**仅出现在 stage3，rank0–23 完全没有。
    - Stage3 独有的 loss/MTP 辅助预测头（`num_mtp_layers=1`）反向 GEMM（小 M，`224×4608×153600` 等形状，复用 hidden→vocab 投影权重，~500 ms 级别）拉低整体 MatMul MFU。
    - Stage3 的 MoE shared expert 调用总耗时 ~2100 ms（4 个 shared expert 始终激活，每个 2×gate/up + down），是各 stage 固有开销，但 stage3 因 lm_head+loss 被挤到同一 device，时间叠加上升。
    - Stage3 vector 算子（RmsNorm/ElementWise）总耗时 ~1850 ms，是 stage0 的 ~1.5×——深层 sandwich-norm 三段归一化（input/post_attn/final）与 Sink Token 累积开销，随层深递增。
  - 后果：rank0–23 每步在 PP P2P recv 上**空等 `communicationWaitStageTime` ≈ 4.05–4.50 s（均值 4.26 s，占 step ~26%）**，而 stage3 几乎不等（0.50–0.70 s）。`hccl_sum` 佐证：stage0–2 的 `hcom_batchSendRecv_`（PP P2P）单卡 SumNs ~7.6–8.0 s/2step，但 Min 仅 ~1.3 ms、Max 达 120+ ms——绝大部分是**等待**而非传输。
- **影响**：单步 ~16.20 s 被末级计算 gating。全集群每步约 `24 卡 × 4.26 s ≈ 102.2` 卡·秒空耗在 bubble（≈ 集群总算力时间的 20%）。
- **修复建议**：
  - **改动位置**：训练启动并行切分配置（Megatron/MindSpeed 的 PP 层切分参数），目标——把末级 transformer 层数调少以抵消 lm_head+loss+深层 MoE 的额外开销，使各 stage 总耗时拉平。
  1. 采用**非均匀 PP 切分**：减少最后一级的 decoder 层数（Megatron `--decoder-last-pipeline-num-layers`），把末级 MoE 层从 12 层减至 9–10 层，差额 ~4.4 s 对应约 2–3 层的等效负载，让 stage3 计算从 ~12.70 s 降到 ~8.8 s。
  2. 或将 lm_head/loss 的计算量摊薄：开启 lm_head 的张量切分（当前 TP=1，可考虑对输出投影/词表做 TP=2 或 vocab-parallel cross-entropy），把 logits GEMM [4608→153600] 与 CE loss 拆到 stage3 的 2 卡。
  3. 复采时建议把 `num_microbatches` 从当前值提高至 8–16，并确认 1F1B 调度，进一步压低固有 bubble。
- **问题修改完成的验证方式**：重采后再跑 `msprof-analyze -m cluster_time_summary`，确认 `ClusterTimeSummary` 中 stage3 计算与 stage0–2 极差 < 10%，且 stage0–2 的 `communicationWaitStageTime` 从 ~4.26 s 降到 < 2.0 s；单步 stepTime 降到 ~11.5–12.5 s。
- **问题举证视图**：
  - 主：算子视图 — 载入 `evidence/rank24_s3_node4/kernel_details.csv`（源：`D:\Projects\ProfilingTest\pangu2.0flash\rank24_...\ASCEND_PROFILER_OUTPUT\kernel_details.csv`），按 Type 过滤 `MatMulV2`，确认 lm_head 投影 [4608→153600] 只在末级出现且单次 ~20 ms × 64 次；对照 `evidence/rank0_s0_node1/kernel_details.csv` 无该算子。
  - 辅：Timeline 视图（系统调优）— 载入 `evidence/rank0_s0_node1/trace_view.json`（源：`...\rank0_...\trace_view.json`），观察 P2P recv 处长达 ~4.5 s 的 device 空挡（bubble）。

### 3.2 [P0] EP all-to-all 通信入关键路径且零重叠，44 层 MoE 累计 ~1.32 s/step

- **证据**：EP=4 配置下，每层 MoE（L4–L47，共 44 层）在 forward 阶段执行 token dispatch all-to-all（将 token 按 router 决策发送到对应 expert 所在的 EP rank），backward 阶段执行 token combine all-to-all（回收 expert 输出梯度）。
  - `hccl_sum` → `HcclPerRankStats`：每 rank 的 `hcom_all_to_all_v_` 调用次数 = 44 层 × 2（fwd+bwd）= 88 次/step。
  - 单次 all-to-all 耗时（forward dispatch）：均值 **~18 ms**，合计 44 层 forward dispatch ≈ 792 ms。
  - 单次 all-to-all 耗时（backward combine）：均值 **~12 ms**，合计 44 层 backward combine ≈ 528 ms。
  - **EP 通信总开销 = 792 + 528 = 1320 ms/step**（≈ 实际传输 ~3350–3400 ms 中的 ~39%）。
  - `ClusterTimeSummary.communicationOverlapComputation` 全 32 卡为 0.0——all-to-all **完全暴露**，未与 MoE expert FFN 计算重叠。
- **影响**：每 step 约 1.32 s 纯 EP 通信落在关键路径上；若能将其与 MoE expert 计算（gate_proj/up_proj/down_proj）重叠，可隐藏 60–80% 的 all-to-all 耗时。
- **修复建议**：
  - **改动位置**：MindSpeed/Megatron MoE 层的前向/反向调度逻辑。
  1. 开启 **EP communication overlap**：在 dispatch all-to-all 发起后立即执行当前 rank 已有的 expert FFN 计算，不等待远端 token 到达；同理在 combine 阶段先发后算。
  2. 使用**独立通信 stream** 将 all-to-all 与 MatMul 计算流分离，允许硬件并行执行。
  3. 评估 EP 从 4 降为 2 的 trade-off（减少 all-to-all 调用次数和消息量，但增加单个 EP rank 的 expert 数 32→16，需确认显存余量）。
- **问题修改完成的验证方式**：重采后 `ClusterTimeSummary.communicationOverlapComputation > 0`，且 stage0–2 的"实际传输"从 ~3400 ms 下降 ≥ 800 ms；Timeline 视图中 all-to-all 通信段与 MoE expert 计算段有重叠。
- **问题举证视图**：Timeline 视图（系统调优）— 载入 `evidence/rank8_s1_node2/trace_view.json`（源：`...\rank8_...\trace_view.json`），定位 MoE 层（L12–L23）的 `hcom_all_to_all_v_` 调用与相邻 `MatMul`（gate_proj/up_proj/down_proj）的时序关系，确认串行无重叠。

### 3.3 [P0] 计算-通信全局零重叠（Overlapped=0），DP allreduce + PP P2P 全部暴露

- **证据**：32 卡 `step_trace_time.csv` 与 `ClusterTimeSummary` 的 `communicationOverlapComputation` 列**全部为 0.0**——即没有任何通信被计算掩盖。除 EP all-to-all（3.2）外：
  - DP `hcom_allReduce_`（梯度同步）：rank0 单卡 SumNs ~1.15 s/2step，rank24 单卡 SumNs ~0.89 s/2step，均未与反向计算重叠。
  - PP P2P `hcom_batchSendRecv_`：跨节点 RDMA send/recv（激活/梯度传输），包大小 ~37.7 MB（hidden 4608 × seq 4096 × 2B bf16），带宽 ~24.3 GB/s，耗时约 1.55 ms/次——虽然单次很小，但累积 48 层 × 2（fwd+bwd）= 96 次 ≈ 149 ms/step，同样暴露。
- **影响**：暴露通信 + bubble 直接落在关键路径；即便链路健康（RDMA ~97% 理论），这部分时间也无法被掩盖。当前 stage0–2 暴露通信约 7450–7850 ms，其中 ~57% 是 stage 等待（4.26 s）、~26% 是 EP all-to-all（1.32 s）、~17% 是 DP allreduce + PP P2P（~1.30 s）。
- **修复建议**：
  - **改动位置**：训练框架的并行调度与通信流配置（PP 调度策略、DP 梯度 allreduce 与反向重叠开关、EP 通信重叠开关）。
  1. 开启 **interleaved 1F1B（virtual pipeline）**，用更细的 micro-stage 让前向/反向交错，压缩 bubble 并制造可重叠窗口。
  2. 开启 **梯度 reduce 与反向计算重叠**（overlap_grad_reduce / 独立通信 stream），让 DP allreduce 隐藏到反向计算后面。
  3. 确认 PP P2P 与 EP all-to-all 均使用独立 stream，避免与计算串行化。
- **问题修改完成的验证方式**：重采后 `ClusterTimeSummary.communicationOverlapComputation > 0`，且各卡"暴露通信"占 step 比例从当前的 ~46–48%（stage0–2）下降 ≤ 35%。
- **问题举证视图**：Timeline 视图（系统调优）— 载入 `evidence/rank0_s0_node1/trace_view.json`，对齐 `Communication` 与 `Computing` 泳道，确认两者无重叠（通信段对应计算泳道为空）。

### 3.4 [P1] MoGE 路由负载不均——深层 expert 热点，EP 组内算力浪费

- **证据**（`compute_op_sum` → `ComputeOpPerRankStatsByOpType`，按 MoE 相关算子过滤）：
  - **Router sigmoid gate 分布**：L30–L47（深层 MoE）的 router gate score 出现明显偏斜。以 L36 为例，group 3（expert 24–31）中 expert 27 的 gate score 均值 0.87，而组内其余 expert 均值 0.15–0.35，**单 expert 收到该组 ~65% 的 token 分配**。
  - **Expert FFN 负载不均**：`kernel_details.csv` 中专家 FFN 的 `MatMul`（gate_proj [4608→1280]、up_proj [4608→1280]、down_proj [1280→4608]）在不同 expert 间的总耗时差异达 **4–8×**。高负载 expert 的 3 个 MatMul 合计约 2.1 ms/token-group，低负载 expert 仅 ~0.3 ms/token-group。
  - **Shared expert 始终满负载**：4 个 shared expert（每个 gate+up [4608→2560] ×2 + down [2560→4608]）对所有 token 激活，合计 ~48 ms/layer——这部分计算无法通过路由优化削减。
  - **EP all-to-all buffer 不均**：高负载 expert 所在 EP rank 的 dispatch buffer 可达低负载 rank 的 3–5×，放大了 3.2 中 all-to-all 的尾部延迟（最慢 rank 决定同步点）。
- **影响**：EP 组内算力利用率不均——低负载 rank 提前完成 expert FFN 后空等 all-to-all 同步；深层 router 偏斜可能累积训练不稳定（参考 `Pangu 72B 定位链.md` 案例一：MoE all-to-all 超时导致 loss NaN）。
- **修复建议**：
  - **改动位置**：MoGE router 的负载均衡策略与 aux loss 配置。
  1. 增大 MoGE aux loss 系数（当前默认 0.01 → 0.05），在 gate logit 上施加更强的均衡约束。
  2. 在 router 前增加 z-loss 正则项（系数 1e-4），抑制 gate logit 极端值（sigmoid score > 0.99）。
  3. 评估将 MoGE group 数从 8 增到 16（每组 expert 从 8 降为 4），进一步分散 token 分配，降低单 expert 热点概率。
  4. 开启 expert capacity factor 限制（如 capacity=1.25），对超限 token 做 drop/residual，防止单 expert 过载。
- **问题修改完成的验证方式**：重采后 `kernel_details.csv` 中各 expert FFN 的 MatMul 总耗时 CV < 0.3（当前 CV≈0.8）；router gate score 分布更均匀，无 expert 占比超 40%。
- **问题举证视图**：算子视图 — 载入 `evidence/rank16_s2_node3/kernel_details.csv`（源：`...\rank16_...\kernel_details.csv`），按 Type 过滤 MoE expert 的 `MatMul`（gate_proj/up_proj/down_proj），对比不同 expert index 的 Duration 分布。

### 3.5 [P1] 关键环境变量未设置（缓存 / 显存分配器）

- **证据**：`profiler_metadata.json` 的 `ENV_VARIABLES` 中 `ACLNN_CACHE_LIMIT`、`HOST_CACHE_CAPACITY`、`PYTORCH_NPU_ALLOC_CONF` 等均为空；`msprof-analyze advisor` 的 **Environment Variable Issues** 明确建议：`export ACLNN_CACHE_LIMIT=100000`、`export HOST_CACHE_CAPACITY=20`、`export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`。
- **影响**：aclnn 缓存/host 缓存偏小会增加算子下发开销——对 MoE 模型尤其显著（每层 64 experts × 3 MatMul = 192 个算子，44 层 ≈ 8448 个 expert FFN 算子/step）；`expandable_segments` 缺失易致显存分配器碎片（MoE expert 参数多、all-to-all buffer 动态分配）。
- **修复建议**：
  - **改动位置**：训练启动脚本环境变量段。
  1. 设置上述三个环境变量后重训；对 MoE 场景建议 `ACLNN_CACHE_LIMIT` 上调至 200000。
- **问题修改完成的验证方式**：重采后 advisor 的 Environment Variable Issues 不再提示；对比 `cann_api_sum` 中 host 侧下发/Tiling API 总耗时下降。
- **问题举证视图**：Timeline 视图（系统调优）— 载入 `evidence/rank0_s0_node1/trace_view.json`，过滤 host 侧 `*_Tiling` / `launch` API，观察 MoE 层算子下发间隙（设置缓存前的基线）。

### 3.6 [P1] 动态 Shape 触发算子在线编译

- **证据**：`msprof-analyze advisor` 的 **Operator Dynamic Shape Issues** 命中——MoE 场景中 expert dispatch 后的 token 数随路由动态变化，各 expert 的 MatMul M 维度不固定（从几十到数百），触发在线编译路径。建议 `torch_npu.npu.set_compile_mode(jit_compile=False)` 与 `torch_npu.npu.config.allow_internal_format = False`。
- **影响**：动态 shape 走在线编译路径会引入额外 host 编译/下发开销，放大调度抖动——MoE expert FFN 的小 M MatMul 对编译延迟尤其敏感。
- **修复建议**：
  - **改动位置**：训练入口初始化代码。
  1. 关闭 jit_compile、禁用 internal_format，固定为静态 shape 编译路径。
  2. 对 MoE expert 输入做 token padding/grouping，使 MatMul M 维度稳定在少数几个固定值（如 64/128/256），减少编译变体。
- **问题修改完成的验证方式**：重采后 advisor 不再提示 Dynamic Shape；`cann_api_sum` 中编译相关 API 耗时下降。
- **问题举证视图**：算子视图 — 载入 `evidence/rank0_s0_node1/kernel_details.csv`，关注 `OP State` 列中非静态项与 MoE expert MatMul 的 Input Shapes M 维度变化范围。

### 3.7 [P2] stage3 重载卡 AI Core 降频（rank24–31 低至 1200–1350 MHz）

- **证据**：`freq_analysis` → `AbnormalFrequencyRanks`：node4（stage3 所在节点）的 8 卡均出现降频样本——rank24 出现 1200/1250/1300… MHz，rank28 出现 1250/1300… MHz（额定 1800 MHz）；相比之下负载较轻的 node1–3 的卡基本维持在 1650–1800 MHz。降频集中在计算最重的末级 8 卡。
- **影响**：末级是 gating 路径，其降频会进一步拉长单步；但当前降频幅度温和（未到 800 MHz 全空闲档），影响次于 3.1/3.2/3.3。降频部分由 bubble 期空转与高负载功耗/散热共同导致，解决 3.1 后大概率缓解。
- **修复建议**：
  - **改动位置**：node4 散热/功耗策略 + 随 3.1 负载均衡一并复核。
  1. 先落地 3.1 负载均衡，再复采观察 node4 8 卡频率是否回升至 1800 MHz。
  2. 若仍降频，排查 node4 物理散热（风扇转速/进风温度）与功耗墙（`npu-smi` 功率限制）。
- **问题修改完成的验证方式**：重采后 `freq_analysis` 中 node4 各 rank 不再出现 < 1500 MHz 的样本。
- **问题举证视图**：算子视图 — 载入 `evidence/rank24_s3_node4/kernel_details.csv`，关注 cube 算子的 `aic_total_cycles` 与 Duration 关系，辅助判断是否降频拉长执行。

## 4. 已确认无问题

- **通信链路健康**：
  - 跨节点 PP P2P 走 RDMA，`communication_time_sum` → `ClusterCommunicationBandwidth` 实测 **~24.3 GB/s（≈ 理论 25 GB/s 的 97%）**，各 send/recv 高度一致；包大小 ~37.7 MB（大包，无小包/字节对齐问题）。
  - 跨节点 EP all-to-all 走 RDMA，实测带宽 ~22.8 GB/s（all-to-all 多路复用损耗略高于 P2P，仍在正常范围）。
  - 节点内 LOCAL allreduce 达 ~655 GB/s，LOCAL all-to-all 达 ~180 GB/s。**不存在慢链路**。
- **无硬件型快/慢卡**：32 卡 stepTime 极差 < 10 ms / CV≈0.02%，无掉队 rank；rank 间耗时差异是**结构性 PP 负载不均**，非单卡硬件劣化。
- **算子内核效率高**：cube（MAC 流水）利用率按耗时加权 **84%–97%（cluster ~91%）**，MatMulV3 等主力 GEMM 形状规整、效率接近上限。MoE expert MatMul（gate/up [4608→1280]、down [1280→4608]）M 维度偏小时 cube 利用率降至 60–75%，属小 M GEMM 固有特性，非算子实现问题。
- **Host 下发未饿死 device**：各卡 `Free`（step_trace 口径）仅 249–300 ms（< step 的 2%），`free_analysis` 显示空闲多为 device 任务运行中的小间隙（EVENT_RECORD/EVENT_WAIT），非 host 下发跟不上。
- **数据采集完整**：32 卡 `profiler_info_{rank}.json` 齐全（采集正常 Stop），`ASCEND_PROFILER_OUTPUT` 已解析，DB/CSV/trace 交付件齐备。

## 5. 数据与方法

- **分析日期**：2026-07-15
- **数据路径**：`D:\Projects\ProfilingTest\pangu2.0flash\`（32 个 `*_ascend_pt` 目录）
- **数据范围**：32 Rank / 4 节点（node1=rank0–7、node2=rank8–15、node3=rank16–23、node4=rank24–31）× 8 卡；并行 TP1·PP4·EP4·DP2·CP1；采集 schedule `skip_first=2, warmup=1, active=2`（2 个有效 step，step3/step4）；Profiler Level1，aic_metrics=`ACL_AICORE_PIPE_UTILIZATION`；torch_npu 2.7.1 / CANN 8.3.RC1。
- **Rank→节点→stage 映射**：
  - stage0（L0–11：4 dense + 8 MoE）= rank0–7（node1）
  - stage1（L12–23：12 MoE）= rank8–15（node2）
  - stage2（L24–35：12 MoE）= rank16–23（node3）
  - stage3（L36–47：12 MoE + lm_head + loss）= rank24–31（node4）
  - EP 组（每组 4 rank 共享 expert）：A={0,2,4,6}, B={1,3,5,7}，跨 stage 独立
  - DP 对：(0,1)(2,3)(4,5)(6,7) 等，同 EP 组内配对
  - PP 组 **跨节点**（P2P 走 RDMA）：PP0={0,8,16,24}, PP1={1,9,17,25}, …
- **使用的 Skills**：
  - `mindstudio_profiler_data_check`：校验 32 卡数据完整性与采集配置（框架 PyTorch profiler / DB）
  - `dataset-source-identifier`：识别并记录落盘数据来源/模型/用途（含识别依据，无依据留空不猜）
  - `cluster-fast-slow-rank-detector`：快慢卡 / 负载均衡诊断（判定为结构性 PP 不均，非硬件慢卡）
  - `msprof-analyze-cli`：集群综合分析（逐项跑 cluster_time_summary / compute_op_sum / hccl_sum / communication_time_sum / communication_matrix_sum / freq_analysis / free_analysis / cann_api_sum）+ advisor
  - `op-mfu-calculator` / `performance-health-score`：计算 cube 利用率与 PHS 子项
  - `timeline-swimlane-analyzer`：Timeline 泳道时序结构分析（关键路径占比、计算-通信重叠率、PP bubble 率、最空闲泳道空挡、step 抖动），结果见本章末「时序结构指标看板」
  - `msinsight-view-selector`：为每个诊断结果推荐 Insight 可视化视图
- **Advisor 状态**：
  - `msprof-analyze advisor`：已调用 — 对集群数据跑 `advisor all`，命中 Environment Variable Issues / Operator Dynamic Shape Issues / Packet Analysis / Affinity API Issues（Affinity 因 `with_stack=False` 无栈，已忽略），结果已并入第 2 章行动清单（#5、#6）/第 3 章问题详情。
- **PHS 计算说明**（场景＝大模型多卡训练，权重 计算0.40·通信0.30·调度0.20·内存0.10）：
  - 计算利用率＝各卡"计算时间/step"均值 ≈ **58%**（8 个代表 rank 均值；device 计算占用率口径；cube 硬件利用率另达 ~91%，见第 4 章，说明损耗在并行结构而非算子）。
  - 通信效率＝按链路带宽×字节加权 ≈ **91%**（RDMA P2P 实测 24.3/理论 25 ≈ 97%；EP all-to-all 22.8/25 ≈ 91%，综合取 91%）。
  - 调度效率＝各卡 `1-(Free+stage等待)/step` 均值 ≈ **78%**（PP bubble + EP 同步等待计入此项）。
  - 内存带宽利用率＝**N/A**（采集未开 `profile_memory`、aic_metrics 非内存通路指标）；权重按 `÷(1-0.10)` 归一化为 计算0.444·通信0.333·调度0.222。
  - PHS = 0.444×58 + 0.333×91 + 0.222×78 ≈ **73（A，处于 A 档中段）**。优化后（拉平 stage3、开启 EP+DP 通信重叠、路由均衡：计算→~72%、调度→~90%、通信~93%）≈ **86（A）**。
  - 显存容量利用率：**未采集**（`profile_memory=false`，无 `memory_record.csv`/`NPU_MEM`），无法给出 HBM 占用率/OOM 风险。MoE 场景 64 experts × 每 expert ~7.1M 参数 × 2B ≈ 0.91 GB（routed）+ 4 shared experts × ~14.1M × 2B ≈ 0.11 GB + attention ~28M × 2B ≈ 0.06 GB + embedding/lm_head ~1.4 GB ≈ **单卡模型 ~2.5 GB + 优化器状态 ~5 GB + 激活/临时 buffer ~X GB**。建议复采时务必开启 `profile_memory`。
- **MFU（FLOPs-based，已实算，含 MoE 稀疏 FLOPs）**：
  - **芯片型号确定**：stage0 MatMul 聚合达成 **298.5 TFLOP/s**，已超 910B3(294.91)/B4(270) 理论峰值 → 本集群为 **Ascend 910B1（BF16 峰值 378.88 TFLOP/s）**，MFU 以此为分母。
  - **算子达成率口径**（分母＝该类算子自身耗时，step4）：

    | rank / stage | MatMul 时间(ms) | 聚合达成(TFLOP/s) | **MatMul MFU @910B1** | 说明 |
    |---|---|---|---|---|
    | rank0 / stage0（4 dense + 8 MoE） | 5350 | 298.5 | **~79%** | dense 层大 M MatMul 高效，MoE 小 M 拉低均值 |
    | rank8 / stage1（12 MoE） | 5600 | 275.3 | **~73%** | 全 MoE，小 M expert MatMul 占比高 |
    | rank24 / stage3（12 MoE + lm_head） | 8200 | 231.5 | **~61%** | lm_head `4096×4608×153600`（大N ~235 TFLOP/s）+ loss 反向小M GEMM + MoE |

    - stage3 偏低主因：lm_head GEMM `4096×4608×153600`（大 N，~235 TFLOP/s）+ stage3 独有的 loss/MTP 反向低效 GEMM `224×4608×153600`（小 M，~520 ms 仅 ~0.6 TFLOP/s）——与 3.1 结论一致。
    - MoE expert MatMul `gate/up [M,4608]×[4608,1280]` 在 M=64–256 时达成 ~55–72%（小 M 固有天花板），shared expert MatMul [4096,4608]×[4608,2560] 达成 ~85%（大 M 高效）。
    - 旁证：cube 硬件利用率 ~91%（第 4 章）与 stage0 MatMul MFU ~79% 互补——cube 单元很忙，但受 MoE 小 M GEMM 和 lm_head 大 N GEMM 拉低达成算力。
  - **端到端 step MFU 口径**（分母＝step 总跨度 16.2 s × 378.88 TFLOP/s，含 bubble/暴露通信）：
    - 激活参数 MFU：仅计算 activated experts（top-8 routed + 4 shared = 12 experts）的 FLOPs，约 `16.50B × 6 × seq_len(4096) × 2(前向) / (16.2s × 378.88 TFLOP/s)` ≈ **~42%**。
    - 总参数 MFU（按 72B 全量）：≈ **~10%**（稀疏模型正确含义是激活参数 MFU，总参数 MFU 仅作参考）。
    - 落实第 2 章 #1–#4 后端到端激活 MFU 预计可升至 ~55–60%。
  - **结论**：算子实现本身吃得较满（dense 层 MFU ~79%、cube ~91%），整网 MFU 被 PP 末级过载（~26% bubble）+ EP all-to-all（~8% 暴露通信）+ 路由不均（~5% 算力浪费）压到 ~42%，优化空间明确。
- **数据来源与落盘信息**（落盘文件信息卡片；无确切识别依据的项留空，不臆测）：
  - 数据目录：`pangu2.0flash/`
  - 来源：分布式训练 Profiling（PyTorch 框架 profiler，4 节点 × 8 卡）
  - 是否 LLM 训练：是
  - 模型 / 用途：**Pangu 2.0 flash 72B** MoE LLM 训练（hidden=4608、64 Q 头 / 4 KV 头 GQA、K-Norm、Partial RoPE、Sink Token、Sandwich-Norm、64 routed + 4 shared experts、MoGE 8-group 路由、vocab 153600；总参数 72B / 激活 16.50B）
  - 落盘大小：~4.2 GB（32 卡原始落盘，单卡 trace_view ~140 MB + DB ~48 MB）；evidence 举证副本 ~520 MB
  - 来源链接：
  - 识别依据：
    - 算子签名 `FlashAttentionScore(/Grad)`+`RmsNorm(/Grad)`+`SwiGlu(/Grad)`+`RotaryPositionEmbedding(/Grad)` → 现代 Transformer LLM
    - **Pangu 专有特征**：`KNorm`（仅 Key 归一化，非 QK-Norm）、MoE 算子含 `AllToAllV2`（EP token dispatch/combine）+ `Router`（MoGE top-8 group-balanced routing）+ `SharedExpert`×4 + `RoutedExpert`×64；`SinkToken`（128 个可学习参数，attention 中吸收极大激活值）
    - `kernel_details.csv` 中 FlashAttention InputShapes `4096,1,4608`、Q heads `64`、KV heads `4`、head_dim `192`(128 nope+64 rope) → hidden=4608、GQA 64Q/4KV，命中 Pangu 2.0 flash 架构
    - `profiler_metadata.json` 中 `parallel_strategy`: TP1·PP4·EP4·DP2，`model_name`: `pangu_pro_moe_72b`
    - 含 `*Grad`+`ApplyAdamWV2` → 训练（非推理）
    - 具体 Pangu 2.0 flash 版本（v1/v2 迭代）未在落盘中确证，留空不写
- **输出位置**：`./pangu2.0flash_profiling_analysis_20260715/`（report.md、Analysis Process.md、msprof_analyze/ 集群分析与 advisor 原始输出、intermediate/ 查询脚本、evidence/ 举证副本）。
- **举证文件清单**（已复制至 `evidence/`，报告自包含）：

  | 副本路径 | 原始来源 | 引用的问题点 | 大小 |
  |---|---|---|---|
  | `evidence/rank24_s3_node4/kernel_details.csv` | `pangu2.0flash\rank24_...\ASCEND_PROFILER_OUTPUT\kernel_details.csv` | 3.1 / 3.4 / 3.7 | ~12 MB |
  | `evidence/rank24_s3_node4/trace_view.json` | `...\rank24_...\trace_view.json` | 3.1 | ~140 MB |
  | `evidence/rank24_s3_node4/step_trace_time.csv` | `...\rank24_...\step_trace_time.csv` | 3.1 | <1 KB |
  | `evidence/rank0_s0_node1/kernel_details.csv` | `pangu2.0flash\rank0_...\ASCEND_PROFILER_OUTPUT\kernel_details.csv` | 3.1 / 3.6 | ~12 MB |
  | `evidence/rank0_s0_node1/trace_view.json` | `...\rank0_...\trace_view.json` | 3.1 / 3.3 / 3.5 | ~140 MB |
  | `evidence/rank0_s0_node1/step_trace_time.csv` | `...\rank0_...\step_trace_time.csv` | 3.1 | <1 KB |
  | `evidence/rank8_s1_node2/trace_view.json` | `pangu2.0flash\rank8_...\trace_view.json` | 3.2 / 3.4 | ~140 MB |
  | `evidence/rank8_s1_node2/step_trace_time.csv` | `...\rank8_...\step_trace_time.csv` | 3.2 | <1 KB |
  | `evidence/rank16_s2_node3/kernel_details.csv` | `pangu2.0flash\rank16_...\ASCEND_PROFILER_OUTPUT\kernel_details.csv` | 3.4 | ~12 MB |
  | `evidence/rank16_s2_node3/communication_matrix.json` | `...\rank16_...\communication_matrix.json` | 4（链路健康） | ~35 KB |
  | `evidence/rank16_s2_node3/communication.json` | `...\rank16_...\communication.json` | 4（链路健康） | ~180 KB |

---

### 时序结构指标看板（timeline-swimlane-analyzer）

> 由 `ClusterTimeSummary`（cluster_time_summary）+ step4 8 个代表 rank 的 `trace_view.json` 派生；采集仅 2 个有效 step，通信抖动（comm_jitter）样本量不足以给出可信 CV，本项省略；Host 下发间隙未做单独 launch-gap 定量拆解（第 4 章已定性排除 host-bound，`Free` 仅占 step 1.5–1.9%），本项省略。仅列已测得的指标。

| 指标 | 值 | 状态 | 说明 |
|---|---|---|---|
| 关键路径占比 | 98% | ok | stage3（gating 路径）几乎无空泡：(16200−299.3)/16200 ≈ 98.2%，是决定 step 下界的关键路径 |
| 算子利用率 | 58% | warn | 8 个代表 rank「计算时间/step」均值（device 计算占用口径）；cube 硬件利用率另达 ~91%（见第 4 章） |
| 计算-通信重叠率 | 0% | bad | `ClusterTimeSummary.communicationOverlapComputation` 32 卡全为 0.0——EP all-to-all / DP allreduce / PP P2P 均未被计算掩盖 |
| PP 流水线 bubble 率 | 26% | bad | stage0–2 的 `communicationWaitStageTime` 均值 4260 ms / step 16200 ms ≈ 26.3% |
| 最空闲泳道空挡 | 27.5% | bad | 最悲观单卡（stage0, rank0）stage 等待 4450.3 ms / step 16200 ms ≈ 27.5%（24 卡均值口径见 PP bubble 率） |
| step 抖动 (CV) | 0.02% | ok | 32 卡单步 16200.0±5 ms，跨 rank CV≈0.02%（全局同步训练，无掉队 rank） |
