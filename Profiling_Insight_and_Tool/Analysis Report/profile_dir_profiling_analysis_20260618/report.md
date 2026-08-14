# profile_dir 多机多卡训练性能诊断报告（2 节点 × 4 卡 / PP4·DP2）

## 1. 结论速览

- **性能健康度**：75 / 100 (A) → 优化后预估 **85 / 100 (A)** — 计算 64% · 通信 93% · 调度 71% · 内存 N/A
- **结论**：单步 ~10.80 s，被**流水线最后一级（PP stage3，rank6/7）计算过载**主导——8 卡 step 时间几乎完全一致（CV≈0.01%），但负载严重不均。
- **头号瓶颈**：stage3 计算 ~9.26 s/step，是其余各级（~6.15 s）的 **1.51×**（多 ~3.11 s）；其余 6 卡因此每步空等 ~3.77 s 流水线 bubble（占 step **~35%**）。
- **收益上限**：若把 stage3 负载拉平到与其余级一致并开启计算-通信重叠，预计节省 **~26–30% 单步耗时（~2.8–3.0 s）**。

## 2. 行动清单

> 默认按"预期收益"从高到低排序；同档收益按"修改难度"从低到高排序。

| # | 优先级 | 问题 | 预期收益 | 修改难度 |
|---|---|---|---|---|
| 1 | P0 | PP 末级（stage3, rank6/7）计算过载，制造 ~35% 单步 bubble | -26~30% 单步耗时（~2.8–3.0 s） | 中 |
| 2 | P0 | 计算-通信零重叠（Overlapped=0），暴露通信 100% 进关键路径 | -10~15% 单步耗时 | 中 |
| 3 | P1 | 关键环境变量未设置（缓存/显存分配器） | 中（降 host 下发开销、缓解碎片） | 低 |
| 4 | P1 | 动态 Shape 触发算子在线编译 | 低~中 | 低 |
| 5 | P2 | stage3 重载卡 AI Core 降频（rank6/7 低至 1200–1350 MHz） | 低 | 中 |

## 3. 问题详情

### 3.1 [P0] PP 末级（stage3, rank6/7）计算过载，制造 ~35% 单步 bubble

- **证据**（`cluster_time_summary` → `ClusterTimeSummary` 表，step 4，单位换算为 ms）：

  | rank | 节点 | PP stage | DP 对 | 计算(ms) | 暴露通信(ms) | 其中 stage 等待(ms) | 实际传输(ms) | Free(ms) | stepTime(ms) | 计算占比 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 0 | ubuntu122 | s0 | (0,1) | 6325.6 | 4389.6 | 3613.1 | 776.5 | 83.6 | 10799.8 | 58.6% |
  | 1 | ubuntu122 | s0 | (0,1) | 6359.1 | 4293.8 | 3515.7 | 778.0 | 143.7 | 10797.7 | 58.9% |
  | 2 | ubuntu122 | s1 | (2,3) | 6005.1 | 4626.7 | 3786.7 | 840.0 | 168.4 | 10800.6 | 55.6% |
  | 3 | ubuntu122 | s1 | (2,3) | 6003.9 | 4635.5 | 3787.1 | 848.5 | 160.9 | 10799.8 | 55.6% |
  | 4 | localhost | s2 | (4,5) | 6174.0 | 4492.1 | 3886.4 | 605.7 | 130.3 | 10796.6 | 57.2% |
  | 5 | localhost | s2 | (4,5) | 6011.9 | 4670.0 | 4053.7 | 616.3 | 116.1 | 10798.3 | 55.7% |
  | **6** | localhost | **s3** | (6,7) | **9454.0** | 1092.5 | 392.3 | 700.2 | 250.8 | 10798.2 | **87.6%** |
  | **7** | localhost | **s3** | (6,7) | **9060.3** | 1554.2 | 848.3 | 705.9 | 182.7 | 10797.9 | **83.9%** |

  - step 时间在 8 卡间几乎相同（10796.6–10800.6 ms，CV≈0.012%）——典型的**全局同步训练**，不存在硬件型快/慢卡（无掉队 rank）。
  - 但 stage3（rank6/7）计算均值 **9257 ms**，stage0–2 均值 **6147 ms**——**末级多算 3110 ms/step（1.51×）**。
  - 末级"多出来的活"已定位到落盘算子（`compute_op_sum`/`kernel_details.csv`）：**`MatMulV2`（lm_head/logits 投影，平均 16.0 ms/次 ×64 次 ≈ 1024 ms）仅出现在 stage3 的 rank6/7**，rank0–5 完全没有；外加末级二级大 MatMul、loss 相关 vector 算子更重（stage3 vector 算子耗时是 stage0 的 ~1.6×）。
  - 后果：rank0–5 每步在 P2P recv 上**空等 `communicationWaitStageTime` ≈ 3.51–4.05 s（均值 3.77 s，占 step ~35%）**，而 stage3 几乎不等（0.39–0.85 s）。`hccl_sum` 佐证：stage0–2 的 `hcom_batchSendRecv_`（PP P2P）单卡 SumNs ~7.4–7.9 s/2step，但其 Min 仅 1.25 ms、Max 达 100+ ms——绝大部分是**等待**而非传输。
- **影响**：单步 ~10.80 s 被末级计算 gating。全集群每步约 `6 卡 × 3.77 s ≈ 22.6` 卡·秒空耗在 bubble（≈ 集群总算力时间的 26%）。
- **修复建议**：
  - **改动位置**：训练启动并行切分配置（Megatron/MindSpeed 的 PP 层切分参数），目标——把末级 transformer 层数调少以抵消 lm_head+loss 的额外开销，使各 stage 总耗时拉平。
  1. 采用**非均匀 PP 切分**：减少最后一级的 decoder 层数（如 Megatron `--decoder-last-pipeline-num-layers`，把末级层数下调，差额 ~3.1 s 对应约 0.5 层的等效负载，按实测微调），让 stage3 计算从 ~9.26 s 降到 ~6.4 s。
  2. 或将 lm_head/loss 的计算量摊薄：开启 lm_head 的张量切分（当前 TP=1，可考虑对输出投影/词表做 TP 或 vocab-parallel cross-entropy），把 logits GEMM 与 CE 拆到多卡。
  3. 复采时建议把 `num_microbatches` 提高，并确认 1F1B 调度，进一步压低固有 bubble。
- **问题修改完成的验证方式**：重采后再跑 `msprof-analyze -m cluster_time_summary`，确认 `ClusterTimeSummary` 中 stage3 计算与 stage0–2 极差 < 10%，且 stage0–2 的 `communicationWaitStageTime` 从 ~3.77 s 降到 < 1.5 s；单步 stepTime 降到 ~7.5–8.0 s。
- **问题举证视图**：
  - 主：算子视图 — 载入 `evidence/rank6_s3_localhost/kernel_details.csv`（源：`D:\Projects\ProfilingTest\profile_dir\localhost.localdomain_10701_20260120112415091_ascend_pt\ASCEND_PROFILER_OUTPUT\kernel_details.csv`），按 Type 过滤 `MatMulV2`，确认该 lm_head 投影只在末级出现且单次 ~16 ms；对照 `evidence/rank0_s0_ubuntu122/kernel_details.csv`（源：`...\ubuntu122_14296_...\ASCEND_PROFILER_OUTPUT\kernel_details.csv`）无该算子。
  - 辅：Timeline 视图（系统调优）— 载入 `evidence/rank0_s0_ubuntu122/trace_view.json`（源：`...\ubuntu122_14296_...\trace_view.json`），观察 P2P recv 处长达 ~3.8 s 的 device 空挡（bubble）。

### 3.2 [P0] 计算-通信零重叠（Overlapped=0），暴露通信 100% 进关键路径

- **证据**：8 卡 `step_trace_time.csv` 与 `ClusterTimeSummary` 的 `communicationOverlapComputation` 列**全部为 0.0**——即没有任何通信被计算掩盖；`Communication(Not Overlapped)` = `Communication`（rank0 step4：4389.6 ms 全部暴露）。RDMA P2P 与 DP `hcom_allReduce_`（rank0/1 单卡 SumNs ~0.93–0.94 s/2step 的大 allreduce + rank6/7 各 270 个小 allreduce）均未与计算并行。
- **影响**：暴露通信 + bubble 直接落在关键路径；即便链路本身健康，这部分时间也无法被掩盖。结合 3.1，stage0–2 暴露通信里约 80% 实为 stage 等待。
- **修复建议**：
  - **改动位置**：训练框架的并行调度与通信流配置（PP 调度策略、DP 梯度 allreduce 与反向重叠开关）。
  1. 开启 **interleaved 1F1B（virtual pipeline）**，用更细的 micro-stage 让前向/反向交错，压缩 bubble 并制造可重叠窗口。
  2. 开启 **梯度 reduce 与反向计算重叠**（overlap_grad_reduce / 独立通信 stream），让 DP allreduce 隐藏到反向计算后面。
  3. 确认 P2P 使用独立 stream，避免与计算串行化。
- **问题修改完成的验证方式**：重采后 `ClusterTimeSummary.communicationOverlapComputation > 0`，且各卡"暴露通信"占 step 比例下降 ≥ 10 个百分点。
- **问题举证视图**：Timeline 视图（系统调优）— 载入 `evidence/rank0_s0_ubuntu122/trace_view.json`（源：`...\ubuntu122_14296_...\trace_view.json`），对齐 `Communication` 与 `Computing` 泳道，确认两者无重叠（通信段对应计算泳道为空）。

### 3.3 [P1] 关键环境变量未设置（缓存 / 显存分配器）

- **证据**：`profiler_metadata.json` 的 `ENV_VARIABLES` 中 `ACLNN_CACHE_LIMIT`、`HOST_CACHE_CAPACITY`、`PYTORCH_NPU_ALLOC_CONF` 等均为空；`msprof-analyze advisor` 的 **Environment Variable Issues** 明确建议：`export ACLNN_CACHE_LIMIT=100000`、`export HOST_CACHE_CAPACITY=20`、`export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`。
- **影响**：aclnn 缓存/host 缓存偏小会增加算子下发开销；`expandable_segments` 缺失易致显存分配器碎片（本次未开 `profile_memory`，碎片量未量化）。
- **修复建议**：
  - **改动位置**：训练启动脚本环境变量段。
  1. 设置上述三个环境变量后重训。
- **问题修改完成的验证方式**：重采后 advisor 的 Environment Variable Issues 不再提示；对比 `cann_api_sum` 中 host 侧下发/Tiling API 总耗时下降。
- **问题举证视图**：Timeline 视图（系统调优）— 载入 `evidence/rank0_s0_ubuntu122/trace_view.json`（源：`...\ubuntu122_14296_...\trace_view.json`），过滤 host 侧 `*_Tiling` / `launch` API，观察下发间隙（设置缓存前的基线）。

### 3.4 [P1] 动态 Shape 触发算子在线编译

- **证据**：`msprof-analyze advisor` 的 **Operator Dynamic Shape Issues** 命中，建议 `torch_npu.npu.set_compile_mode(jit_compile=False)` 与 `torch_npu.npu.config.allow_internal_format = False`。
- **影响**：动态 shape 走在线编译路径会引入额外 host 编译/下发开销，放大调度抖动。
- **修复建议**：
  - **改动位置**：训练入口初始化代码。
  1. 关闭 jit_compile、禁用 internal_format，固定为静态 shape 编译路径。
- **问题修改完成的验证方式**：重采后 advisor 不再提示 Dynamic Shape；`cann_api_sum` 中编译相关 API 耗时下降。
- **问题举证视图**：算子视图 — 载入 `evidence/rank0_s0_ubuntu122/kernel_details.csv`（源：`...\ubuntu122_14296_...\kernel_details.csv`），关注 `OP State` 列中非静态项，定位动态 shape 算子。

### 3.5 [P2] stage3 重载卡 AI Core 降频（rank6/7 低至 1200–1350 MHz）

- **证据**：`freq_analysis` → `AbnormalFrequencyRanks`：rank6 出现 1200/1250/1300… MHz，rank7 出现 1250/1300… MHz（额定 1800 MHz）；相比之下负载较轻的 rank0 仅 1700–1800 MHz。降频集中在计算最重的末级两卡。
- **影响**：末级是 gating 路径，其降频会进一步拉长单步；但当前降频幅度温和（未到 800 MHz 全空闲档），影响次于 3.1/3.2。降频部分由 bubble 期空转与高负载功耗/散热共同导致，解决 3.1 后大概率缓解。
- **修复建议**：
  - **改动位置**：节点散热/功耗策略 + 随 3.1 负载均衡一并复核。
  1. 先落地 3.1 负载均衡，再复采观察 rank6/7 频率是否回升至 1800 MHz。
  2. 若仍降频，排查该物理节点（localhost）散热与功耗墙。
- **问题修改完成的验证方式**：重采后 `freq_analysis` 中 rank6/7 不再出现 < 1500 MHz 的样本。
- **问题举证视图**：算子视图 — 载入 `evidence/rank6_s3_localhost/kernel_details.csv`（源：`...\localhost.localdomain_10701_...\kernel_details.csv`），关注 cube 算子的 `aic_total_cycles` 与耗时关系，辅助判断是否降频拉长执行。

## 4. 已确认无问题

- **通信链路健康**：跨节点 PP P2P 走 RDMA，`communication_time_sum` → `ClusterCommunicationBandwidth` 实测 **~24.2 GB/s（≈ 理论 25 GB/s 的 97%）**，各 send/recv 高度一致；包大小 29.36 MB（大包，无小包/字节对齐问题）。节点内 LOCAL allreduce 达 ~661 GB/s。**不存在慢链路**。
- **无硬件型快/慢卡**：8 卡 stepTime 极差 4 ms / CV≈0.012%，无掉队 rank；rank 间差异是**结构性 PP 负载不均**，非单卡硬件劣化。
- **算子内核效率高**：cube（MAC 流水）利用率按耗时加权 **86%–98%（cluster ~92%）**，MatMulV3 等主力 GEMM 形状规整、效率接近上限——瓶颈不在单算子实现，而在并行结构。
- **Host 下发未饿死 device**：各卡 `Free`（step_trace 口径）仅 84–251 ms（< step 的 2.5%），`free_analysis` 显示空闲多为 device 任务运行中的小间隙（EVENT_RECORD/EVENT_WAIT），非 host 下发跟不上。
- **数据采集完整**：8 卡 `profiler_info_{rank}.json` 齐全（采集正常 Stop），`ASCEND_PROFILER_OUTPUT` 已解析，DB/CSV/trace 交付件齐备。

## 5. 数据与方法

- **分析日期**：2026-06-18
- **数据路径**：`D:\Projects\ProfilingTest\profile_dir\`（8 个 `*_ascend_pt` 目录）
- **数据范围**：8 Rank / 2 节点（ubuntu122=rank0–3，localhost=rank4–7）× 4 卡；并行 TP1·PP4·DP2·CP1；采集 schedule `skip_first=2, warmup=1, active=2`（2 个有效 step，step3/step4）；Profiler Level1，aic_metrics=`ACL_AICORE_PIPE_UTILIZATION`；torch_npu 2.7.1 / CANN 8.3.RC1。
- **Rank→节点→stage 映射**：stage0={0,1}、stage1={2,3}、stage2={4,5}、stage3={6,7}；DP 对 (0,1)(2,3)(4,5)(6,7)；PP 组 {0,2,4,6}/{1,3,5,7} **跨节点**（P2P 走 RDMA）。
- **使用的 Skills**：
  - `mindstudio_profiler_data_check`：校验 8 卡数据完整性与采集配置（框架 PyTorch profiler / DB）
  - `dataset-source-identifier`：识别并记录落盘数据来源/模型/用途（含识别依据，无依据留空不猜）
  - `cluster-fast-slow-rank-detector`：快慢卡 / 负载均衡诊断（判定为结构性 PP 不均，非硬件慢卡）
  - `msprof-analyze-cli`：集群综合分析（逐项跑 cluster_time_summary / compute_op_sum / hccl_sum / communication_time_sum / communication_matrix_sum / freq_analysis / free_analysis / cann_api_sum）+ advisor
  - `op-mfu-calculator` / `performance-health-score`：计算 cube 利用率与 PHS 子项
  - `msinsight-view-selector`：为每个诊断结果推荐 Insight 可视化视图
- **Advisor 状态**：
  - `msprof-analyze advisor`：已调用 — 对集群数据跑 `advisor all`，命中 Environment Variable Issues / Operator Dynamic Shape Issues / Packet Analysis / Affinity API Issues（Affinity 因 `with_stack=False` 无栈，已忽略），结果已并入第 2 章行动清单（#3、#4）/第 3 章问题详情。
- **PHS 计算说明**（场景＝大模型多卡训练，权重 计算0.40·通信0.30·调度0.20·内存0.10）：
  - 计算利用率＝各卡"计算时间/step"均值 ≈ **64%**（device 计算占用率口径；cube 硬件利用率另达 ~92%，见第 4 章，说明损耗在并行结构而非算子）。
  - 通信效率＝按链路带宽×字节加权 ≈ **93%**（RDMA 实测 24.2/理论 25 ≈ 97%，保守取 93%；链路健康）。
  - 调度效率＝各卡 `1-(Free+stage等待)/step` 均值 ≈ **71%**（PP bubble 计入此项）。
  - 内存带宽利用率＝**N/A**（采集未开 `profile_memory`、aic_metrics 非内存通路指标）；权重按 `÷(1-0.10)` 归一化为 计算0.444·通信0.333·调度0.222。
  - PHS = 0.444×64 + 0.333×93 + 0.222×71 ≈ **75（A，处于 A 档下沿）**。优化后（拉平 stage3、开启重叠：计算→~78%、调度→~88%、通信~93%）≈ **85（A）**。
  - 显存容量利用率：**未采集**（`profile_memory=false`，无 `memory_record.csv`/`NPU_MEM`），无法给出 HBM 占用率/OOM 风险，建议复采时开启。
- **MFU（FLOPs-based，已实算，补算于 2026-06-24）**：
  - **前提澄清（修正初版口径）**：初版报告因 `profiler_info` 中 `record_shapes=false` 而判定"无 shape→只能报 cube 利用率，无法算 MFU"。该判据有误——`record_shapes` 是 PyTorch 框架层开关，只影响 host 侧 trace；`kernel_details.csv` 的 `Input/Output Shapes` 是 **device 侧 CANN 落盘**，与之无关。实测 rank0 的 29350 行算子中 **29152 行有 shape**，MatMul/FA 的 M/N/K 完整可取，**FLOPs-based MFU 可正常算**。
  - **芯片型号确定**：stage0 MatMul 聚合达成 **301.7 TFLOP/s**，已超 910B3(294.91)/B4(270) 理论峰值 → 本集群为 **Ascend 910B1（BF16 峰值 378.88 TFLOP/s）**，MFU 以此为分母。
  - **算子达成率口径**（分母＝该类算子自身耗时，转置安全用 Output shape 锚定 M/N，step4）：

    | rank / stage | MatMul 时间(ms) | 聚合达成(TFLOP/s) | **MatMul MFU @910B1** |
    |---|---|---|---|
    | rank0 / stage0（纯 transformer） | 4252 | 301.7 | **~80%** |
    | rank6 / stage3（含 lm_head+loss） | 6476 | 242.4 | **~64%** |

    - stage3 偏低主因：lm_head GEMM `4096×3584×152064`（大 N，~240 TFLOP/s）+ stage3 独有的低效 GEMM `224×9504×9504`（小 M，514ms 仅 ~3 TFLOP/s，属 loss 反向）——与 3.1 结论一致。
    - 旁证：cube 硬件利用率 ~92%（第 4 章）与 MatMul MFU ~80% 互补——cube 单元很忙，但受 lm_head 大 N、loss 小 M GEMM 拉低达成算力。
  - **端到端 step MFU 口径**（分母＝step 总跨度 × 峰值，含 bubble/暴露通信）：≈ 算子达成率 × 计算占用率(~64%) ≈ **40% 出头**。这是训练整体效率的真实水位，低于算子达成率正是因为 3.1 的 ~35% bubble + 3.2 的零重叠暴露通信。FA 的 FLOPs 暂未并入端到端严格值（causal sparse_mode 待最终确认），当前为占用率推导的估算。
  - **结论**：算子实现/形状本身吃得较满（stage0 MatMul MFU ~80%、cube ~92%），整网 MFU 被并行结构（PP 末级过载 + 零重叠）压到 ~40%，落实第 2 章 #1/#2 后端到端 MFU 预计可升至 ~55–60%。
- **数据来源与落盘信息**（落盘文件信息卡片；无确切识别依据的项留空，不臆测）：
  - 数据目录：`profile_dir/`
  - 来源：分布式训练 Profiling（PyTorch 框架 profiler，2 节点 × 4 卡）
  - 是否 LLM 训练：是
  - 模型 / 用途：Qwen2-7B 架构 LLM 训练（hidden=3584、28 注意力头、GQA 4 KV 头、head_dim=128；Qwen2 / Qwen2.5 具体版本未确证）
  - 落盘大小：~1.0 GB（8 卡原始落盘，单卡 trace_view ~119 MB + DB ~43 MB）；evidence 举证副本 ~260 MB
  - 来源链接：
  - 识别依据：算子签名 `FlashAttentionScore(/Grad)`+`RmsNorm(/Grad)`+`SwiGlu(/Grad)`+`RotaryPositionEmbedding(/Grad)` → 现代 Transformer LLM；含 `*Grad`+`ApplyAdamWV2` → 训练（非推理）。advisor/`kernel_details` 中 FlashAttention InputShapes `4096,1,3584`、heads `1,28,4096,8`、GQA KV 维 `512` → hidden=3584、28 q 头、4 kv 头、head_dim=128，命中 Qwen2-7B 架构。具体词表大小（vocab）未在落盘中确证，Qwen2 vs Qwen2.5 版本与精确参数规模留空不写。
- **输出位置**：`./profile_dir_profiling_analysis_20260618/`（report.md、Analysis Process.md、msprof_analyze/ 集群分析与 advisor 原始输出、intermediate/ 查询脚本、evidence/ 举证副本）。
- **举证文件清单**（已复制至 `evidence/`，报告自包含）：

  | 副本路径 | 原始来源 | 引用的问题点 | 大小 |
  |---|---|---|---|
  | `evidence/rank6_s3_localhost/kernel_details.csv` | `profile_dir\localhost.localdomain_10701_...\ASCEND_PROFILER_OUTPUT\kernel_details.csv` | 3.1 / 3.5 | ~10 MB |
  | `evidence/rank6_s3_localhost/trace_view.json` | `...\localhost.localdomain_10701_...\trace_view.json` | 3.1 | ~119 MB |
  | `evidence/rank6_s3_localhost/step_trace_time.csv` | `...\localhost.localdomain_10701_...\step_trace_time.csv` | 3.1 | <1 KB |
  | `evidence/rank0_s0_ubuntu122/kernel_details.csv` | `profile_dir\ubuntu122_14296_...\ASCEND_PROFILER_OUTPUT\kernel_details.csv` | 3.1 / 3.4 | ~10 MB |
  | `evidence/rank0_s0_ubuntu122/trace_view.json` | `...\ubuntu122_14296_...\trace_view.json` | 3.1 / 3.2 / 3.3 | ~119 MB |
  | `evidence/rank0_s0_ubuntu122/step_trace_time.csv` | `...\ubuntu122_14296_...\step_trace_time.csv` | 3.1 | <1 KB |
  | `evidence/rank4_s2_localhost/communication_matrix.json` | `profile_dir\localhost.localdomain_10699_...\communication_matrix.json` | 4（链路健康） | ~30 KB |
  | `evidence/rank4_s2_localhost/communication.json` | `...\localhost.localdomain_10699_...\communication.json` | 4（链路健康） | ~166 KB |
