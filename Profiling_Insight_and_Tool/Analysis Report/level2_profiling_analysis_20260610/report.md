# level2 单机多卡训练性能诊断报告

> 数据：单机 4 卡（device 0/1/4/5）PyTorch Profiler Level2 DB，采集 1 个 step（step 13）。
> 并行拓扑：PP=2（流水线组 (0,2)、(1,3)）× DP=2，含 mp/embd/tp 通信域。
> 分析日期：2026-06-10

## 1. 结论速览

- **性能健康度**：49 / 100 (B) → 优化后预估 **61 / 100 (B+)** — 计算 39% · 通信 63% · 调度 50% · 内存 N/A
- **结论**：单步 ~810 ms，被**流水线 bubble + 零计算通信重叠**主导；4 卡 step 耗时几乎一致（极差 0.17%，无传统慢卡），瓶颈是**结构性空泡**而非某张卡慢。
- **头号瓶颈**：PP 末级（rank 2/3）计算 400 ms ≫ 首级（rank 0/1）231 ms，导致首级在 P2P (`batchSendRecv`) 上空等 **~313–322 ms（占单步 ~39%）**；末级超载主因是 **LM-head 词表投影 + 未融合的交叉熵**（Cast/Exp/Sub/RealDiv/ArgMax/ReduceSum 全部跑在 vocab=151936 上，仅末级独有 ~160 ms）。
- **收益上限**：行动清单 P0/P1 全部落地后节省 ~15–20% 单步耗时（约 130–160 ms）。

## 2. 行动清单

> 默认按"预期收益"从高到低排序；同档收益按"修改难度"从低到高排序。

| # | 优先级 | 问题 | 预期收益 | 修改难度 |
|---|---|---|---|---|
| 1 | P0 | PP 末级交叉熵/LM-head 未融合，vocab=151936 上堆叠 ~160 ms 向量算子 | -10~15% 单步耗时 | 中 |
| 2 | P0 | PP 阶段切分不均：末级计算 400 ms vs 首级 231 ms，bubble ~313 ms | -8~12% 单步耗时 | 中 |
| 3 | P1 | 计算与通信零重叠（Overlapped = 0），通信全暴露 | -5% 单步耗时 | 中 |
| 4 | P1 | micro-batch 数偏少，放大 PP warmup/cooldown bubble | -5~8% 单步耗时 | 低 |
| 5 | P2 | 动态 shape 算子 NonZero 触发 host 强制同步 + 未设 host 缓存环境变量 | -2~4% 单步耗时 | 低 |
| 6 | P2 | 亲和 API 未用：优化器可替换为 `torch_npu.optim.NpuFusedAdamW` 融合接口 | 低 | 低 |
| 7 | P2 | AI Core 频率 1650 MHz（低于标称 1800 MHz） | 计算吞吐 +~8%（若为throttling） | 低 |

## 3. 问题详情

### 3.1 [P0] PP 末级交叉熵/LM-head 未融合，vocab=151936 上堆叠 ~160 ms 向量算子

- **证据**：`compute_op_sum`（`ComputeOpPerRankStatsByOpName`）显示以下算子的 `InputShapes` 含 `151936`（词表维），且 **Rank 仅出现 2 与 3（末级）**，首级（rank 0/1）完全没有：
  - `Exp` `4096,1,151936`：单 step ~16.1 ms（rank2/3 各 2 次，单次 ~8.0 ms）
  - `Sub` `4096,1,151936;4096,1,1`：~15.7 ms　·　`RealDiv` 同形状：~15.7 ms　·　`Mul` 同形状：~14.6 ms
  - `Cast` `4096,1,151936`：~24.0 ms（单次 ~6.0 ms）
  - `ReduceSum` `4096,1,151936;1`：~6.9 ms　·　`ArgMaxWithValue` `4096,1,151936`：~6.8 ms　·　`TransData` `4096,151936`：~10.4 ms　·　`Add` `151936,1024;151936,1024`：~5.1 ms
  - 词表投影 `MatMulV3` `4096,1024;151936,1024`(dgrad) ~14.8 ms + `4096,151936;151936,1024` ~18.5 ms（MIX_AIC）
  - 合计末级独有 vocab 相关向量/投影开销 **~160 ms**，正是末级（400 ms）相对首级（231 ms）多出的 ~169 ms 的主体。
  - **advisor 佐证**（rank2 `mstt_advisor`）：Overall Summary 将末级计算 400.4 ms 拆为 **Vector 180.1 ms（占 E2E 22.24%）** > Matmul 131.6 ms > FlashAttention 87.5 ms——Vector 是最大计算项；且 advisor 的 Vector"bound 算子"清单把 `Cast/Exp/Sub/RealDiv/Mul 4096,1,151936` 全部判为 **`vec_mte2_mte3`（访存）bound**，词表投影 `MatMulV3 4096,1024;151936,1024` 判为 **`mte2` bound**——即这批算子卡在 HBM 进出而非算力，正是"融合省往返"的典型靶子。
- **影响**：这部分耗时直接抬高 PP 末级 stage time，是 3.2 中 bubble 的根因；按单步 810 ms 计，约占 **20%**。交叉熵被拆成 Cast→Exp→Sub→RealDiv→Mul→ReduceSum→ArgMax 一长串独立 kernel，每个都要把 `[4096,151936]` 大张量在 HBM 往返一遍，访存浪费极大（advisor 实测访存 bound 印证）。
- **修复建议**：
  - **改动位置**：模型 loss 计算层（last pipeline stage 的 `CrossEntropyLoss` / logits 处理），算子 `aclnnExp/aclnnSub/aclnnDiv/aclnnMaxDim` 调用点。
  1. 用**融合交叉熵**替换手写 softmax+CE（如 fused/online-softmax cross-entropy，或按 chunk 计算 logits-loss，避免一次性物化 `[4096,151936]` 的 fp32 中间张量）。
  2. 词表投影考虑 **vocab parallel**（按词表维切分到 DP/TP），减小单卡 N 维与中间张量。
  3. 复核是否有不必要的 `Cast`（fp16/bf16↔fp32），融合 CE 后多数可省。
- **问题修改完成的验证方式**：重采 profiling，确认 `ComputeOpPerRankStatsByOpName` 中末级 `Exp/Sub/RealDiv/ArgMax 4096,1,151936` 系列消失或合并，末级 `computation` 从 ~400 ms 降至接近首级（~250 ms 内）。
- **问题举证视图**：
  - 主：算子视图 — 载入 `evidence/rank_2_ascend_pt/kernel_details.csv`（源：`level2/rank_2_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv`），按耗时排序确认 `Exp/Sub/RealDiv/Cast 4096,1,151936` 与词表 `MatMulV3` 的占比。
  - 辅：Timeline 视图（系统调优）— 载入 `evidence/rank_2_ascend_pt/trace_view.json`（源：`level2/rank_2_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json`），定位 step 尾部 logits/loss 段连续的大向量 kernel。

### 3.2 [P0] PP 阶段切分不均：末级计算 400 ms vs 首级 231 ms，bubble ~313 ms

- **证据**：
  - `cluster_time_summary`（`ClusterTimeSummary`）：computation = rank0 231.3 / rank1 230.9 / **rank2 400.4 / rank3 401.1** ms；`communicationWaitStageTime` = **rank0 318.3 / rank1 339.9** / rank2 175.8 / rank3 159.7 ms，而 `communicationTransmitStageTime` 四卡几乎相同（~101.8 ms）→ 通信里真正传输只占 ~102 ms，其余全是**等待**。
  - `communication_time_sum`（`ClusterCommunicationTime`）：rank0 `hcom_batchSendRecv__128_4_1` 单次 **158.5 ms（wait 158.1）**、`__128_5_1` **154.9 ms**；rank1 对应 164.1 ms / 158.4 ms。首级两次 P2P 空等合计 ~313–322 ms。
  - `HcclPerRankStats`：`hcom_batchSendRecv_` 在 rank0/1 各 **313.9 / 323.0 ms**，在 rank2/3 仅 72.7 / 75.0 ms — 等待全压在首级。
- **影响**：首级 ~313 ms（占单步 **~39%**）纯空泡；4 卡 step 仍同为 ~810 ms，是被这段 bubble"对齐"出来的，并非真有效计算。
- **修复建议**：
  - **改动位置**：PP 流水线切分配置（Megatron 类：`--num-layers-per-virtual-pipeline-stage` / 末级 layer 数 / embedding+loss 归属），PP 组 (0,2)、(1,3)。
  1. 将末级的 LM-head/loss 负载与 transformer 层重新均衡：把 1–2 层 transformer 从首级**移到**末级以外，或反之让末级少算 transformer 层来补偿 loss 开销，使两级 stage time 接近。
  2. 配合 3.1 减小末级 loss 开销后再做细粒度均衡。
- **问题修改完成的验证方式**：重采后 `ClusterTimeSummary.computation` 各 rank 极差 < 10%，`communicationWaitStageTime` 首级从 ~318 ms 降到与末级同量级（< 180 ms）。
- **问题举证视图**：Timeline 视图（系统调优）— 载入 `evidence/rank_0_ascend_pt/trace_view.json`（源：`level2/rank_0_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json`），过滤 `hcom_batchSendRecv`，观察 Ascend Hardware 泳道两段 ~155 ms 的连续空白（首级等末级）；与 `evidence/rank_2_ascend_pt/trace_view.json` 同期对照，末级此时正在算 loss。

### 3.3 [P1] 计算与通信零重叠（Overlapped = 0），通信全暴露

- **证据**：`step_trace_time.csv` 与 `ClusterTimeSummary` 中 **四卡 `Overlapped` / `communicationOverlapComputation` 全为 0**；`Communication(Not Overlapped)` = 261–442 ms，即所有通信都串行暴露在关键路径上。
- **影响**：即便扣除 bubble 等待，纯传输 ~102 ms/卡 也完全未被计算掩盖；这部分若与反向计算并发可基本隐藏。
- **修复建议**：
  - **改动位置**：分布式通信重叠开关（梯度 reduce-scatter / all-gather 与反向计算 overlap；P2P 与计算并发），通信流是否独立 stream。
  1. 开启梯度通信-计算 overlap（如 `overlap_grad_reduce` / `overlap_param_gather`）。
  2. 确认 HCCL 通信走独立 stream，且 P2P 与下一 micro-batch 计算可并发。
- **问题修改完成的验证方式**：重采后 `ClusterTimeSummary.communicationOverlapComputation` > 0，`Communication(Not Overlapped)` 较当前下降。
- **问题举证视图**：Timeline 视图（系统调优）— 载入 `evidence/rank_0_ascend_pt/trace_view.json`（源：`level2/rank_0_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json`），对齐 Ascend Hardware 与 Communication 泳道，确认通信块下方无计算块覆盖。

### 3.4 [P1] micro-batch 数偏少，放大 PP warmup/cooldown bubble

- **证据**：`ClusterCommunicationTime` 中首级 P2P 仅 `__*_3/_4/_5` 三段（其中两段 ~155 ms 为大空等），bubble 呈大块而非被多 micro-batch 摊薄；理论 bubble ≈ (p-1)/(p-1+m)，pp=2 时 m 越小 bubble 越大。
- **影响**：与 3.2 叠加，micro-batch 少使 warmup/cooldown 三角区占比偏高。
- **修复建议**：
  - **改动位置**：训练脚本 global/micro batch 配置（`--micro-batch-size` / `--global-batch-size` 决定的 micro-batch 数）。
  1. 在显存允许范围内增大 micro-batch 数，稀释 bubble；可配合 interleaved 1F1B 调度。
- **问题修改完成的验证方式**：重采后首级 `communicationWaitStageTime` 占比随 micro-batch 数增加而下降，bubble 率趋近理论值。
- **问题举证视图**：Timeline 视图（系统调优）— 载入 `evidence/rank_0_ascend_pt/trace_view.json`（源：`level2/rank_0_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json`），观察 warmup/cooldown 三角空白相对稳态段的占比。

### 3.5 [P2] 动态 shape 算子 NonZero 触发 host 强制同步 + 未设 host 缓存环境变量

- **证据**：`communication_bottleneck` 多条记录的 reason 反复指向 `[Device-bound] ... max start-time-diff op aclnnNonzeroV2_NonzeroAiCore_NonZero, diff 280.7 / 281.0 / 359.3 us`；`cann_api_sum` 中 `aclnnNonzeroV2` count 96、`aclrtSynchronizeDevice/DeviceSynchronize` 合计 ~46%、`StreamSynchronize` 19% — host 侧大量阻塞同步。NonZero 输出 shape 数据相关，必须 host 回读，强制 device 同步。**advisor 佐证**：`Operator Dynamic Shape Issues` 报"找到大量动态 shape 算子"；`Environment Variable Issues`（High）指出 `ACLNN_CACHE_LIMIT`、`HOST_CACHE_CAPACITY` 未设——动态 shape 下 host 侧重复编译/下发未被缓存。
- **影响**：每次 NonZero 同步打断算子异步下发，叠加在 bubble 上放大空泡；动态 shape 还使 host 反复编译，是同步类 API 占比畸高的诱因之一。
- **修复建议**：
  - **改动位置**：①训练脚本入口（环境变量 / 编译模式）；②产生 `NonZero`/`aclnnNonzeroV2` 的代码（通常为 mask/索引、`torch.nonzero`、动态 padding 逻辑）。
  1. 入口设 `torch_npu.npu.set_compile_mode(jit_compile=False)`、`torch_npu.npu.config.allow_internal_format = False`（advisor 建议）。
  2. 设环境变量 `export ACLNN_CACHE_LIMIT=100000`、`export HOST_CACHE_CAPACITY=20`，缓存动态 shape 的 host 下发，降低 host bubble。
  3. 用静态 shape 等价实现替换 `nonzero`（如固定上界 + mask 选择），消除 host 回读同步；复核不必要的 `.item()`/`.cpu()` 引发的 `aclrtSynchronizeDevice`。
- **问题修改完成的验证方式**：重采后 `cann_api_sum` 中 `aclnnNonzeroV2` 与 `aclrtSynchronizeDevice` 调用次数/耗时显著下降；advisor 不再报动态 shape / 环境变量问题。
- **问题举证视图**：Timeline 视图（系统调优）— 载入 `evidence/rank_2_ascend_pt/trace_view.json`（源：`level2/rank_2_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json`），过滤 host 侧 `aclrtSynchronizeDevice` 与 `NonZero`，看其与 device 空白的对齐。

### 3.6 [P2] 亲和 API 未用：优化器可替换为 `torch_npu.optim.NpuFusedAdamW` 融合接口

- **证据**：advisor `Affinity API Issues`（schedule）提示当前可替换的亲和接口 `torch_npu.optim.NpuFusedAdamW`、`torch_npu.npu_confusion_transpose`；对应 `compute_op_sum` 中 `ApplyAdamWV2`（AI_VECTOR_CORE，各卡 ~9 ms/step）与多处 `TransData/Cast`。
- **影响**：未用融合优化器/亲和接口时，optimizer step 与 transpose 走非最优实现，host 下发条数与 device kernel 数偏多（量级较小，故 P2）。
- **修复建议**：
  - **改动位置**：优化器构造处（`AdamW` → `torch_npu.optim.NpuFusedAdamW`）、含 confusion-transpose 的算子调用点。
  1. 替换为 `torch_npu.optim.NpuFusedAdamW`，减少 optimizer 阶段 kernel 数与下发开销。
  2. 评估 `torch_npu.npu_confusion_transpose` 替换现有 transpose 逻辑。
- **问题修改完成的验证方式**：重采后 advisor 不再列出该亲和 API；`ApplyAdamWV2` 相关下发/耗时下降。
- **问题举证视图**：Timeline 视图（系统调优）— 载入 `evidence/rank_2_ascend_pt/trace_view.json`（源：`level2/rank_2_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json`），定位 optimizer 段 `ApplyAdamWV2` 系列 kernel 与 host launch 间隙。

### 3.7 [P2] AI Core 频率 1650 MHz（低于标称 1800 MHz）

- **证据**：`freq_analysis`（`AbnormalFrequencyRanks`）四卡 `aicoreFrequency` 均为 **1650 MHz**，被工具判为异常（既非 1800 MHz 满频也非 800 MHz 空闲）。
- **影响**：若为功耗/温度 throttling，相对 1800 MHz 满频约损失 ~8% 计算吞吐；也可能是该芯片型号的实际工作频率（需结合硬件手册确认）。
- **修复建议**：
  - **改动位置**：节点功耗/散热与 NPU 频率策略（运维层，非代码）。
  1. 用 `npu-smi info` 查看实时频率/温度/功耗，确认是否 throttling；排查散热与功率上限设置。
- **问题修改完成的验证方式**：满载下 AI Core 频率稳定在 1800 MHz；`freq_analysis` 不再标记异常。
- **问题举证视图**：详情视图 — 暂无直接 .bin 落盘文件；以 `evidence/rank_0_ascend_pt/kernel_details.csv`（源：`level2/rank_0_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv`）中 AI Core 算子实测耗时辅助佐证（同 shape 算子是否普遍偏慢）。

## 4. 已确认无问题

- **无传统慢卡**：`slow_rank` 未产出慢卡记录；`ClusterTimeSummary.stepTime` 四卡 809.9–810.8 ms，极差 0.17% < 5%，集群在 step 级高度均衡。
- **通信链路带宽无异常离群**：`communication_matrix_sum` 中 HCCS 带宽 18.7–20.9 GB/s、LOCAL 326–333 GB/s，各同类链路一致，集合通信 `large_packet_ratio≈1.0`，无单条慢链路；带宽偏中等但不是头号瓶颈（瓶颈是暴露与等待，见 3.2/3.3）。
  > 备注（advisor Packet Analysis）：SDMA 通道 64.29% 数据包 < 16 MB、偏小包、host-bound（仅 0.819 ms，量级小）。若后续要进一步压通信，可评估 ZeRO3→ZeRO2/ZeRO1 降低切分粒度、增大通信包；本次因占比小未列入行动清单。
- **计算型慢卡排除**：同 shape 算子跨卡耗时一致（如 `FlashAttentionScore` 各卡 867–883 µs、`FlashAttentionScoreGrad` 各卡 2.24–2.29 ms），无单卡算子劣化；末级耗时高是**负载内容不同**（多了 loss），非硬件劣化。
- **MoE 负载**：`ep_load_balance` 无数据（非 MoE 模型），不适用。
- **未排查项**：内存带宽利用率（HBM 实测带宽）未采集（`aic_metrics=PIPE_UTILIZATION`，未含 HBM 带宽计数），PHS 内存子项记 N/A。

## 5. 数据与方法

- **分析日期**：2026-06-10
- **数据路径**：`level2/rank_{0,1,2,3}_ascend_pt/ASCEND_PROFILER_OUTPUT/`
- **数据范围**：4 卡（device 0/1/4/5），step 13 单步；torch_npu 2.7.1 / CANN 8.3.RC1；Profiler Level2，`record_shapes=true`、`profile_memory=true`、`aic_metrics=ACL_AICORE_PIPE_UTILIZATION`，采集正常 Stop（`profiler_info_*.json` 齐全）。
- **并行拓扑**（`CommunicationGroupMapping`）：pp(p2p) (0,2)/(1,3)；dp (2,3)、dp_cp (0,1)/(2,3)；mp (0,2)/(1,3)；embd (0,2)/(1,3)；default (0,1,2,3)。判定 PP=2、首级=rank0/1、末级=rank2/3。
- **芯片峰值参考**：Ascend 910B 系列 FP16/BF16 ~376 TFLOPs/s（型号未在元数据中明确，MFU 类结论为近似）。
- **使用的 Skills**：
  - `mindstudio_profiler_data_check`：校验数据完整性（valid，DB+部分 text 交付件齐全）
  - `ascend_pytorch_profiler_db_explorer`：理解 DB 表结构与算子/通信视图
  - `cluster-fast-slow-rank-detector`：快慢卡判定（结论：无慢卡，结构性 bubble）
  - `timeline-swimlane-analyzer`：重叠率/bubble/暴露通信等时序结构指标
  - `op-mfu-calculator`：计算利用率估算（占用率代理）
  - `performance-health-score`：PHS 评分
  - `msinsight-view-selector`：为每个诊断结果推荐 Insight 视图
  - `dataset-source-identifier`：识别并记录落盘数据来源/模型/用途（含识别依据，无依据留空不猜）
- **数据来源与落盘信息**（落盘文件信息卡片；无确切识别依据的项留空，不臆测）：
  - 数据目录：`level2/`
  - 来源：分布式训练 Profiling
  - 是否 LLM 训练：是
  - 模型 / 用途：Qwen 系列 LLM（vocab=151936）
  - 落盘大小：~47 MB（evidence 举证副本）
  - 来源链接：[level2.rar](https://gitcode.com/zhangruoyu2/msinsight-quick-start-demo/blob/main/GUI-test-data/training/single-node/level2.rar)
  - 识别依据：vocab=151936 命中 Qwen 系列专属 tokenizer；evidence 中含 `FlashAttentionScore/Grad`、`RmsNormGrad`、`ApplyAdamWV2` 等训练/反向算子 → 判定为 LLM 训练。具体参数规模（hidden/层数）因 lm_head 等被 PP/TP 切分、无法从单卡 shape 反推全模型，留空不写。
- **Advisor 状态**：
  - `msprof-analyze advisor`：**已调用** — 对瓶颈末级卡执行 `msprof-analyze advisor all -d ./level2/rank_2_ascend_pt -o ./level2_profiling_analysis_20260610/advisor/rank_2 --force`，结果（`advisor/rank_2/mstt_advisor_*.html` + `log/*.xlsx`）已并入第 3 章证据：① Overall Summary 印证 E2E 809.8 ms 拆解与 Vector 180.1 ms 为最大计算项（→3.1）；② Vector/Cube bound 清单证明 vocab 算子为 `vec_mte2_mte3`/`mte2` 访存 bound（→3.1）；③ Dynamic Shape + Environment Variable Issues（→3.5）；④ Affinity API（NpuFusedAdamW，→3.6）；⑤ Packet Analysis 提示 SDMA 64.29% 小包、host-bound（建议 ZeRO3→ZeRO2/1，见第 4 章备注）。
  - 集群多模式 recipe（cluster_time_summary / hccl_sum / compute_op_sum / communication_time_sum / communication_matrix_sum / communication_bottleneck / cann_api_sum / free_analysis / freq_analysis / slow_rank）为主证据来源；advisor 为单卡补充深诊。
- **执行的命令**（输出落 `level2_profiling_analysis_20260610/msprof_analyze/<mode>/`）：
  `msprof-analyze -m <mode> -d ./level2 -o ./level2_profiling_analysis_20260610/msprof_analyze/<mode>`，mode 覆盖上列 10 项（`slow_link` 在本版本不存在，已用 `communication_matrix_sum`/`communication_bottleneck` 替代）。
- **PHS 计算说明**：场景=大模型多卡训练（默认权重 计算0.40/通信0.30/调度0.20/内存0.10）；内存子项 N/A，按比例归一化为 计算0.444/通信0.333/调度0.222。
  - 计算利用率 39%：以"compute 占 step 比例"`mean(231,231,400,401)/810≈39%`作为 AI Core 占用率代理（`with_flops=false`，无法直接算 MFU；占用率被 bubble 压低）。
  - 通信效率 63%：主力 HCCS 实测 ~19 GB/s ÷ 理论 ~30 GB/s。
  - 调度效率 50%：有效占比 `(compute+transmit)/step ≈ (315.7+102)/810≈52%`，取 ~50%。
  - PHS = 0.444×39 + 0.333×63 + 0.222×50 ≈ 49（B）；优化后按 P0/P1 收益反推 计算→55、调度→72、通信 63 ⇒ ≈ 61（B+）。
- **输出位置**：`./level2_profiling_analysis_20260610/`（report.md / Analysis Process.md / msprof_analyze/ / intermediate/ / evidence/）。
- **举证文件清单**（已复制至 `evidence/`，报告自包含）：

  | 副本路径 | 原始来源 | 引用的问题点 | 大小 |
  |---|---|---|---|
  | `evidence/rank_2_ascend_pt/kernel_details.csv` | `level2/rank_2_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv` | 3.1 | 845 KB |
  | `evidence/rank_2_ascend_pt/trace_view.json` | `level2/rank_2_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json` | 3.1/3.2/3.5 | 16 MB |
  | `evidence/rank_0_ascend_pt/trace_view.json` | `level2/rank_0_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json` | 3.2/3.3/3.4 | 15 MB |
  | `evidence/rank_0_ascend_pt/kernel_details.csv` | `level2/rank_0_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv` | 3.6 | 808 KB |
  | `evidence/rank_0_ascend_pt/communication_matrix.json` | `level2/rank_0_ascend_pt/ASCEND_PROFILER_OUTPUT/communication_matrix.json` | 4 | 12 KB |
  | `evidence/rank_2_ascend_pt/communication_matrix.json` | `level2/rank_2_ascend_pt/ASCEND_PROFILER_OUTPUT/communication_matrix.json` | 4 | 16 KB |
  | `evidence/rank_0_ascend_pt/memory_record.csv` | `level2/rank_0_ascend_pt/ASCEND_PROFILER_OUTPUT/memory_record.csv` | 备用 | 1.4 MB |
  | `evidence/rank_0_ascend_pt/operator_memory.csv` | `level2/rank_0_ascend_pt/ASCEND_PROFILER_OUTPUT/operator_memory.csv` | 备用 | 612 KB |

### 时序结构指标看板（timeline-swimlane-analyzer）

> 由 timeline-swimlane-analyzer 从 trace 泳道几何派生；仅列已测得的指标，未采集的不列（本次单步 step 13 数据，无法给出 step 抖动等多步指标）。前端"总览—指标看板"消费本表。

| 指标 | 值 | 状态 | 说明 |
|---|---|---|---|
| 关键路径占比 | 62% | warn | 末级 (compute+transmit)/step；约 38% 为 bubble/free |
| 计算-通信重叠率 | 0% | bad | Overlapped 全为 0，通信全暴露 |
| 暴露通信 | 102 ms |  | 纯传输暴露/卡；另有 ~248ms 为 PP bubble 等待(communicationWaitStageTime) |
| PP 流水线 bubble 率 | 39% | warn | 首级 batchSendRecv 空等 ~313ms / 810ms |
