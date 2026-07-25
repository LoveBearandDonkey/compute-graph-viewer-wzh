# Level2 4 卡训练 Profiling 诊断报告（PP=2 / DP=2 / TP=1）

## 1. 结论速览

- **性能健康度**：63 / 100 (B+) → 优化后预估 **75 / 100 (A)** — 计算 39% · 通信 65% · 调度 82% · 内存 N/A · 均衡 46%
- **结论**：单步 810 ms 被 **Pipeline 流水级负载不均** 完全主导：PP 末级（rank 2/3，含 LM Head + Loss）计算 ~401 ms，PP 首级（rank 0/1）计算仅 ~231 ms，多出来的 ~170 ms 让首级被迫在 P2P 接收上空等 ~158 ms。
- **头号瓶颈**：rank 2/3 比 rank 0/1 多 ~169 ms 的「LM Head MatMul + Cross-Entropy 链路」计算（MatMulV3/MIX_AIC 36.8 ms × 1 step、Exp/Sub/RealDiv 各 ~16 ms × 1 step、Cast/Mul 额外 ~35 ms），整段未在 PP 内任何位置与计算 / 通信重叠。
- **收益上限**：行动清单 P0/P1 全部落地后节省 ~20% 单步耗时（约 160 ms / step，810 → ~650 ms）

---

## 2. 行动清单

> 默认按"预期收益"从高到低排序；同档收益按"修改难度"从低到高排序。

| # | 优先级 | 问题 | 预期收益 | 修改难度 | 在哪改 / 对应位置 | 可视化视图 |
|---|---|---|---|---|---|---|
| 1 | P0 | **PP 末级承担 LM Head + Loss，比首级多 169 ms 计算**，首级 PP 接收侧空等 158 ms | -15% 单步耗时（约 120 ms） | 中 | Megatron / 自研训练框架的 `pipeline_model_parallel_split_rank` 或 `num_layers_in_first_pipeline_stage` 配置；将 1~2 层 Transformer Block 从 stage 0 迁移到 stage 1 之外，或把 LM Head/Loss 切到独立的 virtual PP stage | 通信视图 + Timeline 视图（系统调优） — 载入 `rank_0_ascend_pt/.../communication_matrix.json` 与 `rank_2_ascend_pt/.../trace_view.json`，对照 `hcom_batchSendRecv__128_4_1` 在 rank 0 长达 158 ms 的 wait 段 |
| 2 | P0 | **LM Head MatMulV3 单次 18.4 ms × 2**（MIX_AIC 路径），疑似走到非最优 tiling | -3% 单步耗时（约 25 ms，含两次正反向） | 中 | 末级 stage 的 `lm_head` 模块（`output_layer`/`tie_word_embeddings` 相关），检查输入 dtype、shape 对齐与 `aclnnMatmul` 走的算子选型 | 算子视图 + 详情视图 — 载入 `rank_2_ascend_pt/.../kernel_details.csv`，过滤 `MatMulV3` 排序看 Top2 18.4 ms 的实例，再用 `*.bin` 看 Roofline 落点是否 Memory Bound |
| 3 | P1 | **Cross-Entropy 链路（Exp/Sub/RealDiv/Cast/Mul）单 step 累计 ~80 ms 全部串行**，无算子融合 | -4% 单步耗时（约 35 ms） | 中 | 末级 stage 的 Loss 函数：把 `softmax → log → nll_loss` 重写为融合算子（`torch.nn.functional.cross_entropy` + Ascend 内置 `aclnnSoftmaxCrossEntropyWithLogits`），或替换为 `Megatron VocabParallelCrossEntropy` 的融合实现 | 算子视图 — 载入 `rank_2_ascend_pt/.../kernel_details.csv`，过滤 op_type ∈ {Exp, Sub, RealDiv}，确认这些算子均出现在反向期间且 shape 与 vocab_size 一致 |
| 4 | P1 | **DP 集合通信 HCCS 带宽仅 18.5 GB/s（理论 ~30 GB/s，利用率 62%）**，存在 ReduceScatter/AllGather 多次小包 | 通信效率 65% → 75%（约 -1.5% 单步耗时） | 低 | FSDP/ZeRO 的 `bucket_size_mb`（PyTorch DDP）或 `num_communication_buckets`（Megatron-DistOpt）；当前 collective transit size = 1750 MB 分 8+ 次发出，建议合并为 2-3 个大包 | 通信视图 — 载入 `rank_0_ascend_pt/.../communication.json`，按 `Size Distribution` 看 9 个 125 MB + 3 个 ~60 MB 的分布，确认是否能合并 |
| 5 | P2 | **rank 0/1 PP 接收完成后才开始反向**，未与 DP collective 重叠 | 中 | 中 | 检查反向中的 `tensor_parallel.cross_entropy` 是否阻塞 DP collective 流；启用 `--overlap-grad-reduce` / `--overlap-param-gather` | Timeline 视图（系统调优） — 载入 `rank_0_ascend_pt/.../trace_view.json`，找 `hcom_batchSendRecv__128_5_1` (154 ms) 与 `hcom_reduceScatter*` / `hcom_allReduce*` 是否能错峰 |
| 6 | P2 | **优化器单算子 ApplyAdamWV2 单次 3.68 ms 突发**（其余实例 < 200 us） | 低 | 低 | 该实例 max 远大于均值，疑似首/末参数张量未对齐 32B 或被 Cast 拖累；对照 op_statistic 看是不是 embedding/LMHead 大权重；可启用 `--use-distributed-optimizer` 让该 step 分片到多卡 | 算子视图 — 载入 `rank_0_ascend_pt/.../kernel_details.csv`，过滤 `ApplyAdamWV2` 排序后看最大耗时实例的 shape |

---

## 3. 问题详情

### 3.1 [P0] PP 末级承担 LM Head + Loss，比首级多 169 ms 计算

- **证据**：
  - `step_trace_time.csv`（所有 rank）：rank 0/1（PP stage 0）Computing = 231 / 231 ms；rank 2/3（PP stage 1）Computing = 400 / 401 ms。差值 ~170 ms。
  - `op_statistic.csv` 差分（详见 `intermediate/op_diff_r2_vs_r0.csv`）：rank 2 相对 rank 0 多出 169.2 ms 计算，几乎完全由以下算子贡献：
    - `MatMulV3/MIX_AIC` ×2 = 36.8 ms（rank 0 完全没有）→ LM Head 正反向各 1 次
    - `Cast` 额外 20.7 ms、`Mul` 额外 14.6 ms、`Sub` 额外 15.8 ms、`MatMulV3/AI_CORE` 额外 14.7 ms、`MatMulV2/AI_CORE` 额外 11.7 ms
    - `Exp` 16.1 ms、`RealDiv` 15.7 ms、`TransData` 10.4 ms、`ReduceSum` 6.9 ms、`ArgMaxWithValue` 6.8 ms（rank 0 全部为 0）→ Cross-Entropy / softmax / metric 算子
  - `communication.json` rank 0：`hcom_batchSendRecv__128_4_1` 单次 elapse 158.55 ms，但 transit 仅 0.40 ms，**wait 158.10 ms（占 99.7%）** → 是首级在等末级反向梯度发回。
- **影响**：单步 810 ms 中，PP 首级有 158 + 154 ≈ 312 ms 卡在 P2P send/recv 上空等（占单步 38%），所有 4 卡都被对齐到 810 ms。这是性能损失的 **第一原因**。
- **操作步骤**：
  1. 确认你的 PP 实现是否把 LM Head（vocab projection）与 Loss 绑定到了 last stage —— 大多数 Megatron-LM 风格框架默认如此。
  2. **方案 A（推荐，零侵入）**：在框架配置里调整 `num_layers_in_first_pipeline_stage` / `pipeline_model_parallel_split_rank`，把 1~2 个 Transformer Block 从 stage 1 转移到 stage 0；以 hidden=4096、layer 单次正反向 ~5 ms（rank 0 计算 231 ms / 14 层 ≈ 16 ms/层）为粗略基准，移走 1 层会让 stage 0 +16 ms、stage 1 -16 ms，新差距压到 ~140 ms；移走 2 层差距压到 ~110 ms（不够；同时要做下一步）。
  3. **方案 B（最佳，需要框架支持）**：开启 `--num-virtual-pipeline-stages`（VPP / interleaved 1F1B）把 stage 拆细，LM Head 单独占一个 micro virtual stage，使各 virtual stage 计算量均衡。
  4. **方案 C（兜底）**：若框架支持 `tensor_parallel.cross_entropy` 的 `parallel_output=True`，把 Loss 留在 stage 1 之外、并将 Embedding 与 LM Head 权重 tied（`tie_word_embeddings=True`）；可省一份 vocab × hidden 的参数与对应反向，但实际改动较大。
- **验证方法**：修改后重采 4 卡 profiling，断言 `step_trace_time.csv` 各 rank 的 Computing 列极差 < 10%（当前 73%）；同时观察 `communication.json` 中 `hcom_batchSendRecv__128_4_1` 的 Wait Time 应降到 < 30 ms。
- **可视化视图**：通信视图（主） + Timeline 视图（系统调优）（辅）
  - 主：通信视图 — 载入任一 rank 的 `communication_matrix.json`，关注 PP 通信域（rank 0↔2 / rank 1↔3）在 batchSendRecv 的耗时极端不对称（rank 0 elapse=313ms vs rank 2 elapse=73ms）。
  - 辅：Timeline 视图（系统调优）— 载入 `rank_0_ascend_pt/.../trace_view.json` 与 `rank_2_ascend_pt/.../trace_view.json` 横向对照，确认 rank 0 在 `hcom_batchSendRecv` 上的 158 ms 空泡正好对齐到 rank 2 的 MatMulV3/Exp/RealDiv 这一连串计算时段。

### 3.2 [P0] LM Head MatMulV3 单次 18.4 ms × 2，疑似走到非最优 Tiling

- **证据**：
  - `kernel_details.csv` (rank 2)：Top 列表中有 2 个 `MatMulV3` 实例耗时 18432 / 18386 us，远高于 stage 中其它 `MatMulV3` 的均值 ~222 us。
  - 这两次实例的 Core Type 是 **MIX_AIC**（混合 AI Core + Vector），而其余 280 个 MatMulV3 实例走的是纯 `AI_CORE`，说明算子选型不同。
  - LM Head 的 shape 通常是 `[batch×seq, hidden] @ [hidden, vocab]`，vocab 维度通常很大（30k-150k），未对齐到 16/32 的内置 tiling 容易触发 MIX_AIC 兜底路径。
- **影响**：单 step 反向期间额外 36.8 ms，占单步 4.5%。
- **操作步骤**：
  1. 在 `lm_head` 前后插一段打印 `tensor.shape` / `tensor.dtype` 的 hook，确认 vocab 维度与 hidden 维度。
  2. 若 vocab 不是 32 的倍数，把 vocab 向上 padding 到 32 的倍数（Megatron 的 `--make-vocab-size-divisible-by 128` 是常用做法）。
  3. 若仍走 MIX_AIC，把 `lm_head` 显式 cast 成 `bfloat16` 后传入 matmul，可能改走纯 AI_CORE Cube。
- **验证方法**：重采后 `kernel_details.csv` 中不再有 > 5 ms 的 MatMulV3 实例。
- **可视化视图**：算子视图（主） + 详情视图（辅）
  - 主：算子视图 — 载入 `rank_2_ascend_pt/.../kernel_details.csv`，按 Duration 倒序，确认两个 18 ms 实例的 Input/Output Shapes 列就是 LM Head 矩阵。
  - 辅：详情视图 — 拿这两个实例对应的算子 `*.bin`（需通过 msOpProf 二次采集），查看 Roofline 落点定位 Cube/MTE 利用率瓶颈。

### 3.3 [P1] Cross-Entropy 链路 ~80 ms 全部串行无融合

- **证据**：
  - `op_statistic.csv` (rank 2) 中以下算子在 rank 0 完全不存在：`Exp` 16.05 ms、`RealDiv` 15.71 ms、`ArgMaxWithValue` 6.84 ms、`ReduceSum` 6.89 ms；以及 `Sub` 从 rank 0 的 0.14 ms 暴涨到 15.91 ms。
  - 累计这 5 类多出来的纯 Vector 算子 = ~61 ms；加上额外的 Cast 21 ms、Mul 15 ms 大概率也是被 cross-entropy 计算图引入，合计 **~97 ms** 的 Vector 串行计算。
  - 这套算子是经典 "softmax → log → gather → mean" 展开形态。
- **影响**：单 step 额外 ~80 ms，占单步 ~10%。
- **操作步骤**：
  1. 检查训练脚本里 Loss 是手写的 `(-y * log_softmax(x)).sum()` 还是直接 `F.cross_entropy(...)`。
  2. 若是手写，替换为 `torch.nn.functional.cross_entropy(logits, labels, ignore_index=...)`，torch_npu 会路由到 `aclnnSoftmaxCrossEntropyWithLogits` 单算子（一次 fused）。
  3. 若用 Megatron 的 `tensor_parallel.cross_entropy`，确认开启了 `parallel_output=True` 与 `fp32_residual_connection` 不冲突，避免被强制 Cast。
  4. ArgMaxWithValue + ReduceSum 通常用于 accuracy/perplexity 指标计算；若不强求每 step 都算，把它们挪到 `if step % eval_interval == 0` 内。
- **验证方法**：重采后 `op_statistic.csv` (rank 2) 中 Exp / RealDiv 应消失或合并为 `SoftmaxCrossEntropyWithLogits` 等单条目；rank 2 Computing 应再减 ~30 ms。
- **可视化视图**：算子视图 — 载入 `rank_2_ascend_pt/.../kernel_details.csv`，过滤 Type ∈ {Exp, RealDiv, Sub, ArgMaxWithValue}，时间轴上看这些算子是否完全串行紧邻 LM Head 之后。

### 3.4 [P1] DP 集合通信 HCCS 带宽 18.5 GB/s，理论利用率 62%

- **证据**：
  - 4 卡 `communication.json` 的 collective Total Op Info 全部呈 1749.8 MB / 92.17 ms transit → 平均 **18.54 GB/s**（HCCS 理论 ~30 GB/s，利用率 ~62%）。
  - 包大小分布 (`Size Distribution`)：9 个 125.8 MB 大包 + 3 个 ~60 MB 中包 + 若干小包；Large Packet Ratio 普遍 1.0 已经不错，但拆得过细仍有合并空间。
- **影响**：集合通信总 92 ms transit 中，若带宽提升到 25 GB/s（83% 理论值），可省 ~25 ms。受限于 PP imbalance 主导，这部分目前被 PP wait 掩盖；在 P0 落地后才能完全显现。
- **操作步骤**：
  1. 若用 PyTorch FSDP：把 `forward_prefetch=True` + `backward_prefetch=BACKWARD_PRE`，并增大 `auto_wrap_policy` 的 `min_num_params` 到 1e8 减少 bucket 数。
  2. 若用 Megatron 分布式优化器：调大 `--ddp-bucket-size` 至 4e8 或更大，让小 bucket 合并。
  3. 检查 HCCL 算法：`export HCCL_ALGO=...`（当前 env 为空，走默认），单机 4 卡时 `ring` 算法对 large packet 通常更优。
- **验证方法**：重采后 `communication.json` collective HCCS Bandwidth ≥ 25 GB/s，且 transit time ≤ 75 ms。
- **可视化视图**：通信视图 — 载入 `rank_0_ascend_pt/.../communication.json`，展开 `collective` 看 Size Distribution，确认包数与单包大小。

### 3.5 [P2] PP 反向接收完成后才开始 DP collective，未与 P2P 重叠

- **证据**：
  - rank 0 在 `hcom_batchSendRecv__128_5_1`（154.89 ms）结束后，紧跟着 `hcom_reduceScatter__097_4_1`（26.31 ms）→ `_5_1`（18.59 ms）→ `allReduce__170_1_1`（34.62 ms） → `allGather__097_6_1`（12.02 ms）。`Start Timestamp` 严格串行，无明显并行段。
  - 若启用 `overlap-grad-reduce`，DP ReduceScatter 应在反向计算最后一层完成时就开始，与 P2P send 重叠。
- **影响**：估算可节省 ~30 ms（DP collective 总耗时 ~92 ms × 30% 可并行段）。
- **操作步骤**：
  1. Megatron：加 `--overlap-grad-reduce --overlap-param-gather`。
  2. PyTorch FSDP：确保 `use_orig_params=True` 且没有强制 `barrier()`。
  3. 检查训练脚本里是否手动加了 `torch.npu.synchronize()`。
- **验证方法**：Timeline 中 P2P send/recv 与 DP collective 应出现并行段；step time 再降 ~30 ms。
- **可视化视图**：Timeline 视图（系统调优）— 载入 `rank_0_ascend_pt/.../trace_view.json`，过滤 `hcom_` 前缀，看 P2P 与 collective stream 是否错峰。

### 3.6 [P2] ApplyAdamWV2 单实例 3.68 ms 突发

- **证据**：`op_statistic.csv` (rank 0)：`ApplyAdamWV2` count=57，avg=158.7 us，**max=3680.76 us**（约 23× 均值）。
- **影响**：单 step 额外 ~3.5 ms，占比小（< 0.5%）。
- **操作步骤**：
  1. 在 `kernel_details.csv` 找到该实例，看 shape；若是 `[vocab × hidden]` 的 embedding/LM Head 权重，说明优化器更新被集中到这一个张量上。
  2. 启用 `--use-distributed-optimizer`（Megatron）或 `optim_state_sharding`（FSDP）把 Adam 状态切片，让 57 张量分摊到 4 卡。
- **验证方法**：重采后该算子 max 应降到 < 500 us。
- **可视化视图**：算子视图 — 载入 `rank_0_ascend_pt/.../kernel_details.csv`，过滤 Type=ApplyAdamWV2 排序，找出 3680 us 那一条的 Input Shapes。

---

## 4. 已确认无问题

- **Host 下发延迟**：rank 0 `launch` API 均值 15 us，最大 713 us（少量异常），KernelLaunchWithHandle 平均 5.8 us → 下发不构成瓶颈。
- **HCCS 链路硬件**：4 卡 collective transit bandwidth 全部 18.5–18.6 GB/s 高度一致，不存在 SDMA "慢链路"型异常（rank 间差异 < 0.5%）。
- **算子下发拥塞 / CANN-Device 错位**：`aclrtSynchronizeDevice` 长耗时（158 ms）本质是 P2P 等待，不是同步语义异常。
- **HBM/PTA 显存峰值**：rank 0 PTA Reserved peak = 10.37 GB；rank 2 = 10.87 GB，未接近 Atlas 910B 32GB 上限，**未排查** HBM 带宽利用率（无对应采集字段，本次 Level2 + `_l2_cache=false` 配置无法计算）。
- **Free Time 极差**：rank 0/1/2/3 Free = 158 / 137 / 132 / 148 ms，相对均匀，**不存在伪快卡 / Host 下发型慢卡**。

---

## 5. 数据与方法

- **分析日期**：2026-05-26
- **数据路径**：`d:/Projects/ProfilingTest/Profiling_output/level2/`（rank 0/1/2/3，PyTorch 框架 profiler，Level2 采集，torch_npu 2.7.1 + CANN 8.3.RC1）
- **数据范围**：单卡单机 4 卡训练，PP=2 / DP=2 / TP=1 / EP=1；采集 step 13 单步（warmup 12 + active 1）。step 总长 ~810 ms。
- **并行通信域**（从 `profiler_metadata.json` 提取）：default = {0,1,2,3}；dp_cp = {0,1}（PP stage 0 内 DP）；mp = pp = {0,2}（rank 0↔2 PP 通信）；tp = {0}（无 TP）。
- **数据完整性**：
  - 4 rank 均含 `profiler_info_*.json` → 采集正常 Stop
  - 4 rank 均含 `ascend_pytorch_profiler_*.db`（7-8 MB） + `analysis.db`（28-40 KB） + `communication.json` + `step_trace_time.csv` + `communication_matrix.json`
  - **rank 1 / rank 3 缺少** `trace_view.json` / `kernel_details.csv` / `op_statistic.csv` / `api_statistic.csv` / `memory_record.csv` / `npu_module_mem.csv` / `operator_memory.csv` —— 推测是 `_data_simplification=true` 配置下，只为每个 dp_cp / pp 通信域保留一份明细。本次以 rank 0（PP-stage 0 代表）与 rank 2（PP-stage 1 代表）的明细为主分析对象，rank 1/3 通过 `step_trace_time.csv` + `communication.json` 验证一致性。
- **关键采集配置**：`_profiler_level=Level2`、`_aic_metrics=ACL_AICORE_PIPE_UTILIZATION`、`_l2_cache=false`、`profile_memory=true`、`record_shapes=true`、`with_stack=false`。**未启用** L2 Cache 与 system I/O 指标，因此本次无法独立计算 HBM 带宽利用率与 L2 命中率（PHS 内存项 → N/A）。
- **使用的 Skills**：
  - `mindstudio_profiler_data_check`：校验 4 rank 的 profiler_info / 关键交付件完整性
  - `ascend-profiler-db-explorer`：本次未直接对 DB 出 SQL（环境无 sqlite3 CLI / 无 Python），已通过 CSV/JSON 等价产物完成分析；如需更精细的 kernel-level connection 关联，建议在装好 sqlite3 / Python 的环境下再用 Track A `compute_view`/`comm_view`/`dispatch_view` 三宏复跑
  - `cluster-fast-slow-rank-detector`：按 SOP 流程 1 完成"输入数据类型 + Rank 个数 + 文件齐全度"判定；SOP 流程 2 / 4 见下方 Advisor 状态
  - `performance-health-score`：按集群慢卡场景权重（计算 0.20 · 通信 0.30 · 调度 0.30 · 内存 0.10 · 均衡 0.10，内存 N/A 重归一）计算 PHS
  - `msinsight-view-selector`：为每个问题点匹配可视化视图三要素
- **Advisor 状态**：
  - `msprof-analyze advisor` / `msprof-analyze -m cluster_time_summary|compute_op_sum|hccl_sum|slow_rank|slow_link|cann_api_sum`：**失败** — 当前 Windows 主机上 `msprof-analyze` 与 `python` 均不可执行（Python 仅为 WindowsApps store stub，sqlite3 CLI 也不存在），未能落地 `cluster_analysis_output`。建议在已部署 CANN + msprof-analyze 的 Linux 环境上对 `Profiling_output/level2/` 重新执行 `msprof-analyze -m all -d ./Profiling_output/level2 -o ./ascend_analysis_20260526/cluster_analysis_output`，将其结果与本报告 §3 交叉验证。本报告所有结论均基于 CSV / JSON 原始采集文件（per_rank_summary + op_diff），证据链可独立成立。
- **输出位置**：`./ascend_analysis_20260526/`
  - `report.md` — 本报告
  - `intermediate/per_rank_summary.csv` — 4 卡 step_trace + 通信总览
  - `intermediate/op_diff_r2_vs_r0.csv` — PP stage 1 vs stage 0 算子级差分（按 delta 倒序，Top 30）

### PHS 计算细节

按 `performance-health-score` 公式（集群慢卡场景）：

| 子项 | 数值 | 来源 | 权重（重归一后） | 加权 |
|---|---|---|---|---|
| 计算利用率 | 39 | 4 rank 平均 Computing/step = (231+231+400+401)/4 / 810 = 39% | 0.20 → 0.222 | 8.7 |
| 通信效率 | 65 | HCCS 实测 18.54 GB/s ÷ 理论 30 GB/s = 62%，按 transit-size 加权约 65% | 0.30 → 0.333 | 21.7 |
| 调度效率 | 82 | (1 - Free/total) = 1 - 143.6/810 = 82.3%（Wait 计入通信不重叠，未重复扣除）| 0.30 → 0.333 | 27.3 |
| 内存利用率 | N/A | `_l2_cache=false` 且未采集 HBM 带宽计数器 | 0.10 → 0 | — |
| 集群均衡度 | 46 | 用 Computing 极差 (1 - (401-231)/316) = 46%（注：用 step time 极差恒为 ~100%，因被同步对齐，失去诊断意义）| 0.10 → 0.111 | 5.1 |
| **PHS** | **63 (B+)** | 加权求和四舍五入 | — | **62.8** |

**优化后预估**（P0/P1 全部落地）：
- 计算 39 → 45（PP 均衡后整体 NPU 忙率上升）；通信 65 → 75（包合并 + HCCL 算法调优）；调度 82 → 89（Free 缩到 ~80 ms / 700 ms）；均衡 46 → 95（PP 两级差距压到 < 10%）
- 加权后 ≈ 75 → 等级 **A**（75-90）

