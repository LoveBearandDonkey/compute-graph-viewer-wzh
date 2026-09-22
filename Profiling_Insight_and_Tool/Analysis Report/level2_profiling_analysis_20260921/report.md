# level2 单机 4 卡 PP=2×DP=2 训练性能诊断报告

> 数据：`D:\Projects\msagent\data\level2`，单机 4 卡（device 0/1/4/5）PyTorch Profiler **Level2** text 导出，采集 **1 个 step（step 13）**。
> 并行拓扑：TP=1 · **PP=2** · DP=2 · CP=1 · EP=1，pp 组 (0,2)/(1,3)，dp_cp 组 (0,1)/(2,3)，embd 组 (0,2)/(1,3)。
> 软件栈：torch_npu 2.7.1 · CANN 8.3.RC1 · Ascend910B（AI Core 1650 MHz）。
> 分析日期：2026-09-21

## 1. 结论速览

- **性能健康度**：32 / 100 (C) → 优化后预估 **45 / 100 (B)** — 计算 7% · 通信 63% · 调度 50% · 内存 7%
- **结论**：单步 810 ms，**4 卡耗时几乎完全一致（极差 0.17%，没有慢卡）**，瓶颈是结构性的——一个 ~0.6B 的模型被切成 PP=2 且只喂 2 个 micro-batch，**计算-通信重叠率为 0**，每卡真正在算的时间只占 28%（首级）/ 49%（末级）。
- **头号瓶颈**：PP 首级（rank 0/1）在 `hcom_batchSendRecv_` 上**纯空等 313.9 / 323.0 ms（占单步 38.7% / 39.9%）**；根因是末级计算 400 ms ≫ 首级 231 ms，而末级多出的 169 ms 几乎全部来自**未融合的交叉熵在 vocab=151936 上的 167 ms 访存型 kernel**。
- **收益上限**：行动清单 P0/P1 全部落地后，同 token 量下单步 810 ms → **~380–450 ms（-45~53%，吞吐 1.8~2.1×）**

## 2. 行动清单

> 默认按“预期收益”从高到低排序；同档收益按“修改难度”从低到高排序。

| # | 优先级 | 问题 | 预期收益 | 修改难度 |
|---|---|---|---|---|
| 1 | P0 | 0.6B 量级模型上启用 PP=2，首级在 P2P 上纯空等 313 ms（占单步 39%） | -35~45% 单步耗时 | 低 |
| 2 | P0 | 交叉熵未融合，末级在 vocab=151936 上堆叠 167 ms 访存型 kernel | -12~20% 单步耗时 | 中 |
| 3 | P0 | PP 阶段负载失衡：末级计算 400 ms vs 首级 231 ms（1.73×） | -8~12% 单步耗时 | 低 |
| 4 | P1 | 计算与通信零重叠（Overlapped = 0），DP 优化器通信 65 ms 全暴露 | -6~8% 单步耗时 | 低 |
| 5 | P1 | micro-batch 数与 micro-batch-size 严重欠配（2×1），显存仅用 12.7 GB | MFU 7% → 15~20% | 低 |
| 6 | P1 | tied embedding 梯度 AllReduce 622 MB/step，末级暴露 121 ms | -4~8% 单步耗时 | 中 |
| 7 | P2 | `aten::is_nonzero` 触发 D2H 强制同步，末级单步阻塞 109 ms | -2~4% 单步耗时 | 低 |
| 8 | P2 | 小 kernel 下发间隙密集：50 µs–1 ms 空挡 926 段共 135 ms（占单步 17%） | -3~5% 单步耗时 | 低 |
| 9 | P2 | HCCS size-weighted 带宽 19.0 GB/s，仅 30 GB/s 理论值的 63% | -2~3% 单步耗时 | 中 |

## 3. 问题详情

### 3.1 [P0] 0.6B 量级模型上启用 PP=2，首级在 P2P 上纯空等 313 ms（占单步 39%）

- **证据**：
  - `step_trace_time.csv`（4 卡）：`Overlapped = 0.0`、`Bubble = 0`，`Communication(Not Overlapped)` 为 rank0 420.2 ms / rank1 441.7 ms / rank2 277.6 ms / rank3 261.5 ms。
  - `communication.json`（rank0）P2P 明细：`hcom_batchSendRecv__128_4_1` elapse **158.55 ms**，其中 `Wait/Synchronization = 158.10 ms`，`Transit Size = 8.4 MB`；`hcom_batchSendRecv__128_5_1` elapse **154.89 ms**，其中 `Idle = 154.89 ms`，同样只传 8.4 MB。按 HCCS 实测 20.9 GB/s，8.4 MB 的真实传输仅 **0.4 ms** → 这两次 P2P 的 **99.7% 是空等**。
  - 由 DB 重建的 device 时序（`intermediate/comm_sequence.txt`）：rank0 计算流最大两段空挡为 `+154.6 ms 起 160.5 ms` 与 `+397.1 ms 起 156.3 ms`，正好对齐上述两次 `batchSendRecv`。
  - 模型几何按算子 shape 反推：28 层 × (hidden 1024 · q 2048 / kv 1024 GQA 16/8 头 · ffn 3072 gated) + vocab 151936 ≈ **0.6B 参数**；P2P 载荷 8.39 MB = 4096 × 1024 × 2 B 恰好是 **1 个 micro-batch 的 [seq 4096, hidden 1024] bf16 激活**，即 mbs=1、micro-batch 数=2。
  - PP=2 且 m=2 时理论 bubble 比 = (p-1)/(m+p-1) = **33%**，实测 39%（叠加了 3.3 的阶段失衡）。
- **影响**：首级两张卡各有 ~313–323 ms（单步 39%）完全不干活。这是本次单步耗时的最大单一构成项，且 PP 在此规模模型上没有换来任何显存收益（见 3.5：峰值仅 12.7 GB）。
- **修复建议**：
  - **改动位置**：训练启动脚本的并行度参数（Megatron / MindSpeed 系 `--pipeline-model-parallel-size`、`--data-parallel-size` 所在处）
  1. **首选：直接取消流水并行** —— `--pipeline-model-parallel-size 1`，4 卡全部走 DP（DP=4）。0.6B 模型单卡峰值 Reserved 仅 12.7 GB，单卡完整装下 28 层毫无压力；PP 在这里只带来 bubble 和 embd AllReduce（3.6）。
  2. 若因框架/脚本约束必须保留 PP：把 micro-batch 数从 2 提到 ≥ 8（提高 `--global-batch-size`，配合 `--micro-batch-size`），bubble 比从 33% 降到 ~11%；并开启交错流水 `--num-layers-per-virtual-pipeline-stage`（VPP）+ `--overlap-p2p-comm`，把 P2P 藏进计算。
  3. 复采时留意 `step_trace_time.csv` 的 `Bubble` 列当前恒为 0，说明框架未单独上报 bubble，bubble 被并入了 `Communication(Not Overlapped)`；判断是否改善要看 P2P 的 wait/idle，而不是看 `Bubble` 列。
- **问题修改完成的验证方式**：复采 1 个 step，确认 rank0/rank1 的 `communication.json` 中 `p2p / Total Op Info` 的 `Wait Time + Idle Time` 从 313 ms 降到 < 40 ms，且 `step_trace_time.csv` 的 `Computing / Stage` 占比从 28.5% 升到 > 55%。
- **问题举证视图**：Timeline 视图（系统调优） — 载入 `evidence/rank_0_ascend_pt/trace_view.json`（源：`D:\Projects\msagent\data\level2\rank_0_ascend_pt\ASCEND_PROFILER_OUTPUT\trace_view.json`），关注 HCCL 泳道上 `+155 ms` 与 `+397 ms` 两条 `hcom_batchSendRecv_` 长条，及其下方 AI Core 泳道的整段空白；与 `evidence/rank_2_ascend_pt/trace_view.json`（源：`D:\Projects\msagent\data\level2\rank_2_ascend_pt\ASCEND_PROFILER_OUTPUT\trace_view.json`）对照，看末级同期在满负荷计算。

### 3.2 [P0] 交叉熵未融合，末级在 vocab=151936 上堆叠 167 ms 访存型 kernel

- **证据**（`evidence/rank_2_ascend_pt/kernel_details.csv`，筛选 shape 含 `151936`，完整明细见 `intermediate/vocab_ops.txt`）：

  | 耗时 | 次数 | 均次 | BlockDim | 核类型 | aiv_mte2 | 算子 · 形状 |
  |---|---|---|---|---|---|---|
  | 36.82 ms | 2 | 18.41 ms | 20 | MIX_AIC | 0.13 | `MatMulV3 4096,151936;151936,1024 → 4096,1024`（logits dgrad） |
  | 24.01 ms | 4 | 6.00 ms | 40 | AI_VECTOR_CORE | **0.97** | `Cast 4096,1,151936` |
  | 16.05 ms | 2 | 8.03 ms | 40 | AI_VECTOR_CORE | **1.00** | `Exp 4096,1,151936` |
  | 15.75 ms | 2 | 7.87 ms | 40 | AI_VECTOR_CORE | **0.96** | `Sub 4096,1,151936;4096,1,1` |
  | 15.69 ms | 2 | 7.85 ms | 40 | AI_VECTOR_CORE | **0.96** | `RealDiv 4096,1,151936;4096,1,1` |
  | 14.81 ms | 2 | 7.40 ms | 20 | AI_CORE | — | `MatMulV3 4096,1024;151936,1024 → 4096,151936`（logits fwd） |
  | 14.61 ms | 2 | 7.31 ms | 40 | AI_VECTOR_CORE | **0.95** | `Mul 4096,1,151936;4096,1,1` |
  | 8.31 ms | 2 | 4.16 ms | 40 | AI_VECTOR_CORE | **0.96** | `TransData 4096,151936 → NZ` |
  | 6.87 / 6.84 / 5.12 / 1.98 ms | 各 2 | — | 40 | AIV / MIX_AIV | 0.58~1.00 | `ReduceSum` / `ArgMaxWithValue` / `Add 151936,1024` / `TransData NZ→ND` |
  | **合计 167.2 ms** | | | | | | **占该卡 kernel 总时长 678.1 ms 的 24.7%** |

  - `aiv_mte2_ratio` 0.95~1.00 而 `aiv_vec_ratio` 仅 0.03~0.11 → 这批 kernel **卡在 HBM 进出，不是算力不足**；`BlockDim = 40` 说明多核切分已饱和，调 tiling 没有空间，唯一出路是**减少张量往返次数**。
  - 单个 `[4096, 151936]` fp32 张量 = **2.49 GB**，`Cast → Exp → Sub → RealDiv → Mul → ReduceSum → ArgMax` 这条链每个 kernel 读+写各一遍 ≈ 5 GB HBM 流量，一个 micro-batch 光 softmax/CE 就要搬 **~35 GB**。
  - 这 167 ms 在首级（rank0/1）**完全不存在**（`intermediate/kernel_efficiency.txt` 中 rank0 无任何 151936 形状算子），正是 3.3 阶段失衡的全部来源。
- **影响**：直接抬高 PP 末级 stage time，进而放大 3.1 的首级空等；按单步 810 ms 计约占 **20.6%**。
- **修复建议**：
  - **改动位置**：训练启动脚本的 loss 相关开关 + `megatron/core/fusions`（或 MindSpeed 对应融合实现）的启用路径
  1. 开启**融合交叉熵**：Megatron-LM 系为 `--cross-entropy-loss-fusion`（部分版本另有 `--cross-entropy-fusion-impl`），MindSpeed 侧为其 fused CE 开关 —— **开关名按所用版本的 `arguments.py` 核对**。目标是让 logits 的 softmax + NLL 在一个 kernel 内完成，省掉中间 6~7 次 2.49 GB 级往返。
  2. 避免 fp32 全量 logits：确认 CE 输入没有被显式 `.float()` 提前 Cast 成 fp32 全量张量（现象就是那 4 次 `Cast 4096,1,151936`，24 ms）；融合实现内部按行做 fp32 归约即可。
  3. TP=1 无法走词表并行的情况下，可考虑 chunked CE（按 seq 分块算 loss），把峰值中间张量从 2.49 GB 降到 1/N。
- **问题修改完成的验证方式**：复采后在 `kernel_details.csv` 中 grep `151936`，确认 `Exp / Sub / RealDiv / Mul / Cast` 这批形状的 kernel 消失或合计 < 40 ms；`op_statistic.csv` 中 rank2 的 `Cast` 总耗时从 26.7 ms 降到 < 5 ms；rank2 `step_trace_time.csv` 的 `Computing` 从 400 ms 降到 ~280 ms。
- **问题举证视图**：算子视图 — 载入 `evidence/rank_2_ascend_pt/kernel_details.csv`（源：`D:\Projects\msagent\data\level2\rank_2_ascend_pt\ASCEND_PROFILER_OUTPUT\kernel_details.csv`），按 Duration 降序并用 Input Shapes 过滤 `151936`，关注这 12 个算子族合计 167 ms 以及它们的 `aiv_mte2_ratio` 接近 1；配合 内存视图 — 载入 `evidence/rank_2_ascend_pt/operator_memory.csv`（源：同目录 `operator_memory.csv`），关注 `[4096,151936]` 量级临时张量的反复申请/释放。

### 3.3 [P0] PP 阶段负载失衡：末级计算 400 ms vs 首级 231 ms（1.73×）

- **证据**（`step_trace_time.csv` 四卡 + `intermediate/timeline_geometry.txt`）：

  | rank | PP stage | step (ms) | 计算 (ms) | 暴露通信 (ms) | 其中真实传输 | 其中等待/同步 | device 空挡 (ms) | 重叠 |
  |---|---|---|---|---|---|---|---|---|
  | 0 | 首级 | 810.6 | 231.3 (28.5%) | 420.2 (51.8%) | ~92.9 | **~327.3** | 159.2 (19.6%) | **0** |
  | 1 | 首级 | 809.9 | 230.9 (28.5%) | 441.7 (54.5%) | ~92.9 | **~348.9** | 137.3 (17.0%) | **0** |
  | 2 | 末级 | 810.3 | **400.4 (49.4%)** | 277.6 (34.3%) | ~92.8 | ~184.8 | 132.2 (16.3%) | **0** |
  | 3 | 末级 | 810.8 | **401.1 (49.5%)** | 261.5 (32.3%) | ~92.8 | ~168.7 | 148.2 (18.3%) | **0** |

  - 末级比首级多算 **169.1 ms**，而 3.2 统计的 vocab 相关 kernel 恰为 **167.2 ms** —— 失衡 99% 由 LM-head + 未融合 CE 解释，**不是层数切分不均**（两级各 14 层，由 `FlashAttentionScore` 各 28 次 = 14 层 × 2 micro-batch 反推）。
  - 核类型分布佐证：首级 cube 类（AI_CORE + MIX_AIC）145.8 ms / vector 类 85.4 ms；末级 cube 208.7 ms / **vector 191.8 ms**，末级 vector 负载翻了一倍以上。
- **影响**：首级每个 micro-batch 都要等末级多出的 ~85 ms，两个 micro-batch 叠加即 3.1 中的 ~170 ms；是 bubble 从理论 33% 涨到实测 39% 的直接原因。
- **修复建议**：
  - **改动位置**：训练启动脚本的层切分参数
  1. **落地 3.2 的融合 CE 后本项自动消失大半**（169 ms → ~30 ms），因此优先做 3.2。
  2. 若仍需微调：用 Megatron 的不均匀切分参数把末级层数调少（`--decoder-last-pipeline-num-layers`，例如 16/12 切分），使两级 stage time 对齐；**版本不支持该参数时**改用 VPP（`--num-layers-per-virtual-pipeline-stage`）摊平。
  3. 最彻底的做法仍是 3.1 的 `PP=1`，失衡问题不再存在。
- **问题修改完成的验证方式**：复采后 4 卡 `step_trace_time.csv` 的 `Computing` 列极差 < 10%（当前 73%）。
- **问题举证视图**：Timeline 视图（系统调优） — 同时载入 `evidence/rank_0_ascend_pt/trace_view.json` 与 `evidence/rank_2_ascend_pt/trace_view.json`（源：`D:\Projects\msagent\data\level2\rank_0_ascend_pt\ASCEND_PROFILER_OUTPUT\trace_view.json` 与 `...\rank_2_ascend_pt\ASCEND_PROFILER_OUTPUT\trace_view.json`），对齐两卡 step 起点后关注 `+74~+312 ms` 区间：rank2 AI Core 泳道连续满载 238 ms，rank0 同期已空闲；配合 算子视图 — 载入 `evidence/rank_0_ascend_pt/kernel_details.csv`，确认首级完全没有 `151936` 形状算子。

### 3.4 [P1] 计算与通信零重叠（Overlapped = 0），DP 优化器通信 65 ms 全暴露

- **证据**：
  - `step_trace_time.csv` 四卡 `Overlapped` 列全为 `0.0`；由 DB 独立重建的 device 区间并集也给出**计算 ∩ 通信 = 0.0 ms**（`intermediate/timeline_geometry.txt`）。
  - rank0 通信时序（`intermediate/comm_sequence.txt`）显示 DP 优化器通信排在反向**之后**、完全串行：`+648.2→+674.5 reduceScatter(440.5 MB)` → `+677.9→+696.5 reduceScatter(311.2 MB)` → `+696.5→+731.1 allReduce(622.3 MB)` → `+769.7→+781.7 allGather(220.2 MB)` → `+781.7→+790.2 allGather(155.6 MB)`。
  - 其中 ReduceScatter 44.9 ms + AllGather 20.5 ms = **65.4 ms 是分布式优化器的梯度归约 / 参数收集**，按 18.7 / 19.5 GB/s 实测带宽计属真实传输（不是空等），因此只能靠**重叠**消除、不能靠调带宽。
  - `+560~735 ms` 区间统计：计算 76.2 ms / 通信 79.5 ms / 空闲 19.3 ms —— 计算和通信各占一半却零重叠，是最容易拿收益的区间。
- **影响**：rank0 单步 65.4 ms（8.1%）本可被反向计算完全掩盖；4 卡同构，整机等比例损失。
- **修复建议**：
  - **改动位置**：训练启动脚本的分布式优化器开关
  1. 开启 `--overlap-grad-reduce`（让 ReduceScatter 随反向逐 bucket 下发，藏进 backward）和 `--overlap-param-gather`（让 AllGather 随前向逐 bucket 展开）；两者都要求 `--use-distributed-optimizer` 已开——落盘出现 RS+AG 组合而非单次大 AllReduce，说明分布式优化器已在用。
  2. 配合 `--ddp-bucket-size` 调 bucket 粒度：当前单次 RS 传 440.5 MB（26.3 ms），粒度过粗不利于与反向交织，建议降到 ~50 MB 量级。
  3. 保留 PP 时补 `--overlap-p2p-comm`（需 VPP，见 3.1）。
- **问题修改完成的验证方式**：复采后 `step_trace_time.csv` 的 `Overlapped` 列 > 50 ms 且 `Communication(Not Overlapped)` 相应下降；`communication.json` 中 `reduceScatter` / `allGather` 的 elapse 总和不变（传输量没变）但落在计算区间内。
- **问题举证视图**：Timeline 视图（系统调优） — 载入 `evidence/rank_0_ascend_pt/trace_view.json`（源：`D:\Projects\msagent\data\level2\rank_0_ascend_pt\ASCEND_PROFILER_OUTPUT\trace_view.json`），关注 `+648~+790 ms` 区间 HCCL 泳道与 AI Core 泳道**从不同时点亮**；配合 通信视图 — 载入 `evidence/rank_0_ascend_pt/communication.json` 与 `evidence/rank_0_ascend_pt/communication_matrix.json`（源：同目录同名文件），关注 `collective` 分组下 `reduceScatter` / `allGather` 的 `Transit Size` 与 `Bandwidth`。

### 3.5 [P1] micro-batch 数与 micro-batch-size 严重欠配（2×1），显存仅用 12.7 GB

- **证据**：
  - `profiler_info_0.json`：`schedule = {skip_first: 12, wait: 0, warmup: 1, active: 1, repeat: 1}`，故落盘只有 step 13 一个 step。
  - micro-batch 结构由数据反推：`FlashAttentionScore` 每卡 28 次 ÷ 14 层 = **2 个 micro-batch**；P2P 载荷 8.39 MB = 4096 × 1024 × 2 B = **mbs 1 × seq 4096**；DP=2 → 全局 batch = 4 seq = **16 384 token/step**。
  - 显存峰值（`memory_record.csv`）：rank0 Allocated 10.28 GB / Reserved **11.93 GB**（碎片 13.8%）；rank2 Allocated 10.82 GB / Reserved **12.68 GB**（碎片 14.7%）。`NPU_INFO` 只给出 `Ascend910B` 未细分子型号，故按候选容量给区间：**32 GB 机型 ≈ 40% 占用，64 GB 机型 ≈ 20% 占用**（型号待确认）。
  - MFU（从 kernel 实际 shape 精算 GEMM FLOPs + FlashAttention 解析式，`intermediate/mfu.txt`）：rank0 17.56 TFLOP/step → 摊到整 step **21.7 TFLOPS**；rank2 25.21 TFLOP/step → **31.1 TFLOPS**；4 卡均值 **26.4 TFLOPS**，按 910B BF16 峰值 376 TFLOPS 计 **MFU = 7.0%**（按 `op-mfu-calculator` 子型号峰值表给区间：910B1 378.88 → 7.0%，910B3 294.91 → 8.9%）。
  - 对照：**只看 cube 类算子自己在跑的那 146 / 209 ms 窗口内，达成算力是 120.4 / 120.8 TFLOPS（占峰值 32%~41%）** —— 算子本身效率正常，MFU 低完全是“算子只占了 18%~26% 的墙钟时间”造成的。
- **影响**：MFU 7% 意味着买到的算力有 93% 没用上。batch 欠配同时是 3.1 bubble 的放大器（micro-batch 数越少 bubble 占比越高），并让每个 kernel 的固定开销（下发、同步、小 kernel 尾部）无法摊薄。
- **修复建议**：
  - **改动位置**：训练启动脚本的 batch 参数
  1. 先用 `npu-smi info` 确认单卡 HBM 实际容量。按 12.7 GB 峰值推算，32 GB 卡可把 `--micro-batch-size` 提到 2（预计 ~22 GB），64 GB 卡可提到 4（预计 ~44 GB）；同时提高 `--global-batch-size` 增加梯度累积步数，使 micro-batch 数 ≥ 8。
  2. 若要在不增显存的前提下增加 micro-batch 数，只提 `--global-batch-size`（保持 mbs=1），bubble 比即从 33% 降到 ~11%。
  3. 显存留有余量时**不要**额外开重计算（`--recompute-granularity`）；当前没有重计算痕迹（反向算子次数与前向 1:1），保持现状即可。
- **问题修改完成的验证方式**：复采后按同一脚本重算 MFU（`intermediate/mfu_from_shapes.mjs`），确认 4 卡均值从 26.4 TFLOPS 升到 ≥ 55 TFLOPS；`memory_record.csv` 峰值 Reserved 仍低于单卡容量的 85%。
- **问题举证视图**：内存视图 — 载入 `evidence/rank_2_ascend_pt/memory_record.csv` 与 `evidence/rank_2_ascend_pt/npu_module_mem.csv`（源：`D:\Projects\msagent\data\level2\rank_2_ascend_pt\ASCEND_PROFILER_OUTPUT\` 下同名文件），关注 `Total Reserved(MB)` 折线全程不超过 12 700 MB、距单卡容量有大段空白；配合 算子视图 — 载入 `evidence/rank_0_ascend_pt/kernel_details.csv`，关注 `MatMulV3` 的 `cube_utilization(%) = 93.5`（算子本身不差，问题在利用时间）。

### 3.6 [P1] tied embedding 梯度 AllReduce 622 MB/step，末级暴露 121 ms

- **证据**：
  - `communication.json`（rank2）：`hcom_allReduce__170_1_1` elapse **120.99 ms**，`Transit Size = 622.3 MB`，HCCS 实测 19.1 GB/s；同一算子在 rank0 上 elapse 仅 **34.62 ms**。
  - 622.3 MB = 151 936 × 1 024 × 4 B，正是 **fp32 词表权重梯度**；该算子跑在 `parallel_group_info` 的 **`embd` 组（`group_name_20`，global_ranks [0,2]）**，即 PP 首末级之间同步 tied word embedding 梯度。
  - 按 19.1 GB/s，622.3 MB 的真实传输是 **32.6 ms**；rank0 elapse 34.6 ms ≈ 真实传输，rank2 elapse 121.0 ms → **rank2 多出的 ~88 ms 是在等 rank0 到达这个同步点**（时序上 rank2 于 `+609.6 ms` 进入，rank0 于 `+696.5 ms` 才进入，两者同在 `+730.6 / +731.1 ms` 结束）。
  - 该 AllReduce 是 step 尾部最大的单个通信项，且紧邻优化器，无法与反向重叠。
- **影响**：末级 121 ms（14.9% 单步）中 32.6 ms 是无法避免的传输、~88 ms 是等待；即使只消除等待部分也能回收 ~11% 单步耗时。
- **修复建议**：
  - **改动位置**：训练启动脚本的 embedding 绑定 / 并行度参数
  1. **落地 3.1 的 `PP=1` 后该 AllReduce 直接消失**（首末级同卡，无需跨 stage 同步 embedding）——最干净的解法。
  2. 若保留 PP：改用 `--untie-embeddings-and-output-weights` 解绑输入 embedding 与输出投影，去掉这次 622 MB 同步（代价是参数量 +155M、可能影响收敛，需小规模验证 loss 曲线）。
  3. 保留 tied embedding 时，先做 3.3 把两级 stage time 对齐，`~88 ms` 的等待会随之收敛到 0，只剩 32.6 ms 传输。
- **问题修改完成的验证方式**：复采后确认 `communication.json` 中 `embd` 组（`group_name_20`）的 allReduce 消失，或其在 rank0 / rank2 上的 elapse 差值从 86 ms 收敛到 < 10 ms。
- **问题举证视图**：通信视图 — 载入 `evidence/rank_2_ascend_pt/communication.json` 与 `evidence/rank_2_ascend_pt/communication_matrix.json`（源：`D:\Projects\msagent\data\level2\rank_2_ascend_pt\ASCEND_PROFILER_OUTPUT\` 下同名文件），关注 `collective/hcom_allReduce__170_1_1` 的 `Elapse 120.99 ms` vs `Transit Size 622.3 MB`；配合 Timeline 视图（系统调优）载入 `evidence/rank_0_ascend_pt/trace_view.json` 与 `evidence/rank_2_ascend_pt/trace_view.json`，对比两卡进入该 AllReduce 的时刻差（`+696.5` vs `+609.6`）。

### 3.7 [P2] `aten::is_nonzero` 触发 D2H 强制同步，末级单步阻塞 109 ms

- **证据**（`intermediate/host_sync.txt`，源自 DB 的 `PYTORCH_API` 表）：
  - rank2：`aten::is_nonzero` 8 次合计 **109.4 ms**，其中 3 次分别阻塞 **41.1 ms（+148.9 ms）/ 40.7 ms（+390.2 ms）/ 25.9 ms（+554.1 ms）**；每次都伴随同起点同长度的 `aten::item` + `aten::_local_scalar_dense`，即典型的 device→host 标量回读。
  - rank0：`aten::is_nonzero` 4 次合计 35.0 ms，最长 29.4 ms（`+646.2 ms`，正落在梯度归约前）。
  - 同族阻塞型 API：rank2 `aten::to` 371 次 181.1 ms（单次最大 145.6 ms）、`aten::copy_` 256 次 176.0 ms —— 单次上百毫秒说明这些是**跨设备拷贝上的阻塞等待**，不是拷贝本身慢。
  - `META_DATA.ENV_VARIABLES`：`ACLNN_CACHE_LIMIT` 与 `HOST_CACHE_CAPACITY` **均为空**（未设置），host 侧算子缓存未调优。
- **影响**：每次 D2H 同步都会把 host 下发流水排空，之后需重新填满，是 3.8 中密集小空挡的成因之一。这部分与 device 计算有重叠，不能线性叠加到单步耗时，估算净收益 2~4%。
- **修复建议**：
  - **改动位置**：训练脚本中把 NPU 张量当 Python bool / 标量用的位置（`if some_tensor:`、`.item()`、`torch.any(...)` 后直接判断）；典型在 loss scaler / grad-norm 的 inf-nan 检查、early-stop 判据处
  1. 用 grep 定位：在训练代码目录内检索 `.item()`、`is_nonzero`、`torch.any(`、以及直接把张量放进 `if` 的写法，把 step 内的标量回读改成 device 侧张量运算，或合并成一次回读。
  2. 溢出检查改用 torch_npu 的 device 侧接口，避免每个 micro-batch 都回读一次。
  3. 设置 host 侧缓存环境变量：`export ACLNN_CACHE_LIMIT=10000`、`export HOST_CACHE_CAPACITY=20`（按实际 host 内存调），减少重复 aclnn 构图开销。
- **问题修改完成的验证方式**：复采后查 DB——`SELECT COUNT(*), SUM(endNs-startNs)/1e6 FROM PYTORCH_API p JOIN STRING_IDS s ON s.id=p.name WHERE s.value='aten::is_nonzero'`，确认 rank2 总时长从 109 ms 降到 < 10 ms。
- **问题举证视图**：Timeline 视图（系统调优） — 载入 `evidence/rank_2_ascend_pt/trace_view.json`（源：`D:\Projects\msagent\data\level2\rank_2_ascend_pt\ASCEND_PROFILER_OUTPUT\trace_view.json`），在 host PyTorch 泳道定位 `+148.9 / +390.2 / +554.1 ms` 处的 `aten::is_nonzero` 长条，关注其覆盖期间 host 侧不再有新的 `aclnnXxx` 下发。

### 3.8 [P2] 小 kernel 下发间隙密集：50 µs–1 ms 空挡 926 段共 135 ms（占单步 17%）

- **证据**（`intermediate/timeline_geometry.txt`，对 device 侧计算 + 通信区间取并集后统计空挡）：

  | 空挡量级 | rank0 段数 / 合计 | rank2 段数 / 合计 |
  |---|---|---|
  | < 10 µs | 1182 / 0.6 ms | 1473 / 0.6 ms |
  | 10–50 µs | 350 / 11.6 ms | 324 / 10.8 ms |
  | **50–200 µs** | **701 / 69.4 ms** | **616 / 60.9 ms** |
  | **0.2–1 ms** | **225 / 66.1 ms** | **170 / 50.2 ms** |
  | > 1 ms | 6 / 11.5 ms | 5 / 9.7 ms |
  | 合计空挡 | **159.2 ms (19.6%)** | **132.2 ms (16.3%)** |

  - 空挡**不是**集中在少数几段（rank0 最大空挡仅 2.6 ms），而是由 926 段 50 µs–1 ms 的碎片构成，合计 135.5 ms —— 典型的 **host 下发跟不上 device 消费**。
  - 分区间看最严重的是 step 尾部：`+735~811 ms（optimizer + 参数 AllGather）` 窗口 75.6 ms 中空闲 **32.7 ms（43%）**，该区间正是小 kernel 最密集处（`ApplyAdamWV2` 57 次、`LpNormV2` 59 次、`ZerosLike` 116 次、`Cast` 287 次）。
  - `empty_tensor` rank0 2478 次 / rank2 2578 次（36.2 / 33.9 ms host 时间），说明每 step 有数千次张量分配。
- **影响**：135 ms 碎片空挡中，可通过下发优化回收的估计为 1/3~1/2，即单步 3~5%。
- **修复建议**：
  - **改动位置**：训练启动脚本的环境变量 + 优化器构造处
  1. 开启 torch_npu 下发队列：`export TASK_QUEUE_ENABLE=2`（把算子下发放到独立线程，减少 host 侧串行开销）。
  2. 优化器换成融合实现：当前是逐参数 `aten::_fused_adamw_`（57 次）+ `ApplyAdamWV2`（57 次）+ `LpNormV2`（59 次 grad-norm），改用 `torch_npu.optim.NpuFusedAdamW` 把这些合并成少量大 kernel。
  3. 一并设置 3.7 的 `ACLNN_CACHE_LIMIT` / `HOST_CACHE_CAPACITY`。
  4. **绑核方案需补采数据后才能给具体 CPU range**：本次落盘只有 `HOST_INFO.hostName = localhost.localdomain`，没有 CPU 拓扑 / NUMA / cgroup cpuset 信息。请在训练主机上执行只读采集 `lscpu`、`npu-smi info -t topo`、`cat /sys/fs/cgroup/cpuset.cpus.effective`，拿到后走 `mindstudio-cpu-binding` 出具体亲和性方案。**在拿到 cgroup/cpuset 真实可用范围前不给推荐 CPU range**，避免推荐到容器不可用的核上。
- **问题修改完成的验证方式**：复采后用 `intermediate/timeline_geometry.mjs` 重算，确认 50 µs–1 ms 空挡合计从 135 ms 降到 < 90 ms，且 `+735~811 ms` 区间空闲率从 43% 降到 < 25%。
- **问题举证视图**：Timeline 视图（系统调优） — 载入 `evidence/rank_0_ascend_pt/trace_view.json`（源：`D:\Projects\msagent\data\level2\rank_0_ascend_pt\ASCEND_PROFILER_OUTPUT\trace_view.json`），放大 `+735~811 ms`，关注 AI Core 泳道上密集小 kernel 之间反复出现的亚毫秒空白，以及 host 泳道 `aclnnXxx` 下发与 device 执行的错位；配合 算子视图 — 载入 `evidence/rank_0_ascend_pt/kernel_details.csv`，按 Count 排序看 `Add(457) / Cast(287) / Slice(136) / ZerosLike(116)` 这类均次仅 20~45 µs 的碎片 kernel。

### 3.9 [P2] HCCS size-weighted 带宽 19.0 GB/s，仅 30 GB/s 理论值的 63%

- **证据**（`communication.json` 四卡 `Communication Bandwidth Info`）：

  | 通信域 | 算子 | 次数 | 单步传输量 | 实测带宽 | rank0 暴露 | rank2 暴露 |
  |---|---|---|---|---|---|---|
  | `pp` (0,2) | `hcom_batchSendRecv_` | 3 | 16.8 MB | 20.9 GB/s | 313.9 ms | 72.7 ms |
  | `dp_cp` | `hcom_reduceScatter_` | 2 | 751.7 MB | 18.7 GB/s | 44.9 ms | 49.7 ms |
  | `dp_cp` | `hcom_allGather_` | 2 | 375.8 MB | 19.5 GB/s | 20.5 ms | 22.3 ms |
  | `embd` (0,2) | `hcom_allReduce_` | 1 | 622.3 MB | 19.1 GB/s | 34.6 ms | 121.0 ms |
  | `mp` / `dp` / `default` | `hcom_allReduce_` ×9~15 | — | ~0 MB | — | ~6.3 ms | ~11.6 ms |

  - 全部流量走 **SDMA / HCCS（单机内）**，`RDMA / PCIE / SIO` 字段全为 0，无跨节点链路。
  - size-weighted 带宽 = Σ(带宽 × 字节) / Σ字节 = **19.03 GB/s**，对 HCCS 理论 30 GB/s 为 **63.4%**。
  - 小尺寸 allReduce（grad-norm 类，`Transit Size = 0 MB`）共 9~15 次，elapse 6~12 ms 几乎全是 `Wait/Synchronization`，属固定同步开销。
- **影响**：若带宽提到 ~26 GB/s（87% 理论），当前 ~93 ms 真实传输可降到 ~68 ms，单步收益 2~3%。这是本报告里**唯一“通信本身慢”的问题**——3.1 / 3.6 的大头是等待而非带宽。
- **修复建议**：
  - **改动位置**：HCCL 环境变量 + 机内互联链路配置
  1. 当前 `HCCL_ALGO` 为空（未指定），可显式尝试机内算法组合（如 `export HCCL_ALGO="level0:fullmesh"`）并 A/B 对比 ReduceScatter 的实测带宽。
  2. 用 `hccn_tool -i <id> -link -g` / `-net_health -g` 逐卡确认机内 HCCS 链路健康与速率协商正常，排除单条链路降速。
  3. 把 3.4 的 bucket size 调小后，单次传输从 440 MB 降到 ~50 MB，可减少长传输期间的带宽抖动。
  4. 小包 allReduce 合并：grad-norm 的多次 0 字节 allReduce 可合并成一次（Megatron 侧通常由分布式优化器的 norm 归约路径决定）。
- **问题修改完成的验证方式**：复采后重算 size-weighted 带宽（用 `communication.json` 的 `Transit Size(MB)` 与 `Bandwidth(GB/s)` 加权），确认 ≥ 24 GB/s。
- **问题举证视图**：通信视图 — 载入 `evidence/rank_0_ascend_pt/communication_matrix.json` 与 `evidence/rank_0_ascend_pt/communication.json`（源：`D:\Projects\msagent\data\level2\rank_0_ascend_pt\ASCEND_PROFILER_OUTPUT\` 下同名文件），关注 HCCS 链路的带宽热力与各算子 `Bandwidth(GB/s)` 分布（18.7~20.9 区间，无单点异常低值）。

## 4. 已确认无问题

- **没有慢卡**：4 卡 step 耗时 809.9 / 810.3 / 810.6 / 810.8 ms，极差 **0.17%**；瓶颈是结构性空泡，不是某张卡慢。
- **cube 算子自身效率正常**：`MatMulV3` cube_utilization 93.5%（首级）/ 96.7%（末级）、`aic_mac_ratio` 0.75 / 0.56；`MatMulV2` cube_utilization 97.0% / 97.9%、mac_ratio 0.83 / 0.84。GEMM 没有 tiling 或 shape 对齐问题。
- **FlashAttention 已启用融合算子**：`FlashAttentionScore` / `FlashAttentionScoreGrad`（MIX_AIC，cube_utilization 92.9% / 87.3%），未见 eager attention 被拆成 BMM+Softmax 的痕迹；atten_mask 为 `2048,2048` 压缩掩码，走的是 causal sparse 模式。
- **torch_npu 亲和算子已用到位**：`npu::npu_rms_norm` / `npu_rms_norm_backward`、`NpuFusionAttention`、`NPURotaryPositionEmbedding`、`SwiGluGrad`、`aten::_fused_adamw_` 均在调用链里（优化器仍有进一步融合空间，见 3.8）。
- **无 AI_CPU 算子**：核类型分布仅 `AI_CORE / AI_VECTOR_CORE / MIX_AIC / MIX_AIV / COMMUNICATION`，没有掉到 AICPU 的算子。
- **显存无 OOM 风险、分配器配置合理**：峰值 Reserved 11.93 / 12.68 GB，Reserved − Allocated 碎片 13.8% / 14.7%（正常区间），`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` 已开启。
- **无跨节点通信问题**：单机 4 卡，`RDMA / PCIE / SIO` 传输量全为 0，不存在慢链路 / 小包 RDMA 问题。
- **未排查项**（有数据但本次未下钻，不等于无问题）：L2 Cache 命中率 46.7%（`LLC` 表 mode=27）偏低，但缺算子级 `*.bin` 无法定性；AI Core 频率落盘为 1650 MHz（`AICORE_FREQ` 仅 2 个采样点），是否低于该子型号标称值需以 `npu-smi info -t board` 核对，样本量不足不下结论。

## 5. 数据与方法

- **分析日期**：2026-09-21
- **数据路径**：`D:\Projects\msagent\data\level2\`（rank_0/1/2/3_ascend_pt）
- **数据范围**：Rank 0–3（device 0/1/4/5），**仅 step 13 一个 step**（`schedule.skip_first=12, warmup=1, active=1, repeat=1`），单步 ~810 ms
- **数据校验结果**（`mindstudio_profiler_data_check`）：
  - 【类型】框架 profiler (PyTorch) | 多卡（4 卡）
  - 【状态】**valid** —— `profiler_info_{0..3}.json` 齐全（采集正常 Stop），`ASCEND_PROFILER_OUTPUT/` 均已解析
  - 【配置】`_profiler_level=Level2` · `_aic_metrics=ACL_AICORE_PIPE_UTILIZATION` · `record_shapes=true` · `profile_memory=true` · `with_stack=false` · `with_modules=false` · `_export_type=["text"]` · `_data_simplification=true` · `_l2_cache=false` · `_msprof_tx/_mstx=false`
  - 【缺失】**rank1 / rank3 只有 DB 与 communication / step_trace，缺 `trace_view.json` / `kernel_details.csv` / `memory_record.csv` / `op_statistic.csv`**（text 导出件仅 rank0 / rank2 完整）；无 `cluster_analysis_output/`
  - 【建议】补齐 rank1 / rank3 的 text 导出件；采集 ≥ 5 个稳态 step 以便算 step 抖动
  - 【下一步】已执行：全局拆解 → 通信 / 计算 / 时序 / 显存 / 下发 下钻
- **落盘数据来源识别**（`dataset-source-identifier`）：
  - **框架**：Megatron-LM 系（MindSpeed on Ascend）。依据：`PYTORCH_API` 含 `LinearWithGradAccumulationAndAsyncCommunication`（Megatron 并行线性层专有名）；`profiler_metadata.json` 的 `distributed_args` 含 `tensor_model_parallel_size / pipeline_model_parallel_size / expert_model_parallel_size / context_parallel_size`，`parallel_group_info` 的 `dp_cp / mp / tp / pp / embd` 组命名为 Megatron 风格；调用链使用 torch_npu 亲和算子 `npu_rms_norm / NpuFusionAttention / NPURotaryPositionEmbedding`。
  - **是否 LLM**：是（decoder-only 自回归训练）。依据：RMSNorm + GQA FlashAttention + SwiGLU MLP + tied embedding，末级带 `[*, 151936]` 词表投影与交叉熵链路。
  - **模型（按算子 shape 反推，非配置文件）**：28 层 × (hidden 1024 · q_dim 2048 / kv_dim 1024 → GQA 16 q 头 / 8 kv 头 × head_dim 128 · ffn 3072 gated) · vocab 151936 · bf16 ≈ **0.6B 参数**。**落盘中没有模型名或配置文件，故只给几何，不断言具体模型。**
  - **用途**：预训练 / 微调的性能采样（1 个 step，含完整 fwd + bwd + optimizer），非推理、非 RL。依据：存在 `Optimizer.step#AdamW.step`、`ApplyAdamWV2`、梯度 ReduceScatter / AllGather，无 vllm / sglang / verl 打点。
- **使用的 Skills**：
  - `mindstudio_profiler_data_check`：校验 Profiling 数据完整性与采集状态
  - `dataset-source-identifier`：识别落盘来源 / 框架 / 模型几何 / 用途
  - `msagent-profiler-breakdown`：全局阶段拆解（本次因环境无 Python，按其方法论用 Node 重写等效查询，见下“工具替代说明”）
  - `ascend-profiler-db-explorer`：Track A CTE 宏（compute_view / comm_view / dispatch_view）构造 DB 查询
  - `ascend-communication-analysis`：通信域归属、传输 vs 等待取证
  - `ascend-computation-analysis`：算子热点与 kernel 效率（cube_utilization / mac_ratio / mte2_ratio）
  - `timeline-swimlane-analyzer`：device 泳道几何、重叠率、空挡分布、PP bubble
  - `ascend-schedule-analysis`：下发间隙与 host 同步点定位
  - `memory-analysis`：显存峰值 / 碎片 / 容量利用率
  - `op-mfu-calculator`：从落盘算子 shape 精算 GEMM + FA FLOPs 与 MFU（含子型号峰值区间）
  - `mindstudio-cpu-binding`：**仅用于约束**——因缺 CPU 拓扑 / cgroup cpuset 数据，按其硬规则不给推荐 CPU range，改为在 3.8 给出只读采集命令
  - `performance-health-score`：PHS 评分（见第 1 章首行）
  - `msinsight-view-selector`：为每个诊断结果推荐 Insight 可视化视图
  - `profiling-workflow`：报告章节骨架与举证文件规范
- **Advisor 状态**：
  - `msprof-analyze advisor`：**失败** —— 本机 `python` / `python3` 仅为 Windows Store 应用执行别名（调用返回 exit 49，无真实解释器），`pip` 与 `msprof-analyze` 均不可用，因此 advisor 与所有 `msprof-analyze -m <recipe>` 派生统计（`cluster_analysis.db` / `FreeAnalysis` / `HcclTopOpStats` / `ComputeOpPerRankStats`）本次**无法生成**。请在装有 Python 3 与 mstt 工具链的环境执行 `msprof-analyze advisor all -d <数据路径>` 后并入本报告。
- **工具替代说明**：AscendProfKit 各 skill 附带的 `scripts/*.py`（`decompose.py` / `db_query.py` / `get_schema.py` 等）因上述 Python 缺失无法执行。本次改用 **Node 24 内置 `node:sqlite`**，按 skill 规定的表结构与 CTE 宏重写等效查询，脚本全部落盘在 `intermediate/`，可复跑。注意：ns 级时间戳超出 JS double 安全整数范围，脚本内统一用 BigInt 归一化到 step 起点后再转 Number（见 `timeline_geometry.mjs`）；结果已与 `step_trace_time.csv` 交叉校验一致（计算 231.29 vs 231.3 ms、通信 420.17 vs 420.2 ms）。
- **PHS 取值口径**（对应第 1 章首行）：
  - 计算 **7%** = MFU，4 卡实测均值 26.4 TFLOPS ÷ 910B BF16 峰值 376 TFLOPS。FLOPs 来自 `kernel_details.csv` 逐 kernel shape 精算（GEMM 按 2·M·N·K，含 NZ 格式还原）+ FlashAttention 解析式（causal 折半，反向按 2.5× 前向估算——**此系数为通用经验值，是本项最大不确定来源**）。
  - 通信 **63%** = size-weighted 实测 19.03 GB/s ÷ HCCS 理论 30 GB/s。
  - 调度 **50%** = 4 卡均值 `(1 − (Free + 通信等待) / step)`；通信等待 = 暴露通信 − 按实测带宽折算的真实传输（rank0：420.2 − 92.9 = 327.3 ms）。**未把被 CANN 标为 `Idle` 的 SDMA 真实传输时间计入等待**，否则会把正常传输误判成调度问题。
  - 内存 **7%** = HBM 实测均值（读 61.2 + 写 53.5 = 114.8 GB/s，`HBM` 表按 hbmId 求和后取时间均值）÷ 910B HBM 峰值 1.6 TB/s。**这是带宽利用率，低是 device 大量空闲的结果而非独立问题**；瞬时峰值达 451.6 GB/s（28%）。
  - 显存容量利用率（独立诊断项，不计入 PHS）：峰值 Reserved 11.93 / 12.68 GB ÷ 单卡容量 → 32 GB 机型 ≈ 40%，64 GB 机型 ≈ 20%（`NPU_INFO` 仅给 `Ascend910B`，**子型号与容量待确认**）。
  - 优化后预估 **45 (B)** 的假设：P0/P1 全部落地后 计算 20% · 通信 65% · 调度 78% · 内存 20% → 0.4×20 + 0.3×65 + 0.2×78 + 0.1×20 = 45.1。其中“计算 20%”按“同 token 量单步降到 ~400 ms + micro-batch-size 提到 2~4 摊薄固定开销”估算，**属保守外推，未经实测验证**。
- **低置信度清单**：
  1. FlashAttention 反向 FLOPs 的 2.5× 系数为经验值 → 影响 MFU 绝对值约 ±15%（不影响“MFU 为个位数百分比”的结论）。
  2. 910B 子型号未知 → 峰值算力在 270~378.88 TFLOPS 之间、HBM 容量 32 或 64 GB 之间，相关百分比按区间给出。
  3. 只有 1 个 step → **无法给 step 抖动 / CV**，也无法排除本 step 是非典型 step（但 4 卡一致性极好、且 `skip_first=12` 已跳过预热，可信度较高）。
  4. rank1 / rank3 缺 `kernel_details.csv` → 其算子级结论由同 stage 的 rank0 / rank2 推广而来（两者 step 与通信量高度一致，风险低）。
  5. AI Core 频率 1650 MHz 仅 2 个采样点，不足以判定是否 throttling。
  6. 融合交叉熵、不均匀 PP 切分的具体开关名随 Megatron / MindSpeed 版本变化，报告中给的是候选名，**落地前须对照所用版本的 `arguments.py` 核对**。
- **输出位置**：`./Analysis Report/level2_profiling_analysis_20260921/`（本次分析全部产物）
- **图表侧车**：`chart-spec.json` → `chart-data.json`（由 `node MindStudioNext/gen-report-charts.mjs "Analysis Report/level2_profiling_analysis_20260921"` 生成，数据源为 `evidence/` 下的 rank DB）。本次可渲染 2 组：action 3 = 未融合 CE 算子族表（`op-table`），action 5 = blockDim 饱和度表（`blockdim`）。**通信带宽（`hccs-bw` / `rdma`）与空闲泳道（`swimlane`）这 3 类图表依赖 `cluster_analysis.db` / `free_analysis.db`，因上述 advisor / msprof-analyze 不可用而无法生成，故未在 spec 中声明**（不是留空，是无数据源）。
- **举证文件清单**（已复制至 `evidence/`，报告自包含）：

  | 副本路径 | 原始来源 | 引用的问题点 | 大小 |
  |---|---|---|---|
  | `evidence/rank_0_ascend_pt/trace_view.json` | `D:\...\rank_0_ascend_pt\ASCEND_PROFILER_OUTPUT\trace_view.json` | 3.1 / 3.3 / 3.4 / 3.6 / 3.8 | 15.2 MB |
  | `evidence/rank_0_ascend_pt/kernel_details.csv` | 同上目录 `kernel_details.csv` | 3.3 / 3.5 / 3.8 | 808 KB |
  | `evidence/rank_0_ascend_pt/communication.json` | 同上目录 `communication.json` | 3.1 / 3.4 / 3.9 | 17 KB |
  | `evidence/rank_0_ascend_pt/communication_matrix.json` | 同上目录 `communication_matrix.json` | 3.4 / 3.9 | 12 KB |
  | `evidence/rank_0_ascend_pt/ascend_pytorch_profiler_0.db` | 同上目录 `ascend_pytorch_profiler_0.db` | 3.1 / 3.4 / 3.7 / 3.8（图表 action 5） | 6.7 MB |
  | `evidence/rank_0_ascend_pt/memory_record.csv` | 同上目录 `memory_record.csv` | 3.5 | 1.4 MB |
  | `evidence/rank_0_ascend_pt/operator_memory.csv` | 同上目录 `operator_memory.csv` | 3.2 | 598 KB |
  | `evidence/rank_0_ascend_pt/npu_module_mem.csv` | 同上目录 `npu_module_mem.csv` | 3.5 | 141 KB |
  | `evidence/rank_0_ascend_pt/{op_statistic.csv, api_statistic.csv, step_trace_time.csv, analysis.db, profiler_info_0.json, profiler_metadata.json}` | 同上目录同名文件 | 3.1 / 3.2 / 3.3 / 3.5 | 2.2 KB / 6 KB / 280 B / 28 KB / 1.6 KB / 1 KB |
  | `evidence/rank_1_ascend_pt/ascend_pytorch_profiler_1.db` | `D:\...\rank_1_ascend_pt\ASCEND_PROFILER_OUTPUT\ascend_pytorch_profiler_1.db` | 3.1 / 3.3 | 6.7 MB |
  | `evidence/rank_1_ascend_pt/{communication.json, communication_matrix.json, step_trace_time.csv, analysis.db, profiler_info_1.json, profiler_metadata.json}` | 同上目录同名文件 | 3.1 / 3.3 / 3.9 | 17 KB / 13 KB / 279 B / 32 KB / 1.6 KB / 1 KB |
  | `evidence/rank_2_ascend_pt/trace_view.json` | `D:\...\rank_2_ascend_pt\ASCEND_PROFILER_OUTPUT\trace_view.json` | 3.1 / 3.3 / 3.6 / 3.7 | 15.9 MB |
  | `evidence/rank_2_ascend_pt/kernel_details.csv` | 同上目录 `kernel_details.csv` | 3.2 / 3.3 | 845 KB |
  | `evidence/rank_2_ascend_pt/communication.json` | 同上目录 `communication.json` | 3.1 / 3.6 / 3.9 | 25 KB |
  | `evidence/rank_2_ascend_pt/communication_matrix.json` | 同上目录 `communication_matrix.json` | 3.6 / 3.9 | 16 KB |
  | `evidence/rank_2_ascend_pt/ascend_pytorch_profiler_2.db` | 同上目录 `ascend_pytorch_profiler_2.db` | 3.2 / 3.7（图表 action 3） | 7.1 MB |
  | `evidence/rank_2_ascend_pt/memory_record.csv` | 同上目录 `memory_record.csv` | 3.5 | 1.4 MB |
  | `evidence/rank_2_ascend_pt/operator_memory.csv` | 同上目录 `operator_memory.csv` | 3.2 | 620 KB |
  | `evidence/rank_2_ascend_pt/{npu_module_mem.csv, op_statistic.csv, api_statistic.csv, step_trace_time.csv, analysis.db, profiler_info_2.json, profiler_metadata.json}` | 同上目录同名文件 | 3.2 / 3.3 / 3.5 | 140 KB / 2.9 KB / 7 KB / 278 B / 32 KB / 1.6 KB / 1.1 KB |
  | `evidence/rank_3_ascend_pt/ascend_pytorch_profiler_3.db` | `D:\...\rank_3_ascend_pt\ASCEND_PROFILER_OUTPUT\ascend_pytorch_profiler_3.db` | 3.1 / 3.3 | 7.1 MB |
  | `evidence/rank_3_ascend_pt/{communication.json, communication_matrix.json, step_trace_time.csv, analysis.db, profiler_info_3.json, profiler_metadata.json}` | 同上目录同名文件 | 3.1 / 3.3 / 3.9 | 26 KB / 17 KB / 282 B / 40 KB / 1.6 KB / 1.1 KB |

  > `intermediate/` 下另有本次全部分析脚本与文本结果：`timeline_geometry.{mjs,txt}`（泳道几何 / 空挡）、`comm_sequence.{mjs,txt}`（通信时序）、`mfu_from_shapes.mjs` + `mfu.txt`（MFU）、`kernel_efficiency.{mjs,txt}`（kernel 效率）、`op_shapes.{mjs,txt}`（形状反推模型几何）、`vocab_ops.{mjs,txt}`（vocab 算子明细）、`host_sync.{mjs,txt}`（host 同步点）、`db_query.mjs`（通用 SQL 执行器）。

### 时序结构指标看板（timeline-swimlane-analyzer）

> 由 timeline-swimlane-analyzer 从 device 泳道几何派生（`intermediate/timeline_geometry.txt`）；仅列已测得的指标，未采集的不列。前端“总览—指标看板”消费本表。

| 指标 | 值 | 状态 | 说明 |
|---|---|---|---|
| 关键路径占比 | 82% | warn | 4 卡均值；step 内 18% 为碎片空挡 |
| 算子利用率 | 39% | warn | 计算 busy / step，4 卡均值（首级 28.5%，末级 49.4%） |
| 计算-通信重叠率 | 0% | error | `Overlapped = 0.0`，通信 100% 暴露 |
| Host 下发间隙占比 | 18% | warn | 926 段 50 µs–1 ms 碎片空挡共 135 ms |
| PP 流水线 bubble 率 | 39% | error | 首级 rank0/1 在 P2P 上纯空等 313.9 / 323.0 ms；末级 9% |
| 暴露通信 | 350 ms | error | 4 卡均值（rank0 420.2 / rank1 441.7 / rank2 277.6 / rank3 261.5 ms） |
