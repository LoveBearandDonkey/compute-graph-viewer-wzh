# verl RL 训练性能诊断报告（Ascend 910B · rank 0）

## 1. 结论速览

- **性能健康度**：25 / 100 (D) → 优化后预估 **45 / 100 (B)** — 计算 12% · 通信 33% · 调度 39% · 内存 N/A
  - （订正：通信子项由初版 57% 下调至 33%——初版误用了「逐 op 平均带宽」17 GB/s，按字节加权的有效带宽实为 ~10 GB/s，详见 3.7 链路分析与第 5 章）
- **结论**：本次采集的 54.2 s 窗口里，**Rollout 生成阶段独占 48.0 s（88.5%）**，而生成阶段 NPU 只忙 37%、其余 63% 在等 Host 下发——整段被「Eager 模式逐 token 解码」的下发与通信暴露彻底拖慢；真正的训练（update_actor）只占 3.5 s 且健康（设备占用 75%）。
- **头号瓶颈**：生成阶段 Eager 解码，48 s 内下发 **136 万次 CANN API**（≈2.8 万次/s），设备占用仅 **37%**，AI Core MAC 利用率均值仅 **3.8%**。
- **次号瓶颈**：TP All-Reduce 在解码中**完全暴露**——27,687 次 All-Reduce 累计 12.95 s，其中 **9.4 s 是 wait（等其他卡/同步）**，几乎不与计算重叠（全程仅 0.32 s overlap）。
- **收益上限**：生成阶段转图模式 + 通信重叠 + 内存复用落地后，墙钟有望从 54.2 s 压到 ~34–38 s，**最多省 ~30%–37%（约 16–20 s）**。

---

## 2. 行动清单

> 默认按"预期收益"从高到低排序；同档收益按"修改难度"从低到高排序。

| # | 优先级 | 问题 | 预期收益 | 修改难度 | 在哪改 / 对应位置 | 问题举证视图 |
|---|---|---|---|---|---|---|
| 1 | P0 | Rollout 生成 Eager 解码，48 s 内 136 万次下发，设备占用仅 37% | -20%~-35% 墙钟 | 中 | 推理后端开图模式：vllm-ascend `enforce_eager=False` / ACL Graph / torchair 捕获 | Timeline 视图（系统调优）— `ascend_pytorch_profiler_0.db` |
| 2 | P0 | TP All-Reduce 解码中全暴露，27687 次累计 12.95 s（wait 9.4 s），overlap 仅 0.32 s | -10%~-15% 墙钟 | 中 | rollout TP 配置 / 通信-计算重叠（图模式内）/ 评估 rollout TP=1 | 通信视图 + Timeline 视图（系统调优）— `analysis.db` |
| 3 | P1 | `aclrtFreePhysical`/`MallocPhysical` 共 184 次累计 ~2.5 s，rollout↔train 反复申请释放物理内存 | -3%~-5% 墙钟 | 低 | `PYTORCH_NPU_ALLOC_CONF` 调大缓存 / 复用 KV、关闭 offload 抖动 | Timeline 视图（系统调优）— `ascend_pytorch_profiler_0.db` |
| 4 | P1 | 采样开销大：`DSARandomUniform` 565 次 688 ms（占设备算力 10.6%）+ `ArgMaxV2` 76 ms | -2%~-4% 墙钟 | 中 | 采样路径：批量化采样 / 可贪心处温度=0 走 greedy | 算子视图 — `ascend_pytorch_profiler_0.db` |
| 5 | P1 | 解码算术强度极低：AI Core MAC 占比 3.8%、Vector 占比 4.9%，受内存/launch 限制 | 提升单 kernel MFU | 高 | 增大 rollout 有效 batch / continuous batching，把碎 GEMM 喂成大 GEMM | 算子视图 + 详情视图 — `ascend_pytorch_profiler_0.db` |
| 6 | P1 | HCCS 小包通信：有效带宽仅 ~10 GB/s（≈理论 30 GB/s 的 33%），逐 op 低至 5.9 GB/s | 提升带宽利用 | 中 | 增大通信 bucket / 字节对齐 / 减少切分粒度（FSDP `reduce_dtype`、bucket_cap） | 通信视图 — `analysis.db`（`CommAnalyzerMatrix`） |
| 7 | P2 | Host 侧 `aten::copy_`/`aten::to`/`_to_copy` 累计 ~11.5 s，dtype/device 转换冗余 | -2% 墙钟 | 中 | 排查 rollout↔train 张量搬运与 dtype 转换，定长复用 | Timeline 视图（系统调优）— `ascend_pytorch_profiler_0.db` |

---

## 3. 问题详情

### 3.1 [P0] Rollout 生成 Eager 解码，设备占用仅 37%

- **证据**：
  - MSTX 相位标记还原时间轴：`compute_log_prob`(1.08 s) 起于采集第 ~48 s、`update_actor`(3.54 s) 起于第 ~50.7 s——**前 48.0 s 无任何训练标记，即 Rollout 生成阶段，占墙钟 88.5%**。
  - 分相位统计（`phase_split.cjs`）：生成阶段 48,009 ms 内**设备并集忙时仅 17,748 ms = 37.0%**，其中真实计算只有 3,597 ms；同期 `CANN_API` 调用 **1,356,400 次**（≈2.8 万次/s）。
  - 设备算子明细全是 vLLM 解码核：`PagedAttentionMaskNdKernel` 13,464 次、`ReshapeAndCacheNdKernel` 13,560 次、`AtbRopeKernel` 13,560 次——典型 paged-attention 自回归解码。
  - Host 大头 `launch` 287,326 次 11,762 ms、`vllm::unified_ascend_attention_with_output` 13,560 次 5,206 ms——逐算子下发喂不饱设备。
- **影响**：生成阶段 63% 时间设备空等 Host，而生成占整窗 88.5%，是端到端吞吐的根本制约；RL 单次迭代的 rollout 墙钟被 Host 串行下发卡死。
- **修复建议**：
  1. 给 rollout 推理后端开**图模式**：vllm-ascend 关闭 `enforce_eager`、启用 ACL Graph 捕获（或 torchair 图模式），把每 token ~数千次 Host 下发收敛为整图重放，让 device 背靠背执行。
  2. 确认 `ASCEND_LAUNCH_BLOCKING` 未置 1（`META_DATA.ENV_VARIABLES` 中为空，正常），保证下发队列可加深。
  3. 配合 3.2（先消通信暴露）再做图捕获，避免图被同步打断。
- **验证方法**：重采 profiling，确认生成阶段设备并集占用率从 37% 升到 ≥60%，`CANN_API` 调用次数显著下降，墙钟回落。
- **问题举证视图**：Timeline 视图（系统调优）— 载入 `ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_0.db`（本次为 DB 导出，无 `trace_view.json`），对比 Host `launch` 行与 Ascend Hardware device task 行之间的大段空隙（Free），确认设备在等下发。

### 3.2 [P0] TP All-Reduce 在解码中完全暴露，9.4 s 花在 wait

- **证据**：`analysis.db` 通信分析——`hcom_allReduce` **27,687 次累计 12,954 ms，其中 wait_time 9,355 ms（72%）、transit 几乎为 0**；`hcom_allGather` 1,175 次 1,662 ms、`hcom_reduceScatter` 100 次 622 ms。`StepTraceTime`：communication 15,238 ms 中 **comm_not_overlapped 14,918 ms（98%），仅 320 ms 与计算重叠**。带宽侧 HCCS 实测均值 **17.1 GB/s（约理论 30 GB/s 的 57%）**，包小（reduceScatter 仅 4.6 GB/s），属延迟/同步受限而非带宽受限。
- **影响**：解码 batch 小、逐 token 串行，TP=2 的 per-layer All-Reduce 无法被计算掩盖；27,687 次小集合通信的同步等待累计近 9.4 s，是生成阶段设备空转的第二大来源。
- **修复建议**：
  1. 图模式内开启**通信-计算重叠**（comm stream 与 compute stream 并行），把 0.32 s 的 overlap 拉高。
  2. 评估 rollout 阶段降低张量并行：若单卡显存放得下权重+KV，rollout 用 **TP=1** 可直接消除这批 All-Reduce。
  3. 减少集合通信次数：算子/层融合后合并相邻 All-Reduce，提升单次消息体量（当前包过小，带宽利用仅 57%）。
- **验证方法**：重采后 `CommAnalyzerTime` 中 All-Reduce 次数与 wait_time 下降，`StepTraceTime.overlapped` 占 communication 比例从 2% 升到 >30%。
- **问题举证视图**：
  - 主：通信视图 — 载入 `ASCEND_PROFILER_OUTPUT/analysis.db`（`CommAnalyzerMatrix`/`CommAnalyzerBandwidth`），看 HCCS 链路带宽与小包占比。
  - 辅：Timeline 视图（系统调优）— `ascend_pytorch_profiler_0.db`，过滤 `hcom_allReduce`，观察其后 device 计算行是否串行等待（无重叠）。

### 3.3 [P1] 物理内存反复申请/释放占 ~2.5 s

- **证据**：`CANN_API` 中 `aclrtFreePhysical` **92 次 2,019 ms（均 22 ms/次）**、`aclrtMallocPhysical` 92 次 446 ms——合计 ~2.5 s（占墙钟 4.6%）花在物理内存页申请/释放，远高于普通 `aclrtMalloc`/`Free`。
- **影响**：rollout 与 train 共卡（colocate）时反复重建显存/KV 物理映射，单次释放 22 ms 量级，叠加成秒级纯开销且打断流水。
- **修复建议**：
  1. 调大 `PYTORCH_NPU_ALLOC_CONF`（增大缓存段、减少归还），让 allocator 复用而非反复 `FreePhysical`。
  2. 排查 rollout↔train 的权重/KV offload-reload 策略，能常驻则常驻，减少物理内存抖动。
- **验证方法**：重采后 `aclrtFreePhysical`/`MallocPhysical` 次数与总耗时明显下降。
- **问题举证视图**：Timeline 视图（系统调优）— 载入 `ascend_pytorch_profiler_0.db`，过滤 `aclrtFreePhysical`/`aclrtMallocPhysical`，定位其在相位切换处的密集出现。

### 3.4 [P1] 采样开销大：DSARandomUniform 占设备算力 10.6%

- **证据**：设备算子统计 `DSARandomUniform` **565 次累计 688 ms（均 1,218 µs/次），占全部计算设备时间 10.6%**，是仅次于 MatMulV2 的第二大计算开销；配套 `ArgMaxV2` 565 次 76 ms。
- **影响**：解码采样（随机数生成 + argmax）单次毫秒级，且每个生成步都要走一遍，显著抬高生成阶段的有效计算占比中的"非模型"部分。
- **修复建议**：
  1. 批量化采样：把逐步采样合并为按 batch 一次性生成随机数。
  2. 对温度=0 / 贪心解码路径直接走 greedy（argmax），跳过随机数生成。
  3. 检查是否可用更轻量的采样核实现。
- **验证方法**：重采后 `DSARandomUniform` 调用次数/总耗时下降，其占计算设备时间比例回落。
- **问题举证视图**：算子视图 — 载入 `ascend_pytorch_profiler_0.db`，按 opType 聚合查看 `DSARandomUniform`/`ArgMaxV2` 的调用次数与单次耗时。

### 3.5 [P1] 解码算术强度极低，单 kernel MFU 很低

- **证据**：`TASK_PMU_INFO`（`ACL_AICORE_PIPE_UTILIZATION`）均值——**AI Core MAC 占比 `aic_mac_ratio` 仅 3.8%**、Vector 占比 `aiv_vec_ratio` 仅 4.9%，而搬运 `aic_mte2_ratio` 14.7%、标量 `aic_scalar_ratio` 13.3% / `aiv_scalar_ratio` 23.7%——计算单元被搬运与标量主导。`MatMulV2` 57,881 次均 28 µs（小 GEMM），blockDim 22–24（核数已基本铺满，瓶颈不在多核切分而在 shape 太小）。
- **影响**：解码 batch 小、算术强度低，cube/vector 几乎空转；即便解决下发与通信，碎小算子仍限制有效算力，MFU 远低于 910B 能力。
- **修复建议**：
  1. 增大 rollout 有效 batch（continuous batching / 提高并发请求数），把 28 µs 的小 matmul 喂成大 GEMM。
  2. 算子融合（attention 已用 paged 融合核，可进一步融合 RoPE/Cast/Gather 等碎核）。
- **验证方法**：重采后 `aic_mac_ratio` 上升，MatMulV2 平均耗时上升而单位 token 总耗时下降。
- **问题举证视图**：
  - 主：算子视图 — `ascend_pytorch_profiler_0.db`，看 MatMulV2/PagedAttention 的耗时与次数分布。
  - 辅：详情视图 — Roofline 定性（需算子 `*.bin`，本次 DB 导出未含；可在重采时加 `_export_type` 输出 text/bin 以查看 Roofline 落点）。

### 3.6 [P1] HCCS 链路均衡（无慢链路），但小包导致带宽利用仅 ~33%

> 本节按 `cluster-fast-slow-rank-detector` 与 `msprof-analyze-cli` 的「慢链路定位」专家规则，对 rank 0 的 `CommAnalyzerMatrix` 做逐链路（src→dst）带宽对比。msprof-analyze CLI 在本环境不可用（见第 5 章），故用 `node:sqlite` 复刻其 `communication_matrix_sum` / `slow_link` 逻辑。

- **拓扑证据**：rank 0 到 peer 1–7 全部经 **HCCS**（+ 自身 LOCAL），**无 RDMA** → 8 卡同处单节点（单机 8 卡），不存在跨节点慢链路风险。
- **链路均衡（无慢链路）**：DP 组（default_group [0–7]）的 6 条链路 0→2…0→7 有效带宽 **7.63 / 7.63 / 7.63 / 7.63 / 7.63 / 7.67 GB/s**，极差 < 1%，高度一致；0→1 较高（**14.59 GB/s**）是因 [0,1] TP 组承载 ~5× 字节量（14,384 vs 2,869），属流量差异而非故障。**结论：无故障/慢链路。**
- **真问题是小包**：HCCS 按字节加权有效带宽 **≈10 GB/s（31.26 GB ÷ 3.12 s），仅理论 ~30 GB/s 的 33%**；逐 op 平均带宽低至 5.9 GB/s、大量 op 趋近 0——典型**小包/延迟受限**（与 3.2 的 27,687 次微型 All-Reduce 互证），而非链路带宽硬件瓶颈。
- **快慢卡判定（证据不足，需补卡）**：仅有 rank 0 数据，无法点名慢卡 Rank ID。rank 0 在 All-Reduce 上的 **wait_time 高达 9.4 s**——按专家规则，高 wait 既可能是「通信暴露/串行」（最可能，因解码逐 token），也可能是「其他卡为 straggler 致 rank 0 早到空等」。**区分二者必须补采 ≥1–2 张其他卡**：若各卡 wait 对称偏大 → 暴露/序列化问题；若某卡 wait 显著小而 compute 大 → 该卡为计算型慢卡。
- **修复建议**：
  1. 增大通信 bucket、提升单次消息体量（小包是带宽利用低的根因，与 3.2 #2 同向）。
  2. 检查 FSDP `reduce_dtype` / bucket 大小与字节对齐，减少 ZeRO 切分过细导致的碎包。
  3. **补采全部 8 卡的 Level1 DB**，跑 `msprof-analyze -m slow_rank/slow_link/cluster_time_summary` 做真正的快慢卡定位。
- **验证方法**：补采多卡后 `slow_rank` 无明显慢卡候选、各 Rank wait 对称；增大 bucket 后 `CommAnalyzerMatrix` 有效带宽从 ~10 GB/s 升到 >18 GB/s。
- **问题举证视图**：通信视图 — 载入 `ASCEND_PROFILER_OUTPUT/analysis.db`（`CommAnalyzerMatrix`/`CommAnalyzerBandwidth`），看 0→1…0→7 各 HCCS 链路带宽热力是否均衡、小包占比是否偏高。

### 3.7 [P2] Host 侧 dtype/device 转换冗余约 11.5 s

- **证据**：`PYTORCH_API` 中 `aten::copy_` 92,177 次 6,030 ms、`aten::to` 8,507 次 2,793 ms、`aten::_to_copy` 4,016 次 2,699 ms——合计 ~11.5 s Host 时间在张量拷贝与类型/设备转换（注：Host 时间含异步下发，不直接等于墙钟，但反映冗余度）。
- **影响**：rollout 与 train 之间的数据搬运、精度转换重复发生，加重 Host 负载、放大下发气泡。
- **修复建议**：排查 rollout 产出→train 输入的搬运链路，定长缓冲复用、减少 `.to()`/`.cpu()` 与 dtype 反复转换。
- **验证方法**：重采后 `aten::copy_`/`aten::to`/`_to_copy` 次数与总耗时下降。
- **问题举证视图**：Timeline 视图（系统调优）— 载入 `ascend_pytorch_profiler_0.db`，过滤 `aten::copy_`/`aten::to`，观察其在相位边界的密集搬运。

---

## 4. 已确认无问题 / 未排查

- **训练阶段（update_actor）健康**：3.54 s 内设备占用 **75.4%**、计算 1,935 ms、通信仅 971 ms（FSDP 反向 + 优化器），**不是瓶颈**，无需优先优化。
- **频率无降频**：`AICORE_FREQ` 全程在 800–1800 MHz 区间，主体 1800 MHz，无异常降频。
- **多核切分充分**：主要算子 blockDim 22–48（910B 核数已基本铺满），瓶颈在 shape/下发而非多核未铺满。
- **慢链路（已排查，无问题）**：基于 rank 0 的 `CommAnalyzerMatrix` 逐链路对比——单节点 8 卡全 HCCS 互联（无 RDMA），DP 组 6 条链路有效带宽 7.63–7.67 GB/s 极差 < 1%，**无故障/慢链路**（详见 3.7）。链路低带宽（~33% 理论值）已归因为小包而非链路故障。
- **快慢卡（未排查，需补卡）**：仅 rank 0 一张卡，无法点名慢卡 Rank ID；rank 0 的 9.4 s All-Reduce wait 需 **补采 ≥1–2 张其他卡**才能区分「通信暴露」与「其他卡 straggler」（判别方法见 3.7）。
- **内存带宽（未排查）**：采集时 `profile_memory=false`、`_l2_cache=false`，无 HBM/L2 带宽数据，PHS 内存项记 N/A。

---

## 5. 数据与方法

- **分析日期**：2026-06-02
- **数据路径**：`verl/1/e2e/localhost.localdomain_214483_20260116064439460_ascend_pt/`
- **数据范围**：单卡（rank 0 / device 0，Ascend 910B），采集墙钟 **54,234 ms**，整窗按 MSTX 标记分相位：Rollout 生成 ~48.0 s（88.5%）、compute_log_prob 1.08 s、相位间隙 1.57 s、update_actor 3.54 s。作业拓扑（`META_DATA.parallel_group_info`）：default_group = 8 卡 [0–7]，group_17 = [0,1]（疑似 rollout TP=2）。
- **采集配置**（`profiler_info_0.json`）：`profiler_level=Level1`、`aic_metrics=ACL_AICORE_PIPE_UTILIZATION`、`_export_type=[db]`、`_msprof_tx=true`；`record_shapes=false`、`profile_memory=false`、`with_stack=false`、`with_flops=false`。torch_npu 2.7.1 / CANN 8.3.RC1。
  - **配置局限**：未采 shape/flops → 无法精确算 MFU（已用 `aic_mac_ratio` 3.8% 与设备占用率替代定性）；未采 memory → 内存子项 N/A；未采 stack → 无法回溯 Python 源码行；DB-only 导出 → 无 `trace_view.json`/算子 `*.bin`，Insight 需直接载入 `*_ascend_pt` 目录或 `.db`。
- **使用的 Skills**：
  - `mindstudio_profiler_data_check`：校验数据完整性（框架 PyTorch profiler / 单卡 / valid，采集已正常 Stop、已解析）
  - `ascend_pytorch_profiler_db_explorer`：用 Track A compute 宏 + DB 查询做算子/通信/下发分析（经 `node:sqlite` 执行，脚本见 `intermediate/`）
  - `cluster-fast-slow-rank-detector` / `msprof-analyze-cli`：按「慢链路定位」专家规则，对 rank 0 的 `CommAnalyzerMatrix` 做逐链路（src→dst）带宽对比（复刻 `communication_matrix_sum`/`slow_link` 逻辑），判定无慢链路；快慢卡因仅 1 卡数据暂无法点名（见 3.7、第 4 章）
  - `performance-health-score`：按"大模型多卡训练"权重（计算 0.40 / 通信 0.30 / 调度 0.20 / 内存 0.10）计算 PHS，内存 N/A 后归一化（计算 0.444 / 通信 0.333 / 调度 0.222）
  - `op-mfu-calculator`：评估计算受限性（MAC 占比 3.8%，确认解码非计算瓶颈而是内存/下发受限）
  - `msinsight-view-selector`：为每个问题点附 MindStudio Insight 视图
- **Advisor 状态**：
  - `msprof-analyze advisor`：**失败** — 本环境 Python 为 Microsoft Store 占位 stub（执行 exit 49）、无 `sqlite3` CLI 与 `msprof-analyze` 命令；改用 Node.js v24 内置 `node:sqlite` 直接查询 `ascend_pytorch_profiler_0.db` 与 `analysis.db`，并以 `StepTraceTime`/`CommAnalyzer*`/MSTX 相位标记交叉验证，结论已并入第 2、3 章。
- **PHS 计算说明**：
  - 计算利用率 ≈ AI Core 占用率 = 计算设备时间 6,472 ms ÷ 墙钟 54,234 ms = **11.9%**（真实 MFU 更低，MAC 占比仅 3.8%）。
  - 通信效率 = HCCS 按字节加权有效带宽 **~10 GB/s（31.26 GB ÷ 3.12 s）÷ 理论 30 GB/s = 33%**（订正：初版误用逐 op 平均带宽 17.1 GB/s，实际有效带宽由 3.7 链路分析得 ~10 GB/s；通信另受同步/暴露拖累，本子项仅反映带宽维度）。
  - 调度效率 = (1 − Free 占比 60.6%) × 100 = **39.4%**。
  - 内存、（集群均衡度）N/A。
  - 当前 PHS = 0.444×12 + 0.333×33 + 0.222×39 ≈ **25（D）**。优化后按 P0/P1 落地（生成设备占用 37%→~65%、增大消息体后带宽利用 33%→~60%、调度→~65%）代回公式 ≈ **45（B）**。
- **输出位置**：`./ascend_analysis_verl_20260602/`（`report.md` + `intermediate/` 下的 `node:sqlite` 查询脚本：overview / steptrace_comm / comm_detail / mstx_compute / phase_split / pmu_size / slowlink）
