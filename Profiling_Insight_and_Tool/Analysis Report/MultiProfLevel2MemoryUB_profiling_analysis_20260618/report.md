# MultiProfLevel2MemoryUB 16 卡多机训练性能诊断报告

> 数据：`MultiProfLevel2MemoryUB_db/`（2 节点 × 8 卡 = 16 Rank，Ascend910B，Megatron-LM TP/DP，PyTorch Profiler Level2 DB，单 active step）
> 分析日期：2026-06-18 · 报告目录：`MultiProfLevel2MemoryUB_profiling_analysis_20260618/`

---

## 1. 结论速览

- **性能健康度**：64 / 100 (B+) → 优化后预估 **77 / 100 (A)** — 计算 60% · 通信 57% · 调度 84% · 内存 N/A
- **结论**：单步耗时 ~6.33 s，瓶颈是「Host 下发被在线编译/AICPU 拖慢」叠加「TP 通信重叠不足」——计算单元被 host 喂不饱而周期性饿死，整集群在 collective barrier 上互等。
- **头号瓶颈**：Rank 15 设备空闲（free）40.7%（2.57 s/步）、`slowAffectCount=70`（全集群最高阻塞源）；根因是 8390 次在线算子编译（`aclopCompileAndExecute`，352 ms）+ 占 40.4% 的 AICPU 融合算子（`AllGatherMatmulAicpu`/`MatmulReduceScatterAicpu`）。
- **收益上限**：行动清单 P0/P1 全部落地后预估节省 **~12–18% 单步耗时（约 0.8–1.1 s）**。

---

## 2. 行动清单

> 默认按"预期收益"从高到低排序；同档收益按"修改难度"从低到高排序。

| # | 优先级 | 问题 | 预期收益 | 修改难度 |
|---|---|---|---|---|
| 1 | **P0** | 8390 次在线算子编译（aclopCompileAndExecute）拖慢 Host 下发 | -5~8% 单步耗时 | 低 |
| 2 | **P0** | TP 融合算子走 AICPU 路径（占 40.4% 计算时间） | -8~10% 计算 | 中 |
| 3 | **P1** | Rank 15 等节点2多卡 Host 饿死型空闲（free 40.7%/20%，slowAffectCount=70） | -均衡，-5~8% | 中 |
| 4 | **P1** | HCCS/SDMA 节点内带宽仅 ~54% 理论值 + 计算通信带宽抢占 | -3~5% 通信 | 中 |
| 5 | **P1** | 跨节点 RDMA 小包（82% 包 <50% 大包阈值）+ 通信重传 | -通信尾延迟 | 中 |
| 6 | P2 | Block Dim 未饱和（46% 时间占比算子核数不足） | -2~4% 计算 | 中 |
| 7 | P2 | 可用亲和 API 未启用（NpuFusedAdamW / npu_confusion_transpose） | -小幅下发 | 低 |
| 8 | P2 | 跨节点 DP AllReduce 早期迭代巨额互等（node1 等 node2 最高 777 ms） | 消除尖峰 | 中 |

> 改动位置、举证视图均在第 3 章；行动清单不重复。第 3 章小节标题与本表"问题"列一字不差。

---

## 3. 问题详情

### 3.1 [P0] 8390 次在线算子编译（aclopCompileAndExecute）拖慢 Host 下发

- **证据**：advisor（Rank 15）Operator Dispatch Issues：`aclopCompileAndExecute` Counts=**8390**，Elapsed Time=**352379.97 us（352 ms）**。`free_analysis` 配套佐证：多卡出现 "Abnormal CANN layer: long time between two node@launch" 间隙——Rank 11 累计 21.7 ms、Rank 15 20.8 ms、Rank 8 11.5 ms（top-160 free 事件，单条 launch 间隙最高 5.2 ms），以及 "Idle Pytorch layer: no task dispatched" 事件。这些都是 Host 下发跟不上、NPU 饿死的直接特征。
- **影响**：在线编译串行阻塞下发线程 → device 周期性空泡 → 落到 collective barrier 上放大为全集群互等。是 Rank 15 free 40.7% 与 node2 整体 free 18.3%（vs node1 9.1%）的首要驱动。
- **修复建议**：
  - **改动位置**：训练启动脚本/入口（torch_npu 初始化处，所有 rank）
  1. 关闭 JIT 在线编译、禁用内部格式自动转换：
     ```python
     torch_npu.npu.set_compile_mode(jit_compile=False)
     torch_npu.npu.config.allow_internal_format = False
     ```
  2. 确认无动态 shape 触发反复编译（固定 seq_len / pad 到定长）。
- **问题修改完成的验证方式**：重采 profiling，确认 `aclopCompileAndExecute` 次数从 8390 降至接近 0；`free_analysis` 中 "long time between two node@launch" 间隙消失或 <2 ms；Rank 15 free 占比从 40.7% 降至 <15%。
- **问题举证视图**：Timeline 视图（系统调优）— 载入 `evidence/rank15_ascend_pt/ascend_pytorch_profiler_15.db`（源：`d:\Projects\ProfilingTest\MultiProfLevel2MemoryUB_db\ubuntu2204_1660973_20240619060440181_ascend_pt\ASCEND_PROFILER_OUTPUT\ascend_pytorch_profiler_15.db`），过滤 host 侧 `aclopCompileAndExecute` / `launch`，观察与 device task 之间的 Free 间隙。

### 3.2 [P0] TP 融合算子走 AICPU 路径（占 40.4% 计算时间）

- **证据**：advisor（Rank 15）AICPU Issues（最高严重度，红）：`AllGatherMatmulAicpu`（640 次，1712958 us）、`MatmulReduceScatterAicpu`（多组），AICPU 类算子 Elapsed Time 合计 **2826670 us（2826 ms）、时间占比 0.404**。`compute_op_sum` 同口径：这两类算子是各 rank 计算耗时最大头；同 OpType 同 Count 下，Rank 9 比 Rank 15 慢 2.2×（`MatmulReduceScatter` R9 2280 ms vs R15 1056 ms），说明融合算子内部嵌入了通信等待且实现走低效 AICPU 通路。
- **影响**：AICPU 通路需要 host 介入、且不吃满 cube，是计算利用率上不去（计算子项仅 60%）与下发压力的核心。
- **修复建议**：
  - **改动位置**：TP/序列并行通信-计算融合配置（Megatron `--tp-comm-overlap` / MC2 融合算子开关、torch_npu 融合算子注册）
  1. 确认 `AllGatherMatmul`/`MatmulReduceScatter` 走 MIX_AIC（cube）融合路径而非 `*Aicpu` 变体；检查 CANN/torch_npu 版本是否支持该 shape/dtype 的 cube 融合。
  2. 若特定 shape 回退 AICPU，调整切分（TP size / sequence parallel 粒度）使其命中 cube 融合白名单。
- **问题修改完成的验证方式**：重采后 `compute_op_sum` 中 `*Aicpu` 算子 Count→0 或时间占比 <5%；advisor AICPU Issues 消失；计算子项利用率提升。
- **问题举证视图**：算子视图 — 载入 `evidence/rank15_ascend_pt/ascend_pytorch_profiler_15.db`（源同 3.1），按 OpType 聚合查看 `AllGatherMatmulAicpu`/`MatmulReduceScatterAicpu` 耗时占比与 TaskType=AI_CPU 标记。

### 3.3 [P1] Rank 15 等节点2多卡 Host 饿死型空闲（free 40.7%/20%，slowAffectCount=70）

- **证据**：`cluster_time_summary`（单 active step，step≈6.33 s）：Rank 15 computing 仅 52.2%、free **40.7%（2.57 s）**，为全集群最低算占比 / 最高空闲；Rank 8/11/13 free 20–22%；node2 平均 free 18.3% vs node1 9.1%。`slow_rank`：Rank 15 `slowAffectCount=70`（次高 R0/R8=24），是 barrier 最大阻塞源。各 rank 算子 **Count 完全一致**（无负载切分不均），故属"伪快卡/Host 下发型慢卡"：NPU 饿死→空闲→到达 collective 最晚→阻塞全集群。
- **影响**：该症状是 3.1/3.2 的集群级表现，直接决定单步被拉长到 6.33 s。修好 3.1/3.2 后此项自然收敛。
- **修复建议**：
  - **改动位置**：node2 各 rank 的 host 侧（CPU 绑核 / dataloader / 下发线程）
  1. 先落地 3.1（关 JIT）、3.2（去 AICPU）——消除下发阻塞源。
  2. node2 host 侧排查：下发线程绑核（`taskset`/`CPU affinity`）、dataloader worker 是否抢占、Python GC。
  3. 复测 `slowAffectCount` 分布是否趋于均匀。
- **问题修改完成的验证方式**：重采后各 rank free 占比极差 <10%、Rank 15 `slowAffectCount` 回落到与其他 rank 同量级；单步耗时下降。
- **问题举证视图**：
  - 主：Timeline 视图（系统调优）— `evidence/rank15_ascend_pt/ascend_pytorch_profiler_15.db`（源同 3.1），看 device 泳道周期性空泡与 host launch 对齐。
  - 辅：通信视图 — `evidence/cluster/slow_rank.db`（源：`d:\Projects\ProfilingTest\MultiProfLevel2MemoryUB_profiling_analysis_20260618\msprof_analyze\slow_rank\cluster_analysis_output\cluster_analysis.db`），查看 `SlowRank` 表各 rank slowAffectCount。

### 3.4 [P1] HCCS/SDMA 节点内带宽仅 ~54% 理论值 + 计算通信带宽抢占

- **证据**：`ClusterCommunicationBandwidth` 按链路类型聚合（size-weighted）：HCCS 实测 **16.35 GB/s**、SDMA **16.35 GB/s**，均约为节点内理论 ~30 GB/s 的 **54%**；16 卡带宽极差 <5%（16.1–16.8 GB/s，**无慢链路**）。advisor Bandwidth Contention：计算与通信并发时 "SDMA 带宽低于 14.4 GB/s"。（RDMA 24.0 GB/s ≈ 96% 理论值，健康。）
- **影响**：node1 单步通信总量 3.52 s（绝大部分被计算重叠），node2 1.0 s；HCCS 效率不足直接抬高 TP 融合算子内部耗时，是通信子项仅 57% 的主因。
- **修复建议**：
  - **改动位置**：通信算子字节对齐 / 通信-计算 stream 划分（HCCL 配置 + 融合算子 tiling）
  1. 检查 SDMA/HCCS 传输地址与数据块字节对齐（512B/cacheline）。
  2. 让通信与计算尽量独立 stream，降低带宽抢占；评估 `HCCL_INTRA_PCIE_ENABLE`/`HCCL_INTRA_ROCE_ENABLE` 等拓扑配置。
- **问题修改完成的验证方式**：重采后 HCCS/SDMA size-weighted 带宽 ≥ 24 GB/s（>80% 理论），advisor Bandwidth Contention 告警消失。
- **问题举证视图**：通信视图 — 载入 `evidence/cluster/cluster_analysis.db`（源：`d:\Projects\ProfilingTest\MultiProfLevel2MemoryUB_db\cluster_analysis_output\cluster_analysis.db`），查看 `ClusterCommunicationBandwidth` 按 `band_type` 的带宽矩阵。

### 3.5 [P1] 跨节点 RDMA 小包（82% 包 <50% 大包阈值）+ 通信重传

- **证据**：`ClusterCommunicationBandwidth`（band_type=RDMA）：180 条记录中 **82.2% 的 `large_packet_ratio` < 0.5**（均值 0.178），即跨节点传输被小包主导；advisor Packet Analysis "过小的通信数据包可能导致 host 传递瓶颈"，且 Retransmission Analysis 列出重传算子 top10。
- **影响**：小包抬高跨节点通信的 host 开销与尾延迟（虽 RDMA 聚合带宽 24 GB/s 尚可，但小包/重传增加抖动与等待）。
- **修复建议**：
  - **改动位置**：DP 梯度通信粒度 / bucket 聚合（DDP/优化器通信桶大小）+ RDMA 网络配置
  1. 增大梯度通信 bucket（减少小包数量），开启梯度通信聚合。
  2. 排查重传：检查 RoCE/网络 PFC、链路误码、`HCCL_RDMA_*` 超时配置。
- **问题修改完成的验证方式**：重采后 RDMA `large_packet_ratio<0.5` 占比 <30%；advisor Packet/Retransmission 告警消失。
- **问题举证视图**：通信视图 — 载入 `evidence/cluster/cluster_analysis.db`（源同 3.4），过滤 `band_type='RDMA'` 看 `large_packet_ratio` 与 `package_size` 分布。

### 3.6 [P2] Block Dim 未饱和（46% 时间占比算子核数不足）

- **证据**：advisor（Rank 15）Block Dim Issues：部分算子未用满 25 个 AICore / 50 个 AIVector，涉及 `MatmulReduceScatter, Mul, AllGatherMatmul, ZerosLike, MatMul, FlashAttentionScoreGrad, ApplyAdamW, FlashAttentionScore, EmbeddingDenseGrad, GatherV2`，时间占比 0.46。
- **影响**：核间并行不足，单算子吞吐受限，叠加在 3.2 上共同压低计算利用率。
- **修复建议**：
  - **改动位置**：算子 tiling / blockDim 设置（多为框架/CANN 算子，部分需版本升级）
  1. 优先随 3.2 一起解决（cube 融合通常自带更优 blockDim）。
  2. 对自定义算子，按硬件核数设 blockDim（耦合架构用 `GetCoreNumAic/Aiv`）。
- **问题修改完成的验证方式**：重采后 advisor Block Dim Issues 涉及算子时间占比 <0.2。
- **问题举证视图**：算子视图 — 载入 `evidence/rank15_ascend_pt/ascend_pytorch_profiler_15.db`（源同 3.1），查看上述算子的 `blockDim`/`mixBlockDim`（`COMPUTE_TASK_INFO`）。

### 3.7 [P2] 可用亲和 API 未启用（NpuFusedAdamW / npu_confusion_transpose）

- **证据**：advisor（Rank 15）Affinity API Issues：建议启用 `torch_npu.optim.NpuFusedAdamW`、`torch_npu.npu_confusion_transpose`（cann-8.0.0 / torch_npu 环境）。
- **影响**：未用融合优化器/亲和算子，多发若干下发条数与 host 开销（幅度小）。
- **修复建议**：
  - **改动位置**：优化器构造处与对应 transpose 调用点
  1. 用 `torch_npu.optim.NpuFusedAdamW` 替换原 AdamW。
- **问题修改完成的验证方式**：重采后 advisor Affinity API 建议消失，ApplyAdamW 相关下发条数下降。
- **问题举证视图**（本项为 advisor 静态调用栈匹配，无对应 timeline 几何，证据本质为文本）：
  - 主：advisor 报告 — 载入 `evidence/rank15_ascend_pt/mstt_advisor_rank15.html`（源：`d:\Projects\ProfilingTest\MultiProfLevel2MemoryUB_profiling_analysis_20260618\advisor_output\rank15\mstt_advisor_20260618120758.html`），定位「Affinity API Issues」段，看 `NpuFusedAdamW` / `npu_confusion_transpose` 的建议与 code stack。
  - 辅：Timeline 视图（系统调优）— 载入 `evidence/rank15_ascend_pt/ascend_pytorch_profiler_15.db`（源同 3.1），按 `PYTORCH_API` 过滤 `AdamW` / `confusion_transpose` 调用点。

### 3.8 [P2] 跨节点 DP AllReduce 早期迭代巨额互等（node1 等 node2 最高 777 ms）

- **证据**：`hccl_sum` 的 `HcclTopOpStats`：一批 cross-node DP `hcom_allReduce__*`（cnt=2）呈极端双峰——同一算子 min ≈ 36 us、max ≈ **777 ms**，且 max 一律落在 node1（r0–r7）、min 一律落在 node2（r8–r15）对应 DP 对（如 r7↔r15、r0↔r8）。说明早期迭代/初始化阶段 node1 先到、长时间等 node2。该量级远超稳态单步（稳态 `hcom_allReduce__624_0_1` mean 仅 72 us），属 warmup/首迭代尖峰。
- **影响**：不进入稳态单步关键路径，但拉高整段采集墙钟、并提示跨节点同步初始不齐。
- **修复建议**：
  - **改动位置**：跨节点通信初始化 / 首迭代 overlap
  1. 排查首次 DP AllReduce 是否含连接建立/参数广播阻塞；预热通信域。
- **问题修改完成的验证方式**：重采后该批 allReduce 的 max/min 比值收敛（<10×）。
- **问题举证视图**：通信视图 — 载入 `evidence/cluster/hccl_sum.db`（源：`d:\Projects\ProfilingTest\MultiProfLevel2MemoryUB_profiling_analysis_20260618\msprof_analyze\hccl_sum\cluster_analysis_output\cluster_analysis.db`），查询 `HcclTopOpStats` 表，按 `MeanNs*Count` 排序看 `hcom_allReduce__*` 的 `MinNs(MinRank)` / `MaxNs(MaxRank)` 双峰（max≈777 ms 落 node1、min≈36 us 落 node2 对应 DP 对）。辅：`evidence/cluster/cluster_analysis.db`（源同 3.4）`ClusterCommunicationTime` 看跨节点等待。

---

## 4. 已确认无问题

- **数据完整性**：16/16 Rank `profiler_info_*.json`、`analysis.db`、`ascend_pytorch_profiler_*.db` 齐全，均正常 Stop、已解析（DB 模式 37 张表），芯片均 Ascend910B。
- **无慢链路**：HCCS（16.1–16.8 GB/s）、RDMA（24.0 GB/s）各 rank 极差 <5%，无单链路异常；问题是系统性带宽效率（3.4），非某条慢链路。
- **AICore 频率**：16 卡均 1850 MHz（boost 态，非 800 MHz 降频/异常），`freq_analysis` 标记仅因 ≠1800 MHz 的形式判定，非真实降频故障。
- **无硬件慢卡**：HBM 实测平均带宽两节点一致（~20.4 GB/s，峰值 138–186 GB/s），各 rank 算子 Count 一致，排除算力硬件劣化/负载切分不均。
- **单步耗时均衡**：16 卡单步耗时 6.288–6.468 s，跨 rank CV 0.92%，集群在 step 级高度同步（问题在 step 内部时间构成，而非 step 总时长离散）。
- **显存容量**：未确认——见第 5 章数据缺陷说明（`memory_record` 峰值仅 0.31 GB，明显不完整，故不据此判 OOM 风险）。

---

## 5. 数据与方法

- **分析日期**：2026-06-18
- **数据路径**：`d:\Projects\ProfilingTest\MultiProfLevel2MemoryUB_db\`
- **数据范围**：Rank 0–15（2 节点 × 8 卡），单 active step（step id=2，schedule: skip_first=1/wait=1/warmup=0/active=1/repeat=1），单步跨度 ~6.33 s（`STEP_TIME` 6.479e9 ns，advisor E2E 6326.994 ms）。
- **集群拓扑**：node1（hostUid …8541）= Rank 0–7（device 0–7）；ubuntu2204（hostUid …5777）= Rank 8–15（device 0–7）。算法 `Megatron-LM(tp-dp-pp)`；通信域：节点内 collective (0–7)/(8–15) 为 TP=8，跨节点 (i, i+8) p2p/collective 对为 DP，另有全局 (0–15) 组。
- **采集配置**：Level2、`export_type=db`、`record_shapes=true`、`profile_memory=true`、`with_stack=true`、`with_flops=true`、`aic_metrics=ACL_AICORE_MEMORY_UB`。
- **使用的 Skills**：
  - `mindstudio_profiler_data_check`：校验 16 Rank 数据完整性与采集配置
  - `dataset-source-identifier`：识别落盘数据来源/模型/用途（含识别依据，无依据留空）
  - `ascend-profiler-db-explorer`：DB SQL 查询（compute/comm/dispatch 宏 + Track B 表）
  - `cluster-fast-slow-rank-detector`：快慢卡诊断（slow_rank / free / 算子对比）
  - `msprof-analyze-cli`：集群 7 模式分析 + advisor
  - `op-mfu-calculator`：MFU 评估（本数据集 MFU 不可算，见下）
  - `performance-health-score`：PHS 评分
  - `msinsight-view-selector`：为每个诊断结果推荐 Insight 可视化视图
- **Advisor 状态**：
  - `msprof-analyze advisor all`：**已调用**（-d 指向 Rank 15 慢卡目录）— 用于头号慢卡根因下钻，输出（Operator Dispatch / AICPU / Block Dim / Bandwidth Contention / Packet / Retransmission / Affinity API）已并入第 2 章行动清单与第 3 章问题详情。原始输出：`advisor_output/rank15/mstt_advisor_20260618120758.html`。
- **msprof-analyze 集群模式**（均 `-o` 至本报告目录 `msprof_analyze/<mode>/`，未污染原始数据）：`slow_rank`、`cluster_time_summary`、`compute_op_sum`、`hccl_sum`、`cann_api_sum`、`free_analysis`、`freq_analysis`；通信矩阵/耗时复用原始 `cluster_analysis_output/cluster_analysis.db`。
- **MFU 说明**：本数据集**无法计算 FLOPs-based MFU**——主力 matmul 为 TP 融合算子（`AllGatherMatmul`/`MatmulReduceScatter` 及其 `*Aicpu` 变体），`InputShapes` 为 N/A；且 `aic_metrics` 选了 `ACL_AICORE_MEMORY_UB`（UB 带宽计数）而非 PipeUtilization，故无硬件 `cube_utilization`。计算子项改用**时间口径 device-busy ≈ 77%（各 rank computing 占比均值）作为代理**，并按 AICPU 占比 40.4%、Block Dim 占比 0.46 下修至 60%。
- **内存说明**：`MEMORY_RECORD` 各 rank `totalReserved` 峰值仅 0.311 GB（`totalAllocated` 0.233 GB），对该规模 LLM 训练明显偏低/不完整（疑采集窗口未覆盖完整 step 分配），故**显存容量利用率不下结论**；HBM 实测带宽 ~20.4 GB/s（均值，含空闲采样）/ 峰值 138–186 GB/s，约为 910B 峰值 1.6 TB/s 的 ~1.3%（非访存 bound），两节点一致。PHS 内存子项记 **N/A**，权重按 `归一化权重=原权重÷(1-0.10)` 放大（计算 0.444 / 通信 0.333 / 调度 0.222）。
- **PHS 计算**：场景=大模型多卡训练（默认）。计算 60（代理，见上）· 通信 57（size-weighted：HCCS/SDMA 54.5%、RDMA 96%）· 调度 84（1−均值 free 13.7%，按 JIT 下发开销小幅下修）· 内存 N/A。`PHS=0.444×60+0.333×57+0.222×84≈64（B+）`。优化后（计算 75 / 通信 70 / 调度 92）`≈77（A）`。
- **数据来源与落盘信息**（落盘文件信息卡片；无确切识别依据的项留空，不臆测）：
  - 数据目录：`MultiProfLevel2MemoryUB_db/`
  - 来源：分布式训练 Profiling（多机多卡）
  - 是否 LLM 训练：是
  - 模型 / 用途：Megatron-LM 训练的 Transformer LLM（具体模型家族/规模无确证依据，留空）
  - 落盘大小：~1.0 GB（16 × ~62 MB DB）
  - 来源链接：
  - 识别依据：`algorithm=Megatron-LM(tp-dp-pp)`（ClusterBaseInfo）；算子签名含 `FlashAttentionScore(/Grad)`、`RmsNorm(/Grad)`、`SwiGlu(/Grad)`、`RotaryMul`(RoPE)、`ApplyAdamW`、`MatmulReduceScatter`/`AllGatherMatmul`（TP 序列并行融合）→ 现代 Transformer LLM 训练；hidden≈5120、seq≈4096、head_dim=128（取自 Cast/算子 shape）。**vocab 维未在任何算子 shape 中出现（fused/AICPU 算子 shape 为 N/A），无法据 tokenizer 词表定家族，模型名留空不猜。**
- **举证文件清单**（已复制至 `evidence/`，报告自包含）：

  | 副本路径 | 原始来源 | 引用的问题点 | 大小 |
  |---|---|---|---|
  | `evidence/rank15_ascend_pt/ascend_pytorch_profiler_15.db` | `…\ubuntu2204_1660973_…_ascend_pt\ASCEND_PROFILER_OUTPUT\ascend_pytorch_profiler_15.db` | 3.1, 3.2, 3.3, 3.6, 3.7 | ~60 MB |
  | `evidence/rank15_ascend_pt/analysis.db` | 同上目录 `analysis.db` | 3.3 | ~0.3 MB |
  | `evidence/rank15_ascend_pt/mstt_advisor_rank15.html` | `…\advisor_output\rank15\mstt_advisor_20260618120758.html` | 3.7（及 3.1/3.2/3.6 advisor 原始来源） | ~0.12 MB |
  | `evidence/cluster/hccl_sum.db` | `…\msprof_analyze\hccl_sum\cluster_analysis_output\cluster_analysis.db` | 3.8 | ~0.05 MB |
  | `evidence/rank9_ascend_pt/ascend_pytorch_profiler_9.db` | `…\ubuntu2204_1660964_…_ascend_pt\…\ascend_pytorch_profiler_9.db` | 3.2（快慢卡对比基准） | ~60 MB |
  | `evidence/cluster/cluster_analysis.db` | `…\MultiProfLevel2MemoryUB_db\cluster_analysis_output\cluster_analysis.db` | 3.4, 3.5, 3.8 | ~2.5 MB |
  | `evidence/cluster/slow_rank.db` | `…\msprof_analyze\slow_rank\cluster_analysis_output\cluster_analysis.db` | 3.3 | ~0.9 MB |
  | `evidence/cluster/free_analysis.db` | `…\msprof_analyze\free_analysis\cluster_analysis_output\cluster_analysis.db` | 3.1 | ~0.9 MB |

---

### 时序结构指标看板（timeline-swimlane-analyzer）

> 由 step trace（`ClusterStepTraceTime`）与 `OVERLAP_ANALYSIS` 派生（本数据集为纯 DB 导出，无 `trace_view.json`，未跑 `timeline_geometry.py`）；仅列已测得的指标。

| 指标 | 值 | 状态 | 说明 |
|---|---|---|---|
| 计算-通信重叠率 | 51% | warn | node2 ~51%（node1 ~83%），node2 重叠不足 |
| 暴露通信 | 0.45–0.68 s | warn | 单步 not-overlapped 通信，占 7–11% |
| 最空闲泳道空挡 | 40.7% | bad | Rank 15 free 占比（node2 均值 18.3%） |
| step 抖动 (CV) | 0.9% | ok | 跨 rank 单步耗时离散（仅 1 step，非跨步抖动） |
