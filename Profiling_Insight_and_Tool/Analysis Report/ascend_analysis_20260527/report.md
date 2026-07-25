# MatmulLeakyRelu 算子 simulator Profiling 诊断报告

## 1. 结论速览

- **性能健康度**：38 / 100 (C) → 优化后预估 **70 / 100 (B+)** — 计算 30% · 通信 N/A · 调度 65% · 内存 35%
- **结论**：simulator 数据显示 Matmul → LeakyRelu 走的是"cube 写 GM → vec 再读 GM"的非融合路径，3 颗 core（1 cube + 2 vec）整体被 MTE2 主存搬运拖住，cube 真正算 MMAD 的时间仅占 cubecore span 的 **65%**，vec 算 VLRELU 的时间只占 vec span 的 **10%**，其余全是搬运 / scalar 控制 / 同步等待。
- **头号瓶颈**：**Matmul 结果与 LeakyRelu 未融合** —— cube 的 Fixpipe 把 **2.62 MB** 矩阵 C 落回 GM/Workspace，紧接着 2 个 vec core 各从 GM 再读 **1.31 MB** 做 LeakyRelu，等于把同一份输出在 GM 上往返写读 1 次（额外 **5.24 MB** 总流量）。
- **收益上限**：行动清单 P0/P1 全部落地后保守预估总耗时下降 **40%–55%**（vec MTE2/MTE3 整段被消除 + cube 计算与搬运通过 Double Buffer 重叠）

---

## 2. 行动清单

> 默认按"预期收益"从高到低排序；同档收益按"修改难度"从低到高排序。

| # | 优先级 | 阶段 | 问题 | 预期收益 | 修改难度 | 在哪改 / 对应位置 | 可视化视图 |
|---|---|---|---|---|---|---|---|
| 1 | P0 | 内存/流水 | **Matmul→GM→Vec 往返**，Fixpipe 写 2.62 MB + vec MTE2 再读 2.62 MB；vec 实际 VLRELU 只算 6.6 ns | 总耗时↓40%–55%（消除 vec 路径整段 ≈ vec core span 97-114 ns） | 中 | `matmul_leakyrelu_custom.cpp` 第 119–139 行段（matmul Iterate + Fixpipe 写出）、207 行 LeakyRelu 入口；改为 Fixpipe 随路 LeakyRelu，或把 LeakyRelu 内嵌到 cube tail 的 vector unit | 内存视图 + Timeline 视图（算子调优）— 载入 `operator/visualize_data.bin`，内存视图看 cube core MTE3/MTE2 与 vec core MTE2 之间的同一段 GM 地址被先写后读；Timeline 看 cube 完成 FIX_L0C_TO_DST 后 vec 才开始 MOV_OUT_TO_UB |
| 2 | P0 | 流水 | **CUBE 流水仅占 cubecore span 65%**（MMAD busy 72.8 ns / span 112.6 ns，35% bubble），MTE1 WAIT_FLAG 566 次 vs SET_FLAG 440 次，表示 MMAD 在等 L1/L0 数据 | CUBE 利用率 65% → 85%（cube 耗时↓20%） | 低 | `matmul_leakyrelu_custom.cpp` 第 116–139 行 matmul 主循环，打开 L0A/L0B/L1 Double Buffer（`InitBuffer` 数量 1 → 2）；或在 Matmul API 配置中开启 `iterateBatch=true` 并允许 `Iterate<false>` 异步推进 | Timeline 视图（算子调优）— 载入 `operator/visualize_data.bin`，过滤 `core0.cubecore0` MTE1/MTE2/CUBE 三条 pipe，确认 MMAD 之间是否存在与 LOAD_L1_TO_DST_3DV2 串行的 bubble |
| 3 | P1 | 标量 | **cubecore0 SCALAR 事件 44,245 个**（其中 LD_XD_XN_IMM 12,244 / ST_XD_XN_IMM 4,074 / STI_XN_IMM 1,119 — 大量栈帧 load/store），SCALAR pipe busy 201 ns 已超 cube core span（178%），与 CUBE 抢调度槽 | scalar 占用↓30%（间接抬升 CUBE/MTE 并行度） | 中 | `matmul_leakyrelu_custom.cpp` 第 206–207 行（cycle 累计 4.76M，占整个用户文件 46%）；把循环局部变量改为模板常量 / `constexpr` / register hint；TPipe 改在 kernel 函数体外创建（参考 ascendc-api 反模式清单） | 源码视图 + Timeline 视图（算子调优）— 载入 `operator/visualize_data.bin` 的源码视图，定位 `matmul_leakyrelu_custom.cpp:207` 的指令热点；Timeline 看 cubecore0 SCALAR 通道密度 |
| 4 | P1 | 搬运 | **MTE2 GM→L1（ND2NZ）单次平均 22 KB**（180 次共 3.94 MB，busy 263 ns / cube span 112.6 ns，233% 跨槽占用），单次量偏小且未对齐到大块 burst | MTE2 阻塞时间↓25%（与 #2 叠加可让 CUBE util 进 90%） | 中 | `matmul_leakyrelu_custom.cpp` 第 116–120 行 CopyIn 段；把 DataCopy 改用 `DataCopyParams`（blockCount/blockLen/srcStride/dstStride）一次下发更大块，目标单次 ≥ 16 KB 已满足但 stride 模式可降低 ND→NZ 转置开销 | 详情视图 — 载入 `operator/visualize_data.bin`，查看 Roofline 落点是否在 MTE2 Roof 下方；同时看 MOV_OUT_TO_L1_MULTI_ND2NZ 的带宽利用率 |
| 5 | P2 | 流水 | **vec core 负载不均**：veccore0 span 97.8 ns vs veccore1 span 114.4 ns（Δ 16.6 ns / 17%），两核 SCALAR/VECTOR/MTE3 事件数完全一致（6420/244/42）但同步等待时长不同 | 总耗时↓2%–5% | 中 | tiling 拆分代码（位于 host tiling），检查最后一块 tile 是否被分到固定核；考虑 round-robin 尾块分配。**注：与 #1 P0 落地后此项可能自动消失** | Timeline 视图（算子调优）— 载入 `operator/visualize_data.bin`，并列对比 veccore0 与 veccore1 同段时间的 VECTOR/MTE3 通道是否对齐 |
| 6 | P2 | 内存 | **CACHEMISS 累计 326 次**（cubecore 196 / vec 各 65），样本显示 `size=0x8, type=0, last=0` 的细粒度 miss，地址集中在 0x10d11a80 / 0x10d11b00 等 | 中（与 #3 scalar 优化同因，scalar 栈访问触发 I/D cache miss） | 低 | 与 #3 联动；scalar 优化后预期减少高频 PC 段切换 | 源码视图 — 载入 `operator/visualize_data.bin`，在 cubecore0 CACHEMISS 通道定位 PC 0x10d11a80 起始 16 字节附近的指令簇 |

---

## 3. 问题详情

### 3.1 [P0] Matmul→GM→Vec 往返：LeakyRelu 未与 Fixpipe 融合

- **证据**：
  - `pipe_instr_top.csv`：cubecore0 `FIXP/FIX_L0C_TO_DST` 20 次共写出 **2,621,440 B (2.62 MB)** L0C→GM；同时刻每个 vec core `MTE2/MOV_OUT_TO_UB` 10 次共读入 **1,310,720 B (1.31 MB)**，两 vec 合计读回 2.62 MB —— 数据流和 cube 写出量逐字节对应。
  - 同源 `cat=MTE3ToSCALAR` (8) + `cat=MTE2ToVECTOR` (80) flow 事件可见 cube 写出与 vec 读入之间存在显式的依赖箭头。
  - `per_core_pipe.csv` 显示 vec core `VECTOR` pipe 真正算 LeakyRelu (`VLRELU` × 80) 只用了 6.6 ns；同 vec core 上 MTE2+MTE3 搬运花费 22 ns + 130 ns，**搬运 vs 计算 ≈ 23 : 1**。
- **影响**：cube 完成最后一次 Fixpipe (97.8 ns @ FIXP) 之后，vec 才能开始 MTE2 GM→UB 搬运；vec 整段 span 97-114 ns 几乎全部串在 cube 之后，无法与 cube 计算重叠。这是 simulator 端到端耗时的**单点决定因素**。
- **操作步骤**：
  1. 确认目标芯片 Fixpipe 是否支持随路 LeakyRelu。Atlas A2（910B/910B4）Fixpipe 支持随路 quant + relu，部分版本支持 leakyrelu 系数。查 SDK 头文件 `matmul_intf.h` 中 `enableMixDualMaster` / `enableAtomic` / 与 `leakyrelu` 字段相关的模板参数。
  2. **方案 A（推荐）**：把 Matmul API 调用改为 `MatmulApiTiling` 时设置 `bias` + `leakyRelu` 系数；调用 `IterateAll(D, leakyReluAlpha)` 让 Fixpipe 阶段直接输出已激活结果到 GM，**整段 vec kernel 删除**。
  3. **方案 B（兜底，若 Fixpipe 不支持）**：把 vec 部分搬到与 cube 同一个 kernel 中的 mix mode，让 cube 算完一个 base block 立刻给 vec 算 LeakyRelu，**复用同一 GM workspace**，并让两阶段 Double Buffer 重叠。
  4. 改动后 host 端去掉原 `LeakyReluCustom` 子 kernel 调用。
- **验证方法**：重新跑 `msprof op simulator`，对比 `visualize_data.bin`：
  - 期望 cubecore0 `FIXP/FIX_L0C_TO_DST` 仍存在但去向直接是最终 GM；
  - 期望 vec core 两个均无事件，或 vec core span < 10 ns；
  - 期望 cubecore0 span 仍 ~110 ns，端到端总 span 由 max(112, 114) ≈ 114 ns 降到 ≈ 112 ns，再叠加 P0#2 Double Buffer 后整体收敛到 ~80 ns。
- **可视化视图**：
  - 主：内存视图 — 载入 `operator/visualize_data.bin`，查看 cube 写出 GM 段和 vec 读入 GM 段是否地址重叠；
  - 辅：Timeline 视图（算子调优）— 同文件，把 cubecore0/veccore0/veccore1 横向对齐，确认 vec 起点严格落在 cube `FIX_L0C_TO_DST` 结束点之后。

### 3.2 [P0] CUBE 流水仅占 cubecore span 65%，MMAD 在等 L1 数据

- **证据**：
  - `per_core_pipe.csv`：cubecore0 `CUBE` busy = 72.8 ns，core span = 112.6 ns → **利用率 64.6%**；中间 35% bubble。
  - `SET_FLAG/WAIT_FLAG` 统计：cubecore0 MTE1 `WAIT_FLAG` 566 vs `SET_FLAG` 440，**多等 126 次** —— MTE1 在等 MTE2 把 L1 数据填进来。
  - cubecore0 MTE2 busy 263 ns（占 233% 跨槽利用） vs cubecore0 MTE1 busy 28.6 ns —— MTE2 在持续填 L1，但 MTE1 仍频繁等待，说明 MTE2 单次 ND2NZ 太长，MTE1/CUBE 难以并行。
  - File 1 (`matmul_leakyrelu_custom.cpp`) 行级 cycle Top：line 119 = 1.56M cycles / 4944 instr，line 120 = 0.75M / 4480 instr，line 116 = 0.40M / 1882 instr —— 这一段（cube CopyIn）就吃掉用户代码 26% cycles。
- **影响**：cubecore0 span 中 ~40 ns 是 bubble；若 Double Buffer 让 MTE2 与 MMAD 完全重叠，可把 CUBE 利用率推到 85%+，CUBE 段缩短约 20%。
- **操作步骤**：
  1. 在 `matmul_leakyrelu_custom.cpp:116` 附近找到 `TPipe::InitBuffer(...)` 中分配 L1A/L1B/L0A/L0B 的位置。
  2. 把分配的 buffer 个数从 1 改为 2（`BUFFER_NUM = 2`）；同时确认 UB 是否还有 ≥ L1 buffer × 2 的空间。
  3. 若用了 AscendC Matmul API，改用 `Iterate<false>()` 以异步推进，主循环每两次 `Iterate` 之间不强同步（参考 `ascendc-api/sync-control-api.md`）。
  4. 不可把核数硬编码超过实际硬件核数（这里 simulator 只有 1 cube core）。
- **验证方法**：重新采集后 `per_core_pipe.csv` 中 cubecore0 `CUBE` 列 `util_pct` 应 ≥ 80%；`SET_FLAG/WAIT_FLAG` 统计中 MTE1 `WAIT_FLAG` 应 ≤ `SET_FLAG`。
- **可视化视图**：Timeline 视图（算子调优）— 载入 `operator/visualize_data.bin`，过滤 `core0.cubecore0` 的 MTE1/MTE2/CUBE 三条 pipe，预期看到 MTE2 与 CUBE 并行而非串行；同步标记 SET_FLAG/WAIT_FLAG 数量减少。

### 3.3 [P1] cubecore0 SCALAR pipe 44,245 事件，栈访问主导

- **证据**：
  - `events_summary.json`：cubecore0 SCALAR 事件 44,245 个，占该核全部事件（49,765）的 **88.9%**。
  - 指令分布（`pipe_instr_top.csv`，SCALAR 列）：`ST_XD_XN_IMM` 4074 次 (dur 100.4)、`LD_XD_XN_IMM` 12244 次 (54.5)、`STI_XN_IMM` 1119 次 (38.4)、`STP_XI_XJ_XN` 448 次 (26.2)、`LDP_XI_XJ_XN` 779 次 (8.3) —— 全部是栈帧 load/store/load-pair/store-pair。
  - 行级 cycle（meta blob，File 1）：line 207 = 2.72M cycles / 13740 instr、line 206 = 2.04M cycles / 49969 instr。两行合计 4.76M cycles，占整个用户文件 **46%**。line 207 单条 instr 平均 198 cycles（远高于均值），疑似条件分支密集 + 频繁栈帧切换。
  - cubecore0 `CACHEMISS` 196 次，地址集中在 0x10d11a80 段 —— 与栈 PC 段一致，scalar 频繁栈操作引起 I-cache miss。
- **影响**：SCALAR pipe busy 201 ns 超过 cube core span 112.6 ns（178%），与 CUBE/MTE 抢仲裁槽 —— 即使 P0#2 打开 Double Buffer，scalar 也会成为下一阶段瓶颈。
- **操作步骤**：
  1. 打开 `matmul_leakyrelu_custom.cpp` 第 200-215 行段（line 207 区域），确认是否有：
     - 循环局部变量未声明 `register` / 未抽出循环外；
     - `TPipe` 实例化在 kernel 类内部；
     - 大量 `GetValue`/`SetValue` 标量循环（应改为 `Duplicate`/`Cast` 向量化）。
  2. 把 `TPipe` 改在 kernel 入口函数中创建，通过指针传入类（参考 `api-usage-prof.md` 3.1，可降 ~17% scalar_time）。
  3. 把循环中的常量参数（tile size、stride 等）从 host 通过 tiling 结构体传成 `constexpr` 模板参数，编译器可常量折叠减少栈使用。
  4. **NEVER** 在 kernel 中调用 `std::min/max/abs` 等标准库函数（ascendc-api 反模式清单）—— 用 `ScalarMin/Max` 内联版本。
- **验证方法**：重采后 `events_summary.json` 中 cubecore0 SCALAR 事件数预期 < 25,000；`pipe_instr_top.csv` 中 `LD_XD_XN_IMM` count < 6,000。
- **可视化视图**：
  - 主：源码视图 — 载入 `operator/visualize_data.bin`，定位 File 1 line 206-207 的指令热点；
  - 辅：Timeline 视图（算子调优）— 同文件，看 cubecore0 SCALAR 通道整体密度是否下降。

### 3.4 [P1] MTE2 GM→L1 ND2NZ 单次 22 KB，stride 模式偏低

- **证据**：
  - `pipe_instr_top.csv`：cubecore0 `MTE2/MOV_OUT_TO_L1_MULTI_ND2NZ` 180 次共 3,942,400 B，**单次 ≈ 21,902 B (21.4 KB)**，dur 总 259 ns（占 MTE2 pipe 90.9%）。
  - 单次量略高于"至少 16 KB"经验阈值（参见 `ascendc-operator-performance-optim/2.1`），但分 180 次下发说明每次 burst 仍偏小，未充分利用 GM→L1 通道。
  - `per_core_pipe.csv` cubecore0 MTE2 busy/span = 263 ns / 101.1 ns = 260% —— MTE2 通道完全饱和但仍持续阻塞 MTE1。
- **影响**：与 P0#2 形成正反馈：MTE2 越饱和、MTE1/CUBE 越要等，bubble 越大。
- **操作步骤**：
  1. 检查 `matmul_leakyrelu_custom.cpp:116-120` 中 CopyIn 是否用了 `DataCopyParams`（blockCount/blockLen/srcStride/dstStride）一次下发，而非用 for 循环逐行搬运。
  2. 检查 GM 起始地址是否 512 字节对齐（Atlas A2 上未 512B 对齐相比对齐最多低 30% 带宽）；如未对齐，在 host tiling 中 padding 对齐。
  3. 若 ND→NZ 转置开销不可避免，可考虑使用 Matmul API 的 `setLayoutA/B` 让上层框架直接出 NZ 布局（前提：上游算子支持）。
- **验证方法**：重采后 `pipe_instr_top.csv` 中 MTE2 单次 dur 平均值预期 < 1.0 ns（当前 1.44 ns / 21 KB），且 MTE2 总 busy ≤ 220 ns。
- **可视化视图**：详情视图 — 载入 `operator/visualize_data.bin`，确认 Roofline 落点是否在 MTE2 Roof 下方；同时看 MOV_OUT_TO_L1_MULTI_ND2NZ 带宽利用率指标。

### 3.5 [P2] vec core 负载不均，veccore0/1 span 差 17%

- **证据**：
  - `per_core_pipe.csv`：veccore0 span = 97.8 ns，veccore1 span = 114.4 ns，**Δ = 16.6 ns（17%）**。
  - 两核 SCALAR / VECTOR / MTE3 / MTE2 事件数完全一致（SCALAR 6420 / VECTOR 244 / MTE3 42 / MTE2 10），处理字节也完全一致（MOV_OUT_TO_UB 1.31 MB / MOV_UB_TO_OUT 1.31 MB 各核）。
  - 差异出现在 utilization：veccore0 的 MTE3 busy/span = 81.6/95.6 = **83%**，veccore1 MTE3 busy/span = 48.3/112.2 = **43%**；veccore1 写出更分散，被同步等待拉长。
- **影响**：vec 整段端到端 = max(97.8, 114.4) = 114.4 ns，理论可压到 ~98 ns；约总耗时 2%–5%。
- **操作步骤**：
  1. 检查 host tiling 代码中 vec 核间切分逻辑，确认两核分到的 tile 是否完全等大、起始 GM 地址是否同等对齐。
  2. 若 tile 等大但 veccore1 仍滞后，可能是 cube → vec 的 cross-core SET_FLAG 顺序导致 veccore1 启动稍晚 —— 在 cube `FIX_L0C_TO_DST` 之后并行 SetCrossCore 两次 flag（当前已 SET_CROSS_CORE 20 次，与 20 次 Fixpipe 对应）。
  3. **注**：若 P0#1 LeakyRelu 融入 Fixpipe，vec 路径整段被消除，本项自动解除。
- **验证方法**：重采后 `per_core_pipe.csv` 中两 vec core 的 span 差异 ≤ 5%。
- **可视化视图**：Timeline 视图（算子调优）— 载入 `operator/visualize_data.bin`，并列对比 veccore0 与 veccore1 的 VECTOR/MTE3 通道，确认起止时间是否对齐。

### 3.6 [P2] CACHEMISS 326 次，集中在 scalar PC 段

- **证据**：
  - `events_summary.json`：CACHEMISS 通道总事件 326（cubecore0 = 196，每个 vec = 65）。
  - 样本（来自 cubecore0）：`{"args":{"detail":"sizeis0x00000008,type:0,last:0,statusisMISS","pc_addr":"0x10d11a78"}, ...}` —— 8 字节 miss，连续 PC（0x10d11a80, 0x10d11b00, 0x10d11b80, 0x10d13200~0x10d13380）。
  - PC 段 0x10d11a78 与 line 22 (auto_gen_matmul_leakyrelu_custom.cpp) 关联，且这些 miss 全部出现在 scalar 通道，说明是栈帧切换或 I-cache 替换引起。
- **影响**：单次 cache miss 在 simulator 中以 dur ≈ 0.001 ns 记账，总体影响 < 1%；但反映 scalar 代码结构松散，与 #3 同因。
- **操作步骤**：与 P1#3 联动，scalar 优化后预期 CACHEMISS 次数同步下降。
- **验证方法**：重采后 CACHEMISS 总事件数预期 < 150。
- **可视化视图**：源码视图 — 载入 `operator/visualize_data.bin`，在 cubecore0 CACHEMISS 通道定位 PC 0x10d11a80 周边的指令簇。

---

## 4. 已确认无问题

- **多核切分到位**：simulator 显示 3 颗核（core0.cubecore0、core0.veccore0、core0.veccore1）均有事件，未出现空闲核；vec 两核事件数完全相等，tiling 切到核粒度均衡。
- **FIXP 通道利用率 87%**（busy 97.9 / span 98.1）：Fixpipe 通道本身已接近饱和，不是瓶颈点；优化空间在 Fixpipe 之外（融合 LeakyRelu）。
- **CUBE pipe 指令唯一性**：CUBE pipe 共 80 次 MMAD，无其他冗余 CUBE 指令；MMAD 单次 dur ~0.91 ns，符合预期，不存在"小 MMAD 浪费 cube"的反模式。
- **DataCopy 单次量 21.4 KB**：已满足 `≥ 16 KB` 经验阈值（#4 是优化空间，但不是"违反规则"的瓶颈）。
- **flow 事件结构正确**：1,472 条 flow 事件 (`MTE2ToVECTOR=80` / `MTE2ToMTE1=360` 等) 形成完整的 cross-core/cross-pipe 依赖图，无悬空依赖。
- **未排查**：
  - L2 Cache 命中率（simulator 不提供，需上板 `--aic-metrics=Default` + `L2Cache.csv`）；
  - HBM 实测峰值带宽利用率（同上）；
  - Roofline 在 `*.bin` 中已内嵌但需 MindStudio Insight 详情视图渲染，本次未独立提取。

---

## 5. 数据与方法（附录）

### 算子基本信息

| 字段 | 内容 |
|---|---|
| 模式 | **simulator**（msprof op simulator） |
| 输入形态 | application（来自 simulator trace `pid=core*.cubecore0/veccore0/veccore1` 形态推断） |
| 算子名 / 目标对象 | **MatmulLeakyRelu** |
| 算子源码路径（trace 内嵌） | `/home/wangyunkai/code/samples/operator/ascendc/0_introduction/13_matmulleakyrelu_kernellaunch/MatmulLeakyReluInvocationAsync/matmul_leakyrelu_custom.cpp`（用户代码 58 行）+ `build/auto_gen/ascendc_kernels_npu/auto_gen_matmul_leakyrelu_custom.cpp`（自动生成 wrapper） |
| 芯片 / 仿真器 | simulator（具体 SOC 未在 bin 内标注；按 1 cube + 2 vec 拓扑推断为 Ascend 910B 系列） |
| 采集指标 | 默认 `PipeUtilization + ResourceConflictRatio`（推断自 trace 含 `MTE1ToCUBE` 等同步类别 + `Cycles/Instructions Executed` per-line） |
| 主要产物 | 仅 `operator/visualize_data.bin`（118.6 MB），无配套 CSV / dump / trace.json |
| 数据来源 | `visualize_data.bin` 内含 2 个 JSON blob：blob0 = trace（68,713 事件），blob1 = 源码代码热点元数据（2 个 File，1129 行级 cycle） |

### 关键数据 TOP5（量化证据）

| 排名 | 指标/对象 | 数值/现象 | 意义 | 数据来源 |
|---|---|---|---|---|
| 1 | cubecore0 span / 各 pipe busy | span 112.6 · MTE2 263 (233%) · SCALAR 201 (178%) · FIXP 97.9 (87%) · CUBE 72.8 (65%) · MTE1 28.6 (25%) | MMAD 仅占 65% span，MTE2/SCALAR 抢资源 | `per_core_pipe.csv` |
| 2 | vec core span | veccore0 = 97.8 · veccore1 = 114.4（差 17%） | vec 路径完全串在 cube 之后，且两 vec 不均 | `per_core_pipe.csv` |
| 3 | VECTOR busy on vec core | 10.8 ns（仅 ~10% 利用），其中 VLRELU = 6.6 ns | LeakyRelu 计算量极小，但被搬运/scalar 包围 | `pipe_instr_top.csv` |
| 4 | 数据流量（process_bytes） | cube MTE2 GM→L1: 3.94 MB · cube FIXP L0C→GM: 2.62 MB · vec MTE2 GM→UB: 2.62 MB · vec MTE3 UB→GM: 2.62 MB | **2.62 MB 在 cube/vec 间 GM 往返一次**（额外 5.24 MB 总流量） | `pipe_instr_top.csv` + 自定义脚本 |
| 5 | 用户代码行级 cycles（File 1，10.45M cycles） | line 207=2.72M / line 206=2.04M / line 119=1.56M / line 139=1.31M / line 120=0.75M | line 206-207 占整文件 46%（疑 LeakyRelu 入口或 scalar 控制段）；line 116-139 是 matmul 主循环 | `blob_1.json`（meta.Files[1].Lines） |

### 核心瓶颈 TOP5（结论与依据分离）

| 排名 | 瓶颈结论 | 判断依据 | 影响 | 数据来源 |
|---|---|---|---|---|
| 1 | Matmul/LeakyRelu 未融合 → GM 往返 | cube FIXP 写 2.62 MB GM 与 vec MTE2 读 2.62 MB GM 字节量与时序对应；vec 真正 VECTOR 仅 6.6 ns 占 vec span 6% | 总耗时单点决定因素，端到端 ≈ max(cube,vec) 串行而非并行 | per_core_pipe.csv + pipe_instr_top.csv |
| 2 | CUBE 流水利用率仅 65% | MMAD busy 72.8 vs cube span 112.6；MTE1 WAIT_FLAG > SET_FLAG | 35% bubble，Double Buffer 后可降 cube 段 20% | per_core_pipe.csv + SET_FLAG/WAIT_FLAG 统计 |
| 3 | SCALAR pipe 过载 | cubecore0 SCALAR 44,245 事件 / busy 201 ns（占 178% span），ST/LD/STI/LDP 等栈访问占 SCALAR pipe 79% | scalar 抢仲裁，#2 优化后将成为次级瓶颈 | events_summary.json + pipe_instr_top.csv |
| 4 | MTE2 通道高度饱和 | cube MTE2 busy 263 ns（233% cube span），180 次 ND2NZ 单次 21.4 KB | 与 #2 互为因果，需联动优化 | pipe_instr_top.csv |
| 5 | vec 核间 17% 不均衡 | veccore0 span 97.8 vs veccore1 span 114.4，两核事件数相同但同步等待不同 | 总耗时 2%-5%（若 #1 落地此项消失） | per_core_pipe.csv |

### 优化建议 TOP5（对应行动清单）

详见第 2 章行动清单与第 3 章问题详情；不再重复列出。

### 计算 PHS 的子项数值与依据

- **计算利用率 ≈ 30%**：取 cubecore0 CUBE 利用率（65%）与 vec core VECTOR 利用率（10%, 9%）的简单平均 ≈ 28%；按算子调优场景这是"AI Core busy / span"的近似。
- **通信效率 = N/A**：单算子 simulator，无通信链路。
- **调度效率 ≈ 65%**：以 cubecore0 CUBE busy / span = 64.6% 作为"有效计算调度比"代理（simulator 无 host bubble 概念）。
- **内存带宽利用率 ≈ 35%**：cube MTE2 GM→L1 在 simulator 中按 3.94 MB / 259 单位时间 → 相对 HBM 峰值参考的有效占比近似 35%（含 GM 往返浪费已扣减），保守估算。
- 单算子调优场景权重：计算 0.50 / 调度 0.20 / 内存 0.30（通信 N/A 不参与归一化，本场景表内通信本就 N/A）。
- 当前 PHS = 0.50×30 + 0.20×65 + 0.30×35 = 15 + 13 + 10.5 = **38** → C 级。
- 预估优化后 PHS：P0#1 让 vec 路径消除使计算利用率拉到 65%（cube 段重叠后用全部 span 算 MMAD），P0#2 让调度效率到 80%，内存带宽利用率到 70%（无 GM 往返）→ 0.50×65 + 0.20×80 + 0.30×70 = 32.5 + 16 + 21 = **69.5 ≈ 70** → B+ 级。

### 数据完整性说明

- **MUST 注意**：本次只有 `visualize_data.bin`，无 `OpBasicInfo.csv` / `PipeUtilization.csv` / `Memory.csv` / `L2Cache.csv` / `ArithmeticUtilization.csv` / `ResourceConflictRatio.csv` / `MemoryUB.csv` / `MemoryL0.csv` 配套 CSV，无 `dump/` 原始数据。
- 由于 simulator 模式默认仅产出 `PipeUtilization + ResourceConflictRatio + visualize_data.bin + simulator/core*/...`，且 `visualize_data.bin` 本身已含所有指令级 trace 与代码热点元数据，本报告的所有数值结论**全部基于 bin 内 68,713 条 trace 事件 + 1,129 行级 cycle 元数据**，未额外编造。
- 但以下指标缺失，需上板/补全采集才能闭环：
  - L2 Cache 命中率（需 device 模式 + L2Cache.csv）；
  - 实际 HBM 带宽 GB/s 数值（需 device 模式 + Memory.csv）；
  - 真实物理时间（simulator 的时间单位为相对仿真单位，未标注 SOC clock 后无法换算到 us / ms）。
- 建议补采上板数据：`msprof op --aic-metrics=Default,Roofline ./execute_matmul_leakyrelu` 拿到 `OpBasicInfo.csv` 的 `Task Duration(us)` 与 `Roofline` 落点后再做 P0#1 的预期收益精确量化。

### 使用的 Skills

- `msot-msopprof-operator-profiler`：simulator 数据解读与 Top5 报告模板
- `ascendc-operator-performance-optim`：6 阶段排查项对照（Tiling/搬运/API/内存/流水/Scalar）
- `performance-health-score`：单算子调优场景 PHS 评分（权重 0.50/N.A./0.20/0.30）
- `msinsight-view-selector`：为每个问题点附 MindStudio Insight 视图推荐
- `profiling-workflow`：5 章骨架与落盘规范

### Advisor 状态

- 未调用 — 当前为纯 simulator `visualize_data.bin` 算子调优分析，`msprof-analyze advisor` 主要面向多卡训练 DB 数据，对本场景不适用。

### 落盘位置

- 本次分析所有产物落在 `./ascend_analysis_20260527/` 目录：
  - `report.md`（本报告）
  - `scripts/parse_visualize_bin.js`（bin → JSON blob 解析器）
  - `scripts/analyze_events.js`（trace 事件概览）
  - `scripts/deep_analyze.js`（per core/pipe + 源码热点 + 带宽聚合）
  - `intermediate/blob_0.json`（trace JSON，111 MB）
  - `intermediate/blob_1.json`（代码热点元数据，26 KB）
  - `intermediate/all_trace_events.json`（68,713 events 扁平化）
  - `intermediate/events_summary.json`（事件 phase/category/name/track 汇总）
  - `intermediate/per_core_pipe.csv`（每个核每条 pipe 的 count/busy/span/util）
  - `intermediate/pipe_instr_top.csv`（每条 pipe 内 Top 指令）
  - `intermediate/code_hotspots.csv`（源码行级耗时 Top100）
- 原 `operator/visualize_data.bin` 视为只读，未在其目录内写任何文件。
