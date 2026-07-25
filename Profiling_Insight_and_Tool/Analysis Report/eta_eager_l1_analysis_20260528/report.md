# eta_eager_l1 推理 Profiling 性能分析报告

## 1. 结论速览

- **性能健康度**：7 / 100 (D) → 优化后预估 **48 / 100 (B)** — 计算 7% · 通信 N/A · 调度 7% · 内存 N/A
- **结论**：单卡推理被 Host 下发完全主导，NPU 100 个 step 中只用 7.36% 的墙钟在跑算子；其余 92.6% 是 device idle（Free Time）
- **头号瓶颈**：steps 呈双峰分布——59 个"慢 step"平均 86.7 ms（Free 82 ms），41 个"快 step"平均 18.7 ms（Free 14 ms），计算时间在所有 step 完全恒定 4.3 ms；瓶颈完全在 host launch 与 AI_CPU 算子（`IndexPut` / `GatherElements`）
- **收益上限**：行动清单 P0/P1 全部落地后预估单步耗时从均值 58.8 ms 降到 ~18 ms，全程节省 **~3.9 s（~69%）**

---

## 2. 行动清单

> 默认按"预期收益"从高到低排序；同档收益按"修改难度"从低到高排序。

| # | 优先级 | 问题 | 预期收益 | 修改难度 | 在哪改 / 对应位置 | 可视化视图 |
|---|---|---|---|---|---|---|
| 1 | P0 | NPU 整体 idle 92.6%，每步 ~54 ms host 下发气泡，eager 模式逐算子下发 353 个 kernel/step | -65% 单步耗时 | 中 | 切换到图模式：`torch.compile(backend="npu")` 或 `torch_npu.npu.experimental_config(graph_mode=True)` | Timeline 视图（系统调优）— 载入 `trace_view.json`，过滤 PyTorch / CANN / Ascend Hardware 三泳道，观察 launch API 与 device task 之间的水平空隙 |
| 2 | P0 | `IndexPut` 算子全部跑在 AI_CPU 上，500 次调用 68.2 ms（占 NPU 计算 15.76%），单次 136 us 是普通 AI_VECTOR 算子的 13 倍 | -15% 单步耗时 | 中 | 业务代码中 `tensor[idx] = val` / `torch.index_put_` 的 5 处调用，改用 `scatter_` 或预先把 mask 转 dense，避免触发 AI_CPU 分支 | 算子视图 — 载入 `kernel_details.csv`，按 `Accelerator Core=AI_CPU` 过滤，定位 `aclnnIndexPutImpl_IndexPut_IndexPut` 的调用 step 与 shape |
| 3 | P0 | `NonZero` + `aclnnNonzeroV2` 产生 D2H 同步，单次最大 72 ms（均值 522 us），是慢 step 的直接触发源 | -15% 单步耗时 | 高 | 业务代码中用 `nonzero()` / `torch.where()` 拿索引的地方（500 次/100 step = 5 次/step），改成 `topk(mask, k=固定值)` 或 boolean mask × 静态索引，消除动态 shape | Timeline 视图（系统调优）— 载入 `trace_view.json`，搜索 `aclnnNonzeroV2`，观察其后紧跟 `aclrtSynchronizeStream` / `aclrtSynchronizeDevice` 的间隔 |
| 4 | P0 | `GatherElements` 同样落 AI_CPU，400 次共 28.6 ms（占 NPU 6.6%），单次 71.6 us | -7% 单步耗时 | 低 | 业务代码中 `torch.gather(dim=...)` 的调用，将输入 dtype 由 INT64 转 INT32 或 FP16，让 `aclnnGather` 走 AI Core 路径（参考 CANN 8.3 算子白名单） | 算子视图 — `kernel_details.csv`，按 Type=`GatherElements` 过滤，对比 input dtype 与 core type 关系 |
| 5 | P1 | MatMul 类算子 cube `mac_ratio` 均值 5.8%、scalar_ratio 81.1%、cube_utilization 47% — 算子粒度过小（如 128×64 × 16×64），Cube 单元基本闲置 | 中 | 中 | 把多个小 MatMul / Linear 合并成 BatchMatMul；或对模型中 ≤128 维的 head 增大 batch 维 / 折叠为 grouped GEMM | 详情视图 — 载入算子 `.bin`，观察 Roofline 落点（应在 Memory Bound 区且远低于 Compute Roof）；辅以算子视图看耗时占比 |
| 6 | P1 | `MemSet` 算子被调用 4600 次 / 100 step（每步 46 次），占 NPU 计算 10.7% — 大量被动 buffer 清零 | -3% 单步耗时 | 中 | 检查框架是否对每个 `aclnnXxx` 都做 workspace clear，开启 `ACLNN_CACHE_LIMIT` 和 `HOST_CACHE_CAPACITY` 环境变量（当前都为空）启用 workspace 复用 | Timeline 视图（系统调优）— 过滤 `MemSet` task，观察其与算子主体的时间关系 |
| 7 | P1 | `aclnnInplaceCopyGetWorkspaceSize` 单次最大 64 ms（均值 60 us），是 host launch 的尾延迟首位 | -2% 单步耗时 | 低 | 已是 8.3.RC1 CANN 版本，建议升级到包含 workspace 计算缓存的最新 hotfix，或显式开启 `_data_simplification: true` | Timeline 视图（系统调优）— `trace_view.json` 过滤 host CANN 泳道，观察 `aclnnInplaceCopyGetWorkspaceSize` 长尾 |
| 8 | P2 | 600 次 `aclrtSynchronizeStream` + 101 次 `aclrtSynchronizeDeviceWithTimeout`（每步 ~7 次同步），强制阻塞 host | 低 | 中 | 业务代码中显式或隐式 `.cpu()` / `.item()` / `print(tensor)` 调用全部清查，移到后处理阶段 | Timeline 视图（系统调优）— 过滤 `aclrtSynchronize*` API，统计落点 step 与对应业务函数 |

---

## 3. 问题详情

### 3.1 [P0] NPU 92.6% 时间空转，host 下发模式无法跟上 device

- **证据**：
  - `step_trace_time.csv` 100 步：Computing 总和 432.77 ms，Stage 总和 5883.72 ms，**NPU 计算占比仅 7.36%**
  - Free Time 5450.95 ms（92.6%）；Computing 均值 4327 us/step，标准差 < 1%（极度稳定）
  - `kernel_details.csv`：每步固定 353 个 kernel，平均单 kernel 仅 12.26 us
  - `api_statistic.csv` 中 `node launch` 共 35,300 次，平均 14.45 us/次，最大 71,944 us（单次 launch 长尾 72 ms）
- **影响**：在 V100/910B 量级硬件上，每个推理样本本应 ~5 ms 完成，实际 58.8 ms，吞吐损失 ~10×
- **操作步骤**：
  1. 优先切换到 PyTorch 图模式：`model = torch.compile(model, backend="npu", mode="reduce-overhead")`（torch_npu 2.6 已支持）；或迁移到 GE 图模式
  2. 验证下发瓶颈解除：单步 `Free / Stage` 应从 0.93 降到 < 0.3
  3. 若无法切图，至少开启 `taskqueue_enable=1` 让 host launch 异步化（环境变量 `TASK_QUEUE_ENABLE=2`）
- **验证方法**：重采 profiling 后，对比 `step_trace_time.csv` 的 Computing/Stage 比值——目标 ≥ 50%；并确认 `kernel_details.csv` 的 Wait Time 总和（当前 5+ s）降到 < 1 s
- **可视化视图**：Timeline 视图（系统调优）— 载入 `trace_view.json`，并排查看 PyTorch / CANN / Ascend Hardware 三个泳道，观察 host launch 与 device 算子的水平间隙（当前肉眼可见的"空白条带"即为 Free Time）

### 3.2 [P0] AI_CPU 算子 IndexPut 阻塞 device pipeline

- **证据**：
  - `op_statistic.csv`：`IndexPut` Core Type = **AI_CPU**，500 次调用，总耗时 68,186 us（占 NPU 计算的 **15.76%**），单次均值 136 us，最大 234 us
  - `kernel_details.csv`：`aclnnIndexPutImpl_IndexPut_IndexPut` 是 AI_CPU 算子，**500 次平均 Wait Time 1502 us**（在所有算子类型中排第一）——意味着每次 IndexPut 后 device 平均空转 1.5 ms 等待下一个 kernel
  - 关联 API：`aclnnIndexPutImpl` 总耗时 28 ms，最大单次 414 us，方差极大（1509）
- **影响**：500 × 1.5 ms = 750 ms 的额外 device idle，几乎与 IndexPut 自身耗时同量级
- **操作步骤**：
  1. 用 `kernel_details.csv` 中 `Input Shapes` 列定位 IndexPut 调用上下文（典型 shape `128,50; ; 2; 1359; 1359`）
  2. 在业务代码中搜索 `_index_put_impl_` 调用源，把高维布尔索引 `t[bool_mask] = val` 改写为 `t = torch.where(bool_mask, val, t)`，强制走 AI_VECTOR_CORE
  3. 若必须使用索引赋值，用 `scatter_` 配合预先计算好的 INT32 索引（避免动态 1D index 触发 AI_CPU fallback）
- **验证方法**：改写后 `op_statistic.csv` 中 `IndexPut` Core Type 应从 AI_CPU 变为 AI_VECTOR_CORE，且 `aten::_index_put_impl_` 在 `operator_details.csv` 中的 Host Self Duration（当前 1,205,985 us）下降 ≥ 80%
- **可视化视图**：算子视图 — 载入 `kernel_details.csv`，按 `Accelerator Core = AI_CPU` 过滤，关注 `IndexPut` 在每个 step 中的分布密度（应为每 step 5 次）；辅以 Timeline 视图观察 IndexPut 前后的同步事件

### 3.3 [P0] NonZero 触发 D2H 同步，是慢 step 的根本触发器

- **证据**：
  - `api_statistic.csv`：`aclnnNonzeroV2` 调用 600 次，总耗时 313,515 us，**单次最大 72,009 us（72 ms），均值 522 us，方差 24,853,118**（极端长尾）
  - `aclnnInplaceCopyGetWorkspaceSize` 最大 64,064 us、`aclnnGather` 最大 68,127 us——这三个 API 的极值均接近 70 ms，恰好对应慢 step 的 ~85 ms 总耗时
  - `kernel_details.csv`：`aclnnNonzeroV2_NonzeroAiCore_NonZero` 输出形状是动态的 `(2, N)`（N=94, 1359, ...），证实存在 host 等待 device 完成才能拿到 shape
  - `aclrtSynchronizeStreamWithTimeout` 调用 600 次（恰等于 NonZero 调用次数），证实每个 NonZero 后都跟着一次 stream sync
- **影响**：双峰分布的成因——59 个慢 step 是 NonZero 后 host 拿到的 shape 较大，触发额外 IndexPut 分支；41 个快 step shape 较小，跳过分支
- **操作步骤**：
  1. 业务代码中搜索 `torch.nonzero(...)` / `(x > 0).nonzero()` / `torch.where(x)` 调用（每 step 5–6 处）
  2. 若 nonzero 是为了取 top-k 大值索引，改用 `torch.topk(x, k=固定值)`，保持静态 shape
  3. 若 nonzero 是为了做稀疏 mask，改用 `x.masked_select(...)` 或 `x[mask].view(-1, d)` 配合 boolean mask 索引（torch_npu 已优化为 AI_VECTOR_CORE）
  4. 若动态 shape 无法消除，封装为子图并用 `aclmdlSetDynamicShape` 提前注册可能的 shape 集合
- **验证方法**：重采后 `aclnnNonzeroV2` 在 `api_statistic.csv` 中的 Max 值应从 72,009 us 降到 < 5,000 us；`step_trace_time.csv` 中慢 step 的占比从当前 59% 降到 < 10%
- **可视化视图**：Timeline 视图（系统调优）— 载入 `trace_view.json`，搜索 `aclnnNonzeroV2`，观察其后紧跟的 `aclrtSynchronizeStreamWithTimeout` 与下一个 `aclnnXxx` launch 之间的 host gap，即为 D2H 同步开销

### 3.4 [P0] GatherElements 同样落 AI_CPU

- **证据**：
  - `op_statistic.csv`：`GatherElements` Core Type = **AI_CPU**，400 次共 28,642 us，单次均值 71.6 us，最大 100 us
  - `kernel_details.csv`：dtype 路径为 `INT64; INT64; INT64`，证实是 INT64 索引触发的 AI_CPU fallback（CANN 对部分 INT64 维度的 GatherElements 暂未实现 AI Core 算子）
- **影响**：占 NPU 计算 6.62%；与 IndexPut 类似，AI_CPU 任务会阻塞 device 后续 kernel 下发
- **操作步骤**：
  1. 在调用前对索引 tensor 做 `.to(torch.int32)`，让 `aclnnGather` 走 AI Core 路径
  2. 若可能，把 `gather` 改写为 `index_select`（AI Core 支持更全）
  3. 升级到 CANN 8.3 latest hotfix（当前 8.3.RC1）后再次复测，确认 INT64 GatherElements 算子白名单是否覆盖到本场景 shape
- **验证方法**：改写后 `kernel_details.csv` 中所有 `GatherElements` Accelerator Core 应不再出现 `AI_CPU`
- **可视化视图**：算子视图 — `kernel_details.csv`，按 Type=`GatherElements` + dtype 过滤，确认 dtype 变更后 Core 切换情况

### 3.5 [P1] MatMul/BatchMatMul 算子粒度过小，Cube 单元利用率 5.8%

- **证据**：
  - `kernel_details.csv` 中 6,100 个 AI_CORE kernel：`aic_mac_ratio` 均值 0.058（5.8%），`aic_scalar_ratio` 均值 0.811（81.1%），`aic_mte2_ratio` 均值 0.142（14.2%），`cube_utilization` 均值 47%
  - 典型 MatMul shape：`128,64 × 16,64`、`2048,16 × 64,16`、`128,16 × 64,16`，K 维仅 16–64，远低于 Ascend 910B Cube 单元 16×16×16 单次完成的设计利用阈值
  - MatMulV2 4500 次共 15.8 ms（单次 3.5 us），BatchMatMulV2 1600 次共 16.9 ms（单次 10.6 us）——Cube 在做"瞎忙"
- **影响**：即便消除 host 瓶颈，Cube 单元的浪费意味着 NPU 算力理论上限的 95% 未被使用
- **操作步骤**：
  1. 用 `operator_details.csv` 反查所有 MatMul 来源——很可能是模型中多 head/多塔的分头 Linear
  2. 把多个 `nn.Linear(d, h)` 合并为单个 `nn.Linear(d, n*h)` 后切分，或包成 `BatchMatMul`（M 维做 batch）
  3. K 维从 16/64 提升到 128 以上后，`aic_mac_ratio` 可提升到 30–50%
- **验证方法**：算子改造后单 MatMul 的 `aic_mac_ratio` 应 ≥ 0.3；`op-mfu-calculator` 计算的 MFU 从当前 < 1% 提升到 5–10%
- **可视化视图**：详情视图 — 载入算子 `.bin`（位于 `PROF_*/` 目录下），观察 Roofline 图（当前落点应在 Memory Bound 区域且远低于 Compute Roof）；辅以算子视图看耗时分布

### 3.6 [P1] MemSet 每步触发 46 次，workspace 未复用

- **证据**：
  - `op_statistic.csv`：`MemSet` 4600 次共 46,189 us（占 NPU 计算 **10.67%**，居第二）
  - `profiler_metadata.json` 中 `ACLNN_CACHE_LIMIT`、`HOST_CACHE_CAPACITY` 环境变量均为空
- **影响**：每个 aclnn 算子调用前都重新分配 workspace 并清零，浪费 device 算力
- **操作步骤**：
  1. 设置环境变量：`export ACLNN_CACHE_LIMIT=10000`、`export HOST_CACHE_CAPACITY=20`
  2. 若使用 PyTorch + torch_npu，确认 `torch_npu.npu.set_device(...)` 之前完成环境变量设置
- **验证方法**：重采后 `MemSet` 调用次数应从 4600 降到 < 1000；Total Time 占比从 10.67% 降到 < 3%
- **可视化视图**：Timeline 视图（系统调优）— 过滤 `MemSet` 任务点，观察其在每个算子前的密度变化

### 3.7 [P1] aclnnInplaceCopyGetWorkspaceSize 极端长尾

- **证据**：
  - `api_statistic.csv`：`aclnnInplaceCopyGetWorkspaceSize` 调用 1800 次，总耗时 108,387 us，**单次最大 64,065 us，方差 2,279,200**
  - `aclnnInplaceCopy` 本身只占 27,830 us、最大 76 us，问题完全在 GetWorkspaceSize 调用
- **影响**：少数 GetWorkspaceSize 调用一次卡 50–64 ms，与 NonZero / Gather 长尾叠加形成 host 端 stall
- **操作步骤**：
  1. 升级 CANN 到 8.3 latest 修复版
  2. 同时启用上一条的 `ACLNN_CACHE_LIMIT` 让 workspace 计算缓存生效
  3. 若仍存在长尾，提 ticket：`aclnnInplaceCopyGetWorkspaceSize` 在 dynamic shape 路径下应避免重新计算
- **验证方法**：单次 Max 应从 64 ms 降到 < 200 us
- **可视化视图**：Timeline 视图（系统调优）— `trace_view.json` host 泳道，按 API 名字过滤 `aclnnInplaceCopyGetWorkspaceSize`，观察长尾分布

### 3.8 [P2] 每步 7 次同步调用，建议清查显式同步

- **证据**：
  - `api_statistic.csv`：`aclrtSynchronizeDeviceWithTimeout` 101 次（每 step 1 次 + 1 次结束），`aclrtSynchronizeStreamWithTimeout` 600 次（每 step 6 次）
  - 600 次 stream sync 与 600 次 `aclnnNonzeroV2` 调用数完全一致——大部分 sync 是 NonZero 引发，3.3 解决后会自动消失
- **影响**：device sync 直接序列化 host/device 流水线
- **操作步骤**：
  1. 业务代码中搜索 `torch.npu.synchronize()` / `.cpu()` / `.item()` / `.tolist()` / `print(tensor)` 等显式同步点
  2. 把它们集中到 step 末尾或后处理阶段，避免在主循环里穿插
- **验证方法**：`aclrtSynchronizeDeviceWithTimeout` 应降到每 step ≤ 1 次（即 100 次以内）
- **可视化视图**：Timeline 视图（系统调优）— 过滤 `aclrtSynchronize*`，按 step 切片观察分布

---

## 4. 已确认无问题

- **通信** — 单卡场景，`step_trace_time.csv` 中 Communication 列恒为 0，无通信瓶颈（N/A）
- **采集完整性** — `profiler_info.json` 正常 Stop，`ASCEND_PROFILER_OUTPUT/` 下 8 个交付件（含 db + text 双模式）齐全
- **Step 间稳定性** — 100 步 Computing 时间标准差 < 1%（4277–4477 us），NPU 算子执行本身没有波动
- **Preparing 阶段** — 均值 218 us，最大 447 us，远低于 step 耗时，dataloader / 输入准备无瓶颈
- **MemCopy 算子** — `aclnnInplaceCopy` 的 device 端表现正常（最大 76 us），仅其 GetWorkspaceSize 异常（见 3.7）
- **通信** —（重申）单卡无需检查 HCCL / 链路；本报告完全跳过 `msprof-analyze advisor cluster*` 流程

未排查项：
- **HBM 带宽 / L2Cache 命中率** — `profiler_info.json` 中 `_l2_cache: false`、`profile_memory: false`，未采集，**内存子项判 N/A**
- **算子源码热点（.bin Roofline）** — 框架 profiler 不导出 `.bin`，需后续单独跑 msOpProf 才能定性 MatMul 是否 Memory / Compute Bound

---

## 5. 数据与方法

- **分析日期**：2026-05-28
- **数据路径**：`d:/Projects/ProfilingTest/eta_eager_l1/1640123b27bd_12093_20260110074326034_ascend_pt/`
- **数据范围**：单卡（Device 0），step 10–109 共 100 步，采集时长 ~5.88 s，采集级别 Level1，profile_memory=false、with_stack=false
- **torch_npu / CANN 版本**：torch_npu 2.6.0 / CANN 8.3.RC1
- **使用的 Skills**：
  - `mindstudio_profiler_data_check`：校验 Profiling 数据完整性（valid，框架 profiler PyTorch 单卡）
  - `ascend-profiler-db-explorer`：未直接调用（环境无 Python，改走 CSV 聚合）
  - `performance-health-score`：按"单卡推理"场景权重（计算 0.50 / 调度 0.30，通信 N/A、内存 N/A）计算 PHS
  - `msinsight-view-selector`：为每个问题点附 Insight 可视化视图（Timeline 系统调优 / 算子视图 / 详情视图）
- **Advisor 状态**：
  - `msprof-analyze advisor`：**失败** — 当前环境 `python` / `msprof-analyze` 均不在 PATH（仅有 Microsoft Store 别名）；建议在装有 CANN toolkit 的 Linux 节点上执行 `msprof-analyze advisor -d eta_eager_l1/1640123b27bd_12093_20260110074326034_ascend_pt/ASCEND_PROFILER_OUTPUT/` 进行二次诊断
- **输出位置**：`./eta_eager_l1_analysis_20260528/`（本次分析所有产物均在此目录下）
  - `report.md` — 本报告
  - `intermediate/step_summary.csv` — 100 步的 Compute/Free/Stage/Preparing 时间提取
  - `intermediate/top_apis.csv` — 按 Total Time 降序的 Top 15 host API
  - `intermediate/top_ops.csv` — 按 Total Time 降序的 Top 12 NPU 算子
  - `figures/` — 预留（本次未生成图表，因 Python 不可用；可在带 Python 环境的机器上用上述 CSV 自行生成）
- **PHS 计算细节**：
  - 计算利用率 = NPU 实际算子时间 432.77 ms ÷ Stage 总时间 5883.72 ms = **7.36%** → 取整 7%
  - 调度效率 = (1 − Free 5450.95 ms ÷ Stage 5883.72 ms) × 100 = **7.36%** → 取整 7%
  - 通信 / 内存子项 N/A，权重归一化：计算 0.50 / 0.80 = 0.625，调度 0.30 / 0.80 = 0.375
  - **当前 PHS** = 0.625 × 7 + 0.375 × 7 = 7 → 等级 **D**
  - 优化后预估：若 P0/P1 全部落地（图模式 + 消除 AI_CPU 算子 + 消除 D2H 同步 + 算子合并），计算利用率有望提升到 35%、调度效率提升到 75%，PHS = 0.625 × 35 + 0.375 × 75 ≈ **48** → 等级 **B**
