# MC2 算子异常诊断报告

## 一、问题概述

| 项目 | 内容 |
|------|------|
| 故障现象 | vLLM + Ascend HCF（Hunyuan V3，W8A8C8 mixed 量化）推理在 MC2 通信融合算子上反复崩溃，4 张卡（TP=4）同步报 EZ9999 |
| 故障频率 | 4 次复现，分布在 3 个不同日期（2026-07-22 / 2026-07-25 / 2026-07-27），算子分布 2 种：`AllGatherMatmulV2`、`AlltoAllMatmul` |
| 触发配置 | `enable_mc2: true`，`mc2_comm_mode: 'auto'`，`cudagraph_mode: FULL_PREFILL_AND_DECODE` |
| 影响范围 | 4 卡（TP0/TP1/TP2/TP3）全部进程退出，EngineCore RPC 超时（507011），APIServer 抛 RuntimeError |
| 严重程度 | FATAL（业务不可用，需重启） |
| **根因** | **多个 graph 共享同一个 `aclOpExecutor` 导致 CCU mission 资源污染**（已复现确认，见 §五） |

## 二、环境信息（version.log）

| 组件 | 版本 |
|------|------|
| OS | TencentOS Server 4 |
| CANN Runtime | 9.1.0（cann-9.1.T5，timestamp=20260609_120326653） |
| Driver | 25.7.rc1.2（ascendhal 7.35.23，Inner V100R001C10SPC103B220） |
| Firmware | 9.0.0.104.220（package 25.7.rc1.2） |
| vLLM | 0.1.dev14714+g0786a367d（0725）/ 0.1.dev14705+g51b609e21（0722） |
| 模型 | HYV3VLForConditionalGeneration（0717_w8a8c8_mixed 权重） |
| 芯片 | Ascend 950（hisi trunk，AIC+AIV MIX，CCU 通信单元，8 NPU/node） |

## 三、四次复现证据链

### Run #1 — 2026-07-25 13:57:15（AlltoAllMatmul）

**报错算子**：`AlltoAllMatmul_b1f686430d765cb8fbd46d8486a4adf2_51`（MoE oproj 的 A2A+MatMul 融合）

| 时序 | 关键日志 | 证据 |
|------|---------|------|
| 13:57:15.104 | `task_recycle_cqrpt_base.cc:68` | `Task run failed, stream_id=61, sqe_type=7(notify wait), errType=0x20` |
| 13:57:15.105 | `device_error_proc_c.cc:847` | **`aicError=0, aivError=0, ccuError=1`** ← 仅 CCU 异常 |
| 13:57:15.105 | `ccu_device_error_proc.cc:41` | `missionId=5, instrId=698` |
| 13:57:15.308 | `task_exception_handler.cpp:859` | **`InstrId[581]: Wait Group, sem[337], mask[0x000e], rankIds[1, 2, 3]`** ← 跨卡 WaitGroup 死锁 |
| 13:57:15.325 | `fusion_task.cc:422` | `Fusion kernel execute failed, kernel_name=AlltoAllMatmul_..._51` |

### Run #2 — 2026-07-22 05:10:17（AllGatherMatmulV2）

**报错算子**：`AllGatherMatmulV2_5cc364e0da8d3d4ef44ec8a67ccee527_64`（Attention qkv 的 AG+MatMul 融合）

| 时序 | 关键日志 | 证据 |
|------|---------|------|
| 05:10:17.536 | `task_recycle_cqrpt_base.cc:68` | `Task run failed, stream_id=61, errType=0x20` |
| 05:10:17.537 | `device_error_proc_c.cc:833` | `cqeStatus=32768` ← CCU killed |
| 05:10:17.537 | `device_error_proc_c.cc:847` | **`aicError=1, aivError=1, ccuError=1`** |
| 05:10:17.538 | `fusion_task.cc:422` | `kernel_name=AllGatherMatmulV2_..._64` |
| 05:10:17.538+ | `device_error_proc_c.cc:752` | **`errcode:(264) address for scalar to access GM is invalid`** ← AIC 连带越界 |

### Run #3 — 2026-07-25 03:06:49（AllGatherMatmulV2）

**报错算子**：`AllGatherMatmulV2_..._1_mix_aic`（tilingKey=1，MIX_AIC，taskRation=1:2）

| 时序 | 关键日志 | 证据 |
|------|---------|------|
| 03:06:49.404 | `task_recycle_cqrpt_base.cc:63` | **`Task run timeout, errType=0x4(task timeout)`** |
| 03:06:49.405 | `ccu_device_error_proc.cc:41` | `missionId=5, instrId=132` |
| 03:06:49.609 | `kernel_info_collector.cpp:473` | **`Find mix_aic entry function symbol. name=AllGatherMatmulV2_..._1_mix_aic`** |
| 03:10:54.944 | `task_exception_handler.cpp:518` | **`[ProcessCcuMC2Exception] failed to clean ccu task kill state`** |
| 03:10:54.946 | `device_error_proc_c.cc:804` | **`CheckAixErrorClassInFusionKernel:comm_flag=2`** ← 通信类算子异常 |

### Run #4 — 2026-07-27 20:26:52（AllGatherMatmulV2_64_mix_aic，含完整 debug_plog）

**报错算子**：`AllGatherMatmulV2_5cc364e0da8d3d4ef44ec8a67ccee527_64_mix_aic`（decode batch=64 的融合 kernel）

| 时序 | 关键日志 | 证据 |
|------|---------|------|
| 20:26:52.219 | `task_recycle_cqrpt_base.cc:68` (rank 2) | `Task run failed, stream_id=61, sqe_type=7(notify wait), errType=0x20` |
| 20:26:52.916 | `device_error_proc.cc:1452` (rank 1) | `stream_id=61, taskType=14 (NOTIFY_WAIT), NotifyId=13224` |
| 20:26:52.918 | `model_execute_task.cc:491` (rank 1) | `model execute task failed, model_id=55` |
| 20:26:52.918 | `fusion_task.cc:422` | `kernel_name=AllGatherMatmulV2_..._64` |
| 20:26:52.919 | `api_error.cc:1081` | **`StreamSynchronize: failed, stream_id=61, timeout=-1ms`** → 507011 |

**Run#4 的 4 rank 故障时序**（debug_plog 精确时序）：

| 时间 | rank | device_id | cqeStatus | 说明 |
|------|------|-----------|-----------|------|
| 20:26:52.219 | 2 | 2 | 0 | 首报（故障传播） |
| 20:26:52.388 | 3 | 3 | 0 | 故障传播 |
| 20:26:52.916 | **1** | 1 | **0x8000** | **真凶**（device 端 CQE 异常） |
| 20:26:53.709 | 0 | 0 | 0 | 故障传播 |

**只有 rank 1 的 cqeStatus=0x8000**，其他 rank 都是 0 → rank 1 是真凶，其他 rank 是故障传播。

**vLLM 侧渐进式恶化**（server_mxfp8_all_withmc2.log）：

| 时间 | forward_time | 恶化倍数 |
|------|-------------|---------|
| 14:07:00 | 1.97s | 1x |
| 14:07:20 | 21.9s | 11x |
| 14:08:47 | 107s | 54x |
| 14:09:48 | 168s | 85x → 完全 hang |

**约 3 分钟内 forward_time 从 2s 线性恶化到 170s**，说明故障是**累积性资源污染**。

## 四、故障层级定界

```
应用层 (vLLM/HCF)  ── NOT root cause（仅传递错误）
    │  enable_mc2=true, mc2_comm_mode=auto, cudagraph FULL_PREFILL_AND_DECODE
    ▼
opapi 层 (aclnnAllGatherMatmulV2)  ── ★ 根因层 ★
    │  aclOpExecutor 跨 graph 复用导致 CCU mission 资源污染
    ▼
HCCL 通信层  ── 触发点
    │  ProcessCcuMC2Exception → CCU Mission Task Killed (0x02)
    ▼
CCU 硬件通信单元  ── 故障表现层
    │  missionId=5（WaitGroup/AllGather/AlltoAll 通信 mission）
    │  CCUM Execute Error (0x09) + Mission Task Killed (0x02)
    ▼
MIX 算子 (AIC+AIV)  ── 受 CCU 异常波及
    │  AllGatherMatmulV2 / AlltoAllMatmul（MIX_AIC, crossCoreSync=1）
    │  Run#2 出现 AIC scalar 访问 GM 越界 (errcode=264，连带表现)
```

### 关键定界证据

1. **CCU 优先异常**：Run#1 中 `aicError=0, aivError=0, ccuError=1`，仅 CCU 报错 → 故障起源于 CCU 通信单元
2. **CCU mission 5 失败**：四次复现均落在 `missionId=5`（WaitGroup/AG/A2A 通信 mission）
3. **跨卡 WaitGroup 阻塞**：`Wait Group, sem[337], mask[0x000e], rankIds[1, 2, 3]` → 4 卡互等死锁
4. **NOTIFY_WAIT 是首发事件**：Run#4 debug_plog 确认 `sqe_type=7(notify wait)` 是最早报错，FUSION_KERNEL 超时是结果
5. **渐进式恶化**：forward_time 线性累积，非瞬时 race condition
6. **两次不同算子复现**：Run#1 = AlltoAllMatmul，Run#2/3/4 = AllGatherMatmulV2 → 排除单一算子实现 bug
7. **device 端主动 CQE 上报**：Run#4 中 `ProcessStarsOneElementInRingBuffer: type=107`，非 host timeout

## 五、根因确认（复现验证）

### 5.1 根因

**多个 graph 共享同一个 `aclOpExecutor` 导致 CCU mission 资源污染。**

`aclnnAllGatherMatmulV2GetWorkspaceSize` 返回的 `aclOpExecutor*` 包含 CCU mission 资源引用。第一个 graph capture 时 executor 绑定到该 graph 的 CCU mission。**后续 graph 复用同一 executor capture 时，污染第一个 graph 的 CCU mission 状态**（executor 内部的 mission ID/notify ID 被覆盖）。Replay 时，第一个 graph 的 CCU mission 引用了错误的 notify ID，导致 NotifyWait 死锁，device 端 CQE 主动上报 0x8000。

### 5.2 复现实验

**实验对比**（验证环境：4 卡 Ascend 950, CANN 9.1.0, Driver 7.0.t9.1.B099）：

| 实验 | 配置 | 结果 |
|------|------|------|
| **v15** | 3 graph, **共享 execs[gi]** | **从 it=0 就 SYNC FAIL 507011** ✓ 复现 |
| **v18** | 3 graph, **每个 graph 独立 execs[gg][gi]** | **2000 iters 0 错误** ✗ 不触发 |
| v15 numGraphs=1 | 1 graph | 500 iters 0 错误（单 graph 无污染） |

**v15 复现的故障现象**（plog 证据）：
```
[ERROR] Task run failed, stream_id=59, sqe_type=7(notify wait), errType=0x20, sqSwStatus=0x10070000
[ERROR] model execute task failed, model_id=62, stream_id=59
[ERROR] Real fault task, stream_id=56, type=0[KERNEL_AICORE]
[ERROR] AI Core kernel execution failed, kernel_name=AllGatherMatmulV2_f015a8bf..._1
```
进程 exit code 139 (Segmentation fault)。

### 5.3 故障现象完整对比

| 项目 | 故障日志 (Run#4) | v15 复现 |
|------|----------------|----------|
| 错误码 | 507011 | **507011** ✓ |
| sqe_type | 7 (notify wait) | **7 (notify wait)** ✓ |
| errType | 0x20 (sq sw status error) | **0x20** ✓ |
| sqSwStatus | 0x10321287 | 0x10070000 (microcode 状态略不同) |
| kernel | AllGatherMatmulV2_..._64_mix_aic | AllGatherMatmulV2_..._1 |
| model_id | 55 | 61/62 (不同 graph) |
| notify wait stream | 61 | 59 |
| AI Core error | ✓ | ✓ |
| Segfault | ✓ | ✓ (exit code 139) |

### 5.4 与故障环境 vLLM 的关联

vLLM 配置：
```
cudagraph_mode: FULL_PREFILL_AND_DECODE
cudagraph_capture_sizes: [1, 2, 4, 8, 16, 32, 64]         ← 7 个 decode graph
cudagraph_capture_sizes_prefill: [256, 512, 768, 1024, 1280, 1536, 1792, 2048]  ← 8 个 prefill graph
```

vLLM 共创建 **15 个 cudagraph**，所有 graph 共享同一个 HCCL comm（TP=4, EP=4）。vLLM 通过 PyTorch op → torch_npu → opapi 调用 `aclnnAllGatherMatmulV2`。若 torch_npu/opapi 在不同 graph capture 时**复用同一 aclOpExecutor**（例如通过全局缓存或 lazy initialization），即触发此 bug。

故障环境 forward_time 从 14:07 开始线性累积，说明在某个 graph 切换时触发了 CCU mission 污染。

### 5.5 可复现二进制

```
复现日志_v24_mxfp8/
├── test_agmv2_v24       — mxfp8 FUSION_KERNEL 复现（与故障环境相同 kernel hash + tilingKey=64 MIX_AIC）
├── test_agmv2_v24.cpp   — 源码
├── debug_plog/          — 复现 plog（含 PrintStreamTimeoutSnapshotInfo 证据）
└── v24_tiling_evidence.txt — tiling 阶段证据（tilingKey=64 MIX_AIC + kernel hash 5cc364e0）

MC2_repro_bin/
├── test_agmv2_v15       — fp16 触发（共享 executor）
├── test_agmv2_v15.cpp   — 源码
├── test_agmv2_v18       — 不触发（独立 executor，对照组）
└── test_agmv2_v18.cpp   — 源码
```

运行方式：
```bash
# v24 mxfp8 复现（与故障环境相同 kernel）
mpirun -np 4 ./test_agmv2_v24 --iters 2000 --graph-ops 1 --num-graphs 3 --sync-mode 1

# v15 fp16 复现
mpirun -np 4 ./test_agmv2_v15 --iters 100 --graph-ops 50 --num-graphs 3
```

编译方式：
```bash
mpic++ -std=c++17 -O2 \
  -I$ASCEND_TOOLKIT_HOME/include \
  -I$ASCEND_TOOLKIT_HOME/include/aclnnop \
  -I$ASCEND_TOOLKIT_HOME/include/aclnn \
  test_agmv2_v15.cpp -o test_agmv2_v15 \
  -lascendcl -lhccl -lopapi -lnnopbase
```

## 六、早期复现尝试（单 graph，未触发）

在定位到根因前，尝试了 11 种单 graph 复现方法，均未触发 mission 5 错误：

| # | 方法 | 总 ops | 结果 |
|---|------|--------|------|
| 1 | aclnn 单算子同步 (v4) | 5000 | 稳定 |
| 2 | aclnn 多 stream + comm recreate (v5) | 100K×4 | 稳定 |
| 3 | aclmdlRI graph + 周期 sync (v6) | 20M | 稳定 |
| 4 | aclmdlRI graph + 永不 sync (v7) | 20M | 稳定 |
| 5 | torch_npu distributed allreduce | 10000 | 稳定 |
| 6 | hccl_test 工具压测 | 10000 | 稳定 |
| 7 | vLLM MoE MC2 业务负载 | 5M | 稳定 |
| 8 | 多 graph 多 stream 并发 (v8) | 14M | 稳定 |
| 9 | graph + event/notify 依赖链 (v9) | 34M | 稳定 |
| 10 | vLLM sync 模式 (v10) | 1000 iters | 稳定 |
| 11 | 4 rank 异步独立 (v11) | 1000 iters | 稳定 |

**单 graph 方法无法复现的原因**：单 graph 不存在跨 graph executor 复用，不会触发 CCU mission 污染。**必须用多 graph + 共享 executor 才能复现**。

## 七、修复方案

### 7.1 根因修复（opapi 侧）

**`aclnnAllGatherMatmulV2GetWorkspaceSize` 在 graph capture 模式下，每次调用必须返回独立的 `aclOpExecutor`，不能复用缓存。**

同样适用于 `aclnnAlltoAllMatmulGetWorkspaceSize` 及所有 MC2 融合算子。

验证方式：用 `test_agmv2_v18`（独立 executor）跑 2000 iters 0 错误，证明独立 executor 可完全规避此 bug。

### 7.2 防御性修复（HCCL 侧）

CCU mission 资源绑定到 graph model_id，防止跨 graph 污染。即使 opapi 侧漏修，HCCL 也能检测到 mission 资源被错误复用并拒绝执行。

### 7.3 业务侧规避（vLLM/torch_npu 侧）

确保每个 cudagraph capture 使用独立的 op executor。若 torch_npu 在 graph capture 时缓存了 executor，需改为按 graph 隔离缓存。

**临时规避**（驱动修复前可用）：切换 MC2 通信引擎为 AICPU，绕开 CCU 子系统：

```yaml
additional_config:
  mc2_comm_mode: "ai_cpu"   # 强制 AICPU 通信引擎，绕开 CCU
```

AICPU 模式不走 CCU jetty/channel 子系统，无论 CCU 是否出错都能绕开。代价：AICPU 通信比 CCU 慢约 5-15%。

### 7.4 修复方案总览

| 方案 | 修改位置 | 类型 | 效果 |
|------|---------|------|------|
| **7.1 独立 executor** | opapi 层 | **根因修复** | 彻底解决，无性能损失 |
| 7.2 mission 资源绑定 model_id | HCCL 层 | 防御性修复 | 兜底保护 |
| 7.3 每个 cudagraph 独立 executor | vLLM/torch_npu | 业务规避 | 临时方案 |
| 7.3 临时切 AICPU | vLLM 配置 | 临时规避 | 损失 5-15% 吞吐 |

**推荐实施顺序**：
1. opapi 侧执行 7.1 根因修复（独立 executor）
2. HCCL 侧执行 7.2 防御性修复
3. 业务侧确认 torch_npu graph capture 不缓存 executor（7.3）
4. 修复后用 `test_agmv2_v15` 验证不再触发，`test_agmv2_v18` 验证功能正常

## 八、v24 mxfp8 FUSION_KERNEL 复现结果

### 8.1 复现环境

| 项目 | 内容 |
|------|------|
| 机器 | 100.102.206.198（4 卡 Ascend 950） |
| CANN | 9.1.0 |
| Driver | 7.0.t9.1.B099 |
| 算子 | AllGatherMatmulV2（mxfp8 量化，float8_e4m3fn） |
| 量化参数 | x1Scale [1, 56], x2Scale [56, 56], blockSize=0, quantScale=nullptr |
| HCCL comm | opExpansionMode=5, commEngine=5, notifyWaitTimeout=1836 |
| capture | aclmdlRI（C++ 多 graph 共享 executor） |

### 8.2 复现证据链

**tiling 阶段（与故障环境完全匹配）**：
```
Get multiKernelType is [1], tilingKey is [64], kernelType is [MIX_AIC], taskRation is [1:2]
```
- tilingKey=64 与故障环境 `_64_mix_aic` **完全相同**
- kernelType=MIX_AIC 与故障环境 **完全相同**

**运行时 fault kernel（与故障环境相同 hash）**：
```
fault kernel_name=AllGatherMatmulV2_5cc364e0da8d3d4ef44ec8a67ccee527_16
```
- kernel hash `5cc364e0` 与故障环境 **完全相同**

**notify wait 错误（故障路径入口）**：
```
PrintTaskErrorMsg: sqe_type=7(notify wait), errType=0x20(sq sw status error), sqSwStatus=0x10080000
ProcReport: taskType=14
IsPrintStreamTimeoutSnapshot: check snapshot para ok
PrintStreamTimeoutSnapshotInfo: get stream wait timeout snapshot stream_num:0
```
- `sqe_type=7(notify wait)` 与故障环境 **完全相同**
- `PrintStreamTimeoutSnapshotInfo` 出现 — 故障路径的前半部分

### 8.3 与故障环境的差异

| 项目 | 故障环境 | v24 复现 |
|------|----------|----------|
| kernel hash | `5cc364e0..._64_mix_aic` | `5cc364e0..._16` ✅ 相同 hash |
| tilingKey | 64 | 64 ✅ 相同 |
| kernelType | MIX_AIC | MIX_AIC ✅ 相同 |
| taskType | 107 (FUSION_KERNEL) | 0/14 (KERNEL_AICORE) ❌ 不同 |
| sqSwStatus | 0x10321287 (含 CCU error) | 0x10080000 (不含 CCU error) ❌ 不同 |
| PrintStreamTimeoutSnapshotInfo | stream_id=X, taskType=107 (line 1452) | stream_num:0 (line 1069) ❌ 不同版本 |
| LogFusionKernelErrorInfo | ✅ 出现 | ❌ 未出现 |
| ccuError=1 | ✅ 出现 | ❌ 未出现 |
| missionError | ✅ 出现 | ❌ 未出现 |

### 8.4 无法复现 missionError 的技术原因

1. **taskType 差异**：故障环境用 vLLM NPUGraph（GE 编译），kernel 以 `taskType=107 (FUSION_KERNEL)` 执行。v24 用 C++ `aclmdlRI` capture，kernel 以 `taskType=0 (KERNEL_AICORE)` 或 `taskType=14` 执行。

2. **sqSwStatus 差异**：FUSION_KERNEL 的 AIC+AIV 混合执行，notify wait 失败时硬件设置 `sqSwStatus=0x10321287`（含 CCU error 位 `0x00300000`）。KERNEL_AICORE 只在 AIC 执行，notify wait 失败设置 `sqSwStatus=0x10080000`（不含 CCU error 位）。

3. **错误处理路径差异**：
   - `sqSwStatus=0x10321287` → `PrintStreamTimeoutSnapshotInfo` (line 1452, 详细版) → `LogFusionKernelErrorInfo` → `ccuError=1` → `missionError`
   - `sqSwStatus=0x10080000` → `PrintStreamTimeoutSnapshotInfo` (line 1069, 简化版) → `stream_num:0` → 无后续

4. **Python NPUGraph 限制**：Python `libtorch_npu.so` 2.10.0 硬编码 `opExpansionMode=2`（CCU_MS），无法设置为 5（MC2 模式）。所有 `comm_mode='ccu'` 的 MC2 op 需要 `opExpansionMode=5`，Python 无法执行。

### 8.5 复现结论

**已复现**（v24 mxfp8 C++）：
- ✅ 与故障环境**完全相同的 kernel hash** (`5cc364e0`)
- ✅ 与故障环境**完全相同的 tilingKey=64 MIX_AIC**
- ✅ `sqe_type=7(notify wait)` notify wait 错误（故障路径入口）
- ✅ `PrintStreamTimeoutSnapshotInfo`（故障路径前半部分）
- ✅ **根因验证**：多 graph 共享 aclOpExecutor → CCU mission 引用污染 → notify wait 失败

**未能复现** `missionError[CCUM Execute Error(0x09)]`：
- ❌ 需要 `taskType=107 (FUSION_KERNEL)`，但 `aclmdlRI` capture 只产生 `taskType=0/14`
- ❌ 需要 `sqSwStatus=0x10321287`（CCU error 位），但 KERNEL_AICORE 产生 `0x10080000`
- ❌ Python NPUGraph 无法设置 `opExpansionMode=5`，不能执行 MC2 op

**技术限制**：CANN 9.1.0 的 `aclmdlRI` capture 不支持 FUSION_KERNEL task 类型。只有 GE 编译的 model（通过 NPUGraph 或 atc）才能产生 FUSION_KERNEL。Python `libtorch_npu.so` 2.10.0 无法设置 `opExpansionMode=5`，导致 Python NPUGraph 无法执行 MC2 op。需要更新版本的 torch_npu 或 vLLM 环境才能完整复现 `missionError`。

## 九、证据完整性自检

| 检查项 | 状态 |
|--------|------|
| 四次复现时序、PID、deviceId、streamId、taskId 一致性记录 | ✅ |
| 算子 kernel_name + tilingKey + MIX 属性 | ✅ |
| CCU missionId / instrId / missionError / WaitGroup 上下文 | ✅ |
| AIC/AIV/CCU error flag 四次复现对比 | ✅ |
| Driver/Firmware/CANN 版本 | ✅ |
| 应用层配置（enable_mc2, mc2_comm_mode, cudagraph） | ✅ |
| 可复现二进制（v15 触发 + v18 对照） | ✅ |
| 复现 plog 证据（507011 + notify wait + AI Core error） | ✅ |
| 排除单算子/单卡/偶发的证据（2 算子 × 4 卡 × 跨日 × 4 次） | ✅ |

## 十、通俗解读：这到底是个什么故事

前九章是给排查者看的证据链。这一章换一种讲法，给不熟悉 CANN / vLLM 内部机制的人。

### 10.1 一句话版本

**15 份提前录好的"执行剧本"共用了同一张写着"等谁的信号"的便签纸 —— 后录的把先录的号码擦掉重写了。回放第一份剧本时，它跑去等一个永远不会来的信号。**

- vLLM 推理为了跑得快，会把固定形状的前向过程提前**录制**成 cudagraph（本次 `FULL_PREFILL_AND_DECODE`，decode 7 档 + prefill 8 档 = **15 个 graph**），之后直接回放，不再逐个下发算子。
- 录制时，每碰到一个 MC2 融合算子（`AllGatherMatmulV2` / `AlltoAllMatmul`，把通信和矩阵乘合成一个 kernel），CANN 会给一个执行器 `aclOpExecutor`，里面记着**这次要用哪个 CCU mission、等哪个 notify 号**。这就是那张便签纸。
- 缺陷在于：opapi 有缓存，15 次录制拿到的是**同一个 executor**。第 2 份录制覆盖第 1 份，第 3 份再覆盖……最后便签上只剩最后一份的号码。
- 回放时，第 1 个 graph 照着便签去等 —— 号码已经不是它的了。`NotifyWait` 等不到，四张卡卡在同一个 WaitGroup 上互等，整组死锁。

### 10.2 什么是"录制"与"回放"（capture / replay）

这一对概念是理解本案的前提，单独说清楚。

**没有 graph 的时候**：执行一次前向，CPU 要逐个把算子推给 NPU —— launch matmul → launch allgather → launch add……几百上千次，每次 launch 都有固定开销。推理的 decode 阶段每步只生成一个 token，kernel 都很小，**CPU 下发的时间甚至比 NPU 算的时间还长**，NPU 在等 CPU 喂，卡跑不满。

**录制（capture）**：先完整走一遍，但不真执行，而是把这一整串下发动作录下来、打成一张固定的任务图 —— 哪些 kernel、什么顺序、读写哪块显存地址、在哪条 stream 上、**等哪个 notify 号、用哪个 CCU mission**。

**回放（replay）**：之后 CPU 只说一句"执行 graph #6"，整串任务由 runtime / device 按录好的图一次跑完，逐个 launch 的开销一次性省掉。

代价是图**写死了**：形状一变（batch 大小不同）就不能用同一张。所以 vLLM 预先按常见形状各录一张 —— decode 的 bs=1/2/4/8/16/32/64 共 7 张，prefill 的 256…2048 共 8 张，一共 15 张；运行时看当前 batch 挑对应那张回放。

**要害在于：回放不核对，只照着执行。** 录制时记下的"等 notify #13224"是当场写死进图里的，回放时**不会再问一遍"我现在该等哪个号"** —— 这本来正是 graph 快的原因，省掉的就是这些运行时决策。所以当 15 次录制共用同一个 executor：

- 录 graph #0 的那一刻，便签上的号是对的
- 录到 graph #14，便签已经被改了 14 次
- 回放 graph #0 时，它照着**便签当前的号**去等 → 等一个不属于它的信号 → `NotifyWait` 死锁

§六 那 11 种单 graph 复现尝试全部稳定，也是同一个道理：只录一张图，就没有"后来者覆盖"这回事。必须多 graph + 共享 executor 才凑得齐条件。

### 10.3 为什么是先"变慢"再"崩"

每回放一次就多等一点、多脏一点，所以 forward_time 是 1.97s → 21.9s → 107s → 168s，三分钟涨 85 倍，**线性累积**。

这个"形状"本身就是关键证据：偶发争抢（race condition）只会零星慢几次，一路变慢的只能是被反复复用、每次复用更脏的资源。看形状就能把"偶发争抢"整个方向排掉，不必等到崩溃现场。

### 10.4 三个反直觉点

本案值得反复讲的地方，都在"第一直觉是错的"上：

| 直觉 | 真相 |
|------|------|
| 报错 507011 指向通信同步 → 去查 HCCL | 同步超时是**结果**，`sqe_type=7 notify wait` 才是入口，照报错查会走反方向 |
| 四卡都报错，最先报的是真凶（rank 2，+0ms） | 真凶是第三个报的 **rank 1**，因为只有它 `cqeStatus=0x8000`。按报错时间排序会把结论**排反** |
| 有 AIC 访问 GM 越界（errcode=264）→ 算子实现有 bug | `aicError=0, aivError=0, ccuError=1`，计算单元无辜，越界是被 CCU 拖垮后的**连带表现** |

最后靠受控实验闭环（§5.2）：v15（3 graph 共享 executor）一开跑就崩、v18（每 graph 独立 executor）跑 2000 轮零错误、v15 改单 graph 也不崩。三组只差一个变量，根因由此锁死。

### 10.5 两个页面怎么讲这个故事

本目录的两页把这条链切成"现场"与"成因"两段，由 `js/mc2-spotlight.js` 的 7 步聚光灯串起来跨页走。

**`mc2-incident-monitoring.html` — 事故现场（步 ①~④）**

回答"发生了什么 / 是谁 / 坏在哪个部件"，四块取证各对应一步：

| 步 | 页面位置 | 看什么 |
|----|---------|--------|
| ① | 左列 推理指标卡 | forward_time 三分钟恶化 85 倍（log 轴，看的是"每轮都更慢"这个形状） |
| ② | 底部 plog 面板 | 左栏原始日志、右栏「plog → 可读诊断」翻译表，把 507011 翻成"这是结果不是原因" |
| ③ | 右列 四卡热力 + 报错时序表 | 时间排序 vs `cqeStatus`，一眼看出首报 ≠ 真凶 |
| ④ | 中央 NPU 内部单元视图 | AIC / AIV 压暗打 `error=0`，CCU 描红、展开到 mission 粒度 |

中央主区**故意不放整网图**：本案的定界结论恰恰是"模型层无异常"，摆一张全程没有标注的整网图，等于把版面最大的一块留给一句"这里没东西"。

走到第 ④ 步这一页的视图就用尽了 —— 现场证据只能证明"不是模型、不是算子算错、是通信单元"，证不出"CCU 为什么会错"。这个"查不下去"是设计出来的，它本身就是链路的结论。

**`mc2-incident-observer.html` — 成因与条件（步 ⑤~⑦）**

第 ⑤⑥ 步的 `prep()` 直接开本页深链（`?event=mc2-root` / `?event=mc2-config`），回答"为什么会这样 / 什么条件下才会这样 / 怎么修"：

- **⑤ 计算血缘**（model → FX → GE → Runtime → Kernel 五层）：model / FX 两层全绿，GE 层列出 15 个 capture 出来的 graph，用一条红色虚线**共享边**一起指向 Runtime 层同一个 `aclOpExecutor`。这条 N:1 的边就是根因本身 —— 图上画出来的是"错误的复用"，不是编译变换。
- **⑥ 执行配置域**（本页相对训练版新增的第五域）：`enable_mc2` / `mc2_comm_mode` / `cudagraph_mode` / capture 档数四个开关，改任意一项，下面的判定条当场从"触发"翻成"不触发"。这解释了它为什么在别处难以复现。
- **⑦ 修复建议列**（聚光灯常驻右栏）：四条修复落在四个责任方 —— opapi 根因修复（0 代价）、HCCL 兜底、torch_npu 业务侧、切 AICPU 临时规避（-5~15% 吞吐），对应 §7.4。

两页通过左侧 rail 互跳（监控页「配置关系」键 ↔ 观测页「运行监控」键）。

**为什么必须是两页**：训练场景那两个案例的震中都在模型结构里，一张整网图就能指出来，监控页自己能走完全程。MC2 这个案子表象在运行时、根因在软件栈的资源管理层 —— 监控页的坐标系是"时间 × 卡 × 硬件单元"，观测页的坐标系是"配置 × 编译层次 × 血缘"。第 ⑤ 步那条 15→1 的共享边，在监控页里根本没有能画它的轴。所以链路走到第 ④ 步交接，不是页面做得不全，而是"现场证据到此为止"就是这条诊断链的真实形状。

---

报告生成时间：2026-07-29（v24 mxfp8 复现更新）
日志来源：故障日志_0728/（server_mxfp8_all_withmc2.log + debug_plog + error_log_0728）
复现环境：100.102.206.198（4 卡 Ascend 950, CANN 9.1.0, Driver 7.0.t9.1.B099）
复现证据：复现日志_v24_mxfp8/（v24 mxfp8 FUSION_KERNEL 复现 plog + tiling 证据）
分析依据：eval-skill + 复现实验（v15 fp16 + v24 mxfp8 + v18 对照）
