# 昇腾超节点监控可视化 · `ingest` 契约数据源清单

> 目标：为逐-rank 遥测可视化（CubeView 状态色）确定"数据从哪来、什么格式、字段叫什么、能否拿到公开样例"。
> 覆盖三类输入：① 逐-rank 时序遥测 ② 作业并行配置 ③ rank→物理节点放置。
> 日期：2026-07-08

---

## 0. 结论先行

三类输入没有单一数据源，需要"拼装"：

| 类别 | 主数据源（优先落地） | 格式 | 公开样例 |
|---|---|---|---|
| ① 逐-rank 时序遥测（利用率/HBM/通信字节/掉队/时间戳） | `npu-smi info` + CANN `msprof` timeline / `op_summary.csv` + HCCL trace | CSV / JSON(Chrome-trace) / 文本 | ✅ 部分（见 §2、§5） |
| ② 并行配置 TP/PP/EP/DP/VPP/CP | MindSpeed/Megatron `parallel_state` + 训练日志 + `config.json` | Python 运行时 / 文本日志 / JSON | ✅（配置文件公开） |
| ③ rank→物理放置 | `rank_table_file.json`（HCCL rank table）+ `hostfile` | JSON / 文本 | ✅（格式文档 + 示例公开） |

**关键发现**：你参考的 `compute-graph-viewer`（PTO 工作台）里已经有一套成型的 ingest 契约，直接照抄字段名即可（见 §1）。它的 swimlane 同时吃两种格式，其中一种就是 CANN `msprof` 的 Chrome-tracing timeline。

---

## 1. 参考基准：PTO / swimlane 的真实 ingest 契约

来源：`github.com/yinyucheng0601/compute-graph-viewer`（已 clone 逐文件核对，非推测）。

### 1.1 格式 A — CoreTask（`swimlane/data.js` 头部注释 + `samples/*.json`）

泳道主格式，一个数组，每元素是一条"核泳道"，内含该核上的任务序列：

```jsonc
[
  {
    "blockIdx": 0,               // 硬件 block 索引
    "coreType": "AIC_CoreMachine0", // 核类型：Fake Core / AIC / AIV / AICPU
    "tasks": [
      {
        "taskId": 23,            // taskId 编码 = stitchedStatic<<32 | rootIndex<<20 | opIndex
        "subGraphId": 4,         // 所属同构子图 ID
        "execStart": 69128,      // 开始时间（μs）
        "execEnd": 82390,        // 结束时间（μs）
        "semanticLabel": "fake", // 语义标签（如 Query-Linear / Prolog-Quant）
        "taskName": "[Stitch 0] 0-0-5-82-2(fake)"
      }
    ]
  }
]
```

真实样例文件：`swimlane/samples/stitched_before.json`（含流水气泡的实采 profile）、`stitched_after.json`（消气泡理想态）、`open-source-swim.json`。

### 1.2 格式 B — Chrome tracing / msprof timeline（`swimlane/app.js` L731+）

`app.js` 解析 `raw.traceEvents`，逐字段读取——**这正是 CANN `msprof` 导出的 timeline JSON 格式**，字段对齐 Chrome Trace Event Format：

| 字段 | 含义 | app.js 用途 |
|---|---|---|
| `ph:"X"` | complete event（有 dur） | 只收 `ph==='X'` 的算子事件 |
| `ts` | 时间戳（μs） | 泳道起点 |
| `dur` | 持续时间（μs） | 泳道条长度 |
| `pid` / `tid` | 进程 / 线程（≈ device / core） | 泳道分组 |
| `name` | 算子名 | 提取 label |
| `args.color / seqNo / taskId / subGraphId` | 扩展语义 | 着色与下钻 |
| `args['event-hint'] / ['ioperand-hint'] / ['execution-hint']` | 输入输出/执行提示 | tooltip |
| `name:"process_name"/"thread_name"` 的 metadata event | pid/tid 命名 | 泳道命名 |

### 1.3 训练透视契约（`pangu-moe-trainviz/`）—— 与你的三类需求几乎 1:1

这个模块已经把"逐-rank × 时间 × 并行"建模好了，是最值得照抄的：

- **逐-rank 时序**：`data/timeseries.js` → `window.TS_DATA`，字段 `steps[]`、`series{train_loss,val_loss,eval_mmlu,grad_norm,load_balance_loss}`、`anomalies`、`faultStep/collapseStep`、`config{DP,PP,TP,EP,lr,batch,seq,precision}`。
- **1F1B rank×time trace**：`data/strict-1f1b-trace-sim.json`
  ```jsonc
  {
    "schema": "pto.strict-1f1b-trace-sim.v2",
    "fidelity": "schedule-simulated-not-profiler-measured",  // 诚实标注：仿真 vs 实测
    "config": { "dp":2,"pp":4,"tp":2,"ep":2,"cp":1,"vpp":1,"microbatches":8,
                "rankFormula": "rank = (((dp*PP + pp)*TP + tp)*EP + ep)",  // ★ 类别③放置公式
                "stageRanges": {"0":[0,12],"1":[13,25],"2":[26,37],"3":[38,49]} },
    "lanes": [ {"id":"d0p0","dp":0,"stage":0,"label":"D0·PP0","rankRange":"0-3"} ],
    "ticks": [ {"tick":19,"region":"steady","stages":{
        "0":{"phase":"B|F|bubble","micro":0,"layerRange":[0,12],
             "opFocus":{"layer":6,"step":"moe_prenorm"},"comm":"pp|tp|ep|dp",
             "dependsOn":[...],"produces":[...],"explain":"..."} }} ]
  }
  ```
- **掉队/故障建模**：`straggler` 用 `phase:"bubble"` + `region:"warmup/steady/drain"` 表达；单卡故障 vs collective 参与面用 `comm` 字段区分（避免把所有参与 rank 判成坏卡）。

> 建议：你的 `ingest` 契约直接沿用 `schema` 版本串 + `fidelity` 字段 + `rankFormula`，把"实测/仿真"标注固化进契约，CubeView 状态色只从这里取。

---

## 2. 类别① 逐-rank 时序遥测

### 2.1 `npu-smi info` —— 最快落地的实时指标（每卡）

```bash
npu-smi info                      # 快照
npu-smi info watch -i <id> -d 1   # 周期采样（1s）
```

关键字段（可直接映射 CubeView 状态色）：`AICore(%)`、`Memory-Usage(MB)`、`HBM-Usage(MB)`、`Power(W)`、`Temp(C)`、`Health`、`Chip`、`Bus-Id`。
- 导出样例：`npu-smi info watch` 定时抓取重定向成 CSV / 自己包一层脚本加 `timestamp` + `rank_id`。
- 公开样例：✅ 输出格式在华为 Atlas 用户指南公开（见 Sources）。

### 2.2 CANN `msprof` —— 算子级 timeline + 掉队分析（每 rank 一份）

```bash
msprof --application="python train.py" --output=./prof
# 产物目录：PROF_{num}_{ts}_{str}/mindstudio_profiler_output/
```

关键产物与字段：

| 文件 | 关键字段 | 用途 |
|---|---|---|
| `msprof_*.json`（timeline） | `ph/ts/dur/pid/tid/name/args`（Chrome-trace） | 直接喂 §1.2 格式 B |
| `op_summary_*.csv` | `Op Name`,`Op Type`,`Task Duration(us)`,`aiv_time(us)`,`aiv_mte2_time(us)`,`aiv_vec_time(us)`,`aiv_scalar_time(us)`,`aiv_mte3(us)`,`aicore_time(us)` | 算子耗时、pipe 占用 → 掉队定位 |
| `step_trace_*.csv` | 迭代 step 边界、fp/bp/allreduce 耗时 | 每 step 掉队信号 |
| `communication.json` / `communication_matrix.json` | 通信算子耗时、通信量（bytes）、band width | **通信字节数**来源 |

- AICore 指标档位：`AicoreMetrics.PipeUtilization`（Level1 默认）。
- 掉队(straggler)信号：对同一 step，各 rank 的 `step_trace` 计算耗时取 `max-min`（对应论文 Imbalance Score 思路），或 `communication.json` 里的等待时间。

### 2.3 MindSpore / MindSpeed / MindFormers profiling

- MindSpore：`mindspore.Profiler(...)` → 同样落 `mindstudio_profiler_output`（op_summary / step_trace / timeline 同构）。
- MindSpeed / MindFormers（训练栈）：底层仍走 CANN profiling，产物一致；训练脚本里开 profiler 即可拿到逐-rank 目录（每个 rank 一个 `device_{id}` 子目录）。

### 2.4 HCCL 通信日志 / trace

- 环境变量：`HCCL_ENTRY_LOG_ENABLE`、`ASCEND_GLOBAL_LOG_LEVEL` 打开通信算子日志。
- `communication.json`（msprof 产物）给结构化通信量/耗时，比裸日志好解析——**优先用它做"通信字节数"和 A2A/AllReduce 可视化**。

---

## 3. 类别② 作业并行配置（TP/PP/EP/DP/VPP/CP）

### 3.1 运行时读取（MindSpeed / Megatron `parallel_state`）

MindSpeed 复用 Megatron 的 `megatron.core.parallel_state`，逐维度有 getter：

```python
from megatron.core import parallel_state as ps
ps.get_tensor_model_parallel_world_size()      # TP
ps.get_pipeline_model_parallel_world_size()    # PP
ps.get_expert_model_parallel_world_size()      # EP
ps.get_data_parallel_world_size()              # DP
ps.get_context_parallel_world_size()           # CP
# VPP: args.virtual_pipeline_model_parallel_size
ps.get_tensor_model_parallel_rank() / _pipeline_.._rank() ...  # 本 rank 各维坐标
```

### 3.2 静态读取（落地更简单）

- 训练启动命令 / 日志：`--tensor-model-parallel-size 8 --pipeline-model-parallel-size 5 --expert-model-parallel-size 2 --context-parallel-size 1 --num-layers-per-virtual-pipeline-stage ...`
- 模型 `config.json`（如 openPangu-R-72B）：`num_hidden_layers`,`num_experts`,`num_experts_per_tok`,`mlp_only_layers` 等（结构口径）。
- 盘古 Pro MoE 论文实测口径（可直接写死做 demo）：`TP=8, EP=2, PP=5, VPP=5, CP=1`；`DP` 论文未直接披露，按 4K 卡近似推导 `DP≈50`（须标注 inferred，见你的可视化输入规格 errata 风格）。

---

## 4. 类别③ rank→物理节点放置

### 4.1 `rank_table_file.json`（HCCL rank table）—— 权威放置来源

CANN/HCCL 集群拓扑文件，描述每张卡在哪台 server、哪个 device：

```jsonc
{
  "version": "1.0",
  "server_count": "1",
  "server_list": [
    {
      "server_id": "10.0.0.1",          // ★ 物理节点(server) 标识
      "device": [
        { "device_id": "0",             // ★ 节点内本地卡号
          "device_ip": "192.1.1.1",
          "rank_id": "0" },             // ★ global rank ↔ 物理卡 的映射
        { "device_id": "1", "device_ip": "192.1.1.2", "rank_id": "1" }
      ],
      "host_nic_ip": "reserve"
    }
  ],
  "status": "completed"
}
```

- `rank_id`：全局唯一，从 0 起，按"先节点、后节点内卡号"顺序编号 → 天然给出 rank→(server, device) 映射。
- 环境变量 `RANK_TABLE_FILE` 指向该文件。
- 公开样例：✅ MindIE-LLM / MindSpore rank_table 教程有 2/8 卡完整示例。

### 4.2 `hostfile`（多机）

`ip slots=8` 形式，配合启动器给出 node→卡数；与 rank_table 二选一或并用。

### 4.3 逻辑 rank → 物理 rank 公式

来自 PTO 契约（§1.3）：`rank = (((dp*PP + pp)*TP + tp)*EP + ep)`。
用它把并行坐标反算成 global rank，再经 rank_table 落到 (server_id, device_id)。**这条链就是你 CubeView 每个 cube 定位的核心。**

---

## 5. 落地映射表：真实源字段 → 你的 `ingest` 契约

| 你的契约字段（建议） | 真实来源 | 源字段 |
|---|---|---|
| `ts` | msprof timeline / npu-smi watch | `ts` / 采样时刻 |
| `rank_id` | rank_table / 训练框架 | `rank_id` |
| `util_aicore` | npu-smi | `AICore(%)` |
| `hbm_used_mb` | npu-smi | `HBM-Usage(MB)` |
| `op_dur_us` | op_summary.csv | `Task Duration(us)` |
| `comm_bytes` | communication.json | 通信量字段 |
| `straggler` | step_trace 跨 rank `max-min` | 派生 |
| `phase` | 1F1B trace | `F/B/bubble` |
| `placement{server,device}` | rank_table | `server_id`,`device_id` |
| `parallel{tp,pp,ep,dp,cp,vpp}` | parallel_state / 启动参数 | 各 world_size |
| `fidelity` | 自定义 | `measured` / `simulated` |

**优先级（按落地难度）**：`npu-smi`（最易，实时状态色）→ `rank_table_file`（放置）→ 启动参数（并行配置）→ `msprof op_summary/timeline`（算子级，最细但需开 profiling）。

---

## 6. 公开样例数据可得性

| 数据源 | 公开样例 | 位置 |
|---|---|---|
| swimlane CoreTask / trace JSON | ✅ | `compute-graph-viewer/swimlane/samples/*.json` |
| 1F1B rank×time trace | ✅ | `compute-graph-viewer/pangu-moe-trainviz/data/strict-1f1b-trace-sim.json` |
| 逐-rank 训练时序 | ✅（合成但字段真实） | 同上 `data/timeseries.js` |
| rank_table_file.json | ✅ | MindIE-LLM / MindSpore 教程示例 |
| npu-smi 输出格式 | ✅ | 华为 Atlas 用户指南 |
| op_summary.csv 字段 | ⚠️ 部分 | MindSpore Profiler 文档 / CANN 开发工具指南（需查原文档拿完整列名） |
| 真实大规模多-rank profiling 数据集 | ❌ 未见公开 | 需自采（开 profiler 跑一次小 job） |

> 空缺项建议：拿一台/一机 8 卡开 `msprof` 跑几十 step，导出一份真实 `op_summary.csv` + `communication.json` + `rank_table.json`，即可把契约对着真实字段定死。

---

## Sources

- [compute-graph-viewer (PTO 工作台)](https://github.com/yinyucheng0601/compute-graph-viewer) — swimlane/pangu-moe-trainviz 契约（逐文件核对）
- [Ascend rank_table_file 配置 (MindIE-LLM)](https://github.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/user_manual/rank_table_file_configuration.md)
- [rank table Startup (MindSpore 教程)](https://www.mindspore.cn/tutorials/en/r2.7.0/parallel/rank_table.html)
- [Performance Profiling (Ascend) — MindSpore Insight](https://www.mindspore.cn/mindinsight/docs/en/master/performance_profiling_ascend.html)
- [mindspore.Profiler API](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.Profiler.html)
- [msProf 算子调试 (triton-ascend)](https://github.com/Ascend/triton-ascend/blob/master/docs/sources/mindstudio-guide/01-msProf_op.md)
- [npu-smi 命令介绍 (Atlas 用户指南)](https://support.huawei.com/enterprise/en/doc/EDOC1100079295/c8f5b2f7/introduction-to-the-npu-smi-command-for-version-1011-1015)
- [Ascend DCMI API Reference](https://support.huawei.com/enterprise/en/doc/EDOC1100288463)
- [Pangu Pro MoE 论文 (arXiv 2505.21411)](https://arxiv.org/abs/2505.21411) — 并行配置 TP8/EP2/PP5/VPP5/CP1
