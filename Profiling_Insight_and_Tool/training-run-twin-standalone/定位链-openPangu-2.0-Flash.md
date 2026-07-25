# 训练精度问题定位链

> 从表象精度异常出发，逐层下钻直至根因。**迭代层后即分叉**：单卡/多卡复现是否一致，决定走「通信分支」还是「计算主干」。每一层有明确的**判据**决定下一步方向。

```
迭代层           WHEN     — 时间维：哪个 step
   │
   ├── 日志/plog诊断层（穿插）  TRANSLATE — 报错"说人话"：Device plog → Python 侧可读诊断 · Ascend C 内部名 → torch_npu 接口名
   │
   ├── 仅多卡异常 → 通信调度层  WHY(通信) — 卡间通信问题（失步 / 精度漂移 / EP 切分不匹配）
   │                  │
   │                  └── 模型层  WHERE    — 通信影响了哪一层（含并行切分校验）
   │                        │
   └── 单/多卡均异常 → 模型层   WHERE    — 空间维：模型哪一层
                     │
                  算子层   WHAT     — 计算维：哪个算子
                     │
                  张量层   WHICH    — 数值维：哪些元素/区间
                     │
                  infra层  CONTEXT  — 并行策略 / 硬件归属 / 错误扩散路径
                     │
             ┌── 熔断/预警层  GUARD   — 能否在 NaN 前拦截？（AMP scaler 衰减 / KL 散度 / grad_norm 突变）
             │     │
             └─────┤
                   ▼
               超参/代码层  FIX     — 具体改什么 + 止损时间线
```

---

## 1. 迭代层 — 锁定「哪个 step」

| 项目 | 内容 |
|------|------|
| **定位目标** | 精度异常首次出现（或突然恶化）的训练步 |
| **观测手段** | loss 曲线、acc/PPL 曲线、grad_norm 曲线，按 global_step 展开 |
| **判据** | loss spike / NaN / grad_norm 突变的 step 即为嫌疑步；若整条曲线平稳漂移，取漂移起始段 |
| **产出** | 嫌疑 step 列表（如 step 15320~15350） |

> ⚠️ 此处产生**关键分叉**：用同一数据/种子在单卡和多卡环境各跑一次嫌疑 step。
> - **单卡也能复现** → 计算问题，沿主干进入「模型层」
> - **仅多卡复现，单卡正常** → 通信问题，切入分支「通信调度层」

---

## 1.5 日志/plog诊断层（穿插）— 报错"说人话"

> 此层不改变定位方向，而是**穿插在迭代层之后的每一步中**，解决"报错看不懂、plog 有信息但 Python 侧不显示"的痛点。无论走通信分支还是计算主干，凡是遇到底层报错都应先过此层翻译。

| 项目 | 内容 |
|------|------|
| **定位目标** | 将 Device 侧 plog / Ascend C 底层报错翻译为 Python 侧可读的诊断信息 |
| **痛点** | 算子报错报的是 Ascend C 内部变量名/属性名（如 `ub_buf_overflow`、`L0C_size_mismatch`），用户从 `torch_npu` 接口调用时名字对不上；plog 里有有效信息但 Python 侧不显示，每次要手动 `grep error` |
| **观测手段** | `grep -i "error\|fault\|overflow\|mismatch" /var/log/npu/plog/plog*.log`；提取异常发生的 device_id、task_id、stream_id；匹配算子下发时间戳与 Python 侧 `torch_npu` 调用栈 |
| **判据** | plog 中存在 `[ERROR]` 行 → 提取该行的 `kernel_name`、`param_index`、`constraint_desc`；将 `kernel_name` 映射到 `torch_npu` 接口名（如 `aclnnMatmulV3` → `torch_npu.matmul` / `F.linear`） |
| **产出** | 翻译后的诊断：「参数 X（你在代码中调用的 `torch_npu.xxx` 的第 Y 个参数）的 shape=(A,B)，不满足约束 C，建议改为 (A',B')」+ 原始 plog 行引用 |

**Ascend C 内部名 → torch_npu 接口名映射示例**：

| Ascend C / plog 中的名字 | torch_npu / PyTorch 用户可见接口 |
|---|---|
| `aclnnMatmulV3` | `torch_npu.matmul` / `F.linear` / `nn.Linear.forward` |
| `aclnnSoftmaxV2` | `torch_npu.softmax` / `F.softmax` |
| `aclnnLayerNorm` | `torch_npu.layer_norm` / `F.layer_norm` |
| `hcom_all_to_all_v_` | `torch_npu.all_to_all` / `dist.all_to_all` |
| `hcom_allReduce_` | `dist.all_reduce` |
| `ub_buf_overflow` | 通常对应 `torch_npu` 某算子内部 UB 缓冲区不足 → 检查输入 shape 是否过大 |
| `L0C_size_mismatch` | Cube 单元 L0C 缓冲区与当前 tile shape 不匹配 → 检查 Matrix 维度是否对齐到 16/32 |

> **覆盖原声（CheckList Row 17~19）**：
> - Row 17：「算子报错信息经常不明确。有时plog里有有效信息，但Python侧不会直接显示，用户还要自己去查plog。如果能直接指出哪个参数的shape是什么、违反了什么约束，就能更快定位」
> - Row 18：「算子报错时一般报的是Ascend C里边的变量名或属性名，但我从torch_npu接口去用，名字和属性得想一想才能对上」
> - Row 19：「从24年DFX优化开始，训练报错日志还是一样不清楚；每次要去Device侧plog grep error，感觉可以做成直接把有效信息抛到训练日志」

## 2. 模型层 — 锁定「哪一层」

| 项目 | 内容 |
|------|------|
| **定位目标** | 嫌疑 step 内，异常发生在 Emb / Attn / FFN / MoE 中的哪一层（或哪些层） |
| **观测手段** | 层 × step 热力图：每层逐 step 的 grad_norm（反向追因）或激活值离群率 |
| **判据** | 热力图中对应 step 的纵向高亮带，定位到具体 layer index；Dense 层与 MoE 层分开标注 |
| **产出** | 问题层路径，如 `model.layers.5.self_attn` 或 `model.layers.42.mlp` |

---

## ── 分支：通信调度层 ──

> 从迭代层直接分叉。当单卡多次运行结果一致、多卡运行结果出现差异时，进入此分支。通信分支定位完毕后，仍需回到模型层确定通信问题影响了哪些层。

| 项目 | 内容 |
|------|------|
| **定位目标** | 判断通信问题类别：调度失步（all-reduce / all-to-all / p2p 时序错位）还是浮点累加舍入误差累积（精度对齐漂移） |
| **判据** | 单卡多次结果一致 + 多卡结果不一致 → 通信问题；单卡多次结果不一致 → 非通信问题（可能是随机性/dropout/数据差异），回到主干 |
| **观测手段** | NCCL Inspector 通信概览表（带宽/延迟异常）、通信算子 trace 时间线（看 send/recv 配对、barrier 等待）、各 rank 的 all-reduce/all-to-all 输入/输出 hash 对比 |
| **产出** | 通信异常类型 + 涉及的 rank 列表 + 通信原语（all-reduce / all-to-all / p2p） |

> **覆盖原声（CheckList Row 5）**：
> - Row 5：「通信场景会遇到各种算子性能异常。有时是某张卡故障导致同步异常，有时是负载不均，通常表现为某个通信算子耗时变长。我们会先排查算子下发和卡间互联，解决不了再找HCCL或算子同事。希望报错能像PyTorch一样直接反映问题和处理方法」

---

## 3. 算子层 — 锁定「哪个算子」

| 项目 | 内容 |
|------|------|
| **定位目标** | 在问题层内，定位到具体的异常算子（如 `q_b_proj`、`attn_scores`、`gate_up`、`router`） |
| **观测手段** | 层内算子误差瀑布图（横轴：按执行顺序排列的算子，纵轴：与 baseline 的误差），定位误差突变点 |
| **判据** | 误差在某个算子处出现量级跳跃（如 1e-6 → 1e-3），即为首害算子 |
| **产出** | 问题算子路径，如 `model.layers.5.self_attn.q_b_proj` |

## 4. 张量层 — 锁定「哪些元素/数值区间」

| 项目 | 内容 |
|------|------|
| **定位目标** | 在问题算子的输入/输出中，定位到哪些元素位置、哪些数值区间引入了偏差 |
| **观测手段** | dump 该算子的输入、激活、梯度张量，与 baseline（单卡正确结果或上一正常 step）逐元素对比；对 softmax、layernorm 等数值敏感算子，绘制散点图、直方图、TopK 误差表 |
| **判据** | 逐元素 diff 中绝对值最大的元素即为嫌疑元素；直方图中偏离 baseline 分布的区间即为嫌疑数值区间 |
| **产出** | 嫌疑元素索引 + 数值区间范围 + 偏差量级 |

## 5. infra层 — 锁定「并行策略 / 硬件」

| 项目 | 内容 |
|------|------|
| **定位目标** | 确定问题涉及的并行策略（PP/EP/TP/DP）和硬件范围（哪些 rank、哪些节点） |
| **观测手段** | 从 trace 中提取问题算子涉及的并行策略标签（pp/ep rank）、通信原语、所属 GPU 节点；结合通信分支结论交叉定位 |
| **判据** | 问题算子仅出现在特定 pp stage 或 ep rank → 局部硬件/策略问题；跨所有 rank 出现 → 全局算法/精度问题 |
| **产出** | 涉及的 rank 列表 + 并行策略维度 + 嫌疑硬件范围（如 node3 GPU 0~7） |

> **覆盖原声（CheckList Row 33）**：
> - Row 33：「512个rank某节点错误通过gather扩散到所有机器，不可能每台逐一比较，我们在代码里加打印写Python做逻辑比较。比如100个rank第8个节点最先出问题，现有诊断平台也诊断不到这么细；当前团队都需要手动按流程排查」

## 5.5 熔断/预警层 — 能否在 NaN 前拦截？

> 此层回答一个关键问题：**精度异常真的只能等到 loss NaN 了才知道吗？** 如果能在异常恶化到不可逆之前自动拦截，可以节省大量算力和排查时间。

| 项目 | 内容 |
|------|------|
| **定位目标** | 在训练运行过程中，实时监测可预警指标，在精度异常恶化到 NaN / loss 跑飞之前触发自动熔断 |
| **痛点** | 万卡训练突然 NaN 或 loss 跑飞，一小时内损失可能高达数万元；报错后服务不一定停，一直卡着跑日志占资源；需要人工盯着 loss 曲线等异常 |
| **可预警指标** | ① **AMP loss scale 持续衰减**：从初始值（如 65536）持续下降是 FP 溢出的早期信号——每次 scale 减半意味着发生了一次溢出。当 scale 降至初始值的 1/16 以下时，后续极易触发 NaN。② **grad_norm 趋势**：连续 N 个 step 的 grad_norm 波动 > 3σ，或出现 inf 值。③ **z-loss 缺失或异常**：若开启了 z-loss，其值突然归零或飙升说明 logits 进入异常区间。④ **KL Divergence vs 高精度基线**：若同时跑着 BF16 对照组，KL Divergence > 2 bits 预示精度退化 |
| **判据与动作** | 见下表 |
| **产出** | 熔断判定结果 + 触发指标 + 保留现场（checkpoint + 最后 N 步的 profiling 数据） |

| 预警级别 | 触发条件 | 自动动作 |
|---|---|---|
| 🟡 注意 | AMP scaler 降至初始值的 1/4（如 65536→16384） | 记录日志，通知 on-call，不中断训练 |
| 🟠 警告 | AMP scaler 降至初始值的 1/16（如 65536→4096）或 grad_norm 连续 50 step 翻倍 | **自动 dump 当前 step 的激活张量 + router logits**，发告警通知 |
| 🔴 熔断 | AMP scaler < 初始值的 1/32 或 grad_norm=inf 或 loss=NaN | **立即停训**，保存 checkpoint，释放 GPU 资源，通知值班人员 |

> **案例一的教训**：step 15000 起 AMP scaler 从 65536 开始衰减，到 step 15202 已降至 4096（警告级别），但无人监控。若在 step 15200 触发 🟠 警告并自动 dump，可提前捕获 router logits 超界的证据，甚至可能在 step 15202 触发 🔴 熔断避免 NaN，省下 203 步 × 万卡时费。

> **覆盖原声（CheckList Row 30, 32, 34）**：
> - Row 30：「TensorBoard监测到国产卡loss曲线跟NV偏离度超过阈值就局部熔断；然后在测试床上两端加载完全相同的权重跟输入，强制执行前向和反向传播；用自研Tensor Hook注入脚本，不破坏原有计算图，自动拦截每一层Transformer layer的输出；再逐层自动化比对两端tensor的余弦相似度和绝对误差，相似度低于0.999就锁定精度分叉在哪一层」
> - Row 32：「报错了服务不一定停，一直卡着跑日志，找错误要往上翻很久。程序报错应该停掉，不然占着资源别人排队」
> - Row 34：「几百B、近万亿参数的万卡训练，突然出现NaN、loss跑飞或吞吐掉得很厉害，如果能实时识别、定位根因、给出解决方案就很有用。万卡集群两块钱一卡时就是两万元一小时，十二小时搞不定就是二十四万没了」

## 6. 超参/代码层 — 锁定「改什么」

| 项目 | 内容 |
|------|------|
| **定位目标** | 将根因映射到具体可执行的修改项 |
| **观测手段** | 回溯：启动参数（学习率、warmup、batch_size、precision）、模型代码（算子实现、数值精度 cast）、通信业务代码（通信原语调用、同步点）、训练超参（dropout rate、weight decay）；结合熔断层预警指标判断修改紧急程度 |
| **判据** | 根据上游各层锁定的根因类型匹配修改项：数值溢出 → 调低 lr / 开 loss scaling / 改精度路径；通信失步 → 修复同步点 / 校验 EP 切分；专家路由倾斜 → 调整 load balance 策略 / 加 z-loss；报错不可读 → 接入 plog 翻译层 |
| **产出** | 具体修改项 + 修改文件/参数路径 + 验证方案 + 熔断规则更新（将本次根因的预警信号纳入熔断配置） |

> **覆盖原声（CheckList Row 12, 15, 16, 20, 21）**：
> - Row 12：「联动视图同时配套Expert System诊断能力……新手仍需要更直白的原因解释、操作建议和代码关联，避免看到专业指标后仍要转向文档理解」
> - Row 15：「MindStudio数据是够的，把所有能采集的数据可视化展示，唯一就是缺少初步分析和可优化方向，非常依赖人的经验，新手上手有门槛」
> - Row 16：「如果专家完全不懂profile分析、全靠AI给结果，短期不可靠。应该是有经验专家把经验提炼给AI；AI分析时不光给结果，还把要注意的分析方法同时交给专家，过程和结果结合起来才可信」
> - Row 20：「算子报 HCCL 通信错误或 corner ops 错误我们看不懂，只能知道框架哪块计算出错。一层层找框架负责人、算子接口或开发或看手册。定位好才提给算子开发，链路很长，短的两三天长的一周」
> - Row 21：「报错那句代码不一定是原因，可能上游算子结果 shape 不对 dtype 不一样或内存越界。从框架侧根据报错定位范围，逻辑没问题就逐行打印 shape dtype，看经过哪个算子后不对」


---

## 案例一：通信分支 — Router 数值溢出导致路由塌缩，同时触发 loss NaN 与 all-to-all 死锁

> **路径**：迭代层 → 日志/plog诊断层（穿插）→ 仅多卡异常 → 通信调度层（含 EP 切分校验）→ 模型层 → 数值层 → 熔断/预警层 → infra层（含错误扩散分析）→ 超参/代码层 → 止损链路总览

**背景**：2048 NPU 训练 openPangu-2.0-Flash，EP=64，TP=1，PP=4，DP=8，FP8 精度，seq_len=4096，global_batch=1024。训练至 step ~15000 时 loss 突发 NaN。

**关键认知**：死锁本身不会产生 NaN（死锁的典型表现是 hang/无输出），但 router logits 的数值溢出会**同时**导致两个平行后果——softmax 出 NaN 污染 loss，与路由概率塌缩触发 all-to-all 死锁。本案例的诊断从通信表象出发，最终追溯到 router 的数值层根因。本案例覆盖了 **16 条用户原声痛点**，是完整的"从表象到底层再到修复"的专家诊断路径示范。

### 0. 日志/plog 诊断前置 — 报错"说人话"

> 在深入定位之前，先把"看不懂的报错"翻译成可读的诊断信息。这一步穿插在后续每一步中，这里单独演示。

| 步骤 | 内容 |
|------|------|
| **现象** | step 15203 训练中断，Python 侧仅报 `RuntimeError: NCCL timeout in all-to-all`，无法直接定位原因。通常用户此时需要：① 去 Device 侧 `grep plog` 找有效信息 → ② 看不懂 HCCL/corner ops 报错 → ③ 逐层找框架负责人/算子开发 → 链路 2~7 天 |
| **plog 翻译** | `grep -i "error\|timeout\|mismatch" /var/log/npu/plog/plog_*.log` 在 step 15203 附近提取到：<br>① `hcom_all_to_all_v_` rank=23, send_count=0, recv_count=9832 → **send/recv 不匹配**（"你调用的 `dist.all_to_all` 在 rank 23 上 send buffer 为空，但期望接收 9832 个 token 的数据——buffer 大小不匹配导致死锁"）<br>② `aclnnSoftmaxV2` input[router_logits] contains inf values → **softmax 输入存在 inf**（"你代码中 router 的 softmax 收到了 inf 值——上游 `router_logits` 在 FP8 下溢出"） |
| **Ascend C→torch_npu 命名映射** | `hcom_all_to_all_v_` → `dist.all_to_all`（通信库）；`aclnnSoftmaxV2` → `F.softmax`（在 `router.forward` 中调用）；`aclnnMatmulV3` → `F.linear`（router 的 Linear 层） |
| **判据** | plog 已给出两个关键线索：① rank 23 的 all-to-all send/recv 不匹配（通信表象）；② router softmax 输入含 inf（数值根因）。这直接指引了后续的通信调度层→数值层排查方向 |
| **产出** | 翻译后的可读诊断 + 关联的 torch_npu 调用位置：`model.layers.38.mlp.router.forward` 中的 `F.softmax(router_logits)` 收到了 inf 输入 → 继续追查 router_logits 的来源 |

> **覆盖原声**：
> - Row 17：「算子报错信息经常不明确。有时plog里有有效信息，但Python侧不会直接显示，用户还要自己去查plog。如果能直接指出哪个参数的shape是什么、违反了什么约束，就能更快定位」→ 本节的 `grep plog` → 翻译为可读诊断流程直接解决此痛点
> - Row 18：「算子报错时一般报的是Ascend C里边的变量名或属性名，但我从torch_npu接口去用，名字和属性得想一想才能对上」→ 本节的 Ascend C→torch_npu 映射表演示了 `hcom_all_to_all_v_` → `dist.all_to_all`、`aclnnSoftmaxV2` → `F.softmax` 的翻译
> - Row 19：「每次要去Device侧plog grep error，感觉可以做成直接把有效信息抛到训练日志」→ 本节演示了 `grep plog` 后直接提取关键行并翻译为可读诊断
> - Row 20：「算子报 HCCL 通信错误或 corner ops 错误我们看不懂……一层层找框架负责人……链路很长，短的两三天长的一周」→ 本节将 `NCCL timeout` 翻译为 `dist.all_to_all` send/recv 不匹配 + `F.softmax` 输入 inf，把 2~7 天的找人链路压缩到一次 grep + 查表

### 1. 迭代层

| 步骤 | 内容 |
|------|------|
| **现象** | step 15203 loss 从 3.1 跳变至 NaN，grad_norm 从 12.4 跳至 inf；step 15200~15202 一切正常 |
| **判据** | 突变发生在单步内（step 15202→15203），排除缓慢漂移 |
| **产出** | 嫌疑 step = 15203 |

### 2. 分叉判定：单卡 vs 多卡

| 步骤 | 内容 |
|------|------|
| **操作** | 锁定 step 15203 的输入数据（dataloader seed 固定），分别在 1 NPU 和 2048 NPU 上重跑该 step |
| **结果** | 单卡：loss=3.21，grad_norm=11.8，完全正常 / 多卡：loss=NaN，grad_norm=inf |
| **判据** | 仅多卡复现，单卡正常 → **切入通信分支** |

> ⚠️ 单卡正常 ≠ 一定是纯粹的通信问题。可能是 router 数值溢出在单卡上被 FP8 截断掩盖（单卡无 all-to-all 则不触发 expert 塌缩的级联效应），需在模型层深挖。

### 3. 通信调度层 — 含 EP 切分校验

| 步骤 | 内容 |
|------|------|
| **观测** | 开启 `NCCL_DEBUG=INFO` 重跑 step 15203。NCCL trace 显示 EP rank 23（node2 GPU 7）在 `all-to-all` 调用处超时（30s timeout）。该调用属于 layer 38 MoE 的 expert dispatch 阶段。<br>↳ 可在 per-rank timeline 中复现：rank 23 的 all-to-all 横条拉满 30s（红），其余 63 rank 同期显示为空等（Wait 段）。 |
| **进一步确认** | 对比各 rank 的 all-to-all send/recv buffer size：rank 23 的 send buffer 为 0（没有 token 被 router 分发到其他 rank 的 expert），而 recv buffer 期望接收大量 token 数据，size 不匹配导致死锁 |
| **EP=64 切分校验** | EP=64 下，256 个 expert 均匀分配到 64 个 EP rank，每个 rank 承载 4 个 expert。正常运行时，all-to-all 的 send/recv 在两个方向上 token 数应大致匹配（每个 rank 发送 token 数 = 其他 63 rank 路由到本 rank 4 个 expert 的 token 总和；接收 token 数 = 本 rank token 被路由到其他 rank 的 expert 数的总和）。当前 rank 23 的 send=0、recv=9832 → **所有 token 被 router 判定应全部送往 rank 23 的 4 个 expert**，其他 252 个 expert 无 token 流入，EP 切分完全失效 |
| **判据** | all-to-all send/recv 不匹配 → 通信调度失步。EP 切分看似均匀（256/64=4），但因 router 输出塌缩为 one-hot，实际上只有 1 个 EP rank 在工作，63 个 rank 闲置。但死锁只是"果"，需继续追"因"——为什么 router 会把几乎所有 token 分配给 rank 23 的 expert 193？ |
| **产出** | 异常通信原语：`all-to-all` / 异常 rank：EP rank 23 / 关联层：layer 38 MoE / EP 切分状态：名义均匀（4 expert/rank），实际塌缩（1 rank 承载 100% token） |

> **覆盖原声**：
> - Row 5：「通信场景会遇到各种算子性能异常。有时是某张卡故障导致同步异常，有时是负载不均，通常表现为某个通信算子耗时变长。我们会先排查算子下发和卡间互联……希望报错能像PyTorch一样直接反映问题和处理方法」→ 本节通过 per-rank timeline 对比，直接定位到 `all-to-all` 超时 + rank 23 为异常卡，给出了"问题是什么（all-to-all send/recv 不匹配）+ 大概率原因（router 塌缩）+ 下一步怎么做（进入模型层查 expert 分布）"的结构化诊断
> - Row 25：「比较容易出现在并行切分不对，或有些融合算子不满足约束条件……如果能直接可视化整网模型、对应到代码，点选节点自动加dump，再可视化对比golden差异」→ 本节通过 EP=64 切分校验（256 experts/64 ranks → 名义均匀，实际 1 rank 承载 100% token），演示了并行切分不匹配的诊断方法

### 4. 模型层

| 步骤 | 内容 |
|------|------|
| **观测** | 提取 step 15203 所有 256 个 expert 的 token 分配统计：expert 193 收到 98% token（约 8028/8192），其余 255 个 expert 合计仅 164 token，其中 247 个 expert 为 dead expert（0 token）。expert 193 恰好位于 EP rank 23 |
| **判据** | 全量 expert 分布严重塌缩——不仅是 expert 193 过载，255 个 expert 几乎完全闲置。这不是普通的路由倾斜（CV=10~20%），而是 router 的 softmax 输出几乎退化为 one-hot |

### 5. 数值层 — 追查 router 的精度路径

> 此层是本案例的**核心转折点**：从"通信怎么死的"下钻到"数值为什么先崩了"。

| 步骤 | 内容 |
|------|------|
| **观测** | ① dump step 15203 时 layer 38 router 的 raw logits（softmax 之前），发现 max(logits)=**1846**（正常应 < 50），且存在 `inf` 值——FP8 E4M3 下 `exp(1846)` 直接溢出为 inf。<br>② 检查 router 计算精度路径：当前实现中 router 的 softmax 在 **FP8** 下计算（`router_logits → FP8 cast → softmax`），而非业界建议的 FP32。对应 Ascend C 侧为 `aclnnSoftmaxV2` 的输入 dtype=FP8，这在大动态范围 logits 上是危险的。<br>③ AMP scaler 日志显示 loss scale 从 step 15000 起从 65536 持续衰减至 step 15202 的 4096，说明训练已处于持续 FP 溢出的临界状态。衰减曲线：`15000:65536 → 15050:32768 → 15100:16384 → 15150:8192 → 15202:4096`，每 ~50 step 减半一次。<br>④ **z-loss 缺失**：当前训练配置中未开启 z-loss。若开启 z-loss（系数 1e-4），可在 logits 趋向极端值前施加正则化惩罚，抑制其漂移出 FP8 安全区间 |
| **判据** | FP8 下 router logits 溢出 → softmax 产生 NaN/inf → 路由概率退化为一组非法值 → top-k 选取极端集中于单个 expert（expert 193）→ 同时触发两个后果：**A)** NaN 沿 forward 传播到 loss；**B)** 所有 token 路由到 rank 23 → all-to-all 死锁。**死锁和 NaN 是同一 root cause（router FP8 overflow）的两个平行后果，而非因果关系** |
| **Ascend C 命名映射（知识沉淀）** | 本案例中的关键算子命名链路：`aclnnSoftmaxV2`（Ascend C）↔ `torch_npu.softmax` / `F.softmax`（PyTorch）↔ `router.forward` 中的 `softmax(router_logits)`（用户代码）。后续遇到类似问题可直接按此映射快速定位 |
| **产出** | 根因：router softmax 在 FP8 精度下计算，logits 动态范围超出 FP8 表示能力 / 前置信号：AMP loss scale 持续衰减（65536→4096，4 次减半）是 NaN 的预警指标，z-loss 缺失使 logits 无约束地向极端漂移 / dead expert 占比 96.5%（247/256） |

> **覆盖原声**：
> - Row 21：「报错那句代码不一定是原因，可能上游算子结果 shape 不对 dtype 不一样或内存越界。从框架侧根据报错定位范围，逻辑没问题就逐行打印 shape dtype，看经过哪个算子后不对」→ 本节演示了从"通信 timeout 表象"逆流追溯到"router FP8 overflow 根因"的完整因果推理，死锁和 NaN 是 router FP8 溢出的两个平行后果
> - Row 28：「我们会把低精度训练任务的监控数据画出来，看loss、z-loss、load balance loss、learning rate、grad norm和吞吐。有些格式下训练收敛会出现问题，比如BF16和MXFP8正常，FP8不行。要定位从哪个位置开始出现较大跳变或引入误差」→ 本节通过 AMP scaler 衰减曲线 + z-loss 缺失分析 + router logits 分布，给出了 FP8 低精度训练的退化链路和观测方法

### 5.5 熔断/预警层 — 如果当时有熔断……

> 此层复盘：如果当时部署了熔断机制，能多早拦截这个事故？损失能减少多少？

| 步骤 | 内容 |
|------|------|
| **复盘时间线** | step 15000：AMP scaler=65536（正常）。step 15050：AMP scaler → 32768（第 1 次减半，🟡 注意级）。step 15100：AMP scaler → 16384（第 2 次减半，仍 🟡）。step 15150：AMP scaler → 8192（第 3 次减半，接近 🟠 警告线）。step 15200：AMP scaler → 4096（第 4 次减半，🟠 警告级——**应触发自动 dump**）。step 15202：AMP scaler=4096，loss=3.1 尚正常——**这是最后的拦截窗口**。step 15203：loss NaN，🔴 熔断级——**但无人监控，训练在 NaN 后仍可能空跑日志** |
| **如果部署了熔断** | step 15150（scaler=8192）触发 🟡 通知 → on-call 收到告警。step 15200（scaler=4096）触发 🟠 自动 dump router logits + 激活张量 → 可在 loss NaN 之前就发现 max(logits) 已从正常的 ~30 飙升至 ~800、z-loss 缺失使 logits 无约束。**如果 step 15202 触发 🔴 熔断（scaler < 65536/32=2048），可在 NaN 之前停训**——虽然本案例中 scaler=4096 尚未跌破 2048，但可设置更灵敏的规则：`scaler < 初始值/8 且持续 ≥100 step` 即熔断 |
| **万卡成本核算** | 本案例 2048 NPU，从 step 15000（scaler 开始衰减）到 step 15203（loss NaN）共 203 step。若在 step 15150 拦截，可省 53 step × 2048 NPU × 2 元/卡时 ≈ **21.7 万元**。若是万卡集群同样模式，203 step 空跑 ≈ **百万元级损失** |
| **产出** | 熔断规则建议：将 `AMP scaler < 初始值/8 且连续 50 step 未恢复` 设为 🔴 熔断条件；将 `AMP scaler < 初始值/4` 设为 🟠 自动 dump 条件。本案例的 AMP scaler 衰减曲线应作为后续训练监控的 baseline 参考 |

> **覆盖原声**：
> - Row 30：「TensorBoard监测到国产卡loss曲线跟NV偏离度超过阈值就局部熔断……逐层自动化比对两端tensor的余弦相似度和绝对误差，相似度低于0.999就锁定精度分叉在哪一层」→ 本节通过 AMP scaler 衰减（65536→4096）作为熔断预警信号，对标了"偏离阈值即熔断"的能力，给出了三级预警体系和自动 dump 机制
> - Row 32：「报错了服务不一定停，一直卡着跑日志，找错误要往上翻很久。程序报错应该停掉，不然占着资源别人排队」→ 本节通过 🔴 熔断级（loss=NaN 或 scaler < 1/32）立即停训 + 释放 GPU 资源，直接解决"报错不停、占卡位"的痛点
> - Row 34：「万卡训练突然出现NaN、loss跑飞或吞吐掉得很厉害，如果能实时识别、定位根因、给出解决方案就很有用。万卡集群两块钱一卡时就是两万元一小时，十二小时搞不定就是二十四万没了」→ 本节通过复盘时间线 + 万卡成本核算，给出了"实时识别→自动 dump→熔断止损"的完整方案，将损失窗口从 203 step 压缩到 50 step 以内

### 6. infra层 — 含错误扩散路径分析

| 步骤 | 内容 |
|------|------|
| **观测** | 问题集中在 EP rank 23（node2 GPU 7），属于 PP stage 3（layers 34~45）。AMP scaler 衰减在全部 64 rank 上同步发生，但 only rank 23 因 expert 193 的地理位置成为死锁的"引爆点"——如果 expert 193 位于其他 rank，只会换一个 rank 触发死锁 |
| **错误扩散路径** | 这是一个典型的"单点故障→全局扩散"模式：① EP rank 23 的 router softmax 最先溢出（数值层根因）→ ② rank 23 的 expert 193 被分配 98% token，all-to-all send=0 / recv=9832 导致死锁（通信调度层表象）→ ③ all-to-all 是同步屏障操作，rank 23 未完成意味着所有 64 个 EP rank 全部卡在 barrier 上（扩散到全集群）→ ④ PP stage 3 的 rank 23 卡死 → PP pipeline 断裂 → **所有 PP stage 的 rank 全部等待** → ⑤ NCCL timeout 30s 后报错，但报的是"通信 timeout"而非"router 溢出"——表象与根因分离，这正是"报错那句代码不一定是原因"的典型场景 |
| **扩散可视化** | `EP rank 23 (node2 GPU 7) router FP8 overflow → rank 23 all-to-all 死锁 → 64 EP ranks barrier 同步等 → PP stage 3 断裂 → 4 PP stages 全卡 → 2048 NPUs 全部 hang → 30s 后 NCCL timeout 报错` |
| **判据** | 问题聚集在单个 EP rank → 局部路由塌缩，非全局硬件故障。但根因（router FP8 overflow）是系统性的——只是 expert 193 恰好落在 rank 23 上使其成为"引爆点"。这种"单点引爆、全局扩散"的模式是分布式训练中最具迷惑性的一类故障：报错位置 ≠ 根因位置 |
| **产出** | 嫌疑范围：node2 GPU 7（EP rank 23），PP stage 3，layer 38 MoE / 扩散范围：全部 64 rank（因 all-to-all barrier + PP 依赖链）/ 关键教训：在 512+ rank 的大规模训练中，一个 rank 的数值溢出可通过 gather/all-to-all 扩散到数百 rank，必须自动做跨 rank 的首因定位而非人工逐一比对 |

> **覆盖原声**：
> - Row 22：「多卡多机训练，很多卡打日志，最终报通信 timeout，实为别的进程提早退出或某算子挂；要收集几百 rank 逐个找，日志对定位无用、全凭经验」→ 本节通过 5 步错误扩散路径分析，演示了"timeout 只是表象，rank 23 才是第一张多米诺"——从 64 rank 全部 hang 的通信 timeout 表象，逆流追溯到 rank 23 router FP8 overflow 首发根因，免去人工逐一收集比对几百 rank 日志
> - Row 33：「512个rank某节点错误通过gather扩散到所有机器，不可能每台逐一比较，我们在代码里加打印写Python做逻辑比较。比如100个rank第8个节点最先出问题，现有诊断平台也诊断不到这么细」→ 本节通过 5 步错误扩散路径分析（rank 23 单点溢出 → all-to-all barrier → 64 rank 全卡 → PP pipeline 断裂），演示了如何自动定位"谁先出问题"而不需人工逐一比对 64 张卡

### 7. 超参/代码层

| 步骤 | 内容 |
|------|------|
| **诊断总结** | 根因是 router softmax 在 FP8 下计算 + 缺乏 logits 正则化。三个问题叠加：① 精度路径错误（FP8 softmax，应 FP32）；② 无 z-loss 抑制 logits 极端值；③ router 学习率与 expert 相同（应降低）。AMP scaler 持续衰减是可在 NaN 前捕获的预警信号 |
| **修改** | 按优先级：① **router softmax 改 FP32** ——`router_logits = router(x.float()); probs = softmax(router_logits); probs = probs.to(dtype)`，这是最关键的修复，消除 logits 溢出的可能性；② **加 z-loss** ——系数 1e-4，抑制 logits 向极端漂移，同时将 z-loss 值纳入训练监控面板；③ **降低 router 学习率** ——router lr = expert lr × 0.1；④ **gradient clipping** ——`clip_grad_norm=1.0`，MoE 训练的标配；⑤ 增大 `n_group` 8→16 作为路由多样性的辅助保障；⑥ NCCL timeout 30s→60s 作为训练不中断的兜底（仅兜底，不解决根因）；⑦ **部署熔断规则** ——将 AMP scaler 衰减纳入监控，`scaler < 初始值/8` 触发 🟠 自动 dump，`scaler < 初始值/16 且持续 50 step` 触发 🔴 熔断停训 |
| **验证** | ①~④ 从 step 15000 续跑：router logits max 稳定在 18~35（安全范围），AMP scaler 维持在 65536 不衰减，256 expert 的 token CV 降至 8~15%。step 15203 正常通过，继续训练 5000 step 无 NaN 无死锁。⑦ 熔断规则：在 step 15000~17000 窗口内，若 scaler 衰减至 4096 以下将触发 🟠 自动 dump 告警 |

> **覆盖原声**：
> - Row 12：「配套 Expert System……把定位从大半天缩到几十分钟……新手仍需要更直白的原因解释、操作建议和代码关联」→ 本节 7 条修改项均含"改什么 + 怎么改 + 为什么这样改 + 怎么验证"，完整示范了 Expert System 的诊断→修复闭环
> - Row 15：「MindStudio数据是够的……唯一就是缺少初步分析和可优化方向，非常依赖人的经验，新手上手有门槛」→ 本案例从 loss NaN 到 7 条可执行修改的全链路，就是"初步分析 + 可优化方向"的完整示范
> - Row 16：「应该是有经验专家把经验提炼给AI；AI分析时不光给结果，还把要注意的分析方法同时交给专家，过程和结果结合起来才可信」→ 本案例每一层都标注了判据、观测手段和产出，将专家的分析方法显式化、可复用
> - Row 34：「万卡训练突然出现NaN……如果能实时识别、定位根因、给出解决方案就很有用」→ 本节 7 项修改 + 验证方案，从根因（router FP8 overflow）到修复（改 FP32 + z-loss + lr + clip）到熔断部署，给出了完整的解决方案

### 8. 止损链路总览 — 从 NaN 到修复的完整时间线

> 此节汇总本案例涉及的全部 CheckList 原声痛点，以及每个痛点在诊断路径中的解决位置。

| 阶段 | 时间节点 | 事件 | 对应 CheckList 痛点 |
|---|---|---|---|
| 🔍 发现 | step 15203 | loss NaN，训练中断 | Row 34：万卡训练突然 NaN/loss 跑飞，希望实时识别 |
| 📋 日志翻译 | 排查开始 | plog 翻译 → `dist.all_to_all` send/recv 不匹配 + `F.softmax` 输入 inf | Row 17~19：plog 有信息但 Python 侧不显示，需手动 grep → 翻译为可读诊断；Ascend C 内部名→torch_npu 接口映射 |
| 🔀 分叉判定 | 排查中 | 单卡重跑正常，多卡复现 NaN → 切入通信分支 | 定位链「分叉判定」机制 |
| 📡 通信调度 | 排查中 | rank 23 all-to-all timeout，send=0/recv=9832 → EP 切分校验失败 | Row 5：通信死锁逐层排查；Row 25：EP=64 并行切分不匹配分析 |
| 🧠 模型层 | 排查中 | expert 193 收到 98% token，247 dead experts → 路由塌缩 | Row 20：HCCL all-to-all timeout 需逐层排查 |
| 🔢 数值层 | 排查中 | router logits max=1846，FP8 softmax → inf，AMP scaler 65536→4096，z-loss 缺失 | Row 21：通信死锁是"果"，router FP8 溢出是"因"；Row 28：FP8 低精度训练 z-loss 缺失 |
| 🛡️ 熔断预警 | 复盘 | 若 step 15200 部署 🟠 自动 dump，可在 NaN 前捕获证据；若 scaler < 2048 触发 🔴 熔断，可避免 NaN | Row 30：AMP scaler 衰减作为熔断预警；Row 32：报错即停，别占卡位 |
| 🌐 扩散分析 | 排查中 | rank 23 单点溢出 → all-to-all barrier → 64 EP rank 全卡 → PP pipeline 断裂 → 2048 NPU 全部 hang | Row 22：多卡多机日志淹没、通信 timeout 表象→首因定位；Row 33：512 rank 某节点错误经 gather 扩散到所有机器 |
| 🔧 修复 | 诊断完成 | 6 项修改（softmax FP32 + z-loss + router lr + clip + n_group + 熔断），5000 step 验证通过 | Row 34：实时识别→定位根因→给出方案；Row 12/15/16：Expert System 诊断路径 |

**总止损时间估算**：
- 无工具/无经验：2~7 天（从看不懂报错→逐层找人→定位→修复，典型链路）
- 按本定位链 + plog 翻译 + 熔断：~30 分钟（plog 即时翻译 → 分叉判定 5min → 通信层 5min → 模型/数值层 10min → 修复 10min）

---

## 案例二：计算分支 — q_b_proj FP8 溢出导致 grad_norm 缓慢发散

> **路径**：迭代层 → 单/多卡均异常 → 模型层 → 算子层 → 张量层 → infra层 → 超参/代码层

**背景**：同上 2048 NPU 训练 openPangu-2.0-Flash，EP=64，TP=1，PP=4，DP=8，FP8 精度。训练至 step ~8000 后 grad_norm 持续上升，loss 缓慢恶化。

### 1. 迭代层

| 步骤 | 内容 |
|------|------|
| **现象** | step 8200 起 grad_norm 从 ~10 逐步升至 ~85（step 8600），loss 从 2.95 升至 4.82，未现 NaN 但趋势持续恶化 |
| **判据** | 平稳漂移 → 取漂移起始段 step 8200~8600 |
| **产出** | 嫌疑 step 范围：8200~8600 |

### 2. 分叉判定：单卡 vs 多卡

| 步骤 | 内容 |
|------|------|
| **操作** | 在单卡和多卡环境分别重跑 step 8200~8600 |
| **结果** | 单卡：grad_norm 同样从 10 升至 85 / 多卡：grad_norm 从 10 升至 85 |
| **判据** | 单卡也能复现 → **沿计算主干，进入模型层** |

### 3. 模型层

| 步骤 | 内容 |
|------|------|
| **观测** | 绘制 46 层 × step（8200~8600）grad_norm 热力图。layer 38（MoE 层）的 grad_norm 热力值（~450）是其他层（~30~50）的约 10 倍，且随 step 持续增长 |
| **判据** | 热力图纵向高亮带锁定 layer 42 |
| **产出** | 问题层：`model.layers.38`（MoE TransformerLayer，Sparse MLA attention + openPangu MoE） |

### 4. 算子层

| 步骤 | 内容 |
|------|------|
| **观测** | 层内算子误差瀑布：按执行顺序对比 layer 38 各算子输出与 baseline（step 8000 正常时的值）。`input_layernorm`（1e-7）→ `q_a_proj`（1e-7，MLA 低秩下投影 [2560→1024]）→ `q_b_proj`（**1e-7→3.2e-2 跳跃**，MLA 低秩上投影 [1024→9216]）→ `kv_a_proj/kv_b_proj`（1e-7）→ `core_attention`（8.1e-2，Sparse FlashAttention 进一步放大）→ 后续算子持续偏高 |
| **判据** | 误差在 `q_b_proj` 处出现量级跳跃（1e-7→1e-2），为首害算子 |
| **产出** | 问题算子：`model.layers.38.self_attn.q_b_proj`（Linear [1024→9216]，FP8 精度，MLA 低秩上投影） |

### 5. 张量层

| 步骤 | 内容 |
|------|------|
| **观测** | dump step 8500 时 layer 38 `q_b_proj` 的输入激活张量（shape [4096, 1024], FP8 E4M3，来自 q_a_proj 的低秩输出）。绘制数值分布直方图： |
|  | • 正常区间（0~448）：占 96.8% 元素，分布与 baseline 一致 |
|  | • 溢出区间（>448，即 FP8 E4M3 max）：占 **3.2%** 元素，最大值 2.3×10⁴ |
|  | TopK 误差表：diff 最大的 100 个元素索引集中在 latent dim [896, 1024] 区间，对应 q_a_proj 输出的特定频段 |
| **判据** | 3.2% 的输入元素超过 FP8 E4M3 表示范围（max=448），在 `q_b_proj` 的 Linear 计算中产生截断误差，经 MLA 低秩上投影（1024→9216）进一步放大到 48 个 attention head |
| **产出** | 嫌疑元素：latent dim [896, 1024] 的尾部 128 维 / 数值区间：[448, 2.3×10⁴] / 偏差量级：3.2e-2（算子输出级） |

### 6. infra层

| 步骤 | 内容 |
|------|------|
| **观测** | 检查问题是否局限在特定 PP stage 或 EP rank。layer 38 属于 PP stage 3（layers 34~45），但在所有 64 个 EP rank 上均观测到相同的 q_b_proj 溢出模式 |
| **判据** | 跨所有 rank 复现 → 全局精度问题，非硬件/特定节点故障 |
| **产出** | 全局问题，与 PP/EP 切分无关，根因在 FP8 数值表示能力不足 |

### 7. 超参/代码层

| 步骤 | 内容 |
|------|------|
| **修改** | ① 对 layer 38 的 `q_b_proj` 输入增加 per-tensor dynamic scaling：`input = input / max(|input|) * 448`，将值域动态映射到 FP8 安全区间后再做 Linear；② 或者在 `q_b_proj` 前插入一层 `Fp8Cast` 时使用 delayed scaling 策略（参考 openPangu-2.0-Flash 的 FP8 训练方案），对 latent dim 尾部高频分量单独 scale；③ 长期方案：评估是否对 L30+ 深层的 q_b_proj 改用 BF16 |
| **验证** | 方案① 从 step 8000 续跑，layer 38 的 q_b_proj 误差降至 1e-6 量级，grad_norm 稳定在 10~15，继续训练 10000 step 无发散 |


---

## 案例三：计算分支 — 低精训练 loss 尾部不收敛，量化误差累积导致梯度信号淹没

> **路径**：迭代层 → 单/多卡均异常 → 张量数值分析（分布曲线 + 宏观指标 + 量化风险 + 算子定位）→ 误差传递路径（逐层对比高精度基线）→ infra层 → 超参/代码层

**背景**：2048 NPU 训练 openPangu-2.0-Flash，EP=64，TP=1，PP=4，DP=8，HiF8 混合精度（forward: FP8 E4M3 hybrid, backward: BF16, master weights: FP32），同时跑 BF16 全精度对照组。训练至 step ~25000 后 HiF8 与 BF16 的 loss 曲线开始分叉——BF16 继续下降，HiF8 停滞不前。step ~31000 后 HiF8 loss 微幅反弹，grad_norm 持续衰减至接近 0——模型进入"低精训练陷阱"：梯度信号被量化噪声淹没。

### 1. 迭代层

| 步骤 | 内容 |
|------|------|
| **现象** | step 0~25000：HiF8 与 BF16 基线 loss 紧密跟随，8.5→2.5 平稳下降，grad_norm 8~15 正常波动。**step 25000 起两条 loss 曲线分叉**——BF16 继续下降至 ~1.8，HiF8 停滞在 2.1 附近。step 31000 后 HiF8 loss 微幅反弹（2.08→2.15），grad_norm 从 ~10 持续衰减至 0.3（step 35000），模型几乎不再更新 |
| **判据** | HiF8 与 BF16 的分叉 + 梯度消失 → FP8 混合精度引入的渐进式数值退化。取分叉起始段 step 25000~35000 |
| **产出** | 嫌疑 step 范围：25000~35000（分叉起点 25000，反弹点 31000） |

### 2. 分叉判定：单卡 vs 多卡

| 步骤 | 内容 |
|------|------|
| **操作** | 固定 seed+dataloader，在单卡（1 NPU, BF16 全精度对照组）和多卡（2048 NPU, HiF8 混合精度）上分别重跑 step 25000~35000 |
| **结果** | 单卡 BF16：loss 继续从 2.5 下降至 1.95，grad_norm 稳定在 8~12 / 多卡 HiF8：loss 在 2.1 附近停滞，grad_norm 衰减至 0.3 |
| **判据** | 单卡（BF16 全精度）正常、多卡（HiF8 混合精度）异常 → **沿计算主干，进入张量数值分析**。根因指向 FP8 量化引入的数值误差 |
| **产出** | 精度模式差异确认为根因方向：HiF8 量化 → 数值退化 |

### 3. 张量数值分析 — 分布曲线 + 宏观指标 + 量化风险 + 算子定位

> 此层是本案例的核心：锁定 layer 35 为异常层后，通过张量数值分布曲线、宏观统计指标、量化风险评估，从数值层面完整刻画 FP8 低精度训练的退化机理，并定位首害算子。

#### 3.1 分布曲线对比

| 步骤 | 内容 |
|------|------|
| **观测** | dump step 32000 时 layer 35 各关键张量，绘制数值分布直方图 + KDE 密度曲线，与 BF16 baseline 叠图对比： |
|  | • **q_b_proj 输入（MLA 低秩 latent，来自 q_a_proj 输出）**：BF16 baseline → 分布在 [-3.2, 3.8]，近似正态 N(0.12, 1.45)。FP8 step 32000 → 分布显著右偏（skewness=+1.8），均值漂移至 +2.4，尾部在 +448 处形成显著截断堆积峰（6.8% 元素被 clip 到 max），左尾相对正常 |
|  | • **core_attention 输出（attention weights 加权后的上下文表示）**：BF16 baseline → 分布 [-1.8, 2.0]，多模态（Sparse MLA 48 个 attention head 产生不同的子分布）。FP8 step 32000 → 分布严重畸变：主峰塌缩至 [-0.3, 0.3]（信息被 softmax 的 FP8 截断抹平），右尾在 +448 处堆积（大值被 clip），整体方差从 0.85 缩至 0.21——attention 输出几乎失去区分度 |
|  | • **shared_expert.gate_up 输出（FFN 中间激活，gate+up 合并）**：BF16 baseline → 分布在 [-12, 15]，长尾（Linear [2560→2048] 各通道激活幅度不同，1024 gate + 1024 up）。FP8 step 32000 → 双侧尾部均在 ±448 处出现截断峰（总计 12.4% 元素被 clip），主峰被压缩至 [-200, 200]——FFN 的非线性表达能力被严重破坏 |
| **判据** | 三张分布图一致揭示：FP8 E4M3 的有限动态范围（max=448）无法容纳深层激活值的自然长尾，截断误差经 softmax（信息抹平）→ FFN（非线性压缩）逐级放大，最终梯度信号被量化噪声淹没 |
| **产出** | 分布偏移三阶段：激活值右偏 → attention 信息坍缩 → FFN 输出截断饱和 |

#### 3.2 宏观统计指标

> 在 step 32000 时 layer 35 各关键张量上计算以下宏观指标，并与 BF16 baseline 对比。每个指标服务于特定的诊断目的。

| 张量 | 指标 | FP8 (step 32000) | BF16 baseline | 诊断含义 |
|------|------|------------------|---------------|----------|
| q_b_proj 输入 | **Mean** | +2.41 | +0.12 | 均值显著右偏 → 激活分布整体向 FP8 正上限漂移 |
| q_b_proj 输入 | **Std** | 3.82 | 1.45 | 标准差扩大 2.6× → 分散度增加意味更多元素超出 FP8 范围 |
| q_b_proj 输入 | **Skewness** | +1.83 | +0.08 | 严重右偏 → 右侧长尾超出 FP8 max=448 |
| q_b_proj 输入 | **Kurtosis** (excess) | **+7.42** | -0.12 | 高峰度说明分布有重尾 + 尖峰：少数极端大值主导数值范围，大量中等值被挤压到窄区间 |
| q_b_proj 输入 | **Outlier Ratio** (>3σ) | **8.7%** | 0.9% | 离群率飙升至接近 10× baseline |
| q_b_proj 输入 | **p99** | 378.4 | 4.12 | p99 已接近 FP8 max=448 |
| q_b_proj 输入 | **p99.9** | **447.8** (clip) | 4.89 | p99.9 被硬截断在 FP8 上限 → 最强 0.1% 激活完全丢失 |
| core_attn 输出 | **Mean** | +0.08 | +0.11 | 均值回中（softmax 归一化），但信息已丢失 |
| core_attn 输出 | **Std** | 0.21 | 0.85 | 方差缩至 1/4 → attention 输出区分度丧失 |
| core_attn 输出 | **KL Divergence** (vs BF16) | **2.31 bits** | 0 (self) | 与 BF16 baseline 的 KL 散度 > 2 bits → 分布已发生根本性变化 |
| gate_up 输出 | **Outlier Ratio** (>3σ) | **12.4%** | 1.1% | FFN 中间层离群率爆炸 |
| gate_up 输出 | **Quantization SNR** | **6.8 dB** | 42.1 dB | 量化信噪比从 42dB 降至 6.8dB → 有效信号被量化噪声严重污染 |
| gate_up 输出 | **Kurtosis** (excess) | **+15.3** | +0.45 | 极端高峰度 → 分布呈"尖峰 + 超级重尾"形态，量化 truncation 人为制造了双峰 |

> **指标解读**：峰度（Kurtosis）是本案例最关键的单一指标——从 baseline 的接近正态（-0.12）到退化后的 +7.42（q_b_proj 输入）乃至 +15.3（gate_up 输出），表明 FP8 截断在分布的两个尾部累积了被 clip 的值（形成截断峰），中间则因数值范围受限而过度集中（形成尖峰）。这种"尖峰+截断双尾"是低精训练退化的典型数值指纹。

#### 3.3 量化风险评估矩阵

| 风险维度 | q_b_proj 输入 | core_attn 输出 | gate_up 输出 | 综合风险 |
|----------|-------------|----------------|----------------|----------|
| **动态范围适配度** | ⚠️ 临界（p99=378，已逼近 448） | 🔴 危险（FP8 截断抹平 softmax 多模态） | 🔴 危险（12% 元素 clip，双侧饱和） | 🔴 高风险 |
| **量化噪声比 (QSNR)** | 18.3 dB | 11.5 dB | **6.8 dB** | 🔴 < 10dB 不可用 |
| **KL 散度 (vs BF16)** | 0.87 bits | 2.31 bits | **3.45 bits** | 🔴 > 1 bit 显著偏移 |
| **梯度信号可恢复性** | 🟡 可部分恢复（dynamic scaling） | 🔴 不可恢复（信息论极限，softmax 抹平不可逆） | 🔴 不可恢复（截断饱和，梯度=0） | 🔴 需结构性修改 |

#### 3.4 算子误差定位

| 步骤 | 内容 |
|------|------|
| **观测** | 在 layer 35 内做算子误差瀑布（横轴：按执行顺序排列的算子，纵轴：与 BF16 baseline 的 MSE）。`input_layernorm`（3e-8）→ `q_a_proj`（6e-7，MLA 低秩下投影 [2560→1024]）→ `q_b_proj`（**6e-7→4.1e-3 跳跃**，MLA 低秩上投影 [1024→9216]）→ `kv_a_proj/kv_b_proj`（8e-7）→ `core_attention`（**4.1e-3→1.6e-1 跳跃**，Sparse FlashAttention）→ `o_proj`（1.8e-1，[6144→2560]）→ `shared_expert.gate_up`（**1.8e-1→7.3e-1 激增**，[2560→2048]）→ 后续 MoE expert 算子持续恶化（> 0.5） |
| **判据** | 三处关键跳跃——`q_b_proj`（首次误差放大，1e-7→1e-3，20000×，MLA 低秩上投影展开到 48 head × 192 dim）、`core_attention`（softmax 放大，1e-3→1e-1，40×）、`shared_expert.gate_up`（FFN 最终放大，1e-1→7e-1，4×）。首害算子为 `q_b_proj`（MLA Q 路低秩上投影），但真正的误差放大器是 `core_attention` 的 softmax 和 `gate_up` 的 FP8 cast |
| **产出** | 首害算子：`model.layers.35.self_attn.q_b_proj`（Linear [1024→9216]，MLA Q 路低秩上投影）/ 误差放大器：`core_attention`（softmax 的 FP8 截断）、`shared_expert.gate_up`（FFN 入口 FP8 cast） |

### 4. 误差传递路径 — 逐层对比高精度基线

> 核心诊断方法：以 BF16 全精度训练为 baseline，沿 forward 计算图逐层对比 FP8 混合精度训练的激活输出和梯度，绘制**误差累积曲线**，找到偏差首次显著偏离的起点 layer。

#### 4.1 误差累积曲线

| 步骤 | 内容 |
|------|------|
| **方法** | 对 step 32000 时 46 层中每一层的中间激活（post-layernorm 的输出），计算 FP8 vs BF16 baseline 的 MSE（逐层累积）。横轴为 layer index（1→46），纵轴为 log10(MSE) |
| **曲线形态** | layer 1~25：MSE 在 1e-7~1e-6 平稳（FP8 在前半段保持良好精度）。layer 26~34：MSE 从 1e-6 缓慢爬升至 5e-4（激活值尾部逐渐累积 FP8 截断误差，但仍在可控范围）。**layer 35**：MSE 从 5e-4 **跳跃至 2.3e-1**（误差量级跃升 3 个数量级）。layer 36~41：MSE 在 0.2~0.8 高位震荡（误差已固化）。layer 42~45：MSE 飙升至 3.5（最终输出层误差不可接受） |
| **判据** | layer 35 是误差累积曲线的"拐点"——在此之前误差平缓，在此之后误差爆发。与模型层 grad_norm 热力图的结论一致 |
| **产出** | 偏差起点：**layer 35** / 误差阶跃幅度：5e-4 → 2.3e-1（460×） |

#### 4.2 逐层对比详表（偏差起点附近）

| Layer | Attn 输出 MSE | FFN 输出 MSE | Grad MSE | 状态 |
|-------|--------------|-------------|----------|------|
| 33 | 8.2e-7 | 1.1e-6 | 2.3e-6 | 🟢 正常 |
| 34 | 3.5e-6 | 6.8e-6 | 1.2e-5 | 🟢 正常（略有抬升） |
| **35** | **2.3e-1** | **7.3e-1** | **1.8e+1** | 🔴 **偏差起点 —— Attn 与 FFN 同时爆炸** |
| 36 | 0.41 | 0.55 | 8.7e+0 | 🔴 误差向下游传播 |
| 37 | 0.38 | 0.62 | 6.2e+0 | 🔴 误差保持 |
| 38 | 0.45 | 0.78 | 9.1e+0 | 🔴 误差放大 |

> **关键发现**：layer 35 不是孤立异常——其 Attn 和 FFN 模块在同一 step 同时出现 MSE 跳跃，说明问题不是某个特定算子 bug，而是**该层的激活值整体数值分布已超出 FP8 的可表示范围**。layer 34→35 的 hidden state 经残差连接 + RMSNorm 后，在深层积累了足够的数值漂移，最终在 layer 35 突破了 FP8 的容限。

#### 4.3 误差传递因果链

```
layer 1~34 激活值数值漂移（FP8 截断误差逐层微量累积）
    │
    ├── layer 34 输出 hidden state 均值已漂移至 +0.8（baseline: +0.1）
    │   尾部 p99.9 从 4.9 升至 68.3（但仍安全 < FP8 max=448）
    │
    ▼
layer 35 RMSNorm → 归一化后方差放大
    │
    ▼
layer 35 q_b_proj (Linear [1024→9216]，MLA 低秩上投影) → 权重矩阵与右偏输入的乘积进一步放大离群值
    │  → 输出中 6.8% 元素被 clip 到 FP8 max=448，展开到 48 head × 192 dim
    │
    ▼
layer 35 core_attention softmax → FP8 截断令 attention weights 失去稀疏性
    │  → 上下文表示坍缩为近均匀向量（信息抹平）
    │
    ▼
layer 35 o_proj (Linear [6144→2560]) → 从坍缩的 attention 输出重建 hidden state
    │  → 输出的有效信息量急剧下降，噪声占比 > 50%
    │
    ▼
layer 35 shared_expert.gate_up → FP8 截断在双侧尾部 clip 12% 元素
    │  → FFN 的激活函数（SiLU）输入被截断 → 非线性区被削平
    │  → 梯度反向传播时 gate_up 的 grad 接近 0（截断区 grad=0）
    │
    ▼
layer 35 → layer 36 残差连接传播受损的 hidden state
    │  → 下游 layer 36~45 的输入已严重退化
    │  → grads 在第 35 层被切断，上游 layer 1~34 的 grad 也逐渐衰减
    │
    ▼
全局梯度消失（grad_norm → 0.3）→ 模型停止学习 → loss 反弹
```

### 5. infra层

| 步骤 | 内容 |
|------|------|
| **观测** | 检查问题是否局限在特定 PP stage 或 EP rank。layer 35 属于 PP stage 3（layers 34~45），但在所有 64 个 EP rank 上均观测到相同的分布偏移模式。进一步检查各 rank 的 FP8 量化参数（scale factor）：layer 35 的 per-tensor scale 从 step 25000 的 0.62 持续下降至 step 32000 的 **0.18**——scale 过小意味着 FP8 的量化粒度变粗，每个量化 bin 代表更大的实数值间隔，舍入误差增大 |
| **判据** | 跨所有 rank 复现 + scale factor 持续衰减 → 全局精度问题。FP8 per-tensor scaling 策略的 scale 衰减是量化误差累积的放大器 |
| **产出** | 全局问题。FP8 per-tensor scale 衰减（0.62→0.18）使有效量化精度从 ~4.5 bit 退化至 ~2.8 bit |

### 6. 超参/代码层

| 步骤 | 内容 |
|------|------|
| **诊断总结** | 根因是 FP8 E4M3 per-tensor 量化在深层激活值上的动态范围不足。不是某一层的 bug，而是**深层激活值的自然长尾分布 + 静态 per-tensor FP8 量化**这一组合的系统性缺陷。量化误差经 softmax（信息抹平）→ FFN（非线性饱和）形成正反馈：截断→信息损失→更大截断 |
| **修改** | ① **per-token + per-channel 混合量化**：对 attention 模块的 q/k/v projection（含 q_b_proj/kv_b_proj 低秩上投影）采用 per-token scaling（每个 token 独立 scale），对 FFN 的 gate_up projection 采用 per-channel scaling（每个输出 channel 独立 scale），替代原有的 per-tensor 统一 scale。这可将有效量化精度恢复至 ~5.5 bit；② **动态 scale 上界保护**：在 FP8 cast 时设置 scale 的最小值下限（`scale = max(computed_scale, 512 / 448)`），防止 scale 过度衰减导致量化粒度过粗；③ **layer 35 起启用 BF16 attention softmax**：对 L30+ 深层，attention softmax 的计算和输出保持 BF16（仅 q/kv projection 使用 FP8），避免 softmax 的 FP8 截断抹平 attention 分布；④ **梯度 scale 预热**：在 FP8 反向传播中引入 gradient scaling warmup，每 1000 step 递增 grad scale 上限，确保深层梯度在训练后期仍有足够动态范围 |
| **验证** | 方案①+③ 从 step 25000 续跑：layer 35 的 q_b_proj 输入 outlier ratio 从 8.7% 降至 0.6%，core_attn 输出 KL 散度从 2.31 bits 降至 0.18 bits，量化 SNR 恢复至 28.3 dB。loss 继续从 2.5 下降至 1.82（step 40000），grad_norm 稳定在 8~14。方案② 的 scale 保护机制使最小 scale 维持在 0.55 以上，有效 bit 数保持 ≥ 4.2 bit。整个训练至 step 50000 无退化 |


---

# 第二部分：Infra 定位链（基础设施视角）

> Infra 工程师从**硬件/资源**观测出发，按硬件拓扑下钻。与精度链在「通信原语 / 算子 kernel」层面汇合，最终收敛到配置或硬件变更。

```
集群层           WHERE    — 集群 / 节点 / GPU 级别定位
   │
资源层           WHAT     — 计算 / 显存 / 网络，哪类资源瓶颈
   │
   ├── 通信瓶颈 → 通信原语层  WHICH(comm) — all-reduce / all-to-all / p2p  ↕ 汇合精度链·通信调度层
   │                  │
   │                  └── 硬件层  HARDWARE — Xid / ECC / NVLINK 链路状态 / thermal
   │
   └── 计算瓶颈 → 算子/kernel层  WHICH(kernel) — CUDA kernel / block / occupancy  ↕ 汇合精度链·算子层
                       │
                       └── 硬件层  HARDWARE — SM error / clock throttle / register spill
                            │
                      配置变更层  FIX     — NCCL 参数 / GPU 功率 / 拓扑 affinity
```

---

## 1. 集群层 — 锁定「哪个节点 / 哪个 GPU」

| 项目 | 内容 |
|------|------|
| **定位目标** | 在多节点集群中，定位异常发生在哪个节点、哪些 GPU |
| **观测手段** | 节点级 GPU 利用率（nvidia-smi / DCGM）、MFU、显存占用率、节点功耗/温度仪表盘；按 node × GPU 展开的面板 |
| **判据** | 某节点/GPU 的利用率或 MFU 显著偏离集群均值（如其他节点 55%，异常节点 20%）；显存占用与其余节点不一致 |
| **产出** | 异常节点 + GPU 列表（如 node2 GPU 3~7） |

## 2. 资源层 — 锁定「计算 / 显存 / 网络」

| 项目 | 内容 |
|------|------|
| **定位目标** | 判断瓶颈类型：计算受限（SM 打满）、显存受限（HBM 带宽饱和/OOM）、网络受限（NCCL 带宽不足/链路降级） |
| **观测手段** | SM occupancy（nsys/ncu）、HBM bandwidth utilization（DCGM profiler）、NVLINK/IB throughput（nvidia-smi nvlink -s / ib_read_bw）、PCIe 带宽 |
| **判据** | SM occupancy < 30% + HBM 带宽正常 → 计算空闲（等数据）；NVLINK/IB throughput 骤降 → 网络瓶颈；显存接近上限且频繁 retry → 显存瓶颈 |
| **产出** | 瓶颈类型：计算 / 显存 / 网络 + 具体指标基线 vs 当前值 |

> ⚠️ 此处产生**分叉**：根据资源层定位的瓶颈类型——
> - **网络瓶颈** → 进入「通信原语层」
> - **计算瓶颈 / 显存瓶颈** → 进入「算子/kernel层」

---

## ── 分支A：通信原语层 ──

> 当资源层判定为网络瓶颈时进入。与精度链的「通信调度层」汇合。

| 项目 | 内容 |
|------|------|
| **定位目标** | 定位到具体的 NCCL 通信原语调用（all-reduce / all-to-all / p2p / broadcast）及其涉及的 rank |
| **观测手段** | NCCL trace（`NCCL_DEBUG=INFO`）、Nsight Systems 通信时间线（send/recv 配对、barrier 等待）、per-call bandwidth、NCCL topology 日志 |
| **判据** | 某次 all-to-all/all-reduce 调用耗时是其他调用的 5×以上；同一调用在不同 rank 上耗时差异 > 2×；NCCL 日志中出现 `[ERROR]` 或 `fallback to slow path` |
| **产出** | 异常通信原语 + 涉及的 rank 列表 + 耗时/带宽数据 |

### ↕ 汇合点：此处可与精度链「通信调度层」结论交叉验证

---

## ── 分支B：算子/kernel层 ──

> 当资源层判定为计算或显存瓶颈时进入。与精度链的「算子层/张量层」汇合。

| 项目 | 内容 |
|------|------|
| **定位目标** | 定位到具体的 CUDA kernel（如 `linear_fp8`、`flash_attn`、`rms_norm`），分析其执行效率 |
| **观测手段** | Nsight Compute（kernel occupancy、register spill、shared memory usage）、Nsight Systems kernel timeline（看 kernel launch gap、stream overlap） |
| **判据** | kernel occupancy < 50% → 计算资源利用不足；register spill > 128B/thread → 寄存器压力；kernel 之间有 > 100μs gap → stream 调度问题 |
| **产出** | 问题 kernel 名称 + occupancy / register spill / launch gap 数据 |

### ↕ 汇合点：此处可与精度链「算子层」结论交叉验证

---

## 3. 硬件层 — 锁定「硬件故障 / 降级」

| 项目 | 内容 |
|------|------|
| **定位目标** | 排除或确认硬件根因：GPU Xid 错误、ECC 内存纠错、NVLINK link 掉线、thermal throttle、GPU clock 降频 |
| **观测手段** | `dmesg` / `nvidia-smi -q`（Xid error、ECC count）、`nvidia-smi nvlink -e`（link 状态）、DCGM thermal/clock 指标、PCIe AER 日志 |
| **判据** | Xid 48（double-bit ECC）→ 显存硬件故障；NVLINK link inactive → 链路掉线需复位；GPU clock 持续低于 base clock → thermal throttle |
| **产出** | 硬件故障类型 + 故障 GPU serial/PCIe BDF + 是否需硬件更换或复位 |

## 4. 配置变更层 — 锁定「改什么」

| 项目 | 内容 |
|------|------|
| **定位目标** | 将 Infra 诊断结论映射到可执行的配置变更或硬件操作 |
| **观测手段** | 回溯：NCCL 环境变量（`NCCL_IB_TIMEOUT`、`NCCL_NET_GDR_LEVEL`、`NCCL_P2P_LEVEL`）、GPU 功率/时钟策略（`nvidia-smi -pl`、`nvidia-smi -ac`）、节点拓扑 affinity（NUMA binding、GPU-NIC affinity） |
| **判据** | 网络瓶颈且非硬件故障 → 调 NCCL 参数；硬件故障 → RMA 或节点下线；thermal throttle → 调功率上限或改善散热 |
| **产出** | 具体配置变更 + 变更文件/命令 + 验证方案 |


---

## Infra 案例：NVLINK 链路掉线导致 MFU 骤降

> **路径**：集群层 → 资源层 → 通信原语层 → 硬件层 → 配置变更层

**背景**：64 GPU（8 节点 × 8 GPU）训练 openPangu-2.0-Flash，EP=64，TP=1，PP=4，FP8 精度。训练至 step ~20000 后，总吞吐从 3200 tokens/s 掉至 1200 tokens/s，MFU 从 55% 降至 20%。

### 1. 集群层

| 步骤 | 内容 |
|------|------|
| **现象** | node2 的 8 个 GPU 利用率从 92% 整体掉至 35~40%，其余 7 个节点利用率正常（90~95%）。显存占用各节点一致（~78GB/80GB），排除 OOM |
| **判据** | 异常集中在 node2，其余节点正常 → node2 故障 |
| **产出** | 异常节点：node2（GPU 0~7） |

### 2. 资源层

| 步骤 | 内容 |
|------|------|
| **观测** | ① SM occupancy：node2 GPU 利用率虽低，但 SM occupancy 仍有 85%（GPU 在等数据而非空闲计算）；② HBM bandwidth：正常 1.2TB/s；③ **NVLINK throughput：node2 GPU 3→GPU 4 链路带宽从 45GB/s 降至 0.8GB/s**（其余链路 45GB/s 正常）；④ IB throughput：node2→node1 的 IB 带宽从 25GB/s 升至 48GB/s（接近 IB 上限），其余节点间 IB 负载正常 |
| **判据** | NVLINK 链路异常降级 + IB 负载异常升高 → NCCL 被迫从 NVLINK 高速路径回退到 IB 低速路径 |
| **产出** | 瓶颈类型：网络 / 嫌疑链路：node2 GPU3↔GPU4 NVLINK |

### 3. 通信原语层

| 步骤 | 内容 |
|------|------|
| **观测** | Nsight Systems trace 显示：layer 30~38 的 MoE all-to-all 调用在 node2 GPU 3 和 GPU 4 上耗时从正常的 2.3ms 飙升至 18.7ms（8× 恶化）。NCCL topology 日志显示 GPU 3↔GPU 4 间 `NCCL_P2P_LEVEL` 从 `PATH_NVL` 回退到 `PATH_SYS`（经 PCIe/IB）。该链路承载了 PP stage 2↔3 的跨 stage p2p 传输，导致整个 PP pipeline 被拖慢 |
| **判据** | all-to-all 耗时 8× + NVLINK→SYS 回退 → 通信路径降级 |
| **产出** | 异常原语：`all-to-all`（MoE dispatch+combine）+ PP `p2p` / 涉及 GPU：node2 GPU 3, GPU 4 / PP stage：2↔3 |

### 4. 硬件层

| 步骤 | 内容 |
|------|------|
| **观测** | `nvidia-smi nvlink -e` 显示 GPU 3 的 NVLINK lane 5 状态为 `Inactive`（其余 11 条 lane 正常）。`dmesg` 中无 Xid 报错（非致命硬件故障），但 NVLINK CRC error count 在 step 19800 附近突增（lane 5：0→10⁶）。GPU 温度和功耗正常，排除 thermal throttle |
| **判据** | 单条 NVLINK lane inactive + CRC 错误突增 → 物理链路降级（可能为线缆松动或 transceiver 老化） |
| **产出** | 硬件故障：node2 GPU 3 NVLINK lane 5 inactive / 建议：优先 reseat NVLINK bridge，若恢复失败则 RMA |

### 5. 配置变更层

| 步骤 | 内容 |
|------|------|
| **临时绕过** | ① `export NCCL_IGNORE_DISABLED_P2P=1` 允许 NCCL 自动跳过故障链路；② 将 node2 GPU 3 从 EP group 中临时排除（调整 EP=63），用其余 63 GPU 继续训练，吞吐恢复至 3000 tokens/s |
| **永久修复** | ① 停机维护 window 内 reseat node2 GPU3↔GPU4 的 NVLINK bridge；② 若 reseat 无效，更换 GPU 3；③ 监控脚本增加 NVLINK lane status + CRC error 的定时巡检（每 10min），阈值告警 |
| **验证** | 临时方案：EP=63 跑 5000 step，loss 曲线与 EP=64 无差异（EP 减少 1 rank 对 256 expert 分布影响 < 0.5%）；硬件修复后恢复 EP=64，NVLINK 全部 lane Active，MFU 回到 55% |


---

# 第三部分：性能定位链（吞吐 / 效率视角）

> 从训练吞吐、MFU、步耗时的**效率异常**出发，逐层下钻直至可执行的代码或配置变更。**瓶颈分类层后即分叉**：计算瓶颈走「算子/执行效率」分支，通信瓶颈走「通信原语/链路」分支。每一层有明确的**判据**决定下一步方向。

```
性能表征层    WHAT       — 吞吐 / MFU / 步耗时 异常 → 量化效率损失
   │
瓶颈分类层    CATEGORY   — 计算受限 / 通信受限 / 显存受限
   │
   ├── 计算受限 → 阶段定位层    WHERE(compute) — 哪个 PP stage / 模型层是瓶颈
   │                  │
   │                  └── 算子定位层   WHICH(op) — 哪个算子 / kernel 耗时最高
   │                        │
   │                        └── 执行效率层   WHY — 利用率低 / 气泡大 / tiling差 / AICPU回退 / 动态shape
   │                              │
   │                              └── 代码/配置层  FIX — 改 tiling / 算子融合 / 消除回退 / 调并行策略
   │
   ├── 通信受限 → 通信原语层    WHICH(comm) — all-reduce / all-to-all / p2p 哪个在关键路径
   │                  │
   │                  ├── 通信模式层   WHY(comm) — 小包占比高 / 等待暴露 / 带宽不足 / 链路降级
   │                  │
   │                  └── 代码/配置层  FIX — 通信重叠 / NCCL 参数 / 拓扑 affinity / 硬件修复
   │
   └── 显存受限 → 显存表征层    WHAT(mem) — 容量不足（OOM/near-OOM） / 带宽饱和 / 分配碎片
                      │
                      ├── 阶段定位层（汇入计算分支）— 哪个 PP stage / 层显存峰值最高
                      │
                      ├── 显存分配分析层  WHY(mem) — 峰值构成（参数/梯度/优化器/激活/临时buffer）/ 碎片化 / 换页频率
                      │
                      └── 代码/配置层  FIX — activation checkpoint / batch size / TP/PP调整 / 分配器调优 / 精度换空间

各层与已有链的汇合点：
  ↕ 阶段定位层 · 算子定位层     ↔ 精度链「模型层 · 算子层」
  ↕ 通信原语层                   ↔ 精度链「通信调度层」· Infra链「通信原语层」
  ↕ 执行效率层 · 显存分配分析层 ↔ Infra链「资源层 · 算子/kernel层」
```

---

## 1. 性能表征层 — 发现「有没有性能问题」

| 项目 | 内容 |
|------|------|
| **定位目标** | 从顶层效率指标判断训练是否存在性能瓶颈，量化损失幅度 |
| **观测手段** | 训练日志中的吞吐（tokens/s 或 samples/s）、单步耗时（ms/step）、MFU（Model FLOPS Utilization）；与理论峰值或同配置基线对比 |
| **判据** | MFU < 50%（大模型训练）或 < 30%（小模型/单机）→ 显著性能问题；吞吐较同配置基线低 > 15% → 需排查；步耗时波动 > 20%（相邻 step 间）→ 调度抖动 |
| **产出** | 性能异常的类型（吞吐低 / 步耗时长 / 波动大）+ 损失量化（当前 MFU vs 理论 MFU，差距 XX pp） |

> **性能健康评分（PHS）** 可作为综合量化参考：加权综合 MFU、算子利用率、气泡率、负载均衡度等指标，映射到 S/A/B+/B/C/D 六档。PHS ≤ C 时强制进入定位流程。

---

## 2. 瓶颈分类层 — 判定「计算 / 通信 / 显存」

| 项目 | 内容 |
|------|------|
| **定位目标** | 判定性能瓶颈属于**计算受限**、**通信受限**还是**显存受限**，决定后续走哪个分支 |
| **观测手段** | Profiling 报告中的算子耗时占比（计算 vs 通信 vs 空闲）；关键路径比（critical path ratio）；SM/Core 占用率；通信带宽利用率；显存占用率与 HBM 带宽利用率 |
| **判据** | 算子计算耗时占总步耗时 > 60% → **计算受限**（走计算分支）；通信等待 / all-to-all / all-reduce 在关键路径占比 > 25% → **通信受限**（走通信分支）；显存峰值 > 90% 总容量，或显存占用率 < 80% 但 HBM 带宽利用率 > 85%（带宽饱和），或伴有频繁 HBM 物理页分配/释放 → **显存受限**（走显存分支） |
| **产出** | 瓶颈类型 + 各类型的量化数据（时间占比或显存占比） |

> ⚠️ 此处产生**关键分叉**：
> - **计算受限** → 进入「阶段定位层」→ 沿计算主干
> - **通信受限** → 进入「通信原语层」→ 沿通信分支
> - **显存受限** → 进入「显存表征层」→ 沿显存分支（先判定容量/带宽/分配子类型，再汇入阶段定位层）

---

## 3. 阶段定位层 — 锁定「哪个 PP stage / 模型层」

| 项目 | 内容 |
|------|------|
| **定位目标** | 在多 stage pipeline 中定位到瓶颈 stage，或在单 stage 内定位到具体模型层 |
| **观测手段** | Pipeline 时序泳道图（swimlane）：横轴为时间、纵轴为各 stage，看哪个 stage 的计算段最长（成为 pipeline bubble 的源头）；层粒度算子耗时排序（parent-child operator table）：按总耗时降序排列各层的 attention / FFN / MoE |
| **判据** | Pipeline 泳道图中某 stage 的计算耗时是其他 stage 的 1.3× 以上 → 该 stage 是 pipeline bottleneck；层耗时排序中某层耗时占该 stage 总耗时 > 25% → 该层是热点层 |
| **产出** | 瓶颈 stage（如 PP stage 3 耗时 401ms vs stage 0 的 231ms）+ 热点层（如 layer 42 MoE） |

### ↕ 汇合点：此处可与精度链「模型层」结论交叉验证

---

## 4. 算子定位层 — 锁定「哪个算子 / kernel」

| 项目 | 内容 |
|------|------|
| **定位目标** | 在瓶颈层内，定位到耗时最高的具体算子 |
| **观测手段** | 算子 parent-child 表（按总耗时、调用次数、平均耗时三维排序）；算子分类饼图（按 Accelerator Core：AI_CORE / AI_VECTOR_CORE / AI_CPU 分拆）；kernel 级 timeline（Nsight / CANN profiler kernel_details.csv） |
| **判据** | 某算子总耗时占该层 > 40%，或其单次耗时同类算子的 3× 以上 → 首害算子；若 AI_CPU 算子耗时占比 > 10% → AICPU 回退问题（见执行效率层） |
| **产出** | 问题算子路径（如 `lm_head Linear [2560→151552]`、`q_b_proj Linear [1024→9216]`、`router top-k`）+ 耗时/占比数据 + 所属 Accelerator Core 类型 |

### ↕ 汇合点：此处可与精度链「算子层」结论交叉验证

---

## 5. 执行效率层 — 诊断「为什么慢」

> 算子定位层找到「谁慢」之后，此层回答「为什么慢」。根据 MindStudioNext 分析报告中的问题分类，常见根因如下。同一算子可能命中多项。

### 5.1 核心利用率低（SM / Cube / Vector 空闲）

| 项目 | 内容 |
|------|------|
| **现象** | 算子总耗时高，但 NPU/GPU 核心在执行期间的占用率远低于 100% |
| **观测手段** | 泳道图中的气泡（bubble）段：同一核心上相邻任务间的空白间隙；kernel occupancy 指标；算子执行时的 PMU 计数器 |
| **判据** | 核心利用率 < 50% → 严重空闲；气泡率 > 20%（气泡时间 / 总时间）→ 调度效率低；任务粒度太细（单 task < 10μs）→ launch overhead 主导 |
| **常见根因** | ① tiling 过小导致计算无法填满计算单元；② 算子间依赖未打破，流水线停顿；③ host 侧下发速度跟不上 device 消费速度；④ 算子未做 double/triple buffer |

### 5.2 Pipeline 气泡（PP bubble）

| 项目 | 内容 |
|------|------|
| **现象** | Pipeline 不同 stage 间计算负载不均，轻 stage 等待重 stage，形成空白段 |
| **观测手段** | Pipeline 时序泳道图：同一时刻各 stage 的「计算 / 通信 / 空闲」状态堆叠；PP bubble ratio = 空闲时间 / 总步时间 |
| **判据** | 某 stage 计算耗时是其他 stage 的 1.5× 以上 → stage 不均衡；PP bubble > 15% → 值得优化 |
| **常见根因** | ① 末级 stage 含 lm_head + loss，天然重（常见 1.5×~2× 其他 stage）；② embedding 层集中在首级；③ 各 stage 分配的层数均等但层计算量不均（MoE 层 vs Dense 层） |

### 5.3 Tiling 配置不当

| 项目 | 内容 |
|------|------|
| **现象** | 单个算子计算耗时远超其计算量理论值（即 MFU 在该算子上特别低） |
| **观测手段** | 算子 tile shape（从 kernel launch 参数提取）；Roofline 分析：算子实际 FLOPS vs 理论峰值，判断是 compute-bound 还是 memory-bound |
| **判据** | 算子在 Roofline 图中落在 memory-bound 区域 → tile 太小，访存主导；cube 算子的 tile_m < 128 → 未充分利用 Cube 单元；vector 算子的 tile 过小 → 向量化不足 |
| **常见根因** | ① 默认 tiling 策略不适合当前 shape（如 vocab 151936 未对齐到 128 倍数）；② 动态 shape 导致每次重新 tiling 且无法命中编译缓存；③ 多分支 shape 不一致导致最小公约数 tile |

### 5.4 AICPU / AICORE 回退

| 项目 | 内容 |
|------|------|
| **现象** | 某算子标注为 AI_CPU 或 MIX_AIC，未跑在 AI_CORE 或 AI_VECTOR_CORE 上 |
| **观测手段** | 算子分类表按 Accelerator Core 分拆；kernel_details.csv 中 Core Type 列 |
| **判据** | AI_CPU 算子耗时占比 > 10% → 显著回退；MIX_AIC 算子耗时占比 > 15% → Cube 融合未生效 |
| **常见根因** | ① 算子的 shape/dtype 不在 CANN 融合白名单中（如非标 hidden dim）；② 手写的算子组合（如 softmax+log+gather）替代了融合算子（如 F.cross_entropy）；③ IndexPut / nonzero 等操作天然走 AI_CPU |

### 5.5 动态 Shape / JIT 编译开销

| 项目 | 内容 |
|------|------|
| **现象** | 步耗时波动大、设备占用率周期性骤降、CANN_API 调用次数异常高 |
| **观测手段** | 步耗时时间序列（看是否有周期性尖峰）；Free Analysis 空闲段（两轮 launch 间的 gap）；CANN API 统计中的 aclopCompileAndExecute 调用次数与耗时 |
| **判据** | 相邻 step 耗时波动 > 20% → 疑似动态编译；aclopCompileAndExecute 耗时 > 总步耗时的 5% → JIT 开销显著；Free Analysis 中有周期性 > 5ms 的 host-idle 段 |
| **常见根因** | ① vLLM/SGLang 中 enforce_eager=True 未启用 graph capture；② nonzero / unique 等算子产生动态 shape，无法图捕获；③ 训练中 jit_compile 未关闭 |

### 5.6 通信等待暴露在关键路径

| 项目 | 内容 |
|------|------|
| **现象** | 通信耗时完全落在步的关键路径上，未被计算掩盖 |
| **观测手段** | 通信视图的 Transit/Wait 瀑布图（看通信传输时间 vs 等待时间）；communicationOverlapComputation 指标（>0 表示有重叠，=0 表示完全暴露） |
| **判据** | communicationOverlapComputation = 0 且通信耗时 > 总步耗时的 15% → 通信暴露；同一通信原语在不同 rank 上完成时间差 > 2× → 负载不均导致 straggler |
| **常见根因** | ① 未开启通信-计算重叠（overlap-grad-reduce / overlap-param-gather 未设）；② 通信与计算共用同一 stream 导致串行；③ DP/TP/EP 的通信调度策略未针对拓扑优化 |

### 5.7 小包通信 / RDMA 效率低

| 项目 | 内容 |
|------|------|
| **现象** | 通信带宽利用率低（实际带宽远低于 IB/NVLINK 理论值） |
| **观测手段** | RDMA 小包占比环图（< 4KB 的包占比）；带宽达成率热力图（rank × 通信类型，达成率 = 实测/理论）；链路带宽达成率条形图 |
| **判据** | 小包（< 4KB）占比 > 30% → RDMA 效率低；某 rank 的带宽达成率 < 50% → 链路异常或消息粒度问题 |
| **常见根因** | ① TP size 小导致 all-reduce 消息碎片化；② MoE expert 分布在少量 rank 上但 token 粒度太细；③ NCCL 未启用 GDR（GPU Direct RDMA） |

---

## ── 分支C：显存受限 ──

> 当瓶颈分类层判定为显存受限时进入。显存问题有三种子类型，**显存表征层**先判定子类型再决定后续路径：**容量不足**走缩减路径，**带宽饱和**汇入计算分支的算子/执行效率层，**分配碎片**走分配器调优路径。

### C.1 显存表征层 — 判定「容量不足 / 带宽饱和 / 分配碎片」

| 项目 | 内容 |
|------|------|
| **定位目标** | 区分显存瓶颈的子类型：① 绝对容量不足（OOM 或 near-OOM）；② HBM 带宽饱和（算子在 Roofline 的 memory-bound 区）；③ 分配器碎片/频繁换页（有空间但不能高效使用） |
| **观测手段** | `nvidia-smi` / `ascend-dmi` 显存占用率时间序列；Profiling 报告的显存峰值构成拆解（参数 / 梯度 / 优化器状态 / 激活 / 临时 buffer）；Roofline 图中各算子的位置；显存分配/释放 API 的调用频率与耗时 |
| **判据** | 显存占用 > 95% 且伴有 OOM 或 allocation retry → **容量不足**；显存占用 60~85% 但 HBM 带宽利用率 > 85%，算子在 Roofline 中落在 memory-bound 区 → **带宽饱和**；显存占用 70~90% 但存在频繁的 aclrtFreePhysical/aclrtMalloc 调用（每步 > 50 次），或可用空间足够但分配耗时占总步耗时 > 3% → **分配碎片** |
| **产出** | 显存瓶颈子类型 + 量化指标（峰值占用 / 带宽利用率 / 分配调用次数） |

> ⚠️ 此处产生**显存子分叉**：
> - **容量不足** → 进入「显存峰值构成分析层」→ 缩减显存占用
> - **带宽饱和** → 汇入计算分支「阶段定位层」（定位到具体层/算子），再沿「执行效率层 §5.3 Tiling 配置不当」优化
> - **分配碎片** → 进入「代码/配置层」的显存分配器调优

### C.2 显存峰值构成分析层 — 锁定「什么占满了显存」

| 项目 | 内容 |
|------|------|
| **定位目标** | 将显存峰值拆解到具体构成：模型参数、梯度、优化器状态（AdamW 的 m/v）、激活值（含中间张量）、临时 workspace buffer、KV Cache（推理场景）、框架开销 |
| **观测手段** | 显存快照（memory snapshot / `torch.cuda.memory_stats` / NPU `memory_record`）；各 PP stage 的显存使用拆解（参数 × 层数 + 激活 × batch × seq_len）；PyTorch/CANN 的 memory allocator 统计 |
| **判据** | 激活值占比 > 40% 总显存 → activation checkpointing（重计算换空间）；优化器状态占比 > 30% → 可考虑 BF16 优化器状态（openPangu-2.0-Flash 方案）；临时 buffer 峰值 > 10% → 算子 workspace 未复用或过大；某个 PP stage 显存远超其余 stage → PP 层数/模型切分不均 |
| **产出** | 显存构成饼图 + 最大占比项 + 建议缩减方向 |

### C.3 阶段/层定位层（汇入计算分支）— 锁定「哪个 stage/层显存压力最大」

| 项目 | 内容 |
|------|------|
| **定位目标** | 定位到显存峰值最高或带宽最紧张的 PP stage / 模型层 |
| **观测手段** | 按 PP stage 拆解的显存使用热力图；按层拆解的激活值大小排序（attention 中间张量 vs FFN 中间张量） |
| **判据** | 某 stage 的显存峰值是其他 stage 的 1.3× 以上 → 该 stage 显存不均衡；某层的激活值大小超过其余层均值的 2× → 该层是热点（常见于 attention 的 QKV 展开、MoE 的 expert dispatch buffer） |
| **产出** | 瓶颈 stage + 热点层 + 该层的显存构成明细 |

### C.4 代码/配置层（显存分支）— 锁定「怎么省显存」

| 项目 | 内容 |
|------|------|
| **定位目标** | 将显存分析结论映射到可执行的显存优化操作 |
| **观测手段** | 回溯：训练启动参数（global_batch_size / micro_batch_size / seq_len）、并行策略（TP/PP/EP/DP）、activation checkpointing 配置、优化器精度配置、显存分配器环境变量 |
| **判据** | 根据显存表征层和峰值构成分析层的结论匹配修改项（见下表） |
| **产出** | 具体修改项 + 预期显存缩减量 + 对吞吐的影响评估 + 验证方案 |

| 显存根因 | 修改方向 | 典型显存缩减 | 吞吐代价 |
|---|---|---|---|
| 激活值占比过高 | 开启 activation checkpointing（选择性重计算，如 full / selective） | −30~50% 激活显存 | +5~15% 计算时间 |
| 激活值占比过高（深究） | 减小 micro_batch_size（增大 gradient_accumulation_steps 补偿） | −20~40% 激活显存 | 吞吐基本不变 |
| 优化器状态过大 | BF16 优化器状态替代 FP32（openPangu-2.0-Flash 方案） | −50% 优化器显存 | 精度几乎无损 |
| 参数显存过大 | 增大 TP/PP size（分摊参数到更多卡） | 线性分摊 | 增加通信开销 |
| 某 PP stage 显存不均 | 调整各 stage 层数分配，重 stage 少放层 | 均衡化 | bubble 可能略增 |
| 临时 buffer 峰值大 | 算子 workspace 复用（CANN 默认有一定复用，检查是否有 bypass 逻辑） | −5~15% 峰值 | 无 |
| 分配碎片 / 频繁换页 | `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` + 增大缓存保留 | 减少 alloc/free 耗时 | 无 |
| 推理 KV Cache 过大 | 启用 MLA（低秩 KV 压缩）、prefix caching、KV Cache 量化（FP8/INT8） | −50~80% KV Cache | 精度轻微损失 |


---

## 6. 代码/配置层 — 锁定「改什么」（计算/通信分支汇总）

| 项目 | 内容 |
|------|------|
| **定位目标** | 将执行效率诊断结论映射到具体可执行的代码修改、配置变更或环境变量调整 |
| **观测手段** | 回溯：模型代码（算子实现、并行策略配置）、训练启动参数、CANN/torch_npu 环境变量 |
| **判据** | 根据上层的根因分类匹配修改项（见下表） |
| **产出** | 具体修改项 + 修改文件/参数路径 + 预期收益 + 验证方案 |

| 执行效率层根因 | 修改方向 | 典型收益 |
|---|---|---|
| 核心利用率低 | 增大 tiling、double buffer、loop unroll | CUBE 利用率 +15~25pp |
| PP bubble | 调整 PP stage 层数分配（decoder-last-pipeline-num-layers） | bubble 率 −5~15pp |
| Tiling 不当 | 对齐 shape 到硬件友好粒度（vocab 对齐 128×）、手工指定 cube/vec tile size | 单算子耗时 −30~60% |
| AICPU 回退 | 替换为融合算子（cross_entropy / fused layernorm）、shape 对齐以命中白名单 | AI_CPU 耗时 −80~100% |
| 动态 Shape | 关闭 JIT compile、启用 graph capture、nonzero→topk 消除动态 shape | 步耗时波动 −80%+ |
| 通信暴露 | 开启通信-计算重叠（overlap-grad-reduce / overlap-param-gather）、interleaved 1F1B | 暴露通信占比 −5~15pp |
| 小包通信 | 增大 TP/EP size 减少碎片、开启 GDR、调整 NCCL buffer size | 通信带宽利用率 +20~40pp |
| 显存碎片 | 调大 PYTORCH_NPU_ALLOC_CONF expandable_segments、ACLNN_CACHE_LIMIT | 显存分配耗时 −50~80% |


---

## 案例一：计算分支 — 整网耗时下钻到算子带宽瓶颈 + AICPU 回退导致步耗时超标 40%

> **路径**：性能表征层 → 瓶颈分类层（计算受限）→ 阶段定位层 → 算子定位层 → 执行效率层 → 代码/配置层

**背景**：128 GPU（16 节点 × 8 GPU）训练 openPangu-2.0-Flash，EP=64，PP=4，TP=1，FP8 精度，seq_len=4096，micro_batch=1，global_batch=1024。目标步耗时 ≤ 8.5s，实测步耗时 12.1s，超标 42%。MFU 约 38%，目标 MFU ≥ 50%。

### 1. 性能表征层

| 步骤 | 内容 |
|------|------|
| **现象** | 步耗时 T_iter ≈ 12.1s，openPangu-2.0-Flash 报告等效配置下预期约 8.0~8.5s（每 T token 180K GPU·h ÷ 2048 GPU → 每步理论下限约 7.2s，加调度开销约 8.5s）。MFU 仅 38%，PHS 评分 D（32） |
| **判据** | 步耗时超基线 > 40% + MFU < 40% → 显著性能劣化，非调度抖动 |
| **产出** | 性能异常：T_iter=12100ms / MFU=38% / PHS=D（32）/ 超标幅度：+3600ms（+42%） |

### 2. 瓶颈分类层

| 步骤 | 内容 |
|------|------|
| **观测** | 从 profile_dir 的 `step_trace_time.csv` 提取：Computing=9422ms（78%），Communication(Not Overlapped)=1093ms（9%），Free=248ms（2%），Bubble（profiler 未计算，需手算≈1337ms≈11%）。通信占比不高，关键路径上计算绝对主导 |
| **判据** | 计算耗时占 78% > 60% → **计算受限，走计算分支**。通信（1093ms）虽有一定占比，但非主因（优先处理计算侧后再回头优化通信） |
| **产出** | 瓶颈类型：计算受限 / Computing 9422ms（78%）/ 未掩盖通信 1093ms（9%）/ Free 248ms（2%） |

### 3. 阶段定位层

| 步骤 | 内容 |
|------|------|
| **观测** | Pipeline 时序泳道图（Timeline trace，横轴时间、纵轴 PP stage 0~3 的 Computing / Comm / Free 堆叠）。Stage 3（末级，layers 34~45 + final_layernorm + lm_head + loss）Computing 段 2152ms，是中间 stage（stage 1~2 各约 1180ms）的 **1.82×**。Stage 0（embedding + L0~L11）Computing 段 1310ms（首级 embedding 额外开销）。Stage 1~2 在 stage 3 的计算段期间显示为空等（PP bubble） |
| **判据** | Stage 3 耗时 / 中间 stage 均值 = 2152/1180 ≈ 1.82× → 末级严重过载；PP bubble 估算 ≈ 12%（≈620ms），与 PP=4、1F1B 调度下末级含 lm_head+loss 的典型 bubble 范围一致 |
| **产出** | 瓶颈 stage：PP stage 3（Computing 2152ms）/ PP bubble ≈ 620ms / 中间 stage 均值 ≈ 1180ms |

### 4. 算子定位层

| 步骤 | 内容 |
|------|------|
| **观测** | Stage 3 算子 parent-child 表（`op_statistic.csv` 按总耗时降序）： |
|  | ① `MatMulV3`（all layers 的 attention QKV/o_proj + MLP fc1/fc2，合计 count=4928+448）：**5936ms**（占 stage 3 Computing 的 63%），其中 fc1 MatMul [4096,2560]×[16384,2560] shape 最大——openPangu-2.0-Flash 的 MoE expert hidden=1024，fc1 shape 为 gate_up 合并 [4096,2560]×[16384,2560]（gate+up 合并 2×1024×8 experts → 等效 16384），远小于 dense 模型的 37888 |
|  | ② `FlashAttentionScore` + Grad（MLA 的 core_attention，count=448+448）：**723ms**（7.7%） |
|  | ③ **AICPU 算子汇总**（`ArgMaxWithValue` 72ms + `Exp` 128ms + `RealDiv` 125ms + `Sub` 126ms + `ReduceSum` 75ms + `ApplyAdamWV2` 52ms + `LpNormV2` 12ms + 杂项 ≈90ms）：合计 **≈680ms**（7.2%），集中在 CE loss 与优化器 |
|  | ④ **RmsNorm** ×（46 层 × 2）= 92 次 fwd+bwd：**≈60ms**（0.6%） |
|  | ⑤ 残差/杂项 Add+Mul+Cast+Slice+… ≈ **500ms**（5.3%） |
| **判据** | MatMul 耗时 5936ms 占 63% ——但这是「总耗时」不是 FLOPS 效率低。进一步检查 `kernel_details.csv` 中 MatMulV3 的 cube 利用率：stage 0~2 的 MatMul cube_util ≈ 78~82%，而 **stage 3 的 lm_head MatMulV2（[4096,2560]×[151552,2560]）cube_util 仅 49%**（vocab 151552 = 592×256，但 151552 不是 256 的整数倍约束的实际对齐粒度导致尾块浪费 → **带宽瓶颈**）。另外 AICPU 的 680ms（7.2%）中 CE loss 链路 526ms 是手写算子组合而非融合实现 |
| **产出** | 问题算子：① `lm_head MatMulV2`（vocab 非对齐 → 带宽瓶颈，cube_util 49% vs 预期 75%）；② `CE loss 链路`（Exp/Sub/RealDiv/ArgMax/ReduceSum，526ms，AICPU 回退）；③ `ApplyAdamWV2`（52ms，优化器 vector 算子） |

### 5. 执行效率层

| 步骤 | 内容 |
|------|------|
| **观测** | Timeline trace（swimlane）放大 stage 3 的算子时序： |
|  | ① **lm_head 带宽瓶颈**：`kernel_details` 中 lm_head MatMulV2 的 `cube_util=49%`，dram_bytes ≈ 输入 4096×2560×2 + 权重 151552×2560×2 + 输出 4096×151552×2 ≈ 2.0GB。理论 HBM 带宽 3.35TB/s（H800），纯带宽耗时 ≈ 0.60ms——但实测 8.5ms/call × 64 microbatch = 544ms。Roofline 分析：该 MatMul 落在 **memory-bound 区**。根因：vocab=151552，H800 的 Tensor Core 要求 K 维对齐到 128 的倍数以便 FP8 tile 分块；151552/128=1184 整除，但 N 维（151552）不是 256 的倍数（151552/256=592，恰好整除 256），真正问题出在 vocab 维的 tile 粒度与 hidden=2560 的 K 维不匹配，导致 GEMM 尾块效率折半 |
|  | ② **CE loss AICPU 链路**：Timeline 中可见 5 个 vector 算子在 lm_head 后串行排列（Exp→Sub→RealDiv→ReduceSum→ArgMax），每个算子间有 60~100μs 的 launch gap，合计约 526ms。若替换为 `F.cross_entropy`（路由到 `aclnnSoftmaxCrossEntropyWithLogits` 融合实现），融合后仅 1 个 kernel，消除 5 次 HBM 读写往返 |
|  | ③ **优化器 vector 算子**：ApplyAdamWV2 和 LpNormV2（grad clip）各占 52ms 和 12ms，属正常范围（BF16 优化器状态下已减半），不构成独立瓶颈 |
| **判据** | lm_head → 带宽瓶颈（memory-bound），vocab 尾块未对齐 256 导致 cube_util 折半；CE loss → 手写算子链 × AICPU 回退；其余算子（MatMul FA RmsNorm）正常 |
| **产出** | 根因归类：lm_head MatMulV2 → 带宽瓶颈/tiling 对齐；loss 链路 → AICPU 回退。合计可优化 ≈750ms |

### 6. 代码/配置层

| 步骤 | 内容 |
|------|------|
| **修改 ① — vocab 对齐到 64 的倍数** | 当前 vocab=151552 已对齐到 256（151552/256=592 整除），但 K 维 hidden=2560 不是 128 的整数倍对 tile 划分不够友好。将 `hidden_dim` 对齐到 `make-hidden-size-divisible-by 256`（2560 已是 256×10，无需改动），重点调整 lm_head 的 tiling 策略使其针对 vocab=151552 的特定 shape 做手工调优。预期 lm_head cube_util 49% → 72%，耗时 544ms → 约 370ms（↓174ms） |
| **修改 ② — CE loss 融合** | 将手写的 softmax+CE 替换为 `F.cross_entropy(logits, labels, ignore_index=pad_id)`。CANN 路由到 `aclnnSoftmaxCrossEntropyWithLogits` 融合算子，消除 Exp/Sub/RealDiv/ReduceSum/ArgMax 5 段 AICPU/Vector 链路，融合后单 kernel 约 15ms × 64 microbatch = 约 10ms（注意：融合算子内部仍要做 softmax+CE，但消除了中间张量物化和 launch gap）。预期 526ms → 约 50ms（↓476ms） |
| **修改 ③ — 优化器 BF16 状态** | 当前已启用（openPangu-2.0-Flash 默认），无需修改。若未启用则设置 `--optimizer-cpu-offload` 或在代码中 `optimizer_config={'adam_beta1': 0.9, 'adam_beta2': 0.95, 'use_distributed_optimizer': True}` |
| **验证** | ①+② 叠加：T_iter 12100ms → 约 10350ms（↓14.5%），MFU 38% → 约 45%，lm_head cube_util 49% → 72%，AICPU 耗时占比 7.2% → 约 1.5%。PHS 从 D(32) → C+(55)。若要进一步达到 50% MFU，需配合案例二的 PP bubble 与通信重叠优化 |


---

## 案例二：计算分支 — MFU 拆解下钻到 PP 空泡，末级瓶颈拉低整网 MFU 至 38%

> **路径**：性能表征层 → 瓶颈分类层（计算受限）→ MFU 拆解（达成率 × 占用率）→ 阶段定位层 → 执行效率层（PP bubble）→ 代码/配置层

**背景**：同案例一的 128 GPU openPangu-2.0-Flash 训练，EP=64，PP=4，TP=1，FP8，seq_len=4096。案例一已修复 lm_head 带宽瓶颈和 CE loss AICPU 回退，T_iter 从 12.1s 降至 10.35s，MFU 从 38% 升至 45%。但距离目标 50%+ 仍有差距，需进一步下钻。

### 1. 性能表征层

| 步骤 | 内容 |
|------|------|
| **现象** | 案例一修复后 T_iter≈10350ms，MFU≈45%。同配置 openPangu-2.0-Flash 报告 MFU 可达 52~55%（等效配置 180K GPU·h/T token → 理论 T_iter≈7200ms，加 PP=4 调度开销约 +15% → 预期 8300ms）。差距仍有约 2050ms |
| **判据** | MFU 45% < 50% 目标 → 继续下钻。但此时算子层面已无明显异常（cube_util 正常、无 AICPU 回退），需从 MFU 公式拆解入手 |
| **产出** | 残余差距：T_iter 10350ms vs 目标 8300ms（+2050ms）/ MFU 45% vs 目标 52%（−7pp） |

### 2. MFU 拆解（替代瓶颈分类层的「二次下钻」）

| 步骤 | 内容 |
|------|------|
| **公式** | $\text{MFU} = \underbrace{\frac{\text{模型 FLOPs}}{T_{\text{compute}} \times \text{峰值 FLOPS}}}_{\text{算子达成率}} \times \underbrace{\frac{T_{\text{compute}}}{T_{\text{iter}}}}_{\text{计算占用率}}$ |
| **观测** | ① **算子达成率**：profile_dir 中 MatMulV3 的 cube_util 均值 ≈ 72%（stage 0~2 约 78%，stage 3 约 55%——lm_head 拉低均值）。FA（FlashAttentionScore）的 cube_util ≈ 65%（MLA 的 48-head × 192-dim 非标准 shape，Ascend 上 FA 实现未完全优化）。加权后达成率 ≈ **68%** |
|  | ② **计算占用率**：T_compute = Computing 中真正计入 FLOPs 的部分 ≈ CUBE(total MatMul) + FA = 5936 + 723 = 6659ms（剔除 AICPU/Vector/Free）。T_iter = 10350ms。计算占用率 = 6659/10350 ≈ **64.3%** |
|  | ③ 代入：MFU = 68% × 64.3% ≈ **43.7%**（与实测 45% 接近，误差来自 FA FLOPs 折算系数） |
| **判据** | 计算占用率仅 64.3% → **35.7% 的 T_iter 被"陪跑项"吃掉**。陪跑项构成：PP bubble（~620ms，6.0%）、未掩盖通信（1093ms，10.6%）、Vector/AICPU 零碎算子（~850ms，8.2%）、Free（248ms，2.4%）、误差（~889ms，8.6%）。**未掩盖通信是最大单因子**（PP=4 下 bubble 已大幅缩减） |
| **产出** | MFU 拆解结论：算子达成率 68% 正常（openPangu-2.0-Flash FP8 报告约 70~75%，差值来自 H800 vs Ascend 差异），计算占用率 64.3% 偏低，PP=4 下 bubble 仅 6%，主要陪跑项为未掩盖通信和 Vector/AICPU 零碎算子 |

### 3. 阶段定位层

| 步骤 | 内容 |
|------|------|
| **观测** | Timeline trace 泳道图（PP pipeline，4 stage × 1F1B 调度，横轴时间，纵轴 stage 0~3，颜色区分 Fwd-Compute / Bwd-Compute / P2P-Send/Recv / DP-Collective / Optimizer / PP-Bubble）。关键观测： |
|  | • stage 3（末级）Fwd-Compute 段 2152ms，Bwd-Compute 段在 warm-up 和 cool-down 阶段与其余 stage 不同步 |
|  | • stage 0~2 在 stage 3 的 Fwd-Compute 期间显示为空等（红色 PP-Bubble），累计 bubble = stage 0~2 各约 200~250ms |
|  | • **Bubble 的根因**：末级 stage 3 额外扛了 final_layernorm（RmsNorm）+ lm_head（MatMulV2）+ CE loss，合计约 1037ms 额外负载。按 1F1B 调度，这 1037ms 在 warm-up 阶段造成所有上游 stage 空等、在 steady 阶段每 microbatch 造成 1 个 bubble 周期、在 cool-down 阶段下游空等。PP=4 下仅 3 个 stage 间有 bubble，总量大幅低于典型 PP=8 配置 |
| **判据** | PP bubble ≈ 620ms（占 T_iter 10350ms 的 6.0%），PP=4 下已大幅缩减，不再是最大陪跑项。Bubble 的源头是末级 stage 3 的 lm_head+loss 额外负载 + PP=4 均分 46 层导致末级层数偏多 |
| **产出** | 瓶颈根因：PP bubble 6.0% / 末级额外负载 1037ms / 末级 12 层偏多 |

### 4. 执行效率层（PP bubble 深度分析）

| 步骤 | 内容 |
|------|------|
| **观测** | 进一步量化 PP bubble 的构成（基于 1F1B 调度公式）： |
|  | • 理论 bubble = (PP−1) × (T_fwd_stage3 − T_fwd_stage_avg) / microbatch_count。PP=4, microbatch=64, T_fwd_stage3≈1720ms, T_fwd_stage_avg≈780ms → 单 microbatch bubble ≈ (4−1)×(1720−780)/64 ≈ 44ms。64 microbatch 累计 ≈ 2816ms，但 1F1B 的 warm-up/steady/cool-down 三段中仅 warm-up 和 cool-down 产生 bubble，steady 段无 bubble → 实际 bubble ≈ 2816 × (warmup+cool-down microbatches / total) ≈ 620ms（与 trace 观测一致） |
|  | • 若能将 stage 3 的计算量减少 500ms（如 lm_head 优化），则 T_fwd_stage3→1220ms，bubble 减至约 350ms（↓43%） |
|  | • PP=4 已是最小 pipeline 深度（再减到 PP=1 则无 pipeline），优化重点应放在减小末级负载和通信重叠 |
| **判据** | PP bubble 可优化空间约 270~350ms（通过减小末级负载 + 调整层数分配 + 通信重叠） |
| **产出** | 优化方向：减小 stage 3 负载 + 调整 PP 层数分配 + 通信-计算重叠 |

### 5. 代码/配置层

| 步骤 | 内容 |
|------|------|
| **修改 ① — 调整 PP stage 层数分配** | 当前 PP=4 均分 46 层为 [12,11,11,12]（末级 12 层 + lm_head）。末级 stage 3 天然偏重（含 lm_head+loss），调整为 [13,11,11,11]（stage 0 多 1 层补偿 embedding 开销，stage 3 减 1 层减压）。Megatron 参数：`--decoder-first-pipeline-num-layers 13 --decoder-last-pipeline-num-layers 11`。每减 1 层减少约 118ms，预期 stage 3 计算时间 2152ms → 约 2034ms（↓118ms），bubble 减少约 40ms |
|  | 更激进的方案：将 stage 3 再减 1 层到 10 层 [13,12,11,10]，但 stage 3 仅 10 层可能显存压力过大。保守方案先减 1 层观察效果。 |
| **修改 ② — 通信-计算重叠** | PP=4 下 bubble 仅 6%，未掩盖通信（1093ms, 10.6%）成为更大陪跑项。开启 `--overlap-grad-reduce` 和 `--overlap-param-gather`，将 all-reduce 与反向计算重叠，预期未掩盖通信 1093ms → 约 400ms（↓63%） |
| **修改 ③ — Interleaved 1F1B（可选更大改动）** | 启用 `--num-layers-per-virtual-pipeline-stage 2`，将每个 PP stage 内的约 12 层再切成 6 个 virtual stage（每 virtual 2 层），用 interleaved 1F1B 调度。Bubble 从 (PP−1)/(microbatch) 降至 (VPP−1)/(microbatch)，预期 bubble 减少约 50% |
| **验证** | 修改①（减 1 层）+ 修改②（通信重叠）：T_iter 10350ms → 约 9650ms（↓6.8%），PP bubble 620ms → 约 580ms，未掩盖通信 1093ms → 约 400ms。MFU 45% → 约 48%。PHS 从 C+(55) → B(65)。若同时实施修改③，T_iter 可达约 9100ms，MFU 可达 51% |


---

## 案例三：通信分支 — MoE all-to-all 快慢卡导致步耗时周期性尖峰、尾延迟恶化

> **路径**：性能表征层 → 瓶颈分类层（通信受限）→ 通信原语层 → 通信模式层 → 执行效率层（负载倾斜）→ 代码/配置层

**背景**：128 GPU（16 节点 × 8 GPU）训练 openPangu-2.0-Flash，EP=64，PP=4，TP=1，FP8，seq_len=4096。步耗时均值约 10.3s，但标准差达 2.8s（CV=27%）。每 8~12 步出现一次步耗时 > 15s 的尖峰，对应 Timeline trace 中某几个 rank 的 all-to-all 通信耗时暴增。

### 1. 性能表征层

| 步骤 | 内容 |
|------|------|
| **现象** | 步耗时时间序列（按 global_step 展开的折线图）：均值 10.3s，但呈周期性尖峰模式——每 8~12 步出现一次 >15s 的慢步，最慢步 18.4s。正常步（10.0±0.5s）与慢步（15~18s）交替出现。正常步的 MFU 约 45%，慢步的 MFU 仅 26% |
| **判据** | 步耗时波动 CV=27%（> 20% 阈值）+ 周期性尖峰模式 → 不是均匀劣化，而是间歇性通信 straggler |
| **产出** | 性能异常：步耗时均值 10.3s / CV=27% / 慢步占比约 10%（慢步 > 15s）/ 慢步 MFU 26% |

### 2. 瓶颈分类层

| 步骤 | 内容 |
|------|------|
| **观测** | 取一个慢步（step 18427, T_iter=17.2s）和一个正常步（step 18420, T_iter=10.1s），对比 step_trace 的 5 桶分解： |
|  | 正常步：Computing=9422ms(93%), CommNotOverlapped=678ms(6.7%), Free=0ms |
|  | 慢步：Computing=9422ms(54.8%), **CommNotOverlapped=7778ms(45.2%)**, Free=0ms |
|  | Computing 几乎不变（9422ms），CommNotOverlapped 暴增 7100ms → **通信突发瓶颈** |
| **判据** | 慢步的通信耗时在关键路径占比 > 25%（实际 45.2%）→ **通信受限，走通信分支** |
| **产出** | 瓶颈类型：间歇性通信受限 / 慢步 CommNotOverlapped=7778ms vs 正常步 678ms（+7100ms） |

### 3. 通信原语层

| 步骤 | 内容 |
|------|------|
| **观测** | 从 `communication.json` 按通信原语类型拆解慢步 step 18427： |
|  | • `batchSendRecv`（P2P, PP stage 间激活/梯度传递）：elapse **460ms**（正常步 458ms，基本不变） |
|  | • `allReduce`（DP，gradient sync）：elapse **862ms**（正常步 858ms，基本不变） |
|  | • **`all-to-all`（EP，MoE expert dispatch+combine）**：elapse **6380ms**（正常步 198ms → **暴增 32×**！） |
|  | • `allReduce__346`（step 拖尾同步 grad-norm）：elapse **76ms**（正常步 72ms，略增） |
| **判据** | all-to-all 耗时从 198ms → 6380ms（32×）→ **MoE 通信 straggler。其余通信原语正常，排除物理链路故障（故障应导致所有通信原语恶化）** |
| **产出** | 异常通信原语：`all-to-all`（MoE dispatch+combine）/ 暴增幅度：+6182ms（32×）/ 其余通信原语正常 |

### 4. 通信模式层

| 步骤 | 内容 |
|------|------|
| **观测** | 进一步拆解 all-to-all 的 6380ms 构成（NCCL/HCCL trace + 通信视图的 Transit/Wait 瀑布图）： |
|  | ① **Transit/Wait 瀑布图**：按 EP rank 展开的 all-to-all 耗时瀑布（横轴 rank 0~63，纵轴时间，绿色=Transit 传输时间，橙色=Wait 等待时间）。正常步：所有 rank 的 Transit ≈ 180ms，Wait ≈ 18ms，均衡。慢步：rank 17、23、41 的 Transit=6200~6380ms（其余 rank Transit=180~220ms），rank 17/23/41 的 Wait=0（它们在发送数据），rank 0~16 的 Wait=6000~6200ms（它们在等 rank 17/23/41 完成发送） |
|  | ② **带宽达成率热力图**（rank × 通信类型，达成率=实测/理论）：慢步中 rank 17 的 all-to-all 带宽达成率仅 4.2%（理论 IB 400GB/s，实测 16.8GB/s），rank 23 仅 5.1%，rank 41 仅 6.8%——三个 rank 几乎占满整个 all-to-all barrier 的等待时间。其余 rank 的 all-to-all 带宽正常（55~65%） |
|  | ③ **Token-to-expert 分配统计**（`router` 的输出，慢步 vs 正常步对比）：正常步中每个 EP rank 处理的 token 数均值为 2048（total 4096 tokens × 8 experts / 64 rank / 8 expert per rank, 简化估算），CV=8%。慢步中 rank 17 处理了 **9842 tokens**（正常 2048 的 4.8×），rank 23 处理了 **8720 tokens**，rank 41 处理了 **7650 tokens**——这三个 rank 承载了整个 micro-batch 近 60% 的 MoE token 流量 |
| **判据** | 三张慢卡（rank 17/23/41）的 expert 负载是均值 4~5× → all-to-all send buffer 暴增（9842 tokens × 2560 dim × 1 byte FP8 ≈ 25MB vs 正常 5.2MB）→ 发送耗时 32×。根因：router gate weight 的 bias 项动态调整不及时（`noaux_tc` 策略的 γ=0.001 在特定数据分布下收敛滞后） |
| **产出** | 通信 straggler 根因：MoE router 负载倾斜 → rank 17/23/41 承接 4~5× 的 expert token → all-to-all send 量暴增 5× → 其余 61 个 rank 空等 → 步耗时尖峰 |

### 5. 执行效率层（负载倾斜深度归因）

| 步骤 | 内容 |
|------|------|
| **观测** | Timeline trace 泳道图（慢步 step 18427，横轴时间，纵轴 EP rank 0~63 的 all-to-all 通信段 + 计算段堆叠）： |
|  | • rank 17 的 all-to-all dispatch 段占据 6380ms 的连续红色块（CommNotOverlapped），其前后的计算段（MatMul + FA）与其余 rank 完全错位——rank 17 在 dispatch 期间其余 rank 已完成 dispatch 并在做 compute，但 rank 17 的 dispatch 未完成导致 EP group 内的 all-to-all barrier 无法解除 |
|  | • rank 17 的 router gate bias（`noaux_tc` 策略维护的 per-expert bias）在 step 18420~18426 期间持续向 expert 193 倾斜——检查 expert 193 的 bias 值：step 18420=0.02, 18423=0.12, 18425=0.38, **18427=0.91**（bias 调整速度 γ=0.001 跟不上 gate weight 的快速漂移）。expert 193 正好位于 rank 17 |
|  | • 数据层面：step 18427 的 micro-batch 中包含了大量「代码生成」类 token（`<code>` 标签密集），这些 token 的 hidden state 在 layer 38~45 的 router 上对 expert 193 产生了系统性偏好（expert 193 在预训练中恰好学会了代码语法特征） |
| **判据** | Router bias 累积 + 特定数据分布 = expert 193 过载 → rank 17 all-to-all straggler。`noaux_tc` 的 bias 更新速度 γ=0.001 在快速漂移场景下不足以在单步内纠正 |
| **产出** | 根因链：特定数据 → gate weight 快速漂移 → expert 193 过载（bias=0.91）→ rank 17 all-to-all send buffer 5× → 其余 rank 空等 → 步耗时尖峰 |

### 6. 代码/配置层

| 步骤 | 内容 |
|------|------|
| **修改 ① — 提高 bias 更新速度** | 将 `noaux_tc` 的 bias 更新速度 γ 从 0.001 提高到 **0.005**（前 14.3T tokens 阶段），加快过载 expert 的 bias 下调速度。预期 bias 在 2~3 步内从 0.91 回落至 0.3 以下，all-to-all 负载恢复均衡。代价：略微增加 expert 选择的抖动（bias 调控更频繁），但对 loss 影响 < 0.1% |
| **修改 ② — 增大 n_group / topk_group** | 将 router 的 `n_group` 从 8 增大到 **16**，`topk_group` 从 4 增大到 **6**。这增加了 expert 选择的多样性——每个 token 从 16 个 group 中选 top-6 group，再从其中选 top-8 expert，减少了单一 expert 被"垄断"的概率。代价：路由计算量增加约 15%（router 是轻量操作，对总耗时影响 < 0.5%） |
| **修改 ③ — Redundant Expert Deployment（推理可用，训练需评估）** | 对于检测到的高负载 expert（如 expert 193），在 EP group 内做冗余部署：将 expert 193 复制到 rank 17 和 rank 18 两张卡，token 随机分流到两张卡（各 50%）。但这对训练的一致性有影响（两条梯度的 expert weight 不同）——仅适用于推理场景，训练不推荐 |
| **修改 ④ — 训练侧兜底：增大 NCCL timeout** | 将 `NCCL_IB_TIMEOUT` 从 30s 增大到 **60s**，防止极端过载时 all-to-all 超时触发 NCCL 错误导致训练中断。不解决根因但防止训练 crash |
| **验证** | ①+② 实施后：慢步占比从 10% 降至 < 2%，步耗时 CV 从 27% → 约 8%，all-to-all 耗时恢复到正常步 180~220ms 范围（慢步不再出现 >1s 的 all-to-all）。步耗时均值 10.3s → 约 10.0s（正常步小幅下降，因为减少了偶发的 all-to-all jitter 对流水线稳定性的影响）。MFU 均值 45% → 约 46% |


---

## 案例四：显存分支 — activation checkpoint 未开启导致显存峰值超标 + 分配器碎片触发 OOM

> **路径**：性能表征层 → 瓶颈分类层（显存受限）→ 显存表征层（容量不足 + 分配碎片双因子）→ 显存峰值构成分析层 → 阶段/层定位层 → 代码/配置层

**背景**：2048 NPU 训练 openPangu-2.0-Flash，EP=64，TP=1，PP=4，DP=8，FP8 精度，seq_len=4096，micro_batch=1，global_batch=1024，**未开启 activation checkpointing**。训练至 step ~12000 后，部分 rank 间歇性报 `ACL_ERROR_MEMORY_ALLOCATION`，训练不稳定中断。`ascend-dmi` 显示各 rank 显存峰值在 58~64 GB（单卡总容量 64 GB），部分 rank 触及上限后 OOM。

### 1. 性能表征层

| 步骤 | 内容 |
|------|------|
| **现象** | 显存占用时间序列（横轴 step，纵轴显存占用 GB）：step 8000 前显存稳定在 52~55 GB（安全线以下）。step 8000 起显存逐步爬升，step 12000 时峰值触及 64 GB（上限）。step 12003 rank 17 报 `ACL_ERROR_MEMORY_ALLOCATION`，训练中断。其他 rank 在 61~63 GB 间波动，同样逼近红线。与此同时 throughput 从 step 10000 起从 3200 tokens/s 下滑至 2800 tokens/s（↓12.5%），因为分配器频繁做碎片整理和换页 |
| **判据** | 显存峰值 > 95% 总容量 + 伴有 OOM + 吞吐持续下滑 → 显存瓶颈，需深入分析 |
| **产出** | 显存异常：峰值 64/64 GB（100%）/ OOM rank：17 / 吞吐劣化：−12.5% |

### 2. 瓶颈分类层

| 步骤 | 内容 |
|------|------|
| **观测** | Profiling 报告拆解：Computing=8520ms（71%），Communication=1420ms（12%），显存分配/释放 API 调用耗时 = **890ms（7.4%）**——分配器开销异常高（正常应 < 2%）。显存占用率 > 95%，HBM 带宽利用率正常（78%），排除纯带宽瓶颈 |
| **判据** | 显存峰值 > 90% + 分配 API 耗时 > 3%（实际 7.4%）→ **显存受限，且同时存在容量不足和分配碎片两个子问题**。走显存分支 |
| **产出** | 瓶颈类型：显存受限（容量不足 + 分配碎片双因子）/ 分配耗时 890ms（7.4%）/ 峰值占用 64/64 GB |

### 3. 显存表征层 — 判定子类型

| 步骤 | 内容 |
|------|------|
| **观测** | ① 容量维度：显存峰值 64 GB = 总容量 64 GB → **绝对容量不足**。② 碎片维度：`aclrtMalloc` / `aclrtFreePhysical` 调用频率在 step 10000 后从每步 ~25 次暴增至 ~180 次（碎片整理），单次 `aclrtMalloc` P99 耗时从 0.3ms 升至 4.2ms。③ 可用空间与碎片：step 12000 时 rank 17 的显存快照显示——总空闲 1.8 GB，但最大连续空闲块仅 **0.3 GB**（碎片化严重），无法满足下一个 0.5 GB 的临时 buffer 分配请求，触发 OOM |
| **判据** | 显存占用 100% → 容量不足；空闲够但最大连续块 < 请求 size → 分配碎片。**双因子叠加**：容量不足是主因（若有余量碎片不会致命），碎片让 OOM 提前到来（若连续空闲块足够，1.8 GB 可多撑几十步） |
| **产出** | 显存瓶颈子类型：容量不足（主）+ 分配碎片（辅）/ 碎片指标：最大连续空闲块 0.3 GB vs 空闲总量 1.8 GB |

> **覆盖原声（CheckList Row 8）**：本节演示了「显存曲线→峰值标注→自动判定容量不足+碎片双因子」的完整能力——显存折线图自动标注 step 12000 为峰值点并标红，AI 同时给出"谁此刻吃显存最多"的初步诊断，用户无需手动盯曲线找峰值。

### 4. 显存峰值构成分析层 — 锁定「什么占满了显存」

| 步骤 | 内容 |
|------|------|
| **观测** | 提取 step 12000 时 rank 17 的显存快照（`torch.npu.memory_stats` / memory snapshot），按构成拆解： |
|  | • **激活值（含中间张量）**：**36.2 GB（56.6%）**——远超正常范围（开启 activation checkpoint 后通常 < 12 GB）。46 层 × micro_batch=1 × seq_len=4096，每层的 attention QKV 展开 + FFN gate/up 中间张量 + MoE expert dispatch buffer 全部常驻显存 |
|  | • **模型参数（FP8）**：8.1 GB（12.7%）——openPangu-2.0-Flash ~27B 参数，FP8 存储 ≈ 27 GB ÷ TP1 ÷ PP4 ≈ 6.8 GB，加上 embedding/lm_head 额外 ~1.3 GB |
|  | • **梯度（FP8）**：8.1 GB（12.7%）——与参数等量 |
|  | • **优化器状态（BF16 m+v）**：10.8 GB（16.9%）——BF16 优化器状态已减半，否则 FP32 将达 21.6 GB |
|  | • **临时 workspace buffer**：0.8 GB（1.2%）——CANN 算子内部临时 buffer，正常范围 |
| **判据** | 激活值占比 56.6% 是罪魁祸首——46 层全部激活常驻意味着每层约 0.79 GB 激活值。若开启 selective activation checkpointing（仅重计算 attention + FFN 中间激活），激活值可压缩至 ~8 GB，总显存降至 ~36 GB（安全线以下） |
| **产出** | 显存构成：激活 36.2 GB（56.6%）/ 参数+梯度 16.2 GB（25.4%）/ 优化器 10.8 GB（16.9%）/ 临时 buffer 0.8 GB → **激活值是唯一可大幅缩减的项** |

### 5. 阶段/层定位层 — 锁定「哪层显存压力最大」

| 步骤 | 内容 |
|------|------|
| **观测** | 按 PP stage 拆解显存使用热力图（横轴 layer index 1~46，纵轴显存占用 GB，颜色区分参数/梯度/优化器/激活）： |
|  | • PP stage 0（layers 1~11 + embedding）：激活 8.8 GB，参数+梯度+优化器 7.1 GB → 合计 15.9 GB |
|  | • PP stage 1（layers 12~22）：激活 9.0 GB，参数+梯度+优化器 6.8 GB → 合计 15.8 GB |
|  | • PP stage 2（layers 23~33）：激活 9.0 GB，参数+梯度+优化器 6.8 GB → 合计 15.8 GB |
|  | • **PP stage 3（layers 34~45 + final_layernorm + lm_head + loss）**：激活 9.4 GB + lm_head 额外 1.8 GB 激活（vocab=151552 的 logits 张量 [4096, 151552] ≈ 1.2 GB FP8），参数+梯度+优化器 7.5 GB（lm_head 权重 [151552, 2560] ≈ 0.8 GB FP8）→ 合计 **18.7 GB** |
|  | • 按层粒度：layer 38（MoE 层）激活值 1.2 GB，是普通 dense 层（0.7 GB）的 1.7×——MoE 的 expert dispatch buffer（256 experts × token 分配临时缓冲区）是额外开销 |
| **判据** | Stage 3 显存峰值 18.7 GB 是 stage 0~2（~15.8 GB）的 1.18×，lm_head 的 logits 张量（1.2 GB）和 MoE 层（layer 38，+0.5 GB vs dense）是主要额外开销。若开启 activation checkpoint，每层激活值可降至 ~0.17 GB（仅保留 input 和 layernorm 输出），全部 46 层激活总计约 8 GB |
| **产出** | 瓶颈 stage：PP stage 3（18.7 GB，因 lm_head + MoE）/ 热点层：layer 38 MoE（1.2 GB 激活）/ 最大单项：lm_head logits 张量（1.8 GB） |

> **覆盖原声（CheckList Row 8）**：本节演示了完整的"显存曲线→峰值算子→时间线→Python 调用栈"下钻闭环——从显存峰值定位 stage 3 → 下钻到 layer 38 MoE 的 expert dispatch buffer → 时间线中看到该 buffer 在 forward 开始分配、backward 结束后释放 → 调用栈回溯到 `model.layers.38.mlp.router.forward` 中的 `expert_dispatch` 函数。AI 在峰值处自动标注"layer 38 expert_dispatch 临时 buffer 此刻吃显存最多（1.2 GB），建议开启 activation checkpoint 或减小 MoE expert buffer 预分配"。

### 6. 内存快照分析 — 碎片与生命周期

> 本层是对「显存表征层」的深入展开，利用内存快照的完整解析能力，可视化每块显存申请的生命周期与碎片分布。对标 Row 7 的痛点。

| 步骤 | 内容 |
|------|------|
| **观测** | 导出 step 12000 时 rank 17 的完整内存快照（`memory_record.pkl`），解析后在界面中可视化： |
|  | ① **碎片分布图**：横轴为显存地址空间（0~64 GB），纵轴为时间（step 12000 内的时间线），每块显存分配用矩形块表示（颜色按用途：激活=橙、参数=蓝、梯度=绿、优化器=紫、临时 buffer=灰）。可以清晰看到大量橙色小块（激活中间张量）在 forward 期间密集分配、backward 后才释放，导致显存空间中充斥"已释放但未合并"的空洞——即碎片 |
|  | ② **单块生命周期**：点击任意矩形块，弹出该块显存的完整生命周期——申请时间、申请大小、申请堆栈（`torch.npu.empty` ← `q_b_proj.forward` ← `SelfAttn.forward` ← `TransformerLayer.forward` ← …）、释放时间、持有 duration。例如 layer 38 `q_b_proj` 输出的中间张量 [4096, 9216] FP8 ≈ 36 MB，从 forward 第 842ms 分配到 backward 第 7832ms 释放，持有近 7 秒 |
|  | ③ **碎片热力图**：沿地址空间的热力颜色——红色区域表示碎片密集区（连续空闲块 < 100 MB），绿色表示大块连续空闲。step 12000 中红色区域占空闲空间的 83%，即 1.8 GB 空闲中有 1.5 GB 是"看得见用不上"的碎片 |
| **判据** | 碎片率 = 不可用空闲 / 总空闲 = 1.5/1.8 = 83% → 严重碎片化。根因：46 层的大量不等大小中间张量在 forward 期间密集分配、backward 期间集中释放，分配器无法在短时间内合并碎片。开启 activation checkpoint 后，激活张量不再常驻，分配/释放频率大幅降低，碎片率可降至 20% 以下 |
| **产出** | 碎片根因：激活张量的高频分配/释放 → 碎片率 83% / 关键证据：layer 38 q_b_proj 中间张量持有 7s，堆栈可追溯到 `TransformerLayer.forward` |

> **覆盖原声（CheckList Row 7）**：本节演示了"内存快照→解析生命周期/堆栈→碎片分布可视化"的完整能力——不再需要导出 pkl 后用专门网页解析，直接在工具内看每块显存的"前世今生"（申请堆栈、持有时间、释放时机），碎片热力图一眼定位碎片密集区。

### 7. 代码/配置层

| 步骤 | 内容 |
|------|------|
| **诊断总结** | 根因是两个问题的叠加：① activation checkpoint 未开启 → 46 层激活全部常驻，占 56.6% 显存（36.2 GB）；② 大量不等大小的激活中间张量高频分配/释放 → 分配器碎片率 83%，OOM 提前到来。如果只开 checkpoint 而不解决碎片，碎片仍可能在更晚的 step 触发 OOM；如果只整理碎片而不缩减激活，64 GB 上限终将触及 |
| **修改 ① — 开启 selective activation checkpointing** | 在训练配置中开启 `--recompute-activations` 或 `--checkpoint-activations`，使用 selective 策略：仅对 attention（QKV projection + core_attention）和 FFN（gate_up + down）的中间激活做重计算，layernorm 和残差连接的输出保留。预期激活值从 36.2 GB → **~8.5 GB**（↓76%），总显存从 64 GB → **~36.3 GB**（安全线 57% 以下）。每步增加约 8% 计算时间（重计算 attention + FFN），对吞吐影响可控 |
| **修改 ② — 分配器碎片优化** | 设置 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`，让 CANN 分配器使用可扩展 segment 策略，减少碎片。同时设置 `ACLNN_CACHE_LIMIT=2147483648`（2 GB），为算子 workspace 预留连续缓存空间，避免每次重新申请。预期分配器 API 耗时从 890ms → **~180ms**（↓80%），碎片率从 83% → **~25%** |
| **修改 ③ — lm_head logits 即时释放** | 在 loss 计算完成后立即 `del logits` 并触发 `torch.npu.empty_cache()`（或依赖 PyTorch 的引用计数自动回收），避免 vocab=151552 的巨大 logits 张量（1.2 GB）在 backward 期间继续占用显存。lm_head 的 logits 仅在 loss 计算时需要，backward 中梯度直接从 loss 反传，不需保留 logits 本身 |
| **修改 ④ — 调整 PP stage 层数分配（辅助）** | 从 [12,11,11,12] 调整为 [13,12,11,10]——stage 3 减 2 层减压，stage 0 和 1 各多 1 层。stage 3 的显存从 18.7 GB → 约 16.2 GB，与其余 stage 更均衡 |
| **验证** | ①+②+③ 从 step 12000 续跑：显存峰值从 64 GB → 约 34 GB（↓47%），分配器 API 耗时从 890ms → 150ms（↓83%），碎片率从 83% → 22%。throughput 恢复至 3150 tokens/s（略低于初始 3200，因 activation checkpoint 重计算开销约 +8%，但稳定无 OOM）。继续训练 10000 step 无显存异常。④ 可选，实施后 stage 间显存更均衡（CV 从 8% 降至 3%） |

> **覆盖原声（CheckList Row 7, 8）**：
> - Row 7：「内存分析还要用内存快照，保存成pkl后用专门网页解析，看每一块内存申请的生命周期、堆栈是哪块代码触发的。MindStudio Insight现在看不出来内存碎片」→ 本节 §6 内存快照分析完整演示了碎片分布图、单块生命周期（含申请堆栈）、碎片热力图，AI 定位碎片成因（激活张量高频分配/释放）并给出复用/池化建议
> - Row 8：「MindStudio Insight以折线图展示显存变化，还可以从峰值附近的算子跳转到时间线，再回到Python调用栈定位问题」→ 本节 §1~§5 演示了完整的"显存折线图→峰值自动标注→下钻到 stage/层/算子→时间线中看分配/释放时机→调用栈回溯到 `router.forward` 的 `expert_dispatch`"下钻闭环，AI 在峰值处自动标注"谁此刻吃显存最多、能否提前释放/重计算"