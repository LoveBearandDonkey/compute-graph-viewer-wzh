# 1F1B 时序与 Microbatch Lifecycle 核对笔记

> 状态：已核对标准 non-interleaved 1F1B 宏观时序  
> 基准实现：NVIDIA Megatron Core `core_v0.16.0`  
> 对应页面：`op-rank-time-openpangu-flash-css3d.html`  
> 当前演示配置：`DP=2 / PP=4 / TP=2 / EP=2 / VPP=1 / microbatches=8`

## 1. 结论

当前 `simulate1F1BSchedule()` 使用的三条宏观约束符合 Megatron Core 标准 non-interleaved 1F1B：

1. 同一个 microbatch 的 Forward 必须等待前一个 PP stage 产生并传来 activation。
2. 同一个 microbatch 的 Backward 必须等待后一个 PP stage 产生并传来 activation gradient。
3. 同一个物理 PP stage、同一个 model chunk 的主计算一次只执行一个 Forward 或 Backward task。

第三条不表示设备上没有任何并行活动。TP、EP、DP 通信以及其他独立 CUDA stream 上的工作可能重叠；它只约束标准 non-interleaved 1F1B 的 stage 主计算顺序。

当前页面的 PP/MB/F/B 交错顺序可以作为规则正确的模拟时序使用，但每个 task 的耗时、task 内精确算子位置和通信重叠窗口仍是 synthetic，不是实测 profiling trace。

### 1.1 先用一个直观模型理解

可以把 PP 想成一条由多个工位组成的流水线：

```text
输入 -> PP0 -> PP1 -> PP2 -> PP3 -> loss
```

- 一个 PP stage 是一段连续模型层，例如 `PP1=L12-L22`，不是一个算子。
- 一个 microbatch 是沿流水线移动的一小批训练样本，可以理解为一件待加工的数据对象。
- Forward 时，数据从 `PP0` 向 `PP3` 移动，stage 之间传的是 activation。
- Backward 时，梯度从 `PP3` 向 `PP0` 返回，stage 之间传的是 activation gradient。
- 同一时刻，每个 stage 可以加工不同的 MB，所以横向看过去的 MB 编号通常不连续。

NVIDIA 的技术文章把这种方式称为 PipeDream-Flush 风格的 1F1B：先用若干 Forward 填满流水线，随后每个 worker 交替做一个 Forward 和一个 Backward，最后排空剩余 Backward。这样做的主要目的不是让一个 MB 更快，而是让多个 stage 尽量同时有事可做。

### 1.2 这里的“并行”到底是什么

需要区分三种并行：

1. **PP stage 之间并行**：`PP0` 处理 MB3 Forward 时，`PP3` 可以处理 MB1 Backward。
2. **一个 stage 内 TP/EP ranks 协同**：多个 rank 共同计算同一个 MB 的 tensor shard 或 expert token，不是每个 rank 各算一个 MB。
3. **DP replica 之间并行**：`D0:MB2` 和 `D1:MB2` 是两个数据副本流水线里的不同样本流，只是编号相同。

因此“PP1 当前是 MB3 F”描述的是一个 stage task；它背后可能有多个 TP/EP rank 同时参与。

NVIDIA 的 1F1B 示意图通常为了讲清调度，把一个 worker 或 device 画成一个 pipeline stage。组合 TP 后不能机械理解成“一条 stage 线只是一张卡”：一个 stage 可以由一个 TP group 共同完成，同组 ranks 对同一个 MB 做张量分片计算。1F1B 调度的对象仍然是这个 stage/model chunk，而不是组内每个 rank 各自选择 MB。

## 2. 术语

设：

```text
P                  PP stage 数量
M                  每个 DP replica 的 microbatch 数量
s                  stage 编号，0 <= s < P
m                  microbatch 编号，0 <= m < M
F(s,m)             MB m 在 stage s 上的 Forward compute
B(s,m)             MB m 在 stage s 上的 Backward compute
A(s,m)             F(s,m) 输出的 activation
dA(s,m)            B(s,m) 输出、发往前一 stage 的 activation gradient
```

当前页面：

```text
PP0 = L0-L11
PP1 = L12-L22
PP2 = L23-L34
PP3 = L35-L45
M   = 8，编号 MB0-MB7
```

页面中的 `MB2` 表示 microbatch ID，不表示 `micro_batch_size=2`。

### 2.1 一个 MB 在 stage 边界传递什么

Forward 与 Backward 传的不是同一种 tensor：

| 方向 | 发送内容 | 接收方拿它做什么 |
|---|---|---|
| Forward `PPs -> PPs+1` | `A(s,m)`，即该 stage 的输出 activation | 作为下一 stage 的 Forward 输入 |
| Backward `PPs+1 -> PPs` | `dA(s+1,m)`，即 loss 对下一 stage 输入的梯度 | 作为当前 stage 的 Backward 上游梯度 |

这解释了为什么 Forward 等前一 stage，而 Backward 等后一 stage：两种计算依赖的数据来源正好相反。

## 3. 官方核对后的依赖规则

### 3.1 Forward 依赖

对于非首 stage：

```text
F(s,m) 只能在收到 A(s-1,m) 后开始

end(F(s-1,m))
  -> PP send activation
  -> PP recv activation
  -> start(F(s,m))
```

形式化约束：

```text
start(F(s,m)) >= end(F(s-1,m)) + comm_forward(s-1,s,m)
```

首 stage `PP0` 不从前一 stage 接收 activation。它直接从本地 data iterator、Embedding 或模型输入开始该 MB 的 Forward。

Megatron Core 的 `P2PCommunicator.recv_forward()` 明确定义为“从前一个 pipeline rank 接收 tensor”；官方调度在 `forward_step()` 前调用它。

白话解释：`PP1` 持有 `L12-L22`，但它自己无法凭空算 `L12`。它必须先收到 `PP0` 对同一个 MB 计算出的 `L11` 输出。这里的“等待前一 stage 完成”只指前一 stage 对**同一个 MB 的这次 Forward**完成，不是等待前一 stage 完成整个 iteration。

### 3.2 Backward 依赖

对于非末 stage：

```text
B(s,m) 只能在收到后一个 stage 返回的 dA(s+1,m) 后开始

end(B(s+1,m))
  -> PP send backward gradient
  -> PP recv backward gradient
  -> start(B(s,m))
```

形式化约束：

```text
start(B(s,m)) >= end(B(s+1,m)) + comm_backward(s+1,s,m)
```

最后一个 stage 没有下一 stage。它在该 MB 的本地 Forward、输出和 loss 就绪后，从 loss 开始 Backward：

```text
start(B(P-1,m)) >= end(F(P-1,m))
```

Megatron Core 的 `recv_backward()` 定义为从下一个 pipeline rank 接收 gradient。中间 stage 的 `backward_step()` 需要 `output_tensor_grad`；只有最后一个 stage 的该输入可以为 `None`，因为它从 loss 发起反向传播。

白话解释：`PP1` 想计算 `L22-L12` 的梯度，必须先知道“loss 对 PP1 输出的梯度”。这个值由 `PP2` 的 Backward 产生，所以反向传播天然从后向前。最后一个 stage 直接连接 loss，因此它不需要等一个不存在的 `PP4`。

### 3.3 Stage 内主计算互斥

标准 non-interleaved 1F1B 在每个 stage 上执行：

```text
Warmup:   F F F ...
Steady:   F B F B ...
Cooldown: B B B ...
```

同一个 stage 不会同时运行两个完整的 `F(s,*)`，也不会同时运行一个完整 F 和一个完整 B。官方稳态循环先调用一次 `forward_step()`，随后调用一次 `backward_step()`。

形式化约束：

```text
对于同一 stage s，任意两个主计算 task 的时间区间不能重叠。
```

当前模拟器使用 `stageFree[s]` 实现这条约束。

这里的“一次一个 task”不等于一张 GPU 只能执行一个 kernel。一个 stage task 内仍然包含很多算子，也可能由一个 TP group 的多张卡共同执行；通信库还可能把独立通信放到其他 stream。规则约束的是完整 `F(s,m)` 与 `B(s,m)` 的宏观排队顺序。

## 4. 标准 non-interleaved 1F1B 的计算与通信顺序

### 4.1 Warmup

每个 stage 先执行若干 Forward，使流水线被填满。

官方 warmup 数量：

```text
warmup(s) = min(P - s - 1, M)
```

单次 warmup Forward：

```text
PP recv activation from previous stage
  -> Forward compute
  -> PP send activation to next stage
  -> retain activation needed by future Backward
```

边界 stage 中，不存在的 recv/send 是 no-op：

- `PP0` 不执行真实的 `recv_forward`。
- `PP(P-1)` 不向下一 PP stage 发送模型 activation。

对于当前 `P=4`，各 stage 的 warmup 数量不同：

| Stage | Warmup Forward 数 | 原因 |
|---|---:|---|
| PP0 | 3 | 离末 stage 最远，需要先送出更多 MB 才能等到第一个梯度返回 |
| PP1 | 2 | 比 PP0 少经过一个 stage |
| PP2 | 1 | 只需要把一个 MB 送到 PP3 |
| PP3 | 0 | 它拿到第一个 MB 后就能从 loss 发起 Backward |

所以 warmup 不是“四个 stage 一起先做三次 Forward”。越靠后的 stage 越早进入 1F1B；这正是画面上 F/B 状态不整齐的第一个原因。

### 4.2 Steady 1F1B

当流水线填满后，每个 stage 交替处理一个较新的 Forward MB 和一个较早的 Backward MB。

Megatron Core non-interleaved 主循环的顺序可概括为：

```text
已有当前 Forward activation
  -> F(s,new_mb)
  -> send_forward(A(s,new_mb))
     + recv_backward(dA(s+1,old_mb))
  -> B(s,old_mb)
  -> send_backward(dA(s,old_mb))
     + recv_forward(A(s-1,next_mb))
  -> 下一轮
```

`send_forward_recv_backward()` 与 `send_backward_recv_forward()` 把相反方向的 P2P send/recv 合并到一次通信调用中。它们表达的是 stage 边界的数据交换，不表示同一个 stage 上的完整 Forward 和 Backward compute 同时执行。

函数名可以拆开读：

| Megatron 调用 | 这个 stage 做什么 | 对端是谁 |
|---|---|---|
| `recv_forward` | 接收下一次 Forward 所需 activation | 前一 stage |
| `send_forward` | 发送本次 Forward 输出 activation | 后一 stage |
| `recv_backward` | 接收本次 Backward 所需 gradient | 后一 stage |
| `send_backward` | 发送本次 Backward 产生的 gradient | 前一 stage |
| `send_forward_recv_backward` | 向后发送新 activation，同时从后接收旧 MB gradient | 后一 stage |
| `send_backward_recv_forward` | 向前发送旧 MB gradient，同时从前接收下一个 MB activation | 前一 stage |

注意“同时 send/recv”是一次双向 P2P 数据交换，不等于当前 stage 同时执行两个完整模型计算。官方 non-interleaved 主循环仍然是 `forward_step()` 完成后再进入 `backward_step()`。

### 4.2.1 为什么稳态快照看起来没有 MB 顺序

假设某一时刻显示：

```text
PP0: MB0 B
PP1: MB3 F
PP2: MB1 B
PP3: MB2 B
```

逐项解释：

- `PP3: MB2 B`：末 stage 已经完成 MB2 Forward/loss，正在向回算梯度。
- `PP2: MB1 B`：PP2 已收到 PP3 对更早 MB1 返回的梯度。
- `PP1: MB3 F`：PP1 同时可以继续把较新的 MB3 往后送。
- `PP0: MB0 B`：最早的 MB0 梯度已经经过 PP3、PP2、PP1，最终返回 PP0。

它们属于四个不同 MB 的四段工作，不应该按 MB 编号从左到右排序。要看顺序，应固定一个 MB，再沿它自己的 lifecycle 追踪。

### 4.3 Cooldown

所有 Forward 已发出后，剩余 Backward 依次排空：

```text
PP recv backward gradient from next stage
  -> Backward compute
  -> PP send backward gradient to previous stage
```

边界 stage：

- `PP(P-1)` 从本地 loss 发起 Backward，不执行真实的 `recv_backward`。
- `PP0` 计算出的输入梯度不需要继续发给更前面的 PP stage。

Cooldown 产生尾部 bubble：前面的 stage 还在等待最后几份 gradient 返回时，后面的 stage 可能已经做完。NVIDIA 将 iteration 开始和结束时这类设备空闲时间称为 pipeline bubble。增加 microbatch 数通常可以降低 bubble 在整轮训练时间中的占比，但会改变 activation、调度和 kernel 效率之间的权衡。

### 4.4 NVIDIA 如何解释 bubble、MB 数和显存

NVIDIA 的简化模型设：

```text
p   pipeline stage 数
m   一个 batch 中的 microbatch 数
tf  一个 microbatch 的 Forward 时间
tb  一个 microbatch 的 Backward 时间
```

标准 flush schedule 的头尾 bubble 时间近似为：

```text
t_bubble = (p - 1) x (tf + tb)
```

理想计算时间为：

```text
t_ideal = m x (tf + tb)
```

所以 NVIDIA 用下面的比值表达 bubble 相对理想计算量的开销：

```text
t_bubble / t_ideal = (p - 1) / m
```

直观含义不是“增加 MB 能消灭头尾等待”，而是固定的填充/排空成本可以被更多 MB 摊薄。当前 `p=4、m=8` 时，这个简化比值是 `3/8=37.5%`。它是相对理想计算时间的额外开销；如果用实际总时间作分母，比例会是 `3/(8+3)`，不要把两种口径混用。

1F1B 相比“先做完所有 Forward、再做所有 Backward”的 GPipe 风格还有一个显存优势。NVIDIA 说明，1F1B 的 outstanding Forward 数最多约为 pipeline depth，因此通常只需要为 `p` 个或更少的 in-flight MB 保留 activation，而不是为全部 `m` 个 MB 保留。也就是说：

- `m` 增大有利于摊薄 bubble；
- 1F1B 把同时等待 Backward 的 activation 数量限制在 pipeline depth 附近；
- 真实显存仍受 stage layer 数、sequence length、microbatch size、recompute 和 MoE token 分布影响。

### 4.5 为什么当前笔记只讨论 non-interleaved 1F1B

当前页面 `VPP=1`，每个物理 stage 持有一段连续 layer range，因此对应 Megatron 的 non-interleaved 1F1B。

NVIDIA 的 interleaved 方案会把同一设备上的层再拆成多个 model chunks，使一个设备承担多个 virtual stages。这样可以缩短 bubble，但会增加 stage 边界和通信次数。combined/fine-grained 1F1B 还可能在 layer 或子模块级同时调度不同 MB 的 Forward/Backward。

这些优化不能直接套用当前三条宏观可视化规则。未来若把 `VPP` 提高或启用 combined 1F1B，至少需要新增 `modelChunkId/virtualStageId`，并允许一个物理 stage 在多个 chunk task 间切换。

## 5. 一个 stage 内的计算与通信

PP P2P 只负责 stage 边界。一个 MB 进入 stage 后，还会在该 stage 的 TP/EP rank group 内执行算子和通信。

四种通信不能混为一条全局通信链：

| 通信域 | 参与者 | 主要传输内容 | 发生位置 | 回答的问题 |
|---|---|---|---|---|
| PP P2P | 相邻 PP stage | activation / activation gradient | stage 边界，每个 MB | 这个 MB 如何跨模型深度移动 |
| TP/SP collective | 同一 layer 的 TP group | tensor/sequence shard | Attention 或 FFN 算子内部 | 同一层如何由多 rank 协同计算 |
| EP All-to-All | 同一 MoE layer 的 expert group | routed tokens / expert outputs | Router 与 Expert compute 两侧 | token 如何去往持有目标 expert 的 rank |
| DP gradient collective | 相同参数 shard 的 DP replicas | parameter gradients | Backward bucket ready 后 | 不同数据副本如何得到一致梯度 |

PP 决定“这个 MB 现在在哪段 layer”；TP/EP 决定“这段 layer 内多个 rank 如何一起算”；DP 决定“不同数据副本的梯度如何合并”。

### 5.1 Attention 路径

当前 OpenPangu Flash 可视化采用：

```text
Input RMSNorm
  -> Attention TP/SP AllGather
  -> Q/KV Latent Linear
  -> Q/KV Causal Conv1D + residual
  -> Q/KV LayerNorm + Up Linear
  -> Sparse FlashAttention
  -> Output Causal Conv1D + residual
  -> Output Projection
  -> Attention TP/SP Reduce-Scatter
  -> Post-Attention RMSNorm + mHC merge
```

这里 TP rank 处理的是同一个逻辑 MB 的不同 tensor shard，不是不同 MB。

### 5.2 Dense FFN 路径

Dense layer `L0-L1` 当前采用：

```text
Pre-MLP RMSNorm
  -> Dense FFN TP/SP AllGather
  -> Dense Gate / Up Linear
  -> Dense SiLU x Multiply
  -> Dense Down Linear
  -> Dense FFN TP/SP Reduce-Scatter
  -> Post-MLP RMSNorm
  -> mHC FFN Merge
```

### 5.3 MoE FFN 路径

MoE layer `L2-L45` 当前采用 `dispatch_combine` 路径：

```text
Pre-MLP RMSNorm
  -> Router Top-8
  -> EP Dispatch fused All-to-All
  -> Routed Expert + Shared Expert compute
  -> EP Combine fused All-to-All / local branch add
  -> Post-MLP RMSNorm
  -> mHC FFN Merge
```

EP All-to-All 是同一 layer、同一 MB 内的 token dispatch/combine。它与 PP activation/gradient P2P 不是同一个通信域。

### 5.4 DP gradient sync

DP 通信同步不同 data replica 上的参数梯度：

```text
Backward 产生 parameter gradients
  -> gradient bucket ready
  -> DP All-Reduce 或 Reduce-Scatter
  -> iteration finalize
  -> optimizer update
```

DP gradient sync 可以按 bucket 与 Backward 重叠，但 optimizer step 必须等待本轮需要的梯度同步完成。它不决定同一个 MB 如何在 PP stage 间移动。

## 6. 以一个 MB lifecycle 为视角的两条方向规则

选择固定 microbatch `m` 后，应忽略其他 MB 的编号顺序，只追踪它自己的数据依赖。

一个 MB 在一个 stage 上的完整状态机可以写成：

```text
WAIT_ACT
  -> FORWARD_COMPUTE
  -> SEND_ACT
  -> HOLD_ACTIVATION
  -> WAIT_GRAD
  -> BACKWARD_COMPUTE
  -> SEND_GRAD
  -> DONE_ON_THIS_STAGE
```

- `WAIT_ACT`：等待前一 stage 的 Forward 输出；PP0 没有这一步。
- `HOLD_ACTIVATION`：Forward 已完成，但 Backward 尚未到来，activation 需要保留或未来重算。
- `WAIT_GRAD`：等待后一 stage 返回 gradient；PP3 从 loss 启动，没有外部等待。
- `DONE_ON_THIS_STAGE`：只表示该 MB 在这个 stage 上完成，整轮训练还要等其他 MB 和梯度同步。

这也是 MB lifecycle 视图不能只画 F/B 两个色块的原因。中间的通信、hold 和 dependency wait 都是定位 pipeline latency 的重要证据。

### 6.1 规则一：Forward 沿 stage 和 layer 正向移动

```text
PP0 -> PP1 -> PP2 -> PP3
L0  -> ... -> L45 -> Final Norm -> LM Head -> Loss
```

以 `MB2` 为例：

```text
F(PP0,MB2): L0  -> L11
  -> send activation PP0 -> PP1
F(PP1,MB2): L12 -> L22
  -> send activation PP1 -> PP2
F(PP2,MB2): L23 -> L34
  -> send activation PP2 -> PP3
F(PP3,MB2): L35 -> L45 -> Final Norm -> LM Head -> Loss
```

Forward lifecycle 不允许跳过 stage，也不允许 `F(PP2,MB2)` 早于 `F(PP1,MB2)` 完成并传来 activation。

### 6.2 规则二：Backward 沿 stage 和 layer 反向移动

```text
PP3 -> PP2 -> PP1 -> PP0
Loss -> L45 -> ... -> L0
```

同一个 `MB2`：

```text
B(PP3,MB2): Loss/Head -> L45 -> L35
  -> send gradient PP3 -> PP2
B(PP2,MB2): L34 -> L23
  -> send gradient PP2 -> PP1
B(PP1,MB2): L22 -> L12
  -> send gradient PP1 -> PP0
B(PP0,MB2): L11 -> L0
```

Backward lifecycle 不允许 `B(PP1,MB2)` 在 `B(PP2,MB2)` 返回 gradient 前开始。

把 Forward 与 Backward 合在一起看，`MB2` 的路径不是一个连续无停顿的动画，而是：

```text
向后逐 stage Forward
  -> 在各 stage 留下待 Backward 使用的 activation
  -> PP3 产生 loss
  -> 向前逐 stage Backward
```

在 MB2 等待下一 stage 时，当前 stage 会转去计算其他 MB。这就是 1F1B 能提高设备利用率的核心。

### 6.3 Activation retention

同一 stage 完成 `F(s,m)` 后，通常需要保留该 MB 的部分 activation，直到 `B(s,m)` 使用：

```text
end(F(s,m))
  -> activation retained / checkpointed / recomputed
  -> start(B(s,m))
```

这个区间是 MB lifecycle 的 `hold`，不是计算，也不是网络通信。Activation checkpointing 会改变保留量和重计算量，但不改变上述 PP 依赖方向。

## 7. 可视化中的两种观察规则

### 7.1 Stage-centric snapshot

问题是：

```text
当前时刻，每个 PP stage 正在做什么？
```

显示规则：

- 每个 PP stage 显示一个当前主 task。
- 不同 stage 可以显示不同 MB。
- 不同 stage 可以同时处于 F 或 B。
- tag 的颜色表示当前 stage/MB task，不代表同一 MB 的连续路径。

因此下面这种快照可以是合法的：

```text
PP0: MB0 B
PP1: MB3 F
PP2: MB1 B
PP3: MB2 B
```

它不是按 MB 编号排序的列表，而是 1F1B steady state 的并行截面。

这个视角适合判断：

- stage 是否忙碌；
- pipeline bubble；
- 当前 F/B 混合状态；
- stage load imbalance。

### 7.2 MB-centric lifecycle

问题是：

```text
选中的 MB 从输入到 loss，再从 loss 返回梯度，走到了哪里？
```

显示规则：

- 固定一个 `DP replica + MB ID`，例如 `D0:MB2`。
- Forward 只按 `PP0 -> PP3` 高亮。
- Backward 只按 `PP3 -> PP0` 高亮。
- 显式区分 compute、PP send/recv、activation hold 和 bubble/wait。
- 其他 MB 可以保留为上下文，但必须降权，不能与选中 MB 混成一条路径。

这个视角适合判断：

- 单个 MB 的端到端 latency；
- activation 在哪个 stage 保留最久；
- PP 边界传输是否拖慢该 MB；
- Forward/Backward 的因果链是否完整。

## 8. 当前模拟器正确与不精确的部分

先区分“规则来源”和“画面数据来源”：

| 页面内容 | 来源 | 当前可信度 |
|---|---|---|
| PP stage layer range | 当前 OpenPangu 模型配置与产品映射 | 配置级 |
| warmup / steady / cooldown 顺序 | Megatron Core non-interleaved 1F1B | 官方规则 |
| Forward/Backward 跨 stage 依赖 | Megatron P2P API 与调度源码 | 官方规则 |
| 某个 F/B task 持续多少微秒 | `compDuration()` synthetic 生成 | 演示值 |
| task 内此刻落在哪个算子 | 按 program 长度均匀插值 | 演示推断 |
| TP/EP/DP 通信精确起止 | 固定比例和 jitter 生成 | 演示推断 |
| activation/gradient 实际数值 | 未接 profiler/runtime trace | 当前没有实测证据 |

因此当前页面可以回答“依赖方向和 1F1B 排队是否合理”，不能证明“真实训练在第 625 微秒恰好执行这个算子”。

### 8.1 已核对、可作为不变量

```text
F(s,m) waits for F(s-1,m) activation
B(s,m) waits for B(s+1,m) gradient
B(P-1,m) waits for local F(P-1,m)/loss
one main F/B compute task per stage at a time
warmup -> steady 1F1B -> cooldown
```

### 8.2 当前仍是 synthetic

- F/B task duration 由基准值和 deterministic hash 扰动生成。
- task 内 layer/operator 使用均匀进度切分，不代表真实 kernel duration。
- Backward 暂时通过反转 Forward program 表达，没有独立的 dgrad/wgrad 图。
- TP/EP/DP 通信窗口由固定比例插入，不是 profiling trace 的真实起止时间。
- 页面存在 `TRACE ticks` 与 simulated wall-clock runtime 两层数据，必须避免把两者混称为实测 trace。

因此 UI 文案应使用：

```text
simulated 1F1B schedule
simulated operator position
inferred communication window
```

没有接入 profiler 数据前，不应使用：

```text
actual kernel timeline
measured operator duration
real communication overlap
```

## 9. 对播放与联动的约束

1. 播放必须只有一个 schedule time source。
2. Stage tag 表示 stage-centric snapshot，允许四个 PP stage 显示不同 MB。
3. 选择某个 MB 后切换为 MB-centric lifecycle，不再把四个 stage 的当前 task 误连成同一个 MB。
4. 算子高亮必须同时由 `stage + layer + operator + phase + MB` 定位。
5. 同名算子在不同 PP stage、不同 layer 上必须可以区分。
6. PP communication 必须画在 stage 边界，TP/EP communication 必须画在 layer 内部。
7. Activation hold 必须作为状态区间显示，不能伪装成 compute 或 communication。
8. 播放前保持模型原色；播放时才降低无关算子权重。
9. Stage-centric 与 MB-centric 的标题、tooltip 和 inspector 必须明确当前观察口径。

## 10. 官方依据

### 10.1 这些资料分别说明了什么

- **Parallelism Strategies Guide**：确认 PP 切模型深度，TP 切单层 tensor，DP 切 batch。它用来确定不同并行维度的职责边界。
- **NVIDIA Megatron 技术文章**：用图解释 microbatch、warmup、steady 1F1B、cooldown 和 pipeline bubble，说明为什么多个 worker 会处理不同 MB。
- **P2PCommunicator API**：定义 `recv_forward`、`recv_backward`、`send_forward`、`send_backward` 的通信方向和 payload 语义。
- **`schedules.py` 官方源码**：给出真实调用顺序，证明 Forward 前接 activation、Backward 前接 gradient，以及 non-interleaved 稳态按一次 F、一次 B 串行调度。

资料的角色不同：指南解释概念，技术文章建立直觉，API 解释通信接口，源码才是核对精确执行顺序的最终依据。

- [Megatron Core Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/0.16.0/user-guide/parallelism-guide.html)
- [NVIDIA Technical Blog: Scaling Language Model Training to a Trillion Parameters Using Megatron](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/)
- [Megatron Core Pipeline Parallel Schedules API](https://docs.nvidia.com/megatron-core/developer-guide/0.16.0/apidocs/core/core.pipeline_parallel.schedules.html)
- [Megatron Core P2PCommunicator API](https://docs.nvidia.com/megatron-core/developer-guide/0.16.0/apidocs/core/core.pipeline_parallel.p2p_communication.html)
- [Megatron-LM `core_v0.16.0` non-interleaved 1F1B source](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/pipeline_parallel/schedules.py#L1972-L2305)

关键官方源码位置：

- warmup 数量和 warmup Forward：[`schedules.py#L2092-L2180`](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/pipeline_parallel/schedules.py#L2092-L2180)
- steady 1F1B：[`schedules.py#L2182-L2264`](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/pipeline_parallel/schedules.py#L2182-L2264)
- cooldown Backward：[`schedules.py#L2266-L2292`](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/pipeline_parallel/schedules.py#L2266-L2292)

## 11. 当前实现位置

- 模拟 1F1B 调度：`js/analysis-data.js` 中的 `simulate1F1BSchedule()`
- 运行时 rank/task 构造：`js/analysis-data.js` 中的 `buildSimulated1F1BRuntime()`
- 当前 stage task 判定：`op-rank-time-openpangu-flash-css3d.html` 中的 `activeStageComputeTask()`
- task 到 layer/operator 的映射：同文件中的 `stageExecutionProgram()` 与 `currentCardActivityStageEvents()`
- MB lifecycle swimlane：同文件中的 `buildMicrobatchLifecycleViewModel()` 与 `paintMicrobatchBody()`
