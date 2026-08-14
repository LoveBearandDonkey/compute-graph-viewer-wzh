## 整网级训练数据可视化需求提炼

### 一、核心判断

训练任务目前常见的监控指标，例如 Loss、吞吐量、MFU、显存占用、Step Time、通信耗时等，只能说明任务整体是否健康，属于任务级的表面指标。

当这些指标发生异常时，还需要进一步回答：

* 异常最早出现在哪个 Layer、Block 或 Stage？
* 是 Attention、FFN、MoE、Residual 还是通信边界引起的？
* 是前向激活异常、反向梯度异常，还是参数更新异常？
* 异常只发生在某个 Microbatch、某个 Rank，还是整网普遍存在？
* 数值异常与通信、计算耗时、并行切分之间是否存在关联？

因此，需要在训练任务概览之下增加一层**整网级全局数据视图**，把模型结构、运行时数据和分布式执行信息关联起来。

---

### 二、整网图需要标识的不是“所有原始值”，而是关键对象的统计状态

整网模型包含大量参数和中间张量，不适合直接展示完整 tensor。整网图上应展示经过聚合的数值摘要和异常状态，完整数据再通过下钻查看。

主要包含四类对象。

#### 1. 参数对象

包括：

* Weight
* Bias
* LayerNorm / RMSNorm 参数
* Embedding 参数
* MoE Expert 参数
* Router 参数

建议标识：

* Shape、dtype
* 参数量
* 是否存在 bias
* 是否被 TP、EP、PP 等策略切分
* 参数或分片所在 Rank
* min、max、mean、std
* L1/L2 norm、amax
* NaN、Inf、异常值比例
* 参数更新量或更新前后差值
* 是否冻结、是否参与优化

整网图主要显示参数状态，不宜直接显示参数矩阵中的所有元素。

#### 2. 前向运行对象

包括：

* Layer 输入、输出
* Attention 的 Q、K、V
* Attention Score、Softmax 输出
* FFN / MoE 中间激活
* Residual Add 前后张量
* PP Stage 间 Send/Recv 激活
* MoE Router Score、Expert 输入输出
* Loss 输入及 Logits

建议标识：

* Shape、dtype
* min、max、mean、std
* amax、动态范围
* L1/L2 norm
* 零值比例、饱和比例
* NaN、Inf
* 激活分布是否突然漂移
* 同一对象在不同 MB、Rank、Step 间的差异

这些数据用于判断数值异常从哪个节点开始产生，并沿数据流向后传播。

#### 3. 反向运行对象

包括：

* Layer 输出梯度和输入梯度
* Weight Gradient
* Bias Gradient
* Residual 分支梯度
* PP Stage 间反向 Send/Recv 梯度
* MoE Expert Gradient
* 梯度归约前后的结果

建议标识：

* min、max、mean、std
* L1/L2 norm
* 梯度动态范围
* NaN、Inf
* 梯度消失、梯度爆炸
* 梯度裁剪前后差异
* 梯度累积状态
* DP/TP/EP 归约前后差异
* 不同 Rank 间梯度一致性

偏置梯度属于其中一种参数梯度，应被纳入参数清单，但通常不需要在整网主画布上单独占据与激活同等的视觉层级。

#### 4. 运行时与并行执行对象

需要把数值数据与实际执行位置关联起来，包括：

* Step
* Global Batch
* Microbatch
* Forward / Backward
* Pipeline Stage
* DP、TP、PP、EP、CP、SP Group
* Rank、Device、Node
* 算子实例
* Send/Recv 和 Collective Communication

建议关联：

* 算子耗时
* 通信耗时
* 等待时间
* Pipeline Bubble
* 计算与通信重叠率
* HBM / Memory 占用
* 通信量
* MFU
* Straggler Rank
* 数值异常发生时对应的 MB、Rank 和时间位置

---

### 三、整网图应该解决的核心问题

整网视图的价值不是展示更多数据，而是建立异常定位链路：

```text
训练任务异常
→ 定位异常 Step
→ 定位异常 Stage / Rank / Microbatch
→ 定位异常 Layer / Block
→ 定位 Attention / FFN / MoE / Residual 等模块
→ 定位具体张量、参数、梯度或通信操作
→ 下钻到算子和原始数据
```

例如：

```text
Loss 突然升高
→ 第 860 Step 开始异常
→ PP Stage 3 的 MB 7 最先出现
→ Decoder Block 18 的 FFN 输出 amax 突增
→ FFN down projection 输出出现 Inf
→ 对应 Rank 的 FP16 MatMul 溢出
```

或者：

```text
Step Time 变长
→ PP Stage 5 出现长尾
→ MoE Block 24 的 All-to-All 耗时上升
→ Router 负载不均衡
→ 少数 Expert 接收 Token 数量过高
→ 对应 Rank 同时出现显存峰值和计算排队
```

整网图应同时支撑数值问题和性能问题的关联分析。

---

### 四、主视图不应直接堆满所有值

建议采用分层表达。

#### 第一层：整网鸟瞰

展示：

* Block / Layer / Stage 拓扑
* 节点健康状态
* 数值异常热力
* 性能异常热力
* 前向、反向或参数更新状态
* 异常传播路径

这一层回答“问题大致在哪里”。

#### 第二层：模块展开

点击 Attention、FFN、MoE 或 Residual 模块后，展示：

* 内部计算链路
* 关键张量
* 参数与 bias 是否存在
* 前向与反向统计
* 通信边界
* 不同 MB / Rank 的差异

这一层回答“是哪类对象异常”。

#### 第三层：算子与 Tensor 下钻

进一步展示：

* 真实参数名
* 规范化语义名称
* 运行时算子 ID
* 完整 Shape 和分片信息
* 统计分布或直方图
* Rank 对比
* Step 对比
* 原始采样数据
* Profiler 时间线

这一层回答“异常的具体原因是什么”。

---

### 五、标识体系需要区分三个维度

整网图中的每一个数据对象都应至少能够由以下维度唯一定位：

```text
模型位置
+ 分布式执行位置
+ 训练时间位置
```

例如：

```text
模型位置：
decoder.block.18.ffn.down_proj.output

执行位置：
pp_stage=3, tp_rank=2, dp_rank=6, device=17

时间位置：
step=860, microbatch=7, phase=forward
```

对于参数和算子，还需要关联：

```text
语义对象 ID
原始框架参数名
运行时算子 ID
```

这样才能把整网结构图、Checkpoint、Profiler 和底层算子日志对齐。

---

### 六、Weight 和 Bias 在整网图中的合理表达

Weight 和 Bias 都应作为参数对象被系统识别，但不需要全部直接摊在主视图中。

推荐方式：

* 在线性层节点上标识 `W` 和 `b` 是否存在；
* 默认显示参数健康摘要；
* 点击后展开 weight、bias 及其 gradient；
* 无 bias 的层明确显示 `bias=False` 或 `absent by design`；
* 不将不存在的 bias 伪装成零值参数；
* 参数梯度与前向激活、反向梯度分别管理；
* 参数监控需要结合并行分片和 optimizer 状态。

因此，Weight、Bias 是整网数据体系中的一部分，但不是整网图的唯一重点。更核心的是它们与激活、梯度、残差、通信、Microbatch 和 Rank 之间的关联。

---

### 七、最终产品定位

这个视图可以定义为：

**训练任务的整网运行数据图谱。**

它处在两层之间：

```text
任务级指标看板
        ↓
整网运行数据图谱
        ↓
算子 / Tensor / Rank 级调试
```

其核心价值是把训练任务的表面异常，映射到模型结构中的具体位置，并进一步关联数值对象、分布式策略、Microbatch 调度和底层运行时数据。

整网图不应成为一个“所有指标平铺的大屏”，而应成为一个支持异常发现、传播分析和逐层下钻的定位入口。
