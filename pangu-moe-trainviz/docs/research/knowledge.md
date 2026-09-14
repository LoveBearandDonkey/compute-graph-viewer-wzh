# 并行分配知识说明

这份文档的目的，是把训练并行里最容易混淆的几件事讲清楚：

- 什么叫“卡”
- 什么叫 `rank`
- 什么叫“这张卡承载哪些层/算子”
- 什么情况下这个说法准确，什么情况下不准确
- 为什么 `ParallelDemo` 里的“并行分配逻辑”是对的，但不能直接把它的示意图当成真实大规模训练拓扑

这份文档尽量不用抽象术语堆砌，而是用“快递仓库”“流水线工厂”“多人抄写一本书”这类类比来解释。

---

## 阅读目录

- [1. 先把几个基本名词分开](#1-先把几个基本名词分开)
- [1.2 stage、训练样本、batch、step、tensor 是什么](#12-stage训练样本batchsteptensor-是什么)
- [2. 最大的误区：是不是“卡按层和算子分配”？](#2-最大的误区是不是卡按层和算子分配)
- [3. DP：数据并行 Data Parallel](#3-dp数据并行-data-parallel)
- [4. FSDP：全分片数据并行 Fully Sharded Data Parallel](#4-fsdp全分片数据并行-fully-sharded-data-parallel)
- [5. PP：流水线并行 Pipeline Parallel](#5-pp流水线并行-pipeline-parallel)
- [6. TP：张量并行 Tensor Parallel](#6-tp张量并行-tensor-parallel)
- [7. SP / Sequence Parallel 与 CP / Context Parallel](#7-sp--sequence-parallel-与-cp--context-parallel)
- [8. EP：专家并行 Expert Parallel](#8-ep专家并行-expert-parallel)
- [9. 把几种并行放在一起看](#9-把几种并行放在一起看)
- [9.1 ParallelDemo 的 dp×pp×cp×tp 网格公式如何匹配](#91-和-paralleldemo-的逻辑关系是否匹配)
- [9.2 EP / MoE 在 ParallelDemo 中为什么要单独说明](#92-ep--moe-在-paralleldemo-里要单独说明)
- [9.3 并行数量是不是开发者可以自己设置](#93-这些并行数量是不是开发者可以自己设置)
- [9.4 盘古 MoE 单卡内部会呈现哪些计算](#94-一个卡内部会受到并行策略怎样影响)
- [10. 为什么 ParallelDemo 的逻辑对，但示意图不能直接当真实盘古拓扑](#10-为什么-paralleldemo-的逻辑对但示意图不能直接当真实盘古拓扑)
- [10.4 如何把 ParallelDemo 落实成确定的盘古训练配置](#104-如何把-paralleldemo-落实成确定的盘古训练配置)
- [10.5 整网模型架构、训练步骤、单卡任务的关系](#105-整网模型架构训练步骤单卡任务的关系)
- [10.5.4 一张卡的时间轴是不是乱序任务](#1054-一张卡的时间轴是不是乱序任务)
- [10.5.5 block、layer、隐藏层、hidden state 的关系](#1055-blocklayer隐藏层hidden-state-的关系)
- [11. 对 TrainScope 的产品表达建议](#11-对-trainscope-的产品表达建议)
- [11.4 推荐的单卡详情结构](#114-推荐的单卡详情结构)
- [12. 给页面文案直接可用的几种模板](#12-给页面文案直接可用的几种模板)
- [13. 最终结论](#13-最终结论)
- [14. 官方与可靠代码依据](#14-官方与可靠代码依据)

---

## 1. 先把几个基本名词分开

### 1.1 卡、device、rank、worker 不是一回事

很多讨论里，大家会把这些词混着说，但它们其实不是同一个层级。

#### 卡 / device

- 指物理硬件，比如 1 张 Ascend 910B、1 张 H100。
- 它是“机器上的一块板子/一个计算设备”。

类比：

- 一张卡，像工厂里的一台机器。

#### rank

- 指分布式训练里的一个进程身份，或者说一个通信成员编号。
- 很多 demo 里常见“1 个 rank 绑定 1 张卡”，所以大家容易把 rank 直接等同于卡。
- 但严格说，`rank` 是软件/运行时语义，`card` 是硬件语义。

类比：

- `rank` 像工厂里的一个工位编号。
- 工位通常坐在某台机器前面工作，所以“工位”和“机器”经常配对，但不是同一个概念。

#### worker

- 常是更宽泛的说法，可以指一个训练进程，也可以指一个参与计算的执行单元。
- 在不同框架里，`worker` 的精确含义可能略有不同。

#### 为什么产品里最好写 rank 和 card 两层

因为产品里一旦要表达：

- 物理位置：host / slot / device_id
- 运行时位置：global rank / local rank
- 并行坐标：TP / PP / DP / CP / EP

你就不能只写“这张卡是什么”，也不能只写“这个 rank 是什么”。

更准确的表达应该是：

`rank_237 运行在 host_03 的 device_5 上，属于 PP5 / TP2 / DP7 / EP19`

### 1.2 stage、训练样本、batch、step、tensor 是什么

这些词容易混在一起，但它们不是同一个维度。

| 概念 | 准确定义 | 类比 | 在图里应该怎么表达 |
|---|---|---|---|
| `训练样本 sample` | 一条训练数据。对语言模型来说，通常是一段 token 序列，模型用前面的 token 预测后面的 token。 | 一份学生作业。 | `sample 0-511`、`样本 shard A`。 |
| `token` | 文本被 tokenizer 切出来的最小训练单位之一，例如一个字、词片段或符号。 | 作业里的一个字/词。 | `sequence length`、`token 2048-4095`。 |
| `tensor` | 计算机里承载数据、权重、激活、梯度的多维数组。样本被编码后会变成 tensor，但 tensor 不等于样本。 | 装在表格里的数字矩阵。 | `input_ids tensor`、`hidden_states tensor`、`QKV weight tensor`。 |
| `batch` | 一个训练 step 一次送进模型的一组样本。 | 一摞作业。 | `one step batch · 32 samples`。 |
| `mini-batch shard` | 在 DP/CP 等并行下，一个 rank 实际处理的 batch 子集。 | 每个老师分到的一部分作业。 | `rank_1: samples 512-1023`。 |
| `step` | 一次训练迭代。通常包括取 batch、forward、loss、backward、通信同步、optimizer update。 | 完整批改一摞作业并更新答案标准的一轮。 | `step 1997`、`current step`。 |
| `stage` | 在 PP（Pipeline Parallel，流水线并行）里，把整网层序列切成的一个流水线阶段。 | 工厂流水线的一段工位。 | `Stage 0`、`PP5 · layers 20-23`。 |

关键区别：

- `sample / batch / mini-batch shard` 是数据维度，回答“这次喂了哪些训练数据”。
- `tensor` 是计算表示，回答“数据、权重、激活、梯度在机器里以什么数组形态存在”。
- `stage` 是模型切分维度，回答“整网前向链路中哪一段层放在这个流水线阶段”。
- `step` 是时间维度，回答“训练跑到了第几次迭代”。

所以，`DP` 图里“不同 rank 吃不同数据”说的是 `mini-batch shard` 不同，不是模型结构不同。`PP` 图里的 `Stage 0` 说的是模型层段不同，不是第 0 个训练 step。

以 `ParallelDemo` 截图为例：

```text
one step batch · 32 训练样本
        |
        v
Stage 0
CP0
EP组0
H100_0
Emb (第1/4片)
Attn-0 (第1/4片)
MoE-0 [E0-E1]
```

这句话的含义是：在当前训练 step 里，某一组样本进入流水线；`H100_0` 处在 `Stage 0`，所以它执行整网开头部分的 Embedding 和 Block 0；同时它是 TP 第 1/4 片，所以这些算子内部的权重/激活不是完整一份，而是第 1 个张量分片。

---

## 2. 最大的误区：是不是“卡按层和算子分配”？

答案是：

- **有些并行方式下，这么说基本对**
- **有些并行方式下，这么说只对一半**
- **有些并行方式下，这么说基本不对**

最关键的一句是：

> 真实训练里，并不是所有并行都在回答“这张卡负责哪几层”。
> 更常见的是：不同并行维度分别决定“这张卡持有哪些层段、哪些参数分片、哪些专家、哪些序列分片、以及它属于哪些通信组”。

也就是说：

- `PP` 更像“按层段切”
- `TP` 更像“同一层内部按矩阵/算子切”
- `DP` 更像“整模型复制多份，各吃不同数据”
- `FSDP` 更像“参数/梯度/优化器状态切片保存”
- `SP/CP` 更像“序列切片”
- `EP` 更像“专家切片”

下面一类一类讲。

---

## 3. DP：数据并行 Data Parallel

### 3.1 它到底做什么

数据并行的核心不是切模型，而是：

- **每个 rank 都有同一份模型**
- **每个 rank 吃不同的数据 batch shard**
- **算完梯度后，把梯度同步**

官方依据：

- PyTorch DDP 文档明确说：`DistributedDataParallel` 通过在每个模型副本之间同步梯度来提供 data parallelism。
- 同时文档也明确说：DDP **不会**自动把输入切到各 GPU，用户要自己决定怎么切，例如用 `DistributedSampler`。

来源：

- https://docs.pytorch.org/docs/2.12/generated/torch.nn.parallel.DistributedDataParallel.html

### 3.2 类比

想象有 8 个老师一起批改同一本练习册的不同页：

- 每个老师手里都有同样的“答案标准”
- 但每个人批改的学生作业页不同
- 最后大家把“改出来的经验”汇总一下，保证答案标准继续保持一致

这里：

- “答案标准” = 模型副本
- “不同页作业” = 不同数据分片
- “最后汇总经验” = 梯度同步

### 3.3 它不做什么

它**不**意味着：

- GPU0 负责 Embedding
- GPU1 负责 Attention
- GPU2 负责 MLP

这不是 DP 的语义。

### 3.4 所以页面里怎么说才准确

不要写：

- `这张卡承载了第 12-24 层（DP）`

应该写：

- `这个 rank 属于 DP7 副本组，运行完整模型副本，处理本 step 的一部分数据 shard`

---

## 4. FSDP：全分片数据并行 Fully Sharded Data Parallel

### 4.1 它到底做什么

FSDP 还是数据并行体系，但它进一步节省显存：

- 参数是分片存的
- 梯度是分片存的
- 优化器状态也是分片存的

官方文档里 `FULL_SHARD` 的定义非常直接：

- 参数、梯度、优化器状态都被切分
- 前向前 `all-gather` 把需要的参数拼出来
- 前向后再重新分片
- 反向前再拼
- 反向后再分片

来源：

- https://docs.pytorch.org/docs/2.12/fsdp.html

### 4.2 类比

想象一本特别厚的参考书：

- DDP：每个老师都抱着一本完整书
- FSDP：8 个老师每人只带书的 1/8
- 轮到某个章节要用时，大家临时把这章拼起来看
- 用完再拆回去，各自只带自己那一份

### 4.3 它是不是按层分配

**不是它的核心语义。**

它的重点是：

- 参数怎么切
- 什么时候 gather
- 什么时候 scatter

虽然工程实现上常常“按模块粒度 wrap”，看起来像“这一块参数归这个 wrapper 管”，但 FSDP 本质上不是在表达“哪张卡负责哪几层”，而是在表达“参数和状态如何被分片管理”。

### 4.4 页面里怎么说才准确

不要写：

- `卡 12 负责 MLP 层`

可以写：

- `该 rank 持有模型参数/梯度/优化器状态的一部分分片；前后向期间按需 all-gather`

---

## 5. PP：流水线并行 Pipeline Parallel

### 5.1 它到底做什么

这类并行最接近“按层分配”。

官方 PyTorch pipeline 文档写得很明确：

- 要先构造 `PipelineStage`
- 一个 `PipelineStage` 包装“这个 stage 上运行的那一部分模型”
- 官方示例里甚至是直接删掉本 stage 不需要的层，再创建 stage

来源：

- https://docs.pytorch.org/docs/2.12/distributed.pipelining.html

### 5.2 类比

像工厂流水线：

- 第 1 站负责上料和前几道工序
- 第 2 站负责中间加工
- 第 3 站负责最后组装

一件产品从第 1 站流到第 2 站，再到第 3 站。

这里：

- `PP stage 0` 可能负责 embedding + 前几层 block
- `PP stage 1` 负责中间一段 block
- `PP stage 2` 负责最后几层 + LM Head

### 5.3 所以“每张卡承载哪些层”在这里准不准

**在 PP 语义下，这句话基本是准确的。**

更精确一点：

- 不是“每张卡”
- 而是“每个 pipeline stage 上的 rank / device”

它承载的是：

- 一段连续层
- 或一个 stage 的模型子模块

### 5.4 页面里怎么说才准确

可以写：

- `PP5 承载第 36-42 层`
- `该 rank 处于 pipeline stage 5，执行这段层的前后向`

这类说法是靠谱的。

---

## 6. TP：张量并行 Tensor Parallel

### 6.1 它到底做什么

TP 不是把不同层分给不同卡，而是把**同一层内部的线性层/张量**切开。

官方 PyTorch Tensor Parallel 文档说明：

- `ColwiseParallel`：按列切 compatible `nn.Module`
- `RowwiseParallel`：按行切 compatible `nn.Module`
- 二者可以组合起来实现更复杂模块，比如 `MLP`、`Attention`

来源：

- https://docs.pytorch.org/docs/2.12/distributed.tensor.parallel.html

### 6.2 类比

假设一张超大的表格要算矩阵乘法：

- 不是让 A 同学算前 10 页、B 同学算后 10 页
- 而是让 4 个人同时算**同一页上的不同列块/行块**

也就是说：

- 大家都在算“同一层”
- 只是每个人算的是这一层里面不同的权重分片或输出分片

### 6.3 为什么“承载哪些层/算子”只对一半

如果你说：

- `这张卡正在参与 Attention`

这没问题。

但如果你说：

- `这张卡独自承载了 Attention 这一层`

这就不对了，因为 TP 下这层通常是多卡一起完成的。

更准确的表达是：

- `这张卡承载 Attention/MLP 的 TP 分片`
- `它持有该层某个线性算子的列分片或行分片`

### 6.4 页面里怎么说才准确

不要写：

- `卡 7 负责 Attention 层`

应该写：

- `卡 7 / rank_7 持有 Attention QKV 的 TP shard 2/8`
- `卡 7 参与该层张量并行组的同层协同计算`

---

## 7. SP / Sequence Parallel 与 CP / Context Parallel

这两个名字在不同框架/论文里有时会有差异，但对产品表达来说，核心抓住一点就够了：

- 它们主要在表达**序列维度被切开**
- 不是在表达“层归哪张卡”

### 7.1 官方语义里 Sequence Parallel 是什么

PyTorch 文档对 `SequenceParallel` 的定义是：

- 模块参数是 replicated
- 计算在 sequence 维已经切开的输入上进行
- 输出继续沿 sequence 维切分

来源：

- https://docs.pytorch.org/docs/2.12/distributed.tensor.parallel.html

### 7.2 类比

想象一本很长的小说：

- 不是把前半本交给甲老师、后半本交给乙老师去“拥有”
- 而是把同一句子序列切成前半段和后半段，大家各自处理一段 token

更像：

- 同样的处理规则
- 不同的人处理长序列的不同片段

### 7.3 它是不是按层分配

**不是。**

它主要回答的是：

- 序列如何切
- 长上下文如何分摊到多个 rank

### 7.4 页面里怎么说才准确

不要写：

- `CP3 承载第 20 层`

可以写：

- `CP3 负责当前序列窗口的第 4 个 context shard`
- `参数不因 CP 而天然按层切走，重点是 token / sequence 维切分`

---

## 8. EP：专家并行 Expert Parallel

### 8.1 它到底做什么

这是 MoE 最关键的一类。

Megatron Core 官方文档写得很清楚：

- 专家被分配到不同 worker
- 每个 worker 在每个 MoE 层处理一个或多个专家

DeepSpeed 官方文档也写得很直接：

- 一个 `ep_size` 大小的 expert-parallel group 中，参与的 GPUs/ranks 会分配该层的总专家数

来源：

- https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/moe.html
- https://www.deepspeed.ai/tutorials/mixture-of-experts/

### 8.2 类比

想象一家医院：

- 有心内科专家、骨科专家、眼科专家
- 病人先经过分诊台（router / gate）
- 分诊台决定把病人送去哪个专家
- 不同医生在不同诊室里接待各自擅长的病人

这里：

- 分诊台 = router / gate
- 医生 = experts
- 不同诊室 = 不同 rank/device 上的专家分布

### 8.3 它是不是按层分配

**不该概括成“按层分配”。**

更准确是：

- 在每个 MoE 层内部，有一组 experts
- 这些 experts 被分散到不同 rank 上

所以它回答的是：

- `这个 rank 上放了哪些 expert`

而不是：

- `这个 rank 负责第几层`

### 8.4 为什么这里很容易和“卡”混淆

因为 demo 常常会画成：

- `expert_group_19 -> rank_19 -> card_19`

这只是为了演示方便。

真实大规模训练里，不应该默认：

- 一个 expert bucket 永远对应一张卡
- 一个 rank 永远只放一个 expert group

真正该表达的是：

- `某个 expert bucket 当前 placement 投影到哪些 rank/device`

### 8.5 页面里怎么说才准确

不要写：

- `卡 19 负责 MoE 层`

可以写：

- `rank_19 持有当前 MoE 层的一部分 experts / expert shards`
- `Gate 会把 token 路由到对应 expert 所在的 rank`

---

## 9. 把几种并行放在一起看

真实大模型训练通常不是只开一种并行，而是叠加。

例如一个 rank 可能同时有这些身份：

- 它属于 `PP5`：说明它负责某一段层
- 它属于 `TP2`：说明这一段层里的线性层，它只持有第 2 个张量分片
- 它属于 `DP7`：说明它所在的是第 7 个数据并行副本组
- 它属于 `CP1`：说明它处理长序列中的某个上下文分片
- 它属于 `EP19`：说明它上面放着某些 MoE experts

所以真实描述应该是“多维坐标”，不是单一归类。

类比：

一个工人可能同时具备这些标签：

- 在第 5 条流水线工位上班（PP）
- 只负责组装零件的第 2 个子部件（TP）
- 今天处理第 7 批订单样本（DP）
- 订单说明书只看第 1 段（CP）
- 遇到心脏病类病例时归他处理（EP）

这比“他负责某一层”要丰富得多，也更接近真实训练。

### 9.1 和 `ParallelDemo` 的逻辑关系是否匹配

匹配，但要说准确：

`ParallelDemo` 不是简单地给一张卡贴 5 个标签，而是先根据开发者配置的并行度生成多维 rank 网格，再把同一个 rank 投影到不同并行坐标上。

在当前 `ParallelDemo` 实现里，核心公式是：

```text
world_size = dp * pp * cp * tp
grid = arange(world_size).view(dp, pp, cp, tp)
rank = ((d * pp + p) * cp + c) * tp + t
```

其中：

- `d` 是 DP（Data Parallel，数据并行）坐标
- `p` 是 PP（Pipeline Parallel，流水线并行）坐标
- `c` 是 CP（Context Parallel，上下文并行）坐标
- `t` 是 TP（Tensor Parallel，张量并行）坐标

所以一个 rank 不是“属于某一层”，而是有一组坐标：

```text
rank_6 = d0 / p0 / c1 / t2
```

这表示：

- 它在第 0 个 DP（Data Parallel，数据并行）副本里
- 它在第 0 个 PP（Pipeline Parallel，流水线并行）stage 里
- 它处理第 1 个 CP（Context Parallel，上下文并行）序列分片
- 它持有第 2 个 TP（Tensor Parallel，张量并行）张量分片

类比：

一个工人不是“被分成四个人”，而是同时有四个身份：

- 属于第几条订单线
- 站在第几道工序
- 只看说明书的哪一段
- 只加工一个零件的哪一块

因此白皮书里“同一 rank/device 上叠加多个并行语义标签”这个概念是准确的；更精确的说法是：

> 同一个 rank 在不同并行维度的通信组和切分维度上，各有一个坐标。

### 9.2 EP / MoE 在 `ParallelDemo` 里要单独说明

当前 `ParallelDemo` 的 `world_size` 只由：

```text
dp * pp * cp * tp
```

决定。

MoE / EP（Expert Parallel，专家并行）在这个 demo 里主要影响：

- 当前 rank 持有哪些 experts
- token 经 router / gate 后路由到哪些 experts
- MoE 层是否显示 `all-to-all dispatch/combine`

也就是说，`ParallelDemo` 当前把 EP（Expert Parallel，专家并行）作为专家放置与 token 路由语义叠加显示，而不是把它作为 `world_size` 的一个乘法维度。

这点在产品里必须写清楚，否则容易让人误以为通用公式一定是：

```text
world_size = dp * pp * cp * tp * ep
```

真实框架里是否把 EP（Expert Parallel，专家并行）作为独立 mesh 维度，要看具体框架和配置。对 TrainScope 来说，更稳妥的数据模型是：

- `dp / pp / cp / tp` 用来生成基础 rank mesh
- `ep / expertIds / expertShardRange / gateRoute` 作为 MoE 放置语义单独建模
- 如果真实训练框架确实有独立 EP 维度，再显式加入 placement 数据，而不是默认绑定到 `tp` 或 `cardId`

### 9.3 这些并行数量是不是开发者可以自己设置

可以配置，但不是任意配置。

以 MindSpore Transformers / Ascend 生态的配置口径看，配置里会出现类似：

- `parallel_config.data_parallel`
- `parallel_config.model_parallel`
- `parallel_config.context_parallel`
- `parallel_config.pipeline_stage`
- `layers_per_stage`
- MoE 里的 `expert_num`、`num_experts_chosen`、专家计算/通信并行相关字段

这些字段说明并行度确实是训练配置的一部分，开发者或训练系统可以设置。

但它们必须满足约束，例如：

- 设备数必须够用
- `device_num` 要能被各并行维度的乘积解释
- PP（Pipeline Parallel，流水线并行）的 stage 数要和层数分配匹配
- TP（Tensor Parallel，张量并行）通常要和 hidden size、attention heads、FFN 维度可整除
- CP（Context Parallel，上下文并行）要和 sequence length / context length 以及 attention 实现匹配
- DP（Data Parallel，数据并行）要和 global batch / micro batch / gradient accumulation 匹配
- EP（Expert Parallel，专家并行）要和 expert 数量、top-k、节点内 NPU 数、all-to-all 通信成本匹配

类比：

这像设计工厂：

- 你可以决定开几条流水线
- 每条线分几道工序
- 每道工序几个人协作
- 每批订单拆成几份

但你不能随便填数字。机器数量、零件尺寸、工序顺序、运输成本都要对得上。

### 9.4 一个卡内部会受到并行策略怎样影响

一个卡内部显示什么，不是只由“这张卡号”决定，而是由这个 rank 的多维坐标共同决定。

以盘古 MoE 训练视图为例，可以这样拆：

| 并行策略 | 对单卡 / 单 rank 内部的影响 | 页面上应呈现什么 |
|---|---|---|
| PP（Pipeline Parallel，流水线并行） | 决定该 rank 处于哪个 stage，以及这个 stage 覆盖哪些连续层段 | `PP0 · Embedding + Block 0`、`PP5 · layers 36-42` |
| TP（Tensor Parallel，张量并行） | 决定同一层内部的权重、张量或算子切片 | `Embedding vocab shard`、`QKV column shard 2/4`、`Out row shard`、`MLP up/down shard` |
| CP（Context Parallel，上下文并行） | 决定该 rank 处理哪段 token / context，以及 attention 的 KV 或中间结果如何交换 | `CP1 · token 2048-4095`、`KV exchange`、`Ring Attention` |
| DP（Data Parallel，数据并行） | 决定该 rank 吃哪份样本 / mini-batch shard，以及 backward 后和谁同步梯度 | `samples 512-1023`、`gradient all-reduce group` |
| EP（Expert Parallel，专家并行） / MoE | 决定该 rank 持有哪些 experts，token 被 gate 路由到哪里 | `router / gate`、`experts E4-E5`、`all-to-all dispatch/combine` |

所以盘古 MoE 单卡详情不应该只写：

```text
这张卡负责第几层
```

更应该写成：

```text
rank_6 / device_6
PP0: Embedding + Block 0
TP2: QKV / MLP 第 3/4 片
CP1: 当前 sequence shard
DP0: 当前 batch shard
MoE/EP: 持有 experts E4-E5，参与 token all-to-all 路由
```

这才和 `ParallelDemo` 的逻辑一致，也能扩展到真实盘古训练需要的卡数。

---

## 10. 为什么 `ParallelDemo` 的逻辑对，但示意图不能直接当真实盘古拓扑

### 10.1 它对在哪里

`ParallelDemo` 对的地方，是它把并行分配逻辑算对了：

- `rankOf(d,p,c,t)` 这类 rank 坐标推导
- `tp / pp / dp / cp` 分组关系
- `distributeLayers()` 的 pipeline 层分配

也就是说，它对的是“placement 计算逻辑”。

更细一点说，它对的是这条链路：

```text
开发者配置 dp / pp / cp / tp
  -> 生成 rank mesh
  -> 推导 TP / PP / CP / DP 通信组
  -> 根据 PP 决定层段
  -> 根据 TP / CP / DP / EP 影响卡内计算切片和通信语义
```

### 10.2 它不能直接复用在哪里

它的图还是 demo 视图：

- 只画少量卡
- 用可读性优先的示意块
- 没打算直接承载 1k/4k 卡规模的真实训练集群

所以不能把它的“几张卡示意图”直接当成 `TrainScope` 的真实物理轴。

### 10.3 正确做法

应该复用：

- `ParallelDemo` 的 placement engine

不应该直接复用：

- `ParallelDemo` 的卡片示意排版

也就是：

- **逻辑复用**
- **视图重做**

TrainScope 的图表建议分成两层：

- 总览图：展示 cluster / host / device / rank mesh，不平铺所有卡的内部细节
- 单卡详情：展示该 rank 的 PP / TP / CP / DP / EP 坐标，以及盘古 MoE 的实际计算切片

如果要在白皮书里画图，推荐画成：

```text
公开模型配置
openPangu-Ultra-MoE-718B-V1.1: 61 decoder layers / hidden 7680 / 3 Dense + 58 MoE / 256 routed experts / top-8
        |
        v
派生训练策略
DP=32 / PP=16 / CP=1 / TP=8 / EP=32
        |
        v
rank mesh
grid[d][p][c][t], world=4096
        |
        v
通信组投影
TP group / PP group / CP group / DP group / EP placement
        |
        v
单卡计算切片
按 stage 展示：Stage0 含 Embedding；中间 stage 含 Attention + Dense/MoE FFN；末尾 stage 含 Final Norm / LM Head
再叠加 TP/CP/EP/DP 对应的 tensor shard、sequence shard、expert bucket、batch shard 和通信组
```

### 10.4 如何把 `ParallelDemo` 落实成确定的盘古训练配置

建议把白皮书里的图标题定为：

```text
把 ParallelDemo 落实成真实训练配置 · 从公开模型到单卡计算
```

这里的“真实训练配置”要谨慎理解：它不是说我们已经拿到了 openPangu 官方完整训练 placement，而是说我们用公开模型结构和官方并行配置口径，构造一套可落地、可解释、可扩展到真实卡数的训练诊断样例。

推荐固定成这个样例：

| 层级 | 配置 | 可靠性 |
|---|---|---|
| 模型结构 | openPangu-Ultra-MoE-718B-V1.1：61 个 decoder layers（L0-L2 Dense，L3-L60 MoE）、hidden 7680、vocab 153600、256 路由专家、1 共享专家、top-8、MLA、sandwich norm | 已由本地 `model_architecture.json` 校验为 `model_architecture.v1`：49 节点 / 56 边 / 0 error |
| 基础 rank mesh | `DP=32 / PP=16 / CP=1 / TP=8`，所以 `world_size = 32 × 16 × 1 × 8 = 4096` | 产品派生 placement，不声称是官方真实训练切分 |
| MoE / EP | `EP=32`，把 256 experts 聚合成 32 个 expert bucket，每桶 8 个 experts | 产品派生 expert placement，用于解释专家路由和 All-to-All |
| 动态遥测 | loss、grad norm、router z-loss、expert load、All-to-All bytes、rank util | 需要自采 Ascend run；没有自采时必须标注 synthetic / reprojected |

注意：V1.1 `config.json` 声明了 `num_mtp_layers=1`，但当前 `inference/model.py` forward 路径没有实例化 MTP head；因此 MTP 只能作为 scope note，不能画进 canonical architecture 主图。EP/All-to-All 也是训练 placement / runtime projection 语义；源码事实层只证明 MoE router、NPU init/finalize routing、grouped matmul、conditional MoE All-Reduce 等结构。

这样落图时，可以选一个具体 rank 讲清楚：

```text
rank_299
d=2, p=5, c=0, t=3

DP2: 第 2 个数据并行副本
PP5: 第 5 个流水线 stage，示例层段 layers 20-23
CP0: 本例不做 context 切分
TP3: 第 4/8 个张量分片
EP19: 示例 expert bucket，experts 152-159
```

它对应的页面表达不是：

```text
910B_299 负责第 20-23 层
```

而是：

```text
rank_299 运行在 910B_299 上；
它处在 PP5，所以只执行该 stage 的层段；
它处在 TP3，所以该 stage 内 Attention 的 QKV / Out，以及 FFN / expert 投影只拿第 4/8 片；
它处在 DP2，所以吃的是第 2 个数据副本的 mini-batch shard；
它处在 CP0，所以本例不切上下文；
它叠加 EP19，所以 MoE FFN 区域显示 expert bucket E152-E159 以及 all-to-all dispatch/combine。
```

这和 `ParallelDemo` 截图的关系要分清：

- `ParallelDemo` 截图里的 `H100_0` 是 `Stage 0 / CP0 / EP组0 / TP0`，所以卡内显示 `Emb (第1/4片)`、`Attn-0 (第1/4片)`、`MoE-0 [E0-E1]` 是准确的。
- `rank_299 / PP5` 不是 `Stage 0`，所以不应照搬 `Embedding`。它应显示 `Blocks 20-23` 内部的 `Attention + MoE FFN` 切片；只有 `PP0 / Stage0` 才额外显示 Embedding，最后一个 stage 才额外显示 Norm / LM Head。
- `ParallelDemo` 当前 demo 里，MoE 卡片的专家范围由 `experts / tp` 计算；在小 demo 中常把 `tp` 和 `ep` 设成相同数值，所以看起来像专家并行。盘古产品场景里如果要表达 `EP=32`，就应把 `expert bucket` 独立建模，例如 `EP19 -> E152-E159`，不要把它说成 ParallelDemo 原生代码直接推导出的结果。
- 因此，准确结论是：ParallelDemo 的 `dp × pp × cp × tp` 网格和 `PP stage -> 卡内层段 -> TP/CP 切片` 逻辑应被复用；盘古 MoE 的 `EP bucket` 是在此基础上新增的专家放置语义。

这才是“把 ParallelDemo 落实成真实训练配置”的含义：复用 `ParallelDemo` 的 rank mesh 公式和通信组推导，但把小 demo 数字换成盘古可解释样例，并且明确标注哪些是官方公开、哪些是产品推导、哪些是合成或待自采。

### 10.5 整网模型架构、训练步骤、单卡任务的关系

先回答一个核心问题：**一个模型的架构是不是计算步骤？**

更准确地说：

- 模型架构是“前向计算的结构模板”，定义输入如何经过 Embedding、Attention、MLP/MoE、Norm、Head 变成 logits。
- 训练步骤是“一次训练迭代的执行流程”，它会调用模型架构做 forward，但还会额外包含 loss、backward、通信同步、优化器更新。
- 单卡任务是“整网架构在某个 rank/device 上的投影”，由 PP/TP/CP/DP/EP placement 决定，而不是整网架构本身。

类比：

- 模型架构像一条完整生产线图纸：原料从入口走到出口，中间有哪些工序。
- 训练步骤像一天里生产一批货的完整工作流：上料、生产、质检、发现误差、调整机器参数。
- 单卡任务像某个工位今天实际负责的那部分工序：它不等于整条生产线，但必须能追溯回整条生产线上的位置。

#### 10.5.1 盘古整网模型架构能确定什么

基于本地 `openPangu-Ultra-MoE-718B-V1.1` 配置、推理代码和已校验的架构事实层，可以确定整网前向结构的大框架：

```text
Token IDs / Position IDs / Attention Mask
        |
        v
Parallel Embedding
        |
        v
Decoder Layers × 61
  每个 decoder layer 都包含：
  Input RMSNorm
    -> MLA Attention
       q_a -> q_ln -> q_b
       kv_a -> kv_ln -> K/V absorb + RoPE + compressed KV Cache
       score -> softmax -> context -> output proj -> conditional Attn All-Reduce
  Post-Attn RMSNorm
    -> Pre-MLP RMSNorm
    -> FFN branch:
       L0-L2  : Dense MLP -> conditional Dense All-Reduce
       L3-L60 : MoE FFN
                 Router: linear -> sigmoid -> TopK -> gather -> normalize/scale
                 NPU routing -> grouped up-gate -> SwiGLU -> grouped down -> finalize routing
                 Shared Expert MLP -> combine -> conditional MoE All-Reduce
  Post-MLP RMSNorm
        |
        v
Final RMSNorm
        |
        v
LM Head -> conditional Logits All-Gather -> Logits
```

为了避免误读，可以把它再压缩成面向产品的折叠图：

```text
Token IDs / Position IDs / Attention Mask
        |
        v
Parallel Embedding
        |
        v
Dense Decoder Layers × 3
  每层：MLA Attention + Dense MLP
        |
        v
MoE Decoder Layers × 58
  每层：MLA Attention + MoE FFN
  MoE FFN：Router / Gate Top-8 -> Routed Experts × 256 + Shared Expert × 1 -> Combine
        |
        v
Final RMSNorm
        |
        v
LM Head -> conditional Logits All-Gather -> Logits
```

可靠依据：

- `config.json` 明确给出 `num_hidden_layers=61`、`num_dense_layers=3`、`hidden_size=7680`、`vocab_size=153600`、`num_routed_experts=256`、`num_shared_experts=1`、`num_experts_per_tok=8`、`num_attention_heads=128`、`attention_q_lora_dim=1536`、`attention_kv_lora_dim=512`、`attention_qk_dim=128`、`attention_qk_rope_dim=64`、`attention_v_dim=128`、`intermediate_size=18432`、`moe_intermediate_size=2048`、`sandwich_norm=true`、`max_position_embeddings=131072`、`rope_theta=25600000`、`num_mtp_layers=1` 等结构参数。
- `inference/model.py` 里的 `PanguUltraMoEModel.forward()` 会顺序遍历 `self.layers`，最后做 `self.norm`；`PanguUltraMoEForCausalLM.forward()` 会调用 `self.model(...)` 后接 `lm_head` 得到 logits。
- 同一份代码里，`PanguUltraMoE.forward()` 包含 `gate`、`moe_npu`、`shared_experts`；`moe_npu()` 包含 NPU MoE routing、expert token 统计、expert 计算和 finalize routing；`PanguUltraMoEAttention` 包含 MLA 相关 Q/K/V 低秩投影与输出投影。
- `/Users/yin/gitcode/openPangu-Ultra-MoE-718B-V1.1/outputs/model_architecture.json` 是规范架构事实层，schema 为 `model_architecture.v1`，校验器结果 `ok: True`，共 49 nodes / 56 edges / 0 errors。

边界要说清楚：这些来源能证明“模型前向结构”和“推理侧 NPU MoE 路径”，但不能证明官方完整训练 placement，也不能证明每个训练 step 的真实通信流量。`num_mtp_layers=1` 只证明 config 声明了 MTP；由于当前 forward 未构建 MTP head，产品图里应把 MTP 放在 scope note，而不是画成实际前向节点。残差 add 在源码中由 fused RMSNorm 路径处理，也不应额外画成独立 Add 节点。MLA 的 `KV Cache` 存的是压缩 latent + RoPE key（`512+64`），不是展开后的完整 K/V。

#### 10.5.2 训练步骤比模型架构多什么

训练时不是只跑一遍模型前向。一个标准训练 step 至少包含：

```text
取当前 step 的 batch
        |
        v
forward：按模型架构计算 logits
        |
        v
loss：把 logits 和 label 比较，得到损失
        |
        v
backward：反向传播，计算梯度
        |
        v
distributed communication：按 DP/TP/PP/CP/EP 做 all-reduce、all-gather、send/recv、all-to-all 等
        |
        v
optimizer step：根据梯度更新参数
        |
        v
进入下一个 step
```

可靠依据：

- PyTorch 官方训练示例的核心循环是 `optimizer.zero_grad()`、`outputs = net(inputs)`、`loss = criterion(outputs, labels)`、`loss.backward()`、`optimizer.step()`。这说明 forward 只是训练 step 的一部分。
- PyTorch autograd 语义说明 `.backward()` 负责从 loss 反向计算并累积梯度。
- MindSpore Transformers 文档说明大模型训练中可以组合数据并行、模型并行、流水线并行、序列并行等策略；这些策略会把 forward/backward/通信/显存状态分散到多设备上。

因此，TrainScope 的“计算轴”不应该只画静态模型结构，也要允许叠加训练 step 的动态信号：

- loss、grad norm、weight/activation stats
- router z-loss、expert load、token dispatch
- all-reduce / all-gather / all-to-all / P2P send-recv
- optimizer update 是否发生、是否跳步、是否溢出

#### 10.5.3 每张卡上的任务和整网架构是什么关系

每张卡上的任务不是独立发明出来的，它来自“整网架构 × 并行 placement”的交集。

| 并行维度 | 它怎样投影整网架构 | 对单卡任务的影响 |
|---|---|---|
| PP（Pipeline Parallel，流水线并行） | 把整网层序列切成多个 stage。 | 决定这张卡处于哪段层，例如 `PP5 -> Blocks 20-23`。Stage 0 可能包含 Embedding，最后 stage 可能包含 Final Norm / LM Head。 |
| TP（Tensor Parallel，张量并行） | 把同一层内部的权重矩阵、attention heads 或输出维度切片。 | 决定这张卡持有 Attention QKV / Out、FFN up/down、Embedding / LM Head 的第几片。 |
| CP/SP（Context/Sequence Parallel，上下文/序列并行） | 把序列长度或上下文窗口切片。 | 决定这张卡处理哪段 token，以及 attention/KV/LayerNorm/Dropout 等是否需要跨卡交换。 |
| DP（Data Parallel，数据并行） | 复制同一份模型计算图，让不同副本吃不同样本。 | 决定这张卡处理哪个 mini-batch shard；模型结构相同，样本不同。 |
| EP（Expert Parallel，专家并行） | 在 MoE 层里把 experts 放到不同 rank/device 上。 | 决定这张卡持有哪些 expert bucket，以及 token 是否经 all-to-all dispatch/combine 到这里。 |

把这张表套到 `rank_299 / PP5 / TP3 / CP0 / DP2 / EP19`：

```text
整网架构：openPangu-Ultra-MoE-718B
        |
        v
PP5 选择整网中的 Blocks 20-23
        |
        v
TP3 选择这些 blocks 里 Attention / FFN / expert 投影的第 4/8 片
        |
        v
CP0 表示本例不切上下文
        |
        v
DP2 表示吃第 2 个数据副本的 mini-batch shard
        |
        v
EP19 表示 MoE 层的 expert bucket E152-E159 会投影到该 rank/device
```

所以准确表达不是：

```text
这张卡负责盘古模型的一部分架构
```

而是：

```text
这个 rank/device 执行整网架构在 PP5 层段上的前后向；
在这些层内部，它只持有 TP3 张量分片；
在 MoE FFN 处，它还叠加 EP19 expert bucket；
在当前 step 中，它处理 DP2 的训练样本 shard，并参与对应通信组。
```

#### 10.5.4 一张卡的时间轴是不是乱序任务

不是乱序。按照时间维度看，一张卡上的 profiler timeline 可能会出现很多交错事件，但这些事件由训练 step、pipeline schedule、autograd 依赖和通信依赖共同决定，不是随机排列。

一个 `rank_299 / PP5 / TP3 / DP2 / CP0 / EP19` 的中间 stage，典型时间轴可以理解为：

```text
从 PP4 接收 activation
        |
        v
forward：执行 PP5 覆盖的 Blocks 20-23
  - Attention QKV / Out 的 TP3 张量分片
  - MoE router / gate
  - all-to-all dispatch
  - 本 rank 的 expert bucket E152-E159
  - all-to-all combine
        |
        v
向 PP6 发送 activation
        |
        v
稍后从 PP6 接收 gradient
        |
        v
backward：反向穿过 Blocks 20-23
        |
        v
TP / EP / DP 等通信同步
        |
        v
optimizer update 或等待全局优化器阶段
```

如果采用 pipeline 1F1B 之类调度，同一张卡上会看到类似下面的交错：

```text
forward micro-batch 7
backward micro-batch 3
forward micro-batch 8
backward micro-batch 4
...
```

这看起来像“乱序”，但它是流水线调度为了填满设备而做的有序交错，不是任务随便打乱。

更重要的是，PP/TP/DP/CP/EP 这些“切片标签”不是时间序列，而是每个 runtime event 的 placement metadata。例如：

| 时间线上看到的 event | 这个 event 应挂的切片标签 |
|---|---|
| `QKV Linear forward` | `PP5 / TP3 / DP2 / CP0` |
| `MoE all-to-all dispatch` | `PP5 / EP19 / DP2 / CP0` |
| `Expert FFN forward` | `PP5 / TP3 / EP19` |
| `gradient all-reduce` | `DP group` |
| `P2P send activation` | `PP5 -> PP6` |

所以产品上不要把切片标签画成“这张卡按时间依次跑 PP、TP、DP、CP、EP”。更准确的表达是：时间轴上有 forward、backward、通信、更新等事件；每个事件都带有一个或多个并行切片标签。

这类内容可以直接借用 `/Users/yin/pto/Profiling_Insight_and_Tool` 里的 swimlane 表达方式，而不是再画一张静态卡片图。推荐图表结构：

```text
横轴：同一个 training step 内的相对时间
纵轴：rank lanes，例如 rank_296、rank_297、rank_298、rank_299
条形任务：recv activation、forward block、MoE all-to-all、send activation、backward、DP all-reduce
行标签：rank/device + PP/TP/DP/CP/EP placement metadata
```

示例读法：

| swimlane 元素 | 在本报告里的含义 |
|---|---|
| `rank_299 · 910B_299` 这一行 | 一张卡 / 一个 rank 的 runtime 事件序列。 |
| `PP5 / TP3 / DP2 / CP0 / EP19` 行标签 | 这个 rank 的静态 placement 坐标，不是时间顺序。 |
| `Fwd mb7 · Blocks 20-23` 条形任务 | 当前时刻正在执行 PP5 覆盖的 decoder blocks。 |
| `MoE all-to-all + expert bucket` 条形任务 | EP（Expert Parallel，专家并行）相关的 token dispatch / combine 与专家计算。 |
| `DP all-reduce` 条形任务 | backward 后的梯度同步；同步的是梯度 / 参数更新，不是把样本数据拼回去。 |
| 多个 rank 行错开 | Pipeline schedule、通信等待和计算依赖造成的有序交错，不是随机乱序。 |

实现上应复用 PTO 已沉淀的 `swimlane-task` pattern：

- 共享 renderer：`/Users/yin/pto/vendor/pto-design-system/patterns/swimlane-task/pattern.js`
- 关键 API：`PtoSwimlaneTaskPattern.drawTaskBar`
- 原产品来源：`/Users/yin/pto/Profiling_Insight_and_Tool/AI_Profiling_Tool/js/graph-evidence/swimlane-stage.js`
- 数据来源契约：真实产品应从 `trace_view.json` parser 生成 lane model；白皮书里的图是说明型样例，不应反过来当真实 profiling 数据。

这能补足前面 placement 图的不足：placement 图说明“rank 属于哪些并行坐标”，swimlane 图说明“这些坐标下，rank 在时间轴上实际跑了哪些 runtime events”。

可靠依据：

- PyTorch Pipeline Parallelism 文档把 `PipelineStage` 定义为模型的一段，并提供 `Schedule1F1B` 等微批调度；这说明 pipeline stage 上的 forward/backward 可以按 micro-batch 交错执行。
- PyTorch Profiler 文档把 profiler 结果作为 operator/runtime event timeline 来观察；这说明时间轴记录的是实际执行事件，而不是静态 placement 标签。
- PyTorch 训练循环示例说明一个训练 step 包含 forward、loss、backward、optimizer update；forward 只是 step 的一部分。
- PTO / Profiling_Insight_and_Tool 的 graph-evidence swimlane 已采用 `swimlane-task` renderer，把 Step / Stream / Communication / Overlap / Coverage lane 作为 runtime evidence 表达；白皮书可借用同一视觉与语义，但必须说明它是示意图。

#### 10.5.5 block、layer、隐藏层、hidden state 的关系

这里需要纠正一个中文术语歧义：**“隐藏层”不是一个稳定的单一概念**。

| 词 | 更准确的含义 | 在本报告里的建议用法 |
|---|---|---|
| `block` / `Transformer block` / `decoder block` | Transformer 里重复堆叠的完整计算单元，通常包含 Attention、FFN/MoE、Norm、Residual 等子结构。 | 描述 PP（Pipeline Parallel，流水线并行）切分层段时优先使用，例如 `Blocks 20-23`。 |
| `layer` | 语境依赖很强。它可以指一个完整 Transformer block，也可以指更小的 `Linear layer`、`Attention layer`、`LayerNorm layer`。 | 不单独裸写“layer”，必须说明是 `decoder layer`、`linear layer` 还是 `attention sublayer`。 |
| `num_hidden_layers` | Hugging Face 配置字段，官方文档说明它是模型里的 block 数量。 | 解释配置时可以说 `num_hidden_layers=61` 表示 61 个 decoder/Transformer blocks。 |
| `hidden layer` / 隐藏层 | 中文里容易混用。传统神经网络里可泛指输入层和输出层之间的中间层；在大模型讨论中，经常被误用来指 `hidden state` 或 `num_hidden_layers`。 | 避免直接用“隐藏层”做产品标签。需要时写成“隐藏状态 hidden state”或“隐藏层数量字段 num_hidden_layers”。 |
| `hidden state` | 每个 token 在模型内部的中间向量表示，形状通常是 `[batch_size, seq_len, hidden_size]`。它是 runtime tensor，不是一个模块。 | 描述激活、attention 输入输出、层间传递数据时使用。 |
| `hidden_size` | hidden state 每个 token 向量的宽度，也就是最后一维大小。 | 解释张量形状或模型宽度时使用，例如 Pangu 配置中 `hidden_size=7680`。 |

因此，下面两句话分别在不同语境下成立：

```text
配置语境：
num_hidden_layers=61 表示模型有 61 个 Transformer/decoder blocks。

运行时语境：
hidden state 是每个 token 在某一层输出的中间表示张量，
例如 [batch_size, seq_len, hidden_size]。
```

这也解释了为什么 `Blocks 20-23` 不能翻译成“隐藏状态 20-23”。`Blocks 20-23` 指 PP5 这个 stage 覆盖的第 20 到第 23 个 Transformer/decoder block；这些 block 会接收上一层输出的 hidden state，执行 Attention 和 FFN/MoE，再输出新的 hidden state 给下一层。

官方依据：

- Hugging Face Transformers 配置文档说明 `hidden_size`、`num_attention_heads`、`num_hidden_layers` 是通用配置字段，并明确 `num_hidden_layers` 是模型里的 block 数量。
- Hugging Face Transformers Model Outputs 文档说明 `hidden_states` 是 embedding 输出加每一层输出组成的张量元组，形状为 `(batch_size, sequence_length, hidden_size)`。
- openPangu-Ultra-MoE-718B `config.json` 给出 `num_hidden_layers=61` 与 `hidden_size=7680`，这两个字段一个描述 block 数量，一个描述 hidden state 向量宽度。

产品文案建议：

- 用 `Blocks 20-23` 或 `Decoder blocks 20-23` 表达 PP stage 覆盖范围。
- 用 `hidden state [batch, seq, hidden]` 表达卡上正在传递或保存的中间激活。
- 不要把“隐藏层”同时拿来表示 block 数量和 hidden state，否则会把模型结构和运行时 tensor 混在一起。

#### 10.5.6 对产品图表的落地要求

TrainScope 页面建议把“整网模型架构”和“单卡任务”做成可追溯的父子关系：

```text
整网架构节点
Parallel Embedding / Dense Layer / MLA Attention / Router / Experts / LM Head
        |
        | placement projection
        v
rank/device 任务
stageLayers / tensorShards / sequenceShard / expertBucket / batchShard / commGroups
        |
        | runtime telemetry
        v
step 级信号
loss / grad / router stats / expert load / communication bytes / overflow
```

产品上要避免两类误导：

- 不要把模型架构图当成训练步骤全貌。架构图只覆盖 forward 主干，训练还包括 loss、backward、通信和 optimizer update。
- 不要把单卡任务画成一张卡拥有完整的模型小副本。混合并行下，单卡通常只拥有某个 stage、某些 tensor shard、某些 expert bucket、某段数据或序列。

可靠代码与文档来源：

- Pangu 模型配置：`https://huggingface.co/FreedomIntelligence/openPangu-Ultra-MoE-718B/blob/main/config.json`
- Pangu 推理代码：`https://huggingface.co/FreedomIntelligence/openPangu-Ultra-MoE-718B/blob/main/inference/model.py`
- Pangu 模型卡：`https://huggingface.co/FreedomIntelligence/openPangu-Ultra-MoE-718B`
- V1.1 规范架构事实层：`/Users/yin/gitcode/openPangu-Ultra-MoE-718B-V1.1/outputs/model_architecture.json`
- V1.1 架构校验报告：`/Users/yin/gitcode/openPangu-Ultra-MoE-718B-V1.1/outputs/model_architecture_validation.md`
- PyTorch 训练循环示例：`https://docs.pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html`
- MindSpore Transformers 分布式并行训练：`https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/parallel_training.html`
- MindSpore Transformers 配置文件说明：`https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/configuration.html`
- 本地整网架构图数据：`/Users/yin/pto/pangu-moe-trainviz/data/graph-ultramoe-718b.js`
- ParallelDemo placement 与卡内任务逻辑：`/Users/yin/pto/Profiling_Insight_and_Tool/ParallelDemo/dist.html`

---

## 11. 对 `TrainScope` 的产品表达建议

### 11.1 不推荐的表达

- `每张卡承载哪些层/算子`

这个说法会让人误以为所有并行都像 PP 一样按层切。

### 11.2 推荐的总标题

- `每个 rank/device 的并行放置`
- `Runtime Placement`
- `Rank / Device Placement`

### 11.3 推荐的字段

- `device`: host / slot / device_id
- `rank`: global_rank / local_rank
- `PP`: 负责的层段范围
- `TP`: 持有的算子/权重分片
- `CP/SP`: 负责的序列分片
- `DP`: 所属副本组
- `EP`: 持有的 expert / expert shard

### 11.4 推荐的单卡详情结构

单卡详情页建议不要只显示“层列表”，而是显示下面几组字段：

```text
物理位置
hostId / nodeId / deviceId / cardType

运行时身份
globalRank / localRank / workerId

并行坐标
dpRank / ppRank / tpRank / cpRank / epRank

通信组
dpGroup / ppGroup / tpGroup / cpGroup / expertGroup

盘古 MoE 卡内计算
stageLayers / tensorShards / sequenceShard / batchShard / expertIds / routeStats
```

这套结构可以同时回答两个问题：

- 它运行在哪张卡上
- 它在本 step 里具体承担哪些并行切片与通信关系

### 11.5 推荐的一句话说明

可以写成：

> 该视图不是简单回答“这张卡是哪几层”，而是展示每个 rank/device 在 PP、TP、DP、CP、EP 多个并行维度上的运行时放置关系。

---

## 12. 给页面文案直接可用的几种模板

### 12.1 适合 PP 的文案

- `PP5 · layers 36-42`
- `该 stage 执行第 36-42 层的前后向`

### 12.2 适合 TP 的文案

- `TP2 · Attention QKV shard 2/8`
- `该 rank 持有该层线性算子的张量分片`

### 12.3 适合 DP 的文案

- `DP7 replica`
- `完整模型副本之一，处理当前 step 的数据 shard`

### 12.4 适合 CP/SP 的文案

- `CP1 · sequence shard 1/4`
- `负责当前上下文窗口的一段 token`

### 12.5 适合 EP 的文案

- `EP19 · experts 152-159`
- `该 rank 持有当前 MoE 层的一部分 experts`

---

## 13. 最终结论

### 13.1 如果只问一句：“每张卡承载哪些层/算子”这个说法准吗？

回答是：

- **对 PP 来说，基本准确**
- **对 TP 来说，只说对了一半**
- **对 DP/FSDP/CP 来说，基本不准确**
- **对 EP 来说，应该改成“承载哪些专家/专家分片”**

### 13.2 更准确的一句话

> 真实大模型训练里，不是简单地“卡按层和算子分配”；更准确的是：每个 rank/device 在 PP、TP、DP、CP、EP 等不同并行维度上，同时拥有层段、参数分片、专家分片、序列分片和通信组身份。

---

## 14. 官方与可靠代码依据

以下链接是这份说明直接参考的官方文档：

- openPangu-Ultra-MoE-718B config: https://huggingface.co/FreedomIntelligence/openPangu-Ultra-MoE-718B/blob/main/config.json

- openPangu-Ultra-MoE-718B inference/model.py: https://huggingface.co/FreedomIntelligence/openPangu-Ultra-MoE-718B/blob/main/inference/model.py

- openPangu-Ultra-MoE-718B model card: https://huggingface.co/FreedomIntelligence/openPangu-Ultra-MoE-718B

- Hugging Face Transformers Configuration: https://huggingface.co/docs/transformers/main_classes/configuration

- Hugging Face Transformers Model Outputs: https://huggingface.co/docs/transformers/main_classes/output

- PyTorch training loop example: https://docs.pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html

- PyTorch Profiler Recipe: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

- PyTorch DistributedDataParallel: https://docs.pytorch.org/docs/2.12/generated/torch.nn.parallel.DistributedDataParallel.html

- PyTorch FullyShardedDataParallel: https://docs.pytorch.org/docs/2.12/fsdp.html

- PyTorch Tensor Parallelism: https://docs.pytorch.org/docs/2.12/distributed.tensor.parallel.html

- PyTorch Pipeline Parallelism: https://docs.pytorch.org/docs/2.12/distributed.pipelining.html

- Megatron Core MoE: https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/moe.html

- DeepSpeed MoE Tutorial: https://www.deepspeed.ai/tutorials/mixture-of-experts/

- MindSpore Transformers 分布式并行训练: https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/parallel_training.html

- MindSpore Transformers 配置文件说明: https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/configuration.html

- MindSpore communication API: https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.communication.html
