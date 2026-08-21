# Config & Relation Observer · 配置兼容规则升级计划

对 `config-relation-observer` 三个模块规则层的一次审视结论与落地清单。

审视范围：

- [js/config-relation-observer.js](js/config-relation-observer.js) — `validate()` / `reconcile()` / `derive()`，整页唯一数据源
- [js/config-relation-capacity.js](js/config-relation-capacity.js) — 单卡容量的显存口径
- [js/config-relation-yaml.js](js/config-relation-yaml.js) — MindFormers yaml + msrun 启动命令的落盘口径

**总结论**：现有 5 条校验规则本身没有写错的，问题是「规则集太薄 + 三个模块各拿一套隐含假设」。
最大的一处偏差是 EP 被硬编码成正交维（`world = DP×PP×TP×CP×EP`），而主流实现里 EP 是从 DP 组内切出来的、不独占 rank。

---

## 落地清单

按建议顺序排，分四批。批次只影响做的先后，行内互相独立。

- **第一批 = 行 1–3**　核心口径（EP 正交性）
- **第二批 = 行 4–8**　结构硬约束
- **第三批 = 行 9–12**　跨模块口径对齐
- **第四批 = 行 13–14**　自动配平手感

改完一行就把首列的 `[ ]` 勾成 `[x]`。

| ✔ | # · 改什么 | 修正了什么业务规则 | 改动后页面变化 |
|---|---|---|---|
| [x] | **1** 加 `moeOrthogonal` 开关（默认 false=切出），`validate` 的 world 公式按开关取 `DP×PP×TP×CP×EP` 或 `DP×PP×TP×CP` | EP 是否独占 rank。主流实现（Megatron/MindSpeed/MindFormers）EP 不进 world，而是从 DP 组内再切一刀 | MoE 区多一枚二选一切换；切到「切出」后同样 2048 卡的配置里 DP 显示从 8 变 512，Total Rank 不变；`#croConfigError` 那行公式文案跟着换 |
| [ ] | **2** `validate` 增 `DP % EP == 0`（切出档生效），`reconcile` 里 ep 的合法值判定同步加这条 | EP 必须整除 DP —— 专家组要能在 DP 组内均分，否则某些 DP 副本拿不到完整专家集 | 把 DP 调到不被 EP 整除时当场报错并标红 DP/EP 两个 stepper；正交档下这条不出现 |
| [x] | **3** 集群矩阵 d 轴与 `coordsOfRank` 的 `dpIdx` 正名为 EDP（`DP/EP`），tooltip / 关系连线文案同步 | 「DP=8」到底是 attention DP 还是 expert DP。参考配置里那个 8 是 EDP，真 DP 是 512 | 矩阵几何**完全不动**（rank 编址本来就对），只有轴标签、格子 tooltip、事件详情里的角色卡文字从「DP 0–7」变成「EDP 0–7」 |
| [ ] | **4** `validate` 增 `heads % TP == 0`；GQA 模型再加 `kvHeads % TP == 0` | TP 切的是注意力头，切不整就非法。Qwen2 只有 4 个 KV head，TP>4 需复制 KV | TP 从 1 往上调时，48 头模型在 TP=64 处被拦、Qwen2 在 TP=8 处被拦，报错说明写清是哪个头数除不尽 |
| [ ] | **5** `validate` 增 `seqLen % (2×CP) == 0`（或至少 `% CP`） | CP 沿序列维切分，ring attention 还要能对半做负载均衡 | CP>1 时 Seq Length 的非法取值被拦；单卡容量栏里 `seq/cp` 不再出现小数序列长度 |
| [ ] | **6** `ranksPerNode` 从 `CARD_SPECS` 取合法枚举（910B=8），`node` 改为受约束派生而非自由 stepper | 每节点卡数是硬件事实，不是任意整除 | Node stepper 只能落在合法值上（2048 卡 → 256 节点），填不出「1 节点 2048 卡」这种；卡型号一换 Node 跟着重算 |
| [ ] | **7** 增 `TP > ranksPerNode` 的**软警告**（不拦截） | TP 每层前反向都要 all-reduce，跨节点是性能悬崖而非功能错误 | `#croConfigError` 下方多一条黄色提示行（现在只有红色错误一档，需要加个 warning 级别样式） |
| [ ] | **8** `validate` 增 `denseIntermediate % TP`、`moeIntermediate % TP` | FFN 沿 intermediate 维切分，moeIntermediate 只有 1024，TP 大了会切碎 | 大 TP + MoE 组合被拦；提示指向 MoE 区而不是 Model Architecture 区 |
| [ ] | **9** `recompute` / `use_seq_parallel` 从 yaml 硬编码提成 config 字段，capacity 的 `actPerLayer` 按这两个开关取值（全重计算 ≈ `2·mb·s·h`／层） | 激活显存口径。现在 capacity 假设「开 SP、不重计算」，yaml 写的是「关 SP、全重计算」，两条都反了 | 单卡容量的激活段体积明显变化（重计算档大幅缩短）；YAML 视图里这两行从死值变成跟随配置；口径浮层的假设行随之改写 |
| [ ] | **10** capacity 的 optim 段按 EDP 切分（`enable_parallel_optimizer` 为真时），或把该开关也提成 config 字段 | 优化器状态是否按数据并行组分片。MindSpore 的 `enable_parallel_optimizer` 就是 ZeRO-1 | 容量柱的「优化器状态」段大幅缩短；判定文案里「DP 不除任何东西」那句要改成有条件成立，这是这一栏最常被引用的一句，改动最扎眼 |
| [ ] | **11** capacity 的 emb/head **不再** `/tp`（`vocab_emb_dp: True` 时） | Embedding 走 DP 不走 TP，每卡背满 151552×2560 ≈ 388M 参数 ≈ 6.2 GB（含梯度+优化器） | TP=1 时数字不变（除以 1），TP>1 时 stage0 / 末 stage 两根柱子明显抬高 —— 「各 PP Stage 峰值」那排小柱的首尾不均会更突出 |
| [ ] | **12** 确认 `full_batch: True` 下 `batch_size` 是全局还是每 DP，据此决定 `BASIS.microBatch` 要不要再除 DP | micro-batch 口径。全局语义下现在的容量估算按 DP 倍数高估激活 | 若结论是全局：激活段随 DP 变化（当前完全不变），会推翻「调 DP 容量柱纹丝不动」这个当前刻意强调的现象 —— **需先确认再动，不建议猜** |
| [ ] | **13** `reconcile` 里 anchor=`node` 那条分支（`node > world` 时撑 world）改为夹取到合法节点数 | 用户改节点数不该反向重排并行度 | 拖 Node 时并行度 stepper 不再被动跳变，Node 自己被夹到最近合法值 |
| [ ] | **14** `fitParallelWorld` 候选序从 `dp→ep→tp→cp→pp` 改为 `dp→tp→cp→ep→pp` | EP 是牵连最广的一维（同时动 MoE 分组、矩阵列数、容量专家段），不该在 DP 之后第一个被牺牲 | 拖 Total Rank 时优先动 DP/TP，MoE 区与集群矩阵的列数保持稳定，画面跳动明显减少 |

### 行 1 落地记录（2026-08-21）

开关是 MoE 区标题右侧的 `#croEpMode`，默认「EP 切出」。实现上多出计划里没写的两处，都是行 1 自身的正确性所必需：

- `derive()` 新增 `edp = 正交 ? DP : DP/EP`，集群矩阵的 d 轴与两处按 DP 副本遍历的查询改读它。不这么做，切出档下 `ranksPerStage` 会比 world 大一个 EP 倍，矩阵直接画错。**几何和标签仍与改动前逐位相同**，行 3 只剩改名。
- `FIELD_SPECS.dp.max` 1024 → 8192：切出档的 DP 大一个 EP 倍，旧上界会在换算时夹掉值、连带改动 Total Rank。

**遗留的口子（行 2 关掉）**：把 DP 调到小于 EP（如 DP32 / EP64）时，world 公式仍然自洽、不报错，但 `edp` 退化到 1，矩阵会按 1 行 × EP 列画出比 Total Rank 更多的格子。补上 `DP % EP == 0` 的校验即消失。

### 行 3 落地记录（2026-08-21）

行 1 落地后页面出现了自相矛盾：表单里 DP 写着 512，矩阵左侧却标着 DP0–7。行 3 紧接着做完。

- 新增 `dAxisName(counts)` 一处判定：切出档且 EP>1 → `EDP`，否则 `DP`（稠密模型 EDP≡DP，不平添新词）。矩阵的行标签 / 组与块的 aria / 格子提示 / 格子 aria 全部走它。
- `coordsOfRank` 那行坐标文案在三处重复（关系卡片、计算血缘、事件详情），合并成 `coordLine(topology, co)` 一份，d 轴名字随口径变。
- Layer 导航的查询范围键（`#croDpScope`）文案改由 `syncDpScopeLabels()` 写：它选的正是「查一个模型副本还是全部」，副本的编号轴就是 d 轴。
- 静态事件样例里的 `PP3 / DP0 / EP23` 改成口径中立的 `PP3 / EP23` —— 它是写死的字符串，跟不了开关。

**表述成本按计划里预判的那样高于实现成本**，换算式落在三处：Cluster 区标题右侧常驻一行（`矩阵纵轴 = EDP 8（DP 512 ÷ EP 64）…`，放标题行是因为矩阵的纵向预算是量 `.cro-cluster__grid` 来的，多一行会直接从格子上扣）、每个格子的悬浮提示、以及容量栏口径浮层新增的 EDP 条目。

### 两点提醒

- **行 12 是唯一一条建议先查证再动的**：结论会推翻页面现在刻意讲的一个卖点（「加 DP 只增吞吐不增余量，容量柱纹丝不动」）。
- **行 3 的实现成本远低于它的表述成本**：代码只改标签，但「DP 8 变 512」这件事需要在容量栏口径浮层里给一句解释，否则老用户会以为算错了。

---

## 附：行 1 的推导（为什么说 EP 不该独占 rank）

现在页面把 EP 当正交维硬编码在两处：

- [js/config-relation-observer.js:10](js/config-relation-observer.js#L10) — `world = DP × PP × TP × CP × EP`，注释写「EP 不从 DP×TP 里切出来」
- [js/config-relation-yaml.js:109](js/config-relation-yaml.js#L109) — `expert_parallel: 64  # 即 EP，与 DP 正交`

但主流实现（Megatron / MindSpeed-LLM / MindFormers）都是切出式：world_size 只由 `DP × PP × TP × CP` 决定，EP **不进乘积**，而是把 DP 组再切一刀，约束是 `DP % EP == 0`（严格些是 `DP × TP % (EP × ETP) == 0`）。EP 不独占 rank —— 一张卡既是某个 DP 副本的成员，又是某个 EP rank 的持有者。

**关键点：参考配置在切出口径下其实是自洽的，只是 DP 被标错了。**

`dp=8, pp=4, tp=1, cp=1, ep=64 → 2048`，换成切出口径：

```
真 DP  = 2048 / (PP4 × TP1 × CP1) = 512
EDP    = 512 / EP64               = 8      ← 这就是配置里写的那个 "8"
```

两种口径算出来的 total rank 恰好都是 2048，所以一直没穿帮。

好消息是 `derive()` 里的 rank 编址
[js/config-relation-observer.js:342-346](js/config-relation-observer.js#L342-L346)
`r = s·(DP·EP·TP·CP) + d·(EP·TP·CP) + p·(TP·CP) + …`
在切出口径下**几何完全正确** —— 那 512 个 rank 本来就该按 `8 EDP × 64 EP` 排成网格。集群矩阵一个格子都不用改，错的只是命名和校验公式。

开关的联动面：

| | 正交（现状） | 切出（建议默认） |
|---|---|---|
| world | `DP×PP×TP×CP×EP` | `DP×PP×TP×CP` |
| 新增校验 | 无 | `DP % EP == 0` |
| 矩阵 d 轴 | DP | EDP（`DP/EP`） |
| yaml 注释 | 「与 DP 正交」 | 「从 DP 内切出，EDP=DP/EP」 |

[js/config-relation-capacity.js:574](js/config-relation-capacity.js#L574) 那句「EP 与 DP 是否正交的口径差异均未计入」说明这个洞已被意识到，只是没落成规则。

> ⚠ 未核实项：本仓内没有 MindFormers 具体版本的校验式证据，上述按 Megatron/MindSpeed 通行做法推得。
> 落地前建议拿实际使用的 MindFormers 版本核一下 `parallel_config` 的 device_num 校验。

---

## 附：行 9–12 的跨模块口径打架明细

同一份配置，capacity 和 yaml 讲的是两套故事：

| capacity 的假设 | yaml 实际写出去的 | 后果 |
|---|---|---|
| 「DP **不除任何东西**，除非上 ZeRO/FSDP」（[capacity.js:553](js/config-relation-capacity.js#L553)） | `enable_parallel_optimizer: True`（[yaml.js:102](js/config-relation-yaml.js#L102)） | MindSpore 这个开关就是 ZeRO-1，优化器态按 DP 切。默认配置下 optim 段（占 12/16 的权重相关字节）被**整体高估** |
| `actPerLayer: 34`，注明「**不重计算**」（[capacity.js:60](js/config-relation-capacity.js#L60)） | `recompute: True`（[yaml.js:116](js/config-relation-yaml.js#L116)） | 全重计算下激活段应掉到 ~`2·mb·s·h`／层，现在高估近一个量级 |
| 同上，注明「**开 SP**」 | `use_seq_parallel: False`（[yaml.js:112](js/config-relation-yaml.js#L112)） | 方向相反，这次是**低估** |
| Embedding/Head 按 `/tp` 切（[capacity.js:420-422](js/config-relation-capacity.js#L420-L422)） | `vocab_emb_dp: True`（[yaml.js:113](js/config-relation-yaml.js#L113)） | emb 走 DP 不切 TP，每卡都要背满 151552×2560 ≈ 388M 参数 ≈ 6.2 GB（含梯度+优化器）。TP=1 时看不出来，一提 TP 就错 |

另外 [js/config-relation-yaml.js:88](js/config-relation-yaml.js#L88) 把 `runner_config.batch_size` 注释成「每 DP 每步喂进去的样本数」，但同时写了 `full_batch: True` —— MindSpore 半自动并行下 full_batch 语义是喂**全局** batch 由框架切分。这一条直接决定 `BASIS.microBatch` 该不该再除以 DP（即行 12）。

**方向建议**：把 `recompute / seq_parallel / parallel_optimizer / vocab_emb_dp` 这四个开关从 yaml 的硬编码里提出来变成 config 字段（哪怕先只在 YAML 视图里可读不可调），让 capacity 读同一份来源。现在这四个是显存估算里影响最大的旋钮，却是唯一 capacity 看不见的。
