# DeepSeek V4 Pro「结构与执行时序」规格

更新：2026-09-09。本文为 training-structure-time-v3.html 的 demo 规格。**DeepSeek V4 是主 demo，ST 仅为参考证据**。只为一个代表层 L2 生成典型算子，其余阶段用概览块表达完整 step。轻量 Demo 限制的是事件实例数，不是真实工具所需的数据契约；参数组、优化器状态分片、集合通信组、运行时结构映射和字段来源均保留，未有实测值的内容显式标记为模拟或 null。

## 1. 产品入口与问题顺序

“执行时序”先回答：选定 step、选定部署范围内，所有 MB 如何在各 PP / Rank 上交错执行？再回答某个 MB 的传播与模型角色的具体执行。

默认：**DeepSeek V4 Pro · Step 18420 · 全部 DP · 完整 step · 每 DP 全部 8 MB**，没有模型或事件预选。

- “完整 step”是当前部署查询范围；默认全部 DP，可手动筛选 DP0 / DP1。
- “全部 DP”可查看当前模拟的 16 Ranks。
- MB 是跨资源泳道的焦点字段，不是资源树或模型树的父节点。
- 单 MB 追踪仍重要，但不能代替完整 step 的调度概览。
- 默认层级：DP → PP → Rank；Rank 的类别与 L2 示例按需展开。
- 首屏用数据生产端提供的 F/B 调度摘要。只有 L2 有算子示例，其他层合并为概览块；“完整 step”不等于“全层算子覆盖”。

## 2. 架构：契约、生产端、消费端

~~~text
官方 Pro 配置摘录 + ST 原始记录 + 明确的模拟训练配置
                          ↓
          独立离线生成器 + 因果 / 数据校验
                          ↓
             training-timeline.v2 JSON
                          ↓
          前端加载并校验 → 泳道 / 结构联动
~~~

本项目是静态 demo，不需要常驻后端服务。“后端”落为可重复运行的离线生产脚本，前端只消费经校验的 JSON。生产契约按真实分析工具设计，Mock 只减少事件数和参数组样本，不删除真实接入时需要的关联字段。

### 2.1 文件职责

| 文件 | 职责 |
| --- | --- |
| training-timeline.schema.json | 前后端共同遵守的版本化 JSON Schema |
| timeline-contract.js | 同一套结构、引用、时间与因果校验；Node 与浏览器共用 |
| data/deepseek-v4-pro.config.json | 官方模型配置所需字段的摘录与来源 |
| data/training-simulation.config.json | 全部训练部署 / batch / 时间尺度假设 |
| st-step23.js / ST-LICENSE.txt | 上游 ST 选定 step 的原始记录、源索引、hash、许可 |
| generate-training-data.cjs | 提取参考统计、构造调度与模型角色事件、输出 JSON |
| data/deepseek-v4-pro.timeline.json | 已生成的数据；页面只读，不运行生成器 |
| training-structure-time-v3.js | 加载 / 校验 JSON，构造可见泳道、选择、并集及渲染 |
| generate-training-data.test.cjs | 生产端、因果、覆盖和确定性回归 |
| training-structure-time-v3.test.cjs | 直接加载实际前端控制器的消费与交互回归 |
| model-profile-validation.md | Pro / Flash、主层 / MTP、模型事实 / trace 证据边界 |

禁止在前端重新计算训练开始时间、通信耗时、反向时间、MB 数对应的模拟公式或猜测 Rank 拓扑。前端可计算可见范围、查询集合与区间统计；这些是显示与分析，不是生产训练事件。

数据加载失败或 schema 不合法时显示错误，不回退到隐藏的旧 mock。

## 3. 版本一致性与模型事实

采用 [DeepSeek-V4-Pro 官方配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/config.json)，不混用 Flash。

| 模型事实 | 当前值 |
| --- | --- |
| 主层 | 61，L0–L60 |
| Hidden size | 7168 |
| Routed / Shared Experts | 384 / 1 |
| 每 token 路由专家 | 6 |
| Hash Router 层 | 前 3 层 |
| 主层注意力 | L0/L1 HCA；L2 起偶数 CSA、奇数 HCA |
| MTP | 额外 1 个聚合训练块，SWA |

重要：compress_ratios 有 62 项。索引 60 仍是 CSA；末尾索引 61 的 0 对应 MTP 的 SWA，不能误称最后一个主层为 SWA。

模型来源是架构配置，**不是官方训练部署或性能参数**。本模拟不从 checkpoint 的推理量化配置推断训练精度、显存或硬件可行性。

左图保留 Pro 整网角色模板，以 L2 展示重复 Decoder 的内部结构。仅 L2 事件有层内算子映射，其余层概览不冒充某个算子。trainingProfile=pro 模式不注入 Flash decode CSA 细节与来源声明；CSA/HCA 保持抽象角色级，MTP / Training Loss 作为聚合训练上下文。训练运行时是独立 auxiliary root，由 timeline JSON 注入，不写入模型源架构。原 Flash 页默认行为不改。

## 4. 真实训练数据如何参与 mock

来源：[ByteDance-Seed / StragglerAnalysis](https://github.com/ByteDance-Seed/StragglerAnalysis)。只用原 timeline，不用 ideal-timeline。

### 4.1 已核对事实

| 记录 | 并行配置 | 采样 step | 每个 DP 每 step 的可信 MB |
| --- | --- | --- | --- |
| ST | DP2 / PP4 / TP1 / DSP1 / VPP1，world 8 | 32 个非连续 step，23–1048 | 4，mb_id 0–3 |
| SE | DP32 / PP2 / TP1 / DSP2 / VPP1，world 128 | 45 个非连续 step，16–838 | 8，mb_id 0–7 |

依据为 F/B 计算记录按 step、DP 去重 MB，不把跨 PP / F/B 的重复出现再计数。SE 仅为设计研究依据，不接入当前数据生产。

[ST 配置](https://github.com/ByteDance-Seed/StragglerAnalysis/blob/main/data/meta-ST.yaml) · [ST 原始 trace](https://github.com/ByteDance-Seed/StragglerAnalysis/blob/main/data/timeline-ST.json.gz) · [SE 配置](https://github.com/ByteDance-Seed/StragglerAnalysis/blob/main/data/meta-SE.yaml) · [SE 原始 trace](https://github.com/ByteDance-Seed/StragglerAnalysis/blob/main/data/timeline-SE.json.gz)。

### 4.2 当前提取的参考量

ST step 23 有 212 条 X 记录，64 条 F/B 计算记录，可配成 32 对 (DP, Rank, MB) 相同的 forward / backward。

生成器保留每对 sourceIndex、forwardUs、backwardUs，计算：

~~~text
backwardForwardRatio = median(backwardUs / forwardUs)
                     = 2.1284895795426912
~~~

这个统计只用作当前模拟的相对计算时长假设。**不迁移 ST 的绝对耗时、不迁移其 stage imbalance、不把 ST 的事件改名后当成 DeepSeek trace。**它不能证明 DeepSeek 各角色的实际 F/B 比值相同；统一套用是明确简化，可在后续替换为更多实测参考。

ST step 23 的 DP0 计算顺序证据：

| PP | 顺序 |
| --- | --- |
| PP0 | F0 F1 F2 F3 B0 B1 B2 B3 |
| PP1 | F0 F1 F2 B0 F3 B1 B2 B3 |
| PP2 | F0 F1 B0 F2 B1 F3 B2 B3 |
| PP3 | F0 B0 F1 B1 F2 B2 F3 B3 |

上述序列用于说明 1F1B，不复制 ST 的 4 MB 数到 Pro。

ST 的 mb_id 只对 F/B compute 有效；8 条零时长 gc 即便带 mb_id=0，也不视为 MB0 的计算。原 ts / dur 单位 μs，保留原始值；配对比值无量纲。原 gzip SHA-256：

~~~text
17ebc492623ec67fefc4c56152732b64f24020535452078309616a2aea57686c
~~~

原记录文件及 Apache-2.0 许可保留，页面运行不依赖 /tmp 或 .local-*。

### 4.3 Optimizer 参考证据与契约边界

ST step 23 不只有 F/B，还为 8 个记录进程各提供一条 `grads-reduce-scatter`、`layernorm-grads-all-reduce`、`optimizer-clip-main-grad`、`optimizer` 和 `params-all-gather`。原记录可证明这些运行时阶段、所在 `pid/tid`、时间区间及 `step/model_chunk/mb_id/seq_id`；不提供 parameter group ID、参数名、optimizer state shard ID 或 DeepSeek V4 的分片归属。

因此生产端保留两层真实性：

- `calibration.optimizerPhases` 保留上述 5 类实测参考的数量、sourceIndex、中位时长和原字段名；它是阶段形状证据，不是 DeepSeek V4 的时长或部署证明。
- `parameterGroups / optimizerStateGroups / collectiveGroups / runtimeGraph` 是面向真实工具的版本化契约。当前只生成每个 PP 的 Dense 与 Routed Expert 两类代表参数组；其分布、状态分片和耗时为 `simulated` 并指向 `simulation-config`。优化器算法未有证据，用 `algorithm:null` 保留，不默认猜 AdamW。

数据缺失必须表达为 null / unknown 或较低 fidelity，不得通过删字段伪装契约完整，也不得把 ST 的字段值改名后当作 DeepSeek V4 实测。

## 5. 模拟训练配置与 MB 数

| 配置 | 模拟值 | 性质 |
| --- | --- | --- |
| Global batch size | 32 样本 | 假设 |
| Micro batch size | 每 DP 每 MB 2 样本 | 假设 |
| Sequence length | 4096 | 上下文假设，不作为已验证 kernel 成本模型 |
| DP / PP / TP / EP | 2 / 4 / 1 / 2 | 教学部署假设 |
| World | 16 Ranks | 当前模拟的独立坐标约定 |
| MTP / Recompute | 1 聚合块 / 不重计算 | 简化假设 |
| Optimizer | Reduce-Scatter → Clip → Optimizer Step → Parameter AllGather | ST 阶段形状 + 模拟参数组 / 分片 / 耗时 |

固定样本 batch 下：

~~~text
M = global_batch_size / (DP × micro_batch_size)
  = 32 / (2 × 2)
  = 8 MB / DP
~~~

PP / TP / EP 不增加逻辑 MB 数。当前 DP 表示完整独立副本，EP 在副本内部拆分局部 MB 的样本 / token；每个 EP Rank 都有 dense 角色与 Shared Expert，Routed Experts 权重分片。

这不是所有框架通用的 DP×EP 公式。某些框架让 EP 复用 DP 轴；本案例明确采用自己的副本坐标契约，不能只把框架参数名称照抄进来。

Dense 梯度在同 stage 的 DP×EP 成员同步；Expert 梯度只在同 stage、固定 EP 坐标的 DP 成员同步。没有任何说明声称 16 台实际设备能容纳该模型。

PP 分配为 L0–14 / L15–29 / L30–44 / L45–60。数据驱动的 Rank 坐标表提供 dp / stage / tp / ep，前端不推导 rank 除法。

## 6. 生产端调度与因果关系

采用非 interleaved 1F1B：

1. 每个 PP 的 warmup 长度为 PP_size − stage − 1。
2. steady-state 交替执行新的 F 与已有 MB 的 B。
3. cooldown 排空剩余 B。
4. 本地计算顺序、上游激活完成、下游梯度完成与本 MB 前向完成共同构成依赖。
5. 通过依赖 DAG 确定可开始时间，不在前端按 MB 索引平移矩形。
6. 所有梯度完成 → Dense / Expert Gradient Reduce-Scatter → Global Norm / Clip → Optimizer Step → Parameter AllGather。

默认 8 MB 时：

~~~text
PP0: F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7
PP3: F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7
~~~

仅 L2 提供 HC Pre / RMSNorm、CSA、Attention HC Post、FFN Pre / Norm、Hash Router、EP Dispatch、Routed Experts、EP Combine、Shared Expert、Combine / FFN HC Post 等典型角色。B 用角色逆序表达梯度归属，不声称是真实反向 kernel 分解。Shared 与 Routed 都有示例。

L0–L1、L3–L14 及其他 PP 内的连续层合并为前向 / 反向概览块（layer=null、granularity=aggregate、layerRange 标注范围）。输出头 / MTP / Loss 合并为一个上下文块。未提供明细的层不显示空算子树，也不把 L2 的数据复制成其他层的精确映射。

当前每 Rank 的 compute 串行；不为了展示重叠而造出同一计算资源的并发。EP 同步事件保留所有参与者；PP 传输在生产者结束之后、消费者开始之前。Shared 的并行调度、细粒度反向 kernel 与专家 token 热点尚未模拟。

仅 L2 的激活驻留从 F 完成延续到 B 消费结束，保留 releaseEventIds；不是设备忙碌。没有根据空白时间虚构 bubble / wait 事件。

当前产物：**1,208 条演示事件 + 256 条 F/B Rank 摘要，JSON 约 1.44 MB**。新增的 28 条事件只表达代表性 optimizer lifecycle，不扩展全层算子 trace。替代之前 37,532 条 / 29 MB 的过量方案，仍保持约 97% 的事件缩减。毫秒仅用于视觉演示，不是性能预测。

## 7. Schema 与校验

~~~javascript
{
  schemaVersion: 'training-timeline.v2',
  id, label, step, fidelity, timeUnit: 'ms',
  model, training, stages, ranks, provenance, calibration, coverage,
  parameterGroups, optimizerStateGroups, collectiveGroups,
  runtimeGraph: { id, label, nodes, edges },
  stepBounds: { start, end },
  events: [{
    id, label, kind, start, end,
    rank, dp, stage, tp, ep, layer,
    scope, microbatchId,          // microbatchId: string | null
    graphNodeIds, templateNodeIds,
    runtimeNodeIds, parameterGroupIds, optimizerStateGroupIds,
    collectiveGroupId, shards, referenceEventCategory,
    participants, dependsOn, sourceEventIds,
    stream, relation, fidelity, provenanceIds
  }],
  summaries: [{
    // Same required event fields, plus:
    isSummary: true,
    sourceEventIds: [/* contributing leaf event IDs */]
  }]
}
~~~

源事件身份稳定；摘要不进入 events，不被再次累计。摘要区间与源 ID 由生产端给出，前端不能为了填满图而推测时间。

microbatchId:null 用于梯度同步 / Optimizer 等 step 级事件；未来真实数据中也可表示未知归属，不能自动贴到最后 MB。缺失字段与 null 不等价，缺失直接报错。

timeline-contract.js 是当前 schema 所用关键字的受限验证器，不宣称是完整通用 JSON Schema 实现；遇到未支持关键字直接失败。另验证：

- ID 唯一、placement 与 ranks 表一致；
- 时间为有限数、非负、end≥start；
- participants、sourceEventIds、provenanceIds 引用存在；
- parameter group、optimizer state group、collective group、runtime node 和 shard 引用存在，collective participants 与组成员一致；
- runtime edge 端点必须是可解析的模型节点或运行时节点；
- 依赖无环且 producer.end≤consumer.start；
- 所有源事件属于声明 stepBounds；
- 摘要源记录存在、MB 一致、时间被摘要包围；
- 数据集与事件 fidelity 一致。

回归只检查 demo 所需边界：仅一个代表层有算子、事件量低于 1,500、MB 数公式、1F1B 顺序、compute 不重叠、Optimizer 时机与产物确定性。不要求全模型明细覆盖。

## 8. 前端视图与交互

独立状态：部署范围、时间窗口、MB 焦点、结构选择、事件选择、左右展开状态、时间表达模式。

- 普通事件点击：查看详情，有映射的源事件定位左图；不自动切 MB / DP / 时间窗口。
- 显式“追踪 MBxx”：只改变 MB 焦点，保留其他 MB 和 null MB 为淡色背景。
- “全部 MB”：只清除 MB 焦点，保留结构选择。
- 清除结构 / 事件选择与 Escape：不清除 MB 焦点。
- 结构节点选择默认覆盖全部 MB；局部 PP 查询可定位至有对应事件的阶段。
- 左右展开互不改写选择、时间范围或对方层级。
- 详情显示来源、作用域、模拟性质，摘要 / 并集可查看源事件。
- 无可靠 MB 的 step 级事件没有“追踪此 MB”。
- 通信投影到所有可见参与 Rank，包括跨 DP / PP；源事件指标不重复计算投影条。
- 左侧的模型源结构与“训练运行时”独立分区；MTP 是模型上下文，Optimizer 是 step 级 runtime node，不冒充 Decoder 算子。
- 选中 runtime node 反向筛选其 `runtimeNodeIds` 事件；选中 Optimizer 事件高亮 runtime node，并在详情中显示参数组、通信组和状态分片。
- 数据来源 popover 显示 batch 公式、模拟配置、ST 配对比值、官方模型配置与 schema 链接。

界面默认摘要只用于可读性；结构选择时摘要作为上下文，前景高亮仍来自准确的源事件集合，避免整段摘要因含一个相关算子就被整体高亮。

## 9. 并集与统计口径

S = 当前部署范围（含已知通信参与关系）∩ 当前时间 / 阶段查询 ∩ MB 焦点 ∩ 结构映射。

- 跨度：max(end) − min(start)。
- 区间并集：S 的半开时间区间并集长度。
- Σ：S 的源事件时长之和。
- 记录 Ranks：S 中源记录 owner rank 的去重数；不等于设备采集覆盖或吞吐。

画布先划分 S 与背景，再分别按 Rank + 活动类别 + MB 合并，保留 sourceEventIds 和时间空隙。点击并集不修改 S。背景摘要不作为前景统计来源。

等待 / 驻留即便有区间，也不是设备 busy。区间并集不能叫 GPU 利用率。模拟毫秒不用于性能归因。

## 10. 视觉与共享系统

继续使用 ide-frame、workbench-shell、swimlane-task 与现有 Pro 模型图 renderer。两主工作区，无新增常驻 Inspector / 性能卡片。新增状态和来源说明复用已有控件与 popover，不增加视觉样式体系。

保持 40 / 60 初始分栏、420 / 680 最小宽度、主题同步、sticky 行头、缩放与滚动。常规行 22px、任务条 16px、Rank 概览 58px、三轨间距 18px、行头 236px、标尺 48px。桌面验收基线 1280×900。

行头采用无边框的纯文字，不绘制 tag 底板。主标题 12px、次级说明 11px；Rank 名称独占第一行，TP / EP 放到第二行。展开按钮和树枝连接线与 Rank 第一行标题居中对齐，不按三轨总高度居中。鼠标悬停行头说明 DP / PP / Rank / 活动类别含义；“泳道 / 对象”总表头解释层级。事件 / 并集复用共享 hover tooltip：前者解释独立事件与 F/B 摘要，后者解释同 Rank / 类别 / MB 的区间合并与非利用率口径。

左侧 L1 / L2 是结构展示层级，不是 Layer 编号。Pro L1 折叠 Hybrid Attention 与 MoE 子模块，L2 展开已有角色 / 路由 / 专家；整网收起 Attention / FFN。三个入口使用 Pro 自身折叠状态，不再只切换 Flash 的 compound IDs。没有 Pro 实现子节点的 CSA / HCA 保持抽象角色且不显示无效展开按钮。

默认展开 DP / PP，Rank 为调度概览；层对象入口仅展开 L2 典型算子。其他 PP 标明“未模拟层内算子”。验收重点是首屏仿真观感、调度可读性和结构 / MB 联动，不以数据量作为完成标准。

### 2026-09-09 可读性修订

- DP / PP 展开后只作为分组标题，不叠画所有子 Rank；折叠后可显示分轨摘要。
- Rank 始终显示 F/B 调度、通信与 step 收尾；激活驻留仅在展开后的独立类别展示，不在“其他活动”重复绘制。Shared Expert 属于计算，不按名称误放到其他活动。
- 所有可见事件按实际绘制宽度做无重叠分轨，行高随子轨数量增加；activation 保留各 MB 的真实模拟时间区间，以 `MBxx · L2` 标注，不裁短时间或合并不同 MB。命中区域不跨相邻子轨。
- 删除默认“完整 step · 全部 MB · 事件数 · 模拟”常驻摘要行；仅在存在选择 / MB 焦点时显示必要的选择操作。
- 普通 hover 仅一句说明，不使用标题、分隔线和键值表；详细元数据留给点击后的详情。
- info 使用现有 panel-shell 标题 / 正文布局，批次、覆盖、调度、参考用紧凑键值表；来源链接复用 `btn btn-ghost btn-sm`，不用浏览器默认蓝色下划线。
- 颜色核对：旧 tooltip 默认 `--surface-3`，light 值为 `#E6E6E6`，不是 token 未加载。简短提示通过共享 pattern 允许的 `--pto-swimlane-tooltip-bg` 改为 `--surface-1`；info 同用 `--surface-1`（light 为白色），不修改全局 token。阴影、圆角和文字继续来自设计系统。

### 2026-09-09 联动与控件修订

- Rank 折叠时显示“计算 / 通信 / 其他活动”三轨摘要；展开后 Rank 本身只作为标题，事件只出现在前向、反向、通信、优化器、激活驻留等分类行。因此通信不会在 Rank 摘要和分类行重复出现。
- Optimizer 的模拟持续时间只有 5 ms，在约 2310 ms 的完整 step 中自然宽度不足 2px；保持真实时间区间，但最低绘制宽度设为 5px，确保橙色事件可见和可点击。这个宽度不参与统计。
- 左侧 Pro 可见结构节点均有演示事件映射：输入 / Embedding、前置层概览、L2 的输入流、残差、Norm、Attention、MoE、输出流、后续层概览、Final Norm、LM Head、Main Logits、MTP 和 Loss。粗粒度节点映射到阶段概览，不能解释为该节点的真实 kernel trace。
- 结构选中后，Rank 摘要让位于匹配的源事件，避免摘要矩形覆盖高亮；选择状态栏给出匹配事件数。折叠 / 展开图结构仍只改变结构层级，不误触发事件筛选。
- dropdown 不再复用 `.btn`。原实现因此继承了按钮的 pill 感、hover 位移与阴影。现在使用 input token：`--input-height-md`、`--input-radius`、`--input-border`、`--input-bg` 和 `--type-body-sm`，保持 34px 高、8px 圆角、无按钮位移动效。

### 2026-09-09 事件关系与详情修订

- 详情面板不再把内部 `dv4/...` 节点 ID 列表当作“结构”。单条事件显示可读的模型位置；F / B 摘要说明所属 PP 阶段和覆盖明细数；并集说明合并口径。“源事件”改为“组成明细 / 合并事件”，避免被误解为依赖源。
- 直接上游由 `dependsOn` 得到，直接下游由反向依赖索引与 `releaseEventIds` 得到。摘要 / 并集事件先把内部依赖排除，只显示跨出其组成集合的直接关系。详情显示完整直接关系计数，画布为可读性最多连接当前可见的 8 个上游和 8 个下游。
- 选中事件后，上游用 accent 实线指向当前事件，下游用 success 实线由当前事件指出；连线固定从源事件右侧 OUT 的垂直中点离开，进入目标事件左侧 IN 的垂直中点，不因屏幕相对位置翻转语义。当前事件及直接上下游保持 100% 不透明度，其他事件使用设计系统 `--button-disabled-opacity` 降权，保持可见而不再接近消失；详情中的色点与连线同义。连线仅表示模拟数据的直接因果关系，不表示数据量或通信带宽。
- 左侧模型节点或 runtime node 选中仍是结构筛选，但未命中的 Rank 保留 F/B 调度摘要，不空白。右侧事件选中则取消结构筛选；模型活动通过 `graphNodeIds`、训练运行时活动优先通过 `runtimeNodeIds` 向左侧发送只读 lineage 高亮。单条、摘要和并集事件都支持该反向联动，且不得回流成泳道过滤。点击泳道空白处同步清除该事件与左侧 lineage 高亮。
- 面板内“聚焦 MB”使用设计系统 `.btn-solid`，作为单一主操作；其他 MB 保留为背景。点击泳道时间区的空白处只取消事件选中与依赖连线，不清除 DP / MB / 结构筛选或时间窗口。
- 详情不重复显示“数据性质：模拟事件”；页面级“模拟配置 / 数据依据”已经承担来源声明。
- 部署范围、时间窗口、Microbatch 焦点三个顶部选择在偏离默认值时使用 `--primary` 边框、`--tone-blue-strong` 底色和左侧指示条；默认与已筛选状态都有单行 hover 说明，并告知如何恢复默认。
- 通信和 Optimizer 保留原始时间统计，但在完整 step 缩放下使用 5px 最小绘制 / 命中宽度，避免毫秒级事件看似消失。该宽度不改变时长与统计口径。

## 11. 生成与验收

从模块目录执行：

~~~sh
rtk node 分布式泳道/generate-training-data.cjs --write
rtk node 分布式泳道/generate-training-data.test.cjs
rtk node 分布式泳道/training-structure-time-v3.test.cjs
~~~

修改模拟配置后重新生成 JSON；前端无需改代码。当前生成器有明确支持范围：TP1 / EP2、M≥PP、无 recompute、非 interleaved 1F1B；其他模式需扩展生产器，不得静默生成不支持的行为。

| ID | 验收项 |
| --- | --- |
| P01 | 默认 DeepSeek V4 Pro / 全部 DP / 完整 step / 每 DP 全部 8 MB |
| P02 | 官方模型字段与模拟训练假设分开；主层与 MTP 配置索引正确 |
| P03 | 前端无 mock 时间公式、无 ST 原始解析；直接读取 schema 与 JSON |
| P04 | 仅 L2 有典型算子；其他层 / 输出头为概览，源事件少于 1,500 |
| P05 | 1F1B 顺序、依赖无环、通信先于消费者、compute 不重叠 |
| P06 | MB 数由 batch 配置推导，改为 GBS16 后得到 4 MB，其他组件无需修改 |
| P07 | 同配置输出逐字确定；保存产物与生成器输出一致 |
| P08 | L2 结构选择覆盖当前范围全部 MB；不选中其他层概览；并集不吞入无关源事件 |
| P09 | 显式 MB 追踪与独立清除；背景保留、时间窗口不变 |
| P10 | 同步与 Optimizer 的 MB=null；跨 DP / PP 投影不漏参与者 |
| P11 | 错误 schema、悬空引用或未来依赖被拒绝，不回退旧 mock |
| P12 | 事件详情不暴露内部节点列表；直接上下游用实线及降亮背景可视化，空白点击只清除事件选中 |
| P13 | 三个主选择器明确区分默认 / 已筛选状态；通信与 Optimizer 在完整 step 下可见可点击 |
| P14 | 结构筛选不清空未命中 Rank；事件选中反向高亮模型 lineage；依赖线固定从源右侧 OUT 连入目标左侧 IN |
| P15 | v2 契约保留参数组、optimizer state shard、collective group、runtime graph 和 ST optimizer 阶段来源；所有引用可解析 |
| P16 | 模型源结构与训练运行时分区；Optimizer 事件双向联动 `Optimizer Step`，详情可解释参数组与分片 |
| V01 | 1280×900 浏览器首屏、结构图、主题、选择、缩放、来源 popover 正常 |

数据量约 1.44 MB（未压缩 JSON）；静态文件加载即可，不增加流式、索引或分块系统。

## 12. 非本次范围与修订

不包含官方 DeepSeek 训练参数发现、真实 DeepSeek profiler 接入、实际硬件容量证明、逐专家 token 路由与 kernel 仿真、HBM、recompute、TP>1 调度、官方 optimizer sharding 部署结论、性能预测或自动根因诊断。已包含的 optimizer state shard 是契约完整性与交互验证用的模拟实例，不表示 DeepSeek V4 官方部署。

准确入口：[训练结构与执行时序](http://127.0.0.1:8765/pangu-moe-trainviz/分布式泳道/training-structure-time-v3.html)。HTTP 服务从 PTO 根目录启动，不能依赖 file://、.local-* 或 /tmp。

2026-09-09：按用户最新范围撤销全层明细，改为 L2 典型算子 + 完整 step 阶段概览。保留 schema / 离线生成 / 前端消费，恢复 Pro 主案例；ST 默认入口与全层仿真方案均不再是本规格。

2026-09-09：纠正“轻量 Demo 可删减生产字段”的错误假设。升级 `training-timeline.v2`，保留 ST 的 optimizer lifecycle 证据形状，新增代表参数组、优化器状态分片、集合通信组与运行时辅助图；模拟值与实测参考保持 provenance 分离。
