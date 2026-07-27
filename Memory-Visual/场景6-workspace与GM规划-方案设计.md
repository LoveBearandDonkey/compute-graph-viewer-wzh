# 场景 6 方案设计：融合算子 / 大算子的 workspace 与 GM 规划

> 版本：v0.1（方案设计）
> 上游：[昇腾算子内存可视化工具-规划.md](昇腾算子内存可视化工具-规划.md) §3 场景 6
> 落地对象：[README.md](README.md) 描述的内存工作台原型（`Memory-Visual/index.html`）
> 数据等级：L2 / schema-generated / share-safe，exploration-only。文中所有形状、容量、
> workspace 数值均为为演示构造的占位值，不对应任何真实产品算子。

---

## 0. 一句话与三个数

场景 1/2/5 解决的是「片上放不下」，场景 6 解决的是「片外放得下但放得太多」。

工具对每个融合算子只需要说清三个数，其余全是它们的展开：

| 数 | 含义 | 谁能改 |
| --- | --- | --- |
| `current` | 当前 tiling 上报的 workspace 值 | 现状 |
| `packed` | 保持子计算顺序与形状不变、仅做地址复用后可达的值 | 改分配策略即可，**改动确定安全** |
| `lowerBound` | 该顺序与形状下的理论下界（最大同时存活字节） | 只能靠改结构（原地、留片上、换执行序）再降 |

场景 6 的成功判据「workspace 显著下降，且改动有据可依」因此变成两个可度量的量：
比值 `current / lowerBound`（目标 < 1.2），以及每条建议背后的 evidence 链
（哪两个 tensor、谁在哪个子计算释放、谁在哪个子计算才申请）。

**关键设计取向**：不报一个孤零零的「理论最小值」。只报下界会让人面对一个够不着的目标；
把差距拆成「策略浪费 `current − packed`」和「装箱碎片 `packed − lowerBound`」两段，
前一段是可以今天就落地的确定收益，后一段才需要动结构。

---

## 1. 场景边界

### 1.1 它和已实现的场景不是同一套坐标

| 维度 | 场景 1/2/3/5（已实现） | 场景 6 |
| --- | --- | --- |
| 空间 | UB / L1 / L0x / VRF / SRF（片上，核内） | GM（片外，chip scope） |
| 主体 | 一个 kernel 内的队列与 TBuf | 一个融合算子内的**多个子计算**及其中间张量 |
| 时间轴 | kernel tick（流水模拟出来的 cycle） | **子计算序号**（拓扑序），tick 只是它的细化 |
| 约束 | 硬容量，超了编译期报错 | 软预算，超了不报错，只是吃显存、挤别的算子 |
| 目标 | 不超限 + 提利用率 | 峰值最小化 |
| 并发 | 单核视角 | 多 block 并发访问同一片 GM，**复用安全性判据不同** |

所以场景 6 不是「给 GM 层级再加一条规则」，它需要中间格式里多一层实体（子计算），
多一个求解器（装箱与下界），以及一套不同于片上的安全护栏。

### 1.2 明确不做的

- 不做模型级、跨算子的显存复用（那是 GE / 框架内存池的职责，属规划文档 §4.1 的「框架侧内存快照」，列入 P2）。
- 不做搬移式（relocating）分配。允许把已写入的数据搬到别处能进一步压低峰值，
  但代价是额外 GM↔GM 拷贝，超出「内存规划」的范畴，只在结论里提示存在这条路。
- 不替代 GE 的融合决策。工具只回答「这个融合体的 workspace 应该是多少」，不回答「该不该融」。

---

## 2. 现状差距（代码级）

| 位置 | 现状 | 差距 |
| --- | --- | --- |
| `data/chip-specs.js:62` | GM 是一个普通 region，`capacity` 注释为「本算子预算而非物理 HBM」 | 预算属于算子/模型上下文，不属芯片规格，应从 chip 挪到 run |
| `data/runs.js:226` | `gmDeclarations()` 六个固定分配，只有 `role` 与 `unused` | 无 producer / consumers，无子计算归属 |
| `data/runs.js:719` | GM 分配的 intervals 被强制拉平成 `[0, totalTicks]` | **GM 张量目前没有生命周期**，复用分析无从谈起 |
| `js/view-lifetime.js:23` | `chip.regions.filter(r => r.scope === 'core')` | GM 根本进不了生命周期视图 |
| `js/diagnostics.js:36,235` | 仅 `GM_UNUSED`（全程无访问） | 没有下界、没有复用组、没有跨 block 安全判据 |
| `js/metrics.js:49` | 对 GM 也算 `peakLive`，但因 interval 被拉平，恒等于 reserved | 峰值曲线在 GM 上是一条直线，不携带信息 |

结论：视图与规则引擎的分层是干净的（都只认中间格式），所以场景 6 的改动可以集中在
**数据模型 + 一个新的求解模块 + 一个新页签**，不需要动已有三个视图的渲染逻辑。

---

## 3. 数据模型扩展

对规划文档 §6 中间格式做**向后兼容的增量**：新增字段全部可选，缺失时降级
（没有 `subgraphs` 就退化成单子计算，此时下界 = 所有 workspace 之和，仍然是正确结论）。

```jsonc
{
  "schema_version": "0.2",

  // 新增：融合体的子计算序列（拓扑序）
  "subgraphs": [
    { "id": "sg0", "name": "QKVProj",   "kind": "matmul",    "order": 0,
      "src": { "file": "mla_block.cpp", "lineStart": 88, "lineEnd": 121 } },
    { "id": "sg1", "name": "RoPE",      "kind": "elementwise", "order": 1 }
    // ...
  ],

  // 新增：workspace 预算与规划结果（三个数的载体）
  "workspace": {
    "budget":     33554432,          // 上游给这个算子的 GM 预算
    "current":    22548480,          // tiling 侧 GetWorkspaceSizes 实际上报值
    "reportedAt": { "file": "mla_block_tiling.cpp", "line": 64 },
    "policy":     "per-subgraph",    // per-subgraph | arena | pool
    "scope":      "shared"           // shared | per-block（per-block 时实际占用 ×blockDim）
  },

  // GM allocation 的增量字段
  "allocations": [
    {
      "id": "wsAttn", "region": "GM", "role": "workspace",
      "offset": 12582912, "size": 2097152, "align": 512,
      "producer": "sg2",                  // 谁写它
      "consumers": ["sg3"],               // 谁读它（最后一个决定死亡点）
      "liveSubgraphs": { "start": 2, "end": 3 },   // 子计算序区间，闭区间
      "blockScope": "shared",             // shared | per-block
      "aliasOf": null,                    // 原地/别名时指向宿主，别名不参与独立装箱
      "mayStayOnChip": false,             // 是否小到能留在片上（见 FUSION_SPILL 规则）
      "shapeRange": null                  // 动态 shape 时 { "min":…, "max":… }
    }
  ]
}
```

三点说明：

1. **`liveSubgraphs` 是子计算序，不是 tick。** GM 张量的死活由「哪个子计算最后读它」决定，
   与片上流水的 cycle 粒度无关。用子计算序号做时间轴，装箱问题的规模从万级 tick 降到个位数区间，
   求解瞬时完成，结论也更容易讲清楚。tick 只在需要下钻到某个子计算内部时才用。
2. **`blockScope` 是 GM 独有的。** 片上 buffer 天然是核内私有，GM 不是。
   `per-block` 的 workspace 实际占用要乘 `blockDim`，且复用只发生在 block 内部；
   `shared` 的 workspace 被所有 block 并发访问，任何复用建议都必须跨 block 成立。
   这一个字段决定了后面安全护栏的走向，不能省。
3. **`reportedAt` 是「有据可依」的落点。** 所有 workspace 结论最终要指回 tiling 函数里
   写 `currentWorkspace[0] = ...` 的那一行，否则开发者拿到一个数不知道去哪儿改。

---

## 4. 分析核心

新增 `js/workspace-planner.js`，纯函数、无 DOM、不依赖芯片型号，便于后续被 CLI 与
Python API（规划文档 §4.6）用同一套实现复用。

### 4.1 口径

- 所有 size 先做 `alignUp(size, region.align)`（GM 为 512B）再参与计算。
  下界若用未对齐的 size 算，会得出一个物理上不可达的目标值。
- `input` / `output` / `const` 角色的 GM 张量由调用方提供地址，**不参与装箱**，
  但要在视图里画出来（它们是「为什么这块地址不能用」的解释）。只有 `role === 'workspace'` 进求解器。
- `aliasOf != null` 的张量并入宿主，不单独占地址。

### 4.2 理论下界：MaxLive

```
lowerBound = max over sg of  Σ { alignUp(size) | tensor 在该 sg 存活 }
```

这是不可再低的下界：在同一个子计算里同时存活的张量必须同时有地址，无法互相覆盖。

它对**不搬移、连续分配**的前提成立；允许搬移时能更低，但要付拷贝代价（§1.2 已排除）。
复杂度 O(n·s)，n 为 workspace 张量数、s 为子计算数，都是个位到两位数量级。

### 4.3 可达布局：装箱

把「地址 × 子计算序」看成二维平面，每个张量是一个高 = size、宽 = 生命周期的矩形，
要求矩形互不重叠且总高度最小 —— 这正是 **Dynamic Storage Allocation** 问题，已知 NP 难。
所以工具报的是**可达值 `packed`**，不是最优值，并且必须把用的是哪种策略写在结论里。

采用带回退的贪心（Best-Fit / 首个可行偏移）：

```
placed = []
for t in order(tensors):                     // 排序策略见下
    busy = union{ [p.offset, p.offset+p.size) | p ∈ placed, p 与 t 生命周期重叠 }
    t.offset = 第一个满足对齐且长度 ≥ t.size 的空隙起点（无空隙则接在 busy 末尾）
    placed.append(t)
packed = max(t.offset + t.size)
```

**排序策略是这个算法里唯一需要调的旋钮，且它真的会影响结果。** 实现三种并取最优者，
三个高度都留在 evidence 里：

| 策略 | 规则 | 弱点 |
| --- | --- | --- |
| `by-size` | size 降序（经典 BFD） | 把「体积小、活得久」的张量排到最后，此时两侧全被占满，只能挤到峰值之上，凭空抬高一层 |
| `by-lifespan` | 生命周期长者先占坑 | 修掉了上一条，但把「只活一个子计算的大块」（FFN 中间量）推到最后，同样吃亏 |
| `by-order` | 按拓扑序 first-fit | 即真实 arena 分配器带空闲链表的行为。前两者都是「先难后易」，这一条反而常常最紧 |

在 §7 的演示数据上实测（`js/workspace-planner.js` 的输出，可复现）：

```
by-size 10244KB    by-lifespan 10244KB    by-order 10240KB = lowerBound
```

`wsLse`（4KB，横跨 sg2–sg5）是那 4KB 差额的来源：前两种策略都把它排到最后，
两侧已被占满，只能落到峰值之上。`by-order` 之所以能打到下界，是因为程序序天然让
相邻子计算的张量挨在一起，`wsLse` 在 sg2 产出时前面刚好空出 wsQ/wsK 的位置。

> 这里修正了本文 v0.1 的一处错误：当时手算认为 `by-lifespan` 能达到下界，
> 实现后实测不能 —— 纯生命周期排序会把最大的 `wsFfn`（lifespan = 0）推到末尾。
> 这也正好说明为什么工具必须**同时报多种策略的结果**而不是只报一个数：
> 贪心的胜负是数据相关的，没有哪一种恒优。

三种都是贪心，谁都不保证最优。事实上本例存在人工构造的 10240KB 解，
而三种策略里只有一种碰到了它 —— 这就是「报可达值而非最优值」的实际含义。
三种都跑一遍是 O(n² s)，n 是个位数，成本可忽略。

### 4.4 可复用组推荐

装箱给出的是一组 offset，开发者要的是一句人话：「A 和 B 可以共用一块」。

- 建**冲突图**：节点 = workspace 张量，边 = 生命周期重叠。不相邻即可复用。
- 生命周期是区间 ⇒ 冲突图是**区间图**（完美图）。在**同尺寸桶内**，按左端点贪心着色是最优的，
  色数恰等于该桶的最大同时存活个数 —— 这一段的结论可证明最优，可以放心地说「这就是最少份数」。
- **跨尺寸**时退化为 §4.3 的装箱，只报可达值。

推荐条目的输出格式（进 `finding.evidence`）：

| 字段 | 例 |
| --- | --- |
| 复用组 | `wsQ` / `wsAttn` / `wsProj`（互不重叠） |
| 组内峰值 | `max(size) = 2048KB` |
| 节省 | `Σsize − max(size) = 4096KB` |
| 生效条件 | `wsQ` 的最后消费者是 sg1，`wsAttn` 在 sg2 才产出，需在 sg1/sg2 之间保留现有同步 |
| 风险 | 组内成员 `blockScope` 是否一致；不一致时**不推荐**，转 `WS_CROSS_BLOCK_UNSAFE` |

### 4.5 安全护栏

`t32reuse` 候选已经演示过片上手工复用踩内存的样子（`ADDR_CONFLICT`）。GM 上同类错误更隐蔽
——不会立刻算错，而是在特定 block 数、特定 shape 下偶发。护栏必须前置到「工具敢不敢建议」这一步：

1. **死亡点取最后一个消费者的读完成**，不是产出者的写完成。缺 `consumers` 信息时**不出复用建议**，
   降级为「信息不足」而不是猜。
2. **`blockScope: shared` 的张量之间复用，需要全局同步点**（所有 block 都已读完）。
   没有可指认的同步点就不推荐，并说明缺什么。
3. **`per-block` 与 `shared` 之间永不互相复用**，两者地址空间语义不同。
4. **动态 shape**：`shapeRange` 存在时，下界与装箱都按 `max` 侧算，结论用区间表述
   （「S ∈ [512, 2048] 时 workspace 在 5.5MB–21.5MB 之间，需按上界预留」），不给单值。
5. **`aliasOf` 链**上的张量整体参与，不拆开。

### 4.6 下界本身怎么降

`packed` 逼近 `lowerBound` 之后，还想再降就只剩三条路，工具应当把它们作为独立结论给出：

| 手段 | 机制 | 对应规则 |
| --- | --- | --- |
| 原地（in-place） | 消掉一个张量，直接压低对应子计算的存活集合 | `WS_INPLACE_CANDIDATE` |
| 留在片上 | 中间量小于 UB 余量时根本不该落 GM | `FUSION_SPILL` |
| 换执行序 | 拓扑序在有并列分支时不唯一，换序能改变峰值所在 | `WS_SCHEDULE_HINT`（P2） |

§7 演示数据里 RoPE 改原地后，下界从 10240KB 直接掉到 8196KB —— 这是复用做到极致也拿不到的收益，
必须和复用建议分开陈述，否则开发者会以为已经到头了。

---

## 5. 视图设计

### 5.1 落位：独立页面 `workspace.html`

v0.1 原计划做成 `index.html` 的第四个页签。实现时改为**同模块下的第二个页面**，原因有三条：

1. 现有三个页签都是**核内视角**（UB/L1/L0x/寄存器，时间轴是 cycle），场景 6 是**片外视角**
   （GM，时间轴是子计算序）。同一套「焦点层级」「时间游标」控件会被迫承担两种语义。
2. 算子也换了 —— 场景 6 的对象是融合体 `MLABlock_fused`，不是 `MatmulLayerNorm_mix`。
   选中融合算子时，另外三个页签没有可显示的内容（融合体没有建片上流水事件）。
3. 两页共享 `data/chip-specs.js`、`js/format.js`、`js/canvas-kit.js` 与设计系统 pattern，
   只是不共享上下文；顶栏互相有入口。

`view-lifetime.js` 的 `scope === 'core'` 过滤因此不必动 —— GM 的生命周期视图在新页面里。

### 5.2 三块内容

| 区块 | 实现 | 说明 |
| --- | --- | --- |
| 子计算带 | `patterns/swimlane-task` 的 `drawTaskBar` | 拓扑序一根任务条，按子计算类型取 lane-kind 家族色；点击移游标 |
| **堆叠列 + 三条参考线** | 页面自有 canvas（`js/view-ws-plan.js`） | 每列 = 该子计算真正同时存活的字节，分段按「产出它的子计算」着色；横线是 current / packed / lowerBound / budget，右侧留白里用斜纹带标出「策略浪费」「装箱碎片」 |
| GM 布局与复用 | `patterns/memory-reuse-viewer` 单实例 | 「当前布局 / 复用后布局」切换 = 同一实例换 data payload |
| 候选对比条 | 页面自有 canvas（`js/view-ws-gap.js`） | 一行一个候选，条形三段分解，竖虚线为预算 |

**为什么不是 v0.1 设想的「三栏并排 memory-reuse-viewer」**——实现前读了 pattern 才发现两个硬问题：

- 该 pattern 是**整面板**组件（1013 行 JS / 535 行 CSS），自带工具栏、buffer 选择器、
  峰值读数、筛选器、源码面板。并排三份塞进一个 pane 必然挤成一团。
- 「理论下界」**不是一份可实现的布局**（它只是每列存活字节的上包络），
  用地址布局图去画它是在虚构一个不存在的地址分配。

所以对比改由**堆叠列图**承担（那里三个数是三条线，一眼可比），布局图只保留
「当前 / 复用后」两个真实存在的方案。堆叠列图与候选对比条属占用曲线一类，
与 `view-watermark.js` 同源，不触碰 `memory-reuse-viewer` 的
`forbiddenOverrides`（「不得为同一张 tensor lifetime 图另写页面局部 canvas 渲染器」）——
生命周期图仍然只有 pattern 那一份。

两个布局方案共用同一纵轴上界（`max(current, packed)`，经 buffers 的 `capacity` 传入，
属 allowedOverrides），否则切过去看不出高度差。

### 5.3 底部水位与状态条

- `view-watermark.js`：GM 曲线上加两条水平参考线（`packed` 虚线、`lowerBound` 点线），
  与已有的 100% 硬边界线同一套画法。这是页面自有 canvas，不涉及 pattern 契约。
- 状态条增加一项 `WS 2.15×`（`current / lowerBound`），与已有的「焦点层级占用 / 峰值 / 诊断计数」并列。
- 分析日志（`memviz analyze` 形态）增加 workspace 段落，格式与现有诊断行一致。

---

## 6. 新增诊断规则

按 `js/diagnostics.js` 现有 `RULE_META` 的四元组格式（问题 / 位置 / 量化影响 / 建议 + evidence）：

已实现于 `js/ws-diagnostics.js`（单独成文件：现有 `diagnostics.js` 的入口签名是
`(run, metrics)`，metrics 依赖片上 region 与流水事件，融合算子这一路没有，硬塞会把两边的
前置条件都搅浑。四元组形状保持一致）。

| rule | severity | category | 触发 | 建议动作 |
| --- | --- | --- | --- | --- |
| `WS_BUDGET_EXCEEDED` | danger | 预算 | `current > budget` | 若 packed 能回到预算内就只需改地址，否则必须改结构 |
| `WS_ADDR_CONFLICT` | danger | 正确性 | 地址重叠 **且** 生命周期重叠 | 撤销手工复用，或把复用点后移到最后一个消费者之后 |
| `WS_CROSS_BLOCK_UNSAFE` | danger | 正确性 | 地址重叠、生命周期错开，但 `blockScope` 不一致 | 撤销；要复用须先统一作用域并补全局同步点 |
| `WS_ABOVE_LOWER_BOUND` | warn | 容量 | `current / lowerBound > 1.2` | 给出 packed、最优策略名与复用组清单，指回 `reportedAt` 行 |
| `WS_REUSE_MISSED` | warn | 复用 | 存在生命周期不重叠却各占地址的组合，节省 ≥ 1MB | 列出复用组、组内峰值、生效条件 |
| `FUSION_SPILL` | warn | 融合 | 某中间量 size ≤ 25% × UB 容量，却落在 GM | 留在片上，workspace 项整块消失（收益大于任何复用） |
| `WS_DYNSHAPE_RANGE` | warn | 动态 shape | 声明了 `shapeRange` | 结论按区间给；上界超预算时考虑按 shape 分档上报 |
| `WS_PEAK_NOT_LARGEST` | info | 容量 | 最大张量不在峰值子计算的存活集合里 | **抑制无效努力**：砍最大那块碰不到峰值，得动峰值那一列 |
| `WS_INPLACE_CANDIDATE` | info | 融合 | elementwise 子计算的输出与某输入同 size 同 dtype | 改原地，下界本身下降 |
| `WS_LONG_LIVED_SMALL` | info | 复用 | 跨度 ≥ 50% 子计算数且体积 ≤ 1% 下界 | 装箱碎片的来源；优先留片上，否则排到地址空间一端 |
| `WS_PACK_FRAGMENT` | info | 复用 | `packed > lowerBound` | 附三种排序策略的高度，说明这是装不进去而非没复用 |
| `WS_SCOPE_BLOCKED` | info | 复用 | 存在被 blockScope 护栏排除的组合 | 明确劝阻手工合并 —— 这些在甘特图上看不出问题 |
| `WS_ALIGN_PADDING` | info | 对齐 | 累计 padding ≥ 64KB | 张量数很多时才值得处理 |

阈值全部进 `DEFAULT_THRESHOLDS`：`wsRatioWarn: 1.2`、`wsReuseSavingBytes: 1MB`、
`wsLongLivedSpanRatio: 0.5`、`wsLongLivedSizeRatio: 0.01`、`wsAlignPaddingBytes: 64KB`、
`wsOnChipHeadroomRatio: 0.25`。规则可关（`disabledRules`）。

**v0.1 里去掉的两条**，因为在这份数据上永不触发，留着就是死规则：
`WS_SINGLE_HOLDER`（峰值单张量占比 > 60% —— 本例峰值由 5 个等大张量平摊，最高 25%）
与 `WS_LIFETIME_UNBOUNDED`（存活覆盖全部子计算 —— 本例无此张量，
且「提前一次性 alloc」这个症状已由 `WS_ABOVE_LOWER_BOUND` 覆盖）。
`WS_PEAK_NOT_LARGEST` 接管了 `WS_SINGLE_HOLDER` 的意图（抑制无效努力），而且真的会触发。

**降噪约定**（沿用现有规则引擎收紧过的思路）：
`WS_REUSE_MISSED` 只出收益最大的 3 条；**已经踩内存（`WS_ADDR_CONFLICT`）时完全不出复用建议**
—— 先解正确性问题，右栏「可复用组合」同步标注「先解冲突」；
同一 elementwise 子计算的多对原地候选（RoPE 的 Q/K）合成一条，处方相同不拆两条；
`policyWaste === 0` 时复用组清单显示「已排满」而不是继续喊「可省 X」。

---

## 7. 演示数据设计

现有 kernel `MatmulLayerNorm_mix` 只有一个 `mmWorkspaceGm`，撑不起场景 6。
新增 `data/fusion-runs.js`，构造融合算子 **`MLABlock_fused`**（M=1024 tokens，H=1024，FFN 中间维 2816，fp16），
六个子计算、十个 GM 张量。沿用「每个候选只承载一个需要被看见的问题」的原则。

### 7.1 张量与生命周期

| 张量 | size | 产出 | 最后消费 | 存活子计算 |
| --- | --- | --- | --- | --- |
| `wsQ` | 2048KB | sg0 QKVProj | sg1 | sg0–sg1 |
| `wsK` | 2048KB | sg0 | sg1 | sg0–sg1 |
| `wsV` | 2048KB | sg0 | sg2 | sg0–sg2 |
| `wsQr` | 2048KB | sg1 RoPE | sg2 | sg1–sg2 |
| `wsKr` | 2048KB | sg1 | sg2 | sg1–sg2 |
| `wsAttn` | 2048KB | sg2 FlashAttn | sg3 | sg2–sg3 |
| `wsLse` | 4KB | sg2 | sg5 | sg2–sg5 |
| `wsProj` | 2048KB | sg3 OutProj | sg4 | sg3–sg4 |
| `wsNorm` | 2048KB | sg4 Add+LN | sg5 | sg4–sg5 |
| `wsFfn` | 5632KB | sg5 FFN | sg5 | sg5 |

### 7.2 三个数（手算，可作为单测基准）

各子计算的同时存活字节：

| sg | 存活集合 | 合计 |
| --- | --- | --- |
| sg0 | Q K V | 6144KB |
| **sg1** | Q K V Qr Kr | **10240KB** ← 峰值 |
| sg2 | V Qr Kr Attn Lse | 8196KB |
| sg3 | Attn Lse Proj | 4100KB |
| sg4 | Lse Proj Norm | 4100KB |
| sg5 | Lse Norm Ffn | 7684KB |

- `lowerBound = 10240KB`（10.0MB），峰值在 **sg1 RoPE**
- `current = 22020KB`（21.5MB，每个子计算各申请各的、全程不释放）
- `packed = 10240KB`（`by-order` 策略；`by-size` / `by-lifespan` 均为 10244KB）
- GM 预算取 16384KB（16MB）→ 基线**超预算 5636KB**

即：**21.5MB → 10.0MB，下降 53.5%**，全部 11780KB 都是策略浪费（改地址即可消除），
最优策略下装箱碎片为 0。

**这个例子刻意让峰值落在 sg1（QKV + RoPE）而不是 FFN。** 直觉上大家都会先去砍 FFN 的
5632KB，但它所在的 sg5 只有 7684KB，砍到 0 也碰不到 10240KB 的峰值 ——
这正是 `WS_PEAK_NOT_LARGEST` 要说的话，也是这张堆叠列图的价值。

### 7.3 候选集合（表内数字为 `workspace-planner` 实测输出）

| 候选 | current | packed | lowerBound | 比值 | 承载的问题 |
| --- | --- | --- | --- | --- | --- |
| `ws-naive` | 21.5MB | 10.0MB | 10.0MB | 2.15× | 基线，超预算 5.50MB；`WS_BUDGET_EXCEEDED` + 2 条复用建议 |
| `ws-packed` | 10.0MB | 10.0MB | 10.0MB | 1.00× | 落地复用组后的样子；复用组清单转为「已排满」 |
| `ws-inplace` | 8.00MB | 8.00MB | 8.00MB | 1.00× | RoPE 原地（`wsQr`/`wsKr` 消失）→ **下界本身**降到 8196KB |
| `ws-onchip` | 10.0MB | 10.0MB | 10.0MB | 1.00× | `wsLse` 留 UB → 三种排序策略**都**能打到下界（碎片归零） |
| `ws-unsafe` | 17.5MB | 10.0MB | 10.0MB | 1.75× | 两处手工复用：省了 4MB、踩了两处内存，**而且仍比工具排的大 7.5MB** |
| `ws-dynshape` | 20.0MB | 20.0MB | 20.0MB | 1.00× | tokens=2048 上界，超预算 4.00MB；结论按区间给 |

`ws-unsafe` 是刻意保留的反例，它同时验证三件事：§4.5 的护栏真的会拦下这两条路径
（而不是靠人工审查）、手工复用的收益远不如工具排的布局、以及
`WS_CROSS_BLOCK_UNSAFE` 那一类「甘特图上完全看不出问题」的错误确实存在
（`wsProj` 压在 `wsQ` 上，两者生命周期 sg3–sg4 与 sg0–sg1 毫无交叠，
唯一的问题是 `wsQ` 是 per-block 切分）。

`ws-onchip` 的收益在数字上只有 4KB，价值在别处：把它拿掉后三种排序策略的结果**完全一致**，
说明这一小块就是让两种贪心各差 4KB 的那根桩子。这条只能靠「同时报多种策略」才看得见。

---

## 8. 与采集层 / CI 的衔接

### 8.1 数据来源（对应规划文档 §4.1）

| 字段 | 来源 | 阶段 |
| --- | --- | --- |
| `subgraphs` | 融合 pass 的融合决策记录 / 源码中子计算函数边界 | MVP（先手工标注，再解析） |
| `producer` / `consumers` | 源码里 GM tensor 的 `SetGlobalBuffer` + `DataCopy` 方向 | MVP |
| `workspace.current` / `reportedAt` | tiling 函数的 `GetWorkspaceSizes` / `currentWorkspace[i]` 赋值行 | MVP |
| `blockScope` | workspace 地址是否带 `block_idx` 偏移 | MVP+ |
| `shapeRange` | 动态 shape 的 tiling 分档表 | V1 |
| 实际访问区间（校验用） | profiling / trace 的 GM 读写事件 | V1 |

缺 `consumers` 时按 §4.5-1 降级：仍能报 `current` 与「所有 workspace 之和」，不报复用建议。

### 8.2 CI

`memviz analyze --fail-on ws-ratio>1.5`：把 `current / lowerBound` 做成门禁指标。
它比绝对字节数更适合做门禁 —— shape 变了绝对值必然变，但比值反映的是分配策略本身的质量，
不会因为业务侧换个 batch size 就误报。

同时保留 `--baseline` 的绝对值回归（workspace 相比基线上涨 > X% 时告警），两者互补。

---

## 9. 分期与改动清单

### 已交付：产品原型 `Memory-Visual/workspace.html`

| 文件 | 职责 |
| --- | --- |
| `data/fusion-source.js` | 融合算子的 kernel 与 tiling 源码（`ws[0] = ...` 上报行即 `reportedAt`） |
| `data/fusion-runs.js` | §7 的六个候选生成器；张量大小由 shape 推、生命周期由产消关系推 |
| `js/workspace-planner.js` | §4 全部：MaxLive 下界、三种排序的装箱、区间图着色的复用组、护栏、冲突检查。纯函数无 DOM |
| `js/ws-diagnostics.js` | §6 的 13 条规则，四元组 + evidence |
| `js/view-ws-plan.js` | 主视图：子计算带 + 堆叠列 + 三条参考线 + 差距带 |
| `js/view-ws-layout.js` | GM 布局与复用：`memory-reuse-viewer` 数据契约翻译 + 布局方案切换 |
| `js/view-ws-gap.js` | 底部候选对比条 |
| `js/ws-app.js` | 状态与渲染编排、子计算游标播放条、面板开关 |
| `workspace.html` | IDE 框架外壳与脚本装载顺序（planner 须先于 fusion-runs） |

未动 `index.html` 与既有 `js/*`：新页面自带上下文，两页只共享 `chip-specs` / `format` /
`canvas-kit` 与设计系统 pattern。原 P0 里「放开 `view-lifetime.js` 的 core 过滤」
「`view-watermark.js` 加参考线」因此都不需要了。

### 仍未做（原 P0/P1 的剩余项）

- `data/runs.js:719` 把 GM interval 拉平成全程那几行仍在 —— 单算子页的 GM 生命周期依旧是直线。
  这只影响 `index.html`，不影响本原型；要在单算子页也看 GM 复用时再改。
- `data/chip-specs.js` 的 GM `capacity` 语义未澄清（本原型用 `run.workspace.budget`，
  没有依赖 chip 上的那个值）。

### P2

- 跨算子 / 图级 workspace 复用（对齐 GE 内存复用策略）。
- 执行序调整建议（`WS_SCHEDULE_HINT`）。
- 动态 shape 分档的区间视图。
- Python API：`plan_workspace(schema) -> {current, packed, lowerBound, groups}`，
  供自动 tiling 搜索当评估函数用。

---

## 10. 已知取舍与未决问题

1. **装箱只报可达值。** DSA 是 NP 难，工具不宣称最优。两种排序策略的结果都写进 evidence，
   开发者可以自己判断还有没有余量。若未来需要更紧的解，可加一路 ILP/CP-SAT 兜底（离线跑，不进交互路径）。
2. **子计算序 vs tick 的粒度。** 用子计算序做时间轴会低估复用机会：同一个子计算内部，
   某个张量可能前半段就死了。这个精度损失换来了求解规模与可解释性，属**有意为之**；
   接入 trace 数据后可选择性下钻到 tick 粒度重算，作为「更激进的复用建议」单列。
3. **`blockScope` 的采集不可靠。** 静态解析未必能判定 workspace 是否带 `block_idx` 偏移。
   采集不到时一律按 `shared` 处理（保守 → 少给建议），并在结论里标注「按最保守假设」。
4. **下界的前提是执行序固定。** 有并列分支的融合体，换序会改变下界本身，
   所以严格说 `lowerBound` 是「当前执行序下的下界」，文案必须这么写，不能简写成「理论最小 workspace」。
5. **与 GE 侧内存复用的关系待对齐。** 若 GE 已经在算子外做了 workspace 池化，
   工具报的 `current` 可能不等于最终显存开销。当前阶段先只对算子上报值负责。
6. **`workspace.html` 复制了 `index.html` 的一份 mv- 布局 CSS。** 两页外观由构造保证一致，
   但同名类有了两处来源。抽成 `Memory-Visual/css/workbench-shell.css` 是正确做法，
   代价是要改动已经在跑的 `index.html`，所以留到下次同时动两页时一起做。
7. **IDE 框架的分栏不随视口 resize 重排**（切窗口大小后需刷新一次）。
   这是 `ide-frame` + 存储的像素分栏的既有行为，`index.html` 同样如此，不是本页引入的。
