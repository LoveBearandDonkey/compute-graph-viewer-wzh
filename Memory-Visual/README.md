# PTO 内存工作台

昇腾算子片上内存可视化工具的可运行原型。入口：`Memory-Visual/index.html`（需 HTTP 服务，`file://` 下取不到设计系统资源）。

```bash
python3 -m http.server 8765
```

然后打开 http://127.0.0.1:8765/Memory-Visual/

需求来源：[昇腾算子内存可视化工具-规划.md](昇腾算子内存可视化工具-规划.md)

---

## 首版范围

覆盖规划文档 §4.3 的三个视图，对应三个高频场景：

| 视图 | 规划文档 | 回答的问题 |
| --- | --- | --- |
| 内存布局 | §4.3-1 + §4.3-3 | 各层级预留了多少、此刻真在用多少、超限超在哪 |
| └ 地址布局 | §4.3-1 | 地址空间摊平的条带图（原有形态） |
| └ 硬件架构 | §4.3-1 | 同一份读数贴回 `patterns/memory-architecture` 的硬件架构图：物理容量、对齐、预留、利用率、峰值直接挂在对应存储卡片上 |
| 生命周期与复用 | §4.3-2 | 哪两个 tensor 生命周期不重叠可以合并、哪两个地址重叠会踩内存 |
| 流水 × 内存 | §4.3-4 | double buffer 到底有没有生效、等待空洞卡在谁身上 |

左栏的 tiling 候选列表同时是 §4.3-6「tiling 对比」的数据基础：五组候选共用同一套推导规则，视图上的差异真实来自参数差异。

暂未实现：利用率热力图（§4.3-5）、并排 Diff 视图（§4.3-6 的差异高亮部分）、层级下钻到模型/算子级（§4.3-7）。

---

## 第二个页面：融合与 workspace（场景 6）

入口 `Memory-Visual/workspace.html`，对应规划文档 §3 场景 6，方案见
[场景6-workspace与GM规划-方案设计.md](场景6-workspace与GM规划-方案设计.md)。

单算子页是**核内视角**（UB/L1/L0x/寄存器，时间轴是 cycle）；这一页是**片外视角**
（GM，时间轴是子计算序），算子也换成融合体 `MLABlock_fused`。因此独立成页而不是加第四个页签
—— 否则「焦点层级」「时间游标」两个控件要同时承担两种语义，而另外三个页签对融合算子无内容可显示。
两页共享 `chip-specs` / `format` / `canvas-kit` 与设计系统 pattern，顶栏互相有入口。

**只回答三个数**，其余都是它们的展开：

| 数 | 含义 | 谁能改 |
| --- | --- | --- |
| `current` | 当前 tiling 上报的 workspace（溯源到 `ws[0] = ...` 那一行） | 现状 |
| `packed` | 保持顺序与形状不变、仅做地址复用可达的值 | 改分配策略即可，**改动确定安全** |
| `lowerBound` | 当前执行序与形状下的理论下界（最大同时存活字节） | 只能靠原地 / 留片上 / 换序再降 |

差距刻意拆成两段：`current − packed` 是**策略浪费**（改地址就能拿回来），
`packed − lowerBound` 是**装箱碎片**（要动结构）。只报一个够不着的最小值会让人放弃。

| 视图 | 回答的问题 |
| --- | --- |
| Workspace 规划 | 每个子计算真正同时存活多少字节（柱子），三个数各在哪条线上，白占了多少 |
| GM 布局与复用 | 当前布局 / 复用后布局的地址分配差别（`memory-reuse-viewer` 单实例换 data） |
| 候选对比（底部） | 六个候选的 `[下界 + 碎片 + 浪费]` 三段分解与预算线 |
| 可复用组合（右栏） | 哪几个张量能共用一段地址、省多少、生效条件，以及**被护栏排除**的组合 |

**装箱只报可达值。** 这是 Dynamic Storage Allocation，NP 难。求解器跑三种排序
（`by-size` / `by-lifespan` / `by-order`）取最紧者，三个高度都进 evidence ——
演示数据上前两者都是 10244KB、只有拓扑序 first-fit 打到下界 10240KB，
可见贪心的胜负是数据相关的，没有哪一种恒优。

**GM 的复用安全判据和片上不同。** 片上 buffer 天然核内私有，GM 不是：`blockScope`
为 `per-block` 的张量被各 block 各持一段，与 `shared` 张量共用地址即使生命周期完全错开也不安全
（`ws-unsafe` 候选专门演示这种「甘特图上看不出问题」的错误）。护栏排除的组合会显式列出来，
避免开发者自己去合。

演示算子 `MLABlock_fused`：6 个子计算、10 个 GM 张量，基线 21.5MB → 下界 10.0MB（降 53.5%）。
峰值刻意落在 QKV+RoPE 而不是 FFN —— 最大的那块张量砍到 0 也碰不到峰值，
这正是 `WS_PEAK_NOT_LARGEST` 要说的话。

```
data/fusion-source.js   融合算子 kernel 与 tiling 源码（workspace 上报行 = 结论的溯源落点）
data/fusion-runs.js     六个候选生成器：大小由 shape 推、生命周期由产消关系推
js/workspace-planner.js 下界 + 装箱 + 复用组 + 护栏 + 冲突检查（纯函数，供 CLI/Python 复用）
js/ws-diagnostics.js    13 条 workspace 规则
js/view-ws-plan.js      主视图：子计算带 + 堆叠列 + 三条参考线
js/view-ws-layout.js    GM 布局：memory-reuse-viewer 数据契约翻译
js/view-ws-gap.js       底部候选对比条
js/ws-app.js            状态与渲染编排
```

---

## IDE 框架分区映射

整个产品外壳是设计系统的 `patterns/ide-frame`（standalone host）：

| IDE 槽位 | 承载内容 |
| --- | --- |
| topbar | 产品标识、当前 kernel/候选、芯片型号切换、三个面板开关 |
| activity rail | Tiling 候选、诊断、分析日志 |
| explorer（左，288px） | kernel 信息卡 + 五组 tiling 候选，每组带占用条与诊断计数 |
| preview tabs | 三个主视图切换 |
| pane header | 焦点层级选择、跳到峰值、跳到首个问题、图例 |
| preview body | 三个视图的画布 |
| inspector（右，340px） | 诊断列表（问题 / 量化影响 / 建议 / 溯源）+ 选中项详情与源码片段 |
| bottom dock · visualization | 各存储层级占用水位曲线（910B 六条，950B 含 VRF/SRF 共八条） |
| bottom dock · terminal | `memviz analyze` CLI 形态的分析日志与退出码（§4.6 CI 集成） |
| status strip | 芯片、候选、tileM、焦点层级占用、峰值、游标、诊断计数、数据等级 |
| floating playback | 时间游标（← → 单步，Shift 加速，空格播放/暂停） |

复用的其他 pattern：`workbench-shell`（分栏）、`memory-architecture` + `aic-core-object` + `aiv-core-object`（内存布局的硬件架构布局）、`memory-reuse-viewer`（整块承担生命周期视图）、`swimlane-task`（流水任务条与悬浮提示）、`floating-playback-control`（播放条）。

---

## 目录结构

```
data/
  chip-specs.js      芯片描述（910B / 950B）—— 容量、bank、对齐、寄存器堆、流水单元集合，表驱动
  kernel-source.js   样例 kernel 源码，供源码联动与代码片段使用
  runs.js            中间格式生成器：tiling 参数 → 静态布局 + 流水事件序列
js/
  format.js          字节 / 比例 / 地址格式化
  canvas-kit.js      DPR 适配、token 取色、圆角、斜纹、悬浮提示
  metrics.js         派生指标：预留/持有双曲线、峰值构成、流水占空比
  diagnostics.js     规则引擎，输出「问题 + 位置 + 量化影响 + 建议」四元组
  view-layout.js     视图 A-1：地址空间分栏条带图
  view-arch.js       视图 A-2：硬件架构布局（memory-architecture pattern + 实时读数）
  view-lifetime.js   视图 B：中间格式 → memory-reuse-viewer 数据契约
  view-pipeline.js   视图 C：流水泳道 + 占用曲线
  view-watermark.js  底部：六层级水位曲线
  app.js             状态与渲染编排
```

---

## 数据模型要点

**两条曲线，不是一条。** `reserved` 是 `TPipe::InitBuffer` 的静态预留，编译期就决定是否超限；`live` 是某一时刻真正被张量持有的字节。两者之差就是「预留了但没在用」的浪费，也是复用建议的量化依据。布局图里实心块 = live，半透明块 = reserved 但当前为空。

**事件序列是模拟出来的，不是编的。** 每条流水线是串行队列，一步的开始时刻 = `max(本流水线空闲时刻, 所读分配的写完时刻, 目标 slot 的释放时刻)`；`buffer_num = N` 的队列有 N 个 slot，第 i 次迭代用 slot `i%N`。double buffer 是否生效、等待空洞出现在哪里，都由这个模型自然产生 —— 所以候选 `tileM=32` 和 `tileM=32 + double buffer` 的时长差（292 → 268 cycle）是算出来的。

**寄存器是一等存储层级（950 起）。** `chip-specs.js` 里 `region.kind === 'register'` 的层级同样表驱动 —— 有容量、有对齐、会超限，只是计量单位是「寄存器个数」而非地址偏移。950 的 A5 RegBase 写法（`loadalign` → VF 计算 → `storealign`）把 Normalize/Cast 的中间量从 UB 挪进 `VRF`，所以同一 tiling 在 950 上 UB 压力下降、寄存器压力上升；活跃寄存器数与展开度线性相关，装不下的部分溢出到 UB（`REG_SPILL`）。SIMT 侧的 `SRF` 按 warp 整块切分，每线程寄存器用量直接决定并发 warp 数（`REG_OCCUPANCY`）。这几条正是「片上内存管理」在 950 上多出来的那一层取舍。

**规则可开关、阈值可配。** `MemVizDiagnostics.analyze(run, metrics, { thresholds, disabledRules })`。每条结论都带 `evidence` 原始数据项引用，不做黑盒推断。刻意收紧过的两处：同一队列的两个 ping-pong slot 不会被建议「互相复用」；只报能归因到具体分配的流水等待，低占空比流水（如 MTE3）天然空闲不算问题。

---

## 候选设计

每组候选只承载一个需要被看见的问题，不把所有毛病堆进一次运行：

| 候选 | 结论 |
| --- | --- |
| `tileM=64` | UB 预留 230KB / 192KB，超 38KB；工具直接给出「降到 48 可释放 57KB」 |
| `tileM=32` | 不超限，但 `mmOutQue` 单份，累计等待 96 cycle 占 33% |
| `tileM=32 + double buffer` | 上一条的解：+32KB 换掉全部可归因等待，Cube/Vector 占空比打满 |
| `tileM=32 + 手工复用` | `normBuf` 复用 `mmOutQue` slot 0 省 32KB，但 Normalize 仍在读 mmOut → 偶数迭代地址冲突 |
| `tileM=16` | 尾块浪费最小，但 UB 只用 43%，搬运次数 13 轮 |

上表是 910B 的读数。切到 950B 时同一组候选走 RegBase 路径，结论随之变化：`normBuf` 消失让 UB 全线降压（`tileM=64` 从超限变成 65%），瓶颈转移到寄存器 —— `tileM=64` 展开 8 组要 68 个向量寄存器、只有 64 个，溢出 4 个到 UB；每线程 96 个寄存器把并发 warp 从 8 压到 5。`tileM=32 + 手工复用` 在 950 上改由 `tmpSqBuf` 复用 `mmOutQue` 首个 slot，同样触发地址冲突。

---

## 数据等级

L2 / schema-generated / share-safe，exploration-only。

`MatmulLayerNorm_mix` 是为演示构造的 mix 算子，不对应任何真实产品算子。芯片容量、bank 数、对齐粒度为规格量级的**占位值**，实际以对应型号官方规格为准（见 `chip-specs.js` 的 `specRef` 字段）。规划文档明确要求这些数值必须来自可配置的芯片描述文件而非硬编码，本原型按此实现。

---

## 已知待补

- `.badge--success/warning/danger` 与 `.inspector-section` / `.inspector-soft-card` 在 `quick-reference.md` 有约定但当前 `css/style.css` 未实现，本页用 `.stat-chip + mv-sev-*` 与 `mv-sec / mv-soft` 局部实现，待设计系统吸收后替换。
- 采集层（§4.1）目前只有生成器一路。接真实数据时替换 `data/runs.js` / `data/fusion-runs.js` 的输出即可，视图与规则引擎只认中间格式。
- `workspace.html` 复制了 `index.html` 的一份 mv- 布局 CSS，两页外观由构造保证一致，但同名类有了两处来源。抽成 `css/workbench-shell.css` 是正确做法，代价是要改动已在跑的 `index.html`，留到下次同时动两页时一起做。
- `data/runs.js:719` 把 GM 分配的生命周期拉平成全程，所以**单算子页**的 GM 复用分析仍然是空的（场景 6 的分析在 `workspace.html`，不受影响）。
