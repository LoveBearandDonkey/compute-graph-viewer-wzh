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
- 采集层（§4.1）目前只有生成器一路。接真实数据时替换 `data/runs.js` 的输出即可，视图与规则引擎只认中间格式。
