# 计算图 Profiling 证据工作台 PRD

**产品代号**：Compute Graph Profiling Evidence Workbench  
**版本**：v0.1 草案  
**日期**：2026-06-06  
**状态**：产品定义中  
**目标页面**：`Profiling_Insight_and_Tool/AI_Profiling_Tool/graph-evidence-workbench.html`  
**来源页面**：`Profiling_Insight_and_Tool/AI_Profiling_Tool/MindStudioNext.html` 的「计算图」tab  
**参考页面**：`graphviz/deepseek_v32_report_overlay_demo.html`

---

## 1. 一句话定义

计算图 Profiling 证据工作台是一个面向 Ascend 模型训练/推理开发者的独立诊断页面。它把模型计算图、Profiling 问题节点、右侧诊断证据和底部 Step/Stream 泳道图放在同一个可联动界面里，让开发者从“模型结构上的哪个模块有问题”直接跳到“运行时哪个 step、stream、kernel、wait 或通信段证明了这个问题”。

### 1.1 已确认范围

- 首个支持模型：P0 只优先支持 Qwen2-7B；DeepSeek V3.2 overlay 作为参考形态和后续扩展，不进入 MVP 主路径。
- 真实数据解析：MVP 必须解析真实 `trace_view.json`，不能只消费当前 HTML 里的 `SWIMLANE_DATA` / `FREE_ANALYSIS_DATA`。
- 数据外置：`GRAPH_PROBLEMS`、`QWEN7B_BASE_NODES`、`QWEN7B_NODE_INFO`、demo report、timeline evidence 等数据必须从 HTML 内联迁出为 JSON。
- 内置 demo：页面提供内置 Qwen2-7B demo 数据，保证无外部文件时也能完整展示产品闭环；demo 数据必须走同一套 JSON schema 和解析链路。

---

## 2. 背景与官方文档依据

### 2.1 官方工具链给出的分析路径

MindStudio Insight 的官方定位是面向 Ascend AI 开发者的可视化调优工具，覆盖系统调优、算子调优、服务化调优和内存调优；系统调优提供 Timeline、Memory、Operator、Summary、Communication 等视图，用于快速定位模型性能瓶颈。官方文档同时强调 Insight 可处理真实软硬件运行数据，并支持大规模集群 Profiling 数据分析。

对本产品最关键的官方信息是：

- Timeline 用于展示训练/推理过程中 host 和 device 的运行细节，关联 host API 耗时和 device task 耗时，帮助识别 host/device 瓶颈。
- Timeline 的层级包含 Python/PyTorch、CANN 层 AscendCL/GE/Runtime，以及 Ascend Hardware 下各 stream task flow、step trace、Communication、Overlap Analysis、Memory 等信息。
- `trace_view.json` 可在 TensorBoard、`chrome://tracing/` 和 Perfetto 中打开；其中包含上层应用、CANN 层、底层 NPU 数据和事件详情。
- `kernel_details.csv` 记录 NPU 上执行的 operators，包括 Step Id、Task Id、Stream ID、Name、Type、Accelerator Core、Start Time、Duration、Wait Time、Block Dim、Input/Output Shapes 等字段。
- Summary 支持通信组识别、计算/通信耗时拆解、慢卡/慢链路分析；Communication 用于查看通信时长、等待时长、链路带宽。
- `msprof-analyze advisor` 可分析 Ascend PyTorch Profiler 采集数据并输出性能调优建议；cluster 分析会生成 `cluster_step_trace_time.csv`、`cluster_communication_matrix.json`、`cluster_communication.json` 等文件。
- Operator 视图提供单算子维度和算子类型维度的耗时统计，并支持从算子详情跳转到 Timeline。
- Source/Details 等算子调优视图可基于 `visualize_data.bin` 展示源码与指令/耗时之间的映射。

### 2.2 当前产品断点

`MindStudioNext.html` 已经有「计算图」tab：左侧模型图用 `PtoModelGraphvizPattern.render` 渲染 Qwen2-7B 结构，右侧展示问题节点的指标、影响、修复建议、验证方式和算子背景。它解决了“问题落在模型结构哪里”的问题，但仍有三个不足：

1. 计算图被塞在大工具的一个 tab 里，独立传播、深链定位和专注分析能力弱。
2. 计算图问题节点与 Timeline 证据是分离的，用户需要在问题详情、Timeline、算子视图之间来回切换。
3. 参考页 `deepseek_v32_report_overlay_demo.html` 底部的 Step / Stream Timeline 仍是本地 DOM bar 图，不符合 PTO 已沉淀的泳道图 pattern，也不足以承载 trace 级别的 stream/task 证据。

### 2.3 产品机会

官方工具链强调多视图分析，但开发者实际调优时经常从某个模型模块或算子开始追问：这个问题到底是 LM Head、Attention、MLP、通信、Host 下发还是某个 stream 上的 wait？本产品把“模型结构图”和“Profiling 时间线证据”前后打通，降低开发者在多个视图之间做人工对齐的成本。

---

## 3. 用户与价值

| 用户 | 典型问题 | 本产品带来的价值 |
|---|---|---|
| 大模型训练开发者 | 不知道慢 step 是哪个模型模块拖慢的 | 在计算图上直接看到 P0/P1/P2 问题节点，并定位到 step/stream 证据 |
| 性能调优工程师 | 需要在 Summary、Communication、Timeline、Operator 多视图之间人工跳转 | 在一个页面完成“结构定位 -> 运行时证据 -> 修复建议 -> 验证指标”闭环 |
| 框架/并行策略工程师 | PP/TP/DP/EP 问题难以映射回模块结构 | 通过图节点和泳道图联动识别 pipeline bubble、通信等待、rank 不均衡 |
| 算子/内核工程师 | 知道某个 op 慢，但缺少上下游上下文 | 在图中看到 op 所属模块、输入输出路径，并在泳道图中看到 stream/task 位置 |
| 技术负责人/评审者 | 调优报告难复核 | 页面天然形成可复核证据链：节点、指标、时间线、数据文件、建议和验收指标 |

核心价值不是替代 MindStudio Insight，而是补上“模型结构语义”和“Profiling 运行时证据”之间的解释层，让开发者更快判断应该改模型并行策略、框架调用、算子实现、通信配置还是采集方式。

---

## 4. 产品目标

### 4.1 用户目标

- 30 秒内判断当前 Profiling 报告是否有关联计算图，以及最严重的问题节点在哪里。
- 2 分钟内从一个 P0/P1 节点看到对应的 step、stream、kernel、wait、communication 或 free/bubble 证据。
- 10 分钟内输出一个可复核的调优判断：问题原因、涉及模块、证据文件、建议修改点、重采后的验证指标。

### 4.2 工程目标

- 从 `MindStudioNext.html` 的「计算图」tab 提取独立页面，不复制整个 AI Assistant/报告工作台。
- 复用 PTO 设计系统和共享 pattern：`model-graphviz`、`swimlane-task`、`panel-shell`、`btn`、`segment-control`、`tag/status-chip`。
- 将 `deepseek_v32_report_overlay_demo.html` 的底部 Step / Stream Timeline 重构为泳道图面板，禁止继续使用页面本地 DOM bar 图表达 stream task。
- 形成可扩展数据契约，MVP 主线支持 Qwen2-7B，后续再扩展 DeepSeek 类模型图和未来 profiling report overlay。
- 将当前 HTML 内联数据迁出为 JSON，并让内置 demo 与真实文件解析共用同一套加载器。

---

## 5. 产品原则

- **结构优先**：计算图是主入口，Timeline 是证据，不反过来用时间线淹没结构。
- **证据可追溯**：每条结论必须能回到 `trace_view.json`、`kernel_details.csv`、`step_trace_time.csv`、`communication*.json`、`analysis.db` 或 `visualize_data.bin`。
- **联动少跳转**：点击图节点、泳道任务、问题列表、mapped node 都应同步 selection。
- **不造新视觉系统**：页面必须消费 PTO design system 与 shared patterns。
- **轻量独立**：作为单独 HTML 页面可直接打开/挂到 launch，不依赖 MindStudioNext 的完整 shell。
- **报告和原始数据分层**：AI 诊断文本可作为解释层，但 timeline/operator/communication evidence 必须保留原始数据来源标记。

---

## 6. 信息架构

目标页面采用 `deepseek_v32_report_overlay_demo.html` 的大图 + 右侧 inspector + 底部证据时间线布局，但用 PTO shared pattern 收敛视觉和行为。

```text
┌──────────────────────────────────────────────────────────────────────┐
│ Header: Home / Report name / Data source / Priority filters / Export  │
├──────────────────────────────────────────────┬───────────────────────┤
│                                              │ Right Inspector        │
│ Model Graph Stage                            │ - Diagnosis           │
│ - model graphviz pattern                     │ - Evidence            │
│ - report priority overlays                   │ - Operators           │
│ - mapped node selection                      │ - Actions             │
│ - pan / zoom / fit                           │ - Mapped Nodes        │
│                                              │ - Data Source         │
├──────────────────────────────────────────────┴───────────────────────┤
│ Bottom Evidence Swimlane                                               │
│ - Step lanes / Stream lanes / Communication lanes / Coverage            │
│ - pattern/swimlane task bars                                            │
│ - playhead / search / filter / focus selected node                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 7. 核心功能需求

### F1. 独立页面抽取

**描述**：从 `MindStudioNext.html` 的「计算图」tab 抽取独立页面，保留计算图、问题节点、节点详情、名称映射、pan/zoom、cluster expand/collapse。

**范围**：

- 保留 `buildQwenGraphData`、pan/zoom、selection、cluster expand/collapse 等页面逻辑。
- 将 `GRAPH_PROBLEMS`、`QWEN7B_BASE_NODES`、`QWEN7B_NODE_INFO` 等数据迁出为 JSON 文件，由页面异步加载。
- 保留 `PtoModelGraphvizPattern.render`，后续优先迁移到 vendored `patterns/model-graphviz/pattern.js`，避免长期内嵌 pattern 代码。
- 不保留 AI 对话、报告 markdown tab、代码 tab、文档 tab、历史记录侧栏。
- MVP 首个模型只支持 Qwen2-7B；非 Qwen2-7B 报告显示明确“不在 MVP 支持范围”的 empty state。

**验收标准**：

- 新页面能独立打开并渲染 Qwen2-7B 计算图。
- 页面渲染所需业务数据来自 JSON，不再来自 HTML 内联常量。
- 无关联图的报告显示明确 empty state。
- 点击 P0/P1/P2 节点，右侧 inspector 更新。
- cluster 折叠/展开后保持视口锚点和 selection。

### F2. 参考 report overlay 交互

**描述**：参考 `graphviz/deepseek_v32_report_overlay_demo.html`，将诊断结果作为 overlay 显示在模型图上，并提供 priority filter。

**需求**：

- Header 提供 All / P0 / P1 / P2 / Off 过滤。
- 右侧 inspector 默认显示最高优先级节点。
- Mapped Nodes 列表可点击并 focus 图节点。
- Selection dimming：选中节点时非相关节点弱化。
- 所有 P0/P1/P2 使用 `model-graphviz` pattern 已定义的 priority overlay，不再画额外可见边框。

**验收标准**：

- 过滤 P0 时仅保留 P0 overlay 和对应 mapped list。
- Off 时保留纯计算图，隐藏报告 overlay 和 inspector 诊断态。
- 从 Mapped Nodes 点击节点时，图 stage 平滑定位到该节点。

### F3. 底部 Step / Stream Timeline 替换为泳道图

**描述**：将参考页底部 Step / Stream Timeline 的 DOM stack/stream bar 替换为 PTO 泳道图面板，使用 `vendor/pto-design-system/patterns/swimlane-task` 的 task bar renderer 和 tooltip。

**泳道分层**：

- Step lane：按 Step ID 展示 Computing、Communication(Not Overlapped)、Free/Bubble、Preparing 等阶段。
- Stream lane：按 Stream ID 展示 Ascend Hardware task flow，突出 selected graph node 相关 kernel。
- Communication lane：展示 HCCL/hcom 通信任务、wait/transit、通信域、rank/group。
- Overlap lane：展示 overlap/free 分析，用于识别计算通信重叠不足。
- Coverage lane：展示当前报告中哪些图节点有 runtime evidence，哪些只有 AI 推断或报告文本。

**交互**：

- 点击泳道 task，高亮相关计算图节点和 inspector evidence。
- 点击计算图节点，泳道自动滚到该节点的 runtime evidence。
- 支持 playhead、拖拽/滚轮缩放、按 task name/operator name 搜索。
- tooltip 必须包含 op/task 名称、lane、start、duration、wait、stream/rank、source file。
- 内置 demo 也必须通过 `trace_view.json` parser 产出泳道数据，不能手写最终泳道条形数据绕过解析器。

**验收标准**：

- 不再使用 `step-row`、`stream-lane-row` 这类本地 DOM bar 作为主 timeline 表达。
- Task bar 由 `PtoSwimlaneTaskPattern.drawTaskBar` 绘制。
- Tooltip 由 `PtoSwimlaneTaskPattern.initHoverTooltip` 提供或兼容其数据结构。
- 能解析真实 `trace_view.json` 并生成 Step/Stream lanes。
- 缺少 trace 数据时显示数据缺口，而不是伪造完整泳道图。

### F4. 证据数据契约

**描述**：定义独立页面消费的数据结构，保证图、问题、timeline、operator、communication 可关联。

**输入数据类型**：

| 数据 | 典型来源 | 用途 |
|---|---|---|
| Model graph schema | 本地模型结构 JSON / `QWEN7B_BASE_NODES` | 渲染计算图 |
| Problem node mapping | AI report / `GRAPH_PROBLEMS` | 将 P0/P1/P2 问题映射到图节点 |
| `trace_view.json` | Ascend PyTorch Profiler | 生成 Step/Stream/Host/CANN/NPU 泳道 |
| `kernel_details.csv` | Ascend PyTorch Profiler | 提供 operator、stream、duration、wait、shape 字段 |
| `step_trace_time.csv` / cluster step trace | profiler / msprof-analyze cluster | Step 对比、computing/communication/free |
| `communication.json` / `communication_matrix.json` | profiler / cluster output | 通信 wait/transit/bandwidth evidence |
| `analysis.db` | profiler parsed output | Summary/Communication 详情数据 |
| `visualize_data.bin` | msProf op simulator | Source/Details 级别算子证据 |

**MVP JSON 文件建议**：

```text
AI_Profiling_Tool/
├── graph-evidence-workbench.html
└── data/
    ├── qwen2-7b.graph.json
    ├── qwen2-7b.node-info.json
    ├── qwen2-7b.problem-map.json
    ├── qwen2-7b.demo-report.json
    ├── qwen2-7b.demo.trace_view.json
    └── qwen2-7b.demo.evidence.json
```

**加载策略**：

- 默认加载内置 Qwen2-7B demo JSON，保证页面无外部文件时可完整展示。
- 当用户提供真实 Profiling 文件时，以真实 `trace_view.json` 为准重新生成泳道 evidence。
- `qwen2-7b.demo.evidence.json` 只能作为 parser regression fixture，不允许作为 MVP 的唯一渲染来源。
- JSON schema 要保留 `schemaVersion`、`modelId`、`reportId`、`sourceFiles`、`generatedAt` 字段，避免 demo、真实报告和后续模型混用时无法追踪来源。

**关联键**：

- `nodeId`：模型图节点 ID。
- `issueId`：报告问题 ID。
- `opName` / `runtimeOpName`：模型语义名与 profiler 算子名。
- `stepId`、`rankId`、`streamId`、`taskId`：runtime 定位。
- `sourceFile`、`sourceLine`：源码/报告来源。
- `confidence`：映射置信度，区分 raw、derived、AI inferred。

**验收标准**：

- 页面无业务数据硬编码；业务 JSON 结构变化时应在加载阶段给出 schema error。
- 每条 inspector evidence 显示数据来源和置信度。
- 对同名算子、多实例算子、cluster folded node 允许一对多映射。
- 无法映射时显示 unmapped reason。

### F5. 右侧 Inspector

**描述**：右侧面板给出当前选中对象的诊断闭环。

**信息结构**：

1. Diagnosis：问题摘要、priority、dimension、影响指标。
2. Evidence：关键事实，含数据文件和字段。
3. Operators：Top operators / mapped runtime ops。
4. Actions：修复建议和参数位置。
5. Verification：重采后应观察的指标。
6. Mapped Nodes：相关计算图节点列表。
7. Data Coverage：当前证据覆盖率和缺失数据。

**验收标准**：

- 所有指标必须带单位。
- 修复建议必须能关联到至少一个 evidence 或标为 inferred。
- Verification 不能只写“重采验证”，必须包含目标指标变化。

### F6. 深链和导出

**描述**：支持页面级分享和复核。

**需求**：

- URL 参数支持 `reportId`、`nodeId`、`priority`、`stepId`。
- 支持导出当前视图为 JSON snapshot。
- 支持复制 evidence summary。

**验收标准**：

- 打开带 `nodeId` 的 URL 后自动选中该节点并定位泳道图。
- 导出内容包含版本、输入文件、selected state、evidence list。

---

## 8. 非功能需求

### 8.1 性能

- 首屏 2 秒内出现 header、empty/loading、inspector shell。
- 10k task 以内泳道图交互保持可用；超过阈值时启用采样/虚拟滚动策略。
- 图节点点击到 inspector 更新小于 100 ms。
- Timeline 搜索结果定位小于 300 ms。

### 8.2 可靠性

- 原始 profiler 文件缺失时，页面降级显示缺失字段，不报错空白。
- AI 推断映射必须和 raw evidence 分层显示。
- 多报告切换时清理 selection、playhead、tooltip、scroll 状态。

### 8.3 可访问性

- Priority filter、playhead、timeline task selection 支持键盘操作。
- 泳道图 tooltip 内容在 inspector 中也应有文本版本，不能只靠 hover。
- P0/P1/P2 不能只用颜色区分，必须有文字标签。

---

## 9. 设计系统与 Pattern 约束

必须复用：

- `vendor/pto-design-system/tokens/foundation.css`
- `vendor/pto-design-system/tokens/semantic.css`
- `vendor/pto-design-system/tokens/components.css`
- `vendor/pto-design-system/css/style.css`
- `vendor/pto-design-system/patterns/model-graphviz/pattern.css`
- `vendor/pto-design-system/patterns/model-graphviz/pattern.js`
- `vendor/pto-design-system/patterns/swimlane-task/pattern.css`
- `vendor/pto-design-system/patterns/swimlane-task/pattern.js`

推荐复用：

- `panel-shell` / `panel-shell-quiet`
- `btn` / `btn-ghost` / `segment-control`
- `tag` / `status-chip` / `priority-badge`
- `inspector-rail` token family

禁止：

- 在业务页重写 model graphviz 的节点几何、priority overlay、cluster title pill、fold control。
- 在业务页用本地 DOM/CSS 重新实现泳道 task bar segment。
- 对卡片/面板使用额外彩色边框、左侧 rail、阴影或非系统圆角。
- 使用远程 CDN 作为核心运行依赖。

---

## 10. 里程碑

### M0. PRD 与数据盘点

- 完成 PRD。
- 盘点 `MindStudioNext.html` 中计算图相关数据和函数。
- 列出可直接迁移、需抽象、需补数据的部分。

### M1. 独立静态页面 MVP

- 新建 standalone page。
- 接入 `model-graphviz` pattern。
- 完成图节点点击、右侧 inspector、priority filter。
- 接入 `AI_Profiling_Tool/data/*.json` 加载器，Qwen2-7B demo 不再内联在 HTML。
- 不接入泳道图，只保留底部 empty/loading shell。

### M2. 泳道图证据面板

- 接入 `swimlane-task` pattern。
- 实现真实 `trace_view.json` parser，至少支持 PyTorch/CANN/Ascend Hardware 事件分层、Step ID、Stream ID、task duration。
- 用 `data/qwen2-7b.demo.trace_view.json` 生成 Step/Stream lanes。
- 完成图节点和泳道 task 双向联动。

### M3. 数据契约与多报告支持

- 完成 HTML 内联数据迁移，业务数据全部落到 `AI_Profiling_Tool/data/*.json`。
- 支持 Qwen2-7B 多个 reportId。
- 支持 trace/kernel/communication 缺失检查和 coverage 面板。

### M4. 验证与发布入口

- 加入 launch 页面入口。
- 做桌面/移动基础响应式验证。
- 完成导出 snapshot、深链 URL、回归测试。

---

## 11. 成功指标

| 指标 | 目标 |
|---|---|
| 首次定位 P0 问题节点时间 | 小于 30 秒 |
| 从问题节点找到 runtime evidence 时间 | 小于 2 分钟 |
| 报告 evidence 覆盖率 | MVP 示例中 P0/P1 节点覆盖率大于 80% |
| 人工跨视图跳转次数 | 从 4-6 次降低到 1-2 次 |
| 调优建议可复核率 | 每条 P0/P1 建议至少 1 条 raw evidence |
| 页面空白/崩溃率 | 缺文件场景 0 空白页 |

---

## 12. 风险与开放问题

### 12.1 风险

- 模型语义节点与 runtime operator 名称天然不一致，容易出现错误映射。
- 大规模 `trace_view.json` 直接渲染可能造成性能问题，需要聚合或虚拟化。
- cluster 数据和单卡数据字段不完全一致，需做适配层。
- AI report 的问题描述可能比 raw evidence 更完整，但不能替代原始数据。
- `MindStudioNext.html` 当前大量代码内嵌，直接复制会形成维护债务。

### 12.2 已确认决策

- MVP 优先支持 Qwen2-7B。
- MVP 解析真实 `trace_view.json`，现有 `SWIMLANE_DATA` / `FREE_ANALYSIS_DATA` 只作为迁移参考，不作为目标数据源。
- `GRAPH_PROBLEMS`、`QWEN7B_BASE_NODES`、`QWEN7B_NODE_INFO` 等从 HTML 内联迁出到 JSON。
- 页面保留内置 demo 数据，且 demo 数据也通过真实解析链路展示。

### 12.3 仍需确认

- `pattern/swimlane` 指的是当前 `swimlane-task` task bar pattern，还是需要抽象完整 swimlane viewport pattern？
- MVP 是否需要支持用户拖入本地 `trace_view.json`，还是先只从内置 JSON URL 加载？
- JSON schema 是否需要在 M1 就固化为独立 `schema/*.json` 文件，还是先在 loader 中做轻量校验？

---

## 13. 官方参考资料

- MindStudio Insight 8.3.0 Introduction: https://www.hiascend.com/document/detail/en/mindstudio/830/GUI_baseddevelopmenttool/MindStudioInsight/Insight_userguide_0002.html
- MindStudio Insight 8.3.0 Timeline GUI Description: https://www.hiascend.com/document/detail/en/mindstudio/830/GUI_baseddevelopmenttool/MindStudioInsight/Insight_userguide_0034.html
- MindStudio Insight 8.3.0 Timeline Basic Functions: https://www.hiascend.com/document/detail/en/mindstudio/830/GUI_baseddevelopmenttool/MindStudioInsight/Insight_userguide_0036.html
- Ascend PyTorch Profiler Timeline and Summary Data: https://www.hiascend.com/document/detail/en/mindstudio/700/TITools/Profiling/atlasprofiling_16_1149.html
- MindStudio Insight 8.3.0 Summary GUI Description: https://www.hiascend.com/document/detail/en/mindstudio/830/GUI_baseddevelopmenttool/MindStudioInsight/Insight_userguide_0049.html
- msprof-analyze Profile Data Analysis: https://www.hiascend.com/document/detail/en/mindstudio/700/quickstart/PTtraingquickstart/pttools_qucikstart_0011.html
- MindStudio Insight Operator fields and Click To Timeline: https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0039.html
- Ascend operator source hot map / `visualize_data.bin`: https://www.hiascend.com/document/detail/zh/mindstudio/70RC2/ODtools/Operatordevelopmenttools/atlasopdev_16_0088.html

## 14. 本地参考资料

- `Profiling_Insight_and_Tool/AI_Profiling_Tool/MindStudioNext.html`
- `graphviz/deepseek_v32_report_overlay_demo.html`
- `vendor/pto-design-system/patterns/model-graphviz/pattern.json`
- `vendor/pto-design-system/patterns/swimlane-task/pattern.json`

# 对比业务

最该先讲的一句是：做性能对比时，泳道图不是第一个被打开的东西，通常也不是主力。

真实工作流大致是：

1. 先过「可比性门禁」。确认两次运行的有效工作量（global batch、token/序列长度分布）、并行配置、精度与重计算、硬件/拓扑、软件栈、profiler 档位、warmup 和采集窗口一致。无法对齐时必须明确降级为「仅支持归一化趋势比较」，不能继续给出算子级显著性结论。
2. 再看汇总数字——step time、吞吐（samples/s、tokens/s）、MFU、E2E 时长。在工作量可比的前提下，这一步才能判断"改动是赚是亏"；动态序列场景优先看有效 tokens/s，E2E 还要区分稳态训练与编译、checkpoint、eval 等周期性成本。
3. 再看分类汇总和算子 Top-N 表。绝大多数团队做 A/B 对比其实是拿 profiler 导出的 CSV/summary 表在 Excel 或脚本里对的，因为可批量、可归档、可进 CI。
4. 只有当汇总表解释不了差异时，才打开泳道。

所以泳道对比真正不可替代的价值只有一个：回答"数字对得上但时间对不上"的问题——每个算子都没变慢，step 却变长了；总通信量没变，通信时间翻倍了。这类问题的答案全在时序结构里，表格看不出来。

这一点会直接改变对比功能的设计取向：它不该试图把汇总表能干的活再干一遍，而该把"时序结构差异"做到无可替代。我上一轮把三点讲得像并列的三个入口，实际它们的业务权重差很多。

## 多次训练任务对比（A/B）

用户嘴上说"对比性能"，实际在验证一个很具体的假设："我改的那件事，生效了吗、副作用是什么"。改动通常是明确的——开了某个融合、换了算子实现、调了并行策略、改了重计算配置。

1. 稳定区间的单 step 时长与构成
关键词是"稳定区间"和"构成"。计算/通信/空闲/host 下发四类的占比变化，决定接下来往哪查。这一层其实是汇总指标，不是泳道特长——但泳道对比必须以它为入口，否则用户会直接扎进算子列表里查错方向。

2. 算子级 diff，重点在"新增/消失"而不是"变快/变慢"
融合验证是这类对比的最高频用途，而融合的证据形态就是"N 条变 1 条"、算子名整个换掉。必须带执行次数——单次变快但次数翻倍是最常见的负优化。

3. 空隙与并行结构
gap、通信与计算的 overlap 是否丢失、关键路径是否换了一条。这条业务频率最低，但只有泳道能答，是这个视图存在的理由。

4. 差异要先过「噪声带」才算数

训练本身就抖，同一份代码跑两次、单 step 差 3~5% 是常态。所以在说"变快了 8%"之前，必须先回答一个前置问题："8% 超出两次运行各自的抖动了吗？"**噪声带就是这条判定线**：它必须同时使用基线和候选运行的多 step 分布估计，不能只看基线自身的 CV。

算法上有两个容易做错的地方，都踩过：

**CV 必须按 rank 分组后再算，取各 rank 的中位数。** 把所有 rank 的所有 step 混在一个数组里算变异系数，会把"慢卡造成的 rank 间系统性差异"也算进噪声——那不是噪声，正是要被检出的结论。混算出来的 CV 明显偏高（实测 7~10% vs 分组后 2~4%），噪声带会被撑宽到吞掉真实差异。

**判据要看两次运行“均值差”的标准误，而不是单次观测的标准差，也不是只拿基线 CV 乘一个固定系数。** 设两侧均值、标准差和有效样本数分别为 `μb/sb/nb`、`μc/sc/nc`，以基线为分母的相对差异可用 `SErel = √(sb²/nb + sc²/nc) / μb`，再用 Welch t 系数形成差异的置信区间。只有在两边方差和样本数近似相等、step 相互独立时，它才会退化成原来的 `√2 / √n` 近似。连续 step 若存在自相关，必须折算有效样本数或做 block bootstrap；如果展示中位数，也应使用 bootstrap 区间，不能套用均值标准误。

这套判据反过来对采样提了要求：**样本少时区间会很宽，只能得出「证据不足」，不能得出「没有变化」。** 十几个 step 可以作为演示和最低采集提示，但是否足够要由实际方差、自相关和目标最小可检测差异共同决定。

界面上必须区分三种结论，而不是统一显示「无显著变化」：
- **显著提升/回退**：置信区间不跨 0，且差异超过业务最小关注阈值；
- **实际等效**：整个置信区间都落在预先定义的等效区间内；
- **证据不足**：区间跨 0 或样本量不足，需要补采，不能按健康状态计分。

原来的 1.5% 下限可以保留为业务最小关注阈值，但它是产品决策线，不是统计置信区间；25% 上限也只能作为界面保护值，不能参与“是否显著”的计算。「显著差异任务 / 泳道」只统计第一类，差异健康度不得把「证据不足」等同于健康。

## 单任务多卡对比（rank 间）

问题完全不同："谁在拖后腿，它是真慢还是在等别人"。集合通信下是木桶效应，一张卡决定全局。

1. 慢卡识别与 rank 间离散度
看 max/median 比值和分布形态，不看均值。区分"一张卡离群"（降频、坏卡、邻居干扰、网络端口）和"整体抖动"（dataloader、存储 IO）。慢卡判定要两个条件 —— 跨 step 一致性（≥60% 的 step 里最晚进入通信）且 显著性（进入时刻极差 ≥ step 中位的 5%）。只看平均值必然能挑出一个「最晚」的 rank，那是抖动不是慢卡。

2. wait 与真实通信的拆分
这是多卡对比里业务价值最高、也最容易误判的一点。同一通信域内的 ring AllReduce，各 rank 的结束时间基本对齐（它本来就是同步点），所以看结束时间永远找不出谁慢；要看的是各 rank 进入通信算子的时刻，最晚进入的那张才是真凶，其余卡的长条其实是空等。工具不做这个拆分，用户会一路去优化那些"看起来通信很慢"的快卡。

3. 负载与任务构成不均
同名算子跨 rank 的耗时/shape 差异（切分不均、动态 shape、MoE 路由不均），以及 rank 特有任务（rank0 的 ckpt 保存、日志、数据预处理）。

## 方法的边界


伪差异是头号敌人。 训练本身就抖——同一份代码跑两次，单 step 差 3~5% 是常态。拿单个 step 做逐条对比，得到的差异里可能大半是噪声。所以任何有价值的对比都必须建立在多 step 统计（中位数/分位数/方差）之上，而不是"挑一个 step 摆两边"。泳道图天然是"看单个 step"的视图，这是它和对比业务之间的根本张力。

profiling 本身有扰动。 打开 timeline 采集会改变时间分布，尤其是 host 下发密集的场景。两次采集的开关档位不同，对比就没意义。

很多时候根本对不齐。 改了并行策略、改了 batch、换了模型结构之后,两边的任务序列完全不同名、不同数、不同长——逐条对齐失效。这时候唯一还成立的对比维度是归一化的构成比例和吞吐，不是泳道上的条与条。工具如果强行对齐，会给出误导性的"新增/消失"。

还有一类对比根本不是双跑并排：新跑 vs 历史基线数字（性能看护、CI 卡点）。这是频率最高的对比形态，但它的产物是一个阈值告警，不是一张泳道图——泳道只在告警之后被打开一次。

这三点合起来的结论是：泳道对比应该定位成归因和取证工具，入口来自汇总层的一个异常结论，而不是让用户从两张并排的泳道图开始自己找差异。

# 指标

## 指标看板为什么分四层

改造前的形态是 12 张同权重卡片平铺 + 一个「编辑」让用户自己挑 4 个显示。问题不在卡片本身，而在于这个信息架构把设计责任推给了用户——用户根本不知道该显示哪 4 个，而这本该是工具替他回答的。

分层的依据是：读一份 profiling 报告时，问题是有先后顺序的，每一层只回答其中一个，答完才轮到下一层。

| 层 | 回答的问题 | 内容 | 特点 |
|---|---|---|---|
| 结果层 | 这次跑多久、多快？ | 单步耗时中位 / P90、吞吐、端到端时长 | 绝对量，不着色 |
| 可信层 | 上面这些数字能信吗？ | step CV、有效 step 数、warmup、采集档位 | 口径脚注，超阈值降级告警 |
| 构成层 | 时间去哪了？ | 计算 / 未掩盖通信 / 调度空泡 / Host 下发 | 闭合到 100% 的堆叠条 |
| 结构层 | 结构上哪里不对？ | 关键路径占比、重叠率、PP bubble、rank 离散度、MFU、显存利用率 | 比率，带阈值着色 |

**改造前 12 张卡全是比率，答不了第一个问题**——"这次跑多久、多快"这种最朴素的数字只以自然语言躺在结论文案里，没法被对比、没法排序、没法进 CI。结果层就是补这一层。

三条约束是从这个顺序推出来的，不是各自单独定的规则：

**结果层不做阈值着色** —— 810 ms 是快是慢取决于模型和集群规模，没有普适健康区间，硬套阈值只会误导。判断留给结构层的比率和跨任务对比。

**可信层不与其他指标平级** —— step CV 的语义是"上面所有数字能不能信"，它是别的指标的前提而不是同类。所以它是常驻的口径脚注而不是第 13 张卡；CV > 10% 时整条加边框并写明「以下比率类指标基于抖动区间，仅供参考」。改造前它排在第 6 张、默认隐藏，等于把"这些数字可能全是噪声"这条警告藏了起来。

**构成层必须闭合** —— 改造前的 op_utilization / overlap_ratio / host_launch_gap_ratio / max_lane_idle 各自分母口径不同、彼此不能相加，用户没法回答"这 810 ms 里计算占多少"。构成层四项之和必须等于单步耗时，偏差 >2% 在标题行标「未闭合」，不让用户把一条不闭合的条当成完整的时间去向读。色板与 Timeline 页签泳道图例同源——同一语义跨页签不能有两种颜色。

另外两条实现层面的取舍：

**「编辑」只作用于结构层。** 前三层是回答固定问题的，不该让用户关掉；只有结构层的比率指标是"按需查看"的。编辑按钮因此从 section 标题行移到结构层标题行，作用域与实际行为一致。

**阈值按 taskType 取，不再全局硬编码。** 同一个「算子利用率 60%」阈值，训练场景合理，推理和 RL rollout 就会天天误报——推理本来就有大量 host 下发与小 batch 空隙，RL rollout 更是 host 串行主导。

结构层里 **MFU 与显存利用率是必备项**：前者是"算力到底用出来多少"的唯一口径，后者决定能不能加 batch、能不能关重计算，两者都是优化决策的直接输入。它们的分母（芯片峰值算力 / HBM 容量）落盘数据里没有、需要用户下拉选型号，这个不确定性由卡片自己的说明气泡承担，不构成把它们降级的理由。

## 这份 fixture 特意构造成一个反直觉的例子
32 卡的 step 耗时几乎完全一致（极差 0.12%,离散度卡是绿的）。只看 step 时长,你会得出"没有慢卡"的结论。

真正的信号在拆分里：

stage0–2 (rank 0/7/8/16)	stage3 (rank 24/31)
计算	8.29 s	12.70 s
通信等待	4.26 s	0.08 s
首次通信进入	8.31 s	12.72 s
stage3 在 12.72 s 才进通信,前面三级 8.31 s 就到了、然后空等 4.26 s。集合通信是同步点,所有卡被拉到同一个 16.20 s——这就是我前面说的「看结束时间永远找不出谁慢,要看进入时刻」。目前这个信息只以文字形式出现在 rank 卡的状态行里,把它做成可视对比是第 4 步「按 Rank 页签」的事。

另外 opStats 也一并补了（分类汇总 4 行 + Top 10,含 lm_head MatMulV2、MoE all-to-all、expert FFN、IndexPut 落 AI_CPU 等）,暂时没有消费方,等第 3 步算子页签落地就能直接用。

# 算子页签


总览页签栏新增「算子」（报告未出表时置灰）。三块内容：

分类汇总 — 按核类型/阶段的横条，配色与总览构成条、Timeline 图例同源。

Top-N 表 — 默认按总耗时降序（不是单次），表头可点换排序，带过滤框，dynamic 标警告色。

与关键问题双向联动 —— 这是相对 MindStudio 的差异化所在：不是「有一张算子表」，而是表里哪几行已经被诊断成问题。Top-N 每行匹配行动清单，命中的打「P0 · 已诊断」徽标、点击回总览选中该问题；问题详情的「影响」块下方给反向入口「在算子表中查看 N 个相关算子」。

