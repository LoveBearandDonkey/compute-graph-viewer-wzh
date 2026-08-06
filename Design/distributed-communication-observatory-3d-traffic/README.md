# Distributed Communication Observatory · 3D Rank Deck + Floor Sankey Experiment

首个可交互产品界面实现 `spec.md` 中的 `MoE Route & Capacity` Lens。当前数据是明确标注的 Mock Scenario，用于验证产品任务闭环和多视图联动，不声称复现真实 Run。

## 黄金任务流

1. 在 openPangu-2.0-Flash 的 128-Rank 全局范围中显示 16 台服务器；
2. 默认展开中央区域 Server 05 上的 8 张卡：R40–R47，作为一个 EP8 通信组；
3. 保留原有 3D Rank 卡体，只在 Server 05 的 ground floor 上铺设二维 Rank→Rank Ribbon 贴图；
4. 通过 Timeline 与 Inspector 核对 Event、Overlap、Wait 和证据来源。

## Pattern 消费

- `ide-frame`：使用标准 standalone 顶栏、Activity Rail、Inspector、可折叠 Bottom Dock 与 Status Strip；右上角提供深浅主题、中英文、Run Info、Timeline 和 Inspector 开关；
- `workbench-shell`：提供主视图 / Timeline 纵向 Resize，以及主视图 / Inspector 横向 Resize；
- `communication-traffic-sankey`：在实验副本中仅作为原始事件聚合控制器，不再显示其 AIC/AIV 子层级；
- `model-parallel-rank-deck`：复用 openPangu 模型载荷与 Rank Manifest；当前使用 8×4 全局排布展示 32 Rank，并通过 `focusCommunicationGroup()` 在不触发并行分组的情况下聚焦一个通信组；
- `model-parallel-rank-deck`：保留 3D Rank 卡体、相机、拖拽与缩放，并通过共享 Floor Traffic API 在 ground plane 上绘制二维 CanvasTexture；
- `rank-floor-sankey`：登记并预览相同的 Rank→Rank / Local Token 语义，供纯二维页面复用；
- `hierarchical-timeline`：负责 Timeline 的固定行头、层级展开、1×–64× 缩放、Profile Window 与事件选择；本 Lens 仅注入 `Layer → Phase → Collective → Rank → Task` 数据；
- `swimlane-task`：作为 `hierarchical-timeline` 的底层绘制原语，负责 Event Task Bar 与 Tooltip，产品中不重写条形编码。

3D Rank Deck + Floor Sankey 回答“当前通信组是谁、组内流量流向哪里、多少”，Timeline 回答“何时、顺序与因果”，Inspector 回答“证据等级、限制与下一步”。

Run、Step、执行位置、并行方案、Baseline 和 Mock 来源收进右上角 Info 弹层，不常驻占用画布。阶段 Tabs 位于 Rank Traffic 标题栏；诊断结论位于 Inspector，3D Rank 卡与地面二维 Ribbon 构成主体视觉。

## 当前场景

- Run：`pangu-flash-train-0421`；
- Step：`18420`；
- Layer：MoE Layer 27；
- 全局范围：128 Rank；显式放置为 16 Servers × 8 Cards；
- 活动通信组：Server 05 上的 EP8，成员为 R40–R47；
- 默认窗口：115–152 ms；
- 主结论：R42 E64/E65 Router / Dispatch Load Skew，派生 4.8 ms 暴露等待；
- Artifact：Router Count、Dispatch Event、Wait Chain 均为 Mock；
- Physical Counter、真实 Task/Core 映射未提供；本实验不展示 AIC/AIV 归因，也不输出链路饱和或 Core 根因。
- 128-Rank 放置是显式 Demo 拓扑，不声明为 openPangu 官方训练部署；模型结构来自设计系统中已核源码的 openPangu Preset。

## 运行

从仓库根目录启动 HTTP Server 后访问：

`http://127.0.0.1:8773/Design/distributed-communication-observatory-3d-traffic/`

## 验收重点

- 首屏是否只有一个结论和一个主视觉；
- Rank→Rank 流量带与 Wait 是否能形成证据闭环；
- 3D 卡体、8-Card Floor 与二维 Ribbon 是否保持清晰的物理层级；
- Window、Selection、Expansion、Phase 是否保持不同语义；
- 页面是否在 3–5 分钟内支持“结论 → 证据 → 动作 → 验证计划”。
