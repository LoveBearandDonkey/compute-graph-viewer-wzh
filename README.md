# PTO — 计算图工作台

PTO 是面向 Ascend NPU 算子开发、编译 Pass 分析、执行泳道和硬件数据路径理解的本地可视化工作台。项目主体是静态前端，优先使用 HTML、CSS 和原生 JavaScript；少数实验模块使用独立的 Vite 或 Node 子工程。

**线上主入口**：[launch-v2.html](https://yinyucheng0601.github.io/compute-graph-viewer/launch-v2.html)  
**Legacy 入口**：`launch.html`（停止维护，仅保留兼容访问）

---

## 快速开始

多数页面需要通过本地 HTTP 服务访问，避免 `file://` 下 fetch、ES 模块或 iframe 资源加载失败。

```bash
cd /Users/yin/pto
python3 -m http.server 8765
```

打开：

```text
http://127.0.0.1:8765/launch-v2.html
```

也可以使用 Node：

```bash
npx serve .
```

首次拉取仓库或设计系统版本发生变化后，初始化 submodule：

```bash
git submodule update --init --recursive
```

不要直接用 `file://` 打开需要 `fetch`、ES Module 或 iframe 的页面；统一从本地 HTTP 地址访问。

---

## 主入口

| 入口 | 文件 / 链接 | 用途 |
|------|-------------|------|
| 主启动台 | [launch-v2.html](https://yinyucheng0601.github.io/compute-graph-viewer/launch-v2.html) | 当前维护的 CANN PTO 项目总入口，聚合工作台、实验模块、白皮书和演示入口 |
| Legacy 启动台 | `launch.html` | 已停止维护，仅为历史链接保留兼容访问 |
| 演示页 | `low-fi/ppt-web.html` | 演示汇报入口 |
| 设计系统技能 | [pto-design-system](https://github.com/yinyucheng0601/pto-design-system) | PTO 设计系统专用技能仓库 |

---

## 核心模块

| 模块 | 入口 | 说明 |
|------|------|------|
| 950B 硬件路径工作台 | `ascend-950-workbench-demo/index.html` | 面向 Ascend 950B 的硬件路径、算子迁移和 tiling 执行理解 |
| AscendPort 迁移工作台 | `ascendport_migration-pangu/ascendport_migration_V3_MLA_pto.html` | CUDA Flash MLA 到 AscendC 的分阶段迁移、架构图、内存与代码联动 |
| A3/A5 差异解读 | `ascend-950-workbench-demo/feature_taxonomy.html` | A3 到 A5 算子迁移差异、分类和硬件联动解读 |
| Ascend A5 架构映射 | `ascend-hardware-map/index.html` | A3/A5 代际对比、数据搬运路径、通信指令和硬件流向地图 |
| A5 PMU 诊断工作台 | `pmu/06-a5-pmu-visualization-group2-loop.html` | PMU 数据、循环分组和泳道式性能诊断 |
| 训练任务监控 | `Profiling_Insight_and_Tool/training-run-twin-standalone/training-monitoring-v2.html` | 训练任务拉起、监控、对比、诊断与配置关系观测 |
| MC2 算子异常定位 | `Profiling_Insight_and_Tool/training-run-twin-standalone/mc2-incident-monitoring.html` | vLLM 推理下 MC2 融合算子 CCU mission 污染的双页定位链路（监控页定界 → 观测页根因与配置耦合） |
| TrainScope 盘古训练透视 | `pangu-moe-trainviz/index.html` | Pangu Pro MoE 训练正确性排障、时空透视与 rank 下钻 |
| Pass IR 计算图 | `pass-ir/index.html` | 编译 Pass 快照浏览、节点分组、语义染色和计算流锁定 |
| 模型 Profiling 报告叠加图 | `deepseek-v32-report-overlay/index.html` | 源码验证的完整模型架构与后端 Profiling、Timeline 局部覆盖层 |
| 昇腾融合算子推荐 | `op-fusion/index.html` / `deepseek-v32-report-V2/index.html` | 从模型图和性能证据生成融合候选与分析建议 |
| 算子支持矩阵 | `Ascend operator matrix/ascend-operator-matrix_V2.html` | 按模型、阶段、数据类型和硬件代际查询算子支持情况 |
| 算子入图工作台 | `op-graph-integration/index.html` | 算子入图流程、执行关系和图集成状态联动查看 |
| 整网到算子精度调试 | `precision-debugger/index.html` | 从整网精度首错定位到算子张量、指令和排布证据下钻 |
| 内存查看器 | `mem_viewer/index.html` | 计算图与 DDR/L1/L0/UB 内存层级联动的逐步执行视图 |
| 内存工作台 | `Memory-Visual/index.html` / `Memory-Visual/workspace.html` | 单算子片上内存、融合 workspace、生命周期和 GM 规划分析 |
| 泳道执行视图 | `swimlane/index.html` | AIC/AIV 任务泳道、目录导入、前后对比和任务下钻 |
| 图执行叠加原型 | `indexer-exec/index.html` | DAG、执行热度、核分配和诊断信息的叠加原型 |
| 模型算子层级架构图 | `model-architecture/index.html` | DeepSeek V3/V3.2 L1 到 L4 的多层级折叠图 |
| TorchVista / Graphviz 预览 | `graphviz/torchvista_graphviz_deepseek_v4.html` | DeepSeek V4 图结构预览 |
| 算子 IDE 助手 | `op-ide-assistant/index.html` / `op-ide-assistant-v2/index.html` | 面向算子开发的 IDE 辅助原型 |
| PTO 性能分析 | `pto-swimlane-profiler/index.html` | 性能数据导入、泳道分析和热点定位工作台 |
| 泳道性能工具 | `pypto-swimlane-perf-tool/index.html` | 泳道性能数据解析、统计和对比工具 |
| 智能性能分析 | `Profiling_Insight_and_Tool/AI_Profiling_Tool/index_v3.html` | Profiling 证据、内存快照、Timeline、泳道和多报告对比 |
| 源码流 | `source-flow/index.html` | 源码计算流实验入口 |
| 图原型实验室 | `graph-prototype-lab/index.html` | 通用图布局、方向切换、分组和检查器实验室 |
| 竞品分析 | `计算领域竞分/index.html` | CUDA / ROCm / Triton 等算子开发体验竞品分析 |

---

## 白皮书页面

| 白皮书 | 本地路径 | 线上预览 | 备注 |
|--------|----------|----------|------|
| PTO / PyPTO Hardware-Native Operator Toolchain 白皮书 | [pypto-toolchain-whitepaper/index.html](pypto-toolchain-whitepaper/index.html) | [GitHub Pages](https://yinyucheng0601.github.io/compute-graph-viewer/pypto-toolchain-whitepaper/) | 工具链总览 |
| Hardware-Native Systems：面向 Ascend NPU 的 AI 编译运行时栈白皮书 | [hw-native-sys/index.html](hw-native-sys/index.html) | [GitHub Pages](https://yinyucheng0601.github.io/compute-graph-viewer/hw-native-sys/) | 系统栈总览 |
| HNSW 白皮书：分层导航小世界图与向量检索工程 | [HNSW/HNSW-whitepaper.html](HNSW/HNSW-whitepaper.html) | [GitHub Pages](https://yinyucheng0601.github.io/compute-graph-viewer/HNSW/HNSW-whitepaper.html) | 图检索方法 |
| VLSI Placement 白皮书：布局算法如何实现芯片核舟记 | [vlsi-placement-whitepaper/index.html](vlsi-placement-whitepaper/index.html) | [GitHub Pages](https://yinyucheng0601.github.io/compute-graph-viewer/vlsi-placement-whitepaper/) | EDA 布局背景 |
| Ascend C Tiling 入门白皮书 | [tiling/tiling-whitepaper.html](tiling/tiling-whitepaper.html) | [GitHub Pages](https://yinyucheng0601.github.io/compute-graph-viewer/tiling/tiling-whitepaper.html) | Tiling 入门 |
| 已合并到 Ascend C Tiling 白皮书 | [tiling/real-operator-workflow-whitepaper.html](tiling/real-operator-workflow-whitepaper.html) | [GitHub Pages](https://yinyucheng0601.github.io/compute-graph-viewer/tiling/real-operator-workflow-whitepaper.html) | 合并提示页 |
| AI CPU 与 AI Core：算子开发初学者的产品视角白皮书 | [ai-cpu-aicore/whitepaper.html](ai-cpu-aicore/whitepaper.html) | [GitHub Pages](https://yinyucheng0601.github.io/compute-graph-viewer/ai-cpu-aicore/whitepaper.html) | 算子开发入门 |
| H-Anchor: 分层锚点 VLSI 布局算法白皮书 | [PycPlacer/pycplacer-whitepaper.html](PycPlacer/pycplacer-whitepaper.html) | [GitHub Pages](https://yinyucheng0601.github.io/compute-graph-viewer/PycPlacer/pycplacer-whitepaper.html) | 布局算法原型 |
| 大模型并行放置白皮书：从 rank 到 PP / TP / DP / CP / EP | [pangu-moe-trainviz/knowledge-whitepaper.html](pangu-moe-trainviz/knowledge-whitepaper.html) | [GitHub Pages](https://yinyucheng0601.github.io/compute-graph-viewer/pangu-moe-trainviz/knowledge-whitepaper.html) | 并行放置知识 |
| openPangu 并行通信可视化白皮书 | [pangu-moe-trainviz/communication-operator-visual-whitepaper.html](pangu-moe-trainviz/communication-operator-visual-whitepaper.html) | [GitHub Pages](https://yinyucheng0601.github.io/compute-graph-viewer/pangu-moe-trainviz/communication-operator-visual-whitepaper.html) | 通信可视化 |
| 白皮书生成规范 | [whitepaper.md](whitepaper.md) | - | 生成与维护说明 |

---

## 目录地图

```text
pto/
├── launch-v2.html                      # 当前维护的项目总启动页
├── launch.html                         # Legacy 启动页，停止维护
├── vendor/pto-design-system/           # 设计系统 submodule，运行时默认来源
├── js/                                 # Pass IR 共享解析、布局、渲染和导航逻辑
├── assets/                             # 启动台和演示图像资源
├── data/                               # 内置样本图数据
├── pass-ir/                            # Pass IR 计算图工作台
├── mem_viewer/                         # 内存层级与计算图联动视图
├── swimlane/                           # 执行泳道主模块
├── ascend-950-workbench-demo/           # 950B 硬件路径和迁移工作台
├── ascend-hardware-map/                # Ascend 数据搬运流向图
├── ascendport_migration-pangu/         # Flash MLA 到 AscendC 的迁移工作台
├── pmu/                                # A5 PMU 可视化原型
├── model-architecture/                 # 大模型算子层级架构图
├── deepseek-v32-report-overlay/        # DeepSeek V3.2 源码架构与 Profiling 覆盖图
├── precision-debugger/                 # 整网到算子的精度调试证据链
├── Memory-Visual/                      # 片上内存与融合 workspace 规划工作台
├── op-graph-integration/               # 算子入图与执行关系集成工作台
├── Profiling_Insight_and_Tool/         # 训练监控、任务管理和智能性能分析
├── graph-prototype-lab/                # 图布局实验室
├── op-ide-assistant*/                  # IDE 助手两版原型
├── pypto-swimlane-perf-tool/           # 泳道性能分析工具
├── hw-native-sys/                      # 硬件原生系统白皮书页面
├── HNSW/                               # HNSW 白皮书资料和页面
├── PycPlacer/                          # H-Anchor / PycPlacer 白皮书页面
├── vlsi-placement-whitepaper/          # VLSI 布局白皮书页面
└── 业务理解/                           # PRD、研究笔记、迁移方案和项目索引
```

---

## 模式库

复用图形模式以 `vendor/pto-design-system/patterns/patterns.json` 为准。

| 模式 | 路径 | 用途 |
|---------|------|------|
| swimlane-task-bar | `vendor/pto-design-system/patterns/swimlane-task/` | 泳道任务条 |
| memory-architecture-layout | `vendor/pto-design-system/patterns/memory-architecture/` | 内存架构层级图 |
| aic-core-object | `vendor/pto-design-system/patterns/aic-core-object/` | AIC 核心对象图形 |
| aiv-core-object | `vendor/pto-design-system/patterns/aiv-core-object/` | AIV 核心对象图形 |
| pass-ir-graph-node | `vendor/pto-design-system/patterns/pass-ir-graph-node/` | Pass IR 图节点 |
新增图形模式时保持 `pattern.html` / `pattern.css` / `pattern.js` / `pattern.json` 结构，并同步更新 `vendor/pto-design-system/patterns/patterns.json`。

---

## 子工程

根目录整体无统一构建流程。以下目录是独立子工程，进入各自目录后按本地 `package.json` 运行：

| 子工程 | 说明 |
|--------|------|
| `ai-for-design-open-slide/` | Open Slide 演示工程 |

---

## 协作流程

开始开发前先同步主线，并从最新 `main` 创建功能分支：

```bash
git fetch origin
git switch main
git pull --ff-only
git switch -c <feature-branch>
```

- 一个 PR 聚焦一个模块或一个明确目标，避免把部署配置夹带在无关改动中。
- 修改 `.github/workflows/`、`.gitmodules` 或 `vendor/pto-design-system` 时，在 PR 说明中单列影响范围和验证结果。
- 合并前检查目标页面、相对路径和 submodule 引用；合并后检查 Pages workflow，而不是重复提交同一个 PR。
- 工作区经常并行存在多个实验，提交和同步前先运行 `git status`，不要清理或覆盖无关改动。

---

## GitHub Pages 发布

仓库只使用一条 Pages 发布链路：

- `Settings → Pages → Source` 必须保持为 **GitHub Actions**，不要同时启用 `Deploy from a branch`。
- 唯一发布工作流是 `.github/workflows/static.yml`；它在 push 到 `main` 后自动运行，也支持 `workflow_dispatch` 手动触发。
- workflow 使用 `submodules: recursive`，确保 `vendor/pto-design-system` 一起进入发布产物。
- workflow 使用 Node.js 24 对应的官方 Action 主版本：`actions/checkout@v7`、`actions/configure-pages@v6`、`actions/upload-pages-artifact@v5`。
- Pages 设置页已经存在 workflow 时，不要点击 Suggested workflows 下的 `Configure`，避免创建第二份部署文件。

发布验证顺序：

1. 在 Actions 中确认 `Deploy static content to Pages` 为绿色 `Success`。
2. 打开 [线上主启动台](https://yinyucheng0601.github.io/compute-graph-viewer/launch-v2.html)。
3. 检查本次修改页面以及设计系统静态资源是否正常加载。

---

## 维护规则

- 新页面优先通过 `launch-v2.html` 暴露入口；`launch.html` 已停止维护，不再新增卡片或功能。若只是实验或归档，放在 `archive/` 或对应模块目录内。
- 设计系统规范以 [pto-design-system](https://github.com/yinyucheng0601/pto-design-system) 为准，README 不再维护展开说明。
- 页面运行时默认引用 `vendor/pto-design-system/...`；`design-system-share/`、根目录 `tokens/`、`css/`、`patterns/` 只在需要兼容旧工具时由同步脚本临时生成。
- 复杂图形先判断是否应沉淀为 `patterns/`，避免在页面里散落重复的 SVG、Canvas 或 DOM 图形实现。
- 白皮书和研究资料优先放在明确模块目录或 `业务理解/`，避免根目录继续堆积临时文件。
- 当前工作区包含较多历史原型和迁移中目录，修改前先看 `git status`，不要顺手清理无关文件。

---

## 版本日志

详见 `CHANGELOG.md`。

**维护者**：Yin Yucheng
