# PTO Tiling

Workspace for PTO tiling visualization experiments and references.
PTO 贴图可视化实验与参考工作的空间。

## References

- https://github.com/Deep-Learning-Profiling-Tools/triton-viz
- https://github.com/gpu-mode/triton-puzzles

参考资料

- https://github.com/Deep-Learning-Profiling-Tools/triton-viz
- https://github.com/gpu-mode/triton-puzzles

## Structure

- `docs/` - notes, PRDs, integration plans, and research summaries.
- `src/` - prototype code for tiling analysis or visualization.
- `data/` - sample traces, tiling metadata, and generated fixtures.
- `assets/` - screenshots, diagrams, and static visual assets.

结构

- `docs/` - 笔记、产品需求文档、集成方案和研究总结。
- `src/` - 用于贴图分析或可视化的原型代码。
- `data/` - 示例轨迹、贴图元数据和生成的固定数据。
- `assets/` - 截图、图表和静态视觉资源。

## Design-System Rule

Any UI built in this module should consume the PTO shared tokens and components first. Do not add a private visual style system here without a preview and explicit approval.

设计系统规则

本模块中构建的任何 UI 应首先使用 PTO 共享令牌和组件。未经预览和明确批准，不要在此处新增私有视觉样式体系。

## Conv + Bias + ReLU Code Recovery Progress

- Stage 1 — Workbench MVP: **completed**
- Stage 2 — Tensor Code Recovery: **completed**
- Stage 3 — Hardware Participation: **next**
- Stage 4 — Execution Dock enrichment: not started
- Stage 5 — Validation and delivery: not started

Current progress: **2/5 stages completed**.

Conv + Bias + ReLU 代码恢复进度

- 阶段 1 — 工作台 MVP：**已完成**
- 阶段 2 — 张量代码恢复：**已完成**
- 阶段 3 — 硬件参与：**下一个**
- 阶段 4 — 执行码头增强：未开始
- 阶段 5 — 验证与交付：未开始

当前进度：**已完成 2/5 阶段**。

Detailed scope and evidence rules:

- `spec.md`
- `Conv + Bias + ReLU Code Recovery 详细实施计划.md`

详细范围与验收规则：

- `spec.md`
- `Conv + Bias + ReLU Code Recovery 详细实施计划.md`

## Local Preview

This page loads JSON fixtures and Ascend C source through `fetch()`. It cannot be opened through `file://`.

From `/Users/songchenfei/Documents/ascend c`, run:

```bash
python3 -m http.server 4180 --bind 127.0.0.1
```

Then open:

```text
http://127.0.0.1:4180/pto_compute-graph-viewer/code%20recovery/index.html
```

本地预览

此页面通过 `fetch()` 加载 JSON 固定数据和 Ascend C 源码，不能通过 `file://` 打开。

在 `/Users/songchenfei/Documents/ascend c` 下运行：

```bash
python3 -m http.server 4180 --bind 127.0.0.1
```

然后打开：

```text
http://127.0.0.1:4180/pto_compute-graph-viewer/code%20recovery/index.html
```
