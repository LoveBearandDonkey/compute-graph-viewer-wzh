# Candidate Population Lineage

用于展示候选集合在多个阶段之间持续发散、收敛并形成 shortlist 的共享 SVG 组件。

## 加载

页面需要先加载 PTO token 链，再加载组件资源：

```html
<link rel="stylesheet" href="/pto-design-system/tokens/foundation.css">
<link rel="stylesheet" href="/pto-design-system/tokens/semantic.css">
<link rel="stylesheet" href="/pto-design-system/tokens/components.css">
<link rel="stylesheet" href="/pto_compute-graph-viewer/components/candidate-population-lineage/pattern.css">

<div id="lineageHost"></div>

<script src="/pto_compute-graph-viewer/components/candidate-population-lineage/pattern.js"></script>
```

组件不设置 `data-theme`，会继承调用页面的 PTO 语义 token，因此自动跟随 light / dark mode。

## 数据配置

发散或收敛属于当前阶段通往下一阶段的操作，所以配置在当前阶段的 `next` 字段中。最终阶段不配置 `next`。

```js
const controller = window.PtoCandidatePopulationLineage.render(
  document.getElementById('lineageHost'),
  {
    data: {
      title: 'Candidate population lineage',
      stages: [
        {
          id: 'seed',
          label: 'SEED',
          kind: 'partial plan',
          count: 80,
          next: {
            id: 'expand-paths',
            type: 'diverge',
            branches: [
              { label: 'Path A · 80', count: 80, tone: 1 },
              { label: 'Path B · 80', count: 80, tone: 2 }
            ]
          }
        },
        {
          id: 'paths',
          label: 'PATHS',
          kind: 'realization',
          count: 160,
          segments: [80, 80],
          next: {
            id: 'filter-contract',
            type: 'converge',
            rejected: [
              { label: 'Contract · 30', count: 30 },
              { label: 'Alignment · 20', count: 20 }
            ]
          }
        },
        {
          id: 'shortlist',
          label: 'SHORTLIST',
          kind: 'decision',
          count: 110
        }
      ]
    },
    initialSelection: 'shortlist',
    onSelect(selection) {
      // selection.kind: "stage" | "transition"
      // 详情面板、路由和业务状态由调用页面处理。
      renderProductDetail(selection);
    }
  }
);
```

## 数量守恒

- `diverge`：下一阶段数量不得小于当前阶段；`branches.count` 总和必须等于下一阶段数量。
- `converge`：下一阶段数量不得大于当前阶段；`rejected.count` 总和必须等于当前阶段数量减去下一阶段数量。
- 配置 `segments` 时，各段数量之和必须等于当前阶段数量。

组件发现不守恒的数据时会抛出包含阶段或转换 ID 的错误。

## 控制器

```js
controller.setData(nextData);
controller.select('stage-id', { emit: false });
controller.clearSelection();
controller.resize();
controller.getSelection();
controller.getData();
controller.destroy();
```

默认最多绘制 240 个候选点。数量更大时组件自动聚合点阵，但阶段标题中的候选总数保持真实；可通过 `maxVisibleCandidates` 调整阈值。

完整契约见 [`pattern.json`](./pattern.json)，独立预览见 [`pattern.html`](./pattern.html)。
