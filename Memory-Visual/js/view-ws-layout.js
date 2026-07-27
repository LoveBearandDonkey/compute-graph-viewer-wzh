/*
  视图 —— GM 布局与复用（场景 6）
  ------------------------------------------------------------------
  整块复用设计系统的 patterns/memory-reuse-viewer：它已经 owns「地址 × 时间」的
  canvas、复用连线、峰值读数、源码/CCE 片段。本文件只做数据契约翻译。

  按该 pattern 的 forbiddenOverrides，不在这里另写同一张生命周期图 ——
  「当前布局 / 复用后布局」的对比因此做成**同一个实例换 data**，
  而不是并排两三个实例：那个组件自带工具栏、buffer 选择器、峰值读数与源码面板，
  并排塞进一个 pane 只会挤成一团，而且「理论下界」本身不是一份可实现的布局、
  不该用布局图去画（它在主视图里以参考线呈现）。

  时间轴单位是子计算序：allocTick = 产出子计算，freeTick = 最后消费子计算 + 1。
*/
(function registerMemVizWsLayoutView(global) {
  'use strict';

  const PLANNER = global.MemVizWorkspacePlanner;
  const SRC = global.MemVizFusionSource;

  function kindOf(tensor) {
    if (tensor.role !== 'workspace') return 'resident';
    if (tensor.manualReuseOf) return 'loop';
    return 'temp';
  }

  /** 中间格式 → memory-reuse-viewer 数据契约 */
  function toViewerData(run, plan, layout, capacity) {
    const tensors = run.tensors.filter((t) => !t.onChip && !t.aliasOf && layout[t.id] != null);

    const reusedBy = new Map();
    tensors.forEach((t) => {
      if (!t.manualReuseOf) return;
      const list = reusedBy.get(t.manualReuseOf) || [];
      list.push(t.id);
      reusedBy.set(t.manualReuseOf, list);
    });

    return {
      kernel: `${run.kernel.name} · ${run.label}`,
      ticks: run.subgraphs.length,
      sourceFiles: SRC.files.map((f) => ({ path: f.path, language: f.language, text: f.text })),
      buffers: [{ name: 'GM', capacity }],
      tensors: tensors.map((t) => ({
        id: t.id,
        name: t.name,
        buffer: 'GM',
        offset: layout[t.id],
        size: t.size,
        allocTick: t.live.start,
        freeTick: t.live.end + 1,
        kind: kindOf(t),
        reuseOf: t.manualReuseOf || null,
        reusedBy: reusedBy.get(t.id) || [],
        srcFile: t.src.file,
        srcLineStart: Math.max(0, t.src.hotLine - 2),
        srcLineEnd: t.src.hotLine + 3,
        srcHotLine: t.src.hotLine,
        code: t.code,
        cce: `// ${t.name} @GM  role=${t.role}  scope=${t.blockScope}\n`
          + `// live ${t.producer || '-'} → ${t.consumers.join(', ') || '-'}`,
      })),
    };
  }

  function create(container, options = {}) {
    let viewer = null;
    let signature = null;

    return {
      /**
       * mode: 'current' 用候选自己的布局；'packed' 用规划器排出来的最紧布局。
       * 两者共用同一个纵轴上界（取两者的较大值），否则视觉上看不出高度差。
       */
      update({ run, plan, mode, selectedTensorId }) {
        const layout = mode === 'packed' ? PLANNER.layoutOf(run) : run.layout;
        const capacity = Math.max(plan.current, plan.packed);
        const key = `${run.id}|${mode}`;
        if (key !== signature) {
          viewer?.destroy?.();
          container.innerHTML = '';
          viewer = global.PtoMemoryReuseViewer.render(
            container,
            toViewerData(run, plan, layout, capacity),
            { initialBuffer: 'GM' },
          );
          signature = key;
          options.onMounted?.(viewer);
        }
        if (selectedTensorId) viewer?.selectTensor?.(selectedTensorId);
      },
      redraw() { viewer?.resize?.(); },
      destroy() {
        viewer?.destroy?.();
        viewer = null;
        signature = null;
      },
    };
  }

  global.MemVizWsLayoutView = { create, toViewerData };
})(window);
