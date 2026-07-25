/*
  视图 B —— 生命周期与复用（规划文档 §4.3-2）
  ------------------------------------------------------------------
  这一屏直接消费设计系统的 patterns/memory-reuse-viewer：它已经owns
  「buffer 地址 × kernel tick」的 canvas、复用连线、峰值读数、源码/CCE 片段。
  本文件只做一件事：把工具的中间格式翻译成该 pattern 的数据契约。
  按 pattern.json 的 forbiddenOverrides，不在这里另写同一张生命周期图。
*/
(function registerMemVizLifetimeView(global) {
  'use strict';

  const MET = global.MemVizMetrics;
  const SRC = global.MemVizKernelSource;

  function kindOf(alloc) {
    if (alloc.persistent) return 'resident';
    if (alloc.bufferNum > 1) return 'loop';
    return 'temp';
  }

  /** 中间格式 → memory-reuse-viewer 数据契约 */
  function toViewerData(run, chip) {
    const coreRegions = chip.regions.filter((r) => r.scope === 'core');
    const coreIds = new Set(coreRegions.map((r) => r.id));
    const allocations = run.allocations.filter((a) => coreIds.has(a.region) && a.intervals.length);

    const reusedBy = new Map();
    allocations.forEach((a) => {
      if (!a.reuseOf) return;
      const list = reusedBy.get(a.reuseOf) || [];
      list.push(a.id);
      reusedBy.set(a.reuseOf, list);
    });

    return {
      kernel: `${run.kernel.name} · ${run.label}`,
      ticks: run.totalTicks,
      sourceFiles: [{ path: SRC.path, language: SRC.language, text: SRC.text }],
      buffers: coreRegions
        .filter((r) => allocations.some((a) => a.region === r.id))
        .map((r) => ({ name: r.id, capacity: r.capacity })),
      tensors: allocations.map((a) => {
        const span = MET.liveSpan(a);
        return {
          id: a.id,
          name: a.name,
          buffer: a.region,
          offset: a.offset,
          size: a.size,
          allocTick: span.start,
          freeTick: span.end,
          kind: kindOf(a),
          reuseOf: a.reuseOf,
          reusedBy: reusedBy.get(a.id) || [],
          srcFile: SRC.path,
          srcLineStart: Math.max(0, a.src.hotLine - 2),
          srcLineEnd: a.src.hotLine + 2,
          srcHotLine: a.src.hotLine,
          code: a.code,
          cce: a.cce,
        };
      }),
    };
  }

  function create(container, options = {}) {
    let viewer = null;
    let currentRunId = null;

    return {
      update({ run, chip, selectedId }) {
        if (run.id !== currentRunId) {
          viewer?.destroy?.();
          container.innerHTML = '';
          viewer = global.PtoMemoryReuseViewer.render(container, toViewerData(run, chip), {
            initialBuffer: 'UB',
          });
          currentRunId = run.id;
          options.onMounted?.(viewer);
        }
        if (selectedId) viewer?.selectTensor?.(selectedId);
      },
      redraw() { viewer?.resize?.(); },
      destroy() {
        viewer?.destroy?.();
        viewer = null;
        currentRunId = null;
      },
    };
  }

  global.MemVizLifetimeView = { create, toViewerData };
})(window);
