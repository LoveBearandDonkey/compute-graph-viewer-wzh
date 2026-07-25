/*
  指标计算层 —— 所有视图与规则引擎共享同一份派生数据。
  ------------------------------------------------------------------
  规划文档 §5「可解释」：任何结论都能溯源到原始数据项。因此这里只做
  聚合，不做推断；每个聚合量都保留贡献它的 allocation / event 引用。

  两条关键曲线（规划文档 §4.3-3 占用水位）：
    reserved —— TPipe::InitBuffer 的静态预留，编译期就决定是否超限；
    live     —— 某一时刻真正被张量持有的字节，反映复用空间。
  两者之差就是「预留了但没在用」的浪费，是场景 1/2 的核心读数。
*/
(function registerMemVizMetrics(global) {
  'use strict';

  function coversTick(interval, tick) {
    return tick >= interval.start && tick < interval.end;
  }

  function liveAt(alloc, tick) {
    for (let i = 0; i < alloc.intervals.length; i += 1) {
      if (coversTick(alloc.intervals[i], tick)) return true;
    }
    return false;
  }

  function liveSpan(alloc) {
    if (!alloc.intervals.length) return null;
    let start = Infinity;
    let end = -Infinity;
    alloc.intervals.forEach((interval) => {
      start = Math.min(start, interval.start);
      end = Math.max(end, interval.end);
    });
    return { start, end };
  }

  function intervalsOverlap(a, b) {
    return a.start < b.end && b.start < a.end;
  }

  function allocsOverlapInTime(a, b) {
    return a.intervals.some((x) => b.intervals.some((y) => intervalsOverlap(x, y)));
  }

  function allocsOverlapInSpace(a, b) {
    return a.offset < b.offset + b.size && b.offset < a.offset + a.size;
  }

  function compute(run, chip) {
    const ticks = run.totalTicks;
    const regions = chip.regions.map((spec) => {
      const allocations = run.allocations.filter((a) => a.region === spec.id);
      const reserved = allocations.reduce((max, a) => Math.max(max, a.offset + a.size), 0);
      const padding = allocations.reduce((sum, a) => sum + (a.size - a.dataBytes), 0);

      const series = new Float64Array(ticks + 1);
      for (let t = 0; t <= ticks; t += 1) {
        let bytes = 0;
        allocations.forEach((a) => { if (liveAt(a, t)) bytes += a.size; });
        series[t] = bytes;
      }
      let peakLive = 0;
      let peakTick = 0;
      for (let t = 0; t <= ticks; t += 1) {
        if (series[t] > peakLive) { peakLive = series[t]; peakTick = t; }
      }
      const topAtPeak = allocations
        .filter((a) => liveAt(a, peakTick))
        .sort((x, y) => y.size - x.size)
        .slice(0, 5)
        .map((a) => ({ alloc: a, bytes: a.size }));

      return {
        id: spec.id,
        label: spec.label,
        scope: spec.scope,
        owner: spec.owner,
        accent: spec.accent,
        align: spec.align,
        banks: spec.banks,
        note: spec.note,
        capacity: spec.capacity,
        allocations,
        reserved,
        reservedRatio: reserved / spec.capacity,
        padding,
        series,
        peakLive,
        peakTick,
        liveRatio: peakLive / spec.capacity,
        idleReserved: Math.max(0, reserved - peakLive),
        topAtPeak,
      };
    });

    const pipes = chip.pipes.map((spec) => {
      const events = run.events.filter((e) => e.pipe === spec.id);
      const busy = events.reduce((sum, e) => sum + e.dur, 0);
      const span = events.length ? events[events.length - 1].end - events[0].t : 0;
      const idle = Math.max(0, span - busy);
      const bytes = events.reduce((sum, e) => sum + (e.bytes || 0), 0);
      return {
        id: spec.id,
        label: spec.label,
        desc: spec.desc,
        kind: spec.kind,
        events,
        busy,
        span,
        idle,
        idleRatio: span ? idle / span : 0,
        occupancy: ticks ? busy / ticks : 0,
        bytes,
      };
    });

    const moveEvents = run.events.filter((e) => e.type === 'copy_in' || e.type === 'copy_out');
    const tailEvent = run.events.find((e) => e.tailRows != null);
    const tilesTotal = run.tiling.tileNum * run.tiling.tileM;
    const tailWasteRows = run.tiling.hasTail ? run.tiling.tileM - run.tiling.tailM : 0;

    return {
      ticks,
      regions,
      regionById: Object.fromEntries(regions.map((r) => [r.id, r])),
      pipes,
      pipeById: Object.fromEntries(pipes.map((p) => [p.id, p])),
      totals: {
        moveCount: moveEvents.length,
        moveBytes: moveEvents.reduce((sum, e) => sum + (e.bytes || 0), 0),
        tailWasteRows,
        tailWasteRatio: tilesTotal ? tailWasteRows / tilesTotal : 0,
        tailEvent,
        criticalPipe: pipes.slice().sort((a, b) => b.busy - a.busy)[0] || null,
      },
    };
  }

  global.MemVizMetrics = {
    compute,
    liveAt,
    liveSpan,
    intervalsOverlap,
    allocsOverlapInTime,
    allocsOverlapInSpace,
  };
})(window);
