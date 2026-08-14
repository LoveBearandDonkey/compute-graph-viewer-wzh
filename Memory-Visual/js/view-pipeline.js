/*
  视图 C —— 流水 × 内存 联合时序（规划文档 §4.3-4）
  ------------------------------------------------------------------
  上半是 MTE1/MTE2/MTE3 + Cube/FixPipe/Vector 六条泳道，下半是同一时间轴上的
  Buffer 生命周期空间图（横轴 cycle，纵轴地址与大小）。判断 double buffer 到底有没有生效，
  靠的就是这两半的对齐关系：
  如果搬运泳道的色块和计算泳道的色块在时间上错不开，占用曲线又始终贴着单份
  buffer 的高度，那 ping-pong 就是没生效。

  任务条一律走 patterns/swimlane-task 的 drawTaskBar / createTaskColormap /
  initHoverTooltip，不在本页重画条形与提示框。
*/
(function registerMemVizPipelineView(global) {
  'use strict';

  const KIT = global.MemVizCanvasKit;
  const F = global.MemVizFormat;

  const PAD_L = 92;
  const PAD_R = 18;
  const AXIS_H = 26;
  const LANE_H = 30;
  const BAR_H = 18;
  const CHART_H = 220;
  const CHART_GAP = 28;

  function withAlpha(hex, alpha) {
    const value = String(hex || '#888').replace('#', '');
    const num = parseInt(value.length === 3 ? value.split('').map((c) => c + c).join('') : value, 16);
    return `rgba(${(num >> 16) & 255}, ${(num >> 8) & 255}, ${num & 255}, ${alpha})`;
  }

  function create(container, options = {}) {
    const SW = global.PtoSwimlaneTaskPattern;
    const colormap = SW.createTaskColormap();

    const canvas = document.createElement('canvas');
    canvas.className = 'mv-canvas';
    container.appendChild(canvas);

    let state = null;
    let hitboxes = [];

    function hitAt(event) {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      return hitboxes.slice().reverse().find((b) => x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h) || null;
    }

    // 悬浮提示复用 swimlane-task 的 tooltip 外壳，只替换内容格式（getTooltipHtml）
    const hover = SW.initHoverTooltip({
      root: canvas,
      targets: [canvas],
      appendTo: container,
      bounds: container,
      getTask: (target, event) => hitAt(event)?.task || null,
      getTooltipHtml: (task) => {
        if (task.allocation) {
          const a = task.allocation;
          const interval = task.interval;
          const reuse = a.reuseOf ? state.allocById.get(a.reuseOf)?.name || a.reuseOf : '';
          return `
            <div class="mv-tip__title">${F.escapeHtml(a.name)}</div>
            <div class="mv-tip__row"><span>生命周期</span><b>${F.tick(interval.start)} – ${F.tick(interval.end)}</b></div>
            <div class="mv-tip__row"><span>内存大小</span><b>${F.bytes(a.size)}</b></div>
            <div class="mv-tip__row"><span>地址区间</span><b>${F.bytes(a.offset)} – ${F.bytes(a.offset + a.size)}</b></div>
            ${reuse ? `<div class="mv-tip__row"><span>复用地址</span><b>${F.escapeHtml(reuse)}</b></div>` : ''}
            <div class="mv-tip__src">${F.escapeHtml(a.src.file)}:${(a.src.hotLine || 0) + 1}</div>
          `;
        }
        const e = task.event;
        const writes = e.writes.map((id) => state.allocById.get(id)?.name).filter(Boolean).join(', ');
        const reads = e.reads.map((id) => state.allocById.get(id)?.name).filter(Boolean).join(', ');
        return `
          <div class="mv-tip__title">${F.escapeHtml(e.label)}</div>
          <div class="mv-tip__row"><span>流水</span><b>${e.pipe}</b></div>
          <div class="mv-tip__row"><span>区间</span><b>${F.tick(e.t)} – ${F.tick(e.end)}（${e.dur} cycle）</b></div>
          ${e.bytes ? `<div class="mv-tip__row"><span>搬运</span><b>${F.bytes(e.bytes)}</b></div>` : ''}
          ${writes ? `<div class="mv-tip__row"><span>写</span><b>${F.escapeHtml(writes)}</b></div>` : ''}
          ${reads ? `<div class="mv-tip__row"><span>读</span><b>${F.escapeHtml(reads)}</b></div>` : ''}
          ${e.gap > 0 && e.blockedBy
            ? `<div class="mv-tip__row is-bad"><span>等待</span><b>${e.gap} cycle · ${F.escapeHtml(state.allocById.get(e.blockedBy)?.name || '')} 未释放</b></div>`
            : ''}
          <div class="mv-tip__src">${F.escapeHtml(state.run.kernel.source)}:${(e.srcLine || 0) + 1}</div>
        `;
      },
    });

    function draw() {
      if (!state) return;
      const {
        run, metrics, tick, focusRegionId, highlightEventIds, selectedEventId,
        highlightIds, conflictIds, selectedAllocId,
      } = state;
      const pipes = metrics.pipes;
      const cssWidth = container.clientWidth;
      const cssHeight = AXIS_H + pipes.length * LANE_H + CHART_GAP + CHART_H + AXIS_H;
      canvas.style.height = `${cssHeight}px`;
      const { ctx, width } = KIT.fitCanvas(canvas, cssWidth, cssHeight);
      const T = KIT.tokens(container);
      const plotW = Math.max(120, width - PAD_L - PAD_R);
      const total = Math.max(1, run.totalTicks);
      const xOf = (t) => PAD_L + (t / total) * plotW;
      hitboxes = [];

      // ---- 时间刻度 ----
      const step = Math.max(20, Math.round(total / 10 / 10) * 10);
      ctx.textBaseline = 'alphabetic';
      ctx.font = `500 10px ${T['font-mono']}`;
      for (let t = 0; t <= total; t += step) {
        const x = xOf(t);
        ctx.strokeStyle = T['border-subtle'];
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, AXIS_H - 6);
        ctx.lineTo(x, cssHeight - AXIS_H);
        ctx.stroke();
        ctx.fillStyle = T['foreground-muted'];
        ctx.fillText(`#${t}`, x + 3, 13);
      }

      // ---- 泳道 ----
      pipes.forEach((pipe, index) => {
        const laneTop = AXIS_H + index * LANE_H;
        const barY = laneTop + (LANE_H - BAR_H) / 2;

        ctx.font = `600 11px ${T['font-sans']}`;
        ctx.fillStyle = T['foreground-secondary'];
        ctx.fillText(pipe.label, 0, barY + BAR_H / 2 + 4);
        ctx.font = `500 9px ${T['font-mono']}`;
        ctx.fillStyle = T['foreground-disabled'];
        ctx.fillText(`${F.pct(pipe.occupancy, 0)}`, 0, barY + BAR_H / 2 + 15);

        ctx.fillStyle = T['surface-2'];
        KIT.roundRect(ctx, PAD_L, barY, plotW, BAR_H, 3);
        ctx.fill();

        const base = colormap.colorForLaneKind(pipe.kind);
        pipe.events.forEach((event, order) => {
          // 等待空洞：可归因的用警示斜纹，纯启动延迟用中性斜纹
          if (order > 0 && event.gap > 0) {
            const gx = xOf(event.t - event.gap);
            KIT.hatch(ctx, gx, barY + 2, xOf(event.t) - gx, BAR_H - 4,
              event.blockedBy ? withAlpha('#FFAA3B', 0.75) : withAlpha('#FFFFFF', 0.1), 4);
          }
          const x = xOf(event.t);
          const w = Math.max(2, xOf(event.end) - x);
          const task = {
            event,
            label: event.label,
            displayName: event.label,
            laneKind: pipe.kind,
            laneId: pipe.id,
            totalCycle: event.dur,
          };
          SW.drawTaskBar(ctx, {
            task,
            x,
            y: barY,
            width: w,
            height: BAR_H,
            baseColor: base,
            fontFamily: T['font-sans'],
            isSelected: event.id === selectedEventId,
            isEmphasized: highlightEventIds.has(event.id),
          });
          hitboxes.push({ x, y: barY, w, h: BAR_H, task });
        });
      });

      // ---- Buffer 生命周期：横轴时间，纵轴内存地址 / 大小 ----
      const region = metrics.regionById[focusRegionId] || metrics.regions[0];
      const chartTop = AXIS_H + pipes.length * LANE_H + CHART_GAP;
      const chartBottom = chartTop + CHART_H;
      const scale = Math.max(1, region.capacity, region.reserved);
      const yOf = (bytes) => chartBottom - (bytes / scale) * CHART_H;

      ctx.fillStyle = T['surface-2'];
      KIT.roundRect(ctx, PAD_L, chartTop, plotW, CHART_H, 4);
      ctx.fill();

      ctx.font = `600 11px ${T['font-sans']}`;
      ctx.fillStyle = T['foreground-secondary'];
      ctx.fillText(`${region.id} 生命周期`, 0, chartTop - 8);
      ctx.font = `500 9px ${T['font-mono']}`;
      ctx.fillStyle = T['foreground-disabled'];
      ctx.fillText('内存大小', 0, chartTop + 10);

      // 纵轴刻度：每个生命周期块的高度就是实际占用字节数。
      ctx.textAlign = 'right';
      [0, 0.25, 0.5, 0.75, 1].forEach((ratio) => {
        const bytes = scale * ratio;
        const y = yOf(bytes);
        ctx.strokeStyle = T['border-subtle'];
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(PAD_L, y);
        ctx.lineTo(PAD_L + plotW, y);
        ctx.stroke();
        ctx.fillStyle = T['foreground-muted'];
        ctx.fillText(F.bytes(bytes), PAD_L - 7, y + (ratio === 0 ? -3 : 3));
      });
      ctx.textAlign = 'left';

      // 容量线
      const capY = yOf(region.capacity);
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = region.reserved > region.capacity ? T.danger : T['border-strong'];
      ctx.beginPath();
      ctx.moveTo(PAD_L, capY);
      ctx.lineTo(PAD_L + plotW, capY);
      ctx.stroke();
      ctx.setLineDash([]);

      region.allocations.forEach((alloc, allocIndex) => {
        alloc.intervals.forEach((interval) => {
          const x = xOf(interval.start);
          const width = Math.max(2, xOf(interval.end) - x);
          const actualTop = yOf(alloc.offset + alloc.size);
          const actualBottom = yOf(alloc.offset);
          const height = Math.max(3, actualBottom - actualTop);
          const y = actualBottom - height;
          const selected = alloc.id === selectedAllocId;
          const highlighted = highlightIds.has(alloc.id);
          const conflicting = conflictIds.has(alloc.id);
          const reused = !!alloc.reuseOf || !!alloc.manualReuse;

          ctx.fillStyle = conflicting
            ? withAlpha(T.danger, 0.72)
            : withAlpha(region.accent, selected || highlighted ? 0.82 : 0.48 + (allocIndex % 3) * 0.1);
          KIT.roundRect(ctx, x, y, width, height, Math.min(3, height / 2));
          ctx.fill();

          ctx.strokeStyle = selected ? T.foreground : (reused ? T.warning : withAlpha(region.accent, 0.9));
          ctx.lineWidth = selected ? 2 : 1;
          ctx.setLineDash(reused ? [4, 3] : []);
          KIT.roundRect(ctx, x + 0.5, y + 0.5, Math.max(1, width - 1), Math.max(1, height - 1), Math.min(3, height / 2));
          ctx.stroke();
          ctx.setLineDash([]);

          if (width > 48 && height >= 10) {
            ctx.save();
            ctx.beginPath();
            ctx.rect(x + 3, y + 1, width - 6, height - 2);
            ctx.clip();
            ctx.fillStyle = T.foreground;
            ctx.font = `600 9px ${T['font-mono']}`;
            ctx.fillText(alloc.name, x + 5, y + Math.min(height - 2, 11));
            ctx.restore();
          }

          hitboxes.push({
            kind: 'allocation', x, y, w: width, h: height,
            task: { allocation: alloc, interval, label: alloc.name },
          });
        });
      });

      ctx.font = `500 9px ${T['font-mono']}`;
      ctx.fillStyle = region.reserved > region.capacity ? T.danger : T['foreground-muted'];
      ctx.fillText(`容量 ${F.bytes(region.capacity)}`, PAD_L + 6, capY + 11);

      // ---- 时间游标 ----
      const cursorX = xOf(Math.min(tick, total));
      ctx.strokeStyle = T.foreground;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cursorX, AXIS_H - 8);
      ctx.lineTo(cursorX, chartBottom);
      ctx.stroke();
      ctx.fillStyle = T.foreground;
      KIT.roundRect(ctx, cursorX - 18, chartBottom + 4, 36, 15, 3);
      ctx.fill();
      ctx.fillStyle = T.background;
      ctx.font = `600 10px ${T['font-mono']}`;
      ctx.textAlign = 'center';
      ctx.fillText(`#${tick}`, cursorX, chartBottom + 15);
      ctx.textAlign = 'left';
    }

    function onClick(event) {
      const hit = hitAt(event);
      if (hit?.kind === 'allocation') { options.onSelectAllocation?.(hit.task.allocation); return; }
      if (hit) { options.onSelectEvent?.(hit.task.event); return; }
      // 空白处点击 = 把时间游标拖到该位置
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      if (x < PAD_L || !state) return;
      const plotW = Math.max(120, canvas.clientWidth - PAD_L - PAD_R);
      options.onSeek?.(Math.round(((x - PAD_L) / plotW) * state.run.totalTicks));
    }

    function onMove(event) {
      const hit = hitAt(event);
      canvas.style.cursor = hit ? 'pointer' : 'crosshair';
      if (!hit) hover?.tooltip?.classList.remove('is-visible');
    }

    canvas.addEventListener('click', onClick);
    canvas.addEventListener('pointermove', onMove);
    const unobserve = KIT.observeSize(container, draw);

    return {
      update(next) {
        state = {
          allocById: new Map((next.run?.allocations || []).map((a) => [a.id, a])),
          ...next,
          highlightEventIds: new Set(next.highlightEventIds || []),
          highlightIds: new Set(next.highlightIds || []),
          conflictIds: new Set(next.conflictIds || []),
        };
        draw();
      },
      redraw: draw,
      destroy() {
        unobserve();
        canvas.removeEventListener('click', onClick);
        canvas.removeEventListener('pointermove', onMove);
        hover?.destroy?.();
        canvas.remove();
      },
    };
  }

  global.MemVizPipelineView = { create };
})(window);
