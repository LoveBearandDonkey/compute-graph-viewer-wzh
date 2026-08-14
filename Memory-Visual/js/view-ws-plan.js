/*
  主视图 —— Workspace 规划（场景 6）
  ------------------------------------------------------------------
  一屏说清三个数与它们的来源：

    上半  子计算带：融合体的拓扑序，一个子计算一根任务条（swimlane-task 的
          drawTaskBar），点击切游标。x 轴自此确定 —— 场景 6 的时间单位是
          「子计算序」而不是 cycle。
    下半  堆叠列：每个子计算上真正同时存活的字节，按产出它的子计算着色。
          柱子的最高点就是理论下界 —— 这是「真实需求」。
    横线  三个数：current（当前预留）/ packed（复用后可达）/ lowerBound（下界），
          外加 budget 预算线。柱子与横线之间的空白就是浪费。

  为什么不用面积图：x 轴是离散的子计算序，不是连续时间，插值出来的斜坡是假的。

  配色：柱子分段取 swimlane-task 的 lane-kind 家族色（编码「哪个子计算产出的字节」），
  属 data-viz exemption；其余一律 token。
*/
(function registerMemVizWsPlanView(global) {
  'use strict';

  const KIT = global.MemVizCanvasKit;
  const F = global.MemVizFormat;

  const PAD_L = 84;
  const PAD_R = 158;
  const BAND_TOP = 22;
  const BAND_H = 26;
  const CHART_TOP = 78;
  const CHART_H = 300;
  const AXIS_H = 34;

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
      return hitboxes.find((b) => x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h) || null;
    }

    const hover = SW.initHoverTooltip({
      root: canvas,
      targets: [canvas],
      appendTo: container,
      bounds: container,
      getTask: (target, event) => hitAt(event)?.task || null,
      getTooltipHtml: (task) => {
        if (task.sgOnly) {
          const sg = task.subgraph;
          const cell = state.plan.perSubgraph[task.index];
          return `<div class="mv-tip__title">${F.escapeHtml(sg.id)} · ${F.escapeHtml(sg.name)}</div>`
            + `<div class="mv-tip__row"><span>作用</span><b>${F.escapeHtml(sg.desc)}</b></div>`
            + `<div class="mv-tip__row"><span>同时存活</span><b>${F.bytes(cell.bytes)}</b></div>`
            + `<div class="mv-tip__row"><span>存活张量</span><b>${cell.members.length}</b></div>`
            + `<div class="mv-tip__src">${F.escapeHtml(cell.members.map((t) => t.name).join(' · ')) || '—'}</div>`;
        }
        const t = task.tensor;
        const run = state.run;
        return `<div class="mv-tip__title">${F.escapeHtml(t.name)}</div>`
          + `<div class="mv-tip__row"><span>大小</span><b>${F.bytes(t.size)}</b></div>`
          + `<div class="mv-tip__row"><span>dtype / shape</span><b>${F.escapeHtml(t.dtype)} ${F.escapeHtml(F.shape(t.shape))}</b></div>`
          + `<div class="mv-tip__row"><span>产出</span><b>${F.escapeHtml(t.producer)}</b></div>`
          + `<div class="mv-tip__row"><span>最后消费</span><b>${F.escapeHtml(t.consumers.join(', '))}</b></div>`
          + `<div class="mv-tip__row"><span>地址</span><b>${F.hex(run.layout[t.id], 6)}</b></div>`
          + (t.blockScope === 'per-block'
            ? `<div class="mv-tip__row is-bad"><span>作用域</span><b>per-block × ${run.kernel.blockDim}</b></div>` : '')
          + `<div class="mv-tip__src">${F.escapeHtml(t.note || '')}</div>`;
      },
    });

    function draw() {
      if (!state) return;
      const { run, plan, sg: cursor, selectedTensorId, highlightIds } = state;
      // 下限只兜住极窄面板；再高会让常规布局白白多出一条横向滚动
      const cssWidth = Math.max(480, container.clientWidth);
      const cssHeight = CHART_TOP + CHART_H + AXIS_H;
      const { ctx, width, height } = KIT.fitCanvas(canvas, cssWidth, cssHeight);
      const T = KIT.tokens(container);
      hitboxes = [];

      const plotW = Math.max(200, width - PAD_L - PAD_R);
      const count = run.subgraphs.length;
      const colW = plotW / count;
      const barW = Math.min(96, colW - 14);
      const chartBottom = CHART_TOP + CHART_H;

      // 纵轴上界：把最高的那条线也留在画面内
      const scale = Math.max(plan.current, plan.lowerBound, plan.budget) * 1.08;
      const yOf = (bytes) => chartBottom - (bytes / scale) * CHART_H;
      const centerOf = (index) => PAD_L + index * colW + colW / 2;

      // ---- 标题 ----
      ctx.font = `500 10px ${T['font-sans']}`;
      ctx.fillStyle = T['foreground-muted'];
      ctx.fillText('子计算', 0, BAND_TOP + BAND_H / 2 + 4);
      ctx.fillText('同时存活', 0, CHART_TOP + 10);
      ctx.font = `500 9px ${T['font-mono']}`;
      ctx.fillStyle = T['foreground-disabled'];
      ctx.fillText('字节', 0, CHART_TOP + 23);

      // ---- 子计算带 ----
      run.subgraphs.forEach((sg, index) => {
        const x = PAD_L + index * colW + (colW - barW) / 2;
        const task = {
          sgOnly: true,
          subgraph: sg,
          index,
          label: sg.name,
          displayName: sg.name,
          laneKind: sg.lane,
          laneId: sg.id,
          totalCycle: 1,
        };
        SW.drawTaskBar(ctx, {
          task,
          x,
          y: BAND_TOP,
          width: barW,
          height: BAND_H,
          baseColor: colormap.colorForLaneKind(sg.lane),
          fontFamily: T['font-sans'],
          isSelected: index === cursor,
          isEmphasized: false,
        });
        hitboxes.push({ x, y: BAND_TOP, w: barW, h: BAND_H, task });
      });

      // ---- 游标列 ----
      const cx = PAD_L + cursor * colW;
      ctx.fillStyle = withAlpha('#FFFFFF', 0.04);
      ctx.fillRect(cx, CHART_TOP, colW, CHART_H);

      // ---- 网格 ----
      ctx.strokeStyle = T['border-subtle'];
      ctx.lineWidth = 1;
      ctx.setLineDash([]);
      for (let i = 0; i <= 4; i += 1) {
        const y = chartBottom - (i / 4) * CHART_H;
        ctx.beginPath();
        ctx.moveTo(PAD_L, y);
        ctx.lineTo(PAD_L + plotW, y);
        ctx.stroke();
      }

      // ---- 堆叠列：每个子计算的同时存活字节 ----
      plan.perSubgraph.forEach((cell, index) => {
        const x = PAD_L + index * colW + (colW - barW) / 2;
        let acc = 0;
        // 按声明序堆叠，保证同一张量在相邻列的纵向位置连续，视觉上能连成一条带
        cell.members.slice().sort((a, b) => a.order - b.order).forEach((t) => {
          const y0 = yOf(acc);
          const y1 = yOf(acc + t.size);
          const h = Math.max(1, y0 - y1);
          // 着色按「产出它的子计算」，所以同一张量在相邻列颜色一致
          const base = colormap.colorForLaneKind(run.subgraphs[t.live.start]?.lane || 'aic');
          const dim = selectedTensorId && selectedTensorId !== t.id;
          ctx.fillStyle = withAlpha(base, dim ? 0.24 : 0.82);
          ctx.fillRect(x, y1, barW, h);

          // 选中/关联张量描边，让诊断条目点过来能一眼定位
          if (t.id === selectedTensorId || highlightIds.has(t.id)) {
            ctx.strokeStyle = t.id === selectedTensorId ? T.foreground : T['foreground-secondary'];
            ctx.lineWidth = t.id === selectedTensorId ? 1.6 : 1;
            ctx.strokeRect(x + 0.5, y1 + 0.5, barW - 1, h - 1);
          } else {
            ctx.strokeStyle = withAlpha('#000000', 0.35);
            ctx.lineWidth = 1;
            ctx.strokeRect(x + 0.5, y1 + 0.5, barW - 1, h - 1);
          }

          if (h >= 14 && barW > 42) {
            ctx.font = `600 9px ${T['font-mono']}`;
            ctx.fillStyle = dim ? T['foreground-disabled'] : T.background;
            ctx.fillText(KIT.truncate(ctx, t.name, barW - 8), x + 4, y1 + h / 2 + 3);
          }

          hitboxes.push({ x, y: y1, w: barW, h, task: { tensor: t, index } });
          acc += t.size;
        });

        // 列顶读数。窄面板下列宽装不住这串数字时只保留峰值那一列 ——
        // 相邻列的读数彼此挤在一起比没有读数更糟。
        const isPeak = cell.bytes === plan.lowerBound;
        ctx.font = `600 10px ${T['font-mono']}`;
        const label = F.bytes(cell.bytes);
        const lw = ctx.measureText(label).width;
        if (lw <= colW - 6 || isPeak) {
          ctx.fillStyle = isPeak ? T.success : T['foreground-secondary'];
          ctx.fillText(label, x + (barW - lw) / 2, yOf(cell.bytes) - 6);
        }
      });

      // ---- 三个数 + 预算线 ----
      const lines = [
        { value: plan.current, color: plan.overBudget > 0 ? T.danger : T.warning, dash: [], label: '当前', note: F.bytes(plan.current) },
        { value: plan.packed, color: T.accent, dash: [5, 4], label: '复用后可达', note: F.bytes(plan.packed) },
        { value: plan.lowerBound, color: T.success, dash: [2, 3], label: '理论下界', note: F.bytes(plan.lowerBound) },
        { value: plan.budget, color: T['foreground-disabled'], dash: [1, 5], label: '预算', note: F.bytes(plan.budget) },
      ];

      const plotRight = PAD_L + plotW;
      const bandX = plotRight + 3;   // 差距带画在右侧留白里，不压到最后一列的柱子上
      const bandW = 8;
      const labelX = plotRight + bandW + 9;

      // ---- 差距带：当前线与可达线之间 = 策略浪费（改地址就能拿回来）----
      if (plan.policyWaste > 0) {
        const yTop = yOf(plan.current);
        const yBottom = yOf(plan.packed);
        ctx.fillStyle = withAlpha('#FFAA3B', 0.14);
        ctx.fillRect(bandX, yTop, bandW, yBottom - yTop);
        KIT.hatch(ctx, bandX, yTop, bandW, yBottom - yTop, withAlpha('#FFAA3B', 0.6), 4);
      }

      // 同值时（如已排满的候选）把标签错开，避免文字叠在一起
      const used = [];
      const putLabel = (y, label, note, color) => {
        let labelY = y + 3.5;
        while (used.some((v) => Math.abs(v - labelY) < 12)) labelY += 13;
        used.push(labelY);
        ctx.font = `600 10px ${T['font-sans']}`;
        ctx.fillStyle = color;
        ctx.fillText(label, labelX, labelY);
        if (!note) return;
        ctx.font = `500 10px ${T['font-mono']}`;
        ctx.fillStyle = T['foreground-secondary'];
        ctx.fillText(note, labelX + ctx.measureText(`${label}  `).width, labelY);
      };

      lines.forEach((line) => {
        if (line.value > scale) return;
        const y = yOf(line.value);
        ctx.setLineDash(line.dash);
        ctx.strokeStyle = line.color;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.moveTo(PAD_L, y);
        ctx.lineTo(plotRight, y);
        ctx.stroke();
        ctx.setLineDash([]);
        putLabel(y, line.label, line.note, line.color);
      });

      // 差距读数跟着差距带走，放在两条线的中间高度
      if (plan.policyWaste > 0) {
        putLabel((yOf(plan.current) + yOf(plan.packed)) / 2,
          '策略浪费', F.bytes(plan.policyWaste), T.warning);
      }
      if (plan.packFragment > 0) {
        putLabel((yOf(plan.packed) + yOf(plan.lowerBound)) / 2,
          '装箱碎片', F.bytes(plan.packFragment), T['foreground-muted']);
      }

      // ---- x 轴刻度 ----
      ctx.font = `500 10px ${T['font-mono']}`;
      run.subgraphs.forEach((sg, index) => {
        const label = sg.id;
        const w = ctx.measureText(label).width;
        ctx.fillStyle = index === cursor ? T.foreground : T['foreground-disabled'];
        ctx.fillText(label, centerOf(index) - w / 2, chartBottom + 16);
      });
      ctx.font = `500 9px ${T['font-sans']}`;
      ctx.fillStyle = T['foreground-disabled'];
      ctx.fillText('子计算序（拓扑序）', PAD_L, chartBottom + 30);
    }

    canvas.addEventListener('click', (event) => {
      const hit = hitAt(event);
      if (!hit) return;
      if (hit.task.sgOnly) options.onPickSubgraph?.(hit.task.index);
      else options.onPickTensor?.(hit.task.tensor.id, hit.task.index);
    });
    canvas.addEventListener('mouseleave', () => hover?.hide?.());

    const unobserve = KIT.observeSize(container, draw);

    return {
      update(next) {
        state = { highlightIds: new Set(), ...next };
        draw();
      },
      redraw: draw,
      destroy() {
        unobserve();
        hover?.destroy?.();
        canvas.remove();
      },
    };
  }

  global.MemVizWsPlanView = { create };
})(window);
