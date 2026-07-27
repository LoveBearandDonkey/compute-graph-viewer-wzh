/*
  底部面板 —— 候选对比与差距分解（场景 6）
  ------------------------------------------------------------------
  一行一个候选，条形按三段分解：

    [   理论下界   ][ 装箱碎片 ][    策略浪费    ]  = 当前 workspace

  这样「workspace 显著下降」这个成功判据在一屏之内可核对：既看到本候选降到了
  哪里，也看到别的结构改动能把下界本身推到多低（ws-inplace 那一行）。
  预算线是竖直的硬边界，越过即超预算。

  三段配色：下界=success（不可再低）、碎片=中性斜纹（要动结构）、
  策略浪费=warning（改地址就能拿回来），与主视图的线色一致。
*/
(function registerMemVizWsGapView(global) {
  'use strict';

  const KIT = global.MemVizCanvasKit;
  const F = global.MemVizFormat;

  const PAD_L = 132;
  const PAD_R = 96;
  const PAD_T = 10;
  const ROW_H = 26;
  const BAR_H = 13;

  function withAlpha(hex, alpha) {
    const value = String(hex || '#888').replace('#', '');
    const num = parseInt(value.length === 3 ? value.split('').map((c) => c + c).join('') : value, 16);
    return `rgba(${(num >> 16) & 255}, ${(num >> 8) & 255}, ${num & 255}, ${alpha})`;
  }

  function create(container, options = {}) {
    const canvas = document.createElement('canvas');
    canvas.className = 'mv-canvas';
    container.appendChild(canvas);

    let state = null;
    let hitboxes = [];
    const tip = KIT.createTooltip(container);

    function draw() {
      if (!state) return;
      const { rows, activeId } = state;
      const cssWidth = Math.max(420, container.clientWidth);
      const cssHeight = Math.max(96, PAD_T * 2 + rows.length * ROW_H + 16);
      const { ctx, width } = KIT.fitCanvas(canvas, cssWidth, cssHeight);
      const T = KIT.tokens(container);
      hitboxes = [];

      const plotW = Math.max(120, width - PAD_L - PAD_R);
      const scale = Math.max(...rows.map((r) => Math.max(r.plan.current, r.plan.budget))) * 1.02;
      const xOf = (bytes) => PAD_L + (bytes / scale) * plotW;

      // 预算线
      const budgetX = xOf(rows[0].plan.budget);
      ctx.setLineDash([2, 4]);
      ctx.strokeStyle = T['border-strong'];
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(budgetX, PAD_T);
      ctx.lineTo(budgetX, PAD_T + rows.length * ROW_H + 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = `500 9px ${T['font-sans']}`;
      ctx.fillStyle = T['foreground-disabled'];
      ctx.fillText(`预算 ${F.bytes(rows[0].plan.budget)}`, budgetX + 4, PAD_T + rows.length * ROW_H + 13);

      rows.forEach((row, index) => {
        const y = PAD_T + index * ROW_H;
        const barY = y + (ROW_H - BAR_H) / 2;
        const active = row.run.id === activeId;

        // 候选名
        ctx.font = `${active ? 600 : 500} 10.5px ${T['font-sans']}`;
        ctx.fillStyle = active ? T.foreground : T['foreground-muted'];
        ctx.fillText(KIT.truncate(ctx, row.run.label, PAD_L - 46), 0, barY + BAR_H - 3);

        // 比值
        ctx.font = `600 10px ${T['font-mono']}`;
        ctx.fillStyle = row.plan.ratio > 1.2 ? T.warning : T['foreground-disabled'];
        ctx.fillText(`${row.plan.ratio.toFixed(2)}×`, PAD_L - 40, barY + BAR_H - 3);

        // 底槽
        ctx.fillStyle = T['surface-2'];
        KIT.roundRect(ctx, PAD_L, barY, plotW, BAR_H, 3);
        ctx.fill();

        const p = row.plan;
        const x0 = PAD_L;
        const x1 = xOf(p.lowerBound);
        const x2 = xOf(p.packed);
        const x3 = xOf(p.current);

        // 下界段
        ctx.fillStyle = withAlpha(T.success || '#4a9568', active ? 0.85 : 0.4);
        ctx.fillRect(x0, barY, Math.max(1, x1 - x0), BAR_H);
        // 装箱碎片段
        if (x2 > x1) {
          ctx.fillStyle = withAlpha('#FFFFFF', active ? 0.16 : 0.08);
          ctx.fillRect(x1, barY, x2 - x1, BAR_H);
          KIT.hatch(ctx, x1, barY, x2 - x1, BAR_H,
            withAlpha('#FFFFFF', active ? 0.4 : 0.2), 3);
        }
        // 策略浪费段
        if (x3 > x2) {
          ctx.fillStyle = withAlpha(T.warning || '#FFAA3B', active ? 0.55 : 0.25);
          ctx.fillRect(x2, barY, x3 - x2, BAR_H);
        }

        // 超预算部分描红
        if (p.overBudget > 0) {
          const ob = Math.max(budgetX, x0);
          ctx.strokeStyle = T.danger;
          ctx.lineWidth = 1.2;
          ctx.strokeRect(ob + 0.5, barY + 0.5, Math.max(1, x3 - ob) - 1, BAR_H - 1);
        }

        if (active) {
          ctx.strokeStyle = T['border-strong'];
          ctx.lineWidth = 1;
          KIT.roundRect(ctx, PAD_L - 1, barY - 2, plotW + 2, BAR_H + 4, 4);
          ctx.stroke();
        }

        // 右侧当前值
        ctx.font = `${active ? 600 : 500} 10px ${T['font-mono']}`;
        ctx.fillStyle = p.overBudget > 0 ? T.danger : T['foreground-secondary'];
        ctx.fillText(F.bytes(p.current), PAD_L + plotW + 8, barY + BAR_H - 3);

        hitboxes.push({ x: 0, y, w: width, h: ROW_H, row });
      });
    }

    function hitAt(event) {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      return hitboxes.find((b) => x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h) || null;
    }

    canvas.addEventListener('mousemove', (event) => {
      const hit = hitAt(event);
      if (!hit) { tip.hide(); return; }
      const p = hit.row.plan;
      tip.show(
        `<div class="mv-tip__title">${F.escapeHtml(hit.row.run.label)}</div>`
        + `<div class="mv-tip__row"><span>当前</span><b>${F.bytes(p.current)}</b></div>`
        + `<div class="mv-tip__row"><span>复用后可达</span><b>${F.bytes(p.packed)}</b></div>`
        + `<div class="mv-tip__row"><span>理论下界</span><b>${F.bytes(p.lowerBound)}</b></div>`
        + `<div class="mv-tip__row"><span>策略浪费</span><b>${F.bytes(p.policyWaste)}</b></div>`
        + `<div class="mv-tip__row"><span>装箱碎片</span><b>${F.bytes(p.packFragment)}</b></div>`
        + (p.overBudget > 0
          ? `<div class="mv-tip__row is-bad"><span>超预算</span><b>${F.bytes(p.overBudget)}</b></div>` : '')
        + `<div class="mv-tip__src">${F.escapeHtml(hit.row.run.kicker)}</div>`,
        event.clientX, event.clientY,
      );
    });
    canvas.addEventListener('mouseleave', () => tip.hide());
    canvas.addEventListener('click', (event) => {
      const hit = hitAt(event);
      if (hit) options.onPickRun?.(hit.row.run.id);
    });

    const unobserve = KIT.observeSize(container, draw);

    return {
      update(next) { state = next; draw(); },
      redraw: draw,
      destroy() { unobserve(); tip.destroy(); canvas.remove(); },
    };
  }

  global.MemVizWsGapView = { create };
})(window);
