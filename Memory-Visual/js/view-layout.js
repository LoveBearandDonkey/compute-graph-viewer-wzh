/*
  视图 A —— 内存布局图（规划文档 §4.3-1）
  ------------------------------------------------------------------
  地址空间为横轴、存储层级分栏的条带图：
    · 色块 = 一次静态分配（TPipe::InitBuffer 的预留区间）；
    · 实心 = 当前时刻真正持有数据；半透明 = 预留着但此刻是空的；
    · 色块之间的空隙 = 碎片，用斜纹标出；
    · 容量线右侧的红色区域 = 超限部分，编译期就会报错的那一段。
  时间游标移动时实心/半透明会变化，于是「预留了多少」和「真在用多少」
  的差距一眼可见 —— 这正是场景 1/2 要回答的问题。

  颜色说明：region.accent 属 data-viz exemption（编码存储层级身份），
  其余全部走设计系统 token。
*/
(function registerMemVizLayoutView(global) {
  'use strict';

  const KIT = global.MemVizCanvasKit;
  const MET = global.MemVizMetrics;
  const F = global.MemVizFormat;

  const ROW_H = 62;
  const PAD_TOP = 14;
  const LABEL_W = 108;
  const META_W = 132;
  const BAR_H = 26;

  function withAlpha(hex, alpha) {
    const value = hex.replace('#', '');
    const num = parseInt(value.length === 3 ? value.split('').map((c) => c + c).join('') : value, 16);
    return `rgba(${(num >> 16) & 255}, ${(num >> 8) & 255}, ${num & 255}, ${alpha})`;
  }

  function create(container, options = {}) {
    const canvas = document.createElement('canvas');
    canvas.className = 'mv-canvas';
    container.appendChild(canvas);
    const tip = KIT.createTooltip(container);

    let state = null;
    let hitboxes = [];
    let hovered = null;

    function draw() {
      if (!state) return;
      const { metrics, tick, selectedId, highlightIds, conflictIds } = state;
      const rows = metrics.regions;
      const cssWidth = container.clientWidth;
      const cssHeight = PAD_TOP * 2 + rows.length * ROW_H;
      canvas.style.height = `${cssHeight}px`;
      const { ctx, width } = KIT.fitCanvas(canvas, cssWidth, cssHeight);
      const T = KIT.tokens(container);
      const barX = LABEL_W;
      const barW = Math.max(80, width - LABEL_W - META_W);
      hitboxes = [];

      rows.forEach((region, index) => {
        const top = PAD_TOP + index * ROW_H;
        const barY = top + 16;
        const scale = Math.max(region.capacity, region.reserved);
        const px = barW / scale;
        const capX = barX + region.capacity * px;

        // --- 左栏：层级标识 ---
        ctx.textBaseline = 'alphabetic';
        ctx.fillStyle = region.accent;
        ctx.font = `600 12px ${T['font-mono']}`;
        ctx.fillText(region.id, 0, top + 14);
        ctx.fillStyle = T['foreground-muted'];
        ctx.font = `500 10px ${T['font-sans']}`;
        ctx.fillText(KIT.truncate(ctx, region.label, LABEL_W - 12), 0, top + 28);
        ctx.fillText(`${region.owner} · ${region.align}B 对齐`, 0, top + 42);

        // --- 轨道 ---
        ctx.fillStyle = T['surface-2'];
        KIT.roundRect(ctx, barX, barY, barW, BAR_H, 4);
        ctx.fill();

        // --- 超限区（容量线右侧）---
        if (region.reserved > region.capacity) {
          ctx.fillStyle = withAlpha('#FF4B7B', 0.14);
          KIT.roundRect(ctx, capX, barY, barX + barW - capX, BAR_H, 4);
          ctx.fill();
        }

        // --- 碎片：相邻分配之间的空隙 ---
        const sorted = region.allocations.slice().sort((a, b) => a.offset - b.offset);
        let cursor = 0;
        sorted.forEach((alloc) => {
          if (alloc.offset > cursor) {
            KIT.hatch(ctx, barX + cursor * px, barY + 1, (alloc.offset - cursor) * px, BAR_H - 2,
              withAlpha('#FFFFFF', 0.08), 4);
          }
          cursor = Math.max(cursor, alloc.offset + alloc.size);
        });

        // --- 分配块 ---
        sorted.forEach((alloc) => {
          const x = barX + alloc.offset * px;
          const w = Math.max(1.5, alloc.size * px);
          const live = MET.liveAt(alloc, tick);
          const isSelected = alloc.id === selectedId;
          const isHighlighted = highlightIds.has(alloc.id);
          const isConflict = conflictIds.has(alloc.id);

          ctx.fillStyle = withAlpha(region.accent, live ? 0.92 : 0.24);
          KIT.roundRect(ctx, x, barY + 1, w, BAR_H - 2, 3);
          ctx.fill();

          if (!live) {
            ctx.strokeStyle = withAlpha(region.accent, 0.5);
            ctx.lineWidth = 1;
            KIT.roundRect(ctx, x + 0.5, barY + 1.5, w - 1, BAR_H - 3, 3);
            ctx.stroke();
          }
          if (isConflict || isHighlighted || isSelected) {
            ctx.strokeStyle = isConflict ? T.danger : (isSelected ? T.foreground : T.warning);
            ctx.lineWidth = isSelected ? 2 : 1.5;
            KIT.roundRect(ctx, x + 1, barY + 2, w - 2, BAR_H - 4, 3);
            ctx.stroke();
          }

          if (w > 34) {
            ctx.fillStyle = live ? T.background : T['foreground-secondary'];
            ctx.font = `600 10px ${T['font-mono']}`;
            ctx.fillText(KIT.truncate(ctx, alloc.name, w - 8), x + 4, barY + BAR_H / 2 + 3.5);
          }

          hitboxes.push({ x, y: barY, w, h: BAR_H, alloc, region });
        });

        // --- 容量线 ---
        ctx.strokeStyle = region.reserved > region.capacity ? T.danger : T['border-strong'];
        ctx.lineWidth = region.reserved > region.capacity ? 1.5 : 1;
        ctx.beginPath();
        ctx.moveTo(capX, barY - 4);
        ctx.lineTo(capX, barY + BAR_H + 4);
        ctx.stroke();

        // --- 右栏：读数 ---
        const over = region.reserved > region.capacity;
        const metaX = barX + barW + 12;
        ctx.font = `600 12px ${T['font-mono']}`;
        ctx.fillStyle = over ? T.danger : T.foreground;
        ctx.fillText(F.pct(region.reservedRatio, 0), metaX, top + 18);
        ctx.font = `500 10px ${T['font-mono']}`;
        ctx.fillStyle = T['foreground-secondary'];
        ctx.fillText(`${F.bytes(region.reserved)} / ${F.bytes(region.capacity)}`, metaX, top + 32);
        ctx.fillStyle = T['foreground-muted'];
        const liveNow = region.series[Math.min(tick, region.series.length - 1)];
        ctx.fillText(`此刻持有 ${F.bytes(liveNow)}`, metaX, top + 46);
      });
    }

    function findAt(clientX, clientY) {
      const rect = canvas.getBoundingClientRect();
      const x = clientX - rect.left;
      const y = clientY - rect.top;
      return hitboxes.find((b) => x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h) || null;
    }

    function onMove(event) {
      const hit = findAt(event.clientX, event.clientY);
      canvas.style.cursor = hit ? 'pointer' : 'default';
      if (!hit) { tip.hide(); hovered = null; return; }
      if (hovered !== hit.alloc) hovered = hit.alloc;
      const a = hit.alloc;
      const span = MET.liveSpan(a);
      const live = MET.liveAt(a, state.tick);
      tip.show(`
        <div class="mv-tip__title">${F.escapeHtml(a.name)}</div>
        <div class="mv-tip__row"><span>层级</span><b>${a.region} · ${F.escapeHtml(a.queue)}</b></div>
        <div class="mv-tip__row"><span>地址</span><b>${F.hex(a.offset)} + ${F.bytes(a.size)}</b></div>
        <div class="mv-tip__row"><span>数据 / 实占</span><b>${F.bytes(a.dataBytes)} / ${F.bytes(a.size)}</b></div>
        <div class="mv-tip__row"><span>dtype</span><b>${F.escapeHtml(a.dtype)} ${F.shape(a.shape)}</b></div>
        <div class="mv-tip__row"><span>buffer_num</span><b>${a.bufferNum}</b></div>
        <div class="mv-tip__row"><span>生命周期</span><b>${span ? `${F.tick(span.start)}–${F.tick(span.end)}` : '未被访问'}</b></div>
        <div class="mv-tip__row"><span>此刻</span><b>${live ? '持有数据' : '空闲预留'}</b></div>
        <div class="mv-tip__src">${F.escapeHtml(a.src.file)}:${a.src.hotLine + 1}</div>
      `, event.clientX, event.clientY);
    }

    function onClick(event) {
      const hit = findAt(event.clientX, event.clientY);
      options.onSelect?.(hit ? hit.alloc : null);
    }

    canvas.addEventListener('mousemove', onMove);
    canvas.addEventListener('mouseleave', () => { tip.hide(); hovered = null; });
    canvas.addEventListener('click', onClick);
    const unobserve = KIT.observeSize(container, draw);

    return {
      update(next) {
        state = {
          highlightIds: new Set(),
          conflictIds: new Set(),
          ...next,
        };
        draw();
      },
      redraw: draw,
      destroy() {
        unobserve();
        canvas.removeEventListener('mousemove', onMove);
        canvas.removeEventListener('click', onClick);
        tip.destroy();
        canvas.remove();
      },
    };
  }

  global.MemVizLayoutView = { create };
})(window);
