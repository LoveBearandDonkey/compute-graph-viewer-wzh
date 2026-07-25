/*
  底部面板 —— 占用水位曲线（规划文档 §4.3-3）
  ------------------------------------------------------------------
  六个存储层级共享一个百分比纵轴，一眼看出「哪一层先顶到容量」。
  实线 = 实际持有 / 容量；虚线 = 静态预留 / 容量；100% 线是硬边界。
  当前聚焦的层级加粗，其余淡出，避免六条线糊成一团。
*/
(function registerMemVizWatermarkView(global) {
  'use strict';

  const KIT = global.MemVizCanvasKit;
  const F = global.MemVizFormat;

  const PAD_L = 46;
  const PAD_R = 12;
  const PAD_T = 16;
  const PAD_B = 22;

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

    function draw() {
      if (!state) return;
      const { run, metrics, tick, focusRegionId } = state;
      const cssWidth = container.clientWidth;
      const cssHeight = Math.max(96, container.clientHeight);
      const { ctx, width, height } = KIT.fitCanvas(canvas, cssWidth, cssHeight);
      const T = KIT.tokens(container);
      const plotW = Math.max(80, width - PAD_L - PAD_R);
      const plotH = Math.max(48, height - PAD_T - PAD_B);
      const total = Math.max(1, run.totalTicks);
      // 纵轴上界：至少 100%，超限时给超出部分留出空间
      const maxRatio = Math.max(1.05, ...metrics.regions.map((r) => r.reservedRatio + 0.05));
      const xOf = (t) => PAD_L + (t / total) * plotW;
      const yOf = (ratio) => PAD_T + plotH - (ratio / maxRatio) * plotH;

      ctx.textBaseline = 'alphabetic';
      ctx.font = `500 9px ${T['font-mono']}`;
      [0, 0.5, 1].forEach((ratio) => {
        const y = yOf(ratio);
        ctx.strokeStyle = ratio === 1 ? withAlpha('#FF4B7B', 0.55) : T['border-subtle'];
        ctx.setLineDash(ratio === 1 ? [4, 4] : []);
        ctx.beginPath();
        ctx.moveTo(PAD_L, y);
        ctx.lineTo(PAD_L + plotW, y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = ratio === 1 ? T.danger : T['foreground-disabled'];
        ctx.fillText(`${ratio * 100}%`, 4, y + 3);
      });

      const focus = focusRegionId;
      metrics.regions.forEach((region) => {
        const isFocus = region.id === focus;
        const alpha = isFocus ? 1 : 0.32;

        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = withAlpha(region.accent, alpha * 0.7);
        ctx.lineWidth = 1;
        const resY = yOf(region.reservedRatio);
        ctx.beginPath();
        ctx.moveTo(PAD_L, resY);
        ctx.lineTo(PAD_L + plotW, resY);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.beginPath();
        for (let t = 0; t <= total; t += 1) {
          const x = xOf(t);
          const y = yOf((region.series[t] || 0) / region.capacity);
          if (t === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = withAlpha(region.accent, alpha);
        ctx.lineWidth = isFocus ? 2 : 1;
        ctx.stroke();

        if (isFocus) {
          const px = xOf(region.peakTick);
          const py = yOf(region.peakLive / region.capacity);
          ctx.fillStyle = region.accent;
          ctx.beginPath();
          ctx.arc(px, py, 3, 0, Math.PI * 2);
          ctx.fill();
          ctx.font = `600 10px ${T['font-mono']}`;
          ctx.fillStyle = T.foreground;
          const text = `峰值 ${F.bytes(region.peakLive)} @${F.tick(region.peakTick)}`;
          const tw = ctx.measureText(text).width;
          ctx.fillText(text, Math.min(px + 6, PAD_L + plotW - tw), py - 6);
        }

        ctx.font = `600 9px ${T['font-mono']}`;
        ctx.fillStyle = withAlpha(region.accent, isFocus ? 1 : 0.5);
        ctx.fillText(region.id, PAD_L + plotW - 22, yOf(region.reservedRatio) - 3);
      });

      const cursorX = xOf(Math.min(tick, total));
      ctx.strokeStyle = T.foreground;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cursorX, PAD_T - 6);
      ctx.lineTo(cursorX, PAD_T + plotH);
      ctx.stroke();
      ctx.font = `500 9px ${T['font-mono']}`;
      ctx.fillStyle = T['foreground-muted'];
      ctx.fillText(`#0`, PAD_L, PAD_T + plotH + 13);
      const endText = `#${total}`;
      ctx.fillText(endText, PAD_L + plotW - ctx.measureText(endText).width, PAD_T + plotH + 13);
      ctx.fillStyle = T.foreground;
      ctx.fillText(`#${tick}`, Math.min(cursorX + 4, PAD_L + plotW - 24), PAD_T + plotH + 13);
    }

    function onClick(event) {
      if (!state) return;
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const plotW = Math.max(80, canvas.clientWidth - PAD_L - PAD_R);
      if (x < PAD_L) return;
      options.onSeek?.(Math.round(((x - PAD_L) / plotW) * state.run.totalTicks));
    }

    canvas.addEventListener('click', onClick);
    canvas.style.cursor = 'crosshair';
    const unobserve = KIT.observeSize(container, draw);

    return {
      update(next) { state = next; draw(); },
      redraw: draw,
      destroy() {
        unobserve();
        canvas.removeEventListener('click', onClick);
        canvas.remove();
      },
    };
  }

  global.MemVizWatermarkView = { create };
})(window);
