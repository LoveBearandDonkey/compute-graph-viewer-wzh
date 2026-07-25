/*
  Canvas 基础设施 —— DPR 适配、token 取色、圆角、悬浮提示。
  ------------------------------------------------------------------
  Canvas 里不能写 var(--x)，所以统一在这里把设计系统 token 解析成具体色值，
  绝不在视图代码里出现硬编码的 UI 色（数据编码色除外，见各视图注释）。
*/
(function registerMemVizCanvasKit(global) {
  'use strict';

  const TOKEN_KEYS = [
    '--background', '--surface-1', '--surface-2', '--surface-3', '--surface-4',
    '--foreground', '--foreground-secondary', '--foreground-muted', '--foreground-disabled',
    '--border-subtle', '--border-default', '--border-strong',
    '--primary', '--success', '--warning', '--danger', '--accent',
    '--font-sans', '--font-mono',
  ];

  function tokens(element) {
    const style = getComputedStyle(element || document.documentElement);
    const map = {};
    TOKEN_KEYS.forEach((key) => {
      map[key.replace(/^--/, '')] = style.getPropertyValue(key).trim();
    });
    return map;
  }

  /** 按容器 CSS 尺寸与 devicePixelRatio 重置画布，返回已缩放的 2D 上下文。 */
  function fitCanvas(canvas, width, height) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(width));
    const h = Math.max(1, Math.floor(height));
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, width: w, height: h };
  }

  function roundRect(ctx, x, y, w, h, r) {
    const radius = Math.max(0, Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2));
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + w - radius, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
    ctx.lineTo(x + w, y + h - radius);
    ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
    ctx.lineTo(x + radius, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
  }

  /** 45° 斜纹填充，用于「等待空洞」「碎片」这类非实体区域。 */
  function hatch(ctx, x, y, w, h, color, gap = 5) {
    if (w <= 0 || h <= 0) return;
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.clip();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = -h; i < w + h; i += gap) {
      ctx.moveTo(x + i, y + h);
      ctx.lineTo(x + i + h, y);
    }
    ctx.stroke();
    ctx.restore();
  }

  function truncate(ctx, text, maxWidth) {
    if (ctx.measureText(text).width <= maxWidth) return text;
    let low = 0;
    let high = text.length;
    while (low < high) {
      const mid = Math.ceil((low + high) / 2);
      if (ctx.measureText(`${text.slice(0, mid)}…`).width <= maxWidth) low = mid;
      else high = mid - 1;
    }
    return low > 0 ? `${text.slice(0, low)}…` : '';
  }

  /** 跟随光标的轻量提示层；沿用设计系统的 surface / border token。 */
  function createTooltip(host) {
    const el = document.createElement('div');
    el.className = 'mv-tip';
    el.hidden = true;
    host.appendChild(el);
    return {
      element: el,
      show(html, clientX, clientY) {
        el.innerHTML = html;
        el.hidden = false;
        const bounds = host.getBoundingClientRect();
        const rect = el.getBoundingClientRect();
        let x = clientX - bounds.left + 14;
        let y = clientY - bounds.top + 14;
        if (x + rect.width > bounds.width - 8) x = clientX - bounds.left - rect.width - 12;
        if (y + rect.height > bounds.height - 8) y = clientY - bounds.top - rect.height - 12;
        el.style.transform = `translate(${Math.max(4, x)}px, ${Math.max(4, y)}px)`;
      },
      hide() { el.hidden = true; },
      destroy() { el.remove(); },
    };
  }

  /**
   * 容器尺寸变化时重绘；返回解绑函数。
   * 直接同步调用 handler 而不套 requestAnimationFrame：页面处于后台标签页时
   * rAF 会被挂起，套一层会让画布停在 0 宽度。ResizeObserver 自身已做批处理，
   * reentrant 标志防止「重绘改高度 → 再次触发观察」的死循环。
   */
  function observeSize(element, handler) {
    let running = false;
    const run = () => {
      if (running) return;
      running = true;
      try { handler(); } finally { running = false; }
    };
    if (!('ResizeObserver' in window)) {
      window.addEventListener('resize', run);
      return () => window.removeEventListener('resize', run);
    }
    const observer = new ResizeObserver(run);
    observer.observe(element);
    return () => observer.disconnect();
  }

  global.MemVizCanvasKit = { tokens, fitCanvas, roundRect, hatch, truncate, createTooltip, observeSize };
})(window);
