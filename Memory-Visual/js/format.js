/* 通用格式化 —— 全站统一字节 / 比例 / 地址的呈现方式。 */
(function registerMemVizFormat(global) {
  'use strict';

  function bytes(value) {
    const v = Number(value) || 0;
    if (Math.abs(v) >= 1024 * 1024) {
      const mb = v / (1024 * 1024);
      return `${mb >= 10 ? mb.toFixed(1) : mb.toFixed(2)}MB`;
    }
    if (Math.abs(v) >= 1024) {
      const kb = v / 1024;
      return `${Number.isInteger(kb) ? kb : kb.toFixed(1)}KB`;
    }
    return `${v}B`;
  }

  function pct(value, digits = 0) {
    return `${(Number(value) * 100).toFixed(digits)}%`;
  }

  function hex(value, width = 5) {
    return `0x${(Number(value) || 0).toString(16).toUpperCase().padStart(width, '0')}`;
  }

  function shape(list) {
    return Array.isArray(list) ? `[${list.join(', ')}]` : '—';
  }

  function tick(value) {
    return `#${Math.round(Number(value) || 0)}`;
  }

  function escapeHtml(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  global.MemVizFormat = { bytes, pct, hex, shape, tick, escapeHtml };
})(window);
