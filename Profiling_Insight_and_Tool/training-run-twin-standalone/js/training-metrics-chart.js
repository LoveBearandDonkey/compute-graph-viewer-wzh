/* ============================================================
   window.PtoTrainingMetricsChart — 训练指标折线图（纯原生、无依赖、自绘 SVG）
   抽自 component-preview.html 的提案 pattern；scope: .pto-tmchart*
   render(container, spec) -> controller
   spec = { steps[], series[{id,label,key,colorVar,axis,emphasis,connectNulls}], data{key:[]},
            anomalies[{step,seriesId}], referenceLines[{value,label,color,axis}], interestWindow, cursor,
            onBrush(win), onCursorHover(step), options, legend }
   controller = { setInterestWindow, setCursor, setData, setAnomalyVisible, destroy }
   ============================================================ */
(function (global) {
  'use strict';
  const SVGNS = 'http://www.w3.org/2000/svg';
  const el = (n, a) => { const e = document.createElementNS(SVGNS, n); if (a) for (const k in a) e.setAttribute(k, a[k]); return e; };
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  let clipSeq = 0;  // clipPath id 计数：同页多图共存时 id 不能撞

  function render(container, spec) {
    const host = typeof container === 'string' ? document.querySelector(container) : container;
    host.classList.add('pto-tmchart');
    host.innerHTML = '';
    /* 悬浮气泡浮层（body 级，见 acquireTip）。同一个 host 会被反复 render（resize/tick 重建图表）而
       未必走 destroy()，故这里认领上一次留下的那个浮层，避免一张卡在 body 上堆出一串孤儿气泡。 */
    let hostTip = host.__ptoTmchartTip || null;
    if (hostTip) hostTip.hidden = true;

    const opt = Object.assign({
      width: 960, height: 300,
      pad: { t: 14, r: 48, b: 26, l: 48 },
      compact: false,
      xTicks: 2,  // x 轴刻度档数（默认 2 = 仅首尾端点，保持原行为）
    }, spec.options || {});
    if (opt.compact) { opt.height = 120; opt.pad = { t: 10, r: 12, b: 20, l: 38 }; }

    const steps = spec.steps;
    const x0 = steps[0], x1 = steps[steps.length - 1];
    const W = opt.width, H = opt.height, P = opt.pad;
    const plotW = W - P.l - P.r, plotH = H - P.t - P.b;

    // 轴域：按 axis 分组自适应
    const axes = {};
    spec.series.forEach(s => {
      const vals = spec.data[s.key].filter(v => v != null && isFinite(v));
      const ax = s.axis || 'left';
      const a = axes[ax] || (axes[ax] = { min: Infinity, max: -Infinity });
      a.min = Math.min(a.min, ...vals); a.max = Math.max(a.max, ...vals);
    });
    Object.values(axes).forEach(a => { if (a.min === a.max) { a.max += 1; } const pad = (a.max - a.min) * 0.08; a.min -= pad; a.max += pad; });
    // spec.yDomain = { left: [min, max] }：调用方显式指定轴域，覆盖上面的自适应结果。
    // 用于指标含极端离群段（如 MFU/显存在事故步跌到 0）时，避免整条正常波动被压成顶部一条窄带；
    // 落在域外的线段由 gLines 的 clip 裁到画面外。
    if (spec.yDomain) {
      Object.keys(spec.yDomain).forEach(ax => {
        const d = spec.yDomain[ax];
        if (!d || !axes[ax]) return;
        const lo = Array.isArray(d) ? d[0] : d.min, hi = Array.isArray(d) ? d[1] : d.max;
        if (isFinite(lo) && isFinite(hi) && hi > lo) axes[ax] = { min: lo, max: hi };
      });
    }

    const mapX = (step) => P.l + ((step - x0) / (x1 - x0 || 1)) * plotW;
    const mapY = (val, ax) => { const a = axes[ax || 'left']; return P.t + plotH - ((val - a.min) / (a.max - a.min || 1)) * plotH; };

    const svg = el('svg', { class: 'pto-tmchart__svg', viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: 'none', role: 'img' });
    host.appendChild(svg);

    // 网格 + 左轴刻度
    const gGrid = el('g', { class: 'pto-tmchart__grid' });
    const rows = opt.compact ? 2 : 4;
    for (let i = 0; i <= rows; i++) {
      const y = P.t + (plotH / rows) * i;
      gGrid.appendChild(el('line', { x1: P.l, y1: y, x2: P.l + plotW, y2: y }));
    }
    svg.appendChild(gGrid);

    // 水平参考线（阈值/健康基线等）。标签贴在绘图区右侧，避免占用左轴刻度空间。
    const gReference = el('g', { class: 'pto-tmchart__reference-lines' });
    (spec.referenceLines || []).forEach(r => {
      const ax = r.axis || 'left';
      const a = axes[ax];
      if (!a || !isFinite(r.value) || r.value < a.min || r.value > a.max) return;
      const y = mapY(r.value, ax);
      const color = r.color || '#ea580c';
      const line = el('line', {
        x1: P.l, y1: y, x2: P.l + plotW, y2: y,
        stroke: color, 'stroke-width': r.width || 1,
        'stroke-dasharray': r.dash || '4 3',
      });
      line.setAttribute('opacity', r.opacity == null ? '0.9' : String(r.opacity));
      gReference.appendChild(line);
      if (r.label) {
        const t = el('text', {
          x: P.l + plotW - 3, y: y - 4, 'text-anchor': 'end',
          fill: color, 'font-size': '9', 'font-weight': '600',
        });
        t.textContent = r.label;
        gReference.appendChild(t);
      }
    });
    svg.appendChild(gReference);

    // 左轴数值
    const gAxis = el('g', { class: 'pto-tmchart__axis' });
    const la = axes.left;
    if (la) for (let i = 0; i <= rows; i++) {
      const v = la.max - ((la.max - la.min) / rows) * i;
      const y = P.t + (plotH / rows) * i;
      const t = el('text', { x: P.l - 6, y: y + 3, 'text-anchor': 'end' });
      const tv = fmt(v);
      t.textContent = tv;
      if (tv === 'NaN' || tv === 'Infinity' || tv === 'inf') t.setAttribute('class', 'pto-tmchart__axis pto-tmchart__val--danger');
      else t.setAttribute('class', 'pto-tmchart__axis');
      gAxis.appendChild(t);
    }
    // 右轴数值
    const ra = axes.right;
    if (ra) for (let i = 0; i <= rows; i++) {
      const v = ra.max - ((ra.max - ra.min) / rows) * i;
      const y = P.t + (plotH / rows) * i;
      const t = el('text', { x: P.l + plotW + 6, y: y + 3, 'text-anchor': 'start' });
      const tv2 = fmt(v);
      t.textContent = tv2;
      if (tv2 === 'NaN' || tv2 === 'Infinity' || tv2 === 'inf') t.setAttribute('class', 'pto-tmchart__axis pto-tmchart__val--danger');
      else t.setAttribute('class', 'pto-tmchart__axis');
      gAxis.appendChild(t);
    }
    // x 轴刻度（opt.xTicks 档，默认 2 = 仅首尾端点）
    const xTicks = Math.max(2, opt.xTicks | 0 || 2);
    for (let i = 0; i < xTicks; i++) {
      const frac = i / (xTicks - 1);
      const v = Math.round(x0 + frac * (x1 - x0));
      const x = P.l + frac * plotW;
      const anchor = i === 0 ? 'start' : i === xTicks - 1 ? 'end' : 'middle';
      const t = el('text', { x, y: H - 6, 'text-anchor': anchor }); t.textContent = v; gAxis.appendChild(t);
    }
    svg.appendChild(gAxis);

    // 区域底色（如路由坍缩区，spec.regions），置于最底层
    const gRegion = el('g'); svg.appendChild(gRegion);
    // 框选 band（先画，置于线下）
    const gBrush = el('g'); svg.appendChild(gBrush);
    // 异常 band + 折线层
    const gAnom = el('g'); svg.appendChild(gAnom);
    // 指定了 yDomain 时给折线层加裁剪：域外线段（如跌到 0 的事故段）不许画到坐标区外面去
    const gLines = el('g'); svg.appendChild(gLines);
    if (spec.yDomain) {
      const clipId = 'pto-tmchart-clip-' + (clipSeq += 1);
      const defs = el('defs');
      const cp = el('clipPath', { id: clipId });
      cp.appendChild(el('rect', { x: P.l, y: P.t, width: plotW, height: plotH }));
      defs.appendChild(cp);
      svg.insertBefore(defs, svg.firstChild);
      gLines.setAttribute('clip-path', 'url(#' + clipId + ')');
    }
    const gDots = el('g'); svg.appendChild(gDots);
    // 事故点等固定标注层：与下面的游标层(gCursor)分开——游标只在 hover 时跟随鼠标临时出现，
    // 这层是不受 hover 影响、常驻显示的红色虚线（spec.markerStep），互不干扰
    const gMarker = el('g'); svg.appendChild(gMarker);
    // 游标层
    const gCursor = el('g'); svg.appendChild(gCursor);

    // 折线（spec.smoothing>0：原曲线淡化、叠加 EMA 平滑曲线；默认 0 = 原行为）
    const smoothing = Math.max(0, Math.min(0.99, +spec.smoothing || 0));
    /* connect（= series.connectNulls）：空点不断线，直接连到下一个实测点。
       稀疏序列（如 val loss/acc 只在 epoch 采样点有值、其余为 null）必须开这个，
       否则每个点都是孤立的 moveto、一条线段都画不出来 —— 图上看着就是"这条曲线没画"。 */
    const buildPath = (arr, ax, stepped, connect) => {
      let d = '';
      let prev = null;
      steps.forEach((st, i) => {
        const v = arr[i];
        if (v == null || !isFinite(v)) { if (!connect) prev = null; return; }
        const x = mapX(st).toFixed(1), y = mapY(v, ax).toFixed(1);
        if (!prev) d += 'M ' + x + ' ' + y;
        else if (stepped) d += ' L ' + x + ' ' + prev.y + ' L ' + x + ' ' + y;
        else d += ' L ' + x + ' ' + y;
        prev = { x, y };
      });
      return d;
    };
    // 中心移动平均（零相位）+ 端点对称收缩：避免 EMA 单边滞后把上升/下降端点（如 2100）拉偏。
    // 在“非空点序列”上做平滑：train 为逐点；val 仅 epoch 点 → 端点（首/末实测点）保持原值。
    const smoothSeries = (arr, w) => {
      const n = arr.length;
      const out = new Array(n).fill(null);
      const idx = [], val = [];
      for (let i = 0; i < n; i++) { const v = arr[i]; if (v != null && isFinite(v)) { idx.push(i); val.push(v); } }
      const m = val.length;
      if (!m) return out;
      const R = Math.max(1, Math.round(w * 40));  // 平滑半径随 smoothing 增大
      for (let j = 0; j < m; j++) {
        const rr = Math.min(R, j, m - 1 - j);     // 边缘对称收缩：首/末点 rr=0 → 保持原值
        let sum = 0, cnt = 0;
        for (let k = j - rr; k <= j + rr; k++) { sum += val[k]; cnt++; }
        out[idx[j]] = sum / cnt;
      }
      return out;
    };
    spec.series.forEach(s => {
      const color = cssVar(s.colorVar) || '#888';
      const raw = spec.data[s.key];
      const cls = 'pto-tmchart__line' + (s.emphasis ? ' pto-tmchart__line--emph' : '');
      const cn = !!s.connectNulls;
      if (smoothing > 0) {
        const base = el('path', { class: 'pto-tmchart__line', d: buildPath(raw, s.axis, s.curve === 'step', cn), stroke: color });
        base.style.opacity = '0.22';  // 原走势淡化但不消失
        gLines.appendChild(base);
        gLines.appendChild(el('path', { class: cls, d: buildPath(smoothSeries(raw, smoothing), s.axis, s.curve === 'step', cn), stroke: color }));
      } else {
        gLines.appendChild(el('path', { class: cls, d: buildPath(raw, s.axis, s.curve === 'step', cn), stroke: color }));
      }
    });

    // 状态
    let anomalyVisible = spec.anomalies && spec.anomalies.length ? true : false;
    let brush = spec.interestWindow || null;
    let cursor = spec.cursor != null ? spec.cursor : null;
    let tooltipVisible = false;  // cursorTooltip 气泡仅在 hover 时显示（由 hover/leave 或外部 setTooltip 控制）

    function drawAnomaly() {
      gAnom.innerHTML = ''; gDots.innerHTML = '';
      if (!anomalyVisible || !spec.anomalies) return;
      // 把连续异常 step 聚成 band
      const aSteps = [...new Set(spec.anomalies.map(a => a.step))].sort((a, b) => a - b);
      if (aSteps.length) {
        const bx0 = mapX(Math.min(...aSteps)), bx1 = mapX(Math.max(...aSteps));
        gAnom.appendChild(el('rect', { class: 'pto-tmchart__anomaly-band', x: bx0, y: P.t, width: Math.max(2, bx1 - bx0), height: plotH }));
      }
      spec.anomalies.forEach(a => {
        const s = spec.series.find(ss => ss.id === a.seriesId); if (!s) return;
        const i = steps.indexOf(a.step); if (i < 0) return;
        const v = spec.data[s.key][i]; if (v == null) return;
        gDots.appendChild(el('circle', { class: 'pto-tmchart__anomaly-dot', cx: mapX(a.step), cy: mapY(v, s.axis), r: 3 }));
      });
    }
    function drawRegions() {
      gRegion.innerHTML = '';
      if (!spec.regions || !spec.regions.length) return;
      spec.regions.forEach(r => {
        const a = Math.max(x0, Math.min(r.start, r.end));
        const b = Math.min(x1, Math.max(r.start, r.end));
        if (b < a) return;
        gRegion.appendChild(el('rect', { class: 'pto-tmchart__anomaly-band', x: mapX(a), y: P.t, width: Math.max(1, mapX(b) - mapX(a)), height: plotH }));
      });
    }
    function drawMarker() {
      gMarker.innerHTML = '';
      if (spec.markerStep == null) return;
      if (spec.markerStep < x0 || spec.markerStep > x1) return;   // 超出当前窗口范围就不画，避免画到坐标区外
      const mx = mapX(spec.markerStep);
      gMarker.appendChild(el('line', { class: 'pto-tmchart__marker', x1: mx, y1: P.t, x2: mx, y2: P.t + plotH }));
    }
    function drawBrush() {
      gBrush.innerHTML = '';
      if (!brush) return;
      const bx0 = mapX(Math.min(brush.start, brush.end)), bx1 = mapX(Math.max(brush.start, brush.end));
      gBrush.appendChild(el('rect', { class: 'pto-tmchart__brush', x: bx0, y: P.t, width: Math.max(1, bx1 - bx0), height: plotH }));
      gBrush.appendChild(el('line', { class: 'pto-tmchart__brush-edge', x1: bx0, y1: P.t, x2: bx0, y2: P.t + plotH }));
      gBrush.appendChild(el('line', { class: 'pto-tmchart__brush-edge', x1: bx1, y1: P.t, x2: bx1, y2: P.t + plotH }));
    }
    function drawCursor() {
      gCursor.innerHTML = '';
      if (cursor == null) { hideTip(); return; }
      // spec.cursorTooltip 图表：定位轴虚线随气泡一起只在 hover 时显示；鼠标移开(tooltipVisible=false)
      // 整组都不画，避免移出后留下一条不再跟随鼠标的虚线残影
      if (spec.cursorTooltip && !tooltipVisible) { hideTip(); return; }
      const cx = mapX(cursor);
      gCursor.appendChild(el('line', { class: 'pto-tmchart__cursor', x1: cx, y1: P.t, x2: cx, y2: P.t + plotH }));
      if (opt.compact) { hideTip(); return; }
      // spec.cursorTooltip：在定位轴旁显示 step + 各序列具体数值的气泡（默认关闭，保持原 Step 胶囊）
      if (spec.cursorTooltip) { drawCursorBubble(cx); return; }
      hideTip();
      const label = 'Step ' + cursor;
      const w = 26 + label.length * 5.4;
      const chipX = Math.min(Math.max(cx - w / 2, P.l), P.l + plotW - w);
      gCursor.appendChild(el('rect', { class: 'pto-tmchart__cursor-chip', x: chipX, y: P.t - 1, width: w, height: 16, rx: 4 }));
      const t = el('text', { class: 'pto-tmchart__cursor-chip-text', x: chipX + w / 2, y: P.t + 10, 'text-anchor': 'middle' }); t.textContent = label;
      gCursor.appendChild(t);
    }
    /* 悬浮气泡不再画进 SVG，改为挂在 <body> 上的浮层（见 hostTip / .pto-tmchart-tip）：
       - 卡片(.twin-accuracy-metric-card 等)是 overflow:hidden 的，画在 SVG 里的气泡在窄卡上会被裁掉；
       - 画在 SVG 里时气泡宽度一旦超过绘图区就只能死钉在左边(bx 被夹到 P.l)，一直挡着背后的曲线。
       浮层按当前游标的屏幕 x 定位 → 天然跟着鼠标左右走，越界时只翻边不夹死。 */
    function drawCursorBubble(cx) {
      const i = steps.indexOf(cursor);
      if (i < 0) { hideTip(); return; }
      const fv = spec.formatValue || ((v) => fmt(v));
      const head = spec.formatCursor ? spec.formatCursor(cursor) : ('Step ' + cursor);
      const rows = [];
      spec.series.forEach(s => {
        let v = spec.data[s.key][i];
        // tipCarryForward：当前点为空（如 val 仅 epoch 边界有值）时，回溯取上一个非空值（上一 epoch 的值）
        if ((v == null || !isFinite(v)) && spec.tipCarryForward) {
          for (let j = i - 1; j >= 0; j--) { const pv = spec.data[s.key][j]; if (pv != null && isFinite(pv)) { v = pv; break; } }
        }
        if (v == null || !isFinite(v)) return;
        rows.push({ label: s.label, val: String(fv(v, s)), color: cssVar(s.colorVar) || '#888' });
      });
      // 命中带 label 的区域（如路由坍缩）→ 气泡右上角角标
      let tag = null;
      if (spec.regions) { const r = spec.regions.find(r => r.label && cursor >= Math.min(r.start, r.end) && cursor <= Math.max(r.start, r.end)); if (r) tag = r.label; }
      const tip = acquireTip();
      if (!tip) return;
      tip.innerHTML =
        '<div class="pto-tmchart-tip__head"><span class="pto-tmchart-tip__step">' + esc(head) + '</span>'
        + (tag ? '<span class="pto-tmchart-tip__tag">' + esc(tag) + '</span>' : '') + '</div>'
        + rows.map(r => '<div class="pto-tmchart-tip__row"><i style="background:' + esc(r.color) + '"></i>'
          + '<span class="pto-tmchart-tip__label">' + esc(r.label) + '</span>'
          + '<span class="pto-tmchart-tip__val">' + esc(r.val) + '</span></div>').join('');
      placeTip(cx);
    }
    // 按游标在屏幕上的位置摆放浮层：x 跟游标（右侧优先，右边放不下就翻到左侧），
    // y 贴住本图绘图区顶部（各图各自锚定，联动时 6 张图的气泡不会叠成一坨）；最后夹进视口，避免被窗口边裁掉。
    function placeTip(cx) {
      const tip = hostTip;
      if (!tip) return;
      const r = svg.getBoundingClientRect();
      // 图表本身不可见（卡片被隐藏、或滚出视口）时不摆气泡：联动会给同组每张图都下发游标，
      // 浮层脱离了容器裁切，不自己判一下就会在页面上飘出没有对应图表的孤儿气泡。
      const offscreen = r.bottom <= 0 || r.top >= window.innerHeight || r.right <= 0 || r.left >= window.innerWidth;
      if (!r.width || !r.height || offscreen) { hideTip(); return; }
      tip.hidden = false;
      const sx = r.width / W, sy = r.height / H;
      const anchorX = r.left + cx * sx;
      const tw = tip.offsetWidth, th = tip.offsetHeight;
      const GAP = 10, EDGE = 4;
      let left = anchorX + GAP;
      if (left + tw > window.innerWidth - EDGE) left = anchorX - GAP - tw;
      left = Math.max(EDGE, Math.min(left, window.innerWidth - tw - EDGE));
      let top = r.top + (P.t + 2) * sy;
      top = Math.max(EDGE, Math.min(top, window.innerHeight - th - EDGE));
      tip.style.transform = 'translate(' + Math.round(left) + 'px,' + Math.round(top) + 'px)';
    }
    function acquireTip() {
      // 图表容器已从文档里摘掉（卡片被重建/切页）时不再新建浮层，顺手把旧的收掉
      if (!host.isConnected) { destroyTip(); return null; }
      if (!hostTip) {
        hostTip = document.createElement('div');
        hostTip.className = 'pto-tmchart-tip';
        hostTip.hidden = true;
        document.body.appendChild(hostTip);
        host.__ptoTmchartTip = hostTip;
      }
      return hostTip;
    }
    function hideTip() { if (hostTip) hostTip.hidden = true; }
    function destroyTip() {
      if (hostTip) { hostTip.remove(); hostTip = null; }
      if (host.__ptoTmchartTip) delete host.__ptoTmchartTip;
    }

    // 交互 hit-rect
    const hit = el('rect', { class: 'pto-tmchart__plot-hit', x: P.l, y: P.t, width: plotW, height: plotH });
    svg.appendChild(hit);
    const stepFromClient = (clientX) => {
      const r = svg.getBoundingClientRect();
      const vx = (clientX - r.left) / r.width * W;
      const frac = Math.min(1, Math.max(0, (vx - P.l) / plotW));
      const target = x0 + frac * (x1 - x0);
      // 吸附到最近的真实采样点（支持非连续 step，如时间轴每 5s；连续整数 step 时等价于四舍五入）
      let best = steps[0], bd = Infinity;
      for (const s of steps) { const d = Math.abs(s - target); if (d < bd) { bd = d; best = s; } }
      return best;
    };
    // hover（游标 + 气泡）与 brush（框选）解耦：onBrush:false 只关掉框选，hover 仍可用
    if (!opt.compact) {
      const brushEnabled = spec.onBrush !== false;
      let dragging = false, anchor = null;
      hit.addEventListener('pointerdown', (e) => { if (!brushEnabled) return; dragging = true; anchor = stepFromClient(e.clientX); hit.setPointerCapture(e.pointerId); brush = { start: anchor, end: anchor }; drawBrush(); });
      hit.addEventListener('pointermove', (e) => {
        if (dragging) { brush = { start: anchor, end: stepFromClient(e.clientX) }; drawBrush(); }
        else { cursor = stepFromClient(e.clientX); tooltipVisible = true; drawCursor(); if (spec.onCursorHover) spec.onCursorHover(cursor); }
      });
      hit.addEventListener('pointerup', (e) => { if (!brushEnabled) return; dragging = false; if (brush && Math.abs(brush.end - brush.start) >= 1 && spec.onBrush) spec.onBrush({ start: Math.min(brush.start, brush.end), end: Math.max(brush.start, brush.end) }); else { brush = null; drawBrush(); } });
      hit.addEventListener('pointerleave', () => { if (dragging) return; tooltipVisible = false; drawCursor(); if (spec.onCursorLeave) spec.onCursorLeave(); });
    }

    drawRegions(); drawAnomaly(); drawBrush(); drawMarker(); drawCursor();

    // legend
    if (spec.legend !== false && !opt.compact) {
      const legend = document.createElement('div'); legend.className = 'pto-tmchart__legend';
      spec.series.forEach(s => {
        const item = document.createElement('span'); item.className = 'pto-tmchart__legend-item';
        const sw = document.createElement('span'); sw.className = 'pto-tmchart__legend-swatch'; sw.style.background = cssVar(s.colorVar);
        item.appendChild(sw); item.appendChild(document.createTextNode(s.label + (s.axis === 'right' ? ' (右轴)' : '')));
        legend.appendChild(item);
      });
      host.appendChild(legend);
    }

    return {
      setInterestWindow(win) { brush = win; drawBrush(); },
      setCursor(step) { cursor = step; drawCursor(); },
      setTooltip(v) { tooltipVisible = !!v; drawCursor(); },
      setAnomalyVisible(v) { anomalyVisible = v; drawAnomaly(); },
      destroy() { destroyTip(); host.innerHTML = ''; },
    };
  }
  function fmt(v) { const a = Math.abs(v); if (a >= 1000) return (v / 1000).toFixed(1) + 'k'; if (a >= 10) return v.toFixed(0); if (a >= 1) return v.toFixed(1); return v.toFixed(2); }

  global.PtoTrainingMetricsChart = { render };
})(window);
