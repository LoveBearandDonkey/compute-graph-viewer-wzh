/*
 * 问题七 · HiF8 精度诊断（定位链嵌入版）
 * ------------------------------------------------------------
 * 把 hif8-precision-workbench-V3.html 的「概览 / 张量分布 / 量化误差 / 误差传播 / 根因分析」
 * 五个页签 100% 搬进「问题诊断」定位链的五个节里。数据模型 / 画布渲染直接沿用工作台，
 * 去掉资源管理器 / inspector 工具壳；顶部「训练步回放」scrubber 保留在概览节，拖动/播放时
 * 驱动 cur 并重绘全部五节图表，回放量化误差的传播与累积（默认停在末步 step 10000）。
 *
 * 用法：training-run-twin.js 里
 *   locateChains["hif8-precision"] = window.PtoHif8Case7.chain();
 * 面板挂载后调用 window.PtoHif8Case7.renderAll() 绘制画布并绑定交互。
 */
window.PtoHif8Case7 = (function () {
  "use strict";

  /* ---------- seeded RNG（与工作台同种子，保证形态一致） ---------- */
  let _s = 1337;
  function rnd() { _s = (_s * 1103515245 + 12345) & 0x7fffffff; return _s / 0x7fffffff; }
  function gauss() { return (rnd() + rnd() + rnd() + rnd() - 2) / 2; }
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const MONO = "'Cascadia Code','Fira Code',ui-monospace,'SF Mono',Menlo,Consolas,monospace";

  /* ---------- 模型 / 数据生成（openPangu 2.0 Flash 算子映射） ---------- */
  const N = 200;                // 采样点（每 50 真实 step → 10000）
  const stepOf = i => i * 50;
  const DIV = 63;               // 发散采样索引（~step 3150）
  const BLOCKS = 6;
  let cur = N - 1;             // 当前训练步索引，由概览页签的「训练步回放」scrubber 驱动（默认停在末步）

  const layers = [];
  layers.push({ name: 'token_embedding', type: 'io', depth: 0 });
  for (let b = 0; b < BLOCKS; b++) {
    const d = 1 + b;
    const isDense = b === 0; // 首块 = Dense L0，其余 = MoE L4+

    // 注意力线性算子（sparse_mla_attention · MLA 压缩投影）
    ['q_a_proj', 'q_b_proj', 'kv_a_proj', 'kv_b_proj', 'o_proj']
      .forEach(op => {
        const t = op.startsWith('q_') || op.startsWith('kv_') || op === 'o_proj' ? 'attn' : 'attn';
        layers.push({ name: `L${b}.${op}`, type: 'attn', depth: d, block: b, op });
      });

    // 注意力归一化
    layers.push({ name: `L${b}.q_a_norm`, type: 'norm', depth: d, block: b });
    layers.push({ name: `L${b}.kv_a_norm`, type: 'norm', depth: d, block: b });
    layers.push({ name: `L${b}.post_attention_norm`, type: 'norm', depth: d, block: b });

    if (isDense) {
      // Dense FFN（L0~L3）
      ['dense_gate_up', 'dense_down']
        .forEach(op => {
          layers.push({ name: `L${b}.${op}`, type: 'mlp', depth: d, block: b, op });
        });
    } else {
      // MoE FFN（L4~L47）
      layers.push({ name: `L${b}.router_gate`, type: 'mlp', depth: d, block: b });
      layers.push({ name: `L${b}.routed_expert_bank`, type: 'mlp', depth: d, block: b });
    }
    layers.push({ name: `L${b}.input_layernorm`, type: 'norm', depth: d, block: b });
  }
  layers.push({ name: 'final_norm', type: 'norm', depth: BLOCKS + 1 });
  layers.push({ name: 'lm_head', type: 'io', depth: BLOCKS + 1 });

  const CULPRIT = { 'L4.q_b_proj': 1.0, 'L3.o_proj': 0.72, 'L5.router_gate': 0.45 };

  layers.forEach(L => {
    const base = L.type === 'norm' ? 42 : L.type === 'io' ? 38 : 35 + gauss() * 2;
    const cul = CULPRIT[L.name] || 0;
    L.sqnr = []; L.over = []; L.under = []; L.cos = []; L.mse = []; L.maxerr = []; L.util = [];
    for (let i = 0; i < N; i++) {
      let sq = base - 1.5 * Math.exp(-i / 40) + gauss() * 0.6;
      let ov = clamp(0.002 + Math.abs(gauss()) * 0.004 + L.depth * 0.0004, 0, 1);
      if (cul > 0 && i > DIV - 4) {
        const prog = clamp((i - (DIV - 4)) / (N - DIV), 0, 1);
        sq -= cul * (6 + 12 * prog) + gauss() * 0.8;
        ov += cul * (0.01 + 0.09 * prog);
      }
      sq = clamp(sq, 14, 48); ov = clamp(ov, 0, 0.6);
      L.sqnr.push(sq);
      L.over.push(ov);
      L.under.push(clamp(0.01 + L.depth * 0.001 + Math.abs(gauss()) * 0.008, 0, 0.3));
      L.cos.push(clamp(1 - Math.pow(10, -sq / 20) * 3.2, 0.90, 0.99999));
      L.mse.push(Math.pow(10, -sq / 10) * (0.5 + Math.abs(gauss()) * 0.1));
      L.maxerr.push(Math.pow(10, -sq / 20) * (1.6 + Math.abs(gauss()) * 0.3));
      L.util.push(clamp(0.55 + L.depth * 0.02 - (cul > 0 && i > DIV ? cul * 0.15 : 0) + gauss() * 0.03, 0.2, 0.98));
    }
    L.sens = cul > 0 ? cul * (0.9 + rnd() * 0.2) : (0.02 + rnd() * 0.05) * (L.type === 'mlp' ? 1.4 : 1);
  });
  const sSum = layers.reduce((a, L) => a + L.sens, 0);
  layers.forEach(L => L.sensPct = L.sens / sSum);

  const lossRef = [], lossMx = [], lossHif = [], dloss = [];
  const spikes = [];
  const logitUnif = [];
  for (let i = 0; i < N; i++) {
    const ref = 4.6 * Math.exp(-i / 32) + 1.55 + gauss() * 0.02;
    lossRef.push(ref);
    lossMx.push(ref + gauss() * 0.015);
    let dv = 0;
    if (i >= DIV) { const p = (i - DIV) / (N - DIV); dv = 0.02 + 0.55 * p * p + (0.03 * Math.sin(i / 3)) * p; }
    let sp = 0;
    if (i < DIV && (i === 14 || i === 27 || i === 41)) { sp = 0.18 + rnd() * 0.12; spikes.push(i); }
    lossHif.push(ref + dv + sp + (i >= DIV ? gauss() * 0.02 : gauss() * 0.008));
    dloss.push(lossHif[i] - lossRef[i]);
    let u = 0.86 + gauss() * 0.02 - (i >= DIV ? 0.15 * (i - DIV) / (N - DIV) : 0);
    if (sp > 0) u -= 0.22;
    logitUnif.push(clamp(u, 0.35, 0.95));
  }

  const meanSqnr = [];
  for (let i = 0; i < N; i++) { let s = 0; layers.forEach(L => s += L.sqnr[i]); meanSqnr.push(s / layers.length); }

  /* ---------- 交互状态 ---------- */
  let selLayer = layers.find(l => l.name === 'L4.q_b_proj');
  let tensorType = 'weight';
  let sortKey = 'sqnr', sortAsc = true;

  const sevOf = sq => sq < 26 ? 'crit' : sq < 32 ? 'warn' : 'ok';
  const sevColor = { ok: '#16a34a', warn: '#ea580c', crit: '#dc2626' };

  const $ = id => document.getElementById(id);

  /* ---------- 画布助手 ---------- */
  function fit(cv, cssH) {
    if (!cv) return null;
    cv.style.width = '100%'; cv.style.height = cssH + 'px';
    const w = cv.clientWidth || cv.getBoundingClientRect().width || 600;
    const dpr = window.devicePixelRatio || 1;
    cv.width = Math.max(1, Math.round(w * dpr));
    cv.height = Math.max(1, Math.round(cssH * dpr));
    const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { c, w, h: cssH };
  }
  function grid(c, w, h, rows) {
    c.strokeStyle = '#e5e7eb'; c.lineWidth = 1;
    for (let i = 0; i <= rows; i++) { const y = h * i / rows + .5; c.beginPath(); c.moveTo(0, y); c.lineTo(w, y); c.stroke(); }
  }
  // 圆角柱：Canvas roundRect 兼容回退
  function drawRoundedBar(c, x, y, w, h, r) {
    if (!c.roundRect) {
      c.beginPath();
      c.moveTo(x + r, y);
      c.lineTo(x + w - r, y);
      c.arcTo(x + w, y, x + w, y + r, r);
      c.lineTo(x + w, y + h);
      c.lineTo(x, y + h);
      c.lineTo(x, y + r);
      c.arcTo(x, y, x + r, y, r);
      c.closePath();
      c.fill();
      return;
    }
    c.beginPath();
    c.roundRect(x, y, w, h, [r, r, 0, 0]);
    c.fill();
  }

  /* ============ 概览 ============ */
  function renderKpis() {
    const host = $('c7kpis'); if (!host) return;
    const dl = dloss[cur], ms = meanSqnr[cur];
    const totOver = layers.reduce((a, L) => a + L.over[cur], 0) / layers.length;
    const totUnder = layers.reduce((a, L) => a + L.under[cur], 0) / layers.length;
    const health = clamp(100 - (dl > 0 ? dl * 90 : 0) - (35 - ms) * 3 - totOver * 120, 0, 100);
    const hSev = health > 70 ? 'ok' : health > 45 ? 'warn' : 'crit';
    const cards = [
      { lab: '精度健康分', val: health.toFixed(0), unit: '/100', sev: hSev, delta: cur > DIV ? '▼ 训练已发散' : '▲ 稳定' },
      { lab: '首次发散步', val: '3150', unit: 'step', sev: 'crit', delta: '|Δloss|>0.05' },
      { lab: '均值 SQNR', val: ms.toFixed(1), unit: 'dB', sev: ms > 32 ? 'ok' : ms > 28 ? 'warn' : 'crit', delta: (ms - 35).toFixed(1) + ' vs 目标' },
      { lab: '平均溢出率', val: (totOver * 100).toFixed(2), unit: '%', sev: totOver < 0.01 ? 'ok' : totOver < 0.03 ? 'warn' : 'crit', delta: cur > DIV ? '↑ 上升中' : '平稳' },
      { lab: '平均下溢率', val: (totUnder * 100).toFixed(2), unit: '%', sev: totUnder < 0.03 ? 'ok' : 'warn', delta: '→0 截断' },
    ];
    host.innerHTML = cards.map(k => `
      <div class="h8-kpi h8-k-${k.sev}">
        <div class="h8-lab">${k.lab}</div>
        <div class="h8-val">${k.val}<small>${k.unit}</small></div>
        <div class="h8-delta ${k.sev === 'crit' ? 'h8-up' : 'h8-down'}">${k.delta}</div>
      </div>`).join('');
  }
  function renderFmtStatus() {
    const host = $('c7fmt'); if (!host) return;
    const fmts = [
      { n: 'BF16', fmt: '1-8-7', st: 'ok', note: '参考基线 · 收敛正常' },
      { n: 'MXFP8', fmt: 'E4M3+block scale', st: 'ok', note: '收敛正常 · 与基线重合' },
      {
        n: 'HiF8 / FP8（候选）', fmt: '1-3-3-1 锥形', st: cur >= DIV ? 'crit' : 'warn',
        note: cur >= DIV ? ('发散 @ step ' + stepOf(DIV) + ' · 需诊断') : '早期 ' + spikes.length + ' 次 loss spike'
      },
    ];
    const ic = { ok: '✔', warn: '⚠', crit: '✘' };
    host.innerHTML = fmts.map(f => `
      <div style="flex:1;min-width:190px;background:var(--h8-inset);border:1px solid var(--h8-stroke);
        border-left:3px solid ${sevColor[f.st]};border-radius:8px;padding:10px 12px">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <b style="font-family:${MONO};font-size:12.5px">${f.n}</b>
          <span style="color:${sevColor[f.st]};font-family:${MONO};font-size:13px">${ic[f.st]}</span>
        </div>
        <div style="font-family:${MONO};font-size:10px;color:var(--h8-dim);margin-top:3px">${f.fmt}</div>
        <div style="font-size:11px;color:var(--h8-mute);margin-top:5px">${f.note}</div>
      </div>`).join('');
  }
  function renderLoss() {
    const r = fit($('c7loss'), 240); if (!r) return; const { c, w, h } = r; c.clearRect(0, 0, w, h);
    const pad = 30; const lo = 1.4, hi = 3.2;
    const X = i => pad + (w - pad - 8) * i / (N - 1), Y = v => h - 18 - (h - 30) * (v - lo) / (hi - lo);
    grid(c, w, h, 4);
    c.fillStyle = 'rgba(220,38,38,.08)'; c.fillRect(X(DIV), 0, w - X(DIV) - 8, h - 18);
    c.strokeStyle = '#dc2626'; c.setLineDash([4, 4]); c.beginPath(); c.moveTo(X(DIV), 0); c.lineTo(X(DIV), h - 18); c.stroke(); c.setLineDash([]);
    const line = (arr, col, wid, dash) => {
      c.strokeStyle = col; c.lineWidth = wid; c.setLineDash(dash || []); c.beginPath();
      arr.forEach((v, i) => { const x = X(i), y = Y(v); i ? c.lineTo(x, y) : c.moveTo(x, y); }); c.stroke(); c.setLineDash([]);
    };
    line(lossMx, '#16a34a', 1.3, [5, 3]);
    line(lossRef, '#0ea5e9', 1.6);
    line(lossHif, '#3b6fe0', 1.8);
    spikes.forEach(si => {
      c.fillStyle = '#ea580c'; c.beginPath(); c.arc(X(si), Y(lossHif[si]), 3, 0, 7); c.fill();
      c.fillStyle = '#ea580c'; c.font = '9px ' + MONO; c.fillText('spike', X(si) - 11, Y(lossHif[si]) - 7);
    });
    c.strokeStyle = 'rgba(15,23,42,.45)'; c.globalAlpha = .5; c.beginPath(); c.moveTo(X(cur), 0); c.lineTo(X(cur), h - 18); c.stroke(); c.globalAlpha = 1;
    c.fillStyle = '#3b6fe0'; c.beginPath(); c.arc(X(cur), Y(lossHif[cur]), 3.5, 0, 7); c.fill();
    c.fillStyle = '#94a3b8'; c.font = '10px ' + MONO;
    c.fillText('loss', 4, 12); c.fillText('step ' + stepOf(cur), X(cur) - 40, h - 4);
  }
  function renderDelta() {
    const r = fit($('c7delta'), 176); if (!r) return; const { c, w, h } = r; c.clearRect(0, 0, w, h);
    const pad = 8; const mx = Math.max(...dloss) * 1.1;
    const X = i => pad + (w - pad * 2) * i / (N - 1), Y = v => h - 16 - (h - 24) * v / mx;
    grid(c, w, h, 4);
    c.strokeStyle = '#ea580c'; c.setLineDash([3, 4]); c.beginPath(); c.moveTo(pad, Y(0.05)); c.lineTo(w - pad, Y(0.05)); c.stroke(); c.setLineDash([]);
    c.fillStyle = '#ea580c'; c.font = '9px ' + MONO; c.fillText('阈值 0.05', w - 58, Y(0.05) - 4);
    const g = c.createLinearGradient(0, 0, 0, h); g.addColorStop(0, 'rgba(220,38,38,.35)'); g.addColorStop(1, 'rgba(220,38,38,0)');
    c.beginPath(); c.moveTo(X(0), Y(0)); dloss.forEach((v, i) => c.lineTo(X(i), Y(v))); c.lineTo(X(N - 1), Y(0)); c.closePath(); c.fillStyle = g; c.fill();
    c.strokeStyle = '#dc2626'; c.lineWidth = 1.6; c.beginPath(); dloss.forEach((v, i) => { i ? c.lineTo(X(i), Y(v)) : c.moveTo(X(i), Y(v)); }); c.stroke();
    c.strokeStyle = 'rgba(15,23,42,.45)'; c.globalAlpha = .4; c.beginPath(); c.moveTo(X(cur), 8); c.lineTo(X(cur), h - 16); c.stroke(); c.globalAlpha = 1;
    const el = $('c7ovDelta'); if (el) { el.textContent = dloss[cur].toFixed(4); el.style.color = dloss[cur] > 0.05 ? 'var(--h8-crit)' : 'var(--h8-ink)'; }
  }
  function renderLogit() {
    const r = fit($('c7logit'), 88); if (!r) return; const { c, w, h } = r; c.clearRect(0, 0, w, h);
    const pad = 8; const X = i => pad + (w - pad * 2) * i / (N - 1), Y = v => h - 14 - (h - 22) * (v - 0.3) / (0.95 - 0.3);
    grid(c, w, h, 3);
    c.strokeStyle = 'rgba(234,88,12,.5)'; c.setLineDash([3, 3]); c.beginPath(); c.moveTo(pad, Y(0.6)); c.lineTo(w - pad, Y(0.6)); c.stroke(); c.setLineDash([]);
    c.strokeStyle = '#3b6fe0'; c.lineWidth = 1.5; c.beginPath(); logitUnif.forEach((v, i) => { i ? c.lineTo(X(i), Y(v)) : c.moveTo(X(i), Y(v)); }); c.stroke();
    spikes.forEach(si => { c.fillStyle = '#ea580c'; c.beginPath(); c.arc(X(si), Y(logitUnif[si]), 2.5, 0, 7); c.fill(); });
    c.strokeStyle = 'rgba(15,23,42,.45)'; c.globalAlpha = .4; c.beginPath(); c.moveTo(X(cur), 8); c.lineTo(X(cur), h - 14); c.stroke(); c.globalAlpha = 1;
    const u = logitUnif[cur]; const el = $('c7ovUnif'); if (el) { el.textContent = u.toFixed(2); el.style.color = u < 0.6 ? 'var(--h8-warn)' : 'var(--h8-ink)'; }
  }
  const EVENTS = [
    { i: 14, sev: 'warn', t: 'FP8 loss spike（logit 打散度骤降）' },
    { i: 52, sev: 'warn', t: 'L4.q_b_proj 溢出率 >1%' },
    { i: 63, sev: 'crit', t: 'Δloss 越过阈值 0.05' },
    { i: 66, sev: 'crit', t: 'L4.q_b_proj SQNR<26dB' },
    { i: 78, sev: 'warn', t: 'L3.o_proj SQNR 塌陷' },
    { i: 120, sev: 'warn', t: 'L5.router_gate 溢出扩散' },
    { i: 170, sev: 'crit', t: '累积 Δloss 达 0.55' },
  ];
  function renderTimeline() {
    // 事件密集处（step 52/63/66/78）文字标签横向必然重叠，改为「贪心分行 + 引导线」布局：
    // 每条标签放到从坐标轴向上、不与同行既有标签相撞的最低一行，右溢出时向左夹紧。
    const c0 = $('c7timeline') && $('c7timeline').getContext('2d');
    if (c0) c0.font = '9.5px ' + MONO;                 // 先量文字宽度以决定行数 → 定高
    const ROW_H = 13, GAP = 8, PAD_TOP = 8, AXIS_PAD = 20;
    // 预排：用一个粗略宽度先算所需行数（真实宽度在下方重排，这里只为定 canvas 高度）
    let probeRows = 1;
    if (c0) {
      const wGuess = ($('c7timeline').clientWidth || 440);
      const Xg = i => 12 + (wGuess - 24) * i / (N - 1);
      const edges = [];
      EVENTS.forEach(e => { const x = Xg(e.i), tw = c0.measureText(e.t).width; let row = 0; while (edges[row] != null && x < edges[row] + GAP) row++; edges[row] = x + tw; if (row + 1 > probeRows) probeRows = row + 1; });
    }
    const cssH = PAD_TOP + probeRows * ROW_H + AXIS_PAD;
    const r = fit($('c7timeline'), cssH); if (!r) return; const { c, w, h } = r; c.clearRect(0, 0, w, h);
    const X = i => 12 + (w - 24) * i / (N - 1), axisY = h - 16;
    c.font = '9.5px ' + MONO; c.textBaseline = 'alphabetic';
    c.strokeStyle = '#e5e7eb'; c.lineWidth = 1; c.beginPath(); c.moveTo(12, axisY); c.lineTo(w - 12, axisY); c.stroke();
    const edges = [];
    EVENTS.forEach(e => {
      const x = X(e.i), tw = c.measureText(e.t).width;
      let row = 0; while (edges[row] != null && x < edges[row] + GAP) row++;
      edges[row] = x + tw;
      const on = e.i <= cur, col = sevColor[e.sev];
      const ly = axisY - AXIS_PAD - row * ROW_H + 4;   // 该行标签基线
      let lx = x + 5; if (lx + tw > w - 4) lx = w - 4 - tw; if (lx < 4) lx = 4;
      // 引导线：从坐标轴刻度上探到标签
      c.strokeStyle = col; c.globalAlpha = on ? .45 : .18; c.lineWidth = 1;
      c.beginPath(); c.moveTo(x, axisY - 6); c.lineTo(x, ly - 8); c.stroke();
      // 刻度 + 圆点
      c.globalAlpha = on ? 1 : .3; c.strokeStyle = col; c.lineWidth = 1.5;
      c.beginPath(); c.moveTo(x, axisY - 6); c.lineTo(x, axisY + 6); c.stroke();
      c.fillStyle = col; c.beginPath(); c.arc(x, axisY, 3, 0, 7); c.fill();
      // 标签
      c.fillStyle = on ? '#1e293b' : '#94a3b8'; c.fillText(e.t, lx, ly);
      c.globalAlpha = 1;
    });
    c.strokeStyle = 'rgba(15,23,42,.45)'; c.globalAlpha = .5; c.lineWidth = 1;
    c.beginPath(); c.moveTo(X(cur), 4); c.lineTo(X(cur), axisY + 8); c.stroke(); c.globalAlpha = 1;
  }

  /* ============ 张量分布 ============ */
  function histFor(L, step, type) {
    const cul = (CULPRIT[L.name] || 0);
    const spreadBase = type === 'grad' ? 0.6 : type === 'act' ? 1.1 : 0.8;
    const drift = cul > 0 && step > DIV ? cul * (step - DIV) / (N - DIV) : 0;
    const std = spreadBase * (1 + drift * 1.4);
    const clip = 448;
    const bins = 90, lohi = clip * 1.15, bw = 2 * lohi / bins;
    const raw = new Array(bins).fill(0), q = new Array(bins).fill(0);
    let over = 0, under = 0, tot = 6000;
    const reps = []; for (let e = -9; e <= 8.8; e += 0.34) { const m = Math.pow(2, e);[m, -m].forEach(v => { if (Math.abs(v) <= clip) reps.push(v); }); }
    reps.push(0); reps.sort((a, b) => a - b);
    const quant = v => { if (Math.abs(v) > clip) return v > 0 ? clip : -clip; let best = reps[0], bd = 1e9; for (const rp of reps) { const d = Math.abs(v - rp); if (d < bd) { bd = d; best = rp; } } return best; };
    const heavy = cul > 0 ? (0.12 * cul + cul * drift * 1.2) : 0.015;
    const tailP = clamp(0.006 + heavy * 0.13, 0, 0.4);
    const outSigma = 3 + heavy * 7;
    const asym = type === 'act' ? 0.32 : 0;
    const ovRate = cul > 0 ? clamp(L.over[step] * 0.9, 0, 0.2) : 0;
    const bulkSd = std * 0.289;
    let s1 = 0, s2 = 0, s3 = 0, s4 = 0, vmin = 1e9, vmax = -1e9;
    for (let n = 0; n < tot; n++) {
      let v = gauss() * std;
      if (rnd() < tailP) { const sign = (rnd() < 0.5 + asym * 0.5) ? 1 : -1; v += sign * bulkSd * outSigma * (0.6 + rnd() * 0.8); }
      s1 += v; s2 += v * v; s3 += v * v * v; s4 += v * v * v * v;
      if (v < vmin) vmin = v; if (v > vmax) vmax = v;
      const bi = Math.floor((v + lohi) / bw);
      if (bi >= 0 && bi < bins) raw[bi]++;
      const qv = quant(v); if (qv === 0 && v !== 0) under++;
      const qi = Math.floor((clamp(qv, -lohi + bw, lohi - bw) + lohi) / bw);
      if (qi >= 0 && qi < bins) q[qi]++;
    }
    const ovN = Math.round(tot * ovRate); over = ovN;
    for (let k = 0; k < ovN; k++) { const b = (rnd() < 0.5 + asym * 0.5) ? bins - 2 : 1; raw[b]++; q[b]++; }
    const m1 = s1 / tot, varr = Math.max(s2 / tot - m1 * m1, 1e-9), sd = Math.sqrt(varr);
    const skew = (s3 / tot - 3 * m1 * (s2 / tot) + 2 * m1 * m1 * m1) / (sd * sd * sd);
    const kurt = (s4 / tot - 4 * m1 * (s3 / tot) + 6 * m1 * m1 * (s2 / tot) - 3 * m1 * m1 * m1 * m1) / (varr * varr) - 3;
    let fit2, fitNote;
    if (kurt > 2.2) { fit2 = 't 分布'; fitNote = '重尾 · 离群值多'; }
    else if (Math.abs(skew) > 0.55) { fit2 = '贝塔分布'; fitNote = '明显偏态'; }
    else { fit2 = '正态分布'; fitNote = '近高斯 · 对称'; }
    const qScore = clamp(100 - Math.max(0, kurt) * 6 - ovRate * 100 * 3.2 - Math.abs(skew) * 14, 0, 100);
    return {
      raw, q, bins, lohi, bw, reps, clip, over: ovRate, under: under / tot, std,
      min: ovRate > 0 ? -clip * 1.04 : vmin, max: ovRate > 0 ? clip * 1.04 : vmax, skew, kurt, fit: fit2, fitNote, qScore
    };
  }
  function renderDist() {
    const L = selLayer;
    const dl = $('c7distLayer'); if (dl) dl.textContent = L.name + ' · ' + ({ weight: '权重', act: '激活', grad: '梯度' }[tensorType]);
    const H = histFor(L, cur, tensorType);
    const scale = (H.clip / (H.max * 1.2)).toFixed(1); const ds = $('c7distScale'); if (ds) ds.textContent = scale;
    const r = fit($('c7hist'), 300); if (!r) return; const { c, w, h } = r; c.clearRect(0, 0, w, h);
    const pad = 8, baseY = h - 22;
    const X = b => pad + (w - pad * 2) * b / H.bins;
    const mx = Math.max(...H.raw, ...H.q);
    const Y = v => baseY - (baseY - 10) * v / mx;

    // 细灰网格（参考 precision-debugger 的 subtle grid）
    c.strokeStyle = 'rgba(148,163,184,.18)'; c.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) { const y = 10 + (baseY - 10) * i / 4 + 0.5; c.beginPath(); c.moveTo(pad, y); c.lineTo(w - pad, y); c.stroke(); }

    const clipB0 = Math.floor((-H.clip + H.lohi) / H.bw), clipB1 = Math.floor((H.clip + H.lohi) / H.bw);

    // 溢出区红晕（参考设计系统 danger-tinted panel）
    c.fillStyle = 'rgba(220,38,38,.07)';
    c.fillRect(pad, 10, X(clipB0) - pad, baseY - 10);
    c.fillRect(X(clipB1), 10, w - pad - X(clipB1), baseY - 10);

    // HiF8 可表示栅格刻度（细灰线，只画到柱区上方 4px）
    c.strokeStyle = 'rgba(148,163,184,.32)'; c.lineWidth = 0.7;
    H.reps.forEach(rp => {
      if (Math.abs(rp) > H.lohi) return;
      const bx = X((rp + H.lohi) / H.bw);
      c.beginPath(); c.moveTo(bx, baseY); c.lineTo(bx, baseY - 4); c.stroke();
    });

    // 量化后柱状图（圆角顶、细间隙、design-system 蓝）
    const bwid = (w - pad * 2) / H.bins;
    const barGap = bwid * 0.12;
    const barW = bwid - barGap;
    const barR = Math.min(2, barW * 0.35);
    H.q.forEach((v, b) => {
      const bx = X(b) + barGap / 2;
      const bh = baseY - Y(v);
      if (bh < 1) return;
      // bar gradient: top=opaque blue, bottom=slightly lighter
      const grad = c.createLinearGradient(0, Y(v), 0, baseY);
      grad.addColorStop(0, 'rgba(59,110,224,.72)');
      grad.addColorStop(1, 'rgba(59,110,224,.38)');
      c.fillStyle = grad;
      drawRoundedBar(c, bx, Y(v), barW, bh, barR);
    });

    // 原始分布密度曲线（参考线）：半透明描边，无填充
    c.strokeStyle = 'rgba(14,165,233,.62)'; c.lineWidth = 1.5;
    c.beginPath();
    H.raw.forEach((v, b) => {
      const x = X(b) + bwid / 2, y = Y(v);
      b ? c.lineTo(x, y) : c.moveTo(x, y);
    });
    c.stroke();

    // FP8 截断边界（红色虚线，带半透明填充条）
    c.fillStyle = 'rgba(220,38,38,.12)';
    c.fillRect(X(clipB0) - 1.5, 10, 3, baseY - 10);
    c.fillRect(X(clipB1) - 1.5, 10, 3, baseY - 10);
    c.strokeStyle = '#dc2626'; c.lineWidth = 1; c.setLineDash([4, 3]);
    [clipB0, clipB1].forEach(b => {
      c.beginPath(); c.moveTo(X(b), 10); c.lineTo(X(b), baseY); c.stroke();
    });
    c.setLineDash([]);

    // 轴标签（muted mono，参考设计系统 label 规范）
    c.fillStyle = 'var(--foreground-muted, #94a3b8)'; c.font = '9.5px ' + MONO;
    c.fillText('−448', X(clipB0) - 14, h - 6);
    c.fillText('+448', X(clipB1) - 14, h - 6);
    c.fillText('0', X(H.bins / 2) - 3, h - 6);

    drawRangeBar(H);
    const sq = L.sqnr[cur];
    const set = (id, html) => { const e = $(id); if (e) e.innerHTML = html; };
    set('c7dMinMax', H.min.toFixed(2) + ' / ' + H.max.toFixed(2));
    set('c7dMeanStd', '0.00 / ' + H.std.toFixed(2));
    set('c7dOver', `<span style="color:${H.over > 0.01 ? 'var(--h8-crit)' : 'var(--h8-ink)'}">${(H.over * 100).toFixed(2)}%</span>`);
    set('c7dUnder', (H.under * 100).toFixed(2) + '%');
    set('c7dEnob', (sq / 6.02).toFixed(2) + ' bit');
    set('c7dSqnr', `<span style="color:${sevColor[sevOf(sq)]}">${sq.toFixed(1)} dB</span>`);
    const qs = H.qScore, qsev = qs > 70 ? 'ok' : qs > 45 ? 'warn' : 'crit';
    const qScoreEl = $('c7qScore'); if (qScoreEl) { qScoreEl.textContent = qs.toFixed(0); qScoreEl.style.color = sevColor[qsev]; }
    const bar = $('c7qScoreBar'); if (bar) { bar.style.width = qs + '%'; bar.style.background = `linear-gradient(90deg,${sevColor[qsev]}88,${sevColor[qsev]})`; }
    set('c7dFit', `<span style="color:${H.fit === '正态分布' ? 'var(--h8-ok)' : H.fit === 't 分布' ? 'var(--h8-crit)' : 'var(--h8-warn)'}">${H.fit}</span>`);
    set('c7dKurt', `<span style="color:${H.kurt > 2.5 ? 'var(--h8-crit)' : 'var(--h8-ink)'}">${H.kurt.toFixed(2)}</span>`);
    set('c7dSkew', H.skew.toFixed(2));
    set('c7dTail', `<span style="color:${H.over > 0.01 ? 'var(--h8-crit)' : 'var(--h8-ink)'}">${(H.over * 100).toFixed(2)}%</span>`);
    const fv = $('c7fitVerdict');
    if (fv) fv.innerHTML = qs > 70
      ? `拟合为<b style="color:var(--h8-ok)">${H.fit}</b>(${H.fitNote}),峰度接近 0、尾部轻,<b>适合直接量化</b>。`
      : qs > 45
        ? `拟合为<b style="color:var(--h8-warn)">${H.fit}</b>(${H.fitNote}),存在一定重尾/偏态,建议 <b>per-channel 缩放</b>后量化。`
        : `拟合为<b style="color:var(--h8-crit)">${H.fit}</b>(${H.fitNote}),峰度 ${H.kurt.toFixed(1)}、离群值多,<b>不适合直接量化</b> → 保留 BF16 或裁剪重尾离群值。`;
  }
  function drawRangeBar(H) {
    const r = fit($('c7range'), 52); if (!r) return; const { c, w, h } = r; c.clearRect(0, 0, w, h);
    const y = h / 2, barH = 16;
    // 浅灰轨道
    c.fillStyle = 'rgba(148,163,184,.16)'; drawRoundedBar(c, 0, y - barH / 2, w, barH, 6);
    const util = selLayer.util[cur];
    const uw = Math.max(4, w * util);
    const sev = util > 0.85 ? 'crit' : 'ok';
    const clr = sevColor[sev];
    const grad = c.createLinearGradient(0, 0, uw, 0);
    grad.addColorStop(0, sev === 'crit' ? '#f87171' : '#60a5fa');
    grad.addColorStop(1, clr);
    c.fillStyle = grad;
    drawRoundedBar(c, 0, y - barH / 2, uw, barH, 6);
    // 利用率文字
    c.fillStyle = '#fff'; c.font = 'bold 10px ' + MONO; c.textAlign = 'center';
    c.fillText((util * 100).toFixed(1) + '%', w / 2, y + 3.5); c.textAlign = 'left';
  }

  /* ============ 量化误差 ============ */
  const fmtE = v => v >= 0.01 ? v.toFixed(3) : v.toExponential(1);
  function cellColor(v, good, bad) {
    const t = clamp((v - good) / (bad - good), 0, 1);
    const r = Math.round(lerp(61, 255, t)), g = Math.round(lerp(220, 90, t)), b = Math.round(lerp(132, 106, t));
    return `rgba(${r},${g},${b},.16)`;
  }
  function renderErrorTable() {
    const table = $('c7etable'); if (!table) return;
    const rows = [...layers].sort((a, b) => {
      let av, bv;
      switch (sortKey) {
        case 'name': return sortAsc ? a.name.localeCompare(b.name) : b.name.localeCompare(a.name);
        case 'sqnr': av = a.sqnr[cur]; bv = b.sqnr[cur]; break;
        case 'cos': av = a.cos[cur]; bv = b.cos[cur]; break;
        case 'mse': av = a.mse[cur]; bv = b.mse[cur]; break;
        case 'maxerr': av = a.maxerr[cur]; bv = b.maxerr[cur]; break;
        case 'over': av = a.over[cur]; bv = b.over[cur]; break;
      }
      return sortAsc ? av - bv : bv - av;
    });
    const tag = { attn: 'attn', mlp: 'mlp', norm: 'norm', io: 'io' };
    const tb = table.querySelector('tbody');
    tb.innerHTML = rows.map(L => {
      const sq = L.sqnr[cur];
      return `<tr class="${L === selLayer ? 'sel' : ''}" data-n="${L.name}">
        <td><span class="h8-tag ${L.type}">${tag[L.type]}</span>${L.name}</td>
        <td><span class="h8-cell" style="background:${cellColor(sq, 40, 22)};color:${sevColor[sevOf(sq)]}">${sq.toFixed(1)}</span></td>
        <td>${L.cos[cur].toFixed(4)}</td>
        <td>${fmtE(L.mse[cur])}</td>
        <td>${fmtE(L.maxerr[cur])}</td>
        <td><span class="h8-cell" style="background:${cellColor(L.over[cur], 0.005, 0.05)}">${(L.over[cur] * 100).toFixed(2)}%</span></td>
      </tr>`;
    }).join('');
    tb.querySelectorAll('tr').forEach(tr => tr.onclick = () => selectLayer(layers.find(l => l.name === tr.dataset.n)));
  }

  /* ---------- 整网图溢出率叠加：把每层当前步溢出率映射到默认 L5 整网图的算子节点 id ----------
     红/绿 2 档阈值 OVER_THRESH（>1% 视为截断风险，与表格「>1% 即为嫌疑」一致）。整网图为单个代表性
     decoder layer，故同名算子（q_b_proj 等）跨 6 个块取「最差块」的溢出率（放大首害算子，命中叙事）。 */
  const OVER_THRESH = 0.01;
  let stepCb = null;
  function graphNodeIdOf(L) {
    if (L.depth === 0) return 'token_embedding';
    if (L.name === 'final_norm') return 'final_norm';
    if (L.name === 'lm_head') return 'lm_head';
    return L.name.replace(/^L\d+\./, '');   // 去掉块前缀 → 整网图算子节点 id
  }
  // 返回 { nodeId: { over, tier, name, sqnr } }，供 training-run-twin 在 #graphStage 上注入右上角溢出率徽标
  function overflowMap() {
    const m = {};
    layers.forEach(L => {
      const id = graphNodeIdOf(L), ov = L.over[cur];
      if (!m[id] || ov > m[id].over) {
        m[id] = { over: ov, tier: ov > OVER_THRESH ? 'crit' : 'ok', name: L.name, sqnr: L.sqnr[cur] };
      }
    });
    return m;
  }
  // 注册训练步/选层变化回调，整网图叠加层据此重绘徽标
  function onStep(cb) { stepCb = typeof cb === 'function' ? cb : null; }

  function renderMetricChart() {
    const es = $('c7errSel'); if (es) es.textContent = selLayer.name;
    const r = fit($('c7metric'), 150); if (!r) return; const { c, w, h } = r; c.clearRect(0, 0, w, h);
    const pad = 8; const X = i => pad + (w - pad * 2) * i / (N - 1);
    grid(c, w, h, 3);
    const slo = 14, shi = 48; const Ys = v => h - 14 - (h - 22) * (v - slo) / (shi - slo);
    c.strokeStyle = '#0ea5e9'; c.lineWidth = 1.6; c.beginPath(); selLayer.sqnr.forEach((v, i) => { i ? c.lineTo(X(i), Ys(v)) : c.moveTo(X(i), Ys(v)); }); c.stroke();
    c.strokeStyle = 'rgba(220,38,38,.4)'; c.setLineDash([3, 3]); c.beginPath(); c.moveTo(pad, Ys(30)); c.lineTo(w - pad, Ys(30)); c.stroke(); c.setLineDash([]);
    c.fillStyle = 'rgba(220,38,38,.6)'; c.font = '9px ' + MONO; c.fillText('30dB 红线', w - 56, Ys(30) - 3);
    const omx = Math.max(0.02, ...selLayer.over); const Yo = v => h - 14 - (h - 22) * v / omx;
    c.strokeStyle = '#dc2626'; c.lineWidth = 1.4; c.beginPath(); selLayer.over.forEach((v, i) => { i ? c.lineTo(X(i), Yo(v)) : c.moveTo(X(i), Yo(v)); }); c.stroke();
    c.strokeStyle = 'rgba(220,38,38,.5)'; c.setLineDash([4, 4]); c.beginPath(); c.moveTo(X(DIV), 8); c.lineTo(X(DIV), h - 14); c.stroke(); c.setLineDash([]);
    c.strokeStyle = 'rgba(15,23,42,.45)'; c.globalAlpha = .5; c.beginPath(); c.moveTo(X(cur), 8); c.lineTo(X(cur), h - 14); c.stroke(); c.globalAlpha = 1;
  }
  function renderHeat() {
    const r = fit($('c7heat'), 220); if (!r) return; const { c, w, h } = r; c.clearRect(0, 0, w, h);
    const show = [...layers].sort((a, b) => a.sqnr[cur] - b.sqnr[cur]).slice(0, 12);
    const rowH = (h - 14) / show.length, colW = (w - 70) / N;
    show.forEach((L, r2) => {
      for (let i = 0; i < N; i++) {
        const sq = L.sqnr[i]; const t = clamp((40 - sq) / (40 - 16), 0, 1);
        c.fillStyle = `rgba(${Math.round(lerp(20, 255, t))},${Math.round(lerp(30, 90, t))},${Math.round(lerp(50, 106, t))},${0.15 + 0.85 * t})`;
        c.fillRect(70 + i * colW, r2 * rowH, colW + 0.5, rowH - 1);
      }
      c.fillStyle = '#64748b'; c.font = '9px ' + MONO; c.textAlign = 'right';
      c.fillText(L.name.replace('_proj', '').slice(0, 10), 66, r2 * rowH + rowH / 2 + 3); c.textAlign = 'left';
    });
    c.strokeStyle = 'rgba(15,23,42,.45)'; c.globalAlpha = .6; c.beginPath(); c.moveTo(70 + cur * colW, 0); c.lineTo(70 + cur * colW, h - 14); c.stroke(); c.globalAlpha = 1;
    c.fillStyle = '#94a3b8'; c.font = '9px ' + MONO; c.fillText('step 0', 70, h - 3); c.fillText('10000', w - 30, h - 3);
  }

  /* ============ 误差传播 ============ */
  function renderProp() {
    const flow = $('c7propFlow'); if (!flow) return;
    const byDepth = {};
    layers.forEach(L => { (byDepth[L.depth] = byDepth[L.depth] || []).push(L); });
    const depths = Object.keys(byDepth).map(Number).sort((a, b) => a - b);
    let cumF = 0; const cum = [];
    flow.innerHTML = depths.map(d => {
      const g = byDepth[d];
      const fwd = Math.max(...g.map(L => Math.max(0, 40 - L.sqnr[cur])));
      const grd = fwd * 0.7 * (1 + d * 0.05);
      cumF += fwd * 0.6;
      cum.push(cumF);
      const fH = clamp(fwd / 24, 0, 1) * 100, gH = clamp(grd / 24, 0, 1) * 60;
      const worst = g.reduce((a, L) => L.sqnr[cur] < a.sqnr[cur] ? L : a);
      return `<div class="h8-pcol" data-n="${worst.name}" title="${worst.name}">
        <div class="h8-bars">
          <div class="h8-b" style="height:${gH}px;background:var(--h8-grad);opacity:.7"></div>
          <div class="h8-b" style="height:${fH}px;background:var(--h8-signal2)"></div>
        </div>
        <div class="h8-lbl">${d === 0 ? 'emb' : d === depths.length - 1 ? 'head' : 'L' + (d - 1)}</div>
      </div>`;
    }).join('');
    flow.querySelectorAll('.h8-pcol').forEach(p => p.onclick = () => selectLayer(layers.find(l => l.name === p.dataset.n)));
    const r = fit($('c7propLine'), 120); if (r) {
      const { c, w, h } = r; c.clearRect(0, 0, w, h);
      const mx = Math.max(...cum, 1); const X = i => 10 + (w - 20) * i / (cum.length - 1), Y = v => h - 14 - (h - 22) * v / mx;
      grid(c, w, h, 3);
      const g2 = c.createLinearGradient(0, 0, 0, h); g2.addColorStop(0, 'rgba(220,38,38,.3)'); g2.addColorStop(1, 'rgba(220,38,38,0)');
      c.beginPath(); c.moveTo(X(0), Y(0)); cum.forEach((v, i) => c.lineTo(X(i), Y(v))); c.lineTo(X(cum.length - 1), Y(0)); c.closePath(); c.fillStyle = g2; c.fill();
      c.strokeStyle = '#dc2626'; c.lineWidth = 2; c.beginPath(); cum.forEach((v, i) => { i ? c.lineTo(X(i), Y(v)) : c.moveTo(X(i), Y(v)); }); c.stroke();
      cum.forEach((v, i) => { c.fillStyle = '#dc2626'; c.beginPath(); c.arc(X(i), Y(v), 2.5, 0, 7); c.fill(); });
      c.fillStyle = '#94a3b8'; c.font = '9px ' + MONO; c.fillText('累积误差', 12, 12);
    }
    const amp = (cum[cum.length - 1] / (cum[1] || 1)).toFixed(1);
    const pr = $('c7propReads');
    if (pr) pr.innerHTML = `
      <div class="h8-read"><div class="l">输入端单层误差</div><div class="v">${(cum[1] || 0).toFixed(2)}</div></div>
      <div class="h8-read"><div class="l">输出端累积误差</div><div class="v" style="color:var(--h8-crit)">${cum[cum.length - 1].toFixed(2)}</div></div>
      <div class="h8-read"><div class="l">累积放大倍数</div><div class="v" style="color:var(--h8-crit)">×${amp}</div></div>
      <div class="h8-read"><div class="l">主要放大深度</div><div class="v">L3–L5</div></div>`;
    const firstBad = [...layers].filter(L => L.sqnr[cur] < 30).sort((a, b) => a.depth - b.depth)[0];
    const ps = $('c7propSource');
    if (ps) ps.innerHTML = firstBad ? `
      <div>▸ 最浅层越红线算子:<br>&nbsp;&nbsp;<b style="color:var(--h8-crit)">${firstBad.name}</b> @ SQNR ${firstBad.sqnr[cur].toFixed(1)}dB</div>
      <div style="margin-top:8px">▸ 误差在其下游 <b style="color:var(--h8-warn)">${layers.length - layers.indexOf(firstBad)}</b> 个算子中逐级累积</div>
      <div style="margin-top:8px;color:var(--h8-dim)">该算子权重分布出现重尾,超出 HiF8 ±448 表示范围,<br>触发截断 → 前向激活失真 → 反向梯度污染 → loss 偏移。</div>`
      : '<div style="color:var(--h8-ok)">当前步无算子越过 30dB 红线。</div>';
  }

  /* ============ 根因分析 ============ */
  function renderSOP() {
    const host = $('c7sop'); if (!host) return;
    const steps = [
      { n: '① 阈值熔断', d: '|Δloss|>0.05 触发', done: cur >= DIV },
      { n: '② 双端同权重前反向', d: 'BF16/候选同载入', done: cur >= DIV },
      { n: '③ Tensor Hook 逐层拦截', d: '不破坏计算图', done: cur >= DIV },
      { n: '④ 余弦<0.999 锁层', d: '逐层相似度比对', done: cur >= DIV },
      { n: '⑤ 算子级下钻', d: '定位注入源+处置', done: cur >= DIV },
    ];
    host.innerHTML = steps.map((s, i) => {
      const col = s.done ? 'var(--h8-signal)' : 'var(--h8-dim)';
      const bg = s.done ? 'color-mix(in srgb,#16a34a 12%,transparent)' : 'var(--h8-inset)';
      return `<div style="flex:1;background:${bg};border:1px solid var(--h8-stroke);
          ${i ? 'border-left:none;' : ''}${i === 0 ? 'border-radius:8px 0 0 8px;' : ''}${i === steps.length - 1 ? 'border-radius:0 8px 8px 0;' : ''}
          padding:10px 12px;position:relative">
        <div style="font-family:${MONO};font-size:11.5px;color:${col}">${s.n}</div>
        <div style="font-size:10.5px;color:var(--h8-dim);margin-top:4px">${s.d}</div>
        <div style="position:absolute;right:8px;top:9px;color:${col};font-size:11px">${s.done ? '●' : '○'}</div>
      </div>`;
    }).join('');
  }
  function renderAttr() {
    renderSOP();
    const r = fit($('c7attr'), 240); if (r) {
      const { c, w, h } = r; c.clearRect(0, 0, w, h);
      const top = [...layers].sort((a, b) => b.sensPct - a.sensPct).slice(0, 9);
      const mx = Math.max(...top.map(L => L.sensPct));
      const bh = (h - 20) / top.length;
      top.forEach((L, i) => {
        const y = 10 + i * bh; const bw = (w - 160) * L.sensPct / mx;
        const barH = bh - 8, barR = Math.min(4, barH / 2);
        c.fillStyle = CULPRIT[L.name] ? 'rgba(220,38,38,.18)' : 'rgba(148,163,184,.14)';
        drawRoundedBar(c, 150, y + 3, (w - 160), barH, barR);
        c.fillStyle = CULPRIT[L.name] ? '#dc2626' : '#3b6fe0';
        c.globalAlpha = CULPRIT[L.name] ? 1 : .6;
        drawRoundedBar(c, 150, y + 3, bw, barH, barR);
        c.globalAlpha = 1;
        c.fillStyle = '#64748b'; c.font = '10px ' + MONO; c.textAlign = 'right'; c.fillText(L.name.replace('_proj', ''), 144, y + bh / 2 + 3); c.textAlign = 'left';
        c.fillStyle = '#1e293b'; c.fillText((L.sensPct * 100).toFixed(1) + '%', 150 + bw + 6, y + bh / 2 + 3);
      });
    }
    const r2 = fit($('c7corr'), 240); if (r2) {
      const { c: cc, w: cw, h: ch } = r2; cc.clearRect(0, 0, cw, ch);
      grid(cc, cw, ch, 4);
      const cul = selLayer && CULPRIT[selLayer.name] ? selLayer : layers.find(l => l.name === 'L4.q_b_proj');
      const pad = 26;
      const xs = cul.sqnr.map(v => 40 - v), ys = dloss;
      const xmx = Math.max(...xs), ymx = Math.max(...ys);
      const X = v => pad + (cw - pad - 8) * v / xmx, Y = v => ch - 18 - (ch - 24) * v / ymx;
      for (let i = 0; i < N; i++) {
        cc.fillStyle = `rgba(${Math.round(lerp(91, 255, i >= DIV ? 1 : .2))},${i >= DIV ? 90 : 157},${i >= DIV ? 106 : 255},.6)`;
        cc.beginPath(); cc.arc(X(xs[i]), Y(ys[i]), 2.3, 0, 7); cc.fill();
      }
      cc.fillStyle = '#94a3b8'; cc.font = '9px ' + MONO; cc.fillText('SQNR塌陷→', pad, ch - 4); cc.save(); cc.translate(10, ch / 2); cc.rotate(-Math.PI / 2); cc.fillText('Δloss→', 0, 0); cc.restore();
      const mX = xs.reduce((a, b) => a + b) / N, mY = ys.reduce((a, b) => a + b) / N;
      let sxy = 0, sx = 0, sy = 0; for (let i = 0; i < N; i++) { sxy += (xs[i] - mX) * (ys[i] - mY); sx += (xs[i] - mX) ** 2; sy += (ys[i] - mY) ** 2; }
      const rr = sxy / Math.sqrt(sx * sy);
      const el = $('c7corrR'); if (el) el.textContent = rr.toFixed(3) + ' (' + cul.name + ')';
    }
    const susp = [...layers].sort((a, b) => b.sensPct - a.sensPct).slice(0, 4);
    const advice = {
      'L4.q_b_proj': '该层保留 BF16 / 提高 per-channel scaling',
      'L3.o_proj': '启用动态 scaling · 裁剪重尾离群值',
      'L5.router_gate': 'per-tensor→per-channel 量化',
    };
    const host = $('c7suspects');
    if (host) host.innerHTML = susp.map((L, i) => `
      <div class="h8-suspect">
        <div class="rank">#${i + 1}</div>
        <div class="s-main">
          <div class="s-name">${L.name}<span class="h8-tag ${L.type}">${L.type}</span><span class="s-contrib">loss 贡献 ${(L.sensPct * 100).toFixed(1)}%</span></div>
          <div class="s-bar"><i style="width:${L.sensPct / susp[0].sensPct * 100}%"></i></div>
          <div class="s-meta">SQNR ${L.sqnr[cur].toFixed(1)}dB · 溢出 ${(L.over[cur] * 100).toFixed(2)}%</div>
          <div class="s-act">${advice[L.name] || '检查数值范围'}</div>
        </div>
      </div>`).join('');
  }

  /* ---------- 层选中联动（张量分布 + 量化误差） ---------- */
  function selectLayer(L) {
    if (!L) return;
    selLayer = L;
    renderDist();
    renderErrorTable();
    renderMetricChart();
    renderHeat();
    renderAttr();
  }

  /* ---------- 训练步回放（概览节 scrubber，驱动全部五节图表） ---------- */
  let playing = false, playTimer = null;

  function updateScrubUI() {
    const pct = cur / (N - 1) * 100;
    const tf = $('c7trackFill'); if (tf) tf.style.width = pct + '%';
    const tm = $('c7trackMarker'); if (tm) tm.style.left = (DIV / (N - 1) * 100) + '%';
    const range = $('c7stepRange'); if (range && +range.value !== cur) range.value = cur;
    const rdStep = $('c7rdStep'); if (rdStep) rdStep.textContent = stepOf(cur);
    const rdLoss = $('c7rdLoss'); if (rdLoss) { rdLoss.textContent = dloss[cur].toFixed(3); rdLoss.style.color = dloss[cur] > 0.05 ? 'var(--h8-crit)' : 'var(--h8-ink)'; }
    const rdSqnr = $('c7rdSqnr'); if (rdSqnr) { rdSqnr.textContent = meanSqnr[cur].toFixed(1) + 'dB'; rdSqnr.style.color = sevColor[meanSqnr[cur] > 32 ? 'ok' : meanSqnr[cur] > 28 ? 'warn' : 'crit']; }
    const dl = $('c7divLabel'); if (dl) dl.textContent = 'step ' + stepOf(DIV);
  }

  // 拖动/播放改变 cur 后，五节所有随步演化的图表都要重绘（工作台里 scrubber 是全局的）
  function redraw() {
    renderKpis(); renderFmtStatus(); renderLoss(); renderDelta(); renderLogit(); renderTimeline();
    renderDist();
    renderErrorTable(); renderMetricChart(); renderHeat();
    renderProp();
    renderAttr();
    updateScrubUI();
    if (stepCb) stepCb();   // 通知整网图叠加层刷新溢出率徽标
  }

  function setPlayIcon() {
    const icon = $('c7playIcon');
    if (icon) icon.innerHTML = playing
      ? '<rect x="3" y="2" width="3" height="10"/><rect x="8" y="2" width="3" height="10"/>'
      : '<path d="M3 2 L12 7 L3 12 Z"/>';
  }
  function togglePlay() {
    playing = !playing;
    setPlayIcon();
    if (playing) {
      if (cur >= N - 1) cur = 0;
      playTimer = setInterval(() => {
        cur++;
        if (cur >= N - 1) { cur = N - 1; playing = false; setPlayIcon(); clearInterval(playTimer); playTimer = null; }
        redraw();
      }, 90);
    } else if (playTimer) { clearInterval(playTimer); playTimer = null; }
  }
  // 关闭定位链面板时停掉回放，避免遗留 interval 空转
  function stop() {
    if (playTimer) { clearInterval(playTimer); playTimer = null; }
    playing = false;
  }

  /* ---------- 交互绑定 ---------- */
  // 图表悬浮 tooltip：复用页面已有的 diagnosis-bubble，鼠标在关键图表上滑动时显示该步指标
  let _chartTip = null;
  function chartTipEl() {
    if (!_chartTip) {
      _chartTip = document.createElement('div');
      _chartTip.className = 'diagnosis-bubble';
      _chartTip.style.cssText = 'position:fixed;padding:5px 9px;font-family:var(--h8-mono);font-size:10.5px;line-height:1.45;pointer-events:none;z-index:200;white-space:nowrap;display:none;border-radius:5px;';
      document.body.appendChild(_chartTip);
    }
    return _chartTip;
  }
  function showChartTip(e, html) {
    const tip = chartTipEl();
    tip.innerHTML = html;
    tip.style.display = 'block';
    tip.style.left = Math.min(e.clientX + 14, window.innerWidth - 200) + 'px';
    tip.style.top = (e.clientY - 36) + 'px';
  }
  function hideChartTip() { const tip = chartTipEl(); tip.style.display = 'none'; }

  // 为折线图画布绑定悬浮：传入 canvas id、数据数组、格式化函数。
  // 图表使用标准 X 映射 X(i) = pad + (canvasW - pad - padR) * i/(N-1)，鼠标就近匹配最近点。
  const _chartHoverBound = {};
  function bindLineChartHover(canvasId, dataArr, fmtFn) {
    const cv = document.getElementById(canvasId);
    if (!cv || _chartHoverBound[canvasId]) return;
    _chartHoverBound[canvasId] = true;
    const onMove = function(e) {
      const rect = cv.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const w = rect.width;
      // 标准图表参数（与 renderLoss/renderDelta 等一致）
      const pad = 30, padR = 8;
      const X = i => pad + (w - pad - padR) * i / (N - 1);
      let bestI = 0, bestDist = 1e9;
      for (let i = 0; i < N; i++) {
        const d = Math.abs(X(i) - mx);
        if (d < bestDist) { bestDist = d; bestI = i; }
      }
      if (bestDist > 28) { hideChartTip(); return; }
      // 动态获取最新数据（selLayer 可能已切换）
      const arr = typeof dataArr === 'function' ? dataArr() : dataArr;
      const val = arr[bestI];
      const sev = bestI >= DIV
        ? (bestI >= DIV + 40 ? 'crit' : 'warn')
        : 'ok';
      const clr = sev === 'crit' ? '#dc2626' : sev === 'warn' ? '#ea580c' : '#16a34a';
      const dot = sev === 'crit' ? '●' : sev === 'warn' ? '●' : '●';
      showChartTip(e, `<span style="color:${clr}">${dot}</span> step&nbsp;${stepOf(bestI)}<br><b>${fmtFn ? fmtFn(val) : (typeof val === 'number' ? val.toFixed(4) : val)}</b>`);
    };
    cv.addEventListener('mousemove', onMove);
    cv.addEventListener('mouseleave', hideChartTip);
    // 保存引用以便后续更新数据源
    cv._hoverData = { getData: Array.isArray(dataArr) ? () => dataArr : dataArr, fmtFn };
  }
  function bind() {
    const range = $('c7stepRange');
    if (range) range.oninput = e => { cur = +e.target.value; redraw(); };
    const play = $('c7play');
    if (play) play.onclick = togglePlay;
    document.querySelectorAll('#c7seg button').forEach(b => b.onclick = () => {
      document.querySelectorAll('#c7seg button').forEach(x => x.classList.remove('on'));
      b.classList.add('on'); tensorType = b.dataset.t; renderDist();
    });
    const table = $('c7etable');
    if (table) table.querySelectorAll('th').forEach(th => th.onclick = () => {
      const k = th.dataset.s;
      if (k === sortKey) sortAsc = !sortAsc; else { sortKey = k; sortAsc = (k === 'sqnr' || k === 'cos'); }
      table.querySelectorAll('th').forEach(x => x.classList.toggle('sorted', x.dataset.s === sortKey));
      renderErrorTable();
    });
    // 图表悬浮展示指标（对标训练监控「精度」面板的 cursor tooltip）
    bindLineChartHover('c7loss', lossHif, v => 'HiF8 loss ' + v.toFixed(4));
    bindLineChartHover('c7delta', dloss, v => 'Δloss ' + (v > 0 ? '+' : '') + v.toFixed(4));
    bindLineChartHover('c7logit', logitUnif, v => '打散度 ' + v.toFixed(3));
    bindLineChartHover('c7metric', () => selLayer.sqnr, v => 'SQNR ' + v.toFixed(1) + ' dB');
    // h8-help 问号气泡：改用页面共享的 #diagnosisTooltip 替代原生 title（后者在 overflow:hidden 祖先中可能不显示）
    bindHelpBubbles();
  }

  const HELP_TEXTS = {
    'error-table': '<b>溢出率</b>是最关键指标：FP8 E4M3 硬截断（&gt;±448→±448）不可逆，SQNR/余弦/MSE 均为后果。先看溢出率找首害算子（&gt;1% 即为嫌疑），再看 SQNR 确认量级。<br><br><b>整网图节点覆盖说明</b>：HiF8 精度诊断仅模拟<b>投影算子、归一化层、MLP 门/专家库</b>的 FP8 溢出行为。通信算子（all-to-all/all-gather/reduce-scatter）、因果卷积、残差连接、注意力核心等节点本身不参与 FP8 量化，故整网图上这些节点暂无溢出率标注。',
  };

  function bindHelpBubbles() {
    const bubble = document.getElementById('diagnosisTooltip');
    if (!bubble) return;
    document.querySelectorAll('.h8-help[data-help-key]').forEach(function (el) {
      if (el.dataset.helpBound) return;
      el.dataset.helpBound = '1';
      var html = HELP_TEXTS[el.dataset.helpKey] || '';
      if (!html) return;
      el.addEventListener('mouseenter', function (e) {
        bubble.innerHTML = html;
        bubble.hidden = false;
        bubble.style.maxWidth = '320px';
        bubble.style.left = Math.min(e.clientX + 14, window.innerWidth - 340) + 'px';
        bubble.style.top = (e.clientY + 10) + 'px';
      });
      el.addEventListener('mousemove', function (e) {
        bubble.style.left = Math.min(e.clientX + 14, window.innerWidth - 340) + 'px';
        bubble.style.top = (e.clientY + 10) + 'px';
      });
      el.addEventListener('mouseleave', function () {
        bubble.hidden = true;
        bubble.style.maxWidth = '';
      });
    });
  }

  /* ---------- 各节 HTML ---------- */
  function sectionOverview() {
    return `<div class="hif8c7">
      <div class="h8-scrub">
        <button class="h8-play" id="c7play" type="button" title="回放误差累积过程">
          <svg id="c7playIcon" width="14" height="14" viewBox="0 0 14 14" fill="currentColor"><path d="M3 2 L12 7 L3 12 Z"/></svg>
        </button>
        <div class="h8-scrub-wrap">
          <div class="h8-scrub-top">
            <span>训练步回放 · 观察量化误差的传播与累积（驱动下方全部图表）</span>
            <span>发散点 <b id="c7divLabel">step 3150</b></span>
          </div>
          <div class="h8-track">
            <div class="h8-track-bg"><div class="h8-track-fill" id="c7trackFill"></div></div>
            <div class="h8-track-marker" id="c7trackMarker"></div>
            <input type="range" id="c7stepRange" min="0" max="${N - 1}" value="${N - 1}">
          </div>
        </div>
        <div class="h8-scrub-read">
          <div><span>STEP</span><b id="c7rdStep">10000</b></div>
          <div><span>ΔLOSS</span><b id="c7rdLoss">—</b></div>
          <div><span>均值 SQNR</span><b id="c7rdSqnr">—</b></div>
        </div>
      </div>
      <div class="h8-kpis" id="c7kpis"></div>
      <div class="h8-card" style="margin-top:14px;padding:12px 14px">
        <h3 style="margin-bottom:10px">候选低精度格式 · 收敛状态 <span class="hint">同权重/同数据下多格式对照 · 依据洞察点 9</span></h3>
        <div id="c7fmt" style="display:flex;gap:10px;flex-wrap:wrap"></div>
      </div>
      <div class="h8-grid" style="grid-template-columns:1.55fr 1fr;margin-top:14px">
        <div class="h8-card">
          <h3>训练损失 · 多格式对照 <span class="hint">发散自动检测 · 阈值 |Δloss| &gt; 0.05</span></h3>
          <canvas id="c7loss"></canvas>
          <div class="h8-legend" style="margin-top:10px">
            <span><i style="background:var(--h8-signal2)"></i>候选 FP8/HiF8</span>
            <span><i style="background:var(--h8-signal)"></i>BF16 基线</span>
            <span><i style="background:var(--h8-ok)"></i>MXFP8(正常)</span>
            <span><i style="background:var(--h8-warn)"></i>loss spike</span>
            <span><i style="background:var(--h8-crit)"></i>发散区间</span>
          </div>
        </div>
        <div class="h8-card">
          <h3>损失偏差 Δloss = 候选 − BF16 <span class="hint">累积漂移</span></h3>
          <canvas id="c7delta"></canvas>
          <div class="h8-readgrid" style="grid-template-columns:1fr 1fr">
            <div class="h8-read"><div class="l">当前 Δloss</div><div class="v" id="c7ovDelta">—</div></div>
            <div class="h8-read"><div class="l">首次超阈步</div><div class="v" style="color:var(--h8-crit)">step 3150</div></div>
          </div>
        </div>
      </div>
      <div class="h8-grid" style="grid-template-columns:1fr 1.55fr;margin-top:14px">
        <div class="h8-card">
          <h3>batch logit 分布 · 打散度 <span class="hint">openPangu 2.0 Flash · 防 spike</span></h3>
          <canvas id="c7logit"></canvas>
          <div class="h8-readgrid" style="grid-template-columns:1fr 1fr">
            <div class="h8-read"><div class="l">当前打散度</div><div class="v" id="c7ovUnif">—</div></div>
            <div class="h8-read"><div class="l">早期 spike 次数</div><div class="v" style="color:var(--h8-warn)">3</div></div>
          </div>
        </div>
        <div class="h8-card">
          <h3>诊断事件时间线 <span class="hint">数值守卫触发记录</span></h3>
          <canvas id="c7timeline"></canvas>
        </div>
      </div>
    </div>`;
  }
  function sectionDist() {
    return `<div class="hif8c7">
      <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;flex-wrap:wrap">
        <div class="h8-seg" id="c7seg">
          <button class="on" data-t="weight">权重</button>
          <button data-t="act">激活</button>
          <button data-t="grad">梯度</button>
        </div>
        <div class="mono" style="color:var(--h8-mute);font-size:12px">当前张量 · <b id="c7distLayer" style="color:var(--h8-ink)">—</b></div>
        <div style="flex:1"></div>
        <div class="mono" style="font-size:11px;color:var(--h8-dim)">HiF8 (1-3-3-1 锥形) · scale=<span id="c7distScale">—</span></div>
      </div>
      <div class="h8-grid" style="grid-template-columns:1.6fr 1fr">
        <div class="h8-card">
          <h3>数值分布 · 原始 vs HiF8 量化后 <span class="hint">灰色刻度=可表示栅格(近零更密) · 红区=截断/溢出</span></h3>
          <canvas id="c7hist"></canvas>
          <div class="h8-legend" style="margin-top:10px">
            <span><i style="background:var(--h8-signal)"></i>原始 FP32 密度</span>
            <span><i style="background:var(--h8-signal2)"></i>HiF8 量化后</span>
            <span><i style="background:#cbd5e1"></i>可表示栅格</span>
            <span><i style="background:#fecaca"></i>溢出区(clip)</span>
          </div>
          <div class="h8-taper-key">
            <div style="background:#93c5fd">次正规 · 密</div>
            <div style="background:#0ea5e9">高精度带 M3</div>
            <div style="background:#3b6fe0">中段 M2</div>
            <div style="background:#ea580c">大值 M1 · 疏</div>
            <div style="background:#dc2626">溢出 &gt; ±448</div>
          </div>
        </div>
        <div class="h8-grid" style="align-content:start">
          <div class="h8-card">
            <h3>动态范围利用率 <span class="hint">HiF8 可表示区间占用</span></h3>
            <canvas id="c7range"></canvas>
            <div class="h8-readgrid">
              <div class="h8-read"><div class="l">min / max</div><div class="v" id="c7dMinMax">—</div></div>
              <div class="h8-read"><div class="l">mean / std</div><div class="v" id="c7dMeanStd">—</div></div>
              <div class="h8-read"><div class="l">溢出率 (overflow)</div><div class="v" id="c7dOver">—</div></div>
              <div class="h8-read"><div class="l">下溢率 (underflow→0)</div><div class="v" id="c7dUnder">—</div></div>
              <div class="h8-read"><div class="l">有效比特 (ENOB)</div><div class="v" id="c7dEnob">—</div></div>
              <div class="h8-read"><div class="l">量化 SNR</div><div class="v" id="c7dSqnr">—</div></div>
            </div>
          </div>
          <div class="h8-card">
            <h3>分布形态 · 量化适配度 <span class="hint">峰度/偏度判断是否适合量化 · 依据洞察点 9</span></h3>
            <div style="display:flex;align-items:center;gap:14px;margin-bottom:12px">
              <div style="flex:0 0 84px;text-align:center">
                <div class="mono" id="c7qScore" style="font-size:30px;line-height:1">—</div>
                <div style="font-size:10px;color:var(--h8-dim);margin-top:2px">量化适配度</div>
              </div>
              <div style="flex:1">
                <div style="height:8px;background:var(--surface-2);border-radius:5px;overflow:hidden">
                  <i id="c7qScoreBar" style="display:block;height:100%;width:0"></i>
                </div>
                <div id="c7fitVerdict" style="font-size:11px;color:var(--h8-mute);margin-top:9px;line-height:1.5"></div>
              </div>
            </div>
            <div class="h8-readgrid" style="grid-template-columns:1fr 1fr">
              <div class="h8-read"><div class="l">最佳拟合分布</div><div class="v" id="c7dFit" style="font-size:14px">—</div></div>
              <div class="h8-read"><div class="l">峰度 kurtosis</div><div class="v" id="c7dKurt">—</div></div>
              <div class="h8-read"><div class="l">偏度 skewness</div><div class="v" id="c7dSkew">—</div></div>
              <div class="h8-read"><div class="l">离群(尾部)占比</div><div class="v" id="c7dTail">—</div></div>
            </div>
          </div>
        </div>
      </div>
    </div>`;
  }
  function sectionError() {
    return `<div class="hif8c7">
      <div class="h8-grid" style="grid-template-columns:1.3fr 1fr">
        <div class="h8-card">
          <h3>层 / 算子级量化误差指标 <span class="hint">点击表头排序 · 点击行钻取</span></h3>
          <div class="h8-table-scroll">
            <table class="h8-etable" id="c7etable">
              <thead><tr>
                <th data-s="name">层 / 算子</th>
                <th data-s="sqnr" class="sorted">SQNR↑<br><span style="font-weight:400">dB</span></th>
                <th data-s="cos">余弦相似度</th>
                <th data-s="mse">MSE</th>
                <th data-s="maxerr">最大误差</th>
                <th data-s="over">溢出率</th>
              </tr></thead>
              <tbody></tbody>
            </table>
          </div>
        </div>
        <div class="h8-grid" style="align-content:start">
          <div class="h8-card">
            <h3>选中层指标随训练步演化 <span class="hint" id="c7errSel">—</span></h3>
            <canvas id="c7metric"></canvas>
            <div class="h8-legend" style="margin-top:8px">
              <span><i style="background:var(--h8-signal)"></i>SQNR (dB)</span>
              <span><i style="background:var(--h8-crit)"></i>溢出率</span>
            </div>
          </div>
          <div class="h8-card">
            <h3>层 × 步 误差热力图 <span class="hint">SQNR 塌陷 = 亮</span></h3>
            <canvas id="c7heat"></canvas>
          </div>
        </div>
      </div>
    </div>`;
  }
  function sectionProp() {
    return `<div class="hif8c7">
      <div class="h8-card">
        <h3>误差沿网络深度的传播与累积 <span class="hint">前向左→右 · 柱高=该处单层量化误差 · 折线=累积误差</span></h3>
        <div class="h8-prop-flow" id="c7propFlow"></div>
        <canvas id="c7propLine" style="margin-top:6px"></canvas>
        <div class="h8-legend" style="margin-top:10px">
          <span><i style="background:var(--h8-signal2)"></i>前向激活误差</span>
          <span><i style="background:var(--h8-grad)"></i>反向梯度误差</span>
          <span><i style="background:var(--h8-crit)"></i>累积误差 (随深度)</span>
        </div>
      </div>
      <div class="h8-grid" style="grid-template-columns:1fr 1fr;margin-top:14px">
        <div class="h8-card">
          <h3>累积放大分析 <span class="hint">相对输入层的误差放大倍数</span></h3>
          <div class="h8-readgrid" style="grid-template-columns:1fr 1fr" id="c7propReads"></div>
        </div>
        <div class="h8-card">
          <h3>误差注入源定位 <span class="hint">首个越过 SQNR 30dB 红线的算子</span></h3>
          <div id="c7propSource" style="font-family:${MONO};font-size:12px;line-height:1.9;color:var(--h8-mute)"></div>
        </div>
      </div>
    </div>`;
  }
  function sectionAttr() {
    return `<div class="hif8c7">
      <div class="h8-card">
        <h3>精度分叉定位流程 <span class="hint">标杆(golden)比对 SOP · 对标月之暗面自动化五步法</span></h3>
        <div id="c7sop" style="display:flex;align-items:stretch;gap:0;flex-wrap:wrap"></div>
      </div>
      <div class="h8-grid" style="grid-template-columns:1fr 1fr;margin-top:14px">
        <div class="h8-card">
          <h3>各层对全局 Δloss 的敏感度 <span class="hint">扰动归因 · Σ = 100%</span></h3>
          <canvas id="c7attr"></canvas>
        </div>
        <div class="h8-card">
          <h3>层 SQNR 塌陷 与 Δloss 尖峰 的相关性 <span class="hint">时序对齐验证</span></h3>
          <canvas id="c7corr"></canvas>
          <div class="h8-read" style="margin-top:10px"><div class="l">Pearson 相关系数 r</div><div class="v" style="color:var(--h8-crit)" id="c7corrR">—</div></div>
        </div>
      </div>
      <div class="h8-card" style="margin-top:14px">
        <h3>可疑算子清单 · 处置建议 <span class="hint">按 loss 贡献排序</span></h3>
        <div id="c7suspects"></div>
      </div>
    </div>`;
  }

  /* ---------- 对外：定位链结构 ---------- */
  function chain() {
    return {
      title: "定位链 · HiF8 低精度训练精度诊断工作台（openPangu 2.0 Flash）",
      meta: "路径:概览 → 张量分布 → 量化误差 → 误差传播 → 根因分析（算子对标 openPangu 2.0 Flash 整网图）",
      steps: [
        { label: "概览", short: "健康分 / 多格式对照", sub: "WHEN · HiF8 候选格式下 openPangu 2.0 Flash 训练至 step 3150 首次发散，BF16 / MXFP8 正常收敛", content: sectionOverview() },
        { label: "分叉判定", sub: "候选格式发散 · 切入数值精度分支", branch: true },
        { label: "张量分布", short: "直方图 / 量化适配度", sub: "WHICH · 权重/激活/梯度分布 vs HiF8 E4M3 ±448 可表示栅格", content: sectionDist() },
        { label: "量化误差", short: "SQNR / 溢出 / 热力图", sub: "WHAT · openPangu 2.0 Flash 层×算子级 SQNR·余弦·MSE·溢出率随步演化", content: sectionError() },
        { label: "误差传播", short: "累积放大 ×N", sub: "WHERE · 误差沿网络深度累积放大，定位首个越 30dB 红线算子", content: sectionProp() },
        { label: "根因分析", short: "敏感度 / 相关性 / 处置", sub: "FIX · 扰动归因 + SQNR-Δloss 相关性锁定嫌疑算子并给处置建议", content: sectionAttr() },
      ],
    };
  }

  function renderAll() {
    stop();               // 重新进入面板时清掉上次可能残留的回放
    cur = N - 1;          // 每次打开定位链从末步（完全发散）快照开始
    playing = false;
    requestAnimationFrame(() => requestAnimationFrame(() => {
      redraw();
      setPlayIcon();
      bind();
    }));
  }

  return { chain, renderAll, stop, overflowMap, onStep, bindHelpBubbles, OVER_THRESH };
})();
