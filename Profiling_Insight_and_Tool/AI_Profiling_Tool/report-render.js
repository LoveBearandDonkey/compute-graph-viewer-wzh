/* ============================================================
   report-render.js —— 报告形态渲染器
   ------------------------------------------------------------
   报告形态是 app.js 里那个 `r` 对象的第二个 renderer：
   同源数据、不同读者。工具形态给"正在排查的人"，报告形态给"没跑过这次
   profiling 的人"——所以这里没有筛选、排序、hover、页签，一切作者先决。

   设计约束（见 REPORT-FORM-PLAN.md）：
   1. 弱交互：验收标准是"打印出来不丢任何信息"。工具态藏在 tooltip 里的阈值、
      口径、芯片型号假设，在这里必须印在版面上或进附录。
   2. 竖版 A4：显式分页容器，不赌 CSS 自动流式分页。
   3. 图多于字：每页至少一个图形元素；长文本裁剪后全文进附录 C。
   4. 总分总：执行摘要（结论是什么）→ 分章 → 行动计划（接下来做什么）。

   图表一律自绘 SVG / CSS，不引 echarts：canvas 打印会糊，CDN 依赖在离线
   归档场景不可靠。
   ============================================================ */
(function () {
  'use strict';

  /* ── 语义色板 ───────────────────────────────────────────────
     源自 swimlane-skill-pack/js/swimlane.js 的 SEMANTIC_COLORS（该常量是
     模块内 const，未挂 window）。报告只用到 Timeline/PP 这一组，在此镜像；
     swimlane.js 若调色，此处需同步。 */
  const SEM_COLOR = {
    'Fwd-Compute': '#2f6fce', 'Bwd-Compute': '#8b3fb5', 'PP-Bubble': '#cf4d86',
    'P2P-Send': '#c15a1c', 'P2P-Recv': '#c15a1c', 'DP-Collective': '#0f9b8e',
    'Optimizer': '#5d6b99', 'Free': '#566173',
    'Computing': '#1a8f52', 'Communication': '#c15a1c', 'Comm-Transmit': '#2f6fce',
    'Comm-Wait': '#d23b3b', 'Comm-Idle': '#6b7280',
  };
  const SEM_ZH = {
    'Fwd-Compute': '前向计算', 'Bwd-Compute': '反向计算', 'PP-Bubble': '流水气泡',
    'P2P-Send': 'P2P 发送', 'P2P-Recv': 'P2P 接收', 'DP-Collective': 'DP 集合通信',
    'Optimizer': '优化器', 'Free': '空闲', 'Computing': '计算',
    'Communication': '通信', 'Comm-Transmit': '通信·传输',
    'Comm-Wait': '通信·等待', 'Comm-Idle': '通信·空转',
  };

  /* ── 指标定义表 ────────────────────────────────────────────
     name/unit/dir/warn 与 index_v3.html 指标卡的 data-* 一致；desc 是工具态
     tooltip 的压缩版。报告态把它们印成"参考区间"和附录 A —— 读者 hover 不到。 */
  const METRIC_DEF = {
    step_cv:               { name: 'step 抖动 (CV)',      unit: '%', dir: 'low',  warn: 10, desc: 'step 耗时离散度（std/mean）。> 10% 说明训练不稳定，应先取典型 step 再分析。' },
    rank_spread:           { name: 'Rank 间 step 离散度', unit: '%', dir: 'low',  warn: 5,  desc: '(最慢 rank − 最快 rank) ÷ 中位数。集合通信是同步点，慢卡会把所有卡拖到同一时长。> 5% 通常意味着慢卡或负载不均。' },
    critical_path_ratio:   { name: '关键路径占比',        unit: '%', dir: 'high', warn: 80, desc: '关键路径（最长链路）÷ step 总耗时。< 80% 说明存在可被掩盖的空泡，应优先做重叠/下发优化，而非压单算子。' },
    op_utilization:        { name: '算子利用率',          unit: '%', dir: 'high', warn: 60, desc: 'AI Core（Cube + Vector）忙时占 step 时长的比例（time-based）。< 60% 通常是调度空泡或访存瓶颈把算力饿死。' },
    overlap_ratio:         { name: '计算-通信重叠率',     unit: '%', dir: 'high', warn: 70, desc: '通信被计算掩盖的比例。< 70% 表示掩盖不足，暴露通信进入关键路径。' },
    pp_bubble_ratio:       { name: 'PP 流水线 bubble 率', unit: '%', dir: 'low',  warn: null, desc: '流水线并行 bubble 占 step 比例，需对比理论值 (p−1)/(p−1+m)。实测远大于理论 → micro-batch 少 / stage 不均。' },
    host_launch_gap_ratio: { name: 'Host 下发间隙占比',   unit: '%', dir: 'low',  warn: 5,  desc: 'device 周期性小间隙中对齐到 host launch/同步 API 的占比。> 5% 判为 Host-bound。' },
    max_lane_idle:         { name: '最空闲泳道空挡',      unit: '%', dir: 'low',  warn: 10, desc: '最空闲泳道的空挡比例。device 计算泳道空挡 > 10% 通常指向 host 下发跟不上。' },
    comm_jitter:           { name: '通信抖动 (CV)',       unit: '%', dir: 'low',  warn: 15, desc: '逐 step 通信耗时的变异系数。> 15% 表示时延不稳定，可能有慢链路、拥塞或 HCCL 重传。' },
    /* mfu / mem_util 的分母是手选的芯片型号假设，不是实测值。工具形态刻意不给这两张卡
       做 ok/warn 着色以免误导为"健康/越界"，报告形态沿用该判断：neutral 表示不判定，
       只印数值与折算依据。 */
    mfu:                   { name: 'MFU（端到端）',       unit: '%', dir: 'high', warn: null, neutral: true, desc: '模型浮点算力利用率。分母为手选芯片峰值算力，本报告按 910B1 BF16 378.88 TF/s 折算，故不做达标判定。' },
    mem_util:              { name: '显存利用率',          unit: '%', dir: 'low',  warn: null, neutral: true, desc: '显存占用峰值 ÷ 单卡 HBM 容量。本报告按 910B1 单卡 64 GB 折算，故不做达标判定。' },
    alloc_api_ratio:       { name: '分配 API 占比',       unit: '%', dir: 'low',  warn: 3,  desc: '显存分配/释放 API 耗时占 step 总耗时；> 3% 通常说明碎片整理或频繁换页。' },
    frag_ratio:            { name: '显存碎片率',          unit: '%', dir: 'low',  warn: 30, desc: '不可用空闲空间 ÷ 总空闲空间；> 30% 需检查分配生命周期与连续空闲块。' },
  };
  const METRIC_ORDER = Object.keys(METRIC_DEF);

  /* 报告固化的芯片假设：工具态是标题旁的下拉，报告态必须定死并印在图注里 */
  const CHIP = { name: '910B1', peakTF: 378.88, hbmGB: 64 };

  /* PHS「当前 / 优化后预估」双色，取自 app.js 的顶层 const（同为经典脚本可读其全局
     词法绑定，与读 REPORTS 同理），工具态调色报告自动跟随；用 typeof 兜底。 */
  const C_CUR = (typeof COLOR_CURRENT !== 'undefined' && COLOR_CURRENT) || '#d66f03';
  const C_EST = (typeof COLOR_ESTIMATED !== 'undefined' && COLOR_ESTIMATED) || '#039c59';

  const RESULT_DEF = {
    step_time_median: { name: '单步耗时中位数' },
    step_time_p90:    { name: '单步耗时 P90' },
    throughput:       { name: '吞吐' },
    e2e_duration:     { name: '采集窗口时长' },
  };

  // ── 通用工具 ────────────────────────────────────────────────
  const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  /** 在句读处截断，避免半个数字被切掉；报告态所有正文块都过这一关 */
  function clamp(s, n) {
    const t = String(s ?? '').replace(/\s+/g, ' ').trim();
    if (t.length <= n) return t;
    const cut = t.slice(0, n);
    const at = Math.max(cut.lastIndexOf('。'), cut.lastIndexOf('；'), cut.lastIndexOf('）'), cut.lastIndexOf('，'));
    return (at > n * 0.55 ? cut.slice(0, at + 1) : cut) + '…';
  }

  function fmtUs(us) {
    const a = Math.abs(us);
    if (a >= 1e6) return (us / 1e6).toFixed(a >= 1e7 ? 1 : 2) + ' s';
    if (a >= 1e3) return Math.round(us / 1e3) + ' ms';
    return Math.round(us) + ' µs';
  }
  function fmtMs(ms) {
    return Math.abs(ms) >= 1000 ? (ms / 1000).toFixed(2) + ' s' : Math.round(ms) + ' ms';
  }
  const PRIO_RANK = { P0: 0, P1: 1, P2: 2 };
  const badge = (p) => `<span class="rp-badge rp-badge--${String(p || '').toLowerCase()}">${esc(p || '—')}</span>`;

  /** 指标判定：报告优先用数据里写死的 status，否则按 dir/warn 阈值判 */
  function metricStatus(key, m) {
    if (m.status) return m.status;
    const d = METRIC_DEF[key];
    if (!d || d.warn == null) return 'na';
    const v = metricBarValue(key, m);
    if (v == null) return 'na';
    return d.dir === 'high' ? (v >= d.warn ? 'ok' : 'bad') : (v <= d.warn ? 'ok' : 'bad');
  }
  /** 条形/判定用的百分数。注意 step_cv 存的是分数(0.0002)但 display 是 '0.02%'，以 display 为准 */
  function metricBarValue(key, m) {
    if (key === 'mfu') return m.e2e_pct != null ? m.e2e_pct : (m.achieved_tflops != null ? +(m.achieved_tflops / CHIP.peakTF * 100).toFixed(1) : null);
    if (key === 'mem_util') return m.reserved_gb != null ? +(m.reserved_gb / CHIP.hbmGB * 100).toFixed(1) : null;
    if (m.display) { const p = parseFloat(m.display); return isNaN(p) ? null : p; }
    return typeof m.value === 'number' ? m.value : null;
  }
  function metricShown(key, m) {
    const v = metricBarValue(key, m);
    return v == null ? '—' : (m.display && !['mfu', 'mem_util'].includes(key) ? m.display : v + '%');
  }

  // ══════════════════════════════════════════════════════════
  //  图表（全部 SVG / CSS，无外部依赖）
  // ══════════════════════════════════════════════════════════

  /* ── 健康度仪表 ────────────────────────────────────────────
     不自绘：直接调用工具形态 app.js 的 `renderPhsGauge()`——分段刻度、等级色渐变、
     指针、增益弧全部同源，工具态改几何或调色时报告自动跟随，不会再出现两边一张图
     两个样子。它写的是 DOM 而不是字符串，所以这里只吐容器，由 mountGauges() 在
     innerHTML 落地之后回填。
     它内部引用 styles.css 的 `--fg` 系列变量与 `.phs-gauge-*` 类，而报告刻意不引
     styles.css，故 report.css 里补了一组同名规则并就地映射这几个变量。 */
  const gaugeBox = () => '<div class="rp-gauge phs-gauge-box" data-rp-gauge></div>';

  function mountGauges(phs) {
    if (typeof renderPhsGauge !== 'function') return;
    document.querySelectorAll('[data-rp-gauge]').forEach((el) => renderPhsGauge(el, phs || {}));
  }

  /* ── 分项雷达 ──────────────────────────────────────────────
     维度顺序、"预估缺失即回落到当前值"、N/A 记法与工具形态的 subChart 一致，双色也
     取自同两个常量；但不复用那边的 echarts 实例——echarts 走 CDN（离线归档取不到）
     且默认 canvas 渲染，按 CSS 像素栅格化，A4 打印必糊。故按同一份配置自绘 SVG。
     图例挪到 SVG 外的 HTML 行：原先压在 viewBox 底部，与下方两个维度的标签打架。 */
  function svgRadar(phs) {
    const items = (phs.subItems || []).filter((s) => s && s.name);
    const n = items.length;
    if (n < 3) return '';
    const estMap = Object.fromEntries((phs.estSubItems || []).map((s) => [s.name, s.value ?? null]));
    const cur = items.map((s) => (s.value == null ? null : s.value));
    // 工具态同款：某维度没给预估就沿用当前值，让虚线在该轴上与实线重合而不是塌到 0
    const est = items.map((s, i) => (estMap[s.name] == null ? cur[i] : estMap[s.name]));

    const W = 300, H = 214, CX = 150, CY = 106, R = 68, LR = R + 25;
    const ang = (i) => -Math.PI / 2 + (i * 2 * Math.PI) / n;
    const P = (i, pct, rad) => [CX + (rad || R) * (pct / 100) * Math.cos(ang(i)), CY + (rad || R) * (pct / 100) * Math.sin(ang(i))];
    const poly = (vals) => vals.map((v, i) => P(i, v == null ? 0 : v).map((k) => k.toFixed(1)).join(',')).join(' ');
    const ringPts = (p) => poly(items.map(() => p));

    // 隔环底色（工具态 splitArea 的等价物）：由外向内 白 / 淡 / 白 / 淡
    const bands = `<polygon points="${ringPts(75)}" fill="#f2f4f8"/><polygon points="${ringPts(50)}" fill="#fff"/><polygon points="${ringPts(25)}" fill="#f2f4f8"/>`;
    const rings = [25, 50, 75, 100].map((p) => `<polygon points="${ringPts(p)}" fill="none" stroke="rgba(33,33,33,.10)" stroke-width="1"/>`).join('');
    const axes = items.map((_, i) => { const [x, y] = P(i, 100); return `<line x1="${CX}" y1="${CY}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="rgba(33,33,33,.10)"/>`; }).join('');
    const dots = (vals, c) => vals.map((v, i) => { const [x, y] = P(i, v == null ? 0 : v); return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2" fill="${c}"/>`; }).join('');

    // 维度名 + 实测值分两行排（工具态 axisName 的 rich 双行）；锚点按象限贴边，
    // 避免最左/最右的标签被 viewBox 切掉
    const labels = items.map((s, i) => {
      const [x, y] = P(i, 100, LR);
      const dx = Math.cos(ang(i));
      const anchor = Math.abs(dx) < 0.25 ? 'middle' : dx > 0 ? 'start' : 'end';
      const val = s.value == null ? 'N/A' : s.value + '%';
      return `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" text-anchor="${anchor}" font-size="10" fill="#8a8a96">
        <tspan x="${x.toFixed(1)}">${esc(s.name)}</tspan>
        <tspan x="${x.toFixed(1)}" dy="12" font-size="11" font-weight="700" fill="${C_CUR}">${val}</tspan></text>`;
    }).join('');

    return `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="健康度分项雷达">
      ${bands}${rings}${axes}
      <polygon points="${poly(est)}" fill="${C_EST}22" stroke="${C_EST}" stroke-width="1.4" stroke-dasharray="4 3"/>
      <polygon points="${poly(cur)}" fill="${C_CUR}24" stroke="${C_CUR}" stroke-width="1.4"/>
      ${dots(est, C_EST)}${dots(cur, C_CUR)}
      ${labels}
    </svg>
    <div class="rp-stack-legend" style="justify-content:center">
      <span><i class="rp-swatch" style="background:${C_CUR}"></i>当前</span>
      <span><i class="rp-swatch" style="background:${C_EST}"></i>优化后预估</span>
      ${items.some((s) => s.value == null) ? '<span>无数据维度记 N/A，在图上按 0 收口</span>' : ''}
    </div>`;
  }

  /** step 时间构成堆叠条：一张图回答"时间去哪了" */
  function stackHtml(bd) {
    const KEY_COLOR = { compute: 'var(--rp-compute)', comm_exposed: 'var(--rp-comm)', bubble: 'var(--rp-bubble)', host_launch: 'var(--rp-host)', idle: 'var(--rp-idle)' };
    const items = bd.items || [];
    const segs = items.map((it) => {
      const c = KEY_COLOR[it.key] || 'var(--rp-idle)';
      return `<div class="rp-stack__seg" style="width:${it.pct}%;background:${c}">${it.pct >= 7 ? it.pct.toFixed(1) + '%' : ''}</div>`;
    }).join('');
    const legend = items.map((it) => `<span><i class="rp-swatch" style="background:${KEY_COLOR[it.key] || 'var(--rp-idle)'}"></i>${esc(it.label)} <b>${fmtMs(it.ms)}</b> · ${it.pct.toFixed(1)}%</span>`).join('');
    return `<div class="rp-stack">${segs}</div><div class="rp-stack-legend">${legend}</div>
      <p class="rp-figcap">四项之和 = ${fmtMs(bd.totalMs)}，与单步耗时中位数闭合。色板与 Timeline 泳道图例同源。</p>`;
  }

  /** 指标体检行：本次值 + 参考区间刻度 —— 报告态最核心的表达 */
  function bulletsHtml(metrics) {
    const rows = METRIC_ORDER.filter((k) => metrics[k]).map((k) => {
      const d = METRIC_DEF[k], m = metrics[k];
      const v = metricBarValue(k, m);
      const st = metricStatus(k, m);
      const w = v == null ? 0 : Math.min(100, Math.max(1.5, v));
      const thr = d.warn == null ? '' : `<i class="rp-bullet__thr" style="left:${Math.min(97, d.warn)}%" data-l="${d.dir === 'high' ? '≥' : '≤'}${d.warn}"></i>`;
      const cls = st === 'ok' ? 'is-ok' : st === 'bad' ? 'is-bad' : st === 'warn' ? 'is-warn' : '';
      const note = m.note ? `<div class="rp-bullet__note">${esc(clamp(m.note, 96))}</div>` : '';
      return `<div class="rp-bullet rp-block">
        <div class="rp-bullet__name">${esc(d.name)}</div>
        <div class="rp-bullet__track"><i class="rp-bullet__fill ${cls}" style="width:${w}%"></i>${thr}</div>
        <div class="rp-bullet__val">${metricShown(k, m)}<small> ${st === 'ok' ? '达标' : st === 'bad' ? '越界' : ''}</small></div>
        ${note}
      </div>`;
    });
    return rows.join('');
  }

  /** 收益 × 难度四象限：一张图回答"先做什么" */
  function svgQuadrant(actions) {
    const withNum = actions.filter((a) => a.benefitNum != null);
    if (!withNum.length) return '';
    const W = 560, H = 300, L = 52, R = 18, T = 16, B = 44;
    const DX = { '低': 1, '中': 2, '高': 3 };
    const maxB = Math.max(...withNum.map((a) => a.benefitNum)) * 1.18;
    const px = (d) => L + ((DX[d] || 2) - 0.5) / 3 * (W - L - R);
    const py = (b) => H - B - (b / maxB) * (H - T - B);
    const midY = py(maxB / 2);
    const bubbles = withNum.map((a, i) => {
      const sameCol = withNum.filter((o) => o.difficulty === a.difficulty);
      const k = sameCol.indexOf(a);
      const off = sameCol.length > 1 ? (k - (sameCol.length - 1) / 2) * 26 : 0;
      const x = px(a.difficulty) + off, y = py(a.benefitNum);
      const rr = 9 + Math.min(9, a.benefitNum * 0.34);
      const c = a.priority === 'P0' ? '#d23b3b' : a.priority === 'P1' ? '#c15a1c' : '#5d6b99';
      return `<g><circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${rr.toFixed(1)}" fill="${c}" fill-opacity=".82"/>
        <text x="${x.toFixed(1)}" y="${(y + 3.6).toFixed(1)}" text-anchor="middle" font-size="10.5" font-weight="700" fill="#fff">${a.id}</text>
        <text x="${x.toFixed(1)}" y="${(y - rr - 4).toFixed(1)}" text-anchor="middle" font-size="9" fill="#4a5160">−${a.benefitNum}%</text></g>`;
    }).join('');
    const yticks = [0, maxB / 2, maxB].map((v) => `<g><line x1="${L}" y1="${py(v).toFixed(1)}" x2="${W - R}" y2="${py(v).toFixed(1)}" stroke="#eceef1"/><text x="${L - 7}" y="${(py(v) + 3.5).toFixed(1)}" text-anchor="end" font-size="9" fill="#9aa1ad">${v.toFixed(0)}%</text></g>`).join('');
    const xlabels = ['低', '中', '高'].map((d) => `<text x="${px(d).toFixed(1)}" y="${H - B + 17}" text-anchor="middle" font-size="10.5" fill="#4a5160">修改难度 ${d}</text>`).join('');
    return `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="收益与难度四象限">
      <rect x="${L}" y="${T}" width="${(px('中') - L).toFixed(1)}" height="${(midY - T).toFixed(1)}" fill="#e6f4ec" fill-opacity=".7"/>
      <text x="${L + 8}" y="${T + 15}" font-size="9.5" fill="#14713f" font-weight="600">优先做：高收益 · 低难度</text>
      ${yticks}
      <line x1="${L}" y1="${T}" x2="${L}" y2="${H - B}" stroke="#b9c0cc"/>
      <line x1="${L}" y1="${H - B}" x2="${W - R}" y2="${H - B}" stroke="#b9c0cc"/>
      ${bubbles}${xlabels}
      <text x="${L - 7}" y="${T + 4}" text-anchor="end" font-size="9" fill="#9aa1ad"></text>
      <text transform="translate(13,${(T + (H - T - B) / 2).toFixed(1)}) rotate(-90)" text-anchor="middle" font-size="10" fill="#4a5160">预计单步耗时收益</text>
    </svg>`;
  }

  /** 分类占比环 */
  function svgDonut(cats) {
    const PAL = ['#2f6fce', '#c15a1c', '#0f9b8e', '#8b3fb5', '#b8860b', '#5d6b99'];
    const total = cats.reduce((s, c) => s + c.pct, 0);
    const CX = 82, CY = 82, R = 62, IR = 38;
    let acc = -Math.PI / 2;
    const arcs = cats.map((c, i) => {
      const a = (c.pct / Math.max(total, 100)) * 2 * Math.PI;
      const [s, e] = [acc, acc + a]; acc = e;
      const p = (r, t) => [CX + r * Math.cos(t), CY + r * Math.sin(t)];
      const [x1, y1] = p(R, s), [x2, y2] = p(R, e), [x3, y3] = p(IR, e), [x4, y4] = p(IR, s);
      const lg = a > Math.PI ? 1 : 0;
      return `<path d="M${x1.toFixed(1)} ${y1.toFixed(1)} A${R} ${R} 0 ${lg} 1 ${x2.toFixed(1)} ${y2.toFixed(1)} L${x3.toFixed(1)} ${y3.toFixed(1)} A${IR} ${IR} 0 ${lg} 0 ${x4.toFixed(1)} ${y4.toFixed(1)} Z" fill="${PAL[i % PAL.length]}"/>`;
    }).join('');
    const rest = Math.max(0, 100 - total);
    const legend = cats.map((c, i) => `<span><i class="rp-swatch" style="background:${PAL[i % PAL.length]}"></i>${esc(c.label)} ${c.pct.toFixed(1)}%</span>`).join('') +
      (rest > 1 ? `<span><i class="rp-swatch" style="background:#eceef1"></i>其他 ${rest.toFixed(1)}%</span>` : '');
    return `<svg viewBox="0 0 164 164" width="132" role="img" aria-label="算子分类占比">
      <circle cx="${CX}" cy="${CY}" r="${(R + IR) / 2}" fill="none" stroke="#eceef1" stroke-width="${R - IR}"/>
      ${arcs}
      <text x="${CX}" y="${CY - 2}" text-anchor="middle" font-size="17" font-weight="700" fill="#1c1f26">${total.toFixed(0)}%</text>
      <text x="${CX}" y="${CY + 13}" text-anchor="middle" font-size="9" fill="#9aa1ad">已归类</text>
    </svg><div class="rp-stack-legend" style="flex-direction:column;gap:3px">${legend}</div>`;
  }

  /* ── 泳道快照（最低成本实现）──────────────────────────────
     直接消费 chart-data.js 里已有的 window.SWIMLANE_DATA[reportId][actionId]，
     不加载 swimlane-data.js（15 MB）、不复用 swimlane.js 渲染器（它面向可缩放
     的交互画布）。这里只做一件事：把 tasks 段按时间轴摊成定宽 SVG，把
     annotations 的 range 画成标注括号 —— 即"我指给你看"。 */
  function swimlaneSvg(snap) {
    const lanes = (snap.data || []).filter((l) => l.tasks && l.tasks.length);
    if (!lanes.length) return '';
    let t0 = Infinity, t1 = -Infinity;
    lanes.forEach((l) => l.tasks.forEach((t) => { t0 = Math.min(t0, t.execStart); t1 = Math.max(t1, t.execEnd); }));
    const span = (t1 - t0) || 1;
    const W = 1000, L = 214, RGT = 16, LH = 30, GAP = 17, TOP = 4;
    const plotW = W - L - RGT;
    const x = (t) => L + ((t - t0) / span) * plotW;
    const ann = (snap.annotations || []).filter((a) => a.type === 'range' && a.startTime != null);
    /* 主线：红色括号标出的区间是这张图要读者先看懂的关系，区间外的任务只作背景
       参考——降低不透明度让主线从一整排同权重的色块里跳出来，而不是靠读者自己
       去逐条对时间戳（白皮书"复杂架构插图"方法论里"必须有主线"的画法）。 */
    const hi = ann[0] || null;
    const axisY = TOP + lanes.length * (LH + GAP) - GAP + 16;
    const annY = axisY + 18;
    const H = annY + (ann.length ? 30 + Math.ceil((ann[0].label || '').length / 78) * 11 : 6);

    const laneSvg = lanes.map((lane, li) => {
      const y = TOP + li * (LH + GAP);
      const parts = String(lane.coreType || '').split(' · ').slice(0, 3);
      const label = parts.map((p, i) => `<tspan x="0" dy="${i ? 11 : 0}">${esc(p.length > 34 ? p.slice(0, 33) + '…' : p)}</tspan>`).join('');
      const bars = lane.tasks.map((t) => {
        const x0 = x(t.execStart), w = Math.max(1, x(t.execEnd) - x0);
        const c = SEM_COLOR[t.semanticLabel] || '#8b93a1';
        const dur = fmtUs(t.execEnd - t.execStart);
        const zh = SEM_ZH[t.semanticLabel] || t.semanticLabel || '';
        const txt = w >= 122 ? `${zh} ${dur}` : w >= 50 ? dur : '';
        const inMain = !hi || (t.execEnd > hi.startTime && t.execStart < hi.endTime);
        const stroke = inMain && hi ? ' stroke="#1c1f26" stroke-width="0.6"' : '';
        return `<g opacity="${inMain ? 1 : 0.32}"><rect x="${x0.toFixed(1)}" y="${y}" width="${w.toFixed(1)}" height="${LH}" rx="2" fill="${c}"${stroke}/>` +
          (txt ? `<text x="${(x0 + w / 2).toFixed(1)}" y="${y + LH / 2 + 3.6}" text-anchor="middle" font-size="9.5" fill="#fff" font-weight="600">${esc(txt)}</text>` : '') + '</g>';
      }).join('');
      return `<g><text x="0" y="${y + (parts.length > 1 ? 11 : LH / 2 + 3)}" font-size="8.6" fill="#4a5160">${label}</text>${bars}</g>`;
    }).join('');

    const ticks = Array.from({ length: 6 }, (_, i) => {
      const t = t0 + (span * i) / 5, xx = x(t);
      return `<g><line x1="${xx.toFixed(1)}" y1="${axisY - 11}" x2="${xx.toFixed(1)}" y2="${axisY - 6}" stroke="#b9c0cc"/><text x="${xx.toFixed(1)}" y="${axisY + 3}" text-anchor="${i === 0 ? 'start' : i === 5 ? 'end' : 'middle'}" font-size="8.6" fill="#9aa1ad">${fmtUs(t - t0)}</text></g>`;
    }).join('');

    const annSvg = ann.slice(0, 1).map((a) => {
      const xa = x(a.startTime), xb = x(a.endTime), mid = (xa + xb) / 2;
      const words = String(a.label || '');
      const lines = words.match(/.{1,78}/g) || [];
      const txt = lines.map((s, i) => `<tspan x="${mid.toFixed(1)}" dy="${i ? 11 : 0}">${esc(s)}</tspan>`).join('');
      return `<g>
        <path d="M${xa.toFixed(1)} ${annY} v6 H${xb.toFixed(1)} v-6" fill="none" stroke="#d23b3b" stroke-width="1.4"/>
        <text x="${mid.toFixed(1)}" y="${annY + 20}" text-anchor="middle" font-size="9.3" fill="#b02525" font-weight="600">${txt}</text>
      </g>`;
    }).join('');

    const used = [...new Set(lanes.flatMap((l) => l.tasks.map((t) => t.semanticLabel)))].filter(Boolean);
    const legend = used.map((s) => `<span><i class="rp-swatch" style="background:${SEM_COLOR[s] || '#8b93a1'}"></i>${esc(SEM_ZH[s] || s)}</span>`).join('');

    return `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="泳道快照">
      <line x1="${L}" y1="${axisY - 6}" x2="${W - RGT}" y2="${axisY - 6}" stroke="#d8dce3"/>
      ${laneSvg}${ticks}${annSvg}
    </svg><div class="rp-stack-legend">${legend}</div>${hi ? '<p class="rp-figcap">区间外任务已降低对比度，仅作时间上下文参考。</p>' : ''}`;
  }

  /** 各 rank 的 step 构成横条 + 首次进入通信时刻标记 */
  function rankBarsHtml(rs) {
    const KEYS = [
      ['computeMs', '计算', 'var(--rp-compute)'],
      ['commExposedMs', '暴露通信', 'var(--rp-comm)'],
      ['commWaitMs', '通信等待/空泡', 'var(--rp-bubble)'],
      ['freeMs', '空闲', 'var(--rp-idle)'],
    ];
    const rows = rs.rows.map((r) => {
      const total = r.stepMs || KEYS.reduce((s, [k]) => s + (r[k] || 0), 0);
      const segs = KEYS.map(([k, , c]) => `<i class="rp-hbar__f" style="width:${((r[k] || 0) / total * 100).toFixed(2)}%;background:${c}"></i>`).join('');
      const entry = r.commEntryMs != null ? `<i style="position:absolute;left:${(r.commEntryMs / total * 100).toFixed(2)}%;top:-3px;bottom:-3px;width:2px;background:#1c1f26"></i>` : '';
      return `<div class="rp-hbar">
        <div class="rp-hbar__n" style="font-family:inherit;font-size:10.5px">rank ${r.rank}${r.verdict === 'slow' ? ' <span class="rp-badge rp-badge--p0" style="font-size:8.5px">慢</span>' : ''}</div>
        <div class="rp-hbar__t" style="position:relative">${segs}${entry}</div>
        <div class="rp-hbar__v">${fmtMs(r.stepMs)}</div>
      </div>`;
    }).join('');
    const legend = KEYS.map(([, l, c]) => `<span><i class="rp-swatch" style="background:${c}"></i>${l}</span>`).join('') +
      '<span><i class="rp-swatch" style="background:#1c1f26;width:2px;height:11px;border-radius:0"></i>首次进入通信</span>';
    return rows + `<div class="rp-stack-legend">${legend}</div>`;
  }

  /** 算子 Top-N 横条 */
  function opBarsHtml(top) {
    const CT = { HCCL: '#c15a1c', AI_CPU: '#8b3fb5', MIX_AIC: '#2f6fce', AI_CORE: '#1a8f52', AI_VECTOR_CORE: '#0f9b8e' };
    const max = Math.max(...top.map((t) => t.pct));
    return top.map((t) => `<div class="rp-hbar rp-block">
      <div class="rp-hbar__n" title="${esc(t.name)}">${esc(t.name)}</div>
      <div class="rp-hbar__t"><i class="rp-hbar__f" style="width:${(t.pct / max * 100).toFixed(1)}%;background:${CT[t.coreType] || '#5d6b99'}"></i></div>
      <div class="rp-hbar__v">${t.pct.toFixed(1)}% · ${fmtMs(t.totalMs)}</div>
    </div>`).join('') +
      `<div class="rp-stack-legend">${Object.entries(CT).map(([k, c]) => `<span><i class="rp-swatch" style="background:${c}"></i>${k}</span>`).join('')}</div>`;
  }

  // ══════════════════════════════════════════════════════════
  //  页面装配
  // ══════════════════════════════════════════════════════════

  let PAGES = [];   // {id, title, sub, html, cover?}
  function page(id, title, html, opts) {
    PAGES.push(Object.assign({ id, title, html }, opts || {}));
  }

  /** 3.N 形式的问题 ↔ 第 2 章行动清单 id（app.js 同款约定） */
  const issueOrdinal = (id) => { const m = /^3\.(\d+)$/.exec(String(id)); return m ? +m[1] : null; };

  function renderReport(r) {
    PAGES = [];
    const actById = new Map((r.actions || []).map((a) => [a.id, a]));
    const swim = (window.SWIMLANE_DATA || {})[r.id] || {};

    // ── 1. 封面 ─────────────────────────────────────────────
    const facts = [
      ['并行配置 / 采集范围', r.meta?.range || r.subtitle],
      ['分析日期', r.reportDate],
      ['软件版本', r.meta?.version],
      ['数据路径', r.meta?.dataPath],
      ['采集档位', r.results?.profiling_level?.raw],
      ['分析技能链', (r.meta?.skills || []).join(' · ')],
    ].filter(([, v]) => v);
    page('cover', '封面', `
      <div class="rp-cover__top"><span>性能分析</span><i class="rp-cover__rule"></i><span>${esc(r.reportDate)}</span></div>
      <div class="rp-cover__mid">
        <span class="rp-cover__type">${esc(r.taskType || '性能分析')}</span>
        <h1 class="rp-cover__title">${esc(r.title)}</h1>
        <p class="rp-cover__sub">${esc(r.subtitle || '')}</p>
        <div class="rp-cover__score">
          <div>
            <div class="rp-cover__scorenum">${r.phs?.current ?? '—'}</div>
            <div class="rp-note" style="text-align:center">性能健康度 / 100</div>
          </div>
          <div class="rp-cover__scoremeta">
            当前评级 <b>${esc(r.phs?.grade || '—')}</b><br>
            落实本报告行动清单后预估 <b>${r.phs?.estimated ?? '—'}</b>（${esc(r.phs?.estGrade || '—')}）<br>
            <span class="rp-note">${esc(clamp(r.summary?.maxGain, 72))}</span>
          </div>
        </div>
        <dl class="rp-cover__facts">
          ${facts.map(([k, v]) => `<div class="rp-cover__fact"><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join('')}
        </dl>
      </div>
      <div class="rp-cover__bot">本报告由性能分析工作台生成，结论基于上述落盘数据；口径与阈值见附录 A，数据来源与可复现信息见附录 B。</div>
    `, { cover: true });

    // ── 2. 执行摘要（总）────────────────────────────────────
    const bd = r.breakdown;
    // 分项不足 3 维出不了雷达，此时整块（含图注）不出，不留空图占位
    const radar = svgRadar(r.phs || {});
    // 头号杠杆：全部行动里预计收益最大的单一动作。借白皮书 callout 的做法——只放
    // 一件事、一句话，和下方 rp-verdicts 那三张说明卡分工："记住哪一件事" vs "为什么"。
    const topAction = (r.actions || []).filter((a) => a.benefitNum != null).sort((a, b) => b.benefitNum - a.benefitNum)[0];
    page('summary', '执行摘要', `
      <p class="rp-kicker">执行摘要</p>
      <h2 class="rp-h2">${esc(clamp(r.summary?.conclusion, 96))}</h2>
      <p class="rp-lead">健康度 ${r.phs?.current} → 预估 ${r.phs?.estimated}；本报告共列出 ${(r.issues || []).length} 个问题、${(r.actions || []).length} 项行动，按预计收益排序。</p>
      ${topAction ? `<div class="rp-callout rp-block"><b>头号杠杆：</b>优先修复 ${esc(topAction.id)} ${esc(clamp(topAction.problem, 60))}，预计单步耗时最多降低 ${esc(topAction.benefit)}——本报告收益最大的单一动作。</div>` : ''}
      <div class="rp-grid2 rp-block" style="align-items:start;margin-bottom:6mm">
        <div>${gaugeBox()}<p class="rp-figcap">环上小段按等级色渐变：实心段为当前健康度（指针所指），半透明段为落实行动清单后的可增益区间，外侧箭头指向预估位置。与工具形态同一份仪表盘。</p></div>
        <div>${radar ? `${radar}<p class="rp-figcap">各分项 0–100，实线为当前、虚线为优化后预估；某维度未给预估时虚线与实线在该轴重合。</p>` : ''}</div>
      </div>
      <dl class="rp-verdicts rp-block">
        <div class="rp-verdict"><dt>核心结论</dt><dd>${esc(r.summary?.conclusion || '—')}</dd></div>
        <div class="rp-verdict rp-verdict--top"><dt>头号瓶颈</dt><dd>${esc(clamp(r.summary?.topBottleneck, 200))}</dd></div>
        <div class="rp-verdict rp-verdict--gain"><dt>收益上限</dt><dd>${esc(r.summary?.maxGain || '—')}</dd></div>
      </dl>
      ${bd ? `<div class="rp-block" style="margin-top:6mm"><h3 class="rp-h3">单步时间去向</h3>${stackHtml(bd)}</div>` : ''}
    `);

    // ── 3. 指标体检（条件：有 results 或 metrics）──────────
    if (r.results || r.metrics) {
      const res = r.results || {};
      const kpis = Object.keys(RESULT_DEF).filter((k) => res[k]).map((k) => `
        <div class="rp-kpi"><div class="rp-kpi__k">${RESULT_DEF[k].name}</div>
        <div class="rp-kpi__v">${esc(res[k].raw || res[k].value)}</div>
        <div class="rp-kpi__n">${esc(clamp(res[k].note, 34))}</div></div>`).join('');
      const trust = [
        ['有效 step', res.active_steps?.raw], ['warmup', res.warmup_steps?.raw],
        ['采集档位', res.profiling_level?.raw],
        ['step 抖动 CV', r.metrics?.step_cv ? metricShown('step_cv', r.metrics.step_cv) : null],
      ].filter(([, v]) => v);
      page('metrics', '指标体检', `
        <p class="rp-kicker">第 1 章 · 指标体检</p>
        <h2 class="rp-h2">${esc(bd ? `单步 ${fmtMs(bd.totalMs)} 中只有 ${(bd.items.find((i) => i.key === 'compute')?.pct ?? 0).toFixed(0)}% 在算` : '关键指标与参考区间对照')}</h2>
        <p class="rp-lead">每项右侧为本次实测值，条上黑色刻度为判定阈值——越过刻度即判为越界。阈值定义见附录 A。</p>
        ${kpis ? `<div class="rp-kpis rp-block" style="margin-bottom:5mm">${kpis}</div>` : ''}
        ${trust.length ? `<p class="rp-note rp-block" style="margin:0 0 5mm;padding:6px 10px;background:var(--surface-2);border-radius:6px">
          <b>可信度口径</b> · ${trust.map(([k, v]) => `${k} ${esc(v)}`).join(' · ')} —— 这几项决定上面所有数字与下面所有比率能不能信。</p>` : ''}
        ${r.metrics ? `<h3 class="rp-h3">结构指标 vs 参考区间</h3><div class="rp-bullets">${bulletsHtml(r.metrics)}</div>
          <p class="rp-figcap">MFU / 显存利用率的分母为固化假设：${CHIP.name}（BF16 ${CHIP.peakTF} TF/s，单卡 ${CHIP.hbmGB} GB HBM）。落盘数据不含芯片型号，换型号需重算。</p>` : ''}
      `);
    }

    // ── 4. 时间去向解剖（条件：有泳道快照）──────────────────
    const firstSwimKey = Object.keys(swim)[0];
    if (firstSwimKey) {
      const snap = swim[firstSwimKey];
      const act = actById.get(+firstSwimKey);
      page('timeline', '时间去向', `
        <p class="rp-kicker">第 2 章 · 时间去向解剖</p>
        <h2 class="rp-h2">${esc(clamp(act?.problem || '关键路径上的时间构成', 88))}</h2>
        <p class="rp-lead">下图是从 trace 中截取的单步泳道定格，红色括号标出的是本报告关注的区间。报告态不提供缩放——需要逐算子游走时请回到工具形态。</p>
        <div class="rp-block">${swimlaneSvg(snap)}</div>
        <p class="rp-figcap">数据来源：${esc(clamp(snap.source, 190))}</p>
        ${bd ? `<div class="rp-block" style="margin-top:6mm"><h3 class="rp-h3">同一单步的时间构成</h3>${stackHtml(bd)}</div>` : ''}
      `);
    }

    // ── 5. 多卡一致性（条件：有 rankStats）──────────────────
    if (r.rankStats?.rows?.length) {
      const rs = r.rankStats;
      page('ranks', '多卡一致性', `
        <p class="rp-kicker">第 3 章 · 多卡一致性</p>
        <h2 class="rp-h2">各卡 step 耗时几乎相同（极差 ${rs.spreadPct?.toFixed(2)}%），问题不在"哪张卡慢"，而在时间构成</h2>
        <p class="rp-lead">这是本页最容易被误读的一点：只看 step 时长永远找不出问题卡。真正的信号在计算/通信等待的拆分，以及首次进入通信的时刻——${rs.slowRanks?.length ? `rank ${rs.slowRanks.join('/')} 明显更晚进入通信，其余卡因此空等。` : '见下方构成条。'}</p>
        <div class="rp-block">${rankBarsHtml(rs)}</div>
        <p class="rp-figcap">按规则 10 只列代表 rank。中位数 ${fmtMs(rs.median)}，极差 ${rs.spreadPct?.toFixed(2)}%${rs.slowRanks?.length ? `，判定慢卡 rank ${rs.slowRanks.join(', ')}` : ''}。</p>
        <h3 class="rp-h3">代表 rank 明细</h3>
        <table class="rp-table rp-block"><thead><tr><th>Rank</th><th class="num">step</th><th class="num">计算</th><th class="num">暴露通信</th><th class="num">通信等待</th><th class="num">空闲</th><th class="num">首次进通信</th></tr></thead><tbody>
          ${rs.rows.map((x) => `<tr class="${x.verdict === 'slow' ? 'is-slow' : ''}"><td>${x.rank}${x.verdict === 'slow' ? ' ⚑' : ''}</td><td class="num">${fmtMs(x.stepMs)}</td><td class="num">${fmtMs(x.computeMs)}</td><td class="num">${fmtMs(x.commExposedMs)}</td><td class="num">${fmtMs(x.commWaitMs)}</td><td class="num">${fmtMs(x.freeMs)}</td><td class="num">${fmtMs(x.commEntryMs)}</td></tr>`).join('')}
        </tbody></table>
      `);
    }

    // ── 6. 算子热点（条件：有 opStats）──────────────────────
    if (r.opStats?.top?.length) {
      const os = r.opStats;
      page('ops', '算子热点', `
        <p class="rp-kicker">第 4 章 · 算子热点</p>
        <h2 class="rp-h2">前 ${os.top.length} 个算子占单步 ${os.top.reduce((s, t) => s + t.pct, 0).toFixed(1)}%</h2>
        <p class="rp-lead">这一章是结论的证据面：把上文引用的算子放在同一张图里，读者不必回到工具就能验一遍。</p>
        <div class="rp-grid-72 rp-block" style="align-items:start">
          <div>${opBarsHtml(os.top)}</div>
          <div>${os.byCategory?.length ? svgDonut(os.byCategory) : ''}</div>
        </div>
        <h3 class="rp-h3">Top 算子明细</h3>
        <table class="rp-table rp-block"><thead><tr><th>算子</th><th>Type</th><th>Core</th><th class="num">次数</th><th class="num">总耗时</th><th class="num">占比</th><th class="num">均值</th><th>Shape</th></tr></thead><tbody>
          ${os.top.map((t) => `<tr><td><code>${esc(t.name)}</code></td><td>${esc(t.type)}</td><td>${esc(t.coreType)}</td><td class="num">${t.count}</td><td class="num">${fmtMs(t.totalMs)}</td><td class="num">${t.pct.toFixed(1)}%</td><td class="num">${(t.avgUs / 1000).toFixed(2)} ms</td><td><code>${esc(t.shape)}</code></td></tr>`).join('')}
        </tbody></table>
      `);
    }

    // ── 7. 问题逐条（分）────────────────────────────────────
    const issues = (r.issues || []).slice().sort((a, b) => {
      const ba = actById.get(issueOrdinal(a.id))?.benefitNum ?? -1;
      const bb = actById.get(issueOrdinal(b.id))?.benefitNum ?? -1;
      if (ba !== bb) return bb - ba;
      return (PRIO_RANK[a.priority] ?? 9) - (PRIO_RANK[b.priority] ?? 9);
    });
    const maxBenefit = Math.max(1, ...(r.actions || []).map((a) => a.benefitNum || 0));
    issues.forEach((is, i) => {
      const ord = issueOrdinal(is.id);
      const act = actById.get(ord);
      const snap = ord != null ? swim[ord] : null;
      const steps = (is.steps || []).slice(0, 3);
      const moreSteps = (is.steps || []).length - steps.length;
      const impactQ = act?.benefitNum != null
        ? `<div class="rp-impact__q">−${act.benefitNum}%<small>预计单步耗时收益</small></div>
           <div class="rp-impact__bar"><i style="width:${(act.benefitNum / maxBenefit * 100).toFixed(0)}%"></i></div>`
        : `<div class="rp-impact__q">${esc(is.priority)}<small>优先级（收益未量化）</small></div>`;
      page('issue-' + is.id, `${is.id} ${is.title}`, `
        <p class="rp-kicker">第 5 章 · 问题 ${i + 1} / ${issues.length}<a class="rp-tool-link rp-screen-only" href="index_v3.html" target="_blank" rel="noopener">在工具中查看 →</a></p>
        <div class="rp-issue__head">
          <span class="rp-issue__id">${esc(is.id)}</span>
          <span class="rp-issue__title">${esc(is.title)}</span>
          ${badge(is.priority)}
        </div>
        <div class="rp-issue__meta">
          ${act ? `<span>预计收益 <b>${esc(act.benefit)}</b></span><span>修改难度 <b>${esc(act.difficulty)}</b></span>` : ''}
          ${is.codeLocations?.length ? `<span>代码位置 <b>${esc(is.codeLocations.join(', '))}</b></span>` : ''}
        </div>
        <div class="rp-grid-72 rp-block" style="align-items:start">
          <div>${snap ? swimlaneSvg(snap) : `<div class="rp-card"><div class="rp-card__t">影响</div><div style="font-size:11px;line-height:1.6">${esc(clamp(is.impact, 260))}</div></div>`}</div>
          <div class="rp-impact">${impactQ}<div class="rp-impact__txt">${esc(clamp(is.impact, 150))}</div></div>
        </div>
        <div class="rp-evidence rp-block"><span class="rp-evidence__t">证据</span>${esc(clamp(is.evidence, 300))}<span class="rp-note"> 完整证据见附录 C。</span></div>
        ${act ? `<p class="rp-note rp-block" style="margin:8px 0 0"><b>改动位置</b> · ${esc(clamp(act.location, 150))}</p>` : ''}
        <h3 class="rp-h3">修复步骤</h3>
        <ol class="rp-steps rp-block">${steps.map((s) => `<li>${esc(clamp(s, 130))}</li>`).join('')}</ol>
        ${moreSteps > 0 ? `<p class="rp-note" style="margin-top:5px">另有 ${moreSteps} 项步骤，见附录 C。</p>` : ''}
        ${is.verification ? `<div class="rp-verify rp-block"><b>验证方式</b> · ${esc(clamp(is.verification, 190))}</div>` : ''}
      `, { sub: true });
    });

    // ── 8. 行动计划（总）────────────────────────────────────
    if (r.actions?.length) {
      const sorted = r.actions.slice().sort((a, b) => (b.benefitNum ?? -1) - (a.benefitNum ?? -1));
      // 收益未量化的报告出不了四象限，此时整块（含图注）不出，不留空图占位
      const quad = svgQuadrant(r.actions);
      page('actions', '行动计划', `
        <p class="rp-kicker">第 6 章 · 行动计划</p>
        <h2 class="rp-h2">${esc(clamp(r.summary?.maxGain, 88))}</h2>
        ${quad ? `<p class="rp-lead">下图横轴为修改难度、纵轴为预计单步收益，气泡大小同收益、颜色同优先级。左上角绿色区域是应当先做的部分。</p>
        <div class="rp-block">${quad}</div>` : '<p class="rp-lead">本报告的行动项未给出量化收益，按优先级排序如下。</p>'}
        <table class="rp-table rp-block" style="margin-top:5mm"><thead><tr><th>#</th><th>优先级</th><th>问题</th><th class="num">预计收益</th><th>难度</th><th>改动位置</th></tr></thead><tbody>
          ${sorted.map((a) => `<tr><td class="num">${a.id}</td><td>${badge(a.priority)}</td><td>${esc(clamp(a.problem, 72))}</td><td class="num">${esc(a.benefit)}</td><td>${esc(a.difficulty)}</td><td>${esc(clamp(a.location, 78))}</td></tr>`).join('')}
        </tbody></table>
      `);
    }

    // ── 9. 阴性结论（条件：有 noProblems）───────────────────
    if (r.noProblems?.length) {
      page('noproblem', '已排查未见异常', `
        <p class="rp-kicker">第 7 章 · 已排查未见异常</p>
        <h2 class="rp-h2">以下 ${r.noProblems.length} 项已排查，未发现问题</h2>
        <p class="rp-lead">阴性结论与阳性结论同等重要：它界定了本次分析的覆盖面，也避免后续重复排查同一方向。</p>
        <div class="rp-checks">${r.noProblems.map((s) => `<div class="rp-check rp-block"><span class="rp-check__m">✓</span><span>${esc(s)}</span></div>`).join('')}</div>
      `);
    }

    // ── 10. 附录 A：口径与阈值 ──────────────────────────────
    if (r.metrics) {
      page('appendix-a', '附录 A · 指标口径与阈值', `
        <p class="rp-kicker">附录 A</p>
        <h2 class="rp-h2">指标口径与判定阈值</h2>
        <p class="rp-lead">工具形态里这些说明藏在指标卡的 <code>?</code> 提示中；报告形态必须印出来，否则打印后即丢失。</p>
        <table class="rp-table rp-block"><thead><tr><th>指标</th><th class="num">本次</th><th>判定阈值</th><th>口径</th></tr></thead><tbody>
          ${METRIC_ORDER.filter((k) => r.metrics[k]).map((k) => {
            const d = METRIC_DEF[k], st = metricStatus(k, r.metrics[k]);
            return `<tr><td>${esc(d.name)}</td><td class="num">${metricShown(k, r.metrics[k])}</td>
              <td>${d.neutral ? '不判定（分母为假设值）' : d.warn == null ? '需对比理论值' : `${d.dir === 'high' ? '≥' : '≤'} ${d.warn}${d.unit}`}${st === 'ok' || st === 'bad' ? ` <span class="rp-badge rp-badge--${st === 'ok' ? 'ok' : 'p0'}" style="font-size:8.5px">${st === 'ok' ? '达标' : '越界'}</span>` : ''}</td>
              <td>${esc(d.desc)}</td></tr>`;
          }).join('')}
        </tbody></table>
      `, { sub: true });
    }

    // ── 11. 附录 B：数据来源与可复现 ────────────────────────
    const df = r.diskFileInfo || {};
    const metaRows = [
      ['分析日期', r.meta?.date], ['数据路径', r.meta?.dataPath], ['数据范围', r.meta?.range],
      ['软件版本', r.meta?.version], ['Advisor 状态', r.meta?.advisorStatus], ['输出位置', r.meta?.output],
      ['分析技能链', (r.meta?.skills || []).join('、')],
      ['落盘目录', df.dir], ['数据来源', df.source], ['模型', df.model],
    ].filter(([, v]) => v);
    page('appendix-b', '附录 B · 数据来源与可复现', `
      <p class="rp-kicker">附录 B</p>
      <h2 class="rp-h2">数据来源与可复现信息</h2>
      <p class="rp-lead">本报告所有结论均可依下列信息复现。缺失字段表示落盘数据未提供，未做推测。</p>
      <table class="rp-table rp-block"><tbody>
        ${metaRows.map(([k, v]) => `<tr><th style="width:130px;border-bottom:1px solid var(--border-subtle);font-weight:600">${esc(k)}</th><td><code>${esc(v)}</code></td></tr>`).join('')}
        <tr><th style="width:130px;border-bottom:1px solid var(--border-subtle);font-weight:600">芯片假设</th><td>${CHIP.name} · BF16 峰值 ${CHIP.peakTF} TF/s · 单卡 HBM ${CHIP.hbmGB} GB（落盘数据不含型号，为报告固化假设）</td></tr>
      </tbody></table>
    `, { sub: true });

    // ── 12. 附录 C：问题完整证据 ────────────────────────────
    if (issues.length) {
      page('appendix-c', '附录 C · 问题完整证据', `
        <p class="rp-kicker">附录 C</p>
        <h2 class="rp-h2">问题完整证据与步骤</h2>
        <p class="rp-lead">正文的问题页对证据与步骤做了裁剪以保证单页版面；此处保留全文。</p>
        ${issues.map((is) => `<div class="rp-block" style="margin-bottom:5mm">
          <div style="font-size:11.5px;font-weight:600;margin-bottom:3px">${esc(is.id)} ${esc(is.title)}</div>
          <div class="rp-note"><b>证据</b> ${esc(is.evidence)}</div>
          ${is.impact ? `<div class="rp-note" style="margin-top:3px"><b>影响</b> ${esc(is.impact)}</div>` : ''}
          ${is.steps?.length ? `<div class="rp-note" style="margin-top:3px"><b>步骤</b> ${is.steps.map((s, i) => `${i + 1}. ${esc(s)}`).join(' ')}</div>` : ''}
          ${is.verification ? `<div class="rp-note" style="margin-top:3px"><b>验证</b> ${esc(is.verification)}</div>` : ''}
        </div>`).join('')}
      `, { sub: true });
    }

    paint(r);
  }

  /** 把 PAGES 落成 DOM：统一页眉页脚、页码、目录 */
  function paint(r) {
    const doc = document.getElementById('rpDoc');
    const toc = document.getElementById('rpToc');
    const total = PAGES.length;
    doc.innerHTML = PAGES.map((p, i) => {
      if (p.cover) return `<section class="rp-page rp-cover" id="pg-${p.id}">${p.html}</section>`;
      return `<section class="rp-page" id="pg-${p.id}">
        <div class="rp-page__head"><span>${esc(r.title)}</span><span>${esc(r.taskType)} · ${esc(r.reportDate)}</span></div>
        <div class="rp-page__body">${p.html}</div>
        <div class="rp-page__foot"><span>性能分析报告 · 口径见附录 A</span><span class="pno">${i + 1} / ${total}</span></div>
      </section>`;
    }).join('');
    // 仪表盘由 app.js 的 renderPhsGauge() 写 DOM，只能等页面 innerHTML 落地后再回填
    mountGauges(r.phs);
    toc.innerHTML = '<h4>目录</h4>' + PAGES.map((p, i) =>
      `<a class="${p.sub ? 'is-sub' : ''}" href="#pg-${p.id}"><span>${esc(clamp(p.title, 26))}</span><span class="n">${i + 1}</span></a>`).join('');
    document.title = r.title + ' · 报告';
  }

  // ══════════════════════════════════════════════════════════
  //  启动
  // ══════════════════════════════════════════════════════════
  function boot() {
    // app.js 用 `const REPORTS = [...]` 声明，顶层 const 不挂 window，但同为经典脚本
    // 可直接读到该全局词法绑定；用 typeof 兜底避免 app.js 未加载时抛 ReferenceError。
    const list = (typeof REPORTS !== 'undefined' && REPORTS) || [];
    const sel = document.getElementById('rpSelect');
    if (!list.length) {
      document.getElementById('rpDoc').innerHTML = '<div class="rp-empty">未找到报告数据（app.js 的 REPORTS 未加载）。</div>';
      return;
    }
    sel.innerHTML = list.map((r) => `<option value="${esc(r.id)}">${esc(r.title)}</option>`).join('');
    const want = new URLSearchParams(location.search).get('r');
    const pick = list.find((r) => r.id === want) || list[list.length - 1];
    sel.value = pick.id;
    renderReport(pick);
    sel.addEventListener('change', () => {
      const r = list.find((x) => x.id === sel.value);
      if (r) { renderReport(r); scrollTo(0, 0); }
    });
    document.getElementById('rpPrint').addEventListener('click', () => window.print());
  }

  window.PTO_REPORT = { render: renderReport };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
