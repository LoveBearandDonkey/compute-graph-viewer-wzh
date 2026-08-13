/* ══════════════════════════════════════════════════════════════════════════════
   问题二 · 显存 OOM —— 底部 dock「性能」页签的 3 张举证图
   ──────────────────────────────────────────────────────────────────────────────
   对应 定位链-openPangu-2.0-Flash.md 案例四，与 js/training-spotlight.js 里
   CASES["mem-oom"] 的步进联动（选择器 [data-mem-card]）：

     ① mem-timeline   §1 性能表征层   显存占用曲线 + 吞吐（触顶 / OOM 断点）
                                     兼作「容量维度」判据：峰值 = 容量 → 绝对容量不足
     ② composition    §4 峰值构成层   显存峰值按用途堆叠（激活 56.6% = 根因）
     ③ fragment-map   §6 内存快照层   时间×地址碎片分布 + 碎片轴 + 碎片判据读数
                                     点块弹出生命周期与申请堆栈

   案例四另外三层不在本面板里取证，各自归到页面上已有的视图，避免重复造图：
     · §2 瓶颈分类层   → 底部 dock 的「Timeline」页签（耗时构成本来就该在泳道上读）
     · §3 显存表征层   → 容量维度看 ①、碎片维度看 ③ 的判据读数，不单独成卡
     · §5 阶段/层定位  → 整网图「侧视图」的逐层指标曲线「单层激活值显存」
                        （见 training-monitoring-v2-deck.js 的 layer_activation_gb）

   数据全部取自 window.PtoTrainingTwinMemoryCase（training-run-twin.js）——
   静态事实走 .facts，逐 step 读数走 .at(step)。本文件不自带任何业务数字。

   布局：3 张卡在 dock 里一行铺满、各自等宽，不滚动（见 training-monitoring-v2.html
   的 .wzh-mem-grid）。
   ══════════════════════════════════════════════════════════════════════════════ */
window.PtoTrainingMemoryPanel = (function () {
  "use strict";

  var host = null;
  var built = false;
  var canvases = {};      // id → <canvas>
  var popover = null;     // ⑥ 的单块生命周期浮层

  /* ── 主题感知取色 ────────────────────────────────────────────────────────────
     页面可切浅/深色，canvas 里不能写死颜色；每次重绘前从 :root 读一次设计令牌。
     语义色（激活/参数/梯度/优化器）与 hif8-case7.js 的 .h8-tag 同一套，跨案例保持一致。 */
  function palette() {
    var cs = getComputedStyle(document.documentElement);
    function v(name, fallback) {
      var got = cs.getPropertyValue(name);
      return (got && got.trim()) || fallback;
    }
    return {
      ink: v("--foreground", "#111827"),
      dim: v("--foreground-secondary", "#4b5563"),
      mute: v("--foreground-muted", "#6b7280"),
      grid: v("--border-subtle", "rgba(128,128,128,.22)"),
      inset: v("--surface-2", "rgba(128,128,128,.10)"),
      bg: v("--background", "#ffffff"),   // 不透明底色：给压在图上的标注文字垫底

      crit: "#dc2626",
      warn: "#ea580c",
      ok: "#16a34a",
      mem: "#0891b2",
      activation: "#ea580c",
      // 碎片图专用：那里的红色被「最大连续空闲块」的标注占了，碎片块换青色，免得抢眼
      frag: "#0891b2",
      params: "#3b6fe0",
      grads: "#16a34a",
      optimizer: "#8b5cf6",
      workspace: "#94a3b8",
      accent: "#3b6fe0",
    };
  }

  var MONO = "ui-monospace,'SF Mono',Menlo,Consolas,monospace";
  var SANS = "system-ui,-apple-system,'Segoe UI',sans-serif";

  function facts() {
    var api = window.PtoTrainingTwinMemoryCase;
    return api ? api.facts : null;
  }
  function constants() {
    var api = window.PtoTrainingTwinMemoryCase;
    return api ? api.constants : null;
  }

  /* ── canvas 助手 ─────────────────────────────────────────────────────────── */
  // 卡片高度由 grid 决定，canvas 用绝对定位撑满 body，这里按实测 CSS 尺寸做 DPR 缩放
  function ctx2d(cv) {
    if (!cv) return null;
    var r = cv.getBoundingClientRect();
    var w = Math.max(1, Math.round(r.width));
    var h = Math.max(1, Math.round(r.height));
    if (w < 8 || h < 8) return null;           // 页签隐藏时量不到尺寸，跳过本次绘制
    var dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
    var c = cv.getContext("2d");
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, w, h);
    return { c: c, w: w, h: h };
  }
  function roundRect(c, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    c.beginPath();
    if (c.roundRect) { c.roundRect(x, y, w, h, r); return; }
    c.moveTo(x + r, y);
    c.arcTo(x + w, y, x + w, y + h, r);
    c.arcTo(x + w, y + h, x, y + h, r);
    c.arcTo(x, y + h, x, y, r);
    c.arcTo(x, y, x + w, y, r);
    c.closePath();
  }
  function text(c, str, x, y, opt) {
    opt = opt || {};
    c.fillStyle = opt.color || "#000";
    c.font = (opt.weight || "400") + " " + (opt.size || 10) + "px " + (opt.mono ? MONO : SANS);
    c.textAlign = opt.align || "left";
    c.textBaseline = opt.baseline || "alphabetic";
    c.fillText(str, x, y);
  }
  // 竖排文字（逆时针 90°）：碎片图里给又高又窄的常驻块贴名字用
  function vtext(c, str, x, y, opt) {
    c.save();
    c.translate(x, y);
    c.rotate(-Math.PI / 2);
    text(c, str, 0, 0, opt);
    c.restore();
  }
  // 窄卡里文字很容易超出：超宽就逐字裁剪加省略号
  function ellipsis(c, str, maxW) {
    if (c.measureText(str).width <= maxW) return str;
    var s = str;
    while (s.length > 1 && c.measureText(s + "…").width > maxW) s = s.slice(0, -1);
    return s + "…";
  }

  /* ══ ① 显存占用曲线 + 吞吐 ══════════════════════════════════════════════════
     区间取 [climbFrom−4000, recoveryEnd+2000]：左边留一段平稳基线做对照，
     右边留到修复后的新稳态，一屏讲完「平稳→爬升→触顶 OOM→回落」。 */
  function drawTimeline(cv) {
    var g = ctx2d(cv);
    if (!g) return;
    var api = window.PtoTrainingTwinMemoryCase;
    var f = facts(), K = constants();
    if (!api || !f || !K) return;
    var c = g.c, w = g.w, h = g.h, p = palette();

    var x0 = Math.max(0, K.climbFrom - 4000), x1 = K.recoveryEnd + 2000;
    var legendH = 14;
    var pad = { l: 26, r: 24, t: 12 + legendH, b: 16 };
    var pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
    var gbMax = f.capacityGB * 1.06, tpMax = 3600;
    var mapX = function (s) { return pad.l + (s - x0) / (x1 - x0) * pw; };
    var mapY = function (gb) { return pad.t + ph - gb / gbMax * ph; };
    var mapT = function (t) { return pad.t + ph - t / tpMax * ph; };

    /* 图例：显存占用 / 吞吐用各自曲线色，容量线与 OOM 断点共用「危险红」——
       红色只留给真正的异常标记，避免整张图看起来「一片红都是坏的」。 */
    var legendItems = [
      { color: p.mem, label: "显存占用" },
      { color: p.accent, label: "吞吐" },
      { color: p.crit, label: "HBM 容量 / OOM" },
    ];
    var lx = pad.l, ly = 9;
    c.font = "400 9px " + SANS;
    legendItems.forEach(function (it) {
      c.fillStyle = it.color;
      roundRect(c, lx, ly - 6, 6, 6, 1.5); c.fill();
      text(c, it.label, lx + 9, ly, { color: p.mute, size: 9 });
      lx += 9 + c.measureText(it.label).width + 10;
    });

    // 采样：等距 + 强制插入事故步前后各一点，保证断点落在准确位置
    var steps = [];
    for (var i = 0; i <= 160; i++) steps.push(Math.round(x0 + (x1 - x0) * i / 160));
    steps.push(K.incidentStep - 1, K.incidentStep, K.incidentStep + 1);
    steps.sort(function (a, b) { return a - b; });

    // 网格 + y 轴刻度（GB）
    c.strokeStyle = p.grid; c.lineWidth = 1;
    [0, 32, 64].forEach(function (gb) {
      var y = Math.round(mapY(gb)) + 0.5;
      c.beginPath(); c.moveTo(pad.l, y); c.lineTo(pad.l + pw, y); c.stroke();
      text(c, gb, pad.l - 4, y + 3, { color: p.mute, size: 9, align: "right", mono: true });
    });

    // 容量红线
    c.save();
    c.strokeStyle = p.crit; c.lineWidth = 1; c.setLineDash([3, 3]);
    var capY = Math.round(mapY(f.capacityGB)) + 0.5;
    c.beginPath(); c.moveTo(pad.l, capY); c.lineTo(pad.l + pw, capY); c.stroke();
    c.restore();
    text(c, "HBM " + f.capacityGB + "GB", pad.l + pw, capY - 3, { color: p.crit, size: 9, align: "right", mono: true });

    // 吞吐（右轴，弱化为背景参照）
    c.strokeStyle = p.accent; c.globalAlpha = 0.5; c.lineWidth = 1.2;
    c.beginPath();
    var pen = false;
    steps.forEach(function (s) {
      var t = api.at(s).throughput;
      if (!isFinite(t)) { pen = false; return; }
      var X = mapX(s), Y = mapT(t);
      if (!pen) { c.moveTo(X, Y); pen = true; } else c.lineTo(X, Y);
    });
    c.stroke();
    c.globalAlpha = 1;

    // 显存占用（主线 + 面积）
    var pts = [];
    steps.forEach(function (s) { pts.push({ s: s, gb: api.at(s).mem_gb }); });
    // 面积：按连续段分批填充（NaN 处断开 = 训练中断，不能把断点连起来）
    var seg = [];
    function flushArea() {
      if (seg.length < 2) { seg = []; return; }
      c.beginPath();
      c.moveTo(mapX(seg[0].s), mapY(0));
      seg.forEach(function (q) { c.lineTo(mapX(q.s), mapY(q.gb)); });
      c.lineTo(mapX(seg[seg.length - 1].s), mapY(0));
      c.closePath();
      c.fillStyle = p.mem; c.globalAlpha = 0.14; c.fill(); c.globalAlpha = 1;
      c.beginPath();
      seg.forEach(function (q, k) { var X = mapX(q.s), Y = mapY(q.gb); k ? c.lineTo(X, Y) : c.moveTo(X, Y); });
      c.strokeStyle = p.mem; c.lineWidth = 1.6; c.stroke();
      seg = [];
    }
    pts.forEach(function (q) { if (isFinite(q.gb)) seg.push(q); else flushArea(); });
    flushArea();

    // OOM 断点
    var ox = Math.round(mapX(K.incidentStep)) + 0.5;
    c.strokeStyle = p.crit; c.lineWidth = 1.2;
    c.beginPath(); c.moveTo(ox, pad.t); c.lineTo(ox, pad.t + ph); c.stroke();
    c.fillStyle = p.crit;
    c.beginPath(); c.arc(ox, mapY(f.peakGB), 3, 0, Math.PI * 2); c.fill();
    text(c, "OOM", ox + 3, pad.t + 8, { color: p.crit, size: 9, weight: "700" });

    // 时光机当前 step 游标：让这张图和页面时钟对上
    var cur = window.twinGetStep ? window.twinGetStep() : null;
    if (cur != null && cur >= x0 && cur <= x1) {
      var cx = Math.round(mapX(cur)) + 0.5;
      c.save();
      c.strokeStyle = p.ink; c.globalAlpha = 0.45; c.lineWidth = 1; c.setLineDash([2, 2]);
      c.beginPath(); c.moveTo(cx, pad.t); c.lineTo(cx, pad.t + ph); c.stroke();
      c.restore();
    }

    // x 轴刻度
    [x0, K.incidentStep, x1].forEach(function (s, k) {
      text(c, (s / 1000).toFixed(0) + "k", mapX(s), h - 4,
        { color: p.mute, size: 9, mono: true, align: k === 0 ? "left" : k === 2 ? "right" : "center" });
    });
  }

  /* ══ ② 显存峰值构成（面积比例气泡图）════════════════════════════════════════════
     用气泡而不是堆叠柱：堆叠柱里「参数 8.1 / 梯度 8.1 / 优化器 10.8」这三段高度接近，
     谁比谁大要贴着量；气泡按面积编码，大小差异一眼可比。
     半径 ∝ √GB（面积才与数值成正比，直接拿 GB 当半径会把差距夸大成平方）。

     排布不用一字排开，走贪心 packing：最大的钉在中心，其余按由大到小依次找「离中心最近
     且不与已放置气泡相交」的位置落座 —— 于是大的自然居中、小的绕在外围，是那种有机的
     热力气泡观感，而不是整齐一行。搜索起始角按下标取定值，布局随机但每次渲染完全一致。 */
  // aspect = 目标区域的宽高比。搜索路径按它拉成椭圆（碰撞判定仍是真实圆距），
  // 于是气泡簇整体跟着卡片的横向长条形状铺开，而不是挤成一个竖着的圆团、白白浪费两侧宽度。
  function packBubbles(items, gap, aspect) {
    var ax = Math.sqrt(aspect || 1), ay = 1 / Math.sqrt(aspect || 1);
    var placed = [];
    items.forEach(function (it, i) {
      if (i === 0) { placed.push({ it: it, x: 0, y: 0, r: it.r }); return; }
      // 起始角错开黄金角，避免所有小球都从同一个方向排过去、堆成一条边
      var a0 = i * 2.39996;
      var best = null;
      for (var rho = it.r + gap; rho < placed[0].r * 8 && !best; rho += 2) {
        for (var k = 0; k < 72; k++) {
          var th = a0 + k * (Math.PI * 2 / 72);
          var x = Math.cos(th) * rho * ax, y = Math.sin(th) * rho * ay;
          var hitAny = placed.some(function (q) {
            var dx = x - q.x, dy = y - q.y;
            return Math.sqrt(dx * dx + dy * dy) < q.r + it.r + gap;
          });
          if (!hitAny) { best = { x: x, y: y }; break; }
        }
      }
      if (!best) best = { x: (placed[0].r + it.r + gap) * Math.cos(a0) * ax, y: (placed[0].r + it.r + gap) * Math.sin(a0) * ay };
      placed.push({ it: it, x: best.x, y: best.y, r: it.r });
    });
    return placed;
  }

  function drawComposition(cv) {
    var g = ctx2d(cv);
    if (!g) return;
    var f = facts(); if (!f) return;
    var c = g.c, w = g.w, h = g.h, p = palette();
    var colorOf = { activation: p.activation, params: p.params, grads: p.grads, optimizer: p.optimizer, workspace: p.workspace };
    var items = f.composition.slice().sort(function (a, b) { return b.gb - a.gb; });
    var totalGB = f.composition.reduce(function (s, it) { return s + it.gb; }, 0);
    var maxGB = items[0].gb;

    var pad = 8, headH = 13, footH = 13, gap = 3;
    var availW = w - pad * 2, availH = h - headH - footH;

    text(c, "峰值 " + totalGB.toFixed(1) + " GB", pad, 9, { color: p.mute, size: 9.5, mono: true });

    // 先按名义半径 100 packing，量出簇的外接框，再整体缩放到可用区域 —— 这样不用预估布局尺寸
    var nominal = items.map(function (it) { return { id: it.id, gb: it.gb, pct: it.pct, short: it.short, label: it.label, reducible: it.reducible, r: 100 * Math.sqrt(it.gb / maxGB) }; });
    var packed = packBubbles(nominal, gap * 4, availW / Math.max(1, availH));
    var x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    packed.forEach(function (q) {
      x0 = Math.min(x0, q.x - q.r); y0 = Math.min(y0, q.y - q.r);
      x1 = Math.max(x1, q.x + q.r); y1 = Math.max(y1, q.y + q.r);
    });
    var scale = Math.min(availW / (x1 - x0), availH / (y1 - y0));
    var offX = pad + (availW - (x1 - x0) * scale) / 2 - x0 * scale;
    var offY = headH + (availH - (y1 - y0) * scale) / 2 - y0 * scale;

    packed.forEach(function (q) {
      var it = q.it;
      var cx = offX + q.x * scale, cy = offY + q.y * scale, r = q.r * scale;
      c.fillStyle = colorOf[it.id] || p.workspace;
      // 激活值(可缩减项)用满不透明、其余压到 0.8，靠明度区分主次即可，不再另加描边
      c.globalAlpha = it.reducible ? 1 : 0.8;
      c.beginPath(); c.arc(cx, cy, r, 0, Math.PI * 2); c.fill();
      c.globalAlpha = 1;

      /* 名称 + GB 写进圆里。可写宽度取圆内接矩形的宽（≈1.4r），字号随半径缩放；
         小到写不下(r<18)的球把标签移到球外下方，仍然读得出来是哪一项。 */
      var inner = r * 1.4;
      var fs = Math.max(8, Math.min(13, r * 0.30));
      if (r >= 18) {
        c.font = "600 " + fs + "px " + SANS;
        text(c, ellipsis(c, it.short || it.label, inner), cx, cy - 1,
          { color: "#fff", size: fs, weight: "600", align: "center" });
        text(c, it.gb.toFixed(1) + " GB", cx, cy + fs + 1,
          { color: "#fff", size: fs * 0.92, weight: "500", align: "center", mono: true });
        // 再大一点的球连百分比一起写下去
        if (r >= 40) {
          c.globalAlpha = 0.82;
          text(c, (it.pct * 100).toFixed(1) + "%", cx, cy + fs * 2.1 + 2,
            { color: "#fff", size: fs * 0.82, align: "center", mono: true });
          c.globalAlpha = 1;
        }
      } else {
        c.font = "400 8.5px " + SANS;
        var outW = 56;
        var lx = Math.max(pad + outW / 2, Math.min(cx, w - pad - outW / 2));
        text(c, ellipsis(c, (it.short || it.label) + " " + it.gb.toFixed(1) + "GB", outW), lx, cy + r + 9,
          { color: p.mute, size: 8.5, align: "center" });
      }
    });

    // 底部结论
    c.font = "600 9px " + SANS;
    text(c, ellipsis(c, "激活值 " + (items[0].pct * 100).toFixed(1) + "% = 唯一可大幅缩减项 → ~8.5 GB", w - pad * 2),
      pad, h - 2, { color: p.crit, size: 9, weight: "600" });
  }

  /* ══ ⑥ 碎片分布图（时间 × 地址）+ 碎片轴 ═══════════════════════════════════
     块数据在模块内用定长种子生成，形态固定可复现：forward 期间大量不等大小的激活中间张量
     密集分配、backward 后才集中释放 —— 正是碎片的成因。其中一块挂上 MEM_CASE_FACTS
     里的真实样本（L38 q_b_proj），点它能看到申请堆栈。

     生成参数对齐 facts.fragment 的读数，图上量出来的和判据格里写的是同一回事：
       · 空洞尺寸压在 largestFreeBlockGB 以下，只在 BIG_GAP_AT 处留一个正好 = 该读数的大空洞
         → 碎片轴上标红的「最大连续空闲块」就是它；
       · 激活区一路铺到 cap，不留尾巴，空闲总量 ≈ totalFreeGB。 */
  var blocksCache = null;
  function buildBlocks() {
    if (blocksCache) return blocksCache;
    var f = facts(); if (!f) return [];
    var seed = 20260727;
    function rnd() { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }
    var cap = f.capacityGB;
    var largestFree = f.fragment.largestFreeBlockGB;
    var list = [];
    // 常驻区（参数/梯度/优化器）：低地址、整个 step 都在
    [["params", 0, 8.1], ["grads", 8.1, 8.1], ["optimizer", 16.2, 10.8]].forEach(function (r) {
      list.push({ kind: r[0], addr: r[1], size: r[2], t0: 0, t1: 1, resident: true });
    });
    // 激活中间张量：高地址、forward 分配 backward 释放，大小不等 → 释放后留空洞
    var acts = [];
    var addr = 27.0;
    var BIG_GAP_AT = 62;                   // 这一块之后留唯一的大空洞
    for (var i = 0; addr < cap - 0.06; i++) {
      var size = Math.min(0.08 + rnd() * 0.52, cap - addr);
      var t0 = 0.06 + rnd() * 0.38;
      var t1 = Math.min(1, t0 + 0.28 + rnd() * 0.62);
      var b = { kind: "activation", addr: addr, size: size, t0: t0, t1: t1, idx: i };
      list.push(b); acts.push(b);
      // 间隙 = 已释放未合并的空洞：平时都是零头，只有一处放到 largestFreeBlockGB
      addr += size + (i === BIG_GAP_AT ? largestFree : 0.002 + rnd() * 0.021);
    }
    /* 临时 workspace：复用某块激活让出来的地址（同地址、不同时间 = 典型的槽位复用），
       这样它不会掉进空洞里、把标红的最大连续块切开。 */
    for (var j = 0; j < 10; j++) {
      var hostBlk = acts[Math.floor(rnd() * acts.length)];
      var wsSize = Math.min(0.05 + rnd() * 0.1, hostBlk.size);
      list.push({ kind: "workspace", addr: hostBlk.addr, size: wsSize,
                  t0: rnd() * 0.7, t1: 0, transient: true });
    }
    list.forEach(function (b) { if (b.transient) b.t1 = Math.min(1, b.t0 + 0.06); });
    /* 指定其中一块为文档里的真实样本：挑中段里最小的一块，改成样本尺寸后留下的零头
       仍远小于那个大空洞，不会抢走「最大连续」的标注。 */
    var sample = f.fragment.sampleBlock;
    var target = null;
    acts.slice(30, 70).forEach(function (b) { if (!target || b.size < target.size) target = b; });
    if (target) {
      target.sample = sample;
      target.size = sample.sizeMB / 1024;
      target.t0 = sample.allocMs / 9000;
      target.t1 = sample.freeMs / 9000;
    }
    blocksCache = list;
    return list;
  }

  /* 沿地址轴求空闲区段：常驻块 + 激活块的占用取并集，剩下的就是空洞。
     transient 的 workspace 不算（它压在激活块的地址上，不改变空闲版图）。 */
  function freeRuns(blocks, cap) {
    var iv = blocks.filter(function (b) { return !b.transient; })
                   .map(function (b) { return [b.addr, b.addr + b.size]; })
                   .sort(function (a, b) { return a[0] - b[0]; });
    var runs = [], cursor = 0;
    iv.forEach(function (r) {
      if (r[0] > cursor) runs.push([cursor, r[0]]);
      cursor = Math.max(cursor, r[1]);
    });
    if (cursor < cap) runs.push([cursor, cap]);
    return runs;
  }

  /* 坐标约定：横轴 = step 内时间（左→右，符合读时序图的习惯），纵轴 = 显存地址（下低上高）。
     于是常驻块画成贯穿整幅的长横条、激活块画成一小段一小段的短横条，"谁一直占着、谁随用随放"
     不用看图例就读得出来。

     颜色只有一条硬规则：红色专门留给「最大连续空闲块」的标注，别的地方一概不用红 ——
     碎片块本身改用青色（p.frag），免得和标注抢眼。

     右侧「碎片轴」= 同一根地址轴上的占用版图：底子是一个空框（框内空白 = 空闲），
     按用途填参数/梯度/优化器/碎片的图例色，最后把最大的那段连续空闲用红框标出来。 */
  function drawFragmentMap(cv) {
    var g = ctx2d(cv);
    if (!g) return;
    var f = facts(); if (!f) return;
    var c = g.c, w = g.w, h = g.h, p = palette();
    var blocks = buildBlocks();
    var colorOf = { params: p.params, grads: p.grads, optimizer: p.optimizer, activation: p.frag, workspace: p.workspace };

    var pad = 8;
    var gutL = 30;                    // 左栏：区名（竖排）+ 地址刻度
    var stripW = 9, stripGap = 5;     // 右侧碎片轴
    var titleH = 11, legendH = 12, axisH = 11, capH = 10, judgeH = 30;
    var top = titleH + legendH + 4;
    var bot = Math.max(top + 40, h - judgeH - capH - axisH - 4);
    var plotX = pad + gutL, plotW = Math.max(20, w - pad * 2 - gutL - stripW - stripGap);
    var stripX = w - pad - stripW;
    var cap = f.capacityGB;
    var mapT = function (t) { return plotX + t * plotW; };               // 时间 → x
    var mapA = function (gb) { return bot - gb / cap * (bot - top); };   // 地址 → y（低地址在下）
    // 常驻区/激活区的地址分界：常驻块的最高地址，不写死数字
    var split = 0;
    blocks.forEach(function (b) { if (b.resident) split = Math.max(split, b.addr + b.size); });

    // ── 轴说明：一句话把两根轴交代掉
    c.font = "400 9.5px " + SANS;
    text(c, ellipsis(c, "横轴 = step 内时间 →　　纵轴 = 显存地址 0–" + cap + " GB（下低上高）", w - pad * 2),
      pad, 9, { color: p.mute, size: 9.5 });

    /* 图例只剩两项：常驻的参数/梯度/优化器已直接把名字写在长横条上，不再重复占图例位。
       右侧留「碎片轴」的栏名，正对下面那根竖条。 */
    var shortOf = {};
    f.composition.forEach(function (it) { shortOf[it.id] = it.short || it.label; });
    var ly = titleH + 9;
    c.font = "400 9px " + SANS;
    var lx = pad;
    [["activation", "碎片块(激活中间张量)"], ["workspace", shortOf.workspace || "workspace"]].forEach(function (it) {
      var name = it[1];
      c.fillStyle = colorOf[it[0]] || p.workspace;
      roundRect(c, lx, ly - 6, 6, 6, 1.5); c.fill();
      text(c, name, lx + 9, ly, { color: p.mute, size: 9 });
      lx += 9 + c.measureText(name).width + 8;
    });
    c.font = "400 8.5px " + SANS;
    if (stripX - 30 > lx + 6) text(c, "碎片轴", stripX + stripW, ly, { color: p.mute, size: 8.5, align: "right" });

    // ── 左栏：地址刻度（0 / 分界 / 容量）+ 竖排区名，说明下段常驻、上段激活
    c.strokeStyle = p.grid; c.lineWidth = 1;
    c.beginPath(); c.moveTo(plotX - 4.5, top); c.lineTo(plotX - 4.5, bot); c.stroke();
    [[0, "0"], [split, String(Math.round(split))], [cap, cap + " GB"]].forEach(function (tk) {
      var y = mapA(tk[0]);
      c.beginPath(); c.moveTo(plotX - 4.5, y + 0.5); c.lineTo(plotX - 1.5, y + 0.5); c.stroke();
      text(c, tk[1], plotX - 7, y + (tk[0] === 0 ? 0 : tk[0] === cap ? 7 : 3),
        { color: p.mute, size: 8, align: "right" });
    });
    c.font = "400 8.5px " + SANS;
    [[0, split, "常驻区"], [split, cap, "激活区"]].forEach(function (seg) {
      var y0 = mapA(seg[0]), y1 = mapA(seg[1]);
      if (y0 - y1 < 26) return;
      vtext(c, seg[2], pad + 5, (y0 + y1) / 2,
        { color: p.mute, size: 8.5, align: "center", baseline: "middle" });
    });
    // 常驻区/激活区分界：横虚线贯穿块图直插碎片轴，点明两者共用同一根地址轴
    c.save();
    c.setLineDash([2, 2]); c.strokeStyle = p.mute; c.globalAlpha = 0.5;
    c.beginPath(); c.moveTo(plotX - 4, mapA(split) + 0.5); c.lineTo(w - pad, mapA(split) + 0.5); c.stroke();
    c.restore();

    // ── 时间轴：底边一条线 + 前向/反向/结束三个刻度
    var BWD = 0.45;        // buildBlocks() 里激活块的申请全部落在 forward 段（t0 ≤ 0.44）
    c.strokeStyle = p.grid; c.lineWidth = 1;
    c.beginPath(); c.moveTo(plotX, bot + 0.5); c.lineTo(plotX + plotW, bot + 0.5); c.stroke();
    [[0, "前向", "left"], [BWD, "反向", "center"], [1, "结束", "right"]].forEach(function (tk) {
      var x = mapT(tk[0]);
      c.beginPath(); c.moveTo(x + 0.5, bot + 0.5); c.lineTo(x + 0.5, bot + 3.5); c.stroke();
      text(c, tk[1], x, bot + axisH, { color: p.mute, size: 8, align: tk[2] });
    });
    // 反向起点：竖虚线一划，就能看出「申请都在线左边、释放都在线右边」
    c.save();
    c.setLineDash([2, 3]); c.strokeStyle = p.grid;
    c.beginPath(); c.moveTo(mapT(BWD) + 0.5, top); c.lineTo(mapT(BWD) + 0.5, bot); c.stroke();
    c.restore();

    // ── 块：常驻的贯穿整幅（整个 step 都在），激活/workspace 只占一段时间
    var fragGeom = [];
    blocks.forEach(function (b) {
      var x = mapT(b.t0), bw = Math.max(1.2, (b.t1 - b.t0) * plotW);
      var y = mapA(b.addr + b.size);
      var bh = Math.max(1.5, b.size / cap * (bot - top));
      c.fillStyle = colorOf[b.kind] || p.workspace;
      c.globalAlpha = b.resident ? 0.7 : 0.9;
      c.fillRect(x, y, bw, bh);
      c.globalAlpha = 1;
      if (b.sample) {                       // 带真实申请堆栈的样本块：描边 + 贴名字，别让人猜这框是什么
        c.strokeStyle = p.ink; c.lineWidth = 1.5;
        c.strokeRect(x - 1, y - 1, bw + 2, bh + 2);
        c.font = "600 8px " + SANS;
        text(c, ellipsis(c, "样本块：点开看申请堆栈", bw), x + 1, y - 4,
          { color: p.ink, size: 8, weight: "600" });
      }
      fragGeom.push({ b: b, x: x, y: y, w: bw, h: bh });
    });
    /* 命中框存在 canvas 自己身上：同一页上碎片图有两张（dock「性能」页签 + 定位链长文），
       两张宽高不同，早先存在模块级变量里会被后画的那张覆盖，先画的那张就点不动了。 */
    cv.__fragGeom = fragGeom;
    // 常驻三条直接把名字写在条上（横过来以后条子够高，横排写得下）
    blocks.forEach(function (b) {
      if (!b.resident) return;
      var y = mapA(b.addr + b.size), bh = b.size / cap * (bot - top);
      if (bh < 9) return;
      c.font = "600 8.5px " + SANS;
      text(c, ellipsis(c, shortOf[b.kind] || b.kind, plotW - 12), plotX + 6, y + bh / 2 + 3,
        { color: "#fff", size: 8.5, weight: "600" });
    });

    /* ── 碎片轴：同一根地址轴上的占用版图。空框垫底 = 空闲，按用途填图例色。
       激活区被切成上百块 0.1~0.6 GB 的小块，填色后连成一片 —— 空白少得可怜正是碎片率 83% 的样子。 */
    c.strokeStyle = p.grid; c.lineWidth = 1;
    c.strokeRect(stripX + 0.5, top + 0.5, stripW - 1, bot - top - 1);
    blocks.forEach(function (b) {
      if (b.transient) return;            // workspace 压在激活块地址上，不另占版图
      var y = mapA(b.addr + b.size), bh = Math.max(0.6, b.size / cap * (bot - top));
      c.fillStyle = colorOf[b.kind] || p.workspace;
      c.globalAlpha = b.resident ? 0.75 : 0.9;
      c.fillRect(stripX + 1, y, stripW - 2, bh);
      c.globalAlpha = 1;
    });

    /* 最大连续空闲块的标注：它本身没有对错 —— 0.3 GB 只是个事实，真正出事的是
       "下一笔 0.5 GB 的临时 buffer 请求接不下它"。所以这块空闲用中性描边标位置，
       红色留给那笔装不进去的请求（红条画在空闲块正上方、按同一地址比例延伸出去，
       露在描边外的那截就是差的量）。 */
    var runs = freeRuns(blocks, cap);
    var best = null;
    runs.forEach(function (r) { if (!best || r[1] - r[0] > best[1] - best[0]) best = r; });
    if (best) {
      var freeGB = best[1] - best[0];
      var reqGB = f.fragment.maxRequestGB || 0;
      var gb2px = (bot - top) / cap;
      var by1 = mapA(best[1]), by0 = mapA(best[0]);
      var bhh = Math.max(3, by0 - by1), byc = (by0 + by1) / 2;
      var fy = byc + bhh / 2;                                  // 空闲块下沿：请求条也从这里起算
      // 块图里对应的地址带：这一条地址全程空着，指得出红框标的是哪一段
      c.fillStyle = p.mute; c.globalAlpha = 0.10;
      c.fillRect(plotX, byc - bhh / 2, plotW, bhh);
      c.globalAlpha = 1;
      // 空闲块本身：中性描边
      c.strokeStyle = p.dim; c.lineWidth = 1;
      c.strokeRect(stripX - 2.5, byc - bhh / 2 - 0.5, stripW + 5, bhh + 1);
      c.beginPath(); c.moveTo(plotX + plotW + 1, byc); c.lineTo(stripX - 3, byc); c.stroke();
      if (reqGB > 0) {
        // 装不下的那笔请求：等比放大到能看清，超出空闲块的部分就是"差的量"
        var scale = Math.max(1, 3 / Math.max(0.001, freeGB * gb2px));
        var reqH = Math.max(4, reqGB * gb2px * scale);
        var rx = stripX - 2.5 - 6;
        c.fillStyle = p.crit; c.globalAlpha = 0.85;
        c.fillRect(rx - 3, fy - reqH, 4, reqH);
        c.globalAlpha = 1;
        c.font = "600 8px " + SANS;
        var lab = "最大申请 " + reqGB + " GB > 最大连续 " + freeGB.toFixed(1) + " GB";
        var lw = c.measureText(lab).width;
        var lxr = rx - 6 - lw, lyr = fy - reqH - 3;
        c.fillStyle = p.bg; c.globalAlpha = 0.9;
        roundRect(c, lxr - 3, lyr - 9, lw + 6, 12, 2); c.fill();
        c.globalAlpha = 1;
        text(c, lab, lxr, lyr, { color: p.crit, size: 8, weight: "600" });
      }
    }

    // 一行说明：碎片轴的读法 + 红色的含义 + 交互提示，不再另开一套色标
    c.font = "400 8.5px " + SANS;
    text(c, ellipsis(c, "碎片轴：填色=已占用，空白=空闲，灰框=最大连续空闲块，红=接不下的那笔申请 · 点任意块看生命周期与堆栈", w - pad * 2),
      pad, bot + axisH + capH - 3, { color: p.mute, size: 8.5 });

    /* 碎片维度的判据读数（原本单独一张「双因子判定」卡，现并到这里 —— 判据紧挨着它的证据图）：
       空闲够但最大连续块接不下请求 size，就是分配碎片。容量维度的判据在 ① 显存曲线上（峰值=容量）。 */
    var fr = f.fragment;
    var jy = bot + axisH + capH + 2;
    /* 前三格都是中性事实（空闲总量 / 最大连续 / 最大申请），单看谁都不算错；
       标红的是「最大申请 > 最大连续」这个关系本身，以及它推出来的碎片率。 */
    var cells = [
      { lab: "空闲总量", val: fr.totalFreeGB + " GB", hot: false },
      { lab: "最大连续", val: fr.largestFreeBlockGB + " GB", hot: false },
      { lab: "最大申请", val: fr.maxRequestGB + " GB", hot: fr.maxRequestGB > fr.largestFreeBlockGB },
      { lab: "碎片率", val: (fr.ratio * 100).toFixed(0) + "%", hot: true },
    ];
    var cw = (w - pad * 2 - (cells.length - 1) * 4) / cells.length;
    cells.forEach(function (cell, i) {
      var cx = pad + i * (cw + 4);
      c.fillStyle = p.inset;
      roundRect(c, cx, jy, cw, judgeH - 4, 4); c.fill();
      text(c, cell.lab, cx + cw / 2, jy + 11, { color: p.mute, size: 8.5, align: "center" });
      text(c, cell.val, cx + cw / 2, jy + 23,
        { color: cell.hot ? p.crit : p.dim, size: 11, weight: "600", align: "center", mono: true });
    });
  }

  /* 单块生命周期浮层：点碎片图里的任意块弹出，样本块带真实申请堆栈。
     按需创建而不是跟着 dock 面板一起建 —— 定位链长文里的碎片图（training-monitoring.html 上
     根本没有 dock「性能」页签）也要能弹出它。 */
  function ensurePopover() {
    if (popover) return popover;
    popover = document.createElement("div");
    popover.className = "wzh-mem-pop";
    popover.hidden = true;
    document.body.appendChild(popover);
    return popover;
  }

  function showBlockPopover(hit, clientX, clientY) {
    ensurePopover();
    var f = facts(); if (!f) return;
    var b = hit.b;
    var s = b.sample;
    var name = s ? s.name : (b.kind === "activation" ? "激活中间张量 #" + (b.idx != null ? b.idx : "?") :
                             b.kind === "params" ? "模型参数 (FP8)" :
                             b.kind === "grads" ? "梯度 (FP8)" :
                             b.kind === "optimizer" ? "优化器状态 (BF16 m+v)" : "临时 workspace buffer");
    var sizeTxt = s ? s.shape + " ≈ " + s.sizeMB + " MB" : (b.size * 1024).toFixed(0) + " MB";
    var allocMs = s ? s.allocMs : Math.round(b.t0 * 9000);
    var freeMs = s ? s.freeMs : Math.round(b.t1 * 9000);
    var holdMs = s ? s.holdMs : Math.max(0, freeMs - allocMs);
    var stack = s ? s.stack : null;

    popover.innerHTML =
      '<div class="wzh-mem-pop__title"></div>' +
      '<div class="wzh-mem-pop__rows">' +
        '<div><span>地址</span><b>' + b.addr.toFixed(2) + ' GB</b></div>' +
        '<div><span>大小</span><b>' + sizeTxt + '</b></div>' +
        '<div><span>申请</span><b>forward +' + allocMs + ' ms</b></div>' +
        '<div><span>释放</span><b>' + (b.resident ? 'step 结束' : 'backward +' + freeMs + ' ms') + '</b></div>' +
        '<div><span>持有</span><b class="is-hot">' + (holdMs / 1000).toFixed(2) + ' s</b></div>' +
      '</div>' +
      (stack ? '<div class="wzh-mem-pop__stack"><span>申请堆栈</span>' +
        stack.map(function (fn, i) { return '<code>' + (i ? '↳ ' : '') + fn + '</code>'; }).join("") + '</div>' : '');
    popover.querySelector(".wzh-mem-pop__title").textContent = name;

    popover.hidden = false;
    // 贴着鼠标摆，越界时翻到另一侧
    var r = popover.getBoundingClientRect();
    var x = Math.min(clientX + 12, window.innerWidth - r.width - 8);
    var y = clientY - r.height - 12;
    if (y < 8) y = clientY + 14;
    popover.style.left = Math.max(8, x) + "px";
    popover.style.top = y + "px";
  }
  function hideBlockPopover() { if (popover) popover.hidden = true; }

  // 碎片图的点击命中：绑在传入的 canvas 上，dock 里那张与定位链长文里那张各绑一次
  function bindFragmentInteraction(cv) {
    if (!cv || cv.dataset.bound) return;
    cv.dataset.bound = "1";
    cv.style.cursor = "crosshair";
    cv.addEventListener("click", function (ev) {
      var fragGeom = cv.__fragGeom;
      if (!fragGeom) return;
      var r = cv.getBoundingClientRect();
      var px = ev.clientX - r.left, py = ev.clientY - r.top;
      // 命中面积小，优先取样本块；其次取最后一个命中的（画在上层的）
      // 激活块本身只有 1~3px 高，命中框统一外扩；带堆栈的样本块再放宽一点，保证点得中
      var hit = null;
      fragGeom.forEach(function (q) {
        var m = q.b.sample ? 6 : 2.5;
        var inside = px >= q.x - m && px <= q.x + q.w + m && py >= q.y - m && py <= q.y + q.h + m;
        if (!inside) return;
        if (!hit || q.b.sample) hit = q;
      });
      if (hit) { ev.stopPropagation(); showBlockPopover(hit, ev.clientX, ev.clientY); }
      else hideBlockPopover();
    });
    document.addEventListener("click", function (ev) {
      if (popover && !popover.hidden && ev.target !== cv && !popover.contains(ev.target)) hideBlockPopover();
    });
  }

  /* ── 装配 ───────────────────────────────────────────────────────────────── */
  var CARDS = [
    { id: "mem-timeline", title: "显存占用 & 吞吐" },
    { id: "composition", title: "峰值构成" },
    { id: "fragment-map", title: "碎片分布" },
  ];

  function build() {
    if (built || !host) return;
    var grid = document.createElement("div");
    grid.className = "wzh-mem-grid";
    CARDS.forEach(function (cd) {
      var card = document.createElement("section");
      card.className = "wzh-mem-card";
      // 聚光灯定位链按 [data-mem-card] 选中并照亮对应卡（见 js/training-spotlight.js）
      card.dataset.memCard = cd.id;
      var head = document.createElement("div");
      head.className = "wzh-mem-card__head";
      head.innerHTML = '<span class="wzh-mem-card__title"></span>';
      head.querySelector(".wzh-mem-card__title").textContent = cd.title;
      var body = document.createElement("div");
      body.className = "wzh-mem-card__body";
      var cv = document.createElement("canvas");
      body.appendChild(cv);
      card.appendChild(head);
      card.appendChild(body);
      grid.appendChild(card);
      canvases[cd.id] = cv;
    });
    host.innerHTML = "";
    host.appendChild(grid);

    built = true;
    bindFragmentInteraction(canvases["fragment-map"]);
  }

  /* 卡片 DOM 无条件建好(哪怕页签还隐藏着)——聚光灯按 [data-mem-card] 找目标，
     元素必须先存在；canvas 绘制则由 ctx2d() 在量不到尺寸时自行跳过。 */
  var DRAWERS = { "mem-timeline": drawTimeline, "composition": drawComposition, "fragment-map": drawFragmentMap };

  /* 把某张图画到任意一块 canvas 上。dock 面板用它画自己那三张；
     「问题二」定位链长文(js/training-memory-case4.js)也用它把同样三张嵌进正文，
     两处共用同一份绘制代码与同一份数据，不会画出两套口径。 */
  function drawInto(id, cv) {
    var fn = DRAWERS[id];
    if (!fn || !cv) return;
    if (id === "fragment-map") bindFragmentInteraction(cv);
    try { fn(cv); } catch (e) {}
  }

  function render() {
    if (!host) return;
    build();
    CARDS.forEach(function (cd) { drawInto(cd.id, canvases[cd.id]); });
  }

  var rafPending = 0;
  var extraRedraw = [];   // 额外注册的重绘回调(定位链长文里的那几张图靠它跟着 resize/主题切换刷新)
  function renderSoon() {
    if (rafPending) return;
    rafPending = window.requestAnimationFrame(function () {
      rafPending = 0;
      render();
      extraRedraw.forEach(function (fn) { try { fn(); } catch (e) {} });
    });
  }
  function onRedraw(fn) { if (typeof fn === "function" && extraRedraw.indexOf(fn) < 0) extraRedraw.push(fn); }

  function mount() {
    host = document.getElementById("dockPanelPerf");
    window.addEventListener("resize", renderSoon);
    // 主题切换（浅/深色）后配色要跟着变：监听 :root 的 data-theme
    if (window.MutationObserver) {
      new MutationObserver(renderSoon).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    }
    render();   // host 不存在(如 training-monitoring.html 没有这个 dock 页签)时内部直接返回
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount);
  else mount();

  return { mount: mount, render: render, renderSoon: renderSoon, drawInto: drawInto, onRedraw: onRedraw };
})();
