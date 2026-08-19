(function (global) {
  "use strict";

  function esc(v) {
    return String(v == null ? "" : v)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/"/g, "&quot;");
  }

  function kind(v) {
    var k = String(v || "").toLowerCase();
    if (k === "temp") return "workspace";
    if (k === "resident") return "parameters";
    return k || "workspace";
  }

  function label(k) {
    return ({
      activation: "激活",
      parameters: "参数",
      gradients: "梯度",
      optimizer: "优化器",
      // 不叫"临时空间"——那是个自造词，看的人无从对应到任何 API 或日志字段。
      // workspace 是算子/通信库自己用的那块临时缓冲的通行叫法，本页「峰值构成」
      // 图里也是这么写的，两处对得上才能互相印证。
      workspace: "临时 workspace"
    })[k] || k;
  }

  /* 图例悬浮解释：类别名再准确也压不进领域知识，workspace 这类词对第一次看的人
     就是个黑盒。写在 title 里，想知道的人一悬停就有，不占版面。 */
  var LEGEND_TIP = {
    activation: "前向算出、必须留到 backward 用完才能释放的中间张量。开重计算就是拿算力换它。",
    parameters: "模型权重分片，整个 step 常驻，不随 micro-batch 变化。",
    gradients: "反向累积出的梯度，与权重同量级。",
    optimizer: "Adam 的 fp32 master weight + 一阶/二阶动量，约 12 B/参数。",
    workspace: "算子或通信库执行期间自用的临时缓冲：如 MoE dispatch/combine 的重排缓存、集合通信 buffer。kernel 调用前申请、结束即释放，不跨算子存活 —— 所以它小、但申请释放极频繁，正是碎片的主要来源。",
  };

  /* ══ 碎片区立体容器（侧视图）════════════════════════════════════════════
     旧版这里是两条平面色条：上面一条把空闲/占用按宽度铺开，下面一条画待分配的
     0.5 GB。数字都在，但"0.5 GB 塞不进 0.3 GB 的空档"要靠读者自己去比两条不同
     的条的长短 —— 读起来是两个进度条，不是一件放不下的事。

     现在把同一份数据画成**一个有厚度的容器**：占用是坐在容器里的实心块，空闲
     就是容器里空着的那几段，待分配的 0.5 GB 是一整块吊在容器口上方、用同一把
     尺子量出来的实体。它比最大那段空档宽出一截，垂直落下去会正好压在中间那块
     占用上。于是"放不下"成了一个几何事实，而不是一行文字。

     ── 投影：为什么不是 config-relation-capacity 那套真等距 ──
     那幅图（单卡容量）的"值"是高度，四段自下而上摞成一根柱子，长轴是竖的，真
     等距（x/z 各偏 30°）正合适。这幅图的"值"是**地址长度**，长轴是横的：34 单位
     长的容器套真等距会在屏幕上同时下沉 17 单位，画出来又高又斜；更要命的是 x 轴
     被 cos30° 压缩、还和 z 轴纠缠在一起，两段长度没法直接目测比较 —— 而"比长度"
     恰恰是这幅图存在的全部意义。所以改用 cabinet 斜投影：x 轴保持水平且不缩放
     （地址轴就是一把真尺子），y 轴竖直，只有深度轴朝右上偏。这也正是"立体容器
     侧视图"该有的样子。

     可复用的部分照搬 training-run-twin-standalone/js/config-relation-capacity.js
     的 buildBox：三面明暗（顶最亮 / 正面居中 / 右侧最暗）、虚线线框 = 容量、实心
     = 内容、越界用 --danger 单独画出来。几何在本文件内独立实现，未跨目录依赖。 */
  var NS = "http://www.w3.org/2000/svg";
  /* 深度轴：背面相对正面在屏幕上的偏移（x 往右、y 往上）。z 只取 0（正面）/ 1（背面）。 */
  var DEPTH = { x: 1.9, y: -1.25 };
  /* len = 整个地址窗口铺开后的长度，h = 容器净高，floor = 底板厚度。全部 user unit，
     viewBox 由内容包围盒反推，所以面板多宽都不用重算。 */
  var CAGE = { len: 34, h: 4.6, floor: 0.42 };
  /* 待分配块：吊在容器口上方 lift 处，本身 h 高。lift 给得足够大 —— 它得读成
     "还没放进去、悬在上面的一块"，贴着容器口会读成"已经摞在上面了"。 */
  var REQ = { lift: 2.3, h: 2.6 };

  function svgEl(tag, attrs) {
    var node = document.createElementNS(NS, tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (attrs[k] !== null && attrs[k] !== undefined) node.setAttribute(k, attrs[k]);
    });
    return node;
  }

  /* host 决定从哪个节点取值：--hbm-* 定义在 .pto-hbm-snapshot 上而不是 :root，
     传根节点才拿得到，否则是空串。 */
  function sceneVar(name, fallback, host) {
    var node = host || document.documentElement;
    var raw = getComputedStyle(node).getPropertyValue(name).trim();
    return raw || fallback;
  }

  function parseColor(input) {
    var str = String(input || "").trim();
    if (str.charAt(0) === "#") {
      var h = str.slice(1);
      var full = h.length === 3 ? h[0] + h[0] + h[1] + h[1] + h[2] + h[2] : h.slice(0, 6);
      var n = parseInt(full, 16);
      if (isNaN(n)) return null;
      return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    }
    var m = str.match(/rgba?\(([^)]+)\)/i);
    if (!m) return null;
    var parts = m[1].split(/[\s,/]+/).filter(Boolean).map(Number);
    if (parts.length < 3 || parts.some(isNaN)) return null;
    return [parts[0], parts[1], parts[2]];
  }

  function shade(rgb, amt) {
    if (!rgb) return "currentColor";
    var out = rgb.map(function (v) {
      var next = amt >= 0 ? v + (255 - v) * amt : v * (1 + amt);
      return Math.max(0, Math.min(255, Math.round(next)));
    });
    return "rgb(" + out[0] + ", " + out[1] + ", " + out[2] + ")";
  }

  function resolveColor(input, host, depth) {
    if (Array.isArray(input)) return input;
    var str = String(input || "").trim();
    var m = str.match(/^var\(\s*(--[\w-]+)\s*(?:,\s*([\s\S]+))?\)$/);
    if (!m) return parseColor(str);
    var raw = sceneVar(m[1], "", host);
    var level = depth || 0;
    if (raw && level < 3) {
      var hit = resolveColor(raw, host, level + 1);
      if (hit) return hit;
    }
    return m[2] && level < 3 ? resolveColor(m[2].trim(), host, level + 1) : null;
  }

  /* spec:
       segments      [{ kind:"free"|"blocker", sizeGB, id, name, allocKind }]，按地址顺序
       total         窗口总长（GB），= segments 之和
       requestedGB   待分配大小
       largestFreeGB 最大连续空闲块
       host          解析 var(--token) 的上下文节点（须已在文档里）
       format        (gb) => "0.30 GB" */
  function buildFragmentScene(spec) {
    var segs = spec.segments || [];
    var total = spec.total > 0 ? spec.total : 1;
    var host = spec.host;
    var unit = CAGE.len / total;
    var fmt = spec.format || function (v) { return v.toFixed(2) + " GB"; };
    var requestedGB = Math.max(0, spec.requestedGB || 0);
    var largestGB = Math.max(0, spec.largestFreeGB || 0);
    var shortageGB = Math.max(0, requestedGB - largestGB);

    /* 待分配块左对齐到**最大空档的起点**，而不是居中盖在它上面：分配器找不到连续
       地址时，最后一次尝试就是从最大的那段空档开头量过去。左对齐之后越界的那一截
       全部落在右邻，图里就能直接指认"是谁挡住的" —— 比两边各露一点更接近真实的
       失败原因。 */
    var largestIndex = -1;
    var gapStart = 0;
    (function () {
      var walk = 0;
      var best = -1;
      segs.forEach(function (seg, i) {
        if (seg.kind === "free" && seg.sizeGB > best) { best = seg.sizeGB; largestIndex = i; gapStart = walk; }
        walk += Number(seg.sizeGB || 0);
      });
    }());
    gapStart *= unit;
    var gapLen = largestGB * unit;
    var gapEnd = gapStart + gapLen;
    var reqEnd = gapStart + requestedGB * unit;

    var bounds = { x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity };
    function grow(px, py) {
      if (px < bounds.x0) bounds.x0 = px;
      if (px > bounds.x1) bounds.x1 = px;
      if (py < bounds.y0) bounds.y0 = py;
      if (py > bounds.y1) bounds.y1 = py;
    }
    /* cabinet 斜投影：x 原样水平、y 竖直、深度轴按 DEPTH 偏。z 只取 0/1。 */
    function P(x, y, z) {
      var px = x + z * DEPTH.x;
      var py = -y + z * DEPTH.y;
      grow(px, py);
      return [px, py];
    }
    function pts(list) {
      return list.map(function (p) { return p[0].toFixed(3) + "," + p[1].toFixed(3); }).join(" ");
    }

    var svg = svgEl("svg", { role: "img", "aria-label": spec.ariaLabel || "碎片地址区立体容器示意" });
    var g = svgEl("g", {});
    svg.appendChild(g);

    /* 一个实心块：只画看得见的三个面，按受光度分三档明暗。按 x 从左往右画 ——
       右邻块的正面正好盖住左块的右侧面，相邻两块的接缝自然消失。 */
    function box(x0, x1, y0, y1, rgb, opts) {
      var o = opts || {};
      var grp = svgEl("g", {
        class: o.cls || null, "data-id": o.id || null, "data-kind": o.kind || null,
      });
      [
        [[P(x0, y1, 0), P(x1, y1, 0), P(x1, y1, 1), P(x0, y1, 1)], 0.30],   // 顶：最亮
        [[P(x0, y0, 0), P(x1, y0, 0), P(x1, y1, 0), P(x0, y1, 0)], -0.06],  // 正面
        [[P(x1, y0, 0), P(x1, y1, 0), P(x1, y1, 1), P(x1, y0, 1)], -0.30],  // 右：最暗
      ].forEach(function (face) {
        grp.appendChild(svgEl("polygon", {
          points: pts(face[0]), fill: shade(rgb, face[1]), stroke: shade(rgb, -0.5),
          "stroke-width": 0.05, "stroke-linejoin": "round",
          // 半透明块（越界投影）要能看见底下压着谁，所以 fill 与 stroke 分开控制
          "fill-opacity": o.fillOpacity || null,
          "stroke-dasharray": o.dashed ? "0.22 0.16" : null,
        }));
      });
      if (o.title) { var t = svgEl("title", {}); t.textContent = o.title; grp.appendChild(t); }
      g.appendChild(grp);
      return grp;
    }

    /* 只描边不填充：画出一个盒子看得见的 9 条棱（正面 4 条 + 顶面后 3 条 + 右面后
       2 条）。用来把"待分配块里超出的那一截"框出来 —— 它和左边那截是同一块实体、
       不该换填充色，但边界得说清楚在哪。 */
    function outlineBox(x0, x1, y0, y1, color, o) {
      o = o || {};
      var grp = svgEl("g", { class: o.cls || null });
      [
        [[x0, y0, 0], [x1, y0, 0]], [[x1, y0, 0], [x1, y1, 0]],
        [[x1, y1, 0], [x0, y1, 0]], [[x0, y1, 0], [x0, y0, 0]],
        [[x0, y1, 0], [x0, y1, 1]], [[x0, y1, 1], [x1, y1, 1]], [[x1, y1, 1], [x1, y1, 0]],
        [[x1, y0, 0], [x1, y0, 1]], [[x1, y0, 1], [x1, y1, 1]],
      ].forEach(function (e) {
        var a = P(e[0][0], e[0][1], e[0][2]);
        var b = P(e[1][0], e[1][1], e[1][2]);
        grp.appendChild(svgEl("line", {
          x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: color,
          "stroke-width": o.width || 0.08, "stroke-dasharray": o.dash || "0.26 0.18",
          "stroke-linecap": "round",
        }));
      });
      g.appendChild(grp);
      return grp;
    }

    /* 文字不经过 P()，包围盒得手动放出去，否则 viewBox 会把它裁掉。中文字宽 ≈ 字号、
       拉丁约 0.6，混排取 0.62 折中（宁可多留一点白，也别让标注被切一半）。
       一律 pointer-events:none —— 压在块面上的那条标注不能挡住块自己的 <title>。 */
    function text(x, y, str, o) {
      o = o || {};
      var size = o.size || 0.6;
      var node = svgEl("text", {
        x: x.toFixed(3), y: y.toFixed(3),
        "text-anchor": o.anchor || "middle", "font-size": size,
        "font-family": "ui-monospace, Menlo, Consolas, monospace",
        "font-weight": o.weight || 500, fill: o.fill || "currentColor",
        "pointer-events": "none",
      });
      node.textContent = str;
      var run = String(str).length * size * 0.62;
      var left = o.anchor === "end" ? x - run : (o.anchor === "start" ? x : x - run / 2);
      grow(left, y - size);
      grow(left + run, y + size * 0.35);
      g.appendChild(node);
      return node;
    }

    var floorRgb = resolveColor(sceneVar("--surface-3", "#2a2f36", host), host);
    var dangerRgb = resolveColor(sceneVar("--danger", "#E5484D", host), host);
    var danger = shade(dangerRgb, 0);
    var wire = sceneVar("--border-strong", "rgba(255,255,255,0.28)", host);
    var muted = sceneVar("--foreground-muted", "#8b93a1", host);
    var fg = sceneVar("--foreground", "#e8ecf1", host);

    /* ── 底板：容器得有个实体的底，空档才读成"容器里空着的一段"而不是"什么都没有" ── */
    box(0, CAGE.len, -CAGE.floor, 0, floorRgb, { cls: "pto-hbm-snapshot__iso-floor" });

    /* ── 最大连续空闲：底板上高亮出那一段。它是唯一有机会容下请求的槽位，
         请求块正是对着它落下来的。 ── */
    if (gapLen > 0.02) {
      g.appendChild(svgEl("polygon", {
        points: pts([P(gapStart, 0, 0), P(gapEnd, 0, 0), P(gapEnd, 0, 1), P(gapStart, 0, 1)]),
        fill: danger, "fill-opacity": 0.18, stroke: danger,
        "stroke-width": 0.06, "stroke-dasharray": "0.24 0.18",
      }));
    }

    /* ── 占用块：坐在容器里、顶到容器口。它们就是把空闲切碎的那些墙 ── */
    var walk = 0;
    segs.forEach(function (seg) {
      var x0 = walk * unit;
      walk += Number(seg.sizeGB || 0);
      var x1 = walk * unit;
      if (seg.kind !== "blocker") return;
      box(x0, x1, 0, CAGE.h, resolveColor("var(--hbm-" + seg.allocKind + ", #738095)", host), {
        cls: "pto-hbm-snapshot__iso-block", id: seg.id, kind: seg.allocKind,
        title: seg.name + " · " + fmt(seg.sizeGB) + " · 点击查看详情",
      });
    });

    /* ── 越界投影：把待分配块超出最大空档的那一截，按原尺寸投影回容器里。
         淡红半透明，压在占用块上 —— 重叠本身就是结论："这块地址已经有人了"。
         所以不用再去描红被压住的那块占用：谁被压住，看重叠区就知道。
         画在占用之后、线框之前：它是落进容器里的体量，不是浮在笼子外的贴片。 ── */
    if (reqEnd > gapEnd) {
      box(gapEnd, reqEnd, 0, CAGE.h, dangerRgb, {
        cls: "pto-hbm-snapshot__iso-overflow", fillOpacity: 0.42, dashed: true,
        title: "待分配块超出的这一截会落在这里 · 与已有占用重叠",
      });
    }

    /* ── 容器线框：虚线 12 条棱，画在实心之后才有"装在笼子里"的读法 ── */
    var L = CAGE.len, H = CAGE.h;
    [
      [[0, 0, 0], [L, 0, 0]], [[0, 0, 1], [L, 0, 1]], [[0, H, 0], [L, H, 0]], [[0, H, 1], [L, H, 1]],
      [[0, 0, 0], [0, H, 0]], [[L, 0, 0], [L, H, 0]], [[0, 0, 1], [0, H, 1]], [[L, 0, 1], [L, H, 1]],
      [[0, 0, 0], [0, 0, 1]], [[L, 0, 0], [L, 0, 1]], [[0, H, 0], [0, H, 1]], [[L, H, 0], [L, H, 1]],
    ].forEach(function (edge) {
      var a = P(edge[0][0], edge[0][1], edge[0][2]);
      var b = P(edge[1][0], edge[1][1], edge[1][2]);
      g.appendChild(svgEl("line", {
        x1: a[0], y1: a[1], x2: b[0], y2: b[1],
        stroke: wire, "stroke-width": 0.06, "stroke-dasharray": "0.26 0.2",
      }));
    });

    var reqY0 = CAGE.h + REQ.lift;
    var reqY1 = reqY0 + REQ.h;

    /* ── 落点引线：把请求块的三条边界垂直投到地址轴上，接住下面那块淡红投影。
         正面（z=0）与背面（z=1）各画一条 —— 请求块是有厚度的，只在靠屏幕这一侧
         画，另一侧就悬空了，读不出"整个体量落下来"。左边界（灰）压在空档开头，
         中间那条红线是空档的尽头，越过它的就是下面那块红体量。 ── */
    [gapStart, gapEnd, reqEnd].forEach(function (x, i) {
      [0, 1].forEach(function (z) {
        var a = P(x, 0, z);
        var b = P(x, reqY0, z);
        g.appendChild(svgEl("line", {
          x1: a[0], y1: a[1], x2: b[0], y2: b[1],
          stroke: i === 0 ? muted : danger, "stroke-width": 0.055,
          "stroke-dasharray": "0.3 0.22", opacity: i === 0 ? 0.55 : 0.85,
        }));
      });
    });

    /* ── 待分配块：同一把尺子量出来，吊在容器口上方，整块一个颜色。
         颜色**沿用它自己那一类**（激活 / 参数 / …）而不是另起一个专用色：待分配
         不是一个新类别，它就是一次还没落地的该类分配 —— 给它单独一个颜色等于在
         图例里凭空多一类。"还没放进去"由位置（悬在容器上方）表达，不由颜色表达。
         它本身也没有"好的一半坏的一半"，放不下这件事由红虚线描边与下面那块越界
         投影去说，不该把结论涂回请求本身。 ── */
    var reqRgb = resolveColor("var(--hbm-" + kind(spec.requestedKind) + ", #d97706)", host);
    box(gapStart, reqEnd, reqY0, reqY1, reqRgb, {
      cls: "pto-hbm-snapshot__iso-request",
      title: "待分配 " + fmt(requestedGB) + "（" + label(kind(spec.requestedKind)) +
        "）· 需要这么长的一段连续地址",
    });

    /* 超出的那一截：不换填充色，改用红虚线把这一小段立体地框出来 —— 说的是
       "这块实体从这里开始就没地方放了"，而不是"这块实体有一半变质了"。 */
    if (reqEnd > gapEnd) {
      outlineBox(gapEnd, reqEnd, reqY0, reqY1, danger, { cls: "pto-hbm-snapshot__iso-req-over" });
    }

    /* ── 标注 ──
       块名写在它靠屏幕的那一面上（正面居中），不吊在半空：悬空的标题和块之间隔着
       一段空白，得靠位置去猜谁是谁；印在面上就没有归属问题了。
       字色按块色亮度在黑白之间二选一 —— 块色现在由数据里的类别决定，写死白字
       碰上浅色类别就糊了。 */
    var ink = reqRgb && (0.299 * reqRgb[0] + 0.587 * reqRgb[1] + 0.114 * reqRgb[2]) / 255 > 0.62
      ? "#14181d" : "#fff";
    text((gapStart + reqEnd) / 2, -(reqY0 + reqY1) / 2 + 0.22, "待分配 " + fmt(requestedGB),
      { size: 0.62, weight: 700, fill: ink });

    if (shortageGB > 0) {
      var lead = P(reqEnd, reqY0 + REQ.h / 2, 1);
      var lx = lead[0] + 0.85;
      g.appendChild(svgEl("line", {
        x1: lead[0], y1: lead[1], x2: lx - 0.2, y2: lead[1],
        stroke: danger, "stroke-width": 0.055, "stroke-dasharray": "0.24 0.18", opacity: 0.85,
      }));
      text(lx, lead[1] + 0.24, "超出 " + shortageGB.toFixed(1) + "GB 放不下",
        { anchor: "start", size: 0.72, weight: 700, fill: danger });
    }

    // 空档尺寸标在底板下方一行：地址轴是水平的，这排数字就是一把刻度尺
    var labelY = CAGE.floor + 1.05;
    walk = 0;
    segs.forEach(function (seg, i) {
      var x0 = walk * unit;
      walk += Number(seg.sizeGB || 0);
      if (seg.kind !== "free") return;
      var mid = (x0 + walk * unit) / 2;
      var largest = i === largestIndex;
      text(mid, labelY, Number(seg.sizeGB).toFixed(2),
        { size: largest ? 0.7 : 0.6, weight: largest ? 700 : 500, fill: largest ? danger : muted });
      if (largest) text(mid, labelY + 0.84, "最大连续空闲", { size: 0.56, fill: muted });
    });

    var pad = 0.6;
    svg.setAttribute("viewBox", [
      (bounds.x0 - pad).toFixed(2), (bounds.y0 - pad).toFixed(2),
      (bounds.x1 - bounds.x0 + pad * 2).toFixed(2), (bounds.y1 - bounds.y0 + pad * 2).toFixed(2),
    ].join(" "));
    return svg;
  }

  function fact(name, value, danger, aux) {
    return '<div class="pto-hbm-snapshot__fact' + (danger ? " is-danger" : "") + '"><span>' +
      name + "</span><strong>" + esc(value) +
      (aux ? ' <em class="pto-hbm-snapshot__fact-aux">' + esc(aux) + "</em>" : "") +
      "</strong></div>";
  }

  function render(container, snapshot, options) {
    options = options || {};
    var s = snapshot.summary || {};
    var rows = snapshot.lifetimes || [];
    var selected = options.initialSelectedId || null;
    var maxT = Math.max.apply(Math, [1].concat(rows.map(function (x) { return Number(x.freeMs) || 0; })));
    var fragments = snapshot.fragmentationMap || [];
    var fragmentTotal = fragments.reduce(function (sum, x) { return sum + Number(x.sizeGB || 0); }, 0) || 1;
    var fragmentWindow = snapshot.fragmentationWindow || {};
    var fragmentStart = Number(fragmentWindow.startGB || 0);
    var fragmentEnd = fragmentStart + fragmentTotal;
    var fragmentLeft = fragmentStart / Number(s.capacityGB || 1) * 100;
    var fragmentRight = fragmentEnd / Number(s.capacityGB || 1) * 100;
    var fragmentTargetLeft = 100 / 6;
    var fragmentTargetRight = 100 - fragmentTargetLeft;
    var fragmentRows = (snapshot.fragmentAllocations || []).map(function (x) {
      var copy = {};
      Object.keys(x).forEach(function (key) { copy[key] = x[key]; });
      return copy;
    });
    var fragmentCursor = fragmentStart;
    fragments.forEach(function (segment) {
      if (segment.kind === "blocker") {
        var allocation = fragmentRows.find(function (x) { return x.id === segment.allocationId; });
        if (allocation) {
          allocation.offsetGB = fragmentCursor;
          allocation.sizeGB = Number(segment.sizeGB || 0);
        }
      }
      fragmentCursor += Number(segment.sizeGB || 0);
    });
    var detailRows = rows.concat(fragmentRows);
    var freeFragmentCount = fragments.filter(function (x) { return x.kind === "free"; }).length;
    container.innerHTML =
      '<section class="pto-hbm-snapshot">' +
        '<div class="pto-hbm-snapshot__evidence">' +
          fact("故障Rank/step", "rank " + snapshot.rank, true, "/ step " + snapshot.step) +
          fact("空闲总量", (s.freeGB || 0) + " GB") +
          fact("最大连续空闲块", (s.largestFreeBlockGB || 0) + " GB") +
          fact("本次请求", (s.requestedGB || 0) + " GB") +
        "</div>" +
        '<div class="pto-hbm-snapshot__body"><div class="pto-hbm-snapshot__plots">' +
          '<section class="pto-hbm-snapshot__section"><div class="pto-hbm-snapshot__title">rank ' +
            esc(snapshot.rank) + ' 内存分配分析</div>' +
            '<div class="pto-hbm-snapshot__verdict"><b>异常分析</b>空闲总量足够，但最大连续块 ' +
              esc(s.largestFreeBlockGB) + " GB 小于请求 " + esc(s.requestedGB) +
              " GB；总空闲 " + esc(s.freeGB) +
              " GB 分散在 " + freeFragmentCount + " 个不连续地址段，不能拼接成 " +
              esc(s.requestedGB) + " GB，引发 OOM —— 训练在 step " + esc(snapshot.incidentStep) +
              " 中断（ACL_ERROR_MEMORY_ALLOCATION），吞吐由 " + esc(s.throughputAtPeak) +
              " tokens/s 跌至 0。</div>" +
            '<div class="pto-hbm-snapshot__axis pto-hbm-snapshot__address-axis"><span>0 GB</span><span>' +
              esc(s.capacityGB / 2) + " GB</span><span>" + esc(s.capacityGB) + " GB</span></div>" +
            '<div class="pto-hbm-snapshot__address-wrap"><div class="pto-hbm-snapshot__address">' +
              rows.map(function (x) {
                return '<button class="pto-hbm-snapshot__address-block" data-id="' + esc(x.id) +
                  '" data-kind="' + kind(x.kind) + '" style="left:' + (x.offsetGB / s.capacityGB * 100) +
                  "%;width:" + (x.sizeGB / s.capacityGB * 100) + '%" title="' + esc(x.name) +
                  " · " + esc(x.sizeGB) + ' GB"></button>';
              }).join("") +
              '<span class="pto-hbm-snapshot__zoom-source" style="left:' + fragmentLeft + '%;width:' +
                (fragmentRight - fragmentLeft) + '%" title="' + esc(fragmentWindow.label || "碎片放大区域") +
                " · " + fragmentStart.toFixed(2) + "–" + fragmentEnd.toFixed(2) + ' GB">' +
                fragments.map(function (x) {
                  var miniAllocation = fragmentRows.find(function (row) { return row.id === x.allocationId; });
                  return '<i class="' + (x.kind === "free" ? "is-free" : "is-occupied") +
                    '" data-kind="' + kind(miniAllocation && miniAllocation.kind) +
                    '" style="width:' + (Number(x.sizeGB || 0) / fragmentTotal * 100) + '%"></i>';
                }).join("") + "</span>" +
            "</div></div>" +
            /* 图例紧跟总览条：这几个颜色是**先在总览条上出现**的，下面立体图只是
               沿用同一套。挂在整节末尾等于让人先看两幅图、再回头找色卡对一遍。
               待分配块不另设条目——它用的就是这几个类别色中的一个（本例是激活），
               「还没放进去」由它悬在容器上方这个位置表达，不由颜色表达。 */
            /* 图例与放大引线叠在同一块区域里：引线必须从总览条的下沿起笔才连得上，
               而图例又要紧跟总览条 —— 两者若前后排开，引线就被图例这一行顶下去、
               和上面那个白框断开了。所以让 region 占住这段高度，图例走正常流排在
               它顶部，引线绝对定位铺满整块从上沿画到下沿，图例压在引线之上。 */
            '<div class="pto-hbm-snapshot__zoom-region">' +
            '<div class="pto-hbm-snapshot__legend">' +
              ["activation", "parameters", "gradients", "optimizer", "workspace"].map(function (k) {
                return '<span title="' + esc(LEGEND_TIP[k] || "") +
                  '"><i style="background:var(--hbm-' + k + ')"></i>' + label(k) + "</span>";
              }).join("") +
              '<span title="这条总览里未逐块展开的其余已占用显存"><i class="is-other"></i>其他已占用（未展开）</span>' +
              '<span><i class="is-gap"></i>空闲碎片（合计 ' + esc(s.freeGB) + " GB）</span></div>" +
            /* 放大关系由两条直线直说：总览里那个白框的左右两边，各连到下方立体图
               白框的左右两边。原先是一条贝塞尔渐变填充 —— 面积大、边界虚，反而看不出
               "这一段被放大成了那一整幅"。两条直线是制图里表达局部放大的通用画法。 */
            '<div class="pto-hbm-snapshot__zoom-bridge" aria-label="总览框选区域展开为下方局部地址图"><span>rank ' +
              esc(snapshot.rank) + ' · ' + fragmentStart.toFixed(2) + "–" + fragmentEnd.toFixed(2) +
              ' GB 空闲区域放大</span><svg viewBox="0 0 100 120" preserveAspectRatio="none">' +
              '<line x1="' + fragmentLeft + '" y1="0" x2="' + fragmentTargetLeft + '" y2="120"></line>' +
              '<line x1="' + fragmentRight + '" y1="0" x2="' + fragmentTargetRight + '" y2="120"></line>' +
              '</svg></div></div>' +
            '<div class="pto-hbm-snapshot__iso"><div class="pto-hbm-snapshot__iso-scene"></div></div>' +
          '</section></div><aside class="pto-hbm-snapshot__detail"></aside></div>' +
      "</section>";

    var root = container.firstElementChild;
    var body = root.querySelector(".pto-hbm-snapshot__body");
    var detail = root.querySelector(".pto-hbm-snapshot__detail");

    /* 立体容器：占用块要带上名字与类别（染色与 <title> 都要用），故先把
       fragmentationMap 与 fragmentAllocations 合成一份自足的段列表再交给 builder
       —— builder 只认识"一串段 + 一个请求"，不该反过来去查 snapshot 的结构。 */
    var isoScene = root.querySelector(".pto-hbm-snapshot__iso-scene");
    if (isoScene) {
      isoScene.appendChild(buildFragmentScene({
        segments: fragments.map(function (x) {
          var allocation = x.kind === "blocker"
            ? fragmentRows.find(function (row) { return row.id === x.allocationId; })
            : null;
          return {
            kind: x.kind, sizeGB: Number(x.sizeGB || 0), id: x.allocationId || null,
            name: allocation ? allocation.name : "活跃分配",
            allocKind: kind(allocation && allocation.kind),
          };
        }),
        total: fragmentTotal,
        requestedGB: Number(s.requestedGB) || 0,
        // 待分配块沿用它自己那一类的颜色；缺省按激活算（这类 OOM 绝大多数栽在激活上）
        requestedKind: s.requestedKind || "activation",
        largestFreeGB: Number(s.largestFreeBlockGB) || 0,
        format: function (v) { return v.toFixed(2) + " GB"; },
        host: root,
      }));
    }

    function select(id) {
      var x = id ? detailRows.find(function (v) { return String(v.id) === String(id); }) : null;
      selected = x ? id : null;
      root.querySelectorAll("[data-id]").forEach(function (el) {
        el.classList.toggle("is-selected", el.dataset.id === selected);
      });
      if (!x) {
        body.classList.remove("has-detail");
        detail.innerHTML = "";
        return;
      }
      body.classList.add("has-detail");
      var duration = x.freeMs - x.allocMs;
      detail.innerHTML = '<div class="pto-hbm-snapshot__detail-head"><h3>' + esc(x.name) +
        '</h3><button class="pto-hbm-snapshot__detail-close" type="button" aria-label="关闭详情">&times;</button></div>' +
        '<div class="pto-hbm-snapshot__detail-kind">' +
        label(kind(x.kind)) + " · " + esc(x.source || "未知来源") + (x.line ? ":" + x.line : "") +
        '</div><div class="pto-hbm-snapshot__detail-life"><div><span>生命周期</span><b>' +
        esc(duration) + ' ms</b></div><div class="pto-hbm-snapshot__detail-life-track"><i data-kind="' +
        kind(x.kind) + '" style="left:' + (x.allocMs / maxT * 100) + "%;width:" +
        (duration / maxT * 100) + '%"></i></div><div class="pto-hbm-snapshot__detail-life-axis"><span>0 ms</span><span>' +
        Math.round(maxT / 2) + " ms</span><span>" + maxT + " ms</span></div>" +
        '</div><div class="pto-hbm-snapshot__kv"><div><span>地址</span><b>' + esc(x.offsetGB) + "–" +
        esc((x.offsetGB + x.sizeGB).toFixed(3)) + " GB</b></div><div><span>大小</span><b>" +
        esc(x.sizeGB) + " GB</b></div><div><span>申请</span><b>" + esc(x.allocMs) +
        " ms</b></div><div><span>释放</span><b>" + esc(x.freeMs) +
        " ms</b></div><div><span>持有</span><b>" + esc(duration) +
        ' ms</b></div></div><div class="pto-hbm-snapshot__stack"><b>调用栈</b><br>' +
        esc(x.stack || "暂无调用栈") +
        '</div><div class="pto-hbm-snapshot__actions"><button class="btn btn-ghost btn-sm" data-action="timeline">Timeline</button>' +
        '<button class="btn btn-ghost btn-sm" data-action="source">源码</button></div>';
    }

    root.addEventListener("click", function (e) {
      if (e.target.closest(".pto-hbm-snapshot__detail-close")) { select(null); return; }
      var block = e.target.closest("[data-id]");
      if (block) select(block.dataset.id);
      var action = e.target.closest("[data-action]");
      if (!action) return;
      var x = detailRows.find(function (v) { return String(v.id) === String(selected); });
      if (action.dataset.action === "timeline") {
        if (options.onOpenTimeline) options.onOpenTimeline(x);
      } else if (options.onOpenSource) {
        options.onOpenSource(x);
      }
    });
    select(selected);
    return {
      select: select,
      resize: function () {},
      destroy: function () { container.innerHTML = ""; }
    };
  }

  global.PtoHbmMemorySnapshot = { render: render };
})(window);
