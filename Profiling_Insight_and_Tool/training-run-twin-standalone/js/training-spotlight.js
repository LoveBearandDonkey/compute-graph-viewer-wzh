/* ══════════════════════════════════════════════════════════════════════════
   问题一 · 聚光灯定位链  (window.PtoTrainingSpotlight)
   —— 进入「问题一」时（training-run-twin.js 的 activateProblemOneLens 调用 open），
      在实页之上覆盖暗遮罩，按 1→6 步进：每一步把当前证据图表挪到可见（展开 infra 列 /
      侧栏滚动 / 切底部 dock 页签），在它上面「开洞」照亮，配编号徽标 + 引出线 + 一句话标注；
      顶部问题名片常驻、右侧「修改建议」列展示 6 处代码修复并高亮当前步关联项。
   —— 数据是本页「问题一」事故（router FP8 溢出 → 路由塌缩 → NaN + all-to-all 死锁双发）的
      固定复盘，一句话结论 + 关键数字与 training-run-twin.js 的 locateChains["moe-a2a"] 同源；
      需要读全文时点名片「详情」→ 复用既有 window.openProblemOneLocateDrawer 抽屉。
   —— 样式见 css/training-spotlight.css；本层 z=1500，详情抽屉 z=2000 在其之上不被遮。
   ══════════════════════════════════════════════════════════════════════════ */
(function () {
  "use strict";

  var MARGIN = 8;

  // ── 名片 ───────────────────────────────────────────────────────────────
  var META = {
    kicker: "问题一",
    severity: "P0",
    title: "Router 数值溢出 → 路由塌缩，同时触发 loss NaN 与 all-to-all 死锁",
    tags: ["Layer 38 · MoE Router", "精度"],
  };

  // ── 右侧「修改建议」6 处代码修复（与 locateChains 超参/代码层同源，压成一屏）──────
  // line = 该片段在原文件中的起始行号（让行号栏贴近真实文件，不是每块都从 1 开始）
  var FIXES = [
    { file: "router.py", line: 142, title: "softmax 改 FP32", diff: [
      ["-", "probs = softmax(router(x))          # FP8 下 logits 溢出"],
      ["+", "probs = softmax(router(x.float()))  # FP32 计算后再 cast 回"] ] },
    { file: "training_args.yaml", line: 68, title: "z-loss + grad clip", diff: [
      ["+", "z_loss_coeff: 1e-4   # 抑制 logits 极端值"],
      ["+", "clip_grad_norm: 1.0  # MoE 训练标配"] ] },
    { file: "optimizer_config.json", line: 31, title: "router 学习率", diff: [
      ["-", "router_lr: 3e-4      # 与 expert 相同"],
      ["+", "router_lr: 3e-5      # expert lr × 0.1"] ] },
    { file: "model_config.json", line: 24, title: "n_group 辅助保障", diff: [
      ["-", "\"n_group\": 8,"],
      ["+", "\"n_group\": 16,     # 分散 expert 选择"] ] },
    { file: "env.sh", line: 17, title: "NCCL 超时兜底", diff: [
      ["-", "export NCCL_IB_TIMEOUT=30"],
      ["+", "export NCCL_IB_TIMEOUT=60  # 训练不中断兜底"] ] },
    { file: "monitor_config.yaml", line: 53, title: "部署熔断规则", diff: [
      ["+", "amp_scaler_dump_ratio: 0.0625   # 1/16 → 🟠 自动 dump"],
      ["+", "amp_scaler_abort_ratio: 0.03125 # 1/32 → 🔴 停训"] ] },
  ];

  // ── 6 个证据步（mockup 编号 ①熔断 ②迭代 ③日志 ④通信 ⑤模型 ⑥infra）────────────
  // target: 返回被照亮的实页元素；prep: 进入该步前把它挪到可见；fix: 关联的修复项下标。
  var STEPS = [
    { n: 1, layer: "熔断 / 预警层", short: "熔断/预警",
      target: function () { return document.querySelector('#accuracyCharts [data-acc-card="loss_scale"]'); },
      prep: function () { scrollCardIntoView('#accuracyCharts [data-acc-card="loss_scale"]'); },
      body: "其实从 step 15000 起，AMP loss scale 就在 65536→4096 一路衰减、连越三级阈值——但无人监控，直到 15203 才爆发。",
      nums: ["scaler 65536→4096", "本可提前 53 step 止损"],
      fix: [5] },
    { n: 2, layer: "迭代层", short: "迭代层",
      // loss 与 grad_norm 两张卡一起框（都在 step 15203 跳变），取二者并集
      target: function () {
        return [
          document.querySelector('#accuracyCharts [data-acc-card="loss"]'),
          document.querySelector('#accuracyCharts [data-acc-card="gradnorm"]'),
        ];
      },
      prep: function () { scrollCardIntoView('#accuracyCharts [data-acc-card="loss"]', "start"); },
      body: "step 15203：loss 跳变 NaN、grad_norm→inf。固定 seed 重跑——单卡 loss=3.21 正常、32 卡即 NaN，仅多卡复现，指向通信。",
      nums: ["loss→NaN", "grad_norm→inf", "仅多卡复现"],
      fix: [] },
    { n: 3, layer: "日志 / plog 诊断层", short: "日志/plog",
      target: function () { return document.getElementById("dockPanelLog") || document.getElementById("bottomDock"); },
      prep: function () {
        // 幂等地展开 dock、切日志页签并渲染日志行（training-log-drawer.js 暴露的 show()）。
        // 之前靠点 #trainLogToggle 触发渲染，受其开关状态时序影响；show() 每次都渲染，最稳。
        if (window.PtoTrainingLogDrawer && window.PtoTrainingLogDrawer.show) {
          window.PtoTrainingLogDrawer.show();
        } else {
          window.PtoTrainingTwinTimelineDock && window.PtoTrainingTwinTimelineDock.setVisible(true);
          window.PtoTrainingTwinDockTabs && window.PtoTrainingTwinDockTabs.select("log");
        }
      },
      body: "Python 侧只报 NCCL timeout；plog 翻译后：rank 23 all-to-all send=0 / recv=9832 buffer 失配，且 router_logits 含 inf——把「看不懂的报错」翻成人话。",
      nums: ["send=0 / recv=9832", "router_logits 含 inf"],
      fix: [] },
    { n: 4, layer: "通信调度层", short: "通信调度",
      target: function () { return document.getElementById("twinTimeline") || document.getElementById("bottomDock"); },
      prep: function () {
        window.PtoTrainingTwinTimelineDock && window.PtoTrainingTwinTimelineDock.setVisible(true);
        window.PtoTrainingTwinDockTabs && window.PtoTrainingTwinDockTabs.select("timeline");
      },
      body: "EP rank 23（node2 GPU7）在 all-to-all 处 30s 超时死锁：send=0，其余 63 rank 卡在同步屏障空等——但死锁只是「果」。",
      nums: ["rank 23 timeout", "63 rank 空等"],
      fix: [4] },
    { n: 5, layer: "模型层 → 数值层（根因）", short: "模型·数值",
      target: function () { return document.getElementById("deckStage"); },
      prep: function () { /* activateProblemOneLens 已聚焦 layer 38 router 并展开 routed_expert_bank */ },
      body: "layer 38 router 把 98% token 路由到 expert 193，247 个 dead expert = 路由彻底塌缩。根因：router softmax 在 FP8 下 max(logits)=1846 → exp 溢出为 inf。",
      nums: ["98% token → E193", "max(logits)=1846→inf", "FP8 softmax 溢出"],
      fix: [0, 1, 2, 3] },
    { n: 6, layer: "infra 层（扩散）", short: "infra 扩散",
      target: function () { return document.getElementById("heat"); },
      prep: function () {
        window.PtoTrainingTwinSideCols && window.PtoTrainingTwinSideCols.setRightVisible(true);
        scrollCardIntoView("#heat");
      },
      body: "单点 rank 23 溢出经 all-to-all 同步屏障扩散 → 64 EP rank 全卡 → PP stage 3 断裂 → 2048 NPU hang，报的却是通信 timeout，而非 router 溢出。",
      nums: ["1 → 2048 NPU", "报错位置 ≠ 根因位置"],
      fix: [4] },
  ];

  function scrollCardIntoView(sel, block) {
    var el = document.querySelector(sel);
    if (el && el.scrollIntoView) {
      try { el.scrollIntoView({ block: block || "center", inline: "nearest", behavior: "smooth" }); }
      catch (e) { el.scrollIntoView(); }
    }
  }

  // ── DOM 装配 ──────────────────────────────────────────────────────────
  var els = null; // 缓存构建好的 DOM
  var open = false;
  var idx = 0;
  var raf = 0;
  var autoTimer = 0;
  var keyHandler = null;

  function h(tag, cls, html) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }
  var SVGNS = "http://www.w3.org/2000/svg";
  function svg(tag, attrs) {
    var e = document.createElementNS(SVGNS, tag);
    if (attrs) for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }

  function build() {
    if (els) return els;
    var root = h("div", "tw-spot");

    // 场景 SVG：光洞四周四片遮板（压暗+拦截交互，光洞区穿透可缩放/拖动底层图表）+ 光框 + 引出线
    var scene = svg("svg", { "class": "tw-spot__scene" });
    var shTop = svg("rect", { "class": "tw-spot__shutter" });
    var shBottom = svg("rect", { "class": "tw-spot__shutter" });
    var shLeft = svg("rect", { "class": "tw-spot__shutter" });
    var shRight = svg("rect", { "class": "tw-spot__shutter" });
    var glow = svg("rect", { "class": "tw-spot__glow", rx: 12, ry: 12 });
    var connector = svg("path", { "class": "tw-spot__connector" });
    var dot = svg("circle", { "class": "tw-spot__connector-dot", r: 3 });
    scene.appendChild(shTop); scene.appendChild(shBottom); scene.appendChild(shLeft); scene.appendChild(shRight);
    scene.appendChild(glow); scene.appendChild(connector); scene.appendChild(dot);

    // 顶部名片
    var card = h("div", "tw-spot__card");
    var kicker = h("div", "tw-spot__card-kicker");
    kicker.appendChild(document.createTextNode(META.kicker));
    var sev = h("span", "tw-spot__sev"); sev.textContent = META.severity; kicker.appendChild(sev);
    var title = h("div", "tw-spot__card-title"); title.textContent = META.title;
    var tags = h("div", "tw-spot__card-tags");
    META.tags.forEach(function (t) { var s = h("span", "tw-spot__tag"); s.textContent = t; tags.appendChild(s); });
    var actions = h("div", "tw-spot__card-actions");
    var detailBtn = h("button", "btn btn-ghost btn-sm"); detailBtn.type = "button"; detailBtn.textContent = "详情"; detailBtn.style.fontSize = "11px";
    detailBtn.addEventListener("click", function () { window.openProblemOneLocateDrawer && window.openProblemOneLocateDrawer(); });
    var closeBtn = h("button", "pto-ide-frame__window-action"); closeBtn.type = "button"; closeBtn.title = "退出聚光灯，回到最新 step"; closeBtn.setAttribute("aria-label", "退出聚光灯"); closeBtn.innerHTML = "&#10005;";
    closeBtn.addEventListener("click", exit);
    actions.appendChild(detailBtn); actions.appendChild(closeBtn);
    card.appendChild(kicker); card.appendChild(title); card.appendChild(tags); card.appendChild(actions);

    // 步进导轨
    var rail = h("div", "tw-spot__rail");
    var stepEls = STEPS.map(function (st, i) {
      var s = h("div", "tw-spot__step");
      var num = h("span", "tw-spot__step-n"); num.textContent = st.n;
      var name = h("span", "tw-spot__step-name"); name.textContent = st.short || st.layer;
      s.appendChild(num); s.appendChild(name);
      s.addEventListener("click", function () { go(i); });
      rail.appendChild(s);
      return s;
    });

    // 证据标注气泡
    var callout = h("div", "tw-spot__callout");

    // 「修改建议」——聚光灯期间挤入页面栅格作第四栏（见 doOpen 的注入），常驻显示
    var fixes = h("aside", "wzh-col-spot-fixes");
    fixes.setAttribute("aria-label", "修改建议");
    var fhead = h("div", "tw-spot__fixes-head");
    fhead.appendChild(h("span", "tw-spot__fixes-title", "修改建议"));
    fhead.appendChild(h("span", "tw-spot__fixes-sub", "6 处 · 按优先级"));
    var fbody = h("div", "tw-spot__fixes-body");
    var fixEls = FIXES.map(function (fx, i) {
      var f = h("div", "tw-spot__fix");
      var fh = h("div", "tw-spot__fix-head");
      var fn = h("span", "tw-spot__fix-n"); fn.textContent = i + 1;
      var ff = h("span", "tw-spot__fix-file"); ff.textContent = fx.file;
      var ft = h("span", "tw-spot__fix-title"); ft.textContent = fx.title;
      fh.appendChild(fn); fh.appendChild(ff); fh.appendChild(ft);
      var fd = h("div", "tw-spot__fix-diff");
      fx.diff.forEach(function (ln, li) {
        var row = h("div", "tw-spot__diff-row " + (ln[0] === "-" ? "del" : "add"));
        var num = h("span", "tw-spot__diff-ln"); num.textContent = (fx.line || 1) + li;
        var code = h("span", "tw-spot__diff-code");
        code.textContent = (ln[0] === "-" ? "− " : "+ ") + ln[1];
        row.appendChild(num); row.appendChild(code);
        fd.appendChild(row);
      });
      f.appendChild(fh); f.appendChild(fd);
      f.addEventListener("click", function () { window.openProblemOneLocateDrawer && window.openProblemOneLocateDrawer(); });
      fbody.appendChild(f);
      return f;
    });
    fixes.appendChild(fhead); fixes.appendChild(fbody);

    // 底部导航
    var nav = h("div", "tw-spot__nav");
    var prev = h("button", "tw-spot__nav-btn"); prev.type = "button"; prev.title = "上一步"; prev.innerHTML = "&#8249;";
    prev.addEventListener("click", function () { go(idx - 1); });
    var play = h("button", "tw-spot__nav-btn"); play.type = "button"; play.title = "自动播放"; play.innerHTML = "&#9654;";
    play.addEventListener("click", toggleAuto);
    var count = h("div", "tw-spot__nav-count");
    var next = h("button", "tw-spot__nav-btn"); next.type = "button"; next.title = "下一步"; next.innerHTML = "&#8250;";
    next.addEventListener("click", function () { go(idx + 1); });
    nav.appendChild(prev); nav.appendChild(play); nav.appendChild(count); nav.appendChild(next);

    // 导轨 + 导航同处顶部工具行（导轨在左、导航在右）
    var toolbar = h("div", "tw-spot__toolbar");
    toolbar.appendChild(rail);
    toolbar.appendChild(nav);

    root.appendChild(scene);
    root.appendChild(card);
    root.appendChild(toolbar);
    root.appendChild(callout);
    // fixes 不进覆盖层：doOpen 时注入 .twin-center-scroll 作第四栏
    document.body.appendChild(root);

    els = {
      root: root, scene: scene,
      sh: { top: shTop, bottom: shBottom, left: shLeft, right: shRight },
      glow: glow, connector: connector, dot: dot,
      rail: rail, stepEls: stepEls, callout: callout,
      fixes: fixes, fixEls: fixEls, nav: nav, prev: prev, next: next, play: play, count: count,
    };
    return els;
  }

  // ── 步导航 ────────────────────────────────────────────────────────────
  function go(i) {
    i = Math.max(0, Math.min(STEPS.length - 1, i));
    idx = i;
    var st = STEPS[i];
    var e = els;

    // 导轨状态
    e.stepEls.forEach(function (s, k) {
      s.classList.toggle("is-active", k === i);
      s.classList.toggle("is-done", k < i);
    });
    // 计数 + 前后按钮
    e.count.innerHTML = "<b>" + st.n + "</b> / " + STEPS.length;
    e.prev.disabled = i === 0;
    e.next.disabled = i === STEPS.length - 1;

    // 标注气泡内容
    var linkedFirst = st.fix.length ? FIXES[st.fix[0]] : null;
    var numsHtml = st.nums.map(function (n) { return '<span class="tw-spot__num"></span>'; }).join("");
    e.callout.innerHTML =
      '<div class="tw-spot__callout-head">' +
      '<span class="tw-spot__callout-n">' + st.n + '</span>' +
      '<span class="tw-spot__callout-layer"></span></div>' +
      '<div class="tw-spot__callout-body"></div>' +
      '<div class="tw-spot__callout-nums">' + numsHtml + '</div>' +
      '<div class="tw-spot__callout-fix"' + (linkedFirst ? '' : ' hidden') + '>' +
      '<b>→ 修复</b> <span class="tw-spot__callout-fix-txt"></span></div>';
    e.callout.querySelector(".tw-spot__callout-layer").textContent = st.layer;
    e.callout.querySelector(".tw-spot__callout-body").textContent = st.body;
    var numEls = e.callout.querySelectorAll(".tw-spot__num");
    st.nums.forEach(function (n, k) { if (numEls[k]) numEls[k].textContent = n; });
    if (linkedFirst) {
      e.callout.querySelector(".tw-spot__callout-fix-txt").textContent =
        linkedFirst.file + " · " + linkedFirst.title + (st.fix.length > 1 ? " 等 " + st.fix.length + " 处" : "");
    }

    // 右列修复高亮
    e.fixes.classList.toggle("has-link", st.fix.length > 0);
    e.fixEls.forEach(function (f, k) { f.classList.toggle("is-linked", st.fix.indexOf(k) >= 0); });
    if (st.fix.length && e.fixEls[st.fix[0]]) {
      try { e.fixEls[st.fix[0]].scrollIntoView({ block: "nearest", behavior: "smooth" }); } catch (x) {}
    }

    // 把当前证据挪到可见
    if (st.prep) { try { st.prep(); } catch (x) {} }

    // 光框脉冲一次
    e.root.classList.remove("is-pulsing");
    void e.root.offsetWidth; // 触发重排以重启动画
    e.root.classList.add("is-pulsing");
    window.setTimeout(function () { e.root.classList.remove("is-pulsing"); }, 620);
  }

  function toggleAuto() {
    if (autoTimer) { stopAuto(); return; }
    els.play.innerHTML = "&#10073;&#10073;"; // ‖
    els.play.title = "暂停";
    autoTimer = window.setInterval(function () {
      if (idx >= STEPS.length - 1) { stopAuto(); return; }
      go(idx + 1);
    }, 4200);
  }
  function stopAuto() {
    if (autoTimer) { window.clearInterval(autoTimer); autoTimer = 0; }
    if (els) { els.play.innerHTML = "&#9654;"; els.play.title = "自动播放"; }
  }

  // ── 几何：每帧按当前证据 rect 重排开洞 / 光框 / 引出线 / 气泡 ──────────────
  function borderPoint(r, tx, ty) {
    var cx = r.x + r.w / 2, cy = r.y + r.h / 2;
    var dx = tx - cx, dy = ty - cy;
    if (dx === 0 && dy === 0) return { x: cx, y: cy };
    var sx = dx !== 0 ? (r.w / 2) / Math.abs(dx) : Infinity;
    var sy = dy !== 0 ? (r.h / 2) / Math.abs(dy) : Infinity;
    var s = Math.min(sx, sy);
    return { x: cx + dx * s, y: cy + dy * s };
  }

  // 目标可以是单个元素或一组元素；取所有可见者的并集外框（含内边距）
  function unionBox(t, vw, vh) {
    var pad = 8;
    if (!t) return { x: 0, y: 0, w: 0, h: 0, ok: false };
    var list = Array.isArray(t) ? t : [t];
    var l = Infinity, tp = Infinity, r = -Infinity, b = -Infinity, any = false;
    list.forEach(function (el) {
      if (!el || !el.getBoundingClientRect) return;
      var rr = el.getBoundingClientRect();
      if (rr.width > 2 && rr.height > 2 && rr.bottom > 0 && rr.top < vh) {
        l = Math.min(l, rr.left); tp = Math.min(tp, rr.top);
        r = Math.max(r, rr.right); b = Math.max(b, rr.bottom); any = true;
      }
    });
    if (!any) return { x: 0, y: 0, w: 0, h: 0, ok: false };
    return { x: l - pad, y: tp - pad, w: (r - l) + pad * 2, h: (b - tp) + pad * 2, ok: true };
  }

  function reflow() {
    raf = window.requestAnimationFrame(reflow);
    if (!open || !els) return;
    var e = els, st = STEPS[idx];
    var vw = window.innerWidth, vh = window.innerHeight;

    // svg 尺寸
    e.scene.setAttribute("width", vw);
    e.scene.setAttribute("height", vh);
    e.scene.setAttribute("viewBox", "0 0 " + vw + " " + vh);

    // 「修改建议」第四栏是真实页面列，聚光灯给它让出一条常亮右栏：遮板右边界止于它左沿(canvasRight)。
    var canvasRight = vw;
    if (e.fixes && e.fixes.parentNode) {
      var fr = e.fixes.getBoundingClientRect();
      if (fr.width > 0) canvasRight = fr.left - 6;
    }

    // 目标 rect（支持一个元素或一组元素取并集；缺失/不可见时退回画布中央的浮框）
    var box = unionBox(st.target ? st.target() : null, vw, vh);
    var hasTarget = box.ok;
    var topR = 104, botR = 16; // 顶部名片 + 工具行约 104px；导航已上移，底部只留边距
    if (!hasTarget) { box = { x: canvasRight * 0.28, y: vh * 0.42, w: canvasRight * 0.34, h: 150 }; }
    // 夹进画布（画布右界 = canvasRight，避免压到常亮的修改建议栏）
    box.x = Math.max(MARGIN, Math.min(box.x, canvasRight - MARGIN - box.w));
    box.y = Math.max(topR, Math.min(box.y, vh - botR - box.h));

    // 四片遮板围出光洞（中间穿透）+ 光框；遮板右界止于 canvasRight，右侧修改建议栏保持常亮
    setRect(e.sh.top, { x: 0, y: 0, w: canvasRight, h: box.y });
    setRect(e.sh.bottom, { x: 0, y: box.y + box.h, w: canvasRight, h: Math.max(0, vh - (box.y + box.h)) });
    setRect(e.sh.left, { x: 0, y: box.y, w: box.x, h: box.h });
    setRect(e.sh.right, { x: box.x + box.w, y: box.y, w: Math.max(0, canvasRight - (box.x + box.w)), h: box.h });
    setRect(e.glow, box);
    e.glow.style.display = hasTarget ? "" : "none";

    // 标注气泡定位：在画布内选空间最大的一侧
    var cw = e.callout.offsetWidth || 300;
    var ch = e.callout.offsetHeight || 150;
    var spaceRight = canvasRight - (box.x + box.w);
    var spaceLeft = box.x - MARGIN;
    var spaceBottom = vh - botR - (box.y + box.h);
    var spaceTop = box.y - topR;
    var cx, cy;
    if (spaceRight >= cw + 20) { cx = box.x + box.w + 16; cy = box.y; }
    else if (spaceLeft >= cw + 20) { cx = box.x - 16 - cw; cy = box.y; }
    else if (spaceBottom >= ch + 20) { cy = box.y + box.h + 16; cx = box.x; }
    else if (spaceTop >= ch + 20) { cy = box.y - 16 - ch; cx = box.x; }
    else { cx = box.x + box.w + 16; cy = box.y; } // 兜底右侧
    cx = Math.max(MARGIN, Math.min(cx, canvasRight - MARGIN - cw));
    cy = Math.max(topR, Math.min(cy, vh - botR - ch));
    e.callout.style.left = cx + "px";
    e.callout.style.top = cy + "px";

    // 引出线：气泡边缘 → 光洞边缘
    var cRect = { x: cx, y: cy, w: cw, h: ch };
    var holeCx = box.x + box.w / 2, holeCy = box.y + box.h / 2;
    var start = borderPoint(cRect, holeCx, holeCy);
    var end = borderPoint(box, cx + cw / 2, cy + ch / 2);
    var mx = (start.x + end.x) / 2, my = (start.y + end.y) / 2;
    e.connector.setAttribute("d", "M" + start.x + "," + start.y + " Q" + mx + "," + my + " " + end.x + "," + end.y);
    e.connector.style.display = hasTarget ? "" : "none";
    e.dot.setAttribute("cx", end.x); e.dot.setAttribute("cy", end.y);
    e.dot.style.display = hasTarget ? "" : "none";
  }
  function setRect(node, b) {
    node.setAttribute("x", b.x); node.setAttribute("y", b.y);
    node.setAttribute("width", b.w); node.setAttribute("height", b.h);
  }

  // ── 开 / 关 ───────────────────────────────────────────────────────────
  function doOpen(caseKey) {
    if (caseKey && caseKey !== "moe-a2a") return false; // 目前仅问题一有聚光灯
    build();
    if (open) { go(idx); return true; }
    open = true;
    idx = 0;
    // 一开场就展开 infra 栏：布局提前定型，⑥ 热力图在场；随后把「修改建议」挤入 workarea 作整高右列
    //（与底部 Timeline 左右并排、撑满整页高）。
    try { window.PtoTrainingTwinSideCols && window.PtoTrainingTwinSideCols.setRightVisible(true); } catch (x) {}
    var workarea = document.querySelector(".pto-ide-frame__workarea");
    if (workarea && els.fixes && els.fixes.parentNode !== workarea) {
      workarea.appendChild(els.fixes);
      workarea.classList.add("is-spot-fixes");
      window.dispatchEvent(new Event("resize"));
    }
    els.root.classList.add("is-open");
    go(0);
    if (!raf) raf = window.requestAnimationFrame(reflow);
    keyHandler = function (ev) {
      if (!open) return;
      if (ev.key === "Escape") { ev.preventDefault(); ev.stopPropagation(); exit(); }
      else if (ev.key === "ArrowRight") { ev.preventDefault(); ev.stopPropagation(); go(idx + 1); }
      else if (ev.key === "ArrowLeft") { ev.preventDefault(); ev.stopPropagation(); go(idx - 1); }
    };
    document.addEventListener("keydown", keyHandler, true);
    return true;
  }

  function doClose() {
    if (!open) return;
    open = false;
    stopAuto();
    if (raf) { window.cancelAnimationFrame(raf); raf = 0; }
    if (els) {
      els.root.classList.remove("is-open");
      // 移除挤入的「修改建议」整高右列，workarea 复原为纵向单块
      var g = els.fixes && els.fixes.parentNode;
      if (g) { g.removeChild(els.fixes); g.classList && g.classList.remove("is-spot-fixes"); window.dispatchEvent(new Event("resize")); }
    }
    if (keyHandler) { document.removeEventListener("keydown", keyHandler, true); keyHandler = null; }
  }

  // 名片 × / ESC：走页面既有「退出时光机」逻辑（复位图表到最新 step），它会回调 close()
  function exit() {
    var btn = document.getElementById("diagnosisLocatorClose");
    if (btn) btn.click();   // → exitTimeMachine() → PtoTrainingSpotlight.close()
    else doClose();
  }

  window.PtoTrainingSpotlight = {
    open: doOpen,
    close: doClose,
    isOpen: function () { return open; },
  };
})();
