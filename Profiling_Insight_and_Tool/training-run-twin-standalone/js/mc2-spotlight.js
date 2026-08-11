/* ══════════════════════════════════════════════════════════════════════════
   MC2 聚光灯定位链  (window.PtoTrainingSpotlight)
   —— fork 自 js/training-spotlight.js：遮罩/开洞/步进/修改建议列这套引擎原样保留，
      只把 CASES 换成 MC2 案例。引擎本身与场景无关，训练版那两个案例不受影响。
   —— 由 js/mc2-incident-monitoring.js 的 openCase() 调 open("mc2-executor") 进入。
   —— 七步与 mc2-incident-observer.js 的 INCIDENT_GROUPS 一一对应；第 5、6 步在本页
      没有取证落点，prep() 直接开观测页深链（?event=<id>）—— 这是本案与训练版最大的
      结构差别：重心在观测页，监控页只承担表象与定界。
   —— 样式复用 css/training-spotlight.css；本层 z=1500。
   ══════════════════════════════════════════════════════════════════════════ */
(function () {
  "use strict";

  var MARGIN = 8;

  function scrollCardIntoView(sel, block) {
    var el = document.querySelector(sel);
    if (el && el.scrollIntoView) {
      try { el.scrollIntoView({ block: block || "center", inline: "nearest", behavior: "smooth" }); }
      catch (e) { el.scrollIntoView(); }
    }
  }

  /* ── 问题注册表 ────────────────────────────────────────────────────────────
     meta  : 顶部名片（kicker/severity/title/tags）
     infraCol : 本问题是否需要「集群监控」右列。有步骤要照亮它才展开，否则进来就收起 ——
                聚光灯期间「修改建议」要挤进同一行做第四栏，留一条用不上的列白占宽度。
     fixes : 右侧「修改建议」列。line = 该片段在原文件中的起始行号（让行号栏贴近真实文件）
     steps : 证据步。target 返回被照亮的实页元素（可以是数组，取并集外框）；
             prep 在进入该步前把它挪到可见；fix 是关联的 fixes 下标。
             ⚠️ target 找不到元素时会退回画布中央的浮框、只展示标注气泡 —— 图表组件尚未落地的
                步骤因此可以先把叙事写全，等组件到位后选择器自动生效。
     ─────────────────────────────────────────────────────────────────────── */
  var CASES = {};

  // 跳到 MC2 观测页的对应事件。第 ⑤⑥ 步在本页没有可照亮的实页元素 ——
  // 根因不在模型结构里，整网图这次帮不上忙 —— 所以它们的取证在观测页，
  // 这里只负责把用户送过去。
  function openObserver(eventId) {
    window.open("mc2-incident-observer.html?event=" + encodeURIComponent(eventId), "_blank", "noopener");
  }

  /* ══ MC2 · 多 graph 共享 aclOpExecutor → CCU mission 污染 → NotifyWait 死锁 ══
     七步与 mc2-incident-observer.js 的 INCIDENT_GROUPS 一一对应（eventId 即深链参数）。
     叙事与数字取自 history.md。与训练版两个案例最大的不同：第 5、6 步在本页
     没有取证落点，是**故意的** —— 「本页已经查不下去了」本身就是这条链的结论。 */
  CASES["mc2-executor"] = {
    meta: {
      kicker: "MC2",
      severity: "P0",
      title: "多 graph 共享 aclOpExecutor → CCU mission 污染 → NotifyWait 死锁",
      tags: ["AllGatherMatmulV2 / AlltoAllMatmul", "通信 / 软件栈"],
    },
    infraCol: true,     // 步 ③④ 要照亮 infra 栏的四卡热力与单元卡
    fixes: [
      { file: "opapi · aclnn_all_gather_matmul_v2.cpp", line: 118, title: "根因：每次返回独立 executor", diff: [
        ["-", "executor = GetCachedExecutor(op_key);      // graph capture 下被跨 graph 复用"],
        ["+", "executor = CreateExecutor(op_key);         // capture 模式禁用缓存"] ] },
      { file: "HCCL · ccu_mission_pool.cc", line: 74, title: "防御：mission 绑定 model_id", diff: [
        ["+", "mission->owner_model_id = ctx.model_id;    // 跨 graph 复用即拒绝执行"],
        ["+", "if (mission->owner_model_id != ctx.model_id) return HCCL_E_INTERNAL;"] ] },
      { file: "torch_npu · npu_graph_capture.py", line: 231, title: "业务：按 graph 隔离 executor 缓存", diff: [
        ["-", "self._exec_cache = _GLOBAL_EXEC_CACHE"],
        ["+", "self._exec_cache = {}   # 每个 capture 各持一份，不共享全局缓存"] ] },
      { file: "vllm_config.yaml", line: 46, title: "临时规避：切 AICPU 通信引擎", diff: [
        ["-", "  mc2_comm_mode: \"auto\"     # 在 950 上实走 CCU"],
        ["+", "  mc2_comm_mode: \"ai_cpu\"   # 绕开 CCU，代价 5-15% 吞吐"] ] },
      { file: "vllm_config.yaml", line: 52, title: "缩小污染面（辅助）", diff: [
        ["-", "  cudagraph_mode: FULL_PREFILL_AND_DECODE   # 15 个 graph"],
        ["+", "  cudagraph_mode: PIECEWISE                 # 只捕 decode，graph 数减半"] ] },
    ],
    // ①耗时恶化 ②plog 表象 ③四卡定界 ④CCU 定界 ⑤根因(观测页) ⑥配置(观测页) ⑦修复
    steps: [
      { n: 1, layer: "表象层 · 耗时", short: "耗时恶化",
        eventId: "mc2-degrade",
        target: function () { return document.querySelector('#accuracyCharts [data-acc-card="forward_time"]'); },
        prep: function () { scrollCardIntoView('#accuracyCharts [data-acc-card="forward_time"]'); },
        body: "先看见的不是报错，是变慢：forward_time 1.97s → 21.9s → 107s → 168s，约三分钟涨 85 倍。关键在形状 —— 每一轮都比上一轮慢，是线性累积而不是零星尖刺。race condition 只会偶尔慢几次；会一路变慢的，是被反复复用、每次复用都更脏的那种资源。这一条就把「偶发争抢」整个方向排掉了。",
        nums: ["1x → 85x", "约 3 分钟", "线性累积，非 race"],
        fix: [] },

      { n: 2, layer: "报错层 · plog 翻译", short: "plog",
        eventId: "mc2-hang",
        target: function () { return document.getElementById("dockPanelLog") || document.getElementById("bottomDock"); },
        prep: function () {
          if (window.PtoTrainingLogDrawer && window.PtoTrainingLogDrawer.show) {
            window.PtoTrainingLogDrawer.show();
          } else {
            window.PtoTrainingTwinTimelineDock && window.PtoTrainingTwinTimelineDock.setVisible(true);
            window.PtoTrainingTwinDockTabs && window.PtoTrainingTwinDockTabs.select("log");
          }
        },
        body: "Python 侧只有一句 RuntimeError / 507011，指向通信同步 —— 照它去查 HCCL 就查反了。plog 翻译后能看到真正的入口：sqe_type=7(notify wait)、errType=0x20，四次复现跨 3 个日期、2 种算子全中同样四项特征。同步超时是结果，notify 等不到才是起点。",
        nums: ["507011", "sqe_type=7 notify wait", "4/4 次复现同构"],
        fix: [] },

      { n: 3, layer: "集群层 · 谁是真凶", short: "四卡定界",
        eventId: "mc2-culprit",
        target: function () {
          return [document.getElementById("heat"), document.getElementById("mc2RankTable")];
        },
        prep: function () {
          window.PtoTrainingTwinSideCols && window.PtoTrainingTwinSideCols.setRightVisible(true);
          scrollCardIntoView("#heat");
        },
        body: "四张卡全报 EZ9999，按报错时间排：rank 2 最先（+0ms）、rank 1 排第三（+697ms）。但只有 rank 1 的 cqeStatus=0x8000，其余三张都是 0 —— 带着 device 端异常状态的那张才是真凶，先喊的那张只是先被拖住。这里按时间排序会把结论整个排反，判据必须是 cqeStatus。",
        nums: ["首报 rank 2 ≠ 真凶", "rank 1 · cqeStatus=0x8000", "其余三卡 CQE 干净"],
        fix: [] },

      { n: 4, layer: "硬件层 · 哪个单元", short: "CCU 定界",
        eventId: "mc2-ccu",
        // 中区的三个单元对象是主证据（AIC/AIV 压暗、CCU 描红到 mission 粒度），
        // infra 栏那张小卡是同一件事的读数，一起框进来
        target: function () {
          return [
            document.getElementById("mc2UnitStage"),
            document.getElementById("mc2UnitCard"),
          ];
        },
        prep: function () {
          window.PtoTrainingTwinSideCols && window.PtoTrainingTwinSideCols.setRightVisible(true);
          window.Mc2Monitoring && window.Mc2Monitoring.fitUnitStage && window.Mc2Monitoring.fitUnitStage();
          scrollCardIntoView("#mc2UnitCard");
        },
        body: "再下一级。先看清一件事：AllGatherMatmulV2_…_64_mix_aic 是**一个** kernel（MIX_AIC、taskRation 1:2、crossCoreSync=1），AIC / AIV / CCU 是它的三个执行部位，不是三个独立部件。所以 aicError=0、aivError=0 并不表示「计算单元与本案无关」—— 它们就在这个 kernel 里，停在 cross-core 同步点上等 CCU 的 notify，而等待不产生错误码。真正断掉的是 AIV → CCU 那条 NotifyWait 边（NotifyId=13224、sqe_type=7），CCU 侧 mission 5 执行失败、notify 再也不会来，四卡随之卡在同一个 WaitGroup（sem[337]、mask 0x000e）。Run#2 那条 AIC 访问 GM 越界（errcode=264）同理是卡死之后的连带表现，顺着它去查算子实现会查空。到这一步「不是模型问题、不是算子算错」已经成立 —— 但也意味着本页的视图用尽了。",
        nums: ["断点在边上，不在盒子里", "missionId=5 · 四次一致", "errcode=264 是连带"],
        fix: [] },

      { n: 5, layer: "根因层 · 在观测页", short: "根因",
        eventId: "mc2-root",
        // 本页没有可照亮的实页元素：整网图是对照组，模型语义层从头到尾没有异常。
        // target 返回 null 时聚光灯退回画布中央浮框，只展示这段叙事 —— 这正是
        // 想要的效果：「这一步在这里看不到」本身就是要传达的信息。
        target: function () { return null; },
        prep: function () { openObserver("mc2-root"); },
        body: "根因不在模型结构里，整网图这次帮不上忙 —— 已为你打开 MC2 配置关系观测页。那里的计算血缘会画出这条边：model / FX 两层全绿，GE 层是 capture 出来的 15 个 graph，它们用一条红色虚线一起指向 Runtime 层同一个 aclOpExecutor。这是 N:1 的错误复用，不是编译变换。executor 内部的 CCU mission / notify id 被后来的 capture 覆盖，replay 时第一个 graph 就等在了错误的 notify 上。",
        nums: ["15 graph → 1 executor", "v15 触发 / v18 不触发", "exit 139"],
        fix: [0, 2] },

      { n: 6, layer: "配置层 · 在观测页", short: "配置耦合",
        eventId: "mc2-config",
        target: function () { return null; },
        prep: function () { openObserver("mc2-config"); },
        body: "观测页右下角的「执行配置」四项就是这条链的开关组：enable_mc2、mc2_comm_mode、cudagraph_mode、capture 档数。把 capture 档数从 15 调到 1，或把 comm_mode 切到 ai_cpu，判定条会当场从「触发」翻成「不触发」—— 同一份代码、同一批卡，配置一改就完全不复现。这也是它在别处难以复现的原因。",
        nums: ["decode 7 + prefill 8 = 15", "四项缺一不触发", "TP=4 / EP=4 共卡"],
        fix: [3, 4] },

      { n: 7, layer: "修复层 · 四个责任方", short: "修复",
        eventId: "mc2-fix",
        target: function () { return null; },
        prep: function () { /* 只看右侧修改建议列，不需要照亮实页元素 */ },
        body: "右侧四条修复分别落在四个责任方：opapi 侧改成每次返回独立 executor 是根因修复，无性能损失；HCCL 侧把 CCU mission 绑到 model_id 做兜底；torch_npu 侧确认 graph capture 不共享全局 executor 缓存；配置侧切 AICPU 只是驱动修复到位前的临时规避，要付 5–15% 吞吐。修复后用 v15 验证不再触发、v18 验证功能正常。",
        nums: ["根因在 opapi 层", "AICPU 规避 -5~15%", "v15 验触发 / v18 验功能"],
        fix: [0, 1, 2, 3] },
    ],
  };
  var els = null;        // 缓存构建好的外壳 DOM（与问题无关，只建一次）
  var activeKey = null;  // 当前问题的 caseKey
  var cur = null;        // = CASES[activeKey]
  var open = false;
  var idx = 0;
  var raf = 0;
  var keyHandler = null;
  var guideOn = true; // 名片「关闭/打开定位链指引」：只隐藏遮罩/光洞/步进导轨/标注气泡，名片本身与问题定位保留

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

  /* build() 只搭「与问题无关」的外壳（遮板/光框/名片骨架/工具行/气泡/修改建议容器），全页只建一次；
     名片文案、步进导轨、修改建议列表这些随问题变化的部分由 applyCase() 每次进入时重填。 */
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

    // 顶部名片（文案由 applyCase 填）
    var card = h("div", "tw-spot__card");
    var kicker = h("div", "tw-spot__card-kicker");
    var kickerText = document.createTextNode("");
    kicker.appendChild(kickerText);
    var sev = h("span", "tw-spot__sev"); kicker.appendChild(sev);
    var title = h("div", "tw-spot__card-title");
    var tags = h("div", "tw-spot__card-tags");
    var actions = h("div", "tw-spot__card-actions");
    // 「定位链指引」开关：只控制遮罩/光洞/步进导轨/标注气泡与修改建议联动高亮，
    // 不等同「详情」左侧的 ✕（那个会退出整个问题定位，见 exit()）。
    var guideToggleBtn = h("button", "tw-spot__guide-toggle");
    guideToggleBtn.type = "button";
    guideToggleBtn.setAttribute("role", "switch");
    guideToggleBtn.setAttribute("aria-checked", "true");
    guideToggleBtn.setAttribute("aria-label", "定位链指引");
    var guideToggleTrack = h("span", "tw-spot__guide-track");
    guideToggleTrack.setAttribute("aria-hidden", "true");
    guideToggleTrack.appendChild(h("span", "tw-spot__guide-thumb"));
    guideToggleBtn.appendChild(guideToggleTrack);
    guideToggleBtn.appendChild(h("span", "tw-spot__guide-label", "定位链指引"));
    guideToggleBtn.addEventListener("click", toggleGuide);
    // 「到性能调优工具查看」→ 与抽屉抬头上同一个入口（#locateDrawerProfilingLink）；
    // 目前只有问题一（mem-oom）在性能调优工具里有对应视图，其余问题由 applyCase 藏掉
    var profilingLink = h("a", "btn btn-ghost btn-sm");
    profilingLink.target = "_blank"; profilingLink.rel = "noopener";
    profilingLink.textContent = "到性能调优工具查看"; profilingLink.style.fontSize = "11px";
    // 「详情」→ 该问题的定位链长文抽屉；没有长文的问题由 applyCase 把按钮藏掉
    var detailBtn = h("button", "btn btn-ghost btn-sm"); detailBtn.type = "button"; detailBtn.textContent = "详情"; detailBtn.style.fontSize = "11px";
    detailBtn.addEventListener("click", openDetail);
    var closeBtn = h("button", "pto-ide-frame__window-action"); closeBtn.type = "button"; closeBtn.title = "退出聚光灯，回到最新 step"; closeBtn.setAttribute("aria-label", "退出聚光灯"); closeBtn.innerHTML = "&#10005;";
    closeBtn.addEventListener("click", exit);
    actions.appendChild(guideToggleBtn); actions.appendChild(profilingLink); actions.appendChild(detailBtn); actions.appendChild(closeBtn);
    card.appendChild(kicker); card.appendChild(title); card.appendChild(tags); card.appendChild(actions);

    // 步进导轨（内容由 applyCase 填）
    var rail = h("div", "tw-spot__rail");

    // 证据标注气泡
    var callout = h("div", "tw-spot__callout");

    // 「修改建议」——聚光灯期间挤入页面栅格作第四栏（见 doOpen 的注入），常驻显示
    var fixes = h("aside", "wzh-col-spot-fixes");
    fixes.setAttribute("aria-label", "修改建议");
    var fhead = h("div", "tw-spot__fixes-head");
    fhead.appendChild(h("span", "tw-spot__fixes-title", "修改建议"));
    var fixesSub = h("span", "tw-spot__fixes-sub");
    fhead.appendChild(fixesSub);
    var fbody = h("div", "tw-spot__fixes-body");
    fixes.appendChild(fhead); fixes.appendChild(fbody);

    // 底部导航
    var nav = h("div", "tw-spot__nav");
    var prev = h("button", "tw-spot__nav-btn tw-spot__nav-btn--strong"); prev.type = "button"; prev.title = "上一步"; prev.innerHTML = "&#8249;";
    prev.addEventListener("click", function () { go(idx - 1); });
    var count = h("div", "tw-spot__nav-count");
    var next = h("button", "tw-spot__nav-btn tw-spot__nav-btn--strong"); next.type = "button"; next.title = "下一步"; next.innerHTML = "&#8250;";
    next.addEventListener("click", function () { go(idx + 1); });
    nav.appendChild(prev); nav.appendChild(count); nav.appendChild(next);

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
      rail: rail, stepEls: [], arrowEls: [], callout: callout,
      fixes: fixes, fixesSub: fixesSub, fixesBody: fbody, fixEls: [],
      nav: nav, prev: prev, next: next, count: count,
      guideToggleBtn: guideToggleBtn, detailBtn: detailBtn, profilingLink: profilingLink,
      kickerText: kickerText, sev: sev, title: title, tags: tags,
    };
    return els;
  }

  /* 把注册表里某个问题的内容灌进已建好的外壳：名片文案 + 步进导轨 + 修改建议列表。
     切换问题时整段重建导轨与修改建议（两者的条数、文案都不同，逐项 diff 得不偿失）。 */
  function applyCase(key) {
    var e = els, c = CASES[key];
    activeKey = key;
    cur = c;

    // 名片
    e.kickerText.nodeValue = c.meta.kicker;
    e.sev.textContent = c.meta.severity;
    e.title.textContent = c.meta.title;
    e.tags.innerHTML = "";
    (c.meta.tags || []).forEach(function (t) {
      var s = h("span", "tw-spot__tag"); s.textContent = t; e.tags.appendChild(s);
    });
    // 该问题的定位链长文还没落地时,「详情」按钮没有去处 —— 直接藏掉,不给死链
    e.detailBtn.hidden = !(window.hasLocateDrawer && window.hasLocateDrawer(key));
    // 「到性能调优工具查看」目标页与抽屉抬头入口保持一致,只对问题一(mem-oom)开放
    if (e.profilingLink) {
      var pBase = window.PTO_BASE_PREFIX || "../../";
      e.profilingLink.href = pBase + "Profiling_Insight_and_Tool/AI_Profiling_Tool/index_v3.html?issue=mem-oom&tab=memory";
      e.profilingLink.hidden = key !== "mem-oom";
    }

    // 步进导轨：相邻步之间插入箭头，强化「链路」观感；箭头随 go() 同步点亮已走过的一段
    e.rail.innerHTML = "";
    e.arrowEls = [];
    e.stepEls = c.steps.map(function (st, i) {
      if (i > 0) {
        var arrow = h("span", "tw-spot__rail-arrow", "&#8594;");
        arrow.setAttribute("aria-hidden", "true");
        e.rail.appendChild(arrow);
        e.arrowEls.push(arrow);
      }
      var s = h("div", "tw-spot__step");
      var num = h("span", "tw-spot__step-n"); num.textContent = st.n;
      var name = h("span", "tw-spot__step-name"); name.textContent = st.short || st.layer;
      s.appendChild(num); s.appendChild(name);
      s.addEventListener("click", function () { go(i); });
      e.rail.appendChild(s);
      return s;
    });

    // 修改建议
    e.fixesSub.textContent = c.fixes.length + " 处 · 按优先级";
    e.fixesBody.innerHTML = "";
    e.fixEls = c.fixes.map(function (fx, i) {
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
      e.fixesBody.appendChild(f);
      return f;
    });
  }

  function openDetail() {
    if (window.openLocateDrawer) window.openLocateDrawer(activeKey);
    else if (window.openProblemOneLocateDrawer) window.openProblemOneLocateDrawer();
  }

  /* 按当前问题决定「集群监控」右列的开合，一开场就定好，布局提前定型：
       · 有步骤要照亮它（问题一 ⑥ infra 扩散）→ 展开，到达那步时图表已在场；
       · 没有任何一步用到（问题二）→ 收起，把宽度让给聚光灯挤进来的「修改建议」第四栏。 */
  function syncInfraCol() {
    if (!cur) return;
    try {
      window.PtoTrainingTwinSideCols &&
        window.PtoTrainingTwinSideCols.setRightVisible(cur.infraCol !== false);
    } catch (x) {}
  }

  // ── 步导航 ────────────────────────────────────────────────────────────
  function go(i) {
    var STEPS = cur.steps;
    i = Math.max(0, Math.min(STEPS.length - 1, i));
    idx = i;
    var st = STEPS[i];
    var e = els;

    // 导轨状态
    e.stepEls.forEach(function (s, k) {
      s.classList.toggle("is-active", k === i);
      s.classList.toggle("is-done", k < i);
    });
    // 箭头：第 k 个箭头连接 step[k] → step[k+1]，走过去了才点亮
    e.arrowEls.forEach(function (a, k) { a.classList.toggle("is-done", k < i); });
    // 计数 + 前后按钮
    e.count.innerHTML = "<b>" + st.n + "</b> / " + STEPS.length;
    e.prev.disabled = i === 0;
    e.next.disabled = i === STEPS.length - 1;

    // 标注气泡内容
    var linkedFirst = st.fix.length ? cur.fixes[st.fix[0]] : null;
    var numsHtml = st.nums.map(function (n) { return '<span class="tw-spot__num"></span>'; }).join("");
    e.callout.innerHTML =
      '<div class="tw-spot__callout-head">' +
      '<span class="tw-spot__callout-n">' + st.n + '</span>' +
      '<span class="tw-spot__callout-layer"></span></div>' +
      '<div class="tw-spot__callout-body"></div>' +
      '<div class="tw-spot__callout-nums">' + numsHtml + '</div>' +
      '<div class="tw-spot__callout-fix"' + (linkedFirst ? '' : ' hidden') + '>' +
      '<b>→ 修复</b> <span class="tw-spot__callout-fix-txt"></span></div>' +
      '<div class="tw-spot__callout-impact"' + (st.eventId ? '' : ' hidden') + '>' +
      '<button type="button" class="tw-spot__callout-impact-btn">查看事件影响范围</button></div>';
    e.callout.querySelector(".tw-spot__callout-layer").textContent = st.layer;
    e.callout.querySelector(".tw-spot__callout-body").textContent = st.body;
    var numEls = e.callout.querySelectorAll(".tw-spot__num");
    st.nums.forEach(function (n, k) { if (numEls[k]) numEls[k].textContent = n; });
    if (linkedFirst) {
      e.callout.querySelector(".tw-spot__callout-fix-txt").textContent =
        linkedFirst.file + " · " + linkedFirst.title + (st.fix.length > 1 ? " 等 " + st.fix.length + " 处" : "");
    }
    var impactBtn = e.callout.querySelector(".tw-spot__callout-impact-btn");
    if (impactBtn && st.eventId) {
      impactBtn.addEventListener("click", function () {
        window.open("config-relation-observer.html?event=" + encodeURIComponent(st.eventId), "_blank", "noopener");
      });
    }

    // 右列修复高亮仅属于定位链指引；指引关闭时各项同权。
    e.fixes.classList.toggle("has-link", guideOn && st.fix.length > 0);
    e.fixEls.forEach(function (f, k) {
      f.classList.toggle("is-linked", guideOn && st.fix.indexOf(k) >= 0);
    });
    if (guideOn && st.fix.length && e.fixEls[st.fix[0]]) {
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
    if (!open || !els || !cur) return;
    var e = els, st = cur.steps[idx];
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
    var topR = 88, botR = 16; // 顶部名片(4~44) + 工具行(44~74)约 88px；导航已上移，底部只留边距
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

  function syncGuideControl() {
    if (!els || !els.guideToggleBtn) return;
    els.guideToggleBtn.setAttribute("aria-checked", guideOn ? "true" : "false");
    els.guideToggleBtn.title = guideOn
      ? "关闭定位链指引"
      : "开启定位链指引";
  }

  // 名片「定位链指引」开关：关闭遮罩/光洞/步进导轨/标注气泡，并取消修改建议关联高亮；
  // 名片本身、修改建议栏与当前问题定位仍保留。再开启时原地恢复到当前步。
  function setGuideVisible(v) {
    guideOn = v;
    if (els) {
      els.root.classList.toggle("is-guide-off", !v);
      syncGuideControl();
      if (!v) {
        els.fixes.classList.remove("has-link");
        els.fixEls.forEach(function (f) { f.classList.remove("is-linked"); });
      }
    }
    if (v) {
      if (!raf) raf = window.requestAnimationFrame(reflow);
      if (open) go(idx);
    } else {
      if (raf) { window.cancelAnimationFrame(raf); raf = 0; }
    }
  }
  function toggleGuide() { setGuideVisible(!guideOn); }

  // ── 开 / 关 ───────────────────────────────────────────────────────────
  function doOpen(caseKey) {
    if (!caseKey || !CASES[caseKey]) return false; // 注册表里没有的问题不开聚光灯
    build();
    // 已经开着但换了问题：原地把名片/导轨/修改建议换成新问题的，回到第 1 步
    if (open && caseKey !== activeKey) {
      applyCase(caseKey);
      syncInfraCol();       // 换问题也要跟着换 infra 栏开合
      idx = 0;
      setGuideVisible(true);
      go(0);
      return true;
    }
    if (open) { setGuideVisible(true); return true; }
    applyCase(caseKey);
    open = true;
    idx = 0;
    guideOn = true;
    els.root.classList.remove("is-guide-off");
    syncGuideControl();
    syncInfraCol();
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
      else if (guideOn && ev.key === "ArrowRight") { ev.preventDefault(); ev.stopPropagation(); go(idx + 1); }
      else if (guideOn && ev.key === "ArrowLeft") { ev.preventDefault(); ev.stopPropagation(); go(idx - 1); }
    };
    document.addEventListener("keydown", keyHandler, true);
    return true;
  }

  function doClose() {
    if (!open) return;
    open = false;
    if (raf) { window.cancelAnimationFrame(raf); raf = 0; }
    guideOn = true;
    if (els) {
      els.root.classList.remove("is-open");
      els.root.classList.remove("is-guide-off");
      syncGuideControl();
      // 移除挤入的「修改建议」整高右列，workarea 复原为纵向单块
      var g = els.fixes && els.fixes.parentNode;
      if (g) { g.removeChild(els.fixes); g.classList && g.classList.remove("is-spot-fixes"); window.dispatchEvent(new Event("resize")); }
    }
    activeKey = null;
    cur = null;
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
    // 当前问题 / 有聚光灯的问题清单（调用方据此决定要不要给某个问题挂"进定位链"的入口）
    activeCase: function () { return activeKey; },
    hasCase: function (caseKey) { return !!CASES[caseKey]; },
  };
})();
