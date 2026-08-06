/* ══════════════════════════════════════════════════════════════════════════
   时光全景 · 顶部浮窗  (window.PtoTrainingPanorama)
   ─────────────────────────────────────────────────────────────────────────
   点击顶栏「训练进度」部件(#progressTopbar)——没点中问题 1/2 标记点的那一击——
   落下一张贴住视口顶部(8px)的全宽浮窗:一条与顶栏进度条同口径的 step 轴，把整段训练里
   已发生 / 计划中的关键事件全部铺开:

     异常(P0 事故) · 告警(P1 苗头) · 产出(ckpt/报告) · 消息推送(值班群/邮件) · 事件(生命周期)

   已发生的事件带时间戳、可点击:命中「问题一/问题二」的走聚光灯定位链
   (PtoTwinGraphBridge.activateProblemLens)，其余直接把时光机拖到那一步；
   计划中的事件给「预计」时间、不可点。× / 点面板外 / ESC 关闭。

   ── 时钟口径 ───────────────────────────────────────────────────────────
   时间戳 = 训练启动时刻 + step × TIME_MACHINE_STEP_SECONDS(8.5s，见
   js/training-run-twin.js)。启动时刻取 js/training-log-drawer.js 里
   "training loop started" 那行(2026-07-16 08:08:10)。这样浮窗里的时间戳、
   「已训练时长」与顶栏进度条读数三者同源，不会自相矛盾。

   ── 排版 ───────────────────────────────────────────────────────────────
   当前 step 只走到 21000/120000，已发生的事件全挤在轴的左侧 ~17%；因此标签不
   强行贴在标记点正下方，而是按「车道 + 向右让位」贪心排布，用引出线接回各自
   的标记点(和设计稿一致)。车道高度/宽度见下面的 LAYOUT。
   ══════════════════════════════════════════════════════════════════════════ */
(function () {
  "use strict";

  // 训练启动墙钟(见文件头「时钟口径」)
  const START_MS = new Date(2026, 6, 16, 8, 8, 10).getTime();

  const TYPES = {
    fault:    { label: "异常",   color: "#dc2626", shape: "dot" },
    warn:     { label: "告警",   color: "#d97706", shape: "dot" },
    artifact: { label: "产出",   color: "#10b981", shape: "diamond" },
    notify:   { label: "消息推送", color: "#3b6fe0", shape: "dot" },
    event:    { label: "事件",   color: "#8b5cf6", shape: "dot" },
  };
  const TYPE_ORDER = ["fault", "warn", "artifact", "notify", "event"];

  /* 关键事件表。step 与 js/training-run-twin.js 的 diagnosisMarkers、artifacts、
     js/training-log-drawer.js 的日志脚本对齐(问题一 12003 显存 OOM、问题二 15203
     Router 溢出、ckpt 每 10000 步、HCCS 掉链路 ~20000)。
     相邻事件至少隔 ~1200 step，否则在 120000 步的轴上两个标记点会叠在一起。
     lens: 有定位链的问题，点击进聚光灯；problem: 顶栏进度条上的问题编号。 */
  const EVENTS = [
    { step: 0,      type: "event",    name: "训练任务启动", detail: "2048 卡 · TP4/PP8/EP64/CP2 · 总步数 120000" },
    { step: 2400,   type: "event",    name: "warmup 结束，进入余弦衰减", detail: "lr 5.0e-6 → 1.2e-4" },
    { step: 6200,   type: "warn",     name: "straggler:node-37 落后 1.8×", detail: "单步落后 1.8×，已纳入慢节点观察" },
    { step: 8500,   type: "warn",     name: "q_proj FP8 溢出 → grad_norm 抬升", detail: "layer 33 · 3.2% token 超 E4M3 max(448)，grad_norm 1.63 → 2.87" },
    { step: 10000,  type: "artifact", name: "ckpt-10000 落盘", detail: "118.4GB · 64 分片(TP4×PP8×CP2)" },
    { step: 12003,  type: "fault",    name: "显存 OOM 中断", detail: "激活值占 56.6%(36.2GB) + 碎片率 83% → 峰值触顶 64GB", lens: "mem-oom", problem: "1" },
    { step: 13500,  type: "notify",   name: "OOM 熔断通知已推送值班群", detail: "WeLink 值班群 + 邮件 · 附现场 dump 与 ckpt 路径" },
    { step: 15203,  type: "fault",    name: "Router 溢出 → NaN + all-to-all 死锁", detail: "layer 38 router logits 512.7 > E4M3 max，98.3% token 塌到 expert 193", lens: "moe-a2a", problem: "2" },
    { step: 16600,  type: "event",    name: "回滚 step_15200 后恢复训练", detail: "跳过损坏的 optimizer state，mfu 0.40 → 0.58 恢复" },
    { step: 18600,  type: "warn",     name: "HCCS lane5 链路抖动 → 回退 RoCE", detail: "node002 NPU3 · pp_group_2 走慢路径，mfu 0.586 → 0.312" },
    { step: 20000,  type: "artifact", name: "ckpt-20000 落盘", detail: "118.6GB · 64 分片 · SHA256 已校验" },

    // ── 以下为计划中(step > liveStep):只给「预计」时间，不可跳转 ──
    { step: 30000,  type: "artifact", name: "ckpt-30000 计划落盘", detail: "按每 10000 步一次的落盘策略" },
    { step: 45000,  type: "event",    name: "阶段性验证集评估", detail: "eval 套件 · 约占用 1 个 step 周期的 12×" },
    { step: 60000,  type: "artifact", name: "中期权重 + 评估报告", detail: "半程权重冻结一份，附收敛/精度报告" },
    { step: 90000,  type: "notify",   name: "训练周报推送", detail: "进度 / 稳定性 / 资源占用汇总 → 项目群" },
    { step: 120000, type: "artifact", name: "最终权重交付 · 训练结束", detail: "释放 2048 卡资源，产出最终 ckpt 与训练总结" },
  ];

  const LAYOUT = {
    LABEL_W: 152,   // 标签卡固定宽度(统一宽度才好按车道让位)
    LANE_H: 66,     // 车道高度:类型药丸 + 两行标题 + 时间戳
    LANE_GAP: 8,
    GAP_X: 10,      // 同车道相邻标签的最小水平间隙
    MAX_LANES: 5,
    ANCHOR_BIAS: 8, // 标签左边缘比标记点略靠左，引出线接得更自然
  };

  const $ = (id) => document.getElementById(id);
  let root = null;      // .tw-pano
  let els = null;       // 缓存的节点引用
  let activeTypes = null;  // null = 全部；否则是 Set
  let placed = [];      // 最近一次排布结果:{ ev, dot, label, lead }
  let resizeRaf = 0;

  // ── 数据/格式化 ──────────────────────────────────────────────────────
  function bridge() { return window.PtoTrainingTimeMachine || null; }

  function ctx() {
    const b = bridge();
    if (b) return b.getState();
    // 兜底(理论上不会走到):训练主脚本还没加载完
    return { step: 0, liveStep: 0, totalSteps: 120000, stepsPerEpoch: 2000, stepSeconds: 8.5 };
  }

  function pad(n) { return String(n).padStart(2, "0"); }

  function stampAt(step, stepSeconds, opts) {
    const d = new Date(START_MS + step * stepSeconds * 1000);
    const date = `${d.getFullYear()}/${pad(d.getMonth() + 1)}/${pad(d.getDate())}`;
    const time = `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
    if (opts && opts.planned) return `预计 ${pad(d.getMonth() + 1)}/${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
    return `${date} ${time}`;
  }

  function fmtDurationCN(seconds) {
    const s = Math.max(0, Math.round(seconds));
    const days = Math.floor(s / 86400);
    const hours = Math.floor((s % 86400) / 3600);
    const mins = Math.floor((s % 3600) / 60);
    if (days > 0) return `${days}天${hours}小时${mins}分`;
    if (hours > 0) return `${hours}小时${mins}分`;
    return `${mins}分`;
  }

  // ── DOM 构建(只建一次) ───────────────────────────────────────────────
  function h(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }

  function build() {
    root = h("div", "tw-pano");
    root.id = "twPanorama";
    root.hidden = true;

    const backdrop = h("div", "tw-pano__backdrop");
    backdrop.addEventListener("click", close);

    const panel = h("section", "tw-pano__panel");
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-label", "时光全景 · 训练关键事件");

    const head = h("div", "tw-pano__head");
    const kpis = h("div", "tw-pano__kpis");
    const closeBtn = h("button", "pto-ide-frame__window-action");
    closeBtn.type = "button";
    closeBtn.title = "关闭时光全景";
    closeBtn.setAttribute("aria-label", "关闭时光全景");
    closeBtn.innerHTML = '<svg class="pto-ide-frame__window-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M18 6 6 18"></path><path d="M6 6l12 12"></path></svg>';
    closeBtn.addEventListener("click", close);
    head.appendChild(kpis);
    head.appendChild(closeBtn);

    const filters = h("div", "tw-pano__filters");

    const stage = h("div", "tw-pano__stage");
    const ticks = h("div", "tw-pano__ticks");
    const axis = h("div", "tw-pano__axis");
    const live = h("div", "tw-pano__axis-fill");
    live.style.opacity = "0.45";
    const fill = h("div", "tw-pano__axis-fill");
    const now = h("div", "tw-pano__now", "<span></span>");
    axis.appendChild(live);
    axis.appendChild(fill);
    axis.appendChild(now);
    const field = h("div", "tw-pano__field");
    const leads = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    leads.setAttribute("class", "tw-pano__leads");
    field.appendChild(leads);
    stage.appendChild(ticks);
    stage.appendChild(axis);
    stage.appendChild(field);

    panel.appendChild(head);
    panel.appendChild(filters);
    panel.appendChild(stage);
    root.appendChild(backdrop);
    root.appendChild(panel);
    document.body.appendChild(root);

    els = { panel, kpis, filters, stage, ticks, axis, fill, live, now, field, leads };
  }

  // ── 渲染 ─────────────────────────────────────────────────────────────
  function renderKpis(c) {
    const pct = c.totalSteps ? (c.step / c.totalSteps) * 100 : 0;
    const epoch = Math.floor(c.step / c.stepsPerEpoch) + 1;
    const totalEpochs = Math.ceil(c.totalSteps / c.stepsPerEpoch);
    const items = [
      { k: "总step", v: `${c.step.toLocaleString()}<span class="tw-pano__kpi-total">/${c.totalSteps.toLocaleString()}</span>`, num: true },
      { k: "总epoch", v: `${epoch}<span class="tw-pano__kpi-total">/${totalEpochs}</span>`, num: true },
      { k: "已训练时长", v: fmtDurationCN(c.step * c.stepSeconds) },
      { k: "预计剩余训练时长", v: fmtDurationCN((c.totalSteps - c.step) * c.stepSeconds) },
      { k: "总进度", v: `${pct.toFixed(1)}%`, num: true },
    ];
    els.kpis.innerHTML = items.map((it) =>
      `<div class="tw-pano__kpi${it.num ? " tw-pano__kpi--num" : ""}"><span>${it.k}</span><strong>${it.v}</strong></div>`
    ).join("");
  }

  function renderFilters() {
    els.filters.innerHTML = "";
    const counts = {};
    EVENTS.forEach((e) => { counts[e.type] = (counts[e.type] || 0) + 1; });
    TYPE_ORDER.forEach((type) => {
      const meta = TYPES[type];
      if (!counts[type]) return;
      const chip = h("button", "tw-pano__chip", `<i></i>${meta.label} <b>${counts[type]}</b>`);
      chip.type = "button";
      chip.style.setProperty("--c", meta.color);
      chip.dataset.panoType = type;
      chip.dataset.shape = meta.shape;
      chip.setAttribute("aria-pressed", "true");
      chip.addEventListener("click", () => toggleType(type));
      els.filters.appendChild(chip);
    });
    const c = ctx();
    const done = EVENTS.filter((e) => e.step <= c.liveStep).length;
    els.filters.appendChild(h("span", "tw-pano__filters-hint",
      `共 ${EVENTS.length} 个关键事件 · 已发生 ${done} / 计划中 ${EVENTS.length - done}`));
    syncFilterChips();
  }

  function syncFilterChips() {
    els.filters.querySelectorAll("[data-pano-type]").forEach((chip) => {
      const on = !activeTypes || activeTypes.has(chip.dataset.panoType);
      chip.classList.toggle("is-on", !!activeTypes && on);
      chip.classList.toggle("is-off", !!activeTypes && !on);
      chip.setAttribute("aria-pressed", String(on));
    });
  }

  function toggleType(type) {
    if (!activeTypes) activeTypes = new Set([type]);          // 首次点击 = 只看这一类
    else if (activeTypes.has(type) && activeTypes.size === 1) activeTypes = null;  // 再点一次放回全部
    else if (activeTypes.has(type)) activeTypes.delete(type);
    else activeTypes.add(type);
    if (activeTypes && !activeTypes.size) activeTypes = null;
    syncFilterChips();
    applyFilter();
  }

  function applyFilter() {
    placed.forEach((p) => {
      const dim = !!activeTypes && !activeTypes.has(p.ev.type);
      p.dot.classList.toggle("is-dimmed", dim);
      p.label.classList.toggle("is-dimmed", dim);
      p.lead.classList.toggle("is-dimmed", dim);
    });
  }

  function renderTicks(c) {
    const marks = [0, 0.25, 0.5, 0.75, 1];
    els.ticks.innerHTML = marks.map((m) => {
      const step = Math.round(c.totalSteps * m);
      return `<div class="tw-pano__tick" style="left:${(m * 100).toFixed(2)}%">${step.toLocaleString()}</div>`;
    }).join("");
  }

  function renderAxis(c) {
    const pct = c.totalSteps ? (c.step / c.totalSteps) * 100 : 0;
    const livePct = c.totalSteps ? (c.liveStep / c.totalSteps) * 100 : 0;
    els.fill.style.width = `${pct.toFixed(2)}%`;
    els.live.style.width = `${livePct.toFixed(2)}%`;
    els.now.style.left = `${pct.toFixed(2)}%`;
    els.now.classList.toggle("is-near-start", pct < 8);
    els.now.classList.toggle("is-near-end", pct > 92);
    const replaying = c.step < c.liveStep;
    els.now.querySelector("span").textContent =
      `${replaying ? "回放" : "现在"} · step ${c.step.toLocaleString()}`;
  }

  /* 事件标记点 + 标签的排布:按 step 升序贪心放车道。
     每个标签优先落在自己标记点正下方(x = markerX - ANCHOR_BIAS)，被同车道左邻挤住时
     整体向右让位，代价最小(让位距离最短、车道尽量靠上)的方案胜出；接不上就开新车道。 */
  /* 车道数上限:浮窗不允许出现纵向滚动条,所以能开几条车道由「顶栏到视口底部还剩多少」决定。
     减掉面板内边距、指标行、筛选行、刻度与轴之后剩下的高度,就是标签场的预算。
     调用前 head/filters/ticks/axis 都已渲染完,offsetHeight 可直接量。 */
  function laneBudget() {
    const { LANE_H, LANE_GAP } = LAYOUT;
    const cs = getComputedStyle(els.panel);
    const padY = parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom);
    const gapY = parseFloat(cs.rowGap) || 0;
    const maxH = parseFloat(els.panel.style.maxHeight) || (window.innerHeight - 80);
    const used = padY + gapY * 2                                   // 面板:指标行 / 筛选行 / 轴区 三段之间的间距
      + els.panel.querySelector(".tw-pano__head").offsetHeight
      + els.filters.offsetHeight
      + els.ticks.offsetHeight + 2 + els.axis.offsetHeight + 2;    // 刻度 margin-bottom / 标签场 margin-top
    const budget = maxH - used - 14;                               // 14 = 标签场底部留白
    return Math.max(1, Math.floor((budget - LANE_H) / (LANE_H + LANE_GAP)) + 1);
  }

  function renderEvents(c) {
    const W = els.stage.clientWidth || 1;
    const { LABEL_W, LANE_H, LANE_GAP, GAP_X, ANCHOR_BIAS } = LAYOUT;
    const MAX_LANES = Math.max(1, Math.min(LAYOUT.MAX_LANES, laneBudget()));

    els.field.querySelectorAll(".tw-pano__label").forEach((el) => el.remove());
    els.axis.querySelectorAll(".tw-pano__dot").forEach((el) => el.remove());
    els.leads.innerHTML = "";
    placed = [];

    const laneRight = [];
    const list = EVENTS.slice().sort((a, b) => a.step - b.step);
    let maxLane = 0;

    list.forEach((ev) => {
      const meta = TYPES[ev.type] || TYPES.event;
      const planned = ev.step > c.liveStep;
      const markerX = Math.max(0, Math.min(W, (ev.step / c.totalSteps) * W));
      const anchorX = Math.max(0, Math.min(W - LABEL_W, markerX - ANCHOR_BIAS));

      let best = null;
      for (let i = 0; i < MAX_LANES; i += 1) {
        const right = laneRight[i] == null ? -Infinity : laneRight[i];
        const x = Math.min(W - LABEL_W, Math.max(anchorX, right + GAP_X));
        if (right !== -Infinity && x < right + GAP_X) continue;   // 这条车道右端已经顶到面板边缘
        const cost = Math.abs(x - anchorX) + i * 0.5;             // 同等让位距离时优先靠上的车道
        if (!best || cost < best.cost) best = { lane: i, x, cost };
        if (cost === i * 0.5) break;                              // 不用让位:就它了
      }
      if (!best) best = { lane: MAX_LANES - 1, x: Math.max(0, W - LABEL_W) };
      laneRight[best.lane] = best.x + LABEL_W;
      maxLane = Math.max(maxLane, best.lane);
      const y = best.lane * (LANE_H + LANE_GAP);

      // 轴上的标记点
      const dot = h("div", "tw-pano__dot" + (planned ? " is-planned" : ""));
      dot.style.setProperty("--c", meta.color);
      dot.style.left = `${markerX.toFixed(1)}px`;
      dot.dataset.shape = meta.shape;
      dot.title = `${meta.label} · step ${ev.step.toLocaleString()} · ${ev.name}`;
      els.axis.appendChild(dot);

      // 引出线:标记点垂下来，再用一段缓和的 S 弯接到标签左上角
      const tx = best.x + 10;
      const lead = document.createElementNS("http://www.w3.org/2000/svg", "path");
      lead.setAttribute("class", "tw-pano__lead");
      lead.setAttribute("d",
        `M ${markerX.toFixed(1)} 0 L ${markerX.toFixed(1)} ${(y + 2).toFixed(1)} ` +
        `C ${markerX.toFixed(1)} ${(y + 14).toFixed(1)} ${tx.toFixed(1)} ${(y + 2).toFixed(1)} ${tx.toFixed(1)} ${(y + 14).toFixed(1)}`);
      lead.style.setProperty("--c", meta.color);
      els.leads.appendChild(lead);

      // 标签卡
      const label = h("button", "tw-pano__label" + (planned ? " is-planned" : "") + (ev.problem ? " is-problem" : ""));
      label.type = "button";
      label.style.setProperty("--c", meta.color);
      label.style.left = `${best.x.toFixed(1)}px`;
      label.style.top = `${(y + 14).toFixed(1)}px`;
      label.style.width = `${LABEL_W}px`;
      label.title = `${meta.label} · step ${ev.step.toLocaleString()}\n${ev.name}\n${ev.detail || ""}`;
      const tag = h("span", "tw-pano__tag", meta.label);
      if (ev.problem) tag.dataset.problem = ev.problem;
      label.appendChild(tag);
      label.appendChild(h("span", "tw-pano__name", ev.name));
      label.appendChild(h("span", "tw-pano__time", stampAt(ev.step, c.stepSeconds, { planned })));
      // 计划中的事件不可跳转,但仍要能悬浮联动高亮 —— 因此用 aria-disabled 而不是 disabled
      // (disabled 的 button 在 Chrome 里连 mouseenter 都不派发)。
      if (planned) label.setAttribute("aria-disabled", "true");
      else label.addEventListener("click", () => jumpTo(ev));
      els.field.appendChild(label);

      const entry = { ev, dot, label, lead };
      placed.push(entry);
      const hot = (on) => {
        dot.classList.toggle("is-hot", on);
        lead.classList.toggle("is-hot", on);
      };
      label.addEventListener("mouseenter", () => hot(true));
      label.addEventListener("mouseleave", () => hot(false));
      label.addEventListener("focus", () => hot(true));
      label.addEventListener("blur", () => hot(false));
      dot.addEventListener("mouseenter", () => { hot(true); label.classList.add("is-hot"); });
      dot.addEventListener("mouseleave", () => { hot(false); label.classList.remove("is-hot"); });
      if (!planned) dot.addEventListener("click", () => jumpTo(ev));
    });

    els.field.style.height = `${maxLane * (LANE_H + LANE_GAP) + LANE_H + 14}px`;
    applyFilter();
  }

  function jumpTo(ev) {
    const b = bridge();
    close();
    if (!b) return;
    // 有定位链的问题(问题一/二):走聚光灯,它内部会把时光机拖到事故步并聚焦整网图
    if (ev.lens && typeof b.activateProblemLens === "function") b.activateProblemLens(ev.lens);
    else b.gotoStep(ev.step);
  }

  function render() {
    const c = ctx();
    renderKpis(c);
    renderFilters();
    renderTicks(c);
    renderAxis(c);
    renderEvents(c);
  }

  // ── 开关 ─────────────────────────────────────────────────────────────
  // 面板贴视口顶部 8px:全景是"看一眼时间轴"的浮层,压在顶栏之上比挂在顶栏下方更靠近视线落点,
  // 也给下方的标签场多留一条车道的高度(顶栏本身被遮罩压住,不需要留出来)。
  function position() {
    const top = 8;
    els.panel.style.top = `${top}px`;
    // 不给滚动条:高度上限交给 laneBudget() 换算成车道数,内容自己收进来(见 renderEvents)
    els.panel.style.maxHeight = `${Math.max(220, window.innerHeight - top - 20)}px`;
  }

  function isOpen() { return !!root && !root.hidden; }

  function open() {
    if (!root) build();
    if (isOpen()) return;
    root.hidden = false;
    position();
    render();
    document.addEventListener("keydown", onKeydown, true);
    window.addEventListener("resize", onResize);
    $("progressTopbar")?.setAttribute("aria-expanded", "true");
  }

  function close() {
    if (!isOpen()) return;
    root.hidden = true;
    document.removeEventListener("keydown", onKeydown, true);
    window.removeEventListener("resize", onResize);
    $("progressTopbar")?.setAttribute("aria-expanded", "false");
  }

  function toggle() { if (isOpen()) close(); else open(); }

  function onKeydown(e) {
    if (e.key === "Escape") { e.stopPropagation(); close(); }
  }

  function onResize() {
    if (resizeRaf) cancelAnimationFrame(resizeRaf);
    resizeRaf = requestAnimationFrame(() => {
      resizeRaf = 0;
      if (!isOpen()) return;
      position();
      renderEvents(ctx());
    });
  }

  // 顶栏部件上「轨道以外」的一击(step 计数 / 已训练时长 / 空白处)也开全景;
  // 轨道上的一击要先和「拖动回放」区分开,由 js/training-run-twin.js 的 bindTimeMachine 分发。
  function bindTopbar() {
    const widget = $("progressTopbar");
    if (!widget) return;
    widget.setAttribute("role", "button");
    widget.setAttribute("aria-expanded", "false");
    /* 拖块/轨道上按下、松手时指针已经飘出轨道(轨道只有 5px 高)的话,click 会派发到两者的
       共同祖先——也就是本部件身上。只看 click 的 target 会把一次回放拖动误当成"点了部件",
       因此用捕获阶段记住这一击是从哪儿按下去的。 */
    let pressedInTrack = false;
    widget.addEventListener("pointerdown", (e) => {
      pressedInTrack = !!e.target.closest("#progressTrack");
    }, true);
    widget.addEventListener("click", (e) => {
      const fromTrack = pressedInTrack || !!e.target.closest("#progressTrack");
      pressedInTrack = false;
      if (fromTrack) return;   // 轨道那一击的开合由 bindTimeMachine 按拖动阈值分发
      toggle();
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", bindTopbar);
  else bindTopbar();

  window.PtoTrainingPanorama = { open, close, toggle, isOpen, events: EVENTS };
})();
