/* ══════════════════════════════════════════════════════════════════════════
   MC2 运行监控 · 页面数据与渲染（window.Mc2Monitoring）
   ------------------------------------------------------------------------
   本页与 training-monitoring-v2.html 共用外壳，但**不复用 training-run-twin.js**：
   那份 410KB 是 openPangu 训练场景的数据源（46 层 / 2048 卡 / loss·grad_norm /
   120k step 时光机），与 vLLM 推理故障没有一个量对得上，接进来只会把两套口径搅在
   一起。这里只提供本案需要的四块取证 + 聚光灯依赖的那几个页面级开关。

   全部数字取自 history.md，来源逐条标在常量旁。凡是 history.md 没记的（Hunyuan V3
   的层数、专家数），只用于给整网图一个可看的骨架，不参与任何结论。
   ══════════════════════════════════════════════════════════════════════════ */
(function (global) {
  "use strict";

  const $ = (id) => document.getElementById(id);

  /* ── 事实表（history.md）───────────────────────────────────────────────── */

  // §三 Run#4「vLLM 侧渐进式恶化」（server_mxfp8_all_withmc2.log）
  const FORWARD_TIME = [
    { at: "14:07:00", value: 1.97 },
    { at: "14:07:20", value: 21.9 },
    { at: "14:08:47", value: 107 },
    { at: "14:09:48", value: 168 },
  ];

  // §三 Run#4「4 rank 故障时序」（debug_plog）。注意 order 与 culprit 是两件事。
  const RANKS = [
    { rank: 2, device: 2, at: "20:26:52.219", offsetMs: 0,    cqe: 0,     role: "victim", note: "首报" },
    { rank: 3, device: 3, at: "20:26:52.388", offsetMs: 169,  cqe: 0,     role: "victim", note: "故障传播" },
    { rank: 1, device: 1, at: "20:26:52.916", offsetMs: 697,  cqe: 32768, role: "origin", note: "真凶 · device 端 CQE 异常" },
    { rank: 0, device: 0, at: "20:26:53.709", offsetMs: 1490, cqe: 0,     role: "victim", note: "故障传播" },
  ];

  // §三 Run#1 device_error_proc_c.cc:847 —— 只有 CCU 报错
  const UNITS = [
    { id: "aic", name: "AIC", full: "AI Cube 计算单元", error: 0 },
    { id: "aiv", name: "AIV", full: "AI Vector 计算单元", error: 0 },
    { id: "ccu", name: "CCU", full: "通信单元", error: 1 },
  ];

  const UNIT_FACTS = [
    { label: "missionId", value: "5", tone: "danger", note: "四次复现全部落在这个 mission" },
    { label: "instrId", value: "698", note: "Run#1 · ccu_device_error_proc.cc:41" },
    { label: "WaitGroup", value: "sem[337] · mask[0x000e] · rankIds[1,2,3]", note: "四卡在同一个 group 上互等" },
    { label: "comm_flag", value: "2", note: "CheckAixErrorClassInFusionKernel：通信类算子异常" },
    { label: "Run#2 的 AIC 越界", value: "errcode=264", tone: "warn", note: "连带表现，不是另一个 bug" },
  ];

  /* plog：左栏原始日志、右栏「plog → 可读诊断」。tab 决定显示哪一侧的来源。
     Python 侧只有最后那一句 —— 这正是本案「报错看不懂」的痛点所在。 */
  const LOG_ROWS = [
    { tab: "plog",   time: "20:26:52.219", level: "ERROR", comp: "runtime", map: "sqe",
      msg: "task_recycle_cqrpt_base.cc:68  Task run failed, stream_id=61, sqe_type=7(notify wait), errType=0x20" },
    { tab: "plog",   time: "20:26:52.916", level: "ERROR", comp: "device", map: "cqe",
      msg: "device_error_proc.cc:1452  stream_id=61, taskType=14 (NOTIFY_WAIT), NotifyId=13224, cqeStatus=0x8000" },
    { tab: "plog",   time: "20:26:52.918", level: "ERROR", comp: "GE", map: "model",
      msg: "model_execute_task.cc:491  model execute task failed, model_id=55" },
    { tab: "plog",   time: "20:26:52.918", level: "ERROR", comp: "fusion", map: "kernel",
      msg: "fusion_task.cc:422  Fusion kernel execute failed, kernel_name=AllGatherMatmulV2_5cc364e0…_64_mix_aic" },
    { tab: "plog",   time: "20:26:52.919", level: "ERROR", comp: "acl", map: "sync",
      msg: "api_error.cc:1081  StreamSynchronize: failed, stream_id=61, timeout=-1ms" },
    { tab: "plog",   time: "13:57:15.105", level: "ERROR", comp: "device", map: "unit",
      msg: "device_error_proc_c.cc:847  aicError=0, aivError=0, ccuError=1     ← Run#1，仅 CCU 异常" },
    { tab: "plog",   time: "13:57:15.308", level: "ERROR", comp: "CCU", map: "waitgroup",
      msg: "task_exception_handler.cpp:859  InstrId[581]: Wait Group, sem[337], mask[0x000e], rankIds[1, 2, 3]" },
    { tab: "python", time: "20:26:52.9xx", level: "ERROR", comp: "vLLM", map: "sync",
      msg: "RuntimeError: EngineCore RPC timeout (507011) — APIServer 侧唯一可见的信息" },
  ];

  const LOG_MAPPING = [
    { id: "sqe", ascend: "sqe_type=7", torch: "notify wait 超时",
      desc: "任务在 notify 等待上失败，errType=0x20 是 sq sw status error。故障路径的入口。" },
    { id: "cqe", ascend: "cqeStatus=0x8000", torch: "device 端主动上报",
      desc: "四张卡里只有 rank 1 带这个状态 —— 判定真凶靠它，不靠报错先后。" },
    { id: "model", ascend: "model_id=55", torch: "第 N 个 cudagraph",
      desc: "GE 编译出的 model。15 个 graph 各有一个 model_id，共享同一个 aclOpExecutor。" },
    { id: "kernel", ascend: "AllGatherMatmulV2_…_64_mix_aic", torch: "npu_all_gather_matmul",
      desc: "MC2 融合算子。tilingKey=64、MIX_AIC，以 FUSION_KERNEL（taskType=107）执行。" },
    { id: "sync", ascend: "StreamSynchronize 507011", torch: "torch.npu.synchronize",
      desc: "同步超时。它是**结果**，不是原因 —— 照它去查 HCCL 会走反方向。" },
    { id: "unit", ascend: "ccuError=1", torch: "通信单元异常",
      desc: "aicError / aivError 均为 0，计算单元无辜，故障起源在 CCU。" },
    { id: "waitgroup", ascend: "Wait Group sem[337]", torch: "集合通信同步点",
      desc: "mask 0x000e = rank 1/2/3，四卡互等，谁都走不掉。" },
  ];

  /* 时间轴：故障当日 14:07（服务开始变慢）→ 20:26:53（四卡退出）。

     ⚠️ 只放**一个**标记。v2 的语汇里「一个编号红点 = 一个独立问题 = 一条独立
        定位链」（问题一显存 OOM / 问题二 Router 溢出，各有各的聚光灯）。本页只有
        一个问题，放两个点会让页面自称有两个问题，而它们点进去是同一条链。
        恶化不归这里管，理由有三：它是一段 3 分钟的区间而不是一个时刻；这段区间
        只占 6 小时轴长的 0.8%，画成带子也是一根看不见的发丝；而它本来就已经在
        推理指标卡上——那张卡自带 14:07:00→14:09:48 的轴，聚光灯第 ① 步取的也是
        它。同一件事在两处各画一遍，只会让人以为是两件事。 */
  const T0 = 14 * 3600 + 7 * 60;          // 14:07:00
  const T1 = 20 * 3600 + 26 * 60 + 53;    // 20:26:53 —— 服务在这里停了，之后没有数据
  // 轴末留一小段尾巴：否则标记压在轨道端点和滑块上，三者叠成一坨看不出是标记
  const T_AXIS_END = 20 * 3600 + 35 * 60;
  const MARKERS = [
    { at: T1, severity: "p0", num: "", label: "MC2 融合算子 CCU mission 污染",
      sub: "20:26:52 四卡同步退出 · 507011；14:07 起 forward_time 已线性恶化 85 倍" },
  ];

  const CASE_KEY = "mc2-executor";   // 与 js/mc2-spotlight.js 的 CASES 键一致

  /* ── 小工具 ───────────────────────────────────────────────────────────── */

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  }

  function hhmmss(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }

  /* ── 折线图 ─────────────────────────────────────────────────────────────
     按容器实测像素作图：1 SVG 单位 = 1 CSS 像素，preserveAspectRatio 用默认值。
     旧版是一张 240×84 的 viewBox 配 preserveAspectRatio="none"，塞进又窄又高的
     卡片格子后被纵向拉伸数倍 —— 线宽、圆点、斜率一起失真，而斜率恰恰是这张图要
     讲的事。改成实测尺寸后，格子怎么变图都不变形，字号与描边也都是真实尺寸。

     纵轴取 log10：4 个点跨两个数量级（1.97 → 168），线性轴会把前三点压成贴底的
     一条直线，「一路在涨」就会被读成「突然崩了」。刻度按十倍档画出并标注，读者
     不必猜轴；轴的选择理由写在卡片标题旁的「?」里。 */
  const SVG_NS = "http://www.w3.org/2000/svg";

  function svgNode(tag, attrs) {
    const node = document.createElementNS(SVG_NS, tag);
    Object.keys(attrs || {}).forEach((k) => node.setAttribute(k, attrs[k]));
    return node;
  }

  function fmtNum(v) {
    if (v >= 100) return String(Math.round(v));
    if (v >= 10) return String(Math.round(v * 10) / 10);
    return String(Math.round(v * 100) / 100);
  }

  /* 悬浮读数气泡：position:fixed 挂在 body 上，而不是塞进卡片里 —— 卡片是
     overflow:hidden 的窄格子，气泡贴边就会被切掉。与「?」说明气泡各用各的元素：
     两者都靠 #diagnosisTooltip 的话，鼠标从「?」滑到图上会互相抢内容。 */
  let chartTip = null;

  function ensureChartTip() {
    if (chartTip) return chartTip;
    chartTip = el("div", "mc2-chart-tip");
    chartTip.hidden = true;
    chartTip.append(
      el("span", "mc2-chart-tip__name"),
      el("span", "mc2-chart-tip__time"),
      el("span", "mc2-chart-tip__val"),
      el("span", "mc2-chart-tip__delta"),
    );
    document.body.appendChild(chartTip);
    return chartTip;
  }

  function showChartTip(spec, i, evt) {
    const tip = ensureChartTip();
    const v = spec.values[i];
    tip.querySelector(".mc2-chart-tip__name").textContent = spec.name || "";
    tip.querySelector(".mc2-chart-tip__time").textContent = (spec.labels && spec.labels[i]) || `#${i + 1}`;
    tip.querySelector(".mc2-chart-tip__val").textContent = `${fmtNum(v)}${spec.suffix || ""}`;
    // 相对上一轮的倍数：本案要看的是「每一轮都比上一轮慢」，绝对差值说明不了这件事
    tip.querySelector(".mc2-chart-tip__delta").textContent = i === 0
      ? "稳态基线"
      : `较上一轮 ×${fmtNum(v / spec.values[i - 1])}`;
    tip.hidden = false;
    const box = tip.getBoundingClientRect();
    const left = Math.min(evt.clientX + 14, window.innerWidth - box.width - 8);
    const top = Math.max(8, Math.min(evt.clientY - box.height - 12, window.innerHeight - box.height - 8));
    tip.style.left = `${Math.max(8, left)}px`;
    tip.style.top = `${top}px`;
  }

  function hideChartTip() {
    if (chartTip) chartTip.hidden = true;
  }

  function drawChart(host) {
    const spec = host.__mc2Chart;
    if (!spec) return;
    const W = Math.max(160, Math.round(host.clientWidth));
    const H = Math.max(96, Math.round(host.clientHeight));
    const PAD = { l: 38, r: 12, t: 16, b: 20 };
    const plotW = W - PAD.l - PAD.r;
    const plotH = H - PAD.t - PAD.b;

    const values = spec.values;
    const tf = (v) => Math.log10(Math.max(v, 1e-6));
    const rawLo = Math.min(...values.map(tf));
    const rawHi = Math.max(...values.map(tf));
    const margin = ((rawHi - rawLo) || 1) * 0.14;   // 端点不贴轴，最后一个点要留得下圆点
    const lo = rawLo - margin;
    const hi = rawHi + margin;
    const x = (i) => PAD.l + (i / Math.max(1, values.length - 1)) * plotW;
    const y = (v) => PAD.t + (1 - (tf(v) - lo) / (hi - lo)) * plotH;

    const svg = svgNode("svg", { class: "mc2-spark", width: W, height: H, viewBox: `0 0 ${W} ${H}` });

    // 纵轴刻度：值域内的十倍档（…1 / 10 / 100…）；不足两条时退回数据两端
    let ticks = [];
    for (let d = Math.ceil(lo); d <= Math.floor(hi); d += 1) ticks.push(Math.pow(10, d));
    if (ticks.length < 2) ticks = [Math.pow(10, rawLo), Math.pow(10, rawHi)];
    ticks.forEach((tv) => {
      const ty = y(tv).toFixed(1);
      svg.appendChild(svgNode("line", { class: "mc2-spark__grid", x1: PAD.l, x2: W - PAD.r, y1: ty, y2: ty }));
      const label = svgNode("text", {
        class: "mc2-spark__tick", x: PAD.l - 6, y: ty,
        "text-anchor": "end", "dominant-baseline": "middle",
      });
      label.textContent = fmtNum(tv);
      svg.appendChild(label);
    });

    // 轴线（纵 + 横）
    svg.appendChild(svgNode("line", { class: "mc2-spark__axis", x1: PAD.l, x2: PAD.l, y1: PAD.t, y2: PAD.t + plotH }));
    svg.appendChild(svgNode("line", { class: "mc2-spark__axis", x1: PAD.l, x2: W - PAD.r, y1: PAD.t + plotH, y2: PAD.t + plotH }));

    // 纵轴量纲（含刻度类型）：不写出来，读者会默认它是线性轴而误判斜率
    if (spec.unit) {
      const cap = svgNode("text", { class: "mc2-spark__cap", x: 2, y: PAD.t - 6 });
      cap.textContent = spec.unit;
      svg.appendChild(cap);
    }

    const points = values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`);

    // 面积 + 折线：面积让「越来越高」在余光里也成立
    svg.appendChild(svgNode("path", {
      class: "mc2-spark__area",
      d: `M ${x(0).toFixed(1)},${(PAD.t + plotH).toFixed(1)} L ${points.join(" L ")} L ${x(values.length - 1).toFixed(1)},${(PAD.t + plotH).toFixed(1)} Z`,
    }));
    svg.appendChild(svgNode("polyline", { class: "mc2-spark__line", points: points.join(" ") }));

    // 数据点不挂 <title>：原生 tooltip 会和下面这套悬浮气泡同时冒出来，读成两份
    values.forEach((v, i) => {
      const last = i === values.length - 1;
      svg.appendChild(svgNode("circle", {
        class: `mc2-spark__dot${last ? " is-last" : ""}`,
        cx: x(i).toFixed(1), cy: y(v).toFixed(1), r: last ? "3.4" : "2.4",
      }));
    });

    // 横轴：首末两个时刻（4 个点全标会在 ~150px 的卡宽里叠在一起）
    if (spec.labels && spec.labels.length) {
      const first = svgNode("text", { class: "mc2-spark__tick", x: PAD.l, y: H - 5, "text-anchor": "start" });
      first.textContent = spec.labels[0];
      svg.appendChild(first);
      const lastLabel = svgNode("text", { class: "mc2-spark__tick", x: W - PAD.r, y: H - 5, "text-anchor": "end" });
      lastLabel.textContent = spec.labels[spec.labels.length - 1];
      svg.appendChild(lastLabel);
    }

    /* 悬浮层：竖线（定位到哪一轮）+ 横线（把该值读回纵轴刻度）+ 焦点环。
       只有 4 个采样点，鼠标不必精确压在点上 —— 按 x 就近吸附到整轮，
       竖线因此永远落在真实数据点上，不会让人以为中间还有连续采样。 */
    const hover = svgNode("g", { class: "mc2-spark__hover", "aria-hidden": "true" });
    const vLine = svgNode("line", { class: "mc2-spark__hover-line", x1: 0, x2: 0, y1: PAD.t, y2: PAD.t + plotH });
    const hLine = svgNode("line", { class: "mc2-spark__hover-line is-h", x1: PAD.l, x2: 0, y1: 0, y2: 0 });
    const ring = svgNode("circle", { class: "mc2-spark__hover-dot", cx: 0, cy: 0, r: 4.6 });
    hover.append(vLine, hLine, ring);
    svg.appendChild(hover);

    // 捕获面：铺满绘图区的透明矩形，鼠标不用压在细线上也能触发
    const capture = svgNode("rect", {
      class: "mc2-spark__capture",
      x: PAD.l, y: PAD.t, width: Math.max(1, plotW), height: Math.max(1, plotH),
    });
    svg.appendChild(capture);

    const nearestIndex = (clientX) => {
      const rect = svg.getBoundingClientRect();
      const px = clientX - rect.left;
      const ratio = plotW > 0 ? (px - PAD.l) / plotW : 0;
      const idx = Math.round(ratio * (values.length - 1));
      return Math.max(0, Math.min(values.length - 1, idx));
    };

    const onMove = (evt) => {
      const i = nearestIndex(evt.clientX);
      const cx = x(i);
      const cy = y(values[i]);
      vLine.setAttribute("x1", cx.toFixed(1));
      vLine.setAttribute("x2", cx.toFixed(1));
      hLine.setAttribute("x2", cx.toFixed(1));
      hLine.setAttribute("y1", cy.toFixed(1));
      hLine.setAttribute("y2", cy.toFixed(1));
      ring.setAttribute("cx", cx.toFixed(1));
      ring.setAttribute("cy", cy.toFixed(1));
      svg.classList.add("is-hovering");
      showChartTip(spec, i, evt);
    };
    const onLeave = () => {
      svg.classList.remove("is-hovering");
      hideChartTip();
    };
    capture.addEventListener("mousemove", onMove);
    capture.addEventListener("mouseleave", onLeave);
    svg.addEventListener("mouseleave", onLeave);

    hideChartTip();   // 重画（改尺寸）时旧气泡的坐标已失效
    host.replaceChildren(svg);
  }

  // 卡片格子会随侧栏收放改变尺寸；实测作图意味着每次改尺寸都要重画一次
  const chartObserver = typeof ResizeObserver === "function"
    ? new ResizeObserver((entries) => entries.forEach((entry) => drawChart(entry.target)))
    : null;

  function mountChart(host, spec) {
    host.__mc2Chart = spec;
    drawChart(host);
    if (chartObserver) chartObserver.observe(host);
  }

  function metricCard(spec) {
    const card = el("div", "twin-accuracy-metric-card");
    card.dataset.accCard = spec.key;         // 聚光灯按 [data-acc-card] 取景
    const head = el("div", "twin-accuracy-metric-card__head");
    // 标题 + 「?」打包进 title-row，head 仍是两个直接子元素（两端对齐不受影响）
    const titleRow = el("div", "wzh-card-title-row");
    titleRow.appendChild(el("span", "twin-accuracy-metric-card__name", spec.name));
    if (spec.help) {
      const help = el("span", "wzh-help", "?");
      help.tabIndex = 0;
      help.dataset.tooltip = spec.help;
      titleRow.appendChild(help);
    }
    head.appendChild(titleRow);
    head.appendChild(el("span", `twin-accuracy-metric-card__val${spec.tone ? ` is-${spec.tone}` : ""}`, spec.value));
    card.appendChild(head);
    const chart = el("div", "twin-accuracy-metric-card__chart");
    card.appendChild(chart);
    if (spec.note) card.appendChild(el("div", "twin-accuracy-metric-card__note", spec.note));
    card.__mc2ChartHost = chart;
    card.__mc2ChartSpec = {
      name: spec.name, values: spec.values, labels: spec.labels,
      unit: spec.unit, suffix: spec.suffix,
    };
    return card;
  }

  /* ── ① 推理指标卡 ─────────────────────────────────────────────────────── */
  function renderMetrics() {
    const host = $("accuracyCharts");
    if (!host) return;
    host.replaceChildren();
    const cards = el("div", "twin-accuracy-cards mc2-metric-cards");

    const values = FORWARD_TIME.map((p) => p.value);
    const labels = FORWARD_TIME.map((p) => p.at);
    const base = values[0];
    cards.appendChild(metricCard({
      key: "forward_time",
      name: "forward_time",
      value: `${values[values.length - 1]} s`,
      values, labels,
      unit: "秒（log10）",
      suffix: " s",
      tone: "danger",
      note: "14:07:00 → 14:09:48 · 每一轮都比上一轮慢",
      help: "衡量什么：vLLM 单次 forward 的墙钟耗时，取自 server_mxfp8_all_withmc2.log 每轮打点（1.97 → 21.9 → 107 → 168 s）。\n"
        + "为什么是 log 轴：四个点跨两个数量级，线性轴会把前三点压成贴底的一条直线，看上去像「一直正常、最后突然崩」。\n"
        + "异常信号：逐轮线性累积上涨，中间没有一次回落。\n"
        + "能做什么判断：零星尖刺是调度抖动或长尾请求，不必追；这种只涨不回的形状说明某个资源被逐次污染（本案是 CCU 的 mission 5），三分钟后必然走到 20:26:52 的同步超时。",
    }));
    cards.appendChild(metricCard({
      key: "degrade_ratio",
      name: "相对稳态倍数",
      value: `${Math.round(values[values.length - 1] / base)}x`,
      values: values.map((v) => v / base),
      labels,
      unit: "倍（log10）",
      suffix: "x",
      tone: "danger",
      note: "同一份数据换算：1x → 11x → 54x → 85x",
      help: "衡量什么：同一份 forward_time 除以 14:07:00 的稳态值（1.97 s），得到与量纲无关的恶化倍数。\n"
        + "为什么要看它：绝对耗时依赖 batch 与序列长度，跨机型不可比；倍数可以直接设阈值告警（>2x 关注，>10x 立即介入）。\n"
        + "异常信号：三分钟内到 85x。\n"
        + "能做什么判断：倍数曲线与绝对耗时同形，说明恶化不是请求变重（负载变化会改变形状），而是执行侧自身在退化 —— 这一步把嫌疑从「业务流量」转到了「硬件/通信」。",
    }));

    host.appendChild(cards);
    // 卡片入 DOM 后才有实际尺寸，这时再作图（实测像素）
    Array.prototype.forEach.call(cards.children, (card) => {
      if (card.__mc2ChartHost) mountChart(card.__mc2ChartHost, card.__mc2ChartSpec);
    });

    const note = el("p", "mc2-metric-note");
    note.textContent = "线性累积而不是零星尖刺 —— 这是资源被逐次污染的形状，不是调度抖动。两段证据分别来自 server log（恶化）与 debug_plog（崩溃）。";
    host.appendChild(note);
    bindHelp(host);
  }

  // 「?」气泡：浮层与 show/position/hide 由页面内联脚本提供（与 v2 同一套），
  // 本文件建完 DOM 后按需重扫一次即可，不重复实现。
  function bindHelp(root) {
    if (typeof global.wzhBindHelpTooltips === "function") global.wzhBindHelpTooltips(root);
  }

  /* ── ③ 四卡热力 + 报错时序 ────────────────────────────────────────────── */
  function renderCluster() {
    const heat = $("heat");
    if (heat) {
      heat.replaceChildren();
      // 训练版是 32 列 2048 格；这里只有 4 张卡，列数跟着改，格子才不会缩成点
      heat.style.gridTemplateColumns = "repeat(4, minmax(0, 1fr))";
      RANKS.slice().sort((a, b) => a.rank - b.rank).forEach((entry) => {
        const cell = el("div", `twin-heat-cell mc2-rank-cell is-${entry.role}`);
        cell.dataset.rank = String(entry.rank);
        cell.dataset.tooltip = `global rank ${entry.rank} · device ${entry.device}\n报错 ${entry.at}\ncqeStatus=${entry.cqe ? "0x8000" : "0"}\n${entry.note}`;
        cell.appendChild(el("span", "mc2-rank-cell__id", `r${entry.rank}`));
        heat.appendChild(cell);
      });
      bindHelp(heat);   // 格子的 data-tooltip 也走同一套浮层
    }

    const table = $("mc2RankTable");
    if (!table) return;
    table.replaceChildren();
    const head = el("div", "mc2-rank-row is-head");
    ["报错时刻", "rank", "cqeStatus", "判读"].forEach((t) => head.appendChild(el("span", null, t)));
    table.appendChild(head);
    RANKS.forEach((entry) => {
      const row = el("div", `mc2-rank-row is-${entry.role}`);
      row.appendChild(el("span", "mc2-rank-row__time", entry.at));
      row.appendChild(el("span", "mc2-rank-row__rank", `rank ${entry.rank}`));
      row.appendChild(el("span", `mc2-rank-row__cqe${entry.cqe ? " is-bad" : ""}`, entry.cqe ? "0x8000" : "0"));
      row.appendChild(el("span", "mc2-rank-row__note", entry.note));
      table.appendChild(row);
    });
    const foot = el("p", "mc2-rank-foot");
    foot.textContent = "按报错时间排序，rank 2 在最上面 —— 但带着 device 端异常状态的是排在第三行的 rank 1。";
    table.appendChild(foot);
  }

  /* ── ④ NPU 内部单元 ───────────────────────────────────────────────────── */
  function renderUnits() {
    const host = $("mc2Units");
    if (host) {
      host.replaceChildren();
      UNITS.forEach((unit) => {
        const tile = el("div", `mc2-unit${unit.error ? " is-error" : " is-ok"}`);
        tile.dataset.unit = unit.id;
        tile.appendChild(el("span", "mc2-unit__name", unit.name));
        tile.appendChild(el("span", "mc2-unit__full", unit.full));
        tile.appendChild(el("span", "mc2-unit__flag", `${unit.id}Error = ${unit.error}`));
        host.appendChild(tile);
      });
    }

    const facts = $("mc2UnitFacts");
    if (!facts) return;
    facts.replaceChildren();
    UNIT_FACTS.forEach((fact) => {
      const row = el("div", "mc2-fact");
      row.appendChild(el("span", "mc2-fact__label", fact.label));
      row.appendChild(el("span", `mc2-fact__value${fact.tone ? ` is-${fact.tone}` : ""}`, fact.value));
      row.appendChild(el("span", "mc2-fact__note", fact.note));
      facts.appendChild(row);
    });
  }

  /* ── ② plog ───────────────────────────────────────────────────────────── */
  let logTab = "all";

  function renderLog() {
    const body = $("trainLogBody");
    const mapping = $("trainLogMappingList");
    const context = $("trainLogContext");
    if (context) {
      context.textContent = logTab === "python"
        ? "Python 侧可见的全部信息 —— 一句 RuntimeError"
        : "Device plog · Run#4（20:26:52）与 Run#1（13:57:15）的关键行";
    }

    if (body) {
      body.replaceChildren();
      const rows = LOG_ROWS.filter((row) => logTab === "all" || row.tab === logTab);
      if (!rows.length) {
        body.appendChild(el("div", "wzh-log-empty", "该分类下没有记录"));
      }
      rows.forEach((row) => {
        const line = el("div", `wzh-log-row lvl-${row.level}`);
        line.dataset.map = row.map;
        line.append(
          el("span", "wzh-log-col-time", row.time),
          el("span", `wzh-log-col-level lvl-${row.level}`, row.level),
          el("span", "wzh-log-col-comp", row.comp),
          el("span", "wzh-log-col-msg", row.msg),
        );
        // 点一行，右栏对应的翻译高亮 —— 与 v2 的日志抽屉同一套读法
        line.addEventListener("click", () => {
          body.querySelectorAll(".wzh-log-row").forEach((n) => n.classList.remove("is-selected"));
          line.classList.add("is-selected");
          mapping?.querySelectorAll(".wzh-log-mapping-row").forEach((n) => {
            n.classList.toggle("is-active", n.dataset.map === row.map);
          });
        });
        body.appendChild(line);
      });
    }

    if (mapping) {
      mapping.replaceChildren();
      LOG_MAPPING.forEach((entry) => {
        const row = el("div", "wzh-log-mapping-row");
        row.dataset.map = entry.id;
        const names = el("div", "wzh-log-mapping-names");
        names.append(
          el("span", "wzh-log-mapping-ascend", entry.ascend),
          el("span", "wzh-log-mapping-arrow", "→"),
          el("span", "wzh-log-mapping-torch", entry.torch),
        );
        row.append(names, el("div", "wzh-log-mapping-desc", entry.desc));
        mapping.appendChild(row);
      });
    }
  }

  function bindLogTabs() {
    const seg = $("trainLogTabSeg");
    if (!seg) return;
    seg.addEventListener("click", (event) => {
      const btn = event.target.closest("[data-log-tab]");
      if (!btn) return;
      logTab = btn.dataset.logTab;
      seg.querySelectorAll("[data-log-tab]").forEach((n) => {
        const on = n === btn;
        n.classList.toggle("on", on);
        n.setAttribute("aria-selected", String(on));
      });
      renderLog();
    });
  }

  /* ── 时间轴 ───────────────────────────────────────────────────────────── */
  function renderTimeline() {
    const track = $("progressTrack");
    if (!track) return;
    track.querySelectorAll(".twin-progress-marker").forEach((n) => n.remove());
    const span = T_AXIS_END - T0;
    const pctOf = (t) => Math.min(1, Math.max(0, (t - T0) / span));
    MARKERS.forEach((marker) => {
      const dot = el("div", `twin-progress-marker is-${marker.severity}`);
      dot.style.left = `${(pctOf(marker.at) * 100).toFixed(2)}%`;
      // 只有一个问题，不编号 —— 一个孤零零的「一」会让人去找「二」
      dot.textContent = marker.num || "";
      dot.title = `${marker.label}\n${marker.sub}`;
      dot.addEventListener("click", (event) => {
        event.stopPropagation();
        openCase(marker);
      });
      track.appendChild(dot);
    });
    // 进度条停在故障时刻而不是轴末：服务就是在那里断的，后面那截是空的
    const fill = $("progressFill");
    if (fill) fill.style.width = `${(pctOf(T1) * 100).toFixed(2)}%`;
    const cur = $("progressStepCurrent");
    if (cur) cur.textContent = hhmmss(T1);
    const elapsed = $("progressElapsed");
    if (elapsed) elapsed.textContent = "故障当日 · 14:07 → 20:26";
  }

  /* 进入聚光灯：先把问题定位卡填上，再交给 spotlight。与 v2 同一套 id，
     聚光灯的 exit() 靠点 #diagnosisLocatorClose 收尾。 */
  function openCase(marker) {
    const locator = $("diagnosisLocator");
    if (locator) {
      locator.classList.add("is-visible");
      const set = (id, text) => { const n = $(id); if (n) n.textContent = text; };
      set("diagnosisLocatorTitle", marker.label);
      set("diagnosisLocatorLayerTag", "MC2 融合算子 · TP=4");
      set("diagnosisLocatorCategoryTag", "通信 / 软件栈");
      set("diagnosisLocatorSeverityTag", marker.severity === "p0" ? "P0" : "P1");
      set("diagnosisLocatorDesc", marker.sub);
    }
    global.PtoTrainingSpotlight?.open?.(CASE_KEY);
  }

  function bindLocatorClose() {
    const btn = $("diagnosisLocatorClose");
    if (!btn) return;
    btn.addEventListener("click", () => {
      $("diagnosisLocator")?.classList.remove("is-visible");
      global.PtoTrainingSpotlight?.close?.();
    });
  }

  /* ══ 中央主区 · MC2 融合算子在一张卡内部的执行图 ══════════════════════════
     这一区不是「三个单元的介绍」，是一张机制图。要讲清楚的只有一件事：

       AllGatherMatmulV2_…_64_mix_aic 是**一个** FUSION_KERNEL（taskType=107，
       MIX_AIC，taskRation 1:2，crossCoreSync=1）。AIC / AIV / CCU 不是三个独立
       的东西，是同一个 kernel 的三个执行部位，靠 cross-core sync 与 notify 串起来。

     为什么必须这样画：`aicError=0 / aivError=0 / ccuError=1` 这三个 flag 摆成三
     张并列卡片时，最自然的读法是「计算单元与本案无关，只有通信单元坏了」——
     而这是错的。AIC / AIV 就在这个 kernel 里，它们停在同步点上等 CCU 的 notify；
     **等待不产生错误码**，所以 flag 是 0。断点在两者之间那条边上，不在任何一个
     盒子里。把红色给那条边、给 flag 配一句「这 0 是什么意思」，误读才堵得住。

     实现上：AIC / AIV 调设计系统 pattern 的 render()，不复制 DOM、不在本页 CSS
     里改它们的壳（pattern.json 的 forbiddenOverrides）；状态与连线都挂在外层包装。
     CCU 没有对应 pattern（设计系统只有 aic-core-object / aiv-core-object），按
     同族视觉自建，只画 history.md 指名到具体位置的结构：mission 槽 + WaitGroup。
     连线用 flex 间隙里的 DOM 元素而不是 SVG overlay —— 无需测量几何，缩放、
     换行、pattern 内部改版都不会让线跑偏。
     ═══════════════════════════════════════════════════════════════════════ */

  // §三 Run#4 / §8.2：故障 kernel 的身份，三个单元共用这一行
  const FUSED_KERNEL = {
    name: "AllGatherMatmulV2_5cc364e0…_64_mix_aic",
    facts: ["FUSION_KERNEL · taskType=107", "tilingKey=64 · MIX_AIC", "taskRation 1:2", "crossCoreSync=1"],
  };

  // history.md §三：四次复现全部落在 missionId=5；WaitGroup mask 0x000e = rank 1/2/3
  const CCU_MISSIONS = 8;
  const CCU_FAULT_MISSION = 5;
  const WAITGROUP = { sem: 337, mask: "0x000e", ranks: [1, 2, 3] };

  /* 单元外框。state 只管描边，flagNote 才是这张卡真正要说的话 ——
     「aicError = 0」单独摆着会被读成「与我无关」。 */
  function unitFrame(id, name, full, flag, state, flagNote, body) {
    const frame = el("section", `mc2-unit-frame is-${state}`);
    frame.dataset.unit = id;
    const head = el("header", "mc2-unit-frame__head");
    const title = el("div", "mc2-unit-frame__title");
    title.append(
      el("span", "mc2-unit-frame__name", name),
      el("span", "mc2-unit-frame__full", full),
    );
    const status = el("div", "mc2-unit-frame__status");
    status.append(
      el("span", `mc2-unit-frame__flag is-${state}`, flag),
      el("span", "mc2-unit-frame__flag-note", flagNote),
    );
    head.append(title, status);
    frame.append(head, body);
    return frame;
  }

  /* 两个单元之间的那条边。本案的断点就在其中一条上，所以边不是装饰：
     它要带名字（谁等谁）、带凭据（notify id）、带判读（等到了 / 等不到）。
     axis：边的朝向。AIC 与 AIV 上下叠放（省横向空间），它们之间那条是竖的；
     计算列 → CCU 那条是横的。CSS 里默认竖排，--x 在宽屏下改回横排。 */
  function unitLink({ label, meta, verdict, broken, axis = "x" }) {
    const link = el("div", `mc2-unit-link mc2-unit-link--${axis}${broken ? " is-broken" : ""}`);
    link.append(el("span", "mc2-unit-link__label", label));
    const line = el("div", "mc2-unit-link__line");
    if (broken) line.appendChild(el("span", "mc2-unit-link__break", "✕"));
    link.appendChild(line);
    link.append(
      el("span", "mc2-unit-link__meta", meta),
      el("span", `mc2-unit-link__verdict${broken ? " is-broken" : ""}`, verdict),
    );
    return link;
  }

  // 顶部横幅：三个单元同属一个 kernel。没有它，下面就是三张各说各话的卡片。
  function kernelBand() {
    const band = el("div", "mc2-kernel-band");
    const head = el("div", "mc2-kernel-band__head");
    head.append(
      el("span", "mc2-kernel-band__tag", "同一个 kernel"),
      el("span", "mc2-kernel-band__name", FUSED_KERNEL.name),
    );
    const facts = el("div", "mc2-kernel-band__facts");
    FUSED_KERNEL.facts.forEach((f) => facts.appendChild(el("span", "mc2-kernel-band__fact", f)));
    band.append(head, facts);
    band.appendChild(el("p", "mc2-kernel-band__note",
      "MC2 把通信融进算子：这一个 kernel 同时占用 AIC、AIV 与 CCU。下面三块不是三个独立部件，是它的三个执行部位。"));
    return band;
  }

  /* CCU 对象：一排 mission 槽，第 5 个描红；下面一行 WaitGroup 的四个 rank 位，
     mask 0x000e 命中的 1/2/3 标出来。这是 history.md 里 CCU 唯一被指名到具体
     位置的两处结构，不多画没有证据支撑的东西。 */
  function buildCcuObject() {
    const body = el("div", "mc2-ccu");

    const engine = el("div", "mc2-ccu__engine");
    engine.append(
      el("span", "mc2-ccu__engine-label", "commEngine = 5"),
      el("span", "mc2-ccu__engine-note", "opExpansionMode=5 · jetty / channel 子系统"),
    );

    const missionsWrap = el("div", "mc2-ccu__section");
    missionsWrap.appendChild(el("span", "mc2-ccu__section-label", "CCU mission"));
    const missions = el("div", "mc2-ccu__missions");
    for (let i = 0; i < CCU_MISSIONS; i += 1) {
      const slot = el("div", `mc2-ccu__mission${i === CCU_FAULT_MISSION ? " is-fault" : ""}`);
      // 字体与居中由 .mc2-ccu__mission 管，这里不另立钩子（与下面的 rank 槽同写法）
      slot.appendChild(el("span", null, String(i)));
      slot.title = i === CCU_FAULT_MISSION
        ? "mission 5 · WaitGroup / AllGather / AlltoAll —— 四次复现全部落在这里\nCCUM Execute Error(0x09) + Mission Task Killed(0x02)"
        : `mission ${i} · 未见异常`;
      missions.appendChild(slot);
    }
    missionsWrap.appendChild(missions);

    const wgWrap = el("div", "mc2-ccu__section");
    wgWrap.appendChild(el("span", "mc2-ccu__section-label", `WaitGroup sem[${WAITGROUP.sem}] · mask ${WAITGROUP.mask}`));
    const wg = el("div", "mc2-ccu__waitgroup");
    RANKS.slice().sort((a, b) => a.rank - b.rank).forEach((entry) => {
      const waiting = WAITGROUP.ranks.includes(entry.rank);
      const slot = el("div", `mc2-ccu__rank${waiting ? " is-waiting" : ""}`);
      slot.appendChild(el("span", null, `r${entry.rank}`));
      slot.title = waiting ? `rank ${entry.rank} 在 WaitGroup 上等待` : `rank ${entry.rank} 不在本 group 的 mask 内`;
      wg.appendChild(slot);
    });
    wgWrap.appendChild(wg);

    const err = el("div", "mc2-ccu__error");
    err.append(
      el("span", "mc2-ccu__error-code", "missionError"),
      el("span", "mc2-ccu__error-text", "CCUM Execute Error(0x09) + Mission Task Killed(0x02)"),
    );

    body.append(engine, missionsWrap, wgWrap, err);
    return body;
  }

  function renderUnitStage() {
    const stage = $("mc2UnitStage");
    if (!stage) return;
    stage.replaceChildren();
    stage.appendChild(kernelBand());

    const row = el("div", "mc2-unit-row");

    // AIC —— pattern 只有 aicDraftV1 一个预设
    const aicBody = el("div", "mc2-unit-frame__object");
    row.appendChild(unitFrame(
      "aic", "AIC", "AI Cube 计算单元", "aicError = 0",
      "blocked", "没算错 —— 停在 cross-core 同步点上，等待不产生错误码", aicBody,
    ));
    global.PtoAicCorePattern?.render?.(aicBody, "aicDraftV1");

    // AIC ↔ AIV：MIX 算子内部的正常协作，这条边是好的。AIV 叠在 AIC 下面，
    // 所以这条边是竖的（两个计算单元并排会把整行拉到 2000+ px，横向全是空转）
    row.appendChild(unitLink({
      label: "crossCoreSync = 1",
      meta: "taskRation 1 : 2",
      verdict: "正常",
      broken: false,
      axis: "y",
    }));

    // AIV —— 故障环境是 Ascend 950，用 ascend950b 预设
    const aivBody = el("div", "mc2-unit-frame__object");
    row.appendChild(unitFrame(
      "aiv", "AIV", "AI Vector 计算单元", "aivError = 0",
      "blocked", "同上：被同一条 notify 卡住，不是它算错了", aivBody,
    ));
    global.PtoAivCorePattern?.render?.(aivBody, "ascend950b");

    /* AIV ↔ CCU：断点就在这条边上。整张图的红色只给它和 CCU ——
       给盒子描红会把「哪个部件坏了」讲成结论，而结论其实是「哪条边断了」。 */
    row.appendChild(unitLink({
      label: "NotifyWait",
      meta: "NotifyId = 13224 · sqe_type=7",
      verdict: "等不到",
      broken: true,
    }));

    // CCU —— 设计系统里没有对应 pattern，本页自建
    row.appendChild(unitFrame(
      "ccu", "CCU", "通信单元", "ccuError = 1",
      "error", "mission 5 执行失败，notify 再也不会来", buildCcuObject(),
    ));

    stage.appendChild(row);

    /* 结论句：这张图存在的理由，别让读者自己去凑。它不进舞台 —— 舞台整块 scale
       到 60% 时这行小字就没法读了。写进区域置顶横幅（#mc2UnitBanner，见 html），
       在缩放之外、在图之前。 */
    const verdict = $("mc2UnitVerdict");
    if (verdict) {
      verdict.innerHTML = "三个 flag 里两个是 0，但那不是「计算单元与本案无关」——"
        + "<strong>它们就在这个 kernel 里，只是停在同步点上等 CCU 的 notify</strong>。"
        + "断点在 AIV → CCU 那条边上，不在任何一个盒子里；"
        + "Run#2 的 AIC 访问 GM 越界（errcode=264）也是卡死之后的连带表现。";
    }

    // 重渲染（如切主题）不该顶掉用户手调的倍率，交给 schedule 判断
    scheduleUnitFit();
  }

  /* 缩放：pattern 是定尺寸的设计对象，装不下时整块等比缩放，不让它们各自压扁
     （压扁会改掉 pattern 的版式契约）。scale 走 transform，不重排内部上千个格子。 */
  const UNIT_SCALE_MIN = 0.2;
  const UNIT_SCALE_MAX = 2;
  let unitScale = 1;
  // 是否还跟着容器走。适配态下窗口/侧栏/分栏一动就重算；用户手动缩放后就钉住，
  // 不能因为拖了一下分栏条就把人家调好的倍率抹掉。
  let unitFollowsViewport = true;
  let unitFitPending = false;

  const clampUnitScale = (v) => Math.max(UNIT_SCALE_MIN, Math.min(UNIT_SCALE_MAX, v));

  /* 舞台的自然尺寸。stage 的 height 被 --mc2-unit-scaled-h 顶着（见 css），直接读
     scrollHeight 量到的是上一次缩放的结果，越量越小。量之前先临时摘掉这个高度。
     transform 不影响 scrollWidth/scrollHeight（它们是布局值），无需拆 transform。 */
  function naturalUnitSize(stage) {
    const prev = stage.style.height;
    stage.style.height = "auto";
    const size = { w: stage.scrollWidth, h: stage.scrollHeight };
    stage.style.height = prev;
    return size;
  }

  function applyUnitScale() {
    const stage = $("mc2UnitStage");
    const viewport = $("mc2UnitViewport");
    if (!stage || !viewport) return;
    const natural = naturalUnitSize(stage);
    stage.style.transform = `scale(${unitScale})`;
    // transform 不改变布局盒，父容器要按缩放后的尺寸留位，否则底部被裁 / 底下空一片
    viewport.style.setProperty("--mc2-unit-scaled-h", `${natural.h * unitScale}px`);
    const readout = $("mc2UnitZoomReadout");
    if (readout) readout.textContent = `${Math.round(unitScale * 100)}%`;
  }

  /* 适配 = 装进视口，两个方向都算。原先只除宽度，高度溢出多少不管 —— 点了「适配」
     图还是被下边缘切掉、竖滚动条照在，所以它看起来根本没适配。取两轴比例的较小值
     才是「整张图都在窗口里」。 */
  function fitUnitStage() {
    const stage = $("mc2UnitStage");
    const viewport = $("mc2UnitViewport");
    if (!stage || !viewport) return;
    const natural = naturalUnitSize(stage);
    if (!natural.w || !natural.h) return;

    // clientWidth/Height 含 padding，可用区要把 padding 扣掉，否则右下总差一截
    const cs = global.getComputedStyle(viewport);
    const availW = viewport.clientWidth
      - (parseFloat(cs.paddingLeft) || 0) - (parseFloat(cs.paddingRight) || 0);
    const availH = viewport.clientHeight
      - (parseFloat(cs.paddingTop) || 0) - (parseFloat(cs.paddingBottom) || 0);
    if (availW <= 0 || availH <= 0) return;

    unitScale = clampUnitScale(Math.min(availW / natural.w, availH / natural.h));
    applyUnitScale();
  }

  // 容器尺寸变化期间连续触发，合到下一帧只算一次
  function scheduleUnitFit() {
    if (unitFitPending) return;
    unitFitPending = true;
    global.requestAnimationFrame(() => {
      unitFitPending = false;
      if (unitFollowsViewport) fitUnitStage();
      else applyUnitScale();
    });
  }

  function zoomUnitStage(factor) {
    unitFollowsViewport = false;      // 手动调过就不再自动改
    unitScale = clampUnitScale(unitScale * factor);
    applyUnitScale();
  }

  function bindUnitZoom() {
    const viewport = $("mc2UnitViewport");
    $("mc2UnitZoomIn")?.addEventListener("click", () => zoomUnitStage(1.15));
    $("mc2UnitZoomOut")?.addEventListener("click", () => zoomUnitStage(1 / 1.15));
    $("mc2UnitZoomFit")?.addEventListener("click", () => {
      unitFollowsViewport = true;     // 点「适配」= 交还给容器
      fitUnitStage();
    });

    /* 改变可视区的不止窗口：收起左栏 / infra 栏、拖上下分栏条、开合底部 dock 都会
       改这块的宽高，而它们一个 resize 事件都不发。盯容器本身才盯得住。 */
    if (viewport && typeof global.ResizeObserver === "function") {
      new global.ResizeObserver(scheduleUnitFit).observe(viewport);
    } else {
      global.addEventListener("resize", scheduleUnitFit);
    }
  }

  /* ── 页面级开关：聚光灯的 prep() 会调这几个 ───────────────────────────── */
  function installShellApis() {
    /* 底部 dock 开关。ide-frame 的 initBottomTerminalToggle 只在页面存在
       data-ide-toggle="terminal" 时才接管 visualization 按钮，本页没有 terminal
       面板，所以这一段要自己补 —— 与 v2 同实现，连分隔条一起收（只 hidden 面板
       会在原位留下一条可拖的空 gutter）。 */
    global.PtoTrainingTwinTimelineDock = {
      setVisible(on) {
        const dock = $("bottomDock");
        if (!dock) return;
        const show = !!on;
        dock.hidden = !show;
        dock.setAttribute("aria-hidden", String(!show));
        const prev = dock.previousElementSibling;
        if (prev && prev.matches(".pto-workbench-shell__split-gutter")) prev.hidden = !show;
        const toggle = $("twinTimelineToggle");
        if (toggle) {
          toggle.classList.toggle("is-selected", show);
          toggle.setAttribute("aria-pressed", String(show));
          toggle.setAttribute("aria-expanded", String(show));
          toggle.title = show ? "隐藏底部面板" : "显示底部面板";
        }
      },
    };

    global.PtoTrainingTwinDockTabs = {
      select(name) {
        document.querySelectorAll("[data-dock-tab]").forEach((tab) => {
          const on = tab.dataset.dockTab === name;
          tab.classList.toggle("is-selected", on);
          tab.setAttribute("aria-selected", String(on));
          tab.tabIndex = on ? 0 : -1;
        });
        document.querySelectorAll("[data-dock-panel]").forEach((panel) => {
          panel.hidden = panel.dataset.dockPanel !== name;
        });
      },
    };

    // 日志入口：展开 dock 并切到 plog 页签，每次都重渲染（与 v2 的 show() 同义）
    global.PtoTrainingLogDrawer = {
      show() {
        global.PtoTrainingTwinTimelineDock.setVisible(true);
        global.PtoTrainingTwinDockTabs.select("log");
        renderLog();
      },
    };

    /* 侧列开合走 .twin-center-scroll 上的 is-left/right-collapsed —— 三列同处一个
       grid，收起是把那条轨道压到 0，不是给列自己加 display:none（那样 grid 仍留着
       一条空轨道和一份 column-gap）。 */
    const scroll = () => document.querySelector(".twin-center-scroll");

    function setColVisible(side, on) {
      scroll()?.classList.toggle(`is-${side}-collapsed`, !on);
      const toggle = $(side === "right" ? "twinRightColToggle" : "twinLeftColToggle");
      if (toggle) {
        toggle.classList.toggle("is-selected", !!on);
        toggle.setAttribute("aria-pressed", String(!!on));
        toggle.setAttribute("aria-expanded", String(!!on));
        toggle.title = `${on ? "隐藏" : "显示"} ${side === "right" ? "infra" : "推理监控"} 栏`;
      }
      // 中列宽度跟着变，单元舞台要重新适配（等这一帧的布局落定）。走 schedule 而不是
      // 直接 fit：用户手动缩放过就该保住他的倍率，收个侧栏不是重置的理由。
      scheduleUnitFit();
    }

    global.PtoTrainingTwinSideCols = {
      setRightVisible(on) { setColVisible("right", on); },
      setLeftVisible(on) { setColVisible("left", on); },
      isRightVisible() { return !scroll()?.classList.contains("is-right-collapsed"); },
      isLeftVisible() { return !scroll()?.classList.contains("is-left-collapsed"); },
    };
  }

  function bindShellButtons() {
    $("trainLogToggle")?.addEventListener("click", () => global.PtoTrainingLogDrawer.show());
    $("bottomDockCloseBtn")?.addEventListener("click", () => global.PtoTrainingTwinTimelineDock.setVisible(false));
    $("twinTimelineToggle")?.addEventListener("click", () => {
      const dock = $("bottomDock");
      global.PtoTrainingTwinTimelineDock.setVisible(!!dock?.hidden);
    });
    $("twinRightColToggle")?.addEventListener("click", () => {
      const cols = global.PtoTrainingTwinSideCols;
      cols.setRightVisible(!cols.isRightVisible());
    });
    $("twinLeftColToggle")?.addEventListener("click", () => {
      const cols = global.PtoTrainingTwinSideCols;
      cols.setLeftVisible(!cols.isLeftVisible());
    });
    document.querySelectorAll("[data-dock-tab]").forEach((tab) => {
      tab.addEventListener("click", () => global.PtoTrainingTwinDockTabs.select(tab.dataset.dockTab));
    });
  }

  function boot() {
    installShellApis();
    bindShellButtons();
    bindLogTabs();
    bindLocatorClose();
    bindUnitZoom();
    renderMetrics();
    renderCluster();
    renderUnits();
    renderUnitStage();
    renderLog();
    renderTimeline();
    /* 三块取证区默认全部在场。v2 把 infra 栏和底部面板默认收起，是因为它那两个
       案例进来先看整网图与精度曲线；本案没有「先看哪张图」的层级 —— 四卡定界、
       单元定界、plog 三者是同一条定界链上连着的三步，开局就该同时看得见。 */
    global.PtoTrainingTwinSideCols.setRightVisible(true);
    global.PtoTrainingTwinTimelineDock.setVisible(true);
    global.PtoTrainingTwinDockTabs.select("log");
    // 侧栏定型后中列宽度才是最终值，这时再适配一次单元舞台
    requestAnimationFrame(fitUnitStage);
  }

  global.Mc2Monitoring = {
    // pattern 自己读 :root[data-theme]，切主题后重渲染一次即可
    setTheme() { renderUnitStage(); },
    openCase() { openCase(MARKERS[MARKERS.length - 1]); },
    fitUnitStage,
    facts: { FORWARD_TIME, RANKS, UNITS, MARKERS },
  };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})(window);
