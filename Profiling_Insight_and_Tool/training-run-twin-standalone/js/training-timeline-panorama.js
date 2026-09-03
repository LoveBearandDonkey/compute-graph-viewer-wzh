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

   ── 分层 ───────────────────────────────────────────────────────────────
   面板从上到下:指标栏 → 三级量尺 → L1 关键事件轴(本浮窗原有内容)。
   三级量尺依次向下放大,三条带子同宽同风格,与下面 L1 的事件轴左右对齐:
     ① epoch:整段训练,块宽 = 该 epoch 实际耗时
     ② step :选中步附近逐步一格,格宽 = 该步实际耗时
     ③ L2   :选中那一步的三大阶段(前向传播 / 反向传播 / 更新),可折叠
   三层共用一套填充语言:蓝色 = 已完成,灰色斜线 = 未完成,进行中的那一格从左往右涨。
   只有已执行的 step 才有真实耗时;liveStep 之后一律按基线 T_iter 等宽占位 —— 没跑过
   的迭代不该在量尺上显出宽窄差别。量尺默认选中时光机当前展示的那一步。

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

  /* ══ 双层量尺 · 迭代耗时模型 ════════════════════════════════════════════
     量尺横向不按 step 均分,而是按「每一步真实花掉多少时间」铺开,所以刻度天然不等距:
     warmup 没进稳态、每 200 步一次的全局指标归约、ckpt 落盘阻塞、straggler 抖动、
     HCCS 掉链路后回退 RoCE 慢路径、事故步的停机与回滚 —— 同样 2000 步的 epoch 会占
     据明显不同的宽度。

     一份 stepCostAt() 同时喂三处,不允许各画各的:
       ① epoch 层:块宽 = 该 epoch 内 2000 步耗时之和
       ② step  层:段宽/刻度位置 = 选中 epoch 内的累计耗时(所以刻度不是均匀排开的)
       ③ L2 三阶段:前向 / 反向 / 更新按选中那一步的耗时切分

     口径边界:这里只表达「相对耗时」。KPI 的已训练时长、事件时间戳仍走顶栏那套
     step × TIME_MACHINE_STEP_SECONDS 的统一口径,两边不会给出互相打架的数字。 */
  const SVG_NS = "http://www.w3.org/2000/svg";
  // step 层一次铺开多少步:编号要写进格子里,格子就得够宽,铺太多就只剩色块了
  const RULER = { WINDOW: 24, LIVE_TAIL: 3 };

  function hash01(n) {
    const x = Math.sin(n * 12.9898 + 78.233) * 43758.5453;
    return x - Math.floor(x);
  }

  // 单步耗时(秒)。base = 稳态 T_iter 基线(8.5s,见 js/training-run-twin.js)。
  function stepCostAt(step, base) {
    const b = base || 8.5;
    let k = 1 + (hash01(step) - 0.5) * 0.18;                  // 常态抖动 ±9%
    if (step % 200 < 4) k += 0.30;                            // 每 200 步一次全局指标归约 + 日志落盘
    if (hash01(step * 1.7 + 11) > 0.972) k += 0.34;           // 偶发慢卡拖一拍
    if (step < 2400) k *= 1.12 - 0.12 * (step / 2400);        // warmup 未进稳态
    if (step >= 6000 && step < 6800) k *= 1.16;               // straggler node-37 落后 1.8×
    if (step >= 8300 && step < 8900) k *= 1.08;               // FP8 溢出后 grad_norm 抬升
    if (step >= 11600 && step < 12003) k *= 1.08 + 0.42 * ((step - 11600) / 403);  // 显存爬升 + 碎片整理
    if (step >= 18600 && step < 20000) k *= 1.88;             // HCCS lane5 掉链路 → 回退 RoCE
    let extra = 0;
    if (step > 0 && step % 10000 === 0) extra += 46;          // ckpt 落盘阻塞(118GB / 64 分片)
    if (step === 12003) extra += 1320;                        // 显存 OOM 中断 + 拉起
    if (step === 15203) extra += 2040;                        // Router 溢出 NaN + 回滚 step_15200
    return b * k + extra;
  }

  let costCache = null;    // 全程逐 epoch 的「真实耗时」表(与 liveStep 无关,只算一次)

  function costModel(c) {
    const key = c.totalSteps + "|" + c.stepsPerEpoch + "|" + c.stepSeconds;
    if (costCache && costCache.key === key) return costCache;
    const stepsPerEpoch = Math.max(1, c.stepsPerEpoch);
    const nEpochs = Math.max(1, Math.ceil(c.totalSteps / stepsPerEpoch));
    const real = new Array(nEpochs).fill(0);
    for (let s = 0; s < c.totalSteps; s += 1) real[Math.floor(s / stepsPerEpoch)] += stepCostAt(s, c.stepSeconds);
    costCache = { key, nEpochs, stepsPerEpoch, base: c.stepSeconds, real };
    return costCache;
  }

  /* 量尺上每个 epoch 占多宽:已跑完的按真实耗时,liveStep 之后的按基线 T_iter 等宽占位
     —— 没执行过的迭代不该显出宽窄差别。跨在 liveStep 上的那个 epoch 逐步算,两边各按各的。 */
  function bandOf(model, live) {
    const spe = model.stepsPerEpoch;
    const flat = spe * model.base;
    const epochs = new Array(model.nEpochs);
    for (let i = 0; i < model.nEpochs; i += 1) {
      const from = i * spe;
      if (from > live) epochs[i] = flat;
      else if (from + spe - 1 <= live) epochs[i] = model.real[i];
      else {
        let sum = 0;
        for (let s = from; s < from + spe; s += 1) sum += s <= live ? stepCostAt(s, model.base) : model.base;
        epochs[i] = sum;
      }
    }
    const offsets = [];
    let acc = 0;
    for (let i = 0; i < model.nEpochs; i += 1) { offsets.push(acc); acc += epochs[i]; }
    return { epochs, offsets, total: acc || 1, stepsPerEpoch: spe, nEpochs: model.nEpochs };
  }

  // 全程 x(0~1):epoch 内线性摊开就够画事件小刺 / 游标了,不必为此展开 2000 步
  function runX(band, step) {
    const e = Math.min(band.nEpochs - 1, Math.max(0, Math.floor(step / band.stepsPerEpoch)));
    const t = (step - e * band.stepsPerEpoch) / band.stepsPerEpoch;
    return (band.offsets[e] + band.epochs[e] * t) / band.total;
  }

  /* ══ L2 · 单步三阶段模型 ══════════════════════════════════════════════
     前向 / 反向 / 更新按 30 / 52 / 18 切分常规耗时;事故与落盘带来的额外阻塞先摘出来,
     挂到它真正发生的那个阶段上(Router 溢出在前向、显存触顶在反向、ckpt 落盘在更新)。 */
  const PHASES = [
    { key: "fwd", name: "前向传播", color: "#3b6fe0" },
    { key: "bwd", name: "反向传播", color: "#8b5cf6" },
    { key: "upd", name: "更新",     color: "#10b981" },
  ];

  function scaleTo(segs, target) {
    const sum = segs.reduce((a, s) => a + s.seconds, 0) || 1;
    const k = target / sum;
    segs.forEach((s) => { s.seconds *= k; });
    return segs;
  }

  function phaseBreakdown(step, base) {
    const total = stepCostAt(step, base);
    const degraded = step >= 18600 && step < 20000;
    const ckptSave = step > 0 && step % 10000 === 0;
    const oom = step === 12003;
    const nanStep = step === 15203;
    const climbing = step >= 11600 && step < 12003;

    let exFwd = 0;
    let exBwd = 0;
    let exUpd = 0;
    if (ckptSave) exUpd += 46;
    if (oom) exBwd += 1320;
    if (nanStep) exFwd += 2040;
    const core = Math.max(0.2, total - exFwd - exBwd - exUpd);
    const cb = degraded ? 2.6 : 1;   // 慢路径下通信段膨胀

    const seg = (name, seconds, kind) => ({ name, seconds, kind });

    const fwd = scaleTo([
      seg("算子计算", 0.60, "compute"),
      seg("TP all-gather", 0.13 * cb, "comm"),
      seg("EP dispatch · all-to-all", 0.19 * cb, "comm"),
      seg("PP 空泡", 0.08, "stall"),
    ], core * 0.30);
    if (exFwd) fwd.push(seg("Router 溢出 → NaN 检测 + 回滚", exFwd, "stall"));

    const bwd = scaleTo([
      seg("梯度计算", 0.44, "compute"),
      seg("重计算(激活重算)", climbing || oom ? 0.30 : 0.17, "compute"),
      seg("EP combine · all-to-all", 0.14 * cb, "comm"),
      seg("梯度 reduce-scatter", 0.15 * cb, "comm"),
      seg("PP 空泡", 0.08, "stall"),
    ], core * 0.52);
    if (exBwd) bwd.push(seg("显存触顶 → 中断与拉起", exBwd, "stall"));

    const upd = scaleTo([
      seg("优化器更新", 0.46, "compute"),
      seg("参数 all-gather", 0.32 * cb, "comm"),
      seg("显存整理", 0.12, "stall"),
      seg("指标采集 / 日志", 0.10, "compute"),
    ], core * 0.18);
    if (exUpd) upd.push(seg("ckpt 落盘(118GB · 64 分片)", exUpd, "io"));

    const notes = [];
    if (degraded) notes.push("HCCS lane5 掉链路,通信段走 RoCE 慢路径");
    if (climbing) notes.push("显存爬升,重计算比例被抬高");
    if (ckptSave) notes.push("本步含 ckpt 落盘阻塞");
    if (oom) notes.push("本步触发显存 OOM 中断");
    if (nanStep) notes.push("本步 Router 溢出 → NaN,已回滚 step_15200");

    const groups = [fwd, bwd, upd];
    return {
      total,
      notes,
      phases: PHASES.map((p, i) => {
        const segs = groups[i];
        const seconds = segs.reduce((a, s) => a + s.seconds, 0);
        return { key: p.key, name: p.name, color: p.color, seconds, pct: seconds / total, segs };
      }),
    };
  }

  /* 当前这一步跑到哪儿了(0~1)。
     首选整网图 deck 的播放节拍 —— liveStep 本来就是它「一轮前向 46 层 + 反向 46 层」
     跑满一圈时 twinAdvanceStep(1) 推进的(见 js/training-monitoring-v2-deck.js),两边共用
     同一个时钟,格子填满的那一刻正好换到下一格,而不是在原地反复播同一格。
     deck 没在播(reduced-motion 静态态)时,退回按观测到的 liveStep 变化节拍自校准。 */
  let liveSeen = { step: -1, at: 0, period: 0 };

  function nowMs() {
    return (window.performance && performance.now) ? performance.now() : Date.now();
  }

  function observeLive(step) {
    const t = nowMs();
    if (liveSeen.step < 0) { liveSeen = { step, at: t, period: 0 }; return; }
    if (step !== liveSeen.step) {
      if (step > liveSeen.step) liveSeen.period = (t - liveSeen.at) / (step - liveSeen.step);
      liveSeen.step = step;
      liveSeen.at = t;
    }
  }

  function liveFraction() {
    const deck = window.PtoTwinGraphAdapter;
    if (deck && typeof deck.stepProgress === "function") {
      const v = deck.stepProgress();
      if (v != null && isFinite(v)) return Math.max(0, Math.min(0.999, v));
    }
    if (!liveSeen.period) return 0;
    return Math.max(0, Math.min(0.999, (nowMs() - liveSeen.at) / liveSeen.period));
  }

  // 进行中的那个 epoch:已跑掉的耗时占这个 epoch 的多少(未跑的部分按基线记)
  function execFraction(model, idx, live) {
    const from = idx * model.stepsPerEpoch;
    let doneSec = 0;
    let allSec = 0;
    for (let s = from; s < from + model.stepsPerEpoch; s += 1) {
      const d = s <= live ? stepCostAt(s, model.base) : model.base;
      allSec += d;
      if (s <= live) doneSec += d;
    }
    return allSec ? doneSec / allSec : 0;
  }

  function fmtSec(s) {
    if (s >= 3600) return (s / 3600).toFixed(1) + "h";
    if (s >= 60) return Math.floor(s / 60) + "分" + Math.round(s % 60) + "秒";
    return (s < 10 ? s.toFixed(2) : s.toFixed(1)) + "s";
  }

  function esc(s) {
    return String(s == null ? "" : s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
  }

  const $ = (id) => document.getElementById(id);
  let root = null;      // .tw-pano
  let els = null;       // 缓存的节点引用
  let activeTypes = null;  // null = 全部；否则是 Set
  let placed = [];      // 最近一次排布结果:{ ev, dot, label, lead }
  let selectedStep = null; // 三级量尺选中的 step;null = 打开时默认落在时光机当前那一步
  let l2Collapsed = false; // L2 阶段量尺是否折叠(折叠后高度让给 L1 的事件车道)
  let lastBand = null;     // renderRuler() 算出的横向坐标:L1 的轴与量尺共用它,事件才对得齐
  let progTimer = 0;       // 进行中的 step / 阶段往前涨的心跳
  let lastLive = -1;       // 上一拍看到的 liveStep,变了就说明这一步跑完了
  let followLive = false;  // 选中的就是「最新一步」时,训练往前走,选中也跟着走
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

    /* ── 三级量尺:epoch → step → L2 阶段 ────────────────────────────────
       三条带子同宽、同风格,依次向下放大,中间用梯形接续。标题连同当前数值放在各自
       条形图「上方」而不是左侧 —— 左侧标题会吃掉横向宽度,量尺就和下面 L1 的事件轴
       对不齐了。 */
    const ruler = h("div", "tw-pano__ruler");

    const tierHead = (name) => {
      const el = h("div", "tw-pano__tier-head");
      el.appendChild(h("span", "tw-pano__tier-name", name));
      const val = h("span", "tw-pano__tier-val");
      el.appendChild(val);
      return { el, val };
    };

    // ① epoch
    const tierE = h("div", "tw-pano__tier");
    const headE = tierHead("epoch");
    const epochsBand = h("div", "tw-pano__ruler-band");
    const epochs = h("div", "tw-pano__epochs");
    const epochPips = h("div", "tw-pano__pips");
    const rulerLive = h("div", "tw-pano__ruler-live");
    epochsBand.appendChild(epochs);
    epochsBand.appendChild(epochPips);
    epochsBand.appendChild(rulerLive);
    tierE.appendChild(headE.el);
    tierE.appendChild(epochsBand);

    // ② step
    const tierS = h("div", "tw-pano__tier");
    const headS = tierHead("step");
    const stepsBand = h("div", "tw-pano__ruler-band tw-pano__ruler-band--steps");
    const stepBars = h("div", "tw-pano__sbars");
    const stepPips = h("div", "tw-pano__pips");
    stepsBand.appendChild(stepBars);
    stepsBand.appendChild(stepPips);
    tierS.appendChild(headS.el);
    tierS.appendChild(stepsBand);

    // ③ L2:那一步的三大阶段,可折叠
    const tierL = h("div", "tw-pano__tier tw-pano__tier--l2");
    const headL = h("div", "tw-pano__tier-head");
    const l2Toggle = h("button", "tw-pano__tier-toggle",
      '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 9 6 6 6-6"></path></svg>');
    l2Toggle.type = "button";
    l2Toggle.setAttribute("aria-expanded", "true");
    l2Toggle.title = "折叠 L2 阶段量尺";
    const l2Val = h("span", "tw-pano__tier-val");
    const l2Notes = h("span", "tw-pano__tier-notes");
    headL.appendChild(l2Toggle);
    headL.appendChild(h("span", "tw-pano__tier-name", "L2 · 阶段"));
    headL.appendChild(l2Val);
    headL.appendChild(l2Notes);
    const l2Body = h("div", "tw-pano__l2-body");
    const phases = h("div", "tw-pano__ruler-band tw-pano__phases");
    const l2Cards = h("div", "tw-pano__l2-cards");
    l2Body.appendChild(phases);
    l2Body.appendChild(l2Cards);
    tierL.appendChild(headL);
    tierL.appendChild(l2Body);

    ruler.appendChild(tierE);
    ruler.appendChild(tierS);
    ruler.appendChild(tierL);

    // step 带上一击 = 选中被点中的那一格(= 那一步);点到靠边的格子会带着窗口一起挪
    stepsBand.addEventListener("click", (e) => {
      const cell = e.target.closest(".tw-pano__sb");
      if (!cell || cell.classList.contains("is-future")) return;
      const step = Number(cell.dataset.step);
      if (Number.isFinite(step)) selectStep(step);
    });
    // 事件小刺:直接选中该事件所在的那一步(未跑到的计划事件不接受选中)
    const pipClick = (e) => {
      const pip = e.target.closest(".tw-pano__pip");
      if (!pip) return;
      e.stopPropagation();
      const step = Number(pip.dataset.step);
      if (!Number.isFinite(step) || step > ctx().liveStep) return;
      selectStep(step);
    };
    epochPips.addEventListener("click", pipClick);
    stepPips.addEventListener("click", pipClick);
    l2Toggle.addEventListener("click", () => setL2Collapsed(!l2Collapsed));

    panel.appendChild(head);
    panel.appendChild(ruler);
    panel.appendChild(stage);
    root.appendChild(backdrop);
    root.appendChild(panel);
    document.body.appendChild(root);

    els = {
      panel, kpis, filters, stage, ticks, axis, fill, live, now, field, leads,
      ruler, epochs, epochPips, rulerLive, epochVal: headE.val,
      stepsBand, stepBars, stepPips, stepVal: headS.val,
      tierL, l2Toggle, l2Val, l2Notes, l2Body, phases, l2Cards,
      epochEls: [], phaseEls: [], phaseSpans: [], phaseLive: false, liveCellEl: null, eventsLabel: null,
    };
    setL2Collapsed(l2Collapsed);
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
    // 「事件」也是一项指标:取值位放类型筛选药丸(兼图例),排在总进度右侧
    els.kpis.innerHTML = items.map((it) =>
      `<div class="tw-pano__kpi${it.num ? " tw-pano__kpi--num" : ""}"><span>${it.k}</span><strong>${it.v}</strong></div>`
    ).join("") + '<div class="tw-pano__kpi tw-pano__kpi--events"><span data-kpi-events></span></div>';
    const slot = els.kpis.querySelector(".tw-pano__kpi--events");
    els.eventsLabel = slot.querySelector("[data-kpi-events]");
    slot.appendChild(els.filters);
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
    if (els.eventsLabel) els.eventsLabel.textContent = "事件";
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

  /* L1 的横向坐标:与三级量尺共用同一套「按实际耗时铺开」的映射。
     两边各算各的(一边按 step 均分、一边按耗时)的话,同一个事件在量尺上和在事件轴上会落在不同位置。 */
  function xOf(c, step) {
    if (lastBand) return runX(lastBand, step);
    return c.totalSteps ? step / c.totalSteps : 0;
  }

  function renderTicks(c) {
    const marks = [0, 0.25, 0.5, 0.75, 1];
    els.ticks.innerHTML = marks.map((m) => {
      const step = Math.round(c.totalSteps * m);
      return `<div class="tw-pano__tick" style="left:${(xOf(c, step) * 100).toFixed(2)}%">${step.toLocaleString()}</div>`;
    }).join("");
  }

  function renderAxis(c) {
    const pct = xOf(c, c.step) * 100;
    const livePct = xOf(c, c.liveStep) * 100;
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
    const used = padY + gapY * 2                                   // 面板:指标行 / 量尺 / 轴区 三段之间的间距
      + els.panel.querySelector(".tw-pano__head").offsetHeight
      + els.ruler.offsetHeight
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
      const markerX = Math.max(0, Math.min(W, xOf(c, ev.step) * W));
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
        markRuler(ev, on);   // 量尺上同步点亮它所在的 epoch 块与事件小刺
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

  /* ── 三级量尺 ────────────────────────────────────────────────────────
     ① epoch:整段训练,块宽 = 该 epoch 实际耗时
     ② step :选中步附近逐步一格,格宽 = 该步实际耗时,编号写在格子里
     ③ L2   :选中那一步的三大阶段(渲染见 renderL2)
     三层只用「宽度」表达耗时,用同一套填充表达进度:--prog 左侧蓝色 = 已完成,
     右侧灰斜线 = 未完成 —— 所以进行中的 epoch 与进行中的 step 长得一模一样。 */
  function renderRuler(c) {
    const model = costModel(c);
    const spe = model.stepsPerEpoch;
    const live = Math.max(0, Math.min(c.totalSteps - 1, c.liveStep));
    // 默认落在时光机当前展示的那一步:实时态就是最新一个 step,回放态则跟着拖块走
    if (selectedStep == null) {
      selectedStep = Math.min(live, c.step);
      followLive = selectedStep === live;   // 实时态默认盯着最新一步,回放态钉在拖块那一步
    }
    selectedStep = Math.max(0, Math.min(live, selectedStep));
    const band = bandOf(model, live);
    lastBand = band;   // L1 的轴/刻度/事件点都走这一份坐标
    const selEpoch = Math.floor(selectedStep / spe);
    const liveEpoch = Math.floor(live / spe);
    const liveFrac = liveFraction();

    // ── ① epoch ──
    els.epochVal.textContent = (selEpoch + 1) + " / " + model.nEpochs;
    els.epochs.innerHTML = "";
    els.epochEls = [];
    let liveEpochFrac = 0;   // 进行中那个 epoch 的进度:进度线要正好落在蓝/灰斜线的交界上
    for (let i = 0; i < model.nEpochs; i += 1) {
      const from = i * spe;
      const future = from > live;
      const running = i === liveEpoch;
      const prog = future ? 0 : (running ? execFraction(model, i, live) : 1);
      if (running) liveEpochFrac = prog;
      const b = h("button", "tw-pano__ep");
      b.type = "button";
      b.style.width = ((band.epochs[i] / band.total) * 100).toFixed(3) + "%";
      b.style.setProperty("--prog", prog.toFixed(4));
      b.classList.toggle("is-future", future);
      b.classList.toggle("is-running", running);
      b.classList.toggle("is-selected", i === selEpoch);
      b.appendChild(h("span", "tw-pano__ep-n", String(i + 1)));
      b.title = "epoch " + (i + 1) + " · step " + from.toLocaleString() + "–" + (from + spe - 1).toLocaleString()
        + (future
          ? " · 计划中，按基线 " + fmtSec(model.base) + "/步 等宽占位"
          : running
            ? " · 进行中，已跑 " + (prog * 100).toFixed(1) + "%"
            : " · 耗时 " + fmtDurationCN(band.epochs[i]) + "，平均单步 " + fmtSec(band.epochs[i] / spe));
      if (future) b.setAttribute("aria-disabled", "true");
      else b.addEventListener("click", () => selectStep(Math.min(live, from + spe - 1)));
      els.epochs.appendChild(b);
      els.epochEls.push(b);
    }
    els.epochEls.forEach((b) => b.classList.toggle("is-narrow", b.offsetWidth < 15));

    els.epochPips.innerHTML = EVENTS.map((ev) => {
      const meta = TYPES[ev.type] || TYPES.event;
      return '<i class="tw-pano__pip" data-step="' + ev.step + '" style="left:'
        + (runX(band, ev.step) * 100).toFixed(3) + '%;--c:' + meta.color + '" title="'
        + esc(meta.label + " · " + ev.name + " · step " + ev.step.toLocaleString()) + '"></i>';
    }).join("");
    // 进度线按耗时口径落在进行中那个 epoch 的蓝/灰交界处(runX 的 epoch 内是线性近似)
    const liveX = (band.offsets[liveEpoch] + band.epochs[liveEpoch] * liveEpochFrac) / band.total;
    els.rulerLive.style.left = (liveX * 100).toFixed(3) + "%";

    /* ── ② step ──
       整个 epoch 有 2000 步,铺不开也没必要:把窗口钉在选中步周围逐步一格,格与格的
       宽度差才看得见(聚合成几十格会把差异平均掉),编号也才写得进格子里。 */
    const epFrom = selEpoch * spe;
    const epTo = Math.min(c.totalSteps, epFrom + spe);
    const win = Math.min(epTo - epFrom, RULER.WINDOW);
    /* 窗口居中对齐选中步,但右边最多只留 LIVE_TAIL 格未执行的占位格 ——
       选在最新一步时若也居中,半条带子都是等宽的「还没跑」格,白白浪费横向空间。 */
    let winFrom = Math.max(epFrom, Math.min(epTo - win, selectedStep - Math.floor(win / 2)));
    if (winFrom + win > live + 1 + RULER.LIVE_TAIL) winFrom = Math.max(epFrom, live + 1 + RULER.LIVE_TAIL - win);
    const winTo = winFrom + win;
    const durs = new Array(win);
    let winTotal = 0;
    for (let i = 0; i < win; i += 1) {
      const s = winFrom + i;
      durs[i] = s <= live ? stepCostAt(s, model.base) : model.base;   // 未执行的一律按基线,等宽
      winTotal += durs[i];
    }
    const xs = new Array(win);
    let acc = 0;
    for (let i = 0; i < win; i += 1) { xs[i] = (acc + durs[i] / 2) / winTotal; acc += durs[i]; }
    const winX = (s) => xs[Math.max(0, Math.min(win - 1, s - winFrom))];

    els.stepVal.textContent = selectedStep.toLocaleString() + " / " + c.totalSteps.toLocaleString();
    let bars = "";
    for (let i = 0; i < win; i += 1) {
      const s = winFrom + i;
      const future = s > live;
      const running = s === live;
      const prog = future ? 0 : (running ? liveFrac : 1);
      bars += '<i class="tw-pano__sb' + (future ? " is-future" : "") + (running ? " is-running" : "")
        + (s === selectedStep ? " is-sel" : "") + '" data-step="' + s
        + '" style="width:' + ((durs[i] / winTotal) * 100).toFixed(3) + '%;--prog:' + prog.toFixed(4) + '" title="'
        + esc("step " + s.toLocaleString()
          + (future ? " · 未执行，按基线 " + fmtSec(model.base) + " 等宽占位"
                    : running ? " · 进行中 · 单步约 " + fmtSec(durs[i])
                              : " · 单步 " + fmtSec(durs[i]))) + '">'
        + '<span class="tw-pano__sb-n">' + s.toLocaleString() + "</span></i>";
    }
    els.stepBars.innerHTML = bars;
    // 格子太窄就藏掉编号;进行中/选中的那一格无论如何都要露出 step 号
    els.liveCellEl = null;
    els.stepBars.querySelectorAll(".tw-pano__sb").forEach((cell) => {
      const keep = cell.classList.contains("is-running") || cell.classList.contains("is-sel");
      cell.classList.toggle("is-narrow", !keep && cell.offsetWidth < 34);
      if (cell.classList.contains("is-running")) els.liveCellEl = cell;
    });

    els.stepPips.innerHTML = EVENTS.filter((ev) => ev.step >= winFrom && ev.step < winTo).map((ev) => {
      const meta = TYPES[ev.type] || TYPES.event;
      return '<i class="tw-pano__pip" data-step="' + ev.step + '" style="left:'
        + (winX(ev.step) * 100).toFixed(3) + '%;--c:' + meta.color + '" title="'
        + esc(meta.label + " · " + ev.name) + '"></i>';
    }).join("");
  }

  // L1 的事件标签悬浮时,量尺上点亮它落在哪个 epoch / 哪一步
  function markRuler(ev, on) {
    if (!els || !els.epochEls) return;
    const c = ctx();
    const block = els.epochEls[Math.floor(ev.step / Math.max(1, c.stepsPerEpoch))];
    if (block) block.classList.toggle("is-hot", on);
    els.ruler.querySelectorAll('.tw-pano__pip[data-step="' + ev.step + '"]')
      .forEach((p) => p.classList.toggle("is-hot", on));
  }

  function selectStep(step) {
    const c = ctx();
    const next = Math.max(0, Math.min(c.liveStep, Math.round(step)));
    if (next === selectedStep) return;
    selectedStep = next;
    followLive = next === c.liveStep;   // 手动挑了一步历史,就别再被实时步进拖走
    renderRuler(c);
    renderL2(c);
  }

  function setL2Collapsed(collapsed) {
    l2Collapsed = !!collapsed;
    if (!els) return;
    els.l2Body.hidden = l2Collapsed;
    els.tierL.classList.toggle("is-collapsed", l2Collapsed);
    els.l2Toggle.setAttribute("aria-expanded", String(!l2Collapsed));
    els.l2Toggle.title = l2Collapsed ? "展开 L2 阶段量尺" : "折叠 L2 阶段量尺";
    if (isOpen()) renderEvents(ctx());   // 折叠腾出来的高度让给 L1 的事件车道
  }

  /* ── ③ L2:选中那一步的三大阶段 ────────────────────────────────────
     沿用上面两层的填充语言:三段一律蓝 + 灰斜线,不另起三种颜色。选中的是正在跑的
     那一步时,前面的阶段已满、当前阶段从左往右涨、后面的阶段还全是斜线。 */
  function renderL2(c) {
    const model = costModel(c);
    const step = selectedStep == null ? c.liveStep : selectedStep;
    const bd = phaseBreakdown(step, model.base);
    const epoch = Math.floor(step / model.stepsPerEpoch) + 1;
    const inEpoch = (step % model.stepsPerEpoch) + 1;
    const ev = EVENTS.find((e) => e.step === step);
    const running = step === c.liveStep;

    els.l2Val.textContent = "step " + step.toLocaleString() + " · epoch " + epoch
      + " 第 " + inEpoch.toLocaleString() + " 步 · 单步 " + fmtSec(bd.total) + (running ? " · 进行中" : "");

    const notes = bd.notes.slice();
    if (ev) notes.unshift((TYPES[ev.type] || TYPES.event).label + " · " + ev.name);
    els.l2Notes.innerHTML = notes.map((t) => "<em>" + esc(t) + "</em>").join("");

    /* 三段各自处在「已进行 / 进行中 / 未进行」的哪一态:看进度线走到了哪一段。
       看历史步时 frac = 1,三段都是已进行。 */
    const frac = running ? liveFraction() : 1;
    els.phaseSpans = bd.phases.map((p) => p.pct);
    els.phaseLive = running;
    let cursor = 0;
    const marks = bd.phases.map((p) => {
      const from = cursor;
      cursor += p.pct;
      const prog = p.pct ? Math.max(0, Math.min(1, (frac - from) / p.pct)) : 0;
      return { prog, state: prog >= 1 ? "done" : (prog > 0 ? "running" : "pending") };
    });

    els.phases.innerHTML = bd.phases.map((p, i) => {
      const m = marks[i];
      return '<div class="tw-pano__ph is-' + m.state + '" style="width:' + (p.pct * 100).toFixed(3)
        + '%;--prog:' + m.prog.toFixed(4) + '" title="'
        + esc(p.name + " " + fmtSec(p.seconds) + " · " + (p.pct * 100).toFixed(1) + "%") + '">'
        + '<span class="tw-pano__ph-n">' + esc(p.name) + " " + fmtSec(p.seconds) + "</span></div>";
    }).join("");
    els.phaseEls = Array.prototype.slice.call(els.phases.querySelectorAll(".tw-pano__ph"));

    // 还没跑到的阶段没有可报的数,只留 key 不填 value —— 不拿模型预估冒充实测
    els.l2Cards.innerHTML = bd.phases.map((p, i) => {
      const st = marks[i].state;
      const blank = st === "pending";
      const rows = p.segs.map((s) =>
        '<div class="tw-pano__kv"><span>' + esc(s.name) + "</span><b>"
        + (blank ? "—" : fmtSec(s.seconds) + " · " + ((s.seconds / p.seconds) * 100).toFixed(0) + "%")
        + "</b></div>").join("");
      return '<div class="tw-pano__l2-card is-' + st + '">'
        + '<div class="tw-pano__l2-card-head"><span class="tw-pano__l2-card-name">' + esc(p.name) + "</span>"
        + (st === "running" ? '<span class="tw-pano__l2-card-tag">进行中</span>' : "")
        + '<span class="tw-pano__l2-card-sec">' + (blank ? "—" : fmtSec(p.seconds)) + "</span>"
        + '<em class="tw-pano__l2-card-pct">' + (blank ? "" : (p.pct * 100).toFixed(1) + "%") + "</em></div>"
        + '<div class="tw-pano__l2-kv">' + rows + "</div></div>";
    }).join("");
  }

  /* 心跳:进行中的那一格/那一段往前涨(只改 --prog,不重建 DOM);
     liveStep 一变 = 这一步真的跑完了,整块重排,进行中的格子自然挪到下一格。 */
  function tickProgress() {
    if (!isOpen() || !els) return;
    const c = ctx();
    observeLive(c.liveStep);
    if (c.liveStep !== lastLive) {
      lastLive = c.liveStep;
      if (followLive) selectedStep = c.liveStep;
      render();
      return;
    }
    const f = liveFraction();
    if (els.liveCellEl) els.liveCellEl.style.setProperty("--prog", f.toFixed(4));
    if (els.phaseLive && els.phaseEls.length) {
      let start = 0;
      let changed = false;
      els.phaseEls.forEach((el, i) => {
        const w = els.phaseSpans[i] || 0;
        const prog = w ? Math.max(0, Math.min(1, (f - start) / w)) : 0;
        start += w;
        el.style.setProperty("--prog", prog.toFixed(4));
        const st = prog >= 1 ? "done" : (prog > 0 ? "running" : "pending");
        if (!el.classList.contains("is-" + st)) changed = true;
        el.classList.toggle("is-done", st === "done");
        el.classList.toggle("is-running", st === "running");
        el.classList.toggle("is-pending", st === "pending");
      });
      // 阶段换了 = 下面三张卡的三态与 value 该重排了(进行中的阶段才填得出数)
      if (changed) renderL2(c);
    }
  }

  function render() {
    const c = ctx();
    renderKpis(c);
    renderRuler(c);
    renderL2(c);
    renderFilters();
    renderTicks(c);
    renderAxis(c);
    renderEvents(c);   // 车道预算要量上面几段的实际高度,必须放在最后
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
    selectedStep = null;   // 每次展开都默认选中最新一个 step
    followLive = false;
    position();
    render();
    document.addEventListener("keydown", onKeydown, true);
    window.addEventListener("resize", onResize);
    lastLive = ctx().liveStep;
    if (progTimer) clearInterval(progTimer);
    progTimer = setInterval(tickProgress, 250);
    $("progressTopbar")?.setAttribute("aria-expanded", "true");
  }

  function close() {
    if (!isOpen()) return;
    root.hidden = true;
    document.removeEventListener("keydown", onKeydown, true);
    window.removeEventListener("resize", onResize);
    if (progTimer) { clearInterval(progTimer); progTimer = 0; }
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
      const c = ctx();
      position();
      renderRuler(c);
      renderL2(c);
      renderEvents(c);
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
