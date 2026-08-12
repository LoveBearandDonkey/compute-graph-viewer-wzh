// 日志:右上角 #trainLogToggle 打开,默认收起。并入底部 Timeline dock 第四个页签
// (见 training-monitoring-v2.html 里的 #dockPanelLog),打开即展开该 dock 并切到这个页签。
// 日志数据是本页事故场景(问题一 router FP8 溢出触发 loss NaN、问题三 q_proj 溢出、问题五 HCCS 掉链路)
// 的固定重演脚本,时间线/step 与 js/training-run-twin.js 里的 INCIDENT_STEP=15203、diagnosisMarkers
// 保持一致,不是随机生成;SQL 搜索是前端对这份静态数组做的简化条件解析,不接真实查询引擎。
(function () {
  const $ = (id) => document.getElementById(id);

  // comp -> 所属页签("task"=训练任务本身，"system"=集群/硬件基础设施)
  const TASK_COMPONENTS = new Set(["trainer", "dataloader", "ckpt", "eval", "router", "oncall"]);
  const SYSTEM_COMPONENTS = new Set(["scheduler", "npu-driver", "network", "node-health", "hccl"]);

  // t: "YYYY-MM-DD HH:mm:ss.SSS"；step: 关联的训练 step（用于 SQL 的 step 比较，无关联时为 null）
  const LOG_DATA = [
    { t: "2026-07-16 08:00:03.120", level: "INFO", comp: "scheduler", step: null, msg: "Volcano job pangu20flash-pretrain-7f3a2 submitted, requested npu=2048 (256 nodes × 8 Ascend 910B)" },
    { t: "2026-07-16 08:02:47.884", level: "INFO", comp: "scheduler", step: null, msg: "pods scheduled 256/256, waiting for NPU driver ready" },
    { t: "2026-07-16 08:04:12.033", level: "INFO", comp: "npu-driver", step: null, msg: "CANN 8.0.RC2 / driver 24.1.0 initialized on all 2048 devices" },
    { t: "2026-07-16 08:05:01.442", level: "INFO", comp: "network", step: null, msg: "HCCL rank_table_file loaded, 256 server groups, RoCEv2 fallback enabled" },
    { t: "2026-07-16 08:06:33.219", level: "INFO", comp: "trainer", step: null, msg: "init process group backend=hccl world_size=2048 rank=0 elapsed=91.2s" },
    { t: "2026-07-16 08:07:02.510", level: "INFO", comp: "dataloader", step: null, msg: "dataset shards resolved: 512 shards × 2048 samples, tokenizer=pangu-bpe-128k" },
    { t: "2026-07-16 08:07:45.771", level: "INFO", comp: "ckpt", step: null, msg: "no existing checkpoint under obs://pangu-ckpt/2.0-flash/, cold start from init weights" },
    { t: "2026-07-16 08:08:10.005", level: "INFO", comp: "trainer", step: 0, msg: "training loop started model=Pangu 2.0 flash task=pretrain total_steps=120000 tp=4 pp=8 ep=64 cp=2" },
    { t: "2026-07-16 08:12:44.332", level: "INFO", comp: "trainer", step: 100, msg: "step 100 | loss=8.214 grad_norm=3.11 lr=5.0e-6 mfu=0.410 tokens/s=298442" },
    { t: "2026-07-16 09:41:20.771", level: "INFO", comp: "trainer", step: 2000, msg: "step 2000 | loss=5.732 grad_norm=2.04 lr=8.4e-5 mfu=0.552 tokens/s=451820" },
    { t: "2026-07-16 13:58:02.410", level: "INFO", comp: "trainer", step: 8000, msg: "step 8000 | loss=3.918 grad_norm=1.63 lr=1.2e-4 mfu=0.579 tokens/s=479330" },
    { t: "2026-07-16 15:03:41.117", level: "WARN", comp: "router", step: 8500, msg: "layer33.q_proj input activation 3.2% of tokens exceed FP8 E4M3 max(448), auto-clamped" },
    { t: "2026-07-16 15:03:41.902", level: "WARN", comp: "trainer", step: 8500, msg: "step 8500 | loss=3.774 grad_norm=2.87(↑ from 1.63) lr=1.2e-4 mfu=0.561 — 已记入问题跟踪(问题三)" },
    { t: "2026-07-16 17:26:24.556", level: "INFO", comp: "ckpt", step: 8500, msg: "checkpoint step_8500 uploaded to obs://pangu-ckpt/2.0-flash/step_8500 (118.7GB, 214s)" },
    { t: "2026-07-16 22:14:09.203", level: "INFO", comp: "trainer", step: 14000, msg: "step 14000 | loss=3.201 grad_norm=1.58 lr=1.2e-4 mfu=0.583 tokens/s=483110" },
    { t: "2026-07-16 23:10:02.114", level: "WARN", comp: "router", step: 15100, msg: "layer38.moe.router logits max drifting upward: p99=402.6 (E4M3 max 448), approaching saturation" },
    { t: "2026-07-16 23:14:18.667", level: "WARN", comp: "router", step: 15150, msg: "expert load imbalance detected: top1 expert(193) share=41.2% (threshold 25%)" },
    { t: "2026-07-16 23:17:02.330", level: "WARN", comp: "trainer", step: 15200, msg: "step 15200 | loss=3.14 grad_norm=2.87(↑ from 1.58) mfu=0.560 — 波动加剧" },
    { t: "2026-07-16 23:17:52.014", level: "ERROR", comp: "router", step: 15202, msg: "FP8 E4M3 softmax overflow at layer38.moe.router, logits max=512.7 (> E4M3 max 448), saturating to inf", mapKeys: ["aclnnSoftmaxV2"] },
    { t: "2026-07-16 23:17:53.228", level: "ERROR", comp: "router", step: 15202, msg: "token routing collapse: 98.3% tokens routed to expert193(EP rank23), capacity_factor exceeded, dropped_tokens=812441" },
    { t: "2026-07-16 23:18:00.406", level: "ERROR", comp: "hccl", step: 15203, msg: "HcclAllToAllV timeout on comm_group ep_group_3, rank=23 peer=45, elapsed=120000ms(> timeout 120000ms)", mapKeys: ["hcom_all_to_all_v_"], defaultSelect: true },
    { t: "2026-07-16 23:18:00.777", level: "ERROR", comp: "trainer", step: 15203, msg: "step 15203 | loss=nan grad_norm=inf — abort optimizer.step(), tensors dumped to /tmp/anomaly_dump/step_15203", focus: true },
    { t: "2026-07-16 23:18:01.115", level: "ERROR", comp: "npu-driver", step: 15203, msg: "device rank23(node2 GPU7) AICORE task timeout on stream14, HCCL watchdog killed pid=88213" },
    { t: "2026-07-16 23:18:01.560", level: "ERROR", comp: "scheduler", step: 15203, msg: "pod pangu20flash-pretrain-7f3a2-worker-23 CrashLoopBackOff restartCount=1" },
    { t: "2026-07-16 23:18:32.220", level: "WARN", comp: "node-health", step: 15203, msg: "node2 GPU7 temperature=78°C, ECC errors=0 — 硬件自检未见异常，判断为数值溢出触发的软件故障" },
    { t: "2026-07-16 23:18:50.000", level: "INFO", comp: "oncall", step: 15203, msg: "plog 翻译：grep -i 'error|timeout|mismatch' plog_*.log → [hcom_all_to_all_v_] rank=23 send_count=0 recv_count=9832 buffer size mismatch；[aclnnSoftmaxV2] input[router_logits] contains inf — 点选本行对照右侧接口映射", mapKeys: ["hcom_all_to_all_v_", "aclnnSoftmaxV2"] },
    { t: "2026-07-16 23:19:10.004", level: "INFO", comp: "trainer", step: 15203, msg: "auto-recovery triggered: rollback to last stable checkpoint step_15200" },
    { t: "2026-07-16 23:21:42.881", level: "INFO", comp: "ckpt", step: 15200, msg: "restored optimizer/model state from obs://pangu-ckpt/2.0-flash/step_15200" },
    { t: "2026-07-16 23:22:03.556", level: "INFO", comp: "scheduler", step: 15203, msg: "worker-23 pod restarted, rejoined process group, rank23 healthy" },
    { t: "2026-07-16 23:22:30.017", level: "INFO", comp: "trainer", step: 15200, msg: "resuming training from step 15200, skipping corrupt optimizer state at step 15203" },
    { t: "2026-07-16 23:28:14.442", level: "INFO", comp: "trainer", step: 15233, msg: "step 15233 | loss=3.982 grad_norm=6.11(恢复期) mfu=0.402" },
    { t: "2026-07-16 23:50:07.330", level: "INFO", comp: "trainer", step: 15773, msg: "step 15773 | loss=2.910 grad_norm=2.44 mfu=0.498(恢复中)" },
    { t: "2026-07-17 03:47:52.660", level: "INFO", comp: "trainer", step: 18500, msg: "step 18500 | loss=2.845 grad_norm=1.49 lr=1.2e-4 mfu=0.586 tokens/s=485224" },
    { t: "2026-07-17 06:11:40.018", level: "WARN", comp: "network", step: 20000, msg: "node002 NPU3 HCCS lane5 link flapping, retry 1/3" },
    { t: "2026-07-17 06:12:00.447", level: "ERROR", comp: "network", step: 20000, msg: "node002 NPU3 HCCS lane5 inactive, HCCL fallback to RoCE slow path for comm_group pp_group_2" },
    { t: "2026-07-17 06:12:00.981", level: "WARN", comp: "trainer", step: 20000, msg: "step 20000 | mfu=0.312(↓ from 0.586), throughput degraded on pp_group_2 — 已记入问题跟踪(问题五)" },
    { t: "2026-07-17 06:14:55.302", level: "INFO", comp: "node-health", step: 20000, msg: "node002 NPU3 physical link diagnostics dispatched to infra on-call" },
    { t: "2026-07-17 06:20:31.774", level: "INFO", comp: "network", step: 20010, msg: "HCCS lane5 recovered after port reset, HCCL comm_group pp_group_2 rebuilt" },
    { t: "2026-07-17 06:21:02.115", level: "INFO", comp: "trainer", step: 20050, msg: "step 20050 | mfu=0.581(recovered) tokens/s=480117" },
    { t: "2026-07-17 06:35:18.220", level: "INFO", comp: "trainer", step: 21000, msg: "step 21000 | loss=2.245 grad_norm=1.43 lr=1.2e-4 mfu=0.583 tokens/s=482900" },
  ];

  // 接口映射表(右栏,静态常驻):Ascend C / plog 内部名 → torch_npu / PyTorch 可见接口。
  // 与「问题一详情与修复建议」抽屉里定位链「日志/plog诊断层」那张表同源,这里复用而不是另写一份;
  // LOG_DATA 里带 mapKeys 的行(见上)点选后会高亮对应 id 的行,把"看到原始报错"和"知道它是什么"接上,
  // 因此原始日志行本身不再需要额外拼一句人工翻译尾注。
  const MAPPING_TABLE = [
    { id: "hcom_all_to_all_v_", ascend: "hcom_all_to_all_v_", torch: "dist.all_to_all", desc: "通信库" },
    { id: "aclnnSoftmaxV2", ascend: "aclnnSoftmaxV2", torch: "F.softmax", desc: "router.forward 中调用" },
    { id: "aclnnMatmulV3", ascend: "aclnnMatmulV3", torch: "F.linear", desc: "router 的 Linear 层" },
  ];

  let activeTab = "all"; // all | task | system
  // 不预置搜索条件,默认展示全量日志;自动滚到关键报错行的逻辑见 scrollToRelevantRow,
  // 不需要靠预置查询语句来"提前过滤出"事故行。
  let activeQuery = "";

  function compTab(comp) {
    if (TASK_COMPONENTS.has(comp)) return "task";
    if (SYSTEM_COMPONENTS.has(comp)) return "system";
    return "task";
  }

  function escHtml(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // ── 简化版 SQL WHERE 解析:支持 level/comp/message/step 列, = != > >= < <= LIKE IN 运算符,
  //    多条件用 AND 连接。解析不出任何列名时整体退化为对 "时间+级别+组件+消息" 的大小写不敏感子串匹配。──
  function parseQuery(raw) {
    const q = raw.trim();
    if (!q) return null;
    const body = q.replace(/^\s*SELECT\s+\*\s+FROM\s+logs\s*/i, "").replace(/^\s*WHERE\s+/i, "");
    const clauses = body.split(/\s+AND\s+/i).map((c) => c.trim()).filter(Boolean);

    const conds = [];
    let recognized = true;
    for (const clause of clauses) {
      let m;
      if ((m = clause.match(/^(\w+)\s+(NOT\s+)?LIKE\s+'?%?([^%']*)%?'?$/i))) {
        conds.push({ col: m[1].toLowerCase(), op: m[2] ? "not-like" : "like", val: m[3] });
      } else if ((m = clause.match(/^(\w+)\s+(NOT\s+)?IN\s*\(([^)]*)\)$/i))) {
        const vals = m[3].split(",").map((s) => s.trim().replace(/^'|'$/g, "").toLowerCase());
        conds.push({ col: m[1].toLowerCase(), op: m[2] ? "not-in" : "in", val: vals });
      } else if ((m = clause.match(/^(\w+)\s*(>=|<=|!=|>|<|=)\s*'?([^']*)'?$/))) {
        conds.push({ col: m[1].toLowerCase(), op: m[2], val: m[3].trim() });
      } else {
        recognized = false;
      }
    }
    if (!recognized || !conds.length) {
      return { fallback: q.toLowerCase() };
    }
    return { conds };
  }

  function rowMatchesConds(row, conds) {
    return conds.every((c) => {
      const col = c.col === "message" || c.col === "msg" ? "msg" : c.col;
      if (col === "level") {
        const v = row.level.toLowerCase();
        if (c.op === "in") return c.val.includes(v);
        if (c.op === "not-in") return !c.val.includes(v);
        if (c.op === "=") return v === String(c.val).toLowerCase();
        if (c.op === "!=") return v !== String(c.val).toLowerCase();
        return false;
      }
      if (col === "comp" || col === "component") {
        const v = row.comp.toLowerCase();
        if (c.op === "in") return c.val.includes(v);
        if (c.op === "not-in") return !c.val.includes(v);
        if (c.op === "=") return v === String(c.val).toLowerCase();
        if (c.op === "!=") return v !== String(c.val).toLowerCase();
        if (c.op === "like") return v.includes(String(c.val).toLowerCase());
        return false;
      }
      if (col === "step") {
        if (row.step == null) return false;
        const n = Number(c.val);
        if (Number.isNaN(n)) return false;
        if (c.op === "=") return row.step === n;
        if (c.op === "!=") return row.step !== n;
        if (c.op === ">") return row.step > n;
        if (c.op === ">=") return row.step >= n;
        if (c.op === "<") return row.step < n;
        if (c.op === "<=") return row.step <= n;
        return false;
      }
      if (col === "msg") {
        const v = row.msg.toLowerCase();
        if (c.op === "like") return v.includes(String(c.val).toLowerCase());
        if (c.op === "not-like") return !v.includes(String(c.val).toLowerCase());
        return v.includes(String(c.val).toLowerCase());
      }
      return true; // 无法识别的列名不参与过滤,避免因笔误把结果清空
    });
  }

  function filteredRows() {
    let rows = LOG_DATA.filter((r) => activeTab === "all" || compTab(r.comp) === activeTab);
    const parsed = parseQuery(activeQuery);
    if (!parsed) return rows;
    if (parsed.fallback) {
      const kw = parsed.fallback;
      rows = rows.filter((r) => (r.t + " " + r.level + " " + r.comp + " " + r.msg).toLowerCase().includes(kw));
    } else {
      rows = rows.filter((r) => rowMatchesConds(r, parsed.conds));
    }
    return rows;
  }

  function renderStatus(rows) {
    const el = $("trainLogStatus");
    if (!el) return;
    const errCount = rows.filter((r) => r.level === "ERROR").length;
    const warnCount = rows.filter((r) => r.level === "WARN").length;
    el.innerHTML =
      "共 <b>" + rows.length + "</b> 条" +
      (errCount ? " · <span class=\"lvl-error\">ERROR " + errCount + "</span>" : "") +
      (warnCount ? " · <span class=\"lvl-warn\">WARN " + warnCount + "</span>" : "") +
      "<span class=\"wzh-log-status-live\"><i></i>已同步至最新 step</span>";
  }

  function renderBody() {
    const body = $("trainLogBody");
    if (!body) return;
    const rows = filteredRows();
    renderStatus(rows);
    highlightMapping([]); // 页签/搜索一变,之前点选高亮的映射行跟着失效,先清空避免留着一份对不上的高亮
    if (!rows.length) {
      body.innerHTML = "<div class=\"wzh-log-empty\">没有匹配的日志，试试清空搜索条件或切换页签。</div>";
      return;
    }
    body.innerHTML = rows.map((r) => {
      const full = r.t + "  [" + r.level + "]  " + r.comp + "  " + r.msg;
      const cls = "wzh-log-row lvl-" + r.level + (r.focus ? " is-focus-row" : "") + (r.defaultSelect ? " is-selected" : "");
      const mapAttr = r.mapKeys && r.mapKeys.length ? " data-map-keys=\"" + r.mapKeys.join(",") + "\"" : "";
      return (
        "<div class=\"" + cls + "\"" + mapAttr + " title=\"" + escHtml(full) + "\">" +
          "<span class=\"wzh-log-col-time\">" + escHtml(r.t) + "</span>" +
          "<span class=\"wzh-log-col-level lvl-" + r.level + "\">" + r.level + "</span>" +
          "<span class=\"wzh-log-col-comp\">" + escHtml(r.comp) + "</span>" +
          "<span class=\"wzh-log-col-msg\">" + escHtml(r.msg) + "</span>" +
        "</div>"
      );
    }).join("");
    // 默认选中 HcclAllToAllV timeout 行(与 loss=nan 事故行相邻一步之遥),点亮右栏映射表,
    // 免得用户打开日志面板还得自己点一下才知道报错和映射表是对上的。
    const defaultRow = LOG_DATA.find((r) => r.defaultSelect);
    if (defaultRow && rows.includes(defaultRow)) {
      highlightMapping(defaultRow.mapKeys || []);
    }
    scrollToRelevantRow(body);
  }

  // 右栏「接口映射表」:静态渲染一次即可,不随日志筛选变化。
  function renderMapping() {
    const list = $("trainLogMappingList");
    if (!list) return;
    list.innerHTML = MAPPING_TABLE.map((m) => (
      "<div class=\"wzh-log-mapping-row\" data-map-id=\"" + escHtml(m.id) + "\">" +
        "<div class=\"wzh-log-mapping-names\">" +
          "<span class=\"wzh-log-mapping-ascend\">" + escHtml(m.ascend) + "</span>" +
          "<span class=\"wzh-log-mapping-arrow\">→</span>" +
          "<span class=\"wzh-log-mapping-torch\">" + escHtml(m.torch) + "</span>" +
        "</div>" +
        "<div class=\"wzh-log-mapping-desc\">" + escHtml(m.desc) + "</div>" +
      "</div>"
    )).join("");
  }

  // 高亮右栏里 id 落在 keys 内的映射行;keys 为空即清空高亮。
  function highlightMapping(keys) {
    const list = $("trainLogMappingList");
    if (!list) return;
    list.querySelectorAll(".wzh-log-mapping-row").forEach((el) => {
      el.classList.toggle("is-active", keys.includes(el.dataset.mapId));
    });
  }

  // 点选日志行(而非 hover,便于停留细看):高亮该行 + 联动右栏对应映射行;
  // 没有 data-map-keys 的行(大多数日志)点选后只留选中态,右栏清空,不误导用户去找不存在的映射。
  function initRowSelection() {
    const body = $("trainLogBody");
    if (!body) return;
    body.addEventListener("click", (e) => {
      const row = e.target.closest(".wzh-log-row");
      if (!row) return;
      body.querySelectorAll(".wzh-log-row.is-selected").forEach((r) => r.classList.remove("is-selected"));
      row.classList.add("is-selected");
      const keys = (row.dataset.mapKeys || "").split(",").filter(Boolean);
      highlightMapping(keys);
    });
  }

  // 自动滚动定位:优先跳到本次事故的关键报错行(问题一 loss=nan,见 LOG_DATA 里的 focus:true),
  // 该行被当前页签/搜索条件过滤掉时退化为跳到筛选结果里第一条 ERROR;完全没有 ERROR 时按
  // "实时日志尾随"惯例滚到最新一条。命中的行加一次红色脉冲高亮,把视线引导过去而不是让用户自己找。
  function scrollToRelevantRow(body) {
    const target = body.querySelector(".wzh-log-row.is-focus-row") || body.querySelector(".wzh-log-row.lvl-ERROR");
    if (!target) { body.scrollTop = body.scrollHeight; return; }
    target.scrollIntoView({ block: "center" });
    target.classList.add("is-flash");
    setTimeout(() => target.classList.remove("is-flash"), 1700);
  }

  function initTabs() {
    const seg = $("trainLogTabSeg");
    if (!seg) return;
    seg.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-log-tab]");
      if (!btn) return;
      activeTab = btn.dataset.logTab;
      seg.querySelectorAll(".segbtn").forEach((b) => {
        b.classList.toggle("on", b === btn);
        b.setAttribute("aria-selected", String(b === btn));
      });
      renderBody();
    });
  }

  function initSearch() {
    const input = $("trainLogSearchInput");
    const searchBtn = $("trainLogSearchBtn");
    const clearBtn = $("trainLogSearchClearBtn");
    if (!input) return;
    input.value = activeQuery; // 搜索框回显默认查询,和 activeQuery 保持同一份状态
    function runSearch() { activeQuery = input.value; renderBody(); }
    searchBtn?.addEventListener("click", runSearch);
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") runSearch(); });
    clearBtn?.addEventListener("click", () => { input.value = ""; activeQuery = ""; renderBody(); });
  }

  function getTrainingContext() {
    return typeof window.twinGetTrainingContext === "function" ? window.twinGetTrainingContext() : null;
  }

  function fmtContextLabel(ctx) {
    if (!ctx) return "训练态未就绪";
    const pct = ctx.totalSteps ? ((ctx.step / ctx.totalSteps) * 100).toFixed(1) : "--";
    return ctx.model.name + " · step " + ctx.step.toLocaleString() + "/" + ctx.totalSteps.toLocaleString() + "（" + pct + "%）";
  }

  function renderContextLabel() {
    const label = $("trainLogContext");
    if (label) label.textContent = fmtContextLabel(getTrainingContext());
  }

  // 日志页签现在是底部 Timeline dock(#bottomDock)的第四个页签,不再是独立浮层:
  // 「打开」= 展开 dock(若已收起) + 切到日志页签;「关闭」= 收起整个 dock(与原抽屉的
  // 完全隐藏行为对齐)。dock 展开/收起、页签切换分别由 training-monitoring-v2.html 里
  // window.PtoTrainingTwinTimelineDock / window.PtoTrainingTwinDockTabs 提供。
  function initPanelToggle() {
    const toggle = $("trainLogToggle");
    const panel = $("dockPanelLog");
    const dock = $("bottomDock");
    if (!toggle || !panel || !dock) return;

    function isActive() {
      return !dock.hidden && !panel.hidden;
    }

    function syncToggle() {
      const active = isActive();
      toggle.classList.toggle("is-active", active);
      toggle.setAttribute("aria-expanded", String(active));
      toggle.setAttribute("aria-pressed", String(active));
      toggle.title = active ? "关闭日志" : "打开日志";
      toggle.setAttribute("aria-label", toggle.title);
    }

    function open() {
      window.PtoTrainingTwinTimelineDock?.setVisible(true);
      window.PtoTrainingTwinDockTabs?.select("log");
      renderContextLabel();
      renderBody();
      syncToggle();
    }

    function close() {
      window.PtoTrainingTwinTimelineDock?.setVisible(false);
      syncToggle();
    }

    toggle.addEventListener("click", () => (isActive() ? close() : open()));
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && isActive()) close();
    });
    // 手动点别的页签(Timeline/制品/事件流)离开日志页时,入口按钮的高亮态要跟着退出;
    // 关键:直接点「日志」页签(不经 #trainLogToggle 的 open())也要渲染日志内容,
    // 否则从 dock 页签进入日志页时,面板是空的(渲染只发生在 open() 里)。
    document.querySelector(".twin-dock-tablist")?.addEventListener("click", (e) => {
      syncToggle();
      const tab = e.target.closest && e.target.closest(".twin-dock-tab");
      if (tab && tab.dataset.dockTab === "log") {
        renderContextLabel();
        renderBody();
      }
    });
    // dock 本身还能被顶栏 #twinTimelineToggle、dock 头部 #bottomDockCloseBtn 等其它入口
    // 显示/隐藏,不经过这里的 open()/close();用 MutationObserver 盯 hidden 属性,
    // 不管从哪条路径关闭都能让「打开日志」按钮的高亮态跟着同步。
    new MutationObserver(syncToggle).observe(dock, { attributes: true, attributeFilter: ["hidden"] });

    syncToggle(); // dock 初始就选中日志页签,按钮高亮态取决于 dock 本身是否展开
  }

  function boot() {
    initTabs();
    initSearch();
    initPanelToggle();
    initRowSelection();
    renderMapping();
    // 日志现在是底部 dock 的默认页签(页面加载即可见),必须在这里就渲染一次。
    // 原来内容只在 open() / 点「日志」页签时才渲染(懒加载,因为那时默认页签是 Timeline),
    // 默认页签改成日志后那套策略会让面板一进来就是空的。
    renderContextLabel();
    renderBody();
    // 供外部（如聚光灯定位链 js/training-spotlight.js 的「日志」步）幂等地展开并渲染日志：
    // 展开 dock + 切到日志页签 + 刷新一次内容（页签本已默认选中时也要重渲，取最新训练态）。
    window.PtoTrainingLogDrawer = {
      show: function () {
        window.PtoTrainingTwinTimelineDock?.setVisible(true);
        window.PtoTrainingTwinDockTabs?.select("log");
        renderContextLabel();
        renderBody();
      },
      render: renderBody,
    };
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }
})();
