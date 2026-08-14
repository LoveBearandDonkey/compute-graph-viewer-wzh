(async function initializeReportWorkbench() {
  "use strict";

  const els = {
    ideFrame: document.getElementById("ideFrame"),
    workspaceTitle: document.getElementById("workspaceTitle"),
    nodeList: document.getElementById("nodeList"),
    architectureGraph: document.getElementById("architectureGraph"),
    architectureGraphPanel: document.getElementById("architectureGraphPanel"),
    operatorTreePanel: document.getElementById("operatorTreePanel"),
    operatorColorLegendMax: document.getElementById("operatorColorLegendMax"),
    architectureViewTab: document.getElementById("architectureViewTab"),
    operatorListViewTab: document.getElementById("operatorListViewTab"),
    nodeViewsRailButton: document.getElementById("nodeViewsRailButton"),
    inspectorTitle: document.getElementById("inspectorTitle"),
    inspectorNodeId: document.getElementById("inspectorNodeId"),
    inspectorSummary: document.getElementById("inspectorSummary"),
    coreMetricGrid: document.getElementById("coreMetricGrid"),
    metricGrid: document.getElementById("metricGrid"),
    fieldTooltip: document.getElementById("reportFieldTooltip"),
    factList: document.getElementById("factList"),
    operatorList: document.getElementById("operatorList"),
    timelineCaption: document.getElementById("timelineCaption"),
    traceToolbarMount: document.getElementById("traceToolbarMount"),
    timelineTabTrace: document.getElementById("timelineTabTrace"),
    timelineTabSteps: document.getElementById("timelineTabSteps"),
    traceTimelinePanel: document.getElementById("traceTimelinePanel"),
    stepTimelinePanel: document.getElementById("stepTimelinePanel"),
    streamTimelinePanel: document.getElementById("streamTimelinePanel"),
    hbmTimelinePanel: document.getElementById("hbmTimelinePanel"),
    streamZoomControls: document.getElementById("streamZoomControls"),
    streamZoomOut: document.getElementById("streamZoomOut"),
    streamZoomReset: document.getElementById("streamZoomReset"),
    streamZoomIn: document.getElementById("streamZoomIn"),
    inspectorPane: document.getElementById("reportInspectorPane"),
    inspectorToggle: document.getElementById("inspectorToggle"),
    inspectorClose: document.getElementById("inspectorClose"),
    languageToggle: document.getElementById("languageToggle"),
    languageToggleLabel: document.getElementById("languageToggleLabel"),
    themeToggle: document.getElementById("themeToggle"),
    themeToggleIcon: document.getElementById("themeToggleIcon"),
    bottomPanelToggle: document.getElementById("bottomPanelToggle"),
    bottomDock: document.getElementById("reportBottomDock"),
    footerStatus: document.getElementById("footerStatus"),
  };

  const state = {
    selectedNodeId: "",
    selectedArchitectureId: "",
    activeArchitectureView: "architecture",
    activeTimelineTab: "trace",
    activeTimelineSegment: -1,
    architectureController: null,
    architectureViewGraph: null,
    repeatInstanceSelections: new Map(),
    pendingArchitectureCenterNodeId: "",
    collapsedArchitectureIds: new Set(),
    visibleArchitectureIds: new Set(),
    operatorTreeExpandedIds: new Set(),
    streamTooltip: null,
    streamResizeTimer: 0,
    streamResizeObserver: null,
    streamZoomIndex: 0,
    traceController: null,
    traceVisibleRange: null,
    hbmFollowsTrace: false,
    hbmEventFocus: false,
    hbmController: null,
    tooltipTrigger: null,
    bottomPanelExpanded: true,
    language: document.documentElement.lang.startsWith("zh") ? "zh" : "en",
    theme: document.documentElement.dataset.theme === "light" ? "light" : "dark",
  };

  const I18N = {
    en: {
      performanceNodes: "Performance Nodes",
      fullModelArchitecture: "Full Model Architecture",
      modelArchitectureView: "Model Architecture",
      operatorListView: "Operator List",
      nodeViews: "Node views",
      stagesGroup: "Model stages",
      layersGroup: "Decoder layers",
      runtimeGroup: "Runtime auxiliary",
      layerSelector: "Layer",
      selectLayer: (index) => `Select decoder layer; current selection is Layer ${index}`,
      aggregateTimeShareHint: (label, value) => `${label}: ${value} is the aggregate time share across all included layers or events. It is a total, not a single-layer hotspot or severity warning.`,
      expand: "Expand",
      collapse: "Collapse",
      operatorTreeStatus: (count) => `${count} backend nodes · hierarchical view`,
      operatorColorLegend: "Operator time-share color",
      operatorColorLegendAria: "Operator time-share color legend",
      inspector: "Inspector",
      coreEventMetrics: "Core Event Metrics",
      performanceMetrics: "Performance Metrics",
      evidence: "Evidence",
      operators: "Operators",
      operatorsDeduplicatedHint: "Deduplicated by operator name within the current node's metric_scope. Each name appears once; the percentage is the aggregated time share for all operators with that name.",
      stepStreamTimeline: "Trace / Step Timeline",
      streams: "Streams",
      traceView: "TraceView",
      timelineZoom: "Timeline zoom",
      zoomOut: "Zoom out",
      zoomIn: "Zoom in",
      resetZoom: "Reset timeline zoom",
      executionLane: "Execution lane",
      laneTotalsHeader: "Lane totals",
      workbenchPanels: "Workbench panels",
      showBottomPanel: "Show bottom panel",
      hideBottomPanel: "Hide bottom panel",
      noSelection: "No selection",
      selectBackendNode: "Select a backend node",
      noNodeId: "No node_id selected",
      selectHint: "Choose a node in the architecture view, operator list, or a mapped timeline event.",
      noMetrics: "No metrics selected.",
      noEvidence: "No evidence selected.",
      noOperatorRatio: "No operator ratio selected.",
      noOperatorRatioForNode: "No operator ratio is present for this node.",
      switchToChinese: "切换到中文",
      switchToEnglish: "Switch to English",
      switchToLight: "Switch to light mode",
      switchToDark: "Switch to dark mode",
      reportTitle: "Profiling Report",
      nodesCount: (count) => `${count} nodes`,
      graphAria: (model) => `${model} complete source architecture with backend performance overlay`,
      decodeLatency: "decode latency",
      kernelSum: "kernel sum",
      eventMapping: "event mapping",
      globalMfu: "global MFU INT8",
      mappedEvent: "Mapped event",
      unmappedEvent: "Unmapped event",
      representative: "representative",
      mappedCount: (mapped, total) => `${mapped} / ${total} mapped`,
      stepCaption: (step) => `Step ${step} latency and direct event mapping summary`,
      streamCaption: (events, lanes) => `${events} raw events grouped into ${lanes} device / stream / core lanes`,
      traceCaption: (events, lanes, range) => `${events} duration events on ${lanes} raw profiler tracks · ${range}`,
      events: (count) => `${count} events`,
      laneTotals: (duration, wait) => `duration Σ ${duration} · raw wait Σ ${wait}`,
      footerStatus: (nodes, events, selection) => `${nodes} nodes · ${events} events · ${selection || "no selection"}`,
      noSelectionShort: "no selection",
      appFailed: "application failed",
      architectureLoadFailed: "architecture data failed to load",
    },
    zh: {
      performanceNodes: "性能节点",
      fullModelArchitecture: "完整模型架构",
      modelArchitectureView: "模型架构",
      operatorListView: "算子列表",
      nodeViews: "节点视图",
      stagesGroup: "模型阶段",
      layersGroup: "解码层",
      runtimeGroup: "运行时辅助",
      layerSelector: "层",
      selectLayer: (index) => `选择解码层；当前为第 ${index} 层`,
      aggregateTimeShareHint: (label, value) => `${label}：${value} 是所含全部层或事件的聚合总耗时占比，表示总量，不代表单层热点或异常等级。`,
      expand: "展开",
      collapse: "收起",
      operatorTreeStatus: (count) => `${count} 个后端节点 · 层级视图`,
      operatorColorLegend: "算子耗时占比着色",
      operatorColorLegendAria: "算子耗时占比颜色图例",
      inspector: "检查器",
      coreEventMetrics: "核心事件指标",
      performanceMetrics: "性能指标",
      evidence: "证据",
      operators: "算子",
      operatorsDeduplicatedHint: "按当前节点 metric_scope 内的算子名称去重汇总；同名算子只展示一次，百分比是该名称下全部算子的合计时间占比。",
      stepStreamTimeline: "Trace / 单步时间线",
      streams: "泳道",
      traceView: "TraceView",
      timelineZoom: "时间线缩放",
      zoomOut: "缩小",
      zoomIn: "放大",
      resetZoom: "重置时间线缩放",
      executionLane: "执行泳道",
      laneTotalsHeader: "泳道汇总",
      workbenchPanels: "工作台面板",
      showBottomPanel: "显示底部面板",
      hideBottomPanel: "隐藏底部面板",
      noSelection: "未选择",
      selectBackendNode: "请选择后端节点",
      noNodeId: "未选择 node_id",
      selectHint: "请在模型架构、算子列表或已映射的时间线事件中选择节点。",
      noMetrics: "未选择性能指标。",
      noEvidence: "未选择证据。",
      noOperatorRatio: "未选择算子占比。",
      noOperatorRatioForNode: "后端未提供该节点的算子占比。",
      switchToChinese: "切换到中文",
      switchToEnglish: "切换到英文",
      switchToLight: "切换到浅色模式",
      switchToDark: "切换到深色模式",
      reportTitle: "性能分析报告",
      nodesCount: (count) => `${count} 个节点`,
      graphAria: (model) => `${model} 完整源码架构与后端性能数据叠加图`,
      decodeLatency: "解码延迟",
      kernelSum: "核函数耗时总和",
      eventMapping: "事件映射率",
      globalMfu: "全局 MFU INT8",
      mappedEvent: "已映射事件",
      unmappedEvent: "未映射事件",
      representative: "代表步骤",
      mappedCount: (mapped, total) => `${mapped} / ${total} 已映射`,
      stepCaption: (step) => `步骤 ${step} 的延迟与事件映射摘要`,
      streamCaption: (events, lanes) => `${events} 个原始事件，按 ${lanes} 条设备 / 流 / 核泳道分组`,
      traceCaption: (events, lanes, range) => `${events} 个持续事件，分布于 ${lanes} 条原始采集泳道 · ${range}`,
      events: (count) => `${count} 个事件`,
      laneTotals: (duration, wait) => `耗时总计 Σ ${duration} · 原始等待总计 Σ ${wait}`,
      footerStatus: (nodes, events, selection) => `${nodes} 个节点 · ${events} 个事件 · ${selection || "未选择"}`,
      noSelectionShort: "未选择",
      appFailed: "应用加载失败",
      architectureLoadFailed: "架构数据加载失败",
    },
  };

  const METRIC_LABELS_ZH = {
    "kernel time": "核函数时间",
    "time share": "时间占比",
    operators: "算子数",
    "HBM estimate": "HBM 估算",
    "MFU INT8": "MFU INT8",
    "MFU BF16": "MFU BF16",
  };

  const METRIC_DEFINITIONS = {
    wall_ms: {
      en: "Wall-clock span from the first assigned kernel start to the last assigned kernel end. It includes execution gaps.",
      zh: "归属到当前节点的首个 kernel 开始至最后一个 kernel 结束之间的墙钟跨度，包含执行间隙。",
      source: { en: "metrics_report.md · four-dimensional base metrics", zh: "metrics_report.md · 四维基础指标" },
    },
    busy_union_ms: {
      en: "Overlap-corrected device busy time after merging all assigned kernel intervals. It measures actual occupied execution time without double-counting overlap.",
      zh: "合并当前节点全部 kernel 时间区间后的设备忙碌时间；重叠区间只计算一次，用于衡量实际占用执行时间。",
      source: { en: "metrics_report.md · four-dimensional base metrics", zh: "metrics_report.md · 四维基础指标" },
    },
    kernel_sum_ms: {
      en: "Arithmetic sum of all assigned kernel durations. Overlap is not removed, so it represents total compute work rather than wall latency.",
      zh: "归属于当前节点的全部 kernel duration 算术和；不扣除并行重叠，因此表示总计算工作量而非墙钟时延。",
      source: { en: "metrics_report.md · four-dimensional base metrics", zh: "metrics_report.md · 四维基础指标" },
    },
    total_cost_ms: {
      en: "Complete event cost calculated as the sum of duration plus wait time for every assigned kernel.",
      zh: "当前节点全部 kernel 的 duration 与 wait 之和，即 Σ(duration + wait)，表示包含等待的完整成本。",
      source: { en: "metrics_report.md · four-dimensional base metrics", zh: "metrics_report.md · 四维基础指标" },
    },
    "time share": {
      en: "This node's kernel time divided by total_time_us in the current performance JSON. Because aggregate scopes can overlap, percentages across the whole tree must not be mechanically summed.",
      zh: "当前节点 kernel time 除以本性能 JSON 的 total_time_us。由于聚合 scope 可能重叠，不能把整棵节点树的百分比机械相加。",
      source: { en: "Time metrics · module-tree display basis", zh: "时间指标 · 模块树展示口径" },
    },
    operators: {
      en: "Count of profiler operators assigned to this node under its metric_scope. Representative or folded scopes may include repeated observed invocations; this is not the number of unique operators in the full model.",
      zh: "按当前 metric_scope 归属于节点的 profiler 算子数量。代表实例或折叠 scope 可能包含重复观测调用，不等于完整模型的唯一算子总数。",
      source: { en: "Folded-layer performance scope", zh: "折叠层性能口径" },
    },
    "HBM estimate": {
      en: "Logical byte estimate derived from tensor shape × dtype, for relative comparison only. It is not actual HBM capacity usage, traffic, bandwidth utilization, or the sampled HBM timeline shown below TraceView.",
      zh: "由 tensor shape × dtype 推导的逻辑字节估算，仅用于相对比较；它不是实际 HBM 容量占用、读写流量、带宽利用率，也不是 TraceView 下方的 HBM 采样时序。",
      source: { en: "HBM metric boundary", zh: "HBM 指标边界" },
    },
    "MFU INT8": {
      en: "INT8 estimate using effective mapped compute ÷ (INT8 peak throughput × time). Only mapped operators are covered; “–” means unavailable. Do not interpret it as trustworthy full-model MFU unless the backend declares complete coverage.",
      zh: "INT8 估算口径：已映射有效计算量 ÷（INT8 峰值算力 × 时间）。它只覆盖已映射算子；“–”表示不可用，除非后端声明完整覆盖，否则不能视为可信的完整模型 MFU。",
      source: { en: "MFU definition and coverage limits", zh: "MFU 口径与覆盖限制" },
    },
    "MFU BF16": {
      en: "BF16 estimate using effective mapped compute ÷ (BF16 peak throughput × time). Only mapped operators are covered; “–” means unavailable. Do not interpret it as trustworthy full-model MFU unless the backend declares complete coverage.",
      zh: "BF16 估算口径：已映射有效计算量 ÷（BF16 峰值算力 × 时间）。它只覆盖已映射算子；“–”表示不可用，除非后端声明完整覆盖，否则不能视为可信的完整模型 MFU。",
      source: { en: "MFU definition and coverage limits", zh: "MFU 口径与覆盖限制" },
    },
  };

  const THEME_ICONS = {
    light: '<circle cx="12" cy="12" r="4"></circle><path d="M12 2v2"></path><path d="M12 20v2"></path><path d="m4.93 4.93 1.41 1.41"></path><path d="m17.66 17.66 1.41 1.41"></path><path d="M2 12h2"></path><path d="M20 12h2"></path><path d="m6.34 17.66-1.41 1.41"></path><path d="m19.07 4.93-1.41 1.41"></path>',
    dark: '<path d="M12 3a6 6 0 0 0 9 7.2A8 8 0 1 1 12 3Z"></path>',
  };
  const STREAM_ZOOM_LEVELS = [1, 1.5, 2, 3, 4, 6, 8];

  function t(key, ...args) {
    const value = I18N[state.language]?.[key] ?? I18N.en[key] ?? key;
    return typeof value === "function" ? value(...args) : value;
  }

  function applyStaticTranslations() {
    document.querySelectorAll("[data-i18n]").forEach((element) => {
      element.textContent = t(element.dataset.i18n);
    });
    document.querySelectorAll("[data-i18n-aria]").forEach((element) => {
      element.setAttribute("aria-label", t(element.dataset.i18nAria));
    });
    document.querySelectorAll("[data-i18n-tooltip]").forEach((element) => {
      element.dataset.reportTooltip = t(element.dataset.i18nTooltip);
    });
  }

  function syncPreferenceControls() {
    const isLight = state.theme === "light";
    const languageLabel = state.language === "en" ? "中" : "EN";
    const languageAction = state.language === "en" ? t("switchToChinese") : t("switchToEnglish");
    const themeAction = isLight ? t("switchToDark") : t("switchToLight");
    document.documentElement.lang = state.language === "zh" ? "zh-CN" : "en";
    document.documentElement.dataset.theme = state.theme;
    if (els.languageToggleLabel) els.languageToggleLabel.textContent = languageLabel;
    els.languageToggle?.setAttribute("aria-label", languageAction);
    if (els.languageToggle) els.languageToggle.dataset.reportTooltip = languageAction;
    els.themeToggle?.classList.toggle("is-selected", isLight);
    els.themeToggle?.setAttribute("aria-pressed", String(isLight));
    els.themeToggle?.setAttribute("aria-label", themeAction);
    if (els.themeToggle) els.themeToggle.dataset.reportTooltip = themeAction;
    if (els.themeToggleIcon) els.themeToggleIcon.innerHTML = isLight ? THEME_ICONS.dark : THEME_ICONS.light;
  }

  applyStaticTranslations();
  syncPreferenceControls();

  const ideFrameInstance = window.PtoIdeFrame?.init(els.ideFrame) || null;
  const runtimeConfig = window.ReportRuntimeConfig || {
    analysis: "../ds3_2_analysis_config.json",
    performance: "../ds3_2_perf_data.json",
    timeline: "../ds3_2_timeline.json",
    trace: "../trace_view.json",
    bindings: "./outputs/trace_bindings.json",
    architecture: "./outputs/model_architecture_graph.json",
    overlay: "./outputs/architecture_overlay_map.json",
    hbm: "./outputs/hbm_series.json",
  };

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  async function loadJson(key, path) {
    const embeddedData = window.ReportEmbeddedData;
    if (embeddedData && Object.prototype.hasOwnProperty.call(embeddedData, key)) {
      return embeddedData[key];
    }
    const response = await fetch(path);
    if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
    return response.json();
  }

  function traceEventsFrom(document) {
    if (Array.isArray(document)) return document;
    if (Array.isArray(document?.traceEvents)) return document.traceEvents;
    if (Array.isArray(document?.events)) return document.events;
    return [];
  }

  function buildRepeatInstanceMetrics(traceDocument) {
    const totals = new Map();
    traceEventsFrom(traceDocument).forEach((event) => {
      if (event?.ph !== "X") return;
      const rawInstanceIndex = event.args?.layer_index ?? event.args?.layerIndex;
      if (rawInstanceIndex == null) return;
      const instanceIndex = Number(rawInstanceIndex);
      const durationUs = Number(event.dur ?? event.duration_us);
      if (!Number.isInteger(instanceIndex) || !Number.isFinite(durationUs) || durationUs < 0) return;
      totals.set(instanceIndex, (totals.get(instanceIndex) || 0) + durationUs);
    });
    const metrics = [...totals].map(([instanceIndex, timeUs]) => ({ instanceIndex, timeUs }));
    const values = metrics.map((metric) => metric.timeUs);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min;
    return metrics.map((metric) => ({
      ...metric,
      heatLevel: span > 0 ? Math.max(1, Math.min(5, Math.round((metric.timeUs - min) / span * 4) + 1)) : 3,
    }));
  }

  let analysisConfig;
  let reportModel;
  let architectureGraphSpec;
  let architectureOverlayMap;
  let architectureGraph;
  let rawTimelineData;
  let rawTraceDocument;
  let traceBindings;
  let hbmData;
  let repeatInstanceMetrics = [];
  try {
    const [analysis, perf, timeline, traceView, bindings, graphSpec, overlayMap, hbm] = await Promise.all([
      loadJson("analysis", runtimeConfig.analysis),
      loadJson("performance", runtimeConfig.performance),
      loadJson("timeline", runtimeConfig.timeline),
      loadJson("trace", runtimeConfig.trace),
      loadJson("bindings", runtimeConfig.bindings),
      loadJson("architecture", runtimeConfig.architecture),
      loadJson("overlay", runtimeConfig.overlay),
      loadJson("hbm", runtimeConfig.hbm),
    ]);
    analysisConfig = analysis;
    architectureGraphSpec = graphSpec;
    architectureOverlayMap = overlayMap;
    rawTimelineData = timeline;
    rawTraceDocument = traceView;
    traceBindings = bindings;
    hbmData = hbm;
    repeatInstanceMetrics = buildRepeatInstanceMetrics(traceView);
    reportModel = window.DeepSeekReportData.createReportModel(analysis, perf, timeline);
    architectureGraph = window.DeepSeekArchitectureData.createArchitectureGraph(graphSpec, reportModel.reports);
    state.collapsedArchitectureIds = new Set(window.DeepSeekArchitectureData.defaultCollapsedIds(graphSpec));
  } catch (error) {
    els.footerStatus.textContent = error.message;
    console.error(error);
    return;
  }

  const REPORTS = reportModel.reports;
  const REPORT_MODEL_TITLE = String(reportModel.identity.modelId).toUpperCase();
  const reportPageTitle = () => state.language === "zh"
    ? `${REPORT_MODEL_TITLE} Profiling 性能报告`
    : `${REPORT_MODEL_TITLE} Profiling Performance Report`;
  const REPORT_ORDER = reportModel.reportOrder;
  const TIMELINE = reportModel.timeline;
  const STREAM_SUMMARY = reportModel.streamSummary;
  const STEP_SUMMARY = reportModel.stepSummary;
  const TIMELINE_NODE_COUNTS = TIMELINE.reduce((counts, event) => {
    if (event.nodeId) counts.set(event.nodeId, (counts.get(event.nodeId) || 0) + 1);
    return counts;
  }, new Map());
  const REPORT_HEAT_VALUES = Object.values(REPORTS)
    .map((report) => metricPercent(report.metricShort))
    .filter((value) => Number.isFinite(value) && value > 0);
  const OPERATOR_HEAT_VALUES = architectureGraph.nodes
    .filter((node) => node.kind === "op")
    .map((node) => metricPercent(node.metricBadge))
    .filter((value) => Number.isFinite(value));
  const operatorHeatMax = OPERATOR_HEAT_VALUES.length ? Math.max(...OPERATOR_HEAT_VALUES) : 0;
  if (els.operatorColorLegendMax) {
    els.operatorColorLegendMax.textContent = `${Number(operatorHeatMax.toFixed(2))}%`;
  }
  const OPERATOR_TREE = createOperatorTree(analysisConfig);
  const OPERATOR_TREE_PARENTS = new Map();
  indexOperatorTree(OPERATOR_TREE, "", OPERATOR_TREE_PARENTS);
  expandOperatorTree(OPERATOR_TREE);
  state.selectedNodeId = REPORT_ORDER[0] || "";
  state.selectedArchitectureId = window.DeepSeekArchitectureData.backendToGraphId(
    architectureGraphSpec,
    state.selectedNodeId,
  );
  expandOperatorAncestors(state.selectedNodeId);

  function formatDuration(us) {
    if (us >= 1000) {
      const value = us / 1000;
      return `${value >= 10 ? value.toFixed(1) : value.toFixed(2)} ms`;
    }
    return `${us < 10 ? us.toFixed(1) : us.toFixed(0)} us`;
  }

  function metricPercent(value) {
    const match = String(value).match(/(\d+(?:\.\d+)?)\s*%/);
    if (!match) return null;
    return Math.max(0, Math.min(100, Number(match[1])));
  }

  function performanceBadgeStyle(value, domainValues = REPORT_HEAT_VALUES) {
    const resolvedValue = metricPercent(value);
    const positiveValues = domainValues.filter((item) => Number.isFinite(item) && item > 0);
    if (!Number.isFinite(resolvedValue) || resolvedValue <= 0 || !positiveValues.length) return "";
    const minValue = Math.min(...positiveValues);
    const maxValue = Math.max(...positiveValues);
    const helper = window.PtoModelGraphvizPattern;
    const fill = helper?.performanceHeatmapColor?.(resolvedValue, maxValue, { minValue });
    const textColor = helper?.performanceHeatmapTextColor?.(fill);
    return fill
      ? `--performance-badge-fill:${fill};--performance-badge-text:${textColor || "#FFFFFF"};`
      : "";
  }

  function isAggregateReport(report) {
    const dimension = String(report?.dimension || "").toLowerCase();
    return dimension.includes("aggregate")
      || report?.facts?.some((fact) => /^Metric scope:\s*(aggregate|phase_aggregate)\b/i.test(String(fact)));
  }

  function aggregateTimeShareHint(report) {
    return t("aggregateTimeShareHint", report.title, report.metricShort);
  }

  function renderList(container, items) {
    container.innerHTML = items.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  }

  function setInspectorSummary(value = "") {
    const summary = String(value || "").trim();
    els.inspectorSummary.textContent = summary;
    els.inspectorSummary.hidden = !summary;
  }

  function metricLabelMarkup(label) {
    return String(label)
      .split("_")
      .map(escapeHtml)
      .join("_<wbr>");
  }

  function metricTileMarkup([label, value], options = {}) {
      const localizedLabel = state.language === "zh" ? (METRIC_LABELS_ZH[label] || label) : label;
      const percent = metricPercent(value);
      const fullWidthClass = !options.core && label === "time share" ? " metric-tile--full" : "";
      const bar = options.core || percent == null
        ? ""
        : `<div class="metric-bar" style="--metric-bar-width:${percent}%"><span></span></div>`;
      return `
        <div class="metric-tile${fullWidthClass}" data-tone="info">
          <button class="metric-label metric-label-trigger" type="button" data-metric-key="${escapeHtml(label)}" aria-label="${escapeHtml(`${localizedLabel}: ${METRIC_DEFINITIONS[label]?.[state.language] || ""}`)}">
            <span>${metricLabelMarkup(localizedLabel)}</span>
            <svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6"></circle><path d="M8 7.1v4M8 4.7h.01"></path></svg>
          </button>
          <div class="metric-value">${escapeHtml(value)}</div>
          ${bar}
        </div>
      `;
  }

  function renderMetrics(report) {
    els.coreMetricGrid.innerHTML = report.coreMetrics.map((metric) => metricTileMarkup(metric, { core: true })).join("");
    els.metricGrid.innerHTML = report.metrics.map((metric) => metricTileMarkup(metric)).join("");
  }

  function positionReportTooltip(trigger) {
    if (!els.fieldTooltip || els.fieldTooltip.hidden || !trigger) return;
    const triggerRect = trigger.getBoundingClientRect();
    const tooltipRect = els.fieldTooltip.getBoundingClientRect();
    const viewportGap = 8;
    const triggerGap = 7;
    const maxLeft = Math.max(viewportGap, window.innerWidth - tooltipRect.width - viewportGap);
    const left = Math.min(maxLeft, Math.max(viewportGap, triggerRect.left));
    const below = triggerRect.bottom + triggerGap;
    const above = triggerRect.top - tooltipRect.height - triggerGap;
    const top = below + tooltipRect.height <= window.innerHeight - viewportGap
      ? below
      : Math.max(viewportGap, above);
    els.fieldTooltip.style.left = `${Math.round(left)}px`;
    els.fieldTooltip.style.top = `${Math.round(top)}px`;
  }

  function showReportTooltip(trigger) {
    const definition = METRIC_DEFINITIONS[trigger?.dataset.metricKey];
    const controlTip = trigger?.dataset.reportTooltip;
    if (!els.fieldTooltip || (!definition && !controlTip)) return;
    if (definition) {
      const metricKey = trigger.dataset.metricKey;
      const label = state.language === "zh" ? (METRIC_LABELS_ZH[metricKey] || metricKey) : metricKey;
      const sourceLabel = state.language === "zh" ? "定义来源" : "Definition source";
      const definitionSource = runtimeConfig.metricDefinitionsSource
        || (state.language === "zh" ? "UI JSON 报告数据契约" : "UI JSON report data contract");
      els.fieldTooltip.innerHTML = `
        <strong class="report-field-tooltip__title">${escapeHtml(label)}</strong>
        <span>${escapeHtml(definition[state.language])}</span>
        <span class="report-field-tooltip__source">${escapeHtml(sourceLabel)} · ${escapeHtml(definitionSource)} · ${escapeHtml(definition.source[state.language])}</span>
      `;
    } else {
      els.fieldTooltip.textContent = controlTip;
    }
    if (state.tooltipTrigger && state.tooltipTrigger !== trigger) {
      state.tooltipTrigger.removeAttribute("aria-describedby");
    }
    state.tooltipTrigger = trigger;
    trigger.setAttribute("aria-describedby", "reportFieldTooltip");
    els.fieldTooltip.hidden = false;
    positionReportTooltip(trigger);
  }

  function hideReportTooltip() {
    if (!els.fieldTooltip) return;
    state.tooltipTrigger?.removeAttribute("aria-describedby");
    state.tooltipTrigger = null;
    els.fieldTooltip.hidden = true;
  }

  function renderOperators(report) {
    if (!report.operators.length) {
      els.operatorList.innerHTML = `<div class="report-inline-empty">${escapeHtml(t("noOperatorRatioForNode"))}</div>`;
      return;
    }
    const heatValues = report.operators.map(([, value]) => metricPercent(value));
    els.operatorList.innerHTML = report.operators.map(([name, value]) => `
      <div class="operator-row">
        <div class="operator-name">${escapeHtml(name)}</div>
        <div class="operator-value performance-value-badge" style="${performanceBadgeStyle(value, heatValues)}">${escapeHtml(value)}</div>
      </div>
    `).join("");
  }

  function createOperatorTree(config) {
    const toNode = (rawNode, overrideChildren) => {
      if (!rawNode?.node_id || !REPORTS[rawNode.node_id]) return null;
      const children = Array.isArray(overrideChildren)
        ? overrideChildren
        : Array.isArray(rawNode.children) ? rawNode.children : [];
      return {
        id: rawNode.node_id,
        nodeId: rawNode.node_id,
        children: children.map((child) => toNode(child)).filter(Boolean),
      };
    };
    const group = (id, labelKey, roots, options = {}) => ({
      id,
      labelKey,
      nodeId: "",
      children: roots.map((rawNode) => toNode(rawNode)).filter(Boolean),
      ...options,
    });
    const stageEntries = Object.entries(config.stages || {});
    const stageByKey = new Map(stageEntries);
    const layerStructure = config.layer_structure || {};
    const layerReferenceKeys = new Set(Object.values(layerStructure)
      .flatMap((value) => Array.isArray(value) ? value : [])
      .filter((value) => typeof value === "string" && stageByKey.has(value)));
    const layerChildren = [...layerReferenceKeys].map((key) => stageByKey.get(key));
    const explicitLayerRoots = Object.values(layerStructure)
      .filter((value) => value && typeof value === "object" && !Array.isArray(value) && value.node_id);
    const decoderEntry = stageEntries.find(([key, node]) => (
      key === "decoder_layers"
      || node?.name === "decoder_layers"
      || /\/decoder_layers$/.test(String(node?.node_id || ""))
    ));
    const layerRoots = explicitLayerRoots.length
      ? explicitLayerRoots
      : decoderEntry
        ? [{ ...decoderEntry[1], children: [...(decoderEntry[1].children || []), ...layerChildren] }]
        : layerChildren;
    const layerBackendNodeId = decoderEntry?.[1]?.node_id || layerRoots[0]?.node_id || "";
    const architectureItems = [...(architectureGraphSpec?.roots || [])];
    let layerRepeatItem = null;
    while (architectureItems.length) {
      const candidate = architectureItems.pop();
      if (candidate?.id === layerBackendNodeId || candidate?.backendNodeId === layerBackendNodeId) {
        layerRepeatItem = candidate;
        break;
      }
      architectureItems.push(...(candidate?.children || []));
    }
    const layerRepeatNodeId = layerRepeatItem?.id || layerBackendNodeId;
    const layerInstanceIndices = Array.isArray(layerRepeatItem?.instanceIndices)
      ? layerRepeatItem.instanceIndices.map(Number).filter(Number.isFinite)
      : Array.from({ length: Number(layerRepeatItem?.repeatCount || 0) }, (_, index) => index);
    const representativeLayerIndex = layerRoots
      .flatMap((root) => [root, ...(root?.children || [])])
      .flatMap((node) => String(node?.node_id || "").split("/"))
      .map(Number)
      .find((index) => layerInstanceIndices.includes(index));
    const layerSelector = layerRepeatItem && layerInstanceIndices.length > 1
      ? {
        repeatNodeId: layerRepeatNodeId,
        instanceIndices: layerInstanceIndices,
        representativeIndex: representativeLayerIndex ?? layerInstanceIndices[0],
      }
      : null;
    const layerNodeIds = new Set(layerRoots.flatMap((root) => [
      root?.node_id,
      ...(root?.children || []).map((child) => child?.node_id),
    ]).filter(Boolean));
    const runtimeRoots = [
      ...stageEntries.map(([, node]) => node).filter((node) => (
        node?.metric_scope === "phase_aggregate"
        || /\/runtime\//.test(String(node?.node_id || ""))
      )),
      ...(config.runtime_auxiliary || []),
    ];
    const runtimeNodeIds = new Set(runtimeRoots.map((node) => node?.node_id).filter(Boolean));
    const stageRoots = stageEntries
      .map(([, node]) => node)
      .filter((node) => !layerNodeIds.has(node?.node_id) && !runtimeNodeIds.has(node?.node_id));
    return [
      group("group/stages", "stagesGroup", stageRoots),
      group("group/layers", "layersGroup", layerRoots, layerSelector ? { layerSelector } : {}),
      group("group/runtime", "runtimeGroup", runtimeRoots),
    ].filter((item) => item.children.length);
  }

  function expandOperatorTree(items) {
    items.forEach((item) => {
      if (item.children.length) state.operatorTreeExpandedIds.add(item.id);
      expandOperatorTree(item.children);
    });
  }

  function indexOperatorTree(items, parentId, parentIndex) {
    items.forEach((item) => {
      if (parentId) parentIndex.set(item.id, parentId);
      indexOperatorTree(item.children, item.id, parentIndex);
    });
  }

  function expandOperatorAncestors(nodeId) {
    let currentId = nodeId;
    while (OPERATOR_TREE_PARENTS.has(currentId)) {
      const parentId = OPERATOR_TREE_PARENTS.get(currentId);
      state.operatorTreeExpandedIds.add(parentId);
      currentId = parentId;
    }
  }

  function operatorTreeDescendantCount(item) {
    return item.children.reduce((count, child) => count + 1 + operatorTreeDescendantCount(child), 0);
  }

  function renderLayerSelector(selector) {
    const savedIndex = Number(state.repeatInstanceSelections.get(selector.repeatNodeId));
    const selectedIndex = selector.instanceIndices.includes(savedIndex)
      ? savedIndex
      : selector.representativeIndex;
    const actionLabel = t("selectLayer", selectedIndex);
    return `
      <label class="operator-layer-selector" data-report-tooltip="${escapeHtml(actionLabel)}">
        <span class="operator-layer-selector-label">${escapeHtml(t("layerSelector"))}</span>
        <select class="operator-layer-select" data-layer-selector="${escapeHtml(selector.repeatNodeId)}" aria-label="${escapeHtml(actionLabel)}">
          ${selector.instanceIndices.map((index) => `
            <option value="${index}"${index === selectedIndex ? " selected" : ""}>${escapeHtml(`${t("layerSelector")} ${index}`)}</option>
          `).join("")}
        </select>
      </label>
    `;
  }

  function renderOperatorTreeItem(item, depth) {
    const isGroup = !item.nodeId;
    const report = item.nodeId ? REPORTS[item.nodeId] : null;
    const label = isGroup ? t(item.labelKey) : report.title;
    const hasChildren = item.children.length > 0;
    const expanded = hasChildren && state.operatorTreeExpandedIds.has(item.id);
    const toggleLabel = `${t(expanded ? "collapse" : "expand")} ${label}`;
    const toggle = hasChildren ? `
      <button type="button" class="operator-tree-toggle" data-tree-toggle="${escapeHtml(item.id)}" aria-label="${escapeHtml(toggleLabel)}" data-report-tooltip="${escapeHtml(toggleLabel)}">
        <svg viewBox="0 0 16 16" aria-hidden="true"><path d="m6 4 4 4-4 4"></path></svg>
      </button>
    ` : '<span class="operator-tree-toggle-placeholder" aria-hidden="true"></span>';
    const groupButton = `
      <button type="button" class="operator-tree-group-button" data-tree-toggle="${escapeHtml(item.id)}">
        <span class="operator-tree-toggle" aria-hidden="true">
          <svg viewBox="0 0 16 16"><path d="m6 4 4 4-4 4"></path></svg>
        </span>
        <span class="operator-tree-group-label">${escapeHtml(label)}</span>
        <span class="operator-tree-count">${operatorTreeDescendantCount(item)}</span>
      </button>
    `;
    const aggregate = !isGroup && isAggregateReport(report);
    const aggregateHint = aggregate ? aggregateTimeShareHint(report) : "";
    const row = isGroup ? (item.layerSelector ? `
      <div class="operator-tree-group-shell">
        ${groupButton}
        ${renderLayerSelector(item.layerSelector)}
      </div>
    ` : groupButton) : `
      ${toggle}
      <button type="button" class="mapped-node-button operator-tree-node-button" data-node-id="${escapeHtml(item.nodeId)}" aria-current="${item.nodeId === state.selectedNodeId ? "true" : "false"}">
        <span class="node-name">${escapeHtml(label)}</span>
        <span class="node-metric performance-value-badge${aggregate ? " is-aggregate" : ""}" style="${aggregate ? "" : performanceBadgeStyle(report.metricShort)}"${aggregate ? ` data-report-tooltip="${escapeHtml(aggregateHint)}" aria-label="${escapeHtml(aggregateHint)}"` : ""}>${escapeHtml(report.metricShort)}</span>
      </button>
    `;
    const children = expanded ? item.children.map((child) => renderOperatorTreeItem(child, depth + 1)).join("") : "";
    return `
      <div class="operator-tree-item${isGroup ? " is-group" : ""}${expanded ? " is-expanded" : ""}" role="treeitem" aria-level="${depth + 1}"${hasChildren ? ` aria-expanded="${expanded}"` : ""} style="--tree-depth:${depth}">
        <div class="operator-tree-row">${row}</div>
        ${children}
      </div>
    `;
  }

  function renderNodeList() {
    els.nodeList.innerHTML = OPERATOR_TREE.map((item) => renderOperatorTreeItem(item, 0)).join("");
    if (state.activeArchitectureView === "operators" && state.selectedNodeId) {
      window.requestAnimationFrame(() => {
        els.nodeList.querySelector('[aria-current="true"]')?.scrollIntoView({ block: "nearest" });
      });
    }
  }

  function formatArchitectureEdge(edge, direction) {
    const endpoint = direction === "incoming" ? edge.source : edge.target;
    const tensor = edge.tensor || {};
    const tensorParts = [tensor.name, tensor.shape, tensor.dtype].filter(Boolean).join(" · ");
    const constraints = Array.isArray(tensor.constraints) && tensor.constraints.length
      ? ` · constraints: ${tensor.constraints.join(", ")}`
      : "";
    return `${direction} edge · ${edge.id} · ${endpoint}${tensorParts ? ` · ${tensorParts}` : ""}${constraints}`;
  }

  function architectureMetadataFacts(graphNodeId) {
    const metadata = window.DeepSeekArchitectureData.metadataForGraphItem(
      architectureGraphSpec,
      graphNodeId,
    );
    if (!metadata) return [];
    const { item, parentId, incomingEdges, outgoingEdges } = metadata;
    const facts = [
      `JSON item · kind: ${item.kind || "unknown"} · origin: ${item.origin || "source"} · dataState: ${item.dataState || "source_only"}`,
    ];
    if (parentId) facts.push(`parentId · ${parentId}`);
    if (item.typeLabel) facts.push(`typeLabel · ${item.typeLabel}`);
    if (item.stateType) facts.push(`stateType · ${item.stateType}`);
    if (item.mappingKind) facts.push(`mappingKind · ${item.mappingKind}`);
    if (item.repeatCount || item.repeatRange || item.instanceIndices || item.observedInstanceIndices) {
      facts.push(`repeat · ${JSON.stringify({
        repeatCount: item.repeatCount,
        repeatRange: item.repeatRange,
        instanceIndices: item.instanceIndices,
        observedInstanceIndices: item.observedInstanceIndices,
      })}`);
    }
    if (item.attrs && Object.keys(item.attrs).length) facts.push(`attrs · ${JSON.stringify(item.attrs)}`);
    (item.sourceRefs || []).forEach((sourceRef) => {
      const lines = sourceRef.line_start || sourceRef.lineStart
        ? `:${sourceRef.line_start || sourceRef.lineStart}-${sourceRef.line_end || sourceRef.lineEnd || sourceRef.line_start || sourceRef.lineStart}`
        : "";
      facts.push(`sourceRef · ${sourceRef.path || sourceRef.source || "unknown"}${lines}${sourceRef.symbol ? ` · ${sourceRef.symbol}` : ""}`);
    });
    incomingEdges.forEach((edge) => facts.push(formatArchitectureEdge(edge, "incoming")));
    outgoingEdges.forEach((edge) => facts.push(formatArchitectureEdge(edge, "outgoing")));
    return facts;
  }

  function renderInspector() {
    hideReportTooltip();
    const report = REPORTS[state.selectedNodeId];
    const architectureMetadata = window.DeepSeekArchitectureData.metadataForGraphItem(
      architectureGraphSpec,
      state.selectedArchitectureId,
    );
    if (!report && architectureMetadata) {
      const item = architectureMetadata.item;
      els.inspectorTitle.textContent = item.label || item.id;
      els.inspectorNodeId.textContent = item.id;
      setInspectorSummary(state.language === "zh"
        ? "此节点直接来自 model_architecture_graph.v1。这里只显示原始架构元数据；它没有后端性能指标，也不会触发 TraceView 联动。"
        : "This node comes directly from model_architecture_graph.v1. Only original architecture metadata is shown; it has no backend metrics and does not activate TraceView bindings.");
      els.coreMetricGrid.innerHTML = `<div class="report-inline-empty">${escapeHtml(t("noMetrics"))}</div>`;
      els.metricGrid.innerHTML = `<div class="report-inline-empty">${escapeHtml(t("noMetrics"))}</div>`;
      renderList(els.factList, architectureMetadataFacts(state.selectedArchitectureId));
      els.operatorList.innerHTML = `<div class="report-inline-empty">${escapeHtml(t("noOperatorRatio"))}</div>`;
    } else if (!report) {
      els.inspectorTitle.textContent = t("selectBackendNode");
      els.inspectorNodeId.textContent = t("noNodeId");
      setInspectorSummary(t("selectHint"));
      els.coreMetricGrid.innerHTML = `<div class="report-inline-empty">${escapeHtml(t("noMetrics"))}</div>`;
      els.metricGrid.innerHTML = `<div class="report-inline-empty">${escapeHtml(t("noMetrics"))}</div>`;
      els.factList.innerHTML = `<li class="report-inline-empty">${escapeHtml(t("noEvidence"))}</li>`;
      els.operatorList.innerHTML = `<div class="report-inline-empty">${escapeHtml(t("noOperatorRatio"))}</div>`;
    } else {
      els.inspectorTitle.textContent = report.title;
      els.inspectorNodeId.textContent = report.nodeId;
      setInspectorSummary();
      renderMetrics(report);
      renderList(els.factList, [
        ...report.facts.filter((fact) => String(fact).trim() !== String(report.summary || "").trim()),
        ...architectureMetadataFacts(state.selectedArchitectureId),
      ]);
      renderOperators(report);
    }
    renderNodeList();
    renderFooterStatus();
  }

  function selectNode(nodeId, options = {}) {
    if (!REPORTS[nodeId]) return;
    if (options.toggle && state.selectedNodeId === nodeId) {
      clearSelection();
      return;
    }
    state.selectedNodeId = nodeId;
    state.hbmFollowsTrace = true;
    // A module selection zooms all three HBM series around its representative
    // kernels. Only an explicit single-kernel selection narrows to occupancy.
    if (options.source !== "trace-event") state.hbmEventFocus = false;
    state.selectedArchitectureId = window.DeepSeekArchitectureData.backendToGraphId(
      architectureGraphSpec,
      nodeId,
    );
    expandOperatorAncestors(nodeId);
    if (options.source !== "timeline") state.activeTimelineSegment = -1;
    renderInspector();
    drawStreamCanvases();
    if (options.syncTrace !== false) {
      const focusEvent = state.traceController?.setSelectedNode(nodeId);
      if (focusEvent) {
        // Zoom all three series to the module's mapped kernel span. The HBM view keeps
        // its own sampling-aware minimum so real samples stay visible.
        state.hbmController?.setRange(focusEvent.ts, focusEvent.end);
      } else {
        state.hbmFollowsTrace = false;
        state.hbmController?.resetRange();
      }
    }
    if (options.syncGraph !== false) {
      const graphNodeId = window.DeepSeekArchitectureData.backendToGraphId(architectureGraphSpec, nodeId);
      if (!graphNodeId) return;
      const centerGraphNode = options.centerGraphNode ?? options.source === "timeline";
      let expandedAncestor = false;
      window.DeepSeekArchitectureData.ancestorIdsForGraphId(architectureGraphSpec, graphNodeId)
        .forEach((collapsedId) => {
        if (!state.collapsedArchitectureIds.has(collapsedId)) return;
        state.collapsedArchitectureIds.delete(collapsedId);
        expandedAncestor = true;
      });
      if (state.activeArchitectureView !== "architecture") {
        state.pendingArchitectureCenterNodeId = centerGraphNode ? graphNodeId : "";
        return;
      }
      if (expandedAncestor) {
        renderArchitecture({
          initialTransform: state.architectureController?.getTransform(),
          activeNodeId: graphNodeId,
          centerNodeId: centerGraphNode ? graphNodeId : "",
        });
      } else {
        state.architectureController?.selectNode(graphNodeId, { source: options.source || "app" });
        if (centerGraphNode) state.architectureController?.centerNode(graphNodeId);
      }
    }
  }

  function selectArchitectureItem(graphNodeId) {
    const metadata = window.DeepSeekArchitectureData.metadataForGraphItem(
      architectureGraphSpec,
      graphNodeId,
    );
    if (!metadata) return;
    const backendNodeId = metadata.item.backendNodeId || "";
    if (REPORTS[backendNodeId]) {
      selectNode(backendNodeId, { syncGraph: false, source: "graph" });
      return;
    }
    state.selectedNodeId = "";
    state.selectedArchitectureId = graphNodeId;
    state.activeTimelineSegment = -1;
    state.pendingArchitectureCenterNodeId = "";
    state.hbmEventFocus = false;
    state.hbmFollowsTrace = false;
    state.traceController?.setSelectedNode("");
    state.hbmController?.resetRange();
    drawStreamCanvases();
    renderInspector();
  }

  function clearSelection() {
    state.selectedNodeId = "";
    state.selectedArchitectureId = "";
    state.activeTimelineSegment = -1;
    state.pendingArchitectureCenterNodeId = "";
    state.hbmEventFocus = false;
    state.hbmFollowsTrace = false;
    state.architectureController?.clearSelection();
    drawStreamCanvases();
    state.traceController?.setSelectedNode("");
    state.hbmController?.resetRange();
    renderInspector();
  }

  function evidenceMap() {
    return Object.fromEntries(Object.entries(REPORTS).flatMap(([nodeId, report]) => {
      const graphNodeId = window.DeepSeekArchitectureData.backendToGraphId(architectureGraphSpec, nodeId);
      return graphNodeId ? [[graphNodeId, {
        dimension: report.dimension,
        metric: report.metricShort,
        what: report.summary,
        evidence: report.facts,
      }]] : [];
    }));
  }

  function architectureItemAnchor(graph, nodeId) {
    const node = graph?.nodes?.find((item) => item.id === nodeId);
    if (node) return { x: node.x, y: node.y };
    const cluster = graph?.clusters?.find((item) => item.id === nodeId);
    return cluster ? { x: cluster.x + cluster.width / 2, y: cluster.y + 18 } : null;
  }

  function architectureSourceItem(nodeId) {
    const stack = [...(architectureGraphSpec?.roots || [])];
    while (stack.length) {
      const item = stack.pop();
      if (item.id === nodeId) return item;
      stack.push(...(item.children || []));
    }
    return null;
  }

  function representativeInstanceIndex(cluster) {
    const indices = (cluster.instanceIndices || []).map(Number).filter(Number.isFinite);
    const saved = Number(state.repeatInstanceSelections.get(cluster.id));
    if (indices.includes(saved)) return saved;
    const sourceItem = architectureSourceItem(cluster.id);
    const stack = sourceItem ? [sourceItem] : [];
    while (stack.length) {
      const item = stack.pop();
      const numericSegments = String(item.backendNodeId || "")
        .split("/")
        .filter((segment) => /^\d+$/.test(segment))
        .map(Number);
      const candidate = numericSegments.find((value) => indices.includes(value));
      if (candidate != null) return candidate;
      stack.push(...(item.children || []));
    }
    return indices[0];
  }

  function decorateRepeatClusters(graph) {
    const metricsByIndex = new Map(repeatInstanceMetrics.map((metric) => [metric.instanceIndex, metric]));
    graph.clusters.forEach((cluster) => {
      if (!cluster.repeat) return;
      const indices = (cluster.instanceIndices || []).map(Number).filter(Number.isFinite);
      cluster.instanceMetrics = indices.map((index) => metricsByIndex.get(index)).filter(Boolean);
      cluster.selectedInstanceIndex = representativeInstanceIndex(cluster);
      state.repeatInstanceSelections.set(cluster.id, cluster.selectedInstanceIndex);
    });
    return graph;
  }

  function activateArchitectureView(viewName) {
    const previousView = state.activeArchitectureView;
    state.activeArchitectureView = viewName === "operators" ? "operators" : "architecture";
    document.querySelectorAll("[data-architecture-view]").forEach((button) => {
      const selected = button.dataset.architectureView === state.activeArchitectureView;
      button.classList.toggle("is-selected", selected);
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
    els.architectureGraphPanel.hidden = state.activeArchitectureView !== "architecture";
    els.operatorTreePanel.hidden = state.activeArchitectureView !== "operators";
    els.nodeViewsRailButton.setAttribute("aria-label", t("operatorListView"));
    els.nodeViewsRailButton.dataset.reportTooltip = t("operatorListView");
    if (state.activeArchitectureView === "operators") renderNodeList();
    if (state.activeArchitectureView === "architecture" && previousView !== "architecture") {
      const initialTransform = state.architectureController?.getTransform();
      const activeNodeId = state.selectedArchitectureId;
      const centerNodeId = state.pendingArchitectureCenterNodeId;
      state.pendingArchitectureCenterNodeId = "";
      window.requestAnimationFrame(() => renderArchitecture({ initialTransform, activeNodeId, centerNodeId }));
    }
  }

  function renderArchitecture(options = {}) {
    const helper = window.PtoModelGraphvizPattern;
    if (!helper) throw new Error("model-graphviz pattern is unavailable");
    const architectureView = decorateRepeatClusters(
      window.DeepSeekArchitectureData.createArchitectureView(
        architectureGraphSpec,
        REPORTS,
        state.collapsedArchitectureIds,
      ),
    );
    let initialTransform = options.initialTransform ? { ...options.initialTransform } : null;
    if (initialTransform && options.anchor) {
      const nextAnchor = architectureItemAnchor(architectureView, options.anchor.nodeId);
      if (nextAnchor) {
        initialTransform.tx += (options.anchor.x - nextAnchor.x) * initialTransform.zoom;
        initialTransform.ty += (options.anchor.y - nextAnchor.y) * initialTransform.zoom;
      }
    }
    state.architectureViewGraph = architectureView;
    state.visibleArchitectureIds = new Set([
      ...architectureView.nodes.map((node) => node.id),
      ...architectureView.clusters.map((cluster) => cluster.id),
    ]);
    state.architectureController?.destroy();
    state.architectureController = helper.renderController(els.architectureGraph, architectureView, {
      ariaLabel: t("graphAria", reportModel.identity.modelId),
      className: "pto-model-architecture-stage",
      autoFit: !initialTransform,
      fitMode: "readable",
      viewportPadding: 28,
      minReadableZoom: 0.68,
      initialTransform,
      activeNodeId: options.activeNodeId,
      selectableClusters: true,
      metricOverlays: true,
      reportOverlays: false,
      edgeTags: false,
      performanceHeatmap: { enabled: true },
      evidenceMap: evidenceMap(),
      colormap: helper.modelArchitectureColormap(architectureView),
      onRepeatInstanceChange({ nodeId, instanceIndex }) {
        const transform = state.architectureController?.getTransform();
        const anchor = architectureItemAnchor(state.architectureViewGraph, nodeId);
        state.repeatInstanceSelections.set(nodeId, Number(instanceIndex));
        renderArchitecture({
          initialTransform: transform,
          activeNodeId: state.selectedArchitectureId,
          anchor: anchor ? { nodeId, ...anchor } : null,
        });
      },
      onToggle({ nodeId, collapsed }) {
        const transform = state.architectureController?.getTransform();
        const anchor = architectureItemAnchor(state.architectureViewGraph, nodeId);
        if (collapsed) {
          state.collapsedArchitectureIds.delete(nodeId);
        } else {
          state.collapsedArchitectureIds.add(nodeId);
        }
        const selectedGraphId = state.selectedArchitectureId;
        const selectedAncestors = window.DeepSeekArchitectureData.ancestorIdsForGraphId(
          architectureGraphSpec,
          selectedGraphId,
        );
        const selectedIsHidden = !collapsed && selectedAncestors.includes(nodeId);
        renderArchitecture({
          initialTransform: transform,
          activeNodeId: selectedIsHidden ? nodeId : selectedGraphId,
          anchor: anchor ? { nodeId, ...anchor } : null,
        });
      },
      onSelect({ nodeId, source }) {
        const backendNodeId = window.DeepSeekArchitectureData.graphToBackendNodeId(
          architectureGraphSpec,
          nodeId,
        );
        if (!REPORTS[backendNodeId]) {
          selectArchitectureItem(nodeId);
          return;
        }
        if (state.selectedNodeId === backendNodeId) {
          if (["graph", "keyboard", "cluster"].includes(source)) {
            selectNode(backendNodeId, { syncGraph: false, source: "graph" });
          }
          return;
        }
        selectNode(backendNodeId, { syncGraph: false, source: "graph" });
      },
      onClearSelection() {
        clearSelection();
      },
    });
    if (options.centerNodeId) {
      window.requestAnimationFrame(() => {
        window.requestAnimationFrame(() => state.architectureController?.centerNode(options.centerNodeId));
      });
    }
  }

  function streamBounds() {
    const minStart = Math.min(...TIMELINE.map((event) => event.startUs));
    const maxEnd = Math.max(...TIMELINE.map((event) => event.endUs));
    return { minStart, maxEnd, span: maxEnd - minStart };
  }

  function renderStepTimeline() {
    const summary = [
      [t("decodeLatency"), formatDuration(STEP_SUMMARY.decodeLatencyUs)],
      [t("kernelSum"), formatDuration(STEP_SUMMARY.kernelSumUs)],
      [t("eventMapping"), `${STEP_SUMMARY.mappingCoveragePct.toFixed(1)}%`],
      [t("globalMfu"), `${STEP_SUMMARY.globalMfuInt8Pct.toFixed(2)}%`],
    ].map(([label, value]) => `
      <div class="timeline-stat">
        <div class="timeline-stat-label">${escapeHtml(label)}</div>
        <div class="timeline-stat-value">${escapeHtml(value)}</div>
      </div>
    `).join("");
    const mappedPct = STEP_SUMMARY.eventCount ? STEP_SUMMARY.mappedEvents / STEP_SUMMARY.eventCount * 100 : 0;
    const unmappedPct = Math.max(0, 100 - mappedPct);
    els.stepTimelinePanel.innerHTML = `
      <div class="timeline-summary-grid">${summary}</div>
      <div class="timeline-legend" aria-label="event mapping legend">
        <span class="legend-item"><span class="legend-swatch" style="--legend-color:var(--success)"></span>${escapeHtml(t("mappedEvent"))}</span>
        <span class="legend-item"><span class="legend-swatch" style="--legend-color:var(--danger)"></span>${escapeHtml(t("unmappedEvent"))}</span>
      </div>
      <div class="step-timeline">
        <div class="step-row" data-kind="representative">
          <div class="step-label"><span class="step-name">${state.language === "zh" ? "步骤" : "Step"} ${escapeHtml(STEP_SUMMARY.step)}</span><span class="step-meta">${escapeHtml(t("representative"))}</span></div>
          <div class="step-stack" data-report-tooltip="${STEP_SUMMARY.mappedEvents} mapped, ${STEP_SUMMARY.unmappedEvents} unmapped events">
            <span class="step-stack-compute" style="width:${mappedPct}%"></span>
            <span class="step-stack-free" style="width:${unmappedPct}%"></span>
          </div>
          <div class="step-values">${escapeHtml(t("mappedCount", STEP_SUMMARY.mappedEvents, STEP_SUMMARY.eventCount))}</div>
        </div>
      </div>
    `;
  }

  function scheduleStreamDraw() {
    window.clearTimeout(state.streamResizeTimer);
    state.streamResizeTimer = window.setTimeout(() => {
      updateStreamChartWidth();
      window.requestAnimationFrame(() => window.requestAnimationFrame(drawStreamCanvases));
    }, 0);
  }

  function streamZoom() {
    return STREAM_ZOOM_LEVELS[state.streamZoomIndex] || 1;
  }

  function syncStreamZoomControls() {
    const zoom = streamZoom();
    const visible = state.activeTimelineTab === "streams";
    els.streamZoomControls.hidden = !visible;
    els.streamZoomControls.setAttribute("aria-label", t("timelineZoom"));
    els.streamZoomOut.disabled = state.streamZoomIndex === 0;
    els.streamZoomIn.disabled = state.streamZoomIndex === STREAM_ZOOM_LEVELS.length - 1;
    els.streamZoomReset.textContent = `${Math.round(zoom * 100)}%`;
    els.streamZoomOut.setAttribute("aria-label", t("zoomOut"));
    els.streamZoomOut.dataset.reportTooltip = t("zoomOut");
    els.streamZoomIn.setAttribute("aria-label", t("zoomIn"));
    els.streamZoomIn.dataset.reportTooltip = t("zoomIn");
    els.streamZoomReset.setAttribute("aria-label", t("resetZoom"));
    els.streamZoomReset.dataset.reportTooltip = t("resetZoom");
  }

  function updateStreamChartWidth() {
    const scroller = els.streamTimelinePanel.querySelector(".stream-lane-scroller");
    const chart = scroller?.querySelector(".stream-lane-chart");
    if (!scroller || !chart) return;
    const fixedColumnsAndGaps = 390;
    const baseTrackWidth = Math.max(240, scroller.clientWidth - fixedColumnsAndGaps);
    chart.style.width = `${Math.ceil(fixedColumnsAndGaps + baseTrackWidth * streamZoom())}px`;
  }

  function setStreamZoom(nextIndex) {
    const scroller = els.streamTimelinePanel.querySelector(".stream-lane-scroller");
    const viewportCenterRatio = scroller?.scrollWidth
      ? (scroller.scrollLeft + scroller.clientWidth / 2) / scroller.scrollWidth
      : 0.5;
    state.streamZoomIndex = Math.max(0, Math.min(STREAM_ZOOM_LEVELS.length - 1, nextIndex));
    syncStreamZoomControls();
    updateStreamChartWidth();
    window.requestAnimationFrame(() => {
      if (scroller) {
        scroller.scrollLeft = viewportCenterRatio * scroller.scrollWidth - scroller.clientWidth / 2;
      }
      scheduleStreamDraw();
    });
  }

  function renderStreamTimeline() {
    const { minStart, span } = streamBounds();
    const rulerTicks = [0, 0.25, 0.5, 0.75, 1].map((ratio, index, ticks) => `
      <span class="stream-time-tick${index === 0 ? " is-start" : index === ticks.length - 1 ? " is-end" : ""}" style="--tick-position:${ratio * 100}%">
        <span>${escapeHtml(formatDuration(minStart + span * ratio))}</span>
      </span>
    `).join("");
    const laneRows = STREAM_SUMMARY.map((lane) => `
      <div class="stream-lane-row pto-pattern-swimlane-task__row" data-lane="${escapeHtml(lane.lane)}">
        <div class="stream-label pto-pattern-swimlane-task__label" data-report-tooltip="${escapeHtml(t("events", lane.ops))}">${escapeHtml(lane.lane)} · ${escapeHtml(t("events", lane.ops))}</div>
        <div class="stream-lane-cell"><canvas class="stream-lane-canvas pto-pattern-swimlane-task__canvas" data-lane="${escapeHtml(lane.lane)}" tabindex="0" aria-label="${escapeHtml(lane.lane)} event timeline"></canvas></div>
        <div class="stream-lane-values">${escapeHtml(t("laneTotals", formatDuration(lane.opUs), formatDuration(lane.waitUs)))}</div>
      </div>
    `).join("");
    els.streamTimelinePanel.innerHTML = `
      <div class="stream-lane-scroller">
        <div class="stream-lane-chart pto-pattern-swimlane-task">
          <div class="stream-lane-header stream-lane-row" aria-hidden="true">
            <div class="stream-label stream-lane-header-cell">${escapeHtml(t("executionLane"))}</div>
            <div class="stream-time-ruler">${rulerTicks}</div>
            <div class="stream-lane-values stream-lane-header-cell">${escapeHtml(t("laneTotalsHeader"))}</div>
          </div>
          ${laneRows}
        </div>
      </div>
    `;
    syncStreamZoomControls();
    updateStreamChartWidth();
    bindStreamCanvasInteractions();
    if (!state.streamResizeObserver && "ResizeObserver" in window) {
      state.streamResizeObserver = new ResizeObserver(scheduleStreamDraw);
      state.streamResizeObserver.observe(els.streamTimelinePanel);
    }
    scheduleStreamDraw();
  }

  function representativeTraceWindow() {
    const events = Array.isArray(rawTimelineData?.events) ? rawTimelineData.events : [];
    const starts = events.map((event) => Number(event.start_time_us_raw)).filter(Number.isFinite);
    const ends = events.map((event) => {
      const start = Number(event.start_time_us_raw);
      const duration = Number(event.duration_us);
      return Number.isFinite(start) && Number.isFinite(duration) ? start + duration : NaN;
    }).filter(Number.isFinite);
    return {
      start: starts.length ? Math.min(...starts) : NaN,
      end: ends.length ? Math.max(...ends) : NaN,
    };
  }

  function renderTraceTimeline() {
    const focus = representativeTraceWindow();
    els.traceTimelinePanel.replaceChildren();
    state.traceController = window.DeepSeekTraceView.createTraceView(
      els.traceTimelinePanel,
      rawTraceDocument,
      {
        language: state.language,
        focusStart: focus.start,
        focusEnd: focus.end,
        bindings: traceBindings.bindings,
        selectedNodeId: state.selectedNodeId,
        toolbarContainer: els.traceToolbarMount,
        stepLabel: `Step ${STEP_SUMMARY.step}`,
        stepLabelZh: `步骤 ${STEP_SUMMARY.step}`,
        onEventSelect(event) {
          if (!event.ownerNodeId || !REPORTS[event.ownerNodeId]) return;
          selectNode(event.ownerNodeId, {
            source: "trace-event",
            syncTrace: false,
            centerGraphNode: true,
          });
          // Apply the single-kernel occupancy focus after selection so any viewport
          // callbacks triggered by selectNode cannot cancel it.
          state.hbmEventFocus = true;
          const focused = state.hbmController?.focusOccupancy(event.ts, event.end);
          if (!focused) {
            state.hbmEventFocus = false;
            state.hbmFollowsTrace = false;
            state.hbmController?.resetRange();
          }
        },
        onVisibleRangeChange(range) {
          state.traceVisibleRange = { start: range.start, end: range.end };
          if (state.hbmFollowsTrace && !state.hbmEventFocus) {
            state.hbmController?.setRange(range.start, range.end);
          }
        },
        onViewportInteraction({ source } = {}) {
          if (source === "fit" || source === "reset") {
            state.hbmEventFocus = false;
            state.hbmFollowsTrace = false;
            state.traceVisibleRange = null;
            state.hbmController?.resetRange();
            return;
          }
          // Keep an explicit single-kernel occupancy focus. Only Fit, reset, or clearing
          // the selection leaves that mode; incidental viewport events must not.
          if (state.hbmEventFocus) return;
          state.hbmFollowsTrace = true;
        },
        onCaption({ eventCount, trackCount, mode }) {
          if (state.activeTimelineTab !== "trace") return;
          const range = mode === "step"
            ? (state.language === "zh" ? `步骤 ${STEP_SUMMARY.step}` : `Step ${STEP_SUMMARY.step}`)
            : (state.language === "zh" ? "完整采集" : "full capture");
          els.timelineCaption.textContent = t("traceCaption", eventCount, trackCount, range);
        },
      },
    );
  }

  function renderHbmTimeline() {
    state.hbmController?.destroy?.();
    state.hbmController = null;
    const hasHbmTimelineData = Array.isArray(hbmData?.bandwidth?.points)
      && hbmData.bandwidth.points.length > 0
      && Array.isArray(hbmData?.occupancy?.points)
      && hbmData.occupancy.points.length > 0;
    if (!hasHbmTimelineData) {
      els.hbmTimelinePanel.hidden = true;
      els.hbmTimelinePanel.replaceChildren();
      return;
    }
    els.hbmTimelinePanel.hidden = false;
    state.hbmController = window.HbmTimelineView.createHbmView(els.hbmTimelinePanel, hbmData, {
      language: state.language,
      range: state.hbmFollowsTrace ? state.traceVisibleRange : null,
    });
  }

  function segmentsForLane(lane) {
    return TIMELINE
      .map((event, index) => ({ ...event, index }))
      .filter((event) => event.lane === lane)
      .sort((left, right) => left.startUs - right.startUs);
  }

  function streamTask(segment, lane) {
    return {
      label: segment.name,
      displayName: segment.name,
      rawName: segment.name,
      opName: segment.name,
      laneKind: segment.core,
      laneId: lane,
      totalCycle: segment.wallUs,
      gap: segment.waitUs,
      status: segment.category,
      dominantCounter: segment.nodeId || "owner_node_id=null",
    };
  }

  function drawStreamCanvases() {
    const helper = window.PtoSwimlaneTaskPattern;
    if (!helper || !els.streamTimelinePanel) return;
    const { minStart, span } = streamBounds();
    const fontFamily = window.getComputedStyle(document.body).fontFamily;
    const hasLinkedTimelineSelection = Boolean(
      state.selectedNodeId && TIMELINE_NODE_COUNTS.get(state.selectedNodeId),
    );
    els.streamTimelinePanel.querySelectorAll(".stream-lane-canvas").forEach((canvas) => {
      const width = Math.max(1, Math.floor(canvas.clientWidth || canvas.parentElement?.clientWidth || 1));
      const height = 12;
      const pixelRatio = Math.max(1, window.devicePixelRatio || 1);
      canvas.width = Math.round(width * pixelRatio);
      canvas.height = Math.round(height * pixelRatio);
      const context = canvas.getContext("2d");
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      context.clearRect(0, 0, width, height);
      const lane = canvas.dataset.lane;
      let linkedEventCount = 0;
      canvas.__reportSegments = segmentsForLane(lane).map((segment) => {
        const x = ((segment.startUs - minStart) / span) * width;
        const segmentWidth = Math.max(1, (segment.wallUs / span) * width);
        const geometry = { x, y: 1, width: Math.min(segmentWidth, width - x), height: 10, segment };
        const isActive = segment.index === state.activeTimelineSegment;
        const isLinked = Boolean(segment.nodeId && segment.nodeId === state.selectedNodeId);
        if (isLinked) linkedEventCount += 1;
        context.save();
        context.globalAlpha = hasLinkedTimelineSelection && !isLinked ? 0.26 : 1;
        helper.drawTaskBar(context, {
          ...geometry,
          baseColor: segment.color,
          fontFamily,
          task: streamTask(segment, lane),
          isSelected: isActive || (isLinked && state.activeTimelineSegment < 0),
          isEmphasized: isLinked,
          isRelated: isLinked && !isActive,
        });
        context.restore();
        return geometry;
      });
      canvas.dataset.linkedEventCount = String(linkedEventCount);
      canvas.closest(".stream-lane-row")?.classList.toggle("has-linked-selection", linkedEventCount > 0);
    });
  }

  function canvasHit(canvas, event) {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    return [...(canvas.__reportSegments || [])].reverse().find((hit) => (
      x >= hit.x && x <= hit.x + hit.width && y >= hit.y && y <= hit.y + hit.height
    )) || null;
  }

  function activateSegment(segment) {
    if (!segment) return;
    state.activeTimelineSegment = segment.index;
    if (segment.nodeId && REPORTS[segment.nodeId]) {
      selectNode(segment.nodeId, { source: "timeline" });
      return;
    }
    state.selectedNodeId = "";
    state.architectureController?.clearSelection();
    renderInspector();
    drawStreamCanvases();
    state.traceController?.setSelectedNode("");
  }

  function bindStreamCanvasInteractions() {
    const helper = window.PtoSwimlaneTaskPattern;
    if (!helper) return;
    state.streamTooltip?.remove();
    state.streamTooltip = helper.createTooltip();
    els.streamTimelinePanel.appendChild(state.streamTooltip);
    els.streamTimelinePanel.querySelectorAll(".stream-lane-canvas").forEach((canvas) => {
      canvas.addEventListener("click", (event) => activateSegment(canvasHit(canvas, event)?.segment));
      canvas.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        const segments = segmentsForLane(canvas.dataset.lane);
        activateSegment(segments.find((segment) => segment.index === state.activeTimelineSegment) || segments[0]);
      });
      canvas.addEventListener("pointermove", (event) => {
        const hit = canvasHit(canvas, event);
        if (!hit) {
          helper.hideTooltip(state.streamTooltip);
          return;
        }
        helper.showTooltip(state.streamTooltip, streamTask(hit.segment, canvas.dataset.lane), event, {
          bounds: els.streamTimelinePanel,
          target: canvas,
          durationUnit: "us",
        });
      });
      canvas.addEventListener("pointerleave", () => helper.hideTooltip(state.streamTooltip));
    });
  }

  function activateTimelineTab(tabName) {
    state.activeTimelineTab = tabName;
    document.querySelectorAll("[data-timeline-tab]").forEach((button) => {
      const selected = button.dataset.timelineTab === tabName;
      button.classList.toggle("is-selected", selected);
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
    els.stepTimelinePanel.hidden = tabName !== "steps";
    els.streamTimelinePanel.hidden = tabName !== "streams";
    els.traceTimelinePanel.hidden = tabName !== "trace";
    if (tabName === "steps") els.timelineCaption.textContent = t("stepCaption", STEP_SUMMARY.step);
    if (tabName === "streams") els.timelineCaption.textContent = t("streamCaption", TIMELINE.length, STREAM_SUMMARY.length);
    syncStreamZoomControls();
    if (tabName === "streams") scheduleStreamDraw();
    if (tabName === "trace") state.traceController?.redraw();
  }

  function renderFooterStatus() {
    const selected = REPORTS[state.selectedNodeId];
    els.footerStatus.textContent = t(
      "footerStatus",
      reportModel.counts.analysisNodes,
      reportModel.counts.timelineEvents,
      selected?.metricShort || t("noSelectionShort"),
    );
  }

  function persistPreference(key, value) {
    try {
      localStorage.setItem(key, value);
    } catch (_error) {
      // Persistence is optional; the visible preference still applies.
    }
  }

  function setLanguage(language) {
    state.language = language === "zh" ? "zh" : "en";
    persistPreference("dsv32-report-language", state.language);
    applyStaticTranslations();
    syncPreferenceControls();
    syncBottomPanelToggle();
    const transform = state.architectureController?.getTransform();
    const activeNodeId = state.selectedArchitectureId;
    if (state.activeArchitectureView === "architecture") {
      renderArchitecture({ initialTransform: transform, activeNodeId });
    }
    if (state.activeArchitectureView === "operators") renderNodeList();
    renderStepTimeline();
    renderStreamTimeline();
    renderHbmTimeline();
    state.traceController?.setLanguage(state.language);
    activateTimelineTab(state.activeTimelineTab);
    renderInspector();
    els.timelineTabSteps.textContent = state.language === "zh"
      ? `步骤 ${STEP_SUMMARY.step}`
      : `Step ${STEP_SUMMARY.step}`;
    document.title = reportPageTitle();
    els.workspaceTitle.textContent = reportPageTitle();
  }

  function setTheme(theme) {
    state.theme = theme === "light" ? "light" : "dark";
    persistPreference("dsv32-report-theme", state.theme);
    syncPreferenceControls();
    const transform = state.architectureController?.getTransform();
    const activeNodeId = state.selectedArchitectureId;
    if (state.activeArchitectureView === "architecture") {
      renderArchitecture({ initialTransform: transform, activeNodeId });
    }
    scheduleStreamDraw();
    state.traceController?.redraw();
  }

  function setInspectorExpanded(expanded) {
    const gutter = els.inspectorPane.previousElementSibling?.matches?.(".pto-workbench-shell__split-gutter")
      ? els.inspectorPane.previousElementSibling
      : null;
    els.inspectorPane.hidden = !expanded;
    els.inspectorPane.setAttribute("aria-hidden", String(!expanded));
    if (gutter) gutter.hidden = !expanded;
    els.inspectorToggle?.classList.toggle("is-selected", expanded);
    els.inspectorToggle?.setAttribute("aria-expanded", String(expanded));
    els.inspectorToggle?.setAttribute("aria-pressed", String(expanded));
    window.requestAnimationFrame(() => {
      ideFrameInstance?.refresh();
      if (state.activeArchitectureView === "architecture") state.architectureController?.fit();
      drawStreamCanvases();
    });
  }

  function syncBottomPanelToggle() {
    const action = t(state.bottomPanelExpanded ? "hideBottomPanel" : "showBottomPanel");
    els.bottomPanelToggle?.classList.toggle("is-selected", state.bottomPanelExpanded);
    els.bottomPanelToggle?.setAttribute("aria-expanded", String(state.bottomPanelExpanded));
    els.bottomPanelToggle?.setAttribute("aria-pressed", String(state.bottomPanelExpanded));
    els.bottomPanelToggle?.setAttribute("aria-label", action);
    if (els.bottomPanelToggle) els.bottomPanelToggle.dataset.reportTooltip = action;
  }

  function setBottomPanelExpanded(expanded) {
    state.bottomPanelExpanded = Boolean(expanded);
    const gutter = els.bottomDock.previousElementSibling?.matches?.(".pto-workbench-shell__split-gutter")
      ? els.bottomDock.previousElementSibling
      : null;
    els.bottomDock.hidden = !state.bottomPanelExpanded;
    els.bottomDock.setAttribute("aria-hidden", String(!state.bottomPanelExpanded));
    if (gutter) gutter.hidden = !state.bottomPanelExpanded;
    syncBottomPanelToggle();
    window.requestAnimationFrame(() => {
      ideFrameInstance?.refresh();
      if (state.activeArchitectureView === "architecture") state.architectureController?.fit();
      if (state.bottomPanelExpanded) drawStreamCanvases();
      if (state.bottomPanelExpanded) state.traceController?.resize();
    });
  }

  els.nodeList.addEventListener("click", (event) => {
    const toggle = event.target.closest("[data-tree-toggle]");
    if (toggle) {
      const itemId = toggle.dataset.treeToggle;
      if (state.operatorTreeExpandedIds.has(itemId)) state.operatorTreeExpandedIds.delete(itemId);
      else state.operatorTreeExpandedIds.add(itemId);
      renderNodeList();
      return;
    }
    const button = event.target.closest("[data-node-id]");
    if (button) selectNode(button.dataset.nodeId, { source: "operator-list" });
  });
  els.nodeList.addEventListener("change", (event) => {
    const selector = event.target.closest("[data-layer-selector]");
    if (!selector) return;
    const instanceIndex = Number(selector.value);
    if (!Number.isFinite(instanceIndex)) return;
    state.repeatInstanceSelections.set(selector.dataset.layerSelector, instanceIndex);
    renderNodeList();
  });
  els.languageToggle?.addEventListener("click", () => {
    setLanguage(state.language === "en" ? "zh" : "en");
  });
  els.themeToggle?.addEventListener("click", () => {
    setTheme(state.theme === "light" ? "dark" : "light");
  });
  els.inspectorToggle?.addEventListener("click", () => setInspectorExpanded(els.inspectorPane.hidden));
  els.inspectorClose?.addEventListener("click", () => setInspectorExpanded(false));
  els.bottomPanelToggle?.addEventListener("click", () => {
    setBottomPanelExpanded(!state.bottomPanelExpanded);
  });
  els.streamZoomOut?.addEventListener("click", () => setStreamZoom(state.streamZoomIndex - 1));
  els.streamZoomReset?.addEventListener("click", () => setStreamZoom(0));
  els.streamZoomIn?.addEventListener("click", () => setStreamZoom(state.streamZoomIndex + 1));
  document.querySelectorAll("[data-timeline-tab]").forEach((button) => {
    button.addEventListener("click", () => activateTimelineTab(button.dataset.timelineTab));
  });
  document.querySelectorAll("[data-architecture-view]").forEach((button) => {
    button.addEventListener("click", () => activateArchitectureView(button.dataset.architectureView));
  });
  els.nodeViewsRailButton?.addEventListener("click", () => activateArchitectureView("operators"));
  document.addEventListener("pointerover", (event) => {
    const trigger = event.target.closest?.("[data-metric-key], [data-report-tooltip]");
    const relatedTarget = event.relatedTarget instanceof Node ? event.relatedTarget : null;
    if (trigger && !trigger.contains(relatedTarget)) showReportTooltip(trigger);
  });
  document.addEventListener("pointerout", (event) => {
    const trigger = event.target.closest?.("[data-metric-key], [data-report-tooltip]");
    const relatedTarget = event.relatedTarget instanceof Node ? event.relatedTarget : null;
    if (trigger && !trigger.contains(relatedTarget)) hideReportTooltip();
  });
  document.addEventListener("focusin", (event) => {
    const trigger = event.target.closest?.("[data-metric-key], [data-report-tooltip]");
    if (trigger) showReportTooltip(trigger);
  });
  document.addEventListener("focusout", (event) => {
    const trigger = event.target.closest?.("[data-metric-key], [data-report-tooltip]");
    const relatedTarget = event.relatedTarget instanceof Node ? event.relatedTarget : null;
    if (trigger && !trigger.contains(relatedTarget)) hideReportTooltip();
  });
  document.addEventListener("click", (event) => {
    const trigger = event.target.closest?.("[data-metric-key], [data-report-tooltip]");
    if (trigger) showReportTooltip(trigger);
  });
  document.addEventListener("pointerdown", (event) => {
    if (!event.target.closest?.("[data-metric-key], [data-report-tooltip], #reportFieldTooltip")) {
      hideReportTooltip();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") hideReportTooltip();
  });
  window.addEventListener("resize", () => {
    positionReportTooltip(state.tooltipTrigger);
    scheduleStreamDraw();
    state.traceController?.resize();
  });
  document.addEventListener("scroll", hideReportTooltip, true);

  document.title = reportPageTitle();
  els.workspaceTitle.textContent = reportPageTitle();
  els.timelineTabSteps.textContent = state.language === "zh"
    ? `步骤 ${STEP_SUMMARY.step}`
    : `Step ${STEP_SUMMARY.step}`;
  const initialGraphNodeId = window.DeepSeekArchitectureData.backendToGraphId(
    architectureGraphSpec,
    state.selectedNodeId,
  );
  renderArchitecture({ activeNodeId: initialGraphNodeId });
  activateArchitectureView("architecture");
  renderStepTimeline();
  renderStreamTimeline();
  renderTraceTimeline();
  renderHbmTimeline();
  activateTimelineTab("trace");
  renderInspector();
  setInspectorExpanded(window.innerWidth > 700);
  setBottomPanelExpanded(true);
})().catch((error) => {
  document.getElementById("footerStatus").textContent = error.message;
  console.error(error);
});
