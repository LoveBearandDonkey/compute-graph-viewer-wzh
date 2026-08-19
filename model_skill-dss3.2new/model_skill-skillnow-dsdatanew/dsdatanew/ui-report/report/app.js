(async function initializeReportWorkbench() {
  "use strict";

  const els = {
    ideFrame: document.getElementById("ideFrame"),
    workspaceTitle: document.getElementById("workspaceTitle"),
    nodeList: document.getElementById("nodeList"),
    architectureGraph: document.getElementById("architectureGraph"),
    architectureGraphPanel: document.getElementById("architectureGraphPanel"),
    operatorTreePanel: document.getElementById("operatorTreePanel"),
    operatorColorLegend: document.getElementById("operatorColorLegend"),
    operatorColorLegendMin: document.getElementById("operatorColorLegendMin"),
    operatorColorLegendMax: document.getElementById("operatorColorLegendMax"),
    architectureViewTab: document.getElementById("architectureViewTab"),
    operatorListViewTab: document.getElementById("operatorListViewTab"),
    nodeViewsRailButton: document.getElementById("nodeViewsRailButton"),
    inspectorTitle: document.getElementById("inspectorTitle"),
    inspectorNodeId: document.getElementById("inspectorNodeId"),
    inspectorSummary: document.getElementById("inspectorSummary"),
    coreMetricGrid: document.getElementById("coreMetricGrid"),
    metricGrid: document.getElementById("metricGrid"),
    diagnosisSection: document.getElementById("diagnosisSection"),
    diagnosisCard: document.getElementById("diagnosisCard"),
    fieldTooltip: document.getElementById("reportFieldTooltip"),
    operatorList: document.getElementById("operatorList"),
    operatorSummaryTab: document.getElementById("operatorSummaryTab"),
    operatorSequenceTab: document.getElementById("operatorSequenceTab"),
    operatorEventTooltip: document.getElementById("operatorEventTooltip"),
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
    reportInfoToggle: document.getElementById("reportInfoToggle"),
    reportInfoPanel: document.getElementById("reportInfoPanel"),
    reportInfoClose: document.getElementById("reportInfoClose"),
    reportInfoTitle: document.getElementById("reportInfoTitle"),
    reportInfoSummary: document.getElementById("reportInfoSummary"),
    reportInfoDetails: document.getElementById("reportInfoDetails"),
    themeToggle: document.getElementById("themeToggle"),
    themeToggleIcon: document.getElementById("themeToggleIcon"),
    bottomPanelToggle: document.getElementById("bottomPanelToggle"),
    bottomDock: document.getElementById("reportBottomDock"),
    footerStatus: document.getElementById("footerStatus"),
  };

  const state = {
    selectedNodeId: "",
    selectedArchitectureId: "",
    selectedLayerIndex: null,
    activeArchitectureView: "architecture",
    activeTimelineTab: "trace",
    activeTimelineSegment: -1,
    inspectorOperatorTab: "summary",
    selectedTraceSourceIndex: null,
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
      allLayers: "All Layers",
      selectAllLayers: "Select all decoder layers",
      selectLayer: (index) => `Select decoder layer; current selection is Layer ${index}`,
      aggregateTimeShareHint: (label, value) => `${label}: ${value} is the aggregate time share across all included layers or events. It is a total, not a single-layer hotspot or severity warning.`,
      operatorColorLegend: "Operator time heatmap",
      operatorColorLegendAria: "Operator time heatmap legend",
      expand: "Expand",
      collapse: "Collapse",
      operatorTreeStatus: (count) => `${count} backend nodes · hierarchical view`,
      inspector: "Inspector",
      coreEventMetrics: "Core Event Metrics",
      performanceMetrics: "Performance Metrics",
      diagnosisAdvice: "Diagnostic Advice",
      diagnosisNextData: "Next data to inspect",
      diagnosisScopeNote: "Scope note",
      operators: "Operators",
      operatorInspectorViews: "Operator Inspector views",
      operatorSummary: "Operator Summary",
      operatorSequence: "Operator Sequence",
      operatorsDeduplicatedHint: "Deduplicated by operator name within the current node's metric_scope. Each name appears once; the percentage is the aggregated time share for all operators with that name.",
      noOperatorSequenceForNode: "No concrete operator instances are mapped to this node and Layer.",
      operatorSequenceLayer: (index) => `Layer ${index}`,
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
      primaryMetric: "Primary metric",
      noOperatorRatio: "No operator ratio selected.",
      noOperatorRatioForNode: "No operator ratio is present for this node.",
      switchToChinese: "切换到中文",
      switchToEnglish: "Switch to English",
      reportInfo: "Report generation info",
      closeReportInfo: "Close report generation info",
      generatedBy: (skills, model, extractor) => `This report was generated by the ${skills} skills from the ${model} model source, extracted with ${extractor}.`,
      generationSkills: "Pipeline skills",
      modelSource: "Model source",
      extractorModel: "Extraction model",
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
      allLayers: "全部层",
      selectAllLayers: "选择全部解码层",
      selectLayer: (index) => `选择解码层；当前为第 ${index} 层`,
      aggregateTimeShareHint: (label, value) => `${label}：${value} 是所含全部层或事件的聚合总耗时占比，表示总量，不代表单层热点或异常等级。`,
      operatorColorLegend: "算子耗时热力图",
      operatorColorLegendAria: "算子耗时热力图例",
      expand: "展开",
      collapse: "收起",
      operatorTreeStatus: (count) => `${count} 个后端节点 · 层级视图`,
      inspector: "检查器",
      coreEventMetrics: "核心事件指标",
      performanceMetrics: "性能指标",
      diagnosisAdvice: "诊断建议",
      diagnosisNextData: "下一步查看",
      diagnosisScopeNote: "口径说明",
      operators: "算子",
      operatorInspectorViews: "算子检查器视图",
      operatorSummary: "算子汇总",
      operatorSequence: "算子序列",
      operatorsDeduplicatedHint: "按当前节点 metric_scope 内的算子名称去重汇总；同名算子只展示一次，百分比是该名称下全部算子的合计时间占比。",
      noOperatorSequenceForNode: "当前节点和层没有映射到具体算子实例。",
      operatorSequenceLayer: (index) => `第 ${index} 层`,
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
      primaryMetric: "主指标",
      noOperatorRatio: "未选择算子占比。",
      noOperatorRatioForNode: "后端未提供该节点的算子占比。",
      switchToChinese: "切换到中文",
      switchToEnglish: "切换到英文",
      reportInfo: "报告生成信息",
      closeReportInfo: "关闭报告生成信息",
      generatedBy: (skills, model, extractor) => `本报告由 ${skills} Skill 基于 ${model} 模型源码，使用 ${extractor} 提取生成。`,
      generationSkills: "流水线 Skill",
      modelSource: "模型源码",
      extractorModel: "提取模型",
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

  function coreMetricCalculation(label, value, details = {}) {
    const metricValue = String(value || "–");
    const formatOperand = (number) => Number(number).toLocaleString("en-US", { maximumFractionDigits: 3 });
    if (label === "wall_ms" && Number.isFinite(details.first_start_ms) && Number.isFinite(details.last_end_ms)) {
      return `${formatOperand(details.last_end_ms)} ms − ${formatOperand(details.first_start_ms)} ms = ${metricValue}`;
    }
    if (label === "busy_union_ms" && Number.isFinite(details.merged_interval_count)) {
      return state.language === "zh"
        ? `Σ ${details.merged_interval_count} 个合并区间 = ${metricValue}`
        : `Σ ${details.merged_interval_count} merged intervals = ${metricValue}`;
    }
    if (label === "kernel_sum_ms" && Number.isFinite(details.kernel_count)) {
      return state.language === "zh"
        ? `Σ ${details.kernel_count} 个 kernel duration = ${metricValue}`
        : `Σ ${details.kernel_count} kernel durations = ${metricValue}`;
    }
    if (label === "total_cost_ms" && Number.isFinite(details.kernel_count)) {
      return state.language === "zh"
        ? `Σ（duration + wait），${details.kernel_count} 个 kernel = ${metricValue}`
        : `Σ (duration + wait), ${details.kernel_count} kernels = ${metricValue}`;
    }
    const fallback = {
      wall_ms: state.language === "zh" ? "末次 kernel 结束时间 − 首次 kernel 开始时间" : "last kernel end − first kernel start",
      busy_union_ms: state.language === "zh" ? "Σ 合并后的 kernel 时间区间长度" : "Σ merged kernel interval lengths",
      kernel_sum_ms: "Σ kernel duration",
      total_cost_ms: "Σ (duration + wait)",
    };
    return fallback[label] || "";
  }

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
    els.reportInfoToggle?.setAttribute("aria-label", t("reportInfo"));
    if (els.reportInfoToggle) els.reportInfoToggle.dataset.reportTooltip = t("reportInfo");
    els.reportInfoClose?.setAttribute("aria-label", t("closeReportInfo"));
  }

  applyStaticTranslations();
  syncPreferenceControls();

  const ideFrameInstance = window.PtoIdeFrame?.init(els.ideFrame) || null;
  const runtimeConfig = window.ReportRuntimeConfig;
  if (!runtimeConfig) throw new Error("Missing window.ReportRuntimeConfig; regenerate this report through Skill 3");

  function reportProvenance() {
    const configured = runtimeConfig.provenance || {};
    const skills = Array.isArray(configured.skills) ? configured.skills.filter(Boolean) : [];
    const sourceRefs = analysisConfig?.architecture?.source_of_truth || [];
    return {
      skills: skills.length ? skills : ["1-perf-breakdown", "2-adapt-breakdown-to-ui-json"],
      modelSource: configured.modelSource || sourceRefs.join("; ") || reportModel?.identity?.modelId || "–",
      extractorModel: configured.extractorModel || analysisConfig?.generation_provenance?.extractor_model || "–",
      status: configured.status || analysisConfig?.generation_provenance?.status || "",
    };
  }

  function renderReportInfo() {
    if (!reportModel || !analysisConfig) return;
    const provenance = reportProvenance();
    const skillLabel = provenance.skills.join(" + ");
    const modelLabel = analysisConfig.model_name || reportModel.identity.modelId;
    els.reportInfoTitle.textContent = t("reportInfo");
    els.reportInfoSummary.textContent = t("generatedBy", skillLabel, modelLabel, provenance.extractorModel);
    const rows = [
      [t("generationSkills"), skillLabel],
      [t("modelSource"), provenance.modelSource],
      [t("extractorModel"), provenance.extractorModel],
    ];
    els.reportInfoDetails.innerHTML = rows.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("");
  }

  function setReportInfoExpanded(expanded) {
    if (!els.reportInfoPanel) return;
    els.reportInfoPanel.hidden = !expanded;
    els.reportInfoToggle?.setAttribute("aria-expanded", String(expanded));
    els.reportInfoToggle?.classList.toggle("is-selected", expanded);
    if (expanded) {
      hideReportTooltip();
      renderReportInfo();
    }
  }

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

  function layerTimingTotals(document, rawTrace = false) {
    const totals = new Map();
    traceEventsFrom(document).forEach((event) => {
      if (rawTrace && event?.ph !== "X") return;
      const rawInstanceIndex = rawTrace
        ? event.args?.layer_index ?? event.args?.layerIndex
        : event.layer_index ?? event.layerIndex;
      if (rawInstanceIndex == null) return;
      const instanceIndex = Number(rawInstanceIndex);
      const durationUs = Number(event.dur ?? event.duration_us);
      if (!Number.isInteger(instanceIndex) || !Number.isFinite(durationUs) || durationUs < 0) return;
      totals.set(instanceIndex, (totals.get(instanceIndex) || 0) + durationUs);
    });
    return totals;
  }

  function buildRepeatInstanceMetrics(traceDocument, timelineDocument) {
    const traceTotals = layerTimingTotals(traceDocument, true);
    const timelineTotals = layerTimingTotals(timelineDocument, false);
    // Prefer the raw capture when it has at least the same layer coverage. Some captures
    // (Gemma) do not tag raw Trace events with layer_index at all even though normalized
    // Timeline ownership is complete enough to color the observed Layer dots.
    const totals = traceTotals.size >= timelineTotals.size ? traceTotals : timelineTotals;
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
  let operatorDetails;
  let hbmData;
  let deviceProfile;
  let metricsFindings;
  let expertInventory;
  let repeatInstanceMetrics = [];
  try {
    const [analysis, perf, timeline, traceView, bindings, details, graphSpec, overlayMap, hbm, findings, experts] = await Promise.all([
      loadJson("analysis", runtimeConfig.analysis),
      loadJson("performance", runtimeConfig.performance),
      loadJson("timeline", runtimeConfig.timeline),
      loadJson("trace", runtimeConfig.trace),
      loadJson("bindings", runtimeConfig.bindings),
      loadJson("operatorDetails", runtimeConfig.operatorDetails),
      loadJson("architecture", runtimeConfig.architecture),
      loadJson("overlay", runtimeConfig.overlay),
      loadJson("hbm", runtimeConfig.hbm),
      loadJson("findings", runtimeConfig.findings),
      loadJson("expertInventory", runtimeConfig.expertInventory),
    ]);
    analysisConfig = analysis;
    architectureGraphSpec = graphSpec;
    architectureOverlayMap = overlayMap;
    rawTimelineData = timeline;
    rawTraceDocument = traceView;
    traceBindings = bindings;
    operatorDetails = details;
    hbmData = hbm;
    deviceProfile = perf?.device_profile || (Number.isFinite(Number(perf?.aicore_freq_mhz))
      ? { aicore_freq_mhz: Number(perf.aicore_freq_mhz), aicore_freq_basis: "reported" }
      : null);
    metricsFindings = findings;
    expertInventory = experts;
    architectureGraphSpec = window.DeepSeekArchitectureData.applyExpertInventory(graphSpec, experts);
    repeatInstanceMetrics = buildRepeatInstanceMetrics(traceView, timeline);
    reportModel = window.DeepSeekReportData.createReportModel(analysis, perf, timeline);
    architectureGraph = window.DeepSeekArchitectureData.createArchitectureGraph(architectureGraphSpec, reportModel.reports);
    state.collapsedArchitectureIds = new Set(window.DeepSeekArchitectureData.defaultCollapsedIds(architectureGraphSpec));
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
  const TIMELINE_LAYER_BY_OP_INDEX = new Map(
    (rawTimelineData?.events || []).map((event) => [Number(event.op_index), event.layer_index]),
  );
  const TRACE_BINDINGS = (traceBindings?.bindings || []).map((binding) => ({
    ...binding,
    model_layer_index: binding.model_layer_index
      ?? TIMELINE_LAYER_BY_OP_INDEX.get(Number(binding.op_index))
      ?? null,
  }));
  const TRACE_MODEL = window.DeepSeekTraceView.parseTrace(rawTraceDocument, TRACE_BINDINGS);
  const TRACE_BINDING_BY_SOURCE = new Map(TRACE_BINDINGS.map((binding) => [Number(binding.raw_source_event_index), binding]));
  const TIMELINE_EVENT_BY_ID = new Map((rawTimelineData?.events || []).map((event) => [event.event_id, event]));
  const OPERATOR_DETAIL_BY_INDEX = new Map(Object.entries(operatorDetails?.details || {}).map(([index, detail]) => [Number(index), detail]));

  function indexAnalysisNodes(value, result = new Map()) {
    if (Array.isArray(value)) {
      value.forEach((item) => indexAnalysisNodes(item, result));
      return result;
    }
    if (!value || typeof value !== "object") return result;
    if (typeof value.node_id === "string") result.set(value.node_id, value);
    Object.values(value).forEach((item) => indexAnalysisNodes(item, result));
    return result;
  }

  const ANALYSIS_NODE_BY_ID = indexAnalysisNodes(analysisConfig);
  const TIMELINE_NODE_COUNTS = TIMELINE.reduce((counts, event) => {
    if (event.nodeId) counts.set(event.nodeId, (counts.get(event.nodeId) || 0) + 1);
    return counts;
  }, new Map());
  function activeReports() {
    return reportModel.reportsForLayer(state.selectedLayerIndex);
  }

  function activeReport(nodeId) {
    return activeReports()[nodeId] || null;
  }

  function alignSelectedNodeToLayer() {
    const layerIndex = Number(state.selectedLayerIndex);
    if (!Number.isInteger(layerIndex) || !state.selectedNodeId) return;
    // Preserve the operator role across sibling templates by translating its path relative
    // to the old layer template. Example: layer_0/self_attn_0/q_proj maps to
    // decoder_layers/self_attn_0/q_proj, not to the decoder_layers root.
    const correspondingItem = window.DeepSeekArchitectureData.correspondingLayerItemForIndex?.(
      architectureGraphSpec,
      state.selectedArchitectureId,
      layerIndex,
    );
    if (!correspondingItem?.backendNodeId || !REPORTS[correspondingItem.backendNodeId]) {
      // Keep the user's operator identity selected. Missing or ambiguous target-layer
      // membership is not permission to jump to another node or cancel the selection.
      return;
    }
    state.selectedNodeId = correspondingItem.backendNodeId;
    state.selectedArchitectureId = correspondingItem.id;
    state.activeTimelineSegment = -1;
    state.pendingArchitectureCenterNodeId = "";
    expandOperatorAncestors(state.selectedNodeId);
  }

  function reportHeatValues() {
    return Object.values(activeReports())
      .map((report) => metricPercent(report.metricShort))
      .filter((value) => Number.isFinite(value) && value > 0);
  }
  const OPERATOR_TREE = createOperatorTree(analysisConfig);
  const DEFAULT_LAYER_SELECTOR = OPERATOR_TREE.find((item) => item.layerSelector)?.layerSelector || null;
  if (DEFAULT_LAYER_SELECTOR) {
    state.selectedLayerIndex = DEFAULT_LAYER_SELECTOR.representativeIndex;
    state.repeatInstanceSelections.set(
      DEFAULT_LAYER_SELECTOR.repeatNodeId,
      DEFAULT_LAYER_SELECTOR.representativeIndex,
    );
  }
  const OPERATOR_TREE_PARENTS = new Map();
  indexOperatorTree(OPERATOR_TREE, "", OPERATOR_TREE_PARENTS);
  expandOperatorTree(OPERATOR_TREE);
  state.selectedNodeId = REPORT_ORDER[0] || "";
  state.selectedArchitectureId = window.DeepSeekArchitectureData.backendToGraphId(
    architectureGraphSpec,
    state.selectedNodeId,
  );
  const initialLayerTemplate = window.DeepSeekArchitectureData.layerTemplateForIndex?.(
    architectureGraphSpec,
    state.selectedLayerIndex,
  );
  if (initialLayerTemplate?.backendNodeId && REPORTS[initialLayerTemplate.backendNodeId]) {
    state.selectedNodeId = initialLayerTemplate.backendNodeId;
    state.selectedArchitectureId = initialLayerTemplate.id;
  }
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

  function performanceBadgeStyle(value, domainValues = reportHeatValues()) {
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

  function updateOperatorColorLegend(architectureView, heatmap) {
    if (!els.operatorColorLegend) return;
    const enabled = state.activeArchitectureView === "architecture"
      && heatmap?.enabled === true
      && Number(heatmap.mappedNodeCount) > 0;
    els.operatorColorLegend.hidden = !enabled;
    if (!enabled) return;
    const metricUnit = "%";
    const formatValue = (value) => {
      const number = Number(value);
      if (!Number.isFinite(number)) return "—";
      const precision = number >= 10 ? 1 : number >= 1 ? 2 : 3;
      return `${Number(number.toFixed(precision))}${metricUnit}`;
    };
    if (els.operatorColorLegendMin) els.operatorColorLegendMin.textContent = formatValue(heatmap.minValue);
    if (els.operatorColorLegendMax) els.operatorColorLegendMax.textContent = formatValue(heatmap.maxValue);
  }

  function isAggregateReport(report) {
    const dimension = String(report?.dimension || "").toLowerCase();
    return dimension.includes("aggregate")
      || report?.facts?.some((fact) => /^Metric scope:\s*(aggregate|phase_aggregate)\b/i.test(String(fact)));
  }

  function aggregateTimeShareHint(report) {
    return t("aggregateTimeShareHint", report.title, report.metricShort);
  }

  function setInspectorSummary(value = "") {
    const summary = String(value || "").trim();
    els.inspectorSummary.textContent = summary;
    els.inspectorSummary.hidden = !summary;
  }

  function metricTileMarkup([label, value], options = {}) {
      const localizedLabel = state.language === "zh" ? (METRIC_LABELS_ZH[label] || label) : label;
      const visibleMetricLabel = escapeHtml(localizedLabel).replaceAll("_", "_<wbr>");
      const percent = metricPercent(value);
      const fullWidthClass = !options.core && label === "time share" ? " metric-tile--full" : "";
      const primaryClass = options.core && label === "wall_ms" ? " metric-tile--primary" : "";
      const formula = options.core ? coreMetricCalculation(label, value, options.details) : "";
      const primaryBadge = primaryClass ? `<span class="metric-primary-badge">${escapeHtml(t("primaryMetric"))}</span>` : "";
      const bar = options.core || percent == null
        ? ""
        : `<div class="metric-bar" style="--metric-bar-width:${percent}%"><span></span></div>`;
      return `
        <div class="metric-tile${fullWidthClass}${primaryClass}" data-tone="info">
          <div class="metric-title-row">
            <button class="metric-label metric-label-trigger" type="button" data-metric-key="${escapeHtml(label)}" aria-label="${escapeHtml(`${localizedLabel}: ${METRIC_DEFINITIONS[label]?.[state.language] || ""}`)}">
              <span>${visibleMetricLabel}</span>
              <svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6"></circle><path d="M8 7.1v4M8 4.7h.01"></path></svg>
            </button>
            ${primaryBadge}
          </div>
          <div class="metric-value">${escapeHtml(value)}</div>
          ${formula ? `<div class="metric-formula">${escapeHtml(formula)}</div>` : ""}
          ${bar}
        </div>
      `;
  }

  function renderMetrics(report) {
    els.coreMetricGrid.innerHTML = report.coreMetrics.map((metric) => metricTileMarkup(metric, { core: true, details: report.coreMetricDetails || {} })).join("");
    els.metricGrid.innerHTML = report.metrics.map((metric) => metricTileMarkup(metric)).join("");
  }

  const FINDING_SEVERITY_RANK = { high: 3, medium: 2, warning: 2, low: 1, info: 0 };

  function diagnosisPathForNodeId(nodeId) {
    const relative = String(nodeId || "").split("/").slice(2);
    if (!relative.length) return "";
    const [group, ...rest] = relative;
    const prefix = {
      decoder_layer_0_dense: "DeepseekV3DecoderLayer_dense_first",
      decoder_layers_dense: "DeepseekV3DecoderLayer_dense",
      decoder_layers_moe: "DeepseekV3DecoderLayer_moe",
      mtp_layer: "DeepseekV3DecoderLayer_mtp",
    }[group];
    if (prefix) return [prefix, ...rest].join("/");
    if (group === "stages" || group === "runtime") return rest.join("/");
    return relative.join("/");
  }

  function findingsForReport(report) {
    if (!report || !Array.isArray(metricsFindings?.nodes)) return null;
    const path = diagnosisPathForNodeId(report.nodeId);
    return metricsFindings.nodes.find((node) => node.path === path) || null;
  }

  function renderDiagnosis(report) {
    if (!els.diagnosisSection || !els.diagnosisCard) return;
    const node = findingsForReport(report);
    const findings = Array.isArray(node?.findings)
      ? [...node.findings].sort((a, b) => (FINDING_SEVERITY_RANK[b.severity] || 0) - (FINDING_SEVERITY_RANK[a.severity] || 0))
      : [];
    els.diagnosisSection.hidden = findings.length === 0;
    if (!findings.length) {
      els.diagnosisCard.innerHTML = "";
      return;
    }
    els.diagnosisCard.innerHTML = findings.map((finding) => {
      const severity = String(finding.severity || "info").toLowerCase();
      const nextData = String(finding.next_data || "").trim();
      const scopeNote = String(finding.not_applicable || "").trim();
      return `
        <article class="diagnosis-finding" data-severity="${escapeHtml(severity)}">
          <div class="diagnosis-finding__header">
            <span class="diagnosis-finding__indicator" aria-hidden="true"></span>
            <strong>${escapeHtml(finding.text || finding.code || t("diagnosisAdvice"))}</strong>
          </div>
          <p class="diagnosis-finding__advice">${escapeHtml(finding.advice_l1 || "")}</p>
          ${nextData ? `<p class="diagnosis-finding__detail"><span>${escapeHtml(t("diagnosisNextData"))}：</span>${escapeHtml(nextData)}</p>` : ""}
          ${scopeNote && scopeNote !== "无。" ? `<p class="diagnosis-finding__detail diagnosis-finding__scope"><span>${escapeHtml(t("diagnosisScopeNote"))}：</span>${escapeHtml(scopeNote)}</p>` : ""}
        </article>
      `;
    }).join("");
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
    if (trigger === els.reportInfoToggle && !els.reportInfoPanel?.hidden) {
      hideReportTooltip();
      return;
    }
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

  function renderOperatorSummary(report) {
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

  function traceEventsForInspector(report) {
    if (!report?.nodeId) return [];
    const layerIndex = report.isLayerScoped ? Number(state.selectedLayerIndex) : null;
    return TRACE_MODEL.durations.filter((event) => (
      event.ownerNodeId
      && (event.ownerNodeId === report.nodeId || event.ownerNodeId.startsWith(`${report.nodeId}/`))
      && (!Number.isInteger(layerIndex) || event.layerIndex === layerIndex)
    )).sort((left, right) => left.ts - right.ts || left.sourceIndex - right.sourceIndex);
  }

  function eventDetail(event) {
    const binding = TRACE_BINDING_BY_SOURCE.get(event.sourceIndex) || event.binding || {};
    const timelineEvent = TIMELINE_EVENT_BY_ID.get(binding.timeline_event_id) || {};
    const operator = OPERATOR_DETAIL_BY_INDEX.get(Number(binding.op_index)) || {};
    const args = event.args || {};
    const readDetail = (patterns) => {
      const entry = Object.entries({ ...timelineEvent, ...args }).find(([key, value]) => (
        value != null && value !== "" && patterns.some((pattern) => pattern.test(key))
      ));
      if (!entry) return "–";
      return typeof entry[1] === "object" ? JSON.stringify(entry[1]) : String(entry[1]);
    };
    const nodeId = binding.node_id || event.ownerNodeId || "";
    const analysisNode = ANALYSIS_NODE_BY_ID.get(nodeId) || {};
    return {
      name: timelineEvent.name || event.name || "event",
      stream: timelineEvent.stream_id ?? args["Physic Stream Id"] ?? event.tid ?? "–",
      duration: formatDuration(event.dur),
      type: timelineEvent.op_type || args.op_type || event.cat || "–",
      shape: [operator.input_shapes, operator.output_shapes].filter(Boolean).join(" → ") || readDetail([/shape/i]),
      dtype: [operator.input_data_types, operator.output_data_types].filter(Boolean).join(" → ") || readDetail([/dtype/i, /data.?type/i]),
      semantic: analysisNode.semantic || "–",
    };
  }

  function operatorTooltipHtml(event) {
    const detail = eventDetail(event);
    return operatorTooltipFields({ sequence: `#${event.streamSequence}`, ...detail });
  }

  function operatorTooltipFields(detail) {
    const labels = state.language === "zh"
      ? { sequence: "序号", name: "算子名称", stream: "Stream", duration: "耗时", type: "类型", shape: "Shape", dtype: "Dtype", semantic: "语义" }
      : { sequence: "Index", name: "Operator", stream: "Stream", duration: "Duration", type: "Type", shape: "Shape", dtype: "Dtype", semantic: "Semantic" };
    const rows = ["sequence", "name", "stream", "duration", "type", "shape", "dtype", "semantic"].map((key) => `
      <div class="operator-tooltip-field">
        <span class="operator-tooltip-field__label">${escapeHtml(labels[key])}：</span>
        <span class="operator-tooltip-field__value">${escapeHtml(detail[key] ?? "–")}</span>
      </div>
    `).join("");
    return `<div class="operator-tooltip-fields">${rows}</div>`;
  }

  function summarizedValues(values, limit = 3) {
    const unique = [...new Set(values.map((value) => String(value || "").trim()).filter((value) => value && value !== "–"))];
    if (!unique.length) return "–";
    const visible = unique.slice(0, limit);
    return `${visible.join("；")}${unique.length > limit ? `；+${unique.length - limit}` : ""}`;
  }

  function architectureTooltipHtml(node, info) {
    const backendNodeId = info?.backendNodeId || window.DeepSeekArchitectureData.graphToBackendNodeId(architectureGraphSpec, node.id) || "";
    const layerIndex = Number(state.selectedLayerIndex);
    const events = TRACE_MODEL.durations.filter((event) => (
      backendNodeId
      && event.ownerNodeId
      && (event.ownerNodeId === backendNodeId || event.ownerNodeId.startsWith(`${backendNodeId}/`))
      && (!Number.isInteger(layerIndex) || event.layerIndex === layerIndex)
    ));
    const details = events.map(eventDetail);
    return operatorTooltipFields({
      sequence: summarizedValues(events.map((event) => `#${event.streamSequence}`), 6),
      name: node.label || summarizedValues(details.map((detail) => detail.name)),
      stream: summarizedValues(details.map((detail) => detail.stream), 6),
      duration: events.length ? formatDuration(events.reduce((sum, event) => sum + Math.max(0, Number(event.dur) || 0), 0)) : "–",
      type: summarizedValues(details.map((detail) => detail.type)),
      shape: summarizedValues(details.map((detail) => detail.shape)),
      dtype: summarizedValues(details.map((detail) => detail.dtype)),
      semantic: info?.what || summarizedValues(details.map((detail) => detail.semantic), 1),
    });
  }

  function placeOperatorTooltip(pointerEvent) {
    const tooltip = els.operatorEventTooltip;
    if (!tooltip) return;
    const gap = 16;
    const bounds = tooltip.getBoundingClientRect();
    let left = pointerEvent.clientX + gap;
    let top = pointerEvent.clientY + gap;
    if (left + bounds.width > window.innerWidth - 10) left = pointerEvent.clientX - bounds.width - gap;
    if (top + bounds.height > window.innerHeight - 10) top = pointerEvent.clientY - bounds.height - gap;
    tooltip.style.left = `${Math.max(10, left)}px`;
    tooltip.style.top = `${Math.max(10, top)}px`;
  }

  function showOperatorTooltip(row, pointerEvent) {
    const sourceIndex = Number(row?.dataset.traceSourceIndex);
    const traceEvent = TRACE_MODEL.durations.find((event) => event.sourceIndex === sourceIndex);
    if (!traceEvent || !els.operatorEventTooltip) return;
    els.operatorEventTooltip.innerHTML = operatorTooltipHtml(traceEvent);
    els.operatorEventTooltip.classList.add("is-visible");
    els.operatorEventTooltip.setAttribute("aria-hidden", "false");
    placeOperatorTooltip(pointerEvent);
  }

  function hideOperatorTooltip() {
    els.operatorEventTooltip?.classList.remove("is-visible");
    els.operatorEventTooltip?.setAttribute("aria-hidden", "true");
  }

  function renderOperatorSequence(report) {
    const events = traceEventsForInspector(report);
    if (!events.length) {
      els.operatorList.innerHTML = `<div class="report-inline-empty">${escapeHtml(t("noOperatorSequenceForNode"))}</div>`;
      return;
    }
    const showLayer = state.selectedLayerIndex == null;
    els.operatorList.innerHTML = events.map((event) => {
      const selected = event.sourceIndex === state.selectedTraceSourceIndex;
      const layerLabel = showLayer && Number.isInteger(event.layerIndex)
        ? t("operatorSequenceLayer", event.layerIndex)
        : "";
      const detail = eventDetail(event);
      return `
        <button type="button" class="operator-row operator-sequence-row${selected ? " is-selected" : ""}" data-trace-source-index="${event.sourceIndex}" aria-pressed="${selected}">
          <span class="operator-sequence-number">#${event.streamSequence}</span>
          <span class="operator-sequence-main">
            <span class="operator-name">${escapeHtml(event.name || "event")}</span>
            <span class="operator-stream-tag">Stream ${escapeHtml(detail.stream)}</span>
            ${layerLabel ? `<span class="operator-layer-tag">${escapeHtml(layerLabel)}</span>` : ""}
          </span>
          <span class="operator-sequence-duration">${escapeHtml(formatDuration(event.dur))}</span>
        </button>
      `;
    }).join("");
    window.requestAnimationFrame(() => {
      els.operatorList.querySelector(".operator-row.is-selected")?.scrollIntoView({ block: "nearest" });
    });
  }

  function renderOperators(report) {
    document.querySelectorAll("[data-operator-tab]").forEach((button) => {
      const selected = button.dataset.operatorTab === state.inspectorOperatorTab;
      button.classList.toggle("is-selected", selected);
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
    if (state.inspectorOperatorTab === "sequence") renderOperatorSequence(report);
    else renderOperatorSummary(report);
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
    const graphLayerNavigation = window.DeepSeekArchitectureData.layerNavigationForGraph?.(
      architectureGraphSpec,
    );
    const layerRepeatNodeId = graphLayerNavigation?.repeatNodeId
      || layerRepeatItem?.id
      || layerBackendNodeId;
    const layerInstanceIndices = graphLayerNavigation?.instanceIndices
      || (Array.isArray(layerRepeatItem?.instanceIndices)
        ? layerRepeatItem.instanceIndices.map(Number).filter(Number.isFinite)
        : Array.from({ length: Number(layerRepeatItem?.repeatCount || 0) }, (_, index) => index));
    const representativeLayerIndex = layerRoots
      .flatMap((root) => [root, ...(root?.children || [])])
      .flatMap((node) => String(node?.node_id || "").split("/"))
      .map(Number)
      .find((index) => layerInstanceIndices.includes(index));
    const layerSelector = layerRepeatNodeId && layerInstanceIndices.length > 1
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
    const selectedIndex = state.selectedLayerIndex == null
      ? null
      : selector.instanceIndices.includes(Number(state.selectedLayerIndex))
        ? Number(state.selectedLayerIndex)
        : selector.instanceIndices.includes(savedIndex) ? savedIndex : selector.representativeIndex;
    const actionLabel = selectedIndex == null ? t("selectAllLayers") : t("selectLayer", selectedIndex);
    return `
      <label class="operator-layer-selector" data-report-tooltip="${escapeHtml(actionLabel)}">
        <span class="operator-layer-selector-label">${escapeHtml(t("layerSelector"))}</span>
        <select class="operator-layer-select" data-layer-selector="${escapeHtml(selector.repeatNodeId)}" aria-label="${escapeHtml(actionLabel)}">
          <option value=""${selectedIndex == null ? " selected" : ""}>${escapeHtml(t("allLayers"))}</option>
          ${selector.instanceIndices.map((index) => `
            <option value="${index}"${index === selectedIndex ? " selected" : ""}>${escapeHtml(`${t("layerSelector")} ${index}`)}</option>
          `).join("")}
        </select>
      </label>
    `;
  }

  function renderOperatorTreeItem(item, depth) {
    const isGroup = !item.nodeId;
    const report = item.nodeId ? activeReport(item.nodeId) : null;
    const nodeMetric = report?.metricShort;
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
        <span class="node-metric performance-value-badge${aggregate ? " is-aggregate" : ""}" style="${aggregate ? "" : performanceBadgeStyle(report.timeSharePct)}"${aggregate ? ` data-report-tooltip="${escapeHtml(aggregateHint)}" aria-label="${escapeHtml(aggregateHint)}"` : ""}>${escapeHtml(nodeMetric)}</span>
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

  function renderInspector() {
    hideReportTooltip();
    const report = activeReport(state.selectedNodeId);
    const architectureMetadata = window.DeepSeekArchitectureData.metadataForGraphItem(
      architectureGraphSpec,
      state.selectedArchitectureId,
    );
    const expertSummary = architectureMetadata?.item?.attributes?.expertInventory;
    if (!report && architectureMetadata) {
      const item = architectureMetadata.item;
      els.inspectorTitle.textContent = item.label || item.id;
      els.inspectorNodeId.textContent = item.id;
      setInspectorSummary(state.language === "zh"
        ? "此节点直接来自 model_architecture_graph.v1；它没有后端性能指标，也不会触发 TraceView 联动。"
        : "This node comes directly from model_architecture_graph.v1; it has no backend metrics and does not activate TraceView bindings.");
      els.coreMetricGrid.innerHTML = `<div class="report-inline-empty">${escapeHtml(t("noMetrics"))}</div>`;
      els.metricGrid.innerHTML = `<div class="report-inline-empty">${escapeHtml(t("noMetrics"))}</div>`;
      renderDiagnosis(null);
      els.operatorList.innerHTML = `<div class="report-inline-empty">${escapeHtml(t("noOperatorRatio"))}</div>`;
    } else if (!report) {
      els.inspectorTitle.textContent = t("selectBackendNode");
      els.inspectorNodeId.textContent = t("noNodeId");
      setInspectorSummary(t("selectHint"));
      els.coreMetricGrid.innerHTML = `<div class="report-inline-empty">${escapeHtml(t("noMetrics"))}</div>`;
      els.metricGrid.innerHTML = `<div class="report-inline-empty">${escapeHtml(t("noMetrics"))}</div>`;
      renderDiagnosis(null);
      els.operatorList.innerHTML = `<div class="report-inline-empty">${escapeHtml(t("noOperatorRatio"))}</div>`;
    } else {
      els.inspectorTitle.textContent = report.title;
      els.inspectorNodeId.textContent = report.nodeId;
      setInspectorSummary(expertSummary ? (state.language === "zh"
        ? `声明 ${expertSummary.declaredRoutedExperts} 个路由专家；每个 token 激活 ${expertSummary.expertsPerToken} 个；当前 rank 常驻 ${expertSummary.localRoutedExperts} 个。该捕获采用融合 GroupedMatmul，不能拆分为单专家耗时${expertSummary.residencyResolved ? "。" : "，且未记录 EP rank，不能确定常驻专家的全局编号。"}`
        : `${expertSummary.declaredRoutedExperts} routed experts are declared; ${expertSummary.expertsPerToken} activate per token; ${expertSummary.localRoutedExperts} reside on this rank. This capture uses fused GroupedMatmul, so per-expert timing is not separable${expertSummary.residencyResolved ? "." : ", and the missing EP rank prevents identifying resident global expert IDs."}`) : "");
      renderMetrics(report);
      renderDiagnosis(report);
      renderOperators(report);
    }
    renderNodeList();
    renderFooterStatus();
  }

  function syncTraceSelection() {
    const report = activeReport(state.selectedNodeId);
    state.traceController?.setSelectedNode(state.selectedNodeId, {
      layerIndex: report?.isLayerScoped ? state.selectedLayerIndex : null,
    });
  }

  function clearOperatorEventSelection() {
    state.selectedTraceSourceIndex = null;
    state.traceController?.clearTimeSelection();
    syncTraceSelection();
    hideOperatorTooltip();
    renderInspector();
  }

  function selectNode(nodeId, options = {}) {
    if (!REPORTS[nodeId]) return;
    if (options.toggle && state.selectedNodeId === nodeId) {
      clearSelection();
      return;
    }
    state.selectedNodeId = nodeId;
    state.hbmFollowsTrace = true;
    state.selectedArchitectureId = window.DeepSeekArchitectureData.backendToGraphId(
      architectureGraphSpec,
      nodeId,
    );
    expandOperatorAncestors(nodeId);
    if (options.source !== "timeline") state.activeTimelineSegment = -1;
    renderInspector();
    drawStreamCanvases();
    if (options.syncTrace !== false) syncTraceSelection();
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
    state.traceController?.setSelectedNode("");
    drawStreamCanvases();
    renderInspector();
  }

  function clearSelection() {
    state.selectedNodeId = "";
    state.selectedArchitectureId = "";
    state.activeTimelineSegment = -1;
    state.pendingArchitectureCenterNodeId = "";
    state.architectureController?.clearSelection();
    drawStreamCanvases();
    state.traceController?.setSelectedNode("");
    renderInspector();
  }

  function evidenceMap() {
    return Object.fromEntries(Object.entries(activeReports()).flatMap(([nodeId, report]) => {
      const graphNodeId = window.DeepSeekArchitectureData.backendToGraphId(architectureGraphSpec, nodeId);
      return graphNodeId ? [[graphNodeId, {
        backendNodeId: nodeId,
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
    if (state.selectedLayerIndex != null && indices.includes(Number(state.selectedLayerIndex))) {
      return Number(state.selectedLayerIndex);
    }
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
    if (els.operatorColorLegend) {
      els.operatorColorLegend.hidden = state.activeArchitectureView !== "architecture"
        || !state.architectureController?.svg?.ptoModelGraphviz?.metadata?.performanceHeatmap?.enabled;
    }
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
        activeReports(),
        state.collapsedArchitectureIds,
        state.selectedLayerIndex,
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
    let activeNodeId = options.activeNodeId;
    if (state.selectedArchitectureId && !state.visibleArchitectureIds.has(state.selectedArchitectureId)) {
      // Selection is semantic state, not a fallback to the active layer's container. If
      // the layer has no corresponding visual operator, keep the Inspector selection and
      // simply render no active graph node for this layer.
      activeNodeId = "";
    }
    state.architectureController?.destroy();
    state.architectureController = helper.renderController(els.architectureGraph, architectureView, {
      ariaLabel: t("graphAria", reportModel.identity.modelId),
      className: "pto-model-architecture-stage",
      autoFit: !initialTransform,
      fitMode: "readable",
      viewportPadding: 28,
      minReadableZoom: 0.68,
      initialTransform,
      activeNodeId,
      selectableClusters: true,
      metricOverlays: true,
      reportOverlays: false,
      edgeTags: false,
      performanceHeatmap: { enabled: true },
      evidenceMap: evidenceMap(),
      renderEvidence: architectureTooltipHtml,
      colormap: helper.modelArchitectureColormap(architectureView),
      onRepeatInstanceChange({ nodeId, instanceIndex }) {
        const transform = state.architectureController?.getTransform();
        const anchor = architectureItemAnchor(state.architectureViewGraph, nodeId);
        state.selectedLayerIndex = Number(instanceIndex);
        state.repeatInstanceSelections.set(nodeId, state.selectedLayerIndex);
        alignSelectedNodeToLayer();
        renderArchitecture({
          initialTransform: transform,
          activeNodeId: state.selectedArchitectureId,
          anchor: anchor ? { nodeId, ...anchor } : null,
        });
        renderInspector();
        syncTraceSelection();
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
    updateOperatorColorLegend(
      architectureView,
      state.architectureController?.svg?.ptoModelGraphviz?.metadata?.performanceHeatmap,
    );
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
        bindings: TRACE_BINDINGS,
        selectedNodeId: state.selectedNodeId,
        selectedLayerIndex: activeReport(state.selectedNodeId)?.isLayerScoped
          ? state.selectedLayerIndex
          : null,
        toolbarContainer: els.traceToolbarMount,
        stepLabel: `Step ${STEP_SUMMARY.step}`,
        stepLabelZh: `步骤 ${STEP_SUMMARY.step}`,
        onEventSelect(event) {
          state.selectedTraceSourceIndex = event.sourceIndex;
          state.inspectorOperatorTab = "sequence";
          if (!event.ownerNodeId || !REPORTS[event.ownerNodeId]) return;
          if (Number.isInteger(event.layerIndex)) {
            state.selectedLayerIndex = event.layerIndex;
            if (DEFAULT_LAYER_SELECTOR?.instanceIndices.includes(event.layerIndex)) {
              state.repeatInstanceSelections.set(
                DEFAULT_LAYER_SELECTOR.repeatNodeId,
                event.layerIndex,
              );
            }
          }
          const currentRangeContainsEvent = Boolean(
            state.selectedNodeId
            && (event.ownerNodeId === state.selectedNodeId || event.ownerNodeId.startsWith(`${state.selectedNodeId}/`)),
          );
          if (currentRangeContainsEvent) {
            renderInspector();
            drawStreamCanvases();
            return;
          }
          selectNode(event.ownerNodeId, {
            source: "trace-event",
            syncTrace: false,
            centerGraphNode: true,
          });
        },
        onVisibleRangeChange(range) {
          state.traceVisibleRange = { start: range.start, end: range.end };
          if (state.hbmFollowsTrace) {
            state.hbmController?.setRange(range.start, range.end);
          }
        },
        onViewportInteraction({ source } = {}) {
          if (source === "fit" || source === "reset") {
            state.hbmFollowsTrace = false;
            state.traceVisibleRange = null;
            state.hbmController?.resetRange();
            return;
          }
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
    els.hbmTimelinePanel.hidden = false;
    state.hbmController = window.HbmTimelineView.createHbmView(els.hbmTimelinePanel, hbmData, {
      language: state.language,
      range: state.hbmFollowsTrace ? state.traceVisibleRange : null,
      deviceProfile,
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
    const selected = activeReport(state.selectedNodeId);
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
    if (!els.reportInfoPanel?.hidden) renderReportInfo();
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
    if (selector.value === "") {
      state.selectedLayerIndex = null;
      state.repeatInstanceSelections.delete(selector.dataset.layerSelector);
    } else {
    const instanceIndex = Number(selector.value);
    if (!Number.isFinite(instanceIndex)) return;
    state.selectedLayerIndex = instanceIndex;
    state.repeatInstanceSelections.set(selector.dataset.layerSelector, instanceIndex);
    }
    alignSelectedNodeToLayer();
    const transform = state.architectureController?.getTransform();
    if (state.activeArchitectureView === "architecture") {
      renderArchitecture({ initialTransform: transform, activeNodeId: state.selectedArchitectureId });
    }
    renderInspector();
    syncTraceSelection();
  });
  els.languageToggle?.addEventListener("click", () => {
    setLanguage(state.language === "en" ? "zh" : "en");
  });
  els.reportInfoToggle?.addEventListener("click", () => setReportInfoExpanded(els.reportInfoPanel?.hidden));
  els.reportInfoClose?.addEventListener("click", () => setReportInfoExpanded(false));
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
  document.querySelectorAll("[data-operator-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      state.inspectorOperatorTab = button.dataset.operatorTab === "sequence" ? "sequence" : "summary";
      renderInspector();
    });
  });
  els.operatorList.addEventListener("click", (event) => {
    const row = event.target.closest("[data-trace-source-index]");
    if (!row) {
      clearOperatorEventSelection();
      return;
    }
    const sourceIndex = Number(row.dataset.traceSourceIndex);
    const traceEvent = TRACE_MODEL.durations.find((item) => item.sourceIndex === sourceIndex);
    if (!traceEvent) return;
    if (state.selectedTraceSourceIndex === sourceIndex) {
      clearOperatorEventSelection();
      return;
    }
    state.selectedTraceSourceIndex = sourceIndex;
    state.inspectorOperatorTab = "sequence";
    state.traceController?.setSelectedEvent(traceEvent, { notify: false });
    renderInspector();
  });
  els.operatorList.addEventListener("pointerover", (event) => {
    const row = event.target.closest("[data-trace-source-index]");
    if (row && !row.contains(event.relatedTarget)) showOperatorTooltip(row, event);
  });
  els.operatorList.addEventListener("pointermove", (event) => {
    if (event.target.closest("[data-trace-source-index]")) placeOperatorTooltip(event);
  });
  els.operatorList.addEventListener("pointerout", (event) => {
    const row = event.target.closest("[data-trace-source-index]");
    if (row && !row.contains(event.relatedTarget)) hideOperatorTooltip();
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
    if (trigger && trigger !== els.reportInfoToggle) showReportTooltip(trigger);
  });
  document.addEventListener("pointerdown", (event) => {
    if (!els.reportInfoPanel?.hidden && !event.target.closest?.("#reportInfoPanel, #reportInfoToggle")) {
      setReportInfoExpanded(false);
    }
    if (!event.target.closest?.("[data-metric-key], [data-report-tooltip], #reportFieldTooltip")) {
      hideReportTooltip();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      hideReportTooltip();
      setReportInfoExpanded(false);
    }
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
