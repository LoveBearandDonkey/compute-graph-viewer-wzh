(function initializeDistributedCommunicationObservatory() {
  'use strict';

  const mock = window.DistributedCommunicationMock;
  const sankeyPattern = window.PtoCommunicationTrafficSankeyPattern;
  const hierarchicalTimelinePattern = window.PtoHierarchicalTimelinePattern;
  const modelRankDeckPattern = window.PtoModelParallelRankDeck;
  const moeRoutingPattern = window.PtoMoeRouting;
  const moeRoutingData = window.PtoMoeRoutingData;
  const workbench = window.PtoWorkbenchShell;
  if (!mock || !sankeyPattern || !hierarchicalTimelinePattern || !modelRankDeckPattern || !moeRoutingPattern || !moeRoutingData || !window.THREE || !workbench) return;
  const ACTIVE_SERVER = 5;
  const ACTIVE_RANKS = [40, 41, 42, 43, 44, 45, 46, 47];
  const ACTIVE_SCOPE = 'Server 05 · EP8 · R40–R47';

  const copy = {
    en: {
      product: 'Communication Observatory', lens: 'MoE Route & Capacity', runContext: 'Run context', demoGuide: 'Demo guide',
      run: 'Run', step: 'Step', stage: 'Context', parallel: 'Parallel plan', baseline: 'Baseline',
      useCase: 'Use case', scope: 'Scope', views: 'Views', interaction: 'Interaction', finding: 'Finding', dataBasis: 'Data', demoTip: 'Start with Rank & Traffic, then inspect L27 in Model & Routing.',
      evidence: 'Evidence', inspector: 'Inspector', mockNotice: 'Mock artifacts for interaction validation', derived: 'DERIVED',
      router: 'Router', dispatch: 'Dispatch', compute: 'Expert Compute', combine: 'Combine',
      tokenCount: 'Token count', attribution: 'Hierarchical route attribution', aggregated: 'AGGREGATED',
      why: 'Why it matters', alternatives: 'Alternative explanations', nextAction: 'Next action',
      validation: 'Create validation plan', timeline: 'Communication timeline',
      timelineMeta: 'Layer → Phase → Collective → Rank → Task', timelineSelection: 'Window drives 3D traffic aggregation',
      complete: 'complete', comparable: 'Comparable', viewLayer: 'Open Model & Routing',
      modelOverview: 'Rank & Traffic', layerOverview: 'Model & Routing', globalRanks: 'Global Rank Deck',
      workspaceCommunication: 'Rank & Traffic', workspaceRouting: 'Model & Routing', singleLayer: 'Layer routing', tokenJourney: 'Token journey', mockSummary: 'PROFILE SCOPE',
      compareLayerLoad: 'Compare L27 layer load', closeComparison: 'Close comparison',
    },
    zh: {
      product: '通信观测台', lens: 'MoE 路由与容量', runContext: '运行上下文', demoGuide: 'Demo 说明',
      run: '运行', step: '步次', stage: '执行位置', parallel: '并行方案', baseline: '基线',
      useCase: '使用场景', scope: '分析范围', views: '核心视图', interaction: '主要功能', finding: '当前发现', dataBasis: '数据说明', demoTip: '先查看 Rank 与流量，再进入模型与路由检查 L27。',
      evidence: '证据', inspector: '检查器', mockNotice: '用于交互验证的 Mock 数据产物', derived: '推导结论',
      router: '路由', dispatch: '分发', compute: '专家计算', combine: '回传',
      tokenCount: 'Token 数量', attribution: '分层路由归因', aggregated: '窗口聚合',
      why: '为何重要', alternatives: '其他可能解释', nextAction: '下一步',
      validation: '创建验证计划', timeline: '通信时间线',
      timelineMeta: 'Layer → 阶段 → 集合通信 → Rank → 任务', timelineSelection: '时间窗口控制 3D 流量聚合',
      complete: '完整', comparable: '可比较', viewLayer: '打开模型与路由',
      modelOverview: 'Rank 与流量', layerOverview: '模型与路由', globalRanks: '全局 Rank Deck',
      workspaceCommunication: 'Rank 与流量', workspaceRouting: '模型与路由', singleLayer: '单层路由', tokenJourney: 'Token 整网', mockSummary: 'Profiling 范围',
      compareLayerLoad: '对比 L27 单层负载', closeComparison: '关闭对比',
    },
  };

  const diagnosisCopy = {
    zh: {
      router: { label: '路由', title: 'Layer 27 路由向 E64/E65 发送了均值 2.1× 的 Token', impact: '在分发开始前，偏斜已经消耗 R42 大部分接收容量。', next: '检查流入 E64/E65 的 Token 来源。' },
      dispatch: { label: '分发', title: 'MoE Layer 27 分发位于关键路径', impact: 'R42 上的 E64/E65 接收均值 2.1× 的负载，造成 4.8 ms 暴露等待，缓冲余量仅 6%。', next: '追踪 E64/E65 的 Token 来源，并比较 Server 05 内的 HCCS Peer 流量。' },
      compute: { label: '专家计算', title: '热点专家拉长了 Layer 27 的计算尾部', impact: '其余七张卡更早完成，但必须等待 R42 上的 E64/E65，回传才能开始。', next: '区分路由偏斜、专家部署和 Kernel 波动的影响。' },
      combine: { label: '回传', title: '回传阶段继承了分发不均衡', impact: 'R42 专家最后完成导致返回路径延迟启动；负载大小本身不是根因。', next: '验证降低分发偏斜后，回传延迟是否消失。' },
    },
  };

  const elements = {
    run: document.getElementById('context-run'),
    step: document.getElementById('context-step'),
    stage: document.getElementById('context-stage'),
    group: document.getElementById('context-group'),
    baseline: document.getElementById('context-baseline'),
    source: document.getElementById('context-source'),
    contextEvidence: document.getElementById('context-evidence'),
    contextToggle: document.getElementById('context-toggle'),
    contextPopover: document.getElementById('context-popover'),
    contextClose: document.getElementById('context-close'),
    trafficPhaseStatus: document.getElementById('traffic-phase-status'),
    sankeyOpen: document.getElementById('sankey-open'),
    sankeyClose: document.getElementById('sankey-close'),
    sankeyOverlay: document.getElementById('sankey-overlay'),
    sankeyOverlayTitle: document.getElementById('sankey-overlay-title'),
    sankeyOverlayMeta: document.getElementById('sankey-overlay-meta'),
    sankeyLegendTitle: document.getElementById('sankey-legend-title'),
    sankeyLegendColor: document.getElementById('sankey-legend-color'),
    sankeyLegendWidth: document.getElementById('sankey-legend-width'),
    sankeyLegendLocal: document.getElementById('sankey-legend-local'),
    workspaceNavigation: document.getElementById('workspace-navigation'),
    phaseScope: document.getElementById('phase-scope'),
    visualSubtitle: document.getElementById('visual-subtitle'),
    primaryTitle: document.getElementById('primary-title'),
    windowReadout: document.getElementById('window-readout'),
    aggregationReadout: document.getElementById('aggregation-readout'),
    sankeyHost: document.getElementById('sankey-host'),
    modelDeckHost: document.getElementById('model-deck-host'),
    modelDeckCanvas: document.getElementById('model-deck-canvas'),
    rankTrafficLegend: document.getElementById('rank-traffic-legend'),
    rankTrafficWorkspace: document.getElementById('rank-traffic-workspace'),
    modelRoutingWorkspace: document.getElementById('model-routing-workspace'),
    routingCanvas: document.getElementById('routing-canvas'),
    moeRoutingHost: document.getElementById('moe-routing-host'),
    moeRoutingDetailHost: document.getElementById('moe-routing-detail-host'),
    routingDetailPane: document.getElementById('routing-detail-pane'),
    routingJourneyLabel: document.getElementById('routing-journey-label'),
    routingCompareClose: document.getElementById('routing-compare-close'),
    routingDetailZoomOut: document.getElementById('routing-detail-zoom-out'),
    routingDetailFit: document.getElementById('routing-detail-fit'),
    routingDetailZoomIn: document.getElementById('routing-detail-zoom-in'),
    routingViewControl: document.getElementById('routing-view-control'),
    routingPrevious: document.getElementById('routing-previous'),
    routingNext: document.getElementById('routing-next'),
    routingReadout: document.getElementById('routing-readout'),
    routingZoomOut: document.getElementById('routing-zoom-out'),
    routingFit: document.getElementById('routing-fit'),
    routingZoomIn: document.getElementById('routing-zoom-in'),
    mainSplit: document.getElementById('main-split'),
    primaryPane: document.getElementById('primary-pane'),
    primaryRow: document.querySelector('[data-ide-pane="primary-row"]'),
    evidenceDock: document.getElementById('evidence-dock'),
    evidenceClose: document.getElementById('evidence-close'),
    evidenceTitle: document.getElementById('evidence-title'),
    evidenceKicker: document.getElementById('evidence-kicker'),
    evidenceImpact: document.getElementById('evidence-impact'),
    evidenceFacts: document.getElementById('evidence-facts'),
    evidenceAlternatives: document.getElementById('evidence-alternatives'),
    evidenceActionCopy: document.getElementById('evidence-action-copy'),
    viewExplanationSection: document.getElementById('view-explanation-section'),
    viewExplanationTitle: document.getElementById('view-explanation-title'),
    viewExplanation: document.getElementById('view-explanation'),
    validationAction: document.getElementById('validation-action'),
    routingCompareAction: document.getElementById('routing-compare-action'),
    layerViewAction: document.getElementById('layer-view-action'),
    validationPlan: document.getElementById('validation-plan'),
    timelineRoot: document.getElementById('timeline-root'),
    timelineSelection: document.getElementById('timeline-selection'),
    timelineWindowInline: document.getElementById('timeline-window-inline'),
    timelineZoomOut: document.getElementById('timeline-zoom-out'),
    timelineZoomIn: document.getElementById('timeline-zoom-in'),
    timelineZoomReadout: document.getElementById('timeline-zoom-readout'),
    timelineFit: document.getElementById('timeline-fit'),
    frame: document.querySelector('[data-ide-frame]'),
    themeToggle: document.getElementById('theme-toggle'),
    languageToggle: document.getElementById('language-toggle'),
    languageLabel: document.getElementById('language-label'),
    bottomToggle: document.getElementById('bottom-toggle'),
    evidencePanelToggle: document.getElementById('evidence-panel-toggle'),
    railEvidence: document.getElementById('rail-evidence'),
    railTimeline: document.getElementById('rail-timeline'),
    bottomDock: document.getElementById('observatory-bottom-dock'),
    statusLeft: document.getElementById('status-left'),
    statusRight: document.getElementById('status-right'),
    live: document.getElementById('observatory-live'),
  };

  const state = {
    phase: 'dispatch',
    window: { ...mock.defaultWindow },
    selection: null,
    dockResize: null,
    timelineController: null,
    language: 'en',
    bottomOpen: true,
    primarySizing: null,
    primaryView: new URLSearchParams(window.location.search).get('view') === 'routing' ? 'routing' : 'communication',
    selectedLayer: 27,
    selectedToken: 37,
    routingView: 'journey',
    routingCompare: false,
    selectedCommunicationGroup: null,
    deckController: null,
    trafficController: null,
    moeRoutingController: null,
    moeLayerDetailController: null,
    sankeyOpen: false,
  };

  function t(key) {
    return copy[state.language][key] || copy.en[key] || key;
  }

  function bilingual(en, zh) {
    return state.language === 'zh' ? zh : en;
  }

  function updateDemoGuide() {
    elements.source.textContent = bilingual('MoE communication diagnosis', 'MoE 通信诊断');
    elements.run.textContent = bilingual('Locate routing skew and correlate it with Rank traffic and critical-path waits.', '定位路由偏斜，并关联 Rank 流量与关键路径等待。');
    elements.step.textContent = '128 ranks · 16 servers × 8 cards · Server 05 EP8';
    elements.stage.textContent = bilingual('Rank & Traffic shows placement and transfers; Model & Routing shows layer load and token journeys.', '“Rank 与流量”展示部署与传输；“模型与路由”展示单层负载和 Token 整网路径。');
    elements.group.textContent = bilingual('Select L27 to inspect the anomaly, then compare its layer load in a linked split view. The timeline controls the active window.', '选择 L27 查看异常，再通过联动分屏对比该层负载；时间线控制当前分析窗口。');
    elements.baseline.textContent = bilingual('L27 routes excess load to R42 · E64/E65, exposing 4.8 ms of wait.', 'L27 向 R42 · E64/E65 路由过量负载，产生 4.8 ms 暴露等待。');
    elements.contextEvidence.textContent = bilingual('Model configuration, routing trace, Rank traffic, and timeline events.', '模型配置、路由 Trace、Rank 流量与时间线事件。');
    elements.contextToggle.title = t('demoGuide');
    elements.contextToggle.setAttribute('aria-label', t('demoGuide'));
    elements.contextClose.setAttribute('aria-label', bilingual('Close demo guide', '关闭 Demo 说明'));
  }

  function currentDiagnosis(phase = state.phase) {
    return state.language === 'zh'
      ? { ...mock.phases[phase], ...diagnosisCopy.zh[phase] }
      : mock.phases[phase];
  }

  function phaseForTime(time) {
    if (time < 115) return 'router';
    if (time < 138) return 'dispatch';
    if (time < 151) return 'compute';
    return 'combine';
  }

  function updateTrafficTools() {
    const label = currentDiagnosis().label;
    const trafficAvailable = state.phase === 'dispatch' || state.phase === 'combine';
    elements.trafficPhaseStatus.textContent = bilingual(
      `Current: ${label} · ${state.window.start}–${state.window.end} ms`,
      `当前：${label} · ${state.window.start}–${state.window.end} ms`,
    );
    elements.sankeyOpen.textContent = bilingual('View Sankey', '查看 Sankey');
    elements.sankeyOpen.hidden = !trafficAvailable;
    elements.sankeyOpen.setAttribute('aria-expanded', String(state.sankeyOpen));
    elements.sankeyOverlayMeta.textContent = `Server 05 · EP8 · ${state.window.start}–${state.window.end} ms`;
    elements.sankeyOverlayTitle.textContent = bilingual(`Rank traffic details · ${label}`, `Rank 流量详情 · ${label}`);
    elements.sankeyClose.setAttribute('aria-label', bilingual('Close Sankey', '关闭 Sankey'));
    elements.sankeyLegendTitle.textContent = bilingual('Legend', '图例');
    elements.sankeyLegendColor.textContent = bilingual('Color = source → target gradient', '颜色 = 来源 → 目标渐变');
    elements.sankeyLegendWidth.textContent = bilingual('Ribbon width = Token count', '带宽宽度 = Token 数量');
    elements.sankeyLegendLocal.textContent = bilingual('LOCAL stays on card and is not drawn', 'LOCAL 留在卡内，不绘制为跨 Rank 流量');
    if (!trafficAvailable && state.sankeyOpen) setSankeyOpen(false);
  }

  function setSankeyOpen(open) {
    state.sankeyOpen = Boolean(open) && (state.phase === 'dispatch' || state.phase === 'combine');
    elements.sankeyOverlay.hidden = !state.sankeyOpen;
    elements.modelDeckHost.classList.toggle('is-sankey-open', state.sankeyOpen);
    elements.sankeyOpen.setAttribute('aria-expanded', String(state.sankeyOpen));
    if (state.sankeyOpen) {
      sankeyController.clearSelection(false);
      state.timelineController?.focusRelatedEventIds([]);
      state.trafficController?.clearFloorTrafficFocus();
      requestAnimationFrame(() => {
        state.deckController?.refresh();
        state.deckController?.fit();
        sankeyController.setWindow(state.window.start, state.window.end, false);
      });
    } else {
      requestAnimationFrame(() => {
        state.deckController?.refresh();
        state.deckController?.fit();
      });
    }
    elements.live.textContent = bilingual(
      `Sankey details ${state.sankeyOpen ? 'opened' : 'closed'}`,
      `Sankey 流量详情已${state.sankeyOpen ? '打开' : '关闭'}`,
    );
  }

  function ensureModelDeck() {
    if (state.deckController) return state.deckController;
    state.deckController = modelRankDeckPattern.render(elements.modelDeckCanvas, {
      model: 'openpangu-flash',
      topology: {
        worldSize: 128,
        dimensions: { tp: 1, pp: 1, cp: 1, ep: 8, edp: 16 },
        rankOrder: ['edp', 'ep', 'pp', 'tp', 'cp'],
        sourceLevel: 'demo',
      },
      partitionRules: { stageRanges: [[0, 45]] },
      overviewLayout: 'server-grid',
      ranksPerServer: 8,
      serverColumns: 4,
      cameraFov: 8,
      showGroundGrid: true,
      showHeader: false,
      showInspector: false,
      syncUrl: false,
      THREE: window.THREE,
      initialState: {
        sceneMode: 'exploded',
        groupingAxes: [],
        selectedRank: null,
        selectedLayer: null,
        theme: document.documentElement.dataset.theme || 'dark',
      },
      onRankSelect(manifest) {
        openEvidence({ type: 'rank-overview', manifest });
      },
    });
    state.trafficController = state.deckController;
    syncTrafficOverlay();
    return state.deckController;
  }

  function trafficData() {
    const combine = state.phase === 'combine';
    const grouped = new Map();
    const localByRank = Object.fromEntries(mock.ranks.map(({ rank }) => [rank, { value: 0, eventIds: [] }]));
    mock.events
      .filter((event) => event.timestamp <= state.window.end && event.timestamp + event.duration >= state.window.start)
      .forEach((event) => {
        const sourceRank = combine ? event.targetRank : event.sourceRank;
        const targetRank = combine ? event.sourceRank : event.targetRank;
        if (sourceRank === targetRank) {
          localByRank[sourceRank].value += event.value;
          localByRank[sourceRank].eventIds.push(combine ? `combine-${event.id}` : event.id);
          return;
        }
        const key = `${sourceRank}:${targetRank}`;
        if (!grouped.has(key)) grouped.set(key, { sourceRank, targetRank, value: 0, eventIds: [] });
        const flow = grouped.get(key);
        flow.value += event.value;
        flow.eventIds.push(combine ? `combine-${event.id}` : event.id);
      });
    return { flows: [...grouped.values()].sort((a, b) => b.value - a.value), localByRank };
  }

  function syncTrafficOverlay() {
    if (!state.trafficController) return;
    const { flows, localByRank } = trafficData();
    const rankColorSlots = Object.fromEntries(mock.ranks.map(({ rank, slot }) => [rank, slot]));
    state.trafficController.setFloorTrafficData({ ranks: ACTIVE_RANKS, flows, localByRank, rankColorSlots });
    const total = flows.reduce((sum, flow) => sum + flow.value, 0);
    const local = Object.values(localByRank).reduce((sum, record) => sum + record.value, 0);
    elements.rankTrafficLegend.textContent = `${state.window.start}–${state.window.end} MS · NETWORK ${total.toLocaleString()} · LOCAL ${local.toLocaleString()}`;
  }

  function syncRoutingControls() {
    elements.routingViewControl.querySelectorAll('[data-routing-view]').forEach((button) => {
      const selected = button.dataset.routingView === state.routingView;
      button.classList.toggle('is-selected', selected);
      button.setAttribute('aria-selected', String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
    elements.modelRoutingWorkspace.dataset.routingView = state.routingView;
    elements.routingCanvas.classList.toggle('is-comparing', state.routingCompare);
    elements.routingDetailPane.hidden = !state.routingCompare;
    elements.routingJourneyLabel.textContent = state.routingView === 'journey'
      ? bilingual(`TOKEN T${String(state.selectedToken).padStart(3, '0')} · WHOLE-NETWORK ROUTE`, `TOKEN T${String(state.selectedToken).padStart(3, '0')} · 整网路径`)
      : bilingual(`L${state.selectedLayer} · LAYER LOAD`, `L${state.selectedLayer} · 单层负载`);
    elements.routingCompareClose.textContent = t('closeComparison');
    elements.routingReadout.textContent = state.routingCompare
      ? `T${String(state.selectedToken).padStart(3, '0')} · L${state.selectedLayer}`
      : state.routingView === 'layer'
      ? `L${state.selectedLayer}`
      : `T${String(state.selectedToken).padStart(3, '0')}`;
  }

  function ensureMoeRouting() {
    if (state.moeRoutingController) return state.moeRoutingController;
    state.moeRoutingController = moeRoutingPattern.render(elements.moeRoutingHost, {
      dataApi: moeRoutingData,
      initialView: state.routingView,
      initialLayer: state.selectedLayer,
      initialToken: state.selectedToken,
      activeRanks: ACTIVE_RANKS,
      anomalyLayers: [{
        layer: 27,
        rankIds: [42],
        expertIds: [64, 65],
        tooltip: bilingual(
          'L27 anomaly · T037 contributes to the R42 · E64/E65 hotspot; verify overload in Layer routing.',
          'L27 异常 · T037 命中 R42 · E64/E65 热点；需在单层路由中验证过载。',
        ),
      }],
      showChrome: false,
      theme: document.documentElement.dataset.theme || 'dark',
      language: state.language,
      showJourneyLegend: false,
      onSelect(selection) {
        openEvidence({ type: 'moe-routing', selection });
      },
      onStateChange(nextState) {
        const structuralChange = state.routingView !== nextState.view || state.selectedLayer !== nextState.layer || state.selectedToken !== nextState.tokenId;
        state.routingView = nextState.view;
        state.selectedLayer = nextState.layer;
        state.selectedToken = nextState.tokenId;
        syncRoutingControls();
        if (structuralChange && state.primaryView === 'routing') {
          updatePrimaryView();
          if (!elements.evidenceDock.hidden && state.selection?.type === 'diagnosis') openEvidence({ type: 'diagnosis' });
        }
      },
    });
    syncRoutingControls();
    return state.moeRoutingController;
  }

  function ensureMoeLayerDetail() {
    if (state.moeLayerDetailController) return state.moeLayerDetailController;
    state.moeLayerDetailController = moeRoutingPattern.render(elements.moeRoutingDetailHost, {
      dataApi: moeRoutingData,
      initialView: 'layer',
      initialLayer: state.selectedLayer,
      initialToken: state.selectedToken,
      activeRanks: ACTIVE_RANKS,
      anomalyLayers: [{
        layer: 27,
        rankIds: [42],
        expertIds: [64, 65],
        tooltip: bilingual(
          'L27 overload · R42 · E64/E65 receive 2.1× the mean load.',
          'L27 过载 · R42 · E64/E65 接收均值 2.1× 的负载。',
        ),
      }],
      showChrome: false,
      theme: document.documentElement.dataset.theme || 'dark',
      language: state.language,
      showJourneyLegend: false,
      onSelect(selection) {
        openEvidence({ type: 'moe-routing', selection });
      },
    });
    return state.moeLayerDetailController;
  }

  function setRoutingComparison(open, layer = 27) {
    state.routingCompare = Boolean(open);
    if (state.routingCompare) {
      state.selectedLayer = layer;
      state.routingView = 'journey';
      ensureMoeRouting().setView('journey');
      const detail = ensureMoeLayerDetail();
      detail.setLayer(layer);
      detail.setToken(state.selectedToken);
      detail.setActiveRanks(ACTIVE_RANKS);
    }
    syncRoutingControls();
    requestAnimationFrame(() => requestAnimationFrame(() => {
      state.moeRoutingController?.fit();
      if (state.routingCompare) state.moeLayerDetailController?.fit();
    }));
  }

  function selectDemoCommunicationGroup() {
    const controller = ensureModelDeck();
    state.selectedCommunicationGroup = controller.focusRankSet(ACTIVE_RANKS, {
      id: 'server-05-ep8', label: 'Server 05 · EP8', server: ACTIVE_SERVER, sourceLevel: 'demo',
    });
    state.moeRoutingController?.setActiveRanks(state.selectedCommunicationGroup.ranks);
    state.moeLayerDetailController?.setActiveRanks(state.selectedCommunicationGroup.ranks);
    state.selection = { type: 'communication-group', group: state.selectedCommunicationGroup };
    syncTrafficOverlay();
    elements.phaseScope.textContent = bilingual('Server 05 · 8-rank EP group', 'Server 05 · 8-Rank EP 通信组');
    elements.live.textContent = bilingual('EP communication group selected', '已选择 EP 通信组');
  }

  function fitRoutingAfterLayout() {
    requestAnimationFrame(() => requestAnimationFrame(() => {
      state.moeRoutingController?.fit();
      state.moeLayerDetailController?.fit();
    }));
  }

  function updatePrimaryView() {
    const communicationView = state.primaryView === 'communication';
    if (communicationView && !state.bottomOpen) setBottomOpen(true);
    if (!communicationView && state.bottomOpen) setBottomOpen(false);
    elements.primaryPane.dataset.primaryView = communicationView ? 'communication' : 'routing';
    elements.primaryTitle.removeAttribute('data-i18n');
    elements.primaryTitle.textContent = communicationView ? t('modelOverview') : t('layerOverview');
    elements.rankTrafficWorkspace.hidden = !communicationView;
    elements.modelRoutingWorkspace.hidden = communicationView;
    elements.workspaceNavigation.querySelectorAll('[data-workspace]').forEach((button) => {
      const selected = button.dataset.workspace === state.primaryView;
      button.classList.toggle('is-selected', selected);
      button.setAttribute('aria-selected', String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
    elements.layerViewAction.hidden = !communicationView;
    if (communicationView) {
      elements.visualSubtitle.textContent = bilingual(
        ACTIVE_SCOPE,
        ACTIVE_SCOPE,
      );
      elements.windowReadout.textContent = `${state.window.start}–${state.window.end} ms`;
      elements.aggregationReadout.hidden = false;
      elements.aggregationReadout.textContent = t('aggregated');
      elements.phaseScope.textContent = state.selectedCommunicationGroup
        ? bilingual('Server 05 · 8-rank EP group', 'Server 05 · 8-Rank EP 通信组')
        : bilingual('No communication group selected', '尚未选择通信组');
      requestAnimationFrame(() => {
        state.deckController?.refresh();
        sankeyController?.resize?.();
      });
      return;
    }
    elements.visualSubtitle.textContent = bilingual(
      'Model Layer context linked to 128-rank MoE routing',
      '模型层上下文与 128-Rank MoE 路由联动',
    );
    elements.windowReadout.textContent = state.routingView === 'layer'
      ? `L${state.selectedLayer} · 128 TOKENS × 256 EXPERTS × 128 RANKS · TOP-8`
      : `TOKEN T${String(state.selectedToken).padStart(3, '0')} · 256 EXPERTS / MOE LAYER · BLACK = TOP-8 · BLUE = WEIGHTED`;
    elements.aggregationReadout.hidden = true;
    elements.phaseScope.textContent = state.routingView === 'layer'
      ? bilingual(`Layer ${state.selectedLayer} · 256 experts · Top-8`, `Layer ${state.selectedLayer} · 256 专家 · Top-8`)
      : bilingual(`Token T${String(state.selectedToken).padStart(3, '0')} · L0–L45`, `Token T${String(state.selectedToken).padStart(3, '0')} · L0–L45`);
    syncRoutingControls();
    fitRoutingAfterLayout();
  }

  function setPrimaryView(view, { force = false } = {}) {
    const nextView = view === 'routing' ? 'routing' : 'communication';
    const domMatchesView = nextView === 'communication'
      ? !elements.rankTrafficWorkspace.hidden && elements.modelRoutingWorkspace.hidden
      : elements.rankTrafficWorkspace.hidden && !elements.modelRoutingWorkspace.hidden;
    if (!force && nextView === state.primaryView && domMatchesView) return;
    state.primaryView = nextView;
    if (nextView === 'communication') {
      ensureModelDeck();
      if (!elements.evidenceDock.hidden) {
        openEvidence(state.selectedCommunicationGroup
          ? { type: 'communication-group', group: state.selectedCommunicationGroup }
          : { type: 'rank-deck-overview' });
      }
      elements.timelineSelection.textContent = bilingual(
        'Rank scope, Traffic, and Timeline share the same selection',
        'Rank 范围、流量与时间线共享当前选择',
      );
    } else {
      ensureMoeRouting();
      elements.timelineSelection.textContent = bilingual(
        'Model context and MoE routing share Layer and Token state',
        '模型上下文与 MoE 路由共享 Layer 和 Token 状态',
      );
      if (!elements.evidenceDock.hidden) openEvidence({ type: 'diagnosis' });
    }
    updatePrimaryView();
    elements.live.textContent = modelViewAnnouncement(nextView);
  }

  function modelViewAnnouncement(view) {
    return view === 'communication'
      ? bilingual('Linked Rank and Traffic workspace opened', '已打开 Rank 与流量联动工作区')
      : bilingual('Linked Model and Routing workspace opened', '已打开模型与路由联动工作区');
  }

  function updateStaticCopy() {
    document.documentElement.lang = state.language === 'zh' ? 'zh-CN' : 'en';
    document.querySelectorAll('[data-i18n]').forEach((element) => {
      element.textContent = t(element.dataset.i18n);
    });
    updateTrafficTools();
    elements.languageLabel.textContent = state.language === 'en' ? '中' : 'EN';
    const languageTarget = state.language === 'en' ? '中文' : 'English';
    elements.languageToggle.title = state.language === 'en' ? '切换为中文' : 'Switch to English';
    elements.languageToggle.setAttribute('aria-label', `${state.language === 'en' ? '切换为' : 'Switch to '}${languageTarget}`);
    updateDemoGuide();
    elements.routingCompareAction.textContent = t('compareLayerLoad');
    syncRoutingControls();
    elements.statusLeft.textContent = mock.context.group;
    elements.statusRight.textContent = `${state.window.start}–${state.window.end} ms · MOCK`;
    state.timelineController?.setData(timelineData());
    state.timelineController?.setActiveRows([state.phase]);
    state.deckController?.refresh();
    if (!state.selection) elements.timelineSelection.textContent = t('timelineSelection');
    document.title = `${t('product')} · ${t('lens')}`;
    updatePrimaryView();
  }

  function phaseData(phase) {
    const combine = phase === 'combine';
    const ranks = mock.ranks.map((rank) => ({ ...rank }));
    const events = combine
      ? mock.events.map((event) => ({
        ...event,
        id: `combine-${event.id}`,
        sourceRank: event.targetRank,
        targetRank: event.sourceRank,
        sourceEngine: event.targetEngine,
        targetEngine: event.sourceEngine,
      }))
      : mock.events;
    const labels = combine
      ? {
        sourceDimensionLabel: state.language === 'zh' ? '来源 · RANK / 引擎归因' : 'SOURCE · RANK / ENGINE ATTRIBUTION',
        targetDimensionLabel: state.language === 'zh' ? '去向 · RANK / 引擎归因' : 'DESTINATION · RANK / ENGINE ATTRIBUTION',
      }
      : {
        sourceDimensionLabel: state.language === 'zh' ? '来源 · RANK / 引擎归因' : 'SOURCE · RANK / ENGINE ATTRIBUTION',
        targetDimensionLabel: state.language === 'zh' ? '去向 · RANK / 引擎归因' : 'DESTINATION · RANK / ENGINE ATTRIBUTION',
      };
    const diagnosis = currentDiagnosis(phase);
    return {
      id: `moe-layer-27-${phase}`,
      title: `MoE Layer 27 · ${diagnosis.label}`,
      ariaLabel: `MoE Layer 27 ${diagnosis.label} token route Sankey`,
      primaryMeasure: 'tokens',
      ranks,
      events,
      timeRange: mock.timeRange,
      defaultWindow: { ...state.window },
      defaultExpanded: combine ? { source: [2], target: [] } : { source: [], target: [2] },
      ...labels,
    };
  }

  function updateDiagnosis() {
    if (state.primaryView !== 'communication') {
      updatePrimaryView();
      return;
    }
    elements.phaseScope.textContent = bilingual(
      'EP group · rank-to-rank traffic only',
      'EP 通信组 · 仅 Rank 间流量',
    );
  }

  function formatTokens(value) {
    return `${new Intl.NumberFormat(state.language === 'zh' ? 'zh-CN' : 'en-US', { maximumFractionDigits: 0 }).format(value)} ${state.language === 'zh' ? '个 Token' : 'tokens'}`;
  }

  function formatWindow() {
    elements.statusRight.textContent = `${state.window.start}–${state.window.end} ms · MOCK`;
    elements.timelineWindowInline.textContent = `${state.window.start}–${state.window.end} ms`;
    if (state.primaryView === 'communication') elements.windowReadout.textContent = `${state.window.start}–${state.window.end} ms`;
    updateTrafficTools();
  }

  function setEvidenceFacts(facts) {
    elements.evidenceFacts.replaceChildren();
    facts.forEach(({ label, value, source }) => {
      const row = document.createElement('div');
      row.className = 'observatory-evidence-fact';
      const term = document.createElement('span');
      term.textContent = label;
      const strong = document.createElement('strong');
      strong.textContent = value;
      row.append(term, strong);
      if (source) {
        const note = document.createElement('small');
        note.textContent = source;
        row.appendChild(note);
      }
      elements.evidenceFacts.appendChild(row);
    });
  }

  function setViewExplanation(items = []) {
    elements.viewExplanation.replaceChildren();
    elements.viewExplanationSection.hidden = items.length === 0;
    elements.viewExplanationTitle.textContent = bilingual('View explanation', '视图说明');
    items.forEach(({ label, copy: description, anomaly = false }) => {
      const item = document.createElement('div');
      item.className = `observatory-view-explanation__item${anomaly ? ' is-anomaly' : ''}`;
      const title = document.createElement('strong');
      title.textContent = label;
      const body = document.createElement('p');
      body.textContent = description;
      item.append(title, body);
      elements.viewExplanation.appendChild(item);
    });
  }

  function ensureDockOpen() {
    if (!elements.evidenceDock.hidden) return;
    elements.evidenceDock.hidden = false;
    elements.evidencePanelToggle.classList.add('is-selected');
    elements.evidencePanelToggle.setAttribute('aria-expanded', 'true');
    elements.evidencePanelToggle.setAttribute('aria-pressed', 'true');
    state.dockResize = workbench.initResizablePanes({
      root: elements.mainSplit,
      panes: [elements.primaryPane, elements.evidenceDock],
      direction: 'horizontal',
      sizes: [72, 28],
      minSize: [980, 300],
      gutterSize: 7,
      storageKey: 'pto-communication-observatory-inspector-v2',
      keyboard: true,
    });
    requestAnimationFrame(renderTimeline);
  }

  function closeEvidence() {
    state.dockResize?.destroy?.();
    state.dockResize = null;
    elements.evidenceDock.hidden = true;
    elements.evidencePanelToggle.classList.remove('is-selected');
    elements.evidencePanelToggle.setAttribute('aria-expanded', 'false');
    elements.evidencePanelToggle.setAttribute('aria-pressed', 'false');
    elements.validationPlan.hidden = true;
    state.selection = null;
    elements.timelineSelection.textContent = t('timelineSelection');
    requestAnimationFrame(renderTimeline);
  }

  function evidenceFor(selection) {
    const diagnosis = currentDiagnosis();
    if ((!selection || selection.type === 'diagnosis') && state.primaryView === 'routing') {
      const layerView = state.routingView === 'layer';
      return {
        title: layerView
          ? bilingual(`Layer ${state.selectedLayer} routing context`, `Layer ${state.selectedLayer} 路由上下文`)
          : bilingual(`Token T${String(state.selectedToken).padStart(3, '0')} whole-network journey`, `Token T${String(state.selectedToken).padStart(3, '0')} 整网路径`),
        kicker: bilingual('MODEL + ROUTING · LINKED WORKSPACE', '模型 + 路由 · 联动工作区'),
        impact: layerView
          ? bilingual('The model structure and the full 128-rank routing surface share the same active MoE layer.', '模型结构与完整 128-Rank 路由视图共享同一个活动 MoE Layer。')
          : bilingual('The whole-network view follows one token across Dense L0-L1 and MoE L2-L45.', '整网视图跟踪一个 Token 经过 Dense L0-L1 与 MoE L2-L45。'),
        facts: [
          { label: bilingual('Layer', 'Layer'), value: `L${state.selectedLayer}`, source: bilingual('Linked model context', '联动模型上下文') },
          { label: bilingual('Token', 'Token'), value: `T${String(state.selectedToken).padStart(3, '0')}`, source: bilingual('MOCK · routing trace', 'MOCK · 路由 Trace') },
          { label: bilingual('Routed experts', '路由专家'), value: '256 · Top-8', source: bilingual('MODEL · openPangu config', '模型 · openPangu 配置') },
          { label: bilingual('Rank placement', 'Rank 部署'), value: '128 · 2 experts/rank', source: bilingual('DEMO · visualization mapping', 'DEMO · 可视化映射') },
        ],
        viewExplanation: layerView ? [
          {
            label: bilingual('Scenario', '适用场景'),
            copy: bilingual(`Fix Layer ${state.selectedLayer} and compare one batch of 128 tokens across 256 routed experts placed on 128 ranks. Use it to find overloaded, idle, or imbalanced experts.`, `固定 Layer ${state.selectedLayer}，比较一批 128 个 Token 在 128 Rank、256 个路由专家上的分发，用于发现专家过载、空闲或负载不均。`),
          },
          {
            label: bilingual('How to read', '阅读方式'),
            copy: bilingual('Each Rank owns two experts. Token dots show queue occupancy; highlighted routes trace the selected token from Router to its Top-8 experts.', '每个 Rank 承载两个专家；圆点表示队列中的 Token，占用高亮线路表示当前 Token 从 Router 分发到 Top-8 专家的路径。'),
          },
          {
            label: bilingual('Current anomaly', '当前异常点'),
            copy: bilingual('R42 · E64/E65 receives 2.1× the mean load; E64 reaches 842/896 capacity and contributes 4.8 ms of exposed wait.', 'R42 · E64/E65 接收均值 2.1× 的负载；E64 容量达到 842/896，并产生 4.8 ms 暴露等待。'),
            anomaly: true,
          },
        ] : [
          {
            label: bilingual('Scenario', '适用场景'),
            copy: bilingual(`Fix Token T${String(state.selectedToken).padStart(3, '0')} and follow it through Dense L0–L1 and MoE L2–L45. Use it to explain why this token reaches particular experts.`, `固定 Token T${String(state.selectedToken).padStart(3, '0')}，跟踪它经过 Dense L0–L1 与 MoE L2–L45，用于解释该 Token 为何在不同层命中特定专家。`),
          },
          {
            label: bilingual('How to read', '阅读方式'),
            copy: bilingual('Each MoE layer shows all 256 experts in gray, its Top-8 experts in black, and their weighted-sum hidden state as a blue node. The blue path connects weighted results between layers.', '每个 MoE Layer 以灰色展示全部 256 个专家，黑色表示 Top-8，蓝点表示 8 个专家输出的加权和；蓝线连接相邻层的加权结果。'),
          },
          {
            label: bilingual('Current anomaly', '当前异常点'),
            copy: bilingual(`At Layer ${state.selectedLayer}, T${String(state.selectedToken).padStart(3, '0')} contributes to the R42 · E64/E65 hotspot. A single-token journey explains the route but must be paired with the layer-load view to prove overload.`, `在 Layer ${state.selectedLayer}，T${String(state.selectedToken).padStart(3, '0')} 命中 R42 · E64/E65 热点；单 Token 路径只能解释路由，需要结合单层负载视图才能证明过载。`),
            anomaly: true,
          },
        ],
        alternatives: bilingual('The 128-rank placement is a demo mapping, not a claim about the official training topology.', '128-Rank 部署是 Demo 映射，不代表官方训练拓扑。'),
        action: bilingual('Select a Rank, routed expert, or weighted-sum node to inspect linked evidence.', '选择 Rank、路由专家或加权和节点，查看联动证据。'),
      };
    }
    if (!selection || selection.type === 'diagnosis') {
      return {
        title: diagnosis.title,
        kicker: bilingual(`${diagnosis.severity} · derived diagnosis`, `${diagnosis.severity} · 推导诊断`),
        impact: diagnosis.impact,
        facts: [
          { label: bilingual('Expert max / mean', '专家最大值 / 均值'), value: '2.1×', source: bilingual('DERIVED · Router token counts', '推导 · Router Token 计数') },
          { label: bilingual('Exposed wait', '暴露等待'), value: '4.8 ms', source: bilingual('DERIVED · Dispatch event + wait chain', '推导 · Dispatch 事件 + 等待链') },
          { label: bilingual('E64 capacity', 'E64 容量'), value: '842 / 896', source: bilingual('MOCK · Runtime capacity contract', 'MOCK · 运行时容量约束') },
          { label: t('evidence'), value: '3 / 3', source: bilingual('Router + Dispatch + Wait complete', 'Router + Dispatch + Wait 证据完整') },
        ],
        alternatives: bilingual('Input-batch locality, expert co-placement, dispatch duplication, or device-side variance could still amplify the observed skew.', '输入批次局部性、专家共置、分发重复或设备侧波动仍可能放大当前偏斜。'),
        action: diagnosis.next,
      };
    }

    if (selection.type === 'rank-deck-overview') {
      return {
        title: bilingual('Pangu global Rank scope', 'Pangu 全局 Rank 范围'),
        kicker: bilingual('MODEL · 128-rank global scope', '模型 · 128-Rank 全局范围'),
        impact: bilingual('Sixteen server tiles preserve the 128-rank context while Server 05 expands into the active eight-card EP group.', '16 个服务器节点保留 128-Rank 上下文，同时将 Server 05 展开为活动的 8 卡 EP 通信组。'),
        facts: [
          { label: bilingual('Model', '模型'), value: 'openPangu-2.0-Flash', source: bilingual('MODEL · source-checked Pattern preset', '模型 · 已核源码的 Pattern Preset') },
          { label: bilingual('Global scope', '全局范围'), value: '128 ranks', source: bilingual('DEMO · profiling scope', 'DEMO · Profiling 范围') },
          { label: bilingual('Physical layout', '物理布局'), value: '16 servers × 8 cards', source: bilingual('DEMO · explicit placement', 'DEMO · 显式部署') },
          { label: bilingual('Model layers', '模型层数'), value: '46', source: bilingual('MODEL · openPangu preset', '模型 · openPangu Preset') },
        ],
        alternatives: bilingual('The 128-rank placement is an explicit demo topology and is not presented as an official openPangu training deployment.', '128-Rank 放置是显式的 Demo 拓扑，不宣称为 openPangu 官方训练部署。'),
        action: bilingual('Inspect the Server 05 EP8 floor and correlate its ribbons with the profile timeline.', '检查 Server 05 的 EP8 Floor，并将 Ribbon 与 Profiling 泳道关联。'),
      };
    }

    if (selection.type === 'server-rank') {
      const { item } = selection;
      return {
        title: `Server ${String(item.server).padStart(2, '0')} · Rank ${item.rank}`,
        kicker: bilingual('PHYSICAL CARD · EP8 MEMBER', '物理卡 · EP8 成员'),
        impact: bilingual('Local routes stay on this card and do not consume inter-rank network bandwidth.', '本地路由停留在当前卡上，不占用 Rank 间网络带宽。'),
        facts: [
          { label: bilingual('Server', '服务器'), value: `S${String(item.server).padStart(2, '0')}`, source: bilingual('DEMO · physical placement', 'DEMO · 物理部署') },
          { label: bilingual('Rank', 'Rank'), value: `R${item.rank}`, source: bilingual('DEMO · one process per card', 'DEMO · 每卡一个进程') },
          { label: bilingual('Local tokens', '本地 Token'), value: Number(item.localTokens || 0).toLocaleString(), source: bilingual('Current profile window', '当前 Profiling 时间窗') },
        ],
        alternatives: bilingual('A high local count is routing load, not network traffic; compare it with compute duration before diagnosing congestion.', '较高的本地计数属于路由负载而非网络流量，诊断拥塞前需要结合计算时长。'),
        action: bilingual('Select a timeline transfer event to isolate the corresponding rank-to-rank ribbon.', '选择泳道中的传输事件，定位对应的 Rank 间 Ribbon。'),
      };
    }

    if (selection.type === 'rank-overview') {
      const { manifest } = selection;
      const coordinate = manifest.coordinate;
      const segment = manifest.layerSegments[0];
      return {
        title: `Rank ${manifest.rank}`,
        kicker: bilingual('Global deck · selected rank', '全局 Deck · 已选 Rank'),
        impact: bilingual('This selection explains placement only. It does not yet create a valid communication-analysis scope.', '当前选择仅说明放置位置，尚未形成有效的通信分析范围。'),
        facts: [
          { label: bilingual('Coordinate', '坐标'), value: `PP${coordinate.pp} · EP${coordinate.ep} · replica${coordinate.edp}`, source: bilingual('DERIVED · demo topology', '推导 · Demo 拓扑') },
          { label: bilingual('Layer payload', 'Layer 载荷'), value: `L${segment.start}–L${segment.end}`, source: bilingual('MODEL · Pangu stage partition', '模型 · Pangu Stage 切分') },
          { label: bilingual('Expert shard', '专家分片'), value: `E${manifest.expertOwnership.start}–E${manifest.expertOwnership.end}`, source: bilingual('DERIVED · EP8', '推导 · EP8') },
        ],
        alternatives: bilingual('A single rank cannot represent All-to-All conservation or identify peer imbalance.', '单个 Rank 无法表达 All-to-All 守恒关系，也无法判断 Peer 间不均衡。'),
        action: bilingual('Choose the supported communication group instead of drilling into this rank alone.', '请选择受支持的通信组，而不是只下钻当前 Rank。'),
      };
    }

    if (selection.type === 'communication-group') {
      return {
        title: bilingual('EP communication group selected', '已选择 EP 通信组'),
        kicker: bilingual('LOCAL SCOPE · SERVER 05 · 8 RANKS', '局部范围 · SERVER 05 · 8 个 Rank'),
        impact: bilingual('This is the active peer set for comparing Dispatch and Combine traffic.', '当前通信组是比较分发与回传流量的活动 Peer 集合。'),
        facts: [
          { label: bilingual('Collective', '集合通信'), value: 'All-to-All / All-to-All-v', source: bilingual('MODEL · MoE Dispatch / Combine', '模型 · MoE 分发 / 回传') },
          { label: bilingual('Scope', '范围'), value: ACTIVE_SCOPE, source: bilingual('DEMO · explicit physical placement', 'DEMO · 显式物理部署') },
          { label: bilingual('Window', '窗口'), value: `${state.window.start}–${state.window.end} ms`, source: bilingual('Current profile filter', '当前 Profiling 过滤条件') },
        ],
        alternatives: bilingual('Selection alone does not indicate an anomaly; profiler traffic is required.', '仅选择通信组不代表存在异常，还需要接入 Profiling 流量。'),
        action: bilingual('Traffic is already linked beside the Rank Deck; open Model & Routing to inspect expert-level evidence.', '流量已经与 Rank Deck 并列联动；可打开“模型与路由”查看专家级证据。'),
      };
    }

    if (selection.type === 'moe-routing') {
      const item = selection.selection || {};
      const layer = Number.isInteger(item.layer) ? item.layer : state.selectedLayer;
      const tokenLabel = `T${String(Number.isInteger(item.tokenId) ? item.tokenId : state.selectedToken).padStart(3, '0')}`;
      if (item.type === 'layer-anomaly') {
        return {
          title: bilingual('Layer 27 routing overload', 'Layer 27 路由过载'),
          kicker: bilingual('ANOMALY · ROUTING SKEW', '异常 · 路由偏斜'),
          impact: bilingual('T037 contributes to the R42 · E64/E65 hotspot. The layer receives 2.1× the mean load and exposes 4.8 ms of critical-path wait.', 'T037 命中 R42 · E64/E65 热点；该位置接收均值 2.1× 的负载，并产生 4.8 ms 关键路径暴露等待。'),
          facts: [
            { label: bilingual('Hot placement', '热点位置'), value: 'R42 · E64/E65', source: bilingual('Layer 27 routing trace', 'Layer 27 路由 Trace') },
            { label: bilingual('Max / mean load', '最大值 / 均值负载'), value: '2.1×', source: bilingual('Router token counts', 'Router Token 计数') },
            { label: bilingual('Expert capacity', '专家容量'), value: '842 / 896', source: bilingual('Layer queue occupancy', '单层队列占用') },
            { label: bilingual('Exposed wait', '暴露等待'), value: '4.8 ms', source: bilingual('Critical-path timeline', '关键路径时间线') },
          ],
          alternatives: bilingual('The token journey explains where T037 was routed; overload is confirmed by the layer queue and timeline evidence.', 'Token 整网路径解释 T037 被路由到哪里；过载由单层队列与时间线证据共同确认。'),
          action: bilingual('Switch to Layer routing at L27 and inspect R42 · E64/E65 queue occupancy.', '切换到 L27 单层路由，检查 R42 · E64/E65 的队列占用。'),
        };
      }
      const title = item.type === 'rank'
        ? `Rank ${item.rankId}`
        : item.type === 'expert' || item.type === 'route'
          ? `L${layer} · E${item.expertId}`
          : item.type === 'aggregate'
            ? `L${layer} · weighted Top-8 result`
            : item.type === 'shared-expert'
              ? `L${layer} · Shared Expert`
              : `${tokenLabel} routing context`;
      const placement = Number.isInteger(item.rankId) ? `R${item.rankId}` : bilingual('all active ranks', '全部活动 Rank');
      const weight = Number.isFinite(item.weight) ? item.weight.toFixed(4) : bilingual('n/a', '无');
      return {
        title,
        kicker: bilingual('MOE ROUTING · LINKED SELECTION', 'MOE 路由 · 联动选择'),
        impact: item.type === 'aggregate'
          ? bilingual('The blue node is the weighted sum of all eight routed expert outputs for this token at the layer.', '蓝色节点表示该 Token 在这一层的 8 个路由专家输出加权和。')
          : item.type === 'rank'
            ? bilingual('This rank remains in the 128-rank layer context and can be compared with the active four-rank traffic slice.', '该 Rank 保留在 128-Rank 单层上下文中，可与当前四个 Rank 的流量切片对照。')
            : bilingual('This selection connects model-layer routing semantics with the current profiling context.', '当前选择将模型层路由语义与 Profiling 上下文关联起来。'),
        facts: [
          { label: bilingual('Object', '对象'), value: item.type || 'routing', source: bilingual('MOCK · shared routing trace', 'MOCK · 共享路由 Trace') },
          { label: bilingual('Layer / Token', 'Layer / Token'), value: `L${layer} · ${tokenLabel}`, source: bilingual('Linked workspace state', '联动工作区状态') },
          { label: bilingual('Placement', '部署位置'), value: placement, source: bilingual('DEMO · 2 experts / rank', 'DEMO · 每 Rank 2 个专家') },
          { label: bilingual('Router weight', 'Router 权重'), value: weight, source: Number.isFinite(item.weight) ? bilingual('MOCK · Top-8 decision', 'MOCK · Top-8 决策') : bilingual('Not applicable', '不适用') },
        ],
        alternatives: bilingual('The routing trace is mock evidence; it demonstrates linkage and does not prove the profiler traffic was caused by this exact token.', '当前路由 Trace 是 Mock 证据，只用于演示联动，不能证明 Profiling 流量由这个具体 Token 导致。'),
        action: bilingual('Compare the selected expert placement with the active four-rank Traffic view.', '将所选专家部署位置与当前四 Rank 流量视图进行对照。'),
      };
    }

    if (selection.type === 'link') {
      const { link } = selection;
      const target = link.targetEngine ? link.targetEngine.toUpperCase() : `R${link.targetRank}`;
      const crossNode = [...link.transports].includes('roce');
      return {
        title: `R${link.sourceRank} → ${target}`,
        kicker: bilingual('Selected token route', '已选 Token 路由'),
        impact: bilingual(`${formatTokens(link.value)} from this source contribute to the hot destination during the selected window.`, `所选窗口内，来自该来源的 ${formatTokens(link.value)} 流向热点目的地。`),
        facts: [
          { label: t('tokenCount'), value: formatTokens(link.value), source: `MOCK · ${link.eventIds.length} ${bilingual('Dispatch events', '个 Dispatch 事件')}` },
          { label: bilingual('Transport', '传输'), value: crossNode ? 'RoCE' : [...link.transports].join(' / ').toUpperCase(), source: bilingual('DERIVED · Rank placement', '推导 · Rank 部署位置') },
          { label: bilingual('Mean event', '事件均值'), value: `${(link.duration / link.eventIds.length).toFixed(2)} ms`, source: bilingual('MOCK · Profiler events', 'MOCK · Profiling 事件') },
          { label: bilingual('Window', '窗口'), value: `${state.window.start}–${state.window.end} ms`, source: bilingual('Current profile filter', '当前 Profiling 过滤条件') },
        ],
        alternatives: bilingual('A large logical route is not automatically exposed time; overlap, expert compute, and collective scheduling remain possible confounders.', '逻辑路由流量大不等于暴露耗时；重叠、专家计算和集合通信调度仍可能是混杂因素。'),
        action: crossNode
          ? bilingual('Compare this route after moving the expert inside the local HCCS domain.', '将专家移入本地 HCCS 域后，对比该路由。')
          : bilingual('Test router balancing while preserving the current placement.', '保持当前部署位置，验证 Router 负载均衡。'),
      };
    }

    if (selection.type === 'node') {
      const node = selection.node;
      return {
        title: node.label || `Rank ${node.rank}`,
        kicker: bilingual('Selected engine attribution', '已选引擎归因'),
        impact: bilingual(`${formatTokens(node.value)} are attributed to ${node.label} on Rank ${node.rank} in the active window.`, `当前窗口内有 ${formatTokens(node.value)} 归因到 Rank ${node.rank} 的 ${node.label}。`),
        facts: [
          { label: bilingual('Attributed', '归因流量'), value: formatTokens(node.value), source: bilingual('MOCK · Runtime task correlation', 'MOCK · Runtime Task 关联') },
          { label: bilingual('Engine', '引擎'), value: node.label, source: bilingual('DERIVED · profiler task type', '推导 · Profiler Task 类型') },
          { label: bilingual('Placement', '部署位置'), value: `Node B · Rank ${node.rank}`, source: bilingual('MOCK · Parallel plan', 'MOCK · 并行方案') },
          { label: bilingual('Window', '窗口'), value: `${state.window.start}–${state.window.end} ms`, source: bilingual('Current profile filter', '当前 Profiling 过滤条件') },
        ],
        alternatives: bilingual('Engine attribution describes where profiled tasks ran; it does not by itself identify the hot expert or prove exposed latency.', '引擎归因描述 Profiling Task 在哪里执行；它本身不能确定热点专家，也不能证明暴露时延。'),
        action: bilingual('Correlate this endpoint with the selected Timeline task, then inspect expert capacity evidence separately.', '将该端点与所选 Timeline Task 关联，再单独检查专家容量证据。'),
      };
    }

    const event = selection.event;
    const row = selection.row;
    const isWait = event.kind === 'wait' || event.id === 'wait-r42';
    return {
      title: event.displayName || event.label || event.id,
      kicker: `${t('timeline')} · ${row.label}`,
      impact: isWait
        ? bilingual('This wait is exposed on the critical path and delays Combine.', '该等待暴露在关键路径上，并延迟回传阶段。')
        : bilingual('This event provides the temporal and hierarchical evidence behind the selected communication route.', '该事件提供了所选通信路由背后的时间与层级证据。'),
      facts: [
        { label: bilingual('Start', '开始'), value: `${event.start.toFixed(1)} ms`, source: bilingual('MOCK · Event timestamp', 'MOCK · 事件时间戳') },
        { label: bilingual('Duration', '时长'), value: `${(event.end - event.start).toFixed(1)} ms`, source: String(event.status || 'mock').toUpperCase() },
        { label: bilingual('Hierarchy', '层级'), value: row.label, source: bilingual('Timeline row contract', 'Timeline 行层级契约') },
        { label: bilingual('Transaction', '事务'), value: event.transactionId || bilingual('not attributed', '未归因'), source: event.transactionId ? bilingual('Stable correlation ID', '稳定关联 ID') : bilingual('Missing artifact field', 'Artifact 字段缺失') },
        { label: bilingual('Window', '窗口'), value: `${state.window.start}–${state.window.end} ms`, source: bilingual('Current profile filter', '当前 Profiling 过滤条件') },
      ],
      alternatives: bilingual('A temporal overlap is not sufficient to prove dependency; the wait-chain relation is required for an exposed-time conclusion.', '时间重叠不足以证明依赖关系；判断暴露耗时还需要等待链关系。'),
      action: isWait
        ? bilingual('Trace the earliest unmet dependency back to R42 E64/E65.', '将最早未满足的依赖追溯到 R42 E64/E65。')
        : bilingual('Correlate this event with the selected Sankey route and Rank wait chain.', '将该事件与所选 Sankey 路由和 Rank 等待链关联。'),
    };
  }

  function openEvidence(selection) {
    state.selection = selection || { type: 'diagnosis' };
    const content = evidenceFor(state.selection);
    ensureDockOpen();
    elements.evidenceTitle.textContent = content.title;
    elements.evidenceKicker.textContent = content.kicker;
    elements.evidenceKicker.classList.toggle('is-diagnosis', state.selection.type === 'diagnosis' || state.selection.selection?.type === 'layer-anomaly');
    elements.evidenceImpact.textContent = content.impact;
    elements.evidenceAlternatives.textContent = content.alternatives;
    elements.evidenceActionCopy.textContent = content.action;
    setViewExplanation(content.viewExplanation || []);
    elements.validationPlan.hidden = true;
    const rankContext = ['rank-deck-overview', 'rank-overview', 'communication-group'].includes(state.selection.type);
    const layerAnomaly = state.selection.type === 'moe-routing' && state.selection.selection?.type === 'layer-anomaly';
    elements.routingCompareAction.hidden = !layerAnomaly;
    elements.validationAction.hidden = rankContext || layerAnomaly;
    elements.layerViewAction.hidden = state.primaryView !== 'communication';
    setEvidenceFacts(content.facts);
    elements.timelineSelection.textContent = content.kicker;
    if (state.selection.type === 'communication-group') {
      elements.phaseScope.textContent = bilingual('Server 05 · 8-rank EP group', 'Server 05 · 8-Rank EP 通信组');
    } else if (state.selection.type === 'rank-deck-overview') {
      elements.phaseScope.textContent = bilingual('No communication group selected', '尚未选择通信组');
    }
    elements.live.textContent = `Evidence opened for ${content.title}`;
    renderTimeline();
  }

  let sankeyController = sankeyPattern.render(elements.sankeyHost, {
    data: phaseData(state.phase),
    onSelect(selection) {
      if (selection) {
        if (selection.type === 'link') {
          state.timelineController?.setActiveRows([state.phase]);
          state.timelineController?.focusRelatedEventIds(selection.link.eventIds || []);
          state.trafficController?.focusFloorTrafficEventIds(selection.link.eventIds || []);
        } else if (selection.type === 'node') {
          const rankRow = `${state.phase}-rank-${selection.node.rank}`;
          state.timelineController?.setActiveRows([state.phase, rankRow]);
          state.timelineController?.focusRelatedEventIds([]);
        }
        openEvidence(selection);
      } else {
        state.timelineController?.setActiveRows([state.phase]);
        state.timelineController?.focusRelatedEventIds([]);
        state.trafficController?.clearFloorTrafficFocus();
        if (!elements.evidenceDock.hidden) openEvidence({ type: 'diagnosis' });
        else state.selection = { type: 'diagnosis' };
      }
    },
    onExpand({ side, rank, expanded }) {
      elements.phaseScope.textContent = expanded
        ? `Server 05 / EP8 / ${side === 'target' ? 'Destination' : 'Source'} Rank ${rank}`
        : 'Server 05 / EP8 / R40–R47';
    },
  });

  function setPhase(phase) {
    if (!mock.phases[phase]) return;
    const changed = phase !== state.phase;
    state.phase = phase;
    state.selection = { type: 'diagnosis' };
    if (changed) sankeyController.setData(phaseData(phase));
    state.trafficController?.clearFloorTrafficFocus();
    syncTrafficOverlay();
    state.timelineController?.setActiveRows([phase]);
    state.timelineController?.focusRelatedEventIds([]);
    updateDiagnosis();
    if (!elements.evidenceDock.hidden) openEvidence(state.selection);
    renderTimeline();
    updateTrafficTools();
    elements.live.textContent = bilingual(`${currentDiagnosis(phase).label} phase selected`, `已选择${currentDiagnosis(phase).label}阶段`);
  }

  function timelinePalette() {
    const root = elements.sankeyHost.querySelector('.pto-traffic-sankey');
    const styles = root ? getComputedStyle(root) : getComputedStyle(document.documentElement);
    const fallback = ['#4369ef', '#04b98a', '#a855f7', '#f97316', '#eab308', '#38bdf8', '#ec4899', '#8b5cf6'];
    return fallback.map((color, slot) => styles.getPropertyValue(`--rank-${slot}`).trim() || color);
  }

  function timelineData() {
    const palette = timelinePalette();
    const styles = getComputedStyle(document.documentElement);
    const colorForSlot = (slot) => {
      if (Number.isInteger(slot)) return palette[slot] || palette[0];
      if (slot === 'wait') return styles.getPropertyValue('--danger').trim() || '#d65b75';
      return styles.getPropertyValue('--foreground-secondary').trim() || '#8b929e';
    };
    return {
      id: mock.timelineHierarchy.id,
      ariaLabel: state.language === 'zh' ? mock.timelineHierarchy.ariaLabelZh : mock.timelineHierarchy.ariaLabel,
      emptyLabel: bilingual('No communication events in this range', '当前范围内没有通信事件'),
      unit: mock.timelineHierarchy.unit,
      timeRange: { ...mock.timeRange },
      window: { ...state.window },
      defaultExpandedIds: [...mock.timelineHierarchy.defaultExpandedIds],
      rows: mock.timelineHierarchy.rows.map((row) => ({
        ...row,
        label: state.language === 'zh' ? (row.labelZh || row.label) : row.label,
        events: (row.events || []).map((event) => ({
          ...event,
          label: state.language === 'zh' ? (event.labelZh || event.label) : event.label,
          displayName: state.language === 'zh' ? (event.displayNameZh || event.displayName) : event.displayName,
          color: colorForSlot(event.colorSlot),
        })),
      })),
    };
  }

  function renderTimeline() {
    state.timelineController?.resize();
  }

  function initializeTimeline() {
    state.timelineController = hierarchicalTimelinePattern.render(elements.timelineRoot, {
      data: timelineData(),
      onSelect(selection) {
        if (selection) {
          const midpoint = (Number(selection.event.start) + Number(selection.event.end)) / 2;
          if (Number.isFinite(midpoint)) setPhase(phaseForTime(midpoint));
          const ids = selection.event.sourceEventIds?.length
            ? selection.event.sourceEventIds
            : [selection.event.id];
          sankeyController.focusEventIds(ids);
          state.trafficController?.focusFloorTrafficEventIds(ids);
          openEvidence(selection);
        } else {
          sankeyController.clearSelection(false);
          state.trafficController?.clearFloorTrafficFocus();
          if (!elements.evidenceDock.hidden) openEvidence({ type: 'diagnosis' });
          else state.selection = { type: 'diagnosis' };
        }
      },
      onWindowChange(nextWindow) {
        applyWindow(nextWindow.start, nextWindow.end, { fromTimeline: true });
      },
      onZoomChange({ zoom }) {
        elements.timelineZoomReadout.textContent = `${zoom}×`;
      },
      onExpand({ row, expanded }) {
        elements.timelineSelection.textContent = bilingual(
          `${row.label} ${expanded ? 'expanded' : 'collapsed'}`,
          `${row.label}已${expanded ? '展开' : '收起'}`,
        );
      },
    });
    state.timelineController.setActiveRows([state.phase]);
  }

  function applyWindow(nextStart, nextEnd, { fromTimeline = false } = {}) {
    const step = mock.timeRange.step;
    state.window.start = Math.max(mock.timeRange.min, Math.min(nextStart, nextEnd - step));
    state.window.end = Math.min(mock.timeRange.max, Math.max(nextEnd, state.window.start + step));
    state.window.start = Math.round(state.window.start / step) * step;
    state.window.end = Math.round(state.window.end / step) * step;
    setPhase(phaseForTime((state.window.start + state.window.end) / 2));
    sankeyController.setWindow(state.window.start, state.window.end);
    state.trafficController?.clearFloorTrafficFocus();
    syncTrafficOverlay();
    state.selection = { type: 'diagnosis' };
    if (!elements.evidenceDock.hidden) openEvidence(state.selection);
    if (!fromTimeline) state.timelineController?.setWindow(state.window.start, state.window.end, false);
    formatWindow();
    elements.live.textContent = bilingual(
      `Profile window ${state.window.start} to ${state.window.end} milliseconds`,
      `Profiling 窗口 ${state.window.start} 至 ${state.window.end} 毫秒`,
    );
  }

  function setTheme(theme) {
    const nextTheme = theme === 'light' ? 'light' : 'dark';
    document.documentElement.dataset.theme = nextTheme;
    const nextLabel = nextTheme === 'dark'
      ? bilingual('Switch to light theme', '切换为浅色模式')
      : bilingual('Switch to dark theme', '切换为深色模式');
    elements.themeToggle.setAttribute('aria-label', nextLabel);
    elements.themeToggle.title = nextLabel;
    elements.themeToggle.setAttribute('aria-pressed', String(nextTheme === 'light'));
    try { window.localStorage.setItem('pto-communication-observatory-theme', nextTheme); } catch (error) { /* no-op */ }
    requestAnimationFrame(() => {
      state.timelineController?.setData(timelineData());
      state.timelineController?.setActiveRows([state.phase]);
      state.deckController?.setTheme(nextTheme).refresh();
      state.moeRoutingController?.setTheme(nextTheme);
      state.moeLayerDetailController?.setTheme(nextTheme);
      renderTimeline();
    });
  }

  function setBottomOpen(open) {
    const shouldOpen = Boolean(open);
    if (shouldOpen === state.bottomOpen && elements.bottomDock.hidden === !shouldOpen) return;
    const gutter = elements.bottomDock.previousElementSibling?.matches?.('.pto-workbench-shell__split-gutter')
      ? elements.bottomDock.previousElementSibling
      : null;
    if (!shouldOpen) {
      state.primarySizing = {
        flex: elements.primaryRow.style.flex,
        flexBasis: elements.primaryRow.style.flexBasis,
        height: elements.primaryRow.style.height,
      };
      elements.primaryRow.style.flex = '1 1 100%';
      elements.primaryRow.style.flexBasis = '100%';
      elements.primaryRow.style.height = '100%';
    } else if (state.primarySizing) {
      elements.primaryRow.style.flex = state.primarySizing.flex;
      elements.primaryRow.style.flexBasis = state.primarySizing.flexBasis;
      elements.primaryRow.style.height = state.primarySizing.height;
      state.primarySizing = null;
    }
    state.bottomOpen = shouldOpen;
    elements.bottomDock.hidden = !shouldOpen;
    elements.bottomDock.setAttribute('aria-hidden', String(!shouldOpen));
    if (gutter) gutter.hidden = !shouldOpen;
    elements.bottomToggle.classList.toggle('is-selected', shouldOpen);
    elements.bottomToggle.setAttribute('aria-expanded', String(shouldOpen));
    elements.bottomToggle.setAttribute('aria-pressed', String(shouldOpen));
    elements.live.textContent = bilingual(
      `Profile timeline ${shouldOpen ? 'opened' : 'closed'}`,
      `Profiling 时间线已${shouldOpen ? '打开' : '关闭'}`,
    );
    requestAnimationFrame(() => {
      renderTimeline();
      window.dispatchEvent(new Event('resize'));
    });
  }

  function toggleEvidenceDock() {
    if (elements.evidenceDock.hidden) {
      openEvidence(state.primaryView === 'communication'
        ? (state.selectedCommunicationGroup
          ? { type: 'communication-group', group: state.selectedCommunicationGroup }
          : { type: 'rank-deck-overview' })
        : { type: 'diagnosis' });
      return;
    }
    sankeyController.clearSelection(false);
    closeEvidence();
  }

  function setContextOpen(open) {
    const shouldOpen = Boolean(open);
    elements.contextPopover.hidden = !shouldOpen;
    elements.contextToggle.classList.toggle('is-selected', shouldOpen);
    elements.contextToggle.setAttribute('aria-expanded', String(shouldOpen));
    elements.contextToggle.setAttribute('aria-pressed', String(shouldOpen));
  }

  function setLanguage(language) {
    state.language = language === 'zh' ? 'zh' : 'en';
    updateStaticCopy();
    updateDiagnosis();
    sankeyController.setData(phaseData(state.phase));
    state.moeRoutingController?.setLanguage(state.language);
    state.moeLayerDetailController?.setLanguage(state.language);
    state.timelineController?.setActiveRows([state.phase]);
    if (state.selection && !elements.evidenceDock.hidden) openEvidence(state.selection);
    renderTimeline();
    setTheme(document.documentElement.dataset.theme);
    elements.live.textContent = state.language === 'zh' ? '界面已切换为中文' : 'Interface switched to English';
  }

  elements.sankeyOpen.addEventListener('click', () => setSankeyOpen(true));
  elements.sankeyClose.addEventListener('click', () => setSankeyOpen(false));
  elements.workspaceNavigation.addEventListener('click', (event) => {
    const button = event.target.closest('[data-workspace]');
    if (button) setPrimaryView(button.dataset.workspace);
  });
  elements.routingViewControl.addEventListener('click', (event) => {
    const button = event.target.closest('[data-routing-view]');
    if (!button) return;
    setRoutingComparison(false);
    state.routingView = button.dataset.routingView === 'journey' ? 'journey' : 'layer';
    ensureMoeRouting().setView(state.routingView);
    updatePrimaryView();
  });
  elements.routingPrevious.addEventListener('click', () => ensureMoeRouting().step(-1));
  elements.routingNext.addEventListener('click', () => ensureMoeRouting().step(1));
  elements.routingZoomOut.addEventListener('click', () => ensureMoeRouting().zoomOut());
  elements.routingFit.addEventListener('click', () => ensureMoeRouting().fit());
  elements.routingZoomIn.addEventListener('click', () => ensureMoeRouting().zoomIn());
  elements.routingDetailZoomOut.addEventListener('click', () => ensureMoeLayerDetail().zoomOut());
  elements.routingDetailFit.addEventListener('click', () => ensureMoeLayerDetail().fit());
  elements.routingDetailZoomIn.addEventListener('click', () => ensureMoeLayerDetail().zoomIn());
  elements.routingCompareAction.addEventListener('click', () => {
    if (state.primaryView !== 'routing') setPrimaryView('routing');
    setRoutingComparison(true, 27);
  });
  elements.routingCompareClose.addEventListener('click', () => setRoutingComparison(false));
  elements.railEvidence.addEventListener('click', () => openEvidence(state.primaryView === 'communication'
    ? (state.selectedCommunicationGroup
      ? { type: 'communication-group', group: state.selectedCommunicationGroup }
      : { type: 'rank-deck-overview' })
    : { type: 'diagnosis' }));
  elements.layerViewAction.addEventListener('click', () => setPrimaryView('routing'));
  elements.timelineZoomOut.addEventListener('click', () => state.timelineController?.zoomOut());
  elements.timelineZoomIn.addEventListener('click', () => state.timelineController?.zoomIn());
  elements.timelineFit.addEventListener('click', () => {
    state.timelineController?.fit();
    elements.timelineZoomReadout.textContent = '1×';
  });
  elements.railTimeline.addEventListener('click', () => setBottomOpen(!state.bottomOpen));
  elements.bottomToggle.addEventListener('click', () => setBottomOpen(!state.bottomOpen));
  elements.evidencePanelToggle.addEventListener('click', toggleEvidenceDock);
  elements.contextToggle.addEventListener('click', () => setContextOpen(elements.contextPopover.hidden));
  elements.contextClose.addEventListener('click', () => setContextOpen(false));
  elements.themeToggle.addEventListener('click', () => {
    setTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light');
  });
  elements.languageToggle.addEventListener('click', () => setLanguage(state.language === 'en' ? 'zh' : 'en'));
  elements.evidenceClose.addEventListener('click', () => {
    sankeyController.clearSelection(false);
    closeEvidence();
  });
  document.addEventListener('pointerdown', (event) => {
    if (elements.contextPopover.hidden) return;
    if (elements.contextPopover.contains(event.target) || elements.contextToggle.contains(event.target)) return;
    setContextOpen(false);
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      setContextOpen(false);
      setSankeyOpen(false);
    }
  });
  elements.validationAction.addEventListener('click', () => {
    elements.validationPlan.hidden = false;
    elements.validationPlan.textContent = bilingual(
      'Candidate: rebalance E64/E65 placement or router load. Pass when Max/Mean < 1.25×, buffer headroom > 20%, exposed wait < 1.5 ms, Step Time improves, and correctness / hang gates remain green.',
      '候选方案：重新平衡 E64/E65 的部署位置或 Router 负载。通过条件：最大值/均值 < 1.25×、缓冲余量 > 20%、暴露等待 < 1.5 ms、Step Time 改善，且正确性与 Hang 门禁保持通过。',
    );
    elements.live.textContent = bilingual('Validation plan created', '验证计划已创建');
  });
  window.addEventListener('resize', () => requestAnimationFrame(() => {
    renderTimeline();
    state.deckController?.refresh();
    state.moeRoutingController?.refresh();
    state.moeLayerDetailController?.refresh();
  }));

  initializeTimeline();
  updateStaticCopy();
  ensureModelDeck();
  if (state.primaryView === 'routing') {
    ensureMoeRouting();
  }
  updateDiagnosis();
  formatWindow();
  const requestedTheme = new URLSearchParams(window.location.search).get('theme');
  let initialTheme = requestedTheme === 'light' || requestedTheme === 'dark' ? requestedTheme : 'dark';
  if (!requestedTheme) {
    try { initialTheme = window.localStorage.getItem('pto-communication-observatory-theme') || 'dark'; } catch (error) { /* no-op */ }
  }
  setTheme(initialTheme);
  if (state.primaryView === 'communication') {
    updatePrimaryView();
    selectDemoCommunicationGroup();
  } else {
    updatePrimaryView();
  }
})();
