(() => {
  'use strict';

  const FIXTURES = [
    { id: 'sample.conv_bias_relu', path: 'data/fixtures/conv_bias_relu.trace.json?v=20260803-allocate-memory-map-v1' },
  ];

  const TENSOR_TONES = {
    default: { fill: 'rgba(116, 128, 142, 0.24)', stroke: 'rgba(220, 230, 240, 0.16)' },
    input: { fill: 'rgba(77, 151, 255, 0.72)', stroke: 'rgba(184, 218, 255, 0.88)' },
    output: { fill: 'rgba(41, 199, 166, 0.72)', stroke: 'rgba(188, 255, 239, 0.9)' },
    compute: { fill: 'rgba(255, 207, 89, 0.74)', stroke: 'rgba(255, 237, 178, 0.9)' },
    reduction: { fill: 'rgba(255, 154, 84, 0.72)', stroke: 'rgba(255, 214, 184, 0.88)' },
    fusion: { fill: 'rgba(184, 146, 255, 0.72)', stroke: 'rgba(229, 216, 255, 0.9)' },
    avoided: { fill: 'rgba(164, 176, 189, 0.20)', stroke: 'rgba(164, 176, 189, 0.42)' },
  };
  const ARCH_ZOOM_LEVELS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2];

  const CPP_KEYWORDS = new Set([
    'alignas', 'auto', 'break', 'case', 'class', 'const', 'constexpr', 'continue', 'default', 'defined', 'do',
    'else', 'false', 'for', 'if', 'inline', 'int', 'namespace', 'new', 'nullptr', 'operator', 'private', 'public',
    'return', 'sizeof', 'static', 'struct', 'switch', 'template', 'this', 'true', 'typedef', 'typename', 'using',
    'void', 'volatile', 'while', '__aicore__', '__global__', '__cube__', '__vector__', '__mix__', '__gm__', '__ubuf__',
    'ASCEND_IS_AIC', 'ASCEND_IS_AIV',
  ]);

  const CPP_TYPES = new Set([
    'bool', 'char', 'double', 'float', 'half', 'int32_t', 'int64_t', 'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t',
    'size_t', 'GM_ADDR', 'AscendC', 'GlobalTensor', 'LocalTensor', 'TPipe', 'TQue', 'TBuf', 'TPosition',
    'DataCopyParams', 'DataCopyPadParams', 'Nd2NzParams', 'LoadData2DParams', 'LoadData2DParamsV2', 'MmadParams',
    'FixpipeParamsV220', 'QuantMode_t', 'HardEvent', 'PIPE_FIX', 'PIPE_V', 'PIPE_M', 'PIPE_MTE1', 'PIPE_MTE2',
    'PIPE_MTE3', 'PIPE_ALL',
  ]);

  const TEXT_ZH = {
    'Host prepares input, launches the vector kernel, and collects output.': 'Host 准备输入、启动 vector kernel，并回收输出。',
    'Derive per-block and per-tile lengths, then initialize GM views and queue buffers.': '计算每个 block 和每个 tile 的长度，然后初始化 GM 视图和队列 buffer。',
    'Allocate local x/y tensors and copy one tile from GM to VECIN queues.': '分配 x/y 的 LocalTensor，并把一个 tile 从 GM 拷入 VECIN 队列。',
    'Deque x/y local tensors, add them, enqueue z, and free input buffers.': '从队列取出 x/y 本地 tensor，执行 Add，写入 z 队列并释放输入 buffer。',
    'Deque z local tensor and copy the tile back to GM.': '从队列取出 zLocal，并把当前 tile 拷回 GM。',
    'Host copies x/y to device, launches 8 vector blocks, waits, and copies z back.': 'Host 将 x/y 拷到 device，启动 8 个 vector block，等待完成后拷回 z。',
    'blockLength=2048; blockIdx 0 owns GM[0:2048]. tileLength=128.': 'blockLength=2048；blockIdx 0 负责 GM[0:2048]；tileLength=128。',
    'Copy x/y GM[0:128] into VECIN queue slots.': '把 x/y 的 GM[0:128] 拷入 VECIN 队列槽位。',
    'Deque x/y, allocate zLocal, run Add over 128 fp32 values.': '取出 x/y，分配 zLocal，对 128 个 fp32 元素执行 Add。',
    'Copy zLocal back to zGm[0:128].': '把 zLocal 拷回 zGm[0:128]。',
    'For blockIdx=3, progress=2 starts at 3*2048 + 2*128 = 6400.': 'blockIdx=3 且 progress=2 时，起始偏移为 3*2048 + 2*128 = 6400。',
    'blockIdx=7, progress=15 writes the final tile GM:z[16256:16384].': 'blockIdx=7 且 progress=15 写回最后一个 tile：GM:z[16256:16384]。',

    'Map one Cube block to one singleCoreM x singleCoreN output partition.': '把一个 Cube block 映射到一个 singleCoreM x singleCoreN 的输出分区。',
    'Copy a baseM x baseK tile from A GM to A1.': '把 A 的 baseM x baseK tile 从 GM 拷到 A1。',
    'Copy a baseK x baseN tile from B GM to B1.': '把 B 的 baseK x baseN tile 从 GM 拷到 B1。',
    'Move A1/B1 tiles into L0A/L0B. 2201 and 3510 use different params.': '把 A1/B1 tile 搬到 L0A/L0B；2201 和 3510 使用不同参数。',
    'Accumulate A2 x B2 into CO1.': '将 A2 x B2 的结果累加到 CO1。',
    'Write CO1 to GM C with Nz->ND and fp32->half conversion.': '把 CO1 写回 GM C，并执行 Nz->ND 与 fp32->half 转换。',
    'mIterIdx=0, nIterIdx=0. GM offsets A=0, B=0, C=0.': 'mIterIdx=0，nIterIdx=0；GM 偏移 A=0、B=0、C=0。',
    'mIterIdx=0, nIterIdx=1. GM C offset=512.': 'mIterIdx=0，nIterIdx=1；GM C 偏移为 512。',
    'Copy A[M 0:128, K 0:64] from GM into A1.': '把 A[M 0:128, K 0:64] 从 GM 拷入 A1。',
    'Copy B[K 0:64, N 0:256] from GM into B1.': '把 B[K 0:64, N 0:256] 从 GM 拷入 B1。',
    'A1/B1 are moved to A2/B2; B is prepared with transpose semantics for Mmad.': 'A1/B1 被搬到 A2/B2；B 会按 Mmad 需要的转置语义准备。',
    'kIndex=0 sets cmatrixInitVal=true, initializing CO1 with first partial result.': 'kIndex=0 时 cmatrixInitVal=true，用第一段部分结果初始化 CO1。',
    'kIndex=7 adds the last baseK slice into CO1.': 'kIndex=7 将最后一段 baseK slice 累加进 CO1。',
    'CO1 is written to GM C[M 0:128, N 0:256] with conversion to half ND layout.': 'CO1 被写回 GM C[M 0:128, N 0:256]，并转换成 half ND layout。',

    'Launches a fused AIC/AIV kernel with one Cube producer for two Vector consumers.': '启动一个融合 AIC/AIV kernel：1 个 Cube 生产者对应 2 个 Vector 消费者。',
    'AIC computes each baseM x baseN C tile and writes it to GM.': 'AIC 计算每个 baseM x baseN 的 C tile，并写到 GM。',
    'AIC notifies the paired AIV blocks that the Matmul result is ready.': 'AIC 通知成对的 AIV block：Matmul 结果已经 ready。',
    'Each AIV block reads half of the AIC result tile, applies LeakyRelu, and writes it back.': '每个 AIV block 读取 AIC 结果 tile 的一半，执行 LeakyRelu 后写回。',
    'The kernel creates a Cube:Vector execution relationship of 1:2.': '该 kernel 建立 1:2 的 Cube:Vector 执行关系。',
    'AIC block 0 starts the Matmul pipeline for C[M 0:256, N 0:512].': 'AIC block 0 为 C[M 0:256, N 0:512] 启动 Matmul 流水。',
    'AIC0 accumulates through K and writes the Matmul output tile to GM C.': 'AIC0 沿 K 维累加，并把 Matmul 输出 tile 写到 GM C。',
    'CrossCoreSetFlag releases AIV block 0 and AIV block 1.': 'CrossCoreSetFlag 放行 AIV block 0 和 AIV block 1。',
    'AIV0 cannot read GM C until the Cube producer has set the flag.': 'Cube 生产者置 flag 之前，AIV0 不能读取 GM C。',
    'AIV0 reads the upper baseM/2 x baseN half tile, applies LeakyRelu, and writes it back.': 'AIV0 读取上半个 baseM/2 x baseN tile，执行 LeakyRelu 后写回。',
    'AIV1 uses GetBlockIdx()%2=1, so its GM offset jumps by baseM/2*N.': 'AIV1 使用 GetBlockIdx()%2=1，因此 GM 偏移会跳过 baseM/2*N。',

    'Host prepares and launches': 'Host 准备并启动',
    'blockIdx 0 maps GM partition': 'blockIdx 0 映射 GM 分区',
    'progress 0 CopyIn': 'progress 0 执行 CopyIn',
    'progress 0 Compute': 'progress 0 执行 Compute',
    'progress 0 CopyOut': 'progress 0 执行 CopyOut',
    'block 3 progress 2 CopyIn': 'block 3 / progress 2 执行 CopyIn',
    'last block last CopyOut': '最后一个 block 写回最后一个 tile',
    'blockIdx 0 selects top-left C partition': 'blockIdx 0 选择左上 C 分区',
    'blockIdx 2 selects top-right C partition': 'blockIdx 2 选择右上 C 分区',
    'kIndex 0 CopyIn A': 'kIndex 0 拷入 A',
    'kIndex 0 CopyIn B': 'kIndex 0 拷入 B',
    'LoadData to L0': 'LoadData 搬入 L0',
    'Mmad initializes CO1': 'Mmad 初始化 CO1',
    'Mmad final K accumulation': 'Mmad 完成最后一段 K 累加',
    'Fixpipe writes C tile': 'Fixpipe 写回 C tile',
    '__mix__(1,2) launch': '__mix__(1,2) 启动',
    'AIC0 copies A/B tiles': 'AIC0 拷入 A/B tile',
    'AIC0 Mmad + Fixpipe': 'AIC0 执行 Mmad + Fixpipe',
    'AIC0 signals AIV pair': 'AIC0 通知 AIV pair',
    'AIV0 waits for AIC0': 'AIV0 等待 AIC0',
    'AIV0 activates upper half': 'AIV0 激活上半 tile',
    'AIV1 activates lower half': 'AIV1 激活下半 tile',
  };

  const state = {
    traces: [],
    sampleId: null,
    sourceFileId: null,
    stepIndex: 0,
    evidence: false,
    playing: false,
    executionView: 'instructions',
    instructionLoopExpanded: false,
    instructionIterationFocus: null,
    instructionIterationRange: null,
    instructionOperationFocus: null,
    allocatedTensorId: null,
    timer: null,
    playback: null,
    webglAvailable: null,
    infoOpen: false,
    tensorTabStepId: null,
    tensorTabKey: null,
    convCoreIndex: 1,
    activeTensorRenderer: 'legacy',
    overviewControllers: {
      feature: null,
      weight: null,
      bias: null,
      output: null,
    },
    hostTilingControllers: {
      source: null,
      cube: null,
    },
    hostLaunchControllers: {
      a: null,
      b: null,
      c: null,
    },
    copyInputControllers: {
      source: null,
      destination: null,
      sourceKind: null,
      destinationKind: null,
    },
    biasC1C2Controllers: {
      source: null,
      destination: null,
    },
    fixpipeOutputControllers: {
      accum: null,
      output: null,
    },
    tensorVolumeController: null,
    tensorMatrixController: null,
   mmadMatrixControllers: {
      a2: null,
      b2: null,
      addend: null,
      co1: null,
    },
    fmapA1VolumeController: null,
    a2LogicalMatrixController: null,
    tensorView: {
      scale: 1,
      panX: 0,
      panY: 0,
      dragging: false,
      startX: 0,
      startY: 0,
      startPanX: 0,
      startPanY: 0,
      moved: false,
      raf: 0,
    },
    architecture: {
      mounted: false,
      frameReady: false,
      pendingFocus: null,
      detailsVisible: true,
      zoomIndex: 2,
      frameWidth: 1900,
      frameHeight: 2400,
      fitOnReady: true,
      panX: 0,
      panY: 0,
      dragging: false,
      startX: 0,
      startY: 0,
      startPanX: 0,
      startPanY: 0,
      moved: false,
      panCaptureTarget: null,
      framePanDocument: null,
    },
    resizeObserver: null,
    resizeRaf: 0,
    playbackIds: {
      shell: 'avz-floating-shell',
      toggle: 'avz-floating-toggle',
      collapsedButton: 'avz-floating-collapsed-btn',
      collapsedIcon: 'avz-floating-collapsed-icon',
      controls: 'avz-controls-row',
      stepBack: 'avz-step-back-btn',
      play: 'avz-play-btn',
      stepForward: 'avz-step-fwd-btn',
      replay: 'avz-replay-btn',
      scrubber: 'avz-scrubber',
      scrubberLabel: 'avz-scrubber-label',
      scrubberOpname: 'avz-scrubber-opname',
      scrubberHover: 'avz-scrubber-hover',
    },
  };

  const byId = (id) => document.getElementById(id);

  const els = {
    operatorMeta: byId('operatorMeta'),
    archReadout: byId('archReadout'),
    evidenceToggle: byId('evidenceToggle'),
    sampleList: byId('sampleList'),
    sourceMeta: byId('sourceMeta'),
    sourceLines: byId('sourceLines'),
    visualTitle: byId('visualTitle'),
    stepMeta: byId('stepMeta'),
    prevStep: byId('prevStep'),
    nextStep: byId('nextStep'),
    convCoreContext: byId('convCoreContext'),
    convCoreOptions: byId('convCoreOptions'),
    tensorTabs: byId('tensorTabs'),
    tensorStage: byId('tensorStage'),
    convTensorOverview: byId('convTensorOverview'),
    memoryAllocationView: byId('memoryAllocationView'),
    featureOverviewCanvas: byId('featureOverviewCanvas'),
    weightOverviewCanvas: byId('weightOverviewCanvas'),
    biasOverviewCanvas: byId('biasOverviewCanvas'),
    outputOverviewCanvas: byId('outputOverviewCanvas'),
    hostTilingView: byId('hostTilingView'),
    hostTilingEquation: byId('hostTilingEquation'),
    hostTilingEvidence: byId('hostTilingEvidence'),
    hostTilingTileCount: byId('hostTilingTileCount'),
    hostTilingSourceTitle: byId('hostTilingSourceTitle'),
    hostTilingSourceMeta: byId('hostTilingSourceMeta'),
    hostTilingSourceCanvas: byId('hostTilingSourceCanvas'),
    hostTilingWeightCount: byId('hostTilingWeightCount'),
    hostTilingWeightCountValue: byId('hostTilingWeightCountValue'),
    hostTilingTransform: byId('hostTilingTransform'),
    hostTilingCubeTitle: byId('hostTilingCubeTitle'),
    hostTilingCubeMeta: byId('hostTilingCubeMeta'),
    hostTilingCubeCanvas: byId('hostTilingCubeCanvas'),
    hostTilingFormula: byId('hostTilingFormula'),
    hostLaunchView: byId('hostLaunchView'),
    hostLaunchEquation: byId('hostLaunchEquation'),
    hostLaunchEvidence: byId('hostLaunchEvidence'),
    hostLaunchBlockDim: byId('hostLaunchBlockDim'),
    hostLaunchATitle: byId('hostLaunchATitle'),
    hostLaunchAMeta: byId('hostLaunchAMeta'),
    hostLaunchACanvas: byId('hostLaunchACanvas'),
    hostLaunchBTitle: byId('hostLaunchBTitle'),
    hostLaunchBMeta: byId('hostLaunchBMeta'),
    hostLaunchBCanvas: byId('hostLaunchBCanvas'),
    hostLaunchCTitle: byId('hostLaunchCTitle'),
    hostLaunchCMeta: byId('hostLaunchCMeta'),
    hostLaunchCCanvas: byId('hostLaunchCCanvas'),
    hostLaunchMapping: byId('hostLaunchMapping'),
    hostLaunchReduction: byId('hostLaunchReduction'),
    tensorPatternView: byId('tensorPatternView'),
    tensorVolumeCanvas: byId('tensorVolumeCanvas'),
    tensorMatrixHost: byId('tensorMatrixHost'),
    tensorMatrixCanvas: byId('tensorMatrixCanvas'),
    copyInputPatternView: byId('copyInputPatternView'),
    copyInputSummary: byId('copyInputSummary'),
    copyInputContext: byId('copyInputContext'),
    copyInputSourceTitle: byId('copyInputSourceTitle'),
    copyInputSourceShape: byId('copyInputSourceShape'),
    copyInputSourceMeta: byId('copyInputSourceMeta'),
    copyInputSourceCanvas: byId('copyInputSourceCanvas'),
    copyInputEngine: byId('copyInputEngine'),
    copyInputTransformation: byId('copyInputTransformation'),
    copyInputDestinationTitle: byId('copyInputDestinationTitle'),
    copyInputDestinationShape: byId('copyInputDestinationShape'),
    copyInputDestinationMeta: byId('copyInputDestinationMeta'),
    copyInputDestinationCanvas: byId('copyInputDestinationCanvas'),
    copyInputLensMount: byId('copyInputLensMount'),
    biasC1C2View: byId('biasC1C2View'),
    biasC1C2Summary: byId('biasC1C2Summary'),
    biasC1C2Context: byId('biasC1C2Context'),
    biasC1C2Engine: byId('biasC1C2Engine'),
    biasC1Title: byId('biasC1Title'),
    biasC1Shape: byId('biasC1Shape'),
    biasC1Meta: byId('biasC1Meta'),
    biasC1Canvas: byId('biasC1Canvas'),
    biasC2Title: byId('biasC2Title'),
    biasC2Shape: byId('biasC2Shape'),
    biasC2Meta: byId('biasC2Meta'),
    biasC2Canvas: byId('biasC2Canvas'),
    fixpipeOutputView: byId('fixpipeOutputView'),
    fixpipeOutputSummary: byId('fixpipeOutputSummary'),
    fixpipeOutputContext: byId('fixpipeOutputContext'),
    fixpipeAccumTitle: byId('fixpipeAccumTitle'),
    fixpipeAccumCanvas: byId('fixpipeAccumCanvas'),
    fixpipeOutputCanvas: byId('fixpipeOutputCanvas'),
    fixpipeAddressCore: byId('fixpipeAddressCore'),
    fixpipeAddress: byId('fixpipeAddress'),
    convMmadMatrixView: byId('convMmadMatrixView'),
    mmadEquation: byId('mmadEquation'),
    mmadEvidence: byId('mmadEvidence'),
    mmadProgress: byId('mmadProgress'),
    mmadA2Title: byId('mmadA2Title'),
    mmadA2Meta: byId('mmadA2Meta'),
    mmadA2Canvas: byId('mmadA2Canvas'),
    mmadB2Title: byId('mmadB2Title'),
    mmadB2Meta: byId('mmadB2Meta'),
    mmadB2Canvas: byId('mmadB2Canvas'),
    mmadAddend: byId('mmadAddend'),
    mmadCo1Title: byId('mmadCo1Title'),
    mmadCo1Meta: byId('mmadCo1Meta'),
    mmadCo1Canvas: byId('mmadCo1Canvas'),
    mmadBiasStatus: byId('mmadBiasStatus'),
    addendMatrixTitle: byId('mmadAddendTitle'),
    addendMatrixMeta: byId('mmadAddendMeta'),
    addendMatrixCanvas: byId('mmadAddendCanvas'),
    tensorCanvas: byId('tensorCanvas'),
    convLoadDataView: byId('convLoadDataView'),
    fmapA1VolumeCanvas: byId('fmapA1VolumeCanvas'),
    fmapA1Meta: byId('fmapA1Meta'),
    fmapA1Params: byId('fmapA1Params'),
    a2LogicalShape: byId('a2LogicalShape'),
    a2LogicalMatrixCanvas: byId('a2LogicalMatrixCanvas'),
    fmapA2TileMeta: byId('fmapA2TileMeta'),
    tensorFallback: byId('tensorFallback'),
    // viewport controls removed: zoomOut/zoomIn/fitView/viewportInfo
    tileLens: byId('tileLens'),
    architectureKicker: byId('architectureKicker'),
    architectureMeta: byId('architectureMeta'),
    architectureViewportRoot: byId('architectureViewportRoot'),
    architectureViewport: byId('architectureViewport'),
    architectureFrame: byId('architectureFrame'),
    architectureBlocks: byId('architectureBlocks'),
    architectureDetailToggle: byId('architectureDetailToggle'),
    archZoomOut: byId('archZoomOut'),
    archZoomIn: byId('archZoomIn'),
    archFitView: byId('archFitView'),
    archZoomReadout: byId('archZoomReadout'),
    timelineKicker: byId('timelineKicker'),
    instructionSequence: byId('instructionSequence'),
    instructionsView: byId('instructionsView'),
    estimatedTimelineView: byId('estimatedTimelineView'),
    executionTabs: Array.from(document.querySelectorAll('[data-execution-view]')),
    traceInfoPanel: byId('traceInfoPanel'),
    traceInfoMeta: byId('traceInfoMeta'),
    traceInfoContent: byId('traceInfoContent'),
    closeTraceInfo: byId('closeTraceInfo'),
    statusText: byId('statusText'),
    statusSample: byId('statusSample'),
    statusStep: byId('statusStep'),
    statusArch: byId('statusArch'),
    playbackMount: byId('playbackMount'),
    frameActions: Array.from(document.querySelectorAll('[data-frame-command]')),
  };

  function escapeHtml(value) {
    return String(value ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function currentTrace() {
    return state.traces.find((trace) => trace.operator.id === state.sampleId) || state.traces[0];
  }

  function currentStep(trace = currentTrace()) {
    return trace?.steps?.[state.stepIndex] || trace?.steps?.[0] || null;
  }

  async function loadTraces() {
    const traces = await Promise.all(FIXTURES.map(async (fixture) => {
      const response = await fetch(fixture.path);
      if (!response.ok) throw new Error(`Failed to load ${fixture.path}: ${response.status}`);
      return response.json();
    }));
    await Promise.all(traces.map(loadTraceSources));
    state.traces = traces;
    state.sampleId = traces[0]?.operator?.id || null;
    state.sourceFileId = sourceFilesForTrace(traces[0])[0]?.id || null;
  }

  async function loadTraceSources(trace) {
    await Promise.all(sourceFilesForTrace(trace).map((source) => loadSourceFile(trace, source)));
  }

  async function loadSourceFile(trace, source) {
    source.keyLines = source.lines || [];
    for (const url of sourceUrlCandidates(trace, source)) {
      try {
        const response = await fetch(url);
        if (!response.ok) continue;
        const text = await response.text();
        source.fullLines = normalizeSourceLines(text);
        source.sourceUrl = url;
        source.partial = false;
        return;
      } catch {
        // Try the next static source candidate.
      }
    }
    source.fullLines = source.keyLines;
    source.partial = true;
  }

  function sourceFilesForTrace(trace) {
    if (Array.isArray(trace?.sources) && trace.sources.length) return trace.sources;
    return trace?.source ? [{ id: 'kernel', label: trace.source.path, ...trace.source }] : [];
  }

  function activeSourceFile(trace) {
    const sources = sourceFilesForTrace(trace);
    return sources.find((source) => source.id === state.sourceFileId) || sources[0] || null;
  }

  function sourceUrlCandidates(trace, source) {
    const path = source?.path || '';
    const candidates = [];
    if (source?.projectPath) candidates.push(source.projectPath);
    if (path) candidates.push(`data/sources/${encodeURIComponent(path)}`);
    const sourcePath = trace.operator?.sourcePath || '';
    const marker = '/asc-devkit-master/';
    const markerIndex = sourcePath.indexOf(marker);
    if (markerIndex >= 0) {
      candidates.push(`/gitcode/asc-devkit-master/${sourcePath.slice(markerIndex + marker.length)}`);
    }
    if (sourcePath.startsWith('/Users/yin/')) {
      candidates.push(`/${sourcePath.slice('/Users/yin/'.length)}`);
    }
    return [...new Set(candidates)];
  }

  function normalizeSourceLines(text) {
    return String(text || '').replace(/\r\n?/g, '\n').split('\n').map((line, index) => ({
      line: index + 1,
      text: line,
    }));
  }

  function initButtons() {
    els.prevStep?.addEventListener('click', () => selectStep(state.stepIndex - 1));
    els.nextStep?.addEventListener('click', () => selectStep(state.stepIndex + 1));
    els.evidenceToggle?.addEventListener('click', () => {
      state.evidence = !state.evidence;
      els.evidenceToggle.setAttribute('aria-pressed', state.evidence ? 'true' : 'false');
      els.evidenceToggle.classList.toggle('is-selected', state.evidence);
    });
    // removed viewport control event listeners (buttons removed from DOM)
    els.closeTraceInfo?.addEventListener('click', () => {
      setInfoOpen(false);
    });
    els.frameActions.forEach((button) => {
      button.addEventListener('click', () => handleFrameCommand(button.dataset.frameCommand));
    });
    els.architectureDetailToggle?.addEventListener('click', () => {
      state.architecture.detailsVisible = !state.architecture.detailsVisible;
      syncArchitectureControls();
      postArchitectureDetails();
    });
    els.archZoomOut?.addEventListener('click', () => {
      setArchitectureZoom(state.architecture.zoomIndex - 1);
    });
    els.archZoomIn?.addEventListener('click', () => {
      setArchitectureZoom(state.architecture.zoomIndex + 1);
    });
    els.archFitView?.addEventListener('click', () => {
      resetArchitectureViewport();
    });
    els.executionTabs.forEach((button) => {
      button.addEventListener('click', () => setExecutionView(button.dataset.executionView));
    });
    els.instructionSequence?.addEventListener('click', handleInstructionSequenceClick);
    els.memoryAllocationView?.addEventListener('click', handleAllocatedTensorClick);
    els.convCoreOptions?.addEventListener('click', (event) => {
      const button = event.target.closest('[data-conv-core-index]');
      if (!button) return;
      state.convCoreIndex = Math.max(0, Number(button.dataset.convCoreIndex) || 0);
      renderConvCoreContext(currentTrace());
      renderTensorViewport(currentTrace());
      renderInfoPanel(currentTrace());
    });
    els.convCoreOptions?.addEventListener('keydown', (event) => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      const buttons = Array.from(els.convCoreOptions.querySelectorAll('[data-conv-core-index]'));
      if (!buttons.length) return;
      const currentIndex = Math.max(0, buttons.indexOf(event.target.closest('[data-conv-core-index]')));
      const nextIndex = event.key === 'Home'
        ? 0
        : event.key === 'End'
          ? buttons.length - 1
          : (currentIndex + (event.key === 'ArrowRight' ? 1 : -1) + buttons.length) % buttons.length;
      event.preventDefault();
      buttons[nextIndex].click();
      buttons[nextIndex].focus();
    });
    initTensorViewportInteractions();
    initArchitecturePan();
  }

  function handleFrameCommand(command) {
    if (command === 'fit-tensor') {
      resetTensorView();
      return;
    }
    if (command === 'fit-architecture') {
      els.archFitView?.click();
      return;
    }
    if (command === 'toggle-info') {
      setInfoOpen(!state.infoOpen);
      return;
    }
  }

  function setInfoOpen(open) {
    state.infoOpen = !!open;
    renderInfoPanel();
    syncFrameActions();
  }

  function setExecutionView(view) {
    state.executionView = view === 'timeline' ? 'timeline' : 'instructions';
    renderExecutionDock();
    if (state.executionView === 'instructions') {
      renderInstructionPanel(currentTrace());
    }
  }

  function renderExecutionDock() {
    const isInstructions = state.executionView === 'instructions';
    els.executionTabs.forEach((button) => {
      const selected = button.dataset.executionView === state.executionView;
      button.classList.toggle('is-selected', selected);
      button.setAttribute('aria-selected', String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
    if (els.instructionsView) els.instructionsView.hidden = !isInstructions;
    if (els.estimatedTimelineView) els.estimatedTimelineView.hidden = isInstructions;
    if (els.timelineKicker) {
      const step = currentStep();
      els.timelineKicker.textContent = isInstructions
        ? `${step?.evidenceKind || 'unknown'} · logical order · repeated iterations grouped`
        : 'Estimated Timeline unavailable · Not Profiling Data';
    }
  }

  function setInstructionLoopExpanded(expanded) {
    state.instructionLoopExpanded = Boolean(expanded);
    renderExecutionDock();
    renderInstructionPanel(currentTrace());
  }

  function syncFrameActions() {
    els.frameActions.forEach((button) => {
      const command = button.dataset.frameCommand;
      const selected = command === 'toggle-info' && state.infoOpen;
      if (command === 'toggle-info') {
        button.classList.toggle('is-selected', selected);
        button.setAttribute('aria-expanded', String(selected));
        button.setAttribute('aria-pressed', String(selected));
      }
    });
    // viewportInfo element removed; frame action toggle still kept via frameActions
  }

  function refreshArchitectureViewport(options = {}) {
    const run = () => {
      if (options.fit) {
        state.architecture.fitOnReady = true;
        fitArchitectureViewport();
      }
      syncArchitectureControls();
    };
    window.requestAnimationFrame(() => window.requestAnimationFrame(run));
  }

  function architectureScale() {
    return ARCH_ZOOM_LEVELS[state.architecture.zoomIndex] || 0.6;
  }

  function setArchitectureZoom(index) {
    state.architecture.fitOnReady = false;
    state.architecture.zoomIndex = Math.max(0, Math.min(ARCH_ZOOM_LEVELS.length - 1, index));
    syncArchitectureControls();
    postArchitectureScale();
  }

  function resetArchitectureViewport() {
    state.architecture.fitOnReady = true;
    fitArchitectureViewport();
    syncArchitectureControls();
    postArchitectureScale();
  }

  function fitArchitectureViewport() {
    const viewportRect = els.architectureViewport?.getBoundingClientRect?.();
    const frameWidth = state.architecture.frameWidth || 1900;
    const frameHeight = state.architecture.frameHeight || 2400;
    const viewportWidth = Math.max(1, viewportRect?.width || 0);
    const viewportHeight = Math.max(1, viewportRect?.height || 0);
    const verticalFitScale = viewportHeight > 1 ? (viewportHeight - 24) / frameHeight : 0.6;
    const maxScale = Math.min(0.6, Math.max(ARCH_ZOOM_LEVELS[0], verticalFitScale));
    let nextIndex = 0;
    ARCH_ZOOM_LEVELS.forEach((level, index) => {
      if (level <= maxScale + 0.001) nextIndex = index;
    });
    state.architecture.zoomIndex = nextIndex;
    const scale = architectureScale();
    const scaledWidth = frameWidth * scale;
    const scaledHeight = frameHeight * scale;
    state.architecture.panX = Math.round((viewportWidth - scaledWidth) / 2);
    state.architecture.panY = Math.max(0, Math.round((viewportHeight - scaledHeight) / 2));
  }

  function applyArchitectureFrameSize(payload = {}) {
    const width = Number(payload.width);
    const height = Number(payload.height);
    let changed = false;
    if (Number.isFinite(width) && width > 0) {
      state.architecture.frameWidth = Math.ceil(width);
      changed = true;
    }
    if (Number.isFinite(height) && height > 0) {
      state.architecture.frameHeight = Math.ceil(height);
      changed = true;
    }
    if (changed && state.architecture.fitOnReady) {
      fitArchitectureViewport();
    }
    syncArchitectureControls();
  }

  function syncArchitectureControls() {
    const scale = architectureScale();
    if (els.architectureViewport) {
      els.architectureViewport.style.setProperty('--avz-hw-scale', String(scale));
      els.architectureViewport.style.setProperty('--avz-hw-pan-x', `${Math.round(state.architecture.panX)}px`);
      els.architectureViewport.style.setProperty('--avz-hw-pan-y', `${Math.round(state.architecture.panY)}px`);
      els.architectureViewport.style.setProperty('--avz-hw-frame-width', `${Math.ceil(state.architecture.frameWidth || 1900)}px`);
      els.architectureViewport.style.setProperty('--avz-hw-frame-height', `${Math.ceil(state.architecture.frameHeight || 2400)}px`);
    }
    if (els.archZoomReadout) els.archZoomReadout.textContent = `${Math.round(scale * 100)}%`;
    if (els.archZoomOut) els.archZoomOut.disabled = state.architecture.zoomIndex <= 0;
    if (els.archZoomIn) els.archZoomIn.disabled = state.architecture.zoomIndex >= ARCH_ZOOM_LEVELS.length - 1;
    if (els.architectureDetailToggle) {
      els.architectureDetailToggle.textContent = state.architecture.detailsVisible ? '细节开' : '细节关';
      els.architectureDetailToggle.title = state.architecture.detailsVisible ? '隐藏细节数据' : '显示细节数据';
      els.architectureDetailToggle.setAttribute('aria-label', els.architectureDetailToggle.title);
      els.architectureDetailToggle.setAttribute('aria-pressed', String(state.architecture.detailsVisible));
    }
  }

  function postArchitectureDetails() {
    if (!els.architectureFrame?.contentWindow) return;
    els.architectureFrame.contentWindow.postMessage({
      type: 'hardware-details',
      visible: state.architecture.detailsVisible,
    }, '*');
  }

  function postArchitectureScale() {
    if (!els.architectureFrame?.contentWindow) return;
    els.architectureFrame.contentWindow.postMessage({
      type: 'hardware-scale',
      scale: architectureScale(),
    }, '*');
  }

  function setArchitecturePanCursor(active) {
    els.architectureViewport?.classList.toggle('is-panning', active);
  }

  function beginArchitecturePan(event) {
    if (event.button !== undefined && event.button !== 0) return;
    const target = event.target?.nodeType === 1 ? event.target : null;
    if (target?.closest?.('button, input, textarea, select, a')) return;
    const point = architecturePointerPoint(event);
    state.architecture.fitOnReady = false;
    state.architecture.dragging = true;
    state.architecture.startX = point.x;
    state.architecture.startY = point.y;
    state.architecture.startPanX = state.architecture.panX;
    state.architecture.startPanY = state.architecture.panY;
    state.architecture.moved = false;
    state.architecture.panCaptureTarget = null;
    setArchitecturePanCursor(true);
  }

  function moveArchitecturePan(event) {
    if (!state.architecture.dragging) return;
    if (event.buttons !== undefined && event.buttons === 0) {
      endArchitecturePan(event);
      return;
    }
    const point = architecturePointerPoint(event);
    const dx = point.x - state.architecture.startX;
    const dy = point.y - state.architecture.startY;
    if (Math.abs(dx) + Math.abs(dy) > 3) state.architecture.moved = true;
    state.architecture.panX = state.architecture.startPanX + dx;
    state.architecture.panY = state.architecture.startPanY + dy;
    syncArchitectureControls();
    event.preventDefault?.();
  }

  function architecturePointerPoint(event) {
    return {
      x: Number.isFinite(event.screenX) ? event.screenX : event.clientX,
      y: Number.isFinite(event.screenY) ? event.screenY : event.clientY,
    };
  }

  function endArchitecturePan(event) {
    if (!state.architecture.dragging) return;
    state.architecture.dragging = false;
    state.architecture.panCaptureTarget = null;
    setArchitecturePanCursor(false);
  }

  function bindArchitectureFramePan() {
    let frameDoc = null;
    try {
      frameDoc = els.architectureFrame?.contentDocument;
    } catch {
      return;
    }
    if (!frameDoc || state.architecture.framePanDocument === frameDoc) return;
    state.architecture.framePanDocument = frameDoc;
    frameDoc.addEventListener('pointerdown', beginArchitecturePan);
    frameDoc.addEventListener('pointermove', moveArchitecturePan);
    frameDoc.addEventListener('pointerup', endArchitecturePan);
    frameDoc.addEventListener('pointercancel', endArchitecturePan);
    frameDoc.addEventListener('lostpointercapture', endArchitecturePan);
  }

  function initArchitecturePan() {
    const viewport = els.architectureViewport;
    if (!viewport) return;
    viewport.addEventListener('pointerdown', beginArchitecturePan);
    viewport.addEventListener('pointermove', moveArchitecturePan);
    viewport.addEventListener('pointerup', endArchitecturePan);
    viewport.addEventListener('pointercancel', endArchitecturePan);
    viewport.addEventListener('lostpointercapture', endArchitecturePan);
    window.addEventListener('pointermove', moveArchitecturePan);
    window.addEventListener('pointerup', endArchitecturePan);
    window.addEventListener('pointercancel', endArchitecturePan);
    window.addEventListener('blur', endArchitecturePan);
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) endArchitecturePan();
    });
  }

  function initTensorViewportInteractions() {
    const canvas = els.tensorCanvas;
    if (!canvas) return;
    canvas.addEventListener('pointerdown', (event) => {
      state.tensorView.dragging = true;
      state.tensorView.startX = event.clientX;
      state.tensorView.startY = event.clientY;
      state.tensorView.startPanX = state.tensorView.panX;
      state.tensorView.startPanY = state.tensorView.panY;
      state.tensorView.moved = false;
      canvas.setPointerCapture?.(event.pointerId);
    });
    canvas.addEventListener('pointermove', (event) => {
      if (!state.tensorView.dragging) return;
      const dx = event.clientX - state.tensorView.startX;
      const dy = event.clientY - state.tensorView.startY;
      if (Math.abs(dx) + Math.abs(dy) > 4) state.tensorView.moved = true;
      state.tensorView.panX = state.tensorView.startPanX + dx;
      state.tensorView.panY = state.tensorView.startPanY + dy;
      if (!state.tensorView.raf) {
        state.tensorView.raf = window.requestAnimationFrame(() => {
          state.tensorView.raf = 0;
          renderTensorViewport(currentTrace());
        });
      }
    });
    canvas.addEventListener('pointerup', (event) => {
      canvas.releasePointerCapture?.(event.pointerId);
      state.tensorView.dragging = false;
    });
    canvas.addEventListener('pointercancel', () => {
      state.tensorView.dragging = false;
    });
    canvas.addEventListener('wheel', (event) => {
      if (!(event.metaKey || event.ctrlKey)) return;   // zoom only with Cmd/Ctrl + wheel
      event.preventDefault();
      const factor = event.deltaY < 0 ? 1.1 : 0.9;
      state.tensorView.scale = Math.max(0.55, Math.min(2.6, (state.tensorView.scale || 1) * factor));
      if (!state.tensorView.raf) {
        state.tensorView.raf = window.requestAnimationFrame(() => {
          state.tensorView.raf = 0;
          renderTensorViewport(currentTrace());
        });
      }
    }, { passive: false });
  }

  function zoomTensorView(multiplier) {
    if (state.activeTensorRenderer === 'matrix' && state.tensorMatrixController) {
      const view = state.tensorMatrixController.getViewState();
      state.tensorMatrixController.setZoom(view.scale * multiplier);
      return;
    }
    state.tensorView.scale = Math.max(0.55, Math.min(2.4, state.tensorView.scale * multiplier));
    renderTensorViewport(currentTrace());
  }

  function resetTensorView() {
    if (state.activeTensorRenderer === 'overview') {
      state.overviewControllers.feature?.resize?.();
      state.overviewControllers.output?.resize?.();
      state.overviewControllers.weight?.resize?.();
      state.overviewControllers.bias?.fit?.();
      return;
    }
    if (state.activeTensorRenderer === 'matrix' && state.tensorMatrixController) {
      state.tensorMatrixController.fit();
      return;
    }
    if (state.activeTensorRenderer === 'mmad-matrix') {
      Object.values(state.mmadMatrixControllers).forEach((controller) => controller?.fit?.());
      return;
    }
    if (state.activeTensorRenderer === 'host-tiling') {
      state.hostTilingControllers.source?.resize?.();
      state.hostTilingControllers.cube?.fit?.();
      return;
    }
    if (state.activeTensorRenderer === 'host-launch') {
      Object.values(state.hostLaunchControllers).forEach((controller) => controller?.fit?.());
      return;
    }
    if (state.activeTensorRenderer === 'copy-input-pattern') {
      ['source', 'destination'].forEach((slot) => {
        const controller = state.copyInputControllers[slot];
        const kind = state.copyInputControllers[`${slot}Kind`];
        if (kind === 'volume') controller?.resize?.();
        else controller?.fit?.();
      });
      return;
    }
    if (state.activeTensorRenderer === 'bias-c1-c2-matrix') {
      Object.values(state.biasC1C2Controllers).forEach((controller) => controller?.fit?.());
      return;
    }
    if (state.activeTensorRenderer === 'volume' && state.tensorVolumeController) {
      state.tensorVolumeController.resize();
      return;
    }
    if (state.activeTensorRenderer === 'load-data') {
      state.fmapA1VolumeController?.resize?.();
      state.a2LogicalMatrixController?.fit?.();
      return;
    }
    state.tensorView.scale = 1;
    state.tensorView.panX = 0;
    state.tensorView.panY = 0;
    renderTensorViewport(currentTrace());
  }

  function initPlayback() {
    const helper = window.PtoFloatingPlaybackControl;
    if (!helper?.createControl) return;
    els.playbackMount.innerHTML = '';
    const control = helper.createControl({
      ids: state.playbackIds,
      className: 'pto-floating-playback--preview pto-floating-playback--tiling',
      showTimeline: false,
    });
    els.playbackMount.appendChild(control);
    state.playback = helper.init({
      root: control,
      isPlaying: () => state.playing,
    });
    byId(state.playbackIds.stepBack)?.addEventListener('click', () => selectStep(state.stepIndex - 1));
    byId(state.playbackIds.stepForward)?.addEventListener('click', () => selectStep(state.stepIndex + 1));
    byId(state.playbackIds.replay)?.addEventListener('click', () => selectStep(0));
    byId(state.playbackIds.play)?.addEventListener('click', togglePlay);
    byId(state.playbackIds.scrubber)?.addEventListener('input', (event) => {
      stopPlayback();
      selectStep(Number(event.target.value) || 0);
    });
  }

  function stopPlayback() {
    state.playing = false;
    if (state.timer) {
      window.clearInterval(state.timer);
      state.timer = null;
    }
    syncPlayback();
  }

  function togglePlay() {
    state.playing = !state.playing;
    if (state.playing) {
      state.timer = window.setInterval(() => {
        const trace = currentTrace();
        const max = Math.max(0, (trace?.steps?.length || 1) - 1);
        if (state.stepIndex >= max) {
          selectStep(0, { keepPlaying: true });
          return;
        }
        selectStep(state.stepIndex + 1, { keepPlaying: true });
      }, 900);
    } else if (state.timer) {
      window.clearInterval(state.timer);
      state.timer = null;
    }
    syncPlayback();
  }

  function selectSample(sampleId) {
    stopPlayback();
    state.sampleId = sampleId;
    state.stepIndex = 0;
    state.instructionIterationFocus = null;
    state.instructionIterationRange = null;
    state.instructionOperationFocus = null;
    state.instructionLoopExpanded = false;
    state.sourceFileId = sourceFilesForTrace(currentTrace())[0]?.id || null;
    render();
  }

  function selectStep(index, options = {}) {
    const trace = currentTrace();
    const max = Math.max(0, (trace?.steps?.length || 1) - 1);
    state.stepIndex = Math.max(0, Math.min(max, index));
    state.instructionIterationFocus = Number.isInteger(options.instructionIteration)
      ? options.instructionIteration
      : null;
    state.instructionIterationRange = Array.isArray(options.instructionIterationRange)
      && options.instructionIterationRange.length === 2
      ? options.instructionIterationRange.map(Number)
      : null;
    state.instructionOperationFocus = options.instructionOperation || null;
    const primaryRef = sourceRefsForStep(currentStep(trace))[0];
    if (!options.keepSource && primaryRef?.fileId && primaryRef.fileId !== state.sourceFileId) {
      state.sourceFileId = primaryRef.fileId;
      renderSourceTabs(trace);
      renderSource(trace);
    }
    if (!options.keepPlaying) stopPlayback();
    renderStep();
  }

  function selectSourceFile(fileId) {
    const trace = currentTrace();
    if (!trace || fileId === state.sourceFileId) return;
    state.sourceFileId = fileId;
    renderSourceTabs(trace);
    renderSource(trace);
    renderChrome(trace);
  }

  function syncPlayback() {
    const trace = currentTrace();
    const steps = trace?.steps || [];
    const scrubber = byId(state.playbackIds.scrubber);
    const label = byId(state.playbackIds.scrubberLabel);
    const opname = byId(state.playbackIds.scrubberOpname);
    const play = byId(state.playbackIds.play);
    const helper = window.PtoFloatingPlaybackControl;
    if (scrubber) {
      scrubber.max = String(Math.max(0, steps.length - 1));
      scrubber.value = String(state.stepIndex);
    }
    if (label) label.textContent = `${state.stepIndex} / ${Math.max(0, steps.length - 1)}`;
    if (opname) opname.textContent = currentStep(trace)?.label || '-';
    if (play && helper?.iconLabel) {
      play.innerHTML = state.playing ? helper.iconLabel('pause', 'Pause') : helper.iconLabel('play', 'Play');
    }
    state.playback?.sync?.({ playing: state.playing });
  }

  // Full render: rebuilds the source listing and sample cards. Only needed on
  // sample switch / init, never on every step.
  function render() {
    const trace = currentTrace();
    if (!trace) return;
    state.stepIndex = Math.max(0, Math.min(state.stepIndex, trace.steps.length - 1));
    renderSourceTabs(trace);
    renderSource(trace);
    renderStep();
  }

  // Light render: runs on every step change / playback tick. Updates only what
  // actually changes per step — no source re-tokenize, no DOM teardown.
  function renderStep() {
    const trace = currentTrace();
    if (!trace) return;
    state.stepIndex = Math.max(0, Math.min(state.stepIndex, trace.steps.length - 1));
    renderChrome(trace);
    updateSourceHighlight(trace);
    renderTensorTabs(trace);
    renderTensorViewport(trace);
    renderTileLens(trace);
    renderArchitectureFocus(trace);
    renderExecutionDock();
    renderInstructionPanel(trace);
    renderInfoPanel(trace);
    syncPlayback();
  }

  function renderChrome(trace) {
    const step = currentStep(trace);
    const source = activeSourceFile(trace);
    const sourceLines = sourceLinesForTrace(trace, source);
    if (els.operatorMeta) els.operatorMeta.textContent = '';
    if (els.archReadout) els.archReadout.textContent = 'Ascend 910B';
    if (els.architectureMeta) els.architectureMeta.textContent = 'Ascend 910B';
    if (els.sourceMeta) {
      const suffix = source?.partial ? '关键行' : `${sourceLines.length} 行`;
      els.sourceMeta.textContent = `${source?.path || 'source'} · ${suffix}`;
    }
    if (els.visualTitle) {
      els.visualTitle.textContent = step?.stageId === 'allocate'
        ? 'Local Memory Map'
        : 'Tensor State & Transformation';
    }
    if (els.stepMeta) {
      els.stepMeta.textContent = step?.stageId === 'allocate'
        ? 'LocalTensor views · no data movement'
        : step ? `${state.stepIndex + 1}/${trace.steps.length}` : '';
    }
    if (els.statusText) els.statusText.textContent = step ? zh(step.label) : 'Ready';
    if (els.statusSample) els.statusSample.textContent = sampleShortName(trace);
    if (els.statusStep) els.statusStep.textContent = step ? `${state.stepIndex + 1}/${trace.steps.length}` : '0/0';
    if (els.statusArch) els.statusArch.textContent = 'Ascend 910B';
    syncFrameActions();
  }

  function tensorTabsForStep(trace, step) {
    if (trace?.operator?.kind !== 'conv2d-cube' || !step) return [];
    if (step.stageId === 'allocate' || step.stageId === 'fixpipe-output') return [];
    if (step.stageId === 'host-tiling') {
      return [
        { key: 'host-tiling:feature', label: 'Input X', location: 'Logical X → Cube A[M,K]' },
        { key: 'host-tiling:weight', label: 'Weight W', location: 'Logical W → Cube B[K,N]' },
        { key: 'host-tiling:output', label: 'Output Y', location: 'Logical Y → Cube C[M,N]' },
      ];
    }
    if (step.stageId === 'copy-inputs') {
      return (step.dataFlows || []).map((flow, index) => {
        const source = (trace.tensors || []).find((item) => item.id === flow.tensorId);
        const labelByRole = {
          input: 'Feature X',
          weight: 'Weight W',
          bias: 'Bias',
        };
        return {
          key: flow.bufferId || flow.tensorId || `copy-input:${index}`,
          label: labelByRole[source?.role] || source?.name || flow.from || `Input ${index + 1}`,
          location: `${flow.from || 'GM'} → ${flow.to || 'L1'}`,
        };
      });
    }
    const snapshots = step.tensorSnapshots || [];
    if (snapshots.length < 2) return [];
    const definitions = [...(trace.tensors || []), ...(trace.buffers || [])];
    return snapshots.map((snapshot) => {
      const definition = definitions.find((item) => item.id === snapshot.tensorId);
      return {
        key: snapshot.tensorId,
        label: definition?.name || snapshot.tensorId?.replace(/^(tensor|buffer):/, '') || 'Tensor',
        location: snapshot.location || definition?.location || '',
      };
    });
  }

  function selectedCopyInputTransfer(trace, step) {
    if (step?.stageId !== 'copy-inputs') return null;
    const flows = step.dataFlows || [];
    const flow = flows.find((item) => item.bufferId === state.tensorTabKey || item.tensorId === state.tensorTabKey)
      || flows[0]
      || null;
    if (!flow) return null;
    const source = (trace?.tensors || []).find((item) => item.id === flow.tensorId) || null;
    const destination = (trace?.buffers || []).find((item) => item.id === flow.bufferId) || null;
    const snapshot = (step.tensorSnapshots || []).find((item) => item.tensorId === flow.bufferId) || null;
    const gateEvent = (trace?.events || []).find((item) => item.eventType === 'MTE2_MTE1') || null;
    const partition = convCorePartition(trace);
    const params = trace?.tiling?.params || {};
    const tileN = num(params.tileN, 16);
    const nStart = partition.nTile * tileN;
    const nEnd = Math.min(num(params.N, 32), nStart + tileN);
    const sourceRole = source?.role || '';
    const coreSource = sourceRole === 'weight'
      ? {
          label: 'GM / Weight W · full matrix',
          slice: `W[K 0:${num(params.K, 144)}, N ${nStart}:${nEnd}]`,
          gmOffsetBytes: nStart * 2,
          addressing: `row stride N=${num(params.N, 32)}`,
        }
      : sourceRole === 'bias'
        ? {
            label: `GM / Bias D · N${nStart}:${nEnd}`,
            slice: `Bias[N ${nStart}:${nEnd}]`,
            gmOffsetBytes: nStart * 4,
            addressing: 'contiguous FP32',
          }
        : {
            label: 'GM / Feature X0 · full input',
            slice: `X[0:1, 0:${num(params.ci, 16)}, 0:${num(params.hi, 8)}, 0:${num(params.wi, 8)}]`,
            gmOffsetBytes: 0,
            addressing: 'shared by all output tiles',
          };
    return {
      flow,
      source,
      destination,
      snapshot,
      gateEvent,
      partition,
      coreSource,
      transferCount: flows.length,
      totalBytes: flows.reduce((total, item) => total + (Number(item.bytes) || 0), 0),
    };
  }

  function convCorePartition(trace, coreIndex = state.convCoreIndex) {
    const params = trace?.tiling?.params || {};
    const derived = trace?.tiling?.derived || {};
    const mTiles = Math.max(1, Number(derived.mTileCount) || Math.ceil(params.M / params.tileM));
    const nTiles = Math.max(1, Number(derived.nTileCount) || Math.ceil(params.N / params.tileN));
    const blockCount = Math.max(
      1,
      Number(derived.outputTileCount) || Number(trace?.launch?.numBlocks) || mTiles * nTiles,
    );
    const index = Math.min(Math.max(0, Number(coreIndex) || 0), blockCount - 1);
    return {
      index,
      outputTile: index,
      mTile: Math.floor(index / nTiles),
      nTile: index % nTiles,
      mTiles,
      nTiles,
      blockCount,
    };
  }

  function renderTensorTabs(trace) {
    if (!els.tensorTabs) return;
    const step = currentStep(trace);
    const tabs = tensorTabsForStep(trace, step);
    if (state.tensorTabStepId !== step?.id || !tabs.some((tab) => tab.key === state.tensorTabKey)) {
      state.tensorTabStepId = step?.id || null;
      state.tensorTabKey = tabs[0]?.key || null;
    }
    els.tensorTabs.hidden = tabs.length < 2;
    if (tabs.length < 2) {
      els.tensorTabs.innerHTML = '';
      return;
    }
    els.tensorTabs.innerHTML = tabs.map((tab) => {
      const selected = tab.key === state.tensorTabKey;
      return `<button class="tab-control-item ${selected ? 'is-selected' : ''}" type="button" role="tab" aria-selected="${selected ? 'true' : 'false'}" data-tensor-tab="${escapeHtml(tab.key)}" title="${escapeHtml(tab.location)}">${escapeHtml(tab.label)}</button>`;
    }).join('');
    els.tensorTabs.querySelectorAll('[data-tensor-tab]').forEach((button) => {
      button.addEventListener('click', () => {
        if (button.dataset.tensorTab === state.tensorTabKey) return;
        state.tensorTabKey = button.dataset.tensorTab;
        renderTensorTabs(trace);
        renderTensorViewport(trace);
        renderTileLens(trace);
        renderInfoPanel(trace);
      });
    });
  }

  function renderSourceTabs(trace) {
    const items = sourceFilesForTrace(trace).map((source) => {
      const active = source.id === state.sourceFileId;
      return `<button class="tab-control-item ${active ? 'is-selected' : ''}" type="button" role="tab" aria-selected="${active ? 'true' : 'false'}" data-source-file="${escapeHtml(source.id)}" title="${escapeHtml(source.path)}">${escapeHtml(source.label || source.path)}</button>`;
    }).join('');
    els.sampleList.innerHTML = `<div class="tab-control avz-source-tabs" role="tablist" aria-label="Source files">${items}</div>`;
    els.sampleList.querySelectorAll('[data-source-file]').forEach((button) => {
      button.addEventListener('click', () => selectSourceFile(button.dataset.sourceFile));
    });
  }

  function sampleShortName(trace) {
    if (trace.operator.kind === 'conv2d-cube') return 'Conv2D + Bias + ReLU';
    if (trace.operator.kind === 'cube') return 'Cube Matmul';
    if (trace.operator.kind === 'fusion') return 'Fusion';
    return 'Vector Add';
  }

  function zh(value) {
    return TEXT_ZH[String(value ?? '')] || String(value ?? '');
  }

  // Build the source listing once per trace. The per-step active highlight is
  // applied separately by updateSourceHighlight (class toggle only).
  function renderSource(trace) {
    const source = activeSourceFile(trace);
    const lines = sourceLinesForTrace(trace, source);
    const keyLines = new Set((source?.keyLines || source?.lines || []).map((line) => line.line));
    trace.steps.forEach((step) => sourceLinesForStep(step, source?.id).forEach((line) => keyLines.add(line)));
    const stageByLine = new Map();
    trace.steps.forEach((step) => {
      const stage = trace.stages.find((item) => item.id === step.stageId) || null;
      sourceLinesForStep(step, source?.id).forEach((ln) => { if (!stageByLine.has(ln)) stageByLine.set(ln, stage); });
    });
    els.sourceLines.innerHTML = lines.map((line) => {
      const stage = stageByLine.get(line.line) || null;
      const hasCode = String(line.text || '').trim().length > 0;
      const isKey = hasCode && keyLines.has(line.line);
      const kind = stageKind(stage);
      const tag = isKey ? `<span class="avz-source-line__tag ${kind ? `is-${kind}` : ''}">${escapeHtml(sourceLineTag(stage))}</span>` : '<span></span>';
      const element = isKey ? 'button' : 'div';
      const attrs = isKey ? `type="button" data-line="${line.line}" role="option" aria-selected="false"` : '';
      return `
        <${element} class="avz-source-line ${isKey ? 'is-key' : ''} ${kind ? `is-${kind}` : ''}" ${attrs}>
          <span class="avz-source-line__number">${line.line}</span>
          <span class="avz-source-line__text">${highlightAscendC(line.text)}</span>
          ${tag}
        </${element}>
      `;
    }).join('');
    els.sourceLines.querySelectorAll('[data-line]').forEach((button) => {
      button.addEventListener('click', () => {
        const line = Number(button.dataset.line);
        const nextIndex = trace.steps.findIndex((step) => sourceLinesForStep(step, source?.id).includes(line));
        if (nextIndex >= 0) selectStep(nextIndex, { keepSource: true });
      });
    });
    updateSourceHighlight(trace);
  }

  function updateSourceHighlight(trace) {
    const container = els.sourceLines;
    if (!container) return;
    const source = activeSourceFile(trace);
    const activeLines = new Set(sourceLinesForStep(currentStep(trace), source?.id));
    let firstActive = null;
    container.querySelectorAll('[data-line]').forEach((el) => {
      const isActive = activeLines.has(Number(el.dataset.line));
      el.classList.toggle('is-active', isActive);
      el.setAttribute('aria-selected', isActive ? 'true' : 'false');
      if (isActive && !firstActive) firstActive = el;
    });
    if (firstActive) {
      window.requestAnimationFrame(() => scrollChildIntoView(container, firstActive));
    }
  }

  function scrollChildIntoView(container, child) {
    if (!container || !child) return;
    const containerRect = container.getBoundingClientRect();
    const childRect = child.getBoundingClientRect();
    const delta = childRect.top - containerRect.top - ((container.clientHeight - childRect.height) / 2);
    container.scrollTop += delta;
  }

  function sourceLinesForTrace(trace, source = activeSourceFile(trace)) {
    return source?.fullLines?.length ? source.fullLines : source?.lines || [];
  }

  function sourceRefsForStep(step) {
    if (Array.isArray(step?.sourceRefs) && step.sourceRefs.length) return step.sourceRefs;
    return (step?.sourceLines || []).length ? [{ fileId: 'kernel', lines: step.sourceLines }] : [];
  }

  function sourceLinesForStep(step, fileId) {
    return sourceRefsForStep(step)
      .filter((ref) => ref.fileId === fileId)
      .flatMap((ref) => ref.lines || []);
  }

  function sourceStageForLine(trace, lineNo) {
    const fileId = activeSourceFile(trace)?.id;
    const current = currentStep(trace);
    if (sourceLinesForStep(current, fileId).includes(lineNo)) return trace.stages.find((stage) => stage.id === current.stageId) || null;
    const step = trace.steps.find((item) => sourceLinesForStep(item, fileId).includes(lineNo));
    return trace.stages.find((stage) => stage.id === step?.stageId) || null;
  }

  function stageKind(stage) {
    const id = String(stage?.id || '').toLowerCase();
    const label = String(stage?.label || '').toLowerCase();
    if (id.startsWith('host-') || id === 'k-loop' || id.includes('sync') || label.includes('sync') || id.includes('init') || id.includes('launch')) return 'control';
    if (id.includes('copy') || id.includes('load') || id.includes('fixpipe') || id.includes('bias-')) return 'memory';
    if (id.includes('compute') || id.includes('mmad') || id.includes('loop-body') || id.includes('matmul') || id.includes('leakyrelu')) return 'compute';
    return '';
  }

  function sourceLineTag(stage) {
    if (!stage) return 'trace';
    const map = {
      init: 'block 切分',
      'copy-in': 'GM -> UB',
      compute: 'Vector 计算',
      'copy-out': 'UB -> GM',
      'load-data': 'L1 -> L0',
      mmad: 'Mmad',
      fixpipe: 'Fixpipe',
      'mix-launch': '__mix__',
      'aic-matmul': 'AIC',
      'cross-core-sync': '同步',
      'aiv-leakyrelu': 'AIV',
    };
    return map[stage.id] || stage.semanticLabel || stage.label || 'trace';
  }

  function highlightAscendC(code) {
    const escaped = escapeHtml(code);
    const re = /(\/\/.*$)|(\/\*.*?\*\/)|(&quot;(?:\\.|[^&])*?&quot;)|(&#39;(?:\\.|[^&])*?&#39;)|(\b\d+(?:\.\d+)?\b)|([A-Za-z_][A-Za-z0-9_]*)/g;
    let out = '';
    let last = 0;
    let match;
    while ((match = re.exec(escaped)) !== null) {
      if (match.index > last) out += escaped.slice(last, match.index);
      if (match[1] || match[2]) out += `<span class="tk-comment">${match[0]}</span>`;
      else if (match[3] || match[4]) out += `<span class="tk-string">${match[0]}</span>`;
      else if (match[5]) out += `<span class="tk-number">${match[5]}</span>`;
      else if (match[6]) {
        const id = match[6];
        const next = escaped[re.lastIndex];
        if (CPP_KEYWORDS.has(id)) out += `<span class="tk-keyword">${id}</span>`;
        else if (CPP_TYPES.has(id)) out += `<span class="tk-type">${id}</span>`;
        else if (next === '(') out += `<span class="tk-fn">${id}</span>`;
        else out += id;
      }
      last = re.lastIndex;
    }
    if (last < escaped.length) out += escaped.slice(last);
    return out;
  }

  function visualStateForStep(trace, step) {
    const derived = deriveVisualState(trace, step);
    const explicit = step?.visualState || {};
    return {
      tensorViewport: {
        ...derived.tensorViewport,
        ...(explicit.tensorViewport || {}),
      },
      onChipLens: {
        ...derived.onChipLens,
        ...(explicit.onChipLens || {}),
      },
      architectureFocus: {
        ...derived.architectureFocus,
        ...(explicit.architectureFocus || {}),
        bufferBlocks: explicit.architectureFocus?.bufferBlocks || derived.architectureFocus.bufferBlocks,
      },
    };
  }

  function deriveVisualState(trace, step) {
    if (trace.operator.kind === 'conv2d-cube') return deriveConvVisualState(step, trace);
    if (trace.operator.kind === 'cube') return deriveCubeVisualState(step, trace);
    if (trace.operator.kind === 'fusion') return deriveFusionVisualState(step, trace);
    return deriveVectorVisualState(step, trace);
  }

  function num(value, fallback) {
    const n = Number(value);
    return Number.isFinite(n) && n > 0 ? n : fallback;
  }

  function deriveVectorVisualState(step, trace) {
    const params = trace?.tiling?.params || {};
    const derived = trace?.tiling?.derived || {};
    const total = num(params.totalLength, 16384);
    const numBlocks = num(params.numBlocks ?? derived.numBlocks, 8);
    const blockLength = num(derived.blockLength, Math.floor(total / numBlocks));
    const tileLength = num(derived.tileLength, 128);
    const blockIdx = Number(step?.blockIdx ?? -1);
    const progress = Number(step?.loop?.progress || 0);
    const stage = step?.stageId || '';
    const isCopyOut = stage.includes('copy-out');
    const isCompute = stage.includes('compute');
    const hasTile = stage.includes('copy') || isCompute;
    const tone = isCopyOut ? 'output' : isCompute ? 'compute' : 'input';
    const safeBlock = Math.max(0, blockIdx);
    const tileStart = safeBlock * blockLength + progress * tileLength;
    const tileEnd = Math.min(total, tileStart + tileLength);
    const segments = Array.from({ length: numBlocks }, (_, i) => ({
      start: i * blockLength,
      end: Math.min(total, (i + 1) * blockLength),
      label: `block ${i}`,
      active: i === blockIdx,
    }));
    const blocks = vectorBufferBlocks(stage, safeBlock, progress);
    return {
      tensorViewport: {
        kind: 'vector',
        layout: '1d',
        title: `1D 逻辑 tensor · GM 线性地址 0 → ${total}（${numBlocks} 个 block × ${blockLength} 元素，tile=${tileLength}）`,
        axisLabels: ['GM element offset'],
        strip: { total, segments, tickStep: blockLength, blockLength, tileLength },
        highlight: hasTile ? {
          x: [tileStart, tileEnd],
          tone,
          state: isCopyOut ? 'committed' : isCompute ? 'computing' : 'loaded',
          label: `${isCompute || isCopyOut ? 'z' : 'x/y'}[${tileStart}:${tileEnd}]`,
          sub: `block ${safeBlock} · tile ${progress}`,
        } : null,
        operationChips: ['DataCopy', 'TQue', isCompute ? 'Add' : isCopyOut ? 'CopyOut' : 'CopyIn'],
      },
      onChipLens: { blocks },
      architectureFocus: {
        selectors: vectorSelectors(stage),
        routes: vectorRoutes(stage),
        bufferBlocks: blocks,
      },
    };
  }

  function deriveCubeVisualState(step, trace) {
    const params = trace?.tiling?.params || {};
    const derived = trace?.tiling?.derived || {};
    const M = num(params.M, 512);
    const N = num(params.N, 1024);
    const baseM = num(params.baseM, 128);
    const baseN = num(params.baseN, 256);
    const singleCoreM = num(params.singleCoreM, M);
    const singleCoreN = num(params.singleCoreN, N);
    const mIter = num(derived.mIter, Math.max(1, Math.round(M / singleCoreM)));
    const kLoop = num(derived.kLoopCount, 8);
    const K = num(params.K, 512);
    const baseK = num(params.baseK, 64);
    const blockIdx = Number(step?.blockIdx || 0);
    const mIndex = step?.loop?.mIndex != null ? Number(step.loop.mIndex) : (blockIdx % mIter);
    const nIndex = step?.loop?.nIndex != null ? Number(step.loop.nIndex) : Math.floor(blockIdx / mIter);
    const kIndex = Number(step?.loop?.kIndex || 0);
    const stage = step?.stageId || '';
    const tone = stage === 'mmad' ? 'reduction' : stage === 'fixpipe' ? 'output' : 'input';
    const rowStart = mIndex * singleCoreM;
    const rowEnd = Math.min(M, rowStart + singleCoreM);
    const colStart = nIndex * singleCoreN;
    const colEnd = Math.min(N, colStart + singleCoreN);
    const tracksK = stage.includes('copy-in') || stage.includes('load-data') || stage === 'mmad';
    const blocks = cubeBufferBlocks(stage, kIndex);
    return {
      tensorViewport: {
        kind: 'matmul',
        layout: '2d',
        title: `C[M=${M}, N=${N}] 输出网格 · 每格 ${baseM}×${baseN} 元素`,
        axisLabels: ['N (列)', 'M (行)', 'K 累加'],
        grid: { rowTotal: M, colTotal: N, rowCell: baseM, colCell: baseN, rowLabel: 'M', colLabel: 'N', kTotal: K, kCell: baseK, kSteps: kLoop, depthLabel: 'K' },
        highlight: {
          row: [rowStart, rowEnd],
          col: [colStart, colEnd],
          tone,
          state: stage === 'mmad' ? 'accumulating' : 'selected',
          label: `C[M ${rowStart}:${rowEnd}, N ${colStart}:${colEnd}]`,
          sub: `block ${blockIdx} · singleCore 分区`,
        },
        progress: tracksK ? { label: 'K 累加', current: kIndex + 1, total: kLoop }
          : stage === 'fixpipe' ? { label: 'K 累加', current: kLoop, total: kLoop }
          : { label: 'K 累加', current: 0, total: kLoop },
        operationChips: cubeOps(stage),
      },
      onChipLens: { blocks },
      architectureFocus: {
        selectors: cubeSelectors(stage),
        routes: cubeRoutes(stage),
        bufferBlocks: blocks,
      },
    };
  }

  function deriveConvVisualState(step, trace) {
    const p = trace?.tiling?.params || {};
    const d = trace?.tiling?.derived || {};
    const stage = step?.stageId || '';
    const focusedIteration = Number.isInteger(state.instructionIterationFocus)
      ? state.instructionIterationFocus
      : null;
    const focusedIterationRange = state.instructionIterationRange;
    const isFocusedLoad = state.instructionOperationFocus === 'load-a2-b2';
    const isFocusedMte1MSync = state.instructionOperationFocus === 'mte1-m';
    const isFocusedMmad = state.instructionOperationFocus === 'mmad-accumulate';
    const isFocusedIterationOperation = isFocusedLoad || isFocusedMte1MSync || isFocusedMmad;
    const kIndex = isFocusedIterationOperation && focusedIteration != null
      ? focusedIteration
      : isFocusedIterationOperation && focusedIterationRange
        ? Number(focusedIterationRange[0])
        : Number(step?.loop?.kIndex || 0);
    const kLoops = num(d.kLoopCount, 9);
    const M = num(p.M, 64);
    const K = num(p.K, 144);
    const N = num(p.N, 32);
    const tileM = num(p.tileM, 16);
    const tileK = num(p.tileK, 16);
    const tileN = num(p.tileN, 16);
    const kRange = isFocusedIterationOperation
      ? [kIndex * tileK, Math.min(K, (kIndex + 1) * tileK)]
      : step?.loop?.kRange || [kIndex * tileK, Math.min(K, (kIndex + 1) * tileK)];
    const presentationStage = isFocusedMte1MSync ? 'sync-mte1-m' : stage;
    const finished = ['sync-m-fix', 'fixpipe-output'].includes(stage);
    const tracksK = ['load-k', 'mmad-init', 'loop-body-middle', 'loop-body-final'].includes(presentationStage);
    const kCurrent = finished ? kLoops
      : stage === 'k-loop' ? 1
      : tracksK ? Math.min(kLoops, kIndex + 1)
      : 0;
    const blocks = convBufferBlocks(presentationStage, kIndex);
    const scene = isFocusedLoad
      ? 'load3d'
      : isFocusedMte1MSync
        ? 'event'
      : isFocusedMmad
        ? 'mmad'
        : convSceneForStage(stage);
    const event = isFocusedMte1MSync
      ? (trace?.events || []).find((item) => item.eventType === 'MTE1_M') || null
      : (trace?.events || []).find((item) => (step?.eventDependencies || []).includes(item.id)) || null;
    const copyTransfer = selectedCopyInputTransfer(trace, step);
    return {
      tensorViewport: {
        kind: 'conv2d',
        layout: 'conv2d',
        title: `Y logical [1, ${p.co}, ${p.ho}, ${p.wo}] · GM ND [${M}, ${N}]`,
        axisLabels: ['Ho×Wo 输出位置', 'Co 输出通道', 'K=Ci×Kh×Kw 归约'],
        conv: {
          scene,
          n: num(p.batch, 1),
          ci: num(p.ci, 16),
          hi: num(p.hi, 8),
          wi: num(p.wi, 8),
          co: num(p.co, 32),
          ho: num(p.ho, 8),
          wo: num(p.wo, 8),
          kh: num(p.kh, 3),
          kw: num(p.kw, 3),
          strideH: num(p.strideH, 1),
          strideW: num(p.strideW, 1),
          padTop: Number(p.padTop ?? 1),
          padLeft: Number(p.padLeft ?? 1),
          M, N, K, tileM, tileN, tileK,
          kIndex, kRange, kCurrent, kLoops,
          kTileSelection: isFocusedIterationOperation && focusedIterationRange
            ? focusedIterationRange
            : [kIndex, kIndex],
          outputPosition: step?.loop?.representativeOutputPosition || [0, 0],
          snapshots: step?.tensorSnapshots || [],
          dataFlows: step?.dataFlows || [],
          event,
          copyTransfer,
        },
        highlight: {
          tone: (presentationStage.includes('mmad') || presentationStage.includes('loop-body')) ? 'reduction' : finished ? 'output' : 'input',
          label: scene === 'copy-in' && copyTransfer
            ? `MTE2 · ${copyTransfer.transferCount} transfers · ${copyTransfer.totalBytes} B · GM → L1`
            : convSceneLabel(scene, kIndex, tileM, tileN),
          sub: scene === 'copy-in' && copyTransfer
            ? `AIC${copyTransfer.partition.index} · OT${copyTransfer.partition.outputTile} · M${copyTransfer.partition.mTile}/N${copyTransfer.partition.nTile} · ${step?.evidenceKind || 'unknown'}`
            : `${step?.evidenceKind || 'unknown'} · logical order only`,
        },
        operationChips: convOperationChips(scene),
      },
      onChipLens: { blocks },
      architectureFocus: { selectors: convSelectors(presentationStage), routes: [], bufferBlocks: blocks },
    };
  }

  function convSceneForStage(stage) {
    if (stage.startsWith('host-') || stage === 'allocate') return 'overview';
    if (stage === 'copy-inputs') return 'copy-in';
    if (stage === 'bias-c1-c2') return 'bias-copy';
    if (stage === 'load-k') return 'load3d';
    if (stage === 'mmad-init') return 'mmad';
    if (stage === 'k-loop') return 'loop-group';
    if (stage === 'loop-body-middle' || stage === 'loop-body-final') return 'loop-body';
    if (stage === 'fixpipe-output') return 'epilogue';
    if (stage.startsWith('sync-')) return 'event';
    return 'overview';
  }

  function convSceneLabel(scene, kIndex, tileM, tileN) {
    if (scene === 'copy-in') return 'MTE2 · 3 transfers · 6720 B · GM → L1';
    if (scene === 'bias-copy') return 'Bias C1 → C2 / Bias Table';
    if (scene === 'load3d') return `Feature window → A2[M=${tileM}, K${kIndex}]`;
    if (scene === 'mmad') {
      return kIndex === 0
        ? 'A[Mi,K0] × B[K0,Nj] + Bias[Nj] → Acc0'
        : `A[Mi,K${kIndex}] × B[K${kIndex},Nj] + Acc${kIndex - 1} → Acc${kIndex}`;
    }
    if (scene === 'loop-group') return 'Iter 1～8：复用同步 → Load Kk → 就绪同步 → Mmad';
    if (scene === 'loop-body') return `完整 Iter ${kIndex} Loop Body：Load K${kIndex} → Mmad accumulate`;
    if (scene === 'epilogue') return 'CO1 FP32 NZ → ReLU → FP16 ND → GM';
    if (scene === 'event') return 'Producer → Event dependency → Consumer';
    return 'Conv2D tensors and local buffers';
  }

  function convOperationChips(scene) {
    const map = {
      overview: ['Shape', '16×16×16 tiling', 'blockDim=8'],
      'copy-in': ['MTE2', 'DataCopy', 'awaiting sync'],
      'bias-copy': ['MTE1', 'DataCopy', 'Bias Table'],
      load3d: ['LoadData3D', 'feature window', 'M×K'],
      'loop-group': ['Loop Group ×8', 'A2/B2 still K0', 'not hardware instruction'],
      'loop-body': ['M_MTE1', 'Load Kk', 'MTE1_M', 'Mmad'],
      event: ['SetFlag', 'WaitFlag', 'dependency'],
      epilogue: ['Fixpipe', 'ReLU', 'FP32→FP16'],
      'copy-out': ['Fixpipe', 'output tile', 'exact offset'],
    };
    return map[scene] || ['Mmad', 'Cube', 'K accumulate'];
  }

  function convBufferBlocks(stage, kIndex) {
    if (stage === 'allocate') {
      return [
        { core: 'mem950-aic', buffer: 'L1', label: 'A1 2048B · B1 4608B · C1 64B', state: 'allocated', tone: 'input', cellRange: [0, 29], sourceTile: 'exact address map' },
        { core: 'mem950-aic', buffer: 'L0C', label: 'CO1 1024B', state: 'allocated', tone: 'output', cellRange: [0, 11], sourceTile: '16×16 fp32' },
      ];
    }
    if (stage === 'copy-inputs' || stage === 'sync-mte2-mte1') {
      return [{ core: 'mem950-aic', buffer: 'L1', label: 'Feature / Weight / Bias', state: 'loaded', tone: 'input', cellRange: [0, 29], sourceTile: '2048B + 4608B + 64B', operation: 'DataCopy' }];
    }
    if (stage === 'bias-c1-c2') {
      return [{ core: 'mem950-aic', buffer: 'BT', label: 'Bias C2', state: 'loaded', tone: 'reduction', cellRange: [0, 3], sourceTile: '16×FP32 · 64B', operation: 'MTE1 / DataCopy' }];
    }
    if (stage === 'k-loop') {
      return [
        { core: 'mem950-aic', buffer: 'L0A', label: 'A2 · K0 reusable', state: 'read-complete', tone: 'input', cellRange: [0, 15], sourceTile: 'K1 not loaded yet', operation: 'Loop entry' },
        { core: 'mem950-aic', buffer: 'L0B', label: 'B2 · K0 reusable', state: 'read-complete', tone: 'input', cellRange: [0, 11], sourceTile: 'K1 not loaded yet', operation: 'Loop entry' },
      ];
    }
    if (stage === 'load-k' || stage === 'sync-mte1-m') {
      return [
        { core: 'mem950-aic', buffer: 'L0A', label: `A2 · K${kIndex}`, state: 'loaded', tone: 'input', cellRange: [0, 15], sourceTile: '16×16 · 512B', operation: 'LoadData3D' },
        { core: 'mem950-aic', buffer: 'L0B', label: `B2 · K${kIndex}`, state: 'loaded', tone: 'input', cellRange: [0, 11], sourceTile: '16×16 · 512B', operation: 'LoadData2D' },
      ];
    }
    if (stage === 'mmad-init' || stage === 'loop-body-middle' || stage === 'loop-body-final') {
      return [
        { core: 'mem950-aic', buffer: 'L0A', label: `A2 · K${kIndex}`, state: 'consumed', tone: 'input', cellRange: [0, 15], sourceTile: '16×16 · 512B', operation: 'Mmad' },
        { core: 'mem950-aic', buffer: 'L0B', label: `B2 · K${kIndex}`, state: 'consumed', tone: 'input', cellRange: [0, 11], sourceTile: '16×16 · 512B', operation: 'Mmad' },
        { core: 'mem950-aic', buffer: 'L0C', label: stage === 'mmad-init' ? 'CO1 + Bias' : stage === 'loop-body-final' ? 'CO1 · Acc8 final' : 'CO1 · Acc1～Acc7', state: 'accumulating', tone: 'reduction', cellRange: [0, 23], sourceTile: 'M[0:16],Co[0:16] · 1024B', operation: 'Mmad' },
      ];
    }
    if (['sync-m-fix', 'fixpipe-output'].includes(stage)) {
      return [{ core: 'mem950-aic', buffer: 'L0C', label: 'CO1 output', state: 'committed', tone: 'output', cellRange: [0, 23], sourceTile: '16×16 · 1024B', operation: stage === 'fixpipe-output' ? 'ReLU + Cast + GM write' : 'M_FIX' }];
    }
    return [];
  }

  function convSelectors(stage) {
    if (stage === 'copy-inputs') return ['[data-mem950-node="rail:GM"]', '#mem950-aic [data-aic-node="buffer:L1"]'];
    if (stage === 'load-k' || stage === 'sync-mte1-m') return ['#mem950-aic [data-aic-node="buffer:L1"]', '#mem950-aic [data-aic-node="buffer:L0A"]', '#mem950-aic [data-aic-node="buffer:L0B"]'];
    if (stage === 'k-loop') return ['#mem950-aic [data-aic-node="buffer:L0A"]', '#mem950-aic [data-aic-node="buffer:L0B"]'];
    if (stage === 'mmad-init' || stage === 'loop-body-middle' || stage === 'loop-body-final') return ['#mem950-aic [data-aic-node="buffer:L1"]', '#mem950-aic [data-aic-node="buffer:L0A"]', '#mem950-aic [data-aic-node="buffer:L0B"]', '#mem950-aic [data-aic-node="cube:CUBE"]', '#mem950-aic [data-aic-node="buffer:L0C"]'];
    if (stage === 'fixpipe-output') return ['#mem950-aic [data-aic-node="buffer:L0C"]', '[data-mem950-node="rail:GM"]'];
    return [];
  }

  function deriveFusionVisualState(step, trace) {
    const params = trace?.tiling?.params || {};
    const M = num(params.M, 512);
    const N = num(params.N, 1024);
    const baseM = num(params.baseM, 128);
    const baseN = num(params.baseN, 256);
    const K = num(params.K, 512);
    const baseK = num(params.baseK, 64);
    const kSteps = Math.max(1, Math.round(K / baseK));
    const singleCoreM = num(params.singleCoreM, 256);
    const singleCoreN = num(params.singleCoreN, 512);
    const aivHalf = Number(step?.blockIdx || 0) % 2;
    const stage = step?.stageId || '';
    const isAiv = stage.includes('aiv');
    const isSync = stage.includes('sync');
    const colStart = 0;
    const colEnd = singleCoreN;
    const half = Math.floor(singleCoreM / 2);
    let row = [0, singleCoreM];
    let tone = 'reduction';
    let state = 'active';
    let label = `AIC 生产 C[M 0:${singleCoreM}, N 0:${singleCoreN}]`;
    let sub = 'Cube block 0';
    if (isAiv) {
      row = aivHalf === 0 ? [0, half] : [half, singleCoreM];
      tone = 'fusion';
      label = `C[M ${row[0]}:${row[1]}, N ${colStart}:${colEnd}]`;
      sub = aivHalf === 0 ? 'AIV0 · 上半 M' : 'AIV1 · 下半 M';
    } else if (isSync) {
      tone = 'output';
      state = 'ready';
      label = `C[M 0:${singleCoreM}, N 0:${singleCoreN}] ready`;
      sub = 'CrossCoreSetFlag';
    }
    const blocks = fusionBufferBlocks(stage, aivHalf);
    return {
      tensorViewport: {
        kind: 'fusion',
        layout: '2d',
        title: `C[M=${M}, N=${N}] · AIC 生产、AIV 上/下半消费 · 每格 ${baseM}×${baseN} 元素`,
        axisLabels: ['N (列)', 'M (行)', 'K'],
        grid: { rowTotal: M, colTotal: N, rowCell: baseM, colCell: baseN, rowLabel: 'M', colLabel: 'N', kTotal: K, kCell: baseK, kSteps, depthLabel: 'K' },
        highlight: { row, col: [colStart, colEnd], tone, state, label, sub },
        progress: null,
        operationChips: ['C tile', 'CrossCoreFlag', 'LeakyRelu'],
      },
      onChipLens: { blocks },
      architectureFocus: {
        selectors: fusionSelectors(stage),
        routes: fusionRoutes(stage),
        bufferBlocks: blocks,
      },
    };
  }

  function vectorBufferBlocks(stage, blockIdx, progress) {
    const sourceBase = `block${blockIdx},progress${progress}`;
    if (stage.includes('copy-out')) {
      return [{ core: 'mem950-aiv1', buffer: 'UB', label: 'zLocal', state: 'committed', tone: 'output', cellRange: [38, 53], sourceTile: `z[${sourceBase},:]`, operation: 'CopyOut' }];
    }
    if (stage.includes('compute')) {
      return [
        { core: 'mem950-aiv1', buffer: 'UB', label: 'xLocal', state: 'dequeued', tone: 'input', cellRange: [0, 15], sourceTile: `x[${sourceBase},:]`, operation: 'DeQue' },
        { core: 'mem950-aiv1', buffer: 'UB', label: 'yLocal', state: 'dequeued', tone: 'input', cellRange: [19, 34], sourceTile: `y[${sourceBase},:]`, operation: 'DeQue' },
        { core: 'mem950-aiv1', buffer: 'UB', label: 'zLocal', state: 'enqueued', tone: 'output', cellRange: [38, 53], sourceTile: `z[${sourceBase},:]`, operation: 'Add' },
      ];
    }
    if (stage.includes('copy-in')) {
      return [
        { core: 'mem950-aiv1', buffer: 'UB', label: 'xLocal', state: 'enqueued', tone: 'input', cellRange: [0, 15], sourceTile: `x[${sourceBase},:]`, operation: 'CopyIn' },
        { core: 'mem950-aiv1', buffer: 'UB', label: 'yLocal', state: 'enqueued', tone: 'input', cellRange: [19, 34], sourceTile: `y[${sourceBase},:]`, operation: 'CopyIn' },
      ];
    }
    return [];
  }

  function cubeBufferBlocks(stage, kIndex) {
    if (stage.includes('copy-in-a')) {
      return [
        { core: 'mem950-aic', buffer: 'L1', label: 'A1 tile', state: 'loaded', tone: 'input', cellRange: [0, 19], sourceTile: `A[m0,k${kIndex}]`, operation: 'DataCopy' },
        { core: 'mem950-aic', buffer: 'L0A', label: 'A2 reserve', state: 'allocated', tone: 'input', cellRange: [0, 9], sourceTile: `A[m0,k${kIndex}]` },
      ];
    }
    if (stage.includes('copy-in-b')) {
      return [
        { core: 'mem950-aic', buffer: 'L1', label: 'B1 tile', state: 'loaded', tone: 'input', cellRange: [30, 49], sourceTile: `B[k${kIndex},n0]`, operation: 'DataCopy' },
        { core: 'mem950-aic', buffer: 'L0B', label: 'B2 reserve', state: 'allocated', tone: 'input', cellRange: [0, 9], sourceTile: `B[k${kIndex},n0]` },
      ];
    }
    if (stage.includes('load-data')) {
      return [
        { core: 'mem950-aic', buffer: 'L0A', label: 'A2 tile', state: 'loaded', tone: 'input', cellRange: [0, 15], sourceTile: `A2[k${kIndex}]`, operation: 'LoadDataA' },
        { core: 'mem950-aic', buffer: 'L0B', label: 'B2 tile', state: 'loaded', tone: 'input', cellRange: [0, 15], sourceTile: `B2[k${kIndex}]`, operation: 'LoadDataB' },
      ];
    }
    if (stage.includes('mmad')) {
      return [
        { core: 'mem950-aic', buffer: 'L0A', label: 'A2 tile', state: 'loaded', tone: 'input', cellRange: [0, 15], sourceTile: `A2[k${kIndex}]`, operation: 'Mmad' },
        { core: 'mem950-aic', buffer: 'L0B', label: 'B2 tile', state: 'loaded', tone: 'input', cellRange: [0, 15], sourceTile: `B2[k${kIndex}]`, operation: 'Mmad' },
        { core: 'mem950-aic', buffer: 'L0C', label: 'C partial', state: 'accumulating', tone: 'accumulator', cellRange: [0, 23], sourceTile: `C[m0,n0,k${kIndex}]`, operation: 'Mmad' },
      ];
    }
    if (stage.includes('fixpipe')) {
      return [{ core: 'mem950-aic', buffer: 'L0C', label: 'C output', state: 'committed', tone: 'output', cellRange: [0, 23], sourceTile: 'C[m0,n0]', operation: 'Fixpipe' }];
    }
    return [];
  }

  function fusionBufferBlocks(stage, aivHalf) {
    if (stage.includes('aic-matmul')) {
      return [
        { core: 'mem950-aic', buffer: 'L0A', label: 'A tile', state: 'loaded', tone: 'input', cellRange: [0, 15], sourceTile: 'A[m0,k*]' },
        { core: 'mem950-aic', buffer: 'L0B', label: 'B tile', state: 'loaded', tone: 'input', cellRange: [0, 15], sourceTile: 'B[k*,n0]' },
        { core: 'mem950-aic', buffer: 'L0C', label: 'C partial', state: 'accumulating', tone: 'accumulator', cellRange: [0, 23], sourceTile: 'C[m0,n0]' },
      ];
    }
    if (stage.includes('sync')) {
      return [{ core: 'mem950-aic', buffer: 'L0C', label: 'C ready', state: 'committed', tone: 'output', cellRange: [0, 23], sourceTile: 'C[m0,n0]', operation: 'CrossCoreSetFlag' }];
    }
    if (stage.includes('aiv-leakyrelu')) {
      return [
        { core: `mem950-aiv${aivHalf + 1}`, buffer: 'UB', label: 'epilogue tile', state: 'enqueued', tone: 'output', cellRange: [0, 31], sourceTile: `C half ${aivHalf}`, operation: 'LeakyRelu' },
        { core: 'mem950-aic', buffer: 'L0C', label: 'C source tile', state: 'committed', tone: 'output', cellRange: [0, 23], sourceTile: 'C[m0,n0]' },
      ];
    }
    return [];
  }

  function vectorSelectors(stage) {
    if (stage.includes('copy-out')) return ['#mem950-aiv1 [data-aiv-node="buffer:UB"]', '[data-mem950-node="rail:GM"]'];
    if (stage.includes('compute')) return ['#mem950-aiv1 [data-aiv-node="buffer:UB"]', '#mem950-aiv1 [data-aiv-node="exec:SIMD"]', '#mem950-aiv1 [data-aiv-node="vector:Vector"]'];
    if (stage.includes('copy-in')) return ['[data-mem950-node="rail:GM"]', '#mem950-aiv1 [data-aiv-node="buffer:UB"]'];
    return [];
  }

  function vectorRoutes(stage) {
    if (stage.includes('copy-out')) return ['aiv1-ub-to-gm'];
    if (stage.includes('copy-in')) return ['gm-to-aiv1-ub'];
    return [];
  }

  function cubeSelectors(stage) {
    if (stage.includes('copy-in') || stage.includes('load-data')) {
      return ['[data-mem950-node="rail:GM"]', '#mem950-aic [data-aic-node="buffer:L1"]', '#mem950-aic [data-aic-node="buffer:L0A"]', '#mem950-aic [data-aic-node="buffer:L0B"]'];
    }
    if (stage.includes('mmad')) {
      return ['#mem950-aic [data-aic-node="buffer:L0A"]', '#mem950-aic [data-aic-node="buffer:L0B"]', '#mem950-aic [data-aic-node="cube:CUBE"]', '#mem950-aic [data-aic-node="buffer:L0C"]'];
    }
    if (stage.includes('fixpipe')) return ['#mem950-aic [data-aic-node="buffer:L0C"]', '[data-mem950-node="rail:GM"]'];
    return [];
  }

  function cubeRoutes(stage) {
    if (stage.includes('copy-in') || stage.includes('load-data')) return ['gm-to-aic-l0a', 'gm-to-aic-l0b'];
    return [];
  }

  function fusionSelectors(stage) {
    if (stage.includes('aiv-leakyrelu')) {
      return ['#mem950-aic [data-aic-node="buffer:L0C"]', '#mem950-aiv1 [data-aiv-node="buffer:UB"]', '#mem950-aiv2 [data-aiv-node="buffer:UB"]', '#mem950-aiv1 [data-aiv-node="vector:Vector"]', '#mem950-aiv2 [data-aiv-node="vector:Vector"]'];
    }
    if (stage.includes('sync')) {
      return ['#mem950-aic [data-aic-node="buffer:L0C"]', '#mem950-aiv1 [data-aiv-node="buffer:UB"]', '#mem950-aiv2 [data-aiv-node="buffer:UB"]'];
    }
    if (stage.includes('aic-matmul')) {
      return ['#mem950-aic [data-aic-node="buffer:L0A"]', '#mem950-aic [data-aic-node="buffer:L0B"]', '#mem950-aic [data-aic-node="cube:CUBE"]', '#mem950-aic [data-aic-node="buffer:L0C"]'];
    }
    return [];
  }

  function fusionRoutes(stage) {
    if (stage.includes('aiv-leakyrelu') || stage.includes('sync')) return ['aic-to-aiv1', 'aiv2-to-aic'];
    return [];
  }

  function cubeOps(stage) {
    if (stage.includes('copy-in')) return ['DataCopy', 'ND->NZ'];
    if (stage.includes('load-data')) return ['LoadData', 'L1->L0'];
    if (stage.includes('mmad')) return ['Mmad', 'K accumulate'];
    if (stage.includes('fixpipe')) return ['Fixpipe', 'CopyOut'];
    return ['GetBlockIdx', 'GM offset'];
  }

  function renderTensorViewport(trace) {
    if (!trace || !els.tensorCanvas) return;
    const step = currentStep(trace);
    const visual = visualStateForStep(trace, step).tensorViewport;
    const snapshot = selectedTensorSnapshot(step);
    const snapshotShape = tensorSnapshotShape(snapshot);
    const isConvLoad = visual.layout === 'conv2d' && visual.conv?.scene === 'load3d';
    const isConvEvent = visual.layout === 'conv2d' && visual.conv?.scene === 'event';
    const isConvCopyInputs = trace.operator?.kind === 'conv2d-cube' && step?.stageId === 'copy-inputs';
    const isBiasC1C2 = trace.operator?.kind === 'conv2d-cube' && step?.stageId === 'bias-c1-c2';
    const useMemoryAllocationMap = trace.operator?.kind === 'conv2d-cube'
      && step?.stageId === 'allocate'
      && !!els.memoryAllocationView;
    const useConvOverview = trace.operator?.kind === 'conv2d-cube'
      && step?.stageId === 'host-shape'
      && !!window.PtoTensorVolumeCanvas
      && !!window.PtoMatrixCanvas
      && !!els.convTensorOverview;
    const useHostTilingPatterns = trace.operator?.kind === 'conv2d-cube'
      && step?.stageId === 'host-tiling'
      && !!window.PtoTensorVolumeCanvas
      && !!window.PtoMatrixCanvas
      && !!els.hostTilingView
      && !!els.hostTilingSourceCanvas
      && !!els.hostTilingCubeCanvas;
    const useHostLaunchPatterns = trace.operator?.kind === 'conv2d-cube'
      && step?.stageId === 'host-launch'
      && !!window.PtoMatrixCanvas
      && !!els.hostLaunchView
      && !!els.hostLaunchACanvas
      && !!els.hostLaunchBCanvas
      && !!els.hostLaunchCCanvas;
    const useLoadDataPatterns = isConvLoad
      && state.tensorTabKey !== 'buffer:weight:b2'
      && !!window.PtoTensorVolumeCanvas
      && !!window.PtoMatrixCanvas
      && !!els.convLoadDataView
      && !!els.fmapA1VolumeCanvas
      && !!els.a2LogicalMatrixCanvas;
    const useMmadMatrixPattern = visual.layout === 'conv2d'
      && visual.conv?.scene === 'mmad'
      && !!window.PtoMatrixCanvas
      && !!els.convMmadMatrixView
      && !!els.mmadA2Canvas
      && !!els.mmadB2Canvas
      && !!els.addendMatrixCanvas
      && !!els.mmadCo1Canvas;
    const useCopyInputPatterns = isConvCopyInputs
      && !!window.PtoTensorVolumeCanvas
      && !!window.PtoMatrixCanvas
      && !!els.copyInputPatternView
      && !!els.copyInputSourceCanvas
      && !!els.copyInputDestinationCanvas;
    const useBiasC1C2MatrixPattern = isBiasC1C2
      && !!window.PtoMatrixCanvas
      && !!els.biasC1C2View
      && !!els.biasC1Canvas
      && !!els.biasC2Canvas;
    const useFixpipeOutputPattern = trace.operator?.kind === 'conv2d-cube'
      && step?.stageId === 'fixpipe-output'
      && !!window.PtoMatrixCanvas
      && !!els.fixpipeOutputView
      && !!els.fixpipeAccumCanvas
      && !!els.fixpipeOutputCanvas;
    const useVolumePattern = !useCopyInputPatterns
      && !useLoadDataPatterns
      && !isConvEvent
      && snapshotShape.length >= 3
      && !!window.PtoTensorVolumeCanvas
      && !!els.tensorVolumeCanvas;
    const useMatrixPattern = !useCopyInputPatterns
      && !useLoadDataPatterns
      && !isConvEvent
      && !!window.PtoMatrixCanvas
      && !!els.tensorMatrixCanvas
      && (
        snapshotShape.length === 2
        || visual.layout === '2d'
        || (isConvLoad && state.tensorTabKey === 'buffer:weight:b2')
      );

    const renderer = useMemoryAllocationMap
      ? 'allocate-memory'
      : useConvOverview
      ? 'overview'
      : useHostTilingPatterns
        ? 'host-tiling'
      : useHostLaunchPatterns
        ? 'host-launch'
      : useCopyInputPatterns
        ? 'copy-input-pattern'
      : useBiasC1C2MatrixPattern
        ? 'bias-c1-c2-matrix'
      : useFixpipeOutputPattern
        ? 'fixpipe-output'
      : useLoadDataPatterns
      ? 'load-data'
      : useMmadMatrixPattern
        ? 'mmad-matrix'
      : useVolumePattern
        ? 'volume'
        : useMatrixPattern
          ? 'matrix'
          : 'legacy';
    setActiveTensorRenderer(renderer);
    [els.zoomOut, els.zoomIn].forEach((button) => {
      if (button) button.disabled = renderer === 'volume'
        || renderer === 'load-data'
        || renderer === 'overview'
        || renderer === 'host-tiling'
        || renderer === 'host-launch';
    });
    if (els.fitView) els.fitView.disabled = renderer === 'volume';

    const tip = tensorViewportTip(visual);
    if (els.tensorStage) els.tensorStage.title = tip;
    if (els.viewportInfo) els.viewportInfo.title = tip;
    if (renderer === 'allocate-memory') {
      renderMemoryAllocationMap(trace);
      if (els.tensorFallback) els.tensorFallback.hidden = true;
      return;
    }
    if (renderer === 'overview') {
      renderConvTensorOverview(trace);
      if (els.tensorFallback) els.tensorFallback.hidden = true;
      return;
    }
    if (renderer === 'host-tiling') {
      renderHostTilingMatrixView(trace);
      if (els.tensorFallback) els.tensorFallback.hidden = true;
      return;
    }
    if (renderer === 'host-launch') {
      renderHostLaunchMatrixView(trace);
      if (els.tensorFallback) els.tensorFallback.hidden = true;
      return;
    }
    if (renderer === 'copy-input-pattern') {
      renderCopyInputPatternView(trace, step, visual);
      if (els.tensorFallback) els.tensorFallback.hidden = true;
      return;
    }
    if (renderer === 'bias-c1-c2-matrix') {
      renderBiasC1C2MatrixView(trace, step);
      if (els.tensorFallback) els.tensorFallback.hidden = true;
      return;
    }
    if (renderer === 'fixpipe-output') {
      renderFixpipeOutputView(trace, step);
      if (els.tensorFallback) els.tensorFallback.hidden = true;
      return;
    }
    if (renderer === 'load-data') {
      renderConvLoadDataView(visual.conv);
      if (els.tensorFallback) els.tensorFallback.hidden = true;
      return;
    }
    if (renderer === 'mmad-matrix') {
      renderConvMmadMatrixPattern(visual);
      if (els.tensorFallback) els.tensorFallback.hidden = true;
      return;
    }
    if (renderer === 'volume') {
      renderTensorVolumePattern(trace, snapshot);
      if (els.tensorFallback) els.tensorFallback.hidden = true;
      return;
    }
    if (renderer === 'matrix') {
      renderTensorMatrixPattern(trace, visual, snapshot);
      if (els.tensorFallback) els.tensorFallback.hidden = true;
      return;
    }

    const canvas = els.tensorCanvas;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(520, Math.floor(rect.width || canvas.clientWidth || 760));
    const height = Math.max(360, Math.floor(rect.height || canvas.clientHeight || 480));
    const ctx = fitCanvas(canvas, width, height);
    drawTensorScene(ctx, width, height, visual);
    if (els.tensorFallback) els.tensorFallback.hidden = true;
  }

  function selectedTensorSnapshot(step) {
    const snapshots = step?.tensorSnapshots || [];
    if (!snapshots.length) return null;
    if (state.tensorTabKey) {
      return snapshots.find((snapshot) => snapshot.tensorId === state.tensorTabKey) || snapshots[0];
    }
    return snapshots[0];
  }

  function tensorSnapshotShape(snapshot) {
    const shape = snapshot?.physicalShape || snapshot?.logicalShape;
    return Array.isArray(shape)
      ? shape.map(Number).filter((value) => Number.isFinite(value) && value > 0)
      : [];
  }

  function setActiveTensorRenderer(renderer) {
    state.activeTensorRenderer = renderer;
    const patternVisible = renderer === 'matrix' || renderer === 'volume';
    if (els.tileLens) {
      if (renderer === 'copy-input-pattern' && els.copyInputLensMount) {
        if (els.tileLens.parentElement !== els.copyInputLensMount) els.copyInputLensMount.appendChild(els.tileLens);
      } else if (els.tensorStage && els.tileLens.parentElement !== els.tensorStage) {
        els.tensorStage.insertBefore(els.tileLens, els.tensorFallback || null);
      }
    }
    if (els.convTensorOverview) els.convTensorOverview.hidden = renderer !== 'overview';
    if (els.memoryAllocationView) els.memoryAllocationView.hidden = renderer !== 'allocate-memory';
    if (els.hostTilingView) els.hostTilingView.hidden = renderer !== 'host-tiling';
    if (els.hostLaunchView) els.hostLaunchView.hidden = renderer !== 'host-launch';
    if (els.copyInputPatternView) els.copyInputPatternView.hidden = renderer !== 'copy-input-pattern';
    if (els.biasC1C2View) els.biasC1C2View.hidden = renderer !== 'bias-c1-c2-matrix';
    if (els.fixpipeOutputView) els.fixpipeOutputView.hidden = renderer !== 'fixpipe-output';
    if (els.tensorPatternView) els.tensorPatternView.hidden = !patternVisible;
    if (els.convMmadMatrixView) els.convMmadMatrixView.hidden = renderer !== 'mmad-matrix';
    if (els.tensorVolumeCanvas) els.tensorVolumeCanvas.hidden = renderer !== 'volume';
    if (els.tensorMatrixHost) els.tensorMatrixHost.hidden = renderer !== 'matrix';
    if (els.convLoadDataView) els.convLoadDataView.hidden = renderer !== 'load-data';
    if (els.tensorCanvas) els.tensorCanvas.hidden = renderer !== 'legacy';
    if (els.tileLens) els.tileLens.hidden = renderer !== 'legacy' && renderer !== 'copy-input-pattern';
  }

  const ALLOCATION_MEMORY_MODEL = {
    'buffer:feature:a1': {
      position: 'A1 / L1', format: 'NC1HWC0', logicalIdentity: 'complete Feature',
      filledBy: 'DataCopy from featureGm_', consumedBy: 'LoadData3D', lifetime: 'staged feature map',
    },
    'buffer:weight:b1': {
      position: 'B1 / L1', format: 'NZ', logicalIdentity: 'B[:,Nj]',
      filledBy: 'DataCopy ND → NZ', consumedBy: 'LoadData2D', lifetime: 'one N tile',
    },
    'buffer:bias:c1': {
      position: 'C1 / L1', format: 'linear', logicalIdentity: 'D[Nj]',
      filledBy: 'DataCopy from biasGm_', consumedBy: 'DataCopy to biasC2', lifetime: 'L1 bias staging',
    },
    'buffer:feature:a2': {
      position: 'A2 / L0A', format: 'ZZ', logicalIdentity: 'A[Mi,Kk]',
      filledBy: 'LoadData3D', consumedBy: 'Mmad', lifetime: 'reused and overwritten across K iterations',
    },
    'buffer:weight:b2': {
      position: 'B2 / L0B', format: 'ZN', logicalIdentity: 'B[Kk,Nj]',
      filledBy: 'LoadData2D', consumedBy: 'Mmad', lifetime: 'reused and overwritten across K iterations',
    },
    'buffer:bias:c2': {
      position: 'C2 / Bias Table', format: 'linear', logicalIdentity: 'D[Nj]',
      filledBy: 'DataCopy from biasC1', consumedBy: 'Iter 0 Mmad', lifetime: 'read by the first Mmad only',
    },
    'buffer:accum:co1': {
      position: 'CO1 / L0C', format: 'NZ', logicalIdentity: 'C[Mi,Nj]',
      filledBy: 'Mmad', consumedBy: 'Fixpipe', lifetime: 'Acc0 → Acc8', writeLabel: 'Written by',
    },
  };

  function allocationTensor(trace, id) {
    const buffer = (trace.buffers || []).find((item) => item.id === id);
    const model = ALLOCATION_MEMORY_MODEL[id];
    if (!buffer || !model) return null;
    const start = Number(buffer.addressBytes) || 0;
    const size = Number(buffer.allocatedBytes) || 0;
    return {
      ...buffer,
      ...model,
      start,
      end: start + size,
      size,
      dtypeLabel: String(buffer.dtype || '').toUpperCase(),
      shapeLabel: `[${(buffer.logicalShape || []).join(',')}]`,
      alignmentLabel: Number(buffer.alignmentBytes) ? `${Number(buffer.alignmentBytes)} B` : 'Not specified',
    };
  }

  function allocationBlock(tensor, compact = false) {
    if (!tensor) return '';
    const selected = state.allocatedTensorId === tensor.id;
    const compactClass = compact ? ' avz-memory-block--compact' : '';
    return `<button class="avz-memory-block${compactClass}${selected ? ' is-selected' : ''}" type="button" data-allocation-tensor="${escapeHtml(tensor.id)}" aria-pressed="${selected}" aria-label="${escapeHtml(`${tensor.name}, ${tensor.position}, [${tensor.start},${tensor.end})`)}">
      <strong>${escapeHtml(tensor.name)}</strong>
      ${compact ? '' : `<span>${escapeHtml(tensor.shapeLabel)}</span><span>${escapeHtml(`${tensor.dtypeLabel} · ${tensor.format}`)}</span>`}
      <span>${escapeHtml(`${tensor.size} B`)}</span>
      ${compact ? '' : `<code>${escapeHtml(`[${tensor.start},${tensor.end})`)}</code>`}
    </button>`;
  }

  function singleAllocationLane(title, tensor) {
    return `<section class="avz-memory-lane" aria-label="${escapeHtml(title)} independent address space">
      <header><strong>${escapeHtml(title)}</strong><span>${tensor.size} B</span></header>
      <div class="avz-memory-axis"><span>0</span><span>${tensor.end}</span></div>
      <div class="avz-memory-track">${allocationBlock(tensor)}</div>
    </section>`;
  }

  function allocationDetail(tensor) {
    if (!tensor) {
      return `<div class="avz-memory-detail__empty"><strong>Tensor details</strong><span>选择任一 Tensor 块，查看完整地址属性与后续生命周期。</span></div>`;
    }
    const rows = [
      ['Variable', tensor.name], ['Position', tensor.position], ['Address', `[${tensor.start},${tensor.end})`],
      ['Size', `${tensor.size} B`], ['Shape', tensor.shapeLabel], ['dtype', tensor.dtypeLabel],
      ['format', tensor.format], ['Logical identity', tensor.logicalIdentity], ['Alignment', tensor.alignmentLabel],
      [tensor.writeLabel || 'Filled by', tensor.filledBy], ['Consumed by', tensor.consumedBy], ['Lifetime', tensor.lifetime],
    ];
    return `<header><strong>${escapeHtml(tensor.name)}</strong><span>${escapeHtml(tensor.position)}</span></header>
      <table><tbody>${rows.map(([key, value]) => `<tr><th>${escapeHtml(key)}</th><td><code>${escapeHtml(value)}</code></td></tr>`).join('')}</tbody></table>`;
  }

  function renderMemoryAllocationMap(trace) {
    const ids = Object.keys(ALLOCATION_MEMORY_MODEL);
    const tensors = Object.fromEntries(ids.map((id) => [id, allocationTensor(trace, id)]));
    const selected = tensors[state.allocatedTensorId] || null;
    const fmapA1 = tensors['buffer:feature:a1'];
    const weightB1 = tensors['buffer:weight:b1'];
    const biasC1 = tensors['buffer:bias:c1'];
    if (!fmapA1 || !weightB1 || !biasC1) return;
    els.memoryAllocationView.innerHTML = `
      <div class="avz-memory-allocation__intro">
        <div><strong>Local Memory Map</strong><span>当前 AI Core 的本地内存地图</span></div>
        <p>Allocate Memory 只建立 LocalTensor 视图并绑定硬件 Buffer 与本地地址；不搬运数据，也不执行计算。</p>
        <small>GM input/output addresses were bound in the previous stage. This stage only creates LocalTensor views.</small>
      </div>
      <div class="avz-memory-lanes">
        <section class="avz-memory-lane avz-memory-lane--l1" aria-label="L1 shared address space">
          <header><strong>L1 Buffer</strong><span>Used: 6720 B · [0,6720)</span></header>
          <div class="avz-memory-axis avz-memory-axis--l1"><span>0</span><span>2048</span><span>6656</span><span>6720</span></div>
          <div class="avz-memory-track avz-memory-track--l1">
            <div>${allocationBlock(fmapA1)}</div>
            <div>${allocationBlock(weightB1)}</div>
            <div>${allocationBlock(biasC1, true)}</div>
          </div>
        </section>
        ${singleAllocationLane('L0A Buffer', tensors['buffer:feature:a2'])}
        ${singleAllocationLane('L0B Buffer', tensors['buffer:weight:b2'])}
        ${singleAllocationLane('Bias Table', tensors['buffer:bias:c2'])}
        ${singleAllocationLane('L0C Buffer', tensors['buffer:accum:co1'])}
      </div>
      <section class="avz-address-spaces" aria-label="Independent address spaces explanation">
        <div class="avz-address-spaces__diagram">
          ${[['L0A', 'fmapA2'], ['L0B', 'weightB2'], ['Bias Table', 'biasC2'], ['L0C', 'accumCo1']].map(([lane, name]) => `<div><strong>${lane}</strong><span>0</span><i></i><code>${name}</code></div>`).join('')}
        </div>
        <p><code>fmapA2</code>、<code>weightB2</code>、<code>biasC2</code>、<code>accumCo1</code> 都从地址 0 开始，但它们属于不同的物理 Buffer，因此不存在地址覆盖或冲突。</p>
      </section>
      <section class="avz-memory-detail" aria-live="polite">${allocationDetail(selected)}</section>
      <footer class="avz-memory-legend">
        <span><i></i> LocalTensor view exists · memory range assigned · data not loaded</span>
        <span>Tensor 块宽经过可读性调整；地址标签和字节数表示真实内存范围。</span>
        <span>所有地址区间均为左闭右开 <code>[start, end)</code>。</span>
      </footer>
      <div class="avz-memory-tooltip" id="memoryAllocationTooltip" role="tooltip" hidden></div>`;

    els.memoryAllocationView.querySelectorAll('[data-allocation-tensor]').forEach((button) => {
      button.addEventListener('pointerenter', showAllocationTooltip);
      button.addEventListener('pointermove', positionAllocationTooltip);
      button.addEventListener('pointerleave', hideAllocationTooltip);
      button.addEventListener('focus', showAllocationTooltip);
      button.addEventListener('blur', hideAllocationTooltip);
    });
  }

  function handleAllocatedTensorClick(event) {
    const button = event.target.closest('[data-allocation-tensor]');
    if (!button) return;
    state.allocatedTensorId = button.dataset.allocationTensor;
    renderMemoryAllocationMap(currentTrace());
  }

  function allocationTooltipMarkup(tensor) {
    const rows = [
      ['Position', tensor.position], ['Address', `[${tensor.start},${tensor.end})`], ['Size', `${tensor.size} B`],
      ['Shape', tensor.shapeLabel], ['dtype', tensor.dtypeLabel], ['format', tensor.format],
      ['Logical tile', tensor.logicalIdentity], ['Alignment', tensor.alignmentLabel],
      [tensor.writeLabel || 'Filled by', tensor.filledBy], ['Consumed by', tensor.consumedBy], ['Lifetime', tensor.lifetime],
    ];
    return `<strong>${escapeHtml(tensor.name)}</strong>${rows.map(([key, value]) => `<span><b>${escapeHtml(key)}</b><code>${escapeHtml(value)}</code></span>`).join('')}`;
  }

  function showAllocationTooltip(event) {
    const trace = currentTrace();
    const tensor = allocationTensor(trace, event.currentTarget.dataset.allocationTensor);
    const tooltip = byId('memoryAllocationTooltip');
    if (!tensor || !tooltip) return;
    tooltip.innerHTML = allocationTooltipMarkup(tensor);
    tooltip.hidden = false;
    positionAllocationTooltip(event);
  }

  function positionAllocationTooltip(event) {
    const tooltip = byId('memoryAllocationTooltip');
    if (!tooltip || tooltip.hidden) return;
    const rootRect = els.memoryAllocationView.getBoundingClientRect();
    const targetRect = event.currentTarget.getBoundingClientRect();
    const x = 'clientX' in event ? event.clientX - rootRect.left + 12 : targetRect.left - rootRect.left + 12;
    const y = 'clientY' in event ? event.clientY - rootRect.top + 12 : targetRect.bottom - rootRect.top + 8;
    tooltip.style.left = `${Math.min(x, Math.max(8, rootRect.width - tooltip.offsetWidth - 12))}px`;
    tooltip.style.top = `${Math.min(y, Math.max(8, rootRect.height - tooltip.offsetHeight - 12))}px`;
  }

  function hideAllocationTooltip() {
    const tooltip = byId('memoryAllocationTooltip');
    if (tooltip) tooltip.hidden = true;
  }

  function createFixpipeAggregateScene(rows, columns, blockRows, blockColumns, activeRange = null) {
    const cells = [];
    const aggregateRows = Math.max(1, Math.min(rows, num(blockRows, rows)));
    const aggregateColumns = Math.max(1, Math.min(columns, num(blockColumns, columns)));
    for (let row = 0; row < rows; row += aggregateRows) {
      const rowSpan = Math.min(aggregateRows, rows - row);
      for (let column = 0; column < columns; column += aggregateColumns) {
        const columnSpan = Math.min(aggregateColumns, columns - column);
        const selected = !!activeRange
          && row < activeRange.rowEnd
          && row + rowSpan > activeRange.rowStart
          && column < activeRange.columnEnd
          && column + columnSpan > activeRange.columnStart;
        cells.push({
          id: `fixpipe-${row}-${column}`,
          row,
          column,
          rowSpan,
          columnSpan,
          style: 'aggregate',
          tone: 'neutral',
          states: selected ? ['selected'] : [],
          summary: {
            rows: rowSpan,
            columns: columnSpan,
            count: rowSpan * columnSpan,
            intensity: selected ? 1 : 0.42,
          },
        });
      }
    }
    return {
      extent: { rows, columns },
      axes: { rows: 'M', columns: 'Co' },
      cells,
    };
  }

  function renderFixpipeOutputView(trace, step) {
    const params = trace?.tiling?.params || {};
    const partition = convCorePartition(trace);
    const rows = num(params.M, 64);
    const columns = num(params.N, 32);
    const tileRows = num(params.tileM, 16);
    const tileColumns = num(params.tileN, 16);
    const rowStart = partition.mTile * tileRows;
    const columnStart = partition.nTile * tileColumns;
    const rowEnd = Math.min(rows, rowStart + tileRows);
    const columnEnd = Math.min(columns, columnStart + tileColumns);
    const elementOffset = rowStart * columns + columnStart;
    const byteOffset = elementOffset * 2;
    const coreLabel = `AIC${partition.index} · OT${partition.outputTile}`;

    if (els.fixpipeOutputSummary) els.fixpipeOutputSummary.textContent = 'accumCo1 → outputGm';
    if (els.fixpipeOutputContext) {
      els.fixpipeOutputContext.textContent = `${coreLabel} · M${partition.mTile}/N${partition.nTile}`;
    }
    if (els.fixpipeAccumTitle) els.fixpipeAccumTitle.textContent = `accumCo1 · AIC${partition.index}`;
    if (els.fixpipeAddressCore) els.fixpipeAddressCore.textContent = coreLabel;
    if (els.fixpipeAddress) {
      els.fixpipeAddress.textContent = `outputGm + ${byteOffset} B · M[${rowStart}:${rowEnd}] · Co[${columnStart}:${columnEnd}] · row stride ${columns} half`;
    }

    const fixtures = {
      accum: {
        canvas: els.fixpipeAccumCanvas,
        scene: createFixpipeAggregateScene(tileRows, tileColumns, tileRows, tileColumns),
        options: {
          ariaLabel: `AIC ${partition.index} accumCo1 one aggregate cell representing ${tileRows} by ${tileColumns} values`,
          showAxes: true,
          showGrid: true,
          interactive: true,
          showTooltip: true,
          autoFit: true,
          padding: { top: 30, right: 20, bottom: 36, left: 42 },
        },
      },
      output: {
        canvas: els.fixpipeOutputCanvas,
        scene: createFixpipeAggregateScene(rows, columns, tileRows, tileColumns, {
          rowStart,
          rowEnd,
          columnStart,
          columnEnd,
        }),
        options: {
          ariaLabel: `outputGm ${rows} by ${columns} values grouped into ${Math.ceil(rows / tileRows) * Math.ceil(columns / tileColumns)} aggregate cells; ${coreLabel} destination M ${rowStart} to ${rowEnd}, Co ${columnStart} to ${columnEnd} selected`,
          showAxes: true,
          showGrid: true,
          interactive: true,
          showTooltip: true,
          autoFit: true,
          padding: { top: 30, right: 20, bottom: 36, left: 42 },
        },
      },
    };

    Object.entries(fixtures).forEach(([key, fixture]) => {
      const controller = state.fixpipeOutputControllers[key];
      if (controller) {
        controller.update(fixture.scene, { ...fixture.options, preserveView: false });
        controller.fit();
      } else {
        state.fixpipeOutputControllers[key] = window.PtoMatrixCanvas.render(
          fixture.canvas,
          fixture.scene,
          fixture.options
        );
      }
    });
  }

  function createBiasTileMatrixScene(prefix, stateName, columns = 16) {
    const resolvedColumns = Math.max(1, Number(columns) || 16);
    return {
      extent: { rows: 1, columns: resolvedColumns },
      axes: { rows: 'Bias tile', columns: `N0:N${resolvedColumns}` },
      cells: Array.from({ length: resolvedColumns }, (_, column) => ({
        id: `${prefix}-${column}`,
        row: 0,
        column,
        label: `N${column}`,
        tone: 'neutral',
        style: 'value',
        states: stateName ? [stateName] : [],
      })),
    };
  }

  function renderBiasC1C2MatrixView(trace, step) {
    if (!window.PtoMatrixCanvas || !step) return;
    const flow = step.dataFlows?.[0] || {};
    const snapshot = step.tensorSnapshots?.[0] || {};
    const source = (trace?.buffers || []).find((item) => item.id === 'buffer:bias:c1') || {};
    const destination = (trace?.buffers || []).find((item) => item.id === 'buffer:bias:c2') || {};
    const elements = Number(snapshot.validElements) || Number(destination.allocatedElements) || 16;
    const bytes = Number(flow.bytes) || Number(snapshot.validBytes) || Number(destination.allocatedBytes) || 64;
    const dtype = String(snapshot.dtype || destination.dtype || source.dtype || 'fp32').toUpperCase();
    const sourceAddress = Number.isFinite(Number(source.addressBytes)) ? `@${Number(source.addressBytes)}` : '@unknown';
    const destinationAddress = Number.isFinite(Number(destination.addressBytes)) ? `@${Number(destination.addressBytes)}` : '@unknown';

    if (els.biasC1C2Summary) els.biasC1C2Summary.textContent = 'Bias C1 → C2 / Bias Table';
    if (els.biasC1C2Context) els.biasC1C2Context.textContent = `MTE1 · ${elements} × ${dtype} · ${formatBytes(bytes)}`;
    if (els.biasC1C2Engine) els.biasC1C2Engine.textContent = flow.transferEngine || 'MTE1 / DataCopy';
    if (els.biasC1Title) els.biasC1Title.textContent = `${source.name || 'biasC1'} · ${flow.from || source.location || 'C1 / L1'}`;
    if (els.biasC1Shape) els.biasC1Shape.textContent = `linear [1,${elements}]`;
    if (els.biasC1Meta) els.biasC1Meta.textContent = `${dtype} · ${sourceAddress} · ${formatBytes(bytes)} · readable after MTE2_MTE1`;
    if (els.biasC2Title) els.biasC2Title.textContent = `${destination.name || 'biasC2'} · ${flow.to || destination.location || 'C2 / Bias Table'}`;
    if (els.biasC2Shape) els.biasC2Shape.textContent = `${snapshot.physicalLayout || 'linear Bias Table'} [1,${elements}]`;
    if (els.biasC2Meta) els.biasC2Meta.textContent = `${dtype} · ${destinationAddress} · ${formatBytes(bytes)} · ready for first Mmad`;

    const options = {
      showAxes: true,
      showGrid: true,
      interactive: true,
      showTooltip: true,
      autoFit: true,
      minZoom: 0.45,
      padding: { top: 34, right: 24, bottom: 40, left: 50 },
    };
    const fixtures = {
      source: {
        canvas: els.biasC1Canvas,
        scene: createBiasTileMatrixScene('bias-c1', 'current', elements),
        ariaLabel: `C1 source Bias tile, ${elements} ${dtype} values, readable by MTE1`,
      },
      destination: {
        canvas: els.biasC2Canvas,
        scene: createBiasTileMatrixScene('bias-c2', 'written', elements),
        ariaLabel: `C2 Bias Table destination, ${elements} ${dtype} values, ready for first Mmad`,
      },
    };

    Object.entries(fixtures).forEach(([key, fixture]) => {
      if (!fixture.canvas) return;
      const controller = state.biasC1C2Controllers[key];
      const nextOptions = { ...options, ariaLabel: fixture.ariaLabel };
      if (controller) {
        controller.update(fixture.scene, { ...nextOptions, preserveView: false });
        controller.fit();
      } else {
        state.biasC1C2Controllers[key] = window.PtoMatrixCanvas.render(
          fixture.canvas,
          fixture.scene,
          nextOptions
        );
      }
    });
  }

  function renderCopyInputPatternView(trace, step, visual) {
    const transfer = visual?.conv?.copyTransfer || selectedCopyInputTransfer(trace, step);
    if (!transfer) return;
    const { flow, source, destination, snapshot, partition, coreSource } = transfer;
    const sourceRole = String(source?.role || '');
    const tone = sourceRole === 'input'
      ? 'neutral'
      : sourceRole === 'bias'
        ? 'reduction'
        : 'input';
    const shape = tensorSnapshotShape(snapshot);
    const sourceLayout = String(source?.physicalLayout || source?.logicalLayout || 'layout unknown').split(' ')[0];
    const destinationLayout = snapshot?.physicalLayout || destination?.physicalLayout || 'layout unknown';
    const dtype = String(snapshot?.dtype || destination?.dtype || source?.dtype || 'dtype unknown').toUpperCase();
    const bytes = Number(flow?.bytes) || Number(snapshot?.validBytes) || 0;
    const allocatedBytes = Number(destination?.allocatedBytes) || Number(snapshot?.allocatedBytes) || bytes;
    const address = Number.isFinite(Number(destination?.addressBytes)) ? Number(destination.addressBytes) : null;
    const alignment = Number.isFinite(Number(destination?.alignmentBytes)) ? Number(destination.alignmentBytes) : null;
    const transformation = sourceRole === 'bias'
      ? `${sourceLayout} → ${destinationLayout} · Bias staging`
      : flow?.status === 'ND-to-NZ'
        ? 'ND → NZ'
        : sourceLayout === destinationLayout
          ? 'layout unchanged'
          : `${sourceLayout} → ${destinationLayout}`;
    const useVolume = sourceRole === 'input' && shape.length >= 3;
    const kind = useVolume ? 'volume' : 'matrix';
    const weightMatrixAggregation = sourceRole === 'weight'
      ? {
          forceAggregate: true,
          blockRows: Math.max(1, num(trace?.tiling?.params?.tileK, 16)),
          blockColumns: Math.max(1, num(trace?.tiling?.params?.tileN, 16)),
        }
      : null;
    const sourcePhysicalShape = Array.isArray(source?.physicalShape)
      ? source.physicalShape.map(Number)
      : shape;
    const sourceMatrixAggregation = weightMatrixAggregation
      ? {
          ...weightMatrixAggregation,
          rows: Math.max(1, sourcePhysicalShape.at(-2) || shape.at(-2) || 1),
          columns: Math.max(1, sourcePhysicalShape.at(-1) || shape.at(-1) || 1),
          selection: {
            rowStart: 0,
            rowEnd: Math.max(1, sourcePhysicalShape.at(-2) || shape.at(-2) || 1),
            columnStart: (partition?.nTile || 0) * Math.max(1, num(trace?.tiling?.params?.tileN, 16)),
            columnEnd: Math.min(
              Math.max(1, sourcePhysicalShape.at(-1) || shape.at(-1) || 1),
              ((partition?.nTile || 0) + 1) * Math.max(1, num(trace?.tiling?.params?.tileN, 16)),
            ),
          },
        }
      : null;
    const sourceScene = useVolume
      ? copyInputVolumeScene(snapshot, tone, 'source')
      : copyInputMatrixScene(snapshot, tone, 'source', sourceMatrixAggregation);
    const destinationScene = useVolume
      ? copyInputVolumeScene(snapshot, tone, 'destination')
      : copyInputMatrixScene(snapshot, tone, 'destination', weightMatrixAggregation);

    if (els.copyInputSummary) {
      els.copyInputSummary.textContent = `MTE2 · ${transfer.transferCount} transfers · ${transfer.totalBytes} B · GM → L1`;
    }
    if (els.copyInputContext) {
      els.copyInputContext.textContent = `AIC${partition?.index ?? 0} · OT${partition?.outputTile ?? 0} · M${partition?.mTile ?? 0}/N${partition?.nTile ?? 0}`;
    }
    if (els.copyInputSourceTitle) els.copyInputSourceTitle.textContent = coreSource?.label || source?.name || flow?.from || 'GM source';
    if (els.copyInputSourceShape) {
      const displayShape = sourceRole === 'weight' ? sourcePhysicalShape : shape;
      const sliceLabel = sourceRole === 'weight' ? `selected ${coreSource?.slice}` : coreSource?.slice || 'current slice';
      els.copyInputSourceShape.textContent = `${sourceLayout} ${formatShape(displayShape)} · ${sliceLabel}`;
    }
    if (els.copyInputSourceMeta) {
      els.copyInputSourceMeta.textContent = `${dtype} · ${formatBytes(bytes)} · GM + ${coreSource?.gmOffsetBytes || 0} B`;
    }
    if (els.copyInputEngine) els.copyInputEngine.textContent = flow?.transferEngine || 'MTE2 / DataCopy';
    if (els.copyInputTransformation) els.copyInputTransformation.textContent = transformation;
    if (els.copyInputDestinationTitle) {
      els.copyInputDestinationTitle.textContent = `${destination?.name || flow?.to || 'L1 buffer'} · ${destination?.location || flow?.to || 'L1'}`;
    }
    if (els.copyInputDestinationShape) {
      els.copyInputDestinationShape.textContent = `${destinationLayout} ${formatShape(shape)}`;
    }
    if (els.copyInputDestinationMeta) {
      const addressText = address == null ? '@unknown' : `@${address}`;
      const alignmentText = alignment == null ? 'alignment unknown' : `align ${alignment} B`;
      els.copyInputDestinationMeta.textContent = `${addressText} · ${formatBytes(bytes)} / ${formatBytes(allocatedBytes)} · ${alignmentText}`;
    }

    renderCopyInputPatternCanvas('source', kind, sourceScene, {
      ariaLabel: `${coreSource?.label || source?.name || 'GM source'} ${sourceLayout} ${formatShape(sourceRole === 'weight' ? sourcePhysicalShape : shape)}${sourceRole === 'weight' ? `, selected ${coreSource?.slice}` : ''}`,
    });
    renderCopyInputPatternCanvas('destination', kind, destinationScene, {
      ariaLabel: `${destination?.name || 'L1 destination'} ${destinationLayout} ${formatShape(shape)}`,
    });
  }

  function copyInputVolumeScene(snapshot, tone, side) {
    const shape = tensorSnapshotShape(snapshot);
    const isNc1hwc0 = String(snapshot?.logicalLayout || snapshot?.physicalLayout || '').toUpperCase().includes('NC1HWC0');
    const columns = Math.max(1, isNc1hwc0 ? shape.at(-2) : shape.at(-1));
    const rows = Math.max(1, isNc1hwc0 ? shape.at(-3) : shape.at(-2));
    const depth = Math.max(1, isNc1hwc0 ? shape.at(-1) : shape.at(-3));
    const voxels = [];
    for (let z = 0; z < depth; z += 1) {
      for (let row = 0; row < rows; row += 1) {
        for (let column = 0; column < columns; column += 1) {
          voxels.push({
            id: `copy-${side}-${column}-${row}-${z}`,
            column,
            row,
            depth: z,
            tone,
            state: 'base',
          });
        }
      }
    }
    return {
      extent: { columns, rows, depth },
      axes: isNc1hwc0
        ? { columns: 'W', rows: 'H', depth: 'C0' }
        : { columns: 'column', rows: 'row', depth: 'depth' },
      voxels,
    };
  }

  function copyInputMatrixScene(snapshot, tone, side, aggregation = null) {
    const shape = tensorSnapshotShape(snapshot);
    const rows = Math.max(1, aggregation?.rows || (shape.length === 1 ? 1 : shape.at(-2)));
    const columns = Math.max(1, aggregation?.columns || shape.at(-1) || shape[0] || 1);
    const rowSpan = Math.max(
      1,
      aggregation?.blockRows || Math.ceil(rows / 12),
    );
    const columnSpan = Math.max(
      1,
      aggregation?.blockColumns || Math.ceil(columns / 16),
    );
    const cells = [];
    for (let row = 0; row < rows; row += rowSpan) {
      for (let column = 0; column < columns; column += columnSpan) {
        const resolvedRowSpan = Math.min(rowSpan, rows - row);
        const resolvedColumnSpan = Math.min(columnSpan, columns - column);
        const aggregate = aggregation?.forceAggregate
          || resolvedRowSpan > 1
          || resolvedColumnSpan > 1;
        const selection = aggregation?.selection;
        const selected = !selection
          || (
            row < selection.rowEnd
            && row + resolvedRowSpan > selection.rowStart
            && column < selection.columnEnd
            && column + resolvedColumnSpan > selection.columnStart
          );
        cells.push({
          id: `copy-${side}-${row}-${column}`,
          row,
          column,
          rowSpan: resolvedRowSpan,
          columnSpan: resolvedColumnSpan,
          tone,
          style: aggregate ? 'aggregate' : 'value',
          states: side === 'source'
            ? selected ? ['current'] : []
            : ['written'],
          summary: aggregate ? {
            rows: resolvedRowSpan,
            columns: resolvedColumnSpan,
            count: resolvedRowSpan * resolvedColumnSpan,
            intensity: side === 'source' ? 0.48 : 0.72,
          } : undefined,
        });
      }
    }
    return {
      extent: { rows, columns },
      axes: rows === 1
        ? { rows: 'Bias slice', columns: 'N tile' }
        : { rows: 'K', columns: 'N tile' },
      cells,
    };
  }

  function renderCopyInputPatternCanvas(slot, kind, scene, customOptions) {
    const canvas = slot === 'source' ? els.copyInputSourceCanvas : els.copyInputDestinationCanvas;
    if (!canvas) return;
    const controllerKey = slot;
    const kindKey = `${slot}Kind`;
    if (state.copyInputControllers[kindKey] !== kind) {
      state.copyInputControllers[controllerKey]?.destroy?.();
      state.copyInputControllers[controllerKey] = null;
      state.copyInputControllers[kindKey] = kind;
    }
    canvas.classList.toggle('pto-tensor-volume-canvas', kind === 'volume');
    canvas.classList.toggle('pto-matrix-canvas', kind === 'matrix');
    canvas.parentElement?.classList.toggle('pto-matrix-canvas-host', kind === 'matrix');
    const options = kind === 'volume'
      ? {
          ...customOptions,
          padding: { top: 34, right: 30, bottom: 36, left: 46 },
          showAxes: true,
          autoLabelDensity: true,
        }
      : {
          ...customOptions,
          showAxes: true,
          showGrid: true,
          interactive: true,
          showTooltip: true,
          autoFit: true,
          padding: { top: 30, right: 24, bottom: 38, left: 48 },
        };
    const controller = state.copyInputControllers[controllerKey];
    const api = kind === 'volume' ? window.PtoTensorVolumeCanvas : window.PtoMatrixCanvas;
    if (controller) {
      controller.update(scene, { ...options, preserveView: false });
      kind === 'volume' ? controller.resize() : controller.fit();
    } else {
      state.copyInputControllers[controllerKey] = api.render(canvas, scene, options);
    }
  }

  function createMmadMatrixScene(rows, columns, axes, prefix) {
    const cells = [];
    for (let row = 0; row < rows; row += 1) {
      for (let column = 0; column < columns; column += 1) {
        cells.push({
          id: `${prefix}-${row}-${column}`,
          row,
          column,
        });
      }
    }
    return {
      extent: { rows, columns },
      axes,
      cells,
    };
  }

  function createMmadBiasBroadcastScene(rows, columns) {
    const cells = [];
    for (let row = 0; row < rows; row += 1) {
      for (let column = 0; column < columns; column += 1) {
        const isSourceRow = row === 0;
        cells.push({
          id: `mmad-bias-${row}-${column}`,
          row,
          column,
          ...(isSourceRow ? {} : { style: 'broadcast' }),
        });
      }
    }
    return {
      extent: { rows, columns },
      axes: { rows: 'M broadcast', columns: 'N' },
      cells,
    };
  }

  function renderConvMmadMatrixPattern(visual) {
    const c = visual?.conv || {};
    const kIndex = Math.max(0, Number(c.kIndex) || 0);
    const kCurrent = Math.max(1, Number(c.kCurrent) || kIndex + 1);
    const kLoops = Math.max(1, Number(c.kLoops) || 1);
    const tileM = Math.max(1, Number(c.tileM) || 16);
    const tileK = Math.max(1, Number(c.tileK) || 16);
    const tileN = Math.max(1, Number(c.tileN) || 16);
    const result = `Acc${kIndex}`;
    const equation = kIndex === 0
      ? 'A[Mi,K0] × B[K0,Nj] + Bias[Nj] → Acc0'
      : `A[Mi,K${kIndex}] × B[K${kIndex},Nj] + Acc${kIndex - 1} → ${result}`;

    if (els.mmadEquation) els.mmadEquation.textContent = equation;
    if (els.mmadEvidence) els.mmadEvidence.textContent = visual?.highlight?.sub || 'confirmed · logical order only';
    if (els.mmadProgress) els.mmadProgress.textContent = `K ${kCurrent}/${kLoops}`;
    if (els.mmadA2Title) els.mmadA2Title.textContent = `A[Mi,K${kIndex}] · A2`;
    if (els.mmadA2Meta) els.mmadA2Meta.textContent = `[${tileM},${tileK}] · FP16 · L0A`;
    if (els.mmadB2Title) els.mmadB2Title.textContent = `B[K${kIndex},Nj] · B2`;
    if (els.mmadB2Meta) els.mmadB2Meta.textContent = `[${tileK},${tileN}] · FP16 · L0B`;
    if (els.mmadAddend) els.mmadAddend.textContent = '+';
    if (els.addendMatrixTitle) {
      els.addendMatrixTitle.textContent = kIndex === 0
        ? 'Bias[Nj] broadcast'
        : `Acc${kIndex - 1} · CO1`;
    }
    if (els.addendMatrixMeta) {
      els.addendMatrixMeta.textContent = kIndex === 0
        ? `[${tileM},${tileN}] · FP32 · C2 / Bias Table`
        : `[${tileM},${tileN}] · FP32 · L0C · previous partial sum`;
    }
    if (els.mmadCo1Title) els.mmadCo1Title.textContent = `${result} · CO1`;
    if (els.mmadCo1Meta) els.mmadCo1Meta.textContent = `[${tileM},${tileN}] · FP32 · L0C · K ${kCurrent}/${kLoops}`;
    if (els.mmadBiasStatus) {
      els.mmadBiasStatus.textContent = kIndex === 0
        ? 'Bias C1 → C2 confirmed · I0 only'
        : `Acc${kIndex - 1} is read from CO1 · Bias is not added again`;
      els.mmadBiasStatus.classList.toggle('is-confirmed', kIndex === 0);
    }

    const matrixOptions = {
      showAxes: false,
      showGrid: true,
      interactive: false,
      showTooltip: false,
      autoFit: true,
      padding: { top: 8, right: 8, bottom: 8, left: 8 },
    };
    const fixtures = {
      a2: {
        canvas: els.mmadA2Canvas,
        scene: createMmadMatrixScene(
          tileM,
          tileK,
          { rows: 'M', columns: 'K' },
          'mmad-a2'
        ),
        ariaLabel: `A2 current matrix operand, M ${tileM} by K ${tileK}, K iteration ${kIndex}`,
      },
      b2: {
        canvas: els.mmadB2Canvas,
        scene: createMmadMatrixScene(
          tileK,
          tileN,
          { rows: 'K', columns: 'N' },
          'mmad-b2'
        ),
        ariaLabel: `B2 current matrix operand, K ${tileK} by N ${tileN}, K iteration ${kIndex}`,
      },
      addend: {
        canvas: els.addendMatrixCanvas,
        scene: kIndex === 0
          ? createMmadBiasBroadcastScene(tileM, tileN)
          : createMmadMatrixScene(
              tileM,
              tileN,
              { rows: 'M', columns: 'N' },
              `mmad-acc${kIndex - 1}`
            ),
        ariaLabel: kIndex === 0
          ? `Bias vector with ${tileN} values logically broadcast across ${tileM} M rows for iteration zero`
          : `Previous accumulator Acc${kIndex - 1} in CO1, M ${tileM} by N ${tileN}, used as the addend for iteration ${kIndex}`,
      },
      co1: {
        canvas: els.mmadCo1Canvas,
        scene: createMmadMatrixScene(
          tileM,
          tileN,
          { rows: 'M', columns: 'N' },
          'mmad-co1'
        ),
        ariaLabel: `CO1 accumulator matrix, M ${tileM} by N ${tileN}, K progress ${kCurrent} of ${kLoops}`,
      },
    };

    Object.entries(fixtures).forEach(([key, fixture]) => {
      const options = { ...matrixOptions, ariaLabel: fixture.ariaLabel };
      const controller = state.mmadMatrixControllers[key];
      if (controller) {
        controller.update(fixture.scene, { ...options, preserveView: false });
        controller.resize();
      } else {
        state.mmadMatrixControllers[key] = window.PtoMatrixCanvas.render(
          fixture.canvas,
          fixture.scene,
          options
        );
      }
    });
  }

  function createVolumeScene({ columns, rows, depth, tone, axes, prefix }) {
    const voxels = [];
    for (let z = 0; z < depth; z += 1) {
      for (let row = 0; row < rows; row += 1) {
        for (let column = 0; column < columns; column += 1) {
          voxels.push({
            id: `${prefix}-${column}-${row}-${z}`,
            column,
            row,
            depth: z,
            tone,
            state: 'base',
          });
        }
      }
    }
    return {
      extent: { columns, rows, depth },
      axes,
      voxels,
    };
  }

  function createOverviewMatrixScene({ rows, columns, rowSpan, columnSpan, tone, axes, prefix }) {
    const cells = [];
    for (let row = 0; row < rows; row += rowSpan) {
      for (let column = 0; column < columns; column += columnSpan) {
        const resolvedRowSpan = Math.min(rowSpan, rows - row);
        const resolvedColumnSpan = Math.min(columnSpan, columns - column);
        const aggregate = resolvedRowSpan > 1 || resolvedColumnSpan > 1;
        cells.push({
          id: `${prefix}-${row}-${column}`,
          row,
          column,
          rowSpan: resolvedRowSpan,
          columnSpan: resolvedColumnSpan,
          tone,
          style: aggregate ? 'aggregate' : 'value',
          summary: aggregate ? {
            rows: resolvedRowSpan,
            columns: resolvedColumnSpan,
            count: resolvedRowSpan * resolvedColumnSpan,
            intensity: 0.62,
          } : undefined,
        });
      }
    }
    return {
      extent: { rows, columns },
      axes,
      cells,
    };
  }

  function createHostTilingFeatureVolume({
    ci, hi, wi, kh, kw, padTop, padLeft, M, tileM, wo, strideH, strideW,
  }) {
    // Reuse the exact Load Data voxel model. Host Tiling has already fixed
    // padList, so its logical read domain must include the same virtual PAD
    // voxels even though those voxels are not physically stored in GM or A1.
    return convFmapA1VolumeScene({
      ci,
      hi,
      wi,
      kh,
      kw,
      padTop,
      padLeft,
      M,
      tileM,
      wo,
      strideH,
      strideW,
    }, 0);
  }

  function createHostTilingWeightVolume({ ci, kh, kw }) {
    const scene = createVolumeScene({
      columns: kw,
      rows: kh,
      depth: ci,
      tone: 'neutral',
      axes: { columns: `Kw=${kw}`, rows: `Kh=${kh}`, depth: `Ci=${ci}` },
      prefix: 'host-tiling-w-volume',
    });
    return {
      ...scene,
      voxels: scene.voxels.map((voxel) => ({
        ...voxel,
        tone: voxel.depth === 0 ? 'input' : 'neutral',
        state: voxel.depth === 0 ? 'window' : 'base',
      })),
    };
  }

  function createHostTilingOutputVolume({ co, ho, wo, tileM, tileN }) {
    const scene = createVolumeScene({
      columns: wo,
      rows: ho,
      depth: co,
      tone: 'neutral',
      axes: { columns: `Wo=${wo}`, rows: `Ho=${ho}`, depth: `Co=${co}` },
      prefix: 'host-tiling-y-volume',
    });
    return {
      ...scene,
      voxels: scene.voxels.map((voxel) => ({
        ...voxel,
        tone: 'neutral',
        state: voxel.depth === 0 ? 'window' : 'base',
      })),
    };
  }

  function renderHostTilingMatrixView(trace) {
    const params = trace?.tiling?.params || {};
    const ci = num(params.ci, 16);
    const hi = num(params.hi, 8);
    const wi = num(params.wi, 8);
    const co = num(params.co, 32);
    const ho = num(params.ho, 8);
    const wo = num(params.wo, 8);
    const kh = num(params.kh, 3);
    const kw = num(params.kw, 3);
    const M = num(params.M, ho * wo);
    const K = num(params.K, ci * kh * kw);
    const N = num(params.N, co);
    const tileM = num(params.tileM, 16);
    const tileK = num(params.tileK, 16);
    const tileN = num(params.tileN, 16);
    const strideH = num(params.strideH, 1);
    const strideW = num(params.strideW, 1);
    const padTop = Math.max(0, Number(params.padTop ?? 1));
    const padRight = Math.max(0, Number(params.padRight ?? padTop));
    const padBottom = Math.max(0, Number(params.padBottom ?? padTop));
    const padLeft = Math.max(0, Number(params.padLeft ?? 1));
    const key = state.tensorTabKey || 'host-tiling:feature';

    const configs = {
      'host-tiling:feature': {
        equation: 'Logical Input X → Cube A[M,K]',
        tileCount: `${Math.ceil(M / tileM)}×${Math.ceil(K / tileK)} tiles`,
        sourceTitle: 'Feature X · logical padded view',
        sourceMeta: `NCHW [1,${ci},${hi},${wi}] · FP16 · padList=[${padLeft},${padRight},${padTop},${padBottom}]`,
        sourceScene: createHostTilingFeatureVolume({
          ci, hi, wi, kh, kw, padTop, padLeft, M, tileM, wo, strideH, strideW,
        }),
        transform: 'logical M/K mapping',
        cubeTitle: `Logical Cube A [M=${M}, K=${K}]`,
        cubeMeta: `tileM=${tileM} · tileK=${tileK} · ${Math.ceil(M / tileM)}×${Math.ceil(K / tileK)} tiles`,
        cubeScene: createOverviewMatrixScene({
          rows: M,
          columns: K,
          rowSpan: tileM,
          columnSpan: tileK,
          tone: 'input',
          axes: { rows: `M=${M}`, columns: `K=${K}` },
          prefix: 'host-tiling-a',
        }),
        formula: `M=Ho×Wo=${ho}×${wo}=${M} · K=Ci×Kh×Kw=${ci}×${kh}×${kw}=${K} · padList fixed by Host Tiling`,
        sourceAria: `Feature X logical padded read domain for NCHW 1 by ${ci} by ${hi} by ${wi}; blue shows valid samples and orange shows virtual padding fixed by Host Tiling`,
        cubeAria: `Cube A logical matrix, M ${M} by K ${K}, divided into ${tileM} by ${tileK} tiles`,
      },
      'host-tiling:weight': {
        equation: 'Logical Weight W → Cube B[K,N]',
        tileCount: `${Math.ceil(K / tileK)}×${Math.ceil(N / tileN)} tiles`,
        sourceTitle: 'Logical Weight W · GM',
        sourceMeta: `OIHW [${co},${ci},${kh},${kw}] · ${co} filters × [${ci},${kh},${kw}] · FP16`,
        sourceScene: createHostTilingWeightVolume({ ci, kh, kw }),
        transform: 'logical K/N mapping',
        cubeTitle: `Logical Cube B [K=${K}, N=${N}]`,
        cubeMeta: `tileK=${tileK} · tileN=${tileN} · ${Math.ceil(K / tileK)}×${Math.ceil(N / tileN)} tiles`,
        cubeScene: createOverviewMatrixScene({
          rows: K,
          columns: N,
          rowSpan: tileK,
          columnSpan: tileN,
          tone: 'input',
          axes: { rows: `K=${K}`, columns: `N=Co=${N}` },
          prefix: 'host-tiling-b',
        }),
        formula: `K=Ci×Kh×Kw=${K} · N=Co=${N}`,
        sourceAria: `Weight W representative filter volume, Ci ${ci} by Kh ${kh} by Kw ${kw}; total output channels ${co}`,
        cubeAria: `Cube B logical matrix, K ${K} by N ${N}, divided into ${tileK} by ${tileN} tiles`,
      },
      'host-tiling:output': {
        equation: 'Logical Output Y → Cube C[M,N]',
        tileCount: `${Math.ceil(M / tileM)}×${Math.ceil(N / tileN)} tiles`,
        sourceTitle: 'Logical Output Y · GM',
        sourceMeta: `NCHW [1,${co},${ho},${wo}] · FP16 · representative [tileM,tileN] region`,
        sourceScene: createHostTilingOutputVolume({ co, ho, wo, tileM, tileN }),
        transform: 'logical M/N mapping',
        cubeTitle: `Logical Cube C [M=${M}, N=${N}]`,
        cubeMeta: `tileM=${tileM} · tileN=${tileN} · ${Math.ceil(M / tileM)}×${Math.ceil(N / tileN)} tiles`,
        cubeScene: createOverviewMatrixScene({
          rows: M,
          columns: N,
          rowSpan: tileM,
          columnSpan: tileN,
          tone: 'neutral',
          axes: { rows: `M=${M}`, columns: `N=Co=${N}` },
          prefix: 'host-tiling-c',
        }),
        formula: `M=Ho×Wo=${M} · N=Co=${N}`,
        sourceAria: `Output Y tensor volume, NCHW 1 by ${co} by ${ho} by ${wo}, with one representative Cube output tile`,
        cubeAria: `Cube C logical matrix, M ${M} by N ${N}, divided into ${tileM} by ${tileN} tiles`,
      },
    };
    const config = configs[key] || configs['host-tiling:feature'];

    if (els.hostTilingEquation) els.hostTilingEquation.textContent = config.equation;
    if (els.hostTilingEvidence) {
      els.hostTilingEvidence.textContent = 'planning only · global logical mapping and tile counts';
    }
    if (els.hostTilingTileCount) els.hostTilingTileCount.textContent = config.tileCount;
    if (els.hostTilingSourceTitle) els.hostTilingSourceTitle.textContent = config.sourceTitle;
    if (els.hostTilingSourceMeta) els.hostTilingSourceMeta.textContent = config.sourceMeta;
    if (els.hostTilingWeightCount) {
      const showWeightCount = key === 'host-tiling:weight';
      els.hostTilingWeightCount.hidden = !showWeightCount;
      els.hostTilingWeightCount.setAttribute(
        'aria-label',
        showWeightCount ? `Co equals ${co} output channels` : 'Output channel count'
      );
    }
    if (els.hostTilingWeightCountValue) {
      els.hostTilingWeightCountValue.textContent = `Co=${co}`;
    }
    if (els.hostTilingTransform) els.hostTilingTransform.textContent = config.transform;
    if (els.hostTilingCubeTitle) els.hostTilingCubeTitle.textContent = config.cubeTitle;
    if (els.hostTilingCubeMeta) els.hostTilingCubeMeta.textContent = config.cubeMeta;
    if (els.hostTilingFormula) els.hostTilingFormula.textContent = config.formula;

    const volumeOptions = {
      showAxes: true,
      autoLabelDensity: true,
      padding: { top: 24, right: 24, bottom: 42, left: 48 },
      ariaLabel: config.sourceAria,
    };
    const sourceController = state.hostTilingControllers.source;
    if (sourceController) {
      sourceController.update(config.sourceScene, volumeOptions);
      sourceController.resize();
    } else {
      state.hostTilingControllers.source = window.PtoTensorVolumeCanvas.render(
        els.hostTilingSourceCanvas,
        config.sourceScene,
        volumeOptions
      );
      state.hostTilingControllers.source.resize();
    }

    const matrixOptions = {
      showAxes: true,
      showGrid: true,
      interactive: true,
      showTooltip: true,
      autoFit: true,
      minZoom: 0.015,
      padding: { top: 28, right: 24, bottom: 38, left: 56 },
      ariaLabel: config.cubeAria,
    };
    const cubeController = state.hostTilingControllers.cube;
    if (cubeController) {
      cubeController.update(config.cubeScene, matrixOptions);
      cubeController.fit();
    } else {
      state.hostTilingControllers.cube = window.PtoMatrixCanvas.render(
        els.hostTilingCubeCanvas,
        config.cubeScene,
        matrixOptions
      );
      state.hostTilingControllers.cube.fit();
    }
  }

  function renderHostLaunchMatrixView(trace) {
    const params = trace?.tiling?.params || {};
    const derived = trace?.tiling?.derived || {};
    const M = num(params.M, 64);
    const K = num(params.K, 144);
    const N = num(params.N, 32);
    const tileM = num(params.tileM, 16);
    const tileK = num(params.tileK, 16);
    const tileN = num(params.tileN, 16);
    const mTiles = num(derived.mTileCount, Math.ceil(M / tileM));
    const kTiles = num(derived.kLoopCount, Math.ceil(K / tileK));
    const nTiles = num(derived.nTileCount, Math.ceil(N / tileN));
    const outputTileCount = num(derived.outputTileCount, mTiles * nTiles);
    const blockDim = num(trace?.launch?.numBlocks, outputTileCount);
    const activePartition = convCorePartition(trace);
    const activeBlockIdx = activePartition.index;
    const activeMTile = activePartition.mTile;
    const activeNTile = activePartition.nTile;
    const selectSceneCells = (scene, predicate) => ({
      ...scene,
      cells: scene.cells.map((cell) => ({
        ...cell,
        states: predicate(cell) ? ['selected'] : [],
      })),
    });

    if (els.hostLaunchEquation) {
      els.hostLaunchEquation.textContent = `A[M=${M},K=${K}] × B[K=${K},N=${N}] → C[M=${M},N=${N}]`;
    }
    if (els.hostLaunchEvidence) {
      els.hostLaunchEvidence.textContent = `confirmed · AIC${activeBlockIdx} / OT${activePartition.outputTile} reads A[M${activeMTile},K0…K${kTiles - 1}] + B[K0…K${kTiles - 1},N${activeNTile}] → writes C[M${activeMTile},N${activeNTile}]`;
    }
    if (els.hostLaunchBlockDim) {
      els.hostLaunchBlockDim.textContent = `blockDim = ${mTiles}×${nTiles} = ${blockDim}`;
    }
    if (els.hostLaunchATitle) els.hostLaunchATitle.textContent = `Input X → A [M=${M},K=${K}]`;
    if (els.hostLaunchAMeta) els.hostLaunchAMeta.textContent = `${mTiles} M tiles × ${kTiles} K tiles`;
    if (els.hostLaunchBTitle) els.hostLaunchBTitle.textContent = `Weight W → B [K=${K},N=${N}]`;
    if (els.hostLaunchBMeta) els.hostLaunchBMeta.textContent = `${kTiles} K tiles × ${nTiles} N tiles`;
    if (els.hostLaunchCTitle) els.hostLaunchCTitle.textContent = `Output Y → C [M=${M},N=${N}]`;
    if (els.hostLaunchCMeta) {
      els.hostLaunchCMeta.textContent = `AIC${activeBlockIdx} owns M${activeMTile}/N${activeNTile} · ${mTiles}×${nTiles} = ${outputTileCount} output tiles`;
    }
    if (els.hostLaunchMapping) {
      els.hostLaunchMapping.textContent = `blockIdx → Mi=floor(blockIdx/${nTiles}), Nj=blockIdx%${nTiles}`;
    }
    if (els.hostLaunchReduction) {
      els.hostLaunchReduction.textContent = `每个 AIC block 沿 K0…K${kTiles - 1} 归约，只写回一个 C[${tileM},${tileN}] tile`;
    }

    const fixtures = {
      a: {
        canvas: els.hostLaunchACanvas,
        scene: selectSceneCells(createOverviewMatrixScene({
          rows: M,
          columns: K,
          rowSpan: tileM,
          columnSpan: tileK,
          tone: 'input',
          axes: { rows: `M=${M}`, columns: `K=${K}` },
          prefix: 'host-launch-a',
        }), (cell) => cell.row === activeMTile * tileM),
        ariaLabel: `Input X logical Cube A matrix, M ${M} by K ${K}, ${mTiles} by ${kTiles} tiles`,
      },
      b: {
        canvas: els.hostLaunchBCanvas,
        scene: selectSceneCells(createOverviewMatrixScene({
          rows: K,
          columns: N,
          rowSpan: tileK,
          columnSpan: tileN,
          tone: 'input',
          axes: { rows: `K=${K}`, columns: `N=${N}` },
          prefix: 'host-launch-b',
        }), (cell) => cell.column === activeNTile * tileN),
        ariaLabel: `Weight W logical Cube B matrix, K ${K} by N ${N}, ${kTiles} by ${nTiles} tiles`,
      },
      c: {
        canvas: els.hostLaunchCCanvas,
        scene: selectSceneCells(createOverviewMatrixScene({
          rows: M,
          columns: N,
          rowSpan: tileM,
          columnSpan: tileN,
          tone: 'output',
          axes: { rows: `M=${M}`, columns: `N=${N}` },
          prefix: 'host-launch-c',
        }), (cell) => (
          cell.row === activeMTile * tileM
          && cell.column === activeNTile * tileN
        )),
        ariaLabel: `Output Y logical Cube C matrix, M ${M} by N ${N}, ${outputTileCount} output tiles`,
      },
    };
    const matrixOptions = {
      showAxes: true,
      showGrid: true,
      interactive: true,
      showTooltip: true,
      autoFit: true,
      minZoom: 0.015,
      padding: { top: 22, right: 12, bottom: 32, left: 42 },
    };

    Object.entries(fixtures).forEach(([key, fixture]) => {
      const options = { ...matrixOptions, ariaLabel: fixture.ariaLabel };
      const controller = state.hostLaunchControllers[key];
      if (controller) {
        controller.update(fixture.scene, options);
        controller.fit();
      } else {
        state.hostLaunchControllers[key] = window.PtoMatrixCanvas.render(
          fixture.canvas,
          fixture.scene,
          options
        );
        state.hostLaunchControllers[key].fit();
      }
    });

  }

  function renderConvTensorOverview(trace) {
    const params = trace?.tiling?.params || {};
    const hi = num(params.hi, 8);
    const wi = num(params.wi, 8);
    const ci = num(params.ci, 16);
    const ho = num(params.ho, 8);
    const wo = num(params.wo, 8);
    const co = num(params.co, 32);
    const kh = num(params.kh, 3);
    const kw = num(params.kw, 3);
    const volumeOptions = {
      showAxes: true,
      autoLabelDensity: true,
      padding: { top: 24, right: 28, bottom: 30, left: 36 },
    };
    const matrixOptions = {
      showAxes: true,
      showGrid: true,
      interactive: true,
      showTooltip: true,
      autoFit: true,
      padding: { top: 26, right: 24, bottom: 34, left: 48 },
    };
    const fixtures = {
      feature: {
        canvas: els.featureOverviewCanvas,
        scene: createVolumeScene({
          columns: wi,
          rows: hi,
          depth: ci,
          tone: 'neutral',
          axes: { columns: 'W=8', rows: 'H=8', depth: 'Ci=16' },
          prefix: 'feature',
        }),
        options: { ...volumeOptions, ariaLabel: `Feature X NCHW 1 by ${ci} by ${hi} by ${wi}` },
        api: window.PtoTensorVolumeCanvas,
      },
      weight: {
        canvas: els.weightOverviewCanvas,
        scene: createVolumeScene({
          columns: kw,
          rows: kh,
          depth: ci,
          tone: 'neutral',
          axes: { columns: `Kw=${kw}`, rows: `Kh=${kh}`, depth: `Ci=${ci}` },
          prefix: 'weight',
        }),
        options: {
          ...volumeOptions,
          showAxes: true,
          padding: { top: 24, right: 28, bottom: 30, left: 36 },
          ariaLabel: `Weight W logical OIHW tensor with ${co} filters; representative filter is Ci ${ci} by Kh ${kh} by Kw ${kw}`,
        },
        api: window.PtoTensorVolumeCanvas,
      },
      bias: {
        canvas: els.biasOverviewCanvas,
        scene: createOverviewMatrixScene({
          rows: 1,
          columns: co,
          rowSpan: 1,
          columnSpan: 2,
          tone: 'neutral',
          axes: { rows: 'broadcast', columns: `Co=${co}` },
          prefix: 'bias',
        }),
        options: { ...matrixOptions, padding: { top: 18, right: 20, bottom: 28, left: 54 }, ariaLabel: `Bias vector with one value for each of ${co} output channels` },
        api: window.PtoMatrixCanvas,
      },
      output: {
        canvas: els.outputOverviewCanvas,
        scene: createVolumeScene({
          columns: wo,
          rows: ho,
          depth: co,
          tone: 'neutral',
          axes: { columns: 'Wo=8', rows: 'Ho=8', depth: 'Co=32' },
          prefix: 'output',
        }),
        options: { ...volumeOptions, ariaLabel: `Output Y NCHW 1 by ${co} by ${ho} by ${wo}` },
        api: window.PtoTensorVolumeCanvas,
      },
    };

    Object.entries(fixtures).forEach(([key, fixture]) => {
      if (!fixture.canvas || !fixture.api) return;
      const controller = state.overviewControllers[key];
      if (controller) {
        controller.update(fixture.scene, { ...fixture.options, preserveView: true });
        controller.resize();
      } else {
        state.overviewControllers[key] = fixture.api.render(
          fixture.canvas,
          fixture.scene,
          fixture.options
        );
      }
    });
  }

  function snapshotPatternTone(snapshot) {
    const tone = snapshotTone(snapshot);
    return tone === 'accumulator' ? 'reduction' : tone;
  }

  function tensorDefinition(trace, snapshot) {
    return [...(trace?.tensors || []), ...(trace?.buffers || [])]
      .find((item) => item.id === snapshot?.tensorId) || null;
  }

  function renderTensorVolumePattern(trace, snapshot) {
    if (!snapshot || !els.tensorVolumeCanvas || !window.PtoTensorVolumeCanvas) return;
    const shape = tensorSnapshotShape(snapshot);
    while (shape.length > 3 && shape[0] === 1) shape.shift();
    while (shape.length > 3) shape.shift();
    const layout = String(snapshot.logicalLayout || snapshot.physicalLayout || '');
    const isNc1hwc0 = layout.toUpperCase().includes('NC1HWC0');
    const rows = isNc1hwc0 ? shape[0] : shape[1];
    const columns = isNc1hwc0 ? shape[1] : shape[2];
    const depth = isNc1hwc0 ? shape[2] : shape[0];
    const voxels = [];
    const tone = snapshotPatternTone(snapshot);
    for (let z = 0; z < depth; z += 1) {
      for (let row = 0; row < rows; row += 1) {
        for (let column = 0; column < columns; column += 1) {
          voxels.push({
            id: `${snapshot.tensorId}-${column}-${row}-${z}`,
            column,
            row,
            depth: z,
            tone,
            state: 'base',
          });
        }
      }
    }
    const definition = tensorDefinition(trace, snapshot);
    const scene = {
      extent: { columns, rows, depth },
      axes: isNc1hwc0
        ? { columns: 'W', rows: 'H', depth: 'C0' }
        : { columns: 'W / column', rows: 'H / row', depth: 'C / depth' },
      voxels,
    };
    const options = {
      ariaLabel: `${definition?.name || snapshot.tensorId} ${formatShape(snapshot.logicalShape)} three-dimensional tensor volume`,
      padding: { top: 42, right: 38, bottom: 42, left: 50 },
      showAxes: true,
      autoLabelDensity: true,
    };
    if (state.tensorVolumeController) {
      state.tensorVolumeController.update(scene, options);
      state.tensorVolumeController.resize();
    } else {
      state.tensorVolumeController = window.PtoTensorVolumeCanvas.render(
        els.tensorVolumeCanvas,
        scene,
        options
      );
    }
  }

  function renderTensorMatrixPattern(trace, visual, snapshot) {
    if (!els.tensorMatrixCanvas || !window.PtoMatrixCanvas) return;
    const scene = snapshot
      ? matrixSceneFromSnapshot(snapshot)
      : matrixSceneFromVisual(visual);
    const definition = tensorDefinition(trace, snapshot);
    const options = {
      ariaLabel: snapshot
        ? `${definition?.name || snapshot.tensorId} ${formatShape(snapshot.logicalShape)} two-dimensional tensor matrix`
        : `${visual.title || 'Two-dimensional tensor matrix'}`,
      showAxes: true,
      showGrid: true,
      interactive: true,
      showTooltip: true,
      autoFit: true,
      padding: { top: 42, right: 34, bottom: 42, left: 52 },
    };
    if (state.tensorMatrixController) {
      state.tensorMatrixController.update(scene, { ...options, preserveView: true });
      state.tensorMatrixController.resize();
    } else {
      state.tensorMatrixController = window.PtoMatrixCanvas.render(
        els.tensorMatrixCanvas,
        scene,
        options
      );
    }
  }

  function matrixSceneFromSnapshot(snapshot) {
    const shape = tensorSnapshotShape(snapshot);
    const rows = Math.max(1, shape.at(-2) || 1);
    const columns = Math.max(1, shape.at(-1) || shape[0] || 1);
    const targetRows = Math.min(rows, 16);
    const targetColumns = Math.min(columns, 20);
    const rowSpan = Math.max(1, Math.ceil(rows / targetRows));
    const columnSpan = Math.max(1, Math.ceil(columns / targetColumns));
    const tone = snapshotPatternTone(snapshot);
    const written = String(snapshot?.role || '').includes('output')
      || String(snapshot?.role || '').includes('accumulator');
    const cells = [];
    for (let row = 0; row < rows; row += rowSpan) {
      for (let column = 0; column < columns; column += columnSpan) {
        const resolvedRowSpan = Math.min(rowSpan, rows - row);
        const resolvedColumnSpan = Math.min(columnSpan, columns - column);
        const aggregate = resolvedRowSpan > 1 || resolvedColumnSpan > 1;
        cells.push({
          id: `${snapshot.tensorId}-${row}-${column}`,
          row,
          column,
          rowSpan: resolvedRowSpan,
          columnSpan: resolvedColumnSpan,
          tone,
          style: aggregate ? 'aggregate' : 'value',
          states: written ? ['written'] : [],
          summary: aggregate ? {
            rows: resolvedRowSpan,
            columns: resolvedColumnSpan,
            count: resolvedRowSpan * resolvedColumnSpan,
            intensity: 0.62,
          } : undefined,
        });
      }
    }
    const layout = snapshot.logicalLayout || snapshot.physicalLayout || 'matrix';
    const axisParts = String(layout)
      .split('×')
      .map((part) => part.trim())
      .filter(Boolean);
    return {
      extent: { rows, columns },
      axes: {
        rows: axisParts[0] || `${layout} · row`,
        columns: axisParts[1] || `${layout} · column`,
      },
      cells,
    };
  }

  function matrixSceneFromVisual(visual) {
    const grid = visual?.grid || {};
    const rows = Math.max(1, Number(grid.rowTotal) || 1);
    const columns = Math.max(1, Number(grid.colTotal) || 1);
    const rowSpan = Math.max(1, Number(grid.rowCell) || 1);
    const columnSpan = Math.max(1, Number(grid.colCell) || 1);
    const highlight = visual?.highlight || {};
    const activeRows = highlight.row || [-1, -1];
    const activeColumns = highlight.col || [-1, -1];
    const cells = [];
    for (let row = 0; row < rows; row += rowSpan) {
      for (let column = 0; column < columns; column += columnSpan) {
        const resolvedRowSpan = Math.min(rowSpan, rows - row);
        const resolvedColumnSpan = Math.min(columnSpan, columns - column);
        const active = row < activeRows[1]
          && row + resolvedRowSpan > activeRows[0]
          && column < activeColumns[1]
          && column + resolvedColumnSpan > activeColumns[0];
        cells.push({
          id: `matrix-${row}-${column}`,
          row,
          column,
          rowSpan: resolvedRowSpan,
          columnSpan: resolvedColumnSpan,
          tone: active ? (highlight.tone || 'input') : 'neutral',
          style: 'aggregate',
          states: active ? ['selected'] : [],
          summary: {
            rows: resolvedRowSpan,
            columns: resolvedColumnSpan,
            count: resolvedRowSpan * resolvedColumnSpan,
            intensity: active ? 1 : 0.38,
          },
        });
      }
    }
    return {
      extent: { rows, columns },
      axes: {
        rows: grid.rowLabel || 'row',
        columns: grid.colLabel || 'column',
      },
      cells,
    };
  }

  function renderConvCoreContext(trace) {
    if (!els.convCoreContext || !els.convCoreOptions) return;
    const visible = trace?.operator?.kind === 'conv2d-cube';
    els.convCoreContext.hidden = !visible;
    if (!visible) return;
    const partition = convCorePartition(trace);
    const { blockCount, mTiles, nTiles } = partition;
    state.convCoreIndex = partition.index;
    const signature = `${blockCount}:${mTiles}:${nTiles}`;
    if (els.convCoreOptions.dataset.signature !== signature) {
      els.convCoreOptions.dataset.signature = signature;
      const buttons = Array.from({ length: blockCount }, (_, index) => {
        const mTile = Math.floor(index / nTiles);
        const nTile = index % nTiles;
        const button = document.createElement('button');
        button.className = 'btn btn-compact btn-ghost';
        button.type = 'button';
        button.role = 'radio';
        button.dataset.convCoreIndex = String(index);
        button.textContent = `AIC${index} · OT${index} · M${mTile}/N${nTile}`;
        button.setAttribute('aria-label', `AIC ${index}, output tile ${index}, M tile ${mTile}, N tile ${nTile}`);
        return button;
      });
      els.convCoreOptions.replaceChildren(...buttons);
    }
    els.convCoreOptions.querySelectorAll('[data-conv-core-index]').forEach((button) => {
      const selected = Number(button.dataset.convCoreIndex) === state.convCoreIndex;
      button.classList.toggle('is-selected', selected);
      button.setAttribute('aria-checked', String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
  }

  function renderConvLoadDataView(c) {
    const mTiles = Math.max(1, Math.ceil(c.M / c.tileM));
    const nTiles = Math.max(1, Math.ceil(c.N / c.tileN));
    const kTiles = Math.max(1, Math.ceil(c.K / c.tileK));
    const mTile = Math.floor(state.convCoreIndex / nTiles);
    const kTile = Math.min(kTiles - 1, c.kIndex);
    const selectionStart = Math.max(0, Math.min(kTiles - 1, Number(c.kTileSelection?.[0] ?? kTile)));
    const selectionEnd = Math.max(selectionStart, Math.min(kTiles - 1, Number(c.kTileSelection?.[1] ?? kTile)));
    const isRangeSelection = selectionEnd > selectionStart;
    if (els.fmapA1Meta) {
      els.fmapA1Meta.textContent = `NC1HWC0 [${c.n},1,${c.hi},${c.wi},${c.ci}]`;
    }
    if (els.fmapA1Params) {
      els.fmapA1Params.textContent = `FP16 · padList=[${c.padLeft},${c.padLeft},${c.padTop},${c.padTop}] · padValue=0`;
    }
    if (els.a2LogicalShape) {
      els.a2LogicalShape.textContent = `M=Ho×Wo=${c.M} · K=Ci×Kh×Kw=${c.K}`;
    }
    if (els.fmapA2TileMeta) {
      els.fmapA2TileMeta.textContent = isRangeSelection
        ? `M${mTile} × K${selectionStart}–K${selectionEnd} · ${selectionEnd - selectionStart + 1} tiles`
        : `M${mTile} × K${kTile} = ${c.tileM} × ${c.tileK}`;
    }
    if (els.a2LogicalMatrixCanvas && window.PtoMatrixCanvas) {
      const cells = [];
      for (let row = 0; row < c.M; row += c.tileM) {
        for (let column = 0; column < c.K; column += c.tileK) {
          const cellKTile = column / c.tileK;
          const selected = row / c.tileM === mTile
            && cellKTile >= selectionStart
            && cellKTile <= selectionEnd;
          const rowSpan = Math.min(c.tileM, c.M - row);
          const columnSpan = Math.min(c.tileK, c.K - column);
          cells.push({
            id: `a2-${row}-${column}`,
            row,
            column,
            rowSpan,
            columnSpan,
            tone: selected ? 'input' : 'neutral',
            style: 'aggregate',
            states: selected ? ['selected'] : [],
            summary: {
              rows: rowSpan,
              columns: columnSpan,
              count: rowSpan * columnSpan,
              intensity: selected ? 1 : 0.36,
            },
          });
        }
      }
      const matrixScene = {
        extent: { rows: c.M, columns: c.K },
        axes: { rows: 'M = Ho × Wo', columns: 'K = Ci × Kh × Kw' },
        cells,
      };
      const matrixOptions = {
        ariaLabel: isRangeSelection
          ? `A2 logical matrix with ${mTiles} M tiles and ${kTiles} K tiles; selected tiles M${mTile}, K${selectionStart} through K${selectionEnd}`
          : `A2 logical matrix with ${mTiles} M tiles and ${kTiles} K tiles; current tile M${mTile}, K${kTile}`,
        showAxes: true,
        showGrid: true,
        interactive: true,
        showTooltip: true,
        autoFit: true,
        padding: { top: 36, right: 24, bottom: 38, left: 46 },
      };
      if (state.a2LogicalMatrixController) {
        state.a2LogicalMatrixController.update(matrixScene, { ...matrixOptions, preserveView: true });
        state.a2LogicalMatrixController.resize();
      } else {
        state.a2LogicalMatrixController = window.PtoMatrixCanvas.render(
          els.a2LogicalMatrixCanvas,
          matrixScene,
          matrixOptions
        );
      }
    }

    const scene = convFmapA1VolumeScene(c, mTile);
    const options = {
      ariaLabel: `fmapA1 NC1HWC0 physical tensor with virtual padding and representative LoadData3D window for M${mTile}, K${kTile}`,
      padding: { top: 34, right: 28, bottom: 36, left: 44 },
      showAxes: true,
      autoLabelDensity: true,
    };
    if (state.fmapA1VolumeController) {
      state.fmapA1VolumeController.update(scene, options);
    } else {
      state.fmapA1VolumeController = window.PtoTensorVolumeCanvas.render(
        els.fmapA1VolumeCanvas,
        scene,
        options
      );
    }
  }

  function convFmapA1VolumeScene(c, mTile) {
    const padTop = Math.max(0, c.padTop);
    const padLeft = Math.max(0, c.padLeft);
    const padBottom = padTop;
    const padRight = padLeft;
    const rows = c.hi + padTop + padBottom;
    const columns = c.wi + padLeft + padRight;
    const depth = c.ci;
    const representativeM = Math.min(c.M - 1, mTile * c.tileM);
    const outputRow = Math.floor(representativeM / c.wo);
    const outputColumn = representativeM % c.wo;
    const windowRow0 = outputRow * c.strideH;
    const windowColumn0 = outputColumn * c.strideW;
    const voxels = [];
    for (let channel = 0; channel < depth; channel += 1) {
      for (let row = 0; row < rows; row += 1) {
        for (let column = 0; column < columns; column += 1) {
          const isPadding = row < padTop
            || row >= rows - padBottom
            || column < padLeft
            || column >= columns - padRight;
          const inWindow = row >= windowRow0
            && row < windowRow0 + c.kh
            && column >= windowColumn0
            && column < windowColumn0 + c.kw;
          let stateName = isPadding ? 'padding' : 'base';
          let tone = 'neutral';
          if (inWindow && isPadding) {
            stateName = 'current';
            tone = 'compute';
          } else if (inWindow) {
            stateName = 'window';
            tone = 'input';
          }
          voxels.push({
            id: `a1-${column}-${row}-${channel}`,
            column,
            row,
            depth: channel,
            tone,
            state: stateName,
          });
        }
      }
    }
    return {
      extent: { columns, rows, depth },
      axes: { columns: 'Wi', rows: 'Hi', depth: 'C0' },
      voxels,
    };
  }

  // Resize the canvas backing store only when the CSS size actually changed,
  // so per-step / per-drag redraws don't reallocate the GPU buffer.
  function fitCanvas(canvas, cssWidth, cssHeight) {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.floor(cssWidth * dpr);
    const h = Math.floor(cssHeight * dpr);
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  function tensorViewportTip(visual) {
    const parts = [];
    if (visual.layout === '1d') {
      parts.push('1D 逻辑 tensor：整条 GM 线性地址；高亮块 = 当前 tile 实际访问的 element 区间。');
    } else if (visual.layout === 'conv2d') {
      if (visual.conv?.scene === 'copy-in') {
        const transfer = visual.conv.copyTransfer;
        const core = transfer?.partition;
        const source = transfer?.coreSource;
        parts.push(`AIC${core?.index ?? 0} · M${core?.mTile ?? 0}/N${core?.nTile ?? 0}：${source?.slice || transfer?.flow?.from || 'GM input'}（GM + ${source?.gmOffsetBytes || 0} B）→ ${transfer?.flow?.to || 'L1'}。数据已由 MTE2 写入目标 Buffer，但必须经过 MTE2_MTE1 同步后，MTE1 才能读取。`);
      } else if (visual.conv?.scene === 'bias-copy') {
        parts.push('Bias C1→C2：MTE2_MTE1 已使 L1 可读，MTE1 再把 16 个 FP32 Bias 值搬入 Bias Table。');
      } else if (visual.conv?.scene === 'load3d') {
        if (state.tensorTabKey === 'buffer:weight:b2') {
          parts.push('LoadData2D：从 B1 的当前 K0 分形取数，按转置语义生成 B2 [16,16] ZN。');
        } else {
          parts.push('LoadData3D：A1 是物理 NC1HWC0 [1,1,8,8,16]；padList 只定义越界读取时写入 A2 的虚拟 PAD 值，不会扩张 A1 的物理 shape。');
        }
      } else {
        parts.push('二维输出 tile + K reduction：K 是归约进度，不是输出 tensor 的第三维。');
      }
    } else {
      parts.push('二维输出 tile：高亮区域 = 当前 M×N 输出分区；K 只作为规约进度展示。Cmd/Ctrl+滚轮缩放。');
    }
    if (visual.title) parts.push(visual.title);
    const operations = visual.layout === 'conv2d'
      && visual.conv?.scene === 'load3d'
      && state.tensorTabKey === 'buffer:weight:b2'
      ? ['LoadData2D', 'K fractal', 'NZ→ZN']
      : (visual.operationChips || []);
    if (operations.length) parts.push(`当前操作：${operations.join(', ')}`);
    return parts.join('\n');
  }

  function drawTensorScene(ctx, width, height, visual) {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = getCss('--surface-2');
    ctx.fillRect(0, 0, width, height);
    drawTensorBackdrop(ctx, width, height);
    if (visual.layout === 'conv2d') drawConvExecution(ctx, width, height, visual);
    else if (visual.layout === '2d') drawTensorGrid(ctx, width, height, visual);
    else drawTensorStrip(ctx, width, height, visual);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    if (visual.highlight && visual.layout !== 'conv2d') {
      drawTileLabels(ctx, width, [{ label: visual.highlight.label, tone: visual.highlight.tone }]);
    }
  }

  function drawConvExecution(ctx, width, height, visual) {
    const c = visual.conv || {};
    const callout = c.scene === 'load3d' && state.tensorTabKey === 'buffer:weight:b2'
      ? {
          ...visual.highlight,
          label: `Weight K tile → B2[K${c.kIndex}, N=${c.tileN}]`,
          sub: 'confirmed · B1 NZ → B2 ZN',
        }
      : visual.highlight;
    drawTensorCallout(ctx, width, height, callout, { x: 24, y: 44 });
    const sceneTop = 78 + (state.tensorView.panY || 0);
    const scale = Math.max(0.7, Math.min(1.5, state.tensorView.scale || 1));
    const sceneLeft = 24 + (state.tensorView.panX || 0);
    const sceneWidth = Math.max(360, width - 48);
    // The tile lens is an overlay, not a reserved second canvas. Keeping more
    // vertical room here lets spatial LoadData3D windows remain readable in
    // the narrow middle workbench pane.
    const sceneHeight = Math.max(190, height - sceneTop - 70);
    ctx.save();
    ctx.translate(sceneLeft, sceneTop);
    ctx.scale(scale, scale);
    const scaledWidth = sceneWidth / scale;
    const scaledHeight = sceneHeight / scale;
    if (c.scene === 'copy-in') drawConvCopyIn(ctx, scaledWidth, scaledHeight, c);
    else if (c.scene === 'bias-copy') drawConvBiasCopy(ctx, scaledWidth, scaledHeight, c);
    else if (c.scene === 'load3d') drawConvLoad3D(ctx, scaledWidth, scaledHeight, c);
    else if (c.scene === 'mmad') drawConvMmad(ctx, scaledWidth, scaledHeight, c);
    else if (c.scene === 'epilogue') drawConvEpilogue(ctx, scaledWidth, scaledHeight, c);
    else if (c.scene === 'copy-out') drawConvCopyOut(ctx, scaledWidth, scaledHeight, c);
    else if (c.scene === 'event') drawConvEvent(ctx, scaledWidth, scaledHeight, c);
    else drawConvOverview(ctx, scaledWidth, scaledHeight, c);
    ctx.restore();
  }

  function drawConvOverview(ctx, width, height, c) {
    const gap = 12;
    const top = 26;
    const boxW = Math.max(118, (width - gap * 3) / 4);
    const boxH = Math.min(104, Math.max(76, height - 54));
    const items = [
      { label: 'Feature X', shape: `[1,${c.ci},${c.hi},${c.wi}]`, meta: 'FP16 · GM', tone: 'input' },
      { label: 'Weight W', shape: `[${c.co},${c.ci},${c.kh},${c.kw}]`, meta: 'FP16 · GM', tone: 'input' },
      { label: 'Bias', shape: `[${c.co}]`, meta: 'FP32 · GM', tone: 'reduction' },
      { label: 'Output Y', shape: `[1,${c.co},${c.ho},${c.wo}]`, meta: 'FP16 · GM', tone: 'output' },
    ];
    items.forEach((item, index) => {
      drawConvObjectBox(ctx, gap + index * (boxW + gap), top, boxW, boxH, item);
    });
    drawConvFooter(ctx, width, height, `M=${c.ho}×${c.wo}=${c.M}   K=${c.ci}×${c.kh}×${c.kw}=${c.K}   N=${c.co}`, 'Host-confirmed fixed tiling: 16×16×16 · blockDim 8');
  }

  function drawConvCopyIn(ctx, width, height, c) {
    const transfer = c.copyTransfer;
    if (!transfer) {
      drawConvFooter(ctx, width, height, 'Copy input data unavailable', 'No DataFlow is linked to the selected tensor');
      return;
    }
    const { flow, source, destination, snapshot, gateEvent, partition, coreSource } = transfer;
    const shape = formatShape(snapshot?.physicalShape || snapshot?.logicalShape || destination?.logicalShape);
    const sourceLayout = String(source?.physicalLayout || source?.logicalLayout || 'layout unknown').split(' ')[0];
    const destinationLayout = snapshot?.physicalLayout || destination?.physicalLayout || 'layout unknown';
    const dtype = String(snapshot?.dtype || destination?.dtype || source?.dtype || 'dtype unknown').toUpperCase();
    const address = Number.isFinite(Number(destination?.addressBytes)) ? `@${Number(destination.addressBytes)}` : '@unknown';
    const alignment = Number.isFinite(Number(destination?.alignmentBytes))
      ? `align ${Number(destination.alignmentBytes)} B`
      : 'alignment unknown';
    const bytes = Number(flow?.bytes) || Number(snapshot?.validBytes) || 0;
    const sourceBoxW = width * 0.35;
    const destinationX = width * 0.65;
    const destinationBoxW = width - destinationX;
    const boxY = 32;
    const boxH = Math.min(108, Math.max(78, height - 112));
    const tone = source?.role === 'bias' ? 'reduction' : 'input';
    const transformation = flow.status === 'ND-to-NZ'
      ? 'ND → NZ'
      : sourceLayout === destinationLayout
        ? 'layout unchanged'
        : `${sourceLayout} → ${destinationLayout}`;

    drawConvObjectBox(ctx, 0, boxY, sourceBoxW, boxH, {
      label: coreSource?.label || flow.from || source?.name || 'GM source',
      shape: coreSource?.slice ? `${sourceLayout} · ${coreSource.slice}` : `${sourceLayout} ${shape}`,
      meta: `${dtype} · ${formatBytes(bytes)} · GM + ${coreSource?.gmOffsetBytes || 0} B`,
      tone,
    });
    drawConvFlowArrow(ctx, sourceBoxW + 14, boxY + boxH / 2, destinationX - sourceBoxW - 28, false);
    drawConvObjectBox(ctx, destinationX, boxY, destinationBoxW, boxH, {
      label: `${destination?.name || flow.to || 'L1 buffer'} · ${destination?.location || flow.to || 'L1'}`,
      shape: `${destinationLayout} ${shape}`,
      meta: `${address} · ${formatBytes(bytes)} / ${formatBytes(destination?.allocatedBytes)} · ${alignment}`,
      tone,
    });

    ctx.textAlign = 'center';
    ctx.fillStyle = getCss('--foreground');
    ctx.font = '700 10px ui-monospace, monospace';
    drawFittedText(ctx, transformation, width * 0.5, boxY + 24, 96);
    ctx.fillStyle = getCss('--foreground-muted');
    ctx.font = '600 9px Inter, sans-serif';
    drawFittedText(ctx, flow.transferEngine || 'MTE2 / DataCopy', width * 0.5, boxY + boxH - 12, 124);
    ctx.textAlign = 'left';

    drawCopyReadiness(ctx, width, Math.min(height - 54, boxY + boxH + 24));
    drawConvFooter(
      ctx,
      width,
      height,
      `Copied by MTE2 → Awaiting ${gateEvent?.eventType || 'MTE2_MTE1'}`,
      `MTE1 blocked · AIC${partition?.index ?? 0} / OT${partition?.outputTile ?? 0} · ${coreSource?.addressing || 'GM source'} → local ${address}`
    );
  }

  function drawCopyReadiness(ctx, width, y) {
    const stages = [
      { label: 'Copied by MTE2', color: getCss('--success') },
      { label: 'Awaiting MTE2_MTE1', color: getCss('--warning') },
      { label: 'MTE1 blocked', color: getCss('--foreground-muted') },
    ];
    const gap = Math.min(178, width / stages.length);
    const startX = Math.max(8, (width - gap * (stages.length - 1)) / 2);
    ctx.save();
    ctx.font = '600 9px Inter, sans-serif';
    ctx.textBaseline = 'middle';
    stages.forEach((stage, index) => {
      const x = startX + index * gap;
      if (index < stages.length - 1) {
        ctx.strokeStyle = getCss('--border-default');
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x + 5, y);
        ctx.lineTo(x + gap - 5, y);
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = stage.color;
      ctx.fill();
      ctx.fillStyle = stage.color;
      ctx.textAlign = index === stages.length - 1 ? 'right' : index === 0 ? 'left' : 'center';
      const labelX = index === stages.length - 1 ? x + 4 : index === 0 ? x - 4 : x;
      ctx.fillText(stage.label, labelX, y - 13);
    });
    ctx.restore();
  }

  function drawConvBiasCopy(ctx, width, height, c) {
    const flow = c.dataFlows?.[0] || {};
    const snapshot = c.snapshots?.[0] || {};
    const boxH = Math.min(96, Math.max(72, height - 70));
    const boxW = width * 0.34;
    const targetX = width * 0.66;
    drawConvObjectBox(ctx, 0, 28, boxW, boxH, {
      label: flow.from || 'C1 / L1',
      shape: `linear ${formatShape(snapshot.logicalShape)}`,
      meta: `${String(snapshot.dtype || 'fp32').toUpperCase()} · ${formatBytes(flow.bytes)}`,
      tone: 'reduction',
    });
    drawConvFlowArrow(ctx, boxW + 14, 28 + boxH / 2, targetX - boxW - 28, false);
    drawConvObjectBox(ctx, targetX, 28, width - targetX, boxH, {
      label: flow.to || snapshot.location || 'C2 / Bias Table',
      shape: `${snapshot.physicalLayout || 'linear Bias Table'} ${formatShape(snapshot.physicalShape)}`,
      meta: `@0 · ${formatBytes(flow.bytes)} · ready for first Mmad`,
      tone: 'reduction',
    });
    ctx.fillStyle = getCss('--foreground-muted');
    ctx.font = '600 9px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(flow.transferEngine || 'MTE1 / DataCopy', width / 2, 48);
    ctx.textAlign = 'left';
    drawConvFooter(ctx, width, height, 'C1 → C2 / Bias Table', 'This is the next stage after MTE2_MTE1 makes L1 readable');
  }

  function drawConvLoad3D(ctx, width, height, c) {
    if (state.tensorTabKey === 'buffer:weight:b2') {
      drawConvLoadB2(ctx, width, height, c);
      return;
    }
    drawConvLoadA2(ctx, width, height, c);
  }

  function drawConvLoadA2(ctx, width, height, c) {
    const top = 28;
    const a1X = 0;
    const a1W = Math.max(190, width * 0.62);
    const matrixX = a1W + 26;
    const oh = c.outputPosition?.[0] || 0;
    const ow = c.outputPosition?.[1] || 0;
    const sourceH0 = oh * c.strideH - c.padTop;
    const sourceW0 = ow * c.strideW - c.padLeft;
    const window = {
      h0: Math.max(0, sourceH0),
      h1: Math.min(c.hi, sourceH0 + c.kh),
      w0: Math.max(0, sourceW0),
      w1: Math.min(c.wi, sourceW0 + c.kw),
    };

    drawConvSpatialVolume(ctx, a1X, top, a1W, height - 72, {
      label: `fmapA1 · A1 NC1HWC0 [1,1,${c.hi},${c.wi},16]`,
      rows: c.hi,
      cols: c.wi,
      channels: c.ci,
      channelLabel: 'C0',
      tone: 'input',
      window: { h0: window.h0, h1: window.h1, w0: window.w0, w1: window.w1 },
      note: `physical A1 · output(${oh},${ow}) reads logical H[${sourceH0}:${sourceH0 + c.kh}), W[${sourceW0}:${sourceW0 + c.kw})`,
    });
    drawConvFlowArrow(ctx, a1X + a1W - 4, top + Math.min(120, height * 0.32), 22, false);
    const matrixW = Math.max(92, width - matrixX);
    const matrixH = Math.min(104, Math.max(70, height * 0.34));
    drawMiniMatrix(ctx, matrixX, top + 28, matrixW, matrixH, 6, 8, 'input');
    ctx.fillStyle = getCss('--foreground');
    ctx.font = '700 11px ui-monospace, monospace';
    ctx.fillText(`A2 [M=${c.tileM}, K=${c.tileK}]`, matrixX, top + 14);
    ctx.fillStyle = getCss('--foreground-secondary');
    ctx.font = '600 10px Inter, sans-serif';
    ctx.fillText(`K${c.kIndex} [${c.kRange[0]}:${c.kRange[1]}]`, matrixX, top + matrixH + 48);
    ctx.fillText('Ci × Kh × Kw → matrix K', matrixX, top + matrixH + 64);
    ctx.fillStyle = getCss('--foreground-muted');
    ctx.font = '600 9px ui-monospace, monospace';
    ctx.fillText('A1 NC1HWC0 → A2 ZZ · 512 B', matrixX, top + matrixH + 80);
    drawConvFooter(ctx, width, height, 'fmapA1 / A1 → LoadData3D → fmapA2 / A2', `padList=[${c.padLeft},${c.padLeft},${c.padTop},${c.padTop}] is virtual · padValue=0`);
  }

  function drawConvLoadB2(ctx, width, height, c) {
    const top = 42;
    const sourceW = Math.max(156, width * 0.42);
    const targetX = width * 0.64;
    const targetW = Math.max(120, width - targetX);
    const matrixH = Math.min(160, Math.max(92, height * 0.42));
    drawMiniMatrix(ctx, 0, top, sourceW, matrixH, 9, 8, 'default');
    drawMiniMatrix(ctx, 0, top, sourceW, Math.max(18, matrixH / 9), 1, 8, 'input');
    drawMatrixLabel(ctx, 0, 16, 'weightB1 [K=144, N=16]', `B1 / L1 · NZ · select K${c.kIndex}`);
    drawConvFlowArrow(ctx, sourceW + 18, top + matrixH * 0.48, Math.max(34, targetX - sourceW - 36), false);
    drawMiniMatrix(ctx, targetX, top, targetW, matrixH, 8, 8, 'input');
    drawMatrixLabel(ctx, targetX, 16, `weightB2 [K=${c.tileK}, N=${c.tileN}]`, 'B2 / L0B · ZN · FP16');
    ctx.fillStyle = getCss('--foreground-secondary');
    ctx.font = '600 10px Inter, sans-serif';
    ctx.fillText(`K${c.kIndex} [${c.kRange[0]}:${c.kRange[1]}] · LoadData2D`, targetX, top + matrixH + 22);
    ctx.fillStyle = getCss('--foreground-muted');
    ctx.font = '600 9px ui-monospace, monospace';
    ctx.fillText('startIndex + transpose semantics', targetX, top + matrixH + 38);
    drawConvFooter(ctx, width, height, 'weightB1 / B1 → LoadData2D → weightB2 / B2', 'NZ → ZN · 16×16 FP16 · 512 B');
  }

  function drawConvSpatialVolume(ctx, x, y, width, height, options) {
    const rows = Math.max(1, options.rows || 1);
    const cols = Math.max(1, options.cols || 1);
    const channels = Math.max(1, options.channels || 1);
    const shownChannels = Math.min(channels, options.maxChannels || channels);
    const fitW = (width - 16) / Math.max(1, cols + shownChannels * 0.52);
    const fitH = (height - 62) / Math.max(1, rows + shownChannels * 0.38);
    const unit = Math.max(3, Math.min(14, fitW, fitH));
    const depthX = unit * 0.52;
    const depthY = unit * 0.38;
    const objectW = cols * unit + shownChannels * depthX;
    const objectH = rows * unit + shownChannels * depthY;
    const ox = x + Math.max(4, (width - objectW) / 2);
    const oy = y + 34 + shownChannels * depthY + Math.max(0, (height - 58 - objectH) / 2);
    const hi = voxelTone(options.tone || 'input');
    const pad = options.padding;
    const window = options.window;
    const cells = [];
    for (let r = 0; r < rows; r += 1) {
      for (let col = 0; col < cols; col += 1) {
        for (let channel = 0; channel < shownChannels; channel += 1) cells.push({ col, r, channel });
      }
    }
    cells.sort((a, b) => (b.channel - a.channel) || (b.r - a.r) || (a.col - b.col));
    cells.forEach((cell) => {
      const isPad = pad && (cell.r < pad.top || cell.r >= rows - pad.bottom || cell.col < pad.left || cell.col >= cols - pad.right);
      const isWindow = window && cell.r >= window.h0 && cell.r < window.h1 && cell.col >= window.w0 && cell.col < window.w1;
      const faces = isWindow ? (isPad ? VOXEL_TONES.compute : hi) : (isPad ? VOXEL_GHOST : VOXEL_GRAY);
      drawDepthVoxel(ctx, ox, oy, unit, depthX, depthY, cell.col, cell.r, cell.channel, faces);
    });
    drawDepthAxes(ctx, ox, oy, unit, depthX, depthY, cols, rows, shownChannels, options.channelLabel || 'Ci');
    ctx.fillStyle = getCss('--foreground');
    ctx.font = '700 10px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(options.label, x, y + 10);
    ctx.fillStyle = getCss('--foreground-muted');
    ctx.font = '600 8.5px ui-monospace, monospace';
    ctx.fillText(options.note, x, y + 24);
    ctx.fillText(`${options.channelLabel || 'Ci'}=${channels} · full physical tensor`, x, y + height - 10);
  }

  function depthPoint(originX, originY, unit, depthX, depthY, col, row, depth) {
    return {
      x: originX + col * unit + depth * depthX,
      y: originY + row * unit - depth * depthY,
    };
  }

  function drawDepthVoxel(ctx, originX, originY, unit, depthX, depthY, col, row, depth, faces) {
    const gap = 0.065;
    const point = (cc, rr, dd) => depthPoint(originX, originY, unit, depthX, depthY, cc, rr, dd);
    const c0 = col + gap;
    const c1 = col + 1 - gap;
    const r0 = row + gap;
    const r1 = row + 1 - gap;
    const d0 = depth + gap;
    const d1 = depth + 1 - gap;
    const f0 = point(c0, r0, d0);
    const f1 = point(c1, r0, d0);
    const f2 = point(c1, r1, d0);
    const f3 = point(c0, r1, d0);
    const b0 = point(c0, r0, d1);
    const b1 = point(c1, r0, d1);
    const b2 = point(c1, r1, d1);
    isoQuad(ctx, f0, f1, b1, b0, faces.top, faces.edge, 1);
    isoQuad(ctx, f1, f2, b2, b1, faces.east, faces.edge, 1);
    isoQuad(ctx, f0, f1, f2, f3, faces.south, faces.edge, 1);
  }

  function drawDepthAxes(ctx, originX, originY, unit, depthX, depthY, cols, rows, channels, channelLabel) {
    const frontTop = depthPoint(originX, originY, unit, depthX, depthY, 0, 0, 0);
    const frontBottomLeft = depthPoint(originX, originY, unit, depthX, depthY, 0, rows, 0);
    const frontBottomRight = depthPoint(originX, originY, unit, depthX, depthY, cols, rows, 0);
    const backTop = depthPoint(originX, originY, unit, depthX, depthY, 0, 0, channels);
    ctx.fillStyle = getCss('--foreground-secondary');
    ctx.font = '700 10px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Wi', (frontBottomLeft.x + frontBottomRight.x) / 2, frontBottomRight.y + 15);
    ctx.textAlign = 'right';
    ctx.fillText('Hi', frontBottomLeft.x - 7, (frontTop.y + frontBottomLeft.y) / 2);
    ctx.textAlign = 'center';
    ctx.fillText(channelLabel, (frontTop.x + backTop.x) / 2, (frontTop.y + backTop.y) / 2 - 8);
  }

  function drawConvMmad(ctx, width, height, c) {
    const matrixTop = 28;
    const matrixH = Math.min(96, Math.max(68, height - 62));
    const aX = 0;
    const aW = width * 0.23;
    const bX = width * 0.32;
    const bW = width * 0.18;
    const cX = width * 0.70;
    const cW = width * 0.30;
    drawMiniMatrix(ctx, aX, matrixTop, aW, matrixH, 8, 6, 'input');
    drawMiniMatrix(ctx, bX, matrixTop, bW, matrixH, 6, 5, 'input');
    drawMiniMatrix(ctx, cX, matrixTop, cW, matrixH, 8, 5, 'reduction', c.kCurrent / c.kLoops);
    drawConvMathGlyph(ctx, width * 0.275, matrixTop + matrixH * 0.52, '×');
    drawConvMathGlyph(ctx, width * 0.57, matrixTop + matrixH * 0.52, c.kIndex === 0 ? '+ Bias' : '+ CO1');
    drawConvFlowArrow(ctx, width * 0.61, matrixTop + matrixH * 0.52, width * 0.07, false);
    drawMatrixLabel(ctx, aX, 10, `A2 [${c.tileM},${c.tileK}]`, `K${c.kIndex} · FP16`);
    drawMatrixLabel(ctx, bX, 10, `B2 [${c.tileK},${c.tileN}]`, `K${c.kIndex} · FP16`);
    drawMatrixLabel(ctx, cX, 10, `CO1 [${c.tileM},${c.tileN}]`, `FP32 · ${c.kCurrent}/${c.kLoops}`);
    const kBarY = matrixTop + matrixH + 14;
    for (let index = 0; index < c.kLoops; index += 1) {
      const segmentW = (cW - 4) / c.kLoops;
      ctx.fillStyle = index < c.kCurrent ? 'rgba(255,154,84,0.78)' : 'rgba(165,175,185,0.12)';
      ctx.fillRect(cX + index * segmentW, kBarY, segmentW - 3, 5);
    }
    ctx.fillStyle = c.kIndex === 0 ? getCss('--success') : getCss('--foreground-muted');
    ctx.font = '600 9px Inter, sans-serif';
    ctx.fillText(c.kIndex === 0 ? 'Bias C1 → C2 confirmed · I0 only' : 'Bias is not added again', width * 0.49, kBarY + 18);
    drawConvFooter(ctx, width, height, `K${c.kIndex} [${c.kRange[0]}:${c.kRange[1]}]`, 'K is reduction progress, not an output tensor axis');
  }

  function drawConvEpilogue(ctx, width, height, c) {
    const items = [
      { label: 'CO1', shape: `[${c.tileM},${c.tileN}]`, meta: 'FP32 · L0C', tone: 'reduction' },
      { label: 'ReLU', shape: 'max(x, 0)', meta: 'reluEn=true', tone: 'compute' },
      { label: 'Cast', shape: 'FP32 → FP16', meta: 'QuantMode F322F16', tone: 'compute' },
      { label: 'GM Output', shape: `[${c.tileM},${c.tileN}]`, meta: 'FP16 ND · 512 B', tone: 'output' },
    ];
    const gap = 28;
    const boxW = Math.max(86, (width - gap * 3) / 4);
    const boxH = Math.min(104, Math.max(70, height - 54));
    items.forEach((item, index) => {
      const x = index * (boxW + gap);
      drawConvObjectBox(ctx, x, 24, boxW, boxH, item);
      if (index < items.length - 1) drawConvFlowArrow(ctx, x + boxW + 4, 24 + boxH / 2, gap - 8, false);
    });
    drawConvFooter(ctx, width, height, 'CO1 / L0C → Fixpipe → GM [M,Co]', 'NZ→ND · FP32→FP16 · ReLU · target compile/run unverified');
  }

  function drawConvCopyOut(ctx, width, height, c) {
    const leftW = width * 0.28;
    const matrixH = Math.min(104, height - 48);
    drawMatrixLabel(ctx, 0, 10, `CO1 [${c.tileM},${c.tileN}]`, 'FP32 NZ · 1024 B');
    drawMiniMatrix(ctx, 0, 30, leftW, matrixH, 8, 5, 'output');
    drawConvFlowArrow(ctx, leftW + 18, 30 + matrixH / 2, width * 0.18, false);
    const gridX = width * 0.56;
    const gridSide = Math.min(matrixH, width - gridX);
    drawMatrixLabel(ctx, gridX, 10, `Output Y[1,${c.co},${c.ho},${c.wo}]`, `M[0:${c.tileM}] · Co[0:${c.tileN}]`);
    const cell = gridSide / c.wo;
    for (let row = 0; row < c.ho; row += 1) {
      for (let col = 0; col < c.wo; col += 1) {
        const active = row * c.wo + col < c.tileM;
        ctx.fillStyle = active ? 'rgba(41,199,166,0.76)' : 'rgba(116,128,142,0.15)';
        ctx.strokeStyle = active ? getCss('--success') : getCss('--border-subtle');
        ctx.fillRect(gridX + col * cell, 30 + row * cell, cell - 1, cell - 1);
        ctx.strokeRect(gridX + col * cell, 30 + row * cell, cell - 1, cell - 1);
      }
    }
    ctx.fillStyle = getCss('--foreground-secondary');
    ctx.font = '600 9px Inter, sans-serif';
    ctx.fillText('offset = mStart × 32 + nStart', leftW + 20, 30 + matrixH / 2 + 20);
    drawConvFooter(ctx, width, height, 'CO1 → Fixpipe → GM', 'Direct strided write · one core writes 512 B');
  }

  function drawConvEvent(ctx, width, height, c) {
    const event = c.event || {};
    const l1Ready = event.eventType === 'MTE2_MTE1';
    const boxW = width * 0.28;
    const boxH = Math.min(92, height - 58);
    const y = 30;
    drawConvObjectBox(ctx, 0, y, boxW, boxH, { label: event.producerEngine || 'Producer', shape: 'SetFlag', meta: l1Ready ? 'L1 writes complete' : 'upstream complete', tone: 'input' });
    drawConvFlowArrow(ctx, boxW + 12, y + boxH / 2, width * 0.13, false);
    const eventX = width * 0.44;
    drawConvObjectBox(ctx, eventX, y, width * 0.18, boxH, { label: event.eventType || 'Event', shape: 'dependency', meta: 'confirmed', tone: 'fusion' });
    drawConvFlowArrow(ctx, width * 0.64, y + boxH / 2, width * 0.10, false);
    drawConvObjectBox(ctx, width * 0.76, y, width * 0.24, boxH, { label: event.consumerEngine || 'Consumer', shape: 'WaitFlag', meta: l1Ready ? 'L1 readable after wait' : 'blocked until ready', tone: 'compute' });
    drawConvFooter(ctx, width, height, event.explanation || 'Execution dependency', l1Ready ? 'Synchronization completes readiness; it does not move tensor data' : 'Event edge is not a data-transfer path');
  }

  function drawConvObjectBox(ctx, x, y, w, h, item) {
    const tone = TENSOR_TONES[item.tone] || TENSOR_TONES.default;
    ctx.beginPath();
    ctx.roundRect(x, y, Math.max(4, w), Math.max(4, h), 8);
    ctx.fillStyle = colorMixCanvas(tone.fill, item.unknown ? 0.35 : 0.72);
    ctx.fill();
    ctx.strokeStyle = item.unknown ? getCss('--danger') : tone.stroke;
    ctx.setLineDash(item.unknown ? [5, 4] : []);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = getCss('--foreground');
    ctx.font = '700 11px Inter, sans-serif';
    drawFittedText(ctx, item.label, x + 10, y + 20, w - 20);
    if (item.shape) {
      ctx.fillStyle = getCss('--foreground-secondary');
      ctx.font = '700 10px ui-monospace, monospace';
      drawFittedText(ctx, item.shape, x + 10, y + 40, w - 20);
    }
    ctx.fillStyle = item.unknown ? getCss('--danger') : getCss('--foreground-muted');
    ctx.font = '600 9px Inter, sans-serif';
    drawFittedText(ctx, item.meta || '', x + 10, y + h - 12, w - 20);
  }

  function drawConvFlowArrow(ctx, x, y, length, unknown) {
    const end = x + Math.max(12, length);
    ctx.save();
    ctx.strokeStyle = unknown ? getCss('--danger') : getCss('--foreground-muted');
    ctx.fillStyle = ctx.strokeStyle;
    ctx.lineWidth = 1.4;
    ctx.setLineDash(unknown ? [5, 4] : []);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(end, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(end, y);
    ctx.lineTo(end - 6, y - 4);
    ctx.lineTo(end - 6, y + 4);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  function drawMiniMatrix(ctx, x, y, w, h, rows, cols, toneKey, fillRatio = 1) {
    const tone = TENSOR_TONES[toneKey] || TENSOR_TONES.default;
    const gap = 2;
    const cellW = (w - gap * (cols - 1)) / cols;
    const cellH = (h - gap * (rows - 1)) / rows;
    const total = rows * cols;
    const activeCount = Math.round(total * Math.max(0, Math.min(1, fillRatio)));
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const index = row * cols + col;
        ctx.fillStyle = index < activeCount ? tone.fill : 'rgba(116,128,142,0.12)';
        ctx.fillRect(x + col * (cellW + gap), y + row * (cellH + gap), Math.max(1, cellW), Math.max(1, cellH));
      }
    }
  }

  function drawMatrixLabel(ctx, x, y, title, meta) {
    ctx.fillStyle = getCss('--foreground');
    ctx.font = '700 10px ui-monospace, monospace';
    ctx.fillText(title, x, y);
    ctx.fillStyle = getCss('--foreground-muted');
    ctx.font = '600 9px Inter, sans-serif';
    ctx.fillText(meta, x, y + 13);
  }

  function drawConvMathGlyph(ctx, x, y, label) {
    ctx.fillStyle = getCss('--foreground-secondary');
    ctx.font = '700 12px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, x, y);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
  }

  function drawConvFooter(ctx, width, height, primary, secondary) {
    const y = Math.max(20, height - 24);
    ctx.fillStyle = getCss('--foreground-secondary');
    ctx.font = '700 9px ui-monospace, monospace';
    drawFittedText(ctx, primary, 0, y, width);
    ctx.fillStyle = getCss('--foreground-muted');
    ctx.font = '600 9px Inter, sans-serif';
    drawFittedText(ctx, secondary, 0, y + 13, width);
  }

  function colorMixCanvas(color, alpha) {
    if (!color || !color.startsWith('rgba(')) return color;
    return color.replace(/rgba\(([^)]+),\s*[\d.]+\)/, `rgba($1, ${alpha})`);
  }

  function drawTensorBackdrop(ctx, width, height) {
    const gradient = ctx.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, 'rgba(255, 255, 255, 0.035)');
    gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);
  }

  // ----- fixed-view isometric helpers (3D, no rotation) -----
  const ISO_COS = Math.cos(Math.PI / 6);
  const ISO_SIN = Math.sin(Math.PI / 6);
  const VOXEL_GRAY = { top: '#474747', east: '#3a3a3a', south: '#2f2f2f', edge: 'rgba(18,18,18,0.65)' };
  const VOXEL_GHOST = { top: 'rgba(165,175,185,0.10)', east: 'rgba(165,175,185,0.07)', south: 'rgba(165,175,185,0.05)', edge: 'rgba(185,195,205,0.18)' };
  const VOXEL_TONES = {
    input:     { top: '#4d97ff', east: '#3f7ed6', south: '#3568b0', edge: 'rgba(8,20,40,0.55)' },
    output:    { top: '#29c7a6', east: '#21a88c', south: '#1b8b73', edge: 'rgba(6,34,28,0.55)' },
    compute:   { top: '#ffcf59', east: '#d9ad44', south: '#b88f34', edge: 'rgba(40,30,6,0.55)' },
    reduction: { top: '#ff9a54', east: '#d98044', south: '#b86836', edge: 'rgba(40,22,8,0.55)' },
    fusion:    { top: '#b892ff', east: '#9a78d9', south: '#7e61b8', edge: 'rgba(24,16,44,0.55)' },
  };
  function voxelTone(key) { return VOXEL_TONES[key] || VOXEL_TONES.reduction; }
  function hexToRgba(hex, a) {
    const m = /^#?([0-9a-fA-F]{6})$/.exec(hex);
    if (!m) return hex;
    const n = parseInt(m[1], 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
  }
  function isoQuad(ctx, p0, p1, p2, p3, fill, stroke, lw) {
    ctx.beginPath();
    ctx.moveTo(p0.x, p0.y); ctx.lineTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.lineTo(p3.x, p3.y);
    ctx.closePath();
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw || 1; ctx.stroke(); }
  }
  // one voxel cube at integer cell (c,r,k); draws the 3 camera-facing faces
  function drawVoxel(ctx, ox, oy, u, zUnit, c, r, k, f) {
    const g = 0.12;
    const P = (cc, rr, kk) => ({ x: ox + (cc - rr) * ISO_COS * u, y: oy + (cc + rr) * ISO_SIN * u - kk * zUnit });
    const c0 = c + g, c1 = c + 1 - g, r0 = r + g, r1 = r + 1 - g, k0 = k + g, k1 = k + 1 - g;
    const T0 = P(c0, r0, k1), T1 = P(c1, r0, k1), T2 = P(c1, r1, k1), T3 = P(c0, r1, k1);
    const E3 = P(c1, r0, k0), E2 = P(c1, r1, k0), S3 = P(c0, r1, k0);
    isoQuad(ctx, T1, E3, E2, T2, f.east, f.edge, 1);
    isoQuad(ctx, T3, T2, E2, S3, f.south, f.edge, 1);
    isoQuad(ctx, T0, T1, T2, T3, f.top, f.edge, 1);
  }

  function drawTensorCallout(ctx, width, height, h, options = {}) {
    if (!h || !h.label) return;
    const x = options.x ?? 24;
    const y = options.y ?? 34;
    ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = getCss('--foreground-secondary');
    ctx.font = '700 13px ui-monospace, monospace';
    ctx.fillText(h.label, x, y);
    if (h.sub) {
      ctx.fillStyle = getCss('--foreground-muted');
      ctx.font = '600 11px Inter, sans-serif';
      ctx.fillText(h.sub, x, y + 22);
    }
  }

  // 1D logical tensor (vector): the GM buffer drawn as an iso row of block
  // cuboids; the current tile is a filled slice on the active block top face.
  function drawTensorStrip(ctx, width, height, visual) {
    const strip = visual.strip || {};
    const total = Math.max(1, strip.total);
    const segs = strip.segments || [];
    const cols = Math.max(1, segs.length);
    const blockLength = Math.max(1, strip.blockLength || Math.round(total / cols));
    const scale = state.tensorView.scale || 1;
    const panX = state.tensorView.panX || 0;
    const panY = state.tensorView.panY || 0;
    const ink = getCss('--foreground-secondary');
    const muted = getCss('--foreground-muted');

    const availW = Math.max(120, width - 120);
    const u = Math.max(12, availW / ((cols + 1) * ISO_COS)) * scale;
    const depthPx = 0.16 * u;
    const ox = width / 2 - ((cols - 1) / 2) * ISO_COS * u + panX;
    const oy = height / 2 - ((cols + 1) / 2) * ISO_SIN * u + panY;
    const P = (c, r, kpx) => ({ x: ox + (c - r) * ISO_COS * u, y: oy + (c + r) * ISO_SIN * u - kpx });

    const activeIdx = segs.findIndex((s) => s.active);
    for (let c = 0; c < cols; c += 1) {
      const active = c === activeIdx;
      const g = 0.04;
      const c0 = c + g, c1 = c + 1 - g, r0 = g, r1 = 1 - g;
      const T0 = P(c0, r0, depthPx), T1 = P(c1, r0, depthPx), T2 = P(c1, r1, depthPx), T3 = P(c0, r1, depthPx);
      const E3 = P(c1, r0, 0), E2 = P(c1, r1, 0), S3 = P(c0, r1, 0);
      const faces = active ? { top: 'rgba(255,207,89,0.10)', east: '#343434', south: '#2d2d2d', edge: 'rgba(220,230,240,0.16)' } : VOXEL_GRAY;
      isoQuad(ctx, T1, E3, E2, T2, faces.east, faces.edge, 1);
      isoQuad(ctx, T3, T2, E2, S3, faces.south, faces.edge, 1);
      isoQuad(ctx, T0, T1, T2, T3, faces.top, active ? VOXEL_TONES.compute.top : faces.edge, active ? 1.6 : 1);
      const lp = P(c + 0.5, 0.5, depthPx);
      ctx.fillStyle = active ? ink : muted;
      ctx.font = `${active ? '700' : '600'} 10px Inter, sans-serif`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(`b${c}`, lp.x, lp.y);
    }

    const h = visual.highlight || {};
    if (Array.isArray(h.x)) {
      const tone = voxelTone(h.tone);
      const g = 0.04;
      const c0 = h.x[0] / blockLength;
      const c1 = Math.max(c0 + 0.004, h.x[1] / blockLength);
      const T0 = P(c0, g, depthPx), T1 = P(c1, g, depthPx), T2 = P(c1, 1 - g, depthPx), T3 = P(c0, 1 - g, depthPx);
      isoQuad(ctx, T0, T1, T2, T3, hexToRgba(tone.top, 0.85), tone.top, 1.5);
    }

    ctx.fillStyle = muted; ctx.font = '600 9px ui-monospace, monospace';
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (let c = 0; c <= cols; c += 1) {
      const p = P(c, 1.45, 0);
      ctx.fillText(String(c * blockLength), p.x, p.y + 2);
    }
    const ap = P(cols / 2, 2.4, 0);
    ctx.fillStyle = ink; ctx.font = '700 11px Inter, sans-serif';
    ctx.fillText(`GM element offset →  (total ${total})`, ap.x, ap.y);

    drawTensorCallout(ctx, width, height, h);
  }

  // 3D iteration space (matmul/fusion): the full M×N×K volume as a voxel grid
  // at tile granularity; the active block's K-columns are highlighted and fill
  // upward as the K reduction accumulates (triton-viz style sub-block highlight).
  function drawTensorGrid(ctx, width, height, visual) {
    const grid = visual.grid || {};
    const rowCell = Math.max(1, grid.rowCell);
    const colCell = Math.max(1, grid.colCell);
    const tilesM = Math.max(1, Math.round(Math.max(1, grid.rowTotal) / rowCell));
    const tilesN = Math.max(1, Math.round(Math.max(1, grid.colTotal) / colCell));
    const kSteps = Math.max(1, Math.round(grid.kSteps || 1));
    const scale = state.tensorView.scale || 1;
    const panX = state.tensorView.panX || 0;
    const panY = state.tensorView.panY || 0;
    const ink = getCss('--foreground-secondary');
    const muted = getCss('--foreground-muted');

    const zRatio = 0.9;
    const availW = Math.max(120, width - 150);
    const availH = Math.max(120, height - 150);
    const fitU = Math.min(
      availW / ((tilesN + tilesM) * ISO_COS),
      availH / ((tilesN + tilesM) * ISO_SIN + kSteps * zRatio)
    );
    const u = Math.max(8, fitU) * scale;
    const zUnit = u * zRatio;
    const ox = width / 2 - ((tilesN - tilesM) / 2) * ISO_COS * u + panX;
    const oy = height / 2 - (((tilesN + tilesM) * ISO_SIN * u - kSteps * zUnit) / 2) + panY;

    const h = visual.highlight || {};
    const rs = Array.isArray(h.row) ? Math.floor(h.row[0] / rowCell) : -1;
    const re = Array.isArray(h.row) ? Math.round(h.row[1] / rowCell) : -1;
    const cs = Array.isArray(h.col) ? Math.floor(h.col[0] / colCell) : -1;
    const ce = Array.isArray(h.col) ? Math.round(h.col[1] / colCell) : -1;
    const isActive = (c, r) => c >= cs && c < ce && r >= rs && r < re;
    const kFill = visual.progress ? Math.max(0, Math.min(kSteps, Number(visual.progress.current) || 0)) : kSteps;
    const hi = voxelTone(h.tone);

    // gray voxels for the whole volume minus the active region, depth-sorted
    const cells = [];
    for (let r = 0; r < tilesM; r += 1) {
      for (let c = 0; c < tilesN; c += 1) {
        if (isActive(c, r)) continue;
        for (let k = 0; k < kSteps; k += 1) cells.push({ c, r, k });
      }
    }
    cells.sort((a, b) => (a.c + a.r + a.k) - (b.c + b.r + b.k));
    for (const cell of cells) drawVoxel(ctx, ox, oy, u, zUnit, cell.c, cell.r, cell.k, VOXEL_GRAY);

    // active output partition: filled columns = accumulated K, ghost = remaining
    if (cs >= 0) {
      for (let r = rs; r < re; r += 1) {
        for (let c = cs; c < ce; c += 1) {
          for (let k = 0; k < kSteps; k += 1) {
            drawVoxel(ctx, ox, oy, u, zUnit, c, r, k, k < kFill ? hi : VOXEL_GHOST);
          }
        }
      }
    }

    // axes ticks + names along the visible front edges
    const P = (c, r, kpx) => ({ x: ox + (c - r) * ISO_COS * u, y: oy + (c + r) * ISO_SIN * u - kpx });
    ctx.fillStyle = muted; ctx.font = '600 9px ui-monospace, monospace';
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (let c = 0; c <= tilesN; c += 1) {
      const p = P(c, tilesM + 0.45, 0);
      ctx.fillText(String(c * colCell), p.x, p.y + 2);
    }
    ctx.fillStyle = ink; ctx.font = '700 11px Inter, sans-serif';
    { const p = P(tilesN / 2, tilesM + 1.4, 0); ctx.fillText(`${grid.colLabel || 'N'} = ${grid.colTotal}`, p.x, p.y); }

    ctx.fillStyle = muted; ctx.font = '600 9px ui-monospace, monospace';
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    for (let r = 0; r <= tilesM; r += 1) {
      const p = P(tilesN + 0.45, r, 0);
      ctx.fillText(String(r * rowCell), p.x + 2, p.y + 4);
    }
    ctx.fillStyle = ink; ctx.font = '700 11px Inter, sans-serif';
    { const p = P(tilesN + 1.5, tilesM / 2, 0); ctx.fillText(`${grid.rowLabel || 'M'} = ${grid.rowTotal}`, p.x, p.y); }

    if (grid.kTotal) {
      const p = P(0, 0, kSteps * zUnit);
      ctx.fillStyle = VOXEL_TONES.reduction.top; ctx.font = '700 11px Inter, sans-serif';
      ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
      ctx.fillText(`${grid.depthLabel || 'K'} = ${grid.kTotal} ↑`, p.x - 6, p.y - 4);
    }

    if (visual.progress) {
      ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
      ctx.fillStyle = muted; ctx.font = '600 11px Inter, sans-serif';
      ctx.fillText(`${visual.progress.label || 'K'} ${kFill}/${kSteps}`, 24, 34);
    }
    drawTensorCallout(ctx, width, height, h, { y: visual.progress ? 58 : 34 });
  }

  function drawTileLabels(ctx, width, tiles) {
    ctx.font = '600 11px Inter, sans-serif';
    tiles.slice(0, 3).forEach((tile, index) => {
      const tone = TENSOR_TONES[tile.tone || 'default'] || TENSOR_TONES.default;
      const y = 28 + index * 24;
      ctx.fillStyle = tone.fill;
      ctx.fillRect(width - 280, y - 12, 10, 10);
      ctx.strokeStyle = tone.stroke;
      ctx.strokeRect(width - 280, y - 12, 10, 10);
      ctx.fillStyle = getCss('--foreground-secondary');
      ctx.fillText(tile.label || `tile ${index + 1}`, width - 264, y - 3);
    });
  }

  function displayCoreName(core) {
    const value = String(core || '');
    if (value === 'mem950-aic') return 'AIC';
    if (value === 'mem950-aiv1') return 'AIV0';
    if (value === 'mem950-aiv2') return 'AIV1';
    return value.replace(/^mem950-/, '').toUpperCase() || 'core';
  }

  function displayBufferTarget(block) {
    const core = displayCoreName(block?.core);
    const buffer = block?.buffer || 'buffer';
    return `${core} · ${buffer}`;
  }

  function renderTileLens(trace) {
    const step = currentStep(trace);
    const visual = visualStateForStep(trace, step);
    const supportsLens = state.activeTensorRenderer === 'legacy'
      || state.activeTensorRenderer === 'copy-input-pattern';
    els.tileLens.classList.toggle('avz-tile-lens--single', state.activeTensorRenderer === 'copy-input-pattern');
    if (!supportsLens) {
      els.tileLens.innerHTML = '';
      els.tileLens.hidden = true;
      return;
    }
    els.tileLens.hidden = false;
    if (trace?.operator?.kind === 'conv2d-cube' && step?.tensorSnapshots?.length) {
      const tabs = tensorTabsForStep(trace, step);
      const snapshots = tabs.length > 1
        ? step.tensorSnapshots.filter((snapshot) => snapshot.tensorId === state.tensorTabKey)
        : step.tensorSnapshots;
      els.tileLens.innerHTML = snapshots.slice(0, 3).map((snapshot) => renderSnapshotLensCard(trace, snapshot)).join('');
      return;
    }
    const blocks = visual.onChipLens?.blocks || visual.architectureFocus?.bufferBlocks || [];
    if (!blocks.length) {
      els.tileLens.innerHTML = '';
      return;
    }
    els.tileLens.innerHTML = blocks.slice(0, 3).map((block) => `
      <div class="avz-lens-card" title="${escapeHtml(block.state || 'loaded')} · ${escapeHtml(block.sourceTile || '')}">
        <header class="avz-lens-card__head">
          <span>${escapeHtml(block.label || block.buffer)}</span>
          <span>${escapeHtml(displayBufferTarget(block))}</span>
        </header>
        <div class="avz-lens-grid">${renderLensCells(block)}</div>
        <div class="avz-card-meta">${escapeHtml(block.state || 'loaded')} · ${escapeHtml(block.sourceTile || '')}</div>
      </div>
    `).join('');
  }

  function renderSnapshotLensCard(trace, snapshot) {
    const definition = [...(trace?.tensors || []), ...(trace?.buffers || [])]
      .find((item) => item.id === snapshot.tensorId);
    const title = definition?.name || snapshot.tensorId?.replace(/^(tensor|buffer):/, '') || 'Tensor';
    const shape = formatShape(snapshot.logicalShape);
    const bytes = snapshot.validBytes ?? snapshot.allocatedBytes;
    const confidence = snapshot.confidence || 'unknown';
    const tone = snapshotTone(snapshot);
    return `
      <div class="avz-lens-card avz-lens-card--snapshot" title="${escapeHtml(snapshot.transformation || '')}">
        <header class="avz-lens-card__head">
          <span>${escapeHtml(title)}</span>
          <span>${escapeHtml(snapshot.location || 'unknown')}</span>
        </header>
        <div class="avz-snapshot-line">
          <strong>${escapeHtml(shape)}</strong>
          <span>${escapeHtml(String(snapshot.dtype || '').toUpperCase())}</span>
        </div>
        <div class="avz-snapshot-meter" aria-hidden="true">
          ${Array.from({ length: 12 }, (_, index) => `<span class="${index < snapshotFillCount(snapshot) ? `is-active is-${tone}` : ''}"></span>`).join('')}
        </div>
        <div class="avz-card-meta">
          <span>${escapeHtml(formatBytes(bytes))}</span>
          <span class="avz-confidence is-${escapeHtml(confidence)}">${escapeHtml(confidence)}</span>
        </div>
      </div>
    `;
  }

  function snapshotTone(snapshot) {
    const role = String(snapshot?.role || '');
    if (role.includes('output')) return 'output';
    if (role.includes('accumulator') || role.includes('bias')) return 'accumulator';
    return 'input';
  }

  function snapshotFillCount(snapshot) {
    const valid = Number(snapshot?.validElements);
    const allocated = Number(snapshot?.allocatedElements);
    if (Number.isFinite(valid) && Number.isFinite(allocated) && allocated > 0) {
      return Math.max(1, Math.min(12, Math.round((valid / allocated) * 12)));
    }
    return snapshot?.confidence === 'unknown' ? 4 : 12;
  }

  function formatShape(shape) {
    return Array.isArray(shape) && shape.length ? `[${shape.join('×')}]` : 'shape unknown';
  }

  function formatBytes(value) {
    if (value == null) return 'bytes unknown';
    const bytes = Number(value);
    if (!Number.isFinite(bytes)) return 'bytes unknown';
    if (bytes >= 1024 && bytes % 1024 === 0) return `${bytes / 1024} KiB`;
    return `${bytes} B`;
  }

  function renderLensCells(block) {
    const count = lensCellCount(block);
    const active = new Set(cellRange(block, count));
    return Array.from({ length: count }, (_, index) => (
      `<span class="${active.has(index) ? `is-active is-${escapeHtml(block.tone || 'input')}` : ''}"></span>`
    )).join('');
  }

  function lensCellCount(block) {
    if (!Array.isArray(block?.cellRange)) return 32;
    const end = Number(block.cellRange[1]);
    if (!Number.isFinite(end)) return 32;
    return end >= 32 ? 64 : 32;
  }

  function cellRange(block, count) {
    if (Array.isArray(block.cellRange)) {
      const start = Math.max(0, Math.min(count - 1, Number(block.cellRange[0] || 0)));
      const end = Math.max(start, Math.min(count - 1, Number(block.cellRange[1] ?? start)));
      return Array.from({ length: end - start + 1 }, (_, offset) => start + offset);
    }
    return [];
  }

  function ensureArchitectureMounted() {
    if (!els.architectureFrame) return false;
    if (state.architecture.mounted) return true;

    const markReady = (payload = {}) => {
      applyArchitectureFrameSize(payload);
      if (state.architecture.frameReady) return;
      state.architecture.frameReady = true;
      bindArchitectureFramePan();
      postArchitectureDetails();
      postArchitectureScale();
      postArchitectureFocus(state.architecture.pendingFocus || {});
    };

    window.addEventListener('message', (event) => {
      if (event.source !== els.architectureFrame.contentWindow) return;
      if (event.data?.type === 'hardware-ready') {
        markReady(event.data);
        return;
      }
      if (event.data?.type === 'hardware-size') {
        applyArchitectureFrameSize(event.data);
        postArchitectureScale();
      }
    });
    els.architectureFrame.addEventListener('load', () => window.requestAnimationFrame(markReady));
    syncArchitectureControls();
    window.requestAnimationFrame(() => {
      try {
        const readyState = els.architectureFrame.contentDocument?.readyState;
        if (readyState === 'interactive' || readyState === 'complete') markReady();
      } catch {
        // Cross-origin protections can block inspection; load/message still handle the normal local case.
      }
    });
    state.architecture.mounted = true;
    return true;
  }

  function postArchitectureFocus(focus = {}) {
    if (!els.architectureFrame?.contentWindow) return;
    els.architectureFrame.contentWindow.postMessage({
      type: 'hardware-focus',
      selectors: focus.selectors || [],
      routes: focus.routes || focus.routeIds || [],
      instructionTags: focus.instructionTags || [],
      ...(Object.prototype.hasOwnProperty.call(focus, 'foldAiv') ? { foldAiv: focus.foldAiv } : {}),
    }, '*');
  }

  function renderArchitectureFocus(trace) {
    const mounted = ensureArchitectureMounted();
    const visual = visualStateForStep(trace, currentStep(trace)).architectureFocus || {};
    const focus = {
      selectors: visual.selectors || [],
      routes: visual.routes || visual.routeIds || [],
      instructionTags: visual.instructionTags || [],
    };
    state.architecture.pendingFocus = focus;
    if (els.architectureKicker) els.architectureKicker.textContent = '';
    if (mounted) postArchitectureFocus(focus);
    if (els.architectureBlocks) {
      els.architectureBlocks.hidden = true;
      els.architectureBlocks.innerHTML = '';
    }
  }

  function instructionLoopGroup(trace) {
    const steps = trace?.steps || [];
    const groupIndex = steps.findIndex((step) => (
      Array.isArray(step?.loop?.childStepIds)
      && Array.isArray(step?.loop?.iterationRange)
      && step.loop.iterationRange.length === 2
    ));
    if (groupIndex < 0) return null;
    const groupStep = steps[groupIndex];
    const childIndexes = groupStep.loop.childStepIds
      .map((id) => steps.findIndex((step) => step.id === id))
      .filter((index) => index >= 0);
    return {
      groupIndex,
      groupStep,
      childIndexes,
      allIndexes: [groupIndex, ...childIndexes],
      iterationStart: Number(groupStep.loop.iterationRange[0]),
      iterationEnd: Number(groupStep.loop.iterationRange[1]),
    };
  }

  function childStepIndexForIteration(trace, loopGroup, iteration) {
    const steps = trace?.steps || [];
    const matchingIndex = loopGroup.childIndexes.find((index) => {
      const range = steps[index]?.loop?.iterationRange;
      return Array.isArray(range) && iteration >= Number(range[0]) && iteration <= Number(range[1]);
    });
    return matchingIndex ?? loopGroup.groupIndex;
  }

  function instructionPanelModel(trace) {
    const steps = trace?.steps || [];
    const loopGroup = instructionLoopGroup(trace);
    if (!loopGroup) {
      return {
        before: steps.map((step, stepIndex) => instructionStepCard(trace, step, stepIndex)),
        loop: null,
        after: [],
      };
    }

    const iter0StageIds = new Set(['load-k', 'sync-mte1-m', 'mmad-init']);
    const iter0Indexes = steps
      .map((step, stepIndex) => ({ step, stepIndex }))
      .filter(({ step, stepIndex }) => (
        stepIndex < loopGroup.groupIndex
        && Number(step?.loop?.kIndex) === 0
        && iter0StageIds.has(step.stageId)
      ))
      .map(({ stepIndex }) => stepIndex);
    const loopIndexes = new Set([...iter0Indexes, ...loopGroup.allIndexes]);
    const firstLoopIndex = Math.min(...loopIndexes);
    const lastLoopIndex = Math.max(...loopIndexes);
    const before = [];
    const after = [];

    steps.forEach((step, stepIndex) => {
      if (loopIndexes.has(stepIndex)) return;
      const card = instructionStepCard(trace, step, stepIndex);
      if (stepIndex < firstLoopIndex) before.push(card);
      if (stepIndex > lastLoopIndex) after.push(card);
    });

    return {
      before,
      loop: {
        ...loopGroup,
        iter0Cards: iter0Indexes.map((stepIndex) => instructionStepCard(trace, steps[stepIndex], stepIndex)),
      },
      after,
    };
  }

  function instructionStepCard(trace, step, stepIndex) {
    const stage = trace?.stages?.find((item) => item.id === step.stageId);
    const titleOverrides = {
      'sync-mte2-mte1': 'MTE2_MTE1 Sync',
      'load-k': 'Load Data A2/B2',
      'sync-mte1-m': 'MTE1_M Sync',
      'mmad-init': 'Mmad + Bias',
    };
    return {
      key: step.stageId,
      title: titleOverrides[step.stageId] || timelineStageTitle(stage, step),
      flow: timelineStageFlow(stage, step),
      stepIndex,
      sourceStepIndexes: [stepIndex],
      iteration: Number.isInteger(step?.loop?.kIndex) ? Number(step.loop.kIndex) : null,
    };
  }

  function repeatedIterationCards(stepIndex, iteration) {
    return [
      {
        key: 'm-mte1',
        title: 'M_MTE1 Sync',
        flow: 'Cube → MTE1',
        stepIndex,
        sourceStepIndexes: [stepIndex],
        iteration,
      },
      {
        key: 'load-a2-b2',
        title: 'Load Data A2/B2',
        flow: `A[Mi,K${iteration}] / B[K${iteration},Nj]`,
        stepIndex,
        sourceStepIndexes: [stepIndex],
        iteration,
      },
      {
        key: 'mte1-m',
        title: 'MTE1_M Sync',
        flow: 'MTE1 → Cube',
        stepIndex,
        sourceStepIndexes: [stepIndex],
        iteration,
      },
      {
        key: 'mmad-accumulate',
        title: 'Mmad Accumulate',
        flow: `A[Mi,K${iteration}] × B[K${iteration},Nj] + Acc${iteration - 1} → Acc${iteration}`,
        stepIndex,
        sourceStepIndexes: [stepIndex],
        iteration,
      },
    ];
  }

  function renderInstructionPanel(trace) {
    if (!trace || !els.instructionSequence || state.executionView !== 'instructions') return;
    renderConvCoreContext(trace);
    const model = instructionPanelModel(trace);
    const track = document.createElement('div');
    track.className = 'avz-instruction-track';

    model.before.forEach((card) => track.appendChild(createInstructionCard(card)));
    if (model.loop) track.appendChild(createLoopGroup(trace, model.loop));
    model.after.forEach((card) => track.appendChild(createInstructionCard(card)));

    els.instructionSequence.replaceChildren(track);
  }

  function createLoopGroup(trace, loop) {
    const group = document.createElement('section');
    group.className = 'avz-instruction-loop';
    group.setAttribute('role', 'listitem');
    group.setAttribute('aria-label', 'K Loop, iterations 0 through 8');
    if (loop.allIndexes.includes(state.stepIndex) || loop.iter0Cards.some((card) => card.stepIndex === state.stepIndex)) {
      group.classList.add('is-active');
    }

    const title = document.createElement('div');
    title.className = 'avz-instruction-loop__title';
    title.textContent = 'K Loop';
    group.appendChild(title);

    const rows = document.createElement('div');
    rows.className = 'avz-instruction-loop__rows';
    rows.appendChild(createInstructionRow({
      label: 'Iter 0',
      meta: 'Initialize',
      cards: loop.iter0Cards,
      sourceStepIndexes: loop.iter0Cards.map((card) => card.stepIndex),
    }));

    const repeatGroup = document.createElement('div');
    repeatGroup.className = 'avz-instruction-repeat';
    const repeatHeader = document.createElement('div');
    repeatHeader.className = 'avz-instruction-repeat__header';
    const repeatLabel = createIterationLabel(
      `Iter ${loop.iterationStart}–${loop.iterationEnd}`,
      `×${loop.iterationEnd - loop.iterationStart + 1}`,
    );
    const toggle = document.createElement('button');
    toggle.className = 'btn btn-sm btn-ghost avz-instruction-loop-toggle';
    toggle.type = 'button';
    toggle.dataset.loopToggle = 'iterations';
    toggle.setAttribute('aria-expanded', String(state.instructionLoopExpanded));
    toggle.textContent = state.instructionLoopExpanded ? 'Group similar' : 'Show all';
    repeatHeader.append(repeatLabel, toggle);
    repeatGroup.appendChild(repeatHeader);

    if (state.instructionLoopExpanded) {
      const expandedRows = document.createElement('div');
      expandedRows.className = 'avz-instruction-repeat__rows';
      for (let iteration = loop.iterationStart; iteration <= loop.iterationEnd; iteration += 1) {
        const stepIndex = childStepIndexForIteration(trace, loop, iteration);
        const kStart = iteration * 16;
        const kEnd = (iteration + 1) * 16;
        expandedRows.appendChild(createInstructionRow({
          label: `Iter ${iteration}`,
          meta: `K[${kStart}:${kEnd}] · Acc${iteration}`,
          cards: repeatedIterationCards(stepIndex, iteration),
          sourceStepIndexes: [stepIndex],
          iteration,
        }));
      }
      repeatGroup.appendChild(expandedRows);
    } else {
      const representativeIteration = Number.isInteger(state.instructionIterationFocus)
        && state.instructionIterationFocus >= loop.iterationStart
        && state.instructionIterationFocus <= loop.iterationEnd
        ? state.instructionIterationFocus
        : loop.iterationStart;
      const stepIndex = childStepIndexForIteration(trace, loop, representativeIteration);
      repeatGroup.appendChild(createInstructionRow({
        label: '',
        meta: '',
        cards: repeatedIterationCards(stepIndex, representativeIteration).map((card) => ({
          ...card,
          sourceStepIndexes: loop.allIndexes,
          iterationRange: [loop.iterationStart, loop.iterationEnd],
        })),
        sourceStepIndexes: loop.allIndexes,
        iteration: representativeIteration,
        hideLabel: true,
      }));
    }

    rows.appendChild(repeatGroup);
    group.appendChild(rows);
    return group;
  }

  function createInstructionRow({
    label,
    meta,
    cards,
    sourceStepIndexes,
    iteration = null,
    hideLabel = false,
  }) {
    const row = document.createElement('div');
    row.className = 'avz-instruction-row';
    if (hideLabel) row.classList.add('is-label-hidden');
    row.setAttribute('role', 'group');
    row.setAttribute('aria-label', label || `Iterations ${iteration || ''}`);
    const rangeSelected = sourceStepIndexes.includes(state.stepIndex);
    const explicitIteration = Number.isInteger(state.instructionIterationFocus);
    const iterationSelected = !explicitIteration || iteration === state.instructionIterationFocus;
    if (rangeSelected && iterationSelected) row.classList.add('is-active');

    if (!hideLabel) row.appendChild(createIterationLabel(label, meta));

    const flow = document.createElement('div');
    flow.className = 'avz-instruction-row__flow';
    cards.forEach((card) => flow.appendChild(createInstructionCard(card)));
    row.appendChild(flow);
    return row;
  }

  function createIterationLabel(label, meta) {
    const wrapper = document.createElement('div');
    wrapper.className = 'avz-instruction-row__label';
    const tag = document.createElement('span');
    tag.className = 'tag avz-iteration-tag';
    tag.textContent = label;
    wrapper.appendChild(tag);
    if (meta) {
      const detail = document.createElement('span');
      detail.className = 'avz-instruction-row__meta';
      detail.textContent = meta;
      wrapper.appendChild(detail);
    }
    return wrapper;
  }

  function createInstructionCard(card) {
    const selectedStep = card.sourceStepIndexes.includes(state.stepIndex);
    const selectedIteration = !Number.isInteger(state.instructionIterationFocus)
      || card.iteration == null
      || card.iteration === state.instructionIterationFocus;
    const selectedOperation = !state.instructionOperationFocus || state.instructionOperationFocus === card.key;
    const selected = selectedStep && selectedIteration && selectedOperation;
    const button = document.createElement('button');
    button.className = 'avz-instruction-card';
    button.type = 'button';
    button.dataset.stepIndex = String(card.stepIndex);
    button.dataset.instructionOperation = card.key;
    if (Number.isInteger(card.iteration)) button.dataset.instructionIteration = String(card.iteration);
    if (Array.isArray(card.iterationRange)) {
      button.dataset.instructionIterationStart = String(card.iterationRange[0]);
      button.dataset.instructionIterationEnd = String(card.iterationRange[1]);
    }
    if (selected) {
      button.classList.add('is-selected');
      button.setAttribute('aria-current', 'step');
    }

    const title = document.createElement('span');
    title.className = 'avz-instruction-card__title';
    title.textContent = card.title;
    const flow = document.createElement('span');
    flow.className = 'avz-instruction-card__flow';
    flow.textContent = card.flow;
    button.append(title, flow);
    return button;
  }

  function handleInstructionSequenceClick(event) {
    const toggle = event.target.closest('[data-loop-toggle]');
    if (toggle) {
      setInstructionLoopExpanded(!state.instructionLoopExpanded);
      return;
    }
    const card = event.target.closest('[data-step-index]');
    if (!card) return;
    const iteration = Number(card.dataset.instructionIteration);
    const iterationStart = Number(card.dataset.instructionIterationStart);
    const iterationEnd = Number(card.dataset.instructionIterationEnd);
    const hasIterationRange = Number.isInteger(iterationStart) && Number.isInteger(iterationEnd);
    selectStep(Number(card.dataset.stepIndex), {
      instructionIteration: hasIterationRange ? null : (Number.isInteger(iteration) ? iteration : null),
      instructionIterationRange: hasIterationRange ? [iterationStart, iterationEnd] : null,
      instructionOperation: card.dataset.instructionOperation || null,
    });
  }

  function drawFittedText(ctx, text, x, y, maxWidth) {
    const value = String(text || '');
    if (ctx.measureText(value).width <= maxWidth) {
      ctx.fillText(value, x, y);
      return;
    }
    let clipped = value;
    while (clipped.length > 1 && ctx.measureText(`${clipped}...`).width > maxWidth) {
      clipped = clipped.slice(0, -1);
    }
    ctx.fillText(`${clipped}...`, x, y);
  }

  function timelineStageTitle(stage, step) {
    if (step?.label) return zh(step.label);
    const map = {
      'host-launch': 'Host 启动',
      init: 'Tiling 初始化',
      'copy-in': 'CopyIn',
      compute: 'Compute',
      'copy-out': 'CopyOut',
      'gm-offset': 'GM Offset',
      'copy-in-a': 'CopyIn A',
      'copy-in-b': 'CopyIn B',
      'load-data': 'LoadData',
      mmad: 'Mmad',
      fixpipe: 'Fixpipe',
      'mix-launch': 'Mix 启动',
      'aic-matmul': 'AIC Matmul',
      'cross-core-sync': '同步',
      'aiv-leakyrelu': 'AIV LeakyRelu',
    };
    return map[stage?.id] || zh(stage?.label || step?.label || 'trace');
  }

  function timelineStageFlow(stage, step) {
    const id = stage?.id || step?.stageId || '';
    const map = {
      'host-shape': 'Host config',
      'host-tiling': 'Host config',
      'host-launch': 'Host/Runtime config',
      allocate: 'Kernel prepare',
      'copy-inputs': 'GM -> L1',
      'sync-mte2-mte1': 'MTE2 -> MTE1',
      'bias-c1-c2': 'C1 -> C2',
      'load-k': 'L1 -> L0A/B',
      'sync-mte1-m': 'MTE1 -> Cube',
      'mmad-init': 'A2/B2 + Bias -> CO1',
      'k-loop': 'Loop container ×8',
      'loop-body-middle': 'Sync + Load + Mmad ×7',
      'loop-body-final': 'Sync + Load + Mmad ×1',
      'sync-m-fix': 'Cube -> Fixpipe',
      'fixpipe-output': 'L0C -> GM',
      init: 'block/tile',
      'copy-in': 'GM -> UB',
      compute: 'UB -> SIMD',
      'copy-out': 'UB -> GM',
      'gm-offset': 'blockIdx -> C',
      'copy-in-a': 'GM A -> L1',
      'copy-in-b': 'GM B -> L1',
      'load-data': 'L1 -> L0',
      mmad: 'L0A/B -> L0C',
      fixpipe: 'L0C -> GM',
      'mix-launch': 'AIC:AIV = 1:2',
      'aic-matmul': 'A/B -> C',
      'cross-core-sync': 'flag',
      'aiv-leakyrelu': 'GM <-> UB',
    };
    return map[id] || unitLabel(step?.unit || stage?.unit) || 'trace';
  }

  const cssCache = new Map();
  function getCss(name) {
    if (cssCache.has(name)) return cssCache.get(name);
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    cssCache.set(name, value);
    return value;
  }

  function renderInfoPanel(trace = currentTrace()) {
    if (els.traceInfoPanel) els.traceInfoPanel.hidden = !state.infoOpen;
    if (els.viewportInfo) {
      els.viewportInfo.setAttribute('aria-expanded', state.infoOpen ? 'true' : 'false');
      els.viewportInfo.classList.toggle('is-selected', state.infoOpen);
    }
    if (!state.infoOpen) return;
    const step = currentStep(trace);
    const stage = trace?.stages?.find((item) => item.id === step?.stageId);
    if (!step || !stage || !els.traceInfoContent) return;
    const visual = visualStateForStep(trace, step);
    const axes = visual.tensorViewport?.axisLabels || [];
    const blocks = visual.architectureFocus?.bufferBlocks || [];
    if (els.traceInfoMeta) {
      els.traceInfoMeta.textContent = `${state.stepIndex + 1}/${trace.steps.length} · ${zh(stage.label)}`;
    }
    const axisText = axes.length ? `当前轴名是 ${formatListCn(axes)}。` : '';
    els.traceInfoContent.innerHTML = `
      <section class="avz-info-panel__section">
        <p class="avz-info-panel__eyebrow">Tensor View</p>
        <h3>Conv2D Output Tile + K Reduction</h3>
        <p>${escapeHtml(`${axisText}${tensorSceneNarrative(trace, step, stage, visual)}`)}</p>
      </section>
      <section class="avz-info-panel__section">
        <p class="avz-info-panel__eyebrow">Logical Execution Sequence</p>
        <h3>${escapeHtml(timelineStageTitle(stage, step))}</h3>
        <p>${escapeHtml(timelineInfoNarrative(trace, step, stage))}</p>
      </section>
      <section class="avz-info-panel__section">
        <p class="avz-info-panel__eyebrow">Memory Architecture</p>
        <h3>硬件链路和片上 buffer</h3>
        <p>${escapeHtml(memoryArchitectureInfoNarrative(visual, blocks))}</p>
      </section>
    `;
  }

  function tensorSceneNarrative(trace, step, stage, visual) {
    const kind = trace?.operator?.kind || visual.tensorViewport?.kind || 'vector';
    const intro = `中央视口参考 Triton-Viz 的 trace-driven 逻辑：画的是完整的 logical tensor，高亮块是当前这一步实际触碰的 element 区间。坐标轴有真实刻度和单位（element 数），不是抽象的折叠维度。GM 始终是线性地址，shape、blockIdx 和循环偏移决定高亮落在哪里。静止画面只是当前帧；播放或切换步骤时，高亮块会跟着 CopyIn、Compute、CopyOut 或同步阶段移动。`;

    if (kind === 'vector') {
      return `${intro} 这个 Vector Add 是 1D tensor，所以整条横轴就是 GM 线性地址 0 → totalLength，按 numBlocks 切成等长 block 段。当前 tile 的高亮区间 = blockIdx * blockLength + progress * tileLength 起、长 tileLength 个 element，刻度上能直接读出它对应的 GM 偏移。`;
    }

    if (kind === 'cube') {
      return `${intro} 这个 Cube Matmul 的输出是二维 C[M,N]，视口就把它画成真实的 M×N 网格，每格是一个 baseM×baseN 的输出 tile。高亮块是当前 Cube block 负责的 singleCoreM×singleCoreN 输出分区。K 是规约维、不是输出 tensor 的轴，所以单独用右侧的 K 累加进度条表示：Mmad 沿 K 把部分和累加到 L0C/CO1，Fixpipe 再把完成的 C tile 写回 GM。`;
    }

    if (kind === 'conv2d-cube') {
      const c = visual.tensorViewport?.conv || {};
      const scene = c.scene || 'overview';
      const base = `M=${c.ho}×${c.wo}=${c.M} 是输出位置，N=${c.N} 是输出通道，K=${c.ci}×${c.kh}×${c.kw}=${c.K} 只表示规约范围，不是 Y 的第三个轴。`;
      if (scene === 'copy-in') {
        const transfer = c.copyTransfer;
        const flow = transfer?.flow || {};
        const snapshot = transfer?.snapshot || {};
        const destination = transfer?.destination || {};
        const source = transfer?.source || {};
        const core = transfer?.partition || {};
        const coreSource = transfer?.coreSource || {};
        const shape = formatShape(snapshot.physicalShape || snapshot.logicalShape || destination.logicalShape);
        const sourceLayout = String(source.physicalLayout || source.logicalLayout || 'unknown').split(' ')[0];
        const destinationLayout = snapshot.physicalLayout || destination.physicalLayout || 'unknown';
        return `${intro} ${base} 当前是 AIC${core.index ?? 0} / OT${core.outputTile ?? 0} / M${core.mTile ?? 0}/N${core.nTile ?? 0}，页签显示 ${coreSource.slice || flow.from || 'GM input'}（GM + ${coreSource.gmOffsetBytes || 0} B）→ ${flow.to || destination.location || 'L1'}：${sourceLayout} ${shape} 经 ${flow.transferEngine || 'MTE2 / DataCopy'} 写入 ${destinationLayout}，共 ${formatBytes(flow.bytes)}。切换核时 Instruction 步骤保持不变，只比较该步骤下各核的 GM 范围与本地搬运结果。此时状态是 Copied by MTE2，但 MTE1 仍被阻塞；只有下一步 MTE2_MTE1 完成后，A1/B1/C1 才成为 MTE1 可读输入。`;
      }
      if (scene === 'bias-copy') {
        return `${intro} ${base} 当前只表达下一阶段的 Bias C1→C2：MTE2_MTE1 已使 C1 可读，MTE1 再搬运 16 个 FP32 值、共 64 B 到 C2 / Bias Table，供第一次 Mmad 使用。它不属于上一阶段的 GM→L1 Copy Inputs。`;
      }
      if (scene === 'load3d') {
        if (state.tensorTabKey === 'buffer:weight:b2') {
          return `${intro} ${base} 当前页签显示 weightB1→weightB2：从 B1 NZ [K=144,N=16] 选中 K${c.kIndex} [${c.kRange?.[0]}:${c.kRange?.[1]}] 分形，经 LoadData2D 的转置语义生成 B2 ZN [${c.tileK},${c.tileN}]。`;
        }
        return `${intro} ${base} 当前页签显示 fmapA1→fmapA2：左侧是物理 A1 NC1HWC0 [1,1,${c.hi},${c.wi},16]，蓝色格是代表性输出位置在 A1 内实际命中的值；越界部分由 LoadData3D 的 padList/padValue 虚拟生成，不扩张 A1。右侧把当前 K[${c.kRange?.[0]}:${c.kRange?.[1]}] 展开成 A2[${c.tileM},${c.tileK}]。`;
      }
      if (scene === 'mmad') {
        return `${intro} ${base} 当前是 K${c.kIndex}：A2[${c.tileM},${c.tileK}] × B2[${c.tileK},${c.tileN}] → CO1[${c.tileM},${c.tileN}]。I0 从 C2 加入 Bias；I1～I8 只累加 partial sum，不重复加入 Bias。`;
      }
      if (scene === 'epilogue') {
        return `${intro} ${base} 完成的 CO1[${c.tileM},${c.tileN}] 由 Fixpipe 直接写入 GM：NZ→ND、FP32→FP16 并融合 ReLU；输出物理视图是 [64,32]，等价 NHWC，不是 NCHW。API 能否在目标 CANN 编译运行仍需验证。`;
      }
      if (scene === 'copy-out') {
        return `${intro} ${base} 当前 tile 覆盖 M[0:${c.tileM}]、Co[0:${c.tileN}]，写入 Y[1,${c.co},${c.ho},${c.wo}]；helper 没有函数体，因此精确 GM offset 和 block 映射保持 unknown。`;
      }
      if (scene === 'event') {
        return `${intro} ${base} 当前画面表达 SetFlag/WaitFlag 的 producer→consumer 依赖；这条边用于阻止下游过早读取，不表示 Tensor 搬运路径。`;
      }
      return `${intro} ${base} 固定参考源码明确给出 M×K×N tile = ${c.tileM}×${c.tileK}×${c.tileN}、4×9×2 个 tile 与 blockDim=8；它是代码事实，但不是自动 tiling 或性能最优结论。`;
    }

    if (kind === 'fusion') {
      return `${intro} 这个 Fusion 同样画 C[M,N] 真实网格：AIC/Cube 先生产一块 singleCore 的 C 分区，CrossCoreSetFlag 把它标记为 ready，AIV0 和 AIV1 再分别消费这块分区的上半和下半 M rows 做 LeakyRelu——所以 AIV 步骤的高亮只覆盖一半行。AIC/AIV 的 handoff 是同步关系，要看底部 Execution Timeline 和右侧 Memory Architecture 的链路高亮，而不是 tensor 的某个轴。`;
    }

    return intro;
  }

  function timelineInfoNarrative(trace, step, stage) {
    const refs = sourceRefsForStep(step).map((ref) => `${ref.fileId}:${(ref.lines || []).join(',')}`).join('；');
    const sourceText = refs ? `源码 ${refs}` : '当前源码片段';
    return `底部时间线按 trace step 展示执行顺序，当前步骤是 ${state.stepIndex + 1}/${trace.steps.length}：${zh(step.label)}。阶段数据流是 ${timelineStageFlow(stage, step)}，对应 ${sourceText}。播放只是在这些离散步骤之间移动当前帧，高亮块和右侧链路会跟着当前步骤切换。`;
  }

  function memoryArchitectureInfoNarrative(visual, blocks) {
    const focus = visual.architectureFocus || {};
    const routes = focus.routes || focus.routeIds || [];
    const routeText = routes.length
      ? `右侧 Memory Architecture 高亮 ${formatListCn(routes)} 这些硬件链路，用来表达当前步骤的数据搬运或跨核同步路径。`
      : '右侧 Memory Architecture 只展示当前步骤涉及的硬件单元，没有额外高亮跨单元链路。';
    const blockText = blocks.length
      ? `buffer grid 中额外标出的 data block 是片上局部驻留，例如 ${formatListCn(blocks.map((block) => `${displayBufferTarget(block)} ${block.label || ''}`))}；它们不是完整 logical tensor grid。`
      : '当前步骤没有需要单独标出的片上 buffer data block。';
    return `${routeText}${blockText}`;
  }

  function formatListCn(items) {
    const values = (items || []).filter(Boolean).map(String);
    if (values.length <= 1) return values[0] || '无';
    if (values.length === 2) return `${values[0]} 和 ${values[1]}`;
    return `${values.slice(0, -1).join('、')} 和 ${values[values.length - 1]}`;
  }

  function unitLabel(unit) {
    const labels = {
      host: 'Host',
      vector: 'Vector',
      cube: 'Cube',
      aic: 'AIC',
      aiv: 'AIV',
      sync: '同步',
      mixed: '混合',
    };
    return labels[unit] || unit || '';
  }

  async function init() {
    initButtons();
    try {
      await loadTraces();
      window.PtoIdeFrame?.initAll?.();
      initPlayback();
      render();
      initResizeObservers();
      window.addEventListener('resize', () => {
        const trace = currentTrace();
        renderTensorViewport(trace);
        refreshArchitectureViewport();
      });
    } catch (error) {
      if (els.statusText) els.statusText.textContent = error.message;
      console.error(error);
    }
  }

  function initResizeObservers() {
    if (state.resizeObserver || typeof ResizeObserver !== 'function') return;
    state.resizeObserver = new ResizeObserver(() => {
      if (state.resizeRaf) return;
      state.resizeRaf = window.requestAnimationFrame(() => {
        state.resizeRaf = 0;
        const trace = currentTrace();
        renderTensorViewport(trace);
        refreshArchitectureViewport();
      });
    });
    [els.tensorStage, els.architectureViewport].forEach((target) => {
      if (target) state.resizeObserver.observe(target);
    });
  }

  document.addEventListener('DOMContentLoaded', init);
})();
