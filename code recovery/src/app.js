(() => {
  'use strict';

  // ---- Synthetic Tensor Data (fixed seed 20260805) ----
  var syntheticData = (function() {
    var SEED = 20260805;

    function mulberry32(a) {
      return function() {
        a |= 0; a = a + 0x6D2B79F5 | 0;
        var t = Math.imul(a ^ a >>> 15, 1 | a);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
      };
    }

    var rand = mulberry32(SEED);

    // X0 logical shape: [1, 16, 8, 8] → value range [-1, 1]
    var x0Data = {};
    function getX0Value(n, c, h, w) {
      var key = n + ',' + c + ',' + h + ',' + w;
      if (x0Data[key] !== undefined) return x0Data[key];
      // Use deterministic seed based on coordinates
      var subSeed = SEED + n * 10000 + c * 1000 + h * 100 + w;
      var r = mulberry32(subSeed)();
      var value = Math.round((r * 2 - 1) * 100) / 100; // round to 2 decimals in [-1, 1]
      x0Data[key] = value;
      return value;
    }

    // fmapA1 value = X0 value (DataCopy doesn't change values)
    // Physical: NC1HWC0 -> c1=floor(c/16), c0=c%16
    function getA1Value(n, c1, h, w, c0) {
      var c = c1 * 16 + c0;
      return getX0Value(n, c, h, w);
    }

    return {
      getX0Value: getX0Value,
      getA1Value: getA1Value,
      seed: SEED
    };
  })();


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
    timer: null,
    playback: null,
    webglAvailable: null,
    infoOpen: false,
    selectedTensorId: null,
    dataViewMode: 'data-dump',
    selectedChannel: 0,
    selectedDataElement: null,
    animationStep: 0,
    animationPlaying: false,
    animationTimer: null,
    animationState: {
      H: 8, W: 8, Ci: 16, C0: 16, filterH: 3, filterW: 3,
      strideH: 1, strideW: 1, padTop: 1, padRight: 1, padBottom: 1, padLeft: 1,
      padValue: 0, dilationH: 1, dilationW: 1, repeatMode: 0, repeatTime: 1, showChannelPadding: true },
    sourceExplorerWidth: 350,
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
    titleControllers: {},
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
    loadDataBControllers: {
      source: null,
      target: null,
    },
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
    tensorSection: byId('tensorSection'),
    tensorStage: byId('tensorStage'),
    sourceExplorerPane: byId('standalone-explorer-pane'),
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
    hostTilingSourceCanvas: byId('hostTilingSourceCanvas'),
    hostTilingWeightCount: byId('hostTilingWeightCount'),
    hostTilingWeightCountValue: byId('hostTilingWeightCountValue'),
    hostTilingTransform: byId('hostTilingTransform'),
    hostTilingCubeCanvas: byId('hostTilingCubeCanvas'),
    hostTilingFormula: byId('hostTilingFormula'),
    hostLaunchView: byId('hostLaunchView'),
    hostLaunchEquation: byId('hostLaunchEquation'),
    hostLaunchEvidence: byId('hostLaunchEvidence'),
    hostLaunchBlockDim: byId('hostLaunchBlockDim'),
    hostLaunchACanvas: byId('hostLaunchACanvas'),
    hostLaunchBCanvas: byId('hostLaunchBCanvas'),
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
    copyInputSourceCanvas: byId('copyInputSourceCanvas'),
    copyInputEngine: byId('copyInputEngine'),
    copyInputTransformation: byId('copyInputTransformation'),
    copyInputDestinationCanvas: byId('copyInputDestinationCanvas'),
    copyInputLensMount: byId('copyInputLensMount'),
    biasC1C2View: byId('biasC1C2View'),
    biasC1C2Summary: byId('biasC1C2Summary'),
    biasC1C2Context: byId('biasC1C2Context'),
    biasC1C2Engine: byId('biasC1C2Engine'),
    biasC1Canvas: byId('biasC1Canvas'),
    biasC2Canvas: byId('biasC2Canvas'),
    fixpipeOutputView: byId('fixpipeOutputView'),
    fixpipeOutputSummary: byId('fixpipeOutputSummary'),
    fixpipeOutputContext: byId('fixpipeOutputContext'),
    fixpipeAccumCanvas: byId('fixpipeAccumCanvas'),
    fixpipeOutputCanvas: byId('fixpipeOutputCanvas'),
    fixpipeAddressCore: byId('fixpipeAddressCore'),
    fixpipeAddress: byId('fixpipeAddress'),
    convMmadMatrixView: byId('convMmadMatrixView'),
    mmadEquation: byId('mmadEquation'),
    mmadEvidence: byId('mmadEvidence'),
    mmadProgress: byId('mmadProgress'),
    mmadA2Canvas: byId('mmadA2Canvas'),
    mmadB2Canvas: byId('mmadB2Canvas'),
    mmadAddend: byId('mmadAddend'),
    mmadCo1Canvas: byId('mmadCo1Canvas'),
    mmadBiasStatus: byId('mmadBiasStatus'),
    addendMatrixCanvas: byId('mmadAddendCanvas'),
    tensorCanvas: byId('tensorCanvas'),
    convLoadDataView: byId('convLoadDataView'),
    fmapA1VolumeCanvas: byId('fmapA1VolumeCanvas'),
    a2LogicalMatrixCanvas: byId('a2LogicalMatrixCanvas'),
    loadDataBView: byId('loadDataBView'),
    weightB1MatrixCanvas: byId('weightB1MatrixCanvas'),
    weightB2MatrixCanvas: byId('weightB2MatrixCanvas'),
    loadDataBSummary: byId('loadDataBSummary'),
    loadDataBContext: byId('loadDataBContext'),
    loadDataBEngine: byId('loadDataBEngine'),
    loadDataBTransformation: byId('loadDataBTransformation'),
    loadDataBDetail: byId('loadDataBDetail'),
    loadDataBKRange: byId('loadDataBKRange'),
    loadDataBAddress: byId('loadDataBAddress'),
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
    tensorJourneyView: byId('tensorJourneyView'),
    tensorJourneyContent: byId('tensorJourneyContent'),
    tensorDataDumpPanel: byId('tensorDataDumpPanel'),
    dataDumpTensorInfo: byId('dataDumpTensorInfo'),
    dataDumpVolumeCanvas: byId('dataDumpVolumeCanvas'),
    dataDumpVolumeMeta: byId('dataDumpVolumeMeta'),
    channelControls: byId('channelControls'),
    channelSelect: byId('channelSelect'),
    prevChannelBtn: byId('prevChannelBtn'),
    nextChannelBtn: byId('nextChannelBtn'),
    channelTotal: byId('channelTotal'),
    heatmapGrid: byId('heatmapGrid'),
    heatmapHeader: byId('heatmapHeader'),
    dataDumpBody: byId('dataDumpBody'),
    dataDumpHeatmap: byId('dataDumpHeatmap'),
    channelControls: byId('channelControls'),
    animateTagBtn: byId('animateTagBtn'),
    animationBody: byId('animationBody'),
    animationA1Canvas: byId('animationA1Canvas'),
    animationA1Status: byId('animationA1Status'),
    animationA2Grid: byId('animationA2Grid'),
    animationA2Meta: byId('animationA2Meta'),
    animationA2Pills: byId('animationA2Pills'),
    animationPlaybackStatus: byId('animationPlaybackStatus'),
    cellInspector: byId('cellInspector'),
    cellInspectorBody: byId('cellInspectorBody'),
    closeInspectorBtn: byId('closeInspectorBtn'),
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
    // Tensor Data Dump event listeners
    els.prevChannelBtn?.addEventListener('click', function() {
      if (state.selectedChannel > 0) {
        state.selectedChannel -= 1;
        renderTensorDataDump(currentTrace());
      }
    });
    els.nextChannelBtn?.addEventListener('click', function() {
      if (state.selectedChannel < 15) {
        state.selectedChannel += 1;
        renderTensorDataDump(currentTrace());
      }
    });
    els.channelSelect?.addEventListener('change', function() {
      state.selectedChannel = parseInt(els.channelSelect.value, 10) || 0;
      renderTensorDataDump(currentTrace());
    });
    els.closeInspectorBtn?.addEventListener('click', function() {
      state.selectedDataElement = null;
      els.cellInspector.hidden = true;
    });
    // Data Dump tab buttons
    var dumpTabs = document.querySelectorAll('[data-dump-tab]');
    dumpTabs.forEach(function(btn) {
      btn.addEventListener('click', function() {
        var mode = btn.dataset.dumpTab;
        if (mode === 'dump') mode = 'data-dump';
        state.dataViewMode = mode;
        dumpTabs.forEach(function(b) {
          b.classList.toggle('is-selected', b.dataset.dumpTab === btn.dataset.dumpTab);
          b.setAttribute('aria-selected', String(b.dataset.dumpTab === btn.dataset.dumpTab));
        });
        // Stop animation when switching away from animation tab
        if (mode !== 'animation') stopAnimationPlayback();
        // Start animation when switching to animation tab
        if (mode === 'animation') startAnimationPlayback();
        renderTensorDataDump(currentTrace());
      });
    });

    // Animate tag button: switch to Animation tab
    if (els.animateTagBtn) {
      els.animateTagBtn.addEventListener('click', function() {
        state.dataViewMode = 'animation';
        var dumpTabs = document.querySelectorAll('[data-dump-tab]');
        dumpTabs.forEach(function(b) {
          b.classList.toggle('is-selected', b.dataset.dumpTab === 'animation');
          b.setAttribute('aria-selected', String(b.dataset.dumpTab === 'animation'));
        });
        startAnimationPlayback();
        renderTensorDataDump(currentTrace());
      });
    }

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
    if (view === 'tensor-journey') state.executionView = 'tensor-journey';
    else if (view === 'timeline') state.executionView = 'timeline';
    else state.executionView = 'instructions';

    // Switching away from tensor-journey clears Data Dump mode
    if (view === 'instructions' || view === 'timeline') {
      if (window._tjResizeObserver) { window._tjResizeObserver.disconnect(); window._tjResizeObserver = null; }
      if (state.selectedTensorId) {
        state.selectedTensorId = null;
        state.selectedDataElement = null;
        state.dataViewMode = 'data-dump';
        stopAnimationPlayback();
        // Restore standard views immediately
        renderDataDumpPanelVisibility(false);
        var trace = currentTrace();
        renderTensorTabs(trace);
        renderTensorViewport(trace);
        renderTileLens(trace);
        renderArchitectureFocus(trace);
      }
    }

    renderExecutionDock();
    if (state.executionView === 'instructions') {
      renderInstructionPanel(currentTrace());
    }
    if (state.executionView === 'tensor-journey') {
      renderTensorJourney(currentTrace());
    }
  }

  function renderDataDumpPanelVisibility(visible) {
    // Hide the entire tensor section during data dump, instead of hiding inner children.
    if (els.tensorSection) els.tensorSection.hidden = visible;
    if (els.tensorStage) els.tensorStage.hidden = visible;
    if (els.tensorTabs) els.tensorTabs.hidden = visible;
    if (els.tensorDataDumpPanel) els.tensorDataDumpPanel.hidden = !visible;

    // Hide hardware participation panel during data dump
    if (els.architectureViewportRoot) {
      els.architectureViewportRoot.style.display = visible ? 'none' : '';
    }

    // Update pane title
    if (els.visualTitle) {
      els.visualTitle.textContent = visible ? 'Tensor' : 'Tensor State & Transformation';
    }
  }

  function renderExecutionDock() {
    els.executionTabs.forEach((button) => {
      const selected = button.dataset.executionView === state.executionView;
      button.classList.toggle('is-selected', selected);
      button.setAttribute('aria-selected', String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
    if (els.sourceExplorerPane) {
      if (!Number.isFinite(state.sourceExplorerWidth) || state.sourceExplorerWidth <= 0) {
        state.sourceExplorerWidth = 350;
      }
      const width = `${state.sourceExplorerWidth}px`;
      els.sourceExplorerPane.style.flex = `0 0 ${width}`;
      els.sourceExplorerPane.style.width = width;
      els.sourceExplorerPane.style.minWidth = width;
      els.sourceExplorerPane.style.maxWidth = width;
    }
    if (els.instructionsView) els.instructionsView.hidden = state.executionView !== 'instructions';
    if (els.tensorJourneyView) els.tensorJourneyView.hidden = state.executionView !== 'tensor-journey';
    if (els.estimatedTimelineView) els.estimatedTimelineView.hidden = state.executionView !== 'timeline';
    if (els.timelineKicker) {
      const step = currentStep();
      if (state.executionView === 'instructions') {
        els.timelineKicker.textContent = `${step?.evidenceKind || 'unknown'} · logical order · repeated iterations grouped`;
      } else if (state.executionView === 'tensor-journey') {
        els.timelineKicker.textContent = 'Tensor data flow by memory location';
      } else {
        els.timelineKicker.textContent = 'Estimated Timeline unavailable · Not Profiling Data';
      }
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
    if (state.activeTensorRenderer === 'load-data-b') {
      state.loadDataBControllers.source?.fit?.();
      state.loadDataBControllers.target?.fit?.();
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

    // Data Dump mode: show tensor panel, hide standard views
    var isDataDump = state.selectedTensorId === 'buffer:feature:a1' || state.selectedTensorId === 'fmapA1';
    renderDataDumpPanelVisibility(isDataDump);

    if (isDataDump) {
      renderTensorDataDump(trace);
    } else {
      renderTensorTabs(trace);
      renderTensorViewport(trace);
      renderTileLens(trace);
      renderArchitectureFocus(trace);
    }

    renderExecutionDock();
    renderInstructionPanel(trace);
    if (state.executionView === 'tensor-journey') {
      renderTensorJourney(trace);
    }
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
    if (state.instructionOperationFocus === 'load-a2-b2') {
      return [
        { key: 'buffer:feature:a2', label: 'fmapA2', location: 'A2 / L0A' },
        { key: 'buffer:weight:b2', label: 'weightB2', location: 'B2 / L0B' },
      ];
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
    const isFocusedMMte1Sync = state.instructionOperationFocus === 'm-mte1';
    const isFocusedLoad = state.instructionOperationFocus === 'load-a2-b2';
    const isFocusedMte1MSync = state.instructionOperationFocus === 'mte1-m';
    const isFocusedMmad = state.instructionOperationFocus === 'mmad-accumulate';
    const isFocusedIterationOperation = isFocusedMMte1Sync || isFocusedLoad || isFocusedMte1MSync || isFocusedMmad;
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
    const presentationStage = isFocusedMMte1Sync
      ? 'sync-m-mte1'
      : isFocusedMte1MSync
        ? 'sync-mte1-m'
        : stage;
    const finished = ['sync-m-fix', 'fixpipe-output'].includes(stage);
    const tracksK = ['load-k', 'mmad-init', 'loop-body-middle', 'loop-body-final'].includes(presentationStage);
    const kCurrent = finished ? kLoops
      : stage === 'k-loop' ? 1
      : tracksK ? Math.min(kLoops, kIndex + 1)
      : 0;
    const blocks = convBufferBlocks(presentationStage, kIndex);
    const scene = isFocusedMMte1Sync || isFocusedMte1MSync
      ? 'event'
      : isFocusedLoad
        ? 'load3d'
        : isFocusedMmad
          ? 'mmad'
          : convSceneForStage(stage);
    const event = isFocusedMMte1Sync
      ? (trace?.events || []).find((item) => item.eventType === 'M_MTE1') || null
      : isFocusedMte1MSync
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
    if (stage === 'sync-m-mte1') {
      const previousK = Math.max(0, kIndex - 1);
      return [
        { core: 'mem950-aic', buffer: 'L0A', label: `A2 · K${previousK} read complete`, state: 'read-complete', tone: 'input', cellRange: [0, 15], sourceTile: `safe to overwrite with K${kIndex}`, operation: 'M_MTE1' },
        { core: 'mem950-aic', buffer: 'L0B', label: `B2 · K${previousK} read complete`, state: 'read-complete', tone: 'input', cellRange: [0, 11], sourceTile: `safe to overwrite with K${kIndex}`, operation: 'M_MTE1' },
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
    if (stage === 'sync-m-mte1') return ['#mem950-aic [data-aic-node="buffer:L0A"]', '#mem950-aic [data-aic-node="buffer:L0B"]', '#mem950-aic [data-aic-node="cube:CUBE"]'];
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
    const useLoadDataBPatterns = isConvLoad
      && state.tensorTabKey === 'buffer:weight:b2'
      && !!window.PtoMatrixCanvas
      && !!els.loadDataBView
      && !!els.weightB1MatrixCanvas
      && !!els.weightB2MatrixCanvas;
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
      : useLoadDataBPatterns
        ? 'load-data-b'
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
    if (renderer === 'load-data-b') {
      renderConvLoadDataBView(trace, visual.conv);
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
    if (els.loadDataBView) els.loadDataBView.hidden = renderer !== 'load-data-b';
    if (els.tensorCanvas) els.tensorCanvas.hidden = renderer !== 'legacy';
    if (els.tileLens) els.tileLens.hidden = renderer !== 'legacy' && renderer !== 'copy-input-pattern';
  }

  function renderTensorTitle(key, scene, options) {
    const host = document.getElementById(`${key}TitleMount`);
    if (!host || !window.PtoTensorTitle) return;
    const controller = state.titleControllers[key];
    const nextOptions = options || {};
    if (controller) {
      controller.update(scene, nextOptions);
      return;
    }
    state.titleControllers[key] = window.PtoTensorTitle.render(host, scene, nextOptions);
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

  const ALLOCATION_LANES = [
    {
      id: 'l1',
      title: 'L1',
      capacityBytes: 512 * 1024,
      tensorIds: ['buffer:feature:a1', 'buffer:weight:b1', 'buffer:bias:c1'],
    },
    {
      id: 'l0a',
      title: 'L0A',
      capacityBytes: 64 * 1024,
      tensorIds: ['buffer:feature:a2'],
    },
    {
      id: 'l0b',
      title: 'L0B',
      capacityBytes: 64 * 1024,
      tensorIds: ['buffer:weight:b2'],
    },
    {
      id: 'bias-table',
      title: 'Bias Table',
      capacityBytes: 64 * 1024,
      tensorIds: ['buffer:bias:c2'],
    },
    {
      id: 'l0c',
      title: 'L0C',
      capacityBytes: 512 * 1024,
      tensorIds: ['buffer:accum:co1'],
    },
  ];
  const ALLOCATION_VISIBLE_RATIO = 80;

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

  function allocationBlock(tensor, referenceBytes) {
    if (!tensor) return '';
    const startRatio = referenceBytes > 0 ? (tensor.start / referenceBytes) * 100 : 0;
    const sizeRatio = referenceBytes > 0 ? (tensor.size / referenceBytes) * 100 : 0;
    return `<div class="avz-memory-block" tabindex="0" data-allocation-tensor="${escapeHtml(tensor.id)}" style="--avz-block-start:${startRatio}%;--avz-block-size:${sizeRatio}%" aria-label="${escapeHtml(`${tensor.name}, ${tensor.position}, [${tensor.start},${tensor.end}), ${tensor.size} bytes`)}"></div>`;
  }

  function allocationCapacityLabel(bytes) {
    if (bytes % (1024 * 1024) === 0) return `${bytes / (1024 * 1024)} MiB`;
    if (bytes % 1024 === 0) return `${bytes / 1024} KiB`;
    return `${bytes} B`;
  }

  function allocationUsageLabel(usedBytes, capacityBytes) {
    const ratio = capacityBytes > 0 ? (usedBytes / capacityBytes) * 100 : 0;
    return `${ratio.toFixed(1)}%`;
  }

  function allocationTicks(tensors, referenceBytes, capacityBytes) {
    const ticks = [{ address: tensors[0]?.start || 0, position: 0 }];
    tensors.forEach((tensor) => {
      ticks.push({
        address: tensor.end,
        position: referenceBytes > 0 ? Math.min(100, (tensor.end / referenceBytes) * 100) : 0,
      });
    });
    return `<div class="avz-memory-scale__used">${ticks.map((tick, index) => {
      const previousPosition = ticks[index - 1]?.position;
      const nextPosition = ticks[index + 1]?.position;
      const isCrowded = index > 0 && (
        (previousPosition !== undefined && tick.position - previousPosition < 18)
        || (nextPosition !== undefined && nextPosition - tick.position < 18)
      );
      const edgeClass = [
        index === 0 ? 'is-start' : '',
        index === ticks.length - 1 ? 'is-reference-end' : '',
        isCrowded && index % 2 === 1 ? 'is-staggered' : '',
      ].filter(Boolean).map((name) => ` ${name}`).join('');
      return `<span class="avz-memory-tick${edgeClass}" style="--avz-tick-position:${tick.position}%"><code>${tick.address}</code></span>`;
    }).join('')}</div>
      <div class="avz-memory-scale__capacity"><span class="avz-memory-tick is-capacity-end"><code>${capacityBytes}</code></span></div>`;
  }

  function allocationLane(lane, tensorsById, referenceBytes) {
    const tensors = lane.tensorIds.map((id) => tensorsById[id]).filter(Boolean);
    if (!tensors.length) return '';
    const usedBytes = Math.max(...tensors.map((tensor) => tensor.end));
    return `<section class="avz-memory-lane avz-memory-lane--${escapeHtml(lane.id)}" aria-label="${escapeHtml(`${lane.title} address space`)}">
      <div class="avz-memory-lane__map">
        <strong class="avz-memory-lane__title">${escapeHtml(lane.title)}</strong>
        <div class="avz-memory-lane__plot">
          <div class="avz-memory-scale">
            ${allocationTicks(tensors, referenceBytes, lane.capacityBytes)}
          </div>
          <div class="avz-memory-capacity-bar">
            <div class="avz-memory-used">
              ${tensors.map((tensor) => allocationBlock(tensor, referenceBytes)).join('')}
            </div>
            <div class="avz-memory-collapsed" aria-label="Collapsed unused address space"><span>⋯</span></div>
          </div>
        </div>
      </div>
      <div class="avz-memory-lane__summary">
        <div class="avz-memory-lane__names">${tensors.map((tensor) => `<code>${escapeHtml(tensor.name)}</code>`).join('<span aria-hidden="true">·</span>')}</div>
        <div class="avz-memory-lane__capacity">
          <strong>${allocationUsageLabel(usedBytes, lane.capacityBytes)}</strong>
          <span aria-hidden="true"></span>
          <strong>${allocationCapacityLabel(lane.capacityBytes)}</strong>
        </div>
      </div>
    </section>`;
  }

  function renderMemoryAllocationMap(trace) {
    const ids = Object.keys(ALLOCATION_MEMORY_MODEL);
    const tensors = Object.fromEntries(ids.map((id) => [id, allocationTensor(trace, id)]));
    if (!tensors['buffer:feature:a1']) return;
    const referenceBytes = Math.max(...ALLOCATION_LANES.map((lane) => {
      const laneTensors = lane.tensorIds.map((id) => tensors[id]).filter(Boolean);
      return laneTensors.length ? Math.max(...laneTensors.map((tensor) => tensor.end)) : 0;
    }));
    els.memoryAllocationView.innerHTML = `
      <div class="avz-memory-lanes">
        ${ALLOCATION_LANES.map((lane) => allocationLane(lane, tensors, referenceBytes)).join('')}
      </div>
      <footer class="avz-memory-legend">
        <span><i class="avz-memory-legend__tensor"></i> Tensor 色块表示 LocalTensor 视图，数据尚未装载</span>
        <span><i class="avz-memory-legend__collapsed"></i> 斜线区域表示折叠的未使用容量，不按真实剩余比例绘制</span>
        <span>所有 Tensor 共用同一比例尺；当前最大占用 ${referenceBytes} B 映射为泳道宽度的 ${ALLOCATION_VISIBLE_RATIO}%。</span>
        <span>所有地址区间均为左闭右开 <code>[start, end)</code>。</span>
      </footer>
      <div class="avz-memory-tooltip" id="memoryAllocationTooltip" role="tooltip" hidden></div>`;

    els.memoryAllocationView.querySelectorAll('[data-allocation-tensor]').forEach((block) => {
      block.addEventListener('pointerenter', showAllocationTooltip);
      block.addEventListener('pointermove', positionAllocationTooltip);
      block.addEventListener('pointerleave', hideAllocationTooltip);
      block.addEventListener('focus', showAllocationTooltip);
      block.addEventListener('blur', hideAllocationTooltip);
    });
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
    if (els.fixpipeAddressCore) els.fixpipeAddressCore.textContent = coreLabel;
    if (els.fixpipeAddress) {
      els.fixpipeAddress.textContent = `outputGm + ${byteOffset} B · M[${rowStart}:${rowEnd}] · Co[${columnStart}:${columnEnd}] · row stride ${columns} half`;
    }
    renderTensorTitle('fixpipeAccum', {
      label: `accumCo1 · AIC${partition.index}`,
      role: 'reduction',
      logicalShape: { label: 'aggregate', dims: [tileRows, tileColumns] },
      dtype: 'FP32',
      format: 'NZ · CO1 / L0C',
      memory: { tier: 'L0C', sizeBytes: tileRows * tileColumns * 4 },
      state: 'produced',
      constraints: ['1 aggregate cell · 16×16 values'],
    });
    renderTensorTitle('fixpipeOutput', {
      label: 'outputGm',
      role: 'output',
      logicalShape: { label: 'ND [M,Co]', dims: [rows, columns] },
      dtype: 'FP16',
      format: 'ND',
      memory: { tier: 'GM', sizeBytes: rows * columns * 2 },
      state: 'ready',
      constraints: ['8 aggregate cells · each 16×16'],
      status: `${coreLabel} · M${rowStart}:${rowEnd} · Co${columnStart}:${columnEnd}`,
    });

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

    if (els.biasC1C2Summary) els.biasC1C2Summary.textContent = 'Bias C1 → C2 / Bias Table';
    if (els.biasC1C2Context) els.biasC1C2Context.textContent = `MTE1 · ${elements} × ${dtype} · ${formatBytes(bytes)}`;
    if (els.biasC1C2Engine) els.biasC1C2Engine.textContent = flow.transferEngine || 'MTE1 / DataCopy';
    renderTensorTitle('biasC1', {
      label: source.name || 'biasC1',
      role: 'bias',
      logicalShape: { label: 'linear', dims: [1, elements] },
      dtype,
      format: 'C1 / L1',
      memory: {
        tier: 'L1',
        sizeBytes: bytes,
        offset: Number.isFinite(Number(source.addressBytes)) ? Number(source.addressBytes) : undefined,
        alignment: Number.isFinite(Number(source.alignmentBytes)) ? Number(source.alignmentBytes) : undefined,
      },
      state: 'current',
      constraints: ['readable after MTE2_MTE1'],
    });
    renderTensorTitle('biasC2', {
      label: destination.name || 'biasC2',
      role: 'bias',
      logicalShape: { label: snapshot.physicalLayout || 'linear Bias Table', dims: [1, elements] },
      dtype,
      format: 'C2 / Bias Table',
      memory: {
        tier: 'Bias Table',
        sizeBytes: bytes,
        offset: Number.isFinite(Number(destination.addressBytes)) ? Number(destination.addressBytes) : undefined,
      },
      state: 'written',
      constraints: ['ready for first Mmad'],
    });

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
    if (els.copyInputEngine) els.copyInputEngine.textContent = flow?.transferEngine || 'MTE2 / DataCopy';
    if (els.copyInputTransformation) els.copyInputTransformation.textContent = transformation;
    const sourceDisplayShape = sourceRole === 'weight' ? sourcePhysicalShape : shape;
    const sourceSliceLabel = sourceRole === 'weight'
      ? `selected ${coreSource?.slice}`
      : coreSource?.slice || 'current slice';
    renderTensorTitle('copyInputSource', {
      label: coreSource?.label || source?.name || flow?.from || 'GM source',
      role: sourceRole === 'bias' ? 'broadcast' : sourceRole === 'weight' ? 'weight' : 'input',
      logicalShape: { label: sourceLayout, dims: sourceDisplayShape },
      dtype,
      format: sourceLayout,
      memory: { tier: 'GM', sizeBytes: bytes, offset: coreSource?.gmOffsetBytes || 0 },
      state: 'current',
      constraints: [sourceSliceLabel],
    });
    renderTensorTitle('copyInputDestination', {
      label: destination?.name || flow?.to || 'L1 buffer',
      role: 'scratch',
      logicalShape: { label: destinationLayout, dims: shape },
      dtype,
      format: destinationLayout,
      memory: {
        tier: destination?.location || 'L1',
        sizeBytes: bytes,
        offset: address,
        alignment,
      },
      state: 'written',
    });

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
    if (els.mmadAddend) els.mmadAddend.textContent = '+';
    renderTensorTitle('mmadA2', {
      label: `A[Mi,K${kIndex}] · A2`,
      role: 'input',
      logicalShape: { label: 'A2', dims: [tileM, tileK] },
      dtype: 'FP16',
      format: 'A2',
      memory: { tier: 'L0A' },
      state: 'current',
      status: `K ${kCurrent}/${kLoops}`,
    });
    renderTensorTitle('mmadB2', {
      label: `B[K${kIndex},Nj] · B2`,
      role: 'input',
      logicalShape: { label: 'B2', dims: [tileK, tileN] },
      dtype: 'FP16',
      format: 'B2',
      memory: { tier: 'L0B' },
      state: 'current',
      status: `K ${kCurrent}/${kLoops}`,
    });
    renderTensorTitle('mmadAddend', kIndex === 0
      ? {
          label: 'Bias[Nj] broadcast',
          role: 'broadcast',
          logicalShape: { label: 'broadcast', dims: [tileM, tileN] },
          dtype: 'FP32',
          format: 'C2 / Bias Table',
          memory: { tier: 'Bias Table' },
          state: 'current',
        }
      : {
          label: `Acc${kIndex - 1} · CO1`,
          role: 'reduction',
          logicalShape: { label: 'CO1', dims: [tileM, tileN] },
          dtype: 'FP32',
          format: 'CO1 / L0C',
          memory: { tier: 'L0C' },
          state: 'accumulating',
          constraints: ['previous partial sum'],
        });
    renderTensorTitle('mmadCo1', {
      label: `${result} · CO1`,
      role: 'output',
      logicalShape: { label: 'CO1', dims: [tileM, tileN] },
      dtype: 'FP32',
      format: 'CO1 / L0C',
      memory: { tier: 'L0C' },
      state: 'accumulating',
      status: `K ${kCurrent}/${kLoops}`,
    });
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
    const sourceTitleScene = key === 'host-tiling:feature'
      ? {
          label: 'Feature X',
          role: 'input',
          logicalShape: { label: 'NCHW', dims: [1, ci, hi, wi] },
          dtype: 'FP16',
          format: 'NCHW',
          memory: { tier: 'GM' },
          constraints: [config.sourceMeta],
        }
      : key === 'host-tiling:weight'
        ? {
            label: 'Weight W',
            role: 'weight',
            logicalShape: { label: 'OIHW', dims: [co, ci, kh, kw] },
            dtype: 'FP16',
            format: 'logical OIHW',
            memory: { tier: 'GM' },
            constraints: [config.sourceMeta],
          }
        : {
            label: 'Output Y',
            role: 'output',
            logicalShape: { label: 'NCHW', dims: [1, co, ho, wo] },
            dtype: 'FP16',
            format: 'NCHW semantic',
            memory: { tier: 'GM' },
            constraints: [config.sourceMeta],
          };
    const cubeTitleScene = key === 'host-tiling:feature'
      ? {
          label: 'Cube A',
          role: 'input',
          logicalShape: { label: 'logical matrix', dims: [M, K] },
          format: 'Cube A',
          constraints: [config.cubeMeta],
          status: config.tileCount,
        }
      : key === 'host-tiling:weight'
        ? {
            label: 'Cube B',
            role: 'weight',
            logicalShape: { label: 'logical matrix', dims: [K, N] },
            format: 'Cube B',
            constraints: [config.cubeMeta],
            status: config.tileCount,
          }
        : {
            label: 'Cube C',
            role: 'output',
            logicalShape: { label: 'logical matrix', dims: [M, N] },
            format: 'Cube C',
            constraints: [config.cubeMeta],
            status: config.tileCount,
          };
    renderTensorTitle('hostTilingSource', sourceTitleScene);
    renderTensorTitle('hostTilingCube', cubeTitleScene);
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
    renderTensorTitle('hostLaunchA', {
      label: 'Input X',
      role: 'input',
      logicalShape: { label: 'Cube A', dims: [M, K] },
      format: 'Cube A',
      constraints: [`${mTiles} M tiles × ${kTiles} K tiles`],
      status: `AIC${activeBlockIdx}`,
    });
    renderTensorTitle('hostLaunchB', {
      label: 'Weight W',
      role: 'weight',
      logicalShape: { label: 'Cube B', dims: [K, N] },
      format: 'Cube B',
      constraints: [`${kTiles} K tiles × ${nTiles} N tiles`],
      status: `AIC${activeBlockIdx}`,
    });
    renderTensorTitle('hostLaunchC', {
      label: 'Output Y',
      role: 'output',
      logicalShape: { label: 'Cube C', dims: [M, N] },
      format: 'Cube C',
      constraints: [`${mTiles}×${nTiles} = ${outputTileCount} output tiles`],
      status: `AIC${activeBlockIdx} owns M${activeMTile}/N${activeNTile}`,
    });
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
        title: {
          label: 'Feature X',
          role: 'input',
          logicalShape: { label: 'NCHW', dims: [1, ci, hi, wi] },
          dtype: 'FP16',
          format: 'NCHW',
          memory: { tier: 'GM' },
        },
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
        title: {
          label: 'Weight W',
          role: 'weight',
          logicalShape: { label: 'OIHW [Co,Ci,Kh,Kw]', dims: [co, ci, kh, kw] },
          dtype: 'FP16',
          format: 'logical OIHW',
          memory: { tier: 'GM' },
          constraints: ['representative filter volume'],
        },
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
        title: {
          label: 'Bias',
          role: 'broadcast',
          logicalShape: { label: 'one value per output channel', dims: [co] },
          dtype: 'FP32',
          format: 'GM',
          memory: { tier: 'GM', sizeBytes: co * 4 },
          constraints: ['broadcast across output channels'],
        },
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
        title: {
          label: 'Output Y',
          role: 'output',
          logicalShape: { label: 'NCHW semantic', dims: [1, co, ho, wo] },
          dtype: 'FP16',
          format: 'NCHW semantic',
          memory: { tier: 'GM' },
        },
      },
    };

    Object.entries(fixtures).forEach(([key, fixture]) => {
      if (!fixture.canvas || !fixture.api) return;
      if (fixture.title) renderTensorTitle(`${key}Overview`, fixture.title);
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
    const fmapA1Shape = [num(c.n, 1), 1, num(c.hi, 8), num(c.wi, 8), num(c.ci, 16)];
    const padLeft = num(c.padLeft, 1);
    const padTop = num(c.padTop, 1);
    const fmapA2TileText = isRangeSelection
      ? `M${mTile} × K${selectionStart}–K${selectionEnd} · ${selectionEnd - selectionStart + 1} tiles`
      : `M${mTile} × K${kTile} = ${c.tileM} × ${c.tileK}`;
    renderTensorTitle('fmapA1', {
      label: 'fmapA1',
      role: 'scratch',
      logicalShape: { label: 'NC1HWC0', dims: fmapA1Shape },
      dtype: 'FP16',
      format: 'A1 / L1',
      memory: { tier: 'L1' },
      constraints: [`padList=[${padLeft},${padLeft},${padTop},${padTop}]`, 'padValue=0'],
      status: `M${mTile} · K${kTile}`,
    });
    renderTensorTitle('a2Logical', {
      label: 'A2 逻辑展开矩阵',
      role: 'reduction',
      logicalShape: { label: 'M × K', dims: [num(c.M, 64), num(c.K, 144)] },
      format: 'A2 / L0A',
      memory: { tier: 'L0A' },
      constraints: ['每格代表一个 [tileM, tileK] tile'],
      status: `M${mTile} × K${kTile}`,
    });
    renderTensorTitle('fmapA2', {
      label: 'fmapA2',
      role: 'output',
      logicalShape: { label: 'tile', dims: [num(c.tileM, 16), num(c.tileK, 16)] },
      dtype: 'FP16',
      format: 'ZZ · A2 / L0A',
      memory: { tier: 'L0A', sizeBytes: 512 },
      status: fmapA2TileText,
    });
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

  function renderConvLoadDataBView(trace, c) {
    if (!window.PtoMatrixCanvas || !els.weightB1MatrixCanvas || !els.weightB2MatrixCanvas) return;
    const params = trace?.tiling?.params || {};
    const tileK = Math.max(1, Number(c?.tileK) || num(params.tileK, 16));
    const tileN = Math.max(1, Number(c?.tileN) || num(params.tileN, 16));
    const K = Math.max(1, Number(c?.K) || num(params.K, 144));
    const N = tileN;
    const kIndex = Math.max(0, Number(c?.kIndex) || 0);
    const kTiles = Math.ceil(K / tileK);
    const kTile = Math.min(kTiles - 1, kIndex);
    const kStart = kTile * tileK;
    const kEnd = Math.min(K, kStart + tileK);
    const selectedRow = kTile * tileK;

    if (els.loadDataBSummary) els.loadDataBSummary.textContent = `weightB1 → weightB2 · K${kTile}`;
    if (els.loadDataBContext) {
      els.loadDataBContext.textContent = `MTE1 / LoadData2D · B1 NZ → B2 ZN · ${tileK}×${tileN} FP16 · ${tileK * tileN * 2} B`;
    }
    if (els.loadDataBEngine) els.loadDataBEngine.textContent = 'MTE1 / LoadData2D';
    if (els.loadDataBTransformation) els.loadDataBTransformation.textContent = 'NZ → ZN';
    if (els.loadDataBDetail) {
      els.loadDataBDetail.textContent = `startIndex=${kStart} · ifTranspose=true`;
    }
    renderTensorTitle('weightB1Matrix', {
      label: 'weightB1',
      role: 'weight',
      logicalShape: { label: 'NZ', dims: [K, N] },
      dtype: 'FP16',
      format: 'NZ · B1 / L1',
      memory: { tier: 'L1', sizeBytes: K * N * 2 },
      constraints: [`${kTiles}×${Math.ceil(N / tileN)} aggregate cells`],
      status: `K${kTile} [${kStart}:${kEnd}]`,
    });
    renderTensorTitle('weightB2Matrix', {
      label: 'weightB2',
      role: 'weight',
      logicalShape: { label: 'ZN', dims: [tileK, tileN] },
      dtype: 'FP16',
      format: 'ZN · B2 / L0B',
      memory: { tier: 'L0B', sizeBytes: tileK * tileN * 2 },
      constraints: ['1 aggregate cell'],
      status: `K${kTile}`,
    });
    if (els.loadDataBKRange) els.loadDataBKRange.textContent = `K${kTile} [${kStart}:${kEnd}]`;
    if (els.loadDataBAddress) {
      els.loadDataBAddress.textContent = `weightB1 + ${kStart * tileN * 2} B → weightB2 @0 · ${tileK * tileN * 2} B`;
    }

    const fixtures = {
      source: {
        canvas: els.weightB1MatrixCanvas,
        scene: createWeightB1MatrixScene(K, N, tileK, tileN, selectedRow),
        ariaLabel: `weightB1 NZ matrix, K ${K} by N ${N}, grouped into ${kTiles} by ${Math.ceil(N / tileN)} aggregate cells; K${kTile} aggregate cell selected`,
      },
      target: {
        canvas: els.weightB2MatrixCanvas,
        scene: createWeightB2MatrixScene(tileK, tileN, kTile),
        ariaLabel: `weightB2 ZN matrix, K ${tileK} by N ${tileN}, one aggregate cell selected`,
      },
    };
    const matrixOptions = {
      showAxes: true,
      showGrid: true,
      interactive: true,
      showTooltip: true,
      autoFit: true,
      minZoom: 0.015,
      padding: { top: 30, right: 24, bottom: 38, left: 54 },
    };

    Object.entries(fixtures).forEach(([key, fixture]) => {
      const options = { ...matrixOptions, ariaLabel: fixture.ariaLabel };
      const controller = state.loadDataBControllers[key];
      if (controller) {
        controller.update(fixture.scene, options);
        controller.fit();
      } else {
        state.loadDataBControllers[key] = window.PtoMatrixCanvas.render(
          fixture.canvas,
          fixture.scene,
          options
        );
        state.loadDataBControllers[key].fit();
      }
    });
  }

  function createWeightB1MatrixScene(rows, columns, rowSpan, columnSpan, selectedRow) {
    const cells = [];
    for (let row = 0; row < rows; row += rowSpan) {
      for (let column = 0; column < columns; column += columnSpan) {
        const resolvedRowSpan = Math.min(rowSpan, rows - row);
        const resolvedColumnSpan = Math.min(columnSpan, columns - column);
        const selected = row === selectedRow && column === 0;
        cells.push({
          id: `weight-b1-${row}-${column}`,
          row,
          column,
          rowSpan: resolvedRowSpan,
          columnSpan: resolvedColumnSpan,
          tone: selected ? 'input' : 'neutral',
          style: 'aggregate',
          states: selected ? ['selected'] : [],
          summary: {
            rows: resolvedRowSpan,
            columns: resolvedColumnSpan,
            count: resolvedRowSpan * resolvedColumnSpan,
            intensity: selected ? 1 : 0.36,
          },
        });
      }
    }
    return {
      extent: { rows, columns },
      axes: { rows: `K=${rows}`, columns: `N=${columns}` },
      cells,
    };
  }

  function createWeightB2MatrixScene(rows, columns, kTile = 0) {
    return {
      extent: { rows, columns },
      axes: { rows: `K${kTile}=${rows}`, columns: `N=${columns}` },
      cells: [{
        id: 'weight-b2-0-0',
        row: 0,
        column: 0,
        rowSpan: rows,
        columnSpan: columns,
        tone: 'input',
        style: 'aggregate',
        states: ['selected'],
        summary: {
          rows,
          columns,
          count: rows * columns,
          intensity: 1,
        },
      }],
    };
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
    const eventType = event.eventType || 'Event';
    const previousK = Math.max(0, Number(c.kIndex || 0) - 1);
    const eventCopy = {
      MTE2_MTE1: {
        producerMeta: 'L1 writes complete',
        consumerMeta: 'L1 readable after wait',
        producerTone: 'input',
        consumerTone: 'input',
        footer: 'Synchronization completes readiness; it does not move tensor data',
      },
      M_MTE1: {
        producerMeta: `A2/B2 K${previousK} reads complete`,
        consumerMeta: `overwrite with K${c.kIndex} after wait`,
        producerTone: 'reduction',
        consumerTone: 'input',
        footer: 'Dependency only · A2/B2 remain in L0A/L0B',
      },
      MTE1_M: {
        producerMeta: `A2/B2 K${c.kIndex} loads complete`,
        consumerMeta: `K${c.kIndex} readable after wait`,
        producerTone: 'input',
        consumerTone: 'reduction',
        footer: 'Readiness only · no tensor data moves through the event edge',
      },
      M_FIX: {
        producerMeta: 'Acc8 accumulation complete',
        consumerMeta: 'CO1 readable after wait',
        producerTone: 'reduction',
        consumerTone: 'output',
        footer: 'Readiness only · CO1 remains in L0C',
      },
    }[eventType] || {
      producerMeta: 'upstream complete',
      consumerMeta: 'unblocked after wait',
      producerTone: 'input',
      consumerTone: 'compute',
      footer: 'Event edge is not a data-transfer path',
    };
    const boxW = width * 0.28;
    const boxH = Math.min(92, height - 58);
    const y = 30;
    drawConvObjectBox(ctx, 0, y, boxW, boxH, {
      label: event.producerEngine || 'Producer',
      shape: 'SetFlag',
      meta: eventCopy.producerMeta,
      tone: eventCopy.producerTone,
    });
    drawConvFlowArrow(ctx, boxW + 12, y + boxH / 2, width * 0.13, false);
    const eventX = width * 0.44;
    drawConvObjectBox(ctx, eventX, y, width * 0.18, boxH, { label: eventType, shape: 'dependency', meta: 'confirmed', tone: 'fusion' });
    drawConvFlowArrow(ctx, width * 0.64, y + boxH / 2, width * 0.10, false);
    drawConvObjectBox(ctx, width * 0.76, y, width * 0.24, boxH, {
      label: event.consumerEngine || 'Consumer',
      shape: 'WaitFlag',
      meta: eventCopy.consumerMeta,
      tone: eventCopy.consumerTone,
    });
    drawConvFooter(ctx, width, height, event.explanation || 'Execution dependency', eventCopy.footer);
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
    const useFocusedMMte1Blocks = trace?.operator?.kind === 'conv2d-cube'
      && state.instructionOperationFocus === 'm-mte1';
    if (!useFocusedMMte1Blocks && trace?.operator?.kind === 'conv2d-cube' && step?.tensorSnapshots?.length) {
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

  // ================================================================
  // Tensor Journey — unified config-driven data flow panel
  // ================================================================

  var TJ_CONFIG = {
    locations: [
      { id: 'gmInput',  title: 'GM', subtitle: 'Global Memory' },
      { id: 'l1',       title: 'L1', subtitle: 'Local Memory' },
      { id: 'a2b2',     title: 'A2 / B2', subtitle: 'Buffer Memory' },
      { id: 'co1',      title: 'CO1', subtitle: 'Compute Memory' },
      { id: 'gmOutput', title: 'GM', subtitle: 'Global Memory (Out)' },
    ],
    tensors: [
      { id: 'featureX0',  location: 'gmInput',  name: 'feature X0', shape: '[1, 1, 8, 8, 16]', dtype: 'FP16', size: '2,048 B', branch: 'feature' },
      { id: 'weightW',    location: 'gmInput',  name: 'weight W',   shape: '[144, 32]',       dtype: 'FP16', size: '9,216 B', branch: 'weight' },
      { id: 'fmapA1',     location: 'l1',       name: 'fmapA1',     shape: '[1, 1, 8, 8, 16]', dtype: 'FP16', size: '2,048 B', branch: 'feature' },
      { id: 'weightB1',   location: 'l1',       name: 'weightB1',   shape: '[144, 16]',       dtype: 'FP16', size: '4,608 B', branch: 'weight' },
      { id: 'biasC1',     location: 'l1',       name: 'biasC1',     shape: '[16]',            dtype: 'FP16', size: '32 B',   branch: 'bias' },
      { id: 'fmapA2',     location: 'a2b2',     name: 'fmapA2',     shape: '[16, 16]',        dtype: 'FP16', size: '512 B',  branch: 'feature' },
      { id: 'weightB2',   location: 'a2b2',     name: 'weightB2',   shape: '[16, 16]',        dtype: 'FP16', size: '512 B',  branch: 'weight' },
      { id: 'biasC2',     location: 'a2b2',     name: 'biasC2',     shape: '[16]',            dtype: 'FP16', size: '32 B',   branch: 'bias' },
      { id: 'accumCo1',   location: 'co1',      name: 'accumCo1',   shape: '[16, 16]',        dtype: 'FP16', size: '512 B',  branch: 'accumulation' },
      { id: 'outputGM',   location: 'gmOutput', name: 'outputGM',   shape: '[64, 32]',        dtype: 'FP16', size: '4,096 B', branch: 'output' },
    ],
    connections: [
      { from: 'featureX0', to: 'fmapA1',   op: 'DataCopy',                                          branch: 'feature' },
      { from: 'fmapA1',    to: 'fmapA2',   op: 'ND2NZ',                                             branch: 'feature' },
      { from: 'weightW',   to: 'weightB1', op: 'DataCopy',                                          branch: 'weight' },
      { from: 'weightB1',  to: 'weightB2', op: 'LoadData3D',                                        branch: 'weight' },
      { from: 'weightB2',  to: 'accumCo1', op: 'LoadData2D',                                        branch: 'weight' },
      { from: 'biasC1',    to: 'biasC2',   op: 'LoadData2D',                                        branch: 'bias' },
      { from: 'biasC2',    to: 'accumCo1', op: 'Bias Init',                                         branch: 'bias' },
      { from: 'fmapA2',    to: 'accumCo1', op: 'Mmad',           isMerge: true, mergeOf: ['fmapA2','weightB2','biasC2'], branch: 'feature' },
      { from: 'accumCo1',  to: 'outputGM', op: 'K0\u2013K8 Accumulate \u00b7 Fixpipe \u00b7 ReLU \u00b7 FP32\u2192FP16 \u00b7 Write GM', branch: 'output' },
    ],
    branchColors: {
      feature: '#35DDF5',
      weight: '#22c55e',
      bias: '#a855f7',
      accumulation: '#FA8838',
      output: '#35DDF5',
      mmad: '#4d97ff',
    },
    branchOrder: ['feature', 'weight', 'bias', 'accumulation', 'output'],
  };

  var TJ_STATE = {
    selectedTensorId: 'fmapA1',
    highlightTensorId: null,
    showFeaturePath: true,
    showWeightPath: true,
  };

  function renderTensorJourney(trace) {
    if (!trace || !els.tensorJourneyContent || state.executionView !== 'tensor-journey') return;

    var cfg = TJ_CONFIG;
    var tjs = TJ_STATE;

    // Ensure default selection; all paths visible (no auto-highlight)
    if (!tjs.selectedTensorId) tjs.selectedTensorId = 'fmapA1';

    var tensorById = {};
    cfg.tensors.forEach(function(t) { tensorById[t.id] = t; });

    // ---- Build Canvas ----
    var canvasHtml = '<div class="avz-journey-canvas-wrap" id="tjCanvasWrap">';
    canvasHtml += '<div class="avz-journey-canvas" id="tjCanvas">';

    cfg.locations.forEach(function(loc, locIdx) {
      var tensors = cfg.tensors.filter(function(t) { return t.location === loc.id; });

      canvasHtml += '<div class="avz-journey-region" data-location="' + escapeHtml(loc.id) + '">';
      // Region header
      canvasHtml += '<div class="avz-journey-region-header">';
      canvasHtml += '<div class="avz-journey-region-icon-base"><svg class="avz-journey-region-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">' + hardwareIconPath(loc.id) + '</svg></div>';
      canvasHtml += '<div class="avz-journey-region-text-group">';
      canvasHtml += '<div class="avz-journey-region-title-row">';
      canvasHtml += '<span class="avz-journey-region-title">' + escapeHtml(loc.title) + '</span>';
      canvasHtml += '</div>';
      canvasHtml += '<span class="avz-journey-region-subtitle">' + escapeHtml(loc.subtitle) + '</span>';
      canvasHtml += '</div>';
      canvasHtml += '</div>';
      // Tracks
      // 多输入区域只保留 feature / weight / bias 三条对齐轨道；
      // CO1 与输出 GM 各自只保留一条结果轨道，避免无意义的空轨道压缩卡片。
      var alignBranches = ['feature', 'weight', 'bias'];
      var isMultiCard = loc.id === 'gmInput' || loc.id === 'l1' || loc.id === 'a2b2';
      var isSingleCard = loc.id === 'co1' || loc.id === 'gmOutput';
      canvasHtml += '<div class="avz-journey-tracks">';
      var renderBranches = isMultiCard
        ? alignBranches
        : [loc.id === 'co1' ? 'accumulation' : 'output'];
      renderBranches.forEach(function(branch) {
        var branchTensors = tensors.filter(function(t) { return t.branch === branch; });
        // 单卡片区域：只渲染有 tensor 的分支
        if (isSingleCard && branchTensors.length === 0) return;
        // 多卡片区域：空占位保持等高度以便跨区对齐
        var isEmpty = branchTensors.length === 0;
        var isCo1Center = loc.id === 'co1' && branch === 'accumulation';
        var isGmOutCenter = loc.id === 'gmOutput' && branch === 'output';
        var trackClass = 'avz-journey-track' + (isCo1Center || isGmOutCenter ? ' center' : '');
        if (isEmpty) trackClass += ' empty';
        canvasHtml += '<div class="' + trackClass + '" data-branch="' + branch + '">';
        if (isEmpty) {
          canvasHtml += '</div>';
          return;
        }
        branchTensors.forEach(function(t) {
          var isSelected = t.id === tjs.selectedTensorId;
          var selectedClass = isSelected ? ' is-selected' : '';
          canvasHtml += '<div class="avz-journey-card' + selectedClass + '" data-tensor-id="' + escapeHtml(t.id) + '" data-branch="' + escapeHtml(branch) + '">';
          canvasHtml += '<div class="avz-journey-card-head">';
          canvasHtml += '<span class="avz-journey-card-name ' + escapeHtml(branch) + '">' + escapeHtml(t.name) + '</span>';
          canvasHtml += '<span class="avz-journey-card-shape" title="' + escapeHtml(t.shape) + '">' + escapeHtml(t.shape) + '</span>';
          canvasHtml += '</div>';
          canvasHtml += '<span class="avz-journey-card-field"><span class="avz-journey-card-key">DType</span><span class="avz-journey-card-value">' + escapeHtml(t.dtype) + '</span></span>';
          canvasHtml += '<span class="avz-journey-card-field"><span class="avz-journey-card-key">Size</span><span class="avz-journey-card-value">' + escapeHtml(t.size) + '</span></span>';
          canvasHtml += '</div>';
        });
        canvasHtml += '</div>';
      });
      canvasHtml += '</div>';
      canvasHtml += '</div>';

    });

    // SVG overlay inside canvas for correct coordinate space
    canvasHtml += '<svg class="avz-journey-svg" id="tjSvg"></svg>';
    canvasHtml += '</div>'; // .avz-journey-canvas
    canvasHtml += '</div>'; // .avz-journey-canvas-wrap

    els.tensorJourneyContent.innerHTML = canvasHtml;

    // ---- Bind Events ----
    bindJourneyCardEvents();

    // ---- Draw SVG lines after DOM is ready ----
    requestAnimationFrame(function() {
      drawJourneyLines(cfg, tjs);
    });

    // ---- ResizeObserver for line updates ----
    setupJourneyResizeObserver(cfg, tjs);

    // Sync state.selectedTensorId with TJ_STATE
    state.selectedTensorId = tjs.selectedTensorId;
  }

  // ---- Hardware icons ----
  function hardwareIconPath(locId) {
    switch (locId) {
      case 'gmInput':
      case 'gmOutput':
        return '<rect x="3" y="4" width="18" height="14" rx="2"/><path d="M7 10h10"/><path d="M7 13h8"/>';
      case 'l1':
        return '<rect x="4" y="6" width="16" height="12" rx="2"/><path d="M6 9h12"/><path d="M6 12h9"/><path d="M6 15h6"/>';
      case 'a2b2':
        return '<rect x="3" y="4" width="8" height="7" rx="1"/><rect x="13" y="4" width="8" height="7" rx="1"/><rect x="3" y="13" width="8" height="7" rx="1"/><rect x="13" y="13" width="8" height="7" rx="1"/>';
      case 'co1':
        return '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 10l3-4 3 8 3-6 2 5"/>';
      default:
        return '<rect x="4" y="4" width="16" height="16" rx="2"/>';
    }
  }

  // ---- Build highlight set: selected tensor + upstream/downstream ----
  function buildHighlightSet(highlightId, cfg) {
    var set = new Set();
    if (!highlightId) return set;
    set.add(highlightId);

    // Walk downstream
    var queue = [highlightId];
    while (queue.length) {
      var cur = queue.shift();
      cfg.connections.forEach(function(c) {
        if (c.from === cur && !set.has(c.to)) {
          set.add(c.to);
          queue.push(c.to);
        }
        if (c.mergeOf && c.mergeOf.indexOf(cur) >= 0 && !set.has(c.to)) {
          set.add(c.to);
          queue.push(c.to);
        }
      });
    }

    // Walk upstream
    queue = [highlightId];
    while (queue.length) {
      var cur2 = queue.shift();
      cfg.connections.forEach(function(c) {
        if (c.to === cur2 && c.from && !set.has(c.from)) {
          set.add(c.from);
          queue.push(c.from);
        }
        if (c.to === cur2 && c.mergeOf) {
          c.mergeOf.forEach(function(m) {
            if (!set.has(m)) { set.add(m); queue.push(m); }
          });
        }
      });
    }

    return set;
  }

  // ---- Draw SVG connection lines ----
  function drawJourneyLines(cfg, tjs) {
    var svg = document.getElementById('tjSvg');
    var canvasWrap = document.getElementById('tjCanvasWrap');
    var canvas = document.getElementById('tjCanvas');
    if (!svg || !canvasWrap || !canvas) return;

    var canvasRect = canvas.getBoundingClientRect();

    svg.setAttribute('viewBox', '0 0 ' + canvas.scrollWidth + ' ' + canvas.scrollHeight);
    svg.style.width = canvas.scrollWidth + 'px';
    svg.style.height = canvas.scrollHeight + 'px';

    var svgNS = 'http://www.w3.org/2000/svg';
    svg.innerHTML = '';

    var highlightSet = tjs.highlightTensorId ? buildHighlightSet(tjs.highlightTensorId, cfg) : null;

    // 卡片 anchor 工具
    function anchor(cardId) {
      var card = canvas.querySelector('.avz-journey-card[data-tensor-id="' + cardId + '"]');
      if (!card) return null;
      var r = card.getBoundingClientRect();
      return {
        x: r.right - canvasRect.left,
        y: r.top + r.height / 2 - canvasRect.top,
        left: r.left - canvasRect.left
      };
    }
    function regionBounds(locationId) {
      var region = canvas.querySelector('.avz-journey-region[data-location="' + locationId + '"]');
      if (!region) return null;
      var r = region.getBoundingClientRect();
      return {
        left: r.left - canvasRect.left,
        right: r.right - canvasRect.left
      };
    }
    function edgeClass(branch, highlighted) {
      return 'edge-' + branch + (highlighted ? '' : ' edge-dimmed');
    }
    function addPath(cls, d) {
      var p = document.createElementNS(svgNS, 'path');
      p.setAttribute('d', d);
      p.setAttribute('class', cls);
      svg.appendChild(p);
    }
    function addArrow(cls, x, y, dir) {
      var s = 5;
      var a = document.createElementNS(svgNS, 'polygon');
      var pts;
      if (dir < 0) pts = (x + s) + ',' + (y - s / 2) + ' ' + x + ',' + y + ' ' + (x + s) + ',' + (y + s / 2);
      else pts = (x - s) + ',' + (y - s / 2) + ' ' + x + ',' + y + ' ' + (x - s) + ',' + (y + s / 2);
      a.setAttribute('points', pts);
      a.setAttribute('class', cls);
      svg.appendChild(a);
    }
    function addLabel(text, x, y, cls, anchorMode) {
      var t = document.createElementNS(svgNS, 'text');
      t.setAttribute('x', x);
      t.setAttribute('y', y);
      t.setAttribute('class', cls || 'op-label');
      t.setAttribute('dominant-baseline', 'middle');
      if (anchorMode === 'end') t.setAttribute('text-anchor', 'end');
      else if (anchorMode === 'start') t.setAttribute('text-anchor', 'start');
      else t.setAttribute('text-anchor', 'middle');
      t.textContent = text;
      svg.appendChild(t);
      return t;
    }

    // ---- 普通连接：GM→L1 / L1→A2B2 / CO1→GM(Out) ----
    cfg.connections.forEach(function(conn) {
      if (conn.from === 'fmapA2' || conn.from === 'weightB2' || conn.from === 'biasC2') return; // merge 单独处理

      var A = anchor(conn.from);
      var B = anchor(conn.to);
      if (!A || !B) return;

      var isHl = highlightSet ? (highlightSet.has(conn.from) && highlightSet.has(conn.to)) : true;
      var eCls = edgeClass(conn.branch, isHl);
      var lCls = isHl ? 'op-label' : 'op-label op-label-dimmed';

      // 普通连接必须落到目标卡片左边；B.x 是目标卡片右边，不能作为入点。
      var targetX = B.left;
      var midX = (A.x + targetX) / 2;
      addPath(eCls, 'M' + A.x + ',' + A.y + ' L' + midX + ',' + A.y + ' L' + midX + ',' + B.y + ' L' + targetX + ',' + B.y);
      addArrow(eCls, targetX, B.y, 1);

      // 操作标签
      if (conn.op) {
        if (conn.from === 'accumCo1' && conn.to === 'outputGM') {
          // 两行：K0–K8 Accumulate / Fixpipe · ReLU · FP32→FP16 · Write GM
          var parts = conn.op.split(' \u00b7 ');
          var line1 = parts[0];
          var line2 = parts.slice(1).join(' \u00b7 ');
          var labelX = midX;
          var labelY = (A.y + B.y) / 2;
          addLabel(line1, labelX, labelY - 8, lCls);
          addLabel(line2, labelX, labelY + 8, lCls);
        } else {
          addLabel(conn.op, midX, A.y - 10, lCls);
        }
      }
    });

    // ---- A2/B2 → CO1 三路汇合（圆滑曲线）----
    var accum = anchor('accumCo1');
    if (accum) {
      var mergeSources = [
        { card: 'fmapA2',  branch: 'mmad',    bend: -1 },  // 蓝色向下弯曲
        { card: 'weightB2', branch: 'weight', bend: 0 },   // 绿色水平
        { card: 'biasC2',  branch: 'bias',    bend: 1 },   // 紫色向上弯曲
      ];
      var mergeEndX = accum.left;
      var mergeEndY = accum.y;
      var sourceRegion = regionBounds('a2b2');

      mergeSources.forEach(function(ms) {

        var S = anchor(ms.card);
        if (!S) return;
        var isHl = highlightSet ? (highlightSet.has(ms.card) && highlightSet.has('accumCo1')) : true;
        var eCls = edgeClass(ms.branch, isHl);
        var lCls = isHl ? 'op-label' : 'op-label op-label-dimmed';

        // 汇合点：三路在 CO1 左侧同一点汇合，箭头指向 accumCo1 左缘中点
        var cx1 = S.x;
        var cy1 = S.y;
        var cx2 = mergeEndX;
        var cy2 = mergeEndY;
        if (ms.branch === 'weight') {
          // 水平直连
          addPath(eCls, 'M' + cx1 + ',' + cy1 + ' L' + cx2 + ',' + cy2);
          addArrow(eCls, cx2, cy2, 1);
          // LoadData2D 标签
          addLabel('LoadData2D', (cx1 + cx2) / 2, cy1 - 10, lCls);
        } else {
          // 参考图的连接方式：先沿 A2/B2 的空白连接带水平走到 Stage 边界，
          // 再进入 CO1 区域弯向 accumCo1，避免曲线过早侵入卡片区。
          var hx = sourceRegion && sourceRegion.right > cx1
            ? sourceRegion.right
            : cx1 + (cx2 - cx1) * 0.42;
          var curveSpan = Math.max(24, cx2 - hx);
          var d = 'M' + cx1 + ',' + cy1 +
                  ' L' + hx + ',' + cy1 +
                  ' C' + (hx + curveSpan * 0.28) + ',' + cy1 +
                  ' ' + (cx2 - curveSpan * 0.32) + ',' + cy2 +
                  ' ' + cx2 + ',' + cy2;
          addPath(eCls, d);
          addArrow(eCls, cx2, cy2, 1);
          if (ms.branch === 'bias') addLabel('Bias Init', (cx1 + hx) / 2, cy1 - 10, lCls);
        }
      });

      // Mmad 标签放在 A2 → CO1 路径的中段，避免固定偏移在面板缩放后漂移
      if (accum) {
        var mSrc = anchor('fmapA2');
        if (mSrc) {
          var labelStartX = sourceRegion && sourceRegion.right > mSrc.x ? sourceRegion.right : mSrc.x;
          addLabel('Mmad', labelStartX + (accum.left - labelStartX) * 0.38, (mSrc.y + accum.y) / 2, 'op-label');
        }
      }
    }
  }


  function setupJourneyResizeObserver(cfg, tjs) {
    if (window._tjResizeObserver) window._tjResizeObserver.disconnect();
    var canvasWrap = document.getElementById('tjCanvasWrap');
    var canvas = document.getElementById('tjCanvas');
    if (!canvasWrap || !canvas || typeof ResizeObserver !== 'function') return;

    window._tjResizeObserver = new ResizeObserver(function() {
      if (window._tjDrawRaf) return;
      window._tjDrawRaf = requestAnimationFrame(function() {
        window._tjDrawRaf = 0;
        drawJourneyLines(cfg, tjs);
      });
    });
    window._tjResizeObserver.observe(canvasWrap);
    window._tjResizeObserver.observe(canvas);
    canvas.querySelectorAll('.avz-journey-region, .avz-journey-card').forEach(function(node) {
      window._tjResizeObserver.observe(node);
    });
  }

  // ---- Card click handler ----
  function bindJourneyCardEvents() {
    var cards = els.tensorJourneyContent.querySelectorAll('.avz-journey-card');
    cards.forEach(function(card) {
      card.addEventListener('click', function() {
        var tensorId = card.dataset.tensorId;
        // Toggle selection
        if (TJ_STATE.selectedTensorId === tensorId) {
          TJ_STATE.selectedTensorId = null;
        } else {
          TJ_STATE.selectedTensorId = tensorId;
        }

        // Just update card selection classes, no dimming or line redraw
        refreshJourneyCards();

        // Status feedback
        var tensor = TJ_CONFIG.tensors.find(function(t) { return t.id === tensorId; });
        if (els.statusText) els.statusText.textContent = TJ_STATE.selectedTensorId
          ? 'Selected: ' + (tensor ? tensor.name : tensorId)
          : 'Ready';
        state.selectedTensorId = TJ_STATE.selectedTensorId;

        // Update visualizer pane above: show data dump panel for fmapA1
        var trace = currentTrace();
        var isDataDump = state.selectedTensorId === 'buffer:feature:a1' || state.selectedTensorId === 'fmapA1';
        renderDataDumpPanelVisibility(isDataDump);
        if (isDataDump) {
          renderTensorDataDump(trace);
        } else {
          renderTensorTabs(trace);
          renderTensorViewport(trace);
          renderTileLens(trace);
          renderArchitectureFocus(trace);
        }
      });
    });
  }

  // ---- Refresh card selection after state change ----
  function refreshJourneyCards() {
    var cards = els.tensorJourneyContent.querySelectorAll('.avz-journey-card');
    cards.forEach(function(card) {
      var tid = card.dataset.tensorId;
      var isSelected = tid === TJ_STATE.selectedTensorId;
      card.classList.toggle('is-selected', isSelected);
    });
  }

  function handleJourneyTensorClick(tensorId) {
    // Legacy stub - handled by new card click handler
    TJ_STATE.selectedTensorId = tensorId;
    if (els.statusText) els.statusText.textContent = 'Selected: ' + tensorId;
  }

  function handleJourneyTransformClick(stepId, label) {
    // Legacy stub - no longer used with new implementation
    var trace = currentTrace();
    if (!trace) return;
    var steps = trace.steps || [];
    var stepIndex = steps.findIndex(function(s) { return s.id === stepId || s.stageId === stepId; });
    if (stepIndex >= 0) {
      selectStep(stepIndex, { instructionOperation: label });
      setExecutionView('instructions');
    }
  }


  function renderTensorDataDump(trace) {
    if (!trace || !els.tensorDataDumpPanel) return;
    var isDumpMode = state.dataViewMode === 'data-dump' || state.dataViewMode === 'dump';
    var isAnimMode = state.dataViewMode === 'animation';

    // Hide standard data dump body in animation mode
    if (els.dataDumpBody) {
      els.dataDumpBody.hidden = isAnimMode || !isDumpMode;
      els.dataDumpBody.style.display = (isDumpMode && !isAnimMode) ? '' : 'none';
    }

    // Show/hide animation body
    if (els.animationBody) {
      els.animationBody.hidden = !isAnimMode;
    }

    // Handle animation mode
    if (isAnimMode) {
      renderAnimationView(trace);
      return;
    }

    if (!isDumpMode) return;
    var tensorId = state.selectedTensorId;
    if (!tensorId || (tensorId !== 'buffer:feature:a1' && tensorId !== 'fmapA1')) return;
    renderDataDumpHeader(trace, tensorId);
    renderDataDumpVolume(trace);
    renderChannelSelector();
    renderHeatmap(trace);
    if (state.selectedDataElement && els.cellInspector) {
      els.cellInspector.hidden = false;
      renderCellInspector(trace, state.selectedDataElement);
    } else if (els.cellInspector) {
      els.cellInspector.hidden = true;
    }
  }

  /* ================================================================
     LoadData3D Animation: fmapA1 -> fmapA2
     Ported from recommendation-trace-V3.html
     ================================================================ */

  function animationParams() {
    var s = state.animationState;
    return {
      Hi: s.H, Wi: s.W, Ci: s.Ci, C0: s.C0,
      Hk: s.filterH, Wk: s.filterW,
      strideH: s.strideH, strideW: s.strideW,
      padTop: s.padTop, padRight: s.padRight, padBottom: s.padBottom, padLeft: s.padLeft,
      padValue: s.padValue, dilationH: s.dilationH, dilationW: s.dilationW,
      repeatMode: s.repeatMode, repeatTime: s.repeatTime,
      showChannelPadding: s.showChannelPadding
    };
  }

  function animationShape(p) {
    var eKH = (p.Hk - 1) * p.dilationH + 1;
    var eKW = (p.Wk - 1) * p.dilationW + 1;
    var Ho = Math.floor((p.Hi + p.padTop + p.padBottom - eKH) / p.strideH) + 1;
    var Wo = Math.floor((p.Wi + p.padLeft + p.padRight - eKW) / p.strideW) + 1;
    var C1 = Math.ceil(p.Ci / p.C0);
    return {
      Ho: Ho, Wo: Wo, C1: C1,
      rows: Math.max(0, Ho * Wo),
      cols: Math.max(0, p.Hk * p.Wk * p.Ci),
      valid: Ho > 0 && Wo > 0
    };
  }

  function animationPatches(p, shape) {
    if (!shape.valid) return [];
    var patches = [];
    for (var oh = 0; oh < shape.Ho; oh++) {
      for (var ow = 0; ow < shape.Wo; ow++) {
        var baseH = oh * p.strideH - p.padTop;
        var baseW = ow * p.strideW - p.padLeft;
        var row = oh * shape.Wo + ow;
        var elements = [];
        for (var kh = 0; kh < p.Hk; kh++) {
          for (var kw = 0; kw < p.Wk; kw++) {
            for (var c = 0; c < p.Ci; c++) {
              var srcH = baseH + kh * p.dilationH;
              var srcW = baseW + kw * p.dilationW;
              var col = ((kh * p.Wk) + kw) * p.Ci + c;
              var isPad = srcH < 0 || srcH >= p.Hi || srcW < 0 || srcW >= p.Wi;
              elements.push({ oh: oh, ow: ow, row: row, kh: kh, kw: kw, c: c, col: col, srcH: srcH, srcW: srcW, isPadding: isPad });
            }
          }
        }
        patches.push({ oh: oh, ow: ow, row: row, baseH: baseH, baseW: baseW, elements: elements });
      }
    }
    return patches;
  }

  function animationSampleGroups(patch) {
    if (!patch) return [];
    var groups = [];
    var seen = {};
    patch.elements.forEach(function(el) {
      var key = el.kh + ',' + el.kw;
      if (!seen[key]) { seen[key] = true; groups.push({ kh: el.kh, kw: el.kw, elements: [] }); }
    });
    patch.elements.forEach(function(el) {
      var key = el.kh + ',' + el.kw;
      groups.forEach(function(g) { if (g.kh === el.kh && g.kw === el.kw) g.elements.push(el); });
    });
    return groups;
  }

  function animationTotalSteps(patches) {
    var total = 0;
    patches.forEach(function(p) { total += animationSampleGroups(p).length; });
    return total;
  }

  function animationContextAtStep(patches, step) {
    if (!patches.length) return { patch: null, element: null, sampleElements: [], step: 0, maxStep: 0 };
    var maxStep = Math.max(0, animationTotalSteps(patches) - 1);
    var boundedStep = Math.max(0, Math.min(maxStep, Number(step) || 0));
    var cursor = boundedStep;
    var patch = patches[0], sampleGroups = animationSampleGroups(patch), sampleGroup = sampleGroups[0];
    for (var i = 0; i < patches.length; i++) {
      var groups = animationSampleGroups(patches[i]);
      if (cursor < groups.length) {
        patch = patches[i]; sampleGroups = groups; sampleGroup = groups[cursor]; break;
      }
      cursor -= groups.length;
    }
    return {
      patch: patch, element: sampleGroup.elements[0] || patch.elements[0],
      sampleElements: sampleGroup.elements, step: boundedStep, maxStep: maxStep
    };
  }

  /* ---- Canvas drawing for A1 layout ---- */
  function animTensorPoint(ox, oy, unit, dx, dy, col, row, depth) {
    return { x: ox + col * unit + depth * dx, y: oy + row * unit - depth * dy };
  }

  function animDrawQuad(ctx, p0, p1, p2, p3, fill, stroke, lw) {
    lw = lw || 1;
    ctx.beginPath();
    ctx.moveTo(p0.x, p0.y); ctx.lineTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.lineTo(p3.x, p3.y);
    ctx.closePath();
    ctx.fillStyle = fill; ctx.fill();
    ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke();
  }

  function animDrawVoxel(ctx, ox, oy, unit, dx, dy, col, row, depth, faces) {
    var gap = 0.065;
    var pt = function(c, r, d) { return animTensorPoint(ox, oy, unit, dx, dy, c, r, d); };
    var c0 = col + gap, c1 = col + 1 - gap, r0 = row + gap, r1 = row + 1 - gap, d0 = depth + gap, d1 = depth + 1 - gap;
    var f0 = pt(c0, r0, d0), f1 = pt(c1, r0, d0), f2 = pt(c1, r1, d0), f3 = pt(c0, r1, d0);
    var b0 = pt(c0, r0, d1), b1 = pt(c1, r0, d1), b2 = pt(c1, r1, d1);
    animDrawQuad(ctx, f0, f1, b1, b0, faces.top, faces.topEdge || faces.edge, faces.topLineWidth || faces.lineWidth || 1);
    animDrawQuad(ctx, f1, f2, b2, b1, faces.east, faces.edge, faces.lineWidth || 1);
    animDrawQuad(ctx, f0, f1, f2, f3, faces.south, faces.edge, faces.lineWidth || 1);
    return { center: { x: (f0.x + f3.x) / 2, y: (f0.y + f2.y) / 2 } };
  }

  function animFrameLabels(ctx, ox, oy, unit, dx, dy, cols, rows, depth) {
    var pt = function(c, r, d) { return animTensorPoint(ox, oy, unit, dx, dy, c, r, d); };
    var f0 = pt(0, 0, 0), f2 = pt(cols, rows, 0), f3 = pt(0, rows, 0), b0 = pt(0, 0, depth);
    ctx.save();
    ctx.fillStyle = 'rgba(236,242,248,0.82)';
    ctx.font = '700 13px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('W', (f2.x + f3.x) / 2, f2.y + 24);
    ctx.textAlign = 'right';
    ctx.fillText('H', f3.x - 10, (f0.y + f3.y) / 2);
    ctx.textAlign = 'center';
    ctx.fillText('C0', (f0.x + b0.x) / 2, (f0.y + b0.y) / 2 - 14);
    ctx.restore();
  }

  function animColorToken(token, fallback) {
    var v = getComputedStyle(document.documentElement).getPropertyValue(token).trim();
    return v || fallback;
  }

  function animFaces(kind) {
    var blue = animColorToken('--primary', '#4369ef');
    var blueH = animColorToken('--primary-hover', '#5a92e6');
    var orange = animColorToken('--warning', '#ffaa3b');
    var palettes = {
      base: { top: 'rgba(74,80,88,0.80)', east: 'rgba(59,65,74,0.80)', south: 'rgba(48,54,64,0.80)', edge: 'rgba(10,12,16,0.62)' },
      pad: { top: 'rgba(90,96,104,0.25)', east: 'rgba(54,60,68,0.25)', south: 'rgba(42,48,56,0.25)', edge: 'rgba(190,200,212,0.18)' },
      channelPad: { top: 'rgba(74,80,88,0.22)', east: 'rgba(59,65,74,0.20)', south: 'rgba(48,54,64,0.20)', edge: 'rgba(190,200,212,0.14)' },
      window: { top: blue, east: blueH, south: blue, edge: blueH, lineWidth: 1.15, topLineWidth: 1.4 },
      sampled: { top: blue, east: blueH, south: blue, edge: blueH, lineWidth: 1.25, topLineWidth: 1.6 },
      current: { top: orange, east: orange, south: orange, edge: orange, lineWidth: 1.5, topLineWidth: 2 },
      skipped: { top: 'rgba(255,255,255,0.08)', east: 'rgba(255,255,255,0.055)', south: 'rgba(255,255,255,0.045)', edge: 'rgba(255,255,255,0.18)' }
    };
    return palettes[kind] || palettes.base;
  }

  function drawAnimationA1Canvas(canvas, p, shape, patch, sample) {
    if (!canvas || !patch || !sample) return;
    var rect = canvas.getBoundingClientRect();
    var dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    var totalH = p.Hi + p.padTop + p.padBottom;
    var totalW = p.Wi + p.padLeft + p.padRight;
    var layers = p.showChannelPadding ? p.C0 : p.Ci;
    var fitW = (rect.width - 96) / Math.max(1, totalW + layers * 0.56);
    var fitH = (rect.height - 94) / Math.max(1, totalH + layers * 0.42);
    var unit = Math.max(8, Math.min(30, fitW, fitH));
    var dx = unit * 0.56, dy = unit * 0.42;
    var objectW = totalW * unit + layers * dx;
    var objectH = totalH * unit + layers * dy;
    var ox = Math.max(54, (rect.width - objectW) / 2 + 14);
    var oy = Math.max(58 + layers * dy, (rect.height - objectH) / 2 + layers * dy + 18);

    var windowCoords = {};
    var skippedCoords = {};
    var sampledKeys = {};
    patch.elements.forEach(function(item) { sampledKeys[item.srcH + ',' + item.srcW + ',' + item.c] = true; });

    for (var kh = 0; kh < p.Hk; kh++) {
      for (var kw = 0; kw < p.Wk; kw++) {
        var h = patch.baseH + kh * p.dilationH;
        var w = patch.baseW + kw * p.dilationW;
        for (var c = 0; c < p.Ci; c++) windowCoords[h + ',' + w + ',' + c] = true;
      }
    }

    animFrameLabels(ctx, ox, oy, unit, dx, dy, totalW, totalH, layers);

    var items = [];
    for (var c = 0; c < layers; c++) {
      for (var ph = 0; ph < totalH; ph++) {
        for (var pw = 0; pw < totalW; pw++) {
          var h = ph - p.padTop, w = pw - p.padLeft;
          var key = h + ',' + w + ',' + c;
          var isPad = h < 0 || h >= p.Hi || w < 0 || w >= p.Wi;
          var isChPad = p.showChannelPadding && c >= p.Ci;
          var kind = isChPad ? 'channelPad' : (isPad ? 'pad' : 'base');
          if (windowCoords[key]) kind = 'window';
          if (sampledKeys[key]) kind = 'sampled';
          if (h === sample.srcH && w === sample.srcW) kind = 'current';
          items.push({ ph: ph, pw: pw, h: h, w: w, c: c, kind: kind, isPad: isPad, isChPad: isChPad });
        }
      }
    }
    items.sort(function(a, b) { return (b.c - a.c) || (b.ph - a.ph) || (a.pw - b.pw); });

    ctx.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    var showPadLabels = totalH * totalW <= 36;

    for (var i = 0; i < items.length; i++) {
      var item = items[i];
      var result = animDrawVoxel(ctx, ox, oy, unit, dx, dy, item.pw, item.ph, item.c, animFaces(item.kind));
      var itemKey = item.h + ',' + item.w + ',' + item.c;
      var isMapped = sampledKeys[itemKey] || windowCoords[itemKey];
      if (item.kind !== 'skipped' && !item.isChPad && (isMapped || (item.isPad && showPadLabels))) {
        ctx.fillStyle = item.kind === 'current' ? '#211603' : 'rgba(240,247,252,0.86)';
        ctx.fillText(item.isPad ? 'PAD' : 'x' + item.h + item.w + item.c, result.center.x, result.center.y + 1);
      }
    }
  }

  /* ---- Render full animation view ---- */
  function renderAnimationView(trace) {
    if (!els.animationBody) return;
    // Ensure data dump body is hidden during animation
    if (els.dataDumpBody) {
      els.dataDumpBody.hidden = true;
      els.dataDumpBody.style.display = 'none';
    }
    if (els.channelControls) els.channelControls.style.display = 'none';
    if (els.dataDumpHeatmap) els.dataDumpHeatmap.style.display = 'none';
    if (els.cellInspector) els.cellInspector.hidden = true;
    var p = animationParams();
    var shape = animationShape(p);
    if (!shape.valid) {
      if (els.animationA1Status) els.animationA1Status.textContent = '当前参数无法产生输出位置';
      return;
    }
    var patches = animationPatches(p, shape);
    var maxSteps = animationTotalSteps(patches);
    state.animationStep = Math.max(0, Math.min(state.animationStep, Math.max(0, maxSteps - 1)));
    var context = animationContextAtStep(patches, state.animationStep);

    // Update A2 meta
    if (els.animationA2Meta) {
      els.animationA2Meta.textContent = '形状 = ' + shape.Ho + '×' + shape.Wo + ' × ' + p.Hk + '×' + p.Wk + '×' + p.Ci + ' = ' + shape.rows + ' × ' + shape.cols;
    }

    // Draw A1 canvas
    drawAnimationA1Canvas(els.animationA1Canvas, p, shape, context.patch, context.element);

    // Update A1 status
    if (els.animationA1Status && context.element) {
      els.animationA1Status.textContent = '当前：窗口行 ' + context.element.row + ', 采样 (kh=' + context.element.kh + ', kw=' + context.element.kw + ')，沿 C 轴 ' + context.sampleElements.length + ' 个值';
    }

    // Render A2 grid
    renderAnimationA2Grid(patches, shape, context, p);

    // Update pills
    if (els.animationA2Pills && context.element) {
      var sampEls = context.sampleElements;
      var firstCol = sampEls[0] ? sampEls[0].col : context.element.col;
      var lastCol = sampEls[sampEls.length - 1] ? sampEls[sampEls.length - 1].col : context.element.col;
      var isPad = sampEls.some(function(e) { return e.isPadding; });
      els.animationA2Pills.innerHTML =
        '<span class="avz-animation-pill is-hot">row = oh×Wo + ow = ' + context.element.row + '</span>' +
        '<span class="avz-animation-pill">cols ' + firstCol + '…' + lastCol + '：固定 kh=' + context.element.kh + ', kw=' + context.element.kw + '，遍历 c' + (isPad ? ' (PAD)' : '') + '</span>';
    }

    // Update playback status
    if (els.animationPlaybackStatus) {
      els.animationPlaybackStatus.textContent = state.animationPlaying
        ? 'Playing · Step ' + (state.animationStep + 1) + ' / ' + maxSteps
        : 'Paused · Step ' + (state.animationStep + 1) + ' / ' + maxSteps;
    }
  }

  function renderAnimationA2Grid(patches, shape, context, p) {
    var grid = els.animationA2Grid;
    if (!grid) return;
    var cols = Math.max(1, shape.cols);
    var rows = Math.max(1, shape.rows);
    var cellMin = cols > 28 ? '14px' : cols > 16 ? '22px' : '34px';
    var cellIdeal = cols > 28 ? '26px' : cols > 16 ? '34px' : '46px';
    var showText = rows * cols <= 180;

    // Track written cells
    var written = {};
    var priorSteps = 0;
    patches.forEach(function(candidate) {
      var groups = animationSampleGroups(candidate);
      groups.forEach(function(group, si) {
        if (priorSteps + si <= context.step) {
          group.elements.forEach(function(item) { written[item.row + ',' + item.col] = true; });
        }
      });
      priorSteps += groups.length;
    });

    var element = context.element;
    var cells = [];
    for (var row = 0; row < rows; row++) {
      for (var col = 0; col < cols; col++) {
        var mapped = patches[row] ? patches[row].elements[col] : null;
        var isCurSample = mapped && mapped.kh === element.kh && mapped.kw === element.kw;
        var active = row === element.row && isCurSample;
        var classes = ['avz-animation-cell'];
        if (row === element.row) classes.push('is-current-row');
        if (isCurSample) classes.push('is-current-col');
        if (written[row + ',' + col]) classes.push('is-written');
        if (mapped && mapped.isPadding) classes.push('is-pad');
        if (active) classes.push('is-active');
        // Always highlight top-left 16x16
        if (row < 16 && col < 16) classes.push('is-top-left-16x16');
        var content = '';
        if (showText && mapped) content = mapped.isPadding ? 'PAD' : 'x' + mapped.srcH + mapped.srcW + mapped.srcC;
        var title = mapped ? 'row=' + row + ', col=' + col + ', kh=' + mapped.kh + ', kw=' + mapped.kw + ', c=' + mapped.c + ', src=(' + mapped.srcH + ',' + mapped.srcW + ')' : '';
        cells.push('<div class="' + classes.join(' ') + '" title="' + title + '">' + content + '</div>');
      }
    }

    grid.style.setProperty('--cols', String(cols));
    grid.style.setProperty('--cell-min', cellMin);
    grid.style.setProperty('--cell-ideal', cellIdeal);
    grid.innerHTML = cells.join('');
  }

  function startAnimationPlayback() {
    stopAnimationPlayback();
    state.animationPlaying = true;
    state.animationStep = 0;
    var p = animationParams();
    var shape = animationShape(p);
    var patches = animationPatches(p, shape);
    var maxSteps = animationTotalSteps(patches);

    state.animationTimer = setInterval(function() {
      if (!state.animationPlaying) return;
      if (state.animationStep >= maxSteps - 1) {
        state.animationStep = 0;
      } else {
        state.animationStep += 1;
      }
      renderAnimationView(currentTrace());
    }, 200);
  }

  function stopAnimationPlayback() {
    state.animationPlaying = false;
    if (state.animationTimer) { clearInterval(state.animationTimer); state.animationTimer = null; }
    // Restore data dump body and controls when leaving animation
    if (els.dataDumpBody) {
      els.dataDumpBody.hidden = false;
      els.dataDumpBody.style.display = '';
    }
    if (els.channelControls) els.channelControls.style.display = '';
    if (els.dataDumpHeatmap) els.dataDumpHeatmap.style.display = '';
  }

  function renderDataDumpHeader(trace, tensorId) {
    if (!els.dataDumpTensorInfo) return;
    var name = 'fmapA1';
    var role = 'Feature staging buffer';
    var location = 'A1 / L1';
    var logicalLayout = 'NCHW';
    var physicalLayout = 'NC1HWC0';
    var dtype = 'FP16';
    var bytes = '2048 B';
    els.dataDumpTensorInfo.innerHTML = [
      fieldHtml(null, name, true, true),
      fieldHtml('Role', role),
      fieldHtml('Location', location),
      fieldHtml('LOGICAL', '[1,1,8,8,16] NCHW', false, true),
      fieldHtml('PHYSICAL', '[1,1,8,8,16] NC1HWC0', false, true),
      fieldHtml('Dtype', dtype),
      fieldHtml('Bytes', bytes),
    ].join('');
  }

  function fieldHtml(label, value, isName, isMono) {
    var valClass = 'avz-data-dump-field-value';
    if (isName) valClass += ' avz-data-dump-field-value--name';
    if (isMono) valClass += ' avz-data-dump-field-value--mono';
    var labelHtml = label != null
      ? '<span class="avz-data-dump-field-label">' + escapeHtml(label) + '</span>'
      : '';
    return '<div class="avz-data-dump-field">' + labelHtml +
      '<span class="' + valClass + '">' + escapeHtml(value) + '</span></div>';
  }

  function renderChannelSelector() {
    if (!els.channelSelect || !els.channelTotal) return;
    var c = state.selectedChannel;
    els.channelSelect.innerHTML = '';
    for (var i = 0; i < 16; i++) {
      var opt = document.createElement('option');
      opt.value = String(i);
      opt.textContent = 'C' + i;
      if (i === c) opt.selected = true;
      els.channelSelect.appendChild(opt);
    }
    els.channelTotal.textContent = '/ 16';
    if (els.prevChannelBtn) els.prevChannelBtn.disabled = c <= 0;
    if (els.nextChannelBtn) els.nextChannelBtn.disabled = c >= 15;
  }

  function renderDataDumpVolume(trace) {
    var canvas = els.dataDumpVolumeCanvas;
    if (!canvas) return;
    var c = state.selectedChannel;
    var H = 8, W = 8, Ci = 16;
    var c1 = Math.floor(c / 16);
    var c0 = c % 16;
    var scene = createVolumeScene({
      columns: W, rows: H, depth: Ci, tone: 'neutral',
      axes: { columns: 'W=8', rows: 'H=8', depth: 'Ci=16' },
      prefix: 'fmapa1-dump'
    });
    for (var row = 0; row < H; row++) {
      for (var col = 0; col < W; col++) {
        for (var d = 0; d < Ci; d++) {
          var vid = 'fmapa1-dump-' + col + '-' + row + '-' + d;
          var voxel = scene.voxels.find(function(v) { return v.id === vid; });
          if (voxel) {
            voxel.tone = d === c0 ? 'input' : 'neutral';
            voxel.state = d === c0 ? 'current' : 'base';
          }
        }
      }
    }
    var PtoVolume = window.PtoTensorVolumeCanvas;
    var options = { showAxes: true, showGrid: true, ariaLabel: 'fmapA1 volume channel ' + c };
    if (PtoVolume) {
      if (!state._dataDumpVolumeController) {
        state._dataDumpVolumeController = PtoVolume.render(canvas, scene, options);
      } else {
        state._dataDumpVolumeController.update(scene, options);
      }
    }
    if (els.dataDumpVolumeMeta) {
      els.dataDumpVolumeMeta.textContent = 'H=8 \u00b7 W=8 \u00b7 C=16 \u00b7 Ci=' + c + ' (c1=' + c1 + ', c0=' + c0 + ')';
    }
  }

  function computeSliceStats(values) {
    var n = values.length, min = Infinity, max = -Infinity, sum = 0, zeros = 0, nans = 0, infs = 0;
    for (var i = 0; i < n; i++) {
      var v = values[i];
      if (typeof v !== 'number' || isNaN(v)) { nans++; continue; }
      if (!isFinite(v)) { infs++; continue; }
      if (v === 0) zeros++;
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
    }
    var validN = n - nans - infs;
    var mean = validN > 0 ? sum / validN : 0;
    var sumSq = 0;
    for (var i = 0; i < n; i++) {
      var v = values[i];
      if (typeof v !== 'number' || isNaN(v) || !isFinite(v)) continue;
      sumSq += (v - mean) * (v - mean);
    }
    return {
      min: validN > 0 ? min : 0, max: validN > 0 ? max : 0,
      mean: mean, stdDev: validN > 1 ? Math.sqrt(sumSq / validN) : 0,
      zeros: zeros, nans: nans, infs: infs
    };
  }

  function renderHeatmap(trace) {
    var grid = document.getElementById('heatmapGrid');
    var headerEl = document.getElementById('heatmapHeader');
    if (!grid || !headerEl) return;
    var c = state.selectedChannel;
    var H = 8, W = 8;
    var maxAbs = 0;
    for (var h = 0; h < H; h++)
      for (var w = 0; w < W; w++)
        maxAbs = Math.max(maxAbs, Math.abs(syntheticData.getA1Value(0, 0, h, w, c)));
    if (maxAbs === 0) maxAbs = 1;
    headerEl.textContent = 'Channel C' + c + ' \u00b7 Slice H\u00d7W = ' + H + '\u00d7' + W;
    var html = '<thead><tr><th></th>';
    for (var w = 0; w < W; w++) html += '<th>W' + w + '</th>';
    html += '</tr></thead><tbody>';
    for (var h = 0; h < H; h++) {
      html += '<tr><th>H' + h + '</th>';
      for (var w = 0; w < W; w++) {
        var val = syntheticData.getA1Value(0, 0, h, w, c);
        var bg = heatmapColor(val, maxAbs);
        var sel = state.selectedDataElement;
        var sc = (sel && sel.h === h && sel.w === w && sel.channel === c) ? ' is-selected' : '';
        html += '<td><div class="avz-heatmap-cell' + sc + (val === 0 ? ' is-zero' : '') + '" style="' + bg + '" data-h="' + h + '" data-w="' + w + '" data-channel="' + c + '" data-value="' + val + '">' + val.toFixed(2) + '</div></td>';
      }
      html += '</tr>';
    }
    html += '</tbody>';
    grid.innerHTML = html;
    var cells = grid.querySelectorAll('.avz-heatmap-cell');
    for (var ci = 0; ci < cells.length; ci++) {
      (function(cell) {
        cell.addEventListener('click', function() {
          var h = parseInt(cell.dataset.h, 10), w = parseInt(cell.dataset.w, 10);
          var ch = parseInt(cell.dataset.channel, 10), v = parseFloat(cell.dataset.value);
          state.selectedDataElement = { tensorId: 'buffer:feature:a1', h: h, w: w, channel: ch, value: v };
          renderHeatmap(trace);
          renderCellInspector(trace, state.selectedDataElement);
          if (els.cellInspector) els.cellInspector.hidden = false;
        });
      })(cells[ci]);
    }

    // Stats bar — inserted between header and grid, never touches existing elements
    var values = [];
    for (var h = 0; h < H; h++) {
      for (var w = 0; w < W; w++) {
        values.push(syntheticData.getA1Value(0, 0, h, w, c));
      }
    }
    var stats = computeSliceStats(values);
    var oldStats = headerEl.parentNode.querySelector('.avz-heatmap-stats');
    if (oldStats) oldStats.remove();
    var sd = [
      ['Min', stats.min.toFixed(2), '检查是否出现异常负值、数值范围是否超出预期'],
      ['Max', stats.max.toFixed(2), '检查溢出、异常峰值，以及激活后的上界变化'],
      ['Mean', stats.mean.toFixed(2), '观察数据整体是否偏正或偏负，判断分布中心是否发生漂移'],
      ['Std Dev', stats.stdDev.toFixed(2), '表示数值分布的离散程度；越大说明数值差异越明显'],
      ['Zeros', String(stats.zeros), '可用于识别 padding、稀疏性，以及 ReLU 后有多少负值被置零'],
      ['NaNs', String(stats.nans), '通常意味着非法运算，如 0/0、无效开方或数值传播异常'],
      ['Infs', String(stats.infs), '通常表示溢出、除以零或计算结果超过数据类型可表达范围'],
    ];
    var sh = '<div class="avz-heatmap-stats">';
    for (var si = 0; si < sd.length; si++) {
      var d = sd[si];
      sh += '<div class="avz-heatmap-stat" title="' + escapeHtml(d[2]) + '"><span class="avz-heatmap-stat-label">' + escapeHtml(d[0]) + '</span><span class="avz-heatmap-stat-value">' + escapeHtml(d[1]) + '</span></div>';
    }
    sh += '</div>';
    headerEl.insertAdjacentHTML('afterend', sh);
  }

  function heatmapColor(value, maxAbs) {
    var intensity = Math.abs(value) / maxAbs;
    var r, g, b;
    if (value > 0) { r = Math.round(48 + intensity * 180); g = Math.round(48 + intensity * 100); b = Math.round(56 - intensity * 30); }
    else if (value < 0) { r = Math.round(48 + intensity * 40); g = Math.round(48 + intensity * 100); b = Math.round(56 + intensity * 180); }
    else { r = 48; g = 48; b = 56; }
    return 'background-color: rgb(' + r + ',' + g + ',' + b + ')';
  }

  function renderCellInspector(trace, element) {
    if (!els.cellInspectorBody) return;
    var ch = element.channel, h = element.h, w = element.w, value = element.value;
    var c1 = Math.floor(ch / 16), c0 = ch % 16;
    var items = [
      { label: 'Value', value: '<span class="avz-inspector-value--number">' + value.toFixed(2) + '</span>' },
      { label: 'Dtype', value: 'FP16' },
      { label: 'Logical', value: '[n=0,c=' + ch + ',h=' + h + ',w=' + w + ']' },
      { label: 'Physical', value: '[n=0,c1=' + c1 + ',h=' + h + ',w=' + w + ',c0=' + c0 + ']' },
      { label: 'Source', value: 'X0[n=0,c=' + ch + ',h=' + h + ',w=' + w + ']' },
      { label: 'Transform', value: 'DataCopy' },
      { label: 'Value Changed', value: 'No' },
      { label: 'Downstream', value: 'Multiple LoadData3D refs' },
    ];
    els.cellInspectorBody.innerHTML = items.map(function(item) {
      return '<div class="avz-inspector-item"><span class="avz-inspector-item-label">' + escapeHtml(item.label) + '</span><span class="avz-inspector-item-value">' + item.value + '</span></div>';
    }).join('');
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
