(() => {
  'use strict';

  const FIXTURES = [
    { id: 'sample.conv_bias_relu', path: 'data/fixtures/conv_bias_relu.trace.json' },
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
    stepIndex: 0,
    evidence: false,
    playing: false,
    timer: null,
    playback: null,
    webglAvailable: null,
    inspectorOpen: false,
    infoOpen: false,
    selectedObject: null,
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
    tensorStage: byId('tensorStage'),
    tensorCanvas: byId('tensorCanvas'),
    tensorFallback: byId('tensorFallback'),
    zoomOut: byId('zoomOut'),
    zoomIn: byId('zoomIn'),
    fitView: byId('fitView'),
    viewportInfo: byId('viewportInfo'),
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
    timelineCanvas: byId('timelineCanvas'),
    traceInfoPanel: byId('traceInfoPanel'),
    traceInfoMeta: byId('traceInfoMeta'),
    traceInfoContent: byId('traceInfoContent'),
    closeTraceInfo: byId('closeTraceInfo'),
    inspectorDrawer: byId('inspectorDrawer'),
    inspectorMeta: byId('inspectorMeta'),
    closeInspector: byId('closeInspector'),
    inspector: byId('inspector'),
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
    await Promise.all(traces.map(loadTraceSource));
    state.traces = traces;
    state.sampleId = traces[0]?.operator?.id || null;
  }

  async function loadTraceSource(trace) {
    if (!trace?.source) return;
    trace.source.keyLines = trace.source.lines || [];
    for (const url of sourceUrlCandidates(trace)) {
      try {
        const response = await fetch(url);
        if (!response.ok) continue;
        const text = await response.text();
        trace.source.fullLines = normalizeSourceLines(text);
        trace.source.sourceUrl = url;
        trace.source.partial = false;
        return;
      } catch {
        // Try the next static source candidate.
      }
    }
    trace.source.fullLines = trace.source.keyLines;
    trace.source.partial = true;
  }

  function sourceUrlCandidates(trace) {
    const path = trace.source?.path || '';
    const candidates = [];
    if (trace.source?.projectPath) candidates.push(trace.source.projectPath);
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
      renderInspector();
    });
    els.zoomOut?.addEventListener('click', () => zoomTensorView(0.86));
    els.zoomIn?.addEventListener('click', () => zoomTensorView(1.16));
    els.fitView?.addEventListener('click', resetTensorView);
    els.viewportInfo?.addEventListener('click', () => {
      setInfoOpen(!state.infoOpen);
    });
    els.closeTraceInfo?.addEventListener('click', () => {
      setInfoOpen(false);
    });
    els.closeInspector?.addEventListener('click', () => {
      setInspectorOpen(false);
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
    els.sourceLines?.addEventListener('click', (event) => {
      const btn = event.target.closest('[data-line]');
      if (!btn || !els.sourceLines.contains(btn)) return;
      const line = Number(btn.dataset.line);
      const trace = currentTrace();
      if (!trace) return;
      const nextIndex = trace.steps.findIndex((step) => step.sourceLines?.includes(line));
      if (nextIndex >= 0) {
        state.selectedObject = { type: 'source', line };
        state.inspectorOpen = true;
        selectStep(nextIndex);
      }
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
    if (command === 'toggle-inspector') {
      if (!state.inspectorOpen && !state.selectedObject) {
        state.selectedObject = { type: 'timeline step', stepIndex: state.stepIndex };
      }
      setInspectorOpen(!state.inspectorOpen);
    }
  }

  function setInfoOpen(open) {
    state.infoOpen = !!open;
    renderInfoPanel();
    syncFrameActions();
  }

  function setInspectorOpen(open) {
    state.inspectorOpen = !!open;
    renderInspector();
    syncFrameActions();
  }

  function syncFrameActions() {
    els.frameActions.forEach((button) => {
      const command = button.dataset.frameCommand;
      const selected = (command === 'toggle-info' && state.infoOpen)
        || (command === 'toggle-inspector' && state.inspectorOpen);
      if (command === 'toggle-info' || command === 'toggle-inspector') {
        button.classList.toggle('is-selected', selected);
        button.setAttribute('aria-expanded', String(selected));
        button.setAttribute('aria-pressed', String(selected));
      }
    });
    if (els.viewportInfo) {
      els.viewportInfo.classList.toggle('is-selected', state.infoOpen);
      els.viewportInfo.setAttribute('aria-expanded', String(state.infoOpen));
      els.viewportInfo.setAttribute('aria-pressed', String(state.infoOpen));
    }
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
      const moved = state.tensorView.moved;
      state.tensorView.dragging = false;
      if (!moved) openInspector('tensor');
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
    state.tensorView.scale = Math.max(0.55, Math.min(2.4, state.tensorView.scale * multiplier));
    renderTensorViewport(currentTrace());
  }

  function resetTensorView() {
    state.tensorView.scale = 1;
    state.tensorView.panX = 0;
    state.tensorView.panY = 0;
    renderTensorViewport(currentTrace());
  }

  function openInspector(type, payload = {}) {
    state.selectedObject = { type, ...payload };
    state.inspectorOpen = true;
    renderInspector();
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
    render();
  }

  function selectStep(index, options = {}) {
    const trace = currentTrace();
    const max = Math.max(0, (trace?.steps?.length || 1) - 1);
    state.stepIndex = Math.max(0, Math.min(max, index));
    if (!options.keepPlaying) stopPlayback();
    renderStep();
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
    renderSamples(trace);
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
    renderTensorViewport(trace);
    renderTileLens(trace);
    renderArchitectureFocus(trace);
    renderTimeline(trace);
    renderInfoPanel(trace);
    renderInspector(trace);
    syncPlayback();
  }

  function renderChrome(trace) {
    const step = currentStep(trace);
    const sourceLines = sourceLinesForTrace(trace);
    if (els.operatorMeta) els.operatorMeta.textContent = '';
    if (els.archReadout) els.archReadout.textContent = 'Ascend 910B';
    if (els.architectureMeta) els.architectureMeta.textContent = 'Ascend 910B';
    if (els.sourceMeta) {
      const suffix = trace.source?.partial ? '关键行' : `${sourceLines.length} 行`;
      els.sourceMeta.textContent = `${trace.source?.path || 'source'} · ${suffix}`;
    }
    if (els.visualTitle) els.visualTitle.textContent = 'Conv Execution Model';
    if (els.stepMeta) els.stepMeta.textContent = step ? `${state.stepIndex + 1}/${trace.steps.length}` : '';
    if (els.inspectorMeta) els.inspectorMeta.textContent = inspectorTypeLabel(state.selectedObject?.type) || '选中对象';
    if (els.timelineKicker) els.timelineKicker.textContent = `${step?.evidenceKind || 'unknown'} · order only`;
    if (els.statusText) els.statusText.textContent = step ? zh(step.label) : 'Ready';
    if (els.statusSample) els.statusSample.textContent = sampleShortName(trace);
    if (els.statusStep) els.statusStep.textContent = step ? `${state.stepIndex + 1}/${trace.steps.length}` : '0/0';
    if (els.statusArch) els.statusArch.textContent = 'Ascend 910B';
    syncFrameActions();
  }

  function renderSamples(trace) {
    const items = state.traces.map((item) => {
      const active = item.operator.id === trace.operator.id;
      return `<button class="tab-control-item ${active ? 'is-selected' : ''}" type="button" role="tab" aria-selected="${active ? 'true' : 'false'}" data-sample="${escapeHtml(item.operator.id)}">${escapeHtml(sampleShortName(item))}</button>`;
    }).join('');
    els.sampleList.innerHTML = `<div class="tab-control" role="tablist" aria-label="Operator samples">${items}</div>`;
    els.sampleList.querySelectorAll('[data-sample]').forEach((button) => {
      button.addEventListener('click', () => {
        state.inspectorOpen = false;
        state.selectedObject = null;
        selectSample(button.dataset.sample);
      });
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
    const lines = sourceLinesForTrace(trace);
    const keyLines = new Set((trace.source?.keyLines || trace.source?.lines || []).map((line) => line.line));
    trace.steps.forEach((step) => (step.sourceLines || []).forEach((line) => keyLines.add(line)));
    const stageByLine = new Map();
    trace.steps.forEach((step) => {
      const stage = trace.stages.find((item) => item.id === step.stageId) || null;
      (step.sourceLines || []).forEach((ln) => { if (!stageByLine.has(ln)) stageByLine.set(ln, stage); });
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
    updateSourceHighlight(trace);
  }

  function updateSourceHighlight(trace) {
    const container = els.sourceLines;
    if (!container) return;
    const activeLines = new Set(currentStep(trace)?.sourceLines || []);
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

  function sourceLinesForTrace(trace) {
    return trace?.source?.fullLines?.length ? trace.source.fullLines : trace?.source?.lines || [];
  }

  function sourceStageForLine(trace, lineNo) {
    const current = currentStep(trace);
    if (current?.sourceLines?.includes(lineNo)) return trace.stages.find((stage) => stage.id === current.stageId) || null;
    const step = trace.steps.find((item) => item.sourceLines?.includes(lineNo));
    return trace.stages.find((stage) => stage.id === step?.stageId) || null;
  }

  function stageKind(stage) {
    const id = String(stage?.id || '').toLowerCase();
    const label = String(stage?.label || '').toLowerCase();
    if (id.includes('sync') || label.includes('sync') || id.includes('init') || id.includes('launch')) return 'control';
    if (id.includes('copy') || id.includes('load') || id.includes('fixpipe')) return 'memory';
    if (id.includes('compute') || id.includes('mmad') || id.includes('matmul') || id.includes('leakyrelu')) return 'compute';
    return '';
  }

  function sourceLineTag(stage) {
    if (!stage) return 'trace';
    const map = {
      'host-launch': '启动',
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
    const kIndex = Number(step?.loop?.kIndex || 0);
    const kLoops = num(d.kLoopCount, 3);
    const finished = ['sync-m-fix', 'epilogue', 'copy-output'].includes(stage);
    const tracksK = ['load-k', 'mmad-init', 'mmad'].includes(stage);
    const kCurrent = finished ? kLoops : tracksK ? Math.min(kLoops, kIndex + 1) : 0;
    const blocks = convBufferBlocks(stage, kIndex);
    return {
      tensorViewport: {
        kind: 'conv2d',
        layout: 'conv2d',
        title: `Y[1, ${p.co}, ${p.ho}, ${p.wo}] · demo context`,
        axisLabels: ['Ho×Wo 输出位置', 'Co 输出通道', 'K=Ci×Kh×Kw 归约'],
        conv: {
          ho: p.ho, wo: p.wo, co: p.co, M: p.M, N: p.N, K: p.K,
          tileM: p.tileM, tileN: p.tileN, tileK: p.tileK,
          kIndex, kCurrent, kLoops,
        },
        highlight: {
          tone: stage.includes('mmad') ? 'reduction' : finished ? 'output' : 'input',
          label: stage === 'mmad-init' ? 'K0 Mmad · Bias C1→C2 unresolved' : `Output tile M[0:${p.tileM}], Co[0:${p.tileN}]`,
          sub: `${step?.evidenceKind || 'unknown'} · logical order only`,
        },
        operationChips: ['LoadData3D', 'Mmad', 'ReLU epilogue'],
      },
      onChipLens: { blocks },
      architectureFocus: { selectors: convSelectors(stage), routes: [], bufferBlocks: blocks },
    };
  }

  function convBufferBlocks(stage, kIndex) {
    if (stage === 'allocate') {
      return [
        { core: 'mem950-aic', buffer: 'L1', label: 'A1/B1 size ?', state: 'unknown', tone: 'avoided', cellRange: [0, 9], sourceTile: 'helpers return 0' },
        { core: 'mem950-aic', buffer: 'L0C', label: 'CO1 reserve', state: 'allocated', tone: 'output', cellRange: [0, 11], sourceTile: '32×16 fp32' },
      ];
    }
    if (stage === 'copy-inputs' || stage === 'sync-mte2-mte1') {
      return [{ core: 'mem950-aic', buffer: 'L1', label: 'Feature / Filter / Bias', state: 'inferred', tone: 'input', cellRange: [0, 29], sourceTile: 'GM → A1/B1/C1', operation: 'helper calls' }];
    }
    if (stage === 'load-k' || stage === 'sync-mte1-m') {
      return [
        { core: 'mem950-aic', buffer: 'L0A', label: `A2 · K${kIndex}`, state: 'loaded', tone: 'input', cellRange: [0, 15], sourceTile: '32×48', operation: 'LoadData3D' },
        { core: 'mem950-aic', buffer: 'L0B', label: `B2 · K${kIndex}`, state: 'inferred', tone: 'input', cellRange: [0, 11], sourceTile: '48×16', operation: 'helper missing' },
      ];
    }
    if (stage === 'mmad-init' || stage === 'mmad') {
      return [
        { core: 'mem950-aic', buffer: 'L0A', label: `A2 · K${kIndex}`, state: 'consumed', tone: 'input', cellRange: [0, 15], sourceTile: '32×48', operation: 'Mmad' },
        { core: 'mem950-aic', buffer: 'L0B', label: `B2 · K${kIndex}`, state: 'consumed', tone: 'input', cellRange: [0, 11], sourceTile: '48×16', operation: 'Mmad' },
        { core: 'mem950-aic', buffer: 'L0C', label: stage === 'mmad-init' ? 'CO1 + Bias ?' : 'CO1 partial', state: stage === 'mmad-init' ? 'unknown' : 'accumulating', tone: 'reduction', cellRange: [0, 23], sourceTile: 'M[0:32],Co[0:16]', operation: 'Mmad' },
      ];
    }
    if (['sync-m-fix', 'epilogue', 'copy-output'].includes(stage)) {
      return [{ core: 'mem950-aic', buffer: 'L0C', label: 'CO1 output', state: stage === 'copy-output' ? 'unknown' : 'committed', tone: 'output', cellRange: [0, 23], sourceTile: '32×16', operation: stage === 'epilogue' ? 'ReLU + Cast' : 'CopyOut' }];
    }
    return [];
  }

  function convSelectors(stage) {
    if (stage === 'copy-inputs') return ['[data-mem950-node="rail:GM"]', '#mem950-aic [data-aic-node="buffer:L1"]'];
    if (stage === 'load-k' || stage === 'sync-mte1-m') return ['#mem950-aic [data-aic-node="buffer:L1"]', '#mem950-aic [data-aic-node="buffer:L0A"]', '#mem950-aic [data-aic-node="buffer:L0B"]'];
    if (stage === 'mmad-init' || stage === 'mmad') return ['#mem950-aic [data-aic-node="buffer:L0A"]', '#mem950-aic [data-aic-node="buffer:L0B"]', '#mem950-aic [data-aic-node="cube:CUBE"]', '#mem950-aic [data-aic-node="buffer:L0C"]'];
    if (stage === 'epilogue' || stage === 'copy-output') return ['#mem950-aic [data-aic-node="buffer:L0C"]', '[data-mem950-node="rail:GM"]'];
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
    const canvas = els.tensorCanvas;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(520, Math.floor(rect.width || canvas.clientWidth || 760));
    const height = Math.max(360, Math.floor(rect.height || canvas.clientHeight || 480));
    const ctx = fitCanvas(canvas, width, height);
    drawTensorScene(ctx, width, height, visual);
    const tip = tensorViewportTip(visual);
    els.tensorStage.title = tip;
    els.viewportInfo.title = tip;
    if (els.tensorFallback) els.tensorFallback.hidden = true;
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
      parts.push('二维输出 tile + K reduction：K 是归约进度，不是输出 tensor 的第三维。');
    } else {
      parts.push('二维输出 tile：高亮区域 = 当前 M×N 输出分区；K 只作为规约进度展示。Cmd/Ctrl+滚轮缩放。');
    }
    if (visual.title) parts.push(visual.title);
    if ((visual.operationChips || []).length) parts.push(`当前操作：${visual.operationChips.join(', ')}`);
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
    if (visual.highlight) {
      drawTileLabels(ctx, width, [{ label: visual.highlight.label, tone: visual.highlight.tone }]);
    }
  }

  function drawConvExecution(ctx, width, height, visual) {
    const c = visual.conv || {};
    const scale = state.tensorView.scale || 1;
    const left = 42 + (state.tensorView.panX || 0);
    const top = 108 + (state.tensorView.panY || 0);
    const gridSize = Math.max(150, Math.min(270, height - 190, width * 0.42)) * scale;
    const cell = gridSize / Math.max(1, c.wo);
    ctx.fillStyle = getCss('--foreground');
    ctx.font = '700 13px Inter, sans-serif';
    ctx.fillText(`Output spatial tile · Ho×Wo = ${c.ho}×${c.wo}`, left, top - 24);
    for (let r = 0; r < c.ho; r += 1) {
      for (let col = 0; col < c.wo; col += 1) {
        const active = r * c.wo + col < c.tileM;
        ctx.fillStyle = active ? 'rgba(77,151,255,0.72)' : 'rgba(116,128,142,0.18)';
        ctx.strokeStyle = active ? 'rgba(184,218,255,0.88)' : getCss('--border-default');
        ctx.fillRect(left + col * cell, top + r * cell, cell - 2, cell - 2);
        ctx.strokeRect(left + col * cell, top + r * cell, cell - 2, cell - 2);
      }
    }
    ctx.fillStyle = getCss('--foreground-secondary');
    ctx.font = '600 11px ui-monospace, monospace';
    ctx.fillText(`M[0:${c.tileM}] · Co[0:${c.tileN}]`, left, top + gridSize + 22);

    const bx = Math.min(width - 230, left + gridSize + 54);
    const bw = Math.max(150, width - bx - 28);
    ctx.fillStyle = getCss('--foreground');
    ctx.font = '700 13px Inter, sans-serif';
    ctx.fillText(`K reduction · Ci×Kh×Kw = ${c.K}`, bx, top - 24);
    for (let i = 0; i < c.kLoops; i += 1) {
      const y = top + i * 58;
      const active = i < c.kCurrent;
      ctx.fillStyle = active ? 'rgba(255,154,84,0.72)' : 'rgba(165,175,185,0.09)';
      ctx.strokeStyle = active ? 'rgba(255,214,184,0.88)' : getCss('--border-default');
      ctx.beginPath();
      ctx.roundRect(bx, y, bw, 40, 6);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = active ? getCss('--foreground') : getCss('--foreground-muted');
      ctx.font = '700 11px ui-monospace, monospace';
      ctx.fillText(`K${i} [${i * c.tileK}:${Math.min(c.K, (i + 1) * c.tileK)}]`, bx + 12, y + 25);
    }
    ctx.fillStyle = getCss('--foreground-muted');
    ctx.font = '600 11px Inter, sans-serif';
    ctx.fillText('K is reduction progress, not a tensor axis.', bx, top + c.kLoops * 58 + 8);
    drawTensorCallout(ctx, width, height, visual.highlight, { x: left, y: 44 });
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
    const visual = visualStateForStep(trace, currentStep(trace));
    const blocks = visual.onChipLens?.blocks || visual.architectureFocus?.bufferBlocks || [];
    if (!blocks.length) {
      els.tileLens.innerHTML = '';
      return;
    }
    els.tileLens.innerHTML = blocks.slice(0, 3).map((block, index) => `
      <button class="avz-lens-card" type="button" data-block-index="${index}" title="${escapeHtml(block.state || 'loaded')} · ${escapeHtml(block.sourceTile || '')}">
        <header class="avz-lens-card__head">
          <span>${escapeHtml(block.label || block.buffer)}</span>
          <span>${escapeHtml(displayBufferTarget(block))}</span>
        </header>
        <div class="avz-lens-grid">${renderLensCells(block)}</div>
        <div class="avz-card-meta">${escapeHtml(block.state || 'loaded')} · ${escapeHtml(block.sourceTile || '')}</div>
      </button>
    `).join('');
    els.tileLens.querySelectorAll('[data-block-index]').forEach((button) => {
      button.addEventListener('click', () => {
        const block = blocks[Number(button.dataset.blockIndex) || 0];
        openInspector('buffer', { block });
      });
    });
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

  function renderTimeline(trace) {
    if (!trace || !els.timelineCanvas) return;
    const canvas = els.timelineCanvas;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(320, Math.floor(rect.width || canvas.clientWidth || 640));
    const height = 92;
    const ctx = fitCanvas(canvas, width, height);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = getCss('--surface-2');
    ctx.fillRect(0, 0, width, height);

    const helper = window.PtoSwimlaneTaskPattern;
    const palette = helper?.createTaskColormap?.() || null;
    const gap = 5;
    const left = 10;
    const top = 28;
    const barHeight = 34;
    const stepCount = trace.steps.length;
    const barWidth = Math.max(26, (width - left * 2 - gap * (stepCount - 1)) / stepCount);

    ctx.fillStyle = getCss('--foreground-muted');
    ctx.font = '600 10px Inter, sans-serif';
    ctx.fillText('Logical order · duration unavailable', left, 17);

    trace.steps.forEach((step, index) => {
      const stage = trace.stages.find((item) => item.id === step.stageId);
      const x = left + index * (barWidth + gap);
      const color = palette?.colorForLaneKind?.(step.unit) || helper?.colorFromColormap?.(stage?.label || step.stageId) || getCss('--primary-hover');
      drawTimelineStep(ctx, {
        x,
        y: top,
        width: barWidth,
        height: barHeight,
        color,
        selected: index === state.stepIndex,
        title: timelineStageTitle(stage, step),
        flow: timelineStageFlow(stage, step),
      });
      ctx.fillStyle = getCss('--foreground-muted');
      ctx.font = '600 9px ui-monospace, monospace';
      ctx.fillText(String(index + 1), x + 2, top + barHeight + 15);
    });

    canvas.onclick = (event) => {
      const bounds = canvas.getBoundingClientRect();
      const x = event.clientX - bounds.left;
      const index = Math.floor((x - left) / (barWidth + gap));
      if (index >= 0 && index < trace.steps.length) {
        state.selectedObject = { type: 'timeline step', stepIndex: index };
        state.inspectorOpen = true;
        selectStep(index);
      }
    };
  }

  function drawTimelineStep(ctx, item) {
    const radius = 5;
    const fg = getCss('--foreground');
    const muted = getCss('--foreground-muted');
    ctx.save();
    ctx.beginPath();
    ctx.roundRect(item.x, item.y, item.width, item.height, radius);
    ctx.globalAlpha = item.selected ? 0.34 : 0.22;
    ctx.fillStyle = item.color;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = item.selected ? fg : getCss('--border-strong');
    ctx.lineWidth = item.selected ? 1.8 : 1;
    ctx.stroke();

    ctx.save();
    ctx.beginPath();
    ctx.roundRect(item.x, item.y, item.width, item.height, radius);
    ctx.clip();
    ctx.fillStyle = fg;
    ctx.font = '700 12px Inter, Source Han Sans SC, sans-serif';
    drawFittedText(ctx, item.title, item.x + 8, item.y + 15, item.width - 16);
    ctx.fillStyle = muted;
    ctx.font = '600 9px ui-monospace, SFMono-Regular, Menlo, monospace';
    drawFittedText(ctx, item.flow, item.x + 8, item.y + 29, item.width - 16);
    ctx.restore();
    ctx.restore();
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
      'host-launch': 'Host -> GM',
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

  function renderInspector(trace = currentTrace()) {
    if (els.inspectorDrawer) els.inspectorDrawer.hidden = !state.inspectorOpen;
    if (!state.inspectorOpen) return;
    const step = currentStep(trace);
    const stage = trace.stages.find((item) => item.id === step?.stageId);
    if (!step || !stage) return;
    const visual = visualStateForStep(trace, step);
    const selected = selectedObjectNarrative(trace, step, stage, visual);
    els.inspectorMeta.textContent = selected.meta;
    const metrics = (step.metrics || []).map((metric) => (
      `${metric.label}=${metric.value}${metric.unit ? ` ${metric.unit}` : ''}`
    ));
    const events = [...(step.queueEvents || []), ...(step.syncEvents || [])];
    const regions = step.memoryRegions || [];
    const blocks = visual.architectureFocus?.bufferBlocks || [];

    els.inspector.innerHTML = `
      <section class="inspector-section inspector-soft-card avz-inspector-hero">
        <p class="avz-inspector-eyebrow">${escapeHtml(selected.meta)}</p>
        <h3>${escapeHtml(selected.title)}</h3>
        <p>${escapeHtml(selected.body)}</p>
      </section>

      <section class="inspector-section">
        <header class="inspector-section-head">
          <span class="inspector-section-title">当前步骤</span>
        </header>
        <p class="avz-inspector-copy">${escapeHtml(stepNarrative(trace, step, stage))}</p>
      </section>

      ${regions.length ? `
        <section class="inspector-section">
          <header class="inspector-section-head">
            <span class="inspector-section-title">数据位置</span>
          </header>
          <p class="avz-inspector-copy">${escapeHtml(memoryNarrative(regions, blocks))}</p>
        </section>
      ` : ''}

      ${events.length ? `
        <section class="inspector-section">
          <header class="inspector-section-head">
            <span class="inspector-section-title">队列和同步</span>
          </header>
          <p class="avz-inspector-copy">${escapeHtml(`这一帧会触发 ${formatListCn(events)}。这些事件用于表达 LocalTensor 入队/出队、buffer 释放，或者 AIC 与 AIV 之间的 flag 同步关系。`)}</p>
        </section>
      ` : ''}

      ${metrics.length ? `
        <section class="inspector-section">
          <header class="inspector-section-head">
            <span class="inspector-section-title">关键参数</span>
          </header>
          <dl class="avz-inspector-facts">
            ${(step.metrics || []).map((metric) => `
              <div>
                <dt>${escapeHtml(metric.label)}</dt>
                <dd>${escapeHtml(metric.value)}${metric.unit ? ` ${escapeHtml(metric.unit)}` : ''}</dd>
              </div>
            `).join('')}
          </dl>
        </section>
      ` : ''}
    `;
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

  function selectedObjectNarrative(trace, step, stage, visual) {
    const selected = state.selectedObject;
    if (selected.block) {
      const block = selected.block;
      return {
        title: block.label || block.buffer || '本地数据块',
        meta: 'Architecture Buffer',
        body: `${displayBufferTarget(block)} 正在承载 ${block.sourceTile || '当前 tile'}。状态是 ${block.state || 'unknown'}，对应操作是 ${block.operation || stage.label}。这个对象只表示片上 buffer 中的一小块驻留数据，不代表完整 tensor；输出分区与 K 规约进度要看中央 Conv2D 视口。`,
      };
    }
    if (selected.type === 'source') {
      return {
        title: `源码第 ${selected.line} 行`,
        meta: 'Source Line',
        body: `这一行源码被 trace 映射到当前执行步骤。选中它时，左侧代码、中央逻辑 tensor、右侧硬件链路和底部时间线会同步到同一帧，帮助你从代码语句追到实际搬运或计算的数据范围。`,
      };
    }
    if (selected.type === 'timeline step') {
      return {
        title: `时间线步骤 ${Number(selected.stepIndex || 0) + 1}`,
        meta: 'Execution Step',
        body: `这是执行序列中的一个可播放切片。时间线负责表达阶段顺序和当前帧位置；播放按钮只控制上一步、下一步、播放和重播，不改变 trace 本身。`,
      };
    }
    if (selected?.type === 'tensor') {
      return {
        title: 'Conv2D Output Tile + K Reduction',
        meta: 'Tensor View',
        body: '这是中央 Trace Visual 的 logical tensor 视口对象，用来显示当前步骤在 logical tensor 上触碰的 tile 或 element 区间。Inspector 这里只保留对象级说明；完整读图语义由右上角 info 面板承载。',
      };
    }
    return {
      title: zh(step.label),
      meta: unitLabel(step.unit || stage.unit) || 'Trace Step',
      body: zh(step.summary),
    };
  }

  function stepNarrative(trace, step, stage) {
    const sourceLines = (step.sourceLines || []).length ? `源码行 ${step.sourceLines.join(', ')}` : '当前源码片段';
    const opText = (stage.operations || []).length ? `涉及 ${formatListCn(stage.operations)}。` : '';
    return `${sourceLines} 对应 ${zh(stage.label)} 阶段。${zh(stage.description)} ${opText}${zh(step.summary)}`;
  }

  function visualNarrative(trace, step, stage, visual, axes, blocks) {
    const axisText = axes.length ? `当前轴名是 ${formatListCn(axes)}。` : '';
    const blockText = blocks.length
      ? `右侧架构图会把 ${blocks.length} 个片上本地数据块标在对应 buffer grid 上，例如 ${formatListCn(blocks.map((block) => `${displayBufferTarget(block)} ${block.label || ''}`))}。`
      : '当前步骤没有片上 buffer data block 需要单独标出。';
    return `${axisText}${tensorSceneNarrative(trace, step, stage, visual)} ${blockText}`;
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

    if (kind === 'fusion') {
      return `${intro} 这个 Fusion 同样画 C[M,N] 真实网格：AIC/Cube 先生产一块 singleCore 的 C 分区，CrossCoreSetFlag 把它标记为 ready，AIV0 和 AIV1 再分别消费这块分区的上半和下半 M rows 做 LeakyRelu——所以 AIV 步骤的高亮只覆盖一半行。AIC/AIV 的 handoff 是同步关系，要看底部 Execution Timeline 和右侧 Memory Architecture 的链路高亮，而不是 tensor 的某个轴。`;
    }

    return intro;
  }

  function timelineInfoNarrative(trace, step, stage) {
    const sourceLines = (step.sourceLines || []).length ? `源码行 ${step.sourceLines.join(', ')}` : '当前源码片段';
    return `底部时间线按 trace step 展示执行顺序，当前步骤是 ${state.stepIndex + 1}/${trace.steps.length}：${zh(step.label)}。阶段数据流是 ${timelineStageFlow(stage, step)}，对应 ${sourceLines}。播放只是在这些离散步骤之间移动当前帧，高亮块和右侧链路会跟着当前步骤切换。`;
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

  function memoryNarrative(regions, blocks) {
    const local = blocks.length
      ? `片上驻留点是 ${formatListCn(blocks.map((block) => `${block.core || 'core'} ${block.buffer || 'buffer'} 中的 ${block.label || 'tile'}`))}。`
      : '';
    return `这一步读写的数据区域包括 ${formatListCn(regions)}。${local}`;
  }

  function formatListCn(items) {
    const values = (items || []).filter(Boolean).map(String);
    if (values.length <= 1) return values[0] || '无';
    if (values.length === 2) return `${values[0]} 和 ${values[1]}`;
    return `${values.slice(0, -1).join('、')} 和 ${values[values.length - 1]}`;
  }

  function inspectorTypeLabel(type) {
    const labels = {
      tensor: '逻辑 Tensor 视口',
      buffer: '片上数据块',
      'architecture buffer': '架构图数据块',
      source: '源码行',
      'timeline step': '时间线步骤',
    };
    return labels[type] || type || '';
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
        renderTimeline(trace);
        refreshArchitectureViewport();
      });
    } catch (error) {
      if (els.statusText) els.statusText.textContent = error.message;
      if (els.inspector) els.inspector.innerHTML = `<div class="inspector-soft-card is-danger">${escapeHtml(error.message)}</div>`;
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
        renderTimeline(trace);
        refreshArchitectureViewport();
      });
    });
    [els.tensorStage, els.timelineCanvas, els.architectureViewport].forEach((target) => {
      if (target) state.resizeObserver.observe(target);
    });
  }

  document.addEventListener('DOMContentLoaded', init);
})();
