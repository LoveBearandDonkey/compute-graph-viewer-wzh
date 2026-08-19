(() => {
  'use strict';

  const FIXTURE_URL = 'data/fixtures/matmul_cann_samples.trace.json?v=20260814-matmul-v7';
  const SOURCE_URL = 'src/matmul/main.asc?v=20260811-matmul-v2';
  const PLAYBACK_IDS = {
    shell: 'matmul-floating-shell',
    toggle: 'matmul-floating-toggle',
    collapsedButton: 'matmul-floating-collapsed-btn',
    collapsedIcon: 'matmul-floating-collapsed-icon',
    controls: 'matmul-controls-row',
    stepBack: 'matmul-step-back-btn',
    play: 'matmul-play-btn',
    stepForward: 'matmul-step-fwd-btn',
    replay: 'matmul-replay-btn',
    scrubber: 'matmul-scrubber',
    scrubberLabel: 'matmul-scrubber-label',
    scrubberOpname: 'matmul-scrubber-opname',
    scrubberHover: 'matmul-scrubber-hover'
  };
  const MATMUL_CORE_COUNT = 8;
  const MATMUL_SHARED_AGGREGATE_GRID = Object.freeze({ rows: 4, columns: 4 });
  const MATMUL_INPUT_SHAPE_SCALE_SPEC = Object.freeze({
    axisScales: Object.freeze({ M: 256, K: 256, N: 256 }),
    hardBoundaries: Object.freeze({ M: Object.freeze([256]), N: Object.freeze([256]) }),
    reason: 'Aligned with the 256×256 output-tile/core partition; K uses the same comparison scale.'
  });
  const MATMUL_KERNEL_TILING_SCALE_SPEC = Object.freeze({
    axisScales: Object.freeze({ M: 256, K: 512, N: 256 }),
    hardBoundaries: Object.freeze({ M: Object.freeze([256]), K: Object.freeze([512]), N: Object.freeze([256]) }),
    reason: 'Aligned with the kernel tile: M/N output tile 256 and K-L1 slice 512.'
  });
  const state = {
    trace: null,
    context: null,
    source: [],
    sourceRole: 'all',
    executionView: 'instructions',
    matmulCoreIndex: 0,
    selectedTileIdx: 0,
    stepIndex: 0,
    frames: [],
    frameIndex: 0,
    playing: false,
    timer: null,
    playback: null,
    playbackHover: null,
    loopExpanded: { l1: true, l0: true },
    instructionIterationFocus: null,
    instructionL0IterationFocus: null,
    instructionOperationFocus: null,
    detailTab: 'tensor',
    selectedTensorId: 'tensor:a',
    matrixController: null,
    tensorTitleController: null,
    overviewMatrixControllers: {},
    overviewTitleControllers: {},
    copyTensorStepId: null,
    copyTensorId: null,
    copyMatrixControllers: { source: null, destination: null },
    copyTitleControllers: { source: null, destination: null },
  };

  const CPP_KEYWORDS = new Set([
    'alignas', 'alignof', 'asm', 'auto', 'bool', 'break', 'case', 'catch', 'char', 'class',
    'const', 'constexpr', 'continue', 'decltype', 'default', 'delete', 'do', 'double',
    'else', 'enum', 'explicit', 'extern', 'false', 'float', 'for', 'friend', 'if',
    'inline', 'long', 'namespace', 'new', 'noexcept', 'nullptr', 'operator', 'private',
    'protected', 'public', 'register', 'reinterpret_cast', 'return', 'short', 'signed',
    'sizeof', 'static', 'static_cast', 'struct', 'switch', 'template', 'this', 'throw',
    'true', 'try', 'typedef', 'typename', 'union', 'unsigned', 'using', 'virtual', 'void',
    'volatile', 'while'
  ]);
  const CPP_TYPES = new Set([
    'GM_ADDR', '__aicore__', '__global__', '__gm__', 'aclError', 'aclrtStream', 'bfloat16_t',
    'size_t', 'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t', 'int32_t', 'int64_t', 'float',
    'double', 'bool', 'char', 'void', 'AscendC', 'HardEvent', 'MmadParams'
  ]);
  const CPP_CONSTANTS = new Set(['ACL_SUCCESS', 'ACL_MEMCPY_DEVICE_TO_HOST', 'ACL_MEMCPY_HOST_TO_DEVICE', 'NULL']);
  const CPP_TOKEN_PATTERN = /#[A-Za-z_]+|\/\/.*|\/\*.*?\*\/|\/\*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b[A-Za-z_]\w*\b|\b\d+(?:\.\d+)?(?:[uUlLfF]*)?\b/g;

  const $ = (selector) => document.querySelector(selector);
  const escapeHtml = (value) => String(value == null ? '' : value)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');

  function highlightCpp(value, lexerState = { inBlockComment: false }) {
    const text = String(value == null ? '' : value);
    let html = '';
    let cursor = 0;
    if (lexerState.inBlockComment) {
      const closeIndex = text.indexOf('*/');
      if (closeIndex === -1) return '<span class="syntax-comment">' + escapeHtml(text) + '</span>';
      html += '<span class="syntax-comment">' + escapeHtml(text.slice(0, closeIndex + 2)) + '</span>';
      cursor = closeIndex + 2;
      lexerState.inBlockComment = false;
    }
    for (const match of text.matchAll(CPP_TOKEN_PATTERN)) {
      const token = match[0];
      const index = match.index ?? 0;
      if (index < cursor) continue;
      html += escapeHtml(text.slice(cursor, index));
      let kind = '';
      if (token.startsWith('#')) kind = '';
      else if (token.startsWith('//') || token.startsWith('/*')) kind = 'comment';
      else if (token.startsWith('"') || token.startsWith("'")) kind = 'string';
      else if (/^\d/.test(token)) kind = 'number';
      else if (CPP_CONSTANTS.has(token)) kind = 'constant';
      else if (CPP_TYPES.has(token)) kind = 'type';
      else if (CPP_KEYWORDS.has(token)) kind = 'keyword';
      else if (/^\s*\(/.test(text.slice(index + token.length))) kind = 'function';
      if (token === '/*') {
        html += escapeHtml(text.slice(index + token.length));
        lexerState.inBlockComment = true;
        cursor = text.length;
        break;
      }
      html += kind ? '<span class="syntax-' + kind + '">' + escapeHtml(token) + '</span>' : escapeHtml(token);
      cursor = index + token.length;
    }
    return html + escapeHtml(text.slice(cursor));
  }

  function roleForLine(line) {
    const ranges = state.trace?.source?.roleRanges;
    if (ranges?.kernel && line >= ranges.kernel[0] && line <= ranges.kernel[1]) return 'kernel';
    if (ranges?.host && line >= ranges.host[0] && line <= ranges.host[1]) return 'host';
    return 'shared';
  }

  function currentStep() {
    return state.trace.steps[state.stepIndex];
  }

  function coreTileSchedule(context, coreIndex) {
    if (!context) return [];
    const tiles = [];
    for (let tileIdx = coreIndex; tileIdx < context.outputTileNum; tileIdx += MATMUL_CORE_COUNT) {
      tiles.push(tileIdx);
    }
    return tiles;
  }

  function currentFrame() {
    const frame = state.frames[state.frameIndex] || { stepId: currentStep()?.id };
    if (!state.context || frame.tileIdx == null) return frame;
    const assignedTiles = coreTileSchedule(state.context, state.matmulCoreIndex);
    const selectedTile = assignedTiles.includes(state.selectedTileIdx) ? state.selectedTileIdx : assignedTiles[0];
    const tileIdx = selectedTile ?? Math.min(state.context.outputTileNum - 1, state.matmulCoreIndex);
    const mTileIdx = Math.floor(tileIdx / state.context.nTileNum);
    const nTileIdx = tileIdx % state.context.nTileNum;
    return {
      ...frame,
      tileIdx,
      mTileIdx,
      nTileIdx,
      curM: Math.min(state.context.baseM, state.context.M - mTileIdx * state.context.baseM),
      curN: Math.min(state.context.baseN, state.context.N - nTileIdx * state.context.baseN),
      aicIndex: state.matmulCoreIndex
    };
  }

  function outputTileLabel(context, tileIdx) {
    const mTileIdx = Math.floor(tileIdx / context.nTileNum);
    const nTileIdx = tileIdx % context.nTileNum;
    return {
      tileIdx,
      mTileIdx,
      nTileIdx,
      label: 'Tile ' + String(tileIdx).padStart(2, '0'),
      coordinate: 'M' + mTileIdx + '/N' + nTileIdx
    };
  }

  function ceilDiv(value, divisor) {
    return Math.ceil(value / divisor);
  }

  function deriveValidationContext(trace, scenario) {
    const shape = scenario?.shape || trace.tiling;
    const { baseM, baseN, baseK, kL1 } = trace.tiling;
    const { M, K, N } = shape;
    const mTileNum = ceilDiv(M, baseM);
    const nTileNum = ceilDiv(N, baseN);
    const outputTileNum = mTileNum * nTileNum;
    const tileIdx = outputTileNum - 1;
    const mTileIdx = Math.floor(tileIdx / nTileNum);
    const nTileIdx = tileIdx % nTileNum;
    const curM = M - mTileIdx * baseM;
    const curN = N - nTileIdx * baseN;
    const kSlices = [];
    let totalMmad = 0;
    for (let iter0 = 0; iter0 < ceilDiv(K, kL1); iter0 += 1) {
      const curKL1 = Math.min(kL1, K - iter0 * kL1);
      const l0Slices = [];
      for (let iter1 = 0; iter1 < ceilDiv(curKL1, baseK); iter1 += 1) {
        l0Slices.push(Math.min(baseK, curKL1 - iter1 * baseK));
      }
      kSlices.push({ iter0, curKL1, l0Slices, mmadsBefore: totalMmad });
      totalMmad += l0Slices.length;
    }
    return {
      scenario,
      M, K, N, baseM, baseN, baseK, kL1,
      mTileNum, nTileNum, outputTileNum, tileIdx, mTileIdx, nTileIdx, curM, curN,
      kSlices,
      kL1TileNum: kSlices.length,
      kL0TileNum: ceilDiv(kL1, baseK),
      mmadPerOutputTile: totalMmad,
      tailKL1: kSlices[kSlices.length - 1].curKL1,
      tailKL0: kSlices[kSlices.length - 1].l0Slices.at(-1)
    };
  }

  function validationScenarios() {
    return state.trace?.validationScenarios || [];
  }

  function currentScenario() {
    return validationScenarios().find((scenario) => scenario.id === state.scenarioId) || validationScenarios()[0];
  }

  function evidenceClass(evidence) {
    if (evidence === 'confirmed') return 'badge--success';
    if (evidence === 'derived') return 'badge--warning';
    if (evidence === 'inferred') return '';
    return '';
  }

  function buildExecutionFrames(trace, context) {
    const frames = [];
    const stepIds = new Set(trace.steps.map((step) => step.id));
    const tileIdx = context.tileIdx;
    const l1Count = context.kL1TileNum;
    const push = (stepId, frameContext = {}) => {
      if (stepIds.has(stepId)) frames.push({ stepId, ...frameContext });
    };

    const outputContext = { tileIdx, curM: context.curM, curN: context.curN };
    push('kernel-block-map', outputContext);
    for (let iter0 = 0; iter0 < l1Count; iter0 += 1) {
      const kSlice = context.kSlices[iter0];
      const l0Count = kSlice.l0Slices.length;
      const l1Context = { ...outputContext, iter0, curKL1: kSlice.curKL1, l0Count, mmadsBefore: kSlice.mmadsBefore };
      push('gm-to-l1', l1Context);
      push('sync-mte2-mte1', l1Context);
      for (let iter1 = 0; iter1 < l0Count; iter1 += 1) {
        const l0Context = { ...l1Context, iter1, curKL0: kSlice.l0Slices[iter1], mmadsBefore: kSlice.mmadsBefore + iter1 };
        push('l1-to-l0', l0Context);
        push('mmad', {
          ...l0Context,
          mmadOrdinal: kSlice.mmadsBefore + iter1 + 1,
          mmadMode: iter0 === 0 && iter1 === 0 ? 'initialize CO1' : 'accumulate CO1'
        });
      }
    }
    push('sync-m-fix', outputContext);
    push('l0c-to-gm', outputContext);
    return frames;
  }

  function frameLabel(frame = currentFrame()) {
    const step = state.trace.steps.find((item) => item.id === frame.stepId);
    if (!step) return 'Logical execution';
    if (frame.stepId === 'mmad') {
      return 'Mmad ' + frame.mmadOrdinal + '/' + state.context.mmadPerOutputTile + ' · ' + frame.mmadMode;
    }
    const context = [];
    if (frame.aicIndex != null && frame.tileIdx != null) context.push('AIC' + frame.aicIndex + ' · OT' + frame.tileIdx);
    if (frame.iter0 != null) context.push('iter0=' + frame.iter0);
    if (frame.iter1 != null) context.push('iter1=' + frame.iter1);
    return step.label + (context.length ? ' · ' + context.join(' · ') : '');
  }

  function primaryAction(step) {
    const value = String(step.action || 'Configure');
    if (value.includes('Store')) return 'Store';
    if (value.includes('Compute')) return 'Compute';
    if (value.includes('Sync')) return 'Sync';
    if (value.includes('Move')) return 'Move';
    return 'Configure';
  }

  function renderValidationControls() {
    const scenario = currentScenario();
    const context = state.context;
    if (!scenario || !context) return;
    $('#validationCases').innerHTML = validationScenarios()
      .filter((item) => item.id !== 'combined-tail')
      .map((item) => {
        const active = item.id === state.scenarioId;
        return '<button class="tab-control-item' + (active ? ' is-selected' : '') + '" type="button" data-validation-scenario="' +
          escapeHtml(item.id) + '" aria-pressed="' + active + '">' + escapeHtml(item.label) + '</button>';
      }).join('');
    $('#validationCases').querySelectorAll('[data-validation-scenario]').forEach((button) => {
      button.addEventListener('click', () => selectValidationScenario(button.dataset.validationScenario));
    });
    $('#shapeReadout').textContent = 'M=' + context.M + ' · K=' + context.K + ' · N=' + context.N;
    $('#tileReadout').textContent = 'baseM/N/K=' + context.baseM + '/' + context.baseN + '/' + context.baseK + ' · BF16';
    $('#validationSummary').innerHTML =
      '<span><strong>Last output tile</strong>#' + context.tileIdx + ' · curM=' + context.curM + ' · curN=' + context.curN + '</span>' +
      '<span><strong>Final K slices</strong>' + context.tailKL1 + ' → ' + context.kSlices.at(-1).l0Slices.join(' / ') + '</span>' +
      '<span class=\"badge badge--warning\">' + escapeHtml(scenario.evidence || 'derived') + '</span>';
    $('#validationDescription').textContent = scenario.description || '';
  }

  function selectValidationScenario(id) {
    const scenario = validationScenarios().find((item) => item.id === id);
    if (!scenario || scenario.id === state.scenarioId) return;
    setPlaying(false);
    state.scenarioId = scenario.id;
    state.context = deriveValidationContext(state.trace, scenario);
    state.frames = buildExecutionFrames(state.trace, state.context);
    const focusIndex = scenario.focus === 'last-mmad'
      ? state.frames.map((frame) => frame.stepId).lastIndexOf('mmad')
      : 0;
    renderValidationControls();
    selectFrame(Math.max(0, focusIndex));
  }

  function sourceLineIsVisible(line) {
    return state.sourceRole === 'all' || roleForLine(line) === state.sourceRole;
  }

  function linkedStepIndexes(line) {
    const sourceFileId = state.trace?.source?.id || 'main.asc';
    return state.trace.steps.reduce((indexes, step, index) => {
      const linked = (step.sourceRefs || []).some((ref) => ref.fileId === sourceFileId && ref.lines.includes(line));
      if (linked) indexes.push(index);
      return indexes;
    }, []);
  }

  function sourceTagKind(stage) {
    if (!stage) return 'control';
    if (stage.unit === 'mte1' || stage.unit === 'mte2') return 'memory';
    if (stage.unit === 'cube' || stage.unit === 'output') return 'compute';
    return 'control';
  }

  function sourceTagForLine(line) {
    const [stepIndex] = linkedStepIndexes(line);
    if (stepIndex == null) return null;
    const step = state.trace.steps[stepIndex];
    const stage = state.trace.stages.find((item) => item.id === step.stageId) || null;
    return {
      label: stage?.semanticLabel || stage?.label || step.label,
      kind: sourceTagKind(stage)
    };
  }

  function renderSource() {
    const active = currentStep();
    const activeLines = new Set(active.sourceRefs.flatMap((ref) => ref.lines));
    const visible = state.source.filter((entry) => sourceLineIsVisible(entry.line));
    const lexerState = { inBlockComment: false };
    $('#sourceLines').innerHTML = visible.map((entry) => {
      const current = activeLines.has(entry.line);
      const stepIndexes = linkedStepIndexes(entry.line);
      const linked = stepIndexes.length > 0;
      const taggable = linked && entry.text.trim().length > 0;
      const element = taggable ? 'button' : 'div';
      const linkedLabels = stepIndexes.map((index) => state.trace.steps[index].label).join(' / ');
      const attrs = taggable
        ? ' type="button" data-source-line="' + entry.line + '" title="' + escapeHtml(linkedLabels) + '" aria-label="Line ' + entry.line + ' · ' + escapeHtml(linkedLabels) + '"'
        : ' data-source-line="' + entry.line + '"';
      const tag = taggable ? sourceTagForLine(entry.line) : null;
      const tagMarkup = tag
        ? '<span class="matmul-source-line__tag is-' + tag.kind + '">' + escapeHtml(tag.label) + '</span>'
        : '<span class="matmul-source-line__tag-placeholder" aria-hidden="true"></span>';
      return '<' + element + ' class="matmul-source-line' + (taggable ? ' is-linked' : '') + (current ? ' is-current' : '') + '"' + attrs + '>' +
        '<span class="matmul-source-line__number">' + entry.line + '</span>' +
        '<code class="matmul-source-line__text">' + highlightCpp(entry.text, lexerState) + '</code>' +
        tagMarkup + '</' + element + '>';
    }).join('');
    $('#sourceLines').querySelectorAll('button[data-source-line]').forEach((button) => {
      button.addEventListener('click', () => selectSourceLine(Number(button.dataset.sourceLine)));
    });
    $('#sourceLines .is-current')?.scrollIntoView({ block: 'center' });
  }

  function renderEvents() {
    const stepId = currentStep().id;
    $('#eventList').innerHTML = state.trace.events.map((event) => {
      const isSet = event.setStepId === stepId;
      const isWait = event.waitStepId === stepId;
      const current = isSet || isWait;
      const phase = isSet && isWait ? 'SET + WAIT' : isSet ? 'SET' : isWait ? 'WAIT' : event.scope;
      return '<div class="matmul-event-item' + (current ? ' is-current' : '') + '">' +
        '<div class="matmul-event-item__header"><strong>' + escapeHtml(event.eventType) + '</strong><span class="badge ' + (current ? 'badge--warning' : '') + '">' + escapeHtml(phase) + '</span></div>' +
        '<div class="matmul-event-item__route"><span>' + escapeHtml(event.producerEngine) + '</span><b>→</b><span>' + escapeHtml(event.consumerEngine) + '</span></div>' +
        '<p>' + escapeHtml(event.explanation) + '</p>' +
      '</div>';
    }).join('');
  }

  function tensorView(id) {
    const baseView = state.trace.tensorViews.find((view) => view.id === id) || state.trace.tensorViews[0];
    const frame = currentFrame();
    const context = state.context;
    const curM = frame.curM ?? context.curM;
    const curN = frame.curN ?? context.curN;
    const curKL1 = frame.curKL1 ?? Math.min(context.kL1, context.K);
    const curKL0 = frame.curKL0 ?? Math.min(context.baseK, curKL1);
    const dimensions = {
      'tensor:a': { logicalShape: [context.M, context.K], tileShape: [curM, curKL1], grid: { rowSpan: context.baseM, columnSpan: context.kL1 }, sizeBytes: context.M * context.K * 2 },
      'tensor:b': { logicalShape: [context.K, context.N], tileShape: [curKL1, curN], grid: { rowSpan: context.kL1, columnSpan: context.baseN }, sizeBytes: context.K * context.N * 2 },
      'tensor:c': { logicalShape: [context.M, context.N], tileShape: [curM, curN], grid: { rowSpan: context.baseM, columnSpan: context.baseN }, sizeBytes: context.M * context.N * 2 },
      'tensor:a1': { logicalShape: [curM, curKL1], tileShape: [curM, curKL0], grid: { rowSpan: curM, columnSpan: context.baseK }, sizeBytes: curM * curKL1 * 2 },
      'tensor:b1': { logicalShape: [curKL1, curN], tileShape: [curKL0, curN], grid: { rowSpan: context.baseK, columnSpan: curN }, sizeBytes: curKL1 * curN * 2 },
      'tensor:a2': { logicalShape: [curM, curKL0], tileShape: [curM, curKL0], grid: { rowSpan: curM, columnSpan: curKL0 }, sizeBytes: curM * curKL0 * 2 },
      'tensor:b2': { logicalShape: [curKL0, curN], tileShape: [curKL0, curN], grid: { rowSpan: curKL0, columnSpan: curN }, sizeBytes: curKL0 * curN * 2 },
      'tensor:co1': { logicalShape: [curM, curN], tileShape: [16, 16], grid: { rowSpan: 16, columnSpan: 16 }, sizeBytes: curM * curN * 4 }
    };
    return { ...baseView, ...(dimensions[baseView.id] || {}) };
  }

  function tensorLabel(id) {
    return tensorView(id)?.label || id;
  }

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes)) return 'size n/a';
    const units = ['B', 'KiB', 'MiB', 'GiB'];
    let value = bytes;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
      value /= 1024;
      unit += 1;
    }
    return (value >= 10 ? Math.round(value) : Math.round(value * 10) / 10) + ' ' + units[unit];
  }

  function resolvedDataFlows() {
    const frame = currentFrame();
    const context = state.context;
    const curM = frame.curM ?? context.curM;
    const curN = frame.curN ?? context.curN;
    const curKL1 = frame.curKL1 ?? Math.min(context.kL1, context.K);
    const curKL0 = frame.curKL0 ?? Math.min(context.baseK, curKL1);
    const bytesById = {
      'flow:a-a1': curM * curKL1 * 2,
      'flow:b-b1': curKL1 * curN * 2,
      'flow:a1-a2': curM * curKL0 * 2,
      'flow:b1-b2': curKL0 * curN * 2,
      'flow:co1-c': curM * curN * 2
    };
    return state.trace.dataFlows.map((flow) => ({
      ...flow,
      bytes: Object.hasOwn(bytesById, flow.id) ? bytesById[flow.id] : flow.bytes
    }));
  }

  function tensorState(view) {
    const stepId = currentStep().id;
    const frame = currentFrame();
    if (view.id === 'tensor:co1') {
      if (stepId === 'mmad') return frame.mmadOrdinal === 1 ? 'initialize CO1' : 'accumulated ' + frame.mmadOrdinal + '/' + state.context.mmadPerOutputTile;
      if (stepId === 'l1-to-l0' && (frame.iter0 > 0 || frame.iter1 > 0)) {
        return 'accumulated ' + frame.mmadsBefore + '/' + state.context.mmadPerOutputTile;
      }
      if (stepId === 'sync-m-fix') return 'ready';
      if (stepId === 'l0c-to-gm' || stepId === 'host-verify') return 'written';
      return stepId === 'kernel-block-map' ? 'allocated' : 'not active';
    }
    const lifecycle = view.lifecycle.find((event) => event.stepId === stepId);
    if (lifecycle) return lifecycle.state;
    if (view.memoryTier === 'GM') return stepId === 'host-args' ? 'declared' : 'ready';
    if ((view.id === 'tensor:a1' || view.id === 'tensor:b1') && stepId === 'mmad') return 'loaded';
    if ((view.id === 'tensor:a2' || view.id === 'tensor:b2') && stepId === 'sync-m-fix') return 'released';
    return 'not active';
  }

  function tensorDirection(view) {
    const flow = resolvedDataFlows().find((item) => item.stepId === currentStep().id && (item.from === view.id || item.to === view.id));
    if (!flow) return '';
    return tensorLabel(flow.from) + ' → ' + tensorLabel(flow.to) + ' · ' + flow.transformation;
  }

  const MATMUL_ALLOCATION_LANES = [
    { id: 'l1', title: 'L1', capacityBytes: 512 * 1024, tensorIds: ['tensor:a1', 'tensor:b1'] },
    { id: 'l0a', title: 'L0A', capacityBytes: 64 * 1024, tensorIds: ['tensor:a2'] },
    { id: 'l0b', title: 'L0B', capacityBytes: 64 * 1024, tensorIds: ['tensor:b2'] },
    { id: 'l0c', title: 'L0C', capacityBytes: 512 * 1024, tensorIds: ['tensor:co1'] }
  ];

  const MATMUL_ALLOCATION_DETAILS = {
    'tensor:a1': { position: 'A1 / L1', filledBy: 'Copy GM → L1', consumedBy: 'Copy L1 → L0A', lifetime: 'one K-L1 slice' },
    'tensor:b1': { position: 'B1 / L1', filledBy: 'Copy GM → L1', consumedBy: 'Copy L1 → L0B', lifetime: 'one K-L1 slice' },
    'tensor:a2': { position: 'A2 / L0A', filledBy: 'Copy L1 → L0A', consumedBy: 'Mmad', lifetime: 'one K-L0 slice' },
    'tensor:b2': { position: 'B2 / L0B', filledBy: 'Copy L1 → L0B', consumedBy: 'Mmad', lifetime: 'one K-L0 slice' },
    'tensor:co1': { position: 'CO1 / L0C', filledBy: 'Mmad', consumedBy: 'Copy L0C → GM', lifetime: 'Acc0 → Acc8', writeLabel: 'Written by' }
  };

  function matmulAllocationTensor(id) {
    const view = tensorView(id);
    const detail = MATMUL_ALLOCATION_DETAILS[id];
    if (!view || !detail) return null;
    const localL1A = tensorView('tensor:a1');
    const start = id === 'tensor:b1' ? localL1A?.sizeBytes || 0 : 0;
    return {
      id,
      name: view.label,
      position: detail.position,
      start,
      end: start + view.sizeBytes,
      size: view.sizeBytes,
      dtypeLabel: String(view.dtype || '').toUpperCase(),
      shapeLabel: '[' + view.logicalShape.join(',') + ']',
      alignmentLabel: 'Not specified',
      format: view.format,
      logicalIdentity: view.label + ' ' + view.axes.join(' / '),
      filledBy: detail.filledBy,
      consumedBy: detail.consumedBy,
      lifetime: detail.lifetime,
      writeLabel: detail.writeLabel,
      confidence: id === 'tensor:a1' || id === 'tensor:b1' ? 'derived' : 'confirmed'
    };
  }

  function matmulAllocationBlock(tensor, referenceBytes) {
    if (!tensor) return '';
    const startRatio = referenceBytes > 0 ? (tensor.start / referenceBytes) * 100 : 0;
    const sizeRatio = referenceBytes > 0 ? (tensor.size / referenceBytes) * 100 : 0;
    return '<div class="avz-memory-block" tabindex="0" data-allocation-tensor="' + escapeHtml(tensor.id) + '" style="--avz-block-start:' + startRatio + '%;--avz-block-size:' + sizeRatio + '%" aria-label="' + escapeHtml(tensor.name + ', ' + tensor.position + ', [' + tensor.start + ',' + tensor.end + '), ' + tensor.size + ' bytes') + '"></div>';
  }

  function matmulAllocationCapacityLabel(bytes) {
    if (bytes % (1024 * 1024) === 0) return bytes / (1024 * 1024) + ' MiB';
    if (bytes % 1024 === 0) return bytes / 1024 + ' KiB';
    return bytes + ' B';
  }

  function matmulAllocationUsageLabel(usedBytes, capacityBytes) {
    return (capacityBytes > 0 ? (usedBytes / capacityBytes) * 100 : 0).toFixed(1) + '%';
  }

  function matmulAllocationTicks(tensors, referenceBytes, capacityBytes) {
    const ticks = [{ address: tensors[0]?.start || 0, position: 0 }];
    tensors.forEach((tensor) => ticks.push({
      address: tensor.end,
      position: referenceBytes > 0 ? Math.min(100, (tensor.end / referenceBytes) * 100) : 0
    }));
    return '<div class="avz-memory-scale__used">' + ticks.map((tick, index) => {
      const previousPosition = ticks[index - 1]?.position;
      const nextPosition = ticks[index + 1]?.position;
      const crowded = index > 0 && ((previousPosition !== undefined && tick.position - previousPosition < 18)
        || (nextPosition !== undefined && nextPosition - tick.position < 18));
      const classes = [
        index === 0 ? 'is-start' : '',
        index === ticks.length - 1 ? 'is-reference-end' : '',
        crowded && index % 2 === 1 ? 'is-staggered' : ''
      ].filter(Boolean).join(' ');
      return '<span class="avz-memory-tick' + (classes ? ' ' + classes : '') + '" style="--avz-tick-position:' + tick.position + '%"><code>' + tick.address + '</code></span>';
    }).join('') + '</div><div class="avz-memory-scale__capacity"><span class="avz-memory-tick is-capacity-end"><code>' + capacityBytes + '</code></span></div>';
  }

  function matmulAllocationLane(lane, tensorsById, referenceBytes) {
    const tensors = lane.tensorIds.map((id) => tensorsById[id]).filter(Boolean);
    if (!tensors.length) return '';
    const usedBytes = Math.max(...tensors.map((tensor) => tensor.end));
    return '<section class="avz-memory-lane avz-memory-lane--' + lane.id + '" aria-label="' + escapeHtml(lane.title + ' address space') + '">' +
      '<div class="avz-memory-lane__map"><strong class="avz-memory-lane__title">' + escapeHtml(lane.title) + '</strong>' +
      '<div class="avz-memory-lane__plot"><div class="avz-memory-scale">' + matmulAllocationTicks(tensors, referenceBytes, lane.capacityBytes) + '</div>' +
      '<div class="avz-memory-capacity-bar"><div class="avz-memory-used">' + tensors.map((tensor) => matmulAllocationBlock(tensor, referenceBytes)).join('') + '</div>' +
      '<div class="avz-memory-collapsed" aria-label="Collapsed unused address space"><span>⋯</span></div></div></div></div>' +
      '<div class="avz-memory-lane__summary"><div class="avz-memory-lane__names">' + tensors.map((tensor) => '<code>' + escapeHtml(tensor.name) + '</code>').join('<span aria-hidden="true">·</span>') + '</div>' +
      '<div class="avz-memory-lane__capacity"><strong>' + matmulAllocationUsageLabel(usedBytes, lane.capacityBytes) + '</strong><span aria-hidden="true"></span><strong>' + matmulAllocationCapacityLabel(lane.capacityBytes) + '</strong></div></div></section>';
  }

  function matmulAllocationTooltipMarkup(tensor) {
    const rows = [
      ['Position', tensor.position], ['Address', '[' + tensor.start + ',' + tensor.end + ')'], ['Size', tensor.size + ' B'],
      ['Shape', tensor.shapeLabel], ['dtype', tensor.dtypeLabel], ['format', tensor.format],
      ['Logical view', tensor.logicalIdentity], ['Alignment', tensor.alignmentLabel],
      [tensor.writeLabel || 'Filled by', tensor.filledBy], ['Consumed by', tensor.consumedBy], ['Lifetime', tensor.lifetime],
      ['Evidence', tensor.confidence]
    ];
    return '<strong>' + escapeHtml(tensor.name) + '</strong>' + rows.map((row) => '<div><span>' + escapeHtml(row[0]) + '</span><code>' + escapeHtml(row[1]) + '</code></div>').join('');
  }

  function showMatmulAllocationTooltip(event) {
    const tooltip = $('#matmulAllocationTooltip');
    const tensor = matmulAllocationTensor(event.currentTarget.dataset.allocationTensor);
    if (!tooltip || !tensor) return;
    tooltip.innerHTML = matmulAllocationTooltipMarkup(tensor);
    tooltip.hidden = false;
    positionMatmulAllocationTooltip(event);
  }

  function positionMatmulAllocationTooltip(event) {
    const tooltip = $('#matmulAllocationTooltip');
    const root = $('#matmulMemoryAllocation');
    if (!tooltip || !root || tooltip.hidden) return;
    const rootRect = root.getBoundingClientRect();
    const targetRect = event.currentTarget.getBoundingClientRect();
    const x = 'clientX' in event ? event.clientX - rootRect.left + 12 : targetRect.left - rootRect.left + 12;
    const y = 'clientY' in event ? event.clientY - rootRect.top + 12 : targetRect.bottom - rootRect.top + 8;
    tooltip.style.left = Math.min(x, Math.max(8, rootRect.width - tooltip.offsetWidth - 12)) + 'px';
    tooltip.style.top = Math.min(y, Math.max(8, rootRect.height - tooltip.offsetHeight - 12)) + 'px';
  }

  function hideMatmulAllocationTooltip() {
    const tooltip = $('#matmulAllocationTooltip');
    if (tooltip) tooltip.hidden = true;
  }

  function renderMatmulMemoryAllocation() {
    const mount = $('#matmulMemoryAllocation');
    if (!mount) return;
    const tensors = Object.fromEntries(Object.keys(MATMUL_ALLOCATION_DETAILS).map((id) => [id, matmulAllocationTensor(id)]));
    const referenceBytes = Math.max(...MATMUL_ALLOCATION_LANES.map((lane) => {
      const laneTensors = lane.tensorIds.map((id) => tensors[id]).filter(Boolean);
      return laneTensors.length ? Math.max(...laneTensors.map((tensor) => tensor.end)) : 0;
    }));
    mount.innerHTML = '<div class="avz-memory-lanes">' + MATMUL_ALLOCATION_LANES.map((lane) => matmulAllocationLane(lane, tensors, referenceBytes)).join('') + '</div>' +
      '<footer class="avz-memory-legend"><span><i class="avz-memory-legend__tensor"></i> Tensor 色块表示 LocalTensor 视图，数据尚未装载</span>' +
      '<span><i class="avz-memory-legend__collapsed"></i> 斜线区域表示折叠的未使用容量，不按真实剩余容量比例绘制</span>' +
      '<span>所有 Tensor 共用同一比例尺；当前最大占用 ' + referenceBytes + ' B 映射为泳道宽度的 80%。</span>' +
      '<span>所有地址区间均为左闭右开 <code>[start, end)</code>。</span></footer>' +
      '<div class="avz-memory-tooltip" id="matmulAllocationTooltip" role="tooltip" hidden></div>';
    mount.querySelectorAll('[data-allocation-tensor]').forEach((block) => {
      block.addEventListener('pointerenter', showMatmulAllocationTooltip);
      block.addEventListener('pointermove', positionMatmulAllocationTooltip);
      block.addEventListener('pointerleave', hideMatmulAllocationTooltip);
      block.addEventListener('focus', showMatmulAllocationTooltip);
      block.addEventListener('blur', hideMatmulAllocationTooltip);
    });
  }

  function tileCoordinates() {
    const frame = currentFrame();
    const tileIdx = frame.tileIdx ?? 0;
    const nTileNum = state.context.nTileNum;
    return {
      mTile: Math.floor(tileIdx / nTileNum),
      nTile: tileIdx % nTileNum,
      iter0: frame.iter0 ?? 0,
      iter1: frame.iter1 ?? 0
    };
  }

  function sharedAggregateGridForViews(views) {
    const validViews = views.filter((view) => Array.isArray(view?.logicalShape) && view.logicalShape.length >= 2);
    if (!validViews.length) return MATMUL_SHARED_AGGREGATE_GRID;
    return {
      rows: Math.max(1, Math.min(MATMUL_SHARED_AGGREGATE_GRID.rows, ...validViews.map((view) => Math.floor(view.logicalShape[0])))),
      columns: Math.max(1, Math.min(MATMUL_SHARED_AGGREGATE_GRID.columns, ...validViews.map((view) => Math.floor(view.logicalShape[1]))))
    };
  }

  function matrixSceneForTensor(view, aggregation = null) {
    const rows = view.logicalShape[0];
    const columns = view.logicalShape[1];
    const tileRowSpan = view.grid.rowSpan;
    const tileColumnSpan = view.grid.columnSpan;
    const aggregateCells = window.PtoMatrixCanvas.createAggregateLayoutCells(
      { rows, columns },
      {
        blockRows: aggregation?.rowSpan ?? tileRowSpan,
        blockColumns: aggregation?.columnSpan ?? tileColumnSpan,
        thumbnailRows: aggregation?.rows,
        thumbnailColumns: aggregation?.columns,
        intensity: 0.5
      }
    );
    const coords = tileCoordinates();
    let activeRow = 0;
    let activeColumn = 0;
    if (view.id === 'tensor:a') [activeRow, activeColumn] = [coords.mTile * tileRowSpan, coords.iter0 * tileColumnSpan];
    if (view.id === 'tensor:b') [activeRow, activeColumn] = [coords.iter0 * tileRowSpan, coords.nTile * tileColumnSpan];
    if (view.id === 'tensor:c') [activeRow, activeColumn] = [coords.mTile * tileRowSpan, coords.nTile * tileColumnSpan];
    if (view.id === 'tensor:a1') [activeRow, activeColumn] = [0, coords.iter1 * tileColumnSpan];
    if (view.id === 'tensor:b1') [activeRow, activeColumn] = [coords.iter1 * tileRowSpan, 0];

    const showCurrentTile = currentStep().id !== 'host-args';
    const outputStageActive = ['mmad', 'sync-m-fix', 'l0c-to-gm'].includes(currentStep().id);
    const tone = view.role === 'output' ? 'output' : view.role === 'reduction' ? 'reduction' : 'input';
    const cells = aggregateCells.map((aggregateCell) => {
      const row = aggregateCell.row;
      const column = aggregateCell.column;
      const cellRowSpan = aggregateCell.rowSpan;
      const cellColumnSpan = aggregateCell.columnSpan;
      const isActiveAggregate = view.id === 'tensor:co1'
        ? showCurrentTile && outputStageActive
        : showCurrentTile
          && activeRow >= row && activeRow < row + cellRowSpan
          && activeColumn >= column && activeColumn < column + cellColumnSpan;
      const states = [];
      if (isActiveAggregate) states.push('current');
      else if (view.id === 'tensor:co1' && tensorState(view).startsWith('accumulated')) states.push('written');
      return {
        ...aggregateCell,
        id: view.id + ':' + aggregateCell.id,
        tone,
        summary: {
          ...aggregateCell.summary,
          intensity: 0.5
        },
        states
      };
    });
    return { extent: { rows, columns }, axes: { rows: view.axes[0], columns: view.axes[1] }, cells };
  }

  function resolveMatmulInputShapeScale(views) {
    return window.PtoMatrixCanvas.resolveSharedAggregateScale({
      tensors: views.map((view) => ({
        id: view.id,
        extent: { rows: view.logicalShape[0], columns: view.logicalShape[1] },
        axes: { rows: view.axes[0], columns: view.axes[1] }
      })),
      axisScales: MATMUL_INPUT_SHAPE_SCALE_SPEC.axisScales,
      hardBoundaries: MATMUL_INPUT_SHAPE_SCALE_SPEC.hardBoundaries
    });
  }

  function resolveMatmulKernelTilingScale(views) {
    return window.PtoMatrixCanvas.resolveSharedAggregateScale({
      tensors: views.map((view) => ({
        id: view.id,
        extent: { rows: view.logicalShape[0], columns: view.logicalShape[1] },
        axes: { rows: view.axes[0], columns: view.axes[1] }
      })),
      axisScales: MATMUL_KERNEL_TILING_SCALE_SPEC.axisScales,
      hardBoundaries: MATMUL_KERNEL_TILING_SCALE_SPEC.hardBoundaries
    });
  }

  function tensorTitleScene(view) {
    const frame = currentFrame();
    const context = [];
    if (frame.tileIdx != null) context.push('tileIdx=' + frame.tileIdx);
    if (frame.iter0 != null) context.push('K-L1=' + frame.iter0 + '/' + (state.context.kL1TileNum - 1));
    if (frame.iter1 != null) context.push('K-L0=' + frame.iter1 + '/' + ((frame.l0Count || state.context.kL0TileNum) - 1));
    return {
      label: view.label,
      role: view.role,
      logicalShape: { label: 'logical', dims: view.logicalShape },
      tileShape: { label: 'tile', dims: view.tileShape },
      dtype: view.dtype,
      format: view.format,
      axes: view.axes,
      memory: { tier: view.memoryTier, sizeBytes: view.sizeBytes },
      state: tensorState(view),
      direction: tensorDirection(view),
      provenance: view.provenance,
      step: {
        phase: frameLabel(),
        operationChips: [primaryAction(currentStep())],
        stepIndex: state.frameIndex + 1,
        totalSteps: state.frames.length
      },
      constraints: view.constraints,
      status: context.length ? context.join(' · ') : frameLabel()
    };
  }

  function overviewDomKey(tensorId) {
    return tensorId.replace(/^tensor:/, '').replace(/[^a-zA-Z0-9]+/g, '-').toLowerCase();
  }

  function matmulTilingPartition(view, tilingScale) {
    const plan = tilingScale?.tensors.find((item) => item.id === view.id);
    if (!plan || !state.context) return null;
    const coords = tileCoordinates();
    const frame = currentFrame();
    const tileIndex = frame.tileIdx ?? (coords.mTile * state.context.nTileNum + coords.nTile);
    const starts = {
      M: coords.mTile * state.context.baseM,
      K: coords.iter0 * state.context.kL1,
      N: coords.nTile * state.context.baseN
    };
    const rowStart = starts[view.axes[0]] ?? 0;
    const columnStart = starts[view.axes[1]] ?? 0;
    const rowEnd = Math.min(view.logicalShape[0], rowStart + plan.rowSpan);
    const columnEnd = Math.min(view.logicalShape[1], columnStart + plan.columnSpan);
    return {
      blockCount: state.context.outputTileNum,
      blockLength: state.context.baseM + '×' + state.context.baseN + ' output',
      tileLength: plan.rowSpan + '×' + plan.columnSpan,
      tileIndex,
      tileRange: view.axes[0] + '[' + rowStart + ':' + rowEnd + ') · ' + view.axes[1] + '[' + columnStart + ':' + columnEnd + ')'
    };
  }

  function matmulTilingOverlay(view, tilingScale) {
    const plan = tilingScale?.tensors.find((item) => item.id === view.id);
    if (!plan || !state.context) return '';
    const coords = tileCoordinates();
    const frame = currentFrame();
    const starts = {
      M: coords.mTile * state.context.baseM,
      K: coords.iter0 * state.context.kL1,
      N: coords.nTile * state.context.baseN
    };
    const rowStart = starts[view.axes[0]] ?? 0;
    const columnStart = starts[view.axes[1]] ?? 0;
    const rowEnd = Math.min(view.logicalShape[0], rowStart + plan.rowSpan);
    const columnEnd = Math.min(view.logicalShape[1], columnStart + plan.columnSpan);
    const tileIndex = frame.tileIdx ?? (coords.mTile * state.context.nTileNum + coords.nTile);
    const focus = view.id === 'tensor:c'
      ? 'OT' + tileIndex + ' · M' + coords.mTile + '/N' + coords.nTile
      : view.id === 'tensor:a'
        ? 'M' + coords.mTile + ' · K-L1 ' + coords.iter0
        : 'K-L1 ' + coords.iter0 + ' · N' + coords.nTile;
    const range = view.axes[0] + '[' + rowStart + ':' + rowEnd + ') · ' + view.axes[1] + '[' + columnStart + ':' + columnEnd + ')';
    return '<div class="matmul-overview-overlay matmul-overview-overlay--tiling" aria-label="' + escapeHtml(view.label + ' tile ' + plan.rowSpan + '×' + plan.columnSpan + ' · ' + range) + '">' +
      '<span class="matmul-overview-overlay__kicker">KERNEL TILE</span>' +
      '<strong>' + escapeHtml(plan.rowSpan + ' × ' + plan.columnSpan) + '</strong>' +
      '<span>' + escapeHtml(focus) + '</span>' +
      '<code>' + escapeHtml(range) + '</code>' +
    '</div>';
  }

  function matmulHostLaunchOverlay(view, hostLaunchScale) {
    const plan = hostLaunchScale?.tensors.find((item) => item.id === view.id);
    if (!plan || !state.context) return '';
    const buffer = {
      'tensor:a': 'deviceInput',
      'tensor:b': 'deviceWeight',
      'tensor:c': 'deviceOutput'
    }[view.id] || view.label;
    const role = view.id === 'tensor:c' ? 'output' : view.id === 'tensor:b' ? 'weight' : 'input';
    return '<div class="matmul-overview-overlay matmul-overview-overlay--host" aria-label="' + escapeHtml(view.label + ' host launch buffer ' + buffer) + '">' +
      '<span class="matmul-overview-overlay__kicker">HOST LAUNCH</span>' +
      '<strong>' + escapeHtml(role + ' · ' + plan.rowSpan + ' × ' + plan.columnSpan + ' cell scale') + '</strong>' +
      '<span>' + escapeHtml(buffer + ' · ' + MATMUL_CORE_COUNT + ' AIC blocks') + '</span>' +
      '<code>' + escapeHtml(view.logicalShape[0] + ' × ' + view.logicalShape[1] + ' · ' + formatBytes(view.sizeBytes) + ' · ' + view.dtype) + '</code>' +
    '</div>';
  }

  function matmulOverviewOverlay(view, mode, scale) {
    if (mode === 'kernel-tiling') return matmulTilingOverlay(view, scale);
    if (mode === 'host-launch') return matmulHostLaunchOverlay(view, scale);
    return '';
  }

  function matmulOverviewTitleScene(view, aggregation = null, tilingScale = null) {
    const scene = tensorTitleScene(view);
    const aggregateShape = Number.isFinite(aggregation?.rowSpan) && Number.isFinite(aggregation?.columnSpan)
      ? { label: 'aggregate', dims: [aggregation.rowSpan, aggregation.columnSpan] }
      : scene.physicalShape;
    return {
      ...scene,
      physicalShape: aggregateShape,
      step: null,
      constraints: [],
      partition: tilingScale ? matmulTilingPartition(view, tilingScale) : null,
      provenance: null,
      status: ''
    };
  }

  function removeMatmulOverviewTitleFooter(mount) {
    mount.querySelector('.pto-tensor-title__footer')?.remove();
  }

  function updateMatmulOverviewTitle(tensorId, view, aggregation = null, tilingScale = null) {
    const key = overviewDomKey(tensorId);
    const mount = document.getElementById('matmulOverviewTitle-' + key);
    if (!mount || !window.PtoTensorTitle) return;
    const scene = matmulOverviewTitleScene(view, aggregation, tilingScale);
    const options = { density: 'full', showShapes: true, showChips: true, showStatus: false };
    const controller = state.overviewTitleControllers[tensorId];
    if (controller) controller.update(scene, options);
    else state.overviewTitleControllers[tensorId] = window.PtoTensorTitle.render(mount, scene, options);
    removeMatmulOverviewTitleFooter(mount);
  }

  function updateMatmulOverviewMatrix(tensorId, view, aggregation) {
    const key = overviewDomKey(tensorId);
    const canvas = document.getElementById('matmulOverviewCanvas-' + key);
    if (!canvas || !window.PtoMatrixCanvas) return;
    const scene = matrixSceneForTensor(view, aggregation);
    const options = {
      ariaLabel: view.label + ' overview matrix',
      showGrid: false,
      interactive: false,
      autoFit: true,
      minZoom: 0.001,
      padding: { top: 18, right: 24, bottom: 46, left: 56 }
    };
    const controller = state.overviewMatrixControllers[tensorId];
    if (controller) {
      controller.update(scene, options);
      controller.resize?.();
      controller.fit?.();
    } else {
      state.overviewMatrixControllers[tensorId] = window.PtoMatrixCanvas.render(canvas, scene, options);
      state.overviewMatrixControllers[tensorId].resize?.();
      state.overviewMatrixControllers[tensorId].fit?.();
    }
  }

  function matmulCopyTabsForStep(stepId) {
    const tabsByStep = {
      'gm-to-l1': [
        { id: 'tensor:a', label: 'A', location: 'A / GM → A1 / L1' },
        { id: 'tensor:b', label: 'B', location: 'B / GM → B1 / L1' }
      ],
      'l1-to-l0': [
        { id: 'tensor:a1', label: 'A1', location: 'A1 / L1 → A2 / L0A' },
        { id: 'tensor:b1', label: 'B1', location: 'B1 / L1 → B2 / L0B' }
      ]
    };
    return tabsByStep[stepId] || [];
  }

  function matmulCopyPeerTensorId(tensorId) {
    return {
      'tensor:a': 'tensor:a1',
      'tensor:b': 'tensor:b1',
      'tensor:a1': 'tensor:a2',
      'tensor:b1': 'tensor:b2'
    }[tensorId] || tensorId;
  }

  function matmulCopyMetadata(stepId) {
    if (stepId === 'gm-to-l1') {
      return {
        engine: 'MTE2 / DataCopy',
        transformation: 'ND → NZ',
        sourceTier: 'GM',
        destinationTier: 'L1',
        readiness: ['Copied by MTE2', 'Awaiting MTE2_MTE1', 'MTE1 blocked']
      };
    }
    return {
      engine: 'MTE1 / CopyL12L0',
      transformation: 'NZ → NZ',
      sourceTier: 'L1',
      destinationTier: null,
      readiness: ['L1 ready', 'Copied by MTE1', 'Awaiting MTE1_M']
    };
  }

  function renderMatmulCopyTabs() {
    const mount = $('#matmulCopyTabs');
    if (!mount) return [];
    const stepId = currentStep().id;
    const tabs = matmulCopyTabsForStep(stepId);
    const selectedTab = tabs.find((tab) => tab.id === state.copyTensorId);
    if (state.copyTensorStepId !== stepId || !selectedTab) {
      state.copyTensorStepId = stepId;
      state.copyTensorId = tabs[0]?.id || null;
    }
    mount.hidden = tabs.length < 2;
    mount.innerHTML = tabs.map((tab) => {
      const selected = tab.id === state.copyTensorId;
      return '<button class="tab-control-item' + (selected ? ' is-selected' : '') + '" type="button" role="tab" data-matmul-copy-tensor="' + escapeHtml(tab.id) + '" aria-selected="' + selected + '" title="' + escapeHtml(tab.location) + '">' + escapeHtml(tab.label) + '</button>';
    }).join('');
    mount.querySelectorAll('[data-matmul-copy-tensor]').forEach((button) => {
      button.addEventListener('click', () => {
        if (state.copyTensorId === button.dataset.matmulCopyTensor) return;
        state.copyTensorId = button.dataset.matmulCopyTensor;
        renderMatmulCopyTabs();
        renderMatmulCopyView();
      });
    });
    return tabs;
  }

  function matmulCopyTitleScene(view, role, tier, stateLabel, offset) {
    const scene = tensorTitleScene(view);
    return {
      ...scene,
      role,
      memory: { tier, sizeBytes: view.sizeBytes, offset },
      state: stateLabel,
      direction: '',
      provenance: null,
      step: null,
      constraints: [],
      status: ''
    };
  }

  function renderMatmulCopyTitle(slot, view, role, tier, stateLabel, offset) {
    const mount = $('#matmulCopy' + (slot === 'source' ? 'Source' : 'Destination') + 'TitleMount');
    if (!mount || !window.PtoTensorTitle) return;
    const scene = matmulCopyTitleScene(view, role, tier, stateLabel, offset);
    const options = { density: 'full', showShapes: true, showChips: true, showStatus: false };
    const controller = state.copyTitleControllers[slot];
    if (controller) controller.update(scene, options);
    else state.copyTitleControllers[slot] = window.PtoTensorTitle.render(mount, scene, options);
    removeMatmulOverviewTitleFooter(mount);
  }

  function renderMatmulCopyMatrix(slot, view, aggregateGrid) {
    const canvas = $('#matmulCopy' + (slot === 'source' ? 'Source' : 'Destination') + 'Canvas');
    if (!canvas || !window.PtoMatrixCanvas) return;
    const scene = matrixSceneForTensor(view, aggregateGrid);
    const options = {
      ariaLabel: view.label + ' copy matrix',
      showAxes: true,
      showGrid: true,
      interactive: false,
      showTooltip: false,
      autoFit: true,
      minZoom: 0.001,
      padding: { top: 30, right: 24, bottom: 38, left: 48 }
    };
    const controller = state.copyMatrixControllers[slot];
    if (controller) {
      controller.update(scene, { ...options, preserveView: false });
      controller.resize?.();
      controller.fit?.();
    } else {
      state.copyMatrixControllers[slot] = window.PtoMatrixCanvas.render(canvas, scene, options);
    }
  }

  function resolveMatmulCopyScale(stepId, views) {
    const frame = currentFrame();
    const curM = frame.curM ?? state.context.curM;
    const curN = frame.curN ?? state.context.curN;
    const curKL1 = frame.curKL1 ?? Math.min(state.context.kL1, state.context.K);
    const curKL0 = frame.curKL0 ?? Math.min(state.context.baseK, curKL1);
    const isGmToL1 = stepId === 'gm-to-l1';
    const axisScales = isGmToL1
      ? { M: state.context.baseM, K: curKL1, N: state.context.baseN }
      : { M: curM, K: curKL0, N: curN };
    return window.PtoMatrixCanvas.resolveSharedAggregateScale({
      tensors: views.map((view) => ({
        id: view.id,
        extent: { rows: view.logicalShape[0], columns: view.logicalShape[1] },
        axes: { rows: view.axes[0], columns: view.axes[1] }
      })),
      axisScales,
      hardBoundaries: {
        M: [axisScales.M],
        K: [axisScales.K],
        N: [axisScales.N]
      }
    });
  }

  function renderMatmulCopyView() {
    const stepId = currentStep().id;
    const tabs = matmulCopyTabsForStep(stepId);
    const sourceId = tabs.some((tab) => tab.id === state.copyTensorId) ? state.copyTensorId : tabs[0]?.id;
    const destinationId = matmulCopyPeerTensorId(sourceId);
    const source = tensorView(sourceId);
    const destination = tensorView(destinationId);
    const metadata = matmulCopyMetadata(stepId);
    if (!source || !destination) return;
    const frame = currentFrame();
    const coords = tileCoordinates();
    const kLabel = stepId === 'gm-to-l1'
      ? 'K-L1=' + (frame.iter0 ?? 0) + '/' + (state.context.kL1TileNum - 1)
      : 'K-L0=' + (frame.iter1 ?? 0) + '/' + ((frame.l0Count || state.context.kL0TileNum) - 1);
    const sourceOffset = source.id === 'tensor:a'
      ? ((coords.mTile * state.context.baseM) * state.context.K + (frame.iter0 ?? 0) * (frame.curKL1 ?? state.context.kL1)) * 2
      : source.id === 'tensor:b'
        ? ((frame.iter0 ?? 0) * (frame.curKL1 ?? state.context.kL1) * state.context.N + coords.nTile * state.context.baseN) * 2
        : 0;
    const destinationOffset = destination.id === 'tensor:b1' ? tensorView('tensor:a1').sizeBytes : 0;
    const sourceRole = source.id === 'tensor:b' ? 'weight' : source.id === 'tensor:b1' ? 'scratch' : 'input';
    const destinationRole = destination.id === 'tensor:b2' ? 'weight' : 'scratch';
    const sourceState = stepId === 'gm-to-l1' ? 'current' : 'ready';
    const destinationState = stepId === 'gm-to-l1' ? 'written' : 'loading';
    $('#matmulCopySummary').textContent = metadata.engine.split(' / ')[0] + ' · ' + source.label + ' transfer · ' + formatBytes(source.sizeBytes) + ' · ' + metadata.sourceTier + ' → ' + (metadata.destinationTier || destination.memoryTier);
    $('#matmulCopyContext').textContent = 'AIC' + (frame.aicIndex ?? 0) + ' · OT' + (frame.tileIdx ?? 0) + ' · M' + coords.mTile + '/N' + coords.nTile + ' · ' + kLabel;
    $('#matmulCopyEngine').textContent = metadata.engine;
    $('#matmulCopyTransformation').textContent = metadata.transformation;
    const evidence = currentStep().evidence || 'derived';
    $('#matmulCopyEvidence').textContent = evidence;
    $('#matmulCopyEvidence').className = 'tag ' + (evidence === 'confirmed' ? 'status-success' : evidence === 'derived' ? 'status-warning' : 'tag-accent');
    $('#matmulCopyReadiness').innerHTML = metadata.readiness.map((item, index) => (index ? '<span aria-hidden="true">→</span>' : '') + '<span class="tag' + (index === 0 ? ' status-success' : '') + '">' + escapeHtml(item) + '</span>').join('');
    renderMatmulCopyTitle('source', source, sourceRole, metadata.sourceTier, sourceState, sourceOffset);
    renderMatmulCopyTitle('destination', destination, destinationRole, metadata.destinationTier || destination.memoryTier, destinationState, destinationOffset);
    const copyScale = resolveMatmulCopyScale(stepId, [source, destination]);
    const sourceScale = copyScale.tensors.find((plan) => plan.id === source.id);
    const destinationScale = copyScale.tensors.find((plan) => plan.id === destination.id);
    renderMatmulCopyMatrix('source', source, sourceScale);
    renderMatmulCopyMatrix('destination', destination, destinationScale);
    const fitMatrices = () => Object.values(state.copyMatrixControllers).forEach((controller) => {
      controller?.resize?.();
      controller?.fit?.();
    });
    if (window.requestAnimationFrame) window.requestAnimationFrame(() => window.requestAnimationFrame(fitMatrices));
    else fitMatrices();
  }

  function activeTensorIdsForStep(stepId) {
    const activeByStep = {
      'host-args': ['tensor:a', 'tensor:b', 'tensor:c'],
      'host-launch': ['tensor:a', 'tensor:b', 'tensor:c'],
      'kernel-tiling': ['tensor:a', 'tensor:b', 'tensor:c'],
      'kernel-block-map': ['tensor:c', 'tensor:co1'],
      'gm-to-l1': ['tensor:a', 'tensor:b', 'tensor:a1', 'tensor:b1'],
      'sync-mte2-mte1': ['tensor:a1', 'tensor:b1'],
      'l1-to-l0': ['tensor:a1', 'tensor:b1', 'tensor:a2', 'tensor:b2'],
      'mmad': ['tensor:a2', 'tensor:b2', 'tensor:co1'],
      'sync-m-fix': ['tensor:co1'],
      'l0c-to-gm': ['tensor:co1', 'tensor:c'],
      'host-verify': ['tensor:c']
    };
    return activeByStep[stepId] || [];
  }

  function renderMatmulOverview() {
    const context = state.context;
    const grid = $('#matmulOverviewGrid');
    if (!context || !grid) return;
    const current = currentInstructionState();
    const activeIds = activeTensorIdsForStep(current.step.id);
    const views = activeIds.map((tensorId) => tensorView(tensorId)).filter(Boolean);
    const inputShapeScale = current.step.id === 'host-args'
      ? resolveMatmulInputShapeScale(views)
      : null;
    const kernelTilingScale = current.step.id === 'kernel-tiling'
      ? resolveMatmulKernelTilingScale(views)
      : null;
    const hostLaunchScale = current.step.id === 'host-launch'
      ? resolveMatmulInputShapeScale(views)
      : null;
    const sharedScale = inputShapeScale || kernelTilingScale || hostLaunchScale;
    const overviewOverlayMode = kernelTilingScale ? 'kernel-tiling' : hostLaunchScale ? 'host-launch' : null;
    const aggregateGrid = sharedScale ? null : sharedAggregateGridForViews(views);
    Object.values(state.overviewTitleControllers).forEach((controller) => controller?.destroy?.());
    Object.values(state.overviewMatrixControllers).forEach((controller) => controller?.destroy?.());
    state.overviewTitleControllers = {};
    state.overviewMatrixControllers = {};
    $('#matmulTensorOverview').classList.toggle('is-kernel-tiling', !!kernelTilingScale);
    $('#matmulTensorOverview').classList.toggle('is-host-execution', !!hostLaunchScale);
    $('#matmulOverviewTitle').textContent = kernelTilingScale
      ? 'Kernel tiling matrix'
      : hostLaunchScale
        ? 'Host execution matrix'
        : 'Logical MatMul contract';
    $('#matmulOverviewEquation').textContent = kernelTilingScale
      ? 'M/N output tile × K-L1 slice · shared semantic-axis scale'
      : hostLaunchScale
        ? 'GM buffers · BF16 · ' + MATMUL_CORE_COUNT + ' AIC launch blocks'
        : 'A[' + context.M + ',' + context.K + '] × B[' + context.K + ',' + context.N + '] → C[' + context.M + ',' + context.N + ']';
    grid.dataset.tensorCount = String(views.length);
    grid.innerHTML = views.map((view) => {
      const key = overviewDomKey(view.id);
      return '<figure class="avz-tensor-overview__item avz-tensor-overview__item--matrix' + (view.role === 'weight' ? ' avz-tensor-overview__item--weight' : '') + (view.role === 'output' ? ' avz-tensor-overview__item--output' : '') + '" data-overview-tensor="' + escapeHtml(view.id) + '">' +
        '<figcaption><div class="avz-tensor-title-host" id="matmulOverviewTitle-' + key + '"></div></figcaption>' +
        '<div class="pto-matrix-canvas-host avz-tensor-overview__canvas matmul-overview-canvas"><canvas class="pto-matrix-canvas" id="matmulOverviewCanvas-' + key + '" aria-label="' + escapeHtml(view.label) + ' overview matrix"></canvas>' + matmulOverviewOverlay(view, overviewOverlayMode, kernelTilingScale || hostLaunchScale) + '</div>' +
      '</figure>';
    }).join('');
    const evidence = current.step.evidence;
    $('#matmulOverviewEvidence').textContent = evidence;
    $('#matmulOverviewEvidence').className = 'tag ' + (evidence === 'confirmed' ? 'status-success' : evidence === 'derived' ? 'status-warning' : 'tag-accent');
    $('#matmulOverviewM').textContent = 'M=' + context.M;
    $('#matmulOverviewK').textContent = 'K=' + context.K;
    $('#matmulOverviewN').textContent = 'N=' + context.N;
    const curM = current.frame.curM ?? context.curM;
    const curN = current.frame.curN ?? context.curN;
    const curKL0 = current.frame.curKL0 ?? context.baseK;
    const inputShapeGridText = inputShapeScale?.tensors
      .map((plan) => tensorView(plan.id).label + ' ' + plan.grid.rows + '×' + plan.grid.columns)
      .join(' · ');
    const kernelTilingGridText = kernelTilingScale?.tensors
      .map((plan) => tensorView(plan.id).label + ' ' + plan.grid.rows + '×' + plan.grid.columns)
      .join(' · ');
    const hostLaunchGridText = hostLaunchScale?.tensors
      .map((plan) => tensorView(plan.id).label + ' ' + plan.grid.rows + '×' + plan.grid.columns)
      .join(' · ');
    $('#matmulOverviewTile').textContent = inputShapeScale
      ? 'Aggregate Cell [' + inputShapeScale.axisScales.M + ',' + inputShapeScale.axisScales.N + '] · ' + inputShapeGridText + ' · BF16'
      : kernelTilingScale
        ? 'Tile M/N/K=' + context.baseM + '/' + context.baseN + '/' + context.baseK + ' · K-L1=' + context.kL1 + ' · ' + kernelTilingGridText + ' · BF16'
      : hostLaunchScale
        ? 'GM buffer cell scale [' + hostLaunchScale.axisScales.M + ',' + hostLaunchScale.axisScales.K + ',' + hostLaunchScale.axisScales.N + '] · ' + hostLaunchGridText + ' · ' + MATMUL_CORE_COUNT + ' AIC · BF16'
      : 'Output Tile [' + curM + ',' + curN + '] · K Slice ' + curKL0 + ' · BF16';
    views.forEach((view) => {
      const aggregation = sharedScale?.tensors.find((plan) => plan.id === view.id) || aggregateGrid;
      updateMatmulOverviewTitle(view.id, view, aggregation, kernelTilingScale);
      updateMatmulOverviewMatrix(view.id, view, aggregation);
    });
    const fitMatrices = () => Object.values(state.overviewMatrixControllers).forEach((controller) => {
      controller?.resize?.();
      controller?.fit?.();
    });
    const fitAndLinkMatrices = () => {
      fitMatrices();
      if (sharedScale) {
        window.PtoMatrixCanvas.synchronizeScale(Object.values(state.overviewMatrixControllers));
      }
    };
    if (window.requestAnimationFrame) {
      window.requestAnimationFrame(() => window.requestAnimationFrame(fitAndLinkMatrices));
    } else {
      fitAndLinkMatrices();
    }
  }

  function renderTensorTabs() {
    $('#tensorTabs').innerHTML = state.trace.tensorViews.map((view) => {
      const selected = view.id === state.selectedTensorId;
      const related = currentStep().memory.includes(view.bufferId);
      return '<button class="tab-control-item' + (selected ? ' is-selected' : '') + (related && !selected ? ' is-related' : '') + '" type="button" role="tab" data-tensor-id="' + escapeHtml(view.id) + '" aria-selected="' + selected + '">' + escapeHtml(view.label) + '</button>';
    }).join('');
    $('#tensorTabs').querySelectorAll('[data-tensor-id]').forEach((button) => {
      button.addEventListener('click', () => {
        state.selectedTensorId = button.dataset.tensorId;
        renderTensorTabs();
        renderTensorDetail();
      });
    });
  }

  function renderTensorLifecycle(view) {
    const stepOrder = new Map(state.trace.steps.map((step, index) => [step.id, index]));
    const currentOrder = stepOrder.get(currentStep().id);
    $('#tensorLifecycle').innerHTML = view.lifecycle.map((event, index) => {
      const order = stepOrder.get(event.stepId);
      const current = event.stepId === currentStep().id;
      const completed = order < currentOrder;
      let label = event.label;
      if (view.id === 'tensor:co1' && event.stepId === 'mmad' && current) label = currentFrame().mmadMode;
      return (index ? '<span class="matmul-lifecycle-arrow" aria-hidden="true">→</span>' : '') +
        '<span class="tag matmul-lifecycle-state' + (current ? ' is-current' : '') + (completed ? ' is-complete' : '') + '">' + escapeHtml(label) + '</span>';
    }).join('');
  }

  function renderTensorDetail() {
    const view = tensorView(state.selectedTensorId);
    const titleScene = tensorTitleScene(view);
    const matrixScene = matrixSceneForTensor(view);
    if (!state.tensorTitleController) {
      state.tensorTitleController = window.PtoTensorTitle?.render($('#tensorTitleMount'), titleScene, { density: 'full' });
    } else {
      state.tensorTitleController.update(titleScene, { density: 'full' });
    }
    if (!state.matrixController) {
      state.matrixController = window.PtoMatrixCanvas?.render($('#tensorMatrixCanvas'), matrixScene, {
        ariaLabel: view.label + ' logical matrix view',
        showGrid: false
      });
    } else {
      state.matrixController.update(matrixScene, { ariaLabel: view.label + ' logical matrix view' });
    }
    renderTensorLifecycle(view);
  }

  function renderMemoryFlowMap() {
    $('#memoryFlowMap').innerHTML = resolvedDataFlows().map((flow) => {
      const current = flow.stepId === currentStep().id;
      return '<button class="matmul-memory-flow' + (current ? ' is-current' : '') + '" type="button" data-flow-target="' + escapeHtml(flow.to) + '">' +
        '<span class="matmul-memory-flow__route"><strong>' + escapeHtml(tensorLabel(flow.from)) + '</strong><b>→</b><strong>' + escapeHtml(tensorLabel(flow.to)) + '</strong></span>' +
        '<span class="matmul-memory-flow__meta">' + escapeHtml(flow.transferEngine) + ' · ' + escapeHtml(flow.transformation) + ' · ' + escapeHtml(formatBytes(flow.bytes)) + '</span>' +
        '<span class="badge ' + evidenceClass(flow.confidence) + '">' + escapeHtml(flow.confidence) + '</span></button>';
    }).join('');
    $('#memoryFlowMap').querySelectorAll('[data-flow-target]').forEach((button) => {
      button.addEventListener('click', () => {
        state.selectedTensorId = button.dataset.flowTarget;
        selectDetailTab('tensor');
      });
    });
  }

  function currentInstructionState() {
    const step = currentStep();
    const frame = currentFrame();
    const tensorId = state.trace.tensorFocusByStep[step.id] || state.selectedTensorId;
    const view = tensorView(tensorId);
    const flow = resolvedDataFlows().find((item) => item.stepId === step.id);
    return {
      step,
      frame,
      frameLabel: frameLabel(frame),
      tensorId,
      tensor: view,
      tensorState: tensorState(view),
      flow
    };
  }

  function selectDetailTab(tab) {
    if (tab === 'tensor') state.detailTab = tab;
  }

  function syncTensorFocusForStep() {
    state.selectedTensorId = state.trace.tensorFocusByStep[currentStep().id] || state.selectedTensorId;
  }

  function publishExecutionState() {
    const frameRoot = document.querySelector('.matmul-frame');
    const step = currentStep();
    const instruction = currentInstructionState();
    const detail = {
      stepIndex: state.stepIndex,
      stepId: step.id,
      frameIndex: state.frameIndex,
      frame: { ...currentFrame() },
      action: primaryAction(step),
      unit: step.unit,
      panelState: {
        tensorId: instruction.tensorId,
        tensorState: instruction.tensorState,
        tensorFlow: instruction.flow ? {
          from: instruction.flow.from,
          to: instruction.flow.to,
          transformation: instruction.flow.transformation
        } : null,
        hardwareUnits: []
      }
    };
    if (frameRoot) {
      frameRoot.dataset.currentStep = detail.stepId;
      frameRoot.dataset.currentAction = detail.action.toLowerCase();
      frameRoot.dataset.currentUnit = String(detail.unit).toLowerCase();
    }
    window.dispatchEvent(new CustomEvent('matmul:execution-state', { detail }));
  }

  function instructionCardModel(stepIndex, title, flow, key, iteration = null, iterationRange = null, sourceStepIndexes = null, l0Iteration = null) {
    return {
      stepIndex,
      title,
      flow,
      key,
      iteration,
      l0Iteration,
      iterationRange,
      sourceStepIndexes: sourceStepIndexes || [stepIndex],
    };
  }

  function syncInstructionCard(eventType, stepIndex, iteration = null, l0Iteration = null) {
    const event = state.trace?.events?.find((item) => item.eventType === eventType);
    const operationKeys = {
      MTE1_MTE2: 'sync-mte1-mte2',
      MTE2_MTE1: 'sync-mte2-mte1',
      M_MTE1: 'm-mte1',
      MTE1_M: 'mte1-m',
      M_FIX: 'sync-m-fix',
      FIX_M: 'fix-m',
    };
    const producer = event?.producerEngine || eventType.split('_')[0];
    const consumer = event?.consumerEngine || eventType.split('_')[1];
    return instructionCardModel(
      stepIndex,
      eventType + ' Sync',
      producer + ' -> ' + consumer,
      operationKeys[eventType] || 'sync-' + eventType.toLowerCase(),
      iteration,
      null,
      null,
      l0Iteration,
    );
  }

  function instructionL1SetupCards(iteration) {
    return [
      syncInstructionCard('MTE1_MTE2', 4, iteration),
      instructionCardModel(4, 'Copy Inputs', 'GM -> L1', 'gm-to-l1', iteration),
      syncInstructionCard('MTE2_MTE1', 5, iteration),
    ];
  }

  function instructionL0Cards(iteration, l0Iteration) {
    const initialize = iteration === 0 && l0Iteration === 0;
    return [
      syncInstructionCard('M_MTE1', 6, iteration, l0Iteration),
      instructionCardModel(6, 'Load A2 / B2', 'L1 -> L0A / L0B', 'load-a2-b2', iteration, null, null, l0Iteration),
      syncInstructionCard('MTE1_M', 6, iteration, l0Iteration),
      instructionCardModel(7, initialize ? 'Mmad Initialize' : 'Mmad Accumulate', initialize ? 'A2 × B2 -> CO1' : 'A2 × B2 + CO1', initialize ? 'mmad-initialize' : 'mmad-accumulate', iteration, null, null, l0Iteration),
    ];
  }

  function createMatmulInstructionCard(card) {
    const selectedStep = card.sourceStepIndexes.includes(state.stepIndex);
    const selectedIteration = Number.isInteger(card.iteration)
      ? (!Number.isInteger(state.instructionIterationFocus) || card.iteration === state.instructionIterationFocus)
      : !Number.isInteger(state.instructionIterationFocus);
    const selectedL0Iteration = Number.isInteger(card.l0Iteration)
      ? (!Number.isInteger(state.instructionL0IterationFocus) || card.l0Iteration === state.instructionL0IterationFocus)
      : !Number.isInteger(state.instructionL0IterationFocus);
    const selectedOperation = !state.instructionOperationFocus || state.instructionOperationFocus === card.key;
    const selected = selectedStep && selectedIteration && selectedL0Iteration && selectedOperation;
    return '<button class="avz-instruction-card' + (selected ? ' is-selected' : '') + '" type="button" data-step-index="' + card.stepIndex + '" data-instruction-operation="' + escapeHtml(card.key) + '"' +
      (Number.isInteger(card.iteration) ? ' data-instruction-iteration="' + card.iteration + '"' : '') +
      (Number.isInteger(card.l0Iteration) ? ' data-instruction-l0-iteration="' + card.l0Iteration + '"' : '') +
      (card.iterationRange ? ' data-instruction-iteration-start="' + card.iterationRange[0] + '" data-instruction-iteration-end="' + card.iterationRange[1] + '"' : '') +
      (selected ? ' aria-current="step"' : '') + '>' +
      '<span class="avz-instruction-card__title">' + escapeHtml(card.title) + '</span>' +
      '<span class="avz-instruction-card__flow">' + escapeHtml(card.flow) + '</span></button>';
  }

  function createMatmulInstructionRow(label, meta, cards, options = {}) {
    const hideLabel = options.hideLabel;
    const sourceStepIndexes = options.sourceStepIndexes || cards.flatMap((card) => card.sourceStepIndexes);
    const active = sourceStepIndexes.includes(state.stepIndex)
      && (!Number.isInteger(options.iteration) || options.iteration === state.instructionIterationFocus)
      && (!Number.isInteger(options.l0Iteration) || options.l0Iteration === state.instructionL0IterationFocus);
    const labelMarkup = Number.isInteger(options.iteration)
      ? instructionIterationButton(options.l0Iteration == null ? 4 : 6, label, options.iteration, options.l0Iteration)
      : '<span class="tag avz-iteration-tag">' + escapeHtml(label) + '</span>';
    return '<div class="avz-instruction-row' + (hideLabel ? ' is-label-hidden' : '') + (active ? ' is-active' : '') + '" role="group" aria-label="' + escapeHtml(label || 'K iterations') + '">' +
      (hideLabel ? '' : '<div class="avz-instruction-row__label">' + labelMarkup + (meta ? '<span class="avz-instruction-row__meta">' + escapeHtml(meta) + '</span>' : '') + '</div>') +
      '<div class="avz-instruction-row__flow">' + cards.map(createMatmulInstructionCard).join('') + '</div></div>';
  }

  function instructionIterationButton(stepIndex, label, iteration, l0Iteration = null) {
    const activeIteration = Number.isInteger(state.instructionIterationFocus) ? state.instructionIterationFocus : 0;
    const activeL0Iteration = Number.isInteger(state.instructionL0IterationFocus) ? state.instructionL0IterationFocus : 0;
    const active = activeIteration === iteration
      && (l0Iteration == null || activeL0Iteration === l0Iteration);
    return '<button class="avz-instruction-iteration-button' + (active ? ' is-active' : '') + '" type="button" role="tab" data-step-index="' + stepIndex + '" data-instruction-iteration="' + iteration + '"' +
      (l0Iteration == null ? '' : ' data-instruction-l0-iteration="' + l0Iteration + '"') + ' aria-selected="' + active + '">' + escapeHtml(label) + '</button>';
  }

  function createMatmulIterationRail(label, count, stepIndex, l1Iteration = null) {
    const buttons = Array.from({ length: count }, (_, iteration) => {
      const l0Iteration = l1Iteration == null ? null : iteration;
      return instructionIterationButton(stepIndex, 'Iter' + iteration, l1Iteration == null ? iteration : l1Iteration, l0Iteration);
    }).join('');
    return '<div class="matmul-loop-iteration-rail" role="tablist" aria-label="' + escapeHtml(label) + ' iterations">' +
      '<span class="matmul-loop-iteration-rail__label">' + escapeHtml(label) + '</span>' +
      '<div class="matmul-loop-iteration-rail__options">' + buttons + '</div>' +
      '</div>';
  }

  function createMatmulL0Loop(iteration) {
    const slices = state.context?.kSlices?.[iteration]?.l0Slices || [];
    const selectedL0Iteration = slices.length
      ? Math.max(0, Math.min(slices.length - 1, Number.isInteger(state.instructionL0IterationFocus) ? state.instructionL0IterationFocus : 0))
      : 0;
    const selectedSlice = slices[selectedL0Iteration] || 0;
    const selectedRow = createMatmulInstructionRow(
      'K-L0 Iter ' + selectedL0Iteration,
      'K[' + (selectedL0Iteration * (state.context?.baseK || 0)) + ':' + (selectedL0Iteration * (state.context?.baseK || 0) + selectedSlice) + ')',
      instructionL0Cards(iteration, selectedL0Iteration),
      { iteration, l0Iteration: selectedL0Iteration, sourceStepIndexes: [6, 7], hideLabel: true }
    );
    return '<section class="avz-instruction-loop avz-instruction-loop--nested" role="list" aria-label="K-L0 Loop for K-L1 Iter ' + iteration + '">' +
      '<header class="avz-instruction-loop__header">' + createMatmulIterationRail('L0 K loop', slices.length, 6, iteration) + '</header>' +
      '<div class="avz-instruction-loop__rows">' + selectedRow + '</div></section>';
  }

  function createMatmulL1Iteration(iteration) {
    const slice = state.context?.kSlices?.[iteration];
    const setupCards = instructionL1SetupCards(iteration);
    const l1Context = 'K[' + (iteration * (state.context?.kL1 || 0)) + ':' + (iteration * (state.context?.kL1 || 0) + (slice?.curKL1 || 0)) + ') · ' + (iteration === 0 ? 'Initialize' : 'Accumulate');
    return '<section class="matmul-l1-iteration" role="listitem" aria-label="K-L1 Iter ' + iteration + '">' +
      '<div class="matmul-l1-instructions" aria-label="' + escapeHtml(l1Context) + '" title="' + escapeHtml(l1Context) + '">' +
        '<div class="matmul-l1-iteration__setup">' + setupCards.map(createMatmulInstructionCard).join('') + '</div>' +
      '</div>' +
      createMatmulL0Loop(iteration) +
      '</section>';
  }

  function createMatmulLoop() {
    const iterationCount = state.context?.kL1TileNum || 1;
    const lastIteration = Math.max(0, iterationCount - 1);
    const selectedIteration = Math.max(0, Math.min(lastIteration, Number.isInteger(state.instructionIterationFocus) ? state.instructionIterationFocus : 0));
    const active = [4, 5, 6, 7].includes(state.stepIndex);
    return '<section class="avz-instruction-loop avz-instruction-loop--l1' + (active ? ' is-active' : '') + '" role="list" aria-label="K-L1 Loop, iterations 0 through ' + lastIteration + '">' +
      '<header class="avz-instruction-loop__header">' + createMatmulIterationRail('L1 K loop', iterationCount, 4) + '</header>' +
      '<div class="avz-instruction-loop__rows">' + createMatmulL1Iteration(selectedIteration) + '</div></section>';
  }

  function renderMatmulCoreContext() {
    const context = state.context;
    const mount = $('#matmulCoreContext');
    const options = $('#matmulCoreOptions');
    if (!mount || !options || !context) return;
    mount.hidden = false;
    const signature = MATMUL_CORE_COUNT + ':' + context.nTileNum + ':' + context.mTileNum;
    if (options.dataset.signature !== signature) {
      options.dataset.signature = signature;
      options.innerHTML = Array.from({ length: MATMUL_CORE_COUNT }, (_, index) => {
        const assignedTiles = coreTileSchedule(context, index);
        const firstTile = assignedTiles[0] ?? 0;
        const lastTile = assignedTiles.at(-1) ?? firstTile;
        const tileSummary = assignedTiles.join(', ');
        return '<button class="btn btn-compact btn-ghost" type="button" role="radio" data-matmul-core-index="' + index + '" aria-label="AIC ' + index + ', output tiles ' + escapeHtml(tileSummary) + '" title="AIC' + index + ' · output tiles ' + escapeHtml(tileSummary) + '">AIC' + index + ' · OT' + firstTile + '→' + lastTile + '</button>';
      }).join('');
    }
    const selectedTiles = coreTileSchedule(context, state.matmulCoreIndex);
    if (!selectedTiles.includes(state.selectedTileIdx)) state.selectedTileIdx = selectedTiles[0] ?? 0;
    const selectedFirst = selectedTiles[0] ?? 0;
    const selectedLast = selectedTiles.at(-1) ?? selectedFirst;
    const scheduleMeta = $('#matmulCoreScheduleMeta');
    if (scheduleMeta) {
      scheduleMeta.textContent = MATMUL_CORE_COUNT + ' AIC · round-robin · ' + context.outputTileNum + ' output tiles · ' + selectedTiles.length + ' tiles/AIC · selected AIC' + state.matmulCoreIndex + ': OT' + selectedFirst + '→OT' + selectedLast;
    }
    options.querySelectorAll('[data-matmul-core-index]').forEach((button) => {
      const selected = Number(button.dataset.matmulCoreIndex) === state.matmulCoreIndex;
      button.classList.toggle('is-selected', selected);
      button.setAttribute('aria-checked', String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
    renderMatmulTileSequence(selectedTiles);
  }

  function renderMatmulTileSequence(selectedTiles = []) {
    const mount = $('#matmulTileSequence');
    const context = state.context;
    if (!mount || !context) return;
    mount.innerHTML = '<header class="matmul-tile-sequence__header"><strong>AIC' + state.matmulCoreIndex + ' Output Tiles</strong><span>' + selectedTiles.length + ' tiles · execution order</span></header>' +
      '<div class="matmul-tile-sequence__list" role="list">' + selectedTiles.map((tileIdx) => {
        const tile = outputTileLabel(context, tileIdx);
        const selected = tileIdx === state.selectedTileIdx;
        return '<button class="matmul-tile-item' + (selected ? ' is-selected' : '') + '" type="button" role="listitem" data-matmul-tile-index="' + tileIdx + '" aria-pressed="' + selected + '">' +
          '<span class="matmul-tile-item__index">' + escapeHtml(tile.label) + '</span><span class="matmul-tile-item__coord">' + escapeHtml(tile.coordinate) + '</span>' +
          '<span class="matmul-tile-item__arrow" aria-hidden="true">→</span></button>';
      }).join('') + '</div>' +
      '<footer class="matmul-tile-sequence__footer">Selected: <code>Tile ' + String(state.selectedTileIdx).padStart(2, '0') + '</code> · C[M' + Math.floor(state.selectedTileIdx / context.nTileNum) + ',N' + (state.selectedTileIdx % context.nTileNum) + ']</footer>';
  }

  function renderFlow() {
    const steps = state.trace.steps;
    const before = [
      instructionCardModel(3, 'Resolve Output Tile', 'C[M' + (currentFrame().mTileIdx ?? 0) + ',N' + (currentFrame().nTileIdx ?? 0) + '] · ' + currentFrame().curM + '×' + currentFrame().curN, 'kernel-block-map', null, null, [3]),
    ];
    const after = [
      syncInstructionCard('M_FIX', 8),
      instructionCardModel(9, 'Copy L0C → GM', 'L0C → GM', steps[9].id),
      syncInstructionCard('FIX_M', 9),
    ];
    const track = '<div class="avz-instruction-track">' +
      before.map(createMatmulInstructionCard).join('') +
      createMatmulLoop() +
      after.map(createMatmulInstructionCard).join('') +
      '</div>';
    $('#instructionSequence').innerHTML = track;
  }

  function renderTensorJourney() {
    const mount = $('#tensorJourneyContent');
    if (!mount) return;
    const flows = resolvedDataFlows();
    mount.innerHTML =
      '<div class="matmul-tensor-journey">' +
        '<div class="matmul-tensor-journey__summary"><strong>Tensor lifecycle</strong><span>GM → L1 → L0 → GM · logical data movement</span></div>' +
        '<div class="matmul-tensor-journey__list">' +
          flows.map((flow) => {
            const targetStep = state.trace.steps.findIndex((step) => step.id === flow.stepId);
            const active = targetStep === state.stepIndex;
            return '<button class="matmul-memory-flow matmul-journey-flow' + (active ? ' is-current' : '') + '" type="button" data-journey-step="' + targetStep + '" data-journey-tensor="' + escapeHtml(flow.to) + '">' +
              '<span class="matmul-memory-flow__route"><strong>' + escapeHtml(tensorLabel(flow.from)) + '</strong><span aria-hidden="true">→</span><strong>' + escapeHtml(tensorLabel(flow.to)) + '</strong></span>' +
              '<span class="matmul-memory-flow__meta">' + escapeHtml(flow.engine) + ' · ' + escapeHtml(flow.transformation) + ' · ' + escapeHtml(flow.bytes) + ' logical bytes</span>' +
              '<span class="badge ' + evidenceClass(flow.evidence) + '">' + escapeHtml(flow.evidence) + '</span>' +
            '</button>';
          }).join('') +
        '</div>' +
      '</div>';
  }

  function renderExecutionDock() {
    const validViews = ['instructions', 'tensor-journey', 'timeline'];
    if (!validViews.includes(state.executionView)) state.executionView = 'instructions';
    document.querySelectorAll('[data-execution-view]').forEach((button) => {
      const active = button.dataset.executionView === state.executionView;
      button.classList.toggle('is-selected', active);
      button.setAttribute('aria-selected', String(active));
      button.tabIndex = active ? 0 : -1;
    });
    document.querySelectorAll('[data-execution-view-panel]').forEach((panel) => {
      panel.hidden = panel.dataset.executionViewPanel !== state.executionView;
    });
    const kicker = $('#timelineKicker');
    if (kicker) {
      kicker.textContent = state.executionView === 'timeline'
        ? 'Logical order · duration unavailable · Not Profiling Data'
        : state.executionView === 'tensor-journey'
          ? 'Tensor data flow by memory location'
          : currentStep().evidence + ' · logical order · repeated iterations grouped';
    }
    if (state.executionView === 'instructions') renderFlow();
    if (state.executionView === 'tensor-journey') renderTensorJourney();
  }

  const MATMUL_SYNC_OPERATION_TYPES = Object.freeze({
    'sync-mte1-mte2': 'MTE1_MTE2',
    'sync-mte2-mte1': 'MTE2_MTE1',
    'm-mte1': 'M_MTE1',
    'mte1-m': 'MTE1_M',
    'sync-m-fix': 'M_FIX',
    'fix-m': 'FIX_M'
  });

  const MATMUL_SYNC_COPY = Object.freeze({
    MTE1_MTE2: Object.freeze({
      producerMeta: 'L0 reads complete',
      consumerMeta: 'L1 writable after wait',
      producerTone: 'reduction',
      consumerTone: 'input',
      footer: 'Loop-carried dependency · L1 reuse is gated before the next GM→L1 copy'
    }),
    MTE2_MTE1: Object.freeze({
      producerMeta: 'L1 writes complete',
      consumerMeta: 'L1 readable after wait',
      producerTone: 'input',
      consumerTone: 'input',
      footer: 'Synchronization completes readiness; it does not move tensor data'
    }),
    M_MTE1: Object.freeze({
      producerMeta: 'A2/B2 reads complete',
      consumerMeta: 'A2/B2 writable after wait',
      producerTone: 'reduction',
      consumerTone: 'input',
      footer: 'Dependency only · A2/B2 remain in L0A/L0B'
    }),
    MTE1_M: Object.freeze({
      producerMeta: 'A2/B2 loads complete',
      consumerMeta: 'K slice readable after wait',
      producerTone: 'input',
      consumerTone: 'reduction',
      footer: 'Readiness only · no tensor data moves through the event edge'
    }),
    M_FIX: Object.freeze({
      producerMeta: 'Final CO1 accumulation complete',
      consumerMeta: 'CO1 readable after wait',
      producerTone: 'reduction',
      consumerTone: 'output',
      footer: 'Readiness only · CO1 remains in L0C'
    }),
    FIX_M: Object.freeze({
      producerMeta: 'Output tile write complete',
      consumerMeta: 'Next output tile may reuse resources after wait',
      producerTone: 'output',
      consumerTone: 'input',
      footer: 'Loop-carried dependency · output path and L0C reuse are ordered'
    })
  });

  function focusedMatmulSyncEventType() {
    const focusedType = MATMUL_SYNC_OPERATION_TYPES[state.instructionOperationFocus];
    if (focusedType) return focusedType;
    return ({
      'sync-mte2-mte1': 'MTE2_MTE1',
      'sync-m-fix': 'M_FIX'
    })[currentStep().id] || null;
  }

  function focusedMatmulSyncEvent() {
    const eventType = focusedMatmulSyncEventType();
    return eventType ? state.trace?.events?.find((event) => event.eventType === eventType) || null : null;
  }

  function matmulSyncCssColor(variable, fallback) {
    const value = getComputedStyle(document.documentElement).getPropertyValue(variable).trim();
    return value || fallback;
  }

  function matmulSyncTone(name) {
    return ({
      input: { fill: 'rgba(70, 132, 216, 0.82)', stroke: 'rgba(184, 218, 255, 0.92)' },
      reduction: { fill: 'rgba(220, 125, 55, 0.82)', stroke: 'rgba(255, 211, 165, 0.92)' },
      output: { fill: 'rgba(52, 157, 135, 0.82)', stroke: 'rgba(168, 244, 223, 0.92)' },
      fusion: { fill: 'rgba(145, 112, 207, 0.86)', stroke: 'rgba(226, 210, 255, 0.92)' }
    })[name] || { fill: 'rgba(93, 108, 126, 0.82)', stroke: 'rgba(207, 216, 226, 0.92)' };
  }

  function matmulSyncRoundRect(ctx, x, y, width, height, radius) {
    const r = Math.min(radius, width / 2, height / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + width, y, x + width, y + height, r);
    ctx.arcTo(x + width, y + height, x, y + height, r);
    ctx.arcTo(x, y + height, x, y, r);
    ctx.arcTo(x, y, x + width, y, r);
    ctx.closePath();
  }

  function matmulSyncFittedText(ctx, value, x, y, maxWidth) {
    const text = String(value == null ? '' : value);
    if (ctx.measureText(text).width <= maxWidth) {
      ctx.fillText(text, x, y);
      return;
    }
    let fitted = text;
    while (fitted.length > 1 && ctx.measureText(fitted + '…').width > maxWidth) fitted = fitted.slice(0, -1);
    ctx.fillText(fitted + '…', x, y);
  }

  function drawMatmulSyncBox(ctx, x, y, width, height, item) {
    const tone = matmulSyncTone(item.tone);
    ctx.fillStyle = tone.fill;
    ctx.strokeStyle = tone.stroke;
    ctx.lineWidth = 1.2;
    matmulSyncRoundRect(ctx, x, y, width, height, 11);
    ctx.fill();
    ctx.stroke();

    const padding = Math.max(12, Math.min(18, width * 0.06));
    const textWidth = Math.max(20, width - padding * 2);
    const compact = width < 140;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = matmulSyncCssColor('--foreground', '#f4f6f8');
    ctx.font = (compact ? '700 11px' : '700 15px') + ' Inter, system-ui, sans-serif';
    matmulSyncFittedText(ctx, item.engine, x + padding, y + 30, textWidth);
    ctx.fillStyle = 'rgba(244, 246, 248, 0.78)';
    ctx.font = (compact ? '700 10px' : '700 13px') + ' ui-monospace, SFMono-Regular, Menlo, monospace';
    matmulSyncFittedText(ctx, item.shape, x + padding, y + 58, textWidth);
    ctx.fillStyle = 'rgba(232, 239, 248, 0.68)';
    ctx.font = (compact ? '600 9px' : '600 11px') + ' Inter, system-ui, sans-serif';
    matmulSyncFittedText(ctx, item.meta, x + padding, y + height - 18, textWidth);
  }

  function drawMatmulSyncArrow(ctx, x, y, width) {
    const color = matmulSyncCssColor('--foreground-secondary', '#aeb7c1');
    const head = Math.min(10, Math.max(6, width * 0.08));
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + width - head, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x + width - head, y - head * 0.65);
    ctx.lineTo(x + width, y);
    ctx.lineTo(x + width - head, y + head * 0.65);
    ctx.closePath();
    ctx.fill();
  }

  function drawMatmulSyncScene(ctx, width, height, event) {
    const copy = MATMUL_SYNC_COPY[event.eventType] || {};
    const background = matmulSyncCssColor('--surface-2', '#202020');
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, width, height);

    const inset = Math.max(24, Math.min(48, width * 0.045));
    const titleColor = matmulSyncCssColor('--foreground', '#f4f6f8');
    const secondaryColor = matmulSyncCssColor('--foreground-secondary', '#aeb7c1');
    const mutedColor = matmulSyncCssColor('--foreground-muted', '#7c8793');
    ctx.textAlign = 'left';
    ctx.fillStyle = titleColor;
    ctx.font = (width < 640 ? '700 14px' : '700 18px') + ' ui-monospace, SFMono-Regular, Menlo, monospace';
    matmulSyncFittedText(ctx, 'Producer → Event dependency → Consumer', inset, 42, width - inset * 2);
    ctx.fillStyle = secondaryColor;
    ctx.font = '600 13px Inter, system-ui, sans-serif';
    ctx.fillText((event.confidence || 'confirmed') + ' · logical order only', inset, 70);

    const usableWidth = Math.max(280, width - inset * 2);
    const compact = width < 640;
    const gap = compact ? Math.max(10, Math.min(18, usableWidth * 0.035)) : Math.max(18, Math.min(54, usableWidth * 0.055));
    const eventWidth = Math.min(210, Math.max(compact ? 86 : 150, usableWidth * 0.19));
    const boxWidth = Math.min(330, Math.max(compact ? 96 : 190, (usableWidth - eventWidth - gap * 2) / 2));
    const totalWidth = boxWidth * 2 + eventWidth + gap * 2;
    const startX = Math.max(inset, (width - totalWidth) / 2);
    const boxHeight = Math.min(126, Math.max(104, height * 0.2));
    const y = Math.max(118, Math.min(height * 0.42, 154));
    const producerX = startX;
    const eventX = producerX + boxWidth + gap;
    const consumerX = eventX + eventWidth + gap;

    drawMatmulSyncBox(ctx, producerX, y, boxWidth, boxHeight, {
      engine: event.producerEngine,
      shape: 'SetFlag',
      meta: copy.producerMeta || 'producer writes complete',
      tone: copy.producerTone
    });
    drawMatmulSyncBox(ctx, eventX, y, eventWidth, boxHeight, {
      engine: event.eventType,
      shape: 'dependency',
      meta: 'confirmed',
      tone: 'fusion'
    });
    drawMatmulSyncBox(ctx, consumerX, y, boxWidth, boxHeight, {
      engine: event.consumerEngine,
      shape: 'WaitFlag',
      meta: copy.consumerMeta || 'consumer readable after wait',
      tone: copy.consumerTone
    });
    const arrowGap = Math.max(8, gap - 10);
    drawMatmulSyncArrow(ctx, producerX + boxWidth + 5, y + boxHeight / 2, arrowGap);
    drawMatmulSyncArrow(ctx, eventX + eventWidth + 5, y + boxHeight / 2, arrowGap);

    ctx.textAlign = 'left';
    ctx.fillStyle = secondaryColor;
    ctx.font = '600 13px Inter, system-ui, sans-serif';
    matmulSyncFittedText(ctx, event.explanation || 'Execution dependency', inset, height - 48, width - inset * 2);
    ctx.fillStyle = mutedColor;
    ctx.font = '500 12px Inter, system-ui, sans-serif';
    matmulSyncFittedText(ctx, copy.footer || 'Synchronization changes readiness, not tensor ownership', inset, height - 24, width - inset * 2);
  }

  function renderMatmulSyncView() {
    const canvas = $('#matmulSyncCanvas');
    const event = focusedMatmulSyncEvent();
    if (!canvas || !event) return;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, rect.width);
    const height = Math.max(1, rect.height);
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.max(1, Math.floor(width * dpr));
    canvas.height = Math.max(1, Math.floor(height * dpr));
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawMatmulSyncScene(ctx, width, height, event);
    canvas.dataset.eventType = event.eventType;
  }

  function renderCurrent() {
    const step = currentStep();
    const syncEvent = focusedMatmulSyncEvent();
    const copyMeta = {
      'gm-to-l1': 'MTE2 · GM → L1 · logical order · duration unavailable',
      'l1-to-l0': 'MTE1 · L1 → L0 · logical order · duration unavailable'
    }[step.id];
    $('#flowMeta').textContent = syncEvent
      ? (syncEvent.confidence || 'confirmed') + ' · logical order only'
      : copyMeta || (step.id === 'kernel-block-map'
      ? 'LocalTensor views · no data movement'
      : primaryAction(step) + ' · ' + step.unit + ' · logical order · duration unavailable');
  }

  function renderDetails() {
    const stepId = currentStep().id;
    const syncEvent = focusedMatmulSyncEvent();
    const copyActive = ['gm-to-l1', 'l1-to-l0'].includes(stepId);
    const allocationActive = stepId === 'kernel-block-map';
    const syncActive = Boolean(syncEvent);
    $('#matmulCopyTabs').hidden = !copyActive || syncActive;
    $('#matmulCopyView').hidden = !copyActive || syncActive;
    $('#matmulTensorOverview').hidden = allocationActive || copyActive || syncActive;
    $('#matmulMemoryAllocation').hidden = !allocationActive || syncActive;
    $('#matmulSyncView').hidden = !syncActive;
    if (syncActive) {
      renderMatmulSyncView();
    } else if (copyActive) {
      renderMatmulCopyTabs();
      renderMatmulCopyView();
    } else if (allocationActive) {
      renderMatmulMemoryAllocation();
    } else {
      renderMatmulCopyTabs();
      renderMatmulOverview();
    }
  }

  function renderRoleTabs() {
    document.querySelectorAll('[data-source-role]').forEach((button) => {
      const active = button.dataset.sourceRole === state.sourceRole;
      button.classList.toggle('is-selected', active);
      button.setAttribute('aria-selected', String(active));
    });
  }

  function stopPlaybackTimer() {
    if (state.timer != null) window.clearInterval(state.timer);
    state.timer = null;
  }

  function syncPlayback() {
    const helper = window.PtoFloatingPlaybackControl;
    const scrubber = document.getElementById(PLAYBACK_IDS.scrubber);
    const label = document.getElementById(PLAYBACK_IDS.scrubberLabel);
    const opname = document.getElementById(PLAYBACK_IDS.scrubberOpname);
    const play = document.getElementById(PLAYBACK_IDS.play);
    if (scrubber) {
      scrubber.max = String(Math.max(0, state.frames.length - 1));
      scrubber.value = String(state.frameIndex);
    }
    if (label) label.textContent = (state.frameIndex + 1) + ' / ' + state.frames.length;
    if (opname) opname.textContent = frameLabel();
    if (play && helper?.iconLabel) {
      play.innerHTML = state.playing ? helper.iconLabel('pause', 'Pause') : helper.iconLabel('play', 'Play');
    }
    state.playback?.sync({ playing: state.playing });
  }

  function selectFrame(index, options = {}) {
    state.frameIndex = Math.max(0, Math.min(state.frames.length - 1, index));
    const frame = currentFrame();
    const stepIndex = state.trace.steps.findIndex((step) => step.id === frame.stepId);
    if (stepIndex >= 0) selectStep(stepIndex, { keepFrame: true, fromPlayback: options.fromPlayback });
  }

  function setPlaying(playing) {
    stopPlaybackTimer();
    state.playing = !!playing;
    if (state.playing && state.frameIndex >= state.frames.length - 1) {
      selectFrame(0, { fromPlayback: true });
    }
    if (state.playing) {
      state.timer = window.setInterval(() => {
        if (state.frameIndex >= state.frames.length - 1) {
          setPlaying(false);
          return;
        }
        selectFrame(state.frameIndex + 1, { fromPlayback: true });
      }, 900);
    }
    syncPlayback();
  }

  function initPlayback() {
    const helper = window.PtoFloatingPlaybackControl;
    const mount = $('#playbackMount');
    if (!helper?.createControl || !mount) return;
    mount.innerHTML = '';
    const control = helper.createControl({
      ids: PLAYBACK_IDS,
      className: 'pto-floating-playback--preview'
    });
    mount.appendChild(control);
    const timelineLabel = control.querySelector('.pto-floating-playback__label');
    if (timelineLabel) timelineLabel.textContent = 'Logical order';
    state.playback = helper.init({
      root: control,
      isPlaying: () => state.playing
    });
    state.playbackHover = helper.initScrubberHover({
      root: control,
      getTotalSteps: () => state.frames.length,
      getLabelForStep: (index) => frameLabel(state.frames[index])
    });

    document.getElementById(PLAYBACK_IDS.play)?.addEventListener('click', () => setPlaying(!state.playing));
    document.getElementById(PLAYBACK_IDS.stepBack)?.addEventListener('click', () => {
      setPlaying(false);
      selectFrame(state.frameIndex - 1);
    });
    document.getElementById(PLAYBACK_IDS.stepForward)?.addEventListener('click', () => {
      setPlaying(false);
      selectFrame(state.frameIndex + 1);
    });
    document.getElementById(PLAYBACK_IDS.replay)?.addEventListener('click', () => {
      setPlaying(false);
      selectFrame(0);
    });
    document.getElementById(PLAYBACK_IDS.scrubber)?.addEventListener('input', (event) => {
      setPlaying(false);
      selectFrame(Number(event.target.value) || 0);
    });
    syncPlayback();
  }

  function selectSourceLine(line) {
    const indexes = linkedStepIndexes(line);
    if (!indexes.length) return;
    const targetIndex = indexes.includes(state.stepIndex) ? state.stepIndex : indexes[0];
    selectStep(targetIndex);
  }

  function frameMatchesInstruction(frame, iteration, l0Iteration, operation) {
    if (Number.isInteger(iteration) && frame.iter0 !== iteration) return false;
    if (Number.isInteger(l0Iteration) && frame.iter1 !== l0Iteration) return false;
    if (operation === 'mmad-initialize') return frame.mmadMode === 'initialize CO1';
    if (operation === 'mmad-accumulate') return frame.mmadMode === 'accumulate CO1';
    return true;
  }

  function selectStep(index, options = {}) {
    if (!options.fromPlayback) {
      state.playing = false;
      stopPlaybackTimer();
    }
    state.stepIndex = Math.max(0, Math.min(state.trace.steps.length - 1, index));
    state.instructionIterationFocus = options.instructionIteration ?? null;
    state.instructionL0IterationFocus = options.instructionL0Iteration ?? null;
    state.instructionOperationFocus = options.instructionOperation ?? null;
    if (!options.keepFrame) {
      const nextFrameIndex = state.frames.findIndex((frame) => frame.stepId === currentStep().id
        && frameMatchesInstruction(frame, options.instructionIteration, options.instructionL0Iteration, options.instructionOperation));
      if (nextFrameIndex >= 0) state.frameIndex = nextFrameIndex;
    }
    syncTensorFocusForStep();
    const role = currentStep().role.toLowerCase();
    if (role === 'kernel' || role === 'host') state.sourceRole = role;
    renderMatmulCoreContext();
    renderRoleTabs();
    renderSource();
    renderFlow();
    renderCurrent();
    renderDetails();
    renderExecutionDock();
    syncPlayback();
    publishExecutionState();
    $('#statusText').textContent = frameLabel() + ' · source-linked · logical playback';
    $('#statusStep').textContent = (state.stepIndex + 1) + ' / ' + state.trace.steps.length;
  }

  function selectMatmulTile(tileIdx) {
    state.selectedTileIdx = Number(tileIdx);
    const tileStepIndex = state.trace.steps.findIndex((step) => step.id === 'kernel-block-map');
    selectStep(tileStepIndex >= 0 ? tileStepIndex : state.stepIndex);
  }

  function bindControls() {
    $('#matmulCoreOptions').addEventListener('click', (event) => {
      const button = event.target.closest('[data-matmul-core-index]');
      if (!button) return;
      state.matmulCoreIndex = Math.max(0, Math.min(MATMUL_CORE_COUNT - 1, Number(button.dataset.matmulCoreIndex) || 0));
      selectMatmulTile(coreTileSchedule(state.context, state.matmulCoreIndex)[0] ?? 0);
    });
    $('#matmulCoreOptions').addEventListener('keydown', (event) => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      const buttons = Array.from($('#matmulCoreOptions').querySelectorAll('[data-matmul-core-index]'));
      if (!buttons.length) return;
      event.preventDefault();
      const currentIndex = Math.max(0, buttons.findIndex((button) => Number(button.dataset.matmulCoreIndex) === state.matmulCoreIndex));
      const nextIndex = event.key === 'Home'
        ? 0
        : event.key === 'End'
          ? buttons.length - 1
          : (currentIndex + (event.key === 'ArrowRight' ? 1 : -1) + buttons.length) % buttons.length;
      state.matmulCoreIndex = nextIndex;
      selectMatmulTile(coreTileSchedule(state.context, state.matmulCoreIndex)[0] ?? 0);
      buttons[nextIndex]?.focus();
    });
    $('#matmulTileSequence').addEventListener('click', (event) => {
      const button = event.target.closest('[data-matmul-tile-index]');
      if (!button) return;
      selectMatmulTile(button.dataset.matmulTileIndex);
    });
    $('#instructionSequence').addEventListener('click', (event) => {
      const iterationButton = event.target.closest('.avz-instruction-iteration-button');
      if (iterationButton) {
        const stepIndex = Number(iterationButton.dataset.stepIndex);
        const iteration = Number(iterationButton.dataset.instructionIteration);
        const l0Iteration = iterationButton.dataset.instructionL0Iteration === undefined
          ? null
          : Number(iterationButton.dataset.instructionL0Iteration);
        if (Number.isInteger(stepIndex) && Number.isInteger(iteration)) {
          selectStep(stepIndex, {
            instructionIteration: iteration,
            instructionL0Iteration: Number.isInteger(l0Iteration) ? l0Iteration : null,
            instructionOperation: null,
          });
        }
        return;
      }
      const card = event.target.closest('[data-step-index]');
      if (!card) return;
      const iterationStart = Number(card.dataset.instructionIterationStart);
      const iterationEnd = Number(card.dataset.instructionIterationEnd);
      const hasRange = Number.isInteger(iterationStart) && Number.isInteger(iterationEnd);
      selectStep(Number(card.dataset.stepIndex), {
        instructionIteration: hasRange ? null : Number.isInteger(Number(card.dataset.instructionIteration)) ? Number(card.dataset.instructionIteration) : null,
        instructionL0Iteration: hasRange ? null : Number.isInteger(Number(card.dataset.instructionL0Iteration)) ? Number(card.dataset.instructionL0Iteration) : null,
        instructionOperation: card.dataset.instructionOperation || null,
      });
    });
    document.addEventListener('click', (event) => {
      const button = event.target.closest?.('[data-journey-step]');
      if (!button) return;
      state.selectedTensorId = button.dataset.journeyTensor || state.selectedTensorId;
      selectStep(Number(button.dataset.journeyStep));
      selectDetailTab('tensor');
    });
    document.querySelectorAll('[data-detail-tab]').forEach((button) => {
      button.addEventListener('click', () => selectDetailTab(button.dataset.detailTab));
    });
    document.querySelectorAll('[data-execution-view]').forEach((button) => {
      button.addEventListener('click', () => {
        state.executionView = button.dataset.executionView;
        renderExecutionDock();
      });
      button.addEventListener('keydown', (event) => {
        if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
        event.preventDefault();
        const buttons = [...document.querySelectorAll('[data-execution-view]')];
        const currentIndex = buttons.indexOf(button);
        const offset = event.key === 'ArrowRight' ? 1 : -1;
        const next = buttons[(currentIndex + offset + buttons.length) % buttons.length];
        state.executionView = next.dataset.executionView;
        renderExecutionDock();
        next.focus();
      });
    });
    document.querySelectorAll('[data-source-role]').forEach((button) => {
      button.addEventListener('click', () => {
        state.sourceRole = button.dataset.sourceRole;
        renderRoleTabs();
        renderSource();
      });
    });
    $('#previousStep').addEventListener('click', () => {
      setPlaying(false);
      selectFrame(state.frameIndex - 1);
    });
    $('#nextStep').addEventListener('click', () => {
      setPlaying(false);
      selectFrame(state.frameIndex + 1);
    });
    document.addEventListener('keydown', (event) => {
      if (event.target.closest('button, a, input, [role="tab"], [role="group"]')) return;
      if (event.key === 'ArrowLeft') {
        setPlaying(false);
        selectFrame(state.frameIndex - 1);
      }
      if (event.key === 'ArrowRight') {
        setPlaying(false);
        selectFrame(state.frameIndex + 1);
      }
    });
  }

  async function load() {
    const responses = await Promise.all([fetch(FIXTURE_URL), fetch(SOURCE_URL)]);
    if (!responses[0].ok || !responses[1].ok) throw new Error('MatMul fixture or source snapshot failed to load');
    state.trace = await responses[0].json();
    const sourceText = await responses[1].text();
    state.source = sourceText.replace(/\r\n?/g, '\n').split('\n').map((text, index) => ({
      line: index + 1,
      text
    }));
    state.scenarioId = validationScenarios()[0]?.id || 'divisible';
    state.context = deriveValidationContext(state.trace, currentScenario());
    state.selectedTileIdx = coreTileSchedule(state.context, state.matmulCoreIndex)[0] ?? 0;
    state.frames = buildExecutionFrames(state.trace, state.context);
    bindControls();
    initPlayback();
    renderExecutionDock();
    const initialStepIndex = state.trace.steps.findIndex((step) => step.id === 'kernel-block-map');
    selectStep(initialStepIndex >= 0 ? initialStepIndex : 0);
    window.PtoIdeFrame?.initAll?.();
  }

  load().catch((error) => {
    $('#statusText').textContent = 'Load failed · ' + error.message;
    $('#sourceLines').innerHTML = '<p class="matmul-load-error">' + escapeHtml(error.message) + '<br>请通过 HTTP 服务打开此页面。</p>';
  });
})();
