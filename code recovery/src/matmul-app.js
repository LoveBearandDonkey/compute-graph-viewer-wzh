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
  const state = {
    trace: null,
    scenarioId: 'divisible',
    context: null,
    source: [],
    sourceRole: 'all',
    executionView: 'instructions',
    stepIndex: 0,
    frames: [],
    frameIndex: 0,
    playing: false,
    timer: null,
    playback: null,
    playbackHover: null,
    loopExpanded: { l1: true, l0: true },
    instructionLoopExpanded: false,
    instructionIterationFocus: null,
    instructionOperationFocus: null,
    detailTab: 'tensor',
    selectedTensorId: 'tensor:a',
    matrixController: null,
    tensorTitleController: null,
    hardwareViewport: null,
    hardwareRouteOverlay: null,
    hardwareInitialized: false
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

  function currentFrame() {
    return state.frames[state.frameIndex] || { stepId: currentStep()?.id };
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
    return state.trace.validationScenarios || [];
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

    push('host-args');
    push('host-launch');
    push('kernel-tiling');
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
    push('host-verify');
    return frames;
  }

  function frameLabel(frame = currentFrame()) {
    const step = state.trace.steps.find((item) => item.id === frame.stepId);
    if (!step) return 'Logical execution';
    if (frame.stepId === 'mmad') {
      return 'Mmad ' + frame.mmadOrdinal + '/' + state.context.mmadPerOutputTile + ' · ' + frame.mmadMode;
    }
    const context = [];
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

  function loopToggleIcon(expanded) {
    return '<svg class="matmul-loop-toggle__icon" viewBox="0 0 24 24" aria-hidden="true"><path d="m' +
      (expanded ? '6 9 6 6 6-6' : '9 6 6 6-6 6') + '"></path></svg>';
  }

  function renderValidationControls() {
    const scenario = currentScenario();
    const context = state.context;
    const scenarios = validationScenarios().filter((item) => item.id !== 'combined-tail');
    $('#validationCases').innerHTML = scenarios.map((item) => {
      const active = item.id === state.scenarioId;
      return '<button class="tab-control-item' + (active ? ' is-selected' : '') + '" type="button" data-validation-scenario="' + escapeHtml(item.id) + '" aria-pressed="' + active + '">' + escapeHtml(item.label) + '</button>';
    }).join('');
    $('#validationCases').querySelectorAll('[data-validation-scenario]').forEach((button) => {
      button.addEventListener('click', () => selectValidationScenario(button.dataset.validationScenario));
      button.addEventListener('keydown', (event) => {
        if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
        event.preventDefault();
        const currentIndex = scenarios.findIndex((item) => item.id === button.dataset.validationScenario);
        const offset = event.key === 'ArrowRight' ? 1 : -1;
        const next = scenarios[(currentIndex + offset + scenarios.length) % scenarios.length];
        selectValidationScenario(next.id);
        window.requestAnimationFrame(() => $('#validationCases [data-validation-scenario="' + next.id + '"]')?.focus());
      });
    });
    $('#shapeReadout').textContent = 'M=' + context.M + ' · K=' + context.K + ' · N=' + context.N;
    $('#tileReadout').textContent = 'baseM/N/K=' + context.baseM + '/' + context.baseN + '/' + context.baseK + ' · ' + state.trace.tiling.dtype;
    $('#validationSummary').innerHTML =
      '<span><strong>Last output tile</strong> #' + context.tileIdx + ' · curM=' + context.curM + ' · curN=' + context.curN + '</span>' +
      '<span><strong>Final K slices</strong> ' + context.tailKL1 + ' → ' + context.kSlices.at(-1).l0Slices.join(' / ') + ' · ' + context.mmadPerOutputTile + ' Mmad</span>' +
      '<span class="badge ' + evidenceClass(scenario.evidence) + '" title="source lines ' + escapeHtml(scenario.sourceLines.join(', ')) + '">' + escapeHtml(scenario.evidence) + '</span>';
    $('#validationDescription').textContent = scenario.description + ' Deterministic source-formula validation; not a device run.';
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

  function renderLoopContext() {
    const frame = currentFrame();
    const execution = state.trace.execution;
    const loops = new Map((execution?.loops || []).map((loop) => [loop.id, loop]));
    const outputLoop = loops.get('output-tile-loop');
    const l1Loop = loops.get('l1-k-loop');
    const l0Loop = loops.get('l0-k-loop');
    const value = (current, count) => current == null ? 'not entered' : current + ' / ' + Math.max(0, count - 1);
    const tileValue = value(frame.tileIdx, state.context.outputTileNum);
    const l1Value = value(frame.iter0, state.context.kL1TileNum);
    const l0Value = value(frame.iter1, frame.l0Count || state.context.kL0TileNum);

    $('#frameCounter').textContent = 'Frame ' + (state.frameIndex + 1) + ' / ' + state.frames.length;
    $('#loopTree').innerHTML =
      '<div class="matmul-loop-row is-root">' +
        '<span class="matmul-loop-row__spacer" aria-hidden="true"></span>' +
        '<div><strong>' + escapeHtml(outputLoop?.label || 'Output tile loop') + '</strong><span>tileIdx · ' + tileValue + '</span></div>' +
        '<span class="tag">×' + escapeHtml(state.context.outputTileNum) + '</span>' +
      '</div>' +
      '<div class="matmul-loop-row is-l1">' +
        '<button class="btn btn-icon btn-ghost matmul-loop-toggle" type="button" data-loop-toggle="l1" aria-expanded="' + state.loopExpanded.l1 + '" aria-label="Toggle L1 K loop">' + loopToggleIcon(state.loopExpanded.l1) + '</button>' +
        '<div><strong>' + escapeHtml(l1Loop?.label || 'L1 K loop') + '</strong><span>iter0 · ' + l1Value + ' · K=' + escapeHtml(frame.curKL1 ?? state.context.kL1) + '</span></div>' +
        '<span class="tag">×' + escapeHtml(state.context.kL1TileNum) + '</span>' +
      '</div>' +
      (state.loopExpanded.l1 ?
        '<div class="matmul-loop-row is-l0">' +
          '<button class="btn btn-icon btn-ghost matmul-loop-toggle" type="button" data-loop-toggle="l0" aria-expanded="' + state.loopExpanded.l0 + '" aria-label="Toggle L0 K loop">' + loopToggleIcon(state.loopExpanded.l0) + '</button>' +
          '<div><strong>' + escapeHtml(l0Loop?.label || 'L0 K loop') + '</strong><span>iter1 · ' + l0Value + ' · K=' + escapeHtml(frame.curKL0 ?? state.context.baseK) + '</span></div>' +
          '<span class="tag">×' + escapeHtml(frame.l0Count || state.context.kL0TileNum) + '</span>' +
        '</div>' +
        (state.loopExpanded.l0 ? '<div class="matmul-loop-leaf"><span class="badge ' + (frame.mmadMode === 'initialize CO1' ? 'badge--warning' : 'badge--success') + '">' + escapeHtml(frame.mmadMode || 'waiting for Mmad') + '</span><span>' + escapeHtml(frame.mmadOrdinal ? 'Mmad ' + frame.mmadOrdinal + '/' + state.context.mmadPerOutputTile : state.context.mmadPerOutputTile + ' Mmad per output tile') + '</span></div>' : '')
        : '');

    $('#loopTree').querySelectorAll('[data-loop-toggle]').forEach((button) => {
      button.addEventListener('click', () => {
        const key = button.dataset.loopToggle;
        state.loopExpanded[key] = !state.loopExpanded[key];
        renderLoopContext();
      });
    });

    const validationBranches = [
      { id: 'm-tail', selected: state.context.curM < state.context.baseM ? 'tail' : 'full-tile', value: state.context.curM, explanation: 'curM on the selected last output tile · source line 112' },
      { id: 'n-tail', selected: state.context.curN < state.context.baseN ? 'tail' : 'full-tile', value: state.context.curN, explanation: 'curN on the selected last output tile · source line 113' },
      { id: 'k-l1-tail', selected: state.context.tailKL1 < state.context.kL1 ? 'tail' : 'full-tile', value: state.context.tailKL1, explanation: 'curGmBKL1 on the final L1 K slice · source line 129' },
      { id: 'k-l0-tail', selected: state.context.tailKL0 < state.context.baseK ? 'tail' : 'full-tile', value: state.context.tailKL0, explanation: 'curKL0 on the final L0 K slice · source line 159' }
    ];
    $('#branchState').innerHTML = '<span class="matmul-branch-state__label">Branches</span>' + validationBranches.map((branch) =>
      '<span class="tag" title="' + escapeHtml(branch.explanation) + '">' +
      escapeHtml(branch.id) + ' · ' + escapeHtml(branch.selected) + ' · ' + escapeHtml(branch.value) + '</span>'
    ).join('');
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

  function matrixSceneForTensor(view) {
    const rows = view.logicalShape[0];
    const columns = view.logicalShape[1];
    const rowSpan = view.grid.rowSpan;
    const columnSpan = view.grid.columnSpan;
    const coords = tileCoordinates();
    const cells = [];
    const rowCount = Math.ceil(rows / rowSpan);
    const columnCount = Math.ceil(columns / columnSpan);
    let activeRow = 0;
    let activeColumn = 0;
    if (view.id === 'tensor:a') [activeRow, activeColumn] = [coords.mTile, coords.iter0];
    if (view.id === 'tensor:b') [activeRow, activeColumn] = [coords.iter0, coords.nTile];
    if (view.id === 'tensor:c') [activeRow, activeColumn] = [coords.mTile, coords.nTile];
    if (view.id === 'tensor:a1') [activeRow, activeColumn] = [0, coords.iter1];
    if (view.id === 'tensor:b1') [activeRow, activeColumn] = [coords.iter1, 0];

    for (let rowIndex = 0; rowIndex < rowCount; rowIndex += 1) {
      for (let columnIndex = 0; columnIndex < columnCount; columnIndex += 1) {
        const active = view.id === 'tensor:co1'
          ? ['mmad', 'sync-m-fix', 'l0c-to-gm'].includes(currentStep().id)
          : rowIndex === activeRow && columnIndex === activeColumn;
        const states = [];
        if (active) states.push('current');
        else if (view.id === 'tensor:co1' && tensorState(view).startsWith('accumulated')) states.push('written');
        cells.push({
          id: view.id + ':' + rowIndex + ':' + columnIndex,
          row: rowIndex * rowSpan,
          column: columnIndex * columnSpan,
          rowSpan: Math.min(rowSpan, rows - rowIndex * rowSpan),
          columnSpan: Math.min(columnSpan, columns - columnIndex * columnSpan),
          label: view.id === 'tensor:co1' ? 'C0 (' + rowIndex + ',' + columnIndex + ')' : view.label + '[' + rowIndex + ',' + columnIndex + ']',
          tone: view.role === 'output' ? 'output' : view.role === 'reduction' ? 'reduction' : 'input',
          style: 'value',
          states
        });
      }
    }
    return { extent: { rows, columns }, axes: { rows: view.axes[0], columns: view.axes[1] }, cells };
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
        phase: currentStep().label,
        operationChips: [primaryAction(currentStep())],
        stepIndex: state.frameIndex + 1,
        totalSteps: state.frames.length
      },
      constraints: view.constraints,
      status: context.length ? context.join(' · ') : currentStep().label
    };
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

  function hardwareContract() {
    return state.trace.hardwareParticipation;
  }

  function hardwareFocus() {
    const contract = hardwareContract();
    return contract.focusByStep[currentStep().id] || { units: [], routes: [], summary: 'No on-chip participation asserted.' };
  }

  function focusedHardwareUnits() {
    const contract = hardwareContract();
    const active = new Set(hardwareFocus().units);
    return contract.units.filter((unit) => active.has(unit.id));
  }

  function hardwareFrameSize() {
    const graph = $('#hardwareArchitectureGraph');
    const root = graph?.querySelector('.pto-mem950') || graph?.firstElementChild || graph;
    if (!root) return { width: 1200, height: 820 };
    const rect = root.getBoundingClientRect();
    return {
      width: Math.max(1, root.offsetWidth || root.scrollWidth || rect.width || 1200),
      height: Math.max(1, root.offsetHeight || root.scrollHeight || rect.height || 820)
    };
  }

  function applyHardwareFocus() {
    if (!state.hardwareInitialized) return;
    const helper = window.PtoMemoryArchitecturePattern;
    const graph = $('#hardwareArchitectureGraph');
    const contract = hardwareContract();
    const focus = hardwareFocus();
    const selectors = contract.units.filter((unit) => focus.units.includes(unit.id)).map((unit) => unit.selector);
    helper?.setPathFocus?.(graph, contract.visualPreset, { selectors, routes: focus.routes });
    state.hardwareRouteOverlay?.render?.();
  }

  function ensureHardwareViewport() {
    if (state.hardwareInitialized) return;
    const contract = hardwareContract();
    const memoryHelper = window.PtoMemoryArchitecturePattern;
    const viewportHelper = window.PtoHardwareArchitectureViewport;
    const graph = $('#hardwareArchitectureGraph');
    if (!memoryHelper?.renderArchitecture || !viewportHelper?.mount || !graph) return;
    memoryHelper.renderArchitecture(graph, contract.visualPreset);
    state.hardwareRouteOverlay = memoryHelper.createRouteOverlay(graph, contract.visualPreset);
    state.hardwareRouteOverlay?.render?.();
    state.hardwareViewport = viewportHelper.mount('#hardwareViewport', {
      mode: 'inline',
      viewport: '[data-hardware-stage]',
      scaleEl: '[data-hardware-scale]',
      inlineHost: '#hardwareArchitectureGraph',
      detailToggle: '[data-hardware-detail]',
      zoomOut: '[data-hardware-zoom-out]',
      zoomIn: '[data-hardware-zoom-in]',
      fit: '[data-hardware-fit]',
      readout: '[data-hardware-readout]',
      frameSize: hardwareFrameSize(),
      scale: 0.25,
      defaultScale: 0.25,
      fitOnMount: true,
      fitPaddingX: 20,
      fitPaddingY: 20,
      onScaleChange: () => state.hardwareRouteOverlay?.schedule?.(),
      onPanChange: () => state.hardwareRouteOverlay?.schedule?.(),
      onDetailChange: () => state.hardwareRouteOverlay?.schedule?.()
    });
    state.hardwareInitialized = true;
    window.requestAnimationFrame(() => {
      const size = hardwareFrameSize();
      state.hardwareViewport?.setFrameSize(size.width, size.height);
      state.hardwareViewport?.fit();
      applyHardwareFocus();
    });
  }

  function renderHardwareParticipation() {
    const contract = hardwareContract();
    const focus = hardwareFocus();
    const active = new Set(focus.units);
    $('#hardwareSummary').innerHTML = '<strong>' + escapeHtml(currentStep().label) + '</strong><span>' + escapeHtml(focus.summary) + '</span>';
    $('#hardwareUnits').innerHTML = contract.units.map((unit) =>
      '<span class="tag matmul-hardware-unit is-' + escapeHtml(unit.kind) + (active.has(unit.id) ? ' is-current' : '') + '">' + escapeHtml(unit.label) + '</span>'
    ).join('');
    $('#hardwareEventPath').innerHTML = state.trace.events.map((event) => {
      const isSet = event.setStepId === currentStep().id;
      const isWait = event.waitStepId === currentStep().id;
      const current = isSet || isWait;
      const phase = isSet && isWait ? 'SET + WAIT' : isSet ? 'SET' : isWait ? 'WAIT' : event.scope;
      return '<div class="matmul-hardware-event' + (current ? ' is-current' : '') + '">' +
        '<span>' + escapeHtml(event.producerEngine) + '</span><b>→ ' + escapeHtml(event.eventType) + ' →</b><span>' + escapeHtml(event.consumerEngine) + '</span>' +
        '<span class="badge ' + (current ? 'badge--warning' : '') + '">' + escapeHtml(phase) + '</span></div>';
    }).join('');
    applyHardwareFocus();
  }

  function renderDetailTabs() {
    document.querySelectorAll('[data-detail-tab]').forEach((button) => {
      const active = button.dataset.detailTab === state.detailTab;
      button.classList.toggle('is-selected', active);
      button.setAttribute('aria-selected', String(active));
    });
    document.querySelectorAll('[data-detail-view]').forEach((view) => {
      view.hidden = view.dataset.detailView !== state.detailTab;
    });
  }

  function selectDetailTab(tab) {
    state.detailTab = tab;
    renderDetailTabs();
    if (tab === 'tensor') {
      renderTensorTabs();
      renderTensorDetail();
      window.requestAnimationFrame(() => state.matrixController?.resize());
    }
    if (tab === 'hardware') {
      ensureHardwareViewport();
      renderHardwareParticipation();
      window.requestAnimationFrame(() => {
        const size = hardwareFrameSize();
        state.hardwareViewport?.setFrameSize(size.width, size.height);
        state.hardwareViewport?.fit();
        applyHardwareFocus();
      });
    }
  }

  function syncTensorFocusForStep() {
    state.selectedTensorId = state.trace.tensorFocusByStep[currentStep().id] || state.selectedTensorId;
  }

  function publishExecutionState() {
    const frameRoot = document.querySelector('.matmul-frame');
    const step = currentStep();
    const detail = {
      stepIndex: state.stepIndex,
      stepId: step.id,
      frameIndex: state.frameIndex,
      frame: { ...currentFrame() },
      scenarioId: state.scenarioId,
      validationContext: { ...state.context, kSlices: state.context.kSlices.map((slice) => ({ ...slice, l0Slices: [...slice.l0Slices] })) },
      action: primaryAction(step),
      unit: step.unit
    };
    if (frameRoot) {
      frameRoot.dataset.currentStep = detail.stepId;
      frameRoot.dataset.currentAction = detail.action.toLowerCase();
      frameRoot.dataset.currentUnit = String(detail.unit).toLowerCase();
    }
    window.dispatchEvent(new CustomEvent('matmul:execution-state', { detail }));
  }

  function instructionCardModel(stepIndex, title, flow, key, iteration = null, iterationRange = null, sourceStepIndexes = null) {
    return {
      stepIndex,
      title,
      flow,
      key,
      iteration,
      iterationRange,
      sourceStepIndexes: sourceStepIndexes || [stepIndex],
    };
  }

  function instructionLoopCards(iteration) {
    return [
      instructionCardModel(4, 'Copy Inputs', 'GM -> L1', 'gm-to-l1', iteration),
      instructionCardModel(5, 'MTE2_MTE1 Sync', 'MTE2 -> MTE1', 'sync-mte2-mte1', iteration),
      instructionCardModel(6, 'M_MTE1 Sync', 'Cube -> MTE1', 'm-mte1', iteration),
      instructionCardModel(6, 'Load Data A2 B2', 'L1 -> L0', 'load-a2-b2', iteration),
      instructionCardModel(6, 'MTE1_M Sync', 'MTE1 -> Cube', 'mte1-m', iteration),
      instructionCardModel(7, iteration === 0 ? 'Mmad Initialize' : 'Mmad Accumulate', iteration === 0 ? 'A2 x B2 -> CO1' : 'A2 x B2 + CO1', 'mmad-' + (iteration === 0 ? 'initialize' : 'accumulate'), iteration),
    ];
  }

  function createMatmulInstructionCard(card) {
    const selectedStep = card.sourceStepIndexes.includes(state.stepIndex);
    const selectedIteration = Number.isInteger(card.iteration)
      ? (!Number.isInteger(state.instructionIterationFocus) || card.iteration === state.instructionIterationFocus)
      : !Number.isInteger(state.instructionIterationFocus);
    const selectedOperation = !state.instructionOperationFocus || state.instructionOperationFocus === card.key;
    const selected = selectedStep && selectedIteration && selectedOperation;
    return '<button class="avz-instruction-card' + (selected ? ' is-selected' : '') + '" type="button" data-step-index="' + card.stepIndex + '" data-instruction-operation="' + escapeHtml(card.key) + '"' +
      (Number.isInteger(card.iteration) ? ' data-instruction-iteration="' + card.iteration + '"' : '') +
      (card.iterationRange ? ' data-instruction-iteration-start="' + card.iterationRange[0] + '" data-instruction-iteration-end="' + card.iterationRange[1] + '"' : '') +
      (selected ? ' aria-current="step"' : '') + '>' +
      '<span class="avz-instruction-card__title">' + escapeHtml(card.title) + '</span>' +
      '<span class="avz-instruction-card__flow">' + escapeHtml(card.flow) + '</span></button>';
  }

  function createMatmulInstructionRow(label, meta, cards, options = {}) {
    const hideLabel = options.hideLabel;
    const sourceStepIndexes = options.sourceStepIndexes || cards.flatMap((card) => card.sourceStepIndexes);
    const active = sourceStepIndexes.includes(state.stepIndex)
      && (!Number.isInteger(options.iteration) || options.iteration === state.instructionIterationFocus);
    return '<div class="avz-instruction-row' + (hideLabel ? ' is-label-hidden' : '') + (active ? ' is-active' : '') + '" role="group" aria-label="' + escapeHtml(label || 'K iterations') + '">' +
      (hideLabel ? '' : '<div class="avz-instruction-row__label"><span class="tag avz-iteration-tag">' + escapeHtml(label) + '</span>' + (meta ? '<span class="avz-instruction-row__meta">' + escapeHtml(meta) + '</span>' : '') + '</div>') +
      '<div class="avz-instruction-row__flow">' + cards.map(createMatmulInstructionCard).join('') + '</div></div>';
  }

  function createMatmulLoop() {
    const iterationCount = state.context?.kL1TileNum || 1;
    const lastIteration = Math.max(0, iterationCount - 1);
    const repeatedStart = Math.min(1, lastIteration);
    const repeatedEnd = lastIteration;
    const loopIndexes = [4, 5, 6, 7];
    const loopRows = [createMatmulInstructionRow('Iter 0', 'Initialize', instructionLoopCards(0), { sourceStepIndexes: loopIndexes })];
    const repeatGroup = [];
    if (repeatedEnd >= repeatedStart) {
      if (state.instructionLoopExpanded) {
        for (let iteration = repeatedStart; iteration <= repeatedEnd; iteration += 1) {
          repeatGroup.push(createMatmulInstructionRow('Iter ' + iteration, 'Accumulate', instructionLoopCards(iteration), { sourceStepIndexes: loopIndexes, iteration }));
        }
      } else {
        const representative = Number.isInteger(state.instructionIterationFocus)
          && state.instructionIterationFocus >= repeatedStart
          && state.instructionIterationFocus <= repeatedEnd
          ? state.instructionIterationFocus
          : repeatedStart;
        const cards = instructionLoopCards(representative).map((card) => ({
          ...card,
          sourceStepIndexes: loopIndexes,
          iterationRange: [repeatedStart, repeatedEnd],
        }));
        repeatGroup.push(createMatmulInstructionRow('', '', cards, { hideLabel: true, sourceStepIndexes: loopIndexes }));
      }
    }
    const repeatHeader = '<div class="avz-instruction-repeat__header"><div class="avz-instruction-row__label"><span class="tag avz-iteration-tag">Iter ' + repeatedStart + '–' + repeatedEnd + '</span><span class="avz-instruction-row__meta">×' + (repeatedEnd - repeatedStart + 1) + '</span></div><button class="btn btn-sm btn-ghost avz-instruction-loop-toggle" type="button" data-loop-toggle="iterations" aria-expanded="' + state.instructionLoopExpanded + '">' + (state.instructionLoopExpanded ? 'Group similar' : 'Show all') + '</button></div>';
    const repeat = '<div class="avz-instruction-repeat">' + repeatHeader + (repeatGroup.length ? '<div class="avz-instruction-repeat__rows">' + repeatGroup.join('') + '</div>' : '') + '</div>';
    const active = loopIndexes.includes(state.stepIndex);
    return '<section class="avz-instruction-loop' + (active ? ' is-active' : '') + '" role="listitem" aria-label="K Loop, iterations 0 through ' + lastIteration + '">' +
      '<div class="avz-instruction-loop__title">K Loop</div><div class="avz-instruction-loop__rows">' + loopRows.join('') + repeat + '</div></section>';
  }

  function renderFlow() {
    const steps = state.trace.steps;
    const before = [
      instructionCardModel(0, 'Input Shape', 'Host config', steps[0].id),
      instructionCardModel(1, 'Kernel Tiling', 'Kernel config', steps[1].id),
      instructionCardModel(2, 'Host执行配置', 'Host/Runtime config', steps[2].id),
      instructionCardModel(3, 'Allocate Memory', 'Kernel prepare', steps[3].id),
    ];
    const after = [
      instructionCardModel(8, 'M_FIX Sync', 'Cube -> Fix', steps[8].id),
      instructionCardModel(9, 'Copy L0C2GM', 'L0C -> GM', steps[9].id),
      instructionCardModel(10, 'Host Verify', 'Device -> Host', steps[10].id),
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

  function renderCurrent() {
    const step = currentStep();
    const frame = currentFrame();
    const contextualFacts = [];
    if (frame.tileIdx != null) contextualFacts.push('tileIdx=' + frame.tileIdx);
    if (frame.curM != null) contextualFacts.push('curM=' + frame.curM);
    if (frame.curN != null) contextualFacts.push('curN=' + frame.curN);
    if (frame.iter0 != null) contextualFacts.push('iter0=' + frame.iter0 + '/' + (state.context.kL1TileNum - 1));
    if (frame.curKL1 != null) contextualFacts.push('curKL1=' + frame.curKL1);
    if (frame.iter1 != null) contextualFacts.push('iter1=' + frame.iter1 + '/' + ((frame.l0Count || state.context.kL0TileNum) - 1));
    if (frame.curKL0 != null) contextualFacts.push('curKL0=' + frame.curKL0);
    if (frame.mmadMode) contextualFacts.push(frame.mmadMode);
    $('#currentRole').textContent = step.role + ' · ' + step.unit;
    $('#currentLabel').textContent = frame.stepId === 'mmad' ? frameLabel(frame) : step.label;
    $('#currentEvidence').textContent = step.evidence;
    $('#currentEvidence').className = 'badge ' + evidenceClass(step.evidence);
    $('#currentSummary').textContent = step.summary;
    $('#currentFacts').innerHTML = step.facts.concat(contextualFacts).map((fact) =>
      '<span class="matmul-fact">' + escapeHtml(fact) + '</span>'
    ).join('');
    $('#flowMeta').textContent = primaryAction(step) + ' · ' + step.unit + ' · logical order · duration unavailable';
  }

  function renderDetails() {
    const step = currentStep();
    const trace = state.trace;
    $('#launchFacts').innerHTML = trace.launchFacts.map((fact) =>
      '<dt>' + escapeHtml(fact[0]) + '</dt><dd>' + escapeHtml(fact[1]) + '</dd>'
    ).join('');
    const tierMembers = {
      gm: ['gm-a', 'gm-b', 'gm-c'],
      l1: ['a1', 'b1'],
      l0a: ['a2'],
      l0b: ['b2'],
      l0c: ['l0c']
    };
    $('#memoryList').innerHTML = trace.memory.tiers.map((item) =>
      '<div class="matmul-memory-item' + (tierMembers[item.id]?.some((id) => step.memory.includes(id)) ? ' is-current' : '') + '">' +
      '<span class="matmul-memory-item__dot" aria-hidden="true"></span><div>' +
      '<strong class="matmul-memory-item__name">' + escapeHtml(item.label) + '</strong>' +
      '<span class="matmul-memory-item__meta">' + escapeHtml(item.role) + ' · ' +
      escapeHtml(item.capacity) + '</span></div></div>'
    ).join('');
    renderDetailTabs();
    renderTensorTabs();
    renderTensorDetail();
    renderMemoryFlowMap();
    renderTensorJourney();
    renderHardwareParticipation();
    renderEvents();
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

  function selectStep(index, options = {}) {
    if (!options.fromPlayback) {
      state.playing = false;
      stopPlaybackTimer();
    }
    state.stepIndex = Math.max(0, Math.min(state.trace.steps.length - 1, index));
    state.instructionIterationFocus = options.instructionIteration ?? null;
    state.instructionOperationFocus = options.instructionOperation ?? null;
    if (!options.keepFrame) {
      const nextFrameIndex = state.frames.findIndex((frame) => frame.stepId === currentStep().id);
      if (nextFrameIndex >= 0) state.frameIndex = nextFrameIndex;
    }
    syncTensorFocusForStep();
    const role = currentStep().role.toLowerCase();
    if (role === 'kernel' || role === 'host') state.sourceRole = role;
    renderRoleTabs();
    renderSource();
    renderFlow();
    renderCurrent();
    renderLoopContext();
    renderDetails();
    renderExecutionDock();
    syncPlayback();
    publishExecutionState();
    $('#statusText').textContent = currentScenario().label + ' · ' + frameLabel() + ' · source-linked · logical playback';
    $('#statusScenario').textContent = currentScenario().label;
    $('#statusStep').textContent = (state.stepIndex + 1) + ' / ' + state.trace.steps.length;
  }

  function bindControls() {
    $('#instructionSequence').addEventListener('click', (event) => {
      const toggle = event.target.closest('[data-loop-toggle]');
      if (toggle) {
        state.instructionLoopExpanded = !state.instructionLoopExpanded;
        renderExecutionDock();
        return;
      }
      const card = event.target.closest('[data-step-index]');
      if (!card) return;
      const iterationStart = Number(card.dataset.instructionIterationStart);
      const iterationEnd = Number(card.dataset.instructionIterationEnd);
      const hasRange = Number.isInteger(iterationStart) && Number.isInteger(iterationEnd);
      selectStep(Number(card.dataset.stepIndex), {
        instructionIteration: hasRange ? null : Number.isInteger(Number(card.dataset.instructionIteration)) ? Number(card.dataset.instructionIteration) : null,
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
    state.frames = buildExecutionFrames(state.trace, state.context);
    bindControls();
    initPlayback();
    renderValidationControls();
    renderExecutionDock();
    selectStep(0);
    window.PtoIdeFrame?.initAll?.();
  }

  load().catch((error) => {
    $('#statusText').textContent = 'Load failed · ' + error.message;
    $('#sourceLines').innerHTML = '<p class="matmul-load-error">' + escapeHtml(error.message) + '<br>请通过 HTTP 服务打开此页面。</p>';
  });
})();
