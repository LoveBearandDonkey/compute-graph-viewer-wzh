/*
  应用控制器 —— 把数据层、规则引擎、三个视图挂到 IDE 框架的槽位上。
  ------------------------------------------------------------------
  单一状态源 + 单向渲染：所有交互只改 state，然后走一次 render()。
  时间游标 tick 是三个视图与底部水位曲线的共同坐标，选中项在视图间互相高亮。
*/
(function bootMemViz(global) {
  'use strict';

  const F = global.MemVizFormat;
  const $ = (id) => document.getElementById(id);

  const state = {
    chipId: 'ascend-910b',
    runId: 't32',
    view: 'layout',
    layoutMode: 'address', // address = 地址条带图；arch = 硬件架构图
    focusRegionId: 'UB',
    tick: 0,
    selectedAllocId: null,
    selectedEventId: null,
    selectedFindingId: null,
    explorerView: 'files',
    selectedFile: 'kernel_cpp',
    expandedFolders: new Set(['root', 'op_host', 'op_kernel', 'scripts', 'tests']),
    plannerLayoutMode: 'list',
    plannerRegionId: 'UB',
    plannerBufferName: 'gammaBuf',
  };

  let chip = null;
  let runs = [];
  let run = null;
  let metrics = null;
  let findings = [];
  let runIndex = new Map(); // runId -> { run, metrics, findings, summary }
  let views = {};
  let frameController = null;
  let plannerArchController = null;

  // ---------------------------------------------------------------
  // 数据装载
  // ---------------------------------------------------------------
  function loadChip(chipId) {
    chip = global.MemVizChips.get(chipId);
    runs = global.MemVizRuns.buildAll(chip);
    runIndex = new Map(runs.map((item) => {
      const m = global.MemVizMetrics.compute(item, chip);
      const f = global.MemVizDiagnostics.analyze(item, m);
      return [item.id, { run: item, metrics: m, findings: f, summary: global.MemVizDiagnostics.summarize(f) }];
    }));
    if (!runIndex.has(state.runId)) state.runId = runs[0].id;
    if (!chip.regions.some((r) => r.id === state.focusRegionId)) state.focusRegionId = chip.regions[0].id;
  }

  function selectRun(runId) {
    state.runId = runId;
    const entry = runIndex.get(runId);
    run = entry.run;
    metrics = entry.metrics;
    findings = entry.findings;
    state.tick = Math.min(state.tick, run.totalTicks);
    state.selectedAllocId = null;
    state.selectedEventId = null;
    state.selectedFindingId = null;
  }

  // ---------------------------------------------------------------
  // 顶栏 / 工具栏
  // ---------------------------------------------------------------
  function renderChipSwitch() {
    const host = $('chipSwitch');
    host.innerHTML = '';
    global.MemVizChips.list().forEach((item) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `btn btn-sm${item.id === state.chipId ? ' is-selected' : ''}`;
      btn.textContent = item.name;
      btn.title = `规格来源 ${item.specRef}（占位值）`;
      btn.addEventListener('click', () => {
        if (state.chipId === item.id) return;
        state.chipId = item.id;
        loadChip(item.id);
        selectRun(state.runId);
        views.lifetime?.destroy?.();
        views.lifetime = global.MemVizLifetimeView.create($('viewLifetime'));
        renderChipSwitch();
        renderRegionSwitch();
        render();
      });
      host.appendChild(btn);
    });
  }

  const LAYOUT_MODES = [
    {
      id: 'address', label: '地址布局',
      title: '按地址空间摊平的条带图',
      note: '横轴是各层级自己的地址空间。实心块 = 当前时刻真正持有数据，半透明块 = 预留着但此刻是空的，'
        + '斜纹 = 分配之间的碎片，竖线右侧的红区 = 超出容量的部分。',
    },
    {
      id: 'arch', label: '硬件架构',
      title: '把同一份读数贴回硬件架构图',
      note: '同一份数据贴回硬件本身：每块存储下方是物理容量、对齐要求、静态预留与利用率，'
        + '格子按此刻占用着色（实色 = 持有，灰色 = 预留未用，警告色 = 该层级已超限）。'
        + '悬停看完整读数，点击把焦点层级切过去；拖拽平移，⌘/Ctrl + 滚轮缩放。',
    },
  ];

  function renderLayoutModeSwitch() {
    const host = $('layoutModeSwitch');
    host.innerHTML = '';
    LAYOUT_MODES.forEach((mode) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `btn btn-sm${mode.id === state.layoutMode ? ' is-selected' : ''}`;
      btn.textContent = mode.label;
      btn.title = mode.title;
      btn.addEventListener('click', () => {
        if (state.layoutMode === mode.id) return;
        state.layoutMode = mode.id;
        renderLayoutModeSwitch();
        render();
        window.requestAnimationFrame(redrawViews);
      });
      host.appendChild(btn);
    });
  }

  function renderRegionSwitch() {
    const host = $('regionSwitch');
    host.innerHTML = '';
    chip.regions.forEach((region) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `btn btn-sm${region.id === state.focusRegionId ? ' is-selected' : ''}`;
      btn.textContent = region.id;
      btn.title = region.kind === 'register'
        ? `${region.label} · ${Math.round(region.capacity / region.regBytes)} regs × ${region.regBytes}B`
        : `${region.label} · ${F.bytes(region.capacity)} · ${region.align}B 对齐`;
      btn.addEventListener('click', () => {
        state.focusRegionId = region.id;
        renderRegionSwitch();
        render();
      });
      host.appendChild(btn);
    });
  }

  const LEGENDS = {
    layout: [
      { cls: '', label: '当前持有' },
      { cls: 'is-ghost', label: '预留未用' },
      { cls: 'is-gap', label: '碎片' },
      { cls: 'is-over', label: '超出容量' },
    ],
    arch: [
      { cls: '', label: '当前持有' },
      { cls: 'is-ghost', label: '预留未用' },
      { cls: 'is-warn-fill', label: '超出容量' },
    ],
    lifetime: [],
    pipeline: [
      { cls: '', label: '流水任务' },
      { cls: 'is-wait', label: '可归因等待' },
      { cls: 'is-gap', label: '启动/空闲' },
    ],
  };

  function renderLegend() {
    const host = $('viewLegend');
    const key = state.view === 'layout' && state.layoutMode === 'arch' ? 'arch' : state.view;
    const items = LEGENDS[key] || [];
    host.innerHTML = items.map((item) => `
      <span class="mv-legend-item"><span class="mv-legend-swatch ${item.cls}"></span>${item.label}</span>
    `).join('');
  }

  // ---------------------------------------------------------------
  // 左栏：Ascend C 工程 / tiling 候选
  // ---------------------------------------------------------------
  const ASCEND_C_PROJECT = [
    {
      id: 'root', label: 'MatmulLayerNorm_mix', type: 'folder', children: [
        {
          id: 'op_host', label: 'op_host', type: 'folder', children: [
            { id: 'op_host_cpp', label: 'matmul_layernorm_mix.cpp', type: 'cpp' },
            { id: 'tiling_h', label: 'matmul_layernorm_mix_tiling.h', type: 'header' },
          ],
        },
        {
          id: 'op_kernel', label: 'op_kernel', type: 'folder', children: [
            { id: 'kernel_cpp', label: 'matmul_layernorm_mix.cpp', type: 'cpp', source: true },
            { id: 'tiling_data_h', label: 'matmul_layernorm_mix_tiling_data.h', type: 'header' },
          ],
        },
        {
          id: 'scripts', label: 'scripts', type: 'folder', children: [
            { id: 'gen_data', label: 'gen_data.py', type: 'python' },
          ],
        },
        {
          id: 'tests', label: 'tests', type: 'folder', children: [
            { id: 'test_case', label: 'test_matmul_layernorm_mix.py', type: 'python' },
          ],
        },
        { id: 'cmake', label: 'CMakeLists.txt', type: 'cmake' },
        { id: 'build', label: 'build.sh', type: 'shell' },
      ],
    },
  ];

  const TREE_ICON = {
    folder: '<path d="M3 6.5h6l2 2h10v9.5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"></path><path d="M3 9h18"></path>',
    cpp: '<path d="M6 2h8l4 4v16H6Z"></path><path d="M14 2v5h5"></path><path d="m11 12-2 2 2 2"></path><path d="m14 12 2 2-2 2"></path>',
    header: '<path d="M6 2h8l4 4v16H6Z"></path><path d="M14 2v5h5"></path><path d="M9 12h6M9 16h6"></path>',
    python: '<path d="M6 2h8l4 4v16H6Z"></path><path d="M14 2v5h5"></path><path d="M9 13h6M12 10v6"></path>',
    cmake: '<path d="M6 2h8l4 4v16H6Z"></path><path d="M14 2v5h5"></path><circle cx="12" cy="14" r="2.5"></circle>',
    shell: '<path d="M6 2h8l4 4v16H6Z"></path><path d="M14 2v5h5"></path><path d="m9 12 2 2-2 2M12.5 16H15"></path>',
  };

  function treeIcon(type) {
    return `<svg class="mv-tree-icon${type === 'folder' ? ' is-folder' : ''}" viewBox="0 0 24 24" aria-hidden="true">${TREE_ICON[type] || TREE_ICON.cpp}</svg>`;
  }

  const CPP_KEYWORDS = new Set([
    'alignas', 'auto', 'break', 'case', 'class', 'const', 'constexpr', 'continue', 'default', 'do',
    'else', 'for', 'if', 'inline', 'namespace', 'private', 'public', 'return', 'sizeof', 'static',
    'struct', 'switch', 'template', 'typename', 'using', 'while', '__aicore__', '__global__',
  ]);
  const CPP_TYPES = new Set([
    'bool', 'char', 'double', 'float', 'half', 'int', 'int8_t', 'int16_t', 'int32_t', 'int64_t',
    'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t', 'void', 'GM_ADDR', 'GlobalTensor', 'LocalTensor',
    'TBuf', 'TPipe', 'TQue', 'QuePosition', 'MLNTiling', 'MatmulLayerNormMixKernel',
  ]);

  function highlightCppLine(line) {
    const tokenPattern = /(\/\/.*$|\/\*.*?\*\/|#[A-Za-z_][A-Za-z0-9_]*|<[^>]+>|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b(?:0x[\dA-Fa-f]+|\d+(?:\.\d+)?)\b|\b[A-Za-z_][A-Za-z0-9_]*\b)/g;
    let result = '';
    let cursor = 0;
    let match;
    while ((match = tokenPattern.exec(line)) !== null) {
      result += F.escapeHtml(line.slice(cursor, match.index));
      const token = match[0];
      const rest = line.slice(match.index + token.length);
      let cls = 'mv-tok-variable';
      if (token.startsWith('//') || token.startsWith('/*')) cls = 'mv-tok-comment';
      else if (token.startsWith('#')) cls = 'mv-tok-preprocessor';
      else if (token.startsWith('<') || token.startsWith('"') || token.startsWith("'")) cls = 'mv-tok-string';
      else if (/^(?:0x[\dA-Fa-f]+|\d)/.test(token)) cls = 'mv-tok-number';
      else if (CPP_KEYWORDS.has(token)) cls = 'mv-tok-keyword';
      else if (/^\s*\(/.test(rest)) cls = 'mv-tok-function';
      else if (/^[A-Z][A-Z0-9_]*$/.test(token)) cls = 'mv-tok-constant';
      else if (CPP_TYPES.has(token) || /^[A-Z][A-Za-z0-9_]*$/.test(token)) cls = 'mv-tok-type';
      result += `<span class="${cls}">${F.escapeHtml(token)}</span>`;
      cursor = match.index + token.length;
      if (token.startsWith('//')) break;
    }
    return result + F.escapeHtml(line.slice(cursor));
  }

  function renderSourceEditor() {
    const source = global.MemVizKernelSource;
    $('sourceEditor').innerHTML = `<div class="mv-source-list">${source.lines.map((line, index) => `
      <div class="mv-source-line"><span class="mv-source-ln">${index + 1}</span><code class="mv-source-text">${highlightCppLine(line) || ' '}</code></div>
    `).join('')}</div>`;
  }

  const QUEUE_POSITION_REGION = {
    A1: 'L1', B1: 'L1', A2: 'L0A', B2: 'L0B', CO1: 'L0C',
    VECIN: 'UB', VECOUT: 'UB', VECCALC: 'UB',
  };

  function evaluateArithmetic(expression, variables) {
    let normalized = expression
      .replace(/sizeof\s*\(\s*(half|float|double|int8_t|uint8_t|int16_t|uint16_t|int32_t|uint32_t|int64_t|uint64_t)\s*\)/g,
        (_, type) => String({ half: 2, float: 4, double: 8, int8_t: 1, uint8_t: 1, int16_t: 2, uint16_t: 2, int32_t: 4, uint32_t: 4, int64_t: 8, uint64_t: 8 }[type]))
      .replace(/\b[A-Za-z_][A-Za-z0-9_]*\b/g, (name) => Object.hasOwn(variables, name) ? String(variables[name]) : `?${name}?`)
      .replace(/\s+/g, '');
    if (/[^\d.+\-*/()?]/.test(normalized) || normalized.includes('?')) return NaN;

    const tokens = normalized.match(/\d+(?:\.\d+)?|[()+\-*/]/g) || [];
    let cursor = 0;
    const parseFactor = () => {
      const token = tokens[cursor];
      if (token === '+' || token === '-') {
        cursor += 1;
        const value = parseFactor();
        return token === '-' ? -value : value;
      }
      if (token === '(') {
        cursor += 1;
        const value = parseExpression();
        if (tokens[cursor] !== ')') return NaN;
        cursor += 1;
        return value;
      }
      cursor += 1;
      return Number(token);
    };
    const parseTerm = () => {
      let value = parseFactor();
      while (tokens[cursor] === '*' || tokens[cursor] === '/') {
        const op = tokens[cursor++];
        const right = parseFactor();
        value = op === '*' ? value * right : value / right;
      }
      return value;
    };
    function parseExpression() {
      let value = parseTerm();
      while (tokens[cursor] === '+' || tokens[cursor] === '-') {
        const op = tokens[cursor++];
        const right = parseTerm();
        value = op === '+' ? value + right : value - right;
      }
      return value;
    }
    const value = parseExpression();
    return cursor === tokens.length && Number.isFinite(value) ? value : NaN;
  }

  function analyzeSourceBuffers() {
    const source = global.MemVizKernelSource.text;
    const variables = {
      tileM_: run.tiling.tileM,
      tileNum_: run.tiling.tileNum,
      A_L1_DB: run.tiling.bufferNum.aL1,
      A_L0A_DB: run.tiling.bufferNum.aL0A,
      C_L0C_DB: run.tiling.bufferNum.cL0C,
      MM_OUT_DB: run.tiling.bufferNum.mmOut,
      Y_DB: run.tiling.bufferNum.yUb,
    };
    for (const match of source.matchAll(/constexpr\s+\w+\s+(\w+)\s*=\s*(\d+)/g)) {
      variables[match[1]] = Number(match[2]);
    }

    const declarations = new Map();
    for (const match of source.matchAll(/\b(TQue|TBuf)<QuePosition::([A-Z0-9]+)(?:,\s*[^>]+)?>\s+([^;]+);/g)) {
      const kind = match[1];
      const position = match[2];
      match[3].split(',').forEach((part) => {
        const name = part.trim().match(/([A-Za-z_][A-Za-z0-9_]*)$/)?.[1];
        if (name) declarations.set(name, { kind, position, regionId: QUEUE_POSITION_REGION[position] || 'UB' });
      });
    }

    const buffers = [];
    for (const match of source.matchAll(/InitBuffer\(\s*(\w+)\s*,\s*([^,]+)\s*,\s*([^;]+?)\s*\);/g)) {
      const name = match[1];
      const declaration = declarations.get(name) || { kind: 'Buffer', position: 'UNKNOWN', regionId: 'UB' };
      const slots = evaluateArithmetic(match[2], variables);
      const bytesPerSlot = evaluateArithmetic(match[3], variables);
      const region = chip.regions.find((item) => item.id === declaration.regionId);
      if (!Number.isFinite(slots) || !Number.isFinite(bytesPerSlot) || !region) continue;
      const alignedPerSlot = Math.ceil(bytesPerSlot / region.align) * region.align;
      buffers.push({
        name, ...declaration, slots, bytesPerSlot, alignedPerSlot,
        bytes: alignedPerSlot * slots,
        expression: match[3].trim(),
        line: source.slice(0, match.index).split('\n').length,
      });
    }

    const regions = chip.regions
      .filter((region) => region.kind !== 'register' && region.id !== 'GM')
      .map((region) => {
        const items = buffers.filter((buffer) => buffer.regionId === region.id);
        const used = items.reduce((sum, item) => sum + item.bytes, 0);
        return { ...region, items, used, ratio: used / region.capacity, remaining: region.capacity - used };
      })
      .filter((region) => region.items.length);

    return {
      buffers, regions,
      queueCount: buffers.filter((item) => item.kind === 'TQue').length,
      tbufCount: buffers.filter((item) => item.kind === 'TBuf').length,
      allocTensorCount: (source.match(/\.AllocTensor\s*</g) || []).length,
    };
  }

  function plannerListMarkup(plan) {
    return plan.regions.map((region) => {
      const level = region.ratio > 1 ? 'danger' : region.ratio >= 0.85 ? 'warning' : '';
      return `<div class="mv-plan-region">
        <div class="mv-plan-region-head">
          <span class="mv-plan-region-name">${region.id}</span>
          <span class="mv-plan-region-value ${level ? `is-${level}` : ''}">${F.bytes(region.used)} / ${F.bytes(region.capacity)} · ${F.pct(region.ratio, 0)}</span>
        </div>
        <div class="mv-plan-layout" aria-label="${region.id} 静态内存布局">
          ${region.items.map((item, index) => `<span class="mv-plan-block" style="flex:0 0 ${Math.max(0.5, item.bytes / region.capacity * 100).toFixed(2)}%;background:${region.accent};opacity:${(0.48 + index % 4 * 0.12).toFixed(2)}" title="${F.escapeHtml(item.name)} · ${F.bytes(item.bytes)}"></span>`).join('')}
        </div>
        <div class="mv-plan-waterline"><span>0</span><span>${region.remaining >= 0 ? `剩余 ${F.bytes(region.remaining)}` : `超出 ${F.bytes(-region.remaining)}`}</span><span>${F.bytes(region.capacity)}</span></div>
        <div class="mv-plan-buffer-list">
          ${region.items.map((item) => `<div class="mv-plan-buffer">
            <span class="mv-plan-buffer-name">${F.escapeHtml(item.name)}</span>
            <span class="mv-plan-buffer-size">${F.bytes(item.bytes)}</span>
            <span class="mv-plan-buffer-meta">${item.kind}&lt;${item.position}&gt; · ${item.slots} × ${F.bytes(item.bytesPerSlot)} · align ${region.align}B · L${item.line}</span>
          </div>`).join('')}
        </div>
      </div>`;
    }).join('');
  }

  function plannerArchitectureMarkup() {
    return `<div class="mv-plan-arch">
      <div id="planArchitectureHost"></div>
      <div>
        <div class="mv-sec-head"><span class="mv-label" id="planArchRegionLabel">当前硬件节点</span><span class="mv-label">选择 Buffer</span></div>
        <div class="mv-plan-arch-buffers" id="planArchBufferList"></div>
      </div>
      <div class="mv-plan-api-detail" id="planArchBufferDetail"></div>
    </div>`;
  }

  function apiUsagesForBuffer(bufferName) {
    const escaped = bufferName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const namePattern = new RegExp(`\\b${escaped}\\b`);
    const methodPattern = new RegExp(`${escaped}\\.([A-Za-z_][A-Za-z0-9_]*)(?:<([^>]+)>)?`);
    return global.MemVizKernelSource.lines.flatMap((line, index) => {
      if (!namePattern.test(line)) return [];
      const trimmed = line.trim();
      const method = trimmed.match(methodPattern);
      let api = '引用';
      if (/\bInitBuffer\s*\(/.test(trimmed)) api = 'TPipe::InitBuffer';
      else if (method) api = `${method[1]}${method[2] ? `<${method[2]}>` : ''}`;
      else if (/\b(?:TQue|TBuf)</.test(trimmed)) api = 'Buffer 声明';
      return [{ api, line: index + 1, code: trimmed }];
    });
  }

  function renderPlannerArchSelection(plan, stage, preset) {
    let region = plan.regions.find((item) => item.id === state.plannerRegionId);
    if (!region?.items.length) region = plan.regions.find((item) => item.items.length);
    if (!region) return;
    state.plannerRegionId = region.id;
    let buffer = region.items.find((item) => item.name === state.plannerBufferName);
    if (!buffer) buffer = region.items.find((item) => item.name === 'gammaBuf') || region.items[0];
    state.plannerBufferName = buffer.name;

    $('planArchRegionLabel').textContent = `${region.id} · ${region.label}`;
    $('planArchBufferList').innerHTML = plan.regions
      .filter((item) => item.items.length)
      .flatMap((item) => item.items.map((candidate) => `
        <button class="btn btn-sm${candidate.name === buffer.name ? ' is-selected' : ''}" type="button" data-plan-region="${F.escapeHtml(item.id)}" data-plan-buffer="${F.escapeHtml(candidate.name)}">${F.escapeHtml(item.id)} · ${F.escapeHtml(candidate.name)}</button>
      `)).join('');
    const usages = apiUsagesForBuffer(buffer.name);
    $('planArchBufferDetail').innerHTML = `
      <div class="mv-plan-buffer-list">
        <div class="mv-plan-buffer">
          <span class="mv-plan-api-name">${F.escapeHtml(buffer.name)}</span>
          <span class="mv-plan-api-size">${F.bytes(buffer.bytes)}</span>
          <span class="mv-plan-buffer-meta">${buffer.kind}&lt;${buffer.position}&gt; · ${buffer.slots} × ${F.bytes(buffer.bytesPerSlot)} · align ${region.align}B · L${buffer.line}</span>
        </div>
      </div>
      <div class="mv-sec-head"><span class="mv-label">使用该 Buffer 的 API</span><span class="mv-label">${usages.length} 处</span></div>
      <div class="mv-plan-buffer-list">
        ${usages.map((usage) => `<div class="mv-plan-buffer">
          <span class="mv-plan-buffer-name">${F.escapeHtml(usage.api)}</span>
          <span class="mv-plan-buffer-size">L${usage.line}</span>
          <code class="mv-plan-api-code">${F.escapeHtml(usage.code)}</code>
        </div>`).join('')}
      </div>
      <div class="mv-plan-buffer-list">
        <div class="mv-plan-buffer">
          <span class="mv-plan-buffer-name">大小表达式</span><span class="mv-plan-buffer-size">${F.escapeHtml(buffer.expression)}</span>
          <span class="mv-plan-buffer-meta">硬件归属 · ${region.id} · ${F.escapeHtml(region.owner)}</span>
        </div>
      </div>
    `;
    $('planArchBufferList').querySelectorAll('[data-plan-buffer]').forEach((button) => {
      button.addEventListener('click', () => {
        state.plannerRegionId = button.dataset.planRegion;
        state.plannerBufferName = button.dataset.planBuffer;
        renderPlannerArchSelection(plan, stage, preset);
      });
    });

    const helper = global.PtoMemoryArchitecturePattern;
    const target = global.MemVizArchView?.TARGETS?.[region.id];
    helper?.clearPathFocus?.(stage);
    if (target?.detail) helper?.setPathFocus?.(stage, preset, { selectors: [target.detail], routes: [] });
  }

  function mountPlannerArchitecture(plan) {
    const host = $('planArchitectureHost');
    const helper = global.PtoMemoryArchitecturePattern;
    if (!host || !helper) {
      if (host) host.innerHTML = '<p class="mv-empty">未加载 memory-architecture pattern。</p>';
      return;
    }

    const viewport = document.createElement('div');
    viewport.className = 'pto-memory-architecture-viewport mv-plan-arch-viewport';
    viewport.setAttribute('aria-label', '昇腾抽象硬件架构');
    const sizer = document.createElement('div');
    sizer.className = 'pto-memory-architecture-sizer';
    const canvas = document.createElement('div');
    canvas.className = 'pto-memory-architecture-canvas';
    const stage = document.createElement('div');
    canvas.appendChild(stage);
    sizer.appendChild(canvas);
    viewport.appendChild(sizer);
    host.appendChild(viewport);

    const presetKey = chip.id === 'ascend-910b' ? 'ascend910b' : 'ascend950b';
    const preset = helper.presets?.[presetKey] || helper.resolvePreset?.(presetKey);
    if (!preset) return;
    helper.renderArchitecture(stage, preset);
    helper.setDetailVisibility?.(stage, false);
    const overlay = helper.createRouteOverlay?.(stage, preset);
    overlay?.render?.();
    const hover = helper.attachHoverInteractions?.(stage, preset);
    const activation = helper.attachNodeActivation?.(stage, preset, {
      selector: '[data-aic-node^="buffer:"], [data-aiv-node^="buffer:"]',
      label: (target) => `查看 ${target.dataset.aicNode || target.dataset.aivNode || 'buffer'}`,
      onActivate: (target) => {
        const key = target.dataset.aicNode || target.dataset.aivNode || '';
        const regionId = global.MemVizArchView?.REGION_BY_NODE?.[key];
        const region = plan.regions.find((item) => item.id === regionId && item.items.length);
        if (!region) return;
        state.plannerRegionId = region.id;
        state.plannerBufferName = region.items[0].name;
        renderPlannerArchSelection(plan, stage, preset);
      },
    });
    activation?.targets?.forEach((target) => target.setAttribute('data-no-pan', ''));
    const zoom = helper.createZoomController?.({
      viewport, sizer, canvas,
      defaultZoom: 0.35, min: 0.35, max: 1.2, step: 0.1,
      pan: true, wheelZoom: true, centerOnReset: true,
      centerTarget: '.pto-mem950__rails, .pto-mem950__engine-stack, .pto-mem950__stack',
      onZoom: () => overlay?.render?.(),
      onPan: () => overlay?.render?.(),
    });

    const blocks = [];
    const targets = global.MemVizArchView?.TARGETS || {};
    plan.regions.forEach((region) => {
      const target = targets[region.id];
      if (!target?.cores) return;
      target.cores.forEach((coreId) => {
        const slot = stage.querySelector(`[id="${coreId}"]`);
        const bufferNode = slot?.querySelector(
          `[data-buffer-key="${target.buffer}"], [data-aic-node="buffer:${target.buffer}"], [data-aiv-node="buffer:${target.buffer}"]`,
        );
        const cells = bufferNode?.querySelectorAll('.pto-aiv-core__cell, .pto-aic-core__cell').length || 0;
        if (!cells) return;
        let cursor = 0;
        region.items.forEach((item) => {
          if (cursor >= cells) return;
          const cellCount = Math.max(1, Math.round(item.bytes / region.capacity * cells));
          const end = Math.min(cells - 1, cursor + cellCount - 1);
          blocks.push({
            core: coreId, buffer: target.buffer, state: region.ratio > 1 ? 'accumulating' : 'loaded', tone: region.ratio > 1 ? 'accumulator' : 'input',
            label: `${item.name} · ${F.bytes(item.bytes)} · ${item.kind}<${item.position}> · ${item.slots} × ${F.bytes(item.bytesPerSlot)} · align ${region.align}B · L${item.line}`,
            sourceTile: item.name, cellRange: [cursor, end],
          });
          cursor = end + 1;
        });
      });
    });
    helper.setBufferBlocks?.(stage, blocks);
    stage.querySelectorAll('[data-buffer-block-source-tile]').forEach((cell) => cell.setAttribute('data-no-pan', ''));
    const onArchitectureSelect = (event) => {
      const cell = event.target?.closest?.('[data-buffer-block-source-tile]');
      const node = event.target?.closest?.('[data-aic-node^="buffer:"], [data-aiv-node^="buffer:"]');
      if ((!cell && !node) || (cell && !stage.contains(cell)) || (node && !stage.contains(node))) return;
      const bufferName = cell?.dataset.bufferBlockSourceTile || '';
      const nodeKey = node?.dataset.aicNode || node?.dataset.aivNode || '';
      const regionId = bufferName
        ? plan.regions.find((item) => item.items.some((buffer) => buffer.name === bufferName))?.id
        : global.MemVizArchView?.REGION_BY_NODE?.[nodeKey];
      const region = plan.regions.find((item) => item.id === regionId && item.items.length);
      if (!region) return;
      event.stopPropagation();
      state.plannerRegionId = region.id;
      state.plannerBufferName = bufferName || region.items[0].name;
      renderPlannerArchSelection(plan, stage, preset);
    };
    stage.addEventListener('click', onArchitectureSelect, true);
    renderPlannerArchSelection(plan, stage, preset);
    window.requestAnimationFrame(() => {
      zoom?.center?.();
      overlay?.render?.();
    });

    plannerArchController = {
      destroy() {
        overlay?.destroy?.();
        hover?.destroy?.();
        activation?.destroy?.();
        zoom?.destroy?.();
        stage.removeEventListener('click', onArchitectureSelect, true);
        helper.clearPathFocus?.(stage);
      },
    };
  }

  function renderBufferPlanner() {
    plannerArchController?.destroy?.();
    plannerArchController = null;
    const plan = analyzeSourceBuffers();
    const ub = plan.regions.find((region) => region.id === 'UB');
    const critical = plan.regions.reduce((worst, region) => !worst || region.ratio > worst.ratio ? region : worst, null);
    const severity = critical?.ratio > 1 ? 'danger' : critical?.ratio >= 0.85 ? 'warning' : 'success';
    const summary = ub
      ? `UB 已占 <b>${F.bytes(ub.used)}</b> / ${F.bytes(ub.capacity)}，${ub.remaining >= 0 ? `剩余 <b>${F.bytes(ub.remaining)}</b>` : `超出 <b>${F.bytes(-ub.remaining)}</b>`}`
      : '源码中未解析到 UB buffer';

    $('sourcePlannerBody').innerHTML = `
      <section class="mv-sec">
        <div class="mv-sec-head"><span class="mv-sec-title">TilingData 试算</span><span class="mv-label">实时</span></div>
        <div class="segmented-control segmented-control-muted mv-plan-candidates" role="group" aria-label="TilingData 试算候选">
          ${runs.map((item) => `<button class="btn btn-sm${item.id === state.runId ? ' is-selected' : ''}" type="button" data-plan-run="${item.id}" title="${F.escapeHtml(item.note)}">${F.escapeHtml(item.label)}</button>`).join('')}
        </div>
        <p class="mv-plan-source-note">${F.escapeHtml(run.note)}</p>
      </section>
      <section class="mv-sec">
        <div class="mv-sec-head"><span class="mv-sec-title">静态规划结果</span><span class="stat-chip mv-sev-${severity === 'warning' ? 'warn' : severity}">${severity === 'danger' ? '超限' : severity === 'warning' ? '接近上限' : '容量安全'}</span></div>
        <div class="mv-soft ${severity === 'danger' ? 'is-danger' : severity === 'warning' ? 'is-warning' : ''} mv-plan-summary">
          <div class="mv-plan-summary-line">${summary}</div>
          <div class="mv-chip-row">
            <span class="stat-chip">tileM ${run.tiling.tileM}</span>
            <span class="stat-chip">tileNum ${run.tiling.tileNum}</span>
            <span class="stat-chip">block_dim ${run.kernel.blockDim}</span>
          </div>
        </div>
        <p class="mv-plan-source-note">无需编译 · 基于 TilingData 与源码声明实时推导</p>
      </section>
      <section class="mv-sec">
        <div class="mv-sec-head">
          <span class="mv-sec-title">静态内存布局</span>
          <div class="segmented-control segmented-control-muted mv-plan-layout-switch" role="group" aria-label="静态内存布局表达">
            <button class="btn btn-sm${state.plannerLayoutMode === 'list' ? ' is-selected' : ''}" type="button" data-plan-layout-mode="list">列表布局</button>
            <button class="btn btn-sm${state.plannerLayoutMode === 'architecture' ? ' is-selected' : ''}" type="button" data-plan-layout-mode="architecture">硬件架构</button>
          </div>
        </div>
        ${state.plannerLayoutMode === 'architecture' ? plannerArchitectureMarkup() : plannerListMarkup(plan)}
      </section>
      <section class="mv-sec">
        <div class="mv-sec-head"><span class="mv-sec-title">解析证据</span><span class="mv-label">SOURCE</span></div>
        <div class="mv-kv"><span>InitBuffer</span><b>${plan.buffers.length}</b></div>
        <div class="mv-kv"><span>TQue / TBuf</span><b>${plan.queueCount} / ${plan.tbufCount}</b></div>
        <div class="mv-kv"><span>AllocTensor</span><b>${plan.allocTensorCount}</b></div>
        <div class="mv-kv"><span>TilingData</span><b>tileM=${run.tiling.tileM}, tileNum=${run.tiling.tileNum}</b></div>
        <p class="mv-plan-source-note">容量来自 ${F.escapeHtml(chip.specRef)}；当前 demo 为占位规格，接入工程后替换为目标 SoC 官方规格。</p>
      </section>
    `;
    $('sourcePlannerBody').querySelectorAll('[data-plan-run]').forEach((button) => {
      button.addEventListener('click', () => {
        if (button.dataset.planRun === state.runId) return;
        selectRun(button.dataset.planRun);
        render();
        window.requestAnimationFrame(redrawViews);
      });
    });
    $('sourcePlannerBody').querySelectorAll('[data-plan-layout-mode]').forEach((button) => {
      button.addEventListener('click', () => {
        if (button.dataset.planLayoutMode === state.plannerLayoutMode) return;
        state.plannerLayoutMode = button.dataset.planLayoutMode;
        renderBufferPlanner();
      });
    });
    if (state.plannerLayoutMode === 'architecture') mountPlannerArchitecture(plan);
  }

  function renderWorkspaceSurface() {
    const sourceMode = state.explorerView === 'files';
    $('sourceTabstrip').hidden = !sourceMode;
    $('sourceEditor').hidden = !sourceMode;
    $('analysisTabstrip').hidden = sourceMode;
    $('analysisToolbarHeader').hidden = sourceMode;
    $('previewStage').hidden = sourceMode;
    document.querySelector('[data-ide-toggle="terminal"]').hidden = sourceMode;
    $('railTerminalToggle').hidden = sourceMode;
    $('statusStrip').parentElement.hidden = sourceMode;
    $('sourcePlannerBody').hidden = !sourceMode;
    $('analysisInspectorBody').hidden = sourceMode;
    $('inspectorTitle').textContent = sourceMode ? 'Buffer 规划' : '诊断与详情';
    $('inspectorMeta').textContent = sourceMode ? 'STATIC · 未编译' : `${findings.length} 条`;
    if (sourceMode) {
      renderSourceEditor();
      renderBufferPlanner();
    } else {
      plannerArchController?.destroy?.();
      plannerArchController = null;
    }
  }

  function renderFileTree() {
    const host = $('explorerBody');
    host.innerHTML = '';
    $('explorerTitle').textContent = 'Ascend C 工程';
    $('explorerMeta').textContent = 'WORKSPACE';

    const tree = document.createElement('div');
    tree.className = 'mv-tree';
    tree.setAttribute('role', 'tree');

    const appendNodes = (nodes, depth) => {
      nodes.forEach((node) => {
        const folder = node.type === 'folder';
        const expanded = folder && state.expandedFolders.has(node.id);
        const row = document.createElement('button');
        row.type = 'button';
        row.className = `mv-tree-row${node.id === 'root' ? ' is-root' : ''}${!folder && state.selectedFile === node.id ? ' is-selected' : ''}`;
        row.style.setProperty('--mv-tree-depth', String(depth));
        row.setAttribute('role', 'treeitem');
        row.dataset.nodeId = node.id;
        if (folder) row.setAttribute('aria-expanded', String(expanded));
        row.innerHTML = `${folder
          ? '<svg class="mv-tree-chevron" viewBox="0 0 16 16" aria-hidden="true"><path d="m4 6 4 4 4-4" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></svg>'
          : '<span class="mv-tree-spacer"></span>'}
          ${treeIcon(node.type)}<span class="mv-tree-name">${F.escapeHtml(node.label)}</span>${node.source ? '<span class="mv-tree-meta">KERNEL</span>' : ''}`;
        row.addEventListener('click', () => {
          if (folder) {
            if (expanded) state.expandedFolders.delete(node.id);
            else state.expandedFolders.add(node.id);
          } else {
            state.selectedFile = node.id;
            state.selectedAllocId = null;
            state.selectedEventId = null;
            state.selectedFindingId = null;
          }
          renderFileTree();
          if (!folder && node.source) {
            renderSourceEditor();
            renderBufferPlanner();
            if ($('inspectorPane').hidden) $('topInspectorToggle').click();
          }
        });
        tree.appendChild(row);
        if (folder && expanded) appendNodes(node.children || [], depth + 1);
      });
    };

    appendNodes(ASCEND_C_PROJECT, 0);
    host.appendChild(tree);
  }

  function renderTilingCandidates() {
    const host = $('explorerBody');
    host.innerHTML = '';
    $('explorerTitle').textContent = '算子与 Tiling';
    $('explorerMeta').textContent = chip.name;

    const info = document.createElement('div');
    info.className = 'mv-kernel-block';
    info.innerHTML = `
      <div class="mv-kernel-name">${F.escapeHtml(run.kernel.name)}</div>
      <div class="mv-chip-row">
        <span class="stat-chip">mix · Cube + Vector</span>
        <span class="stat-chip">block_dim ${run.kernel.blockDim}</span>
      </div>
      <div class="mv-kernel-facts">
        <div class="mv-kv"><span>源文件</span><b>${F.escapeHtml(run.kernel.source)}</b></div>
        <div class="mv-kv"><span>形状 M×N×K</span><b>${run.kernel.shape.M}×${run.kernel.shape.N}×${run.kernel.shape.K}</b></div>
        <div class="mv-kv"><span>L0 分形块 K0</span><b>${run.kernel.shape.K0}</b></div>
        <div class="mv-kv"><span>规格来源</span><b>${F.escapeHtml(chip.specRef)}</b></div>
      </div>
    `;
    host.appendChild(info);

    const group = document.createElement('div');
    group.className = 'mv-group';
    group.innerHTML = `
      <div class="mv-sec-head">
        <span class="mv-sec-title">Tiling 候选</span>
        <span class="mv-label">${runs.length} 组</span>
      </div>
    `;
    host.appendChild(group);

    const list = document.createElement('div');
    list.className = 'mv-run-list';
    runs.forEach((item) => {
      const entry = runIndex.get(item.id);
      const ub = entry.metrics.regionById[state.focusRegionId] || entry.metrics.regions[0];
      const over = ub.reserved > ub.capacity;
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `mv-run${item.id === state.runId ? ' is-selected' : ''}`;
      btn.innerHTML = `
        <div class="mv-run-head">
          <span class="mv-run-title">${F.escapeHtml(item.label)}</span>
          <span class="stat-chip">${F.escapeHtml(item.kicker)}</span>
        </div>
        <div class="mv-run-note">${F.escapeHtml(item.note)}</div>
        <div class="mv-bar"><i style="width:${Math.min(100, ub.reservedRatio * 100).toFixed(1)}%;background:${over ? 'var(--danger)' : ub.accent}"></i></div>
        <div class="mv-run-metrics">
          <span class="mv-metric">
            <span class="mv-metric-value ${over ? 'mv-sev-danger' : ''}">${F.pct(ub.reservedRatio, 0)}</span>
            <span class="mv-metric-label">${ub.id} 占用</span>
          </span>
          <span class="mv-metric">
            <span class="mv-metric-value">${entry.run.totalTicks}</span>
            <span class="mv-metric-label">总 cycle</span>
          </span>
          <span class="mv-metric">
            <span class="mv-metric-value">${entry.metrics.totals.moveCount}</span>
            <span class="mv-metric-label">搬运次数</span>
          </span>
          <span class="mv-metric">
            <span class="mv-metric-value">
              ${entry.summary.danger ? `<span class="mv-sev-danger">${entry.summary.danger}</span>` : ''}${entry.summary.danger && entry.summary.warn ? ' / ' : ''}${entry.summary.warn ? `<span class="mv-sev-warn">${entry.summary.warn}</span>` : ''}${!entry.summary.danger && !entry.summary.warn ? '<span class="mv-sev-success">✓</span>' : ''}
            </span>
            <span class="mv-metric-label">危险 / 警告</span>
          </span>
        </div>
      `;
      btn.addEventListener('click', () => {
        if (state.runId === item.id) return;
        selectRun(item.id);
        render();
      });
      list.appendChild(btn);
    });
    group.appendChild(list);
  }

  function renderExplorer() {
    if (state.explorerView === 'tiling') renderTilingCandidates();
    else renderFileTree();
  }

  // ---------------------------------------------------------------
  // 右栏：诊断与详情
  // ---------------------------------------------------------------
  const SEVERITY_LABEL = { danger: '危险', warn: '警告', info: '提示' };

  function renderFindings() {
    const summary = global.MemVizDiagnostics.summarize(findings);
    $('severitySummary').innerHTML = ['danger', 'warn', 'info']
      .filter((key) => summary[key] > 0)
      .map((key) => `<span class="stat-chip mv-sev-${key === 'warn' ? 'warn' : key}">${SEVERITY_LABEL[key]} ${summary[key]}</span>`)
      .join('') || '<span class="stat-chip mv-sev-success">无问题</span>';
    $('inspectorMeta').textContent = `${findings.length} 条`;

    const host = $('findingList');
    host.innerHTML = '';
    if (!findings.length) {
      host.innerHTML = '<p class="mv-empty">当前配置未触发任何规则。</p>';
      return;
    }
    findings.forEach((item) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `mv-finding${item.id === state.selectedFindingId ? ' is-selected' : ''}`;
      btn.dataset.severity = item.severity;
      btn.innerHTML = `
        <div class="mv-finding-head">
          <span class="mv-finding-title">${F.escapeHtml(item.title)}${item.region ? ` · ${item.region}` : ''}</span>
          <span class="mv-finding-rule">${item.rule}</span>
        </div>
        <div class="mv-finding-text">${F.escapeHtml(item.detail)}</div>
        <div class="mv-finding-body">
          <div class="mv-soft ${item.severity === 'danger' ? 'is-danger' : item.severity === 'warn' ? 'is-warning' : ''}">
            <span class="mv-soft-label">量化影响</span>
            <span class="mv-soft-text">${F.escapeHtml(item.impact)}</span>
          </div>
          <div class="mv-soft">
            <span class="mv-soft-label">建议动作</span>
            <span class="mv-soft-text">${F.escapeHtml(item.suggest)}</span>
          </div>
          <div class="mv-evidence">
            <span class="mv-soft-label">溯源</span>
            ${item.evidence.map((e) => `<span class="mv-kv"><span>${F.escapeHtml(e.label)}</span><b>${F.escapeHtml(e.value)}</b></span>`).join('')}
          </div>
        </div>
      `;
      btn.addEventListener('click', () => {
        state.selectedFindingId = state.selectedFindingId === item.id ? null : item.id;
        if (state.selectedFindingId) {
          if (item.region) state.focusRegionId = item.region;
          const firstEvent = item.eventRefs.length
            ? run.events.find((e) => e.id === item.eventRefs[0]) : null;
          if (firstEvent) state.tick = firstEvent.t;
          if (item.refs.length) state.selectedAllocId = item.refs[0];
          renderRegionSwitch();
        }
        render();
      });
      host.appendChild(btn);
    });
  }

  function currentFinding() {
    return findings.find((f) => f.id === state.selectedFindingId) || null;
  }

  function highlightSets() {
    const finding = currentFinding();
    return {
      highlightIds: new Set(finding ? finding.refs : []),
      highlightEventIds: new Set(finding ? finding.eventRefs : []),
      conflictIds: new Set(findings
        .filter((f) => f.rule === 'ADDR_CONFLICT')
        .flatMap((f) => f.refs)),
    };
  }

  function renderDetail() {
    const host = $('detailBody');
    const kicker = $('detailKicker');
    const alloc = run.allocations.find((a) => a.id === state.selectedAllocId);
    const event = run.events.find((e) => e.id === state.selectedEventId);

    if (alloc) {
      const kindLabel = alloc.isRegister ? '寄存器'
        : alloc.isSpill ? '溢出区'
          : alloc.kind === 'queue' ? 'TQue'
            : alloc.kind === 'gm' ? 'GlobalTensor' : 'TBuf';
      kicker.textContent = `${alloc.region} · ${kindLabel}`;
      const span = global.MemVizMetrics.liveSpan(alloc);
      const related = findings.filter((f) => f.refs.includes(alloc.id));
      const regBytes = metrics.regionById[alloc.region]?.regBytes || 1;
      const location = alloc.isRegister
        ? `v${alloc.regIndex} – v${alloc.regIndex + Math.round(alloc.size / regBytes) - 1}`
        : `${F.hex(alloc.offset)} – ${F.hex(alloc.offset + alloc.size)}`;
      host.innerHTML = `
        <div class="mv-sec" style="padding:0;border:0;gap:var(--space-2)">
          <div class="mv-detail-title">${F.escapeHtml(alloc.name)}</div>
          <div class="mv-chip-row">
            <span class="stat-chip">${F.escapeHtml(alloc.queue)}</span>
            <span class="stat-chip">buffer_num ${alloc.bufferNum}</span>
            ${alloc.manualReuse ? '<span class="stat-chip mv-sev-danger">手工复用</span>' : ''}
          </div>
          <div class="mv-kv"><span>${alloc.isRegister ? '寄存器区间' : '地址区间'}</span><b>${location}</b></div>
          <div class="mv-kv"><span>实占 / 数据</span><b>${F.bytes(alloc.size)} / ${F.bytes(alloc.dataBytes)}</b></div>
          <div class="mv-kv"><span>对齐粒度</span><b>${alloc.align}B</b></div>
          ${alloc.regsPerThread ? `<div class="mv-kv"><span>每线程寄存器</span><b>${alloc.regsPerThread}</b></div>` : ''}
          ${alloc.note ? `<div class="mv-kv"><span>说明</span><b>${F.escapeHtml(alloc.note)}</b></div>` : ''}
          <div class="mv-kv"><span>dtype · shape</span><b>${F.escapeHtml(alloc.dtype)} ${F.shape(alloc.shape)}</b></div>
          <div class="mv-kv"><span>生命周期</span><b>${span ? `${F.tick(span.start)} – ${F.tick(span.end)}` : '未被访问'}</b></div>
          <div class="mv-kv"><span>活跃区间数</span><b>${alloc.intervals.length}</b></div>
          ${alloc.padReason ? `<div class="mv-soft is-warning"><span class="mv-soft-label">对齐说明</span><span class="mv-soft-text">${F.escapeHtml(alloc.padReason)}</span></div>` : ''}
          ${related.length ? `<div class="mv-soft ${related[0].severity === 'danger' ? 'is-danger' : 'is-warning'}"><span class="mv-soft-label">关联诊断</span><span class="mv-soft-text">${related.map((f) => F.escapeHtml(f.title)).join('、')}</span></div>` : ''}
          <span class="mv-soft-label">源码 ${F.escapeHtml(alloc.src.file)}:${alloc.src.hotLine + 1}</span>
          <pre class="mv-code">${F.escapeHtml(alloc.code)}</pre>
          <span class="mv-soft-label">CCE</span>
          <pre class="mv-code">${F.escapeHtml(alloc.cce)}</pre>
        </div>
      `;
      return;
    }

    if (event) {
      kicker.textContent = `${event.pipe} · 事件`;
      const names = (ids) => ids.map((id) => run.allocations.find((a) => a.id === id)?.name).filter(Boolean).join('、') || '—';
      host.innerHTML = `
        <div class="mv-sec" style="padding:0;border:0;gap:var(--space-2)">
          <div class="mv-detail-title">${F.escapeHtml(event.label)}</div>
          <div class="mv-chip-row">
            <span class="stat-chip">${event.pipe}</span>
            <span class="stat-chip">${event.type}</span>
            ${event.iter >= 0 ? `<span class="stat-chip">迭代 #${event.iter}</span>` : '<span class="stat-chip">前导</span>'}
          </div>
          <div class="mv-kv"><span>时间区间</span><b>${F.tick(event.t)} – ${F.tick(event.end)}（${event.dur} cycle）</b></div>
          ${event.bytes ? `<div class="mv-kv"><span>搬运量</span><b>${F.bytes(event.bytes)}</b></div>` : ''}
          <div class="mv-kv"><span>写</span><b>${names(event.writes)}</b></div>
          <div class="mv-kv"><span>读</span><b>${names(event.reads)}</b></div>
          ${event.gap > 0 && event.blockedBy
            ? `<div class="mv-soft is-warning"><span class="mv-soft-label">等待 ${event.gap} cycle</span><span class="mv-soft-text">等 ${F.escapeHtml(run.allocations.find((a) => a.id === event.blockedBy)?.name || '')} 的 slot 释放</span></div>`
            : ''}
          ${event.tailRows != null ? `<div class="mv-soft is-warning"><span class="mv-soft-label">尾块</span><span class="mv-soft-text">本次只有 ${event.tailRows} / ${run.tiling.tileM} 行有效</span></div>` : ''}
          <span class="mv-soft-label">源码 ${F.escapeHtml(run.kernel.source)}:${(event.srcLine || 0) + 1}</span>
          <pre class="mv-code">${F.escapeHtml(global.MemVizKernelSource.snippet(Math.max(0, (event.srcLine || 0) - 1), (event.srcLine || 0) + 1))}</pre>
        </div>
      `;
      return;
    }

    kicker.textContent = '未选中';
    host.innerHTML = '<p class="mv-empty">点击布局图色块、泳道任务或诊断条目查看详情。</p>';
  }

  // ---------------------------------------------------------------
  // 底部：分析日志（CLI 形态，服务规划文档 §4.6 的 CI 集成）
  // ---------------------------------------------------------------
  function renderTerminal() {
    const summary = global.MemVizDiagnostics.summarize(findings);
    const exitCode = summary.danger ? 2 : summary.warn ? 1 : 0;
    const lines = [
      `<span class="is-dim">$</span> memviz analyze ${run.kernel.source} --chip ${chip.id} --tiling tileM=${run.tiling.tileM} --fail-on danger`,
      `<span class="is-dim">loaded  </span> chip spec ${chip.specRef} <span class="is-dim">(占位规格)</span>`,
      `<span class="is-dim">parsed  </span> ${run.allocations.length} allocations · ${run.events.length} events · ${run.totalTicks} cycles`,
      '',
      ...metrics.regions.map((region) => {
        const over = region.reserved > region.capacity;
        const cls = over ? 'is-danger' : region.reservedRatio >= 0.9 ? 'is-warn' : 'is-ok';
        const used = region.isRegister ? `${region.reservedRegs} regs` : F.bytes(region.reserved);
        const cap = region.isRegister ? `${region.capacityRegs} regs` : F.bytes(region.capacity);
        const tag = region.isRegister ? 'regfile ' : 'region  ';
        return `<span class="is-dim">${tag}</span> ${region.id.padEnd(4)} ${String(used).padStart(9)} / ${String(cap).padStart(9)}  <span class="${cls}">${F.pct(region.reservedRatio, 1).padStart(7)}</span>`;
      }),
      '',
      ...findings.map((item) => {
        const cls = item.severity === 'danger' ? 'is-danger' : item.severity === 'warn' ? 'is-warn' : 'is-dim';
        return `<span class="${cls}">${item.severity.toUpperCase().padEnd(7)}</span> ${item.rule.padEnd(26)} ${F.escapeHtml(item.impact)}`;
      }),
      '',
      `<span class="is-dim">summary </span> ${summary.danger} danger · ${summary.warn} warn · ${summary.info} info`,
      `<span class="${exitCode === 2 ? 'is-danger' : exitCode === 1 ? 'is-warn' : 'is-ok'}">exit ${exitCode}</span>`,
    ];
    $('terminalBody').innerHTML = lines.join('\n');
  }

  // ---------------------------------------------------------------
  // 状态条
  // ---------------------------------------------------------------
  function renderStatus() {
    const region = metrics.regionById[state.focusRegionId];
    const summary = global.MemVizDiagnostics.summarize(findings);
    const plan = run.registers;
    const regItem = plan ? `
      <span class="mv-status-item">寄存器 <b class="${plan.spillRegs ? 'mv-sev-danger' : ''}">${plan.requestedRegs} / ${plan.capacityRegs}</b>${plan.spillRegs ? `（溢出 ${plan.spillRegs}）` : ''}</span>
      <span class="mv-status-item">warp <b class="${plan.activeWarps < plan.warpsMax ? 'mv-sev-warn' : ''}">${plan.activeWarps} / ${plan.warpsMax}</b></span>
    ` : '';
    $('statusStrip').innerHTML = `
      <span class="mv-status-item">芯片 <b>${F.escapeHtml(chip.name)}</b></span>
      <span class="mv-status-item">候选 <b>${F.escapeHtml(run.label)}</b></span>
      <span class="mv-status-item">tileM <b>${run.tiling.tileM}</b> × <b>${run.tiling.tileNum}</b>${run.tiling.hasTail ? `（尾块 ${run.tiling.tailM}）` : ''}</span>
      <span class="mv-status-item">${region.id} <b class="${region.reserved > region.capacity ? 'mv-sev-danger' : ''}">${region.isRegister
        ? `${region.reservedRegs} / ${region.capacityRegs} regs`
        : `${F.bytes(region.reserved)} / ${F.bytes(region.capacity)}`}</b></span>
      <span class="mv-status-item">峰值持有 <b>${F.bytes(region.peakLive)} @ ${F.tick(region.peakTick)}</b></span>
      ${regItem}
      <span class="mv-status-item">游标 <b>${F.tick(state.tick)} / ${F.tick(run.totalTicks)}</b></span>
      <span class="mv-status-spacer"></span>
      <span class="mv-status-item">诊断 <b class="${summary.danger ? 'mv-sev-danger' : summary.warn ? 'mv-sev-warn' : 'mv-sev-success'}">${summary.danger}D / ${summary.warn}W / ${summary.info}I</b></span>
      <span class="mv-status-item">数据等级 <b>L2 · 构造样例</b></span>
    `;
  }

  // ---------------------------------------------------------------
  // 视图
  // ---------------------------------------------------------------
  function renderViews() {
    ['layout', 'lifetime', 'pipeline'].forEach((id) => {
      const el = $(`view${id[0].toUpperCase()}${id.slice(1)}`);
      el.classList.toggle('is-active', state.view === id);
    });
    const marks = highlightSets();

    // 布局切换只影响「内存布局」页签内部：两种布局共用同一份 metrics 与时间游标
    const archMode = state.view === 'layout' && state.layoutMode === 'arch';
    const layoutPane = $('viewLayout');
    layoutPane.classList.toggle('is-arch', archMode);
    $('layoutModeSwitch').hidden = state.view !== 'layout';
    $('layoutModeLabel').hidden = state.view !== 'layout';
    const mode = LAYOUT_MODES.find((item) => item.id === state.layoutMode) || LAYOUT_MODES[0];
    if (state.view === 'layout') $('layoutNote').textContent = mode.note;

    if (state.view === 'layout' && archMode) {
      views.arch.update({
        run, metrics, chip, tick: state.tick,
        selectedId: state.selectedAllocId,
        focusRegionId: state.focusRegionId,
      });
    } else if (state.view === 'layout') {
      views.layout.update({
        run, metrics, chip, tick: state.tick,
        selectedId: state.selectedAllocId,
        highlightIds: marks.highlightIds,
        conflictIds: marks.conflictIds,
      });
    } else if (state.view === 'lifetime') {
      views.lifetime.update({ run, chip, selectedId: state.selectedAllocId });
    } else {
      views.pipeline.update({
        run, metrics, tick: state.tick,
        focusRegionId: state.focusRegionId,
        highlightEventIds: marks.highlightEventIds,
        selectedEventId: state.selectedEventId,
      });
    }
    views.watermark.update({ run, metrics, tick: state.tick, focusRegionId: state.focusRegionId });
  }

  function render() {
    document.querySelectorAll('#viewTabs .tab-control-item').forEach((tab) => {
      const active = tab.dataset.view === state.view;
      tab.classList.toggle('is-selected', active);
      tab.setAttribute('aria-selected', String(active));
    });
    renderLegend();
    renderExplorer();
    renderFindings();
    renderWorkspaceSurface();
    renderDetail();
    renderTerminal();
    renderStatus();
    renderViews();
  }

  function redrawViews() {
    views.layout?.redraw?.();
    if (state.view === 'layout' && state.layoutMode === 'arch') views.arch?.redraw?.();
    views.pipeline?.redraw?.();
    views.watermark?.redraw?.();
    views.lifetime?.redraw?.();
  }

  // ---------------------------------------------------------------
  // 时间游标
  // ---------------------------------------------------------------
  function setTick(next) {
    const clamped = Math.max(0, Math.min(run.totalTicks, Math.round(next)));
    if (clamped === state.tick) return;
    state.tick = clamped;
    renderStatus();
    renderViews();
  }

  // ---------------------------------------------------------------
  // 面板开关（顶栏按钮镜像 activity rail 的行为）
  // ---------------------------------------------------------------
  function bindPanelToggles() {
    const explorerCollapseControl = $('explorerCollapseControl');
    const railExplorer = $('railFilesToggle');
    const railTiling = $('railTilingToggle');
    const explorerPane = $('explorerPane');
    const topExplorer = $('topExplorerToggle');
    const inspectorPane = $('inspectorPane');
    const topInspector = $('topInspectorToggle');
    const bottomDock = $('bottomDock');

    const setInspectorVisible = (visible) => {
      inspectorPane.hidden = !visible;
      inspectorPane.setAttribute('aria-hidden', String(!visible));
      const gutter = inspectorPane.previousElementSibling;
      if (gutter?.classList.contains('pto-workbench-shell__split-gutter')) gutter.hidden = !visible;
      topInspector.classList.toggle('is-selected', visible);
      topInspector.setAttribute('aria-pressed', String(visible));
      topInspector.setAttribute('aria-expanded', String(visible));
      frameController?.refresh?.();
      window.requestAnimationFrame(redrawViews);
    };

    const setBottomDockVisible = (visible) => {
      bottomDock.hidden = !visible;
      bottomDock.setAttribute('aria-hidden', String(!visible));
      const gutter = bottomDock.previousElementSibling;
      if (gutter?.classList.contains('pto-workbench-shell__split-gutter')) gutter.hidden = !visible;
      frameController?.refresh?.();
      window.requestAnimationFrame(redrawViews);
    };
    const syncExplorerNavigation = () => {
      const expanded = !explorerPane.hidden;
      railExplorer.classList.toggle('is-selected', expanded && state.explorerView === 'files');
      railExplorer.setAttribute('aria-pressed', String(expanded && state.explorerView === 'files'));
      railExplorer.setAttribute('aria-expanded', String(expanded));
      railTiling.classList.toggle('is-selected', expanded && state.explorerView === 'tiling');
      railTiling.setAttribute('aria-pressed', String(expanded && state.explorerView === 'tiling'));
      railTiling.setAttribute('aria-expanded', String(expanded));
      topExplorer.classList.toggle('is-selected', expanded);
      topExplorer.setAttribute('aria-pressed', String(expanded));
      topExplorer.setAttribute('aria-expanded', String(expanded));
    };

    railExplorer.addEventListener('click', () => {
      const collapseCurrent = state.explorerView === 'files' && !explorerPane.hidden;
      if (collapseCurrent || explorerPane.hidden) explorerCollapseControl.click();
      state.explorerView = 'files';
      state.selectedFile = 'kernel_cpp';
      state.selectedAllocId = null;
      state.selectedEventId = null;
      state.selectedFindingId = null;
      render();
      setInspectorVisible(true);
      setBottomDockVisible(false);
      syncExplorerNavigation();
      window.requestAnimationFrame(redrawViews);
    });

    railTiling.addEventListener('click', () => {
      const collapseCurrent = state.explorerView === 'tiling' && !explorerPane.hidden;
      if (collapseCurrent || explorerPane.hidden) explorerCollapseControl.click();
      state.explorerView = 'tiling';
      state.selectedAllocId = null;
      state.selectedEventId = null;
      state.selectedFindingId = null;
      render();
      setInspectorVisible(true);
      setBottomDockVisible(true);
      syncExplorerNavigation();
      window.requestAnimationFrame(redrawViews);
    });

    topExplorer.addEventListener('click', () => {
      explorerCollapseControl.click();
      syncExplorerNavigation();
      window.requestAnimationFrame(redrawViews);
    });

    const toggleInspector = () => {
      setInspectorVisible(inspectorPane.hidden);
    };
    topInspector.addEventListener('click', toggleInspector);
    $('railDiagnosticsToggle').addEventListener('click', () => {
      if (state.explorerView !== 'tiling') $('railTilingToggle').click();
      if (inspectorPane.hidden) toggleInspector();
      $('findingList').firstElementChild?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    });

    const railTerminal = $('railTerminalToggle');
    railTerminal.addEventListener('click', () => {
      document.querySelector('[data-ide-toggle="terminal"]')?.click();
      window.requestAnimationFrame(redrawViews);
    });

    setInspectorVisible(true);
    setBottomDockVisible(false);
  }

  function bindTabs() {
    document.querySelectorAll('#viewTabs .tab-control-item').forEach((tab) => {
      tab.addEventListener('click', () => {
        state.view = tab.dataset.view;
        render();
        window.requestAnimationFrame(redrawViews);
      });
    });
  }

  function bindToolbar() {
    $('gotoPeak').addEventListener('click', () => {
      const region = metrics.regionById[state.focusRegionId];
      setTick(region.peakTick);
    });
    $('gotoIssue').addEventListener('click', () => {
      if (state.explorerView !== 'tiling') $('railTilingToggle').click();
      const first = findings.find((f) => f.severity === 'danger') || findings[0];
      if (!first) return;
      state.selectedFindingId = first.id;
      if (first.region) { state.focusRegionId = first.region; renderRegionSwitch(); }
      if (first.refs.length) state.selectedAllocId = first.refs[0];
      const ev = first.eventRefs.length ? run.events.find((e) => e.id === first.eventRefs[0]) : null;
      if (ev) state.tick = ev.t;
      render();
    });
  }

  function bindKeyboard() {
    window.addEventListener('keydown', (event) => {
      if (event.target.matches('input, textarea')) return;
      if (event.key === 'ArrowRight') { setTick(state.tick + (event.shiftKey ? 10 : 1)); event.preventDefault(); }
      if (event.key === 'ArrowLeft') { setTick(state.tick - (event.shiftKey ? 10 : 1)); event.preventDefault(); }
    });
  }

  // ---------------------------------------------------------------
  function init() {
    loadChip(state.chipId);
    selectRun(state.runId);

    views.layout = global.MemVizLayoutView.create($('layoutHost'), {
      onSelect: (alloc) => {
        if (state.explorerView !== 'tiling') return;
        state.selectedAllocId = alloc ? alloc.id : null;
        state.selectedEventId = null;
        renderDetail();
        renderViews();
      },
    });
    views.arch = global.MemVizArchView.create($('archHost'), {
      onFocusRegion: (regionId) => {
        state.focusRegionId = regionId;
        renderRegionSwitch();
        render();
      },
    });
    views.lifetime = global.MemVizLifetimeView.create($('viewLifetime'));
    views.pipeline = global.MemVizPipelineView.create($('pipelineHost'), {
      onSelectEvent: (event) => {
        if (state.explorerView !== 'tiling') return;
        state.selectedEventId = event.id;
        state.selectedAllocId = event.writes[0] || event.reads[0] || null;
        renderDetail();
        renderViews();
      },
      onSeek: setTick,
    });
    views.watermark = global.MemVizWatermarkView.create($('watermarkHost'), { onSeek: setTick });

    frameController = global.PtoIdeFrame?.init($('ideFrame'), {
      splitOptions: { default: { onResize: () => window.requestAnimationFrame(redrawViews) } },
    });

    renderChipSwitch();
    renderLayoutModeSwitch();
    renderRegionSwitch();
    bindPanelToggles();
    bindTabs();
    bindToolbar();
    bindKeyboard();
    render();
    redrawViews();
    // 后台标签页 / 隐藏 iframe（例如 launch.html 的预览卡）里，首帧渲染尚未发生，
    // clientWidth 会是 0 且 ResizeObserver 不投递，画布会停在零宽度。
    // 用一次宏任务和 load 事件各补一遍，再在页面转为可见时补最后一次。
    setTimeout(redrawViews, 0);
    window.addEventListener('load', redrawViews, { once: true });
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden) redrawViews();
    });
  }

  // 同步启动：ide-frame 的 pattern.js 会在 DOMContentLoaded 上自动 initAll()，
  // 本页必须先拿到 frame 才能注入 splitOptions.onResize。脚本位于 body 末尾，DOM 已就绪。
  init();
})(window);
