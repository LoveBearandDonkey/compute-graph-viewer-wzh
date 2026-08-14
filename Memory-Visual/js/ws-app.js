/*
  应用控制器 —— 融合与 workspace（场景 6）
  ------------------------------------------------------------------
  与 js/app.js 同构：单一状态源 + 单向渲染，所有交互只改 state 再走一次 render()。
  唯一的坐标是**子计算序**（不是 cycle）—— 主视图的柱子、GM 布局的时间轴、
  播放条游标共用它。

  三个数（current / packed / lowerBound）一律来自 MemVizWorkspacePlanner.plan()，
  视图与规则都不自己重算，避免同一个数在两处对不上。
*/
(function bootMemVizWorkspace(global) {
  'use strict';

  const F = global.MemVizFormat;
  const PLANNER = global.MemVizWorkspacePlanner;
  const DIAG = global.MemVizWorkspaceDiagnostics;
  const $ = (id) => document.getElementById(id);

  const state = {
    chipId: 'ascend-910b',
    runId: 'ws-naive',
    view: 'plan',
    layoutMode: 'current', // current = 候选自己的布局；packed = 规划器排出的最紧布局
    sg: 0,
    playing: false,
    selectedTensorId: null,
    selectedFindingId: null,
    selectedGroupIndex: null,
  };

  let chip = null;
  let runs = [];
  let run = null;
  let plan = null;
  let findings = [];
  let summary = null;
  let runIndex = new Map();
  let views = {};
  let frameController = null;
  let playback = null;
  let playTimer = null;

  // ---------------------------------------------------------------
  // 数据装载
  // ---------------------------------------------------------------
  function loadChip(chipId) {
    chip = global.MemVizChips.get(chipId);
    // 候选本身与芯片无关（GM 侧的形状与预算由算子决定）；
    // 芯片只影响「能不能留在片上」这类判据，所以规则引擎才吃 chip。
    runs = global.MemVizFusionRuns.buildAll();
    runIndex = new Map(runs.map((item) => {
      const p = PLANNER.plan(item);
      const f = DIAG.analyze(item, p, chip);
      return [item.id, { run: item, plan: p, findings: f, summary: DIAG.summarize(f) }];
    }));
    if (!runIndex.has(state.runId)) state.runId = runs[0].id;
  }

  function selectRun(runId) {
    state.runId = runId;
    const entry = runIndex.get(runId);
    run = entry.run;
    plan = entry.plan;
    findings = entry.findings;
    summary = entry.summary;
    state.sg = Math.min(state.sg, run.subgraphs.length - 1);
    state.selectedTensorId = null;
    state.selectedFindingId = null;
    state.selectedGroupIndex = null;
    syncPlaybackRange();
  }

  function tensorById(id) {
    return run.tensors.find((t) => t.id === id) || null;
  }

  function severityClass(severity) {
    if (severity === 'danger') return 'mv-sev-danger';
    if (severity === 'warn') return 'mv-sev-warn';
    return 'mv-sev-info';
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
      btn.addEventListener('click', () => {
        if (item.id === state.chipId) return;
        state.chipId = item.id;
        loadChip(item.id);
        selectRun(state.runId);
        renderChipSwitch();
        render();
      });
      host.appendChild(btn);
    });
  }

  function renderLayoutModeSwitch() {
    const label = $('layoutModeLabel');
    const host = $('layoutModeSwitch');
    const visible = state.view === 'gm';
    label.hidden = !visible;
    host.hidden = !visible;
    if (!visible) return;
    host.innerHTML = '';
    [
      { id: 'current', label: '当前布局', title: '候选自己声明的地址分配' },
      { id: 'packed', label: '复用后布局', title: `规划器按 ${plan.bestOrder} 排出的最紧布局` },
    ].forEach((mode) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `btn btn-sm${mode.id === state.layoutMode ? ' is-selected' : ''}`;
      btn.textContent = mode.label;
      btn.title = mode.title;
      btn.addEventListener('click', () => {
        state.layoutMode = mode.id;
        renderLayoutModeSwitch();
        renderViews();
      });
      host.appendChild(btn);
    });
  }

  function renderLegend() {
    const host = $('viewLegend');
    const items = state.view === 'plan'
      ? [
        ['line is-current', '当前预留'],
        ['line is-packed', '复用后可达'],
        ['line is-bound', '理论下界'],
        ['line is-budget', 'GM 预算'],
        ['swatch is-waste', '策略浪费'],
      ]
      : [
        ['swatch', '实心 = 张量占用'],
        ['line is-packed', '复用连线'],
      ];
    host.innerHTML = items.map(([kind, text]) =>
      `<span class="mv-legend-item"><span class="mv-legend-${kind}"></span>${F.escapeHtml(text)}</span>`).join('');
  }

  // ---------------------------------------------------------------
  // 左栏
  // ---------------------------------------------------------------
  function renderExplorer() {
    const body = $('explorerBody');
    body.innerHTML = '';
    const shape = run.kernel.shape;

    const block = document.createElement('div');
    block.className = 'mv-kernel-block';
    block.innerHTML = `
      <span class="mv-kernel-name">${F.escapeHtml(run.kernel.name)}</span>
      <div class="mv-kernel-facts">
        <span class="mv-kv"><span>子计算</span><b>${run.subgraphs.length} 个</b></span>
        <span class="mv-kv"><span>GM 张量</span><b>${plan.tensors.length} workspace</b></span>
        <span class="mv-kv"><span>tokens × hidden</span><b>${shape.tokens} × ${shape.hidden}</b></span>
        <span class="mv-kv"><span>FFN 中间维</span><b>${shape.ffnInner}</b></span>
        <span class="mv-kv"><span>block_dim</span><b>${run.kernel.blockDim}</b></span>
        <span class="mv-kv"><span>GM 预算</span><b>${F.bytes(plan.budget)}</b></span>
        <span class="mv-kv"><span>上报位置</span><b>${F.escapeHtml(run.workspace.reportedAt.file)}:${run.workspace.reportedAt.line + 1}</b></span>
      </div>`;
    body.appendChild(block);

    const group = document.createElement('div');
    group.className = 'mv-group';
    const title = document.createElement('span');
    title.className = 'mv-label';
    title.textContent = 'workspace 候选';
    group.appendChild(title);

    const list = document.createElement('div');
    list.className = 'mv-run-list';
    runs.forEach((item) => {
      const entry = runIndex.get(item.id);
      const p = entry.plan;
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `mv-run${item.id === state.runId ? ' is-selected' : ''}`;

      const boundPct = (p.lowerBound / p.current) * 100;
      const packPct = ((p.packed - p.lowerBound) / p.current) * 100;
      const wastePct = Math.max(0, 100 - boundPct - packPct);
      const counts = [
        entry.summary.danger ? `<span class="mv-sev-danger">${entry.summary.danger} 危</span>` : '',
        entry.summary.warn ? `<span class="mv-sev-warn">${entry.summary.warn} 警</span>` : '',
        entry.summary.info ? `<span class="mv-sev-info">${entry.summary.info} 提示</span>` : '',
      ].filter(Boolean).join(' · ');

      btn.innerHTML = `
        <span class="mv-run-head">
          <span class="mv-run-title">${F.escapeHtml(item.label)}</span>
          <span class="stat-chip">${F.escapeHtml(item.kicker)}</span>
        </span>
        <span class="mv-run-metrics">
          <span class="mv-metric">
            <span class="mv-metric-value ${p.overBudget > 0 ? 'mv-sev-danger' : ''}">${F.bytes(p.current)}</span>
            <span class="mv-metric-label">当前</span>
          </span>
          <span class="mv-metric">
            <span class="mv-metric-value mv-sev-success">${F.bytes(p.lowerBound)}</span>
            <span class="mv-metric-label">下界</span>
          </span>
          <span class="mv-metric">
            <span class="mv-metric-value ${p.ratio > 1.2 ? 'mv-sev-warn' : ''}">${p.ratio.toFixed(2)}×</span>
            <span class="mv-metric-label">比值</span>
          </span>
        </span>
        <span class="mv-bar">
          <i style="width:${boundPct.toFixed(1)}%;background:var(--success)"></i>
          <i style="width:${packPct.toFixed(1)}%;background:var(--foreground-disabled)"></i>
          <i style="width:${wastePct.toFixed(1)}%;background:var(--warning)"></i>
        </span>
        <span class="mv-run-note">${F.escapeHtml(item.note)}</span>
        <span class="mv-run-note">${counts || '<span class="mv-sev-success">无问题</span>'}</span>`;

      btn.addEventListener('click', () => {
        if (item.id === state.runId) return;
        selectRun(item.id);
        render();
        window.requestAnimationFrame(redrawViews);
      });
      list.appendChild(btn);
    });
    group.appendChild(list);
    body.appendChild(group);

    $('explorerMeta').textContent = `${runs.length} 个候选`;
  }

  // ---------------------------------------------------------------
  // 右栏
  // ---------------------------------------------------------------
  function renderFindings() {
    const host = $('findingList');
    host.innerHTML = '';
    $('severitySummary').innerHTML = [
      summary.danger ? `<span class="stat-chip mv-sev-danger">${summary.danger} 危</span>` : '',
      summary.warn ? `<span class="stat-chip mv-sev-warn">${summary.warn} 警</span>` : '',
      summary.info ? `<span class="stat-chip">${summary.info} 提示</span>` : '',
    ].filter(Boolean).join('') || '<span class="stat-chip mv-sev-success">无问题</span>';

    if (!findings.length) {
      host.innerHTML = '<p class="mv-empty">当前候选没有触发任何规则。</p>';
      return;
    }

    findings.forEach((f) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `mv-finding${f.id === state.selectedFindingId ? ' is-selected' : ''}`;
      btn.dataset.severity = f.severity;
      const src = f.srcRef ? `${f.srcRef.file}:${f.srcRef.line + 1}` : '';
      btn.innerHTML = `
        <span class="mv-finding-head">
          <span class="mv-finding-title ${severityClass(f.severity)}">${F.escapeHtml(f.title)}</span>
          <span class="mv-finding-rule">${F.escapeHtml(f.rule)}</span>
        </span>
        <span class="mv-finding-text">${F.escapeHtml(f.detail)}</span>
        <span class="mv-finding-body">
          <span class="mv-soft ${f.severity === 'danger' ? 'is-danger' : f.severity === 'warn' ? 'is-warning' : ''}">
            <span class="mv-soft-label">量化影响</span>
            <span class="mv-soft-text">${F.escapeHtml(f.impact)}</span>
          </span>
          <span class="mv-soft">
            <span class="mv-soft-label">建议动作</span>
            <span class="mv-soft-text">${F.escapeHtml(f.suggest)}</span>
          </span>
          <span class="mv-evidence">
            ${f.evidence.map((e) => `<span class="mv-kv"><span>${F.escapeHtml(e.label)}</span><b>${F.escapeHtml(e.value)}</b></span>`).join('')}
            ${src ? `<span class="mv-kv"><span>溯源</span><b>${F.escapeHtml(src)}</b></span>` : ''}
          </span>
        </span>`;
      btn.addEventListener('click', () => {
        state.selectedFindingId = f.id === state.selectedFindingId ? null : f.id;
        state.selectedGroupIndex = null;
        if (f.refs.length) state.selectedTensorId = f.refs[0];
        if (f.subgraph != null) state.sg = f.subgraph;
        render();
      });
      host.appendChild(btn);
    });
  }

  function renderGroups() {
    const host = $('groupList');
    host.innerHTML = '';
    const usable = plan.groups.filter((g) => g.saving > 0);
    // 已经排满时不能再喊「可省 X」——那部分收益已经拿到了，
    // 此时这些组合只是「当前布局为什么这么紧」的解释。
    const settled = plan.policyWaste === 0;
    // 已经踩内存时不催人继续合并 —— 与规则引擎里 hasConflict 的抑制保持一致
    const blocked = plan.conflicts.some((c) => c.kind === 'lifetime');
    $('groupMeta').textContent = !usable.length ? '无'
      : settled ? '已排满'
        : `可省 ${F.bytes(Math.min(plan.policyWaste, usable.reduce((sum, g) => sum + g.saving, 0)))}`
          + (blocked ? ' · 先解冲突' : '');

    if (!usable.length && !plan.excluded.length) {
      host.innerHTML = '<p class="mv-empty">没有生命周期互不重叠的可合并组合。</p>';
      return;
    }

    usable.forEach((g, index) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `ws-group${index === state.selectedGroupIndex ? ' is-selected' : ''}`;
      btn.innerHTML = `
        <span class="ws-group-head">
          <span class="ws-group-members">${F.escapeHtml(g.members.map((t) => t.name).join(' + '))}</span>
          <span class="ws-group-saving${settled ? ' is-blocked' : ''}">−${F.bytes(g.saving)}</span>
        </span>
        <span class="ws-group-note">
          组内峰值 ${F.bytes(g.peak)} · ${g.blockScope} ·
          ${F.escapeHtml(g.members.map((t) => `${t.name} ${t.live.start === t.live.end ? `sg${t.live.start}` : `sg${t.live.start}–sg${t.live.end}`}`).join('，'))}
        </span>
        ${settled ? '<span class="ws-group-note mv-sev-success">当前布局已吃到这部分收益</span>' : ''}`;
      btn.addEventListener('click', () => {
        state.selectedGroupIndex = index === state.selectedGroupIndex ? null : index;
        state.selectedFindingId = null;
        state.selectedTensorId = g.members[0].id;
        render();
      });
      host.appendChild(btn);
    });

    // 护栏排除的组合也要露出来 —— 它们在甘特图上看着完全可以合并
    plan.excluded.slice(0, 3).forEach((e) => {
      const item = document.createElement('div');
      item.className = 'ws-group is-blocked';
      item.innerHTML = `
        <span class="ws-group-head">
          <span class="ws-group-members">${F.escapeHtml(`${e.a.name} + ${e.b.name}`)}</span>
          <span class="ws-group-saving is-blocked">−${F.bytes(e.saving)}</span>
        </span>
        <span class="ws-group-note mv-sev-warn">护栏排除：${F.escapeHtml(e.reason)}</span>`;
      host.appendChild(item);
    });
  }

  function renderDetail() {
    const host = $('detailBody');
    const kicker = $('detailKicker');

    const group = state.selectedGroupIndex != null ? plan.groups[state.selectedGroupIndex] : null;
    if (group) {
      kicker.textContent = '复用组';
      host.innerHTML = `
        <div class="mv-sec" style="padding:0;border:0;gap:8px">
          <span class="mv-detail-title">${F.escapeHtml(group.members.map((t) => t.name).join(' + '))}</span>
          <span class="mv-kv"><span>组内峰值</span><b>${F.bytes(group.peak)}</b></span>
          <span class="mv-kv"><span>合并后可省</span><b class="mv-sev-success">${F.bytes(group.saving)}</b></span>
          <span class="mv-kv"><span>作用域</span><b>${F.escapeHtml(group.blockScope)}</b></span>
          <div class="mv-soft">
            <span class="mv-soft-label">生效条件</span>
            <span class="mv-soft-text">按现有子计算顺序，${group.members.map((t, i) => i === 0
              ? `${t.name} 在 sg${t.live.end} 后不再被读`
              : `${t.name} 到 sg${t.live.start} 才产出`).join('；')}。保留这些同步关系即可安全共用一段地址。</span>
          </div>
          <div class="mv-evidence">
            ${group.members.map((t) => `<span class="mv-kv"><span>${F.escapeHtml(t.name)}</span><b>${F.bytes(t.size)} · sg${t.live.start}–sg${t.live.end}</b></span>`).join('')}
          </div>
        </div>`;
      return;
    }

    const tensor = state.selectedTensorId ? tensorById(state.selectedTensorId) : null;
    if (!tensor) {
      kicker.textContent = '未选中';
      host.innerHTML = '<p class="mv-empty">点击柱子分段、复用组或诊断条目查看详情。</p>';
      return;
    }

    kicker.textContent = tensor.role === 'workspace' ? 'workspace 张量' : tensor.role;
    const packedLayout = PLANNER.layoutOf(run);
    const currentOffset = run.layout[tensor.id];
    host.innerHTML = `
      <div class="mv-sec" style="padding:0;border:0;gap:8px">
        <span class="mv-detail-title">${F.escapeHtml(tensor.name)}</span>
        <span class="mv-kv"><span>大小</span><b>${F.bytes(tensor.size)}${tensor.size !== tensor.dataBytes ? `（数据 ${F.bytes(tensor.dataBytes)}）` : ''}</b></span>
        <span class="mv-kv"><span>dtype / shape</span><b>${F.escapeHtml(tensor.dtype)} ${F.escapeHtml(F.shape(tensor.shape))}</b></span>
        <span class="mv-kv"><span>产出 → 最后消费</span><b>${F.escapeHtml(tensor.producer || '—')} → ${F.escapeHtml(tensor.consumers.join(', ') || '—')}</b></span>
        <span class="mv-kv"><span>存活子计算</span><b>sg${tensor.live.start} – sg${tensor.live.end}</b></span>
        <span class="mv-kv"><span>作用域</span><b class="${tensor.blockScope === 'per-block' ? 'mv-sev-warn' : ''}">${F.escapeHtml(tensor.blockScope)}${tensor.blockScope === 'per-block' ? ` × ${run.kernel.blockDim}` : ''}</b></span>
        ${currentOffset != null ? `<span class="mv-kv"><span>当前地址</span><b>${F.hex(currentOffset, 6)}</b></span>` : ''}
        ${packedLayout[tensor.id] != null ? `<span class="mv-kv"><span>复用后地址</span><b>${F.hex(packedLayout[tensor.id], 6)}</b></span>` : ''}
        ${tensor.aliasOf ? `<span class="mv-kv"><span>原地宿主</span><b>${F.escapeHtml(tensor.aliasOf)}</b></span>` : ''}
        ${tensor.onChip ? `<span class="mv-kv"><span>已下沉片上</span><b class="mv-sev-success">${F.escapeHtml(tensor.onChipRegion)}</b></span>` : ''}
        ${tensor.manualReuseOf ? `<span class="mv-kv"><span>手工复用</span><b class="mv-sev-danger">${F.escapeHtml(tensor.manualReuseOf)}</b></span>` : ''}
        ${tensor.note ? `<div class="mv-soft"><span class="mv-soft-label">说明</span><span class="mv-soft-text">${F.escapeHtml(tensor.note)}</span></div>` : ''}
        <span class="mv-label">${F.escapeHtml(tensor.src.file)}:${tensor.src.hotLine + 1}</span>
        <pre class="mv-code">${F.escapeHtml(tensor.code)}</pre>
      </div>`;
  }

  // ---------------------------------------------------------------
  // 底部日志
  // ---------------------------------------------------------------
  function renderTerminal() {
    const lines = [];
    lines.push(`<span class="is-dim">$ memviz analyze --scenario workspace --kernel ${run.kernel.name} --candidate ${run.id}</span>`);
    lines.push(`<span class="is-dim">  chip=${chip.name}  budget=${F.bytes(plan.budget)}  subgraphs=${run.subgraphs.length}  tensors=${plan.tensors.length}</span>`);
    lines.push('');
    lines.push(`  current      ${F.bytes(plan.current)}   ← ${run.workspace.reportedAt.file}:${run.workspace.reportedAt.line + 1}`);
    lines.push(`  packed       ${F.bytes(plan.packed)}   (${plan.packings.map((p) => `${p.order} ${F.bytes(p.height)}`).join(', ')})`);
    lines.push(`  lowerBound   ${F.bytes(plan.lowerBound)}   @ sg${plan.peak.index} ${plan.peak.name}`);
    lines.push(`  ratio        ${plan.ratio.toFixed(2)}×   policyWaste=${F.bytes(plan.policyWaste)} packFragment=${F.bytes(plan.packFragment)}`);
    lines.push('');
    findings.forEach((f) => {
      const tag = f.severity === 'danger' ? 'is-danger' : f.severity === 'warn' ? 'is-warn' : 'is-dim';
      const level = f.severity === 'danger' ? 'ERROR' : f.severity === 'warn' ? 'WARN ' : 'INFO ';
      lines.push(`<span class="${tag}">  ${level} ${f.rule}</span>  ${F.escapeHtml(f.detail)}`);
    });
    lines.push('');
    const exit = summary.danger ? 2 : summary.warn ? 1 : 0;
    const cls = exit === 2 ? 'is-danger' : exit === 1 ? 'is-warn' : 'is-ok';
    lines.push(`<span class="${cls}">  ${summary.danger} danger, ${summary.warn} warn, ${summary.info} info → exit ${exit}</span>`);
    lines.push(`<span class="is-dim">  --fail-on ws-ratio>1.5 → ${plan.ratio > 1.5 ? '<span class="is-danger">FAIL</span>' : 'pass'}</span>`);
    $('terminalBody').innerHTML = lines.join('\n');
  }

  // ---------------------------------------------------------------
  // 状态条
  // ---------------------------------------------------------------
  function renderStatus() {
    const sg = run.subgraphs[state.sg];
    const cell = plan.perSubgraph[state.sg];
    const items = [
      ['芯片', chip.name],
      ['候选', run.label],
      ['子计算', `sg${state.sg} ${sg.name}`],
      ['此刻存活', `${F.bytes(cell.bytes)} · ${cell.members.length} 项`],
      ['当前', F.bytes(plan.current)],
      ['可达', F.bytes(plan.packed)],
      ['下界', F.bytes(plan.lowerBound)],
      ['比值', `${plan.ratio.toFixed(2)}×`],
    ];
    const html = items.map(([k, v]) => `<span class="mv-status-item">${F.escapeHtml(k)} <b>${F.escapeHtml(v)}</b></span>`).join('');
    const budget = plan.overBudget > 0
      ? `<span class="mv-status-item mv-sev-danger">超预算 <b class="mv-sev-danger">${F.bytes(plan.overBudget)}</b></span>`
      : `<span class="mv-status-item mv-sev-success">预算内</span>`;
    const diag = `<span class="mv-status-item">诊断 <b>${summary.danger}/${summary.warn}/${summary.info}</b></span>`;
    $('statusStrip').innerHTML = html + budget + diag
      + '<span class="mv-status-spacer"></span>'
      + '<span class="mv-status-item">L2 · schema-generated · exploration-only</span>';
  }

  // ---------------------------------------------------------------
  // 视图
  // ---------------------------------------------------------------
  function highlightIds() {
    const f = findings.find((x) => x.id === state.selectedFindingId);
    return new Set(f ? f.refs : []);
  }

  function renderViews() {
    $('viewPlan').classList.toggle('is-active', state.view === 'plan');
    $('viewGmLayout').classList.toggle('is-active', state.view === 'gm');
    document.querySelectorAll('#viewTabs .tab-control-item').forEach((tab) => {
      const on = tab.dataset.view === state.view;
      tab.classList.toggle('is-selected', on);
      tab.setAttribute('aria-selected', String(on));
    });

    views.plan.update({
      run, plan, sg: state.sg,
      selectedTensorId: state.selectedTensorId,
      highlightIds: highlightIds(),
    });

    if (state.view === 'gm') {
      views.gm.update({
        run, plan, mode: state.layoutMode, selectedTensorId: state.selectedTensorId,
      });
    }

    views.gap.update({
      rows: runs.map((item) => runIndex.get(item.id)),
      activeId: run.id,
    });
  }

  function redrawViews() {
    views.plan?.redraw?.();
    views.gap?.redraw?.();
    if (state.view === 'gm') views.gm?.redraw?.();
  }

  function render() {
    $('kernelName').textContent = `${run.kernel.name} · ${run.label}`;
    $('inspectorMeta').textContent = `${findings.length} 条结论`;
    renderLayoutModeSwitch();
    renderLegend();
    renderExplorer();
    renderFindings();
    renderGroups();
    renderDetail();
    renderTerminal();
    renderStatus();
    renderViews();
    syncPlaybackUi();
  }

  // ---------------------------------------------------------------
  // 播放条 —— 游标单位是子计算序
  // ---------------------------------------------------------------
  function setSubgraph(next) {
    const max = run.subgraphs.length - 1;
    const clamped = Math.max(0, Math.min(max, Math.round(next)));
    if (clamped === state.sg) return;
    state.sg = clamped;
    renderStatus();
    renderViews();
    syncPlaybackUi();
  }

  function initPlayback() {
    const helper = global.PtoFloatingPlaybackControl;
    const mount = $('playbackMount');
    const ids = {
      shell: 'ws-pb-shell', toggle: 'ws-pb-toggle',
      collapsedButton: 'ws-pb-collapsed', collapsedIcon: 'ws-pb-collapsed-icon',
      controls: 'ws-pb-controls', stepBack: 'ws-pb-back', play: 'ws-pb-play',
      stepForward: 'ws-pb-fwd', replay: 'ws-pb-replay', scrubber: 'ws-pb-scrubber',
      scrubberLabel: 'ws-pb-label', scrubberOpname: 'ws-pb-opname', scrubberHover: 'ws-pb-hover',
    };
    const control = helper.createControl({ ids, className: 'pto-ide-frame__floating-playback' });
    mount.appendChild(control);

    const scrubber = $(ids.scrubber);
    const counter = $(ids.scrubberLabel);
    const opname = $(ids.scrubberOpname);

    const instance = helper.init({ root: control, ...ids, isPlaying: () => state.playing });
    helper.initScrubberHover?.({
      root: control,
      totalSteps: run.subgraphs.length,
      getLabelForStep: (step) => run.subgraphs[step]?.name || `sg${step}`,
    });

    const setPlaying = (next) => {
      state.playing = next;
      clearInterval(playTimer);
      if (next) {
        playTimer = setInterval(() => {
          if (state.sg >= run.subgraphs.length - 1) { setPlaying(false); return; }
          setSubgraph(state.sg + 1);
        }, 720);
      }
      syncPlaybackUi();
    };

    $(ids.play)?.addEventListener('click', () => setPlaying(!state.playing));
    $(ids.stepBack)?.addEventListener('click', () => { setPlaying(false); setSubgraph(state.sg - 1); });
    $(ids.stepForward)?.addEventListener('click', () => { setPlaying(false); setSubgraph(state.sg + 1); });
    $(ids.replay)?.addEventListener('click', () => { setPlaying(false); setSubgraph(0); });
    scrubber?.addEventListener('input', () => { setPlaying(false); setSubgraph(Number(scrubber.value)); });

    playback = { control, instance, scrubber, counter, opname, play: $(ids.play), setPlaying };
    syncPlaybackRange();
  }

  function syncPlaybackRange() {
    if (!playback) return;
    playback.scrubber.min = '0';
    playback.scrubber.max = String(run.subgraphs.length - 1);
  }

  function syncPlaybackUi() {
    if (!playback) return;
    const helper = global.PtoFloatingPlaybackControl;
    const sg = run.subgraphs[state.sg];
    playback.scrubber.value = String(state.sg);
    playback.counter.textContent = `sg${state.sg} / sg${run.subgraphs.length - 1}`;
    playback.opname.textContent = `${sg.name} · ${F.bytes(plan.perSubgraph[state.sg].bytes)}`;
    if (playback.play && helper.iconLabel) {
      playback.play.innerHTML = state.playing
        ? helper.iconLabel('pause') : helper.iconLabel('play');
    }
    playback.instance?.sync?.({ playing: state.playing });
  }

  // ---------------------------------------------------------------
  // 绑定
  // ---------------------------------------------------------------
  function bindPanelToggles() {
    const railExplorer = document.querySelector('.pto-ide-frame__activity-rail [data-ide-toggle="explorer"]');
    const topExplorer = $('topExplorerToggle');
    topExplorer.addEventListener('click', () => {
      railExplorer.click();
      const pressed = railExplorer.getAttribute('aria-pressed') === 'true';
      topExplorer.classList.toggle('is-selected', pressed);
      topExplorer.setAttribute('aria-pressed', String(pressed));
      topExplorer.setAttribute('aria-expanded', String(pressed));
      window.requestAnimationFrame(redrawViews);
    });

    const inspectorPane = $('inspectorPane');
    const topInspector = $('topInspectorToggle');
    const toggleInspector = () => {
      const nextHidden = !inspectorPane.hidden;
      inspectorPane.hidden = nextHidden;
      inspectorPane.setAttribute('aria-hidden', String(nextHidden));
      const gutter = inspectorPane.previousElementSibling;
      if (gutter?.classList.contains('pto-workbench-shell__split-gutter')) gutter.hidden = nextHidden;
      topInspector.classList.toggle('is-selected', !nextHidden);
      topInspector.setAttribute('aria-pressed', String(!nextHidden));
      topInspector.setAttribute('aria-expanded', String(!nextHidden));
      frameController?.refresh?.();
      window.requestAnimationFrame(redrawViews);
    };
    topInspector.addEventListener('click', toggleInspector);
    $('railDiagnosticsToggle').addEventListener('click', () => {
      if (inspectorPane.hidden) toggleInspector();
      $('findingList').firstElementChild?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    });

    $('railTerminalToggle').addEventListener('click', () => {
      document.querySelector('[data-ide-toggle="terminal"]')?.click();
      window.requestAnimationFrame(redrawViews);
    });
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
    $('gotoPeak').addEventListener('click', () => setSubgraph(plan.peak.index));
    $('gotoIssue').addEventListener('click', () => {
      const first = findings.find((f) => f.severity === 'danger') || findings[0];
      if (!first) return;
      state.selectedFindingId = first.id;
      state.selectedGroupIndex = null;
      if (first.refs.length) state.selectedTensorId = first.refs[0];
      if (first.subgraph != null) state.sg = first.subgraph;
      render();
    });
  }

  function bindKeyboard() {
    window.addEventListener('keydown', (event) => {
      if (event.target.matches('input, textarea')) return;
      if (event.key === 'ArrowRight') { playback?.setPlaying(false); setSubgraph(state.sg + 1); event.preventDefault(); }
      if (event.key === 'ArrowLeft') { playback?.setPlaying(false); setSubgraph(state.sg - 1); event.preventDefault(); }
      if (event.key === ' ') { playback?.setPlaying(!state.playing); event.preventDefault(); }
    });
  }

  // ---------------------------------------------------------------
  function init() {
    loadChip(state.chipId);
    selectRun(state.runId);

    views.plan = global.MemVizWsPlanView.create($('planHost'), {
      onPickTensor: (id) => {
        state.selectedTensorId = state.selectedTensorId === id ? null : id;
        state.selectedGroupIndex = null;
        renderDetail();
        renderViews();
      },
      onPickSubgraph: setSubgraph,
    });
    views.gm = global.MemVizWsLayoutView.create($('gmLayoutHost'));
    views.gap = global.MemVizWsGapView.create($('gapHost'), {
      onPickRun: (id) => {
        if (id === state.runId) return;
        selectRun(id);
        render();
        window.requestAnimationFrame(redrawViews);
      },
    });

    frameController = global.PtoIdeFrame?.init($('ideFrame'), {
      splitOptions: { default: { onResize: () => window.requestAnimationFrame(redrawViews) } },
    });

    renderChipSwitch();
    initPlayback();
    bindPanelToggles();
    bindTabs();
    bindToolbar();
    bindKeyboard();
    render();
    redrawViews();
    // 后台标签页里首帧可能尚未发生，可见后补一次
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden) window.requestAnimationFrame(redrawViews);
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})(window);
