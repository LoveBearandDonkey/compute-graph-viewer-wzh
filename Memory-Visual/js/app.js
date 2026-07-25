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
    focusRegionId: 'UB',
    tick: 0,
    playing: false,
    selectedAllocId: null,
    selectedEventId: null,
    selectedFindingId: null,
  };

  let chip = null;
  let runs = [];
  let run = null;
  let metrics = null;
  let findings = [];
  let runIndex = new Map(); // runId -> { run, metrics, findings, summary }
  let views = {};
  let frameController = null;
  let playback = null;
  let playTimer = null;

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

  function renderRegionSwitch() {
    const host = $('regionSwitch');
    host.innerHTML = '';
    chip.regions.forEach((region) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `btn btn-sm${region.id === state.focusRegionId ? ' is-selected' : ''}`;
      btn.textContent = region.id;
      btn.title = `${region.label} · ${F.bytes(region.capacity)} · ${region.align}B 对齐`;
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
    lifetime: [],
    pipeline: [
      { cls: '', label: '流水任务' },
      { cls: 'is-wait', label: '可归因等待' },
      { cls: 'is-gap', label: '启动/空闲' },
    ],
  };

  function renderLegend() {
    const host = $('viewLegend');
    const items = LEGENDS[state.view] || [];
    host.innerHTML = items.map((item) => `
      <span class="mv-legend-item"><span class="mv-legend-swatch ${item.cls}"></span>${item.label}</span>
    `).join('');
  }

  // ---------------------------------------------------------------
  // 左栏：tiling 候选
  // ---------------------------------------------------------------
  function renderExplorer() {
    const host = $('explorerBody');
    host.innerHTML = '';
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
        syncPlaybackRange();
        render();
      });
      list.appendChild(btn);
    });
    group.appendChild(list);
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
      kicker.textContent = `${alloc.region} · ${alloc.kind === 'queue' ? 'TQue' : alloc.kind === 'gm' ? 'GlobalTensor' : 'TBuf'}`;
      const span = global.MemVizMetrics.liveSpan(alloc);
      const related = findings.filter((f) => f.refs.includes(alloc.id));
      host.innerHTML = `
        <div class="mv-sec" style="padding:0;border:0;gap:var(--space-2)">
          <div class="mv-detail-title">${F.escapeHtml(alloc.name)}</div>
          <div class="mv-chip-row">
            <span class="stat-chip">${F.escapeHtml(alloc.queue)}</span>
            <span class="stat-chip">buffer_num ${alloc.bufferNum}</span>
            ${alloc.manualReuse ? '<span class="stat-chip mv-sev-danger">手工复用</span>' : ''}
          </div>
          <div class="mv-kv"><span>地址区间</span><b>${F.hex(alloc.offset)} – ${F.hex(alloc.offset + alloc.size)}</b></div>
          <div class="mv-kv"><span>实占 / 数据</span><b>${F.bytes(alloc.size)} / ${F.bytes(alloc.dataBytes)}</b></div>
          <div class="mv-kv"><span>对齐粒度</span><b>${alloc.align}B</b></div>
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
        return `<span class="is-dim">region  </span> ${region.id.padEnd(4)} ${String(F.bytes(region.reserved)).padStart(9)} / ${String(F.bytes(region.capacity)).padStart(9)}  <span class="${cls}">${F.pct(region.reservedRatio, 1).padStart(7)}</span>`;
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
    $('statusStrip').innerHTML = `
      <span class="mv-status-item">芯片 <b>${F.escapeHtml(chip.name)}</b></span>
      <span class="mv-status-item">候选 <b>${F.escapeHtml(run.label)}</b></span>
      <span class="mv-status-item">tileM <b>${run.tiling.tileM}</b> × <b>${run.tiling.tileNum}</b>${run.tiling.hasTail ? `（尾块 ${run.tiling.tailM}）` : ''}</span>
      <span class="mv-status-item">${region.id} <b class="${region.reserved > region.capacity ? 'mv-sev-danger' : ''}">${F.bytes(region.reserved)} / ${F.bytes(region.capacity)}</b></span>
      <span class="mv-status-item">峰值持有 <b>${F.bytes(region.peakLive)} @ ${F.tick(region.peakTick)}</b></span>
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

    if (state.view === 'layout') {
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
    $('kernelName').textContent = `${run.kernel.name} · ${run.label}`;
    document.querySelectorAll('#viewTabs .tab-control-item').forEach((tab) => {
      const active = tab.dataset.view === state.view;
      tab.classList.toggle('is-selected', active);
      tab.setAttribute('aria-selected', String(active));
    });
    renderLegend();
    renderExplorer();
    renderFindings();
    renderDetail();
    renderTerminal();
    renderStatus();
    renderViews();
    syncPlaybackUi();
  }

  function redrawViews() {
    views.layout?.redraw?.();
    views.pipeline?.redraw?.();
    views.watermark?.redraw?.();
    views.lifetime?.redraw?.();
  }

  // ---------------------------------------------------------------
  // 播放条（floating-playback-control）
  // ---------------------------------------------------------------
  function setTick(next) {
    const clamped = Math.max(0, Math.min(run.totalTicks, Math.round(next)));
    if (clamped === state.tick) return;
    state.tick = clamped;
    renderStatus();
    renderViews();
    syncPlaybackUi();
  }

  function initPlayback() {
    const helper = global.PtoFloatingPlaybackControl;
    const mount = $('playbackMount');
    const ids = {
      shell: 'mv-pb-shell', toggle: 'mv-pb-toggle',
      collapsedButton: 'mv-pb-collapsed', collapsedIcon: 'mv-pb-collapsed-icon',
      controls: 'mv-pb-controls', stepBack: 'mv-pb-back', play: 'mv-pb-play',
      stepForward: 'mv-pb-fwd', replay: 'mv-pb-replay', scrubber: 'mv-pb-scrubber',
      scrubberLabel: 'mv-pb-label', scrubberOpname: 'mv-pb-opname', scrubberHover: 'mv-pb-hover',
    };
    const control = helper.createControl({ ids, className: 'pto-ide-frame__floating-playback' });
    mount.appendChild(control);

    const scrubber = $(ids.scrubber);
    const counter = $(ids.scrubberLabel);
    const opname = $(ids.scrubberOpname);
    const stepBack = $(ids.stepBack);
    const play = $(ids.play);
    const stepForward = $(ids.stepForward);
    const replay = $(ids.replay);

    const instance = helper.init({ root: control, ...ids, isPlaying: () => state.playing });
    helper.initScrubberHover?.({
      root: control,
      totalSteps: run.totalTicks + 1,
      getLabelForStep: (step) => `#${step}`,
    });

    const setPlaying = (next) => {
      state.playing = next;
      clearInterval(playTimer);
      if (next) {
        playTimer = setInterval(() => {
          if (state.tick >= run.totalTicks) { setPlaying(false); return; }
          setTick(state.tick + Math.max(1, Math.round(run.totalTicks / 120)));
        }, 40);
      }
      syncPlaybackUi();
    };

    play?.addEventListener('click', () => setPlaying(!state.playing));
    stepBack?.addEventListener('click', () => { setPlaying(false); setTick(state.tick - 1); });
    stepForward?.addEventListener('click', () => { setPlaying(false); setTick(state.tick + 1); });
    replay?.addEventListener('click', () => { setPlaying(false); setTick(0); });
    scrubber?.addEventListener('input', () => { setPlaying(false); setTick(Number(scrubber.value)); });

    playback = { control, instance, scrubber, counter, opname, play, setPlaying };
    syncPlaybackRange();
  }

  function syncPlaybackRange() {
    if (!playback) return;
    playback.scrubber.min = '0';
    playback.scrubber.max = String(run.totalTicks);
  }

  function eventAtTick(tick) {
    return run.events.find((e) => tick >= e.t && tick < e.end) || null;
  }

  function syncPlaybackUi() {
    if (!playback) return;
    const helper = global.PtoFloatingPlaybackControl;
    playback.scrubber.value = String(state.tick);
    playback.counter.textContent = `${state.tick} / ${run.totalTicks}`;
    const active = eventAtTick(state.tick);
    playback.opname.textContent = active ? `${active.pipe} · ${active.label}` : '空闲';
    if (playback.play && helper.iconLabel) {
      playback.play.innerHTML = state.playing
        ? helper.iconLabel('pause', 'Pause')
        : helper.iconLabel('play', 'Play');
    }
    playback.instance?.sync?.({ playing: state.playing });
  }

  // ---------------------------------------------------------------
  // 面板开关（顶栏按钮镜像 activity rail 的行为）
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

    const railTerminal = $('railTerminalToggle');
    railTerminal.addEventListener('click', () => {
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
    $('gotoPeak').addEventListener('click', () => {
      const region = metrics.regionById[state.focusRegionId];
      setTick(region.peakTick);
    });
    $('gotoIssue').addEventListener('click', () => {
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
      if (event.key === 'ArrowRight') { playback?.setPlaying(false); setTick(state.tick + (event.shiftKey ? 10 : 1)); event.preventDefault(); }
      if (event.key === 'ArrowLeft') { playback?.setPlaying(false); setTick(state.tick - (event.shiftKey ? 10 : 1)); event.preventDefault(); }
      if (event.key === ' ') { playback?.setPlaying(!state.playing); event.preventDefault(); }
    });
  }

  // ---------------------------------------------------------------
  function init() {
    loadChip(state.chipId);
    selectRun(state.runId);

    views.layout = global.MemVizLayoutView.create($('layoutHost'), {
      onSelect: (alloc) => {
        state.selectedAllocId = alloc ? alloc.id : null;
        state.selectedEventId = null;
        renderDetail();
        renderViews();
      },
    });
    views.lifetime = global.MemVizLifetimeView.create($('viewLifetime'));
    views.pipeline = global.MemVizPipelineView.create($('pipelineHost'), {
      onSelectEvent: (event) => {
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
    renderRegionSwitch();
    initPlayback();
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
