/* Schema consumer only: all training times, topology and model bindings come from producer JSON. */
window.TrainingStructureReady = (async () => {
  'use strict';
  const $ = id => document.getElementById(id);
  const pattern = window.PtoSwimlaneTaskPattern;
  const viewport = $('timeline-viewport'), canvas = $('timeline'), ctx = canvas.getContext('2d');
  const frame = $('model-graph'), detail = $('detail');
  const responses = await Promise.all([fetch('./training-timeline.schema.json'), fetch('./data/deepseek-v4-pro.timeline.json')]);
  if (responses.some(r => !r.ok)) throw new Error('Schema / timeline JSON 加载失败');
  const [schema, input] = await Promise.all(responses.map(r => r.json()));
  const dataset = window.TimelineContract.validate(input, schema);
  const STAGES = dataset.stages;
  const kinds = { summary: '执行时序', forward: '前向', backward: '反向', comm: '通信', wait: '依赖等待', hold: '激活驻留', optimizer: '优化器', misc: '其他' };
  const relations = { direct: '直接对应', 'backward-owner': '反向归属', 'communication-participant': '通信参与', 'activation-owner': '激活归属', 'runtime-owner': '训练运行时归属' };
  const state = { dataset: dataset.id, dp: 'all', mb: 'all', mode: 'events', view: 'global', zoom: 0, expanded: new Set(), selection: null, eventId: null, eventKey: null, eventSourceIds: [], stage: 0 };
  const events = dataset.events;
  const runtimeNodes = new Map(dataset.runtimeGraph.nodes.map(node => [node.id, node]));
  const parameterGroups = new Map(dataset.parameterGroups.map(group => [group.id, group]));
  const optimizerStateGroups = new Map(dataset.optimizerStateGroups.map(group => [group.id, group]));
  const collectiveGroups = new Map(dataset.collectiveGroups.map(group => [group.id, group]));
  const dependents = new Map();
  events.forEach(event => event.dependsOn.forEach(id => {
    if (!dependents.has(id)) dependents.set(id, []);
    dependents.get(id).push(event.id);
  }));
  const fmt = v => `${v.toFixed(1)} ms`;
  const escape = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  let byId = new Map();
  const mbMatches = e => state.mb === 'all' || e.microbatchId === state.mb;
  const deploymentEvents = () => {
    const ranks = new Set(dataset.ranks.filter(r=>state.dp==='all'||String(r.dp)===state.dp).map(r=>r.rank));
    return events.filter(e=>ranks.has(e.rank)||e.participants.some(r=>ranks.has(r)));
  };
  const scopeEvents = () => {
    const deployed = deploymentEvents(), ranks = new Set(dataset.ranks.filter(r => r.stage === state.stage && (state.dp === 'all' || String(r.dp) === state.dp)).map(r => r.rank));
    return deployed.filter(e => (state.view === 'global' || e.stage === state.stage || e.kind === 'comm' && e.participants.some(r => ranks.has(r))) &&
      (state.view !== 'forward' || ['forward', 'comm'].includes(e.kind)) && (state.view !== 'backward' || ['backward', 'comm'].includes(e.kind)));
  };
  const microbatches = () => [...new Set(deploymentEvents().map(e => e.microbatchId).filter(mb => mb !== null))].sort();
  function activateDataset() {
    byId = new Map(events.map(e => [e.id, e]));
    for (const r of dataset.ranks) { state.expanded.add(`dp${r.dp}`); state.expanded.add(`dp${r.dp}/pp${r.stage}`); }
    $('view').value = 'global';
    $('deployment').innerHTML = [...new Set(dataset.ranks.map(r=>r.dp))].map(dp=>`<option value="${dp}">DP${dp}</option>`).join('') + '<option value="all">全部 DP</option>';
    $('deployment').value = state.dp;
    $('stage').innerHTML = STAGES.map(s => `<option value="${s.id}">PP${s.id} · L${s.first}–L${s.last}</option>`).join('');
    frame.hidden = false; $('unlinked-structure').hidden = true;
    // Send immediately when data is ready; the iframe-ready handshake below
    // repeats this if its listener was not installed yet.
    post('csa-runtime-graph', { runtimeGraph: dataset.runtimeGraph });
    $('structure-title').textContent = dataset.label + ' · 模型结构 + 训练运行时';
    $('structure-meta').textContent = `${dataset.model.numHiddenLayers} 主层 + ${dataset.model.mtpDepth} MTP`;
    $('timeline-meta').textContent = `DP${dataset.training.dp} · PP${dataset.training.pp} · TP${dataset.training.tp} · EP${dataset.training.ep} · ${dataset.ranks.length} Ranks（模拟）`;
    $('graph-level').querySelectorAll('button').forEach(b => { b.disabled = false; });
    $('graph-fit').disabled = false;
    $('view').querySelector('[value="object"]').disabled = false;
    $('fidelity').textContent = '模拟配置 / 数据依据 ⓘ';
    viewport.scrollLeft = 0; viewport.scrollTop = 0;
  }
  function union(input, grouped = false) {
    const groups = new Map();
    input.forEach(e => {
      const key = grouped ? `${e.rank}/${e.kind}/${e.microbatchId}` : 'all';
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(e);
    });
    const out = [];
    groups.forEach(list => {
      let last;
      list.sort((a, b) => a.start - b.start || a.end - b.end).forEach(e => {
        if (last && e.start <= last.end + 0.001) {
          last.end = Math.max(last.end, e.end);
          last.sourceEventIds = [...new Set([...last.sourceEventIds, ...e.sourceEventIds])];
          last.graphNodeIds = [...new Set([...last.graphNodeIds, ...e.graphNodeIds])];
          last.runtimeNodeIds = [...new Set([...last.runtimeNodeIds, ...e.runtimeNodeIds])];
          last.parameterGroupIds = [...new Set([...last.parameterGroupIds, ...e.parameterGroupIds])];
          last.optimizerStateGroupIds = [...new Set([...last.optimizerStateGroupIds, ...e.optimizerStateGroupIds])];
          last.shards = [...last.shards, ...e.shards];
          last.participants = [...new Set([...last.participants, ...e.participants])];
        } else {
          last = { ...e, id: `union:${e.id}`, sourceEventIds: [...e.sourceEventIds], graphNodeIds: [...e.graphNodeIds], participants: [...e.participants], isUnion: true };
          out.push(last);
        }
      });
    });
    return out;
  }
  function metrics(list) {
    if (!list.length) return { span: 0, active: 0, sum: 0, ranks: 0 };
    return { span: Math.max(...list.map(e => e.end)) - Math.min(...list.map(e => e.start)),
      active: union(list).reduce((v, e) => v + e.end - e.start, 0), sum: list.reduce((v, e) => v + e.end - e.start, 0),
      ranks: new Set(list.map(e => e.rank)).size };
  }
  function matches(e, selection = state.selection) {
    if (!selection) return true;
    const bindings=[...e.graphNodeIds,...e.runtimeNodeIds];
    if (selection.ids) return bindings.some(id => selection.ids.some(s => id === s || id.startsWith(`${s}/`)));
    const id = selection.nodeId;
    if (id === 'dv4/model' || id === 'dv4') return true;
    if (id === dataset.runtimeGraph.id) return e.runtimeNodeIds.length > 0;
    if (id.startsWith('runtime/')) return e.runtimeNodeIds.some(nodeId => nodeId === id || nodeId.startsWith(`${id}/`));
    if (id === 'l1_csa' || id.startsWith('csa_') || id.startsWith('l2_')) return e.label.startsWith('CSA');
    return e.templateNodeIds.some(n => n === id || n.startsWith(`${id}/`));
  }
  function selectedEvents() { return scopeEvents().filter(e => mbMatches(e) && matches(e)); }
  let palette, cssStyle;
  function css(name) { return cssStyle.getPropertyValue(name).trim(); }
  function syncColors() {
    cssStyle = getComputedStyle(document.documentElement);
    palette = pattern.createTaskColormap({ labelColors: COLORS });
  }
  const COLORS = {
    summary: '#38bdf8', forward: '#4369ef', backward: '#ff4b7b', wait: '#737373', comm: '#04d793',
    hold: '#a855f7', memory: '#ffaa3b', release: '#87c80f', loss: '#f97316', optimizer: '#ffaa3b', misc: '#737373'
  };
  const TIMELINE_LAYOUT = Object.freeze({
    rowHeight: 22,
    eventHeight: 16,
    expandedRankHeight: 58,
    rankTrackPitch: 18,
    rankTrackTop: 2
  });
  const color = e => palette.colorForTask({ colorKey: e.kind, label: e.kind });
  let rows = [], hits = [], width = 0, height = 0, raf = 0;
  const LABEL = 236, HEADER = 48, ZOOMS = [1, 1.5, 2, 4, 8];
  function range() {
    // MB focus never rewrites the time window; both DPs share the producer's origin.
    const list = scopeEvents();
    return list.length ? [Math.floor(Math.min(...list.map(e => e.start))), Math.ceil(Math.max(...list.map(e => e.end)))] : [0, 1];
  }
  function buildRows() {
    rows = [];
    const row = (id, label, meta, depth, list, expandable = false, summary = false) => {
      const rowType = /\/r\d+$/.test(id) ? 'rank' : 'object';
      rows.push({ id, label, meta, depth, list, expandable, summary, rowType,
        h: rowType === 'rank' ? (state.expanded.has(id) ? 40 : TIMELINE_LAYOUT.expandedRankHeight) : TIMELINE_LAYOUT.rowHeight });
    };
    const scoped = scopeEvents();
    [...new Set(dataset.ranks.filter(r=>state.dp==='all'||String(r.dp)===state.dp).map(r => r.dp))].sort().forEach(dp => {
      const dpRanks = new Set(dataset.ranks.filter(r=>r.dp===dp).map(r=>r.rank));
      const dpList = scoped.filter(e => dpRanks.has(e.rank)||e.participants.some(r=>dpRanks.has(r))), dpKey = `dp${dp}`;
      row(dpKey, `DP${dp}`, `${dpRanks.size} Ranks · 全部活动`, 0, dpList, true, true);
      if (!state.expanded.has(dpKey)) return;
      STAGES.filter(s => state.view === 'global' || s.id === state.stage).forEach(s => {
        const ranks = dataset.ranks.filter(r=>r.dp===dp&&r.stage===s.id).map(r=>r.rank);
        const list = dpList.filter(e=>ranks.includes(e.rank)||e.participants.some(r=>ranks.includes(r))), key = `${dpKey}/pp${s.id}`;
        row(key, `PP${s.id}`, ranks.map(r => `R${r}`).join(' / '), 1, list, true, true);
        if (!state.expanded.has(key)) return;
        ranks.forEach(r => {
          // Project known communication participants to each visible rank. Keep source
          // IDs, so a projected bar is not a duplicate event in metrics.
          const rk = `${key}/r${r}`, rl = scoped.filter(e => e.rank === r || e.kind === 'comm' && e.participants.includes(r));
          row(rk, `Rank ${r}`, `TP${dataset.ranks.find(n=>n.rank===r).tp} · EP${dataset.ranks.find(n=>n.rank===r).ep}`, 2, rl, true);
          if (state.expanded.has(rk)) {
            const categories = ['forward', 'backward', 'comm', 'optimizer', 'hold', 'wait', 'misc'];
            categories.forEach(kind => {
              const items = rl.filter(e => e.kind === kind);
              if (items.length) row(`${rk}/${kind}`, kinds[kind], '', 3, items);
            });
          }
        });
        const detailLayers = [...new Set(list.filter(e=>e.layer!==null).map(e=>e.layer))];
        if (state.view === 'object') row(`${key}/objects`, detailLayers.length ? '层对象 · 算子示例' : '阶段概览', detailLayers.length ? '仅代表层' : '未模拟层内算子', 2, list.filter(e => e.layer !== null), detailLayers.length>0);
        if (state.view === 'object' && state.expanded.has(`${key}/objects`)) {
          for (const layer of detailLayers) row(`${key}/layer${layer}`, `Layer ${layer}`, '典型算子', 3, list.filter(e=>e.layer===layer));
        }
      });
    });
  }
  function displayEvents(row) {
    // Expanded group headers carry hierarchy only; their Rank children carry time.
    if ((row.summary || row.rowType === 'rank') && state.expanded.has(row.id)) return [];
    const list = row.summary ? row.list.filter(e => !['hold', 'wait'].includes(e.kind)) : row.list;
    if (state.mode === 'events' && (row.summary || row.rowType === 'rank')) {
      const dp = Number(row.id.match(/^dp(\d+)/)[1]), pp = row.id.match(/\/pp(\d+)/), rank = row.id.match(/\/r(\d+)$/);
      const summaries = dataset.summaries.filter(e => e.dp === dp && (!pp || e.stage === Number(pp[1])) && (!rank || e.rank === Number(rank[1])) &&
        (state.view !== 'forward' || e.kind === 'forward') && (state.view !== 'backward' || e.kind === 'backward'));
      if (state.selection) {
        const matched=list.filter(e=>matches(e)&&mbMatches(e));
        // Replace a Rank summary only when that Rank contains a match. Ranks
        // outside the structure selection keep their scheduling context.
        return matched.length
          ? [...matched, ...list.filter(e=>['comm','optimizer','misc'].includes(e.kind))]
          : [...summaries, ...list.filter(e=>['comm','optimizer','misc'].includes(e.kind))];
      }
      return [...summaries, ...list.filter(e=>['comm','optimizer','misc'].includes(e.kind))];
    }
    const visible = row.rowType === 'rank' ? list.filter(e=>!['hold','wait'].includes(e.kind)) : list;
    if (state.mode !== 'union') return visible;
    // Partition before union: adjacent unrelated intervals must never enter S.
    const foreground = visible.filter(e => mbMatches(e) && matches(e));
    const background = visible.filter(e => !mbMatches(e) || !matches(e));
    return [...union(background, true), ...union(foreground, true)];
  }
  function layoutEvents(row, scale) {
    const list=displayEvents(row), rank=row.rowType==='rank';
    const groupFor=e=>rank ? (e.kind==='comm'?1:['forward','backward'].includes(e.kind)?0:2) : 0;
    const positions=new Map(), offsets=[], counts=[];
    let offset=0;
    for(let group=0;group<(rank?3:1);group++) {
      const ends=[];
      list.filter(e=>groupFor(e)===group).sort((a,b)=>a.start-b.start||a.end-b.end||a.id.localeCompare(b.id)).forEach(e=>{
        // Positive durations keep their real scale; tiny serial operators must
        // not be widened into fake overlap. Only zero-duration points use 3px.
        const start=e.start*scale, end=e.end>e.start?e.end*scale:start+3;
        let track=ends.findIndex(t=>t<=start+1e-8);
        if(track<0)track=ends.length;
        ends[track]=end;
        positions.set(e.id,offset+track);
      });
      offsets.push(offset);counts.push(Math.max(1,ends.length));offset+=Math.max(1,ends.length);
    }
    row.trackOffsets=offsets;
    if(list.length) row.h=Math.max(rank?TIMELINE_LAYOUT.expandedRankHeight:TIMELINE_LAYOUT.rowHeight,offset*TIMELINE_LAYOUT.rankTrackPitch+4);
    return list.map(event=>({event,track:positions.get(event.id)}));
  }
  const minimumVisibleWidth = e => ['comm','optimizer'].includes(e.kind) ? 5 : 0;
  // Task glyphs reuse the shared pattern; row headers are plain hierarchy labels.
  function rgbaFromHex(hex, alpha) {
    const clean = String(hex || '').replace('#', '');
    const full = clean.length === 3 ? clean.split('').map(char => char + char).join('') : clean;
    const value = Number.parseInt(full, 16);
    if (!Number.isFinite(value)) return `rgba(168,85,247,${alpha})`;
    const r = (value >> 16) & 255;
    const g = (value >> 8) & 255;
    const b = value & 255;
    return `rgba(${r},${g},${b},${alpha})`;
  }
  const isExpandableRow = lane => lane.expandable;
  const isExpandedRow = lane => state.expanded.has(lane.id);
  const headerCenter = (lane, rowH) => lane.rowType === 'rank'
    ? TIMELINE_LAYOUT.rankTrackTop + TIMELINE_LAYOUT.eventHeight / 2 : rowH / 2;
  const eventsForLane = lane => displayEvents(lane).filter(mbMatches)
    .map(e => ({ ...e, startMs: e.start, endMs: e.end }));
  function eventConnections(sourceIds = state.eventSourceIds) {
    const selected=new Set(sourceIds||[]), upstream=new Set(), downstream=new Set();
    selected.forEach(id=>{
      const event=byId.get(id);
      event?.dependsOn.forEach(dep=>{if(!selected.has(dep))upstream.add(dep);});
      event?.releaseEventIds?.forEach(next=>{if(!selected.has(next))downstream.add(next);});
      (dependents.get(id)||[]).forEach(next=>{if(!selected.has(next))downstream.add(next);});
    });
    return {upstream,downstream};
  }
  function connectionEndpoints(from,to) {
    // Dependencies are directional: OUT leaves the source's right edge and IN
    // enters the target's left edge, both at the vertical centre of the bar.
    return [{x:from.x+from.w,y:from.y+from.h/2},{x:to.x,y:to.y+to.h/2}];
  }
  function drawLaneConnector(ctx, lane, y, rowH, color) {
    const depth = lane.depth || 0;
    if (!depth) return;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.72;
    ctx.setLineDash([2, 3]);
    for (let level = 1; level <= depth; level += 1) {
      const guideX = 12 + level * 15;
      ctx.beginPath();
      ctx.moveTo(guideX + .5, y + 1);
      ctx.lineTo(guideX + .5, y + rowH - 1);
      ctx.stroke();
    }
    const branchX = 12 + depth * 15;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(branchX + .5, y + headerCenter(lane, rowH) + .5);
    ctx.lineTo(branchX + 11.5, y + headerCenter(lane, rowH) + .5);
    ctx.stroke();
    ctx.restore();
  }
  function roundedRect(ctx, x, y, width, height, radius) {
    if (ctx.roundRect) {
      ctx.beginPath();
      ctx.roundRect(x, y, width, height, radius);
      return;
    }
    ctx.beginPath();
    ctx.rect(x, y, width, height);
  }
  function drawChevronTrack(ctx, x, y, width, height, direction, options = {}) {
    if (width <= 0 || height <= 0 || (direction !== 'forward' && direction !== 'backward')) return;
    const step = options.step || 6;
    const inset = options.inset ?? .5;
    const color = options.color || 'rgba(255,255,255,.3)';
    const lineWidth = options.lineWidth || .6;
    ctx.save();
    roundedRect(ctx, x, y, width, height, options.radius ?? 3);
    ctx.clip();
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineJoin = 'miter';
    ctx.lineCap = 'butt';
    for (let px = x - step; px < x + width + step; px += step) {
      ctx.beginPath();
      if (direction === 'forward') {
        ctx.moveTo(px + inset, y + inset);
        ctx.lineTo(px + step - inset, y + height / 2);
        ctx.lineTo(px + inset, y + height - inset);
      } else {
        ctx.moveTo(px + step - inset, y + inset);
        ctx.lineTo(px + inset, y + height / 2);
        ctx.lineTo(px + step - inset, y + height - inset);
      }
      ctx.stroke();
    }
    ctx.restore();
  }
  function drawLaneGlyphTrack(ctx, x, y, width, height, direction, color) {
    if (width <= 0 || height <= 0 || (direction !== 'forward' && direction !== 'backward')) return;
    const glyph = direction === 'forward' ? '›' : '‹';
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, width, height);
    ctx.clip();
    ctx.fillStyle = color;
    ctx.globalAlpha = state.eventSourceIds.length ? .08 : .38;
    ctx.font = `800 23px ${css('--font-mono') || 'monospace'}`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.shadowColor = rgbaFromHex(color, .38);
    ctx.shadowBlur = 8;
    const step = ctx.measureText(glyph).width + 2;
    for (let px = x + 3; px < x + width + step; px += step) {
      ctx.fillText(glyph, px, y + height / 2 + .5);
    }
    ctx.restore();
  }
  function drawDirectionalLaneEdgeTexture(ctx, lane, x, range, y, rowH) {
    if (lane.rowType === 'rank' && isExpandedRow(lane)) return;
    const laneEvents = eventsForLane(lane);
    const forwardEvents = laneEvents.filter(item => item.kind === 'forward');
    const backwardEvents = laneEvents.filter(item => item.kind === 'backward');
    const textureY = y + 2;
    const textureH = rowH - 4;

    if (forwardEvents.length) {
      const forwardStart = Math.min(...forwardEvents.map(item => item.startMs));
      const edgeEnd = Math.min(forwardStart, range.end);
      if (edgeEnd > range.start) {
        const trackX = x(range.start);
        const trackW = x(edgeEnd) - trackX;
        drawLaneGlyphTrack(ctx, trackX, textureY, trackW, textureH, 'forward', COLORS.forward);
      }
    }

    if (backwardEvents.length) {
      const backwardEnd = Math.max(...backwardEvents.map(item => item.endMs));
      const edgeStart = Math.max(backwardEnd, range.start);
      if (edgeStart < range.end) {
        const trackX = x(edgeStart);
        const trackW = x(range.end) - trackX;
        drawLaneGlyphTrack(ctx, trackX, textureY, trackW, textureH, 'backward', COLORS.backward);
      }
    }
  }
  function redrawTaskLabel(ctx, item, barX, barY, barW, barH) {
    if (barW < 28) return;
    const fontFamily = css('--font-mono') || 'monospace';
    const maxChars = Math.max(4, Math.floor((barW - 8) / 6));
    const label = item.label.length > maxChars ? `${item.label.slice(0, Math.max(0, maxChars - 1))}…` : item.label;
    ctx.save();
    roundedRect(ctx, barX + 1, barY + 1, Math.max(0, barW - 2), Math.max(0, barH - 2), 2);
    ctx.clip();
    ctx.fillStyle = 'rgba(255,255,255,.94)';
    ctx.font = `${barW >= 72 ? '600 9px' : '600 8px'} ${fontFamily}`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.shadowColor = 'rgba(0,0,0,.7)';
    ctx.shadowBlur = 2;
    ctx.fillText(label, barX + 5, barY + barH / 2 + .5);
    ctx.restore();
  }
  function drawRowExpand(ctx, lane, x, y, rowH) {
    if (!isExpandableRow(lane)) return null;
    const size = 14;
    const iconY = y + headerCenter(lane, rowH) - size / 2;
    const expanded = isExpandedRow(lane);
    ctx.save();
    roundedRect(ctx, x, iconY, size, size, 4);
    ctx.fillStyle = css('--surface-1') || 'rgba(255,255,255,.9)';
    ctx.strokeStyle = css('--border-strong') || 'rgba(255,255,255,.68)';
    ctx.lineWidth = 1;
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = css('--foreground') || '#111827';
    ctx.font = `800 12px ${css('--font-mono') || 'monospace'}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(expanded ? '−' : '+', x + size / 2, iconY + size / 2 + .25);
    ctx.restore();
    return { x, y: iconY, w: size, h: size };
  }

  function laneHeaderParts(lane) {
    const parts = String(lane.label || '').split(' · ');
    if (parts.length < 2) return { object: lane.label, semantic: '' };
    const owner = parts[0];
    const scopeMatch = parts[1].match(/^(L\d+(?:–L\d+)?)(?:\s+(.+))?$/);
    if (/^PP\d+$/.test(owner) && scopeMatch) {
      return {
        object: `${owner} · ${scopeMatch[1]}`,
        semantic: [scopeMatch[2], ...parts.slice(2)].filter(Boolean).join(' · ')
      };
    }
    return { object: owner, semantic: parts.slice(1).join(' · ') };
  }

  function fitCanvasText(ctx, value, maxWidth) {
    const text = String(value || '');
    if (ctx.measureText(text).width <= maxWidth) return text;
    let fitted = text;
    while (fitted.length > 1 && ctx.measureText(`${fitted}…`).width > maxWidth) fitted = fitted.slice(0, -1);
    return `${fitted}…`;
  }

  function drawLaneHeader(ctx, lane, x, y, rowH, maxWidth, textColor) {
    ctx.save();
    ctx.font = `600 12px ${css('--font-sans') || 'sans-serif'}`;
    ctx.fillStyle = textColor;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    const label = fitCanvasText(ctx, lane.label, maxWidth);
    ctx.fillText(label, x, y + rowH / 2);
    const used = ctx.measureText(label).width + 10;
    if (lane.meta && maxWidth-used>24) {
      ctx.font = `400 11px ${css('--font-sans') || 'sans-serif'}`;
      ctx.fillStyle = css('--foreground-secondary');
      ctx.fillText(fitCanvasText(ctx, lane.meta, maxWidth-used), x+used, y+rowH/2);
    }
    ctx.restore();
    return { x, y, w: maxWidth, h: rowH };
  }

  function drawExpandedRankHeader(ctx, lane, x, y, rowH, maxWidth, textColor) {
    ctx.save();
    ctx.font = `600 12px ${css('--font-sans') || 'sans-serif'}`;
    ctx.fillStyle = textColor;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(lane.label, x, y + headerCenter(lane, rowH));
    ctx.font = `400 11px ${css('--font-sans') || 'sans-serif'}`;
    ctx.fillStyle = css('--foreground-secondary');
    ctx.fillText(lane.meta, x, y + headerCenter(lane, rowH) + TIMELINE_LAYOUT.rankTrackPitch);
    if (state.expanded.has(lane.id)) { ctx.restore(); return { x, y, w: maxWidth, h: rowH }; }
    const trackLabels = ['计算', '通信', '其他活动'];
    ctx.fillStyle = css('--foreground-muted') || '#888';
    ctx.font = `400 11px ${css('--font-sans') || 'sans-serif'}`;
    ctx.textAlign = 'right';
    trackLabels.forEach((label, track) => ctx.fillText(
      label,
      x + maxWidth,
      y + TIMELINE_LAYOUT.rankTrackTop + TIMELINE_LAYOUT.eventHeight / 2 + (lane.trackOffsets?.[track]??track) * TIMELINE_LAYOUT.rankTrackPitch
    ));
    ctx.restore();
    return { x, y, w: maxWidth, h: rowH };
  }

  function scheduleDraw() { if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); }); }
  function draw() {
    width = viewport.clientWidth; height = viewport.clientHeight;
    if (width < 1 || height < 1) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
    canvas.style.width = `${width}px`; canvas.style.height = `${height}px`;
    const totalWidth = LABEL + (width - LABEL) * ZOOMS[state.zoom];
    const [start, end] = range(), scale = (totalWidth - LABEL - 16) / (end - start);
    const layouts=new Map(rows.map(row=>[row.id,layoutEvents(row,scale)]));
    const selectedSourceIds=new Set(state.eventSourceIds), directLinks=eventConnections();
    const linkedSourceIds=new Set([...directLinks.upstream,...directLinks.downstream]);
    const subduedAlpha=Number.parseFloat(css('--button-disabled-opacity'))||.42;
    const totalHeight = HEADER + rows.reduce((sum, r) => sum + r.h, 0);
    $('timeline-extent').style.width = `${totalWidth}px`;
    $('timeline-extent').style.height = `${Math.max(height, totalHeight)}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height); hits = [];
    const x = t => LABEL + (t - start) * scale - viewport.scrollLeft;
    ctx.fillStyle = css('--background'); ctx.fillRect(0, 0, width, height);
    const tick = [1, 2, 5, 10, 20, 40, 80, 100, 200, 500, 1000].find(t => t * scale >= 55) || 2000;
    ctx.strokeStyle = css('--border-subtle');
    for (let t = Math.ceil(start / tick) * tick; t <= end; t += tick) {
      if (x(t) < LABEL) continue;
      ctx.beginPath(); ctx.moveTo(x(t) + .5, HEADER); ctx.lineTo(x(t) + .5, height); ctx.stroke();
    }
    let y = HEADER - viewport.scrollTop;
    rows.forEach((row, index) => {
      const top = y; y += row.h;
      if (y < HEADER || top > height) return;
      if (index % 2) { ctx.fillStyle = css('--surface-disabled'); ctx.fillRect(0, top, width, row.h); }
      ctx.strokeStyle = css('--border-subtle'); ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
      ctx.save(); ctx.beginPath(); ctx.rect(LABEL, Math.max(HEADER, top), width - LABEL, Math.min(row.h, y - HEADER)); ctx.clip();
      const rankLane = row.rowType === 'rank';
      if (rankLane && !state.expanded.has(row.id)) {
        ctx.save(); ctx.strokeStyle = rgbaFromHex('#ffffff', .055);
        for (let track = 1; track < 3; track += 1) {
          const dividerY = top + TIMELINE_LAYOUT.rankTrackTop + row.trackOffsets[track] * TIMELINE_LAYOUT.rankTrackPitch - 1;
          ctx.beginPath(); ctx.moveTo(LABEL, dividerY + .5); ctx.lineTo(width, dividerY + .5); ctx.stroke();
        }
        ctx.restore();
      }
      drawDirectionalLaneEdgeTexture(ctx, row, x, { start, end }, top, row.h);
      const list = layouts.get(row.id);
      list.forEach(({event:e,track}) => {
        if (e.end < start || e.start > end) return;
        const source = e.sourceEventIds.map(id => byId.get(id));
        const related = !e.contextOnly && source.some(s => matches(s));
        const current = mbMatches(e);
        const barY = top + TIMELINE_LAYOUT.rankTrackTop + track * TIMELINE_LAYOUT.rankTrackPitch;
        const barX = x(Math.max(start, e.start)), naturalW = e.start===e.end ? 3 : Math.max(0,x(Math.min(end, e.end)) - barX);
        const barW = Math.max(minimumVisibleWidth(e), naturalW);
        const selected = e.id===state.eventKey || e.sourceEventIds.some(id=>selectedSourceIds.has(id));
        const dependencyLinked = !selected && e.sourceEventIds.some(id=>linkedSourceIds.has(id));
        const item = e, text = css('--foreground');
        ctx.save();
        ctx.globalAlpha = state.eventSourceIds.length
          ? (selected || dependencyLinked ? 1 : subduedAlpha)
          : (!current ? .17 : state.selection && !related ? .2 : 1);
        if (item.kind === 'hold') {
          const bandH = 16;
          const holdColor = color(item);
          ctx.save();
          roundedRect(ctx, barX, barY, barW, bandH, 4);
          ctx.fillStyle = rgbaFromHex(holdColor, selected ? .32 : .20);
          ctx.strokeStyle = selected ? '#ffffff' : rgbaFromHex(holdColor, .78);
          ctx.lineWidth = selected ? 1.6 : 1.1;
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = holdColor;
          ctx.beginPath();
          ctx.moveTo(barX, barY + bandH / 2);
          ctx.lineTo(barX + 4, barY + bandH / 2 - 4);
          ctx.lineTo(barX + 8, barY + bandH / 2);
          ctx.lineTo(barX + 4, barY + bandH / 2 + 4);
          ctx.closePath();
          ctx.fill();
          ctx.beginPath();
          const endX = barX + barW;
          ctx.fillStyle = COLORS.release;
          ctx.moveTo(endX, barY + bandH / 2);
          ctx.lineTo(endX - 4, barY + bandH / 2 - 4);
          ctx.lineTo(endX - 8, barY + bandH / 2);
          ctx.lineTo(endX - 4, barY + bandH / 2 + 4);
          ctx.closePath();
          ctx.fill();
          ctx.save();
          roundedRect(ctx, barX, barY, barW, bandH, 4);
          ctx.clip();
          ctx.fillStyle = text;
          ctx.font = `700 10px ${css('--font-mono') || 'monospace'}`;
          ctx.textAlign = 'left';
          ctx.textBaseline = 'middle';
          ctx.fillText(`${item.microbatchId??'step'} · L${item.layer}`, barX + 10, barY + bandH / 2);
          ctx.restore();
          ctx.restore();
          ctx.restore();
          hits.push({ x: Math.max(LABEL, barX), y: barY, w: Math.min(width, barX + barW) - Math.max(LABEL, barX), h: bandH, event: item });
          return;
        }
        pattern.drawTaskBar(ctx, { x: barX, y: barY, width: barW, height: TIMELINE_LAYOUT.eventHeight, baseColor: color(e), task: { label: e.label, opName: e.label }, isSelected: selected, isRelated: dependencyLinked || Boolean(state.selection && related && current), fontFamily: css('--font-mono') });
        if (e.kind === 'forward' || e.kind === 'backward') {
          drawChevronTrack(ctx, barX, barY, barW, TIMELINE_LAYOUT.eventHeight, e.kind, {
            step: 6, inset: .65, lineWidth: .55, radius: 3, color: 'rgba(255,255,255,.30)'
          });
          redrawTaskLabel(ctx, e, barX, barY, barW, TIMELINE_LAYOUT.eventHeight);
        }
        ctx.restore();
        hits.push({ x: Math.max(LABEL, barX), y: barY, w: Math.min(width, barX + barW) - Math.max(LABEL, barX), h: TIMELINE_LAYOUT.eventHeight, event: e });
      });
      ctx.restore();
      ctx.fillStyle = index % 2 ? css('--surface-1') : css('--background'); ctx.fillRect(0, top, LABEL, row.h);
      drawLaneConnector(ctx, row, top, row.h, css('--border-subtle'));
      const contentX = 12 + row.depth * 15 + (row.depth ? 13 : 0);
      const expandRect = drawRowExpand(ctx, row, contentX, top, row.h);
      const labelX = contentX + (expandRect ? 20 : 0);
      const lane = row;
      ctx.save(); ctx.beginPath(); ctx.rect(labelX, top, LABEL - labelX - 10, row.h); ctx.clip();
      if (rankLane) drawExpandedRankHeader(ctx, lane, labelX, top, row.h, LABEL - labelX - 10, css('--foreground'));
      else drawLaneHeader(ctx, lane, labelX, top, row.h, LABEL - labelX - 10, css('--foreground'));
      ctx.restore();
      hits.push({ x: 0, y: top, w: LABEL, h: row.h, row, expandRect, labelCenterY: top + headerCenter(row,row.h) });
    });
    drawDependencyLines(ctx);
    ctx.fillStyle = css('--surface-2'); ctx.fillRect(0, 0, width, HEADER);
    ctx.fillStyle = css('--foreground-muted'); ctx.font = `500 10px ${css('--font-mono')}`; ctx.textAlign = 'center';
    for (let t = Math.ceil(start / tick) * tick; t <= end; t += tick) if (x(t) >= LABEL + 10 && x(t) <= width - 12) ctx.fillText(`${t} ms`, x(t), 20);
    ctx.textAlign = 'left'; ctx.fillText(`${start.toFixed(2)} ms`, LABEL, 38);
    ctx.textAlign = 'right'; ctx.fillText(`${end.toFixed(2)} ms`, width - 16, 38);
    ctx.fillStyle = css('--surface-2'); ctx.fillRect(0, 0, LABEL, HEADER);
    ctx.fillStyle = css('--foreground'); ctx.textAlign = 'left'; ctx.font = `600 11px ${css('--font-sans')}`; ctx.fillText('泳道 / 对象', 12, 20);
    ctx.strokeStyle = css('--border-default'); ctx.beginPath(); ctx.moveTo(LABEL - .5, 0); ctx.lineTo(LABEL - .5, height); ctx.stroke();
    $('range-label').textContent = `${start}–${end} ms`;
  }
  function drawDependencyLines(ctx) {
    if(!state.eventSourceIds.length)return;
    const selectedIds=new Set(state.eventSourceIds), links=eventConnections(), eventHits=hits.filter(h=>h.event);
    const anchor=eventHits.find(h=>h.event.id===state.eventKey)||eventHits.find(h=>h.event.sourceEventIds.some(id=>selectedIds.has(id)));
    if(!anchor)return;
    const draw=(from,to,color)=>{
      const [a,b]=connectionEndpoints(from,to),bend=Math.max(18,Math.abs(b.x-a.x)*.32);
      ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.bezierCurveTo(a.x+bend,a.y,b.x-bend,b.y,b.x,b.y);
      ctx.strokeStyle=color;ctx.lineWidth=2;ctx.globalAlpha=.9;ctx.setLineDash([]);ctx.stroke();
      const size=5;ctx.beginPath();ctx.moveTo(b.x,b.y);ctx.lineTo(b.x-size,b.y-size*.7);ctx.lineTo(b.x-size,b.y+size*.7);ctx.closePath();ctx.fillStyle=color;ctx.fill();
    };
    const unique=(set)=>{
      const seen=new Set();return eventHits.filter(h=>h!==anchor&&h.event.sourceEventIds.some(id=>set.has(id))).filter(h=>{const key=h.event.id;if(seen.has(key))return false;seen.add(key);return true;}).slice(0,8);
    };
    ctx.save();ctx.beginPath();ctx.rect(LABEL,HEADER,width-LABEL,height-HEADER);ctx.clip();
    unique(links.upstream).forEach(hit=>draw(hit,anchor,css('--accent')));
    unique(links.downstream).forEach(hit=>draw(anchor,hit,css('--success')));
    ctx.restore();
  }
  function update() {
    buildRows();
    const mbList = microbatches(), dpLabel = state.dp === 'all' ? '全部 DP' : `DP${state.dp}`;
    const mbLabel = state.mb === 'all' ? `全部 ${mbList.length} MB / DP` : `${state.mb} 焦点`;
    $('training-context').textContent = `Step ${dataset.step} · ${dpLabel} · ${mbLabel}`;
    $('microbatch').innerHTML = `<option value="all">全部 MB · ${mbList.length} / DP</option>` + mbList.map(mb => `<option value="${mb}">${mb}</option>`).join('');
    $('microbatch').value = state.mb;
    $('stage').hidden = state.view === 'global'; $('stage').value = String(state.stage);
    const controlStates=[
      {id:'deployment',active:state.dp!=='all',label:'部署范围',value:state.dp==='all'?'全部 DP':`DP${state.dp}`,fallback:'全部 DP'},
      {id:'view',active:state.view!=='global',label:'时间窗口',value:{global:'完整 step',stage:'PP 阶段',object:'层对象',forward:'Rank 前向',backward:'Rank 反向'}[state.view],fallback:'完整 step'},
      {id:'microbatch',active:state.mb!=='all',label:'Microbatch 焦点',value:state.mb==='all'?`全部 ${mbList.length} MB / DP`:state.mb,fallback:'全部 MB'}
    ];
    controlStates.forEach(item=>{
      const control=$(item.id),copy=item.active
        ? `${item.label}已限定为 ${item.value}；选择“${item.fallback}”恢复默认。`
        : `${item.label}为默认状态：${item.value}。`;
      control.classList.toggle('is-active',item.active);control.dataset.filterTip=copy;control.title=copy;control.setAttribute('aria-description',copy);
    });
    const picked = byId.get(state.eventId);
    $('selection-actions').hidden = !state.selection && !picked && state.mb === 'all';
    $('selection-context').hidden = $('selection-actions').hidden;
    $('trace').hidden = !picked?.microbatchId || picked.microbatchId === state.mb;
    $('trace').textContent = picked?.microbatchId ? `追踪 ${picked.microbatchId}` : '追踪此 MB';
    $('clear').hidden = !state.selection && !picked;
    $('clear').textContent = state.selection ? '清除结构选择' : '清除事件选择';
    $('clear-mb').hidden = state.mb === 'all';
    if (state.selection) {
      const list = selectedEvents(), m = metrics(list);
      $('selection-copy').textContent = `${state.selection.label} · ${mbLabel} · ${list.length} 事件 · ${m.ranks} 记录 Ranks · 跨度 ${fmt(m.span)} · 区间并集 ${fmt(m.active)} · Σ ${fmt(m.sum)}`;
    } else {
      $('selection-copy').textContent = '';
    }
    scheduleDraw();
  }
  function post(type, data = {}) { frame.contentWindow.postMessage({ type, ...data }, location.origin); }
  function focusGraph(e) {
    const mapped=e.runtimeNodeIds.length ? e.runtimeNodeIds : e.isSummary||e.isUnion ? e.templateNodeIds : e.graphNodeIds;
    const nodeIds=[...new Set((mapped?.length?mapped:e.templateNodeIds)||[])];
    if (!nodeIds.length) return;
    // Lineage focus is visual-only. It cannot echo back as a structure filter
    // and remove unrelated Rank context from the timeline.
    post('csa-lineage-focus', { nodeIds, autoExpand: false });
  }
  function tipRow(key, value) { return `<div class="pto-swimlane-task-tooltip__row"><span class="pto-swimlane-task-tooltip__key">${escape(key)}</span><span class="pto-swimlane-task-tooltip__value">${escape(value)}</span></div>`; }
  function structureSummary(e, source) {
    if(e.runtimeNodeIds.length)return [...new Set(e.runtimeNodeIds.map(id=>runtimeNodes.get(id)?.label).filter(Boolean))].join(' → ');
    if(e.isSummary)return `PP${e.stage} ${e.kind==='forward'?'前向':'反向'}调度摘要；包含 ${source.length} 条阶段与算子明细`;
    if(e.isUnion)return `${kinds[e.kind]}的合并时间段；包含 ${source.length} 条事件`;
    if(e.layerRange)return `Decoder L${e.layerRange[0]}–L${e.layerRange[1]} 阶段概览`;
    if(e.layer!==null)return `Decoder L${e.layer} · ${String(e.label).replace(/ · L\d+(?: ∇)?$/,'')}`;
    if(e.kind==='optimizer')return '全模型参数更新';
    if(e.communicationType)return `${e.communicationType.toUpperCase()} 通信 · ${e.label}`;
    const names={'dv4/model/input_tokens':'输入 Token','dv4/model/token_embedding':'Token Embedding','dv4/model/previous_decoder_layers':'L0–L1 概览','dv4/model/remaining_decoder_layers':'L3–L60 概览','dv4/model/final_rmsnorm':'Final Norm','dv4/model/lm_head':'LM Head','dv4/model/output_tokens':'Main Logits','dv4/model/mtp':'MTP','dv4/model/loss':'Training Loss','dv4/model':'全模型'};
    return [...new Set(e.graphNodeIds.map(id=>names[id]).filter(Boolean))].join(' → ')||'阶段调度上下文';
  }
  function connectionText(ids) {
    const list=[...ids].map(id=>byId.get(id)).filter(Boolean),labels=[...new Set(list.map(e=>e.label))];
    return list.length?`${list.length} 条 · ${labels.slice(0,2).join(' / ')}${labels.length>2?' 等':''}`:'无直接关联';
  }
  function eventHtml(e) {
    const source = e.sourceEventIds.map(id => byId.get(id));
    const links=eventConnections(e.sourceEventIds);
    const scopeLabels={microbatch:'Microbatch',layer:'Layer',step:'完整 step','parameter-group':'参数组'};
    const groupLabels=[...new Set(e.parameterGroupIds.map(id=>parameterGroups.get(id)?.label).filter(Boolean))];
    const collective=collectiveGroups.get(e.collectiveGroupId);
    const shardText=e.shards.length?e.shards.map(shard=>`${optimizerStateGroups.get(shard.optimizerStateGroupId)?.label||shard.optimizerStateGroupId} ${shard.index+1}/${shard.count}`).join('；'):'';
    return `<div class="pto-swimlane-task-tooltip__title">${escape(e.isUnion ? `${kinds[e.kind]} · 活动并集` : e.label)}</div>` +
      tipRow('时间', `${fmt(e.start)} → ${fmt(e.end)} · ${fmt(e.end - e.start)}`) +
      (e.start === e.end ? tipRow('零时长记录', '保留原始 0 时长；最小可点击宽度不代表执行耗时') : '') +
      tipRow('位置', `${e.microbatchId ?? 'MB 未标注 / step 级'} · PP${e.stage} · R${e.rank} · DP${e.dp} / TP${e.tp}${e.ep === null ? '' : ` / EP${e.ep}`}`) +
      tipRow('类型', `${kinds[e.kind]} · ${relations[e.relation] ?? '结构未关联'}`) +
      tipRow('范围', scopeLabels[e.scope]||e.scope) +
      (e.participants.length ? tipRow('参与 Rank', e.participants.map(r => `R${r}`).join(', ')) : '') +
      (groupLabels.length ? tipRow('参数组', groupLabels.join('；')) : '') +
      (collective ? tipRow('通信组', `${collective.label} · ${collective.axes.join(' / ')}`) : '') +
      (shardText ? tipRow('状态分片', shardText) : '') +
      tipRow('模型位置', structureSummary(e,source)) +
      `<div class="dependency-summary"><span><i class="is-upstream"></i>上游：${escape(connectionText(links.upstream))}</span><span><i class="is-downstream"></i>下游：${escape(connectionText(links.downstream))}</span></div>` +
      (e.raw ? tipRow('原始记录', `index=${e.raw.sourceIndex}; ts=${e.raw.ts} μs; dur=${e.raw.dur} μs`) : '') +
      (e.microbatchId !== null ? `<div class="event-primary-action"><button class="btn btn-solid btn-sm" data-trace="${escape(e.microbatchId)}">聚焦 ${escape(e.microbatchId)}</button><span>其他 MB 保留为背景</span></div>` : '') +
      ((e.isUnion || e.isSummary) ? `<details class="source-events"><summary>${source.length} 条${e.isUnion?'合并事件':'组成明细'}</summary>${source.map(s => `<button class="btn btn-ghost btn-sm" data-source="${escape(s.id)}">${escape(`R${s.rank} · ${s.label} · ${fmt(s.start)}–${fmt(s.end)}`)}</button>`).join('')}</details>` : '');
  }
  function closePopover() { detail.hidden = true; detail.classList.remove('is-visible'); $('fidelity').setAttribute('aria-expanded', 'false'); $('hierarchy').setAttribute('aria-expanded', 'false'); }
  function showPopover(html, event, title='事件详情') {
    pattern.hideTooltip(hover.tooltip);
    detail.innerHTML = `<header class="panel-shell-header"><h3 class="panel-shell-title">${escape(title)}</h3><button class="btn btn-ghost btn-sm panel-shell-close" data-close aria-label="关闭详情">✕</button></header><div class="panel-shell-body">${html}</div>`;
    detail.hidden = false; detail.classList.add('is-visible');
    const bounds = detail.getBoundingClientRect();
    detail.style.left = `${Math.max(8, Math.min(innerWidth - bounds.width - 12, event.clientX + 12))}px`;
    detail.style.top = `${Math.max(8, Math.min(innerHeight - bounds.height - 12, event.clientY + 12))}px`;
  }
  function selectEvent(e, event) {
    state.selection = null;
    state.eventId = e.sourceEventIds[0];
    state.eventKey = e.id;
    state.eventSourceIds = [...e.sourceEventIds];
    // Selection, MB focus, deployment and time window are independent dimensions.
    // In particular, a union click may not widen the structure selection to its first event.
    update(); focusGraph(e); showPopover(eventHtml(e), event);
  }
  function hitAt(e) {
    const rect = canvas.getBoundingClientRect(), x = e.clientX - rect.left, y = e.clientY - rect.top;
    if (y < HEADER) return x < LABEL ? {header:true} : null;
    return [...hits].reverse().find(h => x >= h.x && x <= h.x + h.w && y >= h.y && y <= h.y + h.h);
  }
  const kindHelp = {
    forward:'前向：当前 MB 的阶段计算或 L2 算子示例。',
    backward:'反向：为当前 MB 计算梯度。',
    comm:'通信：PP 传输、EP 分发 / 合并及梯度同步。',
    optimizer:'优化器：全部 MB 梯度同步后更新参数。',
    hold:'激活驻留：按 MB 分轨，不代表设备忙碌。',
    wait:'已记录的依赖等待；空白不等于空闲。'
  };
  function headerHtml(hit) {
    if(hit.header) return 'DP → PP → Rank → 活动类别；横轴为 step 时间。';
    const row=hit.row, rankId=row.id.match(/\/r(\d+)$/), stageId=row.id.match(/\/pp(\d+)$/);
    let help=kindHelp[row.id.split('/').at(-1)] || '仅 L2 提供典型算子；其余层使用阶段概览。';
    if(rankId) help=`${row.label}：逻辑训练进程 · ${row.meta}；点击展开类别。`;
    else if(stageId) { const stage=STAGES.find(s=>s.id===Number(stageId[1])); help=`${row.label}：流水线阶段，负责 L${stage.first}–L${stage.last}。`; }
    else if(/^dp\d+$/.test(row.id)) help=`${row.label}：独立处理数据的并行副本，包含各 PP / Rank。`;
    return escape(help);
  }
  const hoverHtml = hit => hit.event ? escape(`${hit.event.microbatchId??'step'} · ${hit.event.label} · ${fmt(hit.event.end-hit.event.start)}；点击详情`) : headerHtml(hit);
  const hover = pattern.initHoverTooltip({ root: canvas, targets: [canvas], appendTo: $('timeline-pane'), bounds: $('timeline-pane'), tooltipOptions:{className:'training-brief-tip'}, getTask: (_, e) => hitAt(e), getTooltipHtml: hoverHtml });
  pattern.initHoverTooltip({root:$('time-mode'),targets:'[data-mode]',appendTo:$('timeline-pane'),bounds:$('timeline-pane'),tooltipOptions:{className:'training-brief-tip'},getTask:target=>({mode:target.dataset.mode}),getTooltipHtml:task=>task.mode==='events'?'逐条展示事件；Rank 概览显示 F/B 摘要。':'合并重叠区间；不跨 Rank 或 MB，不代表利用率。'});
  pattern.initHoverTooltip({root:$('timeline-pane'),targets:'#deployment, #view, #microbatch',appendTo:$('timeline-pane'),bounds:$('timeline-pane'),tooltipOptions:{className:'training-brief-tip'},getTask:target=>target,getTooltipHtml:target=>escape(target.dataset.filterTip||'')});
  canvas.addEventListener('pointermove', e => {
    const hit = hitAt(e); canvas.style.cursor = hit?.event || hit?.row?.expandable ? 'pointer' : 'default';
    if (!detail.hidden || !hit) pattern.hideTooltip(hover.tooltip);
    else pattern.showTooltip(hover.tooltip, hit, e, { bounds: $('timeline-pane'), getTooltipHtml: hoverHtml });
  });
  canvas.addEventListener('click', e => {
    const hit = hitAt(e); if (!hit || !hit.event && !hit.row?.expandable) return clearEvent();
    if (hit.row?.expandable) {
      const id = hit.row.id; state.expanded.has(id) ? state.expanded.delete(id) : state.expanded.add(id);
      closePopover(); update();
    } else if (hit.event) selectEvent(hit.event, e);
  });
  canvas.addEventListener('keydown', e => {
    if (!['ArrowRight', 'ArrowLeft', 'Enter'].includes(e.key)) return;
    e.preventDefault();
    const list = scopeEvents().filter(mbMatches).sort((a, b) => a.start - b.start);
    if (!list.length) return;
    let i = list.findIndex(e => e.id === state.eventId);
    i = e.key === 'ArrowLeft' ? (i - 1 + list.length) % list.length : e.key === 'ArrowRight' ? (i + 1) % list.length : Math.max(0, i);
    const rect = canvas.getBoundingClientRect(); selectEvent(list[i], { clientX: rect.left + LABEL + 20, clientY: rect.top + 80 });
  });
  detail.addEventListener('click', e => {
    if (e.target.closest('[data-close]')) closePopover();
    const button = e.target.closest('[data-source]'); if (button) selectEvent(byId.get(button.dataset.source), e);
    const trace = e.target.closest('[data-trace]'); if (trace) { state.mb = trace.dataset.trace; closePopover(); update(); }
  });
  document.addEventListener('pointerdown', e => { if (!detail.contains(e.target) && !e.target.closest('#fidelity, #hierarchy, #timeline')) closePopover(); });
  function clear() { state.selection = null; state.eventId = null; state.eventKey=null; state.eventSourceIds=[]; post('csa-source-clear'); closePopover(); update(); }
  function clearEvent() { state.eventId=null;state.eventKey=null;state.eventSourceIds=[];if(!state.selection)post('csa-source-clear');closePopover();update(); }
  document.addEventListener('keydown', e => { if (e.key === 'Escape') clear(); });
  $('clear').onclick = clear;
  $('trace').onclick = () => { const mb = byId.get(state.eventId)?.microbatchId; if (mb) state.mb = mb; closePopover(); update(); };
  $('clear-mb').onclick = () => { state.mb = 'all'; closePopover(); update(); };
  $('deployment').onchange = e => { state.dp = e.target.value; state.eventId = null; state.eventKey=null; state.eventSourceIds=[]; if (!microbatches().includes(state.mb)) state.mb = 'all'; viewport.scrollTop = 0; closePopover(); update(); };
  $('microbatch').onchange = e => { state.mb = e.target.value; state.eventId = null; state.eventKey=null; state.eventSourceIds=[]; closePopover(); update(); };
  $('time-mode').onclick = e => {
    const b = e.target.closest('[data-mode]'); if (!b) return; state.mode = b.dataset.mode;
    $('time-mode').querySelectorAll('button').forEach(button => { button.classList.toggle('is-selected', button === b); button.setAttribute('aria-pressed', String(button === b)); }); closePopover(); update();
  };
  $('view').onchange = e => {
    state.view = e.target.value; viewport.scrollLeft = 0; viewport.scrollTop = 0;
    expandScope();
    closePopover(); update();
  };
  function expandScope() {
    if (state.view === 'global') return;
    scopeEvents().forEach(e => {
      const pp = `dp${e.dp}/pp${e.stage}`;
      state.expanded.add(`dp${e.dp}`); state.expanded.add(pp);
      if (state.view === 'object') state.expanded.add(`${pp}/objects`);
      else state.expanded.add(`${pp}/r${e.rank}`);
    });
  }
  $('stage').onchange = e => { state.stage = Number(e.target.value); viewport.scrollLeft = 0; viewport.scrollTop = 0; expandScope(); closePopover(); update(); };
  function zoom(value) {
    const old = ZOOMS[state.zoom]; state.zoom = Math.max(0, Math.min(4, value));
    $('zoom').value = state.zoom; $('zoom-value').textContent = `${ZOOMS[state.zoom]}×`;
    $('zoom-out').disabled = state.zoom === 0; $('zoom-in').disabled = state.zoom === 4;
    $('timeline-extent').style.width = `${LABEL + (width - LABEL) * ZOOMS[state.zoom]}px`;
    viewport.scrollLeft = (viewport.scrollLeft + (width - LABEL) / 2) * ZOOMS[state.zoom] / old - (width - LABEL) / 2;
    closePopover(); scheduleDraw();
  }
  $('zoom').oninput = e => zoom(Number(e.target.value)); $('zoom-out').onclick = () => zoom(state.zoom - 1); $('zoom-in').onclick = () => zoom(state.zoom + 1);
  $('graph-level').onclick = async e => {
    const b = e.target.closest('[data-level]'); if (!b) return;
    $('graph-level').querySelectorAll('button').forEach(button => { button.classList.toggle('is-selected', button === b); button.setAttribute('aria-pressed', String(button === b)); });
    post('csa-root-fold', { folded: b.dataset.level === 'network' });
    if (b.dataset.level !== 'network') post(b.dataset.level === 'l2' ? 'csa-semantic-expand-all' : 'csa-semantic-collapse-all');
  };
  $('graph-fit').onclick = async () => (await frame.contentWindow.csaImplementationController)?.fit({ whole: true });
  $('theme').onclick = () => {
    const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = theme; post('pto-preview-theme', { theme }); syncColors(); scheduleDraw(); closePopover();
  };
  $('fidelity').onclick = e => {
    if ($('fidelity').getAttribute('aria-expanded') === 'true') return closePopover();
    closePopover(); $('fidelity').setAttribute('aria-expanded', 'true');
    const c=dataset.training, ref=dataset.calibration;
    showPopover(`<dl class="simulation-facts"><dt>模型</dt><dd>${escape(dataset.label)}</dd><dt>批次</dt><dd>${c.globalBatchSize} ÷ ${c.dp} DP ÷ ${c.microBatchSize} MBS = ${c.microbatchesPerDP} MB / DP</dd><dt>覆盖</dt><dd>L${c.detailLayer} 典型算子，其余阶段为概览</dd><dt>调度</dt><dd>1F1B · PP / EP 通信示例</dd><dt>Optimizer</dt><dd>ST 的 5 类运行时阶段；参数组与状态分片为模拟</dd><dt>参考</dt><dd>ST 的 32 对 F/B 记录，时长比 ${ref.backwardForwardRatio.toFixed(3)}</dd></dl><p class="simulation-note">耗时与部署均为模拟，不代表 DeepSeek 实际性能。</p><nav class="simulation-links" aria-label="数据来源"><a class="btn btn-ghost btn-sm" href="${escape(dataset.model.source)}" target="_blank" rel="noopener">模型配置 ↗</a><a class="btn btn-ghost btn-sm" href="${escape(ref.source)}" target="_blank" rel="noopener">训练参考 ↗</a><a class="btn btn-ghost btn-sm" href="./data/training-simulation.config.json" target="_blank" rel="noopener">模拟参数 ↗</a><a class="btn btn-ghost btn-sm" href="./training-timeline.schema.json" target="_blank" rel="noopener">数据契约 ↗</a></nav>`, e, '模拟配置与依据');
  };
  $('hierarchy').onclick = e => {
    if ($('hierarchy').getAttribute('aria-expanded') === 'true') return closePopover();
    closePopover(); $('hierarchy').setAttribute('aria-expanded', 'true');
    showPopover(`<div class="pto-swimlane-task-tooltip__title">先看完整 step，再追踪 MB</div><p>DP → PP stage → Rank → 活动类别。默认全部 MB；Rank 概览的三条轨道依次为计算、通信、其他活动，展开可查看分类明细。</p><p>MB 是跨泳道焦点，不是资源树的父节点。点击事件只查看详情和定位结构；显式追踪 MB 后，其他 MB 与未标注事件保留为背景。清除 MB 焦点恢复全部 MB。</p><p>部署范围、时间窗口、MB 焦点和结构选择独立。并集先确定选中集合，再按 Rank、类别和 MB 合并。记录空白不等于设备空闲；等待和驻留不计作设备忙碌。</p>`, e);
  };
  window.addEventListener('message', e => {
    if (e.source !== frame.contentWindow || e.origin !== location.origin) return;
    if (e.data?.type === 'csa-operator-ready') { post('pto-preview-theme', { theme: document.documentElement.dataset.theme }); post('csa-runtime-graph', { runtimeGraph: dataset.runtimeGraph }); post('csa-source-clear'); }
    if (e.data?.source === 'expansion') return;
    if (e.data?.type === 'csa-operator-clear') return clear();
    if (e.data?.type !== 'csa-operator-select') return;
    if (!e.data.nodeId) return clear();
    state.selection = { nodeId: e.data.nodeId, label: String(e.data.label || e.data.nodeId).split('\n')[0] + (e.data.nodeId.startsWith('dv4/layer/') || e.data.nodeId.includes('csa') ? ' · 结构模板' : '') };
    if (state.view !== 'global') {
      const matched = deploymentEvents().filter(e => matches(e));
      if (matched.length && !matched.some(e => e.stage === state.stage)) state.stage = matched[0].stage;
      expandScope();
    }
    state.eventId = null; state.eventKey=null; state.eventSourceIds=[]; closePopover(); update();
  });
  viewport.addEventListener('scroll', () => { pattern.hideTooltip(hover.tooltip); scheduleDraw(); }, { passive: true });
  new ResizeObserver(scheduleDraw).observe(viewport);
  activateDataset(); syncColors(); update(); zoom(0);
  // Read-only inspection surface for deterministic data / UI smoke checks.
  window.TrainingStructureDemo = { dataset, layout: TIMELINE_LAYOUT, layoutEvents, eventConnections, connectionEndpoints, structureSummary, focusGraph, get events() { return events; }, metrics, union, range, displayEvents, getState: () => ({ ...state, expanded: [...state.expanded] }), selectedEvents, getRows: () => rows, getHits: () => hits };
})();
window.TrainingStructureReady.catch(error => {
  document.getElementById('selection-context').hidden = false;
  document.getElementById('selection-copy').textContent = '训练数据加载或校验失败：' + error.message;
  document.getElementById('selection-copy').setAttribute('role','alert');
  console.error(error);
});
