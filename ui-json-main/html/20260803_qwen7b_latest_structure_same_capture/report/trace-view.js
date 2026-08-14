(function exposeTraceView(global) {
  "use strict";

  const LABEL_WIDTH = 236;
  const LANE_HEIGHT = 22;
  const PROCESS_HEIGHT = 24;
  const RULER_HEIGHT = 30;
  const ZOOM_LEVELS = [1, 2, 4, 8, 16, 32, 64];
  const TARGET_EVENT_VIEWPORT_RATIO = 0.5;
  const FOCUS_PAN_VIEWPORTS = 8;
  const MAX_TRACK_WIDTH = 16000;
  const PROCESS_COLORS = ["#4e8fda", "#48a868", "#d6a84b", "#db6d70", "#4fb2aa", "#9b7bd4", "#c6794f"];

  function number(value, fallback = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function keyFor(event) {
    return `${event.pid}:${event.tid}`;
  }

  function flowKey(event) {
    return `${event.name || "flow"}|${event.id ?? event.bind_id ?? ""}`;
  }

  function hash(value) {
    let result = 0;
    for (const character of String(value)) result = ((result << 5) - result + character.charCodeAt(0)) | 0;
    return Math.abs(result);
  }

  function formatDuration(us) {
    if (us >= 1_000_000) return `${(us / 1_000_000).toFixed(2)} s`;
    if (us >= 1000) return `${(us / 1000).toFixed(us >= 10_000 ? 1 : 2)} ms`;
    if (us >= 1) return `${us.toFixed(us >= 100 ? 0 : 1)} us`;
    return `${(us * 1000).toFixed(0)} ns`;
  }

  function matchesNode(event, nodeId) {
    return Boolean(
      nodeId
      && event?.ownerNodeId
      && (event.ownerNodeId === nodeId || event.ownerNodeId.startsWith(`${nodeId}/`)),
    );
  }

  function clampRangeAround(center, span, bounds) {
    const boundsSpan = Math.max(0.001, bounds.end - bounds.start);
    const nextSpan = Math.max(0.001, Math.min(boundsSpan, span));
    let start = center - nextSpan / 2;
    let end = center + nextSpan / 2;
    if (start < bounds.start) {
      end += bounds.start - start;
      start = bounds.start;
    }
    if (end > bounds.end) {
      start -= end - bounds.end;
      end = bounds.end;
    }
    return { start: Math.max(bounds.start, start), end: Math.min(bounds.end, end) };
  }

  function focusRangeForEvent(event, bounds, targetRatio = TARGET_EVENT_VIEWPORT_RATIO) {
    if (!event) return null;
    const visibleStart = Math.max(bounds.start, event.ts);
    const visibleEnd = Math.min(bounds.end, event.end);
    if (visibleEnd < visibleStart) return null;
    const duration = Math.max(0.001, visibleEnd - visibleStart);
    const center = (visibleStart + visibleEnd) / 2;
    return clampRangeAround(center, duration / targetRatio, bounds);
  }

  function representativeEventForNode(events, nodeId, bounds) {
    if (!nodeId) return null;
    const visible = events.filter((event) => (
      matchesNode(event, nodeId)
      && event.ts < bounds.end
      && event.end >= bounds.start
    ));
    const exact = visible.filter((event) => event.ownerNodeId === nodeId);
    return (exact.length ? exact : visible).sort((left, right) => (
      right.dur - left.dur || left.ts - right.ts
    ))[0] || null;
  }

  function normalizeEvents(traceDocument) {
    if (Array.isArray(traceDocument)) return traceDocument;
    return Array.isArray(traceDocument?.traceEvents) ? traceDocument.traceEvents : [];
  }

  function parseTrace(traceDocument, bindings = []) {
    const rawEvents = normalizeEvents(traceDocument);
    const bindingBySourceIndex = new Map(bindings.map((binding) => [Number(binding.raw_source_event_index), binding]));
    const processNames = new Map();
    const threadNames = new Map();
    const processOrder = new Map();
    const threadOrder = new Map();
    const durations = [];
    const startsByKey = new Map();
    const finishes = [];
    let nextProcessOrder = 0;
    let nextThreadOrder = 0;

    rawEvents.forEach((event, sourceIndex) => {
      const pid = event.pid ?? 0;
      const tid = event.tid ?? 0;
      const laneKey = keyFor({ pid, tid });
      if (!processOrder.has(pid)) processOrder.set(pid, nextProcessOrder++);
      if (!threadOrder.has(laneKey)) threadOrder.set(laneKey, nextThreadOrder++);
      if (event.ph === "M" && event.name === "process_name" && event.args?.name) {
        processNames.set(pid, String(event.args.name));
      }
      if (event.ph === "M" && event.name === "thread_name" && event.args?.name) {
        threadNames.set(laneKey, String(event.args.name));
      }
      if (event.ph === "X") {
        const ts = number(event.ts, NaN);
        const dur = Math.max(0, number(event.dur, 0));
        const binding = bindingBySourceIndex.get(sourceIndex) || null;
        if (Number.isFinite(ts)) durations.push({
          ...event,
          pid,
          tid,
          ts,
          dur,
          end: ts + dur,
          sourceIndex,
          laneKey,
          binding,
          ownerNodeId: binding?.node_id || "",
        });
      } else if (event.ph === "s") {
        const ts = number(event.ts, NaN);
        if (!Number.isFinite(ts)) return;
        const start = { ...event, pid, tid, ts, sourceIndex, laneKey };
        const key = flowKey(event);
        if (!startsByKey.has(key)) startsByKey.set(key, []);
        startsByKey.get(key).push(start);
      } else if (event.ph === "f") {
        const ts = number(event.ts, NaN);
        if (Number.isFinite(ts)) finishes.push({ ...event, pid, tid, ts, sourceIndex, laneKey });
      }
    });

    startsByKey.forEach((starts) => starts.sort((left, right) => left.ts - right.ts));
    const streamSequenceCounts = new Map();
    [...durations]
      .sort((left, right) => left.pid - right.pid || left.tid - right.tid || left.ts - right.ts || left.sourceIndex - right.sourceIndex)
      .forEach((event) => {
        const nextSequence = (streamSequenceCounts.get(event.laneKey) || 0) + 1;
        streamSequenceCounts.set(event.laneKey, nextSequence);
        event.streamSequence = nextSequence;
      });
    const pairedFlows = finishes.flatMap((finish) => {
      const starts = startsByKey.get(flowKey(finish)) || [];
      let low = 0;
      let high = starts.length - 1;
      let match = null;
      while (low <= high) {
        const middle = (low + high) >> 1;
        if (starts[middle].ts <= finish.ts) {
          match = starts[middle];
          low = middle + 1;
        } else {
          high = middle - 1;
        }
      }
      return match ? [{ key: flowKey(finish), start: match, finish }] : [];
    });

    const boundDurationsByLane = new Map();
    durations.forEach((event) => {
      if (!event.ownerNodeId) return;
      if (!boundDurationsByLane.has(event.laneKey)) boundDurationsByLane.set(event.laneKey, []);
      boundDurationsByLane.get(event.laneKey).push(event);
    });

    function ownerForEndpoint(endpoint) {
      const candidates = boundDurationsByLane.get(endpoint.laneKey) || [];
      let owner = null;
      candidates.forEach((event) => {
        if (event.ts > endpoint.ts || event.end < endpoint.ts) return;
        if (!owner || event.dur < owner.dur) owner = event;
      });
      return owner;
    }

    const flows = pairedFlows.map((flow) => ({
      ...flow,
      startOwnerEvent: ownerForEndpoint(flow.start),
      finishOwnerEvent: ownerForEndpoint(flow.finish),
    }));

    const minTs = Math.min(...durations.map((event) => event.ts));
    const maxTs = Math.max(...durations.map((event) => event.end));
    return { rawEvents, durations, flows, processNames, threadNames, processOrder, threadOrder, minTs, maxTs };
  }

  function createTraceView(container, traceDocument, options = {}) {
    if (!container) throw new Error("TraceView container is required");
    const trace = parseTrace(traceDocument, options.bindings || []);
    if (!trace.durations.length) throw new Error("TraceView has no duration events");

    const state = {
      language: options.language === "zh" ? "zh" : "en",
      mode: Number.isFinite(options.focusStart) && Number.isFinite(options.focusEnd) ? "step" : "full",
      showSelectedFlows: false,
      zoomIndex: 0,
      focusRange: null,
      focusEvent: null,
      centerSelectionAfterRender: false,
      resizeTimer: 0,
      scrollFrame: 0,
      tooltip: null,
      scroller: null,
      chart: null,
      flowCanvas: null,
      selectionLayer: null,
      selectionSummary: null,
      timeSelection: null,
      pendingScrollAnchor: null,
      trackWidth: 0,
      lanes: [],
      laneY: new Map(),
      rangeStart: trace.minTs,
      rangeEnd: trace.maxTs,
      selectedNodeId: options.selectedNodeId || "",
    };
    const swimlaneHelper = global.PtoSwimlaneTaskPattern;
    const timeSelectionHelper = global.PtoTimelineTimeSelectionPattern;
    if (!timeSelectionHelper) throw new Error("timeline-time-selection pattern is unavailable");
    const taskColormap = swimlaneHelper?.createTaskColormap();

    const labels = {
      en: {
        step: options.stepLabel || "Representative step",
        full: "Full capture",
        events: "events",
        tracks: "tracks",
        zoomOut: "Zoom out",
        zoomIn: "Zoom in",
        reset: "Reset zoom",
        fitAll: "Fit all",
        focusSelection: "Zoom to selection",
        laneInteraction: "Timeline lane. Drag to select time, click an event to select it, use Control or Command plus wheel to zoom, and Left or Right Arrow to pan.",
      },
      zh: {
        step: options.stepLabelZh || "代表步骤",
        full: "完整采集",
        events: "个事件",
        tracks: "条泳道",
        zoomOut: "缩小",
        zoomIn: "放大",
        reset: "重置缩放",
        fitAll: "缩小到全部",
        focusSelection: "聚焦选中项",
        laneInteraction: "时间泳道。拖拽选择时间，单击选择事件，按住 Ctrl 或 Command 并滚动进行缩放，使用左右方向键平移。",
      },
    };
    const text = (key) => labels[state.language][key] || labels.en[key] || key;

    const toolbarMarkup = `
      <div class="raw-trace-toolbar">
        <div class="trace-zoom-controls" role="group">
          <button type="button" data-trace-zoom="out" aria-label="Zoom out" data-report-tooltip="Zoom out">-</button>
          <button type="button" data-trace-zoom="reset">100%</button>
          <button type="button" data-trace-zoom="in" aria-label="Zoom in" data-report-tooltip="Zoom in">+</button>
          <button class="trace-zoom-action" type="button" data-trace-action="fit" aria-label="Fit all" data-report-tooltip="Fit all">
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <path d="M6 2H2v4M10 2h4v4M14 10v4h-4M6 14H2v-4"></path>
            </svg>
          </button>
          <button class="trace-zoom-action" type="button" data-trace-action="focus" aria-label="Zoom to selection" data-report-tooltip="Zoom to selection" disabled>
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <circle cx="7" cy="7" r="4.5"></circle>
              <path d="m10.5 10.5 3 3M7 4.5v5M4.5 7h5"></path>
            </svg>
          </button>
        </div>
      </div>`;
    const hasExternalToolbar = Boolean(options.toolbarContainer);
    const toolbarRoot = options.toolbarContainer || container;
    if (hasExternalToolbar) toolbarRoot.innerHTML = toolbarMarkup;
    container.innerHTML = `
      <div class="raw-trace-view${hasExternalToolbar ? " has-external-toolbar" : ""}">
        ${hasExternalToolbar ? "" : toolbarMarkup}
        <div class="raw-trace-scroller"></div>
        <div class="raw-trace-tooltip" role="tooltip" hidden></div>
      </div>`;
    state.scroller = container.querySelector(".raw-trace-scroller");
    state.tooltip = container.querySelector(".raw-trace-tooltip");
    state.selectionSummary = timeSelectionHelper.createSelectionSummary(
      container.querySelector(".raw-trace-view"),
      { labelWidth: LABEL_WIDTH },
    );
    state.scroller.addEventListener("scroll", () => {
      updateTimeSelectionVisual();
      if (state.scrollFrame) return;
      state.scrollFrame = global.requestAnimationFrame(() => {
        state.scrollFrame = 0;
        notifyVisibleRange("scroll");
      });
    });
    state.scroller.addEventListener("pointerdown", (event) => {
      if (event.target === state.scroller) options.onViewportInteraction?.({ source: "scrollbar" });
    });
    state.scroller.addEventListener("wheel", (event) => {
      if (!event.ctrlKey && !event.metaKey && (event.deltaX || event.shiftKey)) {
        options.onViewportInteraction?.({ source: "pan" });
      }
    }, { passive: true });

    function baseRange() {
      if (state.mode === "step") {
        return {
          start: Math.max(trace.minTs, number(options.focusStart, trace.minTs)),
          end: Math.min(trace.maxTs, number(options.focusEnd, trace.maxTs)),
        };
      }
      return { start: trace.minTs, end: trace.maxTs };
    }

    function activeRange() {
      const base = baseRange();
      if (!state.focusRange) return base;
      const center = (state.focusRange.start + state.focusRange.end) / 2;
      return clampRangeAround(center, state.focusRange.end - state.focusRange.start, base);
    }

    function renderedRange() {
      const visible = activeRange();
      if (!state.focusRange) return { ...visible, trackMultiplier: 1 };
      const base = baseRange();
      const visibleSpan = Math.max(0.001, visible.end - visible.start);
      const center = state.focusEvent
        ? (state.focusEvent.ts + state.focusEvent.end) / 2
        : (visible.start + visible.end) / 2;
      const expanded = clampRangeAround(center, visibleSpan * FOCUS_PAN_VIEWPORTS, base);
      return {
        ...expanded,
        trackMultiplier: Math.max(1, (expanded.end - expanded.start) / visibleSpan),
      };
    }

    function selectedFlows() {
      if (!state.selectedNodeId || !state.showSelectedFlows) return [];
      return trace.flows.filter((flow) => (
        matchesNode(flow.startOwnerEvent, state.selectedNodeId)
        || matchesNode(flow.finishOwnerEvent, state.selectedNodeId)
      ));
    }

    function activateSelectionFocus() {
      state.zoomIndex = 0;
      state.focusRange = null;
      state.focusEvent = representativeEventForNode(trace.durations, state.selectedNodeId, baseRange());
      if (state.focusEvent) {
        state.focusRange = focusRangeForEvent(state.focusEvent, baseRange());
        const base = baseRange();
        if (state.focusRange && state.focusRange.end - state.focusRange.start >= base.end - base.start - 0.001) {
          state.focusRange = null;
        }
      }
      state.centerSelectionAfterRender = Boolean(state.focusEvent);
    }

    function effectiveZoom() {
      const base = baseRange();
      const active = activeRange();
      const rangeZoom = (base.end - base.start) / Math.max(0.001, active.end - active.start);
      return rangeZoom * ZOOM_LEVELS[state.zoomIndex];
    }

    function formatZoom(scale) {
      if (scale >= 10) return `${scale.toFixed(scale < 100 ? 1 : 0)}x`;
      return `${Math.round(scale * 100)}%`;
    }

    function processColor(pid) {
      return PROCESS_COLORS[hash(trace.processNames.get(pid) || pid) % PROCESS_COLORS.length];
    }

    function updateToolbar(visibleEventCount) {
      toolbarRoot.querySelectorAll("[data-trace-range]").forEach((button) => {
        const selected = button.dataset.traceRange === state.mode;
        button.classList.toggle("is-selected", selected);
        button.setAttribute("aria-pressed", String(selected));
        button.textContent = text(button.dataset.traceRange);
      });
      const count = toolbarRoot.querySelector(".trace-view-count");
      if (count) count.textContent = `${visibleEventCount} ${text("events")} / ${state.lanes.length} ${text("tracks")}`;
      const zoom = effectiveZoom();
      const reset = toolbarRoot.querySelector('[data-trace-zoom="reset"]');
      const out = toolbarRoot.querySelector('[data-trace-zoom="out"]');
      const inside = toolbarRoot.querySelector('[data-trace-zoom="in"]');
      const fit = toolbarRoot.querySelector('[data-trace-action="fit"]');
      const focus = toolbarRoot.querySelector('[data-trace-action="focus"]');
      const focusEvent = representativeEventForNode(trace.durations, state.selectedNodeId, baseRange());
      reset.textContent = formatZoom(zoom);
      reset.dataset.reportTooltip = text("reset");
      out.dataset.reportTooltip = text("zoomOut");
      out.setAttribute("aria-label", text("zoomOut"));
      inside.dataset.reportTooltip = text("zoomIn");
      inside.setAttribute("aria-label", text("zoomIn"));
      fit.dataset.reportTooltip = text("fitAll");
      fit.setAttribute("aria-label", text("fitAll"));
      focus.dataset.reportTooltip = text("focusSelection");
      focus.setAttribute("aria-label", text("focusSelection"));
      out.disabled = state.zoomIndex === 0 && !state.focusRange;
      inside.disabled = state.zoomIndex === ZOOM_LEVELS.length - 1;
      fit.disabled = state.zoomIndex === 0 && !state.focusRange;
      focus.disabled = !focusEvent;
      options.onCaption?.({ eventCount: visibleEventCount, trackCount: state.lanes.length, mode: state.mode });
    }

    function laneLabel(lane) {
      return trace.threadNames.get(lane.key) || `Thread ${lane.tid}`;
    }

    function buildLanes(events, flows) {
      const laneKeys = new Set(events.map((event) => event.laneKey));
      flows.forEach((flow) => {
        laneKeys.add(flow.start.laneKey);
        laneKeys.add(flow.finish.laneKey);
      });
      return [...laneKeys].map((key) => {
        const [pidText, tidText] = key.split(":");
        return { key, pid: number(pidText), tid: number(tidText) };
      }).sort((left, right) => (
        (trace.processOrder.get(left.pid) ?? 9999) - (trace.processOrder.get(right.pid) ?? 9999)
        || (trace.threadOrder.get(left.key) ?? 9999) - (trace.threadOrder.get(right.key) ?? 9999)
      ));
    }

    function rulerTicks(start, end) {
      return [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
        const tick = document.createElement("span");
        tick.className = "raw-trace-ruler-tick";
        tick.style.left = `${ratio * 100}%`;
        tick.textContent = ratio === 0 ? "0 us" : formatDuration((end - start) * ratio);
        return tick;
      });
    }

    function configureCanvas(canvas, width, height) {
      const ratio = Math.max(1, global.devicePixelRatio || 1);
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      const context = canvas.getContext("2d");
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, width, height);
      return context;
    }

    function drawLane(canvas, laneEvents, start, end, width, lane) {
      const context = configureCanvas(canvas, width, LANE_HEIGHT);
      const span = Math.max(0.001, end - start);
      context.strokeStyle = "rgba(127, 132, 142, 0.13)";
      [0.25, 0.5, 0.75].forEach((ratio) => {
        context.beginPath();
        context.moveTo(Math.round(width * ratio) + 0.5, 0);
        context.lineTo(Math.round(width * ratio) + 0.5, LANE_HEIGHT);
        context.stroke();
      });
      const eventGeometry = laneEvents.map((event) => {
        const x = Math.max(0, ((event.ts - start) / span) * width);
        const eventWidth = Math.max(1, (event.dur / span) * width);
        return { event, x, drawWidth: Math.max(1, Math.min(eventWidth, width - x)) };
      });
      const labelEvents = new Set();
      const occupiedLabelRanges = [];
      [...eventGeometry]
        .filter(({ drawWidth }) => drawWidth >= 36)
        .sort((left, right) => right.drawWidth - left.drawWidth || left.x - right.x)
        .forEach(({ event, x, drawWidth }) => {
          const range = { start: x, end: x + drawWidth };
          const collides = occupiedLabelRanges.some((occupied) => (
            range.start < occupied.end && range.end > occupied.start
          ));
          if (collides) return;
          labelEvents.add(event);
          occupiedLabelRanges.push(range);
        });
      let linkedCount = 0;
      const renderGeometry = [...eventGeometry].sort((left, right) => (
        Number(labelEvents.has(left.event)) - Number(labelEvents.has(right.event))
        || left.x - right.x
        || left.drawWidth - right.drawWidth
      ));
      const labelOverlays = [];
      canvas.__traceHits = renderGeometry.map(({ event, x, drawWidth }) => {
        const isTimeSelected = Boolean(
          state.timeSelection?.kind === "event"
          && state.timeSelection.event?.sourceIndex === event.sourceIndex,
        );
        const isLinked = Boolean(
          state.selectedNodeId
          && event.ownerNodeId
          && (event.ownerNodeId === state.selectedNodeId || event.ownerNodeId.startsWith(`${state.selectedNodeId}/`)),
        );
        if (isLinked) linkedCount += 1;
        context.globalAlpha = state.selectedNodeId
          ? (isLinked || isTimeSelected ? 1 : 0.12)
          : (event.cat === "cpu_op" ? 0.82 : 0.92);
        const eventName = event.name || "event";
        const displayName = `#${event.streamSequence} · ${eventName}`;
        const task = {
          label: displayName,
          displayName,
          rawName: eventName,
          opName: displayName,
          opType: event.args?.op_type || event.cat || "trace-event",
          colorKey: event.args?.op_type || event.cat || event.name || "trace-event",
          laneKind: trace.processNames.get(lane.pid) || `PID ${lane.pid}`,
        };
        const baseColor = taskColormap?.colorForTask(task, "semantic") || processColor(lane.pid);
        if (swimlaneHelper?.drawTaskBar) {
          swimlaneHelper.drawTaskBar(context, {
            x,
            y: 2,
            width: drawWidth,
            height: 18,
            radius: 2,
            baseColor,
            fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
            task,
            isSelected: isLinked || isTimeSelected,
            isEmphasized: isLinked || isTimeSelected,
            showLabel: false,
          });
        } else {
          context.fillStyle = processColor(lane.pid);
          context.fillRect(x, 2, drawWidth, 18);
        }
        if (labelEvents.has(event)) {
          labelOverlays.push({ x, drawWidth, task, baseColor, isLinked, isTimeSelected });
        }
        return { x, width: drawWidth, event };
      });
      labelOverlays.forEach(({ x, drawWidth, task, baseColor, isLinked, isTimeSelected }) => {
        swimlaneHelper?.drawTaskLabel?.(context, {
          x,
          y: 2,
          width: drawWidth,
          height: 18,
          radius: 2,
          baseColor,
          fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
          task,
          isSelected: isLinked || isTimeSelected,
          isEmphasized: isLinked || isTimeSelected,
          alpha: state.selectedNodeId && !isLinked && !isTimeSelected ? 0.55 : 1,
        });
      });
      context.globalAlpha = 1;
      canvas.dataset.linkedCount = String(linkedCount);
    }

    function drawFlows(flows, start, end, width, height) {
      const canvas = state.flowCanvas;
      if (!canvas) return;
      const context = configureCanvas(canvas, width, height);
      if (!state.selectedNodeId) return;
      const span = Math.max(0.001, end - start);
      context.strokeStyle = "rgba(235, 184, 82, 0.34)";
      context.fillStyle = "rgba(235, 184, 82, 0.62)";
      context.lineWidth = 1;
      flows.forEach((flow) => {
        const startY = state.laneY.get(flow.start.laneKey);
        const finishY = state.laneY.get(flow.finish.laneKey);
        if (!Number.isFinite(startY) || !Number.isFinite(finishY)) return;
        const x1 = Math.max(0, Math.min(width, ((flow.start.ts - start) / span) * width));
        const x2 = Math.max(0, Math.min(width, ((flow.finish.ts - start) / span) * width));
        const y1 = startY + LANE_HEIGHT / 2;
        const y2 = finishY + LANE_HEIGHT / 2;
        const bend = Math.max(8, Math.abs(x2 - x1) * 0.4);
        context.beginPath();
        context.moveTo(x1, y1);
        context.bezierCurveTo(x1 + bend, y1, x2 - bend, y2, x2, y2);
        context.stroke();
        context.beginPath();
        context.moveTo(x2, y2);
        context.lineTo(x2 - 4, y2 - 2.5);
        context.lineTo(x2 - 4, y2 + 2.5);
        context.closePath();
        context.fill();
      });
    }

    function showTooltip(hit, pointerEvent, lane) {
      const event = hit.event;
      const process = trace.processNames.get(lane.pid) || `PID ${lane.pid}`;
      const details = [
        `#${event.streamSequence} · ${event.name || "event"}`,
        `${process} / ${laneLabel(lane)}`,
        `${formatDuration(event.dur)} @ +${formatDuration(event.ts - state.rangeStart)}`,
        event.cat ? `category: ${event.cat}` : "",
        event.ownerNodeId ? `node_id: ${event.ownerNodeId}` : "",
      ].filter(Boolean);
      const args = Object.entries(event.args || {}).slice(0, 4).map(([key, value]) => `${key}: ${String(value)}`);
      state.tooltip.replaceChildren(...details.concat(args).map((line, index) => {
        const item = document.createElement(index === 0 ? "strong" : "span");
        item.textContent = line;
        return item;
      }));
      state.tooltip.hidden = false;
      const bounds = container.getBoundingClientRect();
      state.tooltip.style.left = `${Math.min(bounds.width - 300, Math.max(8, pointerEvent.clientX - bounds.left + 12))}px`;
      state.tooltip.style.top = `${Math.max(8, pointerEvent.clientY - bounds.top - 20)}px`;
    }

    function hideTooltip() {
      state.tooltip.hidden = true;
    }

    function updateTimeSelectionVisual() {
      timeSelectionHelper.updateSelectionLayer(state.selectionLayer, state.timeSelection, {
        rangeStart: state.rangeStart,
        rangeEnd: state.rangeEnd,
        trackLeft: LABEL_WIDTH,
        trackWidth: state.trackWidth,
        top: RULER_HEIGHT,
        height: Math.max(1, (state.chart?.scrollHeight || RULER_HEIGHT + 1) - RULER_HEIGHT),
      });
      timeSelectionHelper.updateSelectionSummary(
        state.selectionSummary,
        state.timeSelection,
        state.language,
        {
          rangeStart: state.rangeStart,
          rangeEnd: state.rangeEnd,
          trackLeft: LABEL_WIDTH,
          trackWidth: state.trackWidth,
          scrollLeft: state.scroller?.scrollLeft || 0,
          viewportWidth: state.scroller?.clientWidth || container.clientWidth,
        },
      );
    }

    function visibleTimelineRange() {
      const trackViewportWidth = Math.max(1, state.scroller.clientWidth - LABEL_WIDTH);
      const startX = Math.max(0, Math.min(state.trackWidth, state.scroller.scrollLeft));
      const endX = Math.max(startX, Math.min(state.trackWidth, startX + trackViewportWidth));
      return {
        start: timeSelectionHelper.xToTime(
          startX,
          state.trackWidth,
          state.rangeStart,
          state.rangeEnd,
        ),
        end: timeSelectionHelper.xToTime(
          endX,
          state.trackWidth,
          state.rangeStart,
          state.rangeEnd,
        ),
      };
    }

    function notifyVisibleRange(source = "render") {
      if (!state.trackWidth) return;
      options.onVisibleRangeChange?.({
        ...visibleTimelineRange(),
        source,
      });
    }

    function setTimeSelection(selection, source = "range") {
      state.timeSelection = timeSelectionHelper.normalizeSelection(selection);
      updateTimeSelectionVisual();
      options.onTimeSelectionChange?.(state.timeSelection, { source });
    }

    function clearTimeSelection(source = "clear") {
      setTimeSelection(null, source);
      hideTooltip();
    }

    function selectEvent(event) {
      const selection = timeSelectionHelper.selectionFromEvent(event);
      if (!selection) return;
      state.timeSelection = selection;
      if (event.ownerNodeId) {
        state.selectedNodeId = event.ownerNodeId;
        state.showSelectedFlows = true;
      }
      hideTooltip();
      options.onEventSelect?.(event);
      render();
      options.onTimeSelectionChange?.(state.timeSelection, { source: "event" });
    }

    function hitAt(canvas, x) {
      return [...(canvas.__traceHits || [])]
        .reverse()
        .find((item) => x >= item.x && x <= item.x + item.width);
    }

    function zoomAt(direction, anchor) {
      if (direction > 0) {
        if (state.zoomIndex >= ZOOM_LEVELS.length - 1) return;
        setZoom(state.zoomIndex + 1, anchor);
        return;
      }
      zoomOut(anchor);
    }

    function panTimeline(direction, accelerated = false) {
      const viewportWidth = Math.max(1, state.scroller.clientWidth - LABEL_WIDTH);
      const distance = Math.max(48, viewportWidth * (accelerated ? 0.48 : 0.12));
      state.scroller.scrollLeft += direction * distance;
    }

    function bindLane(canvas, lane) {
      timeSelectionHelper.bindTimelineInteraction({
        target: canvas,
        getGeometry: () => ({
          width: canvas.getBoundingClientRect().width,
          start: state.rangeStart,
          end: state.rangeEnd,
        }),
        hitTest: (x) => hitAt(canvas, x),
        onInteractionStart: hideTooltip,
        onHover: (event, x) => {
          const hit = hitAt(canvas, x);
          if (hit) showTooltip(hit, event, lane);
          else hideTooltip();
        },
        onDragStart: hideTooltip,
        onPreview: (selection) => setTimeSelection(selection, "drag-preview"),
        onCommit: (selection) => setTimeSelection(selection, "drag"),
        onEventSelect: selectEvent,
        onClear: () => clearTimeSelection("empty-click"),
        onLeave: hideTooltip,
        onZoom: ({ direction, x, event }) => {
          options.onViewportInteraction?.({ source: "zoom" });
          const rect = canvas.getBoundingClientRect();
          const scrollerRect = state.scroller.getBoundingClientRect();
          zoomAt(direction, {
            time: timeSelectionHelper.xToTime(x, rect.width, state.rangeStart, state.rangeEnd),
            viewportX: event.clientX - scrollerRect.left,
          });
        },
        onNavigate: ({ direction, accelerated }) => {
          options.onViewportInteraction?.({ source: "keyboard-pan" });
          panTimeline(direction, accelerated);
        },
      });
    }

    function render() {
      const previousRatio = state.scroller.scrollWidth
        ? (state.scroller.scrollLeft + state.scroller.clientWidth / 2) / state.scroller.scrollWidth
        : 0.5;
      const previousScrollTop = state.scroller.scrollTop;
      const range = renderedRange();
      const laneRange = baseRange();
      state.rangeStart = range.start;
      state.rangeEnd = Math.max(range.start + 0.001, range.end);
      const events = trace.durations.filter((event) => event.ts < state.rangeEnd && event.end >= state.rangeStart);
      const laneEvents = trace.durations.filter((event) => event.ts < laneRange.end && event.end >= laneRange.start);
      const flows = selectedFlows().filter((flow) => (
        Math.max(flow.start.ts, flow.finish.ts) >= state.rangeStart
        && Math.min(flow.start.ts, flow.finish.ts) <= state.rangeEnd
      ));
      state.lanes = buildLanes(laneEvents, flows);
      const byLane = new Map();
      const laneEventCounts = new Map();
      events.forEach((event) => {
        if (!byLane.has(event.laneKey)) byLane.set(event.laneKey, []);
        byLane.get(event.laneKey).push(event);
      });
      laneEvents.forEach((event) => {
        laneEventCounts.set(event.laneKey, (laneEventCounts.get(event.laneKey) || 0) + 1);
      });
      const viewportWidth = Math.max(360, state.scroller.clientWidth - LABEL_WIDTH);
      const requestedTrackWidth = viewportWidth * range.trackMultiplier * ZOOM_LEVELS[state.zoomIndex];
      const trackWidth = Math.ceil(Math.min(MAX_TRACK_WIDTH, requestedTrackWidth));
      state.trackWidth = trackWidth;
      const chart = document.createElement("div");
      chart.className = "raw-trace-chart";
      chart.style.width = `${LABEL_WIDTH + trackWidth}px`;
      state.chart = chart;
      state.laneY.clear();

      const ruler = document.createElement("div");
      ruler.className = "raw-trace-ruler-row";
      const rulerLabel = document.createElement("div");
      rulerLabel.className = "raw-trace-label is-ruler";
      rulerLabel.textContent = state.mode === "step" ? text("step") : text("full");
      const rulerTrack = document.createElement("div");
      rulerTrack.className = "raw-trace-ruler";
      rulerTicks(state.rangeStart, state.rangeEnd).forEach((tick) => rulerTrack.appendChild(tick));
      ruler.append(rulerLabel, rulerTrack);
      chart.appendChild(ruler);

      let lastPid = null;
      state.lanes.forEach((lane) => {
        if (lane.pid !== lastPid) {
          lastPid = lane.pid;
          const group = document.createElement("div");
          group.className = "raw-trace-process-row";
          const groupLabel = document.createElement("div");
          groupLabel.className = "raw-trace-label is-process";
          groupLabel.style.setProperty("--trace-process-color", processColor(lane.pid));
          groupLabel.textContent = trace.processNames.get(lane.pid) || `PID ${lane.pid}`;
          group.appendChild(groupLabel);
          chart.appendChild(group);
        }
        const row = document.createElement("div");
        row.className = "raw-trace-lane-row";
        const label = document.createElement("div");
        label.className = "raw-trace-label is-lane";
        label.dataset.reportTooltip = `${trace.processNames.get(lane.pid) || `PID ${lane.pid}`} / ${laneLabel(lane)}`;
        label.textContent = `${laneLabel(lane)} (${laneEventCounts.get(lane.key) || 0})`;
        const canvas = document.createElement("canvas");
        canvas.className = "raw-trace-lane-canvas";
        canvas.tabIndex = 0;
        canvas.setAttribute("role", "application");
        canvas.setAttribute("aria-label", `${laneLabel(lane)}. ${text("laneInteraction")}`);
        row.append(label, canvas);
        chart.appendChild(row);
        drawLane(canvas, byLane.get(lane.key) || [], state.rangeStart, state.rangeEnd, trackWidth, lane);
        bindLane(canvas, lane);
      });

      state.scroller.replaceChildren(chart);
      chart.querySelectorAll(".raw-trace-lane-row").forEach((row, index) => {
        state.laneY.set(state.lanes[index].key, row.offsetTop);
      });
      state.selectionLayer = timeSelectionHelper.createSelectionLayer(chart);
      state.flowCanvas = null;
      if (flows.length) {
        const flowCanvas = document.createElement("canvas");
        flowCanvas.className = "raw-trace-flow-canvas";
        flowCanvas.style.left = `${LABEL_WIDTH}px`;
        flowCanvas.style.top = "0";
        chart.appendChild(flowCanvas);
        state.flowCanvas = flowCanvas;
        drawFlows(flows, state.rangeStart, state.rangeEnd, trackWidth, chart.scrollHeight);
      }
      updateTimeSelectionVisual();
      if (state.pendingScrollAnchor) {
        const anchor = state.pendingScrollAnchor;
        const contentX = LABEL_WIDTH + timeSelectionHelper.timeToX(
          anchor.time,
          trackWidth,
          state.rangeStart,
          state.rangeEnd,
        );
        const maxScrollLeft = Math.max(0, state.scroller.scrollWidth - state.scroller.clientWidth);
        state.scroller.scrollLeft = Math.max(
          0,
          Math.min(maxScrollLeft, contentX - anchor.viewportX),
        );
        state.pendingScrollAnchor = null;
      } else if (state.centerSelectionAfterRender && state.focusEvent) {
        const center = (state.focusEvent.ts + state.focusEvent.end) / 2;
        const centerRatio = (center - state.rangeStart) / Math.max(0.001, state.rangeEnd - state.rangeStart);
        state.scroller.scrollLeft = Math.max(0, centerRatio * trackWidth - viewportWidth / 2);
      } else {
        state.scroller.scrollLeft = Math.max(0, previousRatio * state.scroller.scrollWidth - state.scroller.clientWidth / 2);
      }
      if (state.selectedNodeId && state.centerSelectionAfterRender) {
        const rows = [...chart.querySelectorAll(".raw-trace-lane-row")];
        const focusedLaneIndex = state.focusEvent
          ? state.lanes.findIndex((lane) => lane.key === state.focusEvent.laneKey)
          : -1;
        const selectedCanvas = [...chart.querySelectorAll(".raw-trace-lane-canvas")]
          .find((canvas) => Number(canvas.dataset.linkedCount) > 0);
        const selectedRow = rows[focusedLaneIndex] || selectedCanvas?.closest(".raw-trace-lane-row");
        if (selectedRow) {
          state.scroller.scrollTop = Math.max(0, selectedRow.offsetTop - state.scroller.clientHeight / 2);
        }
      } else {
        state.scroller.scrollTop = previousScrollTop;
      }
      state.centerSelectionAfterRender = false;
      updateToolbar(events.length);
      notifyVisibleRange("render");
    }

    function setZoom(index, anchor = null) {
      const nextIndex = Math.max(0, Math.min(ZOOM_LEVELS.length - 1, index));
      if (nextIndex === state.zoomIndex) return;
      state.zoomIndex = nextIndex;
      state.pendingScrollAnchor = anchor;
      if (anchor) state.centerSelectionAfterRender = false;
      render();
    }

    function zoomOut(anchor = null) {
      if (state.zoomIndex > 0) {
        setZoom(state.zoomIndex - 1, anchor);
        return;
      }
      if (!state.focusRange) return;
      const base = baseRange();
      const current = activeRange();
      const currentSpan = current.end - current.start;
      const baseSpan = base.end - base.start;
      if (currentSpan * 2 >= baseSpan - 0.001) {
        state.focusRange = null;
      } else {
        const center = state.focusEvent
          ? (state.focusEvent.ts + state.focusEvent.end) / 2
          : (current.start + current.end) / 2;
        state.focusRange = clampRangeAround(center, currentSpan * 2, base);
      }
      state.pendingScrollAnchor = anchor;
      state.centerSelectionAfterRender = anchor ? false : Boolean(state.focusEvent);
      render();
    }

    function resetZoom() {
      state.zoomIndex = 0;
      state.focusRange = null;
      state.pendingScrollAnchor = null;
      state.focusEvent = representativeEventForNode(trace.durations, state.selectedNodeId, baseRange());
      state.centerSelectionAfterRender = Boolean(state.focusEvent);
      render();
    }

    toolbarRoot.querySelectorAll("[data-trace-range]").forEach((button) => {
      button.addEventListener("click", () => {
        state.mode = button.dataset.traceRange;
        state.zoomIndex = 0;
        state.focusRange = null;
        state.focusEvent = representativeEventForNode(trace.durations, state.selectedNodeId, baseRange());
        state.centerSelectionAfterRender = Boolean(state.focusEvent);
        render();
      });
    });
    toolbarRoot.querySelector('[data-trace-zoom="out"]').addEventListener("click", () => {
      options.onViewportInteraction?.({ source: "zoom" });
      zoomOut();
    });
    toolbarRoot.querySelector('[data-trace-zoom="reset"]').addEventListener("click", () => {
      options.onViewportInteraction?.({ source: "reset" });
      resetZoom();
    });
    toolbarRoot.querySelector('[data-trace-zoom="in"]').addEventListener("click", () => {
      options.onViewportInteraction?.({ source: "zoom" });
      setZoom(state.zoomIndex + 1);
    });
    toolbarRoot.querySelector('[data-trace-action="fit"]').addEventListener("click", () => {
      options.onViewportInteraction?.({ source: "fit" });
      resetZoom();
    });
    toolbarRoot.querySelector('[data-trace-action="focus"]').addEventListener("click", () => {
      if (!state.selectedNodeId) return;
      options.onViewportInteraction?.({ source: "selection-focus" });
      activateSelectionFocus();
      render();
    });

    render();
    return {
      resize() {
        global.clearTimeout(state.resizeTimer);
        state.resizeTimer = global.setTimeout(render, 40);
      },
      redraw: render,
      setLanguage(language) {
        state.language = language === "zh" ? "zh" : "en";
        render();
      },
      setSelectedNode(nodeId) {
        const nextNodeId = nodeId || "";
        state.selectedNodeId = nextNodeId;
        state.showSelectedFlows = Boolean(nextNodeId);
        state.timeSelection = null;
        if (nextNodeId) activateSelectionFocus();
        else {
          state.zoomIndex = 0;
          state.focusRange = null;
          state.focusEvent = null;
          state.centerSelectionAfterRender = false;
        }
        render();
        return state.focusEvent ? { ...state.focusEvent } : null;
      },
      setSelectedEvent(event) {
        selectEvent(event);
      },
      clearTimeSelection,
      getTimeSelection() {
        return state.timeSelection ? { ...state.timeSelection } : null;
      },
      getVisibleRange: visibleTimelineRange,
      counts: { events: trace.rawEvents.length, durations: trace.durations.length, flows: trace.flows.length },
    };
  }

  global.DeepSeekTraceView = {
    createTraceView,
    parseTrace,
    focusRangeForEvent,
    representativeEventForNode,
  };
})(window);
