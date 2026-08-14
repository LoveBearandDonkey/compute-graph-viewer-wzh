(function exposeHbmTimelineView(global) {
  "use strict";

  const COLORS = {
    read: "#4e8fda",
    write: "#d6a84b",
    occupancy: "#48a868",
    grid: "rgba(127, 132, 142, 0.18)",
    muted: "#8b949e",
  };
  const MIN_VISIBLE_RAW_SAMPLES = 5;

  function finite(value, fallback = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function formatTime(us) {
    const value = finite(us);
    if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(2)} s`;
    if (Math.abs(value) >= 1000) return `${(value / 1000).toFixed(2)} ms`;
    return `${value.toFixed(0)} us`;
  }

  function percentile(sortedValues, ratio) {
    if (!sortedValues.length) return 0;
    return sortedValues[Math.round((sortedValues.length - 1) * ratio)];
  }

  function adaptiveDomain(values, minimumSpan = 0.01) {
    const sorted = values.map(Number).filter(Number.isFinite).sort((left, right) => left - right);
    if (!sorted.length) return { min: 0, max: 1 };
    const low = percentile(sorted, 0.01);
    const high = percentile(sorted, 0.99);
    const observedSpan = high - low;
    const padding = observedSpan > 0
      ? observedSpan * 0.12
      : Math.max(Math.abs(low) * 0.005, minimumSpan / 2);
    return { min: low - padding, max: high + padding };
  }

  function formatAxisValue(value, span) {
    const digits = span < 0.1 ? 4 : span < 1 ? 3 : span < 10 ? 2 : span < 1000 ? 1 : 0;
    return Number(value).toFixed(digits);
  }

  function lowerBoundIndex(points, targetTime) {
    let low = 0;
    let high = points.length;
    while (low < high) {
      const middle = Math.floor((low + high) / 2);
      if (finite(points[middle][0]) < targetTime) low = middle + 1;
      else high = middle;
    }
    return low;
  }

  function interpolateSeriesWindow(points, valueIndex, start, end) {
    if (!points.length) return [];
    const rangeStart = Math.min(finite(start), finite(end));
    const rangeEnd = Math.max(finite(start), finite(end));
    const sampleAt = (time) => {
      const rightIndex = lowerBoundIndex(points, time);
      if (rightIndex <= 0) return [time, finite(points[0][valueIndex])];
      if (rightIndex >= points.length) return [time, finite(points.at(-1)[valueIndex])];
      const left = points[rightIndex - 1];
      const right = points[rightIndex];
      const leftTime = finite(left[0]);
      const rightTime = finite(right[0]);
      if (rightTime === time || rightTime <= leftTime) return [time, finite(right[valueIndex])];
      const ratio = (time - leftTime) / (rightTime - leftTime);
      return [
        time,
        finite(left[valueIndex]) + (finite(right[valueIndex]) - finite(left[valueIndex])) * ratio,
      ];
    };

    const result = [sampleAt(rangeStart)];
    let index = lowerBoundIndex(points, rangeStart);
    while (index < points.length && finite(points[index][0]) < rangeEnd) {
      const pointTime = finite(points[index][0]);
      if (pointTime > rangeStart) result.push([pointTime, finite(points[index][valueIndex])]);
      index += 1;
    }
    if (rangeEnd > rangeStart) result.push(sampleAt(rangeEnd));
    return result;
  }

  function samplingAwareRange(fullStart, fullEnd, minimumSpan, nextStart, nextEnd) {
    const first = finite(nextStart, fullStart);
    const second = finite(nextEnd, fullEnd);
    const requestedStart = Math.min(first, second);
    const requestedEnd = Math.max(first, second);
    const overlapStart = Math.max(fullStart, requestedStart);
    const overlapEnd = Math.min(fullEnd, requestedEnd);
    if (overlapEnd <= overlapStart) return { start: fullStart, end: fullEnd };
    const center = (overlapStart + overlapEnd) / 2;
    const span = Math.min(
      fullEnd - fullStart,
      Math.max(overlapEnd - overlapStart, minimumSpan),
    );
    let start = center - span / 2;
    let end = center + span / 2;
    if (start < fullStart) {
      end += fullStart - start;
      start = fullStart;
    }
    if (end > fullEnd) {
      start -= end - fullEnd;
      end = fullEnd;
    }
    return { start: Math.max(fullStart, start), end: Math.min(fullEnd, end) };
  }

  function createHbmView(container, data, options = {}) {
    if (!container) throw new Error("HBM timeline container is required");
    const bandwidth = Array.isArray(data?.bandwidth?.points) ? data.bandwidth.points : [];
    const occupancy = Array.isArray(data?.occupancy?.points) ? data.occupancy.points : [];
    if (!bandwidth.length || !occupancy.length) {
      container.hidden = true;
      container.replaceChildren();
      return null;
    }

    const language = options.language === "zh" ? "zh" : "en";
    const labels = language === "zh" ? {
      title: "HBM 带宽与占用量",
      bandwidth: "读 / 写带宽",
      readBandwidth: "读取带宽",
      writeBandwidth: "写入带宽",
      occupancy: "HBM 占用",
      read: "读取",
      write: "写入",
      phase: "阶段",
      step: "步骤",
      op: "窗口主导算子",
      sample: "采样时间",
    } : {
      title: "HBM Bandwidth & Occupancy",
      bandwidth: "Read / write bandwidth",
      readBandwidth: "Read bandwidth",
      writeBandwidth: "Write bandwidth",
      occupancy: "HBM occupancy",
      read: "Read",
      write: "Write",
      phase: "Phase",
      step: "Step",
      op: "Window-dominant op",
      sample: "Sample time",
    };

    container.innerHTML = `
      <div class="hbm-view">
        <div class="hbm-view-header">
          <h3 class="hbm-view-title">${labels.title}</h3>
          <div class="hbm-view-legend" aria-label="HBM series legend">
            <span class="hbm-view-legend-item"><i class="hbm-view-dot" style="--dot-color:${COLORS.read}"></i>${labels.read}</span>
            <span class="hbm-view-legend-item"><i class="hbm-view-dot" style="--dot-color:${COLORS.write}"></i>${labels.write}</span>
            <span class="hbm-view-legend-item"><i class="hbm-view-dot" style="--dot-color:${COLORS.occupancy}"></i>${labels.occupancy}</span>
          </div>
        </div>
        <div class="hbm-chart-stack">
          <div class="hbm-chart-row">
            <div class="hbm-chart-label"><span>${labels.readBandwidth}</span><small>GB/s</small></div>
            <canvas class="hbm-chart-canvas" data-hbm-chart="read" tabindex="0" aria-label="${labels.readBandwidth}"></canvas>
          </div>
          <div class="hbm-chart-row">
            <div class="hbm-chart-label"><span>${labels.writeBandwidth}</span><small>GB/s</small></div>
            <canvas class="hbm-chart-canvas" data-hbm-chart="write" tabindex="0" aria-label="${labels.writeBandwidth}"></canvas>
          </div>
          <div class="hbm-chart-row">
            <div class="hbm-chart-label"><span>${labels.occupancy}</span><small>GiB</small></div>
            <canvas class="hbm-chart-canvas" data-hbm-chart="occupancy" tabindex="0" aria-label="${labels.occupancy}"></canvas>
          </div>
        </div>
        <div class="hbm-tooltip" role="tooltip" hidden></div>
      </div>`;

    const view = container.querySelector(".hbm-view");
    const tooltip = container.querySelector(".hbm-tooltip");
    const canvases = [...container.querySelectorAll(".hbm-chart-canvas")];
    const chartRows = [...container.querySelectorAll(".hbm-chart-row")];
    const legendItems = [...container.querySelectorAll(".hbm-view-legend-item")];
    const allTimes = [...bandwidth, ...occupancy].map((point) => finite(point[0]));
    const fullStart = finite(data?.time?.start_us, Math.min(...allTimes));
    const fullEnd = finite(data?.time?.end_us, Math.max(...allTimes));
    const inferredIntervals = bandwidth
      .slice(1)
      .map((point, index) => finite(point[0]) - finite(bandwidth[index][0]))
      .filter((value) => value > 0)
      .sort((left, right) => left - right);
    const sampleIntervalUs = Math.max(
      Number.EPSILON,
      finite(
        data?.bandwidth?.sample_interval_us,
        percentile(inferredIntervals, 0.5) || 1,
      ),
    );
    const minimumVisibleSpan = Math.min(
      Math.max(Number.EPSILON, fullEnd - fullStart),
      sampleIntervalUs * MIN_VISIBLE_RAW_SAMPLES,
    );
    let rangeStart = fullStart;
    let rangeEnd = Math.max(fullStart + Number.EPSILON, fullEnd);
    let occupancyFocus = null;

    function normalizeRange(nextStart, nextEnd) {
      return samplingAwareRange(fullStart, fullEnd, minimumVisibleSpan, nextStart, nextEnd);
    }

    function resolveFocusRange(nextStart, nextEnd) {
      const first = finite(nextStart, NaN);
      const second = finite(nextEnd, NaN);
      if (!Number.isFinite(first) || !Number.isFinite(second)) return null;
      const candidates = [{ start: Math.min(first, second), end: Math.max(first, second) }];
      const epochOrigin = Number(data?.time?.origin_epoch_us);
      if (Number.isFinite(epochOrigin)) {
        candidates.push({
          start: Math.min(first, second) - epochOrigin,
          end: Math.max(first, second) - epochOrigin,
        });
      }
      const traceOrigin = Number(data?.trace_alignment?.trace_origin_us);
      const hbmOrigin = Number(data?.trace_alignment?.hbm_origin_us);
      if (Number.isFinite(traceOrigin) && Number.isFinite(hbmOrigin)) {
        candidates.push({
          start: Math.min(first, second) - traceOrigin + hbmOrigin,
          end: Math.max(first, second) - traceOrigin + hbmOrigin,
        });
      }
      return candidates.find((candidate) => (
        candidate.end >= fullStart && candidate.start <= fullEnd
      )) || null;
    }

    if (options.range) {
      const initialRange = normalizeRange(options.range.start, options.range.end);
      rangeStart = initialRange.start;
      rangeEnd = initialRange.end;
    }

    function configure(canvas) {
      const width = Math.max(1, Math.floor(canvas.clientWidth));
      const height = Math.max(1, Math.floor(canvas.clientHeight));
      const ratio = Math.max(1, global.devicePixelRatio || 1);
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      const context = canvas.getContext("2d");
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, width, height);
      return { context, width, height };
    }

    function axes(context, width, height, domain) {
      const plot = { left: 52, right: width - 8, top: 6, bottom: height - 18 };
      plot.width = Math.max(1, plot.right - plot.left);
      plot.height = Math.max(1, plot.bottom - plot.top);
      const domainSpan = Math.max(Number.EPSILON, domain.max - domain.min);
      context.strokeStyle = COLORS.grid;
      context.fillStyle = COLORS.muted;
      context.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
      [0, 0.25, 0.5, 0.75, 1].forEach((ratio, index) => {
        const y = Math.round(plot.top + ratio * plot.height) + 0.5;
        context.beginPath();
        context.moveTo(plot.left, y);
        context.lineTo(plot.right, y);
        context.stroke();
        context.textAlign = "right";
        context.textBaseline = "middle";
        context.fillText(formatAxisValue(domain.max - domainSpan * ratio, domainSpan), plot.left - 6, y);
      });
      [0, 0.25, 0.5, 0.75, 1].forEach((ratio, index) => {
        const x = Math.round(plot.left + ratio * plot.width) + 0.5;
        context.beginPath();
        context.moveTo(x, plot.top);
        context.lineTo(x, plot.bottom);
        context.stroke();
        const label = formatTime(rangeStart + (rangeEnd - rangeStart) * ratio);
        context.textAlign = index === 0 ? "left" : index === 4 ? "right" : "center";
        context.textBaseline = "top";
        context.fillText(label, x, plot.bottom + 3);
      });
      return plot;
    }

    function drawOccupancyFocus(context, plot) {
      if (!occupancyFocus) return;
      const span = Math.max(Number.EPSILON, rangeEnd - rangeStart);
      const focusStart = Math.max(rangeStart, occupancyFocus.start);
      const focusEnd = Math.min(rangeEnd, occupancyFocus.end);
      if (focusEnd < focusStart) return;
      const rawLeft = plot.left + ((focusStart - rangeStart) / span) * plot.width;
      const rawRight = plot.left + ((focusEnd - rangeStart) / span) * plot.width;
      const width = Math.max(2, rawRight - rawLeft);
      const left = Math.max(plot.left, Math.min(plot.right - width, (rawLeft + rawRight - width) / 2));
      context.fillStyle = "rgba(72, 168, 104, 0.16)";
      context.fillRect(left, plot.top, width, plot.height);
      context.strokeStyle = COLORS.occupancy;
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(left + 0.5, plot.top);
      context.lineTo(left + 0.5, plot.bottom);
      context.moveTo(left + width - 0.5, plot.top);
      context.lineTo(left + width - 0.5, plot.bottom);
      context.stroke();
    }

    function drawSeries(canvas, points, valueIndex, color, minimumSpan) {
      const { context, width, height } = configure(canvas);
      const visible = interpolateSeriesWindow(points, valueIndex, rangeStart, rangeEnd);
      const domain = adaptiveDomain(visible.map((point) => point[1]), minimumSpan);
      const plot = axes(context, width, height, domain);
      canvas.__hbmPlot = plot;
      canvas.__hbmSourcePointCount = points.length;
      canvas.__hbmRenderedPointCount = visible.length;
      const domainSpan = Math.max(Number.EPSILON, domain.max - domain.min);
      const span = Math.max(Number.EPSILON, rangeEnd - rangeStart);
      context.save();
      context.beginPath();
      context.rect(plot.left, plot.top, plot.width, plot.height);
      context.clip();
      if (canvas.dataset.hbmChart === "occupancy") drawOccupancyFocus(context, plot);
      context.beginPath();
      visible.forEach((point, index) => {
        const x = plot.left + ((finite(point[0]) - rangeStart) / span) * plot.width;
        const normalized = Math.max(0, Math.min(1, (finite(point[1]) - domain.min) / domainSpan));
        const y = plot.bottom - normalized * plot.height;
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      });
      if (visible.length === 1) {
        const normalized = Math.max(0, Math.min(1, (finite(visible[0][1]) - domain.min) / domainSpan));
        const y = plot.bottom - normalized * plot.height;
        context.moveTo(plot.left, y);
        context.lineTo(plot.right, y);
      }
      context.strokeStyle = color;
      context.lineWidth = 1.6;
      context.lineJoin = "round";
      context.lineCap = "round";
      context.stroke();
      context.restore();
    }

    function draw() {
      canvases.forEach((canvas) => {
        if (canvas.closest(".hbm-chart-row")?.hidden) return;
        if (canvas.dataset.hbmChart === "read") drawSeries(canvas, bandwidth, 1, COLORS.read, 1);
        else if (canvas.dataset.hbmChart === "write") drawSeries(canvas, bandwidth, 2, COLORS.write, 0.1);
        else drawSeries(canvas, occupancy, 1, COLORS.occupancy, 0.02);
      });
    }

    function setOccupancyOnly(enabled) {
      chartRows.forEach((row) => {
        const chart = row.querySelector("[data-hbm-chart]")?.dataset.hbmChart;
        row.hidden = enabled && chart !== "occupancy";
      });
      legendItems.forEach((item) => {
        const isOccupancy = item.textContent.trim() === labels.occupancy;
        item.hidden = enabled && !isOccupancy;
      });
      view.dataset.mode = enabled ? "occupancy-focus" : "all-series";
    }

    function closestPoint(points, canvas, event) {
      const rect = canvas.getBoundingClientRect();
      const plot = canvas.__hbmPlot || { left: 0, width: Math.max(1, rect.width) };
      const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left - plot.left) / Math.max(1, plot.width)));
      const targetTime = rangeStart + ratio * Math.max(Number.EPSILON, rangeEnd - rangeStart);
      let low = 0;
      let high = points.length - 1;
      while (low < high) {
        const middle = Math.floor((low + high) / 2);
        if (finite(points[middle][0]) < targetTime) low = middle + 1;
        else high = middle;
      }
      const left = points[Math.max(0, low - 1)];
      const right = points[low];
      return !left || Math.abs(finite(right[0]) - targetTime) < Math.abs(finite(left[0]) - targetTime) ? right : left;
    }

    function showTooltip(canvas, point, event) {
      const bandwidthChart = canvas.dataset.hbmChart !== "occupancy";
      const rows = bandwidthChart ? [
        `<strong>${labels.sample}: ${formatTime(point[0])}</strong>`,
        `${labels.read}: ${finite(point[1]).toFixed(2)} GB/s`,
        `${labels.write}: ${finite(point[2]).toFixed(2)} GB/s`,
        `${labels.phase}: ${point[3] || "-"}`,
        `${labels.step}: ${point[4] ?? "-"}`,
        `${labels.op}: ${point[5] || "-"}`,
      ] : [
        `<strong>${labels.sample}: ${formatTime(point[0])}</strong>`,
        `${labels.occupancy}: ${finite(point[1]).toFixed(4)} GiB`,
        `${labels.phase}: ${point[3] || "-"}`,
      ];
      tooltip.innerHTML = rows.join("<span></span>");
      tooltip.hidden = false;
      const bounds = view.getBoundingClientRect();
      const left = Math.min(bounds.width - 258, Math.max(8, event.clientX - bounds.left + 12));
      const top = Math.min(bounds.height - tooltip.offsetHeight - 8, Math.max(8, event.clientY - bounds.top + 12));
      tooltip.style.left = `${left}px`;
      tooltip.style.top = `${top}px`;
    }

    canvases.forEach((canvas) => {
      const points = canvas.dataset.hbmChart === "occupancy" ? occupancy : bandwidth;
      canvas.addEventListener("pointermove", (event) => {
        const point = closestPoint(points, canvas, event);
        if (point) showTooltip(canvas, point, event);
      });
      canvas.addEventListener("pointerleave", () => { tooltip.hidden = true; });
    });

    const observer = "ResizeObserver" in global ? new ResizeObserver(draw) : null;
    observer?.observe(view);
    global.requestAnimationFrame(draw);
    return {
      redraw: draw,
      setRange(start, end) {
        occupancyFocus = null;
        setOccupancyOnly(false);
        // Same-capture fixed-offset resolution; never a proportional projection.
        const resolved = resolveFocusRange(start, end);
        const next = resolved
          ? normalizeRange(resolved.start, resolved.end)
          : normalizeRange(start, end);
        rangeStart = next.start;
        rangeEnd = next.end;
        tooltip.hidden = true;
        draw();
      },
      resetRange() {
        occupancyFocus = null;
        setOccupancyOnly(false);
        rangeStart = fullStart;
        rangeEnd = Math.max(fullStart + Number.EPSILON, fullEnd);
        tooltip.hidden = true;
        draw();
      },
      focusOccupancy(start, end) {
        const resolved = resolveFocusRange(start, end);
        if (!resolved) return false;
        occupancyFocus = resolved;
        const next = normalizeRange(resolved.start, resolved.end);
        rangeStart = next.start;
        rangeEnd = next.end;
        setOccupancyOnly(true);
        tooltip.hidden = true;
        draw();
        return true;
      },
      getRange() {
        return { start: rangeStart, end: rangeEnd };
      },
      getOccupancyFocus() {
        return occupancyFocus ? { ...occupancyFocus } : null;
      },
      getSourcePointCounts() {
        return { bandwidth: bandwidth.length, occupancy: occupancy.length };
      },
      getSamplingWindow() {
        return { sampleIntervalUs, minimumVisibleSpan, minimumRawSamples: MIN_VISIBLE_RAW_SAMPLES };
      },
      destroy: () => observer?.disconnect(),
    };
  }

  global.HbmTimelineView = { createHbmView, interpolateSeriesWindow, samplingAwareRange };
})(window);
