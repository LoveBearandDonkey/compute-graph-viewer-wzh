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

  function formatFrequency(mhz) {
    const value = finite(mhz, NaN);
    if (!Number.isFinite(value)) return "–";
    return value >= 1000 ? `${(value / 1000).toFixed(2)} GHz` : `${value.toFixed(0)} MHz`;
  }

  function createFrequencyMarkup(profile, labels) {
    const current = finite(profile?.aicore_freq_mhz ?? profile?.derived?.mhz ?? profile?.declared?.mhz, NaN);
    const derived = profile?.derived || {};
    const minimum = finite(derived.min_mhz, current);
    const maximum = finite(derived.max_mhz, current);
    const declared = finite(profile?.declared?.mhz, NaN);
    const derivedMhz = finite(derived.mhz, NaN);
    const hasFrequency = Number.isFinite(current);
    const agreement = profile?.cross_check?.agreement === "match";
    const throttlingObserved = profile?.throttling?.observed === true;
    const declaredSamples = Array.isArray(profile?.declared?.samples)
      ? profile.declared.samples
        .map((sample) => ({ ts: finite(sample?.ts_us, NaN), mhz: finite(sample?.mhz, NaN) }))
        .filter((sample) => Number.isFinite(sample.ts) && Number.isFinite(sample.mhz))
        .sort((left, right) => left.ts - right.ts)
      : [];
    const samples = declaredSamples.length ? declaredSamples : [{ ts: 0, mhz: current }, { ts: 1, mhz: current }];
    const startTs = samples[0].ts;
    const endTs = Math.max(startTs + Number.EPSILON, samples.at(-1).ts);
    const domainMin = Math.min(minimum, ...samples.map((sample) => sample.mhz));
    const domainMax = Math.max(maximum, ...samples.map((sample) => sample.mhz));
    // Keep the derived min/max close to the plot bounds so their labels remain
    // visually tied to the band even in a short, wide timeline section.
    const domainPadding = Math.max(0.05, (domainMax - domainMin) * 0.06);
    const plotMin = domainMin - domainPadding;
    const plotMax = domainMax + domainPadding;
    const pointX = (sample) => 4 + (sample.ts - startTs) / (endTs - startTs) * 992;
    const pointY = (mhz) => 10 + (plotMax - mhz) / Math.max(Number.EPSILON, plotMax - plotMin) * 56;
    let stepPath = `M ${pointX(samples[0]).toFixed(2)} ${pointY(samples[0].mhz).toFixed(2)}`;
    samples.slice(1).forEach((sample) => {
      stepPath += ` H ${pointX(sample).toFixed(2)} V ${pointY(sample.mhz).toFixed(2)}`;
    });
    stepPath += ` H 996`;
    const crossCheckTip = `${labels.frequencyCrossCheck}: ${Number.isFinite(declared) && Number.isFinite(derivedMhz) ? `${declared.toFixed(0)} / ${derivedMhz.toFixed(0)} MHz · ${agreement ? labels.consistent : labels.unverified}` : labels.unverified}`;
    const throttlingTip = `${labels.frequencyThrottling}: ${throttlingObserved ? labels.observed : labels.notObserved}`;
    return {
      hasFrequency,
      html: `
        <section class="aicore-frequency-view">
          <div class="hbm-view-header">
            <div class="hbm-view-heading">
              <h3 class="hbm-view-title">${labels.frequencyTitle}</h3>
              <button class="timeline-section-toggle" type="button" aria-expanded="${hasFrequency}" aria-label="${hasFrequency ? labels.collapseFrequency : labels.expandFrequency}" data-report-tooltip="${hasFrequency ? labels.collapseFrequency : labels.expandFrequency}" data-frequency-toggle><svg viewBox="0 0 16 16" aria-hidden="true"><path d="m5 6 3 3 3-3"></path></svg></button>
            </div>
          </div>
          <div class="aicore-frequency-content"${hasFrequency ? "" : " hidden"}>
            ${hasFrequency ? `
              <div class="aicore-frequency-chart">
                <div class="aicore-frequency-scale" aria-hidden="true">
                  <span class="aicore-frequency-scale__max" style="--frequency-y:${(pointY(maximum) / 82 * 100).toFixed(2)}%">${maximum.toFixed(2)} MHz</span>
                  <strong class="aicore-frequency-scale__current" style="--frequency-y:${(pointY(current) / 82 * 100).toFixed(2)}%">${formatFrequency(current)} / ${current.toFixed(0)} MHz</strong>
                  <span class="aicore-frequency-scale__min" style="--frequency-y:${(pointY(minimum) / 82 * 100).toFixed(2)}%">${minimum.toFixed(2)} MHz</span>
                </div>
                <div class="aicore-frequency-plot" data-report-tooltip="${crossCheckTip} · ${throttlingTip}" aria-label="${labels.frequencyRange}: ${minimum.toFixed(2)}–${maximum.toFixed(2)} MHz. ${crossCheckTip}. ${throttlingTip}" tabindex="0">
                  <svg viewBox="0 0 1000 82" preserveAspectRatio="none" role="img" aria-hidden="true">
                    <rect class="aicore-frequency-derived-band" x="4" y="${pointY(maximum).toFixed(2)}" width="992" height="${Math.max(1, pointY(minimum) - pointY(maximum)).toFixed(2)}"></rect>
                    <path class="aicore-frequency-gridline" d="M4 ${pointY(current).toFixed(2)} H996"></path>
                    <path class="aicore-frequency-step" d="${stepPath}"></path>
                  </svg>
                </div>
              </div>` : `<div class="hbm-empty-state">${labels.noFrequencyData}</div>`}
          </div>
        </section>`,
    };
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
    const hasData = bandwidth.length > 0 && occupancy.length > 0;

    const language = options.language === "zh" ? "zh" : "en";
    const labels = language === "zh" ? {
      frequencyTitle: "AICore 频率",
      frequencyRange: "推导频率范围",
      frequencyCrossCheck: "声明值 / 推导值",
      frequencyThrottling: "降频状态",
      consistent: "一致",
      unverified: "未校验",
      observed: "观察到降频",
      notObserved: "未观察到降频",
      noFrequencyData: "未采集到 AICore 频率数据",
      expandFrequency: "展开 AICore 频率",
      collapseFrequency: "折叠 AICore 频率",
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
      noData: "未采集到 HBM 带宽与占用量数据",
      expand: "展开 HBM",
      collapse: "折叠 HBM",
    } : {
      frequencyTitle: "AICore Frequency",
      frequencyRange: "Derived frequency range",
      frequencyCrossCheck: "Declared / derived",
      frequencyThrottling: "Throttling",
      consistent: "consistent",
      unverified: "not verified",
      observed: "observed",
      notObserved: "not observed",
      noFrequencyData: "AICore frequency data was not captured",
      expandFrequency: "Expand AICore frequency",
      collapseFrequency: "Collapse AICore frequency",
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
      noData: "HBM bandwidth and occupancy data was not captured",
      expand: "Expand HBM",
      collapse: "Collapse HBM",
    };

    const frequency = createFrequencyMarkup(options.deviceProfile, labels);
    container.innerHTML = `
      <div class="hardware-metrics-view">
        ${frequency.html}
      <section class="hbm-view">
        <div class="hbm-view-header">
          <div class="hbm-view-heading">
            <h3 class="hbm-view-title">${labels.title}</h3>
            <button class="timeline-section-toggle" type="button" aria-expanded="${hasData}" aria-label="${hasData ? labels.collapse : labels.expand}" data-report-tooltip="${hasData ? labels.collapse : labels.expand}"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="m5 6 3 3 3-3"></path></svg></button>
          </div>
          <div class="hbm-view-legend" aria-label="HBM series legend"${hasData ? "" : " hidden"}>
            <span class="hbm-view-legend-item"><i class="hbm-view-dot" style="--dot-color:${COLORS.read}"></i>${labels.read}</span>
            <span class="hbm-view-legend-item"><i class="hbm-view-dot" style="--dot-color:${COLORS.write}"></i>${labels.write}</span>
            <span class="hbm-view-legend-item"><i class="hbm-view-dot" style="--dot-color:${COLORS.occupancy}"></i>${labels.occupancy}</span>
          </div>
        </div>
        <div class="hbm-view-content"${hasData ? "" : " hidden"}>
        ${hasData ? `<div class="hbm-chart-stack">
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
        </div>` : `<div class="hbm-empty-state">${labels.noData}</div>`}
        </div>
        <div class="hbm-tooltip" role="tooltip" hidden></div>
      </section>
      </div>`;

    const view = container.querySelector(".hbm-view");
    const tooltip = view.querySelector(".hbm-tooltip");
    const toggle = view.querySelector(".timeline-section-toggle");
    const content = view.querySelector(".hbm-view-content");
    const frequencyToggle = container.querySelector("[data-frequency-toggle]");
    const frequencyContent = container.querySelector(".aicore-frequency-content");
    frequencyToggle?.addEventListener("click", () => {
      const expanded = frequencyToggle.getAttribute("aria-expanded") === "true";
      frequencyToggle.setAttribute("aria-expanded", String(!expanded));
      frequencyToggle.setAttribute("aria-label", expanded ? labels.expandFrequency : labels.collapseFrequency);
      frequencyToggle.dataset.reportTooltip = expanded ? labels.expandFrequency : labels.collapseFrequency;
      frequencyContent.hidden = expanded;
    });
    toggle.addEventListener("click", () => {
      const expanded = toggle.getAttribute("aria-expanded") === "true";
      toggle.setAttribute("aria-expanded", String(!expanded));
      toggle.setAttribute("aria-label", expanded ? labels.expand : labels.collapse);
      toggle.dataset.reportTooltip = expanded ? labels.expand : labels.collapse;
      content.hidden = expanded;
      if (!expanded && hasData) global.requestAnimationFrame(draw);
    });
    const canvases = [...container.querySelectorAll(".hbm-chart-canvas")];
    if (!hasData) {
      return {
        redraw() {}, setRange() {}, resetRange() {},
        getSourcePointCounts: () => ({ bandwidth: 0, occupancy: 0 }),
        destroy() {},
      };
    }
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

    function normalizeRange(nextStart, nextEnd) {
      return samplingAwareRange(fullStart, fullEnd, minimumVisibleSpan, nextStart, nextEnd);
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
        if (canvas.dataset.hbmChart === "read") drawSeries(canvas, bandwidth, 1, COLORS.read, 1);
        else if (canvas.dataset.hbmChart === "write") drawSeries(canvas, bandwidth, 2, COLORS.write, 0.1);
        else drawSeries(canvas, occupancy, 1, COLORS.occupancy, 0.02);
      });
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
        const next = normalizeRange(start, end);
        rangeStart = next.start;
        rangeEnd = next.end;
        tooltip.hidden = true;
        draw();
      },
      resetRange() {
        rangeStart = fullStart;
        rangeEnd = Math.max(fullStart + Number.EPSILON, fullEnd);
        tooltip.hidden = true;
        draw();
      },
      getRange() {
        return { start: rangeStart, end: rangeEnd };
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
