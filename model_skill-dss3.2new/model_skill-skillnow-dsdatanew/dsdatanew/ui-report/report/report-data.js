(function registerDeepSeekReportData(global) {
  "use strict";

  function collectNodeObjects(value, result = new Map()) {
    if (Array.isArray(value)) {
      value.forEach((item) => collectNodeObjects(item, result));
      return result;
    }
    if (!value || typeof value !== "object") return result;

    if (typeof value.node_id === "string" && value.node_id) {
      const score = [
        value.semantic,
        value.node_kind,
        value.code_ref,
        value.children,
        value.kernels,
        value.op_ratio,
        value.time_us
      ].filter((item) => item != null).length;
      const previous = result.get(value.node_id);
      if (!previous || score > previous.score) result.set(value.node_id, { value, score });
    }

    Object.values(value).forEach((item) => collectNodeObjects(item, result));
    return result;
  }

  function nodeIndex(value) {
    return new Map(Array.from(collectNodeObjects(value), ([id, item]) => [id, item.value]));
  }

  function number(value, fallback = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function formatNumber(value, digits = 2) {
    const numeric = number(value, NaN);
    if (!Number.isFinite(numeric)) return "–";
    return numeric.toLocaleString("en-US", { maximumFractionDigits: digits });
  }

  function formatUs(value) {
    const us = number(value);
    if (us >= 1000) return `${formatNumber(us / 1000, 3)} ms`;
    return `${formatNumber(us, 2)} us`;
  }

  function formatMs(value) {
    const numeric = number(value, NaN);
    return Number.isFinite(numeric) ? `${formatNumber(numeric, 3)} ms` : "–";
  }

  function deriveCoreEventMetrics(events) {
    const intervals = events.map((event) => {
      const startUs = number(event.ts_us, NaN);
      const durationUs = Math.max(0, number(event.duration_us));
      const explicitEndUs = number(event.end_us, NaN);
      const endUs = Number.isFinite(explicitEndUs) ? explicitEndUs : startUs + durationUs;
      return {
        startUs,
        endUs,
        durationUs,
        waitUs: Math.max(0, number(event.wait_time_us)),
      };
    }).filter((interval) => Number.isFinite(interval.startUs) && Number.isFinite(interval.endUs));

    if (!intervals.length) return {};

    intervals.sort((left, right) => left.startUs - right.startUs || left.endUs - right.endUs);
    let busyUnionUs = 0;
    let unionStartUs = intervals[0].startUs;
    let unionEndUs = intervals[0].endUs;
    intervals.slice(1).forEach((interval) => {
      if (interval.startUs <= unionEndUs) {
        unionEndUs = Math.max(unionEndUs, interval.endUs);
        return;
      }
      busyUnionUs += Math.max(0, unionEndUs - unionStartUs);
      unionStartUs = interval.startUs;
      unionEndUs = interval.endUs;
    });
    busyUnionUs += Math.max(0, unionEndUs - unionStartUs);

    const firstStartUs = intervals[0].startUs;
    const lastEndUs = Math.max(...intervals.map((interval) => interval.endUs));
    const kernelSumUs = intervals.reduce((sum, interval) => sum + interval.durationUs, 0);
    const totalCostUs = intervals.reduce((sum, interval) => sum + interval.durationUs + interval.waitUs, 0);
    return {
      wall_ms: Math.max(0, lastEndUs - firstStartUs) / 1000,
      busy_union_ms: busyUnionUs / 1000,
      kernel_sum_ms: kernelSumUs / 1000,
      total_cost_ms: totalCostUs / 1000,
      first_start_ms: firstStartUs / 1000,
      last_end_ms: lastEndUs / 1000,
      merged_interval_count: intervals.reduce((count, interval, index) => (
        index > 0 && interval.startUs > Math.max(...intervals.slice(0, index).map((item) => item.endUs)) ? count + 1 : count
      ), 1),
      kernel_count: intervals.length,
    };
  }

  function coreMetricValue(perf, derived, key) {
    const explicit = perf?.core_event_metrics?.[key]
      ?? perf?.event_metrics?.[key]
      ?? perf?.[key];
    if (explicit != null && Number.isFinite(Number(explicit))) return Number(explicit);
    if (derived[key] != null && Number.isFinite(Number(derived[key]))) return Number(derived[key]);
    if (key === "kernel_sum_ms" && Number.isFinite(Number(perf?.time_us))) return Number(perf.time_us) / 1000;
    return NaN;
  }

  function eventLayerIndex(event) {
    const value = event?.layer_index ?? event?.instance_index;
    if (value == null || value === "") return null;
    const index = Number(value);
    return Number.isInteger(index) ? index : null;
  }

  function operatorRatio(events) {
    const durations = new Map();
    events.forEach((event) => {
      const name = String(event.op_type || event.name || "unknown");
      durations.set(name, (durations.get(name) || 0) + Math.max(0, number(event.duration_us)));
    });
    const total = Array.from(durations.values()).reduce((sum, value) => sum + value, 0);
    if (!(total > 0)) return {};
    return Object.fromEntries(Array.from(durations)
      .sort((left, right) => right[1] - left[1])
      .map(([name, duration]) => [name, duration / total * 100]));
  }

  function titleFor(nodeId, analysis, perf) {
    const raw = analysis?.name || perf?.module || nodeId.split("/").pop() || nodeId;
    if (nodeId.includes("/layers/dense_decoder_layer/") && nodeId !== "model/deepseek-v3.2-exp/layers/dense_decoder_layer") {
      return `Dense · ${raw}`;
    }
    if (nodeId.includes("/layers/moe_decoder_layer/") && nodeId !== "model/deepseek-v3.2-exp/layers/moe_decoder_layer") {
      return `MoE · ${raw}`;
    }
    if (nodeId.includes("/layers/mtp_layer/") && nodeId !== "model/deepseek-v3.2-exp/layers/mtp_layer") {
      return `MTP · ${raw}`;
    }
    return raw;
  }

  function laneKey(event) {
    const device = event.device_id == null ? "?" : event.device_id;
    const stream = event.stream_id == null ? "None" : event.stream_id;
    const core = event.accelerator_core || "UNKNOWN";
    return `D${device} · S${stream} · ${core}`;
  }

  function eventColor(event) {
    const core = String(event.accelerator_core || "").toUpperCase();
    if (core.includes("VECTOR")) return "#c678dd";
    if (core.includes("AI_CORE") || core.includes("MIX")) return "#61afef";
    if (core.includes("HCCL") || core.includes("COMM")) return "#e5c07b";
    if (core.includes("SDMA")) return "#56b6c2";
    return event.owner_node_id ? "#98c379" : "#7f848e";
  }

  function assertCompatible(analysisConfig, perfData, timelineData, analysisNodes, perfNodes) {
    const identity = [analysisConfig, perfData, timelineData].map((item) => `${item.model_id}::${item.report_id}`);
    if (new Set(identity).size !== 1) throw new Error(`Data identity mismatch: ${identity.join(", ")}`);

    const missingPerf = Array.from(perfNodes.keys()).filter((id) => !analysisNodes.has(id));
    const missingTimeline = Array.from(new Set(
      (timelineData.events || []).map((event) => event.owner_node_id).filter(Boolean)
    )).filter((id) => !analysisNodes.has(id));
    if (missingPerf.length || missingTimeline.length) {
      throw new Error(`Unresolved node IDs: perf=${missingPerf.length}, timeline=${missingTimeline.length}`);
    }
  }

  function createReportModel(analysisConfig, perfData, timelineData) {
    const analysisNodes = nodeIndex(analysisConfig);
    const perfNodes = nodeIndex(perfData);
    assertCompatible(analysisConfig, perfData, timelineData, analysisNodes, perfNodes);

    const events = Array.isArray(timelineData.events) ? timelineData.events : [];
    const eventsByOwner = new Map();
    events.forEach((event) => {
      if (!event.owner_node_id) return;
      if (!eventsByOwner.has(event.owner_node_id)) eventsByOwner.set(event.owner_node_id, []);
      eventsByOwner.get(event.owner_node_id).push(event);
    });

    const reports = {};
    const layerReports = {};
    perfNodes.forEach((perf, nodeId) => {
      const analysis = analysisNodes.get(nodeId) || {};
      const includeDescendantEvents = perf.metric_scope === "aggregate"
        || perf.metric_scope === "phase_aggregate"
        || perf.aggregate_kind
        || Array.isArray(perf.children);
      const directOwnedEvents = eventsByOwner.get(nodeId) || [];
      const structureInstanceEvents = directOwnedEvents.filter((event) => event.structure_instance_node_id === nodeId);
      const ownedEvents = includeDescendantEvents
        ? events.filter((event) => event.owner_node_id === nodeId || String(event.owner_node_id || "").startsWith(`${nodeId}/`))
        : (structureInstanceEvents.length ? structureInstanceEvents : directOwnedEvents);
      const derivedCoreMetrics = deriveCoreEventMetrics(ownedEvents);
      const opRatio = Object.entries(perf.op_ratio || {})
        .sort((a, b) => number(b[1]) - number(a[1]))
        .slice(0, 6)
        .map(([name, ratio]) => [name, `${formatNumber(ratio, 2)}%`]);
      const facts = [
        analysis.semantic,
        analysis.code_ref ? `Source: ${analysis.code_ref}` : "",
        perf.metric_scope ? `Metric scope: ${perf.metric_scope}` : "",
        Array.isArray(perf.instance_indices) ? `Instances: ${perf.instance_indices.join(", ")}` : "",
        perf.representative_instance != null ? `Representative instance: ${perf.representative_instance}` : "",
        `Timeline mapping: ${ownedEvents.length} direct event${ownedEvents.length === 1 ? "" : "s"}`
      ].filter(Boolean);

      const coreMetrics = [
        ["wall_ms", formatMs(coreMetricValue(perf, derivedCoreMetrics, "wall_ms"))],
        ["busy_union_ms", formatMs(coreMetricValue(perf, derivedCoreMetrics, "busy_union_ms"))],
        ["kernel_sum_ms", formatMs(coreMetricValue(perf, derivedCoreMetrics, "kernel_sum_ms"))],
        ["total_cost_ms", formatMs(coreMetricValue(perf, derivedCoreMetrics, "total_cost_ms"))],
      ];
      const metrics = [
        ["time share", `${formatNumber(perf.time_pct, 2)}%`],
        ["operators", formatNumber(perf.nops, 0)],
        ["HBM estimate", perf.hbm_mb == null ? "–" : `${formatNumber(perf.hbm_mb, 3)} MB`],
        ["MFU INT8", perf.mfu_int8_pct == null ? "–" : `${formatNumber(perf.mfu_int8_pct, 2)}%`],
        ["MFU BF16", perf.mfu_bf16_pct == null ? "–" : `${formatNumber(perf.mfu_bf16_pct, 2)}%`]
      ];

      reports[nodeId] = {
        nodeId,
        timeSharePct: number(perf.time_pct),
        dimension: analysis.node_kind || perf.metric_scope || "performance node",
        title: titleFor(nodeId, analysis, perf),
        metricShort: `${formatNumber(perf.time_pct, 2)}%`,
        summary: analysis.semantic || `${perf.module || nodeId} performance metrics for representative step ${perfData.representative_step}.`,
        instanceIndices: Array.isArray(perf.instance_indices)
          ? perf.instance_indices.map(Number).filter(Number.isInteger)
          : [],
        layerIndex: null,
        isLayerScoped: false,
        coreMetrics,
        coreMetricDetails: derivedCoreMetrics,
        metrics,
        facts,
        operators: opRatio,
        actions: []
      };

      reports[nodeId].instanceIndices.forEach((layerIndex) => {
        const scopedEvents = ownedEvents.filter((event) => eventLayerIndex(event) === layerIndex);
        const scopedDurationUs = scopedEvents.reduce(
          (sum, event) => sum + Math.max(0, number(event.duration_us)),
          0,
        );
        const scopedTimePct = number(perfData.total_time_us) > 0
          ? scopedDurationUs / number(perfData.total_time_us) * 100
          : NaN;
        const scopedCoreMetrics = deriveCoreEventMetrics(scopedEvents);
        const scopedOpRatio = operatorRatio(scopedEvents);
        const singleInstanceBackendMetric = reports[nodeId].instanceIndices.length === 1;
        const layerReport = {
          ...reports[nodeId],
          title: `${reports[nodeId].title} · Layer ${layerIndex}`,
          metricShort: Number.isFinite(scopedTimePct) && scopedEvents.length
            ? `${formatNumber(scopedTimePct, 2)}%`
            : "–",
          timeSharePct: Number.isFinite(scopedTimePct) && scopedEvents.length ? scopedTimePct : NaN,
          dimension: `${reports[nodeId].dimension} · layer ${layerIndex}`,
          layerIndex,
          isLayerScoped: true,
          coreMetrics: [
            ["wall_ms", formatMs(scopedCoreMetrics.wall_ms)],
            ["busy_union_ms", formatMs(scopedCoreMetrics.busy_union_ms)],
            ["kernel_sum_ms", formatMs(scopedCoreMetrics.kernel_sum_ms)],
            ["total_cost_ms", formatMs(scopedCoreMetrics.total_cost_ms)],
          ],
          coreMetricDetails: scopedCoreMetrics,
          metrics: [
            ["time share", Number.isFinite(scopedTimePct) && scopedEvents.length ? `${formatNumber(scopedTimePct, 2)}%` : "–"],
            ["operators", scopedEvents.length ? formatNumber(scopedEvents.length, 0) : "–"],
            ["HBM estimate", singleInstanceBackendMetric && perf.hbm_mb != null ? `${formatNumber(perf.hbm_mb, 3)} MB` : "–"],
            ["MFU INT8", singleInstanceBackendMetric && perf.mfu_int8_pct != null ? `${formatNumber(perf.mfu_int8_pct, 2)}%` : "–"],
            ["MFU BF16", singleInstanceBackendMetric && perf.mfu_bf16_pct != null ? `${formatNumber(perf.mfu_bf16_pct, 2)}%` : "–"],
          ],
          operators: Object.entries(scopedOpRatio)
            .slice(0, 6)
            .map(([name, ratio]) => [name, `${formatNumber(ratio, 2)}%`]),
        };
        if (!layerReports[layerIndex]) layerReports[layerIndex] = {};
        layerReports[layerIndex][nodeId] = layerReport;
      });
    });

    function unavailableLayerReport(report, layerIndex) {
      return {
        ...report,
        title: `${report.title} · Layer ${layerIndex}`,
        metricShort: "–",
        timeSharePct: NaN,
        dimension: `${report.dimension} · layer ${layerIndex}`,
        layerIndex,
        isLayerScoped: true,
        coreMetrics: ["wall_ms", "busy_union_ms", "kernel_sum_ms", "total_cost_ms"].map((key) => [key, "–"]),
        coreMetricDetails: {},
        metrics: [
          ["time share", "–"],
          ["operators", "–"],
          ["HBM estimate", "–"],
          ["MFU INT8", "–"],
          ["MFU BF16", "–"],
        ],
        operators: [],
      };
    }

    function reportsForLayer(layerIndex) {
      if (layerIndex == null || layerIndex === "" || !Number.isInteger(Number(layerIndex))) return reports;
      const selectedLayerIndex = Number(layerIndex);
      return Object.fromEntries(Object.entries(reports).map(([nodeId, report]) => {
        if (!report.instanceIndices.length) return [nodeId, report];
        return [nodeId, layerReports[selectedLayerIndex]?.[nodeId]
          || unavailableLayerReport(report, selectedLayerIndex)];
      }));
    }

    const reportOrder = Object.keys(reports).sort((left, right) => {
      const leftPerf = perfNodes.get(left);
      const rightPerf = perfNodes.get(right);
      return number(rightPerf?.time_us) - number(leftPerf?.time_us) || left.localeCompare(right);
    });

    const timeline = events.map((event) => {
      const lane = laneKey(event);
      return {
        eventId: event.event_id,
        name: event.name || event.op_type || event.event_id,
        nodeId: event.owner_node_id || "",
        instanceIndex: eventLayerIndex(event),
        startUs: number(event.ts_us),
        endUs: number(event.end_us, number(event.ts_us) + number(event.duration_us)),
        wallUs: number(event.duration_us),
        opUs: number(event.duration_us),
        waitUs: number(event.wait_time_us),
        lane,
        stream: event.stream_id == null ? "None" : String(event.stream_id),
        device: event.device_id,
        core: event.accelerator_core || "UNKNOWN",
        dominant: lane,
        category: event.mapping_status || (event.owner_node_id ? "direct" : "unmapped"),
        color: eventColor(event)
      };
    }).sort((left, right) => left.startUs - right.startUs || right.wallUs - left.wallUs);

    const laneMap = new Map();
    timeline.forEach((event) => {
      if (!laneMap.has(event.lane)) {
        laneMap.set(event.lane, {
          lane: event.lane,
          stream: event.stream,
          device: event.device,
          core: event.core,
          opUs: 0,
          waitUs: 0,
          ops: 0
        });
      }
      const lane = laneMap.get(event.lane);
      lane.opUs += event.opUs;
      lane.waitUs += event.waitUs;
      lane.ops += 1;
    });
    const streamSummary = Array.from(laneMap.values()).sort((left, right) => {
      if (left.device !== right.device) return number(left.device) - number(right.device);
      if (left.stream !== right.stream) return number(left.stream, Number.MAX_SAFE_INTEGER) - number(right.stream, Number.MAX_SAFE_INTEGER);
      return left.core.localeCompare(right.core);
    });

    const mapping = timelineData.mapping_summary || {};
    const mappedEvents = number(mapping.mapped_events, timeline.filter((event) => event.nodeId).length);
    const unmappedEvents = number(mapping.unmapped_events, timeline.length - mappedEvents);
    const mappingCoveragePct = number(mapping.mapping_coverage_pct, timeline.length ? mappedEvents / timeline.length * 100 : 0);
    const stepSummary = {
      step: timelineData.representative_step ?? perfData.representative_step,
      decodeLatencyUs: number(perfData.summary?.decode_latency_ms) * 1000,
      kernelSumUs: number(perfData.summary?.kernel_sum_ms) * 1000,
      mappingCoveragePct,
      mappedEvents,
      unmappedEvents,
      eventCount: number(timelineData.event_count, timeline.length),
      globalMfuInt8Pct: number(perfData.summary?.global_mfu_int8_pct)
    };

    return {
      identity: {
        modelId: perfData.model_id,
        reportId: perfData.report_id,
        idNamespace: perfData.id_namespace,
        sku: perfData.sku
      },
      reports,
      reportsForLayer,
      reportOrder,
      timeline,
      streamSummary,
      stepSummary,
      counts: {
        analysisNodes: analysisNodes.size,
        perfNodes: perfNodes.size,
        timelineEvents: timeline.length,
        mappedEvents,
        unmappedEvents,
        lanes: streamSummary.length
      }
    };
  }

  global.DeepSeekReportData = {
    createReportModel
  };
})(window);
