(function registerDeepSeekArchitectureData(global) {
  "use strict";

  const NODE_WIDTH = 336;
  const NODE_HEIGHT = 56;
  const CLUSTER_PADDING_X = 28;
  const CLUSTER_TITLE_HEIGHT = 46;
  // Repeated blocks need two distinct header rows: the block title and layer pager.
  const REPEAT_CLUSTER_TITLE_HEIGHT = 112;
  const REPEAT_PAGER_BUTTON_SIZE = 36;
  const REPEAT_PAGER_CONTROL_GAP = 12;
  const REPEAT_PAGER_DOT_STEP = 24;
  const REPEAT_PAGER_SIDE_PADDING = 52;
  const RESIDUAL_LANE_INSET = 18;
  const RESIDUAL_PORT_SEPARATION = 8;
  const EDGE_NODE_GAP = 6;
  const CLUSTER_PADDING_BOTTOM = 26;
  const ROW_GAP = 40;
  const BRANCH_ROW_GAP = 64;
  const TOP_LEVEL_GAP = 44;
  const BRANCH_GAP = 36;
  const SECTION_GAP = 72;
  const GRAPH_PADDING = 40;

  function allItems(graphSpec) {
    const items = [];
    function visit(item, parentId = "", depth = 0) {
      if (!item || typeof item.id !== "string" || !item.id) return;
      items.push({ item, parentId, depth });
      (item.children || []).forEach((child) => visit(child, item.id, depth + 1));
    }
    (graphSpec.roots || []).forEach((root) => visit(root));
    return items;
  }

  function applyExpertInventory(graphSpec, inventory) {
    if (!inventory || inventory.available === false || !Number.isFinite(Number(inventory.declared?.routed_experts))) {
      return graphSpec;
    }
    const graph = JSON.parse(JSON.stringify(graphSpec));
    const declared = Number(inventory.declared.routed_experts);
    const shared = Number(inventory.declared.shared_experts || 0);
    const activated = Number(inventory.declared.experts_per_token || 0);
    const local = Number(inventory.expert_parallelism?.local_routed_experts || 0);
    const separable = inventory.measurability?.separable_per_expert === true;
    const hasResolvedResidency = Array.isArray(inventory.expert_parallelism?.resident_expert_indices)
      && inventory.expert_parallelism.resident_expert_indices.length > 0;
    const representativeIndices = declared > 2 ? [0, 1, null, declared - 1] : Array.from({ length: declared }, (_, index) => index);

    function decorate(item) {
      if (!item || !Array.isArray(item.children)) return;
      item.children.forEach(decorate);
      const roleOf = (child) => child.architectureRole || child.attributes?.architectureRole
        || child.attributes?.architecture_role || child.expertRole || "";
      const roleAliases = {
        gate: ["moe.router", "router", "gate"],
        dispatch: ["moe.dispatch", "dispatch"],
        routed_experts: ["moe.routed_experts", "routed_experts", "experts"],
        shared_expert: ["moe.shared_expert", "shared_expert", "shared_experts"],
        combine: ["moe.combine", "combine", "expert_merge"],
      };
      const byRole = new Map();
      item.children.forEach((child) => {
        const leaf = String(child.id).split("/").at(-1);
        Object.entries(roleAliases).forEach(([canonical, aliases]) => {
          if (aliases.includes(roleOf(child)) || aliases.includes(leaf)) byRole.set(canonical, child);
        });
      });
      if (["gate", "routed_experts", "shared_expert", "dispatch", "combine"].some((role) => !byRole.has(role))) return;

      const gate = byRole.get("gate");
      const dispatch = byRole.get("dispatch");
      const routed = byRole.get("routed_experts");
      const sharedExpert = byRole.get("shared_expert");
      const combine = byRole.get("combine");
      const expertSummary = {
        declaredRoutedExperts: declared,
        declaredSharedExperts: shared,
        expertsPerToken: activated,
        localRoutedExperts: local,
        expertParallelSize: Number(inventory.expert_parallelism?.moe_ep_size || 0),
        residencyResolved: hasResolvedResidency,
        perExpertTimingSeparable: separable,
        measurementNote: inventory.measurability?.reason || "",
      };
      gate.label = "Router";
      gate.colorKey = gate.colorKey || "opv:gate";
      gate.expertRole = "router";
      dispatch.label = "Dispatch";
      dispatch.expertRole = "dispatch";
      dispatch.visualHidden = true;
      sharedExpert.label = shared === 1 ? "Shared Expert" : `Shared Experts ×${shared}`;
      sharedExpert.colorKey = "opv:linear";
      sharedExpert.expertRole = "shared-expert";
      combine.label = "Combine";
      combine.expertRole = "combine";

      routed.label = "Routed Experts";
      routed.kind = "module";
      routed.expertRole = "routed-group";
      routed.defaultCollapsed = false;
      routed.attributes = { ...(routed.attributes || {}), expertInventory: expertSummary };
      const expertNodes = representativeIndices.map((expertIndex, position) => ({
        id: `${routed.id}/declared_${expertIndex == null ? "ellipsis" : expertIndex}`,
        label: expertIndex == null ? "…" : `Expert ${expertIndex}`,
        kind: "op",
        type: "op",
        origin: "synthetic",
        synthetic: true,
        dataState: "synthetic_visual",
        selectable: false,
        colorKey: "opv:linear",
        expertRole: expertIndex == null ? "expert-ellipsis" : "declared-expert",
        attributes: {
          semantic: expertIndex == null
            ? `${declared - 3} additional declared routed experts`
            : "Declared routed expert; this capture does not identify the resident EP rank",
          expertIndex,
          representativePosition: position,
        },
        children: [],
      }));
      routed.layoutRows = [[gate.id], expertNodes.map((expert) => expert.id)];
      routed.children = [gate, dispatch, ...expertNodes];
      item.label = "Mixture of Experts";
      item.expertRole = "moe-container";
      item.layoutRows = [[routed.id, sharedExpert.id], [combine.id]];
      item.children = [routed, sharedExpert, combine];
    }
    graph.roots.forEach(decorate);

    const expertGroups = allItems(graph).filter(({ item }) => item.expertRole === "routed-group");
    const groupById = new Map(expertGroups.map(({ item }) => [item.id, item]));
    const hiddenDispatchIds = new Set(allItems(graph).filter(({ item }) => item.expertRole === "dispatch").map(({ item }) => item.id));
    const gateToGroup = new Map(expertGroups.map(({ item }) => {
      const gate = item.children.find((child) => child.expertRole === "router");
      return [gate?.id, item];
    }).filter(([gateId]) => gateId));
    const replacementEdges = [];
    (graph.edges || []).forEach((edge) => {
      if (hiddenDispatchIds.has(edge.source) || hiddenDispatchIds.has(edge.target)) return;
      if (groupById.has(edge.source)) {
        const group = groupById.get(edge.source);
        group.children.filter((child) => child.expertRole === "declared-expert").forEach((expert) => {
          replacementEdges.push({ ...edge, id: `${edge.id}::${expert.id}`, source: expert.id, projectionOnly: true });
        });
      } else if (gateToGroup.has(edge.source) && edge.target.endsWith("/shared_expert")) {
        replacementEdges.push(edge);
        const group = gateToGroup.get(edge.source);
        group.children.filter((child) => child.expertRole === "declared-expert").forEach((expert) => {
          replacementEdges.push({ ...edge, id: `expert-routing::${edge.source}->${expert.id}`, target: expert.id, projectionOnly: true });
        });
      } else {
        replacementEdges.push(edge);
      }
    });
    graph.edges = replacementEdges;
    graph.metadata = { ...(graph.metadata || {}), expertInventoryApplied: true };
    return graph;
  }

  function itemIndex(graphSpec) {
    return new Map(allItems(graphSpec).map((entry) => [entry.item.id, entry]));
  }

  function rawInstanceIndices(item) {
    return Array.isArray(item?.instanceIndices)
      ? item.instanceIndices.map(Number).filter(Number.isFinite)
      : [];
  }

  function lowestCommonAncestorId(entries, index) {
    if (!entries.length) return "";
    const chains = entries.map((entry) => {
      const chain = [entry.item.id];
      let current = entry;
      while (current?.parentId) {
        chain.unshift(current.parentId);
        current = index.get(current.parentId);
      }
      return chain;
    });
    let common = "";
    for (let position = 0; position < Math.min(...chains.map((chain) => chain.length)); position += 1) {
      const candidate = chains[0][position];
      if (!chains.every((chain) => chain[position] === candidate)) break;
      common = candidate;
    }
    return common;
  }

  // Stage 2 intentionally propagates instance_indices through a folded layer so every
  // descendant can filter its own metrics. Those metric scopes are not independent visual
  // repeats. Build one UI-only pager at the outer layer template, or promote it to the
  // common container when a model has sibling templates (dense/MoE, layer 0/rest, etc.).
  function visualRepeatPlan(graphSpec) {
    const entries = allItems(graphSpec);
    const index = new Map(entries.map((entry) => [entry.item.id, entry]));
    const candidates = entries.filter((entry) => {
      if (!rawInstanceIndices(entry.item).length) return false;
      // Runtime iteration modules can also carry numeric instance indices, but they are not
      // model-layer navigation (for example DeepSeek's MTP layer after decoder layer 60).
      if (/(mtp|runtime|auxiliary|scaffold)/i.test(
        `${entry.item.id}/${entry.item.label || ""}`,
      )) return false;
      let current = index.get(entry.parentId);
      while (current) {
        if (rawInstanceIndices(current.item).length) return false;
        current = index.get(current.parentId);
      }
      return true;
    });
    const layerIndices = [...new Set(candidates.flatMap(({ item }) => rawInstanceIndices(item)))]
      .sort((a, b) => a - b);
    if (!candidates.length || layerIndices.length < 2) {
      return { hostId: "", instanceIndices: [], suppressedIds: new Set(), templates: [] };
    }
    const hostId = candidates.length === 1
      ? candidates[0].item.id
      : lowestCommonAncestorId(candidates, index);
    const candidateIds = new Set(candidates.map(({ item }) => item.id));
    const suppressedIds = new Set();
    entries.forEach((entry) => {
      if (entry.item.id === hostId) return;
      let current = entry;
      while (current) {
        if (candidateIds.has(current.item.id)) {
          suppressedIds.add(entry.item.id);
          break;
        }
        current = index.get(current.parentId);
      }
    });
    return {
      hostId,
      instanceIndices: layerIndices,
      suppressedIds,
      templates: candidates.map(({ item }) => ({
        id: item.id,
        backendNodeId: item.backendNodeId || "",
        instanceIndices: rawInstanceIndices(item),
      })),
    };
  }

  function pagerHostIdForLayer(repeatPlan, selectedLayerIndex = null) {
    if (repeatPlan.templates.length <= 1) return repeatPlan.hostId;
    if (selectedLayerIndex != null) {
      const matchingTemplates = repeatPlan.templates.filter((template) => (
        template.instanceIndices.includes(Number(selectedLayerIndex))
      ));
      if (matchingTemplates.length) return matchingTemplates[0].id;
    }
    return repeatPlan.templates[0]?.id || repeatPlan.hostId;
  }

  function repeatInfoFor(item, repeatPlan, selectedLayerIndex = null) {
    // The common ancestor of sibling layer templates is only a navigation-state host. It
    // may also contain global stages such as embedding and lm_head, so the visual pager
    // belongs to the currently active decoder template rather than that ancestor frame.
    if (item.id === pagerHostIdForLayer(repeatPlan, selectedLayerIndex)) {
      return { count: repeatPlan.instanceIndices.length, indices: repeatPlan.instanceIndices };
    }
    if (repeatPlan.suppressedIds.has(item.id)) return { count: 0, indices: [] };
    const indices = rawInstanceIndices(item);
    return {
      count: Number(item.repeatCount || indices.length || 0),
      indices,
    };
  }

  function estimateLayoutTextWidth(value) {
    return Math.ceil(Array.from(String(value || "")).reduce((width, character) => {
      if (/\s/.test(character)) return width + 4;
      if (/[ilI1|.,:;'`]/.test(character)) return width + 4.5;
      if (/[MW@#%&]/.test(character)) return width + 10;
      return width + (character.codePointAt(0) > 255 ? 13 : 7.5);
    }, 0));
  }

  function leafMinimumWidth(item) {
    // Leave room for the metric/repeat badges on the left and the expand action on a
    // collapsed node. The centered title must never be clipped by either reservation.
    if (item.expertRole === "expert-ellipsis") return 80;
    if (item.expertRole === "declared-expert") return Math.max(260, estimateLayoutTextWidth(item.label || item.id) + 96);
    return Math.max(NODE_WIDTH, estimateLayoutTextWidth(item.label || item.id) + 128);
  }

  function clusterTitleMinimumWidth(item, repeatInfo) {
    const label = item.label || item.id;
    if (repeatInfo.count > 1) {
      const lastIndex = repeatInfo.indices.at(-1) ?? repeatInfo.count - 1;
      const pagerLabel = `${label} · Layer ${lastIndex} (${repeatInfo.count}/${repeatInfo.count})`;
      return estimateLayoutTextWidth(pagerLabel) + 48;
    }
    // Cluster labels start 20 px from the left; reserve the right corner for the 28 px
    // collapse control plus its edge gap and breathing room.
    return estimateLayoutTextWidth(label) + 88;
  }

  function metricReportFor(item, reports) {
    return item.backendNodeId ? reports[item.backendNodeId] : null;
  }

  function typeLabelFor(item, report, collapsed) {
    const sourceKind = item.typeLabel || item.kind || (collapsed ? "Module" : "Node");
    return collapsed ? `${sourceKind} · folded` : sourceKind;
  }

  // ── DAG-based topological ranking for auto-layout ──
  // Computes ranks for source-scope items: rank = longest path from any source node.
  // Same-rank siblings have no data dependency and can share a row.
  // Returns { ranks: Map<id, rank>, hasCycles: bool }
  function computeDagRanks(graphSpec) {
    const ranks = new Map();
    const activationEdges = (graphSpec.edges || []).filter((e) => e.semanticEdgeType === "activation");
    if (!activationEdges.length) return { ranks, hasCycles: false };

    // Build adjacency and in-degree
    const outAdj = new Map();
    const inDegree = new Map();
    const allIds = new Set();
    for (const edge of activationEdges) {
      if (!outAdj.has(edge.source)) outAdj.set(edge.source, []);
      outAdj.get(edge.source).push(edge.target);
      inDegree.set(edge.target, (inDegree.get(edge.target) || 0) + 1);
      if (!inDegree.has(edge.source)) inDegree.set(edge.source, 0);
      allIds.add(edge.source);
      allIds.add(edge.target);
    }

    // Kahn's algorithm: start with in-degree 0
    const queue = [];
    for (const id of allIds) {
      if ((inDegree.get(id) || 0) === 0) {
        queue.push(id);
        ranks.set(id, 0);
      }
    }

    let processed = 0;
    while (queue.length) {
      const current = queue.shift();
      processed++;
      const currentRank = ranks.get(current) || 0;
      for (const target of outAdj.get(current) || []) {
        const newInDegree = (inDegree.get(target) || 0) - 1;
        inDegree.set(target, newInDegree);
        ranks.set(target, Math.max(ranks.get(target) || 0, currentRank + 1));
        if (newInDegree === 0) queue.push(target);
      }
    }

    const hasCycles = processed < allIds.size;
    // For nodes not reachable from any source, assign rank 0
    for (const id of allIds) {
      if (!ranks.has(id)) ranks.set(id, 0);
    }

    return { ranks, hasCycles };
  }

  // Cached DAG ranks — computed once in projectGraph, consumed by rowsFor
  let _cachedDagRanks = null;
  let _cachedDagHasCycles = false;
  let _currentGraphSpec = null;  // set by projectGraph before layout

  function dagRanks(graphSpec) {
    if (!_cachedDagRanks) {
      const result = computeDagRanks(graphSpec);
      _cachedDagRanks = result.ranks;
      _cachedDagHasCycles = result.hasCycles;
    }
    return { ranks: _cachedDagRanks, hasCycles: _cachedDagHasCycles };
  }

  function clearDagRankCache() {
    _cachedDagRanks = null;
    _cachedDagHasCycles = false;
  }

  function minDagRankForItem(itemId, rankMap) {
    // If the item itself has a rank, use it
    if (rankMap.has(itemId)) return rankMap.get(itemId);
    // Otherwise, it's a container; use the rank of its rightmost (max) predecessor
    // For layout purposes, a container's rank is determined by its leaf descendants
    return null;
  }

  function rowLayout(children, gap = BRANCH_GAP) {
    const width = children.reduce((sum, child) => sum + child.width, 0)
      + Math.max(0, children.length - 1) * gap;
    const height = children.length ? Math.max(...children.map((child) => child.height)) : 0;
    let x = 0;
    const placements = children.map((child) => {
      const placement = { child, x, y: (height - child.height) / 2 };
      x += child.width + gap;
      return placement;
    });
    return { width, height, placements };
  }

  function rowsLayout(rows, rowGap = ROW_GAP) {
    const rowMeasures = rows.filter((row) => row.length).map((row) => rowLayout(row));
    const width = rowMeasures.length ? Math.max(...rowMeasures.map((row) => row.width)) : 0;
    let y = 0;
    const placements = [];
    rowMeasures.forEach((row, index) => {
      row.placements.forEach((placement) => placements.push({
        child: placement.child,
        x: (width - row.width) / 2 + placement.x,
        y: y + placement.y,
      }));
      y += row.height;
      if (index < rowMeasures.length - 1) {
        const nextRow = rowMeasures[index + 1];
        const branchTransition = row.placements.length !== nextRow.placements.length
          && (row.placements.length > 1 || nextRow.placements.length > 1);
        y += branchTransition ? Math.max(rowGap, BRANCH_ROW_GAP) : rowGap;
      }
    });
    return { width, height: y, placements };
  }

  function rowsFor(item, childLayouts) {
    const byId = new Map(childLayouts.map((layout) => [layout.item.id, layout]));
    const used = new Set();
    const rows = [];

    if (Array.isArray(item.layoutRows)) {
      item.layoutRows.forEach((rowIds) => {
        const row = (Array.isArray(rowIds) ? rowIds : [rowIds])
          .map((id) => byId.get(id))
          .filter(Boolean);
        row.forEach((layout) => used.add(layout.item.id));
        if (row.length) rows.push(row);
      });
    } else if (item.layout === "parallel") {
      childLayouts.forEach((layout) => used.add(layout.item.id));
      rows.push(childLayouts);
    } else {
      // No explicit layoutRows: try DAG topological auto-rank
      const spec = _currentGraphSpec || { edges: [] };
      const { ranks } = dagRanks(spec);
      const childRankEntries = [];
      let hasRanks = false;

      for (const layout of childLayouts) {
        const rank = ranks.has(layout.item.id) ? ranks.get(layout.item.id) : 0;
        // Also check descendants: a container's effective rank is max of its leaf descendants
        const leafRanks = [];
        (function collectLeafRanks(it) {
          if (!it) return;
          if (ranks.has(it.id)) { leafRanks.push(ranks.get(it.id)); return; }
          (it.children || []).forEach(collectLeafRanks);
        })(layout.item);
        const effectiveRank = leafRanks.length ? Math.max(...leafRanks) : rank;
        if (ranks.has(layout.item.id) || leafRanks.length) hasRanks = true;
        childRankEntries.push({ layout, rank: effectiveRank });
      }

      if (hasRanks) {
        // Group by rank
        const maxRank = Math.max(...childRankEntries.map((e) => e.rank), 0);
        for (let r = 0; r <= maxRank; r++) {
          const row = childRankEntries
            .filter((e) => e.rank === r)
            .map((e) => e.layout);
          if (row.length) {
            row.forEach((layout) => used.add(layout.item.id));
            rows.push(row);
          }
        }
      }
    }

    childLayouts.forEach((layout) => {
      if (used.has(layout.item.id)) return;
      rows.push([layout]);
    });
    return rows;
  }

  function layoutItem(item, collapsedIds, repeatPlan, selectedLayerIndex = null) {
    const children = item.children || [];
    const collapsed = children.length > 0 && collapsedIds.has(item.id);
    if (!children.length || collapsed) {
      return {
        item,
        collapsed,
        width: leafMinimumWidth(item),
        height: Number(item.height) || NODE_HEIGHT,
        placements: [],
      };
    }

    const childLayouts = children
      .filter((child) => {
        if (child.visualHidden === true) return false;
        const template = repeatPlan.templates.find((entry) => entry.id === child.id);
        return !template || selectedLayerIndex == null
          || template.instanceIndices.includes(Number(selectedLayerIndex));
      })
      .map((child) => layoutItem(child, collapsedIds, repeatPlan, selectedLayerIndex));
    const rowGap = item.synthetic ? TOP_LEVEL_GAP : ROW_GAP;
    const content = rowsLayout(rowsFor(item, childLayouts), rowGap);
    const repeatInfo = repeatInfoFor(item, repeatPlan, selectedLayerIndex);
    const titleHeight = repeatInfo.count > 1
      ? REPEAT_CLUSTER_TITLE_HEIGHT
      : CLUSTER_TITLE_HEIGHT;
    const repeatCount = repeatInfo.count;
    const repeatPagerWidth = repeatCount > 1
      ? REPEAT_PAGER_BUTTON_SIZE * 2
        + REPEAT_PAGER_CONTROL_GAP * 2
        + repeatCount * REPEAT_PAGER_DOT_STEP
        + REPEAT_PAGER_SIDE_PADDING * 2
      : 0;
    const width = Math.max(
      NODE_WIDTH + CLUSTER_PADDING_X * 2,
      content.width + CLUSTER_PADDING_X * 2,
      repeatPagerWidth,
      clusterTitleMinimumWidth(item, repeatInfo),
    );
    const placements = content.placements.map((placement) => ({
      child: placement.child,
      x: CLUSTER_PADDING_X + (content.width
        ? (width - CLUSTER_PADDING_X * 2 - content.width) / 2
        : 0) + placement.x,
      y: titleHeight + placement.y,
    }));

    return {
      item,
      collapsed: false,
      width,
      height: titleHeight + content.height + CLUSTER_PADDING_BOTTOM,
      placements,
    };
  }

  function visualProps(item) {
    return {
      backendNodeId: item.backendNodeId || undefined,
      dataState: item.dataState || (item.backendNodeId ? "mapped" : "source_only"),
      origin: item.origin || "source",
      selectable: item.selectable === true,
      mappingKind: item.mappingKind || undefined,
      expertRole: item.expertRole || undefined,
    };
  }

  function visualPropsForReport(item, report) {
    const common = visualProps(item);
    if (common.dataState === "source_only" || !item.backendNodeId) return common;
    const hasCurrentPerformance = report
      && Number.isFinite(Number(report.timeSharePct))
      && String(report.metricShort || "").trim() !== "–";
    return {
      ...common,
      dataState: hasCurrentPerformance ? common.dataState : "no_performance_data",
    };
  }

  function resolveVisibleEndpoint(itemId, direction, visibleNodes, collapsedIds, index) {
    if (visibleNodes.has(itemId)) return itemId;
    let current = index.get(itemId);
    while (current?.parentId) {
      if (collapsedIds.has(current.parentId) && visibleNodes.has(current.parentId)) return current.parentId;
      current = index.get(current.parentId);
    }

    const entry = index.get(itemId);
    if (!entry) return "";
    const candidates = [];
    function collect(item) {
      if (visibleNodes.has(item.id)) {
        candidates.push(item.id);
        return;
      }
      (item.children || []).forEach(collect);
    }
    collect(entry.item);
    if (!candidates.length) return "";
    return direction === "source" ? candidates[candidates.length - 1] : candidates[0];
  }

  function layerSelectionEdgeProjection(graphSpec, repeatPlan, selectedLayerIndex) {
    if (selectedLayerIndex == null || repeatPlan.templates.length <= 1) {
      return { suppressedEdgeIds: new Set(), bridgeEdges: [] };
    }
    const activeTemplates = repeatPlan.templates.filter((template) => (
      template.instanceIndices.includes(Number(selectedLayerIndex))
    ));
    if (activeTemplates.length !== 1) {
      return { suppressedEdgeIds: new Set(), bridgeEdges: [] };
    }

    const activeTemplateId = activeTemplates[0].id;
    const templateIds = new Set(repeatPlan.templates.map((template) => template.id));
    const activationEdges = (graphSpec.edges || []).filter((edge) => (
      edge.semanticEdgeType === "activation"
      && (templateIds.has(edge.source) || templateIds.has(edge.target))
    ));
    if (!activationEdges.length) {
      return { suppressedEdgeIds: new Set(), bridgeEdges: [] };
    }

    const incoming = new Map();
    const outgoing = new Map();
    activationEdges.forEach((edge) => {
      if (!incoming.has(edge.target)) incoming.set(edge.target, []);
      if (!outgoing.has(edge.source)) outgoing.set(edge.source, []);
      incoming.get(edge.target).push(edge);
      outgoing.get(edge.source).push(edge);
    });

    function boundaryPaths(direction) {
      const results = [];
      const stack = [{ nodeId: activeTemplateId, path: [], visited: new Set([activeTemplateId]) }];
      while (stack.length) {
        const current = stack.pop();
        const adjacent = direction === "incoming"
          ? incoming.get(current.nodeId) || []
          : outgoing.get(current.nodeId) || [];
        adjacent.forEach((edge) => {
          const nextId = direction === "incoming" ? edge.source : edge.target;
          if (current.visited.has(nextId)) return;
          const path = direction === "incoming"
            ? [edge, ...current.path]
            : [...current.path, edge];
          if (!templateIds.has(nextId)) {
            results.push({ boundaryId: nextId, path });
            return;
          }
          stack.push({
            nodeId: nextId,
            path,
            visited: new Set([...current.visited, nextId]),
          });
        });
      }
      return results;
    }

    const bridgeEdges = [];
    const seenBridges = new Set();
    function addBridge(source, target, path, direction) {
      if (!path.length) return;
      const key = `${source}->${target}`;
      if (seenBridges.has(key)) return;
      seenBridges.add(key);
      const adjacentEdge = direction === "incoming" ? path.at(-1) : path[0];
      bridgeEdges.push({
        ...adjacentEdge,
        id: `layer-selection::${source}->${target}`,
        source,
        target,
        sourceItemId: source,
        targetItemId: target,
        projectionOnly: true,
        projectedThroughLayerSelection: true,
        projectedFromEdgeIds: path.map((edge) => edge.id),
        provenance: [...new Set(path.flatMap((edge) => edge.provenance || []))],
      });
    }

    boundaryPaths("incoming").forEach(({ boundaryId, path }) => {
      addBridge(boundaryId, activeTemplateId, path, "incoming");
    });
    boundaryPaths("outgoing").forEach(({ boundaryId, path }) => {
      addBridge(activeTemplateId, boundaryId, path, "outgoing");
    });

    return {
      suppressedEdgeIds: new Set(activationEdges.map((edge) => edge.id)),
      bridgeEdges,
    };
  }

  function projectEdges(
    graphSpec,
    nodes,
    clusters,
    collapsedIds,
    repeatPlan,
    selectedLayerIndex = null,
  ) {
    const index = itemIndex(graphSpec);
    const visibleNodes = new Set(nodes.map((node) => node.id));
    const visibleNodeById = new Map(nodes.map((node) => [node.id, node]));
    const clusterById = new Map(clusters.map((cluster) => [cluster.id, cluster]));
    const residualSources = new Set((graphSpec.edges || [])
      .filter((edge) => edge.semanticEdgeType === "residual")
      .map((edge) => edge.source));
    const residualTargets = new Set((graphSpec.edges || [])
      .filter((edge) => edge.semanticEdgeType === "residual")
      .map((edge) => edge.target));

    function ancestorClusterIds(node) {
      const ids = [];
      let clusterId = node?.parent;
      while (clusterId && clusterById.has(clusterId)) {
        ids.push(clusterId);
        clusterId = clusterById.get(clusterId)?.parent;
      }
      return ids;
    }

    // Match the Qwen gold-standard residual treatment: fixed left-side ports, one stable
    // outer lane inside the nearest common cluster, and a rounded orthogonal polyline.
    // This is renderer geometry only; the source/target still come exclusively from JSON.
    function residualRouting(sourceNode, targetNode) {
      const targetAncestors = new Set(ancestorClusterIds(targetNode));
      const commonClusterId = ancestorClusterIds(sourceNode)
        .find((clusterId) => targetAncestors.has(clusterId));
      const commonCluster = clusterById.get(commonClusterId);
      const sourceLeft = Number(sourceNode.x) - Number(sourceNode.width) / 2;
      const targetLeft = Number(targetNode.x) - Number(targetNode.width) / 2;
      const laneX = commonCluster
        ? Number(commonCluster.x) + RESIDUAL_LANE_INSET
        : Math.max(8, Math.min(sourceLeft, targetLeft) - 48);
      const sourceY = Number(sourceNode.y)
        + (residualTargets.has(sourceNode.id) ? RESIDUAL_PORT_SEPARATION : 0);
      const targetY = Number(targetNode.y)
        - (residualSources.has(targetNode.id) ? RESIDUAL_PORT_SEPARATION : 0);
      return {
        sourceAnchor: "left",
        targetAnchor: "left",
        sourcePoint: { x: sourceLeft - EDGE_NODE_GAP, y: sourceY },
        targetPoint: { x: targetLeft - EDGE_NODE_GAP, y: targetY },
        curve: "orthogonal",
        route: "orthogonal",
        waypoints: [
          { x: laneX, y: sourceY },
          { x: laneX, y: targetY },
        ],
        cornerRadius: 22,
        dashed: true,
        fixedPorts: true,
        routingPolicy: "residual_outer_left",
        residualLaneClusterId: commonClusterId || undefined,
      };
    }

    // Parameters and reusable state are side inputs, even when their source happens to
    // sit far above the consumer. Route long spans through the outer edge of the nearest
    // common frame so they do not cut diagonally across layer pagers and compute nodes.
    function parameterRouting(sourceNode, targetNode, dx, dy) {
      if (Math.abs(dx) > Math.abs(dy)) {
        return {
          sourceAnchor: { side: dx >= 0 ? "right" : "left", dx: dx >= 0 ? EDGE_NODE_GAP : -EDGE_NODE_GAP },
          targetAnchor: { side: dx >= 0 ? "left" : "right", dx: dx >= 0 ? -EDGE_NODE_GAP : EDGE_NODE_GAP },
          curve: "horizontal",
          dashed: true,
          routingPolicy: "parameter_side_input",
        };
      }
      const targetAncestors = new Set(ancestorClusterIds(targetNode));
      const commonClusterId = ancestorClusterIds(sourceNode)
        .find((clusterId) => targetAncestors.has(clusterId));
      const commonCluster = clusterById.get(commonClusterId);
      const useRightLane = Number(sourceNode.x) >= Number(targetNode.x);
      const sourceX = Number(sourceNode.x)
        + (useRightLane ? Number(sourceNode.width) / 2 + EDGE_NODE_GAP : -Number(sourceNode.width) / 2 - EDGE_NODE_GAP);
      const sourceY = Number(sourceNode.y);
      const targetX = Number(targetNode.x);
      const targetY = Number(targetNode.y) - Number(targetNode.height) / 2 - EDGE_NODE_GAP;
      const laneX = commonCluster
        ? Number(commonCluster.x) + (useRightLane
          ? Number(commonCluster.width) - RESIDUAL_LANE_INSET
          : RESIDUAL_LANE_INSET)
        : sourceX + (useRightLane ? 48 : -48);
      return {
        sourcePoint: { x: sourceX, y: sourceY },
        targetPoint: { x: targetX, y: targetY },
        sourceAnchor: useRightLane ? "right" : "left",
        targetAnchor: "top",
        curve: "orthogonal",
        route: "orthogonal",
        waypoints: [
          { x: laneX, y: sourceY },
          { x: laneX, y: targetY },
        ],
        cornerRadius: 16,
        fixedPorts: true,
        dashed: true,
        routingPolicy: "parameter_outer_side",
        parameterLaneClusterId: commonClusterId || undefined,
      };
    }
    const layerProjection = layerSelectionEdgeProjection(
      graphSpec,
      repeatPlan,
      selectedLayerIndex,
    );
    const visibleEdges = [
      ...(graphSpec.edges || []).filter((edge) => !layerProjection.suppressedEdgeIds.has(edge.id)),
      ...layerProjection.bridgeEdges,
    ];
    return visibleEdges.map((edge) => {
      const source = resolveVisibleEndpoint(edge.source, "source", visibleNodes, collapsedIds, index);
      const target = resolveVisibleEndpoint(edge.target, "target", visibleNodes, collapsedIds, index);
      if (!source || !target || source === target) return null;
      const sourceNode = visibleNodeById.get(source);
      const targetNode = visibleNodeById.get(target);
      const dx = Number(targetNode?.x) - Number(sourceNode?.x);
      const dy = Number(targetNode?.y) - Number(sourceNode?.y);
      const parameterSideInput = edge.semanticEdgeType === "parameter"
        && Number.isFinite(dx) && Number.isFinite(dy);
      const routing = edge.semanticEdgeType === "residual"
        ? residualRouting(sourceNode, targetNode)
        : parameterSideInput ? parameterRouting(sourceNode, targetNode, dx, dy) : {
        sourceAnchor: { side: "bottom", dy: EDGE_NODE_GAP },
        targetAnchor: { side: "top", dy: -EDGE_NODE_GAP },
        curve: "vertical",
        routingPolicy: "semantic_top_to_bottom",
      };
      return {
        ...edge,
        id: edge.id || `${edge.source}->${edge.target}`,
        source,
        target,
        sourceItemId: edge.source,
        targetItemId: edge.target,
        projectedThroughCollapse: source !== edge.source || target !== edge.target,
        ...routing,
        dataState: edge.dataState || "source_only",
      };
    }).filter(Boolean);
  }

  function projectGraph(graphSpec, reports, collapsedIds, selectedLayerIndex = null) {
    clearDagRankCache();
    _currentGraphSpec = graphSpec;
    const repeatPlan = visualRepeatPlan(graphSpec);
    const rootLayouts = (graphSpec.roots || [])
      .filter((root) => {
        const template = repeatPlan.templates.find((entry) => entry.id === root.id);
        return !template || selectedLayerIndex == null
          || template.instanceIndices.includes(Number(selectedLayerIndex));
      })
      .map((root) => layoutItem(root, collapsedIds, repeatPlan, selectedLayerIndex));
    const nodes = [];
    const clusters = [];
    const structuralShellIds = new Set();
    function markStructuralShells(item, parentIsStructural = false) {
      if (!item) return;
      const isSourceSection = item.id === "section/source_architecture";
      const isTopStructural = isSourceSection || (!parentIsStructural
        && (graphSpec.roots || []).includes(item)
        && ((graphSpec.roots || []).length === 1
          || (item.synthetic === true && (item.children || []).length === 1)));
      const isChainedStructural = parentIsStructural
        && item.synthetic === true
        && !item.backendNodeId;
      const isStructural = isTopStructural || isChainedStructural;
      if (isStructural) structuralShellIds.add(item.id);
      (item.children || []).forEach((child) => markStructuralShells(child, isStructural));
    }
    (graphSpec.roots || []).forEach((root) => markStructuralShells(root));

    function emit(layout, x, y, parentId = "") {
      const item = layout.item;
      const report = metricReportFor(item, reports);
      const repeatInfo = repeatInfoFor(item, repeatPlan, selectedLayerIndex);
      const repeatCount = repeatInfo.count;
      const common = visualPropsForReport(item, report);
      const hasPerformanceBadge = Boolean(report) && common.dataState !== "no_performance_data"
        && String(report?.metricShort || "").trim() !== "–";
      if (!(item.children || []).length || layout.collapsed) {
        nodes.push({
          id: item.id,
          label: item.label || item.id,
          typeLabel: typeLabelFor(item, report, layout.collapsed),
          kind: layout.collapsed ? "module" : (item.kind || "op"),
          x: x + layout.width / 2,
          y: y + layout.height / 2,
          width: layout.width,
          height: layout.height,
          colorKey: item.colorKey || "opv:op",
          parent: parentId || undefined,
          collapsed: layout.collapsed || undefined,
          repeatCount: repeatCount > 1 ? repeatCount : undefined,
          instanceIndices: repeatCount > 1 ? repeatInfo.indices : undefined,
          repeatRange: item.repeatRange || undefined,
          metricBadge: hasPerformanceBadge ? report.metricShort : undefined,
          timeSharePct: Number.isFinite(Number(report?.timeSharePct)) ? Number(report.timeSharePct) : undefined,
          ...common,
        });
        return;
      }

      const directNodes = layout.placements
        .filter(({ child }) => !(child.item.children || []).length || child.collapsed)
        .map(({ child }) => child.item.id);
      const directClusters = layout.placements
        .filter(({ child }) => (child.item.children || []).length && !child.collapsed)
        .map(({ child }) => child.item.id);
      clusters.push({
        id: item.id,
        label: item.label || item.id,
        x,
        y,
        width: layout.width,
        height: layout.height,
        colorKey: item.colorKey || "module:model",
        parent: parentId || undefined,
        nodes: directNodes,
        children: directClusters,
        repeat: repeatCount > 1,
        repeatCount: repeatCount > 1 ? repeatCount : undefined,
        instanceIndices: repeatCount > 1 ? repeatInfo.indices : undefined,
        repeatRange: item.repeatRange || undefined,
        metric: report?.metricShort || undefined,
        timeSharePct: Number.isFinite(Number(report?.timeSharePct)) ? Number(report.timeSharePct) : undefined,
        collapsible: item.synthetic !== true,
        structuralRoot: structuralShellIds.has(item.id),
        ...common,
      });

      layout.placements.forEach((placement) => {
        emit(placement.child, x + placement.x, y + placement.y, item.id);
      });
    }

    let x = GRAPH_PADDING;
    rootLayouts.forEach((layout) => {
      emit(layout, x, GRAPH_PADDING);
      x += layout.width + SECTION_GAP;
    });

    const graphWidth = rootLayouts.length ? x - SECTION_GAP + GRAPH_PADDING : 960;
    const graphHeight = Math.max(720, ...rootLayouts.map((layout) => layout.height)) + GRAPH_PADDING * 2;
    const visibleItemCount = nodes.length + clusters.filter((cluster) => !cluster.id.startsWith("section/")).length;
    const interactiveItemCount = [...nodes, ...clusters].filter((item) => item.selectable).length;

    // ── Comprehensive degraded-quality diagnostics ──
    const { hasCycles } = dagRanks(graphSpec);
    const hasLayoutRows = (graphSpec.roots || []).some((r) => {
      function check(item) {
        if (Array.isArray(item.layoutRows)) return true;
        return (item.children || []).some(check);
      }
      return check(r);
    });
    const edges = graphSpec.edges || [];
    const hasActivationEdges = edges.some((e) => e.semanticEdgeType === "activation");
    const edgeTypes = new Set(edges.map((e) => e.semanticEdgeType));
    const hasParameterOrStateEdges = edgeTypes.has("parameter") || edgeTypes.has("state");

    // Collect leaf items for colorKey diagnostics
    const allSrcItems = allItems(graphSpec).filter(({ item }) => !item.children || !item.children.length).map(({ item }) => item);
    const colorKeys = new Set(allSrcItems.map((it) => it.colorKey).filter(Boolean));
    const onlyOpvOp = colorKeys.size === 0 || (colorKeys.size === 1 && colorKeys.has("opv:op"));

    // Tensor metadata depth: check activation/residual edges for name/shape/dtype
    const tensorMetadataEdges = edges.filter((e) => e.semanticEdgeType === "activation" || e.semanticEdgeType === "residual");
    const tensorMissingFields = tensorMetadataEdges.length > 0
      && tensorMetadataEdges.every((e) => !e.tensor?.name || !e.tensor?.shape || !e.tensor?.dtype);

    // Aspect ratio: width / height
    const aspectRatio = graphHeight > 0 ? graphWidth / graphHeight : 1;
    const isNarrowAndTall = aspectRatio < 0.3 && graphHeight > 2000;

    // parallel_siblings declared but not projected into layoutRows
    const itemsWithParallelSiblings = allSrcItems.filter((it) => it.parallel_siblings?.length > 0);
    const hasParallelDeclButNoLayoutRows = itemsWithParallelSiblings.length > 0 && !hasLayoutRows;

    const degradedReasons = [];
    if (hasCycles) degradedReasons.push("dag_cycle_detected");
    if (!hasLayoutRows && !hasActivationEdges) degradedReasons.push("no_layout_direction");
    if (!hasLayoutRows && hasCycles) degradedReasons.push("cannot_auto_rank_with_cycles");
    if (onlyOpvOp && allSrcItems.length > 0) degradedReasons.push("all_nodes_use_default_colorKey");
    if (!hasParameterOrStateEdges && edgeTypes.size > 0) degradedReasons.push("no_parameter_or_state_edges");
    if (tensorMissingFields && tensorMetadataEdges.length > 0) degradedReasons.push("tensor_missing_name_shape_dtype");
    if (isNarrowAndTall) degradedReasons.push("narrow_and_tall_aspect");
    if (hasParallelDeclButNoLayoutRows) degradedReasons.push("parallel_siblings_not_in_layoutRows");
    const architectureLayoutDegraded = degradedReasons.length > 0;

    return {
      width: Math.max(graphWidth, 960),
      height: graphHeight,
      nodes,
      clusters,
      edges: projectEdges(
        graphSpec,
        nodes,
        clusters,
        collapsedIds,
        repeatPlan,
        selectedLayerIndex,
      ),
      metadata: {
        ...(graphSpec.metadata || {}),
        extractionScope: "hybrid",
        layoutDirection: "top_to_bottom",
        visibleItemCount,
        interactiveItemCount,
        collapsedIds: [...collapsedIds],
        ...(architectureLayoutDegraded ? {
          architecture_layout_degraded: true,
          degraded_reasons: degradedReasons,
        } : {}),
      },
    };
  }

  function defaultCollapsedIds(graphSpec) {
    return allItems(graphSpec)
      .filter(({ item, depth }) => (item.children || []).length && !item.synthetic
        && (item.defaultCollapsed === true || (item.defaultCollapsed == null && depth >= 4)))
      .map(({ item }) => item.id);
  }

  function createArchitectureGraph(graphSpec, reports = {}) {
    return projectGraph(graphSpec, reports, new Set());
  }

  function createArchitectureView(
    graphSpec,
    reports = {},
    collapsedIds = defaultCollapsedIds(graphSpec),
    selectedLayerIndex = null,
  ) {
    return projectGraph(graphSpec, reports, new Set(collapsedIds), selectedLayerIndex);
  }

  function backendToGraphId(graphSpec, backendNodeId) {
    return allItems(graphSpec).find(({ item }) => item.backendNodeId === backendNodeId)?.item.id || "";
  }

  function graphToBackendNodeId(graphSpec, graphNodeId) {
    return itemIndex(graphSpec).get(graphNodeId)?.item.backendNodeId || "";
  }

  function metadataForGraphItem(graphSpec, graphNodeId) {
    const entry = itemIndex(graphSpec).get(graphNodeId);
    if (!entry) return null;
    const incomingEdges = (graphSpec.edges || []).filter((edge) => edge.target === graphNodeId);
    const outgoingEdges = (graphSpec.edges || []).filter((edge) => edge.source === graphNodeId);
    return {
      item: entry.item,
      parentId: entry.parentId || "",
      incomingEdges,
      outgoingEdges,
    };
  }

  function ancestorIdsForGraphId(graphSpec, graphNodeId) {
    const index = itemIndex(graphSpec);
    const ancestors = [];
    let current = index.get(graphNodeId);
    while (current?.parentId) {
      ancestors.push(current.parentId);
      current = index.get(current.parentId);
    }
    return ancestors;
  }

  function layerNavigationForGraph(graphSpec) {
    const plan = visualRepeatPlan(graphSpec);
    return plan.hostId && plan.instanceIndices.length > 1
      ? {
        repeatNodeId: pagerHostIdForLayer(plan),
        instanceIndices: [...plan.instanceIndices],
      }
      : null;
  }

  function layerTemplateForIndex(graphSpec, layerIndex) {
    const matches = visualRepeatPlan(graphSpec).templates.filter((template) => (
      template.instanceIndices.includes(Number(layerIndex))
    ));
    if (matches.length !== 1) return null;
    return itemIndex(graphSpec).get(matches[0].id)?.item || null;
  }

  function correspondingLayerItemForIndex(graphSpec, graphNodeId, layerIndex) {
    const plan = visualRepeatPlan(graphSpec);
    const graphId = String(graphNodeId || "");
    const sourceTemplate = plan.templates
      .filter((template) => graphId === template.id || graphId.startsWith(`${template.id}/`))
      .sort((left, right) => right.id.length - left.id.length)[0];
    const targetTemplates = plan.templates.filter((template) => (
      template.instanceIndices.includes(Number(layerIndex))
    ));
    if (!sourceTemplate || !targetTemplates.length) return null;
    const index = itemIndex(graphSpec);
    const relativePath = graphId.slice(sourceTemplate.id.length);
    // A folded template represents many physical layers with one visual node. Keep that
    // exact node selected when the destination layer belongs to the same template.
    if (sourceTemplate.instanceIndices.includes(Number(layerIndex))) {
      return index.get(graphId)?.item || null;
    }
    // Across sibling templates (layer-0/rest, dense/MoE), preserve the exact path within
    // the layer block. With ambiguous Stage-2 membership, accept only one exact match.
    const matches = targetTemplates
      .map((template) => index.get(`${template.id}${relativePath}`)?.item)
      .filter(Boolean);
    return matches.length === 1 ? matches[0] : null;
  }

  global.DeepSeekArchitectureData = {
    applyExpertInventory,
    createArchitectureGraph,
    createArchitectureView,
    defaultCollapsedIds,
    backendToGraphId,
    graphToBackendNodeId,
    metadataForGraphItem,
    ancestorIdsForGraphId,
    layerNavigationForGraph,
    layerTemplateForIndex,
    correspondingLayerItemForIndex,
  };
})(typeof window === "undefined" ? globalThis : window);
