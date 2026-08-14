#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const htmlPath = path.join(here, 'assets', 'modelviz.html');
const schemaPath = path.join(here, 'outputs', 'model_architecture.json');
const graphPath = path.join(here, 'outputs', 'model_architecture_graph.json');
const mappingPath = path.join(here, 'outputs', 'operator_mapping.json');

const html = fs.readFileSync(htmlPath, 'utf8');
const schema = JSON.parse(fs.readFileSync(schemaPath, 'utf8'));
const graph = JSON.parse(fs.readFileSync(graphPath, 'utf8'));
const mapping = JSON.parse(fs.readFileSync(mappingPath, 'utf8'));
const scriptMatch = html.match(/<script>\s*([\s\S]*?)<\/script>\s*<\/body>/);
if (!scriptMatch) throw new Error('modelviz inline script was not found');
const initializationMarker = '    const graphStage = document.getElementById(\'graphStage\');';
const initializationIndex = scriptMatch[1].indexOf(initializationMarker);
if (initializationIndex < 0) throw new Error('modelviz initialization marker was not found');
const testableSource = scriptMatch[1].slice(0, initializationIndex)
  + '\nreturn { state, nodeById, defaultCollapsedIds, buildVisibleGraph, aggregateAccuracyOverlay, applyAccuracyOverlay, preservedTransform, popoverPosition, sourceLinesForNode, modelArchitectureColormapFor, mappingIdsForNode, incompatibleMappingsForNode, compatibilityForNode, preferredMappingForNode };';
const api = new Function('window', 'document', testableSource)({}, {
  getElementById() { return { clientWidth: 1440, clientHeight: 960 }; },
});

const pinnedPatternColormap = { coreColors: ['#123456'], ioColors: {} };
if (api.modelArchitectureColormapFor({ standardColormap: pinnedPatternColormap }, graph) !== pinnedPatternColormap) {
  throw new Error('pinned Pages pattern does not fall back to standardColormap');
}
const modernPatternColormap = { semanticColors: { 'sem:act': '#654321' } };
if (api.modelArchitectureColormapFor({ modelArchitectureColormap: () => modernPatternColormap }, graph) !== modernPatternColormap) {
  throw new Error('modern modelArchitectureColormap capability is not preferred when available');
}

function buildHierarchy(sourceGraph) {
  const clusters = new Map(sourceGraph.clusters.map((cluster) => [cluster.id, cluster]));
  const clusterParent = new Map(sourceGraph.clusters.filter((cluster) => cluster.parent).map((cluster) => [cluster.id, cluster.parent]));
  const nodeParent = new Map(sourceGraph.nodes.filter((node) => node.parent).map((node) => [node.id, node.parent]));
  const ancestorClustersForCluster = (clusterId) => {
    const result = [];
    let current = clusterParent.get(clusterId);
    while (current) {
      result.push(current);
      current = clusterParent.get(current);
    }
    return result;
  };
  const ancestorClustersForNode = (nodeId) => {
    const result = [];
    let current = nodeParent.get(nodeId);
    while (current) {
      result.push(current);
      current = clusterParent.get(current);
    }
    return result;
  };
  const descendantNodesOfCluster = (clusterId) => {
    const result = new Set();
    const visit = (id) => {
      const cluster = clusters.get(id);
      if (!cluster) return;
      (cluster.nodes || []).forEach((nodeId) => result.add(nodeId));
      (cluster.children || []).forEach(visit);
    };
    visit(clusterId);
    return result;
  };
  return { clusters, clusterParent, ancestorClustersForCluster, ancestorClustersForNode, descendantNodesOfCluster };
}

function nodeRect(node) {
  return { left: node.x - node.width / 2, top: node.y - node.height / 2, right: node.x + node.width / 2, bottom: node.y + node.height / 2 };
}

function clusterRect(cluster) {
  return { left: cluster.x, top: cluster.y, right: cluster.x + cluster.width, bottom: cluster.y + cluster.height };
}

function overlap(a, b) {
  return { width: Math.min(a.right, b.right) - Math.max(a.left, b.left), height: Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) };
}

function assertProjection(name, projected) {
  const errors = [];
  const nodesById = new Map(projected.nodes.map((node) => [node.id, node]));
  const clustersById = new Map(projected.clusters.map((cluster) => [cluster.id, cluster]));
  projected.nodes.forEach((node, index) => {
    for (const other of projected.nodes.slice(index + 1)) {
      const hit = overlap(nodeRect(node), nodeRect(other));
      if (hit.width > 0 && hit.height > 0) errors.push(`nodes ${node.id}/${other.id} overlap ${hit.width}x${hit.height}`);
    }
    if (node.parent && clustersById.has(node.parent)) {
      const inner = nodeRect(node);
      const outer = clusterRect(clustersById.get(node.parent));
      if (inner.left < outer.left || inner.top < outer.top || inner.right > outer.right || inner.bottom > outer.bottom) {
        errors.push(`node ${node.id} escapes ${node.parent}`);
      }
    }
  });
  projected.clusters.forEach((cluster, index) => {
    for (const other of projected.clusters.slice(index + 1)) {
      if ((cluster.parent || null) !== (other.parent || null)) continue;
      const hit = overlap(clusterRect(cluster), clusterRect(other));
      if (hit.width > 0 && hit.height > 0) errors.push(`sibling clusters ${cluster.id}/${other.id} overlap ${hit.width}x${hit.height}`);
    }
  });
  projected.edges.forEach((edge) => {
    if (!nodesById.has(edge.source) || !nodesById.has(edge.target)) errors.push(`edge ${edge.id} has a hidden endpoint`);
  });
  if (errors.length) throw new Error(`${name}:\n${errors.join('\n')}`);
}

api.state.schema = schema;
api.state.graph = graph;
api.state.mappingDocument = mapping;
api.state.baseHierarchy = buildHierarchy(graph);

const defaultIds = api.defaultCollapsedIds();
api.state.collapsedClusters = new Set(defaultIds);
const defaultGraph = api.buildVisibleGraph();
api.state.visibleGraph = defaultGraph;
assertProjection('default-folded', defaultGraph);
if (defaultIds.length !== 4 || defaultGraph.nodes.length !== 15) {
  throw new Error(`default-folded count drift: ${defaultIds.length} folds, ${defaultGraph.nodes.length} nodes`);
}
const defaultCompatibility = defaultGraph.nodes
  .map((node) => [node.id, api.compatibilityForNode(node)])
  .filter(([, compatibility]) => compatibility);
if (defaultCompatibility.length !== 3) {
  throw new Error(`default incompatibility overlay drift: expected 3 visible nodes, got ${defaultCompatibility.length}`);
}
for (const nodeId of ['kernel_dispatch', 'score_compute_group', 'value_accumulation_group']) {
  if (!defaultCompatibility.some(([id]) => id === nodeId)) {
    throw new Error(`source incompatibility badge is missing from ${nodeId}`);
  }
}
if (api.compatibilityForNode(defaultGraph.nodes.find((node) => node.id === 'online_softmax_group'))) {
  throw new Error('compatible Online Softmax is incorrectly marked source-incompatible');
}
if (api.preferredMappingForNode(api.nodeById('kernel_dispatch'))?.id !== 'map_swizzle') {
  throw new Error('Kernel Dispatch selection does not prioritize the incompatible T.use_swizzle mapping');
}
if (api.preferredMappingForNode(api.nodeById('qk_gemm'))?.id !== 'map_warp_policy') {
  throw new Error('score GEMM selection does not prioritize the incompatible GemmWarpPolicy mapping');
}
const foldedModule = api.nodeById('online_softmax_group');
if (!foldedModule || !foldedModule.collapsed || foldedModule.kind !== 'module') {
  throw new Error('collapsed module representatives are not selectable detail targets');
}
const foldedModuleSourceLines = api.sourceLinesForNode(foldedModule);
for (const expectedLine of [161, 165, 166, 169, 170]) {
  if (!foldedModuleSourceLines.includes(expectedLine)) {
    throw new Error(`collapsed Online Softmax source linkage is missing line ${expectedLine}`);
  }
}
api.state.accuracyOverlay = new Map([
  ['q_stage', { nodeId: 'q_stage', status: 'pass', statusLabel: '通过', error: '0', badge: '通过 · 0' }],
  ['kv_stage', { nodeId: 'kv_stage', status: 'pass', statusLabel: '通过', error: '0', badge: '通过 · 0' }],
  ['qk_gemm', { nodeId: 'qk_gemm', status: 'pass', statusLabel: '通过', error: '2.4e-4', badge: '通过 · 2.4e-4' }],
  ['pe_gemm', { nodeId: 'pe_gemm', status: 'pass', statusLabel: '通过', error: '2.4e-4', badge: '通过 · 2.4e-4' }],
  ['score_block_max', { nodeId: 'score_block_max', status: 'pass', statusLabel: '通过', error: '0', badge: '通过 · 0' }],
  ['running_max_merge', { nodeId: 'running_max_merge', status: 'pass', statusLabel: '通过', error: '0', badge: '通过 · 0' }],
  ['score_exponential', { nodeId: 'score_exponential', status: 'fail', statusLabel: '异常', error: '3.1e-2', badge: '异常 · 3.1e-2' }],
  ['pv_gemm', { nodeId: 'pv_gemm', status: 'pass', statusLabel: '通过', error: '—', badge: '通过 · cos 1.0000' }],
]);
const accuracyDefault = api.buildVisibleGraph();
const foldedSoftmaxAccuracy = accuracyDefault.nodes.find((node) => node.id === 'online_softmax_group')?.accuracyOverlay;
if (foldedSoftmaxAccuracy?.status !== 'fail' || foldedSoftmaxAccuracy.badge !== '1 异常') {
  throw new Error('folded Online Softmax does not aggregate the child accuracy anomaly');
}
const foldedScoreAccuracy = accuracyDefault.nodes.find((node) => node.id === 'score_compute_group')?.accuracyOverlay;
if (foldedScoreAccuracy?.status !== 'pass' || foldedScoreAccuracy.testedCount !== 3) {
  throw new Error('folded score compute does not aggregate its three tested operators');
}
api.state.accuracyOverlay.set('score_exponential', {
  nodeId: 'score_exponential', status: 'fixed', statusLabel: '已修复', error: '8.0e-4', badge: '已修复 · 8.0e-4',
});
const accuracyFixedDefault = api.buildVisibleGraph();
if (accuracyFixedDefault.nodes.find((node) => node.id === 'online_softmax_group')?.accuracyOverlay?.status !== 'fixed') {
  throw new Error('folded Online Softmax does not switch to fixed after the child retest passes');
}
api.state.accuracyOverlay = new Map();
api.applyAccuracyOverlay(accuracyFixedDefault);
if (accuracyFixedDefault.nodes.some((node) => node.accuracyOverlay || node.overlayKind)) {
  throw new Error('in-place accuracy refresh leaves stale node decoration metadata');
}
const mainline = [
  'kernel_dispatch',
  'score_compute_group',
  'online_softmax_group',
  'value_accumulation_group',
  'output_normalize',
  'no_split_store',
  'mla_output',
].map((id) => defaultGraph.nodes.find((node) => node.id === id));
if (mainline.some((node) => !node)) throw new Error('default mainline is incomplete');
if (!mainline.every((node, index) => index === 0 || node.y > mainline[index - 1].y)) {
  throw new Error('default mainline is not strictly top-to-bottom');
}
if (Math.max(...mainline.map((node) => node.x)) - Math.min(...mainline.map((node) => node.x)) > 1) {
  throw new Error('default mainline is not center-aligned');
}
const queryStage = defaultGraph.nodes.find((node) => node.id === 'q_stage');
const kvStage = defaultGraph.nodes.find((node) => node.id === 'kv_stage');
if (!queryStage || !kvStage || queryStage.y !== kvStage.y || Math.abs((queryStage.x + kvStage.x) / 2 - mainline[0].x) > 1) {
  throw new Error('Q/KV staging is not a symmetric natural branch around the mainline');
}
const runtimeConfig = defaultGraph.nodes.find((node) => node.id === 'runtime_config');
const kernelDispatch = defaultGraph.nodes.find((node) => node.id === 'kernel_dispatch');
if (!runtimeConfig || !kernelDispatch || runtimeConfig.x !== kernelDispatch.x || runtimeConfig.y >= kernelDispatch.y) {
  throw new Error('Runtime Config → Kernel Dispatch is not a centered top-to-bottom control spine');
}
const cfgDispatch = defaultGraph.edges.find((edge) => edge.id === 'e_cfg_dispatch');
if (!cfgDispatch || cfgDispatch.sourceAnchor !== 'bottom' || cfgDispatch.targetAnchor !== 'top') {
  throw new Error('Runtime Config → Kernel Dispatch does not use centered vertical ports');
}
const dispatchFanout = defaultGraph.edges.filter((edge) => edge.source === 'kernel_dispatch');
if (dispatchFanout.length !== 2) throw new Error(`Kernel Dispatch fan-out drift: ${dispatchFanout.length} edges`);
if (!dispatchFanout.every((edge) => edge.sourceAnchor === 'bottom' && edge.targetAnchor === 'top')) {
  throw new Error('Kernel Dispatch fan-out does not use the shared bottom-center source port');
}
if (!dispatchFanout.every((edge) => Array.isArray(edge.waypoints) && edge.waypoints.length === 2)) {
  throw new Error('Kernel Dispatch fan-out is missing its shared trunk route');
}
const dispatchQ = dispatchFanout.find((edge) => edge.id === 'e_dispatch_q');
const dispatchKv = dispatchFanout.find((edge) => edge.id === 'e_dispatch_kv');
if (!dispatchQ || !dispatchKv) throw new Error('Kernel Dispatch fan-out edge ids drifted');
if (dispatchQ.waypoints[0].x !== dispatchKv.waypoints[0].x
  || dispatchQ.waypoints[0].y !== dispatchKv.waypoints[0].y
  || dispatchQ.waypoints[0].x !== mainline[0].x) {
  throw new Error('Kernel Dispatch fan-out trunk does not merge before branching');
}
const inputStageCluster = defaultGraph.clusters.find((cluster) => cluster.id === 'input_stage_group');
if (!inputStageCluster || dispatchQ.waypoints[0].y >= inputStageCluster.y) {
  throw new Error('Kernel Dispatch fan-out branch runs through the Input Staging title lane');
}
const dispatchLeft = kernelDispatch.x - kernelDispatch.width / 2;
const dispatchRight = kernelDispatch.x + kernelDispatch.width / 2;
const dispatchBottom = kernelDispatch.y + kernelDispatch.height / 2;
const defaultNodesById = new Map(defaultGraph.nodes.map((node) => [node.id, node]));
const faninGroups = [
  { stage: queryStage, dispatchEdge: dispatchQ, dataEdgeIds: ['e_q_stage', 'e_qpe_stage'], mergeDx: -60, lane: 'left' },
  { stage: kvStage, dispatchEdge: dispatchKv, dataEdgeIds: ['e_kv_stage', 'e_kpe_stage'], mergeDx: 60, lane: 'right' },
];
faninGroups.forEach((group) => {
  const dataEdges = group.dataEdgeIds.map((id) => defaultGraph.edges.find((edge) => edge.id === id));
  if (dataEdges.some((edge) => !edge)) throw new Error(`${group.stage.id} fan-in data edge ids drifted`);
  const sharedSuffix = JSON.stringify(dataEdges[0].waypoints?.slice(1));
  if (!sharedSuffix || dataEdges.some((edge) => !Array.isArray(edge.waypoints)
    || edge.waypoints.length !== 4 || JSON.stringify(edge.waypoints.slice(1)) !== sharedSuffix)) {
    throw new Error(`${group.stage.id} tensor inputs do not merge into a shared route`);
  }
  const stageJunction = group.dispatchEdge.waypoints[group.dispatchEdge.waypoints.length - 1];
  const targetTop = group.stage.y - group.stage.height / 2;
  const expectedMergeX = group.stage.x + group.mergeDx;
  dataEdges.forEach((edge) => {
    const source = defaultNodesById.get(edge.source);
    const dataMerge = edge.waypoints[1];
    const finalJunction = edge.waypoints[edge.waypoints.length - 1];
    if (!source || edge.sourceAnchor !== 'bottom' || edge.targetAnchor !== 'top'
      || edge.waypoints[0].x !== source.x || dataMerge.x !== expectedMergeX) {
      throw new Error(`${edge.id} does not enter its tensor fan-in merge point`);
    }
    if (finalJunction.x !== stageJunction.x || finalJunction.y !== stageJunction.y
      || finalJunction.x !== group.stage.x || finalJunction.y >= inputStageCluster.y
      || targetTop <= inputStageCluster.y) {
      throw new Error(`${edge.id} does not share the Dispatch-to-stage final trunk`);
    }
    const sourceOutsideDispatch = group.lane === 'left'
      ? edge.waypoints[0].x < dispatchLeft && dataMerge.x < group.stage.x
      : edge.waypoints[0].x > dispatchRight && dataMerge.x > group.stage.x;
    if (!sourceOutsideDispatch || dataMerge.y <= dispatchBottom || dataMerge.y >= stageJunction.y) {
      throw new Error(`${edge.id} leaves its outer input corridor before fan-in`);
    }
  });
});

api.state.collapsedClusters = new Set();
const expandedGraph = api.buildVisibleGraph();
assertProjection('expanded', expandedGraph);
if (expandedGraph.nodes.length !== 29 || expandedGraph.clusters.length !== 8) {
  throw new Error(`expanded count drift: ${expandedGraph.nodes.length} nodes, ${expandedGraph.clusters.length} clusters`);
}
const expandedCompatibilityIds = expandedGraph.nodes
  .filter((node) => api.compatibilityForNode(node))
  .map((node) => node.id)
  .sort();
if (JSON.stringify(expandedCompatibilityIds) !== JSON.stringify(['kernel_dispatch', 'pe_gemm', 'pv_gemm', 'qk_gemm'])) {
  throw new Error(`expanded incompatibility overlay drift: ${expandedCompatibilityIds.join(', ')}`);
}
const beforeTransform = { zoom: 0.72, tx: 48, ty: 36 };
const afterTransform = api.preservedTransform(defaultGraph, expandedGraph, beforeTransform, 'online_softmax_group');
const beforeAnchor = defaultGraph.nodes.find((node) => node.id === 'online_softmax_group');
const afterAnchorCluster = expandedGraph.clusters.find((cluster) => cluster.id === 'online_softmax_group');
const beforeScreen = { x: beforeTransform.tx + beforeAnchor.x * beforeTransform.zoom, y: beforeTransform.ty + beforeAnchor.y * beforeTransform.zoom };
const afterScreen = {
  x: afterTransform.tx + (afterAnchorCluster.x + afterAnchorCluster.width / 2) * afterTransform.zoom,
  y: afterTransform.ty + (afterAnchorCluster.y + afterAnchorCluster.height / 2) * afterTransform.zoom,
};
if (Math.hypot(beforeScreen.x - afterScreen.x, beforeScreen.y - afterScreen.y) > 0.01) {
  throw new Error('fold/expand transform did not preserve the clicked module screen anchor');
}

api.state.collapsedClusters = new Set(defaultIds);
api.state.baseHierarchy.ancestorClustersForNode('score_sum').forEach((clusterId) => api.state.collapsedClusters.delete(clusterId));
const mappingFocused = api.buildVisibleGraph();
assertProjection('mapping-focused', mappingFocused);
if (!mappingFocused.nodes.some((node) => node.id === 'score_sum')) {
  throw new Error('mapping focus did not reveal score_sum after expanding ancestors');
}

const rightPopover = api.popoverPosition(
  { left: 0, top: 0, width: 1000, height: 800 },
  { left: 400, right: 500, top: 300, bottom: 360, height: 60 },
  { width: 340, height: 240 },
  12,
  12,
);
if (rightPopover.horizontal !== 'right' || rightPopover.vertical !== 'bottom'
  || rightPopover.left !== 512 || rightPopover.top !== 372) {
  throw new Error('operator popover is not anchored to the target node lower-right corner');
}
const leftPopover = api.popoverPosition(
  { left: 0, top: 0, width: 1000, height: 800 },
  { left: 900, right: 980, top: 700, bottom: 760, height: 60 },
  { width: 340, height: 240 },
  12,
  12,
);
if (leftPopover.horizontal !== 'left' || leftPopover.vertical !== 'top'
  || leftPopover.left !== 548 || leftPopover.top !== 448) {
  throw new Error('operator popover does not flip horizontally and vertically at viewport edges');
}

api.state.collapsedClusters = new Set([...defaultIds, 'tile_loop_group']);
assertProjection('tile-loop-folded', api.buildVisibleGraph());
api.state.collapsedClusters = new Set(['mla_decode_group', ...defaultIds]);
const rootFolded = api.buildVisibleGraph();
assertProjection('root-folded', rootFolded);
if (rootFolded.nodes.length !== 1 || rootFolded.nodes[0].id !== 'mla_decode_group') {
  throw new Error('root collapse did not produce exactly one representative node');
}
if (api.compatibilityForNode(rootFolded.nodes[0])?.mappings?.length !== 2) {
  throw new Error('root-folded MLA module does not aggregate both incompatible source mappings');
}

for (const [id, expected] of [['defaultSchemaJson', schema], ['defaultGraphJson', graph], ['defaultMappingJson', mapping]]) {
  const match = html.match(new RegExp(`<script type="application/json" id="${id}">([\\s\\S]*?)<\\/script>`));
  if (!match) throw new Error(`${id} is missing`);
  const embedded = JSON.parse(match[1]);
  if (JSON.stringify(embedded) !== JSON.stringify(expected)) throw new Error(`${id} is stale`);
}

console.log(JSON.stringify({
  ok: true,
  default: { nodes: defaultGraph.nodes.length, clusters: defaultGraph.clusters.length, folds: defaultIds },
  expanded: { nodes: expandedGraph.nodes.length, clusters: expandedGraph.clusters.length },
  rootFolded: { nodes: rootFolded.nodes.length, clusters: rootFolded.clusters.length },
  anchorDrift: Math.hypot(beforeScreen.x - afterScreen.x, beforeScreen.y - afterScreen.y),
  dispatchFanout: 'shared bottom-center trunk',
  inputRouting: 'Q/KV pair fan-in + shared Dispatch-to-stage trunks',
  mappingFocus: 'score_sum visible',
  sourceCompatibility: '3 folded / 4 expanded nodes with incompatible mapping priority',
  accuracyOverlay: 'pass/fail/fixed node badges + folded parent aggregation',
  operatorPopover: 'lower-right anchored with viewport-edge horizontal/vertical fallback',
  mappings: mapping.mappings.length,
}, null, 2));
