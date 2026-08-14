import * as THREE from '../../hpc-topology-viewer-main/node_modules/three/build/three.module.min.js';

(function initTrainingObjectTopology() {
  'use strict';

  const graph = window.PANGU_GRAPH;
  const registry = window.TRAINING_OBJECT_REGISTRY;
  const weightData = window.WEIGHT_DATA || {};
  const commData = window.COMM_DATA || {};
  const tsData = window.TS_DATA || {};

  if (!graph || !registry) {
    throw new Error('Pangu graph or training object registry is not loaded');
  }

  const THEME_KEY = 'pangu-training-object-topology-theme';
  const GRAPH_SCALE = 0.0105;
  const OBJECT_SIZE = 0.23;
  const layerCount = registry.modelFacts.layers || 46;
  const params = new URLSearchParams(window.location.search);
  const requestedView = params.get('view');
  const requestedTheme = params.get('theme');
  const initialView = ['iso', 'front', 'side'].includes(requestedView) ? requestedView : 'iso';

  if (requestedTheme === 'light' || requestedTheme === 'dark') {
    document.documentElement.dataset.theme = requestedTheme;
  }

  const state = {
    selectedObjectId: 'router_weight',
    hoveredObjectId: null,
    activeTypes: new Set(Object.keys(registry.objectTypes)),
    strategy: 'all',
    view: initialView,
    flowPhase: 'all',
    query: '',
    zoom: 1,
    dragStart: null,
    sceneOffset: new THREE.Vector3(0, 0, 0),
    rootRotation: { x: 0, y: 0 },
  };

  const els = {
    frame: document.querySelector('[data-ide-frame]'),
    root: document.documentElement,
    body: document.body,
    host: document.getElementById('trainingCanvasHost'),
    objectTree: document.getElementById('objectTree'),
    objectCount: document.getElementById('objectCount'),
    typeFilters: document.getElementById('typeFilters'),
    strategySwitch: document.getElementById('strategySwitch'),
    objectSearch: document.getElementById('objectSearch'),
    inspectorBody: document.getElementById('inspectorBody'),
    inspectorMeta: document.getElementById('inspectorMeta'),
    clearSelection: document.getElementById('clearSelection'),
    relationGrid: document.getElementById('relationGrid'),
    sourceStrip: document.getElementById('sourceStrip'),
    bottomMeta: document.getElementById('bottomMeta'),
    statusStrip: document.getElementById('statusStrip'),
    stageMeta: document.getElementById('stageMeta'),
    trainingHud: document.getElementById('trainingHud'),
    axisReadout: document.getElementById('axisReadout'),
    tooltip: document.getElementById('trainingTooltip'),
    tooltipTitle: document.getElementById('tooltipTitle'),
    tooltipBody: document.getElementById('tooltipBody'),
    themeToggle: document.getElementById('themeToggle'),
    terminalObject: document.getElementById('terminalObject'),
  };

  const objectById = new Map(registry.objects.map((object) => [object.id, object]));
  if (objectById.has(params.get('object'))) {
    state.selectedObjectId = params.get('object');
  }
  const objectsByAnchor = registry.objects.reduce((map, object) => {
    const list = map.get(object.anchor) || [];
    list.push(object);
    map.set(object.anchor, list);
    return map;
  }, new Map());
  const nodeById = new Map(graph.nodes.map((node) => [node.id, node]));
  const clusterById = new Map((graph.clusters || []).map((cluster) => [cluster.id, cluster]));

  const sceneState = {
    scene: null,
    camera: null,
    renderer: null,
    root: null,
    raycaster: new THREE.Raycaster(),
    pointer: new THREE.Vector2(),
    hoverables: [],
    objectMeshes: new Map(),
    objectPositions: new Map(),
    graphPositions: new Map(),
    labelSprites: [],
    frameId: null,
    needsRender: true,
  };

  const fallbackState = {
    active: false,
    root: null,
    nodesLayer: null,
    objectsLayer: null,
    flowLayer: null,
    layerRail: null,
    positions: new Map(),
  };

  function escapeHtml(value) {
    return String(value ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function cssVar(name, fallback) {
    const value = getComputedStyle(els.root).getPropertyValue(name).trim();
    return value || fallback;
  }

  function tagKind(tag) {
    const value = String(tag || '').toLowerCase();
    if (value.startsWith('tp')) return 'tp';
    if (value.startsWith('pp')) return 'pp';
    if (value.startsWith('ep')) return 'ep';
    if (value.startsWith('dp')) return 'dp';
    if (value.startsWith('cp')) return 'cp';
    if (value.startsWith('sp')) return 'sp';
    return 'other';
  }

  function selectedObject() {
    return objectById.get(state.selectedObjectId) || null;
  }

  function selectedStrategy() {
    return registry.strategies.find((strategy) => strategy.id === state.strategy) || null;
  }

  function objectMatchesStrategy(object) {
    if (state.strategy === 'all') return true;
    const strategy = selectedStrategy();
    if (!strategy) return true;
    if (strategy.anchors.includes(object.anchor)) return true;
    return object.tags.some((tag) => tagKind(tag) === strategy.id || String(tag).toLowerCase().includes(strategy.id));
  }

  function objectMatchesQuery(object) {
    const query = state.query.trim().toLowerCase();
    if (!query) return true;
    return [
      object.id,
      object.type,
      object.label,
      object.path,
      object.shape,
      object.note,
      ...(object.tags || []),
    ].some((value) => String(value || '').toLowerCase().includes(query));
  }

  function visibleObjects() {
    return registry.objects.filter((object) => (
      state.activeTypes.has(object.type)
      && objectMatchesStrategy(object)
      && objectMatchesQuery(object)
    ));
  }

  function colorKeys() {
    const keys = [];
    (graph.clusters || []).forEach((cluster) => keys.push(cluster.colorKey || `parent:${cluster.id}`));
    (graph.nodes || []).forEach((node) => keys.push(node.colorKey || `type:${node.kind || 'node'}`));
    registry.objects.forEach((object) => {
      const typeInfo = registry.objectTypes[object.type] || {};
      keys.push(typeInfo.colorKey || `object:${object.type}`);
    });
    return keys;
  }

  function colorMap() {
    const pattern = window.PtoModelGraphvizPattern;
    if (!pattern?.buildColorMap || !pattern?.modelArchitectureColormap) return new Map();
    return pattern.buildColorMap(colorKeys(), pattern.modelArchitectureColormap(graph, {
      theme: els.root.dataset.theme || 'dark',
    }));
  }

  function colorForObject(object, map = colorMap()) {
    const typeInfo = registry.objectTypes[object.type] || {};
    const key = typeInfo.colorKey || object.colorKey || `object:${object.type}`;
    return map.get(key) || fallbackObjectColor(object.type);
  }

  function fallbackObjectColor(type) {
    return {
      weight: '#9ca3af',
      bias: '#9ca3af',
      opt: '#8b5cf6',
      activation: '#14b8a6',
      residual: '#f59e0b',
      gradient: '#f43f5e',
      loss: '#38bdf8',
      lr: '#facc15',
      scale: '#a855f7',
      comm: '#22c55e',
      metric: '#fb7185',
    }[type] || '#9ca3af';
  }

  function colorForEdgeType(type) {
    return {
      activation: '#38bdf8',
      residual: '#f59e0b',
      gradient: '#fb7185',
      communication: '#22c55e',
      optimizer: '#a855f7',
      parameter: '#9ca3af',
    }[type] || cssVar('--foreground-secondary', '#a1a1aa');
  }

  function graphToWorld(x, y, z = 0) {
    return new THREE.Vector3(
      (x - graph.width / 2) * GRAPH_SCALE,
      -(y - graph.height / 2) * GRAPH_SCALE,
      z,
    );
  }

  function graphNodeCenter(node) {
    return graphToWorld(node.x, node.y, 0);
  }

  function disposeObject(object) {
    object.traverse((child) => {
      if (child.geometry) child.geometry.dispose();
      if (child.material) {
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        materials.forEach((material) => {
          if (material.map) material.map.dispose();
          material.dispose();
        });
      }
    });
  }

  function clearSceneRoot() {
    if (!sceneState.root) return;
    while (sceneState.root.children.length) {
      const object = sceneState.root.children.pop();
      disposeObject(object);
    }
    sceneState.hoverables = [];
    sceneState.objectMeshes.clear();
    sceneState.objectPositions.clear();
    sceneState.graphPositions.clear();
    sceneState.labelSprites = [];
  }

  function clearFallbackRoot() {
    if (!fallbackState.root) return;
    fallbackState.root.remove();
    fallbackState.positions.clear();
    fallbackState.nodesLayer = null;
    fallbackState.objectsLayer = null;
    fallbackState.flowLayer = null;
    fallbackState.layerRail = null;
  }

  function makeMaterial(color, options = {}) {
    return new THREE.MeshStandardMaterial({
      color: new THREE.Color(color),
      roughness: options.roughness ?? 0.68,
      metalness: options.metalness ?? 0.04,
      transparent: options.opacity !== undefined && options.opacity < 1,
      opacity: options.opacity ?? 1,
      emissive: new THREE.Color(options.emissive || '#000000'),
      emissiveIntensity: options.emissiveIntensity || 0,
    });
  }

  function makeLine(points, color, options = {}) {
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
      color: new THREE.Color(color),
      transparent: true,
      opacity: options.opacity ?? 0.58,
    });
    const line = new THREE.Line(geometry, material);
    line.renderOrder = options.renderOrder || 0;
    return line;
  }

  function makeLabelTexture(text, options = {}) {
    const canvas = document.createElement('canvas');
    const width = options.width || 420;
    const height = options.height || 120;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, width, height);
    const fill = options.fill || (els.root.dataset.theme === 'light' ? 'rgba(255,255,255,0.92)' : 'rgba(10,10,10,0.78)');
    if (options.box !== false) {
      roundRect(ctx, 8, 8, width - 16, height - 16, options.radius || 18);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.strokeStyle = els.root.dataset.theme === 'light' ? 'rgba(0,0,0,0.12)' : 'rgba(255,255,255,0.16)';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    ctx.fillStyle = options.color || (els.root.dataset.theme === 'light' ? '#111827' : '#f8fafc');
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = options.font || '600 34px Inter, PingFang SC, sans-serif';
    const label = String(text || '');
    const trimmed = label.length > (options.maxChars || 26)
      ? `${label.slice(0, (options.maxChars || 26) - 1)}…`
      : label;
    ctx.fillText(trimmed, width / 2, height / 2);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.needsUpdate = true;
    return texture;
  }

  function roundRect(ctx, x, y, width, height, radius) {
    const r = Math.min(radius, width / 2, height / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + width, y, x + width, y + height, r);
    ctx.arcTo(x + width, y + height, x, y + height, r);
    ctx.arcTo(x, y + height, x, y, r);
    ctx.arcTo(x, y, x + width, y, r);
    ctx.closePath();
  }

  function addLabel(text, position, scale, options = {}) {
    const texture = makeLabelTexture(text, options);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false,
      depthWrite: false,
    }));
    sprite.position.copy(position);
    sprite.scale.set(scale.x, scale.y, 1);
    sprite.renderOrder = options.renderOrder || 20;
    sceneState.root.add(sprite);
    sceneState.labelSprites.push(sprite);
    return sprite;
  }

  function buildLayerStack(map) {
    const decoder = clusterById.get('decoder') || { x: 430, y: 239, width: 340, height: 688 };
    const top = decoder.y + 40;
    const bottom = decoder.y + decoder.height - 40;
    const centerX = decoder.x + decoder.width / 2;
    const slabWidth = decoder.width * GRAPH_SCALE * 0.92;
    const layerHeight = Math.max(0.035, ((bottom - top) / layerCount) * GRAPH_SCALE * 0.7);
    const startDepth = -2.35;
    const depthStep = 4.7 / Math.max(1, layerCount - 1);

    for (let index = 0; index < layerCount; index += 1) {
      const yGraph = top + (bottom - top) * (index / Math.max(1, layerCount - 1));
      const position = graphToWorld(centerX, yGraph, startDepth + index * depthStep);
      const dsa = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45].includes(index);
      const dense = index < 2;
      const color = dense ? '#8b949e' : dsa ? (map.get('sem:attention') || '#38bdf8') : (map.get('sem:moe') || '#a855f7');
      const geometry = new THREE.BoxGeometry(slabWidth, layerHeight, 0.07);
      const material = makeMaterial(color, { opacity: 0.72, roughness: 0.82 });
      const slab = new THREE.Mesh(geometry, material);
      slab.position.copy(position);
      slab.userData = {
        kind: 'layer',
        title: `Layer ${String(index).padStart(2, '0')}`,
        body: `${dense ? 'Dense FFN' : 'MoE FFN'} · ${dsa ? 'DSA attention' : 'SWA attention'} · PP${Math.floor(index / Math.ceil(layerCount / 4))}/4`,
      };
      sceneState.root.add(slab);
      sceneState.hoverables.push(slab);

      if (index % 5 === 0 || index === layerCount - 1) {
        addLabel(`L${index}`, position.clone().add(new THREE.Vector3(slabWidth / 2 + 0.34, 0, 0.02)), { x: 0.34, y: 0.11 }, {
          width: 220,
          height: 80,
          box: false,
          font: '700 32px Inter, sans-serif',
          color: cssVar('--foreground-secondary', '#a1a1aa'),
          maxChars: 4,
        });
      }
    }
  }

  function buildGraphNodes(map) {
    graph.nodes.forEach((node) => {
      const position = graphNodeCenter(node);
      sceneState.graphPositions.set(node.id, position.clone());
      const width = Math.max(0.58, (node.width || 180) * GRAPH_SCALE);
      const height = Math.max(0.22, (node.height || 50) * GRAPH_SCALE);
      const depth = node.kind === 'tensor' ? 0.10 : 0.18;
      const key = node.colorKey || `type:${node.kind || 'node'}`;
      const color = node.kind === 'tensor'
        ? (map.get(key) || map.get('io:parameter') || '#d1d5db')
        : (map.get(key) || '#64748b');
      const geometry = new THREE.BoxGeometry(width, height, depth);
      const material = makeMaterial(color, {
        opacity: node.kind === 'tensor' ? 0.82 : 0.92,
        emissive: node.id === 'gate' ? '#26000e' : '#000000',
        emissiveIntensity: node.id === 'gate' ? 0.12 : 0,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.copy(position);
      mesh.userData = {
        kind: 'graph-node',
        nodeId: node.id,
        title: node.label,
        body: node.typeLabel || node.kind || 'node',
      };
      sceneState.root.add(mesh);
      sceneState.hoverables.push(mesh);

      addLabel(node.label, position.clone().add(new THREE.Vector3(0, 0, depth / 2 + 0.035)), {
        x: Math.min(width * 0.86, 2.2),
        y: Math.min(height * 0.58, 0.28),
      }, {
        width: 520,
        height: 140,
        box: false,
        font: '700 30px Inter, PingFang SC, sans-serif',
        color: els.root.dataset.theme === 'light' ? '#111827' : '#f8fafc',
        maxChars: 22,
      });
    });
  }

  function buildGraphEdges() {
    graph.edges.forEach((edge) => {
      const source = sceneState.graphPositions.get(edge.source);
      const target = sceneState.graphPositions.get(edge.target);
      if (!source || !target) return;
      const midDepth = edge.edgeType === 'parameter' ? 0.28 : 0.12;
      const mid = source.clone().lerp(target, 0.5).add(new THREE.Vector3(0, 0, midDepth));
      const line = makeLine([
        source.clone().add(new THREE.Vector3(0, 0, 0.18)),
        mid,
        target.clone().add(new THREE.Vector3(0, 0, 0.18)),
      ], colorForEdgeType(edge.edgeType), {
        opacity: edge.edgeType === 'parameter' ? 0.28 : 0.45,
      });
      sceneState.root.add(line);
    });
  }

  function markerOffset(index, count) {
    const angle = (index / Math.max(1, count)) * Math.PI * 2 - Math.PI / 2;
    const radiusX = count > 3 ? 0.58 : 0.42;
    const radiusY = count > 3 ? 0.38 : 0.28;
    return new THREE.Vector3(Math.cos(angle) * radiusX, Math.sin(angle) * radiusY, 0.82 + index * 0.035);
  }

  function geometryForObject(object) {
    if (object.type === 'activation' || object.type === 'loss' || object.type === 'metric') {
      return new THREE.SphereGeometry(OBJECT_SIZE * 0.58, 24, 16);
    }
    if (object.type === 'comm') {
      return new THREE.OctahedronGeometry(OBJECT_SIZE * 0.72, 0);
    }
    if (object.type === 'residual' || object.type === 'gradient') {
      return new THREE.CylinderGeometry(OBJECT_SIZE * 0.42, OBJECT_SIZE * 0.42, OBJECT_SIZE * 0.22, 24);
    }
    return new THREE.BoxGeometry(OBJECT_SIZE, OBJECT_SIZE, OBJECT_SIZE);
  }

  function buildObjectMarkers(map) {
    const visible = new Set(visibleObjects().map((object) => object.id));
    objectsByAnchor.forEach((objects, anchor) => {
      const anchorPosition = sceneState.graphPositions.get(anchor) || sceneState.graphPositions.get(objects[0]?.anchor);
      if (!anchorPosition) return;
      objects.forEach((object, index) => {
        if (!visible.has(object.id)) return;
        const selected = state.selectedObjectId === object.id;
        const hovered = state.hoveredObjectId === object.id;
        const position = anchorPosition.clone().add(markerOffset(index, objects.length));
        sceneState.objectPositions.set(object.id, position.clone());
        const color = colorForObject(object, map);
        const material = makeMaterial(color, {
          opacity: 0.92,
          emissive: selected || hovered ? color : '#000000',
          emissiveIntensity: selected ? 0.24 : hovered ? 0.14 : 0,
        });
        const mesh = new THREE.Mesh(geometryForObject(object), material);
        mesh.position.copy(position);
        mesh.userData = {
          kind: 'training-object',
          objectId: object.id,
          title: object.label,
          body: `${object.path} · ${object.tags.slice(0, 3).join(' ')}`,
        };
        if (object.type === 'gradient') mesh.rotation.z = Math.PI / 2;
        sceneState.root.add(mesh);
        sceneState.objectMeshes.set(object.id, mesh);
        sceneState.hoverables.push(mesh);

        const typeInfo = registry.objectTypes[object.type] || {};
        addLabel(typeInfo.glyph || '?', position.clone().add(new THREE.Vector3(0, 0, 0.22)), {
          x: 0.22,
          y: 0.22,
        }, {
          width: 112,
          height: 112,
          fill: selected ? 'rgba(255,255,255,0.95)' : (els.root.dataset.theme === 'light' ? 'rgba(255,255,255,0.78)' : 'rgba(12,12,12,0.66)'),
          font: '800 42px Inter, sans-serif',
          maxChars: 2,
          renderOrder: 40,
        });

        if (selected) {
          const ring = new THREE.Mesh(
            new THREE.TorusGeometry(OBJECT_SIZE * 0.86, 0.018, 8, 36),
            makeMaterial(els.root.dataset.theme === 'light' ? '#111827' : '#ffffff', { opacity: 0.88 }),
          );
          ring.position.copy(position.clone().add(new THREE.Vector3(0, 0, 0.03)));
          ring.userData = { kind: 'selection-ring' };
          sceneState.root.add(ring);
        }
      });
    });
  }

  function buildRegistryFlows() {
    registry.flows.forEach((flow) => {
      if (state.flowPhase !== 'all' && flow.type !== state.flowPhase) return;
      const source = sceneState.objectPositions.get(flow.source);
      const target = sceneState.objectPositions.get(flow.target);
      if (!source || !target) return;
      const selected = [flow.source, flow.target].includes(state.selectedObjectId);
      const mid = source.clone().lerp(target, 0.5).add(new THREE.Vector3(0, 0, selected ? 0.74 : 0.42));
      const line = makeLine([
        source.clone(),
        mid,
        target.clone(),
      ], colorForEdgeType(flow.type), {
        opacity: selected ? 0.95 : 0.42,
        renderOrder: selected ? 10 : 1,
      });
      line.userData = {
        kind: 'flow',
        title: flow.label,
        body: `${flow.source} -> ${flow.target}`,
      };
      sceneState.root.add(line);
    });
  }

  function fallbackProject(x, y, z = 0) {
    const xPercent = (x / graph.width) * 100;
    const yPercent = (y / graph.height) * 100;
    if (state.view === 'side') {
      const normalizedDepth = (z + 2.4) / 4.8;
      return {
        x: 18 + Math.max(0, Math.min(1, normalizedDepth)) * 68,
        y: 11 + (y / graph.height) * 78,
      };
    }
    if (state.view === 'front') {
      return { x: xPercent, y: yPercent };
    }
    return {
      x: xPercent + z * 2.9,
      y: yPercent - z * 2.1,
    };
  }

  function fallbackNodeCenter(node) {
    return fallbackProject(node.x, node.y, node.depth || 0);
  }

  function fallbackMarkerOffset(index, count) {
    const angle = (index / Math.max(1, count)) * Math.PI * 2 - Math.PI / 2;
    const radiusX = count > 3 ? 2.7 : 2.0;
    const radiusY = count > 3 ? 2.1 : 1.55;
    return {
      x: Math.cos(angle) * radiusX,
      y: Math.sin(angle) * radiusY,
      z: 0.42 + index * 0.03,
    };
  }

  function fallbackElement(tagName, attrs = {}, text = '') {
    const node = document.createElement(tagName);
    Object.entries(attrs).forEach(([key, value]) => {
      if (value === false || value === undefined || value === null) return;
      if (key === 'className') node.className = value;
      else if (key === 'dataset') Object.entries(value).forEach(([dataKey, dataValue]) => { node.dataset[dataKey] = dataValue; });
      else node.setAttribute(key, String(value));
    });
    if (text) node.textContent = text;
    return node;
  }

  function fallbackSvgElement(tagName, attrs = {}) {
    const node = document.createElementNS('http://www.w3.org/2000/svg', tagName);
    Object.entries(attrs).forEach(([key, value]) => {
      if (value !== undefined && value !== null) node.setAttribute(key, String(value));
    });
    return node;
  }

  function buildFallbackLayers(map) {
    const rail = fallbackElement('div', { className: 'training-fallback-layers', 'aria-hidden': 'true' });
    const decoder = clusterById.get('decoder') || { x: 430, y: 239, width: 340, height: 688 };
    const top = decoder.y + 40;
    const bottom = decoder.y + decoder.height - 40;
    const startDepth = -2.35;
    const depthStep = 4.7 / Math.max(1, layerCount - 1);
    for (let index = 0; index < layerCount; index += 1) {
      const yGraph = top + (bottom - top) * (index / Math.max(1, layerCount - 1));
      const projected = fallbackProject(decoder.x + decoder.width / 2, yGraph, startDepth + index * depthStep);
      const dsa = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45].includes(index);
      const dense = index < 2;
      const color = dense ? '#8b949e' : dsa ? (map.get('sem:attention') || '#38bdf8') : (map.get('sem:moe') || '#a855f7');
      const layer = fallbackElement('div', {
        className: 'training-fallback-layer',
        style: `--layer-x:${projected.x}%;--layer-y:${projected.y}%;--layer-color:${color};--layer-index:${index};`,
        title: `Layer ${String(index).padStart(2, '0')}`,
      });
      if (index % 5 === 0 || index === layerCount - 1) {
        layer.dataset.label = `L${index}`;
      }
      rail.appendChild(layer);
    }
    fallbackState.layerRail = rail;
    fallbackState.root.appendChild(rail);
  }

  function buildFallbackGraphNodes(map) {
    const layer = fallbackElement('div', { className: 'training-fallback-nodes' });
    graph.nodes.forEach((node) => {
      const projected = fallbackNodeCenter(node);
      const width = Math.max(7.2, Math.min(18, ((node.width || 180) / graph.width) * 100));
      const height = Math.max(3.2, Math.min(8.5, ((node.height || 50) / graph.height) * 100));
      const key = node.colorKey || `type:${node.kind || 'node'}`;
      const color = node.kind === 'tensor'
        ? (map.get(key) || map.get('io:parameter') || '#d1d5db')
        : (map.get(key) || '#64748b');
      const el = fallbackElement('button', {
        type: 'button',
        className: `training-fallback-node training-fallback-node--${escapeHtml(node.kind || 'node')}`,
        style: `--node-x:${projected.x}%;--node-y:${projected.y}%;--node-w:${width}%;--node-h:${height}%;--node-color:${color};`,
        title: `${node.label} · ${node.typeLabel || node.kind || 'node'}`,
      }, node.label);
      const anchored = registry.objects.find((object) => object.anchor === node.id);
      if (anchored) el.dataset.objectId = anchored.id;
      layer.appendChild(el);
    });
    fallbackState.nodesLayer = layer;
    fallbackState.root.appendChild(layer);
  }

  function buildFallbackObjects(map) {
    const layer = fallbackElement('div', { className: 'training-fallback-objects' });
    const visible = new Set(visibleObjects().map((object) => object.id));
    objectsByAnchor.forEach((objects, anchor) => {
      const node = nodeById.get(anchor);
      if (!node) return;
      const base = fallbackNodeCenter(node);
      objects.forEach((object, index) => {
        if (!visible.has(object.id)) return;
        const offset = fallbackMarkerOffset(index, objects.length);
        const depthAdjusted = fallbackProject(node.x, node.y, offset.z);
        const x = state.view === 'front' ? base.x + offset.x : depthAdjusted.x + offset.x * 0.22;
        const y = state.view === 'front' ? base.y + offset.y : depthAdjusted.y + offset.y;
        fallbackState.positions.set(object.id, { x, y });
        const info = registry.objectTypes[object.type] || {};
        const selected = object.id === state.selectedObjectId;
        const color = colorForObject(object, map);
        const el = fallbackElement('button', {
          type: 'button',
          className: `training-fallback-object ${selected ? 'is-selected' : ''}`,
          dataset: { objectId: object.id, objectType: object.type },
          style: `--object-x:${x}%;--object-y:${y}%;--object-color:${color};`,
          title: `${object.label} · ${object.path}`,
        }, info.glyph || '?');
        const label = fallbackElement('span', { className: 'training-fallback-object-label' }, object.label);
        el.appendChild(label);
        layer.appendChild(el);
      });
    });
    fallbackState.objectsLayer = layer;
    fallbackState.root.appendChild(layer);
  }

  function buildFallbackFlows() {
    const svg = fallbackSvgElement('svg', {
      class: 'training-fallback-flows',
      viewBox: '0 0 100 100',
      preserveAspectRatio: 'none',
      'aria-hidden': 'true',
    });
    registry.flows.forEach((flow) => {
      if (state.flowPhase !== 'all' && flow.type !== state.flowPhase) return;
      const source = fallbackState.positions.get(flow.source);
      const target = fallbackState.positions.get(flow.target);
      if (!source || !target) return;
      const selected = [flow.source, flow.target].includes(state.selectedObjectId);
      const midX = (source.x + target.x) / 2;
      const midY = (source.y + target.y) / 2 - (selected ? 5.8 : 3.6);
      const path = fallbackSvgElement('path', {
        d: `M ${source.x.toFixed(2)} ${source.y.toFixed(2)} Q ${midX.toFixed(2)} ${midY.toFixed(2)} ${target.x.toFixed(2)} ${target.y.toFixed(2)}`,
        class: selected ? 'is-selected' : '',
        stroke: colorForEdgeType(flow.type),
      });
      svg.appendChild(path);
    });
    fallbackState.flowLayer = svg;
    fallbackState.root.appendChild(svg);
  }

  function applyFallbackView() {
    if (!fallbackState.root) return;
    fallbackState.root.dataset.view = state.view;
    const views = {
      iso: '2.5D · SVG fallback · layer ribs and object depth',
      front: 'Front · SVG fallback · architecture topology and object anchors',
      side: 'Side · SVG fallback · all layer ribs by depth',
    };
    if (els.axisReadout) els.axisReadout.textContent = views[state.view] || views.iso;
  }

  function buildFallbackScene() {
    if (!fallbackState.active) return;
    clearFallbackRoot();
    fallbackState.root = fallbackElement('div', {
      className: 'training-fallback-stage',
      dataset: { view: state.view },
      role: 'img',
      'aria-label': '2.5D fallback model architecture with training objects',
    });
    const map = colorMap();
    buildFallbackLayers(map);
    buildFallbackGraphNodes(map);
    buildFallbackObjects(map);
    buildFallbackFlows();
    applyFallbackView();
    els.host.appendChild(fallbackState.root);
  }

  function initFallbackScene(error) {
    fallbackState.active = true;
    els.host.classList.add('is-fallback');
    els.host.dataset.fallbackReason = error?.message || 'webgl unavailable';
    buildFallbackScene();
    els.body.dataset.renderStatus = 'ready';
  }

  function buildScene() {
    clearSceneRoot();
    const map = colorMap();
    buildLayerStack(map);
    buildGraphEdges();
    buildGraphNodes(map);
    buildObjectMarkers(map);
    buildRegistryFlows();
    applyView();
    sceneState.needsRender = true;
  }

  function initThree() {
    sceneState.scene = new THREE.Scene();
    sceneState.root = new THREE.Group();
    sceneState.scene.add(sceneState.root);

    sceneState.camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
    try {
      sceneState.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch (error) {
      sceneState.scene = null;
      sceneState.camera = null;
      sceneState.root = null;
      initFallbackScene(error);
      return;
    }
    sceneState.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    sceneState.renderer.outputColorSpace = THREE.SRGBColorSpace;
    els.host.appendChild(sceneState.renderer.domElement);

    const ambient = new THREE.AmbientLight(0xffffff, els.root.dataset.theme === 'light' ? 1.65 : 1.25);
    sceneState.scene.add(ambient);
    const key = new THREE.DirectionalLight(0xffffff, els.root.dataset.theme === 'light' ? 2.1 : 1.6);
    key.position.set(4, 7, 8);
    sceneState.scene.add(key);
    const fill = new THREE.DirectionalLight(0x7aa2ff, 0.6);
    fill.position.set(-6, -4, 6);
    sceneState.scene.add(fill);

    els.host.addEventListener('pointermove', onPointerMove);
    els.host.addEventListener('pointerleave', onPointerLeave);
    els.host.addEventListener('pointerdown', onPointerDown);
    els.host.addEventListener('pointerup', onPointerUp);
    els.host.addEventListener('wheel', onWheel, { passive: false });
    window.addEventListener('resize', resizeThree);

    resizeThree();
    buildScene();
    animate();
  }

  function resizeThree() {
    if (!sceneState.renderer || !sceneState.camera) return;
    const rect = els.host.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width));
    const height = Math.max(1, Math.floor(rect.height));
    sceneState.renderer.setSize(width, height, false);
    sceneState.camera.aspect = width / height;
    sceneState.camera.updateProjectionMatrix();
    sceneState.needsRender = true;
  }

  function applyView() {
    if (!sceneState.camera || !sceneState.root) {
      applyFallbackView();
      return;
    }
    const camera = sceneState.camera;
    const target = new THREE.Vector3(0, 0, 0);
    const views = {
      iso: {
        position: new THREE.Vector3(7.2, -5.4, 12.4),
        label: '2.5D · X=module · Y=layer · Z=object depth',
      },
      front: {
        position: new THREE.Vector3(0, 0, 14.2),
        label: 'Front · architecture topology and object anchors',
      },
      side: {
        position: new THREE.Vector3(13.5, 0, 2.4),
        label: 'Side · all layer ribs and object depth',
      },
    };
    const view = views[state.view] || views.iso;
    camera.position.copy(view.position);
    camera.lookAt(target);
    camera.up.set(0, 1, 0);
    sceneState.root.position.copy(state.sceneOffset);
    sceneState.root.scale.setScalar(state.zoom);
    sceneState.root.rotation.x = state.rootRotation.x;
    sceneState.root.rotation.y = state.rootRotation.y;
    if (els.axisReadout) els.axisReadout.textContent = view.label;
    sceneState.needsRender = true;
    applyFallbackView();
  }

  function animate() {
    sceneState.frameId = window.requestAnimationFrame(animate);
    if (!sceneState.needsRender) return;
    sceneState.renderer.render(sceneState.scene, sceneState.camera);
    sceneState.needsRender = false;
  }

  function objectUnderPointer(event) {
    const rect = sceneState.renderer.domElement.getBoundingClientRect();
    sceneState.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    sceneState.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    sceneState.raycaster.setFromCamera(sceneState.pointer, sceneState.camera);
    const hits = sceneState.raycaster.intersectObjects(sceneState.hoverables, false);
    return hits[0]?.object || null;
  }

  function onPointerMove(event) {
    if (state.dragStart) {
      const dx = event.clientX - state.dragStart.x;
      const dy = event.clientY - state.dragStart.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) state.dragStart.dragged = true;
      if (state.view === 'iso') {
        state.rootRotation.y = state.dragStart.rotationY + dx * 0.004;
        state.rootRotation.x = state.dragStart.rotationX + dy * 0.002;
      } else {
        state.sceneOffset.x = state.dragStart.offsetX + dx * 0.009 / state.zoom;
        state.sceneOffset.y = state.dragStart.offsetY - dy * 0.009 / state.zoom;
      }
      applyView();
      return;
    }

    const hit = objectUnderPointer(event);
    if (!hit) {
      clearHover();
      return;
    }
    const objectId = hit.userData.objectId || null;
    if (state.hoveredObjectId !== objectId) {
      state.hoveredObjectId = objectId;
      buildScene();
    }
    showTooltip(event, hit.userData.title, hit.userData.body);
  }

  function onPointerLeave() {
    clearHover();
    state.dragStart = null;
  }

  function onPointerDown(event) {
    if (event.button !== 0) return;
    els.host.setPointerCapture?.(event.pointerId);
    state.dragStart = {
      x: event.clientX,
      y: event.clientY,
      rotationX: state.rootRotation.x,
      rotationY: state.rootRotation.y,
      offsetX: state.sceneOffset.x,
      offsetY: state.sceneOffset.y,
      dragged: false,
    };
  }

  function onPointerUp(event) {
    const drag = state.dragStart;
    state.dragStart = null;
    els.host.releasePointerCapture?.(event.pointerId);
    if (drag?.dragged) return;
    const hit = objectUnderPointer(event);
    if (!hit) return;
    if (hit.userData.objectId) {
      selectObject(hit.userData.objectId);
    } else if (hit.userData.nodeId) {
      const object = registry.objects.find((candidate) => candidate.anchor === hit.userData.nodeId);
      if (object) selectObject(object.id);
    }
  }

  function onWheel(event) {
    event.preventDefault();
    const delta = event.deltaY > 0 ? -0.08 : 0.08;
    state.zoom = Math.max(0.62, Math.min(1.7, state.zoom + delta));
    applyView();
  }

  function showTooltip(event, title, body) {
    if (!title) return;
    els.tooltip.hidden = false;
    els.tooltipTitle.textContent = title;
    els.tooltipBody.textContent = body || '';
    const offset = 14;
    const rect = els.tooltip.getBoundingClientRect();
    const left = Math.min(window.innerWidth - rect.width - 12, event.clientX + offset);
    const top = Math.min(window.innerHeight - rect.height - 12, event.clientY + offset);
    els.tooltip.style.left = `${Math.max(12, left)}px`;
    els.tooltip.style.top = `${Math.max(12, top)}px`;
  }

  function clearHover() {
    if (state.hoveredObjectId) {
      state.hoveredObjectId = null;
      buildScene();
    }
    els.tooltip.hidden = true;
  }

  function setView(view) {
    state.view = view;
    state.sceneOffset.set(0, 0, 0);
    state.rootRotation.x = 0;
    state.rootRotation.y = 0;
    state.zoom = view === 'side' ? 1.06 : 1;
    syncViewButtons();
    applyView();
    if (fallbackState.active) buildFallbackScene();
    updateHud();
  }

  function syncViewButtons() {
    document.querySelectorAll('[data-view-mode]').forEach((button) => {
      button.classList.toggle('is-selected', button.dataset.viewMode === state.view);
    });
  }

  function selectObject(objectId) {
    if (!objectById.has(objectId)) return;
    state.selectedObjectId = objectId;
    renderAll();
  }

  function clearSelection() {
    state.selectedObjectId = null;
    renderAll();
  }

  function typeButton(type, info) {
    const selected = state.activeTypes.has(type);
    return `<button class="btn btn-ghost ${selected ? 'is-selected' : ''}" type="button" data-type-filter="${escapeHtml(type)}">${escapeHtml(info.label)}</button>`;
  }

  function renderTypeFilters() {
    els.typeFilters.innerHTML = Object.entries(registry.objectTypes)
      .map(([type, info]) => typeButton(type, info))
      .join('');
  }

  function renderStrategySwitch() {
    const buttons = [
      { id: 'all', label: 'All' },
      ...registry.strategies.map((strategy) => ({ id: strategy.id, label: strategy.label })),
    ];
    els.strategySwitch.innerHTML = buttons.map((item) => `
      <button class="btn btn-ghost ${state.strategy === item.id ? 'is-selected' : ''}" type="button" data-strategy="${escapeHtml(item.id)}">${escapeHtml(item.label)}</button>
    `).join('');
  }

  function renderObjectTree() {
    const objects = visibleObjects();
    const grouped = new Map();
    objects.forEach((object) => {
      const list = grouped.get(object.type) || [];
      list.push(object);
      grouped.set(object.type, list);
    });
    els.objectCount.textContent = `${objects.length} objects`;

    if (!objects.length) {
      els.objectTree.innerHTML = '<div class="training-empty">No matching training objects</div>';
      return;
    }

    els.objectTree.innerHTML = Array.from(grouped.entries()).map(([type, list]) => {
      const info = registry.objectTypes[type] || { label: type, glyph: '?' };
      return `
        <section class="training-object-group">
          <div class="training-object-group-title"><span>${escapeHtml(info.label)}</span><span>${list.length}</span></div>
          ${list.map((object) => objectButton(object, info)).join('')}
        </section>
      `;
    }).join('');
  }

  function objectButton(object, info = registry.objectTypes[object.type] || {}) {
    return `
      <button class="training-object-button ${state.selectedObjectId === object.id ? 'is-selected' : ''}" type="button" data-object-id="${escapeHtml(object.id)}">
        <span class="training-object-glyph">${escapeHtml(info.glyph || '?')}</span>
        <span class="training-object-main">
          <span class="training-object-label">${escapeHtml(object.label)}</span>
          <span class="training-object-path">${escapeHtml(object.path)}</span>
          ${tagRow(object.tags)}
        </span>
      </button>
    `;
  }

  function tagRow(tags) {
    return `<span class="training-tag-row">${(tags || []).map((tag) => `
      <span class="training-tag" data-kind="${escapeHtml(tagKind(tag))}">${escapeHtml(tag)}</span>
    `).join('')}</span>`;
  }

  function renderInspector() {
    const object = selectedObject();
    if (!object) {
      els.inspectorMeta.textContent = 'overview';
      els.inspectorBody.innerHTML = overviewInspector();
      return;
    }
    const typeInfo = registry.objectTypes[object.type] || {};
    const anchorEvidence = graph.trainingEvidence?.[object.anchor] || {};
    els.inspectorMeta.textContent = object.id;
    els.terminalObject.textContent = object.id;
    els.inspectorBody.innerHTML = `
      <section class="training-inspector-section">
        <div class="training-inspector-kicker">${escapeHtml(typeInfo.label || object.type)}</div>
        <h2 class="training-inspector-title">${escapeHtml(object.label)}</h2>
        <div class="training-inspector-path">${escapeHtml(object.path)}</div>
        ${tagRow(object.tags)}
        <p class="training-inspector-note">${escapeHtml(object.note)}</p>
      </section>
      <section class="training-inspector-section">
        <div class="training-inspector-kicker">Identity</div>
        <dl class="training-detail-grid">
          <dt>shape</dt><dd>${escapeHtml(object.shape || '-')}</dd>
          <dt>anchor</dt><dd>${escapeHtml(object.anchor)}</dd>
          <dt>view</dt><dd>${escapeHtml(state.view)}</dd>
          <dt>strategy</dt><dd>${escapeHtml(state.strategy)}</dd>
        </dl>
      </section>
      <section class="training-inspector-section">
        <div class="training-inspector-kicker">Linked objects</div>
        <div class="training-linked-list">
          ${(object.linked || []).map((id) => {
            const linked = objectById.get(id);
            if (!linked) return '';
            const linkedType = registry.objectTypes[linked.type] || {};
            return `<div class="training-linked-item"><span class="training-object-glyph">${escapeHtml(linkedType.glyph || '?')}</span><button type="button" data-object-id="${escapeHtml(id)}">${escapeHtml(linked.label)}</button></div>`;
          }).join('') || '<div class="training-empty">No linked objects</div>'}
        </div>
      </section>
      <section class="training-inspector-section">
        <div class="training-inspector-kicker">Evidence</div>
        <p class="training-inspector-note">${escapeHtml(anchorEvidence.what || 'No graph evidence mapped to this anchor.')}</p>
        ${Array.isArray(anchorEvidence.evidence) ? `<div class="training-flow-list">${anchorEvidence.evidence.map((item) => `<div class="training-flow-item"><span class="training-flow-kind">fact</span><span>${escapeHtml(item)}</span></div>`).join('')}</div>` : ''}
      </section>
    `;
  }

  function overviewInspector() {
    return `
      <section class="training-inspector-section">
        <div class="training-inspector-kicker">Model facts</div>
        <h2 class="training-inspector-title">${escapeHtml(registry.modelFacts.name)}</h2>
        <p class="training-inspector-note">${escapeHtml(registry.modelFacts.scenario)}</p>
      </section>
      <section class="training-inspector-section">
        <dl class="training-detail-grid">
          <dt>layers</dt><dd>${registry.modelFacts.layers}</dd>
          <dt>MoE</dt><dd>${escapeHtml(registry.modelFacts.moeRange)}</dd>
          <dt>experts</dt><dd>${registry.modelFacts.routedExperts}</dd>
          <dt>context</dt><dd>${registry.modelFacts.contextLength}</dd>
        </dl>
      </section>
    `;
  }

  function renderBottom() {
    const object = selectedObject();
    const relatedIds = new Set(object?.linked || []);
    if (object) relatedIds.add(object.id);
    const flows = registry.flows.filter((flow) => {
      if (state.flowPhase !== 'all' && flow.type !== state.flowPhase) return false;
      if (!object) return true;
      return relatedIds.has(flow.source) || relatedIds.has(flow.target);
    });
    const strategy = selectedStrategy();
    const anomaly = weightData.gate?.anomaly;
    const traffic = commData.flowAt ? commData.flowAt(tsData.defaultStep || 2000) : null;

    els.bottomMeta.textContent = object ? object.label : 'association and flow';
    els.relationGrid.innerHTML = `
      <section class="training-relation-panel">
        <h3>Association</h3>
        <div class="training-linked-list">
          ${object ? (object.linked || []).map((id) => {
            const linked = objectById.get(id);
            const info = registry.objectTypes[linked?.type] || {};
            return linked ? `<div class="training-linked-item"><span class="training-object-glyph">${escapeHtml(info.glyph || '?')}</span><button type="button" data-object-id="${escapeHtml(id)}">${escapeHtml(linked.path)}</button></div>` : '';
          }).join('') : registry.objects.slice(0, 5).map((item) => `<div class="training-linked-item"><span class="training-object-glyph">${escapeHtml((registry.objectTypes[item.type] || {}).glyph || '?')}</span><button type="button" data-object-id="${escapeHtml(item.id)}">${escapeHtml(item.path)}</button></div>`).join('')}
        </div>
      </section>
      <section class="training-relation-panel">
        <h3>Flow</h3>
        <div class="training-flow-list">
          ${flows.slice(0, 8).map((flow) => `<div class="training-flow-item"><span class="training-flow-kind">${escapeHtml(flow.type)}</span><span>${escapeHtml(flow.label)}</span></div>`).join('') || '<div class="training-empty">No matching flow edges</div>'}
        </div>
      </section>
      <section class="training-relation-panel">
        <h3>Strategy</h3>
        <div class="training-flow-list">
          <div class="training-flow-item"><span class="training-flow-kind">${escapeHtml(strategy?.label || 'All')}</span><span>${escapeHtml(strategy?.summary || 'All object identifiers remain visible.')}</span></div>
          <div class="training-flow-item"><span class="training-flow-kind">fault</span><span>${escapeHtml(anomaly?.note || 'no active anomaly')}</span></div>
          <div class="training-flow-item"><span class="training-flow-kind">traffic</span><span>${escapeHtml(traffic ? `TP2 col flow ${traffic.colFlow?.[2]}` : 'not loaded')}</span></div>
        </div>
      </section>
    `;

    const sources = [
      ...registry.sourceManifest.local,
      ...registry.sourceManifest.publicSmallFiles,
    ];
    els.sourceStrip.innerHTML = sources.map((source) => `
      <div class="training-source-item">
        <strong>${escapeHtml(source.id)}</strong>
        <span>${escapeHtml(source.role)}</span>
      </div>
    `).join('');
  }

  function updateHud() {
    const objects = visibleObjects();
    const strategy = selectedStrategy();
    els.stageMeta.textContent = `${registry.modelFacts.layers} layers · ${objects.length} visible objects`;
    els.trainingHud.innerHTML = [
      `${registry.modelFacts.layers} layers`,
      `${registry.modelFacts.routedExperts} experts`,
      `${objects.length} objects`,
      strategy ? strategy.label : 'All strategies',
      state.view,
    ].map((item) => `<span class="training-hud-chip">${escapeHtml(item)}</span>`).join('');
  }

  function renderStatus() {
    const object = selectedObject();
    els.statusStrip.textContent = object
      ? `${object.id} · ${object.tags.join(' · ')}`
      : `ready · ${visibleObjects().length} objects`;
  }

  function renderAll() {
    renderTypeFilters();
    renderStrategySwitch();
    renderObjectTree();
    renderInspector();
    renderBottom();
    updateHud();
    renderStatus();
    if (sceneState.scene) buildScene();
    if (fallbackState.active) buildFallbackScene();
    els.body.dataset.renderStatus = 'ready';
  }

  function bindDomEvents() {
    document.addEventListener('click', (event) => {
      const objectButton = event.target.closest('[data-object-id]');
      if (objectButton) {
        selectObject(objectButton.dataset.objectId);
        return;
      }
      const typeButtonEl = event.target.closest('[data-type-filter]');
      if (typeButtonEl) {
        const type = typeButtonEl.dataset.typeFilter;
        if (state.activeTypes.has(type)) state.activeTypes.delete(type);
        else state.activeTypes.add(type);
        renderAll();
        return;
      }
      const strategyButton = event.target.closest('[data-strategy]');
      if (strategyButton) {
        state.strategy = strategyButton.dataset.strategy;
        renderAll();
        return;
      }
      const viewButton = event.target.closest('[data-view-mode]');
      if (viewButton) {
        setView(viewButton.dataset.viewMode);
        return;
      }
      const flowButton = event.target.closest('[data-flow-phase]');
      if (flowButton) {
        state.flowPhase = flowButton.dataset.flowPhase;
        document.querySelectorAll('[data-flow-phase]').forEach((button) => {
          button.classList.toggle('is-selected', button === flowButton);
        });
        renderAll();
      }
    });

    els.objectSearch.addEventListener('input', () => {
      state.query = els.objectSearch.value;
      renderAll();
    });

    els.themeToggle.addEventListener('click', () => {
      const next = els.root.dataset.theme === 'light' ? 'dark' : 'light';
      els.root.dataset.theme = next;
      window.localStorage?.setItem(THEME_KEY, next);
      renderAll();
    });

    els.clearSelection.addEventListener('click', clearSelection);
  }

  function init() {
    window.PtoIdeFrame?.init?.(els.frame, { onResize: resizeThree });
    bindDomEvents();
    syncViewButtons();
    renderAll();
    initThree();
    window.setTimeout(resizeThree, 60);
  }

  init();
})();
