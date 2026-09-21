// ─────────────────────────────────────────────────────────────────────────────
// WholeNetGraph — 整网图（Model → DecoderLayer → Attention/MoE → QKV/Expert →
// Operator 分层计算图）。直接复用 pto-design-system 的 model-graphviz pattern
// （vendored 于 src/vendor/model-graphviz，全局 window.PtoModelGraphvizPattern），
// 用参考页自身布局引擎导出的「已定位」openPangu 图。
//
// 在原 pattern 之上加两件事，用来和立方重组/整网切分联动：
//   1) 前向/反向方向箭头：训练的 forward（激活下行，青）/ backward（梯度上行，蓝），
//      用箭头方向 + 颜色双重区分（全链=同时显示）。
//   2) 切分维高亮：点算子 → 回调其并行切分维（TP/PP/DP/EP）驱动立方染色；
//      外部指定 highlightDim → 图上高亮属于该维的算子、其余淡出。
// ─────────────────────────────────────────────────────────────────────────────
import { useEffect, useRef } from 'react';
import '../vendor/model-graphviz/pattern.css';
import '../vendor/model-graphviz/pattern.js';
import { OPENPANGU_GRAPH, NODE_DIM, type ParCutDim } from '../vendor/model-graphviz/graph-meta';
import { PARALLEL_COLORS } from '../scene/data';

export type FlowDir = 'fwd' | 'bwd' | 'both';

const FWD_COLOR = '#22d3ee';   // 前向：激活下行（青，对齐 PP 甘特 F 色）
const BWD_COLOR = '#6b8bff';   // 反向：梯度上行（蓝，对齐 PP 甘特 B 色）

export interface WholeNetGraphProps {
  dark: boolean;
  /** 前向 / 反向 / 全链 —— 决定画哪个方向的数据流箭头 */
  direction: FlowDir;
  /** 高亮某并行切分维的算子（其余淡出）；null = 不高亮 */
  highlightDim?: ParCutDim | null;
  /** 受控选中算子 id（联动：立方 → 图 反查时用） */
  selectedNodeId?: string | null;
  /** 点击算子回调：nodeId + 它被哪种并行切分（供立方按维染色） */
  onSelectNode?: (nodeId: string | null, dim: ParCutDim | null) => void;
  style?: React.CSSProperties;
}

let markerSeq = 0;

export function WholeNetGraph({ dark, direction, highlightDim = null, selectedNodeId = null, onSelectNode, style }: WholeNetGraphProps) {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const ctrlRef = useRef<ModelGraphvizController | null>(null);
  const markerRef = useRef<{ fwd: string; bwd: string } | null>(null);
  const curSelRef = useRef<string | null>(null);
  // 最新 props 供「渲染后立即套用」读取（避免重建整图）
  const dirRef = useRef(direction); dirRef.current = direction;
  const dimRef = useRef(highlightDim); dimRef.current = highlightDim;
  const selRef = useRef(selectedNodeId); selRef.current = selectedNodeId;
  const onSelRef = useRef(onSelectNode); onSelRef.current = onSelectNode;

  // 主挂载：theme 变（换深浅色 → 节点色在渲染时烘焙）时整图重建，之后轻量 effect 只调方向/高亮/选中。
  useEffect(() => {
    const helper = window.PtoModelGraphvizPattern;
    const stage = stageRef.current;
    if (!helper || !stage) return;
    // 浅色主题：套参考页同一套 modelArchitecture 柔和调色板（否则默认高饱和色偏"深"）
    const colormap = !dark && helper.modelArchitectureColormap
      ? helper.modelArchitectureColormap(OPENPANGU_GRAPH, { theme: 'light', lightHsl: { hue: 2, saturation: 79, lightness: 76 } })
      : undefined;
    const ctrl = helper.renderController(stage, OPENPANGU_GRAPH, {
      theme: dark ? 'dark' : 'light',
      colormap,
      reportOverlays: false,   // 无 reportPriority → 关掉右侧优先级药丸（否则边缘残留小点）
      evidence: false,
      onSelect: (info: { nodeId?: string | null } | null) => {
        const id = info?.nodeId ?? null;
        curSelRef.current = id;
        onSelRef.current?.(id, id ? (NODE_DIM[id] ?? null) : null);
      },
    } as Record<string, unknown>);
    ctrlRef.current = ctrl;
    if (ctrl?.svg) {
      ctrl.svg.style.width = '100%';
      ctrl.svg.style.height = 'auto';
      ctrl.svg.style.display = 'block';
      // 折叠/展开手柄（cluster collapse toggle）依赖原布局引擎重排；此处用的是静态定位图，
      // 收起不会重新布局 → 隐藏这些手柄，避免右缘残留一排非功能小点。
      ctrl.svg.querySelectorAll<SVGElement>('.pto-model-graphviz-toggle, .pto-model-graphviz-toggle-icon')
        .forEach((el) => { el.style.display = 'none'; });
      markerRef.current = ensureMarkers(ctrl.svg);
      applyDirection(ctrl.svg, dirRef.current, markerRef.current);
      applyDim(ctrl.svg, dimRef.current);
      if (selRef.current) { curSelRef.current = selRef.current; ctrl.selectNode(selRef.current, { source: 'prop' }); }
    }
    return () => { ctrl?.destroy?.(); stage.innerHTML = ''; ctrlRef.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dark]);

  // 方向箭头
  useEffect(() => {
    const ctrl = ctrlRef.current;
    if (ctrl?.svg && markerRef.current) applyDirection(ctrl.svg, direction, markerRef.current);
  }, [direction]);

  // 切分维高亮
  useEffect(() => {
    const ctrl = ctrlRef.current;
    if (ctrl?.svg) applyDim(ctrl.svg, highlightDim);
  }, [highlightDim]);

  // 受控选中（防回环：与刚点击/已套用的选中相同则跳过）
  useEffect(() => {
    const ctrl = ctrlRef.current;
    if (!ctrl) return;
    if ((selectedNodeId ?? null) === curSelRef.current) return;
    curSelRef.current = selectedNodeId ?? null;
    if (selectedNodeId) ctrl.selectNode(selectedNodeId, { source: 'prop' });
    else ctrl.clearSelection();
  }, [selectedNodeId]);

  return (
    <div
      ref={stageRef}
      data-theme={dark ? 'dark' : 'light'}
      aria-label="整网图（openPangu-2.0-Flash 分层计算图）"
      style={{ width: '100%', height: '100%', overflow: 'auto', background: 'var(--panel-solid)', ...style }}
    />
  );
}

// ── 方向箭头：给 SVG 注入前向(青)/反向(蓝)两个 marker，按方向给每条边挂 marker-end/start + 染色 ──
function ensureMarkers(svg: SVGSVGElement): { fwd: string; bwd: string } {
  const defs = svg.querySelector('defs') || svg.insertBefore(document.createElementNS('http://www.w3.org/2000/svg', 'defs'), svg.firstChild);
  const seq = (markerSeq += 1);
  const fwd = `wng-fwd-${seq}`, bwd = `wng-bwd-${seq}`;
  // 前向：marker-end + orient auto → 顺流指向下游（target）
  defs.appendChild(marker(fwd, 'auto', FWD_COLOR));
  // 反向：marker-start + orient auto-start-reverse → 逆流指向上游（source），即梯度方向
  defs.appendChild(marker(bwd, 'auto-start-reverse', BWD_COLOR));
  return { fwd, bwd };
}
function marker(id: string, orient: string, fill: string): SVGMarkerElement {
  const NS = 'http://www.w3.org/2000/svg';
  const m = document.createElementNS(NS, 'marker');
  m.setAttribute('id', id);
  m.setAttribute('viewBox', '0 0 10 10');
  m.setAttribute('refX', '8.5'); m.setAttribute('refY', '5');
  m.setAttribute('markerWidth', '7'); m.setAttribute('markerHeight', '7');
  m.setAttribute('orient', orient);
  const p = document.createElementNS(NS, 'path');
  p.setAttribute('d', 'M 0 0 L 10 5 L 0 10 z');
  p.setAttribute('fill', fill);
  m.appendChild(p);
  return m;
}
function applyDirection(svg: SVGSVGElement, dir: FlowDir, ids: { fwd: string; bwd: string }) {
  const edges = svg.querySelectorAll<SVGPathElement>('path.pto-model-graphviz-edge');
  edges.forEach((el) => {
    if (dir === 'fwd') {
      el.style.markerEnd = `url(#${ids.fwd})`; el.style.markerStart = 'none'; el.style.stroke = FWD_COLOR; el.style.strokeOpacity = '0.85';
    } else if (dir === 'bwd') {
      el.style.markerEnd = 'none'; el.style.markerStart = `url(#${ids.bwd})`; el.style.stroke = BWD_COLOR; el.style.strokeOpacity = '0.85';
    } else {
      el.style.markerEnd = `url(#${ids.fwd})`; el.style.markerStart = `url(#${ids.bwd})`; el.style.stroke = ''; el.style.strokeOpacity = '';
    }
  });
}

// ── 切分维高亮：属于该维的算子保持全亮 + 该维色发光，其余淡出；边同理 ──
function applyDim(svg: SVGSVGElement, dim: ParCutDim | null) {
  const color = dim ? PARALLEL_COLORS[dim] : '';
  svg.querySelectorAll<SVGGElement>('[data-node-id]').forEach((el) => {
    const id = el.getAttribute('data-node-id') || '';
    if (!dim) { el.style.opacity = ''; el.style.filter = ''; return; }
    const match = NODE_DIM[id] === dim;
    el.style.opacity = match ? '1' : '0.16';
    el.style.filter = match ? `drop-shadow(0 0 5px ${color})` : '';
  });
  svg.querySelectorAll<SVGPathElement>('path.pto-model-graphviz-edge').forEach((el) => {
    if (!dim) { el.style.opacity = ''; return; }
    const s = el.getAttribute('data-source') || '', t = el.getAttribute('data-target') || '';
    el.style.opacity = (NODE_DIM[s] === dim && NODE_DIM[t] === dim) ? '0.95' : '0.1';
  });
}
