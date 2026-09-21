/**
 * ConsoleView — 联动控制台. One linked module fusing the three existing views per the reference
 * Smartscape (Dynatrace-style) interaction:
 *   · LEFT  = 平面视图「层级图」改造成 Smartscape 8 级漏斗 (全球 L7→集群 L6→服务池 L5→Pod L4→Host L3→Chip L2 + 卡内 Die L1/Core-Group L0)，
 *             作为控制：点一个实体只展开/高亮它的「链路」(祖先+后代，按方向过滤)。
 *   · RIGHT = 阵列全景 (FullPodScene, 全量 Pod) 作为主视图：scopeOnly 模式只显示左侧链路的内容
 *             (链路内按状态/负载上色，链路外全部压暗)。
 *   · 运行状态 = 分析仪表 (集群 KPI · 实体辅助指标 · DAVIS 根因)。
 *
 * 所有样式/图元/状态/连接/上下层级与层内关系都用既有方案：FullPodScene 组件 + 同一套 data.ts
 * 色彩/状态/负载函数 (loadColor / loadState / stateColor / nodeLoad / isHot / ENTITY_COLORS / PLANES)。
 */
import { useContext, useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { OrbitControls, GizmoHelper, GizmoViewcube } from '@react-three/drei';
import * as THREE from 'three';
import {
  GENERATIONS, ENTITY_COLORS, PARALLEL_COLORS, PARALLEL_COLORS_SP, PARTITION_META, PLANES, LEVEL_PHYS,
  PODS_PER_CLUSTER, POOLS_PER_CLUSTER, PODS_PER_POOL,
  loadColor, loadState, stateColor, STATE_LABELS, nodeLoad, isHot,
  REPLAY, cardLoad01, cardStraggler, cardFault, cardMetric01, parallelMap,
  type Gen, type PartitionDim, type ParDim, type RunPhase, type RunMode, type ViewSync,
} from '../scene/data';
import { FullPodScene, SceneTheme, type CommOverlays } from '../scene/scenes';
import { CoreGroupPattern } from './CoreGroupPattern';
import { SceneVisualProfileContext, sceneSurface } from '../scene/visual-profile';
import { CubeView } from './CubeView';
import { DualModeMonitor } from './DualModeMonitor';
import { CommDeepDive } from './CommDeepDive';
import { LAYOUT_VIEWS, LAYOUT_LABEL, type LayoutView } from '../scene/layout';
import { type LevelKey } from '../scene/data';

// focus.level → CubeView 聚合粒度（层级图选层 → 魔方按该粒度重排/染色）
const FOCUS_AGG: Record<string, LevelKey> = { super: 'super', node: 'node', card: 'card', die: 'die', core: 'core' };

// ── hierarchy fan-out (8×8 schematic shared with FullPodScene full=true): 8 卡/刀片 · 8 刀片/柜 →
//    64 卡/柜. A global card index `k` maps the SAME way in the left 层级 and the right 3D array. ──
export const CPB = 8, BPC = 8, PER_CAB = CPB * BPC;
const STEP_MAX = REPLAY.stepMax, EVT_LO = REPLAY.evtLo, EVT_HI = REPLAY.evtHi, EVT_CAB = REPLAY.evtCab;   // shared replay/event window (data.ts)

// 导出（供统一驾驶舱复用同一套「平面层级图」Smartscape 与选区模型）。
export type Workload = 'pretrain' | 'prefill' | 'decode';
export type Metric = 'util' | 'strag' | 'fault';
type Lens = 'heat' | 'flow' | 'domain' | 'phys';
export type Dir = 'all' | 'up' | 'down';
// hw-native-sys L4→L0 焦点级（L7 全球/L6 集群/L5 服务池 为上层上下文，点击即回到整 Pod，不作 focus）。
// 机柜不是层级（仅物理分组）、Tile 不是层级（L0 Core-Group 内部），均不出现在焦点级里。
export type Level = 'super' | 'node' | 'card' | 'die' | 'core';
export type Focus = { level: Level; card: number; die?: number; core?: number } | null;

export const WL: Record<Workload, { label: string; kind: RunPhase['kind'] }> = {
  pretrain: { label: '预训练', kind: 'compute' },
  prefill: { label: 'Prefill', kind: 'compute' },
  decode: { label: 'Decode', kind: 'comm' },
};
const M_LABEL: Record<Metric, string> = { util: '利用率', strag: '掉队率', fault: '故障度' };
const LENS_LABEL: Record<Lens, string> = { heat: '状态热力', flow: '机柜流量', domain: '通信域', phys: '物理链路' };
const LEVEL_NAME: Record<string, string> = { global: 'Global', cluster: 'Cluster', pool: 'Service Pool', super: 'Pod', node: 'Host', card: 'Chip·NPU', die: 'Die', core: 'Core-Group' };

// ── metric model — thin aliases over the SHARED model in data.ts (same value as 运行状态) ──
const cardLoad = (k: number, wlKind: string, step: number) => cardLoad01(k, wlKind, step);
const isStrag = (k: number, step: number) => cardStraggler(k, step);
const isFault = (k: number, step: number) => cardFault(k, step);
const cardMetric = (k: number, metric: Metric, wlKind: string, step: number) => cardMetric01(k, metric, wlKind, step);

// ── hierarchy navigation / scope — 阶梯（Le）：L4 Pod=3 · L3 Host=4 · L2 Chip=5（含 die/core）。
//    Pod 直接含 1024 Host、Host 含 8 Chip（无机柜档位；机柜仅为 3D 物理分组）。 ──
export function scopeRange(f: Focus, N: number): [number, number] {
  if (!f || f.level === 'super') return [0, N];
  if (f.level === 'node') { const n = Math.floor(f.card / CPB); return [n * CPB, Math.min(N, (n + 1) * CPB)]; }
  return [f.card, f.card + 1];   // card / die / core
}
// Which Chip indices to show in the L2 row. Returns null = overview (show first BUDGET chips).
// Only filters when a Host is selected: show that host's CPB chips so the row reads as containment.
function tierInScope(Le: number, focus: Focus, _dir: Dir, N: number, _nBlades: number): number[] | null {
  if (!focus || focus.level === 'super') return null;
  if (focus.level === 'node' && Le === 5) {
    const n = Math.floor(focus.card / CPB);
    return Array.from({ length: CPB }, (_, i) => n * CPB + i).filter((i) => i < N);
  }
  return null;
}
function entityToFocus(Le: number, idx: number): Focus {
  if (Le === 3) return { level: 'super', card: 0 };   // Pod = whole scope
  if (Le === 4) return { level: 'node', card: idx * CPB };   // Host
  return { level: 'card', card: idx };                 // Chip (Le 5)
}
export function focusToSel(f: Focus): { lv: number; i: number } | null {
  // 3D 全景选择级：lv 0=card · lv 1=blade(Host) · lv 2=cabinet(物理分组)。
  if (!f || f.level === 'super') return null;
  if (f.level === 'node') return { lv: 1, i: Math.floor(f.card / CPB) };
  return { lv: 0, i: f.card };   // card / die / core → 高亮该卡
}
export function selToFocus(s: { lv: number; i: number } | null): Focus {
  if (!s) return null;
  if (s.lv === 1) return { level: 'node', card: s.i * CPB };
  if (s.lv === 2) return { level: 'super', card: 0 };   // 机柜=物理分组，不是层级 → 回到整 Pod
  return { level: 'card', card: s.i };
}
function focusName(f: Focus): string {
  if (!f || f.level === 'super') return '全量 Pod';
  const k = f.card;
  if (f.level === 'node') return `Host B${Math.floor(k / CPB)}`;
  if (f.level === 'card') return `Chip r${k}（device）`;
  if (f.level === 'die') return `Chip ${k} · Die ${f.die ?? 0}`;
  if (f.level === 'core') return `Chip ${k} · Core-Group 核 #${f.core ?? 0}`;
  return `Chip ${k}`;
}

// ── shared button language ──
const ACCENT = '#4369ef';
const SECONDARY: React.CSSProperties = { border: '1px solid var(--button-secondary-border)', background: 'var(--button-secondary-bg)', color: 'var(--foreground-muted)' };
function ink(hex: string): string { const h = hex.replace('#', ''); if (h.length < 6) return '#fff'; const r = parseInt(h.slice(0, 2), 16), g = parseInt(h.slice(2, 4), 16), b = parseInt(h.slice(4, 6), 16); return 0.299 * r + 0.587 * g + 0.114 * b > 150 ? '#10131a' : '#fff'; }
function navBtn(on: boolean): React.CSSProperties { return on ? { border: '1px solid var(--primary)', background: 'var(--primary)', color: 'var(--primary-foreground)', fontWeight: 600, transform: 'translateY(-1px)', boxShadow: '0 1px 3px rgba(67,105,239,0.40)' } : { ...SECONDARY }; }
function toggleBtn(on: boolean, c: string): React.CSSProperties { return on ? { border: `1px solid ${c}`, background: c, color: ink(c), fontWeight: 600 } : { ...SECONDARY }; }
const GLAB: React.CSSProperties = { fontSize: 11, fontWeight: 500, letterSpacing: 0.5, textTransform: 'uppercase', color: 'var(--tx3)', alignSelf: 'center' };
const TNUM: React.CSSProperties = { fontVariantNumeric: 'tabular-nums' };
const btnBase: React.CSSProperties = { padding: '4px 10px', fontSize: 11.5, borderRadius: 8, cursor: 'pointer' };
const OVERLAYS: CommOverlays = { ring: false, a2a: false, tile: true, cores: true };

// collective-comm glyph: ring (环状 AllReduce) / a2a (全互联 All-to-All) / p2p (阶段链)
function CollGlyph({ pat, c }: { pat: 'ring' | 'a2a' | 'p2p'; c: string }) {
  return (
    <svg width={14} height={14} viewBox="0 0 14 14" style={{ flexShrink: 0 }} aria-hidden>
      {pat === 'ring' && <circle cx={7} cy={7} r={4.6} fill="none" stroke={c} strokeWidth={1.5} />}
      {pat === 'a2a' && <><line x1={2} y1={2} x2={12} y2={12} stroke={c} strokeWidth={1.3} /><line x1={12} y1={2} x2={2} y2={12} stroke={c} strokeWidth={1.3} /><line x1={2} y1={7} x2={12} y2={7} stroke={c} strokeWidth={1.3} /></>}
      {pat === 'p2p' && <><line x1={2} y1={7} x2={11} y2={7} stroke={c} strokeWidth={1.5} /><path d="M8 4 L12 7 L8 10" fill="none" stroke={c} strokeWidth={1.5} strokeLinejoin="round" /></>}
    </svg>
  );
}

// Frame the orthographic camera on the focused scope (pan target + zoom); with no focus it settles
// on the whole-field overview. Animates on focus CHANGE then releases, so the user can still orbit/zoom.
function FrameCamera({ bounds, reach, controls, zoomScale = 1 }: {
  bounds: { cx: number; cy: number; cz: number; r: number } | null;
  reach: number;
  controls: React.MutableRefObject<{ target: THREE.Vector3; update: () => void } | null>;
  zoomScale?: number;
}) {
  const { camera, size } = useThree();
  const init = useRef(false);
  const settling = useRef(true);
  const tgt = useMemo(() => (bounds
    ? { pos: new THREE.Vector3(bounds.cx, bounds.cy, bounds.cz), worldH: bounds.r * 2.4 }
    : { pos: new THREE.Vector3(0, Math.min(6, reach * 0.1), 0), worldH: Math.max(14, reach * 1.5) }), [bounds, reach]);
  useEffect(() => { settling.current = true; }, [tgt]);   // re-animate whenever the scope changes
  useEffect(() => {                                        // set the 2.5-D iso direction + distance once
    if (init.current || size.height < 10) return; init.current = true;
    camera.position.copy(tgt.pos).addScaledVector(new THREE.Vector3(1, 0.82, 1).normalize(), reach * 1.3);
    camera.up.set(0, 1, 0); camera.updateProjectionMatrix();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camera, reach, size.height]);
  useFrame(() => {
    if (!controls.current || !settling.current || size.height < 10) return;
    controls.current.target.lerp(tgt.pos, 0.14);
    const oc = camera as THREE.OrthographicCamera, want = size.height / tgt.worldH * zoomScale;
    if (oc.isOrthographicCamera) { oc.zoom += (want - oc.zoom) * 0.14; oc.updateProjectionMatrix(); }
    controls.current.update();
    if (controls.current.target.distanceTo(tgt.pos) < 0.08 && Math.abs((oc.zoom ?? want) - want) < 0.06) settling.current = false;
  });
  return null;
}

// ── stats (per-tier distributions + KPI) computed in one pass over all cards ──
export interface Stats {
  kpi: { util: number; hot: number; strag: number; faultDom: number };
  clusterMean: number; cabVals: number[]; ndVals: number[]; cardVals: number[];
  agg: Record<'cluster' | 'cab' | 'node' | 'card', { p50: number; p95: number; red: number }>;
}
// one-pass distribution/KPI over all cards — shared by 联动控制台 与 统一驾驶舱（同一真值源）。
export function computeStats(N: number, nCabs: number, nBlades: number, metric: Metric, wlKind: string, step: number): Stats {
  const cabSum = new Float64Array(nCabs), cabCnt = new Int32Array(nCabs);
  const ndSum = new Float64Array(nBlades), ndCnt = new Int32Array(nBlades);
  const cardVals: number[] = [], stride = Math.max(1, Math.floor(N / 2048));
  let utilSum = 0, hot = 0, strag = 0; const faultNodes = new Set<number>();
  for (let k = 0; k < N; k++) {
    const u = cardLoad(k, wlKind, step); utilSum += u; if (isHot(u)) hot++;
    if (isStrag(k, step)) strag++; if (isFault(k, step)) faultNodes.add(Math.floor(k / CPB));
    const mv = cardMetric(k, metric, wlKind, step);
    const cb = Math.floor(k / PER_CAB), nd = Math.floor(k / CPB);
    cabSum[cb] += mv; cabCnt[cb]++; ndSum[nd] += mv; ndCnt[nd]++;
    if (k % stride === 0) cardVals.push(mv);
  }
  const cabVals = Array.from({ length: nCabs }, (_, i) => (cabCnt[i] ? cabSum[i] / cabCnt[i] : 0));
  const ndVals = Array.from({ length: nBlades }, (_, i) => (ndCnt[i] ? ndSum[i] / ndCnt[i] : 0));
  const agg = (arr: number[]) => {
    if (!arr.length) return { p50: 0, p95: 0, red: 0 };
    const s = [...arr].sort((a, b) => a - b), q = (p: number) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
    let red = 0; for (const v of arr) if (loadState(v) >= 2) red++;
    return { p50: q(0.5), p95: q(0.95), red: red / arr.length };
  };
  const clusterMean = cabVals.reduce((a, b) => a + b, 0) / Math.max(1, cabVals.length);
  return {
    kpi: { util: utilSum / N, hot, strag, faultDom: faultNodes.size },
    clusterMean, cabVals, ndVals, cardVals,
    agg: { cluster: agg([clusterMean]), cab: agg(cabVals), node: agg(ndVals), card: agg(cardVals) },
  };
}

// ── LEFT: Smartscape 8 级漏斗 (改造自平面视图层级图) — 图元/配色与「层级图」「选中链路·层级图」统一：
//    上三级 L7 全球 / L6 集群 / L5 服务池 = 焦点 pill + 半透明幽灵 sibling（不可点，点焦点回整 Pod）；
//    L4 Pod=玫紫 pill · L3 Host=天蓝 · L2 Chip=teal 卡图元(2×2 Die 点) · L1 Die=网格 · L0 Core-Group=原生 CoreGroupMiniSvg 图元。
//    级间连线挂互联名小 chip（DCN/Scale-Out/Pool 内互联/Scale-Up/PCIe·UB/封装互连/NoC）。 ──
const SVG_W = 600, SVG_H = 476, X0 = 118, X1 = 586, BUDGET = 26;   // 漏斗 L7→L1；L0 完整交互架构图由下方 CoreGroupPattern 面板承担
// ctx-tier centered funnel geometry: current glyph sits ON the center spine, siblings flank symmetrically.
// uniform abstract 2D glyphs per level (like the chip 2×2-die glyph) — small counts render EVERY member.
const CX_SPINE = (X0 + X1) / 2, CTX_GW = 20, CTX_GAP = 5, CTX_SLOT = CTX_GW + CTX_GAP;
// x-center of the k-th sibling glyph on a given side (+1 right / -1 left) of the spine-centered current glyph.
const ctxSibCx = (side: number, k: number) => CX_SPINE + side * k * CTX_SLOT;
// how many sibling glyphs fit per side before hitting the panel edge (so ≤~16 members show in full).
const ctxPerSide = Math.floor(((X1 - CX_SPINE) - CTX_GW / 2 - 4) / CTX_SLOT);
// Le 阶梯：0 全球 · 1 集群 · 2 服务池（ctx 上层上下文）· 3 Pod · 4 Host · 5 Chip。
interface Tier { Le: number; key: string; ctx?: boolean; y: number; h: number; maxW?: number; tag: string; label: string; col: string; ghosts?: string[] }
const TIERS: Tier[] = [
  { Le: 0, key: 'global',  ctx: true, y: 30, h: 16, maxW: 150, tag: 'L7', label: 'Global',       col: ENTITY_COLORS.global,  ghosts: ['Global A', 'Global C'] },
  { Le: 1, key: 'cluster', ctx: true, y: 66, h: 16, maxW: 150, tag: 'L6', label: 'Cluster',      col: ENTITY_COLORS.cluster, ghosts: ['Cluster A', 'Cluster C'] },
  { Le: 2, key: 'pool',    ctx: true, y: 102, h: 16, maxW: 150, tag: 'L5', label: 'Service Pool', col: ENTITY_COLORS.pool,    ghosts: ['Pool 1', 'Pool 3'] },
  { Le: 3, key: 'super',   y: 150, h: 22, maxW: 168, tag: 'L4', label: 'Pod',          col: ENTITY_COLORS.super },
  { Le: 4, key: 'node',    y: 232, h: 15, maxW: 52,  tag: 'L3', label: 'Host',         col: ENTITY_COLORS.node },
  { Le: 5, key: 'card',    y: 314, h: 17, maxW: 20,  tag: 'L2', label: 'Chip·NPU',     col: ENTITY_COLORS.card },
];
interface SubTier { key: string; lvl: Level; y: number; n: number; tag: string; label: string; cols?: number; cell?: number; gap?: number; seed?: number; col?: (i: number) => string }
const SUBTIERS: SubTier[] = [
  { key: 'die',  lvl: 'die',  y: 396, cols: 4, cell: 30, gap: 10, n: 4, seed: 131, tag: 'L1', label: 'Die (optional)', col: (i: number) => (i < 2 ? ENTITY_COLORS.computeDie : ENTITY_COLORS.ioDie) },
  { key: 'core', lvl: 'core', y: 470, n: 32, tag: 'L0', label: 'Core-Group' },   // 原生 CoreGroupMiniSvg 图元（非网格）
];

function Smartscape({ N, nBlades, focus, setFocus, metric, wlKind, step, dir, planeOn, playing, stats, dark, pm }: {
  N: number; nBlades: number; focus: Focus; setFocus: (f: Focus) => void;
  metric: Metric; wlKind: string; step: number; dir: Dir; planeOn: { ub: boolean; rdma: boolean; vpc: boolean }; playing: boolean; stats: Stats; dark: boolean;
  pm: ReturnType<typeof parallelMap>;
}) {
  const P = dark
    ? { ink: '#e6ebf2', ink2: '#9aa6b4', ink3: '#5f6b79', line: '#2a323d', pill: '#1b212b', pillBd: '#2a323d', die: 'rgba(9,13,20,0.55)' }
    : { ink: '#1c2433', ink2: '#5b6573', ink3: '#9099a8', line: '#d6dbe4', pill: '#eef1f6', pillBd: '#d2d8e2', die: 'rgba(255,255,255,0.55)' };
  const [ctxCur, setCtxCur] = useState<Record<string, number>>({});   // 上层每级当前选中的实体索引（可点兄弟切换）
  const total = (Le: number) => [1, 1, 1, 1, nBlades, N][Le];
  const metricOf = (Le: number, idx: number): number =>
    Le <= 3 ? stats.clusterMean : Le === 4 ? (stats.ndVals[idx] ?? 0) : cardMetric(idx, metric, wlKind, step);
  const aggOf = (Le: number) => (Le <= 3 ? stats.agg.cluster : Le === 4 ? stats.agg.node : stats.agg.card);
  const selLe = !focus ? -1 : focus.level === 'super' ? 3 : focus.level === 'node' ? 4 : 5;   // card/die/core → 5 (Chip 卡)
  const selIdx = selLe < 0 ? -1 : selLe === 3 ? 0 : selLe === 4 ? Math.floor(focus!.card / CPB) : focus!.card;
  const focusCard = focus && (focus.level === 'card' || focus.level === 'die' || focus.level === 'core') ? focus.card : null;
  const hasDrillFocus = !!focus && focus.level !== 'super';
  // structure = glyph + position; state = 红黄绿 (only when playing) — else hierarchy colour (同层级图)
  const fillOf = (Le: number, idx: number, base: string) => (playing ? loadColor(metricOf(Le, idx)) : base);

  // build per-tier shown lists + positions（漏斗下钻级只保留 Host(4)/Chip(5)；L4 Pod 与上三级一样是
  //   居中 ctx 行——本 Pod 高亮 + 3 兄弟 Pod，见 ctxRow；这里预置 pos[3] 供 Host→Pod 连线取点）
  const pos: Record<number, Record<number, { x: number; y: number }>> = {};
  const rows = TIERS.filter((t) => !t.ctx && t.Le !== 3).map((t) => {
    const sc = tierInScope(t.Le, focus, dir, N, nBlades);
    const full = sc === null;
    const baseList = full ? Array.from({ length: Math.min(total(t.Le), BUDGET) }, (_, i) => i) : sc;
    const shownIdx = baseList.slice(0, BUDGET);
    const inCount = full ? total(t.Le) : sc.length;
    const fold = inCount - shownIdx.length;
    // Reserve right-edge space for fold pill so it never overlaps the last glyph
    const pillW = fold > 0 ? Math.max(28, String(fold).length * 7 + 22) : 0;
    const availW = X1 - X0 - (fold > 0 ? pillW + 8 : 0);
    const slotW = availW / Math.max(1, shownIdx.length);
    pos[t.Le] = {};
    const shown = shownIdx.map((idx, i) => { const x = X0 + availW * (i + 0.5) / Math.max(1, shownIdx.length); pos[t.Le][idx] = { x, y: t.y }; return { idx, x }; });
    const foldX = fold > 0 ? X1 - pillW / 2 + 2 : null;
    return { t, shown, fold, foldX, inCount, slotW };
  });
  // Pod anchor: index 0 always at CX_SPINE; selected pod (cur) sits at its fixed slot position.
  // The chain line should connect FROM the selected pod's ripple, so store that position at key 0.
  { const podCur = Math.min(PODS_PER_POOL - 1, ctxCur['super'] ?? 0);
    const podCurX = podCur === 0 ? CX_SPINE : ctxSibCx(podCur % 2 === 1 ? 1 : -1, Math.ceil(podCur / 2));
    pos[3] = { 0: { x: podCurX, y: TIERS[3].y } }; }

  const parentOf = (Le: number, idx: number): { Le: number; idx: number } | null =>
    Le === 5 ? { Le: 4, idx: Math.floor(idx / CPB) } : Le === 4 ? { Le: 3, idx: 0 } : null;   // Chip→Host, Host→Pod

  const els: React.ReactNode[] = [];
  // connector language mirrors 平面视图「选中链路·层级图」(SelHierPanel): a solid SEL line +
  // connector dots (色环 + 白芯) at the junctions + 运行时沿线流动的白色彗星 (SMIL marching-ants).
  const tierH = (Le: number) => (TIERS.find((tt) => tt.Le === Le)?.h ?? 16);
  const cdot = (x: number, y: number, c: string, k: string, r = 2.4) => (
    <g key={k}><circle cx={x} cy={y} r={r} fill={c} /><circle cx={x} cy={y} r={r * 0.42} fill="#fff" /></g>
  );
  // 流动彗星：默认仅播放时出现；force=true → 选中链路也常显流动（选中即「联动」，与之前一致）。
  //   选中态彗星用 ACCENT 蓝，播放态用白，二者可区分。
  const cflow = (x1: number, y1: number, x2: number, y2: number, k: string, force = false) => ((playing || force) ? (
    <line key={k} x1={x1} y1={y1} x2={x2} y2={y2} stroke={playing ? '#fff' : ACCENT} strokeWidth={1.6} strokeLinecap="round" strokeDasharray="3 11" opacity={playing ? 0.82 : 0.9}>
      <animate attributeName="stroke-dashoffset" from="14" to="0" dur="0.6s" repeatCount="indefinite" />
    </line>
  ) : null);
  // selection highlight = 涟漪(ripple): a bold outline hugging the glyph + two phase-offset rounded-rect
  //   pulses that grow & fade outward (SVG port of 平面视图 PlaneView's canvas ripple). Replaces the old
  //   fill-box + blue centre dot. w/h = glyph box size; grows ~30% each side (→1.6×) over 1.25s.
  const ripple = (cx: number, cy: number, w: number, h: number, col: string, key: string) => {
    const x = cx - w / 2, y = cy - h / 2, rk = Math.min(w, h) * 0.3, gx = w * 0.3, gy = h * 0.3;
    return (
      <g key={key} style={{ pointerEvents: 'none' }}>
        <rect x={x} y={y} width={w} height={h} rx={rk} fill={col} fillOpacity={0.12} stroke={col} strokeWidth={1.8} />
        {[0, 0.625].map((d, i) => (
          <rect key={i} x={x} y={y} width={w} height={h} rx={rk} fill="none" stroke={col} strokeWidth={2} strokeOpacity={0.45}>
            <animate attributeName="x" values={`${x};${x - gx}`} dur="1.25s" begin={`${d}s`} repeatCount="indefinite" />
            <animate attributeName="y" values={`${y};${y - gy}`} dur="1.25s" begin={`${d}s`} repeatCount="indefinite" />
            <animate attributeName="width" values={`${w};${w + 2 * gx}`} dur="1.25s" begin={`${d}s`} repeatCount="indefinite" />
            <animate attributeName="height" values={`${h};${h + 2 * gy}`} dur="1.25s" begin={`${d}s`} repeatCount="indefinite" />
            <animate attributeName="stroke-opacity" values="0.45;0" dur="1.25s" begin={`${d}s`} repeatCount="indefinite" />
          </rect>
        ))}
      </g>
    );
  };
  // hw-native-sys whitepaper level icons: Global=globe · Cluster=rack · Service Pool=4 linked boxes ·
  //   Pod=UBL128 grid · Host=3 blocks · Chip·NPU=chip shell · Die=dashed 2-die (optional) · Core-Group=V/C/CPU. s = half-size.
  const levelIcon = (kind: string, cx: number, cy: number, s: number, col: string, op: number, key: string) => {
    const sw = Math.max(0.7, s * 0.14), rr = Math.min(2, s * 0.2);
    const box = (x: number, y: number, w: number, h: number, dash?: boolean) => <rect key={`${key}box`} x={cx + x} y={cy + y} width={w} height={h} rx={rr} fill="none" stroke={col} strokeOpacity={op} strokeWidth={sw} strokeDasharray={dash ? `${sw * 2.2} ${sw * 1.6}` : undefined} />;
    const cell = (x: number, y: number, w: number, h: number, k: string, fo = 0.42) => <rect key={`${key}f${k}`} x={cx + x} y={cy + y} width={w} height={h} rx={rr * 0.6} fill={col} fillOpacity={op * fo} stroke={col} strokeOpacity={op * 0.55} strokeWidth={sw * 0.6} />;
    const ln = (x1: number, y1: number, x2: number, y2: number, k: string, o = 0.75) => <line key={`${key}l${k}`} x1={cx + x1} y1={cy + y1} x2={cx + x2} y2={cy + y2} stroke={col} strokeOpacity={op * o} strokeWidth={sw * 0.85} />;
    const dt = (x: number, y: number, k: string) => <circle key={`${key}d${k}`} cx={cx + x} cy={cy + y} r={sw * 0.75} fill={col} fillOpacity={op} />;
    switch (kind) {
      case 'global': return <g key={key} style={{ pointerEvents: 'none' }}>
        <circle cx={cx} cy={cy} r={s} fill="none" stroke={col} strokeOpacity={op} strokeWidth={sw} />
        <ellipse cx={cx} cy={cy} rx={s * 0.42} ry={s} fill="none" stroke={col} strokeOpacity={op * 0.85} strokeWidth={sw * 0.8} />
        {ln(-s, 0, s, 0, 'eq', 0.85)}{ln(-s * 0.86, -s * 0.5, s * 0.86, -s * 0.5, 'la', 0.55)}{ln(-s * 0.86, s * 0.5, s * 0.86, s * 0.5, 'lb', 0.55)}
      </g>;
      case 'cluster': { const w = s * 1.5, h = s * 1.95, x = -w / 2, y = -h / 2;
        return <g key={key} style={{ pointerEvents: 'none' }}>{box(x, y, w, h)}
          {[0.28, 0.52, 0.76].map((f, i) => ln(x, y + h * f, x + w, y + h * f, `sh${i}`, 0.65))}
          {[0.14, 0.4, 0.64, 0.88].map((f, i) => dt(x + s * 0.3, y + h * f, `p${i}`))}
        </g>; }
      case 'pool': { const u = s * 0.66, o = s * 0.56;
        const cells: [number, number][] = [[-o, -o], [o, -o], [-o, o], [o, o]];
        return <g key={key} style={{ pointerEvents: 'none' }}>
          {ln(-o, -o, o, -o, 'ct', 0.65)}{ln(-o, o, o, o, 'cb', 0.65)}{ln(-o, -o, -o, o, 'cl', 0.65)}{ln(o, -o, o, o, 'cr', 0.65)}
          {cells.map(([dx, dy], i) => <rect key={`${key}c${i}`} x={cx + dx - u / 2} y={cy + dy - u / 2} width={u} height={u} rx={rr * 0.7} fill={col} fillOpacity={op * 0.3} stroke={col} strokeOpacity={op} strokeWidth={sw * 0.8} />)}
        </g>; }
      case 'pod': { const w = s * 1.95, h = s * 1.5, x = -w / 2, y = -h / 2, pad = s * 0.24, cols = 4, rws = 2;
        const cw = (w - pad * 2) / cols, ch = (h - pad * 2) / rws, out: React.ReactNode[] = [];
        for (let r = 0; r < rws; r++) for (let c = 0; c < cols; c++) out.push(cell(x + pad + c * cw + 0.3, y + pad + r * ch + 0.3, cw - 0.6, ch - 0.6, `${r}${c}`, 0.5));
        return <g key={key} style={{ pointerEvents: 'none' }}>{box(x, y, w, h)}{out}</g>; }
      case 'host': { const w = s * 1.95, h = s * 1.15, x = -w / 2, y = -h / 2, u = h * 0.52;
        return <g key={key} style={{ pointerEvents: 'none' }}>{box(x, y, w, h)}
          {[0, 1, 2].map((i) => cell(x + w * (0.15 + i * 0.28), -u / 2, u, u, `s${i}`, 0.5))}
        </g>; }
      case 'chip': { const w = s * 1.5, x = -w / 2;
        return <g key={key} style={{ pointerEvents: 'none' }}>{box(x, x, w, w)}
          {cell(x + w * 0.24, x + w * 0.24, w * 0.52, w * 0.52, 'k', 0.35)}
          {[0.32, 0.68].map((f, i) => ln(x - s * 0.26, x + w * f, x, x + w * f, `pl${i}`, 0.8))}
          {[0.32, 0.68].map((f, i) => ln(x + w, x + w * f, x + w + s * 0.26, x + w * f, `pr${i}`, 0.8))}
        </g>; }
      case 'die': { const w = s * 1.95, h = s * 1.35, x = -w / 2, y = -h / 2, u = h * 0.52;
        return <g key={key} style={{ pointerEvents: 'none' }}>{box(x, y, w, h, true)}
          {[0.24, 0.55].map((f, i) => cell(x + w * f, -u / 2, u, u, `s${i}`, 0.5))}
        </g>; }
      default: { const cs = ['#7c5cff', '#ef4444', '#f59e0b'], labels = ['V', 'C', 'CPU'];
        const bw = s * 0.7, gap = s * 0.2, tot = bw * 3 + gap * 2, x0 = -tot / 2;
        return <g key={key} style={{ pointerEvents: 'none' }}>
          {cs.map((cc, i) => <rect key={`${key}v${i}`} x={cx + x0 + i * (bw + gap)} y={cy - bw / 2} width={bw} height={bw} rx={rr * 0.6} fill={cc} fillOpacity={op} />)}
          {s >= 8 && cs.map((_, i) => <text key={`${key}t${i}`} x={cx + x0 + i * (bw + gap) + bw / 2} y={cy + bw * 0.3} fill="#fff" fontSize={bw * (labels[i].length > 1 ? 0.42 : 0.62)} fontWeight={700} textAnchor="middle">{labels[i]}</text>)}
        </g>; }
    }
  };
  // 0) ALWAYS-ON containment funnel — 层级间关系（overview 也画）：
  //    L7→L6→L5→L4 短连线（仅相邻两级当前成员之间 + L4 Pod → L3 Host 漏斗楔形）。
  {
    const podY = TIERS[3].y, podH = TIERS[3].h;
    // Ctx-tier containment connectors: diagonal from selected member of tier N to selected of tier N+1.
    // Uses same fixed-slot formula as ctxRow (e=0→CX_SPINE, odd→right, even→left).
    const ctxSelX = (key: string, tot: number) => {
      const e = Math.min(tot - 1, ctxCur[key] ?? 0);
      return e === 0 ? CX_SPINE : ctxSibCx(e % 2 === 1 ? 1 : -1, Math.ceil(e / 2));
    };
    const ctxPairs: [Tier, string, number, Tier, string, number][] = [
      [TIERS[0], 'global',  1,                TIERS[1], 'cluster', 4],
      [TIERS[1], 'cluster', 4,                TIERS[2], 'pool',    POOLS_PER_CLUSTER],
      [TIERS[2], 'pool',    POOLS_PER_CLUSTER, TIERS[3], 'super',  PODS_PER_POOL],
    ];
    ctxPairs.forEach(([ta, ka, na, tb, kb, nb], i) => {
      const xa = ctxSelX(ka, na), xb = ctxSelX(kb, nb);
      const y1 = ta.y + ta.h / 2, y2 = tb.y - tb.h / 2;
      if (y2 <= y1) return;
      els.push(<line key={`ctx-seg-${i}`} x1={xa} y1={y1} x2={xb} y2={y2} stroke={ACCENT} strokeWidth={1.6} strokeOpacity={0.6} />);
      els.push(cflow(xa, y1, xb, y2, `ctx-flow-${i}`, true));
      els.push(cdot(xa, ta.y, ACCENT, `ctx-dot-a-${i}`, 2.2));
      els.push(cdot(xb, tb.y, ACCENT, `ctx-dot-b-${i}`, 2.2));
    });
    const hostRow = rows.find((r) => r.t.Le === 4);
    if (hostRow && hostRow.shown.length) {
      const hy = TIERS[4].y, hh = TIERS[4].h, xs = hostRow.shown.map((s) => s.x);
      const left = Math.min(...xs) - 4, right = Math.max(...xs) + 4, apexY = podY + podH / 2, baseY = hy - hh / 2 - 2;
      els.push(<path key="ph-wedge" d={`M${CX_SPINE - 16} ${apexY} L${left} ${baseY} L${right} ${baseY} L${CX_SPINE + 16} ${apexY} Z`} fill={ACCENT} fillOpacity={0.07} />);
    }
    // L3 Host → L2 Chip 漏斗楔形（1 Host ⊃ 8 Chip）——续在 Pod→Host 之下
    const chipRow0 = rows.find((r) => r.t.Le === 5);
    if (hostRow && hostRow.shown.length && chipRow0 && chipRow0.shown.length) {
      const cy0 = TIERS[5].y, ch0 = TIERS[5].h, cxs = chipRow0.shown.map((s) => s.x);
      const hBaseY = TIERS[4].y + TIERS[4].h / 2, cTopY = cy0 - ch0 / 2 - 2;
      const cLeft = Math.min(...cxs) - 4, cRight = Math.max(...cxs) + 4;
      els.push(<path key="hc-wedge" d={`M${CX_SPINE - 14} ${hBaseY} L${cLeft} ${cTopY} L${cRight} ${cTopY} L${CX_SPINE + 14} ${hBaseY} Z`} fill={ACCENT} fillOpacity={0.06} />);
    }
  }
  // 1) selected containment chain — UNIFIED across levels: selecting any entity (Host / Chip / Die / Core)
  //    highlights its path up the ancestors (Chip → Host → Pod spine) drawn OVER the full overview — no
  //    collapse, same switch-select feel as the L4–L7 context rows.
  if (focus && focus.level !== 'super') {
    const chipIdx = focus.level === 'node' ? null : focus.card;
    const hostIdx = Math.floor(focus.card / CPB);
    const chain: { Le: number; idx: number }[] = [];
    if (chipIdx != null && pos[5]?.[chipIdx]) chain.push({ Le: 5, idx: chipIdx });   // skip any card scrolled past the BUDGET cap
    if (pos[4]?.[hostIdx]) chain.push({ Le: 4, idx: hostIdx });
    chain.forEach(({ Le, idx }) => {
      const me = pos[Le][idx], par = parentOf(Le, idx); if (!par) return;
      const pp = pos[par.Le]?.[par.idx]; if (!pp) return;
      const cTop = me.y - tierH(Le) / 2, pBot = pp.y + tierH(par.Le) / 2;
      els.push(<line key={`ch-${Le}-${idx}`} x1={me.x} y1={cTop} x2={pp.x} y2={pBot} stroke={ACCENT} strokeWidth={1.8} strokeOpacity={0.7} />);
      els.push(cflow(pp.x, pBot, me.x, cTop, `chf-${Le}-${idx}`, true));   // 选中即流动（联动）
      els.push(cdot(me.x, me.y, ACCENT, `chd-${Le}-${idx}`, 2.4));
      els.push(cdot(pp.x, pBot, ACCENT, `chp-${Le}-${idx}`, 2.2));
    });
    // Host 被选中 → 只向下画到「本 Host 的 8 张 Chip」（L3→L2 下行包含链，与上行同样流动联动）
    if (focus.level === 'node' && pos[4]?.[hostIdx]) {
      const hp = pos[4][hostIdx], hBot = hp.y + tierH(4) / 2;
      const kids = Object.entries(pos[5] ?? {}).filter(([ci]) => Math.floor(+ci / CPB) === hostIdx);
      if (kids.length) els.push(cdot(hp.x, hBot, ACCENT, 'hc-sel-anchor', 2.4));
      kids.forEach(([ci, cpos]) => {
        const cTop = cpos.y - tierH(5) / 2;
        els.push(<line key={`hc-sel-${ci}`} x1={hp.x} y1={hBot} x2={cpos.x} y2={cTop} stroke={ACCENT} strokeWidth={1.6} strokeOpacity={0.6} />);
        els.push(cflow(hp.x, hBot, cpos.x, cTop, `hc-self-${ci}`, true));
        els.push(cdot(cpos.x, cpos.y, ACCENT, `hc-seld-${ci}`, 2.2));
      });
    }
    // 2) UB plane mesh — same-level Host↔Host links among the selected host's shown neighbours (toggle)
    if (planeOn.ub) {
      const nr = rows.find((r) => r.t.Le === 4);
      if (nr) { const ny = nr.t.y; for (let i = 0; i < nr.shown.length - 1; i++) { const a = nr.shown[i], b = nr.shown[i + 1]; if (Math.min(Math.abs(a.idx - hostIdx), Math.abs(b.idx - hostIdx)) > 4) continue; const mx = (a.x + b.x) / 2; els.push(<path key={`ub-${i}`} d={`M${a.x} ${ny} Q ${mx} ${ny - 13} ${b.x} ${ny}`} fill="none" stroke={PLANES[0].color} strokeWidth={1} strokeOpacity={0.5} />); } }
    }
  }
  // 3) tier glyphs — every member is the level's abstract hw-native-sys icon (Host = 3-block, Chip·NPU =
  //    chip shell), same glyph as the context rows above; tinted by live load (fillOf) so utilization
  //    still reads through the hue. Selection = 涟漪 ripple.
  rows.forEach(({ t, shown, fold, foldX, slotW }) => {
    const maxW = t.maxW ?? 40;
    const kind = t.Le === 4 ? 'host' : 'chip';
    shown.forEach(({ idx, x }) => {
      const isSel = t.Le === selLe && idx === selIdx, col = fillOf(t.Le, idx, t.col);
      const strag = playing && t.Le === 5 && isStrag(idx, step);
      const cy = t.y, click = (e: React.MouseEvent) => { e.stopPropagation(); setFocus(isSel ? null : entityToFocus(t.Le, idx)); };
      const gsz = Math.max(9, Math.min(maxW, slotW * 0.82));   // glyph footprint
      const hs = gsz * (t.Le === 5 ? 0.5 : 0.46);              // levelIcon half-size (≈ box ⁄ 1.5–2)
      els.push(
        <g key={`g-${t.Le}-${idx}`} style={{ cursor: 'pointer' }} onClick={click}>
          {/* transparent hit target — the levelIcon itself is pointer-events:none, so without this the
              row would not be clickable (→ no selection, no 3D link). */}
          <rect x={x - gsz / 2 - 2} y={cy - gsz / 2 - 2} width={gsz + 4} height={gsz + 4} rx={gsz * 0.2} fill="transparent" />
          {isSel ? ripple(x, cy, gsz + 5, gsz + 5, t.col, `gr-${t.Le}-${idx}`)
            : strag ? <rect x={x - gsz / 2 - 3} y={cy - gsz / 2 - 3} width={gsz + 6} height={gsz + 6} rx={gsz * 0.34} fill="none" stroke="#b07bff" strokeWidth={1.6} /> : null}
          {levelIcon(kind, x, cy, hs, col, 1, `gi-${t.Le}-${idx}`)}
        </g>,
      );
    });
    if (fold > 0 && foldX != null) {
      const w = Math.max(28, String(fold).length * 7 + 22);
      els.push(
        <g key={`f-${t.Le}`}>
          <rect x={foldX - w / 2} y={t.y - 9} width={w} height={18} rx={9} fill={P.pill} stroke={P.pillBd} />
          <text x={foldX} y={t.y + 4} fill={P.ink2} fontSize={10} textAnchor="middle">{`+${fold}`}</text>
        </g>,
      );
    }
  });
  // 4) tier labels (gutter): Lx · name · total · p50/red — no separate icon (the row members ARE the icon now)
  rows.forEach(({ t }) => {
    const a = aggOf(t.Le);
    els.push(
      <g key={`l-${t.Le}`}>
        {t.tag && <text x={12} y={t.y - 5} fill={t.col} fontSize={8} fontWeight={700}>{t.tag}</text>}
        <text x={12} y={t.y + (t.tag ? 5 : 3)} fill={P.ink} fontSize={10} fontWeight={700}>{t.label}</text>
        <text x={12} y={t.y + (t.tag ? 15 : 13)} fill={P.ink3} fontSize={8}>{`${total(t.Le).toLocaleString()} · p50 ${Math.round(a.p50 * 100)}%`}</text>
      </g>,
    );
  });
  // 5) divider + card-internal sub-tiers — intra-chip links only when drilled into Host/Chip.
  const repCard = focusCard != null ? focusCard : focus ? scopeRange(focus, N)[0] : 0;
  const repReal = focusCard != null;   // true = a real card is selected (else representative card)
  const DIE = SUBTIERS[0];
  els.push(<line key="div" x1={8} y1={352} x2={592} y2={352} stroke={P.line} strokeDasharray="2 4" />);
  els.push(<text key="divt" x={300} y={347} fill={P.ink3} fontSize={9} textAnchor="middle">
    {hasDrillFocus ? `—— On-chip · ${repReal ? 'card' : 'rep. card'} r${repCard} ——` : '—— On-chip ——'}
  </text>);
  // containment connectors — 代表卡 → 2 计算 Die → Core-Group 图元，solid SEL line + connector dots + 流动彗星（封装互连 / NoC）.
  const dieTopY = DIE.y - 6, dieBotY = DIE.y - 6 + DIE.cell!;
  const coreGX = 120, coreAnchorX = coreGX + 34, dieCxArr = [135, 175];
  const rpCardX = pos[5]?.[repCard]?.x ?? CX_SPINE, cardBotY = 322;
  // L2 Chip → L1 计算 Die：代表卡连线（overview 淡显 · focus 加重 + 彗星）——「一张卡 ⊃ 2 计算 + 2 IO Die」
  dieCxArr.forEach((dx, di) => {
    els.push(<line key={`cc-die-${di}`} x1={rpCardX} y1={cardBotY} x2={dx} y2={dieTopY} stroke={ACCENT} strokeWidth={1.3} strokeOpacity={hasDrillFocus ? 0.55 : 0.3} />);
    if (hasDrillFocus) els.push(cflow(rpCardX, cardBotY, dx, dieTopY, `ccf-die-${di}`));
    els.push(cdot(dx, dieTopY, ACCENT, `ccd-die-${di}`, 2));
  });
  els.push(cdot(rpCardX, cardBotY, ACCENT, 'ccd-card', hasDrillFocus ? 2.4 : 2));
  // L1 计算 Die → L0：两块计算 Die 向下汇聚，箭头指向下方「L0 完整存储架构」面板（NoC · 1 计算 Die ⊃ ~16 Core-Group）
  const l0PtrY = SVG_H - 16;
  dieCxArr.forEach((dx, di) => {
    els.push(<line key={`cd-core-${di}`} x1={dx} y1={dieBotY} x2={coreAnchorX} y2={l0PtrY} stroke={ACCENT} strokeWidth={1.3} strokeOpacity={hasDrillFocus ? 0.55 : 0.36} />);
    if (hasDrillFocus) els.push(cflow(dx, dieBotY, coreAnchorX, l0PtrY, `cdf-core-${di}`));
    els.push(cdot(dx, dieBotY, ACCENT, `cdd-die-${di}`, 2));
  });
  els.push(<path key="l0-arrow" d={`M${coreAnchorX - 4} ${l0PtrY - 6} L${coreAnchorX} ${l0PtrY} L${coreAnchorX + 4} ${l0PtrY - 6}`} fill="none" stroke={ACCENT} strokeWidth={1.4} strokeLinejoin="round" />);
  els.push(<text key="l1l0-note" x={coreAnchorX + 12} y={l0PtrY - 2} fill={P.ink3} fontSize={8.5}>↓ L0 Core-Group (panel below)</text>);
  // 5a) L1 Die 子层（网格）
  {
    const st = DIE;
    els.push(<text key="slt-die" x={12} y={st.y - 5} fill={ENTITY_COLORS.computeDie} fontSize={8} fontWeight={700}>{st.tag}</text>);
    els.push(<text key="sl-die" x={12} y={st.y + 5} fill={P.ink} fontSize={10} fontWeight={700}>{st.label}</text>);
    els.push(<text key="scnt-die" x={12} y={st.y + 15} fill={P.ink3} fontSize={8}>{`×${st.n}`}</text>);
    for (let i = 0; i < st.n; i++) {
      const cx = 120 + (i % st.cols!) * (st.cell! + st.gap!), cy = st.y - 6 + Math.floor(i / st.cols!) * (st.cell! + st.gap!);
      const isSel = repReal && focus?.die === i;
      const fill = i >= 2 || !playing ? st.col!(i) : loadColor(Math.max(0, Math.min(1, nodeLoad(repCard * st.seed! + i, wlKind))));
      els.push(
        <rect key={`s-die-${i}`} x={cx} y={cy} width={st.cell!} height={st.cell!} rx={Math.min(3, st.cell! * 0.18)} fill={fill} style={{ cursor: 'pointer' }}
          stroke={isSel ? ENTITY_COLORS.computeDie : 'none'} strokeWidth={isSel ? 2.2 : 0}
          onClick={(e) => { e.stopPropagation(); setFocus({ level: 'die', card: repCard, die: i }); }} />,
      );
    }
    els.push(<text key="die-cap" x={120 + 4 * (st.cell! + st.gap!) + 6} y={st.y + st.cell! / 2} fill={P.ink3} fontSize={8} dominantBaseline="central">compute·IO</text>);
  }
  // L0 Core-Group（最深层级）由下方独立的 CoreGroupPattern 面板完整渲染 memory-architecture 图（可缩放/平移）。
  // 级间互联徽标（DCN/Scale-Out/Intra-Pool/Scale-Up/PCIe·UB/Package/NoC）已移除——竖脊本身即表达包含，
  // 徽标文字标签在中心列显得杂乱；织物名称保留在各行副标题/悬停里即可。
  // 0) upper context (L7 Global / L6 Cluster / L5 Service Pool / L4 Pod) rendered as their own entities.
  // ctx-hint removed (unnecessary gray header)
  const ctxLabel = (t: Tier, sub: string) => els.push(
    <g key={`ctxl-${t.key}`}>
      <text x={12} y={t.y - 3} fill={t.col} fontSize={8} fontWeight={700}>{t.tag}</text>
      <text x={12} y={t.y + 6} fill={P.ink} fontSize={10} fontWeight={700}>{t.label}</text>
      <text x={12} y={t.y + 15} fill={P.ink3} fontSize={8}>{sub}</text>
    </g>,
  );
  // one context row: the SELECTED member sits on the spine (涟漪 ripple), siblings fan out and are CLICKABLE
  //   to switch which one is current. Small counts show every member (else +N fold).
  const ctxRow = (t: Tier, kind: string, total: number, subFn: (c: number) => string, dashed: boolean) => {
    const cur = Math.min(total - 1, ctxCur[t.key] ?? 0);
    ctxLabel(t, subFn(cur));
    const h = t.h, gw = CTX_GW;
    // each member is drawn as the LEVEL'S abstract hw-native-sys icon, not a plain pill;
    //   current = full-opacity + 涟漪 ripple, siblings = faded (dashed tiers extra-faint).
    const iconS = Math.min(gw, h) * 0.46;
    const drawG = (e: number, cx: number) => {
      const isCur = e === cur;
      els.push(
        <g key={`ctx-${t.key}-${e}`} style={{ cursor: 'pointer' }}
          onClick={(ev) => { ev.stopPropagation(); setCtxCur((c) => ({ ...c, [t.key]: e })); if (t.Le === 3) setFocus({ level: 'super', card: 0 }); }}>
          <rect x={cx - gw / 2} y={t.y - h / 2} width={gw} height={h} rx={h * 0.4} fill="transparent" />
          {isCur && ripple(cx, t.y, gw + 3, h + 3, t.col, `ctxr-${t.key}-${e}`)}
          {levelIcon(kind, cx, t.y, iconS, t.col, isCur ? 1 : dashed ? 0.3 : 0.55, `ctxg-${t.key}-${e}`)}
        </g>,
      );
    };
    // FIXED positions (index-based, NOT selection-based): index 0 sits on the spine; rest fan out at
    // fixed slots (1→R, 2→L, 3→R…). Clicking a sibling ripples it IN PLACE, no repositioning.
    drawG(0, CX_SPINE);
    let pr = 0, pl = 0, shown = 0;
    for (let e = 1; e < total && shown < ctxPerSide * 2; e++) {
      const right = e % 2 === 1;
      drawG(e, ctxSibCx(right ? 1 : -1, right ? ++pr : ++pl));
      shown++;
    }
    const foldN = (total - 1) - shown;
    if (foldN > 0) {
      const fx = ctxSibCx(1, pr) + gw / 2 + CTX_GAP, fw = Math.max(28, String(foldN).length * 7 + 18);
      els.push(
        <g key={`ctx-${t.key}-fold`}>
          <rect x={fx} y={t.y - 9} width={fw} height={18} rx={9} fill={P.pill} stroke={P.pillBd} />
          <text x={fx + fw / 2} y={t.y + 4} fill={P.ink2} fontSize={10} textAnchor="middle">{`+${foldN}`}</text>
        </g>,
      );
    }
  };
  // Each row = the level's OWN entities (Pool and Pod are separate rows). Small levels show every member.
  //   Click a sibling to switch which Cluster / Pool / Pod is current.
  ctxRow(TIERS[0], 'global', 1, () => `via DCN`, false);
  ctxRow(TIERS[1], 'cluster', 4, (c) => (c === 0 ? `1 = ${PODS_PER_CLUSTER} Pod` : `Cluster ${c}`), true);
  ctxRow(TIERS[2], 'pool', POOLS_PER_CLUSTER, (c) => `${c === 0 ? 'this Pool' : 'Pool ' + (c + 1)} · ${POOLS_PER_CLUSTER}/Cluster`, false);
  ctxRow(TIERS[3], 'pod', PODS_PER_POOL, (c) => `${c === 0 ? 'this Pod' : 'Pod ' + (c + 1)} · ${PODS_PER_POOL}/Pool`, false);

  // C) 并行关系落到真实层级对象（仅 card 焦点）：TP/PP 描到真实 Host pill · DP 弧到兄弟 Pod · EP 按 epScope。
  if (focus && focus.level === 'card') {
    const k = focus.card, bk = Math.floor(k / CPB);
    const nodeRow = rows.find((r) => r.t.Le === 4);
    const hostW = nodeRow ? Math.max(11, Math.min(TIERS[4].maxW ?? 52, nodeRow.slotW * 0.82)) : 40;
    const hY = TIERS[4].y, hH = TIERS[4].h;
    const badge = (key: string, x: number, y: number, txt: string, col: string, anchor: 'middle' | 'end' = 'middle') => {
      const w = txt.length * 6.6 + 8, lx = anchor === 'end' ? x - w : x - w / 2;
      return (
        <g key={key} style={{ pointerEvents: 'none' }}>
          <rect x={lx} y={y - 6} width={w} height={12} rx={6} fill={col} />
          <text x={lx + w / 2} y={y + 0.5} fill={ink(col)} fontSize={7.5} fontWeight={700} textAnchor="middle" dominantBaseline="central">{txt}</text>
        </g>
      );
    };
    const hostStroke = (key: string, cx: number, col: string, inset: number) => (
      <rect key={key} x={cx - hostW / 2 - inset} y={hY - hH / 2 - inset} width={hostW + inset * 2} height={hH + inset * 2} rx={(hH + inset * 2) * 0.4} fill="none" stroke={col} strokeWidth={1.8} />
    );
    // 1) TP — 焦点 Chip 所在 Host pill 描 tp 色 + 角标 TP×8（TP 组 = 本 Host 8 卡）
    const hp = pos[4]?.[bk];
    if (hp) {
      els.push(hostStroke('c-tp-str', hp.x, PARALLEL_COLORS.tp, 3));
      els.push(badge('c-tp-b', hp.x, hY - hH / 2 - 9, `TP×${pm.tp}`, PARALLEL_COLORS.tp));
    }
    // 2) PP — peersOf(k,'pp') 换算 host 下标 b=⌊peer/8⌋；段内 Host 加 pp 描边 + PP级{stage}；段外合并成行尾角标
    let ppOff = 0;
    pm.peersOf(k, 'pp').forEach((peer) => {
      const pb = Math.floor(peer / CPB); if (pb === bk) return;   // 焦点 Host 已由 TP 标注
      const php = pos[4]?.[pb];
      if (php) {
        els.push(hostStroke(`c-pp-str-${pb}`, php.x, PARALLEL_COLORS.pp, 5));
        els.push(badge(`c-pp-b-${pb}`, php.x, hY + hH / 2 + 9, `PP级${pm.groupOf(peer, 'pp')}`, PARALLEL_COLORS.pp));
      } else ppOff++;
    });
    if (ppOff > 0) els.push(badge('c-pp-off', X1, hY - hH / 2 - 9, `PP ×${pm.pp} 级（跨 Host）`, PARALLEL_COLORS.pp, 'end'));
    // 3) DP — 从焦点 Chip 画虚线弧到 L4 行的兄弟 Pod（DP 副本 = 跨 Pod）+ 副本角标
    const cp = pos[5]?.[k];
    if (cp) {
      const sx = cp.x, sy = TIERS[5].y - 12, tx = ctxSibCx(1, 1), ty = TIERS[3].y + TIERS[3].h / 2;
      const ctlX = Math.max(sx, tx) + 70, ctlY = (sy + ty) / 2;
      els.push(<path key="c-dp-arc" d={`M${sx} ${sy} Q ${ctlX} ${ctlY} ${tx} ${ty}`} fill="none" stroke={PARALLEL_COLORS.dp} strokeWidth={1.5} strokeDasharray="4 4" strokeOpacity={0.9} />);
      els.push(badge('c-dp-b', (sx + tx) / 2 + 24, 180, `DP ×${pm.dp} 副本（本 Pod 内平铺·逻辑上跨 Pod）`, PARALLEL_COLORS.dp));
      // 4) EP — replica 作用域：在 DP 弧旁标相邻副本 A2A
      if (pm.epScope === 'replica') els.push(badge('c-ep-rep', (sx + tx) / 2 + 24, 194, `EP×${pm.ep}（相邻副本 A2A）`, PARALLEL_COLORS.ep));
    }
    // 4) EP — node 作用域：焦点 Host pill 内描边 + 节点内路由角标
    if (pm.epScope === 'node' && hp) {
      els.push(hostStroke('c-ep-str', hp.x, PARALLEL_COLORS.ep, 6));
      els.push(badge('c-ep-node', hp.x, hY + hH / 2 + 9, `EP×${pm.ep}·节点内路由`, PARALLEL_COLORS.ep));
    }
    // 5) 图例：TP■ PP■ DP■ EP■（真实成员来自 parallelMap）
    let lx = 300;
    (['tp', 'pp', 'dp', 'ep'] as const).forEach((d) => {
      const c = PARALLEL_COLORS[d];
      els.push(<rect key={`c-lg-${d}`} x={lx} y={8} width={8} height={8} rx={2} fill={c} />);
      els.push(<text key={`c-lgt-${d}`} x={lx + 11} y={14} fill={P.ink2} fontSize={8.5} fontWeight={700}>{d.toUpperCase()}</text>);
      lx += 40;
    });
    els.push(<text key="c-lg-note" x={lx + 2} y={14} fill={P.ink3} fontSize={8}>（真实成员来自 parallelMap）</text>);
  }

  return (
    // funnel L7→L1 fills the pane WIDTH (aspect-locked, left-aligned) so its gutter lines up with the
    // L0 panel's gutter below; SVG_H is small so the height stays reasonable.
    <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} preserveAspectRatio="xMinYMin meet" width="100%" style={{ display: 'block', width: '100%', height: 'auto' }}>
      <rect x={0} y={0} width={SVG_W} height={SVG_H} fill="transparent" onClick={() => setFocus(null)} />
      {els}
    </svg>
  );
}

export function ConsoleView({ gen, dark, sync, lens: lensP, setLens: setLensP, dir: dirP, setDir: setDirP }: {
  gen: Gen; dark: boolean; sync?: ViewSync;
  lens?: Lens; setLens?: (l: Lens) => void; dir?: Dir; setDir?: (d: Dir) => void;
}) {
  const visualProfile = useContext(SceneVisualProfileContext);
  const workbenchProfile = visualProfile === 'opRankTime';
  const surf = sceneSurface(dark, visualProfile);
  const spec = GENERATIONS[gen];
  const N = spec.totalNpus;
  const nBlades = Math.ceil(N / CPB), nCabs = Math.ceil(nBlades / BPC);

  // 工况/时间/播放 come from the cross-view sync when present → 运行状态 ⇄ 工作台 stay linked
  const [workloadL] = useState<Workload>('decode');
  const workload = sync?.workload ?? workloadL;
  const [metricL] = useState<Metric>('util');
  const metric = (sync?.metric ?? metricL) as Metric;
  const [dirL, setDirL] = useState<Dir>('all');
  const dir = dirP ?? dirL; const setDir = setDirP ?? setDirL;
  const [lensL, setLensL] = useState<Lens>('heat');
  const lens = lensP ?? lensL; const setLens = setLensP ?? setLensL;
  const [partDim, setPartDim] = useState<Exclude<PartitionDim, 'none'>>('tp');
  const [planeOnL] = useState({ ub: true, rdma: true, vpc: false });
  const planeOn = sync?.planeOn ?? planeOnL;
  const [focus, setFocus] = useState<Focus>(null);
  const [scopeB, setScopeB] = useState<{ cx: number; cy: number; cz: number; r: number } | null>(null);
  const [hover, setHover] = useState<string | null>(null);
  // ── 统一驾驶舱增量（仅 workbench profile 启用）：右侧时空折叠魔方 · 左侧诊断抽屉 ──
  const [cubeMode, setCubeMode] = useState(false);          // 右画布：物理机柜(false) ⇄ 时空折叠魔方(true)
  const [cubeLayout, setCubeLayout] = useState<LayoutView>('standard');
  const [leftLower, setLeftLower] = useState<'core' | 'monitor' | 'comm'>('core');   // 左下抽屉：L0 核组 / 双模监控 / 通信深潜
  const [monMode, setMonMode] = useState<'heat' | 'replay'>('heat');
  const [stepL, setStepL] = useState(0);
  const step = sync?.step ?? stepL;
  const setStep = sync?.setStep ?? setStepL;
  const [playingL] = useState(false);
  const playing = sync?.playing ?? playingL;
  const [infoOpen, setInfoOpen] = useState(false);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const controlsRef = useRef<any>(null);
  const splitRef = useRef<HTMLDivElement | null>(null);
  const wlKind = WL[workload].kind;

  useEffect(() => { setFocus(null); setScopeB(null); }, [gen]);
  useEffect(() => {
    if (!workbenchProfile || !splitRef.current) return;

    const helper = window.PtoWorkbenchShell;
    if (!helper?.initResizablePanes) return;

    const root = splitRef.current;
    const leftPane = root.querySelector<HTMLElement>('[data-pane="smartscape"]');
    const rightPane = root.querySelector<HTMLElement>('[data-pane="panorama"]');
    if (!leftPane || !rightPane) return;

    let frame = 0;
    const refreshCanvasLayout = () => {
      if (frame) window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(() => {
        frame = 0;
        window.dispatchEvent(new Event('resize'));
        controlsRef.current?.update?.();
      });
    };

    const split = helper.initResizablePanes({
      root,
      panes: [leftPane, rightPane],
      direction: 'horizontal',
      sizes: [38, 62],
      minSize: [300, 420],
      gutterSize: 10,
      storageKey: 'hpc-topology-console-split-v1',
      gutterLabel: '调整平面视图和阵列全景宽度',
      onResize: refreshCanvasLayout,
    });

    refreshCanvasLayout();

    return () => {
      if (frame) window.cancelAnimationFrame(frame);
      split.destroy();
    };
  }, [workbenchProfile]);
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setStep((s) => (s + 1) % (STEP_MAX + 1)), 650);
    return () => clearInterval(id);
  }, [playing]);

  const stats = useMemo<Stats>(() => computeStats(N, nCabs, nBlades, metric, wlKind, step), [N, nCabs, nBlades, metric, wlKind, step]);

  // focused-entity auxiliary metrics (exact over the scope range; ≤64 cards unless whole)
  const rail = useMemo(() => {
    const [lo, hi] = scopeRange(focus, N), n = hi - lo;
    if (n > PER_CAB) return null;
    const mean = (m: Metric) => { let s = 0; for (let k = lo; k < hi; k++) s += cardMetric(k, m, wlKind, step); return n ? s / n : 0; };
    return { util: mean('util'), strag: mean('strag'), fault: mean('fault'), count: n };
  }, [focus, N, wlKind, step]);

  const problem = useMemo(() => {
    if (step < EVT_LO || step > EVT_HI) return null;
    let strag = 0; for (let k = EVT_CAB * PER_CAB; k < (EVT_CAB + 1) * PER_CAB && k < N; k++) if (isStrag(k, step)) strag++;
    const redR = stats.kpi.hot / N;
    return { root: EVT_CAB, title: `机柜（物理分组）C${EVT_CAB} 过热`, chain: `液冷异常 → ${strag} 卡掉队(straggler) → DP 梯度 AllReduce 阻塞`, impact: `影响 ${Math.min(N, PER_CAB)} 卡 · step 延迟 +${Math.round(redR * 420 + 22)}%` };
  }, [step, N, stats.kpi.hot]);

  // 面包屑：全球 › 集群 › 服务池 › Pod › Host B{n} › Chip r{k} › Die › Core-Group（上三级为上下文，点击回整 Pod）
  const crumbs = useMemo(() => {
    const out: { lvl: string; label: string; card: number }[] = [
      { lvl: 'global', label: '全球', card: 0 }, { lvl: 'cluster', label: '集群', card: 0 },
      { lvl: 'pool', label: '服务池', card: 0 }, { lvl: 'super', label: 'Pod', card: 0 },
    ];
    if (focus && focus.level !== 'super') {
      const b = Math.floor(focus.card / CPB); out.push({ lvl: 'node', label: `Host B${b}`, card: b * CPB });
      if (['card', 'die', 'core'].includes(focus.level)) out.push({ lvl: 'card', label: `Chip r${focus.card}`, card: focus.card });
      if (focus.level === 'die' || focus.level === 'core') out.push({ lvl: 'die', label: 'Die', card: focus.card });
      if (focus.level === 'core') out.push({ lvl: 'core', label: 'Core-Group', card: focus.card });
    }
    return out;
  }, [focus]);

  // panorama config (lens → array presentation); memoised so playback ticks don't churn the 8K recolor
  const panoStatus = playing && (lens === 'heat' || lens === 'flow');
  const panoPeers = playing && lens === 'flow';
  const panoPlanes = lens === 'phys';
  const panoPart: PartitionDim = lens === 'domain' ? partDim : 'none';
  const panoPhase = useMemo<RunPhase | null>(() => (playing && (lens === 'heat' || lens === 'flow')
    ? { id: 'wl', name: WL[workload].label, kind: wlKind, color: wlKind === 'comm' ? '#ff4b7b' : '#22d3ee', collective: lens === 'flow' ? 'ring' : undefined, note: '' }
    : null), [playing, lens, workload, wlKind]);
  const runMode: RunMode = workload === 'pretrain' ? 'train' : 'infer';
  const reach = Math.sqrt(N) * 1.3 + 12;
  const panoSel = useMemo(() => focusToSel(focus), [focus]);

  // ── 魔方联动：层级图 focus.level → 聚合粒度；focus 为具体卡 → 选中 rank；lens=通信域 → 策略着色 ──
  const cubeAgg: LevelKey = focus ? (FOCUS_AGG[focus.level] ?? 'card') : 'card';
  const cubeSel = focus && (focus.level === 'card' || focus.level === 'die' || focus.level === 'core') ? focus.card : null;
  const cubeStrat: PartitionDim = lens === 'domain' ? partDim : 'none';
  const noop = () => { /* CubeView 嵌入时不回写这些通道 */ };
  const cubeSync: ViewSync = {
    workload, step, playing: false, metric, planeOn,
    setWorkload: sync?.setWorkload ?? noop, setStep, setPlaying: sync?.setPlaying ?? noop,
    setMetric: sync?.setMetric ?? noop, setPlaneOn: sync?.setPlaneOn ?? noop,
  };
  const monBaseHost = cubeSel != null ? Math.floor(Math.floor(cubeSel / CPB) / 8) * 8 : 0;

  // parallel groups from the SINGLE SOURCE OF TRUTH — degrees/membership agree with 平面·3D·运行状态
  const pm = useMemo(() => parallelMap(workload, N), [workload, N]);
  // 并行关系（rank↔rank）：每维给出 集合通信形态 + 度数 + 真实对端数（peersOf 真值），把 TP/SP/EP/PP/DP 的「联系」讲清楚
  const COLL: Record<ParDim, string> = { tp: 'AllReduce', sp: 'AllGather+RS', pp: 'P2P send/recv', dp: 'Ring-AllReduce', ep: '层级化 All-to-All' };
  const groups: { d: ParDim; label: string; c: string; coll: string; pat: 'ring' | 'a2a' | 'p2p'; peers: number; deg: number }[] = focus && rail ? (['tp', 'sp', 'pp', 'dp', 'ep'] as ParDim[]).map((d) => {
    const k = focus.card, grp = pm.groupOf(k, d), deg = pm.groupCount(d);
    const label = d === 'sp' && pm.sp <= 1 ? '与 TP 同域' : d === 'tp' ? `切片 ${grp}` : d === 'pp' ? `级 ${grp}/${pm.pp}` : d === 'dp' ? `副本 ${grp}` : `组 ${grp}/${pm.ep}`;
    return { d, label, c: d === 'sp' ? PARALLEL_COLORS_SP : PARALLEL_COLORS[d as Exclude<PartitionDim, 'none'>], coll: COLL[d], pat: pm.collectiveOf(d), peers: pm.peersOf(k, d, 1 << 20).length, deg };
  }) : [];
  const phys = focus && rail ? LEVEL_PHYS[focus.level] : null;
  const card: React.CSSProperties = { background: 'var(--panel)', border: '1px solid var(--bd)', borderRadius: 11, boxShadow: 'var(--shadow-sm)', backdropFilter: 'blur(12px)', WebkitBackdropFilter: 'blur(12px)' };

  // L0 核组图（层级图之下的最深层级面板）——抽出以便经典视图与工作台抽屉复用同一图元。
  const l0Panel = (
    <div style={{ flex: '1 1 0', minHeight: 190, display: 'flex', borderTop: '1px dashed var(--bd)' }}>
      <div style={{ width: '19.7%', minWidth: 80, flexShrink: 0, paddingLeft: '2%', paddingTop: 6, paddingRight: 4, display: 'flex', flexDirection: 'column', gap: 2 }}>
        <span style={{ fontSize: 9, fontWeight: 700, color: '#36e0c4' }}>L0</span>
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--tx)', lineHeight: 1.1 }}>Core-Group</span>
        <span style={{ display: 'inline-flex', gap: 3, marginTop: 3 }}>
          {([['V', '#7c5cff'], ['C', '#ef4444'], ['CPU', '#f59e0b']] as [string, string][]).map(([l, c]) => (
            <span key={l} style={{ fontSize: 8, fontWeight: 700, color: '#fff', background: c, borderRadius: 3, padding: '1px 4px' }}>{l}</span>
          ))}
        </span>
        <span style={{ fontSize: 8.5, color: 'var(--tx3)', lineHeight: 1.35, marginTop: 4 }}>×32 / card · GM/L2 + AIV/AIC</span>
        <span style={{ fontSize: 8, color: 'var(--tx3)', lineHeight: 1.35, marginTop: 'auto' }}>fixed scale · reads as one piece with the funnel</span>
      </div>
      <div style={{ flex: 1, minHeight: 0, position: 'relative' }}>
        <CoreGroupPattern
          phaseKind={playing ? (['load', 'compute', 'comm', 'store'] as const)[step % 4] : undefined}
          load={playing ? Math.max(0.15, Math.min(1, stats.kpi.util)) : 0.5}
          zoom={0.42} detail align="left" height="100%" interactive={false}
        />
      </div>
    </div>
  );
  const shellStyle: React.CSSProperties = workbenchProfile
    ? { position: 'relative', zIndex: 11, flex: '1 1 auto', minHeight: 0, display: 'flex', flexDirection: 'column', background: 'transparent', color: 'var(--tx)' }
    : { position: 'absolute', inset: 0, zIndex: 11, display: 'flex', flexDirection: 'column', background: 'var(--bg)', color: 'var(--tx)' };

  return (
    <div className={workbenchProfile ? 'hpc-console-shell hpc-console-shell--workbench' : 'hpc-console-shell'} style={shellStyle}>
      {/* ── toolbar: 工况 / 指标 / 方向 / 镜头 (+切分) / 平面 · breadcrumb · KPI ── */}
      {workbenchProfile ? (
        <div className="hpc-console-toolbar hpc-console-toolbar--compact">
          <div className="hpc-console-crumbs">
            {crumbs.map((c, i) => (
              <span key={i} className="hpc-console-crumb">
                {i > 0 && <span className="hpc-console-crumb-sep">/</span>}
                <span onClick={() => setFocus(['global', 'cluster', 'pool', 'super'].includes(c.lvl) ? null : { level: c.lvl as Level, card: c.card })}>{c.label}</span>
              </span>
            ))}
          </div>
          <div className="hpc-console-kpis">
            {([
              [`${Math.round(stats.kpi.util * 100)}%`, `集群${M_LABEL.util}`, 'var(--tx)'],
              [`${metric === 'fault' ? stats.kpi.faultDom : stats.kpi.hot}`, metric === 'fault' ? '故障域' : '热点卡', loadColor(0.9)],
              [`${stats.kpi.strag}`, '掉队卡', PARALLEL_COLORS.ep],
            ] as [string, string, string][]).map(([v, l, c], i) => (
              <div key={i} className="hpc-console-kpi">
                <div style={{ color: c, ...TNUM }}>{v}</div>
                <span>{l}</span>
              </div>
            ))}
          </div>
        </div>
      ) : (
      <div style={{ display: 'flex', alignItems: 'center', gap: 9, padding: '6px 12px', borderBottom: '1px solid var(--bd)', flexWrap: 'wrap', background: 'var(--panel-solid)' }}>
        <span style={GLAB}>方向</span>
        <div style={{ display: 'flex', gap: 3 }}>
          {([['all', '全链'], ['up', '上游'], ['down', '下游']] as [Dir, string][]).map(([d, l]) => (<button key={d} onClick={() => setDir(d)} style={{ ...btnBase, ...navBtn(dir === d) }}>{l}</button>))}
        </div>
        <span style={GLAB}>镜头</span>
        <div style={{ display: 'flex', gap: 3 }}>
          {(Object.keys(LENS_LABEL) as Lens[]).map((l) => (<button key={l} onClick={() => setLens(l)} style={{ ...btnBase, ...(lens === l ? { border: '1px solid #5a3a86', background: '#5a3a86', color: '#fff', fontWeight: 600 } : SECONDARY) }}>{LENS_LABEL[l]}</button>))}
        </div>
        {lens === 'domain' && (
          <div style={{ display: 'flex', gap: 3, alignItems: 'center' }}>
            <span style={GLAB}>切分</span>
            {(['tp', 'pp', 'dp', 'ep'] as Exclude<PartitionDim, 'none'>[]).map((d) => (
              <button key={d} onClick={() => setPartDim(d)} title={PARTITION_META[d].label} style={{ ...btnBase, display: 'inline-flex', alignItems: 'center', gap: 5, ...toggleBtn(partDim === d, PARALLEL_COLORS[d]) }}>
                <span style={{ width: 8, height: 8, borderRadius: 2, background: partDim === d ? ink(PARALLEL_COLORS[d]) : PARALLEL_COLORS[d] }} />{d.toUpperCase()}
              </button>
            ))}
          </div>
        )}
        {/* breadcrumb */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, color: 'var(--tx2)', flex: 1, minWidth: 60, overflow: 'hidden' }}>
          {crumbs.map((c, i) => (
            <span key={i} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              {i > 0 && <span style={{ color: 'var(--tx3)' }}>›</span>}
              <span onClick={() => setFocus(['global', 'cluster', 'pool', 'super'].includes(c.lvl) ? null : { level: c.lvl as Level, card: c.card })} style={{ cursor: 'pointer', padding: '2px 5px', borderRadius: 5, color: i === crumbs.length - 1 ? 'var(--tx)' : ACCENT }}>{c.label}</span>
            </span>
          ))}
        </div>
        {/* KPI */}
        <div style={{ display: 'flex', gap: 14 }}>
          {([
            [`${Math.round(stats.kpi.util * 100)}%`, `集群${M_LABEL.util}`, 'var(--tx)'],
            [`${metric === 'fault' ? stats.kpi.faultDom : stats.kpi.hot}`, metric === 'fault' ? '故障域' : '热点卡', loadColor(0.9)],
            [`${stats.kpi.strag}`, '掉队卡', PARALLEL_COLORS.ep],
          ] as [string, string, string][]).map(([v, l, c], i) => (
            <div key={i} style={{ textAlign: 'right' }}>
              <div style={{ fontSize: 12, fontWeight: 600, lineHeight: 1.1, color: c, ...TNUM }}>{v}</div>
              <div style={{ fontSize: 9, color: 'var(--tx3)' }}>{l}</div>
            </div>
          ))}
        </div>
      </div>
      )}

      {/* ── body: left Smartscape 控制 · right panorama (scopeOnly) + 仪表 ── */}
      <div
        ref={splitRef}
        className={workbenchProfile ? 'hpc-console-body hpc-console-body--split workbench-frame-split pto-workbench-shell__panes' : 'hpc-console-body'}
        style={{ flex: 1, display: 'flex', minHeight: 0 }}
      >
        <div
          className={workbenchProfile ? 'hpc-console-left-pane workbench-pane pto-workbench-shell__pane' : 'hpc-console-left-pane'}
          data-pane="smartscape"
          style={{
            flex: workbenchProfile ? '0 0 38%' : '0 0 40%',
            maxWidth: workbenchProfile ? undefined : '46%',
            minWidth: workbenchProfile ? 0 : 340,
            ...(workbenchProfile ? { borderRadius: 'var(--pto-radius-lg)', background: 'var(--background-elevated)', overflow: 'hidden' } : { borderRight: '1px solid var(--bd)', background: 'var(--panel-solid)' }),
            display: 'flex',
            flexDirection: 'column',
            minHeight: 0,
          }}
        >
          <div className="hpc-console-pane-note" style={{ padding: '5px 12px', fontSize: 9, color: 'var(--tx3)', ...(workbenchProfile ? {} : { borderBottom: '1px solid var(--bd)' }), flexShrink: 0 }}>
            Plane view · hierarchy — click an entity to expand its chain (ancestors + descendants) and drive the array on the right; each level shows selected/total · p50 · red%
          </div>
          {/* funnel L7→L1 (fills width, left-aligned) */}
          <div style={{ flexShrink: 0, minHeight: 0, overflow: 'hidden', padding: '2px 0 0' }}>
            <Smartscape N={N} nBlades={nBlades} focus={focus} setFocus={setFocus} metric={metric} wlKind={wlKind} step={step} dir={dir} planeOn={planeOn} playing={playing} stats={stats} dark={dark} pm={pm} />
          </div>
          {/* 层级图之下：经典视图=L0 核组图；工作台驾驶舱=可切换抽屉（L0 核组 / 动态监控双模式 / 集合通信深潜）。
              通信关系仍长在上方的平面层级图（Smartscape 的 TP/PP/DP/EP 徽标与弧）上；这里补两块诊断视图。 */}
          {workbenchProfile ? (
            <div style={{ flex: '1 1 0', minHeight: 200, display: 'flex', flexDirection: 'column', borderTop: '1px dashed var(--bd)', minWidth: 0 }}>
              <div style={{ display: 'flex', gap: 4, padding: '6px 10px 4px', flexShrink: 0, flexWrap: 'wrap', alignItems: 'center' }}>
                <span style={{ fontSize: 9.5, fontWeight: 600, letterSpacing: 0.3, color: 'var(--tx3)', marginRight: 2 }}>层级图之下</span>
                {([['core', 'L0 核组'], ['monitor', '动态监控双模式'], ['comm', '集合通信深潜']] as [typeof leftLower, string][]).map(([k, l]) => (
                  <button key={k} onClick={() => setLeftLower(k)} style={{ ...btnBase, padding: '3px 9px', ...navBtn(leftLower === k) }}>{l}</button>
                ))}
              </div>
              <div style={{ flex: 1, minHeight: 0, overflowY: leftLower === 'core' ? 'hidden' : 'auto', display: 'flex', flexDirection: 'column', padding: leftLower === 'core' ? 0 : '2px 10px 10px' }}>
                {leftLower === 'core' ? l0Panel
                  : leftLower === 'monitor' ? (
                    <DualModeMonitor mode={monMode} setMode={setMonMode} workload={workload} step={step} setStep={setStep}
                      playing={playing} setPlaying={sync?.setPlaying ?? noop} sel={cubeSel} onSelectRank={(r) => setFocus(r == null ? null : { level: 'card', card: r })} baseHost={monBaseHost} dark={dark} />
                  ) : <CommDeepDive dark={dark} />}
              </div>
            </div>
          ) : l0Panel}
        </div>

        <div
          className={workbenchProfile ? 'hpc-console-pano-pane workbench-pane pto-workbench-shell__pane' : 'hpc-console-pano-pane'}
          data-pane="panorama"
          style={{ flex: 1, position: 'relative', minWidth: 0, ...(workbenchProfile ? { borderRadius: 'var(--pto-radius-lg)', overflow: 'hidden', background: '#ffffff' } : {}) }}
        >
          {workbenchProfile && cubeMode ? (
            /* 时空折叠魔方：复用立方重排（物理平铺 ⇄ 按 P 重排飞行动画 + 层级图选层聚合 + 策略着色） */
            <CubeView gen={gen} dark={dark} sync={cubeSync} embedded
              layout={cubeLayout} stratColor={cubeStrat} showComm showAlert
              aggLevel={cubeAgg} sel={cubeSel} onSelectRank={(r) => setFocus(r == null ? null : { level: 'card', card: r })} />
          ) : (
          <Canvas
            orthographic dpr={[1, 2]}
            camera={{ position: [reach, reach * 0.7, reach], zoom: 16, near: 0.1, far: 4000 }}
            gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.1, powerPreference: 'high-performance' }}
            onCreated={({ gl }) => { gl.domElement.addEventListener('webglcontextlost', (e) => e.preventDefault(), false); }}
          >
            <color attach="background" args={[surf.background]} />
            <fog attach="fog" args={[surf.fog, 90, 420]} />
            {visualProfile === 'opRankTime'
              ? <hemisphereLight intensity={surf.ambient} groundColor={dark ? '#10131a' : '#e8edf4'} />
              : <ambientLight intensity={surf.ambient} />}
            <directionalLight position={[8, 14, 6]} intensity={surf.key} />
            {visualProfile === 'opRankTime' && <directionalLight position={[-8, 8, -10]} intensity={surf.fill} />}
            <pointLight position={[0, 10, 0]} intensity={surf.point} color={surf.pointColor} />
            <FrameCamera bounds={scopeB} reach={reach} controls={controlsRef} zoomScale={2} />
            <SceneTheme.Provider value={dark}>
              <FullPodScene
                scale="64P" podCount={1} full gen={spec} overlays={OVERLAYS}
                runMode={runMode} phase={panoPhase} partition={panoPart} peers={panoPeers}
                status={panoStatus} planes={panoPlanes} onHoverInfo={setHover} onPick={() => { /* dbl-click via focus */ }}
                focusSel={panoSel} onSel={(s) => setFocus(selToFocus(s))} dir={dir} scopeOnly onScope={setScopeB}
              />
            </SceneTheme.Provider>
            <OrbitControls
              ref={controlsRef} makeDefault enableDamping dampingFactor={0.08}
              minPolarAngle={0} maxPolarAngle={Math.PI / 2} minDistance={2} maxDistance={600}
              mouseButtons={{ LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.PAN, RIGHT: THREE.MOUSE.PAN }}
            />
            {/* ViewCube 导航正方体 — 点面/棱/角回到标准视角（与 ClusterView 3D 画布一致）。
                bottom margin 抬高避开悬浮播放条。Latin face labels（默认 webfont 无 CJK 字形）。 */}
            <GizmoHelper alignment="bottom-left" margin={[64, 110]}>
              <GizmoViewcube
                faces={['Right', 'Left', 'Top', 'Bottom', 'Front', 'Back']}
                color={dark ? '#2a2e36' : '#eef1f6'} hoverColor="#4369ef"
                textColor={dark ? '#e6e6e6' : '#1c2433'} strokeColor={dark ? '#4a5160' : '#aab4c4'} opacity={0.95}
              />
            </GizmoHelper>
          </Canvas>
          )}

          {/* 右画布场景切换（仅工作台驾驶舱）：物理机柜层级 ⇄ 时空折叠魔方；魔方态附 布局(物理/TP/PP/DP/EP) */}
          {workbenchProfile && (
            <div style={{ position: 'absolute', top: 10, right: 10, zIndex: 8, display: 'flex', gap: 6, flexWrap: 'wrap', justifyContent: 'flex-end', maxWidth: 'calc(100% - 20px)' }}>
              <div style={{ display: 'flex', gap: 3, padding: 3, borderRadius: 9, background: 'var(--panel-shell-bg)', border: '1px solid var(--panel-shell-border)', boxShadow: 'var(--shadow-sm)', backdropFilter: 'blur(12px)' }}>
                {([[false, '物理机柜'], [true, '时空折叠魔方']] as [boolean, string][]).map(([v, l]) => (
                  <button key={l} onClick={() => setCubeMode(v)} style={{ ...btnBase, padding: '4px 10px', ...navBtn(cubeMode === v) }}>{l}</button>
                ))}
              </div>
              {cubeMode && (
                <div style={{ display: 'flex', gap: 3, padding: 3, borderRadius: 9, background: 'var(--panel-shell-bg)', border: '1px solid var(--panel-shell-border)', boxShadow: 'var(--shadow-sm)', backdropFilter: 'blur(12px)' }}>
                  {LAYOUT_VIEWS.map((v) => (
                    <button key={v} onClick={() => setCubeLayout(v)} title={`按 ${LAYOUT_LABEL[v]} 重排`} style={{ ...btnBase, padding: '4px 9px', ...navBtn(cubeLayout === v) }}>{LAYOUT_LABEL[v]}</button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* L0 细节由左侧原生 CoreGroupMiniSvg 图元 + 运行状态视图承担，右侧全景不再覆盖 L0 面板。 */}

          <button
            className={`hpc-console-info-toggle${infoOpen ? ' is-active' : ''}`}
            type="button"
            aria-label="信息面板"
            aria-expanded={infoOpen}
            title="信息面板"
            onClick={() => setInfoOpen((v) => !v)}
          >
            <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
              <circle cx="12" cy="12" r="9" />
              <path d="M12 10.8v5" />
              <path d="M12 7.6h.01" />
            </svg>
          </button>

          {infoOpen && (
            <div className="hpc-console-info-tray">
              <div style={{ ...card, padding: '10px 12px', borderColor: problem ? 'var(--danger, #ef4d4d)' : 'var(--bd)', background: problem ? 'rgba(60,24,24,0.92)' : 'var(--panel)' }}>
                <div style={{ fontSize: 9, letterSpacing: 0.4, color: 'var(--tx3)', display: 'flex', alignItems: 'center', gap: 5, marginBottom: 6 }}>
                  <span style={{ width: 7, height: 7, borderRadius: '50%', background: problem ? '#ef4d4d' : '#2bd47d' }} />DAVIS · 根因分析
                </div>
                {problem ? (
                  <>
                    <div style={{ fontSize: 11, fontWeight: 600, margin: '0 0 6px' }}>{problem.title}</div>
                    <div style={{ fontSize: 10, color: 'var(--tx2)', lineHeight: 1.55, marginBottom: 7 }}>{problem.chain}</div>
                    <div style={{ fontSize: 10, color: '#ef6d6d', marginBottom: 8 }}>{problem.impact}</div>
                    <button onClick={() => { setFocus({ level: 'node', card: problem.root * PER_CAB }); setDir('down'); }} style={{ width: '100%', border: `1px solid ${ACCENT}`, background: ACCENT, color: '#fff', fontSize: 10, padding: 6, borderRadius: 8, cursor: 'pointer' }}>定位根因 →</button>
                  </>
                ) : (
                  <div style={{ fontSize: 10, color: 'var(--tx3)', lineHeight: 1.55 }}>当前无活动问题。拖动下方时间轴到 t=34–46 触发过热事件，看根因链自动聚合与定位。</div>
                )}
              </div>

              <div style={{ ...card, padding: '10px 12px' }}>
                <div style={{ fontSize: 11, fontWeight: 600, margin: '0 0 2px' }}>{focusName(focus)}</div>
                <div style={{ fontSize: 10, color: 'var(--tx2)', marginBottom: 8 }}>{focus && rail ? `${LEVEL_NAME[focus.level]}${rail.count > 1 ? ' · ' + rail.count + ' 卡' : ''}` : `${N.toLocaleString()} 卡 · ${nBlades.toLocaleString()} Host · ${nCabs} 机柜（物理分组）`}</div>
                {focus && rail ? (
                  <>
                    {(['util', 'strag', 'fault'] as Metric[]).map((mm) => {
                      const v = rail[mm];
                      return (
                        <div key={mm}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, margin: '4px 0 2px' }}>
                            <span style={{ color: 'var(--tx2)' }}>{M_LABEL[mm]}</span><span style={{ fontWeight: 600, ...TNUM }}>{Math.round(v * 100)}%</span>
                          </div>
                          <div style={{ height: 5, borderRadius: 3, background: 'var(--btn)', overflow: 'hidden' }}><div style={{ height: '100%', width: `${Math.round(v * 100)}%`, background: loadColor(v), borderRadius: 3 }} /></div>
                        </div>
                      );
                    })}
                    {groups.length > 0 && (
                      <div style={{ marginTop: 9, borderTop: '1px solid var(--bd)', paddingTop: 7 }}>
                        <div style={{ fontSize: 9, color: 'var(--tx3)', marginBottom: 5 }}>并行关系 · rank↔rank（{pm.cfg}）</div>
                        {groups.map((g) => (
                          <div key={g.d} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 9.5, margin: '3px 0' }}>
                            <CollGlyph pat={g.pat} c={g.c} />
                            <span style={{ color: g.c, fontWeight: 700, width: 20, flexShrink: 0 }}>{g.d.toUpperCase()}</span>
                            <span style={{ color: 'var(--tx2)', flex: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{g.label} · {g.coll}</span>
                            <span style={{ color: 'var(--tx3)', flexShrink: 0, ...TNUM }}>{g.peers}对端/{g.deg}</span>
                          </div>
                        ))}
                      </div>
                    )}
                    {phys && (
                      <div style={{ marginTop: 4, borderTop: '1px solid var(--bd)', paddingTop: 7 }}>
                        <div style={{ fontSize: 9, color: 'var(--tx3)', marginBottom: 5 }}>通信平面 · {phys.planeLabel}</div>
                        {PLANES.map((p) => <span key={p.id} style={{ display: 'inline-block', fontSize: 9.5, padding: '2px 8px', borderRadius: 10, background: `${p.color}1f`, color: p.color, margin: '0 4px 4px 0', opacity: phys.plane === p.id || phys.plane === 'multi' ? 1 : 0.4 }}>{p.short}</span>)}
                        <div style={{ fontSize: 9, color: 'var(--tx3)', lineHeight: 1.5, marginTop: 2 }}>{phys.devices}</div>
                      </div>
                    )}
                  </>
                ) : (
                  <div style={{ fontSize: 10, color: 'var(--tx3)', lineHeight: 1.55 }}>左侧层级图驱动右侧阵列全景。点实体只展开其链路；方向(全链/上游/下游)过滤；镜头切阵列呈现；时间轴回放看问题定位。</div>
                )}
              </div>

              <div style={{ ...card, padding: '8px 11px', display: 'flex', flexDirection: 'column', gap: 5 }}>
                <div style={{ fontSize: 9.5, fontWeight: 600, color: 'var(--tx2)' }}>状态（红黄绿+灰 = 状态唯一一套色）</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                  {STATE_LABELS.map((lb, i) => <span key={i} style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 9, color: 'var(--tx2)' }}><span style={{ width: 9, height: 9, borderRadius: 2, background: stateColor(i) }} />{lb}</span>)}
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, borderTop: '1px solid var(--bd)', paddingTop: 4 }}>
                  {([['Chip', ENTITY_COLORS.card], ['Host', ENTITY_COLORS.node], ['机柜（物理分组）', ENTITY_COLORS.cab], ['Pod', ENTITY_COLORS.super]] as [string, string][]).map(([t, c]) => (
                    <span key={t} style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 9, color: 'var(--tx2)' }}><span style={{ width: 9, height: 9, borderRadius: 2, background: c }} />{t}</span>
                  ))}
                </div>
                <div style={{ fontSize: 9, color: 'var(--tx3)' }}>蓝=选中焦点 · 紫环=掉队卡 · 链路外压暗 · 单击实体联动</div>
              </div>
            </div>
          )}

          {hover && (
            <div style={{ position: 'absolute', right: 248, bottom: 12, maxWidth: 320, ...card, padding: '7px 11px', fontSize: 10, lineHeight: 1.5, color: 'var(--tx)', pointerEvents: 'none' }}>{hover}</div>
          )}
        </div>
      </div>

    </div>
  );
}
