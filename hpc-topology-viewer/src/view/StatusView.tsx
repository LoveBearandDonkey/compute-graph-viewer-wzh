/**
 * StatusView — 运行状态总览 (runtime-state / observability dashboard).
 *
 * A status-FIRST lens on the same super-node the rest of the app models: it answers
 * the operations journey "现在好不好 → 慢在哪一层 → 哪张卡/哪段通信 → 哪个时刻". The MAIN
 * lens canvas is the dominant body; a compact KPI strip + a clickable hierarchy
 * status-axis above it are the shared selection; a detail rail on the right shows the
 * selected entity + its communication ASSOCIATIONS (mirrors 平面视图's relationship view).
 *
 * Four lenses share ONE selection (breadcrumb + hierarchy axis): 状态热力 / 互联流量 /
 * 通信域 / 物理链路. The hierarchy drills the FULL hw-native-sys L7→L0 chain (EXACTLY 8 levels,
 * identical to every other view) — like 平面视图's 层级图:
 * 全球(L7)→集群(L6)→服务池(L5)→Pod(L4)→Host(L3)→Chip·NPU(L2·rank)→Die(L1·可选)→Core-Group(L0).
 * 机柜/Tile 不是层级（机柜=Pod 内物理分组、Tile=L0 内部粒度）。L0 (core) renders the
 * memory-architecture pattern (CoreGroupPattern), whose internals carry the Tile granularity.
 *
 * NOTHING is hard-coded: counts come from the generation spec (cabinets = NPU/64, nodes,
 * dies = 2/卡, cores = 32/卡 …), per-card load reuses the SAME `nodeLoad` field the 阵列全景
 * / 平面视图 playback uses, and the comm lenses are derived from the real parallel
 * decomposition (PARTITION_META) + planes (PLANES). Every element (node / link / die /
 * core / plane) is coloured by its OWN load and the replay `step` perturbs the whole field
 * live + injects a localized cabinet event, so the physical & comm lenses actually animate.
 *
 * Display text with brand terms is sourced from ../content (decoded at runtime).
 */
import { useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import {
  GENERATIONS, NODES_PER_CAB, NPUS_PER_NODE, COMPUTE_DIES_PER_CARD, IO_DIES_PER_CARD, CORES_PER_CARD,
  PODS_PER_POOL, PODS_PER_CLUSTER, POOLS_PER_CLUSTER,
  PLANES, PARTITION_META, ENTITY_COLORS, PARALLEL_COLORS, WORKLOAD, WORKLOAD_DETAIL, WORKLOAD_REFS, STEP_DECOMP,
  BENCHMARKS, BENCH_MODELS, BENCH_PANGU_IDX,
  parallelMap, REPLAY, cardLoad01, cardStraggler, cardFault, loadColor, loadState, stateColor, STATE_LABELS,
  type Gen, type ViewSync,
} from '../scene/data';
import { TOK } from '../content';
import { busWire2d } from './wire2d';
import { CoreGroupPattern } from './CoreGroupPattern';
import { SceneVisualProfileContext } from '../scene/visual-profile';

// ── shared button language (matches ClusterView / PlaneView) ──
const ACCENT = '#4369ef';
// PP 关系描边显示色：PARALLEL_COLORS.pp(#04d793 绿) 与状态绿同色 → 高亮时改用青色避让（图例注明）
const PP_HI = '#22d3ee';
const SECONDARY: React.CSSProperties = { border: '1px solid var(--button-secondary-border)', background: 'var(--button-secondary-bg)', color: 'var(--foreground-muted)' };
function inkOf(hex: string): string {
  const h = hex.replace('#', ''); if (h.length < 6) return '#fff';
  const r = parseInt(h.slice(0, 2), 16), g = parseInt(h.slice(2, 4), 16), b = parseInt(h.slice(4, 6), 16);
  return 0.299 * r + 0.587 * g + 0.114 * b > 150 ? '#10131a' : '#fff';
}
function navBtn(active: boolean): React.CSSProperties {
  return active ? { border: '1px solid var(--primary)', background: 'var(--primary)', color: 'var(--primary-foreground)', fontWeight: 600, transform: 'translateY(-1px)' } : { ...SECONDARY };
}
function toggleBtn(active: boolean, c: string): React.CSSProperties {
  return active ? { border: `1px solid ${c}`, background: c, color: inkOf(c), fontWeight: 600 } : { ...SECONDARY };
}
const LBL: React.CSSProperties = { fontSize: 11, fontWeight: 500, letterSpacing: 0.5, textTransform: 'uppercase', color: 'var(--tx3)' };
const TNUM: React.CSSProperties = { fontVariantNumeric: 'tabular-nums' };
const MONO = "'JetBrains Mono','Consolas',ui-monospace,monospace";

type Phase = 'pretrain' | 'prefill' | 'decode';
type Metric = 'util' | 'strag' | 'fault';
type Lens = 'heat' | 'flow' | 'domain' | 'phys' | 'matrix';
// full hierarchy chain (hw-native-sys L7→L0) — EXACTLY 8 levels, identical to every other view.
// L7 global → L6 cluster → L5 pool(服务池) → L4 super(Pod) → L3 node(Host) →
// L2 rank(Chip·NPU) → L1 die(可选) → L0 core(Core-Group·最深层级).
// 机柜(cab)只是 Pod 内物理分组、Tile 只是 L0 内部粒度 —— 均不是层级、不出现在层级轴/漏斗/面包屑。
type Level = 'global' | 'cluster' | 'pool' | 'super' | 'node' | 'rank' | 'die' | 'core';
const SUBCARD: Level[] = ['die', 'core'];

// 工况 → 基准负载 + 驱动的 phaseKind(复用 nodeLoad 的相位语义)
const PH: Record<Phase, { label: string; base: number; kind: string }> = {
  pretrain: { label: '预训练', base: 0.55, kind: 'compute' },
  prefill: { label: 'Prefill', base: 0.50, kind: 'compute' },
  decode: { label: 'Decode', base: 0.62, kind: 'comm' },
};
const NIC_LBL = TOK.qingtian.split(' ')[0];   // 擎天 — brand via TOK (no plaintext brand in source)
const STEP_MAX = 60;
const EVT_LO = 34, EVT_HI = 46;   // 回放事件窗口：局部热点（演示局部故障/拥塞的时间定位）
const HOST_GRID_MAX = 1024;       // Pod 级 Host 大网格采样上限（超出显示 +N）

const rnd = (x: number) => { const v = Math.sin(x * 99.13) * 43758.5453; return v - Math.floor(v); };
const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);

export function StatusView({ gen, dark, sync, lens: lensP, setLens: setLensP, relHi: relHiP, setRelHi: setRelHiP }: {
  gen: Gen; dark: boolean; sync?: ViewSync;
  lens?: Lens; setLens?: (l: Lens) => void; relHi?: boolean; setRelHi?: (v: boolean) => void;
}) {
  const visualProfile = useContext(SceneVisualProfileContext);
  const workbenchProfile = visualProfile === 'opRankTime';
  const spec = GENERATIONS[gen];

  // ── real counts from the spec (NOTHING hard-coded) ──
  const NPC = NODES_PER_CAB * NPUS_PER_NODE;            // 64 NPU / cabinet
  const NPU_TOT = spec.totalNpus;                        // 8192 (A5) / 15488 (A6)
  const CAB = Math.max(1, Math.round(NPU_TOT / NPC));    // cabinets / super-node
  const NODES = CAB * NODES_PER_CAB;                     // nodes / super-node
  const NPN = NPUS_PER_NODE;                             // 8 NPU / node
  const EVT_CAB = Math.min(CAB - 1, REPLAY.evtCab);   // overheating cabinet during the event (shared with 工作台)

  // ── shared selection (drives ALL lenses) · 工况/时间/播放 come from the cross-view sync when present ──
  const [phaseL] = useState<Phase>('decode');
  const phase = sync?.workload ?? phaseL;
  const [metricL] = useState<Metric>('util');
  const metric = (sync?.metric ?? metricL) as Metric;
  const [lensL, setLensL] = useState<Lens>('heat');
  const lens = lensP ?? lensL; const setLens = setLensP ?? setLensL;
  const [relHiL, setRelHiL] = useState(true);
  const relHi = relHiP ?? relHiL; const setRelHi = setRelHiP ?? setRelHiL;
  const [pods, setPods] = useState(4);                   // 集群 = N 个 Pod（示意，跨 Pod DP）
  const [selLevel, setSelLevel] = useState<Level>('super');
  const [selPool, setSelPool] = useState(0);            // 选中服务池（L5）
  const [selSpod, setSelSpod] = useState(0);            // 选中 Pod（L4）· 全局索引 [0,pods)
  const [selCab, setSelCab] = useState(0);
  const [selNode, setSelNode] = useState(0);             // global node index within the super-node [0,NODES)
  const [selNpu, setSelNpu] = useState(-1);              // NPU within the selected node [0,8); -1 = none picked
  const [selCore, setSelCore] = useState(0);             // AI Core within the selected card [0,32)
  const [stepL, setStepL] = useState(0);
  const step = sync?.step ?? stepL;
  const setStep = sync?.setStep ?? setStepL;
  const [playingL] = useState(false);
  const playing = sync?.playing ?? playingL;
  const [tip, setTip] = useState<{ x: number; y: number; t: string } | null>(null);
  const flowRef = useRef(0);   // 连线彗星流动相位 —— 始终推进（即使未播放，连线持续流动）
  // ── N×N 通信矩阵 lens（需求4）：聚合三档 + 泳道 phase(并行维) 过滤 ──
  const [matAgg, setMatAgg] = useState<'pod' | 'host' | 'rank'>('host');
  const [matDim, setMatDim] = useState<'TP' | 'PP' | 'EP' | 'DP'>('EP');
  const matGeom = useRef<{ mx: number; my: number; cs: number; N: number } | null>(null);

  useEffect(() => { setSelPool(0); setSelSpod(0); setSelCab(0); setSelNode(0); setSelNpu(-1); }, [gen]);
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setStep((s) => (s + 1) % (STEP_MAX + 1)), 650);
    return () => clearInterval(id);
  }, [playing]);

  // L5 服务池推导：pools 个服务池、每池 PODS_PER_POOL 个 Pod（示意分组）
  const pools = Math.max(1, Math.ceil(pods / PODS_PER_POOL));
  const podsInPool = Math.max(1, Math.min(PODS_PER_POOL, pods - selPool * PODS_PER_POOL));

  // SINGLE SOURCE OF TRUTH physical tiling — same across 平面/工作台/3D；关系高亮 + 详情栏共用
  const pm = useMemo(() => parallelMap(phase, NPU_TOT), [phase, NPU_TOT]);

  const ev = step >= EVT_LO && step <= EVT_HI;
  const kind = PH[phase].kind;
  const cardJ = selNpu < 0 ? 0 : selNpu;                  // the focused card's NPU index in its node
  const cardRank = selNode * NPN + cardJ;                 // rank id within the super-node
  const cardSelected = selNpu >= 0 || SUBCARD.includes(selLevel);

  // ── live per-NPU field — delegates to the SHARED model (data.ts) so the SAME card reads the SAME
  //    value here and in 工作台. k = card index within the super-node (gnode*8+j), pod = super-node sp. ──
  const isStrag = useCallback((sp: number, gnode: number) => cardStraggler(gnode * NPN, step, sp), [step, NPN]);
  const faultAt = useCallback((sp: number, gnode: number, j: number) => cardFault(gnode * NPN + j, step, sp), [step, NPN]);
  const util01 = useCallback((sp: number, gnode: number, j: number) => cardLoad01(gnode * NPN + j, kind, step, sp, NPU_TOT), [kind, step, NPN, NPU_TOT]);
  const metricVal = useCallback((sp: number, gnode: number, j: number) => {
    if (metric === 'fault') return faultAt(sp, gnode, j) ? 0.95 : 0.12;
    const u = util01(sp, gnode, j);
    if (metric === 'strag') return isStrag(sp, gnode) ? 0.85 + rnd(sp * NPU_TOT + gnode * NPN + j) * 0.15 : Math.max(0, (u - 0.5)) * 0.4;
    return u;
  }, [metric, faultAt, util01, isStrag, NPU_TOT, NPN]);

  const nodeMean = useCallback((sp: number, gnode: number) => { let s = 0; for (let j = 0; j < NPN; j++) s += metricVal(sp, gnode, j); return s / NPN; }, [metricVal, NPN]);
  const cabMean = useCallback((sp: number, cab: number) => { let s = 0; for (let n = 0; n < NODES_PER_CAB; n++) s += nodeMean(sp, cab * NODES_PER_CAB + n); return s / NODES_PER_CAB; }, [nodeMean]);
  const spodMean = useCallback((sp: number) => { let s = 0; for (let c = 0; c < CAB; c++) s += cabMean(sp, c); return s / CAB; }, [cabMean, CAB]);
  // L5 pool mean = 池内 Pod 均值；L6 cluster mean = 全集群 Pod 均值
  const poolMean = useCallback((p: number) => { let s = 0, n = 0; const p0 = p * PODS_PER_POOL; for (let i = p0; i < Math.min(p0 + PODS_PER_POOL, pods); i++) { s += spodMean(i); n++; } return n ? s / n : 0; }, [spodMean, pods]);
  const clusterMean = useCallback(() => { let s = 0; for (let i = 0; i < pods; i++) s += spodMean(i); return s / pods; }, [spodMean, pods]);

  // ── below-card on-chip field (compute Die / AI Core), anchored to the card's load ──
  const cardBaseU = useCallback(() => util01(selSpod, selNode, cardJ), [util01, selSpod, selNode, cardJ]);
  const dieVal = useCallback((d: number) => clamp01(cardBaseU() + (rnd(cardRank * 7 + d * 131 + step * 0.06) - 0.5) * 0.16), [cardBaseU, cardRank, step]);
  const coreVal = useCallback((c: number) => {
    const isVec = c % 8 === 7;   // Cube∶Vector ≈ 8∶1 → 4 Vector / 32 cores
    const b = cardBaseU() + (kind === 'compute' ? 0.06 : -0.05);
    return clamp01(b + (rnd(cardRank * 101 + c * 3.7 + step * 0.05) - 0.5) * 0.34 + (isVec ? -0.05 : 0));
  }, [cardBaseU, cardRank, step, kind]);
  const scopeMean = useCallback(() => {
    if (selLevel === 'global' || selLevel === 'cluster') return clusterMean();
    if (selLevel === 'pool') return poolMean(selPool);
    if (selLevel === 'super' || selLevel === 'rank') return spodMean(selSpod);
    if (selLevel === 'node') return nodeMean(selSpod, selNode);
    return cardBaseU();   // die/core → the card's load
  }, [selLevel, selPool, selSpod, selNode, clusterMean, poolMean, spodMean, nodeMean, cardBaseU]);

  const cardName = `Host${(selNode % NODES_PER_CAB) + 1}·Chip r${cardJ}（机柜${((selNode / NODES_PER_CAB) | 0) + 1} 物理分组）`;
  const scopeName = useCallback(() => {
    if (selLevel === 'global') return `全球 · 本集群（${PODS_PER_CLUSTER} Pod·在线 ${pods}）`;
    if (selLevel === 'cluster') return `集群 = ${PODS_PER_CLUSTER} Pod = ${POOLS_PER_CLUSTER} 服务池 · 在线 ${pods} Pod`;
    if (selLevel === 'pool') return `服务池#${selPool + 1} · ×${PODS_PER_POOL} Pod/池 · 在线 ${podsInPool}`;
    if (selLevel === 'super' || selLevel === 'rank') return `Pod#${selSpod + 1}`;
    if (selLevel === 'node') return `Host${(selNode % NODES_PER_CAB) + 1}（机柜${((selNode / NODES_PER_CAB) | 0) + 1} 物理分组）`;
    if (selLevel === 'die') return `${cardName} · 计算 Die`;
    return `${cardName} · Core-Group`;
  }, [selLevel, pods, pools, selPool, podsInPool, selSpod, selNode, cardName]);

  // ── KPI (utilisation/fault-based, independent of colour metric) ──
  const kpi = useMemo(() => {
    let sum = 0, red = 0, fa = 0;
    for (let gn = 0; gn < NODES; gn++) for (let j = 0; j < NPN; j++) {
      const u = util01(selSpod, gn, j); sum += u; if (loadState(u) >= 2) red++; if (faultAt(selSpod, gn, j)) fa++;
    }
    const n = NODES * NPN, redR = red / n, mfu = Math.round((sum / n) * 100 * (1 - redR * 0.4));
    return { redR, mfu, fa, stepMs: Math.round(92 + redR * 420 + (fa ? 40 : 0)) };
  }, [util01, faultAt, selSpod, NODES, NPN]);

  // ── hierarchy status-axis (typical p50 · red% · peak p95 — exposes outliers); now incl. below-card ──
  const axis = useMemo(() => {
    const cards: number[] = []; for (let gn = 0; gn < NODES; gn++) for (let j = 0; j < NPN; j++) cards.push(metricVal(selSpod, gn, j));
    const nodes: number[] = []; for (let gn = 0; gn < NODES; gn++) nodes.push(nodeMean(selSpod, gn));
    const spods: number[] = []; for (let i = 0; i < pods; i++) spods.push(spodMean(i));
    const poolArr = Array.from({ length: pools }, (_, p) => poolMean(p));
    const dies = Array.from({ length: COMPUTE_DIES_PER_CARD }, (_, d) => dieVal(d));
    const cores = Array.from({ length: CORES_PER_CARD }, (_, c) => coreVal(c));
    const pctl = (a: number[], p: number) => { const b = a.slice().sort((x, y) => x - y); return b[Math.min(b.length - 1, Math.floor(p * b.length))]; };
    const redF = (a: number[]) => { let c = 0; for (const v of a) if (loadState(v) >= 2) c++; return c / a.length; };
    const mk = (id: Level, nm: string, su: string, a: number[]) => ({ id, nm, su, p50: pctl(a, 0.5), p95: pctl(a, 0.95), red: redF(a) });
    return [
      mk('global', '全球 L7', '×1', [clusterMean()]),
      mk('cluster', '集群 L6', `${PODS_PER_CLUSTER} Pod·在线${pods}`, spods),
      mk('pool', '服务池 L5', `×${PODS_PER_POOL} Pod`, poolArr),
      mk('super', `Pod L4 #${selSpod + 1}`, `${NPU_TOT.toLocaleString()} NPU`, cards),
      mk('node', 'Host L3', `${NODES.toLocaleString()} Host`, nodes),
      mk('rank', 'Chip·NPU L2', `${NPU_TOT.toLocaleString()} Chip`, cards),
      mk('die', 'Die L1·可选', `${COMPUTE_DIES_PER_CARD}/卡`, dies),
      mk('core', 'Core-Group L0', `${CORES_PER_CARD}/卡`, cores),
    ];
  }, [metricVal, nodeMean, spodMean, poolMean, clusterMean, dieVal, coreVal, selSpod, pods, pools, NODES, NPN, NPU_TOT]);

  // ── flow-matrix cell intensity (real parallel relationships) ──
  const flowCfg = (): [number, string] => ({
    global: [pods, 'Pod × Pod · 跨 Pool/集群 DP 副本间'] as [number, string],
    cluster: [pods, 'Pod × Pod · DP 副本间'] as [number, string],
    pool: [podsInPool, '池内 Pod × Pod · Pool 内互联'] as [number, string],
    super: [Math.min(NODES, 64), 'Host × Host · Pod 内互联（采样）'] as [number, string],
    rank: [Math.min(NODES, 64), 'Host × Host · Pod 内互联（采样）'] as [number, string],
    node: [NPN, 'Chip·NPU × Chip·NPU · TP AllReduce'] as [number, string],
    die: [NPN, 'Chip·NPU × Chip·NPU（片上无跨卡矩阵→显示所属 Host）'] as [number, string],
    core: [NPN, 'Chip·NPU × Chip·NPU（片上 NoC·非跨卡）'] as [number, string],
  }[selLevel]);
  // Cell intensity keyed to the REAL collective per phase (Pangu Pro MoE, arXiv:2505.21411).
  // loadColor is a DISCRETE 3-state map (green<40 / yellow40–70 / red>70), so cells only read
  // as "changing" when values actually cross those thresholds. A near-uniform field therefore
  // looks like one flat yellow block. We give each cell a per-SOURCE(row) + per-DEST(col) stable
  // hotness (real All-to-All load imbalance → coherent hot/cold bands, NOT a diagonal) plus a
  // live term that swings wide enough to flip colour states as playback advances. selLevel/
  // selCore fold in so sub-card (die/core/tile) levels show a DIFFERENT field, not a frozen one.
  //  · decode  → EP token All-to-All → 密集 any-to-any，行/列热度纹理（无对角）
  //  · prefill → 计算为主 + EP A2A → 近对角软带 + 纹理
  //  · 预训练  → DP Ring 邻近带 + EP A2A 底噪 + 纹理
  // Base intensity anchored to STEP_DECOMP comm fractions (Pangu Pro MoE, arXiv:2505.21411):
  //   decode 12% comm → commBase ≈ 0.39   prefill 16% → 0.45   pretrain 30% → 0.65
  // loadColor is DISCRETE (green<0.4 / yellow 0.4–0.7 / red>0.7), so the base must be chosen
  // such that row/col hotness + live swing cross those thresholds for visible colour variation.
  const commFrac = STEP_DECOMP[phase as 'pretrain' | 'prefill' | 'decode']
    ?.find(p => p.label.includes('通信'))?.frac ?? 0.20;
  const tcell = useCallback((i: number, j: number, N: number) => {
    const d = Math.abs(i - j);
    const salt = selSpod * 4.2 + selCab * 1.7 + selNode * 0.6 + SUBCARD.indexOf(selLevel) * 2.3 + selCore * 0.4;
    const rowHot = rnd(i * 3.1 + salt) - 0.5;          // 该源单元忙/闲（整行纹理）
    const colHot = rnd(j * 2.7 + salt + 5) - 0.5;      // 该目的单元忙/闲（整列纹理）
    const live = rnd(i * 0.7 + j * 0.9 + step * 0.05 + flowRef.current * 0.9) - 0.5;   // 逐帧流动
    const commBase = 0.21 + commFrac * 1.5;              // paper-anchored base intensity
    let v: number;
    if (phase === 'decode') {
      // EP token All-to-All: dense any-to-any. Paper: A2A≈8% of step, moderate network pressure.
      v = commBase + rowHot * 0.50 + colHot * 0.50;
    } else if (phase === 'prefill') {
      // TP AllReduce + SP AllGather: intra-host (near-diagonal). EP A2A adds off-diagonal noise.
      v = commBase + (d <= Math.max(1, N / 8) ? 0.20 : -0.05) + rowHot * 0.30 + colHot * 0.30;
    } else {
      // Pretrain: DP Ring-AllReduce (near-diagonal) + EP A2A background. Paper: 30% comm → high base.
      v = commBase + (d <= Math.max(1, N / 10) ? 0.20 : -0.15) + rowHot * 0.28 + colHot * 0.28;
    }
    v += live * 0.32;
    if (ev && selSpod === 0 && N === CAB && (i === EVT_CAB || j === EVT_CAB)) v += 0.35;
    return clamp01(v);
  }, [phase, commFrac, selSpod, selCab, selNode, selLevel, selCore, step, ev, CAB, EVT_CAB]);

  // ── plane utilisation (UB scale-up / RDMA scale-out / DP / VPC) by scope ──
  const planeUtil = useCallback(() => {
    const sm = scopeMean();
    // UB plane carries TP/EP in-domain collectives. Paper: decode EP A2A=8%, prefill SP=16%,
    // pretrain DP+EP=30% → drive the bar base from STEP_DECOMP comm fraction.
    const ubBase = commFrac * 1.8;   // decode 0.22 / prefill 0.29 / pretrain 0.54
    return [
      { n: `${PLANES[0].short} · ${selLevel === 'node' ? '本 Host' : '域内'}(TP/EP)`, u: clamp01(ubBase + sm * 0.55), c: PLANES[0].color },
      { n: `${PLANES[1].short}(DP/PP)`, u: clamp01(0.24 + sm * 0.4 + (ev ? 0.10 : 0)), c: PLANES[1].color },
      { n: '集群 DP AllReduce', u: clamp01(0.18 + sm * 0.32), c: PLANES[1].color },
      { n: `${PLANES[2].short} · 南北向`, u: clamp01(0.12 + sm * 0.08), c: PLANES[2].color },
    ];
  }, [scopeMean, commFrac, selLevel, phase, ev]);

  // ── communication domains (process↔process) ──
  const domains = useCallback(() => {
    const sm = scopeMean(), adj = (u: number) => Math.max(0.05, Math.min(1, u + (sm - 0.55) * 0.5));
    // parallel degrees + collectives are the REAL Pangu Pro MoE config (arXiv:2505.21411):
    // train TP8·EP2·PP5·VPP5 · infer H2P 注意力 DP2+TP4 / 路由专家 TP2+EP4 / 共享 TP8.
    return [
      { key: 'tp', nm: PARTITION_META.tp.label, pat: 'ring', sc: 'Pod 内 SU', co: 'AllReduce', me: `训练TP${WORKLOAD.train.tp}·推理TP${WORKLOAD.inferAttn.tp} · ${NPN} rank/Host`, u: adj(phase === 'decode' ? 0.5 : 0.72) },
      { key: 'sp', nm: 'SP 序列并行', pat: 'ring', sc: '与 TP 同域', co: 'AllGather+ReduceScatter', me: '与 TP 同域', u: adj(phase === 'decode' ? 0.45 : 0.6) },
      { key: 'ep', nm: PARTITION_META.ep.label, pat: 'a2a', sc: 'Pod 内 SU', co: '层级化 All-to-All', me: `EP${WORKLOAD.inferRouted.ep} · ${WORKLOAD.routedExperts}路由/${WORKLOAD.activatedExperts}激活/${WORKLOAD.sharedExperts}共享`, u: adj(phase === 'decode' ? 0.92 : 0.5) },
      { key: 'pp', nm: PARTITION_META.pp.label, pat: 'p2p', sc: '跨 Host', co: 'P2P send/recv', me: `PP${WORKLOAD.train.pp}·VPP${WORKLOAD.train.vpp} · stage 间`, u: adj(0.35) },
      { key: 'dp', nm: PARTITION_META.dp.label, pat: 'ring', sc: '跨 Pod SO', co: 'Ring-AllReduce', me: `DP${WORKLOAD.inferAttn.dp} · ${pods} 副本`, u: adj(0.4 + (ev ? 0.08 : 0)) },
    ] as { key: string; nm: string; pat: 'ring' | 'a2a' | 'p2p'; sc: string; co: string; me: string; u: number }[];
  }, [scopeMean, phase, ev, NPN, pods]);
  const domActive = useCallback((): Record<string, boolean> => {
    if (SUBCARD.includes(selLevel)) return { tp: false, sp: false, ep: false, pp: false, dp: false };   // 片上：无 rank 间通信
    if (selLevel === 'global' || selLevel === 'cluster' || selLevel === 'pool') return { tp: false, sp: false, ep: false, pp: true, dp: true };   // 跨 Pod：DP/PP
    if (selLevel === 'rank') return { tp: true, sp: true, ep: true, pp: true, dp: true };
    return { tp: true, sp: true, ep: true, pp: false, dp: false };
  }, [selLevel]);

  // ───────────────────────── canvas ─────────────────────────
  const wrapRef = useRef<HTMLDivElement>(null);
  const cvRef = useRef<HTMLCanvasElement>(null);
  const cells = useRef<{ x: number; y: number; w: number; h: number; kind: string; idx: number }[]>([]);
  const P = dark
    ? { bg: '#121418', grid: 'rgba(255,255,255,0.05)', track: '#262E3C', ink: 'rgba(255,255,255,0.88)', ink2: 'rgba(255,255,255,0.58)', mut: '#5A6172', frame: 'rgba(255,255,255,0.10)', neutral: '#39404e' }
    : { bg: '#fbfbfd', grid: 'rgba(67,105,239,0.08)', track: '#e4e7ef', ink: 'rgba(0,0,0,0.80)', ink2: 'rgba(0,0,0,0.55)', mut: '#9aa3b2', frame: 'rgba(0,0,0,0.10)', neutral: '#b9c2d4' };

  const draw = useCallback(() => {
    const cv = cvRef.current, wrap = wrapRef.current; if (!cv || !wrap) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const W = wrap.clientWidth, H = wrap.clientHeight;
    if (cv.width !== W * dpr || cv.height !== H * dpr) { cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + 'px'; cv.style.height = H + 'px'; }
    const ctx = cv.getContext('2d')!; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = P.bg; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = P.grid; for (let gx = 14; gx < W; gx += 22) for (let gy = 14; gy < H; gy += 22) { ctx.beginPath(); ctx.arc(gx, gy, 0.9, 0, 7); ctx.fill(); }
    cells.current = [];
    const PAD = 16, STRIP_H = 54;   // STRIP_H = parent-context strip height (heat lens)
    const tx = (s: string, x: number, y: number, c: string, f = '11px Inter', a: CanvasTextAlign = 'left') => { ctx.fillStyle = c; ctx.font = f; ctx.textAlign = a; ctx.fillText(s, x, y); ctx.textAlign = 'left'; };
    const bar = (x: number, y: number, w: number, h: number, u: number, c?: string) => { ctx.fillStyle = P.track; ctx.fillRect(x, y, w, h); ctx.fillStyle = c ?? loadColor(u); ctx.fillRect(x, y, w * u, h); };
    // 所有连线统一 bus-wiring「平面」样式：管体描边 + connector 接点 + 始终沿线流动彗星（未播放也流动；
    // 状态/状态色仍只随 playing 的 step 变化）。
    const line = (x1: number, y1: number, x2: number, y2: number, c: string, w: number, dash = false, caps = false) => {
      busWire2d(ctx, [[x1, y1], [x2, y2]], c, w, { phase: flowRef.current, flowing: dash || caps, caps, dash: dash ? [6, 5] : null, tube: w >= 1.5, alpha: ctx.globalAlpha });
    };
    const dia = (x: number, y: number, r: number, c: string, lab?: string) => { ctx.beginPath(); ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y); ctx.closePath(); ctx.fillStyle = P.neutral; ctx.fill(); ctx.strokeStyle = c; ctx.lineWidth = 2; ctx.stroke(); if (lab) tx(lab, x, y + 3.5, P.ink, '10px Inter', 'center'); };
    const fbox = (x: number, y: number, w: number, h: number, fill: string, stroke?: string) => { ctx.fillStyle = fill; ctx.fillRect(x, y, w, h); if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.4; ctx.strokeRect(x, y, w, h); } };
    // 关系高亮（B）：焦点=白色粗描边（外加暗 halo 保证浅色主题可见）；图例 = parallelMap 真值
    const focusRing = (x: number, y: number, w: number, h: number) => { ctx.strokeStyle = 'rgba(0,0,0,0.55)'; ctx.lineWidth = 3.4; ctx.strokeRect(x, y, w, h); ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.8; ctx.strokeRect(x, y, w, h); };
    const relLegend = (y: number) => {
      ctx.font = '9.5px Inter';
      let x = PAD;
      const put = (c: string, s: string) => { ctx.fillStyle = c; ctx.fillRect(x, y - 8, 8, 8); tx(s, x + 11, y, P.ink2, '9.5px Inter'); x += 11 + ctx.measureText(s).width + 12; };
      put(PARALLEL_COLORS.tp, `TP ${pm.tp}`); put(PP_HI, `PP ×${pm.pp}`); put(PARALLEL_COLORS.dp, `DP ×${pm.dp}`); put(PARALLEL_COLORS.ep, `EP ×${pm.ep}`);
      tx('（parallelMap 真值 · 采样显示 · PP 用青色避让状态绿 · 白=焦点卡）', x, y, P.mut, '9.5px Inter');
    };

    // parent-context strip is SHARED by all lenses: draw once, offset each lens's top content by topY
    const pc = parentCtx();
    const topY = pc ? PAD + STRIP_H : PAD;
    if (pc) drawParentStrip(pc);
    if (lens === 'heat') drawHeat(topY); else if (lens === 'flow') drawFlow(topY); else if (lens === 'domain') drawDomain(topY); else if (lens === 'matrix') drawMatrix(topY); else drawPhys(topY);

    // parent (上一层) context: the level we drilled FROM; click a sibling to switch without going up
    function parentCtx(): { name: string; n: number; val: (i: number) => number; sel: number; kind: string; lab: (i: number) => string } | null {
      if (selLevel === 'pool') return { name: '集群 · 服务池', n: pools, val: (i) => poolMean(i), sel: selPool, kind: 'ppool', lab: (i) => '池' + (i + 1) };
      if (selLevel === 'super' || selLevel === 'rank') return { name: '服务池 · Pod', n: pods, val: (i) => spodMean(i), sel: selSpod, kind: 'pspod', lab: (i) => 'Pod' + (i + 1) };
      if (selLevel === 'node') { const cb = (selNode / NODES_PER_CAB) | 0; return { name: `Host（机柜${cb + 1} 物理分组）`, n: NODES_PER_CAB, val: (i) => nodeMean(selSpod, cb * NODES_PER_CAB + i), sel: selNode - cb * NODES_PER_CAB, kind: 'pnode', lab: (i) => 'H' + (i + 1) }; }
      if (SUBCARD.includes(selLevel)) return { name: `Host${(selNode % NODES_PER_CAB) + 1} · Chip(rank)`, n: NPN, val: (i) => util01(selSpod, selNode, i), sel: cardJ, kind: 'pcard', lab: (i) => 'r' + i };
      return null;   // global / cluster: no parent strip
    }
    function drawParentStrip(pc: NonNullable<ReturnType<typeof parentCtx>>) {
      tx(`上层 · ${pc.name}（高亮=当前选区 · 点击切换同级）`, PAD, PAD + 11, P.ink2, '11px Inter');
      const y = PAD + 18, h = 26, gap = pc.n > 48 ? 1 : 2;
      const cw = (W - 2 * PAD - (pc.n - 1) * gap) / pc.n;
      for (let i = 0; i < pc.n; i++) {
        const x = PAD + i * (cw + gap), v = pc.val(i);
        fbox(x, y, Math.max(1, cw), h, loadColor(v));
        if (i === pc.sel) { ctx.strokeStyle = ACCENT; ctx.lineWidth = 2.5; ctx.strokeRect(x + 1.25, y + 1.25, cw - 2.5, h - 2.5); }
        if (cw >= 24) tx(pc.lab(i), x + cw / 2, y + h / 2 + 3.5, inkOf(loadColor(v)), '9px Inter', 'center');
        cells.current.push({ x, y, w: Math.max(1, cw), h, kind: pc.kind, idx: i });
      }
      line(PAD, PAD + STRIP_H - 6, W - PAD, PAD + STRIP_H - 6, P.frame, 1);
    }

    // ════════ 状态热力 ════════
    function drawHeat(topY: number) {
      if (selLevel === 'super') { drawHostGrid(topY); return; }   // Pod 直接下钻 Host：大网格采样
      type Item = { val: number; kind: string; idx: number; label: string; color?: string; sub?: string };
      let items: Item[] | null = null;
      if (selLevel === 'global') items = [
        { val: -1, kind: 'gclus', idx: 0, label: 'Global A', color: P.neutral, sub: '兄弟集群（示意）' },
        { val: clusterMean(), kind: 'gclus', idx: 1, label: '本集群', sub: `${PODS_PER_CLUSTER} Pod（>52万卡）· 在线 ${pods} · 点击下钻 Cluster` },
        { val: -1, kind: 'gclus', idx: 2, label: 'Global C', color: P.neutral, sub: '兄弟集群（示意）' },
      ];
      // 真实数量：集群 = 16 服务池（POOLS_PER_CLUSTER），前 ceil(pods/4) 个在线、其余灰=未接入
      else if (selLevel === 'cluster') items = Array.from({ length: POOLS_PER_CLUSTER }, (_, p) => (p < pools
        ? { val: poolMean(p), kind: 'pool', idx: p, label: `服务池#${p + 1}`, sub: `${Math.min(PODS_PER_POOL, pods - p * PODS_PER_POOL)} Pod 在线` }
        : { val: -1, kind: 'pooloff', idx: p, label: `服务池#${p + 1}`, color: P.neutral, sub: '未接入（离线）' }));
      // 真实数量：每池 = 4 Pod（PODS_PER_POOL），在线（全局序 <pods）带状态色、其余灰=未上线
      else if (selLevel === 'pool') items = Array.from({ length: PODS_PER_POOL }, (_, i) => { const g = selPool * PODS_PER_POOL + i; return g < pods
        ? { val: spodMean(g), kind: 'spod', idx: g, label: `Pod#${g + 1}` }
        : { val: -1, kind: 'spodoff', idx: g, label: `Pod#${g + 1}`, color: P.neutral, sub: '未上线' }; });
      else if (selLevel === 'node') items = Array.from({ length: NPN }, (_, j) => ({ val: metricVal(selSpod, selNode, j), kind: 'npuc', idx: j, label: `Chip r${j}`, sub: faultAt(selSpod, selNode, j) ? '故障' : 'rank ' + (selNode * NPN + j) }));
      else if (selLevel === 'die') items = [
        ...Array.from({ length: COMPUTE_DIES_PER_CARD }, (_, d) => ({ val: dieVal(d), kind: 'diec', idx: d, label: `计算 Die ${d}`, sub: '≈16 Core-Group · UMA' })),
        ...Array.from({ length: IO_DIES_PER_CARD }, (_, d) => ({ val: 0, kind: 'iodie', idx: d, label: `IO Die ${d}`, color: ENTITY_COLORS.ioDie, sub: '互联/IO · 无算力负载' })),
      ];
      else if (selLevel === 'core') items = Array.from({ length: CORES_PER_CARD }, (_, c) => ({ val: coreVal(c), kind: 'corec', idx: c, label: c % 8 === 7 ? `AIV${c}` : `AIC${c}`, sub: c % 8 === 7 ? 'Vector' : 'Cube' }));
      if (items) {
        const cap = selLevel === 'cluster' ? `1 集群 = ${PODS_PER_CLUSTER} Pod = ${POOLS_PER_CLUSTER} 服务池 · 在线 ${pods} Pod · 灰=未接入 · 点击在线池下钻`
          : selLevel === 'pool' ? `×${PODS_PER_POOL} Pod / 池 · 在线 ${podsInPool} · 灰=未上线 · 点击在线 Pod 下钻` : '';
        drawUniform(items, topY, cap ? 14 : 0);
        if (cap) tx(cap, PAD, H - 8, P.mut, '11px Inter');
        return;
      }
      drawFull(topY);   // rank → 全量铺开（铺满）
    }
    // Pod 级 Host 大网格：采样至多 HOST_GRID_MAX 台 Host，超出显示 +N；点击 → node（下钻 Host）
    function drawHostGrid(topY: number) {
      const relOn = relHi && selNpu >= 0;
      const availW = W - 2 * PAD, availH = H - topY - PAD - 18 - (relOn ? 14 : 0);
      const shown = Math.min(NODES, HOST_GRID_MAX);
      const cols = Math.max(8, Math.round(Math.sqrt(shown * availW / Math.max(1, availH))));
      const rows = Math.ceil(shown / cols);
      const cell = Math.max(4, Math.min(Math.floor(availW / cols), Math.floor(availH / rows)));
      const gp = cell > 8 ? 2 : 1;
      const x0 = PAD + Math.max(0, (availW - cols * cell) / 2), y0 = topY;
      for (let i = 0; i < shown; i++) {
        const c = i % cols, r = (i / cols) | 0, x = x0 + c * cell, y = y0 + r * cell, v = nodeMean(selSpod, i);
        fbox(x, y, cell - gp, cell - gp, loadColor(v));
        if (i === selNode) { ctx.strokeStyle = ACCENT; ctx.lineWidth = 2; ctx.strokeRect(x + 0.5, y + 0.5, cell - gp - 1, cell - gp - 1); }
        cells.current.push({ x, y, w: cell - gp, h: cell - gp, kind: 'nodec', idx: i });
      }
      // ── 关系高亮（B2）：把含 peer 卡的 Host 格按维度描边（parallelMap 真值 · 采样后再画） ──
      if (relOn) {
        const fk = selNode * NPN + cardJ;
        const hostsOf = (dim: 'tp' | 'pp' | 'dp' | 'ep', cap: number) => { const s = new Set<number>(); for (const k of pm.peersOf(fk, dim, cap)) s.add((k / NPN) | 0); return s; };
        const mark = (hs: Set<number>, col: string, lw: number) => { ctx.strokeStyle = col; ctx.lineWidth = lw; hs.forEach((n) => { if (n === selNode || n >= shown) return; ctx.strokeRect(x0 + (n % cols) * cell + 0.5, y0 + ((n / cols) | 0) * cell + 0.5, cell - gp - 1, cell - gp - 1); }); };
        mark(hostsOf('dp', 32), PARALLEL_COLORS.dp, 1.1);   // DP 副本所在 Host（采样≤32）
        mark(hostsOf('ep', 32), PARALLEL_COLORS.ep, 1.1);   // EP 组所在 Host（推理=本 Host 内）
        mark(hostsOf('pp', 16), PP_HI, 1.6);                // PP stage peer 所在 Host（青避让状态绿）
        // TP = 本 Host（选中 Host 本身）→ 焦点白描边（其 ACCENT 选中框与 TP 靛蓝同色）
        focusRing(x0 + (selNode % cols) * cell + 0.5, y0 + ((selNode / cols) | 0) * cell + 0.5, cell - gp - 1, cell - gp - 1);
        relLegend(H - 22);
      }
      const extra = NODES - shown;
      tx(`${NODES.toLocaleString()} 个 Host（每格=1 Host · 颜色=负载）${extra > 0 ? ` · +${extra.toLocaleString()} 采样外` : ''} · 点击下钻 Host`, PAD, H - 8, P.mut, '11px Inter');
    }
    function drawUniform(items: { val: number; kind: string; idx: number; label: string; color?: string; sub?: string }[], topY: number, botExtra = 0) {
      const n = items.length, cols = n <= 4 ? 2 : n <= 16 ? 4 : n <= 64 ? 8 : Math.ceil(Math.sqrt(n * 1.6));
      const rows = Math.ceil(n / cols), GP = 6;
      const cw = (W - 2 * PAD - (cols - 1) * GP) / cols, ch = (H - topY - PAD - botExtra - (rows - 1) * GP) / rows;
      items.forEach((it, i) => {
        const c = i % cols, r = (i / cols) | 0, x = PAD + c * (cw + GP), y = topY + r * (ch + GP);
        const fill = it.color ?? loadColor(it.val);
        fbox(x, y, cw, ch, fill);
        const picked = (it.kind === 'npuc' && it.idx === selNpu) || (it.kind === 'corec' && it.idx === selCore);
        if (picked) { ctx.strokeStyle = ACCENT; ctx.lineWidth = 3; ctx.strokeRect(x + 1.5, y + 1.5, cw - 3, ch - 3); }
        const ink = inkOf(fill);
        if (cw >= 58 && ch >= 26) {
          tx(it.label, x + 10, y + 22, ink, '600 13px Inter');
          if (it.color) tx(it.sub ?? '', x + 10, y + 40, ink, '10px Inter');
          else tx(Math.round(it.val * 100) + '%', x + 10, y + 42, ink, `15px ${MONO}`);
          if (it.sub && !it.color && ch >= 64) tx(it.sub, x + 10, y + 60, ink, '10px Inter');
        } else if (cw >= 26 && ch >= 18) tx(it.label || (Math.round(it.val * 100) + ''), x + 4, y + 14, ink, '9px Inter');
        cells.current.push({ x, y, w: cw, h: ch, kind: it.kind, idx: it.idx });
      });
    }
    // 全量铺满：每节点=4×2 卡块；按可用区长宽比选列数，cell 尺寸填满（容许很小）
    function drawFull(topY: number) {
      const relOn = relHi && selNpu >= 0;
      const availW = W - 2 * PAD, availH = H - topY - PAD - (relOn ? 30 : 16);
      const cols = Math.max(8, Math.min(NODES, Math.round(Math.sqrt(0.6 * NODES * availW / Math.max(1, availH)))));
      const rows = Math.ceil(NODES / cols);
      const cs = Math.max(1.5, Math.min(availW / (cols * 5), availH / (rows * 3)));
      const sx = 5 * cs, sy = 3 * cs, cg = Math.max(0.8, cs - 0.4);
      const usedW = cols * sx - cs, usedH = rows * sy - cs;
      const x0 = PAD + Math.max(0, (availW - usedW) / 2), y0 = topY + Math.max(0, (availH - usedH) / 2);
      for (let gn = 0; gn < NODES; gn++) {
        const nx = x0 + (gn % cols) * sx, ny = y0 + ((gn / cols) | 0) * sy;
        for (let j = 0; j < NPN; j++) { ctx.fillStyle = loadColor(metricVal(selSpod, gn, j)); ctx.fillRect(nx + (j % 4) * cs, ny + ((j / 4) | 0) * cs, cg, cg); }
        if (gn === selNode) { ctx.strokeStyle = ACCENT; ctx.lineWidth = 1.6; ctx.strokeRect(nx - 1.2, ny - 1.2, 4 * cs, 2 * cs); }
        cells.current.push({ x: nx, y: ny, w: 4 * cs, h: 2 * cs, kind: 'rankfull', idx: gn });
      }
      // ── 关系高亮（B1）：选中卡 → 用 pm.peersOf 给真实格子按维度描边（采样后再画，不遍历全量） ──
      if (relOn) {
        const fk = selNode * NPN + cardJ;
        const posOf = (k: number): [number, number] => { const gn = (k / NPN) | 0, j = k % NPN; return [x0 + (gn % cols) * sx + (j % 4) * cs, y0 + ((gn / cols) | 0) * sy + ((j / 4) | 0) * cs]; };
        const ringCards = (ks: number[], col: string, lw: number) => { ctx.strokeStyle = col; ctx.lineWidth = lw; for (const k of ks) { if (k === fk || k >= NPU_TOT) continue; const [x, y] = posOf(k); ctx.strokeRect(x - 0.6, y - 0.6, cg + 1.2, cg + 1.2); } };
        ringCards(pm.peersOf(fk, 'dp', 32), PARALLEL_COLORS.dp, 1.1);          // DP 副本（采样≤32）
        ringCards(pm.peersOf(fk, 'ep', 32), PARALLEL_COLORS.ep, 1.1);          // EP 组
        ringCards(pm.peersOf(fk, 'pp', 16).slice(0, 16), PP_HI, 1.4);          // PP 链（采样≤16 · 青避让状态绿）
        ringCards(pm.peersOf(fk, 'tp'), PARALLEL_COLORS.tp, 1.4);              // TP 8 卡 = 本 Host
        const [fx, fy] = posOf(fk); focusRing(fx - 1.2, fy - 1.2, cg + 2.4, cg + 2.4);   // 焦点卡白色粗描边
        relLegend(H - 22);
      }
      tx(`全量 ${NPU_TOT.toLocaleString()} 张 Chip·NPU（${cols}×${rows} Host 块铺满 · 每格=1 Chip）· 点击下钻 Host`, PAD, H - 8, P.mut, '11px Inter');
    }

    // ════════ 互联流量：带宽条在上，通信矩阵区域自适应放大为主体 ════════
    function drawFlow(topY: number) {
      const [N, label] = flowCfg();
      tx(`${scopeName()} · ${label}`, PAD, topY + 16, P.ink2, '13px Inter');
      // plane-bandwidth bars: compact row at the TOP (full width)
      const pbY = topY + 32, pbW = (W - 2 * PAD - 3 * 12) / 4;
      planeUtil().forEach((p, i) => { const x = PAD + i * (pbW + 12); tx(p.n.split('·')[0].split('(')[0], x, pbY - 4, P.ink2, '9.5px Inter'); bar(x, pbY, pbW, 13, p.u); tx(Math.round(p.u * 100) + '%', x + pbW - 28, pbY + 11, inkOf(loadColor(p.u)), `9px ${MONO}`); });
      // matrix = main element, area-adaptive: fill the remaining area below the bars (square, centred)
      const mTop = pbY + 32, mLeft = PAD + 26, matH = H - 22 - mTop, matW = W - mLeft - PAD;
      const cs = Math.max(2, Math.min(56, Math.floor(Math.min(matH, matW) / N)));
      const m = N * cs, mx = mLeft + Math.max(0, (matW - m) / 2), my = mTop + Math.max(0, (matH - m) / 2);
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { ctx.fillStyle = loadColor(tcell(i, j, N)); ctx.fillRect(mx + j * cs, my + i * cs, cs - (cs > 5 ? 1 : 0.4), cs - (cs > 5 ? 1 : 0.4)); }
      const lab = N <= 8 ? 1 : N <= 32 ? 4 : 16;
      for (let i = 0; i < N; i += lab) { tx('' + (i + 1), mx + i * cs, my - 3, P.mut, `9px ${MONO}`); tx('' + (i + 1), mx - 22, my + i * cs + cs - 1, P.mut, `9px ${MONO}`); }
      const patNote = phase === 'decode' ? 'Decode·EP All-to-All → 密集 any-to-any + 行/列负载纹理（无对角 · 回放流动）'
        : phase === 'prefill' ? 'Prefill·计算为主+EP A2A → 近对角软带 + 负载纹理'
        : '预训练·DP Ring 邻近带 + EP A2A 底噪 + 负载纹理';
      tx(`行/列 = 通信单元 · ${patNote} · 颜色=通信强度(状态色)` + (SUBCARD.includes(selLevel) ? ' · 卡内片上 NoC 无跨卡矩阵，显示所属 Host' : ''), mLeft, H - 8, P.mut, '10.5px Inter');
    }

    // ════════ N×N 通信矩阵（需求4）：rank/Host/Pod 三档聚合 · 按 phase(并行维) 过滤 ════════
    // 矩阵形态即通信模式指纹：TP=对角块状 · EP AllToAll=簇间棋盘 · DP=环带 · PP=次对角线
    function matPattern(dim: 'TP' | 'PP' | 'EP' | 'DP', i: number, j: number, N: number): number {
      if (i === j) return 0;
      const blk = Math.max(2, N / 8);
      if (dim === 'TP') return ((i / blk) | 0) === ((j / blk) | 0) ? 0.85 : 0.06;          // 对角块状（组内 AllReduce）
      if (dim === 'PP') { const d = Math.abs(i - j); return d === 1 ? 0.85 : d === N - 1 ? 0.4 : 0.05; }  // 次对角线（stage 接力）
      if (dim === 'EP') {                                                                    // 簇间棋盘（A2A 域同位互发）
        const ci = (i / blk) | 0, cj = (j / blk) | 0;
        if (ci !== cj && i % blk === j % blk) return 0.85;
        return ci === cj ? 0.3 : 0.05;
      }
      const s = Math.max(1, (N / 8) | 0), d2 = (j - i + N) % N;                              // 环带（Ring 步进 stride）
      return (d2 === s || d2 === N - s) ? 0.8 : (d2 === 1 || d2 === N - 1) ? 0.3 : 0.05;
    }
    function drawMatrix(topY: number) {
      const aggCfg = { pod: { N: 16, unit: 'Pod', lab: 'Pod×Pod（16 Pod 采样）' }, host: { N: 64, unit: 'Host', lab: `Host×Host（Pod#${selSpod + 1} 采样 64 Host）` }, rank: { N: 64, unit: 'rank', lab: `rank×rank（机柜${((selNode / NODES_PER_CAB) | 0) + 1} · 8 Host×8 rank）` } }[matAgg];
      const N = aggCfg.N;
      tx(`N×N 通信矩阵 · ${aggCfg.lab} · phase=${matDim} · 矩阵形态=通信模式指纹`, PAD, topY + 15, P.ink2, '12.5px Inter');
      // 档位 + phase 过滤 chips
      let bx = PAD;
      const chipBtn = (label: string, on: boolean, kind: string, idx: number, col?: string) => {
        ctx.font = '10.5px Inter'; const w = ctx.measureText(label).width + 18;
        fbox(bx, topY + 24, w, 20, on ? (col ?? ACCENT) : (dark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.04)'));
        tx(label, bx + 9, topY + 38, on ? inkOf(col ?? ACCENT) : P.ink2, '10.5px Inter');
        cells.current.push({ x: bx, y: topY + 24, w, h: 20, kind, idx });
        bx += w + 6;
      };
      (['rank', 'host', 'pod'] as const).forEach((a, i) => chipBtn(a === 'rank' ? 'rank×rank' : a === 'host' ? 'Host×Host' : 'Pod×Pod', matAgg === a, 'magg', i));
      bx += 10;
      (['TP', 'PP', 'EP', 'DP'] as const).forEach((d, i) => chipBtn(d, matDim === d, 'mdim', i, PARALLEL_COLORS[d.toLowerCase() as 'tp' | 'pp' | 'ep' | 'dp']));
      // 矩阵主体（正方形自适应铺满，右侧留指纹图例宽度）
      const LEG_W = 148, mTop = topY + 56, mLeft = PAD + 26;
      const matH = H - 26 - mTop, matW = W - mLeft - PAD - LEG_W;
      const cs = Math.max(2, Math.floor(Math.min(matH, matW) / N));
      const m = N * cs, mx = mLeft, my = mTop + Math.max(0, (matH - m) / 2);
      const salt = selSpod * 3.1 + (matAgg === 'rank' ? selNode * 0.7 : 0);
      const val = (i: number, j: number) => {
        const base = matPattern(matDim, i, j, N);
        const tex = (rnd(i * 3.3 + salt) - 0.5) * 0.14 + (rnd(j * 2.9 + salt + 4) - 0.5) * 0.14;
        const live = (rnd(i * 0.7 + j * 0.9 + step * 0.06 + flowRef.current * 0.7) - 0.5) * 0.22;
        const hot = ev && selSpod === 0 && matAgg !== 'pod' && (i === (EVT_CAB * 2) % N || j === (EVT_CAB * 2) % N) ? 0.3 : 0;
        return clamp01(base + tex + live + hot);
      };
      const tops: { i: number; j: number; v: number }[] = [];
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const v = val(i, j);
        ctx.fillStyle = i === j ? (dark ? '#10141b' : '#e8ebf2') : loadColor(v);
        ctx.globalAlpha = i === j ? 1 : 0.28 + v * 0.72;
        ctx.fillRect(mx + j * cs, my + i * cs, cs - (cs > 5 ? 1 : 0.4), cs - (cs > 5 ? 1 : 0.4));
        if (i !== j) { tops.push({ i, j, v }); if (tops.length > 400) tops.sort((a, b) => b.v - a.v).length = 200; }
      }
      ctx.globalAlpha = 1;
      cells.current.push({ x: mx, y: my, w: m, h: m, kind: 'mcell', idx: 0 });
      // Top-N 热点红框（与 Davis 问题卡联动：通信热点可成为根因证据）
      tops.sort((a, b) => b.v - a.v);
      ctx.strokeStyle = '#e5484d'; ctx.lineWidth = Math.max(1.2, cs * 0.12);
      tops.slice(0, 5).forEach(({ i, j }) => ctx.strokeRect(mx + j * cs - 0.5, my + i * cs - 0.5, cs, cs));
      const labStep = N <= 16 ? 1 : 8;
      for (let i = 0; i < N; i += labStep) { tx('' + i, mx + i * cs + 1, my - 3, P.mut, `8.5px ${MONO}`); tx('' + i, mx - 22, my + i * cs + cs, P.mut, `8.5px ${MONO}`); }
      // 指纹图例：四种矩阵形态小样（8×8 纯模式，不加噪）
      const lx = W - PAD - LEG_W + 10; let ly = mTop + 4;
      tx('指纹图例', lx, ly, P.ink, '700 10.5px Inter'); ly += 8;
      (([['TP', '对角块状 · 组内 AllReduce'], ['EP', '簇间棋盘 · A2A 同位互发'], ['DP', '环带 · Ring 步进'], ['PP', '次对角线 · stage 接力']]) as ['TP' | 'EP' | 'DP' | 'PP', string][]).forEach(([d, note]) => {
        const tcs = 5.5, tn = 8;
        for (let i = 0; i < tn; i++) for (let j = 0; j < tn; j++) {
          const v = matPattern(d, i, j, tn);
          ctx.fillStyle = i === j ? (dark ? '#10141b' : '#e8ebf2') : loadColor(v);
          ctx.globalAlpha = i === j ? 1 : 0.25 + v * 0.75;
          ctx.fillRect(lx + j * tcs, ly + 4 + i * tcs, tcs - 0.6, tcs - 0.6);
        }
        ctx.globalAlpha = 1;
        if (matDim === d) { ctx.strokeStyle = ACCENT; ctx.lineWidth = 1.6; ctx.strokeRect(lx - 2, ly + 2, tn * tcs + 4, tn * tcs + 4); }
        tx(d, lx + tn * tcs + 8, ly + 18, PARALLEL_COLORS[d.toLowerCase() as 'tp' | 'pp' | 'ep' | 'dp'], '700 10px Inter');
        tx(note.split(' · ')[0], lx + tn * tcs + 8, ly + 30, P.ink2, '9px Inter');
        tx(note.split(' · ')[1], lx + tn * tcs + 8, ly + 41, P.mut, '8.5px Inter');
        ly += tn * tcs + 14;
      });
      tx('红框=Top-5 热点', lx, ly + 8, '#e5484d', '9.5px Inter');
      tx('(通信热点=Davis 根因证据)', lx, ly + 20, P.mut, '8.5px Inter');
      tx(`点矩阵格 → 定位对应${aggCfg.unit}对并下钻（联动详情栏 · 物理路径见 集群驾驶舱·需求1）· 颜色=通信强度(状态色) · 回放时逐帧流动`, PAD, H - 8, P.mut, '10px Inter');
      // hover/click 用：把几何参数塞进闭包外的 ref（复用 cells 命中后换算行列）
      matGeom.current = { mx, my, cs, N };
    }

    // ════════ 通信域：每个并行维度画真实集合通信图元 ════════
    function drawDomain(topY: number) {
      tx(`通信域(进程↔进程) · ${scopeName()} · 进程 = rank = 1 NPU（硬件↔进程 1:1）`, PAD, topY + 16, P.ink2, '12.5px Inter');
      tx('上＝概念图(域包含 TP⊂EP⊂DP) · 下＝真实对象拓扑', W - PAD, topY + 16, P.mut, '10px Inter', 'right');
      const onchip = SUBCARD.includes(selLevel);
      const lvlNote = onchip ? '已下钻到卡内：设备内并行 = block_idx / SPMD（核实例），非 rank 间集合通信'
        : selLevel === 'node' ? `本 Host ${NPN} rank = 1 TP 组（域内 AllReduce）`
        : selLevel === 'global' ? `全球：跨集群 DCN 调度（每集群 = 若干 DP 副本）`
        : selLevel === 'pool' ? `服务池：池内 ${podsInPool} Pod 间互联（DP/PP 跨 Pod）`
        : selLevel === 'cluster' ? `每 Pod = 1 个 DP 副本（跨 Pod AllReduce）`
        : `本 Pod：TP/EP 在域内、DP/PP 跨域`;
      tx(lvlNote + ' · 嵌套框=域包含(TP⊂EP⊂DP) · 内嵌图标=集合通信形态 · 颜色=状态 · 蓝框=当前选区所在域', PAD, topY + 34, P.mut, '10.5px Inter');
      const D = domains(), act = domActive();
      const top = topY + 48, rh = (H - top - 12) / D.length;
      const availH2 = (H - 16) - (top + 4);
      const conceptH = Math.max(190, Math.round(availH2 * 0.46));
      // collective-pattern glyph among k representative ranks
      const glyph = (cx: number, cy: number, rad: number, pat: 'ring' | 'a2a' | 'p2p', col: string, on: boolean) => {
        const k = pat === 'p2p' ? 5 : 7;
        const pts: [number, number][] = pat === 'p2p'
          ? Array.from({ length: k }, (_, i) => [cx - rad + (i * 2 * rad) / (k - 1), cy] as [number, number])
          : Array.from({ length: k }, (_, i) => { const t = -Math.PI / 2 + (i * 2 * Math.PI) / k; return [cx + rad * Math.cos(t), cy + rad * Math.sin(t)] as [number, number]; });
        const lc = on ? col : P.neutral;
        if (pat === 'a2a') { for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) line(pts[i][0], pts[i][1], pts[j][0], pts[j][1], lc, on ? 1.4 : 0.8, on); }
        else if (pat === 'ring') { for (let i = 0; i < k; i++) { const a = pts[i], b = pts[(i + 1) % k]; line(a[0], a[1], b[0], b[1], lc, on ? 2 : 1, on); } }
        else { for (let i = 0; i < k - 1; i++) line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], lc, on ? 2 : 1, on); }
        pts.forEach((p) => { ctx.fillStyle = on ? lc : P.neutral; ctx.beginPath(); ctx.arc(p[0], p[1], on ? 4 : 3, 0, 7); ctx.fill(); });
      };
      void rh;
      type Dom = (typeof D)[number];
      const byKey: Record<string, Dom> = {}; D.forEach((d) => { byKey[d.key] = d; });
      const dp = byKey.dp, ep = byKey.ep, tp = byKey.tp, pp = byKey.pp;

      // 片上：设备内并行(block_idx/SPMD)，无 rank 间集合通信 → 说明并停止
      if (onchip) {
        const bx = PAD, by = top + 8, bw = W - 2 * PAD, bh = 118;
        fbox(bx, by, bw, bh, P.neutral); ctx.strokeStyle = P.frame; ctx.lineWidth = 1; ctx.strokeRect(bx, by, bw, bh);
        tx('片上（计算 Die / Core-Group / Tile）= 设备内并行', bx + 14, by + 28, P.ink, '700 13px Inter');
        tx('block_idx · SPMD 核实例 · rank 内不增 rank —— 无 rank↔rank 集合通信（TP/EP/DP/PP 均在卡之上）', bx + 14, by + 52, P.ink2, '11px Inter');
        tx('要看集合通信域，请上钻到 Host(TP 域) / Pod(EP 域) / 集群(DP 域)', bx + 14, by + 76, P.mut, '10.5px Inter');
        return;
      }

      // 当前选区落在哪个域 → 高亮该域框
      const hi = (key: string) => (selLevel === 'node' && key === 'tp') || (selLevel === 'super' && key === 'ep') || ((selLevel === 'cluster' || selLevel === 'global' || selLevel === 'pool') && (key === 'dp' || key === 'pp'));
      // 一个「域框」：状态色描边 + 内嵌集合通信图标(抽象图元作说明) + 名称/scope/集合/成员 + 流量条
      const domainBox = (x: number, y: number, w: number, h: number, d: Dom, tag: string) => {
        const on = act[d.key], col = on ? loadColor(d.u) : P.neutral;
        ctx.globalAlpha = on ? 1 : 0.5;
        ctx.fillStyle = on ? (dark ? 'rgba(255,255,255,0.025)' : 'rgba(0,0,0,0.012)') : 'transparent'; ctx.fillRect(x, y, w, h);
        ctx.strokeStyle = col; ctx.lineWidth = on ? 2 : 1; ctx.setLineDash(on ? [] : [4, 3]); ctx.strokeRect(x, y, w, h); ctx.setLineDash([]);
        if (hi(d.key)) { ctx.strokeStyle = ACCENT; ctx.lineWidth = 3; ctx.strokeRect(x + 2, y + 2, w - 4, h - 4); }
        glyph(x + 24, y + 26, 13, d.pat, col, on);   // 内嵌抽象图元＝该域集合通信形态的说明
        tx(d.nm, x + 46, y + 20, on ? P.ink : P.mut, '700 12.5px Inter');
        tx(tag, x + 46, y + 34, on ? P.ink2 : P.mut, '9.5px Inter');
        tx(`${d.co} · ${d.me}`, x + 46, y + 47, on ? P.ink2 : P.mut, '9px Inter');
        if (on) { const bw2 = Math.min(110, w * 0.4); bar(x + w - bw2 - 12, y + 12, bw2, 10, d.u); tx(Math.round(d.u * 100) + '%', x + w - 44, y + 20, inkOf(col), `9px ${MONO}`); }
        ctx.globalAlpha = 1;
      };

      // ── 嵌套：DP 副本(SO) ⊃ EP 域(SU) ⊃ 多个 TP 组 ；PP 在右侧贯穿；SP 与 TP 同域 ──
      const A = { x: PAD, y: top + 4, w: W - 2 * PAD, h: conceptH };
      const gutter = 92;
      domainBox(A.x, A.y, A.w, A.h, dp, 'SO 广域 · 跨 Pod（全光 scale-out）');
      const ep0 = { x: A.x + 18, y: A.y + 64, w: A.w - 36 - gutter, h: A.h - 64 - 26 };
      domainBox(ep0.x, ep0.y, ep0.w, ep0.h, ep, 'SU 超低延迟 · Pod 内全互联（scale-up）');
      // EP 内：一排 TP 组（代表性 4 个 + ×K 说明）
      const nTP = 4, tgTop = ep0.y + 62, tgH = Math.max(56, Math.min(96, ep0.h - 78)), tgGap = 10;
      const tgW = (ep0.w - 24 - (nTP - 1) * tgGap) / nTP;
      const onTP = act.tp;
      for (let g = 0; g < nTP; g++) {
        const gx = ep0.x + 12 + g * (tgW + tgGap);
        ctx.globalAlpha = onTP ? 1 : 0.5;
        ctx.strokeStyle = onTP ? loadColor(tp.u) : P.neutral; ctx.lineWidth = (selLevel === 'node' && g === 0) ? 2.5 : 1.2;
        ctx.setLineDash(onTP ? [] : [3, 3]); ctx.strokeRect(gx, tgTop, tgW, tgH); ctx.setLineDash([]);
        glyph(gx + tgW / 2, tgTop + tgH / 2 + 2, Math.min(22, tgW / 2 - 8), 'ring', loadColor(tp.u), onTP);
        tx(g === 0 ? `TP 组·${NPN}rank` : 'TP 组', gx + tgW / 2, tgTop + 14, onTP ? P.ink2 : P.mut, '9.5px Inter', 'center');
        ctx.globalAlpha = 1;
      }
      const K = Math.max(1, Math.round(NPC / NPN));
      tx(`… ×${K} TP 组（每组=1 Host ${NPN} Chip · AllReduce）· SP 与 TP 同域（AllGather+ReduceScatter）`, ep0.x + 12, tgTop + tgH + 15, P.mut, '9.5px Inter');
      // PP：右侧竖向流水（stage→stage P2P，贯穿 EP 之间）
      const onPP = act.pp, pcx = A.x + A.w - gutter / 2 - 4, y1 = A.y + 84, y2 = A.y + A.h - 30, stages = 5;
      ctx.globalAlpha = onPP ? 1 : 0.5;
      tx('PP 流水', pcx, A.y + 74, onPP ? P.ink : P.mut, '700 10px Inter', 'center');
      for (let s = 0; s < stages; s++) {
        const yy = y1 + s * (y2 - y1) / (stages - 1);
        if (s < stages - 1) line(pcx, yy + 5, pcx, y1 + (s + 1) * (y2 - y1) / (stages - 1) - 5, onPP ? loadColor(pp.u) : P.neutral, onPP ? 2 : 1, false, onPP);
        ctx.fillStyle = onPP ? loadColor(pp.u) : P.neutral; ctx.beginPath(); ctx.arc(pcx, yy, 4.5, 0, 7); ctx.fill();
      }
      tx('stage→stage', pcx, y2 + 14, onPP ? P.ink2 : P.mut, '9px Inter', 'center'); tx('P2P', pcx, y2 + 25, P.mut, '8.5px Inter', 'center');
      ctx.globalAlpha = 1;
      // SU/SO 分界说明
      tx('SU 超低延迟域(TP/EP · 域内) ↑   ↓ SO 广域(DP/PP · 跨 Pod/全光)', A.x + 14, A.y + A.h - 9, P.mut, '9.5px Inter');

      // ── 真实对象上的关系：把 TP/SP/EP/PP/DP 集合通信直接画在真实 Chip / Host / Pod 上（parallelMap 真值）──
      const entY = A.y + conceptH + 12;
      const entH = (H - 16) - entY;
      if (entH >= 74) {
        tx('真实对象上的拓扑 · TP/SP/EP/PP/DP 集合通信画在真实 Chip / Host / Pod 上（parallelMap 真值 · 颜色=状态）', PAD, entY + 12, P.ink, '700 11px Inter');
        const rTop = entY + 24, annH = 28, rH = Math.max(30, entH - 24 - annH), cyMid = rTop + rH / 2;
        // 集合通信拓扑连线：环(ring)/全互联(a2a)/流水(chain) —— 弧线在对象上方，闭合线在下方
        const arc = (x1: number, x2: number, y: number, lift: number, col: string, w: number, on: boolean) => {
          ctx.strokeStyle = on ? col : P.neutral; ctx.globalAlpha = on ? 0.9 : 0.4; ctx.lineWidth = w;
          ctx.beginPath(); ctx.moveTo(x1, y); ctx.quadraticCurveTo((x1 + x2) / 2, y - lift, x2, y); ctx.stroke(); ctx.globalAlpha = 1;
        };
        const ring = (cxs: number[], yTop: number, yBot: number, col: string, on: boolean) => {
          const gap = Math.min(20, (rTop - (A.y + conceptH)) - 2);
          for (let i = 0; i < cxs.length - 1; i++) arc(cxs[i], cxs[i + 1], yTop, gap, col, on ? 2 : 1, on);
          if (cxs.length > 2) { ctx.strokeStyle = on ? col : P.neutral; ctx.globalAlpha = on ? 0.7 : 0.35; ctx.lineWidth = on ? 2 : 1; ctx.beginPath(); ctx.moveTo(cxs[0], yBot + 3); ctx.lineTo(cxs[cxs.length - 1], yBot + 3); ctx.stroke(); ctx.globalAlpha = 1; }
        };
        const mesh = (cxs: number[], yTop: number, col: string, on: boolean) => {
          const cap = Math.min(26, (rTop - (A.y + conceptH)) - 2);
          for (let i = 0; i < cxs.length; i++) for (let j = i + 1; j < cxs.length; j++) arc(cxs[i], cxs[j], yTop, Math.min(cap, 9 + (j - i) * 6), col, on ? 1.1 : 0.6, on);
        };
        const lbl = (x: number, y: number, s: string, u: number) => { if (rH >= 22) tx(s, x, y, inkOf(loadColor(u)), '8.5px Inter', 'center'); };

        if (selLevel === 'node' || selLevel === 'rank') {
          // Host 级：本 Host 8 张真实 Chip = 1 个 TP 组（AllReduce 环）
          const cW = (W - 2 * PAD - (NPN - 1) * 4) / NPN, cxs: number[] = [];
          for (let j = 0; j < NPN; j++) {
            const x = PAD + j * (cW + 4), u = util01(selSpod, selNode, j); cxs.push(x + cW / 2);
            const sel = selLevel === 'rank' && j === cardJ;
            ctx.fillStyle = loadColor(u); ctx.fillRect(x, rTop, cW, rH);
            ctx.strokeStyle = sel ? ACCENT : (act.tp ? PARALLEL_COLORS.tp : P.frame); ctx.lineWidth = sel ? 2.6 : 1.3; ctx.strokeRect(x, rTop, cW, rH);
            if (cW >= 22) lbl(x + cW / 2, cyMid + 3, `r${selNode * NPN + j}`, u);
          }
          ring(cxs, rTop, rTop + rH, PARALLEL_COLORS.tp, act.tp);
          const ann = rTop + rH + 15;
          tx(`TP×${pm.tp} AllReduce 环 —— 本 Host ${NPN} 张真实 Chip 组成 1 个 TP 组（SP 与 TP 同域 · AllGather+ReduceScatter）`, PAD, ann, PARALLEL_COLORS.tp, '9.5px Inter');
          tx(`更大的域：EP×${pm.ep}（Pod 内 · 上钻到 Pod 看主机间全互联）· DP×${pm.dp} / PP×${pm.pp}（跨 Pod · 上钻到集群看 Pod 间关系）`, PAD, ann + 13, P.mut, '9px Inter');
        } else if (selLevel === 'super') {
          // Pod 级：真实 Host 盒（每盒=1 TP 组）→ EP 全互联 (AllToAll)
          const nH = Math.min(NODES, 8), hGap = 8, hW = (W - 2 * PAD - (nH - 1) * hGap) / nH, cxs: number[] = [];
          for (let hh = 0; hh < nH; hh++) {
            const hx = PAD + hh * (hW + hGap), u = nodeMean(selSpod, hh); cxs.push(hx + hW / 2);
            ctx.fillStyle = loadColor(u); ctx.globalAlpha = 0.92; ctx.fillRect(hx, rTop, hW, rH); ctx.globalAlpha = 1;
            ctx.strokeStyle = act.tp ? PARALLEL_COLORS.tp : P.frame; ctx.lineWidth = 1.5; ctx.strokeRect(hx, rTop, hW, rH);
            if (hW >= 34) { lbl(hx + hW / 2, cyMid - 2, `Host ${hh}`, u); lbl(hx + hW / 2, cyMid + 11, `TP×${pm.tp} 组`, u); }
          }
          mesh(cxs, rTop, PARALLEL_COLORS.ep, act.ep);
          if (NODES > nH) tx(`+${NODES - nH} Host`, W - PAD, rTop + rH + 15, P.mut, '9px Inter', 'right');
          const ann = rTop + rH + 15;
          tx(`EP×${pm.ep} 全互联 AllToAll —— 真实 Host 之间 Pod 内 SU（scale-up 超低延迟）· 每台 Host = 1 个 TP 组`, PAD, ann, PARALLEL_COLORS.ep, '9.5px Inter');
          tx(`DP×${pm.dp} / PP×${pm.pp} 跨 Pod（上钻到集群看 Pod 间 DP 环 / PP 流水链）`, PAD, ann + 13, P.mut, '9px Inter');
        } else {
          // 集群/池/全球：真实 Pod 盒（每盒=1 DP 副本）→ DP 环 (AllReduce) + PP 流水链
          const nP = Math.min(pods, 8), pGap = 10, pW = (W - 2 * PAD - (nP - 1) * pGap) / nP, cxs: number[] = [];
          for (let pi = 0; pi < nP; pi++) {
            const px = PAD + pi * (pW + pGap), u = spodMean(pi); cxs.push(px + pW / 2);
            ctx.fillStyle = loadColor(u); ctx.globalAlpha = 0.92; ctx.fillRect(px, rTop, pW, rH); ctx.globalAlpha = 1;
            ctx.strokeStyle = act.dp ? PARALLEL_COLORS.dp : P.frame; ctx.lineWidth = 1.7; ctx.strokeRect(px, rTop, pW, rH);
            if (pW >= 36) { lbl(px + pW / 2, cyMid - 2, `Pod ${pi + 1}`, u); lbl(px + pW / 2, cyMid + 11, '1 DP 副本', u); }
          }
          ring(cxs, rTop, rTop + rH, PARALLEL_COLORS.dp, act.dp);
          if (act.pp && cxs.length > 1) for (let i = 0; i < cxs.length - 1; i++) line(cxs[i], rTop + rH + 6, cxs[i + 1], rTop + rH + 6, PP_HI, 2, false, true);
          if (pods > nP) tx(`+${pods - nP} Pod`, W - PAD, rTop + rH + 15, P.mut, '9px Inter', 'right');
          const ann = rTop + rH + (act.pp ? 16 : 15);
          tx(`DP×${pm.dp} AllReduce 环 —— 每个真实 Pod = 1 个 DP 副本 · 跨 Pod SO（scale-out 全光）`, PAD, ann, PARALLEL_COLORS.dp, '9.5px Inter');
          if (act.pp) tx(`PP×${pm.pp} 流水 P2P（下方链）· Pod 间顺序穿越（stage→stage）`, PAD, ann + 13, PP_HI, '9px Inter');
        }
      }
    }

    // ════════ 物理链路：结构随层级、每个器件/链路按自身负载上色（随回放变化）、数量真实 ════════
    function drawPhys(topY: number) {
      const Pl = planeUtil();
      // plane-utilisation bars along the bottom (full width, time-varying)
      const pbY = H - 46, pbW = (W - 2 * PAD - 3 * 12) / 4;
      Pl.forEach((p, i) => { const x = PAD + i * (pbW + 12); tx(p.n.split('·')[0].split('(')[0], x, pbY - 5, P.ink2, '9.5px Inter'); bar(x, pbY, pbW, 13, p.u); tx(Math.round(p.u * 100) + '%', x + pbW - 28, pbY + 11, inkOf(loadColor(p.u)), `9px ${MONO}`); });

      // Chip(rank) 及以下 = 卡内/片上视图（每层不同结构）；Host 及以上 = 中心交换 + 子单元
      if (selLevel === 'rank' || SUBCARD.includes(selLevel)) { drawChip(Pl, topY); return; }
      tx(`物理链路 · ${scopeName()} · 器件/链路按各自负载上色 · 随回放变化`, PAD, topY + 16, P.ink2, '12.5px Inter');

      // hub + child units (the real count), each coloured by its OWN load
      const cx = W / 2, hubY = topY + 64, areaTop = hubY + 40, areaBot = pbY - 28;
      const hubU = selLevel === 'global' ? Pl[3].u : (selLevel === 'cluster' || selLevel === 'pool') ? Pl[1].u : Pl[0].u;
      const hubLab = selLevel === 'global' ? 'DCN' : selLevel === 'cluster' ? 'Scale-Out' : selLevel === 'pool' ? 'Pool 内互联' : selLevel === 'super' ? 'Scale-Up · UB-Mesh' : 'L1 交换';

      // ── L7 全球：DCN hub + 本集群 + 2 个幽灵兄弟集群（半透明、点击切到 Cluster） ──
      if (selLevel === 'global') {
        dia(cx, hubY, 26, loadColor(hubU), hubLab);
        const sib = [{ lab: 'Global A', ghost: true }, { lab: '本集群', ghost: false }, { lab: 'Global C', ghost: true }];
        const bw = Math.min(160, (W - 2 * PAD - 2 * 34) / 3), bh = 78, gap = 34, y = areaTop + 24;
        const x0 = cx - (sib.length * bw + (sib.length - 1) * gap) / 2;
        sib.forEach((s, i) => {
          const x = x0 + i * (bw + gap), v = s.ghost ? -1 : clusterMean();
          ctx.globalAlpha = s.ghost ? 0.4 : 1;
          line(x + bw / 2, y, cx, hubY + 18, s.ghost ? P.neutral : loadColor(v), s.ghost ? 1.2 : 2.6, !s.ghost, !s.ghost);
          fbox(x, y, bw, bh, s.ghost ? P.neutral : loadColor(v));
          tx(s.lab, x + bw / 2, y + bh / 2 - 2, s.ghost ? P.mut : inkOf(loadColor(v)), '600 12px Inter', 'center');
          if (s.ghost) tx('兄弟集群（示意）', x + bw / 2, y + bh / 2 + 16, P.mut, '9px Inter', 'center');
          else tx(`${PODS_PER_CLUSTER} Pod（>52万卡）· 在线 ${pods} · ${Math.round(v * 100)}%`, x + bw / 2, y + bh / 2 + 16, inkOf(loadColor(v)), `10px ${MONO}`, 'center');
          ctx.globalAlpha = 1;
          cells.current.push({ x, y, w: bw, h: bh, kind: 'gclus', idx: i });
        });
        tx('DCN 跨地域数据中心网络 · 点击本集群下钻 Cluster', PAD, areaTop - 8, P.mut, '10px Inter');
        return;
      }

      // 真实数量：cluster = 16 服务池、pool = 4 Pod；离线（未接入/未上线）val = -1 → 灰
      const cfg = selLevel === 'cluster' ? { N: POOLS_PER_CLUSTER, val: (i: number) => (i < pools ? poolMean(i) : -1), unit: '服务池', plane: PLANES[1] }
        : selLevel === 'pool' ? { N: PODS_PER_POOL, val: (i: number) => { const g = selPool * PODS_PER_POOL + i; return g < pods ? spodMean(g) : -1; }, unit: 'Pod', plane: PLANES[1] }
        : selLevel === 'super' ? { N: NODES, val: (i: number) => nodeMean(selSpod, i), unit: 'Host', plane: PLANES[0] }
        : { N: NPN, val: (i: number) => util01(selSpod, selNode, i), unit: 'Chip·NPU', plane: PLANES[0] };
      dia(cx, hubY, 26, loadColor(hubU), hubLab);
      const isNode = selLevel === 'node';
      // 子单元的 cell kind / 绝对索引 / 标签（cluster→服务池、pool→Pod、其余沿用）
      const childKind = isNode ? 'pnpu' : selLevel === 'cluster' ? 'pool' : selLevel === 'pool' ? 'spod' : 'punit';
      const childIdx = (i: number) => selLevel === 'pool' ? selPool * PODS_PER_POOL + i : i;
      const childLab = (i: number) => selLevel === 'cluster' ? `服务池${i + 1}`
        : selLevel === 'pool' ? `Pod${selPool * PODS_PER_POOL + i + 1}`
        : selLevel === 'node' ? `${cfg.unit}${i}`
        : `${cfg.unit}${i + 1}`;

      if (cfg.N <= 16) {
        // big boxes filling the width, each linked to the hub by a line coloured by its load
        const cols = Math.min(cfg.N, 8), rows = Math.ceil(cfg.N / cols);
        const areaW = W - 2 * PAD, bw = Math.min(132, (areaW - (cols - 1) * 16) / cols), bh = Math.min(74, ((isNode ? areaBot - 64 : areaBot) - areaTop - (rows - 1) * 16) / rows);
        const x0 = cx - (cols * bw + (cols - 1) * 16) / 2;
        for (let i = 0; i < cfg.N; i++) {
          const c = i % cols, r = (i / cols) | 0, x = x0 + c * (bw + 16), y = areaTop + r * (bh + 16), v = cfg.val(i);
          const off = v < 0, fill = off ? P.neutral : loadColor(v);   // 离线灰（未接入/未上线）
          if (off) ctx.globalAlpha = 0.55;
          line(x + bw / 2, y, cx, hubY + 18, off ? P.neutral : loadColor(v), off ? 1 : 2.4, isNode, !off);
          const picked = isNode && i === selNpu;
          fbox(x, y, bw, bh, fill, picked ? ACCENT : undefined);
          if (picked) { ctx.lineWidth = 3; ctx.strokeStyle = ACCENT; ctx.strokeRect(x + 1.5, y + 1.5, bw - 3, bh - 3); }
          tx(childLab(i), x + bw / 2, y + bh / 2 - 2, off ? P.mut : inkOf(fill), '600 11px Inter', 'center');
          tx(off ? (selLevel === 'cluster' ? '未接入' : '未上线') : Math.round(v * 100) + '%', x + bw / 2, y + bh / 2 + 14, off ? P.mut : inkOf(fill), `11px ${MONO}`, 'center');
          ctx.globalAlpha = 1;
          cells.current.push({ x, y, w: bw, h: bh, kind: off ? 'unitoff' : childKind, idx: childIdx(i) });
        }
        if (isNode) {
          // 鲲鹏 CPU (UB→L1) + 擎天 NIC (VPC) — the third plane
          const cpuY = areaBot - 18, cpw = 96, ch2 = 30;
          fbox(cx - cpw - 30, cpuY - ch2, cpw, ch2, P.neutral, PLANES[0].color); tx(`${TOK.kunpeng}CPU`, cx - cpw - 30 + cpw / 2, cpuY - ch2 / 2 + 4, P.ink, '10px Inter', 'center');
          line(cx - 30, cpuY - ch2 / 2, cx, hubY + 18, loadColor(Pl[0].u), 2, false, true);
          fbox(cx + 30, cpuY - ch2, cpw, ch2, P.neutral, PLANES[2].color); tx(`${NIC_LBL}NIC`, cx + 30 + cpw / 2, cpuY - ch2 / 2 + 4, P.ink, '10px Inter', 'center');
          line(cx + 30, cpuY - ch2 / 2, cx - 30 + cpw, cpuY - ch2 / 2, loadColor(Pl[3].u), 2, true, true);
          tx('CPU→L1 = UB(绿) · CPU→NIC→数据中心 = VPC(紫)', cx, cpuY + 12, P.mut, '9px Inter', 'center');
        }
        tx(`${cfg.N} 个 ${cfg.unit}${selLevel === 'cluster' ? `（真实 ${POOLS_PER_CLUSTER} 池 = ${PODS_PER_CLUSTER} Pod · 在线 ${pods} Pod）` : selLevel === 'pool' ? `（×${PODS_PER_POOL} Pod/池 · 在线 ${podsInPool}）` : ''} · 经 ${hubLab} ${(selLevel === 'cluster' || selLevel === 'pool') ? 'scale-out 全互联' : 'UB 全互联'}`, PAD, areaTop - 8, P.mut, '10px Inter');
      } else {
        // grid sampling the REAL count (capped at HOST_GRID_MAX + "+N"), each square coloured by its own load
        const shown = Math.min(cfg.N, HOST_GRID_MAX), extra = cfg.N - shown;
        const areaW = W - 2 * PAD, areaH = areaBot - areaTop;
        const cols = Math.max(1, Math.round(Math.sqrt(shown * (areaW / Math.max(1, areaH))))), rows = Math.ceil(shown / cols);
        const cell = Math.max(3, Math.min(Math.floor(areaW / cols), Math.floor(areaH / rows))), gp = cell > 10 ? 2 : 1;
        const gw = cols * cell, x0 = cx - gw / 2, y0 = areaTop;
        const sample = Math.max(1, Math.floor(shown / 16));
        for (let i = 0; i < shown; i++) {
          const c = i % cols, r = (i / cols) | 0, x = x0 + c * cell, y = y0 + r * cell, v = cfg.val(i);
          if (i % sample === 0) line(x + cell / 2, y + cell / 2, cx, hubY + 18, loadColor(v), 0.5, false);
          fbox(x, y, cell - gp, cell - gp, loadColor(v));
          cells.current.push({ x, y, w: cell - gp, h: cell - gp, kind: 'punit', idx: i });
        }
        tx(`${cfg.N.toLocaleString()} 个 ${cfg.unit}（真实数量）${extra > 0 ? ` · 采样 ${shown.toLocaleString()}+${extra.toLocaleString()}` : ''} · 经 ${hubLab} UB-Mesh any-to-any（抽样连线）· 每格=1 ${cfg.unit}、颜色=负载`, PAD, areaBot + 14, P.mut, '10px Inter');
      }
    }
    // 卡内/片上物理视图 —— 每层不同：rank=整卡(4 Die) · die=2 计算Die(核组/NoC) · core=AI Core 阵列（L0 最深层级）
    function drawChip(Pl: { u: number }[], topY: number) {
      const pbY = H - 42, cx = W / 2, top = topY + 40, bot = pbY - 38;
      const isRank = selLevel === 'rank', isDie = selLevel === 'die', isCore = selLevel === 'core';
      tx(`物理链路 · ${scopeName()} · ${isRank ? '整卡：2 计算 Die(UMA) + 2 IO Die + 端口' : isDie ? '计算 Die（核组 / 片上 NoC）' : 'AI Core 阵列（Cube/Vector）'}`, PAD, top - 24, P.ink2, '12.5px Inter');
      const dv0 = dieVal(0), dv1 = dieVal(1), gap = 56;
      const dieW = Math.min(isRank ? 200 : 300, (W - 2 * PAD - gap - 80) / 2);
      const dieH = Math.min(isRank ? (bot - top) * 0.46 : bot - top - 20, 340);
      const lx = cx - gap / 2 - dieW, rx = cx + gap / 2, dyTop = top + 14;
      const drawComputeDie = (x: number, v: number, d: number) => {
        fbox(x, dyTop, dieW, dieH, loadColor(v), ENTITY_COLORS.computeDie);
        tx(`计算 Die ${d} · ${Math.round(v * 100)}%`, x + dieW / 2, dyTop - 6, P.ink2, '11px Inter', 'center');
        if (isCore) {                                   // 16 AI Core/die, coloured by coreVal（L0 最深层级）
          const per = CORES_PER_CARD / 2, cc = 4, cr = per / cc, iw = (dieW - 20) / cc, ih = (dieH - 36) / cr;
          for (let k = 0; k < per; k++) {
            const gi = d * per + k, c = k % cc, r = (k / cc) | 0, x2 = x + 10 + c * iw, y2 = dyTop + 26 + r * ih, v2 = coreVal(gi);
            fbox(x2 + 1, y2 + 1, iw - 3, ih - 3, loadColor(v2), gi === selCore ? ACCENT : undefined);
            if (iw > 26) tx(gi % 8 === 7 ? 'V' : 'C', x2 + iw / 2, y2 + ih / 2 + 3, inkOf(loadColor(v2)), '9px Inter', 'center');
            cells.current.push({ x: x2, y: y2, w: iw, h: ih, kind: 'corec', idx: gi });
          }
        } else if (isDie) {                             // NoC mesh hint + core count (no individual cores)
          const mc = inkOf(loadColor(v)); ctx.globalAlpha = 0.28;
          for (let g = 1; g < 4; g++) { line(x + g * dieW / 4, dyTop + 24, x + g * dieW / 4, dyTop + dieH - 8, mc, 1); line(x + 8, dyTop + 24 + g * (dieH - 32) / 4, x + dieW - 8, dyTop + 24 + g * (dieH - 32) / 4, mc, 1); }
          ctx.globalAlpha = 1; tx('≈16 Core-Group · 片上 NoC', x + dieW / 2, dyTop + dieH / 2 + 4, mc, '11px Inter', 'center');
        }
        const hbx = d === 0 ? x - 24 : x + dieW + 4;    // HBM beside
        for (let s = 0; s < 4; s++) fbox(hbx, dyTop + s * (dieH / 4), 18, dieH / 4 - 4, P.neutral, ENTITY_COLORS.ioDie);
        tx('HBM', hbx + 9, dyTop + dieH + 12, P.mut, '8px Inter', 'center');
      };
      drawComputeDie(lx, dv0, 0); drawComputeDie(rx, dv1, 1);
      fbox(cx - gap / 2 + 4, dyTop + dieH / 2 - 8, gap - 8, 16, loadColor((dv0 + dv1) / 2));   // D2D / UMA bridge
      tx('D2D 784GB/s · UMA', cx, dyTop + dieH / 2 - 13, P.mut, '8.5px Inter', 'center');
      if (isRank) {                                     // whole card → also 2 IO Die + UB/RDMA ports
        const ioY = dyTop + dieH + 22, ioH = Math.max(22, bot - ioY - 6);
        [{ x: lx, d: 0 }, { x: rx, d: 1 }].forEach(({ x, d }) => { fbox(x, ioY, dieW, ioH, P.neutral, ENTITY_COLORS.ioDie); tx(`IO Die ${d} · 互联/IO`, x + dieW / 2, ioY + ioH / 2 + 3, P.ink2, '10px Inter', 'center'); });
        fbox(lx + dieW / 2 - 30, bot - 14, 26, 12, loadColor(Pl[0].u)); tx('UB口', lx + dieW / 2 - 17, bot + 2, PLANES[0].color, '8.5px Inter', 'center');
        fbox(rx + dieW / 2 + 4, bot - 14, 26, 12, loadColor(Pl[1].u)); tx('RDMA口', rx + dieW / 2 + 17, bot + 2, PLANES[1].color, '8.5px Inter', 'center');
        tx('1 卡 = 2 计算 Die(UMA·OS 视为单 device) + 2 IO Die · 对外 UB口(绿)/RDMA口(橙)', PAD, H - 8, P.mut, '10px Inter');
      } else tx(isCore ? `32 Core-Group（Cube∶Vector≈8∶1）· 选中核 ${selCore} · L0 内部：GM/L2→UB/L1→L0A/B/C · Tile/lane` : '2 计算 Die（各 ≈16 Core-Group，UMA 合并为单 device）· 在热力镜头点 Die 看核', PAD, H - 8, P.mut, '10px Inter');
    }
  }, [lens, selLevel, selPool, selSpod, selNode, selNpu, selCore, pods, pools, podsInPool, CAB, NODES, NPN, NPC, NPU_TOT, step, P, metricVal, util01, faultAt, nodeMean, cabMean, spodMean, poolMean, clusterMean, dieVal, coreVal, scopeName, planeUtil, domains, domActive, tcell, flowCfg, pm, relHi, cardJ, matAgg, matDim, ev, EVT_CAB, dark]);

  useEffect(() => { draw(); }, [draw]);
  useEffect(() => { const onR = () => draw(); window.addEventListener('resize', onR); return () => window.removeEventListener('resize', onR); }, [draw]);
  // 逐帧重绘：通信域/物理链路始终让连线彗星流动；互联流量矩阵在【播放时】逐帧刷新，
  // 让通信强度真正随回放流动（暂停=定格快照，非静态写死）。
  useEffect(() => {
    const animate = lens === 'domain' || lens === 'phys' || ((lens === 'flow' || lens === 'matrix') && playing);
    if (!animate) return;
    let last = performance.now(), raf = 0;
    const loop = (now: number) => { const dt = Math.min(0.05, (now - last) / 1000); last = now; flowRef.current += dt * 1.2; draw(); raf = requestAnimationFrame(loop); };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [lens, draw, playing]);

  // hover + click on the canvas
  const hitTest = (mx: number, my: number) => { for (const c of cells.current) if (mx >= c.x && mx <= c.x + c.w && my >= c.y && my <= c.y + c.h) return c; return null; };
  const onMove = (e: React.MouseEvent) => {
    const r = cvRef.current!.getBoundingClientRect(), h = hitTest(e.clientX - r.left, e.clientY - r.top);
    if (!h) { if (tip) setTip(null); return; }
    let t = '';
    if (h.kind === 'gclus') t = h.idx === 1 ? `本集群 · ${PODS_PER_CLUSTER} Pod（>52万卡）· 在线 ${pods} · 平均 ${Math.round(clusterMean() * 100)}%（点击下钻 Cluster）` : `${h.idx === 0 ? 'Global A' : 'Global C'} · 兄弟集群（示意 · 点击切到 Cluster）`;
    else if (h.kind === 'pool') t = `服务池#${h.idx + 1} · ${Math.min(PODS_PER_POOL, pods - h.idx * PODS_PER_POOL)} Pod 在线 · 均值 ${Math.round(poolMean(h.idx) * 100)}%`;
    else if (h.kind === 'pooloff' || (h.kind === 'unitoff' && selLevel === 'cluster')) t = `服务池#${h.idx + 1} · 未接入（离线 · 集群共 ${POOLS_PER_CLUSTER} 池）`;
    else if (h.kind === 'spodoff' || (h.kind === 'unitoff' && selLevel === 'pool')) t = `Pod#${h.idx + 1} · 未上线（×${PODS_PER_POOL} Pod/池）`;
    else if (h.kind === 'spod') t = `Pod#${h.idx + 1} · 平均 ${Math.round(spodMean(h.idx) * 100)}%`;
    else if (h.kind === 'nodec' || (h.kind === 'punit' && selLevel === 'super')) t = `Host${(h.idx % NODES_PER_CAB) + 1} · 均值 ${Math.round(nodeMean(selSpod, h.idx) * 100)}%（机柜${((h.idx / NODES_PER_CAB) | 0) + 1} 物理分组）`;
    else if (h.kind === 'npuc' || h.kind === 'pnpu') { const v = metricVal(selSpod, selNode, h.idx); t = `Chip r${h.idx} (rank ${selNode * NPN + h.idx}) · ${Math.round(v * 100)}% ${STATE_LABELS[loadState(v)]}`; }
    else if (h.kind === 'rankfull') { const nl = (h.idx % NODES_PER_CAB) + 1; t = `Host${nl} · 均值 ${Math.round(nodeMean(selSpod, h.idx) * 100)}%（机柜${((h.idx / NODES_PER_CAB) | 0) + 1} 物理分组）${isStrag(selSpod, h.idx) ? ' · straggler' : ''}`; }
    else if (h.kind === 'punit') t = `${selLevel === 'pool' ? 'Pod' : '单元'}${h.idx + 1}`;
    else if (h.kind === 'diec') t = `计算 Die ${h.idx} · ${Math.round(dieVal(h.idx) * 100)}%`;
    else if (h.kind === 'iodie') t = `IO Die ${h.idx} · 互联/IO（无算力负载）`;
    else if (h.kind === 'corec') t = `${h.idx % 8 === 7 ? 'AIV/Vector' : 'AIC/Cube'} #${h.idx} · ${Math.round(coreVal(h.idx) * 100)}%`;
    // parent-context strip (上层) cells
    else if (h.kind === 'ppool') t = `服务池#${h.idx + 1} · 均值 ${Math.round(poolMean(h.idx) * 100)}%（点击切换）`;
    else if (h.kind === 'pspod') t = `Pod#${h.idx + 1} · 平均 ${Math.round(spodMean(h.idx) * 100)}%（点击切换）`;
    else if (h.kind === 'pnode') { const cb = (selNode / NODES_PER_CAB) | 0; t = `Host${h.idx + 1} · 均值 ${Math.round(nodeMean(selSpod, cb * NODES_PER_CAB + h.idx) * 100)}%（机柜${cb + 1} 物理分组）`; }
    else if (h.kind === 'pcard') t = `Chip r${h.idx}（rank ${selNode * NPN + h.idx}）· ${Math.round(util01(selSpod, selNode, h.idx) * 100)}%（点击切换）`;
    // 通信矩阵 lens：档位/维度 chips + 矩阵格（行列换算成单元对）
    else if (h.kind === 'magg') t = `聚合档位：${['rank×rank', 'Host×Host', 'Pod×Pod'][h.idx]}（点击切换）`;
    else if (h.kind === 'mdim') t = `按泳道 phase 过滤：${['TP · 对角块状', 'PP · 次对角线', 'EP · 簇间棋盘', 'DP · 环带'][h.idx]}`;
    else if (h.kind === 'mcell' && matGeom.current) {
      const g = matGeom.current, mxr = e.clientX - r.left, myr = e.clientY - r.top;
      const j = Math.min(g.N - 1, Math.max(0, ((mxr - g.mx) / g.cs) | 0)), i = Math.min(g.N - 1, Math.max(0, ((myr - g.my) / g.cs) | 0));
      const unit = matAgg === 'pod' ? 'Pod' : matAgg === 'host' ? 'Host' : 'rank';
      t = i === j ? `${unit}${i} · 对角线=自身（不通信）` : `${unit}${i} ⇄ ${unit}${j} · ${matDim} 通信对 · 点击定位该${unit}对`;
    }
    if (!t) { if (tip) setTip(null); return; }
    setTip({ x: e.clientX, y: e.clientY, t });
  };
  const onClick = (e: React.MouseEvent) => {
    const r = cvRef.current!.getBoundingClientRect(), h = hitTest(e.clientX - r.left, e.clientY - r.top); if (!h) return;
    if (h.kind === 'gclus') { setSelLevel('cluster'); }
    else if (h.kind === 'pool' || (h.kind === 'punit' && selLevel === 'cluster')) { setSelPool(h.idx); setSelSpod(h.idx * PODS_PER_POOL); setSelCab(0); setSelNode(0); setSelNpu(-1); setSelLevel('pool'); }
    else if (h.kind === 'spod' || (h.kind === 'punit' && selLevel === 'pool')) { setSelPool((h.idx / PODS_PER_POOL) | 0); setSelSpod(h.idx); setSelCab(0); setSelNode(0); setSelNpu(-1); setSelLevel('super'); }
    // Pod 直接下钻 Host：super 的子单元(Host 网格) / Host 全量块 → node
    else if (h.kind === 'nodec' || h.kind === 'rankfull' || (h.kind === 'punit' && selLevel === 'super')) { setSelNode(h.idx); setSelCab((h.idx / NODES_PER_CAB) | 0); setSelNpu(-1); setSelLevel('node'); }
    else if (h.kind === 'npuc' || h.kind === 'pnpu') setSelNpu(h.idx);
    else if (h.kind === 'diec') { setSelNpu((j) => (j < 0 ? 0 : j)); setSelLevel('core'); }
    else if (h.kind === 'corec') { setSelCore(h.idx); }
    // parent-context strip: switch to a sibling at the SAME level (no need to go back up)
    else if (h.kind === 'ppool') { setSelPool(h.idx); setSelSpod(h.idx * PODS_PER_POOL); setSelCab(0); setSelNode(0); setSelNpu(-1); }
    else if (h.kind === 'pspod') { setSelSpod(h.idx); setSelPool((h.idx / PODS_PER_POOL) | 0); setSelCab(0); setSelNode(0); setSelNpu(-1); }
    else if (h.kind === 'pnode') { const cb = (selNode / NODES_PER_CAB) | 0; setSelNode(cb * NODES_PER_CAB + h.idx); setSelNpu(-1); }
    else if (h.kind === 'pcard') setSelNpu(h.idx);
    // 通信矩阵 lens：档位/维度切换 + 点格定位对应卡对（空间锚点=rank：任一处点选全站同步）
    else if (h.kind === 'magg') setMatAgg((['rank', 'host', 'pod'] as const)[h.idx]);
    else if (h.kind === 'mdim') setMatDim((['TP', 'PP', 'EP', 'DP'] as const)[h.idx]);
    else if (h.kind === 'mcell' && matGeom.current) {
      const g = matGeom.current, myr = e.clientY - r.top;
      const i = Math.min(g.N - 1, Math.max(0, ((myr - g.my) / g.cs) | 0));
      if (matAgg === 'pod') { const gp = Math.min(i, pods - 1); setSelPool((gp / PODS_PER_POOL) | 0); setSelSpod(gp); setSelNode(0); setSelNpu(-1); setSelLevel('super'); }
      else if (matAgg === 'host') { setSelNode(Math.min(i, NODES - 1)); setSelCab((i / NODES_PER_CAB) | 0); setSelNpu(-1); setSelLevel('node'); }
      else { const cb = (selNode / NODES_PER_CAB) | 0, hostI = cb * NODES_PER_CAB + (i >> 3); setSelNode(Math.min(hostI, NODES - 1)); setSelNpu(i & 7); setSelLevel('node'); }
    }
  };

  // level navigation (axis + breadcrumb)
  const setLevel = (id: Level) => {
    if ((id === 'node' || SUBCARD.includes(id)) && selNode < 0) setSelNode(selCab * NODES_PER_CAB);
    if (SUBCARD.includes(id) && selNpu < 0) setSelNpu(0);
    setSelLevel(id);
  };

  // breadcrumb segments (hw-native-sys L7→L0 · EXACTLY 8 levels · 无机柜/无 Tile):
  // 全球 › 集群 › 服务池#p › Pod#N › Host N › Chip rN › Die › Core-Group
  const crumbs: { lvl: Level; label: string }[] = [{ lvl: 'global', label: '全球' }];
  if (selLevel !== 'global') crumbs.push({ lvl: 'cluster', label: '集群' });
  if (!['global', 'cluster'].includes(selLevel)) crumbs.push({ lvl: 'pool', label: `服务池#${selPool + 1}` });
  if (!['global', 'cluster', 'pool'].includes(selLevel)) crumbs.push({ lvl: 'super', label: `Pod#${selSpod + 1}` });
  if (['node', ...SUBCARD].includes(selLevel)) crumbs.push({ lvl: 'node', label: `Host ${(selNode % NODES_PER_CAB) + 1}` });
  if (SUBCARD.includes(selLevel)) crumbs.push({ lvl: 'rank', label: `Chip r${cardJ}` });
  if (SUBCARD.includes(selLevel)) crumbs.push({ lvl: 'die', label: 'Die' });
  if (selLevel === 'core') crumbs.push({ lvl: 'core', label: 'Core-Group' });

  // ── detail rail data ──
  const sm = scopeMean();
  const decomp = STEP_DECOMP[phase];   // paper-grounded 计算/通信/访存 split (arXiv:2505.21411)
  const scopeCount = ({ global: 1, cluster: POOLS_PER_CLUSTER, pool: PODS_PER_POOL, super: NODES, node: NPN, rank: NPU_TOT, die: COMPUTE_DIES_PER_CARD, core: CORES_PER_CARD } as Record<Level, number>)[selLevel];
  const scopeUnit = ({ global: '集群', cluster: '服务池', pool: 'Pod', super: 'Host', node: 'Chip·NPU', rank: 'Chip·NPU（rank 1:1）', die: '计算 Die', core: 'Core-Group' } as Record<Level, string>)[selLevel];
  // per-NPU bars for the focused card's node (the TP group) — shown whenever a card is in context
  const showAssoc = cardSelected || selLevel === 'node';
  const nodePeers = showAssoc ? Array.from({ length: NPN }, (_, j) => ({ j, u: util01(selSpod, selNode, j), fault: faultAt(selSpod, selNode, j), strag: isStrag(selSpod, selNode) })) : null;

  // inline SVG: card associations (node-internal 8 NPU ↔ L1 ↔ CPU), focused card highlighted (mirrors 平面视图)
  const assocSVG = () => {
    const W = 240, cx = W / 2, apex: [number, number] = [cx, 42];
    // bus-wiring connector 接点：色环 + 白芯
    const dot = (x: number, y: number, c: string, r = 3.2) => (<><circle cx={x} cy={y} r={r} fill={c} /><circle cx={x} cy={y} r={r * 0.42} fill="#fff" /></>);
    // 运行(playing)时沿线流动的白色彗星（SMIL 连续动画，独立于 step 重渲染）
    const flow = (x1: number, y1: number, x2: number, y2: number) => playing ? (
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#fff" strokeWidth={1.6} strokeLinecap="round" strokeDasharray="3 13" opacity={0.85}>
        <animate attributeName="stroke-dashoffset" from="16" to="0" dur="0.6s" repeatCount="indefinite" />
      </line>
    ) : null;
    return (
      <svg viewBox={`0 0 ${W} 150`} style={{ width: '100%', height: 'auto', display: 'block', marginTop: 4 }}>
        {Array.from({ length: NPN }, (_, j) => {
          const u = util01(selSpod, selNode, j), x = 14 + j * ((W - 28) / (NPN - 1)), focus = j === cardJ;
          return (
            <g key={j}>
              <line x1={x} y1={84} x2={apex[0]} y2={apex[1]} stroke={loadColor(u)} strokeWidth={focus ? 3.2 : 1.8} strokeLinecap="round" opacity={focus ? 1 : 0.6} />
              {flow(x, 84, apex[0], apex[1])}
              {dot(x, 84, loadColor(u), focus ? 3.6 : 3)}
              <rect x={x - 12} y={84} width={24} height={22} rx={3} fill={loadColor(u)} stroke={focus ? ACCENT : 'transparent'} strokeWidth={focus ? 2.5 : 0} />
              <text x={x} y={99} fontSize={8.5} fill={inkOf(loadColor(u))} textAnchor="middle">r{j}</text>
            </g>
          );
        })}
        <line x1={cx} y1={128} x2={apex[0]} y2={apex[1]} stroke={PLANES[0].color} strokeWidth={1.8} strokeLinecap="round" opacity={0.7} />
        {flow(cx, 128, apex[0], apex[1])}
        {dot(cx, 128, PLANES[0].color)}
        <polygon points={`${cx},10 ${cx + 18},26 ${cx},42 ${cx - 18},26`} fill="none" stroke={loadColor(sm)} strokeWidth={2} strokeLinejoin="round" />
        {dot(apex[0], apex[1], loadColor(sm), 3.6)}
        <text x={cx} y={28} fontSize={9} fill="var(--tx)" textAnchor="middle">L1</text>
        <rect x={cx - 34} y={128} width={68} height={20} rx={10} fill="none" stroke="var(--bd2)" />
        <text x={cx} y={142} fontSize={9} fill="var(--tx2)" textAnchor="middle">{TOK.kunpeng} CPU</text>
      </svg>
    );
  };

  return (
    <div className={workbenchProfile ? 'hpc-status-shell hpc-status-shell--workbench' : undefined} data-theme={dark ? 'dark' : 'light'} style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', background: workbenchProfile ? 'var(--background-elevated)' : 'var(--bg)', overflow: 'hidden' }}>
      {/* control header — hidden in workbench (filters live in center-pill dropdown) */}
      {!workbenchProfile && (
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, flexWrap: 'wrap', padding: '8px 14px', borderBottom: '1px solid var(--bd)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12.5, flexWrap: 'wrap' }}>
          <span style={{ color: 'var(--tx3)' }}>选区</span>
          {crumbs.map((c, i) => (
            <span key={c.lvl + i} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              {i > 0 && <span style={{ color: 'var(--tx3)' }}>›</span>}
              <span onClick={() => setLevel(c.lvl)} style={{ cursor: 'pointer', color: c.lvl === selLevel ? 'var(--tx)' : '#5b86ff', fontWeight: c.lvl === selLevel ? 700 : 400 }}>{c.label}</span>
            </span>
          ))}
        </div>
        <div style={{ flex: 1 }} />
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={LBL}>镜头</span>
          {([['heat', '状态热力'], ['flow', '互联流量'], ['domain', '通信域'], ['phys', '物理链路'], ['matrix', '通信矩阵']] as [Lens, string][]).map(([v, l]) => (<button key={v} onClick={() => setLens(v)} style={{ padding: '4px 11px', fontSize: 11.5, borderRadius: 8, cursor: 'pointer', ...navBtn(lens === v) }}>{l}</button>))}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={LBL}>并行</span>
          <button onClick={() => setRelHi(!relHi)} title="状态热力镜头下：在真实网格上给选中卡的 TP/PP/DP/EP 对端描边（parallelMap 真值 · 采样显示）。想看整个并行域拓扑请切到「通信域」镜头。" style={{ padding: '4px 11px', fontSize: 11.5, borderRadius: 8, cursor: 'pointer', ...toggleBtn(relHi, ACCENT) }}>并行对端高亮</button>
        </div>
      </div>
      )}

      {/* KPI strip */}
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', padding: '10px 14px 4px' }}>
        {([
          [`${NPU_TOT.toLocaleString()}`, 'NPU / Pod', 'var(--tx)'],
          phase === 'decode'
            ? [`${WORKLOAD.perf.decodeTPOTms}ms`, 'TPOT(实测)', 'var(--tx)']
            : phase === 'prefill'
              ? [`${WORKLOAD.perf.prefillTTFTms}ms`, 'TTFT(实测)', 'var(--tx)']
              : [`+${WORKLOAD.mfuGainPct}%`, 'MFU增益(实测)', '#22d3ee'],
          phase === 'decode'
            ? [`${WORKLOAD.perf.decodeMtpTokps}`, 'MTP tok/s(实测)', 'var(--tx)']
            : phase === 'prefill'
              ? [`${WORKLOAD.perf.prefillTokps}`, 'Prefill tok/s(实测)', 'var(--tx)']
              : [`${WORKLOAD.trainNpus}`, '训练 NPU', 'var(--tx)'],
          [`${(kpi.redR * 100).toFixed(1)}%`, '红区占比', stateColor(kpi.redR >= 0.1 ? 2 : kpi.redR >= 0.04 ? 1 : 0)],
          [`${kpi.fa}`, '故障 NPU', kpi.fa ? '#e5484d' : 'var(--tx2)'],
          [`${PH[phase].label} · #${step}`, '当前工况 / step', 'var(--tx)'],
        ] as [string, string, string][]).map(([v, l, c]) => (
          <div key={l} style={{ background: 'var(--panel-solid)', border: '1px solid var(--bd)', borderRadius: 10, padding: '7px 14px', minWidth: 96 }}>
            <div style={{ fontSize: 14, fontWeight: 700, fontFamily: MONO, color: c, ...TNUM }}>{v}</div>
            <div style={{ fontSize: 10, fontWeight: 500, letterSpacing: 0.5, textTransform: 'uppercase', color: 'var(--tx3)' }}>{l}</div>
          </div>
        ))}
      </div>

      {/* hierarchy status-axis (full chain incl. below-card) */}
      <div style={{ display: 'flex', alignItems: 'stretch', gap: 0, flexWrap: 'wrap', padding: '6px 14px 2px' }}>
        {axis.map((l, i) => (
          <div key={l.id} style={{ display: 'contents' }}>
            <div onClick={() => setLevel(l.id)} style={{
              flex: '1 1 0', minWidth: 104, cursor: 'pointer', padding: '6px 9px', borderRadius: 9,
              border: `1px solid ${selLevel === l.id ? ACCENT : 'var(--bd)'}`, background: selLevel === l.id ? 'var(--state-sel)' : 'var(--panel-solid)',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                <span style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--tx)' }}>{l.nm}</span>
                <span style={{ fontSize: 10, color: 'var(--tx3)' }}>{l.su}</span>
              </div>
              <div style={{ height: 3, background: dark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)', borderRadius: 2, overflow: 'hidden', margin: '5px 0 3px' }}>
                <div style={{ height: '100%', width: `${Math.round(l.p50 * 100)}%`, background: loadColor(l.p50), borderRadius: 2 }} />
              </div>
              <div style={{ fontSize: 10, color: 'var(--tx2)', fontFamily: MONO }}>{`典型 ${Math.round(l.p50 * 100)}% · 红 ${(l.red * 100).toFixed(1)}% · 峰 ${Math.round(l.p95 * 100)}%`}</div>
            </div>
            {i < axis.length - 1 && <div style={{ alignSelf: 'center', color: 'var(--tx3)', padding: '0 3px', fontSize: 10 }}>▸</div>}
          </div>
        ))}
      </div>

      {/* main stage: enlarged lens canvas + detail rail */}
      <div style={{ flex: 1, display: 'flex', gap: 12, padding: '8px 14px 12px', minHeight: 0 }}>
        <div ref={wrapRef} style={{ flex: 1, minWidth: 0, position: 'relative', borderRadius: 12, ...(workbenchProfile ? {} : { border: '1px solid var(--bd)' }), overflow: 'hidden', background: 'var(--panel-solid)' }}>
          {selLevel === 'core' ? (
            // L0 Core-Group（最深层级）：不再用画布，改渲染 memory-architecture pattern（CoreGroupPattern）
            <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column' }}>
              <div style={{ padding: '8px 12px', borderBottom: '1px solid var(--bd)', fontSize: 12, color: 'var(--tx2)', flexShrink: 0 }}>
                选中：Chip r{cardJ} · L0 Core-Group（AIV·向量 / AIC·Cube / AICPU）<span style={{ color: 'var(--tx3)' }}> · 内部：GM/L2→UB/L1→L0A/B/C · Tile/lane</span>
              </div>
              <div style={{ flex: 1, minHeight: 0 }}>
                <CoreGroupPattern phaseKind={kind} load={util01(selSpod, selNode, cardJ)} zoom={0.45} height="100%" />
              </div>
            </div>
          ) : (
            <canvas ref={cvRef} onMouseMove={onMove} onMouseLeave={() => setTip(null)} onClick={onClick} style={{ display: 'block', width: '100%', height: '100%', cursor: 'pointer' }} />
          )}
        </div>

        {/* detail rail */}
        <div style={{ width: 272, flexShrink: 0, overflowY: 'auto', borderRadius: 12, ...(workbenchProfile ? { boxShadow: 'var(--shadow-sm)' } : { border: '1px solid var(--bd)' }), background: 'var(--panel-solid)', padding: '12px 14px' }}>
          <div style={{ fontSize: 14, fontWeight: 700, color: '#5b86ff', marginBottom: 2 }}>{scopeName()}</div>
          <div style={{ fontSize: 11, color: 'var(--tx3)', marginBottom: 10 }}>{({ heat: '状态热力', flow: '互联流量', domain: '通信域', phys: '物理链路', matrix: 'N×N 通信矩阵' })[lens]} · {({ util: '利用率', strag: 'straggler 落后度', fault: '故障' })[metric]}</div>

          <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
            <div style={{ flex: 1, padding: '7px 9px', borderRadius: 8, background: 'var(--btn)' }}>
              <div style={{ fontSize: 15, fontWeight: 700, fontFamily: MONO, color: loadColor(sm) }}>{Math.round(sm * 100)}%</div>
              <div style={{ fontSize: 10, color: 'var(--tx3)' }}>选区平均负载</div>
            </div>
            <div style={{ flex: 1, padding: '7px 9px', borderRadius: 8, background: 'var(--btn)' }}>
              <div style={{ fontSize: 15, fontWeight: 700, fontFamily: MONO, color: 'var(--tx)' }}>{scopeCount.toLocaleString()}</div>
              <div style={{ fontSize: 10, color: 'var(--tx3)' }}>{scopeUnit}</div>
            </div>
          </div>

          {/* real workload anchor: Pangu Pro MoE (arXiv:2505.21411) — model shape + measured tok/s */}
          <div style={{ borderTop: '1px solid var(--bd)', paddingTop: 9, marginBottom: 4 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--tx2)', marginBottom: 4 }}>工况模型 · {WORKLOAD.name} <span style={{ color: 'var(--tx3)', fontWeight: 400 }}>{WORKLOAD.short}</span></div>
            <div style={{ fontSize: 10, color: 'var(--tx3)', lineHeight: 1.55 }}>
              {WORKLOAD.routedExperts}路由/{WORKLOAD.activatedExperts}激活/{WORKLOAD.sharedExperts}共享专家 · {WORKLOAD.layers}层 · hidden {WORKLOAD.hidden}<br />
              <span style={{ color: 'var(--tx2)', fontFamily: MONO }}>
                {phase === 'decode'
                  ? `Decode ${WORKLOAD.perf.decodeTokps} tok/s·卡 (batch${WORKLOAD.perf.decodeBatch}·TPOT ${WORKLOAD.perf.decodeTPOTms}ms) → MTP ${WORKLOAD.perf.decodeMtpTokps}`
                  : phase === 'prefill'
                    ? `Prefill ${WORKLOAD.perf.prefillTokps} tok/s·卡 (TTFT ${WORKLOAD.perf.prefillTTFTms}ms)`
                    : `预训练 ${WORKLOAD.trainNpus} NPU · ${WORKLOAD.trainTokens} tokens · MFU +${WORKLOAD.mfuGainPct}%`}
              </span>
            </div>
            <div style={{ fontSize: 9.5, color: 'var(--tx3)', marginTop: 4, lineHeight: 1.5 }}>
              {phase === 'decode'
                ? `内核：MulAttention ${WORKLOAD_DETAIL.kernel.mulAttnSpeedup}× · 注意力占时延 ${WORKLOAD_DETAIL.kernel.attnLatencyPct[0]}–${WORKLOAD_DETAIL.kernel.attnLatencyPct[1]}%（KV 搬运占其 ${WORKLOAD_DETAIL.kernel.kvOfAttnPct}%）`
                : phase === 'prefill'
                  ? `Prefill 每 token 仅激活 Top-${WORKLOAD.activatedExperts} 专家 ≈ ${WORKLOAD.activeB}B Dense · SwiftGMM 高并发占时延 >${WORKLOAD_DETAIL.kernel.swiftGmmLatencyPct}%`
                  : `MoGE：${WORKLOAD_DETAIL.moge.note} · 负载不均 ↓>${WORKLOAD_DETAIL.moge.imbalanceReductionPct}%`}
            </div>
            <div style={{ fontSize: 9, color: 'var(--tx3)', marginTop: 3 }}>
              通信优化：AllReduce→RS+AG −{WORKLOAD_DETAIL.comm.allreduceCutPct}% · RMSNorm 重排 −{WORKLOAD_DETAIL.comm.rmsnormCutPct}% · 融合 {WORKLOAD_DETAIL.comm.fusedOps.join('/')}
            </div>
            <div style={{ fontSize: 9, color: 'var(--tx3)', marginTop: 3 }}>真实值 · Ascend 800I A2/300I Duo（arXiv:2505.21411）</div>
            <div style={{ fontSize: 9.5, color: 'var(--tx2)', marginTop: 4, fontFamily: MONO }}>物理平铺 {pm.cfg}</div>
            <div style={{ fontSize: 9, color: 'var(--tx3)', marginTop: 1 }}>{pm.real} · {pm.approxNote}</div>
          </div>

          {/* card associations / comm relationships (mirrors 平面视图's relationship view) */}
          {showAssoc && (
            <div style={{ borderTop: '1px solid var(--bd)', paddingTop: 9, marginBottom: 4 }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--tx2)', display: 'flex', justifyContent: 'space-between' }}>
                <span>通信关系 · 卡 r{cardJ}（rank {selNode * NPN + cardJ}）</span>
                {nodePeers?.[cardJ]?.strag && <span style={{ color: '#e5484d' }}>straggler</span>}
              </div>
              {assocSVG()}
              {/* 并行组归属（parallelMap 真值 · 1 基显示）——与画布『关系高亮』同源 */}
              <div style={{ fontSize: 10, color: 'var(--tx2)', fontFamily: MONO, margin: '4px 0 2px', lineHeight: 1.6 }}>
                TP 组=本 Host {NPN} rank · PP 级 {pm.groupOf(cardRank, 'pp') + 1}/{pm.pp} · DP 副本 {pm.groupOf(cardRank, 'dp') + 1}/{pm.dp} · EP 组 {pm.groupOf(cardRank, 'ep') + 1}/{pm.ep}
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 4, marginTop: 6 }}>
                {nodePeers!.map((p) => (
                  <div key={p.j} onClick={() => { setSelNpu(p.j); if (SUBCARD.includes(selLevel)) setSelLevel('node'); }} title={`NPU${p.j} · ${Math.round(p.u * 100)}%${p.fault ? ' · 故障' : ''}`}
                    style={{ cursor: 'pointer', padding: '4px 2px', borderRadius: 6, background: loadColor(p.u), border: p.j === cardJ ? `2px solid ${ACCENT}` : `1px solid ${p.fault ? '#e5484d' : 'transparent'}`, textAlign: 'center' }}>
                    <div style={{ fontSize: 9.5, fontWeight: 700, color: inkOf(loadColor(p.u)) }}>{p.fault ? '✕' : `r${p.j}`}</div>
                    <div style={{ fontSize: 9, fontFamily: MONO, color: inkOf(loadColor(p.u)) }}>{Math.round(p.u * 100)}</div>
                  </div>
                ))}
              </div>
              {/* association rows: which comm groups this card belongs to */}
              <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 4 }}>
                {([
                  ['TP 组', `本 Host ${NPN} rank · TP${WORKLOAD.train.tp} AllReduce`, PARTITION_META.tp.label.includes('TP') ? '#04d793' : '#04d793', loadColor(nodeMean(selSpod, selNode))],
                  ['EP 组', `EP${WORKLOAD.inferRouted.ep} · ${WORKLOAD.routedExperts}路由/${WORKLOAD.activatedExperts}激活 · All-to-All`, '#ff4b7b', loadColor(cabMean(selSpod, (selNode / NODES_PER_CAB) | 0))],
                  ['DP 副本', `跨 Pod ×${pods} · DP${WORKLOAD.inferAttn.dp} AllReduce`, '#ffaa3b', loadColor(sm)],
                  ['上联链路', `Chip→L1→L2→Pod · UB`, PLANES[0].color, loadColor(planeUtil()[0].u)],
                ] as [string, string, string, string][]).map(([k, v, tag, st]) => (
                  <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 7, fontSize: 10.5 }}>
                    <span style={{ width: 8, height: 8, borderRadius: 2, background: tag, flexShrink: 0 }} />
                    <span style={{ color: 'var(--tx)', width: 52, flexShrink: 0 }}>{k}</span>
                    <span style={{ color: 'var(--tx3)', flex: 1 }}>{v}</span>
                    <span style={{ width: 9, height: 9, borderRadius: '50%', background: st, flexShrink: 0 }} />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* step decomposition */}
          <div style={{ borderTop: '1px solid var(--bd)', paddingTop: 9 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--tx2)', margin: '0 0 5px' }}>step 时间分解（{PH[phase].label} · {WORKLOAD.short}）</div>
            {decomp.map(({ label, frac, color }) => (
              <div key={label} style={{ marginBottom: 5 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10.5, marginBottom: 2 }}><span style={{ color: 'var(--tx)' }}>{label}</span><span style={{ color: 'var(--tx3)', fontFamily: MONO }}>{Math.round(frac * 100)}%</span></div>
                <div style={{ height: 7, borderRadius: 4, background: 'var(--btn)', overflow: 'hidden' }}><div style={{ height: '100%', width: `${frac * 100}%`, background: color }} /></div>
              </div>
            ))}
          </div>

          {/* legend */}
          <div style={{ borderTop: '1px solid var(--bd)', marginTop: 10, paddingTop: 10 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--tx2)', marginBottom: 5 }}>状态色（红黄绿=状态，唯一一套）</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 12px' }}>
              {STATE_LABELS.map((lb, i) => (<span key={lb} style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 10.5, color: 'var(--tx2)' }}><span style={{ width: 11, height: 11, borderRadius: 3, background: stateColor(i) }} />{lb}</span>))}
            </div>
            {(lens === 'flow' || lens === 'phys') && (
              <div style={{ marginTop: 8 }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--tx2)', marginBottom: 5 }}>三平面</div>
                {PLANES.map((p) => (<span key={p.id} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 10.5, color: 'var(--tx2)', marginBottom: 2 }}><span style={{ width: 12, height: 3, background: p.color, borderRadius: 1 }} />{p.short} · {p.parallel}</span>))}
              </div>
            )}
          </div>

          {/* model-quality benchmarks: Pangu Pro MoE vs comparable 27–32B (arXiv:2505.21411 T3) */}
          <div style={{ borderTop: '1px solid var(--bd)', marginTop: 10, paddingTop: 10 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--tx2)', marginBottom: 6 }}>模型质量对照 · {WORKLOAD.name} <span style={{ color: 'var(--tx3)', fontWeight: 400 }}>vs 27–32B</span></div>
            {BENCHMARKS.map((b) => {
              const mine = b.scores[BENCH_PANGU_IDX];
              const others = b.scores.filter((_, i) => i !== BENCH_PANGU_IDX);
              const bestOther = Math.max(...others);
              const lead = mine >= bestOther;
              const scale = (s: number) => Math.max(0, Math.min(1, (s - 40) / 55));   // 40–95 → 0–1 for contrast
              return (
                <div key={b.name} style={{ marginBottom: 5 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, marginBottom: 2 }}>
                    <span style={{ color: 'var(--tx)' }}>{b.name}</span>
                    <span style={{ color: 'var(--tx3)', fontFamily: MONO }}>
                      <span style={{ color: lead ? '#04d793' : 'var(--tx)', fontWeight: 700 }}>{mine.toFixed(1)}</span>
                      <span style={{ color: 'var(--tx3)' }}> · 次优 {bestOther.toFixed(1)} {lead ? `(+${(mine - bestOther).toFixed(1)})` : `(${(mine - bestOther).toFixed(1)})`}</span>
                    </span>
                  </div>
                  <div style={{ position: 'relative', height: 8, borderRadius: 4, background: 'var(--btn)', overflow: 'hidden' }}>
                    <div style={{ position: 'absolute', left: 0, top: 0, height: '100%', width: `${scale(mine) * 100}%`, background: lead ? ACCENT : '#9aa3b2', borderRadius: 4 }} />
                    <div title={`最优对手 ${bestOther.toFixed(1)}`} style={{ position: 'absolute', left: `${scale(bestOther) * 100}%`, top: -1, height: 10, width: 2, background: 'var(--tx)', opacity: 0.65 }} />
                  </div>
                </div>
              );
            })}
            <div style={{ fontSize: 9, color: 'var(--tx3)', marginTop: 5 }}>蓝条=盘古得分 · 竖线=最优对手（{BENCH_MODELS.length - 1} 个 27–32B）· EM/F1/Pass@1 · arXiv:2505.21411 T3</div>
          </div>

          {/* same-family real results on Ascend super-nodes (reference context) */}
          <div style={{ borderTop: '1px solid var(--bd)', marginTop: 10, paddingTop: 10 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--tx2)', marginBottom: 6 }}>同类对照 · 昇腾超节点真实论文</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {WORKLOAD_REFS.map((r) => (
                <div key={r.id} style={{ fontSize: 10, lineHeight: 1.45, paddingLeft: 7, borderLeft: `2px solid ${r.id === 'pangu-pro-moe' ? ACCENT : 'var(--bd2)'}` }}>
                  <div style={{ color: 'var(--tx)', fontWeight: 600 }}>{r.title} <span style={{ color: 'var(--tx3)', fontWeight: 400, fontFamily: MONO }}>arXiv:{r.arxiv}</span></div>
                  <div style={{ color: 'var(--tx3)' }}>{r.scale}</div>
                  <div style={{ color: 'var(--tx2)' }}>{r.metric}</div>
                </div>
              ))}
            </div>
            <div style={{ fontSize: 9, color: 'var(--tx3)', marginTop: 6 }}>注：对照均为昇腾超节点真实论文（非本视图 950 硬件平台），仅作规模/性能参照。</div>
          </div>

          <div style={{ borderTop: '1px solid var(--bd)', marginTop: 10, paddingTop: 10 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--tx2)', marginBottom: 5 }}>集群规模（示意 · Pod 数）</div>
            <div style={{ display: 'flex', gap: 4 }}>
              {[1, 2, 4, 8].map((c) => (<button key={c} onClick={() => { setPods(c); if (selSpod >= c) setSelSpod(0); if (selPool >= Math.max(1, Math.ceil(c / PODS_PER_POOL))) setSelPool(0); }} style={{ padding: '4px 10px', fontSize: 11.5, borderRadius: 8, cursor: 'pointer', ...toggleBtn(pods === c, ACCENT) }}>×{c}</button>))}
            </div>
            <div style={{ fontSize: 10, color: 'var(--tx3)', marginTop: 6 }}>负载状态为示意（含 straggler/故障注入 + 回放事件）。计数与关系由真实层级规模推导；工况/并行/通信/吞吐取自 Pangu Pro MoE 论文（arXiv:2505.21411）；接 profiler 后替换 nodeLoad 即可。</div>
          </div>
        </div>
      </div>

      {tip && (
        <div style={{ position: 'fixed', left: tip.x + 12, top: tip.y + 12, pointerEvents: 'none', zIndex: 30, background: 'var(--panel-solid)', border: '1px solid var(--bd2)', borderRadius: 8, padding: '5px 9px', fontSize: 11.5, color: 'var(--tx)', boxShadow: 'var(--shadow-sm)', whiteSpace: 'nowrap' }}>{tip.t}</div>
      )}
    </div>
  );
}
