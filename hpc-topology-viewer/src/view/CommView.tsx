/**
 * CommView — 通信全景（rank×rank 通信矩阵）. A DEDICATED, NON-hierarchical 2-D view for the
 * communication RELATIONSHIP itself, added as its own 2D 分析 view (NOT inside 运行状态).
 *
 * Why a separate flat view: physical containment is a TREE (超节点⊃机柜⊃节点⊃卡) while
 * communication is a set of OVERLAPPING groups (TP within a node · PP a chain · DP/EP across).
 * Drawing dense collectives ON the hierarchy at 8K scale = an unreadable hairball; squashing
 * them into a band below it loses the link. So we give comm its OWN substrate: a rank×rank
 * matrix where cell (i,j) is coloured by WHICH parallel dimension connects that pair
 * (TP/SP/EP/PP/DP signature colour) and BRIGHTENED by the live traffic. Block-diagonal = TP,
 * stage-stride bands = PP, far bands = DP, dense blocks = EP — each dimension's footprint reads
 * at a glance, WITHOUT any hierarchy lines.
 *
 * Single source of truth: connectivity comes from data.ts `parallelMap` (same groupOf/peersOf
 * the 平面/工作台/3D use) so degrees + membership agree everywhere. It also accepts the shared
 * `sync` (工况/时间/播放) so it stays linked with 运行状态 ⇄ 工作台.
 */
import { useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import {
  GENERATIONS, PARTITION_META, PARALLEL_COLORS, PARALLEL_COLORS_SP, STEP_DECOMP,
  parallelMap, cardLoad01, loadColor,
  type Gen, type ParDim, type ParallelWorkload, type ViewSync,
} from '../scene/data';
import { SceneVisualProfileContext } from '../scene/visual-profile';

const MONO = "'JetBrains Mono','Consolas',ui-monospace,monospace";
const SECONDARY: React.CSSProperties = { border: '1px solid var(--button-secondary-border)', background: 'var(--button-secondary-bg)', color: 'var(--foreground-muted)' };
const LBL: React.CSSProperties = { fontSize: 11, fontWeight: 500, letterSpacing: 0.5, textTransform: 'uppercase', color: 'var(--tx3)' };
const TNUM: React.CSSProperties = { fontVariantNumeric: 'tabular-nums' };
function navBtn(active: boolean): React.CSSProperties {
  return active ? { border: '1px solid var(--primary)', background: 'var(--primary)', color: 'var(--primary-foreground)', fontWeight: 600 } : { ...SECONDARY };
}
function toggleBtn(active: boolean, c: string): React.CSSProperties {
  const h = c.replace('#', ''); const r = parseInt(h.slice(0, 2), 16), g = parseInt(h.slice(2, 4), 16), b = parseInt(h.slice(4, 6), 16);
  const ink = 0.299 * r + 0.587 * g + 0.114 * b > 150 ? '#10131a' : '#fff';
  return active ? { border: `1px solid ${c}`, background: c, color: ink, fontWeight: 600 } : { ...SECONDARY };
}

type Scope = 'intra' | 'inter';
type DimFilter = 'all' | ParDim;
const WL: Record<ParallelWorkload, { label: string; kind: string }> = {
  pretrain: { label: '预训练', kind: 'compute' }, prefill: { label: 'Prefill', kind: 'compute' }, decode: { label: 'Decode', kind: 'comm' },
};
// dimension order + signature colour (de-RYG hues; state stays RYG elsewhere)
const DIMS: { key: ParDim; label: string; color: string }[] = [
  { key: 'tp', label: 'TP', color: PARALLEL_COLORS.tp },
  { key: 'sp', label: 'SP', color: PARALLEL_COLORS_SP },
  { key: 'ep', label: 'EP', color: PARALLEL_COLORS.ep },
  { key: 'pp', label: 'PP', color: PARALLEL_COLORS.pp },
  { key: 'dp', label: 'DP', color: PARALLEL_COLORS.dp },
];
const DIM_COLOR: Record<ParDim, string> = { tp: PARALLEL_COLORS.tp, sp: PARALLEL_COLORS_SP, ep: PARALLEL_COLORS.ep, pp: PARALLEL_COLORS.pp, dp: PARALLEL_COLORS.dp };
// priority for the "全部" cell colour when a pair is connected by more than one dim (most local first)
const PRIORITY: ParDim[] = ['tp', 'pp', 'ep', 'dp'];
const COLL_LABEL: Record<'ring' | 'a2a' | 'p2p', string> = { ring: 'Ring/AllReduce', a2a: 'All-to-All', p2p: 'P2P 链' };

function hexA(hex: string, a: number): string {
  const h = hex.replace('#', ''); const r = parseInt(h.slice(0, 2), 16), g = parseInt(h.slice(2, 4), 16), b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${a})`;
}

export function CommView({ gen, dark, sync, scope: scopeP, setScope: setScopeP, dim: dimP, setDim: setDimP }: {
  gen: Gen; dark: boolean; sync?: ViewSync;
  scope?: Scope; setScope?: (s: Scope) => void; dim?: DimFilter; setDim?: (d: DimFilter) => void;
}) {
  const visualProfile = useContext(SceneVisualProfileContext);
  const workbenchProfile = visualProfile === 'opRankTime';
  const spec = GENERATIONS[gen];
  const N = spec.totalNpus;

  // 工况 / 时间 come from the shared sync when present (linked with 运行状态 ⇄ 工作台); else local.
  const [locWl] = useState<ParallelWorkload>('pretrain');
  const [locStep, setLocStep] = useState(0);
  const [locPlaying] = useState(false);
  const workload = sync?.workload ?? locWl;
  const step = sync?.step ?? locStep;
  const playing = sync?.playing ?? locPlaying;

  const [scopeL, setScopeL] = useState<Scope>('intra');
  const scope = scopeP ?? scopeL; const setScope = setScopeP ?? setScopeL;
  const [dimL, setDimL] = useState<DimFilter>('all');
  const dim = dimP ?? dimL; const setDim = setDimP ?? setDimL;
  const [tip, setTip] = useState<{ x: number; y: number; t: string } | null>(null);

  const pm = useMemo(() => parallelMap(workload, N), [workload, N]);
  const wlKind = WL[workload].kind;

  // local playback when standalone (no sync provides the clock)
  useEffect(() => {
    if (sync || !playing) return;
    const id = setInterval(() => setLocStep((s) => (s + 1) % 61), 650);
    return () => clearInterval(id);
  }, [sync, playing]);

  // ── the scope's rank set + a coarse-label for the axis ──
  // intra = one model-parallel replica (TP block-diagonal · PP stage chain · EP within, if node-scoped)
  // inter = one representative rank per replica (DP ring across replicas · EP a2a between EP-blocks)
  const M = useMemo(() => {
    const replicaSize = pm.pp * pm.tp;             // ranks in one replica (model-parallel group)
    if (scope === 'intra') {
      const n = Math.min(64, Math.max(pm.tp, replicaSize));   // whole replica (train ≈32, infer =8)
      const ranks = Array.from({ length: n }, (_, i) => i);   // replica 0 = ranks [0, replicaSize)
      return { ranks, n, note: `副本 0 内 ${n} rank（模型并行域：TP×PP${pm.epScope === 'node' ? '×EP' : ''}）`, kind: 'intra' as const };
    }
    const cap = 48, reps = Math.min(cap, pm.dp);
    const ranks = Array.from({ length: reps }, (_, r) => r * replicaSize);   // rank 0 of each replica (tp0·stage0)
    return { ranks, n: reps, note: `${reps}${pm.dp > cap ? '/' + pm.dp : ''} 个副本代表 rank（副本间：DP Ring · EP All-to-All）`, kind: 'inter' as const };
  }, [pm, scope]);

  // per-cell connecting dimension: precompute each row's peer sets (restricted to the scope)
  const conn = useMemo(() => {
    const idxOf = new Map<number, number>(); M.ranks.forEach((k, i) => idxOf.set(k, i));
    const activeDims: ParDim[] = M.kind === 'intra'
      ? (pm.epScope === 'node' ? ['tp', 'pp', 'ep'] : ['tp', 'pp'])   // within a replica: TP/PP (+EP if node-scoped)
      : ['dp', 'ep'];                                                  // across replicas: DP/EP
    // cell[i*n+j] = ParDim connecting ranks i,j (or '' none). diagonal handled separately.
    const cell = new Array<ParDim | ''>(M.n * M.n).fill('');
    for (let i = 0; i < M.n; i++) {
      const ki = M.ranks[i];
      const peerByDim: Partial<Record<ParDim, Set<number>>> = {};
      for (const d of activeDims) peerByDim[d] = new Set(pm.peersOf(ki, d, 256));
      for (let j = 0; j < M.n; j++) {
        if (i === j) continue;
        const kj = M.ranks[j];
        for (const d of PRIORITY) {
          if (!activeDims.includes(d)) continue;
          if (peerByDim[d]?.has(kj)) { cell[i * M.n + j] = d; break; }
        }
      }
    }
    return { cell, activeDims, idxOf };
  }, [M, pm]);

  // ── canvas ──
  const wrapRef = useRef<HTMLDivElement>(null);
  const cvRef = useRef<HTMLCanvasElement>(null);
  const geom = useRef<{ x0: number; y0: number; cs: number; n: number } | null>(null);
  const P = dark
    ? { bg: '#121418', grid: 'rgba(255,255,255,0.05)', ink: 'rgba(255,255,255,0.86)', ink2: 'rgba(255,255,255,0.55)', mut: '#5A6172', empty: 'rgba(255,255,255,0.04)', diag: 'rgba(255,255,255,0.14)', frame: 'rgba(255,255,255,0.10)' }
    : { bg: '#fbfbfd', grid: 'rgba(67,105,239,0.07)', ink: 'rgba(0,0,0,0.80)', ink2: 'rgba(0,0,0,0.52)', mut: '#9aa3b2', empty: 'rgba(0,0,0,0.035)', diag: 'rgba(0,0,0,0.16)', frame: 'rgba(0,0,0,0.10)' };

  const draw = useCallback(() => {
    const cv = cvRef.current, wrap = wrapRef.current; if (!cv || !wrap) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const W = wrap.clientWidth, H = wrap.clientHeight;
    if (cv.width !== W * dpr || cv.height !== H * dpr) { cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + 'px'; cv.style.height = H + 'px'; }
    const ctx = cv.getContext('2d')!; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H); ctx.fillStyle = P.bg; ctx.fillRect(0, 0, W, H);
    const n = M.n;
    // live traffic per rank (0..1) — brightens the cells; disabled visual pulse when paused is fine (static snapshot)
    const load = M.ranks.map((k) => cardLoad01(k, wlKind, step));
    // square matrix, centred, leaving a label gutter
    const PAD = 20, GUT = 34, top = 40;
    const avail = Math.min(W - PAD * 2 - GUT, H - top - PAD - GUT);
    const cs = Math.max(3, Math.floor(avail / n));
    const m = cs * n, x0 = PAD + GUT + Math.max(0, (W - PAD * 2 - GUT - m) / 2), y0 = top + Math.max(0, (H - top - PAD - GUT - m) / 2);
    geom.current = { x0, y0, cs, n };

    ctx.font = '12.5px Inter'; ctx.fillStyle = P.ink2; ctx.textAlign = 'left';
    ctx.fillText(`${M.note} · 行/列 = rank · 格色 = 并行维度 · 亮度 = 实时流量`, PAD, 22);

    // cells
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const x = x0 + j * cs, y = y0 + i * cs, gp = cs > 6 ? 1 : 0.4;
      if (i === j) { ctx.fillStyle = P.diag; ctx.fillRect(x, y, cs - gp, cs - gp); continue; }
      const d = conn.cell[i * n + j];
      if (!d) { ctx.fillStyle = P.empty; ctx.fillRect(x, y, cs - gp, cs - gp); continue; }
      const lit = dim === 'all' || dim === d;
      if (!lit) { ctx.fillStyle = P.empty; ctx.fillRect(x, y, cs - gp, cs - gp); continue; }
      const traffic = (load[i] + load[j]) / 2;                 // 0..1 live traffic on this pair
      const a = 0.32 + traffic * 0.68;                          // brightness = traffic (dim hue = relationship)
      ctx.fillStyle = hexA(DIM_COLOR[d], a); ctx.fillRect(x, y, cs - gp, cs - gp);
    }
    // frame + axis ticks
    ctx.strokeStyle = P.frame; ctx.lineWidth = 1; ctx.strokeRect(x0 - 0.5, y0 - 0.5, m + 1, m + 1);
    const lab = n <= 16 ? 1 : n <= 40 ? 4 : 8;
    ctx.fillStyle = P.mut; ctx.font = `9px ${MONO}`;
    for (let i = 0; i < n; i += lab) {
      ctx.textAlign = 'right'; ctx.fillText('' + M.ranks[i], x0 - 5, y0 + i * cs + cs - 1);
      ctx.textAlign = 'center'; ctx.fillText('' + M.ranks[i], x0 + i * cs + cs / 2, y0 - 5);
    }
    ctx.textAlign = 'left'; ctx.fillStyle = P.ink2; ctx.font = '10px Inter';
    ctx.fillText('源 rank ↓', PAD, y0 + m / 2); ctx.fillText('目的 rank →', x0, y0 + m + GUT - 8);
  }, [M, conn, dim, wlKind, step, P]);

  useEffect(() => { draw(); }, [draw]);
  useEffect(() => { const onR = () => draw(); window.addEventListener('resize', onR); return () => window.removeEventListener('resize', onR); }, [draw]);

  const onMove = (e: React.MouseEvent) => {
    const g = geom.current; if (!g) return;
    const r = cvRef.current!.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    const j = Math.floor((mx - g.x0) / g.cs), i = Math.floor((my - g.y0) / g.cs);
    if (i < 0 || j < 0 || i >= g.n || j >= g.n) { if (tip) setTip(null); return; }
    const ki = M.ranks[i], kj = M.ranks[j];
    if (i === j) { setTip({ x: e.clientX, y: e.clientY, t: `rank ${ki} · 自身` }); return; }
    const d = conn.cell[i * g.n + j];
    if (!d) { setTip({ x: e.clientX, y: e.clientY, t: `rank ${ki} ↔ ${kj} · 无直接集合通信` }); return; }
    const traffic = Math.round((cardLoad01(ki, wlKind, step) + cardLoad01(kj, wlKind, step)) / 2 * 100);
    setTip({ x: e.clientX, y: e.clientY, t: `rank ${ki} ↔ ${kj} · ${d.toUpperCase()} ${COLL_LABEL[pm.collectiveOf(d)]} · 流量 ${traffic}%` });
  };

  const dimInfo = (d: ParDim) => {
    const meta = d === 'sp' ? { label: 'SP 序列并行', level: '与 TP 同域', comm: 'AllGather+ReduceScatter' } : PARTITION_META[d as 'tp' | 'pp' | 'dp' | 'ep'];
    return { ...meta, deg: pm.groupCount(d), coll: pm.collectiveOf(d) };
  };
  const decomp = STEP_DECOMP[workload];

  return (
    <div data-theme={dark ? 'dark' : 'light'} style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', background: workbenchProfile ? 'var(--background-elevated)' : 'var(--bg)', overflow: 'hidden' }}>
      {/* toolbar — hidden in workbench (filters live in center-pill dropdown) */}
      {!workbenchProfile && (
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, flexWrap: 'wrap', padding: '8px 14px', borderBottom: '1px solid var(--bd)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={LBL}>范围</span>
          {([['intra', '副本内 (TP/PP/EP)'], ['inter', '副本间 (DP/EP)']] as [Scope, string][]).map(([s, l]) => (<button key={s} onClick={() => setScope(s)} style={{ padding: '4px 11px', fontSize: 11.5, borderRadius: 8, cursor: 'pointer', ...navBtn(scope === s) }}>{l}</button>))}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={LBL}>维度</span>
          <button onClick={() => setDim('all')} style={{ padding: '4px 11px', fontSize: 11.5, borderRadius: 8, cursor: 'pointer', ...navBtn(dim === 'all') }}>全部</button>
          {DIMS.map((d) => (<button key={d.key} onClick={() => setDim(d.key)} style={{ padding: '4px 11px', fontSize: 11.5, borderRadius: 8, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 5, ...toggleBtn(dim === d.key, d.color) }}><span style={{ width: 8, height: 8, borderRadius: 2, background: dim === d.key ? 'currentColor' : d.color }} />{d.label}</button>))}
        </div>
        <div style={{ flex: 1 }} />
      </div>
      )}

      {/* body: matrix + rail */}
      <div style={{ flex: 1, display: 'flex', gap: 12, padding: '8px 14px 12px', minHeight: 0 }}>
        <div ref={wrapRef} style={{ flex: 1, minWidth: 0, position: 'relative', borderRadius: 12, ...(workbenchProfile ? {} : { border: '1px solid var(--bd)' }), overflow: 'hidden', background: 'var(--panel-solid)' }}>
          <canvas ref={cvRef} onMouseMove={onMove} onMouseLeave={() => setTip(null)} style={{ display: 'block', width: '100%', height: '100%', cursor: 'crosshair' }} />
        </div>

        {/* rail */}
        <div style={{ width: 272, flexShrink: 0, overflowY: 'auto', borderRadius: 12, ...(workbenchProfile ? { boxShadow: 'var(--shadow-sm)' } : { border: '1px solid var(--bd)' }), background: 'var(--panel-solid)', padding: '12px 14px' }}>
          <div style={{ fontSize: 14, fontWeight: 700, color: '#5b86ff', marginBottom: 2 }}>通信全景 · rank×rank 矩阵</div>
          <div style={{ fontSize: 11, color: 'var(--tx3)', marginBottom: 10 }}>不分层级 · 只讲「谁跟谁通信」（层级在 3D/平面视图看「在哪」）</div>

          {/* parallel-map truth */}
          <div style={{ borderTop: '1px solid var(--bd)', paddingTop: 9, marginBottom: 8 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--tx2)', marginBottom: 4 }}>并行映射（parallelMap 真值 · {WL[workload].label}）</div>
            <div style={{ fontSize: 11, color: 'var(--tx)', fontFamily: MONO, ...TNUM }}>{pm.cfg}</div>
            <div style={{ fontSize: 10, color: 'var(--tx3)', marginTop: 3, lineHeight: 1.5 }}>{pm.real}</div>
            <div style={{ fontSize: 9.5, color: 'var(--tx3)', marginTop: 2, lineHeight: 1.5 }}>{pm.approxNote}</div>
          </div>

          {/* per-dimension footprint */}
          <div style={{ borderTop: '1px solid var(--bd)', paddingTop: 9, marginBottom: 8 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--tx2)', marginBottom: 6 }}>维度足迹（点上方维度按钮过滤）</div>
            {DIMS.map((d) => {
              const info = dimInfo(d.key); const on = dim === 'all' || dim === d.key;
              return (
                <div key={d.key} onClick={() => setDim(dim === d.key ? 'all' : d.key)} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '4px 6px', borderRadius: 7, cursor: 'pointer', opacity: on ? 1 : 0.45, background: dim === d.key ? hexA(d.color, 0.12) : 'transparent', marginBottom: 2 }}>
                  <span style={{ width: 11, height: 11, borderRadius: 3, background: d.color, flexShrink: 0 }} />
                  <span style={{ width: 26, fontSize: 11.5, fontWeight: 700, color: 'var(--tx)' }}>{d.label}</span>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 10.5, color: 'var(--tx2)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{info.label} · ×{info.deg}</div>
                    <div style={{ fontSize: 9, color: 'var(--tx3)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{COLL_LABEL[info.coll]} · {info.level}</div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* real comm share for this workload */}
          <div style={{ borderTop: '1px solid var(--bd)', paddingTop: 9, marginBottom: 8 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--tx2)', marginBottom: 5 }}>step 时间分解（{WL[workload].label} · 盘古 Pro MoE）</div>
            {decomp.map(({ label, frac, color }) => (
              <div key={label} style={{ marginBottom: 5 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10.5, marginBottom: 2 }}><span style={{ color: 'var(--tx)' }}>{label}</span><span style={{ color: 'var(--tx3)', fontFamily: MONO }}>{Math.round(frac * 100)}%</span></div>
                <div style={{ height: 7, borderRadius: 4, background: 'var(--btn)', overflow: 'hidden' }}><div style={{ height: '100%', width: `${frac * 100}%`, background: color }} /></div>
              </div>
            ))}
          </div>

          {/* legend */}
          <div style={{ borderTop: '1px solid var(--bd)', paddingTop: 9 }}>
            <div style={{ fontSize: 10.5, color: 'var(--tx3)', lineHeight: 1.6 }}>
              格色 = 连接这对 rank 的<b style={{ color: 'var(--tx2)' }}>并行维度</b>（去 RYG，避开状态色）· 亮度 = 该对<b style={{ color: 'var(--tx2)' }}>实时流量</b> · 对角 = 自身 · 空 = 无直接集合通信。<br />
              块对角=TP（节点内）· 阶梯带=PP · 密块=EP · 远带=DP。
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 8, fontSize: 10, color: 'var(--tx3)' }}>
              <span>低流量</span>
              <div style={{ flex: 1, height: 8, borderRadius: 4, background: `linear-gradient(90deg, ${hexA(loadColor(0.3), 0.35)}, ${DIM_COLOR.tp})` }} />
              <span>高流量</span>
            </div>
          </div>
        </div>
      </div>

      {tip && (
        <div style={{ position: 'fixed', left: tip.x + 12, top: tip.y + 12, pointerEvents: 'none', zIndex: 30, background: 'var(--panel-solid)', border: '1px solid var(--bd2)', borderRadius: 8, padding: '5px 9px', fontSize: 11.5, color: 'var(--tx)', boxShadow: 'var(--shadow-sm)', whiteSpace: 'nowrap' }}>{tip.t}</div>
      )}
    </div>
  );
}
