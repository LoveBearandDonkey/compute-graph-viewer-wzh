/**
 * DualModeMonitor — 动态监控双模式（原型第四组件工程化）。
 *
 *   · 实时热力大盘（SRE 视角 / 火灾报警器）：秒级刷新聚合负载，MoE 浪涌爆红并自动触发 Trace 快照。
 *   · 时间轴回放（优化工程师视角 / 黑匣子）：拖拽/播放逐帧查看 AllToAll 星状路由与热点漂移。
 *
 * 关键：数据全部来自单一真值源（cardLoad01 / op-schedule / REPLAY 事件窗口），与右侧 3D、层级图
 * 共用同一 step/工况/播放，因此左右同屏同一个世界。点某卡 → 选中 rank（驱动右侧 3D 高亮）。
 */
import { useMemo } from 'react';
import {
  cardMetric01, loadColor, loadState, inReplayEvent, REPLAY, NPUS_PER_NODE,
  type ParallelWorkload,
} from '../scene/data';
import { opAtCursor, flowLayout } from '../scene/op-schedule';

type Mode = 'heat' | 'replay';
const CPB = NPUS_PER_NODE;   // 8 卡 / Host
const STEP_MAX = REPLAY.stepMax;
const wlKindOf = (w: ParallelWorkload) => (w === 'decode' ? 'comm' : 'compute');

// 网格几何（SVG 内统一绘制，便于叠加星状路由）
const LBLW = 58, CW = 26, CH = 20, GAP = 4, TOPH = 20;
const cellX = (col: number) => LBLW + col * (CW + GAP);
const cellY = (row: number) => TOPH + row * (CH + GAP);
const btn: React.CSSProperties = { padding: '4px 10px', fontSize: 11.5, borderRadius: 8, cursor: 'pointer' };
const SECONDARY: React.CSSProperties = { border: '1px solid var(--button-secondary-border)', background: 'var(--button-secondary-bg)', color: 'var(--foreground-muted)' };
function tab(on: boolean): React.CSSProperties { return on ? { border: '1px solid var(--primary)', background: 'var(--primary)', color: 'var(--primary-foreground)', fontWeight: 600 } : { ...SECONDARY }; }
const MONO = "'JetBrains Mono','Consolas',ui-monospace,monospace";

export function DualModeMonitor({
  mode, setMode, workload, step, setStep, playing, setPlaying, sel, onSelectRank, baseHost, dark,
}: {
  mode: Mode; setMode: (m: Mode) => void;
  workload: ParallelWorkload; step: number; setStep: (n: number) => void; playing: boolean; setPlaying: (b: boolean) => void;
  sel: number | null; onSelectRank: (r: number | null) => void; baseHost: number; dark: boolean;
}) {
  const wlKind = wlKindOf(workload);
  const rows = 8, cols = CPB;                       // 8 Host × 8 卡 = 64 卡窗口
  const base = baseHost * CPB;
  const svgW = LBLW + cols * (CW + GAP);
  const svgH = TOPH + rows * (CH + GAP) + 4;

  const cursor01 = (step % (STEP_MAX + 1)) / STEP_MAX;
  const curOp = useMemo(() => opAtCursor(workload, cursor01), [workload, cursor01]);
  // 当前是否处于 EP AllToAll 阶段（含被计算掩盖的并发通信）→ 画星状路由
  const a2aActive = useMemo(() => {
    if (curOp.kind === 'comm' && curOp.coll === 'a2a') return true;
    const fl = flowLayout(workload);
    return fl.hidden.some((h) => h.op.coll === 'a2a' && cursor01 >= h.x && cursor01 < h.x + h.w);
  }, [curOp, workload, cursor01]);

  // 窗口内每卡的度量值（真实模型）→ 状态色
  const vals = useMemo(() => Array.from({ length: rows * cols }, (_, i) => cardMetric01(base + i, 'util', wlKind, step)), [base, rows, cols, wlKind, step]);
  // 热点集合（星状路由的落点）：窗口内负载最高的若干卡
  const hot = useMemo(() => {
    const idx = vals.map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v).slice(0, 4).map((o) => o.i);
    return idx;
  }, [vals]);
  const srcCell = hot.length ? (hot[0] + 3) % (rows * cols) : 0;   // 源卡（与热点错开，画“发散”）

  const inEvt = inReplayEvent(step);
  const PHASES: [number, number, string][] = [[0, 25, 'Forward · Dense MatMul'], [25, 40, 'MoE Gating（Top-K 路由决策）'], [40, 62, 'AllToAll 分发 + Expert FFN'], [62, 75, 'AllToAll 回收 / Combine'], [75, 101, 'Backward · 梯度 + AllReduce']];
  const tPct = Math.round(cursor01 * 100);
  const phase = PHASES.find((p) => tPct >= p[0] && tPct < p[1]) ?? PHASES[4];

  const cells: React.ReactNode[] = [];
  for (let r = 0; r < rows; r++) {
    cells.push(<text key={`hl${r}`} x={6} y={cellY(r) + CH / 2 + 3} fontSize={9} fill="var(--tx3)" style={{ fontFamily: MONO }}>{`H${baseHost + r}`}</text>);
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c, rank = base + i, v = vals[i];
      const isSel = sel === rank, isHot = a2aActive && hot.includes(i), off = loadState(v) < 0;
      cells.push(
        <rect key={`c${i}`} x={cellX(c)} y={cellY(r)} width={CW} height={CH} rx={3}
          fill={loadColor(v)} stroke={isSel ? '#4369ef' : isHot ? '#f85149' : (dark ? '#0d1117' : '#ffffff')} strokeWidth={isSel ? 2.2 : isHot ? 1.8 : 0.8}
          style={{ cursor: 'pointer' }} onClick={() => onSelectRank(isSel ? null : rank)}>
          <title>{`rank ${rank} · util ${Math.round(v * 100)}%`}</title>
        </rect>,
      );
      if (off) cells.push(<line key={`o${i}`} x1={cellX(c)} y1={cellY(r)} x2={cellX(c) + CW} y2={cellY(r) + CH} stroke="var(--tx3)" strokeWidth={0.8} />);
    }
  }
  // 星状 AllToAll 路由（回放 + a2a 阶段）：源卡 → 热点专家卡，红线=拥堵、紫线=普通
  const routes: React.ReactNode[] = [];
  if (mode === 'replay' && a2aActive) {
    const sc = srcCell % (rows * cols), sx = cellX(sc % cols) + CW / 2, sy = cellY(Math.floor(sc / cols)) + CH / 2;
    hot.forEach((h, k) => {
      const hx = cellX(h % cols) + CW / 2, hy = cellY(Math.floor(h / cols)) + CH / 2;
      routes.push(<path key={`rt${k}`} d={`M${sx},${sy} C${(sx + hx) / 2},${Math.min(sy, hy) - 26} ${(sx + hx) / 2},${Math.min(sy, hy) - 26} ${hx},${hy}`} fill="none" stroke={k === 0 ? '#f85149' : '#a371f7'} strokeWidth={k === 0 ? 2.6 : 1.6} strokeDasharray="5 4" opacity={0.9} />);
    });
    routes.push(<circle key="src" cx={sx} cy={sy} r={4} fill="#d29922" />);
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
        <button onClick={() => setMode('heat')} style={{ ...btn, ...tab(mode === 'heat') }}>实时热力大盘</button>
        <button onClick={() => setMode('replay')} style={{ ...btn, ...tab(mode === 'replay') }}>时间轴回放</button>
        <span style={{ fontSize: 9.5, color: 'var(--tx3)', alignSelf: 'center', marginLeft: 4 }}>{mode === 'heat' ? 'SRE 视角 · 秒级巡航' : '优化工程师视角 · 逐帧'}</span>
      </div>

      {/* 告警条：回放事件窗口 = MoE 浪涌，自动触发 Trace 快照 */}
      {inEvt && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 10px', borderRadius: 8, background: 'rgba(248,81,73,0.14)', border: '1px solid rgba(248,81,73,0.5)', fontSize: 10.5, color: '#f85149' }}>
          <span><b>ALERT</b> 机柜 C{REPLAY.evtCab} MoE AllToAll 浪涌：热点卡负载 &gt;95%，已自动触发 5-Step Trace 快照。</span>
          {mode === 'heat' && <button onClick={() => setMode('replay')} style={{ ...btn, padding: '2px 8px', marginLeft: 'auto', ...tab(false) }}>→ 分析此段 Trace</button>}
        </div>
      )}

      {/* 回放控制条 */}
      {mode === 'replay' && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <button onClick={() => setPlaying(!playing)} style={{ ...btn, ...tab(playing) }}>{playing ? '暂停' : '▶ 播放'}</button>
          <input type="range" min={0} max={STEP_MAX} value={step % (STEP_MAX + 1)} onChange={(e) => setStep(+e.target.value)} style={{ flex: 1 }} />
          <span style={{ fontSize: 10, fontFamily: MONO, color: 'var(--tx2)', minWidth: 150 }}>t={tPct}% · {phase[2]}</span>
        </div>
      )}

      <div style={{ background: 'var(--panel-solid)', border: '1px solid var(--bd)', borderRadius: 8, padding: 8, overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${svgW} ${svgH}`} width="100%" style={{ display: 'block', minWidth: svgW * 0.7 }}>
          <text x={6} y={12} fontSize={9.5} fill="var(--tx3)">窗口：Host H{baseHost}–H{baseHost + rows - 1} × {cols} 卡（{rows * cols} 卡采样 · 点卡下钻）</text>
          {cells}
          {routes}
        </svg>
      </div>

      {/* 图例 + 说明 */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', fontSize: 9.5, color: 'var(--tx2)' }}>
        {[['空闲 <40%', loadColor(0.2)], ['中 40–70%', loadColor(0.55)], ['繁忙 >70%', loadColor(0.9)]].map(([l, c]) => (
          <span key={l} style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}><span style={{ width: 9, height: 9, borderRadius: 2, background: c as string }} />{l}</span>
        ))}
        {mode === 'replay' && <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}><span style={{ width: 12, height: 2, background: '#a371f7' }} />紫=AllToAll 逻辑路由（红=拥堵）</span>}
      </div>
      <div style={{ fontSize: 10.5, color: 'var(--tx2)', lineHeight: 1.55 }}>
        {mode === 'heat'
          ? (inEvt ? '浪涌事件：MoE 极端路由把 Token 洪水涌向热点专家组 → 局部爆红。系统在后台自动开启 5 个 Step 深层打点（Triggered Dump），随后自动关闭探针。' : '集群巡航中：秒级轮询聚合指标（负载/显存/温度），采集开销极低、不影响训练。热力呼吸属正常波动。')
          : (a2aActive ? `回放 t=${tPct}%：Token 从源卡发出 AllToAll 星状路由（紫线=流量、红线=拥堵链路），热点专家组随数据分布漂移。拖动时间轴逐帧对照右侧 3D 定位掩盖失败点。` : `回放 t=${tPct}%：${phase[2]}。此阶段无跨节点 MoE 路由。`)}
      </div>
    </div>
  );
}
