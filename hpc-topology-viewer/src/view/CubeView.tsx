/**
 * CubeView — P1「立方体重排」新视图（加法·独立·不碰 FullPodScene）。
 *
 * 这是承载完整监控愿景的新工作区的起点：同一批卡（rank），按不同并行轴「换一种堆法」，
 * 靠飞行动画在 物理 / TP / PP / DP / EP 视图间重排，状态热力叠在上面。核心价值当场可见——
 * 「注入异常」把某个并行组标红：物理视图里它散成一片，切到对应并行视图它 snap 成一整块，
 * 「异常的形状」直接对应根因类别（方案 §9.1）。
 *
 * 铁律：切视图（结构）改的是「位置」；时间/状态改的是「颜色」——两者不串台。
 * 几何全部来自已验证的 layout.ts（双射不变量），配色来自 data.ts 的 loadRGB（红黄绿=状态唯一色）。
 * 逻辑视图里不画物理刀片/机柜脚手架（那在逻辑重排里没有意义）——只画卡本身。
 */
import { useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, GizmoHelper, GizmoViewcube } from '@react-three/drei';
import * as THREE from 'three';
import {
  GENERATIONS, NODES_PER_CAB, NPUS_PER_NODE, loadRGB, loadState, stateColor, STATE_LABELS, cardLoad01, parallelMap, PARALLEL_COLORS,
  PARTITION_PALETTE,
  type Gen, type ViewSync, type ParallelWorkload, type ParDim, type PartitionDim, type LevelKey,
} from '../scene/data';
import { layoutOf, LAYOUT_VIEWS, LAYOUT_LABEL, type LayoutView } from '../scene/layout';
import { deploymentOf, stageLayerRange } from '../scene/deployment';
import { OP_SCHEDULE, phaseMix, flowLayout, opAtCursor, pipeline1F1B, type OpKind } from '../scene/op-schedule';
import { SceneVisualProfileContext, sceneSurface } from '../scene/visual-profile';
import '../vendor/swimlane-task/pattern.css';
import '../vendor/swimlane-task/pattern.js';

const PITCH = 0.42;                    // 每格世界尺寸
const BOX = 0.72 * PITCH;              // 卡块边长（略小于格，留缝）
const MONO = "'JetBrains Mono','Consolas',ui-monospace,monospace";
// 算子 kind → 色（与 STEP_DECOMP 一致：计算青 / 通信红 / 访存紫）
const OP_COL: Record<OpKind, string> = { compute: '#22d3ee', comm: '#ff4b7b', mem: '#a78bfa' };
const OP_KIND_LBL: Record<OpKind, string> = { compute: '计算', comm: '通信', mem: '访存' };
const CAB_CARDS = NODES_PER_CAB * NPUS_PER_NODE;   // 64 卡 / 机柜（Pod 内物理分组）

function hexRGB(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

// ── 聚合粒度（来自层级轴）：LevelKey → 每个「聚合单元」含多少张连续 rank（物理近邻）。
//    严格对齐 hw-native-sys L0→L7 与参考 8 级详表：整个渲染模型 = 1 个 L6 集群（8192 卡·万卡级），
//    万卡规模落在 Cluster（L6），而非 Pod。逐级换算：
//      L3 Host = 8 卡；L4 Pod · UBL128 = 128 卡（= 16 Host = 2 机柜，Scale-Up 域）；
//      L5 服务池 = 4 Pod = 512 卡；L6 集群 = 整个模型（万卡·一块）；L7 全球 = 整体一块。
//    card/die/core = 满 rank 粒度（1 卡/单元）；机柜(cab)只是 Pod 内物理分组(64)，不是层级。
function aggUnitCards(level: LevelKey | undefined): number {
  switch (level) {
    case 'node': return NPUS_PER_NODE;                 // L3 Host = 8 卡
    case 'cab': return CAB_CARDS;                      // 机柜（物理分组，非层级）= 64 卡
    case 'super': return CAB_CARDS * 2;                // L4 Pod · UBL128 = 128 卡（16 Host = 2 机柜）
    case 'pool': return CAB_CARDS * 8;                 // L5 服务池 = 4 Pod = 512 卡（PODS_PER_POOL=4）
    case 'cluster': return Infinity;                   // L6 集群 = 整个模型（万卡·一块）
    case 'global': return Infinity;                    // L7 全球（整体一块）
    default: return 1;                                 // L2 Chip / L1 Die / L0 Core → 满粒度
  }
}
const nearCols = (n: number) => Math.max(1, Math.ceil(Math.sqrt(n)));
const AGG_SPREAD = 2.0;   // 聚合单元的网格间距倍率（比逐卡稀疏，让宏观大方块之间留缝、不糊成一片）

// 聚合布局：把 N 张卡按 aggUnitCards 分成 U 个单元，铺成近方阵（居中坐标，world unit）。
// size===1 时直接透传逐 rank 的 lay.pos（零行为变化）。返回单元↔rank 的双向映射。
interface AggLayout {
  count: number; cells: { x: number; y: number; z: number }[]; cols: number; rows: number; yExtent: number; size: number;
  rankOfUnit: (u: number) => number;                 // 代表 rank（首个成员）
  unitOfRank: (rank: number) => number;
  membersOfUnit: (u: number) => { start: number; end: number };
}
function aggregateOf(level: LevelKey | undefined, layPos: { x: number; y: number; z: number }[], N: number): AggLayout {
  const size = Math.min(N, aggUnitCards(level));
  if (size <= 1) {
    return {
      count: N, cells: layPos, cols: 0, rows: 0, yExtent: 0, size: 1,
      rankOfUnit: (u) => u, unitOfRank: (r) => r,
      membersOfUnit: (u) => ({ start: u, end: u + 1 }),
    };
  }
  const count = Math.ceil(N / size);
  const cols = nearCols(count), rows = Math.ceil(count / cols);
  const cx = (cols - 1) / 2, cz = (rows - 1) / 2;
  // 稀疏网格：坐标间距 ×AGG_SPREAD×PITCH，大方块之间留缝；cols/rows 同比放大供相机取景。
  const cells = Array.from({ length: count }, (_, u) => ({ x: ((u % cols) - cx) * AGG_SPREAD * PITCH, y: 0, z: (Math.floor(u / cols) - cz) * AGG_SPREAD * PITCH }));
  return {
    count, cells, cols: cols * AGG_SPREAD * PITCH, rows: rows * AGG_SPREAD * PITCH, yExtent: 0, size,
    rankOfUnit: (u) => u * size,
    unitOfRank: (r) => Math.floor(r / size),
    membersOfUnit: (u) => ({ start: u * size, end: Math.min(N, (u + 1) * size) }),
  };
}

export type AnomalyDim = 'none' | 'tp' | 'pp' | 'dp' | 'ep';
export const ANOM_LABEL: Record<AnomalyDim, string> = { none: '无', tp: 'TP 组', pp: 'PP 级', dp: 'DP 副本', ep: 'EP 桶' };
// 每个维度的异常「真实语义 + 物理形状」——诚实区分「散布维(re-layout 必需)」与「结构维(物理已有结构)」。
const ANOM_TYPE: Record<Exclude<AnomalyDim, 'none'>, { scatter: '散布维' | '结构维'; tag: string }> = {
  pp: { scatter: '散布维', tag: 're-layout 必需' },
  ep: { scatter: '散布维', tag: 're-layout 必需' },
  tp: { scatter: '结构维', tag: '物理已有结构' },
  dp: { scatter: '结构维', tag: '物理已有结构' },
};
const ANOM_NOTE: Record<Exclude<AnomalyDim, 'none'>, string> = {
  pp: 'PP 级 0 = 每 PP 台 Host 的同一流水级 · 物理散成条纹 → 切 PP 视图 snap 成一整条竖板。散布维,不重排几乎看不出成组。',
  ep: 'EP 桶 0 = 持有第 0 号专家分桶的所有 rank(每个 A2A 域各出一员,桶↔卡非 1:1)· 物理散成周期条带 → 切 EP 视图 snap 成一面桶墙。散布维,不重排看不出成组(热点专家桶的典型形状)。',
  tp: 'TP 切片 0 = 每台 Host 的第 0 张卡(片内张量分片相同)· 均匀点阵散布 → TP 视图里是规则点阵。结构维:TP 是 Host 内 8 卡并行,故障多为局部,物理视图已能定位。',
  dp: 'DP 副本 0 = 连续几台 Host 的一份模型拷贝 · 物理半聚集 → DP 视图里是干净一块。结构维:物理已有块状结构,重排只是更规整。',
};

// ── 卡阵列（唯一被重排的对象）：位置来自 layout（飞行动画 lerp），颜色来自负载场（逐 step 重染） ──
//    拾取：instanceId == rank；选中/悬停高亮跟随卡的实时(动画中)位置。
const PEER_MAX = 96;   // 对端高亮上限（peersOf 采样）
// 多维叠加对端组（白皮书「一个 rank 的多维身份」）：选中卡的 TP/PP/DP/EP 通信组同时显示，
// active（游标正扫到该维通信算子）加亮加大，其余维淡显——placement 常在，runtime event 点亮。
export interface PeerSet { dim: string; ranks: number[]; color: string; active: boolean; }
function CubeField({ cells, colorOf, recolorKey, onSettleChange, selected, hover, onPick, onHover, peerSets, boxXZ = BOX, boxY = 0.16 }: {
  cells: { x: number; y: number; z: number }[]; colorOf: (k: number) => [number, number, number]; recolorKey: number;
  onSettleChange?: (settling: boolean) => void;
  selected: number | null; hover: number | null;
  onPick: (rank: number | null) => void; onHover: (rank: number | null) => void;
  peerSets: PeerSet[];                   // 选中卡在各并行维的通信对端（四维同时呈现）
  boxXZ?: number; boxY?: number;         // 方块尺寸（聚合单元更大）
}) {
  const N = cells.length;
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const selRef = useRef<THREE.Mesh>(null);
  const hovRef = useRef<THREE.Mesh>(null);
  const peerRefs = useRef<(THREE.InstancedMesh | null)[]>([]);
  const m2 = useMemo(() => new THREE.Matrix4(), []);
  const cur = useRef<{ x: Float32Array; y: Float32Array; z: Float32Array } | null>(null);
  const target = useRef(cells);
  const settling = useRef(true);
  if (!cur.current || cur.current.x.length !== N) {
    const x = new Float32Array(N), y = new Float32Array(N), z = new Float32Array(N);
    for (let k = 0; k < N; k++) { x[k] = cells[k].x; y[k] = cells[k].y; z[k] = cells[k].z; }
    cur.current = { x, y, z };
  }
  // 视图切换 → 新目标位置，开始飞行
  useEffect(() => { target.current = cells; settling.current = true; onSettleChange?.(true); }, [cells, onSettleChange]);

  // 每帧：高亮跟随实时位置（始终）+ 位置 lerp 向目标（稳定后停写省 CPU）
  const m = useMemo(() => new THREE.Matrix4(), []);
  useFrame(() => {
    const mesh = meshRef.current, c = cur.current; if (!mesh || !c) return;
    const hlScale = boxXZ / BOX;   // 聚合单元更大 → 高亮线框同比例放大
    const place = (ref: React.RefObject<THREE.Mesh>, idx: number | null) => {
      if (!ref.current) return;
      if (idx == null || idx < 0 || idx >= N) { ref.current.visible = false; return; }
      ref.current.visible = true; ref.current.position.set(c.x[idx], c.y[idx], c.z[idx]); ref.current.scale.setScalar(hlScale);
    };
    place(selRef, selected); place(hovRef, hover === selected ? null : hover);
    // 多维对端高亮：四个并行维的通信组同时跟随实时位置（active 维加大，其余淡显）
    peerSets.forEach((set, si) => {
      const pm2 = peerRefs.current[si]; if (!pm2) return;
      const n = Math.min(set.ranks.length, PEER_MAX);
      const sXZ = set.active ? 1.7 : 1.42, sY = set.active ? 0.34 : 0.24;
      for (let i = 0; i < n; i++) { const k = set.ranks[i]; if (k < 0 || k >= N) continue; m2.makeScale(boxXZ * sXZ, sY, boxXZ * sXZ); m2.setPosition(c.x[k], c.y[k] + 0.02, c.z[k]); pm2.setMatrixAt(i, m2); }
      pm2.count = n; pm2.instanceMatrix.needsUpdate = true;
    });
    if (!settling.current) return;
    let moving = false;
    for (let k = 0; k < N; k++) {
      const tx = target.current[k].x, ty = target.current[k].y, tz = target.current[k].z;
      const nx = c.x[k] + (tx - c.x[k]) * 0.16, ny = c.y[k] + (ty - c.y[k]) * 0.16, nz = c.z[k] + (tz - c.z[k]) * 0.16;
      if (Math.abs(tx - nx) > 0.004 || Math.abs(ty - ny) > 0.004 || Math.abs(tz - nz) > 0.004) moving = true;
      c.x[k] = nx; c.y[k] = ny; c.z[k] = nz;
      m.makeScale(boxXZ, boxY, boxXZ); m.setPosition(nx, ny, nz); mesh.setMatrixAt(k, m);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (!moving) { settling.current = false; onSettleChange?.(false); }
  });

  // 颜色：由 colorOf 决定（状态红黄绿 或 策略互斥着色）。step / 异常 / 着色 / 粒度变化时重染一次。
  useEffect(() => {
    const mesh = meshRef.current; if (!mesh) return;
    const col = new THREE.Color();
    for (let k = 0; k < N; k++) { const [r, g, b] = colorOf(k); mesh.setColorAt(k, col.setRGB(r / 255, g / 255, b / 255)); }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recolorKey, N]);

  return (
    <>
      <instancedMesh
        ref={meshRef} args={[undefined, undefined, N]} frustumCulled={false}
        onClick={(e) => { e.stopPropagation(); onPick(e.instanceId ?? null); }}
        onPointerMove={(e) => { e.stopPropagation(); if (e.instanceId !== undefined && e.instanceId !== hover) onHover(e.instanceId); }}
        onPointerOut={() => onHover(null)}
      >
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial metalness={0.02} roughness={0.78} />
      </instancedMesh>
      {/* 选中高亮：软件靛色线框（选中的是一个 rank = 软件对象），比卡略大以包住 */}
      <mesh ref={selRef} visible={false} raycast={() => null}>
        <boxGeometry args={[BOX * 1.5, 0.28, BOX * 1.5]} />
        <meshBasicMaterial color="#4369ef" wireframe transparent opacity={0.95} />
      </mesh>
      {/* 悬停高亮：更淡 */}
      <mesh ref={hovRef} visible={false} raycast={() => null}>
        <boxGeometry args={[BOX * 1.35, 0.24, BOX * 1.35]} />
        <meshBasicMaterial color="#8ba3f2" wireframe transparent opacity={0.6} />
      </mesh>
      {/* 多维通信组高亮：选中卡的 TP/PP/DP/EP 对端同时呈现（各维签名色线框），
          游标扫到某维通信算子 → 该维加亮（0.92），其余维淡显（0.26）——「同时存在」可见 */}
      {peerSets.map((set, si) => (
        <instancedMesh key={set.dim} ref={(el) => { peerRefs.current[si] = el; }} args={[undefined, undefined, PEER_MAX]} frustumCulled={false} raycast={() => null}>
          <boxGeometry args={[1, 1, 1]} />
          <meshBasicMaterial color={set.color} wireframe transparent opacity={set.active ? 0.92 : 0.26} />
        </instancedMesh>
      ))}
    </>
  );
}

// 相机随当前视图的三维包围盒自适应取景
function FrameField({ cols, rows, yExtent, controls }: {
  cols: number; rows: number; yExtent: number;
  controls: React.MutableRefObject<{ target: THREE.Vector3; update: () => void } | null>;
}) {
  const { camera, size } = useThree();
  const init = useRef(false);
  const settling = useRef(true);
  const span = useMemo(() => Math.max(cols, rows, yExtent), [cols, rows, yExtent]);
  const worldH = useMemo(() => span * 1.18 + 2.4, [span]);
  useEffect(() => { settling.current = true; }, [cols, rows, yExtent]);
  useEffect(() => {
    if (init.current || size.height < 10) return; init.current = true;
    camera.position.set(1, 0.9, 1).normalize().multiplyScalar(span * 1.4 + 8);
    camera.up.set(0, 1, 0); camera.updateProjectionMatrix();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [size.height]);
  useFrame(() => {
    if (!settling.current || size.height < 10) return;
    const oc = camera as THREE.OrthographicCamera;
    if (controls.current) { controls.current.target.lerp(new THREE.Vector3(0, 0, 0), 0.15); controls.current.update(); }
    const want = size.height / worldH;
    if (oc.isOrthographicCamera) { oc.zoom += (want - oc.zoom) * 0.15; oc.updateProjectionMatrix(); if (Math.abs(oc.zoom - want) < 0.04) settling.current = false; }
  });
  return null;
}

// ── shared button language (matches ClusterView/StatusView) ──
const SECONDARY: React.CSSProperties = { border: '1px solid var(--button-secondary-border)', background: 'var(--button-secondary-bg)', color: 'var(--foreground-muted)' };
function navBtn(on: boolean): React.CSSProperties {
  return on ? { border: '1px solid var(--primary)', background: 'var(--primary)', color: 'var(--primary-foreground)', fontWeight: 600 } : { ...SECONDARY };
}
const btnBase: React.CSSProperties = { padding: '5px 12px', fontSize: 12, borderRadius: 8, cursor: 'pointer' };
const LBL: React.CSSProperties = { fontSize: 11, fontWeight: 500, letterSpacing: 0.4, textTransform: 'uppercase', color: 'var(--tx3)', alignSelf: 'center' };

// ── 泳道（流动面·P2）：一个 step 内的有序算子，canvas 渲染，powered by PtoSwimlaneTaskPattern。
//    x = 关键路径累计时长；三轨 计算/通信/访存；overlapBg 的算子上方叠「背景搬运」带 = 掩盖。
const SW_LANE_KINDS: OpKind[] = ['compute', 'comm', 'mem'];
function Swimlane({ workload, step }: { workload: ParallelWorkload; step: number }) {
  const sched = OP_SCHEDULE[workload];
  const fl = useMemo(() => flowLayout(workload), [workload]);
  const mix = useMemo(() => phaseMix(workload), [workload]);
  const cursor = (step % 61) / 60;
  const LANE_H = 22, GAP = 4, BG_H = 15, TL_H = SW_LANE_KINDS.length * (LANE_H + GAP);
  const hasBg = fl.hidden.length > 0 || fl.bgBand;
  const topPad = hasBg ? BG_H + 3 : 0;
  const phLbl = workload === 'decode' ? 'Decode' : workload === 'prefill' ? 'Prefill' : '预训练';
  const canvasRefs = useRef<(HTMLCanvasElement | null)[]>([]);
  const tooltipRef = useRef<HTMLElement | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const drawLanes = useCallback(() => {
    const helper = window.PtoSwimlaneTaskPattern;
    if (!helper) return;
    SW_LANE_KINDS.forEach((kind, li) => {
      const canvas = canvasRefs.current[li];
      if (!canvas) return;
      const dpr = window.devicePixelRatio || 1;
      const W = canvas.offsetWidth;
      const H = canvas.offsetHeight;
      if (!W || !H) return;
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, W, H);
      fl.placed.filter(({ op }) => op.kind === kind).forEach(({ op, x, w }) => {
        const active = cursor >= x && cursor < x + w;
        const exposed = op.kind === 'comm';
        helper.drawTaskBar(ctx, {
          task: { label: op.name, laneKind: kind },
          x: x * W, y: 1,
          width: Math.max(1, w * W - 2),
          height: H - 2,
          baseColor: OP_COL[op.kind],
          radius: 4,
          isSelected: active,
        });
        if (exposed) {
          ctx.save();
          ctx.strokeStyle = '#b3003b';
          ctx.lineWidth = 2;
          ctx.beginPath();
          if ((ctx as any).roundRect) (ctx as any).roundRect(x * W, 1, Math.max(1, w * W - 2), H - 2, 4);
          else ctx.rect(x * W, 1, Math.max(1, w * W - 2), H - 2);
          ctx.stroke();
          ctx.restore();
        }
      });
    });
  }, [fl, cursor]);

  useEffect(() => { drawLanes(); }, [drawLanes]);

  useEffect(() => {
    const helper = window.PtoSwimlaneTaskPattern;
    if (!helper) return;
    const tip = helper.createTooltip();
    tooltipRef.current = tip;
    const container = containerRef.current;
    if (container) container.appendChild(tip);
    const handlers: Array<{ canvas: HTMLCanvasElement; move: EventListener; leave: EventListener }> = [];
    SW_LANE_KINDS.forEach((kind, li) => {
      const canvas = canvasRefs.current[li];
      if (!canvas) return;
      const opsForLane = fl.placed.filter(({ op }) => op.kind === kind);
      const move = (e: Event) => {
        const me = e as MouseEvent;
        const rect = canvas.getBoundingClientRect();
        const xFrac = (me.clientX - rect.left) / rect.width;
        const found = opsForLane.find(({ x, w }) => xFrac >= x && xFrac < x + w);
        if (!found) { helper.hideTooltip(tip); return; }
        const { op } = found;
        const exposed = op.kind === 'comm';
        helper.showTooltip(tip, { label: op.name, laneKind: OP_KIND_LBL[kind], note: (op as any).note + (exposed ? '（暴露：占墙钟）' : ''), status: exposed ? 'warn' : undefined }, me, { bounds: container });
      };
      const leave = () => helper.hideTooltip(tip);
      canvas.addEventListener('pointermove', move);
      canvas.addEventListener('pointerleave', leave);
      handlers.push({ canvas, move, leave });
    });
    return () => {
      handlers.forEach(({ canvas, move, leave }) => {
        canvas.removeEventListener('pointermove', move);
        canvas.removeEventListener('pointerleave', leave);
      });
      tip.remove();
      tooltipRef.current = null;
    };
  }, [fl]);

  return (
    <div style={{ padding: '8px 12px 10px', display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 12.5, fontWeight: 600 }}>算子泳道 · {phLbl} <span style={{ fontSize: 10.5, color: 'var(--tx3)', fontWeight: 400 }}>一层内 · {sched.bound === 'memory' ? '访存受限' : '计算受限'}</span></span>
        <span style={{ fontSize: 10.5, fontFamily: MONO, color: 'var(--tx2)' }}>
          计算 {Math.round(mix.compute * 100)}% · 通信 {Math.round(mix.comm * 100)}% · 访存 {Math.round(mix.mem * 100)}%
        </span>
        {/* 监控靶子：暴露通信(浪费墙钟) vs 掩盖(藏在计算下) */}
        <span style={{ fontSize: 10.5, fontFamily: MONO, display: 'inline-flex', gap: 10 }}>
          <span style={{ color: '#ff4b7b' }}>暴露通信 {Math.round(fl.exposedComm * 100)}%</span>
          <span style={{ color: 'var(--tx3)' }}>掩盖 {Math.round(fl.hiddenFrac * 100)}%</span>
        </span>
        <span style={{ fontSize: 9.5, color: 'var(--tx3)', marginLeft: 'auto' }}>{sched.src}</span>
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        {/* 左侧轨道名 */}
        <div style={{ width: 44, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: GAP, paddingTop: topPad }}>
          {SW_LANE_KINDS.map((k) => (
            <div key={k} style={{ height: LANE_H, display: 'flex', alignItems: 'center', fontSize: 10, color: OP_COL[k], fontWeight: 600 }}>{OP_KIND_LBL[k]}</div>
          ))}
        </div>
        {/* 时间轴（关键路径 = wall-clock；掩盖带叠在计算之上） */}
        <div ref={containerRef} style={{ position: 'relative', flex: 1, height: TL_H + topPad }}>
          {/* 掩盖带：被计算盖住的通信/访存 + 背景搬运 */}
          {fl.bgBand && (
            <div title={fl.bgBand.note} style={{ position: 'absolute', left: `${fl.bgBand.x * 100}%`, width: `${fl.bgBand.w * 100}%`, top: 0, height: BG_H, borderRadius: 3, background: `${OP_COL.mem}44`, border: `1px dashed ${OP_COL.mem}`, fontSize: 8, color: 'var(--tx2)', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden', whiteSpace: 'nowrap' }}>
              {fl.bgBand.name} · 掩盖
            </div>
          )}
          {fl.hidden.map((h) => (
            <div key={h.op.id} title={`${h.op.name} · ${(h.op as any).note}（被计算掩盖，不占墙钟）`} style={{ position: 'absolute', left: `${h.x * 100}%`, width: `${h.w * 100}%`, top: 0, height: BG_H, borderRadius: 3, background: `${OP_COL[h.op.kind]}44`, border: `1px dashed ${OP_COL[h.op.kind]}`, fontSize: 8, color: 'var(--tx2)', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden', whiteSpace: 'nowrap' }}>
              {h.op.name} · 掩盖
            </div>
          ))}
          {/* canvas lanes (轨道底 + 算子条) */}
          <div style={{ position: 'absolute', left: 0, right: 0, top: topPad, display: 'flex', flexDirection: 'column', gap: GAP }}>
            {SW_LANE_KINDS.map((kind, li) => (
              <canvas
                key={kind}
                ref={(el) => { canvasRefs.current[li] = el; }}
                style={{ display: 'block', width: '100%', height: LANE_H, background: 'var(--btn)', borderRadius: 4, cursor: 'default' }}
              />
            ))}
          </div>
          {/* 时间游标 */}
          <div style={{ position: 'absolute', left: `${cursor * 100}%`, top: 0, bottom: 0, width: 2, background: 'var(--tx)', opacity: 0.75 }} />
        </div>
      </div>
    </div>
  );
}

// ── 算子图（P3·算子整网）：一层内算子 DAG，Attention/MoE 分块 + 残差，标每个算子用哪种并行；
//   当前算子（游标）高亮 → 回答「这张卡此刻在算哪个算子」。结构面看位置、流动面看时间、这里看结构。 ──
function opDim(id: string, coll?: string, dim?: string): string | null {
  if (dim) return dim.toUpperCase();   // 事件自带并行维标签（白皮书：通信事件携带 tp/pp/dp/ep 标签）
  if (coll === 'a2a') return 'EP'; if (coll === 'ring') return 'DP'; if (coll === 'p2p') return 'PP';
  if (/qkv|attn|oproj/i.test(id)) return 'TP'; if (/gmm|expert/i.test(id)) return 'EP·TP';
  if (/^fwd|^bwd/i.test(id)) return '全部'; return null;
}
function OperatorGraph({ workload, step }: { workload: ParallelWorkload; step: number }) {
  const ops = OP_SCHEDULE[workload].ops;
  const cur = opAtCursor(workload, (step % 61) / 60);
  const fine = ops.length > 5;   // 推理层有细算子 → 分 Attention/MoE 块；训练粗算子 → 平铺
  // 分块：Attention = 到 oproj 为止；MoE = router 起
  const moeStart = ops.findIndex((o) => /router/i.test(o.id));
  const attn = fine && moeStart > 0 ? ops.slice(0, moeStart) : ops;
  const moe = fine && moeStart > 0 ? ops.slice(moeStart) : [];
  const node = (o: typeof ops[number]) => {
    const on = o.id === cur.id, dim = opDim(o.id, o.coll, o.dim);
    return (
      <div key={o.id} title={o.note} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, flexShrink: 0 }}>
        <div style={{ padding: '5px 9px', borderRadius: 7, background: on ? OP_COL[o.kind] : `${OP_COL[o.kind]}22`, border: `1.5px solid ${OP_COL[o.kind]}`, color: on ? '#0b0f16' : 'var(--tx)', fontSize: 10.5, fontWeight: on ? 700 : 500, boxShadow: on ? '0 0 0 3px color-mix(in srgb, var(--tx) 30%, transparent)' : 'none', whiteSpace: 'nowrap', position: 'relative' }}>
          {on && <span style={{ position: 'absolute', top: -14, left: '50%', transform: 'translateX(-50%)', fontSize: 8, color: 'var(--tx)', fontWeight: 700 }}>▶ 此刻</span>}
          {o.name}
        </div>
        {dim && <span style={{ fontSize: 8, fontFamily: MONO, color: 'var(--tx3)' }}>{dim}</span>}
      </div>
    );
  };
  const arrow = <span style={{ color: 'var(--tx3)', fontSize: 12, flexShrink: 0 }}>→</span>;
  const block = (title: string, list: typeof ops, color: string) => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 3, border: `1px dashed ${color}66`, borderRadius: 9, padding: '14px 10px 8px', position: 'relative', flexShrink: 0 }}>
      <span style={{ position: 'absolute', top: -8, left: 10, fontSize: 9, fontWeight: 700, color, background: 'var(--panel-solid)', padding: '0 5px' }}>{title} <span style={{ color: 'var(--tx3)', fontWeight: 400 }}>+残差</span></span>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>{list.map((o, i) => <span key={o.id} style={{ display: 'contents' }}>{i > 0 && arrow}{node(o)}</span>)}</div>
    </div>
  );
  return (
    <div style={{ padding: '8px 12px 12px', display: 'flex', flexDirection: 'column', gap: 7 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 12.5, fontWeight: 600 }}>算子图 · {workload === 'decode' ? 'Decode' : workload === 'prefill' ? 'Prefill' : '预训练'} <span style={{ fontSize: 10.5, color: 'var(--tx3)', fontWeight: 400 }}>一层内 DAG · {fine ? 'Attention + MoE 块' : 'Forward / Backward'}</span></span>
        <span style={{ fontSize: 10.5, fontFamily: MONO, color: OP_COL[cur.kind] }}>此刻在算: {cur.name}（{OP_KIND_LBL[cur.kind]}）</span>
        <span style={{ fontSize: 9.5, color: 'var(--tx3)', marginLeft: 'auto' }}>节点色=计算/通信/访存 · 下标=用哪种并行</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, overflowX: 'auto', paddingTop: 8, paddingBottom: 2 }}>
        <div style={{ fontSize: 9.5, color: 'var(--tx3)', flexShrink: 0 }}>输入</div>{arrow}
        {fine && moe.length ? <>{block('Attention', attn, '#4369ef')}{arrow}{block('MoE 专家', moe, '#ff4b7b')}</>
          : <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>{ops.map((o, i) => <span key={o.id} style={{ display: 'contents' }}>{i > 0 && arrow}{node(o)}</span>)}</div>}
        {arrow}<div style={{ fontSize: 9.5, color: 'var(--tx3)', flexShrink: 0 }}>输出</div>
      </div>
    </div>
  );
}

// ── PP 流水甘特（每级一套泳道语义之 L4）：canvas 渲染，powered by PtoSwimlaneTaskPattern。
//   掉队：某 stage 算子耗时×2 → 延迟沿依赖链传播、bubble 变大（真实模拟）。
//   VPP：交错虚拟流水，理论把 bubble 降到 ~1/v（示意）。
const PIPE_MB = 8;
const PIPE_COL_F = '#22d3ee', PIPE_COL_B = '#6b8bff';
function PipelineGantt({ stages, step, straggler, onStraggler, vpp, onVpp }: {
  stages: number; step: number; straggler: number | null; onStraggler: (s: number | null) => void; vpp: number; onVpp: (v: number) => void;
}) {
  const pipe = useMemo(() => pipeline1F1B(stages, PIPE_MB, straggler), [stages, straggler]);
  const baseBubble = useMemo(() => pipeline1F1B(stages, PIPE_MB, null).bubblePct, [stages]);
  const cursor = (step % 61) / 60;
  const curSlot = Math.floor(cursor * pipe.slots);
  const LANE_H = 20, GAP = 4;
  const canvasRefs = useRef<(HTMLCanvasElement | null)[]>([]);
  const tooltipRef = useRef<HTMLElement | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const drawGantt = useCallback(() => {
    const helper = window.PtoSwimlaneTaskPattern;
    if (!helper) return;
    pipe.lanes.forEach((lane, s) => {
      const canvas = canvasRefs.current[s];
      if (!canvas) return;
      const dpr = window.devicePixelRatio || 1;
      const W = canvas.offsetWidth;
      const H = canvas.offsetHeight;
      if (!W || !H) return;
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, W, H);
      const slotW = W / pipe.slots;
      lane.forEach((c) => {
        const active = curSlot >= c.slot && curSlot < c.end;
        const base = c.dir === 'F' ? PIPE_COL_F : PIPE_COL_B;
        const barW = Math.max(1, (c.end - c.slot) * slotW - 1);
        if (vpp > 1 && c.mb % vpp !== 0) ctx.globalAlpha = 0.62;
        helper.drawTaskBar(ctx, {
          task: { label: barW > 14 ? String(c.mb) : '' },
          x: c.slot * slotW, y: 1,
          width: barW,
          height: H - 2,
          baseColor: base,
          radius: 2,
          isSelected: active,
        });
        ctx.globalAlpha = 1;
      });
    });
  }, [pipe, curSlot, vpp]);

  useEffect(() => { drawGantt(); }, [drawGantt]);

  useEffect(() => {
    const helper = window.PtoSwimlaneTaskPattern;
    if (!helper) return;
    const tip = helper.createTooltip();
    tooltipRef.current = tip;
    const container = containerRef.current;
    if (container) container.appendChild(tip);
    const handlers: Array<{ canvas: HTMLCanvasElement; move: EventListener; leave: EventListener }> = [];
    pipe.lanes.forEach((lane, s) => {
      const canvas = canvasRefs.current[s];
      if (!canvas) return;
      const slow = straggler === s;
      const { lo, hi } = stageLayerRange(pipe.stages, s);
      const move = (e: Event) => {
        const me = e as MouseEvent;
        const rect = canvas.getBoundingClientRect();
        const xFrac = (me.clientX - rect.left) / rect.width;
        const slot = Math.floor(xFrac * pipe.slots);
        const cell = lane.find((c) => slot >= c.slot && slot < c.end);
        if (!cell) { helper.hideTooltip(tip); return; }
        helper.showTooltip(tip, { label: `${cell.dir === 'F' ? '前向' : '后向'} · mb ${cell.mb}`, laneKind: `stage ${s} · L${lo}-${hi}${slow ? ' · 掉队(×2)' : ''}`, status: cell.dir === 'F' ? 'ok' : 'warn' }, me, { bounds: container });
      };
      const leave = () => helper.hideTooltip(tip);
      canvas.addEventListener('pointermove', move);
      canvas.addEventListener('pointerleave', leave);
      handlers.push({ canvas, move, leave });
    });
    return () => {
      handlers.forEach(({ canvas, move, leave }) => {
        canvas.removeEventListener('pointermove', move);
        canvas.removeEventListener('pointerleave', leave);
      });
      tip.remove();
      tooltipRef.current = null;
    };
  }, [pipe, straggler]);

  if (stages <= 1) {
    return <div style={{ padding: '12px 12px 14px', fontSize: 11.5, color: 'var(--tx2)', lineHeight: 1.6 }}>PP = 1 · 本工况无流水线（Decode/Prefill 单副本推理，PP 未切分）。切到 <b style={{ color: 'var(--tx)' }}>预训练</b> 工况看 1F1B 流水甘特、掉队与 VPP。</div>;
  }
  const vppBubble = baseBubble / vpp;
  const chip: React.CSSProperties = { ...btnBase, padding: '2px 8px', fontSize: 10.5 };
  return (
    <div style={{ padding: '8px 12px 10px', display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 12.5, fontWeight: 600 }}>PP 流水甘特 · 1F1B <span style={{ fontSize: 10.5, color: 'var(--tx3)', fontWeight: 400 }}>{stages} 级 × {PIPE_MB} mb</span></span>
        <span style={{ fontSize: 10.5, fontFamily: MONO }}><span style={{ color: PIPE_COL_F }}>F</span>·<span style={{ color: PIPE_COL_B }}>B</span>·<span style={{ color: 'var(--tx3)' }}>空档=bubble</span></span>
        <span style={{ fontSize: 10.5, fontFamily: MONO, color: pipe.bubblePct > 0.3 ? '#ff4b7b' : 'var(--tx2)' }}>bubble {Math.round(pipe.bubblePct * 100)}%{straggler != null ? `（掉队前 ${Math.round(baseBubble * 100)}%）` : ''}</span>
        {/* 掉队注入 */}
        <span style={{ ...LBL, marginLeft: 4 }}>掉队</span>
        <button onClick={() => onStraggler(null)} style={{ ...chip, ...navBtn(straggler === null) }}>无</button>
        {pipe.lanes.map((_, s) => <button key={s} onClick={() => onStraggler(s)} style={{ ...chip, ...(straggler === s ? { border: '1px solid #ff4b7b', background: '#ff4b7b', color: '#fff', fontWeight: 600 } : SECONDARY) }}>S{s}</button>)}
        {/* VPP 交错 */}
        <span style={{ ...LBL, marginLeft: 4 }}>VPP</span>
        {[1, 2, 4].map((v) => <button key={v} onClick={() => onVpp(v)} style={{ ...chip, ...navBtn(vpp === v) }}>×{v}</button>)}
        {vpp > 1 && <span style={{ fontSize: 10, fontFamily: MONO, color: '#04d793' }}>交错 → 理论 bubble ~{Math.round(vppBubble * 100)}%</span>}
        <span style={{ fontSize: 9.5, color: 'var(--tx3)', marginLeft: 'auto' }}>schedule-simulated · 对齐 PTO 1F1B</span>
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        {/* stage 泳道名带层段范围（白皮书：PP 是唯一适合「哪段层在哪个 stage」的维度，stage 应带 layers lo-hi） */}
        <div style={{ width: 86, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: GAP }}>
          {pipe.lanes.map((_, s) => {
            const { lo, hi } = stageLayerRange(stages, s);
            return <div key={s} style={{ height: LANE_H, display: 'flex', alignItems: 'center', fontSize: 9.5, color: straggler === s ? '#ff4b7b' : 'var(--tx2)', fontWeight: 600, whiteSpace: 'nowrap' }}>S{s} · L{lo}-{hi}</div>;
          })}
        </div>
        <div ref={containerRef} style={{ position: 'relative', flex: 1 }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: GAP }}>
            {pipe.lanes.map((_, s) => (
              <canvas
                key={s}
                ref={(el) => { canvasRefs.current[s] = el; }}
                style={{ display: 'block', width: '100%', height: LANE_H, background: straggler === s ? 'rgba(255,75,123,0.12)' : 'var(--btn)', borderRadius: 3, cursor: 'default' }}
              />
            ))}
          </div>
          {/* 时间游标 */}
          <div style={{ position: 'absolute', left: `${cursor * 100}%`, top: 0, bottom: 0, width: 2, background: 'var(--tx)', opacity: 0.7, pointerEvents: 'none' }} />
        </div>
      </div>
    </div>
  );
}

export function CubeView({
  gen, dark, sync, layout: layoutP, anom: anomP,
  stratColor: stratColorP, showComm, showAlert, aggLevel: aggLevelP,
  sel: selCtl, onSelectRank, onHoverRank, embedded = false,
}: {
  gen: Gen; dark: boolean; sync?: ViewSync;
  layout?: LayoutView;    // 堆叠方式（受控：由工作台顶部中间控制面板驱动，只读）
  anom?: AnomalyDim;       // 注入异常（受控）
  // ── 统一驾驶舱（CockpitApp）扩展 · 全部可选，不传时行为与升级前一致 ──
  stratColor?: PartitionDim;   // 策略互斥着色（none/tp/pp/dp/ep）——一次只回答一个「谁和谁一组」
  showComm?: boolean;          // 通信连线图层（P1 完整演化；P0 复用选中卡对端高亮）
  showAlert?: boolean;         // 热点/告警图层（P1）
  aggLevel?: LevelKey;         // 聚合粒度（来自层级轴）：L4 Pod/机柜 一柜一块 · L2 满卡粒度 …
  sel?: number | null;         // 受控选中 rank（驾驶舱左右联动共用选区）
  onSelectRank?: (rank: number | null) => void;
  onHoverRank?: (rank: number | null) => void;
  embedded?: boolean;          // 嵌入驾驶舱：隐藏自带右侧详情栏 + 底部流动面 dock（由驾驶舱右栏承载）
}) {
  const visualProfile = useContext(SceneVisualProfileContext);
  const surf = sceneSurface(dark, visualProfile);
  const N = GENERATIONS[gen].totalNpus;
  const stratColor: PartitionDim = sync?.stratColor ?? stratColorP ?? 'none';
  const aggLevel: LevelKey = sync?.aggLevel ?? aggLevelP ?? 'card';
  void showComm; void showAlert;   // P0：图层开关已接线到驾驶舱顶栏，连线渲染留待 P1

  // 堆叠方式 / 工况 / 注入异常 / 回放 均由工作台顶部中间控制面板驱动，本地仅作独立运行的兜底（只读）。
  const [viewL] = useState<LayoutView>('standard');
  const view = layoutP ?? viewL;
  const [workloadL] = useState<ParallelWorkload>('pretrain');
  const workload = sync?.workload ?? workloadL;
  const [anomL] = useState<AnomalyDim>('none');
  const anom = anomP ?? anomL;
  const [stepL, setStepL] = useState(0);
  const step = sync?.step ?? stepL;
  const setStep = sync?.setStep ?? setStepL;
  const [playingL] = useState(false);   // 回放由顶部控制面板驱动（sync.playing）；本地仅兜底
  const playing = sync?.playing ?? playingL;
  const [settling, setSettling] = useState(false);
  const [flowMode, setFlowMode] = useState<'ops' | 'pipe' | 'graph'>('ops');   // 流动面：层内算子序列 / PP 甘特 / 算子图
  const [straggler, setStraggler] = useState<number | null>(null);   // PP 甘特：掉队 stage
  const [vpp, setVpp] = useState(1);                                  // PP 甘特：VPP 交错度
  // 选中 rank：受控（驾驶舱）优先，否则本地 state（独立/ClusterView 用法）。
  const controlledSel = onSelectRank !== undefined;
  const [selL, setSelL] = useState<number | null>(null);
  const sel = controlledSel ? (selCtl ?? null) : selL;
  const setSel = (r: number | null) => { if (onSelectRank) onSelectRank(r); else setSelL(r); };
  const [hover, setHoverL] = useState<number | null>(null);
  const setHover = (r: number | null) => { setHoverL(r); onHoverRank?.(r); };
  useEffect(() => { if (!controlledSel) setSelL(null); setHoverL(null); }, [gen, controlledSel]);  // 代际换 → N 变，清选区

  const controlsRef = useRef<{ target: THREE.Vector3; update: () => void } | null>(null);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setStep((s) => (s + 1) % 61), 650);
    return () => clearInterval(id);
  }, [playing, setStep]);

  const lay = useMemo(() => layoutOf(view, workload, N), [view, workload, N]);
  const pm = useMemo(() => parallelMap(workload, N), [workload, N]);
  const dep = useMemo(() => deploymentOf(workload, N), [workload, N]);   // 部署查询（正查：这张卡担任什么）
  const kind = workload === 'decode' ? 'comm' : 'compute';

  // 负载场：
  //  · 无异常 → 真实负载场（cardLoad01，真实热力）。
  //  · 注入异常（显式合成教学开关）→ 非异常卡给平静绿底、异常组给红热点，让「形状」在
  //    物理(散点)与对应维度视图(成块)里都清晰爆出——不受工况 base 负载高低影响。
  const loadOf = useMemo(() => {
    return (k: number): number => {
      if (anom !== 'none') return pm.groupOf(k, anom as ParDim) === 0 ? 0.93 : 0.2 + ((k % 7) / 7) * 0.12;
      return cardLoad01(k, kind, step, 0, N);
    };
  }, [kind, step, N, anom, pm]);

  // ── 聚合粒度（层级轴驱动）：把逐 rank 布局折成「一柜一块」等宏观单元；card/die/core = 满粒度 ──
  const agg = useMemo(() => aggregateOf(aggLevel, lay.pos, N), [aggLevel, lay, N]);
  const aggregated = agg.size > 1;
  // ── 单元着色：策略互斥着色（PARTITION 调色板）优先；否则状态红黄绿（聚合时取成员均值）──
  const stratRGB = useMemo(() => PARTITION_PALETTE.map(hexRGB), []);
  const colorOf = useMemo(() => {
    return (u: number): [number, number, number] => {
      if (stratColor !== 'none') {
        const g = pm.groupOf(agg.rankOfUnit(u), stratColor as ParDim);
        return stratRGB[g % stratRGB.length];
      }
      if (agg.size === 1) return loadRGB(loadOf(u));
      const { start, end } = agg.membersOfUnit(u);
      let s = 0; for (let k = start; k < end; k++) s += loadOf(k);
      return loadRGB(s / Math.max(1, end - start));
    };
  }, [stratColor, pm, agg, stratRGB, loadOf]);
  // 聚合方块尺寸：填满稀疏格的 ~88%（留细缝），下限比单卡明显更大 → 读作「宏观一块」。
  const boxXZ = aggregated ? Math.min(0.88 * AGG_SPREAD * PITCH, Math.max(BOX * 1.8, PITCH * Math.sqrt(agg.size) * 0.28)) : BOX;
  const boxY = aggregated ? 0.42 : 0.16;

  const recolorKey = useMemo(() =>
    step * 1000 + LAYOUT_VIEWS.indexOf(view) * 7
    + ({ none: 0, tp: 1, pp: 2, dp: 3, ep: 4 } as Record<AnomalyDim, number>)[anom]
    + ({ none: 0, tp: 100, pp: 200, dp: 300, ep: 400 } as Record<PartitionDim, number>)[stratColor]
    + agg.size * 0.0001,
  [step, view, anom, stratColor, agg.size]);

  // ── 流动面 → 结构面：游标扫到的当前算子（与泳道共用 opAtCursor）→ 通信则高亮对端。
  //    当前算子若是计算、但其下有并发的「掩盖」通信（如 Backward 下的 DP AllReduce），也触发对端（标注掩盖）。 ──
  const cursor01 = (step % 61) / 60;
  const fl = useMemo(() => flowLayout(workload), [workload]);
  const curOp = useMemo(() => opAtCursor(workload, cursor01), [workload, cursor01]);
  const activeComm = useMemo(() => {
    if (curOp.kind === 'comm') return { op: curOp, hidden: false };
    const h = fl.hidden.find((hh) => hh.op.kind === 'comm' && cursor01 >= hh.x && cursor01 < hh.x + hh.w);
    return h ? { op: h.op, hidden: true } : null;
  }, [curOp, fl, cursor01]);
  const curDim: Exclude<ParDim, 'sp'> | null = activeComm
    ? (activeComm.op.dim ?? (activeComm.op.coll === 'ring' ? 'dp' : activeComm.op.coll === 'p2p' ? 'pp' : 'ep')) : null;
  // 多维叠加（白皮书「一个 rank 的多维身份」）：选中一张卡 → 它在 TP/PP/DP/EP 四个维度的通信组
  // 同时显示（同一 rank 同时具有多个并行坐标，不是互斥模式）；游标扫到某维通信算子时该维加亮。
  const peerSets = useMemo(() => {
    if (sel == null || aggregated) return [];
    return (['tp', 'pp', 'dp', 'ep'] as const).map((d) => ({
      dim: d, ranks: dep.peersOf(sel, d, PEER_MAX), color: PARALLEL_COLORS[d], active: d === curDim,
    }));
  }, [sel, aggregated, dep, curDim]);
  const activePeers = curDim ? (peerSets.find((s) => s.dim === curDim)?.ranks ?? []) : [];
  const peerColor = curDim ? PARALLEL_COLORS[curDim] : '#4369ef';
  const dimLabel: Record<string, string> = { tp: '张量并行 AllGather/ReduceScatter', ep: '专家 All-to-All', dp: '数据并行 AllReduce', pp: '流水 P2P' };

  const shell: React.CSSProperties = { position: 'absolute', inset: 0, zIndex: 11, display: 'flex', flexDirection: 'column', background: 'var(--bg)', color: 'var(--tx)' };
  const card: React.CSSProperties = { background: 'var(--panel)', border: '1px solid var(--bd)', borderRadius: 11, boxShadow: 'var(--shadow-sm)', backdropFilter: 'blur(12px)', WebkitBackdropFilter: 'blur(12px)' };

  return (
    <div style={shell}>
      {/* 堆叠方式 / 工况 / 注入异常 / 回放 已上移到工作台顶部中间控制面板（Decode ▾ pill）。此处仅留画布 + 流动面。 */}
      {/* ── 3D 立方阵 ── */}
      <div style={{ flex: 1, position: 'relative', minHeight: 0 }}>
        <Canvas
          orthographic dpr={[1, 2]}
          camera={{ position: [40, 34, 40], zoom: 12, near: 0.1, far: 4000 }}
          gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.1, powerPreference: 'high-performance' }}
          onCreated={({ gl }) => { gl.domElement.addEventListener('webglcontextlost', (e) => e.preventDefault(), false); }}
          onPointerMissed={() => setSel(null)}
        >
          <color attach="background" args={[surf.background]} />
          <hemisphereLight intensity={surf.ambient} groundColor={dark ? '#10131a' : '#e8edf4'} />
          <directionalLight position={[8, 14, 6]} intensity={surf.key} />
          <directionalLight position={[-8, 8, -10]} intensity={surf.fill} />
          <FrameField cols={aggregated ? agg.cols : lay.cols} rows={aggregated ? agg.rows : lay.rows} yExtent={aggregated ? agg.yExtent : lay.yExtent} controls={controlsRef} />
          <CubeField cells={agg.cells} colorOf={colorOf} recolorKey={recolorKey} onSettleChange={setSettling} boxXZ={boxXZ} boxY={boxY}
            selected={sel != null ? agg.unitOfRank(sel) : null} hover={hover != null ? agg.unitOfRank(hover) : null}
            onPick={(u) => setSel(u == null ? null : agg.rankOfUnit(u))} onHover={(u) => setHover(u == null ? null : agg.rankOfUnit(u))}
            peerSets={peerSets} />
          <OrbitControls
            ref={controlsRef as never} makeDefault enableDamping dampingFactor={0.08}
            minPolarAngle={0} maxPolarAngle={Math.PI / 2} minDistance={2} maxDistance={600}
            mouseButtons={{ LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.PAN, RIGHT: THREE.MOUSE.PAN }}
          />
          <GizmoHelper alignment="bottom-left" margin={[64, 80]}>
            <GizmoViewcube faces={['Right', 'Left', 'Top', 'Bottom', 'Front', 'Back']}
              color={dark ? '#2a2e36' : '#eef1f6'} hoverColor="#4369ef"
              textColor={dark ? '#e6e6e6' : '#1c2433'} strokeColor={dark ? '#4a5160' : '#aab4c4'} opacity={0.95} />
          </GizmoHelper>
        </Canvas>

        {/* 当前算子横幅（流动面游标 → 结构面）：显示游标此刻在算什么；通信算子+选中卡 → 高亮对端 */}
        {(
          <div style={{ position: 'absolute', left: '50%', top: 12, transform: 'translateX(-50%)', ...card, padding: '7px 14px', pointerEvents: 'none', display: 'flex', alignItems: 'center', gap: 9 }}>
            <span style={{ width: 9, height: 9, borderRadius: 2, background: OP_COL[curOp.kind] }} />
            <span style={{ fontSize: 12, fontWeight: 600 }}>{curOp.name}</span>
            <span style={{ fontSize: 10.5, color: 'var(--tx3)' }}>{OP_KIND_LBL[curOp.kind]}</span>
            {curDim && sel != null && activePeers.length > 0 && (
              <span style={{ fontSize: 10.5, color: peerColor, borderLeft: '1px solid var(--bd)', paddingLeft: 9 }}>
                rank {sel} · 与 {activePeers.length} 张卡做 {dimLabel[curDim]}
                <span style={{ color: 'var(--tx3)', marginLeft: 6 }}>{activeComm?.hidden ? '（掩盖·并发）' : '（暴露）'}</span>
              </span>
            )}
            {!curDim && sel != null && peerSets.length > 0 && (
              <span style={{ fontSize: 10.5, color: 'var(--tx3)', borderLeft: '1px solid var(--bd)', paddingLeft: 9 }}>
                rank {sel} · TP/PP/DP/EP 四维通信组同时淡显 · 游标扫到通信算子时对应维加亮
              </span>
            )}
          </div>
        )}

        {/* 视图说明 + 状态图例 */}
        <div style={{ position: 'absolute', left: 12, top: 12, ...card, padding: '9px 12px', maxWidth: 340, pointerEvents: 'none' }}>
          <div style={{ fontSize: 12.5, fontWeight: 600, marginBottom: 3 }}>{LAYOUT_LABEL[view]}<span style={{ color: 'var(--tx3)', fontWeight: 400, fontFamily: MONO, marginLeft: 8 }}>{settling ? '重排中…' : ''}</span></div>
          <div style={{ fontSize: 10.5, color: 'var(--tx2)', lineHeight: 1.5 }}>{lay.note}</div>
          {/* 多维叠加常驻图例：几种并行是同一批 rank 上叠加的不同维度坐标（同时存在、不互斥）；
              切视图只是换一根投影轴，rank 的多维坐标本身不变（白皮书「多维运行时放置」） */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 5, flexWrap: 'wrap', borderTop: '1px solid var(--bd)', paddingTop: 5 }}>
            {(['tp', 'pp', 'dp', 'ep'] as const).map((d) => (
              <span key={d} style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 9.5, fontFamily: MONO, color: 'var(--tx2)' }}>
                <span style={{ width: 8, height: 8, borderRadius: 2, background: PARALLEL_COLORS[d] }} />{d.toUpperCase()}×{pm.groupCount(d)}
              </span>
            ))}
            <span style={{ fontSize: 9, color: 'var(--tx3)', lineHeight: 1.4, width: '100%' }}>同一 rank 同时具有以上各维坐标（叠加，非互斥）· 切视图只换投影轴 · 点选一张卡看它的四维通信组</span>
          </div>
          {anom !== 'none' && (
            <div style={{ marginTop: 5, borderTop: '1px solid var(--bd)', paddingTop: 5 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
                <span style={{ fontSize: 9, fontWeight: 700, color: '#fff', background: ANOM_TYPE[anom].scatter === '散布维' ? '#ff4b7b' : PARALLEL_COLORS[anom as Exclude<ParDim, 'sp'>], borderRadius: 3, padding: '1px 6px' }}>{ANOM_TYPE[anom].scatter}</span>
                <span style={{ fontSize: 9.5, color: 'var(--tx3)' }}>{ANOM_TYPE[anom].tag}</span>
              </div>
              <div style={{ fontSize: 10.5, color: 'var(--tx2)', lineHeight: 1.5 }}>{ANOM_NOTE[anom]}</div>
            </div>
          )}
        </div>
        {/* 状态图例（策略着色时隐藏——此刻颜色通道给了策略而非状态，避免误读） */}
        {stratColor === 'none' && (
        <div style={{ position: 'absolute', right: (!embedded && sel != null) ? 288 : 12, bottom: 12, ...card, padding: '8px 11px', display: 'flex', gap: 10, pointerEvents: 'none' }}>
          {STATE_LABELS.map((lb, i) => (
            <span key={i} style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 9.5, color: 'var(--tx2)' }}><span style={{ width: 9, height: 9, borderRadius: 2, background: stateColor(i) }} />{lb}</span>
          ))}
        </div>
        )}

        {/* 部署查询详情栏（反查：点一张卡 → 它担任什么并行角色 + 物理位置）· 嵌入驾驶舱时由右栏承载 */}
        {!embedded && sel != null && (() => {
          const phys = dep.physOf(sel), u = loadOf(sel), st = loadState(u);
          const slice = dep.sliceOf(sel);
          // 白皮书口径：只有 PP 说「哪段层」；TP=同层 shard · DP=副本+样本 shard · EP=持有哪些 experts。
          const roleRow: Record<string, { label: string; val: (g: number, deg: number) => string; sub: (g: number, deg: number) => string }> = {
            tp: { label: 'TP 张量并行', val: (g, deg) => `shard ${g + 1}/${deg}`, sub: () => 'QKV/MLP 同层分片 · 协同计算，非整层归属' },
            pp: { label: 'PP 流水并行', val: (g, deg) => `stage ${g}/${deg}`, sub: () => `层段 L${slice.layerLo}-${slice.layerHi}${slice.hasEmbedding ? ' +Embedding' : ''}${slice.hasHead ? ' +LM Head' : ''}` },
            dp: { label: 'DP 数据并行', val: (g, deg) => `replica ${g}/${deg}`, sub: () => '同一完整模型副本 · 处理不同样本 shard' },
            ep: { label: 'EP 专家并行', val: () => `experts ${slice.expertLo}-${slice.expertHi}`, sub: (g, deg) => `专家分桶 ${g}/${deg} · A2A 域 ${slice.epDomain} · 桶↔卡非 1:1` },
          };
          const roles = dep.rolesOf(sel).filter((r) => r.dim !== 'sp');
          const row = (label: string, val: string) => (
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, margin: '3px 0' }}><span style={{ color: 'var(--tx2)' }}>{label}</span><span style={{ fontFamily: MONO, color: 'var(--tx)' }}>{val}</span></div>
          );
          return (
            <div style={{ position: 'absolute', right: 12, top: 12, bottom: 12, width: 264, ...card, padding: '13px 14px', overflowY: 'auto', pointerEvents: 'auto' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <span style={{ fontSize: 13, fontWeight: 700, fontFamily: MONO, color: '#4369ef' }}>rank {sel}</span>
                <button onClick={() => setSel(null)} style={{ ...btnBase, padding: '2px 8px', ...SECONDARY }}>✕</button>
              </div>
              <div style={{ fontSize: 9.5, letterSpacing: 0.4, textTransform: 'uppercase', color: 'var(--tx3)', margin: '2px 0 4px' }}>物理位置（部署在哪台机器）</div>
              {row('集群（L6 · 万卡）', `本集群 · ${N.toLocaleString()} 卡`)}
              {row('Pod（L4 · UBL128）', `Pod ${Math.floor(sel / (CAB_CARDS * 2))} / ${Math.ceil(N / (CAB_CARDS * 2))}`)}
              {row('机柜（物理分组）', `C${phys.cabinet}`)}
              {row('Host · 节点', `${phys.host}（柜内 ${phys.host % NODES_PER_CAB}）`)}
              {row('卡槽 slot', `${phys.slot} / 8`)}
              <div style={{ fontSize: 9.5, letterSpacing: 0.4, textTransform: 'uppercase', color: 'var(--tx3)', margin: '11px 0 5px', borderTop: '1px solid var(--bd)', paddingTop: 9 }}>运行时放置 Runtime Placement · {pm.cfg}</div>
              {roles.map((r) => {
                const c = PARALLEL_COLORS[r.dim as Exclude<ParDim, 'sp'>];
                const rr = roleRow[r.dim];
                return (
                  <div key={r.dim} style={{ margin: '5px 0' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 7, fontSize: 11 }}>
                      <span style={{ width: 9, height: 9, borderRadius: 2, background: c, flexShrink: 0 }} />
                      <span style={{ color: 'var(--tx2)', flex: 1 }}>{rr.label}</span>
                      <span style={{ fontFamily: MONO, color: 'var(--tx)' }}>{rr.val(r.group, r.degree)}</span>
                    </div>
                    <div style={{ fontSize: 9, color: 'var(--tx3)', margin: '1px 0 0 16px', lineHeight: 1.4 }}>{rr.sub(r.group, r.degree)}</div>
                  </div>
                );
              })}
              {/* CP/SP：白皮书要求按配置展示（本工况 CP1 未切分，与 TP 同域），而非省略 */}
              <div style={{ margin: '5px 0' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 7, fontSize: 11 }}>
                  <span style={{ width: 9, height: 9, borderRadius: 2, background: '#22d3ee', flexShrink: 0 }} />
                  <span style={{ color: 'var(--tx2)', flex: 1 }}>CP/SP 序列并行</span>
                  <span style={{ fontFamily: MONO, color: 'var(--tx)' }}>CP1</span>
                </div>
                <div style={{ fontSize: 9, color: 'var(--tx3)', margin: '1px 0 0 16px', lineHeight: 1.4 }}>{slice.tokenNote}</div>
              </div>
              <div style={{ fontSize: 9.5, letterSpacing: 0.4, textTransform: 'uppercase', color: 'var(--tx3)', margin: '11px 0 5px', borderTop: '1px solid var(--bd)', paddingTop: 9 }}>当前状态</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 7, fontSize: 11 }}>
                <span style={{ width: 10, height: 10, borderRadius: 2, background: stateColor(st) }} />
                <span style={{ color: 'var(--tx)' }}>{STATE_LABELS[st]}</span>
                <span style={{ marginLeft: 'auto', fontFamily: MONO, color: 'var(--tx2)' }}>{Math.round(u * 100)}%</span>
              </div>
              <div style={{ fontSize: 9.5, letterSpacing: 0.4, textTransform: 'uppercase', color: 'var(--tx3)', margin: '11px 0 5px', borderTop: '1px solid var(--bd)', paddingTop: 9 }}>此刻在算（t={step}）</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 7, fontSize: 11 }}>
                <span style={{ width: 9, height: 9, borderRadius: 2, background: OP_COL[curOp.kind], flexShrink: 0 }} />
                <span style={{ color: 'var(--tx)', fontWeight: 600 }}>{curOp.name}</span>
                <span style={{ marginLeft: 'auto', fontSize: 10, color: 'var(--tx3)' }}>{OP_KIND_LBL[curOp.kind]}</span>
              </div>
              <div style={{ fontSize: 9.5, color: 'var(--tx3)', lineHeight: 1.5, marginTop: 9 }}>
                反查（rolesOf + opAtCursor）：这张卡此刻在算什么算子 + 它的并行角色。切「算子图」看结构、切「堆叠方式」看它与谁成组。
              </div>
            </div>
          );
        })()}
      </div>

      {/* ── 流动面 docked 在 3D 下方：结构面看位置，这里看时间。每级一套语义：层内算子 / PP 流水甘特 ──
          嵌入驾驶舱时隐藏（右栏②算子下钻承载「此刻在算什么」；驾驶舱主画布只留 3D）。 */}
      {!embedded && (
      <div style={{ flexShrink: 0, borderTop: '1px solid var(--bd)', background: 'var(--panel-solid)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 12px 0' }}>
          <span style={LBL}>流动面</span>
          {([['ops', '层内算子'], ['pipe', 'PP 流水甘特'], ['graph', '算子图']] as [typeof flowMode, string][]).map(([m, l]) => (
            <button key={m} onClick={() => setFlowMode(m)} style={{ ...btnBase, padding: '3px 10px', ...navBtn(flowMode === m) }}>{l}</button>
          ))}
          <span style={{ fontSize: 9.5, color: 'var(--tx3)', marginLeft: 6 }}>{flowMode === 'ops' ? '一层内算子时序 · 掩盖/暴露' : flowMode === 'pipe' ? 'L4 · PP stage × microbatch · 1F1B bubble' : '算子 DAG 结构 · 此刻在算哪个算子'}</span>
        </div>
        {flowMode === 'ops' ? <Swimlane workload={workload} step={step} />
          : flowMode === 'pipe' ? <PipelineGantt stages={pm.pp} step={step} straggler={straggler} onStraggler={setStraggler} vpp={vpp} onVpp={setVpp} />
          : <OperatorGraph workload={workload} step={step} />}
      </div>
      )}
    </div>
  );
}
