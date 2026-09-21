/**
 * layout — 立方体重排布局引擎（P1 · 5 种 3D 堆法，纯函数，不碰渲染）。
 *
 * 「切视图 = 换一种堆法」：同一批卡（rank），按不同并行轴重新排布到三维空间。
 * 5 种视图来自原型 HTML（chipCubeM v22），坐标直接以 THREE.js world unit 返回：
 *   standard    — 标准 3D 立方，X=TP · Y=PP · Z=DP，基准对照视图
 *   dp-tile     — DP 平铺，64 个副本排 8×8 宫格，每格内展示 TP×PP 竖板
 *   ep-cluster  — EP 专家桶墙，同桶成墙 · 墙内 = 各 A2A 域网格（真值派生自 parallelMap）
 *   tp-slice    — TP 切片，8 张权重墙沿 X 展开
 *   pp-pipeline — PP 流水，16 段竖板沿 X（左→右流水）
 *
 * pos[rank] 直接是 Three.js 世界坐标（单位 = THREE.js unit），CubeField 不再乘 PITCH。
 * cols/rows 是 XZ 平面的世界空间跨度（用于相机取景）；yExtent 是 Y 轴跨度。
 */
import { deploymentOf } from './deployment';
import { type ParallelWorkload, type ParallelConfig } from './data';

export type LayoutView = 'standard' | 'dp-tile' | 'ep-cluster' | 'tp-slice' | 'pp-pipeline';
export const LAYOUT_VIEWS: LayoutView[] = ['standard', 'dp-tile', 'ep-cluster', 'tp-slice', 'pp-pipeline'];
export const LAYOUT_LABEL: Record<LayoutView, string> = {
  standard:      '标准',
  'dp-tile':     'DP 平铺',
  'ep-cluster':  'EP 聚簇',
  'tp-slice':    'TP 切片',
  'pp-pipeline': 'PP 流水',
};

/** 每个 rank 的三维世界坐标（THREE.js unit，原点居中）。 */
export interface Cell { x: number; y: number; z: number; }

export interface LayoutResult {
  view: LayoutView;
  cols: number;    // XZ 平面 X 轴跨度（world unit），供相机取景
  rows: number;    // XZ 平面 Z 轴跨度（world unit），供相机取景
  yExtent: number; // Y 轴跨度（world unit），供相机取景
  pos: Cell[];     // pos[rank] — 每个 rank 的三维世界坐标
  note: string;    // 该视图的一句话说明
}

/**
 * layoutOf — 给定视图，返回每个 rank 的三维世界坐标（pos）和视图元信息。
 * 坐标 = THREE.js world unit；CubeField 直接使用，不再乘 PITCH。
 */
export function layoutOf(view: LayoutView, workload: ParallelWorkload, N: number, cfg?: ParallelConfig): LayoutResult {
  const d = deploymentOf(workload, N, cfg);
  const TP = d.pm.groupCount('tp'), PP = d.pm.groupCount('pp'), DP = d.pm.groupCount('dp');
  const pos: Cell[] = new Array(N);

  const tpC = (TP - 1) / 2, ppC = (PP - 1) / 2, dpC = (DP - 1) / 2;
  let cols = 1, rows = 1, yExtent = 0, note = '';

  // DP 折叠：DP 很大时（如 Decode/Prefill：PP=1、DP=1024），把 DP 摆在单轴会退化成一条极长的细线。
  //   → DP>128 时折成近方格铺在平面，任何工况都渲染成紧凑的立体块而非细线；平衡工况（预训练 DP=64）保持经典三轴立方。
  const foldDP = DP > 128;
  const dgx = Math.max(1, Math.ceil(Math.sqrt(DP)));   // DP 折叠网格列数
  const dgz = Math.ceil(DP / dgx);                      // DP 折叠网格行数
  const dpGX = (dp: number) => (dp % dgx) - (dgx - 1) / 2;         // 居中列偏移
  const dpGZ = (dp: number) => Math.floor(dp / dgx) - (dgz - 1) / 2; // 居中行偏移

  if (view === 'standard') {
    if (!foldDP) {
      // 平衡工况（预训练 TP8×PP16×DP64）：经典 3D 立方 X=TP · Y=PP(倒序,低 stage 在顶) · Z=DP
      for (let k = 0; k < N; k++) {
        const c = d.coordsOf(k);
        pos[k] = { x: (c.tp - tpC) * 1.6, y: (ppC - c.pp) * 1.15, z: (c.dp - dpC) * 0.75 };
      }
      cols = (TP - 1) * 1.6 + 1;
      rows = (DP - 1) * 0.75 + 1;
      yExtent = (PP - 1) * 1.15;
      note = `标准立方：X=TP×${TP} · Y=PP×${PP}（模型深度，低 stage 在顶）· Z=DP×${DP}`;
    } else {
      // 大 DP（Decode/Prefill）：DP 折成 dgx×dgz 铺地；(PP,TP) 併成高度 → 紧凑立体块
      const hCount = TP * PP, hC = (hCount - 1) / 2;
      for (let k = 0; k < N; k++) {
        const c = d.coordsOf(k);
        pos[k] = { x: dpGX(c.dp) * 1.4, y: (hC - (c.pp * TP + c.tp)) * 1.2, z: dpGZ(c.dp) * 1.4 };
      }
      cols = (dgx - 1) * 1.4 + 1;
      rows = (dgz - 1) * 1.4 + 1;
      yExtent = (hCount - 1) * 1.2;
      note = `标准立方：DP×${DP} 折成 ${dgx}×${dgz} 铺地（避免退化成细线）· 高度=TP×${TP}${PP > 1 ? `×PP×${PP}` : ''}`;
    }

  } else if (view === 'dp-tile') {
    // DP 平铺：DP 个副本排近方阵宫格，每格内 TP（横） × PP（竖，低 stage 在顶）竖板
    const DGX = Math.max(1, Math.ceil(Math.sqrt(DP)));
    const DGZ = Math.ceil(DP / DGX);
    const dpGapX = TP * 1.3 + 5;   // DP 宫格间距（X）
    const dpGapZ = PP * 0.9 + 2;   // DP 宫格间距（Z）
    for (let k = 0; k < N; k++) {
      const c = d.coordsOf(k);
      const gx = c.dp % DGX, gz = Math.floor(c.dp / DGX);
      pos[k] = {
        x: (gx - (DGX - 1) / 2) * dpGapX + (c.tp - tpC) * 1.3,
        y: (ppC - c.pp) * 0.9,
        z: (gz - (DGZ - 1) / 2) * dpGapZ,
      };
    }
    cols = (DGX - 1) * dpGapX + (TP - 1) * 1.3 + 1;
    rows = (DGZ - 1) * dpGapZ + 1;
    yExtent = (PP - 1) * 0.9;
    note = `DP 平铺：${DP} 个副本排 ${DGX}×${DGZ} 宫格 · 每格内 TP×PP 竖板 · 「整网=一份×${DP} 份」`;

  } else if (view === 'ep-cluster') {
    // EP 专家桶墙（真值派生，替换旧「专家 Host」第二套真值）：
    //   每个 rank 都持有一个专家分桶（experts 不专属少数主机、桶↔卡非 1:1）。
    //   X = 专家分桶（groupOf 'ep'，同墙 = 持有相同 experts）；墙内 = 该桶复现的全部 A2A 域
    //   （训练：⌊replica/EP⌋ · 推理：节点）折成 XZ 网格；Y = 域内成员（训练 PP×TP · 推理 TP/EP 槽）。
    //   于是「桶故障」snap 成一面墙，「A2A 域热点」= 每面墙同一格各取一员——与 parallelMap 的
    //   peersOf('ep') 完全一致（dispatch/combine 的对端 = 同域其余桶）。
    const EP = d.pm.groupCount('ep');
    const folded = d.pm.epScope === 'replica';
    const B = Math.max(1, folded ? Math.ceil(DP / EP) : Math.ceil(N / TP));   // A2A 域数
    const V = Math.max(1, folded ? PP * TP : Math.floor(TP / EP));            // 每域每桶成员数
    const bgx = Math.max(1, Math.ceil(Math.sqrt(B))), bgz = Math.ceil(B / bgx);
    const wallGap = bgx * 0.9 + 4;   // 墙间距 > 墙内网格宽，桶边界直观可见
    const epC = (EP - 1) / 2, vC = (V - 1) / 2;
    for (let k = 0; k < N; k++) {
      const c = d.coordsOf(k);
      const b = folded ? Math.floor(c.dp / EP) : Math.floor(k / TP);          // A2A 域 id
      const v = folded ? c.pp * TP + c.tp : (k % TP) % V;                     // 域内成员序
      pos[k] = {
        x: (c.ep - epC) * wallGap + ((b % bgx) - (bgx - 1) / 2) * 0.9,
        y: (vC - v) * 0.9,
        z: (Math.floor(b / bgx) - (bgz - 1) / 2) * 0.9,
      };
    }
    cols = (EP - 1) * wallGap + (bgx - 1) * 0.9 + 1;
    rows = (bgz - 1) * 0.9 + 1;
    yExtent = (V - 1) * 0.9;
    note = `EP 专家桶墙：${EP} 个分桶成墙（同墙=持有相同 experts）· 墙内 ${B} 个 A2A 域排 ${bgx}×${bgz} 网格 · ` +
      (folded ? `训练 EP 折入 DP 轴（相邻 ${EP} 副本 dispatch/combine）` : `推理 EP=节点内路由（域=节点）`) + ' · 桶↔卡非 1:1';

  } else if (view === 'tp-slice') {
    if (!foldDP) {
      // TP 切片：${TP} 张权重墙沿 X 展开；每墙 = DP×PP 平面（一个 TP 切片的完整模型）
      for (let k = 0; k < N; k++) {
        const c = d.coordsOf(k);
        pos[k] = { x: (c.tp - tpC) * 7, y: (ppC - c.pp) * 1.15, z: (c.dp - dpC) * 0.75 };
      }
      cols = (TP - 1) * 7 + 1;
      rows = (DP - 1) * 0.75 + 1;
      yExtent = (PP - 1) * 1.15;
      note = `TP 切片：${TP} 张权重墙沿 X 展开 · 每墙内 Y=PP×${PP} · Z=DP×${DP} · 张量分片一目了然`;
    } else {
      // 大 DP：每张 TP 墙 = DP 折叠面（DP 折成 dgz 行 × dgx 列铺在 Y-Z）；PP 併入 Z 分层
      const wallGapX = dgx * 0.9 + 5;   // 墙间距（含墙宽）
      const ppLayerZ = dgx * 0.9 + 2;   // PP>1 时的分层深度
      for (let k = 0; k < N; k++) {
        const c = d.coordsOf(k);
        pos[k] = {
          x: (c.tp - tpC) * wallGapX,
          y: -dpGZ(c.dp) * 1.0,
          z: dpGX(c.dp) * 0.9 + (c.pp - ppC) * ppLayerZ,
        };
      }
      cols = (TP - 1) * wallGapX + dgx * 0.9 + 1;
      rows = dgx * 0.9 + (PP - 1) * ppLayerZ + 1;
      yExtent = (dgz - 1) * 1.0;
      note = `TP 切片：${TP} 张权重墙沿 X 展开 · 每墙内 DP×${DP} 折成 ${dgz}×${dgx} 面 · 张量分片一目了然`;
    }

  } else {
    if (!foldDP) {
      // PP 流水（pp-pipeline）：PP 段竖板沿 X 展开（左→右流水方向）；TP 沿 Y · DP 沿 Z
      for (let k = 0; k < N; k++) {
        const c = d.coordsOf(k);
        pos[k] = { x: (c.pp - ppC) * 6.2, y: (c.tp - tpC) * 1.2, z: (c.dp - dpC) * 0.75 };
      }
      cols = (PP - 1) * 6.2 + 1;
      rows = (DP - 1) * 0.75 + 1;
      yExtent = (TP - 1) * 1.2;
      note = `PP 流水：${PP} 段竖板从左（stage 0）到右（stage ${PP - 1}）展开 · Y=TP×${TP} · Z=DP×${DP}`;
    } else {
      // 大 DP（Decode/Prefill：PP=1 单段）：每 PP 段 = DP 折叠板铺地 · TP 沿 Y
      const stageGapX = dgx * 1.4 + 6;
      for (let k = 0; k < N; k++) {
        const c = d.coordsOf(k);
        pos[k] = { x: (c.pp - ppC) * stageGapX + dpGX(c.dp) * 1.4, y: (c.tp - tpC) * 1.2, z: dpGZ(c.dp) * 1.4 };
      }
      cols = (PP - 1) * stageGapX + (dgx - 1) * 1.4 + 1;
      rows = (dgz - 1) * 1.4 + 1;
      yExtent = (TP - 1) * 1.2;
      note = `PP 流水：${PP} 段${PP > 1 ? '沿 X 展开' : '（本工况无流水）'} · 每段 DP×${DP} 折成 ${dgx}×${dgz} 铺地 · 高度=TP×${TP}`;
    }
  }

  return { view, cols, rows, yExtent, pos, note };
}

/**
 * verifyBijection — 校验布局是「同一批卡的置换」（每个三维位置最多一张卡）。
 * 供测试/自检使用；返回 { ok, reason }。
 */
export function verifyBijection(r: LayoutResult, N: number): { ok: boolean; reason: string } {
  if (r.pos.length !== N) return { ok: false, reason: `pos 数量 ${r.pos.length} ≠ N ${N}` };
  const seen = new Set<string>();
  for (let k = 0; k < N; k++) {
    const key = `${r.pos[k].x.toFixed(3)},${r.pos[k].y.toFixed(3)},${r.pos[k].z.toFixed(3)}`;
    if (seen.has(key)) return { ok: false, reason: `位置 (${key}) 被 rank ${k} 重叠` };
    seen.add(key);
  }
  return { ok: true, reason: `${N} 张卡各占唯一三维位置` };
}
