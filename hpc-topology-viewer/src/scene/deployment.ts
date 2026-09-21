/**
 * deployment — P0 地基（加法，不替换）。
 *
 * 在唯一真值源 parallelMap 之上，显式暴露「框架把模型摆到机器」这条链里被压缩掉的三件事：
 *   ① device mesh 形状（并行超立方体 / 魔方的边长）
 *   ② rank ↔ 物理坐标 的放置（placement）
 *   ③ rank ↔ 并行角色 的双向映射（正查：这张卡担任什么；反查：某并行组里有哪些卡）
 *
 * 一切派生自 parallelMap，绝不造第二套真值。本模块不改动任何现有行为，
 * 也暂不接入任何视图（接入是 P1）——纯粹为后续立方体重排 / 监控放置查询提供底座。
 */
import {
  NPUS_PER_NODE, NODES_PER_CAB, parallelMap, WORKLOAD,
  type ParallelWorkload, type ParallelConfig, type ParallelMapping, type ParDim,
} from './data';

// 物理坐标 —— 与 FullPodScene / ConsoleView 用的同一套平铺（8 卡/Host · 8 Host/机柜 · 单 Pod）。
export interface PhysCoord { pod: number; cabinet: number; host: number; slot: number; }
// 逻辑 mesh 坐标 —— 一个 rank 在并行超立方体里的点（魔方格点）。
export interface MeshCoord { pp: number; dp: number; ep: number; tp: number; sp: number; }
// mesh 维度的性质（诚实反映「不是严格立方体」，对应方案漏洞 #14）：
//   spatial   = 正交空间轴，三者乘积 == N（tp·pp·dp）——可给独立空间维度
//   collapsed = 塌缩轴，物理上折入某个空间轴（EP：训练折入 DP / 推理折入 TP 域）——不再乘进 N
//   virtual   = 虚轴，无独立体量（SP 与 TP 同域）——只能用嵌套/时间表达
export type MeshDimKind = 'spatial' | 'collapsed' | 'virtual';
export interface MeshDim { dim: ParDim; size: number; kind: MeshDimKind; note?: string; }
export interface ParallelRole { dim: ParDim; group: number; degree: number; }

// 单卡任务投影（白皮书「多维叠加」§：整网架构经 placement 投影到某个 rank 后的计算切片）。
// 语义边界（白皮书核心结论）：只有 PP 适合说「哪段层」；TP 是同层 shard、DP 是副本+样本 shard、
// EP 是「持有哪些 experts」+ A2A 域、CP/SP 是 token 段——各维口径不同，不能都压成「层归属」。
export interface RankSlice {
  layerLo: number; layerHi: number;          // PP：该 stage 承载的连续 decoder blocks（含端点）
  hasEmbedding: boolean; hasHead: boolean;   // 只有 stage 0 有 Embedding、末 stage 有 LM Head
  expertLo: number; expertHi: number;        // EP：该 rank 持有的专家分桶 experts lo-hi（含端点）
  epDomain: number;                          // EP：所属 A2A 域 id（训练=⌊replica/EP⌋ · 推理=node）
  tokenNote: string;                         // CP/SP：token 分片口径（CP1 → full context）
}

// PP stage → 连续层段（decoder blocks 均分，余数摊给前段；对齐白皮书「PP5 · layers 36-42」口径）。
export function stageLayerRange(stages: number, s: number, totalLayers = WORKLOAD.layers): { lo: number; hi: number } {
  const S = Math.max(1, stages), base = Math.floor(totalLayers / S), extra = totalLayers % S;
  const lo = s * base + Math.min(s, extra);
  const len = base + (s < extra ? 1 : 0);
  return { lo, hi: Math.max(lo, lo + len - 1) };
}

export interface Deployment {
  workload: ParallelWorkload;
  N: number;
  pm: ParallelMapping;          // 派生所依赖的 SSOT（不是第二套真值）
  mesh: MeshDim[];              // 有序的逻辑超立方体维度（魔方形状：外→内嵌套）
  fromIngest: boolean;          // true = 并行度来自外部摄入配置；false = 自动推导
  coordsOf: (rank: number) => MeshCoord;                              // rank → 逻辑 mesh 坐标
  physOf: (rank: number) => PhysCoord;                               // rank → 物理坐标
  rolesOf: (rank: number) => ParallelRole[];                         // 这张卡担任的并行角色（正查）
  sliceOf: (rank: number) => RankSlice;                              // 单卡任务投影（层段/专家桶/token 段）
  ranksInGroup: (dim: ParDim, group: number, cap?: number) => number[]; // 某并行组里有哪些卡（反查）
  peersOf: (rank: number, dim: ParDim, cap?: number) => number[];    // 通信对端（透传 pm）
}

// 一个 rank 会同时担任的所有并行角色（sp 与 tp 同域，单列以便展示）。
const ROLE_DIMS: ParDim[] = ['tp', 'sp', 'pp', 'dp', 'ep'];

export function deploymentOf(workload: ParallelWorkload, N: number, cfg?: ParallelConfig): Deployment {
  const pm = parallelMap(workload, N, cfg);
  const HOST = NPUS_PER_NODE, CAB = NODES_PER_CAB;

  const coordsOf = (rank: number): MeshCoord => ({
    pp: pm.groupOf(rank, 'pp'),
    dp: pm.groupOf(rank, 'dp'),
    ep: pm.groupOf(rank, 'ep'),
    tp: pm.groupOf(rank, 'tp'),
    sp: pm.groupOf(rank, 'sp'),
  });

  // 单 Pod 平铺：rank → host/slot → cabinet（与现有 3D/2D 视图一致，Pod 恒为 0）。
  const physOf = (rank: number): PhysCoord => {
    const host = Math.floor(rank / HOST);
    return { pod: 0, cabinet: Math.floor(host / CAB), host, slot: rank % HOST };
  };

  const rolesOf = (rank: number): ParallelRole[] =>
    ROLE_DIMS.map((dim) => ({ dim, group: pm.groupOf(rank, dim), degree: pm.groupCount(dim) }));

  // 单卡任务投影：整网架构（48 blocks · 64 routed experts）按 placement 投到该 rank。
  const sliceOf = (rank: number): RankSlice => {
    const c = coordsOf(rank);
    const PP = pm.groupCount('pp'), EPd = pm.groupCount('ep');
    const { lo, hi } = stageLayerRange(PP, c.pp);
    const perBucket = Math.max(1, Math.floor(WORKLOAD.routedExperts / EPd));
    const epDomain = pm.epScope === 'replica' ? Math.floor(c.dp / Math.max(1, EPd)) : Math.floor(rank / pm.tp);
    return {
      layerLo: lo, layerHi: hi,
      hasEmbedding: c.pp === 0, hasHead: c.pp === PP - 1,
      expertLo: c.ep * perBucket, expertHi: Math.min(WORKLOAD.routedExperts, (c.ep + 1) * perBucket) - 1,
      epDomain,
      tokenNote: pm.sp > 1 ? `token 段 ${c.sp + 1}/${pm.sp}` : 'CP1 · full context（未切分，与 TP 同域）',
    };
  };

  // 反查：扫描 [0,N) 收集 groupOf 命中该组的卡（cap 上限保底）。查询非每帧调用，O(N) 可接受。
  const ranksInGroup = (dim: ParDim, group: number, cap = 1 << 16): number[] => {
    const out: number[] = [];
    for (let k = 0; k < N && out.length < cap; k++) if (pm.groupOf(k, dim) === group) out.push(k);
    return out;
  };

  // mesh 形状（外→内嵌套）。空间轴 tp·pp·dp 乘积 == N；EP 为塌缩轴（不乘进 N），SP 为虚轴。
  const epNote = pm.epScope === 'replica'
    ? 'EP 折入 DP 轴（训练：相邻副本 A2A）'
    : 'EP 折入 TP 域（推理：节点内路由）';
  const mesh: MeshDim[] = [
    { dim: 'pp', size: pm.groupCount('pp'), kind: 'spatial' },
    { dim: 'dp', size: pm.groupCount('dp'), kind: 'spatial' },
    { dim: 'tp', size: pm.groupCount('tp'), kind: 'spatial' },
    { dim: 'ep', size: pm.groupCount('ep'), kind: 'collapsed', note: epNote },
    { dim: 'sp', size: pm.groupCount('sp'), kind: 'virtual', note: '与 TP 同域（CP1 未独立切分）' },
  ];

  return {
    workload, N, pm, mesh, fromIngest: !!cfg,
    coordsOf, physOf, rolesOf, sliceOf, ranksInGroup,
    peersOf: (rank, dim, cap) => pm.peersOf(rank, dim, cap),
  };
}
