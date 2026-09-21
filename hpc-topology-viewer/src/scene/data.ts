// ─────────────────────────────────────────────────────────────────────────────
// Cluster model data layer — two generations (A5 / A6).
//
// Specs are drawn from public conference / vendor material (see SOURCES in
// ../content). In-cabinet / die / node layouts are schematic abstractions
// (vendor sheet-metal drawings are not public) and do not represent a real
// physical layout.
//
// All product/brand display text is sourced from ../content (stored base64 and
// decoded at runtime), so this source file carries no plaintext product names.
// ─────────────────────────────────────────────────────────────────────────────
import { TOK, INFO, SOURCES, CHANGES } from '../content';

export { INFO, SOURCES, CHANGES };

export type RackKind = 'compute' | 'switch';
export type ViewMode = 'overview' | 'rack' | 'node' | 'topology' | 'matrix' | 'mapping' | 'trace' | 'fullpod' | 'plane' | 'status' | 'console' | 'comm' | 'cube' | 'netcut';
export type Gen = 'A5' | 'A6';

// ─── Generation specs ────────────────────────────────────────────────────────
export interface GenSpec {
  code: Gen;
  name: string;             // pod form-factor display name
  npuLabel: string;         // accelerator display label
  npuShort: string;         // accelerator short label
  totalNpus: number;
  fp8EF: number;            // EFLOPS FP8
  fp4EF: number;            // EFLOPS FP4
  memTB: number;            // total HBM capacity
  memPerChipTBs: number;    // per-chip HBM bandwidth (TB/s)
  interconnectPBs: number;  // total UB interconnect bandwidth (PB/s)
  chipUbTBs: number;        // per-NPU UB bandwidth (TB/s)
  computeCabs: number;
  commCabs: number;
  totalCabs: number;
  footprintM2: number;
  hbm: string;              // self-developed HBM name
  release: string;
  trainTokps: string;
  inferTokps: string;
  superclusterNpu: string;  // cluster-level scale
  // per-chip specs (A5 from the published whitepaper; A6 estimated/derived)
  memGB: number;            // per-chip HBM capacity
  fp4Tflops: number | null; // per-chip MXFP4 TFLOPS
  fp8Tflops: number | null; // per-chip FP8-class TFLOPS
  l2MB: number | null;      // global L2 cache
  ubGBs: number;            // per-chip UB bandwidth (GB/s, bidirectional)
  aiSubsys: number | null;  // AI subsystems (each = 1 Cube + 2 Vector)
  estimated?: boolean;      // per-chip figures are estimates (A6)
}

export const GENERATIONS: Record<Gen, GenSpec> = {
  A5: {
    code: 'A5', name: TOK.atlas950, npuLabel: `${TOK.ascend} ${TOK.n950dt}`, npuShort: TOK.n950dt,
    totalNpus: 8192, fp8EF: 8, fp4EF: 16, memTB: 1152, memPerChipTBs: 4, interconnectPBs: 16, chipUbTBs: 2,
    computeCabs: 128, commCabs: 32, totalCabs: 160, footprintM2: 1000,
    hbm: TOK.hbmZQ, release: '2026 Q4', trainTokps: '4.91M tok/s', inferTokps: '19.6M tok/s',
    superclusterNpu: '>52万',
    memGB: 144, fp4Tflops: 2007, fp8Tflops: 1034, l2MB: 128, ubGBs: 2016, aiSubsys: 36,
  },
  A6: {
    code: 'A6', name: TOK.atlas960, npuLabel: `${TOK.ascend} ${TOK.n960}`, npuShort: TOK.n960,
    totalNpus: 15488, fp8EF: 30, fp4EF: 60, memTB: 4460, memPerChipTBs: 4, interconnectPBs: 34, chipUbTBs: 4,
    computeCabs: 176, commCabs: 44, totalCabs: 220, footprintM2: 2200,
    hbm: TOK.hbmZQ, release: '2027 Q4', trainTokps: '15.9M tok/s', inferTokps: '80.5M tok/s',
    superclusterNpu: '>100万',
    memGB: 288, fp4Tflops: 3874, fp8Tflops: 1937, l2MB: null, ubGBs: 4032, aiSubsys: null, estimated: true,
  },
};

export const DEFAULT_GEN: Gen = 'A5';

// per-node schematic constants (illustrative; real per-node config not public)
export const NPUS_PER_NODE = 8;
export const CPUS_PER_NODE = 4;
// 950-class package = 4 Die per card: 2 compute Die (UMA-merged → OS sees ONE device,
// which the software rank binds to 1:1) + 2 IO Die (interconnect / IO). die-to-die via UB/SIO.
export const COMPUTE_DIES_PER_CARD = 2;
export const IO_DIES_PER_CARD = 2;
export const DIES_PER_NPU = COMPUTE_DIES_PER_CARD + IO_DIES_PER_CARD;   // = 4 (950)
export const NODES_PER_CAB = 8;       // 8 nodes × 8 NPU = 64 NPU per compute cabinet

// ─── hw-native-sys 层级坐标 L7→L0（唯一层级编号，全部视图共用）────────────────
// 递归路径（每级经由固定互联缩放到下一级）：
//   L7 Global ─DCN→ L6 Cluster ─Scale-Out→ L5 Service Pool ─Pool 内互联→
//   L4 Pod(UBL128) ─Scale-Up→ L3 Host(1 OS) ─PCIe/UB→ L2 Chip·NPU ─封装互连→
//   L1 Die(可选) ─NoC→ L0 Core-Group(AIV·向量 / AIC·Cube / AICPU)
// 机柜是 L4 Pod 内的物理分组（无独立 L 级）；AI Core / Tile 归入 L0 Core-Group 内部
// （L0 内部组织复用 pto-design-system 的 memory-architecture pattern：GM/L2 轨道 +
//  AIV1/AIC/AIV2 + UB/L1/L0A/L0B/L0C 缓冲 + MTE/FixPipe 路径）。
export type LevelKey = 'global' | 'cluster' | 'pool' | 'super' | 'cab' | 'node' | 'card' | 'die' | 'core' | 'tile';
export interface HwLevel {
  L: number;            // 层级编号 7..0
  tag: string;          // 'L7'…'L0'；并入级（机柜 / tile）tag 为 ''
  key: LevelKey;        // 视图内部沿用的 key（super=Pod、node=Host、card=Chip、core=Core-Group）
  name: string;         // 中文显示名
  en: string;           // 英文名
  down?: { label: string; color: string; detail: string };   // 通往下一级的互联
  optional?: boolean;   // L1 Die 为可选层级
  folded?: boolean;     // 无独立 L 级、并入相邻级（cab→L4、tile→L0）
  example: string;      // sibling 样例（漏斗图右列）
}
export const HW_LEVELS: HwLevel[] = [
  { L: 7, tag: 'L7', key: 'global',  name: '全球调度', en: 'Global',
    down: { label: 'DCN', color: '#9d7bff', detail: '跨地域数据中心网络（DCN）· 南北向' }, example: 'Global A / C' },
  { L: 6, tag: 'L6', key: 'cluster', name: '集群', en: 'Cluster',
    down: { label: 'Scale-Out', color: '#ffaa3b', detail: `跨 Pool ${TOK.uboe}/RoCE 全光 scale-out` }, example: 'Cluster A / C' },
  { L: 5, tag: 'L5', key: 'pool',    name: '服务池', en: 'Service Pool',
    down: { label: 'Pool 内互联', color: '#04d793', detail: 'Pool 内 Pod 间互联（业务 / 资源分区）' }, example: 'Pool 1 / 3' },
  { L: 4, tag: 'L4', key: 'super',   name: 'Pod', en: 'Pod · UBL128',
    down: { label: 'Scale-Up', color: '#fb5b9a', detail: `UBL128 Scale-Up：${TOK.fullmesh} + UB 交换 Clos（机柜仅为物理分组，不是层级）` }, example: 'Pod α / γ' },
  { L: 3, tag: 'L3', key: 'node',    name: 'Host · 节点', en: 'Host · 1 OS',
    down: { label: 'PCIe / UB', color: '#38bdf8', detail: '主机内多 NPU 经 PCIe/UB 挂到 1 个 OS' }, example: 'Host 1 / 3' },
  { L: 2, tag: 'L2', key: 'card',    name: 'Chip · NPU', en: 'Chip · NPU',
    down: { label: '封装互连', color: '#2dd4bf', detail: '封装内 Die 间 UB/SIO（D2D 784 GB/s）' }, example: 'NPU 1 / 3' },
  { L: 1, tag: 'L1', key: 'die',     name: 'Die', en: 'Die', optional: true,
    down: { label: 'NoC', color: '#22d3ee', detail: '片上 NoC 互联各核组、共享 HBM（单 die 芯片可省略本级）' }, example: 'die 0' },
  { L: 0, tag: 'L0', key: 'core',    name: '核组', en: 'Core-Group',
    down: { label: 'MTE / FixPipe 流水', color: '#7dd3fc', detail: 'L0 内部：AIV·向量 / AIC·Cube / AICPU + GM/L2 轨道（memory-architecture）' }, example: 'AIV · AIC · AICPU' },
];
export const HW_BY_KEY: Record<LevelKey, HwLevel> = (() => {
  const m = {} as Record<LevelKey, HwLevel>;
  for (const l of HW_LEVELS) m[l.key] = l;
  // 非层级：机柜只是 L4 Pod 内的物理分组、Tile 只是 L0 Core-Group 内部粒度，
  // 两者都不出现在层级轴/漏斗/面包屑里（key 仅供旧代码路径兼容）。
  m.cab  = { ...m.super, key: 'cab',  tag: '', folded: true, name: '机柜（物理分组）', en: 'Cabinet (physical)', example: 'C0 / C1' };
  m.tile = { ...m.core,  key: 'tile', tag: '', folded: true, name: 'Tile（L0 内部）', en: 'Tile', example: 'lane 0…' };
  return m;
})();
/** 层级 tag（'L4'…；并入级返回 ''） */
export const levelTag = (k: LevelKey): string => HW_BY_KEY[k].tag;
/** 层级显示名 */
export const levelName = (k: LevelKey): string => HW_BY_KEY[k].name;
/** tag + 名（并入级不带 tag）：'L4 Pod · 超节点' / '机柜' */
export const levelFull = (k: LevelKey): string => { const l = HW_BY_KEY[k]; return l.tag ? `${l.tag} ${l.name}` : l.name; };
// 一个 Service Pool 含的 Pod 数（示意分组值，用于 L5 视图聚合）
export const PODS_PER_POOL = 4;
// 一个 L6 集群（SuperCluster）真实含的 Pod 数：Atlas 950 SuperCluster = 64 × SuperPoD
// （>52万卡 ÷ 8192 卡/Pod = 64；HC2025 徐直军演讲口径）→ 16 个服务池 × 4 Pod
export const PODS_PER_CLUSTER = 64;
export const POOLS_PER_CLUSTER = PODS_PER_CLUSTER / PODS_PER_POOL;   // = 16
// L0 Core-Group 构成（memory-architecture pattern）：1 AIC·Cube + 2 AIV·向量（+ AICPU 调度）
export const AIV_PER_COREGROUP = 2;

// ─── UB interconnect hierarchy (chip → cluster), drives all colour coding ─────
// id 使用互联名（NoC/封装、Host、Scale-Up、Scale-Out），不再占用 L 编号 —— L 编号
// 唯一归属上面的 hw-native-sys 层级坐标。数组顺序/颜色保持不变（大量按下标引用）。
export interface UbLevel { id: string; color: string; label: string; detail: string; }
export const UB_LEVELS: UbLevel[] = [
  { id: 'NoC·D2D',  color: '#2dd4bf', label: '片内（封装互连 + NoC）',    detail: 'L2 Chip 内：Die 间 UB/SIO 封装互连 · Die 内 NoC 至 L0 核组' },
  { id: 'UB·Host',  color: '#38bdf8', label: 'Host 内（PCIe / UB）',      detail: 'L3 Host：板载 UB 2D-Mesh，NPU 直连、同挂 1 OS' },
  { id: 'SU·柜内',  color: '#a78bfa', label: `机柜内 ${TOK.fullmesh}`,    detail: `L4 Pod 内：跨节点 ${TOK.fullmesh} 总线级直连（机柜并入 Pod）` },
  { id: 'SU·Pod',   color: '#fb5b9a', label: 'Pod Clos（Scale-Up）',      detail: 'L4 Pod（UBL128）：经 UB 交换(通信柜) Clos 全互联' },
  { id: 'SO',       color: '#04d793', label: 'Pod 间 Scale-Out',          detail: 'L5 Pool 内互联 / L6 Cluster scale-out（全光）' },
];

// Per UB level: scale-up/scale-out domain + bandwidth/latency + the parallel dims
// that prefer it. (SU = scale-up 超高带宽窄域 全互联 ≤128 卡 → TP·EP; SO = scale-out
// 高带宽广域，双层 UB 交换 → PP·DP. Sources: UB 互联研究 @ Hot Chips.)
export interface UbLevelMeta { domain: 'SU' | 'SO'; bw: string; parallel: string; }
export const UB_LEVEL_META: Record<string, UbLevelMeta> = {
  'NoC·D2D': { domain: 'SU', bw: 'D2D 双向 784 GB/s',           parallel: '片内 die 对等' },
  'UB·Host': { domain: 'SU', bw: '单卡 UB 2016 GB/s · 板载 2D-Mesh', parallel: 'TP 张量并行（窄快）' },
  'SU·柜内': { domain: 'SU', bw: `柜内 ${TOK.fullmesh} · 单跳 200 ns · 1:1 无收敛`, parallel: 'TP·EP（SU 超低延迟域）' },
  'SU·Pod':  { domain: 'SO', bw: 'any-to-any <1 µs · 16 PB/s · 双层 UB 交换', parallel: 'EP·PP（SO 广域）' },
  'SO':      { domain: 'SO', bw: `跨 Pod ${TOK.uboe}/RoCE`,      parallel: 'DP 数据并行（广而省）' },
};

// ─── Physical communication planes (三平面) + physical devices ────────────────
// The single logical "NPU 经 UB 全互联" line actually rides THREE distinct PHYSICAL
// planes — the layer the pure logical bus model omitted. Each NPU exposes TWO
// different port groups (UB scale-up vs RDMA scale-out, physically different SerDes
// groups), and the CPU drives a third plane (VPC) via the NIC. The optical segment
// of the scale-up / scale-out planes is carried by LPO (linear-drive, DSP-less)
// modules. This lets "NPU 经 UB 全互联" expand into the real chain
// "NPU 端口 → 铜/LPO → 交换", and separates TP/EP (scale-up) · DP/PP (scale-out) ·
// 南北向 (VPC). Sources: CloudMatrix384 三平面解读 (InfoQ/知乎, 厂商口径 C) ·
// LPO 功耗/时延 (Vitex, C). Figures标"趋势/待核"处为下一代(LPO 800G/UB2.0)推断。
export type PlaneId = 'ub' | 'rdma' | 'vpc';
export interface PlaneSpec {
  id: PlaneId; name: string; short: string; color: string;
  role: string; members: string; devices: string; scope: string; parallel: string;
  confidence: 'A' | 'B' | 'C';
}
export const PLANES: PlaneSpec[] = [
  { id: 'ub', name: `${TOK.ub} 平面 · Scale-up`, short: 'UB·SU', color: '#04d793',
    role: '超节点内全互联（NPU↔NPU↔CPU）', members: `NPU + CPU`,
    devices: `NPU UB 端口 → 铜缆(柜内)/LPO 光模块(柜间) → L1/L2 ${TOK.ub} 交换`,
    scope: '超节点内 SU 域 · ≤128 卡超低延迟 · >2 TB/s·NPU', parallel: 'TP · EP · SP', confidence: 'B' },
  { id: 'rdma', name: 'RDMA 平面 · Scale-out', short: 'RDMA·SO', color: '#ffaa3b',
    role: '跨超节点 / 外部 RDMA（RoCE）', members: '仅 NPU（自带 RoCE 口）',
    devices: 'NPU RDMA 端口(400G/NPU) → LPO 光模块 → scale-out 交换 → 其它超节点',
    scope: '跨超节点 SO 域', parallel: 'DP · PP', confidence: 'B' },
  { id: 'vpc', name: 'VPC 平面', short: 'VPC', color: '#9d7bff',
    role: '接入数据中心（存储 / 前端 / 管理 · 南北向）', members: `CPU + ${TOK.qingtian} NIC`,
    devices: `CPU → ${TOK.qingtian} NIC → 数据中心网络`,
    scope: '南北向 · 非训练关键路径', parallel: '—', confidence: 'B' },
];

// physical devices the planes traverse (这是现模型最该补的器件层)
export interface PhysDevice { id: string; label: string; color: string; plane: PlaneId | 'multi'; note: string; }
export const PHYS_DEVICES: PhysDevice[] = [
  { id: 'npu_ub_port', label: 'NPU UB 端口', color: '#04d793', plane: 'ub',
    note: 'SerDes/LQC 高速口，进 UB 总线 · 单卡 >2 TB/s · scale-up 最高带宽' },
  { id: 'npu_rdma_port', label: 'NPU RDMA 端口', color: '#ffaa3b', plane: 'rdma',
    note: '集成在 NPU 上的 RoCE 口 · 400 Gbps/NPU · 跨超节点（与 UB 口是 NPU 上不同的 SerDes 组）' },
  { id: 'cpu_ub', label: `${TOK.kunpeng} CPU`, color: '#4a8cff', plane: 'multi',
    note: 'UB 平面与 NPU 平等互联(8×30G LQC 统一编址) · host 侧挂 NIC 接 VPC · 调度/预处理/存储' },
  { id: 'lpo', label: 'LPO 光模块', color: '#36e0c4', plane: 'multi',
    note: '线性直驱(去 DSP)：功耗降 35–50%(→7–8.5W)、单跳<3ns · 柜间光链路介质 · scale-up/out 共用' },
  { id: 'nic', label: `${TOK.qingtian} NIC`, color: '#9d7bff', plane: 'vpc',
    note: '负责 VPC 平面（注意：scale-out RDMA 走 NPU 自带 RoCE 口，不走擎天 NIC）· 南北向接入 DC' },
];

// the physical hop-chain per plane — 把"NPU 经 UB 全互联"那根逻辑线展开成物理链
export interface PhysChain { plane: PlaneId; label: string; hops: string[] }
export const PHYS_CHAINS: PhysChain[] = [
  { plane: 'ub', label: 'Scale-up（超节点内）', hops: ['NPU Die', 'UB 端口', '铜缆 / LPO 光模块', `L1→L2 ${TOK.ub} 交换`] },
  { plane: 'rdma', label: 'Scale-out（跨超节点）', hops: ['NPU Die', 'RDMA/RoCE 口 400G', 'LPO 光模块', 'scale-out 交换', '其它超节点'] },
  { plane: 'vpc', label: 'VPC（南北向）', hops: [`${TOK.kunpeng} CPU`, `${TOK.qingtian} NIC`, '数据中心网络'] },
];

// ── PER-LEVEL physical devices & plane (把物理器件挂到每一层级上) ─────────────────
// Mirrors the reference "物理三平面" layer: each hierarchy level carries WHICH physical
// devices live there and WHICH plane it rides. Keyed by the level `kind` used in the
// 层级图 (LAY.levels.kind) and mapped onto the 阵列全景 bands. `color` = plane accent
// (grey = on-chip, no external port). Consumed by PlaneView (层级图) + FullPodScene bands.
export interface LevelPhys { plane: PlaneId | 'none' | 'multi'; planeLabel: string; color: string; devices: string; short: string }
export const LEVEL_PHYS: Record<string, LevelPhys> = {
  global:  { plane: 'vpc',  planeLabel: 'DCN · 南北向',      color: '#9d7bff', short: 'DCN → 各集群', devices: '跨地域数据中心网络（DCN）· 全局调度' },
  cluster: { plane: 'rdma', planeLabel: 'RDMA · Scale-out', color: '#ffaa3b', short: 'RDMA口→其它 Pool/Pod', devices: `跨 Pool · ${TOK.uboe}/RoCE（NPU RDMA 口）` },
  pool:    { plane: 'rdma', planeLabel: 'Pool 内互联',       color: '#04d793', short: 'Pool 内 Pod 间互联', devices: `Pool 内 Pod 间 ${TOK.uboe}/RoCE · 业务/资源分区` },
  super:   { plane: 'ub',   planeLabel: 'UB · Scale-up',    color: '#04d793', short: 'UB 交换 · LPO', devices: `L2 ${TOK.ub} 交换 · LPO 光模块(柜间)` },
  cab:     { plane: 'ub',   planeLabel: 'UB · Scale-up',    color: '#04d793', short: '柜内 mesh · 铜/LPO', devices: `柜内 ${TOK.fullmesh} · 铜/LPO 上行` },
  node:    { plane: 'multi', planeLabel: 'UB / RDMA / VPC', color: '#9d7bff', short: 'UB口/RDMA口·CPU·LPO·NIC', devices: 'NPU UB口 + RDMA口 · 鲲鹏 CPU · LPO · 擎天 NIC · L1 交换' },
  card:    { plane: 'multi', planeLabel: 'UB + RDMA',       color: '#04d793', short: 'NPU UB口 + RDMA口', devices: 'NPU 封装：UB 端口(绿·scale-up) + RDMA/RoCE 端口(橙·scale-out)' },
  die:     { plane: 'none',  planeLabel: '片上 · 无对外口',  color: '#7c8db8', short: '片上 · 无对外口', devices: 'D2D 784 GB/s · NoC · HBM' },
  core:    { plane: 'none',  planeLabel: '片上 · 无对外口',  color: '#7c8db8', short: '片上 · 无对外口', devices: 'Core-Group：AIC(Cube)/AIV(Vector)/AICPU · GM/L2 轨道' },
  tile:    { plane: 'none',  planeLabel: '片上 · 无对外口',  color: '#7c8db8', short: '片上 · 无对外口', devices: 'Cube/Vector 单元 · UB/L1/L0A/L0B/L0C buffer' },
};
// 阵列全景 band index → LEVEL_PHYS key（band index == L 编号，严格 8 级链 L0→L7；
// 机柜/Tile 不是层级，不在此表——机柜=Pod 内物理分组，Tile=L0 Core-Group 内部粒度）
export const BAND_PHYS_KEY: Record<number, string> = {
  0: 'core', 1: 'die', 2: 'card', 3: 'node', 4: 'super', 5: 'pool', 6: 'cluster', 7: 'global',
};

// Each hierarchy level carries HARDWARE facts (hw) and the SOFTWARE view (sw)
// SEPARATELY — rank is pure software (a collective-comm logical id) bound to a device,
// never the device itself. Tuned for the 950 (4-Die package · UMA · ≈32 AI Core/card).
// 8 entries, one per hw-native-sys level L7→L0（与 PlaneView LAY.defs 按位对齐）。
// 机柜 / Tile 不是层级，不在此表：机柜=Pod 内物理分组，Tile=L0 Core-Group 内部粒度。
export interface LayerInfo { key: string; name: string; intra: string; inter: string; bw: string; domain: string; tag?: string; hw?: string; sw?: string; }
export const LAYER_INFO: LayerInfo[] = [
  { key: 'global', name: 'Global · 全球调度（L7）', intra: '跨地域多集群统一调度 / 容灾 / 就近接入', inter: '↓ DCN：经数据中心网络下挂各 Cluster', bw: 'DCN 广域 · 南北向', domain: '—', sw: '作业全局编排 · 端到端吞吐 / MFU' },
  { key: 'cluster', name: 'Cluster · 集群（L6）', intra: `集群内跨 Pool 调度 · DP/PP 跨 Pod`, inter: `↓ Scale-Out：${TOK.uboe}/RoCE 全光下挂各 Service Pool`, bw: `跨 Pod ${TOK.uboe}/RoCE`, domain: 'SO', sw: 'DP 数据并行域 · 集群通信占比' },
  { key: 'pool',  name: 'Service Pool · 服务池（L5）', intra: '业务 / 资源分区：训练池 · 推理池 · 混部池', inter: '↓ Pool 内互联：池内 Pod 间互联', bw: 'Pool 内 Pod 间全光', domain: 'SO', sw: '任务排布 / 配额边界' },
  { key: 'super', name: 'Pod（L4 · UBL128）', intra: `Scale-Up 域内全互联 · ${TOK.ubmesh}（≤128 卡超低延迟 SU 域，机柜仅为物理分组）`, inter: '↓ Scale-Up：UB 统一编址 → Pod =“一台计算机”', bw: 'any-to-any <1 µs · 16 PB/s', domain: 'SU+SO', sw: 'TP/EP/SP 域(SU) · 部署边界' },
  { key: 'node',  name: 'Host（L3 · 1 OS）', intra: '8 卡 + CPU 经 PCIe/UB（LQC）全互联、平等编址、同挂 1 个 OS', inter: '↓ PCIe / UB：多 NPU 挂入同一 Host OS（单跳 200 ns · 1:1）', bw: 'LQC 8×56G(卡) / 8×30G(CPU)', domain: 'SU' },
  { key: 'card',  name: 'Chip · NPU（L2 · 1 device）', intra: '封装互连：4 Die = 2 计算 Die（UMA 高带宽直连、OS 视为单设备）+ 2 IO Die', inter: '↓ 封装互连：Die 间 UB/SIO（D2D 784 GB/s）', bw: 'D2D · HBM · 单卡 UB 2016 GB/s', domain: '—', tag: 'device ↔ rank 1:1',
    hw: '硬件：1 颗 950 Chip = 1 device = 2 计算 Die（UMA 合并）+ 2 IO Die', sw: `软件：rank = ${TOK.hccl} 逻辑编号（rank 表），与 device 严格 1:1 绑定 · 纯软件、与代际无关` },
  { key: 'die',   name: 'Die（L1 · 可选）', intra: '单计算 Die ≈ 16 AI Core（若干核组），经片上 NoC 互联、共享 HBM', inter: '↓ NoC：片上 NoC 下挂各 Core-Group（单 die 芯片可省略本级）', bw: '片上 NoC · D2D 784 GB/s · HBM 3.2–9.6 TB/s', domain: '—', tag: '设备内（非 rank）· 可选级',
    hw: '硬件：1 计算 Die ≈ 16 AI Core · 整卡 = 2 计算 Die（UMA）+ 2 IO Die', sw: '软件：rank 内 Die 分区 · 同 rank、不增 rank' },
  { key: 'core',  name: 'Core-Group · 核组（L0 · AIV/AIC/AICPU）', intra: '核组 = 1 AIC(Cube) + 2 AIV(Vector)（+ AICPU 调度）· 内部按 memory-architecture 组织：GM/L2 轨道 → UB/L1/L0A/L0B/L0C 缓冲 → MTE/FixPipe 流水', inter: '↓ 内部：Tile / SIMT lane（不再是层级）· block_idx 核实例（SPMD）', bw: 'GM/L2 轨道 · UB/L1/L0A/L0B/L0C · TQue/TPipe 流水', domain: '—', tag: '设备内并行（非 rank）',
    hw: '硬件：约 32 AI Core/卡（16/计算 Die × 2）· AIC(Cube)/AIV(Vector) 分离独立核 · 核组内 1 Cube + 2 Vector', sw: '软件：block_idx 核实例（SPMD）· rank 内不增 rank' },
];

// per-card AI Core count used by the on-chip GRIDS (≈16 AI Core / compute Die × 2 compute Die = 32).
// NOTE: this is the visualisation's REPRESENTATIVE core grid. It is a DIFFERENT granularity from the
// published spec figure `GenSpec.aiSubsys` (A5 = 36 AI 子系统, each = 1 Cube + 2 Vector) — subsystem ≠
// core. Both coexist on purpose: spec panels cite aiSubsys (36 子系统); die/core/tile grids use 32 cores
// (Cube∶Vector ≈ 8∶1). Keep them labelled distinctly so "32" and "36" never read as a contradiction.
export const CORES_PER_CARD = 32;

// ─── Process / thread communication overlays (node view) ─────────────────────
export interface CommPattern { id: string; color: string; label: string; }
export const COMM_PATTERNS: CommPattern[] = [
  { id: 'ring',   color: '#ff4b7b', label: 'Ring-AllReduce · 进程(rank)' },
  { id: 'a2a',    color: '#ffaa3b', label: 'All-to-All MoE · 进程(rank)' },
  { id: 'thread', color: '#22d3ee', label: 'die 内线程 / AI Core 流' },
];

// ─── Trace timeline (illustrative training-iteration schedule, NOT a real profile) ─
export type Phase = 'load' | 'compute' | 'comm' | 'store';
export const TRACE_SCHED: Phase[] = ['load', 'compute', 'compute', 'comm', 'compute', 'compute', 'comm', 'store'];
export const PHASE_META: Record<Phase, { name: string; color: string }> = {
  load:    { name: '加载',           color: '#c2c9d4' },
  compute: { name: '计算（算子/Tile）', color: '#22d3ee' },
  comm:    { name: '通信 AllReduce',  color: '#ff4b7b' },
  store:   { name: '存储',           color: '#aab4c4' },
};

// ─── Full-pod "running" view: train / infer iteration schedules ───────────────
// Drives the phase wash + collectives over the full super-node. A schematic loop
// (illustrative, not a real profile). `kind` selects what lights up: compute →
// AI cores/cards, comm → ranks + the named collective, load/store/mem → data.
export type RunMode = 'train' | 'infer';
export type RunKind = 'load' | 'compute' | 'comm' | 'store' | 'mem';
export interface RunPhase {
  id: string; name: string; kind: RunKind; color: string;
  collective?: 'ring' | 'a2a';   // comm phases: which collective animates
  parallel?: string;             // the parallel dim exercised (TP/PP/DP/EP)
  note: string;
}
// ─── REAL workload profile: Pangu Pro MoE (72BA16B) ──────────────────────────
// The concrete MoE model whose measured config / parallelism / collectives / throughput
// GROUND the 工况·通信 overlays (previously round guesses). Every field below is taken
// from the Pangu Pro MoE technical report (arXiv:2505.21411v2, Tables 1/2/5/6/7 + §4).
//
// IMPORTANT scope note: the paper deploys on Ascend 300I Duo / 800I A2 — a DIFFERENT
// platform than this app's Atlas 950/960 (A5/A6) super-node hardware model. So ONLY the
// workload (model shape, parallel degrees, collective ops, per-card tok/s) is real here;
// the cluster/cabinet/NPU hardware topology stays the A5/A6 model. The overlays cite the
// paper as the workload source, not as this hardware's profile.
export interface WorkloadProfile {
  name: string; short: string;
  totalB: number; activeB: number;        // params (B): total / activated-per-token
  layers: number; noopLayers: number;     // 48 transformer + 2 no-op (PP load-balance) = 50
  hidden: number; intermediate: number;
  queryHeads: number; kvHeads: number; headSize: number;   // GQA
  vocab: number;
  routedExperts: number; activatedExperts: number; sharedExperts: number;   // MoGE: 64 / 8 / 4
  trainTokens: string;                     // pre-training corpus
  trainNpus: number;                       // NPUs used for pre-training
  // parallel degrees
  train: { tp: number; ep: number; pp: number; vpp: number; cp: number };
  inferAttn: { dp: number; tp: number };   // H2P attention: DP2 + TP4
  inferRouted: { tp: number; ep: number }; // H2P routed experts: TP2 + EP4
  inferSharedTP: number;                   // shared experts: TP8
  // measured per-card throughput on Ascend 800I A2 (W8A8)
  perf: {
    prefillTokps: number; prefillTTFTms: number;      // batch 2, seq 2048
    decodeTokps: number; decodeTPOTms: number; decodeBatch: number;
    decodeMtpTokps: number;                           // with MTP speculative decoding
    weightXferPct: number;                            // weight transfer = 29% of decode latency
    epCommPct: number;                                // EP all-gather/reduce-scatter ≈ 8% of net latency
  };
  mfuGainPct: number;                      // +35% relative MFU after H2P/overlap/fused-op optimisation
}
export const WORKLOAD: WorkloadProfile = {
  name: TOK.panguProMoe, short: '72BA16B',
  totalB: 71.99, activeB: 16.5, layers: 48, noopLayers: 2,
  hidden: 5120, intermediate: 1344, queryHeads: 40, kvHeads: 8, headSize: 128, vocab: 153376,
  routedExperts: 64, activatedExperts: 8, sharedExperts: 4,
  trainTokens: '13T', trainNpus: 4096,
  train: { tp: 8, ep: 2, pp: 5, vpp: 5, cp: 1 },
  inferAttn: { dp: 2, tp: 4 }, inferRouted: { tp: 2, ep: 4 }, inferSharedTP: 8,
  perf: {
    // Ascend 800I A2 · 表 6：Decode batch64 → 95.56 ms TPOT / 1148 tok/s（+MTP 1528）；batch1 → 456 tok/s。
    // 修正：原 decodeBatch=456 实为 batch-1 的 tok/s、decodeTPOTms=99 实为 300I Duo 值，均与 800I A2·1148 不配。
    prefillTokps: 4828, prefillTTFTms: 424, decodeTokps: 1148, decodeTPOTms: 96, decodeBatch: 64,
    decodeMtpTokps: 1528, weightXferPct: 29, epCommPct: 8,
  },
  mfuGainPct: 35,
};

// MoGE routing + kernel/comm-optimisation facts (arXiv:2505.21411 §2/§4) — annotate overlays.
export const WORKLOAD_DETAIL = {
  moge: { perGroupTopK: 1, imbalanceScore: 0, imbalanceReductionPct: 50, note: 'N=64 专家均分 M 组，每组 Top-1 → 设备负载天然均衡(IS=0)' },
  kernel: { mulAttnSpeedup: 4.5, attnLatencyPct: [30, 50] as [number, number], kvOfAttnPct: 70, swiftGmmLatencyPct: 50 },
  comm: { allreduceCutPct: 50, rmsnormCutPct: 75, fusedOps: ['GMMRS', 'AGMM'] as string[] },
} as const;

// Same-family REAL results on Ascend super-nodes (each a confidence-A paper) — a 同类对照
// panel. These are NOT this app's platform; shown as reference context alongside the primary
// Pangu Pro MoE workload. (arXiv ids: 2505.04519 / 2506.12708 / 2508.02520 / 2504.07866 / 2509.11662)
export interface WorkloadRef { id: string; title: string; arxiv: string; scale: string; metric: string; }
export const WORKLOAD_REFS: WorkloadRef[] = [
  { id: 'pangu-pro-moe',   title: TOK.panguProMoe,                 arxiv: '2505.21411', scale: '72BA16B · 64 专家',        metric: 'Decode 1148→MTP 1528 · Prefill 4828 tok/s·卡' },
  { id: 'pangu-ultra-moe', title: `${TOK.pangu} Ultra MoE`,        arxiv: '2505.04519', scale: '718B · 256 专家 · 6000 NPU', metric: 'MFU 18.9%→30% · 1.46M tok/s' },
  { id: 'deepseek-cm384',  title: 'DeepSeek-R1 · CloudMatrix384',  arxiv: '2506.12708', scale: '671B MoE · 384 NPU',        metric: 'Prefill 6688 / Decode 1943 tok/s·NPU (TPOT<50ms)' },
  { id: 'xdeepserve',      title: 'xDeepServe MaaS · CM384',       arxiv: '2508.02520', scale: '384×910C · UB 全互联',      metric: '2400 tok/s·NPU · 系统 345K tok/s (DP288)' },
  { id: 'pangu-ultra-135b',title: `${TOK.pangu} Ultra 135B Dense`, arxiv: '2504.07866', scale: '135B Dense · 8192 卡',       metric: '13.2T tokens · depth-scaled sandwich norm' },
  { id: 'mindvl',          title: 'MindVL 多模态 · Ascend',        arxiv: '2509.11662', scale: '原生分辨率 ViT · 447B tokens', metric: '强扩展 >94% (1→128 NPU) · 1/10 数据量' },
];

// Base-model benchmark scores (arXiv:2505.21411 Table 3) — Pangu Pro MoE vs comparable
// 27–32B models. Used for a 模型质量对照 bar panel (Pangu highlighted). Scores are 0–100
// (EM/F1/Pass@1). Pangu is the LAST column.
export const BENCH_MODELS: string[] = ['Qwen2.5-32B', 'GLM4-32B', 'Gemma3-27B', 'Llama4-Scout', TOK.panguProMoe];
export const BENCH_PANGU_IDX = BENCH_MODELS.length - 1;
export interface BenchRow { name: string; scores: number[]; }
export const BENCHMARKS: BenchRow[] = [
  { name: 'MMLU',      scores: [84.2, 82.0, 78.6, 78.3, 87.4] },
  { name: 'MMLU-Pro',  scores: [58.0, 55.8, 50.3, 50.3, 63.5] },
  { name: 'C-Eval',    scores: [87.7, 84.1, 69.4, 74.8, 90.6] },
  { name: 'CMMLU',     scores: [88.9, 83.8, 70.4, 76.8, 89.0] },
  { name: 'GSM8K',     scores: [83.0, 85.4, 82.6, 79.2, 86.5] },
  { name: 'HumanEval', scores: [57.9, 59.1, 48.8, 54.6, 63.7] },
  { name: 'HellaSwag', scores: [93.1, 92.6, 84.1, 81.9, 93.5] },
];

// Per-phase step-time decomposition (计算 / 通信 / 访存), grounded in the Pangu Pro MoE
// report rather than guessed: decode is memory-bound (weight transfer ≈29% of latency + KV),
// EP All-to-All ≈8% of network latency; prefill is compute-bound (Top-8 experts, big GEMMs);
// pre-training carries DP-AllReduce gradient sync + EP All-to-All. Fractions are the paper's
// hard anchors rounded to sum to 1 (illustrative split of the residual).
export type StepPhase = 'pretrain' | 'prefill' | 'decode';
export interface StepPart { label: string; frac: number; color: string; }
export const STEP_DECOMP: Record<StepPhase, StepPart[]> = {
  pretrain: [
    { label: '计算(FWD/BWD)', frac: 0.58, color: '#22d3ee' },
    { label: '通信(DP AllReduce+EP A2A)', frac: 0.30, color: '#ff4b7b' },
    { label: '访存', frac: 0.12, color: '#a78bfa' },
  ],
  prefill: [
    { label: '计算(Top-8 GEMM)', frac: 0.72, color: '#22d3ee' },
    { label: '通信(EP All-to-All)', frac: 0.16, color: '#ff4b7b' },
    { label: '访存(KV 建立)', frac: 0.12, color: '#a78bfa' },
  ],
  decode: [
    { label: '计算', frac: 0.40, color: '#22d3ee' },
    { label: '通信(EP A2A≈8%)', frac: 0.12, color: '#ff4b7b' },
    { label: '访存(权重29%+KV)', frac: 0.48, color: '#a78bfa' },
  ],
};

// ─── Model-parallel partition (maps a sharded model onto the physical levels) ─
// TP = within a host (8 NPU, L3) · PP = hosts within a replica (L4 Pod 内) ·
// DP = replicas across pods (L5 Pool / L6 Cluster) · EP = experts per cabinet (L4 内).
export type PartitionDim = 'none' | 'tp' | 'pp' | 'dp' | 'ep';
export const PARTITION_META: Record<Exclude<PartitionDim, 'none'>, { label: string; level: string; comm: string; same: string }> = {
  tp: { label: 'TP 张量并行', level: 'L3 Host 内（8 卡/节点）', comm: 'AllGather / ReduceScatter', same: '同色 = 同一张量切片（tp rank 0–7，每 Host 复现）' },
  pp: { label: 'PP 流水并行', level: 'L4 Pod 内 · 跨 Host/机柜', comm: '阶段间 P2P 激活传递',        same: '同色 = 同一流水级（承载相同层）' },
  dp: { label: 'DP 数据并行', level: '跨 Pod（L5 Pool / L6 Cluster）', comm: '梯度 AllReduce',       same: '同色 = 同一数据副本' },
  ep: { label: 'EP 专家并行', level: '训练折入 DP 轴（相邻副本）· 推理节点内路由', comm: 'dispatch/combine All-to-All', same: '同色 = 持有同一专家分桶（experts 相同 · 桶↔卡非 1:1，每个 A2A 域各持一桶）' },
};
// cycling palette: group g → PARTITION_PALETTE[g % len] (same colour = same parallel group)
// de-RYG: parallel-group palette uses ONLY non-state hues (blue/indigo/violet/cyan/pink) so it
// never collides with the red/yellow/green state colours. (partition is an opt-in cognition lens)
export const PARTITION_PALETTE = ['#3b82f6', '#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#ec4899', '#06b6d4', '#0ea5e9', '#818cf8', '#c084fc', '#22d3ee', '#f472b6'];

// canonical signature colour per parallel dimension (one colour each for none/TP/PP/DP/EP),
// used for the dimension *selector* chips + legends so each dim reads consistently.
export const PARALLEL_COLORS: Record<PartitionDim, string> = {
  none: '#7c8db8', tp: '#4369ef', pp: '#04d793', dp: '#ffaa3b', ep: '#ff4b7b',
};
export type SpDim = 'sp';
export type ParDim = Exclude<PartitionDim, 'none'> | SpDim;   // tp/pp/dp/ep + sp
export const PARALLEL_COLORS_SP = '#22d3ee';   // SP signature colour (cyan, distinct from TP)

// ─── SINGLE SOURCE OF TRUTH for the model-parallel mapping (TP/SP/EP/PP/DP) ───
// EVERY view (平面 groupOf · 工作台 groups · 3D FullPodScene.part · 运行状态 通信域) must read
// this so the parallel DEGREES + the GROUP membership + the COMM peers agree everywhere.
//
// Placement doctrine (matches 昇腾 SU/SO 域):
//   · TP = 节点内 8 卡 (L1 UB mesh · 最高带宽)  —— tp rank = k % 8, one node = one TP group.
//   · SP = 与 TP 同域 (共组切序列) —— 本工况 CP1 未独立切分，别名到 TP。
//   · PP = 副本内跨节点 (P2P 流水) —— stage = node % PP, replica = ⌊node/PP⌋ (连续 PP 节点=1 副本).
//   · DP = 副本间 (Ring-AllReduce) —— 同 (tp, stage) 跨全部副本.
//   · EP = 专家 All-to-All —— 训练：折叠进 DP 轴 (EP 个相邻副本 A2A)；推理：节点内路由专家 (TP2+EP4).
//
// Real 盘古 Pro MoE (arXiv:2505.21411): 训练 TP8·EP2·PP5·VPP5·CP1；推理 H²P 注意力 DP2+TP4 /
// 路由 TP2+EP4 / 共享 TP8. 真实作业 ~4K 卡除不尽 8192 卡 950 超节点，所以物理平铺取「整除近似」：
// PP 5→与 8192 卡整除的最近约数，DP 填充到铺满 —— 逻辑并行度(real)与物理平铺度(tiled)都保留、都标注。
export type ParallelWorkload = 'pretrain' | 'prefill' | 'decode';

// ─── Cross-view sync: shared 工况 / 时间 / 播放 so 运行状态 ⇄ 工作台 stay linked when you switch
// tabs (combined with the shared load model, both tabs show the SAME world at the same t). Optional —
// a view falls back to its own local state when no sync is provided (standalone use). ───
export interface ViewSync {
  workload: ParallelWorkload;
  step: number;
  playing: boolean;
  metric: 'util' | 'strag' | 'fault';
  planeOn: { ub: boolean; rdma: boolean; vpc: boolean };
  setWorkload: (w: ParallelWorkload) => void;
  setStep: (n: number | ((s: number) => number)) => void;
  setPlaying: (b: boolean | ((b: boolean) => boolean)) => void;
  setMetric: (m: 'util' | 'strag' | 'fault') => void;
  setPlaneOn: (fn: (p: { ub: boolean; rdma: boolean; vpc: boolean }) => { ub: boolean; rdma: boolean; vpc: boolean }) => void;
  // ── 统一驾驶舱（CockpitApp）联动扩展 · 全部可选、向后兼容 ──
  // 着色互斥（策略透镜）· 图层开关（通信线/热点告警）· 层级聚合粒度（来自层级轴）· 选中 rank。
  // 既有 viewSync 不带这些字段时行为逐字节不变。
  stratColor?: PartitionDim;   // 策略着色（none/tp/pp/dp/ep）——互斥图层
  commLayer?: boolean;         // 通信连线图层开关
  alertLayer?: boolean;        // 热点/告警图层开关
  aggLevel?: LevelKey;         // 左画布聚合粒度（层级轴点选驱动）
  selRank?: number | null;     // 当前选中 rank（左右联动共用选区）
}
// ── 外部并行配置（P0 · 加法）：监控可从真实作业「摄入」并行度以覆盖自动推导。
// 全部字段可选；不传任何字段时 parallelMap 的行为与升级前逐字节一致。 ──
export interface ParallelConfig { tp?: number; sp?: number; pp?: number; dp?: number; ep?: number; }
export interface ParallelMapping {
  workload: ParallelWorkload; N: number; training: boolean;
  tp: number; sp: number; ep: number; pp: number; dp: number;   // physical tiling degrees (整除铺满)
  epScope: 'replica' | 'node';                                   // where EP All-to-All lives
  cfg: string;                 // tiled config display, e.g. "TP8×PP4×DP256 · EP2"
  real: string;                // the real paper config (logical) — always shown alongside
  approxNote: string;          // what was approximated to divide N evenly
  groupOf: (k: number, dim: ParDim) => number;                  // colour-group id per dim
  groupCount: (dim: ParDim) => number;                          // number of distinct groups
  peersOf: (k: number, dim: ParDim, cap?: number) => number[];  // cards that actually communicate w/ k
  collectiveOf: (dim: ParDim) => 'ring' | 'a2a' | 'p2p';
}
// nearest integer divisor of n to a target (for 整除近似铺满)
function nearestDivisor(n: number, target: number): number {
  let best = 1, bestD = Infinity;
  for (let d = 1; d <= n; d++) if (n % d === 0) { const gap = Math.abs(d - target); if (gap < bestD) { bestD = gap; best = d; } }
  return best;
}
const TP_NODE = NPUS_PER_NODE;   // 8 卡/节点 = TP 域大小

export function parallelMap(workload: ParallelWorkload, N: number, cfgIn?: ParallelConfig): ParallelMapping {
  // cfgIn（可选·加法）：外部摄入的真实并行度覆盖自动推导。用 `cfgIn?.x ?? <推导>`，
  // 不传 cfgIn 时每个度数与升级前逐字节一致，且覆盖沿依赖链（nodes→PP→DP）正确级联。
  const training = workload === 'pretrain';
  const TP = cfgIn?.tp ?? Math.min(TP_NODE, N);
  const nodes = Math.max(1, Math.floor(N / TP));
  const realPP = training ? 5 : 1, realEP = training ? 2 : 4;   // 训练 EP2 / 推理路由 EP4
  const PP = cfgIn?.pp ?? (training ? nearestDivisor(nodes, realPP) : 1);
  const DP = cfgIn?.dp ?? Math.max(1, Math.floor(nodes / PP));
  const epScope: 'replica' | 'node' = training ? 'replica' : 'node';
  const EP = cfgIn?.ep ?? (training ? Math.max(1, nearestDivisor(DP, realEP)) : Math.min(realEP, TP));
  const SP = cfgIn?.sp ?? 1;   // CP1 → SP 未独立切分

  const nodeOf = (k: number) => Math.floor(k / TP);
  const stageOf = (k: number) => nodeOf(k) % PP;
  const replicaOf = (k: number) => Math.floor(nodeOf(k) / PP);
  const range = (a: number, b: number) => { const o: number[] = []; for (let i = a; i < b; i++) o.push(i); return o; };

  const groupOf = (k: number, dim: ParDim): number => {
    switch (dim) {
      case 'tp': case 'sp': return k % TP;                                  // tensor slice (repeats per node)
      case 'pp': return stageOf(k);                                         // pipeline stage
      case 'dp': return replicaOf(k);                                       // data-parallel replica
      case 'ep': return epScope === 'replica' ? replicaOf(k) % EP           // which expert shard (folded in DP)
                                              : Math.floor((k % TP) / Math.max(1, TP / EP));   // node-internal routed shard
    }
  };
  const groupCount = (dim: ParDim): number => (dim === 'tp' || dim === 'sp' ? TP : dim === 'pp' ? PP : dim === 'dp' ? DP : EP);
  const peersOf = (k: number, dim: ParDim, cap = 64): number[] => {
    const nb = nodeOf(k), tp = k % TP, st = stageOf(k), rep = replicaOf(k);
    if (dim === 'tp' || dim === 'sp') return range(nb * TP, Math.min(N, nb * TP + TP));       // node mates
    if (dim === 'pp') return range(0, PP).map((s) => (rep * PP + s) * TP + tp).filter((x) => x < N);   // stage chain, same replica+tp
    if (dim === 'dp') { const o: number[] = []; for (let r = 0; r < DP && o.length < cap; r++) { const x = (r * PP + st) * TP + tp; if (x < N) o.push(x); } return o; }
    // ep
    if (epScope === 'node') return range(nb * TP, Math.min(N, nb * TP + TP));                 // routed experts A2A within node
    const blk = Math.floor(rep / EP); const o: number[] = [];                                 // EP replicas that A2A (folded in DP)
    for (let r = blk * EP; r < (blk + 1) * EP && r < DP; r++) { const x = (r * PP + st) * TP + tp; if (x < N) o.push(x); }
    return o;
  };
  const collectiveOf = (dim: ParDim): 'ring' | 'a2a' | 'p2p' => (dim === 'pp' ? 'p2p' : dim === 'ep' ? 'a2a' : 'ring');

  const cfg = training ? `TP${TP}×PP${PP}×DP${DP} · EP${EP}` : `TP${TP}×DP${DP} · EP${EP}(节点内路由) · PP1`;
  const real = training
    ? '真实 TP8·EP2·PP5·VPP5·CP1'
    : '真实 H²P：注意力 DP2+TP4 · 路由 TP2+EP4 · 共享 TP8';
  const approxNote = training
    ? (PP !== realPP ? `PP 5→${PP}（整除 ${N.toLocaleString()} 卡）· DP 填充至 ${DP}` : `DP 填充至 ${DP} 铺满 ${N.toLocaleString()} 卡`)
    : `节点=1 H²P 服务实例（8 卡按子层重划分）· 跨节点 DP=${DP} 副本`;
  return { workload, N, training, tp: TP, sp: SP, ep: EP, pp: PP, dp: DP, epScope, cfg, real, approxNote, groupOf, groupCount, peersOf, collectiveOf };
}

// ─── ONE canonical colour per entity, shared by ALL three views (top / layered /
// 3-D) so they correspond: same concept → same colour → same glyph language.
// Hardware accent = teal (the die/device domain); software accent = indigo (rank).
// The two are deliberately different hues so hardware and software never blur.
export const ENTITY_COLORS = {
  global:     '#9d7bff',               // L7 Global — violet（DCN 南北向）
  cluster:    '#ffaa3b',               // L6 Cluster — orange（scale-out）
  pool:       '#c084fc',               // L5 Service Pool — light violet
  super:      UB_LEVELS[3].color,      // L4 Pod · 超节点 — rose（避开状态黄）
  cab:        UB_LEVELS[2].color,      // 机柜（并入 L4）— purple
  node:       UB_LEVELS[1].color,      // L3 Host · 节点 — sky blue
  card:       UB_LEVELS[0].color,      // L2 Chip · NPU device — teal (compute-die domain)
  computeDie: UB_LEVELS[0].color,      // 计算 Die — teal
  ioDie:      '#7c8db8',               // IO Die — accent grey
  cube:       COMM_PATTERNS[2].color,  // AI Core · Cube(AIC) — cyan
  vector:     '#7dd3fc',               // AI Core · Vector(AIV) — light cyan
  rank:       '#4369ef',               // software rank — indigo (distinct from all hardware)
  hw:         UB_LEVELS[0].color,      // generic HARDWARE accent — teal
  sw:         '#4369ef',               // generic SOFTWARE accent — indigo
} as const;

// ─── hw-native-sys L0–L7 坐标对齐表（每个内部 key → 层级编号 + 软件落点）──────────
// L 编号唯一来自 HW_LEVELS（L7 Global → L0 Core-Group）。机柜并入 L4 Pod、Tile 归入
// L0 Core-Group 内部，均无独立 L 级；L1 Die 为可选级（单 die 芯片可省略）。
export interface UbCoordLevel { L: string; scope: string; sw: string; obs: string; note?: string; }
export const UB_COORD: Record<string, UbCoordLevel> = {
  job:     { L: 'L7', scope: '全球域',  sw: 'Global 调度 · 跨地域多集群（经 DCN）', obs: '端到端吞吐 · MFU' },
  cluster: { L: 'L6', scope: '集群域',  sw: 'Cluster · 跨 Pool/Pod 的 DP / PP（Scale-Out）', obs: '集群通信占比 (DP)' },
  pool:    { L: 'L5', scope: '服务池域', sw: 'Service Pool · 业务/资源分区（Pool 内互联）', obs: 'Pool 内 Pod 间带宽 · 任务排布' },
  super:   { L: 'L4', scope: 'Pod 域',  sw: 'Pod（UBL128 Scale-Up）· TP/EP/SP 域(SU)', obs: 'UB 带宽利用 · EP All-to-All' },
  cab:     { L: 'L4', scope: 'Pod 域',  sw: '机柜 = Pod 内物理分组（不是层级）', obs: '柜内 mesh 利用 · 上行带宽',
             note: '机柜不是层级 · 仅物理分组，不出现在层级轴' },
  node:    { L: 'L3', scope: 'Host 域', sw: 'Host（1 OS）· 单机多卡放置（1 CPU + 8 NPU 经 PCIe/UB）', obs: '卡间带宽 · host 开销' },
  card:    { L: 'L2', scope: 'Chip 域', sw: 'Chip·NPU = rank 逻辑设备（950 整卡 UMA）', obs: '算力% · HBM% · 负载均衡' },
  die:     { L: 'L1', scope: 'Die 域',  sw: 'Die（可选级）· NoC 互联核组 / 共享 HBM', obs: 'NoC 争用 · D2D · HBM 带宽',
             note: 'L1 Die 为可选层级（单 die 芯片可省略）' },
  core:    { L: 'L0', scope: '核组域',  sw: 'Core-Group（AIV·向量/AIC·Cube/AICPU）· block_idx 核实例（SPMD）', obs: 'AIC/AIV 利用率 · 同步等待' },
  tile:    { L: 'L0', scope: '核组域',  sw: 'Core-Group 内 Tile/lane · GM/L2→UB/L1→L0A/B/C 流水', obs: '流水气泡 · 访存等待',
             note: '归入 L0 · 按 memory-architecture 组织' },
};
// topology-tier (UB 互联层级，数组下标 0–4) → hw-native-sys L0–L7 坐标
export const UB_COORD_TOPO: Record<number, { L: string; scope: string }> = {
  0: { L: 'L0–L2', scope: 'Chip 内（Core-Group / Die / 封装互连）' },
  1: { L: 'L3',    scope: 'Host · PCIe/UB' },
  2: { L: 'L4',    scope: 'Pod 内（机柜并入）' },
  3: { L: 'L4',    scope: 'Pod · Scale-Up（UBL128）' },
  4: { L: 'L5–L6', scope: 'Service Pool / Cluster · Scale-Out' },
};

// ─── Live status / flow overlay (full-pod): node activity + link state ────────
// Node colour = current activity (from the run phase); link thickness = bandwidth
// (intra-node L1 fattest → scale-out L4 thinnest) with a flow surge on the active
// collective. Status colour takes priority over the partition colour.
export const STATUS_COLORS: Record<string, string> = {
  compute: '#04d793', comm: '#ff4b7b', mem: '#a78bfa', load: '#60a5fa', store: '#94a3b8', idle: '#9aa6b8',
};
export const STATUS_META: { id: string; label: string }[] = [
  { id: 'compute', label: '计算中' }, { id: 'comm', label: '通信中' }, { id: 'mem', label: '访存' },
  { id: 'load', label: '加载' }, { id: 'store', label: '存储' }, { id: 'idle', label: '空闲' },
];

// ─── THE iron rule: 红/黄/绿(+灰) = state-only. ───────────────────────────────
// Observation state = 3 active levels + offline. One state = ONE fixed colour (no gradient).
// Thresholds per the design doctrine: 空闲<40% / 中40–70% / 繁忙>70% · 离线/无数据=灰.
// Used identically by ALL views (阵列全景 + 平面顶视图 + 层级图) for lines AND nodes.
const STATE_RGB: [number, number, number][] = [[0x04, 0xd7, 0x93], [0xfb, 0xbf, 0x24], [0xff, 0x4b, 0x7b]];   // 空闲 绿 / 中 黄 / 繁忙 红 (PTO ramp, aligned to reference)
const OFFLINE_RGB: [number, number, number] = [0x8b, 0x93, 0xa3];   // 离线/无数据 — 冷中性灰
export const STATE_LABELS = ['空闲 <40%', '中 40–70%', '繁忙 >70%', '离线/无数据'];
export function loadState(t: number): number { const x = Math.max(0, Math.min(1, t)); return x < 0.4 ? 0 : x < 0.7 ? 1 : 2; }   // 0 绿 / 1 黄 / 2 红
export function loadRGB(t: number): [number, number, number] { return t < 0 ? OFFLINE_RGB : STATE_RGB[loadState(t)]; }   // t<0 → offline
export function stateColor(i: number): string { const c = i < 0 || i >= 3 ? OFFLINE_RGB : STATE_RGB[i]; return `rgb(${c[0]},${c[1]},${c[2]})`; }
export const isHot = (t: number): boolean => t >= 0 && loadState(t) >= 2;   // only 繁忙(红) → colour the node; else neutral
export function loadColor(t: number): string { const [r, g, b] = loadRGB(t); return `rgb(${r},${g},${b})`; }
// STRUCTURE → a single COOL NEUTRAL BLUE-GREY (NO hue, NO RYG). Any hierarchy/type colour fed
// through here recedes to a lightness-preserving blue-grey, so red/yellow/green belong ONLY to
// state. Levels/types are told apart by SHAPE / POSITION / lightness — never hue.
export function mute(hex: string): string {
  const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);
  const L = Math.max(30, Math.min(206, 0.3 * r + 0.59 * g + 0.11 * b));   // input lightness, clamped off pure black/white
  return `rgb(${Math.round(L * 0.82)},${Math.round(L * 0.9)},${Math.round(Math.min(255, L * 1.08))})`;   // cool blue-grey
}
// deliberate cool blue-grey LIGHTNESS ramp for hierarchy — levels differ by 明度 (浅 finest → 深
// coarsest), so legends/levels stay tellable WITHOUT hue. depth 0..1 (0 = finest level).
export function structColor(depth: number): string {
  const t = Math.max(0, Math.min(1, depth)), L = 196 - t * 130;   // 196 light → 66 dark
  return `rgb(${Math.round(L * 0.82)},${Math.round(L * 0.9)},${Math.round(Math.min(255, L * 1.12))})`;
}
// deterministic, stable per-id "load" 0..1, modulated by the current run-phase kind.
// WIDE per-node spread so within a level you get a clear 绿/黄/橙/红 mix.
export function nodeLoad(id: number, phaseKind?: string): number {
  let h = (id * 2654435761) >>> 0; h ^= h >>> 13; h = (h * 1274126177) >>> 0;
  h ^= h >>> 16; const base = (h >>> 8) / 0xffffff;   // 0..1 stable spread
  const lvl = phaseKind === 'compute' ? 0.62 : phaseKind === 'comm' ? 0.5 : phaseKind === 'mem' ? 0.56 : phaseKind === 'load' || phaseKind === 'store' ? 0.48 : 0.34;
  return Math.max(0, Math.min(1, lvl + (base - 0.5) * 0.95));   // ±0.475 spread → spans green→red
}

// Physics-grounded per-card role bias derived from Pangu Pro MoE parallel config (arXiv:2505.21411).
// k = global card rank; phaseKind = 'compute'|'comm'|'mem'; N = total NPUs in the supernode.
// Returns a bias [-0.12, +0.15] that shifts nodeLoad to reflect real role-level utilization patterns.
export function parallelRoleBias(k: number, phaseKind: string, N = 8192): number {
  const tpRank = k % NPUS_PER_NODE;  // 0–7 within a host (TP8 training, TP4/TP8 inference)
  // Approximate PP stage: PP5 → divide supernode into 5 equal bands
  const ppStage = Math.min(WORKLOAD.train.pp - 1, Math.floor((k / Math.max(1, N)) * WORKLOAD.train.pp));
  const isEdgePP = ppStage === 0 || ppStage === WORKLOAD.train.pp - 1;
  // EP group for decode: groups of cards share an expert partition
  const epGrpSize = Math.max(1, Math.floor(N / (WORKLOAD.inferRouted.ep * 4)));
  const epGroup = Math.floor(k / epGrpSize);

  if (phaseKind === 'compute') {
    // Prefill: TP-heavy GEMMs dominate. PP bubble at boundary stages reduces util ~8%.
    return isEdgePP ? -0.08 : 0.03;
  }
  if (phaseKind === 'comm') {
    // Decode: EP All-to-All dominant. Even with MoGE IS=0 cutting imbalance >50%,
    // popular expert groups still run hotter. Shared experts (TP8) handle ALL tokens.
    const epHot = _rnd01(epGroup * 7.3) > 0.58;          // ~42% of groups are hot
    const isShared = tpRank >= NPUS_PER_NODE - 2;         // last 2 TP ranks = shared experts
    return (epHot ? 0.09 : -0.04) + (isShared ? 0.07 : 0);
  }
  // Pretrain: DP Ring-AllReduce + EP A2A = 30% of step. PP bubble at edges reduces compute.
  return isEdgePP ? -0.06 : 0.02;
}

// ─── SHARED live load / straggler / fault / replay-event model ────────────────
// ONE model, so the SAME card reads the SAME value in 运行状态 AND 工作台 (was two divergent
// copies: util01/faultAt in StatusView vs cardLoad/isFault in ConsoleView). k = global card
// index inside a super-node (node = ⌊k/8⌋, cabinet = ⌊k/64⌋); pod = super-node index.
export const REPLAY = { stepMax: 60, evtLo: 34, evtHi: 46, evtCab: 1, cardsPerCab: 64 } as const;
const _rnd01 = (s: number) => { const x = Math.sin(s * 99.13) * 43758.5453; return x - Math.floor(x); };
const _clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);
export function inReplayEvent(step: number): boolean { return step >= REPLAY.evtLo && step <= REPLAY.evtHi; }
const _evtEnv = (step: number) => Math.sin(((step - REPLAY.evtLo) / (REPLAY.evtHi - REPLAY.evtLo)) * Math.PI);   // 0→1→0 bump
export function cardStraggler(k: number, step: number, pod = 0): boolean {
  const cab = Math.floor(k / REPLAY.cardsPerCab);
  const thr = inReplayEvent(step) && pod === 0 && cab === REPLAY.evtCab ? 0.55 : 0.985;   // event柜掉队概率飙升
  return _rnd01(pod * 131 + k * 1.7 + step * 0.05) > thr;
}
export function cardFault(k: number, step: number, pod = 0): boolean {
  const cab = Math.floor(k / REPLAY.cardsPerCab), nodeInCab = Math.floor(k / 8) % 8;
  const inEvt = inReplayEvent(step) && pod === 0 && cab === REPLAY.evtCab && nodeInCab === 1;
  return inEvt ? _rnd01(k * 0.7) > 0.25 : _rnd01(pod * 5 + k * 2.3 + step) > 0.9994;
}
export function cardLoad01(k: number, phaseKind: string, step: number, pod = 0, N = 8192): number {
  const cab = Math.floor(k / REPLAY.cardsPerCab);
  let u = nodeLoad(k + pod * 100003, phaseKind);
  u += parallelRoleBias(k, phaseKind, N);                    // physics-grounded role bias (PP bubble, EP hot experts)
  u += (_rnd01(pod * 17 + cab * 2.7 + 1) - 0.5) * 0.30;   // per-cabinet hot/cold bias (spatial spread)
  u += (_rnd01(k * 0.91 + step * 0.07) - 0.5) * 0.08;      // live per-step ripple
  if (cardStraggler(k, step, pod)) u += 0.4;               // straggler runs hot
  if (inReplayEvent(step) && pod === 0 && cab === REPLAY.evtCab) u += 0.35 * _evtEnv(step);   // injected 过热事件
  return _clamp01(u);
}
export type CardMetric = 'util' | 'strag' | 'fault';
export function cardMetric01(k: number, metric: CardMetric, phaseKind: string, step: number, pod = 0): number {
  if (metric === 'fault') return cardFault(k, step, pod) ? 0.95 : 0.1;
  if (metric === 'strag') return cardStraggler(k, step, pod) ? 0.88 : Math.max(0, cardLoad01(k, phaseKind, step, pod) - 0.5) * 0.4;
  return cardLoad01(k, phaseKind, step, pod);
}

export const RUN_SCHED: Record<RunMode, RunPhase[]> = {
  train: [
    { id: 'load', name: '加载 batch',      kind: 'load',    color: '#c2c9d4', parallel: 'DP',    note: '各 DP 副本读入各自 micro-batch' },
    { id: 'fwd',  name: '前向 Forward',    kind: 'compute', color: '#22d3ee', parallel: 'TP·PP', note: 'TP 层内并行 + PP 流水级逐级前向' },
    { id: 'bwd',  name: '反向 Backward',   kind: 'compute', color: '#0ea5e9', parallel: 'TP·PP', note: '反向传播逐级回传，产生梯度' },
    { id: 'ar',   name: '梯度 AllReduce',  kind: 'comm',    color: '#ff4b7b', collective: 'ring', parallel: 'DP', note: 'DP 副本间环状 AllReduce 同步梯度' },
    { id: 'opt',  name: '优化器更新',      kind: 'store',   color: '#aab4c4', parallel: '—',     note: '更新参数 / 写回（含 store）' },
  ],
  infer: [
    { id: 'pre', name: 'Prefill 预填充',     kind: 'compute', color: '#22d3ee', parallel: 'TP', note: '提示词整段并行前向，KV-Cache 建立' },
    { id: 'a2a', name: 'MoE All-to-All',     kind: 'comm',    color: '#f59e0b', collective: 'a2a', parallel: 'EP', note: '专家并行 token 分发 All-to-All' },
    { id: 'dec', name: 'Decode 解码(逐token)', kind: 'compute', color: '#34d399', parallel: 'TP', note: '自回归逐 token 前向，吞吐 = tok/s' },
    { id: 'kv',  name: 'KV-Cache 读写',      kind: 'mem',     color: '#a78bfa', parallel: '—', note: '每步读写 KV-Cache（显存带宽受限）' },
  ],
};

// ─── Per-card memory hierarchy (single NPU) — illustrative occupancy ──────────
// Bridges the cluster topology down to the on-chip story PTO focuses on: where a
// rank's bytes live and where the bottleneck usually is. `util` is a schematic
// fill ratio (NOT a measured profile), drawn in the PTO 14%-fill / 34%-stroke style.
export interface MemLayer { id: string; name: string; cap: string; util: number; color: string; note: string; }
export function memLayers(gen: GenSpec): MemLayer[] {
  return [
    { id: 'hbm', name: `HBM（${gen.hbm}）`, cap: `${gen.memGB} GB`, util: 0.78, color: '#4369ef', note: '权重 + 激活 + KV-Cache，带宽常为瓶颈' },
    { id: 'l2',  name: 'L2 全局缓存',        cap: gen.l2MB ? `${gen.l2MB} MB` : '—', util: 0.62, color: '#7c8db8', note: 'die 内共享，算子间数据复用' },
    { id: 'ub',  name: 'UB Memory',          cap: '512 KB', util: 0.5, color: '#04d793', note: '统一编址，跨 NPU 池化访问' },
    { id: 'l1',  name: 'L1（片上 SRAM）',    cap: '512 KB', util: 0.7, color: '#ffaa3b', note: 'Tile 驻留，搬运 HBM→L1→L0' },
    { id: 'l0',  name: 'L0A/B/C',            cap: '64–256 KB', util: 0.85, color: '#ff4b7b', note: 'Cube/Vector 计算缓冲，最贴近算力' },
  ];
}

// ─── On-chip memory ROUTE overlay (HBM→L2→L1→L0→ALU) per workload ──────────────
// Inspired by the reference PTO memory-architecture / route-overlay pattern (910B/950 内存架构图 +
// 访问路由叠加). Spatialises the memory hierarchy at die/AI-Core/Tile level and animates WHERE the
// bytes flow, grounded in REAL Pangu Pro MoE kernel data (arXiv:2505.21411):
//   · Decode  访存受限：权重搬运占时延 29% · KV 占注意力计算 70%（注意力占端到端 30–50%）· MulAttention 4.5×
//   · Prefill 计算受限：SwiftGMM 占端到端 >50%（MTE2 利用率 up to 95%，近权重带宽上限）
//   · 预训练  FWD/BWD 计算 + 梯度回写（访存与计算交织）
// ordered top(最贴近算力 ALU)→bottom(HBM) so the view reads 近算力在上、HBM 在下.
export const MEM_STACK: { id: string; label: string; kind: 'alu' | 'buf' | 'sram' | 'ub' | 'cache' | 'hbm' }[] = [
  { id: 'alu', label: 'Cube / Vector ALU', kind: 'alu' },
  { id: 'l0',  label: 'L0A / L0B / L0C',   kind: 'buf' },
  { id: 'l1',  label: 'L1 片上 SRAM',       kind: 'sram' },
  { id: 'ub',  label: 'UB Memory（池化）',   kind: 'ub' },
  { id: 'l2',  label: 'L2 全局缓存',         kind: 'cache' },
  { id: 'hbm', label: 'HBM',               kind: 'hbm' },
];
export interface MemHop { from: string; to: string; label: string; intensity: number; }   // from/to = MEM_STACK id · intensity 0..1 → 状态色+线宽
export const MEM_ROUTE: Record<'pretrain' | 'prefill' | 'decode', { note: string; bottleneck: string; hops: MemHop[] }> = {
  decode: {
    note: 'Decode 访存受限：权重搬运占时延 29% · KV 占注意力 70%',
    bottleneck: 'hbm',
    hops: [
      { from: 'hbm', to: 'l0', label: '权重 29%', intensity: 0.92 },
      { from: 'hbm', to: 'l1', label: 'KV 搬运', intensity: 0.82 },
      { from: 'l1', to: 'alu', label: 'MulAttn 4.5×', intensity: 0.6 },
      { from: 'l0', to: 'alu', label: '', intensity: 0.5 },
    ],
  },
  prefill: {
    note: 'Prefill 计算受限：SwiftGMM 占端到端 >50%（MTE2 up to 95%）',
    bottleneck: 'alu',
    hops: [
      { from: 'hbm', to: 'l2', label: '权重/激活', intensity: 0.5 },
      { from: 'l2', to: 'l1', label: '', intensity: 0.62 },
      { from: 'l1', to: 'l0', label: 'GEMM', intensity: 0.8 },
      { from: 'l0', to: 'alu', label: 'SwiftGMM >50%', intensity: 0.92 },
    ],
  },
  pretrain: {
    note: '预训练：FWD/BWD 计算 + 梯度回写（访存↔计算交织）',
    bottleneck: 'l1',
    hops: [
      { from: 'hbm', to: 'l2', label: '激活', intensity: 0.55 },
      { from: 'l2', to: 'l1', label: '', intensity: 0.62 },
      { from: 'l1', to: 'l0', label: 'GEMM', intensity: 0.8 },
      { from: 'l0', to: 'alu', label: 'FWD/BWD', intensity: 0.75 },
      { from: 'alu', to: 'hbm', label: '梯度回写', intensity: 0.5 },
    ],
  },
};

export const RACK_COLORS = {
  accent: '#e0252f',
  computeGlow: '#38bdf8',
  switchGlow: '#fb923c',
} as const;

// ─── Overview hall: compute-cabinet grid + communication-cabinet spine ────────
export interface CabinetCell {
  id: string;
  kind: RackKind;
  pos: [number, number, number];
}

const HALL_COLS = 16;
export const CAB_W = 0.34, CAB_H = 1.3, CAB_D = 0.68;
const CAB_GAP_X = 0.12, CAB_GAP_Z = 0.5, BLOCK_GAP_Z = 1.0;

/** Build a schematic data-hall floor for a generation:
 *  compute cabinets in a front grid, communication cabinets as a rear block. */
export function buildHall(gen: GenSpec): CabinetCell[] {
  const cells: CabinetCell[] = [];
  const pitchX = CAB_W + CAB_GAP_X;
  const pitchZ = CAB_D + CAB_GAP_Z;
  const rowW = HALL_COLS * pitchX;
  const x0 = -rowW / 2 + CAB_W / 2;

  const computeRows = Math.ceil(gen.computeCabs / HALL_COLS);
  let z = 0;
  // compute block (front, toward +Z growing away)
  for (let i = 0; i < gen.computeCabs; i++) {
    const r = Math.floor(i / HALL_COLS);
    const c = i % HALL_COLS;
    cells.push({ id: `c-${i}`, kind: 'compute', pos: [x0 + c * pitchX, 0, r * pitchZ] });
    z = r * pitchZ;
  }
  // communication block (rear, separated by an aisle)
  const commZ0 = z + pitchZ + BLOCK_GAP_Z;
  for (let i = 0; i < gen.commCabs; i++) {
    const r = Math.floor(i / HALL_COLS);
    const c = i % HALL_COLS;
    cells.push({ id: `s-${i}`, kind: 'switch', pos: [x0 + c * pitchX, 0, commZ0 + r * pitchZ] });
  }
  // centre the whole hall on the Z origin
  const zs = cells.map((c) => c.pos[2]);
  const zMid = (Math.min(...zs) + Math.max(...zs)) / 2;
  for (const cell of cells) cell.pos[2] -= zMid;
  void computeRows;
  return cells;
}

// ─── Cabinet internals (representative; metres, schematic slots) ──────────────
export const RACK_DIM = { w: 0.6, h: 2.25, d: 1.15 };

export interface RackUnit {
  id: string;
  type: 'node' | 'switch-unit' | 'power' | 'mgmt' | 'cdu';
  label: string;
  labelEn: string;
  y0: number;      // unit bottom (0..1 of rack height)
  hFrac: number;   // unit height fraction (0..1)
  nodeSlot?: number;
}

export const COMPUTE_RACK_UNITS: RackUnit[] = (() => {
  const u: RackUnit[] = [];
  u.push({ id: 'power', type: 'power', label: '集中供电 Busbar / 电源框', labelEn: 'Power / Busbar', y0: 0.93, hFrac: 0.05 });
  u.push({ id: 'mgmt',  type: 'mgmt',  label: '柜管模块 + GE 管理交换',   labelEn: 'Mgmt + GE',      y0: 0.882, hFrac: 0.038 });
  const top = 0.86, bottom = 0.075, gap = 0.004;
  const step = (top - bottom) / NODES_PER_CAB;
  for (let i = 0; i < NODES_PER_CAB; i++) {
    u.push({
      id: `node-${i}`, type: 'node',
      label: `计算节点 ${i + 1}（液冷刀片 · ${NPUS_PER_NODE}× NPU）`,
      labelEn: `Node ${i + 1}`,
      y0: bottom + (NODES_PER_CAB - 1 - i) * step + gap / 2,
      hFrac: step - gap, nodeSlot: i,
    });
  }
  u.push({ id: 'cdu', type: 'cdu', label: 'Manifold 液冷分集水器 / 快接头', labelEn: 'Liquid Manifold', y0: 0.012, hFrac: 0.055 });
  return u;
})();

export const SWITCH_UNIT_COUNT = 6;
export const SWITCH_RACK_UNITS: RackUnit[] = (() => {
  const u: RackUnit[] = [];
  u.push({ id: 'power', type: 'power', label: '电源管理 · 集中供电', labelEn: 'Power Shelf', y0: 0.92, hFrac: 0.06 });
  for (let i = 0; i < SWITCH_UNIT_COUNT; i++) {
    u.push({
      id: `sw-${i}`, type: 'switch-unit',
      label: `${TOK.ub} 交换设备 ${i + 1}（UB Clos 顶层 · 全光）`,
      labelEn: `UB Switch ${i + 1}`,
      y0: 0.10 + (SWITCH_UNIT_COUNT - 1 - i) * 0.128, hFrac: 0.108,
    });
  }
  u.push({ id: 'mgmt', type: 'mgmt', label: '管理 / 全光配线区', labelEn: 'Mgmt / Optical Patch', y0: 0.015, hFrac: 0.07 });
  return u;
})();

// ─── Compute-node internals (abstract blade layout, metres) ──────────────────
export const NODE_DIM = { w: 0.86, h: 0.12, d: 0.72 };

export interface NodePart {
  id: string;
  type: 'npu' | 'cpu' | 'ub-fabric' | 'dpu' | 'optical' | 'dimm';
  label: string;
  pos: [number, number, number];
  size: [number, number, number];
  /** NPU index 0..7 (for overlay wiring) */
  npuIdx?: number;
}

// 8 NPUs in a 2×4 grid — these positions drive the die / process / thread overlays.
export const NPU_GRID = { cols: 4, rows: 2, pitchX: 0.18, pitchZ: 0.17, z0: -0.16 };

export const NODE_PARTS: NodePart[] = (() => {
  const parts: NodePart[] = [];
  for (let i = 0; i < NPUS_PER_NODE; i++) {
    const c = i % NPU_GRID.cols, r = Math.floor(i / NPU_GRID.cols);
    const cx = (c - (NPU_GRID.cols - 1) / 2) * NPU_GRID.pitchX;
    const cz = NPU_GRID.z0 + r * NPU_GRID.pitchZ;
    parts.push({
      id: `npu-${i}`, type: 'npu', npuIdx: i,
      label: `${TOK.ascend} ${TOK.n950dt} #${i + 1}（1 device）· ${DIES_PER_NPU} Die = 2 计算(UMA)+2 IO · UB 2 TB/s · ${TOK.hbmZQ} HBM`,
      pos: [cx, 0.022, cz], size: [0.12, 0.024, 0.115],
    });
  }
  // 4 CPUs (front row)
  for (let i = 0; i < CPUS_PER_NODE; i++) {
    parts.push({
      id: `cpu-${i}`, type: 'cpu',
      label: `${TOK.kunpeng} ${TOK.n950} #${i + 1} · UB 全池化 · NUMA`,
      pos: [(i - 1.5) * 0.18, 0.018, 0.2], size: [0.085, 0.016, 0.085],
    });
  }
  // 2 DIMM banks
  for (let i = 0; i < 2; i++) {
    parts.push({
      id: `dimm-${i}`, type: 'dimm', label: 'DDR5 内存区（统一编址池化）',
      pos: [0, 0.014, 0.3 + i * 0.04], size: [0.72, 0.018, 0.026],
    });
  }
  // central UB fabric chips (L1 node-internal 2D-mesh switching)
  for (let i = 0; i < 2; i++) {
    parts.push({
      id: `ubf-${i}`, type: 'ub-fabric', label: `${TOK.ub} L1 板载 UB 2D-Mesh 交换 fabric`,
      pos: [(i - 0.5) * 0.22, 0.016, 0.02], size: [0.075, 0.014, 0.06],
    });
  }
  // DPU (VPC egress)
  parts.push({ id: 'dpu', type: 'dpu', label: `${TOK.qingtian} · VPC 外网`, pos: [0.36, 0.02, 0.24], size: [0.085, 0.02, 0.16] });
  // rear optical panel (UB uplink to comms cabinets, L3)
  parts.push({ id: 'optical', type: 'optical', label: '光口区 · UB 上行至通信柜（L3 Clos）+ RoCE scale-out', pos: [-0.02, 0.02, -0.34], size: [0.7, 0.026, 0.018] });
  return parts;
})();

export function npuPositions(): [number, number, number][] {
  return NODE_PARTS.filter((p) => p.type === 'npu').map((p) => p.pos);
}

// ─── Small-pod scales (16P / 32P / 64P) + recursive full-mesh adjacency ───────
// Recursive direct-connect full-mesh: 8 NPU/board form a 1D full mesh, boards
// form the next dimension, etc. (single 64-card cabinet = 8×8).
export type Scale = '16P' | '32P' | '32Pi' | '64P';
export interface ScaleSpec {
  id: Scale; label: string; npus: number; dims: number[];
  kind?: 'mesh' | 'switched';   // switched = single-hop fully-switched fabric
  paths?: number;               // parallel switch paths between any two NPUs
  uboe?: [number, number];      // external ethernet uplink ports per NPU (min, max)
}
export const SCALES: Record<Scale, ScaleSpec> = {
  '16P':  { id: '16P',  label: '16P 小超节点',     npus: 16, dims: [8, 2] },
  '32P':  { id: '32P',  label: '32P 小超节点',     npus: 32, dims: [8, 4] },
  '32Pi': { id: '32Pi', label: '32P 一体(单跳)',   npus: 32, dims: [32], kind: 'switched', paths: 6, uboe: [1, 2] },
  '64P':  { id: '64P',  label: '64P 单柜',         npus: 64, dims: [8, 8] },
};
export const DEFAULT_SCALE: Scale = '64P';

/** dim index → UB hierarchy level index (dim0=板内→L1, dim1=跨板→L2, dim2=跨柜→L3). */
export const dimToLevel = (d: number): number => Math.min(d + 1, UB_LEVELS.length - 1);

export interface AdjCell { level: number; direct: boolean; hops: number; paths?: number; }

/** 32P-integrated single-hop fully-switched fabric: any two NPUs reachable in one
 *  hop via the switch, with `paths` parallel switch paths between every pair. */
export function makeSwitchedAdjacency(n: number, paths: number): { n: number; cell: (i: number, j: number) => AdjCell } {
  return {
    n,
    cell: (i, j) => (i === j ? { level: -1, direct: false, hops: 0 } : { level: 3, direct: true, hops: 1, paths }),
  };
}

/** Recursive full-mesh adjacency for `dims`: two NPUs are directly UB-connected iff they
 *  differ in exactly one dimension; otherwise multi-hop. Cell colour = the
 *  (highest) differing dimension's UB level. */
export function makeAdjacency(dims: number[]): { n: number; cell: (i: number, j: number) => AdjCell } {
  const n = dims.reduce((a, b) => a * b, 1);
  const coords = (idx: number) => {
    const c: number[] = [];
    for (const d of dims) { c.push(idx % d); idx = Math.floor(idx / d); }
    return c;
  };
  const cell = (i: number, j: number): AdjCell => {
    if (i === j) return { level: -1, direct: false, hops: 0 };
    const ci = coords(i), cj = coords(j);
    const diff: number[] = [];
    for (let d = 0; d < dims.length; d++) if (ci[d] !== cj[d]) diff.push(d);
    return { level: dimToLevel(diff[diff.length - 1]), direct: diff.length === 1, hops: diff.length };
  };
  return { n, cell };
}

