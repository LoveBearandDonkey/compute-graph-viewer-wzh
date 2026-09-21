/**
 * ingest — P0 监控数据摄入契约（加法，不替换）。
 *
 * 首要任务是运维监控 → app 从「自包含合成」转向「数据驱动」。本模块只做两件事：
 *   ① 定义外部集群/作业喂入的 schema：作业并行配置 · rank→物理放置 · 逐 rank 时序遥测。
 *   ② 把现有合成模型（cardLoad01/cardStraggler/cardFault）包装成「一个数据源」——
 *      于是真实遥测只是换一个 provider，现有合成路径原样保留为默认/演示。零冲击现有代码。
 *
 * 真正的实时/流式接入（采集、时延、降采样）在后续阶段；此处先把契约与可互换性立起来。
 */
import {
  cardLoad01, cardStraggler, cardFault,
  type ParallelWorkload, type ParallelConfig,
} from './data';

// ① 作业并行配置：外部作业声明它「怎么部署的」（喂给 parallelMap / deploymentOf 的 cfg）。
export interface PhysPlacement { pod: number; cabinet: number; host: number; slot: number; }
export interface JobConfig {
  workload: ParallelWorkload;
  N: number;
  parallel?: ParallelConfig;                                   // 真实并行度（缺省 → parallelMap 自动推导）
  framework?: 'mindspeed' | 'megatron' | 'pytorch' | 'unknown';
  placement?: Record<number, PhysPlacement>;                   // 真实 rank→放置（缺省 → 默认平铺推导）
}

// ③ 逐 rank 时序遥测：某 rank 在某时刻的观测。util<0 约定为「无数据/离线」（复用 loadColor 的灰）。
export interface RankSample {
  rank: number;
  t: number;
  util: number;        // 0..1 利用率（<0 = 无数据）
  straggler: boolean;  // 是否掉队
  fault: boolean;      // 是否故障
  commBytes?: number;  // 通信字节（真实遥测可选携带）
}

// 数据源接口：正是「监控上色」的唯一入口。合成与真实实现同一接口 → 可互换。
export interface TelemetryProvider {
  readonly source: 'synthetic' | 'live';
  at: (rank: number, t: number) => RankSample;
  meta: () => { label: string; note: string };
}

// 合成数据源：包装现有 cardLoad01/cardStraggler/cardFault，让当前模型成为「数据源之一」。
// phaseKind: 'compute' | 'comm'（复用现有工况语义）· t 即回放 step。
export function syntheticProvider(N: number, phaseKind: string): TelemetryProvider {
  return {
    source: 'synthetic',
    at: (rank, t) => ({
      rank, t,
      util: cardLoad01(rank, phaseKind, t, 0, N),
      straggler: cardStraggler(rank, t),
      fault: cardFault(rank, t),
    }),
    meta: () => ({ label: '合成（演示）', note: '来自 cardLoad01/cardStraggler/cardFault · 结构真、数值仿真' }),
  };
}

// 真实数据源（表驱动）：用摄入的样本表回答查询，展示 live 契约的形状——真正的实时接入在后续阶段。
// 命中缺失时返回 util<0（→ 离线灰），与现有 loadColor 的「无数据」语义一致。
export function tableProvider(samples: RankSample[]): TelemetryProvider {
  const key = (rank: number, t: number) => `${rank}@${t}`;
  const map = new Map<string, RankSample>();
  for (const s of samples) map.set(key(s.rank, s.t), s);
  return {
    source: 'live',
    at: (rank, t) => map.get(key(rank, t)) ?? { rank, t, util: -1, straggler: false, fault: false },
    meta: () => ({ label: '实时（摄入）', note: `${samples.length} 条样本 · util<0 表示该(rank,t)无数据` }),
  };
}
