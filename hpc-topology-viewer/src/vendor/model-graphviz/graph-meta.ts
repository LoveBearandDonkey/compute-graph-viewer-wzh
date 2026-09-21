// ─────────────────────────────────────────────────────────────────────────────
// model-graphviz 整网图 → 立方重组/整网切分 的桥接元数据。
// 把参考页抽取出来的「已定位」计算图（openpangu-graph.json：Model → DecoderLayer →
// Attention/MoE → QKV/Expert → Operator 五级 DAG）包一层类型，并给每个算子标注
// 「它被哪种并行切分」——这正是整网图与立方（按 TP/PP/DP/EP 重排/染色）联动的关键：
// 点图上的算子 → 立方按该维度染色显形；反选切分维 → 图上高亮属于该维的算子。
// ─────────────────────────────────────────────────────────────────────────────
import rawGraph from './openpangu-graph.json';

export type ParCutDim = 'tp' | 'pp' | 'dp' | 'ep';

export interface GraphNode {
  id: string;
  label: string;
  kind: string;                 // 'module' | 'op' | 'tensor'
  typeLabel?: string;
  x: number; y: number; width: number; height: number;
  colorKey: string;             // 'opv:attention' | 'opv:moe' | 'opv:comm' | 'io:parameter' …
  parent?: string;              // 所属 cluster id
  glyph?: boolean;
  collapsed?: boolean;
}
export interface GraphEdge {
  source: string; target: string; dashed?: boolean;
  sourceAnchor?: unknown; targetAnchor?: unknown; curve?: string;
}
export interface GraphCluster {
  id: string; label: string; colorKey: string;
  parentCluster?: string | null;
  x: number; y: number; width: number; height: number;
}
export interface ModelGraph {
  width: number; height: number;
  nodes: GraphNode[]; edges: GraphEdge[]; clusters: GraphCluster[];
}

export const OPENPANGU_GRAPH = rawGraph as unknown as ModelGraph;

// 算子 → 并行切分维（哪种并行把这个算子切开 / 它落在哪种通信域）。
//   · Attention 块（q/kv/o 投影、FlashAttention、AllGather/Reduce-Scatter）→ TP（张量并行切注意力头）
//   · Dense FFN（gate_up/down/silu、其 TP 通信）→ TP
//   · MoE 块（router/topk/专家bank/combine、All-to-All、专家并行态/权重）→ EP（专家并行）
//   · 逐层结构（DecoderLayer、层级 Norm、embedding/head/logits 的流水级边界）→ PP（流水并行）
//   · 权重/参数张量（DP Ring-AllReduce 同步梯度的对象）→ DP（数据并行）
function deriveDim(n: GraphNode): ParCutDim | null {
  const id = n.id;
  if (/^moe_all_to_all/.test(id)) return 'ep';
  if (/^attention_(all_gather|reduce_scatter)/.test(id)) return 'tp';
  if (/^ffn_(all_gather|reduce_scatter)/.test(id)) return 'tp';
  if (id === 'expert_parallel_state' || id === 'expert_bank_weights') return 'ep';
  if (n.colorKey === 'io:parameter') return 'dp';   // 权重 → DP 梯度同步的对象
  switch (n.parent) {
    case 'attention-block': return 'tp';
    case 'moe-block':       return 'ep';
    case 'ffn-block':       return 'tp';
    case 'decoder-stack':   return 'pp';
  }
  return 'pp';   // model-core 级（输入/embedding/final-norm/head/logits）= 流水级边界
}

/** 算子 id → 并行切分维（null = 不被任一维单独切分，如纯逐元素） */
export const NODE_DIM: Record<string, ParCutDim | null> = Object.fromEntries(
  OPENPANGU_GRAPH.nodes.map((n) => [n.id, deriveDim(n)]),
);

/** 并行切分维 → 属于该维的算子 id 集合（反查：选中切分维 → 图上高亮这些算子） */
export const DIM_NODES: Record<ParCutDim, string[]> = { tp: [], pp: [], dp: [], ep: [] };
OPENPANGU_GRAPH.nodes.forEach((n) => {
  const d = NODE_DIM[n.id];
  if (d) DIM_NODES[d].push(n.id);
});

/** colorKey → 中文类别（图例用） */
export const CATEGORY_LABEL: Record<string, string> = {
  'io:input': '输入', 'io:output': '输出', 'io:state': '运行态', 'io:parameter': '权重',
  'opv:embedding': '嵌入', 'opv:decoder': '解码层', 'opv:attention': '注意力',
  'opv:norm': '归一化', 'opv:linear': '线性', 'opv:act': '激活/残差',
  'opv:mlp': 'MLP', 'opv:gate': '路由', 'opv:moe': '专家', 'opv:head': '输出头',
  'opv:comm': '集合通信',
};
