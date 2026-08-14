/*
  融合算子 workspace 候选构建器 —— 场景 6
  ------------------------------------------------------------------
  方案设计：场景6-workspace与GM规划-方案设计.md §7

  和 data/runs.js 一样，这里是中间格式的**生成器**而不是手写样例：
  张量大小由 shape 推出、生命周期由「谁产出 / 谁最后消费」推出，
  各候选之间的差异真实来自结构差异（原地 / 留片上 / 手工复用 / 动态 shape），
  三个数（current / packed / lowerBound）全部由 js/workspace-planner.js 现算。

  每组候选只承载一个需要被看见的问题，不把所有毛病堆进一次运行。

  数据等级：L2 / schema-generated / share-safe。MLABlock_fused 是为演示构造的
  融合算子，形状与预算均为占位值，不对应任何真实产品算子。
*/
(function registerMemVizFusionRuns(global) {
  'use strict';

  const SRC = global.MemVizFusionSource;
  const PLANNER = global.MemVizWorkspacePlanner;

  const KB = 1024;
  const GM_ALIGN = 512;
  const BLOCK_DIM = 8;
  const BUDGET = 16 * KB * KB;      // 上游给这个算子的 GM 预算：16MB

  const BASE_TOKENS = 1024;
  const HIDDEN = 1024;
  const FFN_INNER = 2816;

  function alignUp(value, align) {
    return Math.ceil(value / align) * align;
  }

  // ---------------------------------------------------------------
  // 子计算序列（拓扑序）
  // ---------------------------------------------------------------
  const SUBGRAPHS = [
    { id: 'sg0', name: 'QKVProj', kind: 'matmul', lane: 'aic', desc: 'x → Q / K / V 投影', fn: 'void QKVProj()' },
    { id: 'sg1', name: 'RoPE', kind: 'elementwise', lane: 'aiv', desc: '旋转位置编码，逐元素', fn: 'void RoPE()' },
    { id: 'sg2', name: 'FlashAttn', kind: 'attention', lane: 'aic', desc: '分块 attention，附带 logsumexp 统计量', fn: 'void FlashAttn()' },
    { id: 'sg3', name: 'OutProj', kind: 'matmul', lane: 'aic', desc: '输出投影', fn: 'void OutProj()' },
    { id: 'sg4', name: 'Add+LN', kind: 'normalize', lane: 'aiv', desc: '残差相加与 LayerNorm', fn: 'void AddLayerNorm()' },
    { id: 'sg5', name: 'FFN', kind: 'ffn', lane: 'aic', desc: '升维 → SwiGLU → 降维写回', fn: 'void FeedForward()' },
  ];

  const SG_INDEX = Object.fromEntries(SUBGRAPHS.map((sg, i) => [sg.id, i]));

  // ---------------------------------------------------------------
  // 张量声明 —— 大小由 shape 推出，生命周期由产消关系推出
  // ---------------------------------------------------------------
  function declarations(tokens) {
    const hidden = tokens * HIDDEN * 2;      // half
    const lse = tokens * 4;                  // float32 每 token 一个统计量
    const ffn = tokens * FFN_INNER * 2;      // half

    return [
      { key: 'wsQ', bytes: hidden, dtype: 'float16', shape: [tokens, HIDDEN], producer: 'sg0', consumers: ['sg1'], note: 'Query 投影结果，只被 RoPE 读一次' },
      { key: 'wsK', bytes: hidden, dtype: 'float16', shape: [tokens, HIDDEN], producer: 'sg0', consumers: ['sg1'], note: 'Key 投影结果，只被 RoPE 读一次' },
      { key: 'wsV', bytes: hidden, dtype: 'float16', shape: [tokens, HIDDEN], producer: 'sg0', consumers: ['sg2'], note: 'Value 不过 RoPE，直接活到 attention' },
      { key: 'wsQr', bytes: hidden, dtype: 'float16', shape: [tokens, HIDDEN], producer: 'sg1', consumers: ['sg2'], note: 'RoPE 后的 Query；形状与 wsQ 完全一致' },
      { key: 'wsKr', bytes: hidden, dtype: 'float16', shape: [tokens, HIDDEN], producer: 'sg1', consumers: ['sg2'], note: 'RoPE 后的 Key；形状与 wsK 完全一致' },
      { key: 'wsAttn', bytes: hidden, dtype: 'float16', shape: [tokens, HIDDEN], producer: 'sg2', consumers: ['sg3'], note: 'attention 输出' },
      { key: 'wsLse', bytes: lse, dtype: 'float32', shape: [tokens], producer: 'sg2', consumers: ['sg4', 'sg5'], note: '每行 logsumexp；体积极小但横跨四个子计算，是典型的装箱碎片来源' },
      { key: 'wsProj', bytes: hidden, dtype: 'float16', shape: [tokens, HIDDEN], producer: 'sg3', consumers: ['sg4'], note: '输出投影结果' },
      { key: 'wsNorm', bytes: hidden, dtype: 'float16', shape: [tokens, HIDDEN], producer: 'sg4', consumers: ['sg5'], note: 'LayerNorm 结果，FFN 的输入' },
      { key: 'wsFfn', bytes: ffn, dtype: 'float16', shape: [tokens, FFN_INNER], producer: 'sg5', consumers: ['sg5'], note: 'SwiGLU 中间量，单个张量里最大的一块' },
    ];
  }

  /** 输入输出：地址由调用方给，不参与装箱，但要画出来解释「这块地址为什么不能用」。 */
  function ioDeclarations(tokens) {
    return [
      { key: 'xGm', bytes: tokens * HIDDEN * 2, dtype: 'float16', shape: [tokens, HIDDEN], role: 'input', note: '算子输入，残差分支在 sg4 还要再读一次' },
      { key: 'yGm', bytes: tokens * HIDDEN * 2, dtype: 'float16', shape: [tokens, HIDDEN], role: 'output', note: '算子输出' },
    ];
  }

  // ---------------------------------------------------------------
  // 构建
  // ---------------------------------------------------------------
  function buildTensors(cfg) {
    const tokens = cfg.tokens || BASE_TOKENS;
    const overrides = cfg.overrides || {};
    const decls = declarations(tokens);

    const tensors = decls.map((d, index) => {
      const over = overrides[d.key] || {};
      const consumers = over.consumers || d.consumers;
      const declLine = SRC.kernel.lineOf(`${d.key}.SetGlobalBuffer`);
      const producerSg = SUBGRAPHS[SG_INDEX[d.producer]];
      const hotLine = SRC.kernel.lineOf(producerSg.fn);
      return {
        id: d.key,
        name: d.key,
        role: 'workspace',
        order: index,
        size: alignUp(d.bytes, GM_ALIGN),
        dataBytes: d.bytes,
        dtype: d.dtype,
        shape: d.shape,
        producer: d.producer,
        consumers,
        live: {
          start: SG_INDEX[d.producer],
          end: Math.max(...consumers.map((id) => SG_INDEX[id])),
        },
        // GM 独有：片上 buffer 天然核内私有，GM 不是。这个字段决定复用是否安全。
        blockScope: over.blockScope || 'shared',
        aliasOf: over.aliasOf || null,
        onChip: over.onChip || false,
        onChipRegion: over.onChipRegion || null,
        note: over.note || d.note,
        src: { file: SRC.kernel.path, declLine, hotLine },
        code: SRC.kernel.snippet(Math.max(0, hotLine - 2), hotLine + 2),
      };
    });

    const io = ioDeclarations(tokens).map((d, index) => ({
      id: d.key,
      name: d.key,
      role: d.role,
      order: 1000 + index,
      size: alignUp(d.bytes, GM_ALIGN),
      dataBytes: d.bytes,
      dtype: d.dtype,
      shape: d.shape,
      producer: null,
      consumers: [],
      live: { start: 0, end: SUBGRAPHS.length - 1 },
      blockScope: 'shared',
      aliasOf: null,
      onChip: false,
      note: d.note,
      src: { file: SRC.kernel.path, declLine: SRC.kernel.lineOf(`${d.key}.SetGlobalBuffer`), hotLine: SRC.kernel.lineOf('void Process()') },
      code: SRC.kernel.snippet(SRC.kernel.lineOf('void Process()'), SRC.kernel.lineOf('void Process()') + 8),
    }));

    // 别名张量并入宿主：宿主的生命周期要延长到别名的最后一个消费者
    tensors.forEach((t) => {
      if (!t.aliasOf) return;
      const host = tensors.find((x) => x.id === t.aliasOf);
      if (!host) return;
      host.live.end = Math.max(host.live.end, t.live.end);
    });

    return tensors.concat(io);
  }

  /**
   * 顺序布局：每个子计算各申请各的一段，谁也不复用谁。
   * reuse 给出的张量直接落在目标的起始地址上、不推进游标 —— 这就是「手工复用」
   * 省容量的方式，也是它踩内存的方式（与 data/runs.js 里片上手工复用同一套写法）。
   */
  function sequentialLayout(tensors, reuse = {}) {
    const layout = {};
    let cursor = 0;
    tensors
      .filter((t) => t.role === 'workspace' && !t.onChip && !t.aliasOf)
      .sort((a, b) => a.order - b.order)
      .forEach((t) => {
        const target = reuse[t.id];
        if (target && layout[target] != null) {
          layout[t.id] = layout[target];
          return;
        }
        cursor = alignUp(cursor, GM_ALIGN);
        layout[t.id] = cursor;
        cursor += t.size;
      });
    return layout;
  }

  function buildRun(cfg) {
    const tokens = cfg.tokens || BASE_TOKENS;
    const tensors = buildTensors(cfg);

    const run = {
      schemaVersion: '0.2',
      id: cfg.id,
      label: cfg.label,
      kicker: cfg.kicker,
      note: cfg.note,
      kernel: {
        name: 'MLABlock_fused',
        source: SRC.kernel.path,
        tilingSource: SRC.tiling.path,
        blockDim: BLOCK_DIM,
        shape: { tokens, hidden: HIDDEN, ffnInner: FFN_INNER },
      },
      subgraphs: SUBGRAPHS,
      tensors,
      workspace: {
        budget: BUDGET,
        align: GM_ALIGN,
        policy: cfg.policy,
        scope: 'shared',
        reportedAt: { file: SRC.tiling.path, line: SRC.tiling.lineOf('ws[0] =') },
      },
      shapeRange: cfg.shapeRange || null,
      layout: {},
    };

    // 布局：顺序申请（可带手工复用）or 规划器给的最紧布局
    run.layout = cfg.policy === 'per-subgraph'
      ? sequentialLayout(tensors, cfg.manualReuse)
      : PLANNER.layoutOf(run);

    Object.entries(cfg.manualReuse || {}).forEach(([id, targetId]) => {
      const t = tensors.find((x) => x.id === id);
      if (t) t.manualReuseOf = targetId;
    });

    return run;
  }

  // ---------------------------------------------------------------
  // 候选集合
  // ---------------------------------------------------------------
  const CONFIGS = [
    {
      id: 'ws-naive',
      label: '逐子计算申请',
      kicker: '基线',
      note: '每个子计算各切各的一段 workspace，全程不释放。这是融合算子「workspace 越滚越大」的原始形态。',
      policy: 'per-subgraph',
    },
    {
      id: 'ws-packed',
      label: '按复用组重排',
      kicker: '候选解',
      note: '顺序与形状都不动，只把生命周期不重叠的张量落到同一段地址上。这一步的收益是确定的，不改变任何计算结果。',
      policy: 'planned',
    },
    {
      id: 'ws-inplace',
      label: 'RoPE 原地',
      kicker: '降低下界',
      note: 'RoPE 是逐元素且形状不变，可以写回自己的输入。wsQr / wsKr 直接消失 —— 这压低的是**下界本身**，是复用做到极致也拿不到的收益。',
      policy: 'planned',
      overrides: {
        wsQr: { aliasOf: 'wsQ', note: 'RoPE 原地写回 wsQ，不再单独申请' },
        wsKr: { aliasOf: 'wsK', note: 'RoPE 原地写回 wsK，不再单独申请' },
        wsQ: { consumers: ['sg2'], note: 'RoPE 原地改写后直接被 FlashAttn 读，生命周期延长到 sg2' },
        wsK: { consumers: ['sg2'], note: 'RoPE 原地改写后直接被 FlashAttn 读，生命周期延长到 sg2' },
      },
    },
    {
      id: 'ws-onchip',
      label: 'Lse 留在片上',
      kicker: '不落 GM',
      note: 'wsLse 只有 4KB 却横跨四个子计算，把地址空间劈成两半 —— 三种排序策略里有两种被它多顶出 4KB。它小到能整个留在 UB，根本不该落 GM。',
      policy: 'planned',
      overrides: {
        wsLse: { onChip: true, onChipRegion: 'UB', note: '4KB 统计量常驻 UB，不占 GM' },
      },
    },
    {
      id: 'ws-unsafe',
      label: '手工复用（危险）',
      kicker: '反例',
      note: '在逐子计算申请的基础上手工压两块地址：wsAttn 压在 wsV 上（生命周期其实重叠），wsProj 压在 wsQ 上（甘特图上完全错开，但 wsQ 是 per-block 切分）。省下 4MB，代价是两处踩内存 —— 而且仍然比工具排出来的布局大。',
      policy: 'per-subgraph',
      overrides: {
        wsQ: { blockScope: 'per-block', note: 'QKVProj 按 token 切到各 block，每 block 只持有自己的 1/8' },
      },
      manualReuse: { wsAttn: 'wsV', wsProj: 'wsQ' },
    },
    {
      id: 'ws-dynshape',
      label: '动态 shape 上界',
      kicker: '最坏 shape',
      note: 'token 数在 [512, 2048] 之间浮动，workspace 必须按上界预留。结论要给区间，不能给单值。',
      policy: 'planned',
      tokens: 2048,
      shapeRange: { field: 'tokens', min: 512, max: 2048, current: 2048 },
    },
  ];

  function buildAll() {
    return CONFIGS.map((cfg) => buildRun(cfg));
  }

  global.MemVizFusionRuns = {
    buildAll, buildRun, CONFIGS, SUBGRAPHS,
    BASE_TOKENS, HIDDEN, FFN_INNER, BUDGET, BLOCK_DIM, GM_ALIGN,
  };
})(window);
