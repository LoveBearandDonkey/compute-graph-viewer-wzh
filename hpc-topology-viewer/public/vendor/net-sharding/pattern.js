/* ════════════════════════════════════════════════════════════════════════════
   net-sharding pattern —— 整网切分 · rank 装载
   ────────────────────────────────────────────────────────────────────────────
   回答的问题：**整网（模型计算图）被六个并行维切开之后，一个 rank 里到底装了什么。**

   逻辑魔方（rubik-cube）回答「谁和谁一组」——它的坐标系是并行超立方，卡是主体。
   本 pattern 的坐标系是**模型本身**：算子是主体，rank 是「这个算子的哪一片落在谁手里」。
   两者是同一批 p 的两种读法，互为反查（见 README 的挂点表）。

   三件事，缺一不可：
   ① 每个算子被**哪个维、沿张量的哪根轴**切开（切分规格 SHARD）；
   ② 每个 rank 因此**持有哪一片**（payloadOf：层段 / head 片 / 专家桶 / 序列块 / 上下文块 / 词表片）；
   ③ HCCL 集合原语是**分片状态的转换器**——AllGather/ReduceScatter/AllToAll/AllReduce/P2P
      各自把张量从一种分片状态搬到另一种（flowOf）。这三件事同屏，才叫「读懂整网怎么装进 rank」。

   与 graph-meta.ts 的 NODE_DIM 的关系：那份是「一个算子贴一个维」的粗标签（且无 CP/SP），
   够用来做整网图↔魔方的着色联动，但答不了「切哪根轴 / 我这张卡持有哪一片」。本 pattern
   是它的**细化**：同一张 openPangu 图，标注升级为 {维, 轴, 份数, 本 rank 持有区间}。
   ════════════════════════════════════════════════════════════════════════════ */
(function (global) {
  'use strict';

  const VERSION = '0.1.0';

  /* ── 维度签名色（与 rubik-cube 同一套，CP/SP 为本 pattern 新增）──────────
     TP/PP/DP/EP 沿用逻辑魔方，两个 pattern 并排看时同一维必须同一色。 */
  const DIM_META = {
    tp: { name: 'TP', full: '张量并行',   cssVar: '--ns-tp', axis: '切单层权重（head / hidden / vocab / ffn 列）' },
    cp: { name: 'CP', full: '上下文并行', cssVar: '--ns-cp', axis: '切上下文长度（seq 的 KV 段）' },
    sp: { name: 'SP', full: '序列并行',   cssVar: '--ns-sp', axis: '切序列 token（norm / dropout 区）' },
    pp: { name: 'PP', full: '流水并行',   cssVar: '--ns-pp', axis: '切模型层（stage）' },
    ep: { name: 'EP', full: '专家并行',   cssVar: '--ns-ep', axis: '切 MoE 专家（expert bank）' },
    dp: { name: 'DP', full: '数据并行',   cssVar: '--ns-dp', axis: '切数据批次（整模型复制）' },
  };
  const DIM_ORDER = ['tp', 'cp', 'sp', 'pp', 'ep', 'dp'];

  /* ── 张量轴 ──────────────────────────────────────────────────────────────
     一个张量在图里流动时，它的每根轴各自「是完整的还是被谁切了」。
     这套轴名是后面所有分片状态（state）的字母表。 */
  const AXES = {
    batch:  '批次',
    seq:    '序列 token',
    ctx:    '上下文（KV 段）',
    head:   '注意力头',
    hidden: '隐藏维',
    ffn:    'FFN 中间维',
    expert: '专家',
    vocab:  '词表',
    layer:  '模型层',
  };

  /* ══════════════════════════════════════════════════════════════════════════
     切分规格表 —— openPangu 图的 55 个节点，每个标注「被哪些维沿哪根轴切」。
     ──────────────────────────────────────────────────────────────────────────
     `by`   : [{dim, axis}]  —— 谁沿哪根轴切它；空数组 = 每个 rank 都有完整一份（复制）
     `prim` : 若本身就是 HCCL 集合原语，写它做的是哪种状态转换
     `note` : 一句话说清「为什么是这样切」（读者不该去猜）

     判据不是「这个算子长什么样」，而是**它的权重/激活的哪根轴上有可分的份数**：
     · 线性层按输出维列切 / 按输入维行切（行切的输出是 partial，必须 AllReduce 才完整）
     · norm / 逐元素 不改变特征维 → 沿 token 切最省显存，这正是 SP 的存在理由
     · MoE 专家 bank 沿 expert 轴切 → EP；token 要先 AllToAll 送到专家所在的卡
     · attention 沿 head 切 → TP；上下文太长时再沿 seq 切 → CP（算 attention 前 AllGather KV）
     ══════════════════════════════════════════════════════════════════════════ */
  const SHARD = {
    /* ── model-core：PP 的首尾两段 ───────────────────────────────────────── */
    input_tokens:   { by: [{ dim: 'dp', axis: 'batch' }, { dim: 'cp', axis: 'seq' }],
                      note: 'DP 切批次；上下文并行开启时序列先按 CP 切成段' },
    token_embedding: { by: [{ dim: 'tp', axis: 'vocab' }],
                      note: '词表并行：词表按 TP 切，查表后 AllReduce 合并（落在 PP 首段）' },
    embedding_weight: { by: [{ dim: 'tp', axis: 'vocab' }], kind: 'param',
                      note: '词表权重按 TP 切；DP 副本间由梯度 AllReduce 保持一致' },
    final_norm:     { by: [{ dim: 'sp', axis: 'seq' }],
                      note: 'RMSNorm 不改变特征维 → 沿 token 切（SP），落在 PP 末段' },
    lm_head:        { by: [{ dim: 'tp', axis: 'vocab' }],
                      note: '输出头按词表切，与 embedding 同一根轴' },
    lm_head_weight: { by: [{ dim: 'tp', axis: 'vocab' }], kind: 'param',
                      note: '输出头权重按 TP 切' },
    logits:         { by: [{ dim: 'tp', axis: 'vocab' }, { dim: 'dp', axis: 'batch' }],
                      note: '词表片上的 logits；求 loss 时跨 TP 归约' },
    mtp_module:     { by: [{ dim: 'tp', axis: 'hidden' }],
                      note: '多 token 预测头，跟随主干 TP 切分' },

    /* ── decoder-stack：PP 切层 · norm 区归 SP ──────────────────────────── */
    decoder_layer:      { by: [{ dim: 'pp', axis: 'layer' }],
                          note: '整个解码层是 PP 的切分单位：每个 stage 持有连续若干层' },
    mhc_attention:      { by: [{ dim: 'tp', axis: 'head' }], note: 'mHC 注意力分支，随 TP 切 head' },
    mhc_attention_post: { by: [{ dim: 'tp', axis: 'head' }], note: 'mHC 合并，随 TP 切 head' },
    input_layernorm:    { by: [{ dim: 'sp', axis: 'seq' }],  note: '进入 TP 区之前的 norm → SP 沿 token 切' },
    post_attention_norm:{ by: [{ dim: 'sp', axis: 'seq' }],  note: '出 TP 区之后的 norm → SP' },
    pre_mlp_norm:       { by: [{ dim: 'sp', axis: 'seq' }],  note: '进 FFN 之前的 norm → SP' },
    post_mlp_norm:      { by: [{ dim: 'sp', axis: 'seq' }],  note: '出 FFN 之后的 norm → SP' },
    block_post_norm:    { by: [{ dim: 'sp', axis: 'seq' }],  note: '块尾 norm → SP' },

    /* ── attention-block：TP + CP + SP ──────────────────────────────────── */
    sparse_mla_attention: { by: [{ dim: 'tp', axis: 'head' }], note: 'MLA 注意力整体按 head 切' },
    q_a_proj:  { by: [{ dim: 'tp', axis: 'hidden' }], note: 'Q 降维投影：列切（输出 latent 维）' },
    q_causal_conv: { by: [{ dim: 'tp', axis: 'hidden' }], note: '逐通道卷积，跟随上游列切，无通信' },
    q_residual_add:{ by: [{ dim: 'tp', axis: 'hidden' }], note: '逐元素相加，跟随分片' },
    q_a_norm:  { by: [{ dim: 'tp', axis: 'hidden' }], note: 'latent 上的 norm，跟随 TP 分片' },
    q_b_proj:  { by: [{ dim: 'tp', axis: 'head' }],   note: 'Q 升维：输出按注意力头切 → 每 rank 持有部分 head' },
    kv_a_proj: { by: [{ dim: 'tp', axis: 'hidden' }], note: 'KV 降维投影：列切' },
    kv_causal_conv:{ by: [{ dim: 'tp', axis: 'hidden' }], note: '逐通道卷积，跟随分片' },
    kv_residual_add:{ by: [{ dim: 'tp', axis: 'hidden' }], note: '逐元素相加，跟随分片' },
    kv_a_norm: { by: [{ dim: 'tp', axis: 'hidden' }], note: 'latent norm，跟随分片' },
    kv_b_proj: { by: [{ dim: 'tp', axis: 'head' }],   note: 'KV 升维：输出按 head 切' },
    query_tensor: { by: [{ dim: 'tp', axis: 'head' }, { dim: 'cp', axis: 'ctx' }], kind: 'tensor',
                    note: 'Q：head 归 TP；上下文并行时 seq 段归 CP' },
    key_tensor:   { by: [{ dim: 'tp', axis: 'head' }, { dim: 'cp', axis: 'ctx' }], kind: 'tensor',
                    note: 'K：算 attention 前需在 CP 组内 AllGather 出完整上下文' },
    value_tensor: { by: [{ dim: 'tp', axis: 'head' }, { dim: 'cp', axis: 'ctx' }], kind: 'tensor',
                    note: 'V：同 K，CP 组内 AllGather' },
    attention_core: { by: [{ dim: 'tp', axis: 'head' }, { dim: 'cp', axis: 'ctx' }],
                    note: '每 rank 只算自己那几个 head；CP 开启时只算本地 token 对全局上下文的注意力' },
    o_causal_conv:  { by: [{ dim: 'tp', axis: 'head' }], note: '输出侧卷积，跟随 head 分片' },
    o_residual_add: { by: [{ dim: 'tp', axis: 'head' }], note: '逐元素，跟随分片' },
    o_proj:    { by: [{ dim: 'tp', axis: 'hidden' }], partial: true,
                 note: '输出投影：行切 → 每 rank 得到的是 partial sum，必须归约才完整' },
    attention_projection_weights: { by: [{ dim: 'tp', axis: 'head' }], kind: 'param',
                 note: 'QKVO 投影权重按 head/hidden 切，是 TP 组内 AllReduce 的对象' },

    /* ── ffn-block：TP（dense）+ EP（MoE）+ SP ──────────────────────────── */
    dense_mlp:       { by: [{ dim: 'tp', axis: 'ffn' }], note: 'Dense FFN 整体按中间维切' },
    dense_gate_up:   { by: [{ dim: 'tp', axis: 'ffn' }], note: '列并行：gate/up 输出按 FFN 中间维切' },
    dense_silu:      { by: [{ dim: 'tp', axis: 'ffn' }], note: 'SiLU 逐元素，跟随列切，无通信' },
    dense_down:      { by: [{ dim: 'tp', axis: 'hidden' }], partial: true,
                       note: '行并行：down 投影输出是 partial sum，需归约' },
    dense_mlp_weights: { by: [{ dim: 'tp', axis: 'ffn' }], kind: 'param', note: 'Dense FFN 权重按中间维切' },

    /* ── moe-block：EP 是主角 ───────────────────────────────────────────── */
    router_gate:  { by: [], note: '路由门是复制的：每 rank 都要独立算出 token 该去哪个专家' },
    router_weight:{ by: [], kind: 'param', note: '路由权重复制（很小），由 DP 梯度同步保持一致' },
    route_topk:   { by: [], note: 'TopK 选专家，复制计算 —— 之后才谈得上把 token 送出去' },
    routed_expert_bank: { by: [{ dim: 'ep', axis: 'expert' }],
                    note: '路由专家 bank 按专家切：每 rank 只持有自己那一桶专家的权重' },
    expert_bank_weights:{ by: [{ dim: 'ep', axis: 'expert' }], kind: 'param',
                    note: '专家权重按 EP 切 —— MoE 显存不爆的根本原因' },
    expert_parallel_state: { by: [{ dim: 'ep', axis: 'expert' }], kind: 'tensor',
                    note: '专家并行运行态：本 rank 这一桶专家的负载/容量' },
    shared_expert_mlp:  { by: [{ dim: 'tp', axis: 'ffn' }],
                    note: '共享专家每个 rank 都有（不按 EP 切），内部按 TP 切中间维' },
    shared_expert_weights: { by: [{ dim: 'tp', axis: 'ffn' }], kind: 'param',
                    note: '共享专家权重按 TP 切，不参与 EP' },
    moe_combine:  { by: [{ dim: 'tp', axis: 'hidden' }], note: '按路由权重加权合并专家输出' },

    /* ── 集合通信算子 = 分片状态的转换器 ────────────────────────────────── */
    attention_all_gather: { by: [], prim: 'AllGather', group: 'tp',
        from: { seq: 'sp' }, to: { seq: null },
        note: 'SP→TP 区入口：把沿 token 切开的激活在 TP 组内收齐成完整序列' },
    attention_reduce_scatter: { by: [], prim: 'ReduceScatter', group: 'tp',
        from: { hidden: 'partial' }, to: { seq: 'sp' },
        note: 'TP 区出口：把 partial sum 归约的同时沿 token 散开 —— 一步顶 AllReduce+Scatter' },
    ffn_all_gather: { by: [], prim: 'AllGather', group: 'tp',
        from: { seq: 'sp' }, to: { seq: null },
        note: '进 FFN 前同样要收齐完整序列' },
    ffn_reduce_scatter: { by: [], prim: 'ReduceScatter', group: 'tp',
        from: { hidden: 'partial' }, to: { seq: 'sp' },
        note: '出 FFN 时归约 + 沿 token 散开，回到 SP 状态' },
    moe_all_to_all_dispatch: { by: [], prim: 'AllToAll', group: 'ep',
        from: { seq: 'local' }, to: { expert: 'ep' },
        note: 'EP 的核心：token 按路由结果重分布到专家所在的 rank —— 从「按 token 分」变成「按专家分」' },
    moe_all_to_all_combine: { by: [], prim: 'AllToAll', group: 'ep',
        from: { expert: 'ep' }, to: { seq: 'local' },
        note: '专家算完把结果送回 token 原属的 rank —— dispatch 的逆转换' },
  };

  /* ══════════════════════════════════════════════════════════════════════════
     createModel —— 纯数据层，无 DOM / 无渲染依赖（可单测、可被别的视图复用）
     ══════════════════════════════════════════════════════════════════════════ */
  function createModel(cfgIn) {
    const c = Object.assign({
      tp: 8, pp: 5, dp: 100, ep: 2, cp: 1,
      layers: 48, heads: 64, experts: 64, sharedExperts: 4,
      seqLen: 4096, hidden: 5120, ffnHidden: 14336, vocab: 153376,
    }, cfgIn || {});

    const TP = Math.max(1, c.tp | 0), CP = Math.max(1, c.cp | 0);
    const PP = Math.max(1, c.pp | 0), DP = Math.max(1, c.dp | 0);
    const EP = Math.max(1, c.ep | 0);
    /* SP 不是独立的 rank 维：序列并行复用 TP 组（Megatron 口径，与 data.ts 的
       「SP 与 TP 同域」一致）。所以 rank 总数只乘 TP·CP·PP·DP。 */
    const SP = TP;
    const N = TP * CP * PP * DP;

    /* rank 分解：TP 最内 → CP → PP → DP 最外。
       CP=1 时与仓库现有约定（tp=k%TP · stage · replica）逐位相同，是它的推广。 */
    function coordOf(rank) {
      const k = ((rank % N) + N) % N;
      const tp = k % TP;
      const cp = Math.floor(k / TP) % CP;
      const pp = Math.floor(k / (TP * CP)) % PP;
      const dp = Math.floor(k / (TP * CP * PP));
      return { rank: k, tp, cp, pp, dp, ep: dp % EP, sp: tp };
    }
    function rankOf(co) {
      const tp = (co.tp | 0) % TP, cp = (co.cp | 0) % CP;
      const pp = (co.pp | 0) % PP, dp = (co.dp | 0) % DP;
      return ((dp * PP + pp) * CP + cp) * TP + tp;
    }

    /* 层段：余数摊给前面的 stage（与 deployment.ts stageLayerRange 同口径） */
    function stageLayers(stage) {
      const base = Math.floor(c.layers / PP), rem = c.layers % PP;
      const lo = stage * base + Math.min(stage, rem);
      const hi = lo + base + (stage < rem ? 1 : 0) - 1;
      return [lo, Math.max(lo, hi)];
    }
    /* 等分某根轴（不能整除时前面的片多拿一个），返回 [lo, hi] 闭区间 */
    function slice(total, parts, idx) {
      const base = Math.floor(total / parts), rem = total % parts;
      const lo = idx * base + Math.min(idx, rem);
      const hi = lo + base + (idx < rem ? 1 : 0) - 1;
      return [lo, Math.max(lo, hi)];
    }

    /* HCCL 通信域成员：固定其余坐标，只让这一维跑遍 */
    function groupOf(rank, dim, cap) {
      const co = coordOf(rank), out = [];
      const lim = cap == null ? Infinity : cap;
      if (dim === 'tp' || dim === 'sp') {
        for (let i = 0; i < TP && out.length < lim; i++) out.push(rankOf({ ...co, tp: i }));
      } else if (dim === 'cp') {
        for (let i = 0; i < CP && out.length < lim; i++) out.push(rankOf({ ...co, cp: i }));
      } else if (dim === 'pp') {
        for (let i = 0; i < PP && out.length < lim; i++) out.push(rankOf({ ...co, pp: i }));
      } else if (dim === 'dp') {
        for (let i = 0; i < DP && out.length < lim; i++) out.push(rankOf({ ...co, dp: i }));
      } else if (dim === 'ep') {
        /* EP 折入 DP 轴：相邻 EP 个副本构成一个 A2A 域 */
        const blk = Math.floor(co.dp / EP) * EP;
        for (let i = 0; i < EP && out.length < lim; i++) out.push(rankOf({ ...co, dp: blk + i }));
      }
      return out;
    }

    /* ── 本 rank 到底装了什么 ───────────────────────────────────────────── */
    function payloadOf(rank) {
      const co = coordOf(rank);
      const [l0, l1] = stageLayers(co.pp);
      const heads = slice(c.heads, TP, co.tp);
      const vocab = slice(c.vocab, TP, co.tp);
      const ffn = slice(c.ffnHidden, TP, co.tp);
      const experts = slice(c.experts, EP, co.ep);
      const ctx = slice(c.seqLen, CP, co.cp);
      /* SP 在 CP 段之内再按 TP 切一次（norm 区的 token 分片） */
      const cpLen = ctx[1] - ctx[0] + 1;
      const spRel = slice(cpLen, SP, co.sp);
      const spTok = [ctx[0] + spRel[0], ctx[0] + spRel[1]];
      return {
        rank: co.rank, coord: co,
        layers: [l0, l1], layerCount: l1 - l0 + 1,
        heads, headCount: heads[1] - heads[0] + 1,
        experts, expertCount: experts[1] - experts[0] + 1,
        sharedExperts: c.sharedExperts,       // 共享专家不按 EP 切，每 rank 都有
        ctx, ctxCount: ctx[1] - ctx[0] + 1,
        spTokens: spTok, spCount: spTok[1] - spTok[0] + 1,
        vocab, vocabCount: vocab[1] - vocab[0] + 1,
        ffn, ffnCount: ffn[1] - ffn[0] + 1,
        isFirstStage: co.pp === 0, isLastStage: co.pp === PP - 1,
      };
    }

    /* 某个算子在本 rank 上持有哪一片（拿切分规格 × payload 求交） */
    function shardOfNode(nodeId, rank) {
      const spec = SHARD[nodeId];
      if (!spec) return null;
      const p = payloadOf(rank), parts = [];
      (spec.by || []).forEach((b) => {
        const dm = DIM_META[b.dim];
        let range = null, total = null;
        if (b.axis === 'head') { range = p.heads; total = c.heads; }
        else if (b.axis === 'expert') { range = p.experts; total = c.experts; }
        else if (b.axis === 'layer') { range = p.layers; total = c.layers; }
        else if (b.axis === 'vocab') { range = p.vocab; total = c.vocab; }
        else if (b.axis === 'ffn') { range = p.ffn; total = c.ffnHidden; }
        else if (b.axis === 'ctx') { range = p.ctx; total = c.seqLen; }
        else if (b.axis === 'seq') { range = p.spTokens; total = c.seqLen; }
        else if (b.axis === 'hidden') { range = slice(c.hidden, TP, p.coord.tp); total = c.hidden; }
        else if (b.axis === 'batch') { range = null; total = null; }
        parts.push({ dim: b.dim, dimName: dm ? dm.name : b.dim, axis: b.axis,
                     axisName: AXES[b.axis] || b.axis, range, total });
      });
      return { nodeId, spec, parts, replicated: parts.length === 0, partial: !!spec.partial };
    }

    /* ── 一个 decoder layer 内的分片状态流水 ────────────────────────────────
       这是「输入输出 × attention/MoE × HCCL 算子」三者关系的正面回答：
       每一步给出**此刻张量处于什么分片状态**，以及**是哪个集合原语把它搬过去的**。 */
    function flowOf() {
      const steps = [];
      const push = (o) => steps.push(o);
      push({ id: 'input_layernorm', label: 'Input RMSNorm', kind: 'op', dim: 'sp',
             state: `seq 切 SP×${SP}`, why: 'norm 不改变特征维 → 沿 token 切最省显存' });
      push({ id: 'attention_all_gather', label: 'AllGather', kind: 'comm', dim: 'tp', prim: 'AllGather',
             state: 'seq 完整', why: '进 TP 区前把序列收齐（TP 组内）' });
      push({ id: 'q_b_proj', label: 'QKV 投影', kind: 'op', dim: 'tp',
             state: `head 切 TP×${TP}`, why: '注意力头按 TP 切，每 rank 只持有部分 head' });
      if (CP > 1) {
        push({ id: 'key_tensor', label: 'CP AllGather(KV)', kind: 'comm', dim: 'cp', prim: 'AllGather',
               state: 'ctx 完整', why: `上下文按 CP×${CP} 切开，算 attention 前在 CP 组内收齐 KV` });
      }
      push({ id: 'attention_core', label: 'FlashAttention', kind: 'op', dim: 'tp',
             state: `head 切 TP×${TP}${CP > 1 ? ` · ctx 切 CP×${CP}` : ''}`,
             why: '只算自己那几个 head' });
      push({ id: 'o_proj', label: 'Output Proj（行切）', kind: 'op', dim: 'tp', partial: true,
             state: 'hidden partial', why: '行并行的输出是部分和，必须归约' });
      push({ id: 'attention_reduce_scatter', label: 'ReduceScatter', kind: 'comm', dim: 'tp', prim: 'ReduceScatter',
             state: `seq 切 SP×${SP}`, why: '归约 partial 的同时沿 token 散开，回到 SP' });
      push({ id: 'pre_mlp_norm', label: 'Pre MLP RMSNorm', kind: 'op', dim: 'sp',
             state: `seq 切 SP×${SP}`, why: '又一个 norm → SP 区' });
      push({ id: 'ffn_all_gather', label: 'AllGather', kind: 'comm', dim: 'tp', prim: 'AllGather',
             state: 'seq 完整', why: '进 FFN 前收齐序列' });
      push({ id: 'route_topk', label: 'Router TopK', kind: 'op', dim: null,
             state: '复制', why: '每 rank 都独立算 token 该去哪个专家' });
      push({ id: 'moe_all_to_all_dispatch', label: 'AllToAll Dispatch', kind: 'comm', dim: 'ep', prim: 'AllToAll',
             state: `expert 切 EP×${EP}`, why: 'token 重分布到专家所在的 rank：从「按 token 分」变「按专家分」' });
      push({ id: 'routed_expert_bank', label: '路由专家计算', kind: 'op', dim: 'ep',
             state: `expert 切 EP×${EP}`, why: `每 rank 只持有 ${Math.ceil(c.experts / EP)} 个专家的权重` });
      push({ id: 'moe_all_to_all_combine', label: 'AllToAll Combine', kind: 'comm', dim: 'ep', prim: 'AllToAll',
             state: 'token 归位', why: '结果送回 token 原属的 rank（dispatch 的逆）' });
      push({ id: 'ffn_reduce_scatter', label: 'ReduceScatter', kind: 'comm', dim: 'tp', prim: 'ReduceScatter',
             state: `seq 切 SP×${SP}`, why: '出 FFN 归约 + 散开' });
      push({ id: 'post_mlp_norm', label: 'Post MLP RMSNorm', kind: 'op', dim: 'sp',
             state: `seq 切 SP×${SP}`, why: '块尾 norm' });
      if (PP > 1) {
        push({ id: '__pp_send', label: 'P2P Send → 下一 stage', kind: 'comm', dim: 'pp', prim: 'P2P',
               state: '激活出栈', why: `PP×${PP} 段之间只传激活，通信量最小` });
      }
      return steps;
    }

    /* 反查：某个维切了哪些算子（整网图高亮用） */
    const dimNodes = { tp: [], cp: [], sp: [], pp: [], dp: [], ep: [] };
    Object.keys(SHARD).forEach((id) => {
      const s = SHARD[id];
      (s.by || []).forEach((b) => { if (dimNodes[b.dim]) dimNodes[b.dim].push(id); });
      if (s.group && dimNodes[s.group] && dimNodes[s.group].indexOf(id) < 0) dimNodes[s.group].push(id);
    });

    /* 某算子的主导维（一个算子可能被多维切，取最内层那个作为着色依据） */
    function primaryDim(nodeId) {
      const s = SHARD[nodeId];
      if (!s) return null;
      if (s.group) return s.group;
      if (!s.by || !s.by.length) return null;
      const rank_ = { tp: 0, cp: 1, sp: 2, ep: 3, pp: 4, dp: 5 };
      return s.by.slice().sort((a, b) => rank_[a.dim] - rank_[b.dim])[0].dim;
    }

    return {
      config: c, N, sizes: { TP, CP, SP, PP, DP, EP },
      coordOf, rankOf, payloadOf, shardOfNode, groupOf, stageLayers, slice,
      flowOf, primaryDim, dimNodes, SHARD, DIM_META, AXES,
    };
  }


  /* ══════════════════════════════════════════════════════════════════════════
     rank 卡 = 整网图的**并行化展开**，不是它的裁剪
     ──────────────────────────────────────────────────────────────────────────
     一张 rank 卡里的节点集合相对逻辑整网图**既减也增**：

       减 —— 别的 stage 的层在这张卡上**根本不存在**（不是画淡，是没有）：
             词嵌入只落 PP 首段、LM Head 只落末段。
       增 —— HCCL 集合原语是**并行化产生的**，逻辑图里没有这些节点：
             TP AllReduce（行切算子的 partial sum）· PP Send/Recv（段边界）·
             DP 梯度 AllReduce · CP AllGather(KV)。
             （AllGather/ReduceScatter/AllToAll 那几个图里本来就有，属「保留」。）
       余下的是**分片**：节点还在，但只持有一片（head 第 i 片 / 专家第 j 桶 / …）。

     显示形式：**共享布局 + 差分着色**。这张图是预定位的（55 个节点各带 x/y），
     所以 small multiples 是白送的——三张卡并排、位置逐点对齐，于是三个结论
     **不靠文字**就能读出来：
       · 同 TP 组：结构完全相同，只有分片区间不同；
       · 跨 PP 段：结构不同（词嵌入/LM Head 出现或消失）；
       · 跨 EP 桶：结构相同，专家桶不同。
     ══════════════════════════════════════════════════════════════════════════ */

  /* 只落在 PP 首/末段的节点——「减」的来源 */
  const FIRST_ONLY = new Set(['input_tokens', 'token_embedding', 'embedding_weight']);
  const LAST_ONLY = new Set(['final_norm', 'lm_head', 'lm_head_weight', 'logits', 'mtp_module']);

  /* 「增」：并行化插入的 HCCL 节点，逻辑图里没有。锚在它插入点那个节点旁边。 */
  const ADDED = [
    { id: '+tp_ar', label: 'TP AllReduce', dim: 'tp', anchor: 'o_proj', dx: 1.06, dy: 0,
      why: '行切算子（o_proj / dense_down）算出的是 partial sum，要在 TP 组内归约才完整' },
    { id: '+cp_ag', label: 'CP AllGather(KV)', dim: 'cp', anchor: 'key_tensor', dx: 1.06, dy: 0,
      when: (m) => m.sizes.CP > 1,
      why: '上下文按 CP 切段后，算 attention 前要在 CP 组内收齐全局 K/V' },
    { id: '+pp_sr', label: 'PP Send/Recv', dim: 'pp', anchor: 'block_post_norm', dx: 1.06, dy: 0,
      when: (m) => m.sizes.PP > 1,
      why: 'stage 边界把激活传给下一段（反向传梯度）' },
    { id: '+dp_ar', label: 'DP 梯度 AllReduce', dim: 'dp', anchor: 'attention_projection_weights', dx: -1.06, dy: 0,
      when: (m) => m.sizes.DP > 1,
      why: '副本间同步梯度，可与反向重叠' },
  ];

  function mount(container, opts) {
    opts = opts || {};
    const el = typeof container === 'string' ? document.querySelector(container) : container;
    if (!el) throw new Error('[net-sharding] container not found');

    let model = createModel(opts.config);
    let theme = opts.theme === 'dark' ? 'dark' : 'light';
    let axis = opts.axis || 'tp';          // 比较轴：tp / pp / ep —— 决定并排的是哪三张卡
    let base = opts.rank != null ? opts.rank : Math.floor(model.N / 2);

    el.classList.add('ns-root');
    el.setAttribute('data-theme', theme);
    el.innerHTML = '<div class="ns-bar"></div><div class="ns-mult"></div><div class="ns-foot"></div>';
    const bar = el.querySelector('.ns-bar');
    const mult = el.querySelector('.ns-mult');
    const foot = el.querySelector('.ns-foot');

    const G = () => global.OPENPANGU_GRAPH;

    /* 三张卡选谁：固定其余坐标、只让比较轴那一维变 —— 与「通信域 = 坐标切面」同一条规则。 */
    function tripletOf() {
      const m = model, co = m.coordOf(base), s = m.sizes, out = [];
      const pick = (n, cap) => {
        const k = Math.min(cap, n), step = Math.max(1, Math.floor(n / k));
        const idx = []; for (let i = 0; idx.length < k && i < n; i += step) idx.push(i);
        return idx;
      };
      if (axis === 'tp') pick(s.TP, 3).forEach((i) => out.push(m.rankOf({ ...co, tp: i })));
      else if (axis === 'pp') pick(s.PP, 3).forEach((i) => out.push(m.rankOf({ ...co, pp: i })));
      else pick(s.EP, 3).forEach((i) => out.push(m.rankOf({ ...co, dp: Math.floor(co.dp / s.EP) * s.EP + i })));
      return out;
    }

    /* 一个节点在这张卡上处于什么状态：absent（根本不存在）/ shard（只持一片）/ full（完整一份） */
    function nodeState(nodeId, rank) {
      const p = model.payloadOf(rank);
      if (FIRST_ONLY.has(nodeId) && !p.isFirstStage) return { k: 'absent', why: '词嵌入只落 PP 首段' };
      if (LAST_ONLY.has(nodeId) && !p.isLastStage) return { k: 'absent', why: 'LM Head 只落 PP 末段' };
      const sh = model.shardOfNode(nodeId, rank);
      if (!sh || sh.replicated) return { k: 'full' };
      const part = sh.parts[0];
      return { k: 'shard', dim: part.dim, range: part.range, of: part.total, label: part.dimName };
    }

    const STATE_COLOR = {
      absent: 'var(--ns-absent)', full: 'var(--ns-full)', added: 'var(--ns-added)',
    };

    /* 一张 rank 卡：共享布局（节点用图自带的 x/y）+ 差分着色 */
    function cardSVG(rank) {
      const g = G(); if (!g) return '<div class="ns-empty">整网图未加载</div>';
      const W = g.width, H = g.height;
      const byId = {}; g.nodes.forEach((n) => { byId[n.id] = n; });
      const p = model.payloadOf(rank), co = p.coord;
      let s = '';
      // 边：两端都在才画（一端被减掉 → 这条边也不存在）
      g.edges.forEach((e) => {
        const a = byId[e.source], b = byId[e.target]; if (!a || !b) return;
        const sa = nodeState(e.source, rank), sb = nodeState(e.target, rank);
        const gone = sa.k === 'absent' || sb.k === 'absent';
        s += `<line x1="${a.x + a.width / 2}" y1="${a.y + a.height / 2}" x2="${b.x + b.width / 2}" y2="${b.y + b.height / 2}"`
          + ` class="ns-e${gone ? ' is-gone' : ''}"/>`;
      });
      // 节点
      g.nodes.forEach((n) => {
        const st = nodeState(n.id, rank);
        const fam = st.k === 'absent' ? STATE_COLOR.absent
          : st.k === 'shard' ? `var(--ns-${st.dim})` : STATE_COLOR.full;
        s += `<rect x="${n.x}" y="${n.y}" width="${n.width}" height="${n.height}" rx="10"`
          + ` class="ns-n is-${st.k}" style="--c:${fam}"><title>${esc(n.label)} · `
          + (st.k === 'absent' ? esc(st.why) : st.k === 'full' ? '完整一份（复制）'
            : `${st.label}切 · 持有 ${esc(String(st.range ? st.range.join('–') : ''))}`) + `</title></rect>`;
      });
      // 增：并行化插入的 HCCL 节点（逻辑图里没有）
      ADDED.forEach((a) => {
        if (a.when && !a.when(model)) return;
        const an = byId[a.anchor]; if (!an) return;
        const w = 300, h = 62;
        const x = an.x + a.dx * an.width * 0.62, y = an.y + a.dy * 60;
        s += `<line x1="${an.x + an.width / 2}" y1="${an.y + an.height / 2}" x2="${x + w / 2}" y2="${y + h / 2}" class="ns-e is-added"/>`
          + `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="31" class="ns-n is-added" style="--c:var(--ns-${a.dim})">`
          + `<title>${esc(a.label)} · 并行化新增 · ${esc(a.why)}</title></rect>`;
      });
      const s2 = model.sizes;
      const badge = axis === 'tp' ? `TP${co.tp}` : axis === 'pp' ? `PP${co.pp}` : `桶${co.ep}`;
      const sub = axis === 'tp' ? `head ${p.heads[0]}–${p.heads[1]}`
        : axis === 'pp' ? `L${p.layers[0]}–L${p.layers[1]}`
          : `E${p.experts[0]}–${p.experts[1]}`;
      return `<figure class="ns-card">`
        + `<figcaption><b>rank ${rank}</b><span class="ns-badge" style="--c:var(--ns-${axis})">${badge}</span>`
        + `<em>${esc(sub)}</em></figcaption>`
        + `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMin meet">${s}</svg></figure>`;
    }

    function esc(t) { return String(t == null ? '' : t).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

    const AXIS_SAY = {
      tp: '同 TP 组：<b>结构完全相同</b>，只有分片区间不同（每卡拿不同的 head）。',
      pp: '跨 PP 段：<b>结构不同</b> —— 词嵌入只在首段、LM Head 只在末段，别的段这些节点<b>根本不存在</b>。',
      ep: '跨 EP 桶：<b>结构相同</b>，专家桶不同（每卡只持有自己那一桶专家的权重）。',
    };

    function render() {
      const s = model.sizes;
      bar.innerHTML = `<div class="ns-title"><b>rank 卡 = 整网的并行化展开</b>`
        + `<span class="ns-sub">节点既减（别的段的层不存在）也增（HCCL 是并行化产生的）</span></div>`
        + `<div class="ns-axis">${['tp', 'pp', 'ep'].map((a) =>
          `<button class="ns-axbtn${a === axis ? ' is-on' : ''}" data-a="${a}" style="--c:var(--ns-${a})">`
          + `并排比 ${a.toUpperCase()}</button>`).join('')}</div>`
        + `<span class="ns-chip">${s.TP}×${s.CP}×${s.PP}×${s.DP} = ${model.N} rank</span>`;
      bar.querySelectorAll('.ns-axbtn').forEach((b) => {
        b.addEventListener('click', () => { axis = b.dataset.a; render(); });
      });
      mult.innerHTML = tripletOf().map(cardSVG).join('');
      foot.innerHTML = `<div class="ns-say">${AXIS_SAY[axis]}</div>`
        + `<div class="ns-key">`
        + `<span class="ns-k"><i style="background:var(--ns-full)"></i>完整一份</span>`
        + `<span class="ns-k"><i style="background:var(--ns-tp)"></i>分片（色=切它的维）</span>`
        + `<span class="ns-k"><i class="is-absent"></i>这张卡上不存在</span>`
        + `<span class="ns-k"><i class="is-added" style="background:var(--ns-added)"></i>并行化新增的 HCCL</span>`
        + `</div>`;
    }

    render();
    return {
      get model() { return model; },
      get state() { return { axis, rank: base, theme }; },
      setAxis(a) { if (['tp', 'pp', 'ep'].includes(a)) { axis = a; render(); } },
      selectRank(r) { base = Math.max(0, Math.min(model.N - 1, r | 0)); render(); },
      setConfig(cfg) {
        const c2 = Object.assign({}, model.config, cfg);
        if (c2.dp % c2.ep !== 0) return { ok: false, error: 'ep 必须整除 dp' };
        model = createModel(c2); if (base >= model.N) base = model.N - 1;
        render(); return { ok: true };
      },
      setTheme(t) { theme = t === 'dark' ? 'dark' : 'light'; el.setAttribute('data-theme', theme); },
      destroy() { el.innerHTML = ''; },
    };
  }

  global.PtoNetShardingPattern = { createModel, mount, VERSION, SHARD, DIM_META, AXES };
})(window);
