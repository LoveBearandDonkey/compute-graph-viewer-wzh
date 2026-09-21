/**
 * pto rubik-cube pattern —— 「逻辑魔方」独立 pattern。
 *
 * 从 cube-cockpit.html 的逻辑魔方抽出（形态/轴标/正交 2D/粒度提示/四维通信组/读图钥匙 全保留），
 * 并把并行度做成配置：默认 TP=2 · PP=4 · EP=8 · DP(A2A 域)=16 → 稠密层副本 128 · rank 总数 1024。
 * 注册为 window.PtoRubikCubePattern：
 *   createModel(config) → 纯布局/拓扑模型（无 Three.js 依赖，可单测/复用）
 *   mount(container, opts) → 完整交互渲染器（需 window.THREE，r128 即可）
 *
 * 保留的核心表达（与 cockpit 逻辑魔方一致，仅数字随配置变化）：
 *   · 5 种形态：标准（X=TP·Y=PP·Z=DP）/ DP平铺（副本宫格）/ EP聚簇（专家桶墙）/
 *     TP切片（权重墙）/ PP流水（段横向展开）——切形态=换投影轴，飞行动画重排；
 *   · 每形态的 3D 坐标网格框 + 轴标注 + 关键结构线（「为什么这样摆 · 这个形状帮你看什么」）；
 *   · 1 小块 = 1 卡（rank）=（TP,PP,DP）坐标交点 · EP 折入 DP 轴（桶↔卡非 1:1）；
 *   · 正交 顶/前/侧 2D 视角 + 被折叠深度维的剖面逐层翻 + 「每格=几张卡」粒度小贴士；
 *   · 选中一张卡 → TP/PP/DP/EP 四维通信组同屏高亮（维度签名色）；
 *   · 状态热力 / 按维分组着色透镜 · 异常注入（异常的形状 → 根因类别）。
 *
 * 之后与整网图（model-graphviz）/ 专家图结合的挂点：handle.selectLayer(l)（整网层 → 魔方水平切片）、
 * handle.selectBucket(e)（专家桶 → 整面墙）、opts.onSelect（rank 反查多维身份）。
 */
(function registerPtoRubikCubePattern(global) {
  'use strict';

  /* ── 并行度配置：rank 总数 = tp × pp × dp（默认 2×4×128 = 1024）。
        EP 不参与乘法（与 cockpit 白皮书语义一致：EP 折入 DP 轴，不新增轴）——
        ep 只要求整除 dp：副本 rep 持有专家桶 rep%ep，相邻 ep 个副本构成
        1 个 A2A 域，共 dp/ep 个域（默认 128/8 = 16）。 ── */
  /* 默认 = 128 卡小规格（2·4·16·8）。落地这一档是为了**第一眼能读**：4000 卡时一张卡只有
     两三个像素，四种形态之间的差别（墙 / 段 / 桶 / 板）都糊成一片色块，读者看到的是一团颜色
     而不是一种堆法；128 卡下每张卡看得清、每一段数得出，形态到底在说什么当场就成立。
     真实规格（盘古 Pro MoE：TP8·EP2·PP5·4K NPU → dp = 4000/(8×5) = 100，EP2 折入其中
     → 50 个 A2A 域，出处 data/ascend-workload-pangu-moe.json ← arXiv 2505.21411）留作
     「并行」那排的预设，一键就能切过去看真实体量。 */
  const DEFAULTS = {
    tp: 2, pp: 4, dp: 16, ep: 8,
    cp: 1,                 // 上下文并行：沿序列切上下文段；与 TP 共用一根轴（TP 在内）
    layers: 48,            // 整网层数 → 每 PP 段 layers/pp 层
    experts: 64,           // 路由专家总数 → 每桶 experts/ep 个
    hotBuckets: [0, 2],    // 示意热点专家桶（标暖色）
    // ── 整网对象透镜用的模型规格（只影响「持有哪一片」的读数，不影响布局）──
    heads: 64,             // 注意力头总数 → TP 沿 head 轴切
    vocab: 153376,         // 词表大小 → TP 沿 vocab 轴切（Embedding / LM Head）
    ffnHidden: 14336,      // FFN 中间维 → TP 沿 ffn 轴切
    sharedExperts: 4,      // 共享专家（每 rank 都有，不按 EP 切）
    seqLen: 4096,          // 序列长度 → SP 沿 token 轴切（norm 区）
    // ── 内存构成透镜（着色 w/act/grad/opt）用的模型宽度 ──
    // heads=64 已经定了「切多细」，hidden 定「每一刀切下来的那片有多大」——
    // 二者缺一都算不出字节数。取 8192 与 heads 对齐给出 headDim=128（真实量级，
    // 同 parallel-topology/demo.html「story128」预设），而不是随手一个数。
    hidden: 8192,
  };

  /* ══ 整网对象目录（本次深化的事实源）════════════════════════════════════
     魔方原本只回答「谁和谁一组」。这张表让它多回答一件事：
     **整网（模型计算图）上的一个对象，是怎么被切开、落到哪些 rank 上的。**

     分带（band）沿用设计系统 `model-architecture-training-sidecar` 契约里的
     `semanticBands.module = ["Embedding","Attention","MoE","Head"]`，不另立一套；
     算子名沿用 openPangu 图里那批有据可查的算子（该契约明令禁止用通用
     Transformer 栈替换它们），另加一个「通信」带装 HCCL 算子。

     每条声明四件事：
       `by`      切它的维 × 张量轴 —— 判据不是算子长什么样，而是**它的权重/激活的
                 哪根轴上有可分的份数**：线性层按输出维列切 / 按输入维行切（行切的
                 输出是 partial sum，必须归约）；norm 不改变特征维、沿 token 切最省
                 显存，这正是 SP 的存在理由；MoE 专家 bank 沿 expert 轴切。
       `carry`   哪些 rank 承载它（PP 首/末段专属的对象只落在那一段）
       `shard`   这张卡持有的是第几片、区间是什么
       `best`    切到哪个形态，这个对象的分片会 snap 成规整的块/墙
                 —— 这是魔方「异常的形状 = 根因类别」那条设计语言的推广：
                    **切分的形状 = 这个对象被哪一维切**。

     通信算子（band='通信'）按 md 笔记 §6 的规则特殊处理：**它不属于单个 rank，
     而属于一个通信组**，在组内每个 rank 上各有一个实例。所以它的「承载集合」
     取选中卡的那个通信组，而不是某批固定的卡。 */
  const OBJ_BAND = ['Embedding', 'Attention', 'MoE', 'Head', '通信'];
  // 等分某根轴：不能整除时前面的片多拿一个（与 stageLayerRange 同口径）
  function axSlice(total, parts, idx) {
    const base = Math.floor(total / parts), rem = total % parts;
    const lo = idx * base + Math.min(idx, rem);
    return { lo, hi: Math.max(lo, lo + base + (idx < rem ? 1 : 0) - 1) };
  }
  const rgOf = (t, n, i, pre) => { const s = axSlice(t, n, i); return `${pre || ''}${s.lo}-${pre || ''}${s.hi}`; };

  /* 算子族色 —— **不是本 pattern 自己定的**，是已发布的整网图 pattern 那一套：
     `patterns/model-architecture-3d-deck` 的 `COLOR_FALLBACKS`，其 key 与本仓库
     `openpangu-graph.json` 节点上的 `opv:* / io:*` colorKey 同源。
     两个 pattern 并排看时，「哪一族算子」必须是同一个颜色——否则读者要在两套色里
     各记一遍。维度签名色（TP/PP/DP/EP/CP/SP）与这套是**两个正交的色系**：
     族色答「这是什么算子」，签名色答「它被哪一维切」，装载清单里同屏出现、各管各的。 */
  const OPV = {
    embedding: '#14B8A6', norm: '#38BDF8', attention: '#3B82F6', linear: '#4F46E5',
    head: '#7C3AED', mlp: '#A855F7', act: '#8B5CF6', gate: '#F59E0B', moe: '#EA580C',
    comm: '#06B6D4', decoder: '#0D9488',
  };

  const NET_OBJ = [
    { id: 'embedding', fam: 'embedding', short: '词嵌入', band: 'Embedding', name: '词嵌入 Vocab Parallel Embedding',
      by: [{ dim: 'TP', axis: 'vocab' }], best: 'ppf', firstStage: true,
      note: '词表并行：词表按 TP 切，查表后组内归约。只落在 PP 首段——别的段没有这个对象。',
      carry: (m, r) => m.ppOf(r) === 0,
      shard: (m, r) => ({ dim: 'TP', axis: '词表', idx: m.tpOf(r), of: m.TP, range: rgOf(m.config.vocab, m.TP, m.tpOf(r)) }) },

    { id: 'qkv', fam: 'linear', short: 'QKV', band: 'Attention', name: 'QKV 投影 Q/KV Up Linear',
      by: [{ dim: 'TP', axis: 'head' }], best: 'tps',
      note: '注意力头按 TP 切：每张卡只算自己那几个 head。一整根纵深行 = 一个 TP 组，组内 8 片拼成全部 head。',
      carry: () => true,
      shard: (m, r) => ({ dim: 'TP', axis: '注意力头', idx: m.tpOf(r), of: m.TP, range: rgOf(m.config.heads, m.TP, m.tpOf(r), 'h') }) },

    { id: 'attn_core', fam: 'attention', short: 'Attn', band: 'Attention', name: 'Sparse FlashAttention（注意力核）',
      by: [{ dim: 'TP', axis: 'head' }, { dim: 'CP', axis: 'ctx' }], best: 'tps',
      note: '只算自己那几个 head。启用 CP 后上下文再沿序列切段，算之前要在 CP 组内 AllGather 收齐 KV —— 这是 CP 唯一会打破「各算各的」的地方。',
      carry: () => true,
      /* 被两维同时切：`shard` 给**主导那一维**（着色与主屏按它），`shards` 给**全部维**。
         主导维的选法：CP>1 时是 CP（此时更值得看，head 切法与 QKV 完全相同），
         CP=1 时退回 TP —— 与 CP 引入前逐位一致。 */
      shard: (m, r) => (m.CP > 1
        ? { dim: 'CP', axis: '上下文段', idx: m.cpOf(r), of: m.CP, range: rgOf(m.config.seqLen, m.CP, m.cpOf(r), 'tok') }
        : { dim: 'TP', axis: '注意力头', idx: m.tpOf(r), of: m.TP, range: rgOf(m.config.heads, m.TP, m.tpOf(r), 'h') }),
      shards: (m, r) => {
        const head = { dim: 'TP', axis: '注意力头', idx: m.tpOf(r), of: m.TP, range: rgOf(m.config.heads, m.TP, m.tpOf(r), 'h') };
        if (m.CP <= 1) return [head];
        /* 两维同时切 → 这张卡持有的是**二维网格里的一格**：
           (head 第 i 片) × (上下文第 j 段)。总份数 = TP × CP。 */
        return [head, { dim: 'CP', axis: '上下文段', idx: m.cpOf(r), of: m.CP, range: rgOf(m.config.seqLen, m.CP, m.cpOf(r), 'tok') }];
      } },

    { id: 'o_proj', fam: 'linear', short: 'O 投影', band: 'Attention', name: 'Output Projection（行切）',
      by: [{ dim: 'TP', axis: 'hidden' }], best: 'tps', partial: true,
      note: '行并行：每张卡算出来的是 partial sum，必须在 TP 组内归约才是完整输出。',
      carry: () => true,
      shard: (m, r) => ({ dim: 'TP', axis: '隐藏维', idx: m.tpOf(r), of: m.TP }) },

    { id: 'norm', fam: 'norm', short: 'RMSNorm', band: 'Attention', name: 'RMSNorm 区（SP）',
      by: [{ dim: 'SP', axis: 'seq' }], best: 'tps',
      note: 'norm 不改变特征维 → 沿 token 切最省显存，这就是 SP。SP 复用 TP 组但不是 TP：同一组卡，TP 区里每卡持有部分 head（AllReduce 合并），SP 区里每卡持有部分 token（AllGather/ReduceScatter 转换布局）——同一批卡在计算图的不同位置扮演两种角色。',
      carry: () => true,
      shard: (m, r) => ({ dim: 'SP', axis: '序列 token', idx: m.tpOf(r), of: m.TP, range: rgOf(m.config.seqLen, m.TP, m.tpOf(r), 'tok') }) },

    { id: 'dense_ffn', fam: 'mlp', short: 'Dense FFN', band: 'MoE', name: 'Dense FFN（列切→行切）',
      by: [{ dim: 'TP', axis: 'ffn' }], best: 'tps',
      note: '列并行算 gate/up，行并行算 down —— 中间维按 TP 切，出口归约。',
      carry: () => true,
      shard: (m, r) => ({ dim: 'TP', axis: 'FFN 中间维', idx: m.tpOf(r), of: m.TP, range: rgOf(m.config.ffnHidden, m.TP, m.tpOf(r)) }) },

    { id: 'router', fam: 'gate', short: 'Router', band: 'MoE', name: 'Router Gate / TopK（复制）',
      by: [], best: null,
      note: '路由门是复制的：每张卡都独立算一遍 token 该去哪个专家——先知道去哪，才谈得上把 token 发出去。',
      carry: () => true, shard: () => null },

    { id: 'experts', fam: 'moe', short: '路由专家', band: 'MoE', name: '路由专家 bank Routed Experts',
      by: [{ dim: 'EP', axis: 'expert' }], best: 'ep',
      note: '专家按 EP 切：每张卡只持有自己那一桶专家的权重 —— MoE 显存不爆的根本原因。',
      carry: () => true,
      shard: (m, r) => ({ dim: 'EP', axis: '专家', idx: m.epOf(r), of: m.EP, range: m.expRange(m.epOf(r)) }) },

    { id: 'shared_expert', fam: 'mlp', short: '共享专家', band: 'MoE', name: '共享专家 Shared Expert',
      by: [{ dim: 'TP', axis: 'ffn' }], best: 'tps',
      note: '共享专家每张卡都有（不按 EP 切），内部按 TP 切中间维 —— 与路由专家正好相反，值得对照着看。',
      carry: () => true,
      shard: (m, r) => ({ dim: 'TP', axis: 'FFN 中间维', idx: m.tpOf(r), of: m.TP }) },

    { id: 'lm_head', fam: 'head', short: 'LM Head', band: 'Head', name: 'LM Head 输出头',
      by: [{ dim: 'TP', axis: 'vocab' }], best: 'ppf', lastStage: true,
      note: '输出头按词表切，与 embedding 同一根轴。只落在 PP 末段。',
      carry: (m, r) => m.ppOf(r) === m.PP - 1,
      shard: (m, r) => ({ dim: 'TP', axis: '词表', idx: m.tpOf(r), of: m.TP, range: rgOf(m.config.vocab, m.TP, m.tpOf(r)) }) },

    /* ── 通信带：属于通信组，不属于单张卡 ── */
    { id: 'c_tp', fam: 'comm', band: '通信', name: 'TP AllReduce（注意力/FFN 输出合并）',
      comm: 'TP', prim: 'AllReduce', best: 'tps',
      note: '行切算子的 partial sum 在 TP 组内求和。HCCL：HcclAllReduce。' },
    { id: 'c_sp', fam: 'comm', band: '通信', name: 'SP AllGather / ReduceScatter（布局转换）',
      comm: 'TP', prim: 'AllGather', best: 'tps',
      note: '进 TP 区收齐序列、出 TP 区归约并散开。绑定的仍是 TP 组（SP 与 TP 同域）。',
      alias: 'SP' },
    { id: 'c_ep', fam: 'comm', band: '通信', name: 'EP AllToAll（Dispatch / Combine）',
      comm: 'EP', prim: 'AllToAll', best: 'ep',
      note: 'token 按路由结果重分布到专家所在的卡，算完再送回。HCCL：HcclAlltoAll（token 不均时用 V 变体）。' },
    { id: 'c_cp', fam: 'comm', band: '通信', name: 'CP AllGather（收齐上下文 KV）',
      comm: 'CP', prim: 'AllGather', best: 'tps',
      note: '上下文按 CP 切段后，算 attention 前要在 CP 组内收齐全局 K/V —— 序列被切开、注意力却要看全局，这一步就是代价。HCCL：HcclAllGather。' },
    { id: 'c_pp', fam: 'comm', band: '通信', name: 'PP Send / Recv（阶段接力）',
      comm: 'PP', prim: 'P2P', best: 'ppf',
      note: '只在 stage 边界传激活与梯度，通信量最小。HCCL：HcclSend / HcclRecv。' },
    { id: 'c_dp', fam: 'comm', band: '通信', name: 'DP AllReduce（梯度同步）',
      comm: 'DP', prim: 'AllReduce', best: 'dpt',
      note: '副本间同步梯度，可与反向重叠。HCCL：HcclAllReduce。' },
  ];
  const NET_OBJ_BY = {}; NET_OBJ.forEach((o) => { NET_OBJ_BY[o.id] = o; });

  /* ══ 布局规则（单一事实源）════════════════════════════════════════════
     目标：任何 (tp,pp,dp,ep) 组合都自动排布好，形态只声明「意图」不写死数值，
     以后加维度（CP/SP…）或加形态只需按同样规则声明。三条规则：

       ① 卡块尺寸恒定（CARD）——换形态、换配置都不改变一张卡的大小；
       ② 单维轴步距 = CARD[轴] + 缝隙(成员数) × 层级倍率。缝隙按成员数分档：
          成员越多缝越紧（长轴不被拉成细丝），越少缝越松（短轴不退化成薄片）；
          层级倍率表达意图：dense 密排 · normal 常规 · spread 留白 · emph 强调分离；
       ③ 复合轴（外维分块 · 内维紧邻，如「板 / 墙」）：外维步距 = 内维总跨度 +
          分块留白（按跨度成比例），于是任何规模下块与块都清楚分开、不重叠；
       ④ 2D 可读性约束：任一正交视角（顶/前/侧）下，屏幕两根轴的「最细步距」之比
          ≤ MAX_RATIO。否则一轴被拉成稀疏条纹——例如 DP 平铺的板在顶视里只有 1 张卡
          厚，若行距按板宽取（方形宫格），每格 98% 是空的、完全读不出结构。因此
          分块轴的块间距以「同屏另一轴最细步距 × MAX_RATIO」封顶（见 clampFor2D）。

     不变量：任一轴步距 ≥ 该轴卡块尺寸 + 最小缝（0.13×0.55 ≈ 0.07）→ 永不重叠。 */
  // 卡块是正方体：同一张卡在任何形态、任何视角下都呈现同样的形状。若做成长方体，
  // 某个维度落到哪根轴上、卡就把哪个面朝向你，读起来像「卡被转了 90°」——同一种东西
  // 在不同视图里长得不一样，正是 cockpit 当年修过的失衡问题。
  const CARD = { x: 0.6, y: 0.6, z: 0.6 };
  const GAP_BY_N = (n) => (n <= 4 ? 0.85 : n <= 12 ? 0.6 : n <= 48 ? 0.3 : 0.13);
  const TIER = { dense: 0.55, normal: 1, spread: 1.8, emph: 4 };
  const stepOf = (ax, n, tier) => CARD[ax] + GAP_BY_N(n) * TIER[tier || 'normal'];
  const padOf = (span) => Math.max(0.9, span * 0.34);        // 块间留白（随块跨度成比例）
  const MAX_RATIO = 4;                                       // 同屏两轴最细步距的比值上限
  // 把分块步距压到「同屏另一轴最细步距」的 MAX_RATIO 倍以内（下限 = 该轴卡块 + 缝）
  const clampFor2D = (want, ax, ...coPitches) =>
    Math.max(CARD[ax] + 0.3, Math.min(want, MAX_RATIO * Math.min(...coPitches)));

  /* 一个训练 step 的通信阶段（与集群驾驶舱 cube-cockpit.html 的 PHASES 同一套语义）：
     时间轴按阶段读，而不是抽象的秒——每个阶段由哪根并行轴主导、走哪层总线、负载多高
     都不同，热力场因此随阶段变化（TP 组齐动 / PP 接力波沿流水级前进 / EP 浪涌点亮热点桶 /
     DP 全网梯度）。 */
  const PHASES = [
    { id: 'TP', dim: 'TP', name: 'TP · 前向 AllReduce', bus: '节点内 UB · 高频', load: 0.92 },
    { id: 'PP', dim: 'PP', name: 'PP · 阶段接力 Send/Recv', bus: 'Pod 内跨 Host · 中频', load: 0.45 },
    { id: 'EP', dim: 'EP', name: 'EP · MoE AllToAll 浪涌', bus: 'Pod 内全互联 · 浪涌', load: 0.88 },
    { id: 'DP', dim: 'DP', name: 'DP · 梯度 AllReduce', bus: '跨 Pod Scale-Out · 低频大包', load: 0.60 },
  ];
  const STEP_SEC = 12;                                       // 一个 step 的墙钟时长（每阶段 3s）

  /* ════════ 设计系统色卡解析（3D 用不了 CSS 变量 → 挂载/切主题时把 token 读成色值）════════
     所有 3D 着色都从 PTO Design System 的 token 取：
       · 负载热力 = --success → --warning → --danger（与图例色带同源，颜色逐格一致）
       · 异常组   = --danger
       · 分组着色 = highlight 六族（copy-blue / accum-orange / l0a-violet / ub-green /
                    mte-amber / l0b-deep-violet）的 400 与 600 两档，共 12 色循环
       · 网格 / 外框 / 字牌底与描边 = --border-* · --surface-* · --foreground-*
       · 字牌字体 = --font-sans
     唯一的例外：TP 的维度签名色需要一个既非红黄绿（状态色专用）又不与 DP 蓝 / EP 紫
     撞色的青，而设计系统色卡里没有青族 → 由 pattern 提供 --dim-tp，并在 README 标记
     为待回portal 上游（同上游对 .toggle-outline 的处理方式）。 */
  const TOKEN_KEYS = [
    '--background', '--surface-1', '--surface-2', '--foreground', '--foreground-secondary',
    '--foreground-muted', '--border-default', '--border-strong', '--primary', '--success',
    '--warning', '--danger', '--accent', '--font-sans', '--font-mono',
    '--highlight-copy-blue-300', '--highlight-accum-orange-300', '--highlight-l0a-violet-300',
    '--highlight-ub-green-300', '--highlight-mte-amber-300', '--highlight-l0b-deep-violet-300',
    '--highlight-copy-blue-400', '--highlight-accum-orange-400', '--highlight-l0a-violet-400',
    '--highlight-ub-green-400', '--highlight-mte-amber-400', '--highlight-l0b-deep-violet-400',
    '--highlight-copy-blue-600', '--highlight-accum-orange-600', '--highlight-l0a-violet-600',
    '--highlight-ub-green-600', '--highlight-mte-amber-600', '--highlight-l0b-deep-violet-600',
    '--highlight-ub-green-700', '--highlight-mte-amber-500', '--highlight-ub-green-600',
    '--dim-tp',
  ];
  // css 颜色 → {r,g,b,a}（支持 #rgb/#rrggbb/#rrggbbaa 与 rgb()/rgba()）
  function cssRGBA(c) {
    c = (c || '').trim();
    let m = /^#([0-9a-f]{3,8})$/i.exec(c);
    if (m) {
      let h = m[1];
      if (h.length === 3) h = h.split('').map((x) => x + x).join('');
      const n = parseInt(h.slice(0, 6), 16);
      return { r: (n >> 16 & 255) / 255, g: (n >> 8 & 255) / 255, b: (n & 255) / 255, a: h.length >= 8 ? parseInt(h.slice(6, 8), 16) / 255 : 1 };
    }
    m = /^rgba?\(([^)]+)\)$/i.exec(c);
    if (m) {
      const p = m[1].split(/[,/\s]+/).filter(Boolean).map(parseFloat);
      return { r: p[0] / 255, g: p[1] / 255, b: p[2] / 255, a: p.length > 3 ? p[3] : 1 };
    }
    return { r: 0.5, g: 0.5, b: 0.5, a: 1 };
  }
  const hex2 = (v) => ('0' + Math.round(Math.max(0, Math.min(1, v)) * 255).toString(16)).slice(-2);

  /* 维度签名色的 token 映射（PP/DP/EP 在色卡里有精确对应，TP 见上文说明） */
  /* 六维签名色。SP 原先借用 TP 的青（因为 SP 复用 TP 组）——但「共用通信组」不等于
     「是同一维」：SP 切的是 seq 轴、跑的是 AllGather/ReduceScatter，TP 切的是 head 轴、
     跑的是 AllReduce。借色的结果是同一屏里两者长得一模一样，读者分不出此刻在说哪一维。
     所以给 SP 与 CP 各自的色：都取色卡里未被四维占用、且与 TP 青拉得开的族。 */
  const DIM_TOKEN = {
    TP: '--dim-tp', PP: '--warning', DP: '--primary', EP: '--highlight-l0a-violet-400',
    SP: '--highlight-mte-amber-500', CP: '--highlight-ub-green-600',
  };
  /* 物理链路层级色（B 档）：同机 UB 用色卡里的 ub-green（名字与语义天然对上），
     Pod 内 rail 用 mte-amber，跨 Pod 用 l0b-deep-violet——三者都不与四维签名色撞。
     不用红黄绿是因为红/黄在本 pattern 里是状态色（负载/异常）专用。 */
  const TIER_TOKEN = {
    ub: '--highlight-ub-green-400',
    rail: '--highlight-mte-amber-400',
    out: '--highlight-l0b-deep-violet-600',
  };
  /* 分组着色调色板：六族 × 两档（深 600 → 浅 300），共 12 色。
     不用 400/500 档：那两档是色卡里彩度最高的一段（ub-green-400 #B3F141、
     mte-amber-400 #F4CB22 尤甚），几百上千张卡同屏铺满时会糊成一片荧光。
     600 档饱和度相当但明度低一截、300 档明度高但彩度降下来——两档都落在"能分辨、
     不刺眼"的区间，且深浅交替本身又多给了一层区分度。
     ub-green 整族偏荧光，深档取 700（橄榄绿）而不是 600。
     排序不是「先排完深档再排浅档」：六族里蓝 / 靛 / 紫三族色相只差 25~55°，一旦同档相邻
     就分不开（PP0 深蓝 vs PP4 深靛）。所以两档交错着排——前 6 位先用四个真正拉得开的
     色相（蓝·橙·橄榄·紫），第 5、6 位改用浅档的琥珀与靛，靠明度差把近色相拆开。
     分组数通常 ≤8，前几位的可分辨度最值钱。 */
  const GROUP_TOKENS = [
    '--highlight-copy-blue-600', '--highlight-accum-orange-600', '--highlight-ub-green-700',
    '--highlight-l0a-violet-600', '--highlight-mte-amber-500', '--highlight-l0b-deep-violet-300',
    '--highlight-accum-orange-300', '--highlight-copy-blue-300', '--highlight-ub-green-300',
    '--highlight-l0a-violet-300', '--highlight-l0b-deep-violet-600', '--highlight-mte-amber-300',
  ];

  /* ════════════════════════ 纯布局模型 ════════════════════════ */
  function createModel(userCfg) {
    const C = Object.assign({}, DEFAULTS, userCfg || {});
    const TP = C.tp | 0, PP = C.pp | 0, EP = C.ep | 0;
    const CP = Math.max(1, C.cp | 0);     // 上下文并行度（真实 rank 维，进乘法）
    const REP = C.dp | 0;                 // 稠密层 DP 副本数（EP 折入其中，不参与乘法）
    if (TP < 1 || PP < 1 || EP < 1 || REP < 1) throw new Error('rubik-cube: tp/pp/dp/ep 均须 ≥ 1');
    if (REP % EP) throw new Error(`ep(${EP}) 须整除 dp(${REP})——EP 折入 DP 轴，不参与乘法`);
    const DOM = REP / EP;                 // A2A 域数（专家数据并行组）
    /* CP 是**真实的 rank 维**（与 SP/EP 不同：SP 复用 TP 组、EP 折入 DP 轴，都不进乘法），
       所以 rank 总数要乘它。空间上 CP 与 TP **共用一根轴**、TP 在内：
       轴上的位置 = cp*TP + tp（记作 lane）。这样 ① TP 组仍然连续（一段 TP 个相邻格）；
       ② CP 组是跨 TP 的等距抽样，读得出「同一 TP 槽位、不同上下文段」；
       ③ CP=1 时 lane ≡ tp、TPL ≡ TP，全部布局与坐标**逐位不变**，已发出的链接不受影响。 */
    const TPL = TP * CP;                  // 「TP 轴」的实际格数（TP 槽 × CP 段）
    const N = TP * CP * PP * REP;         // rank 总数 = tp × cp × pp × dp
    const LPS = Math.max(1, Math.round(C.layers / PP));            // 每段层数
    const EXP_PER = Math.max(1, Math.floor(C.experts / EP));       // 每桶专家数
    /* ── 分组间距放大（groupGap）────────────────────────────────────────────
       默认 1 = 一个字节都不动（已经发出去的链接、别的宿主看到的都还是原来那副
       样子）。给大于 1 的数，把**缝**（不是卡本身）按比例拉开：卡还是那么大，
       块与块之间的空档变宽，于是「这是四段 / 这是八片」在一屏 4000 卡的规模下
       才读得出来——参照的是并行拓扑矩阵那一屏（demo.html）里 PP 段与 EP 组之间
       那道明显的留白。放大的是 stepOf 的缝，所以 padOf（块间留白，按块跨度成
       比例）与 clampFor2D（2D 稀疏条纹的封顶）都跟着等比走，布局不变量
       （步距 ≥ 卡尺寸 + 最小缝）也照旧成立，不必逐形态手调。
       上限 6：再大就不是"分组更明显"而是"卡散成一片星点"，四个形态的块读法
       全都失效。 */
    const GK = Math.max(1, Math.min(6, +C.groupGap || 1));
    const stepK = (ax, n, tier) => CARD[ax] + GAP_BY_N(n) * TIER[tier || 'normal'] * GK;
    // ── 轴步距（布局规则推导）──
    const CY = 9;                                    // 逻辑体离地高度（各形态统一）
    const tpStep = stepK('x', TPL);                  // 板 / 墙内 TP(×CP) 列步距
    const ppStep = stepK('y', PP);                   // 板 / 墙内 PP 行步距
    const blockW = TPL * tpStep;                     // 一面墙的宽度（EP 聚簇：内维 TP(×CP) 一字排开）
    /* DP 平铺的「板」：把板内 TP 折成 (TPC 列 × TPD 排)，让板有厚度——一字排开的板
       只有 1 张卡厚，在顶视/侧视里都退化成稀疏条纹。取「宽 ≥ 深」且世界跨度最接近
       方形的分法（TP=8 → 4×2 · TP=16 → 4×4 · TP=2 → 2×1）。 */
    const tpStepZ = stepK('z', TPL);                // 板内 TP(×CP)「排」的纵深步距
    const TPC = (() => {
      const ideal = Math.sqrt(TPL * tpStepZ / tpStep), lo = Math.sqrt(TPL) - 1e-9;
      let best = TPL, err = Infinity;
      for (let c = 1; c <= TPL; c++) {
        if (TPL % c || c < lo) continue;              // 只取宽 ≥ 深的分法，保住「板」的横向读法
        const e = Math.abs(Math.log(c / ideal));
        if (e < err) { err = e; best = c; }
      }
      return best;
    })();
    const TPD = TPL / TPC;
    // 分块格距统一用同一条公式：「块在该轴的跨度 + padOf(该跨度)」，逐轴各算各的。
    const dptBlockW = TPC * tpStep, dptBlockD = TPD * tpStepZ;
    const dptCellX = dptBlockW + padOf(dptBlockW);
    const dptCellZ = clampFor2D(dptBlockD + padOf(dptBlockD), 'z', tpStep, ppStep);
    // 宫格行列数：按「世界跨度近方形」取，而不是按数量取方阵——板是宽而薄的
    // （列距 ≫ 行距），数量方阵会把顶视与轴测拉成长条。
    const COLS = (() => {
      const ideal = Math.max(1, Math.sqrt(REP * dptCellZ / dptCellX));
      let best = 1, err = Infinity;
      for (let c = 1; c <= REP; c++) {
        if (REP % c) continue;
        const e = Math.abs(Math.log(c / ideal));
        if (e < err) { err = e; best = c; }
      }
      return best;
    })();
    const ROWS = REP / COLS;

    // rank 编码：rank = ((rep*PP + pp)*CP + cp)*TP + tp（TP 最内 → CP → PP → DP）
    const tpOf = (r) => r % TP;
    const cpOf = (r) => ((r / TP) | 0) % CP;          // 持有的上下文段
    const laneOf = (r) => cpOf(r) * TP + tpOf(r);     // 「TP 轴」上的格号（CP=1 时 ≡ tp）
    const ppOf = (r) => ((r / (TP * CP)) | 0) % PP;
    const repOf = (r) => (r / (TP * CP * PP)) | 0;
    const epOf = (r) => repOf(r) % EP;            // 持有的专家桶
    const domOf = (r) => (repOf(r) / EP) | 0;     // 所属 A2A 域
    const gxOf = (r) => repOf(r) % COLS;          // DP 平铺列
    const gzOf = (r) => (repOf(r) / COLS) | 0;    // DP 平铺行
    const rankOf = (tp, pp, rep, cp) => (((rep * PP + pp) * CP + (((cp | 0) % CP) + CP) % CP) * TP + tp);
    /* 层号**从 0 起**，与 rank 侧的 TP0/PP0/DP0/桶0 一致，也与设计系统 sidecar 的
       `PP Stage 0 · L0-L11` 一致。原先写成 1 起（L1-L10）：同一屏里别的都从 0、只有 L 从 1，
       读者得记住这一个例外，而这个例外没有任何好处。 */
    const stageLayerRange = (s) => ({ lo: s * LPS, hi: Math.min(C.layers - 1, (s + 1) * LPS - 1) });
    const expRange = (e) => 'E' + (e * EXP_PER) + '-' + (e * EXP_PER + EXP_PER - 1);

    /* ── 物理落位（可选输入）：逻辑 rank → 装在哪台机、哪个 Pod ──────────────
       逻辑魔方本身只讲「谁和谁一组」，不讲「谁和谁插在一起」。给出落位后，同一条
       逻辑边就能标出它实际跨了哪层链路：
         同机  → UB（节点内全互联，带宽最高、延迟最低）
         同 Pod 跨机 → Pod 内 rail / Scale-Up
         跨 Pod → Scale-Out（最贵的一跳）
       默认按 rank 连号装机（rank 编码本就是 TP 最内层 → TP 组自然落在同一台机内，
       与真实作业的 rank-to-node 映射一致）。宿主可用 placement 覆盖每机卡数/每 Pod
       机数，或直接给 slots 数组做任意映射。 */
    const PLC = Object.assign({ cardsPerHost: 8, hostsPerPod: 32 }, C.placement || {});
    const CPH = Math.max(1, PLC.cardsPerHost | 0);          // 每台机的卡数
    const HPP = Math.max(1, PLC.hostsPerPod | 0);           // 每个 Pod 的机数
    const CPP = CPH * HPP;                                  // 每个 Pod 的卡数
    const slotOf = (r) => (PLC.slots && PLC.slots[r] != null ? PLC.slots[r] | 0 : r);
    const hostOf = (r) => (slotOf(r) / CPH) | 0;
    const podOf = (r) => (slotOf(r) / CPP) | 0;
    const HOSTS = Math.ceil(N / CPH), PODS = Math.ceil(N / CPP);
    // 一条逻辑边实际跨了哪层链路
    const tierOf = (a, b) => (hostOf(a) === hostOf(b) ? 'ub' : podOf(a) === podOf(b) ? 'rail' : 'out');

    // 居中偏移
    const cT = (TPL - 1) / 2, cP = (PP - 1) / 2, cR = (REP - 1) / 2,
      cE = (EP - 1) / 2, cD = (DOM - 1) / 2, cG = (COLS - 1) / 2, cZ = (ROWS - 1) / 2;

    /* 各形态的轴间距——全部由上面的布局规则推导（不再逐形态手调常量）。
       每个形态只声明「哪根轴放哪个维、用什么层级」，换任何并行数字都自动成立。 */
    const SP = {
      /* 标准：三根语义轴各一维，常规层级 —— 位置即多维坐标。
         轴分配 X=PP · Y=DP · Z=TP：把**有编号、要一段段读**的两维（PP 段 · DP 副本）
         摆到屏幕的横纵两轴上，于是「哪一段 × 哪个副本」在前视里是一张现成的表；
         TP 收进纵深，因为一整根 X 行就是一个 TP 组（组内无歧义，翻它用剖面）。
         步距按**各轴自己的枚数**取（GAP_BY_N），不是三轴共用一个数——PP 只有几段、
         DP 常有上百个，共用一个步距要么段挤成一团、要么副本拉得满屏放不下。
         PP 走 spread 档（1.8×）而不是 PP流水 的 emph 档（4×）：段与段之间有看得见的
         空档、读得出是「四组」而不是「一片」，但「把这根轴拉开」这个动作仍然归 PP流水
         ——两者因此还是差一个量级，而不是差一点点。三轴步距比也都在 MAX_RATIO 以内
         （128 卡：2.13 / 0.90 / 1.45），2D 不会散成稀疏条纹。 */
      /* 反馈「现在的间距只有 pp 之间有，dp、tp 之间的间距也要拉开分组」——
         之前只有 X（PP）传了 'spread' 层级，Y（DP）/Z（TP）沿用默认 'normal'（1×），
         于是 PP 只有几段时读得出段与段的留白，DP 这种上百个副本的轴因为
         GAP_BY_N 本来就随数量收缩、又没有额外层级加成，挤成了一条看不出
         断点的斜线。三根轴统一换成 'spread'，缝随各自的枚数走（GAP_BY_N）、
         层级系数三轴相同——比值只由「这根轴有几格」决定，不会因为独厚 PP
         一根轴而失衡（三轴同乘 1.8×，比值反而比原来更收拢，不会破 MAX_RATIO
         这条不变量）。EP 在标准形态里没有自己的轴（折在 DP 里），它的分组
         留白看「EP 聚簇」那个形态（已经是 stepK(..., 'spread') 处理）。 */
      std: { sx: stepK('x', PP, 'spread'), sy: stepK('y', REP, 'spread'), sz: stepK('z', TPL, 'spread'), cy: CY },
      // DP 平铺：外维 = 副本宫格（列距 = 板宽 + 留白 · 行距受 2D 约束）· 内维 = 板内 TP 列 / PP 行
      dpt: { gapX: dptCellX, gapZ: dptCellZ, tp: tpStep, tpz: tpStepZ, pp: ppStep, y0: 1.0, cols: TPC, rows: TPD },
      // EP 聚簇：外维 = 桶墙（墙宽 + 块间留白）· 内维 = 墙内 TP 列 · Z = A2A 域（留白层级，域界可读）
      ep: { gapE: blockW + padOf(blockW), tp: tpStep, pp: stepK('y', PP), dom: stepK('z', DOM, 'spread'), cy: CY },
      // TP切片 / PP流水 是「强调类」形态：主轴用 emph 层级（4×）拉开，强调
      // 「墙拉开查同槽位 / 段拉开找慢段」的读法。这个 4× 正好卡在 MAX_RATIO 上，
      // 2D 里主轴会显得稀疏 —— 靠 axBlockFrames 给每块套框把条纹读成整块，不靠压步距
      // （压了这两个形态就没意义了）。
      // 反馈「现在的间距只有 pp 之间有，dp、tp 之间的间距也要拉开分组」——这两个
      // 形态原来只有主轴传 'emph'，另外两根轴沿用默认 'normal'（1×），PP流水
      // 落地就是 rank-topology-lite 没传 ?mode= 时的默认形态（pick 的落地值 4），
      // 盘古预置那种 tp8·rep100 规模下，没升级层级的那两根轴全挤成一条看不出
      // 断点的斜线——跟标准形态是同一个毛病、同一个修法：非主轴也升到 'spread'，
      // 差数量级但都比 1× 松，读者能看出"这也是分着组的"，又不会跟 4× 的主轴
      // 抢视觉重心（4× 定义的就是"主轴该多显眼"，两次遇到的从来不是它）。
      tps: { gapT: stepK('x', TPL, 'emph'), pp: stepK('y', PP, 'spread'), rep: stepK('z', REP, 'spread'), cy: CY },
      /* 反馈「按照分组拉开间距」「各个切分组之间拉开间距」——moe718b128k 那档
         CP=16，PP流水的 Y 轴摆的是 TP×CP 合并 lane（=128），上面那条 'spread'
         只把 128 个 lane 的步距整体调松，并没有在"这是第几个 CP 组"这件事上
         留出断点：128 条紧挨着的横线，肉眼看不出是 16 组各 8 条，跟"只有 PP
         之间有间距"是同一类问题，只是换了一根轴。这里把 Y 轴从"128 个 lane
         的单一步距"改成"CP 外层分组（块高度+留白）+ TP 内层步距"的复合轴，
         做法照抄 EP 聚簇形态 X 轴的 gapE（桶墙外维 + tp 内维）那一套——tpIn 是
         组内每条 TP 的步距，cpGap 是组与组之间的缝（组高度 + padOf），CP=1
         的预置（tpIn 退化成单组）跟改动前逐位相同。 */
      ppf: { gapP: stepK('x', PP, 'emph'), tpIn: stepK('y', TP, 'spread'),
        cpGap: TP * stepK('y', TP, 'spread') + padOf(TP * stepK('y', TP, 'spread')),
        rep: stepK('z', REP, 'spread'), cy: CY },
      /* 物理平铺：不看任何并行分组，只回答「这张卡插在机房哪个槽位」——X=host 内卡位(slot)
         · Z=host 序号 · Y 恒 0（各形态里唯一不叠高度的一种，真摊平，不是「压扁的立方」）。
         host 数一多会排成一条极长的线，超过 64 台折成 hgx×hgz 近方格（同 DP 平铺的折法）；
         典型工况（≤64 host，如 16 机×8 卡=128 张）不折，摊成整间机房本来的样子——
         这正是它要回答的问题本身，折了反而看不出「插在哪」。 */
      phys: (() => {
        const foldHost = HOSTS > 64;
        const hgx = foldHost ? Math.max(1, Math.ceil(Math.sqrt(HOSTS))) : 1;
        const hgz = foldHost ? Math.ceil(HOSTS / hgx) : HOSTS;
        const slot = stepK('x', CPH);
        const hostBlockW = CPH * slot;
        return { hgx, hgz, slot, gapX: hostBlockW + padOf(hostBlockW), gapZ: stepK('z', hgz) };
      })(),
    };

    // 5 种形态的 rank → 世界坐标（out 为 {x,y,z} 或 THREE.Vector3 均可）
    function posOf(r, mode, out) {
      out = out || { x: 0, y: 0, z: 0 };
      const tp = laneOf(r), pp = ppOf(r), rep = repOf(r);   // tp 在这里是**轴上的格号**（含 CP）
      if (mode === 1) {          // DP 平铺：副本宫格，每副本一块 TP(列×排)×PP 的板（找慢副本）
        const s = SP.dpt, tc = tp % TPC, td = (tp / TPC) | 0;
        out.x = (gxOf(r) - cG) * s.gapX + (tc - (TPC - 1) / 2) * s.tp;
        out.y = s.y0 + (PP - 1 - pp) * s.pp;
        out.z = (gzOf(r) - cZ) * s.gapZ + (td - (TPD - 1) / 2) * s.tpz;
        return out;
      }
      if (mode === 2) {          // EP 专家桶墙：桶成墙（同墙=持有相同专家）· 墙内 X=TP 列 · Y=PP · Z=A2A 域
        const s = SP.ep;
        out.x = (epOf(r) - cE) * s.gapE + (tp - cT) * s.tp;
        out.y = s.cy + (cP - pp) * s.pp;
        out.z = (domOf(r) - cD) * s.dom;
        return out;
      }
      if (mode === 3) {          // TP 切片：权重墙沿 X 拉开，一面墙=全集群同槽位切片
        const s = SP.tps;
        out.x = (tp - cT) * s.gapT;
        out.y = s.cy + (cP - pp) * s.pp;
        out.z = (rep - cR) * s.rep;
        return out;
      }
      if (mode === 4) {          // PP 流水：段横向展开成流水线（找慢段/气泡）
        const s = SP.ppf;
        out.x = (pp - cP) * s.gapP;
        // Y 轴拆成两层：CP 外层分组（cpGap）+ TP 内层步距（tpIn），CP=1 时
        // cpOf(r) 恒为 0，退化成单组，与改动前 (tp - cT) * s.tp 逐位相同。
        out.y = s.cy + (cpOf(r) - (CP - 1) / 2) * s.cpGap + (tpOf(r) - (TP - 1) / 2) * s.tpIn;
        out.z = (rep - cR) * s.rep;
        return out;
      }
      if (mode === 5) {          // 物理平铺：不看并行分组，只看插在机房哪个槽位——Y 恒 0，真摊平
        const s = SP.phys, h = hostOf(r), sl = slotOf(r) % CPH;
        const hx = s.hgx > 1 ? h % s.hgx : 0;
        const hz = s.hgx > 1 ? (h / s.hgx) | 0 : h;
        out.x = (hx - (s.hgx - 1) / 2) * s.gapX + (sl - (CPH - 1) / 2) * s.slot;
        out.y = 0;
        out.z = (hz - (s.hgz - 1) / 2) * s.gapZ;
        return out;
      }
      const s = SP.std;          // 标准：X=PP（左→右 Stage0→末） · Y=DP（上→下 DP0→末） · Z=TP
      out.x = (pp - cP) * s.sx;
      out.y = s.cy + (cR - rep) * s.sy;
      out.z = (tp - cT) * s.sz;
      return out;
    }

    // 正交 2D 被折叠的「深度」维（顶↓Y · 前↓Z · 侧↓X），随形态不同 —— 对齐 cockpit ODEP 表
    const depthDims = {
      tp: { n: TPL, lab: CP > 1 ? 'TP×CP' : 'TP' }, pp: { n: PP, lab: 'PP' }, rep: { n: REP, lab: 'DP' },
      ep: { n: EP, lab: '专家桶' }, dom: { n: DOM, lab: 'A2A域' },
      gx: { n: COLS, lab: '副本列' }, gz: { n: ROWS, lab: '副本行' },
      tpc: { n: TPC, lab: '板内TP列' }, tpd: { n: TPD, lab: '板内TP排' },
      cp: { n: CP, lab: 'CP段' },
      host: { n: HOSTS, lab: 'Host' }, slot: { n: CPH, lab: '槽位' },
    };
    const depthIdxOf = (r, dim) => dim === 'tp' ? laneOf(r) : dim === 'pp' ? ppOf(r)
      : dim === 'rep' ? repOf(r) : dim === 'ep' ? epOf(r) : dim === 'dom' ? domOf(r)
        : dim === 'gx' ? gxOf(r) : dim === 'gz' ? gzOf(r)
          : dim === 'cp' ? cpOf(r)
            : dim === 'tpc' ? laneOf(r) % TPC : dim === 'tpd' ? (laneOf(r) / TPC) | 0
              : dim === 'host' ? hostOf(r) : dim === 'slot' ? (slotOf(r) % CPH) : 0;

    /* 视角收编的判据是**逐屏**的，不是逐形态的：
       一个形态之所以不同于标准，全在它那根 emph 轴（TP切片的墙、PP流水的段，步距 4×）。于是——
         · emph 轴**留在屏幕上**的那两屏（顶 / 前）：墙与墙之间是看得见的空档，
           「这是第几片 / 第几段」在这里才数得清 → 保留；
         · emph 轴**被折进视线**的那一屏（两个形态都是侧视）：空档随之消失，剩下的步距
           签名与标准的同名平面逐位相同（0.730 × 1.200）→ 这一屏就是标准，不重复给出。
       早先「与标准共享三轴 ⇒ 三平面全重合」是把这条判据用到了整个形态上，收多了：
       PP流水 甚至换了轴分配（X=PP · Y=TP），顶/前两屏是标准根本没有的平面。
       剩下的真问题是 emph 步距会让 2D 散成稀疏条纹，交给块框（axBlockFrames）解决。
       bestView = 这个形态「最该看的一屏」——由它要回答的问题决定，不是由好不好看决定。 */
    /* 视角 → 被折进视线的维（可多个）。两套：
         D_STD  标准的 X=PP · Y=DP · Z=TP  →  顶折 DP · 前折 TP · 侧折 PP
         D_TPS  TP切片的 X=TP · Y=PP · Z=DP →  顶折 PP · 前折 DP · 侧折 TP */
    const D_STD = { 1: ['rep'], 2: ['tp'], 3: ['pp'] };
    const D_TPS = { 1: ['pp'], 2: ['rep'], 3: ['tp'] };
    const modes = [
      {
        /* 标准不是「第五种形态」，是前四种的**基准投影**。它自己不回答任何问题——
           另外四个各冲着一个问题去（慢段 / 同槽位坏件 / 桶故障 / 慢副本），它没有。
           留着它是因为三件具体的事，文案就直说这三件，别再讲含糊的「位置即多维坐标」：
             ① 对照系：三根轴地位相同、都是常规密排，「换形态只是换投影轴」才有个基准；
             ② 三个密排 2D 平面的归属地：别的形态把自己那根 emph 轴折进视线时，那一屏
                就退回成标准的同名平面（TP切片/PP流水 的侧视因此收编掉，不重复给）；
             ③ 整网层联动的入口：selectLayer() 的紫色切片只在这个坐标系里成立。 */
        key: 'std', name: '标准', short: '标准',
        sub: `标准 · 基准投影 X=PP(模型深度) Y=DP Z=TP`,
        why: `前四个形态的基准投影，自己不查问题 · 三根轴地位相同（换形态只是换投影轴）· 三个密排 2D 平面的归属地 · 整网层联动的入口`,
        viewLabels: { 1: '顶 PP-TP 面', 2: '前 PP-DP 面', 3: '侧 TP-DP 面' }, depth: D_STD,
        // 三根轴地位相同，3D 才是它本身；2D 三屏都是它的
        views: [0, 1, 2, 3], bestView: 0,
      },
      {
        key: 'dpt', name: 'DP平铺', short: 'DP',
        sub: `DP 平铺：${REP} 副本各自成板（找慢副本）`,
        why: `副本间只在步末做梯度 AllReduce · 发暗/掉队的那块板 = 慢副本`,
        viewLabels: { 1: '顶 副本网格', 2: '前 列-PP 面', 3: '侧 行-PP 面' },
        // 板内 TP 折成「列×排」后板有了厚度：顶视每个副本是一片瓦（而非一条线），
        // 侧视也不再塌陷 → 三个正交视角都成立。
        depth: { 1: ['pp'], 2: ['gz', 'tpd'], 3: ['gx', 'tpc'] },
        // 找慢副本 = 扫一遍整个宫格 → 顶视一屏看全 100 块板
        views: [0, 1, 2, 3], bestView: 1,
      },
      {
        key: 'ep', name: 'EP聚簇', short: 'EP',
        sub: `EP 聚簇：${EP} 专家桶成墙（桶=MoE 组 · 每桶复现于 ${DOM} 个 A2A 域 · 桶↔卡非 1:1）`,
        why: `桶故障 = 整面墙同红 · 域热点 = 横穿 ${EP} 墙的一排过热 · 桶↔卡非 1:1`,
        viewLabels: { 1: '顶 桶-域 面', 2: '前 桶-PP 面', 3: '侧 域-PP 面' },
        depth: { 1: ['pp'], 2: ['dom'], 3: ['ep', 'tp'] },   // 侧视同时折叠墙序与墙内 TP（域数多，仍成阵）
        // 桶故障（整墙）与域热点（横穿一排）是两个正交方向，只有顶视同屏放得下
        views: [0, 1, 2, 3], bestView: 1,
      },
      {
        key: 'tps', name: 'TP切片', short: 'TP',
        sub: `TP 切片：${TP} 片权重墙 · 一面墙=全集群同槽位切片（查同槽位系统性故障）`,
        why: `同槽位系统性故障（整批同号卡坏件）= 一面墙集体异常`,
        viewLabels: { 1: '顶 DP-TP 面', 2: '前 TP-PP 面' }, depth: D_TPS,
        // 侧视折掉 TP（本形态的 emph 轴）→ 剩下的 DP×PP 密排与标准前视是同一批格子
        // （只差一次 90° 转），不重复给。
        // 「同槽位系统性故障」要在全集群范围内看，DP 轴必须在场 → 顶视是这一形态的主屏：
        // 一整列红 = 那个槽位的卡整批坏。
        views: [0, 1, 2], bestView: 1,
      },
      {
        key: 'ppf', name: 'PP流水', short: 'PP',
        sub: `PP 流水：${PP} 段横向展开 · 左=Stage0 右=Stage${PP - 1}（找慢段/气泡）`,
        why: `只有 PP 适合说「哪段层在哪」· ${PP} 段各 ${LPS} 层 · 慢段拖住下游 = 右侧板变暗 · 空档=bubble`,
        viewLabels: { 1: '顶 DP-PP 面', 2: '前 PP-TP 面' },
        depth: { 1: ['tp'], 2: ['rep'] },
        // 侧视折掉 PP（本形态的 emph 轴）→ 退回标准已有的 DP-TP 平面，不重复给。
        // 「哪一段慢」里段身份就是答案、DP 是噪声 → 前视把 100 副本折进每格，
        // 5 列一字排开，慢段=整列偏暗，是这一形态的主屏。
        views: [0, 1, 2], bestView: 2,
      },
      {
        /* 物理平铺：前五个形态问的都是「归哪个并行组」，这一个故意不问——只回答
           「这张卡插在机房哪个槽位」，是并行分组之外的第六个正交问题（rank-segmentation-observability）。
           别的形态换个并行配置，卡就换一批邻居；这一个换任何并行配置，同一张卡永远插在
           同一个槽位——它是最稳定的那个坐标系，切分再怎么变，物理位置不跟着变。 */
        key: 'phys', name: '物理平铺', short: '物理',
        sub: `物理平铺：${HOSTS} 台机 × ${CPH} 卡摊成一张机房俯视图（不看并行分组，只看插在哪）`,
        why: `不问「归哪个并行组」，只问「这张卡插在机房哪个槽位」· 单层摊平（Y 恒 0，各形态里唯一不叠高度的一种）`,
        viewLabels: { 1: '顶 机房俯视（Host×槽位）' },
        depth: { 1: [] },
        // Y 恒为 0（真摊平），3D 与顶视看到的是同一份东西 → 顶视本身就是「最该看的一屏」，
        // 前/侧两屏会把仅有的两根轴之一直接压成一条线，没有信息量，不给。
        views: [0, 1], bestView: 1,
      },
    ];

    // 四维通信组（选中 rank 的对端）——语义与 cockpit activePeerChips 一致
    function commGroup(r, dim) {
      const tp = tpOf(r), pp = ppOf(r), rep = repOf(r), cp = cpOf(r), out = [];
      /* 每个通信域 = 固定其余坐标、只让这一维跑遍。**cp 必须一起固定**，否则
         TP/PP/DP/EP 组会把别的上下文段的卡也算进来——那不是同一个 communicator。 */
      if (dim === 'TP') { for (let t = 0; t < TP; t++) out.push(rankOf(t, pp, rep, cp)); }
      else if (dim === 'CP') { for (let c = 0; c < CP; c++) out.push(rankOf(tp, pp, rep, c)); }
      else if (dim === 'PP') { for (let p = 0; p < PP; p++) out.push(rankOf(tp, p, rep, cp)); }
      else if (dim === 'DP') {                       // 同位副本（全量 AllReduce·显示采样）
        const step = Math.max(1, REP >> 4);
        for (let d = 0; d < REP; d += step) out.push(rankOf(tp, pp, d, cp));
      } else {                                       // EP：A2A 域内同位 rank（每桶各出 1 员互发）
        const d0 = domOf(r) * EP;
        for (let e = 0; e < EP; e++) out.push(rankOf(tp, pp, d0 + e, cp));
      }
      return out;
    }

    // 各形态包围盒（轴标注/取景用）
    const boundsCache = {};
    function boundsOf(mode) {
      if (boundsCache[mode]) return boundsCache[mode];
      const b = { x0: 1e9, x1: -1e9, y0: 1e9, y1: -1e9, z0: 1e9, z1: -1e9 };
      const v = { x: 0, y: 0, z: 0 };
      for (let r = 0; r < N; r++) {
        posOf(r, mode, v);
        if (v.x < b.x0) b.x0 = v.x; if (v.x > b.x1) b.x1 = v.x;
        if (v.y < b.y0) b.y0 = v.y; if (v.y > b.y1) b.y1 = v.y;
        if (v.z < b.z0) b.z0 = v.z; if (v.z > b.z1) b.z1 = v.z;
      }
      return (boundsCache[mode] = b);
    }

    return {
      config: C, TP, CP, TPL, PP, EP, DOM, REP, N, LPS, EXP_PER, COLS, ROWS, TPC, TPD, SP, CARD,
      tpOf, cpOf, laneOf, ppOf, repOf, epOf, domOf, gxOf, gzOf, rankOf,
      stageLayerRange, expRange, posOf, boundsOf,
      modes, depthDims, depthIdxOf, commGroup,
      /* ── 整网对象透镜 ──
         objCarry：这张卡承载这个对象吗（PP 首/末段专属的对象只落在那一段）；
         objShard：这张卡持有的是第几片、区间是什么（通信算子没有「片」，返回 null）。 */
      netObjects: NET_OBJ, netObjBands: OBJ_BAND, netObjBy: NET_OBJ_BY,
      objCarry(id, r) {
        const o = NET_OBJ_BY[id]; if (!o) return true;
        return o.comm ? true : o.carry(this, r);
      },
      objShard(id, r) {
        const o = NET_OBJ_BY[id]; if (!o || o.comm || !o.shard) return null;
        return o.shard(this, r);
      },
      /* 一个对象可能被**多维同时切**（attn_core = TP head × CP ctx）。
         objShard 只给主导那一维（着色与主屏要一个确定的维）；要完整表达就用这个。
         单维对象退化成 [shard]，调用方不必分情况。 */
      objShards(id, r) {
        const o = NET_OBJ_BY[id]; if (!o || o.comm) return [];
        if (o.shards) return o.shards(this, r) || [];
        const one = o.shard ? o.shard(this, r) : null;
        return one ? [one] : [];
      },
      // 物理落位
      placement: { cardsPerHost: CPH, hostsPerPod: HPP, cardsPerPod: CPP, hosts: HOSTS, pods: PODS },
      hostOf, podOf, tierOf,
      hotBuckets: new Set((C.hotBuckets || []).filter((e) => e < EP)),
    };
  }

  /* ════════════════════════ 渲染器 ════════════════════════ */
  function mount(container, opts) {
    opts = opts || {};
    const THREE = global.THREE;
    if (!THREE) throw new Error('PtoRubikCubePattern.mount 需要 window.THREE（three r128）先行加载');
    // 模型可整体重建（工具栏「并行」输入排 / setConfig API 自由改维度）：
    // 维度快照用 let + syncDims 同步，mount 内所有引用自动跟随新配置。
    let model = createModel(opts.config);
    let TP, CP, TPL, PP, EP, DOM, REP, N, LPS;
    const syncDims = () => { ({ TP, CP, TPL, PP, EP, DOM, REP, N, LPS } = model); };
    syncDims();

    /* ── 状态 ── */
    /* 落地形态 = PP流水（4），不是标准。四个形态各回答一个具体问题，标准是它们的基准投影
       ——开门第一眼给一个**能读出东西**的形态，比给一个「全都在这儿」的对照系更有用；
       形态按钮也是按 PP·TP·EP·DP·标准 排的，落地在第一个，选中态就在最左边。 */
    const DEFAULT_MODE = 4;
    const S = {
      mode: opts.mode == null ? DEFAULT_MODE : opts.mode | 0,
      view: 0,                       // 0=轴测 · 1=顶 · 2=前 · 3=侧
      /* 正交剖面**默认开在第 0 层**：2D 一进来每格就只有一张卡，所见即所得；
         关掉才是「把 N 张叠在一起看」。反过来（默认关）会让人第一眼就吃一个折叠，
         还得先发现有这么个开关——把需要解释的那个状态放在默认，是反的。
         3D 下 curDepth() 为空，这个状态不起作用，也不显示控件。 */
      sliceOn: true, sliceVal: 0,
      colorBy: 'load',               // load | tp | pp | dp | ep | host | pod（后两个 = 物理落位透镜）
                                      // | w | act | grad | opt（内存构成透镜，见 memShare）
      anom: 'none',                  // none | tp | pp | dp | ep（异常注入 → 「异常的形状」）
      playing: true,
      theme: opts.theme === 'light' ? 'light' : 'dark',
      sel: null, hover: null,        // 选中/悬停 rank
      // 连线图层（每项都可单独关闭）与集合算法。focus=选中聚焦：与选中卡无关的卡压暗
      wire: { members: true, lines: true, outline: true, movers: true, focus: true, payload: true, board: false, nest: true },
      algo: 'auto',                  // auto（按维选原语）/ ring / tree

      /* 整网对象透镜：选中一个整网对象（算子/模块/HCCL 算子）→ 承载它的卡亮起、
         按「持有第几片」着色，其余退成背景。与异常注入同一个约定——**它接管着色**，
         而不是再加一种并列的着色镜头（否则两套颜色叠在一起，卡色读不准）。 */
      obj: null,                     // null | NET_OBJ 的 id
      selEdge: null,                 // 选中的通信边（C 档：宿主据此点亮物理链路）
      more: false,                   // 工具栏抽屉（着色/注入/连线/时间/并行）是否展开
      selLayer: null,                // 整网层 → 魔方水平切片（整网图联动挂点）
      t: 0,
    };
    const isDark = () => S.theme !== 'light';
    const themeC = (dark, light) => (isDark() ? dark : light);
    /* 设计系统 token → 具体色值（切主题时重读；3D 材质只能吃具体色值） */
    const TOK = {};
    function readTokens() {
      let cs = null;
      try { cs = getComputedStyle(root); } catch (e) { /* noop */ }
      TOKEN_KEYS.forEach((k) => { TOK[k] = cs ? (cs.getPropertyValue(k) || '').trim() : ''; });
    }
    // token → 不透明色（半透明 token 先按 --background 合成，Three 的材质吃不了 alpha 色）
    function tokRGB(key, fallback) {
      const c = cssRGBA(TOK[key] || fallback || '#808080');
      if (c.a >= 0.999) return c;
      const bg = cssRGBA(TOK['--background'] || '#101010');
      return { r: bg.r + (c.r - bg.r) * c.a, g: bg.g + (c.g - bg.g) * c.a, b: bg.b + (c.b - bg.b) * c.a, a: 1 };
    }
    const tokHex = (key, fallback) => { const c = tokRGB(key, fallback); return '#' + hex2(c.r) + hex2(c.g) + hex2(c.b); };
    /* 素色镜头下，维度签名色（TP 青 / PP 橙 / DP 蓝 / EP 紫）连同它们画出来的东西
       ——轴刻度、块分隔线、块框、坐标轴说明、悬停卡的标题色——一起收成灰阶。
       只改这一个出口就够了：整个 3D 标注体系的颜色都是从 dimc() 要来的。
       四维仍然各给一格明度（不是同一个灰），于是"这是 TP 轴还是 DP 轴"在同屏
       对照时还分得开，只是不再靠色相。卡阵本体的素色在 colorOfRank 里，那是
       另一条路（同一个 S.colorBy 开关）。 */
    const DIM_MONO = { TP: '#C8C8C8', SP: '#C8C8C8', CP: '#9A9A9A', PP: '#E2E2E2', DP: '#AEAEAE', EP: '#868686' };
    const dimc = (d) => (S.colorBy === 'neutral' && DIM_MONO[d]) ? DIM_MONO[d] : tokHex(DIM_TOKEN[d]);
    const tierc = (k) => tokHex(TIER_TOKEN[k]);          // 物理链路层级色（同机 / Pod 内 / 跨 Pod）
    const groupColor = (i) => tokHex(GROUP_TOKENS[i % GROUP_TOKENS.length]);
    /* 算子族色（OPV）刻意不认素色开关：反馈原话"卡片还是保留彩色"——卡内
       魔方那圈小方块与装载清单的色点是"结构长什么样"的视觉本体，跟维度
       签名色（TP/PP/DP/EP，答的是"被哪一维切"）不是一回事，素色开关收的
       是后者。这一路一度也接上了同一个开关（famc()，已撤销），选了素色
       之后这几处会跟着退灰——用户随后反馈要把它们的颜色留住，只去掉画布
       里挤在一起读不出来的那些字牌（见 cclabels / axisLabelsOnSelect）。 */

    /* ── DOM 骨架 ── */
    const root = document.createElement('div');
    root.className = 'prc-root';
    root.setAttribute('data-theme', S.theme);
    /* brandName：默认还是"逻辑魔方"（这个 pattern 自己的名字）——只有
       显式传了 opts.brandName 的宿主（目前只有 rank-topology-lite，见它
       自己的 ?brand= 桥接）才会换成别的名字，独立打开 /rubik-pattern.html
       或别处嵌入这份 pattern 都不受影响。这里插进 innerHTML 的字符串，
       跟文件别处的 esc() 一个手法，自己转义一遍，不信任调用方传干净的。 */
    const brandNameHtml = String(opts.brandName || '逻辑魔方').replace(/&/g, '&amp;').replace(/</g, '&lt;');
    root.innerHTML = [
      '<div class="prc-stage"></div>',
      opts.chrome === false
        /* chrome:false 收起的是"操作面板那一整套"（形态/视角/更多抽屉/图例/
           选中卡侧栏 .prc-info）——不是"这张图叫什么"这句话。宿主（比如
           rank-topology-lite）关掉 chrome 之后往往自己接了面包屑逻辑
           （opts.brandTierLabel、选中后追加 rank N），brandname 这个挂点没了，
           syncBrand() 就成了改一个不存在的元素，读者也看不出选中的是哪张卡、
           换到第三档时题面还断了一截（反馈"标题没了？"）。brandname 单独摘出来，
           裸文字浮在左上角——没有 prc-topbar 那张卡包着，也不占用任何操作面板
           的空间，其余控件照常一个都不画。 */
        ? '<div class="prc-brandname is-bare">' + brandNameHtml + '</div>'
        : [
        // 常驻只留「形态 / 视角」——它们决定画面本身怎么摆；其余（着色/注入/连线/时间/
        // 并行）是筛选与工况，收进一个可开合的抽屉，默认收起，画面因此干净。
        // 顶栏（对齐设计系统 sidecar 的页头）：左边是这张图叫什么 + 规格小签，
        // 右边是配置（形态 / 视角两组互斥控件 + 「更多」抽屉）。
        '<div class="prc-topbar">',
        '  <div class="prc-brandname">' + brandNameHtml + '</div>',
        '  <div class="prc-tools">',
        '    <span class="prc-group segmented-control prc-row-modes"></span>',
        '    <span class="prc-group segmented-control prc-row-views"></span>',
        /* 剖面跟在「视角」后面，同一排、不另起一行：它讲的是「这一屏被折掉的那一维翻到
           第几层」——是视角的下游，紧挨着才读得出这层关系；另起一行/另做一张浮卡都会让它
           看起来是与视角平级的第三种设置。说明（折了几张、正翻哪一层）收进它自己的问号
           气泡，与形态/视角两组同一套做法，所以这一排不长个子。 */
        '    <span class="prc-group prc-row-slice"></span>',
        '    <span class="prc-timewrap">',
        '      <button class="prc-playbtn btn btn-sm prc-iconbtn" type="button"></button>',
        '      <div class="prc-timepop panel-shell"><span class="prc-lab">时间</span></div>',
        '    </span>',
        '    <button class="prc-morebtn btn btn-sm prc-iconbtn" type="button"></button>',
        // 宿主控件（主题切换等）的插槽：放进同一张卡，按钮才成套
        '    <span class="prc-toolslot"></span>',
        '  </div>',
        '</div>',
        '<div class="prc-more panel-shell">',
        '  <div class="prc-row prc-row-obj"><span class="prc-lab">对象</span></div>',
        '  <div class="prc-row prc-row-lens"><span class="prc-lab">着色</span></div>',
        '  <div class="prc-row prc-row-anom"><span class="prc-lab">注入</span></div>',
        '  <div class="prc-row prc-row-wire"><span class="prc-lab">连线</span></div>',
        '  <div class="prc-row prc-row-cfg"><span class="prc-lab">并行</span></div>',
        '</div>',
        '<div class="prc-legend panel-shell"></div>',
        '<div class="prc-info panel-shell"></div>',
      ].join(''),
      '<div class="prc-tip"></div>',
    ].join('');
    container.appendChild(root);
    readTokens();   // 挂进文档后才能解析 token（detached 元素读不到 computed style）
    const $ = (sel) => root.querySelector(sel);
    const stageEl = $('.prc-stage'), tipEl = $('.prc-tip');
    /* 招牌名字后面跟一句"选中的是谁"（?brand= 的宿主才用得上这个后缀，
       独立打开时 opts.brandName 是空，brandBase 就是"逻辑魔方"，跟改动前
       一样不带任何后缀）——面包屑挂在名字后面，不再单独占一块地。
       分隔符用 "/" 不用 "·"：反馈「都放成面包屑用/分隔」，跟矩阵那边
       stitle 的写法（见 rank-topology-lite/pattern.js 的 matrixSrcFor）
       对齐，两块题面换形态时读起来是同一套语法。
       opts.brandTierLabel（?tierlabel= 转进来的）：选中之后插在"模型名"和
       "rank N"中间的第三段面包屑，比如 rank-topology-lite 的第二档"选中+
       兄弟"——反馈「面包屑应该是3层」「这一层没有对应的面包屑」：宿主有
       自己的取景档位概念（1 集群 / 2 选中+兄弟 / 3 单卡下钻），只写模型名
       和 rank 号，中间那一档在面包屑里直接被跳过了，读者看不出"选中+
       兄弟"这一步存在过。这个名字是宿主自己的取景语汇，逻辑魔方不该替
       它编，所以走 opts 传入；默认不传，行为跟改动前一样，只有一段
       "模型名 / rank N"。 */
    const brandEl = $('.prc-brandname');
    const brandBase = String(opts.brandName || '逻辑魔方');
    const brandTier = opts.brandTierLabel ? String(opts.brandTierLabel) : '';
    function syncBrand() {
      if (!brandEl) return;
      brandEl.textContent = brandBase
        + (S.sel != null ? (brandTier ? ' / ' + brandTier : '') + ' / rank ' + S.sel : '');
    }

    /* ── three 场景 ── */
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(global.devicePixelRatio || 1, 2));
    stageEl.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -500, 1000);
    /* 打光总增益必须 ≈ 1：MeshStandard 的出射色 = 材质色 ×(环境 + 平行×N·L)，
       原来 0.85+0.55 意味着朝光的顶面拿到 1.4× —— token 里本就高彩度的色一乘就顶到
       通道上限，几百张卡铺满屏幕时集体读成荧光。改成 环境 0.74 + 主光 0.32 + 背光 0.10：
       顶面 ≈1.0（所见即 token 色），立面 ≈0.85，背面靠背光托住不发死，
       三个可见面仍差出 ~15% 的明度阶梯，立体感不丢。 */
    scene.add(new THREE.AmbientLight(0xffffff, 0.74));
    const dl = new THREE.DirectionalLight(0xffffff, 0.32); dl.position.set(18, 30, 12); scene.add(dl);
    const dlFill = new THREE.DirectionalLight(0xffffff, 0.10); dlFill.position.set(-14, 6, -18); scene.add(dlFill);

    const V3 = (x, y, z) => new THREE.Vector3(x, y, z);
    const dummy = new THREE.Object3D(), cTmp = new THREE.Color();

    // 卡阵列：InstancedMesh，1 小块 = 1 卡（rank）。维度改变时整体重建（buildField）。
    const BOXG = new THREE.BoxGeometry(CARD.x, CARD.y, CARD.z);   // 卡块尺寸 = 布局规则的 CARD（恒定）
    const boxMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.55, metalness: 0.02 });
    let chips = null;
    let cur, target, scl;
    let settling = true;
    // 卡块尺寸全形态恒定（用户约定：换形态不改变一张卡的大小）——各形态布局的格步距
    // 均按「装得下固定卡块」设计（见 SP 注释），无需按形态缩放。
    function buildField() {
      if (chips) { scene.remove(chips); if (chips.dispose) chips.dispose(); }
      chips = new THREE.InstancedMesh(BOXG, boxMat, N);
      chips.frustumCulled = false;
      scene.add(chips);
      // 位置缓冲：cur → target 飞行 lerp（切形态的重排动画）
      cur = new Float32Array(N * 3); target = new Float32Array(N * 3); scl = new Float32Array(N);
      const v = { x: 0, y: 0, z: 0 };
      for (let r = 0; r < N; r++) {
        model.posOf(r, S.mode, v);
        cur[r * 3] = v.x; cur[r * 3 + 1] = v.y; cur[r * 3 + 2] = v.z;
        target[r * 3] = v.x; target[r * 3 + 1] = v.y; target[r * 3 + 2] = v.z;
        scl[r] = 1;
      }
      settling = true;
    }
    buildField();
    function retarget() {
      const v = { x: 0, y: 0, z: 0 };
      for (let r = 0; r < N; r++) {
        model.posOf(r, S.mode, v);
        target[r * 3] = v.x; target[r * 3 + 1] = v.y; target[r * 3 + 2] = v.z;
      }
      settling = true;
    }

    // 焦点/悬停/关联标记
    function edgeBox(color) {
      // 焦点线框跟着卡块尺寸走（CARD 是正方体）——沿用旧的各向异性尺寸会框成扁盒子，
      // 与卡对不上、也更难看清选中的是哪一张。
      const g = new THREE.EdgesGeometry(new THREE.BoxGeometry(CARD.x * 1.34, CARD.y * 1.34, CARD.z * 1.34));
      return new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95, depthTest: false }));
    }
    const hovBox = edgeBox(0x9ecbff);
    hovBox.visible = false; hovBox.renderOrder = 7;
    scene.add(hovBox);
    /* 选中标记 = 卡自己的颜色做的「涟漪呼吸灯」，不是外框。
       深色细框在浅底上又硬又抢，而且和卡是两种语言；改成两样东西：
         · 光晕：一个比卡略大的盒子，只渲染背面（BackSide）——于是它只在卡的四周
           透出一圈光，不会盖在卡的正面上（正面一旦被盖，卡自己的着色就读不准了）；
         · 涟漪：两圈线框从卡向外扩散并淡出，错相位循环，像水波一样一圈接一圈。
       两者的颜色都取「这张卡此刻的颜色」（负载/分组/异常都跟着变），所以它读起来是
       这张卡在发光，而不是有人在它外面套了个框。 */
    const selColor = new THREE.Color();
    const selHalo = new THREE.Mesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.3, side: THREE.BackSide, depthWrite: false, depthTest: false }));
    selHalo.renderOrder = 4; selHalo.visible = false; scene.add(selHalo);
    const selRipples = [0, 1].map(() => {
      const m = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
        new THREE.LineBasicMaterial({ transparent: true, opacity: 0.6, depthWrite: false, depthTest: false }));
      m.renderOrder = 7; m.visible = false; scene.add(m);
      return m;
    });
    let selColorAt = -1;
    function updateSelFx(nowMs) {
      const on = S.sel != null && S.sel < N;
      selHalo.visible = on; selRipples.forEach((m) => { m.visible = on; });
      if (!on) return;
      if (nowMs - selColorAt > 200) {              // 颜色跟着卡走（热力随阶段变），不必每帧算
        selColorAt = nowMs;
        selColor.copy(colorOfRank(S.sel));
        // 太暗的卡（暗色主题的低负载）给一点提亮，光晕才透得出来
        const hsl = {}; selColor.getHSL(hsl);
        if (hsl.l < 0.42) selColor.setHSL(hsl.h, Math.min(1, hsl.s * 1.1), 0.52);
        selHalo.material.color.copy(selColor);
        selRipples.forEach((m) => m.material.color.copy(selColor));
      }
      const p = V3(cur[S.sel * 3], cur[S.sel * 3 + 1], cur[S.sel * 3 + 2]);
      // 光晕：随呼吸轻微起伏
      const b = 1 + 0.06 * Math.sin(nowMs / 380);
      selHalo.position.copy(p);
      selHalo.scale.setScalar(CARD.x * 2.05 * b);
      selHalo.material.opacity = 0.26 + 0.08 * (b - 1) / 0.06;
      // 涟漪：两圈错相位向外扩散并淡出
      selRipples.forEach((m, i) => {
        const t = ((nowMs / 1500) + i * 0.5) % 1;
        m.position.copy(p);
        m.scale.setScalar(CARD.x * (1.5 + t * 2.6));
        m.material.opacity = 0.55 * (1 - t) * (1 - t);
      });
    }

    // 选中的那一段通信边：加粗重画一根管（点选后要看得见自己点中了哪一段）
    let selEdgeMesh = null;
    function drawSelEdge() {
      if (selEdgeMesh) { scene.remove(selEdgeMesh); selEdgeMesh.geometry.dispose(); selEdgeMesh.material.dispose(); selEdgeMesh = null; }
      const e = S.selEdge;
      if (!e || e.from >= N || e.to >= N) return;
      const p = (r) => V3(cur[r * 3], cur[r * 3 + 1], cur[r * 3 + 2]);
      const a = p(e.from), b = p(e.to);
      const path = new THREE.CurvePath(); path.add(new THREE.LineCurve3(a, b));
      selEdgeMesh = new THREE.Mesh(
        new THREE.TubeGeometry(path, 2, CARD.x * 0.3, 8, false),
        new THREE.MeshBasicMaterial({ color: new THREE.Color(tokHex('--foreground')), transparent: true, opacity: 0.85, depthTest: false }));
      selEdgeMesh.renderOrder = 9; scene.add(selEdgeMesh);
    }

    /* 四维通信组的成员标记。原先是半透明实体盒罩在卡上——盒色与卡色叠在一起，卡自己的
       着色（负载 / 分组 / 异常）就读不准了；改成「只有棱、没有面」的线框，并且**贴着卡的
       外沿**（1.02×）：比卡大一圈时，一串框在等距视角下棱线互相错开、叠成菱形格纹，
       看着像一排箭头——贴合之后它就只是卡自己的描边。谁是成员主要靠聚焦（无关卡压暗），
       这层描边只是补一个「就是这几张」的确认。
       实例上限要盖住最大的通信域（DP 组可达 dp 张卡），否则线连过去的卡有一半没有标记，
       看上去就是「线没连到卡上」。 */
    const PEER_MAX = 1024;
    const peerDims = ['TP', 'PP', 'DP', 'EP'];
    /* 注意：不能用 InstancedMesh + EdgesGeometry —— InstancedMesh 是 Mesh，会把
       EdgesGeometry 的「每两点一条边」当成「每三点一个三角形」来画，卡面上于是浮出
       一个个斜三角。正确做法是一条 LineSegments，把所有成员的框顶点合并进一个缓冲，
       在 rebuildComm 里按当前成员位置重填。 */
    const EDGE_TPL = (() => {
      const g = new THREE.EdgesGeometry(new THREE.BoxGeometry(CARD.x * 1.02, CARD.y * 1.02, CARD.z * 1.02));
      const a = Float32Array.from(g.attributes.position.array);
      g.dispose();
      return a;                                   // 24 个点（12 条棱）的相对坐标
    })();
    const peerMeshes = peerDims.map((d) => {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(PEER_MAX * EDGE_TPL.length), 3));
      geo.setDrawRange(0, 0);
      const m = new THREE.LineSegments(geo,
        new THREE.LineBasicMaterial({ color: new THREE.Color(dimc(d)), transparent: true, opacity: 0.6, depthTest: false }));
      m.frustumCulled = false; m.renderOrder = 5; m.visible = false; scene.add(m);
      return m;
    });
    // 域轮廓：每维一个线框盒（把该组成员整体包起来），穿透方块可见
    const outlineBoxes = peerDims.map((d) => {
      const geo = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1));
      const mat = new THREE.LineBasicMaterial({ color: new THREE.Color(dimc(d)), transparent: true, opacity: 0.6, depthTest: false });
      const box = new THREE.LineSegments(geo, mat);
      box.renderOrder = 6; box.visible = false; scene.add(box);
      return box;
    });
    // 方向粒子：沿「此刻主导维」的走线跑，进度 = 阶段内进度（Ring 前半 RS / 后半 AG）
    const MOVERS = 10;
    const moverGroup = new THREE.Group(); scene.add(moverGroup);
    const moverMeshes = Array.from({ length: MOVERS }, () => {
      const m = new THREE.Mesh(new THREE.SphereGeometry(CARD.x * 0.15, 8, 8),
        new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.95, depthTest: false }));
      m.renderOrder = 8; m.visible = false; moverGroup.add(m); return m;
    });
    let moverPaths = [];
    const _hsl = new THREE.Color(), _h = { h: 0, s: 0, l: 0 };
    // 折线上按参数 s∈[0,1) 取点
    const _mv = new THREE.Vector3();
    function pointOnPath(pts, s) {
      if (pts.length < 2) return pts[0] || _mv.set(0, 0, 0);
      let total = 0; const segLen = [];
      for (let i = 1; i < pts.length; i++) { const l = pts[i].distanceTo(pts[i - 1]); segLen.push(l); total += l; }
      if (total <= 0) return pts[0];
      let want = (s - Math.floor(s)) * total;
      for (let i = 0; i < segLen.length; i++) {
        if (want <= segLen[i]) return _mv.copy(pts[i]).lerp(pts[i + 1], segLen[i] ? want / segLen[i] : 0);
        want -= segLen[i];
      }
      return pts[pts.length - 1];
    }
    const pathLen = (pts) => { let L = 0; for (let i = 1; i < pts.length; i++) L += pts[i].distanceTo(pts[i - 1]); return L; };
    /* 流动点：点数按「路有多长」分配，不是把 10 个点全撒到同一条路上——
       短路（比如 TP=2 的两卡之间只有一格）本来只放得下一个点，撒十个就挤成一条毛毛虫。 */
    function updateMovers() {
      if (!S.wire.movers || !moverPaths.length) { moverMeshes.forEach((m) => { m.visible = false; }); return; }
      const u = phaseU();
      // Ring：RS 段跑一圈、AG 段再跑一圈。快慢按这一阶段的通信节奏给（图元库里
      // 「速率对应带宽·时延」的同一约定）：TP 节点内 UB 最快 · EP 浪涌次之 ·
      // PP 接力常规 · DP 跨 Pod 低频大包最慢。
      const RATE = { TP: 2.2, EP: 1.6, PP: 1.0, DP: 0.65 };
      const t = ((u < 0.5 ? u * 2 : (u - 0.5) * 2) * (RATE[PHASES[phaseIdx()].dim] || 1)) % 1;
      const lens = moverPaths.map((p) => pathLen(p.pts));
      const total = lens.reduce((a, b) => a + b, 0) || 1;
      const SPACING = CARD.x * 3.2;            // 两点之间至少隔三格卡宽，看得出是「一串在跑」
      let k = 0;
      for (let i = 0; i < moverPaths.length && k < MOVERS; i++) {
        const path = moverPaths[i];
        if (!path || path.pts.length < 2 || lens[i] <= 0) continue;
        const byLen = Math.max(1, Math.floor(lens[i] / SPACING));
        const byShare = Math.max(1, Math.round(MOVERS * lens[i] / total));
        const n = Math.min(byLen, byShare, MOVERS - k);
        for (let j = 0; j < n; j++) {
          const m = moverMeshes[k++];
          m.position.copy(pointOnPath(path.pts, (t + j / n) % 1));
          // 「暗点」取线芯同色的深色调（同色系压暗，不是中性黑）：点落在线上，对比来自
          // 它与线芯的明度差，同色系因此既看得清又与整条线和谐。
          _hsl.copy(new THREE.Color(path.color)).getHSL(_h);
          m.material.color.setHSL(_h.h, Math.min(1, _h.s * 1.05), Math.min(_h.l * 0.42, 0.26));
          m.material.opacity = 0.92;
          m.visible = true;
        }
      }
      for (; k < MOVERS; k++) moverMeshes[k].visible = false;
    }

    // 通信线（TubeGeometry 曲线 + 标签）——穿透方块可见
    const commGroupG = new THREE.Group(); scene.add(commGroupG);
    function clearComm() {
      while (commGroupG.children.length) {
        const o = commGroupG.children.pop();
        if (o.geometry) o.geometry.dispose();
        if (o.material) { if (o.material.map) o.material.map.dispose(); o.material.dispose(); }
      }
    }
    /* 走线 = 逐段直线的管（不是样条！CatmullRom 会在控制点之间外扩成弧，成员散布时
       整条线看起来「不连在卡上」——通信是点到点的，线就必须点到点）。
       画法沿用硬件图元库「连线样式 Pattern」的三层结构（hpc-topology-node 图元库 · 第 6 组）：
         ① casing —— 比线芯粗一圈的「底色外套」，把线从它穿过的卡面上剥离出来，
            于是线清楚、卡也没被染色（图元库里是白色 casing，这里取 --background，
            明暗主题各自成立）；
         ② glow core —— 彩色线芯（维度签名色），细而实；
         ③ 流动暗点 —— 见 moverMeshes：沿线跑的小圆点，方向即数据流向，
            速度对应这一阶段的通信节奏。三层里只有 ③ 表达方向，不用箭头。 */
    /* 一段管 = 一条折线。注意：**不能把整条折线交给一个 TubeGeometry**——
       TubeGeometry 内部按 getPointAt（等弧长）采样，长短不一的段会让采样点落不到折线
       拐点上，线于是从卡旁边抄近路（端点仍然精确，只测端点查不出来，这是「线没连到卡上」
       第三次复发的成因）。所以直线折线由 rebuildComm 拆成逐段调用，这里只画两点之间的
       一根直管；外凸弧（长边）本来就没有中间的卡要对齐，可以整条交给曲线。 */
    function commLine(points, color, opacity, r, meta) {
      if (points.length < 2) return null;
      const path = points.length === 2
        ? new THREE.LineCurve3(points[0], points[1])
        : new THREE.CatmullRomCurve3(points.slice(), false, 'catmullrom', 0);
      const seg = Math.max(2, (points.length - 1) * 3), rad = r || 0.08;
      const casing = new THREE.Mesh(
        new THREE.TubeGeometry(path, seg, rad * 2.3, 5, false),
        new THREE.MeshBasicMaterial({ color: new THREE.Color(tokHex('--background')), transparent: true, opacity: opacity * 0.62, depthWrite: false, depthTest: false }));
      casing.renderOrder = 5; commGroupG.add(casing);
      const mesh = new THREE.Mesh(
        new THREE.TubeGeometry(path, seg, rad, 5, false),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthWrite: false, depthTest: false }));
      mesh.renderOrder = 6;
      if (meta) mesh.userData.edge = meta;      // 这段线是谁到谁 → 可被点选，交给宿主去点亮物理路径
      commGroupG.add(mesh);
      return mesh;
    }

    /* ── 字牌（高分辨率圆角 label，随主题）──────────────────────────────────
       排版纪律对齐设计系统的 model-architecture-training-sidecar pattern：
         · 字牌是「跟着场景缩放的几何标注」，因此允许小于 UI 的 12px 下限，但要有下限
           与兜底——这里给 MIN_LABEL_PX 的屏幕像素地板（按当前相机换算世界尺寸），
           兜底是悬停 tooltip 与右上信息卡（固定屏幕尺寸、同样的信息）；
         · 语义色只落在「名字」上，解释性文字用中性次级前景色（两段式字牌），
           annotation 的对比度始终低于模型本身；
         · 字牌绝不盖住它所标注的东西：远离盒子的横幅用中性引线牵回去，而不是压上去。 */
    // 全局字牌字号：整套标注只此一个入口（mount 的 labelPx · setLabelPx() · URL 的 ?label=）
    // ——不给任何单个标记开小灶，否则「调大一点」就得逐处去找。
    const clampLabelPx = (v) => Math.max(9, Math.min(28, Number(v) > 0 ? Number(v) : 16));
    /* 字牌一律**固定屏幕尺寸**：正文默认 16px，和 UI 文字一个量级。
       曾经让字牌「跟着场景缩放」（世界尺寸固定、再加一个像素地板兜底），代价是同一张牌
       在不同规格/不同缩放下大小不一，小的时候读不出、大的时候压卡；为了补救又长出一堆
       机关（按块步距钉宽、量牌高错行、牵引线指回去），越补越绕，引线本身还细到看不见。
       改成固定屏幕尺寸后这些机关全部不需要：牌永远这么大、永远读得出，摆在它标的那块
       旁边就够了。代价是**缩得很远时牌会互相挤**——这是自觉的取舍：宁可挤，不可读不出。 */
    let LABEL_PX = clampLabelPx(opts.labelPx);
    // 随场景缩放的那些字牌的可读下限（贴边的用 LABEL_PX，不吃这条）
    const MIN_LABEL_PX = 11;
    const worldPerPx = () => (2 * cam.half) / Math.max(1, stageEl.clientHeight);
    /* 字牌样式逐条复刻设计系统 sidecar pattern 的 operator-label / stage-label：
         · 极小号 850 字重的 mono（技术标记的语气，不是标题）；
         · 底 = 语义色 10% 兑底色（几乎只是一层薄纸），描边 = 语义色 30%，圆角 ≈0.47em；
         · 字色 = 语义色 68% 兑 --foreground（明暗主题都够暗/够亮，且保住色相）；
         · 底色光晕（text-shadow 的 canvas 等价物）代替不透明大白板，压在卡上也读得清。 */
    const _mA = new THREE.Color(), _mB = new THREE.Color();
    const mixHex = (a, b, t) => '#' + _mA.set(a).lerp(_mB.set(b), t).getHexString();
    /* sub：第二行「规格」——照搬 sidecar 的 stage-label 分工，第一行是这块**是谁**
       （维度签名色），第二行是这块**里面有什么**（中性次级色、更小号、带字距）。
       两行合一张牌，比「一个名字 + 旁边另起一个标记」少一次视线跳转，也不会各自飘走。 */
    function makeLabel(text, color, w, sub) {
      const SS = 4, fontPx = 40, subPx = 27, gapY = 7, padX = 26, padY = 9;
      const segs = Array.isArray(text) ? text : [{ t: String(text), c: color }];
      text = segs.map((x) => x.t).join('');
      const FONT = `850 ${fontPx}px ${TOK['--font-mono'] || "'JetBrains Mono','Fira Code','Consolas',monospace"}`;
      const SUBF = `600 ${subPx}px ${TOK['--font-mono'] || "'JetBrains Mono','Fira Code','Consolas',monospace"}`;
      const meas = document.createElement('canvas').getContext('2d');
      meas.font = FONT;
      const titleW = Math.ceil(meas.measureText(text).width);
      meas.font = SUBF;
      const subW = sub ? Math.ceil(meas.measureText(sub).width) : 0;
      meas.font = FONT;
      const bw = Math.max(titleW, subW) + padX * 2;
      const bh = fontPx + (sub ? gapY + subPx : 0) + padY * 2;
      // 四周留一圈完全空白的画布边（PAD）：贴图的透明像素 RGB 是黑的，线性过滤/多级
      // 渐远纹理会把黑边混进最外一圈——描边若压在画布边界上，圆角处就会渗出黑边。
      // 底与描边都往里缩，非透明像素永远被同色包着，边缘再插值也只会插到底色。
      const PAD = 5;
      const tw = bw + PAD * 2, th = bh + PAD * 2;
      const cv = document.createElement('canvas'); cv.width = tw * SS; cv.height = th * SS;
      const c = cv.getContext('2d'); c.scale(SS, SS);
      const bg = tokHex('--background'), fg = tokHex('--foreground');
      const rr = fontPx * 0.47;
      c.fillStyle = mixHex(bg, color, 0.1);
      c.beginPath(); c.roundRect(PAD, PAD, bw, bh, rr); c.fill();
      c.lineWidth = 1.6; c.strokeStyle = mixHex(bg, color, 0.3);
      c.beginPath(); c.roundRect(PAD + 1, PAD + 1, bw - 2, bh - 2, rr); c.stroke();
      c.font = FONT; c.textAlign = 'left'; c.textBaseline = 'middle';
      c.shadowColor = bg; c.shadowBlur = 7;                 // 底色光晕：压在卡上也读得清
      const titleY = sub ? PAD + padY + fontPx / 2 : th / 2;
      let x = (tw - titleW) / 2;
      segs.forEach((sg) => {
        c.fillStyle = mixHex(fg, sg.c || color, 0.68);
        c.fillText(sg.t, x, titleY);
        x += meas.measureText(sg.t).width;
      });
      if (sub) {
        c.font = SUBF;
        c.fillStyle = mixHex(bg, tokHex('--foreground-secondary'), 0.86);
        c.fillText(sub, (tw - subW) / 2, titleY + fontPx / 2 + gapY + subPx / 2);
      }
      c.shadowBlur = 0;
      const tex = new THREE.CanvasTexture(cv);
      tex.minFilter = THREE.LinearMipmapLinearFilter; tex.magFilter = THREE.LinearFilter; tex.generateMipmaps = true;
      try { tex.anisotropy = renderer.capabilities.getMaxAnisotropy(); } catch (e) { /* noop */ }
      tex.needsUpdate = true;
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
      /* 两种字牌，语气不同、尺寸规则也不同：
           · **贴边的**（轴刻度、分段规格牌）= 图表框架的一部分，语气同 UI ——
             固定屏幕尺寸（LABEL_PX），怎么缩放都这么大、永远读得出；
           · **贴在几何体上的**（3D 里的 TP0 / 桶0 / 列0 / 板内注记）= 场景的一部分，
             语气是标注 —— 世界尺寸固定，于是跟着视角一起放大缩小，始终"长在"它标的
             那块结构上，缩远了自然退成背景，不会喧宾夺主。
         判定：经 axEmit 落到边上的标 fixedPx=true，其余都随场景。 */
      sp.userData.lab = { fontFrac: fontPx / tw, aspect: th / tw, world: w * 0.92 * LS * tw / 512 };
      syncLabelSize(sp);
      sp.material.userData.baseOp = 1;
      return sp;
    }
    function syncLabelSize(sp) {
      const L = sp.userData.lab;
      if (!L) return;
      const px = (n) => n * worldPerPx() / L.fontFrac;      // 正文 n 屏幕像素 → 牌的世界宽
      /* 贴边的按 LABEL_PX 定死；随场景的保持世界尺寸（于是跟视角一起放大缩小），
         但给一个**屏幕像素地板**——缩远了它会一起变小是对的，小到读不出就不对了。
         地板是兜底不是放大器：只在低于下限时才顶上去。 */
      const worldW = sp.userData.fixedPx ? px(LABEL_PX) : Math.max(L.world, px(MIN_LABEL_PX));
      sp.scale.set(worldW, worldW * L.aspect, 1);
      placeEdgeLabel(sp);
    }
    // 贴边标记：坐标 = 锚点 + 外让；外让量按屏幕算并随相机重算，各边留白因此恒等
    function placeEdgeLabel(sp) {
      const E = sp.userData.edge;
      if (!E) return;
      sp.position.copy(E.anchor);
      sp.position[E.axis] += E.sign * worldPerPx() * LABEL_PX * E.k;
    }
    let lastLabHalf = -1, lastLabH = -1;
    function syncLabelSizes(force) {
      const h = stageEl.clientHeight || 0;
      if (!force && cam.half === lastLabHalf && h === lastLabH) return;
      lastLabHalf = cam.half; lastLabH = h;
      axGroup.traverse((o) => { if (o.isSprite) syncLabelSize(o); });
      syncTickDensity();
    }
    /* 疏密随相机实时算：牌占几像素是定值，两枚之间隔几像素随缩放变——
       两者一比就知道该隔几枚出一枚。缩放时因此会自动加密 / 变疏，像正经的图表轴。
       末位永远保留，且与它前一枚保留项不足一个间隔时把那一枚藏掉。 */
    // 世界点 → 屏幕像素（用当前相机真投影，因此 3D 与 2D 同一套算法）
    const _pv = new THREE.Vector3();
    function toPx(p) {
      const w = stageEl.clientWidth || 1, h = stageEl.clientHeight || 1;
      _pv.copy(p).project(camera);
      return { x: (_pv.x * 0.5 + 0.5) * w, y: (-_pv.y * 0.5 + 0.5) * h };
    }
    function syncTickDensity() {
      axSeries.forEach((ser) => {
        const it = ser.items;
        if (it.length < 2) return;
        /* 两枚相邻标记在**屏幕上**隔多远——直接投影量，而不是拿世界距离除以 worldPerPx。
           后者只在 2D 成立：轴测里一根世界轴投影到屏幕会缩短（且随旋转变），
           照 2D 那样算会高估间距、少估该隔几个，标记就撞上了。 */
        const a = toPx(it[0].sp.position), b2 = toPx(it[1].sp.position);
        const pitchPx = Math.hypot(b2.x - a.x, b2.y - a.y);
        const m = Math.max(1, Math.ceil(ser.needPx / Math.max(1e-6, pitchPx)));
        const lastI = it.length - 1;
        it.forEach((x, k) => { x.sp.userData.tickHide = !(k % m === 0) && k !== lastI; });
        // 末位与上一枚保留项挨得太近 → 藏掉上一枚
        for (let k = lastI - 1; k >= 0; k--) {
          if (x0Hidden(it[k])) continue;
          if ((lastI - k) * pitchPx < ser.needPx) it[k].sp.userData.tickHide = true;
          break;
        }
        it.forEach((x) => { if (x.sp.userData.views) x.sp.visible = x.sp.userData.views.indexOf(S.view) >= 0 && !x.sp.userData.tickHide; });
      });
      dropCornerClashes();
    }
    const x0Hidden = (x) => !!x.sp.userData.tickHide;
    /* 抽样只管**一根轴自己**排不排得下；两根轴共用的那个角落它管不着——
       横轴的末位与纵轴的末位各自合规，投到屏幕上照样叠在一起（3D 里 DP99 压住 PP4·L40-47 就是）。
       所以补一遍跨轴的屏幕盒相交检查：段名（PP4·L40-47）比刻度（DP99）信息量大，冲突时留段名；
       同级则留先建的那枚。同一根轴内部不再动手——那是抽样的地盘，两套解法叠上去会排成锯齿。 */
    /* 往里缩一点再判：精灵的四边是**贴图边**，牌的底与描边在贴图里本就留了一圈空白
       （makeLabel 的 PAD），所以两块牌的贴图刚好挨上时，眼睛看到的是两块分开的牌。
       不缩的话相邻两边一个 0.3px 的接触就会误判成压住——顶视的 DP0 正是这么被判掉的
       （它的上沿与 TP0 的下沿差不到一个像素）。 */
    const RECT_INSET = 3;
    function labelRect(sp) {
      const c = toPx(sp.position), k = worldPerPx();
      const w = sp.scale.x / k, h = sp.scale.y / k;
      const cx = sp.center.x, cy = sp.center.y;
      const ix = Math.min(RECT_INSET, w / 4), iy = Math.min(RECT_INSET, h / 4);
      return {
        x0: c.x - cx * w + ix, x1: c.x + (1 - cx) * w - ix,
        y0: c.y - (1 - cy) * h + iy, y1: c.y + cy * h - iy,
      };
    }
    const rectHit = (a, b) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;
    function dropCornerClashes() {
      const kept = [];
      const cand = [];
      axGroup.children.forEach((sp) => {
        if (!sp.isSprite || !sp.userData.lab) return;
        /* 先按「这一屏该不该出」复位再判叠：上一帧被判掉的牌得有机会回来，
           否则转一次相机就少一枚、越转越空。 */
        if (sp.userData.views) sp.visible = sp.userData.views.indexOf(S.view) >= 0 && !sp.userData.tickHide;
        if (!sp.visible) return;
        /* 冲突时谁留下：**段名 > 刻度**。段名（`PP3·L36-47`）是这一段唯一的那块牌，
           刻度（`TP1`）是一串里的一枚、丢了还能悬停问出来。段标现在也走 series
           （为了跟着抽样一起疏密），所以不能再靠「是不是 series」区分，改由建的时候
           显式声明 rank——否则谁先建谁赢，纯看代码顺序。 */
        const rk = sp.userData.tickSer ? (sp.userData.tickRank || 0) : 2;
        cand.push({ sp, ser: sp.userData.tickSer || null, rank: rk });
      });
      cand.sort((a, b) => b.rank - a.rank);
      cand.forEach((c) => {
        const r = labelRect(c.sp);
        // 只和**别的轴**比：同一系列内部的疏密已由抽样定过
        const clash = kept.some((k) => (k.ser === null || k.ser !== c.ser) && rectHit(r, k.rect));
        if (clash) { c.sp.visible = false; return; }
        kept.push({ ser: c.ser, rect: r });
      });
    }

    /* ── 轴标注（每形态一套：网格框 + 刻度 + 语义标注 + 关键结构线）── */
    const axGroup = new THREE.Group(); scene.add(axGroup);
    /* 分段的**命中区**：一块分段就是一个看不见的盒子，悬停它就报出这一段是谁、里面有什么。
       这是「摆不下就少摆标签」的配套——抽样之后多数块没有牌，但块本身还在那儿；
       没有牌的那几块，靠悬停一样问得出来，信息一条没少，只是不常驻画面。
       材质是「画不出东西但参与拾取」的：opacity 0 + 不写深度，射线照样命中。 */
    const pickGroup = new THREE.Group(); scene.add(pickGroup);
    const PICK_MAT = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false });
    const PICK_GEO = new THREE.BoxGeometry(1, 1, 1);
    function axRegion(q, meta) {
      const m = new THREE.Mesh(PICK_GEO, PICK_MAT);
      m.position.set((q.x0 + q.x1) / 2, (q.y0 + q.y1) / 2, (q.z0 + q.z1) / 2);
      m.scale.set(Math.max(1e-3, q.x1 - q.x0), Math.max(1e-3, q.y1 - q.y0), Math.max(1e-3, q.z1 - q.z0));
      m.userData.region = meta;
      pickGroup.add(m);
    }
    /* 沿一根轴的一串分段各注册一个命中区（其余两轴取满场）。
       区间**按分割线切满**（上一条分割 → 下一条分割），不是只裹住卡本身：
       块与块之间那段空档也属于某一段，鼠标落在空档上照样该答得出「这是第几段」——
       只裹卡的话，TP 切片有 57% 的地方悬停无反应。首尾两段延到场边。 */
    function axRegionsAlong(n, axis, at, bb, mk) {
      const m = BM;
      const base = {
        x0: bb.x0 - CARD.x / 2 - m, x1: bb.x1 + CARD.x / 2 + m,
        y0: bb.y0 - CARD.y / 2 - m, y1: bb.y1 + CARD.y / 2 + m,
        z0: bb.z0 - CARD.z / 2 - m, z1: bb.z1 + CARD.z / 2 + m,
      };
      const cut = (i) => (at(i - 1) + at(i)) / 2;
      for (let i = 0; i < n; i++) {
        const q = Object.assign({}, base);
        q[axis + '0'] = i === 0 ? Math.min(base[axis + '0'], at(0) - m) : cut(i);
        q[axis + '1'] = i === n - 1 ? Math.max(base[axis + '1'], at(n - 1) + m) : cut(i + 1);
        axRegion(q, mk(i));
      }
    }
    // 网格线材质登记表：聚焦（选中压暗）时整体提亮——卡退成背景，格子接手空间参照
    const gridMats = [];
    function clearAxes() {
      gridMats.length = 0;
      while (pickGroup.children.length) pickGroup.remove(pickGroup.children[0]);
      axPad = {}; axSeries = [];       // 边带与轴上标记登记随标注一起重算
      while (axGroup.children.length) {
        const o = axGroup.children.pop();
        if (o.geometry) o.geometry.dispose();
        if (o.material) { if (o.material.map) o.material.map.dispose(); o.material.dispose(); }
      }
    }
    /* 标注锚点的世界包围盒：取景只按卡的包围盒算的话，摆在模型外侧的标记在小规格下会被
       推出画布——相机贴得越近，同样的世界偏移越出框。fitView 把它并进取景范围。
       只并**锚点位置**、不并牌的尺寸：牌是固定屏幕尺寸，尺寸并进去会自我反馈
       （取景放宽 → worldPerPx 变大 → 牌的世界尺寸跟着变大 → 再放宽…never settles）。
       牌本身占的那点屏幕空间，由 fitView 末尾一个固定比例的留白吸收。 */
    let axBox = null;
    /* 贴边标记要占掉画布左边 / 下边一条**固定像素**的带子（牌是固定屏幕尺寸）。
       把这条带子按视角记下来，fitView 先从视口里扣掉它再去装模型——
       这样既不会像「把牌的世界尺寸并进包围盒」那样自我反馈（放宽→牌变大→再放宽），
       也不会出现「牌被挤出画布」（宽的规格牌尤其容易，一张就近 300px）。 */
    let axPad = {};
    let axSeries = [];      // 轴上标记的成组登记：疏密随相机实时定
    const axPadOf = (v) => axPad[v] || { left: 0, bottom: 0 };
    function axBoxAdd(sp) {
      const p = sp.position;
      const e = { x0: p.x, x1: p.x, y0: p.y, y1: p.y, z0: p.z, z1: p.z };
      if (!axBox) { axBox = e; return; }
      axBox.x0 = Math.min(axBox.x0, e.x0); axBox.x1 = Math.max(axBox.x1, e.x1);
      axBox.y0 = Math.min(axBox.y0, e.y0); axBox.y1 = Math.max(axBox.y1, e.y1);
      axBox.z0 = Math.min(axBox.z0, e.z0); axBox.z1 = Math.max(axBox.z1, e.z1);
    }
    function rebuildAxBox() {
      axBox = null;
      axGroup.traverse((o) => { if (o.isSprite && o.visible) axBoxAdd(o); });
    }
    function applyGridEmphasis() {
      const k = focusOn() ? 1.6 : 1;
      gridMats.forEach((m) => { m.opacity = Math.min(1, m.userData.baseOp * k * (settling ? 0.45 : 1)); });
    }
    // 字牌尺度：世界尺寸随模型尺度伸缩，使标注在屏幕上占比恒定（相机按包围盒取景，
    // 固定世界尺寸的字牌在小规格下会被放大到盖满画面——128 卡与 4000 卡差 6 倍）。
    let LS = 1;
    function updateLabelScale() {
      const b = model.boundsOf(S.mode);
      const span = Math.max(b.x1 - b.x0, b.y1 - b.y0, b.z1 - b.z0);
      LS = Math.min(1.6, Math.max(0.3, span / 52));
    }
    // 标注离盒子的偏移量同样随尺度收缩（固定偏移会让小规格下的横幅飘到画布外/被工具栏遮住），
    // 但留 0.5 下限，避免贴到方块上。
    // 标记与网格/轴的距离：横幅撤掉后没必要再留大留白，整体收到原来的 55%——
    // 标记贴着它标注的那根格线，读图时不用来回找对应关系。
    const D = (v) => v * 0.55 * Math.max(0.5, LS);
    // 长文案「读图横幅」（w≥5）只在轴测视图显示：正交 2D 取景很紧，横幅字牌（世界尺寸随文本
    // 长度膨胀）会盖满画面——2D 里只留短刻度标（TP0/DP127/层段标尺…），语义讲解交给 HUD。
    /* 长横幅（w ≥ 5 的整句解释）不再画进 3D：它们是「散文」而不是「标记」——浮在模型上
       又大又抢，还挡卡。按设计系统 sidecar pattern 的分工，几何标注只留短标记
       （TP0 / DP99 / PP0·L0-9 / 桶3 / 列2…），整句解释交给固定屏幕位置的 UI——
       这里收进「形态」问号气泡（axNotes → DYN.modes），信息一句不少。 */
    let axNotes = [];
    /* w 只剩一个语义：**这句是散文还是标记**（w ≥ 5 = 整句解释 → 不画进 3D，收进问号
       气泡）。字号不再由它决定——所有字牌都是 LABEL_PX 的固定屏幕尺寸。
       plate=true 的是块标：文案带「规格」第二行，同样不是横幅。 */
    function axText(text, color, w, pos, anchor, sub, plate) {
      if (w >= 5 && !plate) {
        axNotes.push(Array.isArray(text) ? text.map((x) => x.t).join('') : String(text));
        return null;
      }
      const l = makeLabel(text, color, w, sub);
      l.position.copy(pos);
      l.userData.banner = w >= 5 && !plate;
      axGroup.add(l);
      // 引线：横幅坐在模型外面时，用一根中性细线牵回它标注的位置——既不压住卡，
      // 也不让读者去猜这句话说的是哪根轴（对比度低于模型本身，annotation 不抢戏）。
      return l;
    }
    // 两段式横幅：名字用维度签名色、解释用中性次级前景色
    const seg2 = (head, hc, rest) => [{ t: head, c: hc }, { t: rest, c: tokHex('--foreground-secondary') }];
    /* 只在某几个视角出现的标记。正交 2D 把一根轴折进视线，标记在那根轴上的偏移随之
       失效——3D 里「浮在块顶上方」的一排块标，到顶视会全部塌到 z=0 那一行、互相叠死。
       所以块标做成两份：3D/前视用长名（浮在块顶），顶视换一份短名摆到场外，
       标注轴被折掉的视角（TP 切片的侧视、PP 流水的侧视）干脆不出。
       视角切换只改可见性，不重建场景（renderAxes 很贵）。 */
    const axOnly = (l, views) => { if (l) l.userData.views = views; return l; };
    function applyAxVisibility() {
      const sp = screenPlane();
      axGroup.traverse((o) => {
        /* 线的收放：
             plane='3d'  → 只在 3D 出（块框的整只盒子）
             flat=true   → 只在 2D 出，且只出屏幕那一面（块框拍平成的矩形）
             其余带 plane → 3D 三面拼成角落格箱；2D 只留屏幕那一面 */
        const pl = o.userData.plane;
        if (pl) {
          o.visible = pl === '3d' ? !sp : o.userData.flat ? (!!sp && pl === sp) : (!sp || pl === sp);
          return;
        }
        if (!o.isSprite && !o.isLine) return;
        /* 选中一张卡、且宿主选择收起这一路时：把「贴在几何体上」的那一类字牌
           （TP0/PP3 这种轴刻度、桶号——世界尺寸固定，跟着相机一起放大缩小）
           全部关掉。「贴边」的那一类（fixedPx=true，图表框架的一部分）不在
           这条里——它们本来就是固定屏幕尺寸，不会因为贴近一张卡而爆大。
           反馈原话「彻底去掉 3D 画布中的显示，卡片还是保留彩色」：rank-
           topology-lite 的第二档把相机怼得极近（"局部聚焦"那颗镜头），世界
           尺寸字牌这时候会占据大半个画布、糊住宿主自己的详情卡——而这些
           坐标信息本来就已经在 DOM 侧栏里写着，没有必要在画布里重复一遍，
           还占那么大的地方。独立打开 /rubik-pattern.html 或没传这个选项的
           宿主一个字节不变。 */
        if (opts.axisLabelsOnSelect === false && S.sel != null && o.isSprite && !o.userData.fixedPx) {
          o.visible = false; return;
        }
        if (o.userData.banner) { o.visible = S.view === 0; return; }
        if (o.userData.views) o.visible = o.userData.views.indexOf(S.view) >= 0 && !o.userData.tickHide;
      });
      /* 换屏之后重判一次疏密：这一屏的取景可能与上一屏同缩放（syncLabelSizes 会短路），
         但落边、投影方向都变了，抽样与跨轴判叠都得按新屏重算。 */
      syncTickDensity();
      rebuildAxBox();
    }
    /* 线的分级借鉴设计系统 sidecar 的 layer-guide / stage-guide：
         · **细分格线** = 虚线 · 细 · 中性淡（`layer-guide` 是 0.9px + dash 2 5 + 前景 20%）
         · **分段边界 / 块框** = 实线 · 签名色 · 高一档不透明度（`stage-guide`）
       两级一分开，格子就退成背景，「这是第几片」由实线块框直接说，画面立刻清爽。
       plane：这组线躺在哪个平面（xy / yz / zx）。正交 2D 只画**屏幕所在的那个平面**——
       否则另外两个平面的线全部投影成与它重合的直线，一根线画三遍，就是「很多重合的线」。 */
    function axSeg(pairs, color, opacity, opt) {
      if (!pairs.length) return null;
      const o = opt || {};
      const g = new THREE.BufferGeometry().setFromPoints(pairs);
      const mat = o.dashed
        ? new THREE.LineDashedMaterial({ color, transparent: true, opacity, dashSize: CARD.x * 0.34, gapSize: CARD.x * 0.85 })
        : new THREE.LineBasicMaterial({ color, transparent: true, opacity });
      mat.userData.baseOp = opacity;
      gridMats.push(mat);
      const ln = new THREE.LineSegments(g, mat);
      if (o.dashed) ln.computeLineDistances();
      if (o.plane) ln.userData.plane = o.plane;
      if (o.flat) ln.userData.flat = true;
      axGroup.add(ln);
      return mat;
    }
    // 当前这一屏躺在哪个平面（3D 返回 null = 三个平面全画）
    const screenPlane = () => {
      const A = screenAxes(S.view);
      return A ? [A.h, A.v].sort().join('').replace('xz', 'zx') : null;
    };
    function axLine(a, b, colorHex, r) {
      const dir = b.clone().sub(a), len = dir.length();
      const geo = new THREE.CylinderGeometry(r || 0.07, r || 0.07, len, 8, 1);
      const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: colorHex, transparent: true, opacity: 0.92 }));
      mesh.position.copy(a.clone().add(b).multiplyScalar(0.5));
      mesh.quaternion.setFromUnitVectors(V3(0, 1, 0), dir.normalize());
      axGroup.add(mesh);
    }
    /* 轴向不画任何裸线：圆锥箭头（3D 建模语汇）与「轴脊」细线都试过——散落在模型外的
       短线读者认不出是什么，反而像画错的通信连线。轴的语义交给带该维签名色的网格线，
       范围与方向交给刻度标与文案。axLine 只保留给「有明确所指」的结构线（TP 切片的
       墙间连点）。
       大 3D 坐标网格框（底 XZ + 背 XY + 左 YZ 三张网格 + 描边棱线；floorOnly 只铺地面）。
       tint = {x,y,z} 时，该轴的格边界线用这一维的签名色画并加一档不透明度——
       网格自己就说明「这根轴切的是哪一维、切成几格」，因此不再需要轴向箭头。 */
    function axGridBox(b, xt, yt, zt, floorOnly, tint) {
      const hx = (c) => new THREE.Color(c).getHex();
      const grid = hx(tokHex('--border-default')), frame = hx(tokHex('--border-strong'));
      // 格线退成背景（虚线 + 更低不透明度），实线与不透明度留给块框和外框
      const gridOp = isDark() ? 0.30 : 0.42, tintOp = isDark() ? 0.44 : 0.55, frameOp = isDark() ? 0.7 : 0.8;
      const t = tint || {};
      /* 格线按**它躺在哪个平面**分组（每根轴各出两组：地面一组、竖墙一组）。
         3D 三组全画（就是那个经典的角落格箱）；正交 2D 只画屏幕那一面——
         另外两面的线会整整齐齐投影到同一批位置上，一根线画三遍，就是「很多重合的线」。 */
      const G = { xy: { x: [], y: [] }, zx: { x: [], z: [] }, yz: { y: [], z: [] } };
      xt.forEach((x) => {
        G.zx.x.push(V3(x, b.y0, b.z0), V3(x, b.y0, b.z1));                        // 地面：沿 Z
        if (!floorOnly) G.xy.x.push(V3(x, b.y0, b.z0), V3(x, b.y1, b.z0));        // 背墙：沿 Y
      });
      if (!floorOnly) yt.forEach((y) => {
        G.xy.y.push(V3(b.x0, y, b.z0), V3(b.x1, y, b.z0));                        // 背墙：沿 X
        G.yz.y.push(V3(b.x0, y, b.z0), V3(b.x0, y, b.z1));                        // 左墙：沿 Z
      });
      zt.forEach((z) => {
        G.zx.z.push(V3(b.x0, b.y0, z), V3(b.x1, b.y0, z));                        // 地面：沿 X
        if (!floorOnly) G.yz.z.push(V3(b.x0, b.y0, z), V3(b.x0, b.y1, z));        // 左墙：沿 Y
      });
      // 细分格线一律虚线（对齐 sidecar 的 layer-guide）：格子退成背景，实线留给块框
      Object.keys(G).forEach((pl) => Object.keys(G[pl]).forEach((ax) => {
        axSeg(G[pl][ax], t[ax] ? hx(t[ax]) : grid, t[ax] ? tintOp : gridOp, { plane: pl, dashed: true });
      }));
      /* 外框：2D 用「屏幕那一面的矩形」，3D 用**去重后的角落框**。
         三个平面矩形两两共用一条棱（地面与背墙共用后底棱……），3D 下三个矩形一起画
         就有三条棱各画两遍——正是斜视时框看着「毛」的原因。 */
      const rect = (pts) => { const o = []; for (let i = 0; i < 4; i++) o.push(pts[i], pts[(i + 1) % 4]); return o; };
      const FR = {
        zx: rect([V3(b.x0, b.y0, b.z0), V3(b.x1, b.y0, b.z0), V3(b.x1, b.y0, b.z1), V3(b.x0, b.y0, b.z1)]),
        xy: floorOnly ? [] : rect([V3(b.x0, b.y0, b.z0), V3(b.x1, b.y0, b.z0), V3(b.x1, b.y1, b.z0), V3(b.x0, b.y1, b.z0)]),
        yz: floorOnly ? [] : rect([V3(b.x0, b.y0, b.z0), V3(b.x0, b.y0, b.z1), V3(b.x0, b.y1, b.z1), V3(b.x0, b.y1, b.z0)]),
      };
      Object.keys(FR).forEach((pl) => axSeg(FR[pl], frame, frameOp, { plane: pl, flat: true }));
      const seen = {}, corner = [];
      Object.keys(FR).forEach((pl) => {
        for (let i = 0; i < FR[pl].length; i += 2) {
          const a2 = FR[pl][i], b2 = FR[pl][i + 1];
          const k = [a2.x, a2.y, a2.z, b2.x, b2.y, b2.z].map((v) => v.toFixed(3)).sort().join(',');
          if (seen[k]) continue;
          seen[k] = 1; corner.push(a2, b2);
        }
      });
      axSeg(corner, frame, frameOp, { plane: '3d' });
    }
    /* 强调块的框：TP切片 / PP流水 把主轴按 emph（4×）拉开成「墙 / 段」。轴测下空档本身
       就说明了分段，但正交 2D 里主轴一稀疏，卡就散成条纹——数不清「这是第几片 / 第几段」，
       当年 2D 被收编正是因为这个。给每一块套一圈签名色细框，条纹当场读成整块。
       框贴着卡本身（不是网格框）：只外扩半张卡 + 一点余量，既不与卡穿插，也不像另起一个
       坐标系。四条竖棱只画到块高的两端，中间留空——满框在 3D 里会织成一片网。 */
    /* ── 2D 标注的落位规律（与二维图表一致，不再各处手写偏移）───────────────
       一个标记标的永远是「某根世界轴上的某个位置」。那么它摆哪儿就不该由人逐处去猜：
         · 这根轴在当前这一屏是**横轴** → 摆到**图的下方**；
         · 是**纵轴** → 摆到**图的左侧**；
         · 被折进视线 → 这一屏根本不出这个标。
       右侧留给剖面卡。场内不放任何标记。
       这么定死之后：① 标签之间永远不会互相压（各自占着自己那条边上的一个位置）；
       ② 换视角、顶视转 90°、换配置，标记都自动跟到该去的边上，不需要逐处再调。
       3D 轴测不受此约束——那里没有「屏幕横纵轴」，标记贴着它标注的结构本身。 */
    // 每个 2D 视角的屏幕坐标系：哪根世界轴是横轴/纵轴，以及该轴的正方向在屏幕上是右/上
    function screenAxes(v) {
      if (v === 1) {
        return topRotated()
          ? { h: 'z', v: 'x', hR: 1, vU: 1 }     // 转 90°：横=Z（右为 +）· 纵=X（上为 +）
          : { h: 'x', v: 'z', hR: 1, vU: -1 };   // 不转：  横=X（右为 +）· 纵=Z（上为 −，Z 向下）
      }
      if (v === 2) return { h: 'x', v: 'y', hR: 1, vU: 1 };   // 前视：横=X 纵=Y
      /* 侧视的横轴是**反的**：相机在 +X 看向 −X、up=+Y，算出来的屏幕右方向是 −Z。
         照抄前视写成 hR:+1 的话，「摆到左边」会摆到右边去（PP 刻度跑到图外右侧就是这个）。 */
      if (v === 3) return { h: 'z', v: 'y', hR: -1, vU: 1 };   // 侧视：横=Z（右为 −）纵=Y
      return null;                                            // 3D 没有屏幕轴
    }
    // 沿某根轴的 along 处，落到该屏的「下边」或「左边」，向外让开 gap
    function axEdgePos(v, kind, axis, along, bb, gap) {
      const A = screenAxes(v);
      const lo = { x: bb.x0, y: bb.y0, z: bb.z0 }, hi = { x: bb.x1, y: bb.y1, z: bb.z1 };
      const o = { x: (lo.x + hi.x) / 2, y: (lo.y + hi.y) / 2, z: (lo.z + hi.z) / 2 };
      o[axis] = along;
      // 让开半张卡（bb 是卡心包围盒）再加 gap
      const half = { x: CARD.x / 2, y: CARD.y / 2, z: CARD.z / 2 };
      if (kind === 'bottom') o[A.v] = A.vU > 0 ? lo[A.v] - half[A.v] - gap : hi[A.v] + half[A.v] + gap;
      else o[A.h] = A.hR > 0 ? lo[A.h] - half[A.h] - gap : hi[A.h] + half[A.h] + gap;
      return V3(o.x, o.y, o.z);
    }
    /* 把一个「轴上标记」摆到某一屏的边上。
       贴边用 sprite.center，不把自己的半个身位算进坐标：精灵默认居中，若靠「坐标再退半个
       身位」来让开，就把当时的 scale 烙进了坐标——之后重算尺寸时宽牌退得多、窄牌退得少，
       同一条边上的留白就参差了（实测下方 0.195、左侧 0.48，差 2.5 倍）。改成左侧的牌右
       边缘对齐锚点、下方的牌上边缘对齐锚点，锚点与外让方向存下来随相机重算，留白恒等。
       不做错行：摆不下由 axOnAxisSeries 抽样解决，两个解法一起上只会排成锯齿。 */
    function axEmit(v, kind, text, color, w, axis, along, bb, o) {
      const l = axText(text, color, w, V3(0, 0, 0), null, o.sub, o.plate);
      if (!l) return null;
      l.userData.fixedPx = true;                   // 贴边 = 图表框架 → 固定屏幕尺寸
      syncLabelSize(l);
      if (kind === 'bottom') l.center.set(0.5, 1); else l.center.set(1, 0.5);
      const A = screenAxes(v);
      l.userData.edge = {
        anchor: axEdgePos(v, kind, axis, along, bb, 0),
        axis: kind === 'bottom' ? A.v : A.h,
        sign: (kind === 'bottom' ? A.vU : A.hR) > 0 ? -1 : 1,
        k: 0.55,
      };
      placeEdgeLabel(l);
      // 记下这一屏该留多宽/多高的边带（牌自身 px + 外让 px）
      const L = l.userData.lab, wpx = LABEL_PX / L.fontFrac;
      const need = (kind === 'bottom' ? wpx * L.aspect : wpx) + LABEL_PX * l.userData.edge.k;
      const pad = axPad[v] || (axPad[v] = { left: 0, bottom: 0 });
      pad[kind] = Math.max(pad[kind], need);
      axOnly(l, [v]);
      return l;
    }
    // 这根轴在第 v 屏上落哪条边（null = 被折进视线，这一屏不出）
    function edgeKind(v, axis) {
      const A = screenAxes(v);
      if (!A) return null;
      return A.h === axis ? 'bottom' : A.v === axis ? 'left' : null;
    }
    // 单个轴上标记：在它出得来的每个 2D 视角各摆一份（各自只在自己那屏可见）
    function axOnAxis(text, color, w, axis, along, bb, opt) {
      const o = opt || {};
      [1, 2, 3].forEach((v) => {
        if (o.views && o.views.indexOf(v) < 0) return;
        const kind = edgeKind(v, axis);
        if (kind) axEmit(v, kind, text, color, w, axis, along, bb, o);
      });
    }
    /* 一条边上摆不下就**少摆几枚**（图表轴的老办法）。
       牌是固定屏幕尺寸，所以「摆不摆得下」是可算的：先建一枚量出它在这条边上的占位，
       再和相邻两项的间距一比——隔几个出一枚 = ⌈占位 ÷ (间距 × 错行档数)⌉。
       同一批标记在不同屏上落的边不同（横 vs 纵），占位也就不同（量宽 vs 量高），
       所以抽样必须**逐屏各算各的**，不能算一次用到底。末位必带上，且不与上一枚贴在一起。 */
    function axOnAxisSeries(n, textOf, color, w, axis, at, bb, opt) {
      const o = opt || {};
      [1, 2, 3].forEach((v) => {
        if (o.views && o.views.indexOf(v) < 0) return;
        const kind = edgeKind(v, axis);
        if (!kind) return;
        /* 不错行：错行与「摆不下就少摆」是同一个问题的两个解法，两个一起上就成了
           这种上上下下的锯齿——抽样已经保证摆得下，再错行纯属添乱。
           一条边上只排一行，是图表轴的常规读法。 */
        /* `subOf` = 第二行按序号取（如 PP 段的 `L0-L11`）。分成两行是为了让牌**变窄**：
           落到下方那条边时，占位量的是宽度，`PP0 · L0-L11` 一行摆要 ~140px、而 PP 只有
           几列、列距 ~118px，抽样就会把四段丢掉两段——四段丢两段不是「密了少摆几枚」，
           是把主要信息摆没了。拆成 `PP0` / `L0-L11` 两行后宽度减半，全部摆得下。 */
        const subAt = (i) => (o.subOf ? o.subOf(i) : o.sub);
        // 拿文案最长的一枚当样品（末位通常最长：DP99 / L41-L48），量出它在这条边上占几像素
        const probe = makeLabel(textOf(n - 1), color, w, subAt(n - 1));
        const L = probe.userData.lab, wpx = LABEL_PX / L.fontFrac;
        const need = (kind === 'bottom' ? wpx : wpx * L.aspect) * 1.15;
        if (probe.material.map) probe.material.map.dispose();
        probe.material.dispose();
        /* 疏密**不在建的时候定死**。建的时候相机还没最终取景（fitView 要等标注量出边带才
           算得准），此时定下的步长往往偏大——「明明放得下却少了几枚」就是这么来的；
           缩放之后更是全不作数。所以：先按上限均匀多摆一批，真正显示哪几枚交给
           syncTickDensity 按当前相机算，缩放时自动加密 / 变疏。 */
        const CAP = 28;
        const step0 = Math.max(1, Math.ceil(n / CAP));
        const idx = [];
        for (let i = 0; i < n; i += step0) idx.push(i);
        if (idx[idx.length - 1] !== n - 1) idx.push(n - 1);
        const ser = { view: v, needPx: need, items: [] };
        idx.forEach((i) => {
          const oi = o.subOf ? Object.assign({}, o, { sub: subAt(i) }) : o;
          const l = axEmit(v, kind, textOf(i), o.colorOf ? o.colorOf(i) : color, w, axis, at(i), bb, oi);
          if (l) { l.userData.tickSer = ser; l.userData.tickRank = o.rank || 0; ser.items.push({ sp: l, at: at(i) }); }
        });
        if (ser.items.length) axSeries.push(ser);
      });
    }
    // 轴刻度 = 最简单的一串轴上标记，密了自动减
    function axAxisTicks(fmt, color, w, axis, n, at, bb) {
      axOnAxisSeries(n, fmt, color, w, axis, at, bb, {});
    }
    /* 3D 的轴刻度：**整排贴着轴摆**，与 DP 平铺的「列0…列9 / 行0…行9」同一种做法。
       约定（与 2D 的「下方=横轴·左侧=纵轴」对应，只是轴测里换成了盒子的三条边）：
         X 轴 → 沿地面前沿（z 最大那条边）· Z 轴 → 沿地面右沿（x 最大那条边）
         Y 轴 → 沿左后立棱（x 最小 · z 最小）
       原来只给首尾两枚、块标还浮在场上：首尾读不出中间是第几个，浮着的块标又挡卡。
       疏密仍交给 syncTickDensity 按当前相机实时算（轴测下会随旋转变，所以必须实时）。 */
    function ax3dTicks(fmt, color, w, axis, n, at, bb, opt) {
      /* X 与 Z 都贴着地面走，两条边在近角**交汇**——同一个外让距离会让两串刻度在那儿
         挤成一堆（标准 3D 里 PP3 与 TP1 只差 35px，跨轴判叠必然要枪毙一枚，而 TP 一共
         才两枚，丢一枚就丢掉一半）。所以让 Z 退得更远一档：真正的 3D 图表也是这么摆的
         ——两根轴的标注坐在不同的半径上，角落自然分开，不必靠判叠去裁。 */
      const o = opt || {};
      const DEF_OFF = { x: 1.8, z: 3.4, y: 1.8 };
      const off = D(o.off == null ? DEF_OFF[axis] : o.off);
      const pos = (i) => (axis === 'x' ? V3(at(i), bb.y0, bb.z1 + off)
        : axis === 'z' ? V3(bb.x1 + off, bb.y0, at(i))
          : V3(bb.x0 - off, at(i), bb.z0));
      const CAP = 28, step0 = Math.max(1, Math.ceil(n / CAP));
      const idx = [];
      for (let i = 0; i < n; i += step0) idx.push(i);
      if (idx[idx.length - 1] !== n - 1) idx.push(n - 1);
      // 与 2D 同一套：subOf 把牌拆成两行，宽度减半 → 轴测里投影缩短也摆得下
      const subAt = (i) => (o.subOf ? o.subOf(i) : o.sub);
      const probe = makeLabel(fmt(n - 1), color, w, subAt(n - 1));
      const L = probe.userData.lab, wpx = LABEL_PX / L.fontFrac;
      if (probe.material.map) probe.material.map.dispose();
      probe.material.dispose();
      const ser = { view: 0, needPx: wpx * 1.15, items: [] };
      idx.forEach((i) => {
        const l = axOnly(axText(fmt(i), o.colorOf ? o.colorOf(i) : color, w, pos(i), null, subAt(i), o.plate), [0]);
        if (l) { l.userData.tickSer = ser; l.userData.tickRank = o.rank || 0; ser.items.push({ sp: l, at: at(i) }); }
      });
      if (ser.items.length) axSeries.push(ser);
    }
    const R = (n, f) => Array.from({ length: n }, (_, i) => f(i));
    // boxes：一组 {x0,x1,y0,y1,z0,z1}（已含外扩），画成一组线框
    const BM = 0.34;                     // 分割线/框相对卡阵向外让开的余量
    /* 分段照搬设计系统 sidecar 的 `stage-guide`：**一条分割线**，不给每块套框。
       套框的毛病是同一条线要画好几遍——相邻两块共用边界（各画一遍）、最外块的外沿又与
       外框重合（再画一遍），于是「多种框同时在」。分割线只画在**块与块之间**
       （n 块出 n−1 条），一条边界只有一条线；最外沿交给外框，天生不重叠。
       3D 里一条分割是一张矩形（跨另外两根轴），2D 里就是屏幕上的一根线。 */
    function axBlockDividers(axis, cuts, bb, colorHex, op) {
      if (!cuts || !cuts.length) return;
      const m = BM;
      const lo = { x: bb.x0 - CARD.x / 2 - m, y: bb.y0 - CARD.y / 2 - m, z: bb.z0 - CARD.z / 2 - m };
      const hi = { x: bb.x1 + CARD.x / 2 + m, y: bb.y1 + CARD.y / 2 + m, z: bb.z1 + CARD.z / 2 + m };
      const mid = { x: (lo.x + hi.x) / 2, y: (lo.y + hi.y) / 2, z: (lo.z + hi.z) / 2 };
      const other = ['x', 'y', 'z'].filter((k) => k !== axis);   // 分割面横跨的两根轴
      const at = (p, a, va, b, vb) => { const o = Object.assign({}, mid); o[axis] = p; o[a] = va; o[b] = vb; return V3(o.x, o.y, o.z); };
      const box = [], P = { xy: [], yz: [], zx: [] };
      cuts.forEach((p) => {
        const [a, b] = other;
        // 3D：分割面的矩形轮廓
        const c = [at(p, a, lo[a], b, lo[b]), at(p, a, hi[a], b, lo[b]), at(p, a, hi[a], b, hi[b]), at(p, a, lo[a], b, hi[b])];
        for (let i = 0; i < 4; i++) box.push(c[i], c[(i + 1) % 4]);
        // 2D：这根轴还在屏幕上的那两个平面里，各画一根线（跨该平面的另一根轴）
        other.forEach((k) => {
          const pl = [axis, k].sort().join('').replace('xz', 'zx');
          const o0 = Object.assign({}, mid), o1 = Object.assign({}, mid);
          o0[axis] = o1[axis] = p; o0[k] = lo[k]; o1[k] = hi[k];
          P[pl].push(V3(o0.x, o0.y, o0.z), V3(o1.x, o1.y, o1.z));
        });
      });
      const col = new THREE.Color(colorHex).getHex(), o = op || (isDark() ? 0.62 : 0.72);
      axSeg(box, col, o, { plane: '3d' });
      Object.keys(P).forEach((pl) => axSeg(P[pl], col, o, { plane: pl, flat: true }));
    }
    // 相邻两块中点 = 分割位置（n 块 → n−1 条）
    function cutsBetween(n, centerAt) {
      const out = [];
      for (let i = 1; i < n; i++) out.push((centerAt(i - 1) + centerAt(i)) / 2);
      return out;
    }
    /* 网格线 = 格边界（卡永远落在格子里、不被线穿过）——各形态各轴统一这一条约定。
       原先分块轴画边界、其余轴画「穿过卡中心的刻度」，同一个框里两种约定混用，
       看起来就是「和网格没对齐」。线过多时等间隔抽样（每 k 格一条，仍是边界）。 */
    function cellLines(n, step, center, maxLines) {
      const first = (center || 0) - n * step / 2, last = first + n * step;
      const stride = Math.max(1, Math.ceil(n / (maxLines || 12)));
      const out = [];
      for (let i = 0; i <= n; i += stride) out.push(first + i * step);
      if (out[out.length - 1] !== last) out.push(last);
      return out;
    }
    const span1 = (a) => ({ lo: a[0], hi: a[a.length - 1] });

    // 每种形态 = 换一根投影轴：讲清「为什么这样重排 · 这个形状帮你看什么」——一个小方块 = 1 颗卡（rank）
    function renderAxes() {
      clearAxes();
      axNotes = [];
      updateLabelScale();
      const TPc = dimc('TP'), PPc = dimc('PP'), DPc = dimc('DP'), EPc = dimc('EP'), CPc = dimc('CP'),
        NTc = tokHex('--foreground-secondary');   // 中性注释 = 次级前景色
      const hx = (c) => new THREE.Color(c).getHex();
      const TPw = hx(TPc), PPw = hx(PPc), EPw = hx(EPc), NTw = hx(NTc);
      const sp = model.SP;
      const v = { x: 0, y: 0, z: 0 };
      const pos = (tp, pp, rep) => { model.posOf(model.rankOf(tp, pp, rep), S.mode, v); return V3(v.x, v.y, v.z); };
      if (S.mode === 0) {
        /* 轴分配 X=PP · Y=DP · Z=TP：有编号、要一段段读的两维摆上屏幕横纵轴，
           前 PP-DP 面因此是一张现成的「哪一段 × 哪个副本」表；TP 收进纵深。 */
        const s = sp.std, xS = (p) => (p - (PP - 1) / 2) * s.sx, yD = (d) => s.cy + ((REP - 1) / 2 - d) * s.sy, zT = (t) => (t - (TP - 1) / 2) * s.sz;
        const xL = cellLines(PP, s.sx, 0), yL = cellLines(REP, s.sy, s.cy, 9), zL = cellLines(TP, s.sz, 0);
        const b = { x0: span1(xL).lo, x1: span1(xL).hi, y0: span1(yL).lo, y1: span1(yL).hi, z0: span1(zL).lo, z1: span1(zL).hi };
        const bb = model.boundsOf(0);
        axGridBox(b, xL, yL, zL, false, { x: PPc, y: DPc, z: TPc });
        ax3dTicks((i) => `TP${i}`, TPc, 1.6, 'z', TP, zT, bb);
        axText(seg2(`TP×${TP}`, TPc, ` 同一层切 ${TP} 片 · 层内 AllReduce（一整根纵深行 = 一个 TP 组）`), TPc, 7);
        ax3dTicks((i) => `DP${i}`, DPc, 1.6, 'y', REP, yD, bb);
        axText(seg2(`DP×${REP}`, DPc, ' 完整副本 · 数据不同 · 梯度 AllReduce（上→下 DP0→末）'), DPc, 8);
        // 2D：刻度按「它标的是哪根轴」自动落到下方 / 左侧
        axAxisTicks((i) => `TP${i}`, TPc, 1.6, 'z', TP, zT, bb);
        axAxisTicks((i) => `DP${i}`, DPc, 1.6, 'y', REP, yD, bb);
        axText(seg2(`PP×${PP}`, PPc, ` 模型深度 L0→L${model.config.layers - 1}（左→右 Stage0→末） · 段间 P2P`), PPc, 7.6);
        axText('1 小块 = 1 卡（rank）= (TP,PP,DP) 坐标交点 · 另叠 EP 桶', NTc, 9);
        /* 标准形态的 canonical 分段 = PP 段（模型深度），现在切的方向是 X。
           与 TP切片/PP流水 用同一套语汇——分割线 + 规格牌，几何一动不动：
           分段靠**线**标，间距是留给「强调形态」的唯一手段，标准用掉了 PP流水 就没得拉。 */
        axBlockDividers('x', cutsBetween(PP, xS), bb, PPc);
        axRegionsAlong(PP, 'x', xS, bb, (i) => { const r = model.stageLayerRange(i);
          return { title: `PP Stage ${i} · L${r.lo}-L${r.hi}`, sub: `TP ×${TP} · DP ×${REP} = ${TP * REP} 卡`, color: PPc }; });
        // 块标走 series：摆不下就自动少摆几枚（原来一块一枚，密了就叠成一摞）
        /* 牌上只留**身份 + 这个形态最要紧的那个限定**（哪一段 · 管哪几层），
           规格（里面有多少张卡）收进悬停——牌一短，一条边上就摆得下全部几段，
           不用再靠抽样丢掉几枚。信息一条没少，只是分了两层。 */
        /* 画面上一律用**维度名**（PP0/TP0/DP0/桶0），不掺第二套叫法。
             「Stage」是这一维在训练侧的行话，写在悬停与横幅里够了；常驻标记上再叫一次
             「S0」，读者就得同时记住 S 和 PP 是同一根轴——同一个东西两个名字，最不该有。 */
        // 段标两行：`PP0` / `L0-L11`。PP 现在落在下方那条边，占位量的是**宽度**——
        // 一行摆的话四段只摆得下两段（见 axOnAxisSeries 的 subOf 注释）。
        axOnAxisSeries(PP, (i) => `PP${i}`, PPc, 6, 'x', xS, bb,
          { plate: true, rank: 1, subOf: (i) => { const r = model.stageLayerRange(i); return `L${r.lo}-L${r.hi}`; } });
        // 3D：段标整排贴着 X 轴（与 DP 平铺的列/行同一种做法）
        ax3dTicks((i) => `PP${i}`, PPc, 2.6, 'x', PP, xS, bb,
          { rank: 1, subOf: (i) => { const r = model.stageLayerRange(i); return `L${r.lo}-L${r.hi}`; } });
      } else if (S.mode === 1) {
        const s = sp.dpt, COLS = model.COLS, ROWS = model.ROWS;
        const bb = model.boundsOf(1);
        // 网格 = 宫格的「格边界」（由板中心 ± 半格推出）。若沿用 rank 包围盒，X 会整体
        // 偏掉半个板宽 → 板落在格子左侧、网格与内容对不上。
        const xL = cellLines(COLS, s.gapX, 0, 14), zL = cellLines(ROWS, s.gapZ, 0, 14);
        const b = { x0: span1(xL).lo, x1: span1(xL).hi, y0: 0, y1: bb.y1 + 0.6, z0: span1(zL).lo, z1: span1(zL).hi };
        axGridBox(b, xL, [], zL, true, { x: DPc, z: DPc });
        /* canonical 分段 = 每个副本一块板。板是二维排布（列×行），所以不走 blocksAlong，
           直接按板的实际跨度建框。板多时（默认 100 块）框就是这个形态的主视觉：
           「一块板 = 一份完整副本」这句话，靠框比靠文字说得清。 */
        // 宫格是二维的 → 两根轴各出一组分割线（列间 + 行间），交叉出一格一块板
        axBlockDividers('x', cutsBetween(COLS, (i) => (i - (COLS - 1) / 2) * s.gapX), bb, DPc, isDark() ? 0.4 : 0.5);
        axBlockDividers('z', cutsBetween(ROWS, (i) => (i - (ROWS - 1) / 2) * s.gapZ), bb, DPc, isDark() ? 0.4 : 0.5);
        // 板是二维宫格里的一格 → 命中区逐块建（列×行），悬停任一块板即知它是第几个副本
        const hbx = (model.TPC - 1) / 2 * s.tp + CARD.x / 2, hbz = (model.TPD - 1) / 2 * s.tpz + CARD.z / 2;
        R(REP, (r) => {
          const cx = (r % COLS - (COLS - 1) / 2) * s.gapX, cz = (((r / COLS) | 0) - (ROWS - 1) / 2) * s.gapZ;
          axRegion({ x0: cx - s.gapX / 2, x1: cx + s.gapX / 2, y0: bb.y0 - CARD.y / 2 - BM, y1: bb.y1 + CARD.y / 2 + BM,
            z0: cz - s.gapZ / 2, z1: cz + s.gapZ / 2 },
          { title: `DP 副本 ${r}`, sub: `行${(r / COLS) | 0} 列${r % COLS} · TP ×${TP} · PP ×${PP} = ${TP * PP} 卡`, color: DPc });
        });
        // 列沿 X、行沿 Z：2D 里各自落到「自己那根轴此刻是横还是纵」对应的边上
        // 列沿 X、行沿 Z：3D 整排贴着轴（ax3dTicks），2D 交给落位规律自动落到下方 / 左侧
        const colX = (i) => b.x0 + (i + 0.5) * s.gapX, rowZ2 = (i) => b.z0 + (i + 0.5) * s.gapZ;
        ax3dTicks((i) => '列' + i, DPc, 1.7, 'x', COLS, colX, bb);
        ax3dTicks((i) => '行' + i, DPc, 1.7, 'z', ROWS, rowZ2, bb);
        R(COLS, (i) => axOnAxis('列' + i, DPc, 1.7, 'x', colX(i), bb));
        R(ROWS, (i) => axOnAxis('行' + i, DPc, 1.7, 'z', rowZ2(i), bb));
        /* 板有 REP 块，逐块挂牌会糊掉——只给首尾两块两行规格牌，说明「一块板里装了什么」。
           顶视把 PP（Y）折进视线，「浮在板上方」这个偏移当场失效、牌会压在板上 →
           顶视改用 Z 方向把牌推到宫格外（首块推到近侧、末块推到远侧）。 */
        const dSub = `TP ×${TP} · PP ×${PP} = ${TP * PP} 卡`;
        /* 副本板是**二维宫格里的一个点**，不属于任何一根轴——按落位规律，它没有该去的那条边。
           2D 因此不出这两张牌：板的身份由「列 i」「行 j」两条轴刻度交叉读出来本就够了，
           硬摆到边上只会跟轴刻度抢位置（原先就是「列8」和「DP 副本 99」压在一起）。
           「一块板里装了什么」留在 3D 与形态问号气泡。 */
        const d0 = pos(0, 0, 0), dN = pos(0, 0, REP - 1);
        axOnly(axText('DP 副本 0', DPc, 6, d0.clone().add(V3(0, D(3.4), 0)), null, dSub, true), [0]);
        axOnly(axText(`DP 副本 ${REP - 1}`, DPc, 6, dN.clone().add(V3(0, D(3.4), 0)), null, dSub, true), [0]);
        axText(seg2(`DP 平铺 · ${REP} 块板`, DPc, ` = ${REP} 份完整副本（副本号=行×${COLS}+列 · 参数相同 · 各吃不同数据）`), DPc, 11,
          V3(0, b.y1 + D(3.4), 0), V3(0, b.y1, 0));
        const p00 = pos(0, PP - 1, 0), p10 = pos(TP - 1, PP - 1, 0), pTop = pos(0, 0, 0);
        // 板内两根轴只用文案标注：板内格线试过（一块板上单独铺一层格子，在一字排开的
        // 100 块板里像块悬空面板）、轴脊短线也试过（散在模型外，像画错的连线）——都撤了。
        // 这两句说的是「板里面怎么排」，都挂在板上：顶视把 PP 折掉、板内的列×排本来就一览无余，
        // 再压一张牌上去纯属挡卡 → 顶视不出。
        axOnly(axText(model.TPD > 1 ? `板内 TP×${TP} = ${model.TPC}列×${model.TPD}排` : `板内横=TP×${TP}`,
          TPc, model.TPD > 1 ? 4.6 : 3.4, p00.clone().add(V3(0.6, -1.9, 0))), [0]);
        axOnly(axText(`板内竖=PP×${PP} L0（上）→L${model.config.layers - 1}（下）`, PPc, 5,
          pTop.clone().add(V3(0.4, 1.7, 0))), [0]);
        // PP 是 Y 轴：前视/侧视里是纵轴 → 落左侧；顶视 PP 被折掉 → 不出
        axAxisTicks((i) => `PP${i}`, PPc, 1.6, 'y', PP, (i) => s.y0 + (PP - 1 - i) * s.pp, bb);
      } else if (S.mode === 2) {
        const s = sp.ep;
        const bb = model.boundsOf(2);
        // X 网格 = 墙的格边界（同上，避免整体偏掉半个墙宽）；Z 网格线落在真实的域位置上
        // （域少则逐域画，域多则等间隔抽 5 条），不再是与内容无关的等分线。
        const xL = cellLines(EP, s.gapE, 0), yL = cellLines(PP, s.pp, s.cy), zL = cellLines(DOM, s.dom, 0, 9);
        const b = { x0: span1(xL).lo, x1: span1(xL).hi, y0: span1(yL).lo, y1: span1(yL).hi, z0: span1(zL).lo, z1: span1(zL).hi };
        axGridBox(b, xL, yL, zL, false, { x: EPc, y: PPc, z: DPc });
        /* canonical 分段 = 每个专家桶一面墙。墙内 TP 一字排开，所以块的半厚是
           「(TP-1)/2 个 TP 步距 + 半张卡」，不是单张卡的一半。 */
        axBlockDividers('x', cutsBetween(EP, (e) => (e - (EP - 1) / 2) * s.gapE), bb, EPc);
        axRegionsAlong(EP, 'x', (e) => (e - (EP - 1) / 2) * s.gapE, bb,
          (e) => ({ title: `EP 桶 ${e} · ${model.expRange(e)}${model.hotBuckets.has(e) ? ' 热点' : ''}`,
            sub: `域 ×${DOM} · PP ×${PP} · TP ×${TP} = ${DOM * PP * TP} 卡`,
            color: model.hotBuckets.has(e) ? tokHex('--warning') : EPc }));
        const eSub = `域 ×${DOM} · PP ×${PP} · TP ×${TP} = ${DOM * PP * TP} 卡`;
        axOnAxisSeries(EP, (e) => `桶${e} · ${model.expRange(e)}${model.hotBuckets.has(e) ? ' 热点' : ''}`,
          EPc, 6, 'x', (e) => (e - (EP - 1) / 2) * s.gapE, bb,
          { plate: true, rank: 1, colorOf: (e) => (model.hotBuckets.has(e) ? tokHex('--warning') : EPc) });
        axText(seg2(`${EP} 面墙`, EPc, ` = ${EP} 个专家分桶（桶=MoE 组 · 同墙=同专家 · 热点桶标暖色）`), EPc, 10);
        axText(`1 个 A2A 域 = 横穿 ${EP} 面墙的同一排 · 每桶各出 1 员互发`, EPc, 9);
        const zDm = (i) => (i - (DOM - 1) / 2) * s.dom;
        // 3D：域（Z）与墙内 PP（Y）各补首尾刻度，替掉原来那两句散文
        ax3dTicks((e) => `桶${e}${model.hotBuckets.has(e) ? ' 热点' : ''}`, EPc, 2.4, 'x', EP,
          (e) => (e - (EP - 1) / 2) * s.gapE, bb,
          { rank: 1, colorOf: (e) => (model.hotBuckets.has(e) ? tokHex('--warning') : EPc) });
        ax3dTicks((i) => `域${i}`, DPc, 1.6, 'z', DOM, zDm, bb);
        ax3dTicks((i) => `PP${i}`, PPc, 1.6, 'y', PP, (i) => s.cy + ((PP - 1) / 2 - i) * s.pp, bb);
        axAxisTicks((i) => `域${i}`, DPc, 1.6, 'z', DOM, zDm, bb);
        axAxisTicks((i) => `PP${i}`, PPc, 1.6, 'y', PP, (i) => s.cy + ((PP - 1) / 2 - i) * s.pp, bb);
      } else if (S.mode === 3) {
        const s = sp.tps, zD = (d) => (d - (REP - 1) / 2) * s.rep;
        const bb = model.boundsOf(3);
        const xL = cellLines(TP, s.gapT, 0), yL = cellLines(PP, s.pp, s.cy), zL = cellLines(REP, s.rep, 0, 9);
        const b = { x0: span1(xL).lo, x1: span1(xL).hi, y0: span1(yL).lo, y1: span1(yL).hi, z0: span1(zL).lo, z1: span1(zL).hi };
        axGridBox(b, xL, yL, zL, false, { x: TPc, y: PPc, z: DPc });
        axBlockDividers('x', cutsBetween(TP, (t) => bb.x0 + t * s.gapT), bb, TPc);
        axRegionsAlong(TP, 'x', (t) => bb.x0 + t * s.gapT, bb,
          (t) => ({ title: `TP 切片 ${t} · ${t + 1}/${TP}`, sub: `PP ×${PP} · DP ×${REP} = ${PP * REP} 卡`, color: TPc }));
        /* 块标做成两行牌：第一行「这块是谁」，第二行「这块里面有什么」。
           3D 里等距投影会把 +X（右下）与 +Y（上）几乎抵消，错行救不了、加宽更糟，
           而分段本来就由块框交代清楚了 → 3D 只给一枚身份小牌。
           2D 里落位交给 axOnAxis：TP 是 X 轴，顶视里 X 是纵轴（落左侧）、前视里是横轴
           （落下方），不用再逐屏手写偏移。 */
        const tSub = `PP ×${PP} · DP ×${REP} = ${PP * REP} 卡`;
        axOnAxisSeries(TP, (t) => `TP${t} · ${t + 1}/${TP}`, TPc, 6, 'x',
          (t) => bb.x0 + t * s.gapT, bb, { plate: true, rank: 1 });
        axText(seg2(`${TP} 面墙`, TPc, ` = 每层权重的 ${TP} 个切片 · 一面墙 = 全网同槽位卡`), TPc, 9.5);
        /* 这里曾有一条「墙间连点」（8 个球 + 一根粗线），本意是说「同一 TP 组的卡分属 8 面墙」。
           撤掉了：① 这件事选中任意一张卡就由真正的 TP AllReduce Ring 连线画出来，
           而且画的是实际算法（Ring/Tree），比一根示意横杆准确得多；② 它横在墙顶，
           又粗又抢，小规格下还会顶到工具栏。文字说明留在下面那句（→ 问号气泡）。 */
        axText(`同一 TP 组的 ${TP} 卡 → 分属 ${TP} 面墙 · 层内 AllReduce 拼回完整权重`, TPc, 9.5);
        ax3dTicks((t) => `TP${t}`, TPc, 2, 'x', TP, (t) => bb.x0 + t * s.gapT, bb);
        ax3dTicks((i) => `DP${i}`, DPc, 1.6, 'z', REP, zD, bb);
        ax3dTicks((i) => `PP${i}`, PPc, 1.6, 'y', PP, (i) => s.cy + ((PP - 1) / 2 - i) * s.pp, bb);
        axAxisTicks((i) => `DP${i}`, DPc, 1.6, 'z', REP, zD, bb);
        axAxisTicks((i) => `PP${i}`, PPc, 1.6, 'y', PP, (i) => s.cy + ((PP - 1) / 2 - i) * s.pp, bb);
      } else {
        const s = sp.ppf, zD = (d) => (d - (REP - 1) / 2) * s.rep;
        const bb = model.boundsOf(4);
        const xL = cellLines(PP, s.gapP, 0), zL = cellLines(REP, s.rep, 0, 9);
        const b = { x0: span1(xL).lo, x1: span1(xL).hi, y0: bb.y0 - 0.8, y1: bb.y1 + 0.8, z0: span1(zL).lo, z1: span1(zL).hi };
        axGridBox(b, xL, [], zL, true, { x: PPc, z: DPc });
        axBlockDividers('x', cutsBetween(PP, (st) => bb.x0 + st * s.gapP), bb, PPc);
        axRegionsAlong(PP, 'x', (st) => bb.x0 + st * s.gapP, bb, (st) => { const r = model.stageLayerRange(st);
          return { title: `PP Stage ${st} · L${r.lo}-L${r.hi}`, sub: `TP ×${TP} · DP ×${REP} = ${TP * REP} 卡`, color: PPc }; });
        const pSub = `TP ×${TP} · DP ×${REP} = ${TP * REP} 卡`;
        axOnAxisSeries(PP, (i) => { const r = model.stageLayerRange(i); return `PP${i} · L${r.lo}-L${r.hi}`; },
          PPc, 6, 'x', (i) => bb.x0 + i * s.gapP, bb, { plate: true, rank: 1 });
        axText(seg2(`前向激活 PP0→PP${PP - 1}`, PPc, `（左→右，即 Stage0→Stage${PP - 1}）· 反向梯度 ← · 段间 P2P · 每段=连续 ${LPS} 层`), PPc, 10.5);
        ax3dTicks((i) => `PP${i}`, PPc, 2.6, 'x', PP, (i) => bb.x0 + i * s.gapP, bb,
          { rank: 1, subOf: (i) => { const r = model.stageLayerRange(i); return `L${r.lo}-L${r.hi}`; } });
        ax3dTicks((i) => `DP${i}`, DPc, 1.6, 'z', REP, zD, bb);
        axAxisTicks((i) => `DP${i}`, DPc, 1.6, 'z', REP, zD, bb);
        if (CP > 1) {
          /* 反馈「按照分组拉开间距」之后 Y 轴变成了 CP 外层分组 + TP 内层车道的
             复合轴（见 SP.ppf 定义处注释）——刻度也要跟着换轴：外层 CP 给完整
             刻度 + 分组隔线（仿 EP 聚簇形态桶墙那一套读法），内层 TP 不再单独
             出刻度，同一组内的 8 条车道靠间距本身读，跟 EP 桶墙内的 TP 列是
             同一个读法（那边也没给 TP 单独出刻度）。CP=1 走 else 分支，跟改动
             前的单层 TP 刻度逐位相同。 */
          const cpAt = (i) => s.cy + ((CP - 1) / 2 - i) * s.cpGap;
          axBlockDividers('y', cutsBetween(CP, cpAt), bb, CPc);
          ax3dTicks((i) => `CP${i}`, CPc, 1.6, 'y', CP, cpAt, bb);
          axAxisTicks((i) => `CP${i}`, CPc, 1.6, 'y', CP, cpAt, bb);
        } else {
          ax3dTicks((i) => `TP${i}`, TPc, 1.6, 'y', TP, (i) => s.cy + ((TP - 1) / 2 - i) * s.tpIn, bb);
          axAxisTicks((i) => `TP${i}`, TPc, 1.6, 'y', TP, (i) => s.cy + ((TP - 1) / 2 - i) * s.tpIn, bb);
        }
      }
      applyGridEmphasis();   // 网格材质刚重建 → 若正处于聚焦态，立刻补回加强
    }

    // 选中整网层 → 魔方水平切片（标准形态的紫色 slab —— 整网图联动挂点）
    const slabMat = new THREE.MeshBasicMaterial({ color: 0x9B3CF6, transparent: true, opacity: 0.16, depthWrite: false });
    const slab = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), slabMat);
    slab.visible = false; scene.add(slab);
    function updateSlab() {
      if (S.selLayer == null || S.mode !== 0) { slab.visible = false; return; }
      const st = Math.min(PP - 1, (S.selLayer / LPS) | 0);
      const s = model.SP.std, b = model.boundsOf(0);
      // PP 现在是 X 轴 → 层切片是一张**竖着的**板（跨 Y=DP 与 Z=TP），不再是水平薄片
      slab.scale.set(s.sx * 0.92, (b.y1 - b.y0) + 2.4, (b.z1 - b.z0) + 2.4);
      slab.position.set((st - (PP - 1) / 2) * s.sx, s.cy, 0);
      slab.visible = true;
    }

    /* ── 着色：状态热力 / 分组透镜 / 异常注入 ── */
    const rng = (i) => { const x = Math.sin(i * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };
    // 时间 = 一个 step 内的位置（0→1），按 4 个通信阶段读（对齐集群驾驶舱）
    const phaseIdx = () => Math.min(PHASES.length - 1, (S.t * PHASES.length) | 0);
    const phaseU = () => S.t * PHASES.length - phaseIdx();       // 阶段内进度 0→1
    // 负载场随阶段变化：每个阶段由哪根并行轴主导，就在那根轴上呈现相应的形状
    function load01(r) {
      const ph = PHASES[phaseIdx()], u = phaseU();
      const h1 = rng(r), h2 = rng(r * 7.3);
      let v = ph.load;
      if (ph.id === 'TP') {                          // 层内 AllReduce：同 TP 组齐动，副本间轻微错峰
        v *= 0.88 + 0.24 * Math.sin(6.283 * (u + model.repOf(r) * 0.013));
      } else if (ph.id === 'PP') {                   // 阶段接力：波锋沿流水级前进，其余级在等
        const d = Math.abs((model.ppOf(r) + 0.5) / PP - u);
        v *= d < 0.5 / PP + 0.1 ? 1.35 : 0.5 + 0.3 * h1;
      } else if (ph.id === 'EP') {                   // A2A 浪涌：热点桶更烫，各 A2A 域错峰互发
        v *= (model.hotBuckets.has(model.epOf(r)) ? 1.12 : 0.8)
          * (0.9 + 0.22 * Math.sin(6.283 * (u + model.domOf(r) * 0.07)));
      } else {                                       // 梯度 AllReduce：全网齐动，个体差异露出慢副本
        v *= 0.85 + 0.3 * h2;
      }
      return Math.max(0.04, Math.min(1, v * (0.94 + 0.12 * h1)));
    }
    const anomBucket = () => Math.min(3, EP - 1);            // 注入的示意桶号（EP 小时自动收到合法桶）
    function inAnomGroup(r) {
      if (S.anom === 'tp') return model.tpOf(r) === 0;       // TP 槽 0：全网同槽位卡
      if (S.anom === 'pp') return model.ppOf(r) === 0;       // PP 级 0：一整个流水段
      if (S.anom === 'dp') return model.repOf(r) === 0;      // DP 副本 0：一份完整拷贝
      if (S.anom === 'ep') return model.epOf(r) === anomBucket();   // EP 桶：持有该桶的所有 rank（越区示意）
      return false;
    }
    const rgbCss = (c) => `#${c.getHexString()}`;
    // 负载热力 = 图例那条色带本身：--success → --warning → --danger 三段线性插值
    const RAMP_A = new THREE.Color(), RAMP_B = new THREE.Color();
    function loadColor(v) {
      v = Math.max(0, Math.min(1, v));
      const lo = v < 0.5 ? '--success' : '--warning', hi = v < 0.5 ? '--warning' : '--danger';
      RAMP_A.set(tokHex(lo)); RAMP_B.set(tokHex(hi));
      return cTmp.copy(RAMP_A).lerp(RAMP_B, v < 0.5 ? v * 2 : (v - 0.5) * 2);
    }
    /* 注入态的「其余」：健康色但压成背景。
       用满彩度的 --success 会让红绿各占半屏、势均力敌，注入的那一组反而不跳——
       注入这个模式的全部意义就是「异常组一眼看见」，其余是参照物不是并列项。
       故按低负载取色后再向 --background 拉一半：色相还在（还读得出「这些是好的」），
       但明度/彩度都退到红组之下。图例的「其余」色块走同一个函数，因此永远等于画面。 */
    const REST_BG = new THREE.Color();
    function restColor(r) {
      loadColor(0.16 + rng(r * 3.1) * 0.1);
      REST_BG.set(tokHex('--background'));
      return cTmp.lerp(REST_BG, isDark() ? 0.55 : 0.48);
    }
    /* ── 内存构成透镜（w 权重占比 / act 激活占比 / grad 梯度占比 / opt 优化器态占比）──
       状态热力答「这张卡忙不忙」，这四个答「这张卡的 HBM 被什么占着」——同一套
       0→1 灰度映射（loadColor 那条 --success→--warning→--danger 色带），换一个分子，
       视觉语汇不新造一套。权重/激活是用户点名要看清楚的两档；梯度/优化器态与权重
       同源（都正比于参数量 P，见 memBytesOfRank），归一后长得跟权重一模一样，
       留着纯粹是**对照**——两张图对不上，说明字节口径算错了，不是花纹本身有问题。
       用渐变（不是分组色环）是因为这四档本来就是连续量「占了多少」，不是「属于哪一类」
       ——分组色换个色相只答得出「谁跟谁一样」，答不出「谁比谁多」，渐变才对得上这里
       要读的问题。

       喂给 loadColor 的**不是**「这一档字节数 / 这张卡内存总和」的绝对占比——算过
       才发现那个数几乎不随位置变化：均衡并行本来就设计成不让某张卡背得比别的重，
       而权重/梯度/优化器态又都是参数量 P 的固定倍数（2P/2P/12P），三者之比是个跟 P
       无关的常数；唯一会变的激活项体量比 P 小两个数量级，根本撼不动这个比例——绝对
       占比因此在满屏卡上几乎是同一个数，着色出来是一片死色，白白浪费了这条镜头。
       真正逐卡不同的是**这一档字节数本身**（权重在 PP 首/末段多背 embedding/lm_head、
       专家数除不尽 EP 时多摊一个专家的桶更重；激活在 1F1B 热身期越靠前的 PP 段越重）
       ——所以改用「这张卡在全网同一档里，排在 min→max 之间的哪个位置」（线性归一
       0..1）喂给 loadColor：把结构性差异（哪怕只有百分之一二）拉满整条色带，才看得出
       「谁比谁多」。图例那行 GB 数字给的是归一前的两个真实端点（memByteRange），不是
       归一之后的 0..1——读者由此知道「满屏一种颜色」与「另一种颜色」之间实际差了多少 GB。

       字节口径抄 parallel-topology/demo.html 的 opBytes/memParts（demo.html 那一屏
       正是这批数字唯一的事实源），但只留 pattern.js 本就建模的维度：
         · GQA（demo.html 的 kv 头数）没有对应配置 → 按 MHA 处理，Q/K/V/O 四个
           投影都按 TP 整分，不做「KV 头不整除 TP 时复制」的特例；
         · 专家内 TP 切分（demo.html 的 etp）没有对应配置 → 专家权重只按 EP 切
           （与 netObjects 里 `experts` 条目「专家按 EP 切」的说法一致），不再往下切；
         · 微批数/梯度累积步数（demo.html 的 mbs·ga）没有对应配置 → 在途微批数的
           峰值改用 1F1B 热身期的通用近似 `PP - ppOf(r)`（demo.html 在没给 ga 时
           退化到的同一个公式：`Math.min(D.pp, D.ga) - ppRank`，这里视作 ga≥pp）；
         · demo.html 额外叠了 allocator 抖动 / MoE 路由不均噪声（sin-hash，逐卡不同）
           让「同一段里的卡也不一样」——那是为了配合它的容量告警叙事。这里不需要：
           这个透镜要回答的是「TP/PP/EP 这把刀切完，权重/激活的份额落在哪」，是
           **位置**决定的结构性读数，加一层随机噪声只会让「同一 TP 槽位到底是不是
           真的更重」这件事变得读不准，所以碎片/预留项走一个扁平常量。 */
    const ACT_COEF = 4;                              // 每 token·每隐藏维·每层的激活字节数经验系数，同 demo.html
    const MEM_RESERVE = 1.5 * Math.pow(2, 30);        // 碎片/预留：扁平 1.5GiB（demo.html 同档位的量级中位数）
    const MEM_LENS_KEY = { w: 'w', act: 'act', grad: 'g', opt: 'opt' };  // colorBy 名 → memBytesOfRank 的字段名
    // 单层里，这张卡的参数元素数（未乘 bf16 的 2 字节）：Q/K/V/O 四投影 + 复制的
    // router + 本桶路由专家（gate/up/down 三矩阵）+ 复制但按 TP 切的共享专家。
    function layerElemsOfRank(r) {
      const C = model.config, hid = C.hidden, tpShare = 1 / TP;
      const attn = 4 * hid * hid * tpShare;
      const router = hid * C.experts;                          // 复制；量级远小于专家权重，仅为口径完整
      const routed = model.EXP_PER * 3 * hid * C.ffnHidden;     // 本 rank 的 EP 桶：EXP_PER 个专家，只按 EP 切
      const shared = (C.sharedExperts || 0) * 3 * hid * C.ffnHidden * tpShare;
      return attn + router + routed + shared;
    }
    // 这张卡的总参数元素数：本段层数份的逐层权重，PP 首/末段各自再加一头
    // embedding/lm_head（词表按 TP 精确整分，除不尽时前面的槽位多拿一个——同
    // netObjects 里 embedding/lm_head 用的 axSlice 口径，读数与那边的「持有区间」一致）。
    function paramElemsOfRank(r) {
      const C = model.config, hid = C.hidden;
      let p = layerElemsOfRank(r) * LPS;
      const vs = axSlice(C.vocab, TP, model.tpOf(r)), vOwn = vs.hi - vs.lo + 1;
      if (model.ppOf(r) === 0) p += vOwn * hid;         // 词嵌入：仅 PP 首段持有
      if (model.ppOf(r) === PP - 1) p += vOwn * hid;    // LM Head：仅 PP 末段持有（PP=1 时两者叠在同一 rank）
      return p;
    }
    // 五档字节数（口径同 memParts 的 w/g/opt/act/rsv）：权重与梯度都是 bf16（×2），
    // 优化器态是 fp32 的主参数+一阶矩+二阶矩（×4×3=×12）。
    function memBytesOfRank(r) {
      const C = model.config;
      const P = paramElemsOfRank(r);
      const w = 2 * P, g = 2 * P, opt = 12 * P;
      const peakMb = Math.max(1, PP - model.ppOf(r));   // 在途微批数峰值（1F1B 热身期近似，见上方大注）
      const act = (C.seqLen / CP) * C.hidden * LPS * ACT_COEF * peakMb;
      const total = w + g + opt + act + MEM_RESERVE;
      return { w, g, opt, act, rsv: MEM_RESERVE, total };
    }
    // 全网扫一遍，找这一档字节数的 min/max：既是着色前归一化用的分母，也是图例那两个
    // GB 端点的来源——颜色与图例文字永远读同一份数，不会各说各话。O(N)，N 至多几千张卡，
    // 每张卡几次乘法，可忽略不计；recolor() 每次着色前扫一遍存进 curMemRange 供全屏复用，
    // 不会在 N 次单卡着色里各扫一遍全网（那才是真正的热路径）。
    function memByteRange(colorByKey) {
      const key = MEM_LENS_KEY[colorByKey];
      let lo = Infinity, hi = -Infinity;
      for (let r = 0; r < N; r++) {
        const v = memBytesOfRank(r)[key];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      return { lo, hi };
    }
    function memRangeGB(colorByKey) {
      const b = memByteRange(colorByKey), GiB = Math.pow(2, 30);
      return { lo: b.lo / GiB, hi: b.hi / GiB };
    }
    // colorBy ∈ {w, act, grad, opt} → 这张卡在全网 min→max 之间的相对位置，0..1 供 loadColor 上色
    // （见上方大注：不用绝对占比，用相对位置，否则参数量的量级会把结构性差异整个摊平）。
    function memPos(r, colorByKey, range) {
      const v = memBytesOfRank(r)[MEM_LENS_KEY[colorByKey]];
      const span = range.hi - range.lo;
      return span > 0 ? (v - range.lo) / span : 0.5;
    }
    /* ── 整网对象：承载集合 + 分片序号 ──
       通信算子按 md 笔记 §6 的规则特殊处理：它不属于单张卡而属于一个通信组，
       所以承载集合取**选中卡的那个组**；没选卡时无组可言，整网对象退化成「全网都参与」。 */
    function objCarrySet() {
      if (!S.obj) return null;
      const o = model.netObjBy[S.obj]; if (!o) return null;
      if (o.comm) {
        if (S.sel == null) return null;                       // 没选卡 → 不压暗，图例里说明
        return new Set(model.commGroup(S.sel, o.comm));
      }
      const s = new Set();
      for (let r = 0; r < N; r++) if (model.objCarry(S.obj, r)) s.add(r);
      return s;
    }
    let carrySet = null;
    const objOn = () => S.obj != null;
    // 分片序号 → 组色；没有「片」的对象（复制 / 通信）用同一色，避免假装它被切开
    function objShardIdx(r) {
      const sh = model.objShard(S.obj, r);
      return sh ? sh.idx : -1;
    }
    // recolor() 每次着色前重扫一遍存进来的 min/max（见 memByteRange 注释）；
    // colorOfRank 在单卡调用（选中光晕，见 updateSelFx）时没有这份缓存，退回现扫一遍。
    let curMemRange = null;
    function colorOfRank(r) {
      /* 对象透镜接管着色（同异常注入的约定）：承载者按「持有第几片」着色，
         于是「谁持有同一片」当场同色 —— 切到该对象的 best 形态，同色的会 snap 成一整块。 */
      if (objOn()) {
        const inSet = !carrySet || carrySet.has(r);
        if (!inSet) return restColor(r);
        const i = objShardIdx(r);
        return i < 0 ? cTmp.set(tokHex('--foreground-secondary')) : cTmp.set(groupColor(i));
      }
      if (S.anom !== 'none') {
        if (inAnomGroup(r)) return cTmp.set(tokHex('--danger'));
        return restColor(r);
      }
      // 素色镜头：不按任何维度/负载上色，全阵列一个中性灰——给想要「先看形状、
      // 不看颜色」的读者一个开关，默认仍是 load（负载热力），这颗镜头要显式选。
      if (S.colorBy === 'neutral') return cTmp.set(tokHex('--foreground-secondary'));
      if (S.colorBy === 'load') return loadColor(load01(r));
      // 内存构成透镜：全网相对位置 → 复用负载热力那条渐变（同一色带，换一个分子）。
      if (MEM_LENS_KEY[S.colorBy]) {
        return loadColor(memPos(r, S.colorBy, curMemRange || memByteRange(S.colorBy)));
      }
      // 逻辑分组（TP/PP/DP/EP）与物理分组（主机 / Pod）用同一套分组色环：
      // 「按主机着色」正是看逻辑组与物理落位的亲和度——同色连成块 = 这一组正好装在一台机里。
      const g = S.colorBy === 'tp' ? model.tpOf(r) : S.colorBy === 'pp' ? model.ppOf(r)
        : S.colorBy === 'dp' ? model.repOf(r) : S.colorBy === 'host' ? model.hostOf(r)
          : S.colorBy === 'pod' ? model.podOf(r) : model.epOf(r);
      return cTmp.set(groupColor(g));
    }
    // 正交剖面：非当前层 → 压暗（并在写矩阵时缩小），保持空间参照又不喧宾
    // 一个正交视角可能同时折叠多个维（例：EP 聚簇的侧视把「墙序 ep」与「墙内 tp」
    // 一起折进视线）→ 每格重叠的卡数 = 各折叠维成员数之积。剖面按最外层维逐层翻。
    function curDepth() {
      if (S.view === 0) return null;
      const dims = model.modes[S.mode].depth[S.view];
      if (!dims || !dims.length) return null;
      const info = dims.map((k) => Object.assign({ key: k }, model.depthDims[k]));
      return {
        dims: info,
        slice: info[0],                                           // 剖面翻的是最外层维
        fold: info.reduce((a, d) => a * d.n, 1),                  // 每格重叠卡数
        label: info.map((d) => `${d.lab}×${d.n}`).join(' × '),
      };
    }
    const ghosted = (r) => {
      const d = curDepth();
      return !!(d && S.sliceOn && model.depthIdxOf(r, d.slice.key) !== S.sliceVal);
    };
    /* 选中聚焦：选中一张卡后，与它四个通信域都无关的卡退成背景。
       上千张卡各自有色时，四维高亮会被淹没（尤其分组着色）——把无关的压暗 + 缩小，
       让「这张卡和谁一组」当场跳出来；网格与外框此时反过来加强，空间参照不丢。
       和剖面共用一套 dim 机制，可在「连线」行随时关掉。 */
    const relSet = new Set();
    function buildRelSet() {
      relSet.clear();
      if (S.sel == null) return;
      relSet.add(S.sel);
      peerDims.forEach((d) => model.commGroup(S.sel, d).forEach((r) => relSet.add(r)));
    }
    const focusOn = () => S.wire.focus && S.sel != null;
    // 0=正常 · 1=聚焦压暗（保留形体做参照）· 2=剖面压暗（更狠）
    // 整网对象透镜下，不承载该对象的卡走同一档压暗——「这个对象根本不在这些卡上」
    // 与「这些卡和选中卡无关」是同一件事的两种说法，用同一个视觉档位表达。
    const dimLv = (r) => (ghosted(r) ? 2
      /* 导览作用域：不在当前这一层「里面」的卡退成背景 —— 嵌套关系因此是看得见的
         （全网 → 一个副本 → 一个 Stage → 一个 TP 组，亮着的那块每层缩小一圈）。
         与对象透镜/聚焦共用同一档 dim，图例色与画面永远一致。 */
        : (objOn() && carrySet && !carrySet.has(r)) ? 1
          : focusOn() && !relSet.has(r) ? 1 : 0);
    const BG_C = new THREE.Color();
    // 压暗的唯一算法（图例色块与卡块共用，图例因此永远等于画面）：
    // 打光总增益已归到 ≈1（见上方光源注释），所以这里的系数就是最终看到的压暗幅度；
    // 若哪天又调亮打光，这几个系数要跟着往下压，否则「压暗」会看起来还是一片亮。
    function applyDim(c, lv) {
      if (!lv) return c;
      BG_C.set(tokHex('--background'));
      if (lv === 2) return c.multiplyScalar(isDark() ? 0.22 : 0.55).lerp(BG_C, 0.35);
      return c.multiplyScalar(isDark() ? 0.18 : 0.66).lerp(BG_C, isDark() ? 0.58 : 0.66);
    }
    const _sw = new THREE.Color();
    const dimSwatch = (lv) => rgbCss(applyDim(_sw.set(tokHex('--foreground-secondary')), lv));
    function recolor() {
      // 内存透镜生效时先扫一遍全网 min/max（一次），下面 N 次 colorOfRank 直接复用，
      // 不会各自再扫一遍——真正的热路径只有这一次 O(N)。对象/异常接管着色时用不上，不扫。
      curMemRange = (!objOn() && S.anom === 'none' && MEM_LENS_KEY[S.colorBy]) ? memByteRange(S.colorBy) : null;
      for (let r = 0; r < N; r++) {
        applyDim(colorOfRank(r), dimLv(r));
        chips.setColorAt(r, cTmp);
      }
      if (chips.instanceColor) chips.instanceColor.needsUpdate = true;
    }
    function reScale() {
      let dirty = false;
      for (let r = 0; r < N; r++) {
        /* 剖面外的卡（lv 2）**不画**。剖面只在正交 2D 存在（curDepth 在 3D 返回 null），
           而 2D 里被折的那一维正对着视线——缩小压暗的那一粒和剖面内那张卡在屏幕上完全重合，
           于是每张卡正中多出一个灰方块，既不是数据也不是结构，只是同一个格子的另一层漏了出来。
           「这一格底下还压着几层」由剖面那组控件的读数说，不必在画面上再摆一遍。
           聚焦压暗（lv 1）不这么处理：那些卡各占各的格子，缩小是读得出的对比。 */
        const lv = dimLv(r), want = r === S.sel ? 1.45 : lv === 2 ? 0 : lv === 1 ? 0.42 : 1;
        if (scl[r] !== want) { scl[r] = want; dirty = true; }
      }
      if (dirty) settling = true;
    }
    // 选中/聚焦开关变化后统一刷新（重算关联集合 → 压暗与缩放）
    function refreshFocus() { buildRelSet(); carrySet = objCarrySet(); reScale(); recolor(); applyGridEmphasis(); renderLegend(); buildPayload(); buildShard(); syncShard(); buildDetail(); syncDetailCap(); buildNest(); }
    /* 选中/切换整网对象：承载集合与着色一起重算，并把「该去哪个形态看」交给调用方决定
       （selectObject 会主动飞过去，与 selectLayer/selectBucket 的既有做法一致）。 */
    function refreshObj() { carrySet = objCarrySet(); reScale(); recolor(); renderLegend(); renderInfo(); buildShard(); syncShard(); buildDetail(); syncDetailCap(); }

    /* ── 相机：轴测（等距可旋转）+ 顶/前/侧 正交锁轴（拖动即转回 3D），取景随形态包围盒 ── */
    // 等距轴测（isometric）的标准机位：方位 45°、仰角 asin(tan30°) ≈ 35.26°
    // —— 三根世界轴等比缩短、互成 120°，这才是「轴测」该有的样子。
    const ISO = { theta: Math.PI / 4, phi: Math.asin(Math.tan(Math.PI / 6)) };
    const cam = { theta: ISO.theta, phi: ISO.phi, half: 30, cx: 0, cy: 8, cz: 0, panX: 0, panY: 0 };
    /* opts.zoomOnSelect（?zoomsel= 转进来的一个 0~1 的倍数）：选中一张卡时把
       镜头往那张卡推近一档——反馈截图是 4000 卡的盘古预置，选中的那一列在
       整片阵列里只占几个像素，"rank 1406"那句提示都快找不到自己指的是哪儿。
       默认不传，行为跟改动前一样（选中不动镜头，这是绝大多数消费者一直
       以来的样子，独立打开 /rubik-pattern.html 也不受影响）。
       camTarget 非空时 frame() 每帧把 cam.half/cx/cy/cz 往它那边挪一点，
       挪到位就清空——用的是指数缓动，不是单独再起一套时间轴动画。
       camPreSel 记的是"推近之前"那一份机位，取消选中时原样还回去。 */
    let camTarget = null, camPreSel = null;
    function stepCamera() {
      if (!camTarget) return;
      const k = 0.16;
      cam.half += (camTarget.half - cam.half) * k;
      cam.cx += (camTarget.cx - cam.cx) * k;
      cam.cy += (camTarget.cy - cam.cy) * k;
      cam.cz += (camTarget.cz - cam.cz) * k;
      const done = Math.abs(camTarget.half - cam.half) < 0.02
        && Math.abs(camTarget.cx - cam.cx) < 0.02
        && Math.abs(camTarget.cy - cam.cy) < 0.02
        && Math.abs(camTarget.cz - cam.cz) < 0.02;
      if (done) { cam.half = camTarget.half; cam.cx = camTarget.cx; cam.cy = camTarget.cy; cam.cz = camTarget.cz; camTarget = null; }
    }
    // 顶视是否转 90°：Z 跨度比 X 长就转（依据只此一条，fitView 与 applyCamera 共用）
    function topRotated() {
      const b = model.boundsOf(S.mode);
      return (b.z1 - b.z0) > (b.x1 - b.x0);
    }
    function fitView() {
      const b = model.boundsOf(S.mode);
      // 轴标注留白随模型尺度自适应（固定留白会让小规格模型只占画面一小块）
      const span = Math.max(b.x1 - b.x0, b.y1 - b.y0, b.z1 - b.z0);
      const mx = Math.min(5, Math.max(1.0, span * 0.06));
      // 半尺寸 = rank 中心包围盒 + 卡块自身半尺寸（包围盒只含中心点）+ 标注留白
      let ex = (b.x1 - b.x0) / 2 + CARD.x / 2 + mx;
      let ey = (b.y1 - b.y0) / 2 + CARD.y / 2 + mx * 0.6;
      let ez = (b.z1 - b.z0) / 2 + CARD.z / 2 + mx;
      cam.cx = (b.x0 + b.x1) / 2; cam.cy = (b.y0 + b.y1) / 2; cam.cz = (b.z0 + b.z1) / 2;
      // 并入当前视角下真正会显示的标记：取景以卡心为中心，所以每根轴取「标记探出中心
      // 多远」的最大值。封 3 倍，防某个异常标记把整幅缩成一小团。
      if (axBox) {
        const capX = ex * 3, capY = ey * 3, capZ = ez * 3;
        ex = Math.min(capX, Math.max(ex, Math.abs(axBox.x0 - cam.cx), Math.abs(axBox.x1 - cam.cx)));
        ey = Math.min(capY, Math.max(ey, Math.abs(axBox.y0 - cam.cy), Math.abs(axBox.y1 - cam.cy)));
        ez = Math.min(capZ, Math.max(ez, Math.abs(axBox.z0 - cam.cz), Math.abs(axBox.z1 - cam.cz)));
      }
      cam.panX = 0; cam.panY = 0;
      const w = stageEl.clientWidth || 800, h = stageEl.clientHeight || 600, asp = w / h;
      // 模型在屏幕右/上方向的半跨度（hw, hh）——正交三视图直取，轴测按相机基向量投影
      // （近似式在小尺度下会让方块溢出画布）
      let hw, hh;
      // 顶视把模型的**长边横放**：画布是横的，而顶视保留的两根轴常常一长一短
      // （TP切片 顶视 = TP×8 对 DP×100），不转的话整幅缩成一根细长条，两边全是空白。
      // 转 90° 后 100 张一行横着摆开、8 行竖着叠起来——「一整行红 = 那个槽位整批坏」
      // 这句话才真的看得见。
      if (S.view === 1) { const r = ez > ex; hw = r ? ez : ex; hh = r ? ex : ez; }
      else if (S.view === 2) { hw = ex; hh = ey; }        // 前视：横=X 纵=Y
      else if (S.view === 3) { hw = ez; hh = ey; }        // 侧视：横=Z 纵=Y
      else {
        const st = Math.abs(Math.sin(cam.theta)), ct = Math.abs(Math.cos(cam.theta));
        const sp = Math.abs(Math.sin(cam.phi)), cp = Math.abs(Math.cos(cam.phi));
        hw = ex * st + ez * ct;
        hh = ey * cp + sp * (ex * ct + ez * st);
      }
      // 让开左上角的工具栏卡片：放宽取景后把模型偏向右下。偏移量以「放宽后的实际余量」
      // 封顶，宽扁模型不会被推出画布（面板尺寸从 DOM 实测；chrome:false 的嵌入用法无面板 → 不偏移）。
      const bar = root.querySelector('.prc-topbar');
      const br = bar ? bar.getBoundingClientRect() : null;
      const bx = br && br.width < w * 0.8 ? Math.min(0.18, (br.width / w) * 0.3) : 0;
      const by = br ? Math.min(0.18, (br.height / h) * 0.42) : 0;
      /* 先从视口里扣掉贴边标记占的那条像素带，再去装模型：
         模型的 2·hh 世界高要塞进 (h − padBottom) 像素里 → 所需 cam.half = hh·h/(h−padBottom)。
         宽度同理。像素带与 cam.half 无关，所以这个式子一次就收敛，不会来回放大。 */
      const pad = axPadOf(S.view);
      const fy = h > pad.bottom + 40 ? h / (h - pad.bottom) : 1;
      const fx = w > pad.left + 40 ? w / (w - pad.left) : 1;
      cam.half = Math.max(hh * fy, (hw / asp) * fx) * 1.05 * (1 + Math.max(bx, by));
      const halfW = cam.half * asp;
      cam.panX = -Math.min(bx * halfW, Math.max(0, halfW - hw));    // 相机左移 → 模型右移
      cam.panY = Math.min(by * cam.half, Math.max(0, cam.half - hh)); // 相机上移 → 模型下移
      /* 边带只在**左边和下边**需要，取景放宽却是四边一起放的——不补偿的话右边和上边就白空
         一大条，模型看着偏在一角。所以再把模型往右上推半条边带，空出来的正好落在要用的
         那两侧。（这也是「左右间距要把标签宽度算进去」的意思：留白算的是标签占的地方，
         不是把整幅缩小。） */
      const wpp = (2 * cam.half) / Math.max(1, h);
      cam.panX -= (pad.left * wpp) / 2;      // 相机再左移 → 模型右移，左边空出标签带
      cam.panY -= (pad.bottom * wpp) / 2;    // 相机下移 → 模型上移，下边空出标签带
    }
    function applyCamera() {
      const w = stageEl.clientWidth || 800, h = stageEl.clientHeight || 600, asp = w / h;
      camera.left = -cam.half * asp; camera.right = cam.half * asp;
      camera.top = cam.half; camera.bottom = -cam.half;
      const c = V3(cam.cx, cam.cy, cam.cz);
      const D = 300;
      if (S.view === 1) {
        camera.position.set(c.x, c.y + D, c.z);
        // up=(0,0,-1)：横=X 纵=Z（Z 向下）· up=(1,0,0)：横=Z（左→右）纵=X —— 长边横放时用后者
        camera.up.copy(topRotated() ? V3(1, 0, 0) : V3(0, 0, -1));
      }
      else if (S.view === 2) { camera.position.set(c.x, c.y, c.z + D); camera.up.set(0, 1, 0); }
      else if (S.view === 3) { camera.position.set(c.x + D, c.y, c.z); camera.up.set(0, 1, 0); }
      else {
        const sp = Math.sin(cam.phi), cp = Math.cos(cam.phi);
        camera.position.set(c.x + D * cp * Math.cos(cam.theta), c.y + D * sp, c.z + D * cp * Math.sin(cam.theta));
        camera.up.set(0, 1, 0);
      }
      camera.lookAt(c);
      // 平移（拖拽 / 让开工具栏的取景偏移）：位置与视线目标必须偏移同一个向量，
      // 否则视线方向被改变 → 投影斜切（不再是正经的等距轴测）。
      camera.updateMatrixWorld();
      const right = V3(1, 0, 0).applyQuaternion(camera.quaternion).normalize();
      const up = V3(0, 1, 0).applyQuaternion(camera.quaternion).normalize();
      const shift = right.multiplyScalar(cam.panX).add(up.multiplyScalar(cam.panY));
      camera.position.add(shift);
      camera.lookAt(c.clone().add(shift));
      camera.updateProjectionMatrix();
      /* 必须放在**投影矩阵更新之后**：字牌尺寸只用 worldPerPx（读 cam.half，无所谓顺序），
         但刻度疏密要拿 camera 真投影去量两枚之间隔几像素——放在前面就量的是上一帧的相机，
         一进来会算出一个离谱的步长，整条轴只剩末位一枚。 */
      syncLabelSizes();
      syncPayload();
      syncShard();
    }

    /* ── 嵌套框：把「在谁的里面」画进 3D，而不是只写在面包屑上 ──────────────────
       每往下一层，就在**上一层的范围**外面留一圈、画一个线框盒子。于是画面上真的出现
       一层套一层的容器：全网 ⊃ 这个副本 ⊃ 这一段 ⊃ 这个 TP 组 …
       盒子按该层的维度签名色画，角上挂一枚牌写这一层是谁（DP50 / PP2 / …）。

       盒子从**目标位置**（posOf）算，不是从飞行中的 cur：重排动画期间卡在飞，
       盒子直接落在终点等着，比让它跟着抖更好读。
       一处如实：EP 那层会跨出副本，它的盒子因此**不被上一层的盒子包住** ——
       画面会直接暴露这件事，这正是我们想让人看见的，不去修饰。 */
    const nestGroup = new THREE.Group(); nestGroup.renderOrder = 5; scene.add(nestGroup);
    function clearNest() {
      nestGroup.traverse((o) => {
        if (o === nestGroup) return;
        if (o.geometry) o.geometry.dispose();
        if (o.material) { if (o.material.map) o.material.map.dispose(); o.material.dispose(); }
      });
      while (nestGroup.children.length) nestGroup.remove(nestGroup.children[0]);
    }
    /* 不在导览时也有一条**天然的包含链**：副本 ⊃ 段（= TP 组）⊃ 上下文段。
       嵌套是这套模型本来就有的结构，不该只在导览里才看得见。 */
    function nestChain() {
      const r = S.sel;
      if (r == null) return [];
      const rep = model.repOf(r), pp = model.ppOf(r), cp = model.cpOf(r);
      const out = [
        { key: 'dp', label: `DP${rep}`, pred: (x) => model.repOf(x) === rep },
        { key: 'pp', label: `PP${pp}`, pred: (x) => model.repOf(x) === rep && model.ppOf(x) === pp },
      ];
      if (CP > 1) out.push({ key: 'cp', label: `CP${cp}`,
        pred: (x) => model.repOf(x) === rep && model.ppOf(x) === pp && model.cpOf(x) === cp });
      return out;
    }
    /* ── 未选中时的「结构示例」：DP 副本 ⊃ PP 段 ⊃（CP>1 时）TP 组 ⊃ 卡 ────────
       上一版在这里把**每个副本各画一个壳**，被指出没意义——那是把**同一层级**重复了
       N 遍（16 个一模一样的盒子只是把空间平铺一遍），而要表达的是**层级**。
       现在只拿**一个副本**当样本，把它逐层拆开、每层挂一枚说明牌：
       「一个副本 → 切成几段 → 段里是什么」。一次说清，不铺满屏幕。

       层级数随配置变，不硬画三层——`(dp,pp)` 固定之后剩下的卡数就是 TP×CP：
         CP=1 → 一个 PP 段**正好就是一个 TP 组**（段内 2 卡 = TP 组 2 卡），只有两层容器；
         CP>1 → 段里才真的装着 CP 个 TP 组（16 卡 = 2 个 TP 组），这时第三层才存在。
       牌上照实写，不假装永远有三层。 */
    function buildNestExemplar() {
      const v = { x: 0, y: 0, z: 0 };
      /* 取**中间**那个副本当样本，不取第 0 个：标准形态里 DP 是纵轴、rep0 在最顶上，
         它的说明牌会直接撞进顶栏（实机可见）。中间那个在模型正中，四周都有余地。 */
      const rep = REP >> 1;
      const boxOf = (pred) => {
        const b = { x0: 1e9, x1: -1e9, y0: 1e9, y1: -1e9, z0: 1e9, z1: -1e9 };
        let n = 0;
        for (let r = 0; r < N; r++) {
          if (!pred(r)) continue;
          model.posOf(r, S.mode, v); n++;
          if (v.x < b.x0) b.x0 = v.x; if (v.x > b.x1) b.x1 = v.x;
          if (v.y < b.y0) b.y0 = v.y; if (v.y > b.y1) b.y1 = v.y;
          if (v.z < b.z0) b.z0 = v.z; if (v.z > b.z1) b.z1 = v.z;
        }
        return n ? Object.assign(b, { n }) : null;
      };
      const draw = (b, pad, dim, title, sub, strong) => {
        if (!b) return;
        const w = (b.x1 - b.x0) + CARD.x + pad * 2;
        const h = (b.y1 - b.y0) + CARD.y + pad * 2;
        const d = (b.z1 - b.z0) + CARD.z + pad * 2;
        const cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2, cz = (b.z0 + b.z1) / 2;
        const col = dimc(dim);
        const sh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d),
          new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: strong ? 0.10 : 0.055, depthWrite: false, side: THREE.BackSide }));
        sh.material.userData = { base: strong ? 0.10 : 0.055, noPulse: true };
        sh.position.set(cx, cy, cz); sh.renderOrder = 4; nestGroup.add(sh);
        const eg = new THREE.EdgesGeometry(new THREE.BoxGeometry(w, h, d));
        const em = new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: strong ? 0.9 : 0.5, depthTest: false });
        em.userData = { base: strong ? 0.9 : 0.5, noPulse: true };
        const ln = new THREE.LineSegments(eg, em);
        ln.position.copy(sh.position); ln.renderOrder = 5; nestGroup.add(ln);
        if (title) {
          const lab = makeLabel(title, col, 3.6, sub);
          lab.position.set(cx - w / 2, cy + h / 2 + CARD.y * 0.55, cz - d / 2);
          lab.material.userData = { base: 1, noPulse: true };
          lab.renderOrder = 6; nestGroup.add(lab);
        }
      };
      const stageN = boxOf((r) => model.repOf(r) === rep && model.ppOf(r) === 0);
      const sameAsTP = stageN && stageN.n === TP;      // CP=1 时段就是 TP 组
      /* 三层壳的说明牌全删了：反馈「标签删掉」——嵌套框贴得近，牌子的锚点
         （框角外侧）落到几乎同一块屏幕位置，DP/PP 两张牌的标题+副标题四行
         字挤成一坨，没法既不重叠又不做整套碰撞避让。壳本身（半透明面+
         描边线）还在画，层级关系看框套框就够；具体是哪一层、切了几份，
         交给右侧选中后的详情卡与顶栏坐标读，不在画布里硬挂文字。 */
      // ① 一个副本
      draw(boxOf((r) => model.repOf(r) === rep), CARD.x * 1.15, 'DP', null, null, true);
      // ② 副本切成的每一段
      for (let pp = 0; pp < PP; pp++) {
        draw(boxOf((r) => model.repOf(r) === rep && model.ppOf(r) === pp), CARD.x * 0.5, 'PP', null, null, false);
      }
      // ③ 只有 CP>1 时段内才真的还有一层（CP=1 时段就是 TP 组，画了就是假层级）
      if (!sameAsTP) {
        for (let cp = 0; cp < CP; cp++) {
          draw(boxOf((r) => model.repOf(r) === rep && model.ppOf(r) === 0 && model.cpOf(r) === cp),
            CARD.x * 0.18, 'TP', null, null, false);
        }
      }
    }
    function buildNest() {
      clearNest();
      nestGroup.visible = !!S.wire.nest;
      if (!nestGroup.visible) return;
      /* 没选卡 → **不画壳**。曾经在这里把所有副本各画一个壳（为了「不点击也能看到」），
         但那 N 个一模一样的盒子只是把空间平铺了一遍：它们既不指向任何一张卡，
         也说不出「谁在谁里面」——DP 轴刻度已经把副本分界交代过了，壳只是重复一遍噪音。
         嵌套的意思只在**有一个具体的 rank 作锚**时才成立：DP2 ⊃ PP1 ⊃ 这张卡。
         所以壳跟着选中走；没选中就什么都不画。 */
      if (S.sel == null) { buildNestExemplar(); return; }   // 没选卡：给一个「结构示例」
      const v = { x: 0, y: 0, z: 0 };
      /* 导览时用导览自己的链（含 EP 那层「跨出副本」），平时用天然包含链。
         第 0 层（全网）不画——把整个模型框起来没有信息量。 */
      const levels = nestChain();
      const top = levels.length - 1;
      for (let li = 0; li <= top; li++) {
        const st = levels[li];
        if (!st) continue;
        const pred = st.pred;
        const b = { x0: 1e9, x1: -1e9, y0: 1e9, y1: -1e9, z0: 1e9, z1: -1e9 };
        let cnt = 0;
        for (let r = 0; r < N; r++) {
          if (!pred(r)) continue;
          model.posOf(r, S.mode, v); cnt++;
          if (v.x < b.x0) b.x0 = v.x; if (v.x > b.x1) b.x1 = v.x;
          if (v.y < b.y0) b.y0 = v.y; if (v.y > b.y1) b.y1 = v.y;
          if (v.z < b.z0) b.z0 = v.z; if (v.z > b.z1) b.z1 = v.z;
        }
        if (!cnt) continue;
        /* 外层留更宽的边：里外两个盒子才不会贴在一起看成一个。
           `top - li` = 这一层离最内层还有几级，越外面留白越大。 */
        const pad = CARD.x * (0.62 + (top - li) * 0.55);
        const w = (b.x1 - b.x0) + CARD.x + pad * 2;
        const h = (b.y1 - b.y0) + CARD.y + pad * 2;
        const d = (b.z1 - b.z0) + CARD.z + pad * 2;
        const cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2, cz = (b.z0 + b.z1) / 2;
        const col = dimc(st.key.toUpperCase());
        const cur = li === top;                                   // 最里面那层 = 当前所在
        /* **实体容器壳**：只画线框，在满屏卡块之间根本看不出来（实机反馈就是这条）。
           加一层很淡的半透明填充，多层嵌套时填充叠加、越往里越深 ——
           「里面还有一层」不用数框就读得出。BackSide + 不写深度，才不会糊住里面的卡。 */
        const shell = new THREE.Mesh(new THREE.BoxGeometry(w, h, d),
          new THREE.MeshBasicMaterial({
            color: col, transparent: true, opacity: cur ? 0.15 : 0.08,
            depthWrite: false, side: THREE.BackSide,
          }));
        shell.material.userData = { base: cur ? 0.15 : 0.08, noPulse: true };
        shell.position.set(cx, cy, cz); shell.renderOrder = 4; nestGroup.add(shell);
        const g = new THREE.EdgesGeometry(new THREE.BoxGeometry(w, h, d));
        const m = new THREE.LineBasicMaterial({
          color: col, transparent: true, opacity: cur ? 1 : 0.62, depthTest: false,
        });
        m.userData = { base: cur ? 1 : 0.62, noPulse: true };
        const box = new THREE.LineSegments(g, m);
        box.position.set(cx, cy, cz); box.renderOrder = 5; nestGroup.add(box);
        // 角上挂一枚牌：这一层是谁
        const lab = makeLabel(st.label, col, cur ? 3.2 : 2.6);
        lab.position.set(cx - w / 2, cy + h / 2 + CARD.y * 0.5, cz - d / 2);
        lab.material.opacity = cur ? 1 : 0.6;
        lab.material.userData = { base: cur ? 1 : 0.6, noPulse: true };
        lab.renderOrder = 6; nestGroup.add(lab);
      }
    }

    /* 导览（缩放穿梭）与六面助记参考图已整体移除 —— 一层层轮播视图 + 一堆面板，
       并没有把「卡里装着什么」说清楚，反而堆出很多要读的框。
       嵌套改由结构框表达，卡内交给「卡内魔方」。 */

    /* ── 装载板：把选中的这张卡在 3D 里摊开成「它持有的那些碎片」──────────────
       编码沿用 6D 并行可视化的通行做法（main-horse「Visualizing 6-D Mesh Parallelism」：
       *sharded tensors use subdivision and selective coloring within device squares*）——
       **把设备方块细分，持有的那一片着色、其余留空**。这里把它立体化：

         一条积木 = 一个整网对象的**全部**分片（跨所有卡）
         实心段   = 本卡持有的那一片（族色）· 位置就是第几片
         空槽     = 其余片，在别的卡上
         整条空   = 这张卡完全没有这个对象（如词嵌入只落 PP 首段）
         整条实   = 复制的对象（router），不被任何维切

       为什么不是侧栏那份文字清单的翻版：清单给的是**读数**（5/8 h40-h47），
       这块板给的是**比例与位置**——一眼看出「我只占这么一小段」「专家那条我占了一半」
       「这条整根是空的」。两者互补，所以侧栏那份保留。

       整块板面向相机（billboard）而不是贴世界轴：贴轴的话换到某些正交视角会退化成
       一条线，而它是「随时想看就看」的检视件，不该有看不见的角度。 */
    const payGroup = new THREE.Group(); payGroup.renderOrder = 9; scene.add(payGroup);
    const payPanel = new THREE.Group(); payGroup.add(payPanel);
    let payLeader = null;
    const PAY = { W: 300, H: 15, STEP: 21, D: 6, LAB: 13 };   // px 空间（整块按 worldPerPx 缩放）

    function clearPay() {
      const kill = (o) => {
        if (o.geometry) o.geometry.dispose();
        if (o.material) { if (o.material.map) o.material.map.dispose(); o.material.dispose(); }
      };
      payPanel.traverse((o) => { if (o !== payPanel) kill(o); });
      while (payPanel.children.length) payPanel.remove(payPanel.children[0]);
      if (payLeader) { kill(payLeader); payGroup.remove(payLeader); payLeader = null; }
    }

    function buildPayload() {
      clearPay();
      payGroup.visible = !!(S.wire.board && S.sel != null);
      if (!payGroup.visible) return;
      const r = S.sel;
      const objs = model.netObjects.filter((o) => !o.comm);
      const bg = tokHex('--background'), bd = tokHex('--border-strong');
      const box = (w, h, d, color, op, x, y) => {
        const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d),
          new THREE.MeshBasicMaterial({ color, transparent: true, opacity: op, depthTest: false }));
        m.material.userData = { base: op, noPulse: true };
        m.position.set(x, y, 0); m.renderOrder = 9; payPanel.add(m);
        return m;
      };
      const lab = (text, color, hPx, x, y, anchor) => {
        const sp = makeLabel(text, color, 1);
        const L = sp.userData.lab;
        /* 父组已按 worldPerPx 缩放，这里必须用 px 空间自己定尺 ——
           makeLabel 内部的 syncLabelSize 是按世界尺寸算的，留着会被缩放两次。 */
        const w = hPx / L.aspect;
        sp.scale.set(w, hPx, 1);
        sp.center.set(anchor === 'right' ? 1 : 0, 0.5);
        sp.position.set(x, y, PAY.D); sp.renderOrder = 10;
        payPanel.add(sp);
        return sp;
      };

      let y = 0;
      objs.forEach((o) => {
        const carried = model.objCarry(o.id, r);
        const sh = carried ? model.objShard(o.id, r) : null;
        const fam = OPV[o.fam] || OPV.linear;
        const of = sh ? sh.of : 1, idx = sh ? sh.idx : 0;
        const focused = objOn() && S.obj === o.id;
        // 底槽：这个对象的全部分片（其余片在别的卡上 → 只给一层很淡的底）
        box(PAY.W, PAY.H, PAY.D * 0.6, bd, carried ? 0.30 : 0.16, 0, y);
        // 分片刻线：切成几份，一眼数得出
        if (of > 1 && of <= 32) {
          for (let i = 1; i < of; i++) {
            box(0.7, PAY.H, PAY.D * 0.62, bg, 0.55, -PAY.W / 2 + (PAY.W / of) * i, y);
          }
        }
        // 实心段 = 本卡持有的那一片（复制的对象 of=1 → 整条实心）
        if (carried) {
          const w = PAY.W / of;
          box(w, PAY.H * 1.16, PAY.D, fam, focused ? 1 : 0.9,
            -PAY.W / 2 + w * (idx + 0.5), y);
        }
        // 左侧对象名（缺席的压暗）· 右侧读数
        lab(o.short || o.name.split(/[ （(]/)[0], carried ? fam : bd, PAY.LAB,
          -PAY.W / 2 - 7, y, 'right');
        const read = !carried ? '—' : sh ? `${idx}/${of}` : '整份';
        lab(read, carried ? (sh ? dimc(sh.dim) : bd) : bd,
          PAY.LAB, PAY.W / 2 + 7, y, 'left');
        y -= PAY.STEP;
      });
      // 标题：这块板说的是哪张卡
      lab([{ t: `rank ${r}`, c: tokHex('--foreground') }], tokHex('--foreground'),
        PAY.LAB * 1.15, -PAY.W / 2 - 7, PAY.STEP * 0.95, 'right');
      lab('整条=全部分片 · 实心=本卡这一片', tokHex('--foreground-secondary'),
        PAY.LAB * 0.92, -PAY.W / 2, PAY.STEP * 0.95, 'left');
      // 引线：把板和它说的那张卡连起来（不缩放，走世界坐标，每帧随卡位重算）
      const g = new THREE.BufferGeometry().setFromPoints([V3(0, 0, 0), V3(0, 0, 0)]);
      const mm = new THREE.LineBasicMaterial({ color: bd, transparent: true, opacity: 0.95, depthTest: false });
      mm.userData = { base: 0.95, noPulse: true };
      payLeader = new THREE.Line(g, mm); payLeader.renderOrder = 8; payGroup.add(payLeader);
      syncPayload();
    }

    /* 位置与尺寸随相机走：整块板保持**恒定屏幕尺寸**（同「贴边字牌固定屏幕尺寸」那条纪律
       ——它是检视件，缩远了也得读得出），朝向锁相机，锚点跟着选中卡飞。 */
    const _pc = new THREE.Vector3(), _pr = new THREE.Vector3(), _pu = new THREE.Vector3();
    function syncPayload() {
      if (!payGroup.visible || S.sel == null) return;
      const k = worldPerPx();
      payPanel.scale.setScalar(k);
      payPanel.quaternion.copy(camera.quaternion);
      _pr.set(1, 0, 0).applyQuaternion(camera.quaternion);
      _pu.set(0, 1, 0).applyQuaternion(camera.quaternion);
      _pc.set(cur[S.sel * 3], cur[S.sel * 3 + 1], cur[S.sel * 3 + 2]);
      const objs = model.netObjects.filter((o) => !o.comm).length;
      /* 挂在卡的**正右方**、竖向以卡为中心：挂右上方时板顶常撞顶栏，引线也被拉得老长
         ——引线一长，「这块板说的是哪张卡」就得靠找，而那正是它唯一要交代的事。 */
      const dx = (PAY.W / 2 + 108) * k, dy = (objs - 1) * PAY.STEP * 0.5 * k;
      payPanel.position.copy(_pc).add(_pr.clone().multiplyScalar(dx)).add(_pu.clone().multiplyScalar(dy));
      if (payLeader) {
        const a = _pc.clone();
        const b = payPanel.position.clone()
          .add(_pr.clone().multiplyScalar(-(PAY.W / 2 + 6) * k))
          .add(_pu.clone().multiplyScalar(-(objs - 1) * PAY.STEP * 0.5 * k));
        payLeader.geometry.setFromPoints([a, b]);
        payLeader.geometry.attributes.position.needsUpdate = true;
      }
    }

    /* ── 卡上切分：把选中卡的正方体**就地切开**，点亮它持有的那一片 ──────────────
       这是「切分与正方体结合」最忠实的一版：不再是浮在旁边的板，而是这张卡的方块本身
       沿**当前聚焦对象的切分维**切成 of 段、亮第 idx 段（族色），其余是空槽（细线框）。
       换形态 = 换一根轴切（selectObject 会飞到该对象的主屏，轴自动对上）——正好对应
       「转动魔方去看另一个切面」。轴不写死：采样该维 +1 的邻居卡位置，看世界坐标沿哪根
       轴变、变多少，于是任何形态、任何配置都自对齐；若这一维在当前视图里被折叠成一点
       （邻居位置几乎不动），就画整块族色 + 交给信息卡说明，不画假的切法。 */
    const shardGroup = new THREE.Group(); shardGroup.renderOrder = 7; scene.add(shardGroup);
    const _sp0 = new THREE.Vector3(), _sp1 = new THREE.Vector3();

    function clearShard() {
      shardGroup.traverse((o) => {
        if (o === shardGroup) return;
        if (o.geometry) o.geometry.dispose();
        if (o.material) o.material.dispose();
      });
      while (shardGroup.children.length) shardGroup.remove(shardGroup.children[0]);
    }

    let shardPicks = [];
    /* ── 卡内魔方：把一张卡本身当成一个小魔方 ────────────────────────────────
       「卡内的卡内还是什么都没有」—— 说的就是这个：外面那层嵌套框把「这张卡在谁里面」
       讲清楚了，但**卡自己肚子里装的东西**一直没画。
       一张卡里装的是：它那一段的**每一层** × 每层里的**每类算子**，各持有一片。
       所以把它摊成一个小立方阵：
           X = 本卡持有的层（L20…L29）
           Y = 算子（QKV / Attn / FFN / 专家 / …，按整网图的算子族色）
           每格 = 该层该算子落在本卡上的那一份；本卡不承载的算子画成空框。
       这就是「卡也是一个魔方」——外层魔方按并行维排卡，卡内魔方按模型结构排算子。 */
    function buildCardCube() {
      const r = S.sel;
      const lr = model.stageLayerRange(model.ppOf(r));
      const objs = model.netObjects.filter((o) => !o.comm);
      const LN = Math.min(12, lr.hi - lr.lo + 1);          // 层多时只摆前 12 层，避免糊成一片
      const CW = CARD.x * 0.34, GAP = CW * 0.42;           // 小格边长与缝
      const w = LN * (CW + GAP), h = objs.length * (CW + GAP);
      const bd = tokHex('--border-strong');
      objs.forEach((o, yi) => {
        const carried = model.objCarry(o.id, r);
        const sh = carried ? model.objShard(o.id, r) : null;
        const fam = OPV[o.fam] || OPV.linear;
        for (let xi = 0; xi < LN; xi++) {
          const px = (xi - (LN - 1) / 2) * (CW + GAP);
          const py = ((objs.length - 1) / 2 - yi) * (CW + GAP);
          if (!carried) {
            const g = new THREE.EdgesGeometry(new THREE.BoxGeometry(CW, CW, CW));
            const m = new THREE.LineBasicMaterial({ color: bd, transparent: true, opacity: 0.3, depthTest: false });
            m.userData = { base: 0.3, noPulse: true };
            const ln = new THREE.LineSegments(g, m); ln.position.set(px, py, 0);
            ln.renderOrder = 7; shardGroup.add(ln);
          } else {
            const mm = new THREE.Mesh(new THREE.BoxGeometry(CW, CW, CW),
              new THREE.MeshBasicMaterial({ color: fam, transparent: true, opacity: sh ? 0.9 : 0.55, depthTest: false }));
            mm.material.userData = { base: sh ? 0.9 : 0.55, noPulse: true };
            mm.position.set(px, py, 0); mm.renderOrder = 7; shardGroup.add(mm);
          }
        }
        /* 行末挂算子名（本卡没有的压暗）——只在 opts.cardCubeLabels 不为 false
           时画。这两枚字牌钉在 3D 世界坐标上，跟着相机转，规模一大、卡挤在
           一起时会飘到宿主自己的 DOM 侧栏（选中卡详情、连线图例）那片地界
           上，两层信息叠在一起。右侧详情卡（detailEl，见 syncDetailCap）
           已经把同一句话（rank / 层区间 / 横纵轴 / 对象列表）摆得清清楚楚，
           这两枚字牌因此是可选的——默认还画（不改变任何既有消费者的样子），
           只有显式传了 cardCubeLabels:false 的宿主才会收起来，格子本身
           （下面那圈彩色小方块）照常画，少的只是文字。 */
        if (opts.cardCubeLabels !== false) {
          const lab = makeLabel(o.short || o.name.split(/[ （(]/)[0], carried ? fam : bd, 2.4);
          lab.position.set(-w / 2 - CARD.x * 0.28, ((objs.length - 1) / 2 - yi) * (CW + GAP), 0);
          lab.center.set(1, 0.5);
          lab.material.opacity = carried ? 1 : 0.45;
          lab.material.userData = { base: carried ? 1 : 0.45, noPulse: true };
          lab.renderOrder = 8; shardGroup.add(lab);
        }
      });
      if (opts.cardCubeLabels !== false) {
        // 顶部一枚牌交代这个小魔方的两根轴
        const cap = makeLabel(`卡内 · L${lr.lo}-L${lr.hi}`, tokHex('--foreground'), 3.4,
          `横=层 ×${LN} · 纵=算子 ×${objs.length}`);
        cap.position.set(0, h / 2 + CARD.y * 0.42, 0);
        cap.material.userData = { base: 1, noPulse: true };
        cap.renderOrder = 8; shardGroup.add(cap);
      }
    }

    function buildShard() {
      clearShard();
      shardPicks = [];
      /* 没聚焦某个对象时，卡内画「卡内魔方」（这张卡的层 × 算子）；
         聚焦了某个对象，才切换成那个对象的分片切分。两者都在「切分」图层下。 */
      if (S.wire.payload && !objOn() && S.sel != null) {
        /* opts.cardCube === false（?cc=0）：画布里整块「卡内魔方」彩色格子都不画了，
           不是只收字牌（那是 cardCubeLabels 管的）。用在宿主自己有一张详情卡把同一句话
           （rank / 层区间 / 对象持有情况）说得更清楚的场合——画布这时候只留场景本身，
           不重复摆一份。默认仍画，不改变任何既有消费者的样子。 */
        if (opts.cardCube === false) { shardGroup.visible = false; return; }
        shardGroup.visible = true; buildCardCube(); return;
      }
      const on = S.wire.payload && objOn() && S.sel != null;
      shardGroup.visible = on;
      if (!on) return;
      const r = S.sel, o = model.netObjBy[S.obj];
      if (!o || o.comm) { shardGroup.visible = false; return; }
      const carried = model.objCarry(o.id, r);
      const sh = carried ? model.objShard(o.id, r) : null;
      const fam = OPV[o.fam] || OPV.linear;
      const L = CARD.x * 1.5;                                   // 略大于选中卡（1.45）→ 包住它
      const solid = (w, h, d, x, y, z, color, op) => {
        const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d),
          new THREE.MeshBasicMaterial({ color, transparent: true, opacity: op, depthTest: false }));
        m.material.userData = { base: op, noPulse: true };
        m.position.set(x, y, z); m.renderOrder = 7; shardGroup.add(m); return m;
      };
      const wire = (w, h, d, x, y, z, color, op) => {
        const g = new THREE.EdgesGeometry(new THREE.BoxGeometry(w, h, d));
        const m = new THREE.LineBasicMaterial({ color, transparent: true, opacity: op, depthTest: false });
        m.userData = { base: op, noPulse: true };
        const ln = new THREE.LineSegments(g, m); ln.position.set(x, y, z); ln.renderOrder = 8; shardGroup.add(ln); return ln;
      };
      const bd = tokHex('--border-strong');

      if (!carried) { wire(L, L, L, 0, 0, 0, bd, 0.35); return; }     // 不承载 → 空框
      if (!sh || sh.of <= 1) { solid(L, L, L, 0, 0, 0, fam, 0.5); wire(L, L, L, 0, 0, 0, fam, 0.9); return; } // 复制 → 整块

      const tp = model.tpOf(r), pp = model.ppOf(r), rep = model.repOf(r), cp0 = model.cpOf(r);
      /* 「这一维在当前形态下落在哪根世界轴、朝哪个方向」——采样该维 +1 的邻居卡即可，
         不写死任何形态。双切对象要问两次，所以抽成函数。 */
      function axisOf(dimName) {
        const d = dimName === 'SP' ? 'TP' : dimName;
        let nb = r;
        if (d === 'TP') nb = model.rankOf((tp + 1) % TP, pp, rep, cp0);
        else if (d === 'CP') nb = model.rankOf(tp, pp, rep, (cp0 + 1) % CP);
        else if (d === 'PP') nb = model.rankOf(tp, (pp + 1) % PP, rep, cp0);
        else if (d === 'DP') nb = model.rankOf(tp, pp, (rep + 1) % REP, cp0);
        else if (d === 'EP') { const dom = model.domOf(r); nb = model.rankOf(tp, pp, dom * EP + ((model.epOf(r) + 1) % EP), cp0); }
        model.posOf(r, S.mode, _sp0); model.posOf(nb, S.mode, _sp1);
        const dv = { x: _sp1.x - _sp0.x, y: _sp1.y - _sp0.y, z: _sp1.z - _sp0.z };
        const ax = Math.abs(dv.x) >= Math.abs(dv.y) && Math.abs(dv.x) >= Math.abs(dv.z) ? 'x'
          : Math.abs(dv.y) >= Math.abs(dv.z) ? 'y' : 'z';
        return { ax, sign: dv[ax] >= 0 ? 1 : -1, flat: Math.abs(dv[ax]) < CARD.x * 0.12 };
      }

      /* ── 双切：被两维同时切的对象（attn_core = TP head × CP ctx）────────────
         这张卡持有的不是「第 i 片」，而是**二维网格里的一格** (i, j)。只画一维等于
         把话说了一半。两维落在不同世界轴时就切成网格；落在同一根轴（或有一维在本视图
         折成一点）时退回单维 + 由信息卡补齐另一维 —— 不画假的二维。 */
      const all = model.objShards(o.id, r);
      if (all.length >= 2) {
        const A = axisOf(all[0].dim);
        let B = axisOf(all[1].dim), folded = false;
        /* 两维落在**同一根世界轴**上是常态而非例外：本模型 CP 与 TP 就共用一根轴
           （lane = cp·TP + tp），attn_core 的 TP×CP 因此永远同轴。若就此退回单维，
           3D 里就永远表达不出双切 —— 而这正是要表达的东西。
           解法沿用本 pattern 已有的**折叠** idiom（DP平铺把板内 TP 折成「列×排」）：
           第二维改画到一根垂直轴上，并在牌上标「折」。这不是假坐标——那一维的分片
           确实沿第一根轴排布，折过来只是为了在一张卡上把二维读出来。 */
        if (!A.flat && !B.flat && A.ax === B.ax) {
          B = { ax: A.ax === 'y' ? 'x' : 'y', sign: 1, flat: false };
          folded = true;
        }
        if (!A.flat && !B.flat && A.ax !== B.ax) {
          const segA = L / all[0].of, segB = L / all[1].of, g = 0.16;
          for (let i = 0; i < all[0].of; i++) {
            for (let j = 0; j < all[1].of; j++) {
              const p = { x: 0, y: 0, z: 0 };
              p[A.ax] = A.sign * (i - (all[0].of - 1) / 2) * segA;
              p[B.ax] = B.sign * (j - (all[1].of - 1) / 2) * segB;
              const dims = { x: L, y: L, z: L };
              dims[A.ax] = segA * (1 - g); dims[B.ax] = segB * (1 - g);
              const me = i === all[0].idx && j === all[1].idx;
              const m2 = me ? solid(dims.x, dims.y, dims.z, p.x, p.y, p.z, fam, 0.92)
                            : wire(dims.x, dims.y, dims.z, p.x, p.y, p.z, bd, 0.34);
              m2.userData.shardPick = { obj: o.id, dim: all[0].dim, i, dim2: all[1].dim, j };
              shardPicks.push(m2);
              if (me) {
                const sp = makeLabel(`${all[0].range || i} × ${all[1].range || j}${folded ? ' 折' : ''}`, fam, 3.6);
                sp.position.set(p.x, p.y + L * 0.62, p.z);
                sp.material.userData = { base: 1, noPulse: true };
                sp.renderOrder = 9; shardGroup.add(sp);
              }
            }
          }
          return;
        }
      }

      // 采样切分维在当前形态下落在哪根轴（单维）
      const of = sh.of, idx = sh.idx;
      const dim = sh.dim === 'SP' ? 'TP' : sh.dim;
      const A1 = axisOf(dim);
      const ax = A1.ax;
      if (A1.flat) {                                                 // 这一维在本视图折成一点
        solid(L, L, L, 0, 0, 0, fam, 0.42); wire(L, L, L, 0, 0, 0, fam, 0.85);
        return;
      }
      const sign = A1.sign;
      const seg = L / of, gap = seg * 0.16;
      /* 片上直接标读数（而不是把话都放到旁边那个框里）：每片挂一枚小牌，
         本卡这片给族色 + 区间（E32-39），其余片给淡的片号——「第几片是谁」在片上就读得出。
         片数多时只标本卡这片与首尾，否则牌糊成一片（同轴刻度「摆不下就少摆」那条）。 */
      const labelEvery = of <= 8 ? 1 : Math.ceil(of / 6);
      for (let i = 0; i < of; i++) {
        const off = sign * (i - (of - 1) / 2) * seg;
        const p = { x: 0, y: 0, z: 0 }; p[ax] = off;
        const dims = { x: L, y: L, z: L }; dims[ax] = seg - gap;
        const me = i === idx;
        const m = me ? solid(dims.x, dims.y, dims.z, p.x, p.y, p.z, fam, 0.92)
                     : wire(dims.x, dims.y, dims.z, p.x, p.y, p.z, bd, 0.4);
        /* 片可点：点某片 → 跳到持有该片的 rank（同一形态、同一对象），于是这堆片
           本身就是「切换片」的控件，不用另做一排按钮。 */
        /* 拾取代理：没持有的片是线框，而 Raycaster 对 Line 的命中阈值只有几个世界单位
           的千分之一 —— 肉眼看着点在片里，实际一条棱都没碰到，于是「点不动」。
           补一个整片体积的透明实体专门用来接点击（opacity:0 仍可拾取；visible:false
           会被 Raycaster 直接跳过，不能用）。 */
        const proxy = new THREE.Mesh(new THREE.BoxGeometry(dims.x, dims.y, dims.z),
          new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false }));
        proxy.material.userData = { base: 0, noPulse: true };
        proxy.position.set(p.x, p.y, p.z); proxy.renderOrder = 6;
        proxy.userData.shardPick = { obj: o.id, dim, i };
        shardGroup.add(proxy);
        shardPicks.push(proxy);
        if (me || i === 0 || i === of - 1 || i % labelEvery === 0) {
          const txt = me ? (sh.range || `${i}/${of}`) : String(i);
          const sp = makeLabel(txt, me ? fam : bd, me ? 3.4 : 1.9);
          sp.position.set(p.x, p.y + L * 0.62, p.z);
          sp.material.opacity = me ? 1 : 0.5; sp.material.userData = { base: me ? 1 : 0.5, noPulse: true };
          sp.renderOrder = 9; shardGroup.add(sp);
        }
      }
    }
    /* ── 放大细节视窗 ────────────────────────────────────────────────────────
       卡上切分是「就地」的，忠实但也因此**被埋在卡阵与连线里**（几片只有几个像素，
       还被邻卡挡住）。所以另开一块只画这一件事的小视口：同一份几何克隆一份，
       放在自己的场景里、用自己的相机取景，画在画布左下角。
       朝向跟主相机走——主视图转，细节窗跟着转，两边看到的是同一个东西的同一面，
       不用在心里做一次旋转对应。 */
    /* 细节窗的取景与字号：视锥固定（不随主相机缩放），于是这一窗里「一片有多大」恒定，
       换配置、换对象都一样读得出。字牌按这个视锥自己的像素比定尺（见 buildDetail）。 */
    const DETAIL_PX = 260, DETAIL_LABEL_PX = 11, DETAIL_HALF = CARD.x * 1.5 * 1.22;
    let detailHalf = DETAIL_HALF;      // 实际用的半宽：随内容跨度变（见 buildDetail）
    const detailScene = new THREE.Scene();
    const detailCam = new THREE.OrthographicCamera(-1, 1, 1, -1, -400, 400);
    let detailRoot = null;
    function buildDetail() {
      if (detailRoot) { detailScene.remove(detailRoot); detailRoot = null; }
      if (!shardGroup.visible) return;
      /* 克隆而不是重建：几何与材质共享，因此细节窗与就地那份**永远一模一样**，
         不会出现「改了一处忘了另一处」。共享也意味着这里不能 dispose。 */
      detailRoot = shardGroup.clone(true);
      detailRoot.position.set(0, 0, 0);
      /* 视锥按**内容**定，不用固定常数：分片切分只有一张卡那么大，而卡内魔方是
         「层 × 算子」的一整片，跨度差好几倍——固定视锥会把后者的行标直接切在窗外
         （实机可见左边算子名只剩半个字）。取几何包围盒（不含字牌，字牌自己会缩）。 */
      const bb = new THREE.Box3();
      detailRoot.traverse((o) => { if (o.isMesh || o.type === 'LineSegments') bb.expandByObject(o); });
      if (!bb.isEmpty()) {
        const sz = new THREE.Vector3(); bb.getSize(sz);
        detailHalf = Math.max(CARD.x * 0.9, Math.max(sz.x, sz.y, sz.z) * 0.72);
      } else detailHalf = DETAIL_HALF;
      /* 字牌是**世界尺寸**的（在主场景里由 worldPerPx 定尺）。细节窗的视锥只有一点点，
         同一个世界尺寸搬进来就被放大成半个窗那么大（实机可见「E16-31」糊住整块）。
         所以按细节窗自己的像素比重算一遍：牌在这里也占固定几个像素。 */
      const wpp = (2 * detailHalf) / DETAIL_PX;
      detailRoot.traverse((o) => {
        if (!o.isSprite) return;
        const L = o.userData.lab;
        if (!L || !L.fontFrac) { o.scale.multiplyScalar(0.12); return; }
        const w = (DETAIL_LABEL_PX * wpp) / L.fontFrac;
        o.scale.set(w, w * L.aspect, 1);
      });
      detailScene.add(detailRoot);
    }
    const detailEl = document.createElement('div');
    detailEl.className = 'prc-detail';
    /* 细节窗里有**两条正交的导航**，必须各自说清楚变不变卡，否则方块的样子会让人以为
       它画的是「这张卡的内部」——而点一片其实是跳到别的卡：
         · 换**片**（同一对象的别的分片）→ **换 rank**：片本来就长在别的卡上；
         · 换**对象**（本卡持有的别的东西）→ **不换 rank**：这才是「在 rank 内部走」。
       后者原先只藏在侧栏清单里，3D 里没有入口。 */
    detailEl.innerHTML = '<div class="prc-detail-cap"></div>'
      + '<div class="prc-detail-foot">'
      + '<button class="prc-objstep" data-d="-1" type="button" title="本卡持有的上一个对象（不换卡）">‹</button>'
      + '<span class="prc-objnow"></span>'
      + '<button class="prc-objstep" data-d="1" type="button" title="本卡持有的下一个对象（不换卡）">›</button>'
      + '</div>';
    root.appendChild(detailEl);
    /* 本卡持有的对象序列（不含通信算子与不承载的）——「rank 内部」走的就是这一串。 */
    function carriedObjs(r) {
      return model.netObjects.filter((o) => !o.comm && model.objCarry(o.id, r));
    }
    detailEl.querySelectorAll('.prc-objstep').forEach((b) => {
      b.addEventListener('click', (ev) => {
        ev.stopPropagation();                       // 别让它落到画布的「点片」上
        if (S.sel == null || !objOn()) return;
        const list = carriedObjs(S.sel);
        if (!list.length) return;
        const i = Math.max(0, list.findIndex((o) => o.id === S.obj));
        const j = (i + (+b.dataset.d) + list.length) % list.length;
        api.selectObject(list[j].id, { fly: false });   // 换对象但不换卡、也不飞形态
      });
    });
    function syncDetailCap() {
      const on = shardGroup.visible && S.sel != null;
      detailEl.classList.toggle('show', !!on);
      if (!on) return;
      const o = model.netObjBy[S.obj];
      if (!o) {                                   // 没聚焦对象 → 细节窗里是「卡内魔方」
        const lr = model.stageLayerRange(model.ppOf(S.sel));
        const nObj = model.netObjects.filter((x) => !x.comm).length;
        detailEl.firstChild.innerHTML = `<b>rank ${S.sel}</b> · 卡内`
          + ` <span>L${lr.lo}-L${lr.hi}</span>`
          + `<em>横=层 · 纵=算子 ×${nObj} · 空框=本卡没有</em>`;
        const list0 = carriedObjs(S.sel);
        const now0 = detailEl.querySelector('.prc-objnow');
        if (now0) now0.innerHTML = `本卡对象 <b>${list0.length}</b>`;
        return;
      }
      const sh = model.objCarry(o.id, S.sel) ? model.objShard(o.id, S.sel) : null;
      /* 标题把「这画的是什么」说死：整块是**这个对象的全部分片**，不是这张卡的内部；
         只有亮的那一片属于本卡，其余片在别的卡上（所以点它们会换卡）。 */
      detailEl.firstChild.innerHTML =
        `<b>rank ${S.sel}</b> · ${esc(o.short || o.name)}`
        + (sh ? ` <span>${sh.idx}/${sh.of}</span>` : ' <span>整份</span>')
        + (sh ? `<em>全 ${sh.of} 片 · 亮的是本卡 · 其余在别的卡</em>` : '<em>复制 · 每张卡各一份</em>');
      const list = carriedObjs(S.sel);
      const i = list.findIndex((x) => x.id === S.obj);
      const now = detailEl.querySelector('.prc-objnow');
      if (now) now.innerHTML = `本卡对象 <b>${i < 0 ? '—' : i + 1}/${list.length}</b>`;
    }
    /* ── 细节窗里切换片 ─────────────────────────────────────────────────────
       「点片切换」的正确落点是这里，不是主场景：主场景里片只有几个像素、被邻卡和连线
       盖住，空框的片更是几乎点不中（线框的拾取阈值极小）——那是热区问题的根源。
       细节窗里每片都占几十像素、四周没有东西，点哪片就是哪片。
       主场景那份保留（点得中就点），信息卡那条选择条作为第三条路。 */
    function detailPick(ev) {
      if (!shardGroup.visible || !detailRoot) return null;
      const dr = detailEl.getBoundingClientRect();
      mouse.x = ((ev.clientX - dr.left) / dr.width) * 2 - 1;
      mouse.y = -((ev.clientY - dr.top) / dr.height) * 2 + 1;
      ray.setFromCamera(mouse, detailCam);
      const hits = ray.intersectObjects(detailRoot.children, true);
      for (const h of hits) {
        const d = h.object.userData && h.object.userData.shardPick;
        if (d) return d;
      }
      return null;
    }
    function shardRankOf(i, dim) {
      const r0 = S.sel; if (r0 == null) return null;
      const tp = model.tpOf(r0), pp = model.ppOf(r0), rep = model.repOf(r0);
      const cp = model.cpOf(r0);
      if (dim === 'TP') return model.rankOf(i % TP, pp, rep, cp);
      if (dim === 'CP') return model.rankOf(tp, pp, rep, i % CP);
      if (dim === 'PP') return model.rankOf(tp, i % PP, rep, cp);
      if (dim === 'DP') return model.rankOf(tp, pp, i % REP, cp);
      if (dim === 'EP') return model.rankOf(tp, pp, model.domOf(r0) * EP + (i % EP), cp);
      return null;
    }
    detailEl.addEventListener('click', (ev) => {
      const d = detailPick(ev); if (!d) return;
      const t = shardRankOf(d.i, d.dim);
      if (t != null) api.select(t);
    });
    detailEl.addEventListener('pointermove', (ev) => {
      const d = detailPick(ev);
      detailEl.classList.toggle('is-hit', !!d);
      if (!d) { tipEl.style.display = 'none'; return; }
      const o = model.netObjBy[S.obj];
      const rr = shardRankOf(d.i, d.dim);
      const rc = root.getBoundingClientRect();
      tipEl.style.display = 'block';
      tipEl.style.left = (ev.clientX - rc.left + 14) + 'px';
      tipEl.style.top = (ev.clientY - rc.top + 12) + 'px';
      tipEl.innerHTML = `第 ${d.i} 片 · ${esc(o.short || o.name)}`
        + (rr != null ? `<br><span style="opacity:.75">点一下 → 切到 rank ${rr}</span>` : '');
    });
    detailEl.addEventListener('pointerleave', () => {
      detailEl.classList.remove('is-hit'); tipEl.style.display = 'none';
    });

    /* 在主画面之上开一块自己的视口画细节窗。用 scissor 把这块矩形单独清一次底色，
       于是它是真正的「另一幅图」，不会和主场景的像素混在一起。 */
    function renderDetail() {
      if (!shardGroup.visible) return;
      const cr = renderer.domElement.getBoundingClientRect();
      const dr = detailEl.getBoundingClientRect();
      const w = dr.width, h = dr.height;
      if (w < 8 || h < 8) return;
      const x = dr.left - cr.left, yTop = dr.top - cr.top;
      const y = cr.height - (yTop + h);
      const half = detailHalf;
      const asp = w / h;
      detailCam.left = -half * asp; detailCam.right = half * asp;
      detailCam.top = half; detailCam.bottom = -half;
      /* 朝向与主相机一致、位置沿视线退到原点外：细节窗看到的是同一个东西的同一面，
         主视图转它就跟着转，读者不用在心里做一次旋转对应。 */
      detailCam.quaternion.copy(camera.quaternion);
      detailCam.position.copy(V3(0, 0, 1).applyQuaternion(camera.quaternion).multiplyScalar(60));
      detailCam.updateProjectionMatrix();
      const prevAuto = renderer.autoClear;
      renderer.autoClear = false;
      renderer.setScissorTest(true);
      renderer.setViewport(x, y, w, h);
      renderer.setScissor(x, y, w, h);
      renderer.setClearColor(tokHex('--surface-1') || tokHex('--background'), 1);
      renderer.clear(true, true, false);
      renderer.render(detailScene, detailCam);
      renderer.setScissorTest(false);
      renderer.setViewport(0, 0, cr.width, cr.height);
      renderer.setClearColor(tokHex('--background'), 1);
      renderer.autoClear = prevAuto;
    }

    function syncShard() {
      if (!shardGroup.visible || S.sel == null) return;
      shardGroup.position.set(cur[S.sel * 3], cur[S.sel * 3 + 1], cur[S.sel * 3 + 2]);
    }

    /* ── 通信组重建（选中 rank → 四维对端 + 连线 + 标签）── */
    /* 集合原语的数据流（对齐驾驶舱「算法展开」）：一个通信域按什么算法收发，就画什么形状。
         TP / DP = AllReduce → Ring（成环，前半程 ReduceScatter、后半程 AllGather）或 Tree（二叉树）
         PP      = P2P 接力  → 链
         EP      = AllToAll  → 域内互发（成员少画全连，多则退化成星形，避免边数爆炸）
       S.algo='auto' 时按上表选，也可强制 ring / tree。 */
    // AllToAll 画全连的成员上限：8 个成员就是 28 条边，糊成一片什么也看不清 →
    // 超过就退化成「以选中卡为中心的星形」（它自己收发谁，本来也是读图的重点）。
    const A2A_MESH_MAX = 5;
    function primOf(d) { return d === 'EP' ? 'AllToAll' : d === 'PP' ? 'P2P' : 'AllReduce'; }
    function algoOf(d) {
      if (d === 'EP' || d === 'PP') return d === 'EP' ? 'a2a' : 'chain';
      return S.algo === 'tree' ? 'tree' : 'ring';        // auto/ring → Ring AllReduce
    }
    /* 一个域的「走线」：返回若干条 rank 折线（不是坐标——每一段要知道两端是谁，
       才能判断它实际跨了哪层物理链路），并标出这一段该不该画成外凸弧。
       为什么要弧：一个通信组的成员在多数形态里是共线的（标准形态的 TP 组是一行、
       DP 组是一列），此时 Ring 的闭合边、Tree 的跨层边若也走直线，就与相邻链完全
       重叠——两种算法画出来一模一样，只有文字在变。跨越多个成员的「长边」因此外凸，
       Ring 成了「一条链 + 一道回程弧」，Tree 成了经典的弧形二叉树。 */
    function edgesOf(d, members) {
      const out = [], algo = algoOf(d);
      const seg = (ranks, arc) => out.push({ ranks, arc: !!arc });
      if (algo === 'chain') { seg(members.slice()); return out; }             // PP 接力链
      if (algo === 'ring') {                                                  // Ring：链 + 回程弧
        seg(members.slice());
        if (members.length > 2) seg([members[members.length - 1], members[0]], true);
        return out;
      }
      if (algo === 'tree') {                                                  // Tree：二叉树边
        for (let i = 1; i < members.length; i++) {
          const par = (i - 1) >> 1;
          seg([members[par], members[i]], i - par > 1);
        }
        return out;
      }
      if (members.length <= A2A_MESH_MAX) {                                   // AllToAll：全连
        for (let i = 0; i < members.length; i++) for (let j = i + 1; j < members.length; j++) seg([members[i], members[j]], j - i > 1);
      } else {                                                                // 成员多 → 退化成星形
        members.forEach((r) => { if (r !== S.sel) seg([S.sel, r]); });
      }
      return out;
    }
    /* 长边的外凸弧：二次贝塞尔，两端严格落在卡心（端点不许飘——「线没连到卡上」修过一次），
       只有中间鼓出去。鼓的方向取与边垂直、尽量朝上的那一侧；鼓的高度随边长增长但封顶。 */
    function arcPts(a, b) {
      const dir = b.clone().sub(a), len = dir.length();
      if (len < 1e-6) return [a, b];
      dir.divideScalar(len);
      let up = V3(0, 1, 0);
      if (Math.abs(dir.dot(up)) > 0.9) up = V3(0, 0, 1);
      const perp = up.sub(dir.clone().multiplyScalar(up.dot(dir))).normalize();
      // 鼓的高度：够把长边从相邻链上分开就行——按边长给一点、封顶在两格卡宽以内。
      // 早期按边长 30% 起鼓（封顶 9 格），几条弧就把整个模型盖住了。
      const c = a.clone().add(b).multiplyScalar(0.5).add(perp.multiplyScalar(Math.min(len * 0.12, CARD.x * 2.2)));
      const pts = [], K = 8;   // 8 段足够圆滑；再细会把重排动画期间的每帧重建拖慢
      for (let i = 0; i <= K; i++) {
        const t = i / K, u = 1 - t;
        pts.push(a.clone().multiplyScalar(u * u).add(c.clone().multiplyScalar(2 * u * t)).add(b.clone().multiplyScalar(t * t)));
      }
      return pts;
    }

    /* 一组成员在**当前这一屏**上塌成一个点了没有。
       正交 2D 会把一根世界轴折进视线，沿那根轴排开的通信组于是全部投影到同一个像素——
       实测：标准前 PP-DP 面里 TP 组跨度 0px、顶 PP-TP 面里 DP 与 EP 都是 0px、
       侧 TP-DP 面里 PP 是 0px。每个 2D 视角都至少有一维如此，这不是巧合而是折叠的定义。
       判定只看**屏幕上留下的那两根世界轴**（screenAxes），不看相机——同一个形态在同一屏
       下结论恒定，与缩放平移无关，也就不必挂在渲染循环里。 */
    function groupFlat(members) {
      const A = screenAxes(S.view);
      if (!A || members.length < 2) return false;      // 3D 没有被折的轴
      const v3 = { x: 0, y: 0, z: 0 };
      let h0 = Infinity, h1 = -Infinity, v0 = Infinity, v1 = -Infinity;
      members.forEach((r) => {
        model.posOf(r, S.mode, v3);
        h0 = Math.min(h0, v3[A.h]); h1 = Math.max(h1, v3[A.h]);
        v0 = Math.min(v0, v3[A.v]); v1 = Math.max(v1, v3[A.v]);
      });
      const eps = CARD.x * 0.5;
      return (h1 - h0) < eps && (v1 - v0) < eps;
    }
    const flatDims = () => (S.sel == null ? [] : peerDims.filter((d) => groupFlat(model.commGroup(S.sel, d))));
    function rebuildComm() {
      clearComm();
      peerMeshes.forEach((m) => { m.geometry.setDrawRange(0, 0); m.visible = false; });
      moverPaths = [];
      outlineBoxes.forEach((o) => { o.visible = false; });
      buildRelSet();               // 关联集合与连线同源：谁被画成对端，谁就不被聚焦压暗
      if (S.sel == null) return;
      const gp = (r) => V3(cur[r * 3], cur[r * 3 + 1], cur[r * 3 + 2]);
      const curDim = PHASES[phaseIdx()].dim;
      peerDims.forEach((d, di) => {
        const members = model.commGroup(S.sel, d);
        const mesh = peerMeshes[di];
        const buf = mesh.geometry.attributes.position;
        const V = EDGE_TPL.length;                 // 每个成员 24 点 × 3 分量
        let n = 0;
        members.forEach((r) => {
          if (r === S.sel || n >= PEER_MAX) return;
          const p = gp(r), base = n * V;
          for (let k = 0; k < V; k += 3) {
            buf.array[base + k] = EDGE_TPL[k] + p.x;
            buf.array[base + k + 1] = EDGE_TPL[k + 1] + p.y;
            buf.array[base + k + 2] = EDGE_TPL[k + 2] + p.z;
          }
          n++;
        });
        mesh.geometry.setDrawRange(0, n * (V / 3));
        buf.needsUpdate = true;
        mesh.visible = S.wire.members && n > 0;
        const colorHex = new THREE.Color(dimc(d)).getHex();
        // 当前阶段主导的那一维加一档亮度与粗细（四维一律清晰可见，只是主导维更亮）
        const on = curDim === d;
        // 线要细：一屏可能同时有四维的走线，管壁按卡宽的 1/12 起算，主导维再粗一档。
        const op = on ? 0.95 : 0.45, rad = (on ? 1.2 : 0.8) * CARD.x * 0.085;
        mesh.material.opacity = on ? 0.7 : 0.28;
        /* 折叠维**不画走线**：它的成员在这一屏全落在同一个像素上，画出来是一根长度为 0 的
           管子——读者只会认为图坏了。更糟的是画面与文案会打架：图例把这一维加粗成
           「此刻主导维」、信息卡也写着「此刻 TP 主导」，而画面上一根青线都找不到。
           不画，改由图例那一行注明「这一屏折进视线 · N 张卡重合」，并指去哪一屏看。 */
        const flat = groupFlat(members);
        const segs = flat ? [] : edgesOf(d, members);
        const paths = segs.map((sg) => (sg.arc ? arcPts(gp(sg.ranks[0]), gp(sg.ranks[1])) : sg.ranks.map(gp)));
        if (S.wire.lines) {
          // 折线逐段建管：每段两点，端点与拐点都严格落在卡心（整条折线交给一个
          // TubeGeometry 会按弧长采样，中间拐点被抄近路）。弧（长边）整条画。
          // 走线只按「维」着色：试过按物理层级逐段上色，线太细、又和 TP 组重合，
          // 基本看不出层级差别 → 物理的事交给信息卡的段数统计（文字更准）。
          segs.forEach((sg, i) => {
            if (sg.arc) { commLine(paths[i], colorHex, op, rad, { dim: d, ranks: sg.ranks }); return; }
            const pts = paths[i];
            for (let k = 1; k < pts.length; k++) {
              commLine([pts[k - 1], pts[k]], colorHex, op, rad, { dim: d, ranks: [sg.ranks[k - 1], sg.ranks[k]] });
            }
          });
        }
        if (on) moverPaths = paths.map((pts) => ({ pts, color: colorHex }));   // 粒子只跑「此刻」这一维
        // 域轮廓：把这一组的成员用一个线框包起来——切到对应形态时组会 snap 成整块，
        // 轮廓于是直接画出「这一组在这种堆法下是什么形状」（对齐驾驶舱 COMM 镜头的域轮廓）
        if (S.wire.outline) {
          const box = outlineBoxes[di];
          const bb = new THREE.Box3();
          members.forEach((r) => bb.expandByPoint(gp(r)));
          bb.expandByScalar(CARD.x * 0.75);
          const size = bb.getSize(new THREE.Vector3()), ctr = bb.getCenter(new THREE.Vector3());
          box.scale.set(Math.max(size.x, 0.01), Math.max(size.y, 0.01), Math.max(size.z, 0.01));
          box.position.copy(ctr);
          box.material.color.set(colorHex);
          box.material.opacity = on ? 0.8 : 0.3;
          box.visible = true;
        }
      });
      // 选中卡上方原先挂一句「此刻 TP · Ring ReduceScatter 段 · 其余三维…」：
      // 同样是散文，而且信息卡与图例已经在说同一件事 → 从 3D 里撤掉（同 axNotes 的道理）。
    }

    /* ── HUD / 图例 / 粒度贴士 / 信息卡 ── */
    function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;'); }
    /* 左下的「读图钥匙」HUD 已删除：常驻大段文字太占画面。形态的读法进了「形态」问号
       气泡（随当前形态变化），异常读法在「注入」问号里，当前阶段看时间轴与图例。 */
    function renderHud() { }
    /* 「每格=几张卡」曾经是画面右上角一枚常驻贴士。撤掉了：本 pattern 自己的约定就是
       「解释一律收进问号气泡，画面只留几何」，它是最后一处例外。信息一句没少——
       现在写在「视角」问号气泡的「此刻」一行里（见 DYN.views），且比原来更完整
       （多说了正在翻哪一层）。 */
    function granText() {
      const d = curDepth();
      if (!d) return '';
      const rest = d.fold / d.slice.n;   // 开剖面后仍被折叠的卡数（多折叠维时 > 1）
      if (!S.sliceOn) return `每格 ${d.fold} 张卡重叠（${d.label} 折入视线，可开剖面逐层翻）`;
      return rest > 1
        ? `每格 ${rest} 张卡重叠（正翻 ${d.slice.lab}=${S.sliceVal}，其余维仍折叠）`
        : `每格 1 张卡（正翻 ${d.slice.lab}=${S.sliceVal}）`;
    }
    // 图例必须跟着「当前卡块着色」走：分组着色时列出各组的实际配色，负载着色时给色带，
    // 注入异常时给异常组——否则切了着色图例纹丝不动，读者按图例根本对不上画面。
    /* 图例 = 只回答「这个颜色叫什么」：分「着色 / 连线」两段，每段一行一条，
       条目只有色块 + 名字。为什么、怎么读（此刻主导是什么意思、压暗代表什么、
       同色非同组…）一律收进对应那排控件的问号气泡——图例不承担解释。 */
    function renderLegend() {
      const lg = $('.prc-legend'); if (!lg) return;
      const row = (c, t) => `<div class="prc-lgrow"><i style="background:${c}"></i><span>${esc(t)}</span></div>`;
      const sec = (t) => `<div class="prc-lgsec">${esc(t)}</div>`;
      const parts = [];
      if (objOn()) {
        /* 对象透镜接管了着色 → 图例这一段解释的就是画面上真正出现的那批颜色。 */
        const o = model.netObjBy[S.obj];
        parts.push(sec(`着色 · 整网对象「${o.name}」`));
        if (o.comm) {
          const gl = o.alias || o.comm;
          parts.push(row(dimc(o.comm), `${gl} 组 · ${o.prim}`));
          parts.push(S.sel == null
            ? `<div class="prc-lgrow"><i style="background:transparent"></i><span class="prc-dim">通信算子属于通信组、不属于单张卡 —— 先选一张卡，才有「哪个组」可高亮</span></div>`
            : row(rgbCss(restColor(2)), '组外'));
        } else if (!o.by.length) {
          parts.push(row(tokHex('--foreground-secondary'), '复制 · 每张卡各一份完整的'));
        } else {
          /* 图例的「切成几片、沿哪根轴」必须与着色同源 —— 着色走 objShard（承载者按
             持有第几片着色），所以这里也走 objShard，别用 o.by[0]。否则 attn_core 这类
             被两维同时切的对象，CP>1 时着色按 CP、图例却仍写 TP head，画面与图例打架。
             用选中卡问一份代表分片；没选卡时退回 o.by[0] 的静态口径。 */
          const rep = S.sel != null ? S.sel : (carrySet && carrySet.size ? [...carrySet][0] : 0);
          const rsh = model.objShard(o.id, rep);
          const b = rsh ? { dim: rsh.dim, axisName: rsh.axis } : { dim: o.by[0].dim, axisName: null };
          const n = rsh ? rsh.of : (o.by[0].dim === 'EP' ? EP : TP);
          const axName = b.axisName || (o.by[0].axis === 'expert' ? '专家' : o.by[0].axis === 'head' ? '注意力头' : o.by[0].axis === 'vocab' ? '词表' : o.by[0].axis === 'seq' ? '序列' : o.by[0].axis === 'ctx' ? '上下文' : '张量');
          const MAXC = 6, shown = Math.min(n, MAXC);
          for (let i = 0; i < shown; i++) parts.push(row(groupColor(i), `第 ${i} 片（沿${axName}轴）`));
          if (n > shown) parts.push(`<div class="prc-lgrow"><i style="background:transparent"></i><span class="prc-dim">… 共 ${n} 片（${b.dim} 切）</span></div>`);
          if (carrySet && carrySet.size < N) parts.push(row(rgbCss(restColor(2)), '不承载这个对象的卡'));
        }
        lg.innerHTML = parts.join('');
        return;
      }
      if (S.anom !== 'none') {
        const what = { tp: 'TP 槽 0', pp: 'PP 级 0', dp: 'DP 副本 0', ep: `EP 桶 ${anomBucket()}` }[S.anom];
        parts.push(sec('着色 · 异常注入'), row('var(--danger)', `异常组 ${what}`), row(rgbCss(restColor(2)), '其余'));
      } else if (S.colorBy === 'load') {
        parts.push(sec('着色 · 状态热力'),
          `<div class="prc-lgrow prc-ramp"><i></i><span>负载 低→高</span></div>`);
      } else if (S.colorBy === 'neutral') {
        parts.push(sec('着色 · 素色'),
          row(tokHex('--foreground-secondary'), '不按维度/负载区分——先看阵列形状'));
      } else if (MEM_LENS_KEY[S.colorBy]) {
        /* 颜色画的是「这张卡在全网 min→max 之间排第几」，不是「占这张卡总内存的百分之几」
           ——后者被参数量的量级摊平成一条死色，前者才拉得出结构性差异（见 colorOfRank
           上方大注）。数字给全网真实的两个端点（GB），让「满屏一种颜色」与「另一种颜色」
           之间到底差多少 GB 有个量感——图例只保证颜色与文字读同一份数，不单独再编一遍。 */
        const LAB = { w: '权重占比', act: '激活占比', grad: '梯度占比', opt: '优化器态占比' };
        const WHY = {
          w: '这条纹路 = 「谁背着 embedding/lm_head」（PP 首/末段）+「谁的专家桶因除不尽 EP 多摊一个专家」两件结构性事实的叠加；其余卡在均衡并行下权重完全相等，理应同色。',
          act: '1F1B 热身期里，越靠前的 PP 段攒着越多在途微批，激活字节数随 PP 段号单调走低——一屏就看出流水线的"水位差"，是四档里全网差距最大的一个。',
          grad: '梯度与权重同源（都正比于参数量 P），归一后与权重那张图长得一模一样——留着做对照：两张图对不上，说明某处字节口径算错了，不是花纹本身有问题。',
          opt: '优化器态同样正比于参数量 P（fp32 主参数+一阶矩+二阶矩，是五档里字节数最大的一档），归一后与权重/梯度同一条纹路——真正值得看的是它的绝对体量（上面这行 GB 数），不是它的着色形状。',
        };
        const rg = memRangeGB(S.colorBy);
        parts.push(sec(`着色 · ${LAB[S.colorBy]}`),
          `<div class="prc-lgrow prc-ramp"><i></i><span>全网相对位置 低→高</span></div>`,
          `<div class="prc-lgrow"><i style="background:transparent"></i><span class="prc-dim">全网 ${rg.lo.toFixed(1)}GB → ${rg.hi.toFixed(1)}GB（该档字节数的两个真实端点；色带把这个区间线性铺成 0→1）</span></div>`,
          `<div class="prc-lgrow"><i style="background:transparent"></i><span class="prc-dim">${WHY[S.colorBy]}</span></div>`);
      } else {
        const key = S.colorBy, pl = model.placement;
        const n = key === 'tp' ? TP : key === 'pp' ? PP : key === 'dp' ? REP
          : key === 'host' ? pl.hosts : key === 'pod' ? pl.pods : EP;
        const lab = { tp: 'TP', pp: 'PP', dp: 'DP副本', ep: 'EP桶', host: '主机', pod: 'Pod' }[key];
        const MAXC = 6, shown = Math.min(n, MAXC);
        parts.push(sec(`着色 · 按 ${lab} 分组`));
        for (let i = 0; i < shown; i++) parts.push(row(groupColor(i), `${lab}${i}`));
        if (n > shown) parts.push(`<div class="prc-lgrow"><i style="background:transparent"></i><span class="prc-dim">… 共 ${n} 组</span></div>`);
      }
      // 剖面外的卡已经不画（在屏幕上与剖面内那张完全重合），图例里也就没有这一条可对
      if (focusOn()) parts.push(row(dimSwatch(1), '无关卡'));
      if (S.sel != null) {
        const anyOn = S.wire.members || S.wire.lines || S.wire.outline || S.wire.movers;
        if (anyOn) {
          const cur = PHASES[phaseIdx()].dim;
          const flat = flatDims();
          parts.push(sec('连线 · 通信域'));
          peerDims.forEach((dm) => {
            const algo = dm === 'EP' ? 'AllToAll' : dm === 'PP' ? 'P2P 链'
              : `AllReduce ${S.algo === 'tree' ? 'Tree' : 'Ring'}`;
            /* 画不出来的那一维要**在图例里说清楚**，而不是留一行空指望。
               这一行照旧列出（这一维的通信是真实存在的，只是这一屏画不了），
               但压暗 + 注明「折进视线 · N 卡重合」，并指出去哪一屏看。 */
            const isFlat = flat.indexOf(dm) >= 0;
            const n = model.commGroup(S.sel, dm).length;
            const note = isFlat ? `<span class="prc-dim">（折进视线 · ${n} 卡重合，去 3D 看）</span>` : '';
            parts.push(`<div class="prc-lgrow${dm === cur && !isFlat ? ' is-now' : ''}${isFlat ? ' is-flat' : ''}">`
              + `<i style="background:${dimc(dm)}"></i><span>${esc(dm + ' ' + algo)}${note}</span></div>`);
          });
        }
      }
      lg.innerHTML = parts.join('');
    }
    // 此刻主导维走到集合原语的哪一段（Ring 前半 ReduceScatter / 后半 AllGather）
    function ringStage() {
      const d = PHASES[phaseIdx()].dim;
      if (primOf(d) === 'AllReduce' && algoOf(d) === 'ring') return `Ring ${phaseU() < 0.5 ? 'ReduceScatter' : 'AllGather'} 段`;
      return primOf(d);
    }
    // 选中的那条通信边（点线之后）：谁到谁、跨了哪层物理链路
    function edgeLine() {
      const e = S.selEdge; if (!e) return '';
      return `<b>${e.dim} ${e.prim}</b> rank ${e.from} → ${e.to}` +
        ` · <span style="color:${tierc(e.tier)}">${TIER_LAB[e.tier]}</span>` +
        `<span class="prc-dim">（机${e.hosts[0]}→${e.hosts[1]} · Pod${e.pods[0]}→${e.pods[1]}）</span>`;
    }
    /* 点选详情面板：结构照搬设计系统 sidecar 的 inspector——kicker（类别）→ 标题（是谁）
       → 一句定义 → 键值表（取值）→ 结论条（此刻状态 / 选中的那条边）。 */
    function kv(lab, val) {
      return `<div class="prc-kvrow"><span class="prc-kvlab">${lab}</span><span class="prc-kvval">${val}</span></div>`;
    }
    function renderInfo() {
      const info = $('.prc-info'); if (!info) return;
      if (S.sel == null) { info.classList.remove('show'); info.innerHTML = ''; return; }
      const r = S.sel, st = model.ppOf(r), lr = model.stageLayerRange(st), e = model.epOf(r);
      const ph = PHASES[phaseIdx()], pl = model.placement;
      info.classList.add('show');
      const tally = physTally(r), edge = edgeLine();
      info.innerHTML =
        `<button class="prc-infoclose btn btn-sm" type="button" aria-label="取消选中">${ICON.close}</button>` +
        `<div class="prc-kicker">RANK</div>` +
        `<div class="prc-title">rank ${r} <span class="prc-dim">/ ${N}</span></div>` +
        `<p class="prc-prose">这张卡同时属于四个通信域：换形态只改变它摆在哪，不改变下面这四个身份。</p>` +
        `<div class="prc-kv">` +
        kv('TP 槽位', `<b style="color:${dimc('TP')}">TP${model.tpOf(r)}</b> <span class="prc-dim">/ ${TP}</span>`) +
        kv('PP 段', `<b style="color:${dimc('PP')}">PP${st}</b> <span class="prc-dim">S${st}·L${lr.lo}-${lr.hi}</span>`) +
        kv('DP 副本', `<b style="color:${dimc('DP')}">${model.repOf(r)}</b> <span class="prc-dim">/ ${REP}</span>`) +
        kv('EP 桶 · A2A 域', `<b style="color:${dimc('EP')}">桶${e}</b> <span class="prc-dim">${model.expRange(e)} · 域${model.domOf(r)}</span>`) +
        kv('物理落位', `<b>机${model.hostOf(r)} · Pod${model.podOf(r)}</b> <span class="prc-dim">${pl.cardsPerHost} 卡/机</span>`) +
        `</div>` +
        objLine(r) + objRoster(r) +
        /* 主导维在这一屏画不出来时要**当场说明**：不说的话，这行写着「此刻 TP 主导」、
           图例把 TP 加粗，而画面上一根青线都没有，读者只会认为图坏了。
           通信本身照旧存在（所以这一行照旧给量），缺的只是「这一屏画不出来」这句话。 */
        `<div class="prc-status">此刻 <b style="color:${dimc(ph.dim)}">${ph.dim}</b> 主导 · ${esc(ringStage())}` +
        (tally ? ` · ${tally.replace(/^<br>/, '')}` : '')
        + (flatDims().indexOf(ph.dim) >= 0
          ? `<br><span class="prc-dim">这一屏 ${ph.dim} 被折进视线，整组重合成一点 → 走线不画，切 3D 看</span>` : '')
        + `</div>` +
        (edge ? `<div class="prc-status is-edge">${edge.replace(/^<br>/, '')}</div>` : '');
      const close = info.querySelector('.prc-infoclose');
      if (close) close.addEventListener('click', () => api.select(null));
      /* 清单行 → 选中该对象（走与「整网对象」那一排按钮同一条路径：着色 + 飞主屏），
         选中的卡不变，于是「这张卡的这一片」在新形态里是接着看的，不是重新找。 */
      info.querySelectorAll('.prc-rosterrow').forEach((b) => {
        b.addEventListener('click', () => api.selectObject(b.dataset.obj));
      });
      /* 片选择条 → 跳到持有那一片的 rank（其余坐标不动，形态/对象/视角接着看）。 */
      info.querySelectorAll('.prc-shardchip').forEach((b) => {
        b.addEventListener('click', () => {
          const i = +b.dataset.shard, d = b.dataset.sdim, r0 = S.sel;
          if (r0 == null) return;
          const tp = model.tpOf(r0), pp = model.ppOf(r0), rep = model.repOf(r0);
          let t = null;
          if (d === 'TP') t = model.rankOf(i % TP, pp, rep);
          else if (d === 'PP') t = model.rankOf(tp, i % PP, rep);
          else if (d === 'DP') t = model.rankOf(tp, pp, i % REP);
          else if (d === 'EP') t = model.rankOf(tp, pp, model.domOf(r0) * EP + (i % EP));
          if (t != null) api.select(t);
        });
      });
    }

    /* 片选择条：一排编号 = 这个对象的全部分片，当前这片高亮，点别的片就跳到持有它的 rank。
       片数很多时（如 vocab 按 TP 切成 64 片）只摆一排数字会太长 → 超过 24 片时改成
       「上一片 / 读数 / 下一片」的步进器，两端仍可直达首尾。 */
    function shardStrip(r, o, sh) {
      if (!sh || sh.of <= 1) return '';
      const dim = sh.dim === 'SP' ? 'TP' : sh.dim, c = dimc(sh.dim);   // 组走 TP（SP 复用它），色用 SP 自己的
      const btn = (i, txt, cls) =>
        `<button class="prc-shardchip${cls || ''}${i === sh.idx ? ' is-cur' : ''}" type="button"`
        + ` data-shard="${i}" data-sdim="${dim}" style="--c:${c}"`
        + ` title="第 ${i} 片 / ${sh.of}">${esc(txt)}</button>`;
      let body;
      if (sh.of <= 24) {
        body = Array.from({ length: sh.of }, (_, i) => btn(i, i)).join('');
      } else {
        const prev = (sh.idx - 1 + sh.of) % sh.of, next = (sh.idx + 1) % sh.of;
        body = btn(0, '⏮', ' is-step') + btn(prev, '‹', ' is-step')
          + `<span class="prc-shardnow" style="--c:${c}">${sh.idx} / ${sh.of}</span>`
          + btn(next, '›', ' is-step') + btn(sh.of - 1, '⏭', ' is-step');
      }
      return `<div class="prc-shardrow"><span class="prc-shardlab">切到第几片</span>`
        + `<span class="prc-shardstrip">${body}</span></div>`;
    }

    /* ── 装载清单：把这张卡展开成「整网的哪一堆碎片」──────────────────────────
       与 objLine 正好是一对**互逆**的读法，缺哪一个都只讲了一半：
         objLine  ：选一个对象 → 它被切成什么样、这张卡拿到第几片（对象视角）；
         objRoster：选一张卡  → 它是整网的哪一堆碎片（rank 视角）。
       后者才回答「rank 本身也被整网切分」这件事，而且它给出的是**三种并存的状态**：
         ✗ 完全没有（词嵌入只落 PP 首段——对整网的绝大部分，这张卡持有的是「无」）
         ● 完整一份（router 权重是复制的，不被任何维切）
         ▧ 第 n/m 片（其余）
       「这张卡持有模型的几分之一」因此没有单一答案：每个对象切的维不同、份数不同。
       没选对象时默认给清单（一张卡的全貌），选了对象就收敛到那一个（避免两段重复）。
       行是可点的：点一行 = 选中该对象 → 透镜着色 + 飞到它的主屏，清单于是也是导航。 */
    function objRoster(r) {
      if (objOn()) return '';                       // 已聚焦某个对象 → 由 objLine 负责
      const bands = [];
      model.netObjects.forEach((o) => {
        let b = bands.find((x) => x.name === o.band);
        if (!b) bands.push(b = { name: o.band, items: [] });
        b.items.push(o);
      });
      const row = (o) => {
        const dot = `<i class="prc-fam" style="background:${OPV[o.fam] || OPV.linear}"></i>`;
        let state, cls = '';
        if (o.comm) {
          const g = model.commGroup(r, o.comm);
          const full = o.comm === 'DP' ? REP : g.length;   // DP 组画面按采样显示，这里给真值
          state = `<span style="color:${dimc(o.comm)}">${esc(o.alias || o.comm)} 组</span> <span class="prc-dim">${full} 员</span>`;
          cls = ' is-comm';
        } else if (!model.objCarry(o.id, r)) {
          state = `<span class="prc-dim">完全没有</span>`; cls = ' is-none';
        } else {
          const sh = model.objShard(o.id, r);
          state = sh
            ? `<span style="color:${dimc(sh.dim)}">${sh.idx}</span><span class="prc-dim">/${sh.of}</span>`
              + (sh.range ? ` <span class="prc-dim">${esc(sh.range)}</span>` : '')
            : `<span class="prc-dim">完整一份</span>`;
          cls = sh ? '' : ' is-whole';
        }
        return `<button class="prc-rosterrow${cls}" type="button" data-obj="${o.id}" title="${esc(o.note || '')}">`
          + `${dot}<span class="prc-rostername">${esc(o.name)}</span><span class="prc-rosterval">${state}</span></button>`;
      };
      const lr = model.stageLayerRange(model.ppOf(r));
      return `<div class="prc-kicker" style="margin-top:9px">装载清单 · 这张卡是整网的哪一堆碎片</div>`
        + `<p class="prc-prose">rank 不是先存在、再由对象落上去——它就是六维切分切出来的那个格子。`
        + `本段只有 <b>L${lr.lo}-L${lr.hi}</b>（${lr.hi - lr.lo + 1}/${model.config.layers} 层），`
        + `其余 ${model.config.layers - (lr.hi - lr.lo + 1)} 层这张卡一个字节都没有。点一行看那个对象。</p>`
        + bands.map((b) => `<div class="prc-rosterband">${esc(b.name)}</div>` + b.items.map(row).join('')).join('')
        /* 两个色系的解释放悬停里，不占四行散文（同「行尾说明一律进问号气泡」那条纪律）。 */
        + `<p class="prc-prose prc-dim" title="族色取自已发布的整网图 pattern（model-architecture-3d-deck / …-training-sidecar）的 COLOR_FALLBACKS，其 key 与 openpangu 图节点的 opv:* colorKey 同源，两边同一族算子同一个色。维度签名色则是 TP/PP/DP/EP/CP/SP 那一套。">`
        + `<b>色点</b> = 算子族（同整网图 pattern） · <b>数字色</b> = 切它的那一维</p>`;
    }

    /* 选中整网对象时，信息卡多给一段：**这张卡持有这个对象的哪一片**。
       这正是「把算子装进对应的 rank 里」那句话在单卡尺度上的读数——
       不是「这个算子归 TP」，而是「Q 升维投影沿注意力头轴切成 8 份，本卡持有 h0-h7」。 */
    function objLine(r) {
      if (!objOn()) return '';
      const o = model.netObjBy[S.obj]; if (!o) return '';
      const head = `<div class="prc-kicker" style="margin-top:9px">整网对象</div>`
        + `<div class="prc-objname">${esc(o.name)}</div>`;
      if (o.comm) {
        const g = model.commGroup(r, o.comm);
        return head + `<div class="prc-kv">`
          + kv('通信组', `<b style="color:${dimc(o.comm)}">${esc(o.alias || o.comm)} 组</b> <span class="prc-dim">${g.length} 员</span>`)
          + kv('集合原语', `<b>${esc(o.prim)}</b>`)
          + `</div><p class="prc-prose">通信算子不属于单张卡，而属于一个通信组：组内每张卡上各有一个实例。上面这一组就是本卡参与的那个。</p>`;
      }
      const sh = model.objShard(S.obj, r);
      const carried = model.objCarry(S.obj, r);
      if (!carried) {
        return head + `<div class="prc-status">本卡<b>不承载</b>这个对象`
          + `<span class="prc-dim">（${o.firstStage ? '只落在 PP 首段' : o.lastStage ? '只落在 PP 末段' : '不在本段'}，本卡在 PP${model.ppOf(r)}）</span></div>`;
      }
      if (!sh) {
        return head + `<div class="prc-status">复制 —— 每张卡各持一份完整的<span class="prc-dim">（不被任何维切开）</span></div>`;
      }
      /* 被多维同时切的对象（attn_core = TP head × CP ctx）要**每一维各给一行**：
         只报主导那一维等于把话说了一半。单维对象 objShards 退化成 [shard]，同一段代码。 */
      const alls = model.objShards(o.id, r);
      const grid = alls.length >= 2;
      return head + `<div class="prc-kv">`
        + alls.map((x) => kv(`${x.dim} 沿${esc(x.axis)}切`,
          `<b style="color:${dimc(x.dim)}">第 ${x.idx} 片</b> <span class="prc-dim">/ ${x.of}</span>`
          + (x.range ? ` <span class="prc-dim">${esc(x.range)}</span>` : ''))).join('')
        + (grid ? kv('这张卡 = 网格一格',
          `<b>${alls.map((x) => x.idx).join(' × ')}</b> <span class="prc-dim">/ ${alls.map((x) => x.of).join(' × ')} = ${alls.reduce((a, x) => a * x.of, 1)} 格</span>`) : '')
        + `</div>`
        /* 片选择条：切换片的**可靠**入口。在 3D 里点小方块受遮挡与透视影响，
           很容易误触（尤其片是空框时）；这里一排编号，点哪片就跳到持有那片的 rank。
           两条路并存：场景里点得中就点，点不中来这儿。 */
        + model.objShards(o.id, r).map((x) => shardStrip(r, o, x)).join('')
        + (o.partial ? `<div class="prc-status">行切算子 —— 本卡算出的是 <b>partial sum</b>，要在 TP 组内归约才完整</div>` : '');
    }

    /* 「此刻这一维的走线各跨了哪层」——3D 里画层级色线看不出来（线太细、又和 TP 组
       重合），改用这一行文字给量：跨 Pod 的段越多，这次集合越贵。 */
    function physTally(r) {
      const d = PHASES[phaseIdx()].dim;
      const segs = edgesOf(d, model.commGroup(r, d));
      const cnt = { ub: 0, rail: 0, out: 0 };
      segs.forEach((sg) => { const rs = sg.ranks; for (let i = 1; i < rs.length; i++) cnt[model.tierOf(rs[i - 1], rs[i])]++; });
      const tot = cnt.ub + cnt.rail + cnt.out;
      if (!tot) return '';
      const one = (k, lab) => (cnt[k] ? `<span style="color:${tierc(k)}">${lab} ${cnt[k]}</span>` : '');
      return `<br><span class="prc-dim">此刻 ${d} 走线 ${tot} 段：</span> ` +
        [one('ub', '同机'), one('rail', 'Pod内'), one('out', '跨Pod')].filter(Boolean).join(' · ');
    }

    /* ── 工具栏 ── */
    // Lucide 风格线性图标（内联 SVG · stroke=currentColor · 无 emoji）
    const ICON = {
      play: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="6 3 20 12 6 21 6 3"/></svg>',
      pause: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16" rx="1"/><rect x="14" y="4" width="4" height="16" rx="1"/></svg>',
      grid: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M3 15h18M9 3v18M15 3v18"/></svg>',
      clock: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
      key: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="7.5" cy="15.5" r="4.5"/><path d="m10.7 12.3 8.3-8.3M16 7l2 2M13.5 9.5l2 2"/></svg>',
      alert: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4 3 20h18L12 4z"/><path d="M12 10v4M12 17.5v.5"/></svg>',
      sliders: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/></svg>',
      close: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 6l12 12M18 6L6 18"/></svg>',
      help: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M9.2 9.3a3 3 0 0 1 5.8 1c0 2-2.9 2.6-2.9 4"/><path d="M12 17.5v.5"/></svg>',
    };
    /* 每一排控件的「这是什么」——hover / 键盘聚焦弹出说明。
       写法像一小段文档：标题 → 一句定义 → 选项清单（<dl>）→ 注意事项，
       最后由 DYN 追加一块「此刻」状态。控件名只有两个字，说不清「注入和着色是什么
       关系」这类问题；又不该把长文案常驻在画面上，所以一律收进气泡。 */
    const HELP = {
      modes: `<h4>形态 · 同一批卡换一种堆法</h4>
        <p>只改变卡摆在哪（谁挨着谁），不改变卡本身、也不改变颜色。哪一维被堆成「一整块」，那一维的问题就现出形状。</p>
        <dl>
          <dt>PP</dt><dd>换轴：段沿水平摊成流水线 —— 找慢段 / 气泡</dd>
          <dt>TP</dt><dd>坐标系与标准相同，只把 TP 轴拉开成墙 —— 查同槽位系统性坏件</dd>
          <dt>EP</dt><dd>每个专家桶一面墙 —— 桶故障 = 整面墙同红</dd>
          <dt>DP</dt><dd>每个副本一块板、排成宫格 —— 找慢副本</dd>
          <dt>标准</dt><dd><b>前四个的基准投影，自己不查问题。</b>位置即多维坐标（X=PP 左→右 = Stage0→末 · Y=DP 上→下 · Z=TP 纵深），前 PP-DP 面是一张「哪一段 × 哪个副本」的表；三个密排 2D 平面归它所有；整网层联动也从这儿进</dd>
        </dl>
        <p class="prc-helpnote">前四个各冲着一个问题去，所以开门落在 PP；标准不回答问题、只做参照系，按钮排在最后也是这个意思——先挑一个形态查问题，要看全貌再回来。</p>`,
      views: `<h4>视角 · 怎么看这堆卡</h4>
        <dl>
          <dt>3D</dt><dd>可拖拽旋转的等距轴测</dd>
          <dt>顶 / 前 / 侧</dt><dd>正交锁轴的 2D 投影，会把与视线平行的维折叠</dd>
          <dt>剖面</dt><dd>只看被折叠那一维的某一层，其余压暗</dd>
        </dl>
        <p>不是每个形态都给满三屏：一个形态之所以不同于标准，全在它那根被拉开的轴（TP 切片的墙、PP 流水的段，步距 4×）。这根轴<b>被折进视线</b>的那一屏，空档随之消失，剩下的与标准的同名平面逐位相同——那一屏就是标准，不重复给出（两个形态都少了侧视）。轴还留在屏幕上的顶 / 前两屏，墙与墙之间是看得见的空档，「这是第几片 / 第几段」在那里才数得清。</p>
        <p>按钮上的「DP-TP 面」读作<b>平面</b>而不是乘法：破折号两侧是这一屏留下的两根屏幕轴（横 DP、纵 TP），没写出来的第三根就是被折进视线的那一维。这一格里真正相乘的只有 rank 总数 TP×PP×DP，视角本身不改变任何数量。</p>
        <dl>
          <dt>拖动</dt><dd>转视角。正交视角下一拖就从当前朝向无缝转回 3D</dd>
          <dt>Ctrl / ⌘ + 拖动</dt><dd>平移画布（中键拖同样）。<b>不会</b>把正交视角踢回 3D——挪到哪儿看还是那一屏</dd>
          <dt>滚轮</dt><dd>缩放。切形态或切视角会重新取景，平移量一并归零</dd>
        </dl>
        <p class="prc-helpnote">折叠不隐瞒：每格重叠多少张卡就写在上面这行「此刻」里。</p>`,
      obj: `<h4>对象 · 整网的一个东西被切成了什么样</h4>
        <p>魔方本来回答「谁和谁一组」。选一个<b>整网对象</b>（模型计算图上的一个模块 / 算子 / HCCL 算子），它多回答一件事：<b>这个对象是怎么被切开、落到哪些卡上的</b>。</p>
        <p>选中后<b>对象接管着色</b>（与注入同一个约定，两套颜色不能同时在）：承载它的卡按「持有第几片」着色，不承载的退成背景。于是——</p>
        <dl>
          <dt>路由专家</dt><dd>按 EP 切 → 切到 EP聚簇，每面墙就是一个专家桶</dd>
          <dt>QKV 投影</dt><dd>按 TP 沿注意力头切 → 切到 TP切片，每片墙是一个 head 分片</dd>
          <dt>词嵌入 / LM Head</dt><dd>只落在 PP 首 / 末段 → 切到 PP流水，只有那一段亮着</dd>
          <dt>Router</dt><dd>复制的：每张卡各一份完整的，没有「片」可言</dd>
        </dl>
        <p class="prc-helpnote"><b>切分的形状 = 这个对象被哪一维切</b> —— 这是「异常的形状 = 根因类别」那条读法的推广。每个对象都知道自己该去哪一屏，「去 XX」按钮直接飞过去。</p>
        <p>通信算子（HCCL）那一带按另一条规则：<b>它不属于单张卡，而属于一个通信组</b>，组内每张卡上各有一个实例。所以要先选一张卡，才有「哪个组」可高亮。</p>`,
      slice: `<h4>剖面 · 折掉的那一维翻到第几层</h4>
        <p>正交 2D 会把与视线平行的那一维折进屏幕，于是一格里叠着好几张卡。剖面就是<b>只看这一维的某一层</b>，其余压暗——它跟着视角走（换一屏，折掉的是另一维，剖面翻的也就换成那一维），所以排在「视角」后面而不是自成一档。</p>
        <dl>
          <dt>剖面</dt><dd>开 / 关。关掉 = 把 N 张叠在一起看</dd>
          <dt>滑杆</dt><dd>翻到第几层。拖它会自动开——抓住把手本身就是「我要逐层看」</dd>
        </dl>
        <p class="prc-helpnote">3D 没有被折叠的维，因此这一组不出现。多折叠维的形态（如 EP 聚簇的 3D）开了剖面仍可能剩下重叠，「此刻」那行会照实写。</p>`,
      lens: `<h4>着色 · 给卡上色的镜头</h4>
        <p>只改颜色，不改结构。</p>
        <dl>
          <dt>状态热力</dt><dd>当前通信阶段的负载，绿→黄→红，跟着时间轴走</dd>
          <dt>TP / PP / DP / EP</dt><dd>按该维的组号上色，同色即同组——用来肉眼验证「这种堆法下同组是不是真的连成一块」</dd>
          <dt>主机 / Pod</dt><dd>按物理落位上色，看 rail 亲和：同色连成块 = 这一组正好装在一台机 / 一个 Pod 里</dd>
          <dt>权重占比 / 激活占比 / 梯度占比 / 优化器态占比</dt><dd>这张卡在全网同一档字节数里排 min→max 的第几位，绿→黄→红，与状态热力共用同一条色带（换分子不换调色板）。<b>结构性</b>读数——同一 TP/PP/EP 坐标永远算出同一个数，不随时间轴变化；切一下并行度或形态，花纹会跟着重新排布。图例那两个 GB 数字是真实端点，颜色是这两个端点之间的相对位置，不是占卡内总内存的百分比</dd>
        </dl>
        <p class="prc-helpnote">右下角图例只列「颜色 + 名字」。组数超过色环时会 12 色循环，<b>同色不一定同组</b>，以「… 共 N 组」为准；图例里那条灰色是「与选中卡无关」的压暗卡，不是另一个组。内存构成四档给的是全网 min→max 的 GB 数，不是某张代表卡的读数。</p>`,
      anom: `<h4>注入 · 假装某一维出故障</h4>
        <p>看它在各形态下长什么形状。</p>
        <p><b>与着色的关系</b>：注入不是另一种镜头，而是<b>接管</b>着色——一旦注入非「无」，卡色改由故障决定（受影响的卡＝危险红，其余按低负载淡色），上面选的着色镜头暂时让位，图例也随之切换；选回「无」即恢复。</p>
        <p class="prc-helpnote">例：注入 EP桶3 → 标准形态下是一圈周期条带，切到 EP 聚簇就 snap 成一整面墙，这就是「热点桶」的形状。</p>`,
      wire: `<h4>连线 · 选中卡的四个通信域怎么收发</h4>
        <p>必须先选中一张卡（点画面里的小方块），否则没有对象可画。</p>
        <dl>
          <dt>成员</dt><dd>同域对端卡的线框描边</dd>
          <dt>通信线</dt><dd>按集合算法画的走线</dd>
          <dt>域轮廓</dt><dd>把该组整体框起来，看这组在当前堆法下是什么形状</dd>
          <dt>粒子</dt><dd>沿「此刻主导维」走线跑的暗点，方向即数据流向</dd>
          <dt>聚焦</dt><dd>把与选中卡无关的卡压暗、网格反过来加强</dd>
          <dt>算法</dt><dd>AllReduce 画成 Ring（前半 ReduceScatter / 后半 AllGather）还是 Tree；PP 恒为 P2P 链、EP 恒为 AllToAll</dd>
        </dl>
        <p class="prc-helpnote">图例「连线」段里<b>加粗的那条 = 此刻主导维</b>（走线加亮、粒子只跑它）。走线可点：悬停报这一段是谁到谁、跨的是同机 UB / Pod 内 rail / 跨 Pod；点一下选中该段并抛给宿主（onSelectEdge）去点亮物理链路。物理落位默认 8 卡/机 · 32 机/Pod（rank 连号装机，TP 组因此天然同机），要改走 setPlacement API。</p>`,
      time: `<h4>时间 · 一个训练 step 的 4 个通信阶段</h4>
        <dl>
          <dt>TP</dt><dd>前向 AllReduce —— 节点内 UB · 高频</dd>
          <dt>PP</dt><dd>阶段接力 Send/Recv —— Pod 内跨 Host · 中频</dd>
          <dt>EP</dt><dd>MoE AllToAll 浪涌 —— Pod 内全互联 · 浪涌</dd>
          <dt>DP</dt><dd>梯度 AllReduce —— 跨 Pod Scale-Out · 低频大包</dd>
        </dl>
        <p class="prc-helpnote">热力着色与方向粒子都跟着阶段走；轨道可拖拽定位，播放/暂停在顶栏。当前阶段也写在选中卡的详情面板里。</p>`,
      cfg: `<h4>并行 · 这套魔方由多少卡、怎么切</h4>
        <p>rank 总数 = TP×PP×DP。<b>EP 不乘进卡数</b>——它折在 DP 轴上（要求 EP 整除 DP），DP/EP = AllToAll 域的个数。</p>
        <p class="prc-helpnote">改完数字按 Apply 整体重建。两个预设：盘古 Pro MoE 真实训练策略（8·5·100·2 = 4000 卡）、128 卡小规格（2·4·16·8）。</p>`,
    };
    /* 问号气泡里的「当前状态」部分：形态的读法、注入的读法、连线的空态提示——
       这些原先常驻在画面上（左下 HUD、工具栏行尾的说明），太占地方，一律收进气泡。 */
    const DYN = {
      modes: () => {
        const m = model.modes[S.mode];
        // 当前形态「给你看什么」——一格之隔 vs 一堵墙之隔，各自对应哪类问题
        const DETAIL = {
          std: `三根语义轴各放一维（X=PP 左→右 Stage0→末 · Y=DP 上→下 · Z=TP 纵深），`
            + `一张卡的位置就是它的 (TP,PP,DP) 坐标；第 4 维 EP 靠着色透镜叠上去。`
            + `<b>前 PP-DP 面是一张现成的「哪一段 × 哪个副本」表</b>——两根有编号、要一段段读的`
            + `轴都在屏幕上；TP 收进纵深，因为一整根纵深行就是一个 TP 组，翻它用剖面。`
            + `这是「查身份」的形态，不是「找形状」的形态。`,
          dpt: `一块板 = 一份完整副本（板内 TP×${TP} 折成 ${model.TPC}列×${model.TPD}排、竖向是 PP×${PP}），`
            + `${REP} 块板排成宫格。副本之间只在步末做梯度 AllReduce → <b>慢副本 = 宫格里干净的一整块板发暗</b>。`,
          ep: `一面墙 = 一个专家桶（同墙 = 持有同一批专家），每桶复现于 ${DOM} 个 A2A 域。`
            + `<b>热点/坏桶 = 一整面墙同色</b>；横穿所有墙的同一排 = 一个 A2A 域（每桶各出 1 员互发）。桶↔卡非 1:1。`,
          tps: `坐标系与标准完全相同，只把 TP 轴的间距拉到「强调」档：<b>一面墙 = 全网 TP 槽位相同的卡</b>`
            + `（横跨所有 PP 段与所有 DP 副本，共 ${PP * REP} 张）。同槽位的系统性问题——固件/驱动版本不一致、`
            + `某槽位风道差、某条 rail 上的第 k 张卡——在标准形态里是每隔 ${TP} 张出现一次的周期细条纹，`
            + `在这里是<b>一整面墙同红</b>（注入「TP槽0」可当场对照）。选中一张卡时，它的 TP AllReduce 组`
            + `= 每面墙各一张，连线横穿全部 ${TP} 面墙——这就是「每次层内 AllReduce 要跨过多少堵墙」。`,
          ppf: `这个形态<b>换了轴</b>：X=PP（段，左→右就是前向数据流）· Y=TP · Z=DP。`
            + `<b>一面墙 = 一个流水段的所有卡</b>（连续 ${LPS} 层 × 所有 TP × 所有 DP）。`
            + `慢段 = 一整面墙偏暗，而且<b>下游的墙跟着暗</b>（等上游喂数据 = 气泡）；`
            + `PP 轴上的 PP0·L0-${LPS - 1} 标尺回答「这段管哪些层」。选中卡的 PP 接力组在这里是一条`
            + `从左到右穿过所有墙的链——正是流水线本身的形状。`,
        }[m.key] || '';
        return `<div class="prc-helpnow"><b>此刻：${esc(m.sub)}</b>` + (DETAIL ? `<p>${DETAIL}</p>` : '') +
          `<p>为什么这样摆：${esc(m.why)}</p>` +
          (axNotes.length ? '<ul>' + axNotes.map((t) => `<li>${esc(t)}</li>`).join('') + '</ul>' : '') +
          (S.selLayer != null && S.mode === 0 ? `<p>正高亮整网 L${S.selLayer} 切片</p>` : '') + '</div>';
      },
      views: () => {
        const md = model.modes[S.mode], d = curDepth();
        const bv = md.bestView, bvName = bv === 0 ? '3D' : (md.viewLabels[bv] || '');
        const bvWhy = { std: '三根轴地位相同，位置即坐标——3D 才是它本身',
          dpt: '找慢副本要扫一遍整个宫格，顶视一屏看全所有板',
          ep: '桶故障（整墙同色）与域热点（横穿一排）是两个正交方向，只有这一屏同屏放得下',
          tps: '同槽位系统性故障要在全集群范围内看，DP 轴必须在场——一整列红 = 那个槽位整批坏',
          ppf: '「哪一段慢」里段身份就是答案、DP 是噪声，5 列一字排开，慢段 = 整列偏暗' }[md.key];
        const what = S.view === 0
          ? '3D 等距轴测 · 三根轴同屏，拖动可转'
          : `${md.viewLabels[S.view]} —— 屏幕两根轴之外，${d ? esc(d.label) : '第三根轴'} 被折进视线`;
        return `<div class="prc-helpnow"><b>此刻：</b>${esc(md.name)} · ${what}`
          + (d ? `，${granText()}` : '')
          + (bv != null ? `<br><b>这个形态最该看：${esc(bvName)}</b>——${esc(bvWhy || '')}` : '')
          + '</div>';
      },
      slice: () => {
        const d = curDepth();
        if (!d) return '';
        return `<div class="prc-helpnow"><b>此刻：</b>这一屏折掉的是 ${esc(d.label)}（共 ${d.fold} 层）`
          + `，${esc(granText())}</div>`;
      },
      anom: () => {
        const note = {
          none: '', 
          tp: '当前注入 TP 槽 0：全网同槽位卡集体标红 → 切「TP切片」= 一面墙集体异常（同槽位系统性坏件的形状）',
          pp: '当前注入 PP 级 0：物理上散成条纹 → 切「PP流水」= 最左一整段全红（慢段/坏段的形状）',
          dp: '当前注入 DP 副本 0：切「DP平铺」= 宫格里干净的一块板全红（慢副本的形状）',
          ep: `当前注入 EP 桶 ${anomBucket()}：标准形态下是周期条带 → 切「EP聚簇」= 一整面墙同红（热点/坏桶的形状 · 桶↔卡非 1:1）`,
        }[S.anom];
        return note ? `<div class="prc-helpnow"><b>${esc(note)}</b></div>` : '';
      },
      wire: () => (S.sel == null
        ? '<div class="prc-helpnow"><b>现在还没选卡 → 没有连线可画</b>：点画面里任意一个小方块（点上面任一图层按钮也会自动替你选一张）。</div>'
        : `<div class="prc-helpnow"><b>此刻：</b>TP/DP=AllReduce(${S.algo === 'tree' ? 'Tree' : 'Ring'}) · PP=P2P 链 · EP=AllToAll` +
          (S.wire.focus ? ' · 聚焦开：无关卡已压暗' : '') + '</div>'),
    };
    const helpBubbles = {};
    function syncHelp() {
      Object.keys(DYN).forEach((k) => {
        const b = helpBubbles[k];
        if (b) b.innerHTML = (HELP[k] || '') + DYN[k]();
      });
    }
    function helpDot(key) {
      const s = document.createElement('span');
      s.className = 'prc-help';
      s.tabIndex = 0;
      s.setAttribute('aria-label', '说明');
      s.innerHTML = ICON.help;
      // 气泡贴边翻转：问号靠右时改成右对齐、靠下时改成向上弹，免得跑到屏幕外
      const place = () => {
        const rr = root.getBoundingClientRect(), dr = s.getBoundingClientRect();
        s.classList.toggle('is-right', dr.left - rr.left > rr.width * 0.5);
        s.classList.toggle('is-up', dr.bottom - rr.top > rr.height * 0.6);
      };
      s.addEventListener('pointerenter', place);
      s.addEventListener('focus', place);
      const bub = document.createElement('span');
      bub.className = 'prc-helptip';
      bub.innerHTML = (HELP[key] || '') + (DYN[key] ? DYN[key]() : '');
      helpBubbles[key] = bub;
      s.appendChild(bub);
      return s;
    }
    // 设计系统按钮：secondary(.btn) + 小号(.btn-sm)，选中态用 .is-selected
    function chipBtn(label, onClick) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'btn btn-sm';
      b.textContent = label;
      b.addEventListener('click', onClick);
      return b;
    }
    let modeBtns = [], viewBtns = [], lensBtns = [], anomBtns = [], playBtn = null, sliceBox = null, sliceRange = null, sliceLab = null;
    let objSel = null, objGo = null;
    let cfgInputs = null, cfgRead = null, cfgErr = null;
    let wireBtns = [], algoBtns = [];
    let timeTrack = null, timeHead = null, moreBtn = null;
    function syncTimeUI() {
      if (!timeTrack) return;
      const pi = phaseIdx();
      timeTrack.querySelectorAll('.prc-pseg').forEach((el, i) => el.classList.toggle('on', i === pi));
      timeHead.style.left = (S.t * 100) + '%';
      // 阶段名/总线不再挂在时间行：文案长度随阶段变化，会把整排挤成两行、上下跳动。
      // 常驻位置交给左下 HUD 的「此刻」一行，解释交给行首问号，轨道段自己带 title。
    }
    // 「并行」输入排：TP/PP/DP/EP 任意填数 → setConfig 整体重建魔方（回车或「应用」提交）
    function applyCfg() {
      if (!cfgInputs) return;
      const res = api.setConfig({ tp: +cfgInputs.tp.value, cp: +cfgInputs.cp.value, pp: +cfgInputs.pp.value, dp: +cfgInputs.dp.value, ep: +cfgInputs.ep.value });
      if (!res.ok && cfgErr) cfgErr.textContent = '✗ ' + res.error;
    }
    // 顶栏会随宽度换行变高 → 把实际高度写回 CSS 变量，抽屉/贴士/详情卡都据此让位
    function syncBarH() {
      const bar = root.querySelector('.prc-topbar');
      if (bar) root.style.setProperty('--prc-barh', Math.round(bar.getBoundingClientRect().height) + 'px');
      const more = root.querySelector('.prc-more');
      const open = root.classList.contains('is-open');
      root.style.setProperty('--prc-moreh', (open && more ? Math.round(more.getBoundingClientRect().height) : 0) + 'px');
    }
    function syncCfgUI() {
      if (!cfgInputs) return;
      cfgInputs.tp.value = TP; cfgInputs.cp.value = CP; cfgInputs.pp.value = PP; cfgInputs.dp.value = REP; cfgInputs.ep.value = EP;
      // 读数只补输入框没说的部分（乘法与折叠结果），别把已经摆在旁边的四个数再抄一遍
      cfgRead.textContent = `= ${N} rank${CP > 1 ? ` · CP${CP} 与 TP 共轴` : ''} · EP 折入 DP → ${DOM} 域`;
      cfgErr.textContent = '';
    }
    function syncChrome() {
      syncBrand(); syncHelp(); syncBarH();                              // 招牌名字/问号气泡/标题规格随状态更新
      if (anomBtns[4]) anomBtns[4].textContent = `EP桶${anomBucket()}`;   // 示意桶号随 EP 收缩
      /* 形态按钮的**显示顺序**：PP · TP · EP · DP · 标准。
         前四个按并行维排（与设计系统 sidecar 的并行切换同一顺序，读者的肌肉记忆一致），
         「标准」排到最后——它不是第五种问题导向的形态，而是那四种的基准投影：
         三根轴地位相同、三个密排平面的归属地。摆在末位，「先挑一个问题，实在要看全貌
         再回到标准」这层关系不用读文案就看得出来。
         用 CSS order 调换，DOM 顺序与形态序号保持一致，免得 URL 参数 / 键盘序 / 测试跟着错位。 */
      const MODE_ORDER = ['ppf', 'tps', 'ep', 'dpt', 'std'];
      modeBtns.forEach((b, i) => {
        b.classList.toggle('is-selected', i === S.mode);
        const k = MODE_ORDER.indexOf(model.modes[i].key);
        b.style.order = k < 0 ? MODE_ORDER.length : k;
      });
      const md = model.modes[S.mode];
      const vlist = md.views || [0, 1, 2, 3];
      viewBtns.forEach((b, i) => {
        b.style.display = vlist.includes(i) ? '' : 'none';        // 视角收编：重合平面不出按钮
        b.classList.toggle('is-selected', i === S.view);
        if (i > 0) b.textContent = md.viewLabels[i];
        /* 主屏（bestView）排到 3D 后面的第一个：一排按钮里位置本身就是推荐——
           顺序按「顶/前/侧」的固有次序排，读者只会理解成方位表，不会理解成
           「这个形态该先看哪一屏」。用 CSS order 调换，DOM 顺序与视角序号保持一致，
           免得别处（键盘序、URL 参数、测试）跟着错位。 */
        b.style.order = i === 0 ? 0 : i === md.bestView ? 1 : i + 1;
      });


      // 只剩 3D 一个视角的形态（TP切片 / PP流水，2D 与标准重合）：整组不显示，
      // 一个孤零零的「3D」按钮既没有可切换的对象，也让人以为别的被禁用了
      const vg = root.querySelector('.prc-row-views');
      if (vg) vg.style.display = vlist.length > 1 ? '' : 'none';
      const vhelp = vg && vg.nextElementSibling && vg.nextElementSibling.classList.contains('prc-help') ? vg.nextElementSibling : null;
      if (vhelp) vhelp.style.display = vlist.length > 1 ? '' : 'none';
      const lensKeys = ['load', 'tp', 'pp', 'dp', 'ep', 'host', 'pod', 'w', 'act', 'grad', 'opt', 'neutral'];
      lensBtns.forEach((b, i) => b.classList.toggle('is-selected', lensKeys[i] === S.colorBy));
      // 素色镜头选中时，顶栏那圈"选中态=主色蓝"的胶囊按钮也一并退成中性灰——
      // 不然卡阵已经素净了，顶栏还留一圈蓝，"素色"这句话只说了一半。
      root.classList.toggle('is-neutral-theme', S.colorBy === 'neutral');
      if (objSel) objSel.value = S.obj || '';
      if (objGo) {
        const o = S.obj && model.netObjBy[S.obj];
        // 没选对象、或这个对象没有「该去哪一屏」的答案（如复制的 Router）→ 按钮不出现，
        // 留一个按下去没反应的按钮比不留更费解（同剖面那一排的处理）。
        const on = !!(o && o.best);
        objGo.style.display = on ? '' : 'none';
        if (on) {
          const m = model.modes.find((mm) => mm.key === o.best);
          objGo.textContent = m ? `去 ${m.short || m.name}` : '去主屏';
          objGo.classList.toggle('is-selected', model.modes[S.mode].key === o.best);
        }
      }
      // 对象透镜接管着色时，把「着色」那一排整体压暗并说明——两套颜色不能同时在
      const rowLensEl = $('.prc-row-lens');
      if (rowLensEl) rowLensEl.classList.toggle('is-superseded', objOn());
      const anomKeys = ['none', 'tp', 'pp', 'dp', 'ep'];
      anomBtns.forEach((b, i) => b.classList.toggle('is-selected', anomKeys[i] === S.anom));
      const wireKeys = ['members', 'lines', 'outline', 'movers', 'focus'];
      wireBtns.forEach((b, i) => b.classList.toggle('is-selected', !!S.wire[wireKeys[i]]));
      const algoKeys = ['auto', 'ring', 'tree'];
      algoBtns.forEach((b, i) => b.classList.toggle('is-selected', algoKeys[i] === S.algo));

      if (moreBtn) {
        // 只留图标：顶栏右侧那两个按钮不带文字（标题交给 title/aria-label）
        moreBtn.innerHTML = ICON.sliders;
        moreBtn.title = moreBtn.ariaLabel = S.more ? '收起设置' : '更多设置';
        moreBtn.setAttribute('aria-label', moreBtn.title);
        moreBtn.classList.toggle('is-selected', S.more);
      }
      if (playBtn) {
        playBtn.innerHTML = S.playing ? ICON.pause : ICON.play;
        playBtn.title = S.playing ? `暂停（此刻 ${PHASES[phaseIdx()].id}）` : `播放（此刻 ${PHASES[phaseIdx()].id}）`;
        playBtn.setAttribute('aria-label', playBtn.title);
        playBtn.classList.toggle('is-selected', S.playing);
      }
      if (sliceBox) {
        const d = curDepth();
        // 没有折叠维（3D）就整组连同它的问号一起收起——留一个翻不动的滑杆比不留更费解
        const shown = !!d;
        sliceBox.style.display = shown ? '' : 'none';
        const sh = sliceBox.nextElementSibling;
        if (sh && sh.classList.contains('prc-help')) sh.style.display = shown ? '' : 'none';
        if (d) {
          sliceRange.max = String(d.slice.n - 1);
          if (S.sliceVal > d.slice.n - 1) S.sliceVal = 0;
          sliceRange.value = String(S.sliceVal);
          // 这一排只放最短的读数，「折了几张、正翻哪一层」的完整说法在它自己的问号气泡里
          sliceLab.textContent = S.sliceOn ? `${d.slice.lab}=${S.sliceVal}` : `关（${d.label} 折叠）`;
          sliceBox.querySelector('.btn').classList.toggle('is-selected', S.sliceOn);
        }
      }
    }
    if (opts.chrome !== false) {
      // 每排行首「名称 + 问号」：问号 hover/聚焦弹出这一排是什么、和别的排什么关系
      [['modes', '.prc-row-modes'], ['views', '.prc-row-views'], ['slice', '.prc-row-slice'], ['obj', '.prc-row-obj'],
        ['lens', '.prc-row-lens'],
        ['anom', '.prc-row-anom'], ['wire', '.prc-row-wire'], ['time', '.prc-timepop'], ['cfg', '.prc-row-cfg']]
        .forEach(([k, sel]) => {
          const lab = $(sel + ' .prc-lab');
          if (lab) { lab.insertAdjacentElement('afterend', helpDot(k)); return; }
          const row = $(sel);                       // 顶栏里的形态/视角没有行首标签 → 问号跟在这一组后面
          if (row) row.insertAdjacentElement('afterend', helpDot(k));
        });
      const rowModes = $('.prc-row-modes'), rowViews = $('.prc-row-views'), rowLens = $('.prc-row-lens'), rowAnom = $('.prc-row-anom');
      modeBtns = model.modes.map((m, i) => {
        const b = rowModes.appendChild(chipBtn(m.short || m.name, () => api.setMode(i)));
        b.title = m.name; b.setAttribute('aria-label', m.name);   // 短名按钮，全名进 title
        return b;
      });
      viewBtns = ['3D', '顶', '前', '侧'].map((t, i) => rowViews.appendChild(chipBtn(t, () => api.setView(i))));
      // 抽屉开关（着色 / 注入 / 连线 / 时间 / 并行）——独立按钮，不挤在视角行里
      moreBtn = $('.prc-morebtn');
      moreBtn.addEventListener('click', () => {
        S.more = !S.more;
        root.classList.toggle('is-open', S.more);
        syncChrome(); resize();
      });
      sliceBox = $('.prc-row-slice');
      sliceBox.appendChild(chipBtn('剖面', () => { S.sliceOn = !S.sliceOn; refresh2D(); }));
      sliceRange = document.createElement('input'); sliceRange.type = 'range'; sliceRange.min = '0'; sliceRange.max = '1'; sliceRange.value = '0';
      /* 滑杆**不置灰**：置灰的滑杆读作「这功能不可用」，而不是「先按一下旁边那个按钮」——
         剖面因此看着像坏的。改成拖它就自动开：抓住把手本身就是「我要逐层看」的意思，
         按钮留着做显式开关（也用来关）。 */
      sliceRange.addEventListener('input', () => {
        S.sliceVal = sliceRange.value | 0;
        if (!S.sliceOn) S.sliceOn = true;
        refresh2D();
      });
      sliceLab = document.createElement('span'); sliceLab.className = 'prc-mono prc-scread';
      sliceBox.appendChild(sliceRange); sliceBox.appendChild(sliceLab);
      /* ── 整网对象那一排 ──
         对象有十几个、还分带，用互斥胶囊摆不下 → 走 <select>（按 band 分组）。
         旁边一个「去主屏」按钮：每个对象都知道自己在哪个形态下 snap 成规整的块/墙，
         按一下就飞过去——这是本次深化最要紧的动作，不该让人自己去猜该切哪个形态。 */
      const rowObj = $('.prc-row-obj');
      objSel = document.createElement('select');
      objSel.className = 'prc-objsel';
      objSel.setAttribute('aria-label', '选择整网对象');
      const optNone = document.createElement('option');
      optNone.value = ''; optNone.textContent = '— 不选（看卡本身）—';
      objSel.appendChild(optNone);
      model.netObjBands.forEach((band) => {
        const list = model.netObjects.filter((o) => o.band === band);
        if (!list.length) return;
        const g = document.createElement('optgroup'); g.label = band;
        list.forEach((o) => {
          const op = document.createElement('option');
          op.value = o.id; op.textContent = o.name; op.title = o.note || '';
          g.appendChild(op);
        });
        objSel.appendChild(g);
      });
      objSel.addEventListener('change', () => { api.selectObject(objSel.value || null); });
      rowObj.appendChild(objSel);
      objGo = chipBtn('去主屏', () => {
        const o = S.obj && model.netObjBy[S.obj];
        if (!o || !o.best) return;
        const i = model.modes.findIndex((m) => m.key === o.best);
        if (i >= 0) api.setMode(i);
      });
      objGo.title = '切到这个对象的分片会 snap 成整块/整墙的那个形态';
      rowObj.appendChild(objGo);

      const lensSeg = rowLens.appendChild(Object.assign(document.createElement('span'), { className: 'segmented-control' }));
      lensBtns = [['状态热力', 'load'], ['TP', 'tp'], ['PP', 'pp'], ['DP', 'dp'], ['EP', 'ep'],
        ['主机', 'host'], ['Pod', 'pod'],
        // 内存构成：权重/激活是用户明确点名「现在比较少、要看清楚」的两档，排在前面；
        // 梯度/优化器态是同一套机制顺手加的对照档（见 memBytesOfRank 顶注），排在后面。
        ['权重占比', 'w'], ['激活占比', 'act'], ['梯度占比', 'grad'], ['优化器态占比', 'opt'],
        // 素色：不按任何维度/负载上色，先看阵列形状、不看颜色——排最后，默认仍是「状态热力」。
        ['素色', 'neutral']]
        .map(([t, k]) => lensSeg.appendChild(chipBtn(t, () => { S.colorBy = k; recolor(); renderLegend(); syncChrome(); })));
      /* 时间轴 = 一个 step 的 4 个通信阶段（对齐集群驾驶舱）。
         播放/暂停常驻顶栏（只有图标），阶段轨道悬停时才弹出——它不是常用控件，
         但要随手够得着；弹层里可以直接拖拽定位，拖拽期间不收起。 */
      const rowTime = $('.prc-timepop');
      playBtn = $('.prc-playbtn');
      playBtn.addEventListener('click', () => { S.playing = !S.playing; syncChrome(); });
      timeTrack = document.createElement('div'); timeTrack.className = 'prc-phasetrack';
      PHASES.forEach((ph, i) => {
        const seg = document.createElement('div');
        seg.className = `prc-pseg prc-ph-${ph.id}`;
        seg.textContent = ph.id;
        seg.title = `${ph.name} · ${ph.bus}`;
        seg.dataset.ph = String(i);
        timeTrack.appendChild(seg);
      });
      timeHead = document.createElement('div'); timeHead.className = 'prc-playhead';
      timeTrack.appendChild(timeHead);
      const scrub = (ev) => {
        const r = timeTrack.getBoundingClientRect();
        S.t = Math.min(0.999, Math.max(0, (ev.clientX - r.left) / r.width));
        if (S.colorBy === 'load' && S.anom === 'none') recolor();
        rebuildComm(); syncTimeUI(); renderHud();
      };
      const timeWrap = $('.prc-timewrap');
      timeTrack.addEventListener('pointerdown', (ev) => {
        timeTrack.setPointerCapture(ev.pointerId); timeWrap.classList.add('is-dragging'); scrub(ev);
      });
      timeTrack.addEventListener('pointermove', (ev) => { if (ev.buttons & 1) scrub(ev); });
      global.addEventListener('pointerup', () => timeWrap.classList.remove('is-dragging'));
      rowTime.appendChild(timeTrack);
      // 连线图层：五个独立开关（都可关）+ 集合算法选择
      const rowWire = $('.prc-row-wire');
      const wireSeg = rowWire.appendChild(Object.assign(document.createElement('span'), { className: 'prc-chips' }));
      wireBtns = [['成员', 'members'], ['通信线', 'lines'], ['域轮廓', 'outline'], ['粒子', 'movers'], ['聚焦', 'focus'], ['切分', 'payload'], ['嵌套', 'nest'], ['装载板', 'board']]
        .map(([t, k]) => wireSeg.appendChild(chipBtn(t, () => {
          S.wire[k] = !S.wire[k];
          if (k === 'movers' && !S.wire.movers) moverMeshes.forEach((m) => { m.visible = false; });
          // 连线只在「选中一张卡」之后才有东西可画：开图层时若还没选卡，先替用户选一张
          // 居中的代表卡（否则按钮亮着、画面却毫无变化，看上去像开关坏了）。
          if (S.sel == null && (S.wire.members || S.wire.lines || S.wire.outline || S.wire.movers)) {
            api.select(model.rankOf(TP >> 1, PP >> 1, REP >> 1));
          }
          else { rebuildComm(); refreshFocus(); renderInfo(); syncChrome(); }
        })));
      rowWire.appendChild(Object.assign(document.createElement('span'), { className: 'prc-lab prc-lab-inline', textContent: '算法' }));
      const algoSeg = rowWire.appendChild(Object.assign(document.createElement('span'), { className: 'segmented-control' }));
      algoBtns = [['自动', 'auto'], ['Ring', 'ring'], ['Tree', 'tree']]
        .map(([t, k]) => algoSeg.appendChild(chipBtn(t, () => { S.algo = k; rebuildComm(); renderLegend(); syncChrome(); })));

      const anomSeg = rowAnom.appendChild(Object.assign(document.createElement('span'), { className: 'segmented-control' }));
      anomBtns = [['无', 'none'], ['TP槽0', 'tp'], ['PP级0', 'pp'], ['DP副本0', 'dp'], ['EP桶3', 'ep']]
        .map(([t, k]) => anomSeg.appendChild(chipBtn(t, () => { S.anom = k; recolor(); renderHud(); renderLegend(); syncChrome(); })));
      const rowCfg = $('.prc-row-cfg');
      const mkDim = (lab) => {
        const wrap = document.createElement('span'); wrap.className = 'prc-cfgitem';
        const l = document.createElement('span'); l.textContent = lab; wrap.appendChild(l);
        const inp = document.createElement('input');
        inp.type = 'number'; inp.min = '1'; inp.step = '1';
        inp.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') applyCfg(); });
        wrap.appendChild(inp); rowCfg.appendChild(wrap);
        return inp;
      };
      cfgInputs = { tp: mkDim('TP'), cp: mkDim('CP'), pp: mkDim('PP'), dp: mkDim('DP'), ep: mkDim('EP') };
      { const b = chipBtn('Apply', applyCfg); b.classList.add('btn-solid'); rowCfg.appendChild(b); }
      cfgErr = document.createElement('span'); cfgErr.className = 'prc-cfgerr'; rowCfg.appendChild(cfgErr);
      const cfgFoot = document.createElement('div'); cfgFoot.className = 'prc-rowfoot'; rowCfg.appendChild(cfgFoot);
      cfgRead = document.createElement('span'); cfgRead.className = 'prc-mono'; cfgFoot.appendChild(cfgRead);
      // 快捷预设（标签按 TP·PP·DP·EP 顺序）：
      //  · 盘古 Pro MoE 真实训练策略（data/ascend-workload-pangu-moe.json，
      //    TP8·EP2·PP5·4K NPU → dp = 4000/(8×5) = 100，EP2 折入其中）；
      //  · 128 卡小规格（单超节点量级）：tp2×pp4×dp16 = 128，EP8 折入 DP → 2 个 A2A 域。
      cfgFoot.appendChild(chipBtn('盘古ProMoE 8·5·100·2', () => api.setConfig({ tp: 8, pp: 5, dp: 100, ep: 2 })));
      cfgFoot.appendChild(chipBtn('128卡 2·4·16·8', () => api.setConfig({ tp: 2, pp: 4, dp: 16, ep: 8 })));
    }
    function refresh2D() { reScale(); recolor(); syncHelp(); renderLegend(); syncChrome(); }

    /* ── 交互：悬停 tooltip / 点选 / 拖拽旋转（任何视角，正交下转回 3D）/ 滚轮缩放 ── */
    const ray = new THREE.Raycaster(), mouse = new THREE.Vector2();
    function aimAt(ev) {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
      ray.setFromCamera(mouse, camera);
    }
    function pick(ev) {
      aimAt(ev);
      const hit = ray.intersectObject(chips)[0];
      return hit && hit.instanceId != null ? hit.instanceId : null;
    }
    /* 连线也可点选（C 档挂点）：命中哪根管 → 再按命中点就近判定是这根折线的哪一段，
       于是拿到「谁 → 谁、哪一维、哪种原语、跨了哪层物理链路」。宿主可据此去点亮
       自己那套物理路径（本 pattern 不搬运物理拓扑几何）。 */
    const _p1 = new THREE.Vector3(), _p2 = new THREE.Vector3(), _ab = new THREE.Vector3(), _ap = new THREE.Vector3();
    function segDist(a, b, p) {
      _ab.copy(b).sub(a); _ap.copy(p).sub(a);
      const l2 = _ab.lengthSq();
      const t = l2 ? Math.max(0, Math.min(1, _ap.dot(_ab) / l2)) : 0;
      return _p2.copy(a).addScaledVector(_ab, t).distanceTo(p);
    }
    function pickEdge(ev) {
      if (!S.wire.lines || S.sel == null) return null;
      aimAt(ev);
      const hits = ray.intersectObjects(commGroupG.children, false);
      const hit = hits.find((h) => h.object.userData.edge);
      if (!hit) return null;
      const meta = hit.object.userData.edge, rs = meta.ranks;
      const gp = (r) => _p1.set(cur[r * 3], cur[r * 3 + 1], cur[r * 3 + 2]).clone();
      let bi = 1, bd = Infinity;
      for (let i = 1; i < rs.length; i++) {
        const d = segDist(gp(rs[i - 1]), gp(rs[i]), hit.point);
        if (d < bd) { bd = d; bi = i; }
      }
      const a = rs[bi - 1], b = rs[bi];
      return {
        dim: meta.dim, from: a, to: b,
        prim: primOf(meta.dim), algo: algoOf(meta.dim),
        tier: model.tierOf(a, b),
        hosts: [model.hostOf(a), model.hostOf(b)],
        pods: [model.podOf(a), model.podOf(b)],
        distance: hit.distance,
      };
    }
    /* 点某一片 → 持有该片的 rank。「切换片」不另做一排按钮：片本身就摆在那儿，
       点它就是「我要看这一片」，而看这一片 = 去到持有它的那张卡（其余坐标不动，
       于是形态、对象、视角全都接着看，只有这一维的坐标换了）。 */
    function pickShard(ev) {
      if (!shardGroup.visible || !shardPicks.length || S.sel == null) return null;
      aimAt(ev);
      const hit = ray.intersectObjects(shardPicks, false)[0];
      const d = hit && hit.object.userData.shardPick;
      if (!d) return null;
      const r = S.sel, tp = model.tpOf(r), pp = model.ppOf(r), rep = model.repOf(r);
      const cp = model.cpOf(r);
      if (d.dim === 'TP') return model.rankOf(d.i % TP, pp, rep, cp);
      if (d.dim === 'CP') return model.rankOf(tp, pp, rep, d.i % CP);
      if (d.dim === 'PP') return model.rankOf(tp, d.i % PP, rep, cp);
      if (d.dim === 'DP') return model.rankOf(tp, pp, d.i % REP, cp);
      if (d.dim === 'EP') return model.rankOf(tp, pp, model.domOf(r) * EP + (d.i % EP), cp);
      return null;
    }
    function pickRegion(ev) {
      if (!pickGroup.children.length) return null;
      aimAt(ev);
      const hit = ray.intersectObjects(pickGroup.children, false)[0];
      return hit ? hit.object.userData.region : null;
    }
    const TIER_LAB = { ub: '同机 UB', rail: 'Pod 内跨机 rail', out: '跨 Pod Scale-Out' };
    /* 拖拽有两种：**空手拖 = 转**（任何视角都转回 3D 轴测）、**按住 Ctrl / ⌘ 拖 = 平移**。
       平移复用相机本就有的 pan 向量（原本只用于让开左上角工具栏）——位置与视线目标偏移
       同一个向量，视线方向不变，正交投影不会被斜切。
       两点刻意的差别：
         · 平移**不把正交视角踢回 3D**。在顶视里挪一下看别的区域是很自然的动作，
           若跟着转成轴测，用户会觉得「碰一下就散架」；转是转、挪是挪，两件事。
         · 模式在 pointerdown 就锁定，拖到一半按下 Ctrl 不会中途改变行为（免得手一抖
           从转变成挪）。中键拖同样是平移（很多人手上的肌肉记忆）。 */
    let drag = null;
    const panMode = (ev) => ev.ctrlKey || ev.metaKey || ev.button === 1;
    renderer.domElement.addEventListener('pointerdown', (ev) => {
      drag = { x: ev.clientX, y: ev.clientY, moved: false, pan: panMode(ev) };
      if (drag.pan) ev.preventDefault();
    });
    // Ctrl+拖在部分平台会被当成右键 → 拖动期间吞掉右键菜单
    renderer.domElement.addEventListener('contextmenu', (ev) => { if (drag && drag.pan) ev.preventDefault(); });
    global.addEventListener('pointerup', () => { drag = null; });
    renderer.domElement.addEventListener('pointermove', (ev) => {
      if (drag && drag.pan && (ev.buttons & 5)) {          // 左键或中键
        const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
        if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
        // 内容跟手：往右拖 → 模型右移 → 相机左移（panX 减），往下拖同理（panY 增）
        const wpp = worldPerPx();
        cam.panX -= dx * wpp; cam.panY += dy * wpp;
        drag.x = ev.clientX; drag.y = ev.clientY;
        tipEl.style.display = 'none';
        return;
      }
      if (drag && (ev.buttons & 1)) {
        const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
        if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
        // 任何视角拖动都是旋转：正交 顶/前/侧 下拖动 → 从当前朝向无缝转回 3D（轴测态）
        if (S.view !== 0 && drag.moved) {
          if (S.view === 1) cam.phi = 1.35;                                    // 顶 → 近俯视起步
          else { cam.phi = 0.14; cam.theta = S.view === 2 ? Math.PI / 2 : 0; } // 前/侧 → 对应方位起步
          S.view = 0;
          applyAxVisibility(); refresh2D(); syncHelp();
        }
        cam.theta += dx * 0.006;
        cam.phi = Math.max(0.08, Math.min(1.45, cam.phi + dy * 0.005));
        drag.x = ev.clientX; drag.y = ev.clientY;
        return;
      }
      const r = pick(ev);
      const e = r == null ? pickEdge(ev) : null;      // 卡优先；没命中卡再看连线
      const g = r == null && !e ? pickRegion(ev) : null;   // 都没中 → 落在哪一段上
      S.hover = r;
      const showTip = (html) => {
        const rc = root.getBoundingClientRect();
        tipEl.style.display = 'block';
        tipEl.style.left = (ev.clientX - rc.left + 14) + 'px';
        tipEl.style.top = (ev.clientY - rc.top + 12) + 'px';
        tipEl.innerHTML = html;
      };
      if (r != null) {
        const st = model.ppOf(r), lr = model.stageLayerRange(st);
        showTip(`rank ${r} · TP${model.tpOf(r)} PP${st}(L${lr.lo}-${lr.hi}) DP${model.repOf(r)} · 桶${model.epOf(r)} 域${model.domOf(r)}`);
      } else if (e) {
        showTip(`${e.dim} ${e.prim} · rank ${e.from} → ${e.to} · <span style="color:${tierc(e.tier)}">${TIER_LAB[e.tier]}</span>`);
      } else if (g) {
        // 分段的悬停：与规格牌一模一样的两行，牌被抽掉的那几块靠这里问出来
        showTip(`<b style="color:${esc(g.color)}">${esc(g.title)}</b><br>${esc(g.sub)}`);
      } else tipEl.style.display = 'none';
    });
    renderer.domElement.addEventListener('pointerleave', () => { S.hover = null; tipEl.style.display = 'none'; });
    // 光标提示：按住 Ctrl / ⌘ 就变成「可抓」，不用先拖一下才知道有这个功能
    const syncPanCursor = (on) => stageEl.classList.toggle('is-pan', !!on);
    global.addEventListener('keydown', (ev) => { if (ev.key === 'Control' || ev.key === 'Meta') syncPanCursor(true); });
    global.addEventListener('keyup', (ev) => { if (ev.key === 'Control' || ev.key === 'Meta') syncPanCursor(false); });
    global.addEventListener('blur', () => syncPanCursor(false));
    renderer.domElement.addEventListener('click', (ev) => {
      if (drag && (drag.moved || drag.pan)) return;
      /* 切分片优先于卡：点某一片 = 「切到这一片去看」——跳到持有该片的 rank。
         片就摞在选中卡原地，不抢先判的话点上去命中的永远是它盖住的那张卡。 */
      const spk = pickShard(ev);
      if (spk) { api.select(spk); return; }
      const r = pick(ev);
      if (r == null) {
        const e = pickEdge(ev);
        if (e) { api.selectEdge(e); return; }      // 点在连线上 → 只报边，不动选中的卡
      }
      /* 再点一次已经选中的那张卡 = 下钻，不是「重新选一遍」（跟并行拓扑矩阵
         自己那条「再点一次选中的卡 = 往里走一档」的手势同一个读法——两边
         应该是同一句话）。只在 opts.onDrill 存在时才拦：没传这个回调的宿主
         （独立打开 /rubik-pattern.html、或者只想要普通选中语义的嵌入方）
         不受影响，行为跟改动前逐字节相同。 */
      if (r != null && r === S.sel && opts.onDrill) {
        opts.onDrill({
          rank: r, tp: model.tpOf(r), cp: model.cpOf(r), pp: model.ppOf(r), rep: model.repOf(r),
          bucket: model.epOf(r), domain: model.domOf(r), stage: model.stageLayerRange(model.ppOf(r)),
        });
        return;
      }
      api.select(r == null ? null : r);
    });
    renderer.domElement.addEventListener('wheel', (ev) => {
      ev.preventDefault();
      camTarget = null;   /* 手动滚轮缩放优先：打断 zoomOnSelect 还没走完的那截自动推近 */
      cam.half = Math.max(4, Math.min(220, cam.half * (ev.deltaY > 0 ? 1.1 : 0.9)));
    }, { passive: false });

    /* ── 主循环 ── */
    let raf = 0, lastRecolor = -1, lastMs = null, lastTimeUi = -1, lastPhase = -1, lastSettling = false;
    function frame(nowMs) {
      raf = global.requestAnimationFrame(frame);
      // 累加制时钟：时间轴游标可拖动定位（绝对时钟会把拖动的 S.t 覆盖掉）
      const dt = lastMs == null ? 0 : (nowMs - lastMs) / 1000;
      lastMs = nowMs;
      if (S.playing) S.t = (S.t + Math.min(dt, 0.1) / STEP_SEC) % 1;   // t ∈ [0,1) = 一个 step 内的位置
      // 状态热力随阶段流动（280ms 重染一次；透镜/异常静态无需重染）
      if (S.playing && S.colorBy === 'load' && S.anom === 'none' && nowMs - lastRecolor > 280) { lastRecolor = nowMs; recolor(); }
      if (S.playing && nowMs - lastTimeUi > 200) {
        lastTimeUi = nowMs;
        const ph = phaseIdx(); syncTimeUI();
        if (ph !== lastPhase) { lastPhase = ph; renderHud(); renderLegend(); renderInfo(); rebuildComm(); }   // 换阶段 → 主导维/图例/信息卡随之切换
      }
      // 重排飞行期间标注降一档不透明度（结束后恢复）：飞行中卡在动、标注还在原位，
      // 满不透明的字牌会看着像卡在半空的贴纸。
      if (settling !== lastSettling) {
        lastSettling = settling;
        axGroup.traverse((o) => {
          const m = o.material;
          if (!m || m.opacity == null) return;
          if (m.userData.baseOp == null) m.userData.baseOp = m.opacity;
          m.opacity = m.userData.baseOp * (settling ? 0.45 : 1);
        });
      }
      /* 位置飞行（切形态重排动画；稳定后停写省 CPU）。
         用「按时间」的缓动而不是「按帧」的固定系数：4000 卡时每帧要写 4000 个矩阵、
         还要重建连线，帧率掉到 10fps 上下——固定 0.14/帧 就变成了要飞好几秒，而且
         期间卡与线都停在半路，看上去就像「线没连到卡上」。
         收尾一律精确吸附（cur = target），杜绝残留误差。 */
      if (settling) {
        const k = dt > 0 ? 1 - Math.pow(0.0015, Math.min(dt, 0.25) / 0.5) : 1;   // 0.5s 走完 99.85%
        let moving = false;
        for (let r = 0; r < N; r++) {
          const i = r * 3;
          for (let c = 0; c < 3; c++) {
            const d = target[i + c] - cur[i + c];
            if (Math.abs(d) > 0.004) { cur[i + c] += d * k; moving = true; }
            else cur[i + c] = target[i + c];                                     // 到位就吸附
          }
          dummy.position.set(cur[i], cur[i + 1], cur[i + 2]);
          dummy.rotation.set(0, 0, 0); dummy.scale.setScalar(scl[r]); dummy.updateMatrix();
          chips.setMatrixAt(r, dummy.matrix);
        }
        chips.instanceMatrix.needsUpdate = true;
        if (S.sel != null) { rebuildComm(); if (S.selEdge) drawSelEdge(); }   // 通信线/对端/选中边随重排飞行
        if (!moving) settling = false;
      }
      // 焦点/悬停框跟随实时位置
      const place = (box, r) => {
        if (r == null || r < 0 || r >= N) { box.visible = false; return; }
        box.visible = true; box.position.set(cur[r * 3], cur[r * 3 + 1], cur[r * 3 + 2]);
      };
      place(hovBox, S.hover === S.sel ? null : S.hover);
      updateSelFx(nowMs);
      updateMovers();
      stepCamera();
      applyCamera();
      renderer.render(scene, camera);
      renderDetail();          // 细节窗画在主画面之上（自己的视口 + scissor）
    }

    /* ── 尺寸 ── */
    function resize() {
      const w = stageEl.clientWidth || 800, h = stageEl.clientHeight || 600;
      renderer.setSize(w, h);
      syncBarH();                 // 顶栏换行会变高，抽屉与右上角的卡都靠这个变量让位
      fitView();
    }
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(resize) : null;
    if (ro) ro.observe(stageEl);

    /* ── 对外 API ── */
    const api = {
      get model() { return model; }, state: S,
      // 自由改并行度：整体重建（校验 ep 整除 dp、rank 上限），布局/轴标/图例/HUD 全部跟随新配置
      setConfig(cfg) {
        let next;
        try { next = createModel(Object.assign({}, model.config, cfg || {})); }
        catch (e) { return { ok: false, error: e.message.replace(/^rubik-cube: /, '') }; }
        if (next.N > 65536) return { ok: false, error: `rank = ${next.N} 超出渲染上限 65536` };
        model = next; syncDims();
        S.sel = null; S.hover = null; S.sliceVal = 0;
        carrySet = objCarrySet();          // 承载集合按新维度重算（EP/PP 一变，谁承载就变了）
        buildField();
        clearComm(); peerMeshes.forEach((m2) => { m2.geometry.setDrawRange(0, 0); m2.visible = false; });
        fitView(); renderAxes(); applyAxVisibility(); fitView(); updateSlab();
        refresh2D(); syncHelp();
        renderHud(); renderLegend(); renderInfo(); syncCfgUI();
        return { ok: true, ranks: model.N };
      },
      setMode(m) {
        S.mode = Math.max(0, Math.min(model.modes.length - 1, m | 0));
        // 收编后的形态只允许自己声明的视角；正交下切过去自动落回轴测
        if (!(model.modes[S.mode].views || [0, 1, 2, 3]).includes(S.view)) S.view = 0;
        /* 换形态是全新的一次 fitView（下面就会立刻按新形态重算机位），camTarget/
           camPreSel 记的是旧形态那份机位，留着不清会让 stepCamera 在这之后
           反过来把镜头拉回旧形态的框——zoomOnSelect 只管"同一形态内选中/取消
           选中"这一步，跨形态直接信 fitView 的结果。 */
        camTarget = null; camPreSel = null;
        retarget(); fitView(); renderAxes(); applyAxVisibility(); fitView(); updateSlab(); buildShard(); buildDetail(); syncDetailCap(); buildNest();
        renderHud(); syncHelp(); syncChrome(); refresh2D();
      },
      setView(v) {
        if (!(model.modes[S.mode].views || [0, 1, 2, 3]).includes(v | 0)) return;
        // 点「轴测」= 回到标准等距机位（拖歪之后也能一键复位，按钮在任何时候都有反馈）
        if ((v | 0) === 0) { cam.theta = ISO.theta; cam.phi = ISO.phi; }
        camTarget = null; camPreSel = null;   /* 同 setMode：换屏信这里马上算的新机位 */
        /* 换屏要重建走线：哪一维塌成点是**逐屏**的判断（groupFlat 读 screenAxes(S.view)），
           不重建就还留着上一屏的画法。 */
        S.view = v | 0; fitView(); applyAxVisibility(); fitView(); rebuildComm(); refresh2D(); syncHelp(); renderInfo();
      },
      setSlice(on, val) { S.sliceOn = !!on; if (val != null) S.sliceVal = val | 0; refresh2D(); },
      /* 素色这档不只换卡的颜色，还把维度签名色（dimc）整体换掉，而轴刻度、块框、
         嵌套壳、通信线都是**建好之后不再重算颜色**的 Three 对象——只调 recolor()
         的话卡阵素了、四周那圈青/橙/蓝/紫的标注还在（实机就是"卡灰了、字还是彩的"）。
         所以进出素色时连它们一起重建。非素色之间互相切（load↔tp↔w…）不受影响，
         走的还是原来那条最轻的路。 */
      setColorBy(k) {
        const wasMono = S.colorBy === 'neutral', nowMono = k === 'neutral';
        S.colorBy = k; recolor(); renderLegend(); syncChrome();
        if (wasMono !== nowMono) { renderAxes(); applyAxVisibility(); rebuildComm(); buildNest(); refresh2D(); }
      },
      setAnomaly(k) { S.anom = k; recolor(); renderHud(); renderLegend(); syncChrome(); },
      select(r) {
        S.sel = r;
        if (S.selEdge) { S.selEdge = null; drawSelEdge(); if (opts.onSelectEdge) opts.onSelectEdge(null); }
        /* opts.zoomOnSelect：选中飞近一档，取消选中飞回选中前那份机位（见
           cam 声明旁边的注释）。camPreSel 只在"从没选中→选中"这一刻记一次
           ——连续换选（选中 A 又直接点 B）不会把"没选中"那份机位弄丢，
           一路换到最后一次取消选中，飞回去的还是最初那份。 */
        if (opts.zoomOnSelect) {
          if (r != null) {
            if (!camPreSel) camPreSel = { half: cam.half, cx: cam.cx, cy: cam.cy, cz: cam.cz };
            camTarget = {
              half: Math.max(4, camPreSel.half * opts.zoomOnSelect),
              cx: cur[r * 3], cy: cur[r * 3 + 1], cz: cur[r * 3 + 2],
            };
          } else if (camPreSel) {
            camTarget = camPreSel; camPreSel = null;
          }
        }
        rebuildComm(); refreshFocus(); renderInfo(); syncChrome();
        /* opts.axisLabelsOnSelect===false 那条只在 applyAxVisibility 里判——
           选中态一变就得重跑一遍，不然刚选中那一刻世界尺寸字牌还亮着，要等
           下一次换形态/换视角才会被收掉。 */
        if (opts.axisLabelsOnSelect === false) applyAxVisibility();
        if (opts.onSelect) {
          opts.onSelect(r == null ? null : {
            /* cp：CP=1（绝大多数消费者）时恒为 0，字段一直都在，只是没人问过——
               这次是第一个 cp>1 的宿主（rank-topology-lite 接 moe718b128k，
               cp=16）要靠它换算矩阵 rank，见 model.cpOf 的注释：CP 与 TP 共轴，
               不是折叠维，选中态不给这个数就没法往回换算。纯加字段，旧消费者
               不读它不受影响。 */
            rank: r, tp: model.tpOf(r), cp: model.cpOf(r), pp: model.ppOf(r), rep: model.repOf(r),
            bucket: model.epOf(r), domain: model.domOf(r), stage: model.stageLayerRange(model.ppOf(r)),
          });
        }
      },
      /* C 档挂点：选中一条通信边 → 回调宿主（例如驾驶舱据此点亮对应的物理链路），
         同时在场景里把这一段加粗高亮。传 null 取消。 */
      selectEdge(e) {
        S.selEdge = e || null;
        drawSelEdge();
        if (opts.onSelectEdge) opts.onSelectEdge(S.selEdge);
        renderInfo();
        return S.selEdge;
      },
      /* 整网图 → 魔方水平切片。紫色切片只在标准的坐标系里成立，所以主动切过去——
         与同类挂点 selectBucket()（切到 EP聚簇）保持一致。
         原先只记状态不切形态：在别的形态里调用它，画面**静默地**什么都不发生。 */
      selectLayer(l) {
        S.selLayer = l;
        if (l != null && S.mode !== 0) { api.setMode(0); return; }   // setMode 内部已 updateSlab + syncChrome
        updateSlab(); syncChrome();
      },
      selectBucket(e) {                                                        // 专家图 → 整面墙（切 EP 聚簇并选中桶内代表卡）
        if (e == null) { api.select(null); return; }
        api.setMode(2); api.select(model.rankOf(0, 0, (e | 0) % EP));
      },
      /* 整网对象透镜（本次深化的主入口）。
         与 selectLayer / selectBucket 一致：**主动切到该对象的主屏**——那两个挂点当年
         就是因为「只记状态不切形态」而在别的形态里静默地什么都不发生。这里不重犯：
         对象的分片只有在它的 best 形态下才 snap 成规整的块/墙，不飞过去等于没给答案。
         `fly:false` 可关掉（宿主自己控形态时用）。 */
      selectObject(id, o2) {
        const next = id && model.netObjBy[id] ? id : null;
        S.obj = next;
        const ob = next && model.netObjBy[next];
        if (ob && ob.best && (!o2 || o2.fly !== false)) {
          const i = model.modes.findIndex((m) => m.key === ob.best);
          if (i >= 0 && i !== S.mode) { api.setMode(i); }
        }
        refreshObj(); syncChrome();
        if (opts.onSelectObject) opts.onSelectObject(next, ob || null);
      },
      get netObjects() { return model.netObjects; },
      setTheme(theme) {
        S.theme = theme === 'light' ? 'light' : 'dark';
        root.setAttribute('data-theme', S.theme);
        readTokens();                     // 色卡随主题重解析
        applySceneBg();
        renderAxes(); applyAxVisibility(); fitView(); recolor(); rebuildComm(); renderLegend(); renderHud();
      },
      setPlaying(p) { S.playing = !!p; syncChrome(); },
      // 全局字牌字号（唯一入口，见 LABEL_PX 注释）：9–28px，越界或非数字回落到 16
      setLabelPx(px) { LABEL_PX = clampLabelPx(px); syncLabelSizes(true); fitView(); },
      get labelPx() { return LABEL_PX; },
      // 连线图层开关（可全关）与集合算法
      setWire(w) {
        Object.assign(S.wire, w || {});
        if (!S.wire.movers) moverMeshes.forEach((m) => { m.visible = false; });
        rebuildComm(); refreshFocus(); renderInfo(); syncChrome();
      },
      // 物理落位（B 档）：每机卡数 / 每 Pod 机数，或 slots 数组做任意 rank→槽位映射
      setPlacement(pl) {
        const keep = S.sel;        // 只是换装机方式，逻辑魔方没变 → 选中的卡不该被清掉
        const res = api.setConfig({ placement: Object.assign({}, model.config.placement, pl || {}) });
        if (res.ok && keep != null && keep < model.N) api.select(keep);
        return res;
      },
      setAlgo(a) { S.algo = a === 'tree' ? 'tree' : a === 'ring' ? 'ring' : 'auto'; rebuildComm(); renderLegend(); syncChrome(); },
      // 定位到 step 内的某个位置（0→1）或某个阶段：t 可传 0..1，或 {phase:'EP'}
      setTime(t) {
        const v = (t && typeof t === 'object' && t.phase)
          ? (Math.max(0, PHASES.findIndex((p) => p.id === t.phase)) + 0.5) / PHASES.length
          : Math.min(0.999, Math.max(0, +t || 0));
        S.t = v;
        recolor(); rebuildComm(); syncTimeUI(); renderHud(); renderLegend(); renderInfo();
        return { t: S.t, phase: PHASES[phaseIdx()].id };
      },
      phases: PHASES,
      scene, camera,               // 只读挂点：宿主联动与自动化校验（连线端点是否落在卡心）用
      resize,
      destroy() {
        global.cancelAnimationFrame(raf);
        if (ro) ro.disconnect();
        clearComm(); clearAxes();
        renderer.dispose();
        root.remove();
      },
    };

    /* ── 启动 ── */
    // 场景底色 = 设计系统的 --background（同页面外壳，明暗由 token 层切换）
    function applySceneBg() {
      let c = '';
      try { c = getComputedStyle(root).getPropertyValue('--background').trim(); } catch (e) { /* noop */ }
      scene.background = new THREE.Color(c || (isDark() ? '#101010' : '#F5F5F5'));
    }
    applySceneBg();
    resize(); fitView(); renderAxes(); applyAxVisibility(); fitView(); updateSlab();
    // 整网对象初值（mount({obj}) / ?obj=）：不飞形态——落地形态由 mount({mode}) 说了算，
    // 若这里再飞一次，带 obj 的链接会覆盖掉同一条链接里写明的 mode。
    if (opts.obj && model.netObjBy[opts.obj]) S.obj = opts.obj;
    carrySet = objCarrySet();
    recolor(); renderHud(); syncHelp(); renderLegend(); renderInfo(); syncChrome(); syncCfgUI(); syncTimeUI(); buildNest();   // 一进来就把结构框摆出来
    raf = global.requestAnimationFrame(frame);
    return api;
  }

  global.PtoRubikCubePattern = { version: '0.2.0', DEFAULTS, DIM_TOKEN, GROUP_TOKENS, createModel, mount };
})(typeof window !== 'undefined' ? window : this);
