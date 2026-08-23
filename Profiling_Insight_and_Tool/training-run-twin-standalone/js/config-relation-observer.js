/* ══════════════════════════════════════════════════════════════════════════
   配置与关系观测 · 拓扑模型层（第 2 项）
   ------------------------------------------------------------------------
   本文件是整页唯一的数据源。四个视图（整网 / Layer 导航 / MoE / Cluster）
   全部从 CroTopology.derive(config) 的产物渲染，不各自维护状态。

   并行维度语义 —— EP 占不占自己的 rank 有两套口径，由 config.moeOrthogonal 选，
   页内「文档」视图 #doc-ep 一章是它的完整说明：
     切出（moeOrthogonal=false，默认，Megatron / MindSpeed / MindFormers 的做法）
       world = DP × PP × TP × CP，EP 不进乘积，而是把 DP 组再切一刀
       512 × 4 × 1 × 1 = 2048 ✓   EDP = DP/EP = 512/64 = 8
     正交（moeOrthogonal=true）
       world = DP × PP × TP × CP × EP
       8 × 4 × 1 × 1 × 64 = 2048 ✓   EDP ≡ DP
   两种口径下 Node = 2048 / 8卡每节点 = 256 ✓，**rank 编址的几何也完全一致** ——
   那 512 张卡本来就排成 8×64 的网格，区别只在把行叫 DP 还是 EDP。

   确定性映射（无随机、无数据文件）：
     layer  ℓ → PP stage  s     : 按 PP 把 L 层尽量均分，前 (L mod PP) 段多 1 层
     expert e → EP rank   p     : p = floor(e / (E / EP))
     (s,d,p,t,c) → global rank r: r = s·(EDP·EP·TP·CP) + d·(EP·TP·CP) + p·(TP·CP) + c·TP + t
                                  d 走的是 EDP 轴（正交档下 EDP ≡ DP，与改动前逐位相同）
     rank   r → node n          : n = floor(r / ranksPerNode)
   ══════════════════════════════════════════════════════════════════════════ */
(function (global) {
  "use strict";

  /* ── 模型预设：非并行的结构常量，来自 openPangu-2.0-Flash 架构参考.md §4 ── */
  const MODEL_PRESETS = {
    "openpangu-flash": {
      id: "openpangu-flash",
      label: "openPangu 2.0 flash 92B",
      hidden: 2560,
      vocab: 151552,
      heads: 48,
      firstKDense: 2,       // L0~L1 是 Dense MLP，其余是 MoE FFN
      dsaEvery: 3,          // L0,3,6,…,45 走 DSA Indexer，其余 SWA
      mtpLayers: 3,         // MTP L46~L48
      denseIntermediate: 9216,
      moeIntermediate: 1024,
      // deck 侧的模型细节：与 patterns/model-architecture-3d-deck 的 openpangu-flash
      // preset 同值，在这里显式持有，避免依赖 pattern 内部 PRESETS 的可见性。
      deck: { depthGap: 46, blockPostLayers: [0, 4, 9, 14, 19, 24, 29, 34, 39] },
      /* defaults 一律按**切出口径**记（页面默认档）：dp 是含 EP 组在内的真 DP。
         参考配置里写的 "dp: 8" 是 EDP，真 DP = 2048/(PP4×TP1×CP1) = 512。
         切到正交档时由 controller.setEpMode() / setModel() 按 EP 换算回 8。 */
      defaults: {
        totalLayer: 46,
        dp: 512, pp: 4, tp: 1, cp: 1,
        microBatch: 1, seqLen: 4096,
        routedExpert: 256, topK: 8, sharedExpert: 1, ep: 64,
        totalRank: 2048, node: 256, card: "910b-64",
      },
    },
    /* Qwen2-7B：稠密（非 MoE）参考案例，用于验证「整网」栏在无 MoE 模型下的呈现。
       结构参数取自用户给定的参考配置（GQA 28Q:4KV，PP 28 层均分 4 段，无 VPP）。 */
    "qwen2-7b": {
      id: "qwen2-7b",
      label: "Qwen2-7B",
      hidden: 3584,
      vocab: 152064,
      heads: 28,
      kvHeads: 4,            // GQA：28 Q head : 4 KV head，head_dim 128
      attentionLabel: "gqa",
      noMoe: true,            // 稠密模型：derive() / deckConfigFrom() / renderMoe() 据此跳过 MoE 分支
      denseIntermediate: 18944,
      archType: "Qwen2ForCausalLM",
      deck: {
        depthGap: 46,
        blockPostLayers: [],
        residualLabel: "Residual stream",
        sideRows: [
          { label: "Input RMSNorm", ids: ["input_norm"] },
          { label: "Q / K / V Projection", ids: ["q_proj", "k_proj", "v_proj"] },
          { label: "RoPE", ids: ["q_rope", "k_rope"] },
          { label: "FlashAttention · GQA", ids: ["attention_core"] },
          { label: "Output Projection", ids: ["o_proj"] },
          { label: "Attention Residual Add", ids: ["attn_residual_add"] },
          { label: "Post-Attention RMSNorm", ids: ["post_attn_norm"] },
          { label: "Gate / Up Linear", ids: ["ffn_gate_up"] },
          { label: "SiLU × Multiply", ids: ["ffn_act"] },
          { label: "Down Linear", ids: ["ffn_down"] },
          { label: "FFN Residual Add", ids: ["ffn_residual_add"] },
        ],
      },
      defaults: {
        totalLayer: 28,
        dp: 2, pp: 4, tp: 1, cp: 1,
        microBatch: 1, seqLen: 4096,
        routedExpert: 1, topK: 1, sharedExpert: 0, ep: 1,
        totalRank: 8, node: 2, card: "910b-64",
      },
    },
  };

  /* ── 卡型号 ───────────────────────────────────────────────────────────────
     只有 hbmGB 参与计算（它是「单卡容量」那个线框盒的高度）；specs 是纯说明，
     出现在容量栏口径浮层里。

     ⚠️ HBM 数字的来源强度不一样，别当成同一档证据：
       · 910B —— 本仓 AI_Profiling_Tool/AscendProfKit/skills/performance-health-score/
         SKILL.md 记「常见 32GB / 64GB 两种规格，需从 NPU_INFO 表确认型号后定」。
         两款都是真实存在的规格，故拆成 910b-32 / 910b-64 两个选项，而不是取其一。
       · 950  —— Profiling_Insight_and_Tool/KNOWLEDGE.md §3.1 只给了**整片** DDR
         128 GB / 1.6 TB/s，没有单卡 HBM 容量。64 是按「与 910B 高配对齐」的
         要求取的占位值，**待确认**。
     拿到准确规格后只改这一个表，容量栏与集群下拉会一起跟上。 */
  const CARD_SPECS = {
    /* label 给口径浮层（那里空间宽裕，带「昇腾」读着完整）；short 给下拉选项
       （集群表单那一格只有 128px，选项文本还要缀上容量）。 */
    "910b-32": {
      id: "910b-32", label: "昇腾 910B", short: "910B", hbmGB: 32,
      hbmNote: "常见 32 / 64 GB 两种规格，此处取 32 GB 款",
      specs: "HBM2e 1.6 TB/s（来源：本仓 performance-health-score 技能卡）",
    },
    "910b-64": {
      id: "910b-64", label: "昇腾 910B", short: "910B", hbmGB: 64,
      hbmNote: "常见 32 / 64 GB 两种规格，此处取 64 GB 款",
      specs: "HBM2e 1.6 TB/s（来源：本仓 performance-health-score 技能卡）",
    },
    "950": {
      id: "950", label: "昇腾 950", short: "950", hbmGB: 64,
      hbmNote: "KNOWLEDGE.md 未给单卡 HBM 容量，64 GB 为占位值，待确认",
      specs: "32 Cube × 64 Vector @1.65 GHz；FP16 432 / FP8 864 / FP4 1728 TFLOPS；"
        + "整片 DDR 128 GB · 1.6 TB/s；Chiplet 2×Compute Die + 2×IO Die，CCU 在 IO-Die；"
        + "超节点 128P / 1024P（来源：KNOWLEDGE.md §3.1、§4.4）",
    },
  };

  const CARD_ORDER = ["910b-32", "910b-64", "950"];
  const DEFAULT_CARD = "910b-64";

  /* ── stepper 字段规格：min/max/step 与取值方式（pow2 = 按 2 的幂增减） ── */
  const FIELD_SPECS = {
    totalLayer:   { label: "Total Layer",    group: "parallel", min: 1,  max: 256,  step: 1 },
    /* dp 的上界比其余并行维宽：切出口径下它记的是**含 EP 组**的真 DP，
       与正交口径的读数差一个 EP 倍（参考配置 8 ↔ 512），1024 会在 EP 大时把
       换算夹掉、Total Rank 跟着变。8192 = node 的上界，两者是同一个量级。 */
    dp:           { label: "DP",             group: "parallel", min: 1,  max: 8192, pow2: true },
    pp:           { label: "PP",             group: "parallel", min: 1,  max: 128,  pow2: true },
    tp:           { label: "TP",             group: "parallel", min: 1,  max: 64,   pow2: true },
    cp:           { label: "CP",             group: "parallel", min: 1,  max: 64,   pow2: true },
    /* 下面两项不参与切分、不进 world_size，只决定**单卡装多少**，所以排在 Cluster
       区、接在卡型号之后：卡型号给容量框的高度（单卡显存），这两项给往框里装的量，
       三者凑成「单卡容量」那一栏的全部可调输入，读下来是一句完整的话。
       （早先它们和 DP/PP/TP/CP 同排在 Model Architecture，那一行讲的是「模型怎么
       切开」，而这两项一刀不切，混在里面反而要多解释一遍。）
       注意是 micro-batch 不是 global batch：GBS 只决定梯度累积步数
       GBS/(MBS×DP)，一步也不进显存 —— 与「DP 不减容器」是同一件事。 */
    /* 写全称不写 MBS / Seq：DP·PP·TP·CP 是这个领域里没人会认错的通用缩写，这两个
       不是 —— MBS 还容易和 GBS 混，而两者对显存的作用完全相反（见容量栏口径）。 */
    microBatch:   { label: "Micro Batch",    group: "batch",    min: 1,  max: 64,   step: 1 },
    seqLen:       { label: "Seq Length",     group: "batch",    min: 128, max: 131072, pow2: true },
    routedExpert: { label: "Routed",         group: "moe",      min: 1,  max: 1024, pow2: true },
    topK:         { label: "Top-K",          group: "moe",      min: 1,  max: 64,   step: 1 },
    sharedExpert: { label: "Shared",         group: "moe",      min: 0,  max: 8,    step: 1 },
    ep:           { label: "EP",             group: "moe",      min: 1,  max: 1024, pow2: true },
    totalRank:    { label: "Total Rank",     group: "cluster",  min: 1,  max: 65536, pow2: true },
    node:         { label: "Node",           group: "cluster",  min: 1,  max: 8192, pow2: true },
  };

  /* batch 单独成组而不是并进 cluster：它要挂到卡型号下拉**之后**的那个容器里
     （#croBatchSteppers），而 cluster 组的容器排在下拉之前。 */
  const FIELD_ORDER = {
    parallel: ["totalLayer", "dp", "pp", "tp", "cp"],
    moe: ["routedExpert", "topK", "sharedExpert", "ep"],
    cluster: ["totalRank", "node"],
    batch: ["microBatch", "seqLen"],
  };

  /* ── 合法邻域 ─────────────────────────────────────────────────────────────
     stepper 只按 2 的幂走时，修复约束靠反复减半就够了；手输放开任意整数之后
     减半修不动了 —— 120 减半是 60，仍然不整除 256。所以「把某个字段挪到离现值
     最近的合法值」升格成基本操作：reconcile 的修复分支和报错横幅的建议修法都用
     它。三个纯函数，都不碰 config。 */
  function gcd(a, b) { return b ? gcd(b, a % b) : a; }

  function clampToSpec(value, spec) { return Math.min(spec.max, Math.max(spec.min, value)); }

  /* base 的倍数里离 preferred 最近的那个。至少取到 base 本身 —— 0 不是合法并行度。
     等距时取大的：约束多为「≥」（专家数 ≥ Top-K、DP ≥ EP），往大了取更容易同时成立。 */
  function nearestMultiple(base, preferred, spec) {
    if (!(base >= 1)) return clampToSpec(preferred, spec);
    const candidates = [Math.floor(preferred / base) * base, Math.ceil(preferred / base) * base, base]
      .filter((v) => v >= base && v >= spec.min && v <= spec.max);
    if (!candidates.length) return clampToSpec(base, spec);
    return candidates.sort((a, b) => Math.abs(a - preferred) - Math.abs(b - preferred) || b - a)[0];
  }

  /* target 的因子里离 preferred 最近的那个。target 上界是 8192，直接扫一遍即可，
     不值得为它写筛法 —— 这个函数只在配置变动时跑，不在渲染路径上。 */
  function nearestDivisor(target, preferred, spec) {
    let best = spec.min;
    let bestGap = Infinity;
    for (let d = 1; d <= target; d += 1) {
      if (target % d !== 0 || d < spec.min || d > spec.max) continue;
      const gap = Math.abs(d - preferred);
      if (gap < bestGap || (gap === bestGap && d > best)) { best = d; bestGap = gap; }
    }
    return best;
  }

  /* ── 取值增减 ─────────────────────────────────────────────────────────── */
  /* 2 的幂梯子上的上一档 / 下一档。手输过 120 之后再点加减，应当落回 64 / 128
     这样的档位，而不是从 120 继续翻倍到 240 —— 手输是「精确指定」，加减是「回到
     常规档位」，两者的职责不同。值本来就在梯子上时与旧的 ×2 / ÷2 逐位相同。 */
  function snapPow2(value, direction) {
    if (direction > 0) {
      let next = 1;
      while (next <= value) next *= 2;
      return next;
    }
    if (value <= 1) return 1;
    let next = 1;
    while (next * 2 < value) next *= 2;
    return next;
  }

  function stepValue(field, value, direction) {
    const spec = FIELD_SPECS[field];
    let next;
    if (spec.pow2) {
      next = snapPow2(value, direction);
      if (next < spec.min) next = spec.min;
    } else {
      next = value + direction * (spec.step || 1);
    }
    return Math.min(spec.max, Math.max(spec.min, next));
  }

  /* ── EP 口径 ──────────────────────────────────────────────────────────────
     正交档 EP 独占 rank，进 world 的乘积；切出档 EP 是从 DP 组内再切一刀，
     不进乘积。整页只此一处判断，validate / reconcile / derive / yaml 都走它。 */
  function epInWorld(config) {
    return config.moeOrthogonal ? config.ep : 1;
  }

  /* d 轴（集群矩阵那一行、rank 编址的 dpIdx）的组数：正交档就是 DP，切出档是
     EDP = DP/EP —— 专家切完之后，专家权重只在剩下的这一维上复制。
     DP % EP 由 validate 拦、reconcile 修，除不尽的配置到不了这里；floor 只是
     防御性的兜底，保证万一漏过去几何仍是整数格。 */
  function expertDataParallel(config) {
    if (config.moeOrthogonal) return config.dp;
    return Math.max(1, Math.floor(config.dp / Math.max(1, config.ep)));
  }

  /* d 轴对人显示的名字。切出档下集群矩阵的每一行是 EDP 而不是 DP —— 表单里
     写着 DP 512、矩阵左侧却标 DP0–7，是两个不同的量重名，必须分开叫。
     EP=1（稠密模型）时 EDP ≡ DP，仍叫 DP，不给没有专家的模型平添一个新词。
     凡是要把 dpIdx 写给人看的地方都走这个函数，别再各写各的。 */
  function dAxisName(counts) {
    return !counts.moeOrthogonal && counts.ep > 1 ? "EDP" : "DP";
  }

  /* rank 的坐标一行文案：关系卡片、计算血缘、事件详情三处共用一份 */
  function coordLine(topology, co) {
    const d = dAxisName(topology.counts);
    return `global rank ${co.rank} · PP${co.stage} / ${d}${co.dpIdx} / EP${co.epIdx}`
      + ` / TP${co.tpIdx} / CP${co.cpIdx} · Node ${co.node}`;
  }

  /* 两种口径下 config.dp 记的不是同一个量，切换口径时按 EP 换算，使 Total Rank
     不变（参考配置的 8 与 512 就是同一份卡的两种读法）。 */
  function convertDpAcrossEpMode(dp, ep, toOrthogonal) {
    const factor = Math.max(1, ep);
    const next = toOrthogonal ? Math.floor(dp / factor) : dp * factor;
    return Math.min(FIELD_SPECS.dp.max, Math.max(FIELD_SPECS.dp.min, next || 1));
  }

  /* ── 校验：返回 [] 表示配置自洽 ───────────────────────────────────────── */
  function validate(config) {
    const errors = [];
    const { totalLayer, dp, pp, tp, cp, routedExpert, topK, ep, totalRank, node } = config;

    if (totalLayer < pp) {
      errors.push(`层数 ${totalLayer} 少于 PP ${pp}，至少每个 stage 要有 1 层`);
    }
    if (routedExpert % ep !== 0) {
      errors.push(`路由专家 ${routedExpert} 不能被 EP ${ep} 整除，专家无法均分到 EP rank`);
    }
    /* 切出档：EDP = DP/EP 必须是整数 —— 专家组要能在 DP 组内均分，否则总有模型
       副本拿不到完整的专家集；页面上还会直接穿帮成「矩阵格子数多于 Total Rank」。
       正交档 EP 独占 rank，与 DP 之间没有整除关系，这条不生效。
       文案里 DP 与 EP 两个 label 都要出现 —— emit() 按 label 子串匹配标红
       stepper，这条错是两个字段共同造成的，两个都该红。 */
    if (!config.moeOrthogonal && dp % ep !== 0) {
      errors.push(`DP ${dp} 不能被 EP ${ep} 整除（EDP = DP ÷ EP 必须是整数），专家组无法在一个数据并行组内均分`);
    }
    if (topK > routedExpert) {
      errors.push(`Top-K ${topK} 超过路由专家总数 ${routedExpert}`);
    }
    const world = dp * pp * tp * cp * epInWorld(config);
    if (world !== totalRank) {
      /* 公式文案跟着口径换：切出档 EP 不占 rank，写进乘积会读成"少算了一维"。
         括注里写「专家并行」而不是「EP」—— emit() 是按 FIELD_SPECS 的 label 在
         错误文案里做子串匹配来标红 stepper 的，出现 "EP" 会把无辜的 EP 也标红。 */
      const expr = config.moeOrthogonal
        ? `DP${dp}×PP${pp}×TP${tp}×CP${cp}×EP${ep}`
        : `DP${dp}×PP${pp}×TP${tp}×CP${cp}（专家并行从 DP 内切出，不进乘积）`;
      errors.push(`${expr} = ${world}，与 Total Rank ${totalRank} 不符`);
    }
    if (totalRank % node !== 0) {
      errors.push(`Total Rank ${totalRank} 不能被节点数 ${node} 整除，每节点卡数不是整数`);
    }
    return errors;
  }

  /* ── 自动配平：保留用户刚调整的字段，只改满足约束所需的最少依赖项 ─────── */
  /* 展开写而不是对字段表求积：EP 是否进乘积由口径决定，一个 reduce 表达不了。 */
  function parallelWorld(config) {
    return config.dp * config.pp * config.tp * config.cp * epInWorld(config);
  }

  function isAllowedParallelValue(field, value, config) {
    const spec = FIELD_SPECS[field];
    if (!Number.isInteger(value) || value < spec.min || value > spec.max) return false;
    if (spec.pow2 && (value & (value - 1)) !== 0) return false;
    if (field === "pp" && value > config.totalLayer) return false;
    if (field === "ep" && config.routedExpert % value !== 0) return false;
    /* 切出档下 DP 必须是 EP 的整数倍（见 validate 的 EDP 整除）。不挡在这里，
       拖 Total Rank 时 fitParallelWorld 会自己配平出一个当场报错的 DP。
       EP 不需要对称的一条：切出档下它已被 fitParallelWorld 的候选列表摘掉。 */
    if (field === "dp" && !config.moeOrthogonal && value % config.ep !== 0) return false;
    return true;
  }

  function nearestDivisorNode(totalRank, preferred) {
    let node = Math.min(preferred, totalRank, FIELD_SPECS.node.max);
    while (node > FIELD_SPECS.node.min && totalRank % node !== 0) node = Math.floor(node / 2);
    return Math.max(FIELD_SPECS.node.min, node);
  }

  /* 切出档下 DP 被「EDP 整除」钉住了下限（DP ≥ EP），当 pp/tp/cp 都已降到 1、
     dp 又正好等于 ep 时，一轮配平会无维可动 —— Total Rank 拖不下去，页面卡在
     一条红字上，而用户看不出该去动哪个 stepper。此时唯一正确的联动是 EP 跟着
     DP 一起降一档：切出档 EP 不进 world，降它不改变乘积，只是把 DP 的下限放开，
     再补一轮就收敛了。ep 每轮严格减半，必然终止。
     只在「DP 确实被 EP 顶住」（dp ≤ ep）且目标在下方时才降 EP —— 别的原因导致的
     不收敛（比如目标本就超出字段量程）降 EP 也没用，白白打乱 MoE 分组。
     anchor === "ep" 时不降：那是用户刚拨的那一枚，配平不该把它拨回去。 */
  function fitParallelWorld(config, target, anchor) {
    fitParallelWorldOnce(config, target, anchor);
    while (!config.moeOrthogonal && anchor !== "ep" && config.ep > 1
      && parallelWorld(config) > target && config.dp <= config.ep) {
      config.ep = Math.floor(config.ep / 2);
      fitParallelWorldOnce(config, target, anchor);
    }
  }

  function fitParallelWorldOnce(config, target, anchor) {
    /* 切出档下 EP 不进 world，改它一分钱也补不上差额，必须从候选里摘掉，
       否则第一轮就会把 EP 调成一个既不解决问题又打乱 MoE 分组的值。 */
    const candidates = ["dp", "ep", "tp", "cp", "pp"]
      .filter((field) => field !== anchor && (field !== "ep" || config.moeOrthogonal));

    // 常见的 Rank ±2 倍只需改一个并行维度；优先 DP，避免扰动模型切分。
    for (const field of candidates) {
      const otherProduct = parallelWorld(config) / config[field];
      const value = target / otherProduct;
      if (isAllowedParallelValue(field, value, config)) {
        config[field] = value;
        return;
      }
    }

    // 单字段无法容纳时，再按优先级逐级配平；所有 stepper 都是 2 的幂，
    // 因而在字段范围允许时可精确收敛到目标值。
    let world = parallelWorld(config);
    const direction = target > world ? 2 : 0.5;
    while (world !== target) {
      const field = candidates.find((name) => {
        const next = config[name] * direction;
        return isAllowedParallelValue(name, next, config)
          && (direction > 1 ? world * direction <= target : world * direction >= target);
      });
      if (!field) break;
      config[field] *= direction;
      world = parallelWorld(config);
    }
  }

  function reconcile(config, anchor) {
    // 模型结构约束：锚点不回退，调整与它直接相依的字段。
    if (config.totalLayer < config.pp) {
      if (anchor === "totalLayer") {
        while (config.pp > config.totalLayer) config.pp = Math.max(1, Math.floor(config.pp / 2));
      } else {
        config.totalLayer = config.pp;
      }
    }

    if (config.topK > config.routedExpert) {
      if (anchor === "topK") {
        /* 专家数要容得下 Top-K，同时仍要能被 EP 整除 —— 一步取到位，别让下面的
           EP 修复再为一个刚抬上来的数把 EP 削一遍。 */
        config.routedExpert = nearestMultiple(config.ep, config.topK, FIELD_SPECS.routedExpert);
        if (config.routedExpert < config.topK) config.topK = config.routedExpert;
      } else {
        config.topK = config.routedExpert;
      }
    }

    /* EP 的两条整除约束合成一条：EP 要整除专家数，切出档下还要整除 DP ——
       也就是 EP 必须是 gcd(专家数, DP) 的因子（正交档只看专家数）。
       手输放开之前这里是反复减半，现在改成直接取最近的合法值：120 减半是 60，
       仍然不整除 256，减半那套修不动手输进来的数。 */
    const epBasis = config.moeOrthogonal ? config.routedExpert : gcd(config.routedExpert, config.dp);
    if (epBasis % config.ep !== 0) {
      if (anchor === "ep") {
        /* 锚在 EP 上：不动 EP，改让专家数（切出档还有 DP）成为它的倍数。
           抬 DP 会让 Total Rank 跟着涨 —— 加大专家并行度本来就要更多卡。 */
        config.routedExpert = nearestMultiple(config.ep, config.routedExpert, FIELD_SPECS.routedExpert);
        if (!config.moeOrthogonal) config.dp = nearestMultiple(config.ep, config.dp, FIELD_SPECS.dp);
        if (config.topK > config.routedExpert) config.topK = config.routedExpert;
      }
      // 抬不动（撞量程）时仍会不整除，兜底还是把 EP 收到最近的合法因子
      const basis = config.moeOrthogonal ? config.routedExpert : gcd(config.routedExpert, config.dp);
      if (basis % config.ep !== 0) config.ep = nearestDivisor(basis, config.ep, FIELD_SPECS.ep);
    }

    if (anchor === "totalRank") {
      fitParallelWorld(config, config.totalRank, anchor);
    } else if (anchor === "node" && config.node > parallelWorld(config)) {
      fitParallelWorld(config, config.node, anchor);
      config.totalRank = parallelWorld(config);
    } else {
      let world = parallelWorld(config);
      const maxRank = FIELD_SPECS.totalRank.max;
      if (world > maxRank) {
        fitParallelWorld(config, maxRank, anchor);
        world = parallelWorld(config);
      }
      config.totalRank = world;
    }

    // Node 是集群展示维度：能沿用就不动，不能整除时取不大于原值的最近合法节点数。
    config.node = nearestDivisorNode(config.totalRank, config.node);
  }

  /* ── 派生：把配置展开成四个视图共用的实体表 ───────────────────────────── */
  function derive(config) {
    const preset = MODEL_PRESETS[config.model] || MODEL_PRESETS["openpangu-flash"];
    const { totalLayer, dp, pp, tp, cp, routedExpert, sharedExpert, ep, totalRank, node } = config;
    const errors = validate(config);

    /* 层 → PP stage：尽量均分，前 (L mod PP) 段各多 1 层。46/4 → 12,12,11,11 */
    const base = Math.floor(totalLayer / pp);
    const remainder = totalLayer % pp;
    const stages = [];
    let cursor = 0;
    for (let s = 0; s < pp; s += 1) {
      const count = base + (s < remainder ? 1 : 0);
      stages.push({ stage: s, lo: cursor, hi: cursor + count - 1, count });
      cursor += count;
    }

    const stageOfLayer = new Array(totalLayer);
    stages.forEach((entry) => {
      for (let l = entry.lo; l <= entry.hi; l += 1) stageOfLayer[l] = entry.stage;
    });

    const layers = [];
    for (let l = 0; l < totalLayer; l += 1) {
      const dense = preset.noMoe ? true : l < preset.firstKDense;
      layers.push({
        index: l,
        stage: stageOfLayer[l],
        ffn: dense ? "dense" : "moe",
        attention: preset.dsaEvery ? (l % preset.dsaEvery === 0 ? "dsa" : "swa") : (preset.attentionLabel || "std"),
      });
    }

    /* 专家 → EP rank */
    const expertsPerEpRank = ep > 0 && routedExpert % ep === 0 ? routedExpert / ep : 0;
    const epRanks = [];
    for (let p = 0; p < ep; p += 1) {
      const experts = [];
      for (let k = 0; k < expertsPerEpRank; k += 1) experts.push(p * expertsPerEpRank + k);
      epRanks.push({ epRank: p, experts, lo: experts[0], hi: experts[experts.length - 1] });
    }
    const epRankOfExpert = (e) => (expertsPerEpRank ? Math.floor(e / expertsPerEpRank) : 0);

    /* rank 编址：stage 最外，EDP 次之，ep 最内（TP/CP 内联在最内层）。
       d 轴走 EDP 而不是 DP —— 切出档下那 DP 个副本里已经含了 EP 组，再乘一次 EP
       会把每个 stage 撑成 world 的 EP 倍。正交档 EDP ≡ DP，与改动前逐位相同。 */
    const edp = expertDataParallel(config);
    const ranksPerEp = tp * cp;
    const ranksPerDp = ep * ranksPerEp;
    const ranksPerStage = edp * ranksPerDp;
    const ranksPerNode = node > 0 ? totalRank / node : 0;

    const rankOf = (stage, dpIdx, epIdx, inner = 0) =>
      stage * ranksPerStage + dpIdx * ranksPerDp + epIdx * ranksPerEp + inner;
    const rankOfCoords = (stage, dpIdx, epIdx, tpIdx = 0, cpIdx = 0) =>
      rankOf(stage, dpIdx, epIdx, cpIdx * tp + tpIdx);
    const nodeOfRank = (rank) => (ranksPerNode ? Math.floor(rank / ranksPerNode) : 0);

    /* ── 关系查询（第 7 项的双向互查全部走这几个函数） ── */
    function ranksOfStage(stage) {
      const out = [];
      const start = stage * ranksPerStage;
      for (let i = 0; i < ranksPerStage; i += 1) out.push(start + i);
      return out;
    }

    function ranksOfLayer(layerIndex) {
      return ranksOfStage(stageOfLayer[layerIndex]);
    }

    /* 单 DP 口径下某个 PP stage 的 rank：整段 stage 里只取一个 DP 副本那一块。
       Emb / Final Norm / LM Head 这类端点对象没有层号、只有驻留的 stage，
       与「层」走的是同一个查询口径，所以口径函数要按 stage 提供一份。 */
    function ranksOfStageInDp(stage, dpIdx = 0) {
      const safeDpIdx = Math.max(0, Math.min(edp - 1, dpIdx));
      const out = [];
      const start = rankOf(stage, safeDpIdx, 0);
      for (let i = 0; i < ranksPerDp; i += 1) out.push(start + i);
      return out;
    }

    function ranksOfLayerInDp(layerIndex, dpIdx = 0) {
      return ranksOfStageInDp(stageOfLayer[layerIndex], dpIdx);
    }


    /* 某层里某个专家实际落在哪些 rank 上：该层所在 stage × 全部 DP 副本 × 该专家的 EP rank */
    function ranksOfExpertInLayer(layerIndex, expert) {
      const stage = stageOfLayer[layerIndex];
      const epIdx = epRankOfExpert(expert);
      const out = [];
      for (let d = 0; d < edp; d += 1) {
        for (let inner = 0; inner < ranksPerEp; inner += 1) out.push(rankOf(stage, d, epIdx, inner));
      }
      return out;
    }

    function ranksOfEpRankInStage(stage, epIdx) {
      const out = [];
      for (let d = 0; d < edp; d += 1) {
        for (let inner = 0; inner < ranksPerEp; inner += 1) out.push(rankOf(stage, d, epIdx, inner));
      }
      return out;
    }

    function nodesOfRanks(ranks) {
      const seen = new Set();
      ranks.forEach((r) => seen.add(nodeOfRank(r)));
      return Array.from(seen).sort((a, b) => a - b);
    }

    /* rank → 反查坐标，供集群图格子点击后回溯层/专家 */
    function coordsOfRank(rank) {
      const stage = Math.floor(rank / ranksPerStage);
      const withinStage = rank - stage * ranksPerStage;
      const dpIdx = Math.floor(withinStage / ranksPerDp);
      const withinDp = withinStage - dpIdx * ranksPerDp;
      const epIdx = Math.floor(withinDp / ranksPerEp);
      const inner = withinDp - epIdx * ranksPerEp;
      const tpIdx = inner % tp;
      const cpIdx = Math.floor(inner / tp);
      return { rank, stage, dpIdx, epIdx, tpIdx, cpIdx, inner, node: nodeOfRank(rank) };
    }

    return {
      config, preset, errors, valid: errors.length === 0,
      // 卡型号：只有 hbmGB 参与计算（单卡容量框的高度），其余是说明性规格
      card: CARD_SPECS[config.card] || CARD_SPECS[DEFAULT_CARD],
      hasMoe: !preset.noMoe,
      stages, layers, epRanks,
      counts: {
        totalLayer,
        denseLayers: preset.noMoe ? totalLayer : Math.min(preset.firstKDense, totalLayer),
        moeLayers: preset.noMoe ? 0 : Math.max(0, totalLayer - preset.firstKDense),
        dsaLayers: layers.filter((l) => l.attention === "dsa").length,
        swaLayers: layers.filter((l) => l.attention === "swa").length,
        routedExpert, topK: config.topK, sharedExpert, ep, expertsPerEpRank,
        /* dp 是配置里那个数（切出档 = 真 DP，正交档 = EP 之外的数据并行度）；
           edp 是集群矩阵 d 轴、rank 编址真正用的组数。凡是「有几个副本要画 / 要
           遍历」一律读 edp，凡是「显示这个配置项的值」读 dp。 */
        dp, edp, moeOrthogonal: Boolean(config.moeOrthogonal),
        pp, tp, cp, totalRank, node, ranksPerNode,
        ranksPerStage, ranksPerDp, ranksPerEp,
      },
      stageOfLayer: (l) => stageOfLayer[l],
      epRankOfExpert,
      expertsOfEpRank: (p) => (epRanks[p] ? epRanks[p].experts : []),
      rankOf, rankOfCoords, nodeOfRank, coordsOfRank,
      ranksOfStage, ranksOfLayer, ranksOfStageInDp, ranksOfLayerInDp,
      ranksOfExpertInLayer, ranksOfEpRankInStage, nodesOfRanks,
    };
  }

  /* ══ stepper UI：复用 .zoom-control-group / .zoom-control-readout / .btn ══ */
  /* stroke-width 3（不是图标常用的 2）：这两枚画在 28px 的圆键里、实际渲染 18px，
     24 格坐标系下的 2 格描边落地不到 1.5px，在深色底上几乎看不见笔画。
     笔画长度也收到 ±7 而不是满格 ±14，粗笔画配满格会糊成一坨。
     width/height 只是没有 CSS 时的兜底，真实尺寸由 .cro-stepper__control .btn svg 给。 */
  const MINUS = '<svg viewBox="0 0 24 24" aria-hidden="true" width="18" height="18" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round"><path d="M5.5 12h13"></path></svg>';
  const PLUS = '<svg viewBox="0 0 24 24" aria-hidden="true" width="18" height="18" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round"><path d="M12 5.5v13"></path><path d="M5.5 12h13"></path></svg>';

  /* ── 建议修法 ─────────────────────────────────────────────────────────────
     手输了一个与其它字段不兼容的数之后，横幅要答出「把哪几个字段改成多少就兼容
     了」。这件事 reconcile 本来就在做，区别只是结果不落到 config 上 —— 所以直接
     拿它跑一份副本。reconcile 的所有修复分支都避开 anchor，唯一的例外是末尾那句
     无条件重算 node，所以这里再校一次锚点：动了锚点就不算「兼容你输入的数」的
     建议，宁可返回 null 让横幅退化成只有「取消修改」一个出口。 */
  function proposeFix(config, anchor) {
    const proposal = { ...config };
    reconcile(proposal, anchor);
    if (proposal[anchor] === config[anchor] && !validate(proposal).length) return proposal;

    /* Total Rank 是唯一一个 reconcile 修不出建议的常用锚点：补它的差额要解一个
       整数分解，而 fitParallelWorld 只在 2 的幂梯子上找解 —— 那是加减键该有的
       手感，不该为手输放开。所以这里补一条直解：让 DP 独自吃掉整个差额，再由
       reconcile 把 EP 收到合法因子上。手输 1000 卡时它答「DP 512→250、EP 64→2」，
       而不是两手一摊。 */
    if (anchor === "totalRank") {
      const alt = { ...config };
      const rest = alt.pp * alt.tp * alt.cp * epInWorld(alt);
      const dp = rest > 0 ? alt.totalRank / rest : 0;
      if (Number.isInteger(dp) && dp >= FIELD_SPECS.dp.min && dp <= FIELD_SPECS.dp.max) {
        alt.dp = dp;
        reconcile(alt, anchor);
        if (alt.totalRank === config.totalRank && !validate(alt).length) return alt;
      }
    }
    return null;
  }

  function buildStepper(field, value, onStep, onType) {
    const spec = FIELD_SPECS[field];
    const wrap = document.createElement("div");
    wrap.className = "cro-stepper";
    wrap.dataset.field = field;

    const label = document.createElement("span");
    label.className = "cro-stepper__label";
    label.textContent = spec.label;

    const control = document.createElement("div");
    control.className = "zoom-control-group cro-stepper__control";

    const dec = document.createElement("button");
    dec.type = "button";
    dec.className = "btn btn-ghost btn-icon btn-sm";
    dec.innerHTML = MINUS;
    dec.setAttribute("aria-label", `减少 ${spec.label}`);

    /* 读数是 input 不是 span：加减键只走 2 的幂档位，手输负责「精确指定」——
       DP=120、PP=3 这类真实存在却不在 2 的幂梯子上的值只能由输入框给。
       仍复用 .zoom-control-readout 的排版，只在本作用域抹掉输入框的原生外观。 */
    const readout = document.createElement("input");
    readout.className = "zoom-control-readout cro-stepper__input";
    readout.type = "text";
    readout.inputMode = "numeric";
    readout.autocomplete = "off";
    readout.spellcheck = false;
    readout.value = String(value);
    readout.setAttribute("aria-label", spec.label);
    /* size 决定输入框的固有宽度。不给的话浏览器按 20 个字符算 —— 这正是换成
       input 之后整行 stepper 变宽的原因。按当前值的位数给（下限 3），读数的宽度
       就与还是 <span> 时的内容自适应一致。 */
    readout.size = Math.max(3, String(value).length);
    // 打字过程中同步跟宽，但**不提交**（提交仍在 Enter / 失焦，见下面的注释）
    readout.addEventListener("input", () => {
      readout.size = Math.max(3, readout.value.length || 1);
    });

    const inc = document.createElement("button");
    inc.type = "button";
    inc.className = "btn btn-ghost btn-icon btn-sm";
    inc.innerHTML = PLUS;
    inc.setAttribute("aria-label", `增加 ${spec.label}`);

    dec.addEventListener("click", () => onStep(field, -1));
    inc.addEventListener("click", () => onStep(field, 1));

    /* Enter / 失焦才提交，Esc 放弃。**不监听 input 事件** —— 输 "120" 要先经过
       "1" 和 "12" 两个中间态，逐键提交会让整页在打字过程中反复标红又复原，
       而且 "1" 这种中间态多半就是不兼容的。 */
    readout.addEventListener("focus", () => readout.select());
    readout.addEventListener("keydown", (event) => {
      if (event.key === "Enter") { event.preventDefault(); readout.blur(); }
      else if (event.key === "Escape") { event.preventDefault(); readout.dataset.abort = "1"; readout.blur(); }
      // 上下键当加减键使：焦点在输入框里时，这是最顺手的「回到常规档位」的动作
      else if (event.key === "ArrowUp") { event.preventDefault(); onStep(field, 1); }
      else if (event.key === "ArrowDown") { event.preventDefault(); onStep(field, -1); }
    });
    readout.addEventListener("blur", () => {
      const abort = readout.dataset.abort;
      delete readout.dataset.abort;
      onType(field, abort ? null : readout.value);
    });

    control.append(dec, readout, inc);
    wrap.append(label, control);
    return wrap;
  }

  /* ══ 控制器：持有 config，渲染 stepper，广播 cro:change ══════════════════ */
  function createController(options = {}) {
    const modelId = options.model || "openpangu-flash";
    /* moeOrthogonal 不进 MODEL_PRESETS.defaults：它是读配置的**口径**而不是模型
       的属性，换模型时应当沿用用户当前选的那一档（见 setModel）。 */
    const config = Object.assign(
      { model: modelId, moeOrthogonal: false },
      MODEL_PRESETS[modelId].defaults,
      options.config,
    );
    const readouts = new Map();
    const wraps = new Map();
    const linkedHighlightTimers = new Map();
    const listeners = [];
    /* 手输进来的不兼容值：页面停在这一态，下游图形一律不更新，直到用户走三个出口
       之一 —— 横幅的「一键应用」/「取消修改」，或用加减键把它步进回合法档位。 */
    let invalidTyped = null;
    // 超量程 / 非整数的一次性提示，与上面那种「参数互不兼容」是两回事
    let rangeHint = null;
    // 最近一次自洽的配置快照：「取消修改」整份退回到它，保证报错态一定有出口
    let lastValidConfig = null;
    // 同一时刻的拓扑，供 topology getter 在冻结期间兜底（见下面的 getter）
    let lastValidTopology = null;

    function mount(container, group) {
      if (!container) return;
      container.innerHTML = "";
      FIELD_ORDER[group].forEach((field) => {
        const stepper = buildStepper(field, config[field], apply, commitTyped);
        readouts.set(field, stepper.querySelector(".zoom-control-readout"));
        wraps.set(field, stepper);
        container.appendChild(stepper);
      });
    }

    function highlightLinkedChanges(before, anchor) {
      Object.keys(FIELD_SPECS).forEach((field) => {
        if (field === anchor || before[field] === config[field]) return;
        const wrap = wraps.get(field);
        if (!wrap) return;

        clearTimeout(linkedHighlightTimers.get(field));
        // 先移除并触发布局，再加回 class，使连续联动也能重新播放 3 秒提示。
        wrap.classList.remove("is-auto-adjusted");
        void wrap.offsetWidth;
        wrap.classList.add("is-auto-adjusted");
        linkedHighlightTimers.set(field, setTimeout(() => {
          wrap.classList.remove("is-auto-adjusted");
          linkedHighlightTimers.delete(field);
        }, 3000));
      });
    }

    function apply(field, direction) {
      rangeHint = null;
      const next = stepValue(field, config[field], direction);
      if (next === config[field]) return;
      const before = { ...config };
      config[field] = next;
      reconcile(config, field);
      /* 加减键永远经 reconcile 落到自洽态，所以它同时也是手输报错态的第三个出口：
         「把错数步进回合理档位」—— snapPow2 保证从 120 加减是回到 128 / 64，
         而不是在 240 / 60 上继续翻倍。 */
      invalidTyped = null;
      highlightLinkedChanges(before, field);
      emit();
    }

    /* 手输提交。与加减键的根本区别：**不做联动修复**。
       联动一步能改掉三四个数（EP 输 120 会连带 Routed / DP / Total Rank 一起动），
       而用户手输时是在明确指定一个值，替他改掉另外几个是错的手感。
       所以这里只有两条路：能自洽就直接落；不能自洽就停在报错态，把「建议怎么改」
       交给横幅，由用户按下「一键应用」才真的改。 */
    function commitTyped(field, raw) {
      rangeHint = null;
      if (raw === null) { emit(); return; }               // Esc：还原显示，不动配置
      const spec = FIELD_SPECS[field];
      const parsed = Number(String(raw).trim());
      /* 非整数 / 超量程不进报错态：那不是「参数之间不兼容」，是这个数根本不在这一
         维的取值范围里，没有可建议的修法。直接还原，并把量程写给用户看。 */
      if (!Number.isInteger(parsed) || parsed < spec.min || parsed > spec.max) {
        rangeHint = { field, raw: String(raw).trim() };
        emit();
        return;
      }
      if (parsed === config[field]) { emit(); return; }
      config[field] = parsed;
      if (!validate(config).length) {
        // 手输的数本身就自洽（PP=3、DP=120 配上对应的 Total Rank）：直接落，无需横幅
        invalidTyped = null;
        emit();
        return;
      }
      invalidTyped = { field, value: parsed, proposal: proposeFix(config, field) };
      emit();
    }

    function set(field, value) {
      rangeHint = null;
      invalidTyped = null;                 // 程序化赋值走 reconcile，必然落到自洽态
      if (config[field] === value) return;
      const before = { ...config };
      config[field] = value;
      reconcile(config, field);
      highlightLinkedChanges(before, field);
      emit();
    }

    /* 整网切换模型：不是单字段的 stepper 调整，而是把并行/批次/MoE 全部换成
       新模型的 defaults（两个模型的层数、并行拓扑、是否有 MoE 都不同，不能只改
       一个字段再 reconcile）。defaults 本身已自洽（见 MODEL_PRESETS 里的注释
       校验），不需要再跑 reconcile()。 */
    function setModel(modelId) {
      const preset = MODEL_PRESETS[modelId];
      if (!preset || config.model === modelId) return;
      // 整份换成新模型的 defaults（本身自洽），手输留下的报错态随之作废
      rangeHint = null;
      invalidTyped = null;
      config.model = modelId;
      Object.assign(config, preset.defaults);
      // defaults 里的 dp 一律按切出口径记；正交档下要按 EP 换算回去，否则 world 差一个 EP 倍
      if (config.moeOrthogonal) {
        config.dp = convertDpAcrossEpMode(config.dp, config.ep, true);
      }
      emit();
    }

    /* EP 口径切换：不是一个 stepper 字段 —— 它改的是 world 公式本身，所以单开
       一个入口而不是走 set()。切换的同时按 EP 换算 DP，让 Total Rank 不变：
       同一份 2048 卡，切出档读作 DP512（EDP=8），正交档读作 DP8。
       anchor 传 dp：reconcile 走 else 分支，由换算后的 world 反推 Total Rank。 */
    function setEpMode(orthogonal) {
      const next = Boolean(orthogonal);
      if (config.moeOrthogonal === next) return;
      rangeHint = null;
      invalidTyped = null;                 // 换口径后走 reconcile，落到自洽态
      const before = { ...config };
      config.moeOrthogonal = next;
      config.dp = convertDpAcrossEpMode(config.dp, config.ep, next);
      reconcile(config, "dp");
      // anchor 传 null：DP 这一跳正是要提示的联动，不该被当成"用户自己改的"而排除
      highlightLinkedChanges(before, null);
      emit();
    }

    /* 报错横幅：错在哪 → 建议怎么改 → 两个出口。
       ⚠️ 设计系统没有 alert / banner 组件（css/style.css 里只有 .btn 系列），
       这里用 tokens 拼一个最小实现，按钮复用 .btn / .btn-sm / .btn-ghost；
       与本文件的 select 同属「缺失样式」，待批准后应吸收进共享系统。 */
    function renderConfigError(topology) {
      const el = document.getElementById("croConfigError");
      if (!el) return;
      el.textContent = "";
      el.classList.toggle("is-blocking", !topology.valid);

      if (rangeHint) {
        const spec = FIELD_SPECS[rangeHint.field];
        const line = document.createElement("p");
        line.className = "cro-config-error__msg";
        line.textContent = `${spec.label} 只接受 ${spec.min}–${spec.max} 之间的整数`
          + `，「${rangeHint.raw}」已还原`;
        el.appendChild(line);
        rangeHint = null;                      // 一次性提示，下一次操作即消失
        if (topology.valid) return;
      }
      if (topology.valid) return;

      const msg = document.createElement("p");
      msg.className = "cro-config-error__msg";
      msg.textContent = topology.errors.join("；");
      /* 停更接在错误文字后面同一行：它是这条错误的**后果**，不是第二条错误，
         另起一行会读成并列的两件事。 */
      const frozen = document.createElement("span");
      frozen.className = "cro-config-error__frozen";
      frozen.textContent = "图形已暂停更新，仍显示上一组自洽的参数";
      msg.appendChild(frozen);
      el.appendChild(msg);

      const fix = document.createElement("div");
      fix.className = "cro-config-error__fix";
      const proposal = invalidTyped && invalidTyped.proposal;
      if (proposal) {
        const label = FIELD_SPECS[invalidTyped.field].label;
        const changes = Object.keys(FIELD_SPECS)
          .filter((f) => f !== invalidTyped.field && proposal[f] !== config[f])
          .map((f) => `${FIELD_SPECS[f].label} ${config[f]} → ${proposal[f]}`);
        fix.textContent = changes.length
          ? `为了兼容 ${label} = ${invalidTyped.value}，建议把 ${changes.join("、")}`
          : `为了兼容 ${label} = ${invalidTyped.value}，无需改动其它字段`;
      } else {
        fix.textContent = "没能算出兼容这个数的改法 —— 换一个值，或退回上一组参数";
      }
      el.appendChild(fix);

      const actions = document.createElement("div");
      actions.className = "cro-config-error__actions";
      if (proposal) {
        const applyBtn = document.createElement("button");
        applyBtn.type = "button";
        applyBtn.className = "btn btn-sm";
        applyBtn.textContent = "一键应用";
        applyBtn.addEventListener("click", () => {
          const before = { ...config };
          const anchor = invalidTyped.field;
          Object.assign(config, proposal);
          invalidTyped = null;
          highlightLinkedChanges(before, anchor);   // 被建议改掉的那几枚闪一下
          emit();
        });
        actions.appendChild(applyBtn);
      }
      const cancelBtn = document.createElement("button");
      cancelBtn.type = "button";
      cancelBtn.className = "btn btn-sm";
      cancelBtn.textContent = "取消修改";
      cancelBtn.addEventListener("click", () => {
        // 整份退回最近一次自洽快照：手输可能连着改了几个字段，只还原最后一个不够
        if (lastValidConfig) Object.assign(config, lastValidConfig);
        invalidTyped = null;
        emit();
      });
      actions.appendChild(cancelBtn);
      el.appendChild(actions);
    }

    function emit() {
      const topology = derive(config);
      readouts.forEach((el, field) => {
        // 正在输入的那一枚不覆写，否则会打断光标和选区
        if (el === document.activeElement) return;
        el.value = String(config[field]);
        el.size = Math.max(3, el.value.length);   // 跟着位数收放，见 buildStepper
      });
      // 校验失败时给出提示：把相关 stepper 标红，并在 #croConfigError 写出原因
      const badFields = new Set();
      if (!topology.valid) {
        topology.errors.forEach((message) => {
          Object.keys(FIELD_SPECS).forEach((field) => {
            if (message.includes(FIELD_SPECS[field].label)) badFields.add(field);
          });
        });
        // 手输的那一枚一定标红：错误文案里未必出现它的 label（比如 Total Rank）
        if (invalidTyped) badFields.add(invalidTyped.field);
        /* 横幅建议要改的那几枚也一起标红。只靠错误文案的 label 匹配会漏一大片：
           「Routed 232 不能被 EP 53 整除」只点了 Routed 与 EP 的名，可 DP 与
           Total Rank 同样与这个输入值不兼容，一个字都没出现。
           **红圈的名单必须与横幅列的名单一致** —— 否则用户看到横幅让改三个数、
           页面只红了一个，会以为横幅算错了。 */
        if (invalidTyped && invalidTyped.proposal) {
          Object.keys(FIELD_SPECS).forEach((field) => {
            if (invalidTyped.proposal[field] !== config[field]) badFields.add(field);
          });
        }
      }
      if (rangeHint) badFields.add(rangeHint.field);
      wraps.forEach((el, field) => el.classList.toggle("is-invalid", badFields.has(field)));
      renderConfigError(topology);

      /* 参数不兼容时**不往下游发**：集群矩阵、整网 deck、单卡容量、YAML 一律停在
         上一组自洽参数上。这页最会骗人的错误就是「数字和图形各说各话」——
         宁可图形滞后一步，也不让它按一组自相矛盾的数字画出一个看着挺合理的样子。 */
      if (!topology.valid) return;
      lastValidConfig = { ...config };
      lastValidTopology = topology;
      listeners.forEach((fn) => fn(topology));
      document.dispatchEvent(new CustomEvent("cro:change", { detail: topology }));
    }

    return {
      config,
      mount,
      set,
      setModel,
      setEpMode,
      /* 冻结期间也返回上一组自洽拓扑：视图侧除了监听 cro:change，还会在别的时机
         直接读它（比如窗口尺寸变化时重算集群矩阵的列数）。只掐事件不掐这里的话，
         一次拖窗口就能把不自洽的参数画进矩阵，冻结就漏了。 */
      get topology() {
        const current = derive(config);
        return current.valid || !lastValidTopology ? current : lastValidTopology;
      },
      onChange(fn) { listeners.push(fn); return () => listeners.splice(listeners.indexOf(fn), 1); },
      refresh: emit,
    };
  }

  /* ══ 整网 deck（第 3 项）══════════════════════════════════════════════════
     直接消费 patterns/model-architecture-3d-deck，不复刻它的 DOM / 投影数学 /
     视图 CSS。只做两件事：
       1. 用 options.config 把 layerCount / stageRanges / dense·DSA 层号 / 专家数
          换成本页 topology 派生出来的值（pattern.json 的 allowedOverrides）。
       2. options.showChrome=false 去掉 pattern 自带的 title + 工具栏
          （3D/正视/侧视 切换、主题、适配），只留正视图 + pan/zoom。
     ═══════════════════════════════════════════════════════════════════════ */
  function deckConfigFrom(topology) {
    const { counts, stages, layers, preset } = topology;
    const lastLayer = Math.max(0, counts.totalLayer - 1);
    const base = {
      id: preset.id,
      label: preset.label,
      layerCount: counts.totalLayer,
      depthGap: preset.deck.depthGap,
      frontLayer: Math.floor(lastLayer / 2),   // 正视图默认停在中间层（46 层 → L23）
      blockPostLayers: preset.deck.blockPostLayers.filter((l) => l <= lastLayer),
      stageRanges: stages.map((s) => [s.lo, s.hi]),
      representativeLayers: stages.map((s) => s.lo),
    };
    if (preset.noMoe) {
      return {
        ...base,
        firstMoeLayer: counts.totalLayer,     // 全 Dense：永不触发 MoE 分支
        denseLayers: layers.map((l) => l.index),
        dsaLayers: [],
        routedExperts: 0,
        topK: 0,
        heads: preset.heads,
        kvHeads: preset.kvHeads,
        mtp: false,
        residualLabel: preset.deck.residualLabel,
        sideRows: preset.deck.sideRows,
      };
    }
    return {
      ...base,
      firstMoeLayer: counts.denseLayers,
      denseLayers: layers.filter((l) => l.ffn === "dense").map((l) => l.index),
      dsaLayers: layers.filter((l) => l.attention === "dsa").map((l) => l.index),
      routedExperts: counts.routedExpert,
      topK: counts.topK,
    };
  }

  /* 只有这些量变了才值得重建 deck（46 层 × ~30 节点，重挂不便宜） */
  function deckSignature(topology) {
    const c = topology.counts;
    return [c.totalLayer, c.pp, c.routedExpert, c.topK, c.denseLayers].join("/");
  }

  function createDeck(hostId, options = {}) {
    const host = document.getElementById(hostId);
    if (!host || !global.PtoModelArchitecture3dDeck) return null;
    const initialPanY = -18; // 正视图视觉重心略偏下，本页默认上移少许，免去每次手拖。
    let controller = null;
    let signature = null;
    let muted = false;   // applyRelation 回写 deck 时，屏蔽它的回调，避免自激

    function build(topology) {
      const next = deckSignature(topology);
      if (controller && next === signature) return controller;
      signature = next;
      controller?.destroy?.();
      host.innerHTML = "";
      controller = global.PtoModelArchitecture3dDeck.render(host, {
        config: deckConfigFrom(topology),
        initialView: "front",          // 只要正视图
        showChrome: false,             // 去掉视图切换 / 主题 / 适配工具栏
        initialTheme: document.documentElement.dataset.theme === "light" ? "light" : "dark",
        // 整网图 → 其余三个视图的反查入口
        onNodeSelect: (selected) => { if (!muted) options.onNodeSelect?.(selected); },
      });
      controller.setPose?.({ panY: initialPanY });
      global.croDeckController = controller;
      return controller;
    }

    return {
      build,
      get controller() { return controller; },
      // 回写 deck 选中态时静音回调
      silently(fn) { muted = true; try { fn(controller); } finally { muted = false; } },
    };
  }

  /* deck 节点 id → 结构条的 (segment, bar)，用于「点整网图算子」反查其余视图 */
  function deckNodeIndex(topology) {
    const index = new Map();
    activeColumns(topology).forEach((col) => {
      col.bars.forEach((bar) => {
        if (bar.deckNode && !index.has(bar.deckNode)) {
          index.set(bar.deckNode, { segment: col.id, bar: bar.id, experts: bar.experts || null, layers: col.layers });
        }
      });
    });
    return index;
  }

  /* ══ 与整网 deck 严格同色（第 4 项 · 修订）═══════════════════════════════
     结构条不再自己算色。deck 在自己的根节点上写了 --pto-model-deck-{op} 这批
     变量（见 pattern.js applySemanticPalette），节点填充是
       linear-gradient(180deg, C 0%, color-mix(C 75%, #000) 100%) + inset 高光。
     这里把那批变量原样搬到 .cro-board 上，bar 用同名 op + 同一条渐变，
     色值与整网图逐位一致，不做二次映射。 */
  const DECK_COLOR_VARS = [
    "embedding", "norm", "attention", "linear", "head", "mlp",
    "act", "gate", "moe", "comm", "decoder", "input", "output", "parameter", "state",
  ];

  function syncDeckPalette(deckRoot, target) {
    if (!deckRoot || !target) return;
    const style = getComputedStyle(deckRoot);
    DECK_COLOR_VARS.forEach((op) => {
      const value = style.getPropertyValue(`--pto-model-deck-${op}`).trim();
      if (value) target.style.setProperty(`--pto-model-deck-${op}`, value);
    });
  }

  /* 整网 deck 的「相关/不相关」标注。
     关系集大多只覆盖流水线的一段 —— 一个 rank 只持有它那个 PP stage 的层，
     外加一端的 Emb 或 Final Norm/LM Head/MTP —— 所以整网里也该只留这一段有色。
     算子粒度的去色（点具体节点）走 .is-selected，那条 CSS 规则要求确实有节点
     被选中；点 rank / stage / 层这些粗粒度对象时一个 .is-selected 都没有，
     必须另有一套按层/按静态段的标注，否则整网永远是满色的。
     判定：层内节点看所在层卡的层号是否在关系集里；静态段节点看 id 是否在
     rel.staticNodes（由相关的端点列贡献）。 */
  function markDeckRelated(rel) {
    const host = document.getElementById("croDeckHost");
    if (!host) return;
    /* deck 正视图一次只显示 rel.deckLayer 那一张卡，而"相关"是按真实层号判的。
       两者平时是同一个层（deckLayer 一律取自 rel.layers）；万一正在显示的卡
       不在 rel.layers 里，它的节点会全被判成不相关，而"亮"在 deck 上是靠**其余
       节点变灰**表达的，结果就是整张卡一个节点都不亮。
       所以把当前展示的那一层也算作相关。只在关系确实覆盖到层时才生效：Emb /
       Norm / Head 这类端点选择 rel.layers 是空的，它们该亮的是 staticNodes 里
       那几个静态节点，不能顺手把 L0 整张卡点亮。 */
    const proxy = rel && rel.layers.size && Number.isFinite(rel.deckLayer) ? rel.deckLayer : null;
    const relatedLayer = (l) => Boolean(rel) && (rel.layers.has(l) || l === proxy);
    const layerOf = (el) => Number(el.closest(".pto-model-deck__layer")?.dataset.layer);
    host.querySelectorAll(".pto-model-deck__layer").forEach((card) => {
      card.classList.toggle("is-related", relatedLayer(Number(card.dataset.layer)));
    });
    host.querySelectorAll(".pto-model-deck__node, .pto-model-deck__experts").forEach((node) => {
      const layer = layerOf(node);
      const related = Boolean(rel) && (Number.isFinite(layer)
        ? relatedLayer(layer)
        : rel.staticNodes.has(node.dataset.node));
      node.classList.toggle("is-related", related);
    });
  }

  /* ══ 结构条：五段（第 4 项）══════════════════════════════════════════════
     bar.deckNode 对应 patterns/model-architecture-3d-deck 的节点 id，
     第 7 项据此调 deck.selectNode() 联动高亮。 */
  /* bar.op 就是 deck 里同一个节点的 data-op，保证两边取到同一个色变量。
     每列的 units = 该列在 Layer 导航里占的刻度：Dense/MoE 是真实层，
     Emb / Norm / Head 各占 1 格（46 层 + 3 格 = 49 格）。
     col.stageAnchor —— 端点列没有 layers，但它们真实驻留在流水线两端的
     PP stage 上（Emb 在首段、Final Norm / LM Head 在末段）。关系引擎靠它
     把端点算子接回 PP 段与集群 rank，否则点 Emb/Norm/Head 只亮结构条自己。 */
  function structureColumns(topology) {
    const { counts } = topology;
    const denseLast = counts.denseLayers - 1;
    const moeFirst = counts.denseLayers;
    const moeLast = counts.totalLayer - 1;
    const range = (lo, hi) => Array.from({ length: hi - lo + 1 }, (_, i) => lo + i);

    const attnBar = { id: "attn", label: "Attn", op: "attention", deckNode: "attention_core" };
    const normBar = { id: "post_mlp_norm", label: "Post-MLP Norm", op: "norm", deckNode: "post_mlp_norm" };

    const columns = [
      {
        id: "emb", name: "Emb", layers: [], units: ["emb"], stageAnchor: "first",
        bars: [{ id: "embedding", label: "Token Embedding", op: "embedding", deckNode: "embedding" }],
      },
    ];

    if (counts.denseLayers > 0) {
      columns.push({
        id: "dense",
        name: `Dense x${counts.denseLayers}（L0~L${denseLast}）`,
        layers: range(0, denseLast),
        bars: [
          attnBar,
          { id: "dense_gate_up", label: "Gate / Up Linear", op: "linear", deckNode: "dense_gate_up" },
          { id: "dense_down", label: "Dense Down", op: "linear", deckNode: "dense_down" },
          normBar,
        ],
      });
    }

    if (counts.moeLayers > 0) {
      columns.push({
        id: "moe",
        name: `MoE x${counts.moeLayers}（L${moeFirst}~L${moeLast}）`,
        layers: range(moeFirst, moeLast),
        bars: [
          attnBar,
          { id: "gate", label: `Router · Top-${counts.topK}`, op: "gate", deckNode: "gate", experts: "routed" },
          { id: "a2a_dispatch", label: "EP Dispatch", op: "comm", deckNode: "a2a_dispatch", experts: "routed" },
          { id: "expert_pool", label: `Expert Pool ×${counts.routedExpert}`, op: "moe", deckNode: "expert_pool", experts: "routed" },
          { id: "shared_expert", label: `Shared Expert ×${counts.sharedExpert}`, op: "mlp", deckNode: "shared_expert", experts: "shared" },
          { id: "a2a_combine", label: "EP Combine", op: "comm", deckNode: "a2a_combine", experts: "routed" },
          normBar,
        ],
      });
    }

    columns.push(
      {
        id: "norm", name: "Norm", layers: [], units: ["norm"], stageAnchor: "last",
        bars: [{ id: "final_norm", label: "Final RMSNorm", op: "norm", deckNode: "final_norm" }],
      },
      {
        id: "head", name: "Head", layers: [], units: ["head"], stageAnchor: "last",
        bars: [
          { id: "lm_head", label: "LM Head", op: "head", deckNode: "lm_head" },
          { id: "logits", label: "Logits", op: "output", deckNode: "logits" },
        ],
      },
    );
    return columns;
  }

  /* ══ 结构条 = 整网 deck 的算子投影 ═══════════════════════════════════════
     不再另写一份算子清单。deck 每层卡片真实渲染了 ~25 个节点，结构条直接读
     它的 DOM（节点 id / data-op / 文案），保证「整网图里有的算子，五列里都有」，
     且颜色天然一致（用的就是同一个 data-op）。deck 不可用时回落到骨架清单。 */
  const DECK_NODE_ALIASES = {
    q_residual_add: "Q Residual Add",
    kv_residual_add: "KV Residual Add",
    o_residual_add: "Output Residual Add",
    moe_branch_add: "MoE Branch Add",
    ffn_residual_add: "FFN Residual Add",
  };

  const EXPERT_ROLE = {
    gate: "routed",            // Router 对全部路由专家打分，关系上牵连整池
    expert_pool: "routed",
    a2a_dispatch: "routed",
    a2a_combine: "routed",
    shared_expert: "shared",
  };

  function readDeckNodes(scope) {
    if (!scope) return [];
    const out = [];
    scope.querySelectorAll(".pto-model-deck__node, .pto-model-deck__experts").forEach((el) => {
      const id = el.dataset.node;
      const op = el.dataset.op;
      if (!id || op === "mhc-state") return;   // mhc-state 只在侧视图出现
      let label = (el.textContent || "").trim();
      if (!label || label === "+") label = DECK_NODE_ALIASES[id] || id.replace(/_/g, " ");
      if (el.classList.contains("pto-model-deck__experts")) {
        label = el.getAttribute("aria-label") || "Expert Pool";
      }
      if (out.some((n) => n.id === id)) return;
      out.push({ id, label, op, deckNode: id, experts: EXPERT_ROLE[id] || null });
    });
    return out;
  }

  /* 用 deck 的真实节点填充五列的 bars；任何一段读不到就保留骨架里的那一段 */
  function projectDeckOntoColumns(columns, deckRoot, topology) {
    if (!deckRoot) return columns;
    const firstDense = topology.layers.find((l) => l.ffn === "dense");
    const firstMoe = topology.layers.find((l) => l.ffn === "moe");
    const layerScope = (layer) => (layer
      ? deckRoot.querySelector(`.pto-model-deck__layer[data-layer="${layer.index}"]`)
      : null);

    const input = readDeckNodes(deckRoot.querySelector(".pto-model-deck__static--input"));
    const output = readDeckNodes(deckRoot.querySelector(".pto-model-deck__static--output"));
    const dense = readDeckNodes(layerScope(firstDense));
    const moe = readDeckNodes(layerScope(firstMoe));

    // 输出段以 final_norm 为界：它归 Norm 列，其后的 LM Head / Logits / MTP 归 Head 列
    const normAt = output.findIndex((n) => n.id === "final_norm");
    const normBars = normAt >= 0 ? output.slice(0, normAt + 1) : [];
    const headBars = normAt >= 0 ? output.slice(normAt + 1) : output;

    const bySegment = { emb: input, dense, moe, norm: normBars, head: headBars };
    return columns.map((col) => {
      const bars = bySegment[col.id];
      return bars && bars.length ? { ...col, bars } : col;
    });
  }

  /* 全页统一从这里拿列定义：骨架（列名 / 层归属 / 刻度数）+ deck 投影的算子。 */
  function activeColumns(topology) {
    return projectDeckOntoColumns(
      structureColumns(topology),
      document.getElementById("croDeckHost"),
      topology,
    );
  }

  /* Layer 导航与结构条共用同一套列宽，两块严格对齐成一个整体。
     五列等宽（不再按刻度数配比）——MoE 有 44 层，按比例分会把 Emb/Norm/Head
     压成窄条，五个典型层面板宽度也就参差不齐。 */
  function columnTemplate(columns) {
    return `repeat(${columns.length}, minmax(0, 1fr))`;
  }

  /* ══ 渲染：Layer 导航（第 4 项 · 修订 2）═══════════════════════════════════
     严格照 default.png：一条**连续**刻度带，Emb 1 格 + 46 个 decoder 层 +
     Norm / Head 2 格 = 49 格，全带等宽等距（4px 刻度 / 4px 间隙）。
     带子按「PP 边界 ∪ Dense|MoE 起止」切成若干组，组间留 NAV_SPLIT 的空当，
     空当正中画一条竖分隔线；两套分组各自标在带子的上下两侧：
       上 —— PP0…PPn 纯文字，分隔线从标签行顶画到刻度行下方；
       下 —— Dense / MoE 纯文字，分隔线从刻度行顶画到标签行下方。
     没有卡片底色、没有胶囊标签、没有横向分割线 —— 参考图里都不存在。
     几何（分隔线 x、标签左右边界）一律实测写入：PP 边界会落在 Dense|MoE
     之间，按比例硬算会错位。 */

  /* 组间空当 / 刻度宽 = 26 : 4，量自参考图。带子比参考图宽时整体等比放大，
     刻度与空当一起变粗，而不是把余量全丢给某一边。 */
  const NAV_SPLIT_RATIO = 6.5;
  const NAV_TICK_MIN = 1.5;
  const NAV_TICK_MAX = 8;

  /* 把五列摊平成一条刻度槽序列，并解出两套分组的切点。
     切点一律用「组下标」表达（第 g 组之前的那道缝），layoutLayerNav 只需把
     组下标换算成缝的中点，不必再关心层号。 */
  function navModel(topology) {
    const columns = activeColumns(topology);
    const slots = [];
    const columnStart = [];
    const slotOfLayer = new Map();

    columns.forEach((col) => {
      columnStart.push(slots.length);
      if (col.layers.length) {
        col.layers.forEach((l) => { slotOfLayer.set(l, slots.length); slots.push({ layer: l }); });
      } else {
        slots.push({ unit: col.id, column: col });   // Emb / Norm / Head 各占 1 格
      }
    });

    // 分区起止：每道列缝都断开。Emb / Norm / Head 底部也各自出注记，它们就得是
    // 独立的组 —— 否则 groupAt 找不到对应切点，会一路退回 0 组，注记全挤到左端。
    // 这也补上了 Norm|Head 之间原先缺的那道分隔线。
    const ffnCuts = [];
    columns.forEach((col, i) => {
      if (i === 0) return;
      ffnCuts.push(columnStart[i]);
    });
    // PP 的起止：每个 stage 的首层
    const ppCuts = topology.stages.slice(1)
      .map((entry) => slotOfLayer.get(entry.lo))
      .filter((v) => Number.isFinite(v));

    const splits = Array.from(new Set([...ffnCuts, ...ppCuts]))
      .filter((v) => v > 0 && v < slots.length)
      .sort((a, b) => a - b);

    const groups = [];
    let from = 0;
    splits.concat(slots.length).forEach((cut) => { groups.push({ from, to: cut }); from = cut; });

    // 组下标：slots.length → groups.length（带子右端），0 → 0（带子左端）
    const groupAt = (slot) => (slot >= slots.length
      ? groups.length
      : Math.max(0, groups.findIndex((g) => g.from === slot)));

    const ppSpans = topology.stages.map((entry, i) => ({
      stage: entry.stage,
      title: `PP${entry.stage} · L${entry.lo}~L${entry.hi}（${entry.count} 层）`,
      g0: i === 0 ? 0 : groupAt(slotOfLayer.get(entry.lo)),
      g1: i === topology.stages.length - 1 ? groups.length : groupAt(slotOfLayer.get(topology.stages[i + 1].lo)),
    }));

    // 底部注记覆盖全部五列：有层的列报 Dense / MoE，Emb / Norm / Head 报列名
    const ffnSpans = columns.map((col, i) => ({
      segment: col.id,
      label: col.layers.length
        ? (topology.layers[col.layers[0]].ffn === "dense" ? "Dense" : "MoE")
        : col.name,
      g0: groupAt(columnStart[i]),
      g1: groupAt(columnStart[i] + Math.max(1, col.layers.length)),
    }));

    return {
      slots, groups, ppSpans, ffnSpans,
      ppRules: [0, ...ppCuts.map(groupAt), groups.length],   // 含带子两端
      ffnRules: ffnCuts.map(groupAt),
    };
  }

  /* ══ 悬浮气泡（全页统一）══════════════════════════════════════════════════
     两条老路都不好使：
       · 原生 title —— 要按住不动 ~1s 才弹，Layer 刻度只有 3~4px 宽，光是"停稳"
         就够费劲，再等一秒等于查不了层号；样式也完全不可控。
       · 伪元素气泡（training-run-twin.css 的 .twin-heat-cell::after）—— 画在格子
         内部，会被 .cro-heat / 刻度带自己的滚动裁剪切掉，边缘一圈格子只能看到
         半个气泡。
     所以统一挂一个 body 级的 position:fixed 气泡，事件委托到 document：谁带
     data-tip 就给谁弹，位置按目标 rect 现算并夹在视口内，与任何祖先的 overflow
     都无关。 */
  let tipEl = null;
  let tipTarget = null;
  let tipTimer = 0;

  function placeTip(target) {
    const rect = target.getBoundingClientRect();
    const box = tipEl.getBoundingClientRect();
    const GAP = 8;
    const EDGE = 8;
    // 默认贴在目标上方；上方装不下（刻度带在页面顶部、集群图首行同理）翻到下方
    let top = rect.top - box.height - GAP;
    if (top < EDGE) top = Math.min(rect.bottom + GAP, global.innerHeight - box.height - EDGE);
    const half = box.width / 2;
    const left = Math.min(
      Math.max(rect.left + rect.width / 2 - half, EDGE),
      Math.max(EDGE, global.innerWidth - box.width - EDGE),
    );
    tipEl.style.top = `${Math.max(EDGE, top)}px`;
    tipEl.style.left = `${left}px`;
  }

  function hideTip() {
    global.clearTimeout(tipTimer);
    tipTarget = null;
    if (tipEl) tipEl.classList.remove("is-visible");
  }

  function showTip(target) {
    const text = target.dataset.tip;
    if (!text) return;
    tipTarget = target;
    tipEl.textContent = text;
    tipEl.classList.add("is-visible");
    // 先可见才量得到尺寸（气泡宽度随文字走），再定位
    placeTip(target);
  }

  function installTipLayer() {
    if (tipEl) return;
    tipEl = document.createElement("div");
    tipEl.className = "cro-tip";
    tipEl.setAttribute("role", "tooltip");
    document.body.appendChild(tipEl);

    // 60ms 只为压掉快速划过时的连闪，不构成"等待"
    const arm = (target) => {
      if (target === tipTarget) return;
      global.clearTimeout(tipTimer);
      tipTimer = global.setTimeout(() => showTip(target), 60);
    };
    document.addEventListener("pointerover", (event) => {
      const target = event.target.closest?.("[data-tip]");
      if (target) arm(target);
      else if (tipTarget) hideTip();
    });
    document.addEventListener("pointerdown", hideTip);
    // 滚动/缩放后 rect 已经不是气泡当初贴的那个位置，直接收掉而不是跟着漂
    global.addEventListener("scroll", hideTip, true);
    global.addEventListener("resize", hideTip);
    // 键盘走查同样要看得到（刻度带/集群网格用方向键、Tab 移动焦点）。
    // 只认 :focus-visible —— 鼠标点击也会 focus，那时刚被 pointerdown 收掉的气泡
    // 会立刻弹回来挡住刚选中的东西。
    document.addEventListener("focusin", (event) => {
      const target = event.target.closest?.("[data-tip]");
      if (target && target.matches?.(":focus-visible")) showTip(target);
      else hideTip();
    });
    document.addEventListener("focusout", hideTip);
    // 配置一改，集群网格/刻度带整体重建，气泡贴着的那个元素已经不在了 ——
    // 光标没动就不会有 pointerover，得主动收掉，否则悬在半空
    document.addEventListener("cro:change", hideTip);
  }

  function renderLayerNav(container, topology, emit) {
    if (!container) return;
    const model = navModel(topology);
    container.innerHTML = "";

    const band = document.createElement("div");
    band.className = "cro-layer-nav__band";

    // ── 中：连续刻度带，按切点分组 ──
    const strip = document.createElement("div");
    strip.className = "cro-layer-nav__strip";
    model.groups.forEach((group) => {
      const cell = document.createElement("div");
      cell.className = "cro-layer-nav__group";
      for (let i = group.from; i < group.to; i += 1) {
        const slot = model.slots[i];
        const tick = document.createElement("button");
        tick.type = "button";
        tick.className = "cro-tick";
        if (slot.unit) {
          // Emb / Norm / Head：不是层，但和层刻度同宽同高（参考图里没有区别）
          const col = slot.column;
          tick.classList.add("is-endpoint");
          tick.dataset.unit = col.id;
          // 用 data-tip 而不是 title：原生 title 有 ~1s 延迟，刻度只有几像素宽，
          // 悬浮查层号是这条带子的主要用法，等不起（见 installTipLayer）
          tick.dataset.tip = col.name;
          tick.setAttribute("aria-label", col.name);
          tick.addEventListener("click", () => emit({
            kind: "segment", segment: col.id, bar: col.bars[0].id,
            deckNode: col.bars[0].deckNode, layers: [],
          }));
        } else {
          const layer = topology.layers[slot.layer];
          tick.dataset.layer = String(slot.layer);
          tick.dataset.ffn = layer.ffn;
          tick.dataset.attn = layer.attention;
          tick.dataset.tip = `Layer ${slot.layer} · PP${layer.stage} · ${layer.ffn === "dense" ? "Dense" : "MoE"} · ${layer.attention.toUpperCase()}`;
          tick.setAttribute("aria-label", tick.dataset.tip);
          tick.addEventListener("click", () => emit({ kind: "layer", layer: slot.layer }));
        }
        cell.appendChild(tick);
      }
      strip.appendChild(cell);
    });
    band.appendChild(strip);

    // ── 上：PP 标签 ──
    model.ppSpans.forEach((entry) => {
      const span = document.createElement("button");
      span.type = "button";
      span.className = "cro-pp-span";
      span.dataset.stage = String(entry.stage);
      span.dataset.g0 = String(entry.g0);
      span.dataset.g1 = String(entry.g1);
      span.textContent = `PP${entry.stage}`;
      span.dataset.tip = entry.title;
      span.setAttribute("aria-label", entry.title);
      span.addEventListener("click", () => emit({ kind: "stage", stage: entry.stage }));
      band.appendChild(span);
    });

    // ── 下：Dense / MoE 标签（纯文字，不可点，只是分区注记） ──
    model.ffnSpans.forEach((entry) => {
      const span = document.createElement("span");
      span.className = "cro-ffn-span";
      span.dataset.segment = entry.segment;
      span.dataset.g0 = String(entry.g0);
      span.dataset.g1 = String(entry.g1);
      span.textContent = entry.label;
      band.appendChild(span);
    });

    // ── 分隔线 ──
    const addRule = (kind, g) => {
      const rule = document.createElement("div");
      rule.className = `cro-nav-rule cro-nav-rule--${kind}`;
      rule.dataset.g = String(g);
      band.appendChild(rule);
    };
    model.ppRules.forEach((g) => addRule("pp", g));
    model.ffnRules.forEach((g) => addRule("ffn", g));

    container.appendChild(band);
    requestAnimationFrame(() => layoutLayerNav(container));
  }

  /* 布局两步走：
     1. 解出刻度宽度 —— 带子恰好填满可用宽度。刻度与间隙同宽 t，一组 k 格占
        (2k-1)t；组间与带子两端各留一个 split，且 split = 6.5t（参考图比例）：
          width = (2n - g)·t + g·6.5t   （n 格、g 组）
        t 被上下限夹住时（层数很多 / 带子特别宽）余量反过来吃进 split，
        保证带子既不横向溢出、也不在右侧留一截空当。
     2. 分隔线 / 标签的左右边界按实测组位置写入：切点 = 相邻两组之间那道缝的
        中点，带子两端 = strip 的 padding-box 边。 */
  function layoutLayerNav(container) {
    if (!container) return;
    const band = container.querySelector(".cro-layer-nav__band");
    const strip = container.querySelector(".cro-layer-nav__strip");
    if (!band || !strip) return;
    const groups = Array.from(strip.querySelectorAll(".cro-layer-nav__group"));
    const ticks = strip.querySelectorAll(".cro-tick").length;
    if (!groups.length || !ticks) return;

    const width = strip.clientWidth;
    const span = 2 * ticks - groups.length;
    const tick = Math.max(NAV_TICK_MIN, Math.min(NAV_TICK_MAX,
      width / (span + NAV_SPLIT_RATIO * groups.length)));
    const split = Math.max(4, (width - tick * span) / groups.length);
    container.style.setProperty("--cro-tick-w", `${tick}px`);
    container.style.setProperty("--cro-nav-split", `${split}px`);

    const base = band.getBoundingClientRect();
    const stripRect = strip.getBoundingClientRect();
    const rects = groups.map((g) => g.getBoundingClientRect());
    const boundaryX = (g) => {
      if (g <= 0) return stripRect.left - base.left;
      if (g >= rects.length) return stripRect.right - base.left;
      return (rects[g - 1].right + rects[g].left) / 2 - base.left;
    };

    band.querySelectorAll(".cro-nav-rule").forEach((rule) => {
      rule.style.left = `${boundaryX(Number(rule.dataset.g))}px`;
      rule.style.visibility = "visible";
    });
    band.querySelectorAll(".cro-pp-span, .cro-ffn-span").forEach((el) => {
      const left = boundaryX(Number(el.dataset.g0));
      const right = boundaryX(Number(el.dataset.g1));
      el.style.left = `${left}px`;
      el.style.width = `${Math.max(0, right - left)}px`;
      el.style.visibility = "visible";
    });
  }

  /* ══ 典型层里的并行分支 ══════════════════════════════════════════════════
     deck（model-architecture-3d-deck）把两组算子真的画成并排的两条竖直支路：
       · 注意力的 Q 路径 ∥ KV 路径（deck 里 x=98 vs x=446，同一 y）；
       · MoE 的路由专家支路（Router→Dispatch→Expert Pool→Combine）∥ 共享专家
         支路（shared_expert 在 x=508）。
     投影成典型层时若一律竖排，就把「并行」读成了「串行」。下表按 deck 的
     SIDE_ROWS 配对声明每组并行支路的左/右分栏成员（lanes[0]=左、lanes[1]=右，
     与 deck 的 x 顺序一致），renderStructure 据此把这一段渲染成左右两条子栈；
     在 deck 里两支汇合的节点（attention_core / moe_branch_add）本身不属于任何
     分栏，会自然收束回整条竖排。deck 换布局时改这里即可，其余逻辑不动。 */
  const PARALLEL_GROUPS = [
    { id: "attn_qkv", lanes: [
      ["q_a_proj", "q_causal_conv", "q_residual_add", "q_a_norm", "q_b_proj", "query_tensor"],
      ["kv_a_proj", "kv_causal_conv", "kv_residual_add", "kv_a_norm", "kv_b_proj", "key_tensor"],
    ] },
    { id: "moe_branch", lanes: [
      ["gate", "a2a_dispatch", "expert_pool", "a2a_combine"],
      ["shared_expert"],
    ] },
    // Qwen2-7B（GQA，无 mHC 低秩投影）：Q/K/V 三路并排，见 layerHtmlQwen
    { id: "qwen_qkv", lanes: [["q_proj"], ["k_proj"], ["v_proj"]] },
  ];
  /* deckNode id → { group, lane, laneCount }：同一 id 只属于一组一栏。 */
  const PARALLEL_LOOKUP = (() => {
    const map = new Map();
    PARALLEL_GROUPS.forEach((group) => {
      group.lanes.forEach((ids, lane) => {
        ids.forEach((id) => map.set(id, { group: group.id, lane, laneCount: group.lanes.length }));
      });
    });
    return map;
  })();

  /* ══ 渲染：五段结构条 ════════════════════════════════════════════════════ */
  function renderStructure(container, topology, emit) {
    if (!container) return;
    const columns = activeColumns(topology);
    container.innerHTML = "";
    // 与 Layer 导航同一套列宽，两块对齐成一个整体
    container.style.gridTemplateColumns = columnTemplate(columns);

    const makeBar = (bar, col) => {
      const el = document.createElement("button");
      el.type = "button";
      el.className = "cro-bar";
      el.dataset.segment = col.id;
      el.dataset.bar = bar.id;
      if (bar.deckNode) el.dataset.deckNode = bar.deckNode;
      if (bar.experts) el.dataset.experts = bar.experts;
      el.dataset.op = bar.op;   // 与 deck 节点同名 op → 取到同一个 --pto-model-deck-* 色
      el.textContent = bar.label;
      el.addEventListener("click", () => emit({
        kind: "segment",
        segment: col.id,
        bar: bar.id,
        deckNode: bar.deckNode,
        experts: bar.experts || null,
        layers: col.layers,
      }));
      return el;
    };

    columns.forEach((col) => {
      const wrap = document.createElement("div");
      wrap.className = "cro-structure__col";
      wrap.dataset.segment = col.id;
      /* 整列（名字 + 底板）都是「选中这一整个典型层」的热区：命中列内某根 .cro-bar
         时直接放行（那颗算子条自己的 click 走单算子通路），否则发一个 wholeColumn
         选择 —— 锚点是整列底板（.cro-structure__stack）、整网侧是整张层卡，而不是
         列里的第一个算子。resolveRelation 据 wholeColumn 覆盖整段的层/专家/rank。 */
      wrap.addEventListener("click", (event) => {
        if (event.target.closest(".cro-bar")) return;
        emit({ kind: "segment", segment: col.id, wholeColumn: true, layers: col.layers });
      });

      const name = document.createElement("button");
      name.type = "button";
      name.className = "cro-structure__name";
      name.textContent = col.name;
      name.title = col.name;
      name.setAttribute("aria-label", col.name);

      const stack = document.createElement("div");
      stack.className = "cro-structure__stack";

      /* 把 bars 切成「整条竖排段」与「并行分支段」交替的块：连续且属于同一
         并行组的 bar 收进一块，用左右分栏子栈渲染；其余 bar 直接整条竖排。
         bar 在 col.bars 里本就按 deck 的 y 顺序排列，故各栏内竖排顺序天然正确。 */
      let pending = null;   // { group, lanes: bar[][] }
      const flush = () => {
        if (!pending) return;
        const lanes = document.createElement("div");
        lanes.className = "cro-structure__lanes";
        pending.lanes.forEach((barsInLane) => {
          const lane = document.createElement("div");
          lane.className = "cro-structure__lane";
          barsInLane.forEach((bar) => lane.appendChild(makeBar(bar, col)));
          lanes.appendChild(lane);
        });
        stack.appendChild(lanes);
        pending = null;
      };

      col.bars.forEach((bar) => {
        const info = bar.deckNode ? PARALLEL_LOOKUP.get(bar.deckNode) : null;
        if (!info) { flush(); stack.appendChild(makeBar(bar, col)); return; }
        if (!pending || pending.group !== info.group) {
          flush();
          pending = { group: info.group, lanes: Array.from({ length: info.laneCount }, () => []) };
        }
        pending.lanes[info.lane].push(bar);
      });
      flush();

      wrap.append(name, stack);
      container.appendChild(wrap);
    });
  }

  /* ══ 渲染：MoE 专家面板（第 5 项）════════════════════════════════════════
     共享专家 SE0…（始终激活，不参与路由）+ 路由专家按 EP rank 分组，
     每组 routedExpert / ep 个专家。分组与成员全部由 topology.epRanks 派生，
     改 Routed Expert / EP 立即重建。 */
  function renderMoe(sharedHost, routedHost, topology, emit) {
    const { counts, epRanks, hasMoe } = topology;
    if (sharedHost) sharedHost.innerHTML = "";
    if (routedHost) routedHost.innerHTML = "";
    // 稠密模型（如 Qwen2-7B）没有 MoE：整个 MoE 区被 CSS 的 .is-no-moe 隐去，
    // 这里只需清空，不必渲染「1 个专家」这类无意义的退化态。
    if (!hasMoe) return;

    if (sharedHost) {
      if (counts.sharedExpert > 0) {
        for (let i = 0; i < counts.sharedExpert; i += 1) {
          const chip = document.createElement("button");
          chip.type = "button";
          chip.className = "cro-expert cro-expert--shared";
          chip.dataset.shared = String(i);
          chip.dataset.op = "mlp";           // 与结构条 shared_expert bar 同色
          chip.textContent = `SE${i}`;
          chip.dataset.tip = `共享专家 SE${i} · 每个 token 都经过，不参与 top-${counts.topK} 路由`;
          chip.setAttribute("aria-label", chip.title);
          chip.addEventListener("click", () => emit({
            kind: "sharedExpert", shared: i, deckNode: "shared_expert",
          }));
          sharedHost.appendChild(chip);
        }
      } else {
        const empty = document.createElement("span");
        empty.className = "cro-empty";
        empty.textContent = "无共享专家";
        sharedHost.appendChild(empty);
      }
    }

    if (!routedHost) return;
    if (!epRanks.length || !counts.expertsPerEpRank) {
      const empty = document.createElement("span");
      empty.className = "cro-empty";
      empty.textContent = `路由专家 ${counts.routedExpert} 无法均分到 EP ${counts.ep}`;
      routedHost.appendChild(empty);
      return;
    }

    epRanks.forEach((entry) => {
      const group = document.createElement("div");
      group.className = "cro-moe-group";
      group.dataset.epRank = String(entry.epRank);

      /* 整张卡片都是「选中这个 EP 组」的热区 —— 组名那几个字太小，点不中。
         专家胶囊有自己的 kind:"expert"，让它们的 click 冒到这里就会被这一组
         盖掉，所以命中 .cro-expert 时直接放行。组名按钮不再单独挂 listener，
         它的 click 冒上来走同一条路径，键盘可达性照旧由它承担。 */
      group.dataset.tip = `EP rank ${entry.epRank} · 持有专家 E${entry.lo}~E${entry.hi}（${entry.experts.length} 个）`;
      group.addEventListener("click", (event) => {
        if (event.target.closest(".cro-expert")) return;
        emit({ kind: "epRank", epRank: entry.epRank, experts: entry.experts, deckNode: "expert_pool" });
      });

      const name = document.createElement("button");
      name.type = "button";
      name.className = "cro-moe-group__name";
      name.textContent = `EP${entry.epRank}`;
      name.setAttribute("aria-label", group.title);

      const experts = document.createElement("div");
      experts.className = "cro-moe-group__experts";
      entry.experts.forEach((e) => {
        const dot = document.createElement("button");
        dot.type = "button";
        dot.className = "cro-expert";
        dot.dataset.expert = String(e);
        dot.dataset.epRank = String(entry.epRank);
        dot.dataset.op = "moe";            // 与结构条 expert_pool bar 同色
        dot.textContent = `E${e}`;
        dot.dataset.tip = `路由专家 E${e} · 驻留 EP rank ${entry.epRank}`;
        dot.setAttribute("aria-label", dot.title);
        dot.addEventListener("click", () => emit({
          kind: "expert", expert: e, epRank: entry.epRank, deckNode: "expert_pool",
        }));
        experts.appendChild(dot);
      });

      group.append(name, experts);
      routedHost.appendChild(group);
    });
  }

  /* ══ 渲染：集群图（第 6 项）══════════════════════════════════════════════
     完全参数化，不再是 training-run-twin.js 里写死的 DP4×8行×64列。
     几何直接来自 rank 编址 r = s·(EDP·EP·TP·CP) + d·(EP·TP·CP) + p·(TP·CP) + inner：

        列组 = PP stage              （pp 个 Stage 块左右并排）
        列   = 块内 EP rank          （每块 ep 列）
        行   = 模型副本 × tp × cp    （最左一块带 EDP0…EDPn 标签，见 dAxisName）
        格   = 1 个 rank，总数 = edp·pp·ep·tp·cp = Total Rank

     ⚠️ 行数走的是 EDP 不是表单里的 DP：切出档下 DP 里已经含了 EP 组
     （EDP = DP/EP），拿 DP 当行数会画出 world 的 EP 倍。正交档 EDP ≡ DP。

     ⚠️ 默认 4 块 × 64 列 = 256 列，要在不横向滚动的前提下全部显示完，
     所以格间距必须为 0、列轨必须是 minmax(0, 1fr)（可无限收缩）。
     格宽会小到 2~3px，此时 inset 描边会把格子填满，故静息态改用背景填充。
     格高由 CSS 显式给定，与宽度解耦。

     复用 training-run-twin.css 的 .twin-heat / .twin-heat-cell /
     .twin-heat-dp-group 视觉，不新造网格样式。不用 .ep-tint-N（EP 列的 8 色
     循环底色）—— 本页格子是描边态，那批底色会透出来变成一片杂色。 */
  const CLUSTER_CELL_CAP = 16384;

  /* ══ 矩阵折几行：由「这一行能给多少高」算，不写死 ══════════════════════════
     每个 DP 在每个 stage 块里折成 epRows × epCols（64 EP → 2×32 / 3×22 / 4×16）。
     行数一多，横向列数就成比例减少、格子变宽，可读性直线上升；代价是占高。
     以前这个数写死为 2，于是矩阵高度成了常量：.cro-board 第 2 行是 fit-content()，
     而 fit-content 只会 ≤ 内容高，永远不会为了填满空间变大 —— 屏幕一高，富余
     高度全被第 1 行的 1fr 吞掉，摊成典型层卡片里的空白，集群图却还是那么矮。
     所以行数必须反过来由可用高度倒推。
     格高跟着行数走：列数减半格子就宽一倍，高度不同步放大就成了扁条。 */
  const CLUSTER_CELL_H = { 1: 4, 2: 4, 3: 6, 4: 8 };

  /* 矩阵可用的高度预算 = 它要装进去的那个视口本身。
     原来是拿板面高度 × 46%/62% 再减去实测的区 chrome 去"估"，估出来的数和
     syncBoardRows() 真正批给 Cluster 那一行的高度是两笔独立的账，对不上就一头
     出滚动条、另一头空一截。现在直接读 .cro-cluster__grid 的 clientHeight：
     两行的高度在 syncBoardRows() 里已经定好，且**只由 Model Architecture 的内容
     需求决定、与矩阵无关**（见那边的注释），所以这里读到的是终局值，不会出现
     "矩阵撑高预算 → 预算再撑高矩阵"的自激。
     留 2px 余量：视口高度是分数值经 clientHeight 圆整来的，贴着铺满可能差半像素
     顶出一条拖不动的滚动条。 */
  function clusterHeightBudget(host) {
    const viewport = host.closest?.(".cro-cluster__grid");
    if (viewport && viewport.clientHeight > 0) return viewport.clientHeight - 2;
    /* 量不到视口的两种情况：首帧还没布局；事件详情的角色卡里另开的那几份矩阵
       （host 不在 .cro-board 里，见 ROLE_DOMAINS 的重建）。退回板面比例估算。 */
    const board = host.closest?.(".cro-board");
    if (!board || !board.clientHeight) return 0;
    const capRatio = board.classList.contains("is-view-single") ? 0.62 : 0.46;
    return board.clientHeight * capRatio - 132;
  }

  /* 一幅矩阵里除了格子本身之外的那些高度（下面统称 chrome），几何按真实 DOM
     逐层算，不塞魔数：
       block  = epRows 行 × 格高，行间 1px（.cro-heat-block 的 gap）
       DP 组  = innerRows 个 block 竖排 + 上下各 2px padding（.cro-heat-dp）
                + 上下各 1.5px 边框（.twin-heat-dp-group，padding 被本页改过
                但 border 没有）→ 每组固定 7px
       整幅   = dp 个组 + 组间 3px（.cro-heat-body 的 gap）+ 上下两条标签行
                与它们各自 4px 的 flex gap，合计约 32px
     pickEpRows 与 verticalCellHeight 用的是同一份账，抽出来只写一处。 */
  function clusterChromeH(dp, innerRows, epRows) {
    return dp * (innerRows * (epRows - 1) + 7) + (dp - 1) * 3 + 32;
  }

  /* 下限恒为 2：那是改动前的行为，量不到高度或实在放不下时不该比原来更差
     （放不下就照旧由 .cro-cluster__grid 内部滚动）。

     只收 ep 能**整除**的行数。折不尽的话末行会缺格：64 EP 折 3 行是
     ceil(64/3)=22 列，3×22=66 > 64，右下角空两格 —— 一个 stage 块的方阵是
     「这一组的 64 个 EP rank」，缺角会读成"这里少了两张卡"，而它只是排版余数。
     宁可退回 2 行（满格、只是格子窄一半）也不要缺角，所以预算够 3 行但不够
     4 行时（ep=64），best 停在 2。 */
  function pickEpRows(host, counts) {
    // d 轴的行数是 EDP 不是 DP（切出档 EP 已在组内，见 derive 的 edp）
    const dp = counts.edp;
    const ep = counts.ep;
    const innerRows = counts.ranksPerEp;
    const floor = Math.min(2, ep);
    const budget = clusterHeightBudget(host);
    if (!(budget > 0)) return floor;

    const heightOf = (rows) => {
      const cell = CLUSTER_CELL_H[rows] || 4;
      return dp * innerRows * rows * cell + clusterChromeH(dp, innerRows, rows);
    };

    let best = floor;
    for (let rows = floor + 1; rows <= Math.min(4, ep); rows += 1) {
      if (ep % rows === 0 && heightOf(rows) <= budget) best = rows;
    }
    return best;
  }

  /* 格高的上下限。rank 格不要求正方形，但要求**不高于自己的宽**（竖着长的格子
     读起来像"条"不像"卡"），所以格宽是它的另一道硬上界，见 syncCellWidth。
     CLUSTER_CELL_H_MAX 只是防止卡数很少时（如 qwen2-7b 默认 dp2×pp4×ep1，一个
     stage 块只有 1×1 格）一格独吞整块预算、涨成一面墙。
     CLUSTER_CELL_H_MIN = 3：再挤也得看得见一格是一格 —— 1~2px 时格子与 1px 描边
     同量级，整片矩阵会糊成一条实心色带，rank 的疏密结构全丢了。 */
  const CLUSTER_CELL_H_MAX = 32;
  const CLUSTER_CELL_H_MIN = 3;

  /* 按 pickEpRows 那份同源的账反解：这个 epRows 下，纵向预算刨掉 chrome 之后
     摊到每一格行上能给多高。pickEpRows 用 CLUSTER_CELL_H 那张保守表只是为了
     决定折几行，真正落到 --cro-cell-h 的是这里算出来的值 —— 折行定下来之后，
     剩余的纵向空间应该全部摊进格子里，而不是留白在矩阵下方。 */
  function verticalCellHeight(host, counts, epRows) {
    const dp = counts.edp;
    const innerRows = counts.ranksPerEp;
    const budget = clusterHeightBudget(host);
    if (!(budget > 0) || dp <= 0 || innerRows <= 0) return CLUSTER_CELL_H_MAX;
    const totalCellRows = dp * innerRows * epRows;
    const available = budget - clusterChromeH(dp, innerRows, epRows);
    if (!(available > 0) || totalCellRows <= 0) return CLUSTER_CELL_H[epRows] || 4;
    return Math.min(CLUSTER_CELL_H_MAX, Math.floor(available / totalCellRows));
  }

  /* 格宽必须整数像素、锁死到每个 block 上——不能再让 CSS Grid 的 1fr 各自取整。
     cellTemplate 原来是 repeat(epCols, minmax(0,1fr))：同一个 block 内 1fr 按列
     各自求值，折成十几二十列时哪怕只差 1px，摊到几像素宽的格子上就是肉眼可见
     的「有的宽有的窄」（列数越多越明显，2 行仅 4px 格高时更甚）。
     用已经排好版的第一个 block 反量出真实可用宽度，连它自己的 column-gap 一起
     读出来（不在 JS 里重复硬编码 --space-2 这类 gap 常量），算出整数像素的轨道
     列表后统一写回所有 block；block 之间原有的 stageTemplate（1fr）留着不动
     ——那是 4 个 stage 块互相分宽度，不是本次要锁的"块内格子互相分宽度"。

     余数不能丢在右边。floor 出来的统一格宽最多浪费 epCols-1 px（16 列时近
     一格半的宽度），全堆在 block 右缘就是一道空隙：格子明明可以再宽一点，却
     让 stage 块的右边空着。所以先取 base = floor(可用宽 / 列数)，再把余下的
     extra 像素按 Bresenham **均匀间隔**地摊给 extra 个列（每列至多 +1px），
     整行正好填满 block。
     和当初 1fr 的区别在"均匀"二字：1fr 是每列各自取整、误差落在哪儿由浏览器
     的累积舍入决定，会出现连着几个窄的再连着几个宽的；这里的 +1px 是等间隔
     插入的，且格子早已不是当年的 4px 细条（现在按格宽做成正方形，动辄十几
     二十像素），1px 的差别摊在上面看不出来，空着的那道缝反而更扎眼。

     格高（--cro-cell-h）同一处一并写：取 base（不含 +1 的那档，保证没有格子
     高过自己的宽）与纵向预算两个上限里更小的那个。

     ⚠️ growHeight:false —— renderCluster 之后的那次补量（refitClusterCells）只许
     格子变宽、不许变高。格高那时已经按终局视口算过一次并铺进 DOM 了，补量时若
     矩阵恰好顶出一条滚动条，重算出来的值反而可能偏大，一涨就真溢出。变矮不会
     溢出，所以只封上界、不封下界。 */
  function syncCellWidth(host, epCols, counts, epRows, { growHeight = true } = {}) {
    if (!(epCols > 0)) return;
    const firstBlock = host.querySelector(".cro-heat-block");
    if (!firstBlock) return;
    const gap = parseFloat(getComputedStyle(firstBlock).columnGap) || 0;
    const inner = Math.floor(firstBlock.clientWidth - gap * (epCols - 1));
    const base = Math.floor(inner / epCols);
    if (!(base > 0)) return;
    const extra = inner - base * epCols;          // 0 … epCols-1
    const tracks = [];
    for (let i = 0; i < epCols; i += 1) {
      // 第 i 列是否吃到 +1：等间隔地插 extra 次，不扎堆在头尾
      const take = Math.floor(((i + 1) * extra) / epCols) - Math.floor((i * extra) / epCols);
      tracks.push(`${base + take}px`);
    }
    const fixedTemplate = tracks.join(" ");
    host.querySelectorAll(".cro-heat-block").forEach((block) => {
      block.style.gridTemplateColumns = fixedTemplate;
    });
    /* 格高的三道闸，从松到紧：
         · 纵向预算摊到每格行上能给多少（verticalCellHeight）；
         · 不高过自己的宽 —— 竖着长的格子读起来像"条"不像"卡"，横向再宽也补不回来，
           所以 base 是硬上界（base 是不含 Bresenham +1 的那档，取窄的一边）；
         · 不低于 CLUSTER_CELL_H_MIN —— 再挤也得看得见一格是一格。
       下限压过上限时以下限为准：宁可矩阵溢出让 .cro-cluster__grid 去滚，也不要
       糊成一条实心色带。 */
    let cellH = Math.min(verticalCellHeight(host, counts, epRows), base);
    if (!growHeight) {
      const now = parseFloat(host.style.getPropertyValue("--cro-cell-h")) || cellH;
      cellH = Math.min(cellH, now);
    }
    host.style.setProperty("--cro-cell-h", `${Math.max(CLUSTER_CELL_H_MIN, cellH)}px`);
  }

  /* ══ Model Architecture / Cluster 两行怎么分：内容各自实测，不用写死的比例 ══
     .cro-board 原来是「第 1 行 minmax(260px,1fr) + 第 2 行 fit-content(46%)」：
     46% 是给 openPangu（46 层 · Dense/MoE 两段 · 64 EP 大矩阵）估的经验值，换成
     qwen2-7b（28 层单 Dense 段 · 默认仅 8 卡）内容量整个反过来——Model
     Architecture 变短、Cluster 里单卡容量的等距图反而要更多高度才撑得开。写死
     的比例只能顾一头，另一头必然「一边挤出滚动条、一边空出一大截」。
     scrollHeight 天然反映"不裁剪会有多高"，不受当前 grid 行高影响（MDN：包含
     因 overflow 而未显示的内容），两块各量一次，按各自实际需要分这一行的高度；
     只有两块之和超出可用高度时，才按「超出 260px 下限的那部分」等比例收缩——
     两块各自的下限不因为对方要得多就被挤没，真收缩到底则由 .cro-board 自身的
     overflow:auto 兜底（滚动可以接受，重叠/挤压不行，与原设计同一条准则）。
     YAML 视图（css 的 .cro-board.is-yaml）另有一套「一行铺满 + 第二行归零」的
     模板，那时 arch/cluster 整块隐藏、量出来的 scrollHeight 没有意义，交还给
     那条 class 规则，不写内联样式。

     ⚠️ 量之前不能只把 grid-template-rows 清空退回静态规则——静态规则第 1 行仍是
     minmax(260px,1fr)，1fr 会把"这一行还剩多少空间"全部塞给 arch 这个网格项：
     网格项默认 align-self:stretch，箱子本身就被撑到那么高，此时量 scrollHeight
     量到的是"这一行给了多少"，不是"内容真正需要多少"——.cro-section--structure
     的 flex:1 1 auto、#croCapacity 的 height:0+min-height:100% 这两处"主动填满
     父容器"的机关（分别见本文件与 config-relation-capacity.css 的注释）会因此
     被吃进读数里，两个模型来回切换只会越垒越大，降不回去。
     量的时候把两行都临时设成 max-content：网格按"内在尺寸"排布轨道时，子项的
     flex-grow 与百分比高度按 CSS 规范一律不计入内在尺寸贡献（因为这轮计算本来
     就是在求"容器该多大"，用还没求出来的答案反过来定输入是循环定义），于是两个
     机关在这一步自动失效，量到的是两块各自的真实内容高度，不必逐个手动去关。

     ⚠️ 两行之和必须恒等于 available，不能"内容够用就不填满"——.cro-region--net
     横跨这两行（grid-row:1/3），整网 3D deck 的可视高度就是这两行之和撑出来的。
     早先按"内容不够就不硬撑"设计，两块都不需要太高时两行之和会小于 available，
     直接后果是大屏（1080p）下整网画布下方空出一大截。所以余量必须找地方落。

     ⚠️ 谁先拿够，取决于谁**不能变形**，而不是谁的读数大：
       · Model Architecture 是**刚性**的 —— 典型层那 5 列（Dense×2 / MoE×44 …）
         有多少根算子条就是多少根，给不够就只能滚动才看得完，那是实打实的信息
         损失；Layer 导航、stepper 行更是一格都压不得（css 里全是 flex:0 0 auto）。
       · Cluster 是**弹性**的 —— 矩阵的格高（--cro-cell-h）是自由变量，同一批
         rank 铺在 200px 里和铺在 400px 里都成立，只是格子胖瘦不同；分到多少就
         按多少铺满（见 clusterHeightBudget 直接读视口）。
     所以次序是：**arch 按实测内容需求拿够，剩下的全给 Cluster**，矩阵再把拿到
     的那格填满。曾经反过来让 Cluster 先拿（因为它当时"少一像素就出滚动条"），
     结果矩阵一涨就把典型层挤出滚动条 —— 那条滚动条的真正病根是取整误差和残留的
     测量姿势，已经在下面分别修掉了，不该拿版面比例去补。

     Cluster 仍留一道天花板（46% / 只看一边档 62%）：arch 的读数是 max-content，
     典型层算子条一多就能吃掉整块板子；而矩阵矮到一定程度就读不出 rank 分布了。
     吃不完的余量退回给 arch —— 结构条长高本来就是 .cro-structure flex:1 的设计
     意图，不算浪费。 */
  const BOARD_ROW_MIN = 260;

  /* Cluster 那一行的真实下限：矩阵按下限格高（CLUSTER_CELL_H_MIN）、按最少折行数
     铺开时要占多高，加上区里矩阵之外那部分（区标题 + 表单行 + 间距，实测）。
     不能沿用 arch 那个 260 的通用下限 —— 格高有 3px 的硬底（低于它整片矩阵糊成
     一条实心色带），行数又不会低于 pickEpRows 的下限，两者一乘就是一个由 dp / tp /
     cp 决定的具体数字，跟 260 没关系。给少了，矩阵压不下去，.cro-cluster__grid
     就在"这一行明明还能再高"的时候先顶出纵向滚动条。 */
  function clusterMinRowH(counts) {
    if (!counts || !(counts.edp > 0)) return BOARD_ROW_MIN;
    const epRows = Math.max(1, Math.min(2, counts.ep));   // 与 pickEpRows 的 floor 同源
    const cellRows = counts.edp * counts.ranksPerEp * epRows;
    const matrix = cellRows * CLUSTER_CELL_H_MIN + clusterChromeH(counts.edp, counts.ranksPerEp, epRows);
    const region = document.querySelector(".cro-region--cluster");
    const viewport = document.querySelector(".cro-cluster__grid");
    let chrome = 132;   // 还没布局时的保守估计
    if (region && viewport && viewport.clientHeight) {
      const measured = region.offsetHeight - viewport.clientHeight;
      if (measured > 0 && measured < region.offsetHeight) chrome = measured;
    }
    return Math.ceil(matrix + chrome);
  }

  function syncBoardRows(counts) {
    const board = document.getElementById("croBoard");
    if (!board) return;
    if (board.classList.contains("is-yaml")) {
      board.style.gridTemplateRows = "";
      return;
    }
    const arch = document.querySelector(".cro-region--arch");
    const cluster = document.querySelector(".cro-region--cluster");
    if (!arch || !cluster) return;

    /* 量之前先记下上一次的结果：下面那句 max-content 是**临时**的测量姿势，
       一旦板子这会儿量不出高度（首帧未布局，或本页启动即进事件详情、.cro-board
       整块 hidden）就必须原样退回去 —— 否则 "max-content max-content" 会以内联
       样式的形式留在板子上，等它重新显示时两行各自按内容全高铺开（arch 那行会
       把 .cro-structure__stack 里所有算子条一次摊平），远超板面，当场滚出条来。 */
    const prevRows = board.style.gridTemplateRows;
    board.style.gridTemplateRows = "max-content max-content";
    /* 可用高度必须**向下**取整到整像素，且不能只信 clientHeight。
       clientHeight 是把真实的分数高度（flex 链一路算下来，989.6px 这种再正常
       不过）四舍五入成的整数，可能比实际大半像素；两行之和照它铺满，就会把
       .cro-board 顶出零点几像素的溢出 —— 浏览器照样给一条滚动条，可拖动量却
       不到 1px，就是"有条但滚不动"的那个恶心现象。
       rect.height 是精确的分数值但不扣横向滚动条，clientHeight 扣了滚动条但被
       圆整过，两者取小再 floor，才是"一定装得下"的整数。 */
    const style = getComputedStyle(board);
    const rowGap = parseFloat(style.rowGap) || 0;
    const paddingV = (parseFloat(style.paddingTop) || 0) + (parseFloat(style.paddingBottom) || 0);
    const borderV = (parseFloat(style.borderTopWidth) || 0) + (parseFloat(style.borderBottomWidth) || 0);
    const innerH = Math.min(board.getBoundingClientRect().height - borderV, board.clientHeight);
    const available = Math.floor(innerH - rowGap - paddingV);
    // 板子还没铺开（首帧/隐藏）：退回上一次的结果，别把测量姿势留在样式里
    if (!(available > 0)) { board.style.gridTemplateRows = prevRows; return; }

    /* 用 rect.height 而不是 scrollHeight：scrollHeight 是**四舍五入**过的整数，
       内容真高 456.4px 时它报 456，照它批下来的行高就比内容矮 0.4px —— 于是
       那半当场冒出一条拖不动的滚动条（刷新即见的那个 1~2px）。
       两行都是 max-content，网格项被拉伸到的正是自己的内在高度，rect.height
       就是精确的内容高，再向上取整才保证一定装得下。 */
    const archNeed = Math.max(BOARD_ROW_MIN, Math.ceil(arch.getBoundingClientRect().height));
    /* Cluster 的下限是算出来的，不是那个通用的 260（见 clusterMinRowH）。
       再封一道 available 的一半：卡数极多时（上万格）它能要到天上去，那种规模
       本来就只能靠 .cro-cluster__grid 内部滚动，不该拿整块板子去填。 */
    const clusterMin = Math.min(clusterMinRowH(counts), Math.round(available * 0.5));
    // Cluster 的天花板：矩阵是弹性的，但不该把整块板子都吃掉（见上方注释）。
    // 天花板低于下限时以下限为准 —— 下限是"不出滚动条"的硬条件。
    const clusterCap = Math.max(
      clusterMin,
      Math.round(available * (board.classList.contains("is-view-single") ? 0.62 : 0.46)),
    );
    let row1;
    let row2;
    if (available < BOARD_ROW_MIN + clusterMin) {
      // 窗口实在太矮，两个下限之和都装不下：各自守住下限，超出的那截交还
      // .cro-board 自身的 overflow:auto 兜底（滚动可以接受，压穿下限不行）。
      row1 = BOARD_ROW_MIN;
      row2 = clusterMin;
    } else {
      // 刚性的那半先拿够，弹性的那半接住余量；余量超过天花板就退回给 arch
      row1 = Math.min(archNeed, available - clusterMin);
      row2 = available - row1;
      if (row2 > clusterCap) { row2 = clusterCap; row1 = available - row2; }
    }
    // 两行都是整数、之和恒等于 available，不再有取整余数顶出零点几像素的溢出
    board.style.gridTemplateRows = `${row1}px ${row2}px`;
  }

  /* ── d 轴口径的两处常驻文案 ──────────────────────────────────────────────
     只改文字，不碰任何几何。之所以要有这两个函数：切出档下表单里的 DP 是 512、
     矩阵左侧标的是 EDP0–7，光把标签改名还不够 —— 得把「512 ÷ 64 = 8」这句
     换算摆在不用悬浮就看得见的地方，否则读者只会认为其中一个数算错了。 */
  function syncClusterAxisNote(topology) {
    const el = document.getElementById("croClusterAxisNote");
    if (!el) return;
    const c = topology.counts;
    const text = dAxisName(c) === "EDP"
      ? `矩阵纵轴 = EDP ${c.edp}（DP ${c.dp} ÷ EP ${c.ep}）· 一行是一个完整模型副本，含全部 ${c.ep} 个 EP rank`
      : `矩阵纵轴 = DP ${c.dp} · 一行是一个完整模型副本`;
    el.textContent = text;
    // 窄窗口下这行会被省略号截断，完整句子留在 title 里
    el.title = text;
  }

  function renderCluster(host, topology, emit) {
    if (!host) return;
    const { counts } = topology;
    const { pp, ep, tp, cp, totalRank } = counts;
    // 矩阵的 d 轴是 EDP：切出档下 DP 里已经含了 EP 组，用 DP 会多画 EP 倍的行
    const dp = counts.edp;
    const innerRows = counts.ranksPerEp;   // tp × cp
    host.innerHTML = "";
    delete host.dataset.epRows;

    if (!topology.valid) {
      const note = document.createElement("span");
      note.className = "cro-empty";
      note.textContent = "配置不自洽，集群网格暂不重建（见上方提示）";
      host.appendChild(note);
      return;
    }
    if (totalRank > CLUSTER_CELL_CAP) {
      const note = document.createElement("span");
      note.className = "cro-empty";
      note.textContent = `${totalRank} 卡超过 ${CLUSTER_CELL_CAP} 格上限，不逐卡绘制`;
      host.appendChild(note);
      return;
    }

    /* 每个 DP 在每个 stage 块里不再挤成一行 ep 个格子，而是折成
       epRows × epCols 的小方阵（默认 64 → 4 行 × 16 列），行主序填。
       总列数 = pp × epCols = 4×16 = 64，总行数 = dp × epRows = 8×4 = 32，
       格子从 2.5px 宽放大到 ~10px，仍然是 2048 格、不横向滚动。
       两级列轨都必须能收缩到 0：任意一级留下限，另一级就会溢出压在隔壁块上。 */
    /* 折几行 / 格子多高：全部由可用高度倒推（见 pickEpRows）。视图档只是通过
       行高封顶影响预算，不再单独判档 —— 「只看整网」与「只看典型 Layer」两档
       预算相同，集群图自然长得一样。
       行数是建 DOM 时定的，纯 CSS 改不动：所以切档（cro:view）与窗口高度变化
       （resize）都要回来重建一次，两处都在文件末尾。dataset 留一份当前值，供
       resize 判断"行数其实没变，别白重建 2048 个格子"。
       事件详情的角色卡里也会调本函数各开一份矩阵，那些 host 不在 .cro-board
       里，clusterHeightBudget 量不到板子，天然回落到 2 行。
       两级列轨都用 1fr：宽度随本列自适应，既不溢出也不需要横向滚动。 */
    const epRows = pickEpRows(host, counts);
    const epCols = Math.ceil(ep / epRows);
    host.dataset.epRows = String(epRows);
    // 先给个粗略初值（格宽还没测出来），appendChild 之后 syncCellWidth 会按
    // 实测格宽把它改成正方形的准确值。
    host.style.setProperty("--cro-cell-h", `${CLUSTER_CELL_H[epRows] || 4}px`);
    const stageTemplate = `repeat(${pp}, minmax(0, 1fr))`;
    const cellTemplate = `repeat(${epCols}, minmax(0, 1fr))`;

    /* ── TP 分片 → 具体哪几张卡 ────────────────────────────────────────────
       编址里 TP 是最内的一维：inner = cpIdx·tp + tpIdx，于是同一个 TP 组的 tp
       张卡全局编号连号（rank, rank+1, … rank+tp-1），优先落在同一节点内 ——
       TP 每层前反向都要 all-reduce 一次激活，是通信最密的一维，必须吃机内互联
       （HCCS/NVLink），把它排在最内是各家框架（Megatron 的 tp-cp-ep-dp-pp 序）
       的一致做法。CP 次之，两者共同占满 ranksPerEp。
       所以分片序号 tpIdx = inner % tp，而 inner 正是集群图里 DP 组内的行序 ——
       每 tp 行走完一轮分片，横向天然成条带，斑马纹按行铺就是客观的分片分布。 */
    const tpShardOf = (inner) => inner % tp;
    const cpIdxOf = (inner) => Math.floor(inner / tp);
    /* 亮度是**间隔**的，不是渐变的：单调递减的斜坡在整片格子上会读成一团渐变，
       看不出"一份一份"的边界；明暗交替才切得出条带。
         偶数份 → 100%（与单 TP 时的高亮同强度）
         奇数份 → 50% / 40% 交替（tp≥4 时两条暗纹也分得出先后）
       最暗一档仍亮于静息态的 45% 中性灰描边（白 40% ≠ 灰 45%），"暗的那条也是
       被点亮的"这层意思不能丢。 */
    const tpFade = (shard) => {
      if (tp <= 1 || shard % 2 === 0) return 100;
      return shard % 4 === 1 ? 50 : 40;
    };

    // ── 上：PP stage 标签，与下方 stage 块同列 ──
    const stageLabels = document.createElement("div");
    stageLabels.className = "twin-heat-pp-labels";
    stageLabels.style.gridTemplateColumns = stageTemplate;
    for (let s = 0; s < pp; s += 1) {
      const label = document.createElement("span");
      label.textContent = `Stage${s}`;
      stageLabels.appendChild(label);
    }
    host.appendChild(stageLabels);

    /* ── 中：每个模型副本一个横贯全宽的分组（左侧带 d 轴标签），
          组内是 pp 个 stage 小方阵并排。
          标签写 EDP 还是 DP 由口径定（见 dAxisName）：切出档下这一行是
          「DP/EP 组」，与表单里那个 DP 512 不是同一个量，重名会直接读错。 ── */
    const dName = dAxisName(counts);
    // EDP 比 DP 多一个字符，::before 的左侧留白要跟着放宽（见 .cro-heat[data-d-axis]）
    host.dataset.dAxis = dName.toLowerCase();
    /* 换算式进每一个格子的悬浮提示：表单上写着 DP 512、这里标着 EDP0–7，
       两个数对不上是本页最容易被当成算错的一处，光改名不够，得把桥给出来。 */
    const dTip = dName === "EDP" ? `\nEDP = DP ${counts.dp} ÷ EP ${ep} = ${dp}` : "";
    /* 格子提示的最后一行：这张卡到底背着几个专家。EP 列号只说明它属于哪个专家组，
       换算成「几个」还要再除一次，而这正是看矩阵时最想知道的那个数。
       专家只存在于 MoE 层，所以整段都是 dense 层的 stage 上不写这一行
       （稠密模型全程没有，naturally 也就整块不出现）。 */
    const stageHasMoe = topology.stages.map((entry) =>
      topology.layers.slice(entry.lo, entry.hi + 1).some((layer) => layer.ffn === "moe"));
    const body = document.createElement("div");
    body.className = "cro-heat-body";

    for (let d = 0; d < dp; d += 1) {
      const group = document.createElement("div");
      group.className = "twin-heat-dp-group cro-heat-dp";
      group.style.gridTemplateColumns = stageTemplate;
      group.dataset.dp = String(d);
      group.dataset.dpLabel = `${dName}${d}`;
      group.setAttribute("role", "rowgroup");
      group.setAttribute("aria-label", `${dName}${d} 副本`);

      for (let inner = 0; inner < innerRows; inner += 1) {
        for (let s = 0; s < pp; s += 1) {
          const block = document.createElement("div");
          block.className = "cro-heat-block";
          block.style.gridTemplateColumns = cellTemplate;
          block.dataset.dp = String(d);
          block.dataset.stage = String(s);
          block.setAttribute("role", "row");
          block.setAttribute("aria-label", `Stage${s} · ${dName}${d} · ${ep} 个 EP rank`);

          for (let p = 0; p < ep; p += 1) {
            const rank = topology.rankOf(s, d, p, inner);
            const node = topology.nodeOfRank(rank);
            const cell = document.createElement("div");
            // 不带 ep-tint-N：那是 8 色循环的 EP 列底色，格子改描边后会透出来
            // 变成「五颜六色」，本页不用它编码任何信息。
            cell.className = "twin-heat-cell";
            cell.dataset.rank = String(rank);
            cell.dataset.stage = String(s);
            cell.dataset.dp = String(d);
            cell.dataset.ep = String(p);
            const shard = tpShardOf(inner);
            const cpIdx = cpIdxOf(inner);
            cell.dataset.tp = String(shard);
            cell.dataset.cp = String(cpIdx);
            cell.dataset.node = String(node);
            // TP/CP 展开时把这张卡在最内两维里的位置写进提示：光看格子只知道
            // 「被点亮了」，知道是第几份权重才谈得上定位。
            const shardTip = tp > 1 ? `\nTP 分片 ${shard + 1}/${tp}（持有该层权重的 1/${tp}）` : "";
            const cpTip = cp > 1 ? `\nCP 分片 ${cpIdx + 1}/${cp}` : "";
            const epEntry = topology.epRanks[p];
            const sharedTip = counts.sharedExpert
              ? ` + ${counts.sharedExpert} 个共享专家（每卡一份）` : "";
            const expertTip = stageHasMoe[s] && epEntry && counts.expertsPerEpRank
              ? `\n当前 rank 有 ${counts.expertsPerEpRank} 个路由专家`
                + `（每个 MoE 层各一份，编号 ${epEntry.lo}–${epEntry.hi}）${sharedTip}`
              : "";
            cell.dataset.tip = `rank ${rank}\nStage${s} · ${dName}${d} · EP${p} · TP${shard} · CP${cpIdx}${shardTip}${cpTip}${dTip}\nNode ${node}${expertTip}`;
            if (tp > 1) {
              cell.dataset.tpShard = String(shard);
              // 高亮亮度分档由 CSS 读这枚变量（见 .twin-heat-cell.is-related）。
              // tp=1 时不写，回落到 100% —— 单 TP 的观感与改动前完全一致。
              cell.style.setProperty("--cro-tp-fade", `${tpFade(shard).toFixed(1)}%`);
            }
            // 2048 个格子不能各占一个 Tab 站；用 roving tabindex + 方向键在网格内移动
            cell.setAttribute("role", "gridcell");
            cell.setAttribute("tabindex", rank === 0 ? "0" : "-1");
            cell.setAttribute("aria-label",
              `rank ${rank}，Stage${s}、${dName}${d}、EP${p}、TP${shard}、CP${cpIdx}，节点 ${node}`);
            cell.addEventListener("click", () => emit({
              kind: "rank", rank, stage: s, dpIdx: d, epRank: p,
              tpIdx: shard, cpIdx, node,
            }));
            block.appendChild(cell);
          }
          group.appendChild(block);
        }
      }
      body.appendChild(group);
    }
    host.appendChild(body);
    // block 已经挂到文档上、stageTemplate 分好了每个 stage 的宽度，这时才量得到
    // 真实可用宽度，把 cellTemplate 从 1fr 换成锁死的整数像素值
    syncCellWidth(host, epCols, counts, epRows);

    /* ── 下：每个 stage 块底部标一次 EP 覆盖范围。
          EP 在块内是折行排布的（4 行 × 16 列），列位置不再一一对应某个 EP 序号，
          所以这里不逐列标 EP0/EP8，只给区间，精确值走格子的悬浮提示。 ── */
    const epLabels = document.createElement("div");
    epLabels.className = "cro-heat-ep-labels";
    epLabels.style.gridTemplateColumns = stageTemplate;
    for (let s = 0; s < pp; s += 1) {
      const caption = document.createElement("span");
      caption.textContent = epRows > 1 ? `EP0–EP${ep - 1}（${epRows}×${epCols}）` : `EP0–EP${ep - 1}`;
      epLabels.appendChild(caption);
    }
    host.appendChild(epLabels);

    enableGridKeyboard(host, epCols, emit);
  }

  /* 集群网格的键盘导航：整张网格只占 1 个 Tab 站，进去后用方向键在
     rank 之间移动（左右 = EP rank，上下 = 跨 stage / DP 的同一列），
     Enter/Space 触发选择。 */
  function enableGridKeyboard(host, cols, emit) {
    const cells = Array.from(host.querySelectorAll(".twin-heat-cell"));
    if (!cells.length) return;
    const rows = Math.ceil(cells.length / cols);

    const focusAt = (index) => {
      const next = cells[Math.max(0, Math.min(cells.length - 1, index))];
      if (!next) return;
      cells.forEach((cell) => cell.setAttribute("tabindex", "-1"));
      next.setAttribute("tabindex", "0");
      next.focus();
    };

    host.addEventListener("keydown", (event) => {
      const current = cells.indexOf(event.target);
      if (current < 0) return;
      const row = Math.floor(current / cols);
      const col = current % cols;
      let next = null;
      switch (event.key) {
        case "ArrowLeft": next = current - 1; break;
        case "ArrowRight": next = current + 1; break;
        case "ArrowUp": next = current - cols; break;
        case "ArrowDown": next = current + cols; break;
        case "Home": next = row * cols; break;
        case "End": next = row * cols + cols - 1; break;
        case "PageUp": next = col; break;
        case "PageDown": next = (rows - 1) * cols + col; break;
        case "Enter": case " ":
          event.preventDefault();
          event.target.click();
          return;
        default: return;
      }
      event.preventDefault();
      focusAt(next);
    });
  }

  /* ══ 关系引擎（第 7 项）══════════════════════════════════════════════════
     把任意一个视图里的点击，解析成「整网 / Layer / 专家 / 集群」四者的
     全量关系集。全部走 topology 的确定性查询，不猜、不缓存。
     解析结果是无向的：从哪个视图点进来，其余三个视图都被点亮，所以
     layer ↔ 专家 ↔ rank ↔ 算子 是双向互查的。 */
  function resolveRelation(topology, payload) {
    const { counts } = topology;
    const columns = activeColumns(topology);
    const rel = {
      primary: payload,
      layers: new Set(), stages: new Set(),
      segment: null, bar: null, unit: null, deckNode: payload.deckNode || null, deckLayer: null,
      // wholeColumn：点了典型层的名字/底板 = 选中整列。锚点是整块底板、整网侧是整张
      // 层卡，而不是列里某个算子；关系集覆盖整段的层/专家/rank。
      wholeColumn: Boolean(payload.wholeColumn),
      // deckStatic：目标算子在 deck 的 input / output 静态段里（Emb / Final Norm /
      // LM Head / MTP…），不属于任何一张层卡片。selectNode(id, layer) 会把查找
      // 限死在那张层卡内，静态节点永远找不到 —— 于是既选不中也连不出线。
      deckStatic: false,
      // 一次选择往往横跨多列（一个 rank 压住它那段 PP 的 Dense+MoE+端点列），
      // 单值 segment 只够记「点了哪一列」，列级高亮/去色必须看这个集合。
      segments: new Set(), units: new Set(), staticNodes: new Set(),
      experts: new Set(), epRanks: new Set(), shared: new Set(),
      ranks: new Set(), nodes: [],
      labels: {},
    };
    const moeLayers = topology.layers.filter((l) => l.ffn === "moe").map((l) => l.index);
    const addRanks = (list) => list.forEach((r) => rel.ranks.add(r));
    const addLayers = (list) => list.forEach((l) => { rel.layers.add(l); rel.stages.add(topology.stageOfLayer(l)); });
    const allRoutedExperts = () => { for (let e = 0; e < counts.routedExpert; e += 1) rel.experts.add(e); };
    const allEpRanks = () => { for (let p = 0; p < counts.ep; p += 1) rel.epRanks.add(p); };
    const allShared = () => { for (let i = 0; i < counts.sharedExpert; i += 1) rel.shared.add(i); };
    // 端点列（Emb / Norm / Head）驻留的 PP stage
    const anchorStage = (col) => (col.stageAnchor === "first" ? 0 : Math.max(0, counts.pp - 1));
    /* 结构对象（层、典型层算子、Emb / Norm / Head 端点）一律查**全部 DP/EDP**：
       问的是「这个结构对象落在哪些卡上」，答案本就横跨所有模型副本。
       只有明确带了 dpIdx 的 payload（点某张 rank 卡）才收窄到那一个副本。
       不区分「分片 / 副本」：Dense 层与 Emb / Norm / Head 在 EP 维度上确实是
       副本，但副本也是"这张卡上有这一层"，照样要亮 —— 只亮一份会读成"这个 DP
       里其余的卡不含这一层"，那是错的。副本结构本身由斑马纹表达：同一亮度的
       那批卡持有同一份 TP 切片，彼此互为副本。 */
    const stageRanks = (stage) => (Number.isFinite(payload.dpIdx)
      ? topology.ranksOfStageInDp(stage, payload.dpIdx)
      : topology.ranksOfStage(stage));
    // 整段 stage 被选中（点 PP 标签 / 点某张卡）时，端点列也在这段流水线上；
    // 点某个算子条时不算 —— MoE 算子横跨全部 stage，不该把 Norm/Head 也拖亮。
    let wholeStage = false;

    switch (payload.kind) {
      case "layer": {
        addLayers([payload.layer]);
        rel.deckLayer = payload.layer;
        const layer = topology.layers[payload.layer];
        rel.segment = layer.ffn;
        if (layer.ffn === "moe") { allRoutedExperts(); allEpRanks(); allShared(); }
        addRanks(stageRanks(topology.stageOfLayer(payload.layer)));
        break;
      }
      case "stage": {
        wholeStage = true;
        const entry = topology.stages[payload.stage];
        if (entry) {
          const list = [];
          for (let l = entry.lo; l <= entry.hi; l += 1) list.push(l);
          addLayers(list);
          rel.deckLayer = entry.lo;   // 整网转到这段流水线的首层
          if (list.some((l) => topology.layers[l].ffn === "moe")) { allRoutedExperts(); allEpRanks(); allShared(); }
        }
        addRanks(topology.ranksOfStage(payload.stage));
        break;
      }
      case "segment": {
        const col = columns.find((c) => c.id === payload.segment);
        rel.segment = payload.segment;
        // 整列点击不落到单个算子条：rel.bar 留空，arch 锚点走整块底板；deckNode 也
        // 留空，net 锚点退回整张层卡（见 collectAnchors）。单算子点击才设 rel.bar。
        if (!payload.wholeColumn) rel.bar = { segment: payload.segment, bar: payload.bar };
        if (col && col.layers.length) {
          // 已经选中某一层时，点算子条只收敛到那一层（select.png 的
          //「EP Combine in Layer 3」），否则覆盖整列
          const scoped = Number.isFinite(payload.scopeLayer) && col.layers.includes(payload.scopeLayer);
          addLayers(scoped ? [payload.scopeLayer] : col.layers);
          // preferLayer：从整网图点进来时停在用户正看着的那一层，别把 deck
          // 甩到该列中间去（关系集仍是整列，只是取哪一层做展示锚点）
          const prefer = Number.isFinite(payload.preferLayer) && col.layers.includes(payload.preferLayer)
            ? payload.preferLayer
            : col.layers[Math.floor(col.layers.length / 2)];
          rel.deckLayer = scoped ? payload.scopeLayer : prefer;
          rel.stages.forEach((s) => addRanks(stageRanks(s)));
        } else if (col && col.stageAnchor) {
          // Emb / Norm / Head：没有层，但驻留在首/末 PP stage，按 stage 接回集群
          const stage = anchorStage(col);
          const entry = topology.stages[stage];
          rel.stages.add(stage);
          rel.unit = col.id;
          // Emb / Final Norm / LM Head / MTP 都画在 deck 的静态段里，
          // deckLayer 只用来把 deck 转到流水线对应的一端，不能拿去限定查找范围
          rel.deckStatic = true;
          if (entry) rel.deckLayer = col.stageAnchor === "first" ? entry.lo : entry.hi;
          // 端点列（Emb/Norm/Head）就一个概念块，整列点击时用它的代表算子做 deck 静态
          // 节点，让 net 侧仍能连到 deck 里的 embedding / final_norm / lm_head。
          if (payload.wholeColumn && col.bars[0]) rel.deckNode = col.bars[0].deckNode;
          addRanks(stageRanks(stage));
        }
        if (payload.wholeColumn && col && col.id === "moe") {
          // 整列点 MoE：这一整段 MoE 典型层横跨全部路由专家 + 共享专家 + 全部 EP rank
          allRoutedExperts(); allEpRanks(); allShared();
        } else if (payload.experts === "routed") { allRoutedExperts(); allEpRanks(); }
        else if (payload.experts === "shared") allShared();
        else if (col && col.id === "moe") {
          // MoE 列里其余算子（Attn / 各 Norm / 残差 Add）不落在某几个专家身上，
          // 但整段 MoE 块是横跨所有 EP rank 的。这里至少把 EP 分组接上，否则
          // MoE 区一个 is-related 都没有，collectAnchors().moe 为 null，
          // drawRelationLinks 会整条跳过，表现为「点整网/典型层从不连 MoE」。
          allEpRanks();
        }
        break;
      }
      case "expert":
      case "epRank": {
        const list = payload.kind === "expert" ? [payload.expert] : (payload.experts || []);
        list.forEach((e) => rel.experts.add(e));
        rel.epRanks.add(payload.epRank);
        rel.segment = "moe";
        rel.bar = { segment: "moe", bar: "expert_pool" };
        // 【全展开】一个路由槽位（专家编号 e）在**每个 MoE 层**都有一份实例（各层权重
        // 独立、互不相干，只共享编号与「编号→EP rank」的分片公式）；它的 EP 组在**每个
        // PP stage** 内都占一块 rank。点专家就把这个编号涉及的全部 MoE 层 + 全部 stage 的
        // 该 EP 组 rank（× DP 副本）一并连上，让「这个编号散布在哪里」一眼看全。连线侧
        // 会按 stage 拆成多条（见 drawRelationLinks），而非缩成一个巨框。
        addLayers(moeLayers);
        rel.deckLayer = moeLayers[Math.floor(moeLayers.length / 2)];
        rel.stages.forEach((s) => addRanks(topology.ranksOfEpRankInStage(s, payload.epRank)));
        break;
      }
      case "sharedExpert": {
        rel.shared.add(payload.shared);
        rel.segment = "moe";
        rel.bar = { segment: "moe", bar: "shared_expert" };
        // 共享专家同样每个 MoE 层各一份，每个 token 都过 → 连上全部 MoE 层 + 每个 stage
        // 的全部 rank。
        addLayers(moeLayers);
        rel.deckLayer = moeLayers[Math.floor(moeLayers.length / 2)];
        rel.stages.forEach((s) => addRanks(topology.ranksOfStage(s)));
        break;
      }
      case "rank": {
        wholeStage = true;
        const co = topology.coordsOfRank(payload.rank);
        rel.stages.add(co.stage);
        const entry = topology.stages[co.stage];
        if (entry) for (let l = entry.lo; l <= entry.hi; l += 1) rel.layers.add(l);
        rel.epRanks.add(co.epIdx);
        topology.expertsOfEpRank(co.epIdx).forEach((e) => rel.experts.add(e));
        allShared();
        rel.ranks.add(payload.rank);
        // 一张卡不属于某一列典型层：它持有的是自己那个 PP stage 的整段层
        // （Dense + MoE 都算），相关列由下面按 rel.layers 派生，这里不预设。
        // 整网 deck 转到这段流水线的首层，否则点末段的卡、图还停在中间层上。
        if (entry) rel.deckLayer = entry.lo;
        break;
      }
      default: break;
    }

    /* 关系覆盖到哪几列典型层：凡有层落进关系集的列都算相关；端点列没有层，
       按它驻留的 PP stage 判定，且只在整段 stage 被选中时才接上。
       以前这里只有单值 rel.segment，点一个 rank 无论压住哪几列都写死 "moe"，
       Dense / Norm / Head 既不高亮也不去色 —— 「点 rank 只连 MoE」就是这个。 */
    columns.forEach((col) => {
      if (col.layers.length) {
        if (col.layers.some((l) => rel.layers.has(l))) rel.segments.add(col.id);
      } else if (col.stageAnchor && wholeStage && rel.stages.has(anchorStage(col))) {
        rel.segments.add(col.id);
        rel.units.add(col.id);
      }
    });
    // 端点列在整网 deck 里对应静态段（input / output）的那批节点。层内节点靠
    // 层号判定即可，静态段没有层号，只能按 id 收一份名单给去色用。
    columns.forEach((col) => {
      if (col.layers.length || !rel.segments.has(col.id)) return;
      col.bars.forEach((bar) => { if (bar.deckNode) rel.staticNodes.add(bar.deckNode); });
    });
    if (rel.bar && rel.segment) rel.segments.add(rel.segment);
    // 整列点击没有 rel.bar，但被点的这一列本身当然在关系集里（端点列 col.layers 为空，
    // 上面按层号那轮不会加进来，这里补上，否则整列高亮/去色都读不到自己）。
    if (rel.wholeColumn && rel.segment) rel.segments.add(rel.segment);
    if (rel.unit) rel.units.add(rel.unit);

    rel.nodes = topology.nodesOfRanks(Array.from(rel.ranks));
    rel.labels = relationLabels(topology, rel, columns);
    return rel;
  }

  function summarizeRuns(values) {
    const sorted = Array.from(values).sort((a, b) => a - b);
    const runs = [];
    sorted.forEach((v) => {
      const last = runs[runs.length - 1];
      if (last && v === last[1] + 1) last[1] = v;
      else runs.push([v, v]);
    });
    return runs;
  }

  function formatRuns(values, prefix, maxRuns = 3) {
    const runs = summarizeRuns(values);
    if (!runs.length) return "";
    const shown = runs.slice(0, maxRuns)
      .map(([a, b]) => (a === b ? `${prefix}${a}` : `${prefix}${a}~${b}`))
      .join("+");
    return runs.length > maxRuns ? `${shown} 等 ${runs.length} 段` : shown;
  }

  function relationLabels(topology, rel, columns) {
    const labels = {};
    const c = topology.counts;

    // 整列点击：主标签直接报这一整个典型层的名字（如「Dense x2（L0~L1）」/「MoE
    // x44（L2~L45）」/「Emb」），表示连的是整块而非某个算子。
    if (rel.wholeColumn) {
      const col = columns.find((x) => x.id === rel.segment);
      if (col) {
        labels.arch = col.layers.length || rel.stages.size !== 1
          ? col.name
          : `${col.name} · PP${Array.from(rel.stages)[0]}`;
      }
    } else if (rel.bar) {
      const col = columns.find((x) => x.id === rel.bar.segment);
      const barDef = col && col.bars.find((b) => b.id === rel.bar.bar);
      const name = barDef ? barDef.label : rel.bar.bar;
      if (rel.layers.size === 1) {
        const only = Array.from(rel.layers)[0];
        // 单层定位一律带上 PP 段，把「这个算子/专家究竟落在哪一段流水线」写死在标签上
        labels.arch = `${name} in Layer ${only} · PP${topology.stageOfLayer(only)}`;
      } else if (rel.layers.size) {
        labels.arch = `${name} · ${formatRuns(rel.layers, "L", 1)}`;
      } else {
        // Emb / Norm / Head 不是层，只有 PP 归属可报
        labels.arch = rel.stages.size === 1 ? `${name} · PP${Array.from(rel.stages)[0]}` : name;
      }
    } else if (rel.layers.size === 1) {
      const l = Array.from(rel.layers)[0];
      const layer = topology.layers[l];
      labels.arch = `Layer ${l} · PP${layer.stage} · ${layer.ffn === "dense" ? "Dense" : "MoE"} · ${layer.attention.toUpperCase()}`;
    } else if (rel.stages.size === 1) {
      labels.arch = `PP${Array.from(rel.stages)[0]} · ${formatRuns(rel.layers, "L", 1)}`;
    }

    // 专家 / EP 组 / 共享专家：主标签点明「该编号在全部相关 MoE 层各有一份」，既表达
    // 全展开的分布范围，又不误导成「同一个专家横跨各层」（各层是独立权重实例）。
    const pk = rel.primary && rel.primary.kind;
    if ((pk === "expert" || pk === "epRank" || pk === "sharedExpert") && rel.layers.size) {
      const who = pk === "expert" ? `E${rel.primary.expert}`
        : pk === "sharedExpert" ? `SE${rel.primary.shared}`
        : `EP${rel.primary.epRank}`;
      labels.arch = `${who} · ${formatRuns(rel.layers, "L", 1)} 各一份`;
    }

    // MoE：EP 组 + 组内专家区间，专家全量时改用摘要，避免拼出 64 段
    const epParts = [];
    if (rel.epRanks.size && rel.epRanks.size < c.ep) {
      Array.from(rel.epRanks).sort((a, b) => a - b).slice(0, 3).forEach((p) => {
        const own = topology.expertsOfEpRank(p).filter((e) => rel.experts.has(e));
        epParts.push(own.length ? `EP${p}(${formatRuns(own, "E", 1)})` : `EP${p}`);
      });
      if (rel.epRanks.size > 3) epParts.push(`等 ${rel.epRanks.size} 个 EP rank`);
    } else if (rel.epRanks.size) {
      // 只牵连到 EP 分组、没点到具体专家时（MoE 列里的 Attn / Norm / Add），
      // 不能报「N 专家」，那是没被点亮的
      epParts.push(rel.experts.size
        ? `全部 ${c.ep} 个 EP rank · ${c.routedExpert} 专家`
        : `全部 ${c.ep} 个 EP rank`);
    }
    if (rel.shared.size) epParts.push(rel.shared.size === 1 ? "Share Expert" : `Share Expert ×${rel.shared.size}`);
    if (epParts.length) labels.moe = epParts.join("+");

    // 集群：节点区间 + 卡数。节点常常是等距散布（同一 EP rank 的 DP 副本每隔
    // ranksPerNode 一跳），逐段列会拼成「0+8+16 等 32 段」这种噪音，故退化成
    // 首尾 + 个数。
    if (rel.ranks.size) {
      const runs = summarizeRuns(rel.nodes);
      const span = runs.length <= 2
        ? formatRuns(rel.nodes, "", 2)
        : `${rel.nodes[0]}…${rel.nodes[rel.nodes.length - 1]}（${rel.nodes.length} 个）`;
      /* 光看卡数容易读成"每张卡各存一份完整层权重"，所以 TP≥2 时补两行口径。
         份额分两种：Attn / 共享专家 / Dense MLP 是纯 TP 切，单卡 1/tp；MoE 层的
         大头是路由专家，先被 EP 切成 1/ep 再被 TP 切一刀，单卡实际持有 1/(tp×ep)
         —— 只写 1/tp 会把真实份额高估两个数量级。
         不解释 EP 对 Dense / Emb / Norm / Head 是副本这件事：这几段本来就不涉及
         专家，是业务常识，写在气泡里是噪音。
         卡片挂在连线中点上，行不能长：一行控制在 ~24 个汉字内。 */
      labels.cluster = `Node ${span} · ${rel.ranks.size} 卡`;
      if (pk === "rank" && Number.isFinite(rel.primary.rank)) {
        const co = topology.coordsOfRank(rel.primary.rank);
        labels.cluster = coordLine(topology, co);
      }
      // 只给「结构对象 → 卡」这类选择加口径说明：点 rank / 专家问的是别的事
      const structural = rel.primary
        && (rel.primary.kind === "layer" || rel.primary.kind === "segment");
      if (structural && c.tp > 1) {
        const withMoe = Array.from(rel.layers).some((l) => topology.layers[l]?.ffn === "moe");
        labels.cluster = [
          labels.cluster,
          `明暗相间 = TP 切出的 ${c.tp} 份，每条一份`,
          withMoe
            ? `单卡持有 Attn/共享专家 1/${c.tp} · 路由专家 1/(${c.tp}×${c.ep})`
            : `单卡持有 1/${c.tp}（只按 TP 切）`,
        ];
      }
    }
    return labels;
  }

  /* ══ 关系连线层（第 7 项）════════════════════════════════════════════════
     以被点中的那个视图为 hub，向其余视图各拉一条曲线，中点挂标签。
     用 viewport 坐标直接画在 position:fixed 的 SVG 上，滚动/缩放时重画。 */
  const SVG_NS = "http://www.w3.org/2000/svg";

  /* 一组元素的并集包围盒。关系集经常是「一整组」——某层的全部 rank、某个
     EP rank 的全部专家、某列的全部算子 —— 这时连线应该接到整组，而不是挑
     组里的某一个元素。 */
  function unionRect(elements) {
    let left = Infinity; let top = Infinity; let right = -Infinity; let bottom = -Infinity;
    elements.forEach((el) => {
      const r = el.getBoundingClientRect();
      if (!r.width && !r.height) return;
      left = Math.min(left, r.left); top = Math.min(top, r.top);
      right = Math.max(right, r.right); bottom = Math.max(bottom, r.bottom);
    });
    if (!Number.isFinite(left)) return null;
    return { left, top, right, bottom, width: right - left, height: bottom - top };
  }

  /* 锚点夹回宿主的可视矩形。
     元素被自己所在的滚动/裁剪容器裁掉时 getBoundingClientRect 照样返回有效
     几何，只是那个位置压根不在屏幕上 —— 连线就从可视区里一头扎出去，看着
     就是「高亮有了、线没有」。整网 deck 最典型：正视图下 input / output 静态段
     分别落在层卡上下 700px / 520px 处（Emb、Final Norm、LM Head、MTP 全在
     里面），几乎必定在 deck 视口之外。夹回之后线终止在区域边界上，指向正确。 */
  function clampRectTo(rect, host) {
    if (!rect || !host) return rect;
    const box = host.getBoundingClientRect();
    if (!box.width && !box.height) return rect;
    const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
    const left = clamp(rect.left, box.left, box.right);
    const right = clamp(rect.right, box.left, box.right);
    const top = clamp(rect.top, box.top, box.bottom);
    const bottom = clamp(rect.bottom, box.top, box.bottom);
    return { left, top, right, bottom, width: right - left, height: bottom - top };
  }

  /* 每个视图返回 { rect, group }：group=true 表示这是一整组，
     连线端点会落在整组包围盒的边上，并额外画一圈虚线框把范围圈出来。
     第三个参数是该视图的可视宿主，锚点一律夹在它的边界内。 */
  function collectAnchors() {
    const qsa = (sel) => Array.from(document.querySelectorAll(sel));
    const board = document.getElementById("croBoard");
    const pick = (selectedSel, relatedSel, hostSel) => {
      const host = hostSel ? document.querySelector(hostSel) : null;
      // 宿主自己也可能被 .cro-board 滚出去，两级都夹
      const fit = (rect) => clampRectTo(clampRectTo(rect, host), board);
      const one = document.querySelector(selectedSel);
      if (one) {
        // 零尺寸 = 元素在被隐藏的区域里（deck 正视图的非 front 层是
        // display:none，折叠的整网区同理）。此时 rect 全 0，直接当锚点用
        // 会把连线拉到视口左上角，必须判无锚点。
        // 有效性判在**未夹取**的原始 rect 上：夹取会把「在可视区外但确实存在」
        // 的元素压成零高/零宽的一条边，那是合法锚点，不能当成不存在。
        const rect = one.getBoundingClientRect();
        if (rect.width || rect.height) return { rect: fit(rect), group: false };
      }
      const many = qsa(relatedSel);
      if (!many.length) return null;
      // 元素全部为零尺寸（比如所在区域被折叠）时 unionRect 返回 null，
      // 这里直接判无锚点，别把 null 传下去让 centerOf 炸掉。
      const rect = unionRect(many);
      return rect ? { rect: fit(rect), group: many.length > 1 } : null;
    };
    return {
      // 粗粒度选择（rank / stage / 层）没有被选中的算子节点，退到「被牵连的层卡」
      // 上取锚点。正视图下非 front 的层卡是 display:none、rect 全 0，unionRect
      // 会把它们跳过，实际落到当前正视的那张卡上，不会拉出一个巨大的包围盒。
      net: pick(
        "#croDeckHost .pto-model-deck__node.is-selected",
        "#croDeckHost .pto-model-deck__node.is-selected, #croDeckHost .pto-model-deck__layer.is-selected, #croDeckHost .pto-model-deck__layer.is-related",
        "#croDeckHost",
      ),
      nav: pick(".cro-tick.is-selected", ".cro-tick.is-related", "#croLayerNav"),
      // 整列点击时没有单个 .cro-bar.is-selected，锚点取整块底板（.cro-structure__col
      // .is-selected .cro-structure__stack）；单算子点击则仍锚在那根算子条上。
      arch: pick(
        ".cro-bar.is-selected, .cro-structure__col.is-selected .cro-structure__stack",
        ".cro-structure__col.is-related .cro-structure__stack",
        "#croStructure",
      ),
      // 选中整个 EP 组时，连线要接到组卡片本身（与白描边同一个框），
      // 而不是退化成组内专家的并集包围盒再补一圈虚线。
      moe: pick(
        ".cro-moe-group.is-selected, .cro-expert.is-selected",
        ".cro-expert.is-related, .cro-moe-group.is-related",
        ".cro-region--moe",
      ),
      // 夹取宿主是滚动视口 .cro-cluster__grid 而不是矩阵本身 —— rank 多到矩阵
      // 要内部滚动时，#croHeat 的 rect 比看得见的那块高，锚点会落到区域之外
      cluster: pick("#croHeat .twin-heat-cell.is-selected", "#croHeat .twin-heat-cell.is-related", ".cro-cluster__grid"),
    };
  }

  /* 集群里被牵连的格子按 PP stage 拆成多个锚点。点专家/EP 组/共享专家时，该编号的
     EP 组在**每个 stage** 内都占一块 rank —— 集群图正好横向分成 pp 个 stage 块，把
     每块的并集包围盒各作一个锚点，drawRelationLinks 就能对每个 stage 各拉一条线，
     而不是把 4 段并成一个横跨整幅热力图的巨框。 */
  function clusterStageAnchors() {
    const host = document.querySelector("#croHeat");
    const viewport = document.querySelector(".cro-cluster__grid") || host;
    const board = document.getElementById("croBoard");
    if (!host) return [];
    // 同 collectAnchors：夹在滚动视口上，矩阵内部滚动时锚点不跑出可视区
    const fit = (rect) => clampRectTo(clampRectTo(rect, viewport), board);
    const byStage = new Map();
    host.querySelectorAll(".twin-heat-cell.is-related, .twin-heat-cell.is-selected").forEach((cell) => {
      const s = Number(cell.dataset.stage);
      if (!Number.isFinite(s)) return;
      if (!byStage.has(s)) byStage.set(s, []);
      byStage.get(s).push(cell);
    });
    const out = [];
    byStage.forEach((cells, stage) => {
      const rect = unionRect(cells);
      if (rect) out.push({ stage, rect: fit(rect), group: cells.length > 1 });
    });
    return out.sort((a, b) => a.stage - b.stage);
  }

  const centerOf = (r) => ({ x: (r.left + r.right) / 2, y: (r.top + r.bottom) / 2 });

  /* 端点落在包围盒朝向对方的那条边上，而不是几何中心 —— 否则一整组的连线
     会从组的正中穿出来，看着像指向组里某一格。 */
  function edgePoint(r, toward) {
    const c = centerOf(r);
    const dx = toward.x - c.x;
    const dy = toward.y - c.y;
    if (Math.abs(dx) * r.height > Math.abs(dy) * r.width) {
      return { x: dx > 0 ? r.right : r.left, y: c.y };
    }
    return { x: c.x, y: dy > 0 ? r.bottom : r.top };
  }

  function appendGroupOutline(layer, r) {
    // 夹到可视区边界后可能只剩一条线（目标整体在区域外），这时画虚线框没有意义
    if (r.width < 4 || r.height < 4) return;
    const box = document.createElementNS(SVG_NS, "rect");
    box.setAttribute("class", "cro-link-group");
    box.setAttribute("x", String(r.left - 3));
    box.setAttribute("y", String(r.top - 3));
    box.setAttribute("width", String(r.width + 6));
    box.setAttribute("height", String(r.height + 6));
    box.setAttribute("rx", "4");
    layer.appendChild(box);
  }

  function drawRelationLinks(layer, rel) {
    if (!layer) return;
    while (layer.firstChild) layer.removeChild(layer.firstChild);
    if (!rel) return;

    const anchors = collectAnchors();
    const order = ["net", "nav", "arch", "moe", "cluster"];
    const preferred = hubOf(rel);
    const hubKey = anchors[preferred] ? preferred : order.find((key) => anchors[key]);
    const hub = hubKey && anchors[hubKey];
    if (!hub) return;

    const hubCenter = centerOf(hub.rect);
    const labelFor = {
      net: rel.labels.arch, arch: rel.labels.arch, nav: rel.labels.arch,
      moe: rel.labels.moe, cluster: rel.labels.cluster,
    };

    // 一条 hub→target 的曲线 + 可选中点标签 + 可选整组虚线框
    const drawLink = (targetRect, isGroup, labelText) => {
      const toCenter = centerOf(targetRect);
      const from = edgePoint(hub.rect, toCenter);
      const to = edgePoint(targetRect, hubCenter);
      if (!Number.isFinite(to.x) || !Number.isFinite(from.x)) return;
      if (isGroup) appendGroupOutline(layer, targetRect);
      const bend = Math.max(48, Math.abs(to.x - from.x) * 0.45);
      const path = document.createElementNS(SVG_NS, "path");
      path.setAttribute("d", `M${from.x} ${from.y}C${from.x + (to.x > from.x ? bend : -bend)} ${from.y},${to.x + (to.x > from.x ? -bend : bend)} ${to.y},${to.x} ${to.y}`);
      path.setAttribute("class", "cro-link");
      layer.appendChild(path);
      if (labelText) appendLinkLabel(layer, labelText, (from.x + to.x) / 2, (from.y + to.y) / 2);
    };

    if (hub.group) appendGroupOutline(layer, hub.rect);

    // 点专家/EP 组/共享专家时，该编号的 rank 分布在每个 PP stage 里 —— 集群侧按 stage
    // 拆成多条线（每段一条 + 一圈虚线框），整段的「Node… · N 卡」标签只挂在离 hub 最近
    // 的那条上，其余段只留虚线框，避免 4 个标签堆叠。
    const fanCluster = rel.primary
      && (rel.primary.kind === "expert" || rel.primary.kind === "epRank" || rel.primary.kind === "sharedExpert");

    order.forEach((key) => {
      if (key === hubKey) return;
      if (key === "cluster" && fanCluster) {
        const stageAnchors = clusterStageAnchors();
        if (stageAnchors.length) {
          // 离 hub（MoE 列，在右侧）最近的一段挂总标签
          let nearest = 0; let best = Infinity;
          stageAnchors.forEach((a, i) => {
            const d = Math.abs(centerOf(a.rect).x - hubCenter.x) + Math.abs(centerOf(a.rect).y - hubCenter.y);
            if (d < best) { best = d; nearest = i; }
          });
          stageAnchors.forEach((a, i) => drawLink(a.rect, a.group, i === nearest ? rel.labels.cluster : null));
          return;
        }
      }
      const target = anchors[key];
      if (!target) return;
      drawLink(target.rect, target.group, labelFor[key]);
    });
  }

  function hubOf(rel) {
    switch (rel.primary.kind) {
      case "rank": return "cluster";
      case "expert": case "epRank": case "sharedExpert": return "moe";
      case "layer": case "stage": return "nav";
      default: return "arch";
    }
  }

  /* text 可以是一行字符串，也可以是多行数组（第二行起是补充说明，用 __sub 弱化）。
     多行时整块以 (x, y) 为竖直中心排布，行距 LINE_H。 */
  function appendLinkLabel(layer, text, x, y) {
    const lines = (Array.isArray(text) ? text : [text]).filter(Boolean);
    if (!lines.length) return;
    const LINE_H = 16;
    const group = document.createElementNS(SVG_NS, "g");
    const box = document.createElementNS(SVG_NS, "rect");
    const label = document.createElementNS(SVG_NS, "text");
    label.setAttribute("class", "cro-link-label__text");
    label.setAttribute("x", String(x));
    label.setAttribute("y", String(y));
    label.setAttribute("dominant-baseline", "middle");
    label.setAttribute("text-anchor", "middle");
    const top = y - ((lines.length - 1) * LINE_H) / 2;
    lines.forEach((line, i) => {
      const tspan = document.createElementNS(SVG_NS, "tspan");
      tspan.setAttribute("x", String(x));
      tspan.setAttribute("y", String(top + i * LINE_H));
      if (i > 0) tspan.setAttribute("class", "cro-link-label__sub");
      tspan.textContent = line;
      label.appendChild(tspan);
    });
    box.setAttribute("class", "cro-link-label__box");
    group.append(box, label);
    layer.appendChild(group);
    // 先入 DOM 才能量到文字尺寸，再把底板补到文字后面
    const bbox = label.getBBox();
    const padX = 8;
    const padY = 5;
    box.setAttribute("x", String(bbox.x - padX));
    box.setAttribute("y", String(bbox.y - padY));
    box.setAttribute("width", String(bbox.width + padX * 2));
    box.setAttribute("height", String(bbox.height + padY * 2));
    box.setAttribute("rx", "4");
  }

  global.CroTopology = {
    MODEL_PRESETS,
    FIELD_SPECS,
    FIELD_ORDER,
    validate,
    derive,
    stepValue,
    reconcile,
    createController,
    deckConfigFrom,
    structureColumns,
    columnTemplate,
    resolveRelation,
    deckNodeIndex,
  };

  /* ── 选中项自动露出 ───────────────────────────────────────────────────────
     只滚 container 自己，绝不用 el.scrollIntoView()：后者会把**所有**祖先滚动
     容器一起滚（.cro-board 是 overflow:auto，document 也可滚），点一个专家会
     把整块面板连同整网图一起挪走。这里手算容器与目标的相对位置，只在目标真的
     不在可视区内时补差值，已经露着就一动不动。 */
  const REVEAL_PAD = 10;

  function revealIn(container, el) {
    if (!container || !el || !container.contains(el)) return;
    const box = container.getBoundingClientRect();
    const rect = el.getBoundingClientRect();
    let delta = 0;
    if (rect.top < box.top + REVEAL_PAD) delta = rect.top - box.top - REVEAL_PAD;
    else if (rect.bottom > box.bottom - REVEAL_PAD) delta = rect.bottom - box.bottom + REVEAL_PAD;
    if (!delta) return;
    // 目标比可视区还高时上面的算法会把它顶到底边，改为对齐顶部
    if (rect.height > box.height - REVEAL_PAD * 2) delta = rect.top - box.top - REVEAL_PAD;
    container.scrollBy({ top: delta, behavior: "smooth" });
  }

  /* 按优先级取第一个命中的元素。querySelector 传选择器列表是按**文档顺序**
     返回的，不是按列表顺序，会出现「先选中了某个专家，却滚到了排在它前面的
     某个 is-related 组」。 */
  function firstMatch(root, selectors) {
    if (!root) return null;
    for (const selector of selectors) {
      const el = root.querySelector(selector);
      if (el) return el;
    }
    return null;
  }

  /* ══ 事件内涵 · 证据图表 ═══════════════════════════════════════════════════
     三种形态，按「这组数要回答什么」选，不按好看选：
       line  —— 随时间怎么变（loss scale 衰减、显存爬升）
       bars  —— 谁比谁大（专家 token 份额、逐层激活）
       stack —— 一个总量由什么构成（显存构成、step 耗时构成、2048 卡状态构成）
     两条硬规矩：
       1. 一张图只有一根 y 轴。量纲不同的第二个指标进读数区，不叠双轴。
       2. 取色只用设计系统 token。--primary 是常规量；--warning / --danger 专门
          留给「确实是问题」的那一段，不当第 N 个分类色使唤。
     这套取色跑过 CVD 校验：相邻色对在色盲与常视觉下都可分；浅色主题下绿/橙对
     底色的对比度不足 3:1，故每段都带可见数值标签，不靠颜色单独承载信息。 */
  const CHART_TONE = {
    neutral: "var(--primary)",
    /* 第二个分类色。不用状态色（绿/橙/红各有语义），也不用 deck 的 comm 青
       —— 青与 --primary 蓝在常视觉下 ΔE 只有 11，低于 15 的可分辨下限，
       两段挨着放根本认不出是两类。紫是本页现成的 deck 语义色，与蓝的 ΔE 18。 */
    alt: "var(--pto-model-deck-mlp)",
    good: "var(--success)",
    warning: "var(--warning)",
    danger: "var(--danger)",
  };

  function svgNode(name, attrs = {}, text) {
    const el = document.createElementNS(SVG_NS, name);
    Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, String(value)));
    if (text != null) el.textContent = text;
    return el;
  }

  function fmtValue(value, unit = "") {
    const abs = Math.abs(value);
    const digits = abs >= 100 || Number.isInteger(value) ? 0 : abs >= 10 ? 1 : 2;
    return `${value.toLocaleString("en-US", { maximumFractionDigits: digits })}${unit}`;
  }

  /* 数据端圆角、基线端方角（rx 会把两头都磨圆，读起来像浮在轨道上的胶囊）。 */
  function barPath(x, y, w, h, r = 4) {
    const rr = Math.max(0, Math.min(r, w, h / 2));
    if (w <= 0) return "";
    return `M${x},${y}h${w - rr}a${rr},${rr} 0 0 1 ${rr},${rr}v${h - rr * 2}a${rr},${rr} 0 0 1 ${-rr},${rr}h${-(w - rr)}z`;
  }

  /* 数据端圆角、基线端方角的竖版（圆角在顶部） */
  function columnPath(x, y, w, h, r = 4) {
    const rr = Math.max(0, Math.min(r, w / 2, h));
    if (h <= 0) return "";
    return `M${x},${y + h}V${y + rr}a${rr},${rr} 0 0 1 ${rr},${-rr}h${w - rr * 2}a${rr},${rr} 0 0 1 ${rr},${rr}V${y + h}Z`;
  }

  /* 阈值判定。direction 缺省是「越过上界为坏」；loss scale 这类指标反过来，
     跌破下界才是坏，写 direction: "below"。 */
  function isOverThreshold(value, threshold) {
    if (!threshold || !Number.isFinite(value)) return false;
    return threshold.direction === "below" ? value < threshold.value : value > threshold.value;
  }

  let chartClipSeq = 0;   // 同页多图共存，clipPath id 不能撞

  function chartLine(spec, width, budget) {
    const W = width || 560, P = { l: 56, r: 20, t: 16, b: 26 };
    // 小屏下先压高度：折线的形状靠横向趋势读，压扁比出滚动条好
    const H = Math.max(104, Math.min(budget || 180, 180));
    const values = spec.values;
    const threshold = spec.threshold;
    // 阈值线必须落在画面内，否则「越线」这件事无从读起 —— 把它并进 y 轴域
    const domain = threshold ? values.concat(threshold.value) : values;
    const lo = Math.min(...domain), hi = Math.max(...domain);
    const span = hi - lo || 1;
    const plotBottom = H - P.b;
    const x = (i) => P.l + (i / Math.max(1, values.length - 1)) * (W - P.l - P.r);
    const y = (v) => P.t + (1 - (v - lo) / span) * (H - P.t - P.b);

    const svg = svgNode("svg", {
      class: "cro-chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": spec.title,
    });
    [0, 0.5, 1].forEach((t) => {
      const yy = P.t + t * (H - P.t - P.b);
      svg.appendChild(svgNode("line", { class: "cro-chart__grid", x1: P.l, x2: W - P.r, y1: yy, y2: yy }));
    });
    // y 轴只标上下界，中间靠网格线读
    svg.appendChild(svgNode("text", { class: "cro-chart__tick", x: P.l - 8, y: P.t + 4, "text-anchor": "end" }, fmtValue(hi, spec.unit)));
    svg.appendChild(svgNode("text", { class: "cro-chart__tick", x: P.l - 8, y: H - P.b + 4, "text-anchor": "end" }, fmtValue(lo, spec.unit)));

    const points = values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`);
    const d = `M${points.join("L")}`;
    // 面积填充：折线之下到基线围成的闭合区域，弱填充，只用来压出「量」的体感
    const area = `M${x(0).toFixed(1)},${plotBottom}L${points.join("L")}L${x(values.length - 1).toFixed(1)},${plotBottom}Z`;
    svg.appendChild(svgNode("path", { class: "cro-chart__area", d: area }));

    /* 越过阈值的那一段单独标红：不改折线的取值，只用一个裁剪框把「阈值之外」
       的画面切出来，在里面把面积与线重画一遍成红色 —— 与显存曲线同一套画法。 */
    if (threshold) {
      const ty = y(threshold.value);
      const clipId = `croChartOver-${chartClipSeq += 1}`;
      const defs = svgNode("defs");
      const clip = svgNode("clipPath", { id: clipId });
      clip.appendChild(svgNode("rect", {
        x: P.l, width: W - P.l - P.r,
        y: threshold.direction === "below" ? ty : P.t,
        height: Math.max(0, threshold.direction === "below" ? plotBottom - ty : ty - P.t),
      }));
      defs.appendChild(clip);
      svg.appendChild(defs);
      svg.appendChild(svgNode("path", {
        class: "cro-chart__area is-over", d: area, "clip-path": `url(#${clipId})`,
      }));
      svg.appendChild(svgNode("line", {
        class: "cro-chart__threshold", x1: P.l, x2: W - P.r, y1: ty, y2: ty,
      }));
      svg.appendChild(svgNode("text", {
        class: "cro-chart__threshold-label", x: W - P.r, y: ty - 5, "text-anchor": "end",
      }, threshold.label));
      svg.appendChild(svgNode("path", { class: "cro-chart__line", d }));
      svg.appendChild(svgNode("path", {
        class: "cro-chart__line is-over", d, "clip-path": `url(#${clipId})`,
      }));
    } else {
      svg.appendChild(svgNode("path", { class: "cro-chart__line", d }));
    }

    // x 轴只标首尾，异常点另有直标
    svg.appendChild(svgNode("text", { class: "cro-chart__tick", x: P.l, y: H - 6 }, spec.x[0]));
    svg.appendChild(svgNode("text", { class: "cro-chart__tick", x: W - P.r, y: H - 6, "text-anchor": "end" }, spec.x[spec.x.length - 1]));

    const mark = spec.mark;
    if (mark && Number.isFinite(mark.index)) {
      const mx = x(mark.index), my = y(values[mark.index]);
      // 2px 底色描边，让标记点从线上浮起来
      svg.appendChild(svgNode("circle", { class: "cro-chart__mark-ring", cx: mx, cy: my, r: 6 }));
      const dot = svgNode("circle", { class: "cro-chart__mark", cx: mx, cy: my, r: 4 });
      dot.style.fill = CHART_TONE[mark.tone || "danger"];
      svg.appendChild(dot);
      const anchor = mark.index > values.length / 2 ? "end" : "start";
      const label = svgNode("text", {
        class: "cro-chart__mark-label", x: mx + (anchor === "end" ? -10 : 10), y: my + 4, "text-anchor": anchor,
      }, mark.label);
      svg.appendChild(label);
    }
    return svg;
  }

  /* 条目多 / 高度紧张时，横条改竖排（一根一根并排的柱子）：横条一根占一行，
     12 项就是 12 行，高度只增不减；竖柱把「多」摊到宽度上，高度恒定。
     横条留给条目少、类目名长的情形（send/recv 那种），它读起来更稳。 */
  const BARS_ROW_MIN = 15;

  function chartBars(spec, width, budget) {
    const padY = 6, labelW = 128, valueW = 78;
    const W = width || 560;
    const items = spec.items;
    const rowsNeed = padY * 2 + items.length * BARS_ROW_MIN;
    if (items.length > 6 || (budget && rowsNeed > budget)) return chartColumns(spec, width, budget);

    const cap = Math.max(120, Math.min(budget || 210, 210));
    const rowH = Math.max(BARS_ROW_MIN, Math.min(22, (cap - padY * 2) / items.length));
    const barH = Math.min(10, Math.max(5, rowH - 9));
    const H = padY * 2 + items.length * rowH;
    const threshold = spec.threshold;
    const max = Math.max(...items.map((i) => i.value), threshold ? threshold.value : 0) || 1;
    const trackW = W - labelW - valueW;
    const svg = svgNode("svg", {
      class: "cro-chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": spec.title,
    });
    items.forEach((item, index) => {
      const y = padY + index * rowH;
      const barY = y + (rowH - barH) / 2;
      svg.appendChild(svgNode("text", {
        class: "cro-chart__cat", x: labelW - 10, y: barY + barH - 2, "text-anchor": "end",
      }, item.label));
      svg.appendChild(svgNode("path", {
        class: "cro-chart__track", d: barPath(labelW, barY, trackW, barH),
      }));
      const width = Math.max(item.value > 0 ? 2 : 0, (item.value / max) * trackW);
      if (width) {
        const bar = svgNode("path", { class: "cro-chart__bar", d: barPath(labelW, barY, width, barH) });
        // 越过警戒线的自动标红：红色的来由就是那条线，不靠手工指定
        const over = isOverThreshold(item.value, threshold);
        bar.style.fill = CHART_TONE[over ? "danger" : (item.tone || "neutral")];
        bar.appendChild(svgNode("title", {}, `${item.label}：${fmtValue(item.value, spec.unit)}`));
        svg.appendChild(bar);
      }
      svg.appendChild(svgNode("text", {
        class: "cro-chart__value", x: labelW + trackW + 10, y: barY + barH - 2,
      }, fmtValue(item.value, spec.unit)));
    });
    if (threshold) {
      const tx = labelW + (threshold.value / max) * trackW;
      svg.appendChild(svgNode("line", {
        class: "cro-chart__threshold", x1: tx, x2: tx, y1: padY, y2: H - padY,
      }));
      svg.appendChild(svgNode("text", {
        class: "cro-chart__threshold-label", x: tx + 4, y: padY + 9,
      }, threshold.label));
    }
    return svg;
  }

  /* 竖排柱：类目摊在横轴上，高度恒定不随条目数增长。
     直标只给「有问题」的那几根（tone 非 neutral），12 根全标必然叠字。 */
  function chartColumns(spec, width, budget) {
    const W = width || 560;
    const H = Math.max(104, Math.min(budget || 160, 168));
    const P = { l: 44, r: 10, t: 16, b: 20 };
    const items = spec.items;
    const threshold = spec.threshold;
    const plotW = W - P.l - P.r, plotH = H - P.t - P.b;
    const max = Math.max(...items.map((i) => i.value), threshold ? threshold.value : 0) || 1;
    const slot = plotW / items.length;
    const barW = Math.max(4, Math.min(28, slot - 8));

    const svg = svgNode("svg", {
      class: "cro-chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": spec.title,
    });
    [0, 0.5, 1].forEach((t) => {
      const y = P.t + t * plotH;
      svg.appendChild(svgNode("line", { class: "cro-chart__grid", x1: P.l, x2: W - P.r, y1: y, y2: y }));
    });
    svg.appendChild(svgNode("text", {
      class: "cro-chart__tick", x: P.l - 8, y: P.t + 4, "text-anchor": "end",
    }, fmtValue(max, spec.unit)));
    svg.appendChild(svgNode("text", {
      class: "cro-chart__tick", x: P.l - 8, y: P.t + plotH + 4, "text-anchor": "end",
    }, "0"));

    if (threshold) {
      const ty = P.t + plotH - (threshold.value / max) * plotH;
      svg.appendChild(svgNode("line", {
        class: "cro-chart__threshold", x1: P.l, x2: W - P.r, y1: ty, y2: ty,
      }));
      svg.appendChild(svgNode("text", {
        class: "cro-chart__threshold-label", x: W - P.r, y: ty - 4, "text-anchor": "end",
      }, threshold.label));
    }

    items.forEach((item, index) => {
      const cx = P.l + slot * (index + 0.5);
      const h = (item.value / max) * plotH;
      const x = cx - barW / 2;
      // 越过警戒线的自动标红：红色的来由就是那条线，不靠手工指定
      const over = isOverThreshold(item.value, threshold);
      if (h > 0) {
        const bar = svgNode("path", {
          class: "cro-chart__bar", d: columnPath(x, P.t + plotH - h, barW, h),
        });
        bar.style.fill = CHART_TONE[over ? "danger" : (item.tone || "neutral")];
        bar.appendChild(svgNode("title", {}, `${item.label}：${fmtValue(item.value, spec.unit)}`));
        svg.appendChild(bar);
      }
      svg.appendChild(svgNode("text", {
        class: "cro-chart__cat", x: cx, y: H - 6, "text-anchor": "middle",
      }, item.label));
      if (over || (item.tone && item.tone !== "neutral")) {
        svg.appendChild(svgNode("text", {
          class: "cro-chart__value", x: cx, y: P.t + plotH - h - 5, "text-anchor": "middle",
        }, fmtValue(item.value, spec.unit)));
      }
    });
    return svg;
  }

  /* 构成条：与 pto-swimlane-profiler inspector 里的 .sl-meter + .sl-kv 同款——
     一条 pill 轨道，几段并排铺满不留缝，读数一律落到下面的直标行里。
     不再走 SVG：这张图没有坐标轴，用 DOM 反而能直接吃到 token 和 pill 圆角。 */
  function chartStack(spec) {
    const items = spec.items;
    const total = items.reduce((sum, item) => sum + item.value, 0) || 1;
    const wrap = document.createElement("div");
    wrap.className = "cro-chart-stack";

    const meter = document.createElement("div");
    meter.className = "cro-meter";
    meter.setAttribute("role", "img");
    meter.setAttribute("aria-label", spec.title);
    const addSeg = (share, tone, tip, over) => {
      const seg = document.createElement("div");
      seg.className = over ? "cro-meter__seg cro-meter__seg--over" : "cro-meter__seg";
      seg.style.width = `${share * 100}%`;
      seg.style.setProperty("--cro-seg", CHART_TONE[tone] || tone);
      if (tip) seg.dataset.tip = tip;
      meter.appendChild(seg);
    };

    items.forEach((item) => {
      const share = item.value / total;
      const tip = `${item.label}：${fmtValue(item.value, spec.unit)}`;
      /* limitShare = 这一段「本该占多少」。实际超出时就地切两截：两截同色
         （它们是同一件事，不该被读成两种严重度），超出的那截换成同色细斜纹
         —— 一眼读出「多出来的是哪一块、有多大」，而不是只知道总数偏大。 */
      const limit = Number.isFinite(item.limitShare) ? item.limitShare / 100 : null;
      const tone = item.tone || "neutral";
      if (limit !== null && share > limit) {
        addSeg(limit, tone, `${tip}（其中正常水位 ${item.limitShare}%）`, false);
        addSeg(share - limit, tone, `${tip}（超出水位的部分）`, true);
      } else {
        addSeg(share, tone, tip, false);
      }
    });
    wrap.appendChild(meter);

    wrap.appendChild(chartLegend(items, total, spec.unit));
    return wrap;
  }

  /* 图例即直标：色块只管身份，数值与占比一律用文字色，不靠颜色读数。
     构成条与等距容器（chartCapacity）共用这一份 —— 两张图讲的是同一种「构成」，
     直标行长得不一样会让人以为是两套读数。
     total 由调用方给：构成条按各段之和算占比，等距容器按**容量**算（装了 80%
     和「这一段占已装部分的 80%」是两回事）。 */
  function chartLegend(items, total, unit, options) {
    const opts = options || {};
    const legend = document.createElement("ul");
    legend.className = "cro-chart-legend";
    const addRow = (dotClass, tone, label, value, sub) => {
      const li = document.createElement("li");
      li.className = sub ? "cro-chart-legend__item cro-chart-legend__item--sub" : "cro-chart-legend__item";
      const dot = document.createElement("span");
      dot.className = dotClass;
      dot.style.setProperty("--cro-seg", CHART_TONE[tone] || tone);
      const name = document.createElement("span");
      name.className = "cro-chart-legend__label";
      name.textContent = label;
      const val = document.createElement("span");
      val.className = "cro-chart-legend__value";
      val.textContent = value;
      li.append(dot, name, val);
      legend.appendChild(li);
    };
    (opts.reverse ? items.slice().reverse() : items).forEach((item) => {
      const tone = item.tone || "neutral";
      const pct = (item.value / total) * 100;
      const overrun = Number.isFinite(item.limitShare) && pct > item.limitShare;
      // 空当段（碎片/预留）在等距容器里画的是虚线棱，图例键也换成同义的斜纹块
      addRow(item.void ? "cro-chart-legend__dot cro-chart-legend__dot--void" : "cro-chart-legend__dot",
        tone, item.label, Number.isFinite(item.limitShare)
          ? `${fmtValue(item.value, unit)} · ${pct.toFixed(1)}%（正常 ${item.limitShare}%）`
          : `${fmtValue(item.value, unit)} · ${Math.round(pct)}%`);
      // 斜纹那截自己占一行，纹样即图例键——否则条上多出来的纹理没人解释
      if (overrun) {
        const excess = item.value - (item.limitShare / 100) * total;
        addRow("cro-chart-legend__dot cro-chart-legend__dot--over", tone, "超出正常水位",
          `${fmtValue(excess, unit)} · ${(pct - item.limitShare).toFixed(1)}%`, true);
      }
    });
    return legend;
  }

  /* 等距容器：与 Cluster 区「单卡容量」栏同一个 builder（config-relation-capacity.js
     的 global.croCapacityBox），不在这里再画一份 3D。
     用它而不是构成条的判据是「有没有一个固定的容量上限」：显存构成图的分母是那张
     卡的 64 GB，装不下就是 OOM —— 平面色条只能表达比例，表达不了「框满了」。
     spec 比 stack 多一个 cap（容量，与 items[].value 同单位）；items 自底向上排，
     图例按视觉从上往下读，所以倒序。 */
  function chartCapacity(spec, width, budget, host) {
    const api = global.croCapacityBox;
    // 容量脚本没加载（或被单独引用本文件的页面复用）时退回构成条，不留空白
    if (!api || !(spec.cap > 0)) return chartStack(spec);

    const wrap = document.createElement("div");
    wrap.className = "cro-chart-capacity";
    const scene = document.createElement("div");
    scene.className = "cro-chart-capacity__scene";
    /* 盒子是竖长的，高度就是它的「量纲」——给到预算上限（下限 148px，再矮
       阈值环上的百分比标签会挤成一团）。宽度由 CSS 固定，SVG 自己等比缩放。 */
    scene.style.height = `${Math.max(148, Math.min(budget || 220, 260))}px`;
    wrap.appendChild(scene);
    wrap.appendChild(chartLegend(spec.items, spec.cap, spec.unit, { reverse: true }));

    /* var(--token) 要在**已挂到文档上**的节点上解析（deck 语义色定义在
       .cro-incident-view 上，不在 :root）。wrap 此刻还是游离的，所以传 host。 */
    scene.appendChild(api.build({
      cap: spec.cap,
      host: host || document.body,
      ariaLabel: spec.title,
      format: (v) => fmtValue(v, spec.unit),
      segments: spec.items.map((item) => ({
        label: item.label,
        value: item.value,
        color: CHART_TONE[item.tone || "neutral"],
        // 空当不是「装进去的东西」，与单卡容量栏的预留段同一种画法
        dashed: !!item.void,
        opacity: item.void ? 0.62 : null,
      })),
      thresholds: [
        { at: api.THRESHOLD.tight, color: CHART_TONE.warning },
        { at: api.THRESHOLD.alert, color: CHART_TONE.danger },
      ],
    }));
    return wrap;
  }

  const CHART_BUILDERS = { line: chartLine, bars: chartBars, stack: chartStack, capacity: chartCapacity };

  /* ══ 计算血缘 ════════════════════════════════════════════════════════════
     这里刻意不把 FX / GE / Runtime 写进 CroTopology：拓扑是配置映射，血缘是
     编译与执行映射，两者生命周期不同。本页尚未接 profiler / compiler dump，
     因此下面构造一条完整、可交互的演示链。 */
  const LINEAGE_LOWERING = {
    embedding: {
      fx: ["aten.embedding.default"], ge: ["GatherV2"],
      runtime: ["GatherV2 task"], kernel: ["gather_v2_aicore_tiling_v3"],
    },
    norm: {
      fx: ["aten.rms_norm.default"], ge: ["RmsNorm"],
      runtime: ["RmsNorm task"], kernel: ["rms_norm_vector_tiling_v2"],
    },
    attention: {
      fx: ["aten.matmul.default", "aten.softmax.int", "aten.matmul.default"],
      ge: ["FlashAttentionScore"], runtime: ["FlashAttentionScore task"],
      kernel: ["flash_attention_score_aicore_v4"],
    },
    linear: {
      fx: ["aten.linear.default"], ge: ["MatMul", "Add"],
      runtime: ["MatMul task", "Vector Add task"],
      kernel: ["matmul_cube_tiling_v5", "add_vector_tiling_v2"],
    },
    head: {
      fx: ["aten.linear.default"], ge: ["MatMul"],
      runtime: ["LM Head MatMul task"], kernel: ["matmul_cube_split_k_v3"],
    },
    mlp: {
      fx: ["aten.linear.default", "aten.silu.default", "aten.linear.default"],
      ge: ["MatMul", "Swish", "MatMul"], runtime: ["Fused MLP task"],
      kernel: ["fused_mlp_cube_vector_v2"],
    },
    act: {
      fx: ["aten.silu.default"], ge: ["Swish"],
      runtime: ["Swish task"], kernel: ["swish_vector_tiling_v2"],
    },
    gate: {
      fx: ["aten.linear.default", "aten.softmax.int", "aten.topk.default"],
      ge: ["MatMul", "SoftmaxV2", "TopK"],
      runtime: ["Router MatMul task", "Router select task"],
      kernel: ["router_matmul_cube_v3", "softmax_topk_vector_v4"],
    },
    moe: {
      fx: ["call_function.moe_expert_pool"], ge: ["MoeGatingTopK", "MoeFinalizeRouting"],
      runtime: ["MoE expert task group"], kernel: ["moe_expert_ffn_aicore_v3"],
    },
    comm: {
      fx: ["call_function.npu_all_to_all"], ge: ["HcomAllToAll"],
      runtime: ["HCCL collective task"], kernel: ["HCCL AllToAll executor"],
    },
    output: {
      fx: ["call_function.output"], ge: ["NetOutput"],
      runtime: ["Graph output task"], kernel: ["device_to_host_completion"],
    },
    decoder: {
      fx: ["call_module.decoder_block"], ge: ["PartitionedCall"],
      runtime: ["Decoder task group"], kernel: ["decoder_kernel_group"],
    },
    input: {
      fx: ["placeholder.input"], ge: ["Data"],
      runtime: ["Graph input task"], kernel: ["host_to_device_enqueue"],
    },
    parameter: {
      fx: ["get_attr.parameter"], ge: ["Const"],
      runtime: ["Weight binding"], kernel: ["parameter_address_binding"],
    },
    state: {
      fx: ["get_attr.state"], ge: ["Variable"],
      runtime: ["State binding"], kernel: ["state_address_binding"],
    },
  };

  const LINEAGE_DEFAULT = {
    fx: ["call_function.custom_op"], ge: ["CustomOp"],
    runtime: ["Custom operator task"], kernel: ["custom_aicore_kernel"],
  };

  const LINEAGE_STAGE_META = [
    { id: "model", title: "模型语义", system: "模型语义层，用于定位用户模型中的结构与算子意图" },
    { id: "fx", title: "FX Graph", system: "PyTorch FX 图层，用于记录框架捕获后的算子级表达" },
    { id: "ge", title: "GE Graph", system: "GE 图编译层，用于呈现昇腾侧 Lowering、融合与图优化结果" },
    { id: "runtime", title: "Runtime", system: "CANN 运行时层，用于展示任务下发、Stream 与 Rank 执行位置" },
    { id: "kernel", title: "Kernel / Executor", system: "Kernel 执行层，用于说明最终 Kernel、通信执行器与 Tiling 选择" },
  ];

  const lineageStageMeta = (id) => LINEAGE_STAGE_META.find((stage) => stage.id === id);
  const LINEAGE_NODE_NAME_TIPS = {
    model: "模型语义节点名称：表示用户模型中的结构或算子意图。",
    fx: "FX 节点名称：表示 PyTorch 捕获后的算子级表达。",
    ge: "GE 算子名称：表示 Lowering、融合与图优化后的昇腾侧算子。",
    runtime: "Runtime 任务名称：表示 CANN 下发到 Stream 的执行任务。",
    kernel: "Kernel / Executor 名称：表示设备侧最终执行的 Kernel 或通信执行器。",
  };
  const LINEAGE_NODE_ID_TIP = "节点 ID：用于在当前定位链中标识节点并建立跨层映射。点击后可打开节点证据详情，查看输入输出、转换依据及关联日志（暂未上线）。";

  function lineageIdPart(value) {
    return String(value || "node").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
  }

  function lineageTransition(stageId, sourceCount, targetCount) {
    if (stageId === "model") return { kind: "capture", label: sourceCount < targetCount ? "捕获展开" : "捕获" };
    if (stageId === "fx") {
      if (sourceCount > targetCount) return { kind: "fusion", label: "融合" };
      if (sourceCount < targetCount) return { kind: "split", label: "拆分" };
      return { kind: "lowering", label: "Lowering" };
    }
    if (stageId === "ge") {
      return sourceCount > targetCount
        ? { kind: "fusion", label: "任务聚合" }
        : { kind: "dispatch", label: "任务下发" };
    }
    return sourceCount > targetCount
      ? { kind: "fusion", label: "执行融合" }
      : sourceCount < targetCount
        ? { kind: "split", label: "Kernel 拆分" }
        : { kind: "dispatch", label: "Kernel 选择" };
  }

  function buildLineageEdges(stages) {
    const edges = [];
    stages.slice(0, -1).forEach((stage, stageIndex) => {
      const next = stages[stageIndex + 1];
      const sources = stage.available === false ? [] : stage.nodes.filter((node) => node.id);
      const targets = next.available === false ? [] : next.nodes.filter((node) => node.id);
      if (!sources.length || !targets.length) return;
      const relation = lineageTransition(stage.id, sources.length, targets.length);
      const pairs = [];
      if (sources.length === 1 || targets.length === 1) {
        sources.forEach((source) => targets.forEach((target) => pairs.push([source, target])));
      } else if (sources.length === targets.length) {
        sources.forEach((source, index) => pairs.push([source, targets[index]]));
      } else {
        sources.forEach((source, index) => {
          pairs.push([source, targets[Math.min(targets.length - 1, Math.floor(index * targets.length / sources.length))]]);
        });
        targets.forEach((target, index) => {
          pairs.push([sources[Math.min(sources.length - 1, Math.floor(index * sources.length / targets.length))], target]);
        });
      }
      const seen = new Set();
      pairs.forEach(([source, target]) => {
        const key = `${source.id}>${target.id}`;
        if (seen.has(key)) return;
        seen.add(key);
        edges.push({
          id: key,
          from: source.id,
          to: target.id,
          fromStage: stage.id,
          toStage: next.id,
          kind: relation.kind,
          label: relation.label,
        });
      });
    });
    return edges;
  }

  function completeLineage(data) {
    data.edges = buildLineageEdges(data.stages);
    return data;
  }

  const INCIDENT_LINEAGE_ROLES = [
    { id: "origin", label: "传播源" },
    { id: "victim", label: "受影响" },
  ];

  function incidentRoleStage(scope, spec, roleId) {
    const nodeRef = spec?.[`${roleId}Node`];
    if (LINEAGE_STAGE_META.some((stage) => stage.id === nodeRef?.stage)) return nodeRef.stage;
    const explicit = spec?.[`${roleId}Stage`];
    if (LINEAGE_STAGE_META.some((stage) => stage.id === explicit)) return explicit;
    if (scope?.ranks != null || scope?.stages != null || scope?.experts != null) return "runtime";
    if (scope?.layers != null || scope?.segments != null) return "model";
    const stages = spec?.stages || [];
    return roleId === "origin" ? stages[0] : stages[stages.length - 1];
  }

  function attachIncidentLineageRoles(data, event, spec) {
    INCIDENT_LINEAGE_ROLES.forEach((role) => {
      const stageId = incidentRoleStage(event?.[role.id], spec, role.id);
      const stage = data.stages.find((entry) => entry.id === stageId);
      if (!stage) return;
      const nodeRef = spec?.[`${role.id}Node`];
      const node = nodeRef?.stage === stage.id ? stage.nodes[nodeRef.index] : null;
      // 只有事件显式声明了层与节点序号，角色才下沉到卡片；不能靠“该层恰好只有
      // 一张卡”猜测异常落点，否则会把层级证据误读成节点级证据。
      if (stage.available && node?.id) {
        node.roles.push(role);
      } else {
        stage.roles.push(role);
      }
    });
    return data;
  }

  function buildMockLineage(topology, rel) {
    if (!rel?.bar || rel.primary?.kind !== "segment" || rel.primary.wholeColumn) return null;
    const col = activeColumns(topology).find((entry) => entry.id === rel.bar.segment);
    const bar = col?.bars.find((entry) => entry.id === rel.bar.bar);
    if (!bar) return null;

    const p = rel.primary;
    const layers = Array.from(rel.layers).sort((a, b) => a - b);
    const explicitLayer = [p.scopeLayer, p.preferLayer]
      .find((value) => Number.isFinite(value) && (!layers.length || layers.includes(value)));
    const representativeLayer = Number.isFinite(explicitLayer)
      ? explicitLayer
      : Number.isFinite(rel.deckLayer) ? rel.deckLayer : null;
    const scopeText = Number.isFinite(explicitLayer)
      ? `Layer ${explicitLayer}`
      : layers.length
        ? `${formatRuns(layers, "L", 2)} · 代表实例 L${representativeLayer}`
        : `${col.name} · PP${Array.from(rel.stages)[0] ?? 0}`;

    const ranks = Array.from(rel.ranks).sort((a, b) => a - b);
    const representativeStage = Number.isFinite(representativeLayer)
      ? topology.stageOfLayer(representativeLayer)
      : Array.from(rel.stages)[0];
    const rank = ranks.find((value) => topology.coordsOfRank(value).stage === representativeStage) ?? ranks[0];
    const co = Number.isFinite(rank) ? topology.coordsOfRank(rank) : null;
    const placement = co
      ? coordLine(topology, co)
      : "未绑定代表执行 Rank";
    const layerKey = Number.isFinite(representativeLayer) ? `l${representativeLayer}` : col.id;
    const nodeKey = lineageIdPart(bar.deckNode || bar.id);
    const lowering = LINEAGE_LOWERING[bar.op] || LINEAGE_DEFAULT;
    const stream = Number.isFinite(representativeLayer) ? representativeLayer % 8 : 0;

    const nodes = (stage, names, detail) => names.map((name, index) => ({
      name,
      id: `lineage:${stage}:${layerKey}:${nodeKey}:${index}`,
      detail: typeof detail === "function" ? detail(name, index) : detail,
    }));

    return completeLineage({
      key: `${layerKey}/${col.id}/${bar.id}/${rank}`,
      operator: bar.label,
      scope: scopeText,
      placement,
      stages: [
        {
          ...lineageStageMeta("model"),
          nodes: nodes("model", [bar.label], `${scopeText} · ${col.name}`),
        },
        {
          ...lineageStageMeta("fx"),
          nodes: nodes("fx", lowering.fx, (_name, index) => `call_function · node ${index + 1}/${lowering.fx.length}`),
        },
        {
          ...lineageStageMeta("ge"),
          nodes: nodes("ge", lowering.ge, (_name, index) => `lowering / fusion result · op ${index + 1}/${lowering.ge.length}`),
        },
        {
          ...lineageStageMeta("runtime"),
          nodes: nodes("runtime", lowering.runtime, () => `stream ${stream} · ${placement}`),
        },
        {
          ...lineageStageMeta("kernel"),
          nodes: nodes("kernel", lowering.kernel, (name) => name.startsWith("HCCL")
            ? "communication executor · execution selection"
            : "AI Core / Vector Core · tiling selection"),
        },
      ],
    });
  }

  function buildIncidentLineage(event, topology) {
    const spec = event?.lineage || {};
    const available = new Set(spec.stages || []);
    const columns = activeColumns(topology);
    const col = spec.segment ? columns.find((entry) => entry.id === spec.segment) : null;
    const bar = spec.bar ? col?.bars.find((entry) => entry.id === spec.bar) : null;
    const lowering = bar ? (LINEAGE_LOWERING[bar.op] || LINEAGE_DEFAULT) : null;
    const operator = spec.operator || bar?.label || event?.title || "未定位算子";
    const scope = Number.isFinite(spec.layer)
      ? `Layer ${spec.layer}${col ? ` · ${col.name}` : ""}`
      : "尚未定位到具体 Layer";
    const co = Number.isFinite(spec.rank) ? topology.coordsOfRank(spec.rank) : null;
    const placement = co
      ? coordLine(topology, co)
      : "尚未定位到具体执行 Rank";
    const layerKey = Number.isFinite(spec.layer) ? `l${spec.layer}` : "unscoped";
    const nodeKey = lineageIdPart(bar?.deckNode || spec.bar || operator);
    const stream = Number.isFinite(spec.layer) ? spec.layer % 8 : 0;

    const namesByStage = {
      model: [operator],
      fx: lowering?.fx || [],
      ge: lowering?.ge || [],
      runtime: lowering?.runtime || (available.has("runtime") ? [operator] : []),
      kernel: lowering?.kernel || (available.has("kernel") ? [operator] : []),
    };
    const detailByStage = {
      model: scope,
      fx: "框架捕获后的算子表达",
      ge: "Lowering / 融合结果",
      runtime: `${placement} · stream ${stream}`,
      kernel: "Kernel / Executor 与 Tiling 选择",
    };

    const data = {
      event: `${event.code || event.id} · ${event.title}`,
      operator,
      scope,
      placement,
      stages: LINEAGE_STAGE_META.map((meta) => {
        const isAvailable = available.has(meta.id);
        const names = isAvailable && namesByStage[meta.id]?.length
          ? namesByStage[meta.id]
          : ["本事件未定位到该层"];
        return {
          ...meta,
          available: isAvailable,
          roles: [],
          nodes: names.map((name, index) => ({
            name,
            id: isAvailable ? `event:${event.id}:${meta.id}:${layerKey}:${nodeKey}:${index}` : "",
            detail: isAvailable ? detailByStage[meta.id] : "缺少该层的直接定位信息",
            roles: [],
          })),
        };
      }),
    };
    return completeLineage(attachIncidentLineageRoles(data, event, spec));
  }

  function createLineageDrawer() {
    const drawer = document.getElementById("croLineageDrawer");
    const closeButton = document.getElementById("croLineageClose");
    const summary = document.getElementById("croLineageSummary");
    const stages = document.getElementById("croLineageStages");
    let currentKey = null;

    const close = () => {
      if (!drawer) return;
      drawer.hidden = true;
      currentKey = null;
    };

    const addSummaryRow = (list, label, value) => {
      const row = document.createElement("div");
      row.className = "cro-lineage-summary__row";
      const dt = document.createElement("dt");
      dt.textContent = label;
      const dd = document.createElement("dd");
      dd.textContent = value;
      row.append(dt, dd);
      list.appendChild(row);
    };

    const open = (data) => {
      if (!drawer || !summary || !stages || !data) return;
      currentKey = data.key;
      summary.replaceChildren();
      stages.replaceChildren();

      const list = document.createElement("dl");
      list.className = "cro-lineage-summary__list";
      addSummaryRow(list, "模型算子", data.operator);
      addSummaryRow(list, "结构范围", data.scope);
      addSummaryRow(list, "代表执行位置", data.placement);
      summary.appendChild(list);

      data.stages.forEach((stage, stageIndex) => {
        const item = document.createElement("li");
        item.className = "cro-lineage-stage";
        item.dataset.stage = stage.id;

        const head = document.createElement("div");
        head.className = "cro-lineage-stage__head";
        const index = document.createElement("span");
        index.className = "cro-lineage-stage__index";
        index.textContent = String(stageIndex + 1).padStart(2, "0");
        const heading = document.createElement("div");
        const title = document.createElement("h3");
        title.className = "cro-lineage-stage__title";
        title.textContent = stage.title;
        const system = document.createElement("p");
        system.className = "cro-lineage-stage__system";
        system.textContent = stage.system;
        heading.append(title, system);
        head.append(index, heading);
        item.appendChild(head);

        const nodeList = document.createElement("div");
        nodeList.className = "cro-lineage-stage__nodes";
        stage.nodes.forEach((node) => {
          const nodeEl = document.createElement("article");
          nodeEl.className = "cro-lineage-node";
          const name = document.createElement("div");
          name.className = "cro-lineage-node__name";
          name.textContent = node.name;
          name.dataset.tip = LINEAGE_NODE_NAME_TIPS[stage.id] || "当前血缘层的节点名称。";
          const id = document.createElement("code");
          id.className = "cro-lineage-node__id";
          id.textContent = node.id;
          id.dataset.tip = LINEAGE_NODE_ID_TIP;
          const detail = document.createElement("p");
          detail.className = "cro-lineage-node__detail";
          detail.textContent = node.detail;
          nodeEl.append(name, id, detail);
          nodeList.appendChild(nodeEl);
        });
        item.appendChild(nodeList);
        stages.appendChild(item);
      });
      drawer.hidden = false;
    };

    closeButton?.addEventListener("click", (event) => {
      event.stopPropagation();
      close();
    });

    return {
      open, close,
      get isOpen() { return Boolean(drawer && !drawer.hidden); },
      get currentKey() { return currentKey; },
    };
  }

  /* 运行态事件与 training-monitoring-v2 的问题一/问题二同源。事件保留自己的
     性能语义；scope 描述“本次运行实际涉及谁”，不覆盖静态配置映射公式。
     evidence 是本事件的「内涵」：一张证据图 + 几个关键读数，落在详情下区。 */
  const INCIDENT_GROUPS = [
    {
      id: "problem-2",
      name: "问题2 · Router 溢出与通信死锁",
      context: { layers: [38], experts: [193], ranks: [1559], segments: ["moe"] },
      // 桥接句只交代「这是哪条问题线上的哪一步、怎么传的」。传播源与最大影响
      // 各自挂在画布上对应角色的标题下，这里不再重复一遍。
      bridge: (event) => `${event.title}（${event.time}）· 沿“${event.path}”传导`,
      events: [
        {
          id: "p1-warning", time: "15k", dimension: "数值 · 预警", title: "Loss scale 连续衰减",
          focus: { kind: "layer", layer: 38 }, origin: { layers: [38], segments: ["moe"] },
          lineage: { operator: "Layer 38 Router", layer: 38, segment: "moe", bar: "gate", stages: ["model"] },
          victim: { layers: [38], segments: ["moe"] },
          conclusion: "Layer 38 的数值健康已提前恶化，AMP scaler 从 65536 衰减到 4096。",
          root: "Router 输出的数值分布右移，AMP scaler 连续四次减半", path: "Layer 38 → AMP scaler 三级预警",
          impact: "异常仍关在本层内，还没传到通信侧——距离崩溃尚有 53 step",
          evidence: {
            chart: {
              kind: "line", title: "AMP loss scale 逐次减半", unit: "",
              x: ["step 14800", "14900", "15000", "15100", "15150"],
              threshold: { value: 8192, label: "三级预警 · 8192", direction: "below" },
              values: [65536, 65536, 32768, 32768, 16384, 8192, 8192, 4096],
              mark: { index: 7, label: "4096 · 三级预警", tone: "warning" },
              note: "scale 每减半一次，就是一次梯度溢出后的回退；连续 4 次说明数值分布已整体右移，不是偶发。",
            },
            metrics: [
              { label: "衰减级数", value: "4 级（65536 → 4096）" },
              { label: "观察窗口", value: "400 step" },
              { label: "相对崩溃的提前量", value: "53 step" },
            ],
          },
        },
        {
          id: "p1-nan", time: "15203", dimension: "耗时 · 数值", title: "Loss NaN / grad_norm Inf",
          focus: { kind: "layer", layer: 38 }, origin: { layers: [38], segments: ["moe"] },
          lineage: { operator: "Layer 38 Router", layer: 38, segment: "moe", bar: "gate", stages: ["model"] },
          propagation: { stages: [3] },
          victim: { layers: [34,35,36,37,38,39,40,41,42,43,44,45], ranks: "stage" },
          conclusion: "异常只在多卡复现，Layer 38 是首个数值病灶候选。",
          root: "Router logits 越界成 Inf，反向传播时梯度随之溢出", path: "Layer 38 → 梯度 Inf → Loss NaN",
          impact: "本轮迭代的梯度整段作废，训练无法继续收敛",
          evidence: {
            // loss 与 grad_norm 量纲差三个数量级，不并到一根轴上：曲线只画
            // grad_norm，loss 的状态进读数区。
            chart: {
              kind: "line", title: "grad_norm 末段指数级发散", unit: "",
              x: ["step 15196", "15198", "15200", "15202", "15203"],
              threshold: { value: 10, label: "正常波动上界 · 10" },
              values: [1.2, 1.3, 1.6, 2.4, 4.1, 12.7, 86, 860],
              mark: { index: 7, label: "860 → 下一步 Inf", tone: "danger" },
              note: "最后 4 个 step 每步涨约一个数量级，越界发生在同一层的反向传播里。",
            },
            metrics: [
              { label: "loss", value: "NaN", tone: "danger" },
              { label: "grad_norm", value: "Inf", tone: "danger" },
              { label: "多卡复现", value: "8 / 8 次" },
              { label: "单卡复现", value: "0 / 8 次" },
            ],
          },
        },
        {
          id: "p1-log", time: "+8ms", dimension: "通信 · 日志", title: "Plog 暴露 buffer 失配",
          focus: { kind: "rank", rank: 1559 }, origin: { ranks: [1559] },
          lineage: { operator: "EP Dispatch / All-to-All", layer: 38, segment: "moe", bar: "a2a_dispatch", rank: 1559, stages: ["runtime", "kernel"], originNode: { stage: "runtime", index: 0 } },
          propagation: { layers: [38], experts: [193], segments: ["moe"] },
          victim: { ranks: [1559] },
          conclusion: "运行时 EP rank 23 的 send=0、recv=9832；通信报错同时携带 router_logits Inf 证据。",
          root: "all-to-all 的收发量对不上：send=0、recv=9832", path: "Rank 1559 buffer 失配 → 通信阻塞",
          impact: "第一个卡住的就是它自己（PP3 / EP23），此刻尚未波及同组其他卡",
          evidence: {
            chart: {
              kind: "bars", title: "rank 1559 的 all-to-all 收发量", unit: " token",
              items: [
                { label: "send（本卡发出）", value: 0 },
                { label: "recv（本卡待收）", value: 9832, tone: "danger" },
                { label: "同组正常卡均值", value: 154 },
              ],
              note: "一发一收本该同量级。send 归零、recv 堆到 60 倍，说明这张卡被路由指定成了唯一收方。",
            },
            metrics: [
              { label: "运行时 EP rank", value: "23" },
              { label: "global rank", value: "1559（PP3 / EP23）" },
              { label: "日志同时携带", value: "router_logits = Inf", tone: "danger" },
            ],
          },
        },
        {
          id: "p1-a2a", time: "+30s", dimension: "通信 · 耗时", title: "All-to-all 超时，63 rank 空等",
          focus: { kind: "rank", rank: 1559 }, origin: { ranks: [1559] },
          lineage: { operator: "HCCL All-to-All", layer: 38, segment: "moe", bar: "a2a_dispatch", rank: 1559, stages: ["runtime", "kernel"] },
          propagation: { layers: [38], experts: [193], segments: ["moe"], ranks: "ep-stage" },
          victim: { ranks: "ep-stage-peers" },
          conclusion: "EP rank 23 是首个阻塞者，其余 63 个 EP rank 是 barrier 受害者，不应被判为 64 个独立根因。",
          root: "recv 过载，迟迟进不了 all-to-all barrier", path: "Rank 1559 → All-to-all barrier → 63 rank 空等",
          impact: "同组其余成员在 barrier 上空等 30 s，整个 EP 通信组停止前进",
          evidence: {
            chart: {
              kind: "stack", title: "EP 通信组 64 张卡的角色构成", unit: " 卡",
              items: [
                { label: "阻塞者（rank 1559）", value: 1, tone: "danger" },
                { label: "barrier 空等", value: 63, tone: "warning" },
              ],
              note: "64 张卡同时报 timeout，但只有 1 张是原因、63 张是结果——按报错数排根因会把结论整个搞反。",
            },
            metrics: [
              { label: "HCCL 超时阈值", value: "30 s" },
              { label: "空等卡数", value: "63 / 64" },
              { label: "误判为独立根因", value: "64 个", tone: "warning" },
            ],
          },
        },
        {
          id: "p1-root", time: "-30s", dimension: "数值 · 负载", title: "Router FP8 溢出，E193 吸收 98% token",
          focus: { kind: "segment", segment: "moe", bar: "gate", scopeLayer: 38, deckNode: "gate" },
          lineage: { operator: "Layer 38 Router", layer: 38, segment: "moe", bar: "gate", rank: 1559, stages: ["model", "fx", "ge", "runtime", "kernel"], originNode: { stage: "model", index: 0 }, victimStage: "runtime" },
          origin: { layers: [38], segments: ["moe"] },
          propagation: { ranks: [1559] },
          victim: { experts: "all" },
          conclusion: "这是问题2的根因事件：FP8 softmax 溢出导致路由塌缩，而不是 HCCL 自身故障。",
          root: "Router 的 max(logits)=1846，FP8 下 exp() 直接溢出成 Inf", path: "Router → Expert 193（98% token）→ EP rank 23",
          impact: "路由塌缩，98% token 全压到 E193，其余 247 个再没收到过 token",
          evidence: {
            chart: {
              kind: "bars", title: "Layer 38 本 step 的 token 路由份额", unit: "%",
              threshold: { value: 20, label: "单专家健康上限 · 20%" },
              items: [
                { label: "E193", value: 98, tone: "danger" },
                { label: "其余 8 个活跃专家", value: 2 },
                { label: "247 个 dead expert", value: 0 },
              ],
              note: "在一个 batch / 观测窗口内，负载均衡应避免 token 长期塌缩到单个专家；softmax 出现 Inf 后，路由选择持续落在 E193。",
            },
            metrics: [
              { label: "max(router logits)", value: "1846", tone: "danger" },
              { label: "FP8 E4M3 可表示上限", value: "448" },
              { label: "exp(logits)", value: "Inf → softmax 塌缩", tone: "danger" },
              { label: "dead expert", value: "247 / 256" },
            ],
          },
        },
        {
          id: "p1-spread", time: "+30.1s", dimension: "通信 · 扩散", title: "PP3 断裂，2048 NPU hang",
          focus: { kind: "stage", stage: 3 }, origin: { ranks: [1559] },
          lineage: { operator: "EP Barrier / PP3 等待链", layer: 38, segment: "moe", bar: "a2a_dispatch", rank: 1559, stages: ["runtime", "kernel"] },
          propagation: { stages: [3], ranks: "stage" }, victim: { ranks: "all" },
          conclusion: "报错点是通信 timeout，异常震中却在 Layer 38 Router；单点经 EP barrier 和 PP 依赖扩散至整网。",
          root: "all-to-all 就阻塞在这里，它是整网停摆的起点", path: "Expert 193 过载 → Rank 1559 阻塞 → EP barrier → PP3 断裂 → 全网等待",
          impact: "沿 EP barrier 与 PP 依赖逐级传导，4 个 stage 全部停在等待上",
          evidence: {
            chart: {
              kind: "stack", title: "2048 张卡按「离震中多远」的构成", unit: " 卡",
              items: [
                { label: "直接阻塞", value: 1, tone: "danger" },
                { label: "同 EP 组空等", value: 63, tone: "warning" },
                { label: "同 stage 其余", value: 448 },
                { label: "其他 stage 等待", value: 1536, tone: "good" },
              ],
              note: "99.95% 的卡是被依赖链拖住的，它们的 timeout 日志与根因无关——扩散范围大不等于根因分散。",
            },
            metrics: [
              { label: "受影响 PP stage", value: "4 / 4" },
              { label: "受影响 NPU", value: "2048" },
              { label: "首个阻塞卡", value: "global rank 1559", tone: "danger" },
              { label: "震中", value: "Layer 38 Router（非 HCCL）" },
            ],
          },
        }
      ]
    },
    {
      id: "problem-1",
      name: "问题1 · 显存峰值与碎片 OOM",
      context: {
        layers: [34,35,36,37,38,39,40,41,42,43,44,45],
        stages: [3], experts: "all", epRanks: "all", ranks: "stage", segments: ["moe", "head"]
      },
      bridge: (event) => `${event.title}（${event.time}）· 沿“${event.path}”传导`,
      events: [
        {
          id: "p2-rise", time: "8000+", dimension: "显存 · 趋势", title: "显存从 55 GB 持续爬升",
          focus: { kind: "stage", stage: 3 },
          lineage: { operator: "PP3 激活生命周期", layer: 38, stages: ["model"] },
          origin: { layers: [34,35,36,37,38,39,40,41,42,43,44,45], segments: ["moe"] },
          victim: { layers: [34,35,36,37,38,39,40,41,42,43,44,45], segments: ["head"] },
          conclusion: "PP stage 3 的显存不再回落，吞吐同期下降 12.5%。",
          root: "这一段的激活自前向起就常驻不释放，显存只涨不落", path: "46 层激活常驻 → 显存持续爬升",
          impact: "叠上 LM Head 的 logits 后，本 stage 的显存余量被吃到见底",
          evidence: {
            // 显存(GB) 与吞吐(tokens/s) 不并轴：曲线画显存，吞吐进读数区。
            chart: {
              kind: "line", title: "PP stage 3 显存占用（step 8000 → 12000）", unit: " GB",
              x: ["step 8000", "9000", "10000", "11000", "12000"],
              threshold: { value: 60.8, label: "95% 阈值 · 60.8 GB" },
              values: [55, 56.2, 57.9, 59.4, 60.8, 62.1, 63, 63.7],
              // 不再单点直标：95% 警戒线 + 越线那截红色面积已经把「涨到哪儿了」说清楚了
              note: "每个 step 之间不再回落到基线——说明被留住的不是临时 buffer，而是一直活着的激活。",
            },
            metrics: [
              { label: "起始 / 当前", value: "55 → 63.7 GB" },
              { label: "吞吐", value: "3200 → 2800 tokens/s（−12.5%）", tone: "warning" },
              { label: "未回落持续", value: "4000 step" },
            ],
          },
        },
        {
          id: "p2-cost", time: "12000", dimension: "耗时 · 显存", title: "分配/释放 API 占时 7.4%",
          focus: { kind: "stage", stage: 3 }, origin: { layers: [38], segments: ["moe"] },
          lineage: { operator: "显存分配 / 释放 API", stages: ["runtime"] },
          propagation: { layers: [34,35,36,37,38,39,40,41,42,43,44,45] },
          victim: { ranks: "stage" },
          conclusion: "显存管理耗时 890 ms，明显高于正常值 2%；带宽利用率 78%，可排除纯带宽瓶颈。",
          root: "分配器在这一层反复做碎片整理与换页", path: "碎片整理 / 换页 → step 耗时增加",
          impact: "每个 step 多花 890 ms 在显存管理上，吞吐从 3200 掉到 2800 tokens/s",
          evidence: {
            chart: {
              kind: "stack", title: "单 step 12.0 s 的耗时构成", unit: " s",
              items: [
                { label: "计算", value: 8.9 },
                { label: "通信", value: 2.21, tone: "alt" },
                // limitShare：这一段本该只占 2%，超出的部分在条上单独标红
                { label: "显存分配 / 释放", value: 0.89, tone: "danger", limitShare: 2 },
              ],
              note: "分配释放本该是 2% 量级的边角开销，这里占到 7.4%——斜纹那截就是多出来的碎片整理与换页。",
            },
            metrics: [
              { label: "显存管理耗时", value: "890 ms（7.4%）", tone: "danger" },
              { label: "正常水位", value: "约 2%" },
              { label: "HBM 带宽利用率", value: "78%（非带宽瓶颈）" },
            ],
          },
        },
        {
          id: "p2-peak", time: "12000", dimension: "显存 · 容量", title: "激活值占用 36.2 GB",
          focus: { kind: "stage", stage: 3 }, origin: { layers: [38], segments: ["moe"] },
          lineage: { operator: "PP3 激活值生命周期", layer: 38, stages: ["model"] },
          propagation: { segments: ["moe", "head"] },
          victim: { segments: ["moe", "head"] },
          conclusion: "激活值占峰值的 56.6%，是唯一可大幅缩减的组成。",
          root: "激活在反向用到之前一直留在显存里，逐层累加不释放", path: "逐层激活累积 → Stage 3 叠加 LM Head logits",
          impact: "两段合计 36.2 GB 激活，占满 64 GB 峰值的 56.6%，容量再无余量",
          evidence: {
            /* 用等距容器而不是构成条：这张图的分母是**一张卡的 64 GB**，不是各段
               之和。装满了就是 OOM，平面色条表达不了「框满了」这件事。
               items 自底向上摞，读法与 Cluster 区「单卡容量」栏完全一致。 */
            chart: {
              kind: "capacity", title: "64 GB 显存峰值的构成", unit: " GB", cap: 64,
              items: [
                { label: "权重", value: 14.1 },
                { label: "优化器状态", value: 9.8, tone: "good" },
                { label: "激活值", value: 36.2, tone: "warning" },
                { label: "碎片空洞", value: 3.9, tone: "danger", void: true },
              ],
              note: "权重与优化器状态由并行切分固定，改不动；能靠重计算换回来的只有那 36.2 GB 激活。",
            },
            metrics: [
              { label: "峰值 / 容量", value: "64.0 / 64 GB", tone: "danger" },
              { label: "激活占比", value: "56.6%" },
              { label: "安全余量", value: "0 GB", tone: "danger" },
            ],
          },
        },
        {
          id: "p2-layer", time: "12000", dimension: "显存 · Layer", title: "L38 单层激活达到 1.2 GB",
          focus: { kind: "layer", layer: 38 }, origin: { layers: [38], segments: ["moe"] },
          lineage: { operator: "Layer 38 Expert Dispatch Buffer", layer: 38, segment: "moe", bar: "a2a_dispatch", stages: ["model"] },
          propagation: { stages: [3] },
          victim: { layers: [34,35,36,37,38,39,40,41,42,43,44,45], segments: ["head"] },
          conclusion: "Layer 38 比普通 Dense 层高 1.7 倍，额外占用来自 expert dispatch buffer。",
          root: "expert dispatch buffer 让它的激活比同段普通层高 1.7 倍，单层 1.2 GB", path: "Layer 38 → PP stage 3 → 峰值叠加",
          impact: "连同 LM Head 一起，把这个 stage 顶成全网最重的一段",
          evidence: {
            chart: {
              kind: "bars", title: "PP stage 3 逐层激活占用", unit: " GB",
              threshold: { value: 1, label: "异常线 · 1.0 GB" },
              items: [
                { label: "L34", value: 0.7 }, { label: "L35", value: 0.72 },
                { label: "L36", value: 0.69 }, { label: "L37", value: 0.71 },
                { label: "L38", value: 1.2, tone: "danger" },
                { label: "L39", value: 0.73 }, { label: "L40", value: 0.7 },
                { label: "L41", value: 0.72 }, { label: "L42", value: 0.71 },
                { label: "L43", value: 0.7 }, { label: "L44", value: 0.73 },
                { label: "L45", value: 0.71 },
              ],
              note: "12 层里只有 L38 突出，其余层彼此在 ±3% 内——多出来的 0.5 GB 有明确出处，不是统计噪声。",
            },
            metrics: [
              { label: "L38 / 同段普通层", value: "1.20 vs 0.71 GB（×1.7）", tone: "danger" },
              { label: "额外占用来源", value: "expert dispatch buffer" },
              { label: "本段 12 层合计", value: "9.0 GB" },
            ],
          },
        },
        {
          id: "p2-oom", time: "12003", dimension: "显存 · OOM", title: "EP rank 17（global rank 1553）触顶并发生碎片 OOM",
          focus: { kind: "rank", rank: 1553 }, origin: { ranks: [1553] },
          lineage: { operator: "0.5 GB 临时 Buffer 申请", rank: 1553, stages: ["runtime"], originNode: { stage: "runtime", index: 0 }, victimStage: "runtime" },
          propagation: { stages: [3], ranks: "stage" }, victim: { ranks: "all" },
          conclusion: "64/64 GB 容量不足是主因，83% 碎片率让 0.5 GB 临时 buffer 更早申请失败。",
          root: "64 GB 占满，0.5 GB 的临时 buffer 申请失败", path: "Rank 1553 OOM → PP3 中断 → 全网等待",
          impact: "它一崩 PP3 就断，全网跟着停在等待上",
          evidence: {
            /* 与上一张同款等距容器（同一个 builder），两张图并排看就是「装满了」
               和「装满之后长什么样」。
               空闲的 3.9 GB 在这里拆成两段：3.58 的零碎 + 0.32 的最大连续块。原先
               三段并列（60.1 + 3.9 + 0.32）把最大块又数了一遍，合计 64.32 GB 超过
               容量本身；拆开之后总量正好是 64，而且盒顶那道薄片就是「最大的一个洞
               只有这么薄」——这正是这条证据要说的话。 */
            chart: {
              kind: "capacity", title: "rank 1553 触顶时的 64 GB 分布", unit: " GB", cap: 64,
              items: [
                { label: "已分配", value: 60.1 },
                { label: "碎片空洞 · 零碎", value: 3.58, tone: "danger", void: true },
                { label: "碎片空洞 · 最大连续块", value: 0.32, tone: "warning", void: true },
              ],
              note: "空闲共 3.9 GB（3.58 零碎 + 0.32 最大连续块）＞ 申请量 0.5 GB，申请照样失败——决定成败的是最大连续块，不是空闲总量。",
            },
            metrics: [
              { label: "容量", value: "64 / 64 GB", tone: "danger" },
              { label: "碎片率", value: "83%", tone: "danger" },
              { label: "失败的申请", value: "0.5 GB 临时 buffer" },
              { label: "首个失败卡", value: "global rank 1553（EP17）" },
            ],
          },
        }
      ]
    }
  ].sort((a, b) => a.id.localeCompare(b.id));

  /* ── 页面接线 ─────────────────────────────────────────────────────────── */
  function boot() {
    installTipLayer();
    const controller = createController();
    controller.mount(document.getElementById("croParallelSteppers"), "parallel");
    controller.mount(document.getElementById("croMoeSteppers"), "moe");
    controller.mount(document.getElementById("croClusterSteppers"), "cluster");
    // Micro Batch / Seq Length：与卡型号一起决定单卡装多少，故排在下拉之后
    controller.mount(document.getElementById("croBatchSteppers"), "batch");

    /* 卡型号：硬件属性，和 Total Rank / Node 同属 Cluster 区。两个选项而不是
       stepper —— 型号是枚举不是量，±键在两项之间来回跳读不出「选了哪个」。
       它只影响单卡容量框的高度（CARD_SPECS[].hbmGB）与口径说明，不进 world_size。 */
    (() => {
      const select = document.getElementById("croCardSelect");
      if (!select) return;
      select.innerHTML = "";
      CARD_ORDER.forEach((id) => {
        const spec = CARD_SPECS[id];
        const option = document.createElement("option");
        option.value = id;
        // 容量直接写进选项：选卡的当下就是在选容量框的高度，不该等到看口径才知道
        option.textContent = `${spec.short}（${spec.hbmGB}G）`;
        option.title = `${spec.label} · ${spec.hbmGB} GB HBM · ${spec.hbmNote}`;
        if (id === controller.config.card) option.selected = true;
        select.appendChild(option);
      });
      const syncTitle = () => {
        const spec = CARD_SPECS[select.value] || CARD_SPECS[DEFAULT_CARD];
        select.title = `${spec.label} · ${spec.hbmGB} GB HBM（${spec.hbmNote}）`;
      };
      syncTitle();
      select.addEventListener("change", () => {
        controller.set("card", select.value);
        syncTitle();
      });
    })();

    /* EP 口径：与卡型号、模型一样是"不进 stepper"的配置，但它改的是 world 的
       公式本身（EP 进不进乘积），所以走 controller.setEpMode() 单独一条通路。
       两枚键而不是下拉：只有两档，且切换的后果要能一眼比出来。 */
    (() => {
      const group = document.getElementById("croEpMode");
      if (!group) return;
      const buttons = Array.from(group.querySelectorAll("[data-ep-mode]"));
      const sync = () => {
        const current = controller.config.moeOrthogonal ? "orthogonal" : "split";
        buttons.forEach((button) => {
          const on = button.dataset.epMode === current;
          button.classList.toggle("is-selected", on);
          button.setAttribute("aria-pressed", on ? "true" : "false");
        });
      };
      sync();
      group.addEventListener("click", (event) => {
        const button = event.target.closest?.("[data-ep-mode]");
        if (!button || !group.contains(button)) return;
        controller.setEpMode(button.dataset.epMode === "orthogonal");
        sync();
      });
    })();

    /* 模型：整网 / 典型 Layer / MoE 区 / Cluster 四域全部跟着重派。与卡型号不同，
       这不是单字段调整 —— 见 createController 里 setModel() 的注释。 */
    (() => {
      const select = document.getElementById("croModelSelect");
      if (!select) return;
      select.value = controller.config.model;
      select.addEventListener("change", () => controller.setModel(select.value));
    })();

    /* 整网图 → 其余视图：点 deck 里的算子节点，反查成结构条的 (segment, bar)
       再走同一条 emitSelect 通路，与其他三个方向完全对称。 */
    const deck = createDeck("croDeckHost", {
      onNodeSelect(selected) {
        if (!selected) return;
        const topology = controller.topology;
        let hit = deckNodeIndex(topology).get(selected.nodeId);
        const layer = Number.isFinite(selected.layer) ? selected.layer : null;

        // attention_core / post_mlp_norm 这类节点 Dense 和 MoE 两列都有，
        // 用节点所在层的 FFN 类型消歧，别一律落到先注册的那一列
        if (hit && layer !== null) {
          const ffn = topology.layers[layer]?.ffn;
          if (ffn && (hit.segment === "dense" || hit.segment === "moe") && hit.segment !== ffn) {
            const col = activeColumns(topology).find((c) => c.id === ffn);
            const bar = col && col.bars.find((b) => b.deckNode === selected.nodeId);
            if (bar) hit = { segment: col.id, bar: bar.id, experts: bar.experts || null, layers: col.layers };
          }
        }

        if (hit) {
          // 不传 scopeLayer：整网图的一个算子（EP Combine、Attn…）在同类型的
          // 每一层都存在，直接点它就该亮出整列的层。要收敛到单层得先在 Layer
          // 导航里选中那层，再点算子 —— 这条收敛规则统一由 emitSelect 施加，
          // 与结构条的点击路径保持同一套语义（select.png 的
          //「EP Combine in Layer 3」正是先选层后点条）。
          emitSelect({
            kind: "segment", segment: hit.segment, bar: hit.bar,
            deckNode: selected.nodeId, experts: hit.experts, layers: hit.layers,
            preferLayer: layer,   // deck 停在用户正看的那一层，不跳走
          });
        } else if (layer !== null) {
          emitSelect({ kind: "layer", layer });
        }
      },
    });
    const layerNav = document.getElementById("croLayerNav");
    const structure = document.getElementById("croStructure");

    const linkLayer = document.getElementById("croLinkLayer");
    let relation = null;
    let railLayoutTimer = 0;

    function emitSelect(payload) {
      const topology = controller.topology;
      if (!payload?.incidentId) {
        activeIncident = null;
        setIncidentLayout(false);
        clearIncidentBanner();
        document.querySelectorAll(".cro-event").forEach((button) => {
          button.classList.remove("is-selected");
          button.setAttribute("aria-pressed", "false");
        });
      }
      // 「先选层、再点算子条」时把结构条收敛到那一层（select.png 的 EP Combine in Layer 3）
      if (payload && payload.kind === "segment" && !Number.isFinite(payload.scopeLayer)) {
        const prev = relation && relation.primary;
        if (prev && prev.kind === "layer" && (payload.layers || []).includes(prev.layer)) {
          payload = { ...payload, scopeLayer: prev.layer };
        }
      }
      relation = payload ? resolveRelation(topology, payload) : null;
      applyRelation(relation);
      document.dispatchEvent(new CustomEvent("cro:select", { detail: relation }));
    }

    function clearSelection() { emitSelect(null); }

    /* 配置一改，四域整块重建：选中/关联的 class 挂在旧 DOM 上，跟着一起没了，
       而关系连线画在独立的 overlay 上、不随重建消失 —— 于是留下「线还在、
       高亮没了」这种半截状态。何况关系集本来就是配置的函数（rank 集合跟着
       TP/DP/EP 走），调完 TP 旧连线指向的已经是另一批卡，光补高亮也不对。
       所以按新 topology 把同一个选择重解析一遍再铺；选中对象在新配置里已经
       不存在（层数/卡数/专家数被调小）就整体清空，不留悬空高亮。 */
    function reapplySelection(topology) {
      const p = relation?.primary;
      if (!p) return;
      const c = topology.counts;
      const survives = {
        layer: () => p.layer < c.totalLayer,
        stage: () => p.stage < c.pp,
        rank: () => p.rank < c.totalRank,
        expert: () => p.expert < c.routedExpert,
        epRank: () => p.epRank < c.ep,
        sharedExpert: () => p.shared < c.sharedExpert,
      }[p.kind];
      if (survives && !survives()) { clearSelection(); return; }
      const next = { ...p };
      // 专家 ↔ EP rank 的归属是 (routedExpert, ep) 的函数，配置一变就得重算，
      // 否则拿旧 epRank 去点亮新分组，亮的是别的组
      if (next.kind === "expert") next.epRank = topology.epRankOfExpert(next.expert);
      if (next.kind === "epRank") next.experts = topology.expertsOfEpRank(next.epRank);
      emitSelect(next);
    }

    function redrawLinks() {
      // 事件模式下四域整块隐藏，没有可连的锚点：连线直接收掉，否则 scroll/resize
      // 会把线画到 0×0 的隐藏元素上（退化成射向视口左上角的射线）。
      if (activeIncident) {
        linkLayer?.replaceChildren();
        return;
      }
      drawRelationLinks(linkLayer, relation);
    }

    // deck 当前实际处在的状态，用于去重（见 applyRelation 里的 deck.silently）。
    // deck 重建后 DOM 是全新的，这两个缓存要一并作废。
    let deckFrontLayer = null;
    let deckNodeKey = null;

    let rankCellMap = new Map();
    let rankCellMapHost = null;
    let rankCellMapFirst = null;
    let paintedRankCells = new Set();
    let paintedSelectedRankCell = null;

    function currentRankCellMap() {
      const host = document.getElementById("croHeat");
      const cells = host?.querySelectorAll(".twin-heat-cell") || [];
      // Cluster 在配置变化时会整体重建；只在宿主或子节点数量变化时重建索引。
      if (host !== rankCellMapHost || cells.length !== rankCellMap.size || cells[0] !== rankCellMapFirst) {
        rankCellMapHost = host;
        rankCellMapFirst = cells[0] || null;
        rankCellMap = new Map(Array.from(cells, (cell) => [Number(cell.dataset.rank), cell]));
        paintedRankCells = new Set();
        paintedSelectedRankCell = null;
      }
      return rankCellMap;
    }

    /* 把关系集铺到四个视图。selected = 用户点中的那一个，related = 被它牵连出来的。
       rel 为 null 表示清空，回到「默认不预选、不高亮」的静息态。 */
    function applyRelation(rel) {
      const p = rel ? rel.primary : null;
      // 收关系集时必须传属性名而不是 rel.xxx —— 后者是实参，会在 has 执行
      // 之前就求值，rel 为 null（清空）时第一次调用就 TypeError，整个清空中断。
      const has = (key, v) => Boolean(rel) && rel[key].has(v);
      const board = document.getElementById("croBoard");
      board?.classList.toggle("is-focused", Boolean(rel));

      // ── Layer 导航 ──
      layerNav?.querySelectorAll(".cro-tick").forEach((tick) => {
        // 端点刻度（Emb / Norm / Head）没有 layer，按 segment 匹配
        if (tick.dataset.unit) {
          const unit = tick.dataset.unit;
          const selected = Boolean(rel) && rel.unit === unit;
          tick.classList.toggle("is-selected", selected);
          // 端点刻度以前恒为 false：点 stage / rank 时 Emb / Norm / Head 明明
          // 在那段流水线上，刻度带上却是灰的
          tick.classList.toggle("is-related", !selected && Boolean(rel) && rel.units.has(unit));
          return;
        }
        const l = Number(tick.dataset.layer);
        const selected = Boolean(p) && p.kind === "layer" && l === p.layer;
        tick.classList.toggle("is-selected", selected);
        tick.classList.toggle("is-related", !selected && has("layers", l));
      });
      layerNav?.querySelectorAll(".cro-pp-span").forEach((el) => {
        const s = Number(el.dataset.stage);
        const incident = Boolean(p?.incidentId);
        const selected = !incident && Boolean(p) && p.kind === "stage" && s === p.stage;
        el.classList.toggle("is-selected", selected);
        el.classList.toggle("is-related", !incident && !selected && has("stages", s));
      });
      // Dense / MoE / Emb / Norm / Head 注记：跟着关系集走，不在范围内的整条压暗，
      // 免得选中一层后五个分区名仍是同一亮度、读不出这一层属于哪一段。
      layerNav?.querySelectorAll(".cro-ffn-span").forEach((el) => {
        el.classList.toggle("is-related", Boolean(rel) && rel.segments.has(el.dataset.segment));
      });

      // ── 结构条 ──
      structure?.querySelectorAll(".cro-bar").forEach((bar) => {
        const selected = Boolean(rel && rel.bar)
          && bar.dataset.segment === rel.bar.segment && bar.dataset.bar === rel.bar.bar;
        bar.classList.toggle("is-selected", selected);
      });
      structure?.querySelectorAll(".cro-structure__col").forEach((col) => {
        // 整列点击：被点的那一列进 is-selected（整块底板高亮描边，作连线锚点），
        // 其余被牵连的列仍是 is-related。单算子点击时没有 wholeColumn，全走 is-related。
        const selected = Boolean(rel && rel.wholeColumn) && col.dataset.segment === rel.segment;
        col.classList.toggle("is-selected", selected);
        col.classList.toggle("is-related", !selected && Boolean(rel) && rel.segments.has(col.dataset.segment));
      });

      // ── 层刻度取选中算子的语义色 ──
      // 选中的是单个算子（整网节点 / 典型层算子条，两条通路都会落到同一根
      // .cro-bar 上）时，把它的 op 写到 Layer 导航上，被点亮的那一层或那一组
      // 层就用与算子条完全相同的渐变填充，而不是统一的 --primary 蓝。
      if (layerNav) {
        const op = structure?.querySelector(".cro-bar.is-selected")?.dataset.op;
        if (op) layerNav.dataset.op = op;
        else delete layerNav.dataset.op;
      }

      // ── MoE ──
      // 只扫 board：事件详情的角色卡里也有一整套 .cro-expert / .cro-moe-group，
      // 那是只读证据，不该被静态查询的选中态涂到。
      board?.querySelectorAll(".cro-expert[data-expert]").forEach((dot) => {
        const e = Number(dot.dataset.expert);
        const selected = Boolean(p) && p.kind === "expert" && e === p.expert;
        dot.classList.toggle("is-selected", selected);
        dot.classList.toggle("is-related", !selected && has("experts", e));
      });
      /* 关系集只点到个别 EP 组时（点一张卡 = 它所在的那一个），描边提到与共享
         专家同一档的白：那一次点击里 allShared() 让共享专家亮的是白 1.5px 描边，
         EP 组却只有一档灰边，同一次选择两处亮度差一大截，读起来像 EP 组根本没
         被点上。
         只在「个别」时提亮 —— 点整网节点 / 典型层会走 allEpRanks() 把全部 EP 都
         收进关系集，那时整列几十个组一起白框，等于什么都没突出。判据与关系摘要
         那句「EP0、EP1… / 全部 EP」同源（见 relationSummary 的 rel.epRanks.size
         < c.ep）。 */
      const epPinpoint = Boolean(rel) && rel.epRanks.size > 0
        && rel.epRanks.size < controller.topology.counts.ep;
      board?.querySelectorAll(".cro-moe-group").forEach((group) => {
        const ep = Number(group.dataset.epRank);
        // 点 EP 组名 = 把整组选中：组本身进 is-selected（白描边由 CSS 给），
        // 组内专家仍留 is-related 以免被聚焦降噪压暗，但底色被 CSS 压回中性。
        const selected = Boolean(p) && p.kind === "epRank" && ep === p.epRank;
        const related = !selected && has("epRanks", ep);
        group.classList.toggle("is-selected", selected);
        group.classList.toggle("is-related", related);
        group.classList.toggle("is-pinpoint", related && epPinpoint);
      });
      board?.querySelectorAll(".cro-expert--shared").forEach((chip) => {
        const i = Number(chip.dataset.shared);
        const selected = Boolean(p) && p.kind === "sharedExpert" && i === p.shared;
        chip.classList.toggle("is-selected", selected);
        chip.classList.toggle("is-related", !selected && has("shared", i));
      });

      // ── 集群 ──
      const cellsByRank = currentRankCellMap();
      paintedRankCells.forEach((cell) => cell.classList.remove("is-related"));
      paintedRankCells.clear();
      paintedSelectedRankCell?.classList.remove("is-selected");
      paintedSelectedRankCell = null;

      if (rel) {
        rel.ranks.forEach((rank) => {
          const cell = cellsByRank.get(rank);
          if (!cell) return;
          if (p?.kind === "rank" && rank === p.rank) {
            cell.classList.add("is-selected");
            paintedSelectedRankCell = cell;
          } else {
            cell.classList.add("is-related");
            paintedRankCells.add(cell);
          }
        });
      }

      // ── 整网 deck（回写时静音它的 onNodeSelect，否则会自激成死循环）──
      /* deck 的两个写入都做**幂等去重**：setFrontLayer 内部会遍历全部 46 张层卡、
         逐张重置专家池（replaceChildren + 四条内联几何）再重算边线，正视图下换卡
         还是一次 display:none → block 的整卡重绘。值没变还照写一遍，画面上就是
         连点同一张卡时无谓地闪一下。
         selectNode 与 front layer 绑在一起：换了卡，同名节点要在新卡里重新标。 */
      deck?.silently((api) => {
        if (!api) return;
        const nextLayer = rel && Number.isFinite(rel.deckLayer) ? rel.deckLayer : null;
        // 第二参必须是 undefined 而不是 null —— deck 里判的是
        // Number.isFinite(Number(layer))，Number(null) === 0 会把查找锁进 L0。
        const scope = !rel || rel.deckStatic || nextLayer === null ? undefined : nextLayer;
        // 没有对应算子节点时也要清掉上一次的：正视图下非 front 层是 display:none，
        // 留在旧层里的 .is-selected 节点会退化成 0×0 矩形，collectAnchors 拿到它
        // 之后关系连线就朝视口左上角画出去。
        const nextNode = rel ? (rel.deckNode || null) : null;
        const movedFront = nextLayer !== null && nextLayer !== deckFrontLayer;
        if (movedFront) {
          api.setFrontLayer?.(nextLayer);
          deckFrontLayer = nextLayer;
        }
        const nodeKey = `${nextNode}|${scope}`;
        if (movedFront || nodeKey !== deckNodeKey) {
          api.selectNode?.(nextNode, scope);
          deckNodeKey = nodeKey;
        }
      });
      markDeckRelated(rel);

      // ── 把选中项滚进可视区 ──
      // 典型层的算子条（每列各有一条 44 层长的滚动栈）与路由专家（64 个 EP 组）
      // 都远高于各自的视口，选中项十有八九在折叠区里，不滚出来等于没高亮。
      if (rel) {
        const bar = structure?.querySelector(".cro-bar.is-selected");
        revealIn(bar?.closest(".cro-structure__stack"), bar);

        const routed = document.getElementById("croRoutedExperts");
        // 没有直接选中物时退而露出第一个被牵连的组／专家，至少把关系集的
        // 起点带到眼前（选一层 → 该层用到的 EP 组）
        revealIn(routed, firstMatch(routed, [
          ".cro-expert.is-selected",
          ".cro-moe-group.is-selected",
          ".cro-expert.is-related",
          ".cro-moe-group.is-related",
        ]));

        // 集群矩阵 rank 多到要内部滚动时同理：被点亮的那批卡可能整个在折叠区里
        const clusterView = document.querySelector(".cro-cluster__grid");
        revealIn(clusterView, firstMatch(clusterView, [
          ".twin-heat-cell.is-selected",
          ".twin-heat-cell.is-related",
        ]));
      }

      requestAnimationFrame(redrawLinks);
    }

    const board = document.getElementById("croBoard");
    let activeIncident = null;

    function expandIncidentValues(spec, key, topology) {
      if (!spec || spec[key] == null) return [];
      const value = spec[key];
      if (value === "all") {
        const countByKey = {
          ranks: topology.counts.totalRank,
          experts: topology.counts.routedExpert,
          epRanks: topology.counts.ep,
          layers: topology.counts.totalLayer,
        };
        const count = countByKey[key];
        return Number.isFinite(count) ? Array.from({ length: count }, (_, i) => i) : [];
      }
      if (value === "stage" && key === "ranks") return topology.ranksOfStage(3);
      if (value === "ep-stage" && key === "ranks") {
        const start = topology.rankOf(3, 0, 0);
        return Array.from({ length: topology.counts.ep }, (_, i) => start + i);
      }
      if (value === "ep-stage-peers" && key === "ranks") {
        const start = topology.rankOf(3, 0, 0);
        return Array.from({ length: topology.counts.ep }, (_, i) => start + i).filter((rank) => rank !== 1559);
      }
      return Array.isArray(value) ? value : [];
    }

    function addIncidentScope(rel, event, topology) {
      // 静态查询会把“某层/某算子理论上可关联的全部对象”铺开；运行态事件必须
      // 改用本次采样的实际范围，否则 E193 病灶会误亮全部 256 专家和 2048 rank。
      ["layers", "stages", "experts", "epRanks", "segments", "ranks", "shared", "units", "staticNodes"]
        .forEach((key) => rel[key].clear());
      ["context", "origin", "propagation", "victim"].forEach((role) => {
        const spec = event[role];
        if (!spec) return;
        ["layers", "stages", "experts", "epRanks", "segments", "ranks"].forEach((key) => {
          expandIncidentValues(spec, key, topology).forEach((value) => rel[key].add(value));
        });
      });
      rel.layers.forEach((layer) => rel.stages.add(topology.stageOfLayer(layer)));
      rel.nodes = topology.nodesOfRanks(Array.from(rel.ranks));
      rel.labels = relationLabels(topology, rel, activeColumns(topology));
      return rel;
    }

    /* 事件的角色范围曾经是直接描在四域上的（红框=传播源、橙框=受影响，外加一枚
       跟着震中飘的「传播源」浮标）。四域在事件模式下已整块收起，这套描边随之下线，
       角色范围改由中区两张卡各自重建的域承担。这里只保留「退出事件」时收横幅。 */
    function clearIncidentBanner() {
      const banner = document.getElementById("croIncidentBanner");
      if (banner) banner.hidden = true;
    }

    /* 事件模式与配置仿真模式互斥。
       已发生的运行事件是既成事实：配置表单不可调，四域「点一个对象看它理论上
       牵连谁」的静态查询口径也不成立（范围由本次采样写死）。所以进事件详情就把
       .cro-board 整块收起，换成上（横幅）/ 中（传播源→受影响）/ 下（事件内涵）
       三段视图；关闭横幅或收起运行事件栏再切回来。 */
    function setIncidentLayout(on) {
      const view = document.getElementById("croIncidentView");
      if (view && view.hidden !== !on) view.hidden = !on;
      if (board && board.hidden !== on) {
        board.hidden = on;
        if (!on) {
          // 回配置态时清掉详情内容：角色卡里那两套域各带着 2048 个格子，留着白占内存
          ["croOriginDomains", "croVictimDomains", "croIncidentDetail", "croIncidentLineage"].forEach((id) => {
            const el = document.getElementById(id);
            if (el) el.innerHTML = "";
          });
        }
        // 隐藏期间刻度带量不到宽度，layoutLayerNav 会解出一堆 0 宽刻度。
        // 切回配置态时重算一次，否则 Layer 导航是塌的。
        // 两行的高度分配同理：hidden 期间 syncBoardRows 量不到板子只能原样退回，
        // 中途若改过配置/拉过窗口，留着的就是过期分配（Cluster 差几十像素就会
        // 滚出条来）。等 layout 落定后补量一次。
        // rAF 里跑，顺带避开 refitClusterCells 的 TDZ（它声明在本函数之后）
        if (!on) requestAnimationFrame(() => {
          layoutLayerNav(layerNav);
          syncBoardRows(controller.topology.counts);
          refitClusterCells();
        });
      }
    }

    const setText = (id, text) => {
      const el = document.getElementById(id);
      if (el) el.textContent = text || "";
    };

    /* ── 中区 · 角色卡的域重建 ─────────────────────────────────────────────
       事件的 origin / victim 只可能落在三个域上：Model Architecture（层 / PP 段 /
       结构段）、MoE（专家 / EP 组）、Cluster（rank）。哪个域被触及，就在角色卡里
       用**原样的那一域图形**重建一份 —— 不缩略、不换编码，四域里认得的东西在角色
       卡里还是同一个东西，只是范围被裁到本角色：命中对象按角色色点亮，其余压暗。
       两侧的域清单往往不同（如「L38 Router」→「256 个专家全部塌缩」），这正是
       「事件从哪来、打到哪去」最直接的读法。 */
    const ROLE_DOMAINS = [
      { id: "arch", title: "Model Architecture", keys: ["layers", "stages", "segments"] },
      { id: "moe", title: "MoE", keys: ["experts", "epRanks"] },
      { id: "cluster", title: "Cluster", keys: ["ranks"] },
    ];

    const NOOP = () => {};

    /* deck 的语义色变量要搬到「结构条 / 专家点所在的那棵子树」上。四域在
       .cro-board 里，角色卡里的那几份在 .cro-incident-view 里，两处都要写，
       否则事件模式下的算子条会退回 CSS 兜底色，与整网图对不上。 */
    function syncPalette() {
      const deckRoot = document.getElementById("croDeckHost");
      syncDeckPalette(deckRoot, board);
      syncDeckPalette(deckRoot, document.getElementById("croIncidentView"));
    }

    /* 范围 chip 是那一行的**主语**，后面紧跟着「发生了什么」，所以要短：
       单层直接写 L38（不必再加「· 1 层」），段名用短标签而不是结构条里那种
       带层区间的全名（「MoE x44（L2~L45）」在这里只是噪声）。 */
    const SEGMENT_LABELS = {
      emb: "Emb", dense: "Dense 层", moe: "MoE 层", norm: "Final Norm", head: "LM Head",
    };

    function roleScopeChips(spec, topology) {
      const chips = [];
      const add = (key, text) => { if (text) chips.push({ key, text }); };
      const layers = expandIncidentValues(spec, "layers", topology);
      if (layers.length) {
        add("layers", layers.length === 1
          ? `L${layers[0]}`
          : `${formatRuns(layers, "L")} · ${layers.length} 层`);
      }
      const stages = expandIncidentValues(spec, "stages", topology);
      if (stages.length) add("stages", stages.map((s) => `PP${s}`).join(" + "));
      const segments = expandIncidentValues(spec, "segments", topology);
      if (segments.length) {
        const names = new Map(activeColumns(topology).map((col) => [col.id, col.name]));
        add("segments", segments.map((s) => SEGMENT_LABELS[s] || names.get(s) || s).join(" / "));
      }
      const experts = expandIncidentValues(spec, "experts", topology);
      if (experts.length) add("experts", experts.length > 8 ? `${experts.length} 个专家` : formatRuns(experts, "E"));
      const epRanks = expandIncidentValues(spec, "epRanks", topology);
      if (epRanks.length) add("epRanks", epRanks.length > 8 ? `${epRanks.length} 个 EP 组` : formatRuns(epRanks, "EP"));
      const ranks = expandIncidentValues(spec, "ranks", topology);
      if (ranks.length) add("ranks", ranks.length > 4 ? `${ranks.length} 张卡` : formatRuns(ranks, "rank "));
      return chips;
    }

    function buildRoleDomain(domain, spec, topology, summary) {
      const section = document.createElement("section");
      section.className = "cro-role-domain";
      section.dataset.domain = domain.id;

      /* 域头一行读成一句话：左边的范围 chip 是主语（谁），右边的 summary 是谓语
         （发生了什么）。域名（Model Architecture / MoE / Cluster）不再单列 ——
         chip 本身已经点明了对象类型，多一个分类名只是噪声。 */
      const head = document.createElement("div");
      head.className = "cro-role-domain__head";
      // chip 只挂在描述它的那个域上，免得 Cluster 那行挂一串层号
      roleScopeChips(spec, topology)
        .filter((chip) => domain.keys.includes(chip.key))
        .forEach((chip) => {
          const el = document.createElement("span");
          el.className = "cro-role-domain__chip";
          el.textContent = chip.text;
          head.appendChild(el);
        });
      if (summary) {
        const text = document.createElement("p");
        text.className = "cro-role-domain__summary";
        text.textContent = summary;
        head.appendChild(text);
      }

      const body = document.createElement("div");
      body.className = "cro-role-domain__body";

      if (domain.id === "arch") {
        const nav = document.createElement("div");
        nav.className = "cro-layer-nav";
        renderLayerNav(nav, topology, NOOP);
        body.appendChild(nav);
        // 结构段只在事件确实点名了段时才铺 —— 只涉及层/PP 的事件（如「显存爬升」）
        // 摆一条五段结构条纯属噪声。
        if (spec.segments != null) {
          const structure = document.createElement("div");
          structure.className = "cro-structure";
          renderStructure(structure, topology, NOOP);
          body.appendChild(structure);
        }
      } else if (domain.id === "moe") {
        const routed = document.createElement("div");
        routed.className = "cro-moe-groups";
        renderMoe(null, routed, topology, NOOP);
        body.appendChild(routed);
      } else {
        const heat = document.createElement("div");
        heat.className = "twin-heat cro-heat";
        renderCluster(heat, topology, NOOP);
        body.appendChild(heat);
      }

      section.append(head, body);
      return section;
    }

    /* 命中着色。逐个 querySelectorAll(`[data-x="v"]`) 在 2048 格集群上是 2048 次
       全子树查询；改成「一次拿全、用 Set 判」。 */
    function paintRoleScope(root, spec, topology) {
      const hit = (selector, attr, values, cast) => {
        const set = new Set(values.map(cast));
        if (!set.size) return;
        root.querySelectorAll(selector).forEach((el) => {
          if (set.has(cast(el.dataset[attr]))) el.classList.add("is-hit");
        });
      };
      const segments = expandIncidentValues(spec, "segments", topology);
      hit(".cro-tick[data-layer]", "layer", expandIncidentValues(spec, "layers", topology), Number);
      hit(".cro-pp-span", "stage", expandIncidentValues(spec, "stages", topology), Number);
      hit(".cro-structure__col", "segment", segments, String);
      hit(".cro-ffn-span", "segment", segments, String);
      hit(".cro-expert[data-expert]", "expert", expandIncidentValues(spec, "experts", topology), Number);
      hit(".cro-moe-group", "epRank", expandIncidentValues(spec, "epRanks", topology), Number);
      hit(".twin-heat-cell", "rank", expandIncidentValues(spec, "ranks", topology), Number);
    }

    function renderRoleDomains(host, spec, topology, options = {}) {
      const { focusBar, summary } = options;
      if (!host) return;
      host.innerHTML = "";
      const domains = spec ? ROLE_DOMAINS.filter((d) => d.keys.some((k) => spec[k] != null)) : [];
      if (!domains.length) {
        const empty = document.createElement("span");
        empty.className = "cro-empty";
        empty.textContent = "本次采样未记录该角色的范围";
        host.appendChild(empty);
        return;
      }
      // 这句话说的是整个角色发生了什么，挂在第一个域的头上（该角色的第一行）
      domains.forEach((domain, index) => {
        host.appendChild(buildRoleDomain(domain, spec, topology, index === 0 ? summary : null));
      });
      paintRoleScope(host, spec, topology);
      if (focusBar) {
        const bar = host.querySelector(
          `.cro-bar[data-segment="${focusBar.segment}"][data-bar="${focusBar.bar}"]`,
        );
        // 画布上算子条是全量铺开的，不需要再滚进视口
        if (bar) bar.classList.add("is-hit");
      }
      // 角色卡是只读证据而非查询入口：摘掉 Tab 站（集群网格自己会给首格 tabindex=0），
      // 点击由 CSS 的 pointer-events 挡在按钮上。
      host.querySelectorAll("[tabindex], button").forEach((el) => el.setAttribute("tabindex", "-1"));
    }

    /* ── 中区画布的平移 / 缩放 ────────────────────────────────────────────
       舞台按内容排到自然尺寸（三栏定宽、高度随内容），视口只做取景：
         · 进入事件 / 窗口变化 → fit()，整幅按视口自动缩放并居中
         · 滚轮 → 以指针为锚缩放（指针下那一点保持不动）
         · 拖拽 → 平移
       缩放走 transform，不重排内部那两千多个格子。 */
    const stage = (() => {
      const viewport = document.getElementById("croIncidentFlow");
      const surface = document.getElementById("croIncidentStage");
      if (!viewport || !surface) return { fit() {} };

      const MIN = 0.12, MAX = 2, PAD = 16;
      let scale = 1, tx = 0, ty = 0;

      /* 「连的是哪一坨」：一个域里命中面很小（几个专家、一根算子条）时就接到那几个
         对象本身；命中的是「全部 2048 卡」「256 个专家」这类整体时，接到这个域的
         整块，并把范围虚线圈出来 —— 逐个去并 2048 个格子的包围盒既慢又没意义。 */
      const GROUP_THRESHOLD = 24;

      function anchorGroups(hostId) {
        const host = document.getElementById(hostId);
        if (!host) return [];
        const groups = [];
        host.querySelectorAll(".cro-role-domain").forEach((section) => {
          const hits = section.querySelectorAll(".is-hit");
          if (!hits.length) return;
          if (hits.length > GROUP_THRESHOLD) {
            const body = section.querySelector(".cro-role-domain__body");
            if (body) groups.push({ rect: unionRect([body]), whole: true });
          } else {
            groups.push({ rect: unionRect(Array.from(hits)), whole: hits.length > 1 });
          }
        });
        return groups.filter((g) => g.rect);
      }

      /* 连线画在舞台自身坐标系里，跟着画布一起缩放。只在 fit() 里 transform 归零的
         窗口调用 —— 那时 getBoundingClientRect 读到的才是未缩放的真实几何。 */
      function drawLinks(sw, sh) {
        const svg = document.getElementById("croStageLinks");
        const label = document.getElementById("croIncidentArrow");
        if (!svg) return;
        svg.replaceChildren();
        svg.setAttribute("viewBox", `0 0 ${sw} ${sh}`);

        const base = surface.getBoundingClientRect();
        const toLocal = (r) => ({
          left: r.left - base.left, right: r.right - base.left,
          top: r.top - base.top, bottom: r.bottom - base.top,
        });
        const origin = anchorGroups("croOriginDomains").map((g) => ({ ...g, rect: toLocal(g.rect) }));
        const victim = anchorGroups("croVictimDomains").map((g) => ({ ...g, rect: toLocal(g.rect) }));
        // 两个显示区的栏位边界：路径标签靠它避让，不靠曲线
        const boxOf = (role) => {
          const el = surface.querySelector(`.cro-stage-role[data-role="${role}"]`);
          return el ? toLocal(el.getBoundingClientRect()) : { left: 0, right: 0 };
        };
        const originBox = boxOf("origin"), victimBox = boxOf("victim");
        if (!origin.length || !victim.length) {
          // 某一侧没有命中物（理论上不该发生）：标签退回舞台正中，不留在 auto 位置
          if (label) {
            label.style.left = `${sw / 2}px`;
            label.style.top = `${sh / 2}px`;
          }
          return;
        }

        const spanOf = (groups) => groups.reduce((acc, g) => ({
          left: Math.min(acc.left, g.rect.left), right: Math.max(acc.right, g.rect.right),
          top: Math.min(acc.top, g.rect.top), bottom: Math.max(acc.bottom, g.rect.bottom),
        }), { left: Infinity, right: -Infinity, top: Infinity, bottom: -Infinity });

        const outline = (group, tone) => {
          if (!group.whole) return;
          const r = group.rect;
          const rect = svgNode("rect", {
            class: "cro-stage-link-group",
            x: r.left - 6, y: r.top - 6,
            width: (r.right - r.left) + 12, height: (r.bottom - r.top) + 12,
            rx: 10,
          });
          rect.style.stroke = tone;
          svg.appendChild(rect);
        };
        origin.forEach((g) => outline(g, "var(--danger)"));
        victim.forEach((g) => outline(g, "var(--warning)"));

        const from = spanOf(origin), to = spanOf(victim);
        const HEAD = 9;
        const x1 = from.right + 6, y1 = (from.top + from.bottom) / 2;
        const tipX = to.left - 6, y2 = (to.top + to.bottom) / 2;
        // 曲线在箭头根部收笔，否则圆头笔画会从三角形尖端探出去
        const x2 = tipX - HEAD + 1;
        /* 红→黄渐变按实际端点建（userSpaceOnUse），走向永远跟着这一条线自己的
           两端走，而不是跟着 viewBox 的左右。 */
        const defs = svgNode("defs");
        const gradient = svgNode("linearGradient", {
          id: "croStageLinkGradient", gradientUnits: "userSpaceOnUse",
          x1, y1, x2: tipX, y2,
        });
        [["0", "var(--danger)"], ["1", "var(--warning)"]].forEach(([offset, color]) => {
          const stop = svgNode("stop", { offset });
          stop.style.stopColor = color;
          gradient.appendChild(stop);
        });
        defs.appendChild(gradient);
        svg.appendChild(defs);

        // 控制点水平外推半个跨距：出发与落点都是水平切线，读起来是「流出去/流进来」
        const bend = Math.max(48, (x2 - x1) / 2);
        svg.appendChild(svgNode("path", {
          class: "cro-stage-link",
          d: `M${x1},${y1} C${x1 + bend},${y1} ${x2 - bend},${y2} ${x2},${y2}`,
        }));
        svg.appendChild(svgNode("circle", { class: "cro-stage-link-dot", cx: x1, cy: y1, r: 3 }));
        svg.appendChild(svgNode("path", {
          class: "cro-stage-link-head",
          d: `M${tipX},${y2} L${tipX - HEAD},${y2 - HEAD * 0.58} L${tipX - HEAD},${y2 + HEAD * 0.58} Z`,
        }));

        /* 路径标签避两样东西：
           横向 —— 钉在两个显示区中间的空当里。跟着曲线中点走的话，传播源命中物
           偏左时标签会压到传播源那一列上。
           纵向 —— 整块挪到曲线的上方或下方（挑空间大的一侧），不再骑在线上。
           骑在线上时卡片把曲线截成两截，看着像连线断了。 */
        if (label) {
          const half = label.offsetHeight / 2;
          const curveY = (y1 + y2) / 2;          // 对称控制点下 t=0.5 即两端点中点
          const CLEARANCE = 14;
          const offset = half + CLEARANCE;
          const top = curveY >= sh / 2 ? curveY - offset : curveY + offset;
          label.style.left = `${(originBox.right + victimBox.left) / 2}px`;
          label.style.top = `${Math.min(Math.max(top, half), Math.max(half, sh - half))}px`;
        }
      }

      function apply() {
        surface.style.transform = `translate(${tx.toFixed(1)}px, ${ty.toFixed(1)}px) scale(${scale})`;
        // 底纹跟着平移走，推画布时才有「在一张纸上移动」的感觉
        viewport.style.backgroundPosition = `${tx.toFixed(1)}px ${ty.toFixed(1)}px`;
        setText("croStageZoomReadout", `${Math.round(scale * 100)}%`);
      }

      function fit() {
        // 自然尺寸必须在未缩放时量，否则量到的是上一次缩放后的结果
        surface.style.transform = "none";
        /* 刻度带的分隔线与 PP/FFN 标签位置是 layoutLayerNav 用 getBoundingClientRect
           实测出来、再以 px 写回的 —— 那个读数会被 transform 缩放污染（写回去的值
           被再缩一次，标签整体错位）。所以在这里、transform 已归零的窗口里重排一次。 */
        surface.querySelectorAll(".cro-layer-nav").forEach((nav) => layoutLayerNav(nav));
        const sw = surface.offsetWidth, sh = surface.offsetHeight;
        drawLinks(sw, sh);
        const box = viewport.getBoundingClientRect();
        if (!sw || !sh || !box.width || !box.height) { apply(); return; }
        scale = Math.max(MIN, Math.min(
          (box.width - PAD * 2) / sw,
          (box.height - PAD * 2) / sh,
          1,
        ));
        tx = Math.max(PAD, (box.width - sw * scale) / 2);
        ty = Math.max(PAD, (box.height - sh * scale) / 2);
        apply();
      }

      function zoomAt(clientX, clientY, factor) {
        const box = viewport.getBoundingClientRect();
        const px = clientX - box.left, py = clientY - box.top;
        const next = Math.max(MIN, Math.min(MAX, scale * factor));
        if (next === scale) return;
        tx = px - (px - tx) * (next / scale);
        ty = py - (py - ty) * (next / scale);
        scale = next;
        apply();
      }

      function zoomCenter(factor) {
        const box = viewport.getBoundingClientRect();
        zoomAt(box.left + box.width / 2, box.top + box.height / 2, factor);
      }

      viewport.addEventListener("wheel", (event) => {
        event.preventDefault();
        zoomAt(event.clientX, event.clientY, event.deltaY < 0 ? 1.12 : 1 / 1.12);
      }, { passive: false });

      let drag = null;
      viewport.addEventListener("pointerdown", (event) => {
        // 缩放控件不是画布的一部分
        if (event.target.closest?.(".cro-stage-zoom")) return;
        drag = { x: event.clientX, y: event.clientY, tx, ty, id: event.pointerId };
        viewport.setPointerCapture?.(event.pointerId);
        viewport.classList.add("is-panning");
      });
      viewport.addEventListener("pointermove", (event) => {
        if (!drag) return;
        tx = drag.tx + (event.clientX - drag.x);
        ty = drag.ty + (event.clientY - drag.y);
        apply();
      });
      const endDrag = () => {
        if (!drag) return;
        viewport.releasePointerCapture?.(drag.id);
        drag = null;
        viewport.classList.remove("is-panning");
      };
      viewport.addEventListener("pointerup", endDrag);
      viewport.addEventListener("pointercancel", endDrag);

      document.getElementById("croStageZoomIn")?.addEventListener("click", () => zoomCenter(1.2));
      document.getElementById("croStageZoomOut")?.addEventListener("click", () => zoomCenter(1 / 1.2));
      document.getElementById("croStageZoomFit")?.addEventListener("click", fit);

      return { fit };
    })();

    /* 箭头列：path 拆成链路步骤，再挂上传导中途扫到的范围（propagation）。
       propagation 与 origin/victim 同构，直接复用同一套 chip。 */
    function renderIncidentArrow(event, topology) {
      const chain = document.getElementById("croIncidentArrowChain");
      if (chain) {
        chain.innerHTML = "";
        String(event.path || "").split(/\s*(?:→|->)\s*/).filter(Boolean).forEach((step) => {
          const li = document.createElement("li");
          li.className = "cro-incident-arrow__step";
          li.textContent = step;
          chain.appendChild(li);
        });
      }
      const via = document.getElementById("croIncidentArrowVia");
      if (!via) return;
      via.innerHTML = "";
      const chips = event.propagation ? roleScopeChips(event.propagation, topology) : [];
      if (!chips.length) return;
      const label = document.createElement("span");
      label.className = "cro-incident-arrow__via-label";
      label.textContent = "途经";
      via.appendChild(label);
      chips.forEach((chip) => {
        const el = document.createElement("span");
        el.className = "cro-incident-arrow__via-chip";
        el.textContent = chip.text;
        via.appendChild(el);
      });
    }

    let incidentDetailMode = "detail";

    function setIncidentDetailMode(mode) {
      incidentDetailMode = mode === "lineage" ? "lineage" : "detail";
      const detailTab = document.getElementById("croIncidentDetailTab");
      const lineageTab = document.getElementById("croIncidentLineageTab");
      const detail = document.getElementById("croIncidentDetail");
      const lineage = document.getElementById("croIncidentLineage");
      const showDetail = incidentDetailMode === "detail";
      detailTab?.classList.toggle("is-selected", showDetail);
      detailTab?.setAttribute("aria-selected", String(showDetail));
      lineageTab?.classList.toggle("is-selected", !showDetail);
      lineageTab?.setAttribute("aria-selected", String(!showDetail));
      if (detail) detail.hidden = !showDetail;
      if (lineage) lineage.hidden = showDetail;
      if (showDetail) requestAnimationFrame(paintDetailChart);
      else requestAnimationFrame(() => lineage?.repaintLineage?.());
    }

    function lineageNodeById(data, id) {
      for (const stage of data.stages) {
        const node = stage.nodes.find((entry) => entry.id === id);
        if (node) return { ...node, stage };
      }
      return null;
    }

    function lineageSelection(data, nodeId, recursive) {
      const selected = new Set(nodeId ? [nodeId] : []);
      if (!nodeId) return selected;
      if (!recursive) {
        data.edges.forEach((edge) => {
          if (edge.from === nodeId) selected.add(edge.to);
          if (edge.to === nodeId) selected.add(edge.from);
        });
        return selected;
      }
      const queue = [nodeId];
      while (queue.length) {
        const current = queue.shift();
        data.edges.forEach((edge) => {
          const adjacent = edge.from === current ? edge.to : edge.to === current ? edge.from : null;
          if (adjacent && !selected.has(adjacent)) {
            selected.add(adjacent);
            queue.push(adjacent);
          }
        });
      }
      return selected;
    }

    function lineageRoleNodeIds(data, roleId) {
      const ids = new Set();
      data.stages.forEach((stage) => {
        stage.nodes.forEach((node) => {
          if (node.id && node.roles.some((role) => role.id === roleId)) ids.add(node.id);
        });
        if (stage.roles.some((role) => role.id === roleId)) {
          stage.nodes.forEach((node) => {
            if (node.id) ids.add(node.id);
          });
        }
      });
      return ids;
    }

    function lineagePathBetween(data, origins, victims) {
      if (!origins.size || !victims.size) return new Set();
      const forward = new Set(origins);
      const forwardQueue = Array.from(origins);
      while (forwardQueue.length) {
        const current = forwardQueue.shift();
        data.edges.forEach((edge) => {
          if (edge.from !== current || forward.has(edge.to)) return;
          forward.add(edge.to);
          forwardQueue.push(edge.to);
        });
      }
      const backward = new Set(victims);
      const backwardQueue = Array.from(victims);
      while (backwardQueue.length) {
        const current = backwardQueue.shift();
        data.edges.forEach((edge) => {
          if (edge.to !== current || backward.has(edge.from)) return;
          backward.add(edge.from);
          backwardQueue.push(edge.from);
        });
      }
      const path = new Set(Array.from(forward).filter((id) => backward.has(id)));
      const hasPathEdge = data.edges.some((edge) => path.has(edge.from) && path.has(edge.to));
      return hasPathEdge ? path : new Set();
    }

    function renderLineageInspector(host, data, nodeId, pinned) {
      host.replaceChildren();
      const node = nodeId ? lineageNodeById(data, nodeId) : null;
      if (!node) {
        host.hidden = true;
        return;
      }

      const incoming = data.edges.filter((edge) => edge.to === nodeId);
      const outgoing = data.edges.filter((edge) => edge.from === nodeId);
      if (!incoming.length && !outgoing.length) {
        host.hidden = true;
        return;
      }
      host.hidden = false;
      const connectedText = (edges, side) => edges.length
        ? edges.map((edge) => lineageNodeById(data, edge[side])?.name).filter(Boolean).join("、")
        : "无直接节点";
      const reasons = Array.from(new Set([...incoming, ...outgoing].map((edge) => edge.label)));

      const heading = document.createElement("div");
      heading.className = "cro-incident-lineage__inspector-heading";
      const title = document.createElement("strong");
      title.textContent = node.name;
      const state = document.createElement("span");
      state.textContent = pinned ? "路径已固定" : "当前悬浮";
      heading.append(title, state);

      const grid = document.createElement("dl");
      grid.className = "cro-incident-lineage__inspector-grid";
      const add = (label, value) => {
        const cell = document.createElement("div");
        const dt = document.createElement("dt");
        const dd = document.createElement("dd");
        dt.textContent = label;
        dd.textContent = value;
        cell.append(dt, dd);
        grid.appendChild(cell);
      };
      add("所在层", node.stage.title);
      add("直接输入", connectedText(incoming, "from"));
      add("直接输出", connectedText(outgoing, "to"));
      add("转换依据", reasons.length ? `${reasons.join(" / ")}；${node.detail}` : node.detail);
      host.append(heading, grid);
    }

    function paintIncidentLineageEdges(shell, svg, data, initialSelection = new Set()) {
      const width = shell.offsetWidth;
      const height = shell.offsetHeight;
      if (!width || !height) return;
      svg.replaceChildren();
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
      const shellRect = shell.getBoundingClientRect();
      const scaleX = shellRect.width / width || 1;
      const scaleY = shellRect.height / height || 1;
      const nodes = new Map(Array.from(shell.querySelectorAll("[data-lineage-id]"))
        .map((element) => [element.dataset.lineageId, element]));
      const labels = [];
      data.edges.forEach((edge) => {
        const source = nodes.get(edge.from);
        const target = nodes.get(edge.to);
        if (!source || !target) return;
        const a = source.getBoundingClientRect();
        const b = target.getBoundingClientRect();
        const x1 = (a.right - shellRect.left) / scaleX;
        const y1 = (a.top + a.height / 2 - shellRect.top) / scaleY;
        const x2 = (b.left - shellRect.left) / scaleX;
        const y2 = (b.top + b.height / 2 - shellRect.top) / scaleY;
        const bend = Math.max(12, (x2 - x1) * 0.5);
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`);
        path.classList.add("cro-incident-lineage__edge", `is-${edge.kind}`);
        path.dataset.from = edge.from;
        path.dataset.to = edge.to;
        const isInitiallyActive = initialSelection.has(edge.from) && initialSelection.has(edge.to);
        path.classList.toggle("is-active", isInitiallyActive);
        path.classList.toggle("is-muted", initialSelection.size > 0 && !isInitiallyActive);
        svg.appendChild(path);

        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.classList.add("cro-incident-lineage__edge-label");
        label.setAttribute("x", String((x1 + x2) / 2));
        label.setAttribute("y", String((y1 + y2) / 2 - 5));
        label.setAttribute("text-anchor", "middle");
        label.dataset.from = edge.from;
        label.dataset.to = edge.to;
        label.textContent = edge.label;
        label.classList.toggle("is-active", isInitiallyActive);
        labels.push(label);
      });
      labels.forEach((label) => svg.appendChild(label));
    }

    function appendIncidentLineageRoleTags(host, roles) {
      (roles || []).forEach((role) => {
        const tag = document.createElement("span");
        tag.className = `cro-incident-lineage__role-tag is-${role.id}`;
        tag.textContent = role.label;
        tag.title = role.id === "origin"
          ? "当前事件的传播源"
          : "当前事件中受影响的对象";
        host.appendChild(tag);
      });
    }

    function renderIncidentLineage(host, event, topology) {
      if (!host) return;
      host.replaceChildren();
      const data = buildIncidentLineage(event, topology);

      const trackShell = document.createElement("div");
      trackShell.className = "cro-incident-lineage__track-shell";
      const edgeLayer = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      edgeLayer.classList.add("cro-incident-lineage__edges");
      edgeLayer.setAttribute("aria-hidden", "true");
      const track = document.createElement("ol");
      track.className = "cro-incident-lineage__track";
      data.stages.forEach((stage, stageIndex) => {
        const item = document.createElement("li");
        item.className = `cro-incident-lineage__stage${stage.available ? " is-available" : " is-dimmed"}`;
        item.dataset.stage = stage.id;

        const head = document.createElement("div");
        head.className = "cro-incident-lineage__stage-head";
        const index = document.createElement("span");
        index.className = "cro-incident-lineage__stage-index";
        index.textContent = String(stageIndex + 1).padStart(2, "0");
        const heading = document.createElement("div");
        const titleRow = document.createElement("div");
        titleRow.className = "cro-incident-lineage__stage-title-row";
        const stageTitle = document.createElement("h3");
        stageTitle.className = "cro-incident-lineage__stage-title";
        stageTitle.textContent = stage.title;
        titleRow.appendChild(stageTitle);
        appendIncidentLineageRoleTags(titleRow, stage.roles);
        const system = document.createElement("p");
        system.className = "cro-incident-lineage__stage-system";
        system.textContent = stage.system;
        heading.append(titleRow, system);
        head.append(index, heading);
        item.appendChild(head);

        const nodes = document.createElement("div");
        nodes.className = "cro-incident-lineage__nodes";
        stage.nodes.forEach((node) => {
          const nodeEl = document.createElement("article");
          nodeEl.className = "cro-lineage-node";
          const hasCrossStageEdge = node.id && data.edges.some(
            (edge) => edge.from === node.id || edge.to === node.id,
          );
          if (hasCrossStageEdge) {
            nodeEl.dataset.lineageId = node.id;
            nodeEl.tabIndex = 0;
            nodeEl.setAttribute("role", "button");
            nodeEl.setAttribute("aria-pressed", "false");
            const roleText = node.roles.length ? `，${node.roles.map((role) => role.label).join("、")}` : "";
            nodeEl.setAttribute("aria-label", `${stage.title}：${node.name}${roleText}。悬浮查看直接上下游，点击固定关联路径。`);
          }
          const name = document.createElement("div");
          name.className = "cro-lineage-node__name";
          name.textContent = node.name;
          name.dataset.tip = stage.available
            ? (LINEAGE_NODE_NAME_TIPS[stage.id] || "当前血缘层的节点名称。")
            : "本事件没有定位到该层，因此这里保留链条位置并暗化。";
          const nameRow = document.createElement("div");
          nameRow.className = "cro-incident-lineage__node-name-row";
          nameRow.appendChild(name);
          appendIncidentLineageRoleTags(nameRow, node.roles);
          nodeEl.appendChild(nameRow);
          if (node.id) {
            const id = document.createElement("code");
            id.className = "cro-lineage-node__id";
            id.textContent = node.id;
            id.dataset.tip = LINEAGE_NODE_ID_TIP;
            // ID 的点击下钻尚未上线，先阻止它冒泡成“固定整张卡片路径”。
            id.addEventListener("click", (clickEvent) => clickEvent.stopPropagation());
            nodeEl.appendChild(id);
          }
          const detail = document.createElement("p");
          detail.className = "cro-lineage-node__detail";
          detail.textContent = node.detail;
          nodeEl.appendChild(detail);
          nodes.appendChild(nodeEl);
        });
        item.appendChild(nodes);
        track.appendChild(item);
      });
      trackShell.append(edgeLayer, track);
      const inspector = document.createElement("section");
      inspector.className = "cro-incident-lineage__inspector";
      inspector.setAttribute("aria-live", "polite");
      inspector.hidden = true;
      host.append(trackShell, inspector);

      const defaultOrigins = lineageRoleNodeIds(data, "origin");
      const defaultVictims = lineageRoleNodeIds(data, "victim");
      const defaultSelection = lineagePathBetween(data, defaultOrigins, defaultVictims);
      let pinnedId = null;
      const paintSelection = (selected, nodeId = null, pinned = false) => {
        host.classList.toggle("is-tracing", selected.size > 0);
        track.querySelectorAll("[data-lineage-id]").forEach((element) => {
          const isActive = element.dataset.lineageId === nodeId;
          const isRelated = selected.has(element.dataset.lineageId);
          element.classList.toggle("is-lineage-active", isActive);
          element.classList.toggle("is-lineage-related", isRelated && !isActive);
          element.classList.toggle("is-lineage-muted", selected.size > 0 && !isRelated);
          element.setAttribute("aria-pressed", String(Boolean(pinnedId && element.dataset.lineageId === pinnedId)));
        });
        edgeLayer.querySelectorAll(".cro-incident-lineage__edge").forEach((edge) => {
          const active = selected.has(edge.dataset.from) && selected.has(edge.dataset.to);
          edge.classList.toggle("is-active", active);
          edge.classList.toggle("is-muted", selected.size > 0 && !active);
        });
        edgeLayer.querySelectorAll(".cro-incident-lineage__edge-label").forEach((label) => {
          const active = selected.has(label.dataset.from) && selected.has(label.dataset.to);
          label.classList.toggle("is-active", active);
        });
        // 悬浮只做原位高亮，不能展开下方检查区：检查区改变容器高度后会让指针
        // 反复进出卡片，造成详情、高亮与连线一起闪烁。只有点击固定路径才显示详情。
        renderLineageInspector(inspector, data, pinned ? nodeId : null, pinned);
      };
      const applyDefault = () => paintSelection(
        defaultSelection,
        null,
        false,
      );
      const applySelection = (nodeId, recursive, pinned = false) => {
        const hasCrossStageEdge = nodeId && data.edges.some((edge) => edge.from === nodeId || edge.to === nodeId);
        if (nodeId && !hasCrossStageEdge) nodeId = null;
        paintSelection(lineageSelection(data, nodeId, recursive), nodeId, pinned);
      };

      track.querySelectorAll("[data-lineage-id]").forEach((nodeEl) => {
        const restore = () => pinnedId
          ? applySelection(pinnedId, true, true)
          : applyDefault();
        const hasCrossStageEdge = data.edges.some(
          (edge) => edge.from === nodeEl.dataset.lineageId || edge.to === nodeEl.dataset.lineageId,
        );
        const preview = () => hasCrossStageEdge
          ? applySelection(nodeEl.dataset.lineageId, false, false)
          : restore();
        const togglePin = () => {
          if (!hasCrossStageEdge) return;
          pinnedId = pinnedId === nodeEl.dataset.lineageId ? null : nodeEl.dataset.lineageId;
          if (pinnedId) applySelection(pinnedId, true, true);
          else applyDefault();
        };
        nodeEl.addEventListener("pointerenter", preview);
        nodeEl.addEventListener("pointerleave", restore);
        nodeEl.addEventListener("focus", preview);
        nodeEl.addEventListener("blur", restore);
        nodeEl.addEventListener("click", (clickEvent) => {
          clickEvent.stopPropagation();
          togglePin();
        });
        nodeEl.addEventListener("keydown", (keyEvent) => {
          if (keyEvent.key !== "Enter" && keyEvent.key !== " ") return;
          keyEvent.preventDefault();
          togglePin();
        });
      });
      trackShell.addEventListener("click", () => {
        pinnedId = null;
        applyDefault();
      });
      host.addEventListener("keydown", (keyEvent) => {
        if (keyEvent.key !== "Escape") return;
        pinnedId = null;
        applyDefault();
      });
      host.repaintLineage = () => {
        // 刷新后默认页签是“事件详情”，此时血缘面板 hidden，首次量宽高会得到 0。
        // 页签真正可见时必须重画；若用户已固定卡片，则保留固定路径。
        const selection = pinnedId
          ? lineageSelection(data, pinnedId, true)
          : defaultSelection;
        paintIncidentLineageEdges(trackShell, edgeLayer, data, selection);
        if (pinnedId) applySelection(pinnedId, true, true);
        else applyDefault();
      };
      requestAnimationFrame(() => host.repaintLineage());
    }

    /* 中区两张角色卡 + 下区事件内涵。第 6–8 项接入下区图表。 */
    function renderIncidentView(event) {
      const topology = controller.topology;
      renderIncidentArrow(event, topology);
      // 事件 focus 指向具体算子时（如 p1-root 的 Router gate），把那根算子条也
      // 点亮——四域里这件事原来由整网 deck 承担，现在结构条是唯一的层内算子视图。
      const focusBar = event.focus?.kind === "segment" && event.focus.bar
        ? { segment: event.focus.segment, bar: event.focus.bar }
        : null;
      /* 根因与影响就是这两个角色各自「发生了什么」，接在各自第一行的范围 chip 后面，
         与 chip 合成一句「谁 · 怎么了」。它们原先在横幅桥接句和下区各出现过一次，
         现在全页只在这里。 */
      renderRoleDomains(document.getElementById("croOriginDomains"), event.origin, topology,
        { focusBar, summary: event.root });
      renderRoleDomains(document.getElementById("croVictimDomains"), event.victim, topology,
        { summary: event.impact });
      renderIncidentDetail(document.getElementById("croIncidentDetail"), event);
      renderIncidentLineage(document.getElementById("croIncidentLineage"), event, topology);
      setIncidentDetailMode(incidentDetailMode);
      // 舞台尺寸随事件变（涉及的域不同，高度差一大截），每次换事件重新适配一次
      requestAnimationFrame(() => stage.fit());
    }

    /* ── 下区 · 问题详情 ───────────────────────────────────────────────────
       中区回答「打到了谁」，这里回答「这件事本身是什么」。左边一张证据图，右边
       一句结论 + 几个关键读数 —— 结论与读数是同一件事的两种粒度（一句话 / 几个
       数），放在一起与图形成左右对读。根因与影响不在这里，它们跟在画布上两个角色
       的标题下面。 */
    let detailChart = null;   // { host, spec } —— 图表按宿主实测宽度重画，见 paintDetailChart

    function renderIncidentDetail(host, event) {
      if (!host) return;
      host.innerHTML = '';
      detailChart = null;
      const evidence = event.evidence;
      if (!evidence?.chart) return;

      const grid = document.createElement('div');
      grid.className = 'cro-incident-detail__grid';

      const figure = document.createElement('figure');
      figure.className = 'cro-figure';
      const caption = document.createElement('figcaption');
      caption.className = 'cro-figure__caption';
      caption.textContent = evidence.chart.title;
      figure.appendChild(caption);
      /* 图表按宿主的实际像素宽度出图，viewBox 宽度 = CSS 宽度 → 缩放比恒为 1：
         柱子粗细、字号、行高都是设计值本身，不会被「SVG 按比例放大填满容器」
         连带撑高，下区也就不会为了一张 12 行的图冒出滚动条。 */
      const chartHost = document.createElement('div');
      chartHost.className = 'cro-figure__chart';
      figure.appendChild(chartHost);
      detailChart = { host: chartHost, spec: evidence.chart };
      if (evidence.chart.note) {
        const note = document.createElement('p');
        note.className = 'cro-figure__note';
        note.textContent = evidence.chart.note;
        figure.appendChild(note);
      }
      grid.appendChild(figure);

      const aside = document.createElement('div');
      aside.className = 'cro-incident-detail__aside';
      if (event.conclusion) {
        const conclusion = document.createElement('p');
        conclusion.className = 'cro-incident-detail__conclusion';
        conclusion.textContent = event.conclusion;
        aside.appendChild(conclusion);
      }
      if (evidence.metrics?.length) {
        const list = document.createElement('dl');
        list.className = 'cro-readout';
        evidence.metrics.forEach((metric) => {
          const row = document.createElement('div');
          row.className = 'cro-readout__row';
          const dt = document.createElement('dt');
          dt.textContent = metric.label;
          const dd = document.createElement('dd');
          dd.textContent = metric.value;
          if (metric.tone) dd.dataset.tone = metric.tone;
          row.append(dt, dd);
          list.appendChild(row);
        });
        aside.appendChild(list);
      }
      grid.appendChild(aside);
      host.appendChild(grid);
      requestAnimationFrame(paintDetailChart);
    }

    /* 图按宿主的实测宽高出图。高度预算 = 下区可用净高 − figure 里图表之外的部分
       （图题 + 读法那两行）。小屏下预算变小，各图各自的收敛方式不同：
         line     —— 压扁（趋势靠横向读，压矮不失真）
         stack    —— 不吃预算（DOM 构成条本来就只有一条 pill + 几行直标，够矮）
         bars     —— 条目多就整个换成竖排柱，高度不再随条目数长
         capacity —— 盒子高度按预算取（148–260px），SVG 自己等比缩放
       目的只有一个：这一栏不出滚动条。 */
    function paintDetailChart() {
      if (!detailChart) return;
      const { host, spec } = detailChart;
      const build = CHART_BUILDERS[spec.kind];
      if (!build || !host.isConnected) return;
      const width = Math.max(280, Math.round(host.clientWidth || 560));
      const body = host.closest(".cro-incident-detail__body");
      const figure = host.closest(".cro-figure");
      // figure 当前高度里除掉图表自身，剩下的就是图题 + 读法 + 内边距的固定开销
      const overhead = figure ? Math.max(0, figure.offsetHeight - host.offsetHeight) : 60;
      const budget = Math.round(Math.max(96, (body?.clientHeight || 260) - overhead - 8));
      // 第四参 host：等距容器要在一个**已挂到文档上**的节点上解析 var(--token)，
      // 其余 builder 用不到，多传一个参数无害。
      host.replaceChildren(build(spec, width, budget, host));
    }

    function selectIncident(event) {
      activeIncident = event;
      const topology = controller.topology;
      relation = addIncidentScope(resolveRelation(topology, { ...event.focus, incidentId: event.id }), event, topology);
      setIncidentLayout(true);
      renderIncidentView(event);
      const banner = document.getElementById("croIncidentBanner");
      if (banner) banner.hidden = false;
      setText("croIncidentBannerTag", event.code);
      setText("croIncidentBannerMessage", event.banner || event.path || event.title);
      document.querySelectorAll(".cro-event").forEach((button) => {
        const selected = button.dataset.eventId === event.id;
        button.classList.toggle("is-selected", selected);
        button.setAttribute("aria-pressed", String(selected));
      });
      document.dispatchEvent(new CustomEvent("cro:incident", { detail: { event, relation } }));
    }

    function renderIncidentRail() {
      const host = document.getElementById("croEventGroups");
      if (!host) return;
      host.innerHTML = "";
      INCIDENT_GROUPS.forEach((group) => {
        const section = document.createElement("section");
        const expandedByDefault = group.id === "problem-1" || group.id === "problem-2";
        section.className = `cro-event-group${expandedByDefault ? " is-expanded" : ""}`;
        section.innerHTML = `
          <button class="cro-event-group__toggle" type="button" aria-expanded="${expandedByDefault}">
            <span class="cro-event-group__toggle-main">
              <svg class="cro-event-group__chevron" viewBox="0 0 12 12" aria-hidden="true"><path d="m4 2 4 4-4 4"></path></svg>
              <span class="cro-event-group__name">${group.name}</span>
            </span>
          </button>
          <div class="cro-event-list"></div>`;
        const toggle = section.querySelector(".cro-event-group__toggle");
        toggle.addEventListener("click", () => {
          const expanded = !section.classList.contains("is-expanded");
          section.classList.toggle("is-expanded", expanded);
          toggle.setAttribute("aria-expanded", String(expanded));
        });
        const list = section.querySelector(".cro-event-list");
        group.events.forEach((event, index) => {
          event.context = group.context;
          // 事件编号 = 问题线号.本组内序号，如「问题1.3」。组号取自 group.id
          // （problem-1 / problem-2），与组名里的数字同源，不另立一份。
          event.code = `问题${group.id.replace("problem-", "")}.${index + 1}`;
          event.banner = group.bridge(event);
          const button = document.createElement("button");
          button.type = "button";
          button.className = "cro-event";
          button.dataset.eventId = event.id;
          button.setAttribute("aria-pressed", "false");
          button.innerHTML = `
            <span class="cro-event__index">${String(index + 1).padStart(2, "0")}</span>
            <span>
              <span class="cro-event__name">${event.title}</span>
              <span class="cro-event__dimension">${event.dimension}</span>
            </span>
            <span class="cro-event__time">${event.time}</span>`;
          button.addEventListener("click", () => selectIncident(event));
          list.appendChild(button);
        });
        host.appendChild(section);
      });
    }

    /* select:false 只翻栏宽、不动选择 —— 给「顺手收起」的调用方用（起播、关闭
       事件横幅）。它们各自已经把选择处理妥当了：起播要留着 relation 供退出时
       还原，横幅关闭自己就调了 clearSelection，这里再来一次会把刚定好的状态
       又推翻一遍。rail 键本身仍走默认的 true。 */
    function setEventRailCollapsed(collapsed, { select = true } = {}) {
      const workarea = document.querySelector(".pto-ide-frame__workarea");
      workarea?.classList.toggle("is-event-rail-collapsed", collapsed);
      document.getElementById("navRelationEvents")?.setAttribute("aria-pressed", String(!collapsed));
      if (select) {
        if (collapsed) clearSelection();
        else selectIncident(INCIDENT_GROUPS[0].events[0]);
      }
      // 侧栏宽度不再走过渡（见 css 里 .cro-event-rail 的说明），class 翻转后
      // 宽度即已确定。下一帧读一次几何就够，不必再等一个动画时长。
      cancelAnimationFrame(railLayoutTimer);
      railLayoutTimer = requestAnimationFrame(() => {
        // 栏宽一翻，板面横向差 250 多像素：集群矩阵的列轨是固定像素写死的，
        // 不在这儿补量就会一直按翻转前的宽度挤着（见 resyncClusterGeometry）。
        resyncClusterGeometry();
        layoutLayerNav(layerNav);
        redrawLinks();
      });
    }

    /* 补量一次格宽/格高（不重建那 2048 个格子，只改 grid-template-columns 与一条
       CSS 变量）。renderCluster 是在刚清空的矩阵上量的宽度，那一刻 .cro-cluster__grid
       还没有纵向滚动条；铺完之后若因为几何误差冒出一条，block 的可用宽度就窄了
       一条滚动条 —— 补量把它纠回来。
       growHeight:false 用在 renderCluster 之后的那两处：那时格高已经按终局视口
       算过一次了，再让它涨只会溢出（见 syncCellWidth 的注释）。 */
    const refitClusterCells = (opts) => {
      const heat = document.getElementById("croHeat");
      if (!heat || !heat.dataset.epRows) return;   // 无矩阵（配置非法/超格数上限）时不管
      const counts = controller.topology.counts;
      const epRows = Number(heat.dataset.epRows) || 1;
      syncCellWidth(heat, Math.ceil(counts.ep / epRows), counts, epRows, opts);
    };

    /* 版面尺寸变了之后把矩阵重新对齐到新视口：先定两行高度（只看 arch 的内容
       需求），再看折行数要不要改 —— 改了只能重建（行数是建 DOM 时定的），没改
       就只补量格宽/格高。窗口 resize 与事件栏折叠/展开（栏宽 292px ⇄ 40px，板面
       横向差 250 多像素）走同一条通路：syncCellWidth 写进 block 的是**固定像素**
       的列轨，板面一变宽而不补量，每个 stage 块的格子就还挤在原来那点宽度里、
       右边空出一道缝，矩阵下方也留白 —— 就是「首屏排布畸形、切一次模型（走
       onChange 整条重铺）反而规整」的由来。 */
    const resyncClusterGeometry = () => {
      syncBoardRows(controller.topology.counts);
      const heat = document.getElementById("croHeat");
      if (!heat || !heat.dataset.epRows) return;   // 无矩阵（配置非法/超格数上限）时不管
      const epRows = pickEpRows(heat, controller.topology.counts);
      if (String(epRows) !== heat.dataset.epRows) { rebuildCluster(); return; }
      refitClusterCells();
    };

    controller.onChange((topology) => {
      // 配置非法时不重建 deck，保留上一版可读的图，错误信息由 #croConfigError 承担
      if (topology.valid || !deck?.controller) deck?.build(topology);
      // deck 可能整棵重建了，去重缓存记的是旧 DOM 的状态，作废
      deckFrontLayer = null;
      deckNodeKey = null;
      // deck 的语义色变量搬到 board 上，结构条 bar 与整网节点取到同一个色值
      syncPalette();
      // 稠密模型（无 MoE）：整块 MoE 区收起，Model Architecture 撑到右侧（见 css 的 .is-no-moe）
      document.getElementById("croBoard")?.classList.toggle("is-no-moe", !topology.hasMoe);
      renderLayerNav(layerNav, topology, emitSelect);
      renderStructure(structure, topology, emitSelect);
      renderMoe(
        document.getElementById("croSharedExperts"),
        document.getElementById("croRoutedExperts"),
        topology, emitSelect,
      );
      /* 次序要紧：两行的高度先定 —— 它只看 Model Architecture 的内容需求，与矩阵
         无关（见 syncBoardRows 的注释），所以不必等矩阵铺完。定完之后 Cluster 那
         一行的视口高度才是终局值，renderCluster 照着它算折几行、格子多高，铺出来
         正好填满，既不留白也不溢出。 */
      syncBoardRows(topology.counts);
      syncClusterAxisNote(topology);
      renderCluster(document.getElementById("croHeat"), topology, emitSelect);
      refitClusterCells({ growHeight: false });
      if (activeIncident) requestAnimationFrame(() => selectIncident(activeIncident));
      else if (relation) reapplySelection(topology);
    });

    /* 整网视图切档（见 html 的 croNetView）。四域里只有 Cluster 需要重建：
       其余三块的差异纯粹是显示/隐藏与排布，CSS 就够；而集群矩阵的 epRows
       是建 DOM 时定的行数（见 renderCluster），改不了类就改不了。
       重建会连带清掉格子上的 is-related / is-selected，随后按当前选择重铺一次。 */
    const rebuildCluster = () => {
      const topology = controller.topology;
      syncBoardRows(topology.counts);   // 同上：先定行高，矩阵再照着终局视口铺
      renderCluster(document.getElementById("croHeat"), topology, emitSelect);
      refitClusterCells({ growHeight: false });
      if (activeIncident) requestAnimationFrame(() => selectIncident(activeIncident));
      else if (relation) reapplySelection(topology);
    };

    document.addEventListener("cro:view", rebuildCluster);

    /* 窗口一变，两行的分配与矩阵的几何都要跟着重算，而且**必须按这个次序**：
       先定行高（只看 arch 的内容需求），Cluster 那一行的视口高度才是终局值；
       折几行（pickEpRows）与格子多高（verticalCellHeight）都是照着它算的。
       曾经写成两条各自防抖的监听，谁先跑取决于注册次序，结果矩阵按旧行高量完
       几何、行高才改 —— 合成一条就没有这个隐式依赖了。
       net-view / YAML 视图切换也走这条通路（见 html 内联脚本与
       config-relation-yaml.js 里各自那句 window.dispatchEvent(new Event("resize")))。
       行数是建 DOM 时定的，改不了类就改不了，只有重建一途；但重建动辄 2048 个
       格子，不能每帧都来，所以先算一遍目标行数，跟当前渲染的一样就只调样式。 */
    let clusterResizeTimer = 0;
    global.addEventListener("resize", () => {
      clearTimeout(clusterResizeTimer);
      clusterResizeTimer = setTimeout(resyncClusterGeometry, 180);
    });

    // 列宽随窗口变化，PP 带的实测定位要跟着重排
    if (layerNav && global.ResizeObserver) {
      let pending = 0;
      new ResizeObserver(() => {
        clearTimeout(pending);
        pending = setTimeout(() => {
          layoutLayerNav(layerNav);
        }, 48);
      }).observe(layerNav);
    }
    // 主题切换会重算 deck 调色板，重新搬一次
    document.addEventListener("cro:theme", syncPalette);
    /* 等距容器图（chart.kind = "capacity"）的三个面明暗是建图时按 token 值算死写进
       SVG 的，不像其余图那样靠 CSS 类跟主题走 —— 换主题得整幅重画。 */
    document.addEventListener("cro:theme", () => {
      if (detailChart && detailChart.spec.kind === "capacity") requestAnimationFrame(paintDetailChart);
    });

    /* ── 连线是画在 viewport 坐标上的，任何位移都要重画 ── */
    ["scroll", "wheel"].forEach((type) => {
      document.addEventListener(type, () => requestAnimationFrame(redrawLinks), { passive: true, capture: true });
    });
    global.addEventListener("resize", () => requestAnimationFrame(() => {
      redrawLinks();
      // 画布里的刻度带宽度也是实测出来的（board 那份有 ResizeObserver），
      // 重排完再按新视口重新适配一次
      document.querySelectorAll("#croIncidentView .cro-layer-nav").forEach((nav) => layoutLayerNav(nav));
      if (activeIncident) {
        stage.fit();
        paintDetailChart();   // 证据图按新宽度重画（它不靠 SVG 缩放去适应容器）
      }
    }));
    // deck 自己的拖拽/缩放会挪动被选中的节点
    document.getElementById("croDeckHost")?.addEventListener("pointermove", () => {
      if (relation) requestAnimationFrame(redrawLinks);
    }, { passive: true });

    /* ── 清空选择：Esc，或点击 board 空白处 ── */
    document.addEventListener("keydown", (event) => {
      if (event.key !== "Escape") return;
      if (relation) clearSelection();
    });
    /* 挂在 document 而不是 board：点顶栏、activity rail、集群右侧留白这些
       board 之外的地方也要能清空。命中任一可选对象则不清。 */
    const SELECTABLE = [
      ".cro-tick", ".cro-pp-span", ".cro-bar", ".cro-expert", ".cro-moe-group",
      ".cro-structure__col",
      ".twin-heat-cell", ".pto-model-deck__node", ".pto-model-deck__experts",
      // 单卡容量栏：点 stage 小柱是「选中该 stage 首卡」，不是点空白。
      // 口径浮层被挂到 body 上（避开 pane 的 overflow/backdrop-filter），不在
      // .cro-capacity 子树里，得单独列一条，否则点它选中文字会清掉当前选择。
      ".cro-capacity", ".cro-capacity__basis",
      // YAML 视图的代码框：在里面拖选文本、点复制键都不是「点空白」
      ".cro-region--yaml",
      // .cro-ep-mode 与 .cro-stepper 同理：EP 口径开关是配置控件而不是画布空白
      ".pto-model-deck__side-rule", ".cro-stepper", ".cro-ep-mode", ".cro-event", ".pto-ide-frame__topbar",
    ].join(", ");
    document.addEventListener("click", (event) => {
      if (!relation) return;
      // 运行事件是一次显式调查上下文：点击画布空白不应误退出。只有横幅关闭键
      // 或其他可响应对象触发新的选择时，才结束当前事件关系。
      if (activeIncident) {
        // 整条运行事件栏都不算「离开当前事件」：除了事件条目本身，分组标题
        // （.cro-event-group__toggle）和收起键也都是 <button>，只判 .cro-event
        // 会让它们掉进下面那条通用 button 分支 —— 点一下展开箭头就白跑一整轮
        // applyRelation（2048 格 + 256 专家）＋ deck 反选 ＋ 连线重画 ＋ 横幅
        // 收起（又改变 board 高度再触发一次全量重排）。
        if (event.target.closest?.(".cro-event-rail")) return;
        // 事件详情视图本身也不算「离开当前事件」：角色卡里重建的那几个域是只读
        // 证据（按钮已被 pointer-events 挡掉），点到它们不该把详情关掉。
        if (event.target.closest?.("#croIncidentView")) return;
        if (event.target.closest?.("button, select, input, [role='button']")) clearSelection();
        return;
      }
      if (!event.target.closest?.(SELECTABLE)) clearSelection();
    });

    global.croObserver = controller;
    global.croDeck = deck;
    global.croSelect = emitSelect;
    global.croSelectIncident = selectIncident;
    renderIncidentRail();
    document.getElementById("croIncidentDetailTab")?.addEventListener("click", () => {
      setIncidentDetailMode("detail");
    });
    document.getElementById("croIncidentLineageTab")?.addEventListener("click", () => {
      setIncidentDetailMode("lineage");
    });
    document.querySelector(".cro-incident-detail__tabs")?.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const next = incidentDetailMode === "detail" ? "lineage" : "detail";
      setIncidentDetailMode(next);
      document.getElementById(next === "detail" ? "croIncidentDetailTab" : "croIncidentLineageTab")?.focus();
    });
    document.getElementById("croEventRailCollapse")?.addEventListener("click", () => setEventRailCollapsed(true));
    document.getElementById("croEventRailExpand")?.addEventListener("click", () => setEventRailCollapsed(false));
    /* 关闭事件横幅 = 结束这次调查，回配置仿真态。事件栏跟着收起：留着它等于
       还在「挑事件看」的上下文里，而横幅一关四域就接管了整块画布，那份列表
       只是在左边占宽。select:false —— clearSelection 上一行刚调过。 */
    document.getElementById("croIncidentBannerClose")?.addEventListener("click", () => {
      clearSelection();
      setEventRailCollapsed(true, { select: false });
    });
    document.getElementById("navRelationEvents")?.addEventListener("click", () => {
      const workarea = document.querySelector(".pto-ide-frame__workarea");
      setEventRailCollapsed(!workarea?.classList.contains("is-event-rail-collapsed"));
    });
    /* 深链接:聚光灯定位链「查看事件影响范围」按 ?event=<id> 从别的问题页跳过来,
       命中就直接选中该运行事件、进事件详情。
       没带参数时**留在配置关系态**:这一页的主业是四域关系图与配置仿真,自动展开
       第一个事件等于替用户做了一次他没提的调查——一进来就是某次故障的详情,四域
       整块被顶掉,反倒要先关掉横幅才能看正事。事件栏一并收起(与关闭事件横幅同一
       套处理:留着它等于还在「挑事件看」的上下文里),左侧那条竖标签随时能展开。 */
    const requestedEventId = new URLSearchParams(global.location.search).get("event");
    const requestedEvent = requestedEventId
      ? INCIDENT_GROUPS.flatMap((group) => group.events).find((event) => event.id === requestedEventId)
      : null;
    /* ⚠️ 收栏必须**先于** controller.refresh()：事件栏 292px ⇄ 40px 一翻，板面
       宽度差 250 多像素，而首帧那次 renderCluster / syncCellWidth 把列轨按当时
       的宽度写成了固定像素。先铺矩阵再收栏，矩阵就一直按「栏还开着」的窄宽度
       挤在每个 stage 块的左半边、下方也留白（首屏那个畸形排布）；随便切一次
       模型走完整条 onChange 才会按终局宽度重铺，于是「切走再切回来就好看了」。
       现在把版面先摆到终局态，第一次铺就是对的。
       带 ?event= 进来时保持 HTML 里的展开态，交给下面的 selectIncident。 */
    if (!requestedEvent) {
      // select:false —— 此刻还没有任何选择,不必再走一次 clearSelection
      setEventRailCollapsed(true, { select: false });
    }
    controller.refresh();
    if (requestedEvent) requestAnimationFrame(() => selectIncident(requestedEvent));
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }
})(window);
