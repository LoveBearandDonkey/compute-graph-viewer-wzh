/* ══════════════════════════════════════════════════════════════════════════
   配置 YAML 视图（顶栏「关系视图 / YAML 视图」的第二档）
   ------------------------------------------------------------------------
   切到 YAML 档时整网列原样留着（这份 yaml 描述的就是左边这张网），右侧三个区
   （Model Architecture / MoE / Cluster）整块换成上下分栏：
     上 · configs/<家族>/run_<全名>.yaml —— MindSpore + MindFormers 口径
     下 · msrun 启动命令 —— 集群规模（多少卡、多少节点）真实的落点

   ⚠ 两块都是**实时**由 croObserver 的当前配置生成的，不是写死的样例：用户刚在
   四域里调完参数切过来，代码框里的数字必须和刚才那一屏对得上，否则整页的可信度
   就没了。本文件只吃 cro:change 事件 + 读 croObserver.topology，一行不改主控制器
   （与 config-relation-capacity.js 同样的接法）。

   ── 为什么集群那几项不在 yaml 里 ──
   对标 MindFormers 的真实 run_*.yaml：模型结构 / parallel_config / moe_config /
   runner_config 确实同处一个文件，但**卡型号、单卡 HBM、节点数、总卡数不在**：
     · 卡型号与 HBM 是硬件事实，由驱动探测；yaml 里只有 context.max_device_memory
       这个「给框架划多少显存」的上限，故写成它的行尾注释；
     · 总卡数 / 节点数由启动器给（msrun 的 worker_num / local_worker_num），
       框架只校验并行度的乘积 == worker_num —— ep 进不进这个乘积取决于页面的
       EP 口径开关：切出档（默认）是 dp×mp×pp×cp，正交档是 dp×mp×pp×cp×ep。
   把它们硬塞进 yaml 会骗人（写 64 GB 也变不出 64 GB），所以下沉到启动命令那一栏。
   ══════════════════════════════════════════════════════════════════════════ */
(function (global) {
  "use strict";

  const doc = global.document;

  /* ── 行模型：{ text, field } ───────────────────────────────────────────
     field  这一行对应的 stepper 字段名（没有就是结构常量行），用于把「与预设
            默认值不同」的行标出来 —— 一份几十行的 yaml 里用户真正改过的就那么
            两三处，不标出来等于让人拿眼睛做 diff。
            允许给一个数组：runner_config.batch_size 这类由几个字段乘出来的行，
            其中任何一个变了这一行就变了，只认头一个会漏标。
     note   行尾注释。由 L() 统一补空格对齐到 NOTE_COL，别在调用处手敲空格：
            值是算出来的（位数会变），手敲的对齐一改参数就散。 */
  const NOTE_COL = 38;
  const L = (text, field, note) => ({
    text: note ? text.padEnd(NOTE_COL) + "# " + note : text,
    field,
  });

  /* yaml 的布尔写法是首字母大写的 Python 风格，不是 js 的 true/false */
  const bool = (v) => (v ? "True" : "False");

  /* openPangu 2.0 flash 92B → openpangu_2_0_flash_92b（MindFormers 的命名口径） */
  const snake = (label) => String(label).toLowerCase().replace(/[\s.-]+/g, "_");

  /* configs/<家族>/run_<全名>.yaml —— 与 MindFormers 仓的 configs/ 目录同形 */
  function yamlPath(preset) {
    const full = snake(preset.label);
    return "configs/" + full.split("_")[0] + "/run_" + full + ".yaml";
  }

  /* ── 上栏：MindFormers run_*.yaml ─────────────────────────────────────── */
  function buildYamlLines(topology) {
    const c = topology.counts;
    const p = topology.preset;
    const card = topology.card;
    const cfg = topology.config;
    /* EP 进不进乘积由页面的 EP 口径开关定（见 observer.js 的 epInWorld）：
       切出档（MindFormers 的实际做法）EP 从 DP 组内切出，不占独立 rank。 */
    const orthogonal = Boolean(c.moeOrthogonal);
    const world = c.dp * c.pp * c.tp * c.cp * (orthogonal ? c.ep : 1);
    const name = snake(p.label);
    const noMoe = Boolean(p.noMoe);

    /* pipeline_stage 的非均分层切分在 MindFormers 里靠 model_config.offset 表达：
       各 stage 相对「均分」多带几层。46/4 → 均分 11，offset [1,1,0,0] = 12,12,11,11 */
    const base = Math.floor(c.totalLayer / c.pp);
    const offset = topology.stages.map((s) => s.count - base);
    const split = topology.stages.map((s) => s.count).join(",");
    /* max_device_memory 不是 HBM 本身，是「给框架划多少」：要留一截给通信与算子
       工作区，业界常见留 6 GB 上下。 */
    const deviceMem = Math.max(1, card.hbmGB - 6);

    /* 流水线微批数：本页没有为它开字段（气泡占比未建模），按 4×PP 取一个常见值；
       PP=1 时没有流水线可分，取 1 —— 写 4 会凭空多出一层梯度累积。
       它同时是下面 batch_size 的一个因子，所以只算这一处，两行的数字必须对得上。 */
    const microBatchNum = c.pp > 1 ? c.pp * 4 : 1;
    /* full_batch 口径下 runner_config.batch_size 填的是**全局** batch —— 见那一行
       上面的长注释（升级计划行 12）。 */
    const globalBatch = cfg.microBatch * c.dp * microBatchNum;

    /* 三档口径读同一个 config 字段（行 15）。兼容老配置里的布尔 parallelOptimizer：
       true 等价于 "zero1" —— 折算只写这一处，与 capacity 的 shardPlan 同一条规矩。 */
    const shardMode = cfg.shardMode || (cfg.parallelOptimizer ? "zero1" : "none");
    const shardDenom = noMoe || c.ep <= 1
      ? "沿 DP 分片，每卡只存 1/" + c.dp
      : "专家 ÷ EDP=" + c.edp + "，其余权重 ÷ " + (c.edp * c.ep);

    const lines = [
      L("# " + yamlPath(p)),
      L("# " + p.label + " · MindSpore + MindFormers 训练配置"),
      L("#"),
      L("# 硬件与集群规模不写在本文件里，由启动命令给出（见下栏）："),
      /* 末节点没装满时照实写出来（只有手输能拨到非整机倍数）—— 这一行是给人抄去
         配集群的，"2 节点 × 8 卡 = 12 rank" 是一句算不平的话。 */
      L("#   " + card.label + " · 单卡 HBM " + card.hbmGB + " GB · "
        + (c.tailRanks === c.ranksPerNode
          ? c.node + " 节点 × " + c.ranksPerNode + " 卡 = " + c.totalRank + " rank"
          : c.node + " 节点 × " + c.ranksPerNode + " 卡（末节点 " + c.tailRanks
            + " 卡）= " + c.totalRank + " rank")),
      L(orthogonal
        ? ("# 框架校验 dp×mp×pp×cp×ep = " + c.dp + "×" + c.tp + "×" + c.pp + "×" + c.cp
          + "×" + c.ep + " = " + world + "，须等于 msrun 的 worker_num")
        : ("# 框架校验 dp×mp×pp×cp = " + c.dp + "×" + c.tp + "×" + c.pp + "×" + c.cp
          + " = " + world + "，须等于 msrun 的 worker_num"
          + "（ep 从 dp 内切出，不进乘积）")),
      L(""),
      L("seed: 0"),
      L("run_mode: 'train'"),
      L("output_dir: './output'"),
      L(""),
      L("trainer:"),
      L("  type: CausalLanguageModelingTrainer"),
      L("  model_name: '" + name + "'"),
      L(""),
      L("runner_config:"),
      L("  epochs: 1"),
      /* ⚠ 这一格填的是**全局** batch，不是表单上那枚 Micro Batch —— 升级计划行 12。
         MindFormers《大模型性能调优指南》讲 IO 瓶颈时对 full_batch 的原话是：
         「在配置为 true 时，每张卡都取 global batch size 的数据量，然后在图内完成
         数据的切分，只取对应 DP 域内所需数据进行训练」；源码侧 base_dataset.py 在
         semi + full_batch 下把分片信息置成 shard_id=0 / num_shards=1，batch 大小
         原样交给 dataset.batch()，没有任何 ×device_num 的补偿。
         所以这里写 MBS 会被框架当成全局 batch：DP=512 时每卡实际只训 1/512 个样本，
         而且 batch 维根本切不开（1 分不成 512 份），这份 yaml 拿去跑就是错的。
         页面文档里「全局 batch = MBS × DP × 累积步数」本来就是这么写的 —— 与前面
         几行一样，这次是 yaml 追平文档；流水线下的累积步数就是 micro_batch_num。
         反过来，**容量栏的 mb 不用再除 DP**：表单上那枚 Micro Batch 按定义就是每卡
         每次前反向的样本数，行 12 的问句到此有了答案。 */
      L("  batch_size: " + globalBatch, ["microBatch", "dp", "pp"],
        "全局 batch = MBS " + cfg.microBatch + " × DP " + c.dp
          + " × micro_batch_num " + microBatchNum
          + "；图内按 dp 切、再分微批 → 每卡每次前反向 " + cfg.microBatch),
      L("  sink_mode: True"),
      L("  sink_size: 1"),
      L(""),
      L("context:"),
      L("  mode: 0", null, "0 = Graph Mode"),
      L("  device_target: 'Ascend'"),
      L("  max_device_memory: '" + deviceMem + "GB'", "card",
        card.label + " 单卡 HBM " + card.hbmGB + " GB，留约 6 GB 给通信与算子工作区"),
      L("  jit_level: 'O1'"),
      L(""),
      L("parallel:"),
      L("  parallel_mode: 1", null, "1 = SEMI_AUTO_PARALLEL"),
      L("  full_batch: True", null,
        "每张卡都读整份全局 batch，再在图内按 dp 切 —— 上面 batch_size 的口径由它定"),
      /* 原先是死值，而 capacity 那边按「DP 不除任何东西」估优化器状态 —— 正好相反，
         是升级计划行 10 要治的病。现在两边读同一个 config 字段。 */
      /* 分母写全：MoE 下专家与其余权重不是同一个数（专家只在 EDP 维上复制），
         这正是 capacity 里那两条口径，注释与容量柱说的必须是同一句话。
         DP=1 时如实说它此刻等同于关 —— 表单上那枚开关此时也是灰的。 */
      /* 三档共用这一段分母文字（升级计划行 15），两处注释读的必须是同一句 */
      L("  enable_parallel_optimizer: " + bool(shardMode !== "none"), "shardMode",
        shardMode === "none" ? "关：每张卡各存一份完整的权重 / 梯度 / 优化器状态"
          : c.dp <= 1 ? "DP=1 无处可分，此档等同于关"
            : (shardMode === "fsdp2" ? "FSDP2（ZeRO-3）：权重 / 梯度 / 优化器三段全切，" : "ZeRO-1：只切优化器状态，")
              + shardDenom),
      /* MindFormers 的 parallel 段里没有单独的 FSDP2 开关（它的 ZeRO-3 走
         optimizer_weight_shard / MindSpeed 侧则是 --use-torch-fsdp2），所以这一行
         **只在 fsdp2 档出现**，且明说它是本页的口径标注而不是可直接照抄的配置项。
         写出来的理由是：容量柱此刻按 ZeRO-3 画，yaml 若一个字不提，
         就又回到了行 9–11 治的那个病 —— 两个视图各讲一套故事。 */
      ...(shardMode === "fsdp2" ? [
        L("  # 本页按 ZeRO-3 口径估算（权重/梯度/优化器三段全切 + 逐层 all-gather）", null,
          "MindFormers 无同名开关：MindSpore 侧对应 optimizer_weight_shard 的全分片档，"
          + "Megatron/MindSpeed 侧对应 --use-torch-fsdp2。落到具体框架时改写这一行"),
      ] : []),
      L(""),
      L("parallel_config:"),
      // EP=1 时两种口径没有差别，不必解释 EDP（稠密模型走的就是这一支）
      L("  data_parallel: " + c.dp, "dp",
        !orthogonal && c.ep > 1 ? "含 EP 组在内的真 DP，专家只在 EDP=" + c.edp + " 维上复制" : null),
      /* TP 的整除对象是模型常量而不是别的并行维，写进行尾注释里 —— 这份 yaml
         是拿去跑的，框架校验不过时这一行的注释就是答案。 */
      L("  model_parallel: " + c.tp, "tp", "即 TP，须整除 num_heads " + p.heads),
      L("  pipeline_stage: " + c.pp, "pp", "即 PP，分层见 model_config.offset"),
      L("  context_parallel: " + c.cp, "cp", "即 CP"),
      L("  expert_parallel: " + c.ep, "ep",
        c.ep <= 1 ? "即 EP"
          : orthogonal ? "即 EP，与 DP 正交，独占自己的 rank"
            : "即 EP，从 DP 内切出：EDP = DP/EP = " + c.edp),
      L("  micro_batch_num: " + microBatchNum, null,
        c.pp > 1 ? "流水线微批数，须 ≥ pipeline_stage；本页未建模，按 4×PP 取"
          : "无流水线，不分微批"),
      /* 这两行原先是死值，与 capacity 的假设正好相反（那边按「不重计算 + 开 SP」
         估激活）。现在两边读同一个 config 字段，改一次两处一起动 —— 升级计划行 9。 */
      L("  use_seq_parallel: " + bool(cfg.seqParallel), "seqParallel",
        cfg.seqParallel
          ? "序列并行：TP 切不到的那部分激活再沿序列维切开"
          : "关：LayerNorm / 残差流的激活在 TP 组内整份复制"),
      /* 同上：capacity 原先无条件把 emb/head ÷TP，而这一行写的是 True（走 DP、
         不切 TP）—— 升级计划行 11。TP=1 时两档数字相同，一提 TP 就分道扬镳。 */
      L("  vocab_emb_dp: " + bool(cfg.vocabEmbDp), "vocabEmbDp",
        cfg.vocabEmbDp
          ? "词表走 DP：每卡背满 " + p.vocab + "×" + p.hidden + "，不沿词表维切 TP"
          : "词表沿词表维切成 TP 份，每卡背 " + Math.floor(p.vocab / c.tp) + " 行"),
      L(""),
      L("recompute_config:"),
      L("  recompute: " + bool(cfg.recompute), "recompute",
        cfg.recompute
          ? "全重计算：每层只留输入，反向重算一遍前向"
          : "关：每层的中间激活全部留在显存里"),
      L("  select_recompute: False"),
      L(""),
    ];
    if (!noMoe) {
      lines.push(
        L("moe_config:"),
        L("  expert_num: " + c.routedExpert, "routedExpert", "路由专家总数"),
        L("  num_experts_chosen: " + c.topK, "topK", "即 Top-K"),
        L("  shared_expert_num: " + c.sharedExpert, "sharedExpert"),
        L("  capacity_factor: 1.5"),
        L("  aux_loss_factor: 0.05"),
        L("  routing_policy: 'TopkRouterV2'"),
        L(""),
      );
    }
    lines.push(
      L("model:"),
      L("  arch:"),
      L("    type: " + (p.archType || "PanguForCausalLM")),
      L("  model_config:"),
      L("    num_layers: " + c.totalLayer, "totalLayer"),
      L("    hidden_size: " + p.hidden),
      L("    num_heads: " + p.heads),
    );
    if (noMoe && p.kvHeads) {
      lines.push(L("    n_kv_heads: " + p.kvHeads, null,
        "GQA：" + p.heads + " Q head : " + p.kvHeads + " KV head"));
    }
    lines.push(
      L("    vocab_size: " + p.vocab),
      L("    seq_length: " + cfg.seqLen, "seqLen"),
    );
    if (noMoe) {
      lines.push(L("    intermediate_size: " + p.denseIntermediate));
    } else {
      lines.push(
        L("    intermediate_size: " + p.denseIntermediate, null,
          "Dense MLP，L0–L" + Math.max(0, c.denseLayers - 1)),
        L("    moe_intermediate_size: " + p.moeIntermediate, null, "MoE FFN，共 " + c.moeLayers + " 层"),
        L("    first_k_dense_replace: " + c.denseLayers),
      );
    }
    lines.push(
      L("    offset: [" + offset.join(", ") + "]", null,
        "相对均分 " + base + " 层的偏移 → 各 stage " + split),
    );
    if (!noMoe) lines.push(L("    mtp_depth: " + p.mtpLayers));
    lines.push(
      L("    compute_dtype: 'bfloat16'"),
      L("    use_flash_attention: True"),
    );

    if (!topology.valid) {
      /* 校验没过时把原因原样顶在文件头：这份 yaml 拿去启动会当场挂，
         让人翻回关系视图才看到那行红字就太晚了。 */
      const head = [L("# ⚠ 当前配置未通过校验，启动前请先修正：")];
      topology.errors.forEach((message) => head.push(L("#   · " + message)));
      head.push(L(""));
      return head.concat(lines);
    }
    return lines;
  }

  /* ── 下栏：msrun 启动命令 ──────────────────────────────────────────────
     集群那三项（总卡数 / 每节点卡数 / 节点数）真实的落点。msrun 是 MindSpore 的
     动态组网启动器，多机不再需要 RANK_TABLE_FILE。 */
  function buildLaunchLines(topology) {
    const c = topology.counts;
    return [
      L("# 每个节点执行一次，node_rank 从 0 递增到 " + Math.max(0, c.node - 1)
        + "；msrun 动态组网，无需 RANK_TABLE_FILE"),
      L("msrun --worker_num=" + c.totalRank + " \\", "totalRank"),
      L("      --local_worker_num=" + c.ranksPerNode + " \\", "node"),
      L("      --master_addr=${MASTER_ADDR} \\"),
      L("      --master_port=8118 \\"),
      L("      --node_rank=${NODE_RANK} \\"),
      L("      --log_dir=./output/msrun_log \\"),
      L("      --join=False \\"),
      L("      --cluster_time_out=300 \\"),
      L("      run_mindformer.py \\"),
      L("      --config " + yamlPath(topology.preset) + " \\"),
      L("      --run_mode train \\"),
      L("      --use_parallel True"),
    ];
  }

  /* ── 高亮：注释 / 键 / 值三类，够读就行，不引外部 highlighter ── */
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  const comment = (text) => '<span class="cro-yaml__comment">' + esc(text) + "</span>";

  function splitComment(text) {
    // 本文件生成的值里不含 '#'，按第一个 " #" 切开即可
    const at = text.indexOf(" #");
    return at >= 0 ? [text.slice(0, at), text.slice(at)] : [text, ""];
  }

  function highlightYaml(text) {
    if (!text) return "";
    if (text.trimStart().startsWith("#")) return comment(text);
    const [code, tail] = splitComment(text);

    const m = code.match(/^(\s*)([A-Za-z0-9_.-]+)(:)(.*)$/);
    let html;
    if (m) {
      const value = m[4];
      const cls = /^\s*(-?\d+(\.\d+)?|\[[^\]]*\])\s*$/.test(value) ? "num"
        : /^\s*(True|False|None|true|false|null)\s*$/.test(value) ? "bool"
        : "str";
      html = m[1]
        + '<span class="cro-yaml__key">' + esc(m[2]) + "</span>"
        + '<span class="cro-yaml__punct">:</span>'
        + (value ? '<span class="cro-yaml__' + cls + '">' + esc(value) + "</span>" : "");
    } else {
      html = esc(code);
    }
    return html + (tail ? comment(tail) : "");
  }

  function highlightShell(text) {
    if (!text) return "";
    if (text.trimStart().startsWith("#")) return comment(text);
    const [code, tail] = splitComment(text);
    const html = esc(code)
      .replace(/(--[a-z_-]+)(=?)/g,
        '<span class="cro-yaml__key">$1</span><span class="cro-yaml__punct">$2</span>')
      .replace(/^(\s*)(msrun)\b/, '$1<span class="cro-yaml__bool">$2</span>')
      .replace(/\\$/, '<span class="cro-yaml__punct">\\</span>');
    return html + (tail ? comment(tail) : "");
  }

  function paint(el, lines, highlighter, changed) {
    if (!el) return;
    el.innerHTML = lines.map((line) => {
      /* field 可以是一组字段（batch_size 由 MBS × DP × micro_batch_num 乘出来），
         其中任何一个动了这一行的数字就变了，只认头一个会漏标。 */
      const fields = Array.isArray(line.field) ? line.field : (line.field ? [line.field] : []);
      const mark = fields.some((key) => changed.has(key)) ? " is-changed" : "";
      return '<li class="cro-yaml__line' + mark + '"><code>' + highlighter(line.text) + "</code></li>";
    }).join("");
    el.dataset.raw = lines.map((line) => line.text).join("\n");
  }

  function render(topology) {
    const codeEl = doc.getElementById("croYamlCode");
    if (!codeEl || !topology) return;

    const defaults = (topology.preset && topology.preset.defaults) || {};
    const changed = new Set(Object.keys(defaults)
      .filter((key) => topology.config[key] !== defaults[key]));

    paint(codeEl, buildYamlLines(topology), highlightYaml, changed);
    paint(doc.getElementById("croLaunchCode"), buildLaunchLines(topology), highlightShell, changed);

    const pathEl = doc.getElementById("croYamlPath");
    if (pathEl) pathEl.textContent = yamlPath(topology.preset);

    /* 标题栏只在**出事**时说话：改了几项由行号旁的 M 标记自己交代（见
       .cro-yaml__line.is-changed），不必再用一句「N 项已改」复述一遍；
       校验没过则必须有一句，否则用户只会看到文件头那几行注释。 */
    const statusEl = doc.getElementById("croYamlStatus");
    if (statusEl) {
      statusEl.textContent = topology.valid
        ? ""
        : "校验未通过 · " + topology.errors.length + " 处冲突";
      statusEl.dataset.level = topology.valid ? "ok" : "bad";
    }
  }

  /* 复制：从 dataset.raw 取纯文本（行号是 CSS counter，不会被一起带走） */
  function bindCopy(buttonId, codeId) {
    const btn = doc.getElementById(buttonId);
    if (!btn) return;
    btn.addEventListener("click", () => {
      const raw = doc.getElementById(codeId)?.dataset.raw || "";
      const label = btn.querySelector(".cro-yaml__copy-label");
      const done = (ok) => {
        if (!label) return;
        label.textContent = ok ? "已复制" : "复制失败";
        setTimeout(() => { label.textContent = "复制"; }, 1600);
      };
      if (global.navigator?.clipboard?.writeText) {
        global.navigator.clipboard.writeText(raw).then(() => done(true), () => done(false));
      } else {
        done(false);
      }
    });
  }

  /* ── 视图切换 ───────────────────────────────────────────────────────────
     三档只改 .cro-board 上的一个类，各档让位的范围不同：
       relation  默认，四域联动
       is-yaml   arch / moe / cluster 三个区 display:none，YAML 区顶上它们的
                 格位；整网列留着 —— 左边回答「这份 yaml 描述的是哪张网」。
       is-doc    连整网列一起让位（文档没有那层对照关系，留半张 3D 图只是
                 噪声），文档区吃满三列。见 css/config-relation-doc.css。
     本函数是 #croViewTabs 的唯一监听方：文档档的内容由 js/config-relation-doc.js
     渲染，但它不碰页签，避免两处各绑一个 click 互相打架。 */
  function setup() {
    const board = doc.getElementById("croBoard");
    const tabs = doc.getElementById("croViewTabs");
    if (!board || !tabs) return;

    const incidentView = doc.getElementById("croIncidentView");
    let mode = "relation";

    const apply = (next) => {
      mode = next === "yaml" || next === "doc" ? next : "relation";
      /* 事件模式与配置仿真模式互斥（主脚本 setIncidentLayout 会把 .cro-board
         整块 hidden）。切到 YAML / 文档时先走横幅关闭键这条既有通路退出事件，
         别在这里另写一份收尾逻辑 —— 这两档的内容都在 .cro-board 里面，事件
         模式下整块被藏起来，不退出就什么都看不到。 */
      if (mode !== "relation" && incidentView && !incidentView.hidden) {
        doc.getElementById("croIncidentBannerClose")?.click();
      }
      board.classList.toggle("is-yaml", mode === "yaml");
      board.classList.toggle("is-doc", mode === "doc");
      /* 运行事件栏在文档档整条不显示（见 css/config-relation-doc.css 的说明）。
         类挂在 .pto-ide-frame__workarea 上而不是 .cro-board 上：事件栏是 board
         的兄弟节点，board 上的类够不着它。
         只加 is-doc-view 这一个类、不碰 is-event-rail-collapsed —— 退出文档档时
         侧栏要回到用户原来那个展开/收起状态。 */
      doc.querySelector(".pto-ide-frame__workarea")
        ?.classList.toggle("is-doc-view", mode === "doc");
      tabs.querySelectorAll("[data-observer-view]").forEach((btn) => {
        const on = btn.dataset.observerView === mode;
        btn.classList.toggle("is-selected", on);
        btn.setAttribute("aria-selected", String(on));
      });
      if (mode === "yaml") render(global.croObserver && global.croObserver.topology);
      // 版面变了：关系连线画在 viewport 坐标上、刻度带宽度也是实测的，
      // 借主脚本已有的 resize 通路重排一次。
      global.dispatchEvent(new Event("resize"));
    };

    tabs.addEventListener("click", (event) => {
      const btn = event.target.closest("[data-observer-view]");
      if (btn) apply(btn.dataset.observerView);
    });

    /* 选中运行事件时主脚本会把 .cro-board 收起换成事件详情：那时 YAML / 文档
       两档都没有落脚点（它们的 DOM 就在 .cro-board 里，整块被藏起来了），所以
         · 先退回关系视图，页签状态才不会和画面对不上；
         · 整组页签随之隐藏 —— 一组点了没反应的页签比没有更糟。
       关闭横幅回到配置仿真态时再放出来。 */
    if (incidentView) {
      const syncTabs = () => {
        if (!incidentView.hidden && mode !== "relation") apply("relation");
        tabs.hidden = !incidentView.hidden;
      };
      new MutationObserver(syncTabs)
        .observe(incidentView, { attributes: true, attributeFilter: ["hidden"] });
      // 本页启动即选中第一个运行事件（主脚本 init 里的 selectIncident），
      // 那一趟发生在本文件挂上 observer 之前，首帧得自己对一次。
      syncTabs();
    }

    doc.addEventListener("cro:change", (event) => {
      if (mode === "yaml") render(event.detail);
    });

    bindCopy("croYamlCopy", "croYamlCode");
    bindCopy("croLaunchCopy", "croLaunchCode");
  }

  if (doc.readyState === "loading") doc.addEventListener("DOMContentLoaded", setup);
  else setup();
})(window);
