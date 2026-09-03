/* ══ 导入配置（升级计划行 20）════════════════════════════════════════════════
   这一页至今是**单向**的：config-relation-yaml.js 把表单写成 yaml，却没有任何入口
   把一份现成的配置读回来（grep 不到 file input / FileReader / 粘贴框）。于是
   「拿一份真配置进来，页面接不接得住」这个问题根本问不出口 —— 第六批（行 21–32）
   的价值也就验不出来。这个文件补上那条通路。

   ── 它做三件事，第三件才是重点 ────────────────────────────────────────────
   1. 解析：yaml / sh / json 三种方言，一律拍平成 { 路径: 值 }；
   2. 映射：认识的键落到表单字段上，**并如实报出页面收下的是什么**
      （reconcile 会改数 —— 不报出来就是静默篡改，比报错更危险）；
   3. **交代没落点的键**。这是行 20 存在的真正理由：一份真配置有几百个键，页面只
      认得其中十几个。不把「读到了、故意没用」和「读到了、确实缺」分开摆出来，
      用户只会以为页面漏读了。四堆分别是：
        · 已识别 · 不建模 —— 通信 / 融合 / 调度器超参 / 数据集，不进显存模型；
        · 已识别 · 缺口   —— 不补数就是错的，逐条挂着升级计划的行号；
        · 结构常量        —— 由计算图给，本页不覆盖（见下面「谁赢」那段）；
        · 未识别          —— 没见过的键。**照样列出来**：宁可承认不认识，
                             也不能悄悄吞掉（吞掉就等于骗用户说「都读了」）。

   ── 谁赢：配置 vs 计算图 ──────────────────────────────────────────────────
   `totalLayer` / `routedExpert` / `topK` / `sharedExpert` 四项两边都给得出，
   升级计划要求「在行 20 里定死并在界面上说明」。规矩是：
     · 这四项 **配置赢** —— 它们本来就是表单里可调的字段，导入的目的就是让表单
       跳到那份配置的档位；不覆盖的话会出现「配置写 61 层、表单显示 46 层」这种
       没人解释得清的状态。
     · 结构常量（hidden / vocab / heads / kvHeads / intermediate / moeIntermediate
       / firstKDense / mtpLayers）**预设赢，一个都不覆盖** —— 它们由计算图给，
       是另一条路线的事。
   ⚠️ 但第二条有一个必须当面说清的后果：把 deepseek3 的并行度导到 openPangu 的
   结构上，算出来的容量**不是 deepseek3 的容量**。所以面板上单列一栏，把配置里的
   结构常量与当前预设的值并排摆出来，差多少一目了然。宁可显得啰嗦。

   ── 与其余模块的耦合面 ────────────────────────────────────────────────────
   只用两个已导出的全局：`croObserver.importConfig()`（本行新增的批量入口）与
   `CroTopology.MODEL_PRESETS / FIELD_SPECS / FLAG_SPECS`。解析层与 DOM 完全无关，
   整块导出到 `global.croImport`，好让 tools/cro-selfcheck.js 拿 11 份样本直接撞它
   —— 那 11 份配置本身就是这一批的验收标准，不该只能靠人点着页面试。
   ═══════════════════════════════════════════════════════════════════════════ */
(function (global) {
  "use strict";

  const doc = global.document;

  /* ══ 1. 解析：三种方言 → 拍平的 { 路径: 值 } ══════════════════════════════ */

  /* 值的还原：yaml / sh 里一切都是字符串，落到表单上要的是数和布尔。
     ⚠️ 不用 JSON.parse 兜底 —— "1e-2"、"99990,8,2"、"56GB" 这类它要么抛要么给出
     一个误导的数；这里只认明确的三种形态，其余原样留字符串。 */
  function coerce(raw) {
    if (typeof raw !== "string") return raw;
    const s = raw.trim().replace(/^['"]|['"]$/g, "");
    if (s === "") return "";
    if (/^(true|True|TRUE)$/.test(s)) return true;
    if (/^(false|False|FALSE)$/.test(s)) return false;
    if (/^(null|None|~)$/.test(s)) return null;
    // 纯整数 / 小数 / 科学计数（1.e-8 这种 yaml 写法也认）
    if (/^-?\d+$/.test(s)) return Number(s);
    if (/^-?\d*\.?\d+([eE][-+]?\d+)?$/.test(s)) return Number(s);
    if (/^-?\d+\.?[eE][-+]?\d+$/.test(s)) return Number(s);
    return s;
  }

  function flatten(value, prefix, out) {
    if (Array.isArray(value)) {
      out[prefix] = value;                    // 数组整体留着（offset / betas / recompute 都要看形状）
      return out;
    }
    if (value && typeof value === "object") {
      Object.keys(value).forEach((k) => flatten(value[k], prefix ? `${prefix}.${k}` : k, out));
      return out;
    }
    out[prefix] = value;
    return out;
  }

  function parseJsonText(text) {
    return flatten(JSON.parse(text), "", {});
  }

  /* ── yaml 子集 ─────────────────────────────────────────────────────────────
     只解析本页需要的那一层：缩进嵌套的映射、`- ` 列表、行内 `[a, b]`、注释、引号。
     **刻意不做**的：多行字符串（| 与 >）、复杂锚点合并（<<: *ref）、多文档（---）。
     锚点定义 `&name` 直接剥掉（deepseek3 的 train_dataset 有一个），引用 `*name`
     当普通字符串留着 —— 它们指向的都是数据集那一块，本页一个字段都不读。
     ⚠️ 这不是一个通用 yaml 解析器，别拿去解析别的东西。真要通用的得引库，而这一页
     是纯静态、无打包器（见 CLAUDE.md），引不进来。 */
  function parseYamlText(text) {
    const out = {};
    const stack = [];             // [{ indent, path }]
    const listIndex = {};         // path -> 下一个下标
    text.split(/\r?\n/).forEach((rawLine) => {
      // 去注释：只在「引号之外」的 # 处切
      let line = "";
      let quote = "";
      for (let i = 0; i < rawLine.length; i += 1) {
        const ch = rawLine[i];
        if (quote) { line += ch; if (ch === quote) quote = ""; continue; }
        if (ch === '"' || ch === "'") { quote = ch; line += ch; continue; }
        if (ch === "#" && (i === 0 || /\s/.test(rawLine[i - 1]))) break;
        line += ch;
      }
      if (!line.trim()) return;
      const indent = line.length - line.replace(/^\s*/, "").length;
      let body = line.trim();
      while (stack.length && indent <= stack[stack.length - 1].indent) stack.pop();
      const parent = stack.length ? stack[stack.length - 1].path : "";

      // 列表项：`- 值` 或 `- key: 值`
      if (body.startsWith("- ")) {
        const idx = listIndex[parent] || 0;
        listIndex[parent] = idx + 1;
        const item = body.slice(2).trim();
        const path = `${parent}[${idx}]`;
        const kv = item.match(/^([\w.-]+)\s*:\s*(.*)$/);
        if (kv) {
          out[`${path}.${kv[1]}`] = coerce(kv[2]);
          stack.push({ indent, path });
        } else {
          out[path] = coerce(item);
        }
        return;
      }

      const kv = body.match(/^([\w.$-]+)\s*:\s*(.*)$/);
      if (!kv) return;                                   // 解析不了的行整行跳过
      const key = kv[1];
      let val = kv[2].trim().replace(/^&\S+\s*/, "");     // 剥掉锚点定义
      const path = parent ? `${parent}.${key}` : key;
      if (val === "") {                                   // 下面是个子块
        stack.push({ indent, path });
        listIndex[path] = 0;
        return;
      }
      if (val.startsWith("[")) {                          // 行内数组：整行原样交给一个小解析
        out[path] = parseInlineList(val);
        return;
      }
      out[path] = coerce(val);
    });
    return out;
  }

  /* 行内数组，允许一层嵌套（deepseek3 的 offset 是 [[…],[…]]） */
  function parseInlineList(text) {
    try {
      const json = text
        .replace(/'/g, '"')
        .replace(/\bTrue\b/g, "true").replace(/\bFalse\b/g, "false")
        .replace(/,\s*]/g, "]");
      const v = JSON.parse(json);
      return Array.isArray(v) ? v : [v];
    } catch (e) {
      return text.replace(/^\[|\]$/g, "").split(",").map((s) => coerce(s));
    }
  }

  /* ── Megatron 命令行（.sh）───────────────────────────────────────────────
     两种形态都要收：`NAME=value` 的环境变量与 `--flag value` 的参数。
     `--flag` 后面若跟着另一个 `--` 或行尾，就是布尔开关（--sequence-parallel）。
     `$(( … ))` 这类算式不求值，但 WORLD_SIZE 那个乘法太常见，单独认一下 ——
     它是这份脚本里唯一说得出总卡数的地方。 */
  function parseShText(text) {
    const out = {};
    const vars = {};
    const lines = text.split(/\r?\n/);
    lines.forEach((rawLine) => {
      const line = rawLine.replace(/\\$/, "").trim();
      if (!line || line.startsWith("#")) return;

      const assign = line.match(/^([A-Za-z_][A-Za-z0-9_]*)=(.*)$/);
      if (assign) {
        let v = assign[2].trim();
        const def = v.match(/^\$\{[^:}]+:-\s*"?([^"}]*)"?\}$/);   // ${A:-"8"}
        if (def) v = def[1];
        vars[assign[1]] = v;
        if (!/^\$/.test(v)) out[assign[1]] = coerce(v);
        return;
      }

      // 一行里可能有多个 --flag
      const tokens = line.split(/\s+/);
      for (let i = 0; i < tokens.length; i += 1) {
        /* 引号里的参数也是参数：megatron_llama3_8b_fp8.sh 把 fp8 那几项放在一个
           bash 数组里（`\"--fp8-format hybrid\"`），不剥引号就整组读不到 ——
           而那份脚本整个主题就是 fp8（升级计划行 21）。只剥**紧挨着 --** 的那一个
           引号，免得把普通字符串里的横杠也当成参数。 */
        const tok = tokens[i].replace(/^["']--/, "--");
        if (!tok.startsWith("--")) continue;
        const key = tok.replace(/^--/, "").replace(/["']$/, "");
        const next = tokens[i + 1] === undefined ? undefined : tokens[i + 1].replace(/^["']/, "");
        if (next === undefined || next.startsWith("--") || next === ")") {
          out[key] = true;                       // 布尔开关
        } else {
          out[key] = next.replace(/[")]+$/, "");
          i += 1;
        }
      }
    });

    /* ⚠️ 真实脚本里那几个要紧的数**几乎全是变量引用**：
       `--tensor-model-parallel-size $TP_SIZE`。不解引用的话，落到表单上的会是
       字符串 "$TP_SIZE" —— 那比读不到更糟（一个看着像配置的假值）。
       所以最后统一回代一次，最多三层（变量套变量的情形有，套三层的没见过）。 */
    const resolve = (v, depth) => {
      if (typeof v !== "string" || depth > 3) return v;
      const hit = v.match(/^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?$/);
      if (!hit) return v;
      const next = vars[hit[1]];
      return next === undefined ? v : resolve(next, depth + 1);
    };
    Object.keys(out).forEach((k) => { out[k] = coerce(resolve(out[k], 0)); });
    Object.keys(vars).forEach((k) => { if (!(k in out)) out[k] = coerce(resolve(vars[k], 0)); });

    /* WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))：两个都是数就把它算出来。
       页面的 Total Rank 只有这一处来源 —— 两份 yaml 根本不写卡数（它在启动命令里，
       见 config-relation-yaml.js 开头那段解释）。节点数的变量名各家不一，都认一下。 */
    const per = Number(resolve(vars.GPUS_PER_NODE, 0));
    const nodes = Number(resolve(vars.NNODES !== undefined ? vars.NNODES : vars.NUM_NODES, 0));
    if (per > 0 && nodes > 0) out.__worldSize = per * nodes;
    return out;
  }

  function detectDialect(text) {
    const head = text.slice(0, 400).trim();
    if (head.startsWith("{")) return "json";
    if (/^\s*--[a-z-]+/m.test(text) && /\b(torchrun|GPUS_PER_NODE|MODEL_ARGS|TRAINING_ARGS)\b/.test(text)) return "sh";
    if (/^\s*[A-Z_]+=/m.test(text) && /--[a-z-]+/.test(text)) return "sh";
    return "yaml";
  }

  function parseAny(text, hintName) {
    const name = String(hintName || "");
    let dialect = detectDialect(text);
    if (/\.json$/i.test(name)) dialect = "json";
    else if (/\.(sh|bash)$/i.test(name)) dialect = "sh";
    else if (/\.ya?ml$/i.test(name)) dialect = "yaml";
    const flat = dialect === "json" ? parseJsonText(text)
      : dialect === "sh" ? parseShText(text)
        : parseYamlText(text);
    return { dialect, flat };
  }

  /* ══ 2. 键 → 表单字段 ═════════════════════════════════════════════════════
     `keys` 按**优先级**排，命中第一个就停：同一个概念在一份配置里可能出现好几次
     （deepseek3 的 seq_length 在 dataset 与 model_config 下各有一份），谁更权威
     要写死，不能靠遍历顺序碰运气。
     匹配走「整路径相等」或「以 .key 结尾」，所以 `parallel_config.data_parallel`
     与裸的 `data_parallel` 用同一条规则接得住。 */
  const MAP = [
    { field: "totalLayer", label: "层数",
      keys: ["model.model_config.num_layers", "model.model_config.num_hidden_layers",
        "num_hidden_layers", "num_layers", "num-layers"] },
    { field: "dp", label: "DP", keys: ["parallel_config.data_parallel", "data_parallel"] },
    { field: "tp", label: "TP",
      keys: ["parallel_config.model_parallel", "model_parallel", "tensor-model-parallel-size"] },
    { field: "pp", label: "PP",
      keys: ["parallel_config.pipeline_stage", "pipeline_stage", "pipeline-model-parallel-size"] },
    { field: "cp", label: "CP",
      keys: ["parallel_config.context_parallel", "context_parallel", "context-parallel-size"] },
    { field: "ep", label: "EP",
      keys: ["parallel_config.expert_parallel", "expert_parallel", "expert-model-parallel-size"] },
    { field: "vpp", label: "VPP", keys: ["model.model_config.pp_interleave_num", "pp_interleave_num"] },
    { field: "seqLen", label: "Seq Length",
      keys: ["model.model_config.seq_length", "seq-length", "cutoff_len", "seq_length"] },
    /* ⚠️ MindFormers 的 runner_config.batch_size **不是**每卡 micro-batch：
       full_batch: True 下它是全局 batch（升级计划行 12 查证过）。所以这里只接
       Megatron 与 HF Trainer 那两种明确写「每卡每次前反向」的键，MindFormers 那个
       交给缺口清单（行 22）—— 接错的后果是容量柱静默偏出几百倍。 */
    { field: "microBatch", label: "Micro Batch",
      keys: ["micro-batch-size", "per_device_train_batch_size"] },
    { field: "routedExpert", label: "Routed",
      keys: ["moe_config.expert_num", "expert_num", "num-experts", "n_routed_experts",
        "num_local_experts", "num_experts"] },
    { field: "topK", label: "Top-K",
      keys: ["moe_config.num_experts_chosen", "num_experts_chosen", "moe-router-topk",
        "num_experts_per_tok", "num_experts_per_token"] },
    { field: "sharedExpert", label: "Shared",
      keys: ["moe_config.shared_expert_num", "shared_expert_num", "n_shared_experts"] },
    { field: "totalRank", label: "Total Rank", keys: ["__worldSize"] },
    { field: "seqParallel", label: "序列并行 SP",
      keys: ["parallel_config.use_seq_parallel", "use_seq_parallel", "sequence-parallel"],
      to: (v) => Boolean(v) },
    { field: "vocabEmbDp", label: "词表走 DP",
      keys: ["parallel_config.vocab_emb_dp", "vocab_emb_dp"], to: (v) => Boolean(v) },
    /* 微批数（行 22）。MindFormers 与 HF 各有一个直接的键，Megatron 那支要反推，
       在 DERIVED 里。顺序即优先级：文件里明写的数永远赢反推的。 */
    { field: "microBatchNum", label: "micro_batch_num",
      keys: ["parallel_config.micro_batch_num", "micro_batch_num", "gradient_accumulation_steps"],
      /* DeepSpeed 那份写的是 "auto"（由外层 Trainer 填）—— 落一个字符串进去，
         报告里就会出现一枚看着像配置的假值。只收正整数。 */
      to: (v) => (Number.isInteger(Number(v)) && Number(v) >= 1 ? Number(v) : null) },
    { field: "loraRank", label: "LoRA Rank", keys: ["lora_rank"] },
  ];

  /* 几项要看好几个键才定得下来的，单独写成函数 —— 塞进 MAP 里会把那张表变成
     「一半是对照表、一半是逻辑」，读不下去。 */
  const DERIVED = [
    /* LoRA：LLaMA-Factory 用 finetuning_type 区分全参与 LoRA */
    {
      field: "lora", label: "LoRA",
      from: ["finetuning_type"],
      to: (flat) => {
        const t = pick(flat, ["finetuning_type"]);
        return t.hit ? { value: String(t.value).toLowerCase() === "lora", from: t.key } : null;
      },
    },
    /* 权重分片三档：三种方言各有各的说法，落点是同一枚控件。
       ⚠️ ZeRO-2 页面没有这一档（行 31 记着），命中时落到 zero1 并进缺口清单。 */
    {
      field: "shardMode", label: "权重分片",
      from: ["enable_parallel_optimizer", "use-distributed-optimizer", "zero_optimization.stage",
        "use-torch-fsdp2"],
      to: (flat) => {
        const fsdp = pick(flat, ["use-torch-fsdp2"]);
        if (fsdp.hit && fsdp.value) return { value: "fsdp2", from: fsdp.key };
        const stage = pick(flat, ["zero_optimization.stage"]);
        if (stage.hit) {
          const n = Number(stage.value);
          return { value: n >= 3 ? "fsdp2" : (n >= 1 ? "zero1" : "none"), from: stage.key };
        }
        const mf = pick(flat, ["parallel.enable_parallel_optimizer", "enable_parallel_optimizer"]);
        if (mf.hit) return { value: mf.value ? "zero1" : "none", from: mf.key };
        const mg = pick(flat, ["use-distributed-optimizer"]);
        if (mg.hit && mg.value) return { value: "zero1", from: mg.key };
        return null;
      },
    },
    /* 重计算四档：MindFormers 的 recompute 可能是布尔、也可能是逐 stage 的数组；
       Megatron 走 --recompute-granularity + --recompute-num-layers。 */
    {
      field: "recomputeMode", label: "重计算",
      from: ["recompute_config.recompute", "recompute_config.select_recompute",
        "recompute-granularity", "recompute-num-layers"],
      to: (flat) => {
        const gran = pick(flat, ["recompute-granularity"]);
        if (gran.hit) {
          const g = String(gran.value);
          if (g === "selective") return { value: "none", from: gran.key, note: "Megatron 的 selective 重算的是 attention 里的 softmax/dropout，本页按 FlashAttention 建模、这一项本来就不计入，等价于「关」" };
          const n = pick(flat, ["recompute-num-layers"]);
          return n.hit ? { value: "layers", from: n.key } : { value: "full", from: gran.key };
        }
        const sel = pick(flat, ["recompute_config.select_recompute", "select_recompute"]);
        const rec = pick(flat, ["recompute_config.recompute", "recompute"]);
        if (rec.hit && Array.isArray(rec.value)) return { value: "layers", from: rec.key };
        if (sel.hit && sel.value === true) return { value: "selective", from: sel.key };
        if (rec.hit) return { value: rec.value ? "full" : "none", from: rec.key };
        return null;
      },
    },
    {
      field: "recomputeLayers", label: "重算层数",
      from: ["recompute_config.recompute", "recompute-num-layers"],
      to: (flat) => {
        const n = pick(flat, ["recompute-num-layers"]);
        if (n.hit && Number(n.value) > 0) return { value: Number(n.value), from: n.key };
        const rec = pick(flat, ["recompute_config.recompute", "recompute"]);
        if (rec.hit && Array.isArray(rec.value)) {
          const flatArr = rec.value.flat ? rec.value.flat(2) : rec.value;
          const max = Math.max.apply(null, flatArr.map(Number).filter((v) => Number.isFinite(v)));
          if (max > 0) return { value: max, from: rec.key, note: "取数组里的最大值（本页的重算层数是一个统一的数，不逐 stage 配）" };
        }
        return null;
      },
    },
    /* CP 口径 */
    {
      field: "cpMode", label: "CP 口径",
      from: ["context_parallel_algo", "context-parallel-algo"],
      to: (flat) => {
        const a = pick(flat, ["parallel_config.context_parallel_algo", "context_parallel_algo",
          "context-parallel-algo"]);
        if (!a.hit) return null;
        const v = String(a.value);
        return { value: /ulysses/i.test(v) ? "ulysses" : "ring", from: a.key };
      },
    },
    /* ── 微批数（升级计划行 22）───────────────────────────────────────────
       三种方言给的是同一件事的三种写法，落点都是那枚「微批数」stepper：
         MindFormers  parallel_config.micro_batch_num —— 直接就是它；
         HF / DeepSpeed  gradient_accumulation_steps —— 同义词；
         Megatron  没有这个键，只给 --global-batch-size 与 --micro-batch-size，
                   微批数 = GBS / (MBS × DP)，而 DP 本身也是反推出来的。
       前两种在 MAP 里直接接；这里只补 Megatron 那一支，且**只在前两种都没命中时**
       才算 —— 否则会用一个反推值盖掉文件里明写着的数。
       除不尽就不落：那说明这份脚本的 GBS 与它自己的并行度对不上（公开样例多是模板，
       节点数等着人填），硬凑一个整数还不如让缺口清单如实报出来。 */
    {
      field: "microBatchNum", label: "micro_batch_num",
      from: ["global-batch-size", "micro-batch-size"],
      to: (flat) => {
        if (pick(flat, ["parallel_config.micro_batch_num", "micro_batch_num",
          "gradient_accumulation_steps"]).hit) return null;
        const g = pick(flat, ["global-batch-size"]);
        if (!g.hit) return null;
        const gbs = Number(g.value);
        const mbs = Number(pick(flat, ["micro-batch-size"]).value) || 1;
        const tp = Number(pick(flat, ["tensor-model-parallel-size"]).value) || 1;
        const pp = Number(pick(flat, ["pipeline-model-parallel-size"]).value) || 1;
        const cp = Number(pick(flat, ["context-parallel-size"]).value) || 1;
        const w = pick(flat, ["__worldSize"]);
        const dp = w.hit ? Number(w.value) / (tp * pp * cp) : 1;
        const num = gbs / (mbs * (Number.isInteger(dp) && dp >= 1 ? dp : 1));
        if (!Number.isFinite(num) || !Number.isInteger(num) || num < 1) return null;
        return {
          value: num, from: "--global-batch-size ÷ (--micro-batch-size × DP)",
          note: `Megatron 不直接写 micro_batch_num，由 GBS ${gbs} ÷ (MBS ${mbs} × DP `
            + `${Number.isInteger(dp) && dp >= 1 ? dp : 1}) 反推`,
        };
      },
    },
    /* ── 精度两档（升级计划行 21）─────────────────────────────────────────
       计算精度：四种写法，按**可信度**排而不是按方言排 ——
         --fp8-format / DTYPE="fp8" / quantization_config 里的 fp8 → FP8 档；
         compute_dtype: float16 / --fp16 → FP16；其余（含 --bf16）→ BF16。
       ⚠️ `quantization_config` 说的是**发布权重的量化格式**（DeepSeek-V3 的
       checkpoint 本来就是 fp8 e4m3 存的），不是训练精度。本页的这一档问的正是
       「权重按几个字节存」，所以接得住，但要在报告里把这句话说明白。
       DeepSpeed 的 `bf16.enabled` / `fp16.enabled` 常写成 "auto"（由外层 Trainer
       填），只认 === true，不把 "auto" 当开着。 */
    {
      field: "dtype", label: "计算精度",
      from: ["fp8-format", "compute_dtype", "bf16", "fp16", "quantization_config.quant_method"],
      to: (flat) => {
        const f8 = pick(flat, ["fp8-format"]);
        if (f8.hit) {
          const gather = pick(flat, ["fp8-param-gather"]);
          return { value: "fp8", from: f8.key,
            note: gather.hit && gather.value
              ? "同时开着 --fp8-param-gather（权重直存 fp8）—— 正是本页 FP8 档建模的那一种"
              : "⚠️ 没看到 --fp8-param-gather：不开它时 fp8 只是 matmul 前的一次转换，"
                + "权重仍按 2 B 常驻、另有一份 fp8 缓存；本页 FP8 档按「权重直存」估，这份配置会被高估省下的量" };
        }
        const q = pick(flat, ["quantization_config.quant_method", "quantization_config.fmt"]);
        if (q.hit && /fp8|e4m3|e5m2/i.test(String(q.value))) {
          return { value: "fp8", from: q.key,
            note: "这是**发布权重的量化格式**（checkpoint 本来就按 fp8 存），不是训练精度 —— "
              + "本页这一档问的正是「权重按几个字节存」，所以接得住，但别把它读成训练配方" };
        }
        const cd = pick(flat, ["model.model_config.compute_dtype", "compute_dtype"]);
        if (cd.hit) {
          const v = String(cd.value);
          if (/float16|fp16/i.test(v) && !/bfloat16/i.test(v)) return { value: "fp16", from: cd.key };
          return { value: "bf16", from: cd.key };
        }
        const f16 = pick(flat, ["fp16"]);
        if (f16.hit && f16.value === true) return { value: "fp16", from: f16.key };
        const b16 = pick(flat, ["bf16"]);
        if (b16.hit && b16.value === true) return { value: "bf16", from: b16.key };
        return null;
      },
    },
    {
      field: "paramsDtype", label: "主权重精度",
      from: ["params_dtype"],
      to: (flat) => {
        const d = pick(flat, ["model.model_config.params_dtype", "params_dtype"]);
        if (!d.hit) return null;
        const fp32 = /float32|fp32/i.test(String(d.value));
        return { value: fp32 ? "fp32" : "bf16", from: d.key,
          note: fp32
            ? "参数本身以 fp32 常驻：权重段 4 B/参数，优化器那一段少 4 B（master 就是参数）—— "
              + "总量仍是 16 B，但 ZeRO-1 能切走的从 12 B 掉到 8 B"
            : undefined };
      },
    },
    /* ── EP 口径（升级计划行 23）──────────────────────────────────────────
       MindFormers 的 expert_parallel 是在 **dp × mp 域**上切的，与页面默认的
       「切出档」（从 dp 里切）不是同一件事。两份 MindFormers 样本都踩在这个差别上：
         deepseek3  dp:4 / mp:8 / ep:32 —— 切出档 4 % 32 除不尽，当场红；
         qwen3      dp:1 / mp:4 / ep:4  —— 同样除不尽，配平会把 ep 收成 1，
                    专家并行度**被静默改没了**（报告里能看见，但那是一次无谓的改动）。
       所以判据是「这份配置写的是 MindFormers 的 parallel_config」而不是「除不尽」——
       按框架选档，不按算得通不通选档：否则同一个框架的两份配置会落到两个口径上。
       mp = 1 时两档等价（域就是 dp），仍落 mf —— 它说的是这份配置的出身。 */
    {
      field: "epMode", label: "EP 口径",
      from: ["parallel_config.expert_parallel", "parallel_config.model_parallel"],
      to: (flat) => {
        const ep = pick(flat, ["parallel_config.expert_parallel"]);
        const mp = pick(flat, ["parallel_config.model_parallel"]);
        if (!ep.hit || !mp.hit) return null;          // 不是 MindFormers 的 parallel_config
        const epv = Number(ep.value); const mpv = Number(mp.value);
        if (!(epv > 1)) return null;                  // EP=1 时三档无差别，不必换
        const dp = Number(pick(flat, ["parallel_config.data_parallel"]).value) || 1;
        return {
          value: "mf", from: ep.key,
          note: `MindFormers 在 dp×mp 域上切专家并行：${dp}×${mpv} = ${dp * mpv} ÷ EP ${epv}`
            + `${(dp * mpv) % epv === 0 ? ` = EDP ${(dp * mpv) / epv}` : "（除不尽，配平会收 EP）"}`
            + `${dp % epv !== 0 ? "；按页面默认的「切出」档这份配置会当场报错" : ""}`,
        };
      },
    },
    /* Megatron 不写 DP —— 它由 world / (TP×PP×CP) 得来。这是「切出」口径，
       正是页面的默认档（EP 从 DP 组内切出，不进乘积）。 */
    {
      field: "dp", label: "DP",
      from: ["__worldSize"],
      to: (flat) => {
        if (pick(flat, ["parallel_config.data_parallel", "data_parallel"]).hit) return null;
        const w = pick(flat, ["__worldSize"]);
        if (!w.hit) return null;
        const tp = Number(pick(flat, ["tensor-model-parallel-size"]).value) || 1;
        const pp = Number(pick(flat, ["pipeline-model-parallel-size"]).value) || 1;
        const cp = Number(pick(flat, ["context-parallel-size"]).value) || 1;
        const dp = Number(w.value) / (tp * pp * cp);
        if (Number.isInteger(dp) && dp >= 1) {
          return { value: dp, from: "GPUS_PER_NODE × NNODES ÷ (TP×PP×CP)" };
        }
        /* 除不尽：脚本里的总卡数与它自己写的并行度对不上 —— 公开样例常是**模板**
           （NUM_NODES=1 等着人自己填，而 TP8×PP16 至少要 128 张）。
           这时**必须落一个 DP**，不能返回 null：不落的话表单里那个 DP 还是上一个
           预设的 512，配平会把 Total Rank 顶到 512×TP×PP —— 导入一份 175B 的脚本
           得到 65536 张卡，比读不到还离谱。按 1 记，并在报告里说清这一跳。 */
        return {
          value: 1,
          from: "配置未写 DP",
          note: `脚本声明的总卡数 ${w.value} 与它自己的 TP${tp}×PP${pp}×CP${cp} = ${tp * pp * cp} 对不上`
            + `（公开样例多是模板，节点数等着自己填），DP 按 1 记 —— `
            + `Total Rank 因此由乘积反推，与脚本里那个数不同`,
        };
      },
    },
  ];

  /* ⚠️ epMode 不在 FIELD_SPECS / FLAG_SPECS 里（它走 controller.setEpMode，不是一枚
     stepper），所以 importConfig 会把它跳过 —— 面板那边单独调一次 setEpMode，
     见 applyImport()。这条别忘了，忘了的表现是「报告里说落了 EP 口径，页面却没换」。 */

  /* 路径匹配：整路径相等，或以 `.key` 结尾（`parallel_config.data_parallel` ↔ `data_parallel`） */
  function pick(flat, keys) {
    for (let i = 0; i < keys.length; i += 1) {
      const want = keys[i];
      if (Object.prototype.hasOwnProperty.call(flat, want)) return { hit: true, key: want, value: flat[want] };
      const suffix = "." + want;
      const found = Object.keys(flat).find((k) => k.endsWith(suffix));
      if (found) return { hit: true, key: found, value: flat[found] };
    }
    return { hit: false };
  }

  /* ── ③ 已识别 · 不建模 ────────────────────────────────────────────────────
     通信 / 融合 / 调度 / 数据流程侧。**整块前缀**优先（lr_schedule 那种一块几十个
     键，逐个列出来只会把真正要看的两栏淹掉），单键其次。 */
  const IGNORED_PREFIX = [
    ["lr_schedule", "学习率调度"], ["train_dataset", "数据集"], ["eval_dataset", "数据集"],
    ["dataset", "数据集"], ["callbacks", "回调 / 保存"], ["runner_wrapper", "训练包装"],
    ["trainer", "训练器类型"], ["tokenizer", "分词器"], ["processor", "预处理"],
    ["profile", "性能采集"], ["monitor_config", "监控"], ["swap_config", "换入换出策略"],
    ["metric", "评测"], ["eval_callbacks", "回调 / 保存"], ["auto_tune", "自动调优"],
    ["remote_save_url", "保存路径"], ["strategy_ckpt", "切分策略落盘"],
    ["train_dataset_task", "数据集"], ["eval_dataset_task", "数据集"],
    ["model", "模型定义（要紧的几项已落到表单或列进缺口）"],
    ["parallel", "并行上下文细节（要紧的几项已单列）"],
    /* 这三块里各有一两个键是缺口（batch_size 行 22 / max_device_memory 行 24 /
       swap 行 26 / type 行 31），已在归类时先被摘走，剩下的才折成一条。 */
    ["runner_config", "训练轮次与下沉（batch_size 已列进缺口）"],
    ["context", "执行上下文（max_device_memory 已列进缺口）"],
    ["optimizer", "优化器超参（type / swap 已列进缺口）"],
    ["parallel_optimizer_config", "分片细节（shard_size 已列进缺口）"],
    ["pipeline_config", "流水调度（已列进缺口 行 29）"],
    ["recompute_config", "重计算细节（已落到「重计算」四档）"],
    ["moe_config", "MoE 细节（专家数 / Top-K / 共享专家已落到表单）"],
    /* DeepSpeed：stage 已被「权重分片」用掉（见 DERIVED），这一块剩下的是通信与
       实现细节（overlap_comm / contiguous_gradients / bucket size…），不进显存模型。 */
    ["zero_optimization", "ZeRO 通信与实现细节（stage 已落到「权重分片」）"],
    /* DeepSpeed 的 fp16/bf16 是一整块（enabled + loss_scale + hysteresis…）。
       enabled 为 true 时已被「计算精度」档接走（行 21）；样本里写的是 auto，读不出来。 */
    ["fp16", "混合精度块（enabled 为 true 时已落到「计算精度」档；auto 读不出来）"],
    ["bf16", "混合精度块（同上）"],
    ["auto_map", "HF 远程代码映射"], ["rope_scaling", "位置编码缩放"],
    ["quantization_config", "量化配置（quant_method / fmt 已落到「计算精度」档 —— 那是发布权重的存储格式，不是训练精度）"],
  ];

  /* 逐个列名字列不完的两大类，用前缀正则收口。**只收这两类**：
       · 模型定义 —— 激活函数、位置编码、归一化、融合开关…… 由计算图给；
       · 训练流程 —— 学习率、步数、日志、断点、数据路径…… 不进显存模型。
     ⚠️ 正则是**兜底**不是主力：能写清楚名字的都写在上面两张表里。这里只负责把
     「一份真配置里必然存在的几十个与显存无关的键」折起来，免得它们淹掉正题。
     它排在缺口 / 结构常量 / 整块前缀之后，所以不会误吞要紧的键。 */
  const IGNORED_RE = [
    [/^(use-mcore-models|disable-bias-linear|max-position-embeddings|init-method-std|attention-dropout|hidden-dropout|normalization|position-embedding-type|swiglu|untie-|no-masked-softmax-fusion|no-position-embedding|rotary-|kv-channels|attention-backend|transformer-impl|attention-softmax|group-query-attention|squared-relu|apply-|add_bias|use_attn_mask|mla_|qk_layernorm|multi_latent|gated_linear|router_dense|moe_router|moe_aux|n_group|topk_group|use_pad_tokens|hidden_act|head_dim|rms_norm|rope_|use_cache|tie_word|torch_dtype|transformers_version|architectures|model_type|max_window|sliding_window|output_router|attention_bias|initializer_range|print_separate_loss|use_legacy|mtp_loss_factor|num_nextn)/,
      "模型定义（激活函数 / 位置编码 / 归一化 / 融合开关）—— 由计算图那条路线给"],
    [/^(lr|adam-|weight-decay|clip-grad|min-lr|train-iters|train_iters|exit-|log-|logging|eval-|eval_|save|load|tensorboard|wandb|vocab-file|merge-file|tokenizer|data-|split|seed|ckpt-format|ckpt_format|distributed-|manual-gc|timing|profile|train-samples|decoupled-|grad-reduce|cross-entropy-loss-fusion|calculate-per-token-loss|empty-unused-memory|moe-router-load|moe-aux|no-save|no-load|finetune|use-checkpoint-args|step-batch-size-schedule|accumulate-allreduce|no-gradient-accumulation-fusion|model_name_or_path|trust_remote_code|stage|do_train|template|max_samples|preprocessing|dataloader|plot_loss|overwrite|report_to|learning_rate|num_train_epochs|lr_scheduler|warmup|ddp_|val_size|per_device_eval|resume_|packing|flash_attn|auto_trans_ckpt|only_save_strategy|src_strategy|train_precision_sync|gradient_clipping|zero_allow|train_micro_batch_size_per_gpu|mock-data|no-create-attention-mask|no-mmap|num-workers|tiktoken)/,
      "训练流程 / 数据 / 日志 / 断点 —— 不进显存模型"],
  ];

  /* Shell 变量：启动器 / 路径 / 主机名那一堆全大写的赋值。它们在 sh 方言里动辄
     二三十个，逐条列出来只会把真正要看的两栏淹掉 —— 但也不能不提（不提就等于
     悄悄吞了）。所以整类折成一条，见 analyze()。 */
  const SH_VAR = /^[A-Z][A-Z0-9_]*$/;
  /* 启动器参数（torchrun / msrun 那一层）：它们决定的是「起几个进程」，
     总卡数已由 GPUS_PER_NODE × NNODES 单独接走。 */
  const LAUNCHER_KEY = ["nproc_per_node", "nnodes", "node_rank", "master_addr", "master_port",
    "distributed-backend", "tokenizer-type", "tokenizer-model", "data-path", "save", "load",
    "tensorboard-dir", "log-interval", "save-interval", "eval-interval", "eval-iters"];
  const IGNORED_KEY = [
    ["moe_token_dispatcher_type", "MoE 分发实现"], ["moe-token-dispatcher-type", "MoE 分发实现"],
    ["enable_alltoall", "通信实现"], ["gradient_aggregation_group", "梯度聚合分组"],
    ["overlap-grad-reduce", "通信与计算重叠"], ["overlap-param-gather", "通信与计算重叠"],
    ["moe_grouped_gemm", "算子融合"], ["moe-grouped-gemm", "算子融合"],
    ["use_fused_ops_topkrouter", "算子融合"], ["apply_rope_fusion", "算子融合"],
    ["bias_swiglu_fusion", "算子融合"], ["mp_comm_recompute", "通信重计算"],
    ["micro_batch_interleave_num", "多副本并行（通信掩盖）"],
    ["gradient_accumulation_shard", "梯度累积分片（通信侧）"],
    ["full_batch", "数据切分口径（决定 batch_size 怎么读，不进显存）"],
    ["dataset_strategy", "数据切分策略"], ["parallel_mode", "并行模式枚举"],
    ["search_mode", "自动并行搜索"], ["gradients_mean", "梯度归约方式"],
    ["parallel_optimizer_threshold", "分片阈值（下限以下不切）"],
    ["deepspeed", "指向**另一份** DeepSpeed 配置文件 —— 页面只读你贴进来的这一份，那一份里的 ZeRO 档要单独导一次"],
    /* 行 22 落地后这两个不再是缺口（页面有「微批数」那枚控件了），但仍然**不接**，
       各有各的理由 —— 写清楚比列进缺口更诚实。 */
    ["batch_size", "MindFormers 的 runner_config.batch_size 在 full_batch 下是**全局** batch（行 12），不是每卡 micro-batch。本页从 MBS × DP × 微批数 反过来算它，不读它 —— 读它就得反推 MBS，而这三个数里错一个全错"],
    ["train_batch_size", "DeepSpeed 的全局 batch，样本里写的是 auto（由外层 Trainer 填）；这份文件也没有卡数与 MBS，反推不出 micro_batch_num —— 导入后手拨那枚控件即可"],
    ["train_micro_batch_size_per_gpu", "同上，样本里是 auto"],
    ["gradient_accumulation_steps", "梯度累积步数 —— 已接到 micro_batch_num 那枚控件；这份文件写的是 auto，读不出来"],
    ["fp8-param-gather", "fp8 权重直存 —— 本页 FP8 档正是按它建模，已在「计算精度」那一行的备注里点名"],
    ["fp8-amax-history-len", "FP8 缩放因子的统计窗口，不进显存模型"],
    ["fp8-amax-compute-algo", "FP8 缩放因子的统计方式，不进显存模型"],
    ["init_start_profile", "性能采集"],
    ["seed", "随机种子"], ["output_dir", "输出路径"], ["load_checkpoint", "权重路径"],
    ["resume_training", "续训"], ["run_mode", "运行模式"], ["use_parallel", "是否分布式"],
    ["jit_level", "编译级别"], ["jit_config.jit_level", "编译级别"],
    ["train-iters", "训练步数"], ["lr", "学习率"], ["min-lr", "学习率"],
    ["weight-decay", "优化器超参"], ["clip-grad", "梯度裁剪"], ["split", "数据集划分"],
  ];

  /* ── ② 已识别 · 缺口（逐条挂着升级计划的行号）───────────────────────────── */
  const GAPS = [
    /* ⚠️ 行 21 / 22 已落地：精度两档与微批数都成了真控件，那 12 条从这里撤走了 ——
       params_dtype / compute_dtype / fp8-format / bf16 / fp16 / quantization_config
       与 micro_batch_num / global-batch-size / gradient_accumulation_steps 现在都在
       MAP 或 DERIVED 里接住（见上）；读不出来的那两个（DeepSpeed 的 auto、
       MindFormers 那个全局 batch）落到「已识别·不建模」并各自说明为什么。 */
    ["fp8", 21, "⚠️ 只在 DTYPE=\"fp8\" 这类 shell 变量里出现时才会走到这条 —— 真正的开关是 --fp8-format，那一个已接到「计算精度」档"],
    ["max_device_memory", 24, "框架实际可用显存 —— 容量框的高度现在取自卡型号的 HBM，比它高 6–8 GB"],
    ["offset", 25, "各 stage 层数偏移 —— 本页按「均分、除不尽前几段各多 1」分配"],
    ["swap", 26, "优化器 offload —— 开着时 12 B/参数整段移出 HBM，本页无此档"],
    ["offload_optimizer.device", 26, "优化器 offload，同上"],
    ["optimizer_weight_shard_size", 27, "ZeRO-1 的分母是这个显式子组，本页按整个 DP 域算"],
    ["use_flash_attention", 28, "本页把 FlashAttention 当公理（不计 5·a·s²/h 那一项），没有关掉它的档"],
    ["pipeline_scheduler", 29, "流水调度器（seqpipe / dualpipe）—— warmup 深度不同，本页只建了 1F1B"],
    ["pipeline_interleave", 29, "交错开关，本页由 VPP > 1 隐含"],
    ["moe_token_drop_policy", 30, "MoE 丢弃策略 —— 决定专家侧激活有没有上界"],
    ["capacity_factor", 30, "MoE 容量因子，同上"],
    ["moe-expert-capacity-factor", 30, "MoE 容量因子，同上"],
    ["optimizer.type", 31, "优化器类型 —— 本页按 Adam 的 12 B/参数写死"],
    ["lora_target", 32, "LoRA 作用面 —— all 会覆盖到 FFN，本页只按注意力四件套算"],
  ];

  /* ── 结构常量：由计算图给，本页不覆盖，但要把差异摆出来 ───────────────── */
  const STRUCTURAL = [
    ["hidden_size", "hidden", "hidden size"],
    ["hidden-size", "hidden", "hidden size"],
    ["vocab_size", "vocab", "词表"],
    ["vocab-size", "vocab", "词表"],
    ["num_attention_heads", "heads", "注意力头"],
    ["num-attention-heads", "heads", "注意力头"],
    ["num_key_value_heads", "kvHeads", "KV 头"],
    ["num-query-groups", "kvHeads", "KV 头（GQA 组数）"],
    ["intermediate_size", "denseIntermediate", "Dense FFN intermediate"],
    ["ffn-hidden-size", "denseIntermediate", "Dense FFN intermediate"],
    ["moe_intermediate_size", "moeIntermediate", "MoE 专家 intermediate"],
    ["moe-ffn-hidden-size", "moeIntermediate", "MoE 专家 intermediate"],
    ["first_k_dense_replace", "firstKDense", "前几层走 Dense"],
    ["num_nextn_predict_layers", "mtpLayers", "MTP 层数"],
  ];

  function matchTable(key, table) {
    const tail = key.split(".").pop();
    for (let i = 0; i < table.length; i += 1) {
      const want = table[i][0];
      if (key === want || tail === want || key.endsWith("." + want)) return table[i];
    }
    return null;
  }

  /* ══ 3. 归类：一份配置 → 一张可以照着读的报告 ═══════════════════════════ */
  function analyze(text, hintName, currentConfig) {
    const { dialect, flat } = parseAny(text, hintName);
    const presets = (global.CroTopology && global.CroTopology.MODEL_PRESETS) || {};
    const preset = presets[(currentConfig && currentConfig.model) || "openpangu-flash"] || {};

    const partial = {};
    const mapped = [];
    const used = new Set();

    MAP.forEach((rule) => {
      const hit = pick(flat, rule.keys);
      if (!hit.hit) return;
      const value = rule.to ? rule.to(hit.value) : hit.value;
      if (value === null || value === undefined || value === "") return;
      if (typeof value === "number" && !Number.isFinite(value)) return;
      partial[rule.field] = value;
      mapped.push({ field: rule.field, label: rule.label, from: hit.key, value });
      used.add(hit.key);
    });

    DERIVED.forEach((rule) => {
      const got = rule.to(flat);
      if (!got) return;
      partial[rule.field] = got.value;
      mapped.push({ field: rule.field, label: rule.label, from: got.from, value: got.value, note: got.note });
      rule.from.forEach((k) => {
        const hit = pick(flat, [k]);
        if (hit.hit) used.add(hit.key);
      });
    });

    const ignored = [];
    const gaps = [];
    const structural = [];
    const unknown = [];
    const seenPrefix = new Set();
    let shVars = 0;
    let launcher = 0;

    /* HF 的 config.json 整份就是**模型定义**（架构、层数、激活函数、生成超参），
       不是一份训练配置。所以那里没落点的键不该报成「未识别」—— 它们属于计算图
       那条路线，一律折成一条说清楚。判据用 architectures / model_type：
       DeepSpeed 的 json（zero_optimization / train_batch_size）没有这两个键。 */
    const isHfModelConfig = dialect === "json"
      && (Object.prototype.hasOwnProperty.call(flat, "architectures")
        || Object.prototype.hasOwnProperty.call(flat, "model_type"));
    let hfKeys = 0;

    /* 归类的**顺序即优先级**，改动前要想清楚：
         缺口 → 结构常量 → 整块前缀 → 单键不建模 → 启动器 → Shell 变量 → 未识别
       缺口排在最前，因为同一个键两边都可能命中（`bf16` 既是精度档也是个开关，
       `optimizer.swap` 的父块整块不建模），而挂着行号的那一堆是这张清单的正题。 */
    let accounted = 0;      // 每处理一个键 +1 —— 见下面 counts.accounted 的用途
    let usedKeys = 0;
    Object.keys(flat).forEach((key) => {
      if (key === "__worldSize") return;   // 派生量，不是文件里的键
      accounted += 1;
      if (used.has(key)) { usedKeys += 1; return; }
      const gap = matchTable(key, GAPS);
      if (gap) { gaps.push({ key, value: flat[key], row: gap[1], why: gap[2] }); return; }
      const st = matchTable(key, STRUCTURAL);
      if (st) {
        structural.push({ key, value: flat[key], field: st[1], label: st[2], preset: preset[st[1]] });
        return;
      }
      const root = key.split(/[.[]/)[0];
      const pfx = IGNORED_PREFIX.find((p) => p[0] === root);
      if (pfx) {
        if (!seenPrefix.has(root)) { seenPrefix.add(root); ignored.push({ key: root + " ·（整块）", count: 0, why: pfx[1] }); }
        ignored.find((x) => x.key === root + " ·（整块）").count += 1;
        return;
      }
      const ign = matchTable(key, IGNORED_KEY);
      if (ign) { ignored.push({ key, why: ign[1] }); return; }
      const tail = key.split(".").pop();
      const re = IGNORED_RE.find((r) => r[0].test(key) || r[0].test(tail));
      if (re) {
        const bucket = re[1];
        if (!seenPrefix.has(bucket)) { seenPrefix.add(bucket); ignored.push({ key: bucket, why: "", count: 0 }); }
        const row = ignored.find((x) => x.key === bucket);
        row.count += 1;
        return;
      }
      if (LAUNCHER_KEY.indexOf(key) >= 0) { launcher += 1; return; }
      // 全大写的裸赋值只出现在 sh 方言里，整类折成一条（否则二三十行淹掉正题）
      if (dialect === "sh" && SH_VAR.test(key)) { shVars += 1; return; }
      if (isHfModelConfig) { hfKeys += 1; return; }
      unknown.push({ key, value: flat[key] });
    });
    if (hfKeys) {
      ignored.push({ key: "模型定义", count: hfKeys, why: "HF config.json 描述的是模型本身（架构 / 激活函数 / 位置编码 / 生成超参），由计算图那条路线给，不是并行配置" });
    }
    if (shVars) ignored.push({ key: "Shell 变量", count: shVars, why: "启动器 / 路径 / 主机名（总卡数已由 GPUS_PER_NODE × 节点数单独接走）" });
    if (launcher) ignored.push({ key: "启动器参数", count: launcher, why: "torchrun / msrun 那一层：起几个进程、数据与权重路径" });

    /* 每个键都必须有归宿 —— 这是行 20 的核心承诺（不静默吞键）。
       ignoredKeys 数的是**键**不是行：整块折叠的那几条各自压着几十个键，
       用行数去核对总数会核不平，而核不平就说明有键被悄悄丢了。 */
    const ignoredKeys = ignored.reduce((n, i) => n + (i.count === undefined ? 1 : i.count), 0);
    // accounted 是**逐键**数出来的：每个键要么被用掉、要么落进某一堆，没有第三条路。
    // 它与 total 相等是行 20 的核心承诺（不静默吞键），自检台就断言这一条。
    const totalKeys = Object.keys(flat).filter((k) => k !== "__worldSize").length;
    const counts = {
      total: totalKeys, accounted, used: usedKeys, mapped: mapped.length, gaps: gaps.length,
      structural: structural.length, ignored: ignoredKeys, unknown: unknown.length,
    };
    return { dialect, flat, partial, mapped, ignored, gaps, structural, unknown, counts };
  }

  /* 导入之后表单**实际**停在哪 —— reconcile 会改数（EP 不整除 DP 时会被收回去），
     不把差异摆出来就是静默篡改。 */
  function diffApplied(partial, before, after) {
    const rows = [];
    Object.keys(partial).forEach((field) => {
      const want = partial[field];
      const got = after[field];
      if (got === want) return;
      rows.push({ field, want, got, was: before[field] });
    });
    return rows;
  }

  /* ── 样例配置目录 ─────────────────────────────────────────────────────────
     除了「自己贴 / 自己传」，再给一条**一步就能试**的路：直接切 config-test/ 里
     那 11 份公开配置。它们本来就是第六批（行 20–32）的验收材料，用户拿它们试一遍，
     等于把这一批的结论自己复核了一遍。

     ⚠️ 每一份都必须带**来源**与**它代表什么**：一个只写文件名的下拉框是没法选的
     —— 用户凭什么知道 `mixtral_8x7b.sh` 与 `megatron_175b.sh` 该先试哪个。
     `note` 那一句写的就是「选它能看到什么」，与 config-test/README.md 同源。

     数据全部于 2026-08-26 从公开仓下载（见 config-test/README.md 的来源表）。 */
  const SAMPLES = [
    { group: "MindFormers YAML（昇腾侧，最贴页面口径）", items: [
      { file: "mf_pretrain_deepseek3_671b.yaml", label: "DeepSeek-V3 671B 预训练",
        source: "mindspore-ai/mindformers · configs/deepseek3/",
        note: "并行度最全的一份：dp4 / mp8 / pp8 / ep32 + 交错流水 + ZeRO-1。也是唯一**必然被页面改数**的一份 —— 它的 EP 在 dp×mp 域上切（缺口行 23）" },
      { file: "mf_pretrain_qwen3_30b_a3b.yaml", label: "Qwen3-30B-A3B 预训练",
        source: "mindspore-ai/mindformers · configs/qwen3_moe/",
        note: "中等规模 MoE（128 专家 / topk8 / 无共享专家），且是唯一 `enable_parallel_optimizer: False` 的一份 —— 权重分片会落到「关」档" },
    ] },
    { group: "Megatron 命令行（GPU 侧）", items: [
      { file: "mixtral_8x7b.sh", label: "Mixtral 8×7B",
        source: "NVIDIA/Megatron-LM · examples/mixtral/",
        note: "MoE + EP8 + PP4 + SP，也是**唯一说得出总卡数**的一类（GPUS_PER_NODE × NNODES）—— 页面的 Total Rank 只能从这里来" },
      { file: "megatron_175b.sh", label: "GPT-3 175B",
        source: "NVIDIA/Megatron-LM · examples/gpt3/",
        note: "稠密大模型 TP8 × PP16、seq 2048。公开样例是**模板**（节点数留空），拿它能看到「配置自己对不上」时页面怎么报" },
      { file: "megatron_llama3_8b_fp8.sh", label: "Llama3 8B · FP8",
        source: "NVIDIA/Megatron-LM · examples/llama/",
        note: "唯一带 `--context-parallel-size` 的样本（验 CP），整份脚本的主题是 FP8 —— 而精度档正是缺口行 21" },
    ] },
    { group: "HF config.json（模型定义，不是训练配置）", items: [
      { file: "hf_deepseekv3_config.json", label: "DeepSeek-V3",
        source: "HF · deepseek-ai/DeepSeek-V3",
        note: "MLA + 256 专家 + 共享专家 + first_k_dense + MTP，与页面的 openPangu 预设同形 —— 拿它看「结构常量不覆盖」那一栏最直观" },
      { file: "hf_mixtral_config.json", label: "Mixtral 8×7B",
        source: "HF · mistralai/Mixtral-8x7B-v0.1",
        note: "结构侧的 MoE 字段（num_local_experts / num_experts_per_tok）怎么被接住" },
      { file: "hf_qwen2_7b_config.json", label: "Qwen2-7B",
        source: "HF · Qwen/Qwen2-7B",
        note: "与页面内置的 qwen2-7b 预设逐项对表 —— 体检用：结构常量该一个不差" },
    ] },
    { group: "HF Trainer + DeepSpeed（微调侧）", items: [
      { file: "lf_qwen3_lora_sft.yaml", label: "Qwen3 LoRA 微调",
        source: "hiyouga/LLaMA-Factory · examples/train_lora/",
        note: "验 LoRA 开关与 Rank。它写的是 `lora_target: all`（会覆盖到 FFN），而页面只按注意力四件套算 —— 缺口行 32" },
      { file: "lf_qwen3_full_sft.yaml", label: "Qwen3 全参微调",
        source: "hiyouga/LLaMA-Factory · examples/train_full/",
        note: "反例：微调侧口径（per_device_batch_size / 梯度累积 / 外挂 deepspeed 文件），并行度一个字都不写" },
      { file: "ds_z3_config.json", label: "DeepSpeed ZeRO-3",
        source: "hiyouga/LLaMA-Factory · examples/deepspeed/",
        note: "只有一件事：`zero_optimization.stage: 3` 落到「权重分片 → FSDP2」那一档" },
    ] },
  ];

  global.croImport = { parseAny, analyze, diffApplied, MAP, DERIVED, GAPS, STRUCTURAL, SAMPLES };

  /* ══ 4. UI ═══════════════════════════════════════════════════════════════
     ⚠️ 设计系统没有 dialog / drawer 原语（css/style.css 只有 .btn 系列），这里用
     tokens 拼一个最小实现，按钮一律复用 .btn —— 与本页的横幅、开关同属「缺失样式」，
     待批准后一并吸收进共享系统。 */
  function boot() {
    const openBtn = doc.getElementById("croImportConfig");
    const panel = doc.getElementById("croImportPanel");
    if (!openBtn || !panel) return;
    const el = {
      text: doc.getElementById("croImportText"),
      file: doc.getElementById("croImportFile"),
      parse: doc.getElementById("croImportParse"),
      apply: doc.getElementById("croImportApply"),
      close: doc.getElementById("croImportClose"),
      report: doc.getElementById("croImportReport"),
      name: doc.getElementById("croImportName"),
    };
    let current = null;          // 最近一次解析的报告

    const open = () => { panel.hidden = false; el.text.focus(); };
    const close = () => { panel.hidden = true; };

    openBtn.addEventListener("click", open);
    el.close.addEventListener("click", close);
    panel.addEventListener("click", (event) => { if (event.target === panel) close(); });
    doc.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !panel.hidden) close();
    });

    el.file.addEventListener("change", () => {
      const file = el.file.files && el.file.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        el.text.value = String(reader.result || "");
        el.name.textContent = file.name;
        runParse(file.name);
      };
      reader.readAsText(file);
    });

    el.parse.addEventListener("click", () => runParse(el.name.textContent));

    function runParse(hintName) {
      const text = el.text.value || "";
      if (!text.trim()) { el.report.innerHTML = `<p class="cro-import__empty">先贴一份配置，或选一个文件。</p>`; el.apply.disabled = true; return; }
      try {
        current = analyze(text, hintName, global.croObserver && global.croObserver.config);
      } catch (err) {
        current = null;
        el.apply.disabled = true;
        el.report.innerHTML = `<p class="cro-import__empty">解析失败：${escapeHtml(err.message)}`
          + `<br>本页只认 yaml / sh / json 三种方言的常见写法（多行字符串、锚点合并等未支持）。</p>`;
        return;
      }
      el.apply.disabled = !current.mapped.length;
      el.report.innerHTML = renderReport(current, null);
    }

    el.apply.addEventListener("click", () => {
      if (!global.croObserver || !global.croObserver.importConfig || !current) return;
      const result = global.croObserver.importConfig(current.partial);
      const adjusted = diffApplied(current.partial, result.before, result.after);
      el.report.innerHTML = renderReport(current, { result, adjusted });
      el.report.scrollTop = 0;

      /* 贴/传进来的这一份同样接管 yaml 栏的显示（与样例菜单同一条通路）：
         用户刚导入的文件就是他此刻关心的那份配置，那一栏却还画着本页生成的
         yaml，两边说的不是一件事。表单一动就退回生成视图。 */
      doc.dispatchEvent(new CustomEvent("cro:source", { detail: {
        name: el.name.textContent || "导入的配置",
        label: el.name.textContent || "导入的配置",
        dialect: current.dialect,
        text: el.text.value || "",
        marks: current.mapped.map((m) => {
          const off = adjusted.find((a) => a.field === m.field);
          return {
            field: m.field, token: String(m.from || "").split(".").pop(),
            label: m.label, value: m.value, got: off ? off.got : null,
          };
        }),
      } }));
    });
  }

  const escapeHtml = (s) => String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  /* SAMPLES 的 note 是照着行文写的（**着重**、`键名`），卡片上按原样渲染成
     粗体与等宽 —— 那两处标记本来就是「这句里哪个词要紧」的信息。 */
  const rich = (s) => escapeHtml(s)
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");

  const fmt = (v) => {
    if (Array.isArray(v)) return escapeHtml(JSON.stringify(v));
    if (typeof v === "boolean") return v ? "True" : "False";
    return escapeHtml(String(v));
  };

  /* 报告分五段，顺序就是用户该按的阅读顺序：落到哪 → 页面改了什么 →
     结构常量没换 → 缺口 → 不建模 / 未识别。 */
  function renderReport(r, applied) {
    const flagText = (global.CroTopology && global.CroTopology.flagText) || ((f, v) => String(v));
    const DIALECT = { yaml: "YAML", sh: "Shell 命令行", json: "JSON" };
    const out = [];

    /* 这一行是整张报告的账：**五堆之和必须等于总键数**。数的是键不是行
       （整块折叠的那几条各自压着几十个键），这样它才是一句可以核对的话 ——
       核不平就说明有键被悄悄丢了，而「不静默吞键」正是行 20 的核心承诺。 */
    const c = r.counts;
    out.push(`<p class="cro-import__lede">识别为 <b>${DIALECT[r.dialect] || r.dialect}</b>，`
      + `共读到 <b>${c.total}</b> 个键 ＝ `
      + `落到表单 <b>${c.used}</b>`
      + ` ＋ 已知缺口 <b>${c.gaps}</b>`
      + ` ＋ 结构常量 <b>${c.structural}</b>`
      + ` ＋ 读到但不建模 <b>${c.ignored}</b>`
      + ` ＋ 没认出来 <b>${c.unknown}</b>`
      + (c.accounted === c.total ? "" : `　⚠️ 有 ${c.total - c.accounted} 个键没交代（这是本页的 bug）`)
      + `</p>`);

    if (applied && applied.adjusted.length) {
      out.push(`<div class="cro-import__block is-warn"><h4>页面收下的与你给的不一样</h4>`
        + `<p class="cro-import__hint">配置之间要自洽（乘积、整除），所以导入后走了一遍配平。`
        + `下面这几项是<b>页面改过</b>的 —— 不说出来就是静默篡改。</p><ul>`
        + applied.adjusted.map((a) => `<li><code>${escapeHtml(a.field)}</code> `
          + `你给 <b>${fmt(a.want)}</b> → 页面停在 <b>${fmt(a.got)}</b></li>`).join("")
        + `</ul></div>`);
    }

    out.push(section("落到表单的", r.mapped.map((m) => `<li><b>${escapeHtml(m.label)}</b> = `
      + `<b>${fmt(m.value)}</b><span class="cro-import__from">← ${escapeHtml(m.from)}</span>`
      + (m.note ? `<br><span class="cro-import__note">${escapeHtml(m.note)}</span>` : "")
      + `</li>`), "配置赢：这几项表单里本来就可调，导入的目的就是让它们跳到这份配置的档位。"));

    if (r.structural.length) {
      out.push(`<div class="cro-import__block is-struct"><h4>结构常量 · 未覆盖（由计算图给）</h4>`
        + `<p class="cro-import__hint">这些是模型自身的形状，不是并行配置，本页<b>一个都不改</b> ——`
        + `它们由计算图那条路线给。<b>后果要当面说清：</b>把别人的并行度导到当前预设的结构上，`
        + `算出来的容量<b>不是那份配置的容量</b>。差多少见下表。</p><ul>`
        + r.structural.map((s) => `<li><b>${escapeHtml(s.label)}</b> 配置 <b>${fmt(s.value)}</b>`
          + ` · 当前预设 <b>${s.preset === undefined ? "—" : fmt(s.preset)}</b>`
          + (s.preset !== undefined && String(s.preset) !== String(s.value)
            ? ` <span class="cro-import__diff">不一致</span>` : "")
          + `</li>`).join("")
        + `</ul></div>`);
    }

    if (r.gaps.length) {
      out.push(`<div class="cro-import__block is-gap"><h4>已识别 · 缺口</h4>`
        + `<p class="cro-import__hint">页面读到了，但没有对应的旋钮 —— 每条挂着升级计划的行号。</p><ul>`
        + r.gaps.map((g) => `<li><code>${escapeHtml(g.key)}</code> = <b>${fmt(g.value)}</b>`
          + ` <span class="cro-import__row">行 ${g.row}</span><br>`
          + `<span class="cro-import__note">${escapeHtml(g.why)}</span></li>`).join("")
        + `</ul></div>`);
    }

    out.push(section("已识别 · 不建模", r.ignored.map((i) => `<li><code>${escapeHtml(i.key)}</code>`
      + (i.count ? ` <span class="cro-import__row">${i.count} 个键</span>` : "")
      + (i.why ? `<span class="cro-import__from">${escapeHtml(i.why)}</span>` : "")
      + `</li>`),
    "通信、算子融合、调度与数据流程侧 —— 它们不进显存模型，页面<b>故意</b>没用。", "is-skip"));

    if (r.unknown.length) {
      out.push(`<details class="cro-import__block is-unknown"><summary>未识别 ${r.unknown.length} 个键</summary>`
        + `<p class="cro-import__hint">没见过的键照样列出来 —— 宁可承认不认识，也不能悄悄吞掉。</p>`
        + `<ul>` + r.unknown.map((u) => `<li><code>${escapeHtml(u.key)}</code>`
          + `<span class="cro-import__from">${fmt(u.value)}</span></li>`).join("") + `</ul></details>`);
    }
    return out.join("");
  }

  function section(title, items, hint, cls) {
    if (!items.length) return "";
    return `<div class="cro-import__block ${cls || ""}"><h4>${title} <span>${items.length}</span></h4>`
      + (hint ? `<p class="cro-import__hint">${hint}</p>` : "")
      + `<ul>${items.join("")}</ul></div>`;
  }

  /* ══ 5. 样例配置切换（YAML 视图的文件名那一格）═══════════════════════════════
     入口不在导入面板里，而在 .cro-yaml__file —— 那一格本来就在回答「你现在看的是
     哪份配置」，换一份的动作理应从同一处发起。

     单层选择器：左边卡片菜单，右边解析报告，底部统一放【应用】【取消】。报告这步
     不能省 —— 一份陌生配置直接糊到表单上，用户看不出页面替他改了什么（EP 口径
     不合、VPP 被配平、总卡数反推……这些都写在报告里）。

     ⚠️ 模型不换。importConfig 只收 FIELD/FLAG 认得的字段，MAP 里根本没有 model ——
     样例的结构常量（hidden / layers / 专家数）一律不覆盖，那是计算图那条路线的活。 */
  function bootYamlPicker() {
    const trigger = doc.getElementById("croYamlPickerBtn");
    const menu = doc.getElementById("croYamlMenu");
    const options = doc.getElementById("croYamlOptions");
    if (!trigger || !menu || !options) return;
    const el = {
      tag: doc.getElementById("croYamlPickedTag"),
      name: doc.getElementById("croPreviewName"),
      source: doc.getElementById("croPreviewSource"),
      report: doc.getElementById("croPreviewReport"),
      apply: doc.getElementById("croPreviewApply"),
      cancel: doc.getElementById("croPreviewCancel"),
    };
    const DEFAULT_FILE = "__default__";
    let pending = null;   // 当前预览的那一份：{ meta, partial, adjustedNote }
    let previewToken = 0; // 连续点不同卡片时，较早返回的 fetch 不得覆盖当前报告

    const presetNow = () => {
      const presets = (global.CroTopology && global.CroTopology.MODEL_PRESETS) || {};
      const id = (global.croObserver && global.croObserver.config.model) || "openpangu-flash";
      return presets[id] || null;
    };

    /* 页面默认配置此前没有名字 —— 一个「别的选项都有名字、唯独你现在看的这一屏
       没有」的菜单是读不通的，用户会以为默认配置不在其中。 */
    function defaultMeta() {
      const p = presetNow();
      const d = (p && p.defaults) || {};
      return {
        file: DEFAULT_FILE,
        label: `页面默认 · ${(p && p.label) || "内置预设"}`,
        source: d.totalRank ? `本页内置 · ${d.totalRank} 卡` : "本页内置",
        note: "这一屏刚打开时的档位。卡型号与 EP 口径**不在恢复范围** —— 它们是硬件与读法，不是这份配置的内容。",
      };
    }

    const sampleOf = (file) => SAMPLES.reduce((hit, g) =>
      hit || g.items.find((i) => i.file === file), null);

    /* 卡片两行：第一行「名字 + 来源」，第二行「选它能看到什么」。
       不用 title —— 代表性是**选之前**要读到的东西，藏进悬浮里等于没写。 */
    const card = (meta) => `<button class="cro-yaml__opt" type="button" role="option"`
      + ` aria-selected="false" data-file="${escapeHtml(meta.file)}">`
      + `<span class="cro-yaml__opt-line"><b>${escapeHtml(meta.label)}</b>`
      + `<em>${escapeHtml(meta.source)}</em></span>`
      + `<span class="cro-yaml__opt-note">${rich(meta.note)}</span></button>`;

    function fillMenu() {
      const rows = [`<p class="cro-yaml__menu-head">换一份配置：先给解析报告，看完再决定应不应用。`
        + `模型与结构常量不跟着换。</p>`, card(defaultMeta())];
      SAMPLES.forEach((g) => {
        rows.push(`<div class="cro-yaml__opt-group">${escapeHtml(g.group)}</div>`);
        g.items.forEach((i) => rows.push(card(i)));
      });
      options.innerHTML = rows.join("");
    }

    function resetPreview() {
      pending = null;
      previewToken += 1;
      el.name.textContent = "请选择左侧配置文件";
      el.source.textContent = "";
      el.report.innerHTML = `<p class="cro-import__empty">选择一份配置后，这里会显示它能落到当前表单的配置项、页面配平会调整的数值，以及未建模或尚未支持的内容。</p>`;
      el.report.scrollTop = 0;
      el.apply.disabled = true;
    }

    function positionMenu() {
      const margin = 16;
      const anchor = trigger.getBoundingClientRect();
      const availableWidth = Math.max(280, global.innerWidth - anchor.left - margin);
      menu.style.width = `${Math.min(920, availableWidth)}px`;
    }

    function openMenu() {
      fillMenu();                       // 每次重建：模型可能已经切过，默认那张卡要换名字
      resetPreview();
      menu.hidden = false;
      trigger.setAttribute("aria-expanded", "true");
      positionMenu();
    }
    function closeMenu() {
      menu.hidden = true;
      trigger.setAttribute("aria-expanded", "false");
      resetPreview();
    }

    function showPreview(meta, html, canApply) {
      el.name.textContent = meta.label;
      el.source.innerHTML = `<b>来源</b> ${escapeHtml(meta.source)}　·　${rich(meta.note)}`;
      el.report.innerHTML = html;
      el.report.scrollTop = 0;
      el.apply.disabled = !canApply;
    }

    /* 「页面默认」没有文件可解析，走的是**恢复**而不是导入：把预设自己的 defaults
       整片喂回去（它本身自洽，配平不会改动它）。预览这一步照样给报告 ——
       列的是「这一下会改回哪几项」。 */
    function previewDefault() {
      const p = presetNow();
      if (!p) return;
      previewToken += 1;
      const meta = defaultMeta();
      const now = (global.croObserver && global.croObserver.config) || {};
      const rows = Object.keys(p.defaults || {})
        .filter((k) => now[k] !== p.defaults[k])
        .map((k) => `<li><code>${escapeHtml(k)}</code>`
          + `<span class="cro-import__diff">${fmt(now[k])} → ${fmt(p.defaults[k])}</span></li>`);
      pending = { meta, partial: p.defaults };
      showPreview(meta,
        `<div class="cro-import__block is-warn"><h4>恢复页面默认配置 <span>${rows.length}</span></h4>`
        + `<p class="cro-import__hint">${escapeHtml(p.label || "")} 的内置参考配置。`
        + (rows.length ? `应用会改回下面这几项：` : `表单现在就停在默认档位，应用不会动任何数。`)
        + `</p>` + (rows.length ? `<ul>${rows.join("")}</ul>` : "") + `</div>`,
        true);
    }

    function choose(file) {
      options.querySelectorAll(".cro-yaml__opt").forEach((cardEl) => {
        const selected = cardEl.dataset.file === file;
        cardEl.classList.toggle("is-selected", selected);
        cardEl.setAttribute("aria-selected", String(selected));
      });
      if (file === DEFAULT_FILE) { previewDefault(); return; }
      const meta = sampleOf(file);
      if (!meta) return;
      const token = ++previewToken;
      pending = null;
      showPreview(meta, `<p class="cro-import__empty">正在读取 <code>config-test/${escapeHtml(file)}</code> …</p>`, false);
      /* 走 fetch 读同目录下的文件：这一页本来就必须用 http 打开（见 CLAUDE.md，
         整页到处 fetch json / 嵌 iframe），file:// 下会被拦掉，如实说清楚。 */
      global.fetch(`./config-test/${file}`)
        .then((res) => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.text(); })
        .then((text) => {
          if (token !== previewToken) return;
          const r = analyze(text, file, global.croObserver && global.croObserver.config);
          /* token = 扁平化路径的末段（parallel_config.expert_parallel →
             expert_parallel）：yaml 视图拿它在原文里找「页面读的是哪几行」。 */
          pending = { meta, partial: r.partial, text, dialect: r.dialect,
            marks: r.mapped.map((m) => ({
              field: m.field, token: String(m.from || "").split(".").pop(),
              label: m.label, value: m.value, got: null,
            })) };
          el.report.innerHTML = renderReport(r, null);
          el.report.scrollTop = 0;
          el.apply.disabled = !r.mapped.length;
        })
        .catch((err) => {
          if (token !== previewToken) return;
          el.apply.disabled = true;
          el.report.innerHTML = `<p class="cro-import__empty">读不到 <code>config-test/${escapeHtml(file)}</code>`
            + `（${escapeHtml(err.message)}）。<br>这一页需要用 http 打开`
            + `（<code>python3 -m http.server</code>）—— <code>file://</code> 下浏览器不允许读同目录的文件。`
            + `也可以用顶栏「导入配置」把文件内容直接贴进去。</p>`;
        });
    }

    trigger.addEventListener("click", (event) => {
      event.stopPropagation();
      if (menu.hidden) openMenu(); else closeMenu();
    });
    options.addEventListener("click", (event) => {
      const btn = event.target.closest && event.target.closest(".cro-yaml__opt");
      if (btn) choose(btn.dataset.file);
    });
    doc.addEventListener("click", (event) => {
      if (!menu.hidden && !menu.contains(event.target) && !trigger.contains(event.target)) closeMenu();
    });

    el.cancel.addEventListener("click", closeMenu);
    doc.addEventListener("keydown", (event) => {
      if (event.key !== "Escape") return;
      if (!menu.hidden) closeMenu();
    });
    global.addEventListener("resize", () => { if (!menu.hidden) positionMenu(); });

    el.apply.addEventListener("click", () => {
      if (!pending || !global.croObserver || !global.croObserver.importConfig) return;
      const meta = pending.meta;
      const result = global.croObserver.importConfig(pending.partial);
      // 配平改掉的那几项：表单上会闪，这里再记进文件名旁的角标，免得一闪就没了
      const adjusted = diffApplied(pending.partial, result.before, result.after);

      /* yaml 栏换成这份文件的**原文**，文件名也换成它 —— 那一格回答的是「你现在
         看的是哪份配置」，名字写着 mixtral_8x7b.sh、内容却是按 openPangu 结构生成的
         yaml，两边说的不是一件事。顺带把「哪几行页面读了、哪几行没照收」标上。
         「页面默认」那一档没有原文，发一个空 detail 让它回到生成视图。 */
      const detail = pending.text ? {
        name: `config-test/${meta.file}`,
        label: meta.label,
        dialect: pending.dialect,
        text: pending.text,
        marks: (pending.marks || []).map((m) => {
          const off = adjusted.find((a) => a.field === m.field);
          return { ...m, got: off ? off.got : null };
        }),
      } : null;
      doc.dispatchEvent(new CustomEvent("cro:source", { detail }));

      closeMenu();
      if (el.tag) {
        // hidden 由 yaml 模块按「此刻显示的是不是原文」定（见那里的 render）
        el.tag.dataset.name = meta.file === DEFAULT_FILE ? "页面默认" : meta.file;
        el.tag.hidden = Boolean(detail);
        el.tag.textContent = el.tag.dataset.name;
        el.tag.title = `当前档位来自 ${meta.label}（${meta.source}）`
          + (adjusted.length ? `\n页面配平改了 ${adjusted.length} 项：`
            + adjusted.map((a) => `${a.field} ${a.want}→${a.got}`).join("、") : "")
          + `\n模型与结构常量未跟随切换。`;
      }
    });
  }

  if (doc.readyState === "loading") {
    doc.addEventListener("DOMContentLoaded", () => { boot(); bootYamlPicker(); });
  } else { boot(); bootYamlPicker(); }
})(window);
