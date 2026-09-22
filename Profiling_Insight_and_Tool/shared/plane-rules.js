/* ═══════════════════════════════════════════════════════════════════════════
   plane-rules.js —— 「rank × layer 展平视图」两页共用的**规则层**

   ── 谁在用 ──────────────────────────────────────────────────────────────
     · training-run-twin-standalone/config-relation-plane.html
         展平画布是 **DOM**（绝对定位 div + transform），长在
         config-relation-observer.js 那套控制器上（croSelect / cro:change）。
     · rank-intro/rank-intro.html
         第 10 步与实战案例四的展平画布是 **canvas**（第 10 步要做「3D 卡阵压成
         平面」的逐格飞入插值，非 canvas 做不了），配置表单是那页的手写精简版。

   ── 为什么只抽规则、不抽渲染 ────────────────────────────────────────────
   两页的「展平 rank×layer + 左侧配置面板 + 联动」在产品上是同一个模式，但渲染
   是 DOM 与 canvas 两套、且是**有意分开**的实现（见上）。把渲染合成一个组件，
   必然有一边的呈现要变。所以这里只收**与画法无关、两页语义必须一致**的东西：
   口径换算、约束谓词、字段量程、配色与刻度梯子。改这一处，两页同时生效；
   格子怎么画、面板长什么样，仍然各归各的文件。

   ── 收进来的每一项都核对过「两页此刻逐位相同」 ──────────────────────────
   这是本文件的准入条件 —— 只有当两边现有实现算出来的结果完全一样时才收，
   否则会在「统一」的名义下悄悄改掉一边的行为。核对结论逐条写在下面各段里。
   两边**本来就不同**的规则（最典型的是「层数与 PP」：observer 允许不均分、
   rank-intro 要求整除）不合并，而是各留一个具名谓词并排放着 —— 让分歧显式、
   可读、可查，而不是藏在两个文件里互不知情。

   ── 刻意**没有**收进来的东西（不是漏做）────────────────────────────────
     · 渲染。DOM 那套在 config-relation-plane.js、canvas 那套在 rank-intro 页内，
       两套都保持原样。格子几何（列宽 / 行高 / 缝宽 / 三档详情门槛）跟着各自的
       画法走，也留在各页。
     · 错误文案。observer 侧要靠字段 label 的子串匹配去标红对应的 stepper，两页
       措辞不能合并；这里只给判据与该标红哪几格（RULES 的 fields）。
     · 建议修法（rank-intro 的 propose / observer 的 reconcile + proposeFix）。
       两者算法不同、结果也不同（一个是 6 轮就近修，一个在 2 的幂梯子上找解），
       合并会改掉两边的行为。判据共用，修法各归各。
     · rank-intro 案例四 setMode 里那句 DP 换算。它取 Math.round 且不夹上界，与
       convertDpAcrossEpMode 的 floor + 夹量程不同；那页切口径时 config 可能正处在
       「校验未过」的挂起态，改成共享那一份会动到现有行为，故原样留着。
     · observer 侧 FIELD_SPECS 的 label / group / step / digits / title。只有
       min / max / pow2 由本层统一（observer 里有一趟覆盖），理由见第七段。

   ── 改完请跑一遍自检 ────────────────────────────────────────────────────
     node Profiling_Insight_and_Tool/training-run-twin-standalone/tools/plane-rules-selfcheck.js
   它拿两页**改动前**的实现快照对拍本层的每一项（约 23 万项），所以能直接回答
   「这次改动有没有顺手动到别的行为」。有意改行为时，把它里面对应的快照基线一起
   改掉并记一句为什么。（tools/ 在 .gitignore 里，与 cro-selfcheck.js 同一约定。）

   ── 加载方式 ────────────────────────────────────────────────────────────
   经典全局脚本（与本仓 js/ 下那一套同风格，不是 ES module），挂
   window.PTOPlaneRules。必须排在读它的脚本之前：
     config-relation-plane.html      → 在 config-relation-observer.js 之前
     config-relation-observer.html   → 同上（该页自己没有展平画布，引它只为
                                       observer.js 跑得起来）
     rank-intro.html                 → 在页内主脚本之前
   ═══════════════════════════════════════════════════════════════════════════ */
(function (global) {
  "use strict";

  /* ══ 一、数值小工具 ═══════════════════════════════════════════════════════
     两页各有一份逐字相同的实现（rank-intro 的 clamp/niceStep、
     config-relation-plane.js 的 clamp/niceStep），合成一份。 */

  function gcd(a, b) { return b ? gcd(b, a % b) : a; }

  function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }

  /* 量尺刻度的 1/2/5 梯子：给定「一格在屏幕上多少像素」与「刻度之间至少要隔开
     多少像素」，返回该隔几格标一次。两页的左侧 rank 刻度都走它。
     ⚠️ 与两页原实现逐位相同（含 `step < 100000` 这个防跑飞的上界）。 */
  function niceStep(unitPx, minPx) {
    if (!(unitPx > 0)) return 1;
    var step = 1;
    var ladder = [1, 2, 5];
    var decade = 1;
    var i = 0;
    while (step * unitPx < minPx && step < 100000) {
      i += 1;
      if (i % 3 === 0) decade *= 10;
      step = ladder[i % 3] * decade;
    }
    return step;
  }

  /* ══ 二、EP 口径 ══════════════════════════════════════════════════════════
     三档：
       split       切出 —— EP 组从 DP 里切，不进 world 的乘积（主流）
       orthogonal  正交 —— EP 独占 rank、是 DP 之外的一根轴，进 world 的乘积
       mf          MindFormers —— EP 在 dp×mp（= dp×tp）域上切，同样不进乘积
     ⚠️ 两页的**拼写不同**：observer 写 'orthogonal'，rank-intro 写 'ortho'。
     本层认两种拼写并归一，两页因此都不必改自己已有的字面量。 */

  var EP_SPLIT = "split";
  var EP_ORTHOGONAL = "orthogonal";
  var EP_MF = "mf";
  var EP_MODES = [EP_SPLIT, EP_ORTHOGONAL, EP_MF];

  function normalizeEpMode(mode) {
    if (mode === EP_ORTHOGONAL || mode === "ortho") return EP_ORTHOGONAL;
    if (mode === EP_MF) return EP_MF;
    return EP_SPLIT;
  }

  /* ── 输入归一 ──────────────────────────────────────────────────────────
     两页的 config 字段名不一样，本层不强迫谁改名，而是两种拼写都读：
       层数      totalLayer | layers
       路由专家  routedExpert | routed
       Top-K     topK | topk
       EP 口径   epMode | mode（另兼容 observer 的老字段 moeOrthogonal）
     取值一律过 Math.max(1, …)，与 observer 里那批 `Math.max(1, config.x || 1)`
     同一个防御口径（0 / undefined 不该把除法炸成 Infinity）。 */
  function read(config) {
    var c = config || {};
    var mode = c.epMode !== undefined ? c.epMode
      : c.mode !== undefined ? c.mode
        : (c.moeOrthogonal ? EP_ORTHOGONAL : EP_SPLIT);
    return {
      mode: normalizeEpMode(mode),
      layers: num(c.totalLayer !== undefined ? c.totalLayer : c.layers, 1),
      dp: num(c.dp, 1),
      pp: num(c.pp, 1),
      tp: num(c.tp, 1),
      cp: num(c.cp, 1),
      ep: num(c.ep, 1),
      routed: num(c.routedExpert !== undefined ? c.routedExpert : c.routed, 1),
      topk: num(c.topK !== undefined ? c.topK : c.topk, 1),
    };
  }

  function num(v, lo) {
    var n = Number(v);
    return Number.isFinite(n) ? Math.max(lo, Math.floor(n)) : lo;
  }

  function isOrthogonal(config) { return read(config).mode === EP_ORTHOGONAL; }
  function isMf(config) { return read(config).mode === EP_MF; }

  /* ══ 三、拓扑标量 ═════════════════════════════════════════════════════════
     这一段是本文件的**核心**：两页原先各算一遍、算法写法不同但结果恒等的那批数。
     恒等性逐条验算见每个函数上面的注释 —— 收进来的前提就是它。 */

  /* EP 从哪个域里切出来。正交档没有「切出」这回事，返回 dp 只是让调用方不必分支
     （正交档的整除校验本来就不看这个数）。 */
  function epDomain(config) {
    var c = read(config);
    if (c.mode === EP_MF) return c.dp * c.tp;
    return c.dp;
  }

  /* EP 进不进 world 的乘积：只有正交档进。 */
  function epInWorld(config) {
    var c = read(config);
    return c.mode === EP_ORTHOGONAL ? c.ep : 1;
  }

  /* Total Rank / World Size。
     ── 恒等性验算 ──
       observer  : dp*pp*tp*cp*epInWorld
       rank-intro: dpSplit(c)*tp*pp*cp，其中正交档 dpSplit = dp*ep
       正交 → dp*ep*tp*pp*cp ≡ dp*pp*tp*cp*ep   ✓
       切出/mf → dp*tp*pp*cp ≡ dp*pp*tp*cp*1    ✓ */
  function parallelWorld(config) {
    var c = read(config);
    return c.dp * c.pp * c.tp * c.cp * epInWorld(config);
  }

  /* 「切出口径下的 DP」—— 也就是一份**非专家**权重在数据并行域里复制了多少份
     （ZeRO / FSDP 的分母）。三档：
       切出  dp
       正交  dp×ep   （每个 DP 副本横跨全部 EP rank）
       mf    dp       ← 不能写成 EDP×EP：那会把 TP 那一维重复算进来
     ── 恒等性验算：observer 的 dpReplicaOf ≡ rank-intro 的 dpS，三档逐档相同 ✓ */
  function dpReplica(config) {
    var c = read(config);
    return c.mode === EP_ORTHOGONAL ? c.dp * c.ep : c.dp;
  }

  /* EDP —— 「一整套完整专家」共有几个副本，也就是集群矩阵 / 展平图 d 轴的组数：
       正交  EDP = DP        （EP 是另一根轴，每个 DP 副本自带一整套专家）
       切出  EDP = DP ÷ EP
       mf    EDP = DP×TP ÷ EP
     ── 恒等性验算 ──
       observer  : 正交 → dp；否则 floor(epDomain / ep)
       rank-intro: groups = rows / groupRows = (dpSplit*tp) / groupRows
         正交 → (dp*ep*tp)/(ep*tp) = dp          ✓
         切出 → (dp*tp)/(ep*tp)    = dp/ep       ✓
         mf   → (dp*tp)/ep                       ✓

     ── ⚠️ 除不尽时的取整：两页原本不同，故做成参数而不是悄悄统一 ──────────
     整除性由各页的校验（RULES.epDomainDivisibleByEp）去拦，所以**凡是画得出来的
     配置这里都除得尽**，取整方式落不到实处 —— 对拍 20 万项的结论是：合法配置上
     两页零分歧，分歧只出现在那 34 类校验本来就会拦掉的配置上（如 DP 3 ÷ EP 2）。
     但那些配置仍会流过一些兜底路径（rank-intro 的 peakGB 会对尚未过校验的建议稿
     试算一遍显存），为了让「两页呈现逐位不变」连这条路也成立，这里保留两种取整：
       floor  observer 侧的防御性兜底（宁可少算一组，保证几何是整数格）
       round  rank-intro 侧 topoOf 的 Math.round(rows/groupRows)
     默认 floor；rank-intro 的替代口（topology）传 round。两者都夹了下界 1。 */
  function expertDataParallel(config, opts) {
    var c = read(config);
    if (c.mode === EP_ORTHOGONAL) return c.dp;
    var exact = epDomain(config) / c.ep;
    var rounding = opts && opts.rounding === "round" ? Math.round : Math.floor;
    return Math.max(1, rounding(exact));
  }

  /* 一个 EDP 副本组在展平图上占几行（一行 = 一个 rank）：
       mf      ep        （EP 已吃掉 mp 维）
       其余    ep × tp
     ── 与 rank-intro 的 topoOf.groupRows 逐档相同 ✓ */
  function epGroupRows(config) {
    var c = read(config);
    return c.mode === EP_MF ? c.ep : c.ep * c.tp;
  }

  /* 一个 PP stage 内有几行 rank（展平图的纵轴长度）。
     ── 与 rank-intro 的 topoOf.rows = dpSplit*tp 相同 ✓ */
  function ranksPerStage(config) {
    return dpReplica(config) * read(config).tp;
  }

  /* d 轴对人显示的名字。切出 / mf 档下矩阵的一行是 EDP 而不是 DP —— 表单写着
     DP 512、矩阵左侧却标 DP0–7，是两个不同的量重名，必须分开叫。EP=1（稠密）时
     EDP ≡ DP，仍叫 DP，不给没有专家的模型平添一个新词。
     ── 与 observer 的 dAxisName 判据相同（那边读 counts，这里读 config）✓ */
  function dAxisName(config) {
    var c = read(config);
    return c.mode !== EP_ORTHOGONAL && c.ep > 1 ? "EDP" : "DP";
  }

  /* 切换 EP 口径时按 EP 换算 DP，使 Total Rank 不变（同一份卡的两种读法：
     切出档读作 DP 512、正交档读作 DP 8）。上下界夹在 DP 的字段量程内。
     ── 与 observer 的 convertDpAcrossEpMode 相同 ✓ */
  function convertDpAcrossEpMode(dp, ep, toOrthogonal) {
    var spec = FIELD_SPECS.dp;
    var factor = Math.max(1, ep);
    var next = toOrthogonal ? Math.floor(dp / factor) : dp * factor;
    return Math.min(spec.max, Math.max(spec.min, next || 1));
  }

  /* 一次算齐展平图要的那批数。rank-intro 的 topoOf 现在就是拿它拼的；
     config-relation-plane 侧同名的量散在 counts 里，逐个取用。
     ⚠️ groups 取 round 而不是 floor —— 与 rank-intro topoOf 原有行为保持逐位一致，
     理由见 expertDataParallel 上面那段（只在被校验拦掉的配置上才有差别）。 */
  function topology(config) {
    var c = read(config);
    return {
      mode: c.mode,
      orthogonal: c.mode === EP_ORTHOGONAL,
      mf: c.mode === EP_MF,
      dp: dpReplica(config),            // 切出口径的 DP（= 每段行数 ÷ TP）
      pp: c.pp,
      tp: c.tp,
      cp: c.cp,
      ep: c.ep,
      rows: ranksPerStage(config),      // 一个 stage 内的行数
      groupRows: epGroupRows(config),   // 一个 EDP 副本组占几行
      groups: expertDataParallel(config, { rounding: "round" }), // EDP：副本组共几个
      world: parallelWorld(config),
      dAxis: dAxisName(config),
    };
  }

  /* ══ 四、约束谓词 ═════════════════════════════════════════════════════════
     只答「成不成立、涉及哪几个字段、相关的数是多少」，**不产文案**：
     两页的错误文案不一样（observer 那套还要靠 label 子串去标红对应的 stepper），
     合并文案就会改掉一边的呈现。所以这里给事实，文案各页自己写。

     返回 null = 这条约束此刻不成立（没有错）；返回对象 = 违反了，对象里带着
     写文案要用的那几个数与 fields（该标红哪几格）。 */

  var RULES = {
    /* 路由专家必须能均分到 EP rank。两页都有、口径相同 ✓ */
    routedDivisibleByEp: function (config) {
      var c = read(config);
      if (c.routed % c.ep === 0) return null;
      return { rule: "routedDivisibleByEp", fields: ["routedExpert", "ep"],
               routed: c.routed, ep: c.ep, remainder: c.routed % c.ep };
    },

    /* EP 组必须凑得齐：切出档看 DP，mf 档看 DP×TP 域，正交档不受这条管
       （那里每个 DP 副本自带一整套专家，DP 与 EP 之间无须整除）。
       两页都有、口径相同 ✓ —— 这也是本层最值钱的一条：三档的域怎么取，
       以后只在 epDomain 一处改。 */
    epDomainDivisibleByEp: function (config) {
      var c = read(config);
      if (c.mode === EP_ORTHOGONAL) return null;
      var domain = epDomain(config);
      if (domain % c.ep === 0) return null;
      return { rule: "epDomainDivisibleByEp",
               fields: c.mode === EP_MF ? ["dp", "tp", "ep"] : ["dp", "ep"],
               mode: c.mode, dp: c.dp, tp: c.tp, ep: c.ep,
               domain: domain, remainder: domain % c.ep };
    },

    /* Top-K 不能超过路由专家总数。两页都有、口径相同 ✓ */
    topKWithinRouted: function (config) {
      var c = read(config);
      if (c.topk <= c.routed) return null;
      return { rule: "topKWithinRouted", fields: ["topK", "routedExpert"],
               topk: c.topk, routed: c.routed };
    },

    /* ── 层数与 PP：两页**本来就不同**，故并排两条，不合并 ────────────────
       observer（config-relation-plane）允许不均分 —— openPangu 的 46 层配 PP 4
       会摆成 12,12,11,11，它只要求每个 stage 至少有 1 层；
       rank-intro 的案例四要求整除 —— 那页的展平图按等长分块画，除不尽画不出来。
       谁调用哪一条，由调用方自己定；这里只把两种口径都备好、并写清差别。 */
    layersAtLeastPp: function (config) {          // observer 侧用这条
      var c = read(config);
      if (c.layers >= c.pp) return null;
      return { rule: "layersAtLeastPp", fields: ["totalLayer", "pp"],
               layers: c.layers, pp: c.pp };
    },
    layersDivisibleByPp: function (config) {      // rank-intro 侧用这条
      var c = read(config);
      if (c.layers % c.pp === 0) return null;
      return { rule: "layersDivisibleByPp", fields: ["totalLayer", "pp"],
               layers: c.layers, pp: c.pp, remainder: c.layers % c.pp };
    },

    /* 并行度的乘积必须等于 Total Rank（observer 侧：Total Rank 是可输入的一格）。 */
    worldMatchesTotalRank: function (config, totalRank) {
      var world = parallelWorld(config);
      if (world === totalRank) return null;
      return { rule: "worldMatchesTotalRank", fields: ["totalRank"],
               world: world, totalRank: totalRank, epInWorld: epInWorld(config) !== 1 };
    },

    /* 卡数必须是整机的整数倍（rank-intro 侧：整机 8 卡）。 */
    worldMultipleOfNode: function (config, ranksPerNode) {
      var per = Math.max(1, ranksPerNode || 8);
      var world = parallelWorld(config);
      if (world % per === 0) return null;
      return { rule: "worldMultipleOfNode", fields: ["totalRank"],
               world: world, ranksPerNode: per, remainder: world % per };
    },
  };

  /* ══ 五、专家权重着色 ═════════════════════════════════════════════════════
     颜色说的是**这一层的专家权重是哪一份**：索引只取 层号 % 4，左右相邻换层就
     换色。副本号**不参与** —— 上下两个 EDP 副本持的是同一份权重（步末在 EDP 组里
     all-reduce 对齐），换色会被读成「两套不同的专家」。副本边界交给行距与左量尺。
     色值取 training-run-twin.css 里 .ep-tint-N 那套的前四色，不另造色板。
     ── 两页此刻逐位相同 ✓：config-relation-plane.css 的 --crop-set-0..3 与
        rank-intro 的 SETC 是同四个十六进制值，中性灰也同值；两边的索引都是
        `layer % 4`、Dense 层都退到中性灰。 */
  var SET_TINTS = 4;
  var SET_COLORS = ["#3B82F6", "#F59E0B", "#10B981", "#8B5CF6"];
  /* 着色**关掉**时（以及 Dense 层）退到的那一色。必须是真中性的灰蓝：退到 MoE 的
     语义色（橙红）会让满屏读起来像一片告警，而这一档的本意恰恰是「不表达分组」。 */
  var SET_NEUTRAL = "#64748B";

  function setTintIndex(layer) {
    return ((layer % SET_TINTS) + SET_TINTS) % SET_TINTS;
  }

  /* opts.dense = 这一列是 Dense 层（没有专家，不参与着色）
     opts.paint = 「按层给专家权重着色」开着吗（关掉就整幅退到中性灰） */
  function setColorOf(layer, opts) {
    var o = opts || {};
    if (o.dense || o.paint === false) return SET_NEUTRAL;
    return SET_COLORS[setTintIndex(layer)];
  }

  /* ══ 六、热力色阶 ═════════════════════════════════════════════════════════
     冷蓝 → 火红，中途**绕开绿**：绿在这套色板里是「安全 / 正常」的语义色，热力图
     中段出现一片绿会被读成一档状态，而它其实只是「不冷不热」。
     ── 两页此刻逐位相同 ✓：config-relation-plane.js 的 HEAT_RAMP 与 rank-intro
        案例二的 RAMP_D 是同一张表，插值算法也逐字相同。
     rank-intro 另有一张浅色主题的 RAMP_L（那页可切主题，config-relation-plane
     固定深色），故 heatColor 收一个可选的 ramp 参数，默认这张深色表。 */
  var HEAT_RAMP = [
    [0.00, 13, 30, 66],
    [0.22, 38, 86, 178],
    [0.44, 104, 78, 196],
    [0.64, 176, 66, 148],
    [0.82, 226, 68, 84],
    [1.00, 255, 124, 46],
  ];

  function heatColor(u, ramp) {
    var R = ramp || HEAT_RAMP;
    var t = clamp(u, 0, 1);
    var i = 1;
    while (i < R.length - 1 && t > R[i][0]) i += 1;
    var a = R[i - 1];
    var b = R[i];
    var f = (t - a[0]) / ((b[0] - a[0]) || 1);
    var mix = function (j) { return Math.round(a[j] + (b[j] - a[j]) * f); };
    return "rgb(" + mix(1) + "," + mix(2) + "," + mix(3) + ")";
  }

  function heatRampCss(ramp) {
    var R = ramp || HEAT_RAMP;
    return "linear-gradient(90deg, " + R.map(function (s) {
      return "rgb(" + s[1] + "," + s[2] + "," + s[3] + ") " + Math.round(s[0] * 100) + "%";
    }).join(", ") + ")";
  }

  /* ══ 七、字段量程 ═════════════════════════════════════════════════════════
     这里是**规范值**（口径来自 config-relation-observer.js 的 FIELD_SPECS）。
     pow2 两页一致；min/max 与 label 两页**有意不同**，都由调用方用
     specOf(field, overrides) 显式覆盖 —— 分歧因此看得见、查得到，而不是两个文件
     各写一份、谁也不知道另一边是几：
       · min/max：rank-intro 的案例四是 128 卡量级的缩样，TP/CP 上界收到 16、
         Total Rank 下界抬到 8、Seq Length 上界放到 262144。
       · label：⚠️ 这里的 routedExpert / sharedExpert 取 **observer 侧**的写法
         （"Routed" / "Shared"）而不是 rank-intro 的（"Routed Expert" /
         "Share Expert"）—— observer 的 emit() 拿这些 label 在错误文案里做子串匹配
         来决定标红哪一枚 stepper，改了红圈会落错格子，是承重的。rank-intro 侧
         显式传自己那一套 label。
       pow2  加减键按 2 的幂走（手输仍可以是任意整数）
       两页已经一致的那些（Total Layer 1..256、DP 1..8192、PP 1..128、Top-K 1..64、
       Shared 0..8、Routed / EP 1..1024、Total Rank 上界 65536）从此只有这一处。 */
  var FIELD_SPECS = {
    totalLayer:   { label: "Total Layer",    min: 1,   max: 256,   step: 1 },
    dp:           { label: "DP",             min: 1,   max: 8192,  pow2: true },
    pp:           { label: "PP",             min: 1,   max: 128,   pow2: true },
    tp:           { label: "TP",             min: 1,   max: 64,    pow2: true },
    cp:           { label: "CP",             min: 1,   max: 64,    pow2: true },
    ep:           { label: "EP",             min: 1,   max: 1024,  pow2: true },
    routedExpert: { label: "Routed",         min: 1,   max: 1024,  pow2: true },
    topK:         { label: "Top-K",          min: 1,   max: 64,    step: 1 },
    sharedExpert: { label: "Shared",         min: 0,   max: 8,     step: 1 },
    totalRank:    { label: "Total Rank",     min: 1,   max: 65536, pow2: true },
    microBatch:   { label: "Micro Batch",    min: 1,   max: 64,    step: 1 },
    seqLen:       { label: "Seq Length",     min: 128, max: 131072, pow2: true },
  };

  /* 取一个字段的规格，可就地覆盖（只覆盖传进来的键，其余沿用规范值）。 */
  function specOf(field, overrides) {
    var base = FIELD_SPECS[field];
    if (!base) return null;
    var out = {};
    var k;
    for (k in base) if (Object.prototype.hasOwnProperty.call(base, k)) out[k] = base[k];
    if (overrides) for (k in overrides) if (Object.prototype.hasOwnProperty.call(overrides, k)) out[k] = overrides[k];
    return out;
  }

  /* ══ 导出 ═════════════════════════════════════════════════════════════════ */
  var API = {
    /* 小工具 */
    gcd: gcd, clamp: clamp, niceStep: niceStep,
    /* EP 口径 */
    EP_SPLIT: EP_SPLIT, EP_ORTHOGONAL: EP_ORTHOGONAL, EP_MF: EP_MF, EP_MODES: EP_MODES,
    normalizeEpMode: normalizeEpMode, readConfig: read,
    isOrthogonal: isOrthogonal, isMf: isMf,
    /* 拓扑标量 */
    epDomain: epDomain, epInWorld: epInWorld, parallelWorld: parallelWorld,
    dpReplica: dpReplica, expertDataParallel: expertDataParallel,
    epGroupRows: epGroupRows, ranksPerStage: ranksPerStage,
    dAxisName: dAxisName, convertDpAcrossEpMode: convertDpAcrossEpMode,
    topology: topology,
    /* 约束谓词 */
    RULES: RULES,
    /* 专家着色 */
    SET_TINTS: SET_TINTS, SET_COLORS: SET_COLORS, SET_NEUTRAL: SET_NEUTRAL,
    setTintIndex: setTintIndex, setColorOf: setColorOf,
    /* 热力色阶 */
    HEAT_RAMP: HEAT_RAMP, heatColor: heatColor, heatRampCss: heatRampCss,
    /* 字段量程 */
    FIELD_SPECS: FIELD_SPECS, specOf: specOf,
  };

  global.PTOPlaneRules = API;
  if (typeof module === "object" && module && module.exports) module.exports = API;
}(typeof globalThis !== "undefined" ? globalThis : this));
