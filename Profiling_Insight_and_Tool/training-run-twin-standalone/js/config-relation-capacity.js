/* ══ 单卡容量（Cluster 区右栏）══════════════════════════════════════════════
   这页原本只做**语法校验**（整除、world 一致、层数 ≥ PP）—— 配得通。但用户真正
   要知道的是**跑得下**，而单卡显存一直是个黑洞。这一栏把它点亮：把已经在手里的
   配置乘起来，算出「这张卡里装了什么、还剩多少」，把 OOM 从运行期挪到配置期。

   ── 与本页其余部分的耦合面 ────────────────────────────────────────────────
   刻意做成独立文件，只吃两个已有的 document 事件，不改 config-relation-observer.js
   的任何逻辑：
       cro:change  detail = topology   配置变了，重算
       cro:select  detail = relation   选中变了，切到被选中那张卡 / 那层所在 stage
   反向只用一个已导出的全局：global.croSelect（点 stage 小柱 = 选中该 stage 首卡）。

   ── 口径为什么不能照抄 dense 模型 ────────────────────────────────────────
   通行的 `12H²/层` 是 dense 口径。openPangu flash 92B 约 97% 的参数在专家里
   （256 路由专家 × 3 × H × I_moe ≈ 2.0B/层 × 44 层 ≈ 89B），套 dense 公式会算出
   一个完全错的数。所以这里逐层分算 attention / dense-MLP / 路由专家 / 共享专家 /
   router / emb / head 六项，各自除以真正切它的那一维：
       路由专家  ÷ EP × TP      ← MoE 下减容器的主力
       其余权重  ÷ TP
       所有权重  ÷ PP（体现为「这张卡只背本 stage 那几层」）
       DP        不除任何东西   ← 见 verdict 里那句话
   DP 那条是最反直觉也最有用的一条：把 DP 从 2 拉到 8，Total Rank 翻两番、集群矩阵
   多出几百格，容量柱纹丝不动。「显存不够就加卡」只有加 TP/PP/EP 时才成立。

   ── 三条不能省的建模细节 ────────────────────────────────────────────────
   1. 各 stage 层数不等（46/4 → 12,12,11,11），且 stage0 多背 embedding、末 stage
      多背 head —— 容量在集群上本来就不均，OOM 只需要一张卡爆。
   2. 1F1B 下 stage s 同时在飞 (PP−s) 份 micro-batch 的激活，stage0 最紧。
      不算这一项会系统性低估最危险的那张卡。
   3. 运行时开销必须算进去，且**不能做成 core 的固定百分比**。只画四段得到的是一
      个更好看的黑洞；而按已用量抽 10%，等于把一个几乎不随 core 变的量做成了正比
      项 —— 大 EP/大 PP 的轻卡被低估（光驱动+HCCL 就不止那点），重卡又虚高。这里
      拆成四项，各自跟着各自的标度量走（见 RUNTIME）：
        底座      固定，与配置无关         → 摞在盒底那一段
        通信 buffer  ∝ HCCL 通信域个数
        workspace  ∝ **一层**的 token 张量（不是全部层）
        碎片      ∝ 已用量                ← 只有这一项本来就该按比例
      后三项合成盒顶的「预留」段：它们会随配置动，正是这一页要给人看的东西。
   ═══════════════════════════════════════════════════════════════════════════ */
(function (global) {
  "use strict";

  const doc = global.document;
  const GIB = 1024 * 1024 * 1024;

  /* 口径常量。前三项（microBatch / seq / capGB）是**用户可调**的：它们是除并行维
     之外仅有的、能改变单卡占用的输入，写死了「事前配置校验」就缺一角。它们由
     Cluster 区表单那一行持有：Micro Batch / Seq Length 两个 stepper（训练超参）与
     卡型号下拉（硬件属性，HBM 决定容量）并排，每次渲染前由 syncBasis() 从
     topology.config / topology.card 同步过来；这里的值只是它们还没就绪时的兜底。
     其余是估算模型本身的系数，改它等于改整栏口径，同一组值也要出现在 basisHtml()
     （标题右侧那枚问号）里说给用户听 —— 不写等于让人猜。 */
  const BASIS = {
    seq: 4096,            // 序列长度 S ← config.seqLen
    microBatch: 1,        // 单卡 micro-batch ← config.microBatch
    capGB: 64,            // 单卡显存 ← topology.card.hbmGB（Cluster 区卡型号下拉）
    bytesWeight: 2,       // bf16
    bytesGrad: 2,         // bf16
    bytesOptim: 12,       // Adam：fp32 master weight + momentum + variance
    actPerLayer: 34,      // 每层激活 ≈ 34·mb·s·h 字节（开 SP、不重计算）
  };

  /* ══ 运行时开销的四项系数 ═══════════════════════════════════════════════
     这四个数是**经验值，待实测标定**。标定方法（写在口径浮层脚注里，别只留在
     注释中）：固定并行度、只改 micro-batch 跑两三次，取
         实测显存峰值 − 理论四段(weight/grad/optim/act)
     两点拟合 —— 截距 = 底座 + 通信 buffer（不随 mb 动），斜率 = workspace + 碎片
     （∝ mb）；再换一组 EP/TP 复跑一次，就能把通信域那一项从截距里分离出来。
     标定出来的值是跟着**卡型号**走的常量，届时应挪进 CARD_SPECS。 */
  const RUNTIME = {
    baseGB: 2.0,        // 驱动 + CANN/ACL context + kernel binary + 通信域元数据
    hcclBufGB: 0.2,     // 单个通信域的 HCCL_BUFFSIZE（默认 200 MB）
    hcclDouble: 2,      // 收发双缓冲
    wsFactor: 2,        // MoE permute/unpermute：进出各一份临时区
    wsFloorGB: 1.0,     // 纯 dense stage 由 FA 之类兜底的 workspace 下限
    fragRatio: 0.05,    // caching allocator 碎片，按已用量
  };

  /* config 里那两个字段名与 BASIS 的键名不同（一个是配置项名、一个是口径量名），
     这张表是唯一的对应关系，别在别处再写一遍。
     capGB 不在表里：它不是训练超参而是硬件属性，来自 Cluster 区卡型号下拉选中的
     那款卡的 HBM（topology.card.hbmGB）。 */
  const BASIS_FROM_CONFIG = { microBatch: "microBatch", seq: "seqLen" };

  function syncBasis(topo) {
    const config = (topo && topo.config) || {};
    Object.keys(BASIS_FROM_CONFIG).forEach((key) => {
      const v = config[BASIS_FROM_CONFIG[key]];
      if (Number.isFinite(v) && v > 0) BASIS[key] = v;
    });
    const hbm = topo && topo.card && topo.card.hbmGB;
    if (Number.isFinite(hbm) && hbm > 0) BASIS.capGB = hbm;
  }

  /* 两条警戒线。只有一条 OOM 线是不够的：预留段算的是**稳态值**，而通信 buffer
     与碎片本身还在逐 step 波动，越过 70% 之后这点波动就足以把余量吃光，那时候
     「还没满」不等于「跑得下」。 */
  const THRESHOLD = { tight: 0.70, alert: 0.88 };

  /* 自底向上的堆叠顺序。底座贴盒底、预留摞盒顶，中间四段才是「模型本身」：
     底座固定不动，调 EP/PP 时只有中间四段和盒顶那段会变，一眼能看出哪部分是
     配置能管的、哪部分是给运行时的死钱。 */
  const SEGS = [
    { key: "base", label: "运行时底座" },
    { key: "weight", label: "权重分片" },
    { key: "grad", label: "梯度" },
    { key: "optim", label: "优化器状态" },
    { key: "act", label: "激活分片" },
    { key: "reserve", label: "预留 buffer/碎片" },
  ];

  let topology = null;
  let pinnedStage = null;   // 选中某张卡/某层后钉住的 stage；null = 回落到最紧的卡
  let pinnedRank = null;

  const el = {};

  /* ══ 等距投影 ═══════════════════════════════════════════════════════════
     容器是有体积的东西，画成平面色条会丢掉「装」这层意思。这里用最小一套
     等轴测：y 轴朝上为高度（= 字节数），x/z 只是给盒子一个底面。
       屏幕 x = (x − z)·cos30°
       屏幕 y = (x + z)·sin30° − y
     全部几何在 user unit 里算，viewBox 由内容包围盒反推，所以面板宽高怎么
     变都不用重算——SVG 自己等比缩放。 */
  const ISO_C = Math.cos(Math.PI / 6);
  const ISO_S = Math.sin(Math.PI / 6);
  const NS = "http://www.w3.org/2000/svg";

  /* 盒子尺寸（user unit）。W/D 只决定底面观感，H 是「一张卡的显存」这段高度的
     标尺。H 相对 W/D 越高，整幅在面板里就越瘦长、越吃行高 —— 9.6 是压过一轮的
     值（原 12.0 太高，把 Cluster 那一行顶得过深）。 */
  const BOX = { w: 5.2, d: 4.0, h: 9.6, inset: 0.26 };

  function svgNode(tag, attrs) {
    const node = doc.createElementNS(NS, tag);
    Object.keys(attrs || {}).forEach((k) => {
      if (attrs[k] !== null && attrs[k] !== undefined) node.setAttribute(k, attrs[k]);
    });
    return node;
  }

  /* 颜色：段色来自 deck 语义色变量，可能是 #rgb / #rrggbb / rgb() / rgba()。
     三个面要按受光程度分明暗，所以必须拿到分量而不是直接用字符串。 */
  function parseColor(input) {
    const str = String(input || "").trim();
    if (str.startsWith("#")) {
      const h = str.slice(1);
      const full = h.length === 3 ? h[0] + h[0] + h[1] + h[1] + h[2] + h[2] : h.slice(0, 6);
      const n = parseInt(full, 16);
      if (Number.isNaN(n)) return null;
      return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    }
    const m = str.match(/rgba?\(([^)]+)\)/i);
    if (!m) return null;
    const parts = m[1].split(/[\s,/]+/).filter(Boolean).map(Number);
    if (parts.length < 3 || parts.some(Number.isNaN)) return null;
    return [parts[0], parts[1], parts[2]];
  }

  /* amt > 0 提亮、< 0 压暗。深浅主题都适用：受光面永远比背光面亮。 */
  function shade(rgb, amt) {
    if (!rgb) return "currentColor";
    const out = rgb.map((v) => {
      const next = amt >= 0 ? v + (255 - v) * amt : v * (1 + amt);
      return Math.max(0, Math.min(255, Math.round(next)));
    });
    return `rgb(${out[0]}, ${out[1]}, ${out[2]})`;
  }

  /* host 决定从哪个节点取值。deck 语义色定义在 .cro-board / .cro-incident-view
     上而不是 :root，调用方在哪棵子树里就得传哪棵的节点，否则拿到空串。 */
  function cssVar(name, fallback, host) {
    const node = host || el.root || doc.documentElement;
    if (!node) return fallback;
    const raw = global.getComputedStyle(node).getPropertyValue(name).trim();
    return raw || fallback;
  }

  /* 颜色允许写成 `var(--token)` 或 `var(--token, #fallback)`：三个面要按受光度分
     明暗，必须先解析成 RGB 分量。浏览器计算自定义属性时已经把嵌套的 var() 代换
     掉了，但深浅主题切换 / 未定义 token 的兜底链仍可能再套一层，故最多递归 3 层。 */
  function resolveColor(input, host, depth) {
    if (Array.isArray(input)) return input;                 // 已经是分量
    const str = String(input || "").trim();
    const m = str.match(/^var\(\s*(--[\w-]+)\s*(?:,\s*([\s\S]+))?\)$/);
    if (!m) return parseColor(str);
    const raw = cssVar(m[1], "", host);
    const level = depth || 0;
    if (raw && level < 3) {
      const hit = resolveColor(raw, host, level + 1);
      if (hit) return hit;
    }
    return m[2] && level < 3 ? resolveColor(m[2].trim(), host, level + 1) : null;
  }

  /* 建场景：一个虚线线框（容量）+ 若干实心盒（内容）+ 阈值环 + 溢出盒。
     视觉语法三条铁律：容量是线框、内容是实心、越界摞在盒口之上。

     ── 这个函数不认识「单卡显存」，只认识「一个容量 + 一摞内容」 ──
     事件详情里那两张显存构成图（问题1.3 的 64 GB 峰值构成、1.5 的触顶分布）讲的
     是同一件事，故走同一个 builder（见 config-relation-observer.js 的
     chartCapacity），经 global.croCapacityBox 导出。spec：
       cap        容量，与 segments[].value 同单位且 > 0
       segments   [{ label, value, color, dashed, opacity }]，**自底向上**摞
       thresholds [{ at, color }]，at ∈ (0,1) 的水位环，可省
       host       解析 var(--token) 的上下文节点（须已在文档里）
       format     (value) => "12.3 GB"，写进各段的原生 <title>
       ariaLabel  整幅图的可读名 */
  function buildBox(spec) {
    const cap = spec.cap > 0 ? spec.cap : 1;
    const host = spec.host || el.root;
    const fmt = spec.format || ((v) => String(v));
    const bounds = { x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity };
    // 阈值标签向左伸出的那截：只用来撑 viewBox 的左边界，不算进 bounds（见末尾）
    let overhangX0 = Infinity;
    const P = (x, y, z) => {
      const px = (x - z) * ISO_C;
      const py = (x + z) * ISO_S - y;
      if (px < bounds.x0) bounds.x0 = px;
      if (px > bounds.x1) bounds.x1 = px;
      if (py < bounds.y0) bounds.y0 = py;
      if (py > bounds.y1) bounds.y1 = py;
      return [px, py];
    };
    const pts = (list) => list.map((p) => `${p[0].toFixed(3)},${p[1].toFixed(3)}`).join(" ");

    const svg = svgNode("svg", { role: "img", "aria-label": spec.ariaLabel || "容量等距示意" });
    const g = svgNode("g", {});
    svg.appendChild(g);

    /* 一个实心盒：只画三个可见面（顶 / 右 / 前），按受光度分三档明暗。 */
    const solid = (y, h, rgb, opts) => {
      const o = opts || {};
      const x = BOX.inset;
      const z = BOX.inset;
      const w = BOX.w - BOX.inset * 2;
      const d = BOX.d - BOX.inset * 2;
      const p = (a, b, c) => P(x + a, y + b, z + c);
      const faces = [
        [[p(0, h, 0), p(w, h, 0), p(w, h, d), p(0, h, d)], 0.30],   // 顶：最亮
        [[p(0, 0, d), p(0, h, d), p(w, h, d), p(w, 0, d)], -0.06],  // 前
        [[p(w, 0, 0), p(w, h, 0), p(w, h, d), p(w, 0, d)], -0.30],  // 右：最暗
      ];
      const grp = svgNode("g", { class: o.cls || null, opacity: o.opacity || null });
      faces.forEach((face) => {
        grp.appendChild(svgNode("polygon", {
          points: pts(face[0]),
          fill: shade(rgb, face[1]),
          stroke: shade(rgb, -0.46),
          "stroke-width": 0.062,
          "stroke-linejoin": "round",
          "stroke-dasharray": o.dashed ? "0.18 0.14" : null,
        }));
      });
      if (o.title) {
        const t = svgNode("title", {});
        t.textContent = o.title;
        grp.appendChild(t);
      }
      return grp;
    };

    /* ── 内容：从底往上摞，超出容量的部分不进框 ── */
    let cursor = 0;
    let overflow = 0;
    let loaded = 0;
    (spec.segments || []).forEach((seg) => {
      const v = seg.value;
      if (!(v > 0)) return;
      loaded += v;
      const hFull = v / cap * BOX.h;
      const room = Math.max(0, BOX.h - cursor);
      const h = Math.min(hFull, room);
      overflow += hFull - h;
      if (h > 0.012) {
        g.appendChild(solid(cursor, h, resolveColor(seg.color, host), {
          cls: "cro-capacity__solid",
          // 「留出来的余量 / 用不上的空当」不是「已装进去的东西」，虚线棱以示区别
          dashed: seg.dashed,
          opacity: seg.opacity,
          title: `${seg.label} ${fmt(v)}`,
        }));
      }
      cursor += h;
    });

    /* ── 越界：摞在盒口之上，不涂红盒身。OOM 是结构性溢出，不是颜色变深。 ── */
    if (overflow > 0.012) {
      const dangerRgb = resolveColor(cssVar("--danger", "#E5484D", host), host);
      g.appendChild(solid(BOX.h + 0.30, Math.max(0.5, overflow), dangerRgb, {
        cls: "cro-capacity__solid",
        title: `溢出 ${fmt(loaded - cap)} → OOM`,
      }));
    }

    /* ── 容量：虚线线框（12 条棱）。画在实心之后，才有「装在笼子里」的读法。 ── */
    const wireStroke = cssVar("--border-strong", "rgba(255,255,255,0.24)", host);
    const W = BOX.w;
    const D = BOX.d;
    const H = BOX.h;
    [
      [[0, 0, 0], [W, 0, 0]], [[0, 0, 0], [0, 0, D]], [[W, 0, 0], [W, 0, D]], [[0, 0, D], [W, 0, D]],
      [[0, 0, 0], [0, H, 0]], [[W, 0, 0], [W, H, 0]], [[0, 0, D], [0, H, D]], [[W, 0, D], [W, H, D]],
      [[0, H, 0], [W, H, 0]], [[0, H, 0], [0, H, D]], [[W, H, 0], [W, H, D]], [[0, H, D], [W, H, D]],
    ].forEach((edge) => {
      const a = P(edge[0][0], edge[0][1], edge[0][2]);
      const b = P(edge[1][0], edge[1][1], edge[1][2]);
      g.appendChild(svgNode("line", {
        x1: a[0], y1: a[1], x2: b[0], y2: b[1],
        stroke: wireStroke, "stroke-width": 0.075, "stroke-dasharray": "0.26 0.2",
      }));
    });

    /* ── 阈值环：贴着盒壁的一圈虚线 + 左侧百分比标签 ── */
    (spec.thresholds || []).forEach((mark) => {
      const y = mark.at * H;
      /* 颜色先落地成具体值：这些是 SVG 呈现属性（不是 CSS 声明），把 `var(--danger)`
         原样写进 stroke/fill 各浏览器行为不一，解析不出来就整条线不显示。 */
      const rgb = resolveColor(mark.color, host);
      const color = rgb ? shade(rgb, 0) : mark.color;
      [
        [[0, y, 0], [W, y, 0]], [[0, y, 0], [0, y, D]],
        [[W, y, 0], [W, y, D]], [[0, y, D], [W, y, D]],
      ].forEach((edge) => {
        const a = P(edge[0][0], edge[0][1], edge[0][2]);
        const b = P(edge[1][0], edge[1][1], edge[1][2]);
        g.appendChild(svgNode("line", {
          x1: a[0], y1: a[1], x2: b[0], y2: b[1],
          stroke: color, "stroke-width": 0.062,
          "stroke-dasharray": "0.2 0.18", opacity: 0.8,
        }));
      });
      const anchor = P(0, y, D);
      const label = svgNode("text", {
        x: anchor[0] - 0.26, y: anchor[1] + 0.24,
        "text-anchor": "end", "font-size": 0.78,
        "font-family": "ui-monospace, Menlo, Consolas, monospace",
        "font-weight": 600, fill: color, opacity: 0.9,
      });
      label.textContent = `${Math.round(mark.at * 100)}%`;
      g.appendChild(label);
      /* 标签向左伸出的那截也要算进包围盒，否则会被 viewBox 裁掉。
         但它**不进 bounds** —— bounds 要保持"柱体自身"的范围，viewBox 才能围着
         柱体的中心对称展开（见下方）。混进去的话柱体就被推到右边了。 */
      if (anchor[0] - 2.0 < overhangX0) overhangX0 = anchor[0] - 2.0;
    });

    /* viewBox 围着**柱体的横向中心**对称展开，而不是紧贴包围盒。
       紧贴的话，左边多出来的那截阈值标签（2.0 user unit）会把柱体整体挤到右半边
       —— SVG 是 xMidYMid meet，居中的是 viewBox，viewBox 偏了柱体就跟着偏。
       取"柱体中心到左右两侧最远处"里更大的那个做半宽，两侧等宽，柱体就正好落在
       正中；短的那一侧多出来的空白无非是留白，本来也该有。 */
    const pad = 0.35;
    const cx = (bounds.x0 + bounds.x1) / 2;
    const half = Math.max(cx - Math.min(bounds.x0, overhangX0), bounds.x1 - cx) + pad;
    svg.setAttribute("viewBox", [
      (cx - half).toFixed(2),
      (bounds.y0 - pad).toFixed(2),
      (half * 2).toFixed(2),
      (bounds.y1 - bounds.y0 + pad * 2).toFixed(2),
    ].join(" "));
    return svg;
  }

  /* 导出给事件详情用。写在 boot 之外（模块解析时就挂上）：主脚本比本文件先加载，
     但它建图是在 DOMContentLoaded 之后的 rAF 里，那时这个导出早已就绪。
     一并给出阈值档位，两处的水位环才是同两道线。 */
  global.croCapacityBox = { build: buildBox, THRESHOLD };

  /* 本栏自己的那一幅：把「一张卡的显存」翻译成通用 spec。段色是 CSS 变量而不是
     解析好的分量 —— 主题切换后变量会变，交给 buildBox 每次现解析。 */
  function buildScene(m) {
    return buildBox({
      cap: m.cap,
      host: el.root,
      ariaLabel: "单卡容量等距示意",
      format: (v) => `${gb(v)} GB`,
      segments: SEGS.map((seg) => ({
        label: seg.label,
        value: m.values[seg.key],
        color: `var(--cro-cap-${seg.key}, #7C8A94)`,
        // 预留是「留给意外的空当」故虚线棱；底座是实打实占掉的，只压暗不虚线
        dashed: seg.key === "reserve",
        opacity: seg.key === "reserve" ? 0.62 : (seg.key === "base" ? 0.72 : null),
      })),
      thresholds: [
        { at: THRESHOLD.tight, color: cssVar("--warning", "#F5A524") },
        { at: THRESHOLD.alert, color: cssVar("--danger", "#E5484D") },
      ],
    });
  }

  /* ── 参数量：逐层分算，每项除以真正切它的那一维 ───────────────────────── */
  function paramsOfStage(topo, stage) {
    const { preset, counts, layers, stages } = topo;
    const seg = stages[stage];
    if (!seg || seg.count <= 0) return 0;

    const H = preset.hidden;
    const tp = Math.max(1, counts.tp);
    let sharded = 0;   // 沿 h / intermediate 被 TP 切的部分
    let intact = 0;    // router 这类不切的小项

    for (let l = seg.lo; l <= seg.hi; l += 1) {
      const layer = layers[l];
      if (!layer) continue;
      // Attention：q/k/v/o 四个 [H,H]。TP 切头，PP 已经体现在「只遍历本 stage 的层」。
      sharded += 4 * H * H;
      if (layer.ffn === "dense") {
        sharded += 3 * H * preset.denseIntermediate;   // SwiGLU：gate/up/down
      } else {
        // 路由专家：这张卡只持有 EP 分到的那几个（expertsPerEpRank = routed / EP）
        sharded += counts.expertsPerEpRank * 3 * H * preset.moeIntermediate;
        // 共享专家：在 EP 域内复制，每张卡都有一份完整的
        sharded += counts.sharedExpert * 3 * H * preset.moeIntermediate;
        intact += H * counts.routedExpert;             // router / gate，量级可忽略但别漏
      }
    }
    // Embedding 落在 stage0，LM Head 落在末 stage —— 首尾两段天然比中间重
    if (stage === 0) sharded += preset.vocab * H;
    if (stage === counts.pp - 1) sharded += preset.vocab * H;

    return sharded / tp + intact;
  }

  /* ── 激活：1F1B 下 stage s 同时压着 (PP−s) 份 micro-batch ──────────────── */
  function activationBytes(topo, stage) {
    const { preset, counts, stages } = topo;
    const seg = stages[stage];
    if (!seg || seg.count <= 0) return 0;
    const inflight = Math.max(1, counts.pp - stage);
    return BASIS.actPerLayer
      * BASIS.microBatch
      * (BASIS.seq / Math.max(1, counts.cp))
      * preset.hidden
      * seg.count
      / Math.max(1, counts.tp)
      * inflight;
  }

  /* ── 通信 buffer：∝ 通信域个数，与这张卡背了多少参数无关 ────────────────
     每个 >1 的并行维各建一个 HCCL 通信域；MoE 的 dispatch/combine 另起一条
     all-to-all 域。TP=1 时不建 TP 域，所以这一项会跟着并行配置一起动 —— 这正是
     它该画在「随配置动的预留」里、而不是揉进固定底座的原因。 */
  function commDomains(topo) {
    const c = topo.counts;
    const n = [c.tp, c.pp, c.dp, c.cp, c.ep].filter((v) => v > 1).length;
    return n + (c.ep > 1 ? 1 : 0);
  }

  function commBytes(topo) {
    return commDomains(topo) * RUNTIME.hcclBufGB * RUNTIME.hcclDouble * GIB;
  }

  /* ── 算子 workspace：峰值由**单个最大算子**决定，∝ 一层的 token 张量 ──────
     不是全部层的和 —— workspace 是算子跑完就还的临时区，同一时刻只有一个算子
     在占。MoE 的 permute/unpermute + GroupedMatMul 要 topK 份 token 的临时区，
     通常就是这个峰值；本 stage 全是 dense 层时由 FA 之类兜底，取下限。 */
  function workspaceBytes(topo, stage) {
    const { preset, counts, stages, layers } = topo;
    const seg = stages[stage];
    let peak = RUNTIME.wsFloorGB * GIB;
    if (!seg || seg.count <= 0) return peak;
    let hasMoe = false;
    for (let l = seg.lo; l <= seg.hi; l += 1) {
      if (layers[l] && layers[l].ffn !== "dense") { hasMoe = true; break; }
    }
    if (!hasMoe) return peak;
    const tokens = BASIS.microBatch * (BASIS.seq / Math.max(1, counts.cp));
    const tokenBytes = tokens * preset.hidden * BASIS.bytesWeight / Math.max(1, counts.tp);
    return Math.max(peak, RUNTIME.wsFactor * counts.topK * tokenBytes);
  }

  function measure(topo, stage) {
    const params = paramsOfStage(topo, stage);
    const weight = params * BASIS.bytesWeight;
    const grad = params * BASIS.bytesGrad;
    const optim = params * BASIS.bytesOptim;
    const act = activationBytes(topo, stage);
    const core = weight + grad + optim + act;
    /* 底座摞在盒底而不是从 cap 里扣掉：两种算法对占比完全等价（都是
       (base+core+reserve)/hbm），但画成盒底那一段能让人看见它有多大 —— 「64 GB
       的卡为什么一开机就少 2 GB」是这一栏最常被问的一句。 */
    const base = RUNTIME.baseGB * GIB;
    const comm = commBytes(topo);
    const workspace = workspaceBytes(topo, stage);
    const frag = core * RUNTIME.fragRatio;
    const reserve = comm + workspace + frag;
    const cap = BASIS.capGB * GIB;
    const total = base + core + reserve;
    return {
      stage, params, cap, total,
      ratio: total / cap,
      inflight: Math.max(1, topo.counts.pp - stage),
      layers: topo.stages[stage] ? topo.stages[stage].count : 0,
      values: { base, weight, grad, optim, act, reserve },
      // 预留段在图上是一整段，拆项只在图例 tooltip 与口径浮层里给
      reserveParts: { comm, workspace, frag },
    };
  }

  function levelOf(ratio) {
    if (ratio > 1) return "over";
    if (ratio > THRESHOLD.alert) return "alert";
    if (ratio > THRESHOLD.tight) return "tight";
    return "safe";
  }

  const LEVEL_LABEL = { safe: "安全", tight: "偏满", alert: "预警", over: "越界" };

  const gb = (bytes) => (bytes / GIB).toFixed(1);

  /* ── 判定文案：越紧越要给出下一步动作，且必须点破 DP 不是那个动作 ──────── */
  function verdictText(m, topo) {
    const pct = Math.round(m.ratio * 100);
    const room = m.cap - m.total;
    if (m.ratio > 1) {
      return `占用 ${pct}%，溢出 ${gb(-room)} GB —— 该配置预计 OOM。`
        + `其中底座与预留 ${gb(m.values.base + m.values.reserve)} GB 压不掉，`
        + `减容器要动 EP / PP / TP，加 DP 只增吞吐不增余量。`;
    }
    if (m.ratio > THRESHOLD.alert) {
      return `占用 ${pct}%，余量仅 ${gb(room)} GB —— 已越过 ${Math.round(THRESHOLD.alert * 100)}% 预警线。`
        + `建议提高 EP（切路由专家）或 PP（少背几层）。`;
    }
    if (m.ratio > THRESHOLD.tight) {
      return `占用 ${pct}%，余量 ${gb(room)} GB —— 已越过 ${Math.round(THRESHOLD.tight * 100)}% 偏满线，`
        + `通信 buffer 与碎片的正常波动即可能触发 OOM。`;
    }
    return `占用 ${pct}%，余量 ${gb(room)} GB。`
      + `本 stage 背 ${m.layers} 层、在飞 ${m.inflight} 份 micro-batch。`;
  }

  /* 口径浮层内容：谁除以谁写清楚，尤其「DP 不除任何东西」那一行 —— 它是这一栏
     最反直觉、也最该被人当场核对的一条。
     第二段专门回答「为什么各卡不一样」：这是看到 stage 小柱高低不齐时的第一个
     疑问，不解释就会被当成算错了。 */
  function basisHtml(topo) {
    const c = topo.counts;
    const card = topo.card;
    const row = (k, v) => `<dt>${k}</dt><dd>${v}</dd>`;
    const stageSizes = topo.stages.map((s) => s.count).join(" / ");
    return `<dl>`
      + row("权重 / 梯度", `bf16，各 ${BASIS.bytesWeight} B/参数`)
      + row("优化器状态", `Adam ${BASIS.bytesOptim} B/参数（fp32 master + momentum + variance）`)
      + row("激活", `<code>34·mb·(S/CP)·H·层数/TP</code> × 在飞份数，开 SP、不重计算`)
      + row("micro-batch", `<b>mb=${BASIS.microBatch}</b>，激活与它成正比 —— 这是 batch 里唯一进显存的一半`)
      + row("global batch", `<b>不进显存</b>。GBS 只决定梯度累积步数 <code>GBS/(MBS×DP)</code>，一步也不占容量 —— 与「DP 不减容器」是同一件事`)
      + row("序列长度", `S=${BASIS.seq}，激活与它成正比，CP(${c.cp}) 会把它切开`)
      + row("路由专家", `÷ EP(${c.ep}) × TP(${c.tp})，是 MoE 下减容器的主力`)
      + row("共享专家", `在 EP 域内复制，每张卡各持一份`)
      + row("其余权重", `÷ TP(${c.tp})；全部 ÷ PP(${c.pp})，体现为这张卡只背本 stage 那几层`)
      + row("DP", `<b>不除任何东西</b>。DP(${c.dp}) 买的是吞吐不是余量，除非上 ZeRO / FSDP`)
      + row("运行时底座", `<b>${RUNTIME.baseGB} GB 固定</b>：驱动 + CANN/ACL context + kernel binary + 通信域元数据，与配置无关 —— 64 GB 的卡一开机就少这么多`)
      + row("通信 buffer", `<code>${RUNTIME.hcclBufGB} GB × 域数(${commDomains(topo)}) × ${RUNTIME.hcclDouble}（双缓冲）</code>；域数 = TP/PP/DP/CP/EP 中 >1 的维度${c.ep > 1 ? " + MoE 的 a2a 域" : ""}`)
      + row("算子 workspace", `峰值由单个最大算子定，∝ <b>一层</b>的 token 张量：<code>${RUNTIME.wsFactor}·topK·mb·(S/CP)·H·2B/TP</code>（MoE permute + GroupedMatMul），下限 ${RUNTIME.wsFloorGB} GB`)
      + row("内存碎片", `已用量的 ${Math.round(RUNTIME.fragRatio * 100)}% —— 四项里只有这一项真按比例；MoE 每 step token 数变长时更差`)
      + row("单卡显存", `${BASIS.capGB} GB，警戒线 ${Math.round(THRESHOLD.tight * 100)}% 偏满 / ${Math.round(THRESHOLD.alert * 100)}% 预警`)
      + (card ? row("卡型号", `<b>${card.label}</b> · ${card.hbmGB} GB HBM<br><span class="cro-capacity__basis-warn">${card.hbmNote}</span>`) : "")
      + (card && card.specs ? row("规格", card.specs) : "")
      + `</dl>`
      + `<p class="cro-capacity__basis-sub">为什么各卡装的不一样多</p>`
      + `<dl>`
      + row("层数不均", `${c.totalLayer} 层分给 PP(${c.pp}) 段，除不尽时前几段各多 1 层 —— 本配置是 ${stageSizes} 层`)
      + row("首尾更重", `Embedding 只在 Stage0、LM Head 只在 Stage${c.pp - 1}，各约 ${gb(topo.preset.vocab * topo.preset.hidden * (BASIS.bytesWeight + BASIS.bytesGrad + BASIS.bytesOptim) / Math.max(1, c.tp))} GB（含梯度与优化器状态）`)
      + row("在飞份数", `1F1B 下 Stage s 同时压着 <code>PP−s</code> 份 micro-batch 的激活，Stage0 压 ${c.pp} 份、末段只压 1 份`)
      + row("同 stage 内", `各 DP / EP / TP / CP 副本切法一致、容量相同，所以差异只到 stage 这一级 —— 底部那排小柱就是全集群的完整分布`)
      + `</dl>`
      + `<p class="cro-capacity__basis-sub">运行时四项怎么标定</p>`
      + `<p class="cro-capacity__basis-note">上面四个系数是<b>经验值，待实测标定</b>：固定并行度、只改 micro-batch 跑两三次，`
      + `取 <code>实测显存峰值 − 理论四段</code>，两点拟合 —— 截距 = 底座 + 通信 buffer（不随 mb 动），`
      + `斜率 = workspace + 碎片（∝ mb）；再换一组 EP/TP 复跑一次，即可把通信域那一项从截距里分离出来。</p>`
      + `<p class="cro-capacity__basis-foot">`
      + `MTP 层、重计算、EP 与 DP 是否正交的口径差异均未计入。量级估算，用于看趋势与相对高低。</p>`;
  }

  /* ── 渲染 ───────────────────────────────────────────────────────────────── */
  function setEmpty(message) {
    if (!el.root) return;
    el.body.hidden = true;
    el.empty.hidden = false;
    el.empty.textContent = message;
    el.scope.textContent = "";
  }

  function render() {
    if (!el.root) return;
    if (!topology) return;
    syncBasis(topology);
    if (!topology.valid) {
      setEmpty("配置不自洽，容量暂不估算（见上方提示）");
      return;
    }
    const pp = topology.counts.pp;
    const all = [];
    for (let s = 0; s < pp; s += 1) all.push(measure(topology, s));
    if (!all.length) { setEmpty("无可估算的 stage"); return; }

    // 默认看**装得最满的那张卡** —— 排容量只有它说了算，OOM 只需要一张卡爆。
    const fullest = all.reduce((a, b) => (b.ratio > a.ratio ? b : a), all[0]);
    const stage = (pinnedStage != null && all[pinnedStage]) ? pinnedStage : fullest.stage;
    const m = all[stage];
    const level = levelOf(m.ratio);

    el.empty.hidden = true;
    el.body.hidden = false;

    /* 默认态要指名道姓：不是「某个 stage」而是**具体哪一张卡**。同 stage 内各
       DP/EP/TP/CP 副本容量相同，取该 stage 的首卡作代表，编号与 stage 一起标出，
       用户拿这个号能直接回集群矩阵里找。 */
    const shownRank = pinnedRank != null ? pinnedRank : topology.rankOf(stage, 0, 0, 0);
    el.scope.textContent = pinnedRank != null
      ? `rank ${shownRank} · Stage${stage}`
      : (pinnedStage != null
        ? `rank ${shownRank} · Stage${stage}`
        : `rank ${shownRank} · Stage${stage} · 最满`);
    el.scope.title = pinnedRank != null
      ? `当前显示选中的 rank ${shownRank}（Stage${stage}）`
      : `未选中具体卡时，显示全集群里装得最满的那一张：rank ${shownRank}`
        + `（Stage${fullest.stage} 的首卡，同 stage 内其余副本容量相同）`;

    /* 等距容器：段色取自 deck 语义色变量（主题切换后会变，所以每次重解） */
    el.scene.innerHTML = "";
    el.scene.appendChild(buildScene(m));

    /* 图例读数。SEGS 是**从底往上**的堆叠顺序，图例是从上往下读的，所以倒过来
       —— 两边顺序不一致时，眼睛要在柱子和文字之间做一次映射才对得上，白费一次
       认知。倒序之后第一行「预留」对的就是柱顶那一段。 */
    el.legend.innerHTML = "";
    SEGS.slice().reverse().forEach((seg) => {
      const li = doc.createElement("li");
      li.className = "cro-capacity__row";
      li.dataset.seg = seg.key;
      li.innerHTML = `<i></i><span>${seg.label}</span><b>${gb(m.values[seg.key])} GB</b>`;
      /* 预留是三项合成的一段，拆项只在这里给：常驻三行会把一栏窄面板压掉半屏，
         而"哪一项占大头"是想清楚了才会问的第二层问题。 */
      if (seg.key === "reserve") {
        li.title = `通信 buffer ${gb(m.reserveParts.comm)} GB（${commDomains(topology)} 个 HCCL 域）`
          + ` + 算子 workspace ${gb(m.reserveParts.workspace)} GB`
          + ` + 碎片 ${gb(m.reserveParts.frag)} GB`;
      } else if (seg.key === "base") {
        li.title = `驱动 + CANN/ACL context + kernel binary，固定 ${RUNTIME.baseGB} GB，不随配置变`;
      }
      el.legend.appendChild(li);
    });
    /* 头条读数：占比放大并按档位换色（安全档不出判定横幅后，它是唯一一眼能看出
       险不险的东西），绝对值压小压暗跟在后面。 */
    el.ratio.dataset.level = level;
    el.pct.textContent = `${Math.round(m.ratio * 100)}%`;
    el.abs.textContent = `${gb(m.total)} / ${BASIS.capGB} GB`;

    /* 判定：安全档不出横幅。这一栏在 Cluster 行里，行高由内容撑；一条「安全」
       横幅每次都白占 ~36px 高度，而它什么也没告诉你 —— 没有警报就是好消息。
       越过警戒线才出现，出现即意味着要动手。 */
    const showVerdict = level !== "safe";
    el.verdict.hidden = !showVerdict;
    if (showVerdict) {
      el.verdict.dataset.level = level;
      el.verdictBadge.textContent = LEVEL_LABEL[level];
      el.verdictText.textContent = verdictText(m, topology);
    }

    /* 各 stage 峰值：容量只随 stage 变，所以这排小柱就是完整分布。
       但判定横幅一出现（level !== "safe"，含"偏满" tight）就已经把「险不险」
       说清楚了，这排小柱这时只是重复信息，却仍占着一整块竖向空间 —— 让位给
       横幅，隐藏整个分区。与 showVerdict 用同一个条件，横幅在就该它不在。 */
    const hideSpread = showVerdict;
    el.spread.hidden = hideSpread;
    el.stages.innerHTML = "";
    if (!hideSpread) {
      const peak = Math.max(1, fullest.ratio);
      all.forEach((entry) => {
        const bar = doc.createElement("button");
        bar.type = "button";
        bar.className = "cro-capacity__stage";
        bar.dataset.stage = String(entry.stage);
        bar.dataset.level = levelOf(entry.ratio);
        bar.classList.toggle("is-selected", entry.stage === stage);
        bar.style.height = `${Math.max(6, entry.ratio / peak * 100).toFixed(1)}%`;
        bar.title = `Stage${entry.stage} · ${gb(entry.total)} GB / ${BASIS.capGB} GB`
          + `（${Math.round(entry.ratio * 100)}%）· ${entry.layers} 层 · 在飞 ${entry.inflight}`;
        bar.setAttribute("aria-label", bar.title);
        bar.addEventListener("click", () => selectStage(entry.stage));
        el.stages.appendChild(bar);
      });
      el.spreadNote.textContent = `最满 Stage${fullest.stage} · ${Math.round(fullest.ratio * 100)}%`;
    }

    // 口径里带着当前的 EP/TP/PP/DP 取值，配置一改就得重写；浮层没打开时写了也不亏
    el.basis.innerHTML = basisHtml(topology);
  }

  /* 点 stage 小柱 = 选中该 stage 的首卡。payload 形状与集群矩阵格子完全一致
     （见 renderCluster 里 cell 的 click），走同一条关系解析，四域一起亮。 */
  function selectStage(stage) {
    if (!topology || !topology.valid) return;
    const rank = topology.rankOf(stage, 0, 0, 0);
    if (typeof global.croSelect !== "function") {
      pinnedStage = stage;
      pinnedRank = null;
      render();
      return;
    }
    global.croSelect({
      kind: "rank", rank, stage, dpIdx: 0, epRank: 0,
      tpIdx: 0, cpIdx: 0, node: topology.nodeOfRank(rank),
    });
  }

  /* ── 挂接 ───────────────────────────────────────────────────────────────── */
  function cache() {
    el.root = doc.getElementById("croCapacity");
    if (!el.root) return false;
    el.body = doc.getElementById("croCapacityBody");
    el.empty = doc.getElementById("croCapacityEmpty");
    el.scope = doc.getElementById("croCapacityScope");
    el.help = doc.getElementById("croCapacityHelp");
    el.scene = doc.getElementById("croCapacityScene");
    el.ratio = doc.getElementById("croCapacityRatio");
    el.pct = doc.getElementById("croCapacityPct");
    el.abs = doc.getElementById("croCapacityAbs");
    el.legend = doc.getElementById("croCapacityLegend");
    el.verdict = doc.getElementById("croCapacityVerdict");
    el.verdictBadge = doc.getElementById("croCapacityBadge");
    el.verdictText = doc.getElementById("croCapacityVerdictText");
    el.stages = doc.getElementById("croCapacityStages");
    el.spread = doc.getElementById("croCapacitySpread");
    el.spreadNote = doc.getElementById("croCapacitySpreadNote");
    el.basis = doc.getElementById("croCapacityBasis");
    return true;
  }

  /* 口径浮层：悬浮即出，移开即收。收起有 140ms 延迟 —— 问号与浮层之间隔着几像素
     的空当，鼠标滑过去的那一瞬会先离开问号再进入浮层，没有延迟就会在半路收掉，
     内容根本选不中、也读不完。浮层自身 hover 时同样保持展开。 */
  let basisTimer = 0;

  /* 贴放并避让视口。默认挂在问号正下方左对齐；下方装不下就翻到问号上方；
     左右两侧再各夹 8px。不这么算的话，这一栏在板面最底行、最右列，浮层往下往右
     都会探出屏幕 —— 探出去的那一截就是页面凭空多出来的滚动条。 */
  function placeBasis() {
    if (!el.help || !el.basis || el.basis.hidden) return;
    const anchor = el.help.getBoundingClientRect();
    const box = el.basis.getBoundingClientRect();
    const gap = 6;
    const margin = 8;
    const vw = global.innerWidth;
    const vh = global.innerHeight;

    let left = Math.min(anchor.left, vw - box.width - margin);
    left = Math.max(margin, left);

    let top = anchor.bottom + gap;
    if (top + box.height > vh - margin) {
      const above = anchor.top - gap - box.height;
      // 上方也放不下就退回下方并顶到底边，总之不让它探出视口
      top = above >= margin ? above : Math.max(margin, vh - box.height - margin);
    }

    el.basis.style.left = `${Math.round(left)}px`;
    el.basis.style.top = `${Math.round(top)}px`;
  }

  function showBasis(open) {
    if (!el.help || !el.basis) return;
    global.clearTimeout(basisTimer);
    if (open) {
      el.basis.hidden = false;
      // 先出现再量：hidden 时 getBoundingClientRect 全是 0，算不出该翻到哪边
      placeBasis();
      el.help.setAttribute("aria-expanded", "true");
    } else {
      basisTimer = global.setTimeout(() => {
        el.basis.hidden = true;
        el.help.setAttribute("aria-expanded", "false");
      }, 140);
    }
  }

  function boot() {
    if (!cache()) return;

    /* 把浮层挪出面板挂到 body：祖先 .pto-ide-frame__pane 既是 overflow:hidden
       又带 backdrop-filter（后者会成为 fixed 的包含块），留在里面既定位不到视口
       也会被裁。挪出去之后它就不属于任何滚动容器，撑不出滚动条。 */
    if (el.basis && doc.body && el.basis.parentNode !== doc.body) doc.body.appendChild(el.basis);

    if (el.help) {
      // pointerenter/leave 而不是 mouseover/out：后者在子元素间冒泡，会反复开合
      el.help.addEventListener("pointerenter", () => showBasis(true));
      el.help.addEventListener("pointerleave", () => showBasis(false));
      // 键盘可达：Tab 到问号同样展开（hover-only 的提示对键盘用户等于不存在）
      el.help.addEventListener("focus", () => showBasis(true));
      el.help.addEventListener("blur", () => showBasis(false));
      // 触屏没有 hover，留一个点击开合
      el.help.addEventListener("click", (event) => {
        event.stopPropagation();
        showBasis(el.basis.hidden);
      });
    }
    if (el.basis) {
      el.basis.addEventListener("pointerenter", () => showBasis(true));
      el.basis.addEventListener("pointerleave", () => showBasis(false));
    }
    doc.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && el.basis && !el.basis.hidden) {
        global.clearTimeout(basisTimer);
        el.basis.hidden = true;
        el.help.setAttribute("aria-expanded", "false");
      }
    });
    /* 浮层脱离了文档流，锚点（问号）却还在会滚动的板面里 —— 面板一滚、窗口一改，
       两者就对不上了，得跟着重贴。capture 是为了收到 .cro-board 这类内部滚动。 */
    global.addEventListener("resize", placeBasis);
    doc.addEventListener("scroll", placeBasis, { passive: true, capture: true });

    // 主题一换，deck 语义色变量与 --border-strong / --warning / --danger 全变了，
    // 而这些颜色是渲染时读进 SVG 的死值，必须重画一遍
    doc.addEventListener("cro:theme", () => render());

    doc.addEventListener("cro:change", (event) => {
      topology = event.detail || topology;
      // 层数/PP 变了之后旧的钉选可能已经不存在
      if (topology && pinnedStage != null && pinnedStage >= topology.counts.pp) {
        pinnedStage = null;
        pinnedRank = null;
      }
      render();
    });

    /* 选中态：卡 → 直接切；层 → 切到它所在 stage（层与卡本来就是同一条查询口径）。
       其余对象（专家/算子/EP 组）跨多个 stage，没有唯一答案，回落到最紧的卡。 */
    doc.addEventListener("cro:select", (event) => {
      const p = event.detail && event.detail.primary;
      if (p && p.kind === "rank" && Number.isFinite(p.stage)) {
        pinnedStage = p.stage;
        pinnedRank = Number.isFinite(p.rank) ? p.rank : null;
      } else if (p && p.kind === "layer" && topology && Number.isFinite(p.layer)) {
        pinnedStage = topology.stageOfLayer(p.layer);
        pinnedRank = null;
      } else {
        pinnedStage = null;
        pinnedRank = null;
      }
      render();
    });

    // 首帧：cro:change 是在主脚本初始化时发的，可能早于本文件挂上监听，
    // 所以直接从已导出的 controller 上取一次当前拓扑。
    topology = (global.croObserver && global.croObserver.topology) || null;
    if (topology) render();
    else requestAnimationFrame(() => {
      topology = (global.croObserver && global.croObserver.topology) || null;
      if (topology) render();
    });
  }

  if (doc.readyState === "loading") doc.addEventListener("DOMContentLoaded", boot);
  else boot();
})(window);
