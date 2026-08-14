/* ══════════════════════════════════════════════════════════════════════════════
   training-monitoring-v2.html 整网图适配器
   ──────────────────────────────────────────────────────────────────────────────
   v2 新版把旧 v2 的 SVG 整网图(opv-modelviz)换成了 pto-design-system 的
   model-architecture-3d-deck pattern —— CSS 3D 层叠 + DOM 节点,46 层沿深度铺开。
   两者的 DOM 结构、节点 id、坐标系都不一样,但 v2 在整网图上加出来的那几样东西
   (常驻问题标注、点问题后的聚焦、routed_expert_bank 原地展开 + 专家负载热力动画、
   算子去色、问题二的溢出率徽标)都要原样保留。

   做法是把「画在图上」这件事抽成一层适配器:training-run-twin.js 里那些函数一旦发现
   window.PtoTwinGraphAdapter 存在,就把绘制让给本文件,自己只保留业务语义(哪个问题命中
   哪些节点、时光机跳到哪一步、抽屉里放什么)。v2 / training-monitoring.html 不注册适配器,
   继续走原来的 SVG 实现,一行行为都不变。

   坐标系:deck 的每一层是一个 .pto-model-deck__layer > .pto-model-deck__graph,
   层内节点用 left/top 绝对定位(px)。标注/展开卡片直接以同样的方式插进 __graph 里,
   于是自动继承该层的 3D 变换,跟着旋转/缩放/平移走,不需要自己做任何投影计算。
   ══════════════════════════════════════════════════════════════════════════════ */
(function initTrainingMonitoringV3Deck(global) {
  'use strict';

  var HOST_ID = 'deckStage';

  // 本页只用正交投影(正视/侧视),不出 pattern 的等轴 3D(iso)视角。
  // 影响两处 pattern 行为,都是我们想要的:
  //   · iso 专属的「非代表层压暗成 foreground-muted」规则不再生效,每层都保持语义色;
  //   · 拖拽在非 iso 下是平移而不是旋转,与正交投影的读图方式一致。
  //
  // 动线是「侧视总览 → 正视下钻」:
  //   · 侧视(OVERVIEW_VIEW)下 46 层全部可见,5 枚问题标注同时在场,承接原 3D 视角
  //     「一眼看全哪几层有问题」的作用,所以拿它当落地视图;
  //   · 正视(FOCUS_VIEW)下只有 is-front-layer 那一层不透明且可交互(见 pattern.css
  //     [data-view="front"] 规则),天然就是聚焦效果,所以点问题时切到它。
  var OVERVIEW_VIEW = 'right';
  var FOCUS_VIEW = 'front';

  // ── v2(opv-modelviz schema)节点 id → deck 节点 id ────────────────────────────
  // deck 用的是 openPangu 架构参考里的算子命名,和 opv schema 只是叫法不同;
  // 未列出的 id 原样透传(q_b_proj / o_proj / attention_core 等两边同名)。
  var NODE_MAP = {
    router_gate: 'gate',
    router_weight: 'gate',                 // deck 未单列 router 权重节点,并到 Router 本体
    routed_expert_bank: 'expert_pool',
    shared_expert_mlp: 'shared_expert',
    moe_all_to_all_dispatch: 'a2a_dispatch',
    moe_all_to_all_combine: 'a2a_combine',
    attention_all_gather: 'attn_all_gather',
    attention_reduce_scatter: 'attn_reduce_scatter',
    token_embedding: 'embedding',
  };

  // ── 各问题落在哪一层 ────────────────────────────────────────────────────────
  // 用户选择「只挂问题实际发生的层」:层号取自 diagnosisCases[].layer 的文案。
  // 'input' / 'output' 表示挂在 deck 首尾两块静态区(Embedding / LM Head+MTP)。
  // nvlink 是跨整个 decoder stack 的链路问题,没有单一事故层,挂到 deck 的默认前置层 23
  // (PP2 的首层,本身就是全彩代表层)当作"任意一层的通信算子都受影响"的代表。
  // mem-oom(问题二 · 显存 OOM)的热点层就是 38 —— MoE 的 expert dispatch buffer 让该层激活值
  // 达 1.2 GB,是普通 dense 层的 1.7×;它与 moe-a2a 同层,徽标由下面的重叠避让逻辑纵向堆叠。
  var CASE_LAYER = {
    'moe-a2a': 38,
    'mem-oom': 38,
    'qproj-overflow': 33,
    'low-precision-training': 35,
    nvlink: 23,
    'perf-compute-bottleneck': 'output',
  };
  var ROUTED_EXPAND_LAYER = CASE_LAYER['moe-a2a'];

  var EXPAND_W = 560, EXPAND_H = 210;      // 与 training-run-twin.js 的同名常量一致

  var root = null;             // .pto-model-deck 根(= #deckStage)
  var controller = null;       // pattern 的 controller
  var focusedCase = null;
  var pendingMarkers = null;   // 首次 renderMarkers 的入参,resize/重画时复用
  // 专家热力卡片挂在哪一层:由节点点击写入(点哪层展哪层);null = 用事故层。
  // 进问题定位链时由 focus() 复位成 null,保证问题一永远展在 L38。
  var expandLayer = null;

  function qs(sel, ctx) { return (ctx || document).querySelector(sel); }
  function qsa(sel, ctx) { return Array.prototype.slice.call((ctx || document).querySelectorAll(sel)); }
  function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }

  // ── 层容器定位 ──────────────────────────────────────────────────────────────
  // layer 为数字 → 第 n 个 decoder 层;'input'/'output' → 首尾静态区。
  function layerGraph(layer) {
    if (!root) return null;
    if (layer === 'input' || layer === 'output') {
      return qs('.pto-model-deck__static--' + layer, root);
    }
    var card = qs('.pto-model-deck__layer[data-layer="' + layer + '"]', root);
    return card ? qs('.pto-model-deck__graph', card) : null;
  }

  // 在指定层里找节点;找不到就在整个 deck 里兜底找第一个同 id 节点
  // (跨区节点如 token_embedding 挂在 input 区,但案例的锚点层可能是 output)。
  function findNode(deckId, layer) {
    if (!root) return null;
    var scope = layerGraph(layer);
    var sel = '[data-node="' + (global.CSS && CSS.escape ? CSS.escape(deckId) : deckId) + '"]';
    return (scope && qs(sel, scope)) || qs(sel, root);
  }

  function mapId(v2Id) { return NODE_MAP[v2Id] || v2Id; }

  // cluster 在 pattern 里没有 id,只能按标题文案区分:注意力块 vs FFN 块。
  function findCluster(v2ClusterId, layer) {
    var scope = layerGraph(layer);
    if (!scope) return null;
    var wanted = v2ClusterId === 'attention-block' ? 'MLA' : v2ClusterId === 'moe-block' ? 'FFN' : null;
    if (!wanted) return null;   // decoder-stack 是整摞层,没有对应的层内分组框
    return qsa('.pto-model-deck__cluster', scope).filter(function (c) {
      return (c.textContent || '').indexOf(wanted) >= 0;
    })[0] || null;
  }

  /* ════════════════════════════════════════════════════════════════════════════
     1. 常驻问题标注
     每个案例在它的事故层上画一枚两行标签条(上排「问题N」+ 下排问题标题),
     贴在锚点节点正上方,与节点同宽;同层多枚重叠时向上堆叠避让 —— 与 v2 的
     drawBadge 口径一致,只是从 SVG <g> 换成了绝对定位的 div。
     ════════════════════════════════════════════════════════════════════════════ */
  function clearMarkers() {
    qsa('.v3-problem-badge', root).forEach(function (el) { el.remove(); });
    qsa('.is-diagnosis-active', root).forEach(function (el) { el.classList.remove('is-diagnosis-active'); });
  }

  function renderMarkers(cases, markers, severityColor) {
    if (!root) return;
    clearMarkers();

    // 常态标红:命中节点/分组框加描边,不分层级 —— 侧视总览下这是"哪几层有问题"的第一眼信号
    Object.keys(cases).forEach(function (key) {
      var info = cases[key];
      var layer = CASE_LAYER[key];
      (info.nodeIds || []).forEach(function (id) {
        var el = findNode(mapId(id), layer);
        if (el) el.classList.add('is-diagnosis-active');
      });
      (info.clusterIds || []).forEach(function (cid) {
        var cluster = findCluster(cid, layer);
        if (cluster) cluster.classList.add('is-diagnosis-active');
      });
    });

    // 问题一的 MoE 分组框:点框本体 = 进入 MoE 层展开图(与 v2 一致)
    var moeCluster = findCluster('moe-block', ROUTED_EXPAND_LAYER);
    if (moeCluster && !moeCluster.dataset.lvBound) {
      moeCluster.dataset.lvBound = '1';
      moeCluster.style.cursor = 'pointer';
      moeCluster.addEventListener('click', function (e) {
        e.stopPropagation();
        global.PtoTwinGraphBridge && global.PtoTwinGraphBridge.enterProblemOneLayerView();
      });
    }

    // 点一下就地展开 256 专家负载热力,再点收起。不进问题详情、不动时光机 ——
    // 训练过程中随时可以打开看当前这一步 token 是怎么分到 256 个专家上的,
    // 步数推进/时光机拖动时热力跟着重画(twin.js 的 syncExpertHeatToStep)。
    //
    // 两个节点都绑:'gate'(Router · Top-8,做出路由决策的算子)是语义入口,
    // 'expert_pool'(Expert Pool,展开卡片就长在它身上)是几何入口——从"点哪个能看到专家"
    // 的直觉出发,用户更可能点后者。pattern 自己的 syncExpertExpansion() 只会把 pool 收起、
    // 没有展开行为,原先点它除了 selectNode 高亮什么都不发生。
    //
    // 逐层绑、点哪层在哪层展开:正视下 pattern 把非前置层整个 display:none(见 pattern.css
    // 「正视只显示当前一层」),而默认前置层是 23、不是事故层 38 —— 只绑事故层的话,用户在
    // 别的层上看到的 Expert Pool 根本不是被绑的那个元素,点了没有任何反应。
    // (dense 层没有这两个节点,自然跳过;侧视总览下全部层 pointer-events:none,点不动是 pattern 的既有行为。)
    qsa('.pto-model-deck__layer[data-layer]', root).forEach(function (card) {
      var layer = Number(card.dataset.layer);
      ['gate', 'expert_pool'].forEach(function (id) {
        var el = qs('[data-node="' + id + '"]', card);
        if (!el || el.dataset.routerExpandBound) return;
        el.dataset.routerExpandBound = '1';
        el.style.cursor = 'pointer';
        el.title = '展开 / 收起 256 专家负载热力';
        // pattern 的 pointerDown 只对 <button>/[data-stage-ui] 让路,其余一律给 viewport
        // 设 pointer capture 用于拖拽平移。捕获一旦生效,浏览器就把随后的 click 派发到捕获
        // 元素(viewport)而不是真正被点的元素 —— expert_pool 是个 <div role="button">,
        // 不在让路名单里,于是它身上的 click 永远收不到(pattern 自己的 selectNode 同样收不到)。
        // 这里在目标阶段截住 pointerdown、不让它冒泡到 viewport,capture 就不会建立,click 恢复正常。
        // 代价是不能从这个节点上起手拖拽平移 —— 与 pattern 里 <button> 类节点的既有行为一致。
        el.addEventListener('pointerdown', function (e) { e.stopPropagation(); });
        el.addEventListener('click', function (e) {
          e.stopPropagation();
          expandLayer = layer;
          global.PtoTwinGraphBridge && global.PtoTwinGraphBridge.toggleExpertExpand();
        });
      });
    });

    // 标签条:每层各自维护一份"已占位矩形"用于避让
    var placed = {};
    Object.keys(cases).forEach(function (key) {
      var info = cases[key];
      var marker = (markers || []).filter(function (m) { return m.key === key; })[0];
      if (!marker || !info.nodeIds || !info.nodeIds.length) return;
      var layer = CASE_LAYER[key];
      var anchor = findNode(mapId(info.nodeIds[0]), layer);
      if (!anchor) return;
      var graph = anchor.parentElement;
      if (!graph) return;

      var GAP = 6, H = 40;
      var w = Math.max(anchor.offsetWidth, 150);
      var left = anchor.offsetLeft;
      var top = anchor.offsetTop - H - GAP;

      var key2 = String(layer);
      placed[key2] = placed[key2] || [];
      placed[key2].forEach(function (box) {
        var overX = left < box.left + box.w && left + w > box.left;
        var overY = top < box.top + box.h && top + H > box.top;
        if (overX && overY) top = box.top - H - GAP;
      });
      placed[key2].push({ left: left, top: top, w: w, h: H });

      var badge = document.createElement('div');
      badge.className = 'v3-problem-badge';
      badge.dataset.diagnosisKey = key;
      badge.style.cssText = 'left:' + left + 'px;top:' + top + 'px;width:' + w + 'px;height:' + H + 'px;' +
        '--v3-badge-color:' + (severityColor ? severityColor(key) : '#dc2626');
      // displayNum 由 training-run-twin.js 的 visibleDiagnosisMarkers() 按当前页面算好(v2 有自己
      // 一套序号,见那里的编号说明);老数据没有该字段时退回 num。
      badge.innerHTML = '<b>问题' + (marker.displayNum || marker.num) + '</b><span>' + String(marker.label || '').replace(/[&<>]/g, '') + '</span>';
      badge.title = (marker.label || '') + '\n' + (info.note || '');
      badge.addEventListener('click', function (e) {
        e.stopPropagation();
        global.PtoTwinGraphBridge && global.PtoTwinGraphBridge.activateProblemLens(key);
      });
      graph.appendChild(badge);
    });

    pendingMarkers = { cases: cases, markers: markers, severityColor: severityColor };
  }

  function highlightBadge(caseKey) {
    if (!root) return;
    qsa('.v3-problem-badge', root).forEach(function (b) {
      var match = b.dataset.diagnosisKey === caseKey;
      b.classList.toggle('is-dimmed', !!caseKey && !match);
      b.classList.toggle('is-active', !!caseKey && match);
    });
  }

  /* ════════════════════════════════════════════════════════════════════════════
     2. 点问题后的聚焦
     v2 是"命中节点保持不透明 + 画布平移缩放到命中区域"。deck 上等价、且读起来更清楚的
     做法是:切到正视(front)并把事故层提到最前(setFrontLayer)—— 正视下只有 is-front-layer
     那一层是完全不透明且可交互的,其余层自动退到背景,天然就是 v2 想要的"聚焦"效果;
     命中节点再单独加一层高亮,同层未命中的节点压暗。
     ════════════════════════════════════════════════════════════════════════════ */
  function clearFocus() {
    focusedCase = null;
    if (!root) return;
    root.classList.remove('is-diagnosis-focus');
    qsa('.is-diagnosis-focus-active', root).forEach(function (el) { el.classList.remove('is-diagnosis-focus-active'); });
  }

  function focus(caseKey, info) {
    if (!root || !controller) return;
    clearFocus();
    // 进问题定位链:展开卡片回到该问题的事故层,不再沿用用户上次手点的那一层
    expandLayer = null;
    focusedCase = { key: caseKey, info: info };
    highlightBadge(caseKey);

    var layer = CASE_LAYER[caseKey];
    root.classList.add('is-diagnosis-focus');
    (info.nodeIds || []).forEach(function (id) {
      var el = findNode(mapId(id), layer);
      if (el) el.classList.add('is-diagnosis-focus-active');
    });
    (info.clusterIds || []).forEach(function (cid) {
      var cluster = findCluster(cid, layer);
      if (cluster) cluster.classList.add('is-diagnosis-focus-active');
    });

    if (typeof layer === 'number') {
      controller.setFrontLayer(layer);
      // 聚焦一律回正视:正视下只有 is-front-layer 那层完全不透明且可交互,
      // 天然就是 v2 那套"命中区域留亮、其余淡出"的效果。若用户当时停在侧视,
      // 这里会改变视图,所以工具栏的高亮也要跟着回正。
      controller.setView(FOCUS_VIEW);
      syncSeg('deckViewSeg', 'data-deck-view', FOCUS_VIEW);
      // setView() 内部 fit() 只会把"整层"塞进视口,红框(如问题一的 MoE FFN 分组框)在
      // 整层里往往只占一小块,还得再手动缩放/拖拽才看得清楚 —— 这里紧接着再做一次
      // 精确到命中区域的缩放平移,同步覆盖掉 fit() 刚写的 transform。
      zoomToFocusTarget(layer, info);
    }
  }

  // 把画面缩放平移到本次命中的 cluster(优先,如「MoE FFN」红框)或节点并集的中心,
  // 让红框在视口里保持合适大小,不需要用户再手动缩放/拖拽找它。坐标沿用 showExpertExpand()
  // 的做法:cluster/节点是 .pto-model-deck__graph 内 left/top 绝对定位的 DOM,offsetLeft/Top
  // 是 3D 变换前的原始布局坐标,与 pivot 的 x/y 同一套坐标系,不需要再做投影换算。
  function zoomToFocusTarget(layer, info) {
    var viewport = qs('.pto-model-deck__viewport', root);
    if (!viewport) return;
    var bounds = null;
    var expand = function (el) {
      if (!el) return;
      var x = el.offsetLeft, y = el.offsetTop, w = el.offsetWidth, h = el.offsetHeight;
      if (!w || !h) return;
      if (!bounds) bounds = { x1: x, y1: y, x2: x + w, y2: y + h };
      else {
        bounds.x1 = Math.min(bounds.x1, x); bounds.y1 = Math.min(bounds.y1, y);
        bounds.x2 = Math.max(bounds.x2, x + w); bounds.y2 = Math.max(bounds.y2, y + h);
      }
    };
    // 1) 优先用命中的 cluster(红框本体)定边界
    (info.clusterIds || []).forEach(function (cid) { expand(findCluster(cid, layer)); });
    // 2) 没有 cluster 命中(纯节点类问题)则退化为命中节点的并集
    if (!bounds) {
      (info.nodeIds || []).forEach(function (id) { expand(findNode(mapId(id), layer)); });
    }
    if (!bounds) return;

    var cx = (bounds.x1 + bounds.x2) / 2, cy = (bounds.y1 + bounds.y2) / 2;
    var w = viewport.clientWidth, h = viewport.clientHeight;
    var padX = 60, padY = 60;
    var zoom = Math.min(w / (bounds.x2 - bounds.x1 + padX * 2), h / (bounds.y2 - bounds.y1 + padY * 2));
    zoom = clamp(zoom, 0.4, 1.3);
    controller.setPose({
      view: FOCUS_VIEW, zoom: zoom, panX: 0, panY: 0,
      pivot: { x: cx, y: cy, z: -layer * controller.config.depthGap },
    });
  }

  /* ════════════════════════════════════════════════════════════════════════════
     3. routed_expert_bank(deck 里叫 expert_pool)原地展开 + 256 专家负载热力动画
     卡片内容与动画驱动完全复用 training-run-twin.js 的 buildExpertBankExpandMarkup()
     与 startExpertHeat()(通过参数传进来),这里只负责把那段 SVG 放到 deck 上正确的位置:
     以 expert_pool 节点中心为原点、560×210 的 viewBox —— 与 markup 自身的坐标系
     (ox=-W/2, oy=-H/2)恰好对齐,一个数都不用改。
     ════════════════════════════════════════════════════════════════════════════ */
  function hideExpertExpand() {
    if (!root) return;
    qsa('.v3-expert-expand', root).forEach(function (el) { el.remove(); });
    qsa('[data-expand-dimmed]', root).forEach(function (el) {
      el.style.opacity = '';
      delete el.dataset.expandDimmed;
    });
  }

  function showExpertExpand(buildMarkup, startA2A) {
    if (!root || !controller) return;
    // 先取层号再 hideExpertExpand():点击写进 expandLayer,问题定位链走 null → 事故层
    var layer = expandLayer == null ? ROUTED_EXPAND_LAYER : expandLayer;
    hideExpertExpand();
    var pool = findNode('expert_pool', layer);
    if (!pool || !pool.parentElement) return;
    var graph = pool.parentElement;

    var cx = pool.offsetLeft + pool.offsetWidth / 2;
    var cy = pool.offsetTop + pool.offsetHeight / 2;
    var left = cx - EXPAND_W / 2, top = cy - EXPAND_H / 2;

    var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    // pto-node-expand-card 沿用单屏原地展开卡片的类名,卡片视觉与整网图一致;
    // 卡片内 256 个专家格子的热力过渡由 css/training-run-twin.css 的 .lv-heat-cell 负责
    // (无祖先作用域,三个展开入口共用一条规则),无需在本文件重复一遍。
    svg.setAttribute('class', 'v3-expert-expand pto-node-expand-card');
    svg.setAttribute('viewBox', (-EXPAND_W / 2) + ' ' + (-EXPAND_H / 2) + ' ' + EXPAND_W + ' ' + EXPAND_H);
    svg.style.cssText = 'position:absolute;left:' + left + 'px;top:' + top + 'px;width:' + EXPAND_W + 'px;height:' + EXPAND_H + 'px;overflow:visible;z-index:6';
    svg.innerHTML = buildMarkup();
    graph.appendChild(svg);

    // 卡片比原节点大得多,几何上被盖住的同层节点、问题标注与连到 expert_pool 的连线先淡出,
    // 收起时原样淡回(deck 的节点是 DOM 绝对定位,判包围盒相交即可,不必解析 transform)。
    // 问题标注(.v3-problem-badge)也要算进来 —— 它是贴在锚点节点上方的两行标签条,
    // 卡片一撑开就会有标注浮在卡片上面,读起来像是这张热力图在讲那个问题,必须一起让开。
    qsa('.pto-model-deck__node,.pto-model-deck__experts,.v3-problem-badge', graph).forEach(function (el) {
      if (el === pool) return;
      var hitX = el.offsetLeft < left + EXPAND_W && el.offsetLeft + el.offsetWidth > left;
      var hitY = el.offsetTop < top + EXPAND_H && el.offsetTop + el.offsetHeight > top;
      if (!hitX || !hitY) return;
      el.style.transition = 'opacity 200ms ease';
      el.style.opacity = '0';
      el.dataset.expandDimmed = 'expert_pool';
    });
    qsa('path[data-source="expert_pool"],path[data-target="expert_pool"]', graph).forEach(function (p) {
      p.style.transition = 'opacity 200ms ease';
      p.style.opacity = '0';
      p.dataset.expandDimmed = 'expert_pool';
    });

    // 展开卡片所在层必须是当前前置层,否则正视下它整层都是 display:none
    controller.setFrontLayer(layer);
    // 侧视下 46 层是侧着排的,卡片会跟着该层一起被压成一条,读不出任何东西 ——
    // 训练中点 router/expert pool 展开时可能正停在侧视总览,这里一并切回正视。
    if (controller.state.view !== FOCUS_VIEW) {
      controller.setView(FOCUS_VIEW);
      syncSeg('deckViewSeg', 'data-deck-view', FOCUS_VIEW);
    }
    startA2A(svg);
  }

  /* ════════════════════════════════════════════════════════════════════════════
     4. 问题二(HiF8)溢出率徽标:命中算子右上角的红/绿药丸
     ════════════════════════════════════════════════════════════════════════════ */
  function clearOverflowBadges() {
    if (!root) return;
    qsa('.v3-over-badge', root).forEach(function (el) { el.remove(); });
    qsa('.c7over-node', root).forEach(function (el) {
      el.classList.remove('c7over-node', 'c7over-node-crit', 'c7over-node-ok');
    });
  }

  function refreshOverflowBadges(map) {
    if (!root || !map) return 0;
    clearOverflowBadges();
    // 溢出率是"单个代表性 decoder layer"的口径(见 hif8-case7.js graphNodeIdOf),
    // 因此统一落在当前前置层上,与 deck 正视图看到的那一层一致。
    var frontCard = qs('.pto-model-deck__layer.is-front-layer', root);
    var frontLayer = frontCard ? Number(frontCard.dataset.layer) : ROUTED_EXPAND_LAYER;
    var hit = 0;
    Object.keys(map).forEach(function (id) {
      var info = map[id];
      var el = findNode(mapId(id), frontLayer);
      if (!el || !el.parentElement) return;
      el.classList.add('c7over-node', 'c7over-node-' + info.tier);
      var badge = document.createElement('div');
      badge.className = 'v3-over-badge v3-over-' + info.tier;
      badge.style.cssText = 'left:' + (el.offsetLeft + el.offsetWidth + 4) + 'px;top:' + (el.offsetTop - 4) + 'px';
      badge.textContent = (info.over * 100).toFixed(2) + '%';
      badge.title = info.name + ' · 溢出率 ' + (info.over * 100).toFixed(2) + '% · SQNR ' + info.sqnr.toFixed(1) + 'dB';
      el.parentElement.appendChild(badge);
      hit += 1;
    });
    return hit;
  }

  /* ════════════════════════════════════════════════════════════════════════════
     5. 算子去色
     deck 的节点色全部走 --pto-model-deck-<op> 变量,关掉染色 = 把这些变量统一改写成
     中性灰;打开 = 删掉行内覆盖,让 pattern 的 applySemanticPalette() 写回的语义色生效。
     ════════════════════════════════════════════════════════════════════════════ */
  var OP_VARS = ['embedding', 'norm', 'attention', 'linear', 'head', 'mlp', 'act', 'gate', 'moe', 'comm', 'decoder', 'input', 'output', 'parameter', 'state'];
  var NEUTRAL = '#9ca3af';

  function applyOpColorMode() {
    if (!root || !controller) return;
    if (global._opColorMode === 'off') {
      OP_VARS.forEach(function (name) { root.style.setProperty('--pto-model-deck-' + name, NEUTRAL); });
    } else {
      OP_VARS.forEach(function (name) { root.style.removeProperty('--pto-model-deck-' + name); });
      controller.setTheme(controller.state.theme);   // 重新写回当前主题的语义色
    }
  }

  /* ════════════════════════════════════════════════════════════════════════════
     6. 逐层指标曲线(仅侧视)
     参考 precision-debugger 整网 2D 侧视图顶部的「逐层 cosine」折线:一条对齐每一层的
     趋势曲线叠在整网上方。这里把单一 cosine 扩成 temp.md 那张表的 9 个指标(精度/性能/Infra
     各 Top1~3),做成勾选项,默认每类只勾 Top1。曲线数据结合业务按 46 层构造(见 metricSeries)。

     几何沿用 pattern 内置侧视覆盖层的思路:曲线画进 viewport 里的一张屏幕坐标 SVG,
     每帧(pan/zoom/rotate/切视图)通过 pattern 的 onOverlay 钩子回调,按 layer card 的
     getBoundingClientRect() 逐帧重算 x,于是与整网本体严丝合缝地一起动。曲线带固定落在
     viewport 顶部(侧视下 46 层卡片在竖直中部,顶部一条带天然"在上方")。
     ════════════════════════════════════════════════════════════════════════════ */
  var LAYER_COUNT = 46;
  // 指标定义:key/中文名/类别/类内排名/曲线色/单位/方向(up=越大越好,down=越小越好)。
  // 颜色按类别分色系:精度=暖(红橙黄)、性能=蓝青、Infra=紫粉,类内 Top1 最饱和。
  // badHi/badLo:temp.md「判定异常」阈值,越界的点标红并贴数值(参考 precision-debugger 的坏点读数)
  var METRIC_DEFS = [
    { key: 'grad_weight_l2_norm', name: '权重梯度 L2 范数', cat: '精度', catKey: 'acc', rank: 1, color: '#92400e', unit: '', phase: 'Bwd', badHi: 1.0, badLo: 0.0001 },
    { key: 'hidden_states_std', name: 'hidden-state 标准差', cat: '精度', catKey: 'acc', rank: 2, color: '#f59e0b', unit: '', phase: 'Fwd', badHi: 2.0 },
    { key: 'attention_entropy', name: '注意力权重熵', cat: '精度', catKey: 'acc', rank: 3, color: '#eab308', unit: '', phase: 'Fwd', badHi: 7, badLo: 2 },
    { key: 'layer_fwd_bwd_latency', name: '单层总耗时', cat: '性能', catKey: 'perf', rank: 1, color: '#2563eb', unit: 'ms', phase: 'Fwd+Bwd', badHi: 26 },
    { key: 'layer_mfu', name: '单层 MFU 利用率', cat: '性能', catKey: 'perf', rank: 2, color: '#0891b2', unit: '%', phase: 'Fwd/Bwd', badLo: 40 },
    { key: 'effective_flops_ratio', name: '有效 FLOPs 占比', cat: '性能', catKey: 'perf', rank: 3, color: '#0d9488', unit: '%', phase: 'Fwd/Bwd', badLo: 70 },
    { key: 'peak_activation_mem', name: '激活峰值显存', cat: 'Infra', catKey: 'infra', rank: 1, color: '#7c3aed', unit: 'GB', phase: 'Fwd', badHi: 18 },
    { key: 'hbm_bandwidth_util', name: 'HBM 带宽利用率', cat: 'Infra', catKey: 'infra', rank: 2, color: '#c026d3', unit: '%', phase: 'Fwd+Bwd', badHi: 95 },
    { key: 'pp_transfer_bytes', name: 'PP 层间传输字节', cat: 'Infra', catKey: 'infra', rank: 3, color: '#db2777', unit: 'MB', phase: 'Fwd', badHi: 60 },
    /* 「单层激活值显存」——问题二(显存 OOM)的层级举证,是本页显存故事的层级切面。
       注意它与上面的 peak_activation_mem 不是一回事,两者口径不同、不可互换:
         · peak_activation_mem  执行到本层时的显存水位,沿深度累积(8→19.5 GB);
         · layer_activation_gb  本层自己持有的激活张量(dense 0.70 / MoE 约 0.78 / L38 1.2 GB),
                                46 层求和 = 36.2 GB,正是「峰值构成」里激活值那一项。
       阈值 badHi=1.0:正常层都在 0.7~0.81,只有 L38 因 expert dispatch buffer 越界标红。
       defaultOn:常驻默认勾选(不按 rank===1 那条规则)——它是问题二定位链的层级切面,
       默认就在场,进出问题二时才不必临时改动用户的指标勾选。 */
    { key: 'layer_activation_gb', name: '单层激活值显存', cat: 'Infra', catKey: 'infra', rank: 4, defaultOn: true, color: '#4f46e5', unit: 'GB', phase: 'Fwd', badHi: 1.0 },
  ];
  function metricBad(def, v) {
    if (def.badHi != null && v > def.badHi) return true;
    if (def.badLo != null && v < def.badLo) return true;
    return false;
  }
  // 训练动画里每个指标算「前向数据」还是「反向数据」,决定它的打点方向与时机:
  //   fwd → 前向扫层时(L0→L45)逐层描点;bwd → 等所有层前向亮完,反向扫层(L45→L0)时才往回描点。
  // 依据 temp.md 采集阶段:纯 Fwd 归 fwd;梯度类(Bwd)归 bwd;Fwd+Bwd/分别采集的合并指标,
  // 其显著信号(耗时拖尾、MFU 掉、带宽打满,见 temp.md 补充说明②③)都出现在反向,故归 bwd。
  var METRIC_FLOW = {
    grad_weight_l2_norm: 'bwd', hidden_states_std: 'fwd', attention_entropy: 'fwd',
    layer_fwd_bwd_latency: 'bwd', layer_mfu: 'bwd', effective_flops_ratio: 'bwd',
    peak_activation_mem: 'fwd', hbm_bandwidth_util: 'bwd', pp_transfer_bytes: 'fwd',
    layer_activation_gb: 'fwd',
  };
  function metricFlow(key) { return METRIC_FLOW[key] || 'fwd'; }
  // 默认勾每一类的 Top1,外加显式标了 defaultOn 的常驻项(单层激活值显存)
  var selectedMetrics = {};
  METRIC_DEFS.forEach(function (m) { if (m.rank === 1 || m.defaultOn) selectedMetrics[m.key] = true; });

  // 事故层(与问题标注一致):L33 q_proj 溢出 / L35 低精长尾 / L38 router 塌缩。曲线在这些层做出
  // 与问题一致的形变,让"整网哪几层有问题"在曲线与红色标注上互相印证。
  var INCIDENT = { 33: 1, 35: 1, 38: 1 };
  var DSA_LAYERS = { 0: 1, 3: 1, 6: 1, 9: 1, 12: 1, 15: 1, 18: 1, 21: 1, 24: 1, 27: 1, 30: 1, 33: 1, 36: 1, 39: 1, 42: 1, 45: 1 };
  var STAGE_STARTS = { 12: 1, 23: 1, 35: 1 };   // PP 分段边界(stageRanges 的非首段起点)

  // 确定性伪噪声,保证每次渲染同一条曲线(不抖动)
  function wob(L, seed) { return Math.sin(L * 12.9898 + seed * 78.233) * 0.5; }

  // 按业务构造每个指标的 46 层数据。数值落在 temp.md「判定优秀」区间内,事故层越界。
  var _seriesCache = {};
  function metricSeries(key) {
    if (_seriesCache[key]) return _seriesCache[key];
    var out = [];
    for (var L = 0; L < LAYER_COUNT; L++) {
      var depth = L / (LAYER_COUNT - 1);         // 0..1
      var dense = L < 2, dsa = !!DSA_LAYERS[L], inc = !!INCIDENT[L], boundary = !!STAGE_STARTS[L];
      var v;
      switch (key) {
        // 噪声幅度整体收窄(曲线更平、抖动更小),事故层做出与红色标注一致的形变
        case 'grad_weight_l2_norm':               // 正常 0.001~0.1;L38 梯度爆炸越 1.0 阈值
          v = 0.02 + depth * 0.05 + wob(L, 1) * 0.006;
          if (L === 38) v = 1.15; else if (inc) v = 0.34 + wob(L, 2) * 0.02;
          break;
        case 'hidden_states_std':                 // 缓慢小幅上行;事故层飙升(数值溢出)越 2.0
          v = 0.85 + depth * 0.7 + wob(L, 3) * 0.03;
          if (inc) v += 0.7;
          break;
        case 'attention_entropy':                 // 区间 3~6;DSA 层更聚焦(偏低),L38 过低越 2.0
          v = 4.8 - depth * 0.9 + wob(L, 4) * 0.26 + (dsa ? -0.55 : 0.28);
          if (L === 38) v = 1.7;
          break;
        case 'layer_fwd_bwd_latency':             // ms;MoE 比 Dense 重,DSA 更重,L38 拖尾越 26
          v = 12 + (dense ? 0 : 4.5) + (dsa ? 3.2 : 0) + depth * 2 + wob(L, 5) * 0.55;
          if (L === 38) v += 10;
          break;
        case 'layer_mfu':                         // %;GEMM 层高,MoE/事故层低,L38 跌破 40
          v = 68 - (dense ? 0 : 8) - (dsa ? 5 : 0) + wob(L, 6) * 1.5;
          if (L === 38) v = 31;
          break;
        case 'effective_flops_ratio':             // %;访存重的层偏低,事故层跌破 70
          v = 88 - (dsa ? 9 : 0) - depth * 4 + wob(L, 7) * 1.3;
          if (inc) v -= 12;
          break;
        case 'peak_activation_mem':               // GB;随深度累积,深层越 18 逼近 HBM 上限
          v = 8 + depth * 9 + (dsa ? 1.2 : 0) + wob(L, 8) * 0.28;
          if (L === 38) v += 2.5;
          break;
        /* GB;本层自己持有的激活张量。数值口径来自定位链-openPangu-2.0-Flash.md 案例四 §5:
           dense 层 0.70、MoE 层约 0.78(DSA 层因注意力中间张量再多一点)、L38 因 expert
           dispatch buffer 达 1.2(dense 的 1.7×)。46 层求和 ≈ 36.2 GB,与「峰值构成」里
           激活值 36.2 GB / 56.6% 对得上 —— 改这里的常数要同步核对那个总和。 */
        case 'layer_activation_gb':
          v = dense ? 0.70 : 0.77 + (dsa ? 0.03 : 0) + wob(L, 11) * 0.012;
          if (L === 38) v = 1.2;
          break;
        case 'hbm_bandwidth_util':                // %;注意力/KV 重的层带宽高,L38 越 95 打满
          v = 78 + (dsa ? 12 : 0) + depth * 4 + wob(L, 9) * 1.6;
          if (L === 38) v = 98;
          if (v > 99.5) v = 99.5;
          break;
        case 'pp_transfer_bytes':                 // MB;分段边界出口张量更大,越 60 通信瓶颈
          v = 42 + wob(L, 10) * 2.2 + (boundary ? 26 : 0);
          break;
        default: v = 0;
      }
      out.push(v);
    }
    _seriesCache[key] = out;
    return out;
  }

  /* ── 训练过程动画 ───────────────────────────────────────────────────────────
     侧视图播放"一个训练 step":前向 L0→L45(1s/层)逐层点亮 + 前向指标逐层描点;
     全部亮完后,反向 L45→L0(0.2s/层)沿途回描反向指标。未执行到的层压到 30% 透明。
     一个 step 走完短暂停顿后循环,持续体现"训练在进行"。仅侧视(right)播放,离开即停并复位。

     进度用两个整数表达,renderMetricCurve 与层点亮都读它:
       fwdDone:已完成前向的层数(0..46) → fwd 指标画 L∈[0,fwdDone) 的点;层 L<fwdDone 点亮。
       bwdDone:已完成反向的层数(从后往前 0..46) → bwd 指标画 L∈[46-bwdDone,46) 的点。 */
  var FWD_MS = 1000, BWD_MS = 200, HOLD_MS = 700;   // 前向 1s/层、反向 0.2s/层、相位间停顿
  var animStatic = !!(global.matchMedia && global.matchMedia('(prefers-reduced-motion: reduce)').matches);
  var anim = { running: false, raf: 0, t0: 0, fwdDone: 0, bwdDone: 0, phase: 'idle' };
  /* 「回顾模式」：进入问题定位链时把动画定格。这时页面讲的是一件已经发生完的事故——
     层还在一层层点亮、指标还在往前描点，会读成"训练正在跑"，与回放语义冲突。
     定格 = 停 rAF + 把进度钉在"一个完整 step 已走完"，于是 46 层全亮、所有指标点都在，
     曲线是这一步的最终形态，可以静态对照。退出时光机时解除(见 setFrozen)。 */
  var animFrozen = false;

  function animProgress() {
    return animStatic
      ? { fwd: LAYER_COUNT, bwd: LAYER_COUNT }
      : { fwd: Math.max(0, anim.fwdDone), bwd: Math.max(0, anim.bwdDone) };
  }

  // 按当前 fwdDone 点亮层:L<fwdDone 已完成前向 → 正常;其余 → is-pending(CSS 压到 30%)。
  // lastLitFwd 去抖:同一进度不重复改 46 个 class。关键是——点亮与曲线揭示都用同一个 prog.fwd,
  // 且都在 renderMetricCurve 里一次算完(见下),从根上保证"亮到第几层"与"曲线画到第几层"永远一致。
  var lastLitFwd = -1;
  function applyLayerLighting(fwdDone) {
    if (!root || fwdDone === lastLitFwd) return;
    lastLitFwd = fwdDone;
    qsa('.pto-model-deck__layer[data-layer]', root).forEach(function (card) {
      card.classList.toggle('is-pending', Number(card.dataset.layer) >= fwdDone);
    });
  }
  function clearLayerLighting() {
    lastLitFwd = -1;
    if (!root) return;
    qsa('.pto-model-deck__layer.is-pending', root).forEach(function (c) { c.classList.remove('is-pending'); });
  }
  function updateAnimStatus() {
    if (!metricStatusEl) return;
    if (animFrozen) { metricStatusEl.textContent = '回顾模式 · 已定格在完整一步'; return; }
    if (animStatic) { metricStatusEl.textContent = '训练过程动画（已按系统偏好关闭）'; return; }
    var p = animProgress();
    if (anim.phase === 'fwd') metricStatusEl.textContent = '前向传播 · L' + Math.min(p.fwd, LAYER_COUNT - 1) + ' / ' + (LAYER_COUNT - 1) + '（1s/层)';
    else if (anim.phase === 'bwd') metricStatusEl.textContent = '反向传播 · L' + Math.max(0, LAYER_COUNT - p.bwd) + ' / 0（0.2s/层)';
    else if (anim.phase === 'fwd-hold') metricStatusEl.textContent = '前向完成 · 46 层已点亮，待反向';
    else metricStatusEl.textContent = '一个 step 完成 · 循环下一轮';
  }

  function animTick(now) {
    if (!anim.running) return;
    var fwdDur = LAYER_COUNT * FWD_MS, bwdDur = LAYER_COUNT * BWD_MS;
    var cycle = fwdDur + HOLD_MS + bwdDur + HOLD_MS;
    var tc = (now - anim.t0) % cycle;
    var fwdDone, bwdDone, phase;
    if (tc < fwdDur) { phase = 'fwd'; fwdDone = Math.floor(tc / FWD_MS); bwdDone = 0; }
    else if (tc < fwdDur + HOLD_MS) { phase = 'fwd-hold'; fwdDone = LAYER_COUNT; bwdDone = 0; }
    else if (tc < fwdDur + HOLD_MS + bwdDur) { phase = 'bwd'; fwdDone = LAYER_COUNT; bwdDone = Math.floor((tc - fwdDur - HOLD_MS) / BWD_MS); }
    else { phase = 'bwd-hold'; fwdDone = LAYER_COUNT; bwdDone = LAYER_COUNT; }

    if (fwdDone !== anim.fwdDone || bwdDone !== anim.bwdDone || phase !== anim.phase) {
      // 进入 bwd-hold = 这一轮前向+反向刚播完,即完成一个 step;与旧 v2(training-monitoring-v2.html
      // 按「跑满一圈」推进 twinAdvanceStep)口径一致,让页面顶栏的训练进度计数跟着层视图的
      // 播放节奏走,而不是只在「实时监控」标签页打开时才推进。
      var justFinishedCycle = phase === 'bwd-hold' && anim.phase !== 'bwd-hold';
      anim.fwdDone = fwdDone; anim.bwdDone = bwdDone; anim.phase = phase;
      updateAnimStatus();
      // 只推进状态并触发重画;层点亮交给 renderMetricCurve 用同一个 prog.fwd 一次算完,
      // 避免"这里按 fwdDone 点亮、那边按 prog.fwd 画曲线"两处各算导致的错位。
      controller && controller.refresh();
      if (justFinishedCycle && typeof global.twinAdvanceStep === 'function') global.twinAdvanceStep(1);
    }
    anim.raf = requestAnimationFrame(animTick);
  }

  function startAnim() {
    if (animStatic) { anim.running = true; anim.fwdDone = anim.bwdDone = LAYER_COUNT; anim.phase = 'bwd-hold'; clearLayerLighting(); updateAnimStatus(); return; }
    if (anim.running) return;
    anim.running = true;
    anim.t0 = (global.performance && performance.now) ? performance.now() : Date.now();
    anim.fwdDone = -1; anim.bwdDone = -1; anim.phase = 'idle'; lastLitFwd = -1;   // 强制首帧刷新
    anim.raf = requestAnimationFrame(animTick);
  }
  function stopAnim() {
    anim.running = false;
    if (anim.raf) cancelAnimationFrame(anim.raf);
    anim.raf = 0;
  }

  /* 进入/退出「回顾模式」。定格时把进度钉满(46 层全亮、fwd/bwd 指标点全部画出),
     解除后回到 renderMetricCurve 的常规判断——只要还在侧视就会重新 startAnim()。 */
  function setFrozen(v) {
    v = !!v;
    if (animFrozen === v) return;
    animFrozen = v;
    if (v) {
      stopAnim();
      anim.fwdDone = LAYER_COUNT; anim.bwdDone = LAYER_COUNT; anim.phase = 'bwd-hold';
      clearLayerLighting();
      lastLitFwd = -1;
    }
    updateAnimStatus();
    controller && controller.refresh();
  }

  var curveSvg = null;         // viewport 里的曲线 SVG
  var hoverSvg = null;         // 悬浮数据线 + 气泡,独立于 curveSvg(curveSvg 每帧整体重画一次,
                                // 悬浮态要跟着 pointermove 即时刷新,不能等下一次 overlay 帧)
  var lastFrame = null;        // renderMetricCurve 每帧缓存的 xs/lane 几何,供悬浮态按鼠标位置反查层号与数值
  var metricPanel = null;      // 下拉里的勾选面板
  var metricStatusEl = null;   // 面板里的训练进度状态行
  var metricDD = null;         // 下拉容器(工具栏「算子染色」右侧,见 #deckMetricDD 静态 HTML)
  var metricDDLabel = null;    // 下拉按钮上的「N项层指标」文案
  var NS_SVG = 'http://www.w3.org/2000/svg';

  // 逐层指标「?」说明:含义/采集阶段/判定优秀/判定异常,取自 temp.md 对照表;
  // 复用页面已有的 .wzh-help + window.wzhBindHelpTooltips 浮层(training-monitoring-v2.html 文末),
  // 不重新实现一套 tooltip。
  var METRIC_HELP = {
    grad_weight_l2_norm: '含义:本层权重梯度 L2 范数\n采集:反向 Bwd\n优秀:稳定区间 0.001–0.1,迭代间波动小\n异常:＜0.0001 梯度消失;＞1.0 梯度爆炸;剧烈震荡',
    hidden_states_std: '含义:本层前向输出 hidden-state 标准差\n采集:前向 Fwd\n优秀:迭代缓慢平稳收敛,逐步小幅下降\n异常:标准差持续飙升=数值溢出;持续不变=本层停止学习',
    attention_entropy: '含义:注意力权重熵(self-attention 子模块)\n采集:前向 Fwd\n优秀:熵缓慢下降,逐步收敛;区间 3–6\n异常:熵过高(>7)注意力分散不收敛;熵过低(<2)过度聚焦 token,过拟合',
    layer_fwd_bwd_latency: '含义:单层前向+反向总耗时\n采集:Fwd+Bwd(合并采集,可拆分上报)\n优秀:同 batch/seq_len 下,时延波动 ±5% 以内,不持续上涨\n异常:单层时延显著高于其余层;seq 增大时非线性暴涨(Attention 瓶颈)',
    layer_mfu: '含义:单层 MFU 算力利用率\n采集:Fwd、Bwd 分别采集\n优秀:MFU ≥70%(GEMM 层);≥50%(Softmax-Attention)\n异常:MFU＜40%,访存瓶颈,算子低效,算力未打满',
    effective_flops_ratio: '含义:有效浮点算力占本层理论 FLOPs 占比\n采集:Fwd、Bwd 分别采集\n优秀:有效 FLOPs 占比＞85%\n异常:＜70%,大量时间消耗在数据读写,而非计算',
    peak_activation_mem: '含义:本层激活张量峰值显存占用\n采集:前向 Fwd(激活在前向生成,反向复用)\n优秀:单层显存占用稳定,不随迭代持续递增\n异常:显存持续上涨;接近单卡 HBM 上限,会触发 OOM',
    hbm_bandwidth_util: '含义:本层张量读写对应的 HBM 带宽利用率\n采集:Fwd+Bwd(反向读取激活、梯度读写带宽更高)\n优秀:带宽利用率＜85%,留有余量\n异常:持续＞95% 带宽打满,访存阻塞拖累整体训练速度(KV-cache 层典型)',
    pp_transfer_bytes: '含义:PP 流水线,本层输出传递给下一层的传输字节\n采集:前向 Fwd(层间传递前向 hidden-state)\n优秀:层间传输数据量均衡,各切割点字节接近\n异常:某一层出口张量字节远大于其他层,产生通信瓶颈,PP 切分不合理',
    layer_activation_gb: '含义:本层自己持有的激活张量显存(不含参数/梯度/优化器)。注意与「激活峰值显存」口径不同——那条是执行到本层时的累积水位,这条是本层单独占多少,46 层求和即整卡激活总量\n采集:前向 Fwd(激活在前向生成,反向用完才释放)\n优秀:各层接近,dense 约 0.7 GB、MoE 约 0.78 GB;总和留有余量\n异常:单层显著高于同类层(MoE 的 expert dispatch buffer 是典型);总和占满显存过半 → 未开 activation checkpoint,是 OOM 的头号来源',
  };

  function ensureCurveEls() {
    if (!root) return;
    var viewport = root.querySelector('.pto-model-deck__viewport');
    if (viewport && !curveSvg) {
      curveSvg = document.createElementNS(NS_SVG, 'svg');
      curveSvg.setAttribute('class', 'deck-metric-curve');
      curveSvg.setAttribute('aria-hidden', 'true');
      // none:viewBox 单位与像素在 x/y 各自 1:1,绝不因宽高比取整而再加缩放,曲线 x 严格贴合模型层 x
      curveSvg.setAttribute('preserveAspectRatio', 'none');
      viewport.appendChild(curveSvg);       // 与 side-guides 同级,共享 viewport 屏幕坐标
    }
    if (viewport && !hoverSvg) {
      // 悬浮数据线/气泡单独一层,盖在 curveSvg 之上:curveSvg 只在 renderMetricCurve 整帧重画
      // 时更新,悬浮态要跟手,靠 pointermove 直接改这一层,不等下一次 overlay 帧。
      hoverSvg = document.createElementNS(NS_SVG, 'svg');
      hoverSvg.setAttribute('class', 'deck-metric-curve-hover');
      hoverSvg.setAttribute('aria-hidden', 'true');
      hoverSvg.setAttribute('preserveAspectRatio', 'none');
      viewport.appendChild(hoverSvg);
      viewport.addEventListener('pointermove', onDeckHoverMove);
      viewport.addEventListener('pointerleave', onDeckHoverLeave);
    }
    if (!metricPanel) buildMetricPanel();
  }

  // ── 悬浮数据线:侧视下鼠标划过层堆 → 一条纵向线从画布底部拉到曲线堆叠区顶部,
  // 贴着最近的那一层,气泡里报 Layer 号 + 当前勾选各指标在该层的具体值(text-anchor 左右自适应)。
  // 只读 renderMetricCurve 每帧缓存的 lastFrame,不重新量一次 DOM,和曲线本体严格同源不会错位。
  function onDeckHoverMove(e) {
    if (!lastFrame || !hoverSvg) return;
    var viewport = root && root.querySelector('.pto-model-deck__viewport');
    if (!viewport || viewport.classList.contains('is-grabbing')) { hideHoverCrosshair(); return; }
    var rect = viewport.getBoundingClientRect();
    var mx = e.clientX - rect.left;
    if (mx < lastFrame.minX - 24 || mx > lastFrame.maxX + 24) { hideHoverCrosshair(); return; }
    var bestL = -1, bestD = Infinity;
    lastFrame.xs.forEach(function (x, L) {
      if (x == null) return;
      var d = Math.abs(x - mx);
      if (d < bestD) { bestD = d; bestL = L; }
    });
    if (bestL < 0) { hideHoverCrosshair(); return; }
    drawHoverCrosshair(bestL);
  }
  function onDeckHoverLeave() { hideHoverCrosshair(); }
  function hideHoverCrosshair() { if (hoverSvg) hoverSvg.innerHTML = ''; }
  function resetHoverFrame() { lastFrame = null; hideHoverCrosshair(); }

  // 近似文本宽度(CJK 按 10px、其余 6.1px 估算),参考 training-metrics-chart.js 的 textW,
  // 用来给气泡量出刚好够用的宽度,不让中文指标名撑爆或裁切。
  function hoverTextW(str) {
    var w = 0;
    for (var i = 0; i < str.length; i++) w += str.charCodeAt(i) > 0x2e80 ? 10 : 6.1;
    return w;
  }
  // 气泡样式复用 css/training-run-twin.css 里 op-rank-time/precision 系图表已有的
  // .pto-tmchart__cursor-tip-* 类,和页面其它悬浮气泡一个视觉语言,不用重新定义一套。
  function hoverBubbleHtml(cx, topY, lines, canvasWidth) {
    var lh = 14, padX = 8, padY = 6, sw = 8;
    var bw = 0;
    lines.forEach(function (l) { bw = Math.max(bw, (l.color ? sw + 5 : 0) + hoverTextW(l.text)); });
    bw = Math.ceil(bw) + padX * 2;
    var bh = lines.length * lh + padY * 2;
    var bx = cx + 10;
    if (bx + bw > canvasWidth - 4) bx = cx - 10 - bw;
    if (bx < 4) bx = 4;
    // 气泡贴在曲线堆叠区顶部之上(底边落在 topY-6),而不是从 topY 往下长——
    // 否则气泡会整个盖在它刚标出来的那圈打点高亮上面。
    var by = Math.max(4, topY - bh - 6);
    var html = '<g class="deck-metric-curve-hover__tip">' +
      '<rect class="pto-tmchart__cursor-tip-bg" x="' + bx.toFixed(1) + '" y="' + by.toFixed(1) +
      '" width="' + bw + '" height="' + bh + '" rx="5"/>';
    lines.forEach(function (l, k) {
      var ty = by + padY + lh * k + 10;
      var tx = bx + padX;
      if (l.color) {
        html += '<rect x="' + tx.toFixed(1) + '" y="' + (ty - 8).toFixed(1) + '" width="' + sw + '" height="' + sw + '" rx="2" fill="' + l.color + '"/>';
        tx += sw + 5;
      }
      html += '<text class="' + (l.color ? 'pto-tmchart__cursor-tip-val' : 'pto-tmchart__cursor-tip-step') + '" x="' + tx.toFixed(1) + '" y="' + ty.toFixed(1) + '">' + l.text + '</text>';
    });
    html += '</g>';
    return html;
  }
  function drawHoverCrosshair(L) {
    var f = lastFrame, cx = f.xs[L];
    if (cx == null) { hideHoverCrosshair(); return; }
    var prog = animProgress();
    var parts = [];
    parts.push('<line class="pto-tmchart__cursor" x1="' + cx.toFixed(1) + '" y1="' + f.height.toFixed(1) +
      '" x2="' + cx.toFixed(1) + '" y2="' + (f.stackTop - 6).toFixed(1) + '"/>');
    var lines = [{ text: 'Layer ' + L, color: null }];
    f.lanes.forEach(function (lane) {
      var shown = lane.flow === 'bwd' ? (L >= LAYER_COUNT - prog.bwd) : (L < prog.fwd);
      if (!shown) return;                        // 只报已经画出打点的指标,悬浮气泡与曲线本体严格对齐
      var v = lane.series[L];
      if (v == null || !isFinite(v)) return;
      var y = lane.laneBottom - lane.pad - (v - lane.lo) / lane.span * (lane.laneH - 2 * lane.pad);
      parts.push('<circle class="deck-metric-curve-hover__ring" cx="' + cx.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="5" stroke="' + lane.color + '"/>');
      var txt = lane.def.unit === '%' ? v.toFixed(1) + '%' : v.toFixed(v < 10 ? 3 : 1) + (lane.def.unit && lane.def.unit !== '%' ? lane.def.unit : '');
      lines.push({ text: lane.def.name + '  ' + txt, color: lane.color });
    });
    parts.push(hoverBubbleHtml(cx, f.stackTop, lines, f.width));
    hoverSvg.innerHTML = parts.join('');
  }

  function updateMetricDDLabel() {
    if (!metricDDLabel) return;
    var count = METRIC_DEFS.reduce(function (n, m) { return n + (selectedMetrics[m.key] ? 1 : 0); }, 0);
    metricDDLabel.textContent = count + '项层指标';
  }

  function setMetricDDOpen(open) {
    if (!metricDD) return;
    metricDD.classList.toggle('is-open', open);
    var btn = document.getElementById('deckMetricDDBtn');
    if (btn) btn.setAttribute('aria-expanded', String(open));
  }

  function buildMetricPanel() {
    metricDD = document.getElementById('deckMetricDD');
    metricDDLabel = document.getElementById('deckMetricDDLabel');
    if (!metricDD) return;
    metricPanel = document.createElement('div');
    metricPanel.className = 'deck-metric-panel';
    var cats = [
      { key: 'acc', label: '精度' }, { key: 'perf', label: '性能' }, { key: 'infra', label: 'Infra' },
    ];
    var html = '';
    cats.forEach(function (c) {
      html += '<div class="deck-metric-group"><div class="deck-metric-group__h">' + c.label + '</div>';
      METRIC_DEFS.filter(function (m) { return m.catKey === c.key; }).forEach(function (m) {
        html += '<div class="deck-metric-row">' +
          '<label class="deck-metric-row__main">' +
          '<input type="checkbox" data-metric="' + m.key + '"' + (selectedMetrics[m.key] ? ' checked' : '') + '>' +
          '<i class="deck-metric-sw" style="background:' + m.color + '"></i>' +
          '<span class="deck-metric-rank">Top' + m.rank + '</span>' +
          '<span class="deck-metric-name">' + m.name + '</span>' +
          '</label>' +
          '<span class="wzh-help" tabindex="0" data-tooltip="' + (METRIC_HELP[m.key] || '') + '">?</span>' +
          '</div>';
      });
      html += '</div>';
    });
    html += '<div class="deck-metric-panel__status" data-anim-status>训练过程动画</div>';
    html += '<div class="deck-metric-panel__foot">前向 1s/层描点 · 反向待全亮后 0.2s/层回描</div>';
    metricPanel.innerHTML = html;
    metricStatusEl = metricPanel.querySelector('[data-anim-status]');
    metricPanel.addEventListener('change', function (e) {
      var cb = e.target.closest('input[data-metric]');
      if (!cb) return;
      selectedMetrics[cb.dataset.metric] = cb.checked;
      updateMetricDDLabel();
      controller && controller.refresh();       // 触发 onOverlay 重画曲线
    });
    metricDD.appendChild(metricPanel);
    global.wzhBindHelpTooltips && global.wzhBindHelpTooltips(metricPanel);
    updateAnimStatus();
    updateMetricDDLabel();

    var ddBtn = document.getElementById('deckMetricDDBtn');
    if (ddBtn) {
      ddBtn.addEventListener('click', function (e) {
        e.stopPropagation();
        setMetricDDOpen(!metricDD.classList.contains('is-open'));
      });
    }
    document.addEventListener('click', function (e) {
      if (metricDD.classList.contains('is-open') && !metricDD.contains(e.target)) setMetricDDOpen(false);
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && metricDD.classList.contains('is-open')) setMetricDDOpen(false);
    });
  }

  // Catmull-Rom → 三次贝塞尔平滑,把逐层折线抹成顺滑曲线(去锯齿/去抖),参考 op-rank-time 的平滑连线
  function smoothPath(pts) {
    if (pts.length < 2) return '';
    if (pts.length < 3) return 'M' + pts.map(function (p) { return p.x.toFixed(1) + ',' + p.y.toFixed(1); }).join(' L');
    var d = 'M' + pts[0].x.toFixed(1) + ',' + pts[0].y.toFixed(1);
    for (var i = 0; i < pts.length - 1; i++) {
      var p0 = pts[i - 1] || pts[i], p1 = pts[i], p2 = pts[i + 1], p3 = pts[i + 2] || p2;
      var c1x = p1.x + (p2.x - p0.x) / 6, c1y = p1.y + (p2.y - p0.y) / 6;
      var c2x = p2.x - (p3.x - p1.x) / 6, c2y = p2.y - (p3.y - p1.y) / 6;
      d += 'C' + c1x.toFixed(1) + ',' + c1y.toFixed(1) + ' ' + c2x.toFixed(1) + ',' + c2y.toFixed(1) + ' ' + p2.x.toFixed(1) + ',' + p2.y.toFixed(1);
    }
    return d;
  }

  // pattern 每帧覆盖层收尾时回调(见 vendored pattern.js renderOverlays 末尾的 options.onOverlay)
  function renderMetricCurve(ctx) {
    ensureCurveEls();
    var inSide = ctx.view === 'right';
    // 训练动画只在侧视播放:进入侧视启动扫层,离开则停并复位层透明度(其它视图层不受影响)。
    // 「回顾模式」下不自动重启——那时页面在复盘已经发生完的事故,画面应当是定格的。
    if (inSide && !anim.running && !animFrozen) startAnim();
    else if (!inSide && anim.running) { stopAnim(); clearLayerLighting(); }
    if (!curveSvg) return;
    if (!inSide) { curveSvg.innerHTML = ''; curveSvg.style.display = 'none'; resetHoverFrame(); return; }
    curveSvg.style.display = 'block';
    var prog = animProgress();
    // 时序:某层先亮起(开始计算)→ 算完 1s 后才产出该层的曲线打点。
    // 所以「点亮前沿」比「已出点的层」领先一层——第 prog.fwd 层是当前正在计算的层,
    // 已点亮(reached)但还没出点;它的点要等它算完(prog.fwd 进到下一层)才画。
    //   · 层点亮:L <= prog.fwd(已完成 0..prog.fwd-1 + 正在算的 prog.fwd)→ 传 prog.fwd+1;
    //   · 曲线打点(下面):L < prog.fwd,只画已算完的层。
    // 单一真源:两者都由这同一个 prog.fwd 在本次调用里算出,只差"正在算的那层要不要点亮",
    // 天然同源不漂移。放在所有 early-return 之前,保证没勾指标时层点亮也照常推进。
    applyLayerLighting(prog.fwd + 1);

    var viewport = ctx.viewport;
    var cards = qsa('.pto-model-deck__layer[data-layer]', ctx.root);
    // viewBox 必须用「和坐标同一把尺子」的宽高:xs[L] 是按 viewport.getBoundingClientRect()
    // (实际渲染像素框)换算的,所以 viewBox 也要用同一个 rect 的 width/height,而不是 clientWidth
    // ——两者只要差 1px(边框/滚动条/子像素/布局未落定),SVG 就会给 viewBox 乘一个随 x 放大的
    // 缩放系数,曲线越往右越比模型层"跑得快",看起来提前好几层;收起 infra 栏改变视口宽度让两者
    // 恰好相等时才对齐。用同一个 base 的宽高即可从根上消除这个缩放。
    var base = viewport.getBoundingClientRect();
    var width = base.width, height = base.height;
    if (!width || !height || !cards.length) { curveSvg.innerHTML = ''; resetHoverFrame(); return; }
    curveSvg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);

    // 每层中心 x + 卡片顶,曲线要"贴着模型层顶部"而不是独立占一块
    var xs = [], cardTop = Infinity;
    cards.forEach(function (card) {
      var L = Number(card.dataset.layer);
      var r = card.getBoundingClientRect();
      xs[L] = r.left + r.width / 2 - base.left;
      cardTop = Math.min(cardTop, r.top - base.top);
    });
    var minX = Infinity, maxX = -Infinity;
    xs.forEach(function (x) { if (x != null) { minX = Math.min(minX, x); maxX = Math.max(maxX, x); } });

    var keys = METRIC_DEFS.filter(function (m) { return selectedMetrics[m.key]; }).map(function (m) { return m.key; });
    if (!keys.length) { curveSvg.innerHTML = ''; resetHoverFrame(); return; }

    // 曲线组以 layer 顶边为锚点：拖拽或缩放侧视图时，两者保持固定间隔同步移动。
    // lane 尺寸只由 viewport 和指标数量决定，不随拖拽位置变化，避免移动时曲线被挤压变形。
    var LANE_GAP = 9;
    var anchor = cardTop - 14;
    var curveBudget = Math.max(24, height * 0.30 - 8);
    var laneH = clamp((curveBudget - LANE_GAP * Math.max(0, keys.length - 1)) / keys.length, 6, 20);
    // x0 就是曲线最左侧数据点(第0层)的真实 x,不再额外向右夹到「给标签留 96px」的固定下限——
    // 那样夹会导致缩放/平移把模型第0层拉近视口左边时,标签仍摆在夹出来的固定位置,曲线本体
    // (用未夹过的层中心 x 连线)反而画到标签底下。这里情愿让标签在极端情况下被 SVG viewBox
    // 裁掉一截,也不盖住曲线。
    var x0 = minX;
    var xEnd = maxX;

    var parts = [];
    var stackTop = anchor;                                   // 全部 lane 里最靠上的一条的顶,供 PP 分组标注避让
    var lanesInfo = [];                                       // 供悬浮数据线按 L 反查各指标数值,见 drawHoverCrosshair
    keys.forEach(function (key, idx) {
      var def = METRIC_DEFS.find(function (m) { return m.key === key; });
      var series = metricSeries(key);
      // 自底向上堆叠:idx=0 最贴近卡片,后续往上
      var laneBottom = anchor - idx * (laneH + LANE_GAP);
      var laneTop = laneBottom - laneH;
      stackTop = Math.min(stackTop, laneTop);
      var lo = Math.min.apply(null, series), hi = Math.max.apply(null, series), span = (hi - lo) || 1;
      var pad = laneH * 0.14;                                // 上下留白,曲线不贴 lane 边
      var yOf = function (v) { return laneBottom - pad - (v - lo) / span * (laneH - 2 * pad); };

      // lane 基线(虚线,全宽)——先把"轨道"摆出来,曲线随训练进度往里描,参考 op-rank-time 的 lane baseline
      parts.push('<line class="deck-metric-curve__baseline" x1="' + x0.toFixed(1) + '" y1="' + laneBottom.toFixed(1) +
        '" x2="' + xEnd.toFixed(1) + '" y2="' + laneBottom.toFixed(1) + '"/>');

      // 按训练进度 + 前/反向,决定这条曲线当前揭示到哪些层:
      //   fwd → L∈[0,prog.fwd) 随前向扫层从左往右长出;bwd → L∈[46-prog.bwd,46) 反向后从右往左长出。
      var flow = metricFlow(key);
      lanesInfo.push({ def: def, series: series, laneBottom: laneBottom, laneH: laneH, lo: lo, span: span, pad: pad, flow: flow, color: def.color });
      var pts = [];
      for (var L = 0; L < LAYER_COUNT; L++) {
        if (xs[L] == null) continue;
        var shown = flow === 'bwd' ? (L >= LAYER_COUNT - prog.bwd) : (L < prog.fwd);
        if (shown) pts.push({ x: xs[L], y: yOf(series[L]), L: L, v: series[L] });
      }
      // 左侧标签用的竖色标始终画(轨道已建立);数据点还没到时只有轨道+标签
      if (pts.length >= 2) {
        // 平滑曲线主体 + 柔和投影(drop-shadow 在 CSS 里),粗细参考 precision/op-rank-time
        parts.push('<path class="deck-metric-curve__line" d="' + smoothPath(pts) + '" fill="none" stroke="' + def.color + '"/>');
      }

      // 打点:普通层小点带背景描边环(paint-order),事故/越界层加大标红并贴数值(参考 precision-debugger 坏点读数)
      var placedLbl = [];                                    // 贪心留距,数值标注不重叠
      pts.forEach(function (p) {
        var bad = metricBad(def, p.v), inc = !!INCIDENT[p.L];
        var r = bad ? 3.4 : inc ? 2.8 : 2.0;
        var fill = bad ? 'var(--danger, #dc2626)' : def.color;
        parts.push('<circle class="deck-metric-curve__dot" cx="' + p.x.toFixed(1) + '" cy="' + p.y.toFixed(1) +
          '" r="' + r + '" fill="' + fill + '"/>');
        if (bad) {
          var canPlace = placedLbl.every(function (px) { return Math.abs(p.x - px) >= 40; });
          if (canPlace) {
            placedLbl.push(p.x);
            var txt = def.unit === '%' ? p.v.toFixed(0) + '%' : p.v.toFixed(p.v < 10 ? 2 : 1);
            parts.push('<text class="deck-metric-curve__val" x="' + p.x.toFixed(1) + '" y="' + (yOf(p.v) - 6).toFixed(1) +
              '" text-anchor="middle">' + txt + '</text>');
          }
        }
      });

      // 左侧指标名 + 排名 + 单位(text-anchor:end,紧贴曲线起点)。色标从竖线改成小方块,
      // 与左侧算子/模块标注(.pto-model-deck__side-label::before)同款样式,两套标注look一致。
      var cy = (laneTop + laneBottom) / 2, swX = x0 - 14, textX = x0 - 20;
      parts.push('<rect class="deck-metric-curve__sw" x="' + swX.toFixed(1) + '" y="' + (cy - 4).toFixed(1) +
        '" width="8" height="8" rx="2" fill="' + def.color + '"/>');
      parts.push('<text class="deck-metric-curve__label" x="' + textX.toFixed(1) + '" y="' + (cy - 3).toFixed(1) +
        '" text-anchor="end">' + def.name + '</text>');
      parts.push('<text class="deck-metric-curve__sublabel" x="' + textX.toFixed(1) + '" y="' + (cy + 9).toFixed(1) +
        '" text-anchor="end">' + def.cat + (def.unit ? ' · ' + def.unit : '') + '</text>');
    });

    curveSvg.innerHTML = parts.join('');
    lastFrame = { xs: xs, width: width, height: height, stackTop: stackTop, minX: minX, maxX: maxX, lanes: lanesInfo };

    // PP 分组标注(pattern 默认贴卡片顶)在有曲线展示时会被 lane 堆叠区盖住,
    // 这里按实际画出的 lane 堆叠顶再往上让一截,曲线勾选越多让得越高。
    var ppGroups = ctx.root.querySelector('.pto-model-deck__pp-groups');
    if (ppGroups) {
      var groupTop = Math.max(4, stackTop - 14);
      qsa('.pto-model-deck__pp-group-label, .pto-model-deck__pp-group-divider', ppGroups).forEach(function (el) {
        el.style.top = groupTop.toFixed(1) + 'px';
      });
    }
  }

  /* ════════════════════════════════════════════════════════════════════════════
     7. 挂载 + 工具栏接线
     ════════════════════════════════════════════════════════════════════════════ */
  function syncSeg(segId, attr, value) {
    qsa('#' + segId + ' .segbtn').forEach(function (b) {
      b.classList.toggle('on', b.getAttribute(attr) === value);
    });
  }

  function mount() {
    var host = document.getElementById(HOST_ID);
    if (!host || !global.PtoModelArchitecture3dDeck) return;

    controller = global.PtoModelArchitecture3dDeck.render(host, {
      preset: 'openpangu-flash',
      initialView: OVERVIEW_VIEW,
      initialTheme: document.documentElement.dataset.theme === 'light' ? 'light' : 'dark',
      showChrome: false,          // 工具栏用页面自己那条 .opv-topbar,不出组件自带的
      onOverlay: renderMetricCurve,   // 每帧覆盖层收尾:重画侧视逐层指标曲线(仅侧视可见)
    });
    if (!controller) return;
    root = controller.root;

    // 视图 / 缩放:直接打到 controller,状态回写页面的 seg 高亮
    qsa('#deckViewSeg .segbtn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        controller.setView(btn.dataset.deckView);
        syncSeg('deckViewSeg', 'data-deck-view', btn.dataset.deckView);
      });
    });
    var zin = document.getElementById('zoomIn');
    var zout = document.getElementById('zoomOut');
    var zfit = document.getElementById('zoomReset');
    if (zin) zin.addEventListener('click', function () { controller.setZoom(controller.state.zoom * 1.15); });
    if (zout) zout.addEventListener('click', function () { controller.setZoom(controller.state.zoom / 1.15); });
    // Fit = 退出聚焦并回到侧视总览(本页不提供等轴 3D 视角)
    if (zfit) zfit.addEventListener('click', function () {
      clearFocus();
      controller.setView(OVERVIEW_VIEW);
      syncSeg('deckViewSeg', 'data-deck-view', OVERVIEW_VIEW);
    });

    // 算子染色:页面顶栏的 setOpColorMode() 只改 window._opColorMode 并派发 opv-recolor
    document.addEventListener('opv-recolor', applyOpColorMode);

    // 主题:页面顶栏只有一个 #themeToggle,组件跟着 documentElement[data-theme] 走
    new MutationObserver(function () {
      var theme = document.documentElement.dataset.theme === 'light' ? 'light' : 'dark';
      if (controller.state.theme !== theme) {
        controller.setTheme(theme);
        applyOpColorMode();
      }
    }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

    // 工具栏高亮以 controller 的真实状态为准兜底同步一次,避免 HTML 里写死的
    // .on 和 initialView 不一致时,首屏显示的选中项与实际视图对不上。
    syncSeg('deckViewSeg', 'data-deck-view', controller.state.view);

    // deck 内建的 ResizeObserver 在容器尺寸变化时会无条件 fit() 回默认视图
    // (展开 Timeline dock、收起侧列都会触发),把聚焦状态冲掉;这里在其后补一次恢复。
    var restoreTimer = null;
    global.addEventListener('resize', function () {
      if (restoreTimer) clearTimeout(restoreTimer);
      restoreTimer = setTimeout(function () {
        if (focusedCase) focus(focusedCase.key, focusedCase.info);
      }, 120);
    });
  }

  // 注册必须先于 mount():training-run-twin.js 在自己求值时就会跑 boot() →
  // applyDefaultDiagnosisMarkers(),那时适配器必须已经在 window 上。而且一旦 mount()
  // 抛异常(组件没加载、DOM 结构对不上……),先注册能保证 training-run-twin.js 仍然走
  // 适配器分支安全空转,而不是回退去操作本页根本不存在的 #graphStage SVG。
  /* 切回侧视总览 —— 聚光灯定位链的「阶段/层定位」步用它:46 层同时在场,逐层指标曲线
     才读得出「哪几层压力最大」。不碰指标勾选:「单层激活值显存」是 defaultOn 的常驻项,
     本来就在场,不需要为了讲一个问题去临时改动用户的勾选、事后再还原。 */
  function showSideOverview() {
    if (!controller) return;
    // 先解掉「点问题后的聚焦」——那套是正视 + 命中节点留亮 + 同层其余压暗，与侧视总览互斥：
    // 留着它会让 46 层里大半是暗的，逐层曲线要对照的层看不清。
    clearFocus();
    controller.setView(OVERVIEW_VIEW);
    syncSeg('deckViewSeg', 'data-deck-view', OVERVIEW_VIEW);
    controller.refresh();
  }

  global.PtoTwinGraphAdapter = {
    renderMarkers: renderMarkers,
    highlightBadge: highlightBadge,
    focus: focus,
    clearFocus: clearFocus,
    showExpertExpand: showExpertExpand,
    hideExpertExpand: hideExpertExpand,
    refreshOverflowBadges: refreshOverflowBadges,
    clearOverflowBadges: clearOverflowBadges,
    showSideOverview: showSideOverview,
    setFrozen: setFrozen,
    get controller() { return controller; },
  };

  try {
    mount();
  } catch (err) {
    // 组件挂载失败不应该连带打挂整页(时光机/精度图/Timeline 都还能用),
    // 上面几个适配器方法在 root/controller 为 null 时都会直接 return。
    if (global.console) console.error('[v3] 整网图 3D deck 挂载失败:', err);
  }
})(window);
