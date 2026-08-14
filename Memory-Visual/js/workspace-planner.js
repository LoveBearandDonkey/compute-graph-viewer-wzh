/*
  Workspace 规划求解器 —— 场景 6 的分析核心
  ------------------------------------------------------------------
  方案设计：场景6-workspace与GM规划-方案设计.md §4

  纯函数、无 DOM、不依赖芯片型号 —— 后续 CLI 与 Python API 复用同一套实现。

  只回答三个数：
    lowerBound  当前执行序与形状下的理论下界 = 各子计算「同时存活字节」的最大值。
                同一子计算里同时存活的张量必须同时有地址，无法互相覆盖，所以这是
                不搬移、连续分配前提下不可再低的值。
    packed      保持顺序与形状不变、只做地址复用后可达的值（装箱结果）。
    current     当前布局的实际高度。

  差距刻意拆成两段（current-packed = 策略浪费，packed-lowerBound = 装箱碎片）：
  只报一个够不着的最小值会让人放弃，而策略浪费这一段是改分配就能拿到的确定收益。

  装箱等价于 Dynamic Storage Allocation，已知 NP 难 —— 所以这里报的是**可达值**
  而不是最优值，并且把用的是哪种排序策略写进结果，让结论可复核。
*/
(function registerMemVizWorkspacePlanner(global) {
  'use strict';

  function alignUp(value, align) {
    if (!align || align <= 1) return value;
    return Math.ceil(value / align) * align;
  }

  /** 生命周期用子计算序号的闭区间表示，端点相接即视为重叠（同一子计算内两者都活着）。 */
  function livesOverlap(a, b) {
    return a.live.start <= b.live.end && b.live.start <= a.live.end;
  }

  function liveAt(tensor, sg) {
    return sg >= tensor.live.start && sg <= tensor.live.end;
  }

  /**
   * 参与装箱的张量：只有 workspace 角色进来。
   * 输入/输出/常量由调用方给地址，别名并入宿主，留片上的根本不在 GM。
   */
  function packable(run) {
    return run.tensors.filter((t) => t.role === 'workspace' && !t.onChip && !t.aliasOf);
  }

  // ---------------------------------------------------------------
  // 下界：MaxLive
  // ---------------------------------------------------------------
  function maxLive(run) {
    const tensors = packable(run);
    const perSubgraph = run.subgraphs.map((sg, index) => {
      const members = tensors.filter((t) => liveAt(t, index));
      return {
        index,
        id: sg.id,
        name: sg.name,
        bytes: members.reduce((sum, t) => sum + t.size, 0),
        members,
      };
    });
    const peak = perSubgraph.reduce((best, item) => (item.bytes > best.bytes ? item : best),
      perSubgraph[0] || { bytes: 0, members: [] });
    return { perSubgraph, lowerBound: peak.bytes, peak };
  }

  // ---------------------------------------------------------------
  // 装箱：贪心 + 首个可行偏移
  // ---------------------------------------------------------------
  /**
   * 排序策略是这里唯一的旋钮，而且真的会影响结果 —— 三种都跑，取更优者，
   * 三个高度都留在 evidence 里，让「为什么是这个数」可复核。
   *
   *   by-size      经典 BFD，size 降序。会把「体积小、活得久」的张量排到最后，
   *                此时两侧全被占满，只能被挤到峰值之上，凭空抬高一层。
   *   by-lifespan  生命周期长者先占坑。它修掉了 by-size 的上述毛病，但会把
   *                「只活一个子计算的大块」（如 FFN 中间量）推到最后，同样吃亏。
   *   by-order     按拓扑序 first-fit —— 即真实 arena 分配器带空闲链表的行为。
   *                前两者都是「先难后易」，这一条反而常常最紧：程序序天然让
   *                相邻子计算的张量挨在一起，留下的空洞正好被后面的张量接住。
   *
   * 三种都是贪心，谁都不保证最优（装箱等价于 DSA，NP 难），所以结果一律叫「可达值」。
   */
  const ORDERS = {
    'by-size': (list) => list.slice().sort((a, b) => b.size - a.size || a.order - b.order),
    'by-lifespan': (list) => list.slice().sort((a, b) => {
      const la = a.live.end - a.live.start;
      const lb = b.live.end - b.live.start;
      return lb - la || b.size - a.size || a.order - b.order;
    }),
    'by-order': (list) => list.slice().sort((a, b) => a.order - b.order),
  };

  function pack(tensors, align, orderId) {
    const sorted = (ORDERS[orderId] || ORDERS['by-size'])(tensors);
    const placed = [];
    const offsets = new Map();

    sorted.forEach((t) => {
      // 占位区间来自：生命周期重叠者（必须错开地址），以及
      // blockScope 不同者（地址语义不同，即使生命周期不重叠也不许共用 —— 见 §4.5-3）
      const busy = placed
        .filter((p) => livesOverlap(p, t) || p.blockScope !== t.blockScope)
        .map((p) => [offsets.get(p.id), offsets.get(p.id) + p.size])
        .sort((x, y) => x[0] - y[0]);

      let cursor = 0;
      let chosen = null;
      for (let i = 0; i < busy.length; i += 1) {
        const [start, end] = busy[i];
        if (start - cursor >= t.size) { chosen = cursor; break; }
        cursor = Math.max(cursor, alignUp(end, align));
      }
      if (chosen === null) chosen = cursor;
      offsets.set(t.id, chosen);
      placed.push(t);
    });

    const height = sorted.reduce((max, t) => Math.max(max, offsets.get(t.id) + t.size), 0);
    return { order: orderId, offsets, height };
  }

  // ---------------------------------------------------------------
  // 可复用组：同尺寸桶内做区间图着色
  // ---------------------------------------------------------------
  /**
   * 生命周期是区间 ⇒ 冲突图是区间图（完美图），同尺寸桶内按左端点贪心着色是最优的，
   * 色数恰等于该桶的最大同时存活个数 —— 这一段可以放心地说「这就是最少份数」。
   * 跨尺寸时退化为上面的装箱，只报可达值，不宣称最优。
   */
  function reuseGroups(run) {
    const tensors = packable(run);
    const buckets = new Map();
    tensors.forEach((t) => {
      const key = `${t.size}|${t.blockScope}`;
      const list = buckets.get(key) || [];
      list.push(t);
      buckets.set(key, list);
    });

    const groups = [];
    buckets.forEach((list) => {
      if (list.length < 2) return;
      const sorted = list.slice().sort((a, b) => a.live.start - b.live.start || a.order - b.order);
      const colors = []; // 每个颜色 = 一份物理地址，元素按左端点递增
      sorted.forEach((t) => {
        const slot = colors.find((members) => members.every((m) => !livesOverlap(m, t)));
        if (slot) slot.push(t);
        else colors.push([t]);
      });
      colors.filter((members) => members.length > 1).forEach((members) => {
        const total = members.reduce((sum, t) => sum + t.size, 0);
        const peak = Math.max(...members.map((t) => t.size));
        groups.push({
          members,
          peak,
          saving: total - peak,
          blockScope: members[0].blockScope,
        });
      });
    });

    return groups.sort((a, b) => b.saving - a.saving);
  }

  /**
   * 被护栏排除的组合：生命周期不重叠、看着可以合并，但 blockScope 不同。
   * 这类在甘特图上完全看不出问题，必须显式列出来，否则开发者会自己去合。
   */
  function excludedPairs(run) {
    const tensors = packable(run);
    const out = [];
    for (let i = 0; i < tensors.length; i += 1) {
      for (let j = i + 1; j < tensors.length; j += 1) {
        const a = tensors[i];
        const b = tensors[j];
        if (livesOverlap(a, b)) continue;
        if (a.blockScope === b.blockScope) continue;
        out.push({
          a, b,
          saving: Math.min(a.size, b.size),
          reason: `${a.name} 是 ${a.blockScope}、${b.name} 是 ${b.blockScope}，两者地址语义不同`,
        });
      }
    }
    return out.sort((x, y) => y.saving - x.saving);
  }

  // ---------------------------------------------------------------
  // 现有布局的冲突检查
  // ---------------------------------------------------------------
  function addressesOverlap(a, b, offsetOf) {
    const as = offsetOf(a);
    const bs = offsetOf(b);
    return as < bs + b.size && bs < as + a.size;
  }

  function conflicts(run) {
    const tensors = packable(run);
    const offsetOf = (t) => run.layout[t.id];
    const out = [];
    for (let i = 0; i < tensors.length; i += 1) {
      for (let j = i + 1; j < tensors.length; j += 1) {
        const a = tensors[i];
        const b = tensors[j];
        if (!addressesOverlap(a, b, offsetOf)) continue;
        const overlapBytes = Math.min(offsetOf(a) + a.size, offsetOf(b) + b.size)
          - Math.max(offsetOf(a), offsetOf(b));
        if (livesOverlap(a, b)) {
          out.push({ kind: 'lifetime', a, b, overlapBytes });
        } else if (a.blockScope !== b.blockScope) {
          out.push({ kind: 'blockScope', a, b, overlapBytes });
        }
      }
    }
    return out;
  }

  // ---------------------------------------------------------------
  // 入口
  // ---------------------------------------------------------------
  function plan(run) {
    const align = run.workspace.align || 512;
    const tensors = packable(run);
    const { perSubgraph, lowerBound, peak } = maxLive(run);

    const packings = Object.keys(ORDERS).map((id) => pack(tensors, align, id));
    const best = packings.reduce((a, b) => (b.height < a.height ? b : a), packings[0]);

    const current = tensors.reduce((max, t) => Math.max(max, run.layout[t.id] + t.size), 0);
    const padding = tensors.reduce((sum, t) => sum + (t.size - t.dataBytes), 0);
    const onChip = run.tensors.filter((t) => t.role === 'workspace' && t.onChip);
    const aliased = run.tensors.filter((t) => t.role === 'workspace' && t.aliasOf);

    return {
      align,
      tensors,
      onChip,
      aliased,
      perSubgraph,
      peak,
      lowerBound,
      current,
      packed: best.height,
      packings,
      bestOrder: best.order,
      // 两段差距 —— 视图与规则都只认这两个字段，别处不再重算
      policyWaste: Math.max(0, current - best.height),
      packFragment: Math.max(0, best.height - lowerBound),
      ratio: lowerBound ? current / lowerBound : 1,
      padding,
      groups: reuseGroups(run),
      excluded: excludedPairs(run),
      conflicts: conflicts(run),
      budget: run.workspace.budget,
      overBudget: Math.max(0, current - run.workspace.budget),
    };
  }

  /**
   * 把装箱结果落成一份布局（供「复用后布局」视图与候选构建使用）。
   * 不指定策略时取三种里最紧的那一种。
   */
  function layoutOf(run, orderId) {
    const align = run.workspace.align || 512;
    const tensors = packable(run);
    const result = orderId
      ? pack(tensors, align, orderId)
      : Object.keys(ORDERS)
        .map((id) => pack(tensors, align, id))
        .reduce((a, b) => (b.height < a.height ? b : a));
    const layout = {};
    result.offsets.forEach((offset, id) => { layout[id] = offset; });
    return layout;
  }

  global.MemVizWorkspacePlanner = {
    plan, pack, layoutOf, maxLive, reuseGroups, excludedPairs, conflicts,
    livesOverlap, liveAt, packable, alignUp, ORDERS,
  };
})(window);
