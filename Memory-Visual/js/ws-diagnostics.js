/*
  Workspace 规则引擎 —— 场景 6
  ------------------------------------------------------------------
  方案设计：场景6-workspace与GM规划-方案设计.md §6
  输出与 js/diagnostics.js 完全同构的四元组：问题 / 位置 / 量化影响 / 建议动作，
  外加 evidence 原始数据项引用 —— 任何结论都能溯源，不做黑盒推断。

  单独成文件而不是并进 diagnostics.js：那一套的入口签名是 (run, metrics)，
  metrics 依赖片上 region 与流水事件；融合算子这一路没有片上事件，
  硬塞进去会把两边的前置条件都搅浑。规则四元组的形状保持一致即可。

  降噪约定（沿用已有规则引擎的思路）：
    · WS_ABOVE_LOWER_BOUND 触发时，复用建议只出收益最大的 3 条；
    · 已经踩内存（WS_ADDR_CONFLICT）时不再劝人「继续复用」；
    · 每条规则都可关（disabledRules）、阈值可配（thresholds）。
*/
(function registerMemVizWorkspaceDiagnostics(global) {
  'use strict';

  const F = global.MemVizFormat;

  const DEFAULT_THRESHOLDS = {
    wsRatioWarn: 1.2,          // current / lowerBound 超过即告警
    wsReuseSavingBytes: 1024 * 1024,
    wsLongLivedSpanRatio: 0.5, // 跨度占子计算总数的比例
    wsLongLivedSizeRatio: 0.01, // 体积占峰值的比例（小而长命 = 碎片来源）
    wsAlignPaddingBytes: 64 * 1024,
    wsOnChipHeadroomRatio: 0.25, // 允许占用焦点核 UB 容量的比例
  };

  const RULE_META = {
    WS_BUDGET_EXCEEDED: { severity: 'danger', category: '预算', title: 'workspace 超出预算' },
    WS_ADDR_CONFLICT: { severity: 'danger', category: '正确性', title: '地址重叠且生命周期重叠' },
    WS_CROSS_BLOCK_UNSAFE: { severity: 'danger', category: '正确性', title: '跨 block 语义的地址复用' },
    WS_ABOVE_LOWER_BOUND: { severity: 'warn', category: '容量', title: 'workspace 高于理论下界' },
    WS_REUSE_MISSED: { severity: 'warn', category: '复用', title: '生命周期不重叠但未复用' },
    FUSION_SPILL: { severity: 'warn', category: '融合', title: '中间量本可留在片上' },
    WS_DYNSHAPE_RANGE: { severity: 'warn', category: '动态 shape', title: 'workspace 随 shape 浮动' },
    WS_PEAK_NOT_LARGEST: { severity: 'info', category: '容量', title: '最大张量不在峰值上' },
    WS_INPLACE_CANDIDATE: { severity: 'info', category: '融合', title: '可改原地的逐元素子计算' },
    WS_LONG_LIVED_SMALL: { severity: 'info', category: '复用', title: '小而长命的张量劈开了地址空间' },
    WS_PACK_FRAGMENT: { severity: 'info', category: '复用', title: '装箱碎片' },
    WS_SCOPE_BLOCKED: { severity: 'info', category: '复用', title: '护栏排除的复用组合' },
    WS_ALIGN_PADDING: { severity: 'info', category: '对齐', title: '对齐 padding 浪费' },
  };

  let seq = 0;
  function finding(rule, over) {
    const meta = RULE_META[rule];
    return {
      id: `w${seq += 1}`,
      rule,
      severity: meta.severity,
      category: meta.category,
      title: meta.title,
      refs: [],
      evidence: [],
      ...over,
    };
  }

  const SEVERITY_ORDER = { danger: 0, warn: 1, info: 2 };

  function sgName(run, index) {
    const sg = run.subgraphs[index];
    return sg ? `${sg.id} ${sg.name}` : `sg${index}`;
  }

  function liveText(run, tensor) {
    return tensor.live.start === tensor.live.end
      ? sgName(run, tensor.live.start)
      : `${sgName(run, tensor.live.start)} → ${sgName(run, tensor.live.end)}`;
  }

  function reportedAtText(run) {
    return `${run.workspace.reportedAt.file}:${run.workspace.reportedAt.line + 1}`;
  }

  // ---------------------------------------------------------------
  // 预算与总量
  // ---------------------------------------------------------------
  function budgetRules(run, plan, th, out) {
    if (plan.overBudget > 0) {
      out.push(finding('WS_BUDGET_EXCEEDED', {
        detail: `当前 workspace ${F.bytes(plan.current)}，超出上游给这个算子的预算 ${F.bytes(plan.budget)}。`,
        impact: `超出 ${F.bytes(plan.overBudget)}。这不会被编译器拦截，只会在上板时挤掉别的算子的显存。`,
        suggest: plan.packed <= plan.budget
          ? `仅做地址复用即可降到 ${F.bytes(plan.packed)}，回到预算内，无需改动任何计算。`
          : `即使做满地址复用也只能降到 ${F.bytes(plan.packed)}，仍超预算 ${F.bytes(plan.packed - plan.budget)}，必须改结构或切分。`,
        refs: plan.peak.members.map((t) => t.id),
        srcRef: run.workspace.reportedAt,
        evidence: [
          { label: '当前值', value: F.bytes(plan.current) },
          { label: '预算', value: F.bytes(plan.budget) },
          { label: '上报位置', value: reportedAtText(run) },
        ],
      }));
    }

    if (plan.ratio > th.wsRatioWarn) {
      out.push(finding('WS_ABOVE_LOWER_BOUND', {
        detail: `当前 workspace ${F.bytes(plan.current)}，而这个执行序与形状下的理论下界是 ${F.bytes(plan.lowerBound)}（峰值在 ${sgName(run, plan.peak.index)}）。`,
        impact: `比值 ${plan.ratio.toFixed(2)}×。其中 ${F.bytes(plan.policyWaste)} 是分配策略浪费（改地址即可拿到），`
          + `${F.bytes(plan.packFragment)} 是装箱碎片（需动顺序或形状）。`,
        suggest: `按「${plan.bestOrder}」策略重排可降到 ${F.bytes(plan.packed)}；右栏「可复用组合」给出具体组合。改完把 ${reportedAtText(run)} 的上报值同步改小。`,
        refs: plan.peak.members.map((t) => t.id),
        subgraph: plan.peak.index,
        srcRef: run.workspace.reportedAt,
        evidence: [
          { label: '当前值', value: F.bytes(plan.current) },
          { label: '复用后可达', value: `${F.bytes(plan.packed)}（${plan.bestOrder}）` },
          { label: '理论下界', value: `${F.bytes(plan.lowerBound)} @ ${sgName(run, plan.peak.index)}` },
          ...plan.packings.map((p) => ({ label: `装箱 ${p.order}`, value: F.bytes(p.height) })),
        ],
      }));
    }

    if (plan.padding >= th.wsAlignPaddingBytes) {
      out.push(finding('WS_ALIGN_PADDING', {
        detail: `${plan.tensors.length} 个 workspace 张量按 ${plan.align}B 对齐后累计 padding ${F.bytes(plan.padding)}。`,
        impact: `占当前 workspace 的 ${F.pct(plan.padding / Math.max(1, plan.current), 1)}。`,
        suggest: '张量数很多时才值得处理；优先合并小张量而不是逐个改对齐。',
        refs: plan.tensors.filter((t) => t.size > t.dataBytes).map((t) => t.id).slice(0, 3),
        evidence: [{ label: '对齐粒度', value: `${plan.align}B` }],
      }));
    }
  }

  // ---------------------------------------------------------------
  // 正确性：现有布局里的冲突
  // ---------------------------------------------------------------
  function conflictRules(run, plan, th, out) {
    plan.conflicts.forEach((c) => {
      const manual = c.a.manualReuseOf || c.b.manualReuseOf;
      if (c.kind === 'lifetime') {
        out.push(finding('WS_ADDR_CONFLICT', {
          detail: `${c.a.name} 与 ${c.b.name} 在 GM 上地址重叠 ${F.bytes(c.overlapBytes)}，且生命周期在 `
            + `${sgName(run, Math.max(c.a.live.start, c.b.live.start))} 上仍然交叠。`,
          impact: '后写方会覆盖前者仍需读取的数据。不报错、不报警，只在特定 shape 与 block 数下偶发错结果。',
          suggest: manual
            ? `撤销手工复用，或把复用点后移到 ${c.a.name} 的最后一个消费者之后。`
            : '为其中一方单独分配地址，或在两者之间插入同步。',
          refs: [c.a.id, c.b.id],
          evidence: [
            { label: c.a.name, value: `${F.hex(run.layout[c.a.id], 6)} + ${F.bytes(c.a.size)} · ${liveText(run, c.a)}` },
            { label: c.b.name, value: `${F.hex(run.layout[c.b.id], 6)} + ${F.bytes(c.b.size)} · ${liveText(run, c.b)}` },
          ],
        }));
      } else {
        const perBlock = c.a.blockScope === 'per-block' ? c.a : c.b;
        const shared = perBlock === c.a ? c.b : c.a;
        out.push(finding('WS_CROSS_BLOCK_UNSAFE', {
          detail: `${shared.name}（shared）压在 ${perBlock.name}（per-block）的地址上，重叠 ${F.bytes(c.overlapBytes)}。`
            + `两者生命周期确实错开 —— 甘特图上看不出任何问题。`,
          impact: `${perBlock.name} 被 ${run.kernel.blockDim} 个 block 各持有一段，`
            + `${shared.name} 却被所有 block 当成同一块整体访问。`
            + '无全局同步点时，快的 block 会覆盖慢的 block 还没读完的数据。',
          suggest: `撤销这处复用；若确实要复用，需在两者之间插入所有 block 都已读完的全局同步点，并把 ${perBlock.name} 改为 shared 语义。`,
          refs: [c.a.id, c.b.id],
          evidence: [
            { label: `${perBlock.name} 作用域`, value: `per-block × ${run.kernel.blockDim}` },
            { label: `${shared.name} 作用域`, value: 'shared' },
            { label: '生命周期是否交叠', value: '否（所以只靠甘特图判断会误判）' },
          ],
        }));
      }
    });
  }

  // ---------------------------------------------------------------
  // 复用
  // ---------------------------------------------------------------
  function reuseRules(run, plan, th, out) {
    // 已经踩内存时不劝人继续复用 —— 先把正确性问题解决
    const hasConflict = plan.conflicts.some((c) => c.kind === 'lifetime');
    if (!hasConflict && plan.policyWaste > 0) {
      plan.groups
        .filter((g) => g.saving >= th.wsReuseSavingBytes)
        .slice(0, 3)
        .forEach((g) => {
          const names = g.members.map((t) => t.name);
          const chain = g.members
            .map((t) => `${t.name} ${liveText(run, t)}`)
            .join('，');
          out.push(finding('WS_REUSE_MISSED', {
            detail: `${names.join(' / ')} 在任何子计算上都不同时存活，却各占一段地址（${chain}）。`,
            impact: `合并为一份可释放 ${F.bytes(g.saving)}，占当前 workspace 的 ${F.pct(g.saving / Math.max(1, plan.current), 1)}。`,
            suggest: `让三者共用同一段地址即可（组内峰值 ${F.bytes(g.peak)}）。生效条件：保留现有子计算顺序与同步 —— `
              + `${g.members[0].name} 的最后一个消费者结束后，${g.members[1].name} 才产出。`,
            refs: g.members.map((t) => t.id),
            evidence: [
              ...g.members.map((t) => ({ label: t.name, value: `${F.bytes(t.size)} · ${liveText(run, t)}` })),
              { label: '组内峰值', value: F.bytes(g.peak) },
              { label: '作用域', value: g.blockScope },
            ],
          }));
        });
    }

    if (plan.packFragment > 0) {
      const spread = plan.packings.map((p) => `${p.order} ${F.bytes(p.height)}`).join('、');
      out.push(finding('WS_PACK_FRAGMENT', {
        detail: `复用做到最紧仍是 ${F.bytes(plan.packed)}，比下界 ${F.bytes(plan.lowerBound)} 高 ${F.bytes(plan.packFragment)}。`,
        impact: '这部分不是「没复用」，是装不进去 —— 张量尺寸与生命周期的组合在地址空间上留下了填不满的空洞。',
        suggest: `装箱是 NP 难问题，工具报的是可达值：${spread}。想再压需要改顺序或形状，见「可改原地」与「本可留在片上」两条。`,
        refs: plan.tensors.slice(0, 2).map((t) => t.id),
        evidence: plan.packings.map((p) => ({ label: p.order, value: F.bytes(p.height) })),
      }));
    }

    if (plan.excluded.length) {
      const top = plan.excluded[0];
      out.push(finding('WS_SCOPE_BLOCKED', {
        detail: `有 ${plan.excluded.length} 组组合生命周期完全错开、看着可以合并，但被 blockScope 护栏排除。`,
        impact: `其中收益最大的是 ${top.a.name} / ${top.b.name}，本可省 ${F.bytes(top.saving)}。`,
        suggest: `不要手工合并这些组合 —— ${top.reason}。要合并须先统一作用域。`,
        refs: [top.a.id, top.b.id],
        evidence: plan.excluded.slice(0, 4).map((e) => ({
          label: `${e.a.name} / ${e.b.name}`,
          value: `${e.a.blockScope} vs ${e.b.blockScope}`,
        })),
      }));
    }
  }

  // ---------------------------------------------------------------
  // 结构：降低下界本身
  // ---------------------------------------------------------------
  function structureRules(run, plan, chip, th, out) {
    // 峰值构成 vs 最大张量 —— 抑制「先去砍最大那块」的无效努力
    const largest = plan.tensors.slice().sort((a, b) => b.size - a.size)[0];
    if (largest && !plan.peak.members.some((t) => t.id === largest.id)) {
      out.push(finding('WS_PEAK_NOT_LARGEST', {
        detail: `单个最大的张量是 ${largest.name}（${F.bytes(largest.size)}），但它只活在 ${liveText(run, largest)}；`
          + `峰值发生在 ${sgName(run, plan.peak.index)}，那里根本没有它。`,
        impact: `把 ${largest.name} 砍到 0 也降不了 workspace —— 峰值由 `
          + `${plan.peak.members.map((t) => t.name).join(' + ')} 共同顶起来。`,
        suggest: `要降峰值就动 ${sgName(run, plan.peak.index)} 的存活集合：复用、原地、或把该子计算再切细。`,
        refs: [largest.id, ...plan.peak.members.map((t) => t.id)],
        subgraph: plan.peak.index,
        evidence: [
          { label: `${largest.name} 存活`, value: liveText(run, largest) },
          { label: '峰值子计算', value: sgName(run, plan.peak.index) },
          { label: '峰值构成', value: plan.peak.members.map((t) => t.name).join(' + ') },
        ],
      }));
    }

    // 逐元素子计算的输入输出同形 ⇒ 可原地，直接压低下界。
    // 同一个子计算里的多对（RoPE 的 Q / K）合成一条 —— 处方是同一个，
    // 拆成多条只会稀释注意力。
    run.subgraphs.forEach((sg, index) => {
      if (sg.kind !== 'elementwise') return;
      const reads = plan.tensors.filter((t) => t.consumers.includes(sg.id));
      const writes = plan.tensors.filter((t) => t.producer === sg.id);
      const pairs = [];
      const taken = new Set();
      writes.forEach((w) => {
        const host = reads.find((r) => r.size === w.size && r.dtype === w.dtype && !taken.has(r.id));
        if (!host) return;
        taken.add(host.id);
        pairs.push({ host, w });
      });
      if (!pairs.length) return;

      const saved = pairs.reduce((sum, p) => sum + p.w.size, 0);
      out.push(finding('WS_INPLACE_CANDIDATE', {
        detail: `${sg.name} 是逐元素计算，${pairs.map((p) => `${p.host.name} → ${p.w.name}`).join('、')} `
          + `形状与 dtype 完全一致（${F.shape(pairs[0].w.shape)} ${pairs[0].w.dtype}）。`,
        impact: `改原地后 ${pairs.map((p) => p.w.name).join(' / ')} 不再申请，`
          + `${sgName(run, index)} 的存活集合直接少 ${F.bytes(saved)} —— `
          + '这压低的是下界本身，复用做到极致也拿不到这部分。',
        suggest: `让 ${sg.name} 写回自己的输入，并把 ${pairs.map((p) => p.host.name).join(' / ')} 的生命周期`
          + '延长到原输出的最后一个消费者。',
        refs: pairs.flatMap((p) => [p.host.id, p.w.id]),
        subgraph: index,
        evidence: [
          ...pairs.flatMap((p) => [
            { label: p.host.name, value: `${F.bytes(p.host.size)} · ${liveText(run, p.host)}` },
            { label: p.w.name, value: `${F.bytes(p.w.size)} · ${liveText(run, p.w)}` },
          ]),
          { label: '子计算类型', value: sg.kind },
        ],
      }));
    });

    // 小到能留在片上的中间量：收益大于任何复用（整块消失）
    const ub = chip.regions.find((r) => r.id === 'UB');
    const headroom = ub ? ub.capacity * th.wsOnChipHeadroomRatio : 0;
    plan.tensors.filter((t) => t.size <= headroom).forEach((t) => {
      out.push(finding('FUSION_SPILL', {
        detail: `${t.name} 只有 ${F.bytes(t.size)}，却作为 workspace 落在 GM 上。`
          + `${ub.label} 容量 ${F.bytes(ub.capacity)}，留下它绰绰有余。`,
        impact: `它横跨 ${t.live.end - t.live.start + 1} 个子计算，在地址空间中间钉了一根桩子 —— `
          + `${plan.packings.filter((p) => p.height > plan.lowerBound).length} / ${plan.packings.length} 种排序策略因它多顶出空洞。`
          + `搬到片上后这一项从 workspace 里整块消失。`,
        suggest: `把 ${t.name} 改为片上常驻（${ub.id}），子计算之间通过片上 buffer 传递，不落 GM。`,
        refs: [t.id],
        evidence: [
          { label: t.name, value: `${F.bytes(t.size)} · ${liveText(run, t)}` },
          { label: `${ub.id} 容量`, value: F.bytes(ub.capacity) },
          { label: '判据', value: `≤ ${F.pct(th.wsOnChipHeadroomRatio, 0)} × ${ub.id}` },
        ],
      }));
    });

    // 小而长命 = 装箱碎片的来源
    const span = run.subgraphs.length;
    plan.tensors.forEach((t) => {
      const tSpan = t.live.end - t.live.start + 1;
      if (tSpan / span < th.wsLongLivedSpanRatio) return;
      if (t.size / Math.max(1, plan.lowerBound) > th.wsLongLivedSizeRatio) return;
      out.push(finding('WS_LONG_LIVED_SMALL', {
        detail: `${t.name} 体积 ${F.bytes(t.size)}（占下界 ${F.pct(t.size / plan.lowerBound, 2)}），`
          + `却横跨 ${tSpan} / ${span} 个子计算。`,
        impact: '它在整段生命周期里都不能被覆盖，把一段连续地址劈成两半，后面的大张量只能绕开它。',
        suggest: `优先考虑留在片上；若必须落 GM，把它排在地址空间的一端（首或尾）而不是中间。`,
        refs: [t.id],
        evidence: [
          { label: '存活跨度', value: liveText(run, t) },
          { label: '产出 / 消费', value: `${t.producer} → ${t.consumers.join(', ')}` },
        ],
      }));
    });

    if (run.shapeRange) {
      const r = run.shapeRange;
      const scale = r.min / r.current;
      out.push(finding('WS_DYNSHAPE_RANGE', {
        detail: `${r.field} 在 [${r.min}, ${r.max}] 之间浮动，workspace 随之线性变化：`
          + `${F.bytes(plan.current * scale)} … ${F.bytes(plan.current)}。`,
        impact: `必须按上界 ${F.bytes(plan.current)} 预留，小 shape 下有 ${F.bytes(plan.current - plan.current * scale)} 白占着。`,
        suggest: '结论按区间给而不是单值；若上界超预算，考虑按 shape 分档上报 workspace，而不是一档吃到底。',
        refs: plan.peak.members.map((t) => t.id),
        srcRef: run.workspace.reportedAt,
        evidence: [
          { label: `${r.field} 范围`, value: `${r.min} … ${r.max}` },
          { label: '下界范围', value: `${F.bytes(plan.lowerBound * scale)} … ${F.bytes(plan.lowerBound)}` },
        ],
      }));
    }
  }

  // ---------------------------------------------------------------
  // 入口
  // ---------------------------------------------------------------
  function analyze(run, plan, chip, options = {}) {
    const th = { ...DEFAULT_THRESHOLDS, ...(options.thresholds || {}) };
    const disabled = new Set(options.disabledRules || []);
    const out = [];

    budgetRules(run, plan, th, out);
    conflictRules(run, plan, th, out);
    reuseRules(run, plan, th, out);
    structureRules(run, plan, chip, th, out);

    return out
      .filter((f) => !disabled.has(f.rule))
      .sort((a, b) => SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity]);
  }

  function summarize(findings) {
    return {
      danger: findings.filter((f) => f.severity === 'danger').length,
      warn: findings.filter((f) => f.severity === 'warn').length,
      info: findings.filter((f) => f.severity === 'info').length,
    };
  }

  global.MemVizWorkspaceDiagnostics = {
    analyze, summarize, RULE_META, DEFAULT_THRESHOLDS,
  };
})(window);
