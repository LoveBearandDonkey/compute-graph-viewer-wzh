/*
  诊断规则引擎 —— 规划文档 §4.4
  ------------------------------------------------------------------
  每条规则输出统一的四元组：问题 / 位置 / 量化影响 / 建议动作，
  并附 evidence（原始数据项引用），保证结论可溯源、不做黑盒推断。

  规则可开关、阈值可配（RULES 表 + options.thresholds），
  避免噪声淹没真问题；后续接 YAML 插件时替换 RULES 的来源即可。
*/
(function registerMemVizDiagnostics(global) {
  'use strict';

  const F = global.MemVizFormat;
  const MET = global.MemVizMetrics;

  const DEFAULT_THRESHOLDS = {
    nearLimitRatio: 0.9,
    underutilizedRatio: 0.5,
    paddingBytes: 512,
    idleRatio: 0.25,
    reuseSavingBytes: 4096,
    tailWasteRatio: 0.15,
  };

  const RULE_META = {
    REGION_OVERFLOW: { severity: 'danger', category: '容量', title: '片上容量超限' },
    REGION_NEAR_LIMIT: { severity: 'warn', category: '容量', title: '容量逼近上限' },
    REGION_UNDERUTILIZED: { severity: 'warn', category: '容量', title: '容量严重闲置' },
    RESERVED_IDLE: { severity: 'info', category: '容量', title: '静态预留远高于实际持有' },
    ALIGN_PADDING: { severity: 'info', category: '对齐', title: '对齐 padding 浪费' },
    ADDR_CONFLICT: { severity: 'danger', category: '复用', title: '地址重叠且生命周期重叠' },
    REUSE_MISSED: { severity: 'info', category: '复用', title: '生命周期不重叠但未复用' },
    DOUBLE_BUFFER_INEFFECTIVE: { severity: 'warn', category: '流水', title: 'double buffer 未生效' },
    PIPE_STALL: { severity: 'warn', category: '流水', title: '流水等待空洞占比过高' },
    TAIL_BLOCK_WASTE: { severity: 'warn', category: '尾块', title: '尾块浪费' },
    GM_UNUSED: { severity: 'info', category: '容量', title: 'workspace 全程未被访问' },
    REG_SPILL: { severity: 'danger', category: '寄存器', title: '向量寄存器溢出' },
    REG_OCCUPANCY: { severity: 'warn', category: '寄存器', title: '寄存器用量压低 warp 并发' },
    REG_HEADROOM: { severity: 'info', category: '寄存器', title: '寄存器余量可换更大展开' },
  };

  let seq = 0;
  function finding(rule, over) {
    const meta = RULE_META[rule];
    return {
      id: `f${seq += 1}`,
      rule,
      severity: meta.severity,
      category: meta.category,
      title: meta.title,
      refs: [],
      eventRefs: [],
      evidence: [],
      ...over,
    };
  }

  // ---------------------------------------------------------------
  // 容量类
  // ---------------------------------------------------------------
  function capacityRules(run, metrics, th, out) {
    const underutilized = [];

    metrics.regions.forEach((region) => {
      // 寄存器层级不吃这套通用处方（"复用地址"/"增大 tileM" 对寄存器堆没有意义），
      // 它们由下面的 registerRules 单独判定。
      if (region.isRegister) return;

      // 预留 vs 实际持有 —— 超限时这条同样有价值（复用就是解超限的手段之一）
      if (region.scope === 'core' && region.idleReserved > 0
        && region.idleReserved / Math.max(1, region.reserved) > 0.2) {
        out.push(finding('RESERVED_IDLE', {
          region: region.id,
          detail: `${region.label} 静态预留 ${F.bytes(region.reserved)}，但峰值实际持有仅 ${F.bytes(region.peakLive)}（${F.tick(region.peakTick)}）。`,
          impact: `有 ${F.bytes(region.idleReserved)} 在整个 kernel 里从未同时被使用。`,
          suggest: '检查是否存在生命周期不重叠却各占一块地址的张量，合并后可腾出容量。',
          refs: region.topAtPeak.map((item) => item.alloc.id),
          evidence: [
            { label: '预留', value: F.bytes(region.reserved) },
            { label: '峰值持有', value: `${F.bytes(region.peakLive)} @ ${F.tick(region.peakTick)}` },
          ],
        }));
      }

      const over = region.reserved - region.capacity;
      if (over > 0) {
        const top = region.allocations.slice().sort((a, b) => b.size - a.size).slice(0, 3);
        const tileM = run.tiling.tileM;
        // 敏感度：把 tileM 降一档能释放多少（与 tileM 线性相关的分配才计入）
        const scalable = region.allocations.filter((a) => Array.isArray(a.shape) && a.shape[0] === tileM);
        const perRow = scalable.reduce((sum, a) => sum + a.size / tileM, 0);
        // 建议值向下取到 8 的倍数，贴合实际可用的切分粒度
        const rowsToCut = perRow ? Math.ceil(over / perRow / 8) * 8 : 0;
        out.push(finding('REGION_OVERFLOW', {
          region: region.id,
          detail: `${region.label} 静态预留 ${F.bytes(region.reserved)}，超出容量 ${F.bytes(region.capacity)}。`,
          impact: `超出 ${F.bytes(over)}（${F.pct(over / region.capacity, 1)}），编译期即报 buffer size exceeds limit。`,
          suggest: rowsToCut
            ? `tileM 从 ${tileM} 降到 ${tileM - rowsToCut} 可释放约 ${F.bytes(rowsToCut * perRow)}；或把 ${top[0].queue} 的 buffer_num 降为 1。`
            : `减小与 tileM 相关的队列深度或切分粒度。`,
          refs: top.map((a) => a.id),
          evidence: [
            { label: '预留', value: F.bytes(region.reserved) },
            { label: '容量', value: F.bytes(region.capacity) },
            ...top.map((a) => ({ label: a.name, value: F.bytes(a.size) })),
          ],
        }));
        return;
      }

      if (region.reservedRatio >= th.nearLimitRatio) {
        out.push(finding('REGION_NEAR_LIMIT', {
          region: region.id,
          detail: `${region.label} 已占用 ${F.pct(region.reservedRatio, 1)}（${F.bytes(region.reserved)} / ${F.bytes(region.capacity)}）。`,
          impact: `剩余 ${F.bytes(region.capacity - region.reserved)}，任何形状放大都会立即超限。`,
          suggest: '若后续要增大 tile 或加 double buffer，需先释放该层级空间。',
          refs: region.allocations.slice().sort((a, b) => b.size - a.size).slice(0, 2).map((a) => a.id),
          evidence: [
            { label: '预留', value: F.bytes(region.reserved) },
            { label: '余量', value: F.bytes(region.capacity - region.reserved) },
          ],
        }));
      } else if (region.scope === 'core' && region.reservedRatio < th.underutilizedRatio) {
        underutilized.push(region);
      }
    });

    // 多个层级同时闲置时只出一条 —— 它们的处方是同一个（增大 tileM），
    // 拆成多条只会稀释注意力。
    if (underutilized.length) {
      // 按闲置字节数而非比例挑代表：UB 闲 109KB 比 L0A 闲 56KB 更值得先说
      const worst = underutilized.slice()
        .sort((a, b) => (b.capacity - b.reserved) - (a.capacity - a.reserved))[0];
      const idle = underutilized.reduce((sum, r) => sum + (r.capacity - r.reserved), 0);
      out.push(finding('REGION_UNDERUTILIZED', {
        region: worst.id,
        detail: underutilized.length === 1
          ? `${worst.label} 只用到 ${F.pct(worst.reservedRatio, 1)}（${F.bytes(worst.reserved)} / ${F.bytes(worst.capacity)}）。`
          : `${underutilized.map((r) => `${r.id} ${F.pct(r.reservedRatio, 0)}`).join('、')} 均低于 ${F.pct(th.underutilizedRatio, 0)} 利用率。`,
        impact: `合计闲置 ${F.bytes(idle)}；切分过细让搬运次数上升、单次搬运效率下降。`,
        suggest: `可尝试把 tileM 从 ${run.tiling.tileM} 增大一档，用容量换搬运次数。`,
        refs: underutilized.flatMap((r) => r.allocations.slice().sort((a, b) => b.size - a.size).slice(0, 1).map((a) => a.id)),
        evidence: underutilized.map((r) => ({
          label: r.id, value: `${F.bytes(r.reserved)} / ${F.bytes(r.capacity)}`,
        })),
      }));
    }
  }

  // ---------------------------------------------------------------
  // 对齐类
  // ---------------------------------------------------------------
  function alignRules(run, metrics, th, out) {
    metrics.regions.forEach((region) => {
      const padded = region.allocations.filter((a) => a.size - a.dataBytes > 0);
      const total = padded.reduce((sum, a) => sum + (a.size - a.dataBytes), 0);
      if (total < th.paddingBytes) return;
      const worst = padded.slice().sort((x, y) => (y.size - y.dataBytes) - (x.size - x.dataBytes))[0];
      out.push(finding('ALIGN_PADDING', {
        region: region.id,
        detail: `${region.label} 因对齐产生 ${F.bytes(total)} padding，最大来源是 ${worst.name}${worst.padReason ? `（${worst.padReason}）` : ''}。`,
        impact: `${worst.name} 数据 ${F.bytes(worst.dataBytes)} 实占 ${F.bytes(worst.size)}，有效率 ${F.pct(worst.dataBytes / worst.size, 1)}。`,
        suggest: '把逐行标量类张量改为按 block 打包（一次算 8 行/32B），或用更紧凑的中间布局。',
        refs: padded.map((a) => a.id),
        evidence: padded.slice(0, 4).map((a) => ({
          label: a.name, value: `${F.bytes(a.dataBytes)} → ${F.bytes(a.size)}`,
        })),
      }));
    });
  }

  // ---------------------------------------------------------------
  // 复用类
  // ---------------------------------------------------------------
  function reuseRules(run, metrics, th, out) {
    metrics.regions.forEach((region) => {
      if (region.isRegister) return; // 寄存器分配由寄存器分配器决定，不给地址复用建议
      const list = region.allocations.filter((a) => a.intervals.length);
      const candidates = [];
      for (let i = 0; i < list.length; i += 1) {
        for (let j = i + 1; j < list.length; j += 1) {
          const a = list[i];
          const b = list[j];
          // 同一个队列的两个 slot 就是 ping-pong 本身，建议它们互相复用等于劝人关掉 double buffer
          if (a.declKey === b.declKey) continue;
          const sameSpace = MET.allocsOverlapInSpace(a, b);
          const sameTime = MET.allocsOverlapInTime(a, b);

          if (sameSpace && sameTime) {
            const overlapBytes = Math.min(a.offset + a.size, b.offset + b.size) - Math.max(a.offset, b.offset);
            out.push(finding('ADDR_CONFLICT', {
              region: region.id,
              detail: `${a.name} 与 ${b.name} 在 ${region.label} 上地址重叠 ${F.bytes(overlapBytes)}，且生命周期存在交叠。`,
              impact: '后写方会覆盖前者仍需读取的数据，表现为偶发精度异常，且不会被编译器拦截。',
              suggest: a.manualReuse || b.manualReuse
                ? `撤销 ${(a.manualReuse ? a : b).name} 的手工地址复用，或把复用点后移到读方 FreeTensor 之后。`
                : '为其中一方单独分配地址，或调整同步位置使两者生命周期分离。',
              refs: [a.id, b.id],
              evidence: [
                { label: a.name, value: `${F.hex(a.offset)} + ${F.bytes(a.size)}` },
                { label: b.name, value: `${F.hex(b.offset)} + ${F.bytes(b.size)}` },
                { label: '重叠区间', value: `${F.hex(Math.max(a.offset, b.offset))} … ${F.hex(Math.min(a.offset + a.size, b.offset + b.size))}` },
              ],
            }));
          } else if (!sameSpace && !sameTime && region.scope === 'core') {
            const saving = Math.min(a.size, b.size);
            if (saving < th.reuseSavingBytes) continue;
            candidates.push({ a, b, saving });
          }
        }
      }

      // 只报收益最大的几对，避免 O(n²) 的建议把真问题淹掉
      candidates.sort((x, y) => y.saving - x.saving).slice(0, 3).forEach(({ a, b, saving }) => {
        // 取首个迭代的相邻区间说明「一个刚释放、另一个才申请」
        const early = a.intervals[0].start <= b.intervals[0].start ? a : b;
        const late = early === a ? b : a;
        const freeAt = early.intervals[0].end;
        const allocAt = late.intervals[0].start;
        out.push(finding('REUSE_MISSED', {
          region: region.id,
          detail: `${early.name} 在 ${F.tick(freeAt)} 释放、${late.name} 到 ${F.tick(allocAt)} 才申请，两者在任何时刻都不同时存活，却各占一块地址。`,
          impact: `复用后可在 ${region.label} 释放 ${F.bytes(saving)}（占容量 ${F.pct(saving / region.capacity, 1)}）。`,
          suggest: `让二者共用同一段地址，或把 ${late.name} 改为从 ${early.queue} 同一块 TBuf 上 Get。`,
          refs: [a.id, b.id],
          evidence: [
            { label: `${early.name} 首次生命周期`, value: `${F.tick(early.intervals[0].start)}–${F.tick(freeAt)} · ${F.bytes(early.size)}` },
            { label: `${late.name} 首次生命周期`, value: `${F.tick(allocAt)}–${F.tick(late.intervals[0].end)} · ${F.bytes(late.size)}` },
            { label: '交叠区间数', value: '0' },
          ],
        }));
      });
    });

    run.allocations.filter((a) => a.unused).forEach((a) => {
      out.push(finding('GM_UNUSED', {
        region: a.region,
        detail: `${a.name} 预留 ${F.bytes(a.size)} workspace，但整个 kernel 没有任何访问事件。`,
        impact: `可直接从 workspace 预算中删除 ${F.bytes(a.size)}。`,
        suggest: a.note ? `${a.note}；确认融合后删除该 workspace 申请。` : '确认后删除该 workspace 申请。',
        refs: [a.id],
        evidence: [{ label: '访问事件数', value: '0' }],
      }));
    });
  }

  // ---------------------------------------------------------------
  // 流水类
  // ---------------------------------------------------------------
  function pipelineRules(run, metrics, th, out) {
    // double buffer：单份队列造成的实际阻塞
    const blocked = new Map();
    run.events.forEach((e) => {
      if (!e.blockedBy || e.gap <= 0) return;
      const alloc = run.allocations.find((a) => a.id === e.blockedBy);
      if (!alloc || alloc.bufferNum > 1) return;
      const entry = blocked.get(alloc.declKey) || { alloc, stall: 0, events: [] };
      entry.stall += e.gap;
      entry.events.push(e.id);
      blocked.set(alloc.declKey, entry);
    });

    blocked.forEach((entry) => {
      const region = metrics.regionById[entry.alloc.region];
      const extraCost = entry.alloc.size;
      const headroom = region.capacity - region.reserved;
      out.push(finding('DOUBLE_BUFFER_INEFFECTIVE', {
        region: entry.alloc.region,
        detail: `${entry.alloc.queue} 的 buffer_num = 1，下一次迭代必须等待上一次的消费者释放同一块地址。`,
        impact: `累计等待 ${entry.stall} cycle，占 kernel 总时长 ${F.pct(entry.stall / metrics.ticks, 1)}；搬运与计算无法重叠。`,
        suggest: headroom >= extraCost
          ? `把 ${entry.alloc.queue} 开成 double buffer，代价 +${F.bytes(extraCost)}，当前 ${region.label} 余量 ${F.bytes(headroom)}，可行。`
          : headroom >= 0
            ? `开 double buffer 需 +${F.bytes(extraCost)}，但 ${region.label} 仅余 ${F.bytes(headroom)}，需先降 tileM 或复用其他张量。`
            : `${region.label} 已超限 ${F.bytes(-headroom)}，必须先解容量问题，再谈 double buffer（还需 +${F.bytes(extraCost)}）。`,
        refs: [entry.alloc.id],
        eventRefs: entry.events,
        evidence: [
          { label: 'buffer_num', value: '1' },
          { label: '累计等待', value: `${entry.stall} cycle` },
          { label: '开双份代价', value: F.bytes(extraCost) },
        ],
      }));
    });

    metrics.pipes.forEach((pipe) => {
      if (!pipe.events.length || pipe.idleRatio < th.idleRatio) return;
      // 首个事件之前的等待是流水启动延迟，不是空洞
      const inner = pipe.events.slice(1);
      // 只报能归因到某个具体分配的等待。低占空比流水（MTE3 只搬一点数据）
      // 天然空闲，把它们报成问题就是噪声，也违背「结论可溯源」。
      const attributable = inner.filter((e) => e.gap > 0 && e.blockedBy);
      if (!attributable.length) return;
      const worst = attributable.slice().sort((a, b) => b.gap - a.gap)[0];
      const cause = run.allocations.find((a) => a.id === worst.blockedBy);
      const stall = attributable.reduce((sum, e) => sum + e.gap, 0);
      out.push(finding('PIPE_STALL', {
        region: cause ? cause.region : null,
        pipe: pipe.id,
        detail: `${pipe.label}（${pipe.desc}）有 ${attributable.length} 段等待可归因到 buffer 未释放，占本流水活跃区间 ${F.pct(stall / Math.max(1, pipe.span), 1)}。`,
        impact: `可归因空洞合计 ${stall} cycle，最大一段 ${worst.gap} cycle 出现在 ${worst.label} 之前（${F.tick(worst.t)}）。`,
        suggest: cause
          ? `该段等待由 ${cause.name} 的 slot 释放引起，优先处理 ${cause.queue} 的缓冲深度。`
          : '检查该流水线上游依赖与同步指令位置。',
        refs: cause ? [cause.id] : [],
        eventRefs: attributable.map((e) => e.id),
        evidence: [
          { label: '忙', value: `${pipe.busy} cycle` },
          { label: '可归因等待', value: `${stall} cycle` },
          { label: '最大空洞', value: `${worst.gap} cycle @ ${F.tick(worst.t)}` },
        ],
      }));
    });
  }

  // ---------------------------------------------------------------
  // 寄存器类（950 起：region.kind === 'register'）
  // ---------------------------------------------------------------
  function registerRules(run, metrics, th, out) {
    const plan = run.registers;
    if (!plan) return;
    const vrf = metrics.regionById[plan.vectorRegionId];
    const srf = metrics.regionById[plan.simtRegionId];

    // --- 向量寄存器溢出 ---
    if (plan.spillRegs > 0) {
      const spill = run.allocations.find((a) => a.isSpill);
      const spillEvents = run.events.filter((e) => e.type === 'spill');
      const spillCycles = spillEvents.reduce((sum, e) => sum + e.dur, 0);
      out.push(finding('REG_SPILL', {
        region: vrf.id,
        detail: `RegBase 循环体同时活跃 ${plan.requestedRegs} 个向量寄存器，${vrf.label} 只有 ${plan.capacityRegs} 个，超出 ${plan.spillRegs} 个。`,
        impact: `编译器把这 ${plan.spillRegs} 个寄存器（${F.bytes(plan.spillBytes)}）溢出到 ${plan.spillRegion}，每次迭代多一趟 store+load，累计 ${spillCycles} cycle（占 ${F.pct(spillCycles / Math.max(1, metrics.ticks), 1)}）。`,
        suggest: `把展开度从 ${plan.unroll} 降一档（tileM 减半即可），或减少同时活跃的中间量 —— 目标是把活跃寄存器压到 ${plan.capacityRegs} 个以内。`,
        refs: [
          ...run.allocations.filter((a) => a.region === vrf.id).slice(-2).map((a) => a.id),
          ...(spill ? [spill.id] : []),
        ],
        eventRefs: spillEvents.map((e) => e.id),
        evidence: [
          { label: '活跃寄存器', value: `${plan.requestedRegs} / ${plan.capacityRegs}` },
          { label: '溢出去向', value: `${plan.spillRegion} · ${F.bytes(plan.spillBytes)}` },
          { label: '展开度', value: `${plan.unroll} 组` },
          { label: '溢出往返', value: `${spillEvents.length} 次 · ${spillCycles} cycle` },
        ],
      }));
    } else if (vrf.reservedRatio < 0.6) {
      out.push(finding('REG_HEADROOM', {
        region: vrf.id,
        detail: `${vrf.label} 只用到 ${plan.requestedRegs} / ${plan.capacityRegs} 个寄存器（${F.pct(vrf.reservedRatio, 0)}）。`,
        impact: `还有 ${plan.capacityRegs - plan.requestedRegs} 个寄存器闲置，当前展开度 ${plan.unroll} 组没有把 VF 喂满。`,
        suggest: '增大 tileM 或手工提高循环展开度，用寄存器余量换更少的 loadalign/storealign 往返。',
        refs: vrf.allocations.slice(0, 2).map((a) => a.id),
        evidence: [
          { label: '活跃寄存器', value: `${plan.requestedRegs} / ${plan.capacityRegs}` },
          { label: '展开度', value: `${plan.unroll} 组` },
        ],
      }));
    }

    // --- SIMT 侧：每线程寄存器用量 vs 并发 warp 数 ---
    if (plan.activeWarps < plan.warpsMax) {
      const lostWarps = plan.warpsMax - plan.activeWarps;
      // 反推：要多跑一个 warp，每线程寄存器数需要降到多少
      const targetPerThread = Math.floor(srf.capacity
        / ((plan.activeWarps + 1) * plan.threadsPerWarp * (srf.regBytes || 4)));
      out.push(finding('REG_OCCUPANCY', {
        region: srf.id,
        detail: `每线程用掉 ${plan.regsPerThread} 个寄存器，一个 warp 就要 ${F.bytes(plan.warpBytes)}，${srf.label} 只装得下 ${plan.activeWarps} 个 warp。`,
        impact: `并发 warp 从 ${plan.warpsMax} 降到 ${plan.activeWarps}（occupancy ${F.pct(plan.warpOccupancy, 0)}），少 ${lostWarps} 个 warp 用来掩盖访存延迟。`,
        suggest: `把每线程寄存器压到 ${targetPerThread} 个以内可多跑一个 warp；展开度 ${plan.unroll} 是当前主要来源。`,
        refs: srf.allocations.slice(0, 2).map((a) => a.id),
        evidence: [
          { label: '每线程寄存器', value: `${plan.regsPerThread}` },
          { label: '单 warp 占用', value: F.bytes(plan.warpBytes) },
          { label: '并发 warp', value: `${plan.activeWarps} / ${plan.warpsMax}` },
          { label: 'SRF 占用', value: `${F.bytes(srf.reserved)} / ${F.bytes(srf.capacity)}` },
        ],
      }));
    }
  }

  // ---------------------------------------------------------------
  // 尾块类
  // ---------------------------------------------------------------
  function tailRules(run, metrics, th, out) {
    const { tailWasteRatio, tailWasteRows } = metrics.totals;
    if (!run.tiling.hasTail || tailWasteRatio < th.tailWasteRatio) return;
    out.push(finding('TAIL_BLOCK_WASTE', {
      region: null,
      detail: `M=${run.kernel.shape.M} 按 tileM=${run.tiling.tileM} 切成 ${run.tiling.tileNum} 块，最后一块只有 ${run.tiling.tailM} 行有效。`,
      impact: `尾块空转 ${tailWasteRows} 行，占总计算量 ${F.pct(tailWasteRatio, 1)}。`,
      suggest: `换用能整除 ${run.kernel.shape.M} 的 tileM，或对尾块单独走一条小 tile 分支。`,
      refs: [],
      eventRefs: metrics.totals.tailEvent ? [metrics.totals.tailEvent.id] : [],
      evidence: [
        { label: '有效行', value: `${run.tiling.tailM} / ${run.tiling.tileM}` },
        { label: '浪费占比', value: F.pct(tailWasteRatio, 1) },
      ],
    }));
  }

  // ---------------------------------------------------------------
  const SEVERITY_ORDER = { danger: 0, warn: 1, info: 2 };

  function analyze(run, metrics, options = {}) {
    const th = { ...DEFAULT_THRESHOLDS, ...(options.thresholds || {}) };
    const disabled = new Set(options.disabledRules || []);
    const out = [];
    capacityRules(run, metrics, th, out);
    alignRules(run, metrics, th, out);
    reuseRules(run, metrics, th, out);
    pipelineRules(run, metrics, th, out);
    registerRules(run, metrics, th, out);
    tailRules(run, metrics, th, out);
    return out
      .filter((item) => !disabled.has(item.rule))
      .sort((a, b) => SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity]);
  }

  function summarize(findings) {
    return {
      danger: findings.filter((f) => f.severity === 'danger').length,
      warn: findings.filter((f) => f.severity === 'warn').length,
      info: findings.filter((f) => f.severity === 'info').length,
    };
  }

  global.MemVizDiagnostics = { analyze, summarize, RULE_META, DEFAULT_THRESHOLDS };
})(window);
