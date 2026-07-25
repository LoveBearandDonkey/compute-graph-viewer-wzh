/*
  Tiling 候选构建器 —— 昇腾算子内存可视化工具
  ------------------------------------------------------------------
  规划文档 §6 定义了工具的中间数据格式；本文件是该格式的**生成器**而非手写样例：
  给定 tiling 参数与各队列的 buffer_num，推导出静态地址布局 (allocations)
  与流水事件序列 (events)。这样 §3 场景 4「tiling 对比」的每个候选都由同一套
  规则推出，视图上的差异真实来自参数差异，而不是人工编的数字。

  片上调度用一个极简的顺序模型模拟：
    · 每条流水线（MTE1/MTE2/Cube/FixPipe/Vector/MTE3）是串行队列；
    · 一步的开始时刻 = max(本流水线空闲时刻, 所读分配的写完时刻, 目标 slot 的释放时刻)；
    · buffer_num = N 的队列有 N 个 slot，第 i 次迭代用 slot i%N。
  double buffer 是否真的生效、等待空洞出现在哪里，都由这个模型自然产生。

  数据等级：L2 / schema-generated / share-safe（构造算子，非真实产品数据）
*/
(function registerMemVizRuns(global) {
  'use strict';

  const SRC = global.MemVizKernelSource;

  // ---- 算子形状常量（与 kernel-source.js 中的 constexpr 一致）----
  const M_TOTAL = 200;
  const N = 256;
  const K = 512;
  const K0 = 128;

  // ---- 相对吞吐率（bytes/cycle 或 MAC/cycle），仅用于生成可比的相对时序 ----
  const RATE = {
    MTE2: 4096,
    MTE1: 2048,
    MTE3: 4096,
    Cube: 65536,
    FixPipe: 4096,
    VectorElem: 1024,
    VectorCast: 2048,
  };

  const UB_BLOCK = 32; // UB 每行按 32B block 对齐 —— 标量类张量的经典 padding 来源

  function alignUp(value, align) {
    if (!align || align <= 1) return value;
    return Math.ceil(value / align) * align;
  }

  function cyc(amount, rate) {
    return Math.max(1, Math.round(amount / rate));
  }

  /**
   * 声明表：把 kernel 里的 TQue / TBuf 声明翻译成与 tiling 参数相关的分配描述。
   * logicalSize = 数据真实字节；physicalSize = 硬件对齐后实际占用。
   */
  function declarations(tileM, db) {
    const decl = (over) => ({
      slots: 1, dtype: 'float32', kind: 'buf', persistent: false, ...over,
    });
    return [
      // ---- L1 ----
      decl({
        key: 'bL1', region: 'L1', queue: 'bL1Buf', dtype: 'float16',
        logical: K * N * 2, shape: [K, N], slots: 1, persistent: true,
        declNeedle: 'InitBuffer(bL1Buf', hotNeedle: 'DataCopy(bL1, wGm',
        cce: 'LD.global.b128 %l1_w,[%gm_w+0]  ; 256KB 权重常驻 L1',
      }),
      decl({
        key: 'aL1', region: 'L1', queue: 'aL1Que', kind: 'queue', dtype: 'float16',
        logical: tileM * K * 2, shape: [tileM, K], slots: db.aL1,
        declNeedle: 'InitBuffer(aL1Que', hotNeedle: 'DataCopy(aL1, xGm',
        cce: 'LD.global.b128 %l1_a,[%gm_x+off]\nSET_FLAG MTE2->MTE1',
      }),
      // ---- L0A / L0B ----
      decl({
        key: 'aL0A', region: 'L0A', queue: 'aL0AQue', kind: 'queue', dtype: 'float16',
        logical: tileM * K0 * 2, shape: [tileM, K0], slots: db.aL0A,
        declNeedle: 'InitBuffer(aL0AQue', hotNeedle: 'LoadData(aL0A, aL1',
        cce: 'LOAD2D %l0a,%l1_a,fractal=zZ\nSET_FLAG MTE1->M',
      }),
      decl({
        key: 'bL0B', region: 'L0B', queue: 'bL0BBuf', dtype: 'float16',
        logical: K0 * N * 2, shape: [K0, N], slots: 1, persistent: true,
        declNeedle: 'InitBuffer(bL0BBuf', hotNeedle: 'LoadData(bL0B, bL1',
        cce: 'LOAD2D %l0b,%l1_w,fractal=nZ',
      }),
      // ---- L0C ----
      decl({
        key: 'cL0C', region: 'L0C', queue: 'cL0CQue', kind: 'queue',
        logical: tileM * N * 4, shape: [tileM, N], slots: db.cL0C,
        declNeedle: 'InitBuffer(cL0CQue', hotNeedle: 'Mmad(cL0C',
        cce: 'MMAD %l0c,%l0a,%l0b,init=1',
      }),
      // ---- UB ----
      decl({
        key: 'gammaUb', region: 'UB', queue: 'gammaBuf',
        logical: N * 4, shape: [N], slots: 1, persistent: true,
        declNeedle: 'InitBuffer(gammaBuf', hotNeedle: 'DataCopy(g, gammaGm',
        cce: 'LD.global.b32 %ub_g,[%gm_gamma]',
      }),
      decl({
        key: 'betaUb', region: 'UB', queue: 'betaBuf',
        logical: N * 4, shape: [N], slots: 1, persistent: true,
        declNeedle: 'InitBuffer(betaBuf', hotNeedle: 'DataCopy(b, betaGm',
        cce: 'LD.global.b32 %ub_b,[%gm_beta]',
      }),
      decl({
        key: 'meanUb', region: 'UB', queue: 'meanBuf',
        logical: tileM * 4, physical: tileM * UB_BLOCK, shape: [tileM, 1], slots: 1,
        declNeedle: 'InitBuffer(meanBuf', hotNeedle: 'ReduceSum(mean',
        padReason: '[tileM,1] 每行独占一个 32B block，实际写入 4B',
        cce: 'VCADD %ub_mean,%ub_mm,rows=tileM',
      }),
      decl({
        key: 'rstdUb', region: 'UB', queue: 'rstdBuf',
        logical: tileM * 4, physical: tileM * UB_BLOCK, shape: [tileM, 1], slots: 1,
        declNeedle: 'InitBuffer(rstdBuf', hotNeedle: 'ReduceSum(rstd',
        padReason: '[tileM,1] 每行独占一个 32B block，实际写入 4B',
        cce: 'VCADD %ub_rstd,%ub_sq,rows=tileM',
      }),
      decl({
        key: 'mmOut', region: 'UB', queue: 'mmOutQue', kind: 'queue',
        logical: tileM * N * 4, shape: [tileM, N], slots: db.mmOut,
        declNeedle: 'InitBuffer(mmOutQue', hotNeedle: 'Fixpipe(mmOut, cL0C',
        cce: 'WAIT_FLAG M->V\nFIXPIPE %ub_mm,%l0c,f32',
      }),
      decl({
        key: 'tmpSq', region: 'UB', queue: 'tmpSqBuf',
        logical: tileM * N * 4, shape: [tileM, N], slots: 1,
        declNeedle: 'InitBuffer(tmpSqBuf', hotNeedle: 'Mul(tmpSq, mmOut',
        cce: 'VMUL %ub_sq,%ub_mm,%ub_mm',
      }),
      decl({
        key: 'normUb', region: 'UB', queue: 'normBuf',
        logical: tileM * N * 4, shape: [tileM, N], slots: 1,
        declNeedle: 'InitBuffer(normBuf', hotNeedle: 'Sub(norm, mmOut, mean',
        cce: 'VSUB %ub_n,%ub_mm,%ub_mean\nVMUL %ub_n,%ub_n,%ub_rstd\nVMUL %ub_n,%ub_n,%ub_g\nVADD %ub_n,%ub_n,%ub_b',
      }),
      decl({
        key: 'yUb', region: 'UB', queue: 'yQue', kind: 'queue', dtype: 'float16',
        logical: tileM * N * 2, shape: [tileM, N], slots: db.yUb,
        declNeedle: 'InitBuffer(yQue', hotNeedle: 'Cast(y, norm',
        cce: 'VCONV %ub_y,%ub_n,f32->f16\nSET_FLAG V->MTE3',
      }),
    ];
  }

  /** GM 侧分配 —— 服务规划文档场景 6（workspace 与 GM 规划）。 */
  function gmDeclarations() {
    return [
      { key: 'xGm', name: 'xGm', logical: M_TOTAL * K * 2, dtype: 'float16', shape: [M_TOTAL, K], role: 'input' },
      { key: 'wGm', name: 'wGm', logical: K * N * 2, dtype: 'float16', shape: [K, N], role: 'input' },
      { key: 'gammaGm', name: 'gammaGm', logical: N * 4, dtype: 'float32', shape: [N], role: 'input' },
      { key: 'betaGm', name: 'betaGm', logical: N * 4, dtype: 'float32', shape: [N], role: 'input' },
      { key: 'yGm', name: 'yGm', logical: M_TOTAL * N * 2, dtype: 'float16', shape: [M_TOTAL, N], role: 'output' },
      {
        key: 'mmWorkspaceGm', name: 'mmWorkspaceGm', logical: M_TOTAL * N * 4,
        dtype: 'float32', shape: [M_TOTAL, N], role: 'workspace',
        note: '融合前 matmul 中间结果落盘用；融合后全程未被访问',
        unused: true,
      },
    ];
  }

  // ---------------------------------------------------------------
  // 构建
  // ---------------------------------------------------------------
  function buildRun(cfg, chip) {
    const tileM = cfg.tileM;
    const tileNum = Math.ceil(M_TOTAL / tileM);
    const tailM = M_TOTAL - (tileNum - 1) * tileM;
    const hasTail = tailM !== tileM;

    const regionOf = (id) => chip.regions.find((r) => r.id === id);
    const allocations = [];
    const byId = new Map();
    const slotIds = new Map(); // declKey -> [allocId per slot]
    const cursors = {};

    // ---- 静态地址布局：按声明顺序在各 region 内顺序摆放 ----
    declarations(tileM, cfg.db).forEach((d) => {
      const region = regionOf(d.region);
      const align = region.align;
      const physical = alignUp(d.physical || d.logical, align);
      const ids = [];
      for (let slot = 0; slot < d.slots; slot += 1) {
        const id = `${d.key}#${slot}`;
        let offset;
        const reuseTarget = cfg.manualReuse && cfg.manualReuse[d.key];
        if (reuseTarget && slot === 0) {
          // 手工复用：直接落在目标分配的起始地址上，不推进游标（省容量、担风险）
          offset = byId.get(`${reuseTarget}#0`).offset;
        } else {
          cursors[d.region] = alignUp(cursors[d.region] || 0, align);
          offset = cursors[d.region];
          cursors[d.region] = offset + physical;
        }
        const declLine = SRC.lineOf(d.declNeedle);
        const hotLine = SRC.lineOf(d.hotNeedle);
        const alloc = {
          id,
          declKey: d.key,
          name: d.slots > 1 ? `${d.key}[${slot}]` : d.key,
          region: d.region,
          queue: d.queue,
          kind: d.kind,
          slot,
          bufferNum: d.slots,
          offset,
          size: physical,
          logicalSize: d.physical || d.logical,
          dataBytes: d.logical,
          align,
          dtype: d.dtype,
          shape: d.shape,
          persistent: !!d.persistent,
          padReason: d.padReason || null,
          reuseOf: reuseTarget && slot === 0 ? `${reuseTarget}#0` : null,
          manualReuse: !!(reuseTarget && slot === 0),
          src: { file: SRC.path, declLine, hotLine },
          code: SRC.snippet(Math.max(0, hotLine - 1), hotLine + 2),
          cce: d.cce,
          intervals: [],
        };
        allocations.push(alloc);
        byId.set(id, alloc);
        ids.push(id);
      }
      slotIds.set(d.key, ids);
    });

    // ---- GM 侧 ----
    const gmRegion = regionOf('GM');
    let gmCursor = 0;
    gmDeclarations().forEach((d) => {
      const physical = alignUp(d.logical, gmRegion.align);
      const offset = alignUp(gmCursor, gmRegion.align);
      gmCursor = offset + physical;
      const alloc = {
        id: `${d.key}#0`,
        declKey: d.key,
        name: d.name,
        region: 'GM',
        queue: 'GlobalTensor',
        kind: 'gm',
        slot: 0,
        bufferNum: 1,
        offset,
        size: physical,
        logicalSize: d.logical,
        dataBytes: d.logical,
        align: gmRegion.align,
        dtype: d.dtype,
        shape: d.shape,
        persistent: true,
        role: d.role,
        note: d.note || null,
        unused: !!d.unused,
        padReason: null,
        reuseOf: null,
        manualReuse: false,
        src: { file: SRC.path, declLine: SRC.lineOf('SetGlobalBuffer'), hotLine: SRC.lineOf('SetGlobalBuffer') },
        code: SRC.snippet(SRC.lineOf('xGm.SetGlobalBuffer'), SRC.lineOf('yGm.SetGlobalBuffer')),
        cce: `// ${d.name} @GM role=${d.role}`,
        intervals: [],
      };
      allocations.push(alloc);
      byId.set(alloc.id, alloc);
      slotIds.set(d.key, [alloc.id]);
    });

    // ---- 流水调度模拟 ----
    const events = [];
    const pipeCursor = {};
    const slotFreeAt = {};
    const writeDoneAt = {};
    const openInterval = {};
    let seq = 0;

    const first = (key) => slotIds.get(key)[0];
    const slotOf = (key, iter) => {
      const ids = slotIds.get(key);
      return ids[iter % ids.length];
    };

    function emit(step) {
      const pipe = step.pipe;
      const before = pipeCursor[pipe] || 0;
      let start = before;
      (step.reads || []).forEach((id) => { start = Math.max(start, writeDoneAt[id] || 0); });
      (step.writes || []).forEach((id) => { start = Math.max(start, slotFreeAt[id] || 0); });
      const end = start + step.dur;
      pipeCursor[pipe] = end;

      (step.writes || []).forEach((id) => {
        const interval = { start, end, iter: step.iter };
        byId.get(id).intervals.push(interval);
        openInterval[id] = interval;
        writeDoneAt[id] = end;
      });
      (step.reads || []).forEach((id) => {
        if (openInterval[id]) openInterval[id].end = Math.max(openInterval[id].end, end);
      });

      events.push({
        id: `e${seq += 1}`,
        t: start,
        dur: step.dur,
        end,
        gap: start - before,
        pipe,
        type: step.type,
        label: step.label,
        iter: step.iter,
        reads: step.reads || [],
        writes: step.writes || [],
        bytes: step.bytes || 0,
        srcLine: step.srcLine != null ? step.srcLine : null,
        blockedBy: step.writes && step.writes.length && (slotFreeAt[step.writes[0]] || 0) > before
          ? step.writes[0] : null,
      });
      return end;
    }

    function release(id, at) {
      if (openInterval[id]) openInterval[id].end = Math.max(openInterval[id].end, at);
      slotFreeAt[id] = at;
      openInterval[id] = null;
    }

    // 前导：权重常驻搬运 + gamma/beta 载入
    emit({
      pipe: 'MTE2', type: 'copy_in', label: 'wGm→bL1', iter: -1,
      reads: [first('wGm')], writes: [first('bL1')],
      bytes: K * N * 2, dur: cyc(K * N * 2, RATE.MTE2),
      srcLine: SRC.lineOf('DataCopy(bL1, wGm'),
    });
    emit({
      pipe: 'MTE1', type: 'load', label: 'bL1→bL0B', iter: -1,
      reads: [first('bL1')], writes: [first('bL0B')],
      bytes: K0 * N * 2, dur: cyc(K0 * N * 2, RATE.MTE1),
      srcLine: SRC.lineOf('LoadData(bL0B, bL1'),
    });
    emit({
      pipe: 'MTE2', type: 'copy_in', label: 'gamma/beta→UB', iter: -1,
      reads: [first('gammaGm'), first('betaGm')],
      writes: [first('gammaUb'), first('betaUb')],
      bytes: N * 8, dur: cyc(N * 8, RATE.MTE2),
      srcLine: SRC.lineOf('DataCopy(g, gammaGm'),
    });
    // bL1 首次也是唯一一次被读走后即可释放（工具应据此提示复用机会）
    release(first('bL1'), writeDoneAt[first('bL0B')]);

    for (let i = 0; i < tileNum; i += 1) {
      const rows = i === tileNum - 1 ? tailM : tileM;
      const sA = slotOf('aL1', i);
      const sL0A = slotOf('aL0A', i);
      const sC = slotOf('cL0C', i);
      const sMM = slotOf('mmOut', i);
      const sY = slotOf('yUb', i);
      const tag = `#${i}`;

      emit({
        pipe: 'MTE2', type: 'copy_in', label: `xGm→aL1 ${tag}`, iter: i,
        reads: [first('xGm')], writes: [sA],
        bytes: tileM * K * 2, dur: cyc(tileM * K * 2, RATE.MTE2),
        srcLine: SRC.lineOf('DataCopy(aL1, xGm'),
      });
      const l0aEnd = emit({
        pipe: 'MTE1', type: 'load', label: `aL1→L0A ${tag}`, iter: i,
        reads: [sA], writes: [sL0A],
        bytes: tileM * K0 * 2, dur: cyc(tileM * K0 * 2, RATE.MTE1),
        srcLine: SRC.lineOf('LoadData(aL0A, aL1'),
      });
      release(sA, l0aEnd);

      const mmadEnd = emit({
        pipe: 'Cube', type: 'compute', label: `MMAD ${tag}`, iter: i,
        reads: [sL0A, first('bL0B')], writes: [sC],
        bytes: 0, dur: cyc(tileM * K0 * N, RATE.Cube),
        srcLine: SRC.lineOf('Mmad(cL0C'),
      });
      release(sL0A, mmadEnd);

      const fixEnd = emit({
        pipe: 'FixPipe', type: 'move', label: `L0C→UB ${tag}`, iter: i,
        reads: [sC], writes: [sMM],
        bytes: tileM * N * 4, dur: cyc(tileM * N * 4, RATE.FixPipe),
        srcLine: SRC.lineOf('Fixpipe(mmOut, cL0C'),
      });
      release(sC, fixEnd);

      const reduceEnd = emit({
        pipe: 'Vector', type: 'compute', label: `ReduceMeanVar ${tag}`, iter: i,
        reads: [sMM], writes: [first('tmpSq'), first('meanUb'), first('rstdUb')],
        bytes: 0, dur: cyc(tileM * N, RATE.VectorElem),
        srcLine: SRC.lineOf('Mul(tmpSq, mmOut'),
      });
      // tmpSq 只在 ReduceMeanVar 内部被写与被读，出了这一步就是死变量
      release(first('tmpSq'), reduceEnd);

      const normEnd = emit({
        pipe: 'Vector', type: 'compute', label: `Normalize ${tag}`, iter: i,
        reads: [sMM, first('meanUb'), first('rstdUb'), first('gammaUb'), first('betaUb')],
        writes: [first('normUb')],
        bytes: 0, dur: cyc(tileM * N, RATE.VectorElem),
        srcLine: SRC.lineOf('Sub(norm, mmOut, mean'),
      });
      release(sMM, normEnd);

      const castEnd = emit({
        pipe: 'Vector', type: 'compute', label: `Cast ${tag}`, iter: i,
        reads: [first('normUb')], writes: [sY],
        bytes: 0, dur: cyc(tileM * N, RATE.VectorCast),
        srcLine: SRC.lineOf('Cast(y, norm'),
      });
      release(first('normUb'), castEnd);

      const outEnd = emit({
        pipe: 'MTE3', type: 'copy_out', label: `yUb→yGm ${tag}`, iter: i,
        reads: [sY], writes: [first('yGm')],
        bytes: tileM * N * 2, dur: cyc(tileM * N * 2, RATE.MTE3),
        srcLine: SRC.lineOf('DataCopy(yGm[i'),
      });
      release(sY, outEnd);

      // 尾块：最后一次迭代只有 rows 行有效，其余为无效算力/带宽
      if (rows !== tileM) {
        events[events.length - 1].tailRows = rows;
      }
    }

    const totalTicks = Math.max(...Object.values(pipeCursor));

    // 未关闭的 interval（常驻张量）延伸到 kernel 结束
    allocations.forEach((alloc) => {
      alloc.intervals.forEach((interval) => {
        if (interval.end < interval.start) interval.end = interval.start;
      });
      if (alloc.persistent && alloc.intervals.length && alloc.region !== 'GM') {
        const last = alloc.intervals[alloc.intervals.length - 1];
        if (openInterval[alloc.id]) last.end = totalTicks;
      }
      if (alloc.region === 'GM' && !alloc.unused) {
        alloc.intervals = [{ start: 0, end: totalTicks, iter: -1 }];
      }
    });

    return {
      schemaVersion: '0.1',
      id: cfg.id,
      label: cfg.label,
      kicker: cfg.kicker,
      note: cfg.note,
      chip: { name: chip.name, specRef: chip.specRef },
      kernel: {
        name: 'MatmulLayerNorm_mix',
        source: SRC.path,
        blockDim: 8,
        shape: { M: M_TOTAL, N, K, K0 },
      },
      tiling: {
        tileM, tileNum, tailM, hasTail,
        bufferNum: { ...cfg.db },
      },
      totalTicks,
      pipeCursor: { ...pipeCursor },
      allocations,
      events,
    };
  }

  // ---------------------------------------------------------------
  // 候选集合 —— 每个候选只承载一个需要被看见的问题，不把所有毛病堆在一次运行里
  // ---------------------------------------------------------------
  const CONFIGS = [
    {
      id: 't64',
      label: 'tileM=64',
      kicker: '大切分',
      note: '搬运次数最少，但 UB 静态预留超出容量，编译期即报 buffer 超限。',
      tileM: 64,
      db: { aL1: 2, aL0A: 2, cL0C: 2, mmOut: 1, yUb: 1 },
    },
    {
      id: 't32',
      label: 'tileM=32',
      kicker: '当前基线',
      note: 'UB 不超限，但 mmOutQue / yQue 都是单份，搬运与计算无法重叠。',
      tileM: 32,
      db: { aL1: 2, aL0A: 2, cL0C: 2, mmOut: 1, yUb: 1 },
    },
    {
      id: 't32db',
      label: 'tileM=32 + double buffer',
      kicker: '候选解',
      note: 'mmOutQue / yQue 开双份，用 UB 余量换取搬运与计算重叠。',
      tileM: 32,
      db: { aL1: 2, aL0A: 2, cL0C: 2, mmOut: 2, yUb: 2 },
    },
    {
      id: 't32reuse',
      label: 'tileM=32 + 手工复用',
      kicker: '风险写法',
      note: 'normBuf 手工复用 mmOutQue 首个 slot 的地址省下 32KB，但 Normalize 仍在读 mmOut。',
      tileM: 32,
      db: { aL1: 2, aL0A: 2, cL0C: 2, mmOut: 2, yUb: 2 },
      manualReuse: { normUb: 'mmOut' },
    },
    {
      id: 't16',
      label: 'tileM=16',
      kicker: '小切分',
      note: '尾块浪费最小，但 UB 大量闲置，搬运次数翻数倍。',
      tileM: 16,
      db: { aL1: 2, aL0A: 2, cL0C: 2, mmOut: 2, yUb: 2 },
    },
  ];

  function buildAll(chip) {
    return CONFIGS.map((cfg) => buildRun(cfg, chip));
  }

  global.MemVizRuns = { buildAll, buildRun, CONFIGS, M_TOTAL, N, K, K0 };
})(window);
