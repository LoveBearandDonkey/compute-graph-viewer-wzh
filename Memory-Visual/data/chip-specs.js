/*
  芯片描述文件 —— 昇腾算子内存可视化工具
  ------------------------------------------------------------------
  规划文档 §2 / §5 要求：存储层级的容量、bank 数、对齐粒度、流水单元集合
  必须是**可配置的芯片描述**，不得硬编码在渲染逻辑里。本文件即该配置层，
  新增型号只需在此追加一条 preset，视图与规则引擎均按 region/pipe 表驱动。

  数据等级：L2 / 占位示例
    容量与对齐数值为规格量级的占位值，仅用于演示工具形态；
    实际接入时以对应芯片型号官方规格为准（见 spec_ref 字段）。

  寄存器层级（950 起）：
    region.kind === 'register' 的层级不是 buffer，而是寄存器堆。
    它们同样走 region 表驱动（有容量、有对齐、会超限），但计量单位是
    「寄存器个数」而不是地址偏移，因此额外声明 regBytes / regCount，
    并由 chip.registers 描述 warp 切分与溢出去向，供占用率与 spill 规则使用。
    950 的 A5 RegBase 写法（loadalign → VF 计算 → storealign）把中间量
    从 UB 搬进寄存器，UB 压力下降、寄存器压力上升 —— 这正是要被看见的取舍。

  颜色：region.accent 属 data-viz exemption（编码存储层级身份，非 UI 语义色），
  取值沿用 patterns/swimlane-task 的 lane-kind 家族色，保证与泳道视图同源。
*/
(function registerMemVizChips(global) {
  'use strict';

  const KB = 1024;

  const CHIPS = {
    'ascend-910b': {
      id: 'ascend-910b',
      name: 'Ascend 910B',
      coreName: 'AI Core (DaVinci)',
      specRef: 'chips/ascend-910b.yaml',
      placeholder: true,
      regions: [
        {
          id: 'UB', label: 'Unified Buffer', scope: 'core', owner: 'Vector',
          capacity: 192 * KB, align: 32, banks: 16, accent: '#4d70ba',
          note: 'Vector 计算输入输出主战场，最核心视图',
        },
        {
          id: 'L1', label: 'L1 Buffer', scope: 'core', owner: 'Core',
          capacity: 512 * KB, align: 512, banks: 8, accent: '#735bb4',
          note: 'Cube 输入暂存与大块中转',
        },
        {
          id: 'L0A', label: 'L0A', scope: 'core', owner: 'Cube',
          capacity: 64 * KB, align: 512, banks: 4, accent: '#4a9568',
          note: '左矩阵，分形(fractal)对齐',
        },
        {
          id: 'L0B', label: 'L0B', scope: 'core', owner: 'Cube',
          capacity: 64 * KB, align: 512, banks: 4, accent: '#3f8f7a',
          note: '右矩阵，分形(fractal)对齐',
        },
        {
          id: 'L0C', label: 'L0C', scope: 'core', owner: 'Cube',
          capacity: 128 * KB, align: 1024, banks: 4, accent: '#ba8053',
          note: '累加输出',
        },
        {
          id: 'GM', label: 'Global / Workspace', scope: 'chip', owner: 'Chip',
          capacity: 4096 * KB, align: 512, banks: 0, accent: '#8c847c',
          note: '算子输入输出与 workspace，容量为本算子预算而非物理 HBM',
        },
      ],
      pipes: [
        { id: 'MTE2', label: 'MTE2', desc: 'GM → 片上', kind: 'MTEIn' },
        { id: 'MTE1', label: 'MTE1', desc: 'L1 → L0', kind: 'MTEIn' },
        { id: 'Cube', label: 'Cube', desc: '矩阵计算', kind: 'aic' },
        { id: 'FixPipe', label: 'FixPipe', desc: 'L0C → UB/GM', kind: 'aic' },
        { id: 'Vector', label: 'Vector', desc: '向量计算', kind: 'aiv' },
        { id: 'MTE3', label: 'MTE3', desc: '片上 → GM', kind: 'MTEOut' },
      ],
    },

    'ascend-950b': {
      id: 'ascend-950b',
      name: 'Ascend 950B',
      coreName: 'AI Core (次世代)',
      specRef: 'chips/ascend-950b.yaml',
      placeholder: true,
      regions: [
        { id: 'UB', label: 'Unified Buffer', scope: 'core', owner: 'Vector', capacity: 256 * KB, align: 32, banks: 32, accent: '#4d70ba', note: 'SIMD/SIMT 共享' },
        { id: 'L1', label: 'L1 Buffer', scope: 'core', owner: 'Core', capacity: 1024 * KB, align: 512, banks: 16, accent: '#735bb4', note: '' },
        { id: 'L0A', label: 'L0A', scope: 'core', owner: 'Cube', capacity: 128 * KB, align: 512, banks: 4, accent: '#4a9568', note: '' },
        { id: 'L0B', label: 'L0B', scope: 'core', owner: 'Cube', capacity: 128 * KB, align: 512, banks: 4, accent: '#3f8f7a', note: '' },
        { id: 'L0C', label: 'L0C', scope: 'core', owner: 'Cube', capacity: 256 * KB, align: 1024, banks: 4, accent: '#ba8053', note: '' },
        {
          id: 'VRF', label: 'Vector Register File', scope: 'core', owner: 'SIMD / VF',
          kind: 'register', capacity: 64 * 256, align: 256, banks: 0, accent: '#c2557a',
          regBytes: 256, regCount: 64, lanes: 64,
          note: 'A5 RegBase 的计算面：loadalign 进寄存器、VF 算完再 storealign 回 UB',
        },
        {
          id: 'SRF', label: 'SIMT Register File', scope: 'core', owner: 'SIMT',
          kind: 'register', capacity: 64 * KB, align: 128, banks: 0, accent: '#a06fd0',
          regBytes: 4, regCount: 16384,
          note: '按 warp 静态切分；每线程寄存器用量直接决定能并发多少个 warp',
        },
        { id: 'GM', label: 'Global / Workspace', scope: 'chip', owner: 'Chip', capacity: 8192 * KB, align: 512, banks: 0, accent: '#8c847c', note: '' },
      ],
      // 寄存器分配模型：VF 上下文按寄存器个数计，SIMT 侧按 warp 整块切分。
      // spillRegion 是溢出去向 —— 寄存器放不下的活跃值会被编译器落回该层级。
      registers: {
        vectorRegionId: 'VRF',
        simtRegionId: 'SRF',
        threadsPerWarp: 32,
        warpsMax: 8,
        spillRegion: 'UB',
        spillCostPerReg: 2, // 一次溢出 = 一条 store + 一条 load
      },
      pipes: [
        { id: 'MTE2', label: 'MTE2', desc: 'GM → 片上', kind: 'MTEIn' },
        { id: 'MTE1', label: 'MTE1', desc: 'L1 → L0', kind: 'MTEIn' },
        { id: 'Cube', label: 'Cube', desc: '矩阵计算', kind: 'aic' },
        { id: 'FixPipe', label: 'FixPipe', desc: 'L0C → UB/GM', kind: 'aic' },
        { id: 'Vector', label: 'Vector', desc: '向量计算', kind: 'aiv' },
        { id: 'VF', label: 'VF', desc: '寄存器读写与计算', kind: 'aiv' },
        { id: 'MTE3', label: 'MTE3', desc: '片上 → GM', kind: 'MTEOut' },
      ],
    },
  };

  /** 该芯片是否有寄存器层级（950 起为真，910B 为假）。 */
  function hasRegisters(chip) {
    return Boolean(chip && chip.registers && chip.regions.some((r) => r.kind === 'register'));
  }

  function get(chipId) {
    return CHIPS[chipId] || CHIPS['ascend-910b'];
  }

  function region(chip, regionId) {
    return chip.regions.find((item) => item.id === regionId) || null;
  }

  function list() {
    return Object.values(CHIPS);
  }

  global.MemVizChips = { get, list, region, hasRegisters, CHIPS };
})(window);
