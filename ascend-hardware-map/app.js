(() => {
  'use strict';

  const N = {
    l2: '[data-mem950-node="rail:L2"]',
    gm: '[data-mem950-node="rail:GM"]',
    ub: '[data-aiv-node="buffer:UB"]',
    scalar: '[data-aiv-node="scalar:Scalar"]',
    vector: '[data-aiv-node="vector:Vector"]',
    simt: '[data-aiv-node="exec:SIMT"]',
    simd: '[data-aiv-node="exec:SIMD"]',
    cache: '[data-aiv-node="buffer:ND-DMA Cache"]',
    l1: '[data-aic-node="buffer:L1"]',
    l0a: '[data-aic-node="buffer:L0A"]',
    l0b: '[data-aic-node="buffer:L0B"]',
    l0c: '[data-aic-node="buffer:L0C"]',
    cube: '[data-aic-node="cube:CUBE"]',
    fp: '[data-aic-node="buffer:FP"]',
  };

  const CATEGORIES = [
    { id: 1, badge: '开发者侧', title: '架构代际声明', sub: '声明目标架构', scenario: 'B', arch: 'ascend910b', essence: '架构号是编译与注册声明，只说明“想跑哪一代”，不能单独证明实现可运行。', signals: ['CMAKE_ASC_ARCHITECTURES', '--npu-arch=dav-3510', '__NPU_ARCH__', 'AddConfig("ascend950")'], actions: ['核对 Device 与 Host 两侧的代际分支', '双架构分别编译、运行并对比精度与性能'], selectors: [], routes: [], related: ['vector', 'cube'] },
    { id: 2, badge: 'AIV 核', title: 'Membase → RegBase', sub: '编程对象变了', scenario: 'A', arch: 'ascend950b', essence: '950 AIV 把寄存器张量、谓词与地址寄存器变成一等编程对象，GM → UB → Reg 路径被显式化。', signals: ['RegTensor', 'MaskReg', 'AddrReg', 'LoadAlign / StoreAlign', 'LocalMemBar'], actions: ['把 UB 布局与寄存器压力一起评估', '为 910B 保留独立 Membase 实现'], selectors: [N.ub, N.vector, N.scalar], routes: ['l2-to-aiv1'], related: ['vector'] },
    { id: 3, badge: 'Vector 核', title: '纯 SIMD → SIMD + SIMT', sub: '执行模型变了', scenario: 'A', arch: 'ascend950b', essence: '950 在 Vector 内新增 SIMT 子系统，不替换 SIMD；离散访存、线程分歧和线程级原子有了原生表达。', signals: ['--enable-simt', 'Warp / ThreadBlock', 'asc_atomic_add', 'Gather / Scatter'], actions: ['仅在离散访存或线程语义明确时启用 SIMT', '分别验证 SIMD 与 SIMT 路径的性能边界'], selectors: [N.simt, N.simd, N.ub], routes: [], related: ['vector'] },
    { id: 4, badge: 'Cube 通路', title: '芯片内互连变了', sub: '物理数据通路重接', scenario: 'C', arch: 'ascend950b', essence: 'Cube 周围的物理连线发生变化：旧直通路径移除，同时增加 UB↔L1、L0C→UB、NDDMA 等新路径。', signals: ['删除 GM → L0A/L0B', '删除 L1 → GM', '新增 UB → L1', '新增 L0C → UB', 'NDDMA / SSBuf'], actions: ['按 950 真实通路重写搬运链', '寻找 C-V 直连带来的融合机会'], selectors: [N.gm, N.l1, N.l0a, N.l0b, N.l0c, N.fp, N.ub], routes: ['l2-to-aic', 'aic-to-aiv1'], related: ['cube', 'gemm-ar'] },
    { id: 5, badge: 'Cube ISA', title: '计算指令移除 / 新增', sub: 'Cube 电路重新规划', scenario: 'C', arch: 'ascend950b', essence: '950 移除部分 int4、结构化稀疏与边界绕回能力，并为 MX 低比特路径重新配置 Cube 电路。', signals: ['MmadWithSparse 移除', 'int4 Cube Matmul 移除', 'SetLoadDataBoundary 移除', 'MX 系列新增'], actions: ['替换已移除指令与隐含硬件假设', '低比特方案转向 FP8 / MX 并重做精度验证'], selectors: [N.cube, N.l0a, N.l0b, N.l0c], routes: [], related: ['cube'] },
    { id: 6, badge: '矩阵分形', title: 'Cube 喂数排布变了', sub: 'L0A 从 ZZ 改成 NZ', scenario: 'C', arch: 'ascend950b', essence: 'L0A 的矩阵 A 分形从 ZZ 改为 NZ；写死切分与地址计算的实现必须迁移。', signals: ['A / L0A: ZZ → NZ', 'B / L0B: ZN 不变', 'C / L0C: NZ 不变'], actions: ['定位写死 L0A 分形的地址公式', '按 950 分形重新生成 tiling 与搬运参数'], selectors: [N.l0a, N.l1, N.cube], routes: [], related: ['cube'] },
    { id: 7, badge: '低比特', title: 'HiF8 / FP8 / MX 系列新增', sub: '低比特成为核心路径', scenario: 'A', arch: 'ascend950b', essence: '新格式不仅改变 dtype，还联动 scale 布局、搬运、舍入、饱和与量化融合。', signals: ['HiF8', 'FP8 E5M2 / E4M3', 'MXFP4 / MXFP8', 'MicroScaling', 'Histograms'], actions: ['把 scale 张量纳入 tiling 主路径', '补齐端到端精度和饱和行为验证'], selectors: [N.cube, N.fp, N.l1], routes: ['l2-to-aic'], related: ['cube'] },
    { id: 8, badge: 'UB / SRAM', title: 'bank 拓扑变了', sub: 'UB SRAM 微架构', scenario: 'C', arch: 'ascend950b', essence: 'UB bank group、每组 bank 数与单 bank 容量变化，旧的错位地址经验不再可靠。', signals: ['bank group 16 → 8', '每组 3 → 2 banks', '单 bank 4KB → 16KB', 'UB 192KB → 256KB'], actions: ['删除写死容量与 bank 错位公式', '用 profiling 验证冲突与带宽'], selectors: [N.ub], routes: [], related: ['vector', 'tput-sync'] },
    { id: 9, badge: '语义', title: '浮点 / 同步 / 调试语义', sub: '数值与控制原语', scenario: 'B', arch: 'ascend950b', essence: '即使结构不变，Subnormal、核间同步和调试接口变化也可能造成结果或诊断方式不同。', signals: ['Subnormal 默认不支持', '核间 Mutex 新增', 'CheckLocalMemoryIA 移除'], actions: ['建立跨代数值回归基线', '将同步与调试接口纳入迁移检查表'], selectors: [N.vector, N.scalar], routes: [], related: ['vector', 'gemm-ar'] },
    { id: 10, badge: '通信', title: 'HCCL 软件通信 → CCU 硬化通信', sub: '集合通信专用引擎', scenario: 'A', arch: 'ascend950b', essence: 'HCCL 仍提供上层语义，但 950 把部分集合通信执行下沉至 CCU，需要联合观察硬件资源与通算重叠。', signals: ['CCU', 'ReduceScatter', 'AllGatherMatMul', 'Dispatch / Combine', 'CCU profiling'], actions: ['区分 HCCL 接口语义与 CCU 执行落点', '用 CCU profiling 验证通信、片上内存和 AI Core 协同'], selectors: [N.l2, N.scalar, N.cube, N.l0c], routes: [], related: ['ccu-collective', 'gemm-ar'] },
  ];

  const CATEGORY_CONTEXT = {
    1: { actor: '算子开发者', goal: '让编译器选择正确 ISA，并让运行时挂载对应实现', impact: '在 CMake、Device 宏和 Host 注册中显式声明架构号', boundary: '架构声明只表达“想运行在哪一代”，不能单独证明 kernel 实际可编译、可运行或性能等价。' },
    2: { actor: 'AIV 核硬件', goal: '让向量计算更细粒度可控并贴近裸 ISA', impact: '计算对象从 UB LocalTensor 延伸到 RegTensor，并新增谓词、地址寄存器和本地屏障', boundary: 'RegBase 不绕开 UB 直读 GM；迁移时必须把 GM → UB → Reg、UB 布局和寄存器压力一起评估。' },
    3: { actor: 'Vector 核硬件', goal: '原生支持离散访存、线程级并发与原子操作', impact: '在 Vector 单元内新增 SIMT 子系统，并通过编译开关显式启用', boundary: 'SIMT 是对 SIMD 的补充而非替代；连续向量计算仍应保留 SIMD 路径并分别验证性能边界。' },
    4: { actor: '芯片内互连', goal: '简化 Cube 数据流并减少中间结果绕路', impact: '删除旧直通通路，同时增加 UB→L1、L0C→UB、NDDMA、SSBuf 等路径', boundary: '这是物理通路变化，不是 API 重命名；旧搬运链必须按 950 实际连线重新组织。' },
    5: { actor: 'Cube 硬件电路', goal: '为 MX 低比特计算路径重新分配电路资源', impact: '移除 int4、4:2 稀疏和边界绕回能力，并强化 MX 相关路径', boundary: '被移除的硬件能力不能通过修改编译参数恢复，必须替换指令、格式或算法路径。' },
    6: { actor: 'Cube 取数接口', goal: '配合新数据通路提升矩阵喂数效率', impact: 'A 矩阵在 L0A 的分形由 ZZ 改为 NZ，L0B 的 ZN 与 L0C 的 NZ 保持不变', boundary: '变化只落在 L0A，但任何写死 L0A 切分和地址公式的实现都必须重新生成参数。' },
    7: { actor: 'Cube / Matmul 硬件与 ISA', goal: '支持 LLM 低比特训练、推理和量化融合', impact: 'HiF8、FP8、MX 与 MicroScaling 进入搬运、计算和 Tiling 主路径', boundary: '低比特迁移不是替换 dtype；scale 布局、舍入、饱和、搬运和端到端精度必须一起验证。' },
    8: { actor: 'UB 片上 SRAM', goal: '减少 bank 冲突并提高单 bank 容量', impact: 'bank group 16→8、每组 bank 3→2、单 bank 4KB→16KB、UB 192KB→256KB', boundary: '代码能编译不代表访问模式仍高效；旧代地址错位经验必须通过 profiling 重新验证。' },
    9: { actor: '硬件与软件运行时', goal: '演进数值支持、同步原语与调试接口', impact: 'Subnormal 默认支持关闭、核间 Mutex 新增、CheckLocalMemoryIA 移除', boundary: '这类变化不一定改变结构图，却可能改变数值结果、同步行为与问题定位方式。' },
    10: { actor: 'CCU 集合通信引擎', goal: '降低通信对内存带宽和 AI Core 的占用', impact: 'HCCL 保留上层语义，部分集合通信执行下沉到 CCU 硬化资源', boundary: '仅调用 HCCL API 不等于已使用好 950 通信能力；还要观察 CCU 资源、片上内存压力和通算重叠。' },
  };

  const CATEGORY_COMPARISONS = {
    1: {
      head: ['信号', '出现处', '含义', '指向'],
      rows: [
        ['CMAKE_ASC_ARCHITECTURES=dav-2201, dav-3510', 'CMake', '同一算子双架构编译', '可能 B'],
        ['--npu-arch=dav-3510 固定', 'CMake / 编译参数', '只编 950 二进制', '偏 A'],
        ['--npu-arch=dav-2201 固定', 'CMake', '只编 A2/A3 二进制', '偏 C（待迁移）'],
        ['__NPU_ARCH__ == 2201 / 3510', 'Device 代码 #if', 'Device 侧代际隔离宏', '已有分支：检查各分支'],
        ['SocVersion::Ascend910B / Ascend950PR_9599', 'Host tiling 代码', 'Host 侧代际隔离', '同上'],
        ['AddConfig("ascend910b")', 'op_host/*_def.cpp', '只注册旧代 SoC', '偏 C'],
        ['AddConfig("ascend950", regbaseCfg)', '同上', '注册 950 实现且带 RegBase 配置', '偏 A'],
        ['同时注册 910b + 950', '同上', '双代际实现并存', '多代际共维护'],
      ],
    },
    2: {
      head: ['指令对象 / 关键词', '它是什么', '2201 上', '3510 上', '提示'],
      rows: [
        ['RegBase', '编程模型总称，不是单个对象', '不存在', '950 AIV 新模型', 'A'],
        ['RegTensor', '矢量寄存器张量', '无', '计算第一类对象', 'A'],
        ['MaskReg', '谓词寄存器', '无（用 selector 间接做）', '显式独立寄存器', 'A'],
        ['AddrReg', '地址寄存器', '无（地址藏在 LocalTensor）', '显式对象', 'A'],
        ['GM → UB → Reg', '显式寄存器路径', '隐式 UB 对象为主', '显式 Load / Store 是入口', 'A'],
        ['LoadAlign / StoreAlign', 'UB ↔ RegTensor 对齐搬运', '无', '寄存器编程出入口', 'A'],
        ['LocalMemBar', '本地内存屏障', 'set_flag / wait_flag', '新屏障', 'A'],
        ['asc_add（寄存器级）', '寄存器算子', '无', '作用在 RegTensor', 'A'],
      ],
      note: 'RegBase 不是绕开 UB 直读 GM，而是显式 GM → UB → Reg；Tiling、UB 布局和寄存器压力要一起看。',
    },
    3: {
      head: ['关键词', '它是什么', '2201 上', '3510 上', '提示'],
      rows: [
        ['--enable-simt', 'SIMT 编译开关', '不存在', '必须显式打开', 'A'],
        ['Warp / ThreadBlock', '线程束 / 线程块', '无', 'SIMT 并发单位', 'A'],
        ['asc_atomic_add', '线程级原子加', '无', '直方图等场景', 'A'],
        ['Gather / Scatter / 直方图样例', '离散访存', 'SIMD 模拟', 'SIMT 自然表达', 'A'],
      ],
    },
    4: {
      head: ['关键词 / 通路', '它是什么', '2201', '3510', '提示'],
      rows: [
        ['LoadData', '数据搬运指令', '通用', '仍存在，但 GM → L0 直通已删', 'C'],
        ['Fixpipe', 'Cube 后处理流水', '通用', '仍存在；增加 L0C → UB', 'C'],
        ['删除 GM → L0A / L0B 直通', '旧快捷路径', '有', '没了，需 GM → L1 → L0', 'C'],
        ['删除 L1 → GM', 'L1 直接写回', '有', '没了，走 L0C / Fixpipe', 'C'],
        ['新增 UB → L1', '减少绕路', '无', '有', '优化机会'],
        ['新增 L0C → UB', '后处理少绕路', '无', '有', '优化机会'],
        ['LoadData + MicroScaling', '搬运附带 scale', '不支持', '新增', 'A'],
        ['NDDMA / loop mode / DN 分型', '高维 DMA', '受限', '新增', '优化机会'],
        ['SSBuf / DualDest / L12UB', 'C-V 同步共享通路', '无或弱', '融合重点', '优化机会'],
      ],
    },
    5: {
      head: ['关键词', '它是什么', '2201', '3510', '提示'],
      rows: [
        ['Mmad', '矩阵乘累加', '通用', '仍是主指令，但格式 / 分形变', 'C'],
        ['MmadWithSparse', '4:2 结构化稀疏', '支持', '不支持', 'C'],
        ['LoadDataWithSparse', '配套稀疏搬运', '支持', '不支持', 'C'],
        ['int4b_t Cube Matmul', '4bit 整数矩阵乘', '支持', 'Cube 不支持，要 Cast int8', 'C'],
        ['SetLoadDataBoundary', 'L1 边界绕回', '支持', '删除，要手工拆 Load', 'C'],
        ['CheckLocalMemoryIA', '调试 API', '支持', '删除', 'C'],
      ],
    },
    6: {
      head: ['矩阵', '2201 分形', '3510 分形', '影响'],
      rows: [
        ['A（L0A）', 'ZZ', 'NZ', '写死 L0A 切分 / 地址计算必改'],
        ['B（L0B）', 'ZN', 'ZN', '不变'],
        ['C（L0C）', 'NZ', 'NZ', '不变'],
      ],
      note: '非 L0A 切分场景可能兼容；L0A 切分场景一定要改。',
    },
    7: {
      head: ['关键词', '它是什么', '2201', '3510', '提示'],
      rows: [
        ['hifloat8_t / HiF8', '8bit 浮点', '无', '一等格式', 'A'],
        ['fp8_e5m2_t / fp8_e4m3fn_t', 'FP8 内置类型', '无或弱', '新增 / 增强', 'A'],
        ['MXFP4', '4bit 浮点 + 共享 scale', '无', '原生 Cube', 'A'],
        ['MXFP8', '8bit 浮点 + 共享 scale', '无', '原生', 'A'],
        ['FP8-MXFP4 / MXFP8-MXFP4 Hybrid', '混合精度 Matmul', '无', 'DeepSeek-V4 路径', 'A'],
        ['MicroScaling（LoadData）', '搬运时 scale 处理', '无', '新增', 'A'],
        ['scale cache / n-buffer / AntiQuant', '缓存 / 伪量化 / 反量化', '无', '性能关键点', '优化机会'],
        ['Histograms 指令', 'LightningIndexer Top-k 等', '无', '950 专用', 'A'],
      ],
    },
    8: {
      head: ['维度', '2201', '3510'],
      rows: [
        ['bank group 数', '16', '8'],
        ['每组 bank 数', '3', '2'],
        ['每 bank 容量', '4KB', '16KB'],
        ['单 UB 总容量', '16 × 3 × 4 = 192KB', '8 × 2 × 16 = 256KB'],
      ],
      note: '写死 UB 大小或 bank 错位策略的实现必须检查。',
    },
    9: {
      head: ['关键词', '类别', '2201', '3510', '提示'],
      rows: [
        ['Subnormal 浮点支持', '数值语义', '默认支持', '默认不支持，影响 Exp / Ln / Reciprocal / Sqrt / Rsqrt / Div', 'B/C'],
        ['Mutex（核间）', '同步原语', '无', '新增', '优化机会'],
        ['CheckLocalMemoryIA', '调试 API', '有', '删除', 'C'],
      ],
      note: 'Profiling 扩展到 Pipe、PC Sampling、SIMT 寄存器、片上带宽和 CCU profiling；数值与同步变化要通过工具链闭环确认。',
    },
    10: {
      head: ['关键词', '它是什么', '2201 / 910B', '3510 / 950', '提示'],
      rows: [
        ['HCCL', '集合通信库', '软件库语义为主', '仍是上层接口入口', 'B'],
        ['CCU', '集合通信专用引擎', '无专用硬化引擎', '新增', 'A'],
        ['CCU 专用引擎', '硬化执行路径', '软件 / 通用资源组织', '硬件通信节点', 'A'],
        ['LoopEngine / Loop Group / Memory Slice / channel', '资源约束', '非主要迁移项', '决定并发通信 / 通信域规模', '优化机会'],
        ['ReduceScatter', '典型集合通信', 'HCCL 软件调度视角', 'CCU 示例重点', 'A'],
        ['AllGatherMatMul', '通算融合', '通信常被当黑盒', '通信 / 计算协同排布', '优化机会'],
        ['Dispatch / Combine', 'MOE 通信', '软件编排为主', 'CCU + 片上内存协同', '优化机会'],
        ['CCU profiling', '内部打点 / 压力 / 协同诊断', '看 HCCL 总耗时', '纳入 950 调优闭环', 'A'],
      ],
      note: '只调用 HCCL 不等于用好 950；出现 CCU、ReduceScatter、AllGatherMatMul、Dispatch / Combine 或 CCU profiling 时，应按 950 原生通信分析。',
    },
  };

  const FLOWS = [
    { id: 'vector', title: 'AIV Vector 计算', short: 'GM/L2 → UB → Vector → UB → GM', confidence: 'verified', arch: 'ascend910b', summary: '单核 AIV 的标准 Vector 路径。数据经 MTE2 进入 UB，计算结果再由 MTE3 写回。', path: 'GM / L2 → MTE2 → UB → Vector → UB → MTE3 → GM / L2', selectors: [N.l2, N.ub, N.vector], routes: ['l2-to-aiv1', 'aiv1-to-l2'], meta: { 'PTO 语义': '普通 AIV tile 计算', 'payload 经过 UB': '是', '适用': 'elementwise / reduce / gather / cast' }, steps: [
      { label: '读 GM/L2', text: '源 tile 位于 GM，全局访问经 L2 层。', selectors: [N.gm, N.l2], routes: [] },
      { label: 'MTE2 入 UB', text: '把源 tile 搬入 AIV Unified Buffer。', selectors: [N.l2, N.ub], routes: ['l2-to-aiv1'] },
      { label: 'Vector 消费', text: 'Vector/SIMD 从 UB 读取并把结果留在 UB。', selectors: [N.ub, N.vector], routes: [] },
      { label: 'MTE3 写回', text: '结果从 UB 写回 GM/L2。', selectors: [N.ub, N.l2], routes: ['aiv1-to-l2'] },
    ] },
    { id: 'cube', title: 'AIC Cube / GEMM 计算', short: 'GM/L2 → L1 → L0A/B → Cube → L0C', confidence: 'verified', arch: 'ascend910b', summary: 'AIC 的矩阵主路径。数据进入 L1 与 L0，Cube 计算后由 L0C/FixPipe 承接结果。', path: 'GM / L2 → L1 → L0A / L0B → Cube → L0C → FixPipe', selectors: [N.l2, N.l1, N.l0a, N.l0b, N.cube, N.l0c, N.fp], routes: ['l2-to-aic'], meta: { 'PTO 语义': 'GEMM / Cube tile', 'payload 经过 UB': '否', '适用': 'GEMM / convolution / matmul' }, steps: [
      { label: 'MTE2 入 L1', text: 'GM/L2 数据先进入 AIC L1 Buffer。', selectors: [N.l2, N.l1], routes: ['l2-to-aic'] },
      { label: 'MTE1 入 L0A/B', text: 'A、B tile 分别进入 L0A 与 L0B。', selectors: [N.l1, N.l0a, N.l0b], routes: [] },
      { label: 'Cube 计算', text: 'Cube 消费 A/B tile，累加结果进入 L0C。', selectors: [N.l0a, N.l0b, N.cube, N.l0c], routes: [] },
      { label: 'FixPipe 承接', text: 'L0C 结果经 FixPipe 写回或随路处理。', selectors: [N.l0c, N.fp], routes: [] },
    ] },
    { id: 'tput-sync', title: 'TPUT / TGET 同步搬运', short: 'GM → UB staging → remote GM', confidence: 'verified', arch: 'ascend910b', summary: '同步点对点搬运由 AIV 控制，payload 经过本地 UB staging，API 返回前 AIV 等待完成。', path: 'AIV issue → GM/L2 → UB staging → remote GM → wait', selectors: [N.l2, N.ub, N.scalar], routes: ['l2-to-aiv1', 'aiv1-to-l2'], meta: { 'PTO API': 'TPUT / TGET', 'payload 经过 UB': '是', '适用': '小块、strided、低启动时延' }, steps: [
      { label: 'AIV 发起', text: 'AIV 决定 tile、双缓冲与 atomic 细节。', selectors: [N.scalar], routes: [] },
      { label: 'MTE2 staging', text: '本地或远端 tile 进入本地 UB。', selectors: [N.l2, N.ub], routes: ['l2-to-aiv1'] },
      { label: 'MTE3 远端写', text: 'UB staging 数据写向目标 GM。', selectors: [N.ub, N.l2], routes: ['aiv1-to-l2'] },
      { label: 'AIV 等待', text: '同步语义阻塞到搬运完成。', selectors: [N.scalar], routes: [] },
    ] },
    { id: 'sdma-async', title: 'TPUT_ASYNC / TGET_ASYNC → SDMA', short: 'descriptor 在 UB；payload GM ↔ GM', confidence: 'verified', arch: 'ascend910b', summary: 'AIV 在 UB scratch 写 descriptor，但 payload 不经过 UB，由 AI Core 外部 SDMA 异步搬运。', path: 'AIV descriptor → SDMA → remote GM → AsyncEvent wait/test', selectors: [N.ub, N.scalar, N.gm], routes: [], meta: { 'PTO API': 'TPUT_ASYNC / TGET_ASYNC', 'payload 经过 UB': '否', '适用': '大块连续数据、通信计算重叠' }, steps: [
      { label: '写 descriptor', text: 'UB scratch 保存描述符、队尾与完成标记。', selectors: [N.ub, N.scalar], routes: [] },
      { label: '提交 SDMA', text: 'AIV 提交后继续执行。', selectors: [N.scalar], routes: [] },
      { label: 'GM↔GM DMA', text: '外部 SDMA 搬运 payload。', selectors: [N.gm, N.l2], routes: [] },
      { label: 'wait/test', text: '稍后通过 AsyncEvent 收敛完成状态。', selectors: [N.scalar], routes: [] },
    ] },
    { id: 'urma-async', title: '950 URMA 异步远程内存', short: 'GM → UnifiedBus / URMA → remote GM', confidence: 'claim', arch: 'ascend950b', summary: 'AIV 提交远程访问后，payload 从本地 GM 进入 UnifiedBus / URMA 路径，并通过 wait/test 收敛完成状态。', path: 'AIV session → local GM → UnifiedBus / URMA → remote GM → wait/test', selectors: [N.scalar, N.l2, N.gm], routes: [], meta: { 'PTO API': 'TPUT_ASYNC / TGET_ASYNC', 'payload 经过 UB': '否' }, steps: [
      { label: 'AIV 建会话', text: '构建 session 并提交远程访问。', selectors: [N.scalar], routes: [] },
      { label: '进入互连面', text: 'payload 从 GM/L2 进入 950 IO 互连。', selectors: [N.gm, N.l2], routes: [] },
      { label: 'URMA 访问', text: '外部子系统承担远程内存语义。', selectors: [N.l2], routes: [] },
      { label: '完成语义', text: 'wait/test 或 Quiet 收敛可见性。', selectors: [N.scalar], routes: [] },
    ] },
    { id: 'ccu-collective', title: '950 CCU 集合通信卸载', short: 'AIV launch；CCU 完成搬运归约', confidence: 'claim', arch: 'ascend950b', summary: 'AIV 负责握手与控制，CCU 路径承担集合通信的数据搬运与规约，结果写回 GM 后向 AIV 返回完成状态。', path: 'AIC tile → AIV launch → CCU fetch/reduce → result GM → CCU_DONE', selectors: [N.cube, N.l0c, N.fp, N.scalar, N.l2], routes: [], meta: { 'PTO 语义': 'TGATHER / TSCATTER / TBROADCAST / TREDUCE', 'payload 经过 UB': '否', '适用': 'AllReduce / ReduceScatter / Broadcast' }, steps: [
      { label: 'AIC 产 tile', text: 'Cube 结果变成 CCU 可访问的数据。', selectors: [N.cube, N.l0c, N.fp], routes: [] },
      { label: 'AIV launch', text: 'AIV 通知外部 CCU 启动任务。', selectors: [N.scalar], routes: [] },
      { label: 'CCU fetch + reduce', text: 'CCU 子系统完成缓冲、搬运与规约。', selectors: [N.l2], routes: [] },
      { label: '写回 + DONE', text: '结果回到 GM，完成信号返回 AIV。', selectors: [N.gm, N.scalar], routes: [] },
    ] },
    { id: 'gemm-ar', title: 'GEMM + AllReduce 流水线', short: 'AIC 产 tile；AIV 通信；device barrier', confidence: 'inferred', arch: 'ascend950b', summary: 'AIC 不等整次 GEMM 结束；AIV 按 ready tile 发起通信，以 tile 粒度形成计算与通信重叠。', path: 'AIC tile → Ready Queue → AIV TTEST → TPUT AtomicAdd → barrier', selectors: [N.cube, N.l0c, N.scalar, N.ub, N.l2], routes: ['l2-to-aiv2', 'aiv2-to-l2'], meta: { 'PTO API': 'TTEST / TPUT AtomicAdd / TNOTIFY / TWAIT', '并行关系': 'AIC 生产，AIV 消费' }, steps: [
      { label: 'AIC 产 tile', text: 'Cube 完成一个可消费的 GEMM tile。', selectors: [N.cube, N.l0c], routes: [] },
      { label: 'Ready Queue', text: 'AIV 用 TTEST 非阻塞检查 tile。', selectors: [N.scalar], routes: [] },
      { label: 'TPUT AtomicAdd', text: 'AIV 把 tile 送向 owner rank。', selectors: [N.ub, N.l2], routes: ['l2-to-aiv2', 'aiv2-to-l2'] },
      { label: 'Device barrier', text: 'TNOTIFY/TWAIT 收敛阶段依赖。', selectors: [N.scalar], routes: [] },
    ] },
  ];

  const SCENARIOS = {
    A: {
      title: 'A5 原生能力兼容 A2/A3', tagline: '以 A5 原生能力为主，兼容旧代时维护独立实现分支',
      features: ['固定 --npu-arch=dav-3510，或启用 --enable-simt', 'README 仅声明支持 Ascend 950PR / 950DT', '出现 RegBase、RegTensor、MaskReg、AddrReg', '注册中出现 AddConfig("ascend950", regbaseCfg)', '使用 Histograms、HiF8、FP8、MXFP8、MXFP4 等低比特能力', '通信链路出现 CCU、ReduceScatter、AllGatherMatMul、Dispatch / Combine 或 CCU profiling'],
      actions: ['默认按 3510 / 950 专用实现看待', '兼容 2201 时另写实现分支，并分别注册 ascend910b 与 ascend950', '按代际拆分 tiling、workspace、scale 布局和量化格式', '联合验证 HCCL 语义、CCU 硬化资源与 AI Core 计算重叠'],
    },
    B: {
      title: '简单 SIMD 样例', tagline: '改架构声明后验证，通常不需要重写硬件路径',
      features: ['CMake 中 CMAKE_ASC_ARCHITECTURES 可声明 dav-2201, dav-3510', 'kernel 主要是 GM ↔ UB 搬运、逐元素计算或简单 reduce', '不写死矩阵分形、L0A/L0B/L1 复杂路径', '不使用 BuiltIn API 或内部 impl 接口'],
      actions: ['用 dav-2201 编译并在 2201 设备验证', '用 dav-3510 编译并在 3510 设备验证', '对照精度与性能，重点检查 Subnormal 和 UB bank 冲突'],
    },
    C: {
      title: 'A2/A3 迁移 A5', tagline: '旧代硬件假设不再成立，需要重写数据与计算路径',
      features: ['使用 L0A/L0B/L0C、LoadData、Mmad、Fixpipe', '使用 int4 Cube Matmul、4:2 结构化稀疏或 SetLoadDataBoundary', '假设 GM → L0 或 L1 → GM 通路存在', '写死 UB 大小、bank group、核数或分形布局'],
      actions: ['不能只改 CMAKE_ASC_ARCHITECTURES', '先逐项检查数据通路和矩阵分形', '重写或拆分搬运路径，例如 GM → L1 → L0', 'int4 路径改为 Vector Cast 到 int8，或采用 950 的 MX / FP8 方案', '重新调优 tiling 与 bank 规避策略'],
    },
  };

  const MIGRATION_SCENARIOS = [
    { key: 'A', title: 'A5 原生能力兼容 A2/A3', overview: 'A5 原生算子开发及跨代兼容', tag: 'A5 主导' },
    { key: 'B', title: '简单 SIMD 样例', overview: '旧架构算子经简单调整迁移至 A5', tag: '轻量迁移' },
    { key: 'C', title: 'A2/A3 迁移 A5', overview: '重写旧代硬件路径与实现假设', tag: '深度迁移' },
  ];

  const MIGRATION_MAP = {
    1: { primary: 'B', secondary: ['A', 'C'] },
    2: { primary: 'A', secondary: [] },
    3: { primary: 'A', secondary: [] },
    4: { primary: 'C', secondary: ['A'] },
    5: { primary: 'C', secondary: [] },
    6: { primary: 'C', secondary: [] },
    7: { primary: 'A', secondary: [] },
    8: { primary: 'C', secondary: [] },
    9: { primary: 'C', secondary: ['B'] },
    10: { primary: 'A', secondary: [] },
  };

  const DIFF_950_NEW = {
    selectors: [N.simt, N.fp],
    routes: ['aic-to-aiv1', 'aiv2-to-aic', 'l2-to-aiv2', 'l2-to-aiv2-dcache', 'aiv2-to-l2'],
  };

  const state = { mode: 'migration', arch: 'ascend950b', selectedId: null, activeStep: -1, playing: false, timer: null, overlay: null, viewport: null, playback: null, playbackHover: null, activation: null, hardwareResizeObserver: null, hardwareFitFrame: 0, diff: false, previewView: 'hardware', memoryReuseViewer: null, memoryReuseData: null };
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const escape = (value) => String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));

  function measureHardwareFrame() {
    const host = $('#hardwareGraph');
    const stage = host?.querySelector('[data-pto-mem-arch-stage]');
    const layout = stage?.querySelector('.pto-mem950__layout');
    const candidates = [host, stage, layout].filter(Boolean);
    return {
      width: Math.max(1, ...candidates.map((element) => Math.max(element.scrollWidth || 0, element.offsetWidth || 0))),
      height: Math.max(1, ...candidates.map((element) => Math.max(element.scrollHeight || 0, element.offsetHeight || 0))),
    };
  }

  function observeHardwareSize() {
    state.hardwareResizeObserver?.disconnect?.();
    if (typeof ResizeObserver !== 'function') return;
    state.hardwareResizeObserver = new ResizeObserver(() => scheduleHardwareFit());
    const stage = $('#hardwareGraph [data-pto-mem-arch-stage]');
    if (stage) state.hardwareResizeObserver.observe(stage);
    const viewport = $('#hardwareViewport');
    if (viewport) state.hardwareResizeObserver.observe(viewport);
  }

  function scheduleHardwareFit() {
    if (state.hardwareFitFrame) cancelAnimationFrame(state.hardwareFitFrame);
    state.hardwareFitFrame = requestAnimationFrame(() => {
      state.hardwareFitFrame = requestAnimationFrame(() => {
        state.hardwareFitFrame = 0;
        if (state.previewView !== 'hardware') return;
        const size = measureHardwareFrame();
        state.viewport?.setFrameSize?.(size.width, size.height);
        state.viewport?.fit?.();
        state.overlay?.update?.();
      });
    });
  }

  function scheduleMemoryFit() {
    requestAnimationFrame(() => requestAnimationFrame(() => {
      if (state.previewView !== 'memory') return;
      state.memoryReuseViewer?.resize?.();
    }));
  }

  function renderLists() {
    const migrationOverview = `
      <button class="entity-button tree-document-button migration-overview-button" type="button" role="treeitem" aria-level="1" data-migration-overview>
        <span class="entity-main"><span class="entity-title">迁移场景总览</span><span class="entity-sub">三类场景定义与全量变化项映射</span></span>
      </button>`;
    const scenarioTrees = MIGRATION_SCENARIOS.map((scenario) => {
      const items = CATEGORIES.filter((item) => item.scenario === scenario.key);
      const groupId = `migration-scenario-${scenario.key}`;
      return `
        <section class="migration-tree-group" data-tree-branch="${scenario.key}">
          <button class="migration-tree-toggle" type="button" role="treeitem" aria-level="1" aria-expanded="true" aria-controls="${groupId}" data-tree-toggle="${scenario.key}">
            <svg class="migration-tree-chevron" viewBox="0 0 24 24" aria-hidden="true"><path d="m6 9 6 6 6-6"></path></svg>
            <span class="migration-tree-toggle-title">迁移场景 ${scenario.key}：${escape(scenario.title)}</span>
          </button>
          <ul class="migration-tree-children" id="${groupId}" role="group">
            <li role="none">
              <button class="entity-button tree-document-button scenario-overview-button" type="button" role="treeitem" aria-level="2" data-scenario-overview="${scenario.key}">
                <span class="entity-index">01</span>
                <span class="entity-main"><span class="entity-title">场景概述</span><span class="entity-sub">${escape(scenario.overview)}</span></span>
              </button>
            </li>
            ${items.map((item, index) => `
              <li role="none"><button class="entity-button tree-document-button" type="button" role="treeitem" aria-level="2" data-category-id="${item.id}">
                <span class="entity-index">${String(index + 2).padStart(2, '0')}</span>
                <span class="entity-main"><span class="entity-title">${escape(item.title)}</span><span class="entity-sub">${escape(item.sub)}</span></span>
                <span class="mini-badge">${escape(item.badge)}</span>
              </button></li>`).join('')}
          </ul>
        </section>`;
    }).join('');
    $('#categoryTree').innerHTML = migrationOverview + scenarioTrees;
    $('#flowList').innerHTML = FLOWS.map((item, index) => `
      <li><button class="entity-button" type="button" data-flow-id="${item.id}">
        <span class="entity-index">${String(index + 1).padStart(2, '0')}</span>
        <span class="entity-main"><span class="entity-title">${escape(item.title)}</span><span class="entity-sub">${escape(item.short)}</span></span>
      </button></li>`).join('');
  }

  function renderHardwareArchitecture(host) {
    const rendered = window.PtoMemoryArchitecturePattern.renderArchitecture(host, state.arch);
    rendered?.stage?.querySelector('.pto-mem950__notes')?.remove();
    return rendered;
  }

  function renderHardware() {
    const host = $('#hardwareGraph');
    state.activation?.destroy?.();
    renderHardwareArchitecture(host);
    state.overlay?.destroy?.();
    state.overlay = window.PtoMemoryArchitecturePattern.createRouteOverlay(host, state.arch);
    window.PtoMemoryArchitecturePattern.attachHoverInteractions(host, state.arch);
    state.activation = window.PtoMemoryArchitecturePattern.attachNodeActivation(host, state.arch, {
      selector: '[data-aiv-node="buffer:UB"]',
      label: () => '打开 UB 内存可视化',
      onActivate: (_target, detail) => openMemoryReuse(detail),
    });
    observeHardwareSize();
    requestAnimationFrame(() => {
      state.overlay?.update?.();
      applyCurrentFocus();
      scheduleHardwareFit();
    });
  }

  function setArch(arch) {
    if (!arch || state.arch === arch) return;
    state.arch = arch;
    if (arch !== 'ascend950b') state.diff = false;
    $$('[data-arch-id]').forEach((button) => {
      const selected = button.dataset.archId === arch;
      button.classList.toggle('is-selected', selected);
      button.setAttribute('aria-pressed', String(selected));
    });
    syncGenerationCompare();
    renderHardware();
  }

  function focusHardware(selectors = [], routes = []) {
    const host = $('#hardwareGraph');
    if (state.arch === 'ascend950b' && state.diff) {
      selectors = [...new Set([...selectors, ...DIFF_950_NEW.selectors])];
      routes = [...new Set([...routes, ...DIFF_950_NEW.routes])];
    }
    window.PtoMemoryArchitecturePattern.clearPathFocus(host);
    if (selectors.length || routes.length) {
      window.PtoMemoryArchitecturePattern.setPathFocus(host, state.arch, { selectors, routes });
    }
    requestAnimationFrame(() => state.overlay?.update?.());
  }

  function applyCurrentFocus() {
    if (state.selectedId === null) return focusHardware();
    const item = state.mode === 'migration'
      ? CATEGORIES.find((entry) => entry.id === Number(state.selectedId))
      : FLOWS.find((entry) => entry.id === state.selectedId);
    if (!item) return focusHardware();
    if (state.mode === 'flow' && state.activeStep >= 0) {
      const step = item.steps[state.activeStep];
      return focusHardware(step.selectors || [], step.routes || []);
    }
    focusHardware(item.selectors || [], item.routes || []);
  }

  function section(title, content, help = '') {
    const helpId = title === '关键差异对照' ? 'comparison-legend-help' : 'migration-map-help';
    const helpClass = title === '关键差异对照' ? ' is-comparison' : '';
    return `<section class="inspector-section">
      <div class="inspector-section-heading">
        <h3>${escape(title)}</h3>
        ${help ? `<span class="inspector-help-wrap">
          <button class="inspector-help-button" type="button" aria-label="${escape(title)}说明" aria-describedby="${helpId}">?</button>
          <span class="inspector-help-tooltip${helpClass}" id="${helpId}" role="tooltip">${help}</span>
        </span>` : ''}
      </div>
      ${content}
    </section>`;
  }

  function renderDocumentHeader({ path, title, summary }) {
    const breadcrumbs = path.map((item) => `<span>${escape(item)}</span>`).join('<i aria-hidden="true">/</i>');
    return `<section class="inspector-section inspector-document-header">
      <div class="inspector-document-path" aria-label="文档路径">${breadcrumbs}</div>
      <h2>${escape(title)}</h2>
      <blockquote class="inspector-document-summary">${escape(summary)}</blockquote>
    </section>`;
  }

  function renderContextCard(content) {
    return `<section class="inspector-section inspector-context-section"><div class="inspector-card">${content}</div></section>`;
  }

  function renderInspector(content) {
    const body = $('#inspectorBody');
    body.innerHTML = content;
    body.scrollTop = 0;
  }

  function comparisonMarker(label) {
    const key = label.includes('优化机会') ? 'opportunity' : (
      label.includes('删除') || label.includes('不支持') ? 'deleted' : (
        label.includes('新增') ? 'new' : (
          label.includes('/') ? 'mixed' : (label.includes('B') ? 'b' : (label.includes('C') ? 'c' : 'a'))
        )
      )
    );
    return `<span class="comparison-marker comparison-marker--${escape(key)}">${escape(label)}</span>`;
  }

  function renderComparisonTable(comparison) {
    const lastColumn = comparison.head.length - 1;
    const hasHints = comparison.head[lastColumn] === '提示' || comparison.head[lastColumn] === '指向';
    return `
      <div class="comparison-table-wrap">
        <table class="comparison-table">
          <thead><tr>${comparison.head.map((head) => `<th scope="col">${escape(head)}</th>`).join('')}</tr></thead>
          <tbody>${comparison.rows.map((row) => `<tr>${row.map((cell, index) => {
            const hint = hasHints && index === lastColumn && /^(A|B|C|B\/C|偏 A|偏 B|偏 C|可能 B|优化机会)/.test(cell);
            return `<td>${hint ? comparisonMarker(cell) : escape(cell)}</td>`;
          }).join('')}</tr>`).join('')}</tbody>
        </table>
      </div>
      ${comparison.note ? `<p class="comparison-note">${escape(comparison.note)}</p>` : ''}
    `;
  }

  function renderComparisonLegendHelp() {
    return `
      <span>${comparisonMarker('A')}<b>A5 原生能力；兼容旧代需独立分支</b></span>
      <span>${comparisonMarker('B')}<b>简单 SIMD；改声明后验证</b></span>
      <span>${comparisonMarker('C')}<b>旧代硬件假设失效；需重写路径</b></span>
      <span>${comparisonMarker('B/C')}<b>轻量迁移风险与深度迁移风险并存</b></span>
      <span>${comparisonMarker('优化机会')}<b>新增能力，主动用</b></span>
      <span>${comparisonMarker('删除')}<b>2201 → 3510 移除</b></span>
      <span>${comparisonMarker('新增')}<b>3510 独有</b></span>
    `;
  }

  function renderCategoryInspector(item) {
    const context = CATEGORY_CONTEXT[item.id];
    const comparison = CATEGORY_COMPARISONS[item.id];
    const related = item.related.map((id) => {
      const flow = FLOWS.find((entry) => entry.id === id);
      return flow ? `<button class="btn btn-ghost related-flow" type="button" data-related-flow="${flow.id}">→ ${escape(flow.title)}</button>` : '';
    }).join('');
    const scenarioTitle = MIGRATION_SCENARIOS.find((scenario) => scenario.key === item.scenario)?.title || item.scenario;
    renderInspector(`<div class="inspector-content">
      ${renderDocumentHeader({
        path: ['A5 算子迁移', `迁移场景 ${item.scenario}：${scenarioTitle}`],
        title: item.title,
        summary: item.essence,
      })}
      ${renderContextCard(`<dl class="meta-grid">
        <div class="meta-row"><dt>变化来源</dt><dd>${escape(context.actor)}</dd></div>
        <div class="meta-row"><dt>设计目标</dt><dd>${escape(context.goal)}</dd></div>
        <div class="meta-row"><dt>直接影响</dt><dd>${escape(context.impact)}</dd></div>
        <div class="meta-row"><dt>标签</dt><dd><span class="mini-badge">${escape(item.badge)}</span></dd></div>
      </dl>`)}
      ${section('关键差异对照', renderComparisonTable(comparison), renderComparisonLegendHelp())}
      ${section('判断信号', `<div class="tag-row">${item.signals.map((signal) => `<span class="path-chip">${escape(signal)}</span>`).join('')}</div>`)}
      ${section('关键边界', `<div class="inspector-card"><small>${escape(context.boundary)}</small></div>`)}
      ${section('建议动作', `<ul class="action-list">${item.actions.map((action) => `<li>${escape(action)}</li>`).join('')}</ul>`)}
      ${section('关联执行流', related)}
    </div>`);
  }

  function renderFlowInspector(flow) {
    const visibleMeta = Object.entries(flow.meta);
    renderInspector(`<div class="inspector-content">
      ${renderDocumentHeader({ path: ['执行流对比'], title: flow.title, summary: flow.summary })}
      ${section('执行路径', `<div class="inspector-card"><strong>路径</strong><small>${escape(flow.path)}</small></div>`)}
      ${section('上下文', `<dl class="meta-grid">${visibleMeta.map(([key, value]) => `<div class="meta-row"><dt>${escape(key)}</dt><dd>${escape(value)}</dd></div>`).join('')}</dl>`)}
      ${section('执行步骤', `<ol class="step-list">${flow.steps.map((step, index) => `<li><button class="step-button${state.activeStep === index ? ' is-selected' : ''}" type="button" data-step="${index}"><span class="step-number">${String(index + 1).padStart(2, '0')}</span><span class="step-copy"><strong>${escape(step.label)}</strong><small>${escape(step.text)}</small></span></button></li>`).join('')}</ol>`)}
    </div>`);
    syncPlayback();
  }

  function selectCategory(id) {
    stopPlayback();
    state.selectedId = Number(id);
    state.activeStep = -1;
    const item = CATEGORIES.find((entry) => entry.id === state.selectedId);
    if (!item) return;
    $$('[data-category-id]').forEach((button) => button.classList.toggle('is-selected', Number(button.dataset.categoryId) === item.id));
    $$('[data-scenario-overview]').forEach((button) => button.classList.remove('is-selected'));
    $('[data-migration-overview]')?.classList.remove('is-selected');
    syncEntitySelectionAccessibility();
    renderCategoryInspector(item);
    if (state.arch !== item.arch) setArch(item.arch);
    else { applyCurrentFocus(); scheduleHardwareFit(); }
  }

  function selectFlow(id) {
    stopPlayback();
    state.selectedId = id;
    state.activeStep = -1;
    const item = FLOWS.find((entry) => entry.id === id);
    if (!item) return;
    $$('[data-flow-id]').forEach((button) => button.classList.toggle('is-selected', button.dataset.flowId === item.id));
    syncEntitySelectionAccessibility();
    renderFlowInspector(item);
    if (state.arch !== item.arch) setArch(item.arch);
    else { applyCurrentFocus(); scheduleHardwareFit(); }
  }

  function selectStep(index) {
    const flow = FLOWS.find((entry) => entry.id === state.selectedId);
    if (!flow) return;
    state.activeStep = Math.max(0, Math.min(flow.steps.length - 1, Number(index)));
    renderFlowInspector(flow);
    applyCurrentFocus();
    scheduleHardwareFit();
  }

  function setMode(mode) {
    if (!['migration', 'flow'].includes(mode)) return;
    stopPlayback();
    state.mode = mode;
    state.selectedId = null;
    state.activeStep = -1;
    $('#navigationMode').value = mode;
    $('#migrationExplorer').hidden = mode !== 'migration';
    $('#flowExplorer').hidden = mode !== 'flow';
    $('#playbackMount').hidden = mode !== 'flow' || state.previewView !== 'hardware';
    $$('.entity-button').forEach((button) => button.classList.remove('is-selected'));
    if (mode === 'migration') {
      showMigrationOverview();
    } else {
      selectFlow(FLOWS[0].id);
    }
    syncPlayback();
  }

  function showScenario(key) {
    const scenario = SCENARIOS[key];
    if (!scenario) return;
    state.selectedId = null;
    $$('[data-scenario-overview]').forEach((button) => button.classList.toggle('is-selected', button.dataset.scenarioOverview === key));
    $$('[data-category-id]').forEach((button) => button.classList.remove('is-selected'));
    $('[data-migration-overview]')?.classList.remove('is-selected');
    syncEntitySelectionAccessibility();
    renderInspector(`<div class="inspector-content">
      ${renderDocumentHeader({
        path: ['A5 算子迁移', `迁移场景 ${key}：${scenario.title}`],
        title: '场景概述',
        summary: scenario.tagline,
      })}
      ${section('识别特征（典型）', `<ul class="signal-list">${scenario.features.map((item) => `<li>${escape(item)}</li>`).join('')}</ul>`)}
      ${section('处理方式', `<ul class="action-list">${scenario.actions.map((item) => `<li>${escape(item)}</li>`).join('')}</ul>`)}
    </div>`);
    focusHardware();
    scheduleHardwareFit();
  }

  function showMigrationOverview() {
    state.selectedId = null;
    $$('[data-scenario-overview], [data-category-id]').forEach((button) => button.classList.remove('is-selected'));
    $('[data-migration-overview]')?.classList.add('is-selected');
    syncEntitySelectionAccessibility();
    const scenarioDefinitions = [
      ['A', 'A5 原生能力兼容 A2/A3', '以 A5 原生能力开发为主；需要覆盖 A2/A3 时，为旧代补充独立实现与注册分支。'],
      ['B', '简单 SIMD 样例', '旧代算子只依赖通用 GM ↔ UB 与简单向量计算；调整架构声明后，重点验证精度和性能。'],
      ['C', 'A2/A3 迁移 A5', '旧实现写死数据通路、矩阵分形、片上存储或已移除指令；必须重写硬件路径。'],
    ];
    const mappingRows = CATEGORIES.map((item) => {
      const mapping = MIGRATION_MAP[item.id];
      return `<tr>
        <td><button class="migration-map-link" type="button" data-category-jump="${item.id}">
          <span class="migration-map-index">${String(item.id).padStart(2, '0')}</span><span>${escape(item.title)}</span>
        </button></td>
        ${MIGRATION_SCENARIOS.map((scenario) => {
          if (mapping.primary === scenario.key) return `<td><span class="migration-map-dot is-primary" aria-label="主要映射到迁移场景 ${scenario.key}"></span></td>`;
          if (mapping.secondary.includes(scenario.key)) return `<td><span class="migration-map-dot is-secondary" aria-label="次要映射到迁移场景 ${scenario.key}"></span></td>`;
          return '<td><span class="migration-map-empty" aria-hidden="true">—</span></td>';
        }).join('')}
      </tr>`;
    }).join('');
    renderInspector(`<div class="inspector-content migration-overview-content">
      ${renderDocumentHeader({
        path: ['A5 算子迁移'],
        title: '迁移场景总览',
        summary: '三类场景按 A5 代际主导后的常见工作顺序组织：优先处理 A5 原生开发与跨代兼容，其次处理可轻量迁移的简单 SIMD，最后处理必须重写硬件假设的旧代实现。',
      })}
      ${section('三类场景', `<div class="scenario-definition-list">${scenarioDefinitions.map(([key, title, description]) => `
        <article class="inspector-card scenario-definition-card">
          <span class="comparison-marker scenario-key-tag">${key}</span>
          <div><strong>${escape(title)}</strong><p>${escape(description)}</p></div>
        </article>`).join('')}</div>`)}
      ${section('全量变化项与场景映射', `<div class="comparison-table-wrap">
          <table class="comparison-table migration-map-table">
            <thead><tr><th scope="col">变化项</th>${MIGRATION_SCENARIOS.map((scenario) => `<th scope="col" title="${escape(scenario.title)}">${scenario.key}</th>`).join('')}</tr></thead>
            <tbody>${mappingRows}</tbody>
          </table>
        </div>`, `
          <span><i class="migration-map-dot is-primary" aria-hidden="true"></i><b>实心圆：主要场景</b></span>
          <span><i class="migration-map-dot is-secondary" aria-hidden="true"></i><b>空心圆：次要场景</b></span>
          <span><i class="migration-map-empty" aria-hidden="true">—</i><b>短横线：不映射</b></span>
        `)}
    </div>`);
    focusHardware();
    scheduleHardwareFit();
  }

  function syncEntitySelectionAccessibility() {
    $$('.entity-button').forEach((button) => {
      if (button.classList.contains('is-selected')) button.setAttribute('aria-current', 'page');
      else button.removeAttribute('aria-current');
    });
  }

  function syncPreviewView() {
    $$('[data-preview-view]').forEach((tab) => {
      const selected = tab.dataset.previewView === state.previewView;
      tab.classList.toggle('is-selected', selected);
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
    });
    $$('[data-view-panel]').forEach((panel) => { panel.hidden = panel.dataset.viewPanel !== state.previewView; });
    $('#playbackMount').hidden = state.previewView !== 'hardware' || state.mode !== 'flow';
    if (state.previewView === 'hardware') scheduleHardwareFit();
    else scheduleMemoryFit();
  }

  function openMemoryReuse(detail = {}) {
    if (detail.node && detail.node !== 'buffer:UB') return;
    const helper = window.PtoMemoryReuseViewer;
    const host = $('#memoryReuseHost');
    if (!helper || !host) return;
    state.memoryReuseViewer?.destroy?.();
    const coreLabel = detail.coreTitle || detail.coreId || 'AIV';
    const data = helper.createDemoData({ coreId: detail.coreId, coreTitle: coreLabel });
    state.memoryReuseData = data;
    $('#memoryReuseTitle').textContent = 'UB 内存复用分析';
    $('#memoryReuseSub').textContent = data.kernel || `${coreLabel} · MatMulAddRelu_mix_aic__kernel0`;
    state.previewView = 'memory';
    syncPreviewView();
    state.memoryReuseViewer = helper.render(host, data, { initialBuffer: 'UB' });
    scheduleMemoryFit();
  }

  function closeMemoryReuse() {
    state.previewView = 'hardware';
    state.memoryReuseViewer?.destroy?.();
    state.memoryReuseViewer = null;
    syncPreviewView();
  }

  function syncGenerationCompare() {
    const enabled = state.arch === 'ascend950b';
    const button = $('#generationCompare');
    button.disabled = !enabled;
    button.classList.toggle('is-selected', enabled && state.diff);
    button.setAttribute('aria-pressed', String(enabled && state.diff));
    button.title = enabled ? '突出 950 新增资源与通路，并查看移除项摘要' : '切换到 950 后查看代际对比';
    $('#generationCompareSummary').hidden = !(enabled && state.diff);
  }

  function toggleGenerationCompare() {
    if (state.arch !== 'ascend950b') return;
    state.diff = !state.diff;
    syncGenerationCompare();
    applyCurrentFocus();
    scheduleHardwareFit();
  }

  function syncPlayback() {
    if (!state.playback) return;
    const root = $('#playbackMount');
    const flow = FLOWS.find((entry) => entry.id === state.selectedId);
    const total = flow?.steps.length || 1;
    const current = state.activeStep >= 0 ? state.activeStep : 0;
    const scrubber = $('.pto-floating-playback__scrubber', root);
    const label = $('.pto-floating-playback__counter', root);
    const opname = $('.pto-floating-playback__opname', root);
    const play = $('.pto-floating-playback__button--primary', root);
    if (scrubber) { scrubber.max = Math.max(0, total - 1); scrubber.value = current; scrubber.disabled = !flow; }
    if (label) label.textContent = flow ? `${current + 1} / ${total}` : '0 / 0';
    if (opname) opname.textContent = flow ? flow.steps[current].label : '选择执行流';
    if (play) play.innerHTML = window.PtoFloatingPlaybackControl.iconLabel(state.playing ? 'pause' : 'play', state.playing ? 'Pause' : 'Play');
    state.playback.sync({ playing: state.playing });
  }

  function stopPlayback() {
    window.clearInterval(state.timer);
    state.timer = null;
    state.playing = false;
    syncPlayback();
  }

  function togglePlayback() {
    const flow = FLOWS.find((entry) => entry.id === state.selectedId);
    if (!flow) return;
    if (state.playing) return stopPlayback();
    if (state.activeStep < 0 || state.activeStep >= flow.steps.length - 1) selectStep(0);
    state.playing = true;
    syncPlayback();
    state.timer = window.setInterval(() => {
      if (state.activeStep >= flow.steps.length - 1) return stopPlayback();
      selectStep(state.activeStep + 1);
      state.playing = true;
      syncPlayback();
    }, 1500);
  }

  function initPlayback() {
    const root = $('#playbackMount');
    const control = window.PtoFloatingPlaybackControl.createControl({ className: 'pto-floating-playback--preview pto-ide-frame__floating-playback', showTimeline: true });
    root.appendChild(control);
    state.playback = window.PtoFloatingPlaybackControl.init({ root, isPlaying: () => state.playing });
    state.playbackHover = window.PtoFloatingPlaybackControl.initScrubberHover({
      root,
      getTotalSteps: () => FLOWS.find((entry) => entry.id === state.selectedId)?.steps.length || 1,
      getLabelForStep: (index) => FLOWS.find((entry) => entry.id === state.selectedId)?.steps[index]?.label || '选择执行流',
    });
    $('.pto-floating-playback__button--primary', root)?.addEventListener('click', togglePlayback);
    $('#step-back-btn', root)?.addEventListener('click', () => selectStep(state.activeStep - 1));
    $('#step-fwd-btn', root)?.addEventListener('click', () => selectStep(state.activeStep + 1));
    $('#replay-btn', root)?.addEventListener('click', () => { stopPlayback(); selectStep(0); });
    $('.pto-floating-playback__scrubber', root)?.addEventListener('input', (event) => { stopPlayback(); selectStep(event.target.value); });
    syncPlayback();
  }

  function initHardware() {
    renderHardwareArchitecture($('#hardwareGraph'));
    state.overlay = window.PtoMemoryArchitecturePattern.createRouteOverlay($('#hardwareGraph'), state.arch);
    window.PtoMemoryArchitecturePattern.attachHoverInteractions($('#hardwareGraph'), state.arch);
    state.viewport = window.PtoHardwareArchitectureViewport.mount($('#hardwareViewport'), {
      mode: 'inline',
      viewport: '[data-stage]',
      scaleEl: '[data-scale]',
      inlineHost: '#hardwareGraph',
      detailToggle: '[data-detail]',
      zoomOut: '[data-zoom-out]',
      zoomIn: '[data-zoom-in]',
      fit: '[data-fit]',
      readout: '[data-readout]',
      zoomLevels: [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
      defaultScale: 0.6,
      fitOnMount: true,
      pan: true,
      wheelZoom: true,
      onDetailChange: () => scheduleHardwareFit(),
      onScaleChange: () => requestAnimationFrame(() => state.overlay?.update?.()),
      onPanChange: () => requestAnimationFrame(() => state.overlay?.update?.()),
    });
    observeHardwareSize();
    scheduleHardwareFit();
  }

  function initEvents() {
    document.addEventListener('click', (event) => {
      const treeToggle = event.target.closest('[data-tree-toggle]');
      if (treeToggle) {
        const expanded = treeToggle.getAttribute('aria-expanded') === 'true';
        const group = document.getElementById(treeToggle.getAttribute('aria-controls'));
        treeToggle.setAttribute('aria-expanded', String(!expanded));
        treeToggle.closest('[data-tree-branch]')?.classList.toggle('is-collapsed', expanded);
        if (group) group.hidden = expanded;
        return;
      }
      const scenarioOverview = event.target.closest('[data-scenario-overview]');
      if (scenarioOverview) return showScenario(scenarioOverview.dataset.scenarioOverview);
      const migrationOverview = event.target.closest('[data-migration-overview]');
      if (migrationOverview) return showMigrationOverview();
      const categoryJump = event.target.closest('[data-category-jump]');
      if (categoryJump) return selectCategory(categoryJump.dataset.categoryJump);
      const category = event.target.closest('[data-category-id]');
      if (category) return selectCategory(category.dataset.categoryId);
      const flow = event.target.closest('[data-flow-id]');
      if (flow) return selectFlow(flow.dataset.flowId);
      const related = event.target.closest('[data-related-flow]');
      if (related) { setMode('flow'); selectFlow(related.dataset.relatedFlow); return; }
      const step = event.target.closest('[data-step]');
      if (step) { stopPlayback(); selectStep(step.dataset.step); return; }
      const preview = event.target.closest('[data-preview-view]');
      if (preview?.dataset.previewView === 'memory') return openMemoryReuse();
      if (preview?.dataset.previewView === 'hardware') return closeMemoryReuse();
    });
    $('#navigationMode').addEventListener('change', (event) => {
      setMode(event.target.value);
      event.target.blur();
    });
    $$('[data-arch-id]').forEach((button) => button.addEventListener('click', () => setArch(button.dataset.archId)));
    $$('[data-preview-view]').forEach((tab) => tab.addEventListener('keydown', (event) => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const tabs = $$('[data-preview-view]');
      const current = tabs.indexOf(tab);
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (current + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
      tabs[next].focus();
      if (tabs[next].dataset.previewView === 'memory') openMemoryReuse();
      else closeMemoryReuse();
    }));
    $('#rightPaneToggle').addEventListener('click', () => {
      const pane = $('#visualizationPane');
      const hidden = !pane.hidden;
      pane.hidden = hidden;
      pane.setAttribute('aria-hidden', String(hidden));
      const gutter = pane.previousElementSibling;
      if (gutter?.matches('.pto-workbench-shell__split-gutter')) gutter.hidden = hidden;
      $('#rightPaneToggle').classList.toggle('is-selected', !hidden);
      $('#rightPaneToggle').setAttribute('aria-expanded', String(!hidden));
      $('#rightPaneToggle').setAttribute('aria-pressed', String(!hidden));
      window.dispatchEvent(new Event('resize'));
      if (!hidden) {
        if (state.previewView === 'hardware') scheduleHardwareFit();
        else scheduleMemoryFit();
      }
    });
    $('#memoryReuseClose').addEventListener('click', closeMemoryReuse);
    $('#generationCompare').addEventListener('click', toggleGenerationCompare);
    const frame = $('[data-ide-frame]');
    frame.addEventListener('pointermove', (event) => {
      const rect = frame.getBoundingClientRect();
      frame.style.setProperty('--ide-cursor-x', `${event.clientX - rect.left}px`);
      frame.style.setProperty('--ide-cursor-y', `${event.clientY - rect.top}px`);
      frame.style.setProperty('--ide-cursor-alpha', '0.16');
    });
    frame.addEventListener('pointerleave', () => frame.style.setProperty('--ide-cursor-alpha', '0'));
  }

  renderLists();
  initHardware();
  initPlayback();
  initEvents();
  syncGenerationCompare();
  syncPreviewView();
  setMode('migration');
  window.addEventListener('load', () => {
    if (state.previewView === 'hardware') scheduleHardwareFit();
    else scheduleMemoryFit();
  }, { once: true });
  document.fonts?.ready?.then(() => {
    if (state.previewView === 'hardware') scheduleHardwareFit();
    else scheduleMemoryFit();
  });
})();
