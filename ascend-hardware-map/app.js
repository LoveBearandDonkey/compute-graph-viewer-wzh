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

  const MIGRATION_DIMENSIONS = [
    { label: '工程适配', description: '检查编译目标、Device / Host 代际分支、算子注册与发布配置。' },
    { label: '编程与执行', description: '检查编程模型、执行范式和指令能力是否需要替换或拆分实现。' },
    { label: '数据通路', description: '检查数据在 GM、片上存储、Cube 与 Vector 之间的搬运路径。' },
    { label: '数据与存储', description: '检查数据格式、矩阵布局、scale 生命周期、容量与 bank 组织。' },
    { label: '运行语义', description: '检查数值行为、同步原语、调试接口及其跨代差异。' },
    { label: '通信协同', description: '检查集合通信的执行落点、资源约束与通算重叠。' },
  ];

  const CATEGORIES = [
    { id: 1, dimension: '工程适配', title: '目标架构声明与双代注册', sub: '编译、Host 分支与注册配置', scenario: 'B', arch: 'ascend950b', essence: '架构标识用于声明编译与注册目标；它只能证明实现面向哪一代产品，不能单独证明 kernel 已具备跨代可运行性。', signals: ['CMAKE_ASC_ARCHITECTURES', '--npu-arch=dav-3510', '__NPU_ARCH__', 'AddConfig("ascend950")'], actions: ['核对 Device 与 Host 两侧的代际分支', '双架构分别编译、运行并对比精度与性能'], selectors: [], routes: [], related: ['vector', 'cube'] },
    { id: 2, dimension: '编程与执行', title: 'Vector 编程模型：Membase 到 RegBase', sub: '寄存器化执行与对象模型', scenario: 'A', arch: 'ascend950b', essence: '950 AIV 将寄存器张量、谓词寄存器与地址寄存器提升为显式编程对象，并明确 GM → UB → Reg 的数据生命周期。', signals: ['RegTensor', 'MaskReg', 'AddrReg', 'LoadAlign / StoreAlign', 'LocalMemBar'], actions: ['把 UB 布局与寄存器压力一起评估', '为 910B 保留独立 Membase 实现'], selectors: [N.ub, N.vector, N.scalar], routes: ['l2-to-aiv1'], related: ['vector'] },
    { id: 3, dimension: '编程与执行', title: 'Vector 执行范式：SIMD 与 SIMT 并存', sub: '连续计算与离散计算分流', scenario: 'A', arch: 'ascend950b', essence: '950 在 Vector 侧新增 SIMT 执行能力，与 SIMD 形成互补：规则连续计算继续使用 SIMD，离散访存与线程语义可由 SIMT 表达。', signals: ['--enable-simt', 'Warp / ThreadBlock', 'asc_atomic_add', 'Gather / Scatter'], actions: ['仅在离散访存或线程语义明确时启用 SIMT', '分别验证 SIMD 与 SIMT 路径的性能边界'], selectors: [N.simt, N.simd, N.ub], routes: [], related: ['vector'] },
    { id: 4, dimension: '数据通路', title: '片上数据通路重构', sub: '旧直通路径移除与 C-V 直连', scenario: 'C', arch: 'ascend950b', essence: '950 调整 Cube 周边片上数据通路：移除部分旧直通路径，并增加 UB↔L1、L0C→UB、NDDMA 等路径。', signals: ['删除 GM → L0A/L0B', '删除 L1 → GM', '新增 UB → L1', '新增 L0C → UB', 'NDDMA / SSBuf'], actions: ['按 950 真实通路重写搬运链', '寻找 C-V 直连带来的融合机会'], selectors: [N.gm, N.l1, N.l0a, N.l0b, N.l0c, N.fp, N.ub], routes: ['l2-to-aic', 'aic-to-aiv1'], related: ['cube', 'gemm-ar'] },
    { id: 5, dimension: '编程与执行', title: 'Cube 指令能力调整', sub: '移除能力与低比特路径替换', scenario: 'C', arch: 'ascend950b', essence: '950 移除部分 int4、结构化稀疏与边界绕回能力，并围绕 MX 低比特计算重新组织 Cube 能力。', signals: ['MmadWithSparse 移除', 'int4 Cube Matmul 移除', 'SetLoadDataBoundary 移除', 'MX 系列新增'], actions: ['替换已移除指令与隐含硬件假设', '低比特方案转向 FP8 / MX 并重做精度验证'], selectors: [N.cube, N.l0a, N.l0b, N.l0c], routes: [], related: ['cube'] },
    { id: 6, dimension: '数据与存储', title: 'L0A 矩阵分形迁移', sub: 'A 矩阵布局由 ZZ 调整为 NZ', scenario: 'C', arch: 'ascend950b', essence: '950 将 A 矩阵在 L0A 中的分形由 ZZ 调整为 NZ；依赖固定切分与地址公式的实现必须重新生成参数。', signals: ['A / L0A: ZZ → NZ', 'B / L0B: ZN 不变', 'C / L0C: NZ 不变'], actions: ['定位写死 L0A 分形的地址公式', '按 950 分形重新生成 tiling 与搬运参数'], selectors: [N.l0a, N.l1, N.cube], routes: [], related: ['cube'] },
    { id: 7, dimension: '数据与存储', title: '低比特数据格式扩展', sub: 'HiF8、FP8 与 MX 系列', scenario: 'A', arch: 'ascend950b', essence: 'HiF8、FP8 与 MX 系列不仅扩展数据类型，还将 scale 布局、搬运、舍入、饱和与量化融合纳入主计算路径。', signals: ['HiF8', 'FP8 E5M2 / E4M3', 'MXFP4 / MXFP8', 'MicroScaling', 'Histograms'], actions: ['把 scale 张量纳入 tiling 主路径', '补齐端到端精度和饱和行为验证'], selectors: [N.cube, N.fp, N.l1], routes: ['l2-to-aic'], related: ['cube'] },
    { id: 8, dimension: '数据与存储', title: 'UB Bank 组织调整', sub: '容量、分组与冲突策略重评估', scenario: 'C', arch: 'ascend950b', essence: '950 调整 UB bank group、每组 bank 数与单 bank 容量；旧代容量假设和地址错位策略需要重新验证。', signals: ['bank group 16 → 8', '每组 3 → 2 banks', '单 bank 4KB → 16KB', 'UB 192KB → 256KB'], actions: ['删除写死容量与 bank 错位公式', '用 profiling 验证冲突与带宽'], selectors: [N.ub], routes: [], related: ['vector', 'tput-sync'] },
    { id: 9, dimension: '运行语义', title: '数值、同步与调试语义差异', sub: '跨代行为与诊断接口校验', scenario: 'B', arch: 'ascend950b', essence: '即使计算结构保持不变，Subnormal、核间同步与调试接口差异仍可能改变数值结果、同步行为和诊断方式。', signals: ['Subnormal 默认不支持', '核间 Mutex 新增', 'CheckLocalMemoryIA 移除'], actions: ['建立跨代数值回归基线', '将同步与调试接口纳入迁移检查表'], selectors: [N.vector, N.scalar], routes: [], related: ['vector', 'gemm-ar'] },
    { id: 10, dimension: '通信协同', title: '集合通信执行路径下沉', sub: 'HCCL 语义与 CCU 执行协同', scenario: 'A', arch: 'ascend950b', essence: 'HCCL 继续提供上层集合通信语义，950 将部分执行下沉至 CCU，需要联合评估硬件资源、片上内存压力与通算重叠。', signals: ['CCU', 'ReduceScatter', 'AllGatherMatMul', 'Dispatch / Combine', 'CCU profiling'], actions: ['区分 HCCL 接口语义与 CCU 执行落点', '用 CCU profiling 验证通信、片上内存和 AI Core 协同'], selectors: [N.l2, N.scalar, N.cube, N.l0c], routes: [], related: ['ccu-collective', 'gemm-ar'] },
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
        ['删除 GM → L0A / L0B 直通', '旧快捷路径', '有', '已移除，需改为 GM → L1 → L0', 'C'],
        ['删除 L1 → GM', 'L1 直接写回', '有', '已移除，需经 L0C / Fixpipe 承接', 'C'],
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
      title: '轻量迁移', tagline: '改架构声明后验证，通常不需要重写硬件路径',
      features: ['CMake 中 CMAKE_ASC_ARCHITECTURES 可声明 dav-2201, dav-3510', 'kernel 主要是 GM ↔ UB 搬运、逐元素计算或简单 reduce', '不写死矩阵分形、L0A/L0B/L1 复杂路径', '不使用 BuiltIn API 或内部 impl 接口'],
      actions: ['用 dav-2201 编译并在 2201 设备验证', '用 dav-3510 编译并在 3510 设备验证', '对照精度与性能，重点检查 Subnormal 和 UB bank 冲突'],
    },
    C: {
      title: '深度迁移', tagline: '旧代硬件假设不再成立，需要重写数据与计算路径',
      features: ['使用 L0A/L0B/L0C、LoadData、Mmad、Fixpipe', '使用 int4 Cube Matmul、4:2 结构化稀疏或 SetLoadDataBoundary', '假设 GM → L0 或 L1 → GM 通路存在', '写死 UB 大小、bank group、核数或分形布局'],
      actions: ['不能只改 CMAKE_ASC_ARCHITECTURES', '先逐项检查数据通路和矩阵分形', '重写或拆分搬运路径，例如 GM → L1 → L0', 'int4 路径改为 Vector Cast 到 int8，或采用 950 的 MX / FP8 方案', '重新调优 tiling 与 bank 规避策略'],
    },
  };

  const MIGRATION_SCENARIOS = [
    { key: 'A', title: 'A5 原生能力兼容 A2/A3', overview: 'A5 原生算子开发及跨代兼容', tag: 'A5 主导' },
    { key: 'B', title: '轻量迁移', overview: '旧架构算子经简单调整迁移至 A5', tag: '轻量迁移' },
    { key: 'C', title: '深度迁移', overview: '重写旧代硬件路径与实现假设', tag: '深度迁移' },
  ];

  const GUIDE_DOCUMENTS = [
    {
      id: 'architecture-cognition',
      title: '架构认知',
      sub: '从代际称谓到 950 硬件拓扑',
      group: 'architecture',
      arch: 'ascend950b',
      selectors: [],
      routes: [],
    },
    {
      id: 'guide-development',
      title: 'A5 算子开发概览',
      sub: '从数据流到优化验证',
      group: 'development',
      arch: 'ascend950b',
      selectors: [],
      routes: [],
    },
    {
      id: 'guide-development-vector',
      title: 'Vector 算子开发',
      sub: 'SIMD 与 SIMT 的数据形态分流',
      group: 'development',
      arch: 'ascend950b',
      scenarioKey: 'vector',
      selectors: [],
      routes: [],
    },
    {
      id: 'guide-development-cube',
      title: 'Cube 算子开发',
      sub: '矩阵主计算与低比特路径',
      group: 'development',
      arch: 'ascend950b',
      scenarioKey: 'cube',
      selectors: [],
      routes: [],
    },
    {
      id: 'guide-development-fusion',
      title: '融合算子开发',
      sub: 'Cube-Vector 直连与协同',
      group: 'development',
      arch: 'ascend950b',
      scenarioKey: 'fusion',
      selectors: [],
      routes: [],
    },
    {
      id: 'guide-migration-checklist',
      title: '迁移检查清单',
      sub: '旧硬件假设与代码 review',
      group: 'scenario-C',
      arch: 'ascend950b',
      selectors: [N.gm, N.ub, N.vector, N.l1, N.l0a, N.l0b, N.cube, N.l0c],
      routes: ['l2-to-aiv1', 'l2-to-aic', 'aic-to-aiv1'],
    },
    {
      id: 'generation-diff',
      title: '代际差异速查',
      sub: 'A5 与 A2/A3 的关键差异',
      group: 'architecture',
      arch: 'ascend950b',
      selectors: [],
      routes: [],
    },
  ];

  const GUIDE_FOCUS = {
    'development-vector': { selectors: [N.gm, N.ub, N.vector, N.simd], routes: ['l2-to-aiv1'] },
    'development-simt': { selectors: [N.gm, N.ub, N.simt], routes: ['l2-to-aiv1'] },
    'development-cube': { selectors: [N.gm, N.l1, N.l0a, N.l0b, N.cube, N.l0c], routes: ['l2-to-aic'] },
    'development-fusion': { selectors: [N.l0c, N.fp, N.ub, N.vector], routes: ['aic-to-aiv1'] },
    'migration-gm-detour': { selectors: [N.gm, N.l0c, N.ub, N.vector], routes: ['aic-to-aiv1', 'l2-to-aiv1'] },
    'migration-regbase': { selectors: [N.gm, N.ub, N.vector, N.simd], routes: ['l2-to-aiv1'] },
    'migration-lowbit': { selectors: [N.gm, N.ub, N.l1, N.l0a, N.l0b, N.cube], routes: ['l2-to-aic', 'l2-to-aiv1'] },
    'migration-tiling': { selectors: [N.ub, N.l1, N.l0a, N.l0b, N.l0c], routes: ['l2-to-aic'] },
  };

  const DEVELOPMENT_SCENARIOS = [
    {
      key: 'vector',
      title: 'Vector 算子开发',
      feature: 'SIMD + SIMT',
      summary: '按连续规则数据与离散线程语义选择 SIMD、SIMT 和 RegBase 路径。',
      shape: '连续规则表达式、规整向量融合，以及离散索引、复杂分支、Gather/Scatter、Hash/Atomic。',
      path: '连续数据：GM → UB → Vector Reg File；离散数据：GM → UB → SIMT Reg File，或 GM → SIMT Reg File。',
      legacy: '910B 以 GM↔UB、LocalTensor 和 Memory-based SIMD 为主；离散逻辑常被展开为循环、mask 或多段搬运。',
      current: '950 按数据形态分流：连续规则数据用 RegBase / SIMD VF，离散线程语义用 Thread Block / Warp / SIMT VF。',
      focusId: 'development-vector',
      flowId: 'vector',
      codeSamples: [
        {
          title: '950 代码样例',
          filename: 'vector_kernel_950.cpp',
          code: `<span class="syntax-comment">// 伪代码：先按数据形态选 SIMD 或 SIMT，而不是按算子名称硬编码</span>
<span class="syntax-keyword">if constexpr</span> (<span class="syntax-literal">is_contiguous</span> &amp;&amp; <span class="syntax-literal">expression_is_regular</span>) {
  <span class="syntax-call">CopyGMToUB</span>();
  <span class="syntax-call">LoadToVectorReg</span>();
  <span class="syntax-call">RunSimdVf</span>();       <span class="syntax-comment">// RegBase / SIMD</span>
  <span class="syntax-call">StoreVectorRegToUB</span>();
} <span class="syntax-keyword">else</span> {
  <span class="syntax-call">MapThreadBlockAndWarp</span>();
  <span class="syntax-call">GatherScatterByLane</span>();
  <span class="syntax-call">RunSimtVf</span>();       <span class="syntax-comment">// SIMT + bounds / sync check</span>
}`,
        },
      ],
    },
    {
      key: 'cube',
      title: 'Cube 算子开发',
      feature: '低比特',
      summary: '围绕矩阵主计算、低比特数据与 scale 生命周期组织 Cube 数据路径。',
      shape: 'MatMul、GEMM、GMM、qbmm、MoE 等矩阵主计算；低比特主数据与分组 scale 共同组成输入。',
      path: 'GM / L2 → L1 → L0A / L0B → Cube → L0C；scale 优先按 tile group 预取并在 UB / L1 复用。',
      legacy: '910B 典型实现围绕 L1、L0A/L0B/L0C 和 MMAD 主循环；低比特容易被当成 dtype 替换，scale 生命周期后置。',
      current: '950 需要同时设计 HiF8 / FP8 / MXFP8 / MXFP4、scale cache、L0A NZ layout、尾块与精度策略。',
      focusId: 'development-cube',
      flowId: 'cube',
      codeSamples: [
        {
          title: '950 代码样例',
          filename: 'lowbit_matmul_950.cpp',
          code: `<span class="syntax-comment">// 伪代码：先识别重复加载风险，再让主数据与 scale 共用 Tiling 计划</span>
<span class="syntax-comment">// 风险写法：scale 随每个 tile 重复从 GM 读取</span>
<span class="syntax-keyword">for</span> (<span class="syntax-type">Tile</span> tile : k_tiles) {
  <span class="syntax-call">LoadA</span>(tile);
  <span class="syntax-call">LoadB</span>(tile);
  <span class="syntax-call">LoadScaleFromGM</span>(tile); <span class="syntax-comment">// 额外 GM 搬运可能抵消低比特收益</span>
  <span class="syntax-call">MmadLowBitWithScale</span>();
}

<span class="syntax-comment">// 950 方向：按 tile group 预加载并复用 scale</span>
<span class="syntax-type">ScaleTile</span> scale_ub = <span class="syntax-call">PreloadScaleToUB</span>(scale_gm);
<span class="syntax-keyword">for</span> (<span class="syntax-type">Tile</span> tile : k_tiles) {
  <span class="syntax-call">LoadAtoL0A_NZ</span>(tile.a);
  <span class="syntax-call">LoadBtoL0B</span>(tile.b);
  <span class="syntax-call">UseScaleFromUB</span>(scale_ub, tile.group);
  <span class="syntax-call">MmadLowBit</span>();       <span class="syntax-comment">// Cube → L0C</span>
}
<span class="syntax-call">FixPipeOrEpilogue</span>();`,
        },
      ],
    },
    {
      key: 'fusion',
      title: '融合算子开发',
      feature: 'C-V 直连',
      summary: '围绕 Cube 主计算与 Vector 后处理重画数据流，减少 GM 中转和 Kernel Launch。',
      shape: 'Cube 主计算 + Vector 后处理，例如 MatMul+Add/Activation、RmsNormQuant、FIA 与 Attention 后处理。',
      path: '优先 L0C → UB → Vector Reg File，或 UB → L1；通过 MIX、DualDest / SSBuf 缩短 AIC 与 AIV 的协作路径。',
      legacy: '910B 常把 AIC 结果写入 GM / workspace，再由 AIV 读回，算子边界和中间存储绑定较紧。',
      current: '950 可围绕 AIC+AIV 协同、L0C→UB、UB→L1 与 DualDest 重画数据流，减少 GM 中转与 Kernel Launch。',
      focusId: 'development-fusion',
      flowId: 'gemm-ar',
      codeSamples: [
        {
          title: '910B 代码样例',
          filename: 'mix_epilogue_910b.cpp',
          code: `<span class="syntax-comment">// 伪代码：910B 常见的 GM / workspace 中转路径</span>
<span class="syntax-call">Mmad</span>(a_l0a, b_l0b, c_l0c);
<span class="syntax-call">CopyL0CToGM</span>(c_l0c, workspace_gm);
<span class="syntax-call">CopyGMToUB</span>(workspace_gm, x_ub);
<span class="syntax-call">VectorEpilogue</span>(x_ub, out_gm);`,
        },
        {
          title: '950 代码样例',
          filename: 'mix_epilogue_950.cpp',
          code: `<span class="syntax-comment">// 伪代码：满足约束时优先评估 C-V 直连</span>
<span class="syntax-call">Mmad</span>(a_l0a, b_l0b, c_l0c);
<span class="syntax-call">CopyL0CToUB</span>(c_l0c, x_ub); <span class="syntax-comment">// 或 DualDest</span>
<span class="syntax-call">VectorEpilogue</span>(x_ub, out_gm);`,
        },
      ],
    },
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

  const SUMMARY_HIGHLIGHTS = {
    overview: ['A5 原生能力兼容 A2/A3', '轻量迁移', '深度迁移'],
    terminology: ['平台代际', '产品型号', '软件架构标识'],
    'architecture-cognition': ['Cube-Vector 协同', '访存与互联', '新的算子表达空间'],
    'guide-development': ['先画数据流', '按推荐路径逐层优化', '用指标验证收益'],
    'guide-development-vector': ['SIMD 与 SIMT', '连续与离散数据', 'RegBase'],
    'guide-development-cube': ['矩阵主计算', '低比特', 'scale cache'],
    'guide-development-fusion': ['Cube-Vector 协同', 'C-V 直连', '减少 GM 中转'],
    'generation-diff': ['执行范式', '数据通路', '片上存储', '低比特与通信'],
    'guide-migration-checklist': ['迁移决策表', '架构升级检查项', '分层验收'],
    'scenario-A': ['A5 原生能力', '独立实现分支'],
    'scenario-B': ['改架构声明后验证', '不需要重写硬件路径'],
    'scenario-C': ['旧代硬件假设不再成立', '重写数据与计算路径'],
    'category-1': ['只说明“想跑哪一代”', '不能单独证明实现可运行'],
    'category-2': ['寄存器张量、谓词与地址寄存器', 'GM → UB → Reg'],
    'category-3': ['新增 SIMT 子系统', '不替换 SIMD'],
    'category-4': ['旧直通路径移除', 'UB↔L1、L0C→UB、NDDMA'],
    'category-5': ['移除部分 int4、结构化稀疏与边界绕回能力', 'MX 低比特路径'],
    'category-6': ['L0A 的矩阵 A 分形从 ZZ 改为 NZ', '必须迁移'],
    'category-7': ['不仅改变 dtype', 'scale 布局、搬运、舍入、饱和与量化融合'],
    'category-8': ['UB bank group、每组 bank 数与单 bank 容量变化', '不再可靠'],
    'category-9': ['Subnormal、核间同步和调试接口变化', '结果或诊断方式不同'],
    'category-10': ['执行下沉至 CCU', '硬件资源与通算重叠'],
  };

  const DIFF_950_NEW = {
    selectors: [N.simt, N.fp],
    routes: ['aic-to-aiv1', 'aiv2-to-aic', 'l2-to-aiv2', 'l2-to-aiv2-dcache', 'aiv2-to-l2'],
  };

  const state = { mode: 'migration', arch: 'ascend950b', selectedId: null, guideFocusId: null, activeStep: -1, playing: false, timer: null, overlay: null, viewport: null, playback: null, playbackHover: null, hardwareResizeObserver: null, hardwareFitFrame: 0, diff: false };
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

  function syncHardwareToolbarLayout() {
    const toolbar = $('#hardwareViewport .pto-hw-viewport__toolbar');
    const title = toolbar?.querySelector('.pto-hw-viewport__title');
    const segmented = title?.querySelector('.pto-hw-viewport__segmented');
    const tools = toolbar?.querySelector('.pto-hw-viewport__tools');
    if (!toolbar || !title || !segmented || !tools) return;
    const toolbarStyle = getComputedStyle(toolbar);
    const availableWidth = toolbar.clientWidth
      - parseFloat(toolbarStyle.paddingLeft)
      - parseFloat(toolbarStyle.paddingRight);
    const rootStyle = getComputedStyle(document.documentElement);
    const toolbarGap = parseFloat(toolbarStyle.columnGap) || parseFloat(rootStyle.getPropertyValue('--space-3')) || 12;
    const requiredWidth = segmented.scrollWidth + toolbarGap + tools.scrollWidth;
    toolbar.classList.toggle('is-title-stacked', Math.ceil(requiredWidth) > Math.floor(availableWidth));
  }

  function observeHardwareSize() {
    state.hardwareResizeObserver?.disconnect?.();
    if (typeof ResizeObserver !== 'function') return;
    state.hardwareResizeObserver = new ResizeObserver(() => {
      syncHardwareToolbarLayout();
      scheduleHardwareFit();
    });
    const stage = $('#hardwareGraph [data-pto-mem-arch-stage]');
    if (stage) state.hardwareResizeObserver.observe(stage);
    const viewport = $('#hardwareViewport');
    if (viewport) state.hardwareResizeObserver.observe(viewport);
    syncHardwareToolbarLayout();
  }

  function scheduleHardwareFit() {
    if (state.hardwareFitFrame) cancelAnimationFrame(state.hardwareFitFrame);
    state.hardwareFitFrame = requestAnimationFrame(() => {
      state.hardwareFitFrame = requestAnimationFrame(() => {
        state.hardwareFitFrame = 0;
        const size = measureHardwareFrame();
        state.viewport?.setFrameSize?.(size.width, size.height);
        state.viewport?.fit?.();
        state.overlay?.update?.();
      });
    });
  }

  function renderLists() {
    const architectureDocs = [
      { type: 'guide', ...GUIDE_DOCUMENTS.find((document) => document.id === 'architecture-cognition') },
      { type: 'guide', ...GUIDE_DOCUMENTS.find((document) => document.id === 'generation-diff') },
      { type: 'terminology', id: 'terminology', title: '术语表', sub: '平台代际、产品型号与软件标识' },
    ];
    const developmentDocs = [
      { type: 'guide', ...GUIDE_DOCUMENTS.find((document) => document.id === 'guide-development') },
      { type: 'guide', ...GUIDE_DOCUMENTS.find((document) => document.id === 'guide-development-vector') },
      { type: 'guide', ...GUIDE_DOCUMENTS.find((document) => document.id === 'guide-development-cube') },
      { type: 'guide', ...GUIDE_DOCUMENTS.find((document) => document.id === 'guide-development-fusion') },
      { type: 'overview', id: 'migration-overview', title: 'A5 算子迁移概览', sub: '三类场景定义与全量变化项映射' },
    ];
    const renderDocumentTree = ({ branchId, title, documents }) => {
      const groupId = `${branchId}-documents`;
      return `
      <section class="migration-tree-group" data-tree-branch="${escape(branchId)}">
        <button class="migration-tree-toggle" type="button" role="treeitem" aria-level="1" aria-expanded="true" aria-controls="${groupId}" data-tree-toggle="${escape(branchId)}">
          <svg class="migration-tree-chevron" viewBox="0 0 24 24" aria-hidden="true"><path d="m6 9 6 6 6-6"></path></svg>
          <span class="migration-tree-toggle-title" data-nav-tooltip="${escape(title)}">${escape(title)}</span>
        </button>
        <ul class="migration-tree-children" id="${groupId}" role="group">
          ${documents.map((document, index) => {
            const dataAttribute = document.type === 'overview'
              ? 'data-migration-overview'
              : document.type === 'terminology'
                ? 'data-terminology-document'
                : `data-guide-document="${escape(document.id)}"`;
            return `<li role="none"><button class="entity-button tree-document-button" type="button" role="treeitem" aria-level="2" ${dataAttribute}>
              <span class="entity-index">${String(index + 1).padStart(2, '0')}</span>
              <span class="entity-main"><span class="entity-title" data-nav-tooltip="${escape(document.title)}">${escape(document.title)}</span><span class="entity-sub" data-nav-tooltip="${escape(document.sub)}">${escape(document.sub)}</span></span>
            </button></li>`;
          }).join('')}
        </ul>
      </section>`;
    };
    const architectureTree = renderDocumentTree({ branchId: 'architecture', title: 'A5 架构', documents: architectureDocs });
    const developmentTree = renderDocumentTree({ branchId: 'development', title: 'A5 算子开发', documents: developmentDocs });
    const scenarioTrees = MIGRATION_SCENARIOS.map((scenario) => {
      const items = CATEGORIES.filter((item) => item.scenario === scenario.key);
      const supplementalDocs = GUIDE_DOCUMENTS.filter((document) => document.group === `scenario-${scenario.key}`);
      const groupId = `migration-scenario-${scenario.key}`;
      return `
        <section class="migration-tree-group" data-tree-branch="${scenario.key}">
          <button class="migration-tree-toggle" type="button" role="treeitem" aria-level="1" aria-expanded="true" aria-controls="${groupId}" data-tree-toggle="${scenario.key}">
            <svg class="migration-tree-chevron" viewBox="0 0 24 24" aria-hidden="true"><path d="m6 9 6 6 6-6"></path></svg>
            <span class="migration-tree-toggle-title" data-nav-tooltip="迁移场景 ${scenario.key}：${escape(scenario.title)}">迁移场景 ${scenario.key}：${escape(scenario.title)}</span>
          </button>
          <ul class="migration-tree-children" id="${groupId}" role="group">
            <li role="none">
              <button class="entity-button tree-document-button scenario-overview-button" type="button" role="treeitem" aria-level="2" data-scenario-overview="${scenario.key}">
                <span class="entity-index">01</span>
                <span class="entity-main"><span class="entity-title" data-nav-tooltip="场景概述">场景概述</span><span class="entity-sub" data-nav-tooltip="${escape(scenario.overview)}">${escape(scenario.overview)}</span></span>
              </button>
            </li>
            ${items.map((item, index) => `
              <li role="none"><button class="entity-button tree-document-button" type="button" role="treeitem" aria-level="2" data-category-id="${item.id}">
                <span class="entity-index">${String(index + 2).padStart(2, '0')}</span>
                <span class="entity-main"><span class="entity-title" data-nav-tooltip="${escape(item.title)}">${escape(item.title)}</span><span class="entity-sub" data-nav-tooltip="${escape(item.sub)}">${escape(item.sub)}</span></span>
                <span class="mini-badge">${escape(item.dimension)}</span>
              </button></li>`).join('')}
            ${supplementalDocs.map((document, index) => `
              <li role="none"><button class="entity-button tree-document-button guide-document-button" type="button" role="treeitem" aria-level="2" data-guide-document="${escape(document.id)}">
                <span class="entity-index">${String(items.length + index + 2).padStart(2, '0')}</span>
                <span class="entity-main"><span class="entity-title" data-nav-tooltip="${escape(document.title)}">${escape(document.title)}</span><span class="entity-sub" data-nav-tooltip="${escape(document.sub)}">${escape(document.sub)}</span></span>
              </button></li>`).join('')}
          </ul>
        </section>`;
    }).join('');
    $('#categoryTree').innerHTML = architectureTree + developmentTree + scenarioTrees;
    $('#flowList').innerHTML = FLOWS.map((item, index) => `
      <li><button class="entity-button" type="button" data-flow-id="${item.id}">
        <span class="entity-index">${String(index + 1).padStart(2, '0')}</span>
        <span class="entity-main"><span class="entity-title" data-nav-tooltip="${escape(item.title)}">${escape(item.title)}</span><span class="entity-sub" data-nav-tooltip="${escape(item.short)}">${escape(item.short)}</span></span>
      </button></li>`).join('');
  }

  function renderHardwareArchitecture(host) {
    const rendered = window.PtoMemoryArchitecturePattern.renderArchitecture(host, state.arch);
    rendered?.stage?.querySelector('.pto-mem950__notes')?.remove();
    return rendered;
  }

  function renderHardware() {
    const host = $('#hardwareGraph');
    renderHardwareArchitecture(host);
    state.overlay?.destroy?.();
    state.overlay = window.PtoMemoryArchitecturePattern.createRouteOverlay(host, state.arch);
    window.PtoMemoryArchitecturePattern.attachHoverInteractions(host, state.arch);
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
    else if (state.mode === 'migration' && state.selectedId === 'generation-diff') state.diff = true;
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
    if (state.mode === 'migration' && typeof state.selectedId === 'string') {
      const guide = GUIDE_DOCUMENTS.find((entry) => entry.id === state.selectedId);
      const guideFocus = state.guideFocusId ? GUIDE_FOCUS[state.guideFocusId] : null;
      if (guideFocus) return focusHardware(guideFocus.selectors || [], guideFocus.routes || []);
      if (guide) return focusHardware(guide.selectors || [], guide.routes || []);
    }
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
    const helpClass = title === '关键差异对照'
      ? ' is-comparison'
      : title === '全量变化项与场景映射'
        ? ' is-mapping'
        : '';
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

  function renderHighlightedText(text, highlights = []) {
    return highlights.reduce((html, phrase) => {
      const escapedPhrase = escape(phrase);
      return html.replace(escapedPhrase, `<mark>${escapedPhrase}</mark>`);
    }, escape(text));
  }

  function renderDocumentHeader({ path, title, summary, highlights = [] }) {
    const breadcrumbs = path.map((item) => `<span>${escape(item)}</span>`).join('<i aria-hidden="true">/</i>');
    return `<section class="inspector-section inspector-document-header">
      <div class="inspector-document-path" aria-label="文档路径">${breadcrumbs}</div>
      <h2>${escape(title)}</h2>
      <blockquote class="inspector-document-summary">${renderHighlightedText(summary, highlights)}</blockquote>
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
        label.includes('新增') ? 'new' : 'scenario'
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
      <span>${comparisonMarker('B')}<b>轻量迁移；改声明后验证</b></span>
      <span>${comparisonMarker('C')}<b>深度迁移；旧代硬件假设失效</b></span>
      <span>${comparisonMarker('B/C')}<b>轻量迁移风险与深度迁移风险并存</b></span>
      <span>${comparisonMarker('优化机会')}<b>新增能力，主动用</b></span>
      <span>${comparisonMarker('删除')}<b>2201 → 3510 移除</b></span>
      <span>${comparisonMarker('新增')}<b>3510 独有</b></span>
    `;
  }

  function renderGuideFocusButton(id, label = '聚焦硬件') {
    const selected = state.guideFocusId === id;
    return `<button class="btn btn-sm guide-focus-button${selected ? ' is-selected' : ''}" type="button" data-guide-focus="${escape(id)}" aria-pressed="${selected}">${escape(label)}</button>`;
  }

  function renderRelatedFlows(ids) {
    return ids.map((id) => {
      const flow = FLOWS.find((entry) => entry.id === id);
      return flow ? `<button class="btn btn-ghost related-flow" type="button" data-related-flow="${flow.id}"><span>→ ${escape(flow.title)}</span><span class="related-flow-action">跳转查看</span></button>` : '';
    }).join('');
  }

  function renderScenarioCodeBlock(filename, code) {
    return `<div class="operator-code-surface">
      <div class="operator-code-header">
        <span>${escape(filename)}</span>
      </div>
      <pre class="guide-code-block operator-code-block"><code>${code}</code></pre>
    </div>`;
  }

  function highlightTemplateCode(code) {
    return String(code).split('\n').map((line) => {
      const escapedLine = escape(line);
      if (line.trimStart().startsWith('#')) return `<span class="syntax-comment">${escapedLine}</span>`;
      return escapedLine
        .replace(/\b([A-Za-z_][A-Za-z0-9_]*)(?=\()/g, '<span class="syntax-call">$1</span>')
        .replace(/\b(if|for|else|return)\b/g, '<span class="syntax-keyword">$1</span>');
    }).join('\n');
  }

  function renderArchitectureCognition() {
    const dimensionRows = [
      ['平台代际', 'A5', 'A2/A3', '开发策略、能力兼容与迁移场景'],
      ['产品型号', 'Ascend 950', 'Ascend 910B', '硬件架构图、产品能力与代际对比'],
      ['软件架构标识', 'DAV_3510 / dav-3510', 'DAV_2201 / dav-2201', '源码枚举、编译参数与运行时分支'],
      ['Device 宏值', '3510', '2201', '__NPU_ARCH__ 条件编译'],
    ];
    const hardwareRows = [
      ['Compute Die', '承载 AI Core、AI CPU、L2、Memory Interface、DVPP、DMA/Clink 与 STARS/D2D。'],
      ['IO Die', '位于封装两侧，承载 PCIe、UnifiedBus 与互联控制。'],
      ['AI CPU / CPU block', '承担设备侧控制、调度与辅助计算，是算子执行链中的控制对象。'],
      ['AI Core', '由 AIC/Cube Core 与 AIV/Vector Core 组成，是算子计算资源的上层组合对象。'],
      ['AIC / Cube Core', '执行矩阵计算，围绕 L1、L0A/L0B/L0C、Cube、MTE1/MTE2 与 FixPipe 工作。'],
      ['AIV / Vector Core', '执行向量与线程级计算，围绕 Unified Buffer、Vector、SIMD/SIMT、MTE2/MTE3 工作。'],
      ['L2 Cache / GM', '连接片外数据与片上计算；AIC/AIV 的通用数据协作仍以 GM/L2 为共享层。'],
      ['UnifiedBus / URMA / CCU', '构成远程内存访问、集合通信卸载和通算重叠的互联能力层。'],
    ];
    const topologyRows = [
      ['Compute Die', '2', '双 DIE 通过高速 Die-to-Die 通道形成 UMA。'],
      ['IO Die', '2', '位于封装两侧，承载互联与控制对象。'],
      ['AI CPU / CPU block', '4', '每个 Compute Die 2 个。'],
      ['AI Core', '32', '每个 Compute Die 16 个。'],
      ['AIC / Cube Core', '32', '每个 AI Core 1 个。'],
      ['AIV / Vector Core', '64', '每个 AI Core 2 个。'],
    ];
    const memoryRows = [
      ['GM / L2', '外层数据入口与跨核共享层；中间结果不应默认回到这里。', '把 workspace 当默认中转，导致 C-V 融合收益消失。'],
      ['L1 / L0A / L0B', 'Cube 主计算前的数据组织层；layout、低比特 scale 与 bank 访问共同影响吞吐。', '沿用旧 Tiling 时忽略 L0A 分形、尾块、对齐和 scale 复用。'],
      ['L0C', 'Cube 结果停留层；融合场景优先检查能否直接交给 AIV。', '仍按 L0C → GM → UB 旧路径实现，增加一次大块 GM 往返。'],
      ['UB / Vector Reg File', 'AIV staging 与 SIMD Reg 计算路径；连续规则表达式尽量减少 UB 反复读写。', 'RegBase 临时变量过多导致 spill，把寄存器优化变成 stall 来源。'],
      ['SIMT Reg File', '离散索引、复杂分支、Gather/Scatter 与线程式表达的寄存器路径。', '把离散数据硬套 SIMD，造成访存合并差且同步、边界更难排查。'],
    ];
    const differences = [
      ['从单点算力到 C/V 协同', 'A5 的核心变化不是只提高 Cube 算力，而是同时增强 Cube、Vector、访存、同步与互联，让 Mix 任务成为主要优化对象。'],
      ['从固定搬运到可编排数据通路', 'L0C→UB、UB→L1、NDDMA 与 128B Sector L2 扩大了减少 GM 往返和处理离散访问的空间。'],
      ['从 SIMD 主导到 SIMD/SIMT 并存', '规则连续计算继续使用 SIMD；离散访存、条件分支、Gather/Scatter 与线程级原子可以由 SIMT 表达。'],
      ['从单核调优到协作节奏调优', 'AIC/AIV 的 TileShape、Mix 子图边界与同步等待共同决定端到端性能。'],
    ];
    renderInspector(`<div class="inspector-content guide-document-content">
      ${renderDocumentHeader({
        path: ['A5 架构与编程指南', 'A5 架构'],
        title: '架构认知',
        summary: '先建立 A5 的硬件对象、拓扑和编程能力全景，再把算子设计落到 Cube-Vector 协同、访存与互联路径上。',
        highlights: SUMMARY_HIGHLIGHTS['architecture-cognition'],
      })}
      ${renderContextCard(`<dl class="meta-grid">
        <div class="meta-row"><dt>核心对象</dt><dd>Ascend 950、Compute Die、AI Core、AIC、AIV、L2、GM 与 IO Die</dd></div>
        <div class="meta-row"><dt>开发主线</dt><dd>数据流、执行范式、片上复用、同步协同与跨代兼容</dd></div>
        <div class="meta-row"><dt>文档定位</dt><dd>架构认知基线</dd></div>
      </dl>`)}
      ${section('架构指代维度', `<p class="section-lead">用平台代际、产品型号和软件架构标识三层口径阅读文档；代码中还会出现对应的 Device 宏值。相邻称谓有关联，但不能在所有上下文中直接画等号。</p>
        <div class="comparison-table-wrap"><table class="comparison-table terminology-dimension-table">
          <thead><tr><th scope="col">维度</th><th scope="col">新平台侧</th><th scope="col">旧平台侧</th><th scope="col">主要使用位置</th></tr></thead>
          <tbody>${dimensionRows.map((row) => `<tr>${row.map((cell) => `<td>${escape(cell)}</td>`).join('')}</tr>`).join('')}</tbody>
        </table></div>`)}
      ${section('950 架构相关硬件对象', `<div class="comparison-table-wrap"><table class="comparison-table">
        <thead><tr><th scope="col">对象</th><th scope="col">在算子开发中的作用</th></tr></thead>
        <tbody>${hardwareRows.map((row) => `<tr>${row.map((cell) => `<td>${escape(cell)}</td>`).join('')}</tr>`).join('')}</tbody>
      </table></div>`)}
      ${section('硬件拓扑计数速览', `<div class="comparison-table-wrap"><table class="comparison-table">
        <thead><tr><th scope="col">对象</th><th scope="col">数量</th><th scope="col">组织关系</th></tr></thead>
        <tbody>${topologyRows.map((row) => `<tr>${row.map((cell) => `<td>${escape(cell)}</td>`).join('')}</tr>`).join('')}</tbody>
      </table></div>`)}
      ${section('片上存储层次速查', `<div class="comparison-table-wrap"><table class="comparison-table">
        <thead><tr><th scope="col">层级</th><th scope="col">开发者心智</th><th scope="col">容易踩坑</th></tr></thead>
        <tbody>${memoryRows.map((row) => `<tr>${row.map((cell) => `<td>${escape(cell)}</td>`).join('')}</tr>`).join('')}</tbody>
      </table></div>`)}
      ${section('查看执行机制', renderRelatedFlows(['vector', 'cube', 'gemm-ar']))}
      ${section('A5 和上一代的最大区别', `<div class="terminology-rule-list">${differences.map(([title, description]) => `
        <article class="inspector-card"><strong>${escape(title)}</strong><p>${escape(description)}</p></article>`).join('')}</div>`)}
    </div>`);
  }

  function renderDevelopmentGuide() {
    const workflow = [
      ['01', '画数据流', '标出 GM、L2、L1、L0A/B/C、UB 与 Reg，先检查中间结果是否绕行 GM。'],
      ['02', '选择执行范式', '连续规则数据优先 SIMD/RegBase，离散索引与复杂分支评估 SIMT，矩阵主计算使用 Cube。'],
      ['03', '设计片上复用', '确定 UB、L1、L0 和 Reg 生命周期；低比特场景同时规划 scale 的加载与复用。'],
      ['04', '验证性能假设', '先对齐 Golden 与边界 shape，再检查 pipe bubble、片上带宽、PC stall、寄存器压力和尾块。'],
    ];
    const layers = [
      ['1', '建立正确性基线', '固定 Golden、边界 shape 与低比特误差预算，形成后续优化可重复对比的基准。', 'Correctness'],
      ['2', '稳定 Tiling 与流水', '调整 Block 切分、Double Buffer 和尾块策略，让搬运与计算形成稳定重叠。', 'Tiling / Pipe'],
      ['3', '优化数据通路与片上复用', '减少不必要的 GM 往返，规划 UB、L1、L0 与 scale cache 的停留和复用。', 'Memory Path'],
      ['4', '优化执行范式与指令', '按数据形态选择 SIMD、SIMT 或 RegBase，并控制 GPR 占用、spill 与重复 scalar 控制。', 'Reg / VF'],
      ['5', '优化融合与端到端调度', '评估 Cube 主计算、Vector 后处理与通信协同，减少 Kernel Launch 和跨算子中转。', 'Fusion / Stream'],
    ];
    const features = [
      ['数据通路', 'L0C→UB、UB→L1、NDDMA 和 128B Sector L2 让中间结果、复杂搬运与小包访问有更多片上优化路径。'],
      ['C/V 融合', 'Cube 主计算与 Vector 后处理可以围绕 Mix 子图协同设计，减少多对多依赖和不必要的 GM 往返。'],
      ['SIMT / SIMD', 'SIMD 处理规则连续向量计算；SIMT 覆盖离散索引、复杂分支、Gather/Scatter 与线程级原子。'],
      ['通信', 'UnifiedBus、URMA 与 CCU 扩展远程内存访问、集合通信卸载和通算重叠的设计空间。'],
      ['低比特', 'HiF8、FP8、MXFP8、MXFP4 与 MicroScaling 把 dtype、scale、layout、搬运和精度验证连接成一条设计链。'],
      ['RegBase', 'RegTensor、MaskReg、AddrReg 与显式 Load/Store 使 GM→UB→Reg 的数据生命周期和寄存器压力进入开发决策。'],
    ];
    const validationRows = [
      ['功能与精度', 'Golden、边界 shape、round / saturate、极值样本', '结果一致且误差满足预算后，才能进入性能比较'],
      ['流水与核间负载', 'Pipe View、各核耗时、尾块分布', 'MTE、Cube、Vector 是否持续工作，是否存在长 bubble 或少数核拖尾'],
      ['访存与片上复用', 'GM 流量、片上带宽、L1 / UB bank 冲突', '数据是否停留在预期层级，GM 往返是否减少且未引入新的冲突'],
      ['寄存器与指令效率', 'PC Sampling、GPR / Reg pressure、spill', 'RegBase、SIMT 或 VF 融合是否减少开销且未造成寄存器溢出'],
      ['端到端协同', 'Kernel Launch、CCU profiling、HCCL 总耗时、通算重叠', '局部优化是否转化为端到端收益，而不是把瓶颈转移到通信或调度'],
    ];
    renderInspector(`<div class="inspector-content guide-document-content development-guide-content">
      ${renderDocumentHeader({
        path: ['A5 架构与编程指南', 'A5 算子开发'],
        title: 'A5 算子开发概览',
        summary: 'A5 算子开发不是先选语法，而是先画数据流、选择执行范式、设计片上复用，再用工具验证性能假设。',
        highlights: SUMMARY_HIGHLIGHTS['guide-development'],
      })}
      ${renderContextCard(`<dl class="meta-grid">
        <div class="meta-row"><dt>目标</dt><dd>把架构认知转成从 0-1 开发与优化的工作方法</dd></div>
        <div class="meta-row"><dt>适用对象</dt><dd>A5 新算子开发、旧方案重构与性能方案评审</dd></div>
        <div class="meta-row"><dt>文档定位</dt><dd>开发与优化方法概览</dd></div>
      </dl>`)}
      ${section('开发工作流', `<ol class="guide-workflow">${workflow.map(([number, title, description]) => `
        <li class="inspector-card guide-workflow-card"><span class="step-number">${number}</span><div><strong>${escape(title)}</strong><p>${escape(description)}</p></div></li>`).join('')}</ol>`)}
      ${section('A5 编程新特性', `<div class="guide-scenario-grid">${features.map(([title, description]) => `
        <article class="inspector-card guide-scenario-card"><div><strong>${escape(title)}</strong><p>${escape(description)}</p></div></article>`).join('')}</div>`)}
      ${section('A5 算子推荐优化路径', `<p class="section-lead">这是从正确性到端到端性能的推荐优化顺序。每一层稳定后再进入下一层，便于确认收益来源并及时发现副作用。</p>
        <div class="guide-method-layers">${layers.map(([number, title, detail, target]) => `
        <article class="inspector-card guide-method-layer"><span class="guide-method-number">${number}</span><div><strong>${escape(title)}</strong><p>${escape(detail)}</p></div><span class="mini-badge">${escape(target)}</span></article>`).join('')}</div>`)}
      ${section('优化效果验证', `<p class="section-lead">本节用于验证前述优化是否真正生效，适合作为开发概览的收尾。建议每次只改变一类策略，并用对应指标确认结果和副作用。</p>
        <div class="comparison-table-wrap"><table class="comparison-table">
        <thead><tr><th scope="col">验证目标</th><th scope="col">关键观测</th><th scope="col">判断标准</th></tr></thead>
        <tbody>${validationRows.map((row) => `<tr>${row.map((cell) => `<td>${escape(cell)}</td>`).join('')}</tr>`).join('')}</tbody>
      </table></div>`)}
    </div>`);
  }

  function renderDevelopmentScenarioGuide(document) {
    const scenario = DEVELOPMENT_SCENARIOS.find((entry) => entry.key === document.scenarioKey);
    if (!scenario) return;
    renderInspector(`<div class="inspector-content guide-document-content development-scenario-guide">
      ${renderDocumentHeader({
        path: ['A5 架构与编程指南', 'A5 算子开发'],
        title: scenario.title,
        summary: scenario.summary,
        highlights: SUMMARY_HIGHLIGHTS[document.id],
      })}
      ${renderContextCard(`<dl class="meta-grid">
        <div class="meta-row"><dt>场景特征</dt><dd><span class="mini-badge">${escape(scenario.feature)}</span></dd></div>
        <div class="meta-row"><dt>数据形态</dt><dd>${escape(scenario.shape)}</dd></div>
        <div class="meta-row"><dt>优先路径</dt><dd><div class="scenario-path-detail"><span>${escape(scenario.path)}</span>${renderGuideFocusButton(scenario.focusId)}</div></dd></div>
      </dl>`)}
      ${section('代际实现对照', `<div class="operator-scenario-facts">
        <div class="operator-scenario-fact"><span>910B 典型实现</span><p>${escape(scenario.legacy)}</p></div>
        <div class="operator-scenario-fact"><span>950 实现差异</span><p>${escape(scenario.current)}</p></div>
      </div>`)}
      ${scenario.codeSamples.map((sample) => section(sample.title, renderScenarioCodeBlock(sample.filename, sample.code))).join('')}
      ${section('关联执行流', renderRelatedFlows([scenario.flowId]))}
    </div>`);
  }

  function renderMigrationChecklist() {
    const reviewRows = [
      ['AIC 输出经 GM / workspace 交给 AIV', '后处理是否满足 C-V 片上直连条件', '优先评估 L0C→UB、DualDest 或 UB→L1，减少中间结果回写'],
      ['Vector 中间值反复落入 UB / LocalTensor', '是否属于规则连续且可融合的向量表达式', '评估 RegBase / SIMD VF，并同步检查寄存器压力与 spill'],
      ['Gather / Scatter 或分支逻辑以 SIMD 循环实现', '访问模式是否具有离散索引、线程分歧或原子语义', '评估 SIMT，并按 Thread Block / Warp 重新建模'],
      ['低比特实现仅替换 dtype，未规划 scale', 'scale 是否重复从 GM 读取，layout 是否匹配矩阵路径', '把 scale cache、cast、round、saturate 与精度校验纳入同一方案'],
      ['Tiling 写死旧代容量、核数或矩阵分形', 'L1 / L0 / UB / Reg / L2 的容量与命中假设是否仍成立', '重建 A5 可运行基线，再逐项启用新通路和优化能力'],
    ];
    const risks = [
      ['检查中间结果是否仍绕行 GM', '定位 L0C → GM → UB 或 workspace 中转；如果属于 Cube-Vector 融合场景，优先评估 C-V 直连。', 'migration-gm-detour', 4],
      ['检查 Vector 是否仍按 Membase 实现', '定位 UB 中间变量和规则表达式，判断能否迁移到 RegBase；同时验证寄存器压力与 spill。', 'migration-regbase', 2],
      ['检查低比特 scale 是否重复搬运', '把 scale 生命周期纳入 Tiling，避免在 inner loop 中反复从 GM 加载并抵消低比特收益。', 'migration-lowbit', 7],
      ['检查 Tiling 与 layout 是否写死旧代假设', '重新核对 L0A 分形、UB bank、片上容量、Double Buffer 和尾块策略，不能直接复用旧参数。', 'migration-tiling', 6],
      ['检查跨代 API 与调试接口', '扫描 L1/L0、DataCopy、LoadData、DumpTensor、低比特类型与调试接口，并按当前工具链逐项确认。', '', 5],
      ['按正确顺序完成迁移验收', '先验证功能与精度，再检查内存和同步，最后比较性能，避免总耗时掩盖实现错误。', '', 9],
    ];
    const scanTemplate = `# 示例模板：先标硬件假设，再改路径；接口与参数以当前工具链为准
scan_legacy_kernel() {
  find_data_path("L0C -> GM -> UB");
  find_memory_based_vector_temporaries();
  find_lowbit_without_scale_cache();
  find_generation_sensitive_api();
}

# 再找高风险硬件与编程模型关键字
rg "L1Buffer|L0A|L0B|L0C|int4b_t|cube_only|LoadData|DataCopy|DumpTensor" ./operator_src
rg "GM.*UB|workspace|LocalTensor|RegTensor|MaskReg|AddrReg" ./operator_src
rg "MXFP|HiF8|FP8|scale|AntiQuant|Quant" ./operator_src`;
    const profilingTemplate = `# 示例模板：迁移 review 固定观察流水、负载与 stall
profile_pipeline() {
  collect_pipe_view();
  collect_core_load();
  collect_pc_sampling();
}

retile_for_a5(shape) {
  choose_block_shape(shape);
  enable_double_buffer_when_memory_allows();
  balance_tail_block_or_pad();
  recheck_l1_bank_and_register_pressure();
}`;
    const simtDebugTemplate = `# 950 示例模板：具体编译参数以当前 CANN 版本为准
ascendc_compile \\
  --npu-arch=dav-3510 \\
  --sanitizer \\
  -g \\
  -o simt_kernel.o

# 先功能，再内存，再同步，最后比较性能
run_correctness();
run_memcheck();
run_synccheck();
run_tracecheck();
profile_pipe_and_gpr();`;
    renderInspector(`<div class="inspector-content guide-document-content">
      ${renderDocumentHeader({
        path: ['A5 架构与编程指南', '迁移场景 C：深度迁移'],
        title: '迁移检查清单',
        summary: '迁移到 A5 时先扫描旧硬件假设，再改数据路径与编程范式，最后按功能、内存、同步和性能顺序验证。',
        highlights: SUMMARY_HIGHLIGHTS['guide-migration-checklist'],
      })}
      ${renderContextCard(`<dl class="meta-grid">
        <div class="meta-row"><dt>检查目标</dt><dd>识别架构升级中最容易沿用的旧硬件假设，并落实为检查、调整和验证动作</dd></div>
        <div class="meta-row"><dt>适用对象</dt><dd>A2/A3 算子迁移、跨代维护与迁移验收</dd></div>
        <div class="meta-row"><dt>文档定位</dt><dd>迁移实施检查与验收</dd></div>
      </dl>`)}
      ${section('迁移检查顺序', `<ol class="guide-workflow">
        <li class="inspector-card guide-workflow-card"><span class="step-number">01</span><div><strong>标记代际与注册分支</strong><p>确认 Device 宏、Host tiling、CMake 和 AddConfig 是否覆盖目标产品。</p></div></li>
        <li class="inspector-card guide-workflow-card"><span class="step-number">02</span><div><strong>扫描旧硬件假设</strong><p>定位旧通路、矩阵分形、UB/bank、低比特、同步和调试接口。</p></div></li>
        <li class="inspector-card guide-workflow-card"><span class="step-number">03</span><div><strong>重画数据路径</strong><p>明确数据停留层级、AIC/AIV 分工、scale 生命周期与可融合节点。</p></div></li>
        <li class="inspector-card guide-workflow-card"><span class="step-number">04</span><div><strong>重建 Tiling baseline</strong><p>先得到可运行基线，再逐项启用 RegBase、SIMT、低比特或 C-V 新通路。</p></div></li>
        <li class="inspector-card guide-workflow-card"><span class="step-number">05</span><div><strong>分层验收</strong><p>按功能、内存、同步、性能顺序验证并记录差异。</p></div></li>
      </ol>`)}
      ${section('旧实现迁移决策表', `<p class="section-lead">根据旧代码特征判断迁移方向，回答“这类实现需要改什么”。确定方向后，再用下方检查项落实到代码 review 与验收。</p>
        <div class="comparison-table-wrap"><table class="comparison-table">
        <thead><tr><th scope="col">旧实现特征</th><th scope="col">迁移判断问题</th><th scope="col">推荐改造方向</th></tr></thead>
        <tbody>${reviewRows.map((row) => `<tr>${row.map((cell) => `<td>${escape(cell)}</td>`).join('')}</tr>`).join('')}</tbody>
      </table></div>`)}
      ${section('架构升级重点检查项', `<p class="section-lead">承接上方迁移决策，用于实施阶段逐项检查“是否改到位、是否验证充分”。每项可联动硬件路径或打开对应架构差异文档。</p>
        <div class="guide-risk-list">${risks.map(([title, description, focusId, categoryId], index) => {
          const categoryTitle = CATEGORIES.find((category) => category.id === categoryId)?.title || `变化项 ${categoryId}`;
          return `
        <article class="inspector-card card-demo guide-risk-card">
          <span class="step-number">${String(index + 1).padStart(2, '0')}</span>
          <div><strong>${escape(title)}</strong><p>${escape(description)}</p></div>
          <div class="guide-card-actions">${focusId ? renderGuideFocusButton(focusId) : ''}<button class="btn btn-sm btn-ghost" type="button" data-category-jump="${categoryId}">查看“${escape(categoryTitle)}”</button></div>
        </article>`;
        }).join('')}</div>`)}
      ${section('代码样例', `<div class="migration-code-samples">
        ${renderScenarioCodeBlock('migration_scan.pseudo', highlightTemplateCode(scanTemplate))}
        ${renderScenarioCodeBlock('profile_review.pseudo', highlightTemplateCode(profilingTemplate))}
      </div>`)}
      ${section('950 代码样例', renderScenarioCodeBlock('simt_debug_950.sh', highlightTemplateCode(simtDebugTemplate)))}
      ${section('关联执行流', renderRelatedFlows(['vector', 'cube', 'gemm-ar']))}
    </div>`);
  }

  function renderGenerationDiff() {
    const rows = [
      ['架构目标', 'A2/A3；常见对照为 Ascend 910B、DAV_2201', 'A5；当前主题为 Ascend 950、DAV_3510', '显式拆分编译、注册、Tiling 与回归配置', 1],
      ['Vector 编程', 'Membase / LocalTensor，SIMD 主导', 'RegBase；SIMD 与 SIMT 并存', '按规则连续与离散分支选择执行范式', 2],
      ['AIC/AIV 数据通路', '中间结果较多依赖 GM/L2 中转', '增加 L0C→UB、UB→L1 与 C/V 融合通路', '先画数据流，再决定 Mix 子图与融合边界', 4],
      ['Cube 取数', 'L0A 使用 ZZ 分形', 'L0A 改为 NZ 分形', '重新生成 L0A 切分、地址与 Tiling 参数', 6],
      ['Cube 指令', '支持 int4 Cube、4:2 稀疏与边界绕回', '相关能力移除，MX/FP8 路径增强', '替换旧指令和算法假设，重做精度基线', 5],
      ['低比特', '以传统整数与浮点格式为主', 'HiF8、FP8、MXFP8、MXFP4 与 MicroScaling', '把 scale、layout、搬运与误差控制一起设计', 7],
      ['片上存储', 'UB 192KB；16 bank groups × 3 banks × 4KB', 'UB 256KB；8 bank groups × 2 banks × 16KB', '删除写死容量与错位经验，重新评估 bank 冲突', 8],
      ['通信', 'HCCL 软件调度与通用资源为主', 'CCU、UnifiedBus、URMA 扩展通信执行路径', '同时观察接口语义、硬件落点与通算重叠', 10],
    ];
    renderInspector(`<div class="inspector-content guide-document-content">
      ${renderDocumentHeader({
        path: ['A5 架构与编程指南', 'A5 架构'],
        title: '代际差异速查',
        summary: '从执行范式、数据通路、片上存储、低比特与通信五条主线，快速判断旧代实现需要保留、轻改还是重写。',
        highlights: SUMMARY_HIGHLIGHTS['generation-diff'],
      })}
      ${renderContextCard(`<dl class="meta-grid">
        <div class="meta-row"><dt>对照范围</dt><dd>A5 / Ascend 950 / DAV_3510 与 A2/A3 / Ascend 910B / DAV_2201</dd></div>
        <div class="meta-row"><dt>使用方式</dt><dd>先定位差异维度，再进入对应变化项查看指令、路径与迁移动作</dd></div>
        <div class="meta-row"><dt>文档定位</dt><dd>代际差异检索入口</dd></div>
      </dl>`)}
      ${section('关键代际差异', `<div class="comparison-table-wrap"><table class="comparison-table generation-diff-table">
        <thead><tr><th scope="col">维度</th><th scope="col">A2/A3</th><th scope="col">A5</th><th scope="col">开发影响</th></tr></thead>
        <tbody>${rows.map(([dimension, legacy, a5, impact, categoryId]) => `<tr>
          <td><button class="migration-map-link generation-diff-link" type="button" data-category-jump="${categoryId}">${escape(dimension)}</button></td>
          <td>${escape(legacy)}</td><td>${escape(a5)}</td><td>${escape(impact)}</td>
        </tr>`).join('')}</tbody>
      </table></div>`)}
    </div>`);
  }

  function renderGuideDocument(document) {
    if (document.id === 'architecture-cognition') renderArchitectureCognition();
    else if (document.id === 'guide-development') renderDevelopmentGuide();
    else if (document.scenarioKey) renderDevelopmentScenarioGuide(document);
    else if (document.id === 'generation-diff') renderGenerationDiff();
    else renderMigrationChecklist();
  }

  function renderCategoryInspector(item) {
    const context = CATEGORY_CONTEXT[item.id];
    const comparison = CATEGORY_COMPARISONS[item.id];
    const related = item.related.map((id) => {
      const flow = FLOWS.find((entry) => entry.id === id);
      return flow ? `<button class="btn btn-ghost related-flow" type="button" data-related-flow="${flow.id}"><span>→ ${escape(flow.title)}</span><span class="related-flow-action">跳转查看</span></button>` : '';
    }).join('');
    const scenarioTitle = MIGRATION_SCENARIOS.find((scenario) => scenario.key === item.scenario)?.title || item.scenario;
    renderInspector(`<div class="inspector-content">
      ${renderDocumentHeader({
        path: ['A5 架构与编程指南', `迁移场景 ${item.scenario}：${scenarioTitle}`],
        title: item.title,
        summary: item.essence,
        highlights: SUMMARY_HIGHLIGHTS[`category-${item.id}`],
      })}
      ${renderContextCard(`<dl class="meta-grid">
        <div class="meta-row"><dt>变化来源</dt><dd>${escape(context.actor)}</dd></div>
        <div class="meta-row"><dt>设计目标</dt><dd>${escape(context.goal)}</dd></div>
        <div class="meta-row"><dt>直接影响</dt><dd>${escape(context.impact)}</dd></div>
        <div class="meta-row"><dt>迁移关注</dt><dd><span class="mini-badge">${escape(item.dimension)}</span></dd></div>
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
      ${renderDocumentHeader({ path: ['跨代际执行流对比'], title: flow.title, summary: flow.summary })}
      ${section('执行路径', `<div class="inspector-card"><strong>路径</strong><small>${escape(flow.path)}</small></div>`)}
      ${section('上下文', `<dl class="meta-grid">${visibleMeta.map(([key, value]) => `<div class="meta-row"><dt>${escape(key)}</dt><dd>${escape(value)}</dd></div>`).join('')}</dl>`)}
      ${section('执行步骤', `<ol class="step-list">${flow.steps.map((step, index) => `<li><button class="step-button${state.activeStep === index ? ' is-selected' : ''}" type="button" data-step="${index}"><span class="step-number">${String(index + 1).padStart(2, '0')}</span><span class="step-copy"><strong>${escape(step.label)}</strong><small>${escape(step.text)}</small></span></button></li>`).join('')}</ol>`)}
    </div>`);
    syncPlayback();
  }

  function revealCategoryInNavigation(id) {
    const button = $(`[data-category-id="${Number(id)}"]`);
    if (!button) return;
    const branch = button.closest('[data-tree-branch]');
    const toggle = branch?.querySelector(':scope > [data-tree-toggle]');
    const group = branch?.querySelector(':scope > [role="group"]');
    if (branch?.classList.contains('is-collapsed')) {
      branch.classList.remove('is-collapsed');
      toggle?.setAttribute('aria-expanded', 'true');
      if (group) group.hidden = false;
    }
    requestAnimationFrame(() => requestAnimationFrame(() => {
      button.scrollIntoView({ block: 'center', inline: 'nearest', behavior: 'smooth' });
    }));
  }

  function selectCategory(id, { revealInNavigation = false } = {}) {
    stopPlayback();
    state.diff = false;
    state.selectedId = Number(id);
    state.guideFocusId = null;
    state.activeStep = -1;
    const item = CATEGORIES.find((entry) => entry.id === state.selectedId);
    if (!item) return;
    $$('[data-category-id]').forEach((button) => button.classList.toggle('is-selected', Number(button.dataset.categoryId) === item.id));
    $$('[data-scenario-overview]').forEach((button) => button.classList.remove('is-selected'));
    $('[data-migration-overview]')?.classList.remove('is-selected');
    $('[data-terminology-document]')?.classList.remove('is-selected');
    $$('[data-guide-document]').forEach((button) => button.classList.remove('is-selected'));
    syncEntitySelectionAccessibility();
    syncGenerationCompare();
    renderCategoryInspector(item);
    if (revealInNavigation) revealCategoryInNavigation(item.id);
    if (state.arch !== item.arch) setArch(item.arch);
    else { applyCurrentFocus(); scheduleHardwareFit(); }
  }

  function selectFlow(id, { syncArch = true } = {}) {
    stopPlayback();
    state.diff = false;
    state.selectedId = id;
    state.guideFocusId = null;
    state.activeStep = -1;
    const item = FLOWS.find((entry) => entry.id === id);
    if (!item) return;
    $$('[data-flow-id]').forEach((button) => button.classList.toggle('is-selected', button.dataset.flowId === item.id));
    syncEntitySelectionAccessibility();
    syncGenerationCompare();
    renderFlowInspector(item);
    if (syncArch && state.arch !== item.arch) setArch(item.arch);
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
    state.guideFocusId = null;
    state.activeStep = -1;
    state.diff = false;
    $('#navigationMode').value = mode;
    $('#migrationExplorer').hidden = mode !== 'migration';
    $('#flowExplorer').hidden = mode !== 'flow';
    $('#playbackMount').hidden = mode !== 'flow';
    $$('.entity-button').forEach((button) => button.classList.remove('is-selected'));
    if (mode === 'migration') {
      selectGuideDocument('architecture-cognition');
    } else {
      if (state.arch !== 'ascend950b') setArch('ascend950b');
      selectFlow(FLOWS[0].id, { syncArch: false });
    }
    syncPlayback();
  }

  function showScenario(key) {
    const scenario = SCENARIOS[key];
    if (!scenario) return;
    state.selectedId = null;
    state.guideFocusId = null;
    state.diff = false;
    $$('[data-scenario-overview]').forEach((button) => button.classList.toggle('is-selected', button.dataset.scenarioOverview === key));
    $$('[data-category-id]').forEach((button) => button.classList.remove('is-selected'));
    $('[data-migration-overview]')?.classList.remove('is-selected');
    $('[data-terminology-document]')?.classList.remove('is-selected');
    $$('[data-guide-document]').forEach((button) => button.classList.remove('is-selected'));
    syncEntitySelectionAccessibility();
    syncGenerationCompare();
    renderInspector(`<div class="inspector-content">
      ${renderDocumentHeader({
        path: ['A5 架构与编程指南', `迁移场景 ${key}：${scenario.title}`],
        title: '场景概述',
        summary: scenario.tagline,
        highlights: SUMMARY_HIGHLIGHTS[`scenario-${key}`],
      })}
      ${section('识别特征（典型）', `<ul class="signal-list">${scenario.features.map((item) => `<li>${escape(item)}</li>`).join('')}</ul>`)}
      ${section('处理方式', `<ul class="action-list">${scenario.actions.map((item) => `<li>${escape(item)}</li>`).join('')}</ul>`)}
    </div>`);
    focusHardware();
    scheduleHardwareFit();
  }

  function showTerminology() {
    state.selectedId = null;
    state.guideFocusId = null;
    state.diff = false;
    $$('[data-scenario-overview], [data-category-id]').forEach((button) => button.classList.remove('is-selected'));
    $('[data-migration-overview]')?.classList.remove('is-selected');
    $$('[data-guide-document]').forEach((button) => button.classList.remove('is-selected'));
    $('[data-terminology-document]')?.classList.add('is-selected');
    syncEntitySelectionAccessibility();
    syncGenerationCompare();
    const dictionaryRows = [
      ['A5', '平台代际', 'Ascend 950 所属的平台代际；用于描述开发策略、能力与迁移方向。'],
      ['A2/A3', '平台代际组', '迁移语境中的旧平台集合，不等同于单一产品型号。'],
      ['Ascend 950', '产品系列', 'A5 代际的产品系列；950PR、950DT 是同系列的不同产品形态。'],
      ['Ascend 910B', '产品型号', 'A2/A3 范围内的当前硬件对照型号，不代表整个旧平台集合。'],
      ['DAV_3510 / dav-3510', '软件架构标识', '前者常见于源码枚举，后者常见于 CMake 或编译参数。'],
      ['DAV_2201 / dav-2201', '软件架构标识', '当前旧平台样例使用的软件目标标识；正确编号是 2201，不是 2210。'],
      ['AI Core', '计算组合对象', '每个 AI Core 由 1 个 AIC / Cube Core 与 2 个 AIV / Vector Core 组成。'],
      ['AIC / Cube Core', '计算单元', '矩阵计算侧，围绕 L1、L0A/L0B/L0C、Cube、MTE 与 FixPipe 工作。'],
      ['AIV / Vector Core', '计算单元', '向量计算侧，围绕 Unified Buffer、Vector、SIMD/SIMT 与 MTE 工作。'],
      ['UB / Unified Buffer', '片上存储', 'AIV 侧的片上缓冲区，承接 GM/L2 与 Vector/Reg 之间的数据。'],
      ['UnifiedBus', '互联对象', '950 的 IO 与互联能力；名称与 Unified Buffer 相似，但不是同一对象。'],
      ['RegBase', '编程模型', '以 RegTensor、MaskReg、AddrReg 和显式 Load/Store 组织向量计算。'],
      ['NDDMA', '数据搬运', '覆盖 transpose、stride、broadcast、slice 等复杂搬运与格式转换。'],
      ['CCU', '通信引擎', '面向集合通信卸载与通算协同的专用硬件对象。'],
    ];
    renderInspector(`<div class="inspector-content terminology-content">
      ${renderDocumentHeader({
        path: ['A5 架构与编程指南', 'A5 架构'],
        title: '术语表',
        summary: '统一架构、硬件对象、编程模型与软件标识的含义，让不同章节中的称谓可以稳定对应。',
        highlights: SUMMARY_HIGHLIGHTS.terminology,
      })}
      ${renderContextCard(`<dl class="meta-grid">
        <div class="meta-row"><dt>代际主称谓</dt><dd>A5 与 A2/A3</dd></div>
        <div class="meta-row"><dt>产品主称谓</dt><dd>Ascend 950 与 Ascend 910B</dd></div>
        <div class="meta-row"><dt>软件标识</dt><dd>DAV_3510 / dav-3510 与 DAV_2201 / dav-2201</dd></div>
      </dl>`)}
      ${section('术语字典', `<div class="comparison-table-wrap">
        <table class="comparison-table terminology-dictionary-table">
          <thead><tr><th scope="col">术语</th><th scope="col">类型</th><th scope="col">定义与边界</th></tr></thead>
          <tbody>${dictionaryRows.map((row) => `<tr>${row.map((cell) => `<td>${escape(cell)}</td>`).join('')}</tr>`).join('')}</tbody>
        </table>
      </div>`)}
    </div>`);
    focusHardware();
    scheduleHardwareFit();
  }

  function showMigrationOverview() {
    state.selectedId = null;
    state.guideFocusId = null;
    state.diff = false;
    $$('[data-scenario-overview], [data-category-id]').forEach((button) => button.classList.remove('is-selected'));
    $('[data-terminology-document]')?.classList.remove('is-selected');
    $$('[data-guide-document]').forEach((button) => button.classList.remove('is-selected'));
    $('[data-migration-overview]')?.classList.add('is-selected');
    syncEntitySelectionAccessibility();
    syncGenerationCompare();
    const scenarioDefinitions = [
      ['A', 'A5 原生能力兼容 A2/A3', '使用 RegBase、SIMT、低比特或 CCU 等 A5 原生能力；需要兼容 A2/A3 时，为旧代保留独立实现与注册分支。'],
      ['B', '轻量迁移', '实现只依赖通用 SIMD、GM ↔ UB 与简单向量计算；调整代际声明后，重点完成精度与性能验证。'],
      ['C', '深度迁移', '旧实现写死数据通路、矩阵分形、片上存储或已移除指令；需要重画数据流并重建实现。'],
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
    const dimensionRows = MIGRATION_DIMENSIONS.map((dimension) => {
      const relatedItems = CATEGORIES.filter((item) => item.dimension === dimension.label);
      return `<tr>
        <td><span class="mini-badge">${escape(dimension.label)}</span></td>
        <td>${escape(dimension.description)}</td>
        <td>${relatedItems.map((item) => `<button class="migration-map-link migration-dimension-link" type="button" data-category-jump="${item.id}">${String(item.id).padStart(2, '0')} ${escape(item.title)}</button>`).join('')}</td>
      </tr>`;
    }).join('');
    renderInspector(`<div class="inspector-content migration-overview-content">
      ${renderDocumentHeader({
        path: ['A5 架构与编程指南', 'A5 算子开发'],
        title: 'A5 算子迁移概览',
        summary: '三类场景按 A5 代际主导后的常见工作顺序组织：优先处理 A5 原生能力兼容 A2/A3，其次处理可轻量迁移的实现，最后处理必须重写硬件假设的深度迁移。',
        highlights: SUMMARY_HIGHLIGHTS.overview,
      })}
      ${section('三类场景', `<div class="scenario-definition-list">${scenarioDefinitions.map(([key, title, description]) => `
        <article class="inspector-card scenario-definition-card">
          <span class="comparison-marker scenario-key-tag">${key}</span>
          <div><strong>${escape(title)}</strong><p>${escape(description)}</p></div>
        </article>`).join('')}</div>`)}
      ${section('全量变化项与场景映射', `<div class="comparison-table-wrap">
          <table class="comparison-table migration-map-table">
            <thead><tr><th scope="col">变化项</th>${MIGRATION_SCENARIOS.map((scenario) => `<th scope="col">
              <span class="migration-scenario-header" tabindex="0" data-scenario-header="${scenario.key}" data-tooltip="${escape(scenario.title)}" aria-describedby="migrationScenarioTooltip">
                场景 ${scenario.key}
              </span>
            </th>`).join('')}</tr></thead>
            <tbody>${mappingRows}</tbody>
          </table>
        </div>`, `
          <span><i class="migration-map-dot is-primary" aria-hidden="true"></i><b>实心圆：主要场景</b></span>
          <span><i class="migration-map-dot is-secondary" aria-hidden="true"></i><b>空心圆：次要场景</b></span>
          <span><i class="migration-map-empty" aria-hidden="true">—</i><b>短横线：不映射</b></span>
        `)}
      ${section('迁移关注维度', `<p class="section-lead">场景 A / B / C 回答“采用什么迁移策略”；导航与文档基本信息中的 tag 回答“这篇文档主要提醒你检查哪一层”。每篇变化文档只标一个主维度，跨层影响在正文的“直接影响”和“判断信号”中展开。</p>
        <div class="comparison-table-wrap"><table class="comparison-table migration-dimension-table">
          <thead><tr><th scope="col">标签</th><th scope="col">主要检查内容</th><th scope="col">对应变化文档</th></tr></thead>
          <tbody>${dimensionRows}</tbody>
        </table></div>`)}
    </div>`);
    focusHardware();
    scheduleHardwareFit();
  }

  function selectGuideDocument(id) {
    stopPlayback();
    const document = GUIDE_DOCUMENTS.find((entry) => entry.id === id);
    if (!document) return;
    state.selectedId = document.id;
    state.guideFocusId = null;
    state.activeStep = -1;
    state.diff = document.id === 'generation-diff';
    $$('[data-category-id], [data-scenario-overview]').forEach((button) => button.classList.remove('is-selected'));
    $('[data-migration-overview]')?.classList.remove('is-selected');
    $('[data-terminology-document]')?.classList.remove('is-selected');
    $$('[data-guide-document]').forEach((button) => button.classList.toggle('is-selected', button.dataset.guideDocument === document.id));
    syncEntitySelectionAccessibility();
    renderGuideDocument(document);
    if (state.arch !== document.arch) setArch(document.arch);
    else {
      syncGenerationCompare();
      applyCurrentFocus();
      scheduleHardwareFit();
    }
  }

  function selectGuideFocus(id) {
    const focus = GUIDE_FOCUS[id];
    if (!focus || !GUIDE_DOCUMENTS.some((document) => document.id === state.selectedId)) return;
    const shouldClear = state.guideFocusId === id;
    state.guideFocusId = shouldClear ? null : id;
    $$('[data-guide-focus]').forEach((button) => {
      const selected = !shouldClear && button.dataset.guideFocus === id;
      button.classList.toggle('is-selected', selected);
      button.setAttribute('aria-pressed', String(selected));
    });
    if (shouldClear) applyCurrentFocus();
    else focusHardware(focus.selectors || [], focus.routes || []);
    scheduleHardwareFit();
  }

  function syncEntitySelectionAccessibility() {
    $$('.entity-button').forEach((button) => {
      if (button.classList.contains('is-selected')) button.setAttribute('aria-current', 'page');
      else button.removeAttribute('aria-current');
    });
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

  function positionFloatingTooltip(tooltip, target, pointerEvent) {
    if (!tooltip || !target) return;
    const tooltipRect = tooltip.getBoundingClientRect();
    const viewportPadding = 8;
    const gap = 12;
    if (pointerEvent && Number.isFinite(pointerEvent.clientX) && Number.isFinite(pointerEvent.clientY)) {
      let left = pointerEvent.clientX + gap;
      let top = pointerEvent.clientY + gap;
      if (left + tooltipRect.width > window.innerWidth - viewportPadding) left = pointerEvent.clientX - tooltipRect.width - gap;
      if (top + tooltipRect.height > window.innerHeight - viewportPadding) top = pointerEvent.clientY - tooltipRect.height - gap;
      tooltip.style.left = `${Math.max(viewportPadding, left)}px`;
      tooltip.style.top = `${Math.max(viewportPadding, top)}px`;
      return;
    }
    const targetRect = target.getBoundingClientRect();
    const left = Math.min(
      window.innerWidth - tooltipRect.width - viewportPadding,
      Math.max(viewportPadding, targetRect.left),
    );
    const top = Math.min(
      window.innerHeight - tooltipRect.height - viewportPadding,
      targetRect.bottom + gap,
    );
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${Math.max(viewportPadding, top)}px`;
  }

  let activeMigrationScenarioTooltipTarget = null;

  function showMigrationScenarioTooltip(target, pointerEvent) {
    const tooltip = $('#migrationScenarioTooltip');
    const content = target?.dataset.tooltip;
    if (!tooltip || !content) return;
    activeMigrationScenarioTooltipTarget = target;
    tooltip.textContent = content;
    tooltip.hidden = false;
    tooltip.classList.add('is-visible');
    positionFloatingTooltip(tooltip, target, pointerEvent);
  }

  function hideMigrationScenarioTooltip() {
    const tooltip = $('#migrationScenarioTooltip');
    if (!tooltip) return;
    activeMigrationScenarioTooltipTarget = null;
    tooltip.classList.remove('is-visible');
    tooltip.hidden = true;
  }

  let activeNavigationTooltipTarget = null;
  let activeNavigationTooltipOwner = null;

  function hideNavigationDescriptionTooltip() {
    const tooltip = $('#navigationDescriptionTooltip');
    if (!tooltip) return;
    activeNavigationTooltipOwner?.removeAttribute('aria-describedby');
    activeNavigationTooltipTarget = null;
    activeNavigationTooltipOwner = null;
    tooltip.classList.remove('is-visible');
    tooltip.hidden = true;
  }

  function isNavigationTextTruncated(target) {
    return Boolean(target && target.scrollWidth > target.clientWidth + 1);
  }

  function showNavigationDescriptionTooltip(target, pointerEvent) {
    const tooltip = $('#navigationDescriptionTooltip');
    const content = target?.dataset.navTooltip;
    if (!tooltip || !content || !isNavigationTextTruncated(target)) {
      hideNavigationDescriptionTooltip();
      return;
    }
    activeNavigationTooltipOwner?.removeAttribute('aria-describedby');
    activeNavigationTooltipTarget = target;
    activeNavigationTooltipOwner = target.closest('button');
    activeNavigationTooltipOwner?.setAttribute('aria-describedby', 'navigationDescriptionTooltip');
    tooltip.textContent = content;
    tooltip.hidden = false;
    tooltip.classList.add('is-visible');
    positionFloatingTooltip(tooltip, target, pointerEvent);
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
      const terminologyDocument = event.target.closest('[data-terminology-document]');
      if (terminologyDocument) return showTerminology();
      const guideDocument = event.target.closest('[data-guide-document]');
      if (guideDocument) return selectGuideDocument(guideDocument.dataset.guideDocument);
      const guideFocus = event.target.closest('[data-guide-focus]');
      if (guideFocus) return selectGuideFocus(guideFocus.dataset.guideFocus);
      const categoryJump = event.target.closest('[data-category-jump]');
      if (categoryJump) return selectCategory(categoryJump.dataset.categoryJump, { revealInNavigation: true });
      const category = event.target.closest('[data-category-id]');
      if (category) return selectCategory(category.dataset.categoryId);
      const flow = event.target.closest('[data-flow-id]');
      if (flow) return selectFlow(flow.dataset.flowId);
      const related = event.target.closest('[data-related-flow]');
      if (related) { setMode('flow'); selectFlow(related.dataset.relatedFlow); return; }
      const step = event.target.closest('[data-step]');
      if (step) { stopPlayback(); selectStep(step.dataset.step); return; }
    });
    document.addEventListener('pointerover', (event) => {
      const scenarioTarget = event.target.closest?.('[data-scenario-header]');
      if (scenarioTarget) showMigrationScenarioTooltip(scenarioTarget, event);
      const navigationTarget = event.target.closest?.('[data-nav-tooltip]');
      if (navigationTarget) showNavigationDescriptionTooltip(navigationTarget, event);
    });
    document.addEventListener('pointermove', (event) => {
      const scenarioTarget = event.target.closest?.('[data-scenario-header]');
      if (scenarioTarget) {
        if (scenarioTarget === activeMigrationScenarioTooltipTarget) positionFloatingTooltip($('#migrationScenarioTooltip'), scenarioTarget, event);
        else showMigrationScenarioTooltip(scenarioTarget, event);
      }
      const navigationTarget = event.target.closest?.('[data-nav-tooltip]');
      if (navigationTarget) {
        if (navigationTarget === activeNavigationTooltipTarget) positionFloatingTooltip($('#navigationDescriptionTooltip'), navigationTarget, event);
        else showNavigationDescriptionTooltip(navigationTarget, event);
      }
    });
    document.addEventListener('pointerout', (event) => {
      const scenarioTarget = event.target.closest?.('[data-scenario-header]');
      if (scenarioTarget && !scenarioTarget.contains(event.relatedTarget)) hideMigrationScenarioTooltip();
      const navigationTarget = event.target.closest?.('[data-nav-tooltip]');
      if (navigationTarget && !navigationTarget.contains(event.relatedTarget)) hideNavigationDescriptionTooltip();
    });
    document.addEventListener('focusin', (event) => {
      const scenarioTarget = event.target.closest?.('[data-scenario-header]');
      if (scenarioTarget) showMigrationScenarioTooltip(scenarioTarget);
      const navigationOwner = event.target.closest?.('.entity-button, .migration-tree-toggle');
      const navigationTarget = [...(navigationOwner?.querySelectorAll('[data-nav-tooltip]') || [])].find(isNavigationTextTruncated);
      if (navigationTarget) showNavigationDescriptionTooltip(navigationTarget);
    });
    document.addEventListener('focusout', (event) => {
      if (event.target.closest?.('[data-scenario-header]')) hideMigrationScenarioTooltip();
      if (event.target.closest?.('.entity-button, .migration-tree-toggle')) hideNavigationDescriptionTooltip();
    });
    document.addEventListener('scroll', () => {
      hideMigrationScenarioTooltip();
      hideNavigationDescriptionTooltip();
    }, true);
    window.addEventListener('resize', () => {
      hideMigrationScenarioTooltip();
      hideNavigationDescriptionTooltip();
    });
    $('#navigationMode').addEventListener('change', (event) => {
      setMode(event.target.value);
      event.target.blur();
    });
    $$('[data-arch-id]').forEach((button) => button.addEventListener('click', () => setArch(button.dataset.archId)));
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
      if (!hidden) scheduleHardwareFit();
    });
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
  setMode('migration');
  window.addEventListener('load', scheduleHardwareFit, { once: true });
  document.fonts?.ready?.then(scheduleHardwareFit);
})();
