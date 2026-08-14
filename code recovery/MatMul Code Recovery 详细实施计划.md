# MatMul Code Recovery 实施计划

> 当前版本：0.1  
> 当前状态：Step 0 · 文档建模中  
> 案例：cann-samples / Samples/0_Introduction/matmul  
> 产品性质：静态 Code Recovery 体验模型，不代表已编译、已上板或已 profiling

## 1. 文档定位

本文档是 MatMul Code Recovery 子页面的内容实施计划。它参考现有 Conv + Bias + ReLU Code Recovery 实施计划，但不复用 Conv 的具体 Tensor、Tiling、Event 或硬件结论。

目标是先建立 MatMul 的内容真值，再进入 Trace、UI 和 Pattern 接入：

~~~
源码事实
→ 数学与 Tiling 模型
→ Instruction / Event 恢复
→ Tensor / Memory 生命周期
→ Hardware Participation
→ UI 交互与验证
~~~

本案例以目标仓库的单文件 main.asc 为权威源码。该文件同时包含 Kernel、Host 和辅助函数，因此产品页面保持一个物理源码文件，通过源码行范围和角色标签区分 Kernel / Host。

## 2. 权威输入

### 2.1 目标样例

- 仓库：cann-samples
- 路径：Samples/0_Introduction/matmul
- 目标源码：src/matmul/main.asc
- 构建文件：src/matmul/CMakeLists.txt
- 使用说明：src/matmul/README.md
- 数据生成：src/matmul/scripts/gen_data.py
- 精度校验：src/matmul/scripts/verify_result.py
- 性能脚本：src/matmul/scripts/profile_matmul.py

### 2.2 当前工作区材料

- code recovery/Conv + Bias + ReLU Code Recovery 详细实施计划.md
- code recovery/Conv2D + Bias + ReLU 执行过程.md
- code recovery/Conv2D + Bias + ReLU tensor数据.md
- code recovery/spec.md
- code recovery/CLAUDE.md
- Kernel Execution Model v0.1.md

### 2.3 证据优先级

1. 当前目标样例 src/matmul/main.asc 的源码事实；
2. 目标样例 README、CMake 和验证脚本；
3. 当前三份 Conv Code Recovery 文档提供的组织结构和表达方式；
4. Ascend C API 语义与 API 文档；
5. 运行、dump 或 profiling 数据；
6. 产品解释和 UX 推断。

## 3. 案例边界

### 3.1 数学契约

~~~
A[M,K] × B[K,N] → C[M,N]
~~~

当前首个固定演示上下文：

| 参数 | 值 |
| --- | ---: |
| M | 1024 |
| K | 2048 |
| N | 4096 |
| Kernel T | bfloat16 |
| GM A | ND |
| GM B | ND |
| GM C | ND |
| NPU ARCH | dav-3510 |

源码中的 Kernel 是模板函数，但当前 Host 以 bfloat16_t 实例化。文档默认使用 BF16 固定演示上下文，同时保留 T 模板带来的参数化说明。

### 3.2 非目标

- 不证明当前源码已通过目标 CANN 版本编译；
- 不证明当前机器具有 dav-3510 设备；
- 不声称 block 数等于固定硬件核数；
- 不声称 Tiling 是自动生成或性能最优；
- 不生成真实 duration、stall、overlap、带宽或利用率；
- 不把逻辑 NZ/ZN 视图自动解释成数学转置；
- 本次还原只覆盖 M、K、N 分别可被 baseM、kL1、baseN 整除的完整 tile 场景；
- 不生成不存在的 Tensor Data Dump 数值；
- 不把 CopyL0C2GM 直接等同于源码中显式 Fixpipe API。

## 4. MatMul 执行上下文

### 4.1 Kernel 参数与分块

源码中的固定参数：

| 参数 | 源码语义 |
| --- | --- |
| baseM | L0 计算视图的 M 基础分块，256 |
| baseN | L0 计算视图的 N 基础分块，256 |
| baseK | 256 / sizeof(T)，BF16 时为 128 |
| kL1 | 1024 / sizeof(T)，BF16 时为 512 |
| mTileNum | M / baseM |
| nTileNum | N / baseN |
| tileNum | mTileNum × nTileNum |
| kL1TileNum | K / kL1 |
| kL0IterNum | kL1 / baseK |

对 M=1024、K=2048、N=4096、BF16：

~~~
baseM = 256
baseN = 256
baseK = 128
kL1 = 512

mTileNum = 4
nTileNum = 16
tileNum = 64
kL1TileNum = 4
kL0IterNum = 4
~~~

每个完整输出 tile 需要 4 个 L1 K tile，每个 L1 K tile 需要 4 个 L0 K tile，因此每个完整输出 tile 需要 16 次 Mmad。

### 4.2 Block 到输出 tile

源码使用：

~~~
for (tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum)
~~~

映射关系：

~~~
mTileIdx = tileIdx / nTileNum
nTileIdx = tileIdx % nTileNum

mStart = mTileIdx × baseM
nStart = nTileIdx × baseN
~~~

Block 负责的是输出 tile 调度，不应直接理解为固定的设备物理核编号。blockNum 来自 GetBlockNum，Host 启动时的 numBlocks 来自 GetCoreNumAic。

### 4.3 两层 K 循环

外层 iter0：

- 处理 GM → L1 的 K 分片；
- BF16 完整场景下每次为 K=512；

内层 iter1：

- 处理 L1 → L0 的 K 分片；
- BF16 完整场景下每次为 K=128；
- 先进入 L0A/L0B，再执行 Mmad。

### 4.4 执行单元与动作

| 角色 | 当前案例中的动作 |
| --- | --- |
| Host / Runtime | 参数解析、ACL 初始化、分配、拷贝、Kernel launch、结果回拷 |
| Scalar / Kernel control | Tiling 计算、Slice 和循环控制 |
| MTE2 | GM → L1 |
| MTE1 | L1 → L0A/L0B |
| Cube | Mmad |
| Output path | L0C → GM 的 CopyL0C2GM |
| Event | HardEvent SetFlag / WaitFlag |

## 5. 内容真值文档的分工

### 5.1 本实施计划

回答：

- 先做什么；
- 每一阶段依赖什么；
- 每一阶段如何验收；
- 哪些事实允许进入 UI；
- 哪些事实必须标记为 derived 或 unverified。

### 5.2 MatMul 执行过程

回答：

- 用户如何从整体理解 MatMul；
- 一个 block 如何选择输出 tile；
- GM、L1、L0A、L0B、L0C 如何连接；
- 外层和内层 K 循环如何工作；
- Event 如何形成依赖；
- Mmad 初始化和累加如何区别；
- 输出如何写回。

### 5.3 MatMul tensor 数据

回答：

- 每个阶段有哪些 Tensor；
- 每个 Tensor 的逻辑 Shape、物理 Layout、dtype、bytes；
- 当前 tile 的坐标范围；
- Tensor 在不同 Memory tier 的状态；
- 哪些内容是实际数据，哪些只是 shape / metadata；
- 没有 dump 时如何降级。

## 6. 实施阶段

### 阶段 0：内容真值文档

交付：

- MatMul Code Recovery 详细实施计划.md
- MatMul 执行过程.md
- MatMul tensor数据.md

验收：

- 三份文档结构与 Conv 对齐；
- 所有阶段都能回链到 src/matmul/main.asc；
- 公式、shape、layout、Event 名称一致；
- confirmed / derived / unverified 边界清楚；
- 不引用 Conv 独有的 Bias、LoadData3D、ReLU 或固定 8 核结论。

### 阶段 1：正式 Trace Contract

交付：

- 完整 schema 0.3 MatMul trace；
- Source、Stages、Steps、Tensors、Buffers、Events、Evidence；
- 首个固定上下文的确定性派生结果。

验收：

- 通过 schema 校验；
- 所有 sourceRef 有效；
- 执行阶段可覆盖文档中的完整主路径；
- 不依赖页面脚本中的隐藏常量解释关键事实。

### 阶段 2：Source 双向定位

交付：

- 单文件 src/matmul/main.asc；
- Host / Kernel 角色过滤；
- Source ↔ Instruction 双向定位。

验收：

- 点击源码可定位步骤；
- 点击步骤可定位源码；
- 切换角色不丢失执行上下文；
- 无效引用不产生虚假高亮。

### 阶段 3：Instruction Flow

交付：

- Sequence、Loop、Event、Branch 语义；
- Execution Dock；
- Playback；
- Timeline unavailable 状态。

验收：

- 能区分 Configure、Move、Compute、Sync、Store；
- 能展开和折叠两层 K loop；
- Mmad 初始化与累加语义正确；
- Playback、Source、Tensor、Hardware 同步。

### 阶段 4：Tensor / Memory / Matrix

交付：

- Matrix Canvas；
- Tensor Title；
- Memory Map；
- Tensor Lifecycle；
- Layout 和 dtype 变化。

验收：

- A/B/C 和 A1/B1/A2/B2/CO1 可定位；
- 当前 tile 和 K slice 状态正确；
- CO1 明确为 FP32 accumulator；
- 无 dump 时不显示猜测数值。

### 阶段 5：Hardware Participation

交付：

- GM、L1、L0A、L0B、L0C；
- MTE2、MTE1、Cube、Output path；
- Event dependency path。

验收：

- 当前 instruction 只高亮相关单元；
- Event 方向与 Source / Dock 一致；
- 不显示虚构 occupancy、stall 或 duration。

### 阶段 6：回归与交付

交付：

- 固定整除 shape 的确定性验证；
- 页面响应式与可访问性；
- 文档、fixture、UI 一致性回归。

验收：

- MatMul 子页面可独立运行；
- Conv 页面行为不受影响；
- 所有关键事实都有来源或证据等级；
- 通过 HTTP 页面验证；
- 无控制台错误。

当前 Step 6 使用固定 fixture 驱动的确定性案例回归，不代表设备实测：

| 案例 | Shape | 输出 tile | L1/L0 K 分片 | 每输出 tile 的 Mmad |
| --- | --- | --- | --- | --- |
| Default | 1024 × 2048 × 4096 | 256 × 256 | 512 → 128/128/128/128 | 16 |

以上数值由固定 shape 和基础分块参数派生；真实编译、运行、profiling 与 Tensor Data Dump 仍为 unverified。

## 7. Step 0 的验收方式

本轮只验收三份文档，不验收 MatMul 页面：

1. 内容结构是否符合 Conv 三份文档的使用方式；
2. MatMul 的执行模型是否正确；
3. M/K/N、分块、Memory、Event 是否符合目标源码；
4. 哪些地方需要补充或改写。

Step 0 通过后，才开始 Step 1 的 Trace Contract。

## 8. 当前 Step 1 的验收方式

本轮只验收 Trace Contract 与页面数据接入，不验收 Instruction Flow 播放、Tensor Data Dump 或真实 NPU 运行结果：

1. `data/fixtures/matmul_cann_samples.trace.json` 为单一合法 JSON 根对象，并通过 `data/schemas/trace.schema.json` 的 schema 0.3 校验；
2. Fixture 同时包含 `Source`、`Stages`、`Steps`、`Tensors`、`Buffers`、`Events`、`Evidence`；
3. 所有 `sourceRefs` 的 `fileId` 与行号都能回链到 `src/matmul/main.asc`；
4. 页面从 `src/matmul/main.asc` 读取源码，从正式 Fixture 读取内存层级、步骤和固定上下文；
5. 固定上下文的派生事实可复核：M=1024、K=2048、N=4096，baseM/N/K=256/256/128，输出 tile=64，单输出 tile 的 Mmad=16；
6. profiling、运行结果和 Tensor Data Dump 继续标记为 unavailable / unverified，不把静态恢复结果冒充真实执行事实。

Step 1 通过后，才开始 Step 2 的 Source 双向定位。
