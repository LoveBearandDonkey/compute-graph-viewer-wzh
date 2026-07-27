# Conv + Bias + ReLU Code Recovery 详细实施计划

## 1. 目标与交付边界

将 [code recovery](</Users/songchenfei/Documents/ascend c/pto_compute-graph-viewer/code recovery>) 收敛为单一的 `Conv2D + Bias + ReLU` 静态 Code Recovery 专题，复用现有三栏 PTO Workbench：

- 左栏：源码与证据定位。
- 中栏：Conv tiling、Load3D 滑窗、Cube tile、逻辑执行序列。
- 右栏：Ascend 910B memory architecture 与当前数据路径。
- 播放器：按源码逻辑顺序回放，不表达真实耗时。

首版只展示：

- `confirmed`：源码或演示输入直接确认。
- `inferred`：从控制流、命名和 API 语义进行的静态推断。
- `unknown`：当前参考文件无法确认的事实。

不展示真实 duration、overlap、stall、吞吐或 profiling 结论；页面常驻显示“Static implementation model · no runtime timing”。

实施开始时，先将本计划落盘为 `code recovery/PLAN.md`。

## 2. 数据模型与可信度设计

### 演示上下文

由于参考文件没有真实 host tiling，使用一组明确标为“演示输入，非源码恢复”的固定 context：

- `batch=1`
- `Ci=16, Hi=8, Wi=8`
- `Co=32, Kh=3, Kw=3`
- `stride=1, pad=1, dilation=1`
- 派生：`Ho=8, Wo=8`
- Cube 映射：`M=Ho×Wo=64`、`K=Ci×Kh×Kw=144`、`N=Co=32`
- 演示 tile：`tileM=32, tileK=48, tileN=16`
- 派生循环：M tile 2 个、N tile 2 个、K reduction 3 次

这些数值只用于让视图可播放；不得标成从 `tilingGm`、host tiling 或真实运行恢复的结果。

### Trace schema 最小扩展

在现有 trace fixture 结构上增加：

- `execution.mode = "logical-order"`，同时记录 `durationStatus = "unknown"`。
- stage、step、memory object 增加：
  - `objectId`：跨源码、tiling、timeline、memory view 使用的稳定身份。
  - `evidenceKind`：`confirmed | inferred | unknown`。
  - `sourceRefs`：一到多个源码行和 symbol。
  - `applicability`：演示 shape、目标 SoC、CANN 版本及限制。
- `tensorViewport.kind = "conv2d-cube"`，包含：
  - feature-map window；
  - output spatial tile；
  - channel range；
  - 当前 K slice；
  - A2/B2/CO1 矩阵 tile。
- unknown 对象必须携带 `missingEvidence`，说明需要 host tiling、函数实现、目标头文件或 profiling 中的哪一种证据。

### 必须暴露的未解析事实

不能用 mock 补齐以下缺口：

- `FmapA1Elements()`、`WeightB1Elements()` 返回 0，真实 L1 分配未知。
- `outputTileIndex` 与 block/core 的映射没有实现。
- `CopyFeatureTileGmToA1`、`CopyFilterTileGmToB1`、`CopyBiasGmToC1`、`CopyOutputTileC1ToGm` 只有声明。
- `LoadFilterToL0B` 没有函数体。
- Load3D 的 pad、start position、extension 仍待目标版本确认。
- Bias 已搬入 C1，Mmad 却读取 `biasC2`，源码中缺少明确的 `C1→C2` 转换/搬运；UI 必须显示为“断开的依赖/待补全路径”，不能自行补一条连线。
- Cube epilogue 的 `reluPre`、量化字段及 C1→GM 路径标为“代码存在、目标 API 行为待验证”。
- 没有运行证据，因此阶段长度和硬件并发全部未知。

## 3. 页面与交互改造

### PTO Shell

保留现有 dark-mode `ide-frame`、可拖拽 `workbench-shell`、activity rail、inspector drawer、status strip 和 `floating-playback-control`。

页面身份改为：

- 标题：`Ascend Code Recovery`
- 副标题：`Conv2D · Static Implementation Model`
- 默认架构：`Ascend 910B`
- 删除旧 Add/MatMul/Fusion 样例切换入口；旧 fixture 文件暂时保留但不注册、不加载。
- 使用目录专属 split/localStorage key，避免读取原 tiling 页布局状态。

### 左栏：Source & Evidence

- 把参考源码复制为 `data/sources/conv_bias_relu_reference.asc`，完整显示真实行号。
- 当前 step 同步高亮相关源码行。
- 行级标记只表达三类证据，不用颜色暗示错误严重度。
- 点击源码行打开 inspector，展示：
  - 对应 stage、buffer、tile；
  - 代码确认事实；
  - 静态推断；
  - unresolved dependency；
  - 所需验证材料。
- 特别标出四类关键位置：tiling struct、LocalTensor 分配、K loop/Mmad、epilogue/未实现 helper。

### 中栏：Conv Execution Visual

替换误导性的 “Tensor 3D Viewport”，标题改为 `Conv Execution Model`，采用联动的二维视图：

1. **Feature Map / Load3D**
   - 展示 `Hi×Wi` 空间网格。
   - 高亮当前 output position 对应的 `Kh×Kw` receptive field。
   - padding 区使用明确的空值样式。
   - channel 不画成空间第三轴；以 `Ci range / C0 group` 标签表达。

2. **Output Tile**
   - 展示 `Ho×Wo` 空间网格和当前 `tileM` 覆盖。
   - 单独显示当前 `Co` 范围，对应 Cube N tile。
   - 显示 `M=Ho×Wo` 是空间位置 flatten，不是原始 tensor rank。

3. **Cube Tile Lens**
   - A2：`tileM×currentK`
   - B2：`currentK×tileN`
   - CO1：`tileM×tileN`
   - K 仅显示为 reduction slice/progress，不把 `M×N×K` 称为三维 tensor。
   - 首个 K step 显示 Bias initialization；后续显示 accumulation。
   - Bias C1→C2 缺口用 unknown connection 表达。

4. **Logical Execution Sequence**
   - 标题明确写 `Order only · duration unavailable`。
   - 使用等宽离散步骤，不设置时间刻度。
   - 调用共享 `swimlane-task` 的单段 task bar renderer 和 colormap，不在业务代码中重写 task bar 视觉规则。
   - 播放器驱动步骤选择；步骤宽度不代表耗时。

### 逻辑回放步骤

固定为以下源码顺序：

1. 初始化 GM tensor views。
2. 分配 A1/B1/C1/C2/A2/B2/CO1/outC1。
3. Feature、Filter、Bias 从 GM 进入本地输入区。
4. `MTE2 → MTE1` 同步。
5. K0：Load3D feature window 到 A2，同时准备 B2。
6. `MTE1 → M` 同步。
7. K0 Mmad：以 Bias 初始化 CO1。
8. K1 Mmad：累加部分和。
9. K2 Mmad：完成 K reduction。
10. `M → FIX` 同步。
11. CO1→C1：FP32→FP16，并请求 ReLU epilogue。
12. C1→GM：helper 存在但实现未知。

每步同时更新源码、feature/output tile、Cube lens、memory path、status strip；只有用户主动选择对象时才打开 inspector。

### 右栏：Memory Architecture

- 使用 `hardware-architecture-viewport` + `memory-architecture-layout` 的 Ascend 910B preset。
- 复用 pattern 的 path focus、buffer occupancy、pan、zoom 和 iframe message 协议。
- 映射路径：
  - Feature GM → L1/A1 → L0A/A2
  - Filter GM → L1/B1 → L0B/B2
  - Bias GM → C1 → C2（最后一段 unknown）
  - A2 + B2 + Bias → Cube → L0C/CO1
  - L0C → C1 → Output GM
- 共享架构图不存在精确 C1/C2 节点时，只高亮其可确认的上层硬件区域；精细 C1/C2 状态留在 on-chip lens，不在架构图中虚构硬件节点。
- 默认不选中节点、不 dim 全图；path focus 只随用户选择或播放步骤出现。

## 4. 实施顺序

1. **文档与数据基线**
   - 写入 `PLAN.md`。
   - 复制 Conv 源码到 fixture source。
   - 创建单一 Conv trace fixture。
   - 扩展 schema 并注册证据字段、logical-order mode 和 Conv viewport payload。

2. **应用模型**
   - 将 fixture registry 收敛为 Conv 单案例。
   - 新增 `conv2d-cube` visual-state derivation。
   - 建立统一 stable object id，确保 source、step、tile、buffer、architecture 使用同一身份。
   - 删除运行路径中对旧 vector/cube/fusion dispatch 的依赖；旧代码可暂留为未调用参考，第二轮再清理。

3. **可视化**
   - 将中心 canvas 改为 Feature Window + Output Tile 的二维联动视图。
   - 改造 on-chip lens 为 A2/B2/CO1/Bias。
   - 将 timeline 改为无 duration 的逻辑步骤。
   - 将架构 preset 切至 910B，并接入 Conv buffer/path payload。

4. **证据与 Inspector**
   - 给所有可见对象增加 evidence badge。
   - Inspector 按“正在发生什么 / 代码证据 / 静态推断 / 未知 / 下一步验证”组织。
   - 把 Bias C1→C2 缺口、0 长度 L1 allocation、未实现 helper 作为首版的三项核心静态发现。

5. **样式与文档收尾**
   - 保留 PTO tokens/patterns，不创建私有 button、card、badge 或 architecture 样式。
   - 只允许 Conv 网格、tile、padding、reduction 状态使用 data-viz 颜色。
   - 更新复制目录中的 README/CLAUDE 运行入口，去除过期 `/Users/yin/pto` 路径。
   - 修改 JS/CSS 后更新 `index.html` cache-busting query。

预计首版完整实施与验证工作量：约 3–5 个工作日；不包含 host tiling、编译、上板 correctness 和 profiling。

## 5. 测试与验收

### 数据一致性

- Fixture 可通过 JSON schema 校验。
- 每个 `sourceRef` 都能定位到真实源码行。
- 每个 step 的 stage、objectId、memory region 均存在。
- `Ho/Wo/M/K/N`、tile 数、K loop 数和尾块公式计算正确。
- 不允许出现没有 evidenceKind 的可见对象。

### 交互验收

- 首次打开只加载 Conv 案例。
- 前进、后退、播放和 scrubber 都能稳定驱动 12 个逻辑步骤。
- 每一步同步更新源码、二维 tiling、Cube lens、architecture focus 和状态栏。
- 点击 source line、tile、timeline step、buffer 后，inspector 展示同一个 stable object。
- Explorer、Inspector、zoom、fit、拖拽和 pane resize 可重复开关并正确持久化。

### 语义验收

- 页面中不存在 `Tensor 3D Viewport` 或把 K 描述为输出 tensor 第三维的文案。
- 清楚区分语义 tensor：
  - `X[N,Ci,Hi,Wi]`
  - `W[Co,Ci,Kh,Kw]`
  - `Y[N,Co,Ho,Wo]`
- 清楚解释 Cube execution domain：
  - `M=Ho×Wo`
  - `K=Ci×Kh×Kw`
  - `N=Co`
- timeline 不出现毫秒、cycle、比例 duration 或 overlap。
- 未实现的函数和未验证 API 不显示为完整、确定的数据路径。
- Bias C1→C2 缺口始终可追溯到源码。

### 视觉与兼容验收

- 在 1440×900、1920×1080 和当前高分辨率窗口验证。
- 中栏缩窄时保持 feature/output 两个主视图可读，tile lens 可降为纵向排列。
- 910B 架构视图支持拖拽、Fit 和缩放，路径线宽不随 focus 改变。
- 检查 legacy decoration residue；generic panel/card 不残留私有左色条、inset rail、重复 border 或私有 gradient。
- 通过本地 HTTP 服务打开 `/code%20recovery/index.html` 完成最终视觉检查。

## 6. 明确假设

- 首版是 Conv 单案例，不是通用源码导入器。
- 默认 dark mode，并保留执行回放控制。
- `conv_bias_relu_reference.asc` 是实现参考，不是可编译 Golden Model。
- 演示 shape/tiling 是 fixture 输入，不是从源码、host 或硬件观测恢复。
- 目标架构按文件注释采用 Ascend 910B；CANN API 的实际可用性仍为待验证状态。
- 不修改原始 `pto_compute-graph-viewer/tiling/`；所有实现仅发生在复制的 `code recovery/` 目录。
