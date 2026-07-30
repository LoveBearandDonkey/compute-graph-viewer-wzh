# Conv + Bias + ReLU Code Recovery 详细实施计划

## 1. 文档定位

本文档是 `code recovery` HTML Demo 的唯一实施顺序和进度记录。它把产品目标、代码事实、实现依赖和验收工作组织为阶段 1～5；文中不再存在另一套 Phase 1～5。

配套规范见 `spec.md`。执行语义基线见 `Conv2D + Bias + ReLU 执行过程.md`。

## 2. 2026-07-28 新代码基线

### 2.1 权威输入

本轮以以下三份本地材料为直接依据：

1. Host Tiling：`src/conv_bias_relu_complete_demo/op_host/conv_bias_relu_tiling.cpp`
2. Device Kernel：`src/conv_bias_relu_complete_demo/op_kernel/conv_bias_relu_reference_complete.asc`
3. 执行解读：`Conv2D + Bias + ReLU 执行过程.md`

目录实际名称是 `conv_bias_relu_complete_demo`。

### 2.2 已被新源码确认的事实

| 对象 | 新基线 |
| --- | --- |
| Feature | 逻辑 `[1,16,8,8]`；GM `NC1HWC0 [1,1,8,8,16]`；FP16；2048 B |
| Filter | 逻辑 `[32,16,3,3]`；GM `ND [144,32]`；FP16；9216 B |
| Bias | `ND [32]`；FP32；128 B |
| Output | 逻辑 `[1,32,8,8]`；GM `ND [64,32]`；FP16；4096 B |
| Cube | `A[64,144] × B[144,32] + Bias[32] → C[64,32]` |
| Tile | `tileM=16`、`tileK=16`、`tileN=16` |
| Tile 数量 | M=4、K=9、N=2；输出 Tile=8 |
| Core 映射 | `OT=Mi×2+Nj`；`blockDim=8`；每核一个 `[16,16]` 输出 Tile |
| K 迭代 | I0～I8，共 9 次；I0 加 Bias，I1～I8 只累加 |
| Output 写回 | Fixpipe 从 CO1 直接写 GM，融合 NZ→ND、FP32→FP16、ReLU |

输出 `[64,32]` 等价于 NHWC `[1,8,8,32]`，不等价于 NCHW。界面不得仅显示逻辑 NCHW 而隐藏实际 GM 合约。

### 2.3 已消除的旧缺口

以下旧 fixture 结论已经失效，必须从 UI、fixture 和文档中删除：

- `tileM=32、tileK=48、tileN=16`
- K loop=3
- A1/B1 大小未知或为 0
- Feature/Filter Copy helper 无函数体
- Bias C1→C2 缺失
- LoadFilterToL0B 无函数体
- blockDim 和 block→output tile 映射未知
- CopyOut helper 和 GM offset 未知
- CO1 先到 outC1、再单独 CopyOut 的两段式路径

### 2.4 仍需保留的不确定性

新源码是“固定参数、内部一致的静态参考”，不是已编译和已测量的真实 trace：

- 未使用目标环境的 BiSheng/CANN 编译；
- 未在 910B/Atlas A2/A3 上运行；
- API 字段、重载和注册名尚未按目标 CANN 版本验证；
- 无 correctness、dump、profiling 或真实 duration；
- 固定 tiling 是源码事实，但不是自动 tiling 结果或性能最优结论；
- 页面硬件总容量仍是 Demo 配置，不能宣称为真实 910B 规格。

## 3. 产品目标

用户在同一工作台内完成四个连续判断：

1. Host 如何把逻辑 Shape 推导成 Cube M/K/N、Tile 数和 blockDim；
2. Kernel 当前代码属于哪一步，涉及哪个 Tensor、Buffer 和 Event；
3. Tensor 如何在 GM、L1、L0A/L0B、Bias Table、L0C 和 GM Output 之间变化；
4. 当前结论是代码确认、计算派生、语义推断，还是仍需目标环境验证。

统一关系：

```text
Host Tiling source ─┐
                    ├─ sourceRefs ─→ TraceStep ─→ Tensor / DataFlow / Event / Hardware
Kernel source ──────┘                       └────→ Instructions / Timeline / Playback
```

所有视图只消费统一的 `selectedStepId`。任何区域不得维护独立步骤状态。

## 4. 页面结构

### 4.1 上方三栏

- 左栏：Source，内部提供 Host/Kernel 两个文件 Tab；
- 中栏：Tensor State & Transformation；
- 右栏：Hardware Participation。

### 4.2 下方全宽 Dock

- Instructions：逻辑执行顺序，不表达真实时间比例；
- Timeline：只有 fixture 具备 `startTime/duration` 时才作为 Estimated Timeline 启用；
- Playback：控制同一 `selectedStepId`；
- Terminal 与 Visualization Dock 互斥。

Inspector 已永久移出产品范围，任何阶段都不得恢复。

## 5. Source 双文件体验

### 5.1 Tab

左栏固定显示：

- `op_host/tiling.cpp`
- `op_kernel/kernel.asc`

Tab 切换只改变当前可见源码，不改变 `selectedStepId`。

### 5.2 双文件事件标记

这里的“事件标记”包括两类：

1. 逻辑步骤标记：Shape、Tiling、容量、blockDim、LocalTensor、CopyIn、Load、Mmad、Fixpipe；
2. 同步事件标记：`MTE2_MTE1`、`M_MTE1`、`MTE1_M`、`M_FIX`。

Host 文件至少标记：

- 行 20～43：固定 Shape、卷积参数、Ho/Wo；
- 行 45～56：M/K/N、16×16×16、4×9×2、8 个 OT；
- 行 58～70：GM 与片上元素数；
- 行 73～91：Tiling callback、`SetBlockDim(8)`、workspace=0。

Kernel 文件至少标记：

- 行 54～64：blockIdx→Mi/Nj→mStart/nStart；
- 行 82～108：Local Memory 地址图；
- 行 110～127、191～219：三路 CopyIn、`MTE2_MTE1`、Bias C1→C2；
- 行 129～165、221～268：K loop、`M_MTE1`、Load3D、Load2D、`MTE1_M`、Mmad；
- 行 167～187：`M_FIX`、Fixpipe、outputOffset。

### 5.3 联动规则

- 点击已标记代码行：切换到对应 TraceStep，并停留在当前文件 Tab；
- 从播放、Instructions 或 Timeline 进入新步骤：自动切到该步骤的 primary `sourceRef` 文件；
- 一个步骤可同时引用 Host 和 Kernel；例如 block 映射同时关联 Host `SetBlockDim` 与 Kernel `GetBlockIdx`；
- 当前文件不存在该步骤的引用时，不伪造高亮；
- Tab、源码行、播放和执行块最终都只更新同一 `selectedStepId`。

## 6. Trace 数据改造

### 6.1 多源码

Trace 根对象新增：

```json
{
  "sources": [
    {
      "id": "host",
      "label": "op_host/tiling.cpp",
      "path": "op_host/conv_bias_relu_tiling.cpp",
      "projectPath": "src/conv_bias_relu_complete_demo/op_host/conv_bias_relu_tiling.cpp"
    },
    {
      "id": "kernel",
      "label": "op_kernel/kernel.asc",
      "path": "op_kernel/conv_bias_relu_reference_complete.asc",
      "projectPath": "src/conv_bias_relu_complete_demo/op_kernel/conv_bias_relu_reference_complete.asc"
    }
  ]
}
```

旧 `source` 字段暂时保留为兼容入口，新 UI 以 `sources` 为准。

### 6.2 文件感知的源码引用

每个步骤使用：

```json
{
  "sourceRefs": [
    {"fileId": "host", "lines": [84, 85]},
    {"fileId": "kernel", "lines": [55, 61, 62]}
  ]
}
```

`sourceLines` 暂时保留为 Kernel 兼容字段，但不得再作为多文件身份的唯一来源。

### 6.3 可信度

- `confirmed`：当前两份源码直接给出；
- `derived`：由源码参数确定性计算；
- `inferred`：依据 API 语义解释，但源码未直接声明；
- `unverified`：源码表达目标行为，尚未在目标 CANN/硬件验证；
- `unknown`：当前材料不足。

代码确认与目标验证是两个维度。例如 `reluEn=true` 是 confirmed；该参数在目标 CANN 上成功编译执行仍是 unverified。

## 7. 新逻辑回放

前 10 步沿用当前阶段命名，但必须区分“Host/Kernel 准备信息”和“运行时硬件动作”。Host Shape、Host Tiling、Buffer 元素数和 blockDim 是执行配置，不得画成 GM 搬运或 AI Core 指令。

| Step | 阶段 | 主要内容 | 执行性质 | 主源码 |
| ---: | --- | --- | --- | --- |
| 1 | Input Shape | 固定 N/C/H/W、卷积参数并推导 Ho/Wo | Host 配置 | Host |
| 2 | Host Tiling | 定义 Cube M/K/N、`16×16×16` Tile 和 `4×9×2` Tile 数 | Host 配置 | Host |
| 3 | 不单独回放 | 原 Host Memory 内容并入 Step 4；只作为 Buffer Size 证据展示 | 非执行步骤 | Host |
| 4 | Host执行配置 | 计算 GM/Local Buffer 元素数；设置 `blockDim=8`；建立 `OT→Mi/Nj` 映射；workspace=0 | Host/Runtime 配置，不访问 GM | Host + Kernel |
| 5 | Allocate Memory | 建立 A1/B1/C1 地址范围以及 A2/B2/C2/CO1 LocalTensor 视图 | Kernel 准备 | Kernel |
| 6 | Copy Inputs | MTE2：GM X0 NC1HWC0→A1；GM W[Nj] ND→B1(NZ)；GM Bias[Nj]→C1 | 运行时数据搬运 | Kernel |
| 7 | Sync | `MTE2_MTE1`：MTE2 发布 L1 ready；MTE1 等待后才能读取 A1/B1/C1 | 运行时同步 | Kernel |
| 8 | Copy Data C2 | MTE1：Bias C1→C2 / Bias Table，FP32 `[16]`，64 B | 运行时数据搬运 | Kernel |
| 9 | Load Data A2 B2 | Iter 0：LoadData3D 生成 A[Mi,K0]；LoadData2D 生成 B[K0,Nj] | 运行时数据搬运 | Kernel |
| 10 | Sync | `MTE1_M`：MTE1 发布 A2/B2 ready；Cube 等待后才能执行 Iter 0 | 运行时同步 | Kernel |
| 11 | Iter 0 Mmad | `A[Mi,K0] × B[K0,Nj] + Bias[Nj] → Acc0/CO1`；Bias 只在本轮加入 | Cube 计算 | Kernel |
| 12 | K Loop · Iter 1～8 ×8 | 后续 8 轮的可展开 Loop Group；不是一条独立硬件指令 | 循环容器 | Kernel |
| 13 | Iter 1～7 Loop Body ×7 | 每轮依次执行 `M_MTE1 → Load Kk → MTE1_M → Mmad accumulate`，得到 Acc1～Acc7 | 同步 + 搬运 + Cube 计算 | Kernel |
| 14 | Iter 8 Final Loop Body ×1 | 执行同一完整 Loop Body，读取 K8 `[128:144]`，最终得到 Acc8 | 同步 + 搬运 + Cube 计算 | Kernel |
| 15 | M_FIX Sync | Cube 发布 Acc8 ready；Fixpipe 等待最终 CO1 | 运行时同步 | Kernel |
| 16 | Fixpipe Output | CO1 FP32 NZ→GM FP16 ND，融合 ReLU，并按 `outputOffset` 直接写回 | 运行时输出 | Kernel |

### 7.1 K Loop 的真实结构

`Iter` 是 `Iteration` 的缩写，用于避免大写 `I` 和小写 `l` 在界面字体中混淆。`Iter 0` 表示执行轮次，`K0` 表示该轮读取的数据 Tile。

首次迭代：

```text
Load K0
→ MTE1_M：A2/B2 ready
→ Iter 0 Mmad：A×B+Bias→Acc0
```

后续每次迭代 `Iter k`，其中 `k=1..8`：

```text
M_MTE1
Cube 表示上一轮 A2/B2 已读取完成，MTE1 可以覆盖
→
LoadData3D A[Mi,Kk] + LoadData2D B[Kk,Nj]
→
MTE1_M
MTE1 表示新的 A2/B2 已准备好，Cube 可以读取
→
Mmad
Acck = Acc(k-1) + A[Mi,Kk] × B[Kk,Nj]
```

因此，Step 13 不是“重复 7 次 Step 11～12”，而是折叠展示 7 个完整 Loop Body；Step 14 是相同 Loop Body 的最后一次执行。Step 12 只作为父级 Loop Group，不应在 Instructions 中与 Step 13、14 表现成三个同级、依次只执行一次的硬件动作。

### 7.2 Step 12 同步画面

`M_MTE1` 发生在下一轮 Load 之前。以 Iter 1 为例，此时 A2/B2 仍保存 K0：

```text
A2/B2 · K0
Cube read complete
→ reusable
→ MTE1 may overwrite with K1
```

因此 Step 12 的 Tensor 视图不得提前显示“K1 已加载”。K1 只能在 `M_MTE1` 通过后的 Load 阶段显示。

### 7.3 Instructions 展示层级

```text
Iter 0
├─ Load K0
├─ MTE1_M
└─ Mmad + Bias

K Loop · Iter 1～8 ×8
├─ Iter 1～7 ×7
│  ├─ M_MTE1
│  ├─ Load Kk
│  ├─ MTE1_M
│  └─ Mmad accumulate
└─ Iter 8 ×1
   ├─ M_MTE1
   ├─ Load K8
   ├─ MTE1_M
   └─ Final Mmad
```

默认折叠时必须显示循环次数；展开后允许定位每个 `Iter k` 的源码、Tensor、Event 和硬件状态。无论折叠或展开，总次数都必须保持每核 9 次 Mmad、全 8 核共 72 次 Mmad。

## 8. Tensor 和 Memory 表达

每核固定驻留：

| Buffer | Shape/元素 | Bytes | 地址/对齐 |
| --- | ---: | ---: | --- |
| A1 | 1024 FP16 | 2048 | 0 / 512 B |
| B1 | 2304 FP16 | 4608 | 2048 / 512 B |
| C1 | 16 FP32 | 64 | 6656 / 128 B |
| C2 | 16 FP32 | 64 | 0 / 64 B |
| A2 | `[16,16]` FP16 | 512 | 0 / 512 B |
| B2 | `[16,16]` FP16 | 512 | 0 / 512 B |
| CO1 | `[16,16]` FP32 | 1024 | 0 |

中栏按步骤展示：

- Host Shape/Tiling：逻辑→物理合约与 M/K/N；
- CopyIn：完整 Feature、一个 Filter N Tile、一个 Bias N Tile；
- Load：Feature window→A2 ZZ；Filter fractal→B2 ZN；
- I0：A2、B2、C2、CO1；
- I1～I8：A2、B2、已有 CO1，不再加入 Bias；
- Fixpipe：CO1 FP32 NZ→Output FP16 ND，并明确 NHWC 等价关系。

## 9. 实施顺序与进度

| 阶段 | 本阶段交付 | 状态 |
| --- | --- | --- |
| 阶段 1：Workbench MVP | 上方三栏、全宽 Dock、两种执行 Tab、播放、Terminal 互斥、统一步骤状态、移除 Inspector | **已完成（2026-07-27）** |
| 阶段 2：Tensor Code Recovery | 双源码 Tab、file-aware sourceRefs、新固定 tiling、精确 Buffer、9 次 K 语义、Bias C1→C2、Fixpipe 直写 GM | **已完成并按新源码刷新（2026-07-28）** |
| 阶段 3：Hardware Participation | 910B 路径、真实步骤参与节点、Buffer occupancy、Active/Idle/Waiting、4 类 Event dependency | **下一阶段，未开始** |
| 阶段 4：Execution Dock 深化 | Instructions 泳道、Loop Group 展开、事件因果；有估算数据后再做 Estimated Timeline | **未开始** |
| 阶段 5：验证与交付 | 错误/空状态、布局/播放回归、schema/PTO 审计、目标环境验证记录 | **未开始** |

当前整体进度：**2/5 个阶段完成**。

阶段 2 的“完成”表示静态模型和交互基线完成，不表示源码已经通过 CANN 编译或上板运行。

## 10. 后续阶段安排

### 阶段 3：Hardware Participation

1. 将每个步骤映射到 MTE2、MTE1、Cube、Fixpipe 和相应存储节点；
2. 使用 fixture 中的每核字节数计算 Demo occupancy；
3. 执行单元显示 Active/Idle/Waiting，不显示“空间占用”；
4. `MTE2_MTE1`、`M_MTE1`、`MTE1_M`、`M_FIX` 使用独立依赖边；
5. Demo capacity 与真实芯片总容量分层标注；
6. 不确定的硬件连线使用 unverified，不因图上存在节点就宣称代码使用它。

### 阶段 4：Execution Dock 深化

1. Instructions 按 Host/Scalar、MTE2、MTE1、Cube、Fixpipe、Event 分泳道；
2. I1～I7 Loop Group 支持展开；
3. 点击事件块展示 producer、consumer、阻止的提前执行；
4. Timeline 保持 unavailable，直至有明确估算值或 profiling；
5. 如增加估算值，必须显示 `Estimated Timeline · Not Profiling Data`。

### 阶段 5：验证与交付

1. JSON/schema 验证；
2. 两份源码所有 sourceRefs 行号存在；
3. 双 Tab 切换不丢失步骤；
4. 播放跨 Host→Kernel 自动切 Tab；
5. file:// 错误提示和 HTTP 启动说明；
6. 窄窗口、长文件名、滚动定位和键盘可访问性；
7. PTO design-system residue check；
8. 记录未完成的 CANN 编译、设备运行和 profiling 验证。

## 11. 验收标准

### 双源码

- 两个 Tab 都加载完整源码，而不是关键行摘录；
- 两个文件都存在可点击的步骤标记；
- 点击 Host `SetBlockDim` 选择 Launch Mapping；
- 点击 Kernel `GetBlockIdx` 选择同一 Launch Mapping；
- 点击 Kernel 任一 HardEvent 行选择对应 Event Step。

### Tiling

- 页面显示 `M/K/N=64/144/32`；
- 页面显示 `tile=16/16/16`；
- 页面显示 `M/K/N tile count=4/9/2`；
- 页面显示 `blockDim=8` 和 OT0～OT7。

### Tensor

- LoadData3D 显示 NC1HWC0→ZZ、`[16,16]`、512 B；
- LoadData2D 显示 NZ→ZN、`[16,16]`、512 B；
- I0 明确包含 Bias C2；
- I1～I8 不重复加入 Bias；
- CO1 为 `[16,16]` FP32、1024 B；
- Fixpipe 显示 FP32→FP16、NZ→ND、ReLU、512 B/核。

### 可信度

- 不再显示已被新源码解决的 missing implementation；
- 固定参数标为代码确认，不标为 profiler 观测；
- 页面明确“未编译、未上板、无真实 duration”；
- Output 明确 ND `[64,32]` 与 NHWC 等价，不误称为 NCHW 物理布局；
- Inspector 永不出现。

## 12. 启动

必须通过本地 HTTP 服务打开。页面会使用 `fetch()` 读取 JSON 和两份源码，`file://` 无法工作。

从工作区根目录 `/Users/songchenfei/Documents/ascend c` 启动服务后访问：

```text
http://127.0.0.1:4180/pto_compute-graph-viewer/code%20recovery/index.html
```
