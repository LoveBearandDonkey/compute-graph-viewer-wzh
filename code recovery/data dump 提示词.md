请在现有 `compute-graph-viewer/code recovery` Demo 中实现 **Tensor Data Inspector 最小可用版本**。

本次只实现以下数据链路：

```text
Feature X0 / GM
→ fmapA1 / A1(L1)
→ fmapA2 / A2(L0A)
```

目标是让用户能够看到 Tensor 中每个坐标的具体数值，并理解 A2 中每个元素来自 Feature X0 的哪个坐标。矩阵中的数值需要与 Heatmap 背景叠加显示。

------

# 1. 开始前先阅读

先阅读并遵守以下文件：

```text
code recovery/AGENTS.md
code recovery/CLAUDE.md
code recovery/spec.md
code recovery/README.md
code recovery/data/fixtures/conv_bias_relu.trace.json
```

重点检查现有实现：

```text
code recovery/index.html
code recovery/src/app.js
code recovery/src/styles.css
code recovery/shared.css
```

保持当前页面的信息架构、设计系统、状态管理和交互方式，不要重写整个页面。

优先在现有 `app.js` 和 `styles.css` 中增量实现。只有确有必要时才修改 fixture 或 HTML。

不要引入新的前端框架、第三方矩阵库、图表库或 Heatmap 库。

------

# 2. 产品目标

当前页面只能表达 Tensor 的 Shape、Layout、Dtype、Bytes 和流转过程，但不能展示 Tensor 中的具体数据。

本次需要增加一个 `Tensor Data Inspector`，让用户能够：

1. 查看 X0、A1 和 A2 中的具体数值；
2. 看到数值叠加在 Heatmap 背景之上；
3. 点击 A2 的任意单元格；
4. 查看该单元格的局部坐标、全局 M/K 坐标和卷积语义坐标；
5. 查看该值来自 X0 和 A1 的哪个坐标；
6. 区分真实输入值与卷积 Padding 产生的 0；
7. 在 X0、A1 和 A2 之间保持选中数据的关联。

本功能仍然属于静态代码还原模型，不得暗示这些数据来自真实设备运行或 Runtime Dump。

------

# 3. 本次范围

## 3.1 必须实现

- Feature X0 数据视图；
- A1 数据视图；
- A2 `16×16` Tile 数据视图；
- 稳定的随机小数数据；
- 数值与 Heatmap 叠加显示；
- Channel 切换；
- Output Tile 与 K Slice 联动；
- Cell Inspector；
- A2 到 X0/A1 的来源坐标追踪；
- Padding 识别；
- 跨视图高亮；
- Synthetic Data 状态说明。

## 3.2 本次不要实现

不要实现以下功能：

- Weight W0；
- B1 或 B2；
- Bias；
- CO1；
- Mmad 数值计算；
- K Loop 累加结果；
- Fixpipe；
- ReLU 计算；
- Output Y0；
- 真实二进制数据导入；
- Runtime Dump；
- 文件上传；
- 任意 Shape 的通用 Tensor Viewer；
- 物理 ZZ Layout 的精确地址映射；
- 大 Tensor 虚拟滚动；
- 后端服务；
- Web Worker；
- 新的全局设计系统。

本次是固定 Conv + Bias + ReLU Demo 的最小可用实现，不要过度抽象。

------

# 4. 固定算子参数

使用当前 Demo 已有参数：

```text
N  = 1
Ci = 16
Hi = 8
Wi = 8

Co = 32
Ho = 8
Wo = 8

Kh = 3
Kw = 3

strideH = 1
strideW = 1

padTop    = 1
padBottom = 1
padLeft   = 1
padRight  = 1

dilationH = 1
dilationW = 1

M = Ho × Wo = 64
K = Ci × Kh × Kw = 144

tileM = 16
tileK = 16
```

A2 的当前 Tile 为：

```text
A2[16,16]
```

其中：

- 行表示当前 M Tile 中的局部 `mLocal`；
- 列表示当前 K Slice 中的局部 `kLocal`。

------

# 5. 随机数据生成

## 5.1 数据要求

为 Feature X0 生成随机小数：

```text
Shape: [1,16,8,8]
Dtype semantic: FP16
Value range: [-1.0, 1.0]
Display precision: 2 位小数
```

不要直接使用 `Math.random()` 在每次 Render 时重新生成数据，否则切换步骤或刷新组件时数值会不断变化。

实现一个固定种子的伪随机生成器，例如：

```text
seed = 20260805
```

要求：

- 同一次页面加载中数值稳定；
- 页面刷新后数值仍然一致；
- X0、A1 和 A2 必须来自同一份数据；
- 内部可保留 3～4 位小数；
- 界面统一显示 2 位小数；
- 不要为了模拟 FP16 而引入复杂浮点转换。

建议将数据初始化为一维数组，并提供坐标访问函数：

```js
getX0Value(n, c, h, w)
```

不要为每个格子存储庞大的对象结构。

------

# 6. 数据映射规则

## 6.1 X0

逻辑 Shape：

```text
X0[N,C,H,W] = [1,16,8,8]
```

逻辑坐标：

```text
X0[n,c,h,w]
```

界面以单个 Channel Slice 展示：

```text
当前 channel = c
显示 8×8 的 H×W 矩阵
```

提供 Channel 控件：

```text
C0 ～ C15
```

可使用下拉框、Stepper 或现有项目中的 Segmented/Select 控件，优先复用当前设计系统。

------

## 6.2 A1

A1 的物理 Layout 为：

```text
NC1HWC0
```

当前固定参数：

```text
C1 = 1
C0 = 16
```

映射关系：

```text
c1 = floor(c / 16)
c0 = c % 16
```

因此：

```text
X0[n,c,h,w]
↔ A1[n,c1,h,w,c0]
```

A1 不改变数值，只改变存储坐标。

A1 视图同样以单 Channel 的 `8×8 H×W` 矩阵显示，便于和 X0 对照。

在 Cell Inspector 中必须同时显示：

```text
X0 logical coordinate: [n,c,h,w]
A1 physical coordinate: [n,c1,h,w,c0]
```

不要把 A1 表述为发生了数值计算。

------

## 6.3 A2

A2 当前 Tile：

```text
A2[mLocal,kLocal]
Shape = [16,16]
```

根据当前 `outputTileIndex` 计算：

```text
Mi = floor(outputTileIndex / 2)
Nj = outputTileIndex % 2

mStart = Mi × 16
```

A2 只依赖 `Mi`，不依赖 `Nj`。

因此 OT0 和 OT1 的 A2 内容相同，OT2 和 OT3 的 A2 内容相同，以此类推。

根据当前 `kIndex`：

```text
kStart = kIndex × 16
```

单元格全局坐标：

```text
mGlobal = mStart + mLocal
kGlobal = kStart + kLocal
```

M 坐标映射为输出空间位置：

```text
oh = floor(mGlobal / Wo)
ow = mGlobal % Wo
```

K 坐标映射为卷积语义：

```text
ci = floor(kGlobal / (Kh × Kw))

kRemainder = kGlobal % (Kh × Kw)

kh = floor(kRemainder / Kw)
kw = kRemainder % Kw
```

输入坐标：

```text
ih = oh × strideH - padTop + kh × dilationH
iw = ow × strideW - padLeft + kw × dilationW
```

有效性判断：

```text
valid =
  ih >= 0 &&
  ih < Hi &&
  iw >= 0 &&
  iw < Wi &&
  ci >= 0 &&
  ci < Ci
```

若有效：

```text
A2[mLocal,kLocal] = X0[0,ci,ih,iw]
```

若无效：

```text
A2[mLocal,kLocal] = 0
invalidReason = "convolution-padding"
```

必须在界面上区分：

- 输入数据本身恰好等于 `0.00`；
- 因卷积 Padding 产生的 `0.00`。

Padding 单元格需要单独的视觉状态，例如：

- 特殊边框；
- 斜线纹理；
- 小型 `P` 标识；
- Tooltip 中明确显示 `Padding`。

不要仅靠数值 `0.00` 判断 Padding，必须根据坐标有效性判断。

------

# 7. UI 放置方式

在现有 `Tensor State & Transformation` 区域中增加一个数据查看区域。

不要删除现有 Tensor 元数据。

推荐结构：

```text
Tensor State & Transformation
├── 现有 Tensor 信息
└── Tensor Data
    ├── Tensor Tab：X0 | A1 | A2
    ├── Context Controls
    ├── Matrix Grid
    └── Cell Inspector
```

也可以使用现有 Pane/Card 结构，但必须保证：

- 不遮挡 Source；
- 不破坏 Hardware Participation；
- 不影响 Execution Dock；
- 页面在现有桌面宽度下仍可正常使用。

------

# 8. Tensor Data Header

每个 Tensor Data 视图顶部显示：

```text
Tensor name
Location
Shape
Layout
Dtype
Data source
```

示例：

```text
Feature X0
GM · NCHW · [1,16,8,8] · FP16

Synthetic Data
Seed 20260805 · Not Runtime Dump
```

A2 示例：

```text
fmapA2
L0A · Logical Tile [16,16] · Physical Layout ZZ · FP16

OT0 · Mi=0 · Nj=0
K0 · Global K [0,15]
Synthetic / Derived
```

不要声称能够展示 ZZ Layout 下的真实物理线性地址。

A2 当前展示的是：

```text
Logical Tile View
```

需要明确标注，避免用户误以为矩阵格子顺序就是 ZZ 的真实连续存储顺序。

------

# 9. Matrix Grid

## 9.1 X0 和 A1

显示：

```text
8 行 × 8 列
```

行列标题：

```text
H0 ～ H7
W0 ～ W7
```

通过 Channel 控件切换 `C0～C15`。

------

## 9.2 A2

显示：

```text
16 行 × 16 列
```

行标题：

```text
M0 ～ M15
```

列标题：

```text
K0 ～ K15
```

这里的行列标题表示 Tile Local Coordinate。

Hover 或 Cell Inspector 中再展示全局坐标。

------

## 9.3 Cell 内容

每个格子同时显示：

1. Heatmap 背景；
2. 数字文本。

示例：

```text
-0.42
 0.18
 0.00
```

数字不能被 Heatmap 遮挡。

需要保证文本对比度，根据背景强度自动选择合适的前景色，或者给文本增加轻微底色/阴影。

------

# 10. Heatmap 规则

Heatmap 默认开启，不需要把 Value 和 Heatmap 做成互斥模式。

本次需求是：

```text
Heatmap background + numeric value overlay
```

颜色规则建议：

```text
负值：使用现有设计系统中的负向/危险语义色
正值：使用现有设计系统中的强调色或正向色
接近 0：接近中性背景
Padding：使用独立的中性纹理或特殊边框
```

强度根据绝对值计算：

```text
intensity = abs(value) / maxAbsValue
```

由于数据范围固定在 `[-1,1]`，也可以直接：

```text
intensity = abs(value)
```

需要限制最低和最高透明度，避免：

- 小值完全不可见；
- 大值导致文字无法阅读。

优先复用现有 CSS Token。

不要建立新的私有颜色体系，不要在大量单元格中硬编码不同 Hex 值。

可以使用：

- CSS Variable；
- `color-mix()`；
- 背景色加透明度；
- `data-sign` 和 `data-intensity`；
- 少量分级 class。

考虑性能，不要给每个格子生成复杂 SVG。

------

# 11. 交互

## 11.1 选择 Tensor

提供：

```text
X0 | A1 | A2
```

三个 Tab。

Tab 切换不改变当前执行步骤。

------

## 11.2 Channel 切换

X0 和 A1 共用当前 Channel 状态：

```js
selectedFeatureChannel
```

切换 X0/A1 时保持相同 Channel。

------

## 11.3 Output Tile 和 K Slice

复用页面现有：

```text
outputTileIndex
kIndex
```

不要建立一套重复状态。

A2 根据这两个状态实时重新计算。

改变 `outputTileIndex` 时：

```text
Mi = floor(OT / 2)
Nj = OT % 2
```

A2 只随 `Mi` 变化。

改变 `Nj` 不应改变 A2 的数值。

改变 `kIndex` 时更新：

- A2 数据；
- Global K 范围；
- Cell Inspector；
- 选中状态。

若旧选中单元格仍然存在，可以保留相同 `mLocal/kLocal`；Inspector 自动更新为新坐标。

------

## 11.4 点击单元格

点击任意 Cell 后设置：

```js
selectedDataElement
```

建议结构：

```js
{
  tensorId,
  localCoordinates,
  globalCoordinates,
  semanticCoordinates,
  value,
  valid,
  invalidReason,
  sourceCoordinates
}
```

不要求严格使用此结构，但不要将选中状态散落在多个无关联变量中。

选中 Cell 需要：

- 明确选中边框；
- 键盘 Focus 状态；
- 更新 Cell Inspector；
- 更新跨 Tensor 高亮。

------

## 11.5 跨 Tensor 高亮

点击有效的 A2 Cell 后：

1. 自动得到 `ci、ih、iw`；
2. X0 和 A1 的 Channel 自动切换到 `ci`；
3. X0 中高亮 `[h=ih,w=iw]`；
4. A1 中高亮相同数据对应的位置；
5. A2 中保持当前 Cell 高亮。

若 A2 Cell 是 Padding：

- 不高亮 X0/A1 中任何格子；
- Inspector 显示 `No source cell`;
- 明确显示该值由卷积 Padding 产生。

点击 X0 或 A1 Cell 时：

- X0/A1 之间同步高亮；
- 不要求反向找出所有引用该数据的 A2 单元；
- 本次不实现一对多反向追踪。

------

# 12. Cell Inspector

Cell Inspector 至少显示以下内容。

## 12.1 X0 Cell

```text
Tensor
Feature X0

Value
-0.42

Logical coordinate
X0[n=0,c=3,h=2,w=5]

Logical linear index
...

Data source
Synthetic Data · Seed 20260805
```

------

## 12.2 A1 Cell

```text
Tensor
fmapA1

Value
-0.42

Logical source
X0[n=0,c=3,h=2,w=5]

Physical coordinate
A1[n=0,c1=0,h=2,w=5,c0=3]

Transform
DataCopy · Value unchanged

Evidence
Derived from fixed layout mapping
```

可以显示 A1 的确定性物理线性 Offset，因为 NC1HWC0 映射明确：

```text
offsetElements =
((((n × C1 + c1) × H + h) × W + w) × C0 + c0)

offsetBytes = offsetElements × 2
```

------

## 12.3 A2 Cell

必须显示：

```text
Tensor
fmapA2

Value
-0.42

Local tile coordinate
A2[mLocal=3,kLocal=7]

Global matrix coordinate
A[mGlobal=3,kGlobal=39]

Output-space coordinate
oh=0, ow=3

Convolution coordinate
ci=4, kh=1, kw=0

Input coordinate
ih=0, iw=2

Source
X0[n=0,c=4,h=0,w=2]
A1[n=0,c1=0,h=0,w=2,c0=4]

Transform
LoadData3D

Validity
Valid input
```

Padding 示例：

```text
Value
0.00

Input coordinate
ih=-1, iw=-1

Validity
Convolution Padding

Source
No source cell
```

对于 A2：

- 不显示未经证实的 ZZ 物理 Offset；
- 可以显示 `Physical layout: ZZ`；
- 同时显示 `Exact physical cell address unavailable in this static model`。

------

# 13. 与执行步骤联动

需要尊重当前 `selectedStepId`。

最低要求：

- X0 始终可查看；
- A1 在数据尚未搬入前显示 `Not resident at this step`；
- A2 在尚未执行对应 LoadData3D 前显示 `Not written at this step`；
- 不得在 Buffer 尚未生成时直接展示为已经存在的有效数据。

优先根据现有 Step、Stage 或 Tensor Snapshot 判断可用性。

如果当前数据模型不足以精确判断，可使用固定 Demo 的 Stage 顺序实现，但需要：

- 集中封装判断逻辑；
- 添加清楚注释；
- 不要把判断散落在 Render 代码中；
- 不要伪造成 Runtime 状态。

建议状态：

```text
available
not-resident
not-written
```

这部分保持简单，不需要模拟真实 Buffer 生命周期。

------

# 14. 可访问性与可用性

矩阵单元格需要：

- 可点击；
- 可通过键盘 Focus；
- 有 `aria-label`；
- Hover 时显示简短 Tooltip；
- 选中状态不能只依赖颜色；
- Padding 状态不能只依赖颜色；
- 数字字号在 16×16 矩阵中仍可辨认。

A2 的 16×16 矩阵允许区域内部滚动，但不要让整个页面产生不可控横向滚动。

------

# 15. 工程要求

请遵守以下要求：

1. 不重构与本需求无关的代码；
2. 不改变现有 Trace 模型语义；
3. 不修改 Host/Kernel 源码；
4. 不伪造 Runtime 数据；
5. 不创建真实 Profiling 时间；
6. 不把 Synthetic Data 写入现有 Tensor 的 confirmed source；
7. 不破坏 Source、Hardware、Execution Dock 的交互；
8. 不引入新的 npm 构建流程；
9. 保持 GitHub Pages 可直接运行；
10. 继续支持当前通过 `python3 -m http.server` 的本地预览方式；
11. 所有数据计算放在独立函数中，不要把坐标公式直接堆在 HTML Render 模板里；
12. 所有新增函数添加简洁注释，说明是 logical view、synthetic data 或 derived mapping；
13. 避免在每次 Render 中重新生成 1024 个随机值；
14. 避免为每个 A2 Cell 预先保存完整来源对象，可按需计算；
15. 复用现有设计 Token 和组件。

建议拆分函数：

```js
createSeededRandom(seed)
createSyntheticFeatureData()
getX0Value(n, c, h, w)

mapLogicalX0ToA1(n, c, h, w)

getOutputTileContext(outputTileIndex)
mapA2CellToSemanticCoordinates(
  outputTileIndex,
  kIndex,
  mLocal,
  kLocal
)

getA2CellValue(...)
getHeatmapStyle(value, valid)
renderTensorDataPanel(...)
renderTensorMatrix(...)
renderCellInspector(...)
```

函数名可根据现有项目风格调整。

------

# 16. 验收用例

实现后请至少手动验证以下情况。

## 用例 1：X0 与 A1 数值一致

选择：

```text
Channel C3
H2
W5
```

应满足：

```text
X0[0,3,2,5]
=
A1[0,0,2,5,3]
```

数值完全一致。

------

## 用例 2：左上角 Padding

选择：

```text
OT0
K0
A2[mLocal=0,kLocal=0]
```

计算：

```text
mGlobal = 0
oh = 0
ow = 0

kGlobal = 0
ci = 0
kh = 0
kw = 0

ih = -1
iw = -1
```

结果应为：

```text
A2[0,0] = 0.00
Validity = Convolution Padding
Source = No source cell
```

------

## 用例 3：第一个有效中心点

选择：

```text
OT0
K0
A2[mLocal=0,kLocal=4]
```

计算：

```text
kGlobal = 4
ci = 0
kh = 1
kw = 1

oh = 0
ow = 0

ih = 0
iw = 0
```

结果应满足：

```text
A2[0,4]
=
X0[0,0,0,0]
=
A1[0,0,0,0,0]
```

并同步高亮 X0/A1 的 `H0,W0`。

------

## 用例 4：最后一个 K 坐标

选择：

```text
K8
kLocal = 15
```

应得到：

```text
kGlobal = 8×16+15 = 143

ci = 15
kh = 2
kw = 2
```

------

## 用例 5：最后一个输出位置

选择：

```text
OT7
mLocal = 15
```

应得到：

```text
Mi = 3
Nj = 1

mGlobal = 3×16+15 = 63

oh = 7
ow = 7
```

------

## 用例 6：Nj 不改变 A2

比较：

```text
OT0 与 OT1
```

两者：

```text
Mi = 0
```

因此在相同 `kIndex` 下，A2 的全部 `16×16` 数值必须一致。

比较：

```text
OT2 与 OT3
OT4 与 OT5
OT6 与 OT7
```

也应满足同样规则。

------

## 用例 7：Heatmap

检查：

- 正值和负值有不同视觉方向；
- 数值越接近绝对值 1，背景强度越高；
- 数字始终可读；
- Padding 与普通 0 有明显区别；
- 选中状态不被 Heatmap 覆盖。

------

## 用例 8：稳定随机数

刷新页面后，同一坐标：

```text
X0[0,3,2,5]
```

数值保持一致。

切换步骤、Tensor Tab、Channel、OT、K Slice 后再返回，数值保持一致。

------

# 17. 完成后的输出

完成后请提供：

1. 修改了哪些文件；
2. 每个文件的修改内容概述；
3. 随机数据如何保证稳定；
4. A2 坐标映射如何实现；
5. Heatmap 如何计算；
6. 执行步骤可用性如何判断；
7. 已验证的验收用例；
8. 尚未实现的范围；
9. 本地预览命令。

不要只描述方案，请直接修改代码并完成可运行实现。
