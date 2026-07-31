# Conv2D + Bias + ReLU 执行过程

> 本文件是该固定 Demo 的唯一执行语义基线，已合并原 `conv_bias_relu执行过程_重构版.md` 的详细解释与原同名文档的结构化执行摘要。
>
> `Conv + Bias + ReLU Code Recovery 详细实施计划.md` 保留为产品实施、进度与验收记录；`tensor数据.md` 保持独立，作为 Trace 的 Tensor 数据清单，不并入本文。

> 基于 `conv_bias_relu_complete_demo` 固定 Demo 整理。
>
> 当前基线假设：`padding=1`、`stride=1`、`dilation=1`。这些参数来自 Demo 补全时的选择，不代表已经从真实业务代码验证。

---

# 第一部分：先建立整体理解

## 1. 阅读前必须区分的三类对象

这份文档同时出现三套坐标和两类“真实 Tensor”。如果不先区分，后续的 `mStart`、`kStart`、`A1/A2` 很容易混在一起。

| 层次 | 示例 | 含义 | 是否真实占用 Buffer |
|---|---|---|---|
| 原始 Tensor 坐标 | `X[n,c1,h,w,c0]` | Feature 在 GM/L1 中的真实数据布局 | 是 |
| Cube 逻辑坐标 | `A[m,k]`、`B[k,n]`、`C[m,n]` | 为描述矩阵乘法建立的逻辑视图 | 完整 A/B/C 不额外分配 Buffer |
| 局部 Tensor 坐标 | `fmapA2[p,q]`、`weightB2[p,q]` | 当前核、当前 K 迭代真正放入 L0A/L0B 的局部 Tile | 是 |

完整关系：

```text
真实源数据 X / W
        │
        │ LoadData3D / DataCopy + LoadData2D
        │ 按逻辑 A/B 坐标生成局部 Tile
        ▼
局部 Tensor fmapA2 / weightB2
        │
        │ Mmad
        ▼
局部累加 Tensor accumCo1
        │
        │ Fixpipe
        ▼
GM Output
```

### 1.1 命名规则

后续统一使用：

```text
变量名：fmapA1
位置：A1 / L1
Shape：[1,1,8,8,16]
dtype：FP16
format：NC1HWC0
```

不要写成：

```text
A1 [1,1,8,8,16]
```

因为：

- `fmapA1` 是 Tensor 变量名；
- `A1` 是逻辑存储位置；
- `L1` 是物理存储层级。

---

## 2. 一页总览

### 2.1 输入、权重、Bias 与输出

| Tensor | 逻辑 Shape | GM 物理 Shape | GM format | dtype |
|---|---:|---:|---|---|
| Feature `X0` | `[1,16,8,8]` | `[1,1,8,8,16]` | NC1HWC0 | FP16 |
| Weight `W0` | `[32,16,3,3]` | `[144,32]` | ND `[K,Co]` | FP16 |
| Bias `D0` | `[32]` | `[32]` | linear | FP32 |
| Output `Y0` | `[1,32,8,8]` | `[64,32]` | ND `[M,Co]` | FP16 |

Padding 只是卷积参数，没有提前创建 `[1,1,10,10,16]` 的实体 Tensor。

### 2.2 卷积转换为 Cube 矩阵乘法

```text
A[M,K] × B[K,N] + D[N] → C[M,N]
```

本例：

```text
M = Nbatch × Ho × Wo
  = 1 × 8 × 8
  = 64

K = Ci × Kh × Kw
  = 16 × 3 × 3
  = 144

N = Co
  = 32
```

所以完整逻辑计算为：

```text
A[64,144] × B[144,32] + D[32] → C[64,32]
```

其中：

- `A[64,144]`：64 个输出空间位置，每个位置需要 144 个输入元素；
- `B[144,32]`：32 个输出通道，每个通道有 144 个权重；
- `C[64,32]`：64 个输出位置，每个位置生成 32 个输出通道。

### 2.3 完整硬件流转

```text
GM
├─ Feature X0：NC1HWC0 [1,1,8,8,16]
├─ Weight W0：ND [144,32]
└─ Bias D0：[32]
        │
        │ MTE2
        ▼
L1
├─ fmapA1：完整 Feature [1,1,8,8,16]
├─ weightB1：当前 Nj 的权重 [144,16]
└─ biasC1：当前 Nj 的 Bias [16]
        │
        │ MTE1，每轮生成一个 K Tile
        ▼
L0A / L0B / Bias Table
├─ fmapA2：A[Mi,Kk] [16,16]
├─ weightB2：B[Kk,Nj] [16,16]
└─ biasC2：D[Nj] [16]
        │
        │ Cube Mmad × 9
        ▼
L0C
└─ accumCo1：C[Mi,Nj] [16,16]，FP32
        │
        │ Fixpipe：NZ→ND、FP32→FP16、ReLU
        ▼
GM Output
└─ 当前核写入 C[Mi,Nj]
```

---

# 第二部分：输出如何分给 8 个核

## 1. Tiling 参数

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `tileM` | 16 | 一个核处理 16 个输出空间位置 |
| `tileK` | 16 | 一次 Mmad 累加 16 个 K 元素 |
| `tileN` | 16 | 一个核处理 16 个输出通道 |
| `mTiles` | 4 | `ceil(64/16)` |
| `kTiles` | 9 | `ceil(144/16)` |
| `nTiles` | 2 | `ceil(32/16)` |
| `outputTileCount` | 8 | `4×2` |
| `blockDim` | 8 | 8 个 Block 任务 |

M、K、N 都可被 16 整除，因此没有 Tail。

## 2. Output Tile 与 Core 映射

```text
OT = blockIdx
Mi = OT / 2
Nj = OT % 2
mStart = Mi × 16
nStart = Nj × 16
```

输出矩阵 `C[64,32]` 被切成：

```text
                         N：输出通道
                   N0：0～15       N1：16～31

M0：位置 0～15        OT0              OT1
M1：位置 16～31       OT2              OT3
M2：位置 32～47       OT4              OT5
M3：位置 48～63       OT6              OT7
```

每个核只计算一个：

```text
C[Mi,Nj] = [16,16]
```

### 2.1 为什么一个 Feature Tile 被两个核使用

固定 `Mi`，`Nj` 有两种：

```text
A[Mi,:] × B[:,N0] → C[Mi,N0]
A[Mi,:] × B[:,N1] → C[Mi,N1]
```

同一批 16 个输出位置需要生成 32 个输出通道，但一个核只计算 16 个通道，因此：

| Feature 逻辑分组 | 使用它的核 |
|---|---|
| M0 | OT0、OT1 |
| M1 | OT2、OT3 |
| M2 | OT4、OT5 |
| M3 | OT6、OT7 |

这里的“使用同一份 Feature”是数学上使用相同的输入区域，不代表两个核共用同一个 L1/A2 Buffer。

### 2.2 为什么一个 Weight Tile 被四个核使用

固定 `Nj`，`Mi` 有四种：

```text
A[M0,:] × B[:,Nj] → C[M0,Nj]
A[M1,:] × B[:,Nj] → C[M1,Nj]
A[M2,:] × B[:,Nj] → C[M2,Nj]
A[M3,:] × B[:,Nj] → C[M3,Nj]
```

同一组 Weight 需要作用于全部输出空间位置，因此：

| Weight 逻辑分组 | 使用它的核 |
|---|---|
| N0：输出通道 0～15 | OT0、OT2、OT4、OT6 |
| N1：输出通道 16～31 | OT1、OT3、OT5、OT7 |

### 2.3 为什么每个核仍搬完整 Feature

当前 Demo 中，每个核都将完整 `X0` 搬入自己的 L1：

```text
featureGm_ [1,1,8,8,16]
        ↓ DataCopy
fmapA1 [1,1,8,8,16]
```

原因是该 Demo 选择了简单实现：

- 不在 GM→L1 阶段计算当前 Mi 对应的局部输入区域；
- 不额外处理 Halo、边界和局部坐标偏移；
- 由后续 `LoadData3D` 根据 Mi、Kk 从完整 Feature 中生成当前 A2 Tile。

这不是卷积 Kernel 的必然要求，而是该固定 Demo 的实现策略。

---

# 第三部分：固定一个核，看完整执行过程

以下固定：

```text
当前核：OT0
Mi = M0
Nj = N0
mStart = 0
nStart = 0
```

OT0 最终负责：

```text
输出位置 m0～m15
输出通道 co0～co15
输出 Tile C[M0,N0] = [16,16]
```

## 1. 循环前：LocalTensor 分配

本阶段只建立 LocalTensor 视图，不搬运数据。

| Tensor 变量 | Shape | Tile 身份 | 位置 | dtype | format | 大小 | 起始地址 | Align |
|---|---:|---|---|---|---|---:|---:|---|
| `fmapA1` | `[1,1,8,8,16]` | 完整 Feature | A1 / L1 | FP16 | NC1HWC0 | 2048 B | 0 | 512 B |
| `weightB1` | `[144,16]` | 当前 N0 的完整 K 权重 | B1 / L1 | FP16 | NZ | 4608 B | 2048 | 512 B |
| `biasC1` | `[16]` | `D[N0]` | C1 / L1 | FP32 | linear | 64 B | 6656 | 128 B |
| `biasC2` | `[16]` | `D[N0]` | C2 / Bias Table | FP32 | linear | 64 B | 0 | 64 B burst |
| `fmapA2` | `[16,16]` | 当前 `A[M0,Kk]` | A2 / L0A | FP16 | ZZ | 512 B | 0 | 独立 Buffer；具体要求待验证 |
| `weightB2` | `[16,16]` | 当前 `B[Kk,N0]` | B2 / L0B | FP16 | ZN | 512 B | 0 | 独立 Buffer；具体要求待验证 |
| `accumCo1` | `[16,16]` | 当前 `C[M0,N0]` | CO1 / L0C | FP32 | NZ | 1024 B | 0 | 独立 Buffer；具体要求待验证 |

## 2. GM → L1

| Instruction | 来源 | 目标 | 变化 |
|---|---|---|---|
| DataCopy | `featureGm_ [1,1,8,8,16]` | `fmapA1 [1,1,8,8,16]` | NC1HWC0→NC1HWC0，完整搬入 |
| DataCopy ND→NZ | `weightGm_[:,0:16]` | `weightB1 [144,16]` | ND `[K,Co]`→NZ |
| DataCopy | `biasGm_[0:16]` | `biasC1 [16]` | FP32 linear |

同步：

```text
MTE2 写完 L1
    ↓ MTE2_MTE1
MTE1 才能读取 fmapA1 / weightB1 / biasC1
```

Bias 再执行：

```text
biasC1 [16]  →  biasC2 [16]
C1 / L1          C2 / Bias Table
```

---

# 第四部分：L1 → A2 的真正含义

## 1. 完整 A 矩阵并没有物化

逻辑上存在：

```text
A[64,144]
```

但内存中不存在一块完整的 `A[64,144]` Buffer。

真实存在的是：

```text
fmapA1：X0 的真实数据，位于 L1
fmapA2：本次生成的 A[Mi,Kk]，位于 L0A
```

LoadData3D 的作用：

```text
fmapA1 中的 X
        │
        │ 根据虚拟 A 坐标、卷积参数和边界规则取数
        ▼
fmapA2 = A[Mi,Kk]
```

## 2. 四个 M/K 参数的直接语义

一次 LoadData3D 生成：

```text
fmapA2 = A[
    mStartPt : mStartPt + mExtension,
    kStartPt : kStartPt + kExtension
]
```

| 参数 | 直接含义 | 本例 |
|---|---|---:|
| `mStartPt` | 虚拟 A 的起始行 | `Mi×16` |
| `mExtension` | 从 A 中生成多少行 | 16 |
| `kStartPt` | 虚拟 A 的起始列 | `Kk×16` |
| `kExtension` | 从 A 中生成多少列 | 16 |

它们首先是 **A 矩阵坐标**，不是 X 的内存偏移。

例如 OT0、K4：

```text
mStartPt = 0
mExtension = 16
kStartPt = 64
kExtension = 16

fmapA2 = A[0:16,64:80]
```

## 3. A 坐标如何映射回 X

对于 `fmapA2[p,q]`：

```text
p = 0～15
q = 0～15
```

先得到虚拟 A 的全局坐标：

```text
m = mStartPt + p
k = kStartPt + q
```

### 3.1 M 解码：选择哪个输出位置

```text
ho = floor(m / Wo)
wo = m % Wo
```

本例 `Wo=8`：

```text
m0  → 输出位置 (0,0)
m1  → 输出位置 (0,1)
...
m7  → 输出位置 (0,7)
m8  → 输出位置 (1,0)
...
m15 → 输出位置 (1,7)
```

所以 M0 表示 16 个输出位置，也就是 16 个窗口锚点。

### 3.2 K 解码：选择窗口内部的元素

本例 K 的展开顺序：

```text
k = ((kh × Kw) + kw) × Ci + ci
```

反向解码：

```text
ci = k % Ci
windowIndex = floor(k / Ci)
kh = floor(windowIndex / Kw)
kw = windowIndex % Kw
```

### 3.3 映射为输入坐标

```text
hi = ho × strideH - padTop + kh × dilationH
wi = wo × strideW - padLeft + kw × dilationW
```

最终：

```text
fmapA2[p,q] =
    X[hi,wi,ci]    hi、wi 在有效输入范围内
    0              hi、wi 越界
```

## 4. 一个完整数值例子：OT0、K4、fmapA2[9,5]

当前：

```text
mStartPt = 0
kStartPt = 64
p = 9
q = 5
```

得到：

```text
m = 0 + 9 = 9
k = 64 + 5 = 69
```

M 解码：

```text
ho = floor(9 / 8) = 1
wo = 9 % 8 = 1
```

K 解码：

```text
ci = 69 % 16 = 5
windowIndex = floor(69 / 16) = 4
kh = floor(4 / 3) = 1
kw = 4 % 3 = 1
```

映射到 X：

```text
hi = 1×1 - 1 + 1×1 = 1
wi = 1×1 - 1 + 1×1 = 1
```

所以：

```text
fmapA2[9,5] = A[9,69] = X[1,1,5]
```

这个例子说明：

- A2 第 9 行由 M 坐标决定，对应第 9 个输出位置；
- A2 第 5 列由当前 K Tile 和局部 q 决定；
- LoadData3D 将二者结合，反推出真实 X 坐标。

## 5. K0～K8 与 3×3 的关系

本例恰好满足：

```text
Ci = 16
tileK = 16
```

所以每个 K Tile 恰好覆盖：

```text
一个 (kh,kw) 位置 × 全部 16 个输入通道
```

| K Tile | A 列范围 | 本例解码 |
|---|---:|---|
| K0 | `[0,16)` | `(kh=0,kw=0,ci=0～15)` |
| K1 | `[16,32)` | `(0,1,ci=0～15)` |
| K2 | `[32,48)` | `(0,2,ci=0～15)` |
| K3 | `[48,64)` | `(1,0,ci=0～15)` |
| K4 | `[64,80)` | `(1,1,ci=0～15)` |
| K5 | `[80,96)` | `(1,2,ci=0～15)` |
| K6 | `[96,112)` | `(2,0,ci=0～15)` |
| K7 | `[112,128)` | `(2,1,ci=0～15)` |
| K8 | `[128,144)` | `(2,2,ci=0～15)` |

准确理解：

```text
M：决定窗口滑到哪个输出位置
K：决定从每个窗口内部取哪个元素组
```

“K0 对应左上、K4 对应中心”只对当前 `Ci=tileK=16` 的 Demo 成立，不是通用规则。

---

# 第五部分：Weight 从 L1 到 B2

## 1. 当前核在 B1 中保存什么

OT0 的 `Nj=N0`，因此：

```text
weightB1 = B[:,N0]
         = B[0:144,0:16]
         = [144,16]
```

含义：

- 144 行：完整 K 方向；
- 16 列：输出通道 co0～co15。

## 2. 每次循环从 B1 取什么

每轮执行 LoadData2D：

```text
weightB2 = B[Kk,N0]
         = [16,16]
```

例如 K4：

```text
weightB2 = B[64:80,0:16]
```

B2 的数学含义：

```text
16 个当前 K 元素 × 16 个输出通道
```

本例 K4 中：

```text
行：窗口中心位置的 ci0～ci15
列：输出通道 co0～co15
```

格式变化：

```text
weightB1：NZ，B1 / L1
    ↓ LoadData2D，startIndex=Kk，ifTranspose=true
weightB2：ZN，B2 / L0B
```

---

# 第六部分：核内 9 次循环

## 1. 九次循环是否一样

九次循环使用相同的基本骨架：

```text
生成 A[Mi,Kk]
生成 B[Kk,Nj]
MTE1_M 同步
Mmad
```

但存在两类差异：

1. 每轮的 `kStartPt`、A2 内容和 B2 内容不同；
2. Iter 0 使用 Bias 初始化，Iter 1～8 原位累加。

## 2. 循环结构

```text
K Loop · Iter 0～8
├─ Iter 0 · Initialize
│  ├─ LoadData3D / LoadData2D
│  ├─ MTE1_M
│  └─ Mmad + Bias → Acc0
└─ Iter 1～8 · Accumulate ×8
   ├─ M_MTE1
   ├─ LoadData3D / LoadData2D
   ├─ MTE1_M
   └─ Mmad accumulate → Acc1…Acc8
```

### 2.1 Iter 0

```text
A2 = A[Mi,K0]
B2 = B[K0,Nj]

Acc0 = A[Mi,K0] × B[K0,Nj] + D[Nj]
```

此时：

- A2/B2 中没有上一轮数据，因此不需要 `M_MTE1`；
- Bias 只在本轮加入；
- `accumCo1` 第一次得到有效结果。

### 2.2 Iter 1～8

每轮先等待 Cube 已经读完上一轮 A2/B2：

```text
M_MTE1
```

随后覆盖单 Buffer：

```text
fmapA2：A[Mi,K(k-1)] → A[Mi,Kk]
weightB2：B[K(k-1),Nj] → B[Kk,Nj]
```

继续累加：

```text
Acck = Acc(k-1) + A[Mi,Kk] × B[Kk,Nj]
```

### 2.3 九轮明细

| Iter | K 范围 | A Tile | B Tile | 结果 |
|---:|---:|---|---|---|
| 0 | `[0,16)` | `A[Mi,K0]` | `B[K0,Nj]` | `Acc0 = A×B + Bias` |
| 1 | `[16,32)` | `A[Mi,K1]` | `B[K1,Nj]` | `Acc1` |
| 2 | `[32,48)` | `A[Mi,K2]` | `B[K2,Nj]` | `Acc2` |
| 3 | `[48,64)` | `A[Mi,K3]` | `B[K3,Nj]` | `Acc3` |
| 4 | `[64,80)` | `A[Mi,K4]` | `B[K4,Nj]` | `Acc4` |
| 5 | `[80,96)` | `A[Mi,K5]` | `B[K5,Nj]` | `Acc5` |
| 6 | `[96,112)` | `A[Mi,K6]` | `B[K6,Nj]` | `Acc6` |
| 7 | `[112,128)` | `A[Mi,K7]` | `B[K7,Nj]` | `Acc7` |
| 8 | `[128,144)` | `A[Mi,K8]` | `B[K8,Nj]` | `Acc8` |

九次循环共同完成同一个输出 Tile：

```text
C[Mi,Nj]
=
A[Mi,K0]×B[K0,Nj]
+ A[Mi,K1]×B[K1,Nj]
+ ...
+ A[Mi,K8]×B[K8,Nj]
+ D[Nj]
```

---

# 第七部分：A2、B2 为什么能相乘

当前一次迭代：

```text
fmapA2   = [16,16]
weightB2 = [16,16]
```

两个 Shape 中的 16 含义不同：

| Tensor | 行 | 列 |
|---|---|---|
| `fmapA2` | 16 个输出空间位置 | 当前 16 个 K 元素 |
| `weightB2` | 当前 16 个 K 元素 | 16 个输出通道 |

所以：

```text
A2
[16 个位置, 16 个 K 元素]

×

B2
[16 个 K 元素, 16 个输出通道]

=

局部结果
[16 个位置, 16 个输出通道]
```

结果累加到：

```text
accumCo1：C[Mi,Nj] [16,16]
位置：CO1 / L0C
dtype：FP32
format：NZ
```

---

# 第八部分：Fixpipe 与输出合并

## 1. 九次累加完成

Iter 8 后：

```text
accumCo1 = Acc8[Mi,Nj]
Shape：[16,16]
dtype：FP32
format：NZ
位置：CO1 / L0C
```

同步：

```text
Cube 写完 accumCo1
    ↓ M_FIX
Fixpipe 才能读取
```

## 2. Fixpipe 同时完成四件事

```text
NZ → ND
FP32 → FP16
ReLU
写入 outputGm_
```

| 输入/输出 | Shape | dtype | format | 位置 |
|---|---:|---|---|---|
| `accumCo1` | `[16,16]` | FP32 | NZ | CO1 / L0C |
| 当前写回 Tile | `[16,16]` | FP16 | ND `[M,Co]` | GM Output |

写回参数：

```text
mSize = 16
nSize = 16
srcStride = 16
dstStride = 32
outputOffset = mStart × 32 + nStart
```

## 3. 八个核如何组成完整输出

每个核直接写完整输出中的最终地址：

| 核 | 写入范围 |
|---|---|
| OT0 | `Y[0:16,0:16]` |
| OT1 | `Y[0:16,16:32]` |
| OT2 | `Y[16:32,0:16]` |
| OT3 | `Y[16:32,16:32]` |
| OT4 | `Y[32:48,0:16]` |
| OT5 | `Y[32:48,16:32]` |
| OT6 | `Y[48:64,0:16]` |
| OT7 | `Y[48:64,16:32]` |

这些地址互不重叠，因此不需要额外的“拼接 Kernel”。所有 Block 完成后，GM 中自然形成完整：

```text
Y0 [64,32]
```

---

# 第九部分：Instruction 级 Trace 附录

本附录按页面 Stage 组织源码确定的逻辑顺序。它不是 profiling 时间线，也不表达各硬件流水单元的实测持续时间。

## 1. Stage 总览

| Stage | Instruction / Control | 关键结果 |
| ---: | --- | --- |
| 1. Input Shape | 固定输入、权重、Bias、输出契约 | X/W/D/Y 的逻辑与物理 Shape |
| 2. Host Tiling | 推导 Ho/Wo、M/K/N、Tile 数 | `M/K/N=64/144/32`，Tile 数 `4/9/2` |
| 3. Host 执行配置 | 元素数、`blockDim=8`、workspace、OT→Mi/Nj | 每个 Block 负责一个 `[16,16]` Output Tile |
| 4. Allocate Memory | GM Tensor 句柄、LocalTensor 地址视图 | A1/B1/C1/A2/B2/C2/CO1 可用 |
| 5. Copy Inputs | MTE2 DataCopy / Nd2Nz | GM X/W/D → A1/B1/C1 |
| 6. Sync | `MTE2_MTE1` | MTE1 可以读取 A1/B1/C1 |
| 7. Copy Data C2 | MTE1 DataCopy | `biasC1 → biasC2` |
| 8. K Loop · Iter 0～8 | Load、同步、Mmad | `Acc0 → … → Acc8` |
| 9. M_FIX Sync | `M_FIX` | Fixpipe 可以读取最终 CO1 |
| 10. Fixpipe Output | Fixpipe | NZ→ND、FP32→FP16、ReLU、写回 GM |

## 2. Stage 1～4：Shape、Tiling、执行配置与 Buffer

### 2.1 全局参数

| 参数 | 当前值 | 含义 |
| --- | ---: | --- |
| 输入 / 权重 / Bias | `[1,16,8,8]` / `[32,16,3,3]` / `[32]` | FP16 / FP16 / FP32 |
| 卷积参数 | stride=1，padding=1，dilation=1 | 推导 `Ho/Wo=8/8` |
| Cube M/K/N | `64/144/32` | `A[64,144] × B[144,32] + D[32]` |
| tileM/K/N | `16/16/16` | 每次 Mmad 的 Tile |
| Tile 数 | `4/9/2` | M0～M3、K0～K8、N0～N1 |
| Output Tile / blockDim | `8/8` | `OT=Mi×2+Nj` |
| Tail | 无 | M、K、N 均可被 16 整除 |

### 2.2 Tensor 与 Buffer

| Tensor / 变量 | Shape | 位置 | dtype | format / 作用 |
| --- | --- | --- | --- | --- |
| `featureGm_` | `[1,1,8,8,16]` | GM | FP16 | NC1HWC0；所有核读取完整 Feature |
| `weightGm_` | `[144,32]` | GM | FP16 | ND `[K,Co]`；当前核读取 `[144,16]` |
| `biasGm_` | `[32]` | GM | FP32 | 当前核读取 `D[Nj] [16]` |
| `outputGm_` | `[64,32]` | GM | FP16 | 当前核写 `C[Mi,Nj] [16,16]` |
| `fmapA1` | `[1,1,8,8,16]` | A1 / L1 | FP16 | NC1HWC0，2048 B |
| `weightB1` | `[144,16]` | B1 / L1 | FP16 | NZ，4608 B |
| `biasC1` / `biasC2` | `[16]` / `[16]` | C1 / L1；C2 / BT | FP32 | 当前 `D[Nj]`，各 64 B |
| `fmapA2` / `weightB2` | `[16,16]` / `[16,16]` | A2 / L0A；B2 / L0B | FP16 | ZZ / ZN，各 512 B |
| `accumCo1` | `[16,16]` | CO1 / L0C | FP32 | NZ，1024 B |

## 3. Stage 5～7：输入搬运、同步与 Bias Table

| 顺序 | Instruction | 数据流 | 结果 |
| ---: | --- | --- | --- |
| 1 | MTE2 DataCopy | `featureGm_ → fmapA1` | 完整 Feature 进入 A1 |
| 2 | MTE2 DataCopy Nd2Nz | `weightGm_[Nj] → weightB1` | 当前 N Tile 权重进入 B1 |
| 3 | MTE2 DataCopy | `biasGm_[Nj] → biasC1` | 当前 Bias Tile 进入 C1 |
| 4 | `MTE2_MTE1 Sync` | MTE2 → MTE1 | 发布 A1/B1/C1 ready |
| 5 | MTE1 DataCopy | `biasC1 → biasC2` | Bias 进入 C2 / Bias Table |

## 4. Stage 8：K Loop · Iter 0～8

K Loop 共执行 9 次 Mmad。Iter 0 使用 Bias 初始化 `accumCo1`；Iter 1～8 复用单缓冲 A2/B2，并把新 K Tile 原位累加到同一个 `accumCo1`。

### 4.1 两类迭代的 Instruction 顺序

| Iter 0 · Initialize | Iter 1～8 · Accumulate ×8 |
| --- | --- |
| `Load Data A2/B2` | `M_MTE1 Sync` |
| `MTE1_M Sync` | `Load Data A2/B2` |
| `Mmad + Bias → Acc0` | `MTE1_M Sync` |
|  | `Mmad accumulate → Acc1…Acc8` |

### 4.2 Iter 0

| Instruction | 输入 | 输出 / 状态 |
| --- | --- | --- |
| `LoadData3D` | `fmapA1`，`mStart=Mi×16`，`kStart=0` | `fmapA2=A[Mi,K0]`，ZZ |
| `LoadData2D` | `weightB1`，`startIndex=0`，`ifTranspose=true` | `weightB2=B[K0,Nj]`，ZN |
| `MTE1_M Sync` | MTE1 → Cube | A2/B2 ready |
| `Mmad + Bias` | `fmapA2`、`weightB2`、`biasC2` | `Acc0=A[Mi,K0]×B[K0,Nj]+D[Nj]` |

### 4.3 Iter 1～8

每轮 `k=1..8` 执行同一个 Loop Body：

| Instruction | 依赖 / 数据流 | 输出 / 状态 |
| --- | --- | --- |
| `M_MTE1 Sync` | Cube 已读完上一轮 A2/B2 | A2/B2 可以覆盖 |
| `LoadData3D / LoadData2D` | `A[Mi,Kk] / B[Kk,Nj] → fmapA2 / weightB2` | 新 K Tile 写入 A2/B2 |
| `MTE1_M Sync` | MTE1 → Cube | 新 A2/B2 ready |
| `Mmad accumulate` | `Acc(k-1)+A[Mi,Kk]×B[Kk,Nj]` | `Acck`，最终 Iter 8 得到 `Acc8` |

## 5. Stage 9～10：M_FIX 与 Fixpipe Output

| 顺序 | Instruction | 输入 | 输出 / 变化 |
| ---: | --- | --- | --- |
| 1 | `M_FIX Sync` | 最终 `accumCo1=Acc8` | Fixpipe 可以读取 CO1 |
| 2 | `Fixpipe` | CO1 `[16,16]` FP32 NZ | GM `[16,16]` FP16 ND；F322F16 + ReLU |

Fixpipe 使用 `mSize=16`、`nSize=16`、`srcStride=16`、`dstStride=32`，写入：

```text
outputGm_[mStart × 32 + nStart]
```

---

# 第十部分：最短记忆版

```text
1. 卷积先建立逻辑矩阵视图：
   A[64,144] × B[144,32] → C[64,32]

2. 输出 C 沿 M 切 4 份、沿 N 切 2 份：
   4×2 = 8 个 Output Tile → 8 个 Block 任务

3. 每个核：
   - 搬完整 Feature 到 fmapA1
   - 搬当前 Nj 的 Weight 到 weightB1
   - 搬当前 Nj 的 Bias 到 biasC2

4. 核内循环 9 次：
   - LoadData3D 生成 A[Mi,Kk] → fmapA2
   - LoadData2D 生成 B[Kk,Nj] → weightB2
   - Mmad 累加到同一个 accumCo1

5. Iter 0 加 Bias；Iter 1～8 继续累加。

6. Fixpipe：
   NZ→ND、FP32→FP16、ReLU，并直接写入完整输出中的目标区域。
```

最关键的一句话：

> `M` 决定计算哪些输出位置，`N` 决定计算哪些输出通道，`K` 决定一次累加哪一段输入与权重；完整 A 只是逻辑视图，LoadData3D 每次只物化当前 `A[Mi,Kk]`。
