## Annotation 1

已按你的要求改写：每一行只描述一个 Tensor 变量或一个明确的逻辑 Tensor 对象，不再使用“A1/B1/C1”作为 Tensor 名字，也不再把多个 Tensor、位置、shape 合并在一个单元格里。

先说明当前基线：以下内容来自 `conv_bias_relu_complete_demo` 固定 Demo。`padding=1、stride=1、dilation=1` 是 Demo 补全时选择的参数，不代表已经从真实业务代码验证。

## 全局 Shape 与分块参数

| 参数                       | 当前值   | 对 Shape 的影响                               |
| -------------------------- | -------- | --------------------------------------------- |
| `n`                        | 1        | Batch 维                                      |
| `ci`                       | 16       | Feature 输入通道；等于 C0，不产生通道 padding |
| `hi`                       | 8        | 输入 Feature 高度                             |
| `wi`                       | 8        | 输入 Feature 宽度                             |
| `co`                       | 32       | 输出通道，也是 Cube N                         |
| `kh`                       | 3        | 卷积核高度                                    |
| `kw`                       | 3        | 卷积核宽度                                    |
| `strideH/W`                | 1/1      | 决定相邻输出位置的取窗步幅                    |
| `padTop/Bottom/Left/Right` | 1/1/1/1  | 决定边界取窗和 `Ho/Wo`；不扩大实体 A1         |
| `dilationH/W`              | 1/1      | 有效 kernel 仍为 `3×3`                        |
| `ho/wo`                    | 8/8      | 由 padding=1 推导                             |
| `M`                        | 64       | `Nbatch×Ho×Wo=1×8×8`                          |
| `K`                        | 144      | `Ci×Kh×Kw=16×3×3`                             |
| `N Cube`                   | 32       | 等于输出通道 `Co`                             |
| `tileM/K/N`                | 16/16/16 | 每次 Mmad 的 Tile 规格                        |
| Tile 数                    | 4/9/2    | `M0..M3`、`K0..K8`、`N0..N1`                  |
| Output Tile                | 8        | `OT=Mi×2+Nj`                                  |
| blockDim                   | 8        | 每核处理一个 Output Tile                      |
| Tail                       | 无       | M、K、N 都可被 16 整除                        |

------

## 阶段 1：Input Shape

Host 固定输入、权重、Bias、输出的逻辑契约。

| Tensor / 变量 | Shape                                  | Tile        | Shape 相关参数             | 位置    | dtype | format                                 | Align                   |
| ------------- | -------------------------------------- | ----------- | -------------------------- | ------- | ----- | -------------------------------------- | ----------------------- |
| `Feature X0`  | 逻辑 `[1,16,8,8]`；物理 `[1,1,8,8,16]` | 尚未分 Tile | `n=1, ci=16, hi=8, wi=8`   | GM 输入 | FP16  | 逻辑 NCHW；物理 NC1HWC0                | GM 基址对齐未在源码声明 |
| `Filter W0`   | 逻辑 `[32,16,3,3]`；物理 `[144,32]`    | 尚未分 Tile | `co=32, ci=16, kh=3, kw=3` | GM 输入 | FP16  | 逻辑 OIHW；物理 ND `[K,Co]`            | GM 基址对齐未在源码声明 |
| `Bias D0`     | `[32]`                                 | 尚未分 Tile | `co=32`                    | GM 输入 | FP32  | ND / linear                            | GM 基址对齐未在源码声明 |
| `Output Y0`   | 语义 `[1,32,8,8]`；物理 `[64,32]`      | 尚未分 Tile | `ho=8, wo=8, co=32`        | GM 输出 | FP16  | 语义 NCHW；物理 ND `[M,Co]`，等价 NHWC | GM 基址对齐未在源码声明 |

Padding 在本阶段只是卷积参数。此时没有创建 `[1,1,10,10,16]` Tensor。

------

## 阶段 2：Host Tiling

Host 将卷积转换成 Cube 的 M/K/N 计算空间。

| Tensor / 逻辑对象                 | Shape      | Tile 编号                            | Shape 相关参数                 | 位置         | dtype           | format   | Align         |
| --------------------------------- | ---------- | ------------------------------------ | ------------------------------ | ------------ | --------------- | -------- | ------------- |
| `A`（逻辑 Cube 视图，无独立变量） | `[64,144]` | `A[Mi,Kk]`；`Mi=M0..M3`，`Kk=K0..K8` | `M=Ho×Wo=64`；`K=Ci×Kh×Kw=144` | 逻辑计算空间 | FP16            | 逻辑 M×K | 不分配 Buffer |
| `B`（逻辑 Cube 视图，无独立变量） | `[144,32]` | `B[Kk,Nj]`；`Nj=N0..N1`              | `K=144`；`N=Co=32`             | 逻辑计算空间 | FP16            | 逻辑 K×N | 不分配 Buffer |
| `C`（逻辑 Cube 视图，无独立变量） | `[64,32]`  | `C[Mi,Nj]`                           | `M=64`；`N=32`                 | 逻辑计算空间 | FP32 accumulate | 逻辑 M×N | 不分配 Buffer |
| `D`（逻辑 Bias 视图，无独立变量） | `[32]`     | `D[Nj]`，每 Tile 16 个通道           | `tileN=16`                     | 逻辑计算空间 | FP32            | linear N | 不分配 Buffer |

------

## 阶段 3：Host 执行配置

建立 GM Tensor 句柄、输出 Tile 与 AI Core 的映射。

| Tensor / 变量 | Shape                         | Tile / Core 映射                     | Shape 相关参数                  | 位置 | dtype | format      | Align          |
| ------------- | ----------------------------- | ------------------------------------ | ------------------------------- | ---- | ----- | ----------- | -------------- |
| `featureGm_`  | `[1,1,8,8,16]`；1024 elements | 所有核读取完整 Feature               | `featureGmElements=1024`        | GM   | FP16  | NC1HWC0     | 基址对齐未声明 |
| `filterGm_`   | `[144,32]`；4608 elements     | 当前核读取 `Nj` 对应的 `[144,16]`    | `nStart=Nj×16`                  | GM   | FP16  | ND `[K,Co]` | 基址对齐未声明 |
| `biasGm_`     | `[32]`                        | 当前核读取 `D[Nj]`，shape `[16]`     | `nStart=Nj×16`                  | GM   | FP32  | ND / linear | 基址对齐未声明 |
| `outputGm_`   | `[64,32]`；2048 elements      | 当前核写 `C[Mi,Nj]`，shape `[16,16]` | `outputOffset=mStart×32+nStart` | GM   | FP16  | ND `[M,Co]` | 基址对齐未声明 |

Core 映射：

```
OT = blockIdx
Mi = OT / 2
Nj = OT % 2
mStart = Mi × 16
nStart = Nj × 16
```

------

## 阶段 4：Allocate Memory

本阶段只建立 LocalTensor 视图，不搬运数据。

| Tensor 变量名 | Shape          | Tile 身份                   | 位置            | dtype | format  | 大小   | 起始地址 | Align                               |
| ------------- | -------------- | --------------------------- | --------------- | ----- | ------- | ------ | -------- | ----------------------------------- |
| `fmapA1`      | `[1,1,8,8,16]` | 完整 Feature，不按 Mi 切分  | A1 / L1         | FP16  | NC1HWC0 | 2048 B | 0        | 512 B                               |
| `weightB1`    | `[144,16]`     | 当前 `Nj` 的完整 K 方向权重 | B1 / L1         | FP16  | NZ      | 4608 B | 2048     | 512 B                               |
| `biasC1`      | `[16]`         | 当前 `D[Nj]`                | C1 / L1         | FP32  | linear  | 64 B   | 6656     | 128 B                               |
| `biasC2`      | `[16]`         | 当前 `D[Nj]`                | C2 / Bias Table | FP32  | linear  | 64 B   | 0        | 64 B burst                          |
| `fmapA2`      | `[16,16]`      | 当前 `A[Mi,Kk]`             | A2 / L0A        | FP16  | ZZ      | 512 B  | 0        | 独立 Buffer；具体要求待目标环境验证 |
| `weightB2`    | `[16,16]`      | 当前 `B[Kk,Nj]`             | B2 / L0B        | FP16  | ZN      | 512 B  | 0        | 独立 Buffer；具体要求待目标环境验证 |
| `accumCo1`    | `[16,16]`      | 当前 `C[Mi,Nj]`             | CO1 / L0C       | FP32  | NZ      | 1024 B | 0        | 独立 Buffer；具体要求待目标环境验证 |

------

## 阶段 5：Copy Inputs

MTE2 将 GM 数据搬入 L1。

| 目标 Tensor 变量 | 来源 Tensor  | 目标 Shape     | Tile 身份    | Shape 相关参数 | 目标位置 | dtype | 格式变化        | 搬运/对齐                           |
| ---------------- | ------------ | -------------- | ------------ | -------------- | -------- | ----- | --------------- | ----------------------------------- |
| `fmapA1`         | `featureGm_` | `[1,1,8,8,16]` | 完整 Feature | `1024 FP16`    | A1 / L1  | FP16  | NC1HWC0→NC1HWC0 | DataCopy 2048 B；A1 地址 512 B 对齐 |
| `weightB1`       | `filterGm_`  | `[144,16]`     | 当前 `Nj`    | `nStart=Nj×16` | B1 / L1  | FP16  | ND→NZ           | 4608 B；B1 地址 512 B 对齐          |
| `biasC1`         | `biasGm_`    | `[16]`         | 当前 `D[Nj]` | `nStart=Nj×16` | C1 / L1  | FP32  | ND→linear       | 64 B；C1 地址 128 B 对齐            |

`fmapA1` 仍然是 `[1,1,8,8,16]`，没有物化 padding。

------

## 阶段 6：MTE2_MTE1 Sync

本阶段不改变 Tensor shape 或 format。

| Tensor 变量名 | Shape          | 当前状态                      | 位置    | dtype | format  | Align |
| ------------- | -------------- | ----------------------------- | ------- | ----- | ------- | ----- |
| `fmapA1`      | `[1,1,8,8,16]` | MTE2 写入完成，等待 MTE1 读取 | A1 / L1 | FP16  | NC1HWC0 | 512 B |
| `weightB1`    | `[144,16]`     | MTE2 写入完成，等待 MTE1 读取 | B1 / L1 | FP16  | NZ      | 512 B |
| `biasC1`      | `[16]`         | MTE2 写入完成，等待 MTE1 读取 | C1 / L1 | FP32  | linear  | 128 B |

同步依赖：`MTE2_MTE1`。

------

## 阶段 7：Copy Data C2

MTE1 将 Bias 从 L1 搬到 Bias Table。

| 目标 Tensor 变量 | 来源 Tensor | Shape  | Tile 身份    | 位置            | dtype | format | 搬运/对齐             |
| ---------------- | ----------- | ------ | ------------ | --------------- | ----- | ------ | --------------------- |
| `biasC2`         | `biasC1`    | `[16]` | 当前 `D[Nj]` | C2 / Bias Table | FP32  | linear | 64 B；使用 64 B burst |

`biasC1` 保持在 C1/L1；`biasC2` 是其 Bias Table 副本。

------

## 阶段 8：K Loop 容器 · Iter 0～8

这是源码中真实的 `for (kTileIndex = 0; kTileIndex < 9; ++kTileIndex)` 控制结构，包含全部 9 次 K 迭代。循环容器本身不是硬件指令，不独立搬运或计算 Tensor。

进入循环前：

| Tensor 变量名 | Shape          | 当前状态                                  | 位置            | dtype | format  |
| ------------- | -------------- | ----------------------------------------- | --------------- | ----- | ------- |
| `fmapA1`      | `[1,1,8,8,16]` | 完整 Feature 已在 L1，供 9 次迭代重复读取 | A1 / L1         | FP16  | NC1HWC0 |
| `weightB1`    | `[144,16]`     | 当前 `Nj` 权重已在 L1，供 9 次迭代分段读取 | B1 / L1         | FP16  | NZ      |
| `biasC2`      | `[16]`         | Bias 已在 Bias Table，仅供 Iter 0 使用    | C2 / Bias Table | FP32  | linear  |
| `fmapA2`      | `[16,16]`      | 已分配，尚无有效 K Tile                    | A2 / L0A        | FP16  | ZZ      |
| `weightB2`    | `[16,16]`      | 已分配，尚无有效 K Tile                    | B2 / L0B        | FP16  | ZN      |
| `accumCo1`    | `[16,16]`      | 已分配，尚无有效累加结果                   | CO1 / L0C       | FP32  | NZ      |

循环中的 9 次迭代分为两种执行语义：

| 迭代范围 | 次数 | 与上一轮 A2/B2 的同步 | Mmad 语义 |
| -------- | ---: | ---------------------- | --------- |
| Iter 0   | 1    | 无；A2/B2 尚未保存上一轮数据 | 使用 `A[Mi,K0]`、`B[K0,Nj]` 和 `D[Nj]` 初始化 `Acc0` |
| Iter 1～8 | 8   | 先执行 `M_MTE1`，确认 Cube 已读完上一轮 A2/B2 | 读取新的 K Tile，原位累加到 `accumCo1`，不再加入 Bias |

执行结构：

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

------

## 阶段 9：Load Data A2/B2 · Iter 0

这是 Iter 0 的数据装载，也是 padding、stride、filter、dilation 第一次真正参与取窗的阶段。

### Feature 路径

| Tensor 变量名 | Shape               | Tile 身份               | Shape 相关参数                                               | 位置     | dtype | format  | Align        |
| ------------- | ------------------- | ----------------------- | ------------------------------------------------------------ | -------- | ----- | ------- | ------------ |
| `fmapA1`      | 实体 `[1,1,8,8,16]` | 当前 `Mi` 的取窗来源    | `l1H=8, l1W=8, channelSize=16`                               | A1 / L1  | FP16  | NC1HWC0 | 512 B        |
| `fmapA2`      | `[16,16]`           | `A[Mi,K0]`，`K0=[0,16)` | `mStart=Mi×16`、`kStart=0`、filter=`3×3`、stride=1、dilation=1、padList=`[1,1,1,1]`、padValue=0 | A2 / L0A | FP16  | ZZ      | 512 B Buffer |

Padding 不增加 `fmapA1` 的 shape。LoadData3D 遇到 A1 边界外坐标时，直接向 `fmapA2` 生成 FP16 零值。

### Filter 路径

| Tensor 变量名 | Shape      | Tile 身份               | Shape 相关参数                     | 位置     | dtype | format | Align        |
| ------------- | ---------- | ----------------------- | ---------------------------------- | -------- | ----- | ------ | ------------ |
| `weightB1`    | `[144,16]` | 当前 `Nj` 的完整 K 权重 | `tileN=16`                         | B1 / L1  | FP16  | NZ     | 512 B        |
| `weightB2`    | `[16,16]`  | `B[K0,Nj]`，`K0=[0,16)` | `startIndex=0`、`ifTranspose=true` | B2 / L0B | FP16  | ZN     | 512 B Buffer |

------

## 阶段 10：MTE1_M Sync · Iter 0

本阶段不改变 Tensor 内容，只发布 Iter 0 的 A2/B2 ready。

| Tensor 变量名 | Shape     | Tile 身份  | 状态                         | 位置     | dtype | format |
| ------------- | --------- | ---------- | ---------------------------- | -------- | ----- | ------ |
| `fmapA2`      | `[16,16]` | `A[Mi,K0]` | MTE1 写入完成，Cube 可以读取 | A2 / L0A | FP16  | ZZ     |
| `weightB2`    | `[16,16]` | `B[K0,Nj]` | MTE1 写入完成，Cube 可以读取 | B2 / L0B | FP16  | ZN     |

同步依赖：`MTE1_M`。

------

## 阶段 11：Iter 0 Mmad · Initialize

首次 Mmad 使用 Bias 初始化 `accumCo1`。

| Tensor 变量名 | Shape     | Tile 身份     | 作用                                 | 位置            | dtype | format | 大小   |
| ------------- | --------- | ------------- | ------------------------------------ | --------------- | ----- | ------ | ------ |
| `fmapA2`      | `[16,16]` | `A[Mi,K0]`    | Mmad 左操作数                        | A2 / L0A        | FP16  | ZZ     | 512 B  |
| `weightB2`    | `[16,16]` | `B[K0,Nj]`    | Mmad 右操作数                        | B2 / L0B        | FP16  | ZN     | 512 B  |
| `biasC2`      | `[16]`    | `D[Nj]`       | 初始化每个输出通道的 Bias            | C2 / Bias Table | FP32  | linear | 64 B   |
| `accumCo1`    | `[16,16]` | `Acc0[Mi,Nj]` | 保存 K0 乘积与 Bias 的 FP32 累加结果 | CO1 / L0C       | FP32  | NZ     | 1024 B |

计算：

```text
Acc0 = A[Mi,K0] × B[K0,Nj] + D[Nj]
```

------

## 阶段 12：Iter 1～8 Loop Body · Accumulate ×8

Iter 1～8 使用完全相同的循环体。Iter 8 没有独有指令或条件分支；它只是最后一次产生后续 Fixpipe 所需的最终 `Acc8`。

每轮依次执行：

```text
M_MTE1
→ LoadData3D / LoadData2D
→ MTE1_M
→ Mmad accumulate
```

| Tensor 变量名 | Shape          | Tile 编号                      | 作用                    | 位置      | dtype | format  | Align         |
| ------------- | -------------- | ------------------------------ | ----------------------- | --------- | ----- | ------- | ------------- |
| `fmapA1`      | `[1,1,8,8,16]` | 生成 `A[Mi,K1]` 至 `A[Mi,K8]`  | 每轮 LoadData3D 的来源  | A1 / L1   | FP16  | NC1HWC0 | 512 B         |
| `fmapA2`      | `[16,16]`      | 依次保存 `A[Mi,K1]`…`A[Mi,K8]` | 覆盖上一轮 A Tile       | A2 / L0A  | FP16  | ZZ      | 512 B Buffer  |
| `weightB1`    | `[144,16]`     | 生成 `B[K1,Nj]` 至 `B[K8,Nj]`  | 每轮 LoadData2D 的来源  | B1 / L1   | FP16  | NZ      | 512 B         |
| `weightB2`    | `[16,16]`      | 依次保存 `B[K1,Nj]`…`B[K8,Nj]` | 覆盖上一轮 B Tile       | B2 / L0B  | FP16  | ZN      | 512 B Buffer  |
| `accumCo1`    | `[16,16]`      | `Acc1`…`Acc8`                  | 原位累加，不再加入 Bias | CO1 / L0C | FP32  | NZ      | 1024 B Buffer |

每轮的实际 K 范围与结果：

| 迭代 | K 范围      | 读取 A Tile  | 读取 B Tile  | 产生结果 |
| ---- | ----------- | ------------ | ------------ | -------- |
| Iter 1 | `[16,32)`   | `A[Mi,K1]` | `B[K1,Nj]` | `Acc1` |
| Iter 2 | `[32,48)`   | `A[Mi,K2]` | `B[K2,Nj]` | `Acc2` |
| Iter 3 | `[48,64)`   | `A[Mi,K3]` | `B[K3,Nj]` | `Acc3` |
| Iter 4 | `[64,80)`   | `A[Mi,K4]` | `B[K4,Nj]` | `Acc4` |
| Iter 5 | `[80,96)`   | `A[Mi,K5]` | `B[K5,Nj]` | `Acc5` |
| Iter 6 | `[96,112)`  | `A[Mi,K6]` | `B[K6,Nj]` | `Acc6` |
| Iter 7 | `[112,128)` | `A[Mi,K7]` | `B[K7,Nj]` | `Acc7` |
| Iter 8 | `[128,144)` | `A[Mi,K8]` | `B[K8,Nj]` | `Acc8` |

所有迭代的 K Tile 都包含 16 个元素，不存在 K tail。

------

## 阶段 13：M_FIX Sync

本阶段不改变 Tensor shape。

| Tensor 变量名 | Shape     | Tile 身份          | 当前状态                         | 位置      | dtype | format |
| ------------- | --------- | ------------------ | -------------------------------- | --------- | ----- | ------ |
| `accumCo1`    | `[16,16]` | 最终 `Acc8[Mi,Nj]` | Cube 写入完成，等待 Fixpipe 读取 | CO1 / L0C | FP32  | NZ     |

同步依赖：`M_FIX`。

------

## 阶段 14：Fixpipe Output

完成格式转换、dtype 转换、ReLU 和 GM 写回。

| Tensor 变量名 | Shape                              | Tile 身份     | Shape 相关参数                                  | 位置      | dtype | format      | 搬运/对齐                             |
| ------------- | ---------------------------------- | ------------- | ----------------------------------------------- | --------- | ----- | ----------- | ------------------------------------- |
| `accumCo1`    | `[16,16]`                          | `Acc8[Mi,Nj]` | `mSize=16, nSize=16, srcStride=16`              | CO1 / L0C | FP32  | NZ          | 读取 1024 B                           |
| `outputGm_`   | 当前写回 `[16,16]`；全局 `[64,32]` | `C[Mi,Nj]`    | `dstStride=32`；`outputOffset=mStart×32+nStart` | GM 输出   | FP16  | ND `[M,Co]` | 每 Tile 写回 512 B；GM 基址对齐未声明 |

Fixpipe 同时执行：

```
NZ → ND
FP32 → FP16
ReLU
写入 outputGm_[mStart×32+nStart]
```

## 最关键的命名规则

后续页面和文档建议统一使用：

```
变量名：fmapA1
Shape：[1,1,8,8,16]
位置：A1 / L1
dtype：FP16
format：NC1HWC0
```

不要再写成：

```
A1 [1,1,8,8,16]
```

因为 `A1` 是存储位置，`fmapA1` 才是 Tensor 变量名。
