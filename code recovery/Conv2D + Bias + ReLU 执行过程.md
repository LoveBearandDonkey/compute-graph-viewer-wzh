# Conv2D + Bias + ReLU 执行过程

## 1. 算子计算对象

### 1.1 逻辑输入与输出

| Tensor       | 逻辑 Shape    | GM 物理 Shape  | Format  | Dtype | 元素数 | 数据量 |
| ------------ | ------------- | -------------- | ------- | ----- | ------ | ------ |
| Feature `X0` | `[1,16,8,8]`  | `[1,1,8,8,16]` | NC1HWC0 | FP16  | 1024   | 2048 B |
| Filter `W0`  | `[32,16,3,3]` | `[144,32]`     | ND      | FP16  | 4608   | 9216 B |
| Bias `D0`    | `[32]`        | `[32]`         | ND      | FP32  | 32     | 128 B  |
| Output `Y0`  | `[1,32,8,8]`  | `[64,32]`      | ND      | FP16  | 2048   | 4096 B |

输出物理布局 `[64,32]` 等价于 NHWC `[1,8,8,32]`，不等价于 NCHW `[1,32,8,8]`。

### 1.2 Cube 矩阵表达

```text
A[64,144] × B[144,32] + Bias[32] → C[64,32]
```

| 维度 | 来源           | 数值 |
| ---- | -------------- | ---- |
| M    | `N × Ho × Wo`  | 64   |
| K    | `Ci × Kh × Kw` | 144  |
| N    | `Co`           | 32   |

------

## 2. Tensor 切分与编号

### 2.1 Tile 规格

| 方向 | Tile 大小 | Tile 数量 | 编号     |
| ---- | --------- | --------- | -------- |
| M    | 16        | 4         | `M0～M3` |
| K    | 16        | 9         | `K0～K8` |
| N    | 16        | 2         | `N0～N1` |

每个输出 Tile：

```text
C[Mi,Nj] = [16,16]
```

每次 K 迭代：

```text
A[Mi,Kk] = [16,16]
B[Kk,Nj] = [16,16]
```

### 2.2 编号范围

| 编号     | 覆盖范围                      |
| -------- | ----------------------------- |
| `M0`     | 输出位置 `0～15`              |
| `M1`     | 输出位置 `16～31`             |
| `M2`     | 输出位置 `32～47`             |
| `M3`     | 输出位置 `48～63`             |
| `K0～K8` | 每块覆盖连续 16 个 K 展平元素 |
| `N0`     | 输出通道 `0～15`              |
| `N1`     | 输出通道 `16～31`             |

### 2.3 输出 Tile 与 AI Core 分配

```text
OT = Mi × 2 + Nj
blockDim = 8
```

| AI Core / OT | M Tile | N Tile | 输出位置 | 输出通道 |
| ------------ | ------ | ------ | -------- | -------- |
| `OT0`        | `M0`   | `N0`   | `0～15`  | `0～15`  |
| `OT1`        | `M0`   | `N1`   | `0～15`  | `16～31` |
| `OT2`        | `M1`   | `N0`   | `16～31` | `0～15`  |
| `OT3`        | `M1`   | `N1`   | `16～31` | `16～31` |
| `OT4`        | `M2`   | `N0`   | `32～47` | `0～15`  |
| `OT5`        | `M2`   | `N1`   | `32～47` | `16～31` |
| `OT6`        | `M3`   | `N0`   | `48～63` | `0～15`  |
| `OT7`        | `M3`   | `N1`   | `48～63` | `16～31` |

每个 AI Core 处理一个 `[16,16]` 输出 Tile。

------

## 3. 单个 AI Core 的输入范围

设当前 AI Core 处理 `C[Mi,Nj]`。

| Tensor  | 当前 Core 使用范围 | Shape          | Dtype |
| ------- | ------------------ | -------------- | ----- |
| Feature | 完整 `X0`          | `[1,1,8,8,16]` | FP16  |
| Filter  | `W[Nj]`            | `[144,16]`     | FP16  |
| Bias    | `D[Nj]`            | `[16]`         | FP32  |
| Output  | `C[Mi,Nj]`         | `[16,16]`      | FP16  |

单次 K 迭代参与：

| Tensor     | Shape           | 元素数   | 数据量 |
| ---------- | --------------- | -------- | ------ |
| `A[Mi,Kk]` | `[16,16]`       | 256 FP16 | 512 B  |
| `B[Kk,Nj]` | `[16,16]`       | 256 FP16 | 512 B  |
| `Bias[Nj]` | `[16]`，仅 `I0` | 16 FP32  | 64 B   |
| 累加结果   | `[16,16]`       | 256 FP32 | 1024 B |

------

## 4. Local Memory 分配

| Tensor       | TPosition | 物理位置          | Shape / 元素数 | Dtype | 地址 | 实际对齐 |
| ------------ | --------- | ----------------- | -------------- | ----- | ---- | -------- |
| `X0`         | A1        | L1 Buffer         | 1024           | FP16  | 0    | 512 B    |
| `W[Nj]`      | B1        | L1 Buffer         | 2304           | FP16  | 2048 | 512 B    |
| `D[Nj]`      | C1        | L1 Buffer         | 16             | FP32  | 6656 | 128 B    |
| `D[Nj]`      | C2        | Bias Table Buffer | 16             | FP32  | 0    | 64 B     |
| `A[Mi,Kk]`   | A2        | L0A Buffer        | 256            | FP16  | 0    | 512 B    |
| `B[Kk,Nj]`   | B2        | L0B Buffer        | 256            | FP16  | 0    | 512 B    |
| `Acc[Mi,Nj]` | CO1       | L0C Buffer        | 256            | FP32  | 0    | 64 B     |

A1、B1、C1 共用 L1 Buffer，地址区间互不重叠。A2、B2、C2、CO1 位于不同物理 Buffer，可分别使用地址 0。

官方接口定义中，A2 使用 ZZ 格式，B2 使用 ZN 格式；A1/B1 通常使用 NZ。A1/B1 到 A2/B2 搬运要求源地址 32 B 对齐、目的地址 512 B 对齐。

------

## 5. Instruction 级执行过程

### 5.1 初始化与任务定位

| Instruction / 操作    | Tensor 变化      | 硬件状态            |
| --------------------- | ---------------- | ------------------- |
| `GetBlockIdx()`       | 获得 `OT0～OT7`  | Scalar 执行控制逻辑 |
| `mTileIndex = OT / 2` | 确定 `Mi`        | 不搬运数据          |
| `nTileIndex = OT % 2` | 确定 `Nj`        | 不搬运数据          |
| `mStart = Mi × 16`    | 确定输出行起点   | 不搬运数据          |
| `nStart = Nj × 16`    | 确定输出通道起点 | 不搬运数据          |

### 5.2 GM → L1

| Instruction       | 输入 Tensor | 输出 Tensor | Shape 变化              | Format 变化       | Dtype | 数据量 | 活跃硬件 |
| ----------------- | ----------- | ----------- | ----------------------- | ----------------- | ----- | ------ | -------- |
| `DataCopy`        | GM `X0`     | A1 `X0`     | `[1,1,8,8,16] → 同形状` | NC1HWC0 → NC1HWC0 | FP16  | 2048 B | MTE2     |
| `DataCopy(Nd2Nz)` | GM `W[Nj]`  | B1 `W[Nj]`  | `[144,16] → [144,16]`   | ND → NZ           | FP16  | 4608 B | MTE2     |
| `DataCopy`        | GM `D[Nj]`  | C1 `D[Nj]`  | `[16] → [16]`           | ND → 线性 Bias    | FP32  | 64 B   | MTE2     |

B1 中的 `W[Nj]` 由 9 个 `[16,16]` NZ 分形组成：

```text
W[Nj] = {
    B[K0,Nj],
    B[K1,Nj],
    ...
    B[K8,Nj]
}
```

### 5.3 MTE2 → MTE1 同步

| Instruction           | 作用                                     | 硬件状态            |
| --------------------- | ---------------------------------------- | ------------------- |
| `SetFlag<MTE2_MTE1>`  | MTE2 发布 L1 数据就绪事件                | MTE2 完成           |
| `WaitFlag<MTE2_MTE1>` | MTE1 等待 Feature、Filter、Bias 写入完成 | MTE1 阻塞至数据就绪 |

### 5.4 Bias：C1 → C2

| Instruction | 输入       | 输出       | Shape  | Format    | Dtype | 数据量 | 活跃硬件 |
| ----------- | ---------- | ---------- | ------ | --------- | ----- | ------ | -------- |
| `DataCopy`  | C1 `D[Nj]` | C2 `D[Nj]` | `[16]` | 线性 Bias | FP32  | 64 B   | MTE1     |

C2 对应 Bias Table Buffer，用于 Cube 在首次 MMAD 时读取 Bias。C1、C2、CO1 分别对应 L1/UB、Bias Table/L0C、L0C 等物理位置。

------

## 6. K 方向迭代

当前输出 Tile `C[Mi,Nj]` 共执行 9 次迭代。

### 6.1 迭代 Tensor 编号

| 迭代 | A 输入     | B 输入     | 初始累加输入  | 输出          |
| ---- | ---------- | ---------- | ------------- | ------------- |
| `I0` | `A[Mi,K0]` | `B[K0,Nj]` | `D[Nj]`       | `Acc0[Mi,Nj]` |
| `I1` | `A[Mi,K1]` | `B[K1,Nj]` | `Acc0[Mi,Nj]` | `Acc1[Mi,Nj]` |
| `I2` | `A[Mi,K2]` | `B[K2,Nj]` | `Acc1[Mi,Nj]` | `Acc2[Mi,Nj]` |
| `I3` | `A[Mi,K3]` | `B[K3,Nj]` | `Acc2[Mi,Nj]` | `Acc3[Mi,Nj]` |
| `I4` | `A[Mi,K4]` | `B[K4,Nj]` | `Acc3[Mi,Nj]` | `Acc4[Mi,Nj]` |
| `I5` | `A[Mi,K5]` | `B[K5,Nj]` | `Acc4[Mi,Nj]` | `Acc5[Mi,Nj]` |
| `I6` | `A[Mi,K6]` | `B[K6,Nj]` | `Acc5[Mi,Nj]` | `Acc6[Mi,Nj]` |
| `I7` | `A[Mi,K7]` | `B[K7,Nj]` | `Acc6[Mi,Nj]` | `Acc7[Mi,Nj]` |
| `I8` | `A[Mi,K8]` | `B[K8,Nj]` | `Acc7[Mi,Nj]` | `Acc8[Mi,Nj]` |

### 6.2 每次迭代的 Instruction

#### Instruction A：等待 A2/B2 可覆盖

仅 `I1～I8` 执行。

| Instruction        | 作用                        | 硬件状态            |
| ------------------ | --------------------------- | ------------------- |
| `SetFlag<M_MTE1>`  | Cube 发布上一轮读取完成事件 | Cube 完成读取 A2/B2 |
| `WaitFlag<M_MTE1>` | MTE1 等待后覆盖 A2/B2       | MTE1 阻塞           |

#### Instruction B：`LoadData3D`

| 属性        | 内容                               |
| ----------- | ---------------------------------- |
| 输入        | A1 完整 Feature `X0`               |
| 输入 Shape  | `[1,1,8,8,16]`                     |
| 选择范围    | 输出位置 `Mi`，K 分块 `Kk`         |
| 输出        | A2 `A[Mi,Kk]`                      |
| 输出 Shape  | `[16,16]`                          |
| Format 变化 | NC1HWC0 → ZZ                       |
| Dtype       | FP16 → FP16                        |
| 数据量      | 512 B                              |
| 地址        | A2 offset 0，512 B 对齐            |
| 硬件        | MTE1 / Load3D                      |
| 附加操作    | Padding、滑窗取数、Image-to-Column |

```text
X0[NC1HWC0]
    ↓ LoadData3D
A[Mi,Kk][16,16,ZZ]
```

#### Instruction C：`LoadData2D`

| 属性        | 内容                                   |
| ----------- | -------------------------------------- |
| 输入        | B1 中第 `Kk` 个 Filter 分形            |
| 输入 Tensor | `B[Kk,Nj]`                             |
| 输入 Shape  | `[16,16]`                              |
| 输出        | B2 `B[Kk,Nj]`                          |
| 输出 Shape  | `[16,16]`                              |
| Format 变化 | NZ → ZN                                |
| Dtype       | FP16 → FP16                            |
| 数据量      | 512 B                                  |
| 地址        | B2 offset 0，512 B 对齐                |
| 参数        | `startIndex = k`，`ifTranspose = true` |
| 硬件        | MTE1 / Load2D                          |

```text
B1: B[Kk,Nj][16,16,NZ]
        ↓ LoadData2D + transpose
B2: B[Kk,Nj][16,16,ZN]
```

#### Instruction D：MTE1 → Cube 同步

| Instruction        | 作用                      | 硬件状态            |
| ------------------ | ------------------------- | ------------------- |
| `SetFlag<MTE1_M>`  | MTE1 发布 A2、B2 就绪事件 | MTE1 完成           |
| `WaitFlag<MTE1_M>` | Cube 等待当前 K Tile      | Cube 阻塞至数据就绪 |

#### Instruction E：`Mmad`

##### `I0`

```text
Acc0[Mi,Nj]
= A[Mi,K0] × B[K0,Nj] + D[Nj]
```

| 输入       | Shape     | Format | Dtype | 位置     |
| ---------- | --------- | ------ | ----- | -------- |
| `A[Mi,K0]` | `[16,16]` | ZZ     | FP16  | L0A / A2 |
| `B[K0,Nj]` | `[16,16]` | ZN     | FP16  | L0B / B2 |
| `D[Nj]`    | `[16]`    | Bias   | FP32  | BT / C2  |

| 输出          | Shape     | Format | Dtype | 位置      |
| ------------- | --------- | ------ | ----- | --------- |
| `Acc0[Mi,Nj]` | `[16,16]` | NZ     | FP32  | L0C / CO1 |

活跃硬件：Cube。

##### `I1～I8`

```text
Acck[Mi,Nj]
= Acc(k-1)[Mi,Nj]
+ A[Mi,Kk] × B[Kk,Nj]
```

| 输入       | Shape     | Format | Dtype | 位置      |
| ---------- | --------- | ------ | ----- | --------- |
| `A[Mi,Kk]` | `[16,16]` | ZZ     | FP16  | L0A / A2  |
| `B[Kk,Nj]` | `[16,16]` | ZN     | FP16  | L0B / B2  |
| `Acc(k-1)` | `[16,16]` | NZ     | FP32  | L0C / CO1 |

| 输出   | Shape     | Format | Dtype | 位置      |
| ------ | --------- | ------ | ----- | --------- |
| `Acck` | `[16,16]` | NZ     | FP32  | L0C / CO1 |

活跃硬件：Cube。

------

## 7. 迭代与输出 Tile 的具体对应

| OT    | 9 次 A 输入          | 9 次 B 输入          | `I0` Bias |
| ----- | -------------------- | -------------------- | --------- |
| `OT0` | `A[M0,K0]～A[M0,K8]` | `B[K0,N0]～B[K8,N0]` | `D[N0]`   |
| `OT1` | `A[M0,K0]～A[M0,K8]` | `B[K0,N1]～B[K8,N1]` | `D[N1]`   |
| `OT2` | `A[M1,K0]～A[M1,K8]` | `B[K0,N0]～B[K8,N0]` | `D[N0]`   |
| `OT3` | `A[M1,K0]～A[M1,K8]` | `B[K0,N1]～B[K8,N1]` | `D[N1]`   |
| `OT4` | `A[M2,K0]～A[M2,K8]` | `B[K0,N0]～B[K8,N0]` | `D[N0]`   |
| `OT5` | `A[M2,K0]～A[M2,K8]` | `B[K0,N1]～B[K8,N1]` | `D[N1]`   |
| `OT6` | `A[M3,K0]～A[M3,K8]` | `B[K0,N0]～B[K8,N0]` | `D[N0]`   |
| `OT7` | `A[M3,K0]～A[M3,K8]` | `B[K0,N1]～B[K8,N1]` | `D[N1]`   |

------

## 8. CO1 → GM：Fixpipe 搬出

### 8.1 Cube → Fixpipe 同步

| Instruction       | 作用                            |
| ----------------- | ------------------------------- |
| `SetFlag<M_FIX>`  | Cube 发布最终累加完成事件       |
| `WaitFlag<M_FIX>` | Fixpipe 等待 `Acc8[Mi,Nj]` 就绪 |

### 8.2 `Fixpipe`

| 属性       | 输入          | 输出             |
| ---------- | ------------- | ---------------- |
| Tensor     | `Acc8[Mi,Nj]` | `Y[Mi,Nj]`       |
| Shape      | `[16,16]`     | `[16,16]`        |
| Format     | NZ            | ND               |
| Dtype      | FP32          | FP16             |
| 位置       | CO1 / L0C     | GM               |
| 激活       | —             | ReLU             |
| 转换       | —             | FP32 → FP16      |
| 每核输出量 | —             | 256 FP16 / 512 B |
| 活跃硬件   | Fixpipe       | Fixpipe          |

```text
Y[Mi,Nj] = ReLU(FP16(Acc8[Mi,Nj]))
```

Fixpipe 默认 Row Major 模式执行 NZ→ND；输入 CO1 为 NZ、FP32，源地址要求 64 B 对齐。

### 8.3 GM 输出偏移

```text
outputOffset = mStart × 32 + nStart
```

| OT    | 元素偏移 | 字节偏移 |
| ----- | -------- | -------- |
| `OT0` | 0        | 0 B      |
| `OT1` | 16       | 32 B     |
| `OT2` | 512      | 1024 B   |
| `OT3` | 528      | 1056 B   |
| `OT4` | 1024     | 2048 B   |
| `OT5` | 1040     | 2080 B   |
| `OT6` | 1536     | 3072 B   |
| `OT7` | 1552     | 3104 B   |

每行写入 16 个 FP16，即 32 B；输出行跨度为 32 个 FP16，即 64 B。

------

## 9. 完整硬件执行链

```text
GM
 │
 │ MTE2：DataCopy
 ▼
L1 Buffer
 ├─ A1：完整 Feature，NC1HWC0
 ├─ B1：当前 N Tile，NZ
 └─ C1：当前 Bias Tile
 │
 │ MTE1
 ├───────────────┐
 ▼               ▼
L0A / A2         L0B / B2
[16,16] ZZ       [16,16] ZN
 │               │
 └──────┬────────┘
        │ Cube：Mmad × 9
        ▼
L0C / CO1
[16,16] FP32 NZ
        │
        │ Fixpipe：NZ→ND、FP32→FP16、ReLU
        ▼
GM Output
[16,16] FP16 ND
```

------

## 10. 流水特征

| 项目                     | 当前实现           |
| ------------------------ | ------------------ |
| AI Core 并行             | 8 个输出 Tile 并行 |
| 单核 K 迭代              | 9 次               |
| 总 `Mmad` 次数           | `8 × 9 = 72`       |
| A2/B2 Buffer             | 单 Buffer          |
| 跨迭代 Load/Compute 重叠 | 无                 |
| Buffer 覆盖保护          | `M_MTE1`           |
| Load 完成保护            | `MTE1_M`           |
| Cube 搬出保护            | `M_FIX`            |
| Bias 融合位置            | 首次 `Mmad`        |
| ReLU 融合位置            | `Fixpipe`          |
| 输出格式                 | ND `[64,32]`       |
| Workspace                | 0 B                |

------

## 11. 单核执行摘要

```text
1. 根据 blockIdx 确定 C[Mi,Nj]
2. X0：GM → A1
3. W[Nj]：GM ND → B1 NZ
4. D[Nj]：GM → C1 → C2
5. 对 K0～K8 依次执行：
   a. A[Mi,Kk]：A1 → A2，NC1HWC0 → ZZ
   b. B[Kk,Nj]：B1 → B2，NZ → ZN
   c. Mmad：
      I0：A × B + Bias
      I1～I8：Acc + A × B
6. Acc8：CO1 FP32 NZ
7. Fixpipe：NZ → ND、FP32 → FP16、ReLU
8. 写入 GM 输出 C[Mi,Nj]
```