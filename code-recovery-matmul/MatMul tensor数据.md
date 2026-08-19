# MatMul tensor 数据

## 文档定位

本文按执行 Stage 整理 MatMul 案例中每个 Tensor 的 Shape、Tile 身份、存储位置、dtype、layout 和状态变化。

权威源码：

~~~
页面源码快照 `src/matmul/main.asc`（对应仓库实际源文件 `Samples/0_Introduction/matmul/main.asc`）
~~~

辅助证据：

- src/matmul/README.md
- src/matmul/scripts/gen_data.py
- src/matmul/scripts/verify_result.py
- src/matmul/CMakeLists.txt

固定演示上下文：

~~~
M=1024
K=2048
N=4096
T=bfloat16
baseM=256
baseN=256
baseK=128
kL1=512
~~~

本文明确区分：

- confirmed：源码或脚本直接表达；
- derived：由固定参数确定性推导；
- inferred：依据 API 或 layout 语义的解释；
- unverified：需要目标 CANN 编译、设备运行或 dump 验证。

没有实际 dump 时，本文只提供 Tensor 元数据和逻辑 tile，不生成矩阵数值。

## 全局 Shape 与分块参数

| 参数 | 当前值 | 证据等级 | 对 Shape 的影响 |
| --- | ---: | --- | --- |
| M | 1024 | derived | A 的行数、C 的行数 |
| K | 2048 | derived | A 的列数、B 的行数 |
| N | 4096 | derived | B 的列数、C 的列数 |
| dtype T | BF16 | confirmed | Host 以 bfloat16_t launch，脚本以 torch.bfloat16 生成 |
| baseM | 256 | confirmed | M 输出 tile 的基础行数 |
| baseN | 256 | confirmed | N 输出 tile 的基础列数 |
| baseK | 128 | derived | 256 / sizeof(BF16) |
| kL1 | 512 | derived | 1024 / sizeof(BF16) |
| mTileNum | 4 | derived | M / baseM |
| nTileNum | 16 | derived | N / baseN |
| output tile 数 | 64 | derived | mTileNum × nTileNum |
| K L1 tile 数 | 4 | derived | K / kL1 |
| K L0 tile 数 | 4 | derived | kL1 / baseK |
| 每输出 tile Mmad | 16 | derived | K L1 tile 数 × K L0 tile 数 |
| 逻辑 AIC / block 数 | 8 | assumed | 本页的 8 核 round-robin 演示上下文 |
| 每 AIC 输出 tile 数 | 8 | derived | 64 个输出 tile / 8 个 AIC |
| 每 AIC 逻辑 Mmad 数 | 128 | derived | 8 个输出 tile × 16 次 Mmad |

## 全局 Tensor 契约

| Tensor | 逻辑 Shape | GM / Local 位置 | layout | dtype | 逻辑元素数 | 逻辑字节数 |
| --- | --- | --- | --- | --- | ---: | ---: |
| A | [1024,2048] | GM | ND | BF16 | 2,097,152 | 4 MiB |
| B | [2048,4096] | GM | ND | BF16 | 8,388,608 | 16 MiB |
| C | [1024,4096] | GM | ND | BF16 | 4,194,304 | 8 MiB |
| A1 | [baseM,kL1] | L1 | NZ | T | 固定 | 当前完整 L1 K tile |
| B1 | [kL1,baseN] | L1 | NZ | T | 固定 | 当前完整 L1 K tile |
| A2 | [baseM,baseK] | L0A | NZ | T | 固定 | 当前完整 L0 K tile |
| B2 | [baseK,baseN] | L0B | ZN | T | 固定 | 当前完整 L0 K tile |
| CO1 | [baseM,baseN] | L0C | NZ, C0=16 | FP32 | 固定 | 当前完整输出 tile |

A1、B1、A2、B2 的 Tensor 变量由 Tensor API 在 Kernel 内建立。CO1 的元素类型在源码中明确为 float；GM C 的目标类型来自 Kernel 模板 T，当前 launch 为 BF16。

## 逻辑执行结构

| Stage | Tensor 状态变化 |
| ---: | --- |
| 1 | Host 解析 M/K/N，建立输入输出尺寸 |
| 2 | Kernel 派生 M/N tile、L1 K tile、L0 K tile |
| 3 | 建立 GM A/B/C 的 ND Tensor view |
| 4 | Slice 当前输出 tile 和当前 K tile |
| 5 | GM A/B → L1 A1/B1 |
| 6 | MTE2_MTE1 后，A1/B1 对 MTE1 可读 |
| 7 | L1 A1/B1 → L0A A2/L0B B2 |
| 8 | MTE1_M 后，A2/B2 对 Cube 可读 |
| 9 | Mmad 写入或累加 CO1 |
| 10 | M_FIX 后，CO1 → GM C |

## 阶段 1：Host Shape 与输入输出

Host 侧动态数组：

| Host 变量 | Shape | dtype / storage | 用途 |
| --- | --- | --- | --- |
| hostInput | [M,K] | uint16 位存储 | A 的 BF16 原始位 |
| hostWeight | [K,N] | uint16 位存储 | B 的 BF16 原始位 |
| hostOutput | [M,N] | uint16 位存储 | C 的 BF16 原始位 |
| goldenOutput | [M,N] | uint16 位存储 | Host 侧预留的 golden buffer |

gen_data.py 使用 torch.bfloat16 生成 A、B 和 CPU golden C，并通过 view(torch.uint16) 写入二进制文件。页面可以展示 BF16 dtype，但不能把 uint16 文件存储误解为 uint16 数学计算。

## 阶段 2：M/N 输出 tile

输出 tile 编号：

~~~
tileIdx = 0 ... 63
mTileIdx = tileIdx / 16
nTileIdx = tileIdx % 16
~~~

当前完整 tile：

| 对象 | Shape | 坐标 |
| --- | --- | --- |
| A GM block | [256,2048] | A[mTileIdx×256 : (mTileIdx+1)×256, 0:2048] |
| B GM block | [2048,256] | B[0:2048, nTileIdx×256 : (nTileIdx+1)×256] |
| C GM block | [256,256] | C[mTileIdx×256 : (mTileIdx+1)×256, nTileIdx×256 : (nTileIdx+1)×256] |

例：tileIdx=17：

~~~
mTileIdx = 17 / 16 = 1
nTileIdx = 17 % 16 = 1

A block = A[256:512, 0:2048]
B block = B[0:2048, 256:512]
C block = C[256:512, 256:512]
~~~

block 调度可能以 blockNum 步进处理多个 tile，因此 tileIdx 不等于 blockIdx 的固定一对一映射。

### 8 核 ownership

本页固定使用 `blockNum = 8` 作为逻辑演示假设。对任意 AIC `b`，其输出 tile 序列为：

~~~
{ b, b + 8, b + 16, ... }  < 64
~~~

| AIC | tileIdx | 第一个 tile 的 C tile 坐标 | 最后一个 tile 的 C tile 坐标 |
| --- | --- | --- | --- |
| AIC0 | 0, 8, 16, 24, 32, 40, 48, 56 | M0/N0 | M3/N8 |
| AIC1 | 1, 9, 17, 25, 33, 41, 49, 57 | M0/N1 | M3/N9 |
| AIC2 | 2, 10, 18, 26, 34, 42, 50, 58 | M0/N2 | M3/N10 |
| AIC3 | 3, 11, 19, 27, 35, 43, 51, 59 | M0/N3 | M3/N11 |
| AIC4 | 4, 12, 20, 28, 36, 44, 52, 60 | M0/N4 | M3/N12 |
| AIC5 | 5, 13, 21, 29, 37, 45, 53, 61 | M0/N5 | M3/N13 |
| AIC6 | 6, 14, 22, 30, 38, 46, 54, 62 | M0/N6 | M3/N14 |
| AIC7 | 7, 15, 23, 31, 39, 47, 55, 63 | M0/N7 | M3/N15 |

Tensor 元数据始终描述当前选中的 `AIC / tileIdx / iter0 / iter1` 上下文；它不意味着页面拥有 8 个 AIC 的真实 dump。没有 dump 时，A、B、A1、B1、A2、B2、CO1 仍只显示 shape、layout、dtype、生命周期和逻辑坐标，不显示具体数值。

## 阶段 3：GM Tensor View

Kernel 使用 NDExtLayoutPtn：

| Tensor | Tensor API view | 位置 | layout |
| --- | --- | --- | --- |
| tensorAgm | MakeTensor(GM A, [M,K]) | GM | ND |
| tensorBgm | MakeTensor(GM B, [K,N]) | GM | ND |
| tensorCgm | MakeTensor(GM C, [M,N]) | GM | ND |

当前输出 tile 的 Slice：

| Tensor | Slice 起点 | Slice Shape |
| --- | --- | --- |
| tensorAGmBlock | [mTileIdx×baseM, 0] | [baseM, K] |
| tensorBGmBlock | [0, nTileIdx×baseN] | [K, baseN] |
| tensorCGmBlock | [mTileIdx×baseM, nTileIdx×baseN] | [baseM, baseN] |

## 阶段 4：L1 Buffer View

对完整 BF16 输出 tile：

| Buffer | Shape | layout | 起始 offset | 逻辑字节数 |
| --- | --- | --- | ---: | ---: |
| A1 | [256,512] | NZ | 0 | 256 KiB |
| B1 | [512,256] | NZ | 262144 | 256 KiB |

B1 offset 来自：

~~~
baseM × kL1 × sizeof(T)
= 256 × 512 × 2
= 262144 bytes
~~~

A1 和 B1 共享 L1 地址空间的事实来自同一个 L1 location 和明确的 offset 表达；实际 L1 分配边界、padding 和 bank 使用需要目标环境验证。

## 阶段 5：GM → L1

当前 iter0 的逻辑 Tensor：

| Tensor | 来源 | 目标 | 当前 Shape | layout 变化 |
| --- | --- | --- | --- | --- |
| A1 | A GM Slice | L1 A1 | [baseM,kL1] | GM ND → L1 layoutAL1 |
| B1 | B GM Slice | L1 B1 | [kL1,baseN] | GM ND → L1 layoutBL1 |

当前 transA=false、transB=false 分支下，源码的 layout alias 都选择 NZ。该结论是源码模板分支事实；具体搬运后的物理布局需要编译或运行验证。

阶段结束后等待 MTE2_MTE1。此时状态可以写为：

| Tensor | 状态 | 位置 |
| --- | --- | --- |
| A1 | MTE2 写入完成，等待 MTE1 读取 | L1 |
| B1 | MTE2 写入完成，等待 MTE1 读取 | L1 |

该状态表示逻辑依赖，不代表真实时间点。

## 阶段 6：L1 → L0 K tile

固定 BF16 场景下：

~~~
kL1 = 512
baseK = 128
kL0IterNum = 4
~~~

每个 iter1：

| Tensor | 来源 | 目标 | Shape | layout |
| --- | --- | --- | --- | --- |
| A2 | A1 Slice | L0A | [baseM,baseK] | NZ |
| B2 | B1 Slice | L0B | [baseK,baseN] | ZN |

当前完整 tile 的四个 L0 K 范围：

| iter1 | K 范围 |
| ---: | --- |
| 0 | [0,128) |
| 1 | [128,256) |
| 2 | [256,384) |
| 3 | [384,512) |

A2 的 Slice：

~~~
A1[0:baseM, iter1×baseK : iter1×baseK+baseK]
~~~

B2 的 Slice：

~~~
B1[iter1×baseK : iter1×baseK+baseK, 0:baseN]
~~~

阶段结束后等待 MTE1_M，状态为 A2/B2 对 Cube 可读。

## 阶段 7：L0C CO1 与 Mmad

L0C Tensor：

| Tensor | Shape | 位置 | layout | dtype |
| --- | --- | --- | --- | --- |
| tensorL0C | [baseM,baseN] | L0C | NZ, C0=16 | float |

完整输出 tile 的逻辑累计：

~~~
CO1 = 0
for iter0 in 0..3:
  for iter1 in 0..3:
    CO1 += A2[iter0,iter1] × B2[iter0,iter1]
~~~

第一次 Mmad：

~~~
iter0=0 且 iter1=0
cmatrixInitVal=true
~~~

后续 Mmad：

~~~
cmatrixInitVal=false
沿 K 维累加到同一个 CO1 view
~~~

当前案例没有 Bias Tensor。当前案例也没有 ReLU Tensor 或 Vector output stage。

## 阶段 8：L0C → GM C

输出 Copy：

| 来源 | 目标 | 来源 dtype | 目标 dtype | API |
| --- | --- | --- | --- | --- |
| tensorL0C | tensorCGmBlock | float | T，当前为 BF16 | CopyL0C2GM |

源码明确的是 CopyL0C2GM API 和源/目标 Tensor。float→BF16 的具体转换行为、NZ→ND 的物理转换细节，以及局部 layout 的物理 padding，需要目标 API / 编译 / 设备验证。

因此页面的证据表达建议：

- confirmed：L0C → GM C、CopyL0C2GM、CO1 float；
- derived：当前 C Slice 的行列范围；
- inferred：输出路径中的 layout / dtype conversion；
- unverified：目标环境中的具体转换实现和物理 padding。

## 阶段 9：Host Output 与 Golden

Host：

1. aclrtSynchronizeStream；
2. Device C → hostOutput；
3. 写出 output/npu_out.bin；
4. verify_result.py 读取 npu_out.bin 和 cpu_output.bin；
5. 按 BF16 reshape 为 [M,N]；
6. 计算 absolute error、relative error 和 error ratio。

验证脚本中：

| 指标 | 当前规则 |
| --- | --- |
| point error | relative diff > 0.1 或 non-finite |
| ratio error | absolute diff > 0.001 |
| error ratio | 不超过 0.001 |
| 大矩阵输出 | 只打印 summary 和左上角 4×4 |

这些是 correctness 验证规则，不是 Code Recovery 页面里的运行 profiling 数据。

## Memory 生命周期

### GM

| Tensor | 状态 |
| --- | --- |
| A | Host 输入回拷后存在于 Device GM，按 M/K Slice 被读取 |
| B | Host 权重回拷后存在于 Device GM，按 K/N Slice 被读取 |
| C | Device 输出 buffer，按当前 C Slice 被写回 |

### L1

| Buffer | 生命周期 |
| --- | --- |
| A1 | 当前 iter0 的 A GM tile 搬入后，供内层 iter1 读取 |
| B1 | 当前 iter0 的 B GM tile 搬入后，供内层 iter1 读取 |
| A1/B1 | 通过 MTE2_MTE1 和 M_MTE1 控制读写顺序 |

### L0

| Buffer | 生命周期 |
| --- | --- |
| A2 | 当前 iter1 的 A K tile，MTE1 写入，Cube 消费 |
| B2 | 当前 iter1 的 B K tile，MTE1 写入，Cube 消费 |
| CO1 | 第一次 Mmad 初始化，后续 Mmad 沿 K 累加，输出 Copy 后复用 |

当前文档不把这些生命周期绘制成真实时间条；它们只是源码恢复出的逻辑顺序和状态。

## Tensor Data Dump 边界

当前工作区没有目标样例的真实运行 dump，因此：

- 不填写 A、B、C、A1、B1、A2、B2、CO1 的具体矩阵数值；
- 不使用 README 中的 golden 示例值冒充某个具体 Instruction 的中间数据；
- 不把随机数据生成脚本生成的输入值直接写入页面；
- 页面应显示 shape、tile、dtype、layout 和 No dump data；
- 后续若接入 dump，必须绑定 Run、Instruction、tile、iteration、dtype、layout、location。

## 最关键的命名规则

建议统一使用：

| 变量名 | 位置 | 含义 |
| --- | --- | --- |
| tensorAgm | GM | 全局 A Tensor view |
| tensorBgm | GM | 全局 B Tensor view |
| tensorCgm | GM | 全局 C Tensor view |
| tensorAL1 | L1 | 当前 A 的 L1 Tensor view |
| tensorBL1 | L1 | 当前 B 的 L1 Tensor view |
| tensorAL0 | L0A | 当前 A 的 L0 Tensor view |
| tensorBL0 | L0B | 当前 B 的 L0 Tensor view |
| tensorL0C | L0C | 当前输出累加 Tensor view |

不要把 A1、B1、A2、B2 既当作 Tensor 变量名，又当作 Memory location 名。页面需要同时显示变量和位置，例如：

~~~
tensorAL1 · L1 · NZ · [baseM,kL1]
tensorBL0 · L0B · ZN · [baseK,baseN]
~~~
