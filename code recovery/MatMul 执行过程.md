# MatMul 执行过程

## 第一部分：先建立整体理解

## 1. 阅读前必须区分的三类对象

### 1.1 数学对象

本案例执行：

~~~
A[M,K] × B[K,N] → C[M,N]
~~~

首个固定演示上下文：

~~~
M=1024
K=2048
N=4096
T=bfloat16
~~~

A 是 M×K，B 是 K×N，C 是 M×N。这里没有 Bias、ReLU 或 Vector 后处理。

### 1.2 Tensor API 对象

src/matmul/main.asc 使用 Tensor API 建立 GM 和 Local Memory 的 Tensor 视图：

- GM A：NDExtLayoutPtn；
- GM B：NDExtLayoutPtn；
- GM C：NDExtLayoutPtn；
- A 的 L1 视图：NZ；
- B 的 L1 视图：NZ；
- A 的 L0A 视图：NZ；
- B 的 L0B 视图：ZN；
- C 的 L0C 视图：NZ，元素类型为 float。

NZ、ZN 是 Tensor layout / storage view。不能仅凭 ZN 就把 B 解释成数学意义上的转置矩阵。

### 1.3 执行对象

| 对象 | 含义 |
| --- | --- |
| tile | 当前输出的 M/N 区域或 K 分片 |
| block | 运行时参与 tile 调度的 block |
| iter0 | L1 K 分片循环 |
| iter1 | L0 K 分片循环 |
| Event | HardEvent 的 SetFlag / WaitFlag 依赖 |
| Mmad | 当前 L0A × L0B 对 L0C 的一次计算 |
| output copy | L0C 到 GM C 的输出路径 |

## 2. 一页总览

### 2.1 输入输出

| Tensor | 逻辑 Shape | GM Layout | dtype | 逻辑元素数 | 逻辑字节数 |
| --- | --- | --- | --- | ---: | ---: |
| A | [1024,2048] | ND | BF16 | 2,097,152 | 4 MiB |
| B | [2048,4096] | ND | BF16 | 8,388,608 | 16 MiB |
| C | [1024,4096] | ND | BF16 | 4,194,304 | 8 MiB |

字节数按 BF16 每元素 2 bytes 计算。Tensor API 对局部 layout 的 padding、物理 bank 和真实分配容量，需要目标环境进一步验证。

### 2.2 完整数据流

~~~
GM A [M,K] ── CopyGM2L1 ──> L1 A1 [baseM,kL1] ── CopyL12L0A ──> L0A A2
GM B [K,N] ── CopyGM2L1 ──> L1 B1 [kL1,baseN] ── CopyL12L0B ──> L0B B2
                                                               │
                                                               ▼
                                                        Cube Mmad
                                                               │
                                                               ▼
                                                        L0C CO1 float
                                                               │
                                                        CopyL0C2GM
                                                               │
                                                               ▼
GM C [M,N] <────────────────────────────────────────────── output
~~~

这条路径表达源码定义的逻辑搬运和计算顺序，不表达真实时间重叠。

### 2.3 Kernel 和 Host 的位置

| 代码区间 | 角色 | 主要职责 |
| --- | --- | --- |
| 73–206 | Kernel | Tiling 派生、block 调度、Tensor Slice、Memory 搬运、Mmad、同步 |
| 223–357 | Host | 参数解析、ACL、Host/Device Memory、Kernel launch、结果回拷 |
| 359–471 | Host helper | Tiling 辅助计算、BF16 转换、文件读写 |

main.asc 是单文件，页面不能把它误读成两个独立源码文件。

## 第二部分：M、N、K 如何分块

## 1. 基础 Tiling 参数

Kernel 代码定义：

| 参数 | BF16 下的值 | 源码含义 |
| --- | ---: | --- |
| baseM | 256 | M 方向基础 tile |
| baseN | 256 | N 方向基础 tile |
| baseK | 128 | 256 / sizeof(T) |
| kL1 | 512 | 1024 / sizeof(T) |
| cube C0 | 16 | L0C layout 的内部参数 |

固定上下文的 tile 数：

~~~
baseM = 256
baseN = 256
baseK = 128
kL1 = 512

mTileNum = 1024 / 256 = 4
nTileNum = 4096 / 256 = 16
tileNum = 4 × 16 = 64

kL1TileNum = 2048 / 512 = 4
kL0IterNum = 512 / 128 = 4
~~~

因此：

- 输出空间有 64 个完整输出 tile；
- 每个输出 tile 需要 4 个 L1 K tile；
- 每个 L1 K tile 需要 4 个 L0 K tile；
- 每个输出 tile 需要 16 次 Mmad；
- 固定 shape 下整个逻辑输出空间对应 1024 次 Mmad。

最后一个数字是固定 shape 下的逻辑 Mmad 数，不是某个设备上的实际并行执行次数。

## 2. Output tile 与 block 映射

代码使用：

~~~
for (tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum)
~~~

当前 tile 的二维索引：

~~~
mTileIdx = tileIdx / nTileNum
nTileIdx = tileIdx % nTileNum

mStart = mTileIdx × baseM
nStart = nTileIdx × baseN
~~~

| tileIdx | mTileIdx | nTileIdx | C 行范围 | C 列范围 |
| ---: | ---: | ---: | --- | --- |
| 0 | 0 | 0 | [0,256) | [0,256) |
| 1 | 0 | 1 | [0,256) | [256,512) |
| 15 | 0 | 15 | [0,256) | [3840,4096) |
| 16 | 1 | 0 | [256,512) | [0,256) |
| 63 | 3 | 15 | [768,1024) | [3840,4096) |

blockIdx 只决定 tileIdx 的起始位置和步进，不应被页面固定解释成 block 0 永远只处理左上角 tile。

## 第三部分：Host 侧执行过程

## 1. 参数解析

Host 通过命令行接收：

~~~
matmul m k n
~~~

对应源码 223–262：

- 打印使用说明；
- 解析 m、k、n；
- 检查三个维度为正数。

这一步是 Host 配置，不是 GM 数据搬运。

## 2. ACL 与 Host 数据

Host 依次：

1. aclInit；
2. aclrtSetDevice；
3. aclrtCreateStream；
4. 创建 input、weight、output 和 goldenOutput 的 Host vector；
5. 运行 gen_data.py；
6. 读取 input_a.bin、input_b.bin；
7. 计算输入输出字节数。

当前脚本使用 torch.bfloat16 生成数据，输入和输出以 uint16 原始位模式写入 bin 文件。验证脚本再按 BF16 解释这些位模式。

## 3. Device Memory 与 Host→Device

Host 为 A、B、C 分配 Device Memory：

- deviceInput；
- deviceWeight；
- deviceOutput。

随后执行：

~~~
Host A → Device A
Host B → Device B
~~~

源码 325–328 是 Host→Device 的运行时拷贝。它与 Kernel 内的 GM→L1 CopyGM2L1 属于不同层级的数据路径，页面不能合并为一个 Copy Input 步骤。

## 4. Kernel launch

Host 通过：

~~~
numBlocks = PlatformAscendCManager::GetInstance()->GetCoreNumAic()
~~~

获取启动 block 数，然后调用：

~~~
MatmulKernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(A, B, C, m, k, n)
~~~

确认事实：

- 当前 launch 使用 bfloat16_t；
- numBlocks 来源于平台 AIC core 查询；
- 运行时 M、K、N 传入 Kernel。

不能把 numBlocks 固化为某一个常数，也不能把它直接等同于固定设备物理核数。

## 第四部分：Kernel 初始化与输出 tile

## 1. Kernel 入口

Kernel 在 73–75 行：

- 接收 A、B、C 的 GM 地址；
- 接收 m、k、n；
- 标记 KERNEL_TYPE_AIC_ONLY。

之后在 78–89 行派生 Tiling 变量。

## 2. GM Tensor 视图

代码 94–101 为三个 GM Tensor 创建 NDExtLayoutPtn：

~~~
tensorAgm : GM A [m,k]
tensorBgm : GM B [k,n]
tensorCgm : GM C [m,n]
~~~

输出 tile 的 Slice：

~~~
A block = Slice(mTileIdx × baseM, 0), [baseM, k]
B block = Slice(0, nTileIdx × baseN), [k, baseN]
C block = Slice(mTileIdx × baseM, nTileIdx × baseN), [baseM, baseN]
~~~

L0C 视图：

~~~
L0C layout = NZ(baseM, baseN, C0=16)
L0C type = float
l0cOffset = 0
~~~

## 3. 输出 tile 前的 FIX_M 等待

每个输出 tile 开始前执行：

~~~
WaitFlag<HardEvent::FIX_M>
~~~

这是依赖控制，不是性能时间，也不代表设备实际处于空闲状态。

## 第五部分：GM → L1

## 1. 外层 L1 K 循环

代码 126–192：

~~~
for (iter0 = 0; iter0 < kL1TileNum; ++iter0)
~~~

每一轮：

1. 等待 MTE1_MTE2；
2. 使用固定的 L1 K 长度 kL1；
3. 建立 A1、B1 的 L1 layout；
4. 从 A/B GM Slice 取当前 K 区间；
5. 执行 CopyGM2L1；
6. SetFlag / WaitFlag MTE2_MTE1；
7. 进入内层 L0 K 循环；
8. 本轮结束后 SetFlag MTE1_MTE2。

## 2. A 的搬运

A 当前 L1 tile：

~~~
A1 = A[mStart : mStart+baseM,
       iter0×kL1 : iter0×kL1+kL1]
~~~

A 的 L1 buffer offset：

~~~
l1BufferAOffset = 0
~~~

当前 transA=false 分支的 layoutAL1 使用 NZ。这里的 layout 结论来自源码模板定义，目标 API 的实际物理行为仍属于待编译验证内容。

## 3. B 的搬运

B 当前 L1 tile：

~~~
B1 = B[iter0×kL1 : iter0×kL1+kL1,
       nStart : nStart+baseN]
~~~

B 的 L1 buffer offset：

~~~
l1BufferBOffset = baseM × kL1 × sizeof(T)
~~~

BF16 固定上下文：

~~~
l1BufferBOffset = 256 × 512 × 2 = 262144 bytes
~~~

当前 transB=false 分支的 layoutBL1 使用 NZ。B 在 L1 中仍然是当前 K/N 的二维逻辑 tile，不能在此处直接标记为 ZN。

## 4. MTE2_MTE1

GM→L1 Copy 完成后：

~~~
SetFlag<HardEvent::MTE2_MTE1>
WaitFlag<HardEvent::MTE2_MTE1>
~~~

本阶段不改变 Tensor shape 或 dtype。

## 第六部分：L1 → L0

## 1. 内层 L0 K 循环

当前完整 L1 K tile 进入内层循环。固定 BF16 shape 下：

~~~
baseK = 128
kL0IterNum = 4
~~~

每个 iter1：

1. WaitFlag M_MTE1；
2. 使用固定的 L0 K 长度 baseK；
3. Copy L1 → L0A；
4. Copy L1 → L0B；
5. SetFlag / WaitFlag MTE1_M；
6. Mmad；
7. SetFlag M_MTE1。

## 2. A1 → A2

A2 当前 tile：

~~~
A2 = A1[0 : baseM,
        iter1×baseK : iter1×baseK+baseK]
~~~

A2 位置：

- L0A；
- NZ layout；
- 元素类型 T；
- 起始地址为 0 的局部视图。

CopyL12L0A 是源码确认的 Copy API。A2 的真实物理 padding 和容量需要目标 CANN 环境验证。

## 3. B1 → B2

B2 当前 tile：

~~~
B2 = B1[iter1×baseK : iter1×baseK+baseK,
        0 : baseN]
~~~

B2 位置：

- L0B；
- ZN layout；
- 元素类型 T；
- 起始地址为 0 的局部视图。

页面应描述为 layout preparation / layout conversion，不直接写成数学转置。

## 4. MTE1_M

L1→L0 完成后：

~~~
SetFlag<HardEvent::MTE1_M>
WaitFlag<HardEvent::MTE1_M>
~~~

该依赖使 Cube Mmad 读取 A2、B2 时建立在 MTE1 完成的前提上。

## 第七部分：Mmad 初始化与累加

## 1. Mmad 参数

源码 180–187 设置：

~~~
para.m = baseM
para.n = baseN
para.k = baseK
para.cmatrixInitVal = (iter1 == 0 && iter0 == 0)
~~~

并执行：

~~~
Mmad(MadOp.with(para), tensorL0C, tensorAL0, tensorBL0)
~~~

数学语义：

~~~
C_partial = A2 × B2
~~~

## 2. 第一次 Mmad

当 iter0=0 且 iter1=0：

~~~
cmatrixInitVal = true
~~~

这一轮负责初始化 L0C。源码没有 Bias 输入，因此不是 Conv demo 中的 Mmad + Bias。

## 3. 后续 Mmad

当 iter0 或 iter1 不满足首次条件：

~~~
cmatrixInitVal = false
~~~

后续 Mmad 将当前 A2×B2 结果累加到 L0C。固定完整 shape 下：

~~~
C[Mi,Nj] =
  Σ iter0=0..3
    Σ iter1=0..3
      A[Mi, K(iter0,iter1)] × B[K(iter0,iter1), Nj]
~~~

## 4. M_MTE1

每次 Mmad 后执行：

~~~
SetFlag<HardEvent::M_MTE1>
~~~

下一次内层循环开始时执行：

~~~
WaitFlag<HardEvent::M_MTE1>
~~~

该事件用于控制 Cube 对 L0A/L0B 的消费与下一轮 MTE1 写入之间的依赖。页面只展示位置和方向，不推断等待时长。

## 第八部分：L0C → GM

## 1. M_FIX

当前输出 tile 的所有 K 归约完成后：

~~~
SetFlag<HardEvent::M_FIX>
WaitFlag<HardEvent::M_FIX>
~~~

这让输出 Copy 建立在最终 L0C 结果完成之后。

## 2. CopyL0C2GM

源码 196–200：

~~~
copyL0C2GM = MakeCopy(CopyL0C2GM{})
Copy(copyL0C2GM, tensorCGmBlock, tensorL0C)
SetFlag<HardEvent::FIX_M>
~~~

确认事实：

- 来源是 L0C Tensor；
- 目标是当前 C GM Slice；
- 输出 API 是 CopyL0C2GM；
- 当前代码没有显式写出名为 Fixpipe 的 API 调用。

README 的 profile 表格包含 fixpipe 指标，但这不能单独证明源码 API 名称或具体转换参数。页面可以把它放在 Output / Fixpipe 类别中，但需标记为 API 分类或 profiling 分类，不应覆盖源码事实。

## 3. Host 结果回拷

Kernel 结束后等待：

~~~
WaitFlag<MTE1_MTE2>
WaitFlag<M_MTE1>
WaitFlag<FIX_M>
~~~

Host 再：

1. aclrtSynchronizeStream；
2. Device C → Host output；
3. 写出 npu_out.bin；
4. 调用 verify_result.py；
5. 销毁 stream、reset device、aclFinalize。

## 第九部分：Event 总览

| Event | 源码位置 | 逻辑作用 |
| --- | --- | --- |
| MTE1_MTE2 | 104、191、203 | 初始化/循环间的 MTE1 与 MTE2 资源依赖 |
| M_MTE1 | 105、157、189、204 | Cube 消费 L0 后允许下一次 MTE1 写入 |
| FIX_M | 106、124、200、205 | 输出资源复用前的输出路径依赖 |
| MTE2_MTE1 | 150–151 | GM→L1 完成后允许 L1→L0 |
| MTE1_M | 176–177 | L1→L0 完成后允许 Cube Mmad |
| M_FIX | 193–194 | Mmad 完成后允许 L0C→GM |

同名 HardEvent 的 Set / Wait 组合必须结合循环位置理解，不能只根据字符串名称画成一条独立数据流。

## 第十部分：Instruction 级 Stage 附录

| Stage | 主要源码 | Action | 执行角色 |
| --- | --- | --- | --- |
| 1 | Host 242–262 | Configure | Host / Scalar |
| 2 | Kernel 78–88 | Configure | Scalar |
| 3 | Kernel 94–101 | Configure | Scalar |
| 4 | Kernel 108–124 | Control / Slice | Scalar / AIC |
| 5 | Kernel 126–151 | Move / Sync | MTE2 / Event |
| 6 | Kernel 153–177 | Move / Transform / Sync | MTE1 / Event |
| 7 | Kernel 179–189 | Compute / Sync | Cube / Event |
| 8 | Kernel 193–200 | Sync / Store | Event / Output path |
| 9 | Kernel 202–205 | Finalize | Event |
| 10 | Host 336–356 | Observe / Cleanup | Host / Runtime |

## 最短记忆版

~~~
A[M,K] × B[K,N] → C[M,N]

M/N：输出空间分成 256×256 tile
K：
  GM→L1：每次 512
  L1→L0：每次 128

GM A/B
  → MTE2 → L1 A1/B1
  → MTE1 → L0A A2/L0B B2
  → Cube Mmad → L0C CO1 float
  → CopyL0C2GM → GM C

第一次 Mmad 初始化 CO1
后续 Mmad 沿 K 维累加
没有 Bias，没有 ReLU
逻辑顺序不等于真实时间
~~~
