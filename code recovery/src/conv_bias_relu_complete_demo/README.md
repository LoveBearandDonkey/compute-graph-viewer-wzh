# Conv2D + Bias + ReLU 固定执行参考 / Fixed Execution Reference

本包将原始参考代码中的占位项补充为一组参数完整、执行路径确定的示例配置，适用于 Code-to-Hardware 可视化原型。

This package replaces the placeholders in the original reference with one fully specified demo fixture suitable for a Code-to-Hardware visualization prototype.

---

## 1. 算子配置 / Concrete Operator

| 项目 / Item                                      | 数值 / Value        |
| ------------------------------------------------ | ------------------- |
| 输入 X（逻辑）/ X logical                        | `[1,16,8,8]`, FP16  |
| 权重 W（逻辑）/ W logical                        | `[32,16,3,3]`, FP16 |
| Bias                                             | `[32]`, FP32        |
| 输出 Y（逻辑）/ Y logical                        | `[1,32,8,8]`, FP16  |
| 步长 / 填充 / 膨胀 / stride / padding / dilation | `1 / 1 / 1`         |
| Cube M/K/N                                       | `64 / 144 / 32`     |
| tileM/tileK/tileN                                | `16 / 16 / 16`      |
| M/K/N 分块数 / tile counts                       | `4 / 9 / 2`         |
| 输出块数 / blockDim / output tiles / blockDim    | `8 / 8`             |

---

## 2. 分块编号 / Tile Numbering

- `Mi`：`M0..M3`，每块包含 16 个输出位置。  
  `Mi`: `M0..M3`, each containing 16 output positions.

- `Kk`：`K0..K8`，每块包含 16 个归约元素。  
  `Kk`: `K0..K8`, each containing 16 reduction elements.

- `Nj`：`N0..N1`，每块包含 16 个输出通道。  
  `Nj`: `N0..N1`, each containing 16 output channels.

- 输出块编号：`OT = Mi * 2 + Nj`；每个 AI Core 的 `blockIdx` 处理一个 `OT`。  
  Output-tile numbering: `OT = Mi * 2 + Nj`; each AI Core `blockIdx` processes one `OT`.

- 第 `Ik` 次 K 迭代读取 `A[Mi,Kk]` 和 `B[Kk,Nj]`。  
  K iteration `Ik` consumes `A[Mi,Kk]` and `B[Kk,Nj]`.

- `I0` 额外读取 `Bias[Nj]`；`I1..I8` 从 CO1 中读取已有累加结果并继续累加。  
  `I0` additionally consumes `Bias[Nj]`; `I1..I8` read the existing accumulator from CO1 and continue accumulation.

---

## 3. 输入输出物理格式约定 / Physical Input and Output Contract

该 Kernel 明确定义以下物理格式：

The kernel explicitly uses the following physical formats:

- Feature GM：`NC1HWC0 [1,1,8,8,16]`。  
  Feature GM: `NC1HWC0 [1,1,8,8,16]`.

- Filter GM：行优先 ND，`[K,Co] = [144,32]`。逻辑 OIHW 权重必须在执行 Kernel 前完成展平和转置。  
  Filter GM: row-major ND, `[K,Co] = [144,32]`. The logical OIHW filter must be flattened and transposed before kernel launch.

- Bias GM：ND `[32]`。  
  Bias GM: ND `[32]`.

- Output GM：行优先 ND，`[M,Co] = [64,32]`，等价于 NHWC `[1,8,8,32]`。  
  Output GM: row-major ND, `[M,Co] = [64,32]`, equivalent to NHWC `[1,8,8,32]`.

当前输出不是 NCHW。若外部接口要求 NCHW，需要额外增加布局转换阶段。

The output is not NCHW. Add a separate layout-conversion stage when NCHW is required by the external contract.

---

## 4. 每个 AI Core 的片上存储占用 / Exact Local-Memory Occupancy per AI Core

| 位置 / Position | Tensor                               | 元素数 / Elements | 字节数 / Bytes | 起始地址 / Address |
| --------------- | ------------------------------------ | ----------------: | -------------: | -----------------: |
| A1/L1           | 完整 Feature / full feature          |         1024 half |           2048 |                  0 |
| B1/L1           | 一个权重 N Tile / one weight N-tile  |         2304 half |           4608 |               2048 |
| C1/L1           | 一个 Bias Tile / one bias tile       |          16 float |             64 |               6656 |
| C2/BT           | 一个 Bias Tile / one bias tile       |          16 float |             64 |                  0 |
| A2/L0A          | 一个 A Tile / one A tile             |          256 half |            512 |                  0 |
| B2/L0B          | 一个 B Tile / one B tile             |          256 half |            512 |                  0 |
| CO1/L0C         | 一个累加 Tile / one accumulator tile |         256 float |           1024 |                  0 |

---

## 5. 文件说明 / Files

- `op_kernel/conv_bias_relu_reference_complete.asc`  
  完整的固定参数 Kernel。  
  Complete fixed-parameter kernel.

- `op_kernel/conv_bias_relu_tiling_data.h`  
  Kernel 与 Host 共用的标准 C++ Tiling 数据结构。  
  Shared standard C++ tiling structure used by both kernel and host code.

- `op_host/conv_bias_relu_tiling.cpp`  
  固定参数的 Host 侧 Tiling 回调。  
  Fixed host-side tiling callback.

---

## 6. 集成边界 / Integration Boundary

该示例是一份确定性的参考 Kernel，不是完整生成的自定义算子工程。以下内容不包含在本包中：

This is a deterministic reference kernel, not a complete generated custom-operator project. The following parts are outside this package:

- 算子定义 / operator definition
- 构建脚本 / build scripts
- Kernel 启动代码 / kernel launch code
- 输入数据打包 / input packing
- CPU Golden 实现 / CPU golden implementation

源码已完成内部一致性检查，并移除了原始版本中的零大小分配与未定义占位项。

The source has been checked for internal consistency, and the original zero-sized or undefined placeholders have been removed.

当前环境未安装 BiSheng、完整 CANN 9.1 头文件及 Atlas A2/A3/910B 设备，因此本代码尚未完成真实编译和板端运行验证。头文件字段、API 参数及算子类型注册名称需要与实际安装的 CANN 版本进一步核对。

The code has not been compiled with BiSheng or executed on an Atlas A2/A3/910B device in this environment. Header fields, API parameters, and operator-type registration names must be validated against the exact installed CANN version.