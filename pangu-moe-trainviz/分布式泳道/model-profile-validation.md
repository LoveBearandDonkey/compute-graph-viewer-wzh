# DeepSeek V4 Pro：训练模拟的模型事实边界

日期：2026-09-09。使用 model-architecture-extractor 的来源 / 配置 / trace 分离流程核对；不从 ST 提取 DeepSeek 模型结构，不重建共享模型渲染器。

## 已验证模型配置

来源：[DeepSeek-V4-Pro 官方 config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/config.json)。本目录 data/deepseek-v4-pro.config.json 是该配置中所需架构字段的摘录，不是完整官方训练配置。

- 61 个主 Decoder 层，hidden size 7168。
- 384 个 Routed Experts、1 个 Shared Expert、每 token Top-6，前 3 层 Hash Router。
- compress_ratios 有 **62 项**：索引 0–60 对应主层，索引 61 对应额外 MTP。
- 主层 L0/L1 为 HCA；L2–L60 偶数 CSA、奇数 HCA；L60 为 CSA。最后的 ratio=0 属于 MTP 的 SWA，不能错配到 L60。
- MTP depth=1；模拟里用一个聚合训练块表达，不展开其内部 kernel。
- checkpoint 的 inference dtype / quantization 不是训练精度配置，本模拟不借此宣称 BF16 / FP8 / FP4 训练性能或内存可行性。

## 模型结构与训练轨迹的区别

模型图沿用 Pro 整网角色模板；仅 L2 提供典型算子事件，其他层为无算子映射的阶段概览块，不复制 L2 明细充当全层 trace。模型包含关系不从 profiling 的事件顺序推导。

该模板入口增加 trainingProfile=pro：禁止把 Flash 官方 decode CSA 的细节点与来源声明注入 Pro；CSA/HCA 保持角色级折叠。MTP 与 Training Loss 作为训练上下文加入共享图数据，由同一 renderer 绘制。原 Flash workbench 的默认入口不改变。

这不是声称拥有完整的 Pro 训练源码或真实 kernel 映射。HC Pre / RMSNorm 等组合事件是模拟角色粒度；反向 owner 对应同一个前向模型角色，而不是另造模型节点。

## Trace 覆盖

ST step 23 的来源模型没有绑定到 DeepSeek。它只贡献 32 对可复核 F/B 记录的时长比中位数 2.1284895795426912，以及对 1F1B 调度形态的观察。

DeepSeek JSON 只提供 L2 的典型算子、该层的 EP 通信与激活驻留。其余层和输出头 / MTP / Loss 合并为概览块，辅以 PP 通信、梯度同步及 Optimizer，呈现 8 MB / DP 的完整 step 调度形态。所有事件 fidelity=simulated。这是轻量视觉与交互 demo，不追求全层、全算子覆盖，也不是设备性能或硬件容量的证据。
