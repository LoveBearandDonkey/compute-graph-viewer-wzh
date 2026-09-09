# DeepSeek-V4 Flash CSA 架构图验证说明

> 四层映射、交互行为和视觉编码以同目录的 [`spec.md`](./spec.md) 为准；本文仅记录架构数据与来源校验。

## 结论

lineage-004（2026-09-07）将官方 L1/L2 与 PyPTO L3 分栏对照。CSA 官方语义来自本地 `DeepSeek-V4-Flash-Official/model.py` 及 inference 配置；实现图来自 `deepseek_v4_flash_dspark`。它是带证据的人工语义整理，不是自动从 PyTorch 编译到 PyPTO 的产物。

本轮校验：

- 官方默认 decode、layer=2、ratio=4、window=128；18 个 Op、4 个 Value、4 个 LogicalState，共 26 个 CSA 叶子，Scope 和上下文另计。
- `Attention.kv_cache` 的近期区和压缩区共用一个逻辑身份；窗口位置不再复制；主与 Indexer Compressor 状态属于独立实例。
- window positions 是索引值；主压缩 KV 内容直接服务 attention，不进入 index union；输出 inverse RoPE 位于 attention 与 grouped projection 之间。
- 官方本地行号：Q 496–499，KV 502–507，Indexer 402–433，decode cache/attention 530–534，grouped output 537–543，HC 上下文 690–693。
- 当前 q_proj_rope 定义 244–553，kv_proj_rope 556–760；Indexer query/rotation 在 decode_indexer.py 中，不是 q_proj_rope 的内部计算。
- Q/KV 的共同函数承载标为 GROUPED；其内部展开是 PyPTO 程序步骤，不是 kernel 定义，更不是运行时 task/core 分配。
- source-manifest.js 记录 SHA-256；validate-lineage.mjs 检查范围和实际文件 hash，覆盖全部 64 种混合展开状态。源码可定位与关系数值等价分开；人工映射仍标为 inferred。
- 历史 Pass Dump 尚未证明与当前源/配置快照匹配；L4 关闭静态 verified 声明，仅保留空状态与门禁接口。
- 原有整网上下文模板实为 Pro（61 层、384 experts）；本轮不扩改非 CSA 内容，入口显示“待校准”。该模板不能作为 Flash 整网配置的已验证证据。

以下旧记录保留为来源背景；其中旧远端行号与历史编译信息不自动适用于本地当前快照。

### 本轮交互验证与已知限制

- 自动检查通过：源码 hash/范围与符号、实体/血缘引用、64 种语义展开组合的可逆摘要拓扑、4 种 Q/KV 程序展开组合的端口和包含关系、L4 证据门禁。
- 浏览器检查覆盖：默认/局部/全部展开、双向多对多高亮、独立层级、Diff 与增量突出、调用/定义和多引用、根节点导航、L4 空状态、主题同步及 1280×720 布局。
- 浏览器日志出现 `MutationObserver.observe: parameter 1 is not of type Node`，来源尚未定位；上述已测交互可用，但不能据此宣称浏览器控制台无错误。需后续获取调用栈确认来源。
- 为遵守不改变现有 UI 样式的最新约束，scope 外上下文的既有虚线外观暂未改为 spec 的实线目标；L4 新形式和非 CSA 整网参数校准不属于本轮已完成项。

## 图层定位

| 开发阶段 | 图的职责 | 是否应出现 TP 通信 |
| --- | --- | --- |
| 1. 模型语义架构 | 由 DeepSeek 整网模型源码解析 Module、Op、State 与张量关系 | 否，除非模型源码本身定义分布式语义 |
| 2. 分布式算子实现（本图） | 将 CSA 落到 PyPTO `decode_csa.py` 的 kernel、缓存和 TP 数据交换 | 是；TP=1 时通信退化 |
| 3. Runtime / 硬件映射 | 将已编译任务映射到具体 Rank、NPU、核与 Timeline | 是，并应绑定实际采样证据 |

## 已确认事实

### Official `inference/model.py` 语义事实

- `Attention.forward` 的同一输入 `x` 分别进入 Q 路径、近期窗口 KV 路径、主 Compressor 与 Indexer；四条路径不是按图面顺序串联。
- Q 路径为 `wq_a → q_norm`，随后一支进入主注意力 `wq_b + RoPE`，另一支以 `qr` 形式进入 Indexer 的 query projection、RoPE 与 rotation。
- Recent-window KV 为 `wkv → kv_norm → RoPE`，并由 `get_window_topk_idxs` 生成近期窗口位置；它不是 CSA 内部调用的独立 SWA 模块。
- Main Compressor 与 `Indexer.compressor` 是两个独立实例；两者压缩比均为 4，但 Indexer Compressor 属于 Lightning Indexer。
- `topk_idxs` 由近期窗口位置与压缩位置拼接形成，随后和 Q、KV 一起进入 `sparse_attn`。
- 输出依次经过 group reshape、grouped low-rank projection 和最终 output projection。

官方源码证据：[`deepseek-ai/DeepSeek-V4-Flash-DSpark/inference/model.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/main/inference/model.py)，重点为原始文件 L386–439、L442–548（核对日期：2026-09-04）。

### PyPTO L3 实现事实

- 当前算子绑定 `config.FLASH`：隐藏维度 4096、64 个注意力头、每头 512 维、8 个输出组。
- CSA 压缩比为 4，近期原始窗口为 128，Lightning Indexer 精确选择 512 个压缩位置。
- CSA 使用三套彼此独立的持久状态：Raw KV Cache、Compressed KV Cache、INT8 Index KV Cache；主压缩器和 Indexer 压缩器也各自维护循环状态。
- Q 路径保持 rank-local；KV、主压缩和 Indexer 压缩路径消费 TP All-Gather 后的完整 token stream。
- 稀疏注意力候选由 128 个近期原始位置与 512 个选中压缩位置组成。
- 输出头按 group owner 做 Attention All-to-All，随后执行 grouped low-rank output projection 与 Reduce-Scatter。
- `hc_post` 同时完成 Attention 输出注入与四路 HC residual 重组，因此图中合并为一个语义节点。

## 层级与条件分支

- FLASH 有 43 个主模型层。
- `compress_ratios[layer_id] == 0/4/128` 分别选择 SWA/CSA/HCA。
- CSA 层为 2、4、…、42；HCA 层为 3、5、…、41；这里的 Decoder Layer 3 使用 HCA，不是产品中的 L3 PyPTO 层级。
- MoE 前三层使用 Token-ID Hash Router，3-42 层使用 `sqrt(softplus)` Learned Router。
- FLASH 使用 256 个 Routed Experts、每 token 选择 6 个，并始终运行 1 个 Shared Expert。384 experts 属于 PRO 配置，不能用于当前 FLASH 算子图。

## Trace 覆盖

`_jit_l3_decode_csa_20260903_010617` 是一个 `l3_decode_csa` 独立算子编译与运行采样：

- Backend：Ascend910B；
- World size / TP：2；
- 覆盖：单个 CSA operator invocation；
- 不覆盖：完整 43 层、MoE、LM Head 或端到端 token generation。

这里的 `l3` 是 PyPTO 编译层级命名，不代表模型的 Decoder Layer 3。

## 来源

- `config.py`：FLASH/PRO 配置、层数、压缩比、专家数与路由条件；
- `decode_layer.py`：按压缩比选择 SWA/CSA/HCA；
- `decode_csa.py`：CSA 主数据流、TP 通信、HC Pre/Post；
- `decode_compressor_ratio4.py`：主 ratio-4 KV 压缩器；
- `decode_indexer.py`：Indexer query、INT8 检索与 Exact TopK；
- `decode_sparse_attn_csa.py`：SWA + selected compressed KV 的稀疏 QK/PV；
- `moe.py`：图外的 MoE FFN 上下文；
- `_jit_l3_decode_csa_20260903_010617/distributed_meta.json`：实际编译后端与双 rank 元数据。

## lineage-005 交互回归（2026-09-07）

- 左栏从现有整网 builder 接入 canonical CSA 语义；不再使用独立 CSA iframe 与整网上下文互相切换。原有非 CSA 模板仍为待校准的 Pro 参数，不因此获得 Flash 源码验证声明。
- `validate-lineage.mjs` 通过：64 种 CSA 语义展开状态、4 种 PyPTO Q/KV 展开状态、320 种整网与祖先折叠组合；检查端点解析、去重、父子包含、稳定身份及恢复拓扑。源码引用 hash／范围检查和 JS 语法检查通过。
- 浏览器通过：Query L2 原位展开不进入 L3；CSA 标题鼠标选择显示“查看 PyPTO 实现”；明确入口打开中栏并保持左图 transform；折叠／恢复 Hybrid Attention 保留 Query 局部展开；源码双文件 tab 恢复各自位置；实现定义新增 `qkv_proj_rope.py` tab，基础两个 tab 不被替换。
- 基础 UI 外观沿用，未新增非基础可视化组件。浏览器实际尺寸受宿主影响，本次未作全断点覆盖声明。
- 浏览器仍观察到既有 `MutationObserver.observe` 的目标非 Node 日志，来源未确定；核心已验证路径可操作，不声称 console 零错误。历史编译产物与当前源码／配置对应仍未验证，L4 不新增推测数据。
