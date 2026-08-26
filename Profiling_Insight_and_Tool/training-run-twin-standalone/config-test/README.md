# 外部真实训练配置 · 映射测试样本

用途：拿业界公开的训练配置去撞 `config-relation-observer.html` 的四域表单，
检验「配置里的每个属性能不能在页面上找到落点」。全部于 2026-08-26 从公开仓下载。

| 文件 | 来源 | 方言 | 适合验什么 |
|---|---|---|---|
| `mf_pretrain_deepseek3_671b.yaml` | mindspore-ai/mindformers `configs/deepseek3/` | MindFormers YAML | **最贴页面口径**：dp/mp/pp/ep + recompute_config + parallel_optimizer；MoE 大模型 |
| `mf_pretrain_qwen3_30b_a3b.yaml` | mindspore-ai/mindformers `configs/qwen3_moe/` | MindFormers YAML | 中等规模 MoE（128 专家 / topk8 / 无共享专家），`enable_parallel_optimizer: False` 档 |
| `mixtral_8x7b.sh` | NVIDIA/Megatron-LM `examples/mixtral/` | Megatron 命令行 | MoE + EP8 + PP4 + SP；Megatron 与 MindFormers 的键名对照 |
| `megatron_175b.sh` | NVIDIA/Megatron-LM `examples/gpt3/` | Megatron 命令行 | 稠密大模型 TP8×PP16，seq 2048 |
| `megatron_llama3_8b_fp8.sh` | NVIDIA/Megatron-LM `examples/llama/` | Megatron 命令行 | 唯一带 `--context-parallel-size` 的样本，验 CP 与 CP 口径 |
| `hf_qwen2_7b_config.json` | HF `Qwen/Qwen2-7B` | HF config.json | 与页面内置 qwen2-7b 预设逐项对表（体检用） |
| `hf_mixtral_config.json` | HF `mistralai/Mixtral-8x7B-v0.1` | HF config.json | 结构侧 MoE 字段（num_local_experts / num_experts_per_tok） |
| `hf_deepseekv3_config.json` | HF `deepseek-ai/DeepSeek-V3` | HF config.json | MLA + 256 专家 + 共享专家 + first_k_dense + MTP，与 openPangu 预设同形 |
| `lf_qwen3_full_sft.yaml` | hiyouga/LLaMA-Factory `examples/train_full/` | HF Trainer YAML | 反例：微调侧口径（per_device_bs / grad_accum / deepspeed） |
| `lf_qwen3_lora_sft.yaml` | 同上 `examples/train_lora/` | HF Trainer YAML | 验 LoRA 开关与 LoRA Rank |
| `ds_z3_config.json` | 同上 `examples/deepspeed/` | DeepSpeed JSON | 验「权重分片」三档里的 ZeRO-3/FSDP2 落点 |

## 页面可调项清单（对照用）

- 并行：`totalLayer` / `dp` / `pp` / `tp` / `cp`，高级折叠内 `vpp`、`cpMode`(Ulysses/Ring)
- 批次：`microBatch` / `seqLen`
- MoE：`routedExpert` / `topK` / `sharedExpert` / `ep`（+ EP 口径切出/正交开关）
- 集群：`totalRank` / 卡型号（Node 由每机 8 卡整除得出）
- 开关：`recomputeMode`(关/选择性/按层数/全开) + `recomputeLayers`、`seqParallel`、
  `shardMode`(关/ZeRO-1/FSDP2)、`vocabEmbDp`、`lora` + `loraRank`

**结构常量不可调**：hidden / vocab / heads / kvHeads / intermediate / moeIntermediate /
firstKDense / mtpLayers 全部写死在 `MODEL_PRESETS`（openpangu-flash、qwen2-7b 两档）。
拿外部配置来试时，这一层只能"挑一个形状最近的预设"，属于已知缺口。
