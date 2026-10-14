# 新增业界配置校验样本

首批下载日期：2026-10-12；第二批下载日期：2026-10-13。目录中的上游文件保持原样，版本固定到具体 commit，便于重复测试。

## 六组主测试配置

| 子文件夹 | 主测试配置 | 来源与固定版本 | 配套模型结构 | 主要校验点 |
|---|---|---|---|---|
| `01_mindspeed_deepseek_v4_flash/` | `ascend_mindspeed_deepseek_v4_flash_pretrain.sh` | [Ascend/MindSpeed-LLM](https://github.com/Ascend/MindSpeed-LLM/blob/a9547f618ed4643e386a96be191258b872c4d9dc/examples/mcore/deepseek4_flash/pretrain_deepseek4_flash_4k_A3_ptd.sh) | `deepseek_v4_flash_model_config.json` | TP/PP/VPP/EP/CP、MoE、MTP、重计算、ZeRO-1 类分布式优化器、优化器 swap、BF16，以及新模型字段产生的缺口 |
| `02_nvidia_megatron_gpt_oss_20b/` | `nvidia_megatron_gpt_oss_20b_model_config.sh` | [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM/blob/6ec369d4a13da71d38a6fc95ecc2017b873f711c/examples/post_training/modelopt/conf/openai/gpt-oss-20b.sh) | `gpt_oss_20b_model_config.json` | GPT-OSS 模型结构、32 专家/top-4、BF16、sequence parallel，以及 shell 方言字段映射 |
| `03_microsoft_deepspeed_bert_large/` | `microsoft_deepspeed_bert_squad.json` | [microsoft/DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/blob/127c7a10983c09be471d51686bc29d0924cdae6b/training/BingBertSquad/deepspeed_bsz24_config.json) | `bert_large_model_config.json` | 负向样本：DeepSpeed JSON 中全局/单卡 batch、FP16、优化器未映射时，页面必须明确列出而不能静默通过 |
| `04_mindspeed_glm5_744b/` | `ascend_mindspeed_glm5_744b_pretrain.sh` | [Ascend/MindSpeed-LLM](https://github.com/Ascend/MindSpeed-LLM/blob/8f0d069b657e2c57284c36666eb57c0e50ceb220/examples/mcore/glm5/pretrain_glm5_744b_4k_A3_ptd.sh) | `glm5_model_config.json` | 512 NPU、TP/PP/EP/CP、Ulysses、MoE、MLA/DSA、MTP、重计算、optimizer swap 与 noop layers |
| `05_deepspeed_phi4_superoffload/` | `deepspeed_phi4_superoffload_finetune.sh` | [DeepSpeedExamples](https://github.com/deepspeedai/DeepSpeedExamples/blob/ad2a4bd187e00901475728883ea807ae48967f22/training/DeepSpeed-SuperOffload/finetune_phi-4_1gpu.sh) | `phi4_model_config.json` | shell 内动态生成 ZeRO-3/SuperOffload JSON；验证嵌入配置是否能被发现，而不是只扫描命令行参数 |
| `06_axolotl_qwen3_32b_qlora/` | `axolotl_qwen3_32b_qlora.yaml` | [axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl/blob/e4032fc90f462fd9c2eefd2433c2f4fd0845641f/examples/qwen3/32b-qlora.yaml) | `qwen3_32b_model_config.json` | Axolotl YAML、QLoRA、4-bit、梯度累积、micro batch、checkpoint offload 与框架字段别名 |

## 配套模型结构来源

- `deepseek_v4_flash_model_config.json`：[`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/config.json)。
- `gpt_oss_20b_model_config.json`：[`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/config.json)。
- `bert_large_model_config.json`：与 DeepSpeed SQuAD 配置同仓库、同一提交的[配套 BERT-Large 结构](https://github.com/microsoft/DeepSpeedExamples/blob/127c7a10983c09be471d51686bc29d0924cdae6b/training/BingBertSquad/ckpt/bert-large-uncased-whole-word-masking-config.json)。
- `glm5_model_config.json`：[`zai-org/GLM-5`](https://huggingface.co/zai-org/GLM-5/blob/c183ef8c61faee82855eca1ed9bb3a9a7ce3b0b2/config.json)。
- `phi4_model_config.json`：[`microsoft/phi-4`](https://huggingface.co/microsoft/phi-4/blob/2db69c1c3e91a05d2c64a3185acfbaf36f744e25/config.json)。
- `qwen3_32b_model_config.json`：[`Qwen/Qwen3-32B`](https://huggingface.co/Qwen/Qwen3-32B/blob/9216db5781bf21249d130ec9da846c4624c16137/config.json)。

## 建议测试顺序

1. 先导入一份主测试配置，检查“已应用 / 结构差异 / 已识别但未建模 / 未映射”四类报告。
2. 再导入同套的 `*_model_config.json`，重点检查层数、hidden、词表、注意力头、KV 头、FFN、专家数和 top-k。
3. DeepSeek-V4-Flash 的脚本写 `NUM_LAYERS=44`，发布模型结构写 `num_hidden_layers=43`；这是上游原始文件中的真实差异，适合确认校验页面能否明确报告，而不是静默覆盖。
4. GPT-OSS 的 `mxfp4` 发布权重量化与训练侧 `bf16` 是不同概念；校验结果应保留这种语义差异。
5. GLM-5 训练脚本写 `NUM_LAYERS=80`，发布结构写 78 层，同时脚本声明 `--noop-layers 78,79`；校验时应把这种“物理槽位与有效层”差异解释清楚。
6. Phi-4 脚本通过 heredoc 动态生成 DeepSpeed JSON；当前 shell 解析只扫描 `--参数`，因此尚未读到其中的 ZeRO-3、BF16 和 SuperOffload。
7. Axolotl 用 `adapter: qlora`、`lora_r`、`micro_batch_size`，当前映射只认 LLaMA-Factory/Megatron 的同义字段；这组用于验证字段别名兼容性。

## 当前页面解析器基线

用目标页面的 `js/config-relation-import.js` 直接调用 `analyze()`，结果如下。后续改校验逻辑时可把它当回归基线：

| 文件 | 方言 | 可应用字段 | 结构字段 | 缺口 | 未识别 |
|---|---:|---:|---:|---:|---:|
| `ascend_mindspeed_deepseek_v4_flash_pretrain.sh` | sh | 17 | 5 | 0 | 67 |
| `nvidia_megatron_gpt_oss_20b_model_config.sh` | sh | 7 | 5 | 0 | 15 |
| `microsoft_deepspeed_bert_squad.json` | json | 0 | 0 | 1 | 1 |
| `deepseek_v4_flash_model_config.json` | json | 5 | 6 | 0 | 0 |
| `gpt_oss_20b_model_config.json` | json | 3 | 5 | 0 | 0 |
| `bert_large_model_config.json` | json | 1 | 4 | 0 | 0 |
| `ascend_mindspeed_glm5_744b_pretrain.sh` | sh | 17 | 5 | 0 | 57 |
| `glm5_model_config.json` | json | 4 | 8 | 0 | 0 |
| `deepspeed_phi4_superoffload_finetune.sh` | sh | 0 | 0 | 0 | 8 |
| `phi4_model_config.json` | json | 1 | 5 | 0 | 0 |
| `axolotl_qwen3_32b_qlora.yaml` | yaml | 1 | 0 | 0 | 33 |
| `qwen3_32b_model_config.json` | json | 1 | 5 | 0 | 0 |

MindSpeed 脚本用的是 `NPUS_PER_NODE`，而当前解析器只用 `GPUS_PER_NODE × NNODES` 推总卡数。因此 DeepSeek-V4 暂时无法得到 256 卡和实际 DP，会把梯度累积反推成 1024；GLM-5 也无法得到 512 卡，会把梯度累积反推成 2048。这是一条可由两组样本共同复现的校验缺口。

