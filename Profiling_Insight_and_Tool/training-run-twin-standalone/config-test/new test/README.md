# 新增业界配置校验样本

下载日期：2026-10-12。目录中的上游文件保持原样，版本固定到具体 commit，便于重复测试。

## 三份主测试配置

| 子文件夹 | 主测试配置 | 来源与固定版本 | 配套模型结构 | 主要校验点 |
|---|---|---|---|---|
| `01_mindspeed_deepseek_v4_flash/` | `ascend_mindspeed_deepseek_v4_flash_pretrain.sh` | [Ascend/MindSpeed-LLM](https://github.com/Ascend/MindSpeed-LLM/blob/a9547f618ed4643e386a96be191258b872c4d9dc/examples/mcore/deepseek4_flash/pretrain_deepseek4_flash_4k_A3_ptd.sh) | `deepseek_v4_flash_model_config.json` | TP/PP/VPP/EP/CP、MoE、MTP、重计算、ZeRO-1 类分布式优化器、优化器 swap、BF16，以及新模型字段产生的缺口 |
| `02_nvidia_megatron_gpt_oss_20b/` | `nvidia_megatron_gpt_oss_20b_model_config.sh` | [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM/blob/6ec369d4a13da71d38a6fc95ecc2017b873f711c/examples/post_training/modelopt/conf/openai/gpt-oss-20b.sh) | `gpt_oss_20b_model_config.json` | GPT-OSS 模型结构、32 专家/top-4、BF16、sequence parallel，以及 shell 方言字段映射 |
| `03_microsoft_deepspeed_bert_large/` | `microsoft_deepspeed_bert_squad.json` | [microsoft/DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/blob/127c7a10983c09be471d51686bc29d0924cdae6b/training/BingBertSquad/deepspeed_bsz24_config.json) | `bert_large_model_config.json` | 负向样本：DeepSpeed JSON 中全局/单卡 batch、FP16、优化器未映射时，页面必须明确列出而不能静默通过 |

## 配套模型结构来源

- `deepseek_v4_flash_model_config.json`：[`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/config.json)。
- `gpt_oss_20b_model_config.json`：[`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/config.json)。
- `bert_large_model_config.json`：与 DeepSpeed SQuAD 配置同仓库、同一提交的[配套 BERT-Large 结构](https://github.com/microsoft/DeepSpeedExamples/blob/127c7a10983c09be471d51686bc29d0924cdae6b/training/BingBertSquad/ckpt/bert-large-uncased-whole-word-masking-config.json)。

## 建议测试顺序

1. 先导入一份主测试配置，检查“已应用 / 结构差异 / 已识别但未建模 / 未映射”四类报告。
2. 再导入同套的 `*_model_config.json`，重点检查层数、hidden、词表、注意力头、KV 头、FFN、专家数和 top-k。
3. DeepSeek-V4-Flash 的脚本写 `NUM_LAYERS=44`，发布模型结构写 `num_hidden_layers=43`；这是上游原始文件中的真实差异，适合确认校验页面能否明确报告，而不是静默覆盖。
4. GPT-OSS 的 `mxfp4` 发布权重量化与训练侧 `bf16` 是不同概念；校验结果应保留这种语义差异。

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

MindSpeed 脚本用的是 `NPUS_PER_NODE`，而当前解析器只用 `GPUS_PER_NODE × NNODES` 推总卡数。因此它暂时无法得到 256 卡和实际 DP，会把梯度累积反推成 1024；这也是一条适合用本样本复现的校验缺口。

