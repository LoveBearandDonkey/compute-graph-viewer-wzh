# PanGu-Sophon-PyTorch 问答记录

> **DEMO 说明**：本 demo 使用 **128 rank world size**，对应仓库中 **92B Omni MoE 模型** 的配置。

---

---

## DEMO 并行配置 — 92B Omni MoE（256 expert）

本配置来源于仓库以下三篇文档（stage2, seq 32K）：

| 文档 | 路径 |
|------|------|
| 模型定义 | `https://codehub-g.huawei.com/AI_Infra/AI_Infra_Training/pangu_sophon_pytorch/files?ref=master&filePath=config%2Fomni%2F92BOmni_MOE%2FPILOT%2F92BOMNI_MOE_stage2%2Fmodel_92BOMNI_MOE_stage2_seq_32K.yaml&isFile=true` |
| 并行策略 | `https://codehub-g.huawei.com/AI_Infra/AI_Infra_Training/pangu_sophon_pytorch/files?ref=master&filePath=config%2Fomni%2F92BOmni_MOE%2FPILOT%2F92BOMNI_MOE_stage2%2Fmodel_92BOMNI_MOE_stage2_seq_32K_device_C_accelerate.yaml&isFile=true` |
| 训练配置 | `https://codehub-g.huawei.com/AI_Infra/AI_Infra_Training/pangu_sophon_pytorch/files?ref=master&filePath=config%2Fomni%2F92BOmni_MOE%2FPILOT%2F92BOMNI_MOE_stage2%2Fmodel_92BOMNI_MOE_stage2_seq_32K_train.yaml&isFile=true` |

### 模型架构

| 组件 | 参数 | 一句话说明 |
|------|------|-----------|
| **LLM** | 49层, hidden=2560, **MLA** (q_lora=1024, kv_lora=512, qk_nope_dim=128, qk_rope_dim=64, v_dim=128, head=48), sandwich_norm, post_norm_layers=[1,5,10,15,20,25,30,35,40] | 49层 decoder-only 架构，使用 DeepSeek 式 MLA 注意力省 KV cache，sandwich norm（前后各归一化）稳定训练，部分层后加一次 post norm |
| **FFN** | hidden=9216, SwiGLU | 前馈网络，用 SwiGLU 激活函数（比 ReLU 效果好，参数量略大） |
| **MoE** | **256 expert**, topk=8, 前2层([0,1]) dense, routed_hidden=1024, shared_hidden=1024, noaux_tc, sigmoid_gating, routed_scaling=2.5, expert_bias | 混合专家：256 个专家，每个 token 激活 top-8，前 2 层不用 MoE 保底，共享 expert 处理通用知识，noaux_tc 无辅助损失，sigmoid 门控替代 softmax |
| **MLA 特殊** | SWA sliding window (511→2047 共33层), **ModAttn** 开启 | 滑动窗口注意力：前 30 层窗口 511，后 3 层扩大到 2047 捕捉长距离；ModAttn 增强输出映射 |
| **MTP** | 4 额外预测头, loss_weight=0.3 | 多 token 预测：用 4 个额外头同时预测未来 token，loss_weight=0.3 作为辅助损失 |
| **MHC** | 4 streams, 20 recur norm | 超连接：用 4 条并行流增强跨层信息传递，20 步循环归一化稳定训练 |
| **Visual** | UNIT_1B_anyres ViT, 24层, hidden=1024, head=16, GELU, LayerNorm, 3D-RoPE (section 8,12,12) | 视觉编码器：1B 参数 ViT，支持任意分辨率输入（切 patch 后动态拼凑），3D-RoPE 编码时空位置 |
| **Audio** | Whisper encoder, 16层, hidden=2048, head=16, ffn=8192 | 音频编码器：基于 OpenAI Whisper，16 层 2048 hidden，处理语音输入 |
| **Tokenizer** | Pangu_Vocab, padded=151552 (16对齐) | 词表 151552，16 对齐（硬件友好的 padded vocab，对齐到 16 的倍数） |
| **RoPE** | MRoPE (section 12,10,10), base=6,400,000, max_seq=524K | 多维旋转位置编码：3 个维度分别编码文本/图像/音频位置，超大 base 支持长序列外推 |

### 并行策略

| 维度 | 配置 | 一句话说明 |
|------|------|-----------|
| **TP** | 4 (mc2 overlap) | 张量并行：将单个 Transformer 层切到 4 张卡上，mc2 让计算和通信重叠 |
| **PP** | 1 | 流水线并行=1 即不启用。92B 总参数中绝大部分是 MoE 专家参数，已通过 EP=16 及可能的 TP 拓展 EP 形成 64 路有效专家切分（每 rank 约 4 个专家）；Dense 部分由 TP=4 切分，激活由 CP=2+SP 切分，优化器状态由 Distributed Optimizer 切分，单 rank 显存已足够，无需 PP 再切深度。PP>1 还会引入 stage 间 P2P 通信、流水线空泡和负载均衡问题 |
| **CP** | 2 (Ulysses) | 上下文并行：将序列长度切到 2 张卡上，适用于长序列场景 |
| **EP** | 16 | 专家并行：256 个 expert 分布到 16 个 EP 组。EP 与 TP 共享同一组 rank，若开启 `--moe-tp-extend-ep`，有效专家切分域为 TP×EP=64，每 rank 存放 256÷64=4 个路由专家 |
| **SP** | 开启 | 序列并行：与 TP 配合，将 LayerNorm / Dropout 的序列维度也切分，省显存 |
| **World size** | 128 (Dense: TP4×CP2×DP16×PP1; 专家: ETP4×EP16×EDP2×PP1) | 总卡数 128。PanGu 采用 Parallel Folding 式组织，Dense 和 MoE 路径在同组 rank 上建立两套并行网格 |
| **DP (Dense)** | 16, Distributed Optimizer, bucket=80MB | Dense 路径（Attention/Dense FFN/LayerNorm）的数据并行度 DP=16。优化器状态分布式存储省显存 |
| **EDP (Expert)** | 2 | 专家路径的数据并行度 EDP=2。专家参数已被 EP 进一步分散，完整数据副本只有 2 份 |
| **EP over SP** | 是 | EP 通信走序列并行域，减少通信量 |
| **Shared Expert SP** | 是 | 共享 expert 也走序列并行 |
| **Token Dispatcher** | pangu_dropless | MoE token 分发策略：不丢弃任何 token 的 dropless 方案 |
| **Expert Placement** | 启用 (fine-grained, threshold=0.08, freq=500) | 动态专家放置：运行时根据负载将 expert 迁移到合适的卡上，threshold 0.08 表示负载偏差超 8% 即触发 |
| **Hetero Schedule** | 开启 (visual_mbs=1, audio_mbs=1, swap encoder feature) | 异构调度：解决多模态 encoder 与 LLM decoder 计算量不均衡的问题 |
| **Visual/Audio DP** | encoder_dp=true | 视觉/音频编码器走数据并行，不参与 TP/CP 分片 |

### 训练配置

| 参数 | 值 | 一句话说明 |
|------|-----|-----------|
| **Optimizer** | Muon (beta1=0.9, beta2=0.95) | 优化器：比 AdamW 更节省显存的矩阵正交化优化器，适合大模型 |
| **LR** | cosine, 2e-4 → 2e-5, warmup 3% | 学习率：cosine 衰减策略，从 2e-4 降到 2e-5，前 3% 的 step 线性预热 |
| **Weight decay** | 0.1, 排除 word_embeddings/layernorm | 权重衰减：防止过拟合，但 embedding 和 norm 层不加 decay |
| **Batch size** | 4 | 每张卡的 micro batch size。GBS = MBS × DP × GAS，这里 DP=16（Dense 路径），则 GAS = GBS ÷ (4 × 16) ÷ 1（无梯度累积） |
| **Seq len** | 32K (decoder=0, encoder=32768) | 序列长度 32K，decoder=0 说明是 encoder-only 模式（多模态） |
| **Precision** | bf16 compute, fp32 softmax/logits/grad_accum | 混合精度：计算用 bf16 省显存加速，softmax/logits/梯度累加用 fp32 保精度 |
| **Recompute** | full block (LLM 49层, Visual 26层), activation offload | 重计算：前向不存中间激活，反向重新算一遍，省显存换计算量；offload 到 CPU 进一步省 NPU 显存 |
| **Fused ops** | flash_attention, rmsnorm, swiglu, rope, moe_permute, mhc_ascendc, fused_modattn, sink_fa | 融合算子：将多个小算子合并为一个大算子，减少 kernel launch 开销和显存读写 |
| **Freeze** | ViT/LLM/projector frozen, audio 可训练, partial_freeze_we | 冻结策略：冻结视觉/语言/投影层，只训练音频编码器和部分 embedding token（音频相关 token） |
| **Checkpoint** | aiturbo, 每 500 steps | 保存策略：使用 AIturbo 异步保存，每 500 step 存一次，不阻塞训练 |

---

## Q1：这个仓库是什么，有哪些内容？

**A：** 这是 **PanGu-Sophon-PyTorch**，一个基于 Megatron-LM、面向华为昇腾 NPU (Ascend 910) 的大模型预训练框架。

### 仓库结构

| 目录 | 说明 |
|------|------|
| `PanGu/` | 主源码：入口、分布式策略、模型组件、数据管线、训练流程 |
| `config/` | 模型配置 YAML |
| `launch.py` | 训练启动器 |
| `tests/` | 单元/集成测试 |
| `third_party/` | Megatron-LM、MindSpeed、AITurbo 等子模块 |
| `tools/` | 转换及辅助工具 |
| `doc/` | 架构、并行、训练、推理等设计文档 |
| `scripts/` | 环境脚本和工具脚本 |

核心能力：支持 GPT/BERT/T5 等 Transformer、张量并行/流水并行/数据并行、Checkpoint、容灾、多模态训练和 SFT/DPO。

---

## Q2：PanGu 有哪些模型？有 MoE 吗？训练配置是什么？

### 模型架构一览

| 模型 | 说明 |
|------|------|
| **GPT / LLM** | 标准 decoder-only Transformer，支持 GQA / MLA / DSA 三种 Attention |
| **GPT + MTP** | GPT 扩展 Multi-Token Prediction（同时预测多个未来 token） |
| **GPT + MoE** | 相同 GPT 骨架，将部分/全部 FFN 替换为 MoE Layer |
| **Omni** | 视觉-语言-音频多模态 VLM（ViT + Whisper/Huanyu + LLM backbone），3B/5B/7B/30B/38B/92B |
| **PanguGen** | 文生图模型（ViT + VAE + FlowMatching Diffusion + LLM），18B |
| **PanguGenPILOT** | Transfusion 架构的文生图变体 |

### MoE 支持

**是，MoE 是一等公民特性**，代码完备：

- 核心实现在 `PanGu/models/transformer/moe/`：
  - `moe_layer.py` — MoELayer 顶层封装
  - `router.py` — TopKRouter 路由器
  - `experts.py` — GroupGemmExperts / SequentialExperts 等专家计算策略
  - `token_dispatcher.py` — All-to-All token 调度
  - `expert_placement/` — 细粒度专家放置子系统
- 配置参数：`num_experts` (64/256/384)、`topk`、`shared_experts`、`load_balancing_type` (noaux_tc/group_aux_loss)、`router_type` 等
- 现有 MoE 配置：**92B Omni MoE**（256 expert，3 stage，含 DSA）、**30B A2B Omni**（384 expert）

### 训练配置结构

每个模型目录下有三类 YAML 文件：

```
*_seq_XXK.yaml         → 模型定义（层数、head数、hidden size、attention类型、MoE、MTP、RoPE、tokenizer）
*_train.yaml           → 训练配置（stage、数据集、batch size、seq len、优化器、LR schedule、loss类型）
*_device_*.yaml        → 并行策略（TP/PP/CP/EP/DP/SP、recompute、精度、MoE专用设置、FSDP2）
```

模型规模：3B / 5B / 7B / 18B / 30B / 38B / 92B

训练流水线：pretrain → Stage1 → Stage2 → Stage3 (128K) → SFT → DPO

并行策略：TP 1~8、PP（可配 layer 划分）、CP（长序列）、EP（MoE）、FSDP2/DP、SP，支持异构调度。

---

## Q3：30B A2B Omni（384 expert）的具体模型配置和并行策略？

### 模型架构 (stage2, seq 32K)

| 组件 | 参数 |
|------|------|
| **LLM** | 37层, hidden=2560, GQA (head=24, kv_group=4, kv_channels=128), RoPE-MRoPE (section 6,5,5) |
| **FFN** | hidden=6144, SwiGLU, post_norm_scale=0.03086 |
| **MoE** | **384 expert**, topk=8, 前2层([0,1])保持 dense, routed_hidden=256, shared_hidden=512, router_type=learnable, sigmoid_gating, noaux_tc (aux_loss_coeff=0.0002), z_loss_coeff=1e-6, 带 expert_bias |
| **MHC** | 4 streams, 20 recur norm |
| **Visual** | UNIT_1B_anyres ViT, 26层, hidden=1280, head=16, GELU, GatedMerger, Hiev3 projector, anyres (min 50K~max 1.8M pixels), window_attention |
| **Audio** | Huanyu encoder, 24层, hidden=768, head=8, ffn_hidden=3072 |
| **Tokenizer** | Pangu_Vocab, padded=151552 (16对齐) |

### 并行策略

| 维度 | 配置 |
|------|------|
| **TP** | 2 |
| **PP** | 1 |
| **CP** | 4 |
| **EP** | 32 |
| **SP** | 开启 |
| **DP** | Distributed Optimizer, overlap param gather / grad reduce |
| **EP over SP** | 是 |
| **Shared Expert SP** | 是 |
| **Token Dispatcher** | pangu_dropless |
| **Visual DP** | encoder_dp=true |
| **Audio DP** | encoder_dp=true |

### 训练配置

- **Optimizer**: Muon (beta1=0.9, beta2=0.95), visual 也使用 Muon
- **LR**: cosine, 6e-5 → 6e-6, warmup 3%
- **Weight decay**: 0.1, 排除 word_embeddings/layernorm
- **Batch size**: 2048
- **Seq len**: 32K (decoder=0, encoder=32768)
- **Precision**: bf16 compute, fp32 softmax/logits/grad_accum
- **Recompute**: full block recompute (LLM 37层, Visual 26层, Audio 24层), activation offload
- **Fused ops**: flash_attention, rmsnorm, swiglu, rope, gmm_gradient_accumulation, moe_permute, mhc_ascendc
- **Freeze**: ViT frozen, LLM frozen, projector frozen, audio encoder 可训练, partial_freeze_we (仅可训练音频相关 token)
- **Checkpoint**: aiturbo, 每 400 steps, ViT 从独立路径加载
- **Eval**: 每 1000 interval
- **卡数**: Dense 路径 TP(2) × CP(4) × DP(32) × PP(1) = **256 卡**; 专家路径 ETP(2) × EP(32) × EDP(4) × PP(1)

---

## Q4：92B Omni MoE（256 expert）的具体模型配置和并行策略？

### 模型架构 (stage2, seq 32K)

| 组件 | 参数 |
|------|------|
| **LLM** | 49层, hidden=2560, **MLA** (q_lora=1024, kv_lora=512, qk_nope_dim=128, qk_rope_dim=64, v_dim=128, head=48), sandwich_norm, post_norm_layers=[1,5,10,15,20,25,30,35,40] |
| **FFN** | hidden=9216, SwiGLU |
| **MoE** | **256 expert**, topk=8, 前2层([0,1]) dense, routed_hidden=1024, shared_hidden=1024, noaux_tc, sigmoid_gating, routed_scaling=2.5, expert_bias |
| **MLA 特殊** | SWA sliding window (511→2047 共33层), **ModAttn** 开启 |
| **MTP** | 4 额外预测头, loss_weight=0.3 |
| **MHC** | 4 streams, 20 recur norm |
| **Visual** | UNIT_1B_anyres ViT, 24层, hidden=1024, head=16, GELU, LayerNorm, 3D-RoPE (section 8,12,12) |
| **Audio** | Whisper encoder, 16层, hidden=2048, head=16, ffn=8192 |
| **Tokenizer** | 同 30B, padded=151552 |
| **RoPE** | MRoPE (section 12,10,10), base=6,400,000, max_seq=524K |

### 并行策略

| 维度 | 配置 |
|------|------|
| **TP** | 4 (mc2 overlap) |
| **PP** | 1 |
| **CP** | 2 (Ulysses) |
| **EP** | 16 |
| **SP** | 开启 |
| **DP** | Distributed Optimizer, bucket=80MB |
| **EP over SP** | 是 |
| **Shared Expert SP** | 是 |
| **Token Dispatcher** | pangu_dropless |
| **Expert Placement** | **启用** (fine-grained, threshold=0.08, freq=500) |
| **Hetero Schedule** | 开启 (visual_mbs=1, audio_mbs=1, swap encoder feature) |
| **Visual/Audio DP** | encoder_dp=true |

### 训练配置

- **Optimizer**: Muon (lr=2e-4→2e-5, cosine, warmup=3%)
- **Batch size**: 4
- **Precision**: bf16, fp32 softmax/logits/grad_accum
- **Recompute**: full block (LLM 49层, Visual 26层), activation offload
- **Fused ops**: flash_attention, rmsnorm, swiglu, rope, moe_permute, mhc_ascendc, **fused_modattn, sink_fa**
- **Freeze**: ViT/LLM/projector frozen, audio 可训练, partial_freeze_we
- **Checkpoint**: aiturbo, 每 500 steps
- **Profile**: dynamic (step 10-12)
- **卡数**: Dense 路径 TP(4) × CP(2) × DP(16) × PP(1) = **128 卡**; 专家路径 ETP(4) × EP(16) × EDP(2) × PP(1)

### Stage3 扩展

Stage3 使用 **128K seq** 并包含 **DSA** 变体：
- `dsa_dense_warmup` — 先 dense warmup
- `dsa_sparse_training` — 再切到 sparse training

---

## Q5：92B 49层为什么 PP 只用 1？30B 和 92B 分别用多少卡？

### 卡数推算

| 模型 | 推算公式（Dense路径 DP 视角） | 总卡数 | 详细并行网格 |
|------|------|--------|----------|
| **30B A2B** | TP(2) × CP(4) × PP(1) × DP=32 → **256** | **256** | Dense: TP2×CP4×DP32×PP1; 专家: ETP2×EP32×EDP4×PP1 (384 experts, 每 rank 6 experts) |
| **92B MoE** | TP(4) × CP(2) × PP(1) × DP=16 → **128** | **128** | Dense: TP4×CP2×DP16×PP1; 专家: ETP4×EP16×EDP2×PP1 (256 experts, 每 rank 4 experts) |

**关键理解**：PanGu 采用 Megatron/MindSpeed 的 Parallel Folding 式组织，Dense 路径和 MoE 专家路径在同一组 rank 上建立两套并行网格：

- **Dense 路径（Attention / Dense FFN / LayerNorm）**：\
  `world_size = TP × CP × PP × DP` \
  DP = world_size ÷ (TP × CP × PP)，不受 EP 影响

- **专家路径（MoE 专家参数）**：\
  `world_size = ETP × EP × PP × EDP` \
  EDP = world_size ÷ (ETP × EP × PP)，Dense 的 DP 和专家 EDP 是两个不同的值

### PP=1 的准确原因

92B MoE 用 PP=1 不是因为「EP 替代了 PP」，而是几个条件同时成立：

1. **92B 总参数中绝大部分是 MoE 专家参数**。专家参数已通过 EP=16 及可能的 TP 拓展 EP (ETP=4) 形成 64 路有效专家切分，每 rank 只保存约 4 个路由专家，不保存完整 92B 参数。激活参数约 6B/token
2. **Dense 路径由 TP=4 切分**，Attention 和共享专家不需要单卡完整承担
3. **长序列激活由 CP=2 + SP 切分**，PP 不是唯一的激活显存缓解手段
4. **优化器状态由 Distributed Optimizer 分布式存储**，进一步降低每 rank 模型状态显存
5. **当前配置下单 rank 显存已能容纳**，PP 不是内存可行性的必要条件
6. **PP>1 的代价**：引入 stage 间 P2P 通信、流水线预热/排空空泡、49 层含多种 Attention 类型 (SWA/ModAttn) 和 MTP/MHC，均匀划分 stage 不简单

> ⚠️ 说明：上述解释基于开源 MindSpeed 实现和提供的并行配置推导得出。盘古团队未公开正式说明「为何选择 PP=1」。

---

## Q6：训练流水线 pretrain → Stage1 → Stage2 → Stage3 → SFT → DPO 各阶段说明

| 阶段 | 说明 |
|------|------|
| **Pretrain** | 从零预训练，全参数训练 LLM backbone + vision/audio encoder，海量图文/音视频数据 |
| **Stage 1** | 多模态对齐。冻结部分参数 (ViT/LLM)，主要训练 projector 和音频模块，让视觉/音频特征与 LLM embedding 对齐，通常较短 seq (4K-8K) |
| **Stage 2** | 多模态指令跟随。解冻更多参数（如 LLM），用 SFT 风格数据训练，seq 扩展到 32K，加入 MHC、loss mask 等 |
| **Stage 3** | 长上下文扩展。seq 扩展到 128K 甚至 524K (92B max_seq)，使用 DSA/Dense Sparse Attention 等长序列优化。仅 92B 有 stage3 |
| **SFT** | 监督微调，高质量对话数据提升指令跟随能力 |
| **DPO** | 偏好对齐，用偏好数据对做 Direct Preference Optimization |

不同模型覆盖的阶段不同：30B A2B 有 stages 1/2/SFT，92B 有 stages 1/2/3 (+ DSA)，7B V5 最完整：Pretrain → Stage1/2/3 → SFT → DPO。

---

## Q7：PanGu 模型都是多模态模型吗？

**不全是。** 区分如下：

| 模型 | 是否多模态 | 说明 |
|------|-----------|------|
| **GPT / LLM** | ❌ | 纯文本，decoder-only，入口 `pretrain_gpt.py` |
| **GPT + MTP** | ❌ | 纯文本 + 多 token 预测 |
| **GPT + MoE** | ❌ | 纯文本 MoE |
| **Omni** | ✅ | 视觉 + 音频 + 文本 VLM |
| **PanguGen** | ✅ | 文生图 (ViT + VAE + Diffusion) |

配置文件区分：`vlm/`、`omni/` 是多模态，`sample_4k/` 等是纯语言。

---

## Q8：这个仓库有训练过程的 trace 数据吗？

**没有。** 仓库只有**训练框架代码和配置**，不含任何训练产生的 trace/profile 数据。

训练时 trace 数据来源于以下途径，但数据本身不在仓库中：
1. **torch_npu profiler** (`PanGu/training/utils.py:2660`) — 生成 `profiler_config.json` 和 timeline trace，保存到训练节点本地或 OBS
2. **DeepTrace** (`PanGu/monitor/metrics/writers.py`) — 华为内部 deeptrace 系统，数据上报 ROMA
3. **Monitor 模块** (`PanGu/monitor/`) — 网络 hook、内存 profiler、系统心跳，日志本地落盘

所有 trace 数据只在**实际训练运行时**产生，保存在昇腾 NPU 训练节点上。

---

## Q9：doc/ 目录下有哪些文档？

### 按类别分组

<details>
<summary><b>📐 系统架构 (1篇)</b></summary>

| 文档 | 一句话说明 |
|------|-----------|
| 训练系统架构设计说明书 | PanGu 训练系统整体架构设计，528行 |

</details>

<details>
<summary><b>📡 通信优化 (10篇)</b></summary>

| 文档 | 一句话说明 |
|------|-----------|
| 1F1B通信掩盖特性说明书 | 前向/反向计算的通信与计算重叠，降低 MoE all-to-all 和 PP P2P 等待 |
| DP并行通信掩盖特性说明书 | DP 梯度 reduce-scatter / 参数 all-gather 与计算重叠 |
| PP通信Shape重构方案 | 通过缓存 tensor shape 避免 PP 间冗余 shape 通信 |
| PP并行自定义负载划分特性说明书 | 允许手动调整各 PP stage 的层数分配，解决负载不均 |
| P2P通信融合算子说明书 | 将 PP 多次 isend/irecv 合并为一次发送/接收 |
| MC2通算融合算子说明书 | TP/EP 场景下 Linear 计算与 all-gather/all-reduce 重叠执行 |
| FSDP2通信掩盖优化特性说明书 | 异步融合 AllReduce，跨模块通信/计算重叠 |
| FSDP2反向RS流水深度可配特性说明书 | FSDP2 反向 reduce-scatter 流水线深度 + 双 HCCL communicator |
| 分级EP通信特性说明书 | 跨节点 EP all-to-all 分层处理（节点内高带宽 + 节点间 gather） |
| Group Collective Gather与Scatter通信接口说明书 | 提供等长/变长的 Gather/Scatter 通信接口 |

</details>

<details>
<summary><b>⚙️ 计算融合 (20篇)</b></summary>

| 文档 | 一句话说明 |
|------|-----------|
| SwiGLU融合算子说明书 | SwiGLU 激活函数适配昇腾的融合算子 |
| RMSNorm融合算子说明书 | RMSNorm layer norm 融合实现 |
| RoPE融合算子说明书 | RoPE 位置编码的融合实现 |
| RoPE缓存加速特性说明书 | RoPE 位置编码的缓存加速 |
| Permute融合算子说明书 | MoE token dispatch 的 permute/unpermute 融合 |
| ModAttn融合算子说明书 | Masked Output Map Enhancement 的融合实现 |
| MHC融合算子说明书 | Manifold-Constrained Hyper-Connections 融合算子 |
| Param Sink融合算子说明书 | Attention 固定 sink token 的融合实现 |
| GMM梯度累加融合算子说明书 | Grouped GEMM (MoE 专家计算) 的反向梯度累加融合 |
| Matmul梯度累加融合算子说明书 | 普通矩阵乘法的梯度累加融合 |
| Attention mask压缩特性说明书 | Flash Attention mask 的压缩表示 |
| Attention mask快速构造特性说明书 | 多模态场景下快速重置 attention mask |
| 框架侧对FA结果rescale加权实现sink说明书 | Flash Attention 结果 rescale 加权实现 sink 效果 |
| Core Attention重计算特性说明书 | Attention 部分重计算以省显存 |
| 动态专家迁移特性说明书 | 运行时根据专家负载动态迁移到不同设备 |
| QAT融合算子说明书 | 量化感知训练 (INT4/INT8) 的融合算子 |
| Encoder DP重构支持DP&CP特性说明书 | 多模态 encoder 的 DP + CP 并行重构 |
| TP域数据CPU内存共享特性说明书 | TP 组内 CPU 侧共享数据 buffer |
| 优化器buffer对齐特性说明书 | optimizer buffer 对齐以提升计算效率 |
| Attention负载均衡说明书 | (已废弃) CP Attention 负载均衡已删除 |

</details>

<details>
<summary><b>🧠 模型算法 - MoE (5篇)</b></summary>

| 文档 | 一句话说明 |
|------|-----------|
| 路由特性说明书 | MoE TopKRouter 的设计与实现细节 |
| 共享专家特性说明书 | MoE 中 shared expert 的机制与配置 |
| RouterReplay特性说明书 | Router 路由历史重放，缓解负载不均衡 |
| 专家负载均衡特性说明书 | MoE 各路负载均衡算法的设计与选择 |
| Micro-DP负载均衡域特性说明书 | DP 组内更细粒度的负载均衡域 |

</details>

<details>
<summary><b>🧠 模型算法 - Attention (5篇)</b></summary>

| 文档 | 一句话说明 |
|------|-----------|
| Attention相关特性说明书 | GQA / MLA / Aug GQA 等 Attention 变体 |
| DSA算法设计说明书 | Dynamic Sparse Attention + SWA sliding window 设计 |
| RoPE相对位置编码特性说明书 | RoPE、MRoPE、Partial RoPE 等变体 |
| Param Sink特性说明书 | Attention 中固定 sink token 的位置编码附加 |
| 样本隔离特性说明书 | Reset Attention Mask & Position IDs 实现样本隔离 |

</details>

<details>
<summary><b>🧠 模型算法 - Structure (13篇)</b></summary>

| 文档 | 一句话说明 |
|------|-----------|
| MHC特性说明书 | Manifold-Constrained Hyper-Connections 结构设计 |
| MTP特性说明书 | Multi-Token Prediction，一次预测多 token |
| Muon特性设计说明书 | Muon 优化器原理与实现 (659行，含伪代码) |
| ModAttn特性说明书 | Masked Output Map Enhancement attention 改进 |
| Sandwich Norm特性说明书 | 层前后 sandwich normalization |
| 稠密稀疏层混合特性说明书 | Dense/MoE 层混合配置策略 |
| 共享Embedding特性说明书 | 是否共享 embedding 与 lm_head 的配置 |
| Vanilla激活函数特性说明书 | 原生激活函数（非融合版本） |
| Gradient Scaling机制设计说明书 | 梯度缩放机制设计 |
| Loss Func模块设计说明书 | loss function 模块设计 |
| Preprocess模块设计说明书 | Embedding 前处理模块 |
| 量化感知训练(QAT)算法设计说明书 | 量化感知训练算法设计 |

</details>

<details>
<summary><b>🧠 模型算法 - Multimodal (5篇)</b></summary>

| 文档 | 一句话说明 |
|------|-----------|
| 多模Omni模型设计说明书 | Omni 多模态模型整体架构 |
| 视觉编码器设计说明书 | ViT 视觉编码器设计 |
| Whisper语音编码器设计说明书 | Whisper 音频编码器接入设计 |
| 寰宇编码器算法设计说明书 | 华为自研 Huanyu 音频编码器 |
| 多模态Embedding分布式插入All2All特性设计说明书 | 多模态 token 通过 All2All 插入 LLM embedding |

</details>

<details>
<summary><b>🧠 模型算法 - Low Precision (6篇)</b></summary>

| 文档 | 一句话说明 |
|------|-----------|
| 低精度训练特性说明书 | 整体低精度训练方案 |
| MXFP8配方特性说明书 | MXFP8 精度格式的配方/配置 |a
| MXFP8_FSDP2低精度训练特性说明书 | FSDP2 分布式场景下 MXFP8 低精度训练 |
| Hyper FSDP支持lMXFP8设计说明书 | Hyper FSDP 对低精度 MXFP8 的支持 |
| 细粒度低精度训练控制说明书 | 不同模块/层独立配置精度 |
| 低精度模型参数特性说明书 | 模型参数低精度存储 |

</details>

<details>
<summary><b>🧠 模型算法 - Generate (1篇)</b></summary>

| 文档 | 一句话说明 |
|------|-----------|
| 多模态生成算法设计说明书 | 文生图/图文生成算法设计 |

</details>

<details>
<summary><b>🏋️ 训练 (4篇)</b></summary>

| 文档 | 一句话说明 |
|------|-----------|
| 34BA3模型910B训练说明书 | 34B A3 模型在 910B NPU 上的训练配置 (564行) |
| 34BA3模型910C训练说明书 | 34B A3 模型在 910C NPU 上的训练配置 (780行) |
| 91BA5模型910C训练说明书 | 91B A5 模型在 910C 上的训练配置 (691行) |
| 505BA18模型910C训练说明书 | 505B A18 模型在 910C 上的训练配置 (970行) |

</details>

<details>
<summary><b>🔮 推理 (3篇)</b></summary>

| 文档 | 一句话说明 |
|------|-----------|
| 纯Prefill推理特性说明书 | 不实现 KV cache，每生成一个 token 重新执行完整 forward，用于训推精度对齐 |
| 多模态PurePrefill推理实现说明 | Omni 多模态模型的 PurePrefill 推理实现 |
| Infer_logp特性说明书 | 推理时输出 token log probability 的特性 |

</details>

<details>
<summary><b>🛡️ 可用性/容灾 (9篇)</b></summary>

| 文档 | 一句话说明 |
|------|-----------|
| AIturbo权重快速异步保存加载特性说明书 | 基于昇腾 AIturbo 的异步 checkpoint 加速 |
| 通用断点续训特性说明书 | 基础断点续训方案 |
| 多模态断点续训特性说明书 | 多模态场景下支持 DP size / num_workers / 数据集配比变化的续训 |
| Step级快速恢复特性说明书 | ECC 场景下以 step 粒度快速恢复训练 |
| Hyper_DCP权重保存加载特性说明书 | 大规模分布式 checkpoint 的 DCP 方案 |
| Hyper_DCP支持MTP_Head变化特性设计说明书 | DCP 加载时兼容 MTP head 增减 |
| 重跑跳过特性说明书 | 容灾恢复时跳过已完成的 step |
| IntervalActionCallback特性设计说明书 | 按间隔执行回调动作 (save/eval 等) 的框架 |
| Interval_pre_check特性设计说明书 | 在 interval 到达前检查参数有效性 |

</details>

<details>
<summary><b>📊 数据集 (9篇)</b></summary>

| 文档 | 一句话说明 |
|------|-----------|
| Omni-Dataset-V2特性设计说明书 | 第二代多模态数据集流水线整体设计 |
| BatchCollator输出格式说明 | Omni-Dataset-V2 的 BatchCollator 输出格式 |
| DataConfig格式说明 | Omni-Dataset-V2 的 DataConfig 配置格式 |
| Lance格式说明 | Omni-Dataset-V2 的 Lance 存储格式 |
| Tar存储格式说明 | Omni-Dataset-V2 的 Tar 存储格式 |
| PanguML格式说明 | Omni-Dataset-V2 的 PanguML 数据格式 |
| Processor输出/输入格式说明 | Omni-Dataset-V2 的 Processor 输入输出格式 |
| 离线数据格式说明 | Omni-Dataset-V2 的离线数据处理格式 |
| Omni离线数据缓存特性说明书 | 多模态数据离线缓存加速 |
| Tokenizer并发构建文件锁特性说明书 | 多进程构建 tokenizer 缓存的文件锁机制 |
| PanguGenDataset设计与使用说明书 | 文生图模型的数据集设计与使用 |
| DCD-Lance特性设计说明书 | 数据压缩/去重/Lance 存储格式 |

</details>

<details>
<summary><b>📝 其他 (3篇)</b></summary>

| 文档 | 一句话说明 |
|------|-----------|
| 参数配置checklist | 所有配置参数的完整清单与说明 (526行) |
| 多模态加速特性说明书 | 多模态训练加速方法汇总 |
| 通用加速特性说明书 | LLM 通用加速方法 (VPP、通信掩盖等) |

</details>

<details>
<summary><b>📄 模板 (3篇)</b></summary>

| 文档 | 一句话说明 |
|------|-----------|
| XXX模型训练说明书模板 | 编写模型训练说明书的模板 |
| XXX特性设计说明书模板 | 编写特性设计说明书的模板 |
| 训练系统架构设计说明书模板 | 编写系统架构设计说明书的模板 |

</details>

---

## Q10：基于昇腾/NPU的差异化优化总结

### 一、通信优化

| # | 优化项 | 创新性 | 硬件 | NPU做了什么 |
|---|--------|--------|------|-----------|
| 1 | **TP MC2 通算融合** (`use_ascend_mc2`) | 中等 | NPU专有，GPU不可复用（依赖`npu_all_gather_base_mm`/`npu_mm_reduce_scatter_base`） | Linear matmul 与 all-gather/all-reduce 融合为单算子，计算与通信重叠执行 |
| 2 | **EP MC2 通算融合** (`ep_mc2`) | **高** | NPU专有，GPU不可复用 | GMM 与 AlltoAll 融合（`npu_alltoallv_gmm`/`npu_gmm_alltoallv`），Permute+AlltoAll+GMM 三合一 |
| 3 | **1F1B 通信掩盖** | 工程改进 | GPU通用（基于Megatron-LM + 深度定制），设计思路GPU可复用 | 两层调度架构：调度编排层 + 执行引擎层；MoE 层算子级 fwd/bwd 交错；EP all-to-all 与反向计算重叠；PP P2P 通信掩盖 |
| 4 | **DP 通信掩盖** | 工程改进 | GPU通用（参考Megatron-LM），设计可复用 | DP 梯度 reduce-scatter / 参数 all-gather 与计算重叠 |
| 5 | **FSDP2 通信掩盖 + HCCL 双 Communicator** | **高** | NPU专有，GPU不可复用 | 反向 reduce-scatter 流水深度可配；HCCL 双 communicator 使 RS 与 AG 并行，互不阻塞 |
| 6 | **分级 EP 通信** | 中等 | GPU通用（参考DeepSeek-V2/V3思路），设计思路可复用 | 跨节点 EP all-to-all 拆为两层：节点内 high-bandwidth all-to-all + 节点间 gather |
| 7 | **P2P 通信融合** | 工程改进 | GPU通用（基于PyTorch `batch_isend_irecv`），可复用 | 将 PP 多次 isend/irecv 合并为一次 batch 发送 |
| 8 | **PP 通信 Shape 重构** | 工程改进 | NPU专有（MHC/MTP Shape 相关性高），GPU不可直接复用 | 缓存 tensor shape 避免 PP 间冗余 shape 通信，含 static/variable 两级 cache |
| 9 | **PP 自定义负载划分** | 工程改进 | GPU通用（参考Megatron-LM），可复用 | 允许手动调整各 PP stage 层数分配 |
| 10 | **Group Collective Gather/Scatter** | 工程改进 | NPU专有（依赖HCCL原生操作），GPU不可复用 | 封装等长/变长 Gather/Scatter 通信接口，支持原生 HCCL 和 batch P2P 双路径 |

### 二、计算融合

| # | 优化项 | 创新性 | 硬件 | NPU做了什么 |
|---|--------|--------|------|-----------|
| 1 | **SwiGLU 融合算子** | 低 | NPU专有（`npu_swiglu`），GPU不可复用 | SiLU + 门控 + 乘法合并为单 kernel |
| 2 | **RMSNorm 融合算子** | 低 | NPU专有（`npu_rms_norm`），GPU不可复用 | LayerNorm 多算子融合为单 kernel |
| 3 | **RoPE 融合算子** | 低 | NPU专有（`npu_rotary_position_embedding`），GPU不可复用 | cos/sin 计算 + 旋转变换合并为单 kernel |
| 4 | **RoPE 缓存加速** | 工程改进 | NPU专有，GPU不可复用 | RoPE 位置编码的计算结果缓存，避免重复计算 |
| 5 | **Permute 融合算子** | 中等 | NPU专有，GPU不可复用 | MoE token dispatch 的 permute/unpermute 融合 |
| 6 | **ModAttn 融合算子** | **高** | NPU专有，GPU不可复用 | MLA 场景滑动窗口聚合的 NPU 自定义算子，替代 PyTorch 卷积 |
| 7 | **MHC 融合算子** | **高** | NPU专有（`mhc_ascendc`），GPU不可复用 | Manifold-Constrained Hyper-Connections 的 NPU 融合实现 |
| 8 | **Param Sink 融合算子** (`sink_fa`) | **高** | NPU专有，GPU不可复用 | Attention 固定 sink token 的融合 FA，含 rescale 加权 |
| 9 | **GMM 梯度累加融合** | 中等 | NPU专有，GPU不可复用 | Grouped GEMM (MoE) 反向梯度累加融合为单算子 |
| 10 | **Matmul 梯度累加融合** | 中等 | NPU专有，GPU不可复用 | 普通矩阵乘法梯度累加融合 |
| 11 | **Attention Mask 压缩** | 工程改进 | NPU专有，GPU不可复用 | Flash Attention mask 的压缩表示，减少传输量 |
| 12 | **Attention Mask 快速构造** | 工程改进 | NPU专有，GPU不可复用 | 多模态场景下快速重置 attention mask |
| 13 | **FA Rescale 加权实现 Sink** | 工程改进 | NPU专有，GPU不可复用 | Flash Attention 结果 rescale 加权实现 sink 效果 |
| 14 | **Core Attention 重计算** | 工程改进 | GPU通用，可复用 | Attention 部分重计算省显存（选择性重计算） |
| 15 | **动态专家迁移** | **高** | NPU专有，GPU不可复用 | 运行时根据专家负载动态迁移到不同设备（fine-grained, threshold=0.08） |
| 16 | **QAT 融合算子** | 中等 | NPU专有，GPU不可复用 | INT4/INT8 量化感知训练的融合算子 |
| 17 | **Encoder DP 重构（DP+CP）** | 工程改进 | NPU专有，GPU不可复用 | 多模态 encoder 的 DP + CP 并行重构 |
| 18 | **TP 域数据 CPU 内存共享** | 工程改进 | NPU专有，GPU不可复用 | TP 组内 CPU 侧共享数据 buffer |
| 19 | **优化器 buffer 对齐** | 工程改进 | NPU专有，GPU不可复用 | optimizer buffer 对齐以提升计算效率 |
| 20 | **Flash Attention** | 低 | GPU通用，可复用 | 标准 flash_attention 融合算子 |

### 三、MoE 专项

| # | 优化项 | 创新性 | 硬件 | NPU做了什么 |
|---|--------|--------|------|-----------|
| 1 | **pangu_dropless Token Dispatcher** | 中等 | GPU通用，设计可复用 | 不丢弃任何 token 的 dropless 分发策略 |
| 2 | **Expert Placement（细粒度动态迁移）** | **高** | GPU通用，设计可复用 | 运行时根据负载偏差（阈值8%）将 expert 迁移到合适卡上 |
| 3 | **EP MC2（AlltoAll+GMM 融合）** | **高** | NPU专有，GPU不可复用 | 见通信优化#2 |
| 4 | **分层 EP 通信** | 中等 | GPU通用（参考DeepSeek），设计可复用 | 见通信优化#6 |
| 5 | **RouterReplay** | 中等 | GPU通用，设计可复用 | 路由历史重放，缓解负载不均衡 |

### 四、训练/其他差异化

| # | 优化项 | 创新性 | 硬件 | NPU做了什么 |
|---|--------|--------|------|-----------|
| 1 | **异构调度（Hetero Schedule）** | **高** | GPU通用，设计可复用 | 多模态 encoder 与 LLM decoder 解耦，各自独立 MBS/并行策略/bridge 梯度 |
| 2 | **AIturbo 异步 Checkpoint** | 工程改进 | NPU专有（华为内部工具），GPU不可复用 | 基于昇腾 AIturbo 的异步保存，不阻塞训练 |
| 3 | **Hyper_DCP 分布式 Checkpoint** | 中等 | GPU通用，设计可复用 | 大规模分布式 checkpoint 的 DCP 方案，支持 MTP head 增减 |
| 4 | **Step 级快速恢复** | **高** | GPU通用，设计可复用 | ECC 场景下以 step 粒度快速恢复训练 |
| 5 | **Muon 优化器（NPU 适配）** | 工程改进 | GPU通用，设计可复用 | 矩阵正交化优化器，比 AdamW 省显存 |
| 6 | **Micro-DP 负载均衡域** | 中等 | GPU通用，设计可复用 | DP 组内更细粒度的负载均衡域 |
| 7 | **Hyper FSDP + MXFP8** | 中等 | NPU专有，GPU不可直接复用 | FSDP2 分布式场景下 MXFP8 低精度训练 |
| 8 | **ModAttn / MHC / MTP / Sandwich Norm 等算法** | 模型算法创新 | GPU通用，设计可复用 | 这些是模型结构创新，NPU 适配融合算子实现加速 |

### 五、汇总总结

| 分类 | 创新性强 | 已有开源基础 | GPU 可复用设计 | NPU 专用实现 |
|------|---------|------------|--------------|------------|
| **通信优化** | EP MC2、FSDP2 双 Communicator | 1F1B/DP overlap (Megatron 理念) | 1F1B、DP、PP 自定义划分、分级 EP 设计思路 | TP/EP MC2、HCCL 双 Comm、P2P Shape 重构 |
| **计算融合** | ModAttn、MHC、Param Sink、动态专家迁移 | Core Attention 重计算、FA | Core Attention 重计算 | 全部融合算子均为 NPU `npu_*` 专有 |
| **MoE** | pangu_dropless + EP MC2 | 参考 DeepSeek 分级 EP | Token Dispatcher、Expert Placement 设计 | EP MC2 融合算子 |
| **训练** | 异构调度、Step 级恢复 | -- | 异构调度、DCP、Expert Placement | AIturbo、Muon NPU 适配 |

**核心结论**：
- **NPU 创新项（不可复用）**：MC2 通算融合、FSDP2 双 HCCL Communicator、ModAttn/MHC/Permute/Sink 融合算子、动态专家迁移、AIturbo
- **开源基座**：Megatron-LM（NVIDIA 开源）、MindSpeed（华为开源）、PyTorch
- **GPU可复用设计**：1F1B/DP 通信掩盖方案、分级 EP 设计、PP 负载划分、pangu_dropless、Expert Placement、异构调度、DCP Checkpoint 方案、Muon 优化器、融合算子**设计理念**（但具体算子不可复用）
- **NPU 优化重点**：通信与计算重叠（MC2）、跨节点 EP 通信分层、MoE 融合算子链、长序列 Memory/Attention 优化
