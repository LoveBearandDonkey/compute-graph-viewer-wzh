# DeepSeek V4 Mapping Explorer — CSA 代码血缘规格

- 关联 Task ID：`PTO-MAPPING-L1-L2-REVERSIBLE-ZOOM-003`
- 基础 Task ID：`PTO-MAPPING-FOUR-LEVEL-FOUNDATION-002`
- 规格修订：2026-09-07 · lineage-006 · CSA 整容器选中、L2 基准的 L3 默认展开、L0 整网概览
- 状态：lineage-006 修正三项交互与层级定义；本轮范围及验证见 §12.3。L4 最终形式与非 CSA 整网参数校准仍未完成。
- 页面入口：[workbench.html](workbench.html)
- 架构事实校验：[model_architecture_validation.md](model_architecture_validation.md)
- 本次交付：修复展开／折叠 CSA 选中入口，将 L3 默认展开至 L2 对应实现步骤，使用保留整网主干的 L0 替代单节点根视图；沿用三栏及既有节点、容器、连线和文件 tab。

## 1. 产品目标与范围

让开发者沿着同一条代码血缘，理解模型的一段计算如何从官方开源代码落实为 PyPTO 程序，再经过编译生成 Kernel：

```text
官方模型源码
  → 计算语义
  → 开发者编写的 PyPTO 实现
  → PyPTO 编译 Pass
  → Kernel / Hardware
```

每次下钻都应回答：这段计算来自哪里、输入输出是什么、经过了什么变换、为什么发生变换、由谁引入、证据在哪里。节点连续性、变换原因和证据归属是核心验收要求。

本任务以 CSA 为唯一纵向样板。官方整网图是左栏持续使用的架构载体，不是进入独立 CSA 图之前的临时上下文。保留整网原有的多级父子展开与折叠，将 CSA 的计算语义接入对应节点。不得扩展 MoE、HCA、SWA 的实现或编译映射。保留三栏布局和现有设计系统、图节点、连线、布局及缩放能力。

当前样例中的 `decode_csa.py` 是开发者编写的 PyPTO DSL 实现。现有证据不支持把官方 PyTorch `model.py` 到该文件的关系标为自动编译。界面必须区分“源码解析”“开发者实现映射”和“编译器变换”。

## 2. 三栏职责与层级导航

架构图与 PyPTO 计算／实现图分开查看，以共同的 CSA 模块作为衔接点。两图的源码归属、数据依赖和 scope 各自明确，通过映射关系连接，不将它们拼成一棵虚假的实现继承树。

本文“右侧图／右侧视图”指三栏中的中栏 PyPTO 图；“源码窗”指最右栏。

| 区域 | 内容 | 默认与导航 |
| --- | --- | --- |
| 左栏：开源模型架构图 | 官方整网的原有层次；CSA 内的 L1 摘要与 L2 计算语义 | 默认 L1；在整网对应节点内 expand / fold，不切换独立 CSA 图 |
| 中栏：PyPTO 计算／实现图 | CSA 的 L3 程序实现、后续 L4 编译血缘 | 由“查看 PyPTO 实现”进入 L3；进入后支持内部展开及 L3/L4 导航 |
| 右栏：证据源码窗 | 同时保留官方与 PyPTO 两个源码 tab；后续可查看编译证据 | 当前操作激活对应 tab，不关闭另一份源码；不作为第五层 |

### 2.1 左栏：整网内展开，不切换架构载体

原有整网已经支持模型、Decoder、Attention、CSA 等多级父子展开。CSA 的 L1/L2 必须接入这张 canonical hierarchical graph：例如在整网内展开 Query Projection，其 L2 子节点出现在原父节点容器中，CSA、Attention、Decoder 等祖先与周围节点继续保留。

不得保留“整网上下文 / 进入 CSA”作为两套独立图的切换。聚焦或适配 CSA 只改变观察范围，不卸载整网、不改变节点身份，也不要求用户返回模型根 scope 才能恢复祖先上下文。

左栏 dropdown 工作命名为“开源模型层级”：

- `L0 整网概览`：保留 Input Tokens、Token Embedding、前序 Decoder、当前 Decoder 残差主干、后续 Decoder、Final RMSNorm、LM Head 和 Output Tokens；mHC-Attention 与 mHC-FFN 为可展开的折叠模块。
- `L1 模型架构`：默认展开深度。
- `L2 计算语义`：在整网中的当前 CSA scope 批量展开至已有官方证据支持的默认语义深度；不展开其他模块的新语义。

L0 按用户提供的整网主干截图定义，是开源模型的概览层，不是新增编译证据层。删除“模型根节点／整图变成单个孤立节点”的选项；旧 modelLevel=root 深链兼容恢复为 L0。L0 沿用同一 canonical 整网的节点 ID 与折叠投影，不加载另一套图。Decoder 层号与 L0/L1/L2 抽象层级分开表达。

dropdown 是批量控制当前适用 scope 展开深度的快捷操作，不能取代节点 expand / fold，也不能加载另一套图。左栏状态显示 `L0 · 整网概览`、`L1 · 模型架构`、`L2 · 局部展开` 或 `L2 · 完全展开`；L2 状态描述本轮支持的 CSA 语义范围，不代表整网所有模块已展开到 L2。

### 2.2 跨图衔接：查看 PyPTO 实现

1. 用户在左栏整网选中 CSA 模块，出现明确入口 **“查看 PyPTO 实现”**。CSA 为折叠或展开状态均可选中，不要求先展开全部 L2。展开时标题、边框及没有子节点覆盖的内部空白都属于 CSA 的点击热区；折叠时摘要主体可选中。加减按钮只折叠／展开，不能成为唯一可点击区域。
2. 仅选中 CSA 或展开其 L2 不自动打开、替换中栏实现图。
3. 点击入口，中栏进入该 CSA 对应的 **L3 · PyPTO 实现**，以共同 CSA 模块和适用上下文衔接；展示实际 `decode_csa.py` 实现范围，包括大于官方 CSA scope 的部分。
4. 左栏保留展开状态、CSA 选中状态、缩放与平移；中栏明确当前实现来自左栏哪个 CSA scope，并提供对应源码证据。
5. 进入后两图支持节点级多对多映射联动。L3 内部的函数／scope 可在中栏继续 expand / fold。

**L2 → L3 不使用左栏 L2 叶子节点的 expand。** 同一源码图内细化使用 expand / fold；跨源码实现使用模块级“查看 PyPTO 实现”。前序讨论中的“从 L2 节点继续点击 expand 进入 L3”已被本约定替代。

中栏 dropdown 工作命名为“PyPTO 实现层级”，进入实现后可选择 `L3 PyPTO 程序实现`、`L4 编译与 Kernel`，但它不能取代从 CSA 模块建立映射上下文的入口。首次未进入实现时应说明需从左侧选择 CSA 查看实现，不默认显示没有来源上下文的 L3；恢复已保存或深链状态时须同时恢复来源 CSA 与实现上下文。

每栏只保留一套层级控制，避免 dropdown、分段按钮和第二排全局操作重复。入口按钮、层级菜单与节点折叠控件职责不同，不应互相替代。

## 3. L1：官方模型架构

CSA 的摘要 compound node 保留以下语义分组：

- Query Projection
- Recent-window KV · 128
- Compressed KV Path · ratio 4
- Lightning Indexer
- Sparse Shared-KV Attention
- Grouped Output Projection

`ratio 4` 表示压缩比例，不表示四个 Compressor 实例；不得使用 `KV Compressor ×4`。近期窗口不命名为独立 SWA 模块。

摘要依赖必须来自官方代码：RMSNorm 后的 hidden state 分别进入 Query、Recent-window KV、Compressed KV 和 Lightning Indexer 路径；Query 路径的归一化低秩特征供 Indexer 使用，主 Q 供 attention 使用；近期 KV、压缩 KV 和选择索引供 sparse attention 使用；attention 输出进入 grouped output projection。

不得出现 Query → Main Compressor、Shared KV Projection → Main Compressor、Main Compressor → Indexer Compressor、Main Compressor → Recent-window KV 的错误串联。

## 4. L2：由官方源码独立建立的计算语义

### 4.1 数据来源与解析边界

L2 的输入为官方源码及依赖、官方配置、选定 layer、phase 和必要的输入/分支条件。L2 不依赖开发者的 PyPTO CSA 实现补写官方计算关系。

`model.py + config` 可支持模块调用、张量表达式、归一化、投影、RoPE、压缩 pooling、检索评分、Top-K、索引组合、输出投影等语义。展开深度以可取得的函数定义为限：外部调用没有定义时保留为可追溯的不透明叶子，不凭名称推断内部运算。

必须区分：

- 源码明确表达的计算、数据依赖和逻辑状态。
- 由配置或运行条件特化出的分支、shape、dtype；未固定的条件保留为条件或符号。
- 为阅读而建立的语义名称、分组和摘要布局；这是语义整理，不等于官方类名，也不等于自动解析产物。

现有 L2 是人工整理且带源码引用的语义图，尚不能宣称由通用解析器自动生成。固定“22 个节点”不是产品目标或验收指标。

CSA 默认研究 decode 路径；需保留压缩触发条件，不能把仅特定 token 执行的步骤标成每次无条件执行。prefill 证据不得无标识地与 decode 实现比较。

官方 `kernel.py` 可补充外部调用的参考算法证据，但其中 TileLang 的线程、shared memory、tile 和流水策略属于官方参考实现。不得将其当作 PyPTO 编译结果或通用 L2 语义。

### 4.2 默认语义分组

| L1 分组 | L2 默认展示内容；操作、值和状态须分别建模 |
| --- | --- |
| Query Projection | Q LoRA/down、Q normalization、Main Q projection/scaling/RoPE、Indexer Q projection/RoPE/rotation |
| Recent-window KV · 128 | Shared KV Projection、KV normalization、RoPE；近期 KV 值与窗口索引；逻辑 KV 状态引用 |
| Compressed KV Path · ratio 4 | Main Compressor；按证据可继续展开 projection/gating/pooling/norm/RoPE；压缩 KV 值与持久状态 |
| Lightning Indexer | Indexer Compressor、索引 KV 值/状态、Query–Index KV score、Score weighting、Top-K 选择及索引输出 |
| Sparse Shared-KV Attention | 引用近期窗口索引和选中压缩索引；Candidate/index union；Sparse Attention 调用及输出逆 RoPE 语义 |
| Grouped Output Projection | Group reshape、Grouped low-rank projection、Output projection |

Indexer Compressor 属于 Lightning Indexer，不能作为 Main Compressor 的下游。Indexer Q 的视觉分组与实际定义所在的 `Indexer.forward` 分别记录，不把视觉归属当作源码 ownership。

`Main compressed KV representation`、`Top-K compressed positions` 等名称可能表示值或状态，不能统一标为 Compute。窗口索引在多个分组中被使用时，共用 canonical ID；必要的视觉引用不新增计算实体。

仅从 `model.py` 的调用级证据出发，不把独立 `Selected KV Gather` 宣称为官方独立算子。若进一步展开官方 kernel 中的语义选择，必须标明证据来源和抽象边界；PyPTO 的物理 gather 由 L3 描述。

### 4.3 纯计算视图与逻辑状态

L2 可显示官方源码已有的 Logical State，例如 `Attention.kv_cache`、Compressor 的 `kv_state/score_state`、Indexer 的 `kv_cache`，并表达 read/write 更新关系。这里展示逻辑内容与生命周期，不展示分页地址、物理 KV Pool、Block Table、Slot Mapping 或 rank ownership。

官方源码已有的分布式通信不能在证据里抹去。L2 默认按计算语义投影，隐藏部署细节；原始源码图仍保留通信事实，隐藏项在映射说明中可追溯。不能因 L2 隐藏通信就称 L3 的所有通信均为新增。

## 5. 实体视觉语法与连线

| 实体 | 约定视觉 | 含义 |
| --- | --- | --- |
| Semantic Op | 圆角胶囊 | 计算或数据变换 |
| Tensor / Value | 灰色圆角矩形、实线 border | 输入、输出或中间值 |
| Logical State | 灰色圆角矩形、虚线 border | 跨 token/调用保存、读取和更新的逻辑状态 |
| Scope | 透明内部、灰色 border 包围框 | 组织、语义或实现边界 |

上述 Logical State 与 Scope 样式为用户已明确选择。Runtime metadata 可沿用灰色矩形并在详情说明角色；不能仅因它是函数参数或名为 mapping，就将其标为持久 Logical State。应依据生命周期判断。

Scope 展开时不是运算实体，不计入算子数；边通过边界 port 连接内部实体。Scope 收起后投影为 compound 摘要，保留折叠控件以区别实际算子。边界使用灰色，选择状态不使内部节点全部去色。

Scope 外上下文节点用低透明度文字和实线边框表达，以免与 State 的虚线混淆。State 可在详情展示生命周期；额外图标或副标题属于可选设计，尚非必需。

数据边区分 produce、consume、state read、state write/update；读写使用独立方向，不以无解释的双向箭头代替。控制依赖使用虚线并有图例。血缘映射与执行数据边必须有不同语义和显示开关，不能把 mapsTo 画成执行顺序。

节点计数分别展示 Op、Value、State；Scope 单列或不计入。视觉别名、父容器、kernel 定义数与运行时 task 实例数不能混算。

## 6. 可逆展开与持续联动

官方整网及其 CSA L1/L2 从同一 canonical hierarchical graph 投影，满足 `collapse(L2) = L1`：ID、名称、摘要拓扑、输入输出关系、scope、节点尺寸和相对布局均可恢复。这里恢复的是整网内对应 CSA 的摘要状态，不是切换到独立摘要页面。

- 左栏支持全局展开、全部收起、单 compound 展开与混合状态；最后一个 compound 收起后自动回到 L1。
- 整网原有祖先节点继续支持 expand / fold；折叠某祖先时隐藏其后代，重新展开时保留已有局部展开记录。CSA 的局部操作不得重置其他模块的展开状态。
- 展开时父节点保留为 container；收起隐藏后代并将跨界边重定向至 port，同方向摘要边去重，必要时显示数量。
- 保持展开节点的横向中心、上边缘与周边相对位置，仅为新增内容让出必要空间；并列节点保留所在行的相对布局。
- 标题、展开按钮和内部节点有明确间距。按钮位于主体内侧且垂直居中；不显示 `L1 compound node` 副标题。
- CSA 外框具有充分内边距；节点不得压住边框。沿用统一尺寸的紧凑箭头。
- 左、中画布独立缩放、平移、适配 scope，并复用三栏宽度调整能力。

在通过 CSA 入口打开对应实现后，左栏点击 L2 节点，中栏高亮所有相关 L3 实体；中栏点击 L3 节点，左栏高亮全部源语义实体。多对多映射不能只取第一个结果。源实体尚未展开时高亮所属 compound 并提示命中数量，可由用户开启自动展开。未进入实现时，点击 L2 节点仅选择语义实体并定位官方源码，不隐式触发跨图导航。

左右选择不得重置另一栏层级或整图布局。源码面板跟随最后选择的实体或映射关系。L3/L4 切换保留语义焦点、所选实现和编译上下文；旧画布迟到事件不得覆盖新状态。

这里的“连续”包括两种明确关系：左栏整网内的结构父子展开，以及从共同 CSA 模块到中栏实现的跨源码血缘衔接。两图分开查看，但来源、选择和变换关系不断开。L3 必须以 L2 语义对应为默认阅读基准：每个支持的 L2 实体进入 L3 后都有可见的实现对应或明确的缺失说明，不能只显示更粗的函数摘要而把细化要求列为后续待办。多对多分组与实际共享实现保留唯一 ID，不为增加节点数复制计算。

## 7. L3：PyPTO 程序实现与语义变换

### 7.1 覆盖范围

L3 描述 `decode_csa.py` 及实际调用的 PyPTO 子程序，至少覆盖 HC Pre、Attention RMSNorm、CP Token All-Gather、Q/KV Projection、原始/压缩/索引 KV Cache、Compressor State、Slot Mapping/Block Table、Indexer/Top-K/KV 访问、Sparse Attention、Block Merge、Head-group Redistribution、Attention All-to-All、Grouped Output Projection、Output Reduce-Scatter 和 HC Post。

函数调用、内部计算步骤和运行时资源需采用一致、可展开的展示粒度。首次进入 L3 默认展开 Q、KV、主 Compressor、Indexer Compressor、Lightning Indexer、Exact Top-K、Sparse Attention、Block Merge、Grouped Output Projection，直接展示 L2 对应的 PyPTO 实现步骤及实现增量。每个程序可以独立 fold / expand，层级菜单提供恢复默认展开和收起步骤；这些控件不影响左图层级、展开或 viewport。一个函数不能默认视作一个 kernel；节点数不是完成指标，L2 语义覆盖和可追踪细化才是。

### 7.2 变换类型

变换属于源实体与目标实体之间的关系，可同时具有多种标签，并记录引入者及证据。

| 类型 | 含义 | CSA 应说明的变化 |
| --- | --- | --- |
| PRESERVED | 保留官方计算语义 | 投影、归一化、压缩等语义对应 |
| GROUPED / FUSED | 多个源步骤由共同实现承载 | `q_proj_rope` 承载多个语义步骤；函数级分组先标 GROUPED，证明实际融合后才标 FUSED 并注明层次 |
| SPLIT / EXPLICIT | 一个源步骤被分解或内部步骤外显 | sparse attention 的 KV 访问、分块统计与合并 |
| MATERIALIZED | 逻辑值/状态被落实为显式资源 | 模块内部 cache/state → 显式参数、分页存储、scale 与寻址元数据 |
| DEPLOYED | 分布式所有权和通信方案 | Token All-Gather、head-group publish、All-to-All、Reduce-Scatter |
| SCOPE_INCLUDED | 扩大实现范围，纳入已有模型上下文 | HC Pre、RMSNorm、HC Post 纳入 `decode_csa.py` |

使用 `changeKind` 补充 added、replaced、preserved、grouped、split 等事实；“新增”“替换”均相对于明确的源 scope、phase 和部署条件。名称相近、位于同一函数或共有输入都不足以证明融合或数值等价。

### 7.3 必须解释的实现增量

- 缓存与状态：官方已有逻辑 cache 和 compressor state。L3 展示分页表示、显式参数、寻址及生命周期，不能声称缓存概念由 PyPTO 首次引入。
- Slot Mapping / Block Table：展示实际读写代码中的逻辑位置到物理存储映射；区分元数据与计算该元数据的程序，输入参数不意味着由 `decode_csa` 内部生成。
- Token All-Gather：说明本地 token 输入与 KV/Compressor 所需 token stream 的范围差异，以及该部署条件下的通信需求。
- Head-group 通信：说明 attention 输出的数据拥有者、输出投影拥有者、跨 rank 搬运与归约目的；notify/wait 属于可见实现细节。
- 官方 all-reduce 与 PyPTO Reduce-Scatter：必须对照具体并行布局核实对应关系；不能只凭名称宣称二者直接替换或等价。
- 精度与布局：量化、scale、cast、reshape 等存在时标明来源；区分官方已有操作与 PyPTO 选用的数值/布局策略，尚未验证的数值等价不得标 verified。

L3 同时显示灰色边界框 `CSA model semantic scope` 与 `decode_csa.py implementation scope`。HC Pre、Attention RMSNorm、HC Post 在官方整层代码中已经存在；纳入 PyPTO operator 是实现 scope 扩大，不是增加模型数学语义。

## 8. Mapping Diff

默认关闭 Diff，沿用功能色和实体形状约定；Tensor/State 默认保持灰色。开启后对相关 L3 实体整体染色，保留形状和 State 虚线，并显示完整中文解释与英文标签。不得恢复以节点左侧蓝黄绿竖线为主要映射入口的方式。

| 颜色组 | 复用令牌 | 图例文字 |
| --- | --- | --- |
| 绿色 | `#168a52` | PRESERVED：保留官方源码所表达的计算语义。 |
| 蓝色 | `#2563eb` | GROUPED / FUSED / SPLIT / EXPLICIT：语义步骤在 PyPTO 中被组合承载、融合、拆解或显式展开；详情注明具体变化与证据。 |
| 黄色 | `#d68a00` | MATERIALIZED：官方逻辑值或状态被落实为缓存、显式参数及 Block Table、Slot Mapping 等寻址资源。 |
| 紫色 | `#7c3aed` | DEPLOYED：由于分布式数据所有权与部署方案引入或改写的通信、搬运和同步步骤。 |

`SCOPE_INCLUDED` 通过灰色 scope 框及范围说明表达，不新增填充色。多标签关系不得因单色而丢失；详情显示全部标签，颜色仅表示当前查看的主变换。

提供“实现增量”过滤能力，突出通信、资源物理化、元数据、同步和范围扩大的部分，同时保留必要语义锚点。区分“原有语义外显”与“新增实现机制”。切至 L4 关闭 L3 Diff；L4 的 Pass Diff 是独立功能。

旧字段 `SAME / EXPANDED / STATE / DEPLOY` 为迁移兼容项。不得机械改名：STATE 是实体类别与旧差异类别的混用；SAME 不保证一对一；每条关系应重新核查。

## 9. 源码证据与解释性 # 注释

### 9.1 源码联动

源码窗以真实文件为 tab，同时保留两个已打开的源码 tab：

- `官方 · model.py`：左栏模型架构与 L2 语义的官方证据。
- `PyPTO · decode_csa.py`：中栏 CSA 实现的调用及实现范围证据。

“同时打开”指两个文件 tab 持续存在、可直接切换，不要求将源码窗再分成两栏，也不新增第四栏。未进入 PyPTO 图时可保留其源码 tab，但不能因此自动加载 L3 图；没有有效引用时明确提示，不展示伪造的代码定位。

联动规则：

1. 点击左图节点，激活官方源码 tab，定位对应声明、调用或表达式。
2. 点击“查看 PyPTO 实现”，打开对应 L3 图并激活 `PyPTO · decode_csa.py`，保留官方 tab 的原位置。
3. 点击中图节点，激活对应 PyPTO 源码 tab，默认定位该节点调用位置；没有调用位置时使用实际可用的声明／定义证据并说明。
4. “调用位置 / 实现定义”改为当前实体的源码导航操作，不再作为源码窗的两枚主页签。跳转定义涉及 `qkv_proj_rope.py` 等叶子文件时打开或复用对应文件 tab，不替换、关闭两份基础源码；同一文件不重复开 tab。
5. 每个文件 tab 独立保存滚动位置、定位范围与选择状态。只切换 tab 不重置图的选择、展开和布局；图节点选择可更新所属文件的定位，但不能覆盖另一份源码的状态。
6. 选择映射关系时可分别查看源端和目标端证据；多文件、多表达式继续通过引用列表导航，不能只展示第一个引用。

每个 tab 明确官方／开发者实现／编译产物归属。后续 IR、Pass Diff、Generated Code 属于编译证据模式，不能覆盖或混淆这两个源码归属入口，其最终形式仍按 L4 单独验收。

一个语义节点可对应多个表达式或文件。没有独立函数定义的 Value、State 或语义步骤，应显示表达式/声明证据并解释“无独立定义”，不得伪造函数或复用无关范围。

源码引用必须绑定版本或内容 hash、路径、symbol 和行号。当前本地快照、远端 main 与历史编译快照不能默认相同；行号随版本变化时不得继续宣称命中已验证。源码引用的 verified 不等于转换正确性或编译血缘已验证。

### 9.2 注释内容与归属

在证据面板使用 `#` 形式呈现解释性注释，明确标注“Explorer 注释”，与原始作者注释区分；由 registry 的结构化解释生成，不改写上游源码文件。注释至少回答：

- 对应哪些源/目标实体，属于哪种变化。
- 原始计算或逻辑状态是什么，具体改变了什么。
- 为什么需要这一实现，条件和输入输出范围是什么。
- 来自官方源码、开发者实现还是编译器 Pass。
- 哪些证据支持，哪些结论仍未验证。

以下为注释模板；实际展示前按选定快照、shape 和部署配置校验：

```python
# [Explorer · DEPLOYED / added] CP Token All-Gather
# 来源：RMSNorm 输出供 KV 与 Compressor 路径使用。
# 原因：本 rank 仅持有 local token rows；当前部署的 KV 路径需要组内 token stream。
# 变化：本地 x_normed_t → gathered x_normed_full；具体范围见当前配置。
# 引入者：开发者编写的 decode_csa / token-allgather 程序。
# 证据：链接到调用位置、通信实现与 tensor ownership 定义。

# [Explorer · MATERIALIZED] KV Cache 与寻址元数据
# 官方：Attention.kv_cache 表达跨 decode step 保留的逻辑 KV 状态。
# PyPTO：使用显式 cache 参数，按 ori_slot_mapping 写入物理行。
# 增量：存储表示和寻址机制；KV 历史这一逻辑语义原本已存在。
# Block Table / Slot Mapping 的生成者与读写位置分别链接。

# [Explorer · GROUPED] q_proj_rope
# 来源：Q LoRA/down、Q normalization、Main Q projection/RoPE 等语义步骤。
# 变化：多个步骤在同一 PyPTO 函数中承载，内部仍可继续展开。
# 说明：函数边界不证明 kernel 融合；具体 kernel 划分须追踪编译产物。

# [Explorer · SCOPE_INCLUDED] HC Post
# 官方位置：Block.forward 中，Attention 调用之后。
# 变化：decode_csa.py 将该步骤纳入实现边界，返回更新后的 HC 状态。
# 说明：它是已有模型上下文，不是新增数学运算。
```

All-to-All、Reduce-Scatter、Compressor State、Block Table 等同样使用这一结构；不能只以一句“新增通信”或重复节点名称代替解释。

## 10. L4：编译血缘与 Kernel 证据

### 10.1 已确定原则

从当前 L3 程序/语义焦点出发，追踪该次编译的 Frontend IR、实际 Pass 序列、生成 kernel 和可取得的硬件证据。首个样例优先选择 `qkv_proj_rope.py` 中具有匹配编译快照的程序。

每步变换需说明新增、删除、组合、拆分、tile 化、内存推导或依赖变化，并可回到前一版本。Pass 名称用于导航；声称某节点经过某变换必须有前后 IR 或映射证据。

允许展示已验证 Tile Shape、Layout、Memory Space、kernel 类型、源码位置和任务依赖。编译配置里的 `func_id`/kernel 定义与运行时 task 实例、物理 core id 分开记录。Core 类型不等于具体 core 分配；实际流水排布需要对应编译或运行证据。

Source、IR、Pass Diff、Generated Code 是证据模式，不是额外模型层级。官方 TileLang kernel 是参考实现证据分支，不能作为 PyPTO L4 产物。

没有可匹配、已验证产物时显示：

> 暂无已验证的 Kernel 映射，请先编译并导入 Pass Dump

即使目录中存在 Pass Dump，若其源码/配置版本与当前 L3 不匹配，也不得自动建立 verified 血缘。禁止根据常识或函数名称推测 AIC、AIV、UB、Tile、Core 或流水排布。

### 10.2 待设计部分

L4 最终交互形式尚未确定。Pass 时间轴、相邻 Pass 图 Diff、split/fuse 追踪及生成 kernel 定位是候选方向。现有 `Tensor Graph → Tile Graph → Block Graph → Execution Graph` 四卡展示不再作为最终形式或强制阶段顺序；应尊重真实编译序列。

现有静态摘要和写死的 Pass Diff 文字不等于从产物解析出的结构差异。L4 完整改造需单独设计与验收，本次文档更新不视为该形式获批或已实现。

## 11. 统一数据与证据契约

统一 registry 管理官方 canonical graph、PyPTO 实现图、编译阶段图及它们之间的血缘关系。统一来源不意味着把多对多血缘硬塞进单一父子树。

节点保留既有兼容字段：`id`、`label`、`level`、`semanticLevel`、`type`、`nodeType`、`parentId`、`parentScope`、`children`、`inputPorts`、`outputPorts`、`summaryEdges`、`expandedEdges`、`defaultExpanded`、`applicableScope`、`mapsTo`、`provenance`、`sourceRefs`、`phase`、`applicableLayers`、`artifactStatus`。

新增或明确字段职责：

| 对象/字段 | 要求 |
| --- | --- |
| node.entityKind | Op、Value、LogicalState、RuntimeMetadata、Scope；功能分类独立记录 |
| node.origin | official-source、developer-pypto、compiler-generated；不以层级数字代替证据归属 |
| node.lifecycle | 对 state/value 记录持续范围与可变性；未知时明确未知 |
| node.canonicalId / viewRef | 视觉引用回到唯一实体，不重复统计或复制计算 |
| sourceRefs | artifact/version/hash、file、symbol、start/end、role；跨版本不复用行号结论 |
| graph.context | scope、layer、phase、配置、分支条件和输入约束 |
| implementation entry | sourceScopeId、targetScopeId、relationIds、sourceContext、targetContext、artifactStatus；驱动 CSA 的“查看 PyPTO 实现”，不可凭同名节点认定对应 |
| graph view state | 左栏祖先与局部展开集合、选择、viewport；中栏进入状态、来源 CSA、实现上下文及独立展开集合；切图不得覆盖对侧状态 |
| source tab state | origin、artifact/version/hash、file 作为文件身份；分别记录 active tab、定位范围、scroll、selection；symbol/行号是定位，不是新文件 tab 身份 |
| lineage relation | id、sourceIds[]、targetIds[]、transformTypes[]、changeKind、introducedBy、scope、conditions、evidenceRefs、artifactStatus |
| relation.explanation | originalMeaning、change、rationale、inputOutputChange、limitations；用于生成源码 # 注释 |
| compilation context | compileRunId、源码/配置快照、后端、Pass 序号、前后 artifact、kernel/运行 trace 标识 |

`mapsTo` 可保留为查询索引，但不足以独立表达关系类型、变换理由或验证状态。节点映射不能通过单个点击的硬编码逻辑维护。

`artifactStatus` 支持：

- `verified`：该具体声明有匹配版本的可追溯证据。
- `inferred`：推导或候选对应，界面说明推导依据与限制。
- `unavailable`：缺少证据或无法建立当前上下文的对应，不填入猜测值。

状态按节点、关系、编译字段分别记录，不以某个目录存在产物为整条链统一赋值 verified。源码可定位、数值等价、生成对应和运行行为是不同验证声明。

## 12. 实现现状与迁移清单

下表保留 `lineage-004` 基线与修订目标的差距，作为迁移记录；本轮 `lineage-005` 的实施与验证见 §12.2，不能将历史基线当作当前状态。

| 项目 | lineage-004 基线／既有能力 | 最新目标与待迁移项 |
| --- | --- | --- |
| L1/L2 | 左栏独立 CSA L1/L2 图与外部整网上下文切换，已有局部/全局展开 | 不符合最新要求；将 CSA 语义接入整网原有节点，移除两套架构载体切换，保留祖先展开与 canonical 身份 |
| L2 实体 | 26 个 CSA 叶子：18 Op、4 Value、4 LogicalState；上下文不计入这组数字 | 手工源码整理，非通用自动解析；更深算法展开需补充官方证据 |
| CSA → L3 入口 | 中栏可独立加载 L3；已有多对多双向高亮，默认不自动展开 | 改为选中左栏 CSA 后点击“查看 PyPTO 实现”进入；不通过 L2 expand 跨图，不因普通选择自动进入 |
| L2/L3 联动 | 切换 L4 保留 L3 iframe 与焦点 | 保留来源 CSA 与两图各自状态；进入实现后复用多对多联动，缺失和候选映射明确 |
| L3 内部展开 | lineage-005 仅 Q/KV 可独立展开，未达到完整 L2 基准要求 | lineage-006 默认展开九个程序；源码范围、输入输出端口及映射逐项绑定，见 §12.3 |
| Diff | 复用原有四种颜色与 Diff 按钮，完整图例；菜单提供“突出实现增量” | 多标签在证据注释显示；筛选保留淡化的语义锚点 |
| State / Scope | 按实体类别复用灰色矩形，LogicalState 增加已约定虚线；Scope 沿用容器 | 基础颜色、字体、圆角、箭头样式不重新设计 |
| 源码面板 | 本地 SHA-256 快照、调用/定义页签、多引用菜单、Explorer 注释；异步加载防串台 | 改为官方与 PyPTO 两个持久文件 tab；调用/定义改为导航操作，叶子定义另开／复用文件 tab，各自保存位置；版本不混用 |
| L4 | 已停用写死的四卡与 Pass Diff，显示未匹配源码的空状态 | 实际编译快照导入、Pass 图、Kernel 追踪仍待单独设计验收 |
| 既有整网 | 外部模板使用 Pro 参数，已标注“待校准” | 整网是左栏主体，不再是替代入口；接入 CSA 时仍须明确未校准参数，不扩展其他模块，不将模板当作已验证 Flash 证据 |

上述现状不构成数值等价或编译产物对应的验证。registry 保留多标签关系；当前人工对照关系标为 inferred，而可定位的源码引用经实际 hash 比较后单独报告匹配。

### 12.1 持续有效的 UI 约束与既有复用记录

- 不重设现有三栏、颜色、字体、圆角、图节点、连线和主题样式；新增层级及证据入口复用现有 dropdown/按钮。
- 新增证据菜单的滚动限高复用 [shadcn/ui DropdownMenuContent 源码](https://github.com/shadcn-ui/ui/blob/main/apps/v4/registry/new-york-v4/ui/dropdown-menu.tsx) 的 max-height、overflow-x-hidden、overflow-y-auto 规则（核对：2026-09-07）；保留本页既有外观，原生 details 使用可用视口高度代替 Radix 变量。
- Q/KV 内部展开复用已有 scope/container，不新增独立视觉组件。后续新增非基础可视化组件必须先提供预览，等待用户验收后合入；尤其 L4 不在本轮默认获批。
- 最新“不改变现有 UI 样式”约束优先：既有 scope 外上下文节点暂保留低透明度虚线外观，尚未迁到 §5 的实线目标。它与 LogicalState 的视觉区分仍是待验收事项，不能视为全部视觉规范已完成。
- 既有校验入口为 [validate-lineage.mjs](validate-lineage.mjs)，检查源码 hash/范围、实体类型、映射完整性、全部 64 种 L1/L2 混合折叠状态的摘要拓扑与节点 containment，以及 4 种 Q/KV 程序展开组合的端口与包含关系。后续须补充整网祖先折叠、CSA 入口门禁、两图状态保持和文件 tab 联动测试；既有测试通过不等于本修订验收通过。
- [source-manifest.js](source-manifest.js) 绑定本地证据内容；测试发现源文件改变时必须重新核对引用与解释，再更新 manifest，不能只刷新 hash 以掩盖语义漂移。

### 12.2 lineage-005：本轮三项实施结果

- 整网只保留一个左栏图载体。适配器调用原有整网 builder，将 canonical CSA 的摘要／语义投影接入原 Attention / Hybrid Attention 父子结构，复用原折叠投影、残差连线和容器装饰。CSA 局部展开不替换整网；祖先 fold 后再 expand 保留局部子树状态。
- 选中展开或折叠的 CSA 后显示“查看 PyPTO 实现”。点击才进入中栏 L3，保留左图 viewport 和展开状态；普通 L2 展开不打开 L3。入口由 registry 的 `implementationEntries` 驱动；`implementation=csa` 深链恢复对应来源与实现上下文。
- 源码窗初始保留 `官方 · model.py` 与 `PyPTO · decode_csa.py` 两个文件 tab；文件拥有独立 DOM、滚动与定位状态。调用／定义导航复用现有证据菜单；打开 `qkv_proj_rope.py` 时新增或复用该文件 tab，不替换两份基础源码。
- 沿用现有按钮、dropdown、tab、图节点与容器外观；源码 tab 行仅增加横向溢出处理。共享渲染器只追加折叠／装饰复用接口，原有入口行为不变；可选容器标题开启鼠标命中，不改变视觉样式。
- 自动检查：源码 hash／范围、64 种语义展开状态、4 种 Q/KV 程序展开状态、320 种整网／祖先折叠组合，以及 JS／内联脚本语法。浏览器已验证鼠标选择 CSA 显示入口、L2 展开不自动进入 L3、跨图入口不改变左图 transform、祖先展开恢复局部 Query 子树、双源码位置恢复与叶子定义文件 tab。
- 验证边界：浏览器尺寸随宿主变化，已检查实际三栏布局，不宣称已完成所有窄屏断点；既有 MutationObserver 日志仍有出现，未确定来源，不能宣称 console 零错误。非 CSA 模板参数仍待校准，L4 保持真实证据门禁，其新形式未在本轮实现。

### 12.3 lineage-006：修正入口、默认细化与概览定义

本节覆盖 §12.2 的历史完成声明：此前深链直接打开 L3 不能证明正常入口可用，仅实现 Q/KV 展开也不能视为完整 L3 细化验收通过。

- CSA 展开容器的空白区域可选中；标题、边框和折叠摘要使用原有选择逻辑，折叠按钮独立。正常入口从没有 implementation=csa 的页面验证，不借助深链绕过门禁。
- L3 默认展开九个有源码证据的程序。新增 [implementation-programs.js](implementation-programs.js) 管理步骤、实际源文件／表达式范围、输入输出端口、内部依赖和条件；由 registry 统一生成实体与多对多映射。所有 26 个 L2 叶子均有默认可见的实现对应，Op、Value、State 分开计数。
- Recent Window / Selected KV Gather 属于 sparse_attn_csa 的内部步骤，沿用已有 ID，折叠 Sparse Attention 时一起收起；不在函数外再复制相同 gather。压缩器的 pooling、state commit、RMSNorm/RoPE、条件 cache write 分开表达，state commit 的依赖不会被写成 pooling 的数学输出。
- 同一行的程序展开按最高子树预留空间，保持并行兄弟的相对位置。跨越中间步骤的内部依赖走旁路，去除遮挡节点的自动 activation 标签。适配按钮适配当前所选节点／scope，未选择或 Shift+点击时适配整图，不再设会裁切整图的最小缩放。继续使用现有 Scope、节点、箭头和 Diff；默认展开不代表已验证 kernel 融合、编译分配或数值等价。
- L0 从同一整网投影，保留输入输出链与 Decoder 残差流，折叠 mHC-Attention／mHC-FFN。返回 L1/L2 或原位展开恢复已有 CSA 局部记录；旧 root 深链转为 L0。
- 自动校验增加九个程序的 512 种展开组合、全部 L2 叶子的默认可见映射，以及 L0 主干身份；继续检查源码 hash、范围和既有 64／320 种语义／整网状态。文件快照 hash 未修改。
- 浏览器验收覆盖正常 CSA 入口、展开空白／折叠摘要选中、入口激活 PyPTO tab 且左图 transform 不变、L0 原位展开及 L3 内部折叠不改变左图。新建 Indexer 步骤已验证鼠标点击定位 decode_indexer.py、双向语义高亮及基础源码 tab 保留。浏览器仍出现既有 MutationObserver 日志，不宣称 console 零错误。非 CSA 参数和 L4 仍沿用前述证据边界。

## 13. 验收标准

1. 左栏默认呈现整网的 L1 架构；不使用“整网上下文 / 进入 CSA”替换图。原有祖先节点可逐级展开与收起；L0 保留整网主干及折叠的 Attention／FFN，不能替换成单个模型根节点。
2. 在整网内展开 CSA 的 Query Projection，L2 子节点出现在该节点容器中，祖先和周围节点保留；局部展开不重置其他模块。L2 全部收起无损恢复同 ID、同名称、同摘要拓扑与布局的 L1，共享实体不复制，无悬空/重复边。
3. L2 各语义步骤可仅凭官方来源定位；PyPTO 补充内容归入 L3；source/config/phase 不匹配时明确提示。
4. Op、Value、Logical State、Scope 使用规定形状；计数不混淆容器、值、算子、kernel 定义与运行实例。
5. 左栏选中 CSA 后能看到“查看 PyPTO 实现”；仅选中 CSA、展开 L2 或点击 L2 叶子均不自动进入 L3。点击明确入口后，中栏显示对应 CSA 的 L3，左栏选择、展开及 viewport 不变，来源 scope 可追溯。
6. L3 首次进入即显示 L2 对应的实现步骤：九个程序默认展开，26 个已有 L2 叶子有可见映射；程序可独立收起／展开。明确组合、拆解、资源物理化、通信和 scope 变化，缺失映射可见。不能仅以函数摘要节点或 Q/KV 两条路径验收全部 L3。
7. 官方已有 cache、state、通信与 HC 上下文不被错误标为全新数学语义；函数分组不被误称单 kernel 融合。
8. Diff 整体染色、图例解释完整；关闭后恢复实体视觉；实现增量可筛选，范围与条件可追溯。
9. 源码窗同时保留 `官方 · model.py` 与 `PyPTO · decode_csa.py` 文件 tab。点击左／中图分别激活对应文件并定位，切换 tab 保留各自位置且不改变图；调用/定义是导航操作，跳转叶子定义不关闭两份基础源码。无独立定义时说明实际证据，不伪造函数。
10. L4 展示的每个字段有匹配证据；未知内容不猜测；最终 UI 按后续确定的设计另行验收。
11. 三栏布局、独立缩放、展开按钮、间距和紧凑箭头在默认及展开状态可读可操作。
12. 打开对应实现后，两图可同时对照；节点选择追踪全部已知多对多映射，不静默遗漏分支，不将共享实现复制为多个计算实体。L3 内部 expand / fold 只细化中栏实现，不改变左栏层级。
13. 层级 dropdown 只批量控制适用范围的展开深度；不作为独立图切换器，也不替代 CSA 实现入口。保存／恢复或深链打开 L3 时同时恢复来源 CSA 和实现上下文。
14. 源码窗的 Explorer `#` 注释说明来源、变化、原因、引入者及证据，与原始作者注释可区分；多引用与两端证据不因文件 tab 重构而丢失。

## 14. 来源、变更与维护

本修订的交互依据为 2026-09-07 用户最新讨论及随后的实施请求；源码与编译事实沿用本地快照，并运行 hash／行范围回归，未重新核验远端 main。外部链接的 main 是来源入口；正式数据需固定 commit/hash 后再建立行级引用：

- [官方 inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/main/inference/model.py)
- [官方 inference/kernel.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/main/inference/kernel.py)
- [官方 inference/config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/main/inference/config.json)
- [PyPTO 模型实现参考入口](https://www.pypto.ai/pypto-lib/models/deepseek_v4_flash_mtp/)
- [本地官方模型快照](../../../DeepSeek-V4-Flash-Official/model.py) 与 [本地 inference 配置](../../../DeepSeek-V4-Flash-Official/inference-config.json)
- [本地 decode_csa.py](../../deepseek_v4_flash_dspark/decode_csa.py)、[qkv_proj_rope.py](../../deepseek_v4_flash_dspark/qkv_proj_rope.py)
- [历史编译 Frontend Dump](../../_jit_l3_decode_csa_20260903_010617/passes_dump/00_frontend.py) 与 [Kernel 配置](../../_jit_l3_decode_csa_20260903_010617/next_levels/decode_csa_test/kernel_config.py)

本修订替代旧版以下约定：左栏固定仅 L1、L2 留在中栏、同一 dropdown 切换四张图、L2 排除一切逻辑状态、用旧四类标签涵盖全部变化、L4 固定四卡展示。原有拓扑正确性、可逆布局、间距、按钮、箭头、三栏及真实证据约束继续有效。

最新讨论进一步明确并覆盖上一轮实现假设：

- 左栏 CSA L2 融入整网原有父子结构，不是独立模块图与整网切换。
- lineage-006 按用户最新确认采用 L0 整网概览，覆盖此前“不增加 L0”的记录；Decoder 层号仍不作为语义层级。
- 架构图与 PyPTO 计算／实现图分开查看，共同 CSA 模块承担衔接。
- 左栏 L2 的 expand 不打开 L3；选中 CSA 后通过“查看 PyPTO 实现”入口进入中栏 L3。
- 官方／PyPTO 两份源码使用持久文件 tab 同时保留；“调用位置 / 实现定义”从主页签改为导航操作。
- 上一轮文档修订不构成实现完成；随后本轮按三项要求实施，结果与验证边界见 §12.2，不能沿用 lineage-004 的完成声明。

后续涉及层级定义、实体类型、映射、证据、视觉或联动的变更，必须同步更新本文件。架构事实与逐项证据校验记录在 `model_architecture_validation.md`；本 spec 管理产品行为与验收。方案建议、待设计事项与已实现状态必须分开标注。


### 选中与配色回归约束（lineage-007）

- 左右图选中算子后，仅选中项与跨栏映射命中项保色；选中模块时，其可见子节点保色，其余节点统一灰色填充。该规则优先于 Diff；清除选择后恢复当前配色模式。
- 普通模式共用官方语义配色。Lightning Indexer 模块统一使用 sem:gate；L3 单一来源步骤使用对应 L2 语义颜色，不能统一继承父函数颜色。
- 展开层级不是映射变换分类依据。Q down projection、KV projection、Output inverse RoPE 分别保留官方投影或逆旋转操作，标记 PRESERVED；这不声明精度或数值等价。量化组合、拆分等实现步骤继续依据映射证据标记 EXPLICIT。
- Diff 默认展开视图必须能看到上述绿色节点；不能把全部子节点机械标记 EXPLICIT。普通模式语义色与 Diff 映射色是两套明确的显示模式。


### Diff 以左侧 L2 为基准（lineage-008，取代此前 Mapping Diff 主视图规则）

Diff 的首要问题是 L3 相对官方 L2 展开、显式化或新增了什么。PRESERVED / EXPLICIT 等源码变换标签仍保留在证据详情，但不再决定主图 Diff 颜色；lineage-007 中“Diff 默认应出现绿色”的要求由本节替代。

- 点击 Diff 打开分类下拉框并启用差异染色。提供“启用差异染色”总开关，以及默认全选的三个独立复选项：蓝色“展开的计算步骤”、黄色“显式缓存／状态／索引”、紫色“新增通信／重分布”。
- 比较基准固定为完整 CSA L2 语义，不随左侧临时折叠而改变。依据注册表的 L2 来源映射与显式节点职责分类，不按名称或旧映射颜色推测。
- 展开不等于新增模型算法；已有逻辑状态的物理缓存、读写、更新和寻址表达归显式化。部署引入的通信、head-group ownership 重分布归部署新增。已有独立计算、模型上下文、折叠程序摘要灰色显示，不计新增。
- 计数为当前展开状态下的可见节点数；范围框不重复计数。折叠摘要灰色，展开后显示其内部分类。取消复选只去色，不删节点或连线；全不选时全灰。关闭总开关恢复普通语义色，保留分类选择。
- 算子／模块选中后的范围外灰填充优先于 Diff 分类高亮。清除选中恢复当前筛选结果。普通模式继续保持左右对应节点语义色一致。
- 点击按钮再次收起下拉框不关闭染色；点击框外或 Escape 收起。节点详情说明分类理由及其 L2 来源，完整源码映射证据继续可查。


### Diff 图例色板与说明（lineage-009）

Diff 图例和对应节点使用用户提供色板：计算展开 `#95bcf9`、状态显式化 `#fad595`、通信重分布 `#a69dfa`；保留色令牌统一为 `#90f4b4`，当前保留节点仍按 L2 Diff 规则显示灰色。浅色填充配深色文字。三类图例标题下各显示一行说明，解释计算拆解、逻辑状态落实，以及部署新增的含义。
