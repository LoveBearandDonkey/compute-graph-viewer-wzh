# net-sharding pattern（整网切分 · rank 装载 · 独立迭代）

以**模型计算图**为坐标系理解 rank 的 pattern。独立迭代入口：**`public/net-sharding-pattern.html`**
（`npm run dev` 后访问 `/hpc-topology-viewer/net-sharding-pattern.html`）。
控制台暴露 `window.ns` 句柄可直接调 API。

| file | what it is |
|---|---|
| `pattern.json` | **设计系统契约**（同 `rubik-cube` / `memory-architecture` 约定）：id/type/描述、来源、`useWhen`、`requiredApis`、`layoutRules`、允许与禁止的覆盖、`integrationHooks`、`agentReuseRule`。改动或复用前先读它。 |
| `pattern.js` | 切分规格表 `SHARD` + 纯数据层 `createModel(config)` + DOM/SVG 渲染层 `mount(container, opts)`。无 Three.js 依赖。 |
| `pattern.css` | pattern 独有的样式（维度签名色、装载卡、分片状态流水带、整网图上的切分标记）。控件本身复用设计系统的 `.btn` / `.panel-shell` / `.stat-chip`。 |

## rank 卡 = 整网的**并行化展开**，不是它的裁剪

一张 rank 卡里的节点集合，相对逻辑整网图**既减也增**：

| | 是什么 | 画成什么 |
|---|---|---|
| **减** | 别的 stage 的层在这张卡上**根本不存在**（词嵌入只落 PP 首段、LM Head 只落末段） | 虚线空框——是「没有」，不是「弱」 |
| **增** | HCCL 集合原语是**并行化产生的**，逻辑图里没有这些节点：TP AllReduce · PP Send/Recv · DP 梯度 AllReduce · CP AllGather(KV) | 粗描边胶囊 + 虚线引到插入点 |
| 其余 | 节点还在，但只持有**一片** | 按切它的那一维签名色着色 |

> AllGather / ReduceScatter / AllToAll 那几个**图里本来就有**，属「保留」——
> 只有上面四类是并行策略插进去的。这个区分是这张图的核心，不能含糊。

## 显示形式：共享布局 + 差分着色（small multiples）

这张图是**预定位**的（55 个节点各带 `x/y`），所以 small multiples 是白送的：
三张卡用同一份坐标渲染、位置**逐点对齐**，于是三个结论**不靠文字**就能读出来——

| 并排比 | 读出什么 | 实测 |
|---|---|---|
| **TP** | 结构完全相同，只有分片区间不同 | 三卡缺席节点数一致（8/8/8），head 区间 0–7 / 16–23 / 32–39 |
| **PP** | **结构不同**——词嵌入/LM Head 出现或消失 | 缺席数 5/8/8（PP0 是首段，词嵌入存在） |
| **EP** | 结构相同，专家桶不同 | E0–31 / E32–63 |

样式纪律只有一条：**除了状态，什么都不许变**——卡的尺寸、节点的位置与形状三张完全一致，
变化只能落在颜色与存在与否上。否则「对齐比对」这套读法就不成立了。

入口：`public/net-sharding-pattern.html` · `?axis=tp|pp|ep` 决定并排比哪一维。
程序侧：`handle.setAxis('pp')` / `selectRank(r)` / `setConfig({...})`。

