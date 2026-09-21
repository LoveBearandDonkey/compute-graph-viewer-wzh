# HPC Topology Viewer

> **启动页（一站入口）：<https://cinnnnnnndy.github.io/hpc-topology-viewer/launch.html>**
> —— 工作台、驾驶舱、独立 pattern、参照系 / PRD / 映射简图与 `research/` 研究报告
> 按组整合在一张浅色启动页上，点开即看（dev: `/hpc-topology-viewer/launch.html`）。
> `research/` 顶层报告 HTML 由构建带进站点（`/research/*.html`，见 `vite.config.ts`
> 的 `researchHtml` 插件），仓库里不存第二份。

Interactive 3D viewer for a large-scale HPC accelerator cluster — racks, compute
nodes, and the interconnect fabric — built with **React + Three.js**
(`@react-three/fiber` + `@react-three/drei`). Geometry is **procedural by
default**, with an **optional open-source GLB swap layer**: drop a correctly
named `.glb` into `src/scene/models/` and the matching part (NPU module, CPU,
blade, cabinet, DIMM, optic, DPU, PSU, CDU, switch line-card) renders the real
model instead — no code edits, automatic fall-back if absent. See
[`src/scene/models/README.md`](src/scene/models/README.md) for the part list and
download guide.

## Views

- **Overview** — 16 cabinets (12 compute + 4 switch) with inter-cabinet optical links.
- **Cabinet** — drill into one cabinet: power shelf, management blade, compute nodes, liquid-cooling manifold.
- **Node** — a compute blade (accelerators, CPUs, on-board L1 switch chips, DPU, optics) and the on-board switch device.
- **Topology** — two-tier non-blocking Clos: all compute cabinets → 7 switch planes → cross-node RDMA / VPC planes, with hover-to-highlight uplinks.

Every interactive element shows a hover tooltip. The seven recurring colors map to
the seven independent switch planes (each plane is its own non-blocking fabric).

## Rubik-cube pattern（逻辑魔方 · 独立迭代）

The cockpit's 逻辑魔方 (5 形态重排 · 轴标注 · 正交 2D/剖面 · 四维通信组) is also
extracted as a standalone, parallelism-configurable pattern for independent
iteration — default **TP8×PP5×DP100 = 4000 ranks**, Pangu Pro MoE's real
training strategy (EP2 folded into DP). Entry page:
`public/rubik-pattern.html` (dev: `/hpc-topology-viewer/rubik-pattern.html`);
sources & docs: [`public/vendor/rubik-cube/`](public/vendor/rubik-cube/README.md).
Integration hooks for the whole-network graph / expert graph are pre-wired
(`selectLayer` / `selectBucket` / `onSelect`).

## 整网切分 · rank 装载（net-sharding · 独立迭代）

与逻辑魔方并列的第二个独立 pattern，坐标系是**模型计算图**而不是并行超立方：把 openPangu
整网图的每个算子标注成「被哪个并行维、沿张量的哪根轴切成几份」，据此算出任意一个 rank
到底装了什么（层段 / 注意力头片 / FFN 中间维 / 词表片 / 专家桶 / 上下文段 / 序列片 / 数据副本），
并把 **HCCL 集合原语呈现为分片状态的转换器**——AllGather / ReduceScatter / AllToAll /
AllReduce / P2P 各自把张量从一种分片状态搬到另一种。支持 6 维 **TP·CP·SP·PP·EP·DP**
（CP/SP 是现有 `NODE_DIM` 四维标签所没有的）。默认盘古 Pro MoE 真实策略
tp8·pp5·dp100·ep2 = 4000 rank。入口页：`public/net-sharding-pattern.html`
（dev: `/hpc-topology-viewer/net-sharding-pattern.html`）；
sources & docs：[`public/vendor/net-sharding/`](public/vendor/net-sharding/README.md)。

魔方答「谁和谁一组」，它答「这一组各自装了什么」——两者互为反查，
挂点见 `pattern.json` 的 `integrationHooks`。

## 并行拓扑：参照系与 PRD（两条各自独立的链接）

同一套概念的两份文档，各占一条顶层链接，各自自包含（单文件打开即可，零外部依赖），
互不依赖也不共用目录——打开就是内容本身，中间没有目录页：

| 链接 | 源 | 是什么 |
|---|---|---|
| `/parallel-reference/` | `public/parallel-reference/index.html` | **《分布式训练参照系 —— 五根轴 · 两个对象 · 三组坐标》**：术语、基数、切分、映射、通信、编号、运行时、**Rank 状态图 / 通信状态图**的完整参照系，配可交互示意图与勘误。**这是参照系的唯一正本** |
| `/parallel-prd/` | `docs/parallel-topology-prd.md` | **《并行拓扑可视化工具 PRD》v0.1**：把参照系直接落成信息架构——四类结构性错误、五条设计原则、五个视图族与五个图层、可直接测的验收标准 |

PRD 那张页是**构建产物**，不要手改：改 `docs/parallel-topology-prd.md` 之后跑
`node scripts/build-prd-page.mjs docs/parallel-topology-prd.md public/parallel-prd/index.html`
重新生成（需先 `npm i -D marked`），md 与产物一起提交。手改 HTML 会让两者分叉，
之后没人说得清哪份是准的。源放 `docs/` 而不是 `public/`：`public/` 下的东西会原样发布，
md 跟着发出去就等于同一份内容有两条链接。

两条链接各自独立，但视觉语言同源（同一套 token、版心、章节标尺）——是一对文档，
不该长成两种东西。

> **参照系只有一份正本。** `public/parallel-topology/concept-map.html` 曾经是同一篇文档的
> 第二份副本（`<title>` 一模一样），停在 16 节，而正本已经走到 20 节——两份的章节号差一位，
> 于是「§15」到底指哪一节谁也说不清（2026-09 核对时发现三处交叉引用已经错位一格很久）。
> 现在那份已合并进正本，原路径改成**跳转桩**（带锚点跳到 `/parallel-reference/`）——
> 路径发出去过，不能直接删成 404。**要改参照系，只改 `public/parallel-reference/index.html`。**

## 故障故事线 · 无限画布（`/incident-canvas/` · 独立迭代）

**主视图永远是并行拓扑 pattern 本体**——就是打开 `/combo-workbench/` 在舞台上看到的那张
「模型分片与训练设备映射」，不是另画一份像它的图。画布围着它长：**下钻 = 在画布上把一格
展开出来**，展开得越多画布越大，收起就是那格的 ×。

| 出处 | 拿走的是 |
|---|---|
| `/patterns/net-slicing/` | **主视图**（五刀全落并进卡维度），以及「整网 · 真图展开」那格（同一 pattern 的另一份配置：不落 DP、不进卡维度） |
| `/combo-workbench/swimlane.html` | 「泳道 · 微批次生命周期」那格，原样嵌入 |
| `config-relation-observer` | **故事线与数据**：两条问题线、11 个运行事件、5 套机制相位、11 张证据图与全部读数。数值逐条照搬没改 |
| `pangu_sophon_pytorch` | 「训推大盘」那格的**指标口径**：十条关键少数 + 五个早停钩子。只取口径不取值 |

前两份嵌的是**规范路径上的本体、不是拷贝**，所以「画布里的那一份」和「直接打开那条链接」
永远是同一个文件、同一个版本。整页只有 `var BASE` 一处知道自己离站点根有多深，
发布步骤按入口 sed 它并用断言兜住——BASE 错了不报错，只会让三格静默 404。

**联动搭在层号上，不是 rank。** 故事线讲的是一个 2048 卡的实跑（`rank 1559` 是那个集群的门牌），
被嵌的 pattern 画的是它自己 128 卡的演示集群，两边卡数不同、rank 号天然对不上；
而 L38 在两边都落在 PP3。所以选中事件，主视图点亮**承载那一层的卡**，pattern 自己在角上报
「L38 · 32 张卡 · 各持 1/16，无一张完整」并注明这是来自另一个视图的联动高亮。
（也试过让 pattern 直接画 2048 卡：DP 512 行退化成一条看不清的斜线、12 秒才画完，
那不是更真，是把主视图毁了。）

五种机制（路由塌缩 / barrier 语义 / 依赖链回压 / 存活区间叠加 / 碎片）各有一套自绘 SVG，
不共用「通用示意图」。大盘那格**只取口径不取值**：每个数都能在故事线同一事件里找到原文，
没采到的格子写「—」。

`?open=` 把「当时摊开了哪几格」也写进链接。细节见
[`public/incident-canvas/README.md`](public/incident-canvas/README.md)。

## 超节点集群 · 八步（`/supernode-intro/` · main 自带）

一页讲清**一张卡怎么长成一个 8,192 卡的集群，以及哪一层才是换挡点**。八步各一张简图，
一步只讲一件事：

| 步 | 讲什么 |
|---|---|
| 01 卡 | NPU = 1 rank，是通信域里的最小参与者；die 与 core 不单独参与通信 |
| 02 节点 | 8 卡任意两两直通（排成环画 28 条弦，一眼看得出没有中心） |
| 03 超节点 | 16 节点 = 128 卡，全互联延伸到整域，1:1 无收敛 |
| 04 边界 | 域内 UB / 跨域 RDMA —— **换挡点是超节点边界，不是节点边界** |
| 05 集群 | 64 超节点 = 8,192 卡 |
| 06 盘古 4000 | 盘古 Pro MoE 的 4,000 卡配置：TP8 × PP5 × DP100，EP2 折入 DP → 50 域 |
| 07 通信关系 | 一张卡同时属于四个组，只有 DP 那一个跨域 |
| 08 运行状态 | 八级典型利用率 + 一步的时间分解（访存 48% > 计算 40%） |

第 06 步是这一页的落点：**TP8 恰好一台节点的 8 张卡，一个副本 40 卡又装得进 128 卡的
超节点——所以只有 DP 跨域**。前五步建立的结构，到这里变成一个具体配置能不能放得下的判断。

形制照 `rank-intro` 那类分步讲解页：一个主可视化区居中，标题与图例贴上沿、读数贴右沿、
步骤条贴下沿，四周留白。**URL 带步号**（`?step=1`–`8`），所以任何一步都能单独发出去；
键盘左右/ 上下 / 空格翻页，乱参数退回默认不报错。

视觉：整体中性灰（主文字 `#E8E8E8`、次文字 `#A0A0A0`），纯白只给当前选中对象的边框与
关键高亮；**只保留两支必要的颜色**——域内 UB 与跨超节点，因为这一页全部的意义都压在
这一个对比上。Inter 400–500，数字等宽对齐，不用卡片包裹内容。

单文件、零依赖（不引 vendor、不嵌 iframe），源就是
[`public/supernode-intro/index.html`](public/supernode-intro/index.html)，vite 直接拷进
dist，发布流水线不用为它加任何一步。**它不改动任何现有 demo**：驾驶舱、并行拓扑、
各 pattern 页都原样不动。

## 组合工作台（`/combo-workbench/` · 独立迭代）

一块**摞格子**的应用台面，后续的工作台组合往它上面继续摞。形制与视觉语言同
`public/cube-cockpit.html`：顶栏 + 舞台 + 底部抽屉，铺满视口、自己不滚动（滚动都发生在
格子里面），同一套 PTO token（浅色优先、`[data-theme=dark]` 单块覆盖）、同一种面板圆角与
段控，抽屉就是驾驶舱那张「Profiling 视图」卡的形制。token 是内联的，不引 vendor。

现在三格：

| 格子 | 内容 | 我们给了什么 |
|---|---|---|
| 舞台左 · 并行拓扑（**主视图**） | iframe → `/patterns/net-slicing/pattern.html` | 什么都不加；名字与「＋ 搜索」由它自己在画布左上角出。占满舞台剩下的宽度 |
| 舞台右 · 整网图（**跟随视图**） | iframe → `/patterns/net-slicing/pattern.html?cuts=pcte&rank=0&vtab=side` —— **同一个 pattern 的第二份配置**；格头段控可切到「全局切分」（`view=solid&p=grid`） | 格头：段控（整网图 / 全局切分）+ 收起；藏掉搜索 / Rank / 四把刀 / 搜索结果浮层（见下），定宽可拖 |
| 抽屉 · 微批次生命周期泳道 | iframe → `./swimlane.html?chrome=0`，MB07 · step 18420 | 格头：三档时间范围的段控 + 收起 |

三格之间两条可拖的分隔条（左右一条、上下一条）。

左格与右格是**同一个 pattern 的两份配置**，差别只在参数：

| | 右 · 整网图（跟随） | 左 · 并行拓扑（主视图） |
|---|---|---|
| 落哪几刀 | `cuts=pcte` —— PP / CP / TP·SP / EP 落下，**独独不落 DP** | 五刀全落 |
| 卡维度 | `rank=0`，不进卡维度 | 进卡维度（Rank） |
| 默认机位 | 3D | 3D |

整网图问「一张整网被切成什么」——DP 是「整套安排复印几份」，复印件对这一问没有
信息量，落下来只会把同一张图重复 16 遍，所以那一刀不落；主视图问「切完之后落成
world 张卡」。

**整网图当成一格独立的「整网」来看，不是「并行拓扑的某个状态」。** 目前的实现是从
net-slicing 抽出来的一份配置，这是权宜——后续可以像并行拓扑当初那样把这一屏单独抽成
自己的 pattern（`vendor/model-graphviz/` 就是它的渲染器本体）。抽走那天只要换掉这一格的
`src` 与 `hideCss`，台面别的地方不用动，因为两格之间**只走语义消息、不共享配置**：

| 方向 | 走什么 | 结果 |
|---|---|---|
| 主视图选中一张卡 | `pto:select` → `pto:state{sel}` | 整网图亮起那张卡装的那一段层；算子旁的切分读数跟着那张卡变（实测 r0 → `TP0/32`、r1 → `TP16/32`） |
| 整网图点一个算子 | 读它命中区自带的 `data-li` | 主视图落 `层 · Ln` 标签，高亮持有这一层的那些卡 |
| 任一格选中某一层 | `层 · Ln` 标签 | 两边互相同步 |
| 全局切分里点一个腔格 | `pto:select` → `pto:state{hl:{rank}}` | 主视图与泳道照亮持有那一腔的那张卡 |

换 pattern 时要保住的是这几条语义，不是某一套 query 参数。

**整网图这一格有两个 tab**（`?n=` 进 URL，默认 `net` 不写）：「整网图」照旧；
「全局切分」把同一份 pattern 切到 `view=solid&p=grid`——交集腔的总账：权重 / 梯度 /
优化器态 / 激活四个形态 × 全部的腔（W·G·O 同一套切分，画成一只托盘的三层厚度、
厚度比即字节比 2:2:12；激活自己一套，b·s 被真切）。每一格都是命中区，点一格定位
持有它的那张卡，主视图与泳道跟着照亮。联动走的还是上面那几条语义消息，没有新协议；
唯一新开的口子是 `pto:state{persp}`（demo 侧白名单加了一个字段），让换 tab 不重载
iframe、不丢选中。

整网图是**跟随视图**：搜索、Rank、四把刀、搜索结果浮层全藏掉——选中由主视图发起，
两个搜索框摆在同一屏用户不知道该点哪个，而 `rank=0` / `cuts=pcte` 是这一格的定义、
不是可调项。四个机位留着（它要能换着看）。台面驱动它落标签仍走同一条路
（dispatch 到 `[data-act=qopen]`），**藏起来的节点照样收事件**。

**默认机位是 3D，不是侧视**：侧视那一屏的算子名/切法标注是固定屏幕字号的，而
pattern 按舞台长宽比反解 viewBox（`VW = clamp(692·AR, 1080, 2600)`），AR 低于 1.56 时
VW 被钳在下限 1080——620px 宽的格子要装 1080 单位的内容，缩放只有 0.57，文字不跟着缩
就全撞在一起。实测重叠的文字对数：

| 机位 | 620px | 900px | 1080px |
|---|---|---|---|
| 侧视 | 48 对 | 42 对 | 35 对 |
| 正视 | 1 对 | 0 对 | 0 对 |
| 3D | 0 对 | 1 对 | 1 对 |

也就是说侧视要接近独立打开那种两千像素的宽度才读得下来，不是拖一拖就能解决的。
3D 讲的同样是「整网被这几刀切成什么」，在窄栏里 0 重叠，所以拿它当默认；
侧视一个点击就能回去。

**先前这一格挂的是跨仓 `pto-design-system` 的 training-sidecar**，换掉的原因有两条：
它一次只能选中一层（公开接口只有 `selectLayer(单个层号)`），而这一屏要回答的是
「一张卡装的那十几层」；而且它与右格是两套不同的模型口径（46 层 vs 48 层），层号对不齐。
换成同源的第二份之后这两件事同时消失——层号天然一致，选中走两边同一套搜索标签。

认 `pto:state` 的格子用 `postMessage` 驱动，**不重载 iframe**——切视图不会丢掉格子里
选中的那张卡 / 那条事件。顶栏那颗月亮是**主题桥**，三格都认 `?theme=` / `pto:state`，
一条消息过去就同时换明暗。
台面状态（看哪个视图、左格多宽、抽屉多高、折了谁、明暗）写进 URL，
`?a=solid&b=l34&w=600&h=460&ui=dark` 这样一条链接就是你当时那一屏。

**两格的层联动（双向）**：在任一格选中第 *n* 层，另一格就落一枚同样的
`层 · Ln` 标签——左格点画布上的逐层命中区，或任一格手填「＋ 搜索」都算。落下之后
由 pattern 自己算出这一层落在哪个 stage、被 TP/EP/DP 各切几份、共几张卡与之有关，
并把那些卡点亮；摘掉标签另一格也跟着摘（只摘「层」那一枚，不碰用户手填的别的标签）。
因为高亮的是**卡**不是算子，「关闭整网」进卡阵那个模式一样有效。

走的是 pattern 自己的控件（点「＋ 搜索」→ 点「层」→ 填值 → 回车，全程它自己的事件
处理器），**不重载 iframe**，视角/缩放/选中卡都不丢；它的 `pto:state` 白名单里没有
搜索标签，chips 关在它的 IIFE 里，所以只能这么走。

台面靠轮询两格的标签读数判「谁变了」，**比的是这一轮读到的值**而不是「我们推了什么」：
后者有回声竞态——推过去之后对面要过几十毫秒才真的更新，下一轮读到的还是旧值，会被
当成「对面主动变了」再推回来，一来一回把刚选好的层关掉（实测 3 次里 2 次）。
同一轮只处理一侧，理由相同。

台面在启动页上另有一张卡（「工作台 · 主线」组的头一张），那张卡用的是启动页的样式；
片段本体在 `public/combo-workbench/launch-card.html`，由 `deploy.yml` 叠加插入。

`public/combo-workbench/swimlane.html` 是 **compute-graph-viewer 的上游拷贝**
（`pangu-moe-trainviz/microbatch-lifecycle-swimlane-mock.html`），连同
`vendor/swimlane-task/` 一起搬过来。相对上游只加了三处：文件头出处说明、
`embed.css`、`embed-bridge.js`（后两个是内嵌壳，一律靠 `.click()` 现有控件，
不碰它自己那个 IIFE）。正文一行没动，上游更新时重拷一遍再补这三处即可。
`embed.css` 里还修了一处跨仓类名撞车：本仓 pto-design-system 快照的
`.legend`（色板面板用，`flex-direction: column`）会把泳道底部那排图例竖着摞成
138px 高、压穿 38px 的页脚——独立打开也一样坏，所以那条修复不分内嵌与否都生效。

**与被嵌 pattern 的同步**：舞台那两格嵌的是**规范路径上的本体**，不是拷贝——都指向
`/patterns/net-slicing/pattern.html`（同一个文件，两套 query），由它自己的发布步骤从
main 叠加，所以「工作台里的那一份」和「直接打开那条链接」永远是同一个文件、同一个版本。`?v=<短SHA>` 版本戳**只打本目录自己的文件**（`swimlane.html`）：
拿本分支的 SHA 去戳别人的成品是反效果——它们更新而我们没动时戳不变（该刷的没刷），
我们改台面而它们没动时戳变了（不该刷的刷了）。

台面本体自包含（token 内联，不引外部样式）。舞台两格的内容是**构建期产物**——
`/patterns/net-slicing/` 由发布流水线从 `public/parallel-topology/demo.html` 产出，
仓库里没有那份文件，所以本地直接开 `public/` 时那两格是空的（有兜底文案说明），
要看真样子得看发布后的站点。哪条 404 也只坏那一格，外壳还在。

## Develop

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # type-check + production build to dist/
npm run preview
```

## Content encoding (anti-scrape)

All product/brand display strings live in `src/content.ts` as **base64(UTF-8)**
and are reconstructed at runtime by `src/codec.ts` (`dc()`), so the committed
source tree (and the built bundle) contain no plaintext product names — a
repository grep or code search finds nothing; the terms only materialize in the
browser at runtime.

The plaintext generator that produces `content.ts` is intentionally **kept out
of version control** (`scripts/` is git-ignored) so the plaintext never lands in
the repo. `content.ts` is the committed, encoded artifact.

The deployment is also marked `noindex` (see `index.html` meta tags and
`public/robots.txt`) so crawlers do not index it.

## Notes

Cabinet outer dimensions use the published envelope; in-cabinet and on-board
layouts are schematic abstractions based on public material and do not represent
real engineering layouts.
