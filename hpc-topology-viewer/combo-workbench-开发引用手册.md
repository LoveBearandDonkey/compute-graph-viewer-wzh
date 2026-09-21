# 组合工作台 · 开发引用手册

这是给要接手 `/combo-workbench/` 开发的人看的**索引文档**：每个部分的代码在哪、
上游出处是什么、由哪个分支的哪一步发布上线。产品向的说明（三格是什么、为什么这样切）
看 [README.md 的「组合工作台」一节](README.md)，两份不重复。

核对时间：2026-08-20（已随 E1 迁移更新）。本文档记的是**引用关系**，这类关系不常变；
但如果某个格子换了实现、或部署步骤挪了地方，下面的行号会先过期，发现对不上以文件里的
实际内容为准。

---

## 0. 改动要落在哪个分支（E1 迁移已完成，历史风险record 在此）

**现状（已修复）：** `deploy.yml` 的 "Checkout combo-workbench branch" 步骤现在
`ref: main`，`/combo-workbench/` 的发布内容就是 `main` 上 `public/combo-workbench/`
的实际内容，不再从 `claude/topology-swimlane-card-layout-fdrgiq` 分支单独取。
在 `main` 上改这个目录、推 `main`，会直接反映到线上——不需要再记得「改完要合并回哪」。

`claude/topology-swimlane-card-layout-fdrgiq` 分支还在，可以继续用于开发/PR，但它
**不再是发布源**，只是走 PR 合回 `main` 的一条普通开发分支，和仓库里其它 `claude/*`
分支性质一样。

**历史风险（曾经存在，记录下来避免同类问题在别处重演）：** 在这次迁移之前，
`/combo-workbench/` 一直是从 `fdrgiq` 分支单独 checkout 发布的，`main` 上那份
`public/combo-workbench/` 只是历史 PR 合并进来的镜像——`fdrgiq` 有提交但还没发起/
合并 PR 时，`main` 会悄悄落后于线上实际发布的内容，且**没有任何报错**提示这件事。
这和 `/parallel-topology/` 早年踩过的坑是同一种：那边最初也是从专属分支（`ff10w3`）
checkout 发布，「推 main」与「推 ff10w3」谁最后跑谁赢，同一条链接的内容会在两版之间
随机跳（`deploy.yml:248-254` 的注释原话记着这段历史）。两次的修法相同：把源码彻底
合并进 `main`、checkout 改成 `ref: main`，只留 `main` 一个发布口子。

以后仓库里再出现「产品/pattern 先在专属分支上开发、deploy.yml 按分支名单独
checkout 发布」这种结构时，先假设它迟早会出这个问题，尽早规划迁移到 `ref: main`，
不用等它真的两次线上跳变去踩坑才处理。

---

## 1. 目录 → 部署产物映射

| 站点路径 | 本仓源文件（`main` 分支） | 怎么产出 |
|---|---|---|
| `/combo-workbench/index.html` | `public/combo-workbench/index.html` | `deploy.yml:499-531`："Overlay 组合工作台" 步骤用 `sed` 把 `var BUILD = '';` 替换成本分支短 SHA，其余原样拷贝 |
| `/combo-workbench/swimlane.html` | `public/combo-workbench/swimlane.html` | 同一步骤，`sed` 给所有相对 `href=/src=` 的 `.css/.js` 打 `?v=<SHA>` |
| `/combo-workbench/vendor/swimlane-task/{pattern.css,pattern.js}` | `public/combo-workbench/vendor/swimlane-task/` | 原样 `cp`，不打版本戳 |
| `/combo-workbench/embed.css`<br>`/combo-workbench/embed-bridge.js`<br>`/combo-workbench/favicon.svg` | 同名文件 | 原样 `cp` |
| `/combo-workbench/observatory/**`（30+ 个文件） | `public/combo-workbench/observatory/` | `cp -r` 整目录搬，见 `deploy.yml:522-528` |
| `/launch.html` 里「组合工作台」那张卡 | `public/combo-workbench/launch-card.html` | `deploy.yml:532-559`："Overlay 组合工作台卡片" 步骤把这个片段插进 main 构建出的 `dist/launch.html`「工作台 · 主线」组最前面 |
| 舞台两格实际内容 `/patterns/net-slicing/pattern.html` | **不在本仓库、不在 combo-workbench 目录下** | 由 `main` 分支自己的 `public/parallel-topology/demo.html` 在更早一步（`deploy.yml:248-377`，"Overlay 并行拓扑"）产出。combo-workbench 只是引用这条**规范路径**，构建期不复制第二份 |

本地 `npm run dev` 直接开 `public/combo-workbench/index.html`：舞台两格是空的（有兜底文案），
因为 `/patterns/net-slicing/` 是构建期产物，仓库里没有这份文件。要看两格的真样子只能看
发布后的站点。抽屉（swimlane / observatory）是自包含文件，本地能看到真样子。

---

## 2. 三个格子逐一的引用

代码位置：`public/combo-workbench/index.html:454` 起的 `var SLOTS = [...]` 数组，
三个条目分别是 `id: 'netgraph'`、`id: 'topology'`、`id: 'swimlane'`；对应的 DOM 挂点是
`#slot-topology`（`index.html:326`）、`#slot-netgraph`（`:345`）、`#dock`（`:366`）。

### 格子「并行拓扑」（主视图，`id: topology`）与「整网图」（跟随视图，`id: netgraph`）

- 两格都指向同一个构建产物 `../patterns/net-slicing/pattern.html`，只是 query 参数不同
  （`netgraph` 加 `cuts=pcte&rank=0`，见 `index.html:515-537` 的 `src()` 函数与其上方注释）。
- `netgraph` 这一格有**两个 tab**（`key:'n'`，URL 参数 `?n=`，默认 `net` 不写）：
  「整网图」= 原来的 chain 侧视配置；「全局切分」（`?n=cav`）= 同一份 pattern 的
  `view=solid&p=grid`——权重/梯度/优化器态/激活四形态的交集腔总账，逐腔可点定位 rank。
  换 tab 走 postMessage 不重载（`pto:state{view, persp}`——`persp` 是为这一格在
  demo.html 的 `pto:state` 白名单里**新加的字段**，只认 map/cav/grid）。点腔格后的
  联动走既有的 `pto:select` → `pushHL` 那条路，台面没有新协议。这一格的 hideCss
  多了一条 `[data-act="persp"]`：子视角切换器由格头 tab 代言，格子里那排藏掉。
- 上游来源：`main` 分支的 `public/parallel-topology/demo.html`，经 `deploy.yml` 的
  "Overlay 并行拓扑" 步骤抽取发布。这条链早先已经完成过和 E1 同类的迁移
  （checkout 从专属分支改成 `ref: main`），不受本文档第 0 节说的那类风险影响。
- 两格之间**不共享配置，只走语义消息**：
  - 主视图选中一张卡 → 整网图亮起对应层段（`index.html:975` `wireOpToRank`）；
  - 任一格选中某一层 → 另一格同步高亮（`index.html:1000-1009` `wireLinkage` / `:958` `pushHL`）；
  - 走带进度（训练步 t）→ 两格同时收到 `pto:state{step}`（`index.html:1034` `xpPushStep`）。
  - 收消息的判别逻辑在 `index.html:1303` 起，认 `pto:select` / `pto:state-ack` / `pto:comm` /
    `pto:swimlane-select`——最后这个是后来加的（泳道选中 rank 反过来联动两格），
    没跟着上面三条一起写进这份索引的机制说明里，改这块之前先读一遍当前代码，
    不要只信这三条。
- 要把「整网图」抽成独立 pattern 时（README 里提到的后续计划），只需要换这一格的
  `src` 与 `hideCss`（`index.html:493-514`），上面几条语义消息的挂钩不用动。

### 格子「微批次生命周期泳道」（抽屉，`id: swimlane`，4 个 tab）

- `swimlane.html`：**上游拷贝**，源仓库 `github.com/Cinnnnnnndy/compute-graph-viewer`，
  路径 `pangu-moe-trainviz/microbatch-lifecycle-swimlane-mock.html @ 7ca2142`
  （出处写在文件头注释 `swimlane.html:1-22`）。相对上游只有 4 处改动，逐条登记在同一段注释里；
  正文（DOM / 绘制逻辑 / 模拟数据）没有再动过，上游更新时重拷一遍、补这 4 处即可。
- `vendor/swimlane-task/{pattern.css,pattern.js}`：与 `swimlane.html` 同一来源，逐字节一致。
- `embed.css` / `embed-bridge.js`：**本仓原创**，不是上游文件——内嵌桥，只靠 `.click()`
  驱动上游已有控件，不碰上游 IIFE 里的任何变量（约束写在 `embed-bridge.js:1-16`）。
- Tab「通信观测」（`obs`）指向 `./observatory/`，来源另一个上游目录：同一个
  `compute-graph-viewer` 仓库下的 `distributed-communication-observatory-3d-traffic`
  （出处见 `index.html:588-590`）。内部 `observatory/vendor/pto-design-system/` 是钉版快照，
  钉在上游子模块当时指向的 `pto-design-system@d58dd4eb`——升级这份快照要去上游对应仓库找
  子模块新指向的 commit，不能直接拿本仓 `public/vendor/pto-design-system/`（版本不同源）。
- 走带游标（时间轴上那根线，`index.html` 里 `xpPlaceCursor`/`XP_GUTTER`/`XP_PAD_R`
  一带）读的是 `swimlane.html:1573` 里 `const gutter = 236` 这个**上游字面量**的复述，
  改不了也读不到，只能在宿主这边硬编一份同步。上游这个常量变了，这边要跟着改。

---

## 3. 部署步骤的先后顺序（要碰 `deploy.yml` 时对照）

只列与 combo-workbench 直接相关、或顺序上会影响它的步骤，按文件里的出现顺序：

1. `deploy.yml:76` Checkout `main`（站点框架）
2. `deploy.yml:254` Checkout `parallel-topology` 分支 → 实际 `ref: main`（早先完成的同类迁移，见该步注释）
3. `deploy.yml:264` Overlay 并行拓扑 → 产出 `/patterns/net-slicing/pattern.html`，combo-workbench 两格依赖它
4. `deploy.yml:385` Overlay 并行拓扑卡片 into `/launch.html`
5. `deploy.yml:499` Checkout combo-workbench 分支 → 现在也是 `ref: main`（E1 迁移，见本文档第 0 节）
6. `deploy.yml:509` Overlay 组合工作台 at `/combo-workbench/`
7. `deploy.yml:542` Overlay 组合工作台卡片 into `/launch.html`
8. `deploy.yml:576` Checkout `pto-design-system`（跨仓库，供 `/pto-ds/`、`atlas.html` 用，与 combo-workbench 无直接关系，紧接其后）

第 5-7 步新增文件（比如给 combo-workbench 再加一个子目录）时，要同步改这三步——尤其是
第 6 步："逐页列举（不是 `cp -r`）：新增页面必须补到这一步，否则线上 404"
（`deploy.yml:495-498` 原话），`observatory/` 那一条是唯一的例外（整目录 `cp -r`）。

---

## 4. 已知的口径不一致（交接前必须知道，不是本文档臆测，可复现）

`swimlane.html` 里硬编码的训练步事件（`swimlane.html:407-416`）按 **46 层**切 PP 段：

```
PP0 · L0–L11    PP1 · L12–L22    PP2 · L23–L34    PP3 · L35–L45
```

而「并行拓扑」「整网图」两格当前默认配置（`default128` 预设）是 **48 层**：

```
PP0 · L0–L11    PP1 · L12–L23    PP2 · L24–L35    PP3 · L36–L47
```

`L23`、`L34`/`L35` 这几层在上下两部分屏幕上分属不同的 PP 段。工作台的核心卖点是
「同一件事的空间/时间两面对齐着看」，这条错位直接影响这个卖点——打开 `/combo-workbench/`
同屏对比两侧的 PP 段标注即可复现，不需要特殊操作。

这不是新问题：`index.html:472-474` 的注释记着，此前挂在这一格的
`pto-design-system` training-sidecar 正是因为「46 层 vs 48 层，层号对不齐」被换掉的；
换成同源的两格之后，右边两格互相对齐了，但左边泳道那份 46 层的上游拷贝没有跟着改，
矛盾只是从「两格之间」挪到了「舞台和抽屉之间」，没有消失。

修法二选一：把 `swimlane.html` 的硬编码事件改成按 48 层重新切分（会偏离「正文一行没动」
的上游同步纪律，需要在文件头注释里补登记为第 5 处改动）；或者反过来让舞台默认配置切到
46 层。哪个方向由做产品判断的人定，本文档只负责把矛盾点钉清楚。

---

## 5. 加第四格的改动清单

`index.html:435-438` 的注释原话：「按 **N 格** 写：加第三格只需要往 SLOTS 里加一条 +
照抄一节 `<section class="slot">`，下面这些（段控、折叠、关闭、URL 状态）都不用改」。
具体拆开：

**要改的：**
- `var SLOTS = [...]`（`index.html:454`）里加一条新对象：`id` / `el` / `src()` /
  可选的 `key`+`def`+`tabs`（要不要段控）/ 可选的 `foldable`（要不要「收起」按钮）/
  可选的 `hideCss`（要不要藏掉内嵌页自带的某些 UI）。
- HTML 里照抄一节 `<section class="slot" id="...">`（参照 `:326` 或 `:345` 的写法，
  这两行没有随后面的改动挪过）。

**不用改的（通用逻辑，按 N 格写好的）：**
- `readURL`/`writeURL`（`:682`/`:705`）—— URL 状态编解码
- `buildTabs`/`syncTabs`/`select`/`load`（`:724`起）—— 段控与加载
- `wireFrame`（`:775`）/`applySlotVisibility`（`:791`）/`settleFrame`（`:876`）—— 折叠/关闭/加载态判定
- `applyDockH`/`applySideW`（`:803`/`:807`）—— 分隔条拖拽

**如果新格子要跟主视图联动**，接的是既有语义消息（不要接某一套具体 query 参数）：
- `sel`（选中卡）
- `hl`（层高亮，走 `pushHL`，`:958`）
- `step`（训练步进度，走 `xpPushStep`，`:1034`）
- 还有前面第 2 节提到的 `pto:swimlane-select`（泳道选中 rank 反向联动），接哪几条
  取决于新格子答的是什么问题，不是照单全收

新格子要收/发这几条，照抄 `wireLinkage`（`:1000`）或 `xpPushStep`（`:1034`）里已有的
`postMessage({type:'pto:state', ...})` 写法，白名单字段要跟被嵌页面（如果是别的 pattern）
自己认的 `pto:state` 字段对上——参考 `topology`/`netgraph` 两格已经接好的样子。

---

## 6. 相关文档

- [`README.md` 「组合工作台」一节](README.md)——产品向说明，讲「为什么这样切」
- [`CLAUDE.md`](CLAUDE.md)——提交身份规范；「新增分支目录要同步 `deploy.yml`」的仓库级规则
- `.github/workflows/deploy.yml` 顶部大段注释——全站发布策略，覆盖不止 combo-workbench 这一个分支
