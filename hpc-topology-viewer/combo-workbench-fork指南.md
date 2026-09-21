# 组合工作台 · Fork 修改指南

写给**不在本仓库有提交权限、想 fork 一份自己改**的人。产品向说明看
[README.md「组合工作台」一节](README.md)；本仓维护者接手开发看
[`combo-workbench-开发引用手册.md`](combo-workbench-开发引用手册.md)——两份不重复，
这份只讲「fork 出去之后怎么跑起来、怎么改、怎么发布」。

核对时间：2026-08-27。

---

## 0. 先回答「代码是不是都在 main」

**是。** `/combo-workbench/` 用到的全部代码，包括它舞台两格实际嵌的那个 pattern，
现在都在 `main` 分支上，不用去找别的分支：

- `public/combo-workbench/` —— 台面本体（`index.html`、`swimlane.html`、
  `observatory/`、`vendor/swimlane-task/`、`launch-card.html` 等）。
- `public/parallel-topology/demo.html` —— 舞台「并行拓扑」「整网图」两格共同嵌的
  `/patterns/net-slicing/pattern.html` 就是从这个文件产出的，同样在 `main` 上。
- 两格用到的渲染器 `public/vendor/model-graphviz/`、`public/vendor/net-sharding/`、
  `public/vendor/pto-design-system/`、`public/vendor/three-r128.min.js` 等，也都在
  `public/vendor/` 下，随 `main` 一起在。

这是 2026-08 做的一次迁移（`combo-workbench-开发引用手册.md` 里叫 "E1 迁移"）的结果：
`/combo-workbench/` 和它依赖的 `/parallel-topology/`，`.github/workflows/deploy.yml`
里对应的 checkout 步骤都已经改成 `ref: main`，不再单独指向某条专属分支。

**跟 combo-workbench 无关、还没合并进 main 的东西**（fork 之后如果不碰这些，可以完全
不管）：
- 逻辑魔方相关的几个 pattern（`/rubik/`、`/rubik-spec/`、`/rubik-netobj/`、
  `/rank-card/`）还各自躺在专属分支上（`claude/rubik-view-pattern-extraction-7pk0qt`
  等），走同一份 `deploy.yml` 里另外几步 checkout 叠加发布。
- `/pto-ds/` 和 `/parallel-topology/atlas.html` 用到的「纠正版 rank-deck」来自**另一个
  仓库** `Cinnnnnnndy/pto-design-system` 的 `claude/rank-deck-intersection-payload`
  分支（公开仓库，只读 checkout）。combo-workbench 的三个格子都不引用它。

所以：**如果你只想 fork 改 `/combo-workbench/`，只需要 `main` 这一个分支，不需要去拉
任何 `claude/*` 分支。**

---

## 1. Fork 与本地跑起来

1. GitHub 网页上 Fork `cinnnnnnndy/hpc-topology-viewer` 到自己账号下。
2. `git clone` 自己的 fork，`cd` 进去。
3. ```bash
   npm install
   npm run dev        # http://localhost:5173
   ```

**本地能看到什么、看不到什么**（这点容易踩坑，README 里也提过）：

- 台面外壳（顶栏、分隔条、折叠/展开、URL 状态）、抽屉「微批次生命周期泳道」和
  「通信观测」两个 tab —— 都是自包含文件，本地能看到真实效果。
- 舞台「并行拓扑」「整网图」两格 —— **是构建期产物**。它们 iframe 指向
  `/patterns/net-slicing/pattern.html`，这个文件是发布流水线从
  `public/parallel-topology/demo.html` 现场拼出来的（补版本戳、拍平 vendor 路径），
  仓库里没有这份现成文件，`npm run dev` 直接开 `public/combo-workbench/index.html`
  时这两格是空的（有兜底文案，不会报错），外壳其它部分不受影响。
  要看这两格的真实效果，只能看**发布之后的站点**（见第 2 节），或者本地跑一遍
  `npm run build` 后自己模拟 `deploy.yml` 里 "Overlay 并行拓扑" 那一步的产出路径。

---

## 2. 只想改 combo-workbench、发到自己的 GitHub Pages 上看效果

在自己 fork 的仓库里：

1. `Settings → Pages → Source` 选 `GitHub Actions`（`deploy.yml` 里
   `enablement: true` 会在第一次跑的时候自动开，不用手动点也行，跑一次就有了）。
2. 直接改 `public/combo-workbench/` 下的文件，`git add / commit / push` 到自己 fork
   的 `main` 分支——`.github/workflows/deploy.yml` 监听的是 `push: branches: [main]`，
   推自己 fork 的 `main` 会触发**你自己 fork 里的这份 workflow**，产出**你自己的**
   GitHub Pages 站点，跟原仓库线上的站点毫无关系，改坏了也不会影响任何人。
3. `deploy.yml` 里有一步会 checkout 另一个仓库
   `Cinnnnnnndy/pto-design-system`（供 `/pto-ds/`、`atlas.html` 用）——这行写死了
   `repository: Cinnnnnnndy/pto-design-system`，fork 之后**这一步依然会去读原仓库
   那个公开分支**，不会因为你 fork 了 `hpc-topology-viewer` 就跟着变。这是只读
   checkout，对你改 combo-workbench 没有影响；只有你要改 `/pto-ds/` 那部分内容时
   才需要单独去 fork 那个仓库。
4. 第一次跑这份 workflow 如果 build 成功、deploy 那一步秒失败且没有日志，去
   `Settings → Environments → github-pages → Deployment branches and tags`
   把 `main`（或你实际推送的分支）加进允许列表——这是 GitHub Pages 环境的保护规则，
   跟 workflow 本身对不对没关系。

---

## 3. 只是想改完提 PR 回原仓库，不需要自己发布一份站点

不用碰 Pages 设置，正常流程即可：

1. 在自己 fork 里建一个开发分支（不要直接改 fork 的 `main`，方便对齐上游）。
2. 改 `public/combo-workbench/` 下的文件，`commit` + `push` 到这条分支。
3. 在 GitHub 上从 `你的fork:你的分支` 向 `cinnnnnnndy/hpc-topology-viewer:main`
   开 PR。
4. 原仓库的 `deploy.yml` 只在**原仓库自己的 `main` 分支被 push**（包括 PR 合并）时
   触发发布，你在自己 fork 上的推送、甚至开 PR 本身，都不会碰到原仓库线上的站点——
   合并之前线上内容不会有任何变化。

**提交身份**：本仓库 `CLAUDE.md` 里要求所有 commit 用仓库主人的 GitHub 身份提交，
那是**这个仓库自己的约定**，管的是这个仓库的提交历史；fork 到你自己名下之后，
用你自己的 git 身份正常提交、正常开 PR 即可，不需要照抄那段配置。

---

## 4. 改之前值得先看的几个位置

- `public/combo-workbench/index.html` 里的 `var SLOTS = [...]`（约 510 行起）——
  三个格子（`netgraph` / `topology` / `swimlane`）的定义都在这，每个格子的 `src()`
  函数、`hideCss`、要不要段控（`key`/`def`/`tabs`）、要不要折叠（`foldable`）
  都在这一份对象字面量里，加第四格照抄一条 + HTML 里照抄一节 `<section class="slot">`
  即可，URL 状态编解码 / 段控 / 折叠 / 分隔条拖拽这些通用逻辑不用碰。
- 两格之间、以及未来新格子之间**只走语义消息**（`pto:select` 选中卡、
  层高亮走 `pushHL`、训练步进度走 `xpPushStep`、泳道选中 rank 反向联动走
  `pto:swimlane-select`），不共享 query 参数——想接哪几条消息取决于新格子回答
  什么问题，不是照单全收。
- `public/combo-workbench/swimlane.html` 与 `vendor/swimlane-task/`
  是 `compute-graph-viewer` 仓库的**上游逐字节拷贝**，正文不要改，只在文件头
  注释里登记差异；升级时重拷一遍上游、补回文件头列的那几处改动。
- 已知有一处口径不一致还没修：泳道按 46 层切 PP 段，舞台两格默认配置是
  48 层，`L23`/`L34` 附近的 PP 段标注在两边对不上——细节和两个候选修法见
  `combo-workbench-开发引用手册.md` 第 4 节，这份指南不重复。

---

## 5. 相关文档

- [`README.md`「组合工作台」一节](README.md) —— 产品向说明：三格是什么、为什么这样切
- [`combo-workbench-开发引用手册.md`](combo-workbench-开发引用手册.md) —— 面向本仓
  维护者的索引文档：每部分代码在哪、由 `deploy.yml` 哪一步发布、部署步骤顺序
- [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) 顶部大段注释 ——
  全站发布策略，覆盖不止 combo-workbench 这一个分支/目录
