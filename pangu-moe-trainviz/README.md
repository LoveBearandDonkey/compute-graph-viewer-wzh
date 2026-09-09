# Pangu MoE TrainViz

本目录是 PTO 的训练可视化产品区。Git 只保留正式产品入口、内容页、共享实现和需要协作维护的产品文档。

## 正式产品页

- [`index.html`](./index.html)：TrainScope 训练透视入口。
- [`op-rank-time.html`](./op-rank-time.html)：openPangu-R-72B-MoE 算子、硬件与时间视图。
- [`op-rank-time-openpangu-flash.html`](./op-rank-time-openpangu-flash.html)：openPangu 2.0 Flash 训练可视化。
- [`op-rank-time-openpangu-flash-events.html`](./op-rank-time-openpangu-flash-events.html)：OpenPangu 并行事件预览。
- [`trainscope-live.html`](./trainscope-live.html)：实时训练监护页。
- [`ep-expert-parallel-2d.html`](./ep-expert-parallel-2d.html)：EP Expert Parallel 2D 视图。
- [`training-object-topology-ide.html`](./training-object-topology-ide.html)：训练对象拓扑工作台。
- [`分布式泳道/training-structure-time-v3.html`](./分布式泳道/training-structure-time-v3.html)：DeepSeek V4 训练结构与生命周期泳道，支持双向关联和确定性模拟时序。

## 内容页

- [`knowledge-whitepaper.html`](./knowledge-whitepaper.html)：并行放置知识白皮书。
- [`communication-operator-visual-whitepaper.html`](./communication-operator-visual-whitepaper.html)：并行通信可视化白皮书。

## 文档

- [`docs/`](./docs/)：PRD、工程规格与可复用的并行知识说明。

个人核对笔记、视觉实验、协作临时页和历史副本不纳入 Git；需要时存放在本机的 `.local-*` 目录，并由 `.git/info/exclude` 排除。

## 共享实现

- `css/`、`js/`、`data/`：产品页和实验页共用的实现与数据。
- `vendor/`：本模块本地化的第三方依赖。
- `pangu-palette.js`：多个正式页面共用的语义色板。
- `shared.css`：白皮书内容页共用样式。

从 PTO 根目录启动本地服务：

```sh
cd /Users/yin/pto
python3 -m http.server 8765
```

页面地址统一为 `http://127.0.0.1:8765/pangu-moe-trainviz/...`。
