# Vendored: pto-design-system patterns（memory-architecture + aic/aiv-core-object + tokens）

来源：https://github.com/yinyucheng0601/pto-design-system （main 分支，2026-07 快照）

- `memory-architecture/pattern.{js,css,json}` — L0 Core-Group 芯片内架构 shell：
  Global Memory / L2 Cache 轨道 + AIV1 / AIC / AIV2 核组对象 + MTE/CV 路由 overlay +
  hover / path-focus / buffer-block / 缩放平移 API（`window.PtoMemoryArchitecturePattern`）。
- `../aic-core-object/pattern.{js,css}` — AIC（Cube）核对象渲染器（`window.PtoAicCorePattern`）。
- `../aiv-core-object/pattern.{js,css}` — AIV（Vector）核对象渲染器（`window.PtoAivCorePattern`）。
- `../pto-tokens/{foundation,semantic,components}.css` — pattern 依赖的设计 token（仅 `:root` 变量，无全局副作用）。

接入方式与 `../workbench-shell` 相同：side-effect import 注册全局 API；
React 封装见 `src/view/CoreGroupPattern.tsx`（所有视图的 L0 层级统一走它）。
按 pattern.json 的 agentReuseRule：不要复制渲染出来的 DOM、不要在页面本地改路由几何/配色，
扩展 preset / 路由数据请改 pattern.js。
