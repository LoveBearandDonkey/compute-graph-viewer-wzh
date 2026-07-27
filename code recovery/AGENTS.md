# AGENTS.md

## 模块说明

这是 Ascend Tiling Visualization Workbench，不是独立项目。

## 工作区根目录

完整项目根目录是 `/Users/yin/pto`。

## 入口

- `tiling/index.html`
- 辅助页面：`tiling/matmul-tiling.html`、`tiling/iso3d-preview.html`

本目录里的白皮书页面是内容页；除非明确要求，不要加入白皮书专项流程。

## 运行方式

必须从 PTO 根目录启动本地服务，不能从当前文件夹启动：

```sh
cd /Users/yin/pto
python3 -m http.server 8765
```

打开 `http://127.0.0.1:8765/tiling/index.html`。

## 共享依赖

- `../vendor/pto-design-system/`
- 本地 `src/`、`data/fixtures/`、`data/schemas/`、`data/sources/`

## 给其他 Agent 的规则

- 不要把 `tiling/` 当作独立项目。
- 修改 UI 或 trace 前，先读 `tiling/CLAUDE.md` 里的详细模块规则。
- 保持 trace fixtures 和 `src/app.js` 行为一致。
- 修改后通过根目录服务 URL 验证。
