# MindSpeed Config MCP · GitHub Pages 体验包

这个目录可直接作为 GitHub Pages 的发布目录。`index.html` 是自包含文件，不依赖 CDN、
远程 MCP 服务、NPU、CANN 或 MindSpeed 运行环境。

该目录是一个完整、可独立复制的发布单元。需要同步到其他仓库时，请整体复制目录中的
`.nojekyll`、`index.html`、`manifest.json`、`README.md` 和 `SHA256SUMS`，不要只复制页面文件。

## 能体验什么

- 原始 YAML、JSON、Shell 配置解析；
- 并行配置静态校验；
- Rank 拓扑计算与可视化；
- 配置修复候选；
- 标记为 `theoretical` 的理论容量估算。

页面直接调用打包进 HTML 的同一份 `packages/config-core` 纯计算核心。它展示的是当前五项
Tool 的计算与界面能力，但不是 MCP 协议服务器：GitHub Pages 不能提供 `tools/list`、
`tools/call`、鉴权或 MCP Host/App 通信。需要验证这些协议能力时，仍应使用 stdio、
Streamable HTTP 或 MCP Inspector。

## 发布

最简单的方式是把这个目录里的全部文件推送到 GitHub 仓库的发布分支根目录，再在仓库
Settings → Pages 中选择该分支。也可以在 GitHub Actions 中把整个目录作为 Pages artifact
上传。不要只上传 `index.html`：`README.md`、`manifest.json` 和 `SHA256SUMS` 用于说明与核验。

## 重新打包

在项目根目录运行：

```powershell
npm run package:github-pages
```

命令会先生成浏览器本地计算版 `index.html`，写入产品版本、构建清单和 SHA-256，随后恢复
正常的 MCP App 构建产物，避免服务端资源误用静态体验模式。正式腾讯云发布会在发布提交前
自动执行同一套打包，并要求这个目录和服务端使用相同的产品版本。
