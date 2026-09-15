# Ascend Tiling Visualizer Cloud

这是面向远程 Codex agent 的 MCP 服务包。它不提供独立网站；MCP App 资源随 MCP 服务返回。

## 启动

需要 Node.js 20+，并设置 MCP_AUTH_TOKEN：

```bash
MCP_AUTH_TOKEN="仅在服务器环境中设置" MCP_HOST=127.0.0.1 MCP_PORT=3000 ./start.sh
```

健康检查：`GET /healthz`。MCP 地址：`POST /mcp`。生产环境请在前面配置 HTTPS 反向代理。

原项目说明：

# Ascend C Tiling Visualizer

一个本地 Codex 插件和 MCP App，用于从 Ascend C / C++ 源码中静态恢复 Vector、MatMul，以及 FlashAttention Score Grad（Deter）的核间与核内 Tiling。

它回答五个问题：

- 全局 Tensor 是什么形状；
- 启用了多少个核；
- 每个核负责哪个范围或哪些输出 Tile；
- 选中核内部如何沿 X 或 M/N/K 继续分块；
- 每项结论来自哪段源码，哪些信息仍然未知。

## 边界

