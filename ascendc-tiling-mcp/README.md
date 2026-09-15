# Ascend Tiling Visualizer MCP

这是 Ascend C Tiling Visualizer 的远程 MCP 分发与部署目录。它包含：

- macOS 用户一键安装包；
- 管理员创建、分发、撤销用户 Token 的说明；
- 当前部署到腾讯云的 Streamable HTTP MCP 服务包。

## 服务信息

| 项目 | 内容 |
| --- | --- |
| MCP 名称 | `ascend_tiling_visualizer_cloud` |
| MCP 地址 | `https://42.192.111.92/mcp` |
| 健康检查 | `https://42.192.111.92/healthz` |
| 传输方式 | Streamable HTTP |
| 认证方式 | 每用户独立 Bearer Token |
| MCP 工具 | `visualize_ascend_tiling` |

服务返回 Tiling 分析结果以及关联的 MCP App。它是静态源码分析和可视化服务，不会编译或执行用户提交的 Ascend C 代码。

## 文件结构

```text
tiling-visualization/
├── Ascend-Tiling-MCP-macOS.zip
│   └── 给普通用户分发的 macOS 一键安装包和安装说明
├── Ascend-Tiling-MCP-管理员发放说明.md
│   └── 管理员创建、分发、撤销 Token 的操作说明
├── ascend-tiling-visualizer-cloud-upload/
│   └── 当前部署到腾讯云的文件快照
│       ├── server.js
│       ├── tiling-app.html
│       ├── manage-access.mjs
│       ├── start.sh
│       └── package.json
└── README.md
```

`ascend-tiling-visualizer-cloud-upload/` 是当前云端对应的部署包；不要使用旧的 `ascend-tiling-visualizer-cloud` 目录。云端容器中的 `/app` 文件已经与这个部署包逐文件哈希一致。

## 管理员发放用户

登录腾讯云服务器后，为每个人创建独立访问密钥：

```bash
sudo /usr/local/sbin/ascend-mcp-access create --name "用户标识" --daily-limit 200
```

注意：

- Token 只会显示一次，后台只保存 SHA-256 哈希；
- 不要把 Token 写进文档、台账、命令参数、邮件、群聊或 Git；
- 不要把 `owner` Token 发给其他用户；
- 安装包和 Token 通过两个不同渠道发送；
- 推荐使用一次查看、短有效期的 1Password、Bitwarden Send 或企业密码管理器链接。

详细流程见 [Ascend-Tiling-MCP-管理员发放说明.md](Ascend-Tiling-MCP-管理员发放说明.md)。

## 普通用户安装

普通用户只需要收到：

1. `Ascend-Tiling-MCP-macOS.zip`；
2. 个人 Token，通过单独的安全渠道发送。

用户解压后双击 `setup-ascend-tiling-mcp-macos.command`。安装程序会：

- 通过 macOS `security` 提示隐藏输入 Token；
- 将 Token 保存到当前用户的系统钥匙串；
- 验证 HTTPS 服务和 Token；
- 安装 `http_headers_helper`；
- 备份并更新 `~/.codex/config.toml`；
- 提示用户完全退出并重新打开 Codex。

Token 不会写入 Codex 配置文件，也不会写入安装包。

当前安装包面向 macOS 上的 Codex Desktop、Codex CLI 和 Codex IDE 扩展。ChatGPT 网页版不会读取用户电脑上的本地 Codex MCP 配置，因此不适用这套安装方式。

## 用户测试

安装完成并重启 Codex 后，新建任务，要求实际调用云端 MCP：

```text
请只使用 MCP 服务器 ascend_tiling_visualizer_cloud，实际调用它的 visualize_ascend_tiling 工具。不要使用本地的 ascend-tiling-visualizer。

请用以下输入生成 Tiling 可视化：
sources:
- role: kernel
  language: ascendc
  content: |
    auto offset = GetBlockIdx() * blockLength;
    DataCopy(xLocal, xGm[offset], tileLength);
    Add(y, x, z, tileLength);

context:
  tilingValues:
    blockDim: 2
    totalLength: 256
    blockLength: 128
    tileLength: 64

完成后明确告诉我实际调用的 MCP 服务器名、工具名，以及 MCP App 是否成功显示。
```

用户应看到工具调用结果和关联的 Tiling MCP App。仅仅让 Codex“列出可用工具”不一定会产生工具调用日志；要验证服务是否真正被使用，应执行一次 `visualize_ascend_tiling`。

## Token 管理

```bash
# 撤销用户
sudo /usr/local/sbin/ascend-mcp-access revoke 用户ID

# 恢复用户
sudo /usr/local/sbin/ascend-mcp-access enable 用户ID

# 修改每日额度
sudo /usr/local/sbin/ascend-mcp-access set-limit 用户ID --daily-limit 200

# 查看用户状态、额度和创建时间
sudo /usr/local/sbin/ascend-mcp-access list
```

设备遗失、员工离开、Token 被误发或怀疑泄露时，只撤销对应的用户 ID，再为该用户创建新 Token。不要为了处理单个用户事件而更换所有人的凭证。

## 查看云端日志

在腾讯云服务器上查看最近日志并持续跟踪：

```bash
sudo docker logs --since 10m -f ascend-tiling-mcp
```

日志会记录认证拒绝、配额拒绝和工具调用接受等审计事件，不应包含完整 Token 或源码内容。

## 更新部署包

当前目录中的 `ascend-tiling-visualizer-cloud-upload` 是部署快照，不是完整开发源码。更新流程应是：

```text
Ascend Tiling Visualizer 源码
        │
        ├─ npm run build:deploy
        ▼
ascend-tiling-visualizer-cloud-upload
        │
        ├─ 上传 server.js、tiling-app.html、manage-access.mjs 等文件
        ▼
腾讯云容器 /app
```

不要直接手改云端 `server.js` 或 `tiling-app.html` 后再把它当作源码。更新前应保留当前部署包和云端备份，并在健康检查、MCP 工具调用和 MCP App 显示三项都验证通过后再通知用户。

## 安全边界

- 当前服务是“对外可访问、Token 控制”的邀请制服务，不是匿名公开服务；
- 每用户有独立速率和每日额度，仍应关注腾讯云流量、CPU、内存和日志异常；
- 不要在客户端配置中使用静态 `http_headers` 保存 Token；
- 不要关闭 HTTPS 证书校验；
- 正式扩大用户规模前，建议绑定正式域名和受信任的 TLS 证书；
- 用户数量较大时，建议把当前 Bearer Token 方案升级为 OAuth，减少人工分发和轮换凭证的成本。

