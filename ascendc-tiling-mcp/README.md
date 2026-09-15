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

## 这个 MCP 的作用

Ascend Tiling Visualizer MCP 用于把 Ascend C / C++ 算子源码中的 Tiling 逻辑转化为可理解、可交互的可视化结果。它帮助用户从源码和已知参数中回答：

- 全局 Tensor 的形状、数据范围和切分关系是什么；
- 使用了多少个核，以及不同核分别处理哪些数据范围或输出 Tile；
- 选中某个核后，核内如何沿 X、M、N、K 或其他语义轴继续分块；
- block、tile、循环边界和数据搬运之间如何对应；
- 哪些结论直接来自源码，哪些是推导结果、输入假设或仍然未知。

调用完成后，MCP 会返回带有来源和证据状态的分析结果，并尝试在宿主中显示关联的 Tiling MCP App。它适合用于理解算子结构、检查 Tiling 参数与 kernel 逻辑的对应关系，以及辅助定位边界和切分问题。

它不是编译器、运行器或性能分析器：不会替用户编译、执行算子，也不会凭静态源码给出真实性能、核利用率或搬运重叠结论。涉及真实运行行为时，仍需要结合 CANN 环境、运行日志和 profiling 数据验证。

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

## 使用方法

安装完成并重启 Codex 后，新建任务。用户可以提供以下任意一种输入，Codex 会先读取或整理输入，再调用云端 MCP 的 `visualize_ascend_tiling` 工具：

### 方式一：提供本地算子文件

直接告诉 Codex 文件路径，并说明希望查看的内容，例如：

```text
请读取本地文件 /path/to/my_operator/kernel/add_kernel.cpp 和相关的 tiling 文件，调用 ascend_tiling_visualizer_cloud 的 visualize_ascend_tiling 工具，分析这个算子的核间和核内 Tiling，并显示 MCP App。
```

如果算子依赖多个头文件、host 侧 tiling 文件或配置文件，应同时提供这些文件的路径，避免只分析单个 kernel 文件而缺少上下文。

### 方式二：提供代码链接

提供 GitHub、GitCode、企业 Git 或原始文件链接，并说明需要分析的文件或代码范围，例如：

```text
请读取这个链接中的 Ascend C 算子源码：<代码链接>
找到 kernel、host tiling 和相关参数定义后，调用 ascend_tiling_visualizer_cloud 的 visualize_ascend_tiling 工具，分析核间映射、核内分块和边界处理，并显示 MCP App。
```

链接需要对当前 Codex 环境可访问。如果链接需要登录、位于内网或无法读取，可以改用本地文件或直接粘贴代码。

### 方式三：直接粘贴代码片段

直接粘贴 kernel、tiling 或 host 侧代码，并标注语言和文件角色，例如：

~~~~text
请使用 MCP 服务器 ascend_tiling_visualizer_cloud，实际调用 visualize_ascend_tiling 工具分析下面的 Ascend C kernel。请说明核间切分、核内 Tile、数据搬运范围和无法从代码确认的部分，并显示 MCP App。

文件角色：kernel
语言：ascendc

```cpp
auto offset = GetBlockIdx() * blockLength;
DataCopy(xLocal, xGm[offset], tileLength);
Add(y, x, z, tileLength);
```
~~~~

如果已知 `shape`、`dtype`、`format`、`blockDim`、`tileLength` 或其他 host 侧 tiling 参数，也应一并提供。参数不完整时，MCP 会保留公式或标记为假设、未知，不应把补充的可视化参数误认为真实运行值。

### 推荐调用要求

为了确认使用的是云端服务，而不是本地版本，提示词中应明确写出：

```text
请实际调用 MCP 服务器 ascend_tiling_visualizer_cloud 的 visualize_ascend_tiling 工具，不要使用本地的 ascend-tiling-visualizer。完成后告诉我实际调用的服务器名、工具名，以及 MCP App 是否成功显示。
```

只要求 Codex“列出可用工具”不一定会产生实际工具调用日志。要验证服务是否真正被使用，应提交一次本地文件、代码链接或代码片段，让 Codex 执行 Tiling 分析。

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
