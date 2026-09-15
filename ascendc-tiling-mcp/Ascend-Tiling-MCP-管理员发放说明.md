# Ascend Tiling Visualizer MCP 管理员发放说明

## 发放原则

- 每位用户创建独立 Token，禁止分发 `owner` Token，禁止多人共用 Token。
- 默认每日额度为 200 次工具调用；按实际用途调整。
- 安装包和 Token 分开传递。安装包可以公开，Token 必须走一次性秘密链接或企业密码管理器。
- Token 只展示一次。后台只保存 SHA-256 哈希，丢失后不能找回，只能撤销并重建。

## 创建用户

登录腾讯云服务器后运行：

```bash
sudo /usr/local/sbin/ascend-mcp-access create --name "用户标识" --daily-limit 200
```

将输出中的用户 ID 记录在管理员台账中。立即复制 Token，但不要把 Token 写入台账、命令参数、邮件正文、群聊或文档。

## 推荐分发方式

1. 把 `Ascend-Tiling-MCP-macOS.zip` 发给用户。
2. 使用 1Password、Bitwarden Send 或企业密码管理器创建一次性秘密链接。
3. 设置一次查看和 24 小时以内过期。
4. 通过另一个可信渠道发送安装说明，避免安装包和秘密在同一条消息中出现。
5. 用户确认安装成功后，仅记录用户 ID、发放人、发放时间和额度，不记录 Token。

## 撤销与额度

```bash
sudo /usr/local/sbin/ascend-mcp-access revoke 用户ID
sudo /usr/local/sbin/ascend-mcp-access set-limit 用户ID --daily-limit 200
sudo /usr/local/sbin/ascend-mcp-access list
```

员工离开、设备遗失、Token 被误发或日志出现异常来源时，应立即撤销对应用户 ID。不要通过更换所有人的共享 Token 来处理单个用户事件。

