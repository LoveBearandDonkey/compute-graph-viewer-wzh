# hpc-topology-viewer

## 提交身份（对整个仓库、所有子项目一律生效）

本仓库的每一条 commit 都必须以仓库主人的身份提交，而不是 Claude：

```
user.name  = Cindy_wxd
user.email = 209322477+Cinnnnnnndy@users.noreply.github.com
```

`.claude/hooks/set-git-identity.sh` 会在每次会话开始时自动把这两项写进
`.git/config`（仓库级配置优先于云端沙箱的全局 `~/.gitconfig`，后者写的是
`Claude <noreply@anthropic.com>`）。

**如果那个 hook 因为任何原因没跑到**（hook 被禁用、在别的工具里操作、CI 里提交），
在 commit 之前先手动执行一次：

```bash
git config --local user.name  "Cindy_wxd"
git config --local user.email "209322477+Cinnnnnnndy@users.noreply.github.com"
git config --local commit.gpgsign false
```

提交前用 `git config user.email` 确认一遍；提交后用
`git log -1 --pretty='%an <%ae>'` 复核，出现 `noreply@anthropic.com` 说明配置没生效，
改完要重新提交（`git commit --amend --reset-author`）。

不要在 commit message 里写 `Co-Authored-By: Claude ...` 或 `Claude-Session: ...`
尾注，也不要在 PR 描述里加 Claude Code 的署名——`.claude/settings.json` 里的
`attribution` 已经关掉了它们。

## 部署

`.github/workflows/deploy.yml` 从 main 构建站点框架，再把各 pattern 分支的自包含
静态子目录叠加上去一起发布到 GitHub Pages。**新增分支目录时必须同步更新那份
workflow 的 `on.push.branches` 和叠加步骤**，否则下一次从 main 发布会把它抹掉。
