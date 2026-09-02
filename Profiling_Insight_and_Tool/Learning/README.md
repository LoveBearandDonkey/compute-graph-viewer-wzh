# Transformer 知识漫游

把本地 Markdown 知识库自动切成适合手机上划复习的随机卡片。

## 启动

```powershell
npm start
```

电脑访问 `http://localhost:4173`。手机和电脑在同一 Wi-Fi 时，访问终端中电脑网卡对应的地址，例如 `http://192.168.1.20:4173`。

也可以直接双击 `public/index.html`，无需启动服务。此方式读取最近一次构建的离线知识快照。

应用每次刷新都会读取原始知识库，因此 Markdown 更新后无需重新生成卡片。若文件移动，可指定新路径：

```powershell
$env:KNOWLEDGE_MD='D:\path\to\knowledge.md'
npm start
```

项目默认从相邻目录 `../ParallelDemo/Transformer结构与并行策略知识库.md` 读取知识库，因此整个 `Profiling_Insight_and_Tool` 目录换位置后仍可正常工作。

学习进度、收藏和掌握状态保存在当前浏览器的 `localStorage` 中。

## 手机离线使用 / 部署

```powershell
npm run build
```

该命令会把最新版知识库生成到 `public/cards.json`。将 `public` 目录发布到任意 HTTPS 静态站点后，可以在手机上“添加到主屏幕”，首次打开后支持离线复习。以后源文档更新，重新执行构建并发布即可。
