# 这是一份源码镜像，不是独立维护的模块

本目录是 [Cinnnnnnndy/hpc-topology-viewer](https://github.com/Cinnnnnnndy/hpc-topology-viewer)
`main` 分支的源码快照，同步自 commit `013bd81c3ffacdf7a840f2458b2018b63586da2e`
（2026-09-21）。

- **线上入口以源仓库的 GitHub Pages 为准**：<https://cinnnnnnndy.github.io/hpc-topology-viewer/launch.html>
  —— 这里的静态文件是原始源码，不是构建产物，直接打开本目录下的 `index.html` /
  `public/*.html` 不会正常工作（工作台主体是 Vite + React，需要先构建；启动页里
  标 `°` 的几张 pattern 卡是源仓库 CI 在发布时从其他分支叠加生成的，源码树里
  本来就没有对应文件）。
- **本地想跑起来**：`npm install && npm run build`，产物在 `dist/`。
- **持续迭代仍然发生在源仓库**，不在这里。这份镜像是一次性同步的副本，不会跟着
  源仓库自动更新——要刷新，重新拉取源仓库对应 commit 并覆盖本目录即可，不必
  另开工作。

保留这份源码镜像是为了方便在本仓库里直接读代码、按需参考实现，而不必跳出去看
另一个仓库；真正要点开、要分享给别人看的链接，请用上面的 Pages 地址。
