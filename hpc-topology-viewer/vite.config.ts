import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';

// research/ 顶层的研究报告 HTML 随站点一起发布到 /research/*.html（launch.html 的
// 「研究报告与原型」区链接它们）。目录本身留在仓库根——那里还有 md / json / 会话档案，
// 不适合整体搬进 public/，也不在仓库里复制第二份：dev 时按需从磁盘直读，build 时拷进
// dist/research/（只取顶层 .html，local_* 会话档案不发布）。
function researchHtml(): Plugin {
  const srcDir = fileURLToPath(new URL('./research', import.meta.url));
  // 参考他人资料的报告不对外发布（launch 页也不放它们）：
  const EXCLUDED = new Set(['昇腾超节点-概念可视化报告.html', '昇腾超节点-全栈整合架构图.html']);
  const pages = () => readdirSync(srcDir).filter((f) => f.endsWith('.html') && !EXCLUDED.has(f));
  return {
    name: 'research-html',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const m = /\/research\/([^/?#]+\.html)$/.exec(decodeURIComponent(req.url ?? ''));
        const file = m && !EXCLUDED.has(m[1]) && join(srcDir, m[1]);
        if (!file || !existsSync(file)) return next();
        res.setHeader('Content-Type', 'text/html; charset=utf-8');
        res.end(readFileSync(file));
      });
    },
    closeBundle() {
      const outDir = fileURLToPath(new URL('./dist/research', import.meta.url));
      mkdirSync(outDir, { recursive: true });
      for (const f of pages()) copyFileSync(join(srcDir, f), join(outDir, f));
    },
  };
}

// base is set for project-pages hosting (https://<user>.github.io/hpc-topology-viewer/).
// Override with VITE_BASE=/ for root hosting or local preview.
export default defineConfig({
  base: process.env.VITE_BASE ?? '/hpc-topology-viewer/',
  plugins: [react(), researchHtml()],
  // 每次构建生成唯一 id，用于给非哈希静态资源（如 iframe 里的 cube-cockpit.html）做缓存刷新，
  // 避免部署后浏览器/CDN 仍加载旧的 iframe 内容。
  define: { __BUILD_ID__: JSON.stringify(String(Date.now())) },
  build: {
    rollupOptions: {
      input: new URL('./index.html', import.meta.url).pathname,
    },
  },
});
