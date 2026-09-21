/**
 * 把并行拓扑 PRD 的 markdown 渲染成一张自包含的 HTML 页。
 *
 * 为什么要有这个脚本、而不是手改 HTML：
 *   PRD 会改，而发出去的链接不能变。源文件是 `docs/parallel-topology-prd.md`，
 *   `public/parallel-prd/index.html` 是它的构建产物——改 md、重跑本脚本、提交两者，
 *   链接内容就更新了。手改 HTML 的话，md 与页面会立刻分叉，之后没人说得清哪份是准的。
 *
 * 为什么源文件在 docs/ 而不是 public/：
 *   public/ 下的东西都会原样发布。md 跟着发出去就等于同一份内容有两条链接，
 *   别人拿到哪条全看运气。源留在 docs/，发布出去的只有渲染好的那一张页。
 *
 * 为什么不引 markdown 运行时到页面里：
 *   这条链接要能单文件打开（发给别人、存本地、塞进任何静态托管都一样），
 *   所以 markdown 在构建期就吃掉，产物里只有 HTML + 内联 CSS/JS，零外部依赖。
 *
 * 视觉语言与《分布式训练参照系》那条链接（/parallel-reference/）同源——同一套
 * token、同一套版心与章节标尺。两条链接各自独立，但是一对文档，不该长成两种东西。
 *
 * 用法：
 *   1) 装一次 marked：`npm i -D marked`（只在生成时需要，不进运行时依赖）；
 *   2) `node scripts/build-prd-page.mjs docs/parallel-topology-prd.md public/parallel-prd/index.html`
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { marked } from 'marked';

const SRC = process.argv[2] || 'docs/parallel-topology-prd.md';
const OUT = process.argv[3] || 'public/parallel-prd/index.html';
const md = readFileSync(SRC, 'utf8');

marked.setOptions({ mangle: false, headerIds: false, gfm: true });

/* ---- 拆成「masthead + 若干 section」，而不是一长条 ---- */
const lines = md.split('\n');
const firstSec = lines.findIndex((l) => /^## /.test(l));
const head = lines.slice(0, firstSec).join('\n');
const rest = lines.slice(firstSec).join('\n');

// masthead：# 标题 / **版本** / **依据** / **范围假设**
const title = (head.match(/^#\s+(.+)$/m) || [, 'PRD'])[1].trim();
const metaLines = head
  .split('\n')
  .filter((l) => /^\*\*/.test(l.trim()))
  .map((l) => l.trim());
const sub = metaLines.find((l) => /范围假设/.test(l)) || '';
const meta = metaLines.filter((l) => l !== sub);

const inline = (s) => marked.parseInline(s.replace(/^\*\*(.+?)\*\*\s*/, '<b>$1</b> '));

/* ---- 分节：`## N. 标题` / `## 附录 X · 标题` → <section id="sN"> ---- */
const chunks = rest.split(/\n(?=## )/).filter((c) => c.trim());
const rail = [];
const sections = chunks.map((chunk, i) => {
  const m = chunk.match(/^## +(.+)/);
  const raw = m ? m[1].trim() : `S${i}`;
  const num = (raw.match(/^(\d+|附录 [A-Z])/) || [, String(i)])[1];
  const label = raw.replace(/^(\d+\.|附录 [A-Z] ·)\s*/, '').trim();
  const id = 's' + num.replace(/[^0-9A-Za-z]/g, '');
  const numeric = /^\d+$/.test(num);
  rail.push({ id, num: numeric ? num.padStart(2, '0') : num.replace('附录 ', 'App '), label });

  const body = chunk
    .replace(/^## .+\n/, '')
    .replace(/^---\s*$/gm, '') // md 里的分隔线由 section 边界代劳
    .trim();

  let html = marked.parse(body);
  // 表格套上横向滚动壳：窄屏上宽表格滚动，而不是撑破版心
  html = html.replace(/<table>/g, '<div class="tblbox"><div class="tblscroll"><table class="tbl">');
  html = html.replace(/<\/table>/g, '</table></div></div>');
  // 引用块 → 概念图里的 note 样式
  html = html.replace(/<blockquote>/g, '<div class="note">').replace(/<\/blockquote>/g, '</div>');

  return `<section id="${id}">
  <div class="sec-head"><span class="sec-num">${numeric ? num.padStart(2, '0') : num}</span><h2>${inline(label)}</h2></div>
${html}
</section>`;
});

const CSS = `
:root{
  --bg:#EEF1F3; --paper:#FFFFFF; --ink:#0E1418; --ink-2:#3D4B55; --ink-3:#6B7B86;
  --rule:#D2DADF; --rule-2:#E4EAEE;
  --dp:#2E6E8E; --pp:#B3603A; --cp:#3E7D5E; --tp:#A8404A; --ep:#6A5A9B;
  --mono: ui-monospace, "SFMono-Regular", "JetBrains Mono", Menlo, Consolas, monospace;
  --sans: -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB", "Source Han Sans SC", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:var(--sans); font-size:15.5px; line-height:1.78;
  -webkit-font-smoothing:antialiased;
  background-image:
    linear-gradient(115deg, rgba(14,20,24,.022) 1px, transparent 1px),
    linear-gradient(65deg, rgba(14,20,24,.022) 1px, transparent 1px);
  background-size:34px 20px, 34px 20px;
}
.wrap{max-width:1120px; margin:0 auto; padding:0 28px 140px}

header{padding:74px 0 30px; border-bottom:2px solid var(--ink)}
.eyebrow{font-family:var(--mono); font-size:11.5px; letter-spacing:.22em; color:var(--ink-3); text-transform:uppercase}
h1{margin:16px 0 10px; font-size:clamp(30px,4.6vw,50px); line-height:1.1; font-weight:800; letter-spacing:-.022em}
.sub{color:var(--ink-2); max-width:62ch; margin:0}
.meta{margin-top:22px; display:flex; flex-wrap:wrap; gap:8px 22px; font-family:var(--mono); font-size:11.5px; color:var(--ink-3)}
.meta b{color:var(--ink-2); font-weight:700}

.pair{
  margin-top:26px; display:inline-flex; align-items:center; gap:10px;
  font-family:var(--mono); font-size:11.5px; color:var(--ink-3);
  background:var(--paper); border:1px solid var(--rule); border-radius:5px; padding:9px 14px;
}
.pair a{color:var(--dp); font-weight:700; text-decoration:none; border-bottom:1px solid transparent}
.pair a:hover{border-bottom-color:var(--dp)}

section{padding-top:64px; scroll-margin-top:20px}
.sec-head{display:flex; align-items:baseline; gap:16px; border-bottom:1px solid var(--ink); padding-bottom:9px; margin-bottom:26px}
.sec-num{font-family:var(--mono); font-size:12px; font-weight:600; color:var(--ink-3); letter-spacing:.08em; white-space:nowrap}
h2{margin:0; font-size:clamp(20px,2.5vw,26px); font-weight:750; letter-spacing:-.012em}
h3{margin:38px 0 12px; font-size:17.5px; font-weight:700; letter-spacing:-.005em}
h4{margin:26px 0 8px; font-size:15px; font-weight:700; color:var(--ink-2)}
p{margin:0 0 15px; max-width:74ch}
ul,ol{margin:0 0 15px; padding-left:20px; max-width:74ch}
li{margin-bottom:7px}
code{font-family:var(--mono); font-size:.885em; background:#E2E8EC; padding:1.5px 5px; border-radius:3px; color:var(--ink)}
strong{font-weight:700}
a{color:var(--dp)}
hr{display:none}

pre{
  background:var(--paper); border:1px solid var(--rule); border-radius:5px;
  padding:14px 16px; overflow-x:auto; font-family:var(--mono); font-size:12.5px;
  line-height:1.72; margin:18px 0; color:var(--ink);
}
pre code{background:none; padding:0; font-size:1em}

.tblbox{margin:20px 0 24px}
.tblbox .tblscroll{overflow-x:auto; -webkit-overflow-scrolling:touch}
.tbl{width:100%; border-collapse:collapse; margin:0; font-size:13.5px}
.tbl th,.tbl td{border-bottom:1px solid var(--rule-2); padding:9px 12px; text-align:left; vertical-align:top; line-height:1.6}
.tbl th{background:#F7F9FA; font-weight:700; font-size:12.5px; color:var(--ink-2); border-bottom:1.5px solid var(--rule)}
.tbl td:first-child{font-family:var(--mono); font-weight:650; white-space:nowrap}
.tbl tr:hover td{background:#F9FBFC}

.note{border-left:3px solid var(--ink-3); background:var(--paper); padding:14px 18px; margin:22px 0; border-radius:0 4px 4px 0}
.note p:last-child{margin-bottom:0}

nav.rail{
  position:fixed; left:max(12px, calc(50vw - 640px)); top:50%; transform:translateY(-50%);
  display:flex; flex-direction:column; gap:2px; z-index:20;
}
nav.rail a{
  font-family:var(--mono); font-size:10.5px; color:var(--ink-3); text-decoration:none;
  padding:5px 10px; border-left:2px solid var(--rule); transition:.15s; white-space:nowrap;
  cursor:pointer; display:block;
}
nav.rail a:hover{color:var(--ink); border-left-color:var(--ink); background:rgba(255,255,255,.75)}
nav.rail a:focus-visible{outline:2px solid var(--ink); outline-offset:1px}
nav.rail a.on{color:var(--ink); font-weight:700; border-left-color:var(--ink); background:rgba(255,255,255,.9)}
@media (max-width:1360px){ nav.rail{display:none} }

@media print{
  body{background:#fff}
  nav.rail,.pair{display:none}
  section{padding-top:28px; break-inside:avoid}
}
`;

const JS = `
(function(){
/* 左侧章节标尺：预览沙箱常拦 href="#..." 的原生锚点跳转，这里自己滚 */
function setupRail(){
  var links = [].slice.call(document.querySelectorAll('nav.rail a'));
  var items = [];
  links.forEach(function(a){
    var el = document.getElementById((a.getAttribute('href')||'').replace(/^#/,''));
    if(!el) return;
    items.push({a:a, el:el});
    a.addEventListener('click', function(ev){
      ev.preventDefault();
      try{ el.scrollIntoView({behavior:'smooth', block:'start'}); }
      catch(e){ window.scrollTo(0, el.offsetTop); }
    });
  });
  function spy(){
    var y = window.scrollY + 140, cur = items[0];
    items.forEach(function(it){ if(it.el.offsetTop <= y) cur = it; });
    items.forEach(function(it){ it.a.classList.toggle('on', it === cur); });
  }
  window.addEventListener('scroll', spy, {passive:true});
  window.addEventListener('resize', spy);
  spy();
}
if(document.readyState === 'loading'){
  document.addEventListener('DOMContentLoaded', setupRail);
}else{ setupRail(); }
})();
`;

const html = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<link rel="icon" href="../favicon.svg" type="image/svg+xml">
<title>${title}</title>
<style>${CSS}</style>
</head>
<body>

<nav class="rail" aria-label="章节">
${rail.map((r) => `  <a href="#${r.id}">${r.num} ${r.label}</a>`).join('\n')}
</nav>

<div class="wrap">

<header>
  <div class="eyebrow">Product Requirements</div>
  <h1>${inline(title.replace(/\s*PRD$/, ''))}</h1>
  ${sub ? `<p class="sub">${inline(sub)}</p>` : ''}
  <div class="meta">${meta.map((l) => `<span>${inline(l)}</span>`).join('')}</div>
  <div class="pair">配套阅读 · <a href="../parallel-reference/">分布式训练参照系 —— 五根轴 · 两个对象 · 三组坐标</a></div>
</header>

${sections.join('\n\n')}

</div>

<script>${JS}</script>
</body>
</html>
`;

writeFileSync(OUT, html);
console.log(`wrote ${OUT} — ${rail.length} sections, ${(html.length / 1024).toFixed(1)} KB`);
