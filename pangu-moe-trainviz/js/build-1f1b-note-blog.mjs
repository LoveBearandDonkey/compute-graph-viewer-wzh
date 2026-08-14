import { readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const moduleRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const sourcePath = join(moduleRoot, 'NOTE-1F1B-MB-lifecycle.md');
const outputPath = join(moduleRoot, 'NOTE-1F1B-MB-lifecycle.html');

const markdown = await readFile(sourcePath, 'utf8');

function escapeHtml(value) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function renderInline(value) {
  const tokens = [];
  const stash = (html) => {
    const token = `@@HTML_TOKEN_${tokens.length}@@`;
    tokens.push(html);
    return token;
  };

  let rendered = value
    .replace(/`([^`]+)`/g, (_, code) => stash(`<code>${escapeHtml(code)}</code>`))
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, href) => {
      const external = /^https?:\/\//.test(href);
      const attributes = external ? ' target="_blank" rel="noreferrer"' : '';
      return stash(`<a href="${escapeHtml(href)}"${attributes}>${escapeHtml(label)}</a>`);
    });

  rendered = escapeHtml(rendered)
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/~~([^~]+)~~/g, '<del>$1</del>');

  for (let index = tokens.length - 1; index >= 0; index -= 1) {
    rendered = rendered.replaceAll(`@@HTML_TOKEN_${index}@@`, tokens[index]);
  }

  return rendered;
}

function headingId(text, index) {
  const numbered = text.match(/^(\d+(?:\.\d+)*)/);
  if (numbered) return `section-${numbered[1].replaceAll('.', '-')}`;
  return `section-${index + 1}`;
}

function splitTableRow(line) {
  return line
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map((cell) => cell.trim());
}

const pipelineFigure = `
  <figure class="article-figure" aria-labelledby="pipeline-caption">
    <figcaption id="pipeline-caption">
      <span class="figure-kicker">Forward / Backward</span>
      一个 microbatch 在四个 PP stage 间的往返路径
    </figcaption>
    <div class="pipeline-flow" role="img" aria-label="Forward 从 Input 经 PP0、PP1、PP2、PP3 到 Loss，Backward 反向返回">
      <span class="flow-end">Input</span>
      <span class="flow-arrow flow-arrow--forward">→</span>
      <span class="flow-stage">PP0<small>L0-L11</small></span>
      <span class="flow-arrow flow-arrow--forward">→</span>
      <span class="flow-stage">PP1<small>L12-L22</small></span>
      <span class="flow-arrow flow-arrow--forward">→</span>
      <span class="flow-stage">PP2<small>L23-L34</small></span>
      <span class="flow-arrow flow-arrow--forward">→</span>
      <span class="flow-stage">PP3<small>L35-L45</small></span>
      <span class="flow-arrow flow-arrow--forward">→</span>
      <span class="flow-end">Loss</span>
    </div>
    <div class="direction-key" aria-label="方向图例">
      <span><i class="key-line key-line--forward"></i>Forward：activation 向后传</span>
      <span><i class="key-line key-line--backward"></i>Backward：activation gradient 向前传</span>
    </div>
  </figure>`;

const scheduleFigure = `
  <figure class="article-figure article-figure--wide" aria-labelledby="schedule-caption">
    <figcaption id="schedule-caption">
      <span class="figure-kicker">Non-interleaved 1F1B</span>
      每个 stage 的本地执行模式
    </figcaption>
    <p class="figure-note">下表表达各 stage 的本地顺序，不把四行误读成严格对齐的全局时间轴。</p>
    <div class="schedule-grid" role="table" aria-label="各 pipeline stage 的 warmup、steady 和 cooldown 模式">
      <div class="schedule-cell schedule-cell--head" role="columnheader">Stage</div>
      <div class="schedule-cell schedule-cell--head" role="columnheader">Warmup</div>
      <div class="schedule-cell schedule-cell--head" role="columnheader">Steady</div>
      <div class="schedule-cell schedule-cell--head" role="columnheader">Cooldown</div>
      <div class="schedule-cell schedule-cell--stage" role="rowheader">PP0</div>
      <div class="schedule-cell"><span class="task task--forward">F</span><span class="task task--forward">F</span><span class="task task--forward">F</span></div>
      <div class="schedule-cell"><span class="task-pair"><b>F</b><b>B</b></span><span class="task-pair"><b>F</b><b>B</b></span><span class="task-pair"><b>…</b></span></div>
      <div class="schedule-cell"><span class="task task--backward">B</span><span class="task task--backward">B</span><span class="task task--backward">…</span></div>
      <div class="schedule-cell schedule-cell--stage" role="rowheader">PP1</div>
      <div class="schedule-cell"><span class="task task--forward">F</span><span class="task task--forward">F</span></div>
      <div class="schedule-cell"><span class="task-pair"><b>F</b><b>B</b></span><span class="task-pair"><b>F</b><b>B</b></span><span class="task-pair"><b>…</b></span></div>
      <div class="schedule-cell"><span class="task task--backward">B</span><span class="task task--backward">B</span><span class="task task--backward">…</span></div>
      <div class="schedule-cell schedule-cell--stage" role="rowheader">PP2</div>
      <div class="schedule-cell"><span class="task task--forward">F</span></div>
      <div class="schedule-cell"><span class="task-pair"><b>F</b><b>B</b></span><span class="task-pair"><b>F</b><b>B</b></span><span class="task-pair"><b>…</b></span></div>
      <div class="schedule-cell"><span class="task task--backward">B</span><span class="task task--backward">…</span></div>
      <div class="schedule-cell schedule-cell--stage" role="rowheader">PP3</div>
      <div class="schedule-cell"><span class="task task--idle">0</span></div>
      <div class="schedule-cell"><span class="task-pair"><b>F</b><b>B</b></span><span class="task-pair"><b>F</b><b>B</b></span><span class="task-pair"><b>…</b></span></div>
      <div class="schedule-cell"><span class="task task--backward">B</span><span class="task task--backward">…</span></div>
    </div>
  </figure>`;

const lifecycleFigure = `
  <figure class="article-figure article-figure--wide" aria-labelledby="lifecycle-caption">
    <figcaption id="lifecycle-caption">
      <span class="figure-kicker">MB-centric lifecycle</span>
      固定一个 MB 后，追踪它在一个 stage 上的完整状态
    </figcaption>
    <div class="lifecycle-flow" role="img" aria-label="WAIT ACT、Forward Compute、Send ACT、Hold Activation、Wait Grad、Backward Compute、Send Grad、Done">
      <span class="life-step life-step--wait">Wait act<small>等待 activation</small></span>
      <span class="life-link">→</span>
      <span class="life-step life-step--compute">Forward<small>stage compute</small></span>
      <span class="life-link">→</span>
      <span class="life-step life-step--comm">Send act<small>PP P2P</small></span>
      <span class="life-link">→</span>
      <span class="life-step life-step--hold">Hold<small>保留 activation</small></span>
      <span class="life-link">→</span>
      <span class="life-step life-step--wait">Wait grad<small>等待 gradient</small></span>
      <span class="life-link">→</span>
      <span class="life-step life-step--compute">Backward<small>stage compute</small></span>
      <span class="life-link">→</span>
      <span class="life-step life-step--comm">Send grad<small>PP P2P</small></span>
      <span class="life-link">→</span>
      <span class="life-step life-step--done">Done<small>完成当前 stage</small></span>
    </div>
  </figure>`;

function figureForHeading(text) {
  if (text.startsWith('1.1 ')) return pipelineFigure;
  if (text.startsWith('4. 标准')) return scheduleFigure;
  if (text.startsWith('6. 以一个 MB')) return lifecycleFigure;
  return '';
}

function renderMarkdown(source) {
  const lines = source.split(/\r?\n/);
  const start = lines.findIndex((line) => /^##\s/.test(line));
  const content = start >= 0 ? lines.slice(start) : lines;
  const headings = [];
  const output = [];
  let i = 0;

  const isSpecial = (line, next = '') => (
    line.trim() === ''
    || /^#{2,4}\s/.test(line)
    || /^```/.test(line)
    || /^>\s?/.test(line)
    || /^[-*]\s+/.test(line)
    || /^\d+\.\s+/.test(line)
    || (line.includes('|') && /^\s*\|?\s*:?-{3,}/.test(next))
  );

  while (i < content.length) {
    const line = content[i];
    const next = content[i + 1] || '';

    if (line.trim() === '') {
      i += 1;
      continue;
    }

    const heading = line.match(/^(#{2,4})\s+(.+)$/);
    if (heading) {
      const level = heading[1].length;
      const text = heading[2].trim();
      const id = headingId(text, headings.length);
      headings.push({ level, text, id });
      output.push(`<h${level} id="${id}">${renderInline(text)}</h${level}>`);
      output.push(figureForHeading(text));
      i += 1;
      continue;
    }

    const fence = line.match(/^```\s*([^\s]*)/);
    if (fence) {
      const language = fence[1] || 'text';
      const code = [];
      i += 1;
      while (i < content.length && !/^```/.test(content[i])) {
        code.push(content[i]);
        i += 1;
      }
      i += 1;
      output.push(`<pre data-language="${escapeHtml(language)}"><code>${escapeHtml(code.join('\n'))}</code></pre>`);
      continue;
    }

    if (/^>\s?/.test(line)) {
      const quote = [];
      while (i < content.length && /^>\s?/.test(content[i])) {
        quote.push(content[i].replace(/^>\s?/, ''));
        i += 1;
      }
      output.push(`<aside class="article-note">${quote.map((item) => `<p>${renderInline(item)}</p>`).join('')}</aside>`);
      continue;
    }

    if (line.includes('|') && /^\s*\|?\s*:?-{3,}/.test(next)) {
      const header = splitTableRow(line);
      i += 2;
      const rows = [];
      while (i < content.length && content[i].includes('|') && content[i].trim() !== '') {
        rows.push(splitTableRow(content[i]));
        i += 1;
      }
      output.push(`
        <div class="table-scroll" tabindex="0" role="region" aria-label="可横向滚动的数据表">
          <table>
            <thead><tr>${header.map((cell) => `<th scope="col">${renderInline(cell)}</th>`).join('')}</tr></thead>
            <tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${renderInline(cell)}</td>`).join('')}</tr>`).join('')}</tbody>
          </table>
        </div>`);
      continue;
    }

    if (/^[-*]\s+/.test(line)) {
      const items = [];
      while (i < content.length && /^[-*]\s+/.test(content[i])) {
        items.push(content[i].replace(/^[-*]\s+/, ''));
        i += 1;
      }
      output.push(`<ul>${items.map((item) => `<li>${renderInline(item)}</li>`).join('')}</ul>`);
      continue;
    }

    if (/^\d+\.\s+/.test(line)) {
      const items = [];
      while (i < content.length && /^\d+\.\s+/.test(content[i])) {
        items.push(content[i].replace(/^\d+\.\s+/, ''));
        i += 1;
      }
      output.push(`<ol>${items.map((item) => `<li>${renderInline(item)}</li>`).join('')}</ol>`);
      continue;
    }

    const paragraph = [line.trim()];
    i += 1;
    while (i < content.length && !isSpecial(content[i], content[i + 1] || '')) {
      paragraph.push(content[i].trim());
      i += 1;
    }
    const text = paragraph.join(' ');
    const className = /^(白话解释|注意)/.test(text) ? ' class="plain-language"' : '';
    output.push(`<p${className}>${renderInline(text)}</p>`);
  }

  return { html: output.filter(Boolean).join('\n'), headings };
}

const article = renderMarkdown(markdown);
const toc = article.headings
  .filter(({ level }) => level <= 3)
  .map(({ level, text, id }) => `<a class="toc-link toc-link--h${level}" href="#${id}">${renderInline(text)}</a>`)
  .join('\n');

const document = `<!doctype html>
<html lang="zh-CN" data-theme="light">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="基于 NVIDIA Megatron Core 0.16.0 核对的 non-interleaved 1F1B 时序、通信顺序与 microbatch lifecycle 笔记。">
  <title>1F1B 时序与 Microbatch Lifecycle | TrainScope Notes</title>
  <link rel="stylesheet" href="../vendor/pto-design-system/css/style.css">
  <style>
    :root {
      --article-measure: 72ch;
      --article-rail: 224px;
      --article-page-bg: #F8F8F8;
      --background: var(--article-page-bg);
      --app-background: var(--article-page-bg);
      --canvas-bg: var(--article-page-bg);
    }

    html {
      scroll-behavior: smooth;
      background: var(--article-page-bg);
    }

    body {
      display: block;
      width: 100%;
      min-height: 100vh;
      height: auto;
      overflow: visible;
      color: var(--foreground);
      background: var(--article-page-bg);
      font-family: var(--font-sans);
      font-size: 17px;
      line-height: 1.78;
      letter-spacing: 0;
    }

    a {
      color: var(--primary);
      text-underline-offset: 3px;
      text-decoration-thickness: 1px;
    }

    a:hover { color: var(--primary-hover); }

    .skip-link {
      position: fixed;
      top: var(--space-3);
      left: var(--space-3);
      z-index: var(--z-modal);
      transform: translateY(-160%);
    }

    .skip-link:focus { transform: translateY(0); }

    .reading-progress {
      position: fixed;
      inset: 0 auto auto 0;
      z-index: calc(var(--z-overlay) + 1);
      width: 0;
      height: 3px;
      background: var(--primary);
    }

    .article-header {
      width: min(var(--article-measure), calc(100% - 48px));
      margin: 0 auto;
      padding: 72px 0 56px;
      border-bottom: 1px solid var(--border-subtle);
    }

    .header-copy { max-width: 860px; }

    .eyebrow,
    .figure-kicker {
      display: block;
      margin-bottom: var(--space-3);
      color: var(--primary);
      font-family: var(--font-mono);
      font-size: var(--font-size-label-xs);
      font-weight: var(--font-weight-bold);
      line-height: 1.2;
      letter-spacing: var(--letter-spacing-label);
      text-transform: uppercase;
    }

    .article-header h1 {
      max-width: 820px;
      margin: 0;
      color: var(--foreground);
      font-size: 42px;
      font-weight: var(--font-weight-bold);
      line-height: 1.18;
      letter-spacing: 0;
    }

    .lede {
      max-width: 760px;
      margin: var(--space-5) 0 0;
      color: var(--foreground-secondary);
      font-size: 19px;
      line-height: 1.72;
    }

    .article-meta {
      display: flex;
      flex-wrap: wrap;
      gap: var(--space-2);
      margin-top: var(--space-6);
    }

    .article-layout {
      display: block;
      width: min(var(--article-measure), calc(100% - 48px));
      margin: 0 auto;
      padding: 56px 0 96px;
    }

    .article-toc {
      position: fixed;
      top: 24px;
      right: max(16px, calc((100vw - 1520px) / 2));
      z-index: var(--z-raised);
      width: var(--article-rail);
      max-height: calc(100vh - 48px);
      overflow: auto;
      padding: var(--space-2) 0;
    }

    .toc-title {
      margin-bottom: var(--space-3);
      color: var(--foreground);
      font-size: var(--font-size-body-sm);
      font-weight: var(--font-weight-semibold);
    }

    .toc-links {
      display: grid;
      gap: 2px;
      border-left: 1px solid var(--border-subtle);
    }

    .toc-link {
      display: block;
      padding: 6px 0 6px var(--space-3);
      border-left: 2px solid transparent;
      color: var(--foreground-muted);
      font-size: 12px;
      line-height: 1.35;
      text-decoration: none;
    }

    .toc-link--h3 { padding-left: var(--space-6); }

    .toc-link:hover,
    .toc-link.is-active {
      border-left-color: var(--primary);
      color: var(--foreground);
    }

    .article-body {
      min-width: 0;
      max-width: var(--article-measure);
    }

    .article-body h2,
    .article-body h3,
    .article-body h4 {
      color: var(--foreground);
      letter-spacing: 0;
      scroll-margin-top: 88px;
    }

    .article-body h2 {
      margin: 76px 0 var(--space-5);
      padding-top: var(--space-3);
      border-top: 1px solid var(--border-subtle);
      font-size: 29px;
      font-weight: var(--font-weight-bold);
      line-height: 1.28;
    }

    .article-body > h2:first-child { margin-top: 0; }

    .article-body h3 {
      margin: 48px 0 var(--space-4);
      font-size: 22px;
      font-weight: var(--font-weight-semibold);
      line-height: 1.35;
    }

    .article-body h4 {
      margin: 36px 0 var(--space-3);
      font-size: 18px;
      font-weight: var(--font-weight-semibold);
      line-height: 1.4;
    }

    .article-body p,
    .article-body li {
      color: color-mix(in srgb, var(--foreground) 92%, var(--background));
      font-size: 17px;
      line-height: 1.82;
    }

    .article-body p { margin: 0 0 1.2em; }

    .article-body strong {
      color: var(--foreground);
      font-weight: var(--font-weight-semibold);
    }

    .article-body ul,
    .article-body ol {
      display: grid;
      gap: var(--space-2);
      margin: 0 0 var(--space-6);
      padding-left: 1.35em;
    }

    .article-body li { padding-left: var(--space-1); }

    .article-body :not(pre) > code {
      padding: 2px 6px;
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      background: var(--surface-2);
      color: color-mix(in srgb, var(--foreground) 92%, var(--primary));
      font-family: var(--font-mono);
      font-size: 0.86em;
      word-break: break-word;
    }

    pre {
      position: relative;
      margin: var(--space-5) 0 var(--space-6);
      padding: var(--space-6);
      overflow: auto;
      border: 1px solid var(--border-default);
      border-radius: var(--radius-md);
      background: var(--surface-2);
      color: var(--foreground);
      font-family: var(--font-mono);
      font-size: 14px;
      line-height: 1.68;
      tab-size: 2;
    }

    pre::before {
      position: absolute;
      top: var(--space-2);
      right: var(--space-3);
      color: var(--foreground-muted);
      content: attr(data-language);
      font-size: 10px;
      text-transform: uppercase;
    }

    pre code { font: inherit; }

    .article-note,
    .plain-language {
      margin: var(--space-5) 0 var(--space-6);
      padding: var(--space-4) var(--space-5);
      border-left: 3px solid var(--primary);
      background: color-mix(in srgb, var(--primary) 7%, transparent);
    }

    .article-note p,
    .plain-language { color: var(--foreground-secondary); }
    .article-note p:last-child { margin-bottom: 0; }

    .plain-language::before {
      display: block;
      margin-bottom: var(--space-1);
      color: var(--primary);
      content: '直观解释';
      font-family: var(--font-mono);
      font-size: var(--font-size-label-xs);
      font-weight: var(--font-weight-bold);
    }

    .table-scroll {
      width: 100%;
      margin: var(--space-5) 0 var(--space-6);
      overflow-x: auto;
      border: 1px solid var(--border-default);
      border-radius: var(--radius-md);
    }

    table {
      width: 100%;
      min-width: 620px;
      border-collapse: collapse;
      background: var(--surface-1);
    }

    th,
    td {
      padding: 13px var(--space-4);
      border-bottom: 1px solid var(--border-subtle);
      color: var(--foreground-secondary);
      font-size: 14px;
      line-height: 1.55;
      text-align: left;
      vertical-align: top;
    }

    th {
      background: var(--surface-2);
      color: var(--foreground);
      font-weight: var(--font-weight-semibold);
    }

    tbody tr:last-child td { border-bottom: 0; }

    .article-figure {
      margin: var(--space-6) 0 32px;
      padding: var(--space-6);
      border: 1px solid var(--border-default);
      border-radius: var(--radius-lg);
      background: var(--surface-1);
    }

    .article-figure figcaption {
      margin-bottom: var(--space-5);
      color: var(--foreground);
      font-size: 16px;
      font-weight: var(--font-weight-semibold);
      line-height: 1.45;
    }

    .figure-kicker { margin-bottom: var(--space-1); }

    .figure-note {
      margin: calc(-1 * var(--space-3)) 0 var(--space-5) !important;
      color: var(--foreground-muted) !important;
      font-size: 13px !important;
      line-height: 1.55 !important;
    }

    .pipeline-flow {
      display: grid;
      grid-template-columns: auto 20px repeat(4, minmax(92px, 1fr) 20px) auto;
      align-items: center;
      gap: var(--space-1);
    }

    .flow-stage,
    .flow-end {
      display: grid;
      place-items: center;
      min-height: 58px;
      padding: var(--space-2) var(--space-3);
      border: 1px solid var(--border-default);
      border-radius: var(--radius-md);
      background: var(--surface-2);
      color: var(--foreground);
      font-family: var(--font-mono);
      font-size: 13px;
      font-weight: var(--font-weight-semibold);
      text-align: center;
    }

    .flow-end {
      min-height: 40px;
      background: color-mix(in srgb, var(--primary) 10%, var(--surface-1));
    }

    .flow-stage small,
    .life-step small {
      display: block;
      margin-top: 2px;
      color: var(--foreground-muted);
      font-size: 10px;
      font-weight: var(--font-weight-medium);
    }

    .flow-arrow {
      color: var(--primary);
      font-family: var(--font-mono);
      font-size: 20px;
      text-align: center;
    }

    .direction-key {
      display: flex;
      flex-wrap: wrap;
      gap: var(--space-3) var(--space-5);
      margin-top: var(--space-5);
      color: var(--foreground-secondary);
      font-size: 12px;
    }

    .direction-key span { display: inline-flex; align-items: center; gap: var(--space-2); }

    .key-line {
      width: 24px;
      height: 3px;
      background: var(--primary);
    }

    .key-line--backward { background: var(--warning); }

    .schedule-grid {
      display: grid;
      grid-template-columns: 64px minmax(116px, 0.8fr) minmax(210px, 1.5fr) minmax(116px, 0.8fr);
      overflow: hidden;
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
    }

    .schedule-cell {
      display: flex;
      min-height: 54px;
      align-items: center;
      gap: 5px;
      padding: var(--space-2) var(--space-3);
      border-right: 1px solid var(--border-subtle);
      border-bottom: 1px solid var(--border-subtle);
      color: var(--foreground-secondary);
      font-family: var(--font-mono);
      font-size: 12px;
    }

    .schedule-cell:nth-child(4n) { border-right: 0; }
    .schedule-cell:nth-last-child(-n + 4) { border-bottom: 0; }

    .schedule-cell--head {
      min-height: 36px;
      background: var(--surface-2);
      color: var(--foreground-muted);
      font-size: 10px;
      font-weight: var(--font-weight-bold);
      text-transform: uppercase;
    }

    .schedule-cell--stage {
      color: var(--foreground);
      font-weight: var(--font-weight-bold);
    }

    .task,
    .task-pair {
      display: inline-flex;
      min-width: 24px;
      min-height: 24px;
      align-items: center;
      justify-content: center;
      border-radius: var(--radius-sm);
      font-size: 10px;
    }

    .task--forward { background: color-mix(in srgb, var(--primary) 16%, transparent); color: var(--primary); }
    .task--backward { background: color-mix(in srgb, var(--warning) 18%, transparent); color: color-mix(in srgb, var(--warning) 78%, var(--foreground)); }
    .task--idle { background: var(--surface-2); color: var(--foreground-muted); }

    .task-pair { overflow: hidden; border: 1px solid var(--border-subtle); }
    .task-pair b { display: grid; width: 22px; min-height: 24px; place-items: center; font-weight: var(--font-weight-semibold); }
    .task-pair b:first-child { background: color-mix(in srgb, var(--primary) 16%, transparent); color: var(--primary); }
    .task-pair b:last-child { background: color-mix(in srgb, var(--warning) 18%, transparent); color: color-mix(in srgb, var(--warning) 78%, var(--foreground)); }
    .task-pair b:only-child { background: var(--surface-2); color: var(--foreground-muted); }

    .lifecycle-flow {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: var(--space-2);
    }

    .life-step {
      display: grid;
      min-height: 54px;
      place-items: center;
      padding: var(--space-2) var(--space-3);
      border: 1px solid var(--border-default);
      border-radius: var(--radius-md);
      background: var(--surface-2);
      color: var(--foreground);
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: var(--font-weight-semibold);
      text-align: center;
    }

    .life-step--compute { border-color: color-mix(in srgb, var(--primary) 38%, var(--border-default)); background: color-mix(in srgb, var(--primary) 10%, var(--surface-2)); }
    .life-step--comm { border-color: color-mix(in srgb, var(--success) 38%, var(--border-default)); background: color-mix(in srgb, var(--success) 8%, var(--surface-2)); }
    .life-step--hold { border-color: color-mix(in srgb, var(--warning) 42%, var(--border-default)); background: color-mix(in srgb, var(--warning) 9%, var(--surface-2)); }
    .life-step--done { color: var(--success); }
    .life-link { color: var(--foreground-muted); font-family: var(--font-mono); }

    .article-footer {
      margin-top: 80px;
      padding-top: var(--space-6);
      border-top: 1px solid var(--border-subtle);
      color: var(--foreground-muted);
      font-size: 13px;
      line-height: 1.65;
    }

    @media (max-width: 1240px) {
      .article-toc {
        display: none;
      }
    }

    @media (max-width: 720px) {
      .article-header,
      .article-layout {
        width: min(100% - 32px, var(--article-measure));
      }

      .article-header { padding: 48px 0 40px; }

      .article-header h1 { font-size: 32px; line-height: 1.22; }
      .lede { font-size: 17px; }

      .article-layout { padding-top: 32px; }

      .article-body h2 { margin-top: 60px; font-size: 26px; }
      .article-body h3 { margin-top: 40px; font-size: 21px; }
      .article-body p,
      .article-body li { font-size: 16.5px; line-height: 1.78; }

      pre { margin-inline: -8px; padding: var(--space-5); font-size: 13px; }

      .article-figure { margin-inline: -8px; padding: var(--space-5); }

      .pipeline-flow {
        grid-template-columns: 1fr;
      }

      .flow-arrow { transform: rotate(90deg); }

      .schedule-grid {
        grid-template-columns: 52px minmax(108px, 0.8fr) minmax(190px, 1.5fr) minmax(108px, 0.8fr);
        overflow-x: auto;
      }

      .lifecycle-flow { align-items: stretch; }
      .life-step { flex: 1 1 132px; }
      .life-link { display: none; }
    }

    @media (prefers-reduced-motion: reduce) {
      html { scroll-behavior: auto; }
    }

    @media print {
      .reading-progress,
      .article-toc { display: none; }

      body { color: #111; background: #fff; }
      .article-header,
      .article-layout { width: 100%; }
      .article-layout { display: block; padding-top: 32px; }
      .article-body { max-width: none; }
      a { color: #111; text-decoration: underline; }
      pre, .article-figure, .table-scroll { break-inside: avoid; }
    }
  </style>
</head>
<body>
  <a class="skip-link btn btn-solid" href="#article-content">跳到正文</a>
  <div class="reading-progress" aria-hidden="true"></div>

  <header class="article-header">
    <div class="header-copy">
      <span class="eyebrow">Architecture Note / Verified Schedule</span>
      <h1>1F1B 时序与 Microbatch Lifecycle</h1>
      <p class="lede">从单个 pipeline stage 的 F/B 排队，到一个 microbatch 穿过四个 stage 的完整往返路径：这份笔记用 Megatron Core 0.16.0 的指南、API 与源码核对计算依赖、通信方向和当前可视化的可信边界。</p>
      <div class="article-meta" aria-label="文章元数据">
        <span class="stat-chip">Megatron Core 0.16.0</span>
        <span class="stat-chip">non-interleaved 1F1B</span>
        <span class="stat-chip">DP2 / PP4 / TP2 / EP2</span>
        <span class="stat-chip">最后核对 2026-07-14</span>
      </div>
    </div>
  </header>

  <div class="article-layout">
    <aside class="article-toc" aria-label="文章目录">
      <div class="toc-title">目录</div>
      <nav class="toc-links">${toc}</nav>
    </aside>

    <main class="article-body" id="article-content">
      ${article.html}
      <footer class="article-footer">
        本页由 <a href="./NOTE-1F1B-MB-lifecycle.md">原始核对笔记</a>生成。概念与宏观顺序以 NVIDIA Megatron Core 0.16.0 为依据；页面中的 task 时长、算子位置和通信重叠窗口仍属于模拟数据。
      </footer>
    </main>
  </div>

  <script>
    const progress = document.querySelector('.reading-progress');
    const tocLinks = [...document.querySelectorAll('.toc-link')];
    const headings = tocLinks
      .map((link) => document.querySelector(link.getAttribute('href')))
      .filter(Boolean);

    const updateProgress = () => {
      const scrollable = document.documentElement.scrollHeight - window.innerHeight;
      const ratio = scrollable > 0 ? window.scrollY / scrollable : 0;
      progress.style.width = Math.min(100, Math.max(0, ratio * 100)) + '%';
    };

    const observer = new IntersectionObserver((entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
      if (!visible) return;
      tocLinks.forEach((link) => {
        link.classList.toggle('is-active', link.getAttribute('href') === '#' + visible.target.id);
      });
    }, { rootMargin: '-72px 0px -72% 0px', threshold: 0 });

    headings.forEach((heading) => observer.observe(heading));
    window.addEventListener('scroll', updateProgress, { passive: true });
    updateProgress();
  </script>
</body>
</html>`;

await writeFile(outputPath, document, 'utf8');
console.log(`Wrote ${outputPath}`);
