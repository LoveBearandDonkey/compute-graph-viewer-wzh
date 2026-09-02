import http from 'node:http';
import { createHash } from 'node:crypto';
import { readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const publicDir = path.join(__dirname, 'public');
const defaultKnowledgePath = path.resolve(__dirname, '..', 'ParallelDemo', 'Transformer结构与并行策略知识库.md');
const knowledgePath = process.env.KNOWLEDGE_MD || defaultKnowledgePath;
const port = Number(process.env.PORT || 4173);

const mime = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml'
};

function stripMd(value) {
  return value
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/^>\s*/gm, '')
    .trim();
}

function blockLength(block) {
  return stripMd(block).replace(/[|#>*_`\-]/g, '').trim().length;
}

function splitLogicalBlocks(content) {
  const blocks = [];
  let current = [];
  let inFence = false;
  for (const line of content.split('\n')) {
    if (/^\s*(?:>\s*)?```/.test(line)) inFence = !inFence;
    if (!line.trim() && !inFence) {
      if (current.length) blocks.push(current.join('\n'));
      current = [];
    } else {
      current.push(line);
    }
  }
  if (current.length) blocks.push(current.join('\n'));
  return blocks.filter((block) => block.trim());
}

function cardTitle(base, body, part) {
  if (part === 1) return base;
  const lead = stripMd(body.split('\n').find((line) => line.trim()) || '');
  if (/⚠️|误区/.test(lead)) return `${base} · 常见误区`;
  if (/📖|对照案例/.test(lead)) return `${base} · 对照案例`;
  if (/💡|一句话|核心/.test(lead)) return `${base} · 核心结论`;
  if (/^\|/.test(body.trim()) || /速查表/.test(lead)) return `${base} · 速查表`;
  if (/^\s*(?:>\s*)?```/.test(body)) return `${base} · 结构示意`;
  const boldLead = body.trim().match(/^(?:>\s*)?\*\*([^*：:]{2,28})[：:]?\*\*/);
  if (boldLead) return `${base} · ${stripMd(boldLead[1])}`;
  return `${base} · ${part}`;
}

export function parseKnowledge(markdown, modifiedAt = '') {
  const lines = markdown.replace(/\r\n/g, '\n').split('\n');
  const cards = [];
  let chapter = '知识库';
  let section = '';
  let subsection = '';
  let buffer = [];
  let inFence = false;
  let part = 0;

  const flush = () => {
    const body = buffer.join('\n').replace(/^\s*---\s*$/gm, '').trim();
    buffer = [];
    if (!body || blockLength(body) < 12) return;
    const baseTitle = subsection || section || chapter;
    const title = cardTitle(baseTitle, body, ++part);
    const identity = `${chapter}\n${section}\n${title}\n${body}`;
    cards.push({
      id: createHash('sha1').update(identity).digest('hex').slice(0, 14),
      chapter,
      section: section || chapter,
      title,
      body
    });
  };

  const pushChunked = (content) => {
    const blocks = splitLogicalBlocks(content);
    let chunk = [];
    let size = 0;
    for (const block of blocks) {
      const len = blockLength(block);
      const isBoundary = /^(?:>\s*)?(?:⚠️|💡|📖|\*\*关键|\*\*核心|\*\*一句话)/.test(block.trim());
      if (chunk.length && (size + len > 680 || (isBoundary && size > 220))) {
        buffer = chunk;
        flush();
        chunk = [];
        size = 0;
      }
      chunk.push(block);
      size += len;
    }
    if (chunk.length) {
      buffer = chunk;
      flush();
    }
  };

  let pending = [];
  const commitPending = () => {
    if (!pending.length) return;
    pushChunked(pending.join('\n'));
    pending = [];
  };

  for (const line of lines) {
    if (line.trimStart().startsWith('```')) inFence = !inFence;
    const heading = inFence ? null : line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      commitPending();
      const level = heading[1].length;
      const text = stripMd(heading[2]);
      if (level === 1) {
        chapter = text;
        section = '';
        subsection = '';
      } else if (level === 2) {
        section = text;
        subsection = '';
      } else {
        subsection = text;
      }
      part = 0;
      continue;
    }
    pending.push(line);
  }
  commitPending();

  return {
    title: 'Transformer 结构与并行策略',
    source: knowledgePath,
    modifiedAt,
    cards
  };
}

async function serveApi(res) {
  try {
    const [markdown, info] = await Promise.all([
      readFile(knowledgePath, 'utf8'),
      stat(knowledgePath)
    ]);
    const payload = parseKnowledge(markdown, info.mtime.toISOString());
    res.writeHead(200, { 'Content-Type': mime['.json'], 'Cache-Control': 'no-store' });
    res.end(JSON.stringify(payload));
  } catch (error) {
    res.writeHead(500, { 'Content-Type': mime['.json'] });
    res.end(JSON.stringify({ error: `无法读取知识库：${error.message}`, source: knowledgePath }));
  }
}

async function serveStatic(urlPath, res) {
  const requestPath = urlPath === '/' ? '/index.html' : urlPath;
  const resolved = path.resolve(publicDir, `.${requestPath}`);
  if (!resolved.startsWith(publicDir)) {
    res.writeHead(403).end('Forbidden');
    return;
  }
  try {
    const data = await readFile(resolved);
    res.writeHead(200, { 'Content-Type': mime[path.extname(resolved)] || 'application/octet-stream' });
    res.end(data);
  } catch {
    res.writeHead(404).end('Not found');
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  http.createServer(async (req, res) => {
    const url = new URL(req.url, `http://${req.headers.host}`);
    if (url.pathname === '/api/cards') await serveApi(res);
    else await serveStatic(decodeURIComponent(url.pathname), res);
  }).listen(port, '0.0.0.0', () => {
    console.log(`Transformer Pocket Cards: http://localhost:${port}`);
    console.log(`Knowledge source: ${knowledgePath}`);
  });
}
