import { readFile, writeFile, stat } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { parseKnowledge } from './server.mjs';

const source = process.env.KNOWLEDGE_MD || fileURLToPath(new URL('../ParallelDemo/Transformer结构与并行策略知识库.md', import.meta.url));
const [markdown, info] = await Promise.all([readFile(source, 'utf8'), stat(source)]);
const data = parseKnowledge(markdown, info.mtime.toISOString());
const json = JSON.stringify(data);
await Promise.all([
  writeFile(new URL('./public/cards.json', import.meta.url), json, 'utf8'),
  writeFile(new URL('./public/cards-data.js', import.meta.url), `window.__CARDS__=${json};`, 'utf8')
]);
console.log(`Built ${data.cards.length} cards → public/cards.json + cards-data.js`);
