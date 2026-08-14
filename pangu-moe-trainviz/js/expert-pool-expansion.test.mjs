import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const page = readFileSync(
  new URL('../op-rank-time-openpangu-flash-events.html', import.meta.url),
  'utf8'
);

const generator = page.match(
  /const CSS_DECK_EXPERT_COUNT=72;[\s\S]*?\n}\nfunction cssGraphCenter/
)?.[0];
const selectionGuard = page.match(
  /function cssDeckExpertExpansionSelected\(\)[\s\S]*?\n}\nfunction cssDeckElementBox/
)?.[0];
const expansionSync = page.match(
  /function syncCssDeckExpertExpansion\(\)[\s\S]*?\n}\nfunction cssDeckPpLayerIsSample/
)?.[0];

assert.ok(generator, 'Expert Pool should expose a dedicated collapsed generator');
assert.match(generator, /CSS_DECK_EXPERT_COLLAPSED_BOX=Object\.freeze\(\{x:300,y:700,w:140,h:52\}\)/);
assert.match(generator, /opv-cssdeck-experts is-collapsed/);
assert.match(generator, /aria-expanded="false"/);
assert.match(generator, /data-collapsed-[xywh]=/);
assert.match(generator, /data-expanded-[xywh]=/);
assert.doesNotMatch(generator, /Array\.from\(\{length:72\}/, 'the default model must not eagerly create expert cells');

const generatorSource = generator.slice(0, generator.lastIndexOf('\nfunction cssGraphCenter'));
const buildExpertPool = new Function(
  `${generatorSource}; const nodes={}; const html=cssGraphExperts(nodes,278,674,210,142); return {nodes,html};`
);
const compactPool = buildExpertPool();
assert.deepEqual(compactPool.nodes.expert_pool, { x: 300, y: 700, w: 140, h: 52 });
assert.match(compactPool.html, /top:700px;width:140px;height:52px/);
assert.doesNotMatch(compactPool.html, /<span/, 'the initial compact node should contain no expert-cell DOM');

assert.ok(selectionGuard, 'Expert Pool expansion should have an explicit view guard');
assert.match(selectionGuard, /activeLens==='communication'&&cssDeckView==='front'/);
assert.match(selectionGuard, /parallelEventFilter==='EP'/);
assert.match(selectionGuard, /selected\?\.parallelTags\?\.includes\('EP'\)/);

assert.ok(expansionSync, 'Expert Pool expansion should be synchronized with view state');
assert.match(expansionSync, /cssDeck\.dataset\.epExperts=expansionSelected\?'expanded':'collapsed'/);
assert.match(expansionSync, /expansionSelected&&card\.classList\.contains\('is-front-layer'\)/, 'only the visible front layer may allocate expert cells');
assert.match(expansionSync, /length:CSS_DECK_EXPERT_COUNT/, 'EP view should lazily create all 72 expert cells');
assert.match(expansionSync, /expertPool\.replaceChildren\(\)/, 'leaving EP should release the expanded cells');
assert.match(expansionSync, /syncCssDeckExpertEdges\(card\)/, 'dispatch/combine paths should follow the compact and expanded geometry');
assert.match(expansionSync, /expertPool\.classList\.toggle\('is-expanded',expanded\)/);
assert.match(expansionSync, /expertPool\.classList\.toggle\('is-collapsed',!expanded\)/);

assert.match(page, /parallelEventFilter=tag;\s*clearParallelEventSelection\(\);\s*if\(tag!=='PP'\)setCssGraphView\('front'\);/, 'EP filter should enter the front projection before expansion');
assert.match(page, /function syncParallelEventOverlay\(\)[\s\S]*?ensureParallelEventUi\(\);\s*syncCssDeckExpertExpansion\(\);/, 'lens and event updates should resynchronize Expert Pool state');
assert.match(page, /syncCssDeckViewClasses\(\)[\s\S]*?syncCssDeckExpertExpansion\?\.\(\);/, 'view changes should collapse or expand Expert Pool');
const collapsedStyle = page.match(/\.opv-cssdeck-experts\.is-collapsed\{[^}]+\}/)?.[0];
assert.ok(collapsedStyle, 'collapsed Expert Pool should have a standard op-node style');
assert.match(collapsedStyle, /min-height:32px;display:flex/);
assert.match(collapsedStyle, /padding:6px 8px/);
assert.match(collapsedStyle, /border:1px solid color-mix\(in srgb,var\(--deck-moe\) 28%,transparent\)/);
assert.match(collapsedStyle, /border-radius:8px/);
assert.match(collapsedStyle, /background:color-mix\(in srgb,var\(--deck-moe\) 24%,var\(--surface-1\)\)/);
assert.match(collapsedStyle, /color:var\(--foreground\);font:800 11px\/1\.18 var\(--font-sans\)/);
assert.match(page, /cssGraphNode\(nodes,'shared_expert','Shared Expert','moe',508,700,140,52\)/, 'Expert Pool and Shared Expert should share the same row and 140×52 op-node size');
assert.match(page, /\.opv-cssdeck-experts\.is-collapsed::before\{content:"Expert Pool";position:static/, 'the collapsed node label should live inside the op node');

console.log('Expert Pool expansion tests passed');
