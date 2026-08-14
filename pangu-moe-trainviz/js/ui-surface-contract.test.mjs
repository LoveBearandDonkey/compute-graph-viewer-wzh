import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const page = readFileSync(
  new URL('../op-rank-time-openpangu-flash-events.html', import.meta.url),
  'utf8'
);

assert.match(page, /id="opRankIdeFrame"[^>]*data-surface="solid"/);
assert.match(page, /#opRankIdeFrame\[data-surface="solid"\]\s*\{/);
assert.match(page, /--ide-frame-pane-fill:var\(--ide-frame-pane-bg\)/);
assert.match(page, /--ide-frame-pane-backdrop-filter:none/);
assert.match(page, /--surface-backdrop-filter-soft:none/);

assert.match(page, /\[data-ui-surface\]\s*\{[\s\S]*?background:var\(--opv-ui-surface-bg\)[\s\S]*?box-shadow:none/);
assert.match(page, /\[data-ui-surface="panel"\]\{--opv-ui-surface-bg:var\(--opv-ui-panel-bg\)\}/);
assert.doesNotMatch(page, /(?:-webkit-)?backdrop-filter\s*:\s*blur\(/);

for (const id of [
  'parallelEventLegend',
  'executionPlayback',
  'objectPanel',
  'zoomControl',
  'lensSwitch',
  'sideConfigToggle',
  'filterToggle',
  'sideConfigPanel',
  'infoPanel',
  'swimZoomControl',
]) {
  assert.match(page, new RegExp(`id="${id}"[^>]*data-ui-surface="(?:toolbar|panel)"`));
}

assert.match(page, /class="opv-view-btns" data-ui-surface="toolbar"/);
assert.match(page, /class="opv-stage-tools" data-stage-ui="stage-tools"/);

for (const id of ['parallelEventLegend', 'executionPlayback', 'objectPanel']) {
  assert.match(page, new RegExp(`id="${id}"[^>]*data-stage-ui="[^"]+"`));
}

console.log('UI solid surface contract tests passed');
