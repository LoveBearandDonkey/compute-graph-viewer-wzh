import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const page = readFileSync(
  new URL('../op-rank-time-openpangu-flash-events.html', import.meta.url),
  'utf8'
);

assert.match(page, /id="parallelEventLegend" data-stage-ui="parallel-events" data-ui-surface="toolbar" aria-label="并行事件筛选"/);
assert.doesNotMatch(page, /id="parallelEventLegend"[^>]*data-stage-top=/);
assert.doesNotMatch(page, /id="executionPlayback"[^>]*data-stage-top=/);
assert.match(page, /\.opv-parallel-event-legend\{[^}]*top:12px/);
assert.match(page, /\.opv-layer-playback\{[^}]*top:12px/);
assert.match(page, /function bindStageUiSurfaces\(root=document\)/);
assert.match(page, /surface\.dataset\.stageUiBound==='true'/);
assert.match(page, /\['pointerdown','pointermove','pointerup','click','dblclick','contextmenu'\]/);
assert.match(page, /surface\.addEventListener\('wheel',stop,\{passive:true\}\)/);
assert.match(page, /event\.target\.closest\('\[data-stage-ui\]'\)/);
assert.match(page, /if\(e\.target\.closest\?\.\('\[data-stage-ui\]'\)\)return;/);
assert.doesNotMatch(page, /setCssGraphView\(tag==='PP'\?'right':'front'\)/, 'PP filter must not force the side view');
assert.match(page, /if\(tag!=='PP'\)setCssGraphView\('front'\);/, 'non-PP event filters should preserve their existing front-view behavior');
assert.match(page, /viewIso\?\.addEventListener\('click',\(\)=>setCssGraphView\('iso'\)\);/);
assert.match(page, /viewFront\?\.addEventListener\('click',\(\)=>setCssGraphView\('front'\)\);/);
assert.match(page, /viewRight\?\.addEventListener\('click',\(\)=>setCssGraphView\('right'\)\);/);

const positionFunction = page.match(
  /function positionStageTopControl\(control\)[\s\S]*?\n}\nfunction syncExecutionPlaybackPosition/
)?.[0];
assert.ok(positionFunction, 'stage control positioning helper should exist');
assert.match(positionFunction, /const toolbarTop=toolbarRect\?toolbarRect\.top-parentRect\.top:safeInset;/);
assert.match(positionFunction, /const baseTop=Number\.isFinite\(configuredTop\)\?Math\.max\(safeInset,configuredTop\):Math\.max\(safeInset,toolbarTop\);/);
assert.doesNotMatch(positionFunction, /\|\|86/);
assert.match(positionFunction, /if\(control\.dataset\.stagePosition!==positionSignature\)/);
assert.match(positionFunction, /control\.dataset\.stagePosition=positionSignature;/);

console.log('parallel-event control tests passed');
