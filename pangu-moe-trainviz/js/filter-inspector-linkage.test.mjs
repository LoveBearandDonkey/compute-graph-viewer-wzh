import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const page = readFileSync(
  new URL('../op-rank-time-openpangu-flash-events.html', import.meta.url),
  'utf8'
);

const filterHandler = page.match(
  /objectPanel\?\.addEventListener\('click',[\s\S]*?\n}\);\nfilterToggle/
)?.[0];
const tabHandler = page.match(
  /function setInspectorTab\(id\)[\s\S]*?\n}\ninspectorTabs\?\.addEventListener/
)?.[0];

assert.ok(filterHandler, 'filter click handler should exist');
assert.match(filterHandler, /setInspectorOpen\(true\);/, 'filter selection should open Inspector');
assert.match(filterHandler, /setInspectorTab\('focus'\);/, 'filter selection should activate Focus');
assert.match(filterHandler, /renderObjectFocusInspector\(\);/, 'filter selection should render linked content');
assert.match(filterHandler, /setObjectPanelOpen\(false\);/, 'filter popover should close after selection');

assert.ok(tabHandler, 'Inspector tab handler should exist');
assert.match(tabHandler, /if\(activeInspectorTab==='focus'\)renderObjectFocusInspector\(\);/);
assert.match(tabHandler, /if\(inspectorBody\)inspectorBody\.scrollTop=0;/, 'tab change should reveal content from the top');
assert.match(tabHandler, /frameBootstrap\?\.scheduleLayout\?\./, 'tab change should schedule frame layout');

console.log('filter-inspector linkage tests passed');
