import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

// PTO is intentionally not a Node package. Load the browser ES module without
// adding a package.json that could change script semantics for existing pages.
const moduleUrl = new URL('./interaction-state.js', import.meta.url);
const source = await readFile(moduleUrl, 'utf8');
const loaded = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const { createInteractionStore, interactionReducer } = loaded;

const store = createInteractionStore({
  view: { lens: 'numerics', camera: 'right' },
  time: { step: 100 },
});

assert.deepEqual(store.getState().filter, { objectFocus: 'all' });
assert.equal(store.getState().view.lens, 'numerics');
assert.equal(store.getState().time.step, 100);
assert.equal(store.getState().time.tick, null);

store.setSelection({ type: 'model', id: 'gate', layer: 10 }, { source: 'scene' });
store.setSelection({ severity: 3 }, { mode: 'merge', source: 'inspector' });
assert.deepEqual(store.getState().selection, {
  type: 'model', id: 'gate', layer: 10, severity: 3,
});

let notificationCount = 0;
let lastChange = null;
const unsubscribe = store.subscribe((_state, change) => {
  notificationCount += 1;
  lastChange = change;
});

store.batch(() => {
  store.setObjectFocus('comm:ep', { source: 'filter' });
  store.setLens('communication', { source: 'lens' });
  store.setTimeContext({ tick: 12, microbatch: 2 }, { source: 'playback' });
});

assert.equal(notificationCount, 1);
assert.deepEqual(lastChange.sources, ['filter', 'lens', 'playback']);
assert.deepEqual(lastChange.changedSlices, ['filter', 'view', 'time']);
assert.equal(store.getState().filter.objectFocus, 'comm:ep');
assert.equal(store.getState().view.camera, 'right');
assert.equal(store.getState().time.step, 100);

let lensNotifications = 0;
const unsubscribeLens = store.subscribe(() => { lensNotifications += 1; }, {
  selector: state => state.view.lens,
});
store.setTime({ step: 101 }, { source: 'playback' });
assert.equal(lensNotifications, 0);
store.setLens('execution', { source: 'lens' });
assert.equal(lensNotifications, 1);

const previewToken = store.beginPreview(
  { type: 'finding', id: 'finding-ep-a2a' },
  { source: 'inspector:hover' },
);
store.batch(() => {
  store.setObjectFocus('comm:dp', { source: 'inspector:hover' });
  store.setView({ camera: 'iso' }, { source: 'inspector:hover' });
  store.setSelection({ type: 'finding', id: 'temporary' }, { source: 'inspector:hover' });
});
assert.equal(store.getState().hoverPreview.id, 'finding-ep-a2a');
assert.equal(store.getState().filter.objectFocus, 'comm:dp');
assert.equal(store.getState().selection.id, 'temporary');

assert.equal(store.endPreview(previewToken, { restore: true }), true);
assert.equal(store.getState().hoverPreview, null);
assert.equal(store.getState().filter.objectFocus, 'comm:ep');
assert.equal(store.getState().view.camera, 'right');
assert.equal(store.getState().selection.id, 'gate');

const committedPreview = store.beginPreview({ id: 'candidate' });
store.setObjectFocus('comm:tp');
assert.equal(store.endPreview(committedPreview, { restore: false }), true);
assert.equal(store.getState().hoverPreview, null);
assert.equal(store.getState().filter.objectFocus, 'comm:tp');

store.clearSelection({ source: 'escape' });
store.clearPreview({ source: 'escape' });
assert.equal(store.getState().selection, null);
assert.equal(store.getState().hoverPreview, null);

const reducerState = interactionReducer(store.getState(), {
  type: 'filter/set', payload: { objectFocus: 'all', query: 'expert' }, mode: 'replace',
});
assert.deepEqual(reducerState.filter, { objectFocus: 'all', query: 'expert' });

unsubscribe();
unsubscribeLens();
console.log('interaction-state tests passed');
