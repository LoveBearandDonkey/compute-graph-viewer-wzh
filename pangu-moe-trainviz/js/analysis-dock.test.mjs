import assert from 'node:assert/strict';

class FakeElement {
  constructor() {
    this.hidden = false;
    this.visible = false;
    this.dataset = {};
    this.children = [];
    this.listeners = new Map();
    this.classList = { toggle() {} };
    this.textContent = '';
    this.innerHTML = '';
  }

  appendChild(child) { this.children.push(child); return child; }
  addEventListener(type, listener) { this.listeners.set(type, listener); }
  removeEventListener(type, listener) {
    if (this.listeners.get(type) === listener) this.listeners.delete(type);
  }
  setAttribute(name, value) { this[name] = String(value); }
  getClientRects() { return this.visible && !this.hidden ? [{}] : []; }
}

const frames = new Map();
let nextFrame = 1;
globalThis.document = { createElement: () => new FakeElement() };
globalThis.localStorage = { setItem() {} };
globalThis.requestAnimationFrame = callback => {
  const id = nextFrame++;
  frames.set(id, callback);
  return id;
};
globalThis.cancelAnimationFrame = id => frames.delete(id);

function flushFrames() {
  const pending = [...frames.values()];
  frames.clear();
  pending.forEach(callback => callback(0));
}

const { createAnalysisDock } = await import('./analysis-dock.js');
const tabsRoot = new FakeElement();
const titleEl = new FakeElement();
const metaEl = new FakeElement();
const firstPanel = new FakeElement();
const secondPanel = new FakeElement();
const calls = [];
const view = (id, panel) => ({
  id,
  label: id,
  title: id,
  meta: `${id}-meta`,
  panel,
  mount: () => calls.push(`${id}:mount`),
  render: () => calls.push(`${id}:render`),
  resize: () => calls.push(`${id}:resize`),
  destroy: () => calls.push(`${id}:destroy`),
});

const controller = createAnalysisDock({
  tabsRoot,
  titleEl,
  metaEl,
  initialView: 'first',
  views: [view('first', firstPanel), view('second', secondPanel)],
});

flushFrames();
assert.deepEqual(calls, [], 'hidden dock must not mount or measure a view');

firstPanel.visible = true;
controller.resize();
assert.deepEqual(calls, ['first:mount', 'first:resize']);

controller.refresh();
assert.equal(calls.at(-1), 'first:render');

secondPanel.visible = true;
controller.setActiveView('second');
flushFrames();
assert.deepEqual(calls.slice(-2), ['second:mount', 'second:render']);

controller.destroy();
assert.ok(calls.includes('first:destroy'));
assert.ok(calls.includes('second:destroy'));

console.log('analysis-dock tests passed');
