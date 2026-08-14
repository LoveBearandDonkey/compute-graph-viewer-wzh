import assert from 'node:assert/strict';

class FakeElement {
  constructor() {
    this.nodeType = 1;
    this.isConnected = true;
    this.hidden = false;
    this.dataset = {};
    this.attributes = new Map();
    this.listeners = new Map();
    this.classList = { toggle() {} };
    this.title = '';
  }

  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  getAttribute(name) { return this.attributes.get(name) ?? null; }
  addEventListener(type, listener) { this.listeners.set(type, listener); }
  removeEventListener(type, listener) {
    if (this.listeners.get(type) === listener) this.listeners.delete(type);
  }
  dispatchEvent() { return true; }
  getClientRects() { return this.hidden ? [] : [{}]; }
}

const frame = new FakeElement();
frame.dataset = { inspectorOpen: 'true', analysisOpen: 'true' };
const inspector = new FakeElement();
const analysisPane = new FakeElement();
const analysisDock = new FakeElement();
const stagePane = new FakeElement();
const inspectorToggle = new FakeElement();
const analysisToggle = new FakeElement();
frame.querySelector = selector => {
  if (selector.includes('editor-preview')) return stagePane;
  if (selector.includes('#inspectPane') || selector.includes('pane="inspector"')) return inspector;
  if (selector.includes('#analysisDock')) return analysisDock;
  if (selector.includes('analysis-dock')) return analysisPane;
  if (selector.includes('#inspectorToggle')) return inspectorToggle;
  if (selector.includes('#analysisToggle')) return analysisToggle;
  return null;
};

const stored = new Map([
  ['op-rank-time-analysis-open', 'false'],
  ['op-rank-time-inspector-open', 'true'],
]);
const frames = new Map();
let nextFrame = 1;
let refreshCount = 0;
let initOptions = null;
const windowListeners = new Map();

globalThis.CustomEvent = class CustomEvent {
  constructor(type, options = {}) { this.type = type; this.detail = options.detail; }
};
globalThis.document = {
  hidden: false,
  readyState: 'complete',
  fonts: { ready: Promise.resolve() },
  querySelector: () => frame,
  addEventListener() {},
  removeEventListener() {},
};
globalThis.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};
globalThis.MutationObserver = class MutationObserver {
  observe() {}
  disconnect() {}
};

const windowMock = {
  localStorage: {
    getItem: key => stored.get(key) ?? null,
    setItem: (key, value) => stored.set(key, String(value)),
    removeItem: key => stored.delete(key),
  },
  requestAnimationFrame(callback) {
    const id = nextFrame++;
    frames.set(id, callback);
    return id;
  },
  cancelAnimationFrame(id) { frames.delete(id); },
  getComputedStyle: () => ({ display: 'block', visibility: 'visible' }),
  addEventListener(type, listener) { windowListeners.set(type, listener); },
  removeEventListener(type, listener) {
    if (windowListeners.get(type) === listener) windowListeners.delete(type);
  },
  CustomEvent: globalThis.CustomEvent,
  ResizeObserver: globalThis.ResizeObserver,
  MutationObserver: globalThis.MutationObserver,
  PtoIdeFrame: {
    init(_frame, options) {
      initOptions = options;
      return { refresh: () => { refreshCount += 1; } };
    },
  },
  console,
};
globalThis.window = windowMock;

const flushFrames = () => {
  const pending = [...frames.values()];
  frames.clear();
  pending.forEach(callback => callback(0));
};

await import(`./frame-bootstrap.js?test=${Date.now()}`);
const api = windowMock.OpenPanguFrameBootstrap;
assert.ok(api?.ready);
assert.ok(initOptions?.splitOptions?.default?.onResize);
assert.equal(frame.dataset.analysisOpen, 'false');
assert.equal(frame.dataset.inspectorOpen, 'true');

let redraws = 0;
api.registerLayoutCallbacks({ redraw: () => { redraws += 1; } });
api.scheduleLayout('test-a');
api.scheduleLayout('test-b');
assert.equal(frames.size, 1, 'layout requests must coalesce into one animation frame');
flushFrames();
assert.equal(redraws, 1);

api.setAnalysisOpen(true, { persist: true });
assert.equal(frame.dataset.analysisOpen, 'true');
assert.equal(stored.get('op-rank-time-analysis-open'), 'true');
assert.equal(refreshCount, 1);
flushFrames();

api.destroy();
assert.equal(api.ready, false);

console.log('frame-bootstrap tests passed');
