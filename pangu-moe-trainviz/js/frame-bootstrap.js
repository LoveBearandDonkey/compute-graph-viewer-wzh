(function bootstrapOpenPanguFrame(global) {
  'use strict';

  const GLOBAL_NAME = 'OpenPanguFrameBootstrap';
  const VERSION = '1.0.0';
  const FRAME_SELECTOR = '#opRankIdeFrame, [data-ide-frame]';
  const CALLBACK_NAMES = ['rendererResize', 'axisResize', 'dockResize', 'redraw'];
  const STORAGE_KEYS = Object.freeze({
    inspectorOpen: 'op-rank-time-inspector-open',
    analysisOpen: 'op-rank-time-analysis-open',
  });

  if (global[GLOBAL_NAME]?.version) {
    global[GLOBAL_NAME].boot?.();
    return;
  }

  let frame = null;
  let nativeController = null;
  let booted = false;
  let rafId = 0;
  let flushing = false;
  let rerunRequested = false;
  let resizeObserver = null;
  let visibilityObserver = null;
  const cleanupFns = [];
  const layoutReasons = new Set();
  const callbackRegistries = Object.fromEntries(
    CALLBACK_NAMES.map((name) => [name, new Map()]),
  );
  const observedElements = new Map();

  function withStorage(operation, fallback) {
    try {
      return operation(global.localStorage);
    } catch (_error) {
      return fallback;
    }
  }

  const storage = Object.freeze({
    get(key, fallback = null) {
      const value = withStorage((store) => store.getItem(key), null);
      return value == null ? fallback : value;
    },
    set(key, value) {
      return withStorage((store) => {
        store.setItem(key, String(value));
        return true;
      }, false);
    },
    remove(key) {
      return withStorage((store) => {
        store.removeItem(key);
        return true;
      }, false);
    },
    getBoolean(key, fallback = false) {
      const value = this.get(key, null);
      if (value == null) return !!fallback;
      const normalized = String(value).trim().toLowerCase();
      if (['true', '1', 'yes', 'on'].includes(normalized)) return true;
      if (['false', '0', 'no', 'off'].includes(normalized)) return false;
      return !!fallback;
    },
    setBoolean(key, value) {
      return this.set(key, value ? 'true' : 'false');
    },
    getJSON(key, fallback = null) {
      const value = this.get(key, null);
      if (value == null) return fallback;
      try {
        return JSON.parse(value);
      } catch (_error) {
        return fallback;
      }
    },
    setJSON(key, value) {
      try {
        return this.set(key, JSON.stringify(value));
      } catch (_error) {
        return false;
      }
    },
  });

  function resolveElement(target, root = document) {
    if (!target) return null;
    if (typeof target === 'string') return root.querySelector(target);
    return target.nodeType === 1 ? target : null;
  }

  function resolveFrame() {
    frame = frame && frame.isConnected ? frame : document.querySelector(FRAME_SELECTOR);
    return frame;
  }

  function elementIsVisible(element) {
    if (!element || !element.isConnected || document.hidden) return false;
    if (element.hidden || element.getAttribute('aria-hidden') === 'true') return false;
    const styles = global.getComputedStyle?.(element);
    if (styles?.display === 'none' || styles?.visibility === 'hidden') return false;
    return element.getClientRects().length > 0;
  }

  function panelElements(kind) {
    const root = resolveFrame();
    if (!root) return { root: null, pane: null, toggle: null };
    if (kind === 'inspector') {
      return {
        root,
        pane: root.querySelector('#inspectPane, [data-ide-pane="inspector"]'),
        toggle: root.querySelector('#inspectorToggle'),
      };
    }
    return {
      root,
      pane: root.querySelector('[data-ide-pane="analysis-dock"]'),
      toggle: root.querySelector('#analysisToggle'),
    };
  }

  function applyPanelState(kind, open) {
    const elements = panelElements(kind);
    if (!elements.root) return false;
    const isInspector = kind === 'inspector';
    const datasetKey = isInspector ? 'inspectorOpen' : 'analysisOpen';
    const openTitle = isInspector ? '隐藏 Inspect 面板' : '隐藏底部分析面板';
    const closedTitle = isInspector ? '打开 Inspect 面板' : '打开底部分析面板';
    const normalized = !!open;

    elements.root.dataset[datasetKey] = normalized ? 'true' : 'false';
    elements.pane?.setAttribute('aria-hidden', normalized ? 'false' : 'true');
    if (elements.toggle) {
      elements.toggle.classList.toggle('is-active', normalized);
      elements.toggle.setAttribute('aria-expanded', normalized ? 'true' : 'false');
      elements.toggle.title = normalized ? openTitle : closedTitle;
      elements.toggle.setAttribute('aria-label', elements.toggle.title);
    }
    return normalized;
  }

  function getPanelOpen(kind) {
    const root = resolveFrame();
    if (!root) return false;
    const datasetKey = kind === 'inspector' ? 'inspectorOpen' : 'analysisOpen';
    return root.dataset[datasetKey] !== 'false';
  }

  function restoreFrameState() {
    const root = resolveFrame();
    if (!root) return null;
    const inspectorDefault = root.dataset.inspectorOpen !== 'false';
    const analysisDefault = root.dataset.analysisOpen !== 'false';
    const state = {
      inspectorOpen: storage.getBoolean(STORAGE_KEYS.inspectorOpen, inspectorDefault),
      analysisOpen: storage.getBoolean(STORAGE_KEYS.analysisOpen, analysisDefault),
    };
    applyPanelState('inspector', state.inspectorOpen);
    applyPanelState('analysis', state.analysisOpen);
    return state;
  }

  function dispatchFrameEvent(name, detail) {
    const root = resolveFrame();
    if (!root || typeof global.CustomEvent !== 'function') return;
    root.dispatchEvent(new CustomEvent(name, { detail, bubbles: true }));
  }

  function layoutContext(timestamp, reasons) {
    const root = resolveFrame();
    const stagePane = root?.querySelector('[data-ide-pane="editor-preview"]') || null;
    const analysisPane = root?.querySelector('[data-ide-pane="analysis-dock"]') || null;
    const analysisDock = root?.querySelector('#analysisDock') || null;
    return {
      timestamp,
      reasons,
      frame: root,
      stagePane,
      analysisPane,
      analysisDock,
      inspectorOpen: getPanelOpen('inspector'),
      analysisOpen: getPanelOpen('analysis'),
      frameVisible: elementIsVisible(root),
      stageVisible: elementIsVisible(stagePane),
      analysisVisible: getPanelOpen('analysis') && elementIsVisible(analysisPane),
      isVisible: elementIsVisible,
    };
  }

  function reportCallbackError(name, error, context) {
    global.console?.error?.(`[${GLOBAL_NAME}] ${name} callback failed`, error);
    if (typeof global.CustomEvent === 'function') {
      global.dispatchEvent(new CustomEvent('openpangu-frame-layout-error', {
        detail: { name, error, reasons: context.reasons },
      }));
    }
  }

  function runCallbacks(name, context) {
    if (name === 'rendererResize' && !context.stageVisible) return;
    if (name === 'axisResize' && !context.stageVisible) return;
    if (name === 'dockResize' && !context.analysisVisible) return;
    if (name === 'redraw' && !context.frameVisible) return;
    Array.from(callbackRegistries[name].keys()).forEach((callback) => {
      try {
        callback(context);
      } catch (error) {
        reportCallbackError(name, error, context);
      }
    });
  }

  function flushLayout(timestamp) {
    rafId = 0;
    if (document.hidden) return;
    const reasons = Array.from(layoutReasons);
    layoutReasons.clear();
    const context = layoutContext(timestamp, reasons);
    flushing = true;
    CALLBACK_NAMES.forEach((name) => runCallbacks(name, context));
    flushing = false;
    dispatchFrameEvent('openpangu-frame-layout', context);
    if (rerunRequested || layoutReasons.size) {
      rerunRequested = false;
      scheduleLayout('nested-layout-request');
    }
  }

  function scheduleLayout(reason = 'manual') {
    layoutReasons.add(String(reason || 'manual'));
    if (flushing) {
      rerunRequested = true;
      return 0;
    }
    if (document.hidden || rafId) return rafId;
    rafId = global.requestAnimationFrame(flushLayout);
    return rafId;
  }

  function registerLayoutCallbacks(callbacks, options = {}) {
    const registrations = [];
    CALLBACK_NAMES.forEach((name) => {
      const callback = callbacks?.[name];
      if (typeof callback !== 'function') return;
      const registry = callbackRegistries[name];
      registry.set(callback, (registry.get(callback) || 0) + 1);
      registrations.push([name, callback]);
    });
    let active = true;
    const unregister = () => {
      if (!active) return;
      active = false;
      registrations.forEach(([name, callback]) => {
        const registry = callbackRegistries[name];
        const count = registry.get(callback) || 0;
        if (count <= 1) registry.delete(callback);
        else registry.set(callback, count - 1);
      });
    };
    if (options.immediate !== false) scheduleLayout(options.reason || 'callbacks-registered');
    return unregister;
  }

  function resizeEntryChanged(entry, record) {
    const width = Math.round((entry.contentRect?.width || 0) * 100) / 100;
    const height = Math.round((entry.contentRect?.height || 0) * 100) / 100;
    if (record.width === width && record.height === height) return false;
    record.width = width;
    record.height = height;
    return true;
  }

  function ensureResizeObserver() {
    if (resizeObserver || typeof global.ResizeObserver !== 'function') return resizeObserver;
    resizeObserver = new ResizeObserver((entries) => {
      const changedLabels = new Set();
      entries.forEach((entry) => {
        const record = observedElements.get(entry.target);
        if (record && resizeEntryChanged(entry, record)) changedLabels.add(record.label);
      });
      if (changedLabels.size) scheduleLayout(`resize:${Array.from(changedLabels).join(',')}`);
    });
    return resizeObserver;
  }

  function observeResize(target, label = 'element') {
    const element = resolveElement(target, resolveFrame() || document);
    if (!element) return () => {};
    const existing = observedElements.get(element);
    if (existing) {
      existing.count += 1;
    } else {
      observedElements.set(element, {
        count: 1,
        label: String(label || 'element'),
        width: null,
        height: null,
      });
      ensureResizeObserver()?.observe(element);
    }
    let active = true;
    return () => {
      if (!active) return;
      active = false;
      const record = observedElements.get(element);
      if (!record) return;
      record.count -= 1;
      if (record.count > 0) return;
      resizeObserver?.unobserve(element);
      observedElements.delete(element);
    };
  }

  function refreshFrame(reason = 'frame-refresh') {
    nativeController?.refresh?.();
    scheduleLayout(reason);
  }

  function setPanelOpen(kind, open, options = {}) {
    if (kind !== 'inspector' && kind !== 'analysis') return false;
    const normalized = applyPanelState(kind, open);
    if (options.persist !== false) {
      storage.setBoolean(
        kind === 'inspector' ? STORAGE_KEYS.inspectorOpen : STORAGE_KEYS.analysisOpen,
        normalized,
      );
    }
    refreshFrame(`${kind}-panel-${normalized ? 'opened' : 'closed'}`);
    dispatchFrameEvent('openpangu-frame-panel-change', { kind, open: normalized });
    return normalized;
  }

  function togglePanel(kind, options) {
    return setPanelOpen(kind, !getPanelOpen(kind), options);
  }

  function initNativeFrame() {
    const root = resolveFrame();
    if (!root || nativeController) return nativeController;
    const helper = global.PtoIdeFrame;
    if (!helper?.init) return null;
    nativeController = helper.init(root, {
      splitOptions: {
        default: {
          onResize(_sizes, meta) {
            scheduleLayout(`split-${meta?.direction || 'resize'}:${meta?.phase || 'change'}`);
          },
        },
      },
    });
    return nativeController;
  }

  function installObservers() {
    const root = resolveFrame();
    if (!root) return;
    cleanupFns.push(observeResize(root, 'frame'));
    cleanupFns.push(observeResize(root.querySelector('[data-ide-pane="editor-preview"]'), 'stage'));
    cleanupFns.push(observeResize(root.querySelector('[data-ide-pane="analysis-dock"]'), 'analysis-pane'));
    cleanupFns.push(observeResize(root.querySelector('#analysisDock'), 'analysis-dock'));

    if (typeof global.MutationObserver === 'function') {
      visibilityObserver = new MutationObserver(() => scheduleLayout('panel-visibility'));
      visibilityObserver.observe(root, {
        attributes: true,
        attributeFilter: ['data-inspector-open', 'data-analysis-open'],
      });
      [
        root.querySelector('[data-ide-pane="inspector"]'),
        root.querySelector('[data-ide-pane="analysis-dock"]'),
      ].filter(Boolean).forEach((pane) => visibilityObserver.observe(pane, {
        attributes: true,
        attributeFilter: ['hidden', 'aria-hidden'],
      }));
    }

    const onWindowResize = () => scheduleLayout('window-resize');
    const onVisibilityChange = () => {
      if (!document.hidden) refreshFrame('document-visible');
    };
    const onPageShow = () => refreshFrame('page-show');
    const onBottomPanelChange = () => refreshFrame('bottom-panel-change');
    global.addEventListener('resize', onWindowResize, { passive: true });
    global.addEventListener('orientationchange', onWindowResize, { passive: true });
    global.addEventListener('pageshow', onPageShow);
    global.addEventListener('load', onPageShow, { once: true });
    document.addEventListener('visibilitychange', onVisibilityChange);
    root.addEventListener('pto-ide-bottom-panel-change', onBottomPanelChange);
    cleanupFns.push(
      () => global.removeEventListener('resize', onWindowResize),
      () => global.removeEventListener('orientationchange', onWindowResize),
      () => global.removeEventListener('pageshow', onPageShow),
      () => global.removeEventListener('load', onPageShow),
      () => document.removeEventListener('visibilitychange', onVisibilityChange),
      () => root.removeEventListener('pto-ide-bottom-panel-change', onBottomPanelChange),
    );

    const fontsReady = document.fonts?.ready;
    if (fontsReady?.then) {
      fontsReady.then(() => {
        if (booted) scheduleLayout('fonts-ready');
      }).catch(() => {});
    }
  }

  function boot() {
    if (booted) return api;
    if (!resolveFrame()) return null;
    restoreFrameState();
    initNativeFrame();
    installObservers();
    booted = true;
    scheduleLayout('bootstrap');
    dispatchFrameEvent('openpangu-frame-bootstrap-ready', {
      api,
      controller: nativeController,
    });
    return api;
  }

  function destroy(options = {}) {
    if (rafId) global.cancelAnimationFrame(rafId);
    rafId = 0;
    visibilityObserver?.disconnect();
    visibilityObserver = null;
    resizeObserver?.disconnect();
    resizeObserver = null;
    observedElements.clear();
    cleanupFns.splice(0).forEach((cleanup) => cleanup());
    CALLBACK_NAMES.forEach((name) => callbackRegistries[name].clear());
    if (options.destroyFrame === true) nativeController?.destroy?.();
    if (options.destroyFrame === true) nativeController = null;
    layoutReasons.clear();
    booted = false;
  }

  const api = {
    version: VERSION,
    STORAGE_KEYS,
    storage,
    boot,
    destroy,
    restoreFrameState,
    scheduleLayout,
    registerLayoutCallbacks,
    observeResize,
    refreshFrame,
    getPanelOpen,
    setPanelOpen,
    togglePanel,
    setInspectorOpen: (open, options) => setPanelOpen('inspector', open, options),
    setAnalysisOpen: (open, options) => setPanelOpen('analysis', open, options),
    get frame() {
      return resolveFrame();
    },
    get controller() {
      return nativeController;
    },
    get ready() {
      return booted;
    },
  };

  global[GLOBAL_NAME] = api;

  if (!boot() && document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot, { once: true });
  }
})(window);
