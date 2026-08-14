/**
 * Framework-free interaction state for the OpenPangu training visualizer.
 *
 * The durable selection is intentionally separate from hoverPreview: a hover
 * must never erase the user's pinned selection. Domain payloads stay open so
 * callers can migrate existing model, finding, card, rank, and expert records
 * without teaching this module about Three.js or the DOM.
 */

export const INTERACTION_SLICES = Object.freeze([
  'selection',
  'filter',
  'hoverPreview',
  'view',
  'time',
]);

export const DEFAULT_INTERACTION_STATE = Object.freeze({
  selection: null,
  filter: Object.freeze({ objectFocus: 'all' }),
  hoverPreview: null,
  view: Object.freeze({ lens: 'structure', camera: 'iso' }),
  time: Object.freeze({ step: null, tick: null, microbatch: null, window: null }),
});

const hasOwn = (value, key) => Object.prototype.hasOwnProperty.call(value, key);
const isObject = value => value !== null && typeof value === 'object';
const isPlainObject = value => {
  if (!isObject(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
};

function cloneSnapshotValue(value, seen = new WeakMap()) {
  if (!isObject(value)) return value;
  // DOM nodes, Three.js objects, and other class instances are identity values.
  // Preserve them instead of attempting a lossy structured clone.
  if (!Array.isArray(value) && !isPlainObject(value)) return value;
  if (seen.has(value)) return seen.get(value);
  const clone = Array.isArray(value) ? [] : {};
  seen.set(value, clone);
  Reflect.ownKeys(value).forEach(key => {
    clone[key] = cloneSnapshotValue(value[key], seen);
  });
  return clone;
}

function normalizeSlice(slice, value) {
  if (slice === 'filter' && typeof value === 'string') return { objectFocus: value || 'all' };
  return value;
}

function mergeValue(current, incoming) {
  if (isPlainObject(current) && isPlainObject(incoming)) return { ...current, ...incoming };
  return incoming;
}

function sameShallowValue(left, right) {
  if (Object.is(left, right)) return true;
  if (!isPlainObject(left) || !isPlainObject(right)) return false;
  const leftKeys = Reflect.ownKeys(left);
  const rightKeys = Reflect.ownKeys(right);
  if (leftKeys.length !== rightKeys.length) return false;
  return leftKeys.every(key => hasOwn(right, key) && Object.is(left[key], right[key]));
}

function normalizeInitialState(initialState = {}) {
  const state = {};
  INTERACTION_SLICES.forEach(slice => {
    if (!hasOwn(initialState, slice)) {
      state[slice] = DEFAULT_INTERACTION_STATE[slice];
      return;
    }
    const value = normalizeSlice(slice, initialState[slice]);
    const fallback = DEFAULT_INTERACTION_STATE[slice];
    state[slice] = isPlainObject(fallback) && isPlainObject(value)
      ? { ...fallback, ...value }
      : value;
  });
  return state;
}

function updateSlice(state, slice, value, mode = 'merge') {
  if (!INTERACTION_SLICES.includes(slice)) return state;
  const normalized = normalizeSlice(slice, value);
  const nextValue = mode === 'replace' ? normalized : mergeValue(state[slice], normalized);
  if (sameShallowValue(state[slice], nextValue)) return state;
  return { ...state, [slice]: nextValue };
}

/**
 * Pure reducer. Supported actions:
 *   {type:'slice/set', slice, payload, mode:'merge'|'replace'}
 *   {type:'selection/set'|'filter/set'|'preview/set'|'view/set'|'time/set', ...}
 *   {type:'selection/clear'|'preview/clear'}
 *   {type:'state/restore', payload:{selection, filter, ...}}
 */
export function interactionReducer(state = normalizeInitialState(), action = {}) {
  const mode = action.mode === 'replace' ? 'replace' : 'merge';
  switch (action.type) {
    case 'slice/set':
      return updateSlice(state, action.slice, action.payload, mode);
    case 'selection/set':
      return updateSlice(state, 'selection', action.payload, action.mode || 'replace');
    case 'filter/set':
      return updateSlice(state, 'filter', action.payload, mode);
    case 'preview/set':
      return updateSlice(state, 'hoverPreview', action.payload, action.mode || 'replace');
    case 'view/set':
      return updateSlice(state, 'view', action.payload, mode);
    case 'time/set':
      return updateSlice(state, 'time', action.payload, mode);
    case 'selection/clear':
      return updateSlice(state, 'selection', null, 'replace');
    case 'preview/clear':
      return updateSlice(state, 'hoverPreview', null, 'replace');
    case 'state/restore': {
      const snapshot = action.payload?.state || action.payload;
      if (!isObject(snapshot)) return state;
      return INTERACTION_SLICES.reduce((next, slice) => {
        if (!hasOwn(snapshot, slice)) return next;
        return updateSlice(next, slice, snapshot[slice], 'replace');
      }, state);
    }
    default:
      return state;
  }
}

function changedSlices(previousState, nextState) {
  return INTERACTION_SLICES.filter(slice => !Object.is(previousState[slice], nextState[slice]));
}

function actionMeta(action, fallbackSource) {
  const raw = isPlainObject(action.meta) ? action.meta : {};
  return { ...raw, source: raw.source || action.source || fallbackSource || 'unknown' };
}

/**
 * Create a small synchronous store. Calls inside batch() are reduced
 * immediately but subscribers receive one consolidated notification.
 */
export function createInteractionStore(initialState = {}, options = {}) {
  let state = normalizeInitialState(initialState);
  let revision = 0;
  let batchDepth = 0;
  let pending = null;
  let previewSequence = 0;
  const subscriptions = new Set();
  const previewSessions = [];
  const defaultSource = options.source || 'interaction-store';
  const onSubscriberError = typeof options.onSubscriberError === 'function'
    ? options.onSubscriberError
    : (error => console.error('interaction-state subscriber', error));

  function notify(change) {
    subscriptions.forEach(subscription => {
      const nextSelected = subscription.selector(state);
      if (!subscription.immediatePending && subscription.equality(nextSelected, subscription.selected)) return;
      const previousSelected = subscription.selected;
      subscription.selected = nextSelected;
      subscription.immediatePending = false;
      try {
        subscription.listener(nextSelected, {
          ...change,
          state,
          selected: nextSelected,
          previousSelected,
        });
      } catch (error) {
        onSubscriberError(error, change);
      }
    });
  }

  function flush() {
    if (!pending || batchDepth > 0) return;
    const aggregate = pending;
    pending = null;
    revision += 1;
    const sources = [...new Set(aggregate.actions.map(item => item.meta.source))];
    notify({
      revision,
      source: sources.length === 1 ? sources[0] : 'batch',
      sources,
      meta: aggregate.actions[aggregate.actions.length - 1]?.meta || { source: defaultSource },
      actions: aggregate.actions,
      actionTypes: aggregate.actions.map(item => item.action.type),
      changedSlices: changedSlices(aggregate.previousState, state),
      previousState: aggregate.previousState,
    });
  }

  function dispatch(action = {}) {
    const previousState = state;
    const nextState = interactionReducer(state, action);
    if (Object.is(previousState, nextState)) return state;
    state = nextState;
    const meta = actionMeta(action, defaultSource);
    if (!pending) pending = { previousState, actions: [] };
    pending.actions.push({ action, meta });
    flush();
    return state;
  }

  function batch(callback, meta = {}) {
    if (typeof callback !== 'function') throw new TypeError('batch(callback) expects a function');
    batchDepth += 1;
    try {
      return callback({ dispatch, getState });
    } finally {
      batchDepth -= 1;
      // A batch-level source is useful when its inner actions use generic sources.
      if (pending && meta.source) {
        pending.actions = pending.actions.map(item => ({
          ...item,
          meta: item.meta.source === defaultSource ? { ...item.meta, ...meta } : item.meta,
        }));
      }
      flush();
    }
  }

  function getState() {
    return state;
  }

  function subscribe(listener, config = {}) {
    if (typeof listener !== 'function') throw new TypeError('subscribe(listener) expects a function');
    const selector = typeof config.selector === 'function' ? config.selector : value => value;
    const subscription = {
      listener,
      selector,
      equality: typeof config.equality === 'function' ? config.equality : Object.is,
      selected: selector(state),
      immediatePending: Boolean(config.immediate),
    };
    subscriptions.add(subscription);
    if (subscription.immediatePending) {
      notify({
        revision,
        source: 'subscribe',
        sources: ['subscribe'],
        meta: { source: 'subscribe' },
        actions: [],
        actionTypes: [],
        changedSlices: [],
        previousState: state,
      });
    }
    return () => subscriptions.delete(subscription);
  }

  function setSlice(slice, payload, config = {}) {
    return dispatch({
      type: 'slice/set',
      slice,
      payload,
      mode: config.mode || 'merge',
      meta: config.meta || { source: config.source || defaultSource },
    });
  }

  function snapshot(slices = INTERACTION_SLICES) {
    const included = Array.isArray(slices) ? slices : [slices];
    const snapshotState = {};
    included.forEach(slice => {
      if (INTERACTION_SLICES.includes(slice)) snapshotState[slice] = cloneSnapshotValue(state[slice]);
    });
    return Object.freeze({
      kind: 'interaction-state-snapshot',
      revision,
      state: Object.freeze(snapshotState),
    });
  }

  function restore(savedSnapshot, config = {}) {
    if (savedSnapshot?.kind !== 'interaction-state-snapshot' || !isObject(savedSnapshot.state)) {
      throw new TypeError('restore(snapshot) expects a snapshot created by this store');
    }
    return dispatch({
      type: 'state/restore',
      payload: cloneSnapshotValue(savedSnapshot.state),
      mode: 'replace',
      meta: config.meta || { source: config.source || 'preview:restore' },
    });
  }

  function beginPreview(payload, config = {}) {
    const token = Object.freeze({
      kind: 'interaction-preview-token',
      id: ++previewSequence,
      snapshot: snapshot(config.snapshotSlices || INTERACTION_SLICES),
    });
    previewSessions.push(token);
    dispatch({
      type: 'preview/set',
      payload,
      mode: config.mode || 'replace',
      meta: config.meta || { source: config.source || 'preview:begin' },
    });
    return token;
  }

  function endPreview(token = previewSessions[previewSessions.length - 1], config = {}) {
    const index = previewSessions.lastIndexOf(token);
    if (index < 0) return false;
    previewSessions.splice(index, 1);
    batch(() => {
      if (config.restore !== false) {
        restore(token.snapshot, { source: config.source || 'preview:restore' });
      } else {
        dispatch({
          type: 'preview/clear',
          meta: config.meta || { source: config.source || 'preview:commit' },
        });
      }
    }, { source: config.source || (config.restore === false ? 'preview:commit' : 'preview:restore') });
    return true;
  }

  return Object.freeze({
    getState,
    getRevision: () => revision,
    dispatch,
    batch,
    transact: batch,
    subscribe,
    snapshot,
    restore,
    beginPreview,
    endPreview,
    setSelection: (selection, config = {}) => setSlice('selection', selection, { mode: 'replace', ...config }),
    clearSelection: (config = {}) => dispatch({ type: 'selection/clear', meta: config.meta || { source: config.source || defaultSource } }),
    setFilter: (filter, config = {}) => setSlice('filter', filter, config),
    setObjectFocus: (objectFocus, config = {}) => setSlice('filter', { objectFocus: objectFocus || 'all' }, config),
    setHoverPreview: (preview, config = {}) => setSlice('hoverPreview', preview, { mode: 'replace', ...config }),
    clearPreview: (config = {}) => dispatch({ type: 'preview/clear', meta: config.meta || { source: config.source || defaultSource } }),
    setView: (view, config = {}) => setSlice('view', view, config),
    setLens: (lens, config = {}) => setSlice('view', { lens }, config),
    setTime: (time, config = {}) => setSlice('time', time, config),
    setTimeContext: (time, config = {}) => setSlice('time', time, config),
  });
}
