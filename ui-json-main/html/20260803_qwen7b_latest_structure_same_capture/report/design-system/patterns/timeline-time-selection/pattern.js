(function registerPtoTimelineTimeSelectionPattern(global) {
  'use strict';

  const DEFAULT_DRAG_THRESHOLD = 4;

  function number(value, fallback = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function normalizeSelection(selection) {
    if (!selection) return null;
    const first = number(selection.start, NaN);
    const second = number(selection.end, NaN);
    if (!Number.isFinite(first) || !Number.isFinite(second)) return null;
    return {
      ...selection,
      start: Math.min(first, second),
      end: Math.max(first, second),
    };
  }

  function xToTime(x, width, start, end) {
    const safeWidth = Math.max(1, number(width, 1));
    const safeStart = number(start);
    const safeEnd = Math.max(safeStart, number(end, safeStart));
    const ratio = clamp(number(x) / safeWidth, 0, 1);
    return safeStart + (safeEnd - safeStart) * ratio;
  }

  function timeToX(time, width, start, end) {
    const safeWidth = Math.max(1, number(width, 1));
    const safeStart = number(start);
    const safeEnd = Math.max(safeStart + Number.EPSILON, number(end, safeStart));
    return clamp((number(time, safeStart) - safeStart) / (safeEnd - safeStart), 0, 1) * safeWidth;
  }

  function selectionFromEvent(event) {
    if (!event) return null;
    const start = number(event.ts ?? event.start, NaN);
    const explicitEnd = number(event.end, NaN);
    const duration = Math.max(0, number(event.dur ?? event.duration, 0));
    const end = Number.isFinite(explicitEnd) ? explicitEnd : start + duration;
    if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
    return normalizeSelection({ kind: 'event', start, end, event });
  }

  function formatAbsoluteUs(value, language = 'en') {
    const locale = language === 'zh' ? 'zh-CN' : 'en-US';
    return `${number(value).toLocaleString(locale, {
      minimumFractionDigits: 0,
      maximumFractionDigits: 3,
      useGrouping: true,
    })} µs`;
  }

  function formatDuration(value) {
    const us = Math.max(0, number(value));
    if (us >= 1_000_000) return `${(us / 1_000_000).toFixed(us >= 10_000_000 ? 1 : 2)} s`;
    if (us >= 1000) return `${(us / 1000).toFixed(us >= 10_000 ? 1 : 2)} ms`;
    if (us >= 1) return `${us.toFixed(us >= 100 ? 1 : 3)} µs`;
    return `${(us * 1000).toFixed(us >= 0.1 ? 1 : 2)} ns`;
  }

  function formatSelectionSummary(selection, language = 'en') {
    const normalized = normalizeSelection(selection);
    if (!normalized) return '';
    const values = [
      formatAbsoluteUs(normalized.start, language),
      formatAbsoluteUs(normalized.end, language),
      formatDuration(normalized.end - normalized.start),
    ];
    const labels = language === 'zh'
      ? ['开始', '结束', '持续']
      : ['Start', 'End', 'Duration'];
    return values.map((value, index) => `${labels[index]} ${value}`).join('  ·  ');
  }

  function createSelectionLayer(root) {
    if (!root) throw new Error('Timeline selection layer root is required');
    const layer = document.createElement('div');
    layer.className = 'pto-timeline-time-selection';
    layer.hidden = true;
    layer.setAttribute('aria-hidden', 'true');
    layer.innerHTML = `
      <span class="pto-timeline-time-selection__fill"></span>
      <span class="pto-timeline-time-selection__marker is-start"></span>
      <span class="pto-timeline-time-selection__marker is-end"></span>
    `;
    root.appendChild(layer);
    return layer;
  }

  function updateSelectionLayer(layer, selection, geometry = {}) {
    if (!layer) return;
    const normalized = normalizeSelection(selection);
    const rangeStart = number(geometry.rangeStart, NaN);
    const rangeEnd = number(geometry.rangeEnd, NaN);
    const trackWidth = Math.max(1, number(geometry.trackWidth, 1));
    if (
      !normalized
      || !Number.isFinite(rangeStart)
      || !Number.isFinite(rangeEnd)
      || normalized.end < rangeStart
      || normalized.start > rangeEnd
    ) {
      layer.hidden = true;
      return;
    }
    const startX = timeToX(Math.max(normalized.start, rangeStart), trackWidth, rangeStart, rangeEnd);
    const endX = timeToX(Math.min(normalized.end, rangeEnd), trackWidth, rangeStart, rangeEnd);
    layer.hidden = false;
    layer.dataset.kind = normalized.kind || 'range';
    layer.style.left = `${number(geometry.trackLeft)}px`;
    layer.style.top = `${number(geometry.top)}px`;
    layer.style.width = `${trackWidth}px`;
    layer.style.height = `${Math.max(1, number(geometry.height, 1))}px`;
    layer.style.setProperty('--pto-selection-start', `${startX}px`);
    layer.style.setProperty('--pto-selection-width', `${Math.max(1, endX - startX)}px`);
  }

  function createSelectionSummary(root, options = {}) {
    if (!root) throw new Error('Timeline selection summary root is required');
    const summary = document.createElement('div');
    summary.className = 'pto-timeline-time-selection-summary';
    summary.hidden = true;
    summary.setAttribute('role', 'status');
    summary.setAttribute('aria-live', 'polite');
    summary.style.setProperty('--pto-timeline-label-width', `${number(options.labelWidth)}px`);
    root.appendChild(summary);
    return summary;
  }

  function updateSelectionSummary(summary, selection, language = 'en', geometry = null) {
    if (!summary) return;
    const normalized = normalizeSelection(selection);
    const content = formatSelectionSummary(normalized, language);
    summary.textContent = content;
    summary.hidden = !content;
    summary.dataset.kind = selection?.kind || 'range';
    if (!content || !geometry) return;
    const rangeStart = number(geometry.rangeStart, NaN);
    const rangeEnd = number(geometry.rangeEnd, NaN);
    const trackWidth = Math.max(1, number(geometry.trackWidth, 1));
    const viewportWidth = number(geometry.viewportWidth, NaN);
    if (![rangeStart, rangeEnd, viewportWidth].every(Number.isFinite)) return;

    const selectionCenter = (normalized.start + normalized.end) / 2;
    const contentX = number(geometry.trackLeft)
      + timeToX(selectionCenter, trackWidth, rangeStart, rangeEnd);
    const desiredLeft = contentX - number(geometry.scrollLeft);
    const padding = Math.max(0, number(geometry.viewportPadding, 8));
    const availableWidth = Math.max(1, viewportWidth - padding * 2);
    const halfWidth = Math.min(summary.offsetWidth / 2, availableWidth / 2);
    const resolvedLeft = clamp(
      desiredLeft,
      padding + halfWidth,
      viewportWidth - padding - halfWidth,
    );
    summary.style.left = `${resolvedLeft}px`;
    summary.dataset.alignment = Math.abs(resolvedLeft - desiredLeft) > 1
      ? 'viewport-clamped'
      : 'selection-center';
  }

  function bindTimelineInteraction(options = {}) {
    const target = options.target;
    if (!target) throw new Error('Timeline interaction target is required');
    const threshold = Math.max(0, number(options.dragThreshold, DEFAULT_DRAG_THRESHOLD));
    let drag = null;

    const geometry = () => {
      const next = options.getGeometry?.() || {};
      return {
        width: Math.max(1, number(next.width, target.getBoundingClientRect().width || 1)),
        start: number(next.start),
        end: number(next.end),
      };
    };
    const localX = (event) => {
      const rect = target.getBoundingClientRect();
      return clamp(event.clientX - rect.left, 0, Math.max(1, rect.width));
    };
    const selectionAt = (anchorX, currentX) => {
      const currentGeometry = geometry();
      return normalizeSelection({
        kind: 'range',
        start: xToTime(anchorX, currentGeometry.width, currentGeometry.start, currentGeometry.end),
        end: xToTime(currentX, currentGeometry.width, currentGeometry.start, currentGeometry.end),
      });
    };

    const onPointerDown = (event) => {
      if (event.button !== 0) return;
      target.focus?.({ preventScroll: true });
      drag = {
        pointerId: event.pointerId,
        anchorX: localX(event),
        currentX: localX(event),
        moved: false,
      };
      options.onInteractionStart?.(event);
      try {
        target.setPointerCapture(event.pointerId);
      } catch (_error) {
        // Pointer capture is an enhancement; document-local dragging still works without it.
      }
    };

    const onPointerMove = (event) => {
      if (!drag || drag.pointerId !== event.pointerId) {
        options.onHover?.(event, localX(event));
        return;
      }
      drag.currentX = localX(event);
      if (!drag.moved && Math.abs(drag.currentX - drag.anchorX) < threshold) return;
      if (!drag.moved) {
        drag.moved = true;
        target.classList.add('is-selecting-time');
        options.onDragStart?.(event);
      }
      options.onPreview?.(selectionAt(drag.anchorX, drag.currentX), event);
    };

    const finishPointer = (event, cancelled = false) => {
      if (!drag || drag.pointerId !== event.pointerId) return;
      const finished = drag;
      drag = null;
      target.classList.remove('is-selecting-time');
      try {
        if (target.hasPointerCapture(event.pointerId)) target.releasePointerCapture(event.pointerId);
      } catch (_error) {
        // The pointer may already have been released by the browser.
      }
      if (cancelled) {
        options.onCancel?.(event);
      } else if (finished.moved) {
        options.onCommit?.(selectionAt(finished.anchorX, localX(event)), event);
      } else {
        const hit = options.hitTest?.(localX(event), event);
        if (hit) options.onEventSelect?.(hit.event || hit, event);
        else options.onClear?.(event);
      }
      options.onInteractionEnd?.(event);
    };

    const onPointerUp = (event) => finishPointer(event, false);
    const onPointerCancel = (event) => finishPointer(event, true);
    const onPointerLeave = (event) => {
      if (!drag) options.onLeave?.(event);
    };
    const onWheel = (event) => {
      if (!event.ctrlKey && !event.metaKey) return;
      if (!event.deltaY) return;
      event.preventDefault();
      options.onZoom?.({
        direction: event.deltaY < 0 ? 1 : -1,
        x: localX(event),
        event,
      });
    };
    const onKeyDown = (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        options.onClear?.(event);
        return;
      }
      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
      event.preventDefault();
      options.onNavigate?.({
        direction: event.key === 'ArrowLeft' ? -1 : 1,
        accelerated: event.shiftKey,
        event,
      });
    };

    target.addEventListener('pointerdown', onPointerDown);
    target.addEventListener('pointermove', onPointerMove);
    target.addEventListener('pointerup', onPointerUp);
    target.addEventListener('pointercancel', onPointerCancel);
    target.addEventListener('pointerleave', onPointerLeave);
    target.addEventListener('wheel', onWheel, { passive: false });
    target.addEventListener('keydown', onKeyDown);

    return {
      destroy() {
        target.removeEventListener('pointerdown', onPointerDown);
        target.removeEventListener('pointermove', onPointerMove);
        target.removeEventListener('pointerup', onPointerUp);
        target.removeEventListener('pointercancel', onPointerCancel);
        target.removeEventListener('pointerleave', onPointerLeave);
        target.removeEventListener('wheel', onWheel);
        target.removeEventListener('keydown', onKeyDown);
      },
    };
  }

  global.PtoTimelineTimeSelectionPattern = {
    DEFAULT_DRAG_THRESHOLD,
    bindTimelineInteraction,
    createSelectionLayer,
    createSelectionSummary,
    formatAbsoluteUs,
    formatDuration,
    formatSelectionSummary,
    normalizeSelection,
    selectionFromEvent,
    timeToX,
    updateSelectionLayer,
    updateSelectionSummary,
    xToTime,
  };
})(window);
