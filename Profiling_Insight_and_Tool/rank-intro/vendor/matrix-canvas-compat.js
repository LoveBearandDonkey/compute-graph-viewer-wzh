(function attachRankMatrixCanvasCompat(global) {
  'use strict';

  var base = global.PtoMatrixCanvas;
  if (!base || typeof base.render !== 'function' || base.__rankIntroCompat) return;

  var BASE_CELL = 32;
  var originalRender = base.render;
  var AXIS_OPTION_KEYS = [
    'axisFontSize',
    'axisFontWeight',
    'axisColor',
    'columnAxisPosition',
    'columnAxisGap',
    'rowAxisGap',
  ];
  var TONE_TOKENS = {
    neutral: '--foreground-secondary',
    input: '--primary',
    output: '--success',
    compute: '--warning',
    reduction: '--warning',
    fusion: '--accent',
  };

  function finiteOr(value, fallback) {
    var number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  function positiveInteger(value, fallback) {
    var number = Number(value);
    return Number.isFinite(number) && number > 0 ? Math.floor(number) : fallback;
  }

  function sourceExtent(scene) {
    return {
      rows: positiveInteger(scene && scene.extent && scene.extent.rows, 1),
      columns: positiveInteger(scene && scene.extent && scene.extent.columns, 1),
    };
  }

  function displayExtent(scene, options) {
    var source = sourceExtent(scene);
    var requested = options && options.displayExtent;
    if (!requested || typeof requested !== 'object') return source;
    return {
      rows: positiveInteger(requested.rows, source.rows),
      columns: positiveInteger(requested.columns, source.columns),
    };
  }

  function needsCompat(scene, options) {
    if (options && options.displayExtent) return true;
    if (scene && Array.isArray(scene.cuts) && scene.cuts.length) return true;
    return AXIS_OPTION_KEYS.some(function (key) {
      return options && options[key] != null;
    });
  }

  function mapCell(cell, rowScale, columnScale) {
    var row = finiteOr(cell && cell.row, 0);
    var column = finiteOr(cell && cell.column, 0);
    var rowSpan = Math.max(1, finiteOr(cell && cell.rowSpan, 1));
    var columnSpan = Math.max(1, finiteOr(cell && cell.columnSpan, 1));
    var mappedRow = Math.round(row * rowScale);
    var mappedColumn = Math.round(column * columnScale);
    var mappedRowEnd = Math.round((row + rowSpan) * rowScale);
    var mappedColumnEnd = Math.round((column + columnSpan) * columnScale);
    return Object.assign({}, cell, {
      row: mappedRow,
      column: mappedColumn,
      rowSpan: Math.max(1, mappedRowEnd - mappedRow),
      columnSpan: Math.max(1, mappedColumnEnd - mappedColumn),
    });
  }

  function prepare(scene, options) {
    var source = sourceExtent(scene);
    var display = displayExtent(scene, options);
    var rowScale = display.rows / source.rows;
    var columnScale = display.columns / source.columns;
    var cells = Array.isArray(scene && scene.cells) ? scene.cells : [];
    var cuts = Array.isArray(scene && scene.cuts) ? scene.cuts : [];
    return {
      scene: {
        extent: display,
        axes: scene && scene.axes,
        cells: cells.map(function (cell) {
          return mapCell(cell, rowScale, columnScale);
        }),
      },
      cuts: cuts.map(function (cut) {
        var axis = cut && cut.axis === 'rows' ? 'rows' : 'columns';
        var scale = axis === 'rows' ? rowScale : columnScale;
        return Object.assign({}, cut, { axis: axis, at: finiteOr(cut && cut.at, 0) * scale });
      }).filter(function (cut) {
        return cut.at > 0 && cut.at < display[cut.axis];
      }),
    };
  }

  function cssToken(name, fallback, seen) {
    var visited = seen || {};
    if (!name || visited[name]) return fallback;
    visited[name] = true;
    var value = global.getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    if (!value) return fallback;
    var reference = /var\(\s*(--[\w-]+)/.exec(value);
    return reference ? cssToken(reference[1], fallback, visited) : value;
  }

  function canvasColor(value, fallbackToken, fallback) {
    var input = String(value || '').trim();
    if (input.indexOf('--') === 0) return cssToken(input, fallback);
    var reference = /var\(\s*(--[\w-]+)/.exec(input);
    if (reference) return cssToken(reference[1], fallback);
    return input || cssToken(fallbackToken, fallback);
  }

  function compatRender(canvas, inputScene, inputOptions) {
    var currentScene = inputScene || {};
    var currentOptions = Object.assign({}, inputOptions || {});
    if (!needsCompat(currentScene, currentOptions)) {
      return originalRender(canvas, currentScene, currentOptions);
    }

    var prepared = prepare(currentScene, currentOptions);
    var baseOptions = Object.assign({}, currentOptions, { showAxes: false });
    var inner = originalRender(canvas, prepared.scene, baseOptions);
    var host = canvas.parentElement || document.body;
    var overlay = document.createElement('canvas');
    overlay.className = 'rank-matrix-canvas-compat';
    overlay.setAttribute('aria-hidden', 'true');
    overlay.style.position = 'absolute';
    overlay.style.pointerEvents = 'none';
    overlay.style.zIndex = '1';
    host.appendChild(overlay);

    var observer = null;
    var frame = 0;
    var destroyed = false;

    function positionOverlay() {
      var canvasRect = canvas.getBoundingClientRect();
      var hostRect = host.getBoundingClientRect();
      overlay.style.left = (canvasRect.left - hostRect.left) + 'px';
      overlay.style.top = (canvasRect.top - hostRect.top) + 'px';
      overlay.style.width = canvasRect.width + 'px';
      overlay.style.height = canvasRect.height + 'px';
      return {
        width: Math.max(1, canvasRect.width || canvas.clientWidth || 1),
        height: Math.max(1, canvasRect.height || canvas.clientHeight || 1),
      };
    }

    function draw() {
      if (destroyed) return;
      var size = positionOverlay();
      var dpr = Math.max(1, global.devicePixelRatio || 1);
      var backingWidth = Math.max(1, Math.floor(size.width * dpr));
      var backingHeight = Math.max(1, Math.floor(size.height * dpr));
      if (overlay.width !== backingWidth) overlay.width = backingWidth;
      if (overlay.height !== backingHeight) overlay.height = backingHeight;
      var ctx = overlay.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, size.width, size.height);

      var view = inner.getViewState();
      var extent = prepared.scene.extent;
      var bounds = {
        x: view.offsetX,
        y: view.offsetY,
        width: extent.columns * BASE_CELL * view.scale,
        height: extent.rows * BASE_CELL * view.scale,
      };

      if (prepared.cuts.length) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(bounds.x, bounds.y, bounds.width, bounds.height);
        ctx.clip();
        ctx.lineCap = 'square';
        ctx.lineWidth = 2.4;
        prepared.cuts.forEach(function (cut) {
          var token = TONE_TOKENS[cut.tone] || TONE_TOKENS.input;
          ctx.strokeStyle = canvasColor(currentOptions.cutColor, token, '#7c6cff');
          ctx.beginPath();
          if (cut.axis === 'columns') {
            var x = bounds.x + cut.at * BASE_CELL * view.scale;
            ctx.moveTo(x, bounds.y);
            ctx.lineTo(x, bounds.y + bounds.height);
          } else {
            var y = bounds.y + cut.at * BASE_CELL * view.scale;
            ctx.moveTo(bounds.x, y);
            ctx.lineTo(bounds.x + bounds.width, y);
          }
          ctx.stroke();
        });
        ctx.restore();
      }

      if (currentOptions.showAxes === false) return;
      var axes = currentScene && currentScene.axes || {};
      var fontSize = Math.max(1, finiteOr(currentOptions.axisFontSize, 12));
      var fontWeight = Math.max(100, finiteOr(currentOptions.axisFontWeight, 600));
      var mono = cssToken('--font-mono', 'ui-monospace, monospace');
      var axisColor = canvasColor(currentOptions.axisColor, '--foreground-secondary', '#7b8190');
      var top = currentOptions.columnAxisPosition === 'top';
      var columnGap = Math.max(0, finiteOr(currentOptions.columnAxisGap, top ? 10 : 26));
      var rowGap = Math.max(0, finiteOr(currentOptions.rowAxisGap, 28));
      ctx.fillStyle = axisColor;
      ctx.font = fontWeight + ' ' + fontSize + 'px ' + mono;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(axes.columns || 'column'), bounds.x + bounds.width / 2,
        top ? bounds.y - columnGap : bounds.y + bounds.height + columnGap);
      ctx.save();
      ctx.translate(bounds.x - rowGap, bounds.y + bounds.height / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(String(axes.rows || 'row'), 0, 0);
      ctx.restore();
    }

    function scheduleDraw() {
      if (destroyed || frame) return;
      frame = global.requestAnimationFrame(function () {
        frame = global.requestAnimationFrame(function () {
          frame = 0;
          draw();
        });
      });
    }

    if (typeof ResizeObserver === 'function') {
      observer = new ResizeObserver(scheduleDraw);
      observer.observe(canvas);
    }

    var controller = {
      update: function update(nextScene, nextOptions) {
        currentScene = nextScene || {};
        if (nextOptions && typeof nextOptions === 'object') {
          currentOptions = Object.assign({}, currentOptions, nextOptions);
        }
        prepared = prepare(currentScene, currentOptions);
        inner.update(prepared.scene, Object.assign({}, currentOptions, { showAxes: false }));
        draw();
        return controller;
      },
      resize: function resize() {
        inner.resize();
        draw();
        return controller;
      },
      fit: function fit() {
        inner.fit();
        draw();
        return controller;
      },
      resetView: function resetView() {
        return controller.fit();
      },
      setZoom: function setZoom(nextZoom, anchor) {
        inner.setZoom(nextZoom, anchor);
        draw();
        return controller;
      },
      getViewState: function getViewState() {
        return inner.getViewState();
      },
      destroy: function destroy() {
        if (destroyed) return;
        destroyed = true;
        observer && observer.disconnect();
        observer = null;
        if (frame) global.cancelAnimationFrame(frame);
        frame = 0;
        inner.destroy();
        overlay.remove();
      },
    };

    draw();
    return controller;
  }

  global.PtoMatrixCanvas = Object.freeze(Object.assign({}, base, {
    render: compatRender,
    __rankIntroCompat: true,
  }));
})(window);
