/*
  视图 A-2 —— 内存布局的「硬件架构」布局
  ------------------------------------------------------------------
  布局 1（view-layout.js）把地址空间摊平成条带，回答「谁占了哪一段」；
  布局 2 把同一份读数贴回硬件本身，回答「这块存储长在哪、归谁管、还剩多少」：
    · 每个存储层级在架构图上的卡片下方挂一组读数 —— 物理容量、对齐要求、
      静态预留、利用率、峰值持有，超限时利用率一栏直接写出超了多少；
    · 卡片里的 cell 网格按当前时刻的占用着色：实色 = 此刻持有，
      灰色 = 预留着但空闲，警告色 = 该层级已经超限；
    · 悬停任意硬件节点给出同一份数据的完整读数，点击则把焦点层级切过去。

  设计系统约束（patterns/memory-architecture/pattern.json）：
    · 架构图本体、路由、AIC/AIV 内部结构全部由 memory-architecture 的
      renderArchitecture / createRouteOverlay / attachHoverInteractions /
      createZoomController 产出，本文件不复制也不重排它生成的 DOM；
    · cell 着色只走 setBufferBlocks 这一个官方入口，不用局部 CSS 改颜色；
    · 读数行通过「派生 preset 的 details」交给 pattern 自己渲染 —— 这些行
      随当前 run 变化，无法固化进 vendor 子模块，因此以 preset 对象形式传入
      （renderArchitecture 的 presetOrKey 本就接受 preset 对象）。
*/
(function registerMemVizArchView(global) {
  'use strict';

  const F = global.MemVizFormat;

  const ARCH_PRESET_BY_CHIP = {
    'ascend-910b': 'ascend910b',
    'ascend-950b': 'ascend950b',
  };

  /**
   * 存储层级 → 架构图上的落点。
   *   detail  挂读数的卡片选择器（pattern 的 details.selector）
   *   cores   可着色的核心 slot id；buffer 是该核心内的 buffer key
   */
  const TARGETS = {
    UB: {
      detail: '[data-aiv-node="buffer:UB"]',
      cores: ['mem950-aiv1', 'mem950-aiv2'], buffer: 'UB',
    },
    L1: { detail: '#mem950-aic [data-aic-node="buffer:L1"]', cores: ['mem950-aic'], buffer: 'L1' },
    L0A: { detail: '#mem950-aic [data-aic-node="buffer:L0A"]', cores: ['mem950-aic'], buffer: 'L0A' },
    L0B: { detail: '#mem950-aic [data-aic-node="buffer:L0B"]', cores: ['mem950-aic'], buffer: 'L0B' },
    L0C: { detail: '#mem950-aic [data-aic-node="buffer:L0C"]', cores: ['mem950-aic'], buffer: 'L0C' },
    GM: { detail: '[data-mem950-node="rail:GM"]' },
    VRF: { detail: '[data-aiv-node="exec:SIMD"]' },
    SRF: { detail: '[data-aiv-node="exec:SIMT"]' },
  };

  /** 节点 key → 存储层级；悬停与点击都靠它把硬件位置翻译回数据。 */
  const REGION_BY_NODE = {
    'buffer:UB': 'UB',
    'buffer:L1': 'L1',
    'buffer:L0A': 'L0A',
    'buffer:L0B': 'L0B',
    'buffer:L0C': 'L0C',
    'rail:GM': 'GM',
    'exec:SIMD': 'VRF',
    'vector:Vector': 'VRF',
    'exec:SIMT': 'SRF',
  };

  function nodeKeyOf(target) {
    return target?.dataset?.mem950Node || target?.dataset?.aicNode || target?.dataset?.aivNode || '';
  }

  function capacityText(region) {
    return region.isRegister
      ? `${region.capacityRegs} regs × ${region.regBytes}B`
      : F.bytes(region.capacity);
  }

  function reservedText(region) {
    return region.isRegister
      ? `${region.reservedRegs} regs · ${F.bytes(region.reserved)}`
      : F.bytes(region.reserved);
  }

  /** 一个存储层级的读数行 —— 物理大小 / 对齐 / 预留 / 利用率 / 峰值。 */
  function rowsForRegion(region, run) {
    const over = region.reserved - region.capacity;
    const rows = [
      ['物理容量', capacityText(region)],
      ['对齐', region.isRegister ? `${region.regBytes}B / reg` : `${region.align}B`],
      ['预留', reservedText(region)],
      ['利用率', over > 0
        ? `${F.pct(region.reservedRatio, 0)} · 超 ${F.bytes(over)}`
        : F.pct(region.reservedRatio, 0)],
      ['峰值持有', `${F.bytes(region.peakLive)} @ ${F.tick(region.peakTick)}`],
    ];
    if (region.banks) rows.splice(2, 0, ['bank', `${region.banks} 个`]);
    if (region.padding > 0) rows.push(['对齐浪费', F.bytes(region.padding)]);

    const plan = run.registers;
    if (plan && region.id === plan.vectorRegionId) {
      rows.push(['展开度', `${plan.unroll} 组`]);
      if (plan.spillRegs > 0) {
        rows.push(['溢出', `${plan.spillRegs} regs → ${plan.spillRegion}`]);
      }
    }
    if (plan && region.id === plan.simtRegionId) {
      rows.push(['每线程寄存器', `${plan.regsPerThread}`]);
      rows.push(['并发 warp', `${plan.activeWarps} / ${plan.warpsMax}`]);
    }
    return rows;
  }

  /**
   * 派生 preset：保留原 preset 的结构、路由与非数据类补充（指令序列、bank 网格），
   * 把随 run 变化的读数行换成当前 metrics 算出来的那一组。
   */
  function derivePreset(base, chip, run, metrics) {
    const keepDetails = (base.details || [])
      .filter((detail) => detail.instructionItems || detail.bankGrid)
      .map((detail) => ({
        selector: detail.selector,
        bankGrid: detail.bankGrid,
        instructionItems: detail.instructionItems,
      }));

    const dataDetails = metrics.regions
      .filter((region) => TARGETS[region.id])
      .map((region) => ({
        selector: TARGETS[region.id].detail,
        rows: rowsForRegion(region, run),
      }));

    return { ...base, details: [...keepDetails, ...dataDetails] };
  }

  function create(container, options = {}) {
    const helper = global.PtoMemoryArchitecturePattern;

    const viewport = document.createElement('div');
    viewport.className = 'pto-memory-architecture-viewport mv-arch-viewport';
    viewport.setAttribute('aria-label', '硬件架构内存布局');
    const sizer = document.createElement('div');
    sizer.className = 'pto-memory-architecture-sizer';
    const canvas = document.createElement('div');
    canvas.className = 'pto-memory-architecture-canvas';
    const stage = document.createElement('div');
    sizer.appendChild(canvas);
    canvas.appendChild(stage);
    viewport.appendChild(sizer);
    container.appendChild(viewport);

    if (!helper) {
      stage.innerHTML = '<p class="mv-empty">未加载 memory-architecture pattern，硬件架构布局不可用。</p>';
      return { update() {}, redraw() {}, destroy() { viewport.remove(); } };
    }

    let state = null;
    let preset = null;
    let mountKey = '';
    let overlay = null;
    let hover = null;
    let activation = null;
    let zoom = null;
    let userZoomed = false;
    let fitting = false;

    function regionOf(id) {
      return state?.metrics?.regionById?.[id] || null;
    }

    function teardownMount() {
      overlay?.destroy?.();
      hover?.destroy?.();
      activation?.destroy?.();
      overlay = null;
      hover = null;
      activation = null;
    }

    function tipFor(key) {
      const region = regionOf(REGION_BY_NODE[key]);
      if (!region) return null;
      const over = region.reserved - region.capacity;
      const liveNow = region.series[Math.min(state.tick, region.series.length - 1)];
      const lines = [
        `物理容量 ${capacityText(region)} · 对齐 ${region.isRegister ? `${region.regBytes}B/reg` : `${region.align}B`}`,
        `静态预留 ${reservedText(region)}（${F.pct(region.reservedRatio, 1)}${over > 0 ? `，超出 ${F.bytes(over)}` : ''}）`,
        `此刻持有 ${F.bytes(liveNow)} · 峰值 ${F.bytes(region.peakLive)} @ ${F.tick(region.peakTick)}`,
        region.note || '',
      ];
      return {
        title: `${region.id} · ${region.label}`,
        body: lines.filter(Boolean).join('\n'),
      };
    }

    function paintBlocks() {
      if (!state) return;
      const root = stage.querySelector('.pto-mem950');
      if (!root) return;
      const blocks = [];

      state.metrics.regions.forEach((region) => {
        const target = TARGETS[region.id];
        if (!target?.cores) return;
        const over = region.reserved > region.capacity;
        const liveNow = region.series[Math.min(state.tick, region.series.length - 1)];

        target.cores.forEach((coreId) => {
          const slot = root.querySelector(`[id="${coreId}"]`);
          if (!slot) return;
          const cells = slot.querySelectorAll('.pto-aiv-core__cell, .pto-aic-core__cell').length;
          if (!cells) return;

          const reservedCells = Math.min(cells, Math.max(1,
            Math.round(Math.min(1, region.reservedRatio) * cells)));
          const liveCells = Math.min(reservedCells,
            Math.round(Math.min(1, liveNow / region.capacity) * cells));

          if (over) {
            // 超限：整块打成警告色，读数行里写明超了多少
            blocks.push({
              core: coreId, buffer: target.buffer, state: 'accumulating', tone: 'accumulator',
              label: `${region.id} 超限 ${F.bytes(region.reserved - region.capacity)}`,
              cellRange: [0, cells - 1],
            });
            return;
          }
          if (liveCells > 0) {
            blocks.push({
              core: coreId, buffer: target.buffer, state: 'loaded', tone: 'input',
              label: `${region.id} 此刻持有 ${F.bytes(liveNow)}`,
              cellRange: [0, liveCells - 1],
            });
          }
          if (reservedCells > liveCells) {
            blocks.push({
              core: coreId, buffer: target.buffer, state: 'avoided', tone: 'input',
              label: `${region.id} 预留未用 ${F.bytes(Math.max(0, region.reserved - liveNow))}`,
              cellRange: [liveCells, reservedCells - 1],
            });
          }
        });
      });

      helper.setBufferBlocks?.(stage, blocks);
    }

    function mount() {
      teardownMount();
      const baseKey = ARCH_PRESET_BY_CHIP[state.chip.id] || 'ascend950b';
      const base = helper.presets?.[baseKey] || helper.resolvePreset?.(baseKey);
      if (!base) return;
      preset = derivePreset(base, state.chip, state.run, state.metrics);

      helper.renderArchitecture(stage, preset);
      overlay = helper.createRouteOverlay(stage, preset);
      overlay?.render();
      hover = helper.attachHoverInteractions?.(stage, preset, { getTip: (key) => tipFor(key) });
      activation = helper.attachNodeActivation?.(stage, preset, {
        label: (target) => {
          const region = regionOf(REGION_BY_NODE[nodeKeyOf(target)]);
          return region ? `聚焦 ${region.id} · ${region.label}` : '查看该硬件节点';
        },
        onActivate: (target) => {
          const regionId = REGION_BY_NODE[nodeKeyOf(target)];
          if (regionId && regionOf(regionId)) options.onFocusRegion?.(regionId);
        },
      });

      if (!zoom) {
        zoom = helper.createZoomController?.({
          viewport, sizer, canvas,
          defaultZoom: 0.6, min: 0.35, max: 1.4, step: 0.1,
          pan: true, wheelZoom: true, centerOnReset: true,
          centerTarget: '.pto-mem950__rails, .pto-mem950__engine-stack, .pto-mem950__stack',
          onZoom: ({ zoom: next }) => {
            if (!fitting) userZoomed = true;
            hover?.setViewportScale?.(next);
            overlay?.render();
          },
        });
      }
      zoom?.render?.();
      window.requestAnimationFrame(() => {
        overlay?.render?.();
        fitWidth();
        zoom?.center?.();
      });
      // 分栏与首帧布局要几帧才稳定，宽度量早了会算出偏小的缩放
      window.setTimeout(fitWidth, 120);
    }

    // 首屏尽量把整张图的宽度放进来：0.6 是 pattern 的共享默认值，
    // 但工作台的舞台常常窄于图本身，这时缩到刚好装下整幅宽度更有用。
    // 用户一旦自己缩放过就不再自动改动。
    function fitWidth() {
      const graph = stage.querySelector('.pto-mem950');
      if (!zoom || !graph || userZoomed) return;
      const natural = graph.offsetWidth;
      const available = viewport.clientWidth - 8;
      if (!natural || available <= 0) return;
      fitting = true;
      zoom.setZoom(Math.max(0.35, Math.min(0.6, available / natural)));
      fitting = false;
    }


    return {
      update(next) {
        state = next;
        const key = `${state.chip.id}::${state.run.id}`;
        if (key !== mountKey) {
          mountKey = key;
          mount();
        }
        paintBlocks();
      },
      redraw() {
        fitWidth();
        overlay?.render?.();
        paintBlocks();
      },
      destroy() {
        teardownMount();
        zoom?.destroy?.();
        zoom = null;
        viewport.remove();
      },
    };
  }

  global.MemVizArchView = { create, TARGETS, REGION_BY_NODE };
})(window);
