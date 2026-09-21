// ─────────────────────────────────────────────────────────────────────────────
// L0 Core-Group 内部架构视图 —— 复用 pto-design-system 的 memory-architecture
// pattern（vendored 于 src/vendor/memory-architecture，依赖 aic/aiv-core-object）。
// 所有 2D / 工作台 / 运行状态视图的 L0 层级钻取统一渲染本组件：
//   Global Memory / L2 Cache 轨道 → AIV 1 · AIC · AIV 2 核组对象
//   + UB / L1 / L0A / L0B / L0C / BT / FP 缓冲 + MTE1/MTE2/MTE3 / FixPipe 路由。
// 运行状态由 setPathFocus（路由高亮）+ setBufferBlocks（缓冲块占用）驱动，
// 对应 RunKind：load=MTE2 搬入 · compute=CUBE/SIMD 执行 · comm=UB↔GM 集合通信取数 ·
// mem=L2 轨道访存 · store=MTE3 写回。
// ─────────────────────────────────────────────────────────────────────────────
import { useEffect, useRef } from 'react';
import '../vendor/pto-tokens/foundation.css';
import '../vendor/pto-tokens/semantic.css';
import '../vendor/pto-tokens/components.css';
import '../vendor/aic-core-object/pattern.css';
import '../vendor/aiv-core-object/pattern.css';
import '../vendor/memory-architecture/pattern.css';
import '../vendor/aiv-core-object/pattern.js';
import '../vendor/aic-core-object/pattern.js';
import '../vendor/memory-architecture/pattern.js';

const PRESET = 'ascend950b';

// run-phase kind → pattern 路由 / 高亮节点（route id 来自 preset ascend950b）
const PHASE_FOCUS: Record<string, { routes: string[]; selectors: string[] }> = {
  load: {
    routes: ['gm-to-aiv1-ub', 'gm-to-aiv2-ub', 'gm-to-aic-l0a', 'gm-to-aic-l0b'],
    selectors: [],
  },
  compute: {
    routes: ['aic-to-aiv1', 'aiv2-to-aic'],
    selectors: [
      '#mem950-aic [data-aic-node="cube:CUBE"]',
      '#mem950-aiv1 [data-aiv-node="exec:SIMD"]',
      '#mem950-aiv2 [data-aiv-node="exec:SIMD"]',
    ],
  },
  comm: {
    routes: ['aiv1-ub-to-gm', 'aiv2-ub-to-gm'],
    selectors: ['#mem950-aiv1 [data-aiv-node="buffer:UB"]', '#mem950-aiv2 [data-aiv-node="buffer:UB"]'],
  },
  mem: {
    routes: ['l2-to-aiv1', 'l2-to-aiv2', 'l2-to-aic', 'l2-to-aic-dcache'],
    selectors: [],
  },
  store: {
    routes: ['aiv1-to-l2', 'aiv2-to-l2'],
    selectors: ['#mem950-aic [data-aic-node="buffer:L0C"]'],
  },
};

// 运行相位对应的缓冲块占用（cellRange 端点随 load 伸缩，示意占用而非实测）
function phaseBlocks(kind: string, load: number) {
  const n = Math.max(2, Math.round(4 + Math.max(0, Math.min(1, load)) * 24));
  switch (kind) {
    case 'load':
      return [
        { core: 'mem950-aiv1', buffer: 'UB', label: 'xLocal', state: 'enqueued', tone: 'input', cellRange: [0, n] as [number, number], sourceTile: 'x[block,progress,:]' },
        { core: 'mem950-aic', buffer: 'L0A', label: 'A tile', state: 'enqueued', tone: 'input', cellRange: [0, Math.min(n, 15)] as [number, number], sourceTile: 'A[m,k]' },
        { core: 'mem950-aic', buffer: 'L0B', label: 'B tile', state: 'enqueued', tone: 'input', cellRange: [0, Math.min(n, 15)] as [number, number], sourceTile: 'B[k,n]' },
      ];
    case 'compute':
      return [
        { core: 'mem950-aic', buffer: 'L0C', label: 'C partial', state: 'accumulating', tone: 'accumulator', cellRange: [8, 8 + n] as [number, number], sourceTile: 'C[m,n]' },
        { core: 'mem950-aiv1', buffer: 'UB', label: 'act', state: 'accumulating', tone: 'accumulator', cellRange: [0, Math.min(n, 15)] as [number, number], sourceTile: 'y[block,:]' },
      ];
    case 'comm':
      return [
        { core: 'mem950-aiv1', buffer: 'UB', label: 'AllReduce out', state: 'committed', tone: 'output', cellRange: [0, n] as [number, number], sourceTile: 'grad[block,:]' },
        { core: 'mem950-aiv2', buffer: 'UB', label: 'AllReduce in', state: 'enqueued', tone: 'input', cellRange: [0, n] as [number, number], sourceTile: 'grad[peer,:]' },
      ];
    case 'mem':
      return [
        { core: 'mem950-aiv1', buffer: 'UB', label: 'KV read', state: 'enqueued', tone: 'input', cellRange: [0, n] as [number, number], sourceTile: 'KV[seq,:]' },
        { core: 'mem950-aic', buffer: 'L1', label: 'weights', state: 'enqueued', tone: 'input', cellRange: [0, Math.min(n * 2, 60)] as [number, number], sourceTile: 'W[layer,:]' },
      ];
    case 'store':
      return [
        { core: 'mem950-aic', buffer: 'L0C', label: 'C commit', state: 'committed', tone: 'output', cellRange: [0, n] as [number, number], sourceTile: 'C[m,n]' },
        { core: 'mem950-aiv2', buffer: 'UB', label: 'writeback', state: 'committed', tone: 'output', cellRange: [0, Math.min(n, 15)] as [number, number], sourceTile: 'out[block,:]' },
      ];
    default:
      return [];
  }
}

export interface CoreGroupPatternProps {
  /** 当前运行相位（RUN_SCHED 的 RunKind）；不传或 'idle' 则不做路由聚焦 */
  phaseKind?: string;
  /** 0..1 负载，驱动缓冲块占用规模 */
  load?: number;
  /** 初始缩放（嵌入面板用 0.4–0.6） */
  zoom?: number;
  /** 是否显示 UB bank 明细行 */
  detail?: boolean;
  /** 折叠 AIV 2（窄面板时省空间） */
  foldAiv?: boolean;
  /** 水平对齐：'center'（默认，居中）或 'left'（贴左，紧接上方漏斗的左栏 → 视觉连成一体） */
  align?: 'center' | 'left';
  height?: number | string;
  /** 是否允许滚轮缩放 / 拖拽平移（默认 true）。嵌入固定漏斗（如联动控制台左栏）时设 false，
   *  这样滚动不会意外把 L0 单独放大，L0 与上方漏斗保持同一固定比例、读作一体。 */
  interactive?: boolean;
  style?: React.CSSProperties;
}

/** L0 Core-Group（AIV·向量 / AIC·Cube / AICPU）内部架构 —— memory-architecture pattern 挂载器 */
export function CoreGroupPattern({ phaseKind, load = 0.5, zoom = 0.5, detail = false, foldAiv = false, align = 'center', height = '100%', interactive = true, style }: CoreGroupPatternProps) {
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const apiRef = useRef<{ overlay?: any; hover?: any; zoomCtl?: any } | null>(null);

  // 一次性挂载：渲染 + 路由 overlay + hover + path-focus + 缩放控制器
  useEffect(() => {
    const helper = window.PtoMemoryArchitecturePattern;
    const stage = stageRef.current, viewport = viewportRef.current;
    if (!helper || !stage || !viewport) return;
    const sizer = viewport.querySelector('[data-pto-mem-arch-sizer]') as HTMLElement | null;
    const canvas = viewport.querySelector('[data-pto-mem-arch-canvas]') as HTMLElement | null;

    helper.renderArchitecture(stage, PRESET);
    helper.setDetailVisibility?.(stage, detail);
    helper.setAivFolded?.(stage, foldAiv);
    const overlay = helper.createRouteOverlay(stage, PRESET);
    overlay?.render();
    const hover = helper.attachHoverInteractions?.(stage, PRESET);
    helper.attachPathFocusInteractions?.(stage, PRESET);
    const zoomCtl = helper.createZoomController?.({
      viewport, sizer, canvas,
      defaultZoom: zoom, min: 0.3, max: 1.4, step: 0.1,
      pan: interactive, wheelZoom: interactive, centerOnReset: true,
      centerTarget: '.pto-mem950__rails, .pto-mem950__engine-stack, .pto-mem950__stack',
      onZoom: ({ zoom: z }: { zoom: number }) => { hover?.setViewportScale?.(z); overlay?.render(); },
    });
    zoomCtl?.render?.();
    // center() (vendored) equalises L/R margins → a big empty gap on the left between the funnel's
    // left gutter and this diagram. align='left' shifts X so the rails butt against the gutter, so the
    // figure reads as one continuous piece with the L1 Die section directly above it.
    const settle = () => {
      zoomCtl?.center?.();
      if (align === 'left' && zoomCtl && viewport) {
        const targets = viewport.querySelectorAll('.pto-mem950__rails, .pto-mem950__engine-stack, .pto-mem950__stack');
        if (targets.length) {
          const vr = viewport.getBoundingClientRect();
          let left = Infinity;
          targets.forEach((t) => { const b = (t as HTMLElement).getBoundingClientRect(); if (b.width > 0) left = Math.min(left, b.left); });
          if (Number.isFinite(left)) {
            const cur = zoomCtl.getPan?.() ?? { x: 0 };
            zoomCtl.setPan?.(cur.x + 8 - (left - vr.left), (zoomCtl.getPan?.() ?? { y: 0 }).y);
          }
        }
      }
    };
    requestAnimationFrame(() => requestAnimationFrame(settle));
    apiRef.current = { overlay, hover, zoomCtl };
    return () => {
      zoomCtl?.destroy?.();
      hover?.destroy?.();
      overlay?.destroy?.();
      helper.clearBufferBlocks?.(stage);
      apiRef.current = null;
      stage.innerHTML = '';
    };
    // detail/foldAiv 变化走下面的轻量 effect，不重建整个 stage
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // detail / fold 切换
  useEffect(() => {
    const helper = window.PtoMemoryArchitecturePattern, stage = stageRef.current;
    if (!helper || !stage) return;
    helper.setDetailVisibility?.(stage, detail);
    helper.setAivFolded?.(stage, foldAiv);
    apiRef.current?.overlay?.render?.();
  }, [detail, foldAiv]);

  // 运行状态驱动：相位 → 路由聚焦 + 缓冲块占用
  useEffect(() => {
    const helper = window.PtoMemoryArchitecturePattern, stage = stageRef.current;
    if (!helper || !stage) return;
    const focus = phaseKind ? PHASE_FOCUS[phaseKind] : undefined;
    if (focus) {
      helper.setPathFocus?.(stage, PRESET, { selectors: focus.selectors, routes: focus.routes });
      helper.setBufferBlocks?.(stage, phaseBlocks(phaseKind!, load));
    } else {
      helper.clearPathFocus?.(stage);
      helper.clearBufferBlocks?.(stage);
    }
    apiRef.current?.overlay?.render?.();
  }, [phaseKind, load]);

  return (
    <div
      ref={viewportRef}
      className="pto-memory-architecture-viewport"
      data-pto-mem-arch-viewport
      data-default-zoom={zoom}
      aria-label="L0 Core-Group 内部架构（memory-architecture pattern）"
      style={{ width: '100%', height, overflow: 'hidden', position: 'relative', ...style }}
    >
      <div className="pto-memory-architecture-sizer" data-pto-mem-arch-sizer>
        <div className="pto-memory-architecture-canvas" data-pto-mem-arch-canvas>
          <div ref={stageRef} />
        </div>
      </div>
    </div>
  );
}
