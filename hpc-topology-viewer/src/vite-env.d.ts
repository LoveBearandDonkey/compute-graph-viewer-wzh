/// <reference types="vite/client" />

// 构建期注入的唯一 id（见 vite.config.ts define），用于给 iframe 静态资源做缓存刷新。
declare const __BUILD_ID__: string;

declare module '*workbench-shell/pattern.js';
declare module '*aic-core-object/pattern.js';
declare module '*aiv-core-object/pattern.js';
declare module '*memory-architecture/pattern.js';
declare module '*model-graphviz/pattern.js';
declare module '*swimlane-task/pattern.js';

type PtoWorkbenchSplitDirection = 'horizontal' | 'vertical';

interface PtoWorkbenchResizablePanes {
  destroy(): void;
  getSizes(): number[];
  setSizes(nextSizes: number[]): void;
  refresh(): void;
}

interface PtoWorkbenchResizeMeta {
  phase: 'start' | 'drag' | 'end' | 'api' | 'refresh';
  event?: Event;
  direction: PtoWorkbenchSplitDirection;
}

interface PtoWorkbenchResizablePaneOptions {
  root?: Element | string;
  panes?: Array<Element | string>;
  direction?: PtoWorkbenchSplitDirection;
  sizes?: number[];
  defaultSize?: number[];
  minSize?: number | number[];
  gutterSize?: number;
  storageKey?: string;
  gutterLabel?: string;
  keyboard?: boolean;
  keyboardStep?: number;
  onResize?: (sizes: number[], meta: PtoWorkbenchResizeMeta) => void;
}

interface Window {
  PtoWorkbenchShell?: {
    initResizablePanes(options?: PtoWorkbenchResizablePaneOptions): PtoWorkbenchResizablePanes;
  };
  /** pto-design-system patterns/aic-core-object（vendored，全局注册） */
  PtoAicCorePattern?: Record<string, any>;
  /** pto-design-system patterns/aiv-core-object（vendored，全局注册） */
  PtoAivCorePattern?: Record<string, any>;
  /** pto-design-system patterns/memory-architecture（vendored，全局注册）
   *  L0 Core-Group 芯片内架构 pattern：GM/L2 轨道 + AIV1/AIC/AIV2 + MTE/FixPipe 路由。 */
  PtoMemoryArchitecturePattern?: {
    renderArchitecture(container: Element, presetOrKey?: any): void;
    createRouteOverlay(container: Element, presetOrKey?: any): { render(): void; update?(): void; schedule?(): void; destroy(): void } | null;
    attachHoverInteractions?(container: Element, presetOrKey?: any, options?: any): { setViewportScale?(z: number): void; destroy?(): void } | null;
    attachPathFocusInteractions?(container: Element, presetOrKey?: any, options?: any): { destroy?(): void } | null;
    setDetailVisibility?(container: Element, visible: boolean): void;
    setAivFolded?(container: Element, folded: boolean): void;
    setPathFocus?(container: Element, presetOrKey: any, payload: { selectors?: string[]; routes?: string[]; errorSelectors?: string[] }): void;
    clearPathFocus?(container: Element): void;
    setBufferBlocks?(root: Element, blocks: Array<{ core: string; buffer: string; label?: string; state?: string; tone?: string; cellRange?: [number, number]; sourceTile?: string }>): void;
    clearBufferBlocks?(root: Element): void;
    createZoomController?(options: any): { render?(): void; getZoom?(): number; getPan?(): { x: number; y: number }; setZoom?(z: number): void; setPan?(x: number, y: number): void; center?(): void; zoomAtPoint?(z: number, x: number, y: number): void; reset?(): void; destroy?(): void } | null;
  };
  /** pto-design-system patterns/swimlane-task（vendored，全局注册）
   *  泳道任务条：canvas 渲染 IN/compute/OUT 三段任务条 + 悬停提示，适用于泳道/执行时序/甘特图。 */
  PtoSwimlaneTaskPattern?: {
    defaults: Record<string, unknown>;
    drawTaskBar(ctx: CanvasRenderingContext2D, options: {
      task?: { label?: string; displayName?: string; opName?: string; laneKind?: string; laneId?: string; totalCycle?: number; clcCycle?: number; gap?: number; gapRatio?: number; status?: string; dominantCounter?: string; wrapId?: string; inputRawMagic?: unknown[]; outputRawMagic?: unknown[]; [k: string]: unknown };
      x?: number; y?: number; width?: number; height?: number;
      baseColor?: string; radius?: number; fontFamily?: string;
      isSelected?: boolean; isRelated?: boolean; isEmphasized?: boolean;
    }): { displayColor: string; borderColor: string; segmentWidths: { inW: number; computeW: number; outW: number } };
    createTaskColormap(options?: Record<string, unknown>): { colorForTask(task: unknown, mode?: string): string; colorForLaneKind(kind: string): string; legendForKeys(keys: unknown[], mode?: string): Array<{ key: string; label: string; color: string }> };
    formatTaskTooltip(task: Record<string, unknown>, options?: Record<string, unknown>): string;
    createTooltip(options?: Record<string, unknown>): HTMLElement;
    showTooltip(tooltip: HTMLElement, task: Record<string, unknown>, event: MouseEvent | null, options?: Record<string, unknown>): void;
    moveTooltip(tooltip: HTMLElement, event: MouseEvent, options?: Record<string, unknown>): void;
    hideTooltip(tooltip: HTMLElement): void;
    initHoverTooltip(options: { root: Element; targets?: Element[] | string; tooltip?: HTMLElement; bounds?: Element; appendTo?: Element; getTask?: (target: Element, event?: Event) => Record<string, unknown> | null; durationUnit?: string; [k: string]: unknown }): { tooltip: HTMLElement; destroy(): void } | null;
    alphaHexColor(color: string, alpha: number): string;
    lightenHexColor(hex: string, amount: number): string;
    mixHexColors(base: string, target: string, ratio: number): string;
    stableHash(input: unknown): number;
  };
  /** pto-design-system patterns/model-graphviz（vendored，全局注册）
   *  整网图：Model → DecoderLayer → Attention/MoE → QKV/Expert → Operator 分层计算图渲染引擎。 */
  PtoModelGraphvizPattern?: {
    render(container: Element | string, graph: unknown, options?: Record<string, unknown>): (SVGSVGElement & { ptoModelGraphvizController?: ModelGraphvizController }) | null;
    renderController(container: Element | string, graph: unknown, options?: Record<string, unknown>): ModelGraphvizController | null;
    modelArchitectureColormap?(graph: unknown, options?: Record<string, unknown>): unknown;
  };
}

interface ModelGraphvizController {
  svg: SVGSVGElement;
  graph: unknown;
  selectNode(nodeId: string | null, options?: { source?: string; relatedNodeIds?: string[] | null }): void;
  clearSelection(): void;
  setFocus(focus: string | { nodeId?: string; id?: string; relatedNodeIds?: string[]; source?: string } | null): void;
  fit(): void;
  setTransform(transform: unknown): void;
  getTransform(): unknown;
  destroy(): void;
}
