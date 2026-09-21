// ─────────────────────────────────────────────────────────────────────────────
// 架构图元库 —— 对 pto-design-system memory-architecture pattern 的“理解性重画”。
//
// 图源语义（patterns/memory-architecture + aic/aiv-core-object，Ascend 950 架构）：
//   · 纵向轨道 rail        = 芯片级共享存储（Global Memory 灰 / L2 Cache 蓝），点阵=容量块
//   · 核容器 container     = 执行域（AIC=Cube 矩阵引擎 · AIV=SIMD/SIMT 向量引擎）
//   · 缓冲块 buffer        = 名称+容量徽标+单元格网格（L1/L0A/L0B/BT/FP/L0C/UB），橙格=tile 驻留
//   · 执行单元 exec        = CUBE(青)/Scalar(绿)/SIMD·SIMT 面板
//   · 正交路由 route+chip  = MTE 搬运通路（MTE2 入片、MTE1 L1→L0、MTE3 回写、FixPipe、
//                            L0C→UB CV 直连），黄 chip=搬运指令，实线=数据、虚线=指令
//
// 可泛化语法：共享资源轨道 ｜ 容器块（容量格网）｜ 执行/交换单元 ｜ 带标签 chip 的正交路由。
// 本文件提供与视图无关的 layout（纯几何）+ canvas 渲染器 + SVG 渲染器，
// 层级图 / 平面图 / Smartscape / 运行状态用同一套图元，保证“相同的图元”跨视图一致。
// ─────────────────────────────────────────────────────────────────────────────
import React from 'react';

export const ARCH_COLORS = {
  gmRail: '#9aa3b2',       // Global Memory 轨道（灰）
  l2Rail: '#4a8cff',       // L2 Cache 轨道（蓝）
  cellBase: '#3f6fe0',     // 缓冲单元格（蓝）
  cellOcc: '#f59e0b',      // 占用块（橙 · setBufferBlocks 语义）
  cube: '#36e0c4',         // CUBE 矩阵引擎（青）
  scalar: '#3f9e4d',       // Scalar（绿）
  simd: '#7dd3fc',         // SIMD/SIMT 面板（浅青）
  mteChip: '#f5c84b',      // MTE 搬运 chip（黄）
  cvLane: '#22d3ee',       // L0C→UB CV 直连（青）
  containerDark: '#232833',
  containerLight: '#eef1f7',
  labelDark: '#c7cede',
  labelLight: '#3d4658',
} as const;

// ─── 图元原语（纯数据，两个渲染器共用）─────────────────────────────────────────
export type ArchPrim =
  | { t: 'rail'; x: number; y: number; w: number; h: number; label: string; color: string }
  | { t: 'container'; x: number; y: number; w: number; h: number; label: string }
  | { t: 'buffer'; x: number; y: number; w: number; h: number; name: string; cap?: string; cols: number; rows: number; occKey: string }
  | { t: 'exec'; x: number; y: number; w: number; h: number; label: string; color: string }
  | { t: 'route'; pts: [number, number][]; color: string; dashed?: boolean; hotKeys?: string[] }
  | { t: 'chip'; x: number; y: number; label: string; color: string; hotKeys?: string[] }
  | { t: 'text'; x: number; y: number; label: string; size: number; dim?: boolean };

// 相位 → 高亮的路由/占用的缓冲（与 CoreGroupPattern 的 PHASE_FOCUS 语义一致）
export const ARCH_PHASE: Record<string, { hot: string[]; occ: string[] }> = {
  load:    { hot: ['mte2'], occ: ['UB', 'L0A', 'L0B'] },
  compute: { hot: ['cube', 'cv'], occ: ['L0C', 'UB'] },
  comm:    { hot: ['mte3'], occ: ['UB'] },
  mem:     { hot: ['mte2', 'l2'], occ: ['L1', 'UB'] },
  store:   { hot: ['mte3', 'fixp'], occ: ['L0C'] },
};

/** L0 核组 mini 架构（GM/L2 轨道 + AIV1/AIC/AIV2 + MTE 路由）。
 *  detail 0=极简（行内 glyph）；1=标准（含 L0A/L0B/L0C/UB 格网）；2=展开（含 BT/FP/Scalar）。
 *  返回以 (0,0)-(w,h) 为界的原语列表。 */
export function layoutCoreGroupMini(w: number, h: number, detail: 0 | 1 | 2 = 1): ArchPrim[] {
  const P: ArchPrim[] = [];
  const railW = Math.max(6, w * 0.055), gap = Math.max(3, w * 0.02);
  const gmX = 0, l2X = railW + gap * 0.6, coreX = l2X + railW + gap * 1.6, coreW = w - coreX;
  P.push({ t: 'rail', x: gmX, y: 0, w: railW, h, label: 'GM', color: ARCH_COLORS.gmRail });
  P.push({ t: 'rail', x: l2X, y: 0, w: railW, h, label: 'L2', color: ARCH_COLORS.l2Rail });
  // 三个核容器：AIV1 / AIC / AIV2
  const gapY = Math.max(2, h * 0.03);
  const aivH = h * 0.27, aicH = h - 2 * aivH - 2 * gapY;
  const rows: { label: string; y: number; hh: number; kind: 'aiv' | 'aic' }[] = [
    { label: 'AIV 1', y: 0, hh: aivH, kind: 'aiv' },
    { label: 'AIC', y: aivH + gapY, hh: aicH, kind: 'aic' },
    { label: 'AIV 2', y: aivH + gapY + aicH + gapY, hh: aivH, kind: 'aiv' },
  ];
  for (const r of rows) {
    P.push({ t: 'container', x: coreX, y: r.y, w: coreW, h: r.hh, label: r.label });
    const pad = Math.max(3, coreW * 0.045), innerX = coreX + pad, innerW = coreW - pad * 2;
    if (r.kind === 'aiv') {
      // UB 缓冲（左）+ SIMD/SIMT（右）
      const ubW = innerW * (detail >= 1 ? 0.46 : 0.6), unitH = Math.max(6, r.hh - pad * 2 - 8);
      P.push({ t: 'buffer', x: innerX, y: r.y + pad + 7, w: ubW, h: unitH, name: 'UB', cap: '64kb', cols: detail >= 1 ? 8 : 4, rows: detail >= 1 ? 3 : 2, occKey: 'UB' });
      P.push({ t: 'exec', x: innerX + ubW + pad, y: r.y + pad + 7, w: innerW - ubW - pad, h: unitH, label: detail >= 2 ? 'SIMT · SIMD' : 'SIMD', color: ARCH_COLORS.simd });
      // MTE2：GM → UB
      P.push({ t: 'route', pts: [[gmX + railW, r.y + r.hh * 0.5], [innerX, r.y + r.hh * 0.5]], color: ARCH_COLORS.mteChip, hotKeys: ['mte2'] });
      if (detail >= 1) P.push({ t: 'chip', x: (gmX + railW + innerX) / 2, y: r.y + r.hh * 0.5 - 6, label: 'MTE2', color: ARCH_COLORS.mteChip, hotKeys: ['mte2'] });
    } else {
      // AIC：L1 → L0A/L0B → CUBE → L0C（detail≥2 加 BT/FP/Scalar）
      const colW = innerW / (detail >= 1 ? 4 : 3), unitH = Math.max(6, r.hh - pad * 2 - 8), uy = r.y + pad + 7;
      P.push({ t: 'buffer', x: innerX, y: uy, w: colW * 0.86, h: unitH, name: 'L1', cap: '512kb', cols: 5, rows: detail >= 1 ? 5 : 3, occKey: 'L1' });
      if (detail >= 1) {
        const abH = detail >= 2 ? unitH * 0.42 : unitH * 0.46;
        P.push({ t: 'buffer', x: innerX + colW, y: uy, w: colW * 0.8, h: abH, name: 'L0A', cols: 5, rows: 2, occKey: 'L0A' });
        P.push({ t: 'buffer', x: innerX + colW, y: uy + unitH - abH, w: colW * 0.8, h: abH, name: 'L0B', cols: 5, rows: 2, occKey: 'L0B' });
        if (detail >= 2 && unitH - 2 * abH - 4 >= 8) P.push({ t: 'buffer', x: innerX + colW, y: uy + abH + 2, w: colW * 0.8, h: unitH - 2 * abH - 4, name: 'BT·FP', cols: 5, rows: 1, occKey: 'FP' });
        P.push({ t: 'route', pts: [[innerX + colW * 0.86, uy + unitH * 0.5], [innerX + colW, uy + unitH * 0.5]], color: ARCH_COLORS.mteChip, hotKeys: ['mte1'] });
        P.push({ t: 'chip', x: innerX + colW * 0.93, y: uy + unitH * 0.5 - 6, label: 'MTE1', color: ARCH_COLORS.mteChip, hotKeys: ['mte1'] });
      }
      const cubeX = innerX + colW * (detail >= 1 ? 2 : 1.1);
      P.push({ t: 'exec', x: cubeX, y: uy + unitH * 0.18, w: colW * 0.8, h: unitH * 0.64, label: 'CUBE', color: ARCH_COLORS.cube });
      const l0cX = cubeX + colW * 0.95;
      P.push({ t: 'buffer', x: l0cX, y: uy, w: Math.min(colW * 0.85, coreX + coreW - l0cX - pad), h: unitH, name: 'L0C', cap: '512kb', cols: 5, rows: detail >= 1 ? 5 : 3, occKey: 'L0C' });
      P.push({ t: 'route', pts: [[cubeX + colW * 0.8, uy + unitH * 0.5], [l0cX, uy + unitH * 0.5]], color: ARCH_COLORS.cube, hotKeys: ['cube'] });
      // MTE2：L2 → L1
      P.push({ t: 'route', pts: [[l2X + railW, r.y + r.hh * 0.5], [innerX, r.y + r.hh * 0.5]], color: ARCH_COLORS.mteChip, hotKeys: ['mte2', 'l2'] });
      // L0C→UB CV 直连（到 AIV1 的 UB）
      P.push({ t: 'route', pts: [[l0cX + colW * 0.4, uy], [l0cX + colW * 0.4, rows[0].y + rows[0].hh - 2]], color: ARCH_COLORS.cvLane, dashed: detail < 1, hotKeys: ['cv'] });
      if (detail >= 1) P.push({ t: 'chip', x: l0cX + colW * 0.4, y: (uy + rows[0].y + rows[0].hh) / 2 - 6, label: 'L0C→UB', color: ARCH_COLORS.cvLane, hotKeys: ['cv'] });
      if (detail >= 2) P.push({ t: 'exec', x: innerX, y: r.y + r.hh - pad - 7, w: colW * 0.7, h: 9, label: 'Scalar', color: ARCH_COLORS.scalar });
    }
  }
  // MTE3：AIV2 UB → GM（回写）
  const r2 = rows[2];
  P.push({ t: 'route', pts: [[coreX + Math.max(3, coreW * 0.045), r2.y + r2.hh * 0.78], [gmX + railW, r2.y + r2.hh * 0.78]], color: ARCH_COLORS.mteChip, dashed: true, hotKeys: ['mte3', 'fixp'] });
  if (detail >= 1) P.push({ t: 'chip', x: (gmX + railW + coreX) / 2, y: r2.y + r2.hh * 0.78 - 6, label: 'MTE3', color: ARCH_COLORS.mteChip, hotKeys: ['mte3'] });
  return P;
}

// ─── canvas 渲染器（层级图 / 平面图 / 运行状态画布用）────────────────────────────
export interface ArchDrawOpts { dark?: boolean; phase?: string; load?: number; alpha?: number; /** 字号缩放（提升小尺寸可读性） */ fs?: number }
export function drawArchPrims(ctx: CanvasRenderingContext2D, prims: ArchPrim[], ox: number, oy: number, o: ArchDrawOpts = {}) {
  const hot = o.phase ? ARCH_PHASE[o.phase]?.hot ?? [] : [];
  const occ = o.phase ? ARCH_PHASE[o.phase]?.occ ?? [] : [];
  const occFrac = Math.max(0.15, Math.min(0.9, o.load ?? 0.5));
  ctx.save();
  if (o.alpha != null) ctx.globalAlpha = o.alpha;
  const label = o.dark ? ARCH_COLORS.labelDark : ARCH_COLORS.labelLight;
  const F = o.fs ?? 1;   // 字号缩放
  for (const p of prims) {
    if (p.t === 'rail') {
      ctx.fillStyle = p.color; ctx.globalAlpha = (o.alpha ?? 1) * 0.16;
      ctx.fillRect(ox + p.x, oy + p.y, p.w, p.h);
      ctx.globalAlpha = (o.alpha ?? 1) * 0.75; ctx.fillStyle = p.color;
      const step = 4.6;   // 点阵容量块
      for (let yy = oy + p.y + 3; yy < oy + p.y + p.h - 3; yy += step)
        for (let xx = ox + p.x + 2; xx < ox + p.x + p.w - 2; xx += step) ctx.fillRect(xx, yy, 2, 2);
      ctx.globalAlpha = o.alpha ?? 1; ctx.fillStyle = label; ctx.font = `600 ${7 * F}px sans-serif`; ctx.textAlign = 'center';
      ctx.fillText(p.label, ox + p.x + p.w / 2, oy + p.y + p.h - 3);
    } else if (p.t === 'container') {
      ctx.fillStyle = o.dark ? ARCH_COLORS.containerDark : ARCH_COLORS.containerLight;
      ctx.strokeStyle = o.dark ? '#3a4152' : '#c6cddc'; ctx.lineWidth = 1;
      ctx.beginPath(); (ctx as any).roundRect(ox + p.x, oy + p.y, p.w, p.h, 4); ctx.fill(); ctx.stroke();
      ctx.fillStyle = label; ctx.font = `700 ${8 * F}px sans-serif`; ctx.textAlign = 'left';
      ctx.fillText(p.label, ox + p.x + 4, oy + p.y + 8);
    } else if (p.t === 'buffer') {
      ctx.strokeStyle = o.dark ? '#4a5468' : '#9aa6bd'; ctx.lineWidth = 0.8;
      ctx.beginPath(); (ctx as any).roundRect(ox + p.x, oy + p.y, p.w, p.h, 2.5); ctx.stroke();
      ctx.fillStyle = label; ctx.font = `600 ${7 * F}px sans-serif`; ctx.textAlign = 'left';
      ctx.fillText(p.name + (p.cap ? ` ${p.cap}` : ''), ox + p.x + 2, oy + p.y + 7);
      const gx = ox + p.x + 2, gy = oy + p.y + 9, gw = p.w - 4, gh = p.h - 11;
      const nRows = Math.max(1, Math.min(p.rows, Math.floor(gh / 3)));   // 小尺寸下压缩行数
      if (gh >= 3 && gw >= 3) {
        const cw = gw / p.cols, ch = Math.max(1.5, Math.min(gh / nRows, 5));
        const isOcc = occ.includes(p.occKey);
        const occCells = isOcc ? Math.round(p.cols * nRows * occFrac) : 0;
        let k = 0;
        for (let r = 0; r < nRows; r++) for (let c = 0; c < p.cols; c++, k++) {
          ctx.fillStyle = k < occCells ? ARCH_COLORS.cellOcc : ARCH_COLORS.cellBase;
          ctx.globalAlpha = (o.alpha ?? 1) * (k < occCells ? 0.95 : 0.55);
          ctx.fillRect(gx + c * cw, gy + r * (ch + 1), Math.max(1.5, cw - 1), Math.max(1, ch - 0.5));
        }
      }
      ctx.globalAlpha = o.alpha ?? 1;
    } else if (p.t === 'exec') {
      ctx.fillStyle = p.color; ctx.globalAlpha = (o.alpha ?? 1) * (hot.includes('cube') && p.label === 'CUBE' ? 1 : 0.82);
      ctx.beginPath(); (ctx as any).roundRect(ox + p.x, oy + p.y, p.w, p.h, 3); ctx.fill();
      ctx.globalAlpha = o.alpha ?? 1; ctx.fillStyle = o.dark ? '#10141c' : '#ffffff';
      ctx.font = `700 ${7.5 * F}px sans-serif`; ctx.textAlign = 'center';
      ctx.fillText(p.label, ox + p.x + p.w / 2, oy + p.y + p.h / 2 + 2.5);
    } else if (p.t === 'route') {
      const isHot = p.hotKeys?.some((k) => hot.includes(k));
      ctx.strokeStyle = p.color; ctx.lineWidth = isHot ? 2.2 : 1.2;
      ctx.globalAlpha = (o.alpha ?? 1) * (isHot ? 1 : 0.45);
      ctx.setLineDash(p.dashed ? [3, 2.5] : []);
      ctx.beginPath(); ctx.moveTo(ox + p.pts[0][0], oy + p.pts[0][1]);
      for (const [x, y] of p.pts.slice(1)) ctx.lineTo(ox + x, oy + y);
      ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = o.alpha ?? 1;
    } else if (p.t === 'chip') {
      const isHot = p.hotKeys?.some((k) => hot.includes(k));
      ctx.font = `700 ${6.5 * F}px sans-serif`; const tw = ctx.measureText(p.label).width + 7;
      ctx.fillStyle = p.color; ctx.globalAlpha = (o.alpha ?? 1) * (isHot ? 1 : 0.85);
      ctx.beginPath(); (ctx as any).roundRect(ox + p.x - tw / 2, oy + p.y, tw, 10 * F, 5 * F); ctx.fill();
      ctx.globalAlpha = o.alpha ?? 1; ctx.fillStyle = '#1a1e28'; ctx.textAlign = 'center';
      ctx.fillText(p.label, ox + p.x, oy + p.y + 7 * F);
    } else if (p.t === 'text') {
      ctx.fillStyle = label; ctx.globalAlpha = (o.alpha ?? 1) * (p.dim ? 0.6 : 1);
      ctx.font = `${p.size}px sans-serif`; ctx.textAlign = 'left';
      ctx.fillText(p.label, ox + p.x, oy + p.y); ctx.globalAlpha = o.alpha ?? 1;
    }
  }
  ctx.restore();
}

/** canvas 一步到位：在 (x,y,w,h) 内画 L0 核组 mini 架构 */
export function drawCoreGroupMini(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, o: ArchDrawOpts & { detail?: 0 | 1 | 2 } = {}) {
  drawArchPrims(ctx, layoutCoreGroupMini(w, h, o.detail ?? 1), x, y, o);
}

// ─── SVG 渲染器（Smartscape 等 SVG 视图用）──────────────────────────────────────
export function CoreGroupMiniSvg({ width, height, detail = 1, phase, load = 0.5, dark = true, opacity = 1, fs = 1 }:
  { width: number; height: number; detail?: 0 | 1 | 2; phase?: string; load?: number; dark?: boolean; opacity?: number; /** 字号缩放 */ fs?: number }) {
  const prims = layoutCoreGroupMini(width, height, detail);
  const hot = phase ? ARCH_PHASE[phase]?.hot ?? [] : [];
  const occ = phase ? ARCH_PHASE[phase]?.occ ?? [] : [];
  const occFrac = Math.max(0.15, Math.min(0.9, load));
  const label = dark ? ARCH_COLORS.labelDark : ARCH_COLORS.labelLight;
  return (
    <g opacity={opacity}>
      {prims.map((p, i) => {
        if (p.t === 'rail') {
          const dots: React.ReactNode[] = [];
          for (let yy = p.y + 3, r = 0; yy < p.y + p.h - 3; yy += 4.6, r++)
            for (let xx = p.x + 2, c = 0; xx < p.x + p.w - 2; xx += 4.6, c++)
              dots.push(<rect key={`${r}-${c}`} x={xx} y={yy} width={2} height={2} fill={p.color} opacity={0.75} />);
          return <g key={i}><rect x={p.x} y={p.y} width={p.w} height={p.h} fill={p.color} opacity={0.16} />{dots}
            <text x={p.x + p.w / 2} y={p.y + p.h - 3} fontSize={7 * fs} fontWeight={600} fill={label} textAnchor="middle">{p.label}</text></g>;
        }
        if (p.t === 'container') return <g key={i}>
          <rect x={p.x} y={p.y} width={p.w} height={p.h} rx={4} fill={dark ? ARCH_COLORS.containerDark : ARCH_COLORS.containerLight} stroke={dark ? '#3a4152' : '#c6cddc'} strokeWidth={1} />
          <text x={p.x + 4} y={p.y + 8} fontSize={8 * fs} fontWeight={700} fill={label}>{p.label}</text></g>;
        if (p.t === 'buffer') {
          const cells: React.ReactNode[] = [];
          const gx = p.x + 2, gy = p.y + 9, gh = p.h - 11;
          const nRows = Math.max(1, Math.min(p.rows, Math.floor(gh / 3)));   // 小尺寸下压缩行数
          if (gh >= 3 && p.w >= 7) {
            const cw = (p.w - 4) / p.cols, ch = Math.max(1.5, Math.min(gh / nRows, 5));
            const occCells = occ.includes(p.occKey) ? Math.round(p.cols * nRows * occFrac) : 0;
            let k = 0;
            for (let r = 0; r < nRows; r++) for (let c = 0; c < p.cols; c++, k++)
              cells.push(<rect key={k} x={gx + c * cw} y={gy + r * (ch + 1)} width={Math.max(1.5, cw - 1)} height={Math.max(1, ch - 0.5)}
                fill={k < occCells ? ARCH_COLORS.cellOcc : ARCH_COLORS.cellBase} opacity={k < occCells ? 0.95 : 0.55} />);
          }
          return <g key={i}>
            <rect x={p.x} y={p.y} width={p.w} height={p.h} rx={2.5} fill="none" stroke={dark ? '#4a5468' : '#9aa6bd'} strokeWidth={0.8} />
            <text x={p.x + 2} y={p.y + 7} fontSize={7 * fs} fontWeight={600} fill={label}>{p.name}{p.cap ? ` ${p.cap}` : ''}</text>{cells}</g>;
        }
        if (p.t === 'exec') return <g key={i}>
          <rect x={p.x} y={p.y} width={p.w} height={p.h} rx={3} fill={p.color} opacity={hot.includes('cube') && p.label === 'CUBE' ? 1 : 0.82} />
          <text x={p.x + p.w / 2} y={p.y + p.h / 2 + 2.5} fontSize={7.5 * fs} fontWeight={700} fill={dark ? '#10141c' : '#ffffff'} textAnchor="middle">{p.label}</text></g>;
        if (p.t === 'route') {
          const isHot = p.hotKeys?.some((k) => hot.includes(k));
          return <polyline key={i} points={p.pts.map(([x, y]) => `${x},${y}`).join(' ')} fill="none" stroke={p.color}
            strokeWidth={isHot ? 2.2 : 1.2} strokeDasharray={p.dashed ? '3 2.5' : undefined} opacity={isHot ? 1 : 0.45} />;
        }
        if (p.t === 'chip') {
          const isHot = p.hotKeys?.some((k) => hot.includes(k));
          const tw = p.label.length * 4.4 * fs + 7;
          return <g key={i}>
            <rect x={p.x - tw / 2} y={p.y} width={tw} height={10 * fs} rx={5 * fs} fill={p.color} opacity={isHot ? 1 : 0.85} />
            <text x={p.x} y={p.y + 7 * fs} fontSize={6.5 * fs} fontWeight={700} fill="#1a1e28" textAnchor="middle">{p.label}</text></g>;
        }
        return <text key={i} x={p.x} y={p.y} fontSize={p.size} fill={label} opacity={p.dim ? 0.6 : 1}>{p.label}</text>;
      })}
    </g>
  );
}

// ─── 层级容器图元（其他层级用同一语法“细化”）────────────────────────────────────
// 统一的“容器 + 构件块 + 端口 tab + 路由 chip”画法，供层级图各行/Smartscape 各级使用。
export interface LevelBlockSpec { label: string; color: string; w?: number; kind?: 'unit' | 'switch' | 'mem' }
/** 在 (x,y,w,h) 画一个层级容器：外框 + 标题 + 内部构件块一行 + 可选端口 tab。canvas 版。 */
export function drawLevelContainer(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number,
  title: string, blocks: LevelBlockSpec[], o: ArchDrawOpts & { ports?: { label: string; color: string }[]; ghost?: boolean } = {}) {
  ctx.save();
  ctx.globalAlpha = (o.alpha ?? 1) * (o.ghost ? 0.35 : 1);
  ctx.strokeStyle = o.dark ? '#3a4152' : '#c6cddc'; ctx.lineWidth = 1;
  ctx.fillStyle = o.dark ? 'rgba(35,40,51,0.85)' : 'rgba(238,241,247,0.9)';
  ctx.beginPath(); (ctx as any).roundRect(x, y, w, h, 5); ctx.fill(); ctx.stroke();
  ctx.fillStyle = o.dark ? ARCH_COLORS.labelDark : ARCH_COLORS.labelLight;
  ctx.font = '700 8px sans-serif'; ctx.textAlign = 'left'; ctx.fillText(title, x + 5, y + 10);
  // 构件块一行（unit=圆角块 · switch=菱形调 · mem=点阵条）
  const pad = 5, by = y + 14, bh = h - 20;
  const totalW = blocks.reduce((a, b) => a + (b.w ?? 1), 0);
  let bx = x + pad;
  for (const b of blocks) {
    const bw = ((w - pad * 2 - (blocks.length - 1) * 3) * (b.w ?? 1)) / totalW;
    ctx.fillStyle = b.color; ctx.globalAlpha = (o.alpha ?? 1) * (o.ghost ? 0.3 : b.kind === 'mem' ? 0.35 : 0.75);
    ctx.beginPath();
    if (b.kind === 'switch') { (ctx as any).roundRect(bx, by + bh * 0.15, bw, bh * 0.7, bh * 0.35); }
    else (ctx as any).roundRect(bx, by, bw, bh, 3);
    ctx.fill();
    ctx.globalAlpha = (o.alpha ?? 1) * (o.ghost ? 0.5 : 1);
    ctx.fillStyle = o.dark ? '#e7ebf4' : '#2c3446'; ctx.font = '600 6.5px sans-serif'; ctx.textAlign = 'center';
    if (bw > 16) ctx.fillText(b.label, bx + bw / 2, by + bh / 2 + 2.5);
    bx += bw + 3;
  }
  // 端口 tab（UB 口/RDMA 口这类对外端口，画在容器下沿）
  if (o.ports) {
    let px = x + w - 6 - o.ports.length * 16;
    for (const pt of o.ports) {
      ctx.fillStyle = pt.color; ctx.globalAlpha = (o.alpha ?? 1) * 0.9;
      ctx.fillRect(px, y + h - 3, 13, 5);
      ctx.font = '600 5.5px sans-serif'; ctx.textAlign = 'center'; ctx.fillStyle = o.dark ? '#0f131b' : '#ffffff';
      ctx.fillText(pt.label, px + 6.5, y + h + 1.2);
      px += 16;
    }
  }
  ctx.restore();
}
/** 两级容器之间的互联 chip（DCN/Scale-Out/Pool 内互联/Scale-Up/PCIe·UB/封装互连/NoC）。canvas 版。 */
export function drawInterconnectChip(ctx: CanvasRenderingContext2D, x: number, y: number, label: string, color: string, o: ArchDrawOpts = {}) {
  ctx.save();
  ctx.strokeStyle = color; ctx.globalAlpha = (o.alpha ?? 1) * 0.75; ctx.lineWidth = 1.1;
  ctx.beginPath(); ctx.moveTo(x, y - 9); ctx.lineTo(x, y + 9); ctx.stroke();
  ctx.font = '700 7px sans-serif';
  const tw = ctx.measureText(label).width + 10;
  ctx.fillStyle = color; ctx.globalAlpha = (o.alpha ?? 1) * 0.16;
  ctx.beginPath(); (ctx as any).roundRect(x - tw / 2, y - 6, tw, 12, 6); ctx.fill();
  ctx.globalAlpha = o.alpha ?? 1; ctx.strokeStyle = color; ctx.globalAlpha = (o.alpha ?? 1) * 0.5;
  ctx.beginPath(); (ctx as any).roundRect(x - tw / 2, y - 6, tw, 12, 6); ctx.stroke();
  ctx.globalAlpha = o.alpha ?? 1; ctx.fillStyle = color; ctx.textAlign = 'center';
  ctx.fillText(label, x, y + 2.5);
  ctx.restore();
}
