/**
 * CommDeepDive — 集合通信深潜诊断（原型第八组件工程化）。
 *
 * 把 NCCL 生态的诊断方法平移到昇腾超节点（HCCL + 灵衢）监控，三视图：
 *   ① 环构建与跨域散射（SCATTER_XDC）：环先域内成环、松散端跨域拼接；散射把跨域流量分摊到多对节点/多链路。
 *   ② PXN 轨道路径诊断：rail-optimized + NVLink 换轨避开 Spine（3 跳 → 1 跳，AllToAll 翻倍）。
 *   ③ busbw–消息大小健康基线：峰后回落 = ECMP 哈希不均指纹；平台低于直连基线即告警。
 *
 * 语义/配色沿用原型；面板底色/文字用 PTO 主题 token（深浅色自适应）。
 */
import { useEffect, useRef, useState } from 'react';

type Comm = 0 | 1 | 2;
const MONO = "'JetBrains Mono','Consolas',ui-monospace,monospace";
const SECONDARY: React.CSSProperties = { border: '1px solid var(--button-secondary-border)', background: 'var(--button-secondary-bg)', color: 'var(--foreground-muted)' };
const btn: React.CSSProperties = { padding: '3px 9px', fontSize: 11, borderRadius: 7, cursor: 'pointer' };
function tab(on: boolean): React.CSSProperties { return on ? { border: '1px solid var(--primary)', background: 'var(--primary)', color: 'var(--primary-foreground)', fontWeight: 600 } : { ...SECONDARY }; }
function redBtn(on: boolean): React.CSSProperties { return on ? { border: '1px solid #f85149', background: '#f85149', color: '#fff', fontWeight: 600 } : { ...SECONDARY }; }
function grnBtn(on: boolean): React.CSSProperties { return on ? { border: '1px solid #3fb950', background: '#3fb950', color: '#08130b', fontWeight: 600 } : { ...SECONDARY }; }
const panel: React.CSSProperties = { background: 'var(--panel-solid)', border: '1px solid var(--bd)', borderRadius: 8, overflowX: 'auto' };
const desc: React.CSSProperties = { fontSize: 10.5, color: 'var(--tx2)', lineHeight: 1.55, marginTop: 8 };

// ── ① 环构建与跨域散射 ──
const RA: [number, number][] = [[80, 70], [200, 70], [200, 180], [80, 180]];
const RB: [number, number][] = [[520, 70], [640, 70], [640, 180], [520, 180]];
function ringSVG(on: boolean): { html: string; note: string; noteRed: boolean } {
  const node = (x: number, y: number, l: string) => `<rect x="${x - 22}" y="${y - 16}" width="44" height="32" rx="5" fill="rgba(127,140,170,0.14)" stroke="#7c8db8"/><text x="${x}" y="${y + 4}" fill="#c9d1d9" font-size="10" text-anchor="middle">${l}</text>`;
  const seg = (p: number[], q: number[], c: string, w: number, dash: boolean) => `<line x1="${p[0]}" y1="${p[1]}" x2="${q[0]}" y2="${q[1]}" stroke="${c}" stroke-width="${w}" ${dash ? 'stroke-dasharray="7 5"' : ''} opacity=".9"/>`;
  let h = `<rect x="40" y="30" width="200" height="185" rx="10" fill="none" stroke="#2a3546"/><text x="60" y="22" fill="#8b949e" font-size="11">超节点 A（域内：灵衢/UB FullMesh）</text>
         <rect x="480" y="30" width="200" height="185" rx="10" fill="none" stroke="#2a3546"/><text x="500" y="22" fill="#8b949e" font-size="11">超节点 B</text>`;
  ([[RA, '#58a6ff', 0], [RA, '#a371f7', 5], [RB, '#58a6ff', 0], [RB, '#a371f7', 5]] as [number[][], string, number][]).forEach(([P, c, o]) => {
    for (let k = 0; k < 3; k++) h += seg([P[k][0] + o, P[k][1] + o], [P[k + 1][0] + o, P[k + 1][1] + o], c, 1.6, false);
  });
  let note = '', noteRed = false;
  if (!on) {
    h += seg([RA[1][0], RA[1][1] - 3], [RB[0][0], RB[0][1] - 3], '#f85149', 3.4, true);
    h += seg([RA[1][0], RA[1][1] + 4], [RB[0][0], RB[0][1] + 4], '#f85149', 3.4, true);
    h += seg([RB[3][0], RB[3][1] - 3], [RA[2][0], RA[2][1] - 3], '#f85149', 3.4, true);
    h += seg([RB[3][0], RB[3][1] + 4], [RA[2][0], RA[2][1] + 4], '#f85149', 3.4, true);
    h += `<text x="360" y="52" fill="#f85149" font-size="11" text-anchor="middle">全部环挤在同一对节点跨域：需求 1.6 Tbps / 链路</text>`;
    note = '无散射：环先在各域内构建，再用松散端跨域拼接（跨域链路数恒为 2×(nDC−1)）。但所有环都从同一对节点跨越慢链路——4 GPU×400Gbps 全压到单链路，成为整个集合的瓶颈，对应泳道图上的暴露气泡。'; noteRed = true;
  } else {
    h += seg(RA[1], RB[0], '#3fb950', 2, true);
    h += seg(RB[3], RA[2], '#3fb950', 2, true);
    h += `<path d="M${RA[0][0]},${RA[0][1] - 16} C220,-14 500,-14 ${RB[1][0]},${RB[1][1] - 16}" fill="none" stroke="#3fb950" stroke-width="2" stroke-dasharray="7 5"/>`;
    h += `<path d="M${RB[2][0]},${RB[2][1] + 16} C500,246 220,246 ${RA[3][0]},${RA[3][1] + 16}" fill="none" stroke="#3fb950" stroke-width="2" stroke-dasharray="7 5"/>`;
    h += `<text x="360" y="52" fill="#3fb950" font-size="11" text-anchor="middle">✓ 每个环从不同节点跨域：单链路需求降回 400 Gbps</text>`;
    note = 'SCATTER_XDC=1 散射：环 1 走 A1↔B0/B3↔A2，环 2 改走 A0↔B1/B2↔A3——跨域流量分摊到多对节点与多条链路。昇腾对照：跨超节点的 OCS 光链路同理需要把多环/多通道散射到不同端口。';
  }
  RA.forEach((p, i) => h += node(p[0], p[1], 'A' + i));
  RB.forEach((p, i) => h += node(p[0], p[1], 'B' + i));
  return { html: h, note, noteRed };
}

// ── ② PXN 轨道路径 ──
function pxnSVG(on: boolean): { html: string; note: string; noteRed: boolean } {
  const AG = (k: number) => [70 + 60 * k, 255], AN = (k: number) => [70 + 60 * k, 195], BG = (k: number) => [430 + 60 * k, 255], BN = (k: number) => [430 + 60 * k, 195], L = (k: number) => [140 + 140 * k, 100], SP = [360, 30];
  const box = (x: number, y: number, l: string, c?: string) => `<rect x="${x - 16}" y="${y - 11}" width="32" height="22" rx="4" fill="rgba(127,140,170,0.14)" stroke="${c || '#7c8db8'}"/><text x="${x}" y="${y + 4}" fill="#c9d1d9" font-size="8" text-anchor="middle">${l}</text>`;
  const ln = (p: number[], q: number[], c: string, w: number, dash: boolean) => `<line x1="${p[0]}" y1="${p[1]}" x2="${q[0]}" y2="${q[1]}" stroke="${c}" stroke-width="${w}" ${dash ? 'stroke-dasharray="6 4"' : ''}/>`;
  let h = `<rect x="40" y="170" width="250" height="110" rx="8" fill="none" stroke="#2a3546"/><text x="50" y="164" fill="#8b949e" font-size="10">服务器 A（内部 灵衢 FullMesh）</text>
         <rect x="400" y="170" width="250" height="110" rx="8" fill="none" stroke="#2a3546"/><text x="410" y="164" fill="#8b949e" font-size="10">服务器 B</text>`;
  for (let k = 0; k < 4; k++) h += ln(AN(k), L(k), '#3a4658', 1, false) + ln(BN(k), L(k), '#3a4658', 1, false) + ln(L(k), SP, '#3a4658', 1, false) + ln(AG(k), AN(k), '#3a4658', 1, false) + ln(BG(k), BN(k), '#3a4658', 1, false);
  const path = on ? [AG(0), AG(3), AN(3), L(3), BN(3), BG(3)] : [AG(0), AN(0), L(0), SP, L(3), BN(3), BG(3)];
  const pc = on ? '#3fb950' : '#d29922';
  for (let k = 0; k < path.length - 1; k++) { const crossSpine = !on && (k === 2 || k === 3); h += ln(path[k], path[k + 1], crossSpine ? '#f85149' : pc, crossSpine ? 3.4 : 2.6, true); }
  if (on) h += `<text x="160" y="245" fill="#39c5cf" font-size="9" text-anchor="middle">① 灵衢换轨到同轨 NPU</text>`;
  h += `<rect x="${SP[0] - 30}" y="${SP[1] - 13}" width="60" height="26" rx="5" fill="${!on ? '#67060c' : 'rgba(127,140,170,0.14)'}" stroke="${!on ? '#f85149' : '#7c8db8'}"/><text x="${SP[0]}" y="${SP[1] + 4}" fill="#fff" font-size="10" text-anchor="middle">Spine S</text>`;
  for (let k = 0; k < 4; k++) {
    h += box(L(k)[0], L(k)[1], 'L' + k, (k === 0 && !on) || k === 3 ? '#8b949e' : undefined);
    h += box(AG(k)[0], AG(k)[1], 'G' + k, k === 0 ? '#d29922' : (on && k === 3 ? '#3fb950' : undefined));
    h += box(AN(k)[0], AN(k)[1], 'N' + k);
    h += box(BG(k)[0], BG(k)[1], 'G' + k, k === 3 ? '#3fb950' : undefined);
    h += box(BN(k)[0], BN(k)[1], 'N' + k);
  }
  const note = on
    ? 'PXN 路径：A.G0 先经灵衢把数据挪到与目的地同轨的 A.G3，再走 A.N3 → 叶交换机 L3 → B.G3——全程不穿 Spine，且同目的消息可聚合（最多 8 合 1）。AllToAll 实测性能翻倍。昇腾对照：柜内灵衢 FullMesh 换轨 + 柜间 OCS 同轨直达。'
    : '无 PXN：A.G0 → A.N0 → L0 → Spine → L3 → B.G3，穿越 3 台交换机；Spine 上与其他流量争抢产生拥塞（红色高亮）。监控视图应能对任意 NPU 对展开这样的 hop-by-hop 路径。';
  return { html: h, note, noteRed: !on };
}

// ── ③ busbw 曲线（canvas 端口自 drawBW）──
function BusbwCanvas({ dark }: { dark: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const draw = () => {
      const dpr = window.devicePixelRatio || 1, W = c.clientWidth || 680, H = 260;
      c.width = W * dpr; c.height = H * dpr; const x = c.getContext('2d'); if (!x) return; x.scale(dpr, dpr); x.clearRect(0, 0, W, H);
      const N = 26, padL = 46, padR = 14, padT = 16, padB = 34;
      const X = (i: number) => padL + i * (W - padL - padR) / (N - 1), Y = (v: number) => H - padB - (v / 210) * (H - padT - padB);
      x.strokeStyle = dark ? '#26303e' : '#dfe4ec'; x.fillStyle = dark ? '#8b949e' : '#5b6573'; x.font = '10px monospace'; x.textAlign = 'right';
      [0, 50, 100, 150, 195].forEach((v) => { x.beginPath(); x.moveTo(padL, Y(v)); x.lineTo(W - padR, Y(v)); x.stroke(); x.fillText(String(v), padL - 6, Y(v) + 3); });
      x.textAlign = 'center';
      ['512B', '4KB', '32KB', '256KB', '2MB', '16MB', '128MB', '1GB', '8GB'].forEach((l, k) => x.fillText(l, X(k * 3), H - padB + 14));
      x.fillText('消息大小（对数轴）→', W / 2, H - 6);
      x.strokeStyle = '#58a6ff'; x.setLineDash([5, 5]); x.beginPath(); x.moveTo(padL, Y(195)); x.lineTo(W - padR, Y(195)); x.stroke(); x.setLineDash([]);
      x.fillStyle = '#58a6ff'; x.textAlign = 'left'; x.fillText('网卡直连基线 ≈195 GB/s', padL + 6, Y(195) - 6);
      const sig = (i: number, mid: number, k: number) => 1 / (1 + Math.exp(-(i - mid) / k));
      const bad: number[] = [], good: number[] = [];
      for (let i = 0; i < N; i++) { bad.push(i <= 19 ? 48 * sig(i, 12, 2.2) : 48 - (i - 19) * 2.4); good.push(195 * sig(i, 13, 2.4)); }
      const plot = (d: number[], col: string) => { x.strokeStyle = col; x.lineWidth = 2; x.beginPath(); d.forEach((v, i) => i ? x.lineTo(X(i), Y(v)) : x.moveTo(X(i), Y(v))); x.stroke(); };
      plot(good, '#3fb950'); plot(bad, '#f85149');
      x.fillStyle = '#f85149'; x.beginPath(); x.arc(X(19), Y(48), 4, 0, 7); x.fill();
      x.textAlign = 'center'; x.fillText('峰值 48 → 回落 34 ECMP 哈希不均', X(17), Y(48) - 14);
      x.textAlign = 'left'; x.fillStyle = '#f85149'; x.fillText('— 优化前（跨轨接线，交换机间链路拥堵）', padL + 6, padT + 8);
      x.fillStyle = '#3fb950'; x.fillText('— 轨道优化后（同轨 NIC 接同一交换机）', padL + 6, padT + 22);
    };
    draw();
    window.addEventListener('resize', draw);
    return () => window.removeEventListener('resize', draw);
  }, [dark]);
  return <canvas ref={ref} style={{ width: '100%', height: 260, display: 'block' }} />;
}

export function CommDeepDive({ dark }: { dark: boolean }) {
  const [view, setView] = useState<Comm>(0);
  const [scatter, setScatter] = useState(false);
  const [pxn, setPxn] = useState(false);
  const ring = ringSVG(scatter), px = pxnSVG(pxn);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
        {([[0, '① 环构建与跨域散射'], [1, '② PXN 轨道路径'], [2, '③ busbw 健康基线']] as [Comm, string][]).map(([v, l]) => (
          <button key={v} onClick={() => setView(v)} style={{ ...btn, ...tab(view === v) }}>{l}</button>
        ))}
      </div>

      {view === 0 && (
        <>
          <div style={{ display: 'flex', gap: 5 }}>
            <button onClick={() => setScatter(false)} style={{ ...btn, ...redBtn(!scatter) }}>SCATTER_XDC=0（无散射）</button>
            <button onClick={() => setScatter(true)} style={{ ...btn, ...grnBtn(scatter) }}>SCATTER_XDC=1（散射）</button>
          </div>
          <div style={panel}><svg viewBox="0 0 720 250" style={{ width: '100%', minWidth: 640, display: 'block' }} dangerouslySetInnerHTML={{ __html: ring.html }} /></div>
          <div style={{ ...desc, color: ring.noteRed ? '#f85149' : 'var(--tx2)' }}>{ring.note}</div>
        </>
      )}

      {view === 1 && (
        <>
          <div style={{ display: 'flex', gap: 5 }}>
            <button onClick={() => setPxn(false)} style={{ ...btn, ...redBtn(!pxn) }}>无 PXN（跨轨 3 跳）</button>
            <button onClick={() => setPxn(true)} style={{ ...btn, ...grnBtn(pxn) }}>PXN（换轨 · 避开 Spine）</button>
          </div>
          <div style={panel}><svg viewBox="0 0 720 300" style={{ width: '100%', minWidth: 640, display: 'block' }} dangerouslySetInnerHTML={{ __html: px.html }} /></div>
          <div style={{ ...desc, color: px.noteRed ? '#f85149' : 'var(--tx2)' }}>{px.note}</div>
        </>
      )}

      {view === 2 && (
        <>
          <div style={{ ...panel, padding: 12 }}><BusbwCanvas dark={dark} /></div>
          <div style={desc}><b style={{ color: '#f85149' }}>红线（优化前）</b>：256MB 处达峰 48 GB/s 后随消息增大<b>回落</b>至 34 —— 大象流在交换机间链路 ECMP 哈希不均的典型指纹。<b style={{ color: '#3fb950' }}>绿线（轨道优化后）</b>：单调爬升后在大消息区饱和成平台（≈195 GB/s ≈ 网卡直连基线）。判读铁律：看<b>峰值与平台</b>而非平均值。busbw = algbw × <span style={{ fontFamily: MONO }}>2(n-1)/n</span>（AllReduce）。</div>
        </>
      )}
    </div>
  );
}
