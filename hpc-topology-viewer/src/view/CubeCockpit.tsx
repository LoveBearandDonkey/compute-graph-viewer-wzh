/**
 * CubeCockpit — 「立方重排」工作区：内嵌 AI Infra 统一 3D 拓扑驾驶舱原型（v22）。
 *
 * 原型是一份自包含的 vanilla Three.js（r128）应用，功能完整（物理机房 3D 拓扑、L7→L0 层级
 * 下钻、场景副本/巡检/MoE 越界/慢 DP/单机深潜/部署审视、TP/PP/EP/DP 连线镜头、演练回放、
 * 5 种立方形态重排…）。为 100% 保留其全部功能，这里以 iframe 挂载静态资源
 * `public/cube-cockpit.html`，而非把 2600 行 Three.js 逐行改写成 React-Three-Fiber。
 *
 * 统一适配：
 *   · 明暗主题随宿主项目联动（初始 `?theme=`、切换时 postMessage）；
 *   · 工具栏桥接 —— 原型自带工具栏（着色透镜/连线镜头/剧本/演练/重置）已隐藏，改由宿主顶栏
 *     「Decode」控制面板驱动：宿主发 cockpit-cmd、原型回报 cockpit-state 供高亮。
 */
import { useEffect, useRef } from 'react';

const COCKPIT_URL = `${import.meta.env.BASE_URL}cube-cockpit.html`;

/** 宿主 → 原型 的工具栏指令。 */
export type CockpitCmd =
  | { cmd: 'lens'; value: string }
  | { cmd: 'wire'; value: string }
  | { cmd: 'script'; value: string }
  | { cmd: 'rel'; value: string }
  | { cmd: 'topn'; host: number }
  | { cmd: 'anom' }
  | { cmd: 'reset' }
  | { cmd: 'matrix' };

/** 原型 → 宿主 的工具栏状态（供宿主面板高亮）。 */
export interface CockpitState { lens: string; wire: string; anom: boolean; rel: string; topn: [number, string][]; }

export function CubeCockpit({ dark, onState, cmdApiRef }: {
  dark: boolean;
  onState?: (s: CockpitState) => void;
  cmdApiRef?: React.MutableRefObject<((c: CockpitCmd) => void) | null>;
}) {
  const frameRef = useRef<HTMLIFrameElement>(null);
  const theme = dark ? 'dark' : 'light';

  const post = (msg: unknown) => frameRef.current?.contentWindow?.postMessage(msg, '*');

  // 主题联动
  useEffect(() => { post({ type: 'cockpit-theme', theme }); }, [theme]);

  // 注册「宿主 → 原型」指令发送器，供 ClusterView 的 Decode 面板调用
  useEffect(() => {
    if (!cmdApiRef) return;
    cmdApiRef.current = (c: CockpitCmd) => post({ type: 'cockpit-cmd', ...c });
    return () => { if (cmdApiRef) cmdApiRef.current = null; };
  }, [cmdApiRef]);

  // 接收「原型 → 宿主」状态回报
  useEffect(() => {
    if (!onState) return;
    const onMsg = (e: MessageEvent) => {
      if (e.source !== frameRef.current?.contentWindow) return;
      const d = e.data;
      if (d && d.type === 'cockpit-state') onState({ lens: d.lens, wire: d.wire, anom: !!d.anom, rel: d.rel ?? 'AUTO', topn: d.topn ?? [] });
    };
    window.addEventListener('message', onMsg);
    return () => window.removeEventListener('message', onMsg);
  }, [onState]);

  return (
    <div style={{ position: 'absolute', inset: 0, zIndex: 11, background: 'var(--bg)' }}>
      <iframe
        ref={frameRef}
        title="立方重排 · AI Infra 统一驾驶舱"
        src={`${COCKPIT_URL}?theme=${theme}&v=${__BUILD_ID__}`}
        onLoad={() => post({ type: 'cockpit-theme', theme })}
        style={{ width: '100%', height: '100%', border: 'none', display: 'block', background: 'var(--bg)' }}
      />
    </div>
  );
}
