// ─────────────────────────────────────────────────────────────────────────────
// NetCutView — 「整网切分」：整网图（左，逻辑计算图）× 立方重组（右，物理卡阵）。
// 一句话故事：整网（切什么）── 按并行切 ──▶ 立方（切到哪张卡）。
//   · 左：WholeNetGraph（复用 model-graphviz pattern）—— Model→…→Operator 分层 DAG，
//         带训练前向/反向方向箭头（青=激活下行 / 蓝=梯度上行，用箭头方向区分）。
//   · 右：CubeView（立方重组）—— 卡阵按 TP/PP/DP/EP 重排/染色。
//   · 联动：共享一个「切分维」。点图上的算子 → 立方按该维染色显形 + 图上同维算子高亮；
//         顶栏切「切分」维 → 双侧同时响应。这就是「整网切分」的两半。
// ─────────────────────────────────────────────────────────────────────────────
import { useState } from 'react';
import { CubeView } from './CubeView';
import { WholeNetGraph, type FlowDir } from './WholeNetGraph';
import { type ParCutDim } from '../vendor/model-graphviz/graph-meta';
import { PARALLEL_COLORS, type Gen, type PartitionDim } from '../scene/data';

const FWD_COLOR = '#22d3ee', BWD_COLOR = '#6b8bff';

const btn = (on: boolean): React.CSSProperties => ({
  border: `1px solid ${on ? '#4369ef' : 'var(--bd)'}`,
  background: on ? '#4369ef' : 'var(--btn)',
  color: on ? '#fff' : 'var(--tx2)',
  borderRadius: 7, padding: '3px 10px', fontSize: 11, fontWeight: on ? 600 : 500, cursor: 'pointer',
});
const dimBtn = (on: boolean, c: string): React.CSSProperties => ({
  border: `1px solid ${on ? c : 'var(--bd)'}`,
  background: on ? c : 'var(--btn)',
  color: on ? '#fff' : 'var(--tx2)',
  borderRadius: 7, padding: '3px 10px', fontSize: 11, fontWeight: on ? 600 : 500, cursor: 'pointer',
  display: 'inline-flex', alignItems: 'center', gap: 5,
});
const LBL: React.CSSProperties = { fontSize: 9.5, fontWeight: 600, letterSpacing: 0.4, textTransform: 'uppercase', color: 'var(--tx3)' };

const DIM_LABEL: Record<ParCutDim, string> = { tp: 'TP 张量', pp: 'PP 流水', dp: 'DP 数据', ep: 'EP 专家' };

export function NetCutView({ gen, dark }: { gen: Gen; dark: boolean }) {
  const [strat, setStrat] = useState<PartitionDim>('none');   // 共享切分维（联动双侧）
  const [dir, setDir] = useState<FlowDir>('both');            // 前向/反向/全链
  const [selNode, setSelNode] = useState<string | null>(null);
  const [cubeSel, setCubeSel] = useState<number | null>(null);

  const highlightDim: ParCutDim | null = strat === 'none' ? null : strat;

  return (
    <div data-theme={dark ? 'dark' : 'light'} style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', background: 'var(--bg)', color: 'var(--tx)' }}>
      {/* ── 顶栏：方向 + 切分 + 图例 ── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', padding: '7px 14px', borderBottom: '1px solid var(--bd)', background: 'var(--panel-solid)' }}>
        <span style={{ fontSize: 12.5, fontWeight: 700 }}>整网切分</span>
        <span style={{ fontSize: 10, color: 'var(--tx3)' }}>整网图（切什么）· 按并行切 ▸ 立方（切到哪张卡）</span>

        <span style={{ ...LBL, marginLeft: 6 }}>方向</span>
        <div style={{ display: 'flex', gap: 4 }}>
          {([['fwd', '前向'], ['bwd', '反向'], ['both', '全链']] as [FlowDir, string][]).map(([d, l]) => (
            <button key={d} onClick={() => setDir(d)} style={btn(dir === d)}>{l}</button>
          ))}
        </div>

        <span style={LBL}>切分</span>
        <div style={{ display: 'flex', gap: 4 }}>
          <button onClick={() => { setStrat('none'); setSelNode(null); }} style={btn(strat === 'none')}>无</button>
          {(['tp', 'pp', 'dp', 'ep'] as ParCutDim[]).map((d) => (
            <button key={d} onClick={() => setStrat(d)} title={DIM_LABEL[d]} style={dimBtn(strat === d, PARALLEL_COLORS[d])}>
              <span style={{ width: 8, height: 8, borderRadius: 2, background: strat === d ? '#fff' : PARALLEL_COLORS[d] }} />{d.toUpperCase()}
            </button>
          ))}
        </div>

        {/* 前向/反向图例 */}
        <div style={{ display: 'flex', gap: 12, marginLeft: 'auto', fontSize: 10, color: 'var(--tx2)' }}>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}><Arrow color={FWD_COLOR} /> 前向 · 激活下行</span>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}><Arrow color={BWD_COLOR} up /> 反向 · 梯度上行</span>
        </div>
      </div>

      {/* ── 主体：整网图（左） × 立方重组（右） ── */}
      <div style={{ flex: 1, display: 'flex', minHeight: 0 }}>
        <div style={{ flex: '0 0 44%', minWidth: 320, display: 'flex', flexDirection: 'column', borderRight: '1px solid var(--bd)', minHeight: 0 }}>
          <div style={{ padding: '5px 12px', fontSize: 10, color: 'var(--tx3)', borderBottom: '1px solid var(--bd)', flexShrink: 0 }}>
            整网图 · openPangu-2.0-Flash — 点算子看它落在哪种并行切分{selNode ? ` · 选中 ${selNode}` : ''}
          </div>
          <div style={{ flex: 1, minHeight: 0 }}>
            <WholeNetGraph
              dark={dark}
              direction={dir}
              highlightDim={highlightDim}
              selectedNodeId={selNode}
              onSelectNode={(id, d) => { setSelNode(id); if (d) setStrat(d); }}
            />
          </div>
        </div>

        <div style={{ flex: 1, position: 'relative', minWidth: 0, minHeight: 0 }}>
          <CubeView
            gen={gen}
            dark={dark}
            stratColor={strat}
            sel={cubeSel}
            onSelectRank={setCubeSel}
          />
        </div>
      </div>
    </div>
  );
}

function Arrow({ color, up = false }: { color: string; up?: boolean }) {
  return (
    <svg width="22" height="10" viewBox="0 0 22 10" aria-hidden>
      <line x1={up ? 20 : 2} y1="5" x2={up ? 2 : 20} y2="5" stroke={color} strokeWidth="2" />
      <path d={up ? 'M2 5 L7 2 L7 8 Z' : 'M20 5 L15 2 L15 8 Z'} fill={color} />
    </svg>
  );
}
