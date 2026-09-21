/**
 * WorkloadPanel — a DEEP, reusable real-data panel for the Pangu Pro MoE workload
 * (arXiv:2505.21411). Mirrors the rich panels StatusView shows, so the plane / 3-D
 * views carry the same real numbers (model shape · parallelism · collectives · kernel
 * & comm optimisation · measured tok/s · benchmark comparison · same-family references).
 *
 * Uses the shared CSS vars (--tx / --tx2 / --tx3 / --bd / --btn) already defined by the
 * host view roots, so it themes automatically. `phase` is optional: when given, the
 * throughput line + step decomposition are phase-specific; otherwise all phases are shown.
 */
import { WORKLOAD, WORKLOAD_DETAIL, WORKLOAD_REFS, BENCHMARKS, BENCH_MODELS, BENCH_PANGU_IDX, STEP_DECOMP } from '../scene/data';

const MONO = "'JetBrains Mono','Consolas',ui-monospace,monospace";
const ACCENT = '#4369ef';
type Phase = 'pretrain' | 'prefill' | 'decode';

const HDR: React.CSSProperties = { fontSize: 11, fontWeight: 600, color: 'var(--tx2)', margin: '14px 0 6px' };
const SUB: React.CSSProperties = { fontSize: 10, color: 'var(--tx3)', lineHeight: 1.55 };

export function WorkloadPanel({ phase, style }: { phase?: Phase; style?: React.CSSProperties }) {
  const P = WORKLOAD.perf;
  return (
    <div style={style}>
      {/* ── 工况模型 ── */}
      <div style={{ ...HDR, marginTop: 0 }}>工况模型 · {WORKLOAD.name} <span style={{ color: 'var(--tx3)', fontWeight: 400 }}>{WORKLOAD.short}</span></div>
      <div style={SUB}>
        {WORKLOAD.routedExperts}路由/{WORKLOAD.activatedExperts}激活/{WORKLOAD.sharedExperts}共享专家 · {WORKLOAD.layers}层 · hidden {WORKLOAD.hidden} · GQA {WORKLOAD.queryHeads}/{WORKLOAD.kvHeads}·head {WORKLOAD.headSize}
      </div>
      <div style={{ ...SUB, color: 'var(--tx2)', fontFamily: MONO, marginTop: 3 }}>
        {(!phase || phase === 'decode') && <div>Decode {P.decodeTokps} tok/s·卡 (batch{P.decodeBatch}·TPOT {P.decodeTPOTms}ms) → MTP {P.decodeMtpTokps}</div>}
        {(!phase || phase === 'prefill') && <div>Prefill {P.prefillTokps} tok/s·卡 (TTFT {P.prefillTTFTms}ms)</div>}
        {(!phase || phase === 'pretrain') && <div>预训练 {WORKLOAD.trainNpus} NPU · {WORKLOAD.trainTokens} tokens · MFU +{WORKLOAD.mfuGainPct}%</div>}
      </div>
      <div style={{ ...SUB, marginTop: 4 }}>
        内核：MulAttention {WORKLOAD_DETAIL.kernel.mulAttnSpeedup}× · 注意力占时延 {WORKLOAD_DETAIL.kernel.attnLatencyPct[0]}–{WORKLOAD_DETAIL.kernel.attnLatencyPct[1]}%（KV {WORKLOAD_DETAIL.kernel.kvOfAttnPct}%）· SwiftGMM &gt;{WORKLOAD_DETAIL.kernel.swiftGmmLatencyPct}%
      </div>
      <div style={SUB}>
        通信优化：AllReduce→RS+AG −{WORKLOAD_DETAIL.comm.allreduceCutPct}% · RMSNorm 重排 −{WORKLOAD_DETAIL.comm.rmsnormCutPct}% · 融合 {WORKLOAD_DETAIL.comm.fusedOps.join('/')}
      </div>
      <div style={{ ...SUB, marginTop: 2 }}>真实值 · Ascend 800I A2/300I Duo（arXiv:2505.21411）</div>

      {/* ── 并行策略 ── */}
      <div style={HDR}>并行策略（真实 · H²P）</div>
      <div style={{ ...SUB, color: 'var(--tx2)' }}>
        <div>训练 TP{WORKLOAD.train.tp}·EP{WORKLOAD.train.ep}·PP{WORKLOAD.train.pp}·VPP{WORKLOAD.train.vpp}·CP{WORKLOAD.train.cp}</div>
        <div>推理 注意力 DP{WORKLOAD.inferAttn.dp}+TP{WORKLOAD.inferAttn.tp} · 路由专家 TP{WORKLOAD.inferRouted.tp}+EP{WORKLOAD.inferRouted.ep} · 共享 TP{WORKLOAD.inferSharedTP}</div>
        <div style={SUB}>集合：EP 层级化 All-to-All · DP Ring-AllReduce · SP AllGather+ReduceScatter · PP P2P</div>
        <div style={SUB}>MoGE：{WORKLOAD_DETAIL.moge.note} · 负载不均 ↓&gt;{WORKLOAD_DETAIL.moge.imbalanceReductionPct}%</div>
      </div>

      {/* ── step 时间分解（phase 指定时）── */}
      {phase && (
        <>
          <div style={HDR}>step 时间分解（{phase === 'decode' ? 'Decode' : phase === 'prefill' ? 'Prefill' : '预训练'} · {WORKLOAD.short}）</div>
          {STEP_DECOMP[phase].map(({ label, frac, color }) => (
            <div key={label} style={{ marginBottom: 5 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, marginBottom: 2 }}><span style={{ color: 'var(--tx)' }}>{label}</span><span style={{ color: 'var(--tx3)', fontFamily: MONO }}>{Math.round(frac * 100)}%</span></div>
              <div style={{ height: 7, borderRadius: 4, background: 'var(--btn)', overflow: 'hidden' }}><div style={{ height: '100%', width: `${frac * 100}%`, background: color }} /></div>
            </div>
          ))}
        </>
      )}

      {/* ── 模型质量对照 ── */}
      <div style={HDR}>模型质量对照 · {WORKLOAD.name} <span style={{ color: 'var(--tx3)', fontWeight: 400 }}>vs 27–32B</span></div>
      {BENCHMARKS.map((b) => {
        const mine = b.scores[BENCH_PANGU_IDX];
        const bestOther = Math.max(...b.scores.filter((_, i) => i !== BENCH_PANGU_IDX));
        const lead = mine >= bestOther;
        const scale = (s: number) => Math.max(0, Math.min(1, (s - 40) / 55));
        return (
          <div key={b.name} style={{ marginBottom: 5 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, marginBottom: 2 }}>
              <span style={{ color: 'var(--tx)' }}>{b.name}</span>
              <span style={{ color: 'var(--tx3)', fontFamily: MONO }}>
                <span style={{ color: lead ? '#04d793' : 'var(--tx)', fontWeight: 700 }}>{mine.toFixed(1)}</span>
                <span> · 次优 {bestOther.toFixed(1)} {lead ? `(+${(mine - bestOther).toFixed(1)})` : `(${(mine - bestOther).toFixed(1)})`}</span>
              </span>
            </div>
            <div style={{ position: 'relative', height: 8, borderRadius: 4, background: 'var(--btn)', overflow: 'hidden' }}>
              <div style={{ position: 'absolute', left: 0, top: 0, height: '100%', width: `${scale(mine) * 100}%`, background: lead ? ACCENT : '#9aa3b2', borderRadius: 4 }} />
              <div title={`最优对手 ${bestOther.toFixed(1)}`} style={{ position: 'absolute', left: `${scale(bestOther) * 100}%`, top: -1, height: 10, width: 2, background: 'var(--tx)', opacity: 0.65 }} />
            </div>
          </div>
        );
      })}
      <div style={SUB}>蓝条=盘古得分 · 竖线=最优对手（{BENCH_MODELS.length - 1} 个 27–32B）· EM/F1/Pass@1 · arXiv:2505.21411 T3</div>

      {/* ── 同类对照 ── */}
      <div style={HDR}>同类对照 · 昇腾超节点真实论文</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {WORKLOAD_REFS.map((r) => (
          <div key={r.id} style={{ fontSize: 10, lineHeight: 1.45, paddingLeft: 7, borderLeft: `2px solid ${r.id === 'pangu-pro-moe' ? ACCENT : 'var(--bd2, var(--bd))'}` }}>
            <div style={{ color: 'var(--tx)', fontWeight: 600 }}>{r.title} <span style={{ color: 'var(--tx3)', fontWeight: 400, fontFamily: MONO }}>arXiv:{r.arxiv}</span></div>
            <div style={{ color: 'var(--tx3)' }}>{r.scale}</div>
            <div style={{ color: 'var(--tx2)' }}>{r.metric}</div>
          </div>
        ))}
      </div>
      <div style={{ ...SUB, marginTop: 6 }}>注：对照均为昇腾超节点真实论文（非本视图 950 硬件平台），仅作规模/性能参照。</div>
    </div>
  );
}
