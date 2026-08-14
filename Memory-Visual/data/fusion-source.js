/*
  融合算子样例源码 —— 服务场景 6（workspace 与 GM 规划）
  ------------------------------------------------------------------
  两份源码：
    · mla_block_tiling.cpp —— host 侧 tiling，workspace 上报值就写在这里。
      场景 6 的所有结论最终要指回 `ws[0] = ...` 那一行，否则开发者拿到一个
      数字不知道去哪儿改（见方案设计 §3 的 reportedAt）。
    · mla_block_fused.cpp  —— kernel 侧，六个子计算按拓扑序串行，
      每个中间量在 workspace 上各切一段。

  数据等级：L2 / 构造算子，不对应任何真实产品算子。
*/
(function registerMemVizFusionSource(global) {
  'use strict';

  const TILING_TEXT = `// mla_block_tiling.cpp —— MLABlock_fused 的 host 侧 tiling 与 workspace 上报
#include "mla_block_tiling.h"

namespace optiling {

constexpr uint32_t kTokens   = 1024;   // M：本次 shape 的 token 数
constexpr uint32_t kHidden   = 1024;   // H
constexpr uint32_t kFfnInner = 2816;   // SwiGLU 中间维
constexpr uint32_t kHalf     = 2;      // sizeof(half)
constexpr uint32_t kFloat    = 4;      // sizeof(float)
constexpr uint32_t kGmAlign  = 512;    // GM 分配对齐粒度

static inline uint32_t AlignUp(uint32_t v, uint32_t a) {
  return (v + a - 1) / a * a;
}

ge::graphStatus TilingMLABlockFused(gert::TilingContext* context) {
  MLABlockTilingData tiling;
  tiling.set_tokens(kTokens);
  tiling.set_hidden(kHidden);
  tiling.set_ffnInner(kFfnInner);
  context->SetBlockDim(8);

  // 每个中间量的物理大小
  const uint32_t hidden = AlignUp(kTokens * kHidden * kHalf, kGmAlign);    // 2048KB
  const uint32_t lse    = AlignUp(kTokens * kFloat, kGmAlign);             //    4KB
  const uint32_t ffn    = AlignUp(kTokens * kFfnInner * kHalf, kGmAlign);  // 5632KB

  // 融合体里每个子计算各申请各的一段，谁也不复用谁：
  //   wsQ wsK wsV | wsQr wsKr | wsAttn wsLse | wsProj | wsNorm | wsFfn
  size_t* ws = context->GetWorkspaceSizes(1);
  ws[0] = hidden * 8 + lse + ffn;   // ← 场景 6 的「当前值」就是这一行

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

}  // namespace optiling`;

  const KERNEL_TEXT = `// mla_block_fused.cpp —— MLABlock_fused：QKVProj → RoPE → FlashAttn → OutProj → Add+LN → FFN
#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t TOKENS = 1024;
constexpr uint32_t HIDDEN = 1024;
constexpr uint32_t FFN_INNER = 2816;

class KernelMLABlockFused {
public:
  __aicore__ inline KernelMLABlockFused() {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR wqkv, GM_ADDR wo, GM_ADDR wffn,
                              GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                              GM_ADDR workspace, MLABlockTiling tiling) {
    xGm.SetGlobalBuffer((__gm__ half*)x, TOKENS * HIDDEN);
    yGm.SetGlobalBuffer((__gm__ half*)y, TOKENS * HIDDEN);

    // workspace 顺序切分：每个中间量一段，互不重叠、也互不复用
    __gm__ half* ws = (__gm__ half*)workspace;
    wsQ.SetGlobalBuffer(ws + kOffQ,       TOKENS * HIDDEN);
    wsK.SetGlobalBuffer(ws + kOffK,       TOKENS * HIDDEN);
    wsV.SetGlobalBuffer(ws + kOffV,       TOKENS * HIDDEN);
    wsQr.SetGlobalBuffer(ws + kOffQr,     TOKENS * HIDDEN);
    wsKr.SetGlobalBuffer(ws + kOffKr,     TOKENS * HIDDEN);
    wsAttn.SetGlobalBuffer(ws + kOffAttn, TOKENS * HIDDEN);
    wsLse.SetGlobalBuffer((__gm__ float*)(ws + kOffLse), TOKENS);
    wsProj.SetGlobalBuffer(ws + kOffProj, TOKENS * HIDDEN);
    wsNorm.SetGlobalBuffer(ws + kOffNorm, TOKENS * HIDDEN);
    wsFfn.SetGlobalBuffer(ws + kOffFfn,   TOKENS * FFN_INNER);
  }

  __aicore__ inline void Process() {
    QKVProj();       // sg0: xGm            -> wsQ / wsK / wsV
    RoPE();          // sg1: wsQ, wsK       -> wsQr / wsKr
    FlashAttn();     // sg2: wsQr,wsKr,wsV  -> wsAttn (+ wsLse 统计量)
    OutProj();       // sg3: wsAttn         -> wsProj
    AddLayerNorm();  // sg4: wsProj, xGm    -> wsNorm
    FeedForward();   // sg5: wsNorm         -> wsFfn -> yGm
  }

private:
  // sg0 —— QKV 投影：一次 matmul 出三份，之后 wsQ/wsK 只被 RoPE 读
  __aicore__ inline void QKVProj() {
    MatmulQKV(xGm, wqkvGm, wsQ, wsK, wsV);
  }

  // sg1 —— 旋转位置编码：逐元素，输入输出形状一致（原地候选见 ws-inplace）
  __aicore__ inline void RoPE() {
    RotaryEmbed(wsQ, wsQr);
    RotaryEmbed(wsK, wsKr);
  }

  // sg2 —— FlashAttention：分块累加，wsLse 是每行的 logsumexp 统计量
  __aicore__ inline void FlashAttn() {
    FlashAttentionV2(wsQr, wsKr, wsV, wsAttn, wsLse);
  }

  // sg3 —— 输出投影
  __aicore__ inline void OutProj() {
    MatmulOut(wsAttn, woGm, wsProj);
  }

  // sg4 —— 残差 + LayerNorm，wsLse 在此参与缩放校正
  __aicore__ inline void AddLayerNorm() {
    AddLayerNormCustom(wsProj, xGm, gammaGm, betaGm, wsLse, wsNorm);
  }

  // sg5 —— FFN：升维到 FFN_INNER、SwiGLU、再降回 HIDDEN 写 yGm
  __aicore__ inline void FeedForward() {
    FfnSwiGlu(wsNorm, wffnGm, wsFfn, yGm, wsLse);
  }

  GlobalTensor<half>  xGm, yGm, wqkvGm, woGm, wffnGm, gammaGm, betaGm;
  GlobalTensor<half>  wsQ, wsK, wsV, wsQr, wsKr, wsAttn, wsProj, wsNorm, wsFfn;
  GlobalTensor<float> wsLse;
};`;

  function makeSource(path, text) {
    const lines = text.split('\n');
    return {
      path,
      language: 'cpp',
      text,
      lines,
      /** 按子串定位行号（0 基，与 memory-reuse-viewer 的 srcLine* 约定一致）。 */
      lineOf(needle, from = 0) {
        for (let i = from; i < lines.length; i += 1) {
          if (lines[i].includes(needle)) return i;
        }
        return 0;
      },
      snippet(start, end) {
        return lines.slice(Math.max(0, start), end + 1).join('\n');
      },
    };
  }

  const kernel = makeSource('mla_block_fused.cpp', KERNEL_TEXT);
  const tiling = makeSource('mla_block_tiling.cpp', TILING_TEXT);

  global.MemVizFusionSource = { kernel, tiling, files: [kernel, tiling] };
})(window);
