/*
  样例 kernel 源码 —— 供「源码联动」(规划文档 §4.3-8) 与生命周期视图的代码片段使用。

  数据等级：L2 / 构造样例
    MatmulLayerNorm_mix 是为演示构造的 mix 算子（Cube + Vector 混合），
    覆盖 L1 / L0A / L0B / L0C / UB 全部片上层级，不对应任何真实产品算子。
*/
(function registerMemVizKernelSource(global) {
  'use strict';

  const TEXT = `#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t M_TOTAL = 200;   // 总行数
constexpr int32_t N       = 256;   // 输出列数
constexpr int32_t K       = 512;   // 归约维
constexpr int32_t K0      = 128;   // L0 分形块

class MatmulLayerNormMixKernel {
public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR gamma,
                              GM_ADDR beta, GM_ADDR y, MLNTiling tiling) {
    tileM_   = tiling.tileM;
    tileNum_ = tiling.tileNum;
    xGm.SetGlobalBuffer((__gm__ half*)x);
    wGm.SetGlobalBuffer((__gm__ half*)w);
    gammaGm.SetGlobalBuffer((__gm__ float*)gamma);
    betaGm.SetGlobalBuffer((__gm__ float*)beta);
    yGm.SetGlobalBuffer((__gm__ half*)y);

    // ---- L1 / L0：Cube 侧 ----
    pipe.InitBuffer(bL1Buf,  1,        K * N * sizeof(half));
    pipe.InitBuffer(aL1Que,  A_L1_DB,  tileM_ * K * sizeof(half));
    pipe.InitBuffer(aL0AQue, A_L0A_DB, tileM_ * K0 * sizeof(half));
    pipe.InitBuffer(bL0BBuf, 1,        K0 * N * sizeof(half));
    pipe.InitBuffer(cL0CQue, C_L0C_DB, tileM_ * N * sizeof(float));

    // ---- UB：Vector 侧 ----
    pipe.InitBuffer(gammaBuf, 1, N * sizeof(float));
    pipe.InitBuffer(betaBuf,  1, N * sizeof(float));
    pipe.InitBuffer(meanBuf,  1, tileM_ * sizeof(float));  // 每行按 32B block 对齐
    pipe.InitBuffer(rstdBuf,  1, tileM_ * sizeof(float));
    pipe.InitBuffer(mmOutQue, MM_OUT_DB, tileM_ * N * sizeof(float));
    pipe.InitBuffer(tmpSqBuf, 1,         tileM_ * N * sizeof(float));
    pipe.InitBuffer(normBuf,  1,         tileM_ * N * sizeof(float));
    pipe.InitBuffer(yQue,     Y_DB,      tileM_ * N * sizeof(half));
  }

  __aicore__ inline void Process() {
    LoadWeightOnce();
    for (int32_t i = 0; i < tileNum_; ++i) {
      CopyInA(i);
      LoadL0A(i);
      MatMulTile(i);
      FixpipeToUb(i);
      ReduceMeanVar(i);
      Normalize(i);
      CastAndCopyOut(i);
    }
  }

private:
  __aicore__ inline void LoadWeightOnce() {
    LocalTensor<half> bL1 = bL1Buf.Get<half>();
    DataCopy(bL1, wGm, K * N);                        // MTE2：权重常驻 L1
    LocalTensor<half> bL0B = bL0BBuf.Get<half>();
    LoadData(bL0B, bL1, {0, K0, N, 0});               // MTE1：L1 -> L0B
    LocalTensor<float> g = gammaBuf.Get<float>();
    LocalTensor<float> b = betaBuf.Get<float>();
    DataCopy(g, gammaGm, N);
    DataCopy(b, betaGm, N);
  }

  __aicore__ inline void CopyInA(int32_t i) {
    LocalTensor<half> aL1 = aL1Que.AllocTensor<half>();
    DataCopy(aL1, xGm[i * tileM_ * K], tileM_ * K);   // MTE2
    aL1Que.EnQue(aL1);
  }

  __aicore__ inline void LoadL0A(int32_t i) {
    LocalTensor<half> aL1  = aL1Que.DeQue<half>();
    LocalTensor<half> aL0A = aL0AQue.AllocTensor<half>();
    LoadData(aL0A, aL1, {0, tileM_, K0, 0});          // MTE1
    aL0AQue.EnQue(aL0A);
    aL1Que.FreeTensor(aL1);
  }

  __aicore__ inline void MatMulTile(int32_t i) {
    LocalTensor<half>  aL0A = aL0AQue.DeQue<half>();
    LocalTensor<half>  bL0B = bL0BBuf.Get<half>();
    LocalTensor<float> cL0C = cL0CQue.AllocTensor<float>();
    Mmad(cL0C, aL0A, bL0B, {tileM_, K0, N, true});    // Cube
    cL0CQue.EnQue(cL0C);
    aL0AQue.FreeTensor(aL0A);
  }

  __aicore__ inline void FixpipeToUb(int32_t i) {
    LocalTensor<float> cL0C  = cL0CQue.DeQue<float>();
    LocalTensor<float> mmOut = mmOutQue.AllocTensor<float>();
    Fixpipe(mmOut, cL0C, {tileM_, N});                // FixPipe：L0C -> UB
    mmOutQue.EnQue(mmOut);
    cL0CQue.FreeTensor(cL0C);
  }

  __aicore__ inline void ReduceMeanVar(int32_t i) {
    LocalTensor<float> mmOut = mmOutQue.DeQue<float>();
    LocalTensor<float> tmpSq = tmpSqBuf.Get<float>();
    LocalTensor<float> mean  = meanBuf.Get<float>();
    LocalTensor<float> rstd  = rstdBuf.Get<float>();
    Mul(tmpSq, mmOut, mmOut, tileM_ * N);             // Vector
    ReduceSum(mean, mmOut, tileM_, N);
    ReduceSum(rstd, tmpSq, tileM_, N);
    mmOutQue.EnQue(mmOut);
  }

  __aicore__ inline void Normalize(int32_t i) {
    LocalTensor<float> mmOut = mmOutQue.DeQue<float>();
    LocalTensor<float> norm  = normBuf.Get<float>();
    LocalTensor<float> g     = gammaBuf.Get<float>();
    LocalTensor<float> b     = betaBuf.Get<float>();
    Sub(norm, mmOut, mean, tileM_ * N);               // Vector
    Mul(norm, norm, rstd, tileM_ * N);
    Mul(norm, norm, g,    tileM_ * N);
    Add(norm, norm, b,    tileM_ * N);
    mmOutQue.FreeTensor(mmOut);
  }

  __aicore__ inline void CastAndCopyOut(int32_t i) {
    LocalTensor<half>  y    = yQue.AllocTensor<half>();
    LocalTensor<float> norm = normBuf.Get<float>();
    Cast(y, norm, RoundMode::CAST_RINT, tileM_ * N);  // Vector
    yQue.EnQue(y);
    LocalTensor<half> yOut = yQue.DeQue<half>();
    DataCopy(yGm[i * tileM_ * N], yOut, tileM_ * N);  // MTE3
    yQue.FreeTensor(yOut);
  }

  TPipe pipe;
  TQue<QuePosition::A1,  A_L1_DB>      aL1Que;
  TQue<QuePosition::A2,  A_L0A_DB>     aL0AQue;
  TQue<QuePosition::CO1, C_L0C_DB>     cL0CQue;
  TQue<QuePosition::VECIN,  MM_OUT_DB> mmOutQue;
  TQue<QuePosition::VECOUT, Y_DB>      yQue;
  TBuf<QuePosition::B1>      bL1Buf;
  TBuf<QuePosition::B2>      bL0BBuf;
  TBuf<QuePosition::VECCALC> gammaBuf, betaBuf, meanBuf, rstdBuf, tmpSqBuf, normBuf;
  GlobalTensor<half>  xGm, wGm, yGm;
  GlobalTensor<float> gammaGm, betaGm;
  int32_t tileM_, tileNum_;
};`;

  const LINES = TEXT.split('\n');

  /** 按子串定位行号（0 基，与 memory-reuse-viewer 的 srcLine* 约定一致）。 */
  function lineOf(needle, from = 0) {
    for (let i = from; i < LINES.length; i += 1) {
      if (LINES[i].includes(needle)) return i;
    }
    return 0;
  }

  /** 取 [start, end] 闭区间的源码文本，用于详情面板代码块。 */
  function snippet(start, end) {
    return LINES.slice(start, end + 1).map((line) => line.replace(/^ {4}/, '')).join('\n');
  }

  global.MemVizKernelSource = {
    path: 'matmul_layernorm_mix.cpp',
    language: 'cpp',
    text: TEXT,
    lines: LINES,
    lineOf,
    snippet,
  };
})(window);
