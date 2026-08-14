// flash_mla_decode.cpp · AscendC 核  (AscendPort · S6 双缓冲流水已编排)
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t DIM     = 512;
constexpr int32_t PE_DIM  = 64;
constexpr int32_t BLOCK_N = 128;
constexpr int32_t DEPTH   = 2;              // ← 双缓冲深度

class FlashMLADecode {
public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR qPe, GM_ADDR kv,
                                GM_ADDR kPe, GM_ADDR out,
                                int32_t B, int32_t numHeads,
                                int32_t seqlenKv, float softmaxScale, int32_t nTile) {
        this->batchIdx = GetBlockIdx() / numHeads;
        this->headIdx  = GetBlockIdx() % numHeads;
        this->B = B; this->numHeads = numHeads; this->seqlenKv = seqlenKv;
        this->softmaxScale = softmaxScale; this->nTile = nTile;
        qGm.SetGlobalBuffer((__gm__ half*)q);
        qPeGm.SetGlobalBuffer((__gm__ half*)qPe);
        kvGm.SetGlobalBuffer((__gm__ half*)kv);
        kPeGm.SetGlobalBuffer((__gm__ half*)kPe);
        outGm.SetGlobalBuffer((__gm__ half*)out);
        pipe.InitBuffer(qL1,  1,     (DIM + PE_DIM) * sizeof(half));
        pipe.InitBuffer(kL1,  DEPTH, BLOCK_N * (DIM + PE_DIM) * sizeof(half));  // 深度=2 双缓冲
        pipe.InitBuffer(cO,   DEPTH, BLOCK_N * sizeof(float));
        pipe.InitBuffer(ubQK, DEPTH, BLOCK_N * sizeof(float));
        pipe.InitBuffer(ubOut,1,     DIM * sizeof(float));
    }
    __aicore__ inline void Process() {
        if (batchIdx >= B || headIdx >= numHeads) return;
        LocalTensor<half> qLoc = qL1.AllocTensor<half>();
        DataCopy(qLoc,        qGm[(batchIdx * numHeads + headIdx) * DIM], DIM);
        DataCopy(qLoc[DIM], qPeGm[(batchIdx * numHeads + headIdx) * PE_DIM], PE_DIM);
        qL1.EnQue(qLoc);
        LocalTensor<half> q = qL1.DeQue<half>();

        LocalTensor<float> outAcc = ubOut.Get<float>();
        SetValue(outAcc, DIM, 0.f);
        float mPrev = -CUDART_INF_F, lPrev = 0.f;

        // ---- 软件流水:预取 n+1  ∥  矩阵/向量计算 n  ∥  P·V 累加 ----
        CopyInKV(0);                                        // 预热:载入第 0 块
        for (int32_t tile = 0; tile < nTile; ++tile) {
            if (tile + 1 < nTile) CopyInKV(tile + 1);       // 预取下一块(与计算重叠)
            ComputeTile(q, tile, outAcc, mPrev, lPrev);     // 矩阵 QKᵀ → 向量在线 Softmax
        }
        // 归一化并写回
        Div(outAcc, outAcc, lPrev, DIM);
        DataCopy(outGm[(batchIdx * numHeads + headIdx) * DIM], outAcc, DIM);
        qL1.FreeTensor(q);
    }
private:
    __aicore__ inline void CopyInKV(int32_t tile) {
        int32_t kvStart  = tile * BLOCK_N;
        int32_t tileSize = min(BLOCK_N, seqlenKv - kvStart);
        // KV 分块 (K 非位置部分 + K_pe) 一并载入
        LocalTensor<half> kLoc = kL1.AllocTensor<half>();
        DataCopy(kLoc,      kvGm[(batchIdx * seqlenKv + kvStart) * DIM], tileSize * DIM);
        DataCopy(kLoc[tileSize * DIM], kPeGm[(batchIdx * seqlenKv + kvStart) * PE_DIM], tileSize * PE_DIM);
        kL1.EnQue(kLoc);                                    // 入队 → 与 Compute 并行
    }
    __aicore__ inline void ComputeTile(LocalTensor<half>& q, int32_t tile,
                                       LocalTensor<float>& outAcc, float& mPrev, float& lPrev) {
        int32_t kvStart  = tile * BLOCK_N;
        int32_t tileSize = min(BLOCK_N, seqlenKv - kvStart);

        LocalTensor<half> k = kL1.DeQue<half>();            // 取上一轮预取的块
        LocalTensor<float> logits = cO.AllocTensor<float>();
        Mmad(logits, q, k, {1, tileSize, DIM + PE_DIM});    // 矩阵单元: QKᵀ + PEᵀ
        Muls(logits, logits, softmaxScale, tileSize);
        cO.EnQue(logits);
        LocalTensor<float> lg = cO.DeQue<float>();

        LocalTensor<float> qkScores = ubQK.AllocTensor<float>();
        DataCopy(qkScores, lg, tileSize);
        // use_swizzle / GemmWarpPolicy 在昇腾无对应物 → 分核 + 向量单元片上归约
        float mCurr = ReduceMax(qkScores, tileSize);        // 向量单元规约 reduce_max
        float mNew  = fmaxf(mPrev, mCurr);
        float alpha = expf(mPrev - mNew);
        Muls(outAcc, outAcc, alpha, DIM);

        Subs(qkScores, qkScores, mNew, tileSize);
        Exp(qkScores, qkScores, tileSize);                  // 自然底 exp (非 exp2)
        float localSum = ReduceSum(qkScores, tileSize);     // 向量单元规约 reduce_sum
        float lNew = lPrev * alpha + localSum;

        for (int32_t j = 0; j < tileSize; ++j) {
            float weight = qkScores[j];
            Axpy(outAcc, k[j * (DIM + PE_DIM)], weight, DIM);// P·V 累加
        }
        ubQK.EnQue(qkScores);

        mPrev = mNew; lPrev = lNew;
        kL1.FreeTensor(k); cO.FreeTensor(lg);
    }

    TPipe pipe;
    TQue<TPosition::A1, 1>        qL1;
    TQue<TPosition::B1, DEPTH>    kL1;      // ← 双缓冲
    TQue<TPosition::CO1, DEPTH>   cO;
    TQue<TPosition::VECOUT,DEPTH> ubQK;
    TBuf<TPosition::VECCALC>      ubOut;
    GlobalTensor<half> qGm, qPeGm, kvGm, kPeGm, outGm;
    int32_t batchIdx, headIdx, B, numHeads, seqlenKv, nTile;
    float softmaxScale;
};
