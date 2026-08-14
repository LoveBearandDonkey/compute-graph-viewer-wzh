// flash_mla_decode.cpp · AscendC 核  (AscendPort · S4 内存层次已注入)
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t DIM     = 512;
constexpr int32_t PE_DIM  = 64;
constexpr int32_t BLOCK_N = 128;

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
        // === 片上缓冲层次(S4 注入)===
        pipe.InitBuffer(qL1,  1, (DIM + PE_DIM) * sizeof(half));         // Q|Q_pe: GM→L1→L0A
        pipe.InitBuffer(kL1,  1, BLOCK_N * (DIM + PE_DIM) * sizeof(half));// KV|K_pe: GM→L1→L0B
        pipe.InitBuffer(vL1,  1, BLOCK_N * DIM * sizeof(half));          // V(=KV): GM→L1
        pipe.InitBuffer(cO,   1, BLOCK_N * sizeof(float));               // QKᵀ logits: L0C
        pipe.InitBuffer(ubQK, 1, BLOCK_N * sizeof(float));              // 在线 Softmax 中间: UB
        pipe.InitBuffer(ubOut,1, DIM * sizeof(float));                  // 输出累加: UB
    }
    __aicore__ inline void Process() {
        if (batchIdx >= B || headIdx >= numHeads) return;
        // 加载 Q 与 Q_pe (拼接为 [DIM+PE_DIM])
        LocalTensor<half> qLoc = qL1.AllocTensor<half>();
        DataCopy(qLoc,        qGm[(batchIdx * numHeads + headIdx) * DIM], DIM);
        DataCopy(qLoc[DIM], qPeGm[(batchIdx * numHeads + headIdx) * PE_DIM], PE_DIM);
        qL1.EnQue(qLoc);
        LocalTensor<half> q = qL1.DeQue<half>();

        LocalTensor<float> outAcc = ubOut.Get<float>();
        SetValue(outAcc, DIM, 0.f);                                 // 初始化输出累加器 acc_o
        float mPrev = -CUDART_INF_F, lPrev = 0.f;                   // 在线 Softmax 统计量

        // 沿 KV 序列分块遍历 (dense, 全序列)
        for (int32_t tile = 0; tile < nTile; ++tile) {
            ComputeTile(q, tile, outAcc, mPrev, lPrev);
        }
        // 归一化并写回
        Div(outAcc, outAcc, lPrev, DIM);                           // 向量单元: acc_o /= logsum
        DataCopy(outGm[(batchIdx * numHeads + headIdx) * DIM], outAcc, DIM);
        qL1.FreeTensor(q);
    }
private:
    __aicore__ inline void ComputeTile(LocalTensor<half>& q, int32_t tile,
                                       LocalTensor<float>& outAcc, float& mPrev, float& lPrev) {
        int32_t kvStart  = tile * BLOCK_N;
        int32_t tileSize = min(BLOCK_N, seqlenKv - kvStart);

        // 加载 KV 分块 (K 的非位置部分 + K_pe),GM→L1
        LocalTensor<half> kLoc = kL1.AllocTensor<half>();
        DataCopy(kLoc,      kvGm[(batchIdx * seqlenKv + kvStart) * DIM], tileSize * DIM);
        DataCopy(kLoc[tileSize * DIM], kPeGm[(batchIdx * seqlenKv + kvStart) * PE_DIM], tileSize * PE_DIM);
        kL1.EnQue(kLoc);
        LocalTensor<half> k = kL1.DeQue<half>();

        // 矩阵单元: QKᵀ = Q·KVᵀ + Q_pe·K_peᵀ (两段累加)
        LocalTensor<float> logits = cO.AllocTensor<float>();
        Mmad(logits, q, k, {1, tileSize, DIM + PE_DIM});           // [1, tileSize] logits → L0C
        Muls(logits, logits, softmaxScale, tileSize);              // logits *= softmax_scale
        cO.EnQue(logits);
        LocalTensor<float> lg = cO.DeQue<float>();

        // 在线 Softmax: L0C 无直连 UB → 经 GM/UB 中转,再向量单元规约
        LocalTensor<float> qkScores = ubQK.Get<float>();
        DataCopy(qkScores, lg, tileSize);
        float mCurr = ReduceMax(qkScores, tileSize);              // 向量单元: reduce_max
        float mNew  = fmaxf(mPrev, mCurr);
        float alpha = expf(mPrev - mNew);                         // exp2→exp: 去 log2(e)
        Muls(outAcc, outAcc, alpha, DIM);                        // rescale 历史输出 acc_o

        Subs(qkScores, qkScores, mNew, tileSize);                 // qk -= mNew
        Exp(qkScores, qkScores, tileSize);                        // qk = exp(qk)  自然底
        float localSum = ReduceSum(qkScores, tileSize);          // 向量单元: reduce_sum
        float lNew = lPrev * alpha + localSum;                    // logsum 在线更新

        // P·V 累加:概率 qkScores 逐行加权 V(=KV 的非位置部分)
        for (int32_t j = 0; j < tileSize; ++j) {
            float weight = qkScores[j];
            Axpy(outAcc, k[j * (DIM + PE_DIM)], weight, DIM);    // acc_o += weight * v[j]
        }

        mPrev = mNew; lPrev = lNew;
        kL1.FreeTensor(k); cO.FreeTensor(lg);
    }

    TPipe pipe;
    TQue<TPosition::A1, 1> qL1;
    TQue<TPosition::B1, 1> kL1;
    TQue<TPosition::VECIN,1> vL1;
    TQue<TPosition::CO1,1> cO;
    TBuf<TPosition::VECCALC> ubQK, ubOut;
    GlobalTensor<half> qGm, qPeGm, kvGm, kPeGm, outGm;
    int32_t batchIdx, headIdx, B, numHeads, seqlenKv, nTile;
    float softmaxScale;
};
