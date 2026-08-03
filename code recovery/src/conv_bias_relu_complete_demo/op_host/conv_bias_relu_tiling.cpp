/*
 * Fixed host-side tiling for conv_bias_relu_reference_complete.asc.
 *
 * This file deliberately supplies one concrete fixture rather than a general
 * shape-adaptive tiling algorithm. It makes every number recoverable by the
 * visualization prototype and launches eight AI Cores, one per output tile.
 */

#include "register/op_impl_registry.h"
#include "../op_kernel/conv_bias_relu_tiling_data.h"

namespace optiling {
namespace {

constexpr uint32_t CeilDiv(uint32_t x, uint32_t y)
{
    return (x + y - 1U) / y;
}

void FillFixedDemoTiling(ConvBiasReluTilingData& t)
{
    // Logical operator parameters.
    t.n = 1;
    t.ci = 16;
    t.hi = 8;
    t.wi = 8;
    t.co = 32;
    t.kh = 3;
    t.kw = 3;

    t.strideH = 1;
    t.strideW = 1;
    t.padTop = 1;
    t.padBottom = 1;
    t.padLeft = 1;
    t.padRight = 1;
    t.dilationH = 1;
    t.dilationW = 1;

    const uint32_t effectiveKh = t.dilationH * (t.kh - 1U) + 1U;
    const uint32_t effectiveKw = t.dilationW * (t.kw - 1U) + 1U;
    t.ho = (t.hi + t.padTop + t.padBottom - effectiveKh) / t.strideH + 1U;
    t.wo = (t.wi + t.padLeft + t.padRight - effectiveKw) / t.strideW + 1U;

    // Cube view and fixed tile shape.
    t.m = t.n * t.ho * t.wo;       // 64 output positions
    t.k = t.ci * t.kh * t.kw;      // 144 reduction elements
    t.nCube = t.co;                // 32 output channels
    t.tileM = 16;
    t.tileK = 16;
    t.tileN = 16;

    t.mTiles = CeilDiv(t.m, t.tileM);       // 4: M0..M3
    t.kTiles = CeilDiv(t.k, t.tileK);       // 9: K0..K8
    t.nTiles = CeilDiv(t.nCube, t.tileN);   // 2: N0..N1
    t.outputTileCount = t.mTiles * t.nTiles; // 8: OT0..OT7

    // Exact element counts.
    t.featureGmElements = t.n * t.ci * t.hi * t.wi; // 1024 half
    t.weightGmElements = t.k * t.co;                // 4608 half, ND [K,Co]
    t.biasGmElements = t.co;                        // 32 float
    t.outputGmElements = t.m * t.co;                // 2048 half, ND [M,Co]

    // Each core stages the complete feature map, one N-tile of weights and bias.
    t.fmapA1Elements = t.featureGmElements;          // 1024 half = 2048 B
    t.weightB1Elements = t.k * t.tileN;              // 2304 half = 4608 B
    t.biasTileElements = t.tileN;                    // 16 float = 64 B
    t.fmapA2Elements = t.tileM * t.tileK;            // 256 half = 512 B
    t.weightB2Elements = t.tileK * t.tileN;          // 256 half = 512 B
    t.accumCo1Elements = t.tileM * t.tileN;          // 256 float = 1024 B
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto* tiling = context->GetTilingData<ConvBiasReluTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    FillFixedDemoTiling(*tiling);
    context->SetBlockDim(tiling->outputTileCount);

    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace != nullptr) {
        workspace[0] = 0;
    }
    return ge::GRAPH_SUCCESS;
}

}  // namespace

IMPL_OP_OPTILING(ConvBiasReluReference).Tiling(TilingFunc);

}  // namespace optiling
