#ifndef CONV_BIAS_RELU_TILING_DATA_H
#define CONV_BIAS_RELU_TILING_DATA_H

#include <cstdint>

// Fixed-shape demo contract used by both host tiling and the Ascend C kernel.
// All dimensions are concrete so that a code-to-hardware trace tool can recover
// exact tile IDs, byte counts, formats, locations and loop iterations.
struct ConvBiasReluTilingData {
    // Logical operator shape.
    uint32_t n;
    uint32_t ci;
    uint32_t hi;
    uint32_t wi;
    uint32_t co;
    uint32_t ho;
    uint32_t wo;
    uint32_t kh;
    uint32_t kw;

    // Convolution attributes.
    uint32_t strideH;
    uint32_t strideW;
    uint32_t padTop;
    uint32_t padBottom;
    uint32_t padLeft;
    uint32_t padRight;
    uint32_t dilationH;
    uint32_t dilationW;

    // Cube view: A[M,K] * B[K,N] + Bias[N] -> C[M,N].
    uint32_t m;
    uint32_t k;
    uint32_t nCube;

    // Tile shape and tile counts.
    uint32_t tileM;
    uint32_t tileK;
    uint32_t tileN;
    uint32_t mTiles;
    uint32_t kTiles;
    uint32_t nTiles;
    uint32_t outputTileCount;

    // Exact storage sizes, expressed in elements.
    uint32_t featureGmElements;
    uint32_t weightGmElements;
    uint32_t biasGmElements;
    uint32_t outputGmElements;
    uint32_t fmapA1Elements;
    uint32_t weightB1Elements;
    uint32_t biasTileElements;
    uint32_t fmapA2Elements;
    uint32_t weightB2Elements;
    uint32_t accumCo1Elements;
};

#endif  // CONV_BIAS_RELU_TILING_DATA_H
