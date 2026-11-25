// include/strata/kernels/cuda/handoff_bf16.cuh - Path C: Vectorized BF16 Inter-Stage Activation Handoff
#pragma once

#include <cstdint>

namespace strata::kernels::cuda {

/// Quantize `count` FP32 elements into IEEE BF16 (`uint16_t` bit pattern, round-to-nearest-even) on `stream`.
void pack_activations_bf16(
    const float* src,
    uint16_t* dst,
    int64_t count,
    void* stream
);

/// Expand `count` IEEE BF16 (`uint16_t` bit pattern) elements back to FP32 on `stream`.
void unpack_activations_bf16(
    const uint16_t* src,
    float* dst,
    int64_t count,
    void* stream
);

} // namespace strata::kernels::cuda
