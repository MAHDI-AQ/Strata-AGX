// src/prefill/handoff_fp8.cu - Path B: Block-Microscaled FP8 (MXFP8 B=64) Inter-Stage Activation Handoff Compression
#include "strata/kernels/cuda/handoff_fp8.cuh"

namespace strata::kernels::cuda {

void pack_activations_mxfp8(
    const float* src,
    uint8_t* dst_payload,
    float* dst_scales,
    int64_t tokens,
    int64_t elems_per_token,
    int64_t stride_bytes,
    void* stream
) {
    launch_pack_activations_mxfp8(src, dst_payload, dst_scales, tokens, elems_per_token, stride_bytes, stream);
}

void unpack_activations_mxfp8(
    const uint8_t* src_payload,
    const float* src_scales,
    float* dst,
    int64_t tokens,
    int64_t elems_per_token,
    int64_t stride_bytes,
    void* stream
) {
    launch_unpack_activations_mxfp8(src_payload, src_scales, dst, tokens, elems_per_token, stride_bytes, stream);
}

}  // namespace strata::kernels::cuda
