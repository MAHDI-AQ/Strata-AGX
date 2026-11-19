// include/strata/kernels/cuda/handoff_fp8.cuh - Path B: Block-Microscaled FP8 (MXFP8 B=64) Inter-Stage Activation Handoff Compression
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace strata::kernels::cuda {

constexpr int64_t kHandoffTokensMax = 4096;
constexpr int64_t kHandoffElemsPerToken = 12804; // D(10240) + N(2560) + inj(4)
constexpr int64_t kHandoffBlockSize = 64;
constexpr int64_t kHandoffBlocksPerToken = (kHandoffElemsPerToken + kHandoffBlockSize - 1) / kHandoffBlockSize; // 201 blocks
constexpr int64_t kHandoffFP8PayloadBytes = kHandoffElemsPerToken; // 12,804 B
constexpr int64_t kHandoffFP8PadBytes = 12; // 12 B
constexpr int64_t kHandoffFP8StrideBytes = 12816; // 12,804 bytes payload + 12 bytes pad (16-byte aligned)

static_assert(kHandoffFP8StrideBytes % 16 == 0, "FP8 handoff stride must be 16-byte aligned");

struct HandoffBufferFP8 {
    uint8_t* payload = nullptr; // Pinned host memory: TOKENS * 12816 bytes
    float* scales = nullptr;    // Pinned host memory: TOKENS * 201 * sizeof(float)
    size_t payload_bytes = 0;
    size_t scale_bytes = 0;
    size_t total_bytes = 0;
};

// C++ host launchers
void pack_activations_mxfp8(
    const float* src,
    uint8_t* dst_payload,
    float* dst_scales,
    int64_t tokens,
    int64_t elems_per_token,
    int64_t stride_bytes,
    void* stream
);

void unpack_activations_mxfp8(
    const uint8_t* src_payload,
    const float* src_scales,
    float* dst,
    int64_t tokens,
    int64_t elems_per_token,
    int64_t stride_bytes,
    void* stream
);

#if defined(__CUDACC__) || defined(__HIPCC__)
#include <cuda_fp16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ void store_fp8x2_cs(void* ptr, __nv_fp8x2_e4m3 val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("st.global.cs.b16 [%0], %1;" : : "l"(ptr), "h"(*(const unsigned short*)&val) : "memory");
#else
    *reinterpret_cast<__nv_fp8x2_e4m3*>(ptr) = val;
#endif
}

__device__ __forceinline__ __nv_fp8x2_e4m3 load_fp8x2_cs(const void* ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    unsigned short val;
    asm volatile("ld.global.cs.b16 %0, [%1];" : "=h"(val) : "l"(ptr) : "memory");
    return *reinterpret_cast<const __nv_fp8x2_e4m3*>(&val);
#else
    return *reinterpret_cast<const __nv_fp8x2_e4m3*>(ptr);
#endif
}

// Vectorized, Register-Fused Single-Pass Pack Kernel
__global__ void __launch_bounds__(256) pack_activations_mxfp8_kernel(
    const float* __restrict__ src,
    uint8_t* __restrict__ dst_payload,
    float* __restrict__ dst_scales,
    int64_t tokens,
    int64_t elems_per_token,
    int64_t stride_bytes
) {
    int warp_global_id = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
    int total_blocks = tokens * kHandoffBlocksPerToken;
    if (warp_global_id >= total_blocks) return;

    int lane = threadIdx.x & 31;
    int token_idx = warp_global_id / kHandoffBlocksPerToken;
    int blk_idx = warp_global_id % kHandoffBlocksPerToken;

    int elem0 = blk_idx * kHandoffBlockSize + lane * 2;
    int elem1 = elem0 + 1;

    float f0 = (elem0 < elems_per_token) ? src[(int64_t)token_idx * elems_per_token + elem0] : 0.0f;
    float f1 = (elem1 < elems_per_token) ? src[(int64_t)token_idx * elems_per_token + elem1] : 0.0f;

    // Step 1: Warp-level amax reduction
    float local_max = fmaxf(fabsf(f0), fabsf(f1));
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, ::__shfl_down_sync(0xffffffffu, local_max, offset));
    }
    float blk_amax = ::__shfl_sync(0xffffffffu, local_max, 0);

    // FP8 E4M3 range: max 448.0f
    float scale = (blk_amax > 1e-12f) ? (448.0f / blk_amax) : 1.0f;
    float inv_scale = (blk_amax > 1e-12f) ? (blk_amax / 448.0f) : 1.0f;

    if (lane == 0) {
        dst_scales[warp_global_id] = inv_scale;
    }

    // Step 2: Vectorized conversion via SM89 native cvt.rn.satfinite.e4m3x2.f32
    float2 f2 = make_float2(f0 * scale, f1 * scale);
    __nv_fp8x2_e4m3 q(f2);

    int64_t blk_dst_offset = (int64_t)token_idx * stride_bytes + (int64_t)blk_idx * kHandoffBlockSize;
    if (elem1 < elems_per_token) {
        store_fp8x2_cs(dst_payload + blk_dst_offset + lane * 2, q);
    } else if (elem0 < elems_per_token) {
        dst_payload[blk_dst_offset + lane * 2] = static_cast<uint8_t>(__nv_fp8_e4m3(f0 * scale));
    }
}

// Vectorized Unpack Kernel
__global__ void __launch_bounds__(256) unpack_activations_mxfp8_kernel(
    const uint8_t* __restrict__ src_payload,
    const float* __restrict__ src_scales,
    float* __restrict__ dst,
    int64_t tokens,
    int64_t elems_per_token,
    int64_t stride_bytes
) {
    int warp_global_id = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
    int total_blocks = tokens * kHandoffBlocksPerToken;
    if (warp_global_id >= total_blocks) return;

    int lane = threadIdx.x & 31;
    int token_idx = warp_global_id / kHandoffBlocksPerToken;
    int blk_idx = warp_global_id % kHandoffBlocksPerToken;

    int64_t blk_src_offset = (int64_t)token_idx * stride_bytes + (int64_t)blk_idx * kHandoffBlockSize;
    int elem0 = blk_idx * kHandoffBlockSize + lane * 2;
    int elem1 = elem0 + 1;

    float inv_scale = src_scales[warp_global_id];
    __nv_fp8x2_e4m3 q = load_fp8x2_cs(src_payload + blk_src_offset + lane * 2);
    float2 f = static_cast<float2>(q);
    f.x *= inv_scale;
    f.y *= inv_scale;

    if (elem0 < elems_per_token) dst[(int64_t)token_idx * elems_per_token + elem0] = f.x;
    if (elem1 < elems_per_token) dst[(int64_t)token_idx * elems_per_token + elem1] = f.y;
}

inline void launch_pack_activations_mxfp8(
    const float* src,
    uint8_t* dst_payload,
    float* dst_scales,
    int64_t tokens,
    int64_t elems_per_token,
    int64_t stride_bytes,
    void* stream
) {
    int total_blocks = tokens * kHandoffBlocksPerToken;
    int ctas = (total_blocks + 7) / 8;
    pack_activations_mxfp8_kernel<<<ctas, 256, 0, (cudaStream_t)stream>>>(
        src, dst_payload, dst_scales, tokens, elems_per_token, stride_bytes
    );
}

inline void launch_unpack_activations_mxfp8(
    const uint8_t* src_payload,
    const float* src_scales,
    float* dst,
    int64_t tokens,
    int64_t elems_per_token,
    int64_t stride_bytes,
    void* stream
) {
    int total_blocks = tokens * kHandoffBlocksPerToken;
    int ctas = (total_blocks + 7) / 8;
    unpack_activations_mxfp8_kernel<<<ctas, 256, 0, (cudaStream_t)stream>>>(
        src_payload, src_scales, dst, tokens, elems_per_token, stride_bytes
    );
}

#endif // defined(__CUDACC__) || defined(__HIPCC__)

} // namespace strata::kernels::cuda
