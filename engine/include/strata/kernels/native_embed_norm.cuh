#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <algorithm>
#include <type_traits>

namespace strata {
namespace kernels {

// Intra-warp FP32 reduction via hardware shuffle
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }
    return val;
}

// SM 89 1-cycle FP8 E4M3 to FP16x2 hardware instruction with portable fallback
__device__ __forceinline__ __half2 cvt_fp8x2_to_half2_ptx(uint16_t raw_fp8x2) {
    uint32_t half2_storage;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    // "h" constraint binds to 16-bit register operand required by cvt.rn.f16x2.e4m3x2
    asm volatile("{cvt.rn.f16x2.e4m3x2 %0, %1;}\n" : "=r"(half2_storage) : "h"(raw_fp8x2));
#else
    __half2_raw raw = __nv_cvt_fp8x2_to_halfraw2(raw_fp8x2, __NV_E4M3);
    half2_storage = *reinterpret_cast<uint32_t*>(&raw);
#endif
    return *reinterpret_cast<__half2*>(&half2_storage);
}

// Unpack 16 FP8 E4M3 elements from a 128-bit uint4 into 16 FP32 values
__device__ __forceinline__ void unpack_16_fp8_to_float(
    uint4 raw_bytes,
    float out[16]
) {
    const uint16_t* p16 = reinterpret_cast<const uint16_t*>(&raw_bytes);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        __half2 h2 = cvt_fp8x2_to_half2_ptx(p16[i]);
        float2 f2 = __half22float2(h2);
        out[i * 2]     = f2.x;
        out[i * 2 + 1] = f2.y;
    }
}

/**
 * Fused Embedding Gather + Optional Root RMSNorm + Hyper-Connection Stream Broadcast
 * Validated on SM 89 (RTX 4090)
 *
 * Supports both FP16 (half) and FP32 (float) destination buffers.
 * When gamma == nullptr, Root RMSNorm is bypassed for raw gather + broadcast parity.
 */
template <typename TOut>
__global__ void __launch_bounds__(256, 6) fused_embedding_rmsnorm_hc_broadcast_kernel(
    const uint4* __restrict__ embed_table,
    const int32_t* __restrict__ tokens,
    const half* __restrict__ gamma,
    TOut* __restrict__ out_main,
    TOut* __restrict__ out_residual,
    TOut* __restrict__ out_hc,
    const float eps,
    const int32_t V,
    const int32_t D
) {
    __shared__ float smem_warp_sums[8];
    __shared__ float smem_r_rms;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int token_idx = blockIdx.x;

    const int32_t tok = tokens[token_idx];
    
    // Defensive architectural mandate: Branchless OOB clamp min((uint32_t)max(0, tok), V - 1)
    const uint32_t safe_tok = min(static_cast<uint32_t>(max(0, tok)), static_cast<uint32_t>(V - 1));

    const int num_vectors = D / 16; // 3,584 / 16 = 224 vectors of uint4
    const uint4* tok_embed_row = embed_table + static_cast<size_t>(safe_tok) * num_vectors;

    uint4 raw_vec;
    float vals[16];
    float local_sumsq = 0.0f;

    // Vectorized 128-bit global load: 224 threads fetch all 3,584 elements concurrently
    if (tid < num_vectors) {
        raw_vec = tok_embed_row[tid];
        unpack_16_fp8_to_float(raw_vec, vals);
        if (gamma != nullptr) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                local_sumsq += vals[i] * vals[i];
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            vals[i] = 0.0f;
        }
    }

    float r_rms = 1.0f;
    if (gamma != nullptr) {
        // Step 1: Intra-warp reduction
        float warp_sum = warp_reduce_sum(local_sumsq);

        // Step 2: Bank-conflict-free shared memory staging (only 8 writes)
        if (lane_id == 0) {
            smem_warp_sums[warp_id] = warp_sum;
        }
        __syncthreads();

        // Step 3: Warp 0 computes block sum and Root RMS reciprocal
        if (warp_id == 0) {
            float b_sum = (lane_id < 8) ? smem_warp_sums[lane_id] : 0.0f;
            b_sum = warp_reduce_sum(b_sum);
            if (lane_id == 0) {
                float mean_square = b_sum * (1.0f / static_cast<float>(D));
                mean_square = fmaxf(mean_square, 1e-12f);
                smem_r_rms = rsqrtf(mean_square + eps);
            }
        }
        __syncthreads();
        r_rms = smem_r_rms;
    }

    // Step 4: Normalization (if gamma) + 128-bit Vectorized Broadcast
    if (tid < num_vectors) {
        if (gamma != nullptr) {
            const uint4* gamma_u4 = reinterpret_cast<const uint4*>(gamma);
            uint4 g0_raw = gamma_u4[tid * 2];
            uint4 g1_raw = gamma_u4[tid * 2 + 1];

            const half* g0 = reinterpret_cast<const half*>(&g0_raw);
            const half* g1 = reinterpret_cast<const half*>(&g1_raw);

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                vals[i]     = vals[i] * r_rms * __half2float(g0[i]);
                vals[i + 8] = vals[i + 8] * r_rms * __half2float(g1[i]);
            }
        }

        if constexpr (std::is_same<TOut, float>::value) {
            float4 f4_0 = make_float4(vals[0],  vals[1],  vals[2],  vals[3]);
            float4 f4_1 = make_float4(vals[4],  vals[5],  vals[6],  vals[7]);
            float4 f4_2 = make_float4(vals[8],  vals[9],  vals[10], vals[11]);
            float4 f4_3 = make_float4(vals[12], vals[13], vals[14], vals[15]);

            const size_t out_f4_per_tok = static_cast<size_t>(D / 4); // 3584 / 4 = 896
            const size_t tok_f4_offset = static_cast<size_t>(token_idx) * out_f4_per_tok + tid * 4;

            if (out_main != nullptr) {
                float4* p = reinterpret_cast<float4*>(out_main);
                p[tok_f4_offset + 0] = f4_0;
                p[tok_f4_offset + 1] = f4_1;
                p[tok_f4_offset + 2] = f4_2;
                p[tok_f4_offset + 3] = f4_3;
            }

            if (out_residual != nullptr) {
                float4* p = reinterpret_cast<float4*>(out_residual);
                p[tok_f4_offset + 0] = f4_0;
                p[tok_f4_offset + 1] = f4_1;
                p[tok_f4_offset + 2] = f4_2;
                p[tok_f4_offset + 3] = f4_3;
            }

            if (out_hc != nullptr) {
                #pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const size_t hc_offset_f4 = (static_cast<size_t>(token_idx) * 4 + s) * out_f4_per_tok + tid * 4;
                    float4* p = reinterpret_cast<float4*>(out_hc);
                    p[hc_offset_f4 + 0] = f4_0;
                    p[hc_offset_f4 + 1] = f4_1;
                    p[hc_offset_f4 + 2] = f4_2;
                    p[hc_offset_f4 + 3] = f4_3;
                }
            }
        } else {
            uint4 out_v0, out_v1;
            half* h_out0 = reinterpret_cast<half*>(&out_v0);
            half* h_out1 = reinterpret_cast<half*>(&out_v1);

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                h_out0[i] = __float2half(vals[i]);
                h_out1[i] = __float2half(vals[i + 8]);
            }

            const size_t out_vectors_per_tok = static_cast<size_t>(D / 8); // 3584 / 8 = 448
            const size_t tok_out_offset_u4 = static_cast<size_t>(token_idx) * out_vectors_per_tok + tid * 2;

            if (out_main != nullptr) {
                uint4* main_u4 = reinterpret_cast<uint4*>(out_main);
                main_u4[tok_out_offset_u4]     = out_v0;
                main_u4[tok_out_offset_u4 + 1] = out_v1;
            }

            if (out_residual != nullptr) {
                uint4* res_u4 = reinterpret_cast<uint4*>(out_residual);
                res_u4[tok_out_offset_u4]     = out_v0;
                res_u4[tok_out_offset_u4 + 1] = out_v1;
            }

            if (out_hc != nullptr) {
                #pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const size_t hc_offset_u4 = (static_cast<size_t>(token_idx) * 4 + s) * out_vectors_per_tok + tid * 2;
                    uint4* hc_u4 = reinterpret_cast<uint4*>(out_hc);
                    hc_u4[hc_offset_u4]     = out_v0;
                    hc_u4[hc_offset_u4 + 1] = out_v1;
                }
            }
        }
    }
}

inline cudaError_t launch_fused_embedding_rmsnorm_hc(
    const uint4* embed_table,
    const int32_t* tokens,
    const half* gamma,
    half* out_main,
    half* out_residual,
    half* out_hc,
    float eps,
    int32_t V,
    int32_t D,
    int32_t num_tokens,
    cudaStream_t stream = nullptr
) {
    if (num_tokens <= 0) return cudaSuccess;
    dim3 grid(num_tokens);
    dim3 block(256);
    fused_embedding_rmsnorm_hc_broadcast_kernel<half><<<grid, block, 0, stream>>>(
        embed_table, tokens, gamma, out_main, out_residual, out_hc, eps, V, D
    );
    return cudaGetLastError();
}

inline cudaError_t launch_fused_embedding_rmsnorm_hc(
    const uint4* embed_table,
    const int32_t* tokens,
    const half* gamma,
    float* out_main,
    float* out_residual,
    float* out_hc,
    float eps,
    int32_t V,
    int32_t D,
    int32_t num_tokens,
    cudaStream_t stream = nullptr
) {
    if (num_tokens <= 0) return cudaSuccess;
    dim3 grid(num_tokens);
    dim3 block(256);
    fused_embedding_rmsnorm_hc_broadcast_kernel<float><<<grid, block, 0, stream>>>(
        embed_table, tokens, gamma, out_main, out_residual, out_hc, eps, V, D
    );
    return cudaGetLastError();
}

} // namespace kernels
} // namespace strata
