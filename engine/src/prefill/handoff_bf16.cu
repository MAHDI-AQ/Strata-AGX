// src/prefill/handoff_bf16.cu - Vectorized BF16 Inter-Stage Activation Handoff
#include "strata/kernels/cuda/handoff_bf16.cuh"

#include <cuda_runtime.h>
#if defined(__CUDACC__) && !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
#include <cuda_bf16.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace strata::kernels::cuda {
namespace {

__device__ __forceinline__ uint16_t f32_to_bf16_rn_scalar(float f) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
    const __nv_bfloat16 b = __float2bfloat16_rn(f);
    uint16_t out = 0;
    std::memcpy(&out, &b, sizeof(out));
    return out;
#else
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return static_cast<uint16_t>((u >> 16) | 0x0040u);
    }
    const uint32_t rounding_bias = ((u >> 16) & 1u) + 0x7fffu;
    return static_cast<uint16_t>((u + rounding_bias) >> 16);
#endif
}

__device__ __forceinline__ uint32_t f32x2_to_bf16x2_rn(float a, float b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
    const float2 f2 = make_float2(a, b);
    const __nv_bfloat162 b2 = __float22bfloat162_rn(f2);
    uint32_t out = 0;
    std::memcpy(&out, &b2, sizeof(out));
    return out;
#else
    const uint32_t lo = static_cast<uint32_t>(f32_to_bf16_rn_scalar(a));
    const uint32_t hi = static_cast<uint32_t>(f32_to_bf16_rn_scalar(b));
    return lo | (hi << 16);
#endif
}

__device__ __forceinline__ float bf16_to_f32_scalar(uint16_t b) {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float out = 0.0f;
    std::memcpy(&out, &u, sizeof(out));
    return out;
}

__device__ __forceinline__ float2 bf16x2_to_f32x2(uint32_t packed) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
    __nv_bfloat162 b2;
    std::memcpy(&b2, &packed, sizeof(b2));
    return __bfloat1622float2(b2);
#else
    return make_float2(
        bf16_to_f32_scalar(static_cast<uint16_t>(packed & 0xffffu)),
        bf16_to_f32_scalar(static_cast<uint16_t>(packed >> 16))
    );
#endif
}

__global__ void __launch_bounds__(256) pack_activations_bf16_kernel(
    const float* __restrict__ src,
    uint16_t* __restrict__ dst,
    int64_t count
) {
    const int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const bool can_vec4 =
        ((reinterpret_cast<uintptr_t>(src) & 15u) == 0) &&
        ((reinterpret_cast<uintptr_t>(dst) & 7u) == 0);

    if (can_vec4) {
        const int64_t n4 = count / 4;
        const float4* __restrict__ src4 = reinterpret_cast<const float4*>(src);
        uint2* __restrict__ dst2 = reinterpret_cast<uint2*>(dst);
        for (int64_t i = gid; i < n4; i += stride) {
            const float4 v = src4[i];
            uint2 out;
            out.x = f32x2_to_bf16x2_rn(v.x, v.y);
            out.y = f32x2_to_bf16x2_rn(v.z, v.w);
            dst2[i] = out;
        }
        const int64_t tail_base = n4 * 4;
        if (gid < count - tail_base) {
            const int64_t idx = tail_base + gid;
            dst[idx] = f32_to_bf16_rn_scalar(src[idx]);
        }
    } else {
        for (int64_t i = gid; i < count; i += stride) {
            dst[i] = f32_to_bf16_rn_scalar(src[i]);
        }
    }
}

__global__ void __launch_bounds__(256) unpack_activations_bf16_kernel(
    const uint16_t* __restrict__ src,
    float* __restrict__ dst,
    int64_t count
) {
    const int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const bool can_vec4 =
        ((reinterpret_cast<uintptr_t>(src) & 7u) == 0) &&
        ((reinterpret_cast<uintptr_t>(dst) & 15u) == 0);

    if (can_vec4) {
        const int64_t n4 = count / 4;
        const uint2* __restrict__ src2 = reinterpret_cast<const uint2*>(src);
        float4* __restrict__ dst4 = reinterpret_cast<float4*>(dst);
        for (int64_t i = gid; i < n4; i += stride) {
            const uint2 p = src2[i];
            const float2 lo = bf16x2_to_f32x2(p.x);
            const float2 hi = bf16x2_to_f32x2(p.y);
            dst4[i] = make_float4(lo.x, lo.y, hi.x, hi.y);
        }
        const int64_t tail_base = n4 * 4;
        if (gid < count - tail_base) {
            const int64_t idx = tail_base + gid;
            dst[idx] = bf16_to_f32_scalar(src[idx]);
        }
    } else {
        for (int64_t i = gid; i < count; i += stride) {
            dst[i] = bf16_to_f32_scalar(src[i]);
        }
    }
}

} // namespace

void pack_activations_bf16(
    const float* src,
    uint16_t* dst,
    int64_t count,
    void* stream
) {
    if (count <= 0) return;
    const int64_t work = (count + 3) / 4;
    const unsigned blocks = static_cast<unsigned>(std::clamp<int64_t>((work + 255) / 256, 1, 8192));
    pack_activations_bf16_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(src, dst, count);
}

void unpack_activations_bf16(
    const uint16_t* src,
    float* dst,
    int64_t count,
    void* stream
) {
    if (count <= 0) return;
    const int64_t work = (count + 3) / 4;
    const unsigned blocks = static_cast<unsigned>(std::clamp<int64_t>((work + 255) / 256, 1, 8192));
    unpack_activations_bf16_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(src, dst, count);
}

} // namespace strata::kernels::cuda
