// src/kernels/cuda/fused_gdn.cu - see include/strata/kernels/fused_gdn.hpp.
#include <unistd.h>
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int S = 128;          // state size (rows = cols = 128)
constexpr int WARPS = 16;       // 16 warps in a 512-thread block
constexpr int ROWS_PER_WARP = S / WARPS; // 8 rows per warp

__global__ void __launch_bounds__(512) gdn_step_norm_kernel(float* __restrict__ state, const float* __restrict__ q,
                                                            const float* __restrict__ k, const float* __restrict__ v,
                                                            const float* __restrict__ gate,
                                                            const float* __restrict__ beta,
                                                            const float* __restrict__ z,
                                                            const float* __restrict__ gamma, float eps,
                                                            float* __restrict__ y, int h_k, int h_v,
                                                            uint8_t* __restrict__ y_q8_0) {
    __shared__ float sk[S], sq[S];
    __shared__ float4 red4[WARPS][32];
    __shared__ float s_scale;
    __shared__ float s_y[S];

    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;    // 0..15
    const int lane = tid & 31;       // 0..31 (each lane owns 4 columns: lane*4 .. lane*4+3)
    const int qh = head % h_k;

    if (tid < S) {
        sk[tid] = k[qh * S + tid];
        sq[tid] = q[qh * S + tid];
    }

    // 128-bit vector transactions: 32 lanes load 128 columns in 1 instruction
    float4* state4 = reinterpret_cast<float4*>(state);
    const size_t row_stride4 = (size_t) h_v * 32;
    float4 s[ROWS_PER_WARP];
    const size_t base_offset = ((size_t) (warp_id * ROWS_PER_WARP) * h_v + head) * 32 + lane;

#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        s[r] = state4[base_offset + (size_t) r * row_stride4];
    }
    __syncthreads();

    const float g = __expf(gate[head]);
    float4 kv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const float k_val = sk[warp_id * ROWS_PER_WARP + r];
        kv.x = fmaf(s[r].x, k_val, kv.x);
        kv.y = fmaf(s[r].y, k_val, kv.y);
        kv.z = fmaf(s[r].z, k_val, kv.z);
        kv.w = fmaf(s[r].w, k_val, kv.w);
    }
    red4[warp_id][lane] = kv;
    __syncthreads();

    float4 kv_col = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
        const float4 p = red4[w][lane];
        kv_col.x += p.x; kv_col.y += p.y; kv_col.z += p.z; kv_col.w += p.w;
    }

    const float4 v4 = *reinterpret_cast<const float4*>(v + head * S + lane * 4);
    const float bh = beta[head];
    float4 delta;
    delta.x = (v4.x - g * kv_col.x) * bh;
    delta.y = (v4.y - g * kv_col.y) * bh;
    delta.z = (v4.z - g * kv_col.z) * bh;
    delta.w = (v4.w - g * kv_col.w) * bh;

    float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const float k_val = sk[warp_id * ROWS_PER_WARP + r];
        const float q_val = sq[warp_id * ROWS_PER_WARP + r];
        s[r].x = fmaf(g, s[r].x, k_val * delta.x);
        s[r].y = fmaf(g, s[r].y, k_val * delta.y);
        s[r].z = fmaf(g, s[r].z, k_val * delta.z);
        s[r].w = fmaf(g, s[r].w, k_val * delta.w);

        o.x = fmaf(s[r].x, q_val, o.x);
        o.y = fmaf(s[r].y, q_val, o.y);
        o.z = fmaf(s[r].z, q_val, o.z);
        o.w = fmaf(s[r].w, q_val, o.w);

        state4[base_offset + (size_t) r * row_stride4] = s[r];
    }
    __syncthreads();
    red4[warp_id][lane] = o;
    __syncthreads();

    float4 oc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
        const float4 p = red4[w][lane];
        oc.x += p.x; oc.y += p.y; oc.z += p.z; oc.w += p.w;
    }
    const float inv_sqrt_S = rsqrtf((float) S);
    oc.x *= inv_sqrt_S; oc.y *= inv_sqrt_S; oc.z *= inv_sqrt_S; oc.w *= inv_sqrt_S;

    // Full-warp cooperative RMS reduction across all 128 outputs:
    // Warp 0's 32 lanes hold all 128 elements (4 per lane).
    float sq_part = 0.0f;
    if (warp_id == 0) {
        sq_part = oc.x * oc.x + oc.y * oc.y + oc.z * oc.z + oc.w * oc.w;
#pragma unroll
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if (lane == 0) {
            s_scale = rsqrtf(sq_part / (float) S + eps);
        }
    }
    __syncthreads();
    const float scale = s_scale;

    // Epilogue: evaluate output y and populate shared buffer
    if (warp_id == 0) {
        const float4 gamma4 = *reinterpret_cast<const float4*>(gamma + lane * 4);
        const float4 z4 = *reinterpret_cast<const float4*>(z + head * S + lane * 4);
        float4 y4;
        y4.x = oc.x * scale * gamma4.x * (1.0f / (1.0f + __expf(-z4.x)));
        y4.y = oc.y * scale * gamma4.y * (1.0f / (1.0f + __expf(-z4.y)));
        y4.z = oc.z * scale * gamma4.z * (1.0f / (1.0f + __expf(-z4.z)));
        y4.w = oc.w * scale * gamma4.w * (1.0f / (1.0f + __expf(-z4.w)));

        *reinterpret_cast<float4*>(y + head * S + lane * 4) = y4;

        s_y[lane * 4] = y4.x;
        s_y[lane * 4 + 1] = y4.y;
        s_y[lane * 4 + 2] = y4.z;
        s_y[lane * 4 + 3] = y4.w;
    }
    __syncthreads();

    // Direct Q8_0 Epilogue: Warps 0..3 quantize the 4 blocks of 32 elements simultaneously
    if (y_q8_0 != nullptr && warp_id < 4) {
        const int b = warp_id;
        const int i = lane;
        const float val = s_y[b * 32 + i];
        float amax = fabsf(val);
#pragma unroll
        for (int o2 = 16; o2 > 0; o2 >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o2));

        uint8_t* blk = y_q8_0 + (size_t) (head * 4 + b) * 34;
        if (amax == 0.0f) {
            if (i == 0) {
                const uint16_t zb = f16_from_f32(0.0f);
                blk[0] = (uint8_t) (zb & 0xFF);
                blk[1] = (uint8_t) (zb >> 8);
            }
            blk[2 + i] = 0;
        } else {
            const float d32 = amax / 127.0f;
            if (i == 0) {
                const uint16_t d16 = f16_from_f32(d32);
                blk[0] = (uint8_t) (d16 & 0xFF);
                blk[1] = (uint8_t) (d16 >> 8);
            }
            double q = rint((double) val / (double) d32);
            if (q > 127.0) q = 127.0;
            if (q < -128.0) q = -128.0;
            blk[2 + i] = (uint8_t) (int8_t) q;
        }
    }
}

__global__ void __launch_bounds__(S) gdn_conv_l2_kernel(float* __restrict__ hist, const float* __restrict__ qkv,
                                                        const float* __restrict__ w, float* __restrict__ h,
                                                        int qk_heads, float eps) {
    __shared__ float part[S / 32];
    const int c = blockIdx.x * S + threadIdx.x;
    const float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2], x = qkv[c];
    float sum = v0 * w[c * 4] + v1 * w[c * 4 + 1] + v2 * w[c * 4 + 2] + x * w[c * 4 + 3];
    hist[c * 3] = v1;
    hist[c * 3 + 1] = v2;
    hist[c * 3 + 2] = x;
    float y = sum / (1.0f + __expf(-sum));
    if ((int) blockIdx.x < qk_heads) {
        float sq = y * y;
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
        __syncthreads();
        const float ss = part[0] + part[1] + part[2] + part[3];
        y *= rsqrtf(ss + eps);
    }
    h[c] = y;
}

__global__ void __launch_bounds__(256) gdn_ab_kernel(const float* __restrict__ x, const uint16_t* __restrict__ wa,
                                                     const uint16_t* __restrict__ wb, const float* __restrict__ dt,
                                                     const float* __restrict__ ssm_a, float* __restrict__ gate,
                                                     float* __restrict__ beta, int n, int h_v) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= 2 * h_v) return;
    const bool is_beta = row >= h_v;
    const int r = is_beta ? row - h_v : row;
    const uint4* w4 = reinterpret_cast<const uint4*>((is_beta ? wb : wa) + (size_t) r * n);
    float acc = 0.0f;
    for (int j = lane; j < n / 8; j += 32) {
        const uint4 wv = __ldg(w4 + j);
        const float4 xa = *reinterpret_cast<const float4*>(x + j * 8);
        const float4 xb = *reinterpret_cast<const float4*>(x + j * 8 + 4);
        acc = fmaf(__uint_as_float(wv.x << 16), xa.x, acc); acc = fmaf(__uint_as_float(wv.x & 0xffff0000u), xa.y, acc);
        acc = fmaf(__uint_as_float(wv.y << 16), xa.z, acc); acc = fmaf(__uint_as_float(wv.y & 0xffff0000u), xa.w, acc);
        acc = fmaf(__uint_as_float(wv.z << 16), xb.x, acc); acc = fmaf(__uint_as_float(wv.z & 0xffff0000u), xb.y, acc);
        acc = fmaf(__uint_as_float(wv.w << 16), xb.z, acc); acc = fmaf(__uint_as_float(wv.w & 0xffff0000u), xb.w, acc);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane != 0) return;
    if (is_beta) {
        beta[r] = 1.0f / (1.0f + __expf(-acc));
    } else {
        const float v = acc + dt[r];
        const float sp = v > 20.0f ? v : log1pf(__expf(v));
        gate[r] = sp * ssm_a[r];
    }
}

}  // namespace

void fused_gdn_conv_l2(float* history, const float* qkv, const float* conv_w, float* h, int channels, int qk_heads,
                       float eps, void* stream) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || qk_heads < 0 || qk_heads > channels / S) {
        std::fprintf(stderr, "fused_gdn_conv_l2: invalid arguments\n");
        _exit(1);
    }
    gdn_conv_l2_kernel<<<(unsigned) (channels / S), S, 0, (cudaStream_t) stream>>>(history, qkv, conv_w, h, qk_heads, eps);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "fused_gdn_conv_l2: %s\n", cudaGetErrorString(e)); _exit(1); }
}

void fused_gdn_ab(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, void* stream) {
    if (!x || !w_alpha || !w_beta || !dt || !ssm_a || !gate || !beta || n_embd % 8 != 0 || h_v <= 0) {
        std::fprintf(stderr, "fused_gdn_ab: invalid arguments\n");
        _exit(1);
    }
    gdn_ab_kernel<<<(unsigned) ((2 * h_v + 7) / 8), 256, 0, (cudaStream_t) stream>>>(x, w_alpha, w_beta, dt, ssm_a, gate,
                                                                                     beta, n_embd, h_v);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "fused_gdn_ab: %s\n", cudaGetErrorString(e)); _exit(1); }
}

void fused_gdn_step_norm(float* state, const float* q, const float* k, const float* v, const float* gate,
                         const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k, int h_v,
                         void* stream, uint8_t* y_q8_0) {
    if (!state || !q || !k || !v || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v <= 0 || h_v % h_k) {
        std::fprintf(stderr, "fused_gdn_step_norm: invalid arguments\n");
        _exit(1);
    }
    gdn_step_norm_kernel<<<(unsigned) h_v, 512, 0, (cudaStream_t) stream>>>(
        state, q, k, v, gate, beta, z, gamma, eps, y, h_k, h_v, y_q8_0);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fused_gdn_step_norm: %s\n", cudaGetErrorString(e));
        _exit(1);
    }
}

}  // namespace strata::kernels
