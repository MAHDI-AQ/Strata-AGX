#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <cmath>

namespace strata {
namespace kernels {

struct SoftmaxArgmaxTuple {
    float max_val;
    float sum_exp;
    int32_t argmax_idx;
};

// Milakov-Gimelshein Online Softmax state merger with deterministic lexicographical tie-breaking
__host__ __device__ __forceinline__ SoftmaxArgmaxTuple merge_softmax_tuples(
    const SoftmaxArgmaxTuple& a,
    const SoftmaxArgmaxTuple& b
) {
    SoftmaxArgmaxTuple res;
    if (a.max_val > b.max_val) {
        res.max_val = a.max_val;
#ifdef __CUDA_ARCH__
        res.sum_exp = a.sum_exp + b.sum_exp * __expf(b.max_val - a.max_val);
#else
        res.sum_exp = a.sum_exp + b.sum_exp * expf(b.max_val - a.max_val);
#endif
        res.argmax_idx = a.argmax_idx;
    } else if (b.max_val > a.max_val) {
        res.max_val = b.max_val;
#ifdef __CUDA_ARCH__
        res.sum_exp = b.sum_exp + a.sum_exp * __expf(a.max_val - b.max_val);
#else
        res.sum_exp = b.sum_exp + a.sum_exp * expf(a.max_val - b.max_val);
#endif
        res.argmax_idx = b.argmax_idx;
    } else {
        // Exact tie: deterministic lexicographical tie-breaking (lower token ID wins)
        res.max_val = a.max_val;
        res.sum_exp = a.sum_exp + b.sum_exp;
        res.argmax_idx = (a.argmax_idx < b.argmax_idx) ? a.argmax_idx : b.argmax_idx;
    }
    return res;
}

__device__ __forceinline__ SoftmaxArgmaxTuple warp_reduce_softmax(SoftmaxArgmaxTuple val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        SoftmaxArgmaxTuple peer;
        peer.max_val    = __shfl_down_sync(0xffffffff, val.max_val, mask);
        peer.sum_exp    = __shfl_down_sync(0xffffffff, val.sum_exp, mask);
        peer.argmax_idx = __shfl_down_sync(0xffffffff, val.argmax_idx, mask);
        val = merge_softmax_tuples(val, peer);
    }
    return val;
}

// Asynchronous 16-byte Global to Shared Memory Copy
__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* gmem_ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"(smem_addr), "l"(gmem_ptr));
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

// Hardware Ada 4th-Gen Tensor Core m16n8k32 FP8 GEMM PTX Primitive with FP32 Accumulation
__device__ __forceinline__ void mma_m16n8k32_fp8(
    float d[4],
    uint32_t a[4],
    uint32_t b[2],
    float c[4]
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
#else
    (void)d; (void)a; (void)b; (void)c;
#endif
}

// Philox 4x32 PRNG for In-Register Zero-DRAM Gumbel Sampling
__host__ __device__ __forceinline__ uint32_t philox_mulhi(uint32_t a, uint32_t b) {
#if defined(__CUDA_ARCH__)
    return __umulhi(a, b);
#else
    return static_cast<uint32_t>((static_cast<uint64_t>(a) * static_cast<uint64_t>(b)) >> 32);
#endif
}

__host__ __device__ __forceinline__ void philox4x32_round(uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3,
                                                         uint32_t k0, uint32_t k1) {
    const uint32_t hi0 = philox_mulhi(0x9E3779B9u, c0);
    const uint32_t hi1 = philox_mulhi(0xBB67AE85u, c2);
    const uint32_t lo0 = 0x9E3779B9u * c0;
    const uint32_t lo1 = 0xBB67AE85u * c2;
    const uint32_t n0 = hi1 ^ c1 ^ k0;
    const uint32_t n1 = lo1;
    const uint32_t n2 = hi0 ^ c3 ^ k1;
    const uint32_t n3 = lo0;
    c0 = n0; c1 = n1; c2 = n2; c3 = n3;
}

__host__ __device__ __forceinline__ float philox_uniform(uint64_t seed, uint64_t counter, uint32_t tid, uint32_t col) {
    uint32_t c0 = col;
    uint32_t c1 = static_cast<uint32_t>(counter);
    uint32_t c2 = static_cast<uint32_t>(seed) ^ (tid * 0x9E3779B9u);
    uint32_t c3 = static_cast<uint32_t>(seed >> 32);
    #pragma unroll
    for (int i = 0; i < 7; ++i) {
        philox4x32_round(c0, c1, c2, c3, static_cast<uint32_t>(i), 0u);
    }
    return (static_cast<float>(c0 >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

/**
 * Fused LM Head GEMV + In-Register Online Softmax
 * Tile: M=16, N=128, K=128 (Double Buffered)
 * Supports dynamic per-row activation scaling s_x and per-channel weight scaling s_w.
 * Completely eliminates VRAM logit writeback when out_logits == nullptr.
 * Launch bounds (128, 2): 2 blocks per SM = 73,824 bytes shared memory <= 102,400 bytes limit on Ada SM89.
 */
__global__ void __launch_bounds__(128, 2) fused_lm_head_gemv_online_softmax_kernel(
    const uint8_t* __restrict__ W_head,
    const uint8_t* __restrict__ X_act,
    SoftmaxArgmaxTuple* __restrict__ L2_scratchpad,
    float* __restrict__ out_logits,
    const int32_t T,
    const int32_t V,
    const int32_t K,
    const float temperature,
    const float scale_w,
    const float scale_x,
    const float* __restrict__ scale_w_ptr = nullptr,
    const float* __restrict__ scale_x_ptr = nullptr,
    const bool spec_gumbel = false,
    const uint64_t seed = 0,
    const uint64_t step_counter = 0
) {
    __shared__ alignas(16) uint8_t smem_W[2][128 * 128];
    __shared__ alignas(16) uint8_t smem_X[2][16 * 128];
    __shared__ SoftmaxArgmaxTuple smem_warp_tuples[4];
    __shared__ float smem_scale_w[128];
    __shared__ float smem_scale_x[16];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int group = lane_id >> 2;       // 0..7
    const int tid_in_grp = lane_id & 3;   // 0..3
    const int block_n = blockIdx.x;
    const int global_n_base = block_n * 128;

    if (global_n_base >= V) return;

    // Load per-channel weight scales and per-row activation scales into shared memory
    if (scale_w_ptr != nullptr) {
        int g_col = global_n_base + tid;
        smem_scale_w[tid] = (g_col < V) ? scale_w_ptr[g_col] : 1.0f;
    }
    if (scale_x_ptr != nullptr && tid < 16) {
        smem_scale_x[tid] = (tid < T) ? scale_x_ptr[tid] : 1.0f;
    }

    // 16 FP32 registers for accumulator (M=16, N=32 slice per warp: 4 sub-tiles of N=8)
    float accum[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            accum[i][j] = 0.0f;
        }
    }

    const int num_k_tiles = K / 128;
    int smem_buf = 0;

    // Prologue: Prefetch tile 0
    {
        #pragma unroll
        for (int step = 0; step < 8; ++step) {
            int r = (tid / 8) + step * 16;
            int c = (tid % 8) * 16;
            int g_row = global_n_base + r;
            const uint8_t* src_ptr = (g_row < V) ? (W_head + static_cast<size_t>(g_row) * K + c) : nullptr;
            void* dst_ptr = &smem_W[0][r * 128 + c];
            if (src_ptr) {
                cp_async_16(dst_ptr, src_ptr);
            }
        }
        int x_row = tid / 8;
        int x_col = (tid % 8) * 16;
        void* dst_x = &smem_X[0][x_row * 128 + x_col];
        if (x_row < T) {
            const uint8_t* src_x = X_act + static_cast<size_t>(x_row) * K + x_col;
            cp_async_16(dst_x, src_x);
        } else {
            *reinterpret_cast<uint4*>(dst_x) = make_uint4(0, 0, 0, 0);
        }
        cp_async_commit();
    }

    // Main K-loop over tiles
    #pragma unroll 1
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        cp_async_wait<0>();
        __syncthreads();

        int next_buf = smem_buf ^ 1;
        int next_k_tile = k_tile + 1;

        if (next_k_tile < num_k_tiles) {
            int k_offset = next_k_tile * 128;
            #pragma unroll
            for (int step = 0; step < 8; ++step) {
                int r = (tid / 8) + step * 16;
                int c = (tid % 8) * 16;
                int g_row = global_n_base + r;
                const uint8_t* src_ptr = (g_row < V) ? (W_head + static_cast<size_t>(g_row) * K + k_offset + c) : nullptr;
                void* dst_ptr = &smem_W[next_buf][r * 128 + c];
                if (src_ptr) {
                    cp_async_16(dst_ptr, src_ptr);
                }
            }
            int x_row = tid / 8;
            int x_col = (tid % 8) * 16;
            void* dst_x = &smem_X[next_buf][x_row * 128 + x_col];
            if (x_row < T) {
                const uint8_t* src_x = X_act + static_cast<size_t>(x_row) * K + k_offset + x_col;
                cp_async_16(dst_x, src_x);
            } else {
                *reinterpret_cast<uint4*>(dst_x) = make_uint4(0, 0, 0, 0);
            }
            cp_async_commit();
        }

        // Tensor Core MMA: 4 sub-steps of K_sub=32
        #pragma unroll
        for (int k_sub = 0; k_sub < 4; ++k_sub) {
            uint32_t a_regs[4] = {0};
            int k_base = k_sub * 32;

            // Load A: row group (rows 0..7) and row group+8 (rows 8..15)
            const uint32_t* smem_x_r0 = reinterpret_cast<const uint32_t*>(&smem_X[smem_buf][group * 128 + k_base]);
            const uint32_t* smem_x_r1 = reinterpret_cast<const uint32_t*>(&smem_X[smem_buf][(group + 8) * 128 + k_base]);

            a_regs[0] = smem_x_r0[tid_in_grp];      // row group, cols 0..15
            a_regs[1] = smem_x_r1[tid_in_grp];      // row group+8, cols 0..15
            a_regs[2] = smem_x_r0[tid_in_grp + 4];  // row group, cols 16..31
            a_regs[3] = smem_x_r1[tid_in_grp + 4];  // row group+8, cols 16..31

            #pragma unroll
            for (int n_sub = 0; n_sub < 4; ++n_sub) {
                uint32_t b_regs[2] = {0};
                int col_in_tile = warp_id * 32 + n_sub * 8 + group;
                const uint32_t* smem_w_col = reinterpret_cast<const uint32_t*>(&smem_W[smem_buf][col_in_tile * 128 + k_base]);

                b_regs[0] = smem_w_col[tid_in_grp];     // col group, rows 0..15
                b_regs[1] = smem_w_col[tid_in_grp + 4]; // col group, rows 16..31

                mma_m16n8k32_fp8(accum[n_sub], a_regs, b_regs, accum[n_sub]);
            }
        }

        smem_buf = next_buf;
    }

    // Step 2: In-Register Online Softmax & Optional Logit Materialization
    const float temp_scale = (temperature > 0.0f) ? (1.0f / temperature) : 1.0f;
    const bool apply_gumbel = spec_gumbel && (temperature > 0.0f);
    const uint64_t effective_seed = (seed != 0) ? seed : 0x1337C0DEull;

    #pragma unroll
    for (int t = 0; t < 16; ++t) {
        if (t >= T) break;

        SoftmaxArgmaxTuple local_tuple;
        local_tuple.max_val = -1e30f;
        local_tuple.sum_exp = 0.0f;
        local_tuple.argmax_idx = -1;

        const float cur_sx = (scale_x_ptr != nullptr) ? smem_scale_x[t] : scale_x;

        // Threads with group == (t % 8) hold the computed logits for token t
        if (group == (t % 8)) {
            int d_idx0 = (t < 8) ? 0 : 2;
            int d_idx1 = (t < 8) ? 1 : 3;
            size_t row_offset = static_cast<size_t>(t) * V;

            #pragma unroll
            for (int n_sub = 0; n_sub < 4; ++n_sub) {
                int local_col0 = warp_id * 32 + n_sub * 8 + (tid_in_grp * 2);
                int local_col1 = local_col0 + 1;
                int col0 = global_n_base + local_col0;
                int col1 = col0 + 1;

                if (col0 < V) {
                    float cur_sw0 = (scale_w_ptr != nullptr) ? smem_scale_w[local_col0] : scale_w;
                    float raw_logit0 = accum[n_sub][d_idx0] * (cur_sw0 * cur_sx);
                    if (out_logits != nullptr) {
                        out_logits[row_offset + col0] = raw_logit0;
                    }
                    float logit0 = raw_logit0 * temp_scale;
                    if (apply_gumbel) {
                        float u0 = philox_uniform(effective_seed, step_counter + t, tid, static_cast<uint32_t>(col0));
                        float u0_clamped = fminf(fmaxf(u0, 1e-7f), 1.0f - 1e-7f);
                        float g0 = -__logf(-__logf(u0_clamped));
                        logit0 += g0;
                    }
                    if (logit0 > local_tuple.max_val) {
                        local_tuple.sum_exp = local_tuple.sum_exp * __expf(local_tuple.max_val - logit0) + 1.0f;
                        local_tuple.max_val = logit0;
                        local_tuple.argmax_idx = col0;
                    } else if (logit0 == local_tuple.max_val) {
                        local_tuple.sum_exp += 1.0f;
                        if (col0 < local_tuple.argmax_idx) local_tuple.argmax_idx = col0;
                    } else {
                        local_tuple.sum_exp += __expf(logit0 - local_tuple.max_val);
                    }
                }

                if (col1 < V) {
                    float cur_sw1 = (scale_w_ptr != nullptr) ? smem_scale_w[local_col1] : scale_w;
                    float raw_logit1 = accum[n_sub][d_idx1] * (cur_sw1 * cur_sx);
                    if (out_logits != nullptr) {
                        out_logits[row_offset + col1] = raw_logit1;
                    }
                    float logit1 = raw_logit1 * temp_scale;
                    if (apply_gumbel) {
                        float u1 = philox_uniform(effective_seed, step_counter + t, tid, static_cast<uint32_t>(col1));
                        float u1_clamped = fminf(fmaxf(u1, 1e-7f), 1.0f - 1e-7f);
                        float g1 = -__logf(-__logf(u1_clamped));
                        logit1 += g1;
                    }
                    if (logit1 > local_tuple.max_val) {
                        local_tuple.sum_exp = local_tuple.sum_exp * __expf(local_tuple.max_val - logit1) + 1.0f;
                        local_tuple.max_val = logit1;
                        local_tuple.argmax_idx = col1;
                    } else if (logit1 == local_tuple.max_val) {
                        local_tuple.sum_exp += 1.0f;
                        if (col1 < local_tuple.argmax_idx) local_tuple.argmax_idx = col1;
                    } else {
                        local_tuple.sum_exp += __expf(logit1 - local_tuple.max_val);
                    }
                }
            }
        }

        // Warp reduction across the active threads
        local_tuple = warp_reduce_softmax(local_tuple);

        if (lane_id == 0) {
            smem_warp_tuples[warp_id] = local_tuple;
        }
        __syncthreads();

        if (warp_id == 0) {
            SoftmaxArgmaxTuple block_tuple = (lane_id < 4) ? smem_warp_tuples[lane_id] : SoftmaxArgmaxTuple{-1e30f, 0.0f, -1};
            block_tuple = warp_reduce_softmax(block_tuple);

            if (lane_id == 0) {
                // Write block partial to L2 Scratchpad
                size_t scratch_idx = static_cast<size_t>(t) * gridDim.x + block_n;
                L2_scratchpad[scratch_idx] = block_tuple;
            }
        }
        __syncthreads();
    }
}

/**
 * L2 Scratchpad Epilogue Reduction Kernel
 * Merges block partials for each token row in L2 cache (< 2 us latency)
 */
__global__ void __launch_bounds__(256, 1) l2_scratchpad_epilogue_reduction_kernel(
    const SoftmaxArgmaxTuple* __restrict__ L2_scratchpad,
    int32_t* __restrict__ d_out_tokens,
    float* __restrict__ d_out_normalizers,
    const int32_t num_blocks,
    const int32_t T
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= T) return;

    const int tid = threadIdx.x;
    const SoftmaxArgmaxTuple* token_partials = L2_scratchpad + static_cast<size_t>(token_idx) * num_blocks;

    SoftmaxArgmaxTuple local_t{-1e30f, 0.0f, -1};

    // Stride over block partials
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        local_t = merge_softmax_tuples(local_t, token_partials[i]);
    }

    local_t = warp_reduce_softmax(local_t);

    __shared__ SoftmaxArgmaxTuple smem_warp_t[8];
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    if (lane_id == 0) {
        smem_warp_t[warp_id] = local_t;
    }
    __syncthreads();

    if (warp_id == 0) {
        SoftmaxArgmaxTuple global_t = (lane_id < 8) ? smem_warp_t[lane_id] : SoftmaxArgmaxTuple{-1e30f, 0.0f, -1};
        global_t = warp_reduce_softmax(global_t);
        if (lane_id == 0) {
            d_out_tokens[token_idx] = global_t.argmax_idx;
            if (d_out_normalizers != nullptr) {
                d_out_normalizers[token_idx] = global_t.sum_exp;
            }
        }
    }
}

/**
 * K25b Opt-In Fast Non-Gumbel LM Head FP8 GEMV + Parallel Argmax Kernel
 * XOR-swizzled smem_W/smem_X (zero SMEM bank conflicts) + single-barrier parallel argmax epilogue.
 */
__global__ void __launch_bounds__(128, 2) fused_lm_head_gemv_fast_argmax_kernel(
    const uint8_t* __restrict__ W_head,
    const uint8_t* __restrict__ X_act,
    SoftmaxArgmaxTuple* __restrict__ L2_scratchpad,
    float* __restrict__ out_logits,
    const int32_t T,
    const int32_t V,
    const int32_t K,
    const float temperature,
    const float scale_w,
    const float scale_x,
    const float* __restrict__ scale_w_ptr = nullptr,
    const float* __restrict__ scale_x_ptr = nullptr
) {
    __shared__ alignas(16) uint8_t smem_W[2][128 * 128];
    __shared__ alignas(16) uint8_t smem_X[2][16 * 128];
    __shared__ float smem_best_v[4][16];
    __shared__ int32_t smem_best_i[4][16];
    __shared__ float smem_scale_w[128];
    __shared__ float smem_scale_x[16];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int group = lane_id >> 2;       // 0..7
    const int tid_in_grp = lane_id & 3;   // 0..3
    const int block_n = blockIdx.x;
    const int global_n_base = block_n * 128;

    if (global_n_base >= V) return;

    if (scale_w_ptr != nullptr) {
        const int g_col = global_n_base + tid;
        smem_scale_w[tid] = (g_col < V) ? scale_w_ptr[g_col] : 1.0f;
    }
    if (scale_x_ptr != nullptr && tid < 16) {
        smem_scale_x[tid] = (tid < T) ? scale_x_ptr[tid] : 1.0f;
    }

    float accum[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            accum[i][j] = 0.0f;
        }
    }

    const int num_k_tiles = K / 128;
    int smem_buf = 0;

    // Prologue: Prefetch tile 0 with XOR bank-conflict-free swizzle
    {
        #pragma unroll
        for (int step = 0; step < 8; ++step) {
            const int r = (tid / 8) + step * 16;
            const int c = (tid % 8) * 16;
            const int c_sw = c ^ ((r & 7) << 4);
            const int g_row = global_n_base + r;
            const uint8_t* src_ptr = (g_row < V) ? (W_head + static_cast<size_t>(g_row) * K + c) : nullptr;
            if (src_ptr) {
                cp_async_16(&smem_W[0][r * 128 + c_sw], src_ptr);
            }
        }
        const int x_row = tid / 8;
        const int x_col = (tid % 8) * 16;
        const int x_sw = x_col ^ ((x_row & 7) << 4);
        void* dst_x = &smem_X[0][x_row * 128 + x_sw];
        if (x_row < T) {
            const uint8_t* src_x = X_act + static_cast<size_t>(x_row) * K + x_col;
            cp_async_16(dst_x, src_x);
        } else {
            *reinterpret_cast<uint4*>(dst_x) = make_uint4(0, 0, 0, 0);
        }
        cp_async_commit();
    }

    // Main K-loop over tiles
    #pragma unroll 1
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        cp_async_wait<0>();
        __syncthreads();

        const int next_buf = smem_buf ^ 1;
        const int next_k_tile = k_tile + 1;

        if (next_k_tile < num_k_tiles) {
            const int k_offset = next_k_tile * 128;
            #pragma unroll
            for (int step = 0; step < 8; ++step) {
                const int r = (tid / 8) + step * 16;
                const int c = (tid % 8) * 16;
                const int c_sw = c ^ ((r & 7) << 4);
                const int g_row = global_n_base + r;
                const uint8_t* src_ptr = (g_row < V) ? (W_head + static_cast<size_t>(g_row) * K + k_offset + c) : nullptr;
                if (src_ptr) {
                    cp_async_16(&smem_W[next_buf][r * 128 + c_sw], src_ptr);
                }
            }
            const int x_row = tid / 8;
            const int x_col = (tid % 8) * 16;
            const int x_sw = x_col ^ ((x_row & 7) << 4);
            void* dst_x = &smem_X[next_buf][x_row * 128 + x_sw];
            if (x_row < T) {
                const uint8_t* src_x = X_act + static_cast<size_t>(x_row) * K + k_offset + x_col;
                cp_async_16(dst_x, src_x);
            } else {
                *reinterpret_cast<uint4*>(dst_x) = make_uint4(0, 0, 0, 0);
            }
            cp_async_commit();
        }

        const int x_lane_sw = group << 4;
        #pragma unroll
        for (int k_sub = 0; k_sub < 4; ++k_sub) {
            uint32_t a_regs[4];
            const int k_base = k_sub * 32;
            const int k0 = k_base ^ x_lane_sw;
            const int k1 = (k_base + 16) ^ x_lane_sw;

            const uint32_t* smem_x_r0_0 = reinterpret_cast<const uint32_t*>(&smem_X[smem_buf][group * 128 + k0]);
            const uint32_t* smem_x_r0_1 = reinterpret_cast<const uint32_t*>(&smem_X[smem_buf][group * 128 + k1]);
            const uint32_t* smem_x_r1_0 = reinterpret_cast<const uint32_t*>(&smem_X[smem_buf][(group + 8) * 128 + k0]);
            const uint32_t* smem_x_r1_1 = reinterpret_cast<const uint32_t*>(&smem_X[smem_buf][(group + 8) * 128 + k1]);

            a_regs[0] = smem_x_r0_0[tid_in_grp];
            a_regs[1] = smem_x_r1_0[tid_in_grp];
            a_regs[2] = smem_x_r0_1[tid_in_grp];
            a_regs[3] = smem_x_r1_1[tid_in_grp];

            #pragma unroll
            for (int n_sub = 0; n_sub < 4; ++n_sub) {
                uint32_t b_regs[2];
                const int col_in_tile = warp_id * 32 + n_sub * 8 + group;
                const int w_lane_sw = (col_in_tile & 7) << 4;
                const uint32_t* smem_w_0 = reinterpret_cast<const uint32_t*>(&smem_W[smem_buf][col_in_tile * 128 + (k_base ^ w_lane_sw)]);
                const uint32_t* smem_w_1 = reinterpret_cast<const uint32_t*>(&smem_W[smem_buf][col_in_tile * 128 + ((k_base + 16) ^ w_lane_sw)]);

                b_regs[0] = smem_w_0[tid_in_grp];
                b_regs[1] = smem_w_1[tid_in_grp];

                mma_m16n8k32_fp8(accum[n_sub], a_regs, b_regs, accum[n_sub]);
            }
        }

        smem_buf = next_buf;
    }

    const float temp_scale = (temperature > 0.0f) ? (1.0f / temperature) : 1.0f;

    #pragma unroll
    for (int hi = 0; hi < 2; ++hi) {
        const int t = hi * 8 + group;
        float best_v = -1e30f;
        int32_t best_i = -1;

        if (t < T) {
            const int d0 = hi * 2;
            const int d1 = hi * 2 + 1;
            const float cur_sx = (scale_x_ptr != nullptr) ? smem_scale_x[t] : scale_x;
            const size_t row_offset = static_cast<size_t>(t) * V;

            #pragma unroll
            for (int n_sub = 0; n_sub < 4; ++n_sub) {
                const int local_col0 = warp_id * 32 + n_sub * 8 + (tid_in_grp * 2);
                const int local_col1 = local_col0 + 1;
                const int col0 = global_n_base + local_col0;
                const int col1 = col0 + 1;

                float raw0 = 0.0f;
                float raw1 = 0.0f;
                if (col0 < V) {
                    const float cur_sw0 = (scale_w_ptr != nullptr) ? smem_scale_w[local_col0] : scale_w;
                    raw0 = accum[n_sub][d0] * (cur_sw0 * cur_sx);
                    const float l0 = raw0 * temp_scale;
                    if (l0 > best_v || (l0 == best_v && (best_i < 0 || col0 < best_i))) {
                        best_v = l0;
                        best_i = col0;
                    }
                }
                if (col1 < V) {
                    const float cur_sw1 = (scale_w_ptr != nullptr) ? smem_scale_w[local_col1] : scale_w;
                    raw1 = accum[n_sub][d1] * (cur_sw1 * cur_sx);
                    const float l1 = raw1 * temp_scale;
                    if (l1 > best_v || (l1 == best_v && (best_i < 0 || col1 < best_i))) {
                        best_v = l1;
                        best_i = col1;
                    }
                }
                if (out_logits != nullptr) {
                    if (col1 < V && ((reinterpret_cast<uintptr_t>(out_logits + row_offset + col0) & 7u) == 0u)) {
                        *reinterpret_cast<float2*>(out_logits + row_offset + col0) = make_float2(raw0, raw1);
                    } else {
                        if (col0 < V) out_logits[row_offset + col0] = raw0;
                        if (col1 < V) out_logits[row_offset + col1] = raw1;
                    }
                }
            }
        }

        // Uniform warp-4 argmax reduction across lanes sharing the same token group
        #pragma unroll
        for (int mask = 2; mask >= 1; mask >>= 1) {
            const float pv = __shfl_xor_sync(0xffffffff, best_v, mask);
            const int32_t pi = __shfl_xor_sync(0xffffffff, best_i, mask);
            if (pv > best_v || (pv == best_v && pi >= 0 && (best_i < 0 || pi < best_i))) {
                best_v = pv;
                best_i = pi;
            }
        }
        if (t < T && tid_in_grp == 0) {
            smem_best_v[warp_id][t] = best_v;
            smem_best_i[warp_id][t] = best_i;
        }
    }

    __syncthreads();
    if (tid < T) {
        float fv = smem_best_v[0][tid];
        int32_t fi = smem_best_i[0][tid];
        #pragma unroll
        for (int w = 1; w < 4; ++w) {
            const float wv = smem_best_v[w][tid];
            const int32_t wi = smem_best_i[w][tid];
            if (wv > fv || (wv == fv && wi >= 0 && (fi < 0 || wi < fi))) {
                fv = wv;
                fi = wi;
            }
        }
        L2_scratchpad[static_cast<size_t>(tid) * gridDim.x + block_n] = SoftmaxArgmaxTuple{fv, 0.0f, fi};
    }
}

inline cudaError_t launch_fused_lm_head_gemv_fast_argmax(
    const uint8_t* W_head,
    const uint8_t* X_act,
    SoftmaxArgmaxTuple* L2_scratchpad,
    int32_t* d_out_tokens,
    float* d_out_normalizers,
    float* out_logits,
    int32_t T,
    int32_t V,
    int32_t K,
    float temperature,
    float scale_w,
    float scale_x,
    const float* scale_w_ptr,
    const float* scale_x_ptr,
    cudaStream_t stream = nullptr
) {
    if (T <= 0 || V <= 0 || K <= 0) return cudaSuccess;
    const int num_blocks = (V + 127) / 128;

    fused_lm_head_gemv_fast_argmax_kernel<<<num_blocks, 128, 0, stream>>>(
        W_head, X_act, L2_scratchpad, out_logits, T, V, K, temperature, scale_w, scale_x,
        scale_w_ptr, scale_x_ptr
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    l2_scratchpad_epilogue_reduction_kernel<<<T, 256, 0, stream>>>(
        L2_scratchpad, d_out_tokens, d_out_normalizers, num_blocks, T
    );

    return cudaGetLastError();
}

inline cudaError_t launch_fused_lm_head_gemv_softmax(
    const uint8_t* W_head,
    const uint8_t* X_act,
    SoftmaxArgmaxTuple* L2_scratchpad,
    int32_t* d_out_tokens,
    float* d_out_normalizers,
    float* out_logits,
    int32_t T,
    int32_t V,
    int32_t K,
    float temperature,
    float scale_w,
    float scale_x,
    const float* scale_w_ptr,
    const float* scale_x_ptr,
    bool spec_gumbel,
    uint64_t seed,
    uint64_t step_counter,
    cudaStream_t stream = nullptr
) {
    if (T <= 0 || V <= 0 || K <= 0) return cudaSuccess;
    const int num_blocks = (V + 127) / 128;

    fused_lm_head_gemv_online_softmax_kernel<<<num_blocks, 128, 0, stream>>>(
        W_head, X_act, L2_scratchpad, out_logits, T, V, K, temperature, scale_w, scale_x,
        scale_w_ptr, scale_x_ptr, spec_gumbel, seed, step_counter
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    l2_scratchpad_epilogue_reduction_kernel<<<T, 256, 0, stream>>>(
        L2_scratchpad, d_out_tokens, d_out_normalizers, num_blocks, T
    );

    return cudaGetLastError();
}

inline cudaError_t launch_fused_lm_head_gemv_softmax(
    const uint8_t* W_head,
    const uint8_t* X_act,
    SoftmaxArgmaxTuple* L2_scratchpad,
    int32_t* d_out_tokens,
    float* d_out_normalizers,
    float* out_logits,
    int32_t T,
    int32_t V,
    int32_t K,
    float temperature,
    float scale_w,
    float scale_x,
    const float* scale_w_ptr = nullptr,
    const float* scale_x_ptr = nullptr,
    cudaStream_t stream = nullptr
) {
    bool spec_gumbel = false;
    if (out_logits == nullptr) {
        const char* eg = std::getenv("STRATA_SPEC_GUMBEL");
        const char* ez = std::getenv("STRATA_ZERO_DRAM_SAMPLER");
        if ((eg != nullptr && eg[0] == '1') || (ez != nullptr && ez[0] == '1')) {
            spec_gumbel = true;
        }
    }
    return launch_fused_lm_head_gemv_softmax(
        W_head, X_act, L2_scratchpad, d_out_tokens, d_out_normalizers, out_logits,
        T, V, K, temperature, scale_w, scale_x, scale_w_ptr, scale_x_ptr,
        spec_gumbel, 0, 0, stream
    );
}

inline cudaError_t launch_fused_lm_head_gemv_softmax(
    const uint8_t* W_head,
    const uint8_t* X_act,
    SoftmaxArgmaxTuple* L2_scratchpad,
    int32_t* d_out_tokens,
    float* d_out_normalizers,
    int32_t T,
    int32_t V,
    int32_t K,
    float temperature,
    float scale_w,
    float scale_x,
    const float* scale_w_ptr = nullptr,
    const float* scale_x_ptr = nullptr,
    cudaStream_t stream = nullptr
) {
    return launch_fused_lm_head_gemv_softmax(
        W_head, X_act, L2_scratchpad, d_out_tokens, d_out_normalizers, nullptr,
        T, V, K, temperature, scale_w, scale_x, scale_w_ptr, scale_x_ptr, stream
    );
}

} // namespace kernels
} // namespace strata
