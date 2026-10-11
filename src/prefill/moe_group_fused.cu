#include "strata/prefill/moe_group_fused.cuh"
#include <cstdio>

namespace strata::prefill::gpu_group {

// ============================================================================
// Single-CTA Deterministic Fused Kernel (Handles all total_tokens <= 65536)
// Guarantees 100.000% deterministic bitwise matching with host CPU loop
// ============================================================================
__global__ void moe_fused_cta_kernel(
    const int32_t* __restrict__ ids,
    int64_t total_tokens,
    int k_top,
    int n_expert,
    int32_t* __restrict__ slot_out,
    int32_t* __restrict__ src_out,
    int32_t* __restrict__ global_cnt,
    int32_t* __restrict__ off_out,
    int32_t* __restrict__ order_out,
    int32_t* __restrict__ num_active_out,
    int32_t* __restrict__ bounds_out)
{
    extern __shared__ int32_t smem[];
    int32_t* smem_cnt = smem;                   // [n_expert]
    int32_t* smem_off = smem + n_expert;        // [n_expert + 1]
    int32_t* smem_fill = smem + 2 * n_expert + 1; // [n_expert]

    const int tid = threadIdx.x;
    for (int e = tid; e < n_expert; e += blockDim.x) {
        smem_cnt[e] = 0;
    }
    __syncthreads();

    // 1. Cooperative Histogram
    for (int64_t i = tid; i < total_tokens; i += blockDim.x) {
        int e = ids[i];
        if ((unsigned)e < (unsigned)n_expert) {
            atomicAdd(&smem_cnt[e], 1);
        }
    }
    __syncthreads();

    // 2. Prefix scan & MMQ bounds on thread 0
    if (tid == 0) {
        int running_sum = 0;
        int active_count = 0;
        for (int e = 0; e < n_expert; ++e) {
            int c = smem_cnt[e];
            smem_off[e] = running_sum;
            smem_fill[e] = running_sum;
            if (c > 0) {
                if (order_out) order_out[active_count] = e;
                active_count++;
            }
            running_sum += c;
        }
        smem_off[n_expert] = running_sum;
        if (num_active_out) *num_active_out = active_count;

        // Build MMQ bounds if requested
        if (bounds_out && order_out) {
            const int n = active_count;
            const int ng = (n + MMQ_GROUP - 1) / MMQ_GROUP;
            for (int j = 0; j < n; ++j) {
                bounds_out[j] = smem_off[order_out[j]];
            }
            bounds_out[n] = (int32_t)total_tokens;
            for (int g = 0; g < ng; ++g) {
                int g_base = bounds_out[g * MMQ_GROUP];
                for (int i = 0; i <= MMQ_GROUP; ++i) {
                    int idx = (g * MMQ_GROUP + i < n) ? (g * MMQ_GROUP + i) : n;
                    bounds_out[n + 1 + g * (MMQ_GROUP + 1) + i] = bounds_out[idx] - g_base;
                }
            }
        }
    }
    __syncthreads();

    // Write back global_cnt and off_out if provided
    for (int e = tid; e < n_expert; e += blockDim.x) {
        if (global_cnt) global_cnt[e] = smem_cnt[e];
        if (off_out) off_out[e] = smem_off[e];
    }
    if (tid == 0 && off_out) {
        off_out[n_expert] = smem_off[n_expert];
    }
    __syncthreads();

    // 3. Stable sequential scatter within CTA
    // Executed on thread 0 to guarantee 100.000% monotonic original sequence order
    if (tid == 0) {
        for (int64_t i = 0; i < total_tokens; ++i) {
            int e = ids[i];
            if ((unsigned)e < (unsigned)n_expert) {
                int p = smem_fill[e]++;
                slot_out[i] = p;
                src_out[p] = (int32_t)(i / k_top);
            }
        }
    }
}

void launch_moe_gpu_group(
    const int32_t* ids_dev,
    int64_t total_tokens,
    int k_top,
    int n_expert,
    GroupingDevTables tables,
    cudaStream_t stream)
{
    if (total_tokens <= 0 || n_expert <= 0) return;

    if (tables.cnt) cudaMemsetAsync(tables.cnt, 0, n_expert * sizeof(int32_t), stream);

    size_t smem_bytes = (3 * n_expert + 1) * sizeof(int32_t);
    moe_fused_cta_kernel<<<1, 256, smem_bytes, stream>>>(
        ids_dev, total_tokens, k_top, n_expert,
        tables.slot, tables.src, tables.cnt, tables.off,
        tables.order, tables.num_active, tables.bounds);
}

} // namespace strata::prefill::gpu_group
