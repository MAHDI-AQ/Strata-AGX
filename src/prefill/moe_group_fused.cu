#include "strata/prefill/moe_group_fused.cuh"
#include <cstdio>
#include <algorithm>

namespace strata::prefill::gpu_group {

// ============================================================================
// 1. Single-CTA Fast-Path Kernel (T * K <= 1024)
// ============================================================================
__global__ void moe_fused_small_k_kernel(
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
    int32_t* smem_cnt  = smem;                     // [n_expert]
    int32_t* smem_off  = smem + n_expert;          // [n_expert + 1]
    int32_t* smem_fill = smem + 2 * n_expert + 1;  // [n_expert]

    const int tid = threadIdx.x;
    for (int e = tid; e < n_expert; e += blockDim.x) {
        smem_cnt[e] = 0;
    }
    __syncthreads();

    // Cooperative Histogram
    for (int64_t i = tid; i < total_tokens; i += blockDim.x) {
        int e = ids[i];
        if ((unsigned)e < (unsigned)n_expert) {
            atomicAdd(&smem_cnt[e], 1);
        }
    }
    __syncthreads();

    // Prefix scan & MMQ bounds on thread 0
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

    for (int e = tid; e < n_expert; e += blockDim.x) {
        if (global_cnt) global_cnt[e] = smem_cnt[e];
        if (off_out) off_out[e] = smem_off[e];
    }
    if (tid == 0 && off_out) {
        off_out[n_expert] = smem_off[n_expert];
    }
    __syncthreads();

    // Stable sequential scatter within CTA
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

// ============================================================================
// 2. Multi-CTA Parallel Two-Pass Block-Binning Kernels (T * K > 1024)
// ============================================================================

// Pass 1: Compute private block histogram
__global__ void moe_block_hist_kernel(
    const int32_t* __restrict__ ids,
    int64_t total_tokens,
    int n_expert,
    int64_t chunk_size,
    int32_t* __restrict__ block_hist_out)
{
    extern __shared__ int32_t smem_cnt[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    for (int e = tid; e < n_expert; e += blockDim.x) {
        smem_cnt[e] = 0;
    }
    __syncthreads();

    const int64_t start_idx = bid * chunk_size;
    const int64_t end_idx = min(start_idx + chunk_size, total_tokens);

    // Warp-aggregated atomic counting
    for (int64_t i = start_idx + tid; i < end_idx; i += blockDim.x) {
        int e = ids[i];
        if ((unsigned)e < (unsigned)n_expert) {
            uint32_t mask = __match_any_sync(0xffffffff, e);
            int leader = __ffs(mask) - 1;
            int lane = tid & 31;
            if (lane == leader) {
                atomicAdd(&smem_cnt[e], __popc(mask));
            }
        }
    }
    __syncthreads();

    // Write out block histogram
    int32_t* dst_hist = block_hist_out + bid * n_expert;
    for (int e = tid; e < n_expert; e += blockDim.x) {
        dst_hist[e] = smem_cnt[e];
    }
}

// Pass 2: Exclusive scan across blocks and experts + Metadata generation
__global__ void moe_scan_and_metadata_kernel(
    int num_blocks,
    int n_expert,
    int64_t total_tokens,
    int32_t* __restrict__ block_hist,
    int32_t* __restrict__ global_cnt,
    int32_t* __restrict__ off_out,
    int32_t* __restrict__ order_out,
    int32_t* __restrict__ num_active_out,
    int32_t* __restrict__ bounds_out)
{
    extern __shared__ int32_t smem[];
    int32_t* smem_cnt = smem;              // [n_expert]
    int32_t* smem_off = smem + n_expert;   // [n_expert + 1]

    const int tid = threadIdx.x;

    // 1. Compute total expert counts across all blocks
    for (int e = tid; e < n_expert; e += blockDim.x) {
        int sum = 0;
        for (int b = 0; b < num_blocks; ++b) {
            sum += block_hist[b * n_expert + e];
        }
        smem_cnt[e] = sum;
        if (global_cnt) global_cnt[e] = sum;
    }
    __syncthreads();

    // 2. Thread 0 scans expert totals and generates metadata
    if (tid == 0) {
        int running_sum = 0;
        int active_count = 0;
        for (int e = 0; e < n_expert; ++e) {
            int c = smem_cnt[e];
            smem_off[e] = running_sum;
            if (c > 0) {
                if (order_out) order_out[active_count] = e;
                active_count++;
            }
            running_sum += c;
        }
        smem_off[n_expert] = running_sum;
        if (num_active_out) *num_active_out = active_count;

        // Build MMQ bounds
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

    // Write out off_out
    for (int e = tid; e < n_expert; e += blockDim.x) {
        if (off_out) off_out[e] = smem_off[e];
    }
    if (tid == 0 && off_out) {
        off_out[n_expert] = smem_off[n_expert];
    }

    // 3. Convert block_hist[b][e] into exclusive block start offsets in-place
    // block_hist[b][e] becomes off[e] + sum_{j=0}^{b-1} block_hist[j][e]
    for (int e = tid; e < n_expert; e += blockDim.x) {
        int curr = smem_off[e];
        for (int b = 0; b < num_blocks; ++b) {
            int count = block_hist[b * n_expert + e];
            block_hist[b * n_expert + e] = curr;
            curr += count;
        }
    }
}

// Pass 3: Parallel Deterministic Scatter
__global__ void moe_block_scatter_kernel(
    const int32_t* __restrict__ ids,
    int64_t total_tokens,
    int k_top,
    int n_expert,
    int64_t chunk_size,
    const int32_t* __restrict__ block_base_offsets,
    int32_t* __restrict__ slot_out,
    int32_t* __restrict__ src_out)
{
    extern __shared__ int32_t smem_base[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int32_t* my_bases = block_base_offsets + bid * n_expert;
    for (int e = tid; e < n_expert; e += blockDim.x) {
        smem_base[e] = my_bases[e];
    }
    __syncthreads();

    const int64_t start_idx = bid * chunk_size;
    const int64_t end_idx = min(start_idx + chunk_size, total_tokens);

    // Intra-block sequential dispatch across chunk
    // Thread 0 in each block ensures 100.000% monotonic sequence order within the block
    if (tid == 0) {
        for (int64_t i = start_idx; i < end_idx; ++i) {
            int e = ids[i];
            if ((unsigned)e < (unsigned)n_expert) {
                int p = smem_base[e]++;
                slot_out[i] = p;
                src_out[p] = (int32_t)(i / k_top);
            }
        }
    }
}

// ============================================================================
// Driver Dispatch Function
// ============================================================================
void launch_moe_gpu_group(
    const int32_t* ids_dev,
    int64_t total_tokens,
    int k_top,
    int n_expert,
    GroupingDevTables tables,
    cudaStream_t stream)
{
    if (total_tokens <= 0 || n_expert <= 0) return;

    if (total_tokens <= MULTI_CTA_TOKEN_THRESHOLD || tables.block_hist == nullptr) {
        // Path A: Low latency single CTA (micro-chunk)
        size_t smem_bytes = (3 * n_expert + 1) * sizeof(int32_t);
        moe_fused_small_k_kernel<<<1, 256, smem_bytes, stream>>>(
            ids_dev, total_tokens, k_top, n_expert,
            tables.slot, tables.src, tables.cnt, tables.off,
            tables.order, tables.num_active, tables.bounds);
    } else {
        // Path B: Multi-CTA Parallel Two-Pass (large prefill, saturating SMs)
        constexpr int TPB = 256;
        int num_blocks = (int)((total_tokens + 511) / 512);
        num_blocks = std::max(1, std::min(num_blocks, MAX_BLOCKS));
        int64_t chunk_size = (total_tokens + num_blocks - 1) / num_blocks;

        // Pass 1: Block Histograms
        size_t smem_hist = n_expert * sizeof(int32_t);
        moe_block_hist_kernel<<<num_blocks, TPB, smem_hist, stream>>>(
            ids_dev, total_tokens, n_expert, chunk_size, tables.block_hist);

        // Pass 2: Global Prefix Scan & Metadata
        size_t smem_scan = (2 * n_expert + 1) * sizeof(int32_t);
        moe_scan_and_metadata_kernel<<<1, TPB, smem_scan, stream>>>(
            num_blocks, n_expert, total_tokens, tables.block_hist,
            tables.cnt, tables.off, tables.order, tables.num_active, tables.bounds);

        // Pass 3: Multi-CTA Parallel Scatter
        size_t smem_scatter = n_expert * sizeof(int32_t);
        moe_block_scatter_kernel<<<num_blocks, TPB, smem_scatter, stream>>>(
            ids_dev, total_tokens, k_top, n_expert, chunk_size,
            tables.block_hist, tables.slot, tables.src);
    }
}

} // namespace strata::prefill::gpu_group
