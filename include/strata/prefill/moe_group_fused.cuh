#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace strata::prefill::gpu_group {

constexpr int MMQ_GROUP = 16;
constexpr int MAX_BLOCKS = 64;

struct GroupingDevTables {
    int32_t* slot;          // [T * K]: target slot for each (token, k)
    int32_t* src;           // [T * K]: source token index for each gathered slot
    int32_t* cnt;           // [n_expert]: token count per expert
    int32_t* off;           // [n_expert + 1]: prefix sum offsets
    int32_t* order;         // [n_expert]: list of active experts (cnt > 0)
    int32_t* num_active;    // [1]: count of active experts
    int32_t* bounds;        // MMQ bounds array
    int32_t* block_hist;    // [MAX_BLOCKS * n_expert]: per-block private histograms
};

void launch_moe_gpu_group(
    const int32_t* ids_dev,
    int64_t total_tokens,   // T * K
    int k_top,
    int n_expert,
    GroupingDevTables tables,
    cudaStream_t stream);

} // namespace strata::prefill::gpu_group
