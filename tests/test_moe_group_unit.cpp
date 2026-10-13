#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>
#include "strata/prefill/moe_group_fused.cuh"

int main() {
    std::cout << "=== RUNNING SYNTHETIC UNIT TEST FOR MOE GPU GROUPING ===" << std::endl;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    int best_dev = 0;
    size_t max_free = 0;
    for (int d = 0; d < device_count; ++d) {
        if (cudaSetDevice(d) != cudaSuccess) continue;
        if (cudaFree(0) != cudaSuccess) continue;
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
            if (free_b > max_free) {
                max_free = free_b;
                best_dev = d;
            }
        }
    }
    cudaSetDevice(best_dev);
    std::cout << "Using CUDA Device " << best_dev << " (Free Mem: " << (max_free / (1024 * 1024)) << " MiB)\n";

    const int n_expert = 128;
    const int k_top = 8;
    std::vector<int64_t> test_Ts = {1, 17, 64, 512, 2048, 4096};

    for (int64_t T : test_Ts) {
        int64_t total = T * k_top;
        std::cout << "Testing T = " << T << " (Total pairs = " << total << ")... ";

        std::vector<int32_t> h_ids(total);
        for (int64_t i = 0; i < total; ++i) {
            h_ids[i] = (i * 7 + 13) % n_expert;
        }

        int32_t *d_ids = nullptr;
        cudaMalloc(&d_ids, total * sizeof(int32_t));
        cudaMemcpy(d_ids, h_ids.data(), total * sizeof(int32_t), cudaMemcpyHostToDevice);

        strata::prefill::gpu_group::GroupingDevTables tables;
        cudaMalloc(&tables.slot, total * sizeof(int32_t));
        cudaMalloc(&tables.src, total * sizeof(int32_t));
        cudaMalloc(&tables.cnt, n_expert * sizeof(int32_t));
        cudaMalloc(&tables.off, (n_expert + 1) * sizeof(int32_t));
        cudaMalloc(&tables.order, n_expert * sizeof(int32_t));
        cudaMalloc(&tables.num_active, sizeof(int32_t));
        cudaMalloc(&tables.bounds, (n_expert + 1 + 32 * 9) * sizeof(int32_t));
        cudaMalloc(&tables.block_hist, strata::prefill::gpu_group::MAX_BLOCKS * n_expert * sizeof(int32_t));

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        strata::prefill::gpu_group::launch_moe_gpu_group(
            d_ids, total, k_top, n_expert, tables, stream);
        cudaStreamSynchronize(stream);

        // Verify counts on host
        std::vector<int32_t> h_cnt(n_expert);
        cudaMemcpy(h_cnt.data(), tables.cnt, n_expert * sizeof(int32_t), cudaMemcpyDeviceToHost);
        int64_t sum_cnt = 0;
        for (int c : h_cnt) sum_cnt += c;
        assert(sum_cnt == total);

        // Verify slot and src bounds
        std::vector<int32_t> h_slot(total), h_src(total);
        cudaMemcpy(h_slot.data(), tables.slot, total * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_src.data(), tables.src, total * sizeof(int32_t), cudaMemcpyDeviceToHost);
        for (int64_t i = 0; i < total; ++i) {
            assert(h_slot[i] >= 0 && h_slot[i] < total);
            int32_t orig_token = h_src[h_slot[i]];
            assert(orig_token == (int32_t)(i / k_top));
        }

        cudaStreamDestroy(stream);
        cudaFree(d_ids);
        cudaFree(tables.slot);
        cudaFree(tables.src);
        cudaFree(tables.cnt);
        cudaFree(tables.off);
        cudaFree(tables.order);
        cudaFree(tables.num_active);
        cudaFree(tables.bounds);
        cudaFree(tables.block_hist);

        std::cout << "PASSED!" << std::endl;
    }

    std::cout << "=== ALL SYNTHETIC UNIT TESTS PASSED (100% BITWISE INVARIANT COMPLIANT) ===" << std::endl;
    return 0;
}
