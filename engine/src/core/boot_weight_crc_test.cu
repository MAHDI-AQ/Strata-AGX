// src/core/boot_weight_crc_test.cu - CUDA boot weight CRC verification + fail-closed 1-bit VRAM flip test.
#include "strata/core/boot_weight_crc.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

__global__ void flip_vram_bit_kernel(uint8_t* ptr, uint64_t byte_offset, uint8_t bit_mask) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ptr[byte_offset] ^= bit_mask;
    }
}

}  // namespace

int main() {
    int n_devs = 0;
    if (cudaGetDeviceCount(&n_devs) != cudaSuccess || n_devs <= 0) {
        std::printf("SKIPPED: no CUDA device available\n");
        return 77;
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        std::printf("SKIPPED: cudaSetDevice(0) failed\n");
        return 77;
    }

    strata::core::boot_crc_clear_all();

    const uint64_t big_sz = (4ULL << 20) + 13ULL;
    const uint64_t mis_sz = 2055ULL;
    std::vector<uint8_t> h_big((size_t) big_sz);
    std::vector<uint8_t> h_mis((size_t) mis_sz);
    for (size_t i = 0; i < h_big.size(); ++i) h_big[i] = static_cast<uint8_t>((i * 29 + 3) ^ (i >> 11));
    for (size_t i = 0; i < h_mis.size(); ++i) h_mis[i] = static_cast<uint8_t>((i * 97 + 11) ^ (i >> 4));

    uint8_t* d_big = nullptr;
    uint8_t* d_mis = nullptr;
    if (cudaMalloc(&d_big, (size_t) big_sz) != cudaSuccess ||
        cudaMalloc(&d_mis, (size_t) (mis_sz + 16)) != cudaSuccess) {
        if (d_big) cudaFree(d_big);
        if (d_mis) cudaFree(d_mis);
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }

    const uint8_t* h_mis_unaligned = h_mis.data();
    uint8_t* d_mis_unaligned = d_mis + 3;  // deliberately unaligned in both device and tail

    if (cudaMemcpy(d_big, h_big.data(), (size_t) big_sz, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_mis_unaligned, h_mis_unaligned, (size_t) mis_sz, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(d_big);
        cudaFree(d_mis);
        std::fprintf(stderr, "FAIL: cudaMemcpy failed\n");
        return 1;
    }

    strata::core::boot_crc_record(0, "wt", "test.big", d_big, big_sz, h_big.data());
    strata::core::boot_crc_record(0, "dense", "test.mis", d_mis_unaligned, mis_sz, h_mis_unaligned);

    std::string err;
    if (!strata::core::boot_weight_crc_verify_all(err)) {
        cudaFree(d_big);
        cudaFree(d_mis);
        std::fprintf(stderr, "FAIL: intact VRAM verification failed: %s\n", err.c_str());
        return 1;
    }

    // Flip 1 single bit in the 3rd slice tail (byte big_sz - 1) and verify fail-closed detection.
    flip_vram_bit_kernel<<<1, 1>>>(d_big, big_sz - 1, 0x04);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(d_big);
        cudaFree(d_mis);
        std::fprintf(stderr, "FAIL: flip_vram_bit_kernel sync failed\n");
        return 1;
    }

    std::string fail_err;
    if (strata::core::boot_weight_crc_verify_all(fail_err)) {
        cudaFree(d_big);
        cudaFree(d_mis);
        std::fprintf(stderr, "FAIL: 1-bit VRAM flip was NOT caught!\n");
        return 1;
    }
    if (fail_err.find("CUDA0") == std::string::npos || fail_err.find("wt:test.big#2") == std::string::npos) {
        cudaFree(d_big);
        cudaFree(d_mis);
        std::fprintf(stderr, "FAIL: unexpected mismatch error message: %s\n", fail_err.c_str());
        return 1;
    }

    strata::core::boot_crc_clear_all();
    cudaFree(d_big);
    cudaFree(d_mis);
    std::printf("boot_weight_crc_cuda_test PASS (intact PASS + 1-bit VRAM flip fail-closed: %s)\n", fail_err.c_str());
    return 0;
}
