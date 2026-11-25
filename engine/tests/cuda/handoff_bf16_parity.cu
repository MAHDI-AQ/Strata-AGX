// tests/cuda/handoff_bf16_parity.cu - Verification test for Path C BF16 inter-stage activation handoff
#include "strata/kernels/cuda/handoff_bf16.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

bool ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::printf("FAIL: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

uint16_t host_f32_to_bf16_rn(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return static_cast<uint16_t>((u >> 16) | 0x0040u);
    }
    const uint32_t bias = ((u >> 16) & 1u) + 0x7fffu;
    return static_cast<uint16_t>((u + bias) >> 16);
}

float host_bf16_to_f32(uint16_t b) {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

bool check_shape_and_offsets(
    int64_t count,
    int src_offset,
    int dst_offset,
    cudaStream_t st,
    bool test_pinned_bounce,
    const char* label
) {
    const int64_t pad = 16;
    const int64_t total_f = count + pad * 2;
    std::vector<float> h_src_full((size_t) total_f, 0.0f);
    std::mt19937 rng(20261125u + (unsigned) count + (unsigned) src_offset * 17u);
    std::normal_distribution<float> nd(0.0f, 3.5f);

    float* h_src = h_src_full.data() + src_offset;
    for (int64_t i = 0; i < count; ++i) {
        h_src[i] = nd(rng);
    }
    if (count > 0) h_src[0] = 32768.0f;
    if (count > 5) h_src[5] = -65504.0f;
    if (count > 42) h_src[42] = 120.0f;
    if (count > 7) h_src[7] = -0.0f;

    float* d_src_base = nullptr;
    float* d_dst_base = nullptr;
    uint16_t* d_bf16_base = nullptr;
    uint16_t* h_pinned = nullptr;

    if (!ok(cudaMalloc((void**) &d_src_base, (size_t) total_f * sizeof(float)), "bf16 src") ||
        !ok(cudaMalloc((void**) &d_dst_base, (size_t) total_f * sizeof(float)), "bf16 dst") ||
        !ok(cudaMalloc((void**) &d_bf16_base, (size_t) total_f * sizeof(uint16_t)), "bf16 dev")) {
        return false;
    }
    if (test_pinned_bounce &&
        !ok(cudaHostAlloc((void**) &h_pinned, (size_t) count * sizeof(uint16_t),
                          cudaHostAllocMapped | cudaHostAllocPortable), "bf16 pinned")) {
        cudaFree(d_src_base); cudaFree(d_dst_base); cudaFree(d_bf16_base);
        return false;
    }

    cudaMemsetAsync(d_dst_base, 0xA5, (size_t) total_f * sizeof(float), st);
    cudaMemsetAsync(d_bf16_base, 0x5A, (size_t) total_f * sizeof(uint16_t), st);
    cudaMemcpyAsync(d_src_base, h_src_full.data(), (size_t) total_f * sizeof(float), cudaMemcpyHostToDevice, st);

    float* d_src = d_src_base + src_offset;
    uint16_t* d_bf16 = d_bf16_base + dst_offset;
    float* d_dst = d_dst_base + src_offset;

    strata::kernels::cuda::pack_activations_bf16(d_src, d_bf16, count, st);
    if (test_pinned_bounce && count > 0) {
        cudaMemcpyAsync(h_pinned, d_bf16, (size_t) count * sizeof(uint16_t), cudaMemcpyDeviceToHost, st);
        cudaMemsetAsync(d_bf16, 0, (size_t) count * sizeof(uint16_t), st);
        cudaMemcpyAsync(d_bf16, h_pinned, (size_t) count * sizeof(uint16_t), cudaMemcpyHostToDevice, st);
    }
    strata::kernels::cuda::unpack_activations_bf16(d_bf16, d_dst, count, st);

    if (!ok(cudaStreamSynchronize(st), "bf16 sync")) return false;

    std::vector<float> h_dst_full((size_t) total_f);
    std::vector<uint16_t> h_bf16_full((size_t) total_f);
    cudaMemcpy(h_dst_full.data(), d_dst_base, (size_t) total_f * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bf16_full.data(), d_bf16_base, (size_t) total_f * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    int mismatch = 0;
    double max_rel = 0.0;
    for (int64_t i = 0; i < count; ++i) {
        const uint16_t exp_b = host_f32_to_bf16_rn(h_src[i]);
        const uint16_t got_b = h_bf16_full[(size_t) (dst_offset + i)];
        const float exp_f = host_bf16_to_f32(exp_b);
        const float got_f = h_dst_full[(size_t) (src_offset + i)];
        uint32_t exp_bits = 0, got_bits = 0;
        std::memcpy(&exp_bits, &exp_f, 4);
        std::memcpy(&got_bits, &got_f, 4);
        if (exp_b != got_b || exp_bits != got_bits) {
            if (mismatch < 5) {
                std::printf("FAIL: %s idx=%lld exp_b=0x%04x got_b=0x%04x\n",
                            label, (long long) i, exp_b, got_b);
            }
            ++mismatch;
        }
        const float ax = std::fabs(h_src[i]);
        if (ax > 1e-6f) {
            const double rel = std::fabs((double) got_f - (double) h_src[i]) / (double) ax;
            if (rel > max_rel) max_rel = rel;
        }
    }

    // Check sentinels untouched around slice
    bool sentinel_ok = true;
    for (int64_t i = count + src_offset; i < total_f; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, &h_dst_full[(size_t) i], 4);
        if (bits != 0xA5A5A5A5u) { sentinel_ok = false; break; }
    }
    for (int64_t i = count + dst_offset; i < total_f; ++i) {
        if (h_bf16_full[(size_t) i] != 0x5A5Au) { sentinel_ok = false; break; }
    }

    if (h_pinned) cudaFreeHost(h_pinned);
    cudaFree(d_src_base);
    cudaFree(d_dst_base);
    cudaFree(d_bf16_base);

    if (mismatch == 0 && sentinel_ok && max_rel <= 0.0045) {
        std::printf("PASS: %-24s count=%-7lld max_rel=%.6f (IEEE-754 RNE bit-exact)\n",
                    label, (long long) count, max_rel);
        return true;
    }
    std::printf("FAIL: %-24s count=%-7lld mismatch=%d sentinel_ok=%d max_rel=%.6f\n",
                label, (long long) count, mismatch, sentinel_ok ? 1 : 0, max_rel);
    return false;
}

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 1) {
        std::printf("no CUDA device (skipped)\n");
        return 77;
    }
    cudaStream_t st = nullptr;
    if (!ok(cudaStreamCreate(&st), "stream")) return 1;

    bool all_ok = true;
    all_ok &= check_shape_and_offsets(10240, 0, 0, st, true, "T1_D10240_pinned");
    all_ok &= check_shape_and_offsets(32 * 10240, 0, 0, st, true, "T32_D10240_pinned");
    all_ok &= check_shape_and_offsets(12805, 0, 0, st, false, "tail_12805_vec_plus_1");
    all_ok &= check_shape_and_offsets(7, 0, 0, st, false, "tail_7");
    all_ok &= check_shape_and_offsets(1, 0, 0, st, false, "tail_1");
    all_ok &= check_shape_and_offsets(10243, 1, 1, st, false, "unaligned_src1_dst1");
    all_ok &= check_shape_and_offsets(10243, 0, 1, st, false, "aligned_src0_dst1");

    cudaStreamDestroy(st);
    std::printf("handoff_bf16_parity: %s\n", all_ok ? "OK" : "FAILED");
    return all_ok ? 0 : 1;
}
