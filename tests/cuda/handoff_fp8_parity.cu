// tests/cuda/handoff_fp8_parity.cu - Verification test for Path B Block-Microscaled FP8 Handoff Compression
#include "strata/kernels/cuda/handoff_fp8.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <cassert>

using namespace strata::kernels::cuda;

// Verify vector alignment at compile time
static_assert(kHandoffFP8StrideBytes % 16 == 0, "GATE-B1: FP8 handoff stride must be 16-byte aligned");

struct TestResult {
    double cos_sim;
    double mean_normal_l1;
    double normal_rel_range_pct;
    double outlier_rel_err_pct;
};

TestResult run_profile(const char* name, float sigma, float normal_range, int64_t T) {
    const int64_t elems = kHandoffElemsPerToken; // 12804 elements per token
    const int64_t total_elements = T * elems;

    std::vector<float> h_src(total_elements);
    std::mt19937 gen(1337);
    std::normal_distribution<float> dist(0.0f, sigma);
    for (size_t i = 0; i < h_src.size(); ++i) {
        h_src[i] = dist(gen);
    }

    // Inject sharp channel outliers in channel 42 (120.0f) and channel 5000 (85.0f)
    for (int64_t t = 0; t < T; ++t) {
        h_src[t * elems + 42] = 120.0f;
        h_src[t * elems + 5000] = 85.0f;
    }

    float *d_src = nullptr, *d_dst = nullptr, *d_scales = nullptr;
    uint8_t *d_payload = nullptr;
    size_t payload_bytes = (size_t)T * kHandoffFP8StrideBytes;
    size_t total_blocks = (size_t)T * kHandoffBlocksPerToken;
    size_t scales_bytes = total_blocks * sizeof(float);

    cudaMalloc(&d_src, total_elements * sizeof(float));
    cudaMalloc(&d_dst, total_elements * sizeof(float));
    cudaMalloc(&d_payload, payload_bytes);
    cudaMalloc(&d_scales, scales_bytes);

    cudaMemcpy(d_src, h_src.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_payload, 0, payload_bytes);
    cudaMemset(d_dst, 0, total_elements * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int ctas = (total_blocks + 7) / 8;
    pack_activations_mxfp8_kernel<<<ctas, 256, 0, stream>>>(
        d_src, d_payload, d_scales, T, elems, kHandoffFP8StrideBytes
    );

    unpack_activations_mxfp8_kernel<<<ctas, 256, 0, stream>>>(
        d_payload, d_scales, d_dst, T, elems, kHandoffFP8StrideBytes
    );

    cudaStreamSynchronize(stream);

    std::vector<float> h_dst(total_elements);
    cudaMemcpy(h_dst.data(), d_dst, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

    double dot_prod = 0.0, norm_src = 0.0, norm_dst = 0.0;
    double max_outlier_42_err = 0.0;
    double normal_l1_err = 0.0;
    int64_t normal_count = 0;

    for (int64_t t = 0; t < T; ++t) {
        for (int64_t i = 0; i < elems; ++i) {
            int64_t idx = t * elems + i;
            float orig = h_src[idx];
            float rec = h_dst[idx];

            dot_prod += (double)orig * (double)rec;
            norm_src += (double)orig * (double)orig;
            norm_dst += (double)rec * (double)rec;

            double abs_err = std::abs((double)rec - (double)orig);
            if (i == 42) {
                if (abs_err > max_outlier_42_err) max_outlier_42_err = abs_err;
            } else if (i != 5000) {
                normal_l1_err += abs_err;
                normal_count++;
            }
        }
    }

    double cos_sim = dot_prod / (std::sqrt(norm_src) * std::sqrt(norm_dst));
    double mean_normal_l1 = normal_l1_err / normal_count;
    double normal_rel_pct = (mean_normal_l1 / normal_range) * 100.0;
    double outlier_rel_pct = (max_outlier_42_err / 120.0) * 100.0;

    printf("  [%s Profile (sigma=%.2f)]\n", name, sigma);
    printf("    Cosine Similarity            : %.8f\n", cos_sim);
    printf("    Normal Channels Mean L1 Error: %.6f (rel to range %.2f: %.3f%%)\n",
           mean_normal_l1, normal_range, normal_rel_pct);
    printf("    Outlier Ch42 (120.0f) Max Err: %.6f (rel: %.4f%%)\n",
           max_outlier_42_err, outlier_rel_pct);

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_payload);
    cudaFree(d_scales);
    cudaStreamDestroy(stream);

    return {cos_sim, mean_normal_l1, normal_rel_pct, outlier_rel_pct};
}

int main() {
    printf("==================================================================\n");
    printf("  PATH B: BLOCK-MICROSCALED FP8 (MXFP8 B=64) HANDOFF PARITY SUITE\n");
    printf("==================================================================\n\n");

    // GATE-B1: 16-Byte Vector Alignment
    printf("--- GATE-B1: 16-Byte Vector Memory Alignment ---\n");
    printf("  kHandoffElemsPerToken  = %ld elements (payload: %ld B)\n", (long)kHandoffElemsPerToken, (long)kHandoffFP8PayloadBytes);
    printf("  kHandoffFP8PadBytes    = %ld B padding\n", (long)kHandoffFP8PadBytes);
    printf("  kHandoffFP8StrideBytes = %ld B\n", (long)kHandoffFP8StrideBytes);
    printf("  Alignment: %ld %% 16 = %ld\n", (long)kHandoffFP8StrideBytes, (long)(kHandoffFP8StrideBytes % 16));

    if (kHandoffFP8StrideBytes % 16 != 0) {
        fprintf(stderr, "FAIL: GATE-B1 Vector Alignment FAILED!\n");
        return 1;
    }
    printf("  [PASS] GATE-B1: Vector memory stride is 16-byte aligned.\n\n");

    // Chunk Footprint
    printf("--- Chunk Memory Footprint Verification (T=4096) ---\n");
    const int64_t T_4k = 4096;
    size_t fp32_bytes = (size_t)T_4k * kHandoffElemsPerToken * sizeof(float);
    size_t fp8_payload_bytes = (size_t)T_4k * kHandoffFP8StrideBytes;
    size_t total_blocks_4k = (size_t)T_4k * kHandoffBlocksPerToken;
    size_t fp8_scales_bytes = total_blocks_4k * sizeof(float);
    size_t total_fp8_bytes = fp8_payload_bytes + fp8_scales_bytes;

    printf("  FP32 Uncompressed Payload : %10zu B (%.2f MB)\n", fp32_bytes, fp32_bytes / 1e6);
    printf("  FP8 Compressed Payload    : %10zu B (%.2f MB)\n", fp8_payload_bytes, fp8_payload_bytes / 1e6);
    printf("  FP8 Scales Metadata       : %10zu B (%.2f MB)\n", fp8_scales_bytes, fp8_scales_bytes / 1e6);
    printf("  Total FP8 Transferred     : %10zu B (%.2f MB)\n", total_fp8_bytes, total_fp8_bytes / 1e6);
    printf("  Compression Ratio         : %.2fx (%.1f%% bandwidth reduction)\n\n",
           (double)fp32_bytes / total_fp8_bytes,
           (1.0 - (double)total_fp8_bytes / fp32_bytes) * 100.0);

    // GATE-B2 & GATE-B3: Parity on Transformer Residuals & Standard Normal
    printf("--- GATE-B2 & GATE-B3: Numerical Parity across Activation Profiles ---\n");
    TestResult r_tf = run_profile("Realistic Deep Residual", 0.20f, 1.20f, 64);
    TestResult r_sn = run_profile("Standard Normal Baseline", 1.00f, 1.20f, 64);

    bool pass_b2 = (r_tf.cos_sim > 0.9999);
    bool pass_b3 = (r_tf.normal_rel_range_pct <= 1.5 && r_tf.outlier_rel_err_pct < 0.001);

    printf("\n  Gate Evaluation:\n");
    if (pass_b2) {
        printf("  [PASS] GATE-B2: Transformer residual cosine similarity %.8f > 0.9999\n", r_tf.cos_sim);
    } else {
        printf("  [FAIL] GATE-B2: Cosine similarity %.8f <= 0.9999\n", r_tf.cos_sim);
    }

    if (pass_b3) {
        printf("  [PASS] GATE-B3: Normal channel error %.3f%% <= 1.5%%; outlier channel 42 preserved with %.4f%% error\n",
               r_tf.normal_rel_range_pct, r_tf.outlier_rel_err_pct);
    } else {
        printf("  [FAIL] GATE-B3: Outlier isolation test failed\n");
    }

    bool all_pass = pass_b2 && pass_b3;
    printf("\n==================================================================\n");
    if (all_pass) {
        printf("  VERDICT: ALL PATH B VERIFICATION GATES PASSED (100%%)\n");
    } else {
        printf("  VERDICT: VERIFICATION GATES FAILED\n");
    }
    printf("==================================================================\n");

    return all_pass ? 0 : 1;
}
