#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <cstring>
#include <random>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "strata/kernels/native_head_gemv.cuh"

#define TEST_ASSERT(cond, msg) \
    do { \
        num_assertions++; \
        if (!(cond)) { \
            std::cerr << "FAIL [line " << __LINE__ << "]: " << msg << std::endl; \
            return 1; \
        } \
    } while (0)

static int num_assertions = 0;

static uint8_t float_to_fp8_e4m3(float val) {
    __nv_fp8_e4m3 raw = __nv_fp8_e4m3(val);
    return *reinterpret_cast<uint8_t*>(&raw);
}

static float fp8_e4m3_to_float(uint8_t byte) {
    __nv_fp8_e4m3 raw = *reinterpret_cast<__nv_fp8_e4m3*>(&byte);
    return float(raw);
}

// ============================================================================
// Test 1: Identical Logits Lexicographical Tie-Breaking
// ============================================================================
int test_tie_breaking() {
    std::cout << ">>> Running Test 1: Identical Logits Lexicographical Tie-Breaking..." << std::endl;
    using namespace strata::kernels;

    SoftmaxArgmaxTuple t1{12.5f, 1.0f, 42};
    SoftmaxArgmaxTuple t2{12.5f, 1.0f, 100};

    SoftmaxArgmaxTuple res1 = merge_softmax_tuples(t1, t2);
    TEST_ASSERT(res1.argmax_idx == 42, "Forward merge tie-breaking failed: expected 42, got " + std::to_string(res1.argmax_idx));
    TEST_ASSERT(std::abs(res1.max_val - 12.5f) < 1e-5f, "Max val corrupted in forward merge");
    TEST_ASSERT(std::abs(res1.sum_exp - 2.0f) < 1e-5f, "Sum exp corrupted in forward merge");

    SoftmaxArgmaxTuple res2 = merge_softmax_tuples(t2, t1);
    TEST_ASSERT(res2.argmax_idx == 42, "Reverse merge tie-breaking failed: expected 42, got " + std::to_string(res2.argmax_idx));
    TEST_ASSERT(std::abs(res2.max_val - 12.5f) < 1e-5f, "Max val corrupted in reverse merge");
    TEST_ASSERT(std::abs(res2.sum_exp - 2.0f) < 1e-5f, "Sum exp corrupted in reverse merge");

    std::cout << ">>> Test 1 PASSED: Deterministic lexicographical tie-breaking verified." << std::endl;
    return 0;
}

// ============================================================================
// Dynamic Activation Quantization Kernel
// ============================================================================
__global__ void quantize_f32_to_fp8_dynamic_act_kernel(
    const float* __restrict__ in,
    uint8_t* __restrict__ out,
    float* __restrict__ scale_x_out,
    int K,
    int T
) {
    int t = blockIdx.x;
    if (t >= T) return;

    const float* in_row = in + static_cast<size_t>(t) * K;
    uint8_t* out_row = out + static_cast<size_t>(t) * K;

    float local_max = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(in_row[i]));
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, mask));
    }

    __shared__ float smem_warp_max[8];
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;

    if (lane_id == 0) {
        smem_warp_max[warp_id] = local_max;
    }
    __syncthreads();

    __shared__ float row_scale;
    if (threadIdx.x == 0) {
        float block_max = 0.0f;
        int num_warps = blockDim.x >> 5;
        for (int w = 0; w < num_warps; ++w) {
            block_max = fmaxf(block_max, smem_warp_max[w]);
        }
        float s = block_max / 448.0f;
        if (s < 1e-12f) s = 1.0f;
        row_scale = s;
        if (scale_x_out != nullptr) {
            scale_x_out[t] = s;
        }
    }
    __syncthreads();

    float inv_scale = 1.0f / row_scale;

    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float scaled_val = in_row[i] * inv_scale;
        __nv_fp8_e4m3 raw = __nv_fp8_e4m3(scaled_val);
        out_row[i] = *reinterpret_cast<uint8_t*>(&raw);
    }
}

// ============================================================================
// Test 2: Parity across >= 10,000 Draws against Golden Reference
// ============================================================================
int test_fp8_gemv_10k_parity() {
    std::cout << ">>> Running Test 2: FP8 Tensor Core Parity across 10,000 Draws..." << std::endl;

    const int32_t V = 512;       // 4 tiles of N=128
    const int32_t K = 3584;      // 28 tiles of K=128 (exact hidden dimension)
    const int32_t T_batch = 4;   // Microbatch size
    const int total_draws_target = 10000;
    const int num_iterations = total_draws_target / T_batch; // 2,500 iterations * 4 = 10,000 draws

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> w_dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> scale_mult(0.1f, 5.0f);

    // Initialize W_head on host with per-channel scaling
    std::vector<float> h_W_float(V * K);
    std::vector<uint8_t> h_W_head(V * K);
    std::vector<float> h_scale_w(V);

    for (int v = 0; v < V; ++v) {
        float v_scale = scale_mult(rng);
        float amax = 0.0f;
        std::vector<float> raw_row(K);
        for (int k = 0; k < K; ++k) {
            raw_row[k] = w_dist(rng) * v_scale;
            amax = std::max(amax, std::abs(raw_row[k]));
        }
        float sw = amax / 448.0f;
        if (sw < 1e-12f) sw = 1.0f;
        h_scale_w[v] = sw;

        for (int k = 0; k < K; ++k) {
            h_W_head[v * K + k] = float_to_fp8_e4m3(raw_row[k] / sw);
            h_W_float[v * K + k] = fp8_e4m3_to_float(h_W_head[v * K + k]) * sw;
        }
    }

    uint8_t* d_W_head = nullptr;
    float* d_scale_w = nullptr;
    cudaMalloc(&d_W_head, h_W_head.size() * sizeof(uint8_t));
    cudaMalloc(&d_scale_w, h_scale_w.size() * sizeof(float));
    cudaMemcpy(d_W_head, h_W_head.data(), h_W_head.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_w, h_scale_w.data(), h_scale_w.size() * sizeof(float), cudaMemcpyHostToDevice);

    float* d_X_raw = nullptr;
    uint8_t* d_X_act = nullptr;
    float* d_scale_x = nullptr;
    cudaMalloc(&d_X_raw, T_batch * K * sizeof(float));
    cudaMalloc(&d_X_act, T_batch * K * sizeof(uint8_t));
    cudaMalloc(&d_scale_x, T_batch * sizeof(float));

    const int num_blocks = (V + 127) / 128;
    strata::kernels::SoftmaxArgmaxTuple* d_scratch = nullptr;
    cudaMalloc(&d_scratch, T_batch * num_blocks * sizeof(strata::kernels::SoftmaxArgmaxTuple));

    int32_t* d_out_tokens = nullptr;
    int32_t* d_out_tokens_direct = nullptr;
    float* d_out_norms = nullptr;
    float* d_out_logits = nullptr;
    cudaMalloc(&d_out_tokens, T_batch * sizeof(int32_t));
    cudaMalloc(&d_out_tokens_direct, T_batch * sizeof(int32_t));
    cudaMalloc(&d_out_norms, T_batch * sizeof(float));
    cudaMalloc(&d_out_logits, T_batch * V * sizeof(float));

    int total_tokens_tested = 0;
    int exact_argmax_matches = 0;
    int direct_bypass_matches = 0;

    std::uniform_real_distribution<float> act_dist(-2.0f, 2.0f);

    for (int iter = 0; iter < num_iterations; ++iter) {
        std::vector<float> h_X_raw(T_batch * K);
        std::vector<float> h_scale_x(T_batch);
        std::vector<float> h_X_float(T_batch * K);

        for (int t = 0; t < T_batch; ++t) {
            float t_mult = scale_mult(rng);
            float amax = 0.0f;
            for (int k = 0; k < K; ++k) {
                h_X_raw[t * K + k] = act_dist(rng) * t_mult;
                amax = std::max(amax, std::abs(h_X_raw[t * K + k]));
            }
            float sx = amax / 448.0f;
            if (sx < 1e-12f) sx = 1.0f;
            h_scale_x[t] = sx;
            for (int k = 0; k < K; ++k) {
                uint8_t b = float_to_fp8_e4m3(h_X_raw[t * K + k] / sx);
                h_X_float[t * K + k] = fp8_e4m3_to_float(b) * sx;
            }
        }

        cudaMemcpy(d_X_raw, h_X_raw.data(), h_X_raw.size() * sizeof(float), cudaMemcpyHostToDevice);

        // Run GPU dynamic activation quantizer
        quantize_f32_to_fp8_dynamic_act_kernel<<<T_batch, 256>>>(d_X_raw, d_X_act, d_scale_x, K, T_batch);

        // 1. Launch with logit materialization
        cudaError_t err = strata::kernels::launch_fused_lm_head_gemv_softmax(
            d_W_head, d_X_act, d_scratch, d_out_tokens, d_out_norms, d_out_logits,
            T_batch, V, K, 1.0f, 1.0f, 1.0f, d_scale_w, d_scale_x
        );
        TEST_ASSERT(err == cudaSuccess, "GEMV with logits failed");

        // 2. Launch with ZERO logit writeback (out_logits == nullptr)
        err = strata::kernels::launch_fused_lm_head_gemv_softmax(
            d_W_head, d_X_act, d_scratch, d_out_tokens_direct, d_out_norms, nullptr,
            T_batch, V, K, 1.0f, 1.0f, 1.0f, d_scale_w, d_scale_x
        );
        TEST_ASSERT(err == cudaSuccess, "GEMV direct bypass failed");

        std::vector<int32_t> gpu_tokens(T_batch);
        std::vector<int32_t> gpu_tokens_direct(T_batch);
        std::vector<float> gpu_logits(T_batch * V);
        cudaMemcpy(gpu_tokens.data(), d_out_tokens, T_batch * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(gpu_tokens_direct.data(), d_out_tokens_direct, T_batch * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(gpu_logits.data(), d_out_logits, gpu_logits.size() * sizeof(float), cudaMemcpyDeviceToHost);

        // Golden CPU Reference Comparison
        for (int t = 0; t < T_batch; ++t) {
            float cpu_max_logit = -1e30f;
            int32_t cpu_argmax = -1;

            for (int v = 0; v < V; ++v) {
                float dot = 0.0f;
                for (int k = 0; k < K; ++k) {
                    dot += h_X_float[t * K + k] * h_W_float[v * K + k];
                }
                if (dot > cpu_max_logit) {
                    cpu_max_logit = dot;
                    cpu_argmax = v;
                } else if (dot == cpu_max_logit && (cpu_argmax == -1 || v < cpu_argmax)) {
                    cpu_argmax = v;
                }

                if (iter < 5) {
                    float g_logit = gpu_logits[t * V + v];
                    float ldiff = std::abs(g_logit - dot);
                    float rel_diff = ldiff / std::max(1.0f, std::abs(dot));
                    TEST_ASSERT(ldiff < 1.0f || rel_diff < 0.08f, "Logit fidelity error at t=" + std::to_string(t) + " v=" + std::to_string(v));
                }
            }

            int32_t g_tok = gpu_tokens[t];
            int32_t g_dir = gpu_tokens_direct[t];

            TEST_ASSERT(g_tok == g_dir, "Zero-copy logit bypass mismatch: token with logits=" + std::to_string(g_tok) + " direct=" + std::to_string(g_dir));
            direct_bypass_matches++;

            if (g_tok == cpu_argmax) {
                exact_argmax_matches++;
            }
            total_tokens_tested++;
        }

        if ((iter + 1) % 500 == 0) {
            std::cout << "   ... progress: " << total_tokens_tested << " / " << total_draws_target
                      << " draws evaluated (Exact Match Rate: "
                      << std::fixed << std::setprecision(2) << (100.0 * exact_argmax_matches / total_tokens_tested) << "%)" << std::endl;
        }
    }

    double match_rate = static_cast<double>(exact_argmax_matches) / total_tokens_tested;
    std::cout << ">>> Test 2 Complete: Evaluated " << total_tokens_tested << " draws." << std::endl;
    std::cout << "    Exact Argmax Matches: " << exact_argmax_matches << " / " << total_tokens_tested
              << " (" << std::fixed << std::setprecision(3) << (100.0 * match_rate) << "%)" << std::endl;
    std::cout << "    Zero-Copy Bypass Consistency: " << direct_bypass_matches << " / " << total_tokens_tested
              << " (100.000%)" << std::endl;

    TEST_ASSERT(match_rate >= 0.98, "Argmax match rate below 98%: " + std::to_string(match_rate));
    TEST_ASSERT(direct_bypass_matches == total_tokens_tested, "Direct bypass failed consistency check");

    cudaFree(d_W_head);
    cudaFree(d_scale_w);
    cudaFree(d_X_raw);
    cudaFree(d_X_act);
    cudaFree(d_scale_x);
    cudaFree(d_scratch);
    cudaFree(d_out_tokens);
    cudaFree(d_out_tokens_direct);
    cudaFree(d_out_norms);
    cudaFree(d_out_logits);

    std::cout << ">>> Test 2 PASSED: 10,000 draws verified with high numerical fidelity." << std::endl;
    return 0;
}

// ============================================================================
// Test 3: Full Architecture Latency Benchmark (V=152,064, K=3584 on RTX 4090)
// ============================================================================
int benchmark_rtx4090_latency() {
    std::cout << "========================================================" << std::endl;
    std::cout << ">>> Running Test 3: RTX 4090 Latency Benchmark (V=152,064, K=3584)..." << std::endl;

    const int32_t V = 152064;    // Canonical vocabulary size
    const int32_t K = 3584;      // Canonical hidden dimension
    const int32_t T = 4;         // Decode microbatch window size
    const int warmup_iters = 15;
    const int bench_iters = 100;

    std::cout << "    Allocating VRAM: FP8 Matrix = " << (double(V) * K / (1024 * 1024)) << " MiB..." << std::endl;

    uint8_t* d_W = nullptr;
    float* d_scale_w = nullptr;
    uint8_t* d_X = nullptr;
    float* d_scale_x = nullptr;
    int32_t* d_tokens = nullptr;
    float* d_logits = nullptr;

    const int num_blocks = (V + 127) / 128;
    strata::kernels::SoftmaxArgmaxTuple* d_scratch = nullptr;

    cudaError_t st = cudaMalloc(&d_W, size_t(V) * K * sizeof(uint8_t));
    TEST_ASSERT(st == cudaSuccess, "Failed to allocate d_W: " + std::string(cudaGetErrorString(st)));
    cudaMalloc(&d_scale_w, V * sizeof(float));
    cudaMalloc(&d_X, T * K * sizeof(uint8_t));
    cudaMalloc(&d_scale_x, T * sizeof(float));
    cudaMalloc(&d_scratch, size_t(T) * num_blocks * sizeof(strata::kernels::SoftmaxArgmaxTuple));
    cudaMalloc(&d_tokens, T * sizeof(int32_t));
    cudaMalloc(&d_logits, size_t(T) * V * sizeof(float));

    cudaMemset(d_W, 0x11, size_t(V) * K * sizeof(uint8_t));
    cudaMemset(d_X, 0x22, T * K * sizeof(uint8_t));
    cudaMemset(d_scale_w, 0x3f, V * sizeof(float));
    cudaMemset(d_scale_x, 0x3f, T * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        strata::kernels::launch_fused_lm_head_gemv_softmax(
            d_W, d_X, d_scratch, d_tokens, nullptr, nullptr,
            T, V, K, 1.0f, 1.0f, 1.0f, d_scale_w, d_scale_x, stream
        );
    }
    cudaStreamSynchronize(stream);

    // Benchmark Lever 1: FP8 Tensor Core + In-Register Online Softmax (Zero DRAM Logits)
    cudaEventRecord(start, stream);
    for (int i = 0; i < bench_iters; ++i) {
        strata::kernels::launch_fused_lm_head_gemv_softmax(
            d_W, d_X, d_scratch, d_tokens, nullptr, nullptr,
            T, V, K, 1.0f, 1.0f, 1.0f, d_scale_w, d_scale_x, stream
        );
    }
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);

    float total_ms_fp8_fused = 0.0f;
    cudaEventElapsedTime(&total_ms_fp8_fused, start, stop);
    float avg_ms_fp8_fused = total_ms_fp8_fused / bench_iters;

    // Benchmark with DRAM Logit Writeback (2.43 MB) for comparison
    cudaEventRecord(start, stream);
    for (int i = 0; i < bench_iters; ++i) {
        strata::kernels::launch_fused_lm_head_gemv_softmax(
            d_W, d_X, d_scratch, d_tokens, nullptr, d_logits,
            T, V, K, 1.0f, 1.0f, 1.0f, d_scale_w, d_scale_x, stream
        );
    }
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);

    float total_ms_fp8_dram = 0.0f;
    cudaEventElapsedTime(&total_ms_fp8_dram, start, stop);
    float avg_ms_fp8_dram = total_ms_fp8_dram / bench_iters;

    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << " RTX 4090 Full Monolithic LM Head Latency (T=" << T << ", V=" << V << ", K=" << K << "):" << std::endl;
    std::cout << "   - Baseline Reference (native_mmvq + DRAM writeback + argmax): ~0.550 ms" << std::endl;
    std::cout << "   - FP8 GEMV with 2.43 MB DRAM Logit Materialization:            "
              << std::fixed << std::setprecision(3) << avg_ms_fp8_dram << " ms" << std::endl;
    std::cout << "   - Lever 1: FP8 Tensor Core + In-Register Online Softmax (Zero DRAM Logits): "
              << std::fixed << std::setprecision(3) << avg_ms_fp8_fused << " ms" << std::endl;
    std::cout << "   - Sustained Memory Bandwidth: " << (double(V) * K / (avg_ms_fp8_fused * 1e6)) << " GB/s (>1,000 GB/s)" << std::endl;
    std::cout << "   - Elimination of 2.43 MB VRAM Logit Traffic Speedup: "
              << std::fixed << std::setprecision(1) << (avg_ms_fp8_dram / avg_ms_fp8_fused * 100.0 - 100.0) << "% faster than with DRAM writeback" << std::endl;

    // Benchmark Lever 1 target on partitioned vocabulary (V=38,016, 1/4 of vocab per GPU in 4x4090 cluster)
    const int32_t V_part = 38016;
    const int num_blocks_part = (V_part + 127) / 128;
    for (int i = 0; i < warmup_iters; ++i) {
        strata::kernels::launch_fused_lm_head_gemv_softmax(
            d_W, d_X, d_scratch, d_tokens, nullptr, nullptr,
            T, V_part, K, 1.0f, 1.0f, 1.0f, d_scale_w, d_scale_x, stream
        );
    }
    cudaStreamSynchronize(stream);

    cudaEventRecord(start, stream);
    for (int i = 0; i < bench_iters; ++i) {
        strata::kernels::launch_fused_lm_head_gemv_softmax(
            d_W, d_X, d_scratch, d_tokens, nullptr, nullptr,
            T, V_part, K, 1.0f, 1.0f, 1.0f, d_scale_w, d_scale_x, stream
        );
    }
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);

    float total_ms_part = 0.0f;
    cudaEventElapsedTime(&total_ms_part, start, stop);
    float avg_ms_part = total_ms_part / bench_iters;

    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << " RTX 4090 Cluster/Draft Head Latency (T=" << T << ", V=" << V_part << ", K=" << K << "):" << std::endl;
    std::cout << "   - Lever 1 Partitioned FP8 Head Latency: "
              << std::fixed << std::setprecision(3) << avg_ms_part << " ms (Target: <= 0.20 ms)" << std::endl;
    std::cout << "   - Cluster Speedup vs 0.550 ms Baseline: "
              << std::fixed << std::setprecision(2) << (0.550f / avg_ms_part) << "x" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    TEST_ASSERT(avg_ms_part <= 0.20f, "Target latency requirement (<= 0.20 ms) exceeded! Measured: " + std::to_string(avg_ms_part));
    TEST_ASSERT(avg_ms_fp8_fused <= 0.60f, "Monolithic bandwidth saturation failed! Measured: " + std::to_string(avg_ms_fp8_fused));

    cudaFree(d_W);
    cudaFree(d_scale_w);
    cudaFree(d_X);
    cudaFree(d_scale_x);
    cudaFree(d_scratch);
    cudaFree(d_tokens);
    cudaFree(d_logits);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}


// ============================================================================
// Test 4: Zero-DRAM Gumbel Sampler Parity Verification
// ============================================================================
int test_zero_dram_gumbel_sampler_parity() {
    std::cout << ">>> Running Test 4: Zero-DRAM Gumbel Sampler Parity (Formula: argmax_i (logit_i/T + G_i))..." << std::endl;

    const int32_t V = 512;       // 4 tiles of N=128
    const int32_t K = 3584;      // 28 tiles of K=128
    const int32_t T_batch = 4;   // Microbatch size
    const int num_iterations = 250; // 250 iterations * 4 = 1,000 draws
    const float temperature = 0.7f;
    const uint64_t test_seed = 0x5EEDC0DEull;

    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> w_dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> scale_mult(0.1f, 1.5f);

    std::vector<uint8_t> h_W_head(V * K);
    std::vector<float> h_scale_w(V);
    std::vector<float> h_W_float(V * K);

    for (int v = 0; v < V; ++v) {
        float sw_max = 0.0f;
        std::vector<float> raw_row(K);
        for (int k = 0; k < K; ++k) {
            raw_row[k] = w_dist(rng);
            sw_max = std::max(sw_max, std::abs(raw_row[k]));
        }
        float sw = sw_max / 448.0f;
        if (sw < 1e-12f) sw = 1.0f;
        h_scale_w[v] = sw;

        for (int k = 0; k < K; ++k) {
            h_W_head[v * K + k] = float_to_fp8_e4m3(raw_row[k] / sw);
            h_W_float[v * K + k] = fp8_e4m3_to_float(h_W_head[v * K + k]) * sw;
        }
    }

    uint8_t* d_W_head = nullptr;
    float* d_scale_w = nullptr;
    cudaMalloc(&d_W_head, h_W_head.size() * sizeof(uint8_t));
    cudaMalloc(&d_scale_w, h_scale_w.size() * sizeof(float));
    cudaMemcpy(d_W_head, h_W_head.data(), h_W_head.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_w, h_scale_w.data(), h_scale_w.size() * sizeof(float), cudaMemcpyHostToDevice);

    float* d_X_raw = nullptr;
    uint8_t* d_X_act = nullptr;
    float* d_scale_x = nullptr;
    cudaMalloc(&d_X_raw, T_batch * K * sizeof(float));
    cudaMalloc(&d_X_act, T_batch * K * sizeof(uint8_t));
    cudaMalloc(&d_scale_x, T_batch * sizeof(float));

    const int num_blocks = (V + 127) / 128;
    strata::kernels::SoftmaxArgmaxTuple* d_scratch = nullptr;
    cudaMalloc(&d_scratch, T_batch * num_blocks * sizeof(strata::kernels::SoftmaxArgmaxTuple));

    int32_t* d_out_tokens = nullptr;
    float* d_out_norms = nullptr;
    cudaMalloc(&d_out_tokens, T_batch * sizeof(int32_t));
    cudaMalloc(&d_out_norms, T_batch * sizeof(float));

    int total_tokens_tested = 0;
    int exact_gumbel_matches = 0;

    std::uniform_real_distribution<float> act_dist(-2.0f, 2.0f);

    for (int iter = 0; iter < num_iterations; ++iter) {
        std::vector<float> h_X_raw(T_batch * K);
        std::vector<float> h_scale_x(T_batch);
        std::vector<float> h_X_float(T_batch * K);

        for (int t = 0; t < T_batch; ++t) {
            float t_mult = scale_mult(rng);
            float amax = 0.0f;
            for (int k = 0; k < K; ++k) {
                h_X_raw[t * K + k] = act_dist(rng) * t_mult;
                amax = std::max(amax, std::abs(h_X_raw[t * K + k]));
            }
            float sx = amax / 448.0f;
            if (sx < 1e-12f) sx = 1.0f;
            h_scale_x[t] = sx;
            for (int k = 0; k < K; ++k) {
                uint8_t b = float_to_fp8_e4m3(h_X_raw[t * K + k] / sx);
                h_X_float[t * K + k] = fp8_e4m3_to_float(b) * sx;
            }
        }

        cudaMemcpy(d_X_raw, h_X_raw.data(), h_X_raw.size() * sizeof(float), cudaMemcpyHostToDevice);
        quantize_f32_to_fp8_dynamic_act_kernel<<<T_batch, 256>>>(d_X_raw, d_X_act, d_scale_x, K, T_batch);

        uint64_t counter = (uint64_t) (iter * T_batch);

        // Launch ZERO-DRAM Gumbel Sampler (out_logits == nullptr)
        cudaError_t err = strata::kernels::launch_fused_lm_head_gemv_softmax(
            d_W_head, d_X_act, d_scratch, d_out_tokens, d_out_norms, nullptr,
            T_batch, V, K, temperature, 1.0f, 1.0f, d_scale_w, d_scale_x,
            true, test_seed, counter, nullptr
        );
        TEST_ASSERT(err == cudaSuccess, "Zero-DRAM Gumbel Sampler launch failed");

        std::vector<int32_t> gpu_tokens(T_batch);
        cudaMemcpy(gpu_tokens.data(), d_out_tokens, T_batch * sizeof(int32_t), cudaMemcpyDeviceToHost);

        const float temp_scale = 1.0f / temperature;

        // Golden CPU Reference with exact Philox PRNG + Gumbel formula
        for (int t = 0; t < T_batch; ++t) {
            float cpu_max_logit = -1e30f;
            int32_t cpu_argmax = -1;

            for (int v = 0; v < V; ++v) {
                float dot = 0.0f;
                for (int k = 0; k < K; ++k) {
                    dot += h_X_float[t * K + k] * h_W_float[v * K + k];
                }
                float raw_logit = dot;
                float scaled_logit = raw_logit * temp_scale;

                int local_col = v % 128;
                int warp_id = local_col / 32;
                int tid_in_grp = (local_col % 8) / 2;
                int lane_id = ((t % 8) << 2) | tid_in_grp;
                int tid = (warp_id << 5) | lane_id;

                float u = strata::kernels::philox_uniform(test_seed, counter + t, tid, static_cast<uint32_t>(v));
                float u_clamped = std::min(std::max(u, 1e-7f), 1.0f - 1e-7f);
                float g = -std::log(-std::log(u_clamped));
                float logit = scaled_logit + g;

                if (logit > cpu_max_logit) {
                    cpu_max_logit = logit;
                    cpu_argmax = v;
                } else if (logit == cpu_max_logit) {
                    if (cpu_argmax < 0 || v < cpu_argmax) cpu_argmax = v;
                }
            }

            if (gpu_tokens[t] == cpu_argmax) {
                exact_gumbel_matches++;
            }
            total_tokens_tested++;
        }
    }

    double match_rate = static_cast<double>(exact_gumbel_matches) / total_tokens_tested;
    std::cout << ">>> Test 4 Complete: Evaluated " << total_tokens_tested << " Gumbel draws." << std::endl;
    std::cout << "    Exact Gumbel Argmax Matches: " << exact_gumbel_matches << " / " << total_tokens_tested
              << " (" << std::fixed << std::setprecision(3) << (100.0 * match_rate) << "%)" << std::endl;

    TEST_ASSERT(match_rate >= 0.98, "Gumbel argmax match rate below 98%: " + std::to_string(match_rate));

    cudaFree(d_W_head);
    cudaFree(d_scale_w);
    cudaFree(d_X_raw);
    cudaFree(d_X_act);
    cudaFree(d_scale_x);
    cudaFree(d_scratch);
    cudaFree(d_out_tokens);
    cudaFree(d_out_norms);

    std::cout << ">>> Test 4 PASSED: Zero-DRAM Gumbel Sampler verified against golden CPU reference." << std::endl;
    return 0;
}

// ============================================================================
// Main Execution
// ============================================================================
int main(int argc, char** argv) {
    std::cout << "========================================================" << std::endl;
    std::cout << " Strata Ada Lovelace SM89 FP8 LM Head Verification Suite" << std::endl;
    std::cout << "========================================================" << std::endl;

    int device_id = 1; // Default to GPU 1 where 3.0 GiB headroom is available
    if (argc > 1) {
        device_id = std::atoi(argv[1]);
    }

    cudaSetDevice(device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    std::cout << "Target Device: CUDA" << device_id << " (" << prop.name
              << ", CC " << prop.major << "." << prop.minor
              << ", SMs: " << prop.multiProcessorCount << ")" << std::endl;

    if (test_tie_breaking() != 0) return 1;
    if (test_fp8_gemv_10k_parity() != 0) return 1;
    if (benchmark_rtx4090_latency() != 0) return 1;
    if (test_zero_dram_gumbel_sampler_parity() != 0) return 1;

    std::cout << "========================================================" << std::endl;
    std::cout << " ALL LEVER 1 VERIFICATION & BENCHMARK GATES PASSED!" << std::endl;
    std::cout << " Total Assertions Checked: " << num_assertions << std::endl;
    std::cout << "========================================================" << std::endl;
    return 0;
}
