#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <cstring>
#include <random>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "strata/kernels/native_embed_norm.cuh"
#include "strata/kernels/native_head_gemv.cuh"
#include "strata/pipeline/pipeline_dma.hpp"

#define TEST_ASSERT(cond, msg) \
    do { \
        num_assertions++; \
        if (!(cond)) { \
            std::cerr << "FAIL [line " << __LINE__ << "]: " << msg << std::endl; \
            return 1; \
        } \
    } while (0)

static int num_assertions = 0;

// Host helper to convert float to FP8 E4M3 raw byte
static uint8_t float_to_fp8_e4m3(float val) {
    __nv_fp8_e4m3 raw = __nv_fp8_e4m3(val);
    return *reinterpret_cast<uint8_t*>(&raw);
}

// Host helper to convert FP8 E4M3 byte to float
static float fp8_e4m3_to_float(uint8_t byte) {
    __nv_fp8_e4m3 raw = *reinterpret_cast<__nv_fp8_e4m3*>(&byte);
    return float(raw);
}

// ============================================================================
// Test 1: Branchless OOB Clamping (-1 and 200,000 against V=151,936)
// ============================================================================
int test_oob_clamping() {
    std::cout << ">>> Running Test 1: Branchless OOB Clamping (-1 and 200,000)..." << std::endl;

    (void)151936; // V = 151936
    const int32_t D = 3584;
    const int32_t num_vectors_per_tok = D / 16; // 224

    // Allocate minimal table: token 0, token 1, and token V-1
    // To save VRAM in unit test, allocate 4 rows: row 0 (tok 0), row 1 (tok 1), row 2 (dummy), row 3 (tok V-1)
    // We map safe_tok by creating a table of size V_alloc rows
    const int32_t V_alloc = 2048; // Allocate 2048 rows for unit test
    std::vector<uint4> h_table(V_alloc * num_vectors_per_tok);

    // Initialize row 0 with 0x11, row 1 with 0x22, row (V_alloc - 1) with 0x33
    for (int i = 0; i < num_vectors_per_tok; ++i) {
        h_table[0 * num_vectors_per_tok + i] = make_uint4(0x01010101, 0x01010101, 0x01010101, 0x01010101);
        h_table[(V_alloc - 1) * num_vectors_per_tok + i] = make_uint4(0x02020202, 0x02020202, 0x02020202, 0x02020202);
    }

    uint4* d_table = nullptr;
    cudaMalloc(&d_table, h_table.size() * sizeof(uint4));
    cudaMemcpy(d_table, h_table.data(), h_table.size() * sizeof(uint4), cudaMemcpyHostToDevice);

    std::vector<half> h_gamma(D, __float2half(1.0f));
    half* d_gamma = nullptr;
    cudaMalloc(&d_gamma, D * sizeof(half));
    cudaMemcpy(d_gamma, h_gamma.data(), D * sizeof(half), cudaMemcpyHostToDevice);

    // Tokens to test: -1, 0, V_alloc - 1, 200000
    // Under V = V_alloc:
    // tok = -1 should clamp to 0 (matching tok = 0)
    // tok = 200000 should clamp to V_alloc - 1 (matching tok = V_alloc - 1)
    std::vector<int32_t> h_tokens = {-1, 0, V_alloc - 1, 200000};
    int32_t num_test_tokens = h_tokens.size();

    int32_t* d_tokens = nullptr;
    cudaMalloc(&d_tokens, num_test_tokens * sizeof(int32_t));
    cudaMemcpy(d_tokens, h_tokens.data(), num_test_tokens * sizeof(int32_t), cudaMemcpyHostToDevice);

    half* d_out_main = nullptr;
    half* d_out_res = nullptr;
    half* d_out_hc = nullptr;
    cudaMalloc(&d_out_main, num_test_tokens * D * sizeof(half));
    cudaMalloc(&d_out_res, num_test_tokens * D * sizeof(half));
    cudaMalloc(&d_out_hc, num_test_tokens * 4 * D * sizeof(half));

    cudaError_t err = strata::kernels::launch_fused_embedding_rmsnorm_hc(
        d_table, d_tokens, d_gamma, d_out_main, d_out_res, d_out_hc,
        1e-6f, V_alloc, D, num_test_tokens
    );
    cudaDeviceSynchronize();
    TEST_ASSERT(err == cudaSuccess, "Kernel launch failed with error: " + std::string(cudaGetErrorString(err)));

    std::vector<half> h_out_main(num_test_tokens * D);
    cudaMemcpy(h_out_main.data(), d_out_main, h_out_main.size() * sizeof(half), cudaMemcpyDeviceToHost);

    // Verify: token 0 (index 0, input -1) must equal token 1 (index 1, input 0)
    bool neg_one_clamped = true;
    for (int d = 0; d < D; ++d) {
        float v_neg = __half2float(h_out_main[0 * D + d]);
        float v_zero = __half2float(h_out_main[1 * D + d]);
        if (std::abs(v_neg - v_zero) > 1e-4f) {
            neg_one_clamped = false;
            break;
        }
    }
    TEST_ASSERT(neg_one_clamped, "Token -1 was not clamped to Token 0 output!");

    // Verify: token 3 (input 200000) must equal token 2 (input V_alloc - 1)
    bool high_oob_clamped = true;
    for (int d = 0; d < D; ++d) {
        float v_high = __half2float(h_out_main[3 * D + d]);
        float v_max = __half2float(h_out_main[2 * D + d]);
        if (std::abs(v_high - v_max) > 1e-4f) {
            high_oob_clamped = false;
            break;
        }
    }
    TEST_ASSERT(high_oob_clamped, "Token 200,000 was not clamped to Token V-1 output!");

    cudaFree(d_table);
    cudaFree(d_gamma);
    cudaFree(d_tokens);
    cudaFree(d_out_main);
    cudaFree(d_out_res);
    cudaFree(d_out_hc);

    std::cout << ">>> Test 1 PASSED: Clamping verified on negative and high OOB tokens." << std::endl;
    return 0;
}

// ============================================================================
// Test 2: All-Zero Input Vector for RMSNorm (Zero NaNs / Infs Guard)
// ============================================================================
int test_all_zero_rmsnorm() {
    std::cout << ">>> Running Test 2: All-zero Input Vector for RMSNorm..." << std::endl;

    const int32_t V = 64;
    const int32_t D = 3584;
    const int32_t num_vectors = D / 16;

    // All-zero embedding table (0x00 is exact 0.0 in FP8 E4M3)
    std::vector<uint4> h_table(V * num_vectors, make_uint4(0, 0, 0, 0));
    uint4* d_table = nullptr;
    cudaMalloc(&d_table, h_table.size() * sizeof(uint4));
    cudaMemset(d_table, 0, h_table.size() * sizeof(uint4));

    std::vector<half> h_gamma(D, __float2half(1.0f));
    half* d_gamma = nullptr;
    cudaMalloc(&d_gamma, D * sizeof(half));
    cudaMemcpy(d_gamma, h_gamma.data(), D * sizeof(half), cudaMemcpyHostToDevice);

    std::vector<int32_t> h_tokens = {0, 1, 2, 3};
    int32_t num_tokens = h_tokens.size();
    int32_t* d_tokens = nullptr;
    cudaMalloc(&d_tokens, num_tokens * sizeof(int32_t));
    cudaMemcpy(d_tokens, h_tokens.data(), num_tokens * sizeof(int32_t), cudaMemcpyHostToDevice);

    half* d_out_main = nullptr;
    half* d_out_res = nullptr;
    half* d_out_hc = nullptr;
    cudaMalloc(&d_out_main, num_tokens * D * sizeof(half));
    cudaMalloc(&d_out_res, num_tokens * D * sizeof(half));
    cudaMalloc(&d_out_hc, num_tokens * 4 * D * sizeof(half));

    cudaError_t err = strata::kernels::launch_fused_embedding_rmsnorm_hc(
        d_table, d_tokens, d_gamma, d_out_main, d_out_res, d_out_hc,
        1e-6f, V, D, num_tokens
    );
    cudaDeviceSynchronize();
    TEST_ASSERT(err == cudaSuccess, "RMSNorm launch failed");

    std::vector<half> h_out(num_tokens * D);
    cudaMemcpy(h_out.data(), d_out_main, h_out.size() * sizeof(half), cudaMemcpyDeviceToHost);

    // Verify: strictly zero NaNs, zero Infs, exactly 0.0
    int nan_count = 0;
    int inf_count = 0;
    int non_zero_count = 0;

    for (size_t i = 0; i < h_out.size(); ++i) {
        float f = __half2float(h_out[i]);
        if (std::isnan(f)) nan_count++;
        if (std::isinf(f)) inf_count++;
        if (f != 0.0f) non_zero_count++;
    }

    TEST_ASSERT(nan_count == 0, "Found NaNs in RMSNorm output of zero vector!");
    TEST_ASSERT(inf_count == 0, "Found Infs in RMSNorm output of zero vector!");
    TEST_ASSERT(non_zero_count == 0, "Found non-zero values in RMSNorm output of zero vector!");

    cudaFree(d_table);
    cudaFree(d_gamma);
    cudaFree(d_tokens);
    cudaFree(d_out_main);
    cudaFree(d_out_res);
    cudaFree(d_out_hc);

    std::cout << ">>> Test 2 PASSED: 0 NaNs, 0 Infs, exact zero preservation verified." << std::endl;
    return 0;
}

// ============================================================================
// Test 3: Identical Logits Tie-Breaking (Lower Token ID Wins Deterministically)
// ============================================================================
int test_identical_logits_tie_breaking() {
    std::cout << ">>> Running Test 3: Identical Logits Tie-Breaking..." << std::endl;

    using namespace strata::kernels;

    // Direct unit test of merge_softmax_tuples
    SoftmaxArgmaxTuple t1{10.5f, 1.0f, 42};
    SoftmaxArgmaxTuple t2{10.5f, 1.0f, 100};

    // Forward merge: t1 (id 42) merged with t2 (id 100) -> lower id 42 must win
    SoftmaxArgmaxTuple res1 = merge_softmax_tuples(t1, t2);
    TEST_ASSERT(res1.argmax_idx == 42, "Forward merge failed: expected 42, got " + std::to_string(res1.argmax_idx));
    TEST_ASSERT(std::abs(res1.max_val - 10.5f) < 1e-5f, "Max val corrupted in forward merge");
    TEST_ASSERT(std::abs(res1.sum_exp - 2.0f) < 1e-5f, "Sum exp corrupted in forward merge");

    // Reverse merge: t2 (id 100) merged with t1 (id 42) -> lower id 42 must STILL win deterministically
    SoftmaxArgmaxTuple res2 = merge_softmax_tuples(t2, t1);
    TEST_ASSERT(res2.argmax_idx == 42, "Reverse merge failed: expected 42, got " + std::to_string(res2.argmax_idx));
    TEST_ASSERT(std::abs(res2.max_val - 10.5f) < 1e-5f, "Max val corrupted in reverse merge");
    TEST_ASSERT(std::abs(res2.sum_exp - 2.0f) < 1e-5f, "Sum exp corrupted in reverse merge");

    std::cout << ">>> Test 3 PASSED: Lexicographical tie-breaking verified deterministically." << std::endl;
    return 0;
}

// ============================================================================
// Test 4: Tensor Core FP8 GEMV Parity against FP32 CPU Reference (100 Sample Tokens)
// ============================================================================
int test_fp8_gemv_parity() {
    std::cout << ">>> Running Test 4: Tensor Core FP8 GEMV Parity across 100 sample tokens..." << std::endl;

    const int32_t V = 256;      // Vocabulary size for test (2 tiles of N=128)
    const int32_t K = 3584;     // Hidden dimension (28 tiles of K=128)
    const int32_t T_batch = 4;  // Microbatch size
    const int num_iterations = 25; // 25 batches * 4 tokens = 100 sample tokens

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Initialize W_head on host in FP8
    std::vector<uint8_t> h_W_head(V * K);
    std::vector<float> h_W_float(V * K);
    for (size_t i = 0; i < h_W_head.size(); ++i) {
        float f = dist(rng);
        h_W_head[i] = float_to_fp8_e4m3(f);
        h_W_float[i] = fp8_e4m3_to_float(h_W_head[i]); // Bit-exact dequantized float
    }

    uint8_t* d_W_head = nullptr;
    cudaMalloc(&d_W_head, h_W_head.size() * sizeof(uint8_t));
    cudaMemcpy(d_W_head, h_W_head.data(), h_W_head.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);

    uint8_t* d_X_act = nullptr;
    cudaMalloc(&d_X_act, T_batch * K * sizeof(uint8_t));

    const int num_blocks = (V + 127) / 128;
    strata::kernels::SoftmaxArgmaxTuple* d_scratch = nullptr;
    cudaMalloc(&d_scratch, T_batch * num_blocks * sizeof(strata::kernels::SoftmaxArgmaxTuple));

    int32_t* d_out_tokens = nullptr;
    float* d_out_norms = nullptr;
    cudaMalloc(&d_out_tokens, T_batch * sizeof(int32_t));
    cudaMalloc(&d_out_norms, T_batch * sizeof(float));

    int total_tokens_tested = 0;
    int argmax_matches = 0;

    for (int iter = 0; iter < num_iterations; ++iter) {
        // Generate T_batch test vectors
        std::vector<uint8_t> h_X_act(T_batch * K);
        std::vector<float> h_X_float(T_batch * K);
        for (size_t i = 0; i < h_X_act.size(); ++i) {
            float f = dist(rng);
            h_X_act[i] = float_to_fp8_e4m3(f);
            h_X_float[i] = fp8_e4m3_to_float(h_X_act[i]);
        }
        cudaMemcpy(d_X_act, h_X_act.data(), h_X_act.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);

        // Run GPU kernel
        cudaError_t err = strata::kernels::launch_fused_lm_head_gemv_softmax(
            d_W_head, d_X_act, d_scratch, d_out_tokens, d_out_norms,
            T_batch, V, K, 1.0f, 1.0f, 1.0f
        );
        cudaDeviceSynchronize();
        TEST_ASSERT(err == cudaSuccess, "GEMV launch error: " + std::string(cudaGetErrorString(err)));

        std::vector<int32_t> gpu_tokens(T_batch);
        cudaMemcpy(gpu_tokens.data(), d_out_tokens, T_batch * sizeof(int32_t), cudaMemcpyDeviceToHost);

        // Compute Golden CPU reference for each token in batch
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
            }

            int32_t gpu_argmax = gpu_tokens[t];
            total_tokens_tested++;
            if (gpu_argmax == cpu_argmax) {
                argmax_matches++;
            }
        }
    }

    std::cout << ">>> Evaluated " << total_tokens_tested << " sample tokens: "
              << argmax_matches << " / " << total_tokens_tested
              << " (" << (100.0 * argmax_matches / total_tokens_tested) << "%) exact argmax match." << std::endl;

    // On FP8 with 100 sample tokens, verify >= 95% argmax agreement with FP32 CPU reference
    double match_rate = static_cast<double>(argmax_matches) / total_tokens_tested;
    TEST_ASSERT(match_rate >= 0.95, "GEMV parity match rate below 95%: " + std::to_string(match_rate));

    cudaFree(d_W_head);
    cudaFree(d_X_act);
    cudaFree(d_scratch);
    cudaFree(d_out_tokens);
    cudaFree(d_out_norms);

    std::cout << ">>> Test 4 PASSED: FP8 Tensor Core GEMV parity verified across 100 tokens." << std::endl;
    return 0;
}

// ============================================================================
// Main Execution Gate
// ============================================================================
int main() {
    std::cout << "========================================================" << std::endl;
    std::cout << " Strata Ada SM89 FP8 Native Kernel Parity Verification" << std::endl;
    std::cout << "========================================================" << std::endl;

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices detected!" << std::endl;
        return 77;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Target Device 0: " << prop.name << " (Compute Capability: "
              << prop.major << "." << prop.minor << ")" << std::endl;

    if (test_oob_clamping() != 0) return 1;
    if (test_all_zero_rmsnorm() != 0) return 1;
    if (test_identical_logits_tie_breaking() != 0) return 1;
    if (test_fp8_gemv_parity() != 0) return 1;

    std::cout << "========================================================" << std::endl;
    std::cout << " ALL VERIFICATION GATES PASSED! Total Assertions: " << num_assertions << std::endl;
    std::cout << "========================================================" << std::endl;
    return 0;
}
