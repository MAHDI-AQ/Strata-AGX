#include "strata/core/native_fp8.hpp"
#include "strata/kernels/native_head_gemv.cuh"
#include "strata/kernels/native_embed_norm.cuh"
#include "strata/artifact/dequant.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <vector>
#include <thread>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <cmath>

namespace strata::core {

static bool g_native_head_fp8_enabled = false;
static bool g_native_embed_fp8_enabled = false;
static bool g_pipeline_dma_enabled = false;

void set_native_head_fp8_enabled(bool v) { g_native_head_fp8_enabled = v; }
bool native_head_fp8_enabled() { return g_native_head_fp8_enabled; }

void set_native_embed_fp8_enabled(bool v) { g_native_embed_fp8_enabled = v; }
bool native_embed_fp8_enabled() { return g_native_embed_fp8_enabled; }

void set_pipeline_dma_enabled(bool v) { g_pipeline_dma_enabled = v; }
bool pipeline_dma_enabled() { return g_pipeline_dma_enabled; }

namespace {

static bool dequantize_row_to_float(int type, const uint8_t* src, float* dst, int K) {
    const int n_blocks = K / 256;
    if (type == 14) { // Q6_K
        const size_t blk_bytes = 210;
        for (int b = 0; b < n_blocks; ++b) {
            strata::dequantize_q6_K(src + b * blk_bytes, dst + b * 256);
        }
        return true;
    } else if (type == 13) { // Q5_K
        const size_t blk_bytes = 176;
        for (int b = 0; b < n_blocks; ++b) {
            strata::dequantize_q5_K(src + b * blk_bytes, dst + b * 256);
        }
        return true;
    } else if (type == 23) { // IQ4_XS
        const size_t blk_bytes = 136;
        for (int b = 0; b < n_blocks; ++b) {
            strata::dequantize_iq4_xs(src + b * blk_bytes, dst + b * 256);
        }
        return true;
    } else if (type == 8) { // Q8_0
        for (int b = 0; b < K / 32; ++b) {
            strata::dequantize_q8_0(src + b * 34, dst + b * 32);
        }
        return true;
    } else if (type == 0) { // F32
        const float* fsrc = reinterpret_cast<const float*>(src);
        std::memcpy(dst, fsrc, K * sizeof(float));
        return true;
    } else if (type == 1) { // F16
        for (int b = 0; b < n_blocks; ++b) {
            strata::dequantize_f16(src + b * 512, dst + b * 256, 256);
        }
        return true;
    } else if (type == 30) { // BF16
        for (int b = 0; b < n_blocks; ++b) {
            strata::dequantize_bf16(src + b * 512, dst + b * 256, 256);
        }
        return true;
    }
    return false;
}

static bool dequantize_row_to_scaled_fp8(int type, const uint8_t* src, uint8_t* dst, float* scale_out, int K) {
    std::vector<float> row_f(K);
    if (!dequantize_row_to_float(type, src, row_f.data(), K)) return false;

    float amax = 0.0f;
    for (int i = 0; i < K; ++i) {
        amax = std::max(amax, std::abs(row_f[i]));
    }

    float sw = amax / 448.0f;
    if (sw < 1e-12f) sw = 1.0f;
    *scale_out = sw;

    float inv_sw = 1.0f / sw;
    for (int i = 0; i < K; ++i) {
        __nv_fp8_e4m3 raw = __nv_fp8_e4m3(row_f[i] * inv_sw);
        dst[i] = *reinterpret_cast<uint8_t*>(&raw);
    }
    return true;
}

size_t get_row_bytes(int type, int K) {
    if (type == 14) return static_cast<size_t>(K / 256) * 210;
    if (type == 13) return static_cast<size_t>(K / 256) * 176;
    if (type == 23) return static_cast<size_t>(K / 256) * 136;
    if (type == 8) return static_cast<size_t>(K / 32) * 34;
    if (type == 0) return static_cast<size_t>(K) * 4;
    if (type == 1 || type == 30) return static_cast<size_t>(K) * 2;
    return 0;
}

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

} // namespace

bool init_native_head_fp8(NativeHeadFp8State& state, const void* src_weights, int type, int K, int V, int device, std::string& err) {
    if (K <= 0 || V <= 0 || K % 256 != 0 || src_weights == nullptr) {
        err = "init_native_head_fp8: invalid dimensions or null weights";
        return false;
    }
    const size_t row_bytes = get_row_bytes(type, K);
    if (row_bytes == 0) {
        err = "init_native_head_fp8: unsupported quantization type " + std::to_string(type);
        return false;
    }

    const size_t total_fp8_bytes = static_cast<size_t>(V) * K;
    std::vector<uint8_t> h_fp8(total_fp8_bytes);
    std::vector<float> h_scale_w(V);
    const uint8_t* src_bytes = static_cast<const uint8_t*>(src_weights);

    unsigned int n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 8;
    std::vector<std::thread> workers;
    for (unsigned int tid = 0; tid < n_threads; ++tid) {
        workers.emplace_back([&, tid]() {
            int v_start = (static_cast<size_t>(V) * tid) / n_threads;
            int v_end = (static_cast<size_t>(V) * (tid + 1)) / n_threads;
            for (int v = v_start; v < v_end; ++v) {
                float sw = 1.0f;
                dequantize_row_to_scaled_fp8(
                    type,
                    src_bytes + static_cast<size_t>(v) * row_bytes,
                    h_fp8.data() + static_cast<size_t>(v) * K,
                    &sw,
                    K
                );
                h_scale_w[v] = sw;
            }
        });
    }
    for (auto& w : workers) w.join();

    int prev_dev = 0;
    cudaGetDevice(&prev_dev);
    cudaSetDevice(device);

    cudaError_t st = cudaMalloc(&state.d_W_head, total_fp8_bytes);
    if (st == cudaSuccess) st = cudaMemcpy(state.d_W_head, h_fp8.data(), total_fp8_bytes, cudaMemcpyHostToDevice);
    if (st == cudaSuccess) st = cudaMalloc(&state.d_scale_w, V * sizeof(float));
    if (st == cudaSuccess) st = cudaMemcpy(state.d_scale_w, h_scale_w.data(), V * sizeof(float), cudaMemcpyHostToDevice);
    if (st == cudaSuccess) st = cudaMalloc(&state.d_X_act, 16 * K * sizeof(uint8_t));
    if (st == cudaSuccess) st = cudaMalloc(&state.d_scale_x, 16 * sizeof(float));

    const int num_blocks = (V + 127) / 128;
    if (st == cudaSuccess) st = cudaMalloc(&state.d_scratchpad, 16 * num_blocks * sizeof(strata::kernels::SoftmaxArgmaxTuple));
    if (st == cudaSuccess) st = cudaMalloc(&state.d_out_tokens, 16 * sizeof(int32_t));
    if (st == cudaSuccess) st = cudaMalloc(&state.d_out_norms, 16 * sizeof(float));

    if (st != cudaSuccess) {
        err = "init_native_head_fp8 VRAM allocation error: " + std::string(cudaGetErrorString(st));
        free_native_head_fp8(state);
        cudaSetDevice(prev_dev);
        return false;
    }

    state.V = V;
    state.K = K;
    state.device = device;
    state.enabled = true;

    cudaSetDevice(prev_dev);
    std::fprintf(stderr, "strata head: native Ada SM89 FP8 E4M3 tensor core head active on CUDA%d (%d x %d, %.2f MiB VRAM, per-channel scaled)\n",
                 device, V, K, (double)total_fp8_bytes / 1048576.0);
    return true;
}

void free_native_head_fp8(NativeHeadFp8State& state) {
    if (state.d_W_head) {
        int prev = 0;
        cudaGetDevice(&prev);
        if (state.device >= 0) cudaSetDevice(state.device);
        cudaFree(state.d_W_head);
        if (state.d_scale_w) cudaFree(state.d_scale_w);
        if (state.d_X_act) cudaFree(state.d_X_act);
        if (state.d_scale_x) cudaFree(state.d_scale_x);
        if (state.d_scratchpad) cudaFree(state.d_scratchpad);
        if (state.d_out_tokens) cudaFree(state.d_out_tokens);
        if (state.d_out_norms) cudaFree(state.d_out_norms);
        cudaSetDevice(prev);
    }
    state.d_W_head = nullptr;
    state.d_scale_w = nullptr;
    state.d_X_act = nullptr;
    state.d_scale_x = nullptr;
    state.d_scratchpad = nullptr;
    state.d_out_tokens = nullptr;
    state.d_out_norms = nullptr;
    state.enabled = false;
}

bool run_native_head_fp8(const NativeHeadFp8State& state, const float* mixed, float* logits, int32_t* out_tokens, int T, void* stream, std::string& err,
                         float temperature, bool spec_gumbel, uint64_t seed, uint64_t counter) {
    if (!state.enabled || !state.d_W_head || !mixed) {
        err = "run_native_head_fp8: not initialized or null activations";
        return false;
    }
    if (T <= 0 || T > state.max_T) {
        err = "run_native_head_fp8: batch size T=" + std::to_string(T) + " exceeds max_T=" + std::to_string(state.max_T);
        return false;
    }

    cudaStream_t cs = static_cast<cudaStream_t>(stream);

    // Dynamic per-row activation scaling
    quantize_f32_to_fp8_dynamic_act_kernel<<<T, 256, 0, cs>>>(
        mixed, static_cast<uint8_t*>(state.d_X_act), state.d_scale_x, state.K, T
    );

    bool gumbel_enabled = spec_gumbel;
    if (!gumbel_enabled && logits == nullptr) {
        const char* eg = std::getenv("STRATA_SPEC_GUMBEL");
        const char* ez = std::getenv("STRATA_ZERO_DRAM_SAMPLER");
        if ((eg != nullptr && eg[0] == '1') || (ez != nullptr && ez[0] == '1')) {
            gumbel_enabled = true;
        }
    }

    int32_t* target_out_tokens = (out_tokens != nullptr) ? out_tokens : state.d_out_tokens;
    cudaError_t kerr = strata::kernels::launch_fused_lm_head_gemv_softmax(
        static_cast<const uint8_t*>(state.d_W_head),
        static_cast<const uint8_t*>(state.d_X_act),
        static_cast<strata::kernels::SoftmaxArgmaxTuple*>(state.d_scratchpad),
        target_out_tokens,
        state.d_out_norms,
        logits,
        T, state.V, state.K,
        temperature, 1.0f, 1.0f,
        state.d_scale_w,
        state.d_scale_x,
        gumbel_enabled,
        seed,
        counter,
        cs
    );

    if (kerr != cudaSuccess) {
        err = "run_native_head_fp8 kernel error: " + std::string(cudaGetErrorString(kerr));
        return false;
    }

    return true;
}

bool init_native_embed_fp8(NativeEmbedFp8State& state, const void* src_weights, int type, int K, int V, const void* gamma_weights, int device, std::string& err) {
    if (K <= 0 || V <= 0 || K % 256 != 0 || src_weights == nullptr) {
        err = "init_native_embed_fp8: invalid dimensions or null weights";
        return false;
    }
    const size_t row_bytes = get_row_bytes(type, K);
    if (row_bytes == 0) {
        err = "init_native_embed_fp8: unsupported quantization type " + std::to_string(type);
        return false;
    }

    const size_t total_fp8_bytes = static_cast<size_t>(V) * K;
    std::vector<uint8_t> h_fp8(total_fp8_bytes);
    const uint8_t* src_bytes = static_cast<const uint8_t*>(src_weights);

    unsigned int n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 8;
    std::vector<std::thread> workers;
    for (unsigned int tid = 0; tid < n_threads; ++tid) {
        workers.emplace_back([&, tid]() {
            int v_start = (static_cast<size_t>(V) * tid) / n_threads;
            int v_end = (static_cast<size_t>(V) * (tid + 1)) / n_threads;
            std::vector<float> row_f(K);
            for (int v = v_start; v < v_end; ++v) {
                dequantize_row_to_float(type, src_bytes + static_cast<size_t>(v) * row_bytes, row_f.data(), K);
                for (int i = 0; i < K; ++i) {
                    __nv_fp8_e4m3 raw = __nv_fp8_e4m3(row_f[i]);
                    h_fp8[static_cast<size_t>(v) * K + i] = *reinterpret_cast<uint8_t*>(&raw);
                }
            }
        });
    }
    for (auto& w : workers) w.join();

    int prev_dev = 0;
    cudaGetDevice(&prev_dev);
    cudaSetDevice(device);

    cudaError_t st = cudaMalloc(&state.d_embed_table, total_fp8_bytes);
    if (st == cudaSuccess) st = cudaMemcpy(state.d_embed_table, h_fp8.data(), total_fp8_bytes, cudaMemcpyHostToDevice);
    state.d_gamma = nullptr;
    if (st == cudaSuccess && gamma_weights != nullptr) {
        st = cudaMalloc(&state.d_gamma, K * sizeof(half));
        if (st == cudaSuccess) {
            st = cudaMemcpy(state.d_gamma, gamma_weights, K * sizeof(half), cudaMemcpyHostToDevice);
        }
    }

    if (st != cudaSuccess) {
        err = "init_native_embed_fp8 VRAM allocation error: " + std::string(cudaGetErrorString(st));
        free_native_embed_fp8(state);
        cudaSetDevice(prev_dev);
        return false;
    }

    state.V = V;
    state.K = K;
    state.device = device;
    state.enabled = true;

    cudaSetDevice(prev_dev);
    std::fprintf(stderr, "strata embed: native FP8 E4M3 embedding active on CUDA%d (%d x %d, %.2f MiB VRAM)\n",
                 device, V, K, (double)total_fp8_bytes / 1048576.0);
    return true;
}

void free_native_embed_fp8(NativeEmbedFp8State& state) {
    if (state.d_embed_table) {
        int prev = 0;
        cudaGetDevice(&prev);
        if (state.device >= 0) cudaSetDevice(state.device);
        cudaFree(state.d_embed_table);
        if (state.d_gamma) cudaFree(state.d_gamma);
        cudaSetDevice(prev);
    }
    state.d_embed_table = nullptr;
    state.d_gamma = nullptr;
    state.enabled = false;
}

bool run_native_embed_fp8(const NativeEmbedFp8State& state, const int32_t* tokens, int32_t num_tokens, void* out_main, void* out_res, void* out_hc, void* stream, std::string& err) {
    if (!state.enabled || !state.d_embed_table || !tokens) {
        err = "run_native_embed_fp8: not initialized or null tokens";
        return false;
    }

    cudaStream_t cs = static_cast<cudaStream_t>(stream);
    cudaError_t kerr = strata::kernels::launch_fused_embedding_rmsnorm_hc(
        static_cast<const uint4*>(state.d_embed_table),
        tokens,
        static_cast<const half*>(state.d_gamma),
        static_cast<float*>(out_main),
        static_cast<float*>(out_res),
        static_cast<float*>(out_hc),
        1e-6f,
        state.V,
        state.K,
        num_tokens,
        cs
    );

    if (kerr != cudaSuccess) {
        err = "run_native_embed_fp8 kernel error: " + std::string(cudaGetErrorString(kerr));
        return false;
    }
    return true;
}

} // namespace strata::core
