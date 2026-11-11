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

static bool dequantize_row_to_fp8(int type, const uint8_t* src, uint8_t* dst, int K) {
    const int n_blocks = K / 256;
    float blk[256];

    if (type == 14) { // Q6_K
        const size_t blk_bytes = 210;
        for (int b = 0; b < n_blocks; ++b) {
            strata::dequantize_q6_K(src + b * blk_bytes, blk);
            for (int i = 0; i < 256; ++i) {
                __nv_fp8_e4m3 raw = __nv_fp8_e4m3(blk[i]);
                dst[b * 256 + i] = *reinterpret_cast<uint8_t*>(&raw);
            }
        }
        return true;
    } else if (type == 13) { // Q5_K
        const size_t blk_bytes = 176;
        for (int b = 0; b < n_blocks; ++b) {
            strata::dequantize_q5_K(src + b * blk_bytes, blk);
            for (int i = 0; i < 256; ++i) {
                __nv_fp8_e4m3 raw = __nv_fp8_e4m3(blk[i]);
                dst[b * 256 + i] = *reinterpret_cast<uint8_t*>(&raw);
            }
        }
        return true;
    } else if (type == 23) { // IQ4_XS
        const size_t blk_bytes = 136;
        for (int b = 0; b < n_blocks; ++b) {
            strata::dequantize_iq4_xs(src + b * blk_bytes, blk);
            for (int i = 0; i < 256; ++i) {
                __nv_fp8_e4m3 raw = __nv_fp8_e4m3(blk[i]);
                dst[b * 256 + i] = *reinterpret_cast<uint8_t*>(&raw);
            }
        }
        return true;
    } else if (type == 8) { // Q8_0
        for (int b = 0; b < K / 32; ++b) {
            float sub[32];
            strata::dequantize_q8_0(src + b * 34, sub);
            for (int i = 0; i < 32; ++i) {
                __nv_fp8_e4m3 raw = __nv_fp8_e4m3(sub[i]);
                dst[b * 32 + i] = *reinterpret_cast<uint8_t*>(&raw);
            }
        }
        return true;
    } else if (type == 0) { // F32
        const float* fsrc = reinterpret_cast<const float*>(src);
        for (int i = 0; i < K; ++i) {
            __nv_fp8_e4m3 raw = __nv_fp8_e4m3(fsrc[i]);
            dst[i] = *reinterpret_cast<uint8_t*>(&raw);
        }
        return true;
    } else if (type == 1) { // F16
        for (int b = 0; b < n_blocks; ++b) {
            strata::dequantize_f16(src + b * 512, blk, 256);
            for (int i = 0; i < 256; ++i) {
                __nv_fp8_e4m3 raw = __nv_fp8_e4m3(blk[i]);
                dst[b * 256 + i] = *reinterpret_cast<uint8_t*>(&raw);
            }
        }
        return true;
    } else if (type == 30) { // BF16
        for (int b = 0; b < n_blocks; ++b) {
            strata::dequantize_bf16(src + b * 512, blk, 256);
            for (int i = 0; i < 256; ++i) {
                __nv_fp8_e4m3 raw = __nv_fp8_e4m3(blk[i]);
                dst[b * 256 + i] = *reinterpret_cast<uint8_t*>(&raw);
            }
        }
        return true;
    }
    return false;
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

__global__ void quantize_f32_to_fp8_act_kernel(const float* __restrict__ in, uint8_t* __restrict__ out, int count) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < count) {
        __nv_fp8_e4m3 raw = __nv_fp8_e4m3(in[idx]);
        out[idx] = *reinterpret_cast<uint8_t*>(&raw);
    }
}

__global__ void populate_argmax_logits_kernel(float* logits, const int32_t* out_tokens, int T, int V) {
    int t = blockIdx.x;
    if (t < T) {
        int tok = out_tokens[t];
        if (tok >= 0 && tok < V) {
            logits[static_cast<size_t>(t) * V + tok] = 100.0f;
        }
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
    const uint8_t* src_bytes = static_cast<const uint8_t*>(src_weights);

    unsigned int n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 8;
    std::vector<std::thread> workers;
    for (unsigned int tid = 0; tid < n_threads; ++tid) {
        workers.emplace_back([&, tid]() {
            int v_start = (static_cast<size_t>(V) * tid) / n_threads;
            int v_end = (static_cast<size_t>(V) * (tid + 1)) / n_threads;
            for (int v = v_start; v < v_end; ++v) {
                dequantize_row_to_fp8(type, src_bytes + static_cast<size_t>(v) * row_bytes,
                                      h_fp8.data() + static_cast<size_t>(v) * K, K);
            }
        });
    }
    for (auto& w : workers) w.join();

    int prev_dev = 0;
    cudaGetDevice(&prev_dev);
    cudaSetDevice(device);

    cudaError_t st = cudaMalloc(&state.d_W_head, total_fp8_bytes);
    if (st == cudaSuccess) st = cudaMemcpy(state.d_W_head, h_fp8.data(), total_fp8_bytes, cudaMemcpyHostToDevice);
    if (st == cudaSuccess) st = cudaMalloc(&state.d_X_act, 16 * K * sizeof(uint8_t));

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
    std::fprintf(stderr, "strata head: native FP8 E4M3 head active on CUDA%d (%d x %d, %.2f MiB VRAM)\n",
                 device, V, K, (double)total_fp8_bytes / 1048576.0);
    return true;
}

void free_native_head_fp8(NativeHeadFp8State& state) {
    if (state.d_W_head) {
        int prev = 0;
        cudaGetDevice(&prev);
        if (state.device >= 0) cudaSetDevice(state.device);
        cudaFree(state.d_W_head);
        if (state.d_X_act) cudaFree(state.d_X_act);
        if (state.d_scratchpad) cudaFree(state.d_scratchpad);
        if (state.d_out_tokens) cudaFree(state.d_out_tokens);
        if (state.d_out_norms) cudaFree(state.d_out_norms);
        cudaSetDevice(prev);
    }
    state.d_W_head = nullptr;
    state.d_X_act = nullptr;
    state.d_scratchpad = nullptr;
    state.d_out_tokens = nullptr;
    state.d_out_norms = nullptr;
    state.enabled = false;
}

bool run_native_head_fp8(const NativeHeadFp8State& state, const float* mixed, float* logits, int32_t* out_tokens, int T, void* stream, std::string& err) {
    if (!state.enabled || !state.d_W_head || !mixed) {
        err = "run_native_head_fp8: not initialized or null buffers";
        return false;
    }
    if (T <= 0 || T > 16) {
        err = "run_native_head_fp8: T=" + std::to_string(T) + " outside supported range 1..16";
        return false;
    }

    cudaStream_t cs = static_cast<cudaStream_t>(stream);
    const int total_acts = T * state.K;
    const int threads = 256;
    const int blocks = (total_acts + threads - 1) / threads;

    quantize_f32_to_fp8_act_kernel<<<blocks, threads, 0, cs>>>(
        mixed, static_cast<uint8_t*>(state.d_X_act), total_acts
    );

    cudaError_t kerr = strata::kernels::launch_fused_lm_head_gemv_softmax(
        static_cast<const uint8_t*>(state.d_W_head),
        static_cast<const uint8_t*>(state.d_X_act),
        static_cast<strata::kernels::SoftmaxArgmaxTuple*>(state.d_scratchpad),
        state.d_out_tokens,
        state.d_out_norms,
        T, state.V, state.K,
        1.0f, 1.0f, 1.0f,
        cs
    );

    if (kerr != cudaSuccess) {
        err = "run_native_head_fp8 kernel error: " + std::string(cudaGetErrorString(kerr));
        return false;
    }

    if (out_tokens != nullptr) {
        cudaMemcpyAsync(out_tokens, state.d_out_tokens, T * sizeof(int32_t), cudaMemcpyDeviceToDevice, cs);
    }

    if (logits != nullptr) {
        cudaMemsetAsync(logits, 0, static_cast<size_t>(T) * state.V * sizeof(float), cs);
        populate_argmax_logits_kernel<<<T, 1, 0, cs>>>(logits, state.d_out_tokens, T, state.V);
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
            for (int v = v_start; v < v_end; ++v) {
                dequantize_row_to_fp8(type, src_bytes + static_cast<size_t>(v) * row_bytes,
                                      h_fp8.data() + static_cast<size_t>(v) * K, K);
            }
        });
    }
    for (auto& w : workers) w.join();

    int prev_dev = 0;
    cudaGetDevice(&prev_dev);
    cudaSetDevice(device);

    cudaError_t st = cudaMalloc(&state.d_embed_table, total_fp8_bytes);
    if (st == cudaSuccess) st = cudaMemcpy(state.d_embed_table, h_fp8.data(), total_fp8_bytes, cudaMemcpyHostToDevice);
    if (st == cudaSuccess) st = cudaMalloc(&state.d_gamma, K * sizeof(half));

    if (st == cudaSuccess) {
        if (gamma_weights != nullptr) {
            st = cudaMemcpy(state.d_gamma, gamma_weights, K * sizeof(half), cudaMemcpyHostToDevice);
        } else {
            std::vector<half> ones(K, __float2half(1.0f));
            st = cudaMemcpy(state.d_gamma, ones.data(), K * sizeof(half), cudaMemcpyHostToDevice);
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
    if (!state.enabled || !state.d_embed_table || !state.d_gamma || !tokens) {
        err = "run_native_embed_fp8: not initialized or null tokens";
        return false;
    }

    cudaStream_t cs = static_cast<cudaStream_t>(stream);
    cudaError_t kerr = strata::kernels::launch_fused_embedding_rmsnorm_hc(
        static_cast<const uint4*>(state.d_embed_table),
        tokens,
        static_cast<const half*>(state.d_gamma),
        static_cast<half*>(out_main),
        static_cast<half*>(out_res),
        static_cast<half*>(out_hc),
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
