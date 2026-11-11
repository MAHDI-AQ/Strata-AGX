#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace strata::core {

void set_native_head_fp8_enabled(bool v);
bool native_head_fp8_enabled();

void set_native_embed_fp8_enabled(bool v);
bool native_embed_fp8_enabled();

void set_pipeline_dma_enabled(bool v);
bool pipeline_dma_enabled();

struct NativeHeadFp8State {
    void* d_W_head = nullptr;       // [V, K] uint8_t in FP8 E4M3
    void* d_X_act = nullptr;        // [max_T, K] uint8_t in FP8 E4M3
    void* d_scratchpad = nullptr;   // SoftmaxArgmaxTuple scratchpad
    int32_t* d_out_tokens = nullptr;// [max_T] int32_t argmax tokens
    float* d_out_norms = nullptr;   // [max_T] float normalizers
    int V = 0;
    int K = 0;
    int max_T = 16;
    int device = -1;
    bool enabled = false;
};

struct NativeEmbedFp8State {
    void* d_embed_table = nullptr;  // [V, K] uint8_t in FP8 E4M3
    void* d_gamma = nullptr;        // [K] half RMSNorm weights
    int V = 0;
    int K = 0;
    int device = -1;
    bool enabled = false;
};

bool init_native_head_fp8(NativeHeadFp8State& state, const void* src_weights, int type, int K, int V, int device, std::string& err);
void free_native_head_fp8(NativeHeadFp8State& state);
bool run_native_head_fp8(const NativeHeadFp8State& state, const float* mixed, float* logits, int32_t* out_tokens, int T, void* stream, std::string& err);

bool init_native_embed_fp8(NativeEmbedFp8State& state, const void* src_weights, int type, int K, int V, const void* gamma_weights, int device, std::string& err);
void free_native_embed_fp8(NativeEmbedFp8State& state);
bool run_native_embed_fp8(const NativeEmbedFp8State& state, const int32_t* tokens, int32_t num_tokens, void* out_main, void* out_res, void* out_hc, void* stream, std::string& err);

} // namespace strata::core
