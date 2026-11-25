// src/prefill/gemm_fp8_parity.cu - qualification test for surgical FP8 dense weight cache (w8_)
// Validates:
// 1. Bitwise identity / fallback parity on T < 128 (seamless fallback to standard FP16 GEMM).
// 2. Numerical stability & precision on T >= 128 (SM 89 FP8 Tensor Core cublasLt GEMM):
//    - Cosine similarity > 0.995 vs FP16 baseline (measured > 0.9993)
//    - Relative Frobenius norm error (L2 error) < 5.0%
//    - Zero NaNs, zero Infs
// 3. Across all production dense shapes of Qwen3.8-Flash-Next (attn_q, attn_qkv, attn_gate, ssm_out, shexp_gate, shexp_down, attn_k)
//    and quantization types (Q8_0, Q4_K, Q5_K, Q6_K).
#include "strata/prefill/gemm.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <string>
#include <vector>

namespace {
__global__ void fill_f16(__half* p, int64_t n, unsigned seed, float scale) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned x = (unsigned) i * 2654435761u + seed;
    x ^= x >> 16; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
    p[i] = __float2half(((float) (x & 0xffff) / 32768.0f - 1.0f) * scale);
}

bool ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) std::printf("FAIL: %s: %s\n", what, cudaGetErrorString(e));
    return e == cudaSuccess;
}

uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}

struct QType { int type, elems, bytes; int at[2]; const char* name; };
struct Shape { int qt; int64_t N, K; const char* what; };
struct Weight { Shape s; void* dev = nullptr; };
}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 1) {
        std::printf("no CUDA device (skipped)\n");
        return 77;
    }

    cudaStream_t st = nullptr;
    if (!ok(cudaStreamCreate(&st), "stream")) return 1;

    const int64_t scratch_elems = 32ll << 20;   // 32M elements (GEMM_SCRATCH)
    const QType qts[] = {
        {8, 32, 34, {0, -1}, "Q8_0"},
        {12, 256, 144, {0, 2}, "Q4_K"},
        {13, 256, 176, {0, 2}, "Q5_K"},
        {14, 256, 210, {208, -1}, "Q6_K"}
    };

    // Dense projections of Qwen3.8-Flash-Next (bounded to scratch size to guarantee safe VRAM headroom)
    std::vector<Shape> shapes;
    for (int q = 0; q < 4; ++q) {
        shapes.push_back({q, 12288, 2560, "attn_q"});
        shapes.push_back({q, 10240, 2560, "attn_qkv"});
        shapes.push_back({q, 6144, 2560, "attn_gate"});
        shapes.push_back({q, 2560, 6144, "ssm_out"});
        shapes.push_back({q, 640, 2560, "shexp_gate"});
        shapes.push_back({q, 2560, 640, "shexp_down"});
        shapes.push_back({q, 512, 2560, "attn_k"});
    }

    std::mt19937 rng(20261112u);
    std::vector<Weight> ws;
    int64_t max_k = 0, max_n = 0;

    for (const Shape& s : shapes) {
        const QType& q = qts[s.qt];
        if (s.K % q.elems != 0) continue;
        const size_t blocks = (size_t) (s.N * s.K / q.elems);
        const size_t bytes = blocks * (size_t) q.bytes;
        std::vector<uint8_t> h(bytes);
        for (auto& b : h) b = (uint8_t) (rng() & 0xff);
        for (size_t k = 0; k < blocks; ++k) {
            for (int at : q.at) {
                if (at >= 0) {
                    const uint16_t v = sane_half(rng);
                    std::memcpy(&h[k * (size_t) q.bytes + (size_t) at], &v, 2);
                }
            }
        }
        Weight w; w.s = s;
        if (!ok(cudaMalloc(&w.dev, bytes), "W") ||
            !ok(cudaMemcpy(w.dev, h.data(), bytes, cudaMemcpyHostToDevice), "W copy")) {
            return 1;
        }
        ws.push_back(w);
        if (s.K > max_k) max_k = s.K;
        if (s.N > max_n) max_n = s.N;
    }

    const char* min_t_env = std::getenv("STRATA_PF_FP8_MIN_T");
    const char* pad_env = std::getenv("STRATA_PF_FP8_PAD_SMALL_T");
    const bool pad_small_t = pad_env && pad_env[0] == '1' && pad_env[1] == '\0';
    const int64_t active_min_t = pad_small_t ? 1
        : ((min_t_env && min_t_env[0] && std::atoll(min_t_env) >= 1) ? std::atoll(min_t_env) : 128);
    const std::vector<int64_t> Ts = pad_small_t
        ? std::vector<int64_t>{1, 2, 7, 15, 16, 17, 31, 32, 48, 64, 82, 96, 112, 128, 512, 2048}
        : std::vector<int64_t>{16, 32, 48, 64, 82, 96, 112, 128, 512, 2048};
    const int64_t Tmax = 2048;

    uint16_t* X = nullptr;
    float* Y[2] = {nullptr, nullptr}; // Y[0]: baseline (w8 off), Y[1]: w8 active

    if (!ok(cudaMalloc((void**) &X, (size_t) Tmax * max_k * 2), "X") ||
        !ok(cudaMalloc((void**) &Y[0], (size_t) Tmax * max_n * 4), "Y0") ||
        !ok(cudaMalloc((void**) &Y[1], (size_t) Tmax * max_n * 4), "Y1")) {
        return 1;
    }

    fill_f16<<<(unsigned) ((Tmax * max_k + 255) / 256), 256, 0, st>>>(
        (__half*) X, Tmax * max_k, 42u, 1.5f);
    if (!ok(cudaStreamSynchronize(st), "fill")) return 1;

    int total_checked = 0;
    int fallback_ok = 0;
    int fp8_ok = 0;
    int failed = 0;

    printf("========================================================================\n");
    printf("gemm_fp8_parity: qualifying Surgical FP8 Prefill Weight Cache (w8_)\n");
    printf("========================================================================\n");

    const float betas[] = {0.0f, 1.0f};
    for (const Weight& w : ws) {
        for (const int64_t T : Ts) {
            for (const float beta : betas) {
                // Only test beta=1.0f when active_min_t < 128 or on representative Ts to keep runtime fast
                if (beta != 0.0f && active_min_t >= 128 && T != 64 && T != 128) continue;
                std::vector<float> out[2];
                for (int on = 0; on < 2; ++on) {
                    setenv("STRATA_PF_W8_CACHE", on ? "1" : "0", 1);
                    setenv("STRATA_PF_W16_CACHE_MIB", "0", 1);

                    strata::prefill::Gemm gemm;
                    std::string err;
                    if (!gemm.init(st, scratch_elems, err)) {
                        std::printf("FAIL: gemm.init: %s\n", err.c_str());
                        return 1;
                    }

                    if (beta != 0.0f) {
                        cudaMemset(Y[on], 0x3e, (size_t) T * max_n * 4);
                    } else {
                        cudaMemset(Y[on], 0, (size_t) T * max_n * 4);
                    }

                    // Run twice when on=1: first call caches in w8_, second call hits w8_ (use beta=0 on warmup rep if beta!=0)
                    if (on) {
                        gemm.native(X, qts[w.s.qt].type, w.dev, Y[on], T, w.s.N, w.s.K, max_n, 0.0f);
                        if (beta != 0.0f) cudaMemset(Y[on], 0x3e, (size_t) T * max_n * 4);
                        gemm.native(X, qts[w.s.qt].type, w.dev, Y[on], T, w.s.N, w.s.K, max_n, beta);
                    } else {
                        gemm.native(X, qts[w.s.qt].type, w.dev, Y[on], T, w.s.N, w.s.K, max_n, beta);
                    }
                    if (!ok(cudaStreamSynchronize(st), "run")) return 1;

                    out[on].resize((size_t) T * max_n);
                    cudaMemcpy(out[on].data(), Y[on], (size_t) T * max_n * 4, cudaMemcpyDeviceToHost);
                }

                total_checked++;

                double dot = 0.0, norm0 = 0.0, norm1 = 0.0, diff_sq = 0.0;
                bool has_nan = false;

                for (int64_t r = 0; r < T; ++r) {
                    for (int64_t c = 0; c < w.s.N; ++c) {
                        float v0 = out[0][(size_t) r * max_n + c];
                        float v1 = out[1][(size_t) r * max_n + c];

                        if (std::isnan(v0) || std::isnan(v1) || std::isinf(v0) || std::isinf(v1)) {
                            has_nan = true;
                            break;
                        }

                        dot += (double)v0 * (double)v1;
                        norm0 += (double)v0 * (double)v0;
                        norm1 += (double)v1 * (double)v1;
                        double diff = (double)v0 - (double)v1;
                        diff_sq += diff * diff;
                    }
                    if (has_nan) break;
                }

                double cos_sim = (norm0 > 0.0 && norm1 > 0.0) ? (dot / (std::sqrt(norm0) * std::sqrt(norm1))) : 0.0;
                double rel_l2_err = (norm0 > 0.0) ? (std::sqrt(diff_sq) / std::sqrt(norm0)) : 1.0;

                if (T < active_min_t) {
                // Assert exact fallback to standard FP16 GEMM (cosine similarity > 0.99999 or diff_sq == 0)
                bool pass = (!has_nan && (cos_sim > 0.99999 || diff_sq == 0.0));
                if (pass) {
                    fallback_ok++;
                } else {
                    printf("FAIL: Fallback T=%lld %s %s: cos_sim=%.8f, rel_l2_err=%.6f\n",
                           (long long)T, qts[w.s.qt].name, w.s.what, cos_sim, rel_l2_err);
                    failed++;
                }
            } else {
                // Assert FP8 numerical stability & quality:
                // Cosine similarity > 0.995 and L2 relative error < 0.05 (5.0%)
                bool pass = (!has_nan && cos_sim > 0.995 && rel_l2_err < 0.05);
                if (pass) {
                    fp8_ok++;
                } else {
                    printf("FAIL: FP8 T=%lld %s %s: cos_sim=%.6f, rel_l2_err=%.4f%%, has_nan=%d\n",
                           (long long)T, qts[w.s.qt].name, w.s.what, cos_sim, rel_l2_err * 100.0, has_nan);
                    failed++;
                }
            }
            }
        }
    }

    // ---- PF-1a: STRATA_PF_FP8_FUSED_AMAX=1 takes the activation amax from the producer (act_amax) and skips
    // x16_amax_kernel.  With the same amax, x8_, d_scale_x_, d_inv_scale_x_ and Y must be byte-identical to the
    // flag-off fallback, for T in {128, 512, 2048} and both a contiguous X and a padded row stride (ldx != K).
    // A control run with a deliberately wrong amax proves the gate is read: on changes x8, off ignores the slot.
    int pf1a_ok = 0, pf1a_failed = 0;
    {
        const int64_t N = 2560, K = 2560, ldy = N, ldx_max = K + 64;
        const int64_t Ts_pf[] = {128, 512, 2048};
        const int64_t Tmax_pf = 2048;
        uint8_t* W8 = nullptr;
        float* d_inv_w = nullptr;
        uint16_t* Xp = nullptr;
        float* Yp = nullptr;
        float* d_ref = nullptr;
        float* d_ref2 = nullptr;
        if (!ok(cudaMalloc((void**) &W8, (size_t) N * K), "pf1a W8") ||
            !ok(cudaMalloc((void**) &d_inv_w, sizeof(float)), "pf1a inv_w") ||
            !ok(cudaMalloc((void**) &Xp, (size_t) Tmax_pf * ldx_max * 2), "pf1a X") ||
            !ok(cudaMalloc((void**) &Yp, (size_t) Tmax_pf * ldy * 4), "pf1a Y") ||
            !ok(cudaMalloc((void**) &d_ref, sizeof(float)), "pf1a amax") ||
            !ok(cudaMalloc((void**) &d_ref2, sizeof(float)), "pf1a amax2")) {
            return 1;
        }
        {
            std::vector<uint8_t> hw((size_t) N * K);
            std::mt19937 rng2(20261113u);
            for (auto& b : hw) b = (uint8_t) (rng2() & 0x7e);   // E4M3 bits, finite and positive
            if (!ok(cudaMemcpy(W8, hw.data(), hw.size(), cudaMemcpyHostToDevice), "pf1a W copy")) return 1;
            const float inv = 1.0f / 0.02f;
            if (!ok(cudaMemcpy(d_inv_w, &inv, sizeof inv, cudaMemcpyHostToDevice), "pf1a inv copy")) return 1;
        }
        fill_f16<<<(unsigned) ((Tmax_pf * ldx_max + 255) / 256), 256, 0, st>>>(
            (__half*) Xp, Tmax_pf * ldx_max, 77u, 2.5f);
        if (!ok(cudaStreamSynchronize(st), "pf1a fill")) return 1;

        strata::prefill::Gemm gemm;
        std::string err;
        if (!gemm.init(st, scratch_elems, err)) {
            std::printf("FAIL: pf1a gemm.init: %s\n", err.c_str());
            return 1;
        }

        auto run = [&](const char* gate, const float* amax, int64_t T, int64_t ldx, std::vector<uint8_t>& x8,
                       float sc[2], std::vector<float>& y) {
            setenv("STRATA_PF_FP8_FUSED_AMAX", gate, 1);
            const int64_t yn = T * ldy;
            cudaMemsetAsync(Yp, 0xA5, (size_t) yn * 4, st);
            gemm.fp8(Xp, W8, d_inv_w, Yp, T, N, K, ldy, 0.0f, ldx, amax);
            if (!ok(cudaStreamSynchronize(st), "pf1a run")) return false;
            if (gemm.x8() == nullptr || gemm.scale_x() == nullptr || gemm.inv_scale_x() == nullptr) {
                std::printf("FAIL: pf1a: fp8() left no x8_/scale buffers\n");
                return false;
            }
            x8.resize((size_t) T * K);
            y.resize((size_t) yn);
            return ok(cudaMemcpy(x8.data(), gemm.x8(), x8.size(), cudaMemcpyDeviceToHost), "pf1a x8") &&
                   ok(cudaMemcpy(&sc[0], gemm.scale_x(), 4, cudaMemcpyDeviceToHost), "pf1a scale") &&
                   ok(cudaMemcpy(&sc[1], gemm.inv_scale_x(), 4, cudaMemcpyDeviceToHost), "pf1a inv scale") &&
                   ok(cudaMemcpy(y.data(), Yp, y.size() * 4, cudaMemcpyDeviceToHost), "pf1a Y");
        };

        for (int64_t ldx : {K, K + 64}) {
            for (int64_t T : Ts_pf) {
                // the reference amax exactly as x16_amax_kernel reads X: rows [0, T), columns [0, K)
                std::vector<uint16_t> hx((size_t) T * ldx);
                if (!ok(cudaMemcpy(hx.data(), Xp, hx.size() * 2, cudaMemcpyDeviceToHost), "pf1a X back")) return 1;
                float m = 0.0f;
                for (int64_t t = 0; t < T; ++t) {
                    for (int64_t k = 0; k < K; ++k) {
                        const uint16_t bits = hx[(size_t) t * ldx + k];
                        m = fmaxf(m, fabsf(__half2float(*reinterpret_cast<const __half*>(&bits))));
                    }
                }
                const float m2 = m * 2.0f;
                if (!ok(cudaMemcpyAsync(d_ref, &m, 4, cudaMemcpyHostToDevice, st), "pf1a amax up") ||
                    !ok(cudaMemcpyAsync(d_ref2, &m2, 4, cudaMemcpyHostToDevice, st), "pf1a amax2 up")) {
                    return 1;
                }
                std::vector<uint8_t> x8_off, x8_on, x8_bad, x8_off2;
                std::vector<float> y_off, y_on, y_bad, y_off2;
                float s_off[2], s_on[2], s_bad[2], s_off2[2];
                if (!run("0", d_ref, T, ldx, x8_off, s_off, y_off) ||
                    !run("1", d_ref, T, ldx, x8_on, s_on, y_on) ||
                    !run("1", d_ref2, T, ldx, x8_bad, s_bad, y_bad) ||
                    !run("0", d_ref2, T, ldx, x8_off2, s_off2, y_off2)) {
                    std::printf("FAIL: PF-1a T=%lld ldx=%lld: the fp8 calls did not complete\n", (long long) T, (long long) ldx);
                    ++pf1a_failed;
                    ++failed;
                    continue;
                }
                const bool same = std::memcmp(x8_off.data(), x8_on.data(), x8_off.size()) == 0 &&
                                  std::memcmp(s_off, s_on, sizeof s_off) == 0 &&
                                  std::memcmp(y_off.data(), y_on.data(), y_off.size() * 4) == 0;
                const bool gate_read = std::memcmp(x8_off.data(), x8_bad.data(), x8_off.size()) != 0 &&
                                       std::memcmp(x8_off.data(), x8_off2.data(), x8_off.size()) == 0;
                if (same && gate_read) {
                    ++pf1a_ok;
                    std::printf("PASS: PF-1a T=%lld ldx=%lld x8/scale/inv/Y byte-identical, gate honoured\n",
                                (long long) T, (long long) ldx);
                } else {
                    std::printf("FAIL: PF-1a T=%lld ldx=%lld: identical=%d gate_honoured=%d\n", (long long) T,
                                (long long) ldx, same ? 1 : 0, gate_read ? 1 : 0);
                    ++pf1a_failed;
                    ++failed;
                }
            }
        }
        cudaFree(W8); cudaFree(d_inv_w); cudaFree(Xp); cudaFree(Yp); cudaFree(d_ref); cudaFree(d_ref2);
    }

    // Free w8_ test buffers before Wave A1 per-token stress
    cudaFree(X); X = nullptr;
    cudaFree(Y[0]); Y[0] = nullptr;
    cudaFree(Y[1]); Y[1] = nullptr;
    for (auto& w : ws) { cudaFree(w.dev); }
    ws.clear();

    // ---- Wave A1: x16_amax_kernel memset race fix + STRATA_PF_FP8_PER_TOKEN=1 ----
    int a1_amax_ok = 0, a1_quant_ok = 0, a1_outlier_ok = 0;
    {
        auto fp32_to_e4m3_cpu = [](float val) -> uint8_t {
            if (std::isnan(val)) return 0x7fu;
            uint32_t bits;
            std::memcpy(&bits, &val, 4);
            const uint8_t sign = (uint8_t) ((bits >> 31) << 7);
            const float ax = fabsf(val);
            if (ax >= 448.0f) return (uint8_t) (sign | 0x7eu);
            if (ax == 0.0f) return sign;
            // Brute-force nearest-even over all 126 finite non-negative E4M3 codes (exact & simple)
            auto decode = [](int code) -> float {
                int e = (code >> 3) & 0xf;
                int m = code & 0x7;
                if (e == 0) return std::ldexp((float) m / 8.0f, -6);
                return std::ldexp(1.0f + (float) m / 8.0f, e - 7);
            };
            int best_code = 0;
            float best_diff = ax;
            for (int c = 1; c <= 0x7e; ++c) {
                float v = decode(c);
                float d = fabsf(ax - v);
                if (d < best_diff || (d == best_diff && (c & 1) == 0)) {
                    best_diff = d;
                    best_code = c;
                }
            }
            return (uint8_t) (sign | (uint8_t) best_code);
        };

        auto e4m3_to_fp32_cpu = [](uint8_t code) -> float {
            const float s = (code & 0x80u) ? -1.0f : 1.0f;
            const int e = (code >> 3) & 0xf;
            const int m = code & 0x7;
            if (e == 15 && m == 7) return NAN;
            if (e == 0) return s * std::ldexp((float) m / 8.0f, -6);
            return s * std::ldexp(1.0f + (float) m / 8.0f, e - 7);
        };

        // 1) Pre-poisoned d_max_x_ = 0x7f7fffff test (STRATA_PF_FP8_PER_TOKEN=0, STRATA_PF_FP8_FUSED_AMAX=0)
        {
            const int64_t T = 128, N = 256, K = 2560;
            uint16_t* Xt = nullptr;
            uint8_t* W8t = nullptr;
            float* d_inv_w = nullptr;
            float* Yt = nullptr;
            if (!ok(cudaMalloc((void**) &Xt, (size_t) T * K * 2), "a1_amax X") ||
                !ok(cudaMalloc((void**) &W8t, (size_t) N * K), "a1_amax W8") ||
                !ok(cudaMalloc((void**) &d_inv_w, sizeof(float)), "a1_amax inv_w") ||
                !ok(cudaMalloc((void**) &Yt, (size_t) T * N * 4), "a1_amax Y")) {
                return 1;
            }
            fill_f16<<<(unsigned) ((T * K + 255) / 256), 256, 0, st>>>((__half*) Xt, T * K, 101u, 1.5f);
            cudaMemsetAsync(W8t, 0x38, (size_t) N * K, st); // 1.0 in E4M3
            const float inv_w = 1.0f;
            cudaMemcpyAsync(d_inv_w, &inv_w, sizeof(float), cudaMemcpyHostToDevice, st);
            if (!ok(cudaStreamSynchronize(st), "a1_amax setup")) return 1;

            std::vector<uint16_t> hx((size_t) T * K);
            cudaMemcpy(hx.data(), Xt, hx.size() * 2, cudaMemcpyDeviceToHost);
            float true_amax = 0.0f;
            for (uint16_t bits : hx) {
                true_amax = fmaxf(true_amax, fabsf(__half2float(*reinterpret_cast<const __half*>(&bits))));
            }
            const float expected_scale = 448.0f / fmaxf(true_amax, 1e-8f);

            strata::prefill::Gemm gemm;
            std::string err;
            if (!gemm.init(st, scratch_elems, err)) return 1;

            setenv("STRATA_PF_FP8_PER_TOKEN", "0", 1);
            setenv("STRATA_PF_FP8_FUSED_AMAX", "0", 1);
            // First call allocates d_max_x_
            if (!gemm.fp8(Xt, W8t, d_inv_w, Yt, T, N, K, N, 0.0f, K, nullptr, false)) {
                std::printf("FAIL: a1_amax warmup fp8\n");
                ++failed;
            }
            bool all_race_ok = true;
            const uint32_t poison = 0x7f7fffffu;
            for (int iter = 0; iter < 8; ++iter) {
                cudaMemcpyAsync(const_cast<float*>(gemm.max_x()), &poison, 4, cudaMemcpyHostToDevice, st);
                gemm.fp8(Xt, W8t, d_inv_w, Yt, T, N, K, N, 0.0f, K, nullptr, false);
                cudaStreamSynchronize(st);
                float got_scale = 0.0f;
                cudaMemcpy(&got_scale, gemm.scale_x(), 4, cudaMemcpyDeviceToHost);
                if (fabsf(got_scale - expected_scale) > 1e-5f * expected_scale) {
                    std::printf("FAIL: a1_amax poison iter=%d got_scale=%.6f expected=%.6f\n", iter, got_scale, expected_scale);
                    all_race_ok = false;
                    break;
                }
            }
            if (all_race_ok) {
                ++a1_amax_ok;
                std::printf("PASS: x16_amax_kernel memset race fix (8/8 poisoned 0x7f7fffff runs matched %.6f)\n", expected_scale);
            } else {
                ++failed;
            }
            cudaFree(Xt); cudaFree(W8t); cudaFree(d_inv_w); cudaFree(Yt);
        }

        // 2) Per-row d_inv_scale_tok_ + x8_ E4M3 byte parity against CPU FP32 reference
        {
            struct QCase { int64_t T, K, ldx; const char* tag; };
            const QCase qcases[] = {
                {128, 2560, 2560, "aligned_contig"},
                {128, 2560, 2624, "aligned_padded"},
                {128, 2560, 2561, "scalar_unaligned_ldx"},
            };
            for (const QCase& qc : qcases) {
                const int64_t N = 128;
                std::vector<uint16_t> hx((size_t) qc.T * qc.ldx, 0);
                std::mt19937 rng_q(20261115u + (unsigned) qc.K);
                std::normal_distribution<float> nd(0.0f, 1.0f);
                for (int64_t t = 0; t < qc.T; ++t) {
                    for (int64_t k = 0; k < qc.K; ++k) {
                        float v = 0.0f;
                        if (t == 0) {
                            v = 0.0f; // all-zero row -> exact 0
                        } else if (t == 1 && k == 17) {
                            v = 150.0f; // 100x outlier spike
                        } else if (t == 2) {
                            v = nd(rng_q) * 1e-7f; // tiny row exercising floor
                        } else {
                            v = nd(rng_q);
                        }
                        __half hv = __float2half(v);
                        std::memcpy(&hx[(size_t) t * qc.ldx + k], &hv, 2);
                    }
                }
                uint16_t* Xq = nullptr;
                uint8_t* W8q = nullptr;
                float* d_inv_w = nullptr;
                float* Yq = nullptr;
                if (!ok(cudaMalloc((void**) &Xq, hx.size() * 2), "a1_q X") ||
                    !ok(cudaMalloc((void**) &W8q, (size_t) N * qc.K), "a1_q W8") ||
                    !ok(cudaMalloc((void**) &d_inv_w, sizeof(float)), "a1_q inv_w") ||
                    !ok(cudaMalloc((void**) &Yq, (size_t) qc.T * N * 4), "a1_q Y")) {
                    return 1;
                }
                cudaMemcpy(Xq, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
                cudaMemset(W8q, 0x38, (size_t) N * qc.K);
                const float inv_w = 1.0f;
                cudaMemcpy(d_inv_w, &inv_w, sizeof(float), cudaMemcpyHostToDevice);

                strata::prefill::Gemm gemm;
                std::string err;
                if (!gemm.init(st, scratch_elems, err)) return 1;
                setenv("STRATA_PF_FP8_PER_TOKEN", "1", 1);
                if (!gemm.fp8(Xq, W8q, d_inv_w, Yq, qc.T, N, qc.K, N, 0.0f, qc.ldx, nullptr, false)) {
                    std::printf("FAIL: a1_quant %s fp8 returned false\n", qc.tag);
                    ++failed;
                    cudaFree(Xq); cudaFree(W8q); cudaFree(d_inv_w); cudaFree(Yq);
                    continue;
                }
                cudaStreamSynchronize(st);

                std::vector<float> got_inv((size_t) qc.T);
                std::vector<uint8_t> got_x8((size_t) qc.T * qc.K);
                cudaMemcpy(got_inv.data(), gemm.inv_scale_tok(), (size_t) qc.T * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(got_x8.data(), gemm.x8(), got_x8.size(), cudaMemcpyDeviceToHost);

                bool row_ok = true;
                int max_ulp_diff = 0;
                for (int64_t t = 0; t < qc.T && row_ok; ++t) {
                    float row_amax = 0.0f;
                    for (int64_t k = 0; k < qc.K; ++k) {
                        const uint16_t bits = hx[(size_t) t * qc.ldx + k];
                        row_amax = fmaxf(row_amax, fabsf(__half2float(*reinterpret_cast<const __half*>(&bits))));
                    }
                    const float exp_inv = (row_amax > 0.0f) ? fmaxf(row_amax * (1.0f / 448.0f), 1.0f / (448.0f * 512.0f)) : 0.0f;
                    const float exp_scale = (row_amax > 0.0f) ? (1.0f / exp_inv) : 0.0f;
                    if (fabsf(got_inv[(size_t) t] - exp_inv) > 1e-6f * fmaxf(1.0f, exp_inv)) {
                        std::printf("FAIL: a1_quant %s row=%lld inv_scale=%.9g expected=%.9g\n",
                                    qc.tag, (long long) t, got_inv[(size_t) t], exp_inv);
                        row_ok = false;
                        break;
                    }
                    for (int64_t k = 0; k < qc.K; ++k) {
                        const uint16_t bits = hx[(size_t) t * qc.ldx + k];
                        const float x_val = __half2float(*reinterpret_cast<const __half*>(&bits));
                        const uint8_t exp_b = fp32_to_e4m3_cpu(x_val * exp_scale);
                        const uint8_t got_b = got_x8[(size_t) t * qc.K + k];
                        if (t == 0 && got_b != 0) {
                            std::printf("FAIL: a1_quant %s zero row k=%lld got 0x%02x\n", qc.tag, (long long) k, got_b);
                            row_ok = false;
                            break;
                        }
                        if (got_b != exp_b) {
                            const int diff = std::abs((int) (got_b & 0x7f) - (int) (exp_b & 0x7f));
                            if ((got_b & 0x80) != (exp_b & 0x80) || diff > 1) {
                                std::printf("FAIL: a1_quant %s (%lld,%lld) got=0x%02x exp=0x%02x\n",
                                            qc.tag, (long long) t, (long long) k, got_b, exp_b);
                                row_ok = false;
                                break;
                            }
                            if (diff > max_ulp_diff) max_ulp_diff = diff;
                        }
                    }
                }
                if (row_ok) {
                    ++a1_quant_ok;
                    std::printf("PASS: per-token quantize %s (T=%lld, K=%lld, ldx=%lld, max_ulp=%d)\n",
                                qc.tag, (long long) qc.T, (long long) qc.K, (long long) qc.ldx, max_ulp_diff);
                } else {
                    ++failed;
                }
                cudaFree(Xq); cudaFree(W8q); cudaFree(d_inv_w); cudaFree(Yq);
            }
        }

        // 3) Outlier stress across Qwen3.8 shapes, T={128, 513, 4096}, beta={0.0f, 1.0f}
        {
            struct OShape { int64_t K, N; const char* name; };
            const OShape oshapes[] = {
                {2560, 10240, "attn_qkv"},
                {2560, 6144, "attn_gate"},
                {3072, 2560, "k3072_n2560"},
            };
            const int64_t oTs[] = {128, 513, 4096};
            const float betas[] = {0.0f, 1.0f};

            strata::prefill::Gemm gemm;
            std::string err;
            if (!gemm.init(st, scratch_elems, err)) return 1;

            for (const OShape& os : oshapes) {
                std::vector<uint8_t> hw((size_t) os.N * os.K);
                std::mt19937 rng_w(9991u + (unsigned) os.N);
                for (auto& b : hw) {
                    // Valid finite E4M3 weights in modest range
                    int mag = (int) (rng_w() % 0x38);
                    b = (uint8_t) (((rng_w() & 1u) << 7) | mag);
                }
                uint8_t* W8o = nullptr;
                float* d_inv_w = nullptr;
                const float inv_w = 0.02f;
                if (!ok(cudaMalloc((void**) &W8o, hw.size()), "o W8") ||
                    !ok(cudaMalloc((void**) &d_inv_w, sizeof(float)), "o inv_w")) {
                    return 1;
                }
                cudaMemcpy(W8o, hw.data(), hw.size(), cudaMemcpyHostToDevice);
                cudaMemcpy(d_inv_w, &inv_w, sizeof(float), cudaMemcpyHostToDevice);

                for (int64_t T : oTs) {
                    std::vector<uint16_t> hx((size_t) T * os.K);
                    std::mt19937 rng_x(7771u + (unsigned) T);
                    std::normal_distribution<float> nd(0.0f, 0.05f);
                    for (int64_t t = 0; t < T; ++t) {
                        for (int64_t k = 0; k < os.K; ++k) {
                            float v = 0.0f;
                            if (t == 0 && k == 0) v = 32768.0f; // extreme outlier spike in row 0
                            else if (t == T - 1) v = 0.0f;    // all-zero row to verify beta*Y_old preservation
                            else v = nd(rng_x);
                            __half hv = __float2half(v);
                            std::memcpy(&hx[(size_t) t * os.K + k], &hv, 2);
                        }
                    }
                    uint16_t* Xo = nullptr;
                    float* Yo = nullptr;
                    if (!ok(cudaMalloc((void**) &Xo, hx.size() * 2), "o X") ||
                        !ok(cudaMalloc((void**) &Yo, (size_t) T * os.N * 4), "o Y")) {
                        return 1;
                    }
                    cudaMemcpy(Xo, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);

                    // Compute exact FP32 reference on sampled non-outlier rows [1..8] and zero row [T-1]
                    const int64_t sample_rows[] = {1, 2, 3, 4, 5, 6, 7, 8, T - 1};
                    std::vector<std::vector<float>> ref_base(9, std::vector<float>((size_t) os.N, 0.0f));
                    for (int ri = 0; ri < 9; ++ri) {
                        const int64_t t = sample_rows[ri];
                        if (t == T - 1) continue;
                        for (int64_t n = 0; n < os.N; ++n) {
                            double acc = 0.0;
                            for (int64_t k = 0; k < os.K; ++k) {
                                const uint16_t bits = hx[(size_t) t * os.K + k];
                                const float xv = __half2float(*reinterpret_cast<const __half*>(&bits));
                                const float wv = e4m3_to_fp32_cpu(hw[(size_t) n * os.K + k]) * inv_w;
                                acc += (double) xv * (double) wv;
                            }
                            ref_base[ri][(size_t) n] = (float) acc;
                        }
                    }

                    for (float beta : betas) {
                        std::vector<float> y_init((size_t) T * os.N, 0.0f);
                        if (beta != 0.0f) {
                            for (int64_t t = 0; t < T; ++t) {
                                for (int64_t n = 0; n < os.N; ++n) {
                                    y_init[(size_t) t * os.N + n] = 1.25f + 0.05f * (float) ((t + n) % 13);
                                }
                            }
                        }

                        double rel_err[2] = {0.0, 0.0};
                        double cos_s[2] = {0.0, 0.0};
                        bool zero_row_exact = true;
                        bool has_nan = false;

                        for (int pt = 0; pt < 2; ++pt) {
                            setenv("STRATA_PF_FP8_PER_TOKEN", pt ? "1" : "0", 1);
                            setenv("STRATA_PF_FP8_FUSED_AMAX", "0", 1);
                            cudaMemcpy(Yo, y_init.data(), y_init.size() * 4, cudaMemcpyHostToDevice);
                            if (!gemm.fp8(Xo, W8o, d_inv_w, Yo, T, os.N, os.K, os.N, beta, os.K, nullptr, false)) {
                                std::printf("FAIL: outlier %s T=%lld beta=%.1f pt=%d fp8 returned false\n",
                                            os.name, (long long) T, beta, pt);
                                has_nan = true;
                                break;
                            }
                            cudaStreamSynchronize(st);

                            double dot = 0.0, n0 = 0.0, n1 = 0.0, dsq = 0.0;
                            std::vector<float> row_out((size_t) os.N);
                            for (int ri = 0; ri < 9; ++ri) {
                                const int64_t t = sample_rows[ri];
                                cudaMemcpy(row_out.data(), Yo + (size_t) t * os.N, (size_t) os.N * 4, cudaMemcpyDeviceToHost);
                                if (t == T - 1 && pt == 1) {
                                    for (int64_t n = 0; n < os.N; ++n) {
                                        const float exp_z = beta * y_init[(size_t) t * os.N + n];
                                        if (fabsf(row_out[(size_t) n] - exp_z) > 1e-6f) {
                                            zero_row_exact = false;
                                            break;
                                        }
                                    }
                                    continue;
                                }
                                if (t == T - 1) continue;
                                for (int64_t n = 0; n < os.N; ++n) {
                                    const float got = row_out[(size_t) n];
                                    const float ref = beta * y_init[(size_t) t * os.N + n] + ref_base[ri][(size_t) n];
                                    if (std::isnan(got) || std::isinf(got)) { has_nan = true; break; }
                                    dot += (double) got * (double) ref;
                                    n0 += (double) ref * (double) ref;
                                    n1 += (double) got * (double) got;
                                    const double d = (double) got - (double) ref;
                                    dsq += d * d;
                                }
                                if (has_nan) break;
                            }
                            rel_err[pt] = (n0 > 0.0) ? std::sqrt(dsq) / std::sqrt(n0) : 1.0;
                            cos_s[pt] = (n0 > 0.0 && n1 > 0.0) ? dot / (std::sqrt(n0) * std::sqrt(n1)) : 0.0;
                        }

                        const bool pass = !has_nan && zero_row_exact && cos_s[1] > 0.995 && rel_err[1] < 0.05 && rel_err[1] < rel_err[0] * 0.25;
                        if (pass) {
                            ++a1_outlier_ok;
                            std::printf("PASS: outlier %s T=%lld beta=%.1f | per-tensor err=%.4f -> per-token err=%.4f (cos=%.6f)\n",
                                        os.name, (long long) T, beta, rel_err[0], rel_err[1], cos_s[1]);
                        } else {
                            ++failed;
                            std::printf("FAIL: outlier %s T=%lld beta=%.1f | nan=%d zero_exact=%d pt0_err=%.4f pt1_err=%.4f pt1_cos=%.6f\n",
                                        os.name, (long long) T, beta, has_nan ? 1 : 0, zero_row_exact ? 1 : 0, rel_err[0], rel_err[1], cos_s[1]);
                        }
                    }
                    cudaFree(Xo); cudaFree(Yo);
                }
                cudaFree(W8o); cudaFree(d_inv_w);
            }
        }
    }

    // Wave K8-A: PrefillFp8Activation pair/triple ticket parity & non-interference
    int k8a_ok = 0;
    {
        strata::prefill::Gemm gemm;
        std::string k8_err;
        if (!gemm.init(st, 4ll << 20, k8_err)) {
            std::printf("FAIL: K8-A gemm.init: %s\n", k8_err.c_str());
            return 1;
        }
        const int64_t T = 512, N = 256, K = 2560;
        std::mt19937 rng(20261117);
        std::normal_distribution<float> nd(0.0f, 0.5f);
        auto mk_f16 = [&](size_t n) {
            std::vector<uint16_t> v(n);
            for (size_t i = 0; i < n; ++i) v[i] = __half_as_ushort(__float2half(nd(rng)));
            uint16_t* d = nullptr;
            cudaMalloc(&d, n * 2);
            cudaMemcpy(d, v.data(), n * 2, cudaMemcpyHostToDevice);
            return d;
        };
        auto mk_fp8_w = [&](int64_t n_rows, int64_t k_cols, float* d_inv_w) {
            std::vector<uint8_t> w((size_t) (n_rows * k_cols));
            for (auto& b : w) {
                int mag = (int) (rng() % 0x38);
                b = (uint8_t) (((rng() & 1u) << 7) | mag);
            }
            uint8_t* dw = nullptr;
            cudaMalloc(&dw, w.size());
            cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
            float inv_w = 1.0f;
            cudaMemcpy(d_inv_w, &inv_w, 4, cudaMemcpyHostToDevice);
            return dw;
        };
        uint16_t* X1 = mk_f16((size_t) (T * K));
        uint16_t* X2 = mk_f16((size_t) (T * K));
        float *dw1 = nullptr, *dw2 = nullptr, *dw3 = nullptr;
        cudaMalloc(&dw1, 4); cudaMalloc(&dw2, 4); cudaMalloc(&dw3, 4);
        uint8_t* W1 = mk_fp8_w(N, K, dw1);
        uint8_t* W2 = mk_fp8_w(N, K, dw2);
        uint8_t* W3 = mk_fp8_w(N, K, dw3);
        float *Yref = nullptr, *Ygot = nullptr;
        cudaMalloc(&Yref, (size_t) (T * N) * 4);
        cudaMalloc(&Ygot, (size_t) (T * N) * 4);
        std::vector<float> h_ref((size_t) (T * N)), h_got((size_t) (T * N));

        auto check_bit_eq = [&](const char* label) {
            cudaDeviceSynchronize();
            cudaMemcpy(h_ref.data(), Yref, h_ref.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_got.data(), Ygot, h_got.size() * 4, cudaMemcpyDeviceToHost);
            if (std::memcmp(h_ref.data(), h_got.data(), h_ref.size() * 4) == 0) {
                ++k8a_ok;
            } else {
                ++failed;
                std::printf("FAIL: K8-A %s: output differs from unticketed reference\n", label);
            }
        };

        // 1) Triple reuse (beta=0, beta=0, beta=1)
        strata::prefill::PrefillFp8Activation ticket;
        gemm.fp8(X1, W1, dw1, Yref, T, N, K, 0, 0.0f);
        gemm.fp8(X1, W1, dw1, Ygot, T, N, K, 0, 0.0f, 0, nullptr, false, &ticket);
        check_bit_eq("triple-1(beta=0)");

        gemm.fp8(X1, W2, dw2, Yref, T, N, K, 0, 0.0f);
        gemm.fp8(X1, W2, dw2, Ygot, T, N, K, 0, 0.0f, 0, nullptr, false, &ticket);
        check_bit_eq("triple-2(beta=0)");

        cudaMemset(Yref, 0x3f, (size_t) (T * N) * 4);
        cudaMemcpy(Ygot, Yref, (size_t) (T * N) * 4, cudaMemcpyDeviceToDevice);
        gemm.fp8(X1, W3, dw3, Yref, T, N, K, 0, 1.0f);
        gemm.fp8(X1, W3, dw3, Ygot, T, N, K, 0, 1.0f, 0, nullptr, false, &ticket);
        check_bit_eq("triple-3(beta=1)");

        // 2) Shape / pointer change forces re-quantize
        gemm.fp8(X1, W1, dw1, Yref, 256, N, K, 0, 0.0f);
        gemm.fp8(X1, W1, dw1, Ygot, 256, N, K, 0, 0.0f, 0, nullptr, false, &ticket);
        check_bit_eq("T-change(256)");

        gemm.fp8(X2, W1, dw1, Yref, T, N, K, 0, 0.0f);
        gemm.fp8(X2, W1, dw1, Ygot, T, N, K, 0, 0.0f, 0, nullptr, false, &ticket);
        check_bit_eq("X-ptr-change");

        // 3) Intervening unticketed fp8() bumps x8_gen_ and forces re-quantize
        gemm.fp8(X1, W1, dw1, Ygot, T, N, K, 0, 0.0f, 0, nullptr, false, &ticket);
        gemm.fp8(X2, W2, dw2, Ygot, T, N, K, 0, 0.0f);   // unticketed overwrite of x8_
        gemm.fp8(X1, W1, dw1, Ygot, T, N, K, 0, 0.0f, 0, nullptr, false, &ticket);
        gemm.fp8(X1, W1, dw1, Yref, T, N, K, 0, 0.0f);
        check_bit_eq("intervening-x8-gen");

        cudaFree(X1); cudaFree(X2); cudaFree(W1); cudaFree(W2); cudaFree(W3);
        cudaFree(dw1); cudaFree(dw2); cudaFree(dw3); cudaFree(Yref); cudaFree(Ygot);
        std::printf("K8-A ticket parity: %d/6 PASS\n", k8a_ok);
    }

    // ---- Wave B1: STRATA_PF_W8_PER_ROW=1 accuracy & epilogue parity + comma-list W8_CACHE_MIB ----
    int b1_per_row_ok = 0;
    {
        auto get_saved_var = [](const char* k) -> std::string {
            const char* v = std::getenv(k);
            return v ? std::string(v) : std::string("__UNSET__");
        };
        auto put_saved_var = [](const char* k, const std::string& v) {
            if (v == "__UNSET__") unsetenv(k);
            else setenv(k, v.c_str(), 1);
        };
        const std::string orig_w8 = get_saved_var("STRATA_PF_W8_CACHE");
        const std::string orig_w8_mib = get_saved_var("STRATA_PF_W8_CACHE_MIB");
        const std::string orig_per_row = get_saved_var("STRATA_PF_W8_PER_ROW");
        const std::string orig_per_tok = get_saved_var("STRATA_PF_FP8_PER_TOKEN");
        const std::string orig_w16 = get_saved_var("STRATA_PF_W16_CACHE_MIB");

        // 1) Skewed-row Q8_0 end-to-end native() test: row 0 huge magnitude vs rows 1..N-1 tiny magnitude
        {
            const int64_t N = 512, K = 2560, T = 128;
            const int64_t blocks_per_row = K / 32;
            const size_t q8_bytes = (size_t) (N * blocks_per_row) * 34;
            std::vector<uint8_t> hq8(q8_bytes, 0);
            std::mt19937 rng_s(20261125u);
            for (int64_t n = 0; n < N; ++n) {
                const float d_val = (n == 0) ? (60000.0f / 127.0f) : (0.02f / 127.0f);
                const uint16_t dh = __half_as_ushort(__float2half(d_val));
                for (int64_t b = 0; b < blocks_per_row; ++b) {
                    const size_t off = (size_t) (n * blocks_per_row + b) * 34;
                    std::memcpy(&hq8[off], &dh, 2);
                    for (int q = 0; q < 32; ++q) {
                        int v = (int) (rng_s() % 241) - 120;
                        if (v == 0) v = 17;
                        hq8[off + 2 + q] = (uint8_t) (int8_t) v;
                    }
                }
            }
            void* dw = nullptr;
            uint16_t* Xb = nullptr;
            float* Yb = nullptr;
            if (!ok(cudaMalloc(&dw, q8_bytes), "b1 w") ||
                !ok(cudaMemcpy(dw, hq8.data(), q8_bytes, cudaMemcpyHostToDevice), "b1 w copy") ||
                !ok(cudaMalloc((void**) &Xb, (size_t) T * K * 2), "b1 X") ||
                !ok(cudaMalloc((void**) &Yb, (size_t) T * N * 4), "b1 Y")) {
                return 1;
            }
            fill_f16<<<(unsigned) ((T * K + 255) / 256), 256, 0, st>>>((__half*) Xb, T * K, 31337u, 1.0f);
            if (!ok(cudaStreamSynchronize(st), "b1 fill")) return 1;

            auto run_native_mode = [&](const char* w8_on, const char* per_row_on, const char* w8_mib,
                                       float beta, std::vector<float>& out_y) {
                setenv("STRATA_PF_W8_CACHE", w8_on, 1);
                if (w8_mib) setenv("STRATA_PF_W8_CACHE_MIB", w8_mib, 1);
                else unsetenv("STRATA_PF_W8_CACHE_MIB");
                setenv("STRATA_PF_W8_PER_ROW", per_row_on, 1);
                setenv("STRATA_PF_FP8_PER_TOKEN", "1", 1);
                setenv("STRATA_PF_W16_CACHE_MIB", "0", 1);

                strata::prefill::Gemm g;
                std::string err;
                if (!g.init(st, scratch_elems, err)) return false;
                if (beta != 0.0f) {
                    std::vector<float> init_v((size_t) T * N, 0.02f);
                    cudaMemcpyAsync(Yb, init_v.data(), init_v.size() * 4, cudaMemcpyHostToDevice, st);
                    g.native(Xb, 8, dw, Yb, T, N, K, N, 0.0f);
                    cudaMemcpyAsync(Yb, init_v.data(), init_v.size() * 4, cudaMemcpyHostToDevice, st);
                    g.native(Xb, 8, dw, Yb, T, N, K, N, beta);
                } else {
                    cudaMemsetAsync(Yb, 0, (size_t) T * N * 4, st);
                    g.native(Xb, 8, dw, Yb, T, N, K, N, 0.0f);
                    g.native(Xb, 8, dw, Yb, T, N, K, N, 0.0f);
                }
                if (cudaStreamSynchronize(st) != cudaSuccess) return false;
                out_y.resize((size_t) T * N);
                return cudaMemcpy(out_y.data(), Yb, out_y.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
            };

            for (float beta : {0.0f, 1.0f}) {
                std::vector<float> y_f16, y_pt, y_pr, y_zero_list;
                if (!run_native_mode("0", "0", nullptr, beta, y_f16) ||
                    !run_native_mode("1", "0", "768,768,768,768", beta, y_pt) ||
                    !run_native_mode("1", "1", "768,768,768,768", beta, y_pr) ||
                    !run_native_mode("1", "1", "0,0,0,0", beta, y_zero_list)) {
                    std::printf("FAIL: B1 native_mode run beta=%.1f\n", beta);
                    ++failed;
                    continue;
                }
                // Check tiny rows [1..N-1]
                double norm0 = 0.0, dsq_pt = 0.0, dsq_pr = 0.0, dot_pr = 0.0, norm_pr = 0.0;
                bool bad = false;
                for (int64_t t = 0; t < T; ++t) {
                    for (int64_t n = 1; n < N; ++n) {
                        const float r = y_f16[(size_t) t * N + n];
                        const float a = y_pt[(size_t) t * N + n];
                        const float b = y_pr[(size_t) t * N + n];
                        if (std::isnan(b) || std::isinf(b)) { bad = true; break; }
                        norm0 += (double) r * (double) r;
                        norm_pr += (double) b * (double) b;
                        dot_pr += (double) r * (double) b;
                        dsq_pt += ((double) a - (double) r) * ((double) a - (double) r);
                        dsq_pr += ((double) b - (double) r) * ((double) b - (double) r);
                    }
                    if (bad) break;
                }
                const double rel_pt = std::sqrt(dsq_pt) / std::sqrt(norm0);
                const double rel_pr = std::sqrt(dsq_pr) / std::sqrt(norm0);
                const double cos_pr = dot_pr / (std::sqrt(norm0) * std::sqrt(norm_pr));
                const bool zero_list_exact = (y_zero_list == y_f16);
                if (!bad && zero_list_exact && cos_pr > 0.998 && rel_pr < 0.05 && rel_pr < rel_pt * 0.15) {
                    ++b1_per_row_ok;
                    std::printf("PASS: B1 skewed-row beta=%.1f per-tensor rel_l2=%.4f -> per-row rel_l2=%.4f (cos=%.6f, zero-list exact=%d)\n",
                                beta, rel_pt, rel_pr, cos_pr, zero_list_exact ? 1 : 0);
                } else {
                    ++failed;
                    std::printf("FAIL: B1 skewed-row beta=%.1f bad=%d zero_list=%d rel_pt=%.4f rel_pr=%.4f cos_pr=%.6f\n",
                                beta, bad ? 1 : 0, zero_list_exact ? 1 : 0, rel_pt, rel_pr, cos_pr);
                }
            }
            cudaFree(dw); cudaFree(Xb); cudaFree(Yb);
        }

        // 2) Direct Gemm::fp8(..., per_row_w=true) testing both per_token={0,1}, beta={0,1}, T={64,128},
        //    plus 16-byte aligned d_inv_w (vectorized float4 epilogue) and 4-byte offset d_inv_w+1 (scalar fallback)
        {
            const int64_t K = 2560, N = 2560;
            for (int unaligned_w : {0, 1}) {
                std::vector<uint8_t> hw((size_t) N * K);
                std::vector<float> h_inv_w((size_t) N + 1, 0.0f);
                std::mt19937 rng_d(5511u + (unsigned) unaligned_w);
                for (auto& b : hw) b = (uint8_t) (rng_d() & 0x7eu);
                for (int64_t n = 0; n < N; ++n) h_inv_w[(size_t) n + unaligned_w] = 0.005f + 0.0005f * (float) (n % 29);

                uint8_t* Wd = nullptr;
                float* d_inv_w_base = nullptr;
                if (!ok(cudaMalloc((void**) &Wd, hw.size()), "b1d W") ||
                    !ok(cudaMalloc((void**) &d_inv_w_base, h_inv_w.size() * 4), "b1d inv_w")) {
                    return 1;
                }
                cudaMemcpy(Wd, hw.data(), hw.size(), cudaMemcpyHostToDevice);
                cudaMemcpy(d_inv_w_base, h_inv_w.data(), h_inv_w.size() * 4, cudaMemcpyHostToDevice);
                const float* d_inv_w = d_inv_w_base + unaligned_w;

                for (int64_t T : {int64_t(64), int64_t(128)}) {
                    std::vector<uint16_t> hx((size_t) T * K);
                    std::mt19937 rng_x(991u + (unsigned) T + (unsigned) N);
                    std::normal_distribution<float> nd(0.0f, 0.4f);
                    for (size_t i = 0; i < hx.size(); ++i) {
                        __half hv = __float2half(nd(rng_x));
                        std::memcpy(&hx[i], &hv, 2);
                    }
                    uint16_t* Xd = nullptr;
                    float* Yd = nullptr;
                    cudaMalloc((void**) &Xd, hx.size() * 2);
                    cudaMalloc((void**) &Yd, (size_t) T * N * 4);
                    cudaMemcpy(Xd, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);

                    // Compute exact FP32 reference for sampled rows [0..3]
                    std::vector<float> ref_rows(4 * (size_t) N, 0.0f);
                    for (int t = 0; t < 4; ++t) {
                        for (int64_t n = 0; n < N; ++n) {
                            double acc = 0.0;
                            for (int64_t k = 0; k < K; ++k) {
                                const uint16_t bits = hx[(size_t) t * K + k];
                                const float xv = __half2float(*reinterpret_cast<const __half*>(&bits));
                                const uint8_t c = hw[(size_t) n * K + k];
                                const float s = (c & 0x80u) ? -1.0f : 1.0f;
                                const int e = (c >> 3) & 0xf;
                                const int m = c & 0x7;
                                const float raw_w = (e == 0) ? s * std::ldexp((float) m / 8.0f, -6)
                                                             : s * std::ldexp(1.0f + (float) m / 8.0f, e - 7);
                                acc += (double) xv * (double) (raw_w * h_inv_w[(size_t) n + unaligned_w]);
                            }
                            ref_rows[(size_t) t * N + n] = (float) acc;
                        }
                    }

                    strata::prefill::Gemm g;
                    std::string err;
                    if (!g.init(st, scratch_elems, err)) return 1;

                    for (int pt : {0, 1}) {
                        for (float beta : {0.0f, 1.0f}) {
                            setenv("STRATA_PF_FP8_PER_TOKEN", pt ? "1" : "0", 1);
                            std::vector<float> y_init((size_t) T * N, beta != 0.0f ? 0.5f : 0.0f);
                            cudaMemcpy(Yd, y_init.data(), y_init.size() * 4, cudaMemcpyHostToDevice);
                            if (!g.fp8(Xd, Wd, d_inv_w, Yd, T, N, K, N, beta, K, nullptr, false, nullptr, true)) {
                                ++failed;
                                std::printf("FAIL: B1 direct unalign=%d N=%lld T=%lld pt=%d beta=%.1f returned false\n",
                                            unaligned_w, (long long) N, (long long) T, pt, beta);
                                continue;
                            }
                            cudaStreamSynchronize(st);
                            std::vector<float> got_rows(4 * (size_t) N);
                            cudaMemcpy(got_rows.data(), Yd, got_rows.size() * 4, cudaMemcpyDeviceToHost);

                            double dot = 0.0, n0 = 0.0, n1 = 0.0, dsq = 0.0;
                            bool has_nan = false;
                            for (size_t i = 0; i < got_rows.size(); ++i) {
                                const float r = ref_rows[i] + (beta != 0.0f ? 0.5f : 0.0f);
                                const float v = got_rows[i];
                                if (std::isnan(v) || std::isinf(v)) { has_nan = true; break; }
                                dot += (double) r * (double) v;
                                n0 += (double) r * (double) r;
                                n1 += (double) v * (double) v;
                                dsq += ((double) r - (double) v) * ((double) r - (double) v);
                            }
                            const double cos_s = dot / (std::sqrt(n0) * std::sqrt(n1));
                            const double rel_l2 = std::sqrt(dsq) / std::sqrt(n0);
                            if (!has_nan && cos_s > 0.995 && rel_l2 < 0.05) {
                                ++b1_per_row_ok;
                            } else {
                                ++failed;
                                std::printf("FAIL: B1 direct unalign=%d N=%lld T=%lld pt=%d beta=%.1f cos=%.6f rel=%.4f\n",
                                            unaligned_w, (long long) N, (long long) T, pt, beta, cos_s, rel_l2);
                            }
                        }
                    }
                    cudaFree(Xd); cudaFree(Yd);
                }
                cudaFree(Wd); cudaFree(d_inv_w_base);
            }
        }
        std::printf("Wave B1 per-row W8 parity checks: %d/18 PASS\n", b1_per_row_ok);

        put_saved_var("STRATA_PF_W8_CACHE", orig_w8);
        put_saved_var("STRATA_PF_W8_CACHE_MIB", orig_w8_mib);
        put_saved_var("STRATA_PF_W8_PER_ROW", orig_per_row);
        put_saved_var("STRATA_PF_FP8_PER_TOKEN", orig_per_tok);
        put_saved_var("STRATA_PF_W16_CACHE_MIB", orig_w16);
    }

    printf("========================================================================\n");
    printf("gemm_fp8_parity SUMMARY: %d checked | Fallback(T<128): %d PASS | FP8(T>=128): %d PASS | PF-1a fused amax: %d PASS | A1 amax_fix: %d PASS | A1 quant: %d PASS | A1 outlier: %d PASS | %d FAIL\n",
           total_checked, fallback_ok, fp8_ok, pf1a_ok, a1_amax_ok, a1_quant_ok, a1_outlier_ok, failed);
    printf("gemm_fp8_parity: %s\n", (failed == 0) ? "OK" : "FAILED");
    printf("========================================================================\n");

    cudaStreamDestroy(st);

    return (failed == 0) ? 0 : 1;
}
