// src/prefill/gemm.cu - see include/strata/prefill/gemm.hpp.
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/core/arch_defaults.hpp"   // prompt_f16_resolve (inline)
#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)
#include "strata/prefill/mmq_ids_margin.hpp"
#include "strata/prefill/moe_mmq.hpp"
#endif

#if defined(STRATA_USE_HIP) || (!defined(__HIPCC__) && !defined(__APPLE__))
#include "wmma_gemm.h"
#endif

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if !defined(STRATA_USE_HIP)
#include <cublasLt.h>
#include <cuda_fp8.h>
#endif

#include <algorithm>
#include <atomic>
#include <mutex>

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
// The HIP compatibility shim maps CUDA shuffle spellings to Strata helpers.
// hipBLASLt's public headers declare native HIP shuffle functions, so keep
// those declarations from being macro-expanded in this translation unit.
#undef __shfl_xor_sync
#undef __shfl_down_sync
#undef __shfl_up_sync
#undef __shfl_sync
#undef __ballot_sync
#endif

#include <algorithm>
#include <cctype>
#include <initializer_list>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
#include "hipblaslt_tuning.hpp"
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <map>
#include <set>
#include <tuple>
#endif
#if defined(__HIPCC__) && defined(STRATA_ROCBLAS_AVAILABLE)
#include "rocblas_tuning.hpp"
#include "strata/core/amd_community.hpp"
#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>
#include <filesystem>
#include <map>
#include <string>
#include <system_error>
#include <tuple>
#endif

#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#endif

namespace strata::prefill {
namespace {
#if defined(__HIPCC__)
// STRATA_F16_RANGE (gemm_set_f16_range): the largest |value| and the count beyond 65504 / non-finite, for the BF16
// weights converted to FP16 (slot 0, bf16_to_f16_rows) and the FP16 GEMM outputs widened (slot 1, widen_rows_f16 -
// an output rocBLAS wrote as FP16 is Inf there, not saturated).  As the activation counter in kernels.cu.
__device__ int g_f16_range_on_g = 0;
__device__ unsigned g_f16_range_max_g[2] = {0, 0};
__device__ unsigned long long g_f16_range_over_g[2] = {0, 0};
__device__ __forceinline__ void f16_range_note_g(int k, float f) {
    const float a = fabsf(f);
    const unsigned bits = __float_as_uint(a);
    if (bits > g_f16_range_max_g[k]) atomicMax(&g_f16_range_max_g[k], bits);
    if (a > 65504.0f || isnan(f)) atomicAdd(&g_f16_range_over_g[k], 1ull);
}
// Y's rows, written by an FP16-out GEMM as FP16 at the start of each FP32 row (ldc = 2 ldy halves), widened in place.
// Float c overwrites halves 2c and 2c+1, so a row is walked from its end in blocks: a block's halves are read into
// registers, the block syncs, then writes its floats - which only cover halves of blocks already read.
// STRATA_DBG_NAN: the FP16 outputs that are not finite (an FP32-accumulated sum past 65504 becomes inf in FP16).
__device__ unsigned long long g_f16_nonfinite = 0;
__global__ void widen_rows_f16(float* __restrict__ Y, int64_t n, int64_t ldy, int count_nonfinite) {
    float* y = Y + (int64_t) blockIdx.x * ldy;
    const __half* h = reinterpret_cast<const __half*>(y);
    const int64_t nb = (n + blockDim.x - 1) / blockDim.x;
    for (int64_t b = nb - 1; b >= 0; --b) {
        const int64_t c = b * blockDim.x + threadIdx.x;
        const float v = c < n ? __half2float(h[c]) : 0.0f;
        if (count_nonfinite && c < n && !isfinite(v)) atomicAdd(&g_f16_nonfinite, 1ull);
        if (c < n && g_f16_range_on_g) f16_range_note_g(1, v);
        __syncthreads();
        if (c < n) y[c] = v;
        __syncthreads();
    }
}
// BF16 weight rows -> FP16, saturated (bench/results/2026-10-04-rdna2-fp16-prompt: none leaves FP16's range)
__global__ void bf16_to_f16_rows(const uint16_t* __restrict__ s, __half* __restrict__ d, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const float f = __uint_as_float((uint32_t) s[i] << 16);
        if (g_f16_range_on_g) f16_range_note_g(0, f);
        d[i] = __float2half(isnan(f) ? f : fminf(fmaxf(f, -65504.0f), 65504.0f));   // as hf_sat: a NaN stays NaN
    }
}
#endif
}  // namespace

void gemm_set_f16_range(bool on) {
#if defined(__HIPCC__)
    const int v = on ? 1 : 0;
    if (cudaMemcpyToSymbol(g_f16_range_on_g, &v, sizeof v) != cudaSuccess) {
        std::fprintf(stderr, "prefill gemm: setting the FP16 range check failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        std::exit(1);
    }
#else
    (void) on;
#endif
}
bool gemm_f16_range_read(float max_abs[2], unsigned long long over[2], bool reset) {
#if defined(__HIPCC__)
    unsigned bits[2] = {0, 0}; unsigned long long o[2] = {0, 0};
    if (cudaMemcpyFromSymbol(bits, g_f16_range_max_g, sizeof bits) != cudaSuccess ||
        cudaMemcpyFromSymbol(o, g_f16_range_over_g, sizeof o) != cudaSuccess) { cudaGetLastError(); return false; }
    for (int k = 0; k < 2; ++k) { union { unsigned u; float f; } c; c.u = bits[k]; max_abs[k] = c.f; over[k] = o[k]; }
    if (reset) {
        const unsigned z[2] = {0, 0}; const unsigned long long zz[2] = {0, 0};
        cudaMemcpyToSymbol(g_f16_range_max_g, z, sizeof z); cudaMemcpyToSymbol(g_f16_range_over_g, zz, sizeof zz);
    }
    return true;
#else
    (void) max_abs; (void) over; (void) reset;
    return false;
#endif
}

namespace {

void ck(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: %s: cuBLAS status %d\n", what, (int) s);
        std::exit(1);
    }
}

// #247/#325: on Windows (seen on gfx1201), hipBLAS can return success with the correct BF16/FP16 product for some
// shapes (hc up once T >= 96, the router) and still leave hipErrorInvalidValue set, which the next kernel's error
// check turns into an exit. The multiply has finished, so that one stale error is cleared after a GEMM that succeeded;
// any other error still stops the engine. Windows only: on Linux a stale hipErrorInvalidValue is a real error from
// an earlier call and keeps being reported. A no-op everywhere else (CUDA compiles none of it).
#if defined(__HIPCC__) && defined(_WIN32)
void absorb_hipblas_sticky(const char* what) {
    const hipError_t sticky = hipGetLastError();
    if (sticky == hipSuccess || sticky == hipErrorInvalidValue) return;
    std::fprintf(stderr, "prefill gemm: %s left %s\n", what, hipGetErrorString(sticky));
    std::exit(1);
}
#define STRATA_ABSORB_HIPBLAS_STICKY(what) absorb_hipblas_sticky(what)
#else
#define STRATA_ABSORB_HIPBLAS_STICKY(what) ((void) 0)
#endif

// A setup call whose failure the engine survives (the handle keeps its defaults), as before #240 - but said.
void note(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) std::fprintf(stderr, "prefill gemm: %s: cuBLAS status %d (continuing)\n", what, (int) s);
}

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
struct HipLtCallKey {
    strata::prefill::hipblaslt::InputType type;
    int t;
    int n;
    int k;
    int ldy;
    uint32_t beta_bits;

    bool operator<(const HipLtCallKey& other) const {
        return std::tie(type, n, k, ldy, t, beta_bits) <
               std::tie(other.type, other.n, other.k, other.ldy, other.t, other.beta_bits);
    }
};

struct HipLtCachedAlgo {
    bool supported = false;
    hipblasLtMatmulAlgo_t algo{};
    size_t workspace_bytes = 0;
};

struct HipLtState {
    hipblasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    strata::prefill::hipblaslt::TuningTable table;
    std::map<HipLtCallKey, HipLtCachedAlgo> cache;
    uint64_t lt_launches = 0;
    uint64_t fallbacks = 0;
    std::set<std::tuple<strata::prefill::hipblaslt::InputType, int, int, int, int>> fallback_shapes;

    ~HipLtState() {
        if (std::getenv("STRATA_HIPBLASLT_VERBOSE")) {
            std::fprintf(stderr, "prefill gemm: hipBLASLt summary launches=%llu fallbacks=%llu unique_fallback_shapes=%zu\n",
                         (unsigned long long) lt_launches, (unsigned long long) fallbacks, fallback_shapes.size());
            for (const auto& shape : fallback_shapes) {
                const auto type = std::get<0>(shape);
                std::fprintf(stderr, "prefill gemm: fallback shape dtype=%s T=%d N=%d K=%d ldy=%d\n",
                             type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16",
                             std::get<1>(shape), std::get<2>(shape), std::get<3>(shape), std::get<4>(shape));
            }
        }
        if (handle) hipblasLtDestroy(handle);
    }
};

struct HipLtDescriptors {
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a = nullptr;
    hipblasLtMatrixLayout_t b = nullptr;
    hipblasLtMatrixLayout_t c = nullptr;

    ~HipLtDescriptors() {
        if (op) hipblasLtMatmulDescDestroy(op);
        if (a) hipblasLtMatrixLayoutDestroy(a);
        if (b) hipblasLtMatrixLayoutDestroy(b);
        if (c) hipblasLtMatrixLayoutDestroy(c);
    }

    bool init(hipDataType type, int t, int n, int k, int ldy, int ldb = 0) {
        const hipblasOperation_t trans_a = HIPBLAS_OP_T;
        const hipblasOperation_t trans_b = HIPBLAS_OP_N;
        if (hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)) !=
                HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)) !=
                HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&a, type, k, n, k) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&b, type, k, t, ldb > k ? ldb : k) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&c, HIP_R_32F, n, t, ldy) != HIPBLAS_STATUS_SUCCESS) {
            return false;
        }
        return true;
    }
};

// The engine package's own tuning folder (<engine folder>/tuning), for a start that did not come through the launcher (a hand-made
// config, a CLI run): without a table the prompt GEMMs fall back to plain hipBLAS and a Strix Halo reads prompts at half speed
// (340 against 690 tok/s, IQ3_S 4K).  The launcher sets STRATA_HIPBLASLT_TUNING itself; this is the same table found by the engine.
std::string own_tuning_dir() {
#if defined(_WIN32)
    char p[4096] = {};
    const DWORD n = GetModuleFileNameA(nullptr, p, (DWORD) sizeof(p));
    std::string s(p, n);
    const size_t k = s.find_last_of("\\/");
    return k == std::string::npos ? std::string() : s.substr(0, k + 1) + "tuning/";
#else
    char p[4096] = {};
    const ssize_t n = readlink("/proc/self/exe", p, sizeof(p) - 1);
    std::string s(p, n > 0 ? (size_t) n : 0);
    const size_t k = s.find_last_of('/');
    return k == std::string::npos ? std::string() : s.substr(0, k + 1) + "tuning/";
#endif
}

std::unique_ptr<HipLtState> create_hipblaslt_state(void* workspace, size_t workspace_bytes) {
    std::string path_own;
    const char* path = std::getenv("STRATA_HIPBLASLT_TUNING");
    bool from_package = false, table_forced = false;
    if (!path || !*path) {
        const char* sw = std::getenv("STRATA_HIPBLASLT_TABLE");     // the launcher's switch (strata_control/hipblaslt_tuning.py): off / on / auto
        std::string low = sw != nullptr ? sw : "";                   // trimmed and lower case, as the launcher reads it
        while (!low.empty() && std::isspace((unsigned char) low.back())) low.pop_back();
        while (!low.empty() && std::isspace((unsigned char) low.front())) low.erase(low.begin());
        for (char& ch : low) ch = (char) std::tolower((unsigned char) ch);
        const auto word = [&](std::initializer_list<const char*> ws) {
            for (const char* w : ws) if (low == w) return true;
            return false;
        };
        if (word({"off", "0", "no", "false", "none", "disable", "disabled"})) return nullptr;   // never the packaged table
        table_forced = word({"on", "1", "yes", "true", "force", "enable", "enabled"});          // any card that has a table beside the engine
        from_package = true;                                          // the file is found below, once the card and the version are known
    }

    auto state = std::make_unique<HipLtState>();
    state->workspace = workspace;
    state->workspace_bytes = workspace_bytes;
    if (hipblasLtCreate(&state->handle) != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: hipBLASLt handle creation failed; using hipBLASEx\n");
        return nullptr;
    }

    int version = 0;
    if (hipblasLtGetVersion(state->handle, &version) != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: hipBLASLt version query failed; using hipBLASEx\n");
        return nullptr;
    }
    int device = 0;
    hipDeviceProp_t properties{};
    if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&properties, device) != hipSuccess) {
        std::fprintf(stderr, "prefill gemm: HIP device query failed; using hipBLASEx\n");
        return nullptr;
    }
    std::string arch(properties.gcnArchName);
    const auto suffix = arch.find(':');
    if (suffix != std::string::npos) arch.resize(suffix);

    if (from_package) {
        // Unset or "auto": only Strix Halo (gfx1151) takes the packaged table by itself, the owner's decision of 2026-10-19 (the same
        // rule as AUTO_ARCHS in strata_control/hipblaslt_tuning.py): on other cards the table changes the answers' last digits for ~1%
        // (gfx1201), so they take it with STRATA_HIPBLASLT_TABLE=on only.
        if (!table_forced && arch != "gfx1151") return nullptr;
        const std::string dir = own_tuning_dir();
        if (dir.empty()) return nullptr;
        path_own = dir + arch + "-hipblaslt-" + std::to_string(version) + ".txt";
        std::FILE* f = std::fopen(path_own.c_str(), "rb");
        if (f == nullptr) return nullptr;       // no table for this card and hipBLASLt version beside the engine: plain hipBLAS as before
        std::fclose(f);
        path = path_own.c_str();
    }
    std::string error;
    if (!state->table.load(path, arch, version, error)) {
        std::fprintf(stderr, "prefill gemm: %s; using hipBLASEx\n", error.c_str());
        return nullptr;
    }
    std::fprintf(stderr, "prefill gemm: hipBLASLt tuning enabled (%zu rows, %s, version %d)\n",
                 state->table.rows().size(), arch.c_str(), version);
    return state;
}

HipLtCachedAlgo resolve_hipblaslt_algo(HipLtState& state, strata::prefill::hipblaslt::InputType type, int t,
                                       int n, int k, int ldy, float beta) {
    uint32_t beta_bits = 0;
    static_assert(sizeof(beta_bits) == sizeof(beta));
    std::memcpy(&beta_bits, &beta, sizeof(beta));
    const HipLtCallKey key{type, t, n, k, ldy, beta_bits};
    const auto cached = state.cache.find(key);
    if (cached != state.cache.end()) return cached->second;

    HipLtCachedAlgo resolved;
    const bool verbose = std::getenv("STRATA_HIPBLASLT_VERBOSE") != nullptr;
    const auto* row = state.table.closest(type, n, k, ldy, t);
    if (!row) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; no calibration for dtype=%s T=%d N=%d K=%d ldy=%d\n",
                         type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16", t, n, k, ldy);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    HipLtDescriptors desc;
    const hipDataType input_type = type == strata::prefill::hipblaslt::InputType::bf16 ? HIP_R_16BF : HIP_R_16F;
    if (!desc.init(input_type, t, n, k, ldy)) {
        return state.cache.emplace(key, resolved).first->second;
    }

    std::vector<int> solution_ids{row->solution_id};
    std::vector<hipblasLtMatmulHeuristicResult_t> candidates;
    if (hipblaslt_ext::getAlgosFromIndex(state.handle, solution_ids, candidates) != HIPBLAS_STATUS_SUCCESS ||
        candidates.empty() || candidates.front().state != HIPBLAS_STATUS_SUCCESS ||
        hipblaslt_ext::getIndexFromAlgo(candidates.front().algo) != row->solution_id) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution %d unavailable for T=%d N=%d K=%d ldy=%d\n",
                         row->solution_id, t, n, k, ldy);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    const float alpha = 1.0f;
    size_t required_workspace = 0;
    auto algo = candidates.front().algo;
    if (hipblaslt_ext::matmulIsAlgoSupported(state.handle, desc.op, &alpha, desc.a, desc.b, &beta, desc.c, desc.c,
                                             algo, required_workspace) != HIPBLAS_STATUS_SUCCESS) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution %d rejects actual T=%d N=%d K=%d ldy=%d beta=%.9g\n",
                         row->solution_id, t, n, k, ldy, beta);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    resolved.supported = true;
    resolved.algo = algo;
    resolved.workspace_bytes = required_workspace;
    if (verbose) {
        std::fprintf(stderr,
                     "prefill gemm: Lt solution=%d dtype=%s T=%d N=%d K=%d ldy=%d beta=%.9g workspace=%zu\n",
                     row->solution_id, type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16", t, n,
                     k, ldy, beta, required_workspace);
    }
    return state.cache.emplace(key, resolved).first->second;
}

// the hipBLASLt solution index the tuning table gives this call (-1: none / unsupported)
int hipblaslt_solution_for(void* opaque_state, strata::prefill::hipblaslt::InputType type, int64_t t, int64_t n, int64_t k,
                           int64_t ldy, float beta) {
    auto* state = static_cast<HipLtState*>(opaque_state);
    if (!state || t <= 0 || n <= 0 || k <= 0 || t > INT_MAX || n > INT_MAX || k > INT_MAX || ldy > INT_MAX || ldy < n) return -1;
    const auto resolved = resolve_hipblaslt_algo(*state, type, (int) t, (int) n, (int) k, (int) ldy, beta);
    if (!resolved.supported || resolved.workspace_bytes > state->workspace_bytes) return -1;
    auto algo = resolved.algo;
    return hipblaslt_ext::getIndexFromAlgo(algo);
}

bool try_hipblaslt(void* opaque_state, strata::prefill::hipblaslt::InputType type, const uint16_t* x,
                   const uint16_t* w, float* y, int64_t t, int64_t n, int64_t k, int64_t ldy, float beta,
                   void* stream, int64_t ldx = 0) {
    auto* state = static_cast<HipLtState*>(opaque_state);
    if (!state || t <= 0 || n <= 0 || k <= 0 || t > INT_MAX || n > INT_MAX || k > INT_MAX || ldy > INT_MAX ||
        ldy < n) {
        return false;
    }
    const auto resolved = resolve_hipblaslt_algo(*state, type, (int) t, (int) n, (int) k, (int) ldy, beta);
    if (!resolved.supported) {
        ++state->fallbacks;
        state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
        return false;
    }
    if (resolved.workspace_bytes > state->workspace_bytes) {
        ++state->fallbacks;
        state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
        if (std::getenv("STRATA_HIPBLASLT_VERBOSE")) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution needs %zu workspace bytes, have %zu\n",
                         resolved.workspace_bytes, state->workspace_bytes);
        }
        return false;
    }

    HipLtDescriptors desc;
    const hipDataType input_type = type == strata::prefill::hipblaslt::InputType::bf16 ? HIP_R_16BF : HIP_R_16F;
    if (!desc.init(input_type, (int) t, (int) n, (int) k, (int) ldy, (int) ldx)) return false;
    const float alpha = 1.0f;
    const hipblasStatus_t status = hipblasLtMatmul(state->handle, desc.op, &alpha, w, desc.a, x, desc.b, &beta, y,
                                                   desc.c, y, desc.c, &resolved.algo, state->workspace,
                                                   state->workspace_bytes, (hipStream_t) stream);
    if (status == HIPBLAS_STATUS_SUCCESS) {
        ++state->lt_launches;
        return true;
    }

    std::fprintf(stderr, "prefill gemm: hipBLASLt launch failed with status %d\n", (int) status);
    if (beta != 0.0f) {
        std::fprintf(stderr, "prefill gemm: refusing a fallback after hipBLASLt failed with nonzero beta\n");
        std::exit(1);
    }
    auto* mutable_state = static_cast<HipLtState*>(opaque_state);
    uint32_t beta_bits = 0;
    std::memcpy(&beta_bits, &beta, sizeof(beta_bits));
    auto cached = mutable_state->cache.find(HipLtCallKey{type, (int) t, (int) n, (int) k, (int) ldy, beta_bits});
    if (cached != mutable_state->cache.end()) cached->second.supported = false;
    ++mutable_state->fallbacks;
    mutable_state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
    return false;
}
#endif

#if defined(__HIPCC__) && defined(STRATA_ROCBLAS_AVAILABLE)
// The rocBLAS solution table of f16_inplace (STRATA_ROCBLAS_TUNING: a table file, or a directory holding
// <arch>-rocblas-<version>.txt for this card and rocBLAS build; src/prefill/rocblas_tuning.hpp).  rocBLAS's own
// pick for some shapes is up to 6x slower than the best kernel it holds for them (gfx1030: tools/hip/tune_rocblas.cpp).
struct RocBlasState {
    rocblas_handle handle = nullptr;
    strata::prefill::rocblas_tuning::Table table;
    struct Pick { bool tuned = false; int solution = 0; int refusals = 0; };
    std::map<std::tuple<int, int, int, int>, Pick> cache;   // (T, N, K, ldy) -> the solution to run, or the default
    uint64_t tuned_launches = 0;
    uint64_t fallbacks = 0;

    ~RocBlasState() {
        if (std::getenv("STRATA_ROCBLAS_VERBOSE")) {
            std::fprintf(stderr, "prefill gemm: rocBLAS tuning summary tuned_launches=%llu fallbacks=%llu\n",
                         (unsigned long long) tuned_launches, (unsigned long long) fallbacks);
        }
        if (handle) rocblas_destroy_handle(handle);
    }
};

std::unique_ptr<RocBlasState> create_rocblas_state(void* stream) {
    const char* path = std::getenv("STRATA_ROCBLAS_TUNING");
    if (!path || !*path) return nullptr;
    auto state = std::make_unique<RocBlasState>();
    if (rocblas_create_handle(&state->handle) != rocblas_status_success) {
        std::fprintf(stderr, "prefill gemm: rocBLAS handle creation failed; its default kernels run\n");
        return nullptr;
    }
    rocblas_set_stream(state->handle, (hipStream_t) stream);
    char version[128] = {0};
    if (rocblas_get_version_string(version, sizeof version) != rocblas_status_success) {
        std::fprintf(stderr, "prefill gemm: rocBLAS version query failed; its default kernels run\n");
        return nullptr;
    }
    int device = 0;
    hipDeviceProp_t properties{};
    if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&properties, device) != hipSuccess) {
        std::fprintf(stderr, "prefill gemm: HIP device query failed; rocBLAS's default kernels run\n");
        return nullptr;
    }
    std::string arch(properties.gcnArchName);
    if (const auto suffix = arch.find(':'); suffix != std::string::npos) arch.resize(suffix);
    std::filesystem::path file(path);
    std::error_code ec;
    if (std::filesystem::is_directory(file, ec)) {
        // the table of this card and this rocBLAS build (its version string, trailing dots dropped)
        std::string tag(version);
        while (!tag.empty() && tag.back() == '.') tag.pop_back();
        file /= arch + "-rocblas-" + tag + ".txt";
    }
    // The community-made tables (strata/core/amd_community.hpp) load only with STRATA_AMD_COMMUNITY_TUNING on (the default).
    if (!strata::amd_community::rocblas_table_allowed(arch, version)) {
        std::fprintf(stderr, "prefill gemm: the rocBLAS table for %s / %s is community-made and STRATA_AMD_COMMUNITY_TUNING=0; "
                             "rocBLAS's default kernels run\n", arch.c_str(), version);
        return nullptr;
    }
    std::string error;
    if (!state->table.load(file.string(), arch, version, error)) {
        std::fprintf(stderr, "prefill gemm: %s (%s); rocBLAS's default kernels run\n", error.c_str(),
                     file.string().c_str());
        return nullptr;
    }
    // rocBLAS loads its kernel library on the first product through a handle; a solution-index call before that
    // fails (rocblas_status_internal_error) although the index is valid.  One tiny default product opens it.
    {
        uint16_t* tiny = nullptr;
        if (hipMalloc(&tiny, 4 * 64 * 64 * 2) == hipSuccess) {
            const float one = 1.0f, zero = 0.0f;
            (void) rocblas_gemm_ex(state->handle, rocblas_operation_transpose, rocblas_operation_none, 64, 64, 64, &one,
                                   tiny, rocblas_datatype_f16_r, 64, tiny + 64 * 64, rocblas_datatype_f16_r, 64, &zero,
                                   tiny + 2 * 64 * 64, rocblas_datatype_f16_r, 128, tiny + 2 * 64 * 64,
                                   rocblas_datatype_f16_r, 128, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
            (void) hipStreamSynchronize((hipStream_t) stream);
            (void) hipFree(tiny);
        }
    }
    size_t tuned_rows = 0;
    for (const auto& row : state->table.rows()) tuned_rows += row.tuned ? 1 : 0;
    std::fprintf(stderr, "prefill gemm: rocBLAS tuning enabled (%zu rows, %zu with a solution, %s, rocBLAS %s)\n",
                 state->table.rows().size(), tuned_rows, arch.c_str(), version);
    return state;
}

// Gemm::f16_inplace's product through the table's solution for its shape.  False when the shape has no row, or the
// launch was refused (then the default kernel runs, and for that shape from now on).
bool try_rocblas_f16o(void* opaque_state, const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N,
                      int64_t K, int64_t ldy) {
    auto* state = static_cast<RocBlasState*>(opaque_state);
    if (!state || T <= 0 || N <= 0 || K <= 0 || T > INT_MAX || N > INT_MAX || K > INT_MAX || ldy > INT_MAX / 2) return false;
    const auto key = std::make_tuple((int) T, (int) N, (int) K, (int) ldy);
    auto it = state->cache.find(key);
    if (it == state->cache.end()) {
        RocBlasState::Pick pick;
        if (const auto* row = state->table.closest((int) N, (int) K, (int) ldy, (int) T); row && row->tuned) {
            pick.tuned = true;
            pick.solution = row->solution;
        }
        if (std::getenv("STRATA_ROCBLAS_VERBOSE")) {
            if (pick.tuned)
                std::fprintf(stderr, "prefill gemm: rocBLAS solution %d for T=%lld N=%lld K=%lld ldy=%lld\n",
                             pick.solution, (long long) T, (long long) N, (long long) K, (long long) ldy);
            else
                std::fprintf(stderr, "prefill gemm: rocBLAS default kernel; no row for T=%lld N=%lld K=%lld ldy=%lld\n",
                             (long long) T, (long long) N, (long long) K, (long long) ldy);
        }
        it = state->cache.emplace(key, pick).first;
    }
    if (!it->second.tuned) return false;
    const float one = 1.0f, zero = 0.0f;
    const rocblas_status status = rocblas_gemm_ex(
        state->handle, rocblas_operation_transpose, rocblas_operation_none, (int) N, (int) T, (int) K, &one, W,
        rocblas_datatype_f16_r, (int) K, X, rocblas_datatype_f16_r, (int) K, &zero, Y, rocblas_datatype_f16_r,
        (int) (2 * ldy), Y, rocblas_datatype_f16_r, (int) (2 * ldy), rocblas_datatype_f32_r,
        rocblas_gemm_algo_solution_index, it->second.solution, 0);
    if (status == rocblas_status_success) {
        ++state->tuned_launches;
        return true;
    }
    // A refusal writes nothing (the default kernel runs now).  The first one can be rocBLAS still setting up for the
    // kernel (its device workspace: seen as invalid_value once, success after); a solution refused three times is
    // dropped for this shape.
    ++state->fallbacks;
    if (++it->second.refusals >= 3) {
        std::fprintf(stderr, "prefill gemm: rocBLAS solution %d refused (status %d) for T=%lld N=%lld K=%lld ldy=%lld "
                             "three times; the default kernel runs for this shape\n", it->second.solution, (int) status,
                     (long long) T, (long long) N, (long long) K, (long long) ldy);
        it->second.tuned = false;
    } else if (std::getenv("STRATA_ROCBLAS_VERBOSE")) {
        std::fprintf(stderr, "prefill gemm: rocBLAS solution %d refused (status %d) for T=%lld N=%lld K=%lld ldy=%lld; "
                             "the default kernel this time\n", it->second.solution, (int) status, (long long) T,
                     (long long) N, (long long) K, (long long) ldy);
    }
    return false;
}
#endif

}  // namespace

bool prompt_f16() {
#if defined(__HIPCC__)
    constexpr int kMaxDev = 64;
    static std::atomic<int8_t> cached[kMaxDev] = {};   // 0 unknown, 1 off, 2 on
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= kMaxDev) { (void) hipGetLastError(); return false; }
    if (const int8_t c = cached[dev].load(std::memory_order_relaxed)) return c == 2;
    // (#835) STRATA_HIP_PROMPT_F16=1 turns it on.  It changes the prompt path's numbers (an FP16 rounding of each 16-bit GEMM's output), so a card
    // gets it unasked only where it was measured: the Radeon 780M (gfx1103, integrated, Linux) and the discrete RX 6000 cards (gfx1030 / 1031 / 1032, #5151;
    // strata::core::prompt_f16_resolve), where =0 is the way back.
    const char* e = std::getenv("STRATA_HIP_PROMPT_F16");
    hipDeviceProp_t p{};
    if (hipGetDeviceProperties(&p, dev) != hipSuccess) { (void) hipGetLastError(); p.gcnArchName[0] = 0; p.integrated = 0; }
#if defined(_WIN32)
    constexpr bool kWin = true;
#else
    constexpr bool kWin = false;
#endif
    const bool on = strata::core::prompt_f16_resolve(e, p.gcnArchName, p.integrated != 0, kWin);
    if (on && e != nullptr) std::fprintf(stderr, "strata prefill: HIP device %d%s%s - STRATA_HIP_PROMPT_F16=1: the prompt's 16-bit GEMMs run FP16 in and out "
                                 "(rocBLAS is tuned only for that on gfx103x; not bit-identical to the default)%s",
                         dev, p.gcnArchName[0] ? " " : "", p.gcnArchName, "\n");
    else if (on && std::strncmp(p.gcnArchName, "gfx103", 6) == 0)
        std::fprintf(stderr, "strata prefill: HIP device %d %s - the prompt's 16-bit GEMMs run FP16 in and out by default here (rocBLAS is tuned for that on "
                             "gfx103x; STRATA_HIP_PROMPT_F16=0 keeps the old route)%s", dev, p.gcnArchName, "\n");
    // (#5151: the start-up tip that asked gfx103x owners to set STRATA_HIP_PROMPT_F16=1 is gone: users never set variables.  On a discrete RX 6000 card the
    // FP16 route is the default now (prompt_f16_resolve): against the exact FP32 products it is no further from exact than the old route and 11-14 % faster.)
    cached[dev].store(on ? 2 : 1, std::memory_order_relaxed);
    return on;
#else
    return false;
#endif
}

// A padded X (ldx > K) no kernel took: its rows copied to a contiguous FP16 buffer of stride K.  Grown, never shrunk
// (one buffer per Gemm; a padded X only arrives from STRATA_PF_PAD's producers).  One 2-D copy: the same values, and
// the ordinary paths below then read them exactly as if the producer had written them unpadded.
const uint16_t* Gemm::unpad_x(const uint16_t* X, int64_t T, int64_t K, int64_t ldx) {
    if (X == nullptr || T <= 0 || K <= 0 || ldx <= K) return nullptr;
    const int64_t need = T * K;
    if (need > unpadded_x_elems_) {
        if (unpadded_x_ != nullptr) { cudaFree(unpadded_x_); unpadded_x_ = nullptr; unpadded_x_elems_ = 0; }
        void* p = nullptr;
        if (cudaMalloc(&p, (size_t) need * sizeof(uint16_t)) != cudaSuccess) { cudaGetLastError(); return nullptr; }
        unpadded_x_ = (uint16_t*) p;
        unpadded_x_elems_ = need;
    }
    if (cudaMemcpy2DAsync(unpadded_x_, (size_t) K * sizeof(uint16_t), X, (size_t) ldx * sizeof(uint16_t),
                          (size_t) K * sizeof(uint16_t), (size_t) T, cudaMemcpyDeviceToDevice,
                          (cudaStream_t) stream_) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    return unpadded_x_;
}

bool gemm_dequant_side() {
#if defined(__HIPCC__)
    return false;
#else
    static const bool on = [] { const char* v = std::getenv("STRATA_DEQUANT_SIDE"); return v && v[0] == '1'; }();
    return on;
#endif
}

Gemm::~Gemm() {
    for (W8Entry& e : w8_) {
        if (e.d) (void) cudaFree(e.d);
        if (e.scale_w) (void) cudaFree(e.scale_w);
    }
    for (W16Entry& e : w16_) if (e.d) (void) cudaFree(e.d);
    for (void* e : side_ready_)
        if (e) (void) cudaEventDestroy((cudaEvent_t) e);
    for (void* e : side_free_)
        if (e) (void) cudaEventDestroy((cudaEvent_t) e);
    if (side_stream_) (void) cudaStreamDestroy((cudaStream_t) side_stream_);
#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)
    delete static_cast<strata::prefill::mmq::Context*>(mmq_ctx_);
    if (mmq_buf_) cudaFree(mmq_buf_);
#endif
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    delete static_cast<HipLtState*>(hipblaslt_state_);
#endif
#if defined(__HIPCC__) && defined(STRATA_ROCBLAS_AVAILABLE)
    delete static_cast<RocBlasState*>(rocblas_state_);
#endif
#if !defined(STRATA_USE_HIP)
    if (lt_pref_) cublasLtMatmulPreferenceDestroy((cublasLtMatmulPreference_t) lt_pref_);
    if (lt_handle_) cublasLtDestroy((cublasLtHandle_t) lt_handle_);
    if (x8_) cudaFree(x8_);
    if (d_scale_x_) cudaFree(d_scale_x_);
    if (d_inv_scale_x_) cudaFree(d_inv_scale_x_);
    if (d_inv_scale_tok_) cudaFree(d_inv_scale_tok_);
    if (y_tmp_) cudaFree(y_tmp_);
    if (d_max_x_) cudaFree(d_max_x_);
    if (d_max_w_) cudaFree(d_max_w_);
#endif
    if (handle_) cublasDestroy((cublasHandle_t) handle_);
    if (tc_w_) cudaFree(tc_w_);
    if (tc_x_) cudaFree(tc_x_);
    if (unpadded_x_) cudaFree(unpadded_x_);
    if (!external_) {
        if (scratch_) cudaFree(scratch_);
        if (workspace_) cudaFree(workspace_);
    }
}

bool Gemm::init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                         std::string& err) {
    cublasHandle_t h = nullptr;
    if (const cublasStatus_t s = cublasCreate(&h); s != CUBLAS_STATUS_SUCCESS) {
        err = "prefill gemm: cublasCreate: cuBLAS status " + std::to_string((int) s);
        return false;
    }
    handle_ = h;
    stream_ = stream;
    external_ = true;
    note(cublasSetStream(h, (cudaStream_t) stream), "cublasSetStream");
    workspace_ = workspace;
    ws_bytes_ = ws_bytes;
    note(cublasSetWorkspace(h, workspace_, ws_bytes), "cublasSetWorkspace");
    note(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
#if !defined(STRATA_USE_HIP)
    {
        cublasLtHandle_t lt = nullptr;
        if (cublasLtCreate(&lt) == CUBLAS_STATUS_SUCCESS) {
            lt_handle_ = lt;
            cublasLtMatmulPreference_t pref = nullptr;
            if (cublasLtMatmulPreferenceCreate(&pref) == CUBLAS_STATUS_SUCCESS) {
                size_t ws_bytes = 32u << 20;
                cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes));
                lt_pref_ = pref;
            }
        }
    }
#endif
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    hipblaslt_state_ = create_hipblaslt_state(workspace_, ws_bytes).release();
#endif
#if defined(__HIPCC__) && defined(STRATA_ROCBLAS_AVAILABLE)
    rocblas_state_ = create_rocblas_state(stream).release();
#endif
    return true;
}

void Gemm::rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes) {
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    workspace_ = workspace;
    ws_bytes_ = ws_bytes;
    cublasSetWorkspace((cublasHandle_t) handle_, workspace_, ws_bytes);
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (hipblaslt_state_) {
        auto* state = static_cast<HipLtState*>(hipblaslt_state_);
        state->workspace = workspace_;
        state->workspace_bytes = ws_bytes;
    }
#endif
}

bool Gemm::set_dequant_side(uint16_t* scratch2, std::string& err) {
    side_scratch_ = scratch2;
    side_cur_ = 0;
    if (scratch2 == nullptr) return true;
    if (side_stream_ == nullptr) {   // once: the stream and the four events outlive every rebind
        cudaStream_t s = nullptr;
        if (const cudaError_t e = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking); e != cudaSuccess) {
            err = std::string("prefill gemm: dequant side stream: ") + cudaGetErrorString(e);
            side_scratch_ = nullptr;
            return false;
        }
        side_stream_ = s;
        for (int i = 0; i < 2; ++i) {
            cudaEvent_t r = nullptr, f = nullptr;
            const bool made = cudaEventCreateWithFlags(&r, cudaEventDisableTiming) == cudaSuccess &&
                              cudaEventCreateWithFlags(&f, cudaEventDisableTiming) == cudaSuccess;
            side_ready_[i] = r;   // kept even on a failure: the destructor frees what was made
            side_free_[i] = f;
            if (!made) {
                err = "prefill gemm: dequant side events";
                side_scratch_ = nullptr;
                return false;
            }
        }
    }
    // both scratches are free once what the compute stream holds now is done (a rebind follows a stream sync, so
    // this is belt and braces; a wait on an event never recorded would pass anyway)
    for (int i = 0; i < 2; ++i)
        (void) cudaEventRecord((cudaEvent_t) side_free_[i], (cudaStream_t) stream_);
    return true;
}

bool Gemm::init(void* stream, int64_t scratch_elems, std::string& err) {
    // #240: every failure names the call and the real status, so "no VRAM" can be told from a broken install
    cublasHandle_t h = nullptr;
    if (const cublasStatus_t s = cublasCreate(&h); s != CUBLAS_STATUS_SUCCESS) {
        err = "prefill gemm: cublasCreate: cuBLAS status " + std::to_string((int) s);
        return false;
    }
    handle_ = h;
    stream_ = stream;
    note(cublasSetStream(h, (cudaStream_t) stream), "cublasSetStream");
    // A fixed workspace so the handle never allocates on the way (and graphs could capture it later).
    const size_t ws = 32u << 20;
    if (const cudaError_t e = cudaMalloc(&workspace_, ws); e != cudaSuccess) {
        err = std::string("prefill gemm: workspace of 32 MiB: ") + cudaGetErrorString(e);
        return false;
    }
    ws_bytes_ = ws;
    note(cublasSetWorkspace(h, workspace_, ws), "cublasSetWorkspace");
    note(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
#if !defined(STRATA_USE_HIP)
    {
        cublasLtHandle_t lt = nullptr;
        if (cublasLtCreate(&lt) == CUBLAS_STATUS_SUCCESS) {
            lt_handle_ = lt;
            cublasLtMatmulPreference_t pref = nullptr;
            if (cublasLtMatmulPreferenceCreate(&pref) == CUBLAS_STATUS_SUCCESS) {
                size_t ws_bytes = 32u << 20;
                cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes));
                lt_pref_ = pref;
            }
        }
    }
#endif
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    hipblaslt_state_ = create_hipblaslt_state(workspace_, ws).release();
#endif
#if defined(__HIPCC__) && defined(STRATA_ROCBLAS_AVAILABLE)
    rocblas_state_ = create_rocblas_state(stream).release();
#endif
    if (scratch_elems > 0) {
        if (const cudaError_t e = cudaMalloc((void**) &scratch_, (size_t) scratch_elems * 2); e != cudaSuccess) {
            err = "prefill gemm: dequant scratch of " + std::to_string(scratch_elems * 2 >> 20) + " MiB: " +
                  cudaGetErrorString(e);
            return false;
        }
    }
    scratch_elems_ = scratch_elems;
    return true;
}

#if !defined(__HIPCC__)
// ---- below sm_80: BF16 GEMMs without BF16 tensor cores (PR #655, #540, #395) ----------------------------------------
// Turing and Volta have no BF16 tensor cores: cublasGemmEx on CUDA_R_16BF inputs falls back to a SIMT fp32 kernel
// (magma_sgemmEx), a fifth of a 4K prompt on an RTX 2080 Ti (#655) and ~10% of a V100's prompt (#540).  BF16 -> FP16
// is exact for every value inside FP16's normal range (the 7-bit mantissa fits in 10 bits); weights and normalized
// activations sit there.  With the FP16 path, the weight is converted once per call and the activations in row
// slices, into the instance's own buffers, and the product runs as the FP16 GEMM (fp32 accumulate) on the tensor
// cores.  Finite values beyond FP16's range are clamped to +-65504 (#540) instead of becoming Inf.  A beta = 1
// product (STRATA_PREFILL_BF16X2's remainder, ~2^-9 of the original, deep in FP16's subnormal band) keeps cuBLAS.
//   Default: on for compute capability 7.0 - 7.4 (Volta: only the experimental STRATA_EXPERIMENTAL_SM60 build runs
//   there), OFF for 7.5 (RTX 20 in the ready-made engine: the sums round differently, so it is opt-in until it has
//   been gated).  STRATA_BF16_TC=1 / =0 turns it on / off on any 7.x card (=2, a test mode: on any card).
// Pascal (6.x) has no tensor cores and cuBLAS has no BF16 GEMM for it (#395, measured NOT_SUPPORTED on a P40): both
// operands are widened to fp32 by an exact shift and the product is cublasSgemm with the same fp32 accumulator.
namespace {
__global__ void bf16_to_f16_kernel(const uint16_t* __restrict__ in, __half* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float f = __uint_as_float((uint32_t) in[i] << 16);
        if (f > 65504.0f && !isinf(f)) f = 65504.0f;
        else if (f < -65504.0f && !isinf(f)) f = -65504.0f;
        out[i] = __float2half_rn(f);
    }
}
void bf16_to_f16(const uint16_t* in, uint16_t* out, int64_t n, cudaStream_t st) {
    if (n <= 0) return;
    bf16_to_f16_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>(in, reinterpret_cast<__half*>(out), n);
}
#if defined(STRATA_EXPERIMENTAL_SM60)   // Pascal runs only the experimental build
__global__ void bf16_to_f32_kernel(const uint16_t* __restrict__ in, float* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __uint_as_float((uint32_t) in[i] << 16);
}
void bf16_to_f32(const uint16_t* in, float* out, int64_t n, cudaStream_t st) {
    if (n <= 0) return;
    bf16_to_f32_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>(in, out, n);
}
#endif
// The current device's compute capability as 10 * major + minor, per device (a layer split can mix cards); 0: not
// known, read as "not an old card" so a failed query keeps the cuBLAS BF16 call.
int current_cc() {
    static std::atomic<int> cc[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return 0; }
    int v = cc[dev].load(std::memory_order_relaxed);
    if (v == 0) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        v = 10 * major + minor;
        cc[dev].store(v, std::memory_order_relaxed);
    }
    return v;
}
// 0: cuBLAS's BF16 GEMM, 1: through FP16 (tensor cores), 2: through FP32 (Pascal)
int bf16_path() {
    static const int forced = [] {
        const char* v = std::getenv("STRATA_BF16_TC");
        return v != nullptr && v[0] != '\0' ? std::atoi(v) : -1;
    }();
    const int cc = current_cc();
    if (forced == 2 && cc > 0) return 1;   // a test mode: the FP16 path on any card (gemm_bf16_parity on sm_80+)
    if (cc <= 0 || cc >= 80) return 0;
    if (cc < 70) return 2;
    if (forced >= 0) return forced != 0 ? 1 : 0;
    return cc < 75 ? 1 : 0;
}
bool grow(uint16_t*& p, int64_t& have, int64_t want) {   // `have`, `want`: 2-byte elements
    if (have >= want) return true;
    if (p) cudaFree(p);
    p = nullptr;
    have = 0;
    if (cudaMalloc((void**) &p, (size_t) want * 2) != cudaSuccess) { cudaGetLastError(); p = nullptr; return false; }
    have = want;
    return true;
}
constexpr int64_t kXSliceElems = 16ll << 20;   // 32 MiB of FP16 activations per slice (64 MiB as fp32)

#if defined(STRATA_EXPERIMENTAL_SM60)
// ---- Pascal, opt-in (STRATA_DENSE_F32=1): the dense projections' FP16-operand GEMMs as cublasSgemm ----------------
// Measured on a Tesla P40 (sm_61) with cuBLAS 12.4.5.8 (bench/results/2026-10-12-p40-sm61-kernels).  For FP16
// operands with FP32 sums cuBLAS has two Maxwell kernels there, maxwell_sgemm_fp16_64x64 and _32x128, and picks the
// slow one for most chunk sizes: the 10240 x 2560 qkv projection runs at 7.6 TFLOPS for T 7030-7040 and at 4.3 from
// T 7041 (49 -> 85 ms), and at 4.3-4.6 for two thirds of the T between 256 and 8192; the 12288-row query projection
// likewise.  A 7K-token prompt reads in 16.3 or 19.5 s by its token count.  cublasSgemm on FP32 copies
// (maxwell_sgemm_128x64) runs at 8.4-9.8 TFLOPS for every T from 512 (6.7+ from 256).  The copies are exact
// (FP16 -> FP32), and for the shapes and sizes taken here the two kernels add every output's products in the same
// order: the result is bitwise the cuBLAS FP16 call's (gemm_f32_parity: 5,201 chunk sizes on a P40 with cuBLAS 12.4, a
// measurement of this library, not a guarantee of cuBLAS's; below the sizes taken here - T under
// ~144 at 6144+ rows, under ~408 at 2560 rows - cuBLAS switches one of them to another kernel family and the bits
// differ, so those keep the FP16 call).  The weight goes through the BF16 path's conversion buffer (126 MiB for the
// largest, 21 MiB more than that path's own), the activations through its slices.
// Opt-in, and on Pascal (compute capability 6.x) only: the equality belongs to the cuBLAS build, so run
// gemm_f32_parity with yours before turning it on; newer cards have tensor cores for the FP16 call.
__global__ void f16_to_f32_kernel(const uint16_t* __restrict__ in, float* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(__ushort_as_half(in[i]));
}
void f16_to_f32(const uint16_t* in, float* out, int64_t n, cudaStream_t st) {
    if (n <= 0) return;
    f16_to_f32_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>(in, out, n);
}
int g_dense_f32 = -1;   // gemm_set_dense_f32 (the parity test: 0 / 1 on any card); -1 = STRATA_DENSE_F32
bool dense_f32_on() {
    static const bool env = [] {
        const char* v = std::getenv("STRATA_DENSE_F32");
        return v != nullptr && std::atoi(v) != 0;
    }();
    if (g_dense_f32 >= 0) return g_dense_f32 != 0;
    if (!env) return false;
    const int cc = current_cc();
    return cc >= 60 && cc < 70;
}
// The smallest T taken for a weight of N rows and K columns (0: never): the measured shapes only - the 2560-column
// qkv (10240 rows), query (12288) and gate (6144) projections and the 2560-row output projections (6144 and 4096
// columns), each from a T where the FP32 route (its conversions included) is the faster one or within ~7% of the FP16
// call (the gate projection below ~1300 tokens, the 4096-column output projection near 4096) and well above the T
// where the two stop agreeing bitwise.  Every other shape (a row slice of a weight too) keeps the FP16 call.
int64_t dense_f32_min_t(int64_t N, int64_t K) {
    if (K == 2560 && (N == 10240 || N == 12288)) return 256;
    if (K == 2560 && N == 6144) return 512;
    if (N == 2560 && (K == 6144 || K == 4096)) return 4096;
    return 0;
}
#endif
}  // namespace

// P2 (STRATA_PF_HCDOWN_CUDA=1): fused HC down + inject GEMM on m16n8k16 BF16 MMA (sm_80+).  One padded xn16 read,
// 64 fp32 accumulators per thread, no device heap: dynamic SMEM A[2][64][72] + B[2][128][72] bf16 = 55,296 B.
namespace {
constexpr int HCD_BM = 64, HCD_BN = 128, HCD_BK = 64, HCD_NT = 128, HCD_SMEM_PAD = 8;
__global__ __launch_bounds__(HCD_NT, 1) void kernel_pf_hcdown_cuda_bf16(
    const uint16_t* __restrict__ X, int ldx, const uint16_t* __restrict__ Wd, const uint16_t* __restrict__ Wi,
    int nd, int ni, float* __restrict__ lo, float* __restrict__ inj, int T, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ uint32_t sm[];
    constexpr int ROW_W = (HCD_BK + HCD_SMEM_PAD) / 2;   // 36 words = 72 bf16: 16-byte aligned rows, no bank conflicts
    constexpr int A_W = 2 * HCD_BM * ROW_W;              // 4,608 words
    uint32_t* const A = sm;                              // A[buf][r][k/2] = (buf*BM + r)*ROW_W + k/2
    uint32_t* const B = sm + A_W;                        // B[buf][n][k/2] = (buf*BN + n)*ROW_W + k/2
    constexpr int A_COPIES = HCD_BM * (HCD_BK / 8);
    constexpr int B_COPIES = HCD_BN * (HCD_BK / 8);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;               // PTX m16n8k16 fragment coordinates
    const int m0 = (int) blockIdx.x * HCD_BM, n0 = (int) blockIdx.y * HCD_BN;
    const int kbt = (K + HCD_BK - 1) / HCD_BK;

    // One 16-byte cp.async per 8 bf16.  Out-of-range rows / columns / k zero-fill (src-size 0), so the epilogue
    // needs no extra guards beyond the output bounds and no scratch allocation is made.
    auto load = [&](int kb, int ib) {
        const int k0 = kb * HCD_BK;
        for (int i = tid; i < A_COPIES; i += HCD_NT) {
            const int r = i / (HCD_BK / 8), c = (i % (HCD_BK / 8)) * 8;
            const int m = m0 + r, k = k0 + c;
            const bool ok = m < T && k < K;
            const uint16_t* src = X + (size_t) (m < T ? m : T - 1) * ldx + (k < K ? k : K - 1);
            uint32_t* dst = A + (ib * HCD_BM + r) * ROW_W + c / 2;
            const unsigned sz = ok ? 16u : 0u;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                         :: "r"((unsigned) __cvta_generic_to_shared(dst)), "l"(src), "r"(sz) : "memory");
        }
        for (int i = tid; i < B_COPIES; i += HCD_NT) {
            const int nr = i / (HCD_BK / 8), n = n0 + nr, c = (i % (HCD_BK / 8)) * 8;
            const int k = k0 + c;
            const bool ok = n < nd + ni && k < K;
            const int row = n < nd ? n : (n - nd < ni ? n - nd : ni - 1);
            const uint16_t* w = n < nd ? Wd : Wi;
            const uint16_t* src = w + (size_t) row * K + (k < K ? k : K - 1);
            uint32_t* dst = B + (ib * HCD_BN + nr) * ROW_W + c / 2;
            const unsigned sz = ok ? 16u : 0u;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                         :: "r"((unsigned) __cvta_generic_to_shared(dst)), "l"(src), "r"(sz) : "memory");
        }
    };

    float acc[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;

    load(0, 0);
    asm volatile("cp.async.commit_group;\n");
#pragma unroll 1
    for (int kb = 0; kb < kbt; ++kb) {
        const int ib = kb & 1;
        if (kb + 1 < kbt) {
            load(kb + 1, ib ^ 1);
            asm volatile("cp.async.commit_group;\n\tcp.async.wait_group 1;\n");
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();

        const int ar0 = (ib * HCD_BM + warp * 16 + g) * ROW_W;
        const int ar1 = ar0 + 8 * ROW_W;
#pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            const int br = (ib * HCD_BN + nt * 8 + g) * ROW_W;
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int kw = ks * 8 + t;   // k = ks*16 + 2*t, in uint32 words
                // PTX m16n8k16.row.col fragments: A rows g / g+8, k 2*t / 2*t+8; B col g, k 2*t / 2*t+8;
                // D rows g / g+8, columns 2*t and 2*t+1 (NOT t and t+4).
                const uint32_t a0 = A[ar0 + kw], a1 = A[ar1 + kw];
                const uint32_t a2 = A[ar0 + kw + 4], a3 = A[ar1 + kw + 4];
                const uint32_t b0 = B[br + kw], b1 = B[br + kw + 4];
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(acc[nt][0]), "+f"(acc[nt][1]), "+f"(acc[nt][2]), "+f"(acc[nt][3])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        const int nbase = n0 + nt * 8 + 2 * t;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int m = m0 + warp * 16 + g + (j >> 1) * 8;
            const int n = nbase + (j & 1);
            if (m < T && n < nd + ni) {
                if (n < nd) lo[(size_t) m * nd + n] = acc[nt][j];
                else inj[(size_t) m * ni + (n - nd)] = acc[nt][j];
            }
        }
    }
#else
    (void) X; (void) ldx; (void) Wd; (void) Wi; (void) nd; (void) ni;
    (void) lo; (void) inj; (void) T; (void) K;   // only launched on cc >= 8.0
#endif
}
}  // namespace
}  // namespace strata::prefill

#if !defined(__APPLE__)
bool strata_pf_hcdown_cuda_bf16(const uint16_t* X, int64_t ldx, const uint16_t* Wd, const uint16_t* Wi, int64_t nd,
                                int64_t ni, float* lo, float* inj, int64_t T, int64_t K, void* stream) {
    using namespace strata::prefill;   // HCD_* constants and kernel live in its unnamed namespace
    constexpr int64_t kD = 10240, kLR = 320, kHC = 4, kPad = 64;   // prefill.cpp's D / LR / HC / XN_PAD
    if (!X || !Wd || !Wi || !lo || !inj) return false;
    if (nd != kLR || ni != kHC || K != kD || ldx != K + kPad || ldx == K) return false;
    if (T < 64 || T > (1LL << 30) || K % HCD_BK != 0) return false;
    int dev = 0, cc_major = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    if (cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) return false;
    if (cc_major < 8) return false;   // m16n8k16 BF16 mma.sync begins on sm_80 (Ampere / Ada / Hopper)
    constexpr size_t smem_bytes = 2 * HCD_BM * (HCD_BK + HCD_SMEM_PAD) * 2 +
                                  2 * HCD_BN * (HCD_BK + HCD_SMEM_PAD) * 2;   // 55,296 B
    if (cudaFuncSetAttribute(kernel_pf_hcdown_cuda_bf16, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int) smem_bytes) != cudaSuccess) return false;
    const dim3 grid((unsigned) ((T + HCD_BM - 1) / HCD_BM), (unsigned) ((nd + ni + HCD_BN - 1) / HCD_BN));
    kernel_pf_hcdown_cuda_bf16<<<grid, HCD_NT, smem_bytes, static_cast<cudaStream_t>(stream)>>>(
        X, (int) ldx, Wd, Wi, (int) nd, (int) ni, lo, inj, (int) T, (int) K);
    return cudaGetLastError() == cudaSuccess;
}
#endif
namespace strata::prefill {
#endif
#if !defined(__HIPCC__) && defined(STRATA_EXPERIMENTAL_SM60)
void gemm_set_dense_f32(int mode) { g_dense_f32 = mode; }
#else
void gemm_set_dense_f32(int) {}
#endif
bool Gemm::bf16_hcd_exact(const uint16_t* X, int64_t ldx, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K) {
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    static std::atomic<bool> told{false};
    if (!hipblaslt_state_ || N != 320 || K != 10240 || T < 1) {
        if (N == 320 && K == 10240 && !told.exchange(true))
            std::fprintf(stderr, "strata: STRATA_HCD_EXACT: no hipBLASLt tuning table is loaded, so the exact kernel has nothing to match: "
                                 "the hyper-connection down projection stays on hipBLAS\n");
        return false;
    }
    // The kernel copies hipBLASLt solution 1176 / 1177's summation order, which is only true of the library build the
    // table was calibrated with: take it only when the table makes hipBLASLt pick one of those two for this T.
    const int id = hipblaslt_solution_for(hipblaslt_state_, strata::prefill::hipblaslt::InputType::bf16, T, N, K, N, 0.0f);
    if (id != 1176 && id != 1177) {
        if (T >= 4096 && !told.exchange(true))   // (a short chunk may legitimately get another solution: not worth a line)
            std::fprintf(stderr, "strata: STRATA_HCD_EXACT: the tuning table gives the hyper-connection down projection (N 320, K 10240) "
                                 "solution %d at T %lld, not 1176 / 1177: hipBLASLt runs it (a changed table or library; "
                                 "tools/hip/gfx1151-hipblaslt-100401.txt is the one measured)\n",
                         id, (long long) T);
        return false;
    }
    return strata_pf_hcdown_exact_bf16(X, ldx, W, Y, T, N, K, stream_);
#else
    (void) X; (void) ldx; (void) W; (void) Y; (void) T; (void) N; (void) K;
    return false;
#endif
}

#if defined(__HIPCC__)
// ---- RDNA2 (gfx103x): the prompt path's GEMMs as FP32 SGEMMs ------------------------------------------------------
// The GEMMs here take 16-bit inputs (FP16 or BF16) and write FP32 (rocBLAS's "HS" / "BS" types).  The rocBLAS the engine
// ships has tuned gfx1030 kernels for FP16 -> FP16 ("HH"), int8 and FP32 -> FP32 ("SS") only: every 16-bit-input,
// FP32-output GEMM runs a "fallback" kernel at ~5 TFLOPS.  Measured on an RX 6800 (ROCm 10.2.0a20260930) with the
// engine's exact calls (opA = T, opB = N, fp32 compute): at the Coder's prompt shapes the FP16 GEMMs of one fresh
// 12K-token prompt take 14.1 s, the same GEMMs as SGEMM on widened copies 4.7 s including the widening (e.g.
// N 10240 x T 8192 x K 2560: 86.7 -> 27.7 ms); the BF16 hyper-connection GEMMs 10.9 -> 7.5 ms (N 320, K 10240).
// Widening FP16 / BF16 to FP32 is exact and SGEMM accumulates in FP32 like the native call, so the products are the
// same up to summation order.  The weight is widened once per call, the activations in slices of up to 128 MiB of
// FP32 (an 8,192-token chunk at K = 2560 is one slice).  Small N (< 64: routers, gates) keeps the native call:
// there the widening costs more than the GEMM gains.  If the FP32 buffers cannot be allocated, the native call runs.
//   OPT-IN here: STRATA_RDNA2_SGEMM=1 turns it on for gfx103x devices (other cards keep the native call, also in a
//   mixed layer split); =all turns it on for every AMD card (A/B arms, tests); unset or 0 is the native path.  The
//   author ran it as the gfx103x default (RX 6800 343 -> 534 tok/s); it is not bit-identical to the native call.
namespace {
__global__ void rdna2_bf16_to_f32_kernel(const uint16_t* __restrict__ in, float* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __uint_as_float((uint32_t) in[i] << 16);
}
__global__ void rdna2_f16_to_f32_kernel(const __half* __restrict__ in, float* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}
void rdna2_widen(const uint16_t* in, float* out, int64_t n, bool bf16, cudaStream_t st) {
    if (n <= 0) return;
    const unsigned blocks = (unsigned) ((n + 255) / 256);
    if (bf16) rdna2_bf16_to_f32_kernel<<<blocks, 256, 0, st>>>(in, out, n);
    else rdna2_f16_to_f32_kernel<<<blocks, 256, 0, st>>>(reinterpret_cast<const __half*>(in), out, n);
}
// whether this GEMM runs as an SGEMM (see above)
bool rdna2_sgemm_on(int64_t N) {
    static const int mode = [] {   // 0 off (unset), 1 gfx103x devices, 2 every AMD device
        const char* v = std::getenv("STRATA_RDNA2_SGEMM");
        if (v == nullptr || v[0] == '\0') return 0;
        if (std::strcmp(v, "all") == 0) return 2;
        return std::atoi(v) > 0 ? 1 : 0;
    }();
    if (mode == 0 || N < 64) return false;
    if (mode == 2) return true;
    static std::atomic<int> arch[64] = {};   // per device: 0 not known, 1 gfx103x, 2 another architecture
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) { (void) hipGetLastError(); return false; }
    int v = arch[dev].load(std::memory_order_relaxed);
    if (v == 0) {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, dev) != hipSuccess) { (void) hipGetLastError(); return false; }
        v = std::strncmp(prop.gcnArchName, "gfx103", 6) == 0 ? 1 : 2;
        arch[dev].store(v, std::memory_order_relaxed);
    }
    return v == 1;
}
// `have`, `want`: FP32 elements (the buffers are the instance's 2-byte tc_ buffers, sized in 2-byte elements)
bool rdna2_grow(uint16_t*& p, int64_t& have_u16, int64_t want_f32) {
    const int64_t want = 2 * want_f32;
    if (have_u16 >= want) return true;
    if (p) cudaFree(p);
    p = nullptr;
    have_u16 = 0;
    if (cudaMalloc((void**) &p, (size_t) want * 2) != cudaSuccess) { (void) cudaGetLastError(); p = nullptr; return false; }
    have_u16 = want;
    return true;
}
constexpr int64_t kRdna2SliceF32 = 32ll << 20;   // 128 MiB of FP32 activations per slice
}  // namespace

// Y^T[N, T] (+)= W[N, K] . X^T[K, T] as SGEMMs on FP32 copies; false: not this card / shape, or no buffers (the caller
// runs its native GEMM, nothing was written)
bool Gemm::rdna2_sgemm(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                       float beta, bool bf16) {
    if (K <= 0 || !rdna2_sgemm_on(N)) return false;
    const int64_t rows = std::max<int64_t>(1, std::min<int64_t>(T, kRdna2SliceF32 / K));
    // growing frees a buffer an earlier GEMM on this stream may still read (as the sm_60 FP32 route does)
    if (tc_w_elems_ < 2 * N * K || tc_x_elems_ < 2 * rows * K) (void) hipStreamSynchronize((hipStream_t) stream_);
    if (!rdna2_grow(tc_w_, tc_w_elems_, N * K) || !rdna2_grow(tc_x_, tc_x_elems_, rows * K)) return false;
    float* const wf = reinterpret_cast<float*>(tc_w_);
    float* const xf = reinterpret_cast<float*>(tc_x_);
    const float alpha = 1.0f;
    rdna2_widen(W, wf, N * K, bf16, (cudaStream_t) stream_);
    for (int64_t t0 = 0; t0 < T; t0 += rows) {
        const int64_t n = std::min<int64_t>(rows, T - t0);
        rdna2_widen(X + t0 * K, xf, n * K, bf16, (cudaStream_t) stream_);
        // every slice is a disjoint block of Y, so each gets the caller's beta
        ck(hipblasSgemm((hipblasHandle_t) handle_, HIPBLAS_OP_T, HIPBLAS_OP_N, (int) N, (int) n, (int) K, &alpha, wf,
                        (int) K, xf, (int) K, &beta, Y + t0 * ldy, (int) ldy),
           "hipblasSgemm (RDNA2)");
    }
    return true;
}
#endif

bool Gemm::bf16_converts_x() const {
#if defined(__HIPCC__) || defined(STRATA_EXPERIMENTAL_SM60)
    return false;
#else
    return bf16_path() == 1;
#endif
}
__global__ void f16_to_bf16_kernel(const __half* __restrict__ in, uint16_t* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (uint16_t) (__float_as_uint(__half2float(in[i])) >> 16);   // exact: the value was a BF16
}
void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta, int64_t ldx, bool x_f16) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    if (ldx <= K) ldx = 0;
#ifdef STRATA_USE_HIP
    // #313 (opt-in STRATA_WMMA_GEMM=1; STRATA_WMMA_BF16=0 excludes bf16): RDNA3 / RDNA3.5 WMMA dense GEMM, before
    // hipBLASLt; it returns false for shapes it does not take (and off gfx11) and this falls through
    static const bool wmma_on = [] { const char* v = std::getenv("STRATA_WMMA_GEMM"); return v && v[0] != 0 && v[0] != '0'; }();
    static const bool bf16_on = [] { const char* v = std::getenv("STRATA_WMMA_BF16"); return !v || (v[0] != 0 && v[0] != '0'); }();
    if (wmma_on && bf16_on && ldx == 0 && T >= 16 && (beta == 0.0f || beta == 1.0f) &&
        strata_wmma_gemm_bf16(X, W, Y, T, N, K, ldy, beta, stream_)) {
        return;
    }
#endif
#if defined(__HIPCC__)
    if (f16_io_) {   // X is the FP16 image (set_act_f16); W goes through the dequantization scratch as FP16
        // the only accumulating BF16 product is the BF16X2 low part, which bf16x2_mode() turns off with this path
        if (beta != 0.0f) { std::fprintf(stderr, "prefill gemm: an accumulating BF16 GEMM on the FP16 prompt path\n"); std::exit(1); }
        const int64_t rows = std::min<int64_t>(N, scratch_elems_ / K);
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = std::min(rows, N - r0);
            bf16_to_f16_rows<<<1024, 256, 0, (cudaStream_t) stream_>>>(W + r0 * K, (__half*) scratch_, n * K);
            f16_inplace(X, scratch_, Y + r0, T, n, K, ldy);
        }
        return;
    }
#endif
    const float alpha = 1.0f;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (try_hipblaslt(hipblaslt_state_, strata::prefill::hipblaslt::InputType::bf16, X, W, Y, T, N, K, ldy,
                      beta, stream_, ldx)) {
        STRATA_ABSORB_HIPBLAS_STICKY("hipBLASLt bf16");
        return;
    }
#endif
#if defined(__HIPCC__)
    // a padded X (STRATA_PF_PAD, ldx > K) keeps the native call: the widening copies contiguous rows
    if (ldx == 0 && rdna2_sgemm(X, W, Y, T, N, K, ldy, beta, true)) return;
#endif
#if !defined(__HIPCC__)
#if defined(STRATA_EXPERIMENTAL_SM60)
    constexpr bool kPascalPath = true;
#else
    constexpr bool kPascalPath = false;
#endif
    if (const int path = K > 0 ? bf16_path() : 0; (path == 1 && N > 1 && beta == 0.0f) || (path == 2 && kPascalPath)) {
        // FP16 (path 1) or FP32 (path 2, Pascal) copies of the weight and of X, converted in tiles.  Issue #1757: when
        // the buffers did not fit in the VRAM that was left (a stage of a layer split keeps ~100 MiB), this fell through
        // to the raw BF16 cuBLAS call, which a Pascal card answers with CUBLAS_STATUS_INTERNAL_ERROR (14).  Now the
        // weight is converted in tiles of N rows (the whole weight when it fits, as before) and X in slices; a card with
        // no room even for a tile says so.  A single output row stays cuBLAS's GEMV for path 1 (faster than the
        // conversions).  Every tile is a disjoint block of Y, so each gets the caller's beta.
        const int mult = path == 2 ? 2 : 1;   // 2-byte elements per converted value
        int64_t w_rows = N;
        int64_t x_rows = std::max<int64_t>(1, std::min<int64_t>(T, kXSliceElems / K));
        bool fit = false;
        for (;;) {
            if (grow(tc_w_, tc_w_elems_, mult * w_rows * K) && grow(tc_x_, tc_x_elems_, mult * x_rows * K)) { fit = true; break; }
            if (w_rows <= 64 && x_rows <= 64) break;
            w_rows = std::max<int64_t>(std::min<int64_t>(w_rows, 64), w_rows / 2);
            x_rows = std::max<int64_t>(std::min<int64_t>(x_rows, 64), x_rows / 2);
        }
        if (!fit && path == 2) {   // Pascal has no BF16 cuBLAS call to fall back to; Volta keeps its raw one below
            std::fprintf(stderr, "prefill gemm: no GPU memory left for the %s conversion buffers of a %lld x %lld BF16 weight "
                                 "(Pascal cards run the BF16 products through them); free VRAM on this card: raise "
                                 "--vram-reserve-mib, lower --max-context or --kv-resident, or give it fewer layers\n",
                         "FP32", (long long) N, (long long) K);
            std::exit(1);
        }
        if (fit) {
            for (int64_t n0 = 0; n0 < N; n0 += w_rows) {
                const int64_t nn = std::min<int64_t>(w_rows, N - n0);
                if (path == 2) {
#if defined(STRATA_EXPERIMENTAL_SM60)
                    float* const wf = reinterpret_cast<float*>(tc_w_);
                    float* const xf = reinterpret_cast<float*>(tc_x_);
                    bf16_to_f32(W + n0 * K, wf, nn * K, (cudaStream_t) stream_);
                    for (int64_t t0 = 0; t0 < T; t0 += x_rows) {
                        const int64_t n = std::min<int64_t>(x_rows, T - t0);
                        bf16_to_f32(X + t0 * K, xf, n * K, (cudaStream_t) stream_);
                        ck(cublasSgemm((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) nn, (int) n, (int) K, &alpha,
                                       wf, (int) K, xf, (int) K, &beta, Y + t0 * ldy + n0, (int) ldy),
                           "cublasSgemm (bf16 on Pascal)");
                    }
#endif
                } else {
                    bf16_to_f16(W + n0 * K, tc_w_, nn * K, (cudaStream_t) stream_);
                    for (int64_t t0 = 0; t0 < T; t0 += x_rows) {
                        const int64_t n = std::min<int64_t>(x_rows, T - t0);
                        if (x_f16) {   // already the converted values (the same slices: the same GEMMs)
                            f16(X + t0 * K, tc_w_, Y + t0 * ldy + n0, n, nn, K, ldy, beta);
                        } else {
                            bf16_to_f16(X + t0 * K, tc_x_, n * K, (cudaStream_t) stream_);
                            f16(tc_x_, tc_w_, Y + t0 * ldy + n0, n, nn, K, ldy, beta);
                        }
                    }
                }
            }
            return;
        }
    }
#endif
#if !defined(__HIPCC__)
    if (x_f16) {   // not the tensor-core path after all (one output row, or no room): X back to BF16, a slice at a time
        const int64_t x_rows = std::max<int64_t>(1, std::min<int64_t>(T, kXSliceElems / K));
        if (!grow(tc_x_, tc_x_elems_, x_rows * K)) {
            std::fprintf(stderr, "prefill gemm: no GPU memory for converting an FP16 image back to BF16\n");
            std::exit(1);
        }
        for (int64_t t0 = 0; t0 < T; t0 += x_rows) {
            const int64_t n = std::min<int64_t>(x_rows, T - t0);
            f16_to_bf16_kernel<<<(unsigned) ((n * K + 255) / 256), 256, 0, (cudaStream_t) stream_>>>(
                reinterpret_cast<const __half*>(X + t0 * K), tc_x_, n * K);
            bf16(tc_x_, W, Y + t0 * ldy, n, N, K, ldy, beta, 0, false);
        }
        return;
    }
#endif
    // Column-major view: Y^T[N, T] = W[N, K] (stored K x N col-major, transposed) . X^T[K, T].
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16BF, (int) K, X, CUDA_R_16BF, (int) (ldx ? ldx : K), &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx");
    STRATA_ABSORB_HIPBLAS_STICKY("cublasGemmEx");
}

#if !defined(STRATA_USE_HIP)
namespace {

__device__ __forceinline__ void atomicMaxFloat(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(val));
    }
}

__device__ __forceinline__ float warpReduceMax(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

__global__ void w16_amax_kernel(const half* __restrict__ src, int64_t n, float* __restrict__ d_max) {

    float local_max = 0.0f;
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;

    int64_t total_vec = n / 8;
    const uint4* src_v = reinterpret_cast<const uint4*>(src);
    int64_t idx = (int64_t)blockDim.x * blockIdx.x + tid;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total_vec; i += stride) {
        uint4 v = src_v[i];
        const half2* h2 = reinterpret_cast<const half2*>(&v);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 f2 = __half22float2(h2[j]);
            local_max = fmaxf(local_max, fmaxf(fabsf(f2.x), fabsf(f2.y)));
        }
    }
    for (int64_t i = total_vec * 8 + idx; i < n; i += stride) {
        local_max = fmaxf(local_max, fabsf(__half2float(src[i])));
    }

    local_max = warpReduceMax(local_max);
    __shared__ float s_warp_max[8];
    if (lane == 0) s_warp_max[wid] = local_max;
    __syncthreads();

    if (wid == 0) {
        float block_max = (lane < (blockDim.x / 32)) ? s_warp_max[lane] : 0.0f;
        block_max = warpReduceMax(block_max);
        if (lane == 0 && block_max > 0.0f) {
            atomicMaxFloat(d_max, block_max);
        }
    }
}

__global__ void w16_scale_and_quantize_vectorized_kernel(
    const half* __restrict__ src, uint8_t* __restrict__ dst,
    int64_t n, const float* __restrict__ d_max, float* __restrict__ d_inv_scale_w) {

    __shared__ float s_scale;
    if (threadIdx.x == 0) {
        float m = *d_max;
        if (m < 1e-8f) m = 1e-8f;
        float s = 448.0f / m;
        s_scale = s;
        if (blockIdx.x == 0 && d_inv_scale_w) {
            *d_inv_scale_w = 1.0f / s;
        }
    }
    __syncthreads();

    float s = s_scale;
    int tid = threadIdx.x;
    int64_t total_vec = n / 8;
    const uint4* src_v = reinterpret_cast<const uint4*>(src);
    uint2* dst_v = reinterpret_cast<uint2*>(dst);
    int64_t idx = (int64_t)blockDim.x * blockIdx.x + tid;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total_vec; i += stride) {
        uint4 in_val = src_v[i];
        const half2* h2 = reinterpret_cast<const half2*>(&in_val);
        uint8_t out_b[8];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 f2 = __half22float2(h2[j]);
            __nv_fp8_e4m3 q0(f2.x * s);
            __nv_fp8_e4m3 q1(f2.y * s);
            out_b[j * 2] = *reinterpret_cast<uint8_t*>(&q0);
            out_b[j * 2 + 1] = *reinterpret_cast<uint8_t*>(&q1);
        }
        dst_v[i] = *reinterpret_cast<uint2*>(out_b);
    }

    for (int64_t i = total_vec * 8 + idx; i < n; i += stride) {
        __nv_fp8_e4m3 q(__half2float(src[i]) * s);
        dst[i] = *reinterpret_cast<uint8_t*>(&q);
    }
}

__global__ void x16_amax_kernel(const half* __restrict__ src, int64_t T, int64_t K, int64_t ldx,
                                float* __restrict__ d_max) {

    float local_max = 0.0f;
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;

    if (ldx == K) {
        int64_t total = T * K;
        int64_t total_vec = total / 8;
        const uint4* src_v = reinterpret_cast<const uint4*>(src);
        int64_t idx = (int64_t)blockDim.x * blockIdx.x + tid;
        int64_t stride = (int64_t)blockDim.x * gridDim.x;
        for (int64_t i = idx; i < total_vec; i += stride) {
            uint4 v = src_v[i];
            const half2* h2 = reinterpret_cast<const half2*>(&v);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 f2 = __half22float2(h2[j]);
                local_max = fmaxf(local_max, fmaxf(fabsf(f2.x), fabsf(f2.y)));
            }
        }
        for (int64_t i = total_vec * 8 + idx; i < total; i += stride) {
            local_max = fmaxf(local_max, fabsf(__half2float(src[i])));
        }
    } else {
        int64_t K_vec = K / 8;
        int64_t total_blocks = T * K_vec;
        int64_t idx = (int64_t)blockDim.x * blockIdx.x + tid;
        int64_t stride = (int64_t)blockDim.x * gridDim.x;
        for (int64_t i = idx; i < total_blocks; i += stride) {
            int64_t t = i / K_vec;
            int64_t k_block = (i % K_vec) * 8;
            const uint4* src_v = reinterpret_cast<const uint4*>(src + t * ldx + k_block);
            uint4 v = *src_v;
            const half2* h2 = reinterpret_cast<const half2*>(&v);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 f2 = __half22float2(h2[j]);
                local_max = fmaxf(local_max, fmaxf(fabsf(f2.x), fabsf(f2.y)));
            }
        }
        int64_t rem = K - K_vec * 8;
        if (rem > 0) {
            for (int64_t t = blockIdx.x; t < T; t += gridDim.x) {
                for (int64_t k = K_vec * 8 + tid; k < K; k += blockDim.x) {
                    local_max = fmaxf(local_max, fabsf(__half2float(src[t * ldx + k])));
                }
            }
        }
    }

    local_max = warpReduceMax(local_max);
    __shared__ float s_warp_max[8];
    if (lane == 0) s_warp_max[wid] = local_max;
    __syncthreads();

    if (wid == 0) {
        float block_max = (lane < (blockDim.x / 32)) ? s_warp_max[lane] : 0.0f;
        block_max = warpReduceMax(block_max);
        if (lane == 0 && block_max > 0.0f) {
            atomicMaxFloat(d_max, block_max);
        }
    }
}

__global__ void x16_scale_and_quantize_vectorized_kernel(
    const half* __restrict__ src, uint8_t* __restrict__ dst,
    int64_t T, int64_t K, int64_t ldx,
    const float* __restrict__ d_max,
    float* __restrict__ d_scale_x,
    float* __restrict__ d_inv_scale_x) {

    __shared__ float s_scale;
    if (threadIdx.x == 0) {
        float m = *d_max;
        if (m < 1e-8f) m = 1e-8f;
        float s = 448.0f / m;
        s_scale = s;
        if (blockIdx.x == 0) {
            if (d_scale_x) *d_scale_x = s;
            if (d_inv_scale_x) *d_inv_scale_x = 1.0f / s;
        }
    }
    __syncthreads();

    float s = s_scale;
    int tid = threadIdx.x;

    if (ldx == K) {
        int64_t total = T * K;
        int64_t total_vec = total / 8;
        const uint4* src_v = reinterpret_cast<const uint4*>(src);
        uint2* dst_v = reinterpret_cast<uint2*>(dst);
        int64_t idx = (int64_t)blockDim.x * blockIdx.x + tid;
        int64_t stride = (int64_t)blockDim.x * gridDim.x;

        for (int64_t i = idx; i < total_vec; i += stride) {
            uint4 in_val = src_v[i];
            const half2* h2 = reinterpret_cast<const half2*>(&in_val);
            uint8_t out_b[8];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 f2 = __half22float2(h2[j]);
                __nv_fp8_e4m3 q0(f2.x * s);
                __nv_fp8_e4m3 q1(f2.y * s);
                out_b[j * 2] = *reinterpret_cast<uint8_t*>(&q0);
                out_b[j * 2 + 1] = *reinterpret_cast<uint8_t*>(&q1);
            }
            dst_v[i] = *reinterpret_cast<uint2*>(out_b);
        }

        for (int64_t i = total_vec * 8 + idx; i < total; i += stride) {
            __nv_fp8_e4m3 q(__half2float(src[i]) * s);
            dst[i] = *reinterpret_cast<uint8_t*>(&q);
        }
    } else {
        int64_t K_vec = K / 8;
        int64_t total_blocks = T * K_vec;
        int64_t idx = (int64_t)blockDim.x * blockIdx.x + tid;
        int64_t stride = (int64_t)blockDim.x * gridDim.x;

        for (int64_t i = idx; i < total_blocks; i += stride) {
            int64_t t = i / K_vec;
            int64_t k_block = (i % K_vec) * 8;
            const uint4* in_ptr = reinterpret_cast<const uint4*>(src + t * ldx + k_block);
            uint2* out_ptr = reinterpret_cast<uint2*>(dst + t * K + k_block);

            uint4 in_val = *in_ptr;
            const half2* h2 = reinterpret_cast<const half2*>(&in_val);
            uint8_t out_b[8];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 f2 = __half22float2(h2[j]);
                __nv_fp8_e4m3 q0(f2.x * s);
                __nv_fp8_e4m3 q1(f2.y * s);
                out_b[j * 2] = *reinterpret_cast<uint8_t*>(&q0);
                out_b[j * 2 + 1] = *reinterpret_cast<uint8_t*>(&q1);
            }
            *out_ptr = *reinterpret_cast<uint2*>(out_b);
        }

        int64_t rem = K - K_vec * 8;
        if (rem > 0) {
            for (int64_t t = blockIdx.x; t < T; t += gridDim.x) {
                for (int64_t k = K_vec * 8 + tid; k < K; k += blockDim.x) {
                    __nv_fp8_e4m3 q(__half2float(src[t * ldx + k]) * s);
                    dst[t * K + k] = *reinterpret_cast<uint8_t*>(&q);
                }
            }
        }
    }
}

__global__ void __launch_bounds__(256) x16_per_token_quantize_e4m3_kernel(
    const half* __restrict__ src, uint8_t* __restrict__ dst,
    float* __restrict__ d_inv_scale_tok,
    int64_t T, int64_t K, int64_t ldx) {

    const int64_t t = (int64_t) blockIdx.x;
    if (t >= T) return;

    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lane = tid % 32;

    const half* row_in = src + t * ldx;
    uint8_t* row_out = dst + t * K;
    const bool can_vec = ((reinterpret_cast<uintptr_t>(row_in) & 15u) == 0) &&
                         ((reinterpret_cast<uintptr_t>(row_out) & 7u) == 0) &&
                         ((K & 7) == 0);

    uint4 reg_v0{}, reg_v1{};
    const bool reg_cached = can_vec && (K <= 4096);
    float local_max = 0.0f;
    if (reg_cached) {
        const int64_t K_vec = K / 8;
        const uint4* in_v = reinterpret_cast<const uint4*>(row_in);
        if (tid < K_vec) {
            reg_v0 = in_v[tid];
            const half2* h2 = reinterpret_cast<const half2*>(&reg_v0);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 f2 = __half22float2(h2[j]);
                local_max = fmaxf(local_max, fmaxf(fabsf(f2.x), fabsf(f2.y)));
            }
        }
        if (tid + 256 < K_vec) {
            reg_v1 = in_v[tid + 256];
            const half2* h2 = reinterpret_cast<const half2*>(&reg_v1);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 f2 = __half22float2(h2[j]);
                local_max = fmaxf(local_max, fmaxf(fabsf(f2.x), fabsf(f2.y)));
            }
        }
    } else if (can_vec) {
        const int64_t K_vec = K / 8;
        const uint4* in_v = reinterpret_cast<const uint4*>(row_in);
        for (int64_t i = tid; i < K_vec; i += blockDim.x) {
            uint4 v = in_v[i];
            const half2* h2 = reinterpret_cast<const half2*>(&v);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 f2 = __half22float2(h2[j]);
                local_max = fmaxf(local_max, fmaxf(fabsf(f2.x), fabsf(f2.y)));
            }
        }
    } else {
        for (int64_t k = tid; k < K; k += blockDim.x) {
            local_max = fmaxf(local_max, fabsf(__half2float(row_in[k])));
        }
    }

    local_max = warpReduceMax(local_max);
    __shared__ float s_warp_max[8];
    if (lane == 0) s_warp_max[wid] = local_max;
    __syncthreads();

    __shared__ float s_scale;
    if (wid == 0) {
        float row_amax = (lane < (blockDim.x / 32)) ? s_warp_max[lane] : 0.0f;
        row_amax = warpReduceMax(row_amax);
        if (lane == 0) {
            const float inv_scale = (row_amax > 0.0f) ? fmaxf(row_amax * (1.0f / 448.0f), 1.0f / (448.0f * 512.0f)) : 0.0f;
            const float scale = (row_amax > 0.0f) ? (1.0f / inv_scale) : 0.0f;
            d_inv_scale_tok[t] = inv_scale;
            s_scale = scale;
        }
    }
    __syncthreads();

    const float scale = s_scale;
    if (reg_cached) {
        const int64_t K_vec = K / 8;
        uint2* out_v = reinterpret_cast<uint2*>(row_out);
        if (tid < K_vec) {
            const half2* h2 = reinterpret_cast<const half2*>(&reg_v0);
            uint8_t out_b[8];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 f2 = __half22float2(h2[j]);
                __nv_fp8_e4m3 q0(f2.x * scale);
                __nv_fp8_e4m3 q1(f2.y * scale);
                out_b[j * 2] = *reinterpret_cast<uint8_t*>(&q0);
                out_b[j * 2 + 1] = *reinterpret_cast<uint8_t*>(&q1);
            }
            out_v[tid] = *reinterpret_cast<const uint2*>(out_b);
        }
        if (tid + 256 < K_vec) {
            const half2* h2 = reinterpret_cast<const half2*>(&reg_v1);
            uint8_t out_b[8];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 f2 = __half22float2(h2[j]);
                __nv_fp8_e4m3 q0(f2.x * scale);
                __nv_fp8_e4m3 q1(f2.y * scale);
                out_b[j * 2] = *reinterpret_cast<uint8_t*>(&q0);
                out_b[j * 2 + 1] = *reinterpret_cast<uint8_t*>(&q1);
            }
            out_v[tid + 256] = *reinterpret_cast<const uint2*>(out_b);
        }
    } else if (can_vec) {
        const int64_t K_vec = K / 8;
        const uint4* in_v = reinterpret_cast<const uint4*>(row_in);
        uint2* out_v = reinterpret_cast<uint2*>(row_out);
        for (int64_t i = tid; i < K_vec; i += blockDim.x) {
            uint4 in_val = in_v[i];
            const half2* h2 = reinterpret_cast<const half2*>(&in_val);
            uint8_t out_b[8];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 f2 = __half22float2(h2[j]);
                __nv_fp8_e4m3 q0(f2.x * scale);
                __nv_fp8_e4m3 q1(f2.y * scale);
                out_b[j * 2] = *reinterpret_cast<uint8_t*>(&q0);
                out_b[j * 2 + 1] = *reinterpret_cast<uint8_t*>(&q1);
            }
            out_v[i] = *reinterpret_cast<uint2*>(out_b);
        }
    } else {
        for (int64_t k = tid; k < K; k += blockDim.x) {
            __nv_fp8_e4m3 q(__half2float(row_in[k]) * scale);
            row_out[k] = *reinterpret_cast<uint8_t*>(&q);
        }
    }
}

// 1-pass per-row E4M3 weight quantize: 1 block (256 threads) per row n of W[N, K].
// Writes E4M3 row and per-row inverse scale d_inv_scale_w[n] = fmaxf(amax_n, 1e-8f) / 448.0f without global atomics.
__global__ void __launch_bounds__(256) w16_per_row_quantize_e4m3_kernel(
    const half* __restrict__ W, uint8_t* __restrict__ W8,
    float* __restrict__ d_inv_scale_w, int64_t N, int64_t K) {

    const int64_t n = (int64_t) blockIdx.x;
    if (n >= N) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;

    const half* row_in = W + n * K;
    uint8_t* row_out = W8 + n * K;

    const bool can_vec = ((reinterpret_cast<uintptr_t>(row_in) & 15u) == 0) &&
                         ((reinterpret_cast<uintptr_t>(row_out) & 7u) == 0) &&
                         ((K & 7) == 0);
    const bool reg_cached = can_vec && (K <= 4096);

    float local_max = 0.0f;
    uint4 reg_v0{}, reg_v1{};

    if (reg_cached) {
        const int64_t K_vec = K / 8;
        const uint4* in_v = reinterpret_cast<const uint4*>(row_in);
        if (tid < K_vec) {
            reg_v0 = in_v[tid];
            const half2* h2 = reinterpret_cast<const half2*>(&reg_v0);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 f2 = __half22float2(h2[j]);
                local_max = fmaxf(local_max, fmaxf(fabsf(f2.x), fabsf(f2.y)));
            }
        }
        if (tid + 256 < K_vec) {
            reg_v1 = in_v[tid + 256];
            const half2* h2 = reinterpret_cast<const half2*>(&reg_v1);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 f2 = __half22float2(h2[j]);
                local_max = fmaxf(local_max, fmaxf(fabsf(f2.x), fabsf(f2.y)));
            }
        }
    } else if (can_vec) {
        const int64_t K_vec = K / 8;
        const uint4* in_v = reinterpret_cast<const uint4*>(row_in);
        for (int64_t i = tid; i < K_vec; i += blockDim.x) {
            const uint4 v = in_v[i];
            const half2* h2 = reinterpret_cast<const half2*>(&v);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 f2 = __half22float2(h2[j]);
                local_max = fmaxf(local_max, fmaxf(fabsf(f2.x), fabsf(f2.y)));
            }
        }
    } else {
        for (int64_t k = tid; k < K; k += blockDim.x) {
            local_max = fmaxf(local_max, fabsf(__half2float(row_in[k])));
        }
    }

    local_max = warpReduceMax(local_max);
    __shared__ float s_warp_max[8];
    if (lane == 0) s_warp_max[wid] = local_max;
    __syncthreads();

    __shared__ float s_scale;
    if (wid == 0) {
        float row_amax = (lane < (blockDim.x / 32)) ? s_warp_max[lane] : 0.0f;
        row_amax = warpReduceMax(row_amax);
        if (lane == 0) {
            const float safe_amax = fmaxf(row_amax, 1e-8f);
            const float inv_scale = safe_amax * (1.0f / 448.0f);
            const float scale = 448.0f / safe_amax;
            d_inv_scale_w[n] = inv_scale;
            s_scale = scale;
        }
    }
    __syncthreads();

    const float scale = s_scale;
    if (reg_cached) {
        const int64_t K_vec = K / 8;
        uint2* out_v = reinterpret_cast<uint2*>(row_out);
        if (tid < K_vec) {
            const half2* h2 = reinterpret_cast<const half2*>(&reg_v0);
            uint8_t out_b[8];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 f2 = __half22float2(h2[j]);
                __nv_fp8_e4m3 q0(f2.x * scale);
                __nv_fp8_e4m3 q1(f2.y * scale);
                out_b[j * 2] = *reinterpret_cast<uint8_t*>(&q0);
                out_b[j * 2 + 1] = *reinterpret_cast<uint8_t*>(&q1);
            }
            out_v[tid] = *reinterpret_cast<const uint2*>(out_b);
        }
        if (tid + 256 < K_vec) {
            const half2* h2 = reinterpret_cast<const half2*>(&reg_v1);
            uint8_t out_b[8];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 f2 = __half22float2(h2[j]);
                __nv_fp8_e4m3 q0(f2.x * scale);
                __nv_fp8_e4m3 q1(f2.y * scale);
                out_b[j * 2] = *reinterpret_cast<uint8_t*>(&q0);
                out_b[j * 2 + 1] = *reinterpret_cast<uint8_t*>(&q1);
            }
            out_v[tid + 256] = *reinterpret_cast<const uint2*>(out_b);
        }
    } else if (can_vec) {
        const int64_t K_vec = K / 8;
        const uint4* in_v = reinterpret_cast<const uint4*>(row_in);
        uint2* out_v = reinterpret_cast<uint2*>(row_out);
        for (int64_t i = tid; i < K_vec; i += blockDim.x) {
            const uint4 in_val = in_v[i];
            const half2* h2 = reinterpret_cast<const half2*>(&in_val);
            uint8_t out_b[8];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 f2 = __half22float2(h2[j]);
                __nv_fp8_e4m3 q0(f2.x * scale);
                __nv_fp8_e4m3 q1(f2.y * scale);
                out_b[j * 2] = *reinterpret_cast<uint8_t*>(&q0);
                out_b[j * 2 + 1] = *reinterpret_cast<uint8_t*>(&q1);
            }
            out_v[i] = *reinterpret_cast<uint2*>(out_b);
        }
    } else {
        for (int64_t k = tid; k < K; k += blockDim.x) {
            __nv_fp8_e4m3 q(__half2float(row_in[k]) * scale);
            row_out[k] = *reinterpret_cast<uint8_t*>(&q);
        }
    }
}

__global__ void __launch_bounds__(256) y32_per_token_scale_epilogue_kernel(
    float* __restrict__ Y, const float* __restrict__ d_inv_scale_tok,
    const float* __restrict__ d_inv_scale_w,
    int64_t T, int64_t N, int64_t ldy) {

    const int64_t t = (int64_t) blockIdx.x;
    if (t >= T) return;

    const float s = (d_inv_scale_tok != nullptr) ? d_inv_scale_tok[t] : 1.0f;
    float* row_y = Y + t * ldy;
    const bool can_vec = ((reinterpret_cast<uintptr_t>(row_y) & 15u) == 0) &&
                         (d_inv_scale_w == nullptr || (reinterpret_cast<uintptr_t>(d_inv_scale_w) & 15u) == 0) &&
                         ((N & 3) == 0);

    if (can_vec) {
        const int64_t N_vec = N / 4;
        float4* y_v = reinterpret_cast<float4*>(row_y);
        if (d_inv_scale_w != nullptr) {
            const float4* w_v = reinterpret_cast<const float4*>(d_inv_scale_w);
            for (int64_t i = threadIdx.x; i < N_vec; i += blockDim.x) {
                float4 v = y_v[i];
                const float4 w = w_v[i];
                v.x *= s * w.x;
                v.y *= s * w.y;
                v.z *= s * w.z;
                v.w *= s * w.w;
                y_v[i] = v;
            }
        } else {
            for (int64_t i = threadIdx.x; i < N_vec; i += blockDim.x) {
                float4 v = y_v[i];
                v.x *= s;
                v.y *= s;
                v.z *= s;
                v.w *= s;
                y_v[i] = v;
            }
        }
    } else if (d_inv_scale_w != nullptr) {
        for (int64_t n = threadIdx.x; n < N; n += blockDim.x) {
            row_y[n] *= s * d_inv_scale_w[n];
        }
    } else {
        for (int64_t n = threadIdx.x; n < N; n += blockDim.x) {
            row_y[n] *= s;
        }
    }
}

__global__ void __launch_bounds__(256) y32_per_token_accum_epilogue_kernel(
    float* __restrict__ Y, const float* __restrict__ y_tmp,
    const float* __restrict__ d_inv_scale_tok,
    const float* __restrict__ d_inv_scale_w,
    int64_t T, int64_t N, int64_t ldy, float beta) {

    const int64_t t = (int64_t) blockIdx.x;
    if (t >= T) return;

    const float s = (d_inv_scale_tok != nullptr) ? d_inv_scale_tok[t] : 1.0f;
    float* row_y = Y + t * ldy;
    const float* row_tmp = y_tmp + t * N;
    const bool can_vec = ((reinterpret_cast<uintptr_t>(row_y) & 15u) == 0) &&
                         ((reinterpret_cast<uintptr_t>(row_tmp) & 15u) == 0) &&
                         (d_inv_scale_w == nullptr || (reinterpret_cast<uintptr_t>(d_inv_scale_w) & 15u) == 0) &&
                         ((N & 3) == 0);

    if (can_vec) {
        const int64_t N_vec = N / 4;
        float4* y_v = reinterpret_cast<float4*>(row_y);
        const float4* tmp_v = reinterpret_cast<const float4*>(row_tmp);
        if (d_inv_scale_w != nullptr) {
            const float4* w_v = reinterpret_cast<const float4*>(d_inv_scale_w);
            for (int64_t i = threadIdx.x; i < N_vec; i += blockDim.x) {
                float4 yv = y_v[i];
                const float4 tv = tmp_v[i];
                const float4 w = w_v[i];
                yv.x = beta * yv.x + tv.x * (s * w.x);
                yv.y = beta * yv.y + tv.y * (s * w.y);
                yv.z = beta * yv.z + tv.z * (s * w.z);
                yv.w = beta * yv.w + tv.w * (s * w.w);
                y_v[i] = yv;
            }
        } else {
            for (int64_t i = threadIdx.x; i < N_vec; i += blockDim.x) {
                float4 yv = y_v[i];
                const float4 tv = tmp_v[i];
                yv.x = beta * yv.x + tv.x * s;
                yv.y = beta * yv.y + tv.y * s;
                yv.z = beta * yv.z + tv.z * s;
                yv.w = beta * yv.w + tv.w * s;
                y_v[i] = yv;
            }
        }
    } else if (d_inv_scale_w != nullptr) {
        for (int64_t n = threadIdx.x; n < N; n += blockDim.x) {
            row_y[n] = beta * row_y[n] + row_tmp[n] * (s * d_inv_scale_w[n]);
        }
    } else {
        for (int64_t n = threadIdx.x; n < N; n += blockDim.x) {
            row_y[n] = beta * row_y[n] + row_tmp[n] * s;
        }
    }
}

__global__ void __launch_bounds__(256) y32_per_token_scale_copy_epilogue_kernel(
    float* __restrict__ Y, const float* __restrict__ y_tmp,
    const float* __restrict__ d_inv_scale_tok,
    const float* __restrict__ d_inv_scale_w,
    int64_t T, int64_t N, int64_t ldy) {

    const int64_t t = (int64_t) blockIdx.x;
    if (t >= T) return;

    const float s = (d_inv_scale_tok != nullptr) ? d_inv_scale_tok[t] : 1.0f;
    float* row_y = Y + t * ldy;
    const float* row_tmp = y_tmp + t * N;
    const bool can_vec = ((reinterpret_cast<uintptr_t>(row_y) & 15u) == 0) &&
                         ((reinterpret_cast<uintptr_t>(row_tmp) & 15u) == 0) &&
                         (d_inv_scale_w == nullptr || (reinterpret_cast<uintptr_t>(d_inv_scale_w) & 15u) == 0) &&
                         ((N & 3) == 0);

    if (can_vec) {
        const int64_t N_vec = N / 4;
        float4* y_v = reinterpret_cast<float4*>(row_y);
        const float4* tmp_v = reinterpret_cast<const float4*>(row_tmp);
        if (d_inv_scale_w != nullptr) {
            const float4* w_v = reinterpret_cast<const float4*>(d_inv_scale_w);
            for (int64_t i = threadIdx.x; i < N_vec; i += blockDim.x) {
                const float4 tv = tmp_v[i];
                const float4 w = w_v[i];
                float4 yv;
                yv.x = tv.x * (s * w.x);
                yv.y = tv.y * (s * w.y);
                yv.z = tv.z * (s * w.z);
                yv.w = tv.w * (s * w.w);
                y_v[i] = yv;
            }
        } else {
            for (int64_t i = threadIdx.x; i < N_vec; i += blockDim.x) {
                const float4 tv = tmp_v[i];
                float4 yv;
                yv.x = tv.x * s;
                yv.y = tv.y * s;
                yv.z = tv.z * s;
                yv.w = tv.w * s;
                y_v[i] = yv;
            }
        }
    } else if (d_inv_scale_w != nullptr) {
        for (int64_t n = threadIdx.x; n < N; n += blockDim.x) {
            row_y[n] = row_tmp[n] * (s * d_inv_scale_w[n]);
        }
    } else {
        for (int64_t n = threadIdx.x; n < N; n += blockDim.x) {
            row_y[n] = row_tmp[n] * s;
        }
    }
}

__global__ void __launch_bounds__(256) y32_copy_or_accum_epilogue_kernel(
    float* __restrict__ Y, const float* __restrict__ y_tmp,
    int64_t T, int64_t N, int64_t ldy, float beta) {

    const int64_t t = (int64_t) blockIdx.x;
    if (t >= T) return;

    float* row_y = Y + t * ldy;
    const float* row_tmp = y_tmp + t * N;
    const bool can_vec = ((reinterpret_cast<uintptr_t>(row_y) & 15u) == 0) &&
                         ((reinterpret_cast<uintptr_t>(row_tmp) & 15u) == 0) &&
                         ((N & 3) == 0);

    if (can_vec) {
        const int64_t N_vec = N / 4;
        float4* y_v = reinterpret_cast<float4*>(row_y);
        const float4* tmp_v = reinterpret_cast<const float4*>(row_tmp);
        if (beta == 0.0f) {
            for (int64_t i = threadIdx.x; i < N_vec; i += blockDim.x) {
                y_v[i] = tmp_v[i];
            }
        } else {
            for (int64_t i = threadIdx.x; i < N_vec; i += blockDim.x) {
                float4 yv = y_v[i];
                const float4 tv = tmp_v[i];
                yv.x = beta * yv.x + tv.x;
                yv.y = beta * yv.y + tv.y;
                yv.z = beta * yv.z + tv.z;
                yv.w = beta * yv.w + tv.w;
                y_v[i] = yv;
            }
        }
    } else {
        if (beta == 0.0f) {
            for (int64_t n = threadIdx.x; n < N; n += blockDim.x) {
                row_y[n] = row_tmp[n];
            }
        } else {
            for (int64_t n = threadIdx.x; n < N; n += blockDim.x) {
                row_y[n] = beta * row_y[n] + row_tmp[n];
            }
        }
    }
}

inline int64_t get_t_bucket(int64_t T) {
    static constexpr int64_t kBuckets[] = {16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096};
    for (int64_t b : kBuckets) {
        if (T <= b) return b;
    }
    return ((T + 1023) / 1024) * 1024;
}

inline int64_t pf_fp8_min_t() {
    const char* v = std::getenv("STRATA_PF_FP8_MIN_T");
    if (v != nullptr && v[0] != '\0') {
        const int64_t n = std::atoll(v);
        if (n >= 1) return n;
    }
    return 128;
}

bool env_is_1(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

// A failed query counts as capture: do not time, and do not skip a quantize, inside a graph.
bool stream_capturing(void* stream) {
    if (stream == nullptr) return false;
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing((cudaStream_t) stream, &st) != cudaSuccess) return true;
    return st != cudaStreamCaptureStatusNone;
}

} // namespace

static bool gdn_qkvz_on() {
    static const bool on = [] { const char* v = std::getenv("STRATA_GDN_QKVZ"); return v != nullptr && std::atoi(v) != 0; }();
    return on;
}

static bool pf_fp8_pair_ticket_on() {
    static const bool on = [] { const char* v = std::getenv("STRATA_PF_FP8_PAIR_TICKET"); return v != nullptr && std::atoi(v) != 0; }();
    return on;
}

bool Gemm::fp8(const uint16_t* X, const uint8_t* W8, const float* d_inv_scale_w, float* Y,
               int64_t T, int64_t N, int64_t K, int64_t ldy, float beta, int64_t ldx, const float* act_amax,
               bool reuse_x8, PrefillFp8Activation* act_ticket, bool per_row_w) {
    if (T <= 0 || N <= 0 || K <= 0) return false;
    if (ldy <= 0) ldy = N;
    if (ldx <= 0) ldx = K;

    if (!lt_handle_ || !lt_pref_) return false;
    const bool pad_small_t = (T < pf_fp8_min_t()) && env_is_1("STRATA_PF_FP8_PAD_SMALL_T");
    const int64_t T_pad = pad_small_t ? (T < 16 ? 16 : ((T + 15) / 16) * 16) : T;
    // 1. Ensure transient activation buffer x8_ is allocated
    int64_t want_bytes = T_pad * K;
    if (x8_bytes_ < want_bytes) {
        if (x8_) cudaFree(x8_);
        x8_ = nullptr;
        x8_bytes_ = 0;
        x8_key_valid_ = false;
        ++x8_gen_;
        int64_t alloc_bytes = std::max<int64_t>(want_bytes, 4ll << 20);
        if (cudaMalloc((void**) &x8_, (size_t) alloc_bytes) != cudaSuccess) {
            cudaGetLastError();
            x8_ = nullptr;
            return false;
        }
        x8_bytes_ = alloc_bytes;
    }

    const bool per_token = [] {
        const char* v = std::getenv("STRATA_PF_FP8_PER_TOKEN");
        return v != nullptr && std::atoi(v) != 0;
    }();

    if (per_token) {
        if (inv_scale_tok_cap_ < (size_t) T) {
            if (d_inv_scale_tok_) cudaFree(d_inv_scale_tok_);
            d_inv_scale_tok_ = nullptr;
            inv_scale_tok_cap_ = 0;
            x8_key_valid_ = false;
            ++x8_gen_;
            const size_t alloc_elems = std::max<size_t>((size_t) T, 4096u);
            if (cudaMalloc((void**) &d_inv_scale_tok_, alloc_elems * sizeof(float)) != cudaSuccess) {
                cudaGetLastError();
                d_inv_scale_tok_ = nullptr;
                return false;
            }
            inv_scale_tok_cap_ = alloc_elems;
        }
    } else if (!d_scale_x_) {
        if (cudaMalloc((void**) &d_scale_x_, sizeof(float)) != cudaSuccess ||
            cudaMalloc((void**) &d_inv_scale_x_, sizeof(float)) != cudaSuccess ||
            cudaMalloc((void**) &d_max_x_, sizeof(float)) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
    }

    // 2. Dynamic scaling of activations X (FP16) -> x8_ (FP8 E4M3) on stream_
    // STRATA_PF_FP8_PER_TOKEN=1: 1-pass per-row amax + quantize; else 2-pass per-tensor (with optional PF-1a fused amax).
    // STRATA_GDN_QKVZ: reuse_x8 skips quantization only when mode, buffers, and shape all match.
    const bool bufs_ok = x8_ != nullptr && x8_bytes_ >= want_bytes &&
                         (per_token ? (d_inv_scale_tok_ != nullptr && inv_scale_tok_cap_ >= (size_t) T)
                                    : (d_inv_scale_x_ != nullptr));
    static const bool allow_pad_reuse = [] {
        const char* v = std::getenv("STRATA_PF_FP8_SMALL_T_REUSE");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    const bool gdn_reuse = (allow_pad_reuse || !pad_small_t) && reuse_x8 && x8_key_valid_ && x8_key_x_ == X && x8_key_amax_ == act_amax &&
                           x8_key_t_ == T && x8_key_k_ == K && x8_key_ldx_ == ldx &&
                           x8_key_per_token_ == per_token && bufs_ok;
    const bool ticket_reuse = !gdn_reuse && pf_fp8_pair_ticket_on() && act_ticket != nullptr &&
                              bufs_ok && act_ticket->matches(X, T, K, ldx, act_amax, per_token, x8_gen_);
    if (!gdn_reuse && !ticket_reuse) {
        ++x8_gen_;
        if (per_token) {
            x16_per_token_quantize_e4m3_kernel<<<(unsigned) T, 256, 0, (cudaStream_t) stream_>>>(
                (const half*) X, x8_, d_inv_scale_tok_, T, K, ldx);
        } else {
            const int num_blocks = 64;
            bool fused_amax = act_amax != nullptr;
            if (fused_amax) {
                const char* v = std::getenv("STRATA_PF_FP8_FUSED_AMAX");
                if (v == nullptr || std::atoi(v) == 0) fused_amax = false;
            }
            if (fused_amax) {
                if (cudaMemcpyAsync(d_max_x_, act_amax, sizeof(float), cudaMemcpyDeviceToDevice,
                                    (cudaStream_t) stream_) != cudaSuccess) {
                    (void) cudaGetLastError();   // the producer's slot could not be read: take the amax pass instead
                    fused_amax = false;
                }
            }
            if (!fused_amax) {
                cudaMemsetAsync(d_max_x_, 0, sizeof(float), (cudaStream_t) stream_);
                x16_amax_kernel<<<num_blocks, 256, 0, (cudaStream_t) stream_>>>(
                    (const half*) X, T, K, ldx, d_max_x_);
            }
            x16_scale_and_quantize_vectorized_kernel<<<num_blocks, 256, 0, (cudaStream_t) stream_>>>(
                (const half*) X, x8_, T, K, ldx, d_max_x_, d_scale_x_, d_inv_scale_x_);
        }
    }
    if (!ticket_reuse && act_ticket != nullptr && pf_fp8_pair_ticket_on()) {
        *act_ticket = {X, T, K, ldx, act_amax, per_token, x8_gen_, true};
    }
    if ((allow_pad_reuse ? (!gdn_reuse && !ticket_reuse) : true) && pad_small_t && T_pad > T) {
        cudaMemsetAsync(x8_ + T * K, 0, (size_t) (T_pad - T) * (size_t) K, (cudaStream_t) stream_);
    }
    if (gdn_qkvz_on() && (allow_pad_reuse || !pad_small_t)) {
        x8_key_valid_ = true;
        x8_key_x_ = X;
        x8_key_amax_ = act_amax;
        x8_key_t_ = T;
        x8_key_k_ = K;
        x8_key_ldx_ = ldx;
        x8_key_per_token_ = per_token;
        fp8_image_ready_ = true;
    }

    const bool need_scale_epilogue = per_token || per_row_w;
    float* matmul_y = Y;
    float matmul_beta = beta;
    int64_t c_ldy = ldy;
    if (pad_small_t || (need_scale_epilogue && beta != 0.0f)) {
        const size_t need_bytes = (size_t) T_pad * (size_t) N * sizeof(float);
        if (y_tmp_bytes_ < need_bytes) {
            if (y_tmp_) cudaFree(y_tmp_);
            y_tmp_ = nullptr;
            y_tmp_bytes_ = 0;
            if (cudaMalloc((void**) &y_tmp_, need_bytes) != cudaSuccess) {
                cudaGetLastError();
                y_tmp_ = nullptr;
                return false;
            }
            y_tmp_bytes_ = need_bytes;
        }
        matmul_y = y_tmp_;
        matmul_beta = 0.0f;
        c_ldy = N;
    }

    // 3. cuBLASLt Tiered Memoization & Execution
    int64_t t_bucket = get_t_bucket(T_pad);
    AlgoCacheKey key{N, K, c_ldy, t_bucket, per_token, per_row_w};

    cublasLtMatmulDescOpaque_t opDesc;
    cublasLtMatmulDescInit(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t opA = CUBLAS_OP_T;
    cublasOperation_t opB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(&opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
    cublasLtMatmulDescSetAttribute(&opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));
    if (!per_row_w) {
        cublasLtMatmulDescSetAttribute(&opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_inv_scale_w, sizeof(d_inv_scale_w));
    }
    if (!per_token) {
        cublasLtMatmulDescSetAttribute(&opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_inv_scale_x_, sizeof(d_inv_scale_x_));
    }

    cublasLtMatrixLayoutOpaque_t aLayout, bLayout, cLayout;
    cublasLtMatrixLayoutInit(&aLayout, CUDA_R_8F_E4M3, (uint64_t) K, (uint64_t) N, (int64_t) K);
    cublasLtMatrixLayoutInit(&bLayout, CUDA_R_8F_E4M3, (uint64_t) K, (uint64_t) T_pad, (int64_t) K);
    cublasLtMatrixLayoutInit(&cLayout, CUDA_R_32F, (uint64_t) N, (uint64_t) T_pad, (int64_t) c_ldy);

    size_t actual_ws_bytes = ws_bytes_;
    AlgoCacheVal cached_algo{{}, 0, false};
    AlgoCacheVal first_algo{{}, 0, false};
    auto it = algo_cache_.find(key);
    // Read per call, like STRATA_PF_FP8_FUSED_AMAX, so a parity run can force the fallthrough in one process.
    const bool force_no_algo = [] {
        const char* v = std::getenv("STRATA_PF_FP8_FORCE_NO_ALGO");
        return v != nullptr && std::atoi(v) != 0;
    }();
    // Real prompt shapes only (the FP8 gate is T >= 128). Once per cache key, on the first call outside capture.
    // Timing during capture is illegal; beta != 0 would accumulate into Y across the timed runs.
    const bool tune = !force_no_algo && beta == 0.0f && T >= pf_fp8_min_t() && env_is_1("STRATA_PF_FP8_AUTOTUNE") &&
                      !stream_capturing(stream_);
    if (tune && it != algo_cache_.end() && it->second.tuned && !it->second.valid) return false;
    if (it != algo_cache_.end() && it->second.valid && (!tune || it->second.tuned)) {
        cached_algo = it->second;
    } else if (tune) {
        cublasLtMatrixLayoutOpaque_t bLayoutBucket, cLayoutBucket;
        cublasLtMatrixLayoutInit(&bLayoutBucket, CUDA_R_8F_E4M3, (uint64_t) K, (uint64_t) t_bucket, (int64_t) K);
        cublasLtMatrixLayoutInit(&cLayoutBucket, CUDA_R_32F, (uint64_t) N, (uint64_t) t_bucket, (int64_t) c_ldy);
        cublasLtMatmulHeuristicResult_t found[8];
        int nret = 0;
        cublasStatus_t hst = cublasLtMatmulAlgoGetHeuristic(
            (cublasLtHandle_t) lt_handle_, &opDesc, &aLayout, &bLayoutBucket, &cLayoutBucket, &cLayoutBucket,
            (cublasLtMatmulPreference_t) lt_pref_, 8, found, &nret);
        int valid_i[8];
        int nvalid = 0;
        if (hst == CUBLAS_STATUS_SUCCESS) {
            for (int i = 0; i < nret && i < 8; ++i) {
                if (found[i].state == CUBLAS_STATUS_SUCCESS && found[i].workspaceSize <= actual_ws_bytes)
                    valid_i[nvalid++] = i;
            }
        }
        int best = -1;
        float best_ms = 0.0f;
        cudaEvent_t ev0 = nullptr, ev1 = nullptr;
        const bool can_time = nvalid > 0 && cudaEventCreate(&ev0) == cudaSuccess &&
                              cudaEventCreate(&ev1) == cudaSuccess;
        if (!can_time) {
            if (ev0 != nullptr) (void) cudaEventDestroy(ev0);
            if (ev1 != nullptr) (void) cudaEventDestroy(ev1);
            ev0 = nullptr;
            ev1 = nullptr;
        }
        auto run_one = [&](const cublasLtMatmulAlgo_t& algo) -> cublasStatus_t {
            float alpha = 1.0f;
            return cublasLtMatmul(
                (cublasLtHandle_t) lt_handle_, &opDesc,
                &alpha, W8, &aLayout,
                x8_, &bLayout,
                &matmul_beta, matmul_y, &cLayout,
                matmul_y, &cLayout,
                &algo,
                workspace_, actual_ws_bytes, (cudaStream_t) stream_);
        };
        if (can_time) {
            static bool said = false;
            if (!said) {
                said = true;
                std::fprintf(stderr, "strata prefill: FP8 cuBLASLt autotune on (STRATA_PF_FP8_AUTOTUNE=1); "
                                     "once per prompt shape, never during capture\n");
            }
            for (int v = 0; v < nvalid; ++v) {
                const cublasLtMatmulAlgo_t& algo = found[valid_i[v]].algo;
                if (run_one(algo) != CUBLAS_STATUS_SUCCESS) continue;   // warmup; a failed algo is not timed
                float samples[3];
                bool ok = true;
                for (int r = 0; ok && r < 3; ++r) {
                    ok = cudaEventRecord(ev0, (cudaStream_t) stream_) == cudaSuccess &&
                         run_one(algo) == CUBLAS_STATUS_SUCCESS &&
                         cudaEventRecord(ev1, (cudaStream_t) stream_) == cudaSuccess &&
                         cudaEventSynchronize(ev1) == cudaSuccess &&
                         cudaEventElapsedTime(&samples[r], ev0, ev1) == cudaSuccess;
                }
                if (!ok) continue;
                float a = samples[0], b = samples[1], c = samples[2];
                if (a > b) { float t = a; a = b; b = t; }
                if (b > c) { float t = b; b = c; c = t; }
                if (a > b) { float t = a; a = b; b = t; }
                if (best < 0 || b < best_ms) { best = v; best_ms = b; }
            }
        }
        if (ev0) (void) cudaEventDestroy(ev0);
        if (ev1) (void) cudaEventDestroy(ev1);
        // No timed winner: the first valid heuristic, which is the single-heuristic choice when it is valid.
        const int pick = best >= 0 ? best : (nvalid > 0 ? 0 : -1);
        if (pick >= 0) {
            const cublasLtMatmulHeuristicResult_t& chosen = found[valid_i[pick]];
            cached_algo.workspace_bytes = chosen.workspaceSize;
            std::memcpy(cached_algo.algo_data, &chosen.algo, sizeof(cublasLtMatmulAlgo_t));
            cached_algo.valid = true;
            cached_algo.tuned = true;
            algo_cache_[key] = cached_algo;
            if (pick != 0) {
                const cublasLtMatmulHeuristicResult_t& head = found[valid_i[0]];
                first_algo.workspace_bytes = head.workspaceSize;
                std::memcpy(first_algo.algo_data, &head.algo, sizeof(cublasLtMatmulAlgo_t));
                first_algo.valid = true;
                first_algo.tuned = true;
            }
        }
    } else if (it != algo_cache_.end() && it->second.valid) {
        cached_algo = it->second;
    } else {
        // Query heuristic for bucketed layout
        cublasLtMatrixLayoutOpaque_t bLayoutBucket, cLayoutBucket;
        cublasLtMatrixLayoutInit(&bLayoutBucket, CUDA_R_8F_E4M3, (uint64_t) K, (uint64_t) t_bucket, (int64_t) K);
        cublasLtMatrixLayoutInit(&cLayoutBucket, CUDA_R_32F, (uint64_t) N, (uint64_t) t_bucket, (int64_t) c_ldy);

        cublasLtMatmulHeuristicResult_t heuristicResult;
        int returnedResults = 0;
        cublasStatus_t hst = cublasLtMatmulAlgoGetHeuristic(
            (cublasLtHandle_t) lt_handle_, &opDesc, &aLayout, &bLayoutBucket, &cLayoutBucket, &cLayoutBucket,
            (cublasLtMatmulPreference_t) lt_pref_, 1, &heuristicResult, &returnedResults);

        if (hst == CUBLAS_STATUS_SUCCESS && returnedResults > 0 &&
            heuristicResult.state == CUBLAS_STATUS_SUCCESS &&
            heuristicResult.workspaceSize <= actual_ws_bytes) {
            cached_algo.workspace_bytes = heuristicResult.workspaceSize;
            std::memcpy(cached_algo.algo_data, &heuristicResult.algo, sizeof(cublasLtMatmulAlgo_t));
            cached_algo.valid = true;
            algo_cache_[key] = cached_algo;
        }
    }

    // A cache miss or a rejected heuristic must fall through to FP16/MMQ, not leave Y unwritten.
    if (force_no_algo) return false;

    auto launch = [&](const AlgoCacheVal& slot) -> cublasStatus_t {
        float alpha = 1.0f;
        cublasLtMatmulAlgo_t algo;
        std::memcpy(&algo, slot.algo_data, sizeof(cublasLtMatmulAlgo_t));
        return cublasLtMatmul(
            (cublasLtHandle_t) lt_handle_, &opDesc,
            &alpha, W8, &aLayout,
            x8_, &bLayout,
            &matmul_beta, matmul_y, &cLayout,
            matmul_y, &cLayout,
            &algo,
            workspace_, actual_ws_bytes, (cudaStream_t) stream_);
    };
    if (cached_algo.valid) {
        cublasStatus_t mst = launch(cached_algo);
        if (mst != CUBLAS_STATUS_SUCCESS && first_algo.valid) {
            mst = launch(first_algo);   // the timed choice failed: the first heuristic
            if (mst == CUBLAS_STATUS_SUCCESS) algo_cache_[key] = first_algo;
        }
        if (mst != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata fp8 prefill: cublasLtMatmul failed with status %d\n", (int) mst);
            // Remember a tuned shape that produced nothing, so the next call falls through without timing again.
            if (cached_algo.tuned) {
                cached_algo.valid = false;
                algo_cache_[key] = cached_algo;
            }
            return false;
        }
        if (need_scale_epilogue) {
            const float* tok_ptr = per_token ? d_inv_scale_tok_ : nullptr;
            const float* w_ptr = per_row_w ? d_inv_scale_w : nullptr;
            if (pad_small_t && beta == 0.0f) {
                y32_per_token_scale_copy_epilogue_kernel<<<(unsigned) T, 256, 0, (cudaStream_t) stream_>>>(
                    Y, y_tmp_, tok_ptr, w_ptr, T, N, ldy);
            } else if (beta == 0.0f) {
                y32_per_token_scale_epilogue_kernel<<<(unsigned) T, 256, 0, (cudaStream_t) stream_>>>(
                    Y, tok_ptr, w_ptr, T, N, ldy);
            } else {
                y32_per_token_accum_epilogue_kernel<<<(unsigned) T, 256, 0, (cudaStream_t) stream_>>>(
                    Y, y_tmp_, tok_ptr, w_ptr, T, N, ldy, beta);
            }
        } else if (pad_small_t) {
            y32_copy_or_accum_epilogue_kernel<<<(unsigned) T, 256, 0, (cudaStream_t) stream_>>>(
                Y, y_tmp_, T, N, ldy, beta);
        }
        return true;
    }
    return false;
}
#endif

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
#ifdef STRATA_USE_HIP
    // #313 (opt-in STRATA_WMMA_GEMM=1): RDNA3 / RDNA3.5 WMMA dense GEMM, before hipBLASLt (falls through when false)
    static const bool wmma_on = [] { const char* v = std::getenv("STRATA_WMMA_GEMM"); return v && v[0] != 0 && v[0] != '0'; }();
    static const bool pf_on = [] { const char* v = std::getenv("STRATA_PF_GEMM"); return v && v[0] == '1'; }();
    static const int64_t pf_min_t = [] { const char* v = std::getenv("STRATA_PF_SWITCH_MIN_T"); return v ? (int64_t) std::atoll(v) : (int64_t) 0; }();
    if (pf_on && T >= pf_min_t && strata_pf_gemm_f16(X, W, Y, T, N, K, ldy, beta, stream_)) return;   // S23: opt-in
    if (wmma_on && T >= 16 && (beta == 0.0f || beta == 1.0f) &&
        strata_wmma_gemm_f16(X, W, Y, T, N, K, ldy, beta, stream_)) {
        return;
    }
#endif
#if defined(__HIPCC__)
    if (f16_io_ && beta == 0.0f) { f16_inplace(X, W, Y, T, N, K, ldy); return; }
#endif
    const float alpha = 1.0f;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (try_hipblaslt(hipblaslt_state_, strata::prefill::hipblaslt::InputType::f16, X, W, Y, T, N, K, ldy,
                      beta, stream_)) {
        STRATA_ABSORB_HIPBLAS_STICKY("hipBLASLt f16");
        return;
    }
#endif
#if !defined(__HIPCC__) && defined(STRATA_EXPERIMENTAL_SM60)
    // Pascal, STRATA_DENSE_F32=1: cublasSgemm on FP32 copies (dense_f32_on above: where it is bitwise the FP16 call)
    if (const int64_t min_t = beta == 0.0f ? dense_f32_min_t(N, K) : 0; min_t > 0 && T >= min_t && dense_f32_on()) {
        const int64_t n_sl = (T * K + kXSliceElems - 1) / kXSliceElems;   // even slices of at most 64 MiB as fp32
        const int64_t x_rows = (T + n_sl - 1) / n_sl;
        // grow() frees a buffer an earlier GEMM on this stream may still read
        if (tc_w_elems_ < 2 * N * K || tc_x_elems_ < 2 * x_rows * K) cudaStreamSynchronize((cudaStream_t) stream_);
        if (grow(tc_w_, tc_w_elems_, 2 * N * K) && grow(tc_x_, tc_x_elems_, 2 * x_rows * K)) {
            static bool said = false;
            if (!said) {
                said = true;
                std::fprintf(stderr, "prefill gemm: the dense projections' FP16 GEMMs run as cublasSgemm on FP32 copies "
                                     "(STRATA_DENSE_F32=1; gemm_f32_parity checks the bits against this cuBLAS)\n");
            }
            float* const wf = reinterpret_cast<float*>(tc_w_);
            float* const xf = reinterpret_cast<float*>(tc_x_);
            f16_to_f32(W, wf, N * K, (cudaStream_t) stream_);
            for (int64_t t0 = 0; t0 < T; t0 += x_rows) {
                const int64_t n = std::min<int64_t>(x_rows, T - t0);
                f16_to_f32(X + t0 * K, xf, n * K, (cudaStream_t) stream_);
                ck(cublasSgemm((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) n, (int) K, &alpha, wf, (int) K,
                               xf, (int) K, &beta, Y + t0 * ldy, (int) ldy),
                   "cublasSgemm (FP16 operands, STRATA_DENSE_F32)");
            }
            return;
        }
        cudaGetLastError();   // no room for the copies: the FP16 call below
    }
#endif
#if defined(__HIPCC__)
    if (rdna2_sgemm(X, W, Y, T, N, K, ldy, beta, false)) return;
#endif
#if !defined(__HIPCC__)
    const auto f16_gemm = [&](cublasGemmAlgo_t algo) {
        return cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha,
                            W, CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                            CUBLAS_COMPUTE_32F, algo);
    };
    // Issue #1650: with a second CUDA device initialized in the process, cuBLAS 12 answers the default algorithm of a
    // few shapes - narrow ones, and the set moves with K - with CUBLAS_STATUS_INTERNAL_ERROR; a fixed algorithm keeps
    // a kernel that state does not break.  Only the shapes that fail are moved (the failing call itself is harmless);
    // the wide shapes, for which the fixed algorithms are markedly slower, stay on the default.
    const int64_t shape = ((int64_t) K << 32) | (uint32_t) N;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
    for (int i = 0; i < f16_algo_n_; ++i)
        if (f16_algo_shape_[i] == shape) { algo = CUBLAS_GEMM_ALGO2; break; }
    cublasStatus_t st = f16_gemm(algo);
    if (st == CUBLAS_STATUS_INTERNAL_ERROR && algo == CUBLAS_GEMM_DEFAULT) {
        if (f16_algo_n_ < kF16AlgoSlots) f16_algo_shape_[f16_algo_n_++] = shape;
        st = f16_gemm(CUBLAS_GEMM_ALGO2);
    }
    ck(st, "cublasGemmEx f16");
#else
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx f16");
#endif
    STRATA_ABSORB_HIPBLAS_STICKY("cublasGemmEx f16");
}

#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)
bool Gemm::native_mmq(const uint16_t* X, int type, const void* W, float* Y, int64_t T, int64_t N, int64_t K,
                      int64_t ldy) {
    namespace mmq = strata::prefill::mmq;
    constexpr int64_t kRows = 1024, kMaxK = 8192;
    static const bool enabled = [] {
        const char* e = std::getenv("STRATA_DENSE_MMQ");
        return e && e[0] == '1';
    }();
    if (!enabled || mmq_failed_ || !mmq::built() || !mmq::fits(type, N)) return false;
    // K must be a multiple of 256: llama.cpp's MMQ loads the weights in 256-value K chunks, and the
    // chunk past a partial row reads past the row (the next row's bytes - or, for the last weight
    // row, bytes past the tensor, which no caller guarantees to be zeros). The MoE path is safe
    // because its gather buffers carry a zeroed tail; a dense tensor is passed as-is, so a K that
    // is not a full chunk multiple (e.g. the 640-value shared-expert down: 20 blocks of 32) stays on
    // the dequantize + cuBLAS path.
    if (K > kMaxK || K % 256 != 0) return false;
    int64_t R = scratch_elems_ / (2 * K);   // FP32 activations for a chunk fit in the FP16 dequant scratch
    if (R > kRows) R = kRows;
    if (R > T) R = T;
    if (R < 1) return false;
    if (ldy <= 0) ldy = N;

    const auto up = [](size_t v) { return (v + 255) & ~(size_t) 255; };
    // The row table is read past its end by llama.cpp's MMQ (mul_mat_q: a whole J-column tile from the start of each tile, J up to 128; the stream-k fixup:
    // up to roundup(max_rows, J) from the expert's first row): kRows + the margin of mmq_ids_margin.hpp (max_rows <= kRows), the identity written over all of it.
    // Exact, with only the 256 bytes of the bounds after it, a tile of 112 columns on a full chunk read 384 bytes past the table, 128 past the allocation.
    const size_t ids_rows = (size_t) mmq::mmq_ids_table_rows((uint64_t) kRows, (uint64_t) kRows);
    const size_t ident_off = up(mmq::q8_bytes(kRows, kMaxK));
    const size_t bounds_off = ident_off + up(ids_rows * 4);
    if (!mmq_buf_) {
        if (cudaMalloc(&mmq_buf_, bounds_off + 256) != cudaSuccess) {
            cudaGetLastError();
            mmq_failed_ = true;
            return false;
        }
        mmq_ctx_ = new mmq::Context();
        mmq::iota((int32_t*) ((uint8_t*) mmq_buf_ + ident_off), (int64_t) ids_rows, stream_);
        std::fprintf(stderr, "prefill gemm: dense MMQ on (STRATA_DENSE_MMQ=1)\n");
    }
    void* xq = mmq_buf_;
    const int32_t* ident = (const int32_t*) ((uint8_t*) mmq_buf_ + ident_off);
    int32_t* bounds = (int32_t*) ((uint8_t*) mmq_buf_ + bounds_off);
    float* xf = (float*) scratch_;
    auto* ctx = static_cast<mmq::Context*>(mmq_ctx_);

    for (int64_t r0 = 0; r0 < T; r0 += R) {
        const int64_t rows = (T - r0 < R) ? T - r0 : R;
        mmq::f16_to_f32(X + r0 * K, xf, rows * K, stream_);
        mmq::quantize(xf, nullptr, xq, type, K, K, rows, stream_);
        mmq::set_bounds(bounds, (int32_t) rows, stream_);
        mmq::Product p;
        p.w = W;
        p.type = type;
        p.w_rows = N;
        p.w_cols = K;
        p.expert_bytes = mmq::matrix_bytes(type, N, K);
        p.n = 1;
        p.xq = xq;
        p.bounds = bounds;
        p.ids = ident;
        p.total_rows = rows;
        p.max_rows = rows;
        p.dst = Y + r0 * ldy;
        p.ld_dst = ldy;
        ctx->run(p, stream_);
    }
    return true;
}
#endif
void Gemm::f16_inplace(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy) {
#if defined(__HIPCC__)
    const float one = 1.0f, zero = 0.0f;
    bool done = false;
#if defined(STRATA_ROCBLAS_AVAILABLE)
    done = try_rocblas_f16o(rocblas_state_, X, W, Y, T, N, K, ldy);   // the table's kernel for this shape, if any
#endif
    if (!done) {
        ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &one, W,
                        CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &zero, Y, CUDA_R_16F, (int) (2 * ldy),
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
           "cublasGemmEx f16 out");
    }
    static const bool dbg_nan = std::getenv("STRATA_DBG_NAN") != nullptr;
    widen_rows_f16<<<(unsigned) T, 256, 0, (cudaStream_t) stream_>>>(Y, N, ldy, dbg_nan ? 1 : 0);
    if (dbg_nan) {   // debug only: a sync per GEMM
        unsigned long long bad = 0, zero_count = 0;
        (void) hipStreamSynchronize((hipStream_t) stream_);
        if (hipMemcpyFromSymbol(&bad, HIP_SYMBOL(g_f16_nonfinite), sizeof bad) == hipSuccess && bad != 0) {
            std::fprintf(stderr, "strata prefill: STRATA_DBG_NAN: %llu non-finite FP16 GEMM outputs (T=%lld N=%lld K=%lld; FP16 ends at 65504)%s",
                         bad, (long long) T, (long long) N, (long long) K, "\n");
            (void) hipMemcpyToSymbol(HIP_SYMBOL(g_f16_nonfinite), &zero_count, sizeof zero_count);
        }
    }
#else
    (void) X; (void) W; (void) Y; (void) T; (void) N; (void) K; (void) ldy;
#endif
}

void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta, int64_t ldx, const float* act_amax, bool reuse_x8,
                  PrefillFp8Activation* act_ticket) {
    if (gdn_qkvz_on()) fp8_image_ready_ = false;
    (void) reuse_x8;
#ifdef STRATA_USE_HIP
    // S23 (opt-in STRATA_PF_PAD=1, with STRATA_PF_GEMM=1 on chunks of STRATA_PF_SWITCH_MIN_T+ tokens): the weight
    // dequantized into the scratch with row stride K + 64 halves, so no 4 KB-multiple stride camps on the memory
    // channels (gemm_probe8: K 6144 19 -> 29 TFLOPS from W alone, 35 with X padded too).  Bitwise the same products.
    static const bool pad_on = [] { const char* v = std::getenv("STRATA_PF_PAD"); return v && v[0] == '1'; }();
    static const bool pf_on = [] { const char* v = std::getenv("STRATA_PF_GEMM"); return v && v[0] == '1'; }();
    static const int64_t pf_min_t = [] { const char* v = std::getenv("STRATA_PF_SWITCH_MIN_T"); return v ? (int64_t) std::atoll(v) : (int64_t) 0; }();
    // (strata_pf_padded_gemm_available(): the CURRENT device runs the padded gfx11 product.  On a gfx1151 + gfx1201 split the process-wide switch
    // is on for both stages; the gfx12 stage would dequantize the weight padded, find no product, and do it again unpadded below.)
    if (pad_on && pf_on && T >= pf_min_t && T >= 64 && N >= 512 && K % 32 == 0 && N * (K + 64) <= scratch_elems_ &&
        strata_pf_padded_gemm_available()) {
        const int64_t ldw = K + 64;
        if (strata::kernels::dequant_f16_ld(ggml_type, W_blocks, 0, N, K, ldw, scratch_, stream_) &&
            strata_pf_gemm_f16_ld(X, ldx > 0 ? ldx : K, scratch_, ldw, Y, T, N, K, ldy, beta, stream_)) {
            return;
        }
    }
#endif
    if (ldx > 0 && ldx != K) {
#ifdef STRATA_USE_HIP
        // X arrived padded but the weight cannot be (an i-quant without a strided dequant, or the scratch is too
        // small for N x (K + 64)): the weight unpadded, X read at its stride - the same products, the same bits
        const int64_t rows = scratch_elems_ / K;
        bool ok = rows > 0 && strata_pf_padded_gemm_available();
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; ok && r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            ok = strata_pf_gemm_f16_ld(X, ldx, scratch_, K, Y + r0, T, n, K, ldy, beta, stream_);
        }
        if (ok) return;
#endif
        // Nothing took a padded X: the padded product is the gfx11 matrix cores' (STRATA_PF_PAD asks
        // strata_pf_padded_gemm_available() before padding now, so this is a stale producer or another card's
        // activation).  Copy the rows unpadded and run the ordinary paths - the same products, the same bits - rather
        // than ending the engine.
        const uint16_t* Xu = unpad_x(X, T, K, ldx);
        if (Xu != nullptr) {
            native(Xu, ggml_type, W_blocks, Y, T, N, K, ldy, beta, 0, act_amax, reuse_x8, nullptr);
            return;
        }
        std::fprintf(stderr, "prefill gemm: a padded X (ldx %lld, K %lld) has no path on this card and could not be "
                             "copied unpadded (no device memory)\n", (long long) ldx, (long long) K);
        std::exit(1);
    }
#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)
    if (beta == 0.0f && (ldx == 0 || ldx == K) && native_mmq(X, ggml_type, W_blocks, Y, T, N, K, ldy)) return;
#endif
#ifndef STRATA_USE_HIP
    // Dense FP8 Prefill Weight Cache (w8_):
    // Caches weights in CUDA_R_8F_E4M3 (1 byte/element, saving 50% VRAM vs FP16).
    // When prompt chunk T >= 128, invokes cublasLtMatmul using FP8 Tensor Cores with FP32 accumulation.
    auto parse_w8_mib_bytes = [](const char* v) -> int64_t {
        if (v == nullptr || v[0] == '\0') return -1;
        if (std::strcmp(v, "auto") == 0) return 16ll << 30;
        if (std::strchr(v, ',') != nullptr) {
            int dev = 0;
            if (cudaGetDevice(&dev) != cudaSuccess || dev < 0) dev = 0;
            const char* p = v;
            for (int idx = 0;; ++idx) {
                const char* comma = std::strchr(p, ',');
                if (idx == dev) {
                    if (*p == '\0' || *p == ',') return 0;
                    const int64_t mib = std::atoll(p);
                    return mib > 0 ? (mib << 20) : 0;
                }
                if (comma == nullptr) return 0;
                p = comma + 1;
            }
        }
        const int64_t mib = std::atoll(v);
        return mib > 0 ? (mib << 20) : 0;
    };
    // STRATA_PF_W8_MIN_WEIGHT_MIB=N (CUDA, opt-in): do not cache a weight smaller than N MiB. The scarce byte budget
    // then goes to the larger prefill GEMM layers that would otherwise fall back to BF16. Unset / 0 changes nothing.
    static const int64_t w8_min_weight_bytes = [] {
        const char* v = std::getenv("STRATA_PF_W8_MIN_WEIGHT_MIB");
        const int64_t mib = v != nullptr ? std::atoll(v) : 0;
        return mib > 0 ? (mib << 20) : int64_t(0);
    }();
    // STRATA_PF_W8_SIZE_PRIORITY=1 (CUDA, opt-in): on a free-VRAM-floor skip or an allocation failure, evict the
    // smallest cached weight (when it is at most half the incoming weight) and retry before freezing the budget.
    static const bool w8_size_priority = env_is_1("STRATA_PF_W8_SIZE_PRIORITY");
    if (w8_budget_ < 0) {
        const char* v = std::getenv("STRATA_PF_W8_CACHE_MIB");
        const char* on = std::getenv("STRATA_PF_W8_CACHE");
        const char* w16_env = std::getenv("STRATA_PF_W16_CACHE_MIB");
        if (on != nullptr && std::strcmp(on, "0") == 0) {
            w8_budget_ = 0;
            w8_reserve_ = kW8AutoReserve;
        } else if (v != nullptr && v[0] != '\0') {
            const int64_t parsed = parse_w8_mib_bytes(v);
            w8_budget_ = parsed >= 0 ? parsed : 0;
            w8_reserve_ = kW8AutoReserve;
        } else if (on != nullptr) {
            w8_budget_ = (std::strcmp(on, "1") == 0 || std::atoi(on) != 0) ? (16ll << 30) : 0;
            w8_reserve_ = kW8AutoReserve;
        } else if (w16_env != nullptr && std::strcmp(w16_env, "0") != 0) {
            // Explicit W16 cache test requested: leave w8 off
            w8_budget_ = 0;
            w8_reserve_ = kW8AutoReserve;
        } else {
            // Opt-in only: enabled only when explicitly requested or via w8_auto_default_()
            const bool auto_on = w8_auto_default_().load();
            w8_budget_ = auto_on ? (16ll << 30) : 0;
            w8_reserve_ = kW8AutoReserve;
        }
    }
    if (w8_budget_ > 0) {
        const bool per_row = env_is_1("STRATA_PF_W8_PER_ROW");
        const int64_t bytes = N * K;
        const int64_t scale_bytes = per_row ? (N * (int64_t) sizeof(float)) : (int64_t) sizeof(float);
        const int64_t total_bytes = bytes + scale_bytes;
        uint8_t* cw8 = nullptr;
        float* d_scale_w = nullptr;
        bool cw8_per_row = false;
        for (const W8Entry& e : w8_) {
            if (e.p == W_blocks && e.type == ggml_type && e.n == N && e.k == K && e.per_row == per_row) {
                cw8 = e.d;
                d_scale_w = e.scale_w;
                cw8_per_row = e.per_row;
                break;
            }
        }
        if (cw8 == nullptr && w8_used_ + total_bytes <= w8_budget_ &&
            N * K <= scratch_elems_ && total_bytes >= w8_min_weight_bytes) {
            size_t fb = 0, tb = 0;
            if (cudaMemGetInfo(&fb, &tb) == cudaSuccess) {
                // Hard 1,400 MiB boot-free safety floor whenever PF_W8_CACHE is active (prevents Xid 31).
                const int64_t effective_reserve = std::max<int64_t>(w8_reserve_, kW8AutoReserve);
                // This loop runs once on the default path. With STRATA_PF_W8_SIZE_PRIORITY=1 a floor / allocation
                // failure first evicts the smallest cached weight (when it is at most half the incoming weight) and
                // retries; the budget freezes only when no entry can make room.
                for (;;) {
                    bool amax_alloc_failed = false;
                    if (fb > (size_t) total_bytes + (size_t) effective_reserve &&
                        cudaMalloc((void**) &cw8, (size_t) bytes) == cudaSuccess &&
                        cudaMalloc((void**) &d_scale_w, (size_t) scale_bytes) == cudaSuccess) {
                        bool q_ok = true;
                        if (!per_row) {
                            if (!d_max_w_ && cudaMalloc((void**) &d_max_w_, sizeof(float)) != cudaSuccess) {
                                d_max_w_ = nullptr;
                                q_ok = false;
                            }
                        }
                        if (q_ok) {
                            strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
                            if (per_row) {
                                w16_per_row_quantize_e4m3_kernel<<<(unsigned) N, 256, 0, (cudaStream_t) stream_>>>(
                                    (const half*) scratch_, cw8, d_scale_w, N, K);
                            } else {
                                cudaMemsetAsync(d_max_w_, 0, sizeof(float), (cudaStream_t) stream_);
                                w16_amax_kernel<<<64, 256, 0, (cudaStream_t) stream_>>>((const half*) scratch_, N * K, d_max_w_);
                                w16_scale_and_quantize_vectorized_kernel<<<64, 256, 0, (cudaStream_t) stream_>>>(
                                    (const half*) scratch_, cw8, N * K, d_max_w_, d_scale_w);
                            }
                            w8_.push_back({W_blocks, ggml_type, N, K, cw8, d_scale_w, per_row});
                            cw8_per_row = per_row;
                            w8_used_ += total_bytes;
                            if (!w8_logged_) {
                                w8_logged_ = true;
                                std::fprintf(stderr, "strata prefill: surgical FP8 dense weight cache active (%lld MiB cached, per-row=%d, card keeps %lld MiB free; STRATA_PF_W8_CACHE=0 turns it off)\n",
                                             (long long) (w8_used_ >> 20), per_row ? 1 : 0, (long long) (effective_reserve >> 20));
                            }
                            break;   // the entry is cached
                        }
                        amax_alloc_failed = !q_ok;
                    }
                    (void) cudaGetLastError();
                    if (cw8) { (void) cudaFree(cw8); cw8 = nullptr; }
                    if (d_scale_w) { (void) cudaFree(d_scale_w); d_scale_w = nullptr; }
                    // The original path did not freeze the budget for an amax scratch allocation failure.
                    if (amax_alloc_failed) break;
                    bool retried = false;
                    if (w8_size_priority && !w8_.empty()) {
                        size_t smallest = 0;
                        int64_t smallest_bytes = 0;
                        for (size_t i = 0; i < w8_.size(); ++i) {
                            const W8Entry& e = w8_[i];
                            const int64_t eb = e.n * e.k +
                                (e.per_row ? e.n * (int64_t) sizeof(float) : (int64_t) sizeof(float));
                            if (smallest_bytes == 0 || eb < smallest_bytes) {
                                smallest = i;
                                smallest_bytes = eb;
                            }
                        }
                        if (smallest_bytes > 0 && smallest_bytes <= total_bytes / 2) {
                            W8Entry& e = w8_[smallest];
                            (void) cudaFree(e.d);
                            (void) cudaFree(e.scale_w);
                            w8_used_ -= smallest_bytes;
                            w8_.erase(w8_.begin() + smallest);
                            // Re-check free VRAM before the retry; the hard floor is unchanged.
                            retried = cudaMemGetInfo(&fb, &tb) == cudaSuccess;
                        }
                    }
                    if (!retried) {
                        // Cap budget to currently used cache if any weights cached; do NOT zero out budget on initial skip!
                        if (w8_used_ > 0) {
                            w8_budget_ = w8_used_;
                        }
                        break;
                    }
                }
            }
        }
        static const bool pad_small_t_env = [] {
            const char* v = std::getenv("STRATA_PF_FP8_PAD_SMALL_T");
            return v != nullptr && v[0] == '1' && v[1] == '\0';
        }();
        if (cw8 != nullptr && (T >= pf_fp8_min_t() || (pad_small_t_env && T >= 1))) {
            if (fp8(X, cw8, d_scale_w, Y, T, N, K, ldy, beta, ldx, act_amax, reuse_x8, act_ticket, cw8_per_row)) return;
        }
    }

    // The FP16 weight kept on the device after its first dequant.  NOT bit-exact against the uncached sliced form on every
    // card (see the default below): off unless STRATA_PF_W16_CACHE_MIB is set.  When on it is sized by the card, never by the user.  It takes only VRAM nobody else holds: a weight is
    // cached while the card keeps kW16AutoReserve free afterwards (the MTP prompt capture, the batch windows and a long
    // prompt's buffers are allocated later, out of that room; a replica with 309 MiB free ran out of memory at 14K tokens),
    // so a card whose expert cache fills it caches nothing and a card with room (all experts resident) caches what the
    // prompt path uses (~6 GB for Qwen3.8).  STRATA_PF_W16_CACHE_MIB=N is an explicit budget (2 GiB kept free, as before),
    // =0 turns it off.
    if (w16_budget_ < 0) {
        const char* v = std::getenv("STRATA_PF_W16_CACHE_MIB");
        if (v != nullptr && std::strcmp(v, "auto") == 0) {
            // STRATA_PF_W16_CACHE_MIB=auto (tester switch, round0212 item 7): the automatic sizing - weights are cached while the card
            // keeps kW16AutoReserve free afterwards, so a card whose expert cache fills it caches nothing.  Not bit-exact (see below),
            // so it is not the default until the owner says yes.
            w16_budget_ = 16ll << 30;
            w16_reserve_ = kW16AutoReserve;
        } else if (v != nullptr) {
            w16_budget_ = (int64_t) std::atoll(v) * (1ll << 20);
            w16_reserve_ = 2ll << 30;
        } else {
            // INT3 turned the default off (11-01) because gemm_dequant_side_parity failed with it on sm_86 / sm_120 / sm_121.  That test builds its Gemm
            // with a 10.5 M-element scratch, smaller than four of its weights, so uncached those take the sliced path (several GEMMs) and cached they
            // take one full-N GEMM: not the engine.  The engine's scratch is GEMM_SCRATCH = 32 M elements and every dense projection of Qwen3.8
            // (largest attn_q, 31.5 M) fits it, so cached and uncached are the same GEMM on the same FP16 values (gemm_w16_parity: the engine's scratch,
            // shapes and types, bitwise).  Round0212 item 7: on when the engine says so (set_w16_auto_default: Linux CUDA with batch slots), else off
            // as in v0.2.9.  It takes only VRAM beyond kW16AutoReserve, so a card whose expert cache fills it caches nothing.  =0 turns it off.
            w16_budget_ = w16_auto_default_().load() ? (16ll << 30) : 0;
            w16_reserve_ = kW16AutoReserve;
        }
    }
    if (w8_budget_ <= 0 && w16_budget_ > 0 && !side_scratch_) {
        const int64_t bytes = N * K * 2;
        uint16_t* cw = nullptr;
        for (const W16Entry& e : w16_)
            if (e.p == W_blocks && e.type == ggml_type && e.n == N && e.k == K) { cw = e.d; break; }
        if (cw == nullptr && w16_used_ + bytes <= w16_budget_) {
            size_t fb = 0, tb = 0;
            if (cudaMemGetInfo(&fb, &tb) == cudaSuccess && fb > (size_t) bytes + (size_t) w16_reserve_ &&
                cudaMalloc((void**) &cw, (size_t) bytes) == cudaSuccess) {
                strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, cw, stream_);
                w16_.push_back({W_blocks, ggml_type, N, K, cw});
                w16_used_ += bytes;
            } else {
                (void) cudaGetLastError();
                cw = nullptr;
                if (w16_budget_ > 0 && !w16_logged_) {
                    w16_logged_ = true;
                    std::fprintf(stderr, "strata prefill: FP16 weight cache %lld MiB (the card keeps %lld MiB free; STRATA_PF_W16_CACHE_MIB=0 turns it off)\n",
                                 (long long) (w16_used_ >> 20), (long long) (w16_reserve_ >> 20));
                }
                w16_budget_ = w16_used_ > 0 ? w16_used_ : 0;   // out of room: stop trying
            }
        }
        if (cw != nullptr) {
            f16(X, cw, Y, T, N, K, ldy, beta);
            return;
        }
    }
#endif
    if (N * K > scratch_elems_) {
        // Too large for the scratch at once: in row slices.
        const int64_t rows = scratch_elems_ / K;
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            f16(X, scratch_, Y + r0, T, n, K, ldy, beta);
        }
        // the side stream's next dequant into scratch_ waits for these slices' reads and writes of it
        if (side_scratch_) (void) cudaEventRecord((cudaEvent_t) side_free_[0], (cudaStream_t) stream_);
        return;
    }
    if (side_scratch_) {
        // STRATA_DEQUANT_SIDE: the dequant on the side stream, into the scratch the compute stream read last but one.
        // free[cur] -> dequant -> ready[cur] -> GEMM -> free[cur] orders the dequant after the GEMM that last read
        // this scratch and the GEMM after the dequant that filled it, so every read and write of a scratch is where
        // the serial form had it - the same kernels, the same bits - and the dequant runs while the kernels queued
        // ahead of this GEMM do.
        uint16_t* const s = side_cur_ == 0 ? scratch_ : side_scratch_;
        (void) cudaStreamWaitEvent((cudaStream_t) side_stream_, (cudaEvent_t) side_free_[side_cur_], 0);
        strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, s, side_stream_);
        (void) cudaEventRecord((cudaEvent_t) side_ready_[side_cur_], (cudaStream_t) side_stream_);
        (void) cudaStreamWaitEvent((cudaStream_t) stream_, (cudaEvent_t) side_ready_[side_cur_], 0);
        f16(X, s, Y, T, N, K, ldy, beta);
        (void) cudaEventRecord((cudaEvent_t) side_free_[side_cur_], (cudaStream_t) stream_);
        side_cur_ ^= 1;
        return;
    }
    strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
    f16(X, scratch_, Y, T, N, K, ldy, beta);
}

}  // namespace strata::prefill
