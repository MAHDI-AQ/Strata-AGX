// src/kernels/verify_batch_parity.cpp - the verify window's one-launch forms (the hyper-connection write, the PLE post-ops) against the per-token launches they
// replace, bitwise.
//
//     build/verify_batch_parity
//
// 1. native_gr_post_multi against n native_gr_post calls (the hyper-connection write before the PLE block).
// 2. native_ple_postops_batch_snap (STRATA_DF_PLE) against the per-token loop of the verify window: per token
//    native_ple_postops, ple_history_advance and a copy of the history (the commit's snapshot); the results, the final
//    history and every snapshot, for windows of 2..8 tokens.
//    Skipped where the engine switches the batched PLE post-ops off itself (an engine built with CUDA 12.8 on a compute
//    capability 12.0 card: sm120_series.hpp ple_batch_resolve, the rule the prompt path uses; STRATA_PLE_BATCH=1 runs it anyway).
// GPU, synthetic, no model.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/native_gr_postops.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/core/sm120_series.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_fail = 0;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

// The set-up copies and fills run on the legacy default stream, which does NOT order work on the non-blocking stream
// the kernels run on (and cudaMemset / a pageable cudaMemcpy may return before the device has finished them): each
// one is completed here, or a kernel could read its inputs - or have its outputs overwritten - while they land.
template <typename T>
T* dalloc(size_t n, int fill = 0) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, fill, n * sizeof(T) + 64), "memset");
    ck(cudaDeviceSynchronize(), "memset sync");
    return p;
}

template <typename T>
std::vector<T> read(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "read");
    return h;
}

template <typename T>
T* upload(const std::vector<T>& h) {
    T* d = dalloc<T>(h.size());
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    ck(cudaDeviceSynchronize(), "upload sync");
    return d;
}

void report(const char* what, bool ok) {
    std::printf("%-64s %s\n", what, ok ? "bitwise equal" : "DIFFERENT");
    if (!ok) ++g_fail;
}

// ---- 1. the hyper-connection write
void gr_post(cudaStream_t s) {
    const int N = 2560, HC = 4;
    std::mt19937 rng(12);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int n : {2, 4, 8}) {
        std::vector<float> r((size_t) n * HC * N), bo((size_t) n * N), inj((size_t) n * HC);
        for (auto& x : r) x = nd(rng);
        for (auto& x : bo) x = nd(rng);
        for (auto& x : inj) x = 3.f * nd(rng);
        float* r0 = upload(r);
        float* r1 = upload(r);
        float* d_bo = upload(bo);
        float* d_inj = upload(inj);
        for (int t = 0; t < n; ++t)
            k::native_gr_post(r0 + (size_t) t * HC * N, d_bo + (size_t) t * N, d_inj + (size_t) t * HC,
                              r0 + (size_t) t * HC * N, N, HC, s);
        k::native_gr_post_multi(r1, d_bo, d_inj, r1, N, HC, n, (long long) HC * N, N, HC, s);
        ck(cudaStreamSynchronize(s), "gr post");
        char what[128];
        std::snprintf(what, sizeof what, "native_gr_post_multi, %d tokens (in place)", n);
        report(what, read(r0, r.size()) == read(r1, r.size()));
        cudaFree(r0); cudaFree(r1); cudaFree(d_bo); cudaFree(d_inj);
    }
}

// ---- 2. the PLE post-ops of a window, with the commit's history snapshots
void ple(cudaStream_t s) {
    constexpr int N = k::NG_N_EMBD, D = k::NG_HC_DIM, HS = k::NG_HIST * k::NG_HC_DIM;
    std::mt19937 rng(13);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> nk(D), nq(D), nc(D);
    for (auto* v : {&nk, &nq, &nc})
        for (auto& x : *v) x = 1.f + 0.1f * nd(rng);
    std::vector<uint16_t> conv((size_t) 4 * D);
    for (auto& x : conv) x = k::f16_from_f32(0.3f * nd(rng));
    k::PleWeights w;
    w.norm_key = upload(nk);
    w.norm_query = upload(nq);
    w.norm_conv = upload(nc);
    w.conv1d_f16 = upload(conv);
    std::vector<float> hist0((size_t) HS);
    for (auto& x : hist0) x = nd(rng);
    for (int n : {2, 3, 4, 8}) {
        std::vector<float> key((size_t) n * D), hid((size_t) n * D), val((size_t) n * N);
        for (auto& x : key) x = 4.f * nd(rng);
        for (auto& x : hid) x = nd(rng);
        for (auto& x : val) x = nd(rng);
        // per token, as the verify window's loop: native_ple_postops (ple_block_projected's buffers), the history
        // advance and the snapshot copy
        float* key0 = upload(key);
        float* hid0 = upload(hid);
        float* val0 = upload(val);
        float* hist_a = upload(hist0);
        float* snap_a = dalloc<float>((size_t) n * HS);
        float *q = dalloc<float>(D), *nrm = dalloc<float>(D), *g = dalloc<float>(4), *gd = dalloc<float>(D);
        float* cv = dalloc<float>(D);
        for (int t = 0; t < n; ++t) {
            k::NativePlePostopsBuffers b{q, nrm, g, gd, nrm, cv, hid0 + (size_t) t * D};
            k::native_ple_postops(key0 + (size_t) t * D, hid0 + (size_t) t * D, val0 + (size_t) t * N, hist_a, w, b, s);
            k::ple_history_advance(hist_a, nrm, s);
            ck(cudaMemcpyAsync(snap_a + (size_t) t * HS, hist_a, (size_t) HS * 4, cudaMemcpyDeviceToDevice, s), "snap");
        }
        // the window's one pass
        float* key1 = upload(key);
        float* hid1 = upload(hid);
        float* val1 = upload(val);
        float* hist_b = upload(hist0);
        float* snap_b = dalloc<float>((size_t) n * HS);
        float *qn = dalloc<float>((size_t) n * D), *gated = dalloc<float>((size_t) n * D), *gate = dalloc<float>((size_t) n * 4);
        k::native_ple_postops_batch_snap(key1, hid1, val1, hist_b, w, qn, gated, gate, n, snap_b, s);
        ck(cudaStreamSynchronize(s), "ple");
        char what[128];
        std::snprintf(what, sizeof what, "native_ple_postops_batch_snap, %d tokens: results", n);
        report(what, read(hid0, hid.size()) == read(hid1, hid.size()));
        std::snprintf(what, sizeof what, "native_ple_postops_batch_snap, %d tokens: history", n);
        report(what, read(hist_a, (size_t) HS) == read(hist_b, (size_t) HS));
        std::snprintf(what, sizeof what, "native_ple_postops_batch_snap, %d tokens: snapshots", n);
        report(what, read(snap_a, (size_t) n * HS) == read(snap_b, (size_t) n * HS));
        for (float* p : {key0, hid0, val0, hist_a, snap_a, q, nrm, g, gd, cv, key1, hid1, val1, hist_b, snap_b, qn, gated,
                         gate})
            cudaFree(p);
    }
}

void handoff_u64_pingpong(cudaStream_t s) {
    constexpr int na = 64, nb = 16, nc = 4, N = na + nb + nc;
    float *h0 = nullptr, *d0 = nullptr, *h1 = nullptr, *d1 = nullptr;
    uint64_t *hflag = nullptr, *dflag = nullptr, *hpseq = nullptr, *dpseq = nullptr, *hwseq = nullptr, *dwseq = nullptr;
    uint32_t *herr_h = nullptr, *herr_d = nullptr, *dcnt = nullptr;
    ck(cudaHostAlloc((void**) &h0, N * sizeof(float), cudaHostAllocMapped | cudaHostAllocPortable), "h0");
    ck(cudaHostAlloc((void**) &h1, N * sizeof(float), cudaHostAllocMapped | cudaHostAllocPortable), "h1");
    ck(cudaHostGetDevicePointer((void**) &d0, h0, 0), "d0");
    ck(cudaHostGetDevicePointer((void**) &d1, h1, 0), "d1");
    ck(cudaHostAlloc((void**) &hflag, 64, cudaHostAllocMapped | cudaHostAllocPortable), "hflag");
    ck(cudaHostAlloc((void**) &hpseq, 64, cudaHostAllocMapped | cudaHostAllocPortable), "hpseq");
    ck(cudaHostAlloc((void**) &hwseq, 64, cudaHostAllocMapped | cudaHostAllocPortable), "hwseq");
    ck(cudaHostAlloc((void**) &herr_h, 64, cudaHostAllocMapped | cudaHostAllocPortable), "herr");
    ck(cudaHostGetDevicePointer((void**) &dflag, hflag, 0), "dflag");
    ck(cudaHostGetDevicePointer((void**) &dpseq, hpseq, 0), "dpseq");
    ck(cudaHostGetDevicePointer((void**) &dwseq, hwseq, 0), "dwseq");
    ck(cudaHostGetDevicePointer((void**) &herr_d, herr_h, 0), "derr");
    *hflag = 0; *hpseq = 0; *hwseq = 0; *herr_h = 0;
    ck(cudaMalloc((void**) &dcnt, 64), "dcnt");
    ck(cudaMemset(dcnt, 0, 64), "dcnt0");
    float *sa = dalloc<float>(na), *sb = dalloc<float>(nb), *sc = dalloc<float>(nc);
    float *da = dalloc<float>(na), *db = dalloc<float>(nb), *dc = dalloc<float>(nc);
    bool all_ok = true;
    for (uint64_t seq = 1; seq <= 6; ++seq) {
        std::vector<float> va(na), vb(nb), vc(nc);
        for (int i = 0; i < na; ++i) va[i] = float(seq * 100 + i);
        for (int i = 0; i < nb; ++i) vb[i] = float(seq * 200 + i);
        for (int i = 0; i < nc; ++i) vc[i] = float(seq * 300 + i);
        ck(cudaMemcpy(sa, va.data(), na * sizeof(float), cudaMemcpyHostToDevice), "up a");
        ck(cudaMemcpy(sb, vb.data(), nb * sizeof(float), cudaMemcpyHostToDevice), "up b");
        ck(cudaMemcpy(sc, vc.data(), nc * sizeof(float), cudaMemcpyHostToDevice), "up c");
        *hpseq = seq;
        *hwseq = seq;
        k::handoff_publish_u64(d0, d1, 0, sa, na, sb, nb, sc, nc, dcnt, dflag, dpseq, s, nullptr);
        k::wait_handoff_u64(dflag, dwseq, herr_d, 5000000000ull, s);
        k::handoff_consume_u64(da, na, db, nb, dc, nc, d0, d1, 0, dwseq, s);
        ck(cudaStreamSynchronize(s), "sync");
        if (*herr_h != 0 || *hflag != seq || read(da, na) != va || read(db, nb) != vb || read(dc, nc) != vc) {
            all_ok = false;
            break;
        }
    }
    report("handoff_u64_pingpong monotonic seq=1..6", all_ok);
    *hwseq = 9999ull;
    *herr_h = 0;
    *hflag |= k::kHandoffAbortBit;
    k::wait_handoff_u64(dflag, dwseq, herr_d, 5000000000ull, s);
    ck(cudaStreamSynchronize(s), "abort sync");
    report("handoff_u64_pingpong host abort bit releases without timeout", *herr_h == 0);
    for (float* p : {sa, sb, sc, da, db, dc}) cudaFree(p);
    cudaFree(dcnt);
    for (void* p : {(void*) h0, (void*) h1, (void*) hflag, (void*) hpseq, (void*) hwseq, (void*) herr_h})
        cudaFreeHost(p);
}

}  // namespace

int main() {
    cudaDeviceProp prop{};
    int dev = 0;
    ck(cudaGetDevice(&dev), "device");
    ck(cudaGetDeviceProperties(&prop, dev), "props");
    std::printf("verify_batch_parity: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    cudaStream_t s = nullptr;
    ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    gr_post(s);
    // the engine's own rule for the batched PLE post-ops (prefill.cpp ple_batch_enabled, same toolkit arguments): off for a CUDA 12.8 build on a 12.0 card unless STRATA_PLE_BATCH says otherwise
#if defined(CUDART_VERSION) && !defined(STRATA_USE_HIP) && !defined(__HIPCC__)
    constexpr int kRt = CUDART_VERSION;
    constexpr bool kHip = false;
#else
    constexpr int kRt = 0;      // a HIP build has no CUDA toolkit version: the engine passes 0 and hip = true
    constexpr bool kHip = true;
#endif
    if (strata::ple_batch_resolve(std::getenv("STRATA_PLE_BATCH"), kRt, kHip, prop.major, prop.minor))
        ple(s);
    else
        std::printf("SKIP native_ple_postops_batch_snap: built with CUDA 12.8 for RTX 50: the batched PLE post-ops are switched off in this build "
                    "(STRATA_PLE_BATCH=1 runs the step anyway)\n");
    handoff_u64_pingpong(s);
    std::printf("verify_batch_parity: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
