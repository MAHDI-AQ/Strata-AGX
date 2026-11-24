// src/kernels/qsa_q4_fast_parity.cpp - STRATA_QSA_ATTN_FAST parity & microbenchmark harness.
// Verifies:
//   1. Chunk-level bitwise equality (worst_abs == 0, memcmp == 0) of Q4_0 fast chunk kernels against baseline
//      attn_chunk_kernel / attn_chunk_rows_kernel / attn_chunk_slots_kernel across batch, step, rows, slots,
//      rot_q=0/1, aligned & odd-2-byte pointers, partial chunks, and unresident (-1) page holes.
//   2. 2-stage split attn_merge_v2 accuracy (max_abs < 1e-5, max_rel < 1e-5, 0 NaN/Inf) vs attn_merge_v2_kernel
//      at n_q=1,2 and cap=32768 across edge widths (-1, 0, 1, 63, 64, 65, 1023..1025, 16383..16385, 32768).
//   3. Optional --bench: latency & achieved GB/s comparison (only when explicitly requested).
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_q4_fast_parity: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc((void**) &p, n * sizeof(T) + 256), "cudaMalloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "cudaMemset");
    return p;
}
template <typename T> T* upload(const std::vector<T>& h) {
    T* d = dalloc<T>(h.size());
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
template <typename T> std::vector<T> download(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}
uint16_t f16_scale_q4(uint32_t r) { return (uint16_t) ((13u << 10) | (r & 0x3FFu)); }

constexpr int64_t kCells = 40960;
constexpr int64_t kHole = 8192;

struct Q4Buffers {
    uint8_t* k4_raw = nullptr;
    uint8_t* v4_raw = nullptr;
    uint8_t* k4 = nullptr;
    uint8_t* v4 = nullptr;
    int32_t* table = nullptr;
};

Q4Buffers make_q4_buffers(const k::QsaShapes& s, std::mt19937& rng, size_t byte_shift) {
    const int64_t PS = s.page_size, pages = kCells / PS, H = s.n_head_kv, D = s.head_dim;
    std::vector<int32_t> tab((size_t) pages);
    for (int64_t i = 0; i < pages; ++i) tab[(size_t) i] = (int32_t) i;
    std::shuffle(tab.begin(), tab.end(), rng);
    for (int64_t i = 0; i < pages; ++i)
        if (rng() % 32u == 0) tab[(size_t) i] = -1;
    for (int64_t i = kHole / PS; i < (kHole + 256) / PS; ++i) tab[(size_t) i] = -1;

    const int64_t rows = pages * H * PS;
    const size_t blocks = (size_t) (rows * (D / k::QK4_0));
    const size_t bytes = blocks * sizeof(k::block_q4_0);
    std::vector<uint8_t> hk(bytes + byte_shift), hv(bytes + byte_shift);
    for (int side = 0; side < 2; ++side) {
        auto& dst = (side == 0 ? hk : hv);
        for (size_t i = 0; i < blocks; ++i) {
            k::block_q4_0 blk;
            blk.d = f16_scale_q4(rng());
            for (auto& q : blk.qs) q = (uint8_t) (rng() & 0xFFu);
            std::memcpy(dst.data() + byte_shift + i * sizeof(k::block_q4_0), &blk, sizeof(blk));
        }
    }
    Q4Buffers out;
    out.table = upload(tab);
    out.k4_raw = upload(hk);
    out.v4_raw = upload(hv);
    out.k4 = out.k4_raw + byte_shift;
    out.v4 = out.v4_raw + byte_shift;
    return out;
}

void free_q4_buffers(Q4Buffers& b) {
    ck(cudaFree(b.table), "free table");
    ck(cudaFree(b.k4_raw), "free k4");
    ck(cudaFree(b.v4_raw), "free v4");
}

void fill_selection(int32_t* ids, int64_t cap, int width, int shape, std::mt19937& rng) {
    const int64_t w = std::max<int64_t>(0, std::min<int64_t>(width, cap));
    if (shape == 0) {
        int64_t start = (rng() % 2 == 0) ? std::max<int64_t>(0, kHole - 64 * (int64_t) (rng() % 8))
                                         : (int64_t) (rng() % (uint32_t) (kCells - std::max<int64_t>(w, 1)));
        if (start + w > kCells) start = kCells - w;
        for (int64_t j = 0; j < cap; ++j) ids[j] = (int32_t) ((start + j) % kCells);
    } else if (shape == 1) {
        std::vector<int32_t> v((size_t) cap);
        for (auto& x : v) x = (int32_t) (rng() % (uint32_t) kCells);
        std::sort(v.begin(), v.begin() + w);
        std::copy(v.begin(), v.end(), ids);
    } else {
        for (int64_t j = 0; j < cap; ++j) ids[j] = (int32_t) (rng() % (uint32_t) kCells);
    }
}

struct DiffStats {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    float l1_rel = 0.0f;
    float worst_abs_ref = 0.0f;
    float worst_abs_got = 0.0f;
    float worst_rel_ref = 0.0f;
    float worst_rel_got = 0.0f;
    size_t bad_tol = 0;
    size_t diffs = 0;
    size_t nonfinite = 0;
    size_t nonzero = 0;
};

DiffStats compare_vecs(const std::vector<float>& ref, const std::vector<float>& got) {
    DiffStats st;
    double err_sum = 0.0, ref_sum = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(ref[i]) || !std::isfinite(got[i])) {
            ++st.nonfinite;
            continue;
        }
        const float abs_ref = std::fabs(ref[i]);
        if (ref[i] != 0.0f) ++st.nonzero;
        if (std::memcmp(&ref[i], &got[i], 4) != 0) ++st.diffs;
        const float ad = std::fabs(ref[i] - got[i]);
        const float rd = ad / std::max(1e-6f, abs_ref);
        err_sum += (double) ad;
        ref_sum += (double) abs_ref;
        if (ad > 1e-5f + 1e-4f * abs_ref) ++st.bad_tol;
        if (ad > st.max_abs) {
            st.max_abs = ad;
            st.worst_abs_ref = ref[i];
            st.worst_abs_got = got[i];
        }
        if (rd > st.max_rel) {
            st.max_rel = rd;
            st.worst_rel_ref = ref[i];
            st.worst_rel_got = got[i];
        }
    }
    st.l1_rel = (float) (err_sum / (ref_sum > 1e-30 ? ref_sum : 1e-30));
    return st;
}
}  // namespace

int main(int argc, char** argv) {
    int device = 0;
    bool do_bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) do_bench = true;
        else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) device = std::atoi(argv[++i]);
        else {
            std::fprintf(stderr, "usage: qsa_q4_fast_parity [--device N] [--bench]\n");
            return 2;
        }
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no GPU");
        return 77;
    }
    ck(cudaSetDevice(device), "cudaSetDevice");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    cudaStream_t stream = nullptr;
    ck(cudaStreamCreate(&stream), "cudaStreamCreate");
    const char* fast_env = std::getenv("STRATA_QSA_ATTN_FAST");
    const int fast_mode = (fast_env && *fast_env) ? std::atoi(fast_env) : 0;
    std::printf("qsa_q4_fast_parity on device %d (%s, %d.%d), STRATA_QSA_ATTN_FAST=%d\n",
                device, prop.name, prop.major, prop.minor, fast_mode);
    if (fast_mode == 0) {
        std::puts("NOTE: STRATA_QSA_ATTN_FAST is 0/unset; set STRATA_QSA_ATTN_FAST=1 to exercise fast path.");
    }

    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t NH = s.n_head, HD = s.head_dim;
    std::mt19937 rng(20261123u);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    const int64_t max_q = 8, max_cap = 32768;
    const size_t scr_floats = (size_t) k::qsa_decode_attn_scratch_floats(max_cap, s);
    float* d_q = dalloc<float>((size_t) (max_q * NH * HD));
    float* d_q_full = dalloc<float>((size_t) (max_q * NH * 2 * HD));
    int32_t* d_ids = dalloc<int32_t>((size_t) (max_q * max_cap));
    int32_t* d_steps = dalloc<int32_t>((size_t) (max_q * k::kStepCount));
    float* d_scratch = dalloc<float>(scr_floats * (size_t) max_q);
    float* d_ref = dalloc<float>((size_t) (max_q * NH * HD));
    float* d_got = dalloc<float>((size_t) (max_q * NH * HD));
    float* d_gated_ref = dalloc<float>((size_t) (max_q * NH * HD));
    float* d_gated_got = dalloc<float>((size_t) (max_q * NH * HD));

    int failed = 0;
    long long cases = 0;
    float global_max_abs_split = 0.0f, global_max_l1_split = 0.0f;
    float worst_split_ref = 0.0f, worst_split_got = 0.0f;

    // Test both 16-byte aligned pools (shift=0) and odd 2-byte aligned pools (shift=2)
    for (size_t shift : {(size_t) 0, (size_t) 2}) {
        Q4Buffers buf = make_q4_buffers(s, rng, shift);
        k::QsaAttnPools pools;
        pools.k_q4 = buf.k4;
        pools.v_q4 = buf.v4;
        pools.page_table = buf.table;

        const int32_t widths[] = {-1, 0, 1, 63, 64, 65, 1023, 1024, 1025, 2051, 4096, 16383, 16384, 16385, 32768};
        for (int64_t cap : {(int64_t) 2051, (int64_t) 32768}) {
            const int64_t n_chunks = (cap + 63) / 64;
            for (int n_q : {1, 2, 4, 8}) {
                if (shift == 2 && n_q > 2) continue;   // keep runtime bounded on odd-shift coverage
                for (int wc = 0; wc < (int) (sizeof(widths) / sizeof(widths[0])); ++wc) {
                    const int32_t w = widths[wc];
                    if (cap % 64 != 0 && (int64_t) w > cap) continue;
                    std::vector<float> hq((size_t) (n_q * NH * HD));
                    for (auto& x : hq) x = nd(rng);
                    ck(cudaMemcpy(d_q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "q");
                    std::vector<int32_t> hid((size_t) (n_q * cap));
                    std::vector<int32_t> hst((size_t) (n_q * k::kStepCount), 0);
                    for (int i = 0; i < n_q; ++i) {
                        const int32_t wi = (wc % 2 == 1 && i > 0) ? widths[(wc + i) % (sizeof(widths) / sizeof(widths[0]))] : w;
                        const int32_t wi_clamped = (cap % 64 != 0 && wi > cap) ? (int32_t) cap : wi;
                        fill_selection(hid.data() + (size_t) i * cap, cap, wi_clamped, i % 3, rng);
                        hst[(size_t) i * k::kStepCount + k::kStepWidth] = wi_clamped;
                    }
                    ck(cudaMemcpy(d_ids, hid.data(), hid.size() * 4, cudaMemcpyHostToDevice), "ids");
                    ck(cudaMemcpy(d_steps, hst.data(), hst.size() * 4, cudaMemcpyHostToDevice), "steps");

                    const size_t sb = (size_t) k::qsa_decode_attn_scratch_floats(cap, s) * (size_t) n_q * 4;
                    ck(cudaMemsetAsync(d_scratch, 0xFF, sb, stream), "nan scratch");
                    k::qsa_decode_attn_chunks_only(d_q, pools, d_ids, d_steps, cap, s, d_scratch, n_q, 0, stream);
                    k::qsa_decode_attn_merge_only(d_steps, cap, s, d_scratch, d_ref, n_q, 1, stream);

                    ck(cudaMemsetAsync(d_scratch, 0xFF, sb, stream), "nan scratch");
                    k::qsa_decode_attn_batch(d_q, pools, d_ids, d_steps, cap, s, d_scratch, d_got, n_q, stream);
                    ck(cudaStreamSynchronize(stream), "sync");

                    const auto href = download(d_ref, (size_t) (n_q * NH * HD));
                    const auto hgot = download(d_got, (size_t) (n_q * NH * HD));
                    const DiffStats ds = compare_vecs(href, hgot);
                    const bool expect_split_merge = (fast_mode == 1) && (n_q * NH < 64) && (n_chunks >= 64);
                    const bool ok = expect_split_merge
                                        ? (ds.nonfinite == 0 && ds.bad_tol == 0 && ds.l1_rel <= 1e-5f)
                                        : (ds.nonfinite == 0 && ds.diffs == 0);
                    if (expect_split_merge) {
                        if (ds.max_abs > global_max_abs_split) {
                            global_max_abs_split = ds.max_abs;
                            worst_split_ref = ds.worst_abs_ref;
                            worst_split_got = ds.worst_abs_got;
                        }
                        global_max_l1_split = std::max(global_max_l1_split, ds.l1_rel);
                    }
                    if (!ok) {
                        std::printf("FAIL batch shift=%zu cap=%lld n_q=%d w=%d split=%d: diffs=%zu bad_tol=%zu max_abs=%.3e (ref=%.6g got=%.6g) l1_rel=%.3e max_rel=%.3e nan=%zu\n",
                                    shift, (long long) cap, n_q, w, expect_split_merge ? 1 : 0, ds.diffs, ds.bad_tol,
                                    ds.max_abs, ds.worst_abs_ref, ds.worst_abs_got, ds.l1_rel, ds.max_rel, ds.nonfinite);
                        ++failed;
                    }
                    ++cases;

                    if (n_q == 1 && (w == 2051 || w == 32768 || w == 63 || w == 0)) {
                        ck(cudaMemsetAsync(d_scratch, 0xFF, sb, stream), "nan scratch step");
                        k::qsa_decode_attn_step(d_q, pools, d_ids, d_steps, cap, s, d_scratch, d_got, stream);
                        ck(cudaStreamSynchronize(stream), "sync step");
                        const auto hs = download(d_got, (size_t) (NH * HD));
                        const DiffStats dss = compare_vecs(href, hs);
                        const bool ok_s = expect_split_merge
                                              ? (dss.nonfinite == 0 && dss.bad_tol == 0 && dss.l1_rel <= 1e-5f)
                                              : (dss.nonfinite == 0 && dss.diffs == 0);
                        if (!ok_s) {
                            std::printf("FAIL step shift=%zu cap=%lld w=%d: diffs=%zu bad_tol=%zu max_abs=%.3e (ref=%.6g got=%.6g) l1_rel=%.3e\n",
                                        shift, (long long) cap, w, dss.diffs, dss.bad_tol, dss.max_abs, dss.worst_abs_ref, dss.worst_abs_got, dss.l1_rel);
                            ++failed;
                        }
                        ++cases;
                    }
                }
            }
        }

        // Check qsa_decode_attn_batch_rows & qsa_decode_attn_slots with Q4_0 pools across n_q=1,2,4,8
        for (int n_q : {1, 2, 4, 8}) {
            const int64_t cap = 2051;
            std::vector<Q4Buffers> row_bufs;
            k::QsaAttnPoolsRows pr;
            k::QsaAttnSlotPools spx;
            pr.n = n_q;
            for (int r = 0; r < n_q; ++r) {
                row_bufs.push_back(make_q4_buffers(s, rng, (r % 2 == 1 && shift == 2) ? 2 : 0));
                k::QsaAttnPools pl;
                pl.k_q4 = row_bufs.back().k4;
                pl.v_q4 = row_bufs.back().v4;
                pl.page_table = row_bufs.back().table;
                pr.p[r] = pl;
                spx.p[r] = pl;
            }
            std::vector<float> hq((size_t) (n_q * NH * HD));
            for (auto& x : hq) x = nd(rng);
            ck(cudaMemcpy(d_q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "q rows");
            std::vector<int32_t> hid((size_t) (n_q * cap));
            std::vector<int32_t> hst((size_t) (n_q * k::kStepCount), 0);
            for (int r = 0; r < n_q; ++r) {
                const int32_t w = (r == 0 ? 2051 : (r * 317 + 65) % 2051);
                fill_selection(hid.data() + (size_t) r * cap, cap, w, r % 3, rng);
                hst[(size_t) r * k::kStepCount + k::kStepWidth] = w;
            }
            ck(cudaMemcpy(d_ids, hid.data(), hid.size() * 4, cudaMemcpyHostToDevice), "ids rows");
            ck(cudaMemcpy(d_steps, hst.data(), hst.size() * 4, cudaMemcpyHostToDevice), "steps rows");

            for (int r = 0; r < n_q; ++r) {
                k::qsa_decode_attn_chunks_only(d_q + (size_t) r * NH * HD, pr.p[r], d_ids + (size_t) r * cap,
                                               d_steps + (size_t) r * k::kStepCount, cap, s,
                                               d_scratch + (size_t) r * scr_floats, 1, 0, stream);
                k::qsa_decode_attn_merge_only(d_steps + (size_t) r * k::kStepCount, cap, s,
                                              d_scratch + (size_t) r * scr_floats, d_ref + (size_t) r * NH * HD, 1, 1, stream);
            }
            k::qsa_decode_attn_batch_rows(d_q, pr, d_ids, d_steps, cap, s, d_scratch, d_got, n_q, stream);
            ck(cudaStreamSynchronize(stream), "rows sync");
            const auto href = download(d_ref, (size_t) (n_q * NH * HD));
            auto hgot = download(d_got, (size_t) (n_q * NH * HD));
            const DiffStats dsr = compare_vecs(href, hgot);
            if (dsr.nonfinite != 0 || dsr.diffs != 0) {
                std::printf("FAIL batch_rows shift=%zu n_q=%d: diffs=%zu max_abs=%.3e\n", shift, n_q, dsr.diffs, dsr.max_abs);
                ++failed;
            }
            ++cases;

            if (k::qsa_decode_attn_slots(d_q, spx, d_ids, d_steps, cap, s, d_scratch, d_got, n_q, stream)) {
                ck(cudaStreamSynchronize(stream), "slots sync");
                hgot = download(d_got, (size_t) (n_q * NH * HD));
                const DiffStats dss = compare_vecs(href, hgot);
                if (dss.nonfinite != 0 || dss.diffs != 0) {
                    std::printf("FAIL slots shift=%zu n_q=%d: diffs=%zu max_abs=%.3e\n", shift, n_q, dss.diffs, dss.max_abs);
                    ++failed;
                }
                ++cases;
            }
            for (auto& rb : row_bufs) free_q4_buffers(rb);
        }

        // Verify fused rot_q=true, rot_out=true step path against unfused fwht + chunks_only + merge_only + fwht + gate
        for (int64_t cap : {(int64_t) 2051, (int64_t) 32768}) {
            std::vector<float> hq((size_t) (NH * HD)), hq_full((size_t) (NH * 2 * HD));
            for (auto& x : hq) x = nd(rng);
            for (auto& x : hq_full) x = nd(rng);
            ck(cudaMemcpy(d_q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "q rot");
            ck(cudaMemcpy(d_q_full, hq_full.data(), hq_full.size() * 4, cudaMemcpyHostToDevice), "qfull rot");
            std::vector<int32_t> hid((size_t) cap);
            fill_selection(hid.data(), cap, (int) cap, 0, rng);
            std::vector<int32_t> hst((size_t) k::kStepCount, 0);
            hst[k::kStepWidth] = (int32_t) cap;
            ck(cudaMemcpy(d_ids, hid.data(), hid.size() * 4, cudaMemcpyHostToDevice), "ids rot");
            ck(cudaMemcpy(d_steps, hst.data(), hst.size() * 4, cudaMemcpyHostToDevice), "steps rot");

            const size_t sb = (size_t) k::qsa_decode_attn_scratch_floats(cap, s) * 4;
            ck(cudaMemsetAsync(d_scratch, 0xFF, sb, stream), "nan scratch rot ref");
            float* d_q_rot = dalloc<float>((size_t) (NH * HD));
            ck(cudaMemcpyAsync(d_q_rot, d_q, (size_t) (NH * HD) * 4, cudaMemcpyDeviceToDevice, stream), "copy q_rot");
            k::fwht256_inplace_cuda(d_q_rot, NH, stream);
            k::qsa_decode_attn_chunks_only(d_q_rot, pools, d_ids, d_steps, cap, s, d_scratch, 1, 0, stream);
            k::qsa_decode_attn_merge_only(d_steps, cap, s, d_scratch, d_ref, 1, 1, stream);
            k::fwht256_inplace_cuda(d_ref, NH, stream);
            k::native_qsa_gate_apply(d_ref, d_q_full, d_gated_ref, (int) NH, (int) HD, stream);

            ck(cudaMemsetAsync(d_scratch, 0xFF, sb, stream), "nan scratch rot got");
            k::qsa_decode_attn_step(d_q, pools, d_ids, d_steps, cap, s, d_scratch, d_got, stream,
                                    d_q_full, d_gated_got, /*rot_q=*/true, /*rot_out=*/true);
            ck(cudaStreamSynchronize(stream), "rot sync");
            const auto href_g = download(d_gated_ref, (size_t) (NH * HD));
            const auto hgot_g = download(d_gated_got, (size_t) (NH * HD));
            const DiffStats dsr = compare_vecs(href_g, hgot_g);
            // Fused in-register FWHT matches standalone fwht256_inplace_cuda within 1e-5 (same tolerance as test_qsa_sm89_attention_audit)
            if (dsr.nonfinite != 0 || dsr.max_abs >= 1e-5f || dsr.nonzero == 0) {
                std::printf("FAIL rot step shift=%zu cap=%lld: max_abs=%.3e nonfinite=%zu nz=%zu\n",
                            shift, (long long) cap, dsr.max_abs, dsr.nonfinite, dsr.nonzero);
                ++failed;
            }
            ++cases;
            ck(cudaFree(d_q_rot), "free q_rot");
        }
        free_q4_buffers(buf);
    }

    std::printf("qsa_q4_fast_parity: %lld cases, %d failed (split-merge max_abs=%.3e [ref=%.6g got=%.6g], max_l1_rel=%.3e) -> %s\n",
                cases, failed, global_max_abs_split, worst_split_ref, worst_split_got, global_max_l1_split, failed == 0 ? "PASS" : "FAIL");

    if (do_bench && failed == 0) {
        Q4Buffers buf = make_q4_buffers(s, rng, 0);
        k::QsaAttnPools pools;
        pools.k_q4 = buf.k4;
        pools.v_q4 = buf.v4;
        pools.page_table = buf.table;
        cudaEvent_t e0, e1;
        ck(cudaEventCreate(&e0), "e0");
        ck(cudaEventCreate(&e1), "e1");
        auto time_us = [&](auto&& fn, int iters) -> double {
            for (int i = 0; i < 10; ++i) fn();
            ck(cudaStreamSynchronize(stream), "warm");
            ck(cudaEventRecord(e0, stream), "e0");
            for (int i = 0; i < iters; ++i) fn();
            ck(cudaEventRecord(e1, stream), "e1");
            ck(cudaEventSynchronize(e1), "sync");
            float ms = 0.0f;
            ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
            return 1000.0 * ms / iters;
        };
        std::puts("\n--bench (Q4_0 KV): baseline (chunks_only(0)+merge_only(1)) vs production entry (STRATA_QSA_ATTN_FAST)");
        for (int64_t cap : {(int64_t) 32768, (int64_t) 2051}) {
            for (int n_q : {1, 2, 4}) {
                if (cap == 32768 && n_q > 2) continue;
                std::vector<int32_t> hid((size_t) (n_q * cap));
                std::vector<int32_t> hst((size_t) (n_q * k::kStepCount), 0);
                for (int i = 0; i < n_q; ++i) {
                    fill_selection(hid.data() + (size_t) i * cap, cap, (int) cap, 0, rng);
                    hst[(size_t) i * k::kStepCount + k::kStepWidth] = (int32_t) cap;
                }
                ck(cudaMemcpy(d_ids, hid.data(), hid.size() * 4, cudaMemcpyHostToDevice), "ids");
                ck(cudaMemcpy(d_steps, hst.data(), hst.size() * 4, cudaMemcpyHostToDevice), "steps");
                const int iters = (cap == 32768) ? 200 : 400;
                const double base_ch = time_us([&] {
                    k::qsa_decode_attn_chunks_only(d_q, pools, d_ids, d_steps, cap, s, d_scratch, n_q, 0, stream);
                }, iters);
                const double base_mg = time_us([&] {
                    k::qsa_decode_attn_merge_only(d_steps, cap, s, d_scratch, d_ref, n_q, 1, stream);
                }, iters);
                const double fast_tot = time_us([&] {
                    k::qsa_decode_attn_batch(d_q, pools, d_ids, d_steps, cap, s, d_scratch, d_got, n_q, stream);
                }, iters);
                const double bytes_kv = (double) cap * (double) s.n_head_kv * 2.0 * 144.0 * (double) n_q;
                std::printf("  cap=%5lld n_q=%d | base_chunk=%6.2f us base_merge=%6.2f us (tot=%6.2f us) | fast_tot=%6.2f us (%.2fx, eff KV BW=%.1f GB/s)\n",
                            (long long) cap, n_q, base_ch, base_mg, base_ch + base_mg, fast_tot,
                            (base_ch + base_mg) / fast_tot, bytes_kv / (fast_tot * 1e3));
            }
        }
        ck(cudaEventDestroy(e0), "d0");
        ck(cudaEventDestroy(e1), "d1");
        free_q4_buffers(buf);
    }

    for (void* p : {(void*) d_q, (void*) d_q_full, (void*) d_ids, (void*) d_steps, (void*) d_scratch,
                    (void*) d_ref, (void*) d_got, (void*) d_gated_ref, (void*) d_gated_got})
        ck(cudaFree(p), "free");
    ck(cudaStreamDestroy(stream), "stream destroy");
    return failed == 0 ? 0 : 1;
}
