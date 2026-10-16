// src/core/verify.cpp - see include/strata/core/verify.hpp.
#include "strata/core/verify.hpp"
#if defined(_WIN32)
#include <intrin.h>
#endif
#include "strata/core/native_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/spec/tree_buffer.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/core/progress.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include <algorithm>
#include <atomic>
#include <map>
#include <string>
#include <thread>
#include <vector>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <immintrin.h>
namespace strata::core {
namespace {
constexpr float EPS = 1e-6f;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
const bool g_dbg = std::getenv("STRATA_VERIFY_DEBUG") != nullptr;
inline bool g_mmvq_il() { static const bool on = [] { const char* v = std::getenv("STRATA_MMVQ_IL"); return v == nullptr || v[0] != '0'; }(); return on; }
#define VDBG(...) do { if (g_dbg) { std::fprintf(stderr, "verify dbg: " __VA_ARGS__); std::fflush(stderr); } } while (0)
struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T> T* take(uint64_t n) {
        T* p = base ? (T*) (base + used) : nullptr;
        used += (n * sizeof(T) + 255) & ~255ull;
        return p;
    }
};
bool mapped(size_t bytes, void** h, void** d) {
    if (cudaHostAlloc(h, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) return false;
    std::memset(*h, 0, bytes);
    return cudaHostGetDevicePointer(d, *h, 0) == cudaSuccess;
}
strata::kernels::QsaShapes shapes_of(const ModelGeometry& g) {
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head;
    s.n_head_kv = g.n_head_kv;
    s.head_dim = g.head_dim;
    s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    return s;
}
const WeightRef* need(const LayerView& v, const char* suffix, std::string& err) {
    const WeightRef* r = v.get(suffix);
    if (r == nullptr && err.empty()) err = v.name(suffix) + " is missing";
    return r;
}
bool native_of(const WeightRef* w, const std::string& name, std::string& err) {
    if (w == nullptr) return false;
    if (w->native_data == nullptr) {
        err = "verify: " + name + " is not served natively (run with --native)";
        return false;
    }
    return true;
}
}  // namespace
namespace {
std::atomic<const Verifier*> g_diag_verifier{nullptr};
void diag_active_verifier(std::FILE* f) {
    if (const Verifier* v = g_diag_verifier.load()) v->diag(f);
}
// #267: every live verifier (a layer split has one per stage), for the release before the engine ends
constexpr int kLiveMax = 16;
std::atomic<Verifier*> g_live[kLiveMax];
void release_live_verifiers(std::FILE* f) {
    for (auto& slot : g_live)
        if (Verifier* v = slot.load()) {
            const Clock::time_point t0 = Clock::now();
            const bool done = v->release_gpu_waits(5000);
            if (f != nullptr)
                std::fprintf(f, "strata: released the verify window's GPU waits (#267): the GPU %s\n",
                             done ? ("finished in " + std::to_string((long long) ms_since(t0)) + " ms").c_str()
                                  : "did not finish within 5 s");
        }
    if (f != nullptr) std::fflush(f);
}
std::string released_note(bool drained) {
    return drained ? "; its GPU waits were released and the GPU finished (#267)"
                   : "; its GPU waits were released but the GPU did not finish within 5 s (#267)";
}
// #267 test hook: STRATA_TEST_VERIFY_STALL=N withholds the last layer's flag in the N-th window (1-based), so the
// GPU spins on a flag nobody raises - the bounded window wait and the release are then what ends it.  Unset: never.
const int64_t g_test_stall = [] {
    const char* e = std::getenv("STRATA_TEST_VERIFY_STALL");
    return e != nullptr ? (int64_t) std::atoll(e) : (int64_t) 0;
}();
}  // namespace
bool Verifier::release_gpu_waits(int timeout_ms) {
    released_.store(true);
    // the words the spin kernels read (wait_flag_ge, wait_flag_ge_or) are mapped host memory, so a store here
    // reaches them with no API call; UINT32_MAX is past every ring.  (E-6's skip words are device memory, but
    // wait_flag_ge_or also returns on its flag.)  A host function raising flag B later only raises.
    for (uint32_t* p : {h_flag_, h_flagA_, h_flagB_})
        if (p != nullptr) *(volatile uint32_t*) p = UINT32_MAX;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    _mm_sfence();
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    for (cudaStream_t s : {cs_, copy_}) {
        if (s == nullptr) continue;
        while (cudaStreamQuery(s) == cudaErrorNotReady) {
            if (ms_since(t0) > timeout_ms) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    return true;
}
void Verifier::diag(std::FILE* f) const {
    auto rd = [](const uint32_t* p) { return p ? *(const volatile uint32_t*) p : 0u; };
    std::fprintf(f, "  verify window: %d tokens at position %lld, host at layer step %u; the GPU rang %u; flags: "
                    "served %u, plan (A) %u, copies (B) %u\n", last_t_, (long long) last_pos0_, cur_layer_ + 1,
                 rd(h_seq_), rd(h_flag_), rd(h_flagA_), rd(h_flagB_));
}
Verifier::~Verifier() {
    const Verifier* self = this;
    g_diag_verifier.compare_exchange_strong(self, nullptr);
    for (auto& slot : g_live) {
        Verifier* me = this;
        slot.compare_exchange_strong(me, nullptr);
    }
    if (cs_) cudaStreamSynchronize(cs_);
    for (auto& b : batch_graphs_) if (b.graph) cudaGraphExecDestroy(b.graph);
    if (batch_fork_) cudaEventDestroy(batch_fork_);
    for (auto e : batch_join_) if (e) cudaEventDestroy(e);
    for (auto& row : exec_)
        for (auto& e : row)
            if (e) cudaGraphExecDestroy(e);
    if (commit_exec_) cudaGraphExecDestroy(commit_exec_);
    if (cs_) cudaStreamDestroy(cs_);
    if (copy_) { cudaStreamSynchronize(copy_); cudaStreamDestroy(copy_); }
    if (commit_done_) cudaEventDestroy(commit_done_);
    if (arena_) cudaFree(arena_);
    void* hosts[] = {h_tok_, h_step_, h_pos_, h_commit_, h_ple_, h_out_, h_x_, h_ids_, h_w_, h_seq_, h_flag_, h_ymiss_,
                     h_flagA_, h_plan_, h_flagB_};
    for (void* h : hosts)
        if (h) cudaFreeHost(h);
    if (skip_) cudaFree(skip_);
    if (slot_off_d_) cudaFree(slot_off_d_);
}
bool Verifier::init(const WeightTable& wt, const ModelGeometry& g, SessionState& ss, const VerifyHits& hits,
                    const NativeHead* head, int max_t, std::string& err, bool batch_workspace) {
    g_diag_verifier.store(this);
    diag_verify_fn().store(&diag_active_verifier);
    for (auto& slot : g_live) {
        Verifier* none = nullptr;
        if (slot.load() == this || slot.compare_exchange_strong(none, this)) break;
    }
    release_gpu_fn().store(&release_live_verifiers);
    cudaGetDevice(&device_);   // a layer split's stage on another GPU: its streams, graphs and buffers live there
    wt_ = &wt;
    g_ = &g;
    ss_ = &ss;
    hits_ = hits;
    head_ = head;
    max_t_ = max_t;
    sampling_.greedy = true;      // a fresh verifier samples greedily until set_sampling says otherwise
    sampling_.temperature = 0.0f;
    if (max_t < 2 || max_t > (batch_workspace ? strata::kernels::cpu::MAXT : strata::kernels::kVerifyMaxT) ||
        max_t > strata::kernels::cpu::MAXT) {
        err = "verify: the window must hold 2.." +
              std::to_string(batch_workspace ? strata::kernels::cpu::MAXT : strata::kernels::kVerifyMaxT) + " tokens";
        return false;
    }
    if (hits.d_res == nullptr || hits.cache_base == nullptr || hits.blob <= 0) {
        err = "verify: needs the profile-filled VRAM expert tier (--expert-profile and --expert-cache); with "
              "--expert-cache auto, no VRAM was left for it: lower --max-context, use --kv k8v4 or images on the CPU";
        return false;
    }
    std::string why;
    if (!layer_verify_compatible(why)) {
        err = "verify: " + why + " (the verify window reproduces the default native decode path)";
        return false;
    }
    if (!strata::kernels::fused_gr_supported(g.n_embd, g.hc, g.hc_lr) || ss.k != 10 || g.ssm_state_size != 128 ||
        g.ssm_d_conv != 4) {
        err = "verify: geometry differs from the artifact's";
        return false;
    }
    if (le_ < 0) le_ = g.n_layers;
    if (lb_ < 0 || lb_ >= le_ || le_ > g.n_layers || (lb_ > 0 && hand_in_ == nullptr) ||
        (le_ < g.n_layers && hand_out_ == nullptr)) {
        err = "verify: the stage's layer range or its hand-off buffers are wrong";
        return false;
    }
    const WeightRef* wo = wt.find("output.weight");
    if (wo == nullptr) { err = "verify: output.weight is missing"; return false; }
    n_vocab_ = wo->ne1;
    const strata::kernels::QsaShapes s = shapes_of(g);
    cap_ = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    max_blocks_ = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    attn_scratch_floats_ = (int64_t) strata::kernels::qsa_decode_attn_scratch_floats(cap_, s);
    const uint64_t T = (uint64_t) max_t, N = (uint64_t) g.n_embd, HC = (uint64_t) g.hc, K = (uint64_t) ss.k;
    const uint64_t C = (uint64_t) g.ssm_conv_channels, ZV = (uint64_t) g.ssm_value_dim, HV = (uint64_t) g.ssm_v_heads;
    const uint64_t NH = (uint64_t) g.n_head, HD = (uint64_t) g.head_dim, NKV = (uint64_t) g.n_head_kv;
    const uint64_t IQ = (uint64_t) g.idx_q_heads, ID = (uint64_t) g.idx_key_dim;
    const uint64_t nG = (uint64_t) g.n_gdn_layers(), nQ = (uint64_t) g.n_qsa_layers();
    const uint64_t HS = (uint64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM;
    const uint64_t TS = (uint64_t) (s.idx_block - 1) * ID;
    const int max_in = (int) std::max<uint64_t>(std::max<uint64_t>(N, ZV), NH * HD);
    // ---- mapped staging
    bool ok = mapped(T * 4, (void**) &h_tok_, (void**) &m_tok_) &&
              mapped(T * strata::kernels::kStepCount * 4, (void**) &h_step_, (void**) &m_step_) &&
              mapped(T * (NH + NKV + IQ) * 4, (void**) &h_pos_, (void**) &m_pos_) &&
              mapped((2 + T) * 4 + 16, (void**) &h_commit_, (void**) &m_commit_) &&
              mapped(T * N * 4, (void**) &h_ple_, (void**) &m_ple_) &&
              mapped(T * 4 + 16, (void**) &h_out_, (void**) &m_out_) &&
              mapped(T * N * 4, (void**) &h_x_, (void**) &m_x_) &&
              mapped(T * K * 4, (void**) &h_ids_, (void**) &m_ids_) &&
              mapped(T * K * 4, (void**) &h_w_, (void**) &m_w_) &&
              mapped(64, (void**) &h_seq_, (void**) &m_seq_) &&
              mapped(64, (void**) &h_flag_, (void**) &m_flag_) &&
              mapped(64, (void**) &h_flagA_, (void**) &m_flagA_) &&
              mapped(64, (void**) &h_flagB_, (void**) &m_flagB_) &&
              mapped(T * K * N * 4, (void**) &h_ymiss_, (void**) &m_ymiss_);
    if (!ok) { err = "verify: mapped staging allocation failed"; return false; }
    // the GPU plan: counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad | ptr(cap u64) | ptr2(cap u64) | start2(cap+1)
    {
        const int64_t cap = (int64_t) (T * K);
        const int64_t i32 = 4 + (cap + 1) + cap + cap;
        const int64_t ptr_off = (i32 + 1) & ~1ll;
        plan_i32_ = ptr_off + 4 * cap + (cap + 1) + 1;
        if (!mapped((size_t) plan_i32_ * 4 * 2 + 64, (void**) &h_plan_, (void**) &m_plan_)) {
            err = "verify: mapped plan allocation failed";
            return false;
        }
        sink_.counts = h_plan_;
        sink_.start = h_plan_ + 4;
        sink_.dst = sink_.start + cap + 1;
        sink_.tok = sink_.dst + cap;
        sink_.ptr = (unsigned long long*) (h_plan_ + ptr_off);
        sink_.ptr2 = sink_.ptr + cap;
        sink_.start2 = h_plan_ + ptr_off + 4 * cap;
        sink_.cap = cap;
        sink_.publish = &Verifier::publish_plan;
        sink_.fetch = &Verifier::fetch_dma;
        sink_.ctx = this;
    }
    // ---- the device arena: the same sequence counted, then carved
    auto carve = [&](Bump& b) {
        tok_ = b.take<int32_t>(T); step_ = b.take<int32_t>(T * strata::kernels::kStepCount);
        pos_ = b.take<int32_t>(T * (NH + NKV + IQ)); commit_ = b.take<int32_t>(2 + T);
        ple_ = b.take<float>(T * N); emb_ = b.take<float>(T * N); R_ = b.take<float>(T * HC * N);
        mixed_ = b.take<float>(T * N); bo_ = b.take<float>(T * N);
        inj_ = b.take<float>(T * HC); inj2_ = b.take<float>(T * HC);
        lo_ = b.take<float>(T * (uint64_t) g.hc_lr); rs_ = b.take<float>(T * HC); xn_ = b.take<float>(T * HC * N);
        xq_ = b.take<uint8_t>(T * strata::kernels::native_q8_1_bytes(max_in, 1));
        xil_ = b.take<uint8_t>(strata::kernels::native_q8_1_il_bytes(max_in, 4));
        qkv_L_ = b.take<float>(nG * T * C); h_L_ = b.take<float>(nG * T * C);
        gate_L_ = b.take<float>(nG * T * HV); beta_L_ = b.take<float>(nG * T * HV);
        z_ = b.take<float>(T * ZV); y_ = b.take<float>(T * ZV); y_dummy_ = b.take<float>(T * ZV);
        qfull_ = b.take<float>(T * NH * 2 * HD); qcur_ = b.take<float>(T * NH * HD);
        kcur_ = b.take<float>(T * NKV * HD); vcur_ = b.take<float>(T * NKV * HD);
        idx_raw_L_ = b.take<float>(nQ * T * ID); qidx_ = b.take<float>(T * IQ * ID);
        scores_ = b.take<float>(T * (uint64_t) max_blocks_); sel_ = b.take<int32_t>(T * (uint64_t) cap_);
        attn_ = b.take<float>(T * NH * HD); attn32_ = b.take<float>(T * NH * HD);
        attn_scratch_ = b.take<float>(T * (uint64_t) attn_scratch_floats_);
        tail_snap_ = b.take<float>(nQ * TS);
        logits_ = b.take<float>(T * (uint64_t) g.n_expert); w_ = b.take<float>(T * K); ids_ = b.take<int32_t>(T * K);
        shared_ = b.take<float>(T * N); parts_ = b.take<float>(T * K * N); hit_out_ = b.take<float>(T * K * N);
        hit_slot_ = b.take<int32_t>(T * K); hit_dst_ = b.take<int32_t>(T * K); hit_count_ = b.take<int32_t>(4);
        plan_ = b.take<int32_t>(2 * ((uint64_t) plan_i32_ + 16));
        staging_ = b.take<uint8_t>((uint64_t) kStagingBlobs * strata::kernels::cpu::expert_layout().max_blob);
        hit_xq_ = b.take<uint8_t>(T * (N / 32) * 34); hit_xs_ = b.take<float>(T * (N / 32));
        nat_xq_ = b.take<uint8_t>(T * (N / 32) * 36);
        hit_scratch_ = b.take<uint8_t>(std::max<uint64_t>(
            strata::kernels::moe_hit_grouped_scratch_bytes((int64_t) (T * K), g.n_embd, g.n_ff),
            strata::kernels::native_expert_scratch_bytes((int64_t) (T * K), g.n_ff)));
        head_mixed_ = b.take<float>(T * N); head_inj_ = b.take<float>(HC);
        sh_bf16_ = b.take<uint16_t>(T * N); sh_gate_ = b.take<float>(T * (uint64_t) g.n_ff);
        sh_up_ = b.take<float>(T * (uint64_t) g.n_ff); sh_g_ = b.take<float>(T + 4);
        head_logits_ = b.take<float>(T * (uint64_t) n_vocab_);
        arg_scratch_ = b.take<uint8_t>(strata::kernels::argmax_rows_scratch_bytes((int) T));
        hist_snap_ = b.take<float>(T * HS);
    };
    Bump count;
    carve(count);
    if (cudaMalloc(&arena_, count.used) != cudaSuccess) {
        err = "verify: the device arena (" + std::to_string(count.used >> 20) + " MiB) does not fit";
        return false;
    }
    cudaMemset(arena_, 0, count.used);
    prof_on_ = std::getenv("STRATA_VERIFY_PROFILE") != nullptr;
    if (prof_on_) {
        const size_t np = (size_t) g.n_layers * kProfPer + 4;
        if (cudaMalloc((void**) &prof_, np * 8) != cudaSuccess) { prof_on_ = false; prof_ = nullptr; cudaGetLastError(); }
        else { cudaMemset(prof_, 0, np * 8); prof_h_.assign(np, 0); }
    }
    Bump real;
    real.base = (uint8_t*) arena_;
    carve(real);
    sink_.staging = (unsigned long long) staging_;
    sink_.staging_cap = kStagingBlobs;
    (void) TS;
    if (cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking) != cudaSuccess) {
        err = "verify: copy stream create failed";
        return false;
    }
    if (cudaStreamCreateWithFlags(&cs_, cudaStreamNonBlocking) != cudaSuccess) {
        err = "verify: stream create failed";
        return false;
    }
    if (cudaEventCreateWithFlags(&commit_done_, cudaEventDisableTiming) != cudaSuccess) {
        err = "verify: event create failed";
        return false;
    }
    // SM89 Ada Lovelace 72MB Persistent L2 Cache Window:
    // Pin verify device arena into persistent L2 cache to eliminate GDDR6X trips during token verification.
    {
        size_t max_persisting_l2 = 0;
        cudaDeviceGetLimit(&max_persisting_l2, cudaLimitPersistingL2CacheSize);
        if (max_persisting_l2 == 0) {
            const size_t target_window = 40ULL * 1024 * 1024; // 40 MiB (out of 49.5 MiB max on SM89 RTX 4090)
            if (cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, target_window) == cudaSuccess) {
                max_persisting_l2 = target_window;
            } else {
                cudaGetLastError();
            }
        }
        if (max_persisting_l2 > 0 && arena_ && count.used > 0) {
            cudaStreamAttrValue attr;
            std::memset(&attr, 0, sizeof(attr));
            attr.accessPolicyWindow.base_ptr = arena_;
            attr.accessPolicyWindow.num_bytes = std::min<size_t>(count.used, max_persisting_l2);
            attr.accessPolicyWindow.hitRatio = 1.0f;
            attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
            if (cudaStreamSetAttribute(cs_, cudaStreamAttributeAccessPolicyWindow, &attr) == cudaSuccess) {
                std::fprintf(stderr, "strata verify: SM89 persistent L2 cache window enabled (%.1f MiB on cs)\n", (double) attr.accessPolicyWindow.num_bytes / 1048576.0);
            } else {
                cudaGetLastError();
            }
        }
    }
    if (batch_workspace && batch_parallel_) {
        if (cudaEventCreateWithFlags(&batch_fork_, cudaEventDisableTiming) != cudaSuccess) {
            err = "batch verify: fork event failed"; return false;
        }
        for (auto& e : batch_join_) if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) {
            err = "batch verify: join event failed"; return false;
        }
    }
    // E-6: a layer whose routed experts are all resident is planned on the device (STRATA_VERIFY_DEVICE_PLAN=1: on;
    // exact, but neutral on RIBPC 1-2 GPUs: off by default)
    {
        const char* v = std::getenv("STRATA_VERIFY_DEVICE_PLAN");
        device_plan_ = v != nullptr && std::atoi(v) != 0;
    }
    if (device_plan_) {
        bool ok2 = cudaMalloc((void**) &skip_, 64) == cudaSuccess && cudaMemset(skip_, 0, 64) == cudaSuccess;
        if (ok2 && hits.slot_off != nullptr && hits.n_slots > 0) {
            ok2 = cudaMalloc((void**) &slot_off_d_, (size_t) hits.n_slots * sizeof(unsigned long long)) == cudaSuccess &&
                  cudaMemcpy(slot_off_d_, hits.slot_off, (size_t) hits.n_slots * sizeof(unsigned long long),
                             cudaMemcpyHostToDevice) == cudaSuccess;
        }
        if (!ok2) { cudaGetLastError(); device_plan_ = false; }
    }
    std::fprintf(stderr, "strata verify: window up to %d tokens, %.1f MiB of device buffers\n", max_t,
                 (double) count.used / 1048576.0);
    return true;
}
const float* Verifier::final_R(int t) const { return R_ + (size_t) t * (size_t) (g_->hc * g_->n_embd); }
// ================================ THE WINDOW, AS CAPTURED ================================
//
// Plan v0.3 P6 (split window): with `groups_ == 2` the window's tokens are cut into two groups A = [0, T/2 up) and
// B = the rest, and the stream is ordered
//
//     pre(0,A) pre(0,B) | post(0,A) pre(1,A) | post(0,B) pre(1,B) | post(1,A) pre(2,A) | ...
//
// so the CPU computes A's experts of layer l while the GPU runs B's mixer and router of layer l, and B's experts
// while the GPU combines A and runs A's layer l+1.  B's mixer only needs A's mixer of the same layer (K/V, GDN
// state), never A's experts, so nothing waits that did not wait before.  Every token's arithmetic is unchanged.
bool Verifier::record_window(int T, cudaStream_t cs, std::string& err, int phase, int64_t layer,
                            int64_t row0) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const WeightTable& wt = *wt_;
    SessionState& ss = *ss_;
    const int64_t N = g.n_embd, HC = g.hc, K = ss.k, C = g.ssm_conv_channels, ZV = g.ssm_value_dim;
    const int64_t HV = g.ssm_v_heads, HK = g.ssm_k_heads, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const int64_t IQ = g.idx_q_heads, ID = g.idx_key_dim, NE = g.n_expert, MT = max_t_;
    const QsaShapes s = shapes_of(g);
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    const int64_t TS = (s.idx_block - 1) * ID;
    const bool ple_on = ss.ple.ready() && ple_stage();
    auto Rt = [&](int t) { return R_ + (size_t) t * HC * N; };
    const int G = (split_ && T >= 2) ? 2 : 1;
    static const bool dec_batch = [] { const char* v = std::getenv("STRATA_DEC_BATCH"); return v == nullptr || std::atoi(v) != 0; }();
    auto stamp = [&](int64_t l, int i, int grp) { if (prof_on_ && grp == 0) gpu_stamp(prof_, (int) (l * kProfPer + i), cs); };
    const int tb_[2] = {0, (T + 1) / 2}, te_[2] = {G == 2 ? (T + 1) / 2 : T, T};
    groups_[T] = G;
    // ---- the window's inputs, from mapped staging
    const int64_t HB = Verifier::handoff_floats(g);
    const int32_t* pos_k = pos_ + MT * NH;
    const int32_t* pos_i = pos_k + MT * NKV;
    if (phase <= 0) {
    copy_i32_from_mapped(tok_, m_tok_, T, cs);
    copy_i32_from_mapped(step_, m_step_, (int64_t) T * kStepCount, cs);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) MT * (NH + NKV + IQ), cs);
    // per-ROW positions of the K rows [t][NKV] and the indexer query rows [t][IQ] (for batched RoPE)
    if (ple_on) copy_from_mapped(ple_, m_ple_, (int64_t) T * N, cs);
    // ---- the embeddings, broadcast to the hc streams - or, in a later stage of a layer split, the previous stage's
    // residual, pending write and inject (see set_stage)
    if (lb_ > 0) {
        // C2-A3: a batch round packs every member's rows [0, total); row0 is this member's offset.
        if (row0 < 0 || row0 + T > (int64_t) strata::kernels::cpu::MAXT) { err = "verify: window row offset out of range (row0=" + std::to_string(row0) + ", T=" + std::to_string(T) + ", max_t=" + std::to_string(max_t_) + ")"; return false; }
        for (int t = 0; t < T; ++t) {
            const float* in = hand_in_ + (size_t) (row0 + t) * HB;
            copy_from_mapped(Rt(t), in, HC * N, cs);
            copy_from_mapped(bo_ + (size_t) t * N, in + HC * N, N, cs);
            copy_from_mapped(inj2_ + (size_t) t * HC, in + HC * N + N, HC, cs);
        }
    } else if (const NativeEmbed* ne = native_embed()) {       // plan v0.3 P6: the GGUF-form table
        ne->gather_dev(tok_, T, emb_, cs);
        broadcast_streams(emb_, R_, N, (int) HC, T, cs);
    } else {
        const WeightRef* w = wt.find("token_embd.weight");
        if (w == nullptr || w->codebook_iq4nl || (w->code_bits != 2 && w->code_bits != 4 && w->code_bits != 8)) {
            err = "verify: token_embd.weight is missing or not an S2/S4/S8 tensor";
            return false;
        }
        const auto* codes = (const uint8_t*) w->data;
        const auto* scales = (const float*) (codes + w->codes_bytes);
        const auto* offsets = w->has_offset ? (const float*) (codes + w->codes_bytes + w->scales_bytes) : nullptr;
        const uint64_t row_codes = (uint64_t) (w->ne0 / (8 / w->code_bits));
        const uint64_t row_groups = (uint64_t) (w->ne0 / w->group_elems);
        embedding_gather_dev(codes, scales, offsets, tok_, T, w->ne0, w->code_bits, w->code_bias, w->group_elems,
                             row_codes, row_groups, emb_, cs);
        broadcast_streams(emb_, R_, N, (int) HC, T, cs);
    }
    }
    if (phase == 0) return true;
    // per-layer state indices (GDN and QSA layers are numbered separately)
    std::vector<int64_t> gdn_idx((size_t) g.n_layers, -1), qsa_idx((size_t) g.n_layers, -1);
    {
        int64_t qi = 0, gi = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (is_qsa_layer(g, l)) qsa_idx[(size_t) l] = qi++;
            else gdn_idx[(size_t) l] = gi++;
        }
    }
    // ---------------------------------------------------------------- pre(l, group): up to the ring
    auto pre = [&](int64_t l, int grp) -> bool {
        const int tb = tb_[grp], te = te_[grp], n = te - tb;
        stamp(l, 0, grp);
        const LayerView v(wt, l);
        const char* pfx[2] = {"hc_attn_", "hc_ffn_"};
        const WeightRef *wn[2], *wd[2], *wu[2], *wi[2];
        for (int h = 0; h < 2; ++h) {
            wn[h] = need(v, (std::string(pfx[h]) + "norm.weight").c_str(), err);
            wd[h] = need(v, (std::string(pfx[h]) + "down.weight").c_str(), err);
            wu[h] = need(v, (std::string(pfx[h]) + "up.weight").c_str(), err);
            wi[h] = need(v, (std::string(pfx[h]) + "inject.weight").c_str(), err);
            if (!wn[h] || !wd[h] || !wu[h] || !wi[h]) return false;
        }
        // the previous layer's FFN write, folded into this layer's first read (a control vector after it has
        // already applied it)
        bool pending = l > 0 && !cvec().covers(l - 1);
        if (l == 1 && ple_on) {
            float* normalized = (float*) ((uint8_t*) ss.ple.scratch + ple_block_scratch_bytes());
            for (int t = tb; t < te; ++t) {
                gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
                PleOut po;
                po.normalized = normalized;
                po.result = Rt(t);
                try {
                    ple_block(ple_ + t * N, Rt(t), ss.ple.hist, ss.ple.w, po, ss.ple.scratch, cs);
                    ple_history_advance(ss.ple.hist, normalized, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE: ") + e.what();
                    return false;
                }
                copy_from_mapped(hist_snap_ + (size_t) t * HS, ss.ple.hist, HS, cs);
            }
            pending = false;
        }
        auto gr_read_group = [&](int half, bool apply, float* inj_prev, float* inj_out) {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = tb; t < te; ++t) {
                FusedGrArgs& a = fa[t - tb];
                a.R = Rt(t); a.R_out = Rt(t); a.apply = apply;
                a.bo_prev = bo_ + t * N; a.inj_prev = inj_prev + t * HC;
                a.w_norm = (const float*) wn[half]->data; a.w_down = (const uint16_t*) wd[half]->data;
                a.w_up = (const uint16_t*) wu[half]->data; a.w_inject = (const uint16_t*) wi[half]->data;
                a.eps = EPS; a.lo = lo_ + t * g.hc_lr; a.rs = rs_ + t * HC;
                a.inject_out = inj_out + t * HC; a.mixed = mixed_ + t * N;
            }
            fused_gr_read_multi(fa, n, xn_ + (size_t) tb * HC * N, cs, (prof_on_ && grp == 0) ? prof_ : nullptr,
                                (int) (l * kProfPer + (half == 0 ? 27 : 30)));
        };
        gr_read_group(0, pending, inj2_, inj_);
        stamp(l, 1, grp);
        float* xm = mixed_ + tb * N;
        bool il_ready = false;   // xil_ holds xq_'s interleaved copy (reset whenever xq_ is rewritten)
        auto mm = [&](const WeightRef* w, float* out, int n_in, int n_out) {
            if (g_mmvq_il() && strata::kernels::native_mmvq_il_supported(w->native_type, n, n_out)) {
                if (!il_ready) {
                    strata::kernels::native_q8_1_interleave(xq_, xil_, n_in, n, cs);
                    il_ready = true;
                }
                strata::kernels::native_mmvq_il(w->native_type, w->native_data, xq_, xil_, out, n_in, n_out, n, cs);
            } else {
                native_mmvq(w->native_type, w->native_data, xq_, out, n_in, n_out, n, cs);
            }
        };
        try {
            if (!is_qsa_layer(g, l)) {
                // ======================= GDN =======================
                const WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                                *wout = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                                *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                                *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                                *wsa = need(v, "ssm_a", err);
                if (!wqkv || !wg || !wout || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                if (!native_of(wqkv, v.name("attn_qkv.weight"), err) || !native_of(wg, v.name("attn_gate.weight"), err) ||
                    !native_of(wout, v.name("ssm_out.weight"), err))
                    return false;
                const int64_t gi = gdn_idx[(size_t) l];
                float* state = ss.gdn_state + (size_t) (gi - ss.gdn_ord0) * gdn_floats;
                float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                float* qkv = qkv_L_ + (size_t) gi * MT * C;
                float* hb = h_L_ + (size_t) gi * MT * C;
                float* gate = gate_L_ + (size_t) gi * MT * HV;
                float* beta = beta_L_ + (size_t) gi * MT * HV;
                native_quantize_q8_1(xm, xq_, (int) N, n, cs);
                mm(wqkv, qkv + (size_t) tb * C, (int) N, (int) C);
                stamp(l, 2, grp);
                gdn_conv_l2_multi(conv, qkv, (const float*) wc->data, hb, (int) C, (int) (2 * HK), EPS, n, cs, tb);
                stamp(l, 3, grp);
                gdn_ab_multi(xm, (const uint16_t*) wa->data, (const uint16_t*) wb->data, (const float*) wdt->data,
                             (const float*) wsa->data, gate + (size_t) tb * HV, beta + (size_t) tb * HV, (int) N, (int) HV,
                             n, cs);
                stamp(l, 4, grp);
                mm(wg, z_ + (size_t) tb * ZV, (int) N, (int) ZV);
                stamp(l, 5, grp);
                // the recurrence from the untouched state over tokens [0, te); outputs only for this group's
                gdn_step_norm_multi(state, hb, (int) C, gate, beta, z_, (const float*) wnm->data, EPS, y_, (int) HK,
                                    (int) HV, te, nullptr, cs, tb);
                stamp(l, 6, grp);
                native_quantize_q8_1(y_ + (size_t) tb * ZV, xq_, (int) ZV, n, cs);
                il_ready = false;
                mm(wout, bo_ + tb * N, (int) ZV, (int) N);
            } else {
                // ======================= QSA =======================
                const int64_t qi = qsa_idx[(size_t) l];
                const QsaState& st = ss.qsa_states[qi];
                const WeightRef *wik = need(v, "indexer.k_proj.weight", err), *wq = need(v, "attn_q.weight", err),
                                *wk = need(v, "attn_k.weight", err), *wv = need(v, "attn_v.weight", err),
                                *wo = need(v, "attn_output.weight", err), *wiq = need(v, "indexer.q_proj.weight", err),
                                *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                                *wiqn = need(v, "indexer.q_norm.weight", err), *wikn = need(v, "indexer.k_norm.weight", err);
                if (!wik || !wq || !wk || !wv || !wo || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
                if (!native_of(wq, v.name("attn_q.weight"), err) || !native_of(wk, v.name("attn_k.weight"), err) ||
                    !native_of(wv, v.name("attn_v.weight"), err) || !native_of(wo, v.name("attn_output.weight"), err))
                    return false;
                auto norm_rope = [&](float* data, const WeightRef* norm, int rows, int cols, const int32_t* pos) {
                    if (native_qsa_enabled()) native_qsa_rms_norm_weighted(data, (const float*) norm->data, data, cols, rows, EPS, cs);
                    else rms_norm_weighted(data, (const float*) norm->data, rows, cols, EPS, cs);
                    if (native_rope_enabled()) native_rope_apply(data, data, rows, cols, (int) s.n_rot, rope_scaling(), pos, cs);
                    else rope_neox_apply(data, data, rows, cols, (int) s.n_rot, st.cos_tab, st.sin_tab, pos, cs);
                };
                float* idx_raw = idx_raw_L_ + (size_t) qi * MT * ID;
                // the per-token GEMVs / norms / RoPEs / copies of this layer as one launch over the
                // window's rows each - row-wise identical arithmetic (STRATA_DEC_BATCH=0: token by token)
                // R2-revive: the batched tail is open under Q4_0 KV too. K2: the kv_append_q4 and indexer
                // appends batch a group's rows into ONE launch each via their *_batch_step entries, which
                // read every row's position from its device step row - a host position would go stale on
                // graph replay (batch graphs are keyed by shape, not position). n == 1 and the other
                // storage modes keep the per-token loop. The q rotation MUST stay under kv_q4 (added to
                // the qb path below): without it <Hq,Hk> misaligns.
                const bool qb = dec_batch && n > 1 && native_qsa_enabled() && native_rope_enabled();
                native_quantize_q8_1(xm, xq_, (int) N, n, cs);
                if (qb) bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) wik->data, idx_raw + tb * ID, ID, N, ID, n, cs);
                else for (int t = tb; t < te; ++t)
                    bf16_gemv_fp32_mmvf(mixed_ + t * N, (const uint16_t*) wik->data, idx_raw + t * ID, (int) N, (int) ID, cs);
                stamp(l, 7, grp);
                mm(wk, kcur_ + tb * NKV * HD, (int) N, (int) (NKV * HD));
                mm(wv, vcur_ + tb * NKV * HD, (int) N, (int) (NKV * HD));
                if (qb) norm_rope(kcur_ + tb * NKV * HD, wkn, (int) (n * NKV), (int) HD, pos_k + tb * NKV);
                else for (int t = tb; t < te; ++t) norm_rope(kcur_ + t * NKV * HD, wkn, (int) NKV, (int) HD, pos_ + t * NH);
                if (st.kv_q4) {   // Q4_0 KV (kv_q4.hpp): K and V rotated before they are stored
                    fwht256_inplace_cuda(kcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                    fwht256_inplace_cuda(vcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                } else if (st.kv_hybrid) {   // K8V4: only V is rotated
                    fwht256_inplace_cuda(vcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                }
                stamp(l, 8, grp);
                if (grp == 0) copy_from_mapped(tail_snap_ + (size_t) qi * TS, st.idx_tail, TS, cs);
                // K2: one launch per K/V over the group's rows; each row's position is read from its own
                // device step row at replay. The per-token loop below stays the oracle (and the path for
                // n == 1, kv_hybrid/kv_int8/fp16, and STRATA_DEC_BATCH=0).
                if (dec_batch && n > 1 && st.kv_q4)
                    kv_append_q4_batch_step(st.k_q4, st.v_q4, st.page_table, step_ + tb * kStepCount, n,
                                            kcur_ + tb * NKV * HD, vcur_ + tb * NKV * HD, s, cs, &st.host);
                else
                for (int t = tb; t < te; ++t) {
                    const int32_t* step_t = step_ + t * kStepCount;
                    if (st.kv_hybrid) {   // K8V4: the unused half's lanes folded onto the used pool (layer.cpp)
                        kv_append_q8_step(st.k_q, st.k_q, st.k_scale, st.k_scale, st.page_table, step_t,
                                          kcur_ + t * NKV * HD, kcur_ + t * NKV * HD, s, cs, nullptr);
                        kv_append_q4_step(st.v_q4, st.v_q4, st.page_table, step_t, vcur_ + t * NKV * HD,
                                          vcur_ + t * NKV * HD, s, cs, nullptr);
                    } else if (st.kv_q4)
                        kv_append_q4_step(st.k_q4, st.v_q4, st.page_table, step_t, kcur_ + t * NKV * HD,
                                          vcur_ + t * NKV * HD, s, cs, &st.host);
                    else if (st.kv_int8)
                        kv_append_q8_step(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, step_t,
                                          kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, s, cs, &st.host);
                    else
                        kv_append_step(st.k_pool, st.v_pool, st.page_table, step_t, kcur_ + t * NKV * HD,
                                       vcur_ + t * NKV * HD, s, cs, &st.host);
                }
                const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                // K2: the group's contiguous run in ONE call; the kernels read the run's first cell from
                // its device step row and mask their fixed launch shapes, so any replay position works.
                if (dec_batch && n > 1)
                    native_qsa_indexer_append_batch_step(idx_raw + tb * ID, n, step_ + tb * kStepCount + kStepPos, 0,
                                                         (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                                         rope_scaling(), cs);
                else
                for (int t = tb; t < te; ++t)
                    native_qsa_indexer_append(idx_raw + t * ID, step_ + t * kStepCount + kStepPos, 0,
                                              (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                              rope_scaling(), cs);
                stamp(l, 9, grp);
                mm(wq, qfull_ + tb * NH * 2 * HD, (int) N, (int) (NH * 2 * HD));
                if (qb) {
                    if (cudaMemcpy2DAsync(qcur_ + tb * NH * HD, (size_t) HD * 4, qfull_ + tb * NH * 2 * HD, (size_t) HD * 2 * 4,
                                          (size_t) HD * 4, (size_t) (n * NH), cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
                        err = "verify: the q/gate split failed";
                        return false;
                    }
                    norm_rope(qcur_ + tb * NH * HD, wqn, (int) (n * NH), (int) HD, pos_ + tb * NH);
                    if (st.kv_q4) fwht256_inplace_cuda(qcur_ + tb * NH * HD, (int64_t) (n * NH), cs);   // <Hq, Hk> = <q, k> (mirrors the per-token path)
                    bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) wiq->data, qidx_ + tb * IQ * ID, IQ * ID,
                                              N, IQ * ID, n, cs);
                    norm_rope(qidx_ + tb * IQ * ID, wiqn, (int) (n * IQ), (int) ID, pos_i + tb * IQ);
                } else {
                for (int t = tb; t < te; ++t) {
                    float* qc = qcur_ + t * NH * HD;
                    if (cudaMemcpy2DAsync(qc, (size_t) HD * 4, qfull_ + t * NH * 2 * HD, (size_t) HD * 2 * 4,
                                          (size_t) HD * 4, (size_t) NH, cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
                        err = "verify: the q/gate split failed";
                        return false;
                    }
                    norm_rope(qc, wqn, (int) NH, (int) HD, pos_ + t * NH);
                    if (st.kv_q4) fwht256_inplace_cuda(qc, NH, cs);   // <Hq, Hk> = <q, k>
                }
                for (int t = tb; t < te; ++t) {
                    float* qx = qidx_ + t * IQ * ID;
                    bf16_gemv_fp32_mmvf(mixed_ + t * N, (const uint16_t*) wiq->data, qx, (int) N, (int) (IQ * ID), cs);
                    norm_rope(qx, wiqn, (int) IQ, (int) ID, pos_ + t * NH);
                }
                }
                stamp(l, 10, grp);
                qsa_block_scores(st.idx_pooled, st.idx_dead, qidx_ + tb * IQ * ID, step_ + tb * kStepCount, n, max_blocks_,
                                 s, scores_ + (size_t) tb * max_blocks_, cs);
                qsa_block_topk(scores_ + (size_t) tb * max_blocks_, step_ + tb * kStepCount, n, max_blocks_, cap_, s,
                               sel_ + (size_t) tb * cap_, cs);
                stamp(l, 11, grp);
                // KV streaming: the n selections' blocks resident (device-side, inside the graph)
                qsa_kv_resolve(st, *g_, sel_ + (size_t) tb * cap_, step_ + tb * kStepCount, n, cap_, cs);
                stamp(l, 12, grp);
                const QsaAttnPools pools = qsa_attn_pools(st);
                const bool can_fuse_gate = !st.kv_q4 && !st.kv_hybrid && native_qsa_enabled();
                if (can_fuse_gate) {
                    // Sprint 4 Polish: Direct fused attention merge + sigmoid gate epilogue
                    // Eliminates intermediate global memory round-trip to attn_ and saves 1 kernel launch
                    // Frontier 4: Tree verification causal attention gating
                    // When tree verification is active, candidate tokens evaluate simultaneously
                    // in one target forward pass with intra-tree causal masking.
                    qsa_decode_attn_batch_fused_gate(qcur_ + tb * NH * HD, pools, sel_ + (size_t) tb * cap_,
                                                    step_ + tb * kStepCount, cap_, s,
                                                    attn_scratch_ + (size_t) tb * attn_scratch_floats_,
                                                    nullptr, qfull_ + tb * NH * 2 * HD,
                                                    attn32_ + tb * NH * HD, n, cs);
                    stamp(l, 13, grp);
                } else {
                    qsa_decode_attn_batch(qcur_ + tb * NH * HD, pools, sel_ + (size_t) tb * cap_, step_ + tb * kStepCount, cap_,
                                          s, attn_scratch_ + (size_t) tb * attn_scratch_floats_, attn_ + tb * NH * HD, n, cs);
                    stamp(l, 13, grp);
                    if (st.kv_q4 || st.kv_hybrid) fwht256_inplace_cuda(attn_ + tb * NH * HD, (int64_t) n * NH, cs);   // back: H^-1 = H
                    if (qb) native_qsa_gate_apply(attn_ + tb * NH * HD, qfull_ + tb * NH * 2 * HD, attn32_ + tb * NH * HD,
                                                  (int) (n * NH), (int) HD, cs);
                    else
                    for (int t = tb; t < te; ++t) {
                        if (native_qsa_enabled())
                            native_qsa_gate_apply(attn_ + t * NH * HD, qfull_ + t * NH * 2 * HD, attn32_ + t * NH * HD,
                                                  (int) NH, (int) HD, cs);
                        else
                            qsa_gate_apply_f32(attn_ + t * NH * HD, qfull_ + t * NH * 2 * HD, s, attn32_ + t * NH * HD, cs);
                    }
                }
                stamp(l, 14, grp);
                native_quantize_q8_1(attn32_ + tb * NH * HD, xq_, (int) (NH * HD), n, cs);
                il_ready = false;
                mm(wo, bo_ + tb * N, (int) (NH * HD), (int) N);
            }
        } catch (const std::exception& e) {
            err = "verify layer " + std::to_string(l) + ": " + e.what();
            return false;
        }
        stamp(l, 16, grp);
        gr_read_group(1, true, inj_, inj2_);
        // the window's rows routed in 2 launches (one router GEMV reading the weight once, one
        // top-10) instead of 2 per token; every row's arithmetic is the single-token call's (STRATA_DEC_BATCH=0: old)
        const WeightRef* w_router = v.get("ffn_gate_inp.weight");
        if (dec_batch && n > 1 && w_router != nullptr && native_router_enabled() && NE == 512 && K == 10) {
            try {
                bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) w_router->data, logits_ + tb * NE, NE, N,
                                          NE, n, cs);
                native_router_top10_multi(logits_ + tb * NE, ids_ + tb * K, w_ + tb * K, n, cs);
            } catch (const std::exception& e) { err = "verify router: " + std::string(e.what()); return false; }
        } else
        for (int t = tb; t < te; ++t) {
            MoEBuffers mb = ss.moe;
            mb.logits = logits_ + t * NE; mb.ids = ids_ + t * K; mb.weights = w_ + t * K;
            if (!moe_route(wt, g, l, K, mb, mixed_ + t * N, cs, err, nullptr)) return false;
        }
        if (device_plan_ && phase < 0)   // batch coordinator builds the cross-request plan
            resident_plan(ids_ + tb * K, n * (int) K, (int) K, hits_.d_res + l * g.n_expert, (int) g.n_expert,
                          hits_.cache_base, slot_off_d_, (long long) hits_.blob,
                          plan_ + (size_t) grp * (size_t) (plan_i32_ + 16), (long long) max_t_ * K, skip_ + grp,
                          (uint32_t) ((l - lb_) * G + grp + 1), cs);
        if (phase < 0)
            doorbell_publish(xm, ids_ + tb * K, w_ + tb * K, (int64_t) n * N, (int64_t) n * K, m_x_ + tb * N,
                             m_ids_ + tb * K, m_w_ + tb * K, m_seq_, cs);
        stamp(l, 17, grp);
        {
            const WeightRef *wgi = need(v, "ffn_gate_inp_shexp.weight", err), *wsg = need(v, "ffn_gate_shexp.weight", err),
                            *wsu = need(v, "ffn_up_shexp.weight", err), *wsd = need(v, "ffn_down_shexp.weight", err);
            if (!wgi || !wsg || !wsu || !wsd) return false;
            if (!native_of(wsg, v.name("ffn_gate_shexp.weight"), err) || !native_of(wsu, v.name("ffn_up_shexp.weight"), err) ||
                !native_of(wsd, v.name("ffn_down_shexp.weight"), err))
                return false;
            NativeSharedWeights nsw;
            nsw.gate_type = wsg->native_type; nsw.gate_data = wsg->native_data;
            nsw.up_type = wsu->native_type; nsw.up_data = wsu->native_data;
            nsw.down_type = wsd->native_type; nsw.down_data = wsd->native_data;
            nsw.q8_1 = xq_;
            if (dec_batch) f32_to_bf16_bulk(mixed_ + tb * N, sh_bf16_ + tb * N, (int64_t) n * N, cs);   // contiguous rows
            else for (int t = tb; t < te; ++t) f32_to_bf16_bulk(mixed_ + t * N, sh_bf16_ + t * N, N, cs);
            try {
                shared_expert_multi(n, xm, sh_bf16_ + tb * N, nsw, (const uint16_t*) wgi->data, sh_gate_ + (size_t) tb * g.n_ff,
                                    sh_up_ + (size_t) tb * g.n_ff, sh_g_ + tb, shared_ + tb * N, N, g.n_ff, cs);
            } catch (const std::exception& e) {
                err = std::string("verify shared expert: ") + e.what();
                return false;
            }
        }
        if (strata::kernels::cpu::expert_layout().native)
            quantize_q8_1_rows(xm, n, N, nat_xq_ + (size_t) tb * (N / 32) * 36, cs);
        else
            quantize_q8_0_scaled(xm, hit_xq_ + (size_t) tb * (N / 32) * 34, hit_xs_ + (size_t) tb * (N / 32), (int64_t) n * N, cs);
        stamp(l, 18, grp);
        return true;
    };
    // ---------------------------------------------------------------- post(l, group): experts, combine
    auto post = [&](int64_t l, int grp) -> bool {
        const int tb = tb_[grp], te = te_[grp], n = te - tb;
        const uint32_t ring = (uint32_t) ((l - lb_) * G + grp + 1);
        const int64_t cap = (int64_t) n * K, capx = (int64_t) max_t_ * K;
        if (phase != 3) {
        int32_t* pl = plan_ + (size_t) grp * (size_t) (plan_i32_ + 16);
        if (device_plan_) {   // E-6: skipped when the device planned this group (all its experts resident)
            wait_flag_ge_or(m_flagA_, ring, skip_ + grp, cs);
            copy_i32_from_mapped_unless(pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, skip_ + grp, ring, cs);
        } else {
            wait_flag_ge(m_flagA_, ring, cs);                  // the pool published this group's GPU plan
            copy_i32_from_mapped(pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, cs);
        }
        stamp(l, 19, grp);
        const int32_t* p_counts = pl;
        const int32_t* p_start = pl + 4;
        const int32_t* p_dst = p_start + capx + 1;
        const int32_t* p_tok = p_dst + capx;
        const int64_t ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
        const unsigned long long* p_ptr = (const unsigned long long*) (pl + ptr_off);
        const unsigned long long* p_ptr2 = p_ptr + capx;
        const int32_t* p_start2 = pl + ptr_off + 4 * capx;
        float* parts_out = parts_ + (size_t) tb * K * N;
        const auto& lay = strata::kernels::cpu::expert_layout();
        if (device_plan_) {
            // E-6 / ZERO-COPY RESIDENT MOE PATH:
            // All experts are 100% resident in GPU VRAM (verified by resident_plan on device).
            // Write directly to parts_ (destination of down projection), bypassing:
            // 1. hit_out scratchpad
            // 2. wait_flag_ge_or(m_flagB_) kernel
            // 3. grouped(p_ptr2) PCIe staging kernel (0 experts)
            // 4. wait_flag_ge_or(m_flag_) kernel
            // 5. copy_or_zero_from_mapped kernel (VRAM zeroing)
            // 6. moe_hit_add kernel (parts += hit_out copy)
            if (lay.native) {
                const auto& f = lay.fmt[(size_t) l];
                const NativeExpertLayout L = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
                native_expert_grouped(L, p_ptr, p_start, p_counts, p_dst, p_tok, cap, cap,
                                      nat_xq_ + (size_t) tb * (N / 32) * 36, hit_scratch_, parts_out, cs);
            } else {
                moe_grouped_s2(p_ptr, p_start, p_counts, p_dst, p_tok, cap, cap, hit_xq_ + (size_t) tb * (N / 32) * 34,
                               hit_xs_ + (size_t) tb * (N / 32), hit_scratch_, parts_out, cs);
            }
            stamp(l, 20, grp);
        } else {
            float* hit_out = hit_out_ + (size_t) tb * K * N;
            auto grouped = [&](const unsigned long long* gp, const int32_t* gs, const int32_t* gn) {
                if (lay.native) {
                    const auto& f = lay.fmt[(size_t) l];
                    const NativeExpertLayout L = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
                    native_expert_grouped(L, gp, gs, gn, p_dst, p_tok, cap, cap,
                                          nat_xq_ + (size_t) tb * (N / 32) * 36, hit_scratch_, hit_out, cs);
                } else {
                    moe_grouped_s2(gp, gs, gn, p_dst, p_tok, cap, cap, hit_xq_ + (size_t) tb * (N / 32) * 34,
                                   hit_xs_ + (size_t) tb * (N / 32), hit_scratch_, hit_out, cs);
                }
            };
            grouped(p_ptr, p_start, p_counts);
            stamp(l, 20, grp);
            wait_flag_ge(m_flagB_, ring, cs);                 // the PCIe share is in staging (DMA) or mapped
            if (sink_.pcie_mode == 2) {                            // stage it with a copy kernel, then point at staging
                const int64_t per = G == 2 ? kStagingBlobs / 2 : kStagingBlobs;
                uint8_t* stage = staging_ + (size_t) (grp * per) * lay.max_blob;
                fetch_blobs(p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l), (int) per, cs);
                rebase_ptrs((unsigned long long*) p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l), cs);
            }
            stamp(l, 21, grp);
            grouped(p_ptr2, p_start2, p_counts + 2);
            stamp(l, 22, grp);
            wait_flag_ge(m_flag_, ring, cs);               // the CPU's share is in the mapped rows
            stamp(l, 23, grp);
            if (dec_batch || p_counts != nullptr)   // only the CPU rows cross PCIe (p_dst[0, counts[1]) = the GPU's own rows)
                copy_rows_from_mapped(parts_ + (size_t) tb * K * N, m_ymiss_ + (size_t) tb * K * N, (int64_t) n * K, N,
                                      p_dst, p_counts + 1, cs);
            else
                copy_from_mapped(parts_ + (size_t) tb * K * N, m_ymiss_ + (size_t) tb * K * N, (int64_t) n * K * N, cs);
            moe_hit_add(parts_ + (size_t) tb * K * N, hit_out, p_dst, p_counts + 1, cap, N, cs);
        }
        }
        if (phase == 2) return true;
        if (dec_batch && n > 1 && native_moe_combine_enabled()) {   // one launch for the window's rows
            try {
                native_moe_combine_multi(parts_ + (size_t) tb * K * N, w_ + tb * K, shared_ + tb * N, bo_ + tb * N, N, K, n, cs);
            } catch (const std::exception& e) { err = "verify combine: " + std::string(e.what()); return false; }
        } else
        for (int t = tb; t < te; ++t) {
            MoEBuffers mb = ss.moe;
            mb.weights = w_ + t * K; mb.shared = shared_ + t * N;
            if (!moe_combine_parts(g, l, K, mb, parts_ + (size_t) t * K * N, bo_ + t * N, cs, err)) return false;
        }
        stamp(l, 24, grp);
        if (l == g.n_layers - 1) {
            for (int t = tb; t < te; ++t) gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
            if (cvec().covers(l)) cvec_apply(Rt(tb), l, n, HC * N, nullptr, 0, nullptr, 0, false, cs);
        } else if (cvec().covers(l)) {
            cvec_apply(Rt(tb), l, n, HC * N, bo_ + tb * N, N, inj2_ + tb * HC, HC, true, cs);
        }
        return true;
    };
    if (phase == 1) return pre(layer, 0);
    if (phase == 2 || phase == 3) return post(layer, 0);
    if (phase == 5) {   // C2-A3: a split's earlier stage hands this window's packed rows on
        if (row0 < 0 || row0 + T > (int64_t) strata::kernels::cpu::MAXT) { err = "verify: window row offset out of range (row0=" + std::to_string(row0) + ", T=" + std::to_string(T) + ", max_t=" + std::to_string(max_t_) + ")"; return false; }
        for (int t = 0; t < T; ++t) {
            float* out = hand_out_ + (size_t) (row0 + t) * HB;
            copy_from_mapped(out, Rt(t), HC * N, cs);
            copy_from_mapped(out + HC * N, bo_ + (size_t) t * N, N, cs);
            copy_from_mapped(out + HC * N + N, inj2_ + (size_t) t * HC, HC, cs);
        }
        return true;
    }
    if (phase < 0) {
    for (int grp = 0; grp < G; ++grp)
        if (!pre(lb_, grp)) return false;
    for (int64_t l = lb_; l < le_; ++l)
        for (int grp = 0; grp < G; ++grp) {
            if (!post(l, grp)) return false;
            if (l + 1 < le_ && !pre(l + 1, grp)) return false;
        }
    if (le_ < g.n_layers) {   // a layer split's earlier stage: hand the residual on, no head
        for (int t = 0; t < T; ++t) {
            copy_from_mapped(hand_out_ + (size_t) t * HB, Rt(t), HC * N, cs);
            copy_from_mapped(hand_out_ + (size_t) t * HB + HC * N, bo_ + (size_t) t * N, N, cs);
            copy_from_mapped(hand_out_ + (size_t) t * HB + HC * N + N, inj2_ + (size_t) t * HC, HC, cs);
        }
        return true;
    }
    }
    // ---- the head, T columns, and the argmax of each
    stamp(g.n_layers, 0, 0);
    {
        const WeightRef *hn = wt.find("output_hc_norm.weight"), *hd = wt.find("output_hc_down.weight"),
                        *hu = wt.find("output_hc_up.weight");
        if (!hn || !hd || !hu) { err = "verify: an output_hc_* weight is missing"; return false; }
        for (int t = 0; t < T; ++t) {
            BlockBuffers bb = ss.block;
            bb.R = Rt(t);
            bb.mixed = head_mixed_ + t * N;
            if (head_ != nullptr && head_->loaded()) {
                if (!lm_head_mix(wt, g, bb, cs, err)) return false;
            } else if (!lm_head(wt, g, bb, head_logits_ + (size_t) t * n_vocab_, cs, err)) {
                return false;
            }
        }
        if (phase == 40) return true;
        if (head_ != nullptr && head_->loaded()) {
            try {
                native_quantize_q8_1(head_mixed_, xq_, (int) N, T, cs);
                if (g_mmvq_il() && strata::kernels::native_mmvq_il_supported(head_->type(), (int) T, (int) n_vocab_)) {
                    strata::kernels::native_q8_1_interleave(xq_, xil_, (int) N, (int) T, cs);
                    strata::kernels::native_mmvq_il(head_->type(), head_->weights(), xq_, xil_, head_logits_, (int) N,
                                                    (int) n_vocab_, (int) T, cs);
                } else
                native_mmvq(head_->type(), head_->weights(), xq_, head_logits_, (int) N, (int) n_vocab_, T, cs);
            } catch (const std::exception& e) {
                err = std::string("verify head: ") + e.what();
                return false;
            }
        }
        // Greedy, the default, is recorded here as before (no extra launch or sync per window). A request that
        // samples or penalizes is sampled again host-side after the replay (run()) with its own parameters and a
        // fresh draw counter: a captured sampler would bake them in and replay the same draws forever.
        if (argmax_rows_wanted()) {   // sm_80 to sm_89: the pick over many blocks a row (bitwise the one-block kernel's)
            argmax_rows(head_logits_, T, (int) n_vocab_, arg_scratch_, m_out_, cs);
        } else {
            SamplerParams sp;
            sp.greedy = true;
            sp.temperature = 0.0f;
            sample_tokens(head_logits_, T, (int) n_vocab_, nullptr, 0, sp, m_out_, cs);
        }
    }
    stamp(g.n_layers, 1, 0);
    return true;
}
std::string Verifier::profile_report() {
    if (!prof_on_ || prof_windows_ == 0) return std::string();
    static const char* names[kProfPer] = {"-", "hc-read0", "q8+qkv/q-idx gemv", "conv", "ab", "z", "rec", "q8+kv-idx",
                                          "k/v+norm-rope", "kv+idx append", "q+q-idx", "scores+topk", "kv-resolve",
                                          "attention", "gate", "", "out-proj", "hc-read1+router", "shared+quant",
                                          "waitA", "VRAM hits", "waitB", "PCIe grp", "waitCPU", "copy+combine",
                                          "(gap)", "head", "  hc0 norm", "  hc0 down", "  hc0 up", "", "", ""};
    std::string out;
    char b[80];
    double total = 0;
    for (int k = 0; k < 2; ++k) {
        out += k == 0 ? " GDN layers:" : " | QSA layers:";
        for (int i = 0; i < kProfPer; ++i) {
            if (prof_sum_[k][i] <= 0) continue;
            total += prof_sum_[k][i];
            std::snprintf(b, sizeof b, " %s %.2f", names[i], prof_sum_[k][i] / 1e6 / (double) prof_windows_);
            out += b;
        }
    }
    std::snprintf(b, sizeof b, " | total %.2f ms/window over %lld windows", total / 1e6 / (double) prof_windows_, (long long) prof_windows_);
    out += b;
    for (auto& r : prof_sum_) for (double& d : r) d = 0;
    prof_windows_ = 0;
    return out;
}
bool Verifier::capture(int T, std::string& err) {
    if (exec_[parity_][T] != nullptr) return true;
    const OnDevice on_device(device_);
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin capture failed";
        return false;
    }
    std::string rerr;
    const bool ok = record_window(T, cs_, rerr);
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        err = rerr;
        return false;
    }
    if (ce != cudaSuccess) {
        err = std::string("verify: end capture: ") + cudaGetErrorString(ce);
        return false;
    }
#if !defined(STRATA_USE_HIP)   // a CUDA debug listing (node types, kernel names)
    if (std::getenv("STRATA_VERIFY_NODES") != nullptr) {   // what the window graph holds
        size_t nn = 0;
        cudaGraphGetNodes(graph, nullptr, &nn);
        std::vector<cudaGraphNode_t> nodes(nn);
        cudaGraphGetNodes(graph, nodes.data(), &nn);
        std::map<std::string, int> kinds;
        for (cudaGraphNode_t nd : nodes) {
            cudaGraphNodeType ty;
            cudaGraphNodeGetType(nd, &ty);
            std::string name = "type" + std::to_string((int) ty);
            if (ty == cudaGraphNodeTypeKernel) {
                cudaKernelNodeParams kp{};
                if (cudaGraphKernelNodeGetParams(nd, &kp) == cudaSuccess) {
#if CUDART_VERSION >= 12030   // cudaFuncGetName arrived in CUDA 12.3
                    const char* fn = nullptr;
                    if (cudaFuncGetName(&fn, kp.func) == cudaSuccess && fn) name = fn;
#endif
                }
            } else if (ty == cudaGraphNodeTypeMemcpy) name = "memcpy";
            else if (ty == cudaGraphNodeTypeMemset) name = "memset";
            ++kinds[name];
        }
        std::vector<std::pair<int, std::string>> v;
        for (auto& [k2, c] : kinds) v.push_back({c, k2});
        std::sort(v.rbegin(), v.rend());
        std::fprintf(stderr, "strata verify: the %d-token window graph has %zu nodes:", T, nn);
        for (size_t i = 0; i < v.size() && i < 40; ++i) std::fprintf(stderr, " %d x %.60s;", v[i].first, v[i].second.c_str());
        std::fprintf(stderr, "\n");
    }
#endif
    const cudaError_t ie = cudaGraphInstantiate(&exec_[parity_][T], graph, 0);
    cudaGraphDestroy(graph);
    if (ie != cudaSuccess) {
        err = std::string("verify: instantiate: ") + cudaGetErrorString(ie);
        return false;
    }
    const cudaError_t ue = cudaGraphUpload(exec_[parity_][T], cs_);
    const cudaError_t us = cudaStreamSynchronize(cs_);
    std::fprintf(stderr, "strata verify: captured the %d-token window (upload %s, sync %s)\n", T,
                 cudaGetErrorString(ue), cudaGetErrorString(us));
    return true;
}
bool Verifier::capture_commit(std::string& err) {
    if (commit_exec_ != nullptr) return true;
    const OnDevice on_device(device_);
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const QsaShapes s = shapes_of(g);
    const int64_t C = g.ssm_conv_channels, HV = g.ssm_v_heads, ID = g.idx_key_dim, MT = max_t_;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t TS = (s.idx_block - 1) * ID;
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin commit capture failed";
        return false;
    }
    bool ok = true;
    try {
        copy_i32_from_mapped(commit_, m_commit_, 2 + MT, cs_);
        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < lb_; ++l) (is_qsa_layer(g, l) ? qsa_index : gdn_index) += 1;
        for (int64_t l = lb_; l < le_ && ok; ++l) {
            const LayerView v(*wt_, l);
            if (!is_qsa_layer(g, l)) {
                const WeightRef* wnm = need(v, "ssm_norm.weight", err);
                if (!wnm) { ok = false; break; }
                float* state = ss.gdn_state + (size_t) (gdn_index - ss.gdn_ord0) * gdn_floats;
                float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                const float* qkv = qkv_L_ + (size_t) gdn_index * MT * C;
                gdn_conv_commit(conv, qkv, (int) C, commit_, cs_);
                gdn_step_norm_multi(state, h_L_ + (size_t) gdn_index * MT * C, (int) C, gate_L_ + (size_t) gdn_index * MT * HV,
                                    beta_L_ + (size_t) gdn_index * MT * HV, z_, (const float*) wnm->data, EPS, y_dummy_,
                                    (int) g.ssm_k_heads, (int) HV, (int) MT, commit_, cs_);
                ++gdn_index;
            } else {
                const QsaState& st = ss.qsa_states[qsa_index];
                const WeightRef* wikn = need(v, "indexer.k_norm.weight", err);
                if (!wikn) { ok = false; break; }
                copy_from_mapped(st.idx_tail, tail_snap_ + (size_t) qsa_index * TS, TS, cs_);
                const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                for (int64_t t = 0; t < MT; ++t)
                    native_qsa_indexer_append(idx_raw_L_ + (size_t) (qsa_index * MT + t) * ID, commit_ + 2 + t, 0,
                                              (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                              rope_scaling(), cs_);
                ++qsa_index;
            }
        }
        if (ok && ss.ple.ready() && ple_stage()) copy_indexed(ss.ple.hist, hist_snap_, HS, commit_ + 1, HS, cs_);
    } catch (const std::exception& e) {
        err = std::string("verify commit: ") + e.what();
        ok = false;
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        return false;
    }
    if (ce != cudaSuccess || cudaGraphInstantiate(&commit_exec_, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("verify: commit capture: ") + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    return true;
}
bool Verifier::stage_inputs(int T, const int32_t* tokens, int64_t pos0, std::string& err) {
    using namespace strata::kernels;
    const OnDevice on_device(device_);
    if (T < 1 || T > max_t_ || (!batch_replay_ && T > strata::kernels::kVerifyMaxT)) {
        err = "verify: window size out of range"; return false;
    }
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    if (pos0 < 0 || pos0 + T > ss.qsa_states[ss.qsa_primary()].max_cells) { err = "verify: the window runs past the context"; return false; }
    const QsaShapes s = shapes_of(g);
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        qsa_step_fill(h_step_ + t * kStepCount, pos0 + t, s);
        for (int64_t h = 0; h < g.n_head; ++h) h_pos_[t * g.n_head + h] = (int32_t) (pos0 + t);
        int32_t* pk = h_pos_ + (size_t) max_t_ * g.n_head;
        int32_t* pi = pk + (size_t) max_t_ * g.n_head_kv;
        for (int64_t h = 0; h < g.n_head_kv; ++h) pk[t * g.n_head_kv + h] = (int32_t) (pos0 + t);
        for (int64_t h = 0; h < g.idx_q_heads; ++h) pi[t * g.idx_q_heads + h] = (int32_t) (pos0 + t);
    }
    if (ss.ple.ready() && ple_stage()) {
        uint32_t rows[kVerifyMaxT * PLE_N_HEADS];
        int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
        for (int t = 0; t < T; ++t) {
            ngram_rows(&tokens[t], prev, 1, ss.ple.consts, rows + t * PLE_N_HEADS);
            prev[0] = prev[1];
            prev[1] = tokens[t];
        }
        if (!ss.ple.table->gather_batch(rows, (size_t) T, h_ple_, err)) return false;
    }
    last_t_ = T;
    last_pos0_ = pos0;
    for (int t = 0; t < T; ++t) last_tokens_[t] = tokens[t];
    return true;
}
bool Verifier::run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, void* user, int32_t* out,
                   std::string& err) {
    using namespace strata::kernels;
    const OnDevice on_device(device_);
    g_diag_verifier.store(this);
    const ModelGeometry& g = *g_;
    const Clock::time_point t0 = Clock::now();
    if (T < 1 || T > max_t_ || (!batch_replay_ && T > strata::kernels::kVerifyMaxT)) {
        err = "verify: window size out of range"; return false;
    }
    if (released_.load()) { err = "verify: an earlier window never finished on the GPU (#267); restart the engine"; return false; }
    if (!batch_replay_ && (!capture(T, err) || !capture_commit(err) || !stage_inputs(T, tokens, pos0, err))) return false;
    last_t_ = T;
    last_pos0_ = pos0;
    if (!batch_replay_) for (int t = 0; t < T; ++t) last_tokens_[t] = tokens[t];
    ms_host += ms_since(t0);
    if (!launch_pass(T, nullptr, err)) return false;
    const int G = groups_[T] > 0 ? groups_[T] : 1;
    for (int64_t k = 0; k < (le_ - lb_) * G; ++k) {
        const int64_t l = lb_ + k / G;
        const uint32_t want = (uint32_t) (k + 1);
        const Clock::time_point a = Clock::now();
        auto last_flush = a;
        uint32_t spins = 0;
        progress_at("verify window: waiting for the GPU to reach layer", l);
        while (!pass_doorbell(want)) {
            _mm_pause();
            if ((++spins & 1023u) != 0) continue;
            if (!pass_stall(want, l, a, last_flush, err)) return false;
        }
        const Clock::time_point b = Clock::now();
        if (!service_one(k, pool, user, err)) return false;
        ms_wait += std::chrono::duration<double, std::milli>(b - a).count();
    }
    progress_at("verify window: waiting for the GPU to finish the window (flags A/B/M raised)", (int64_t) T);
    // #267: a window the GPU never finishes (a spin kernel that never sees its flag) holds the host here; the stall
    // watchdog then releases every verifier's GPU waits (release_live_verifiers) before it ends the engine, so no
    // spin kernel outlives the process - the case that left Windows GPUs "lost" until a power cycle.  The wait
    // itself stays a blocking sync: a cudaStreamQuery poll here cost IQ3_S ~3% decode (a core calling the driver
    // beside the expert workers).
    const cudaError_t se = cudaStreamSynchronize(cs_);
    if (se != cudaSuccess) { err = std::string("verify: ") + cudaGetErrorString(se); return false; }
    progress_at("verify window: waiting for the expert copies", (int64_t) T);
    // (sync invariant, rebased onto the pass API): only a fetch_dma(n > 0) of THIS
    // window can queue a copy-stream host function, so a window that queued none
    // leaves the copy stream drained.  launch_pass captured fetches_ at launch.
    if (fetches_.load(std::memory_order_relaxed) != pass_fetches_at_launch_)
        cudaStreamSynchronize(copy_);
    if (batch_replay_) { ++windows; progress_beat(); return true; }
    pass_prof_dump();
    if (le_ < g.n_layers) {   // a layer split's earlier stage: the hand-off is written (synced above)
        ++windows;
        return next_ == nullptr || next_->run(T, tokens, pos0, pool, next_user_, out, err);
    }
    VDBG("window done\n");
    if (!pass_finish_outputs(T, out, err)) return false;
    if (static const bool dbg = std::getenv("STRATA_DBG_NAN") != nullptr; dbg) {   // debug: the first non-finite head
        static bool reported = false;
        if (!reported) {
            std::vector<float> h((size_t) T * (size_t) n_vocab_);
            cudaMemcpy(h.data(), head_logits_, h.size() * 4, cudaMemcpyDeviceToHost);
            for (int t = 0; t < T && !reported; ++t) {
                int64_t bad = 0;
                for (int64_t v = 0; v < n_vocab_; ++v) bad += !std::isfinite(h[(size_t) t * n_vocab_ + v]);
                if (bad) {
                    reported = true;
                    std::fprintf(stderr, "strata dbg: verify window at position %lld, row %d: %lld of %lld logits non-finite "
                                         "(token out %d)\n", (long long) last_pos0_, t, (long long) bad, (long long) n_vocab_, out[t]);
                }
            }
        }
    }
    return true;
}
// C2: the shared batch-round preparation - validation, per-member staging and the graph
// lookup/capture.  run_batch (serial) and begin_pass_batch (stage-pipeline overlap) both call
// it; they differ only in what they do with the captured graph (run it here, or launch it as a
// pass and service it from drive_passes).
// graph-OOM fix: the MMQ pools' flush (defined in ggml_cuda_host.cu with C linkage)
extern "C" void strata_mmq_pool_flush_all();

bool Verifier::prepare_batch(const std::vector<BatchWindow>& batch, int& total,
                             std::vector<std::pair<Verifier*, int>>& shape, cudaGraphExec_t& graph_exec,
                             std::string& err) {
    using namespace strata::kernels;
    const OnDevice on_device(device_);
    const bool chained = le_ < g_->n_layers;   // this stage hands its rows to a next stage
    if (batch.empty() || batch.size() > 16 || split_) {
        err = "batch verify: requires 1..16 member windows"; return false;
    }
    // C2-A1: the chain must be CONTIGUOUS: the next stage starts exactly at this stage's le_,
    // same geometry, and only a last stage may be chained-free.  A missing/mis-ranged next
    // stage is an error, never a silent head-less run.
    if (chained ? (next_ == nullptr || next_->lb_ != le_ || next_->g_ != g_)
                : (next_ != nullptr)) {
        err = "batch verify: the stage chain is not contiguous"; return false;
    }
    assert(!chained || (next_->lb_ == le_ && next_->le_ <= g_->n_layers));   // belt; A1 above is operative
    total = 0;
    shape.clear();
    for (const auto& b : batch) {
        Verifier* v = b.verifier;
        const bool member_chained = v != nullptr && v->next_ != nullptr;
        if (!v || v == this || v->g_ != g_ || v->wt_ != wt_ || v->device_ != device_ || v->split_ ||
            v->lb_ != lb_ || v->le_ != le_ ||                       // THIS stage's range, this device
            member_chained != chained ||                            // a slot chains all its stages or none
            (member_chained && (v->next_->g_ != g_ || v->next_->lb_ != le_)) ||
            (lb_ > 0 && v->hand_in_ != hand_in_) ||                  // one boundary buffer, shared
            (chained && v->hand_out_ != hand_out_) ||
            b.count < 1 || b.count > v->max_t_ || !b.tokens || !b.output) {
            err = "batch verify: incompatible member"; return false;
        }
        for (const auto& previous : shape) if (previous.first == v || previous.first->ss_ == v->ss_) {
            err = "batch verify: duplicate sequence state"; return false;
        }
        total += b.count;
        shape.emplace_back(v, b.count);
    }
    // C2-A6: max_t_ = max(2, c.rows) <= the batch envelope (verify.cpp:207 caps a batch workspace at
    // MAXT = 48; kVerifyMaxT = 8, verify_kernels.hpp:21), and the POOL accepts any call whose n_tok * k
    // fits kMaxWindowEntries (480; src/core/expert_source.cpp:975, guards at :983 and :989) and
    // n_tok <= MAXT (48; include/strata/kernels/cpu/expert.hpp:138).  This model: k = ss.k = 10
    // => the full 48-row workspace envelope is legal (480 entries); the batch configs run --batch-rows 8..48.
    // The split hand-off buffer is max(kVerifyMaxT, batch_rows) rows per boundary (generate.cpp:3893) -
    // a 48-row round fits EXACTLY.  Rows beyond the envelope abort the engine LOUDLY at layer 0
    // (dispatch.failed -> exit 1).  A7's census prints rows= for every capture.
    if (total > max_t_) { err = "batch verify: row budget exceeded"; return false; }
    for (const auto& b : batch)
        if (!b.verifier->stage_inputs(b.count, b.tokens, b.position, err) || !b.verifier->capture_commit(err)) return false;
    // E-6: a layer whose routed experts are all resident is planned on the device
    device_plan_ = (skip_ != nullptr);
    graph_exec = nullptr;
    for (auto it = batch_graphs_.begin(); it != batch_graphs_.end(); ++it) if (it->shape == shape && it->parity == parity_) {
        graph_exec = it->graph;
        std::rotate(it, it + 1, batch_graphs_.end()); // bounded LRU, not creation-order eviction
        break;
    }
    if (!graph_exec) {
        const auto capture_start = Clock::now();
        struct CaptureGuard {
            cudaStream_t cs; bool open = false;
            ~CaptureGuard() { if (open) { cudaGraph_t g = nullptr; cudaStreamEndCapture(cs, &g); if (g) cudaGraphDestroy(g); } }
        } capture_guard{cs_};
        if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
            err = "batch verify: begin capture failed"; return false;
        }
        capture_guard.open = true;
        bool ok = true;
        const int64_t N = g_->n_embd, K = ss_->k;
        auto copy = [&](void* dst, const void* src, size_t bytes, cudaStream_t stream) {
            // Keep the doorbell graph kernel-only. WDDM can wait for a graph memcpy node while
            // holding the driver lock needed by the host that must service its preceding doorbell.
            copy_i32_from_mapped((int32_t*) dst, (const int32_t*) src, (int64_t) (bytes / 4), stream);
        };
        // C2-A4: row0(member i) = sum of counts[j], j < i — a pure function of the shape list,
        // identical in both stages' captures by construction.
        int row_in = 0;
        for (const auto& b : batch) if (ok) {
            ok = b.verifier->record_window(b.count, cs_, err, 0, 0, row_in);
            row_in += b.count;
        }
        if (ok && row_in != total) { err = "batch verify: row mapping broken (in)"; return false; }
        for (int64_t l = lb_; ok && l < le_; ++l) {          // C2: this stage's layers only
            if (prof_on_) gpu_stamp(prof_, (int) (l * kProfPer + 27), cs_);
            if (batch_parallel_ && cudaEventRecord(batch_fork_, cs_) != cudaSuccess) {
                err = "batch verify: fork failed"; ok = false; break;
            }
            int row = 0;
            int member = 0;
            for (const auto& b : batch) {
                Verifier& v = *b.verifier;
                cudaStream_t branch = batch_parallel_ ? v.cs_ : cs_;
                if (batch_parallel_ && cudaStreamWaitEvent(branch, batch_fork_, 0) != cudaSuccess) {
                    err = "batch verify: branch wait failed"; ok = false; break;
                }
                if (!(ok = v.record_window(b.count, branch, err, 1, l))) break;
                copy(mixed_ + row * N, v.mixed_, (size_t) b.count * N * sizeof(float), branch);
                copy(ids_ + row * K, v.ids_, (size_t) b.count * K * sizeof(int32_t), branch);
                copy(w_ + row * K, v.w_, (size_t) b.count * K * sizeof(float), branch);
                if (batch_parallel_ && cudaEventRecord(batch_join_[member], branch) != cudaSuccess) {
                    err = "batch verify: branch join failed"; ok = false; break;
                }
                ++member;
                row += b.count;
            }
            if (batch_parallel_) for (int i = 0; i < member; ++i)
                if (cudaStreamWaitEvent(cs_, batch_join_[i], 0) != cudaSuccess) {
                    err = "batch verify: join wait failed"; ok = false;
                }
            if (!ok) break;
            if (prof_on_) gpu_stamp(prof_, (int) (l * kProfPer + 28), cs_);
            doorbell_publish(mixed_, ids_, w_, total * N, total * K, m_x_, m_ids_, m_w_, m_seq_, cs_);
            if (device_plan_) {
                resident_plan(ids_, total * (int) K, (int) K, hits_.d_res + l * g_->n_expert, (int) g_->n_expert,
                              hits_.cache_base, slot_off_d_, (long long) hits_.blob,
                              plan_, (long long) max_t_ * K, skip_, (uint32_t) (l - lb_ + 1), cs_);
            }
            if (strata::kernels::cpu::expert_layout().native)
                quantize_q8_1_rows(mixed_, total, N, nat_xq_, cs_);
            else { err = "batch verify: requires native experts"; ok = false; break; }
            if (!(ok = record_window(total, cs_, err, 2, l))) break;
            if (prof_on_) gpu_stamp(prof_, (int) (l * kProfPer + 29), cs_);
            if (batch_parallel_ && cudaEventRecord(batch_fork_, cs_) != cudaSuccess) {
                err = "batch verify: fork (phase 3) failed"; ok = false; break;
            }
            row = 0;
            int member3 = 0;
            for (const auto& b : batch) {
                Verifier& v = *b.verifier;
                cudaStream_t branch = batch_parallel_ ? v.cs_ : cs_;
                if (batch_parallel_ && cudaStreamWaitEvent(branch, batch_fork_, 0) != cudaSuccess) {
                    err = "batch verify: branch wait (phase 3) failed"; ok = false; break;
                }
                copy(v.parts_, parts_ + row * K * N, (size_t) b.count * K * N * sizeof(float), branch);
                if (!ok || !(ok = v.record_window(b.count, branch, err, 3, l))) break;
                if (batch_parallel_ && cudaEventRecord(batch_join_[member3], branch) != cudaSuccess) {
                    err = "batch verify: branch join (phase 3) failed"; ok = false; break;
                }
                ++member3;
                row += b.count;
            }
            if (batch_parallel_) for (int i = 0; i < member3; ++i)
                if (cudaStreamWaitEvent(cs_, batch_join_[i], 0) != cudaSuccess) {
                    err = "batch verify: join wait (phase 3) failed"; ok = false;
                }
            if (prof_on_) gpu_stamp(prof_, (int) (l * kProfPer + 30), cs_);
        }
        if (prof_on_) gpu_stamp(prof_, (int) (g_->n_layers * kProfPer + 2), cs_);
        // C2-A4: the same cumulative offsets as the capture head; chained stages hand the rows
        // on (phase 5), the last stage runs the head (phase 4).
        int row_out = 0;
        static const bool batch_head = [] {
            const char* v = std::getenv("STRATA_BATCH_HEAD");
            return v == nullptr || std::atoi(v) != 0;
        }();
        if (!chained && head_ != nullptr && head_->loaded() && batch_head) {
            for (const auto& b : batch) if (ok) {
                ok = b.verifier->record_window(b.count, cs_, err, 40);
                copy(head_mixed_ + (size_t) row_out * N, b.verifier->head_mixed_,
                     (size_t) b.count * N * sizeof(float), cs_);
                row_out += b.count;
            }
            if (ok && row_out == total) {
                try {
                    quantize_q8_1_rows(head_mixed_, total, N, nat_xq_, cs_);
                    for (int r = 0; r < total; r += 8) {
                        const int cur = std::min(8, total - r);
                        const void* x_chunk = (const uint8_t*) nat_xq_ + (size_t) r * (N / 32) * 36;
                        float* log_chunk = head_logits_ + (size_t) r * n_vocab_;
                        native_mmvq(head_->type(), head_->weights(), x_chunk, log_chunk,
                                    (int) N, (int) n_vocab_, cur, cs_);
                    }
                    SamplerParams sp;
                    sp.greedy = true;
                    sp.temperature = 0.0f;
                    int r_member = 0;
                    for (const auto& b : batch) {
                        sample_tokens(head_logits_ + (size_t) r_member * n_vocab_, b.count,
                                      (int) n_vocab_, nullptr, 0, sp, b.verifier->m_out_, cs_);
                        if (b.verifier->head_sampling_ && b.verifier->head_logits_ != nullptr) {
                            copy(b.verifier->head_logits_, head_logits_ + (size_t) r_member * n_vocab_,
                                 (size_t) b.count * n_vocab_ * sizeof(float), cs_);
                        }
                        r_member += b.count;
                    }
                } catch (const std::exception& e) {
                    err = std::string("verify head batch: ") + e.what();
                    ok = false;
                }
            } else if (!ok) {
                // err already set
            } else {
                err = "batch verify: row mapping broken (head)";
                ok = false;
            }
        } else {
            for (const auto& b : batch) if (ok) {
                ok = chained ? b.verifier->record_window(b.count, cs_, err, 5, 0, row_out)
                             : b.verifier->record_window(b.count, cs_, err, 4);
                row_out += b.count;
            }
            if (ok && row_out != total) { err = "batch verify: row mapping broken (out)"; return false; }
        }
        if (prof_on_) gpu_stamp(prof_, (int) (g_->n_layers * kProfPer + 3), cs_);
        cudaGraph_t graph = nullptr;
        const cudaError_t end = cudaStreamEndCapture(cs_, &graph);
        capture_guard.open = false;           // disarm before any early return below
        if (!ok || end != cudaSuccess) {
            if (graph) cudaGraphDestroy(graph);
            if (err.empty()) err = "batch verify: end capture failed";
            return false;
        }
        if (std::getenv("STRATA_VERIFY_NODES") != nullptr) {   // C2-A7: batch hand-off census
            // Prove the capture is wired for THIS stage: a chained stage wrote its packed rows to
            // hand_out (3 copy_from_mapped nodes per row: R, bo, inj), a later stage read them from
            // hand_in, and a full-range coordinator has neither.  "No head on a chained stage" is the
            // A4c either/or tail; the phase-5 count is its evidence.
            size_t nn = 0;
            cudaGraphGetNodes(graph, nullptr, &nn);
            std::vector<cudaGraphNode_t> nodes(nn);
            cudaGraphGetNodes(graph, nodes.data(), &nn);
            const uintptr_t hb = (uintptr_t) Verifier::handoff_floats(*g_);
            const uintptr_t span = (uintptr_t) total * hb * sizeof(float);
            const uintptr_t hin = (uintptr_t) hand_in_, hout = (uintptr_t) hand_out_;
            int in_nodes = 0, out_nodes = 0, named = 0;
#if CUDART_VERSION >= 12030   // cudaFuncGetName arrived in CUDA 12.3
            for (cudaGraphNode_t nd : nodes) {
                cudaGraphNodeType ty;
                cudaGraphNodeGetType(nd, &ty);
                if (ty != cudaGraphNodeTypeKernel) continue;
                cudaKernelNodeParams kp{};
                if (cudaGraphKernelNodeGetParams(nd, &kp) != cudaSuccess || kp.func == nullptr) continue;
                const char* fn = nullptr;
                if (cudaFuncGetName(&fn, kp.func) != cudaSuccess || fn == nullptr) continue;
                if (std::strstr(fn, "copy_from_mapped") == nullptr) continue;   // not copy_i32_from_mapped
                ++named;
                const char* dst = kp.kernelParams ? *(const char* const*) kp.kernelParams[0] : nullptr;
                const char* src = kp.kernelParams ? *(const char* const*) kp.kernelParams[1] : nullptr;
                if (hout != 0 && (uintptr_t) dst >= hout && (uintptr_t) dst < hout + span) ++out_nodes;
                if (hin != 0 && (uintptr_t) src >= hin && (uintptr_t) src < hin + span) ++in_nodes;
            }
#endif
            std::fprintf(stderr, "strata verify: batch capture stage [%lld,%lld) rows=%d hand_out=%d "
                                 "hand_in=%d (%d copy_from_mapped nodes)\n",
                         (long long) lb_, (long long) le_, total, out_nodes, in_nodes, named);
#if CUDART_VERSION >= 12030
            std::string& e2 = err;   // fail the round loudly on a wiring violation, never silently
            if (chained && (out_nodes != 3 * total || in_nodes != 0)) {
                e2 = "batch verify: stage [" + std::to_string(lb_) + "," + std::to_string(le_) +
                     ") captured no complete hand-off out (" + std::to_string(out_nodes) + " of " +
                     std::to_string(3 * total) + " copy_from_mapped nodes)";
                cudaGraphDestroy(graph); return false;
            }
            if (!chained && lb_ > 0 && (in_nodes != 3 * total || out_nodes != 0)) {
                e2 = "batch verify: stage [" + std::to_string(lb_) + "," + std::to_string(le_) +
                     ") captured no complete hand-in (" + std::to_string(in_nodes) + " of " +
                     std::to_string(3 * total) + " copy_from_mapped nodes)";
                cudaGraphDestroy(graph); return false;
            }
            if (!chained && lb_ == 0 && (in_nodes != 0 || out_nodes != 0)) {
                e2 = "batch verify: a full-range coordinator captured hand-off copies";
                cudaGraphDestroy(graph); return false;
            }
#endif
        }
        cudaError_t instantiate = cudaGraphInstantiate(&graph_exec, graph, 0);
        if (instantiate == cudaErrorMemoryAllocation) {
            // graph-OOM fix v1: free the MMQ pools' unused scratch and retry once - the instantiate
            // needs only a few MB, the pools hold the surplus (measured 2026-10-07).
            strata_mmq_pool_flush_all();
            cudaGetLastError();
            instantiate = cudaGraphInstantiate(&graph_exec, graph, 0);
        }
        if (instantiate != cudaSuccess) {
            // graph-fix S1: name the real cause (the boolean alone hid it for the 12288 breach
            // and the 3x262K load alike)
            size_t n_nodes = 0, n_edges = 0;
            cudaGraphGetNodes(graph, nullptr, &n_nodes);
            cudaGraphGetEdges(graph, nullptr, nullptr, &n_edges);
            std::fprintf(stderr, "batch verify: graph instantiation failed: %s (nodes=%zu edges=%zu "
                                 "lb=%lld le=%lld)\n",
                         cudaGetErrorString(instantiate), n_nodes, n_edges, (long long) lb_, (long long) le_);
        }
        cudaGraphDestroy(graph);
        if (instantiate != cudaSuccess) { err = "batch verify: graph instantiation failed"; return false; }
        if (cudaGraphUpload(graph_exec, cs_) != cudaSuccess || cudaStreamSynchronize(cs_) != cudaSuccess) {
            cudaGraphExecDestroy(graph_exec);
            err = "batch verify: graph upload failed"; return false;
        }
        // Bound graph memory when confidence/suffix windows produce many layouts.
        size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
            cudaGraphExecDestroy(graph_exec);
            err = "batch verify: cannot query free VRAM"; return false;
        }
        const auto reserve_bytes = (size_t) std::min<int>(batch_reserve_mib_, 16) * 1048576;
        while (!batch_graphs_.empty() && (batch_graphs_.size() >= (size_t) batch_cache_limit_ || free_bytes < reserve_bytes)) {
            cudaGraphExecDestroy(batch_graphs_.front().graph);
            batch_graphs_.erase(batch_graphs_.begin());
            if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
                cudaGraphExecDestroy(graph_exec);
                err = "batch verify: cannot query free VRAM"; return false;
            }
        }
        // LANE failclean (test hook, default off): STRATA_TEST_REFUSE_CAPTURE is a comma list of
        // 1-based capture ordinals to refuse with the reserve message - the falsification hook for
        // the clean-fail path that needs no VRAM pressure.  Ordinals count every UNCACHED batch
        // capture of the process: "1" refuses the first (stage-0 of the first batch unit), "2" the
        // second (that unit's stage-1 capture), "1,3" a stage-0 refusal then a later stage-1 one.
        // The refusal keeps the real message so the callers' matching and bookkeeping are verbatim.
        static int test_capture_ordinal = 0;
        static const char* test_refuse_list = std::getenv("STRATA_TEST_REFUSE_CAPTURE");
        ++test_capture_ordinal;
        bool test_refuse = false;
        if (test_refuse_list != nullptr) {
            const std::string list = std::string(",") + test_refuse_list + ",";
            test_refuse = list.find("," + std::to_string(test_capture_ordinal) + ",") != std::string::npos;
        }
        if (free_bytes < reserve_bytes) {
            // graph-OOM fix v2: the guard needs only ~16 MiB; the MMQ pools hold surplus scratch
            // (their unused buffers), so flush them and re-read before refusing a request the box
            // can actually hold (measured: the 3rd slot failed at 11 MiB of 16).
            strata_mmq_pool_flush_all();
            if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
                cudaGraphExecDestroy(graph_exec);
                err = "batch verify: cannot query free VRAM"; return false;
            }
        }
        if (free_bytes < reserve_bytes || test_refuse) {
            if (test_refuse)
                std::fprintf(stderr, "strata verify: TEST hook refused batch capture #%d (STRATA_TEST_REFUSE_CAPTURE=%s)\n",
                             test_capture_ordinal, test_refuse_list);
            cudaGraphExecDestroy(graph_exec);
            err = "batch verify: graph leaves " + std::to_string(free_bytes >> 20) +
                  " MiB free, below VRAM reserve; reduce expert cache/context";
            return false;
        }
        batch_graphs_.push_back({std::move(shape), parity_, graph_exec});
        ms_batch_capture += ms_since(capture_start);
        ++batch_captures;
    }
    return true;
}
bool Verifier::run_batch(const std::vector<BatchWindow>& batch, PoolMultiFn pool, void* user, std::string& err) {
    using namespace strata::kernels;
    const OnDevice on_device(device_);
    const bool chained = le_ < g_->n_layers;   // this stage hands its rows to a next stage
    int total = 0;
    std::vector<std::pair<Verifier*, int>> shape;
    cudaGraphExec_t graph_exec = nullptr;
    if (!prepare_batch(batch, total, shape, graph_exec, err)) return false;
    groups_[total] = 1;
    batch_replay_ = graph_exec;
    const bool ran = run(total, nullptr, 0, pool, user, nullptr, err);
    batch_replay_ = nullptr;                                             // reset BEFORE any chain call
    if (!ran) return false;
    for (const auto& b : batch) ++b.verifier->windows;                   // this stage's members ran
    if (prof_on_) {
        cudaMemcpy(prof_h_.data(), prof_, prof_h_.size() * 8, cudaMemcpyDeviceToHost);
        for (int64_t l = 0; l < g_->n_layers; ++l)
            for (int stage = 0; stage < 3; ++stage)
                batch_gpu_ms[stage] += (double) (prof_h_[(size_t) l * kProfPer + 28 + stage] -
                                                 prof_h_[(size_t) l * kProfPer + 27 + stage]) / 1e6;
        batch_gpu_ms[3] += (double) (prof_h_[(size_t) g_->n_layers * kProfPer + 3] -
                                     prof_h_[(size_t) g_->n_layers * kProfPer + 2]) / 1e6;
    }
    if (chained) {
        // C2-A5: the hand-off was written into mapped memory and this stage's stream was synced by
        // run() above; the next stage replays the SAME rows in the SAME order.  Outputs are the
        // last stage's; sampling settings and final_R already chain through set_next().
        std::vector<BatchWindow> chain_batch;
        chain_batch.reserve(batch.size());
        for (const auto& b : batch)
            chain_batch.push_back({b.verifier->next_, b.count, b.tokens, b.position, b.output});
        // belt: the derivation is positional — same size/order/counts, the member's own next stage
        assert(chain_batch.size() == batch.size());
        for (size_t i = 0; i < batch.size(); ++i) {
            assert(chain_batch[i].verifier == batch[i].verifier->next_ &&
                   chain_batch[i].count == batch[i].count &&
                   chain_batch[i].verifier->lb_ == le_);                 // contiguity, per member
        }
        (void) 0;
        return next_->run_batch(chain_batch, pool, next_user_, err);
    }
    // ---- the last stage: the head ran here (members' phase 4) — sampling and outputs
    for (const auto& b : batch) {
        Verifier& v = *b.verifier;
        if (v.head_sampling_ && v.head_logits_ != nullptr && ((!v.sampling_.greedy && v.sampling_.temperature >= 1e-4f) || v.hist_d_)) {
            auto sp = v.sampling_;
            sp.counter = (uint64_t) b.position;
            sample_tokens(v.head_logits_, b.count, (int) v.n_vocab_, v.hist_d_, v.hist_len_, sp, v.m_out_, cs_);
        }
    }
    if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = "batch verify: sampling failed"; return false; }
    for (const auto& b : batch)
        for (int t = 0; t < b.count; ++t) b.output[t] = b.verifier->h_out_[t];
    progress_at("decode");
    progress_beat();
    return true;
}
void Verifier::set_plan_slot(int grp) {
    const int64_t cap = sink_.cap;
    int32_t* base = h_plan_ + (size_t) grp * (size_t) plan_i32_;
    const int64_t i32 = 4 + (cap + 1) + cap + cap;
    const int64_t ptr_off = (i32 + 1) & ~1ll;
    sink_.counts = base;
    sink_.start = base + 4;
    sink_.dst = sink_.start + cap + 1;
    sink_.tok = sink_.dst + cap;
    sink_.ptr = (unsigned long long*) (base + ptr_off);
    sink_.ptr2 = sink_.ptr + cap;
    sink_.start2 = base + ptr_off + 4 * cap;
    const int G = groups_[last_t_] > 0 ? groups_[last_t_] : 1;
    const int64_t per = G == 2 ? kStagingBlobs / 2 : kStagingBlobs;
    sink_.staging = (unsigned long long) (staging_ + (size_t) (grp * per) * strata::kernels::cpu::expert_layout().max_blob);
    sink_.staging_cap = per;
}
// Flag B only rises: a host function of an earlier layer may run after a later layer already raised it directly.
void Verifier::raise_flag(uint32_t* flag, uint32_t value) {
    volatile long* f = (volatile long*) flag;
#if defined(_WIN32)
    long cur = *f;
    while ((uint32_t) cur < value) {
        const long prev = _InterlockedCompareExchange(f, (long) value, cur);
        if (prev == cur) break;
        cur = prev;
    }
#else
    uint32_t cur = __atomic_load_n((uint32_t*) flag, __ATOMIC_SEQ_CST);
    while (cur < value && !__atomic_compare_exchange_n((uint32_t*) flag, &cur, value, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {}
#endif
}
// Plan v0.3 P6: the PCIe share by DMA.  The copy engine moves the blobs while the CPU computes its own share and the
// GPU its VRAM experts; a host function raises flag B when they have landed (the graph waits for it before the PCIe
// groups).  Staging is split between the two token groups of a split window.
void Verifier::fetch_dma(void* ctx, const uint8_t* const* src, int n, size_t bytes) {
    Verifier* v = (Verifier*) ctx;
    const uint32_t want = v->cur_layer_ + 1;
    if (n <= 0) { raise_flag(v->h_flagB_, want); return; }
    v->fetches_.fetch_add(1, std::memory_order_relaxed);
    uint8_t* stage = (uint8_t*) v->sink_.staging;                  // this group's half in a split window
    for (int i = 0; i < n; ++i) cudaMemcpyAsync(stage + (size_t) i * bytes, src[i], bytes, cudaMemcpyHostToDevice, v->copy_);
    FlagSet& fs = v->flag_sets_[v->cur_layer_ % (sizeof v->flag_sets_ / sizeof v->flag_sets_[0])];
    fs.flag = v->h_flagB_;
    fs.value = want;
    cudaLaunchHostFunc(v->copy_, [](void* p) { FlagSet* s = (FlagSet*) p; raise_flag(s->flag, s->value); }, &fs);
}
void Verifier::publish_plan(void* ctx) {
    Verifier* v = (Verifier*) ctx;
    _mm_sfence();
    *(volatile uint32_t*) v->h_flagA_ = v->cur_layer_ + 1;
}
// LANE hostloop: commit() is split so callers can LAUNCH every slot's commit chain first and wait
// afterwards (the per-slot chains then run concurrently instead of one-slot-at-a-time).  The
// contract is unchanged for the historical wrapper: commit() == commit_launch() + commit_wait().
bool Verifier::commit_launch(int n_keep, std::string& err) {
    const OnDevice on_device(device_);
    if (n_keep < 1 || n_keep > last_t_) { err = "verify: commit count out of range"; return false; }
    h_commit_[0] = n_keep;
    h_commit_[1] = n_keep - 1;
    for (int t = 0; t < max_t_; ++t) h_commit_[2 + t] = t < n_keep ? (int32_t) (last_pos0_ + t) : -1;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const cudaError_t le = cudaGraphLaunch(commit_exec_, cs_);
    if (le != cudaSuccess) { err = std::string("verify: commit launch: ") + cudaGetErrorString(le); return false; }
    if (ple_stage())   // stages that share one session must advance it once
        for (int t = 0; t < n_keep; ++t) {
            ss_->ple_prev[0] = ss_->ple_prev[1];
            ss_->ple_prev[1] = last_tokens_[t];
        }
    // A non-tail stage's commit graph writes only that stage's own session state; its consumers are
    // the next launch on the SAME stream (ordered without a host wait) and nothing on the host.
    if (next_ != nullptr) return next_->commit_launch(n_keep, err);
    return true;
}
// Only the CHAIN TAIL waits.  The tail's wait is the caller's dependency: the drafter's own stream
// reads the last stage's committed state next (ms_commit now measures the tail WAIT only).
bool Verifier::commit_wait(std::string& err) {
    if (next_ != nullptr) return next_->commit_wait(err);
    const Clock::time_point t0 = Clock::now();
    const cudaError_t se = cudaStreamSynchronize(cs_);
    if (se != cudaSuccess) { err = std::string("verify: commit: ") + cudaGetErrorString(se); return false; }
    ms_commit += ms_since(t0);
    return true;
}
bool Verifier::commit(int n_keep, std::string& err) {
    return commit_launch(n_keep, err) && commit_wait(err);
}
// ============================ STAGE-PIPELINE OVERLAP (lane overlap) ============================
//
// The serial round runs stage0's pass to completion (host loop, cs_ sync), then stage1's, and the
// two stages of a round are therefore ~serial: with the round split 24/24 the two halves each cost
// ~half the verify phase.  With the hand-off DOUBLE-BUFFERED BY PARITY the two passes of
// CONSECUTIVE units can be in flight at once: stage0[N+1] writes parity (N+1)&1 while stage1[N]
// still reads parity N&1, and stage1[N]'s launch is gated on stage0[N]'s completion event only.
// The host services BOTH passes' doorbells (drive_passes) because each pass's expert dispatch runs
// on the host thread; everything else - captures, plan publication, flags, the stall check - is the
// same machinery the serial path uses (service_one, pass_stall).  Correctness notes:
//   * a slot is eligible for the next unit only once its previous unit's EPILOGUE ran (its window
//     tokens are the epilogue's drafts), so the two units in flight own disjoint slots;
//   * the host must not stage a verifier's mapped inputs or reset its flags while that verifier's
//     graph is still executing, so begin_pass_* follows the pass's completion event (the driver
//     waits it) - see concurrent_serve.cpp's pipeline loop;
//   * the captured graphs bake the ACTIVE hand-off pointers, hence exec_[parity][T] and the
//     parity key of batch_graphs_.
bool Verifier::launch_pass(int T, cudaEvent_t done, std::string& err) {
    *(volatile uint32_t*) h_seq_ = 0;
    *(volatile uint32_t*) h_flag_ = 0;
    *(volatile uint32_t*) h_flagA_ = 0;
    *(volatile uint32_t*) h_flagB_ = 0;
    cur_layer_ = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    VDBG("staged; launching\n");
    // (sync invariant, rebased): the launch-time fetch count; end_pass_* syncs copy_
    // only when this pass queued a fetch_dma(n > 0) after it.
    pass_fetches_at_launch_ = fetches_.load(std::memory_order_relaxed);
    const cudaError_t le = cudaGraphLaunch(batch_replay_ ? batch_replay_ : exec_[parity_][T], cs_);
    if (le != cudaSuccess) { err = std::string("verify: launch: ") + cudaGetErrorString(le); return false; }
    // R2 (split1): NO host CUDA call lands between this launch and the first serviced doorbell.
    // The discarded cudaStreamQuery(cs_) that sat here, and recording the pass completion event on
    // the in-flight stream, were the host's driver touches of the stage hand-off window: on a
    // two-device split the p2 probe parked on a driver lock in exactly this span ("graphed"
    // without "recorded", the host thread spinning inside the driver - the #31 class: the host
    // inside a CUDA call while the GPU spins on a flag the host must raise).  The query is removed
    // (its result was discarded); the event moves to where the pass ends and its stream was waited
    // out (end_pass_window/end_pass_batch).  Its consumers only gate on "the pass completed", and
    // host order establishes that no later than those points, so the meaning is unchanged.
    // v3 (split1-r2fix): the deferred record is legal because the event now lives in the recording
    // stream's own context - ev0/ev1 are created on their own stage's device (concurrent_serve.cpp).
    deferred_done_ = done;
    VDBG("launched\n");
    return true;
}
bool Verifier::service_one(int64_t k, PoolMultiFn pool, void* user, std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const Clock::time_point b = Clock::now();
    const int G = groups_[last_t_] > 0 ? groups_[last_t_] : 1;
    const int64_t l = lb_ + k / G;
    const int grp = (int) (k % G);
    const uint32_t want = (uint32_t) (k + 1);
    const int T = last_t_;
    const int64_t steps = (le_ - lb_) * G;
    const bool test_stall = g_test_stall > 0 && windows + 1 == g_test_stall;   // #267 test hook (off: false)
    const int gtb[2] = {0, (T + 1) / 2}, gte[2] = {G == 2 ? (T + 1) / 2 : T, T};
    VDBG("layer %lld rang\n", (long long) l);
    cur_layer_ = want - 1;
    set_plan_slot(grp);
    const int tb = gtb[grp], n = gte[grp] - gtb[grp];
    progress_at("verify window: the CPU experts of layer", l);
    if (pool != nullptr)
        pool(user, h_x_ + (size_t) tb * g.n_embd, h_ids_ + (size_t) tb * ss.k, n, ss.k,
             h_ymiss_ + (size_t) tb * ss.k * g.n_embd, l);
    VDBG("layer %lld served\n", (long long) l);
    progress_tick();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    _mm_sfence();
    if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
        sink_.counts[0] = 0;
        sink_.counts[1] = 0;
        sink_.counts[2] = 0;
        sink_.start[0] = 0;
        sink_.start2[0] = 0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        *(volatile uint32_t*) h_flagA_ = want;
        raise_flag(h_flagB_, want);
    }
    if (!(test_stall && k + 1 == steps)) *(volatile uint32_t*) h_flag_ = want;
    ms_pool += ms_since(b);
    (void) err;
    return true;
}
bool Verifier::pass_stall(uint32_t want, int64_t l, Clock::time_point wait_start, Clock::time_point& last_flush,
                          std::string& err) {
    const auto now = Clock::now();
    // Latency optimization: avoid frequent cudaStreamQuery calls that acquire driver locks
    // during fast graph execution. Check every 250ms (or STRATA_PASS_STALL_MS) instead of 2ms.
    static const int64_t stall_interval_us = [] {
        const char* v = std::getenv("STRATA_PASS_STALL_MS");
        return (v ? std::max(10, std::atoi(v)) : 250) * 1000;
    }();
    if (now - last_flush > std::chrono::microseconds(stall_interval_us)) {
        last_flush = now;
        const cudaError_t q = cudaStreamQuery(cs_);
        if (q != cudaErrorNotReady && !pass_doorbell(want)) {
            err = "verify: layer " + std::to_string(l) + " never rang (" +
                  (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
            return false;
        }
    }
    if (now - wait_start > std::chrono::seconds(20)) {
        // #267: the caller ends the engine; no spin kernel may outlive it
        err = "verify: timed out at layer " + std::to_string(l) + released_note(release_gpu_waits(5000));
        return false;
    }
    return true;
}
bool Verifier::begin_pass_window(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, void* user,
                                 cudaEvent_t done, std::string& err) {
    const OnDevice on_device(device_);
    g_diag_verifier.store(this);
    const Clock::time_point t0 = Clock::now();
    if (T < 1 || T > max_t_ || T > strata::kernels::kVerifyMaxT) {
        err = "verify: window size out of range"; return false;
    }
    if (released_.load()) { err = "verify: an earlier window never finished on the GPU (#267); restart the engine"; return false; }
    if (!capture(T, err) || !capture_commit(err) || !stage_inputs(T, tokens, pos0, err)) return false;
    last_t_ = T;
    last_pos0_ = pos0;
    for (int t = 0; t < T; ++t) last_tokens_[t] = tokens[t];
    ms_host += ms_since(t0);
    if (!launch_pass(T, done, err)) return false;
    pass_live_ = true;
    pass_k_ = 0;
    const int G = groups_[T] > 0 ? groups_[T] : 1;
    pass_total_ = (le_ - lb_) * G;
    pass_pool_ = pool;
    pass_user_ = user;
    pass_wait_start_ = Clock::now();
    pass_last_flush_ = pass_wait_start_;
    pass_wait_reported_ = false;
    return true;
}
bool Verifier::begin_pass_batch(const std::vector<BatchWindow>& batch, PoolMultiFn pool, void* user,
                                cudaEvent_t done, std::string& err) {
    const OnDevice on_device(device_);
    g_diag_verifier.store(this);
    if (released_.load()) { err = "verify: an earlier window never finished on the GPU (#267); restart the engine"; return false; }
    int total = 0;
    std::vector<std::pair<Verifier*, int>> shape;
    cudaGraphExec_t graph_exec = nullptr;
    if (!prepare_batch(batch, total, shape, graph_exec, err)) return false;
    groups_[total] = 1;
    batch_replay_ = graph_exec;
    last_t_ = total;
    last_pos0_ = 0;
    const bool ok = launch_pass(total, done, err);
    batch_replay_ = nullptr;
    if (!ok) return false;
    pass_live_ = true;
    pass_k_ = 0;
    pass_total_ = le_ - lb_;                 // a batch pass runs one group (groups_[total] = 1)
    pass_pool_ = pool;
    pass_user_ = user;
    pass_wait_start_ = Clock::now();
    pass_last_flush_ = pass_wait_start_;
    pass_wait_reported_ = false;
    return true;
}
int Verifier::pass_step(std::string& err) {
    if (!pass_live_ || pass_k_ >= pass_total_) return 0;
    const int G = groups_[last_t_] > 0 ? groups_[last_t_] : 1;
    const int64_t l = lb_ + pass_k_ / G;
    const uint32_t want = (uint32_t) (pass_k_ + 1);
    if (!pass_doorbell(want)) {
        if (!pass_wait_reported_) {
            progress_at("verify window: waiting for the GPU to reach layer", l);
            pass_wait_reported_ = true;
        }
        if (!pass_stall(want, l, pass_wait_start_, pass_last_flush_, err)) return -1;
        return 0;
    }
    const Clock::time_point b = Clock::now();
    if (!service_one(pass_k_, pass_pool_, pass_user_, err)) return -1;
    ms_wait += std::chrono::duration<double, std::milli>(b - pass_wait_start_).count();
    ++pass_k_;
    pass_wait_start_ = Clock::now();
    pass_last_flush_ = pass_wait_start_;
    pass_wait_reported_ = false;
    return 1;
}
bool Verifier::drive_passes(Verifier* a, Verifier* b, std::string& err) {
    // LANE sched-impl P3 (host-loop O3 counter-only): count consecutive no-progress iterations per
    // call. Default off (STRATA_DRIVE_HISTO unset = old path, one getenv per call); when on, two ALU
    // ops per idle iteration plus one stderr line per 512 calls. No wait-posture or logic change.
    Verifier* acct = a != nullptr ? a : b;
    const bool counting = acct != nullptr && std::getenv("STRATA_DRIVE_HISTO") != nullptr;
    uint64_t burst = 0;
    if (counting) ++acct->drive_calls;
    for (;;) {
        const bool done_a = a == nullptr || a->pass_finished();
        const bool done_b = b == nullptr || b->pass_finished();
        if (done_a && done_b) {
            if (counting && acct->drive_calls % 512 == 0)
                std::fprintf(stderr, "strata drive idle: calls=%llu iters=%llu maxburst=%llu (STRATA_DRIVE_HISTO)\n",
                             (unsigned long long) acct->drive_calls, (unsigned long long) acct->drive_idle_iters,
                             (unsigned long long) acct->drive_idle_maxburst);
            return true;
        }
        int progressed = 0;
        if (!done_a) { const int r = a->pass_step(err); if (r < 0) return false; progressed += r; }
        if (!done_b) { const int r = b->pass_step(err); if (r < 0) return false; progressed += r; }
        if (!progressed) {
            if (counting) {
                ++acct->drive_idle_iters;
                if (++burst > acct->drive_idle_maxburst) acct->drive_idle_maxburst = burst;
            }
            _mm_pause();
        } else if (counting) burst = 0;
    }
}
void Verifier::pass_prof_dump() {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const int G = groups_[last_t_] > 0 ? groups_[last_t_] : 1;
    if (!prof_on_ || G != 1) return;
    cudaMemcpy(prof_h_.data(), prof_, prof_h_.size() * 8, cudaMemcpyDeviceToHost);
    const int64_t L = g.n_layers;
    auto at = [&](int64_t l, int i) { return prof_h_[(size_t) (l * kProfPer + i)]; };
    for (int64_t l = 0; l < L; ++l) {
        const int kind = is_qsa_layer(g, l) ? 1 : 0;
        unsigned long long prev = at(l, 0);
        for (int i = 1; i <= 24; ++i) {
            const unsigned long long x = at(l, i);
            if (x == 0 || x < prev) continue;
            prof_sum_[kind][i] += (double) (x - prev);
            prev = x;
        }
        if (l + 1 < L) prof_sum_[kind][25] += (double) (at(l + 1, 0) - at(l, 24));
        prof_sum_[kind][27] += (double) (at(l, 27) - at(l, 0));    // hc-read0: norm
        prof_sum_[kind][28] += (double) (at(l, 28) - at(l, 27));   //           down
        prof_sum_[kind][29] += (double) (at(l, 1) - at(l, 28));    //           up
        prof_sum_[kind][1] -= (double) (at(l, 1) - at(l, 0));      // (hc-read0 shown split)
    }
    prof_sum_[0][26] += (double) (at(L, 1) - at(L, 0));
    ++prof_windows_;
}
bool Verifier::pass_finish_outputs(int T, int32_t* out, std::string& err) {
    using namespace strata::kernels;
    // ---- a sampled or penalized request: the head's sampling again, host-side so its parameters are this call's
    // own (a captured kernel would replay the same draws forever).  Row t's draw is Philox(seed, pos0 + t): tied to
    // the POSITION it samples, not to how the text was cut into windows, so a seed replays the same text whatever
    // the drafts were. Exact: a rejected row's draw is discarded, and no kept decision depends on a reused draw.
    const bool sampled = !sampling_.greedy && sampling_.temperature > 0.0f;
    if (head_sampling_ && (sampled || hist_d_ != nullptr)) {
        SamplerParams sp = sampling_;
        sp.counter = (uint64_t) last_pos0_;
        sample_tokens(head_logits_, T, (int) n_vocab_, hist_d_, hist_len_, sp, m_out_, cs_);
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {   // m_out_ is the mapped h_out_: synced, it is readable
            err = "verify: the head sampling failed";
            return false;
        }
    }
    if (out != nullptr) for (int t = 0; t < T; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    ++windows;
    progress_at("decode");
    progress_beat();
    return true;
}
bool Verifier::end_pass_window(int T, int32_t* out, std::string& err) {
    const OnDevice on_device(device_);
    pass_live_ = false;
    progress_at("verify window: waiting for the GPU to finish the window (flags A/B/M raised)", (int64_t) T);
    // #267: a window the GPU never finishes (a spin kernel that never sees its flag) holds the host here; the stall
    // watchdog then releases every verifier's GPU waits (release_live_verifiers) before it ends the engine, so no
    // spin kernel outlives the process - the case that left Windows GPUs "lost" until a power cycle.  The wait
    // itself stays a blocking sync: a cudaStreamQuery poll here cost IQ3_S ~3% decode (a core calling the driver
    // beside the expert workers).
    const cudaError_t se = cudaStreamSynchronize(cs_);
    if (se != cudaSuccess) { err = std::string("verify: ") + cudaGetErrorString(se); return false; }
    // R2 (split1): the pass is over and cs_ is drained - record its completion event now (was: at
    // launch time, on the in-flight stream; see launch_pass).  Every consumer runs after this
    // point in host order, so the event's meaning ("the pass completed") is unchanged.
    if (deferred_done_ != nullptr) {
        const cudaError_t de = cudaEventRecord(deferred_done_, cs_);
        deferred_done_ = nullptr;
        if (de != cudaSuccess) {   // split1-v3: name the actual CUDA refusal (P1 forensics)
            std::fprintf(stderr, "strata verify: the pass completion event failed: %s (%s)\n",
                         cudaGetErrorName(de), cudaGetErrorString(de));
            err = "verify: the pass completion event failed";
            return false;
        }
    }
    progress_at("verify window: waiting for the expert copies", (int64_t) T);
    // (sync invariant, rebased): see run() tail - same verifier, same comparison.
    if (fetches_.load(std::memory_order_relaxed) != pass_fetches_at_launch_)
        cudaStreamSynchronize(copy_);
    pass_prof_dump();
    if (le_ < g_->n_layers) {   // a layer split's earlier stage: the hand-off is written; the chain is the driver's
        ++windows;
        return true;
    }
    return pass_finish_outputs(T, out, err);
}
bool Verifier::end_pass_batch(const std::vector<BatchWindow>& batch, std::string& err) {
    using namespace strata::kernels;
    const OnDevice on_device(device_);
    pass_live_ = false;
    if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = "batch verify: the pass stream failed"; return false; }
    // R2 (split1): as in end_pass_window - the launched pass's completion event is recorded after
    // its stream was waited out, never on the in-flight stream (see launch_pass).
    if (deferred_done_ != nullptr) {
        const cudaError_t de = cudaEventRecord(deferred_done_, cs_);
        deferred_done_ = nullptr;
        if (de != cudaSuccess) {   // split1-v3: name the actual CUDA refusal (P1 forensics)
            std::fprintf(stderr, "strata batch verify: the pass completion event failed: %s (%s)\n",
                         cudaGetErrorName(de), cudaGetErrorString(de));
            err = "batch verify: the pass completion event failed";
            return false;
        }
    }
    // (sync invariant, rebased): serial run_batch funnels through run() batch_replay_,
    // which skips this sync when no fetch_dma(n > 0) was queued - same comparison here.
    if (fetches_.load(std::memory_order_relaxed) != pass_fetches_at_launch_)
        cudaStreamSynchronize(copy_);
    for (const auto& b : batch) ++b.verifier->windows;                   // this stage's members ran
    if (prof_on_) {
        cudaMemcpy(prof_h_.data(), prof_, prof_h_.size() * 8, cudaMemcpyDeviceToHost);
        for (int64_t l = 0; l < g_->n_layers; ++l)
            for (int stage = 0; stage < 3; ++stage)
                batch_gpu_ms[stage] += (double) (prof_h_[(size_t) l * kProfPer + 28 + stage] -
                                                 prof_h_[(size_t) l * kProfPer + 27 + stage]) / 1e6;
        batch_gpu_ms[3] += (double) (prof_h_[(size_t) g_->n_layers * kProfPer + 3] -
                                     prof_h_[(size_t) g_->n_layers * kProfPer + 2]) / 1e6;
    }
    if (le_ < g_->n_layers) return true;   // the next stage's pass is the driver's business
    // ---- the last stage: the head ran here (members' phase 4) - sampling and outputs
    for (const auto& b : batch) {
        Verifier& v = *b.verifier;
        if (v.head_sampling_ && v.head_logits_ != nullptr && ((!v.sampling_.greedy && v.sampling_.temperature >= 1e-4f) || v.hist_d_)) {
            auto sp = v.sampling_;
            sp.counter = (uint64_t) b.position;
            sample_tokens(v.head_logits_, b.count, (int) v.n_vocab_, v.hist_d_, v.hist_len_, sp, v.m_out_, cs_);
        }
    }
    if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = "batch verify: sampling failed"; return false; }
    for (const auto& b : batch)
        for (int t = 0; t < b.count; ++t) b.output[t] = b.verifier->h_out_[t];
    progress_at("decode");
    progress_beat();
    return true;
}

namespace { bool g_commit_async = false; }
void Verifier::set_commit_async(bool on) { g_commit_async = on && std::getenv("STRATA_COMMIT_SYNC") == nullptr; }

bool Verifier::wait_commit(std::string& err) {
    if (commit_pending_) {
        const OnDevice on_device(device_);
        commit_pending_ = false;
        const cudaError_t se = cudaEventSynchronize(commit_done_);
        if (se != cudaSuccess) { err = std::string("verify: commit: ") + cudaGetErrorString(se); return false; }
    }
    return next_ == nullptr || next_->wait_commit(err);
}

bool Verifier::copy_logits(int t, float* host) const {
    if (next_ != nullptr) return next_->copy_logits(t, host);   // a layer split: the head is on the last stage
    if (head_logits_ == nullptr || host == nullptr || t < 0 || n_vocab_ <= 0) return false;
    return cudaMemcpy(host, head_logits_ + (size_t) t * (size_t) n_vocab_, (size_t) n_vocab_ * sizeof(float),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool Verifier::window_logprobs(const int32_t* targets, int T, int64_t pos0, int32_t extra_id, std::FILE* out,
                               std::string& err) {
    if (next_ != nullptr) return next_->window_logprobs(targets, T, pos0, extra_id, out, err);
    const OnDevice on_device(device_);
    if (head_logits_ == nullptr || n_vocab_ <= 0 || T <= 0 || targets == nullptr || out == nullptr) {
        err = "window_logprobs: no head logits for this window";
        return false;
    }
    // run() synchronized cs_ before returning, so the head of this window is complete
    std::vector<float> h((size_t) T * (size_t) n_vocab_);
    if (cudaMemcpy(h.data(), head_logits_, h.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = "window_logprobs: the head logits copy failed";
        return false;
    }
    for (int t = 0; t < T; ++t) {
        const float* row = h.data() + (size_t) t * (size_t) n_vocab_;
        const int32_t tgt = targets[t];
        if (tgt < 0 || (int64_t) tgt >= n_vocab_) continue;
        double maxv = -1e30;
        int64_t top = 0;
        for (int64_t v = 0; v < n_vocab_; ++v) {
            const double val = (double) row[v];
            if (val > maxv) { maxv = val; top = v; }
        }
        const bool has_extra = extra_id >= 0 && (int64_t) extra_id < n_vocab_;
        double sum = 0.0, sum_without = 0.0;
        for (int64_t v = 0; v < n_vocab_; ++v) {
            const double e = std::exp((double) row[v] - maxv);
            sum += e;
            if (v != (int64_t) extra_id) sum_without += e;
        }
        const double lse = maxv + std::log(sum);
        const double extra = has_extra ? (double) row[extra_id] - lse : NAN;
        const double without = has_extra && tgt != extra_id && sum_without > 0.0
                                   ? (double) row[tgt] - (maxv + std::log(sum_without)) : NAN;
        std::fprintf(out, "%lld\t%d\t%.9f\t%lld\t%.9f\t%d\t%.9f\t%.9f", (long long) (pos0 + t), (int) tgt,
                     (double) row[tgt] - lse, (long long) top, maxv - lse, (int) (top == (int64_t) tgt), extra,
                     without);
        static const int topk = [] {
            const char* v = std::getenv("STRATA_LOGPOS_TOPK");
            return v != nullptr ? std::max(0, std::min(256, std::atoi(v))) : 0;
        }();
        if (topk > 0) {
            std::vector<int32_t> order((size_t) n_vocab_);
            for (int64_t v = 0; v < n_vocab_; ++v) order[(size_t) v] = (int32_t) v;
            std::partial_sort(order.begin(), order.begin() + topk, order.end(),
                              [&](int32_t a, int32_t b) { return row[a] > row[b]; });
            for (int j = 0; j < topk; ++j)
                std::fprintf(out, "\t%d:%.6f", order[(size_t) j], (double) row[order[(size_t) j]] - lse);
        }
        std::fprintf(out, "\n");
    }
    std::fflush(out);
    return true;
}

}  // namespace strata::core
