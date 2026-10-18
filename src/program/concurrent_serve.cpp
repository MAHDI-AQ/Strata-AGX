// src/program/concurrent_serve.cpp - Multi-Stream Concurrent Engine & Wavefront Dispatch Loop
//
// Cross-Engine Attribution and Research Lineage:
//   - SGLang: RadixAttention tree-structured prefix caching, zero-overhead event-loop scheduling, shared-memory IPC (Zheng et al., LMSYS / UC Berkeley, arXiv:2312.07104)
//   - vLLM: Continuous batching, PagedAttention virtual block tables, chunked prefill interleaving (Kwon et al., UC Berkeley, SOSP 2023)
//   - NanoFlow: Overlapped nanobatch execution, asynchronous device-level DMA scheduling (DeepSeek-AI / Tsinghua, arXiv:2408.12757)
//   - EAGLE-2: Dynamic entropy-gated speculative depth calibration and decay (Li et al., Peking University, arXiv:2406.16858)
//   - Orca: Continuous token-level iteration scheduling (OSDI 2022)
//
#include "strata/program/concurrent_serve.hpp"
#include "strata/core/nanobatch.hpp"
#include "strata/core/triple_buffer_ipc.hpp"
#include "strata/program/batch_schedule.hpp"
#include "strata/prefill/prefill.hpp"
#include "strata/prefill/moe_group_fused.cuh"
#include "strata/core/native_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/progress.hpp"
#include "strata/core/layer.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"   // C4: kVerifyMaxT (hand-off buffer sizing)
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/spec/suffix_drafter.hpp"
#include "strata/spec/tree_spec.hpp"
#include "strata/spec/draft_policy.hpp"
#include "strata/core/radix_tree.hpp"
#include "strata/core/nvme_tier.hpp"
#include "strata/core/shm_ipc.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/core/continuous_scheduler.hpp"
#include "strata/core/radix_compactor.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_set>

#include <csignal>
namespace strata::program {
namespace {
static std::atomic<bool> g_sigterm_received{false};
static void handle_engine_signal(int sig) {
    g_sigterm_received.store(true, std::memory_order_relaxed);
}

using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
bool gpu_alloc(void** p, size_t bytes, int reserve, std::string& err) {
    size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess || free < bytes + (size_t) reserve * 1048576) {
        err = "concurrency: insufficient dedicated VRAM for request states and reserve; reduce concurrency/context";
        return false;
    }
    if (cudaMalloc(p, bytes) != cudaSuccess) { err = "concurrency: allocation failed"; return false; }
    return true;
}
struct Input {
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false, closed = false;
};
struct Request {
    uint64_t id = 0;
    int64_t max_new = 0;
    std::vector<int64_t> tokens;
    kernels::SamplerParams sampling{};
    float spec_min_p = 0.5f;
    uint64_t session = 0;   // D1 wire hint: the CGEN sess= conversation key (0 = none)
    std::string sve_path;
    std::vector<float> img_rows;
    std::vector<const float*> row_ptr;
};
bool parse_request(const std::string& line, const ConcurrentConfig& config, const core::ModelGeometry& g, int64_t vocab, Request& r, std::string& err) {
    std::istringstream input(line);
    std::string command, id, maximum, token;
    input >> command >> id >> maximum;
    auto integer = [](const std::string& s, const char* field = "integer") -> int64_t {
        if (s.empty()) throw std::invalid_argument(std::string("empty ") + field);
        size_t end = 0;
        try {
            const auto n = std::stoll(s, &end);
            if (end != s.size()) throw std::invalid_argument(std::string("invalid ") + field + ": \x27" + s + "\x27");
            return n;
        } catch (const std::exception& e) {
            throw std::invalid_argument(std::string("invalid ") + field + ": \x27" + s + "\x27 (" + e.what() + ")");
        }
    };
    try {
        const auto number = integer(id, "request id");
        if (number < 1) throw std::invalid_argument("request id");
        r.id = (uint64_t) number;
        r.max_new = integer(maximum, "max_new");
        const bool is_cgeni = (command == "CGENI");
        if ((command != "CGEN" && !is_cgeni) || r.max_new < 1 || r.max_new > config.context) throw std::invalid_argument("max_new");
        r.sampling.greedy = true;
        r.sampling.temperature = 0;
        r.sampling.top_p = 1.0f;
        r.spec_min_p = config.spec_min_p;
        bool got_tokens = false;
        bool got_sve = !is_cgeni;
        while (input >> token) {
            const auto eq = token.find('=');
            if (eq == std::string::npos) {
                if (!got_sve) {
                    r.sve_path = token;
                    got_sve = true;
                    continue;
                }
                if (got_tokens) throw std::invalid_argument("trailing data");
                got_tokens = true;
                std::istringstream ids(token);
                std::string cell;
                while (std::getline(ids, cell, ',')) {
                    if (cell.empty()) continue;
                    const auto value = integer(cell, "token id");
                    if (value < 0 || value >= vocab || r.tokens.size() >= (size_t) config.context)
                        throw std::invalid_argument("token/context");
                    r.tokens.push_back(value);
                }
                if (token.back() == ',') throw std::invalid_argument("empty token");
                continue;
            }
            if (got_tokens) throw std::invalid_argument("keys after tokens");
            const std::string key = token.substr(0, eq), value = token.substr(eq + 1);
            size_t used = 0;
            const float f = std::stof(value, &used);
            if (used != value.size() || !std::isfinite(f)) throw std::invalid_argument("sampling value");
            if (key == "temperature") { if (f < 0) throw std::invalid_argument(key); r.sampling.temperature = f; r.sampling.greedy = f == 0; }
            else if (key == "top_p") { if (f <= 0 || f > 1) throw std::invalid_argument(key); r.sampling.top_p = f; }
            else if (key == "min_p") { if (f < 0 || f > 1) throw std::invalid_argument(key); r.sampling.min_p = f; }
            else if (key == "top_k") { const auto n = integer(value, "top_k"); if (n < 1 || n > 64) throw std::invalid_argument(key); r.sampling.top_k = (int) n; }
            else if (key == "seed") { const auto n = integer(value, "seed"); if (n < 0) throw std::invalid_argument(key); r.sampling.seed = (uint64_t) n; }
            else if (key == "penalty_last_n") { const auto n = integer(value, "penalty_last_n"); if (n < 0 || n > 4096) throw std::invalid_argument(key); r.sampling.penalty_last_n = (int) n; }
            else if (key == "penalty_repeat") { if (f <= 0) throw std::invalid_argument(key); r.sampling.penalty_repeat = f; }
            else if (key == "penalty_freq") r.sampling.penalty_freq = f;
            else if (key == "penalty_present") r.sampling.penalty_present = f;
            else if (key == "spec_min_p") { if (f < 0 || f > 1) throw std::invalid_argument(key); r.spec_min_p = f; }
            // D1 wire hint: the conversation key the server derives from the prompt prefix (a u64;
            // 0 = absent).  Strict decimal digits so "-1" cannot wrap into a valid-looking key.
            else if (key == "sess") {
                if (value.empty() || value[0] < '0' || value[0] > '9') throw std::invalid_argument(key);
                size_t sess_end = 0;
                const unsigned long long n = std::stoull(value, &sess_end);
                if (sess_end != value.size()) throw std::invalid_argument(key);
                r.session = (uint64_t) n;
            }
            else if (key == "cvec" && (f == 0 || f == 1)) {} // no control vector can be loaded in this mode
            else throw std::invalid_argument("unsupported key: " + key);
        }
        if (!got_tokens || r.tokens.empty()) throw std::invalid_argument("empty prompt");
        if (!r.sve_path.empty()) {
            constexpr int64_t kImagePad = 248056;
            struct Img { int64_t n, nx, ny; size_t off; };
            std::vector<Img> imgs;
            r.img_rows.clear();
            std::FILE* f = std::fopen(r.sve_path.c_str(), "rb");
            if (!f) throw std::invalid_argument("cannot open " + r.sve_path);
            std::string ve;
            while (f && ve.empty()) {
                int32_t hdr[5];
                const size_t got = std::fread(hdr, sizeof(int32_t), 5, f);
                if (got == 0) break;
                if (got != 5 || hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 ||
                    (int64_t) hdr[2] * hdr[3] != hdr[1] || hdr[4] != (int32_t) g.n_embd) {
                    ve = "bad embeddings file (expected strata-vision records of width " +
                         std::to_string((long long) g.n_embd) + ")";
                    break;
                }
                const size_t off = r.img_rows.size(), cnt = (size_t) hdr[1] * (size_t) hdr[4];
                r.img_rows.resize(off + cnt);
                if (std::fread(r.img_rows.data() + off, sizeof(float), cnt, f) != cnt) { ve = "short embeddings file"; break; }
                imgs.push_back({hdr[1], hdr[2], hdr[3], off});
            }
            if (f) std::fclose(f);
            if (!ve.empty()) throw std::invalid_argument(ve);
            const int64_t n = (int64_t) r.tokens.size();
            r.row_ptr.assign((size_t) n, nullptr);
            int64_t i = 0;
            size_t k = 0;
            while (ve.empty() && i < n) {
                if (r.tokens[(size_t) i] != kImagePad) { ++i; continue; }
                if (k >= imgs.size()) { ve = "the prompt has more images than the embeddings file"; break; }
                const Img& im = imgs[k++];
                for (int64_t j = 0; j < im.n && ve.empty(); ++j)
                    if (i + j >= n || r.tokens[(size_t) (i + j)] != kImagePad)
                        ve = "image " + std::to_string(k) + " has " + std::to_string((long long) im.n) +
                             " rows but fewer <|image_pad|> tokens";
                for (int64_t j = 0; j < im.n && ve.empty(); ++j) {
                    r.row_ptr[(size_t) (i + j)] = r.img_rows.data() + im.off + (size_t) j * (size_t) g.n_embd;
                }
                i += im.n;
            }
            if (ve.empty() && k != imgs.size()) ve = "the embeddings file has more images than the prompt";
            if (ve.empty() && n > 0 && r.tokens[(size_t) (n - 1)] == kImagePad) ve = "the prompt cannot end in an image";
            if (!ve.empty()) throw std::invalid_argument(ve);
        }
        return true;
    } catch (const std::exception& e) { err = "bad concurrent request: " + std::string(e.what()); return false; }
}
static std::mutex s_stdout_mutex;
void error(uint64_t id, const std::string& e) {
    std::lock_guard<std::mutex> lock(s_stdout_mutex);
    std::printf("R %llu ERR %s\n", (unsigned long long) id, e.c_str());
    std::fflush(stdout);
}
}

struct ConcurrentServe::Impl {
    explicit Impl(ConcurrentConfig c) : config(std::move(c)) {}
    struct Slot {
        // C4: one entry per stage of the engine.  Slot 0 borrows the CLI's chain (the primary session and the
        // CLI stage sessions); every other slot owns one per-stage arena.  The drafter stays slot-level and
        // lives on the LAST stage's device; history_device stays slot-level too (run() allocates it there).
        struct StageSlot {
            core::SessionState owned;
            core::SessionState* state = nullptr;
            void* arena = nullptr;
            bool carved = false;      // STRATA_SLOT_LAZY: the arena lives inside the stage cache's allocation
            core::Verifier verify;
            prefill::Prefill prompt;
            void* ple_scratch = nullptr;   // only the stage that holds layer 1 (stage 0 for any legal split)
            // SPRINT 1: Snapshot buffers for cross-slot prefix sharing
            float* gdn_saved = nullptr;
            float* ple_saved = nullptr;
            float* R_saved = nullptr;
            int32_t ple_prev_saved[2] = {-1, -1};
            int32_t ple_token_saved = -1;
            // Frontier 2: Non-blocking multi-stage KV fork overlapping
            cudaEvent_t fork_event = nullptr;
            bool fork_in_flight = false;
            // TODO(C4 run-side): per-stage destruction guard (a core::OnDevice member) is deferred - OnDevice
            // is neither default-constructible nor movable, and StageSlot lives in a std::vector.
        };
        // The stage count is fixed for the engine's life.  Sized here because a vector of StageSlot cannot be
        // resized (Verifier and Prefill are not movable: user-declared destructors with deleted copies).
        explicit Slot(size_t stage_count) : stages(stage_count) {}
        std::vector<StageSlot> stages;
        std::unique_ptr<core::MtpDrafter> draft_owner;
        core::MtpDrafter* draft = nullptr;
        int32_t* history_device = nullptr;
        std::vector<int32_t> history, consumed;
        spec::SuffixDrafter suffix;
        spec::DraftPolicy policy{8};
        Request request;
        int64_t max_context = 0;
        // Lane prefill: `active`/`read`/`position`/`prompt_ms` are touched by the pump thread as well as by
        // the loop.  The gate that keeps them disjoint is `read < position`: only the pump advances `read`
        // during prompt processing, and the loop only decodes (and re-sets `read = position`) once the slot
        // is ready.  `read` is the release/acquire hand-off between the two (see the pump).
        std::atomic<bool> active{false};
        std::atomic<uint64_t> in_flight_prefix_hash{0};
        bool first = true, lookup = false;
        bool brought_up = false;   // STRATA_SLOT_LAZY: false until its first admission initializes it
        // P4 live retention: `consumed` is ALSO the conversation this slot's sessions hold. At idle the
        // sessions are exactly the sequence state for those tokens - the invariant every writer keeps:
        // run_chunk appends each read token as prefill consumes it, retire_unit commits the window's kept
        // tokens (commit() advances GDN/PLE with them; verify.hpp:185). So an admission that finds the
        // full retained history as an exact prefix of its request continues the sequence with no copy and
        // no session_zero. `reused_prefix` = tokens THIS request reused (the DONE trailer's field 9).
        int64_t reused_prefix = 0;
        // D1: the session key of the last request this slot admitted (the CGEN sess= hint), 0 = none.
        // Written only while the retain pre-pass is live; read only by that pre-pass (inert when off).
        uint64_t session = 0;
        std::atomic<int64_t> read{0}, position{0};
        int64_t generated = 0, offered = 0, accepted = 0;
        int32_t current = 0;
        int count = 0, match = 0;
        int32_t drafts[8]{}, lookup_tokens[8]{}, window[8]{}, output[8]{};
        float probability[8]{};
        std::atomic<double> prompt_ms{0};
        Clock::time_point decode_start{};
        int64_t saved_prefix = 0;
        int64_t root_at = -1;            // Phase 1: End of system prompt (<|im_start|>user boundary)
        int64_t turn_at = -1;            // Phase 1: Last turn boundary
        std::vector<int64_t> turn_boundaries;
        size_t next_turn_snap_idx = 0;
        bool root_snap_done = false;     // Phase 1: System root snapshot captured to L2 Host RAM
        bool turn_snap_done = false;     // Phase 3: Conversational turn boundary snapshot
        bool prompt_snap_done = false;   // P18 L1: the prompt-end snapshot was captured this admission
        std::shared_ptr<core::RadixNode> radix_node = nullptr;
        std::string cache_tier = "Fresh";
        std::vector<int32_t> saved_consumed;
        ~Slot() {
            // TODO(C4 run-side): free under OnDevice(last stage / stage device) once stages can be non-zero.
            if (history_device) cudaFree(history_device);
            for (auto& gs : stages) {
                if (gs.ple_scratch) cudaFree(gs.ple_scratch);
                if (gs.gdn_saved) cudaFree(gs.gdn_saved);
                if (gs.ple_saved) cudaFree(gs.ple_saved);
                if (gs.R_saved) cudaFree(gs.R_saved);
                if (gs.fork_event) cudaEventDestroy(gs.fork_event);
            }
        }
    };
    ConcurrentConfig config;
    const core::ModelGeometry* geometry = nullptr;
    std::vector<std::unique_ptr<Slot>> slots;
    std::vector<ServeStage> stages;             // stage list: prepare stores it, run() refreshes it
    core::GpuPlanSink** stage_plans = nullptr;  // C4: per-round publish targets (= split_drive.plan)
    // C4: one hand-off buffer per stage boundary (n_stages - 1), mapped pinned and portable: the stages on
    // either side read it from different devices.  All of a stage's slot verifiers and batch coordinators
    // point at the same buffer (C2's packed row0/phase-5 hand-off writes consume it there).
    struct Boundary { float* host[3] = {}; float* dev[3] = {}; };   // Task 8.2: TripleBufferIPC 3-slot circular ring
    std::vector<Boundary> hand;
    // Lane prefill: the chunk pump.  One thread owns every Prefill::run call; the loop thread keeps running
    // decode rounds while a chunk is in flight (different streams, disjoint slot state).  `pump_paused`
    // holds it at a chunk boundary for the fences (admission, CSTOP, adapt, exit); `pump_busy` is the
    // chunk-in-flight flag the fences wait on; `pump_failed`/`pump_err` carry a chunk failure back to the
    // loop (the loop returns 1 exactly where the inline call used to).
    std::mutex pump_mutex;
    std::condition_variable pump_cv;
    bool pump_paused = false, pump_busy = false;
    std::atomic<bool> pump_failed{false};
    std::string pump_err;
    std::atomic<size_t> prompt_rotation{0};   // thread-safe: accessed by pump and admission loops
    // C1-B/C4: the prompt path is stage-owned, one entry per stage; N=1 keeps exactly one (same resources).
    struct StageRt {
        int device = 0;
        void* prompt_workspace = nullptr;
        uint64_t prompt_bytes = 0;
        cudaStream_t prompt_stream = nullptr;
    };
    std::vector<StageRt> stage_rt;
    // Lane w3-chaingate: the engine-wide "one deferred stage-1 run in flight" gate.  Every slot's
    // stage-0 prompt points at THIS gate (run()'s per-slot block); all slots' stage-1 prompts share the
    // stage-1 prompt workspace/stream, so two live stage-1 runs would alias it (prefill.hpp ChainGate).
    prefill::ChainGate chain_gate;
    // STRATA_SLOT_LAZY (prepare reads the env; run()'s bringup_slot uses these members).
    bool lazy_slots = false;
    int64_t session_k = 0;
    core::SessionState* primary_state = nullptr;
    core::MtpDrafter* shared_draft = nullptr;
    std::vector<int64_t> deferred_bytes;   // session bytes per stage, recorded in prepare()
    ~Impl() {
        // Graphs and draft state must die before the sessions they reference. Slot 0's sessions are borrowed
        // from the CLI; the own per-(slot, stage) arenas are collected first and freed after slots.clear().
        struct OwnedArena { void* base; core::QsaState* qsa; int device; bool carved; };
        std::vector<OwnedArena> allocations;
        for (const auto& s : slots)
            for (size_t st = 0; st < s->stages.size(); ++st) {
                auto& gs = s->stages[st];
                if (gs.arena) allocations.push_back({gs.arena, gs.owned.qsa_states,
                                                     st < stages.size() ? stages[st].device : 0, gs.carved});
            }
        slots.clear();
        for (const auto& a : allocations) {
            if (a.qsa) for (int64_t i = 0; i < geometry->n_qsa_layers(); ++i) {
                if (a.qsa[i].host_step) cudaFreeHost(a.qsa[i].host_step);
                if (a.qsa[i].host_pos) cudaFreeHost(a.qsa[i].host_pos);
            }
            delete[] a.qsa;
            const core::OnDevice on(a.device);   // the arena was allocated on its stage's device
            if (!a.carved) cudaFree(a.base);     // a carved arena is inside the expert cache's allocation
        }
        for (auto& rt : stage_rt) {
            const core::OnDevice on(rt.device);  // the stream/workspace were created on that stage's device
            if (rt.prompt_stream) cudaStreamDestroy(rt.prompt_stream);
            if (rt.prompt_workspace) cudaFree(rt.prompt_workspace);
        }
        for (auto& h : hand) for (int p = 0; p < 3; ++p)   // Task 8.2: 3-slot mapped hand-off buffers
            if (h.host[p]) cudaFreeHost(h.host[p]);
    }
};
ConcurrentServe::ConcurrentServe(ConcurrentConfig c) : impl_(std::make_unique<Impl>(std::move(c))) {}
ConcurrentServe::~ConcurrentServe() = default;

int64_t ConcurrentServe::prompt_workspace_bytes() const {   // LANE m1m2 (M1 telemetry)
    int64_t bytes = 0;
    for (const auto& rt : impl_->stage_rt) bytes += (int64_t) rt.prompt_bytes;
    return bytes;
}

bool ConcurrentServe::prepare(const core::ModelGeometry& g, core::SessionState& primary, core::MtpDrafter& draft,
                              const std::vector<ServeStage>& stages, std::string& err) {
    auto& m = *impl_;
    g_sigterm_received.store(false, std::memory_order_relaxed);
    std::signal(SIGTERM, handle_engine_signal);
    std::signal(SIGINT, handle_engine_signal);
    m.geometry = &g;
    m.stages = stages;   // C4: stage devices and borrowed sessions are read here; run() re-reads the rest
    const auto& c = m.config;
    // LANE alloc-empty: STRATA_SLOT_LAZY defers an owned slot's stage sessions and its drafter to the
    // slot's FIRST admission, and carves each session arena from its stage cache's tail.  A boot whose
    // extra slots stay empty then does not displace expert-cache slots (measured at 5x87.5K: the c8 boot's
    // three empty slots cost the whole 1.73x prefill wall through the expert cache they shrink).  The
    // carve makes the released pairs stream, and the residency updates before the next plan keep the
    // captured graphs safe (adapt()'s doctrine).  The stage hand-off stays serial, so the cross-device
    // stage-overlap mode stays refused (structural - its two in-flight units ride plans published before
    // a mid-serve carve); --batch-parallel composes (lane lazy-enabler): it forks per-member streams
    // WITHIN one round on one plan, so the carve's ordering is unchanged - its extra resource is the
    // per-slot PLE scratch, allocated for each slot below.  Default off: unset, every allocation below
    // is the boot-time one it always was and the engine is byte-identical.
    m.lazy_slots = [] { const char* v = std::getenv("STRATA_SLOT_LAZY"); return v != nullptr && std::atoi(v) != 0; }();
    if (m.lazy_slots) {
        const char* ov = std::getenv("STRATA_STAGE_OVERLAP_CROSSDEV");
        if (ov != nullptr && std::atoi(ov) != 0) {
            err = "concurrency: STRATA_SLOT_LAZY needs the serial stage hand-off (unset STRATA_STAGE_OVERLAP_CROSSDEV)";
            return false;
        }
        m.deferred_bytes.assign(stages.size(), 0);
    }
    m.primary_state = &primary;
    m.shared_draft = &draft;
    m.session_k = primary.k;
    static const core::ModelGeometry draft_geometry{};
    for (int i = 0; i < c.requests; ++i) {
        const int64_t slot_context = c.context;
        // Register ownership before any allocation that can fail partway through initialization.
        m.slots.push_back(std::make_unique<Impl::Slot>(stages.size()));
        auto& s = m.slots.back();
        s->max_context = slot_context;
        s->suffix = spec::SuffixDrafter(std::max(1, c.suffix), 64, (size_t) slot_context + 4096);
        for (size_t st = 0; st < stages.size(); ++st) {
            auto& gs = s->stages[st];
            const core::OnDevice on(stages[st].device);
            cudaEventCreateWithFlags(&gs.fork_event, cudaEventDisableTiming);
            if (i == 0) {
                // Slot 0 borrows the CLI's chain: the primary on stage 0, the CLI stage sessions beyond it.
                gs.state = st == 0 ? &primary : stages[st].session;
            } else if (m.lazy_slots) {
                gs.state = &gs.owned;
                m.deferred_bytes[st] = (int64_t) core::session_bytes(g, slot_context, primary.k, stages[st].lb, stages[st].le);
                if (st == 0 && c.parallel_batch && primary.ple.ready()) {
                    if (!gpu_alloc(&gs.ple_scratch, (size_t) core::ple_run_scratch_bytes(), c.reserve_mib, err)) return false;
                }
            } else {
                if (!gpu_alloc(&gs.arena, core::session_bytes(g, slot_context, primary.k, stages[st].lb, stages[st].le), c.reserve_mib, err)) return false;
                gs.state = &gs.owned;
                if (!core::session_init(g, slot_context, primary.k, gs.arena, gs.owned, stages[st].lb, stages[st].le)) { err = "concurrency: session initialization failed"; return false; }
                if (st == 0) {
                    // The PLE is a layer-1 module: only the stage that holds layer 1 (stage 0 for any legal
                    // split) wires it.  Stages >= 1 leave it unwired there (ready() == false).
                    gs.owned.ple = primary.ple;
                    gs.owned.ple.hist = gs.owned.ple_hist;
                    gs.owned.ple.prev = gs.owned.ple_prev;
                    gs.owned.ple.token = &gs.owned.ple_token;
                    // PLE weights/table are immutable, but its scratch is written during layer 1.
                    // Parallel member streams must not inherit the primary session's scratch pointer.
                    if (c.parallel_batch && primary.ple.ready()) {
                        if (!gpu_alloc(&gs.ple_scratch, (size_t) core::ple_run_scratch_bytes(), c.reserve_mib, err)) return false;
                        gs.owned.ple.scratch = static_cast<float*>(gs.ple_scratch);
                    }
                }
            }
        }
        if (i == 0) { s->draft = &draft; s->brought_up = true; }
        else if (!m.lazy_slots) {
            s->draft_owner = std::make_unique<core::MtpDrafter>();
            s->draft = s->draft_owner.get();
            // The drafter reads the LAST stage's residual and weights, so it lives on that stage's device.
            const core::OnDevice on(stages.back().device);
            if (!s->draft->load(c.mtp_dir, draft_geometry, *s->stages.back().state, c.window, err, c.draft_context, &draft)) return false;
        }
        if (!m.lazy_slots || i == 0) s->draft->set_max_drafts(c.mtp_window_rows - 1);
    }
    // C4: stage-owned prompt path, one entry per stage, sized like the stage list.  For a 1-entry list this is
    // C1-B's single workspace/stream (same allocations, same order as before).
    m.stage_rt.clear();
    m.stage_rt.resize(stages.size());
    for (size_t st = 0; st < stages.size(); ++st) {
        auto& rt = m.stage_rt[st];
        rt.device = stages[st].device;
        const core::OnDevice on(rt.device);
        rt.prompt_bytes = prefill::Prefill::bytes_needed(g, st == 0 ? primary : *stages[st].session, c.prefill_chunk);
        if (!gpu_alloc(&rt.prompt_workspace, (size_t) rt.prompt_bytes, c.reserve_mib, err)) return false;
        if (cudaStreamCreateWithFlags(&rt.prompt_stream, cudaStreamNonBlocking) != cudaSuccess) { err = "concurrency: prompt stream failed"; return false; }
    }
    // C4: one mapped hand-off buffer per stage boundary.  Sized for the widest round: a single window is at
    // most kVerifyMaxT rows, a batch round at most c.rows (the pool's n_tok*k envelope caps a real round
    // lower; the buffer is sized for the config, the cap is enforced by scheduling - C2).
    const size_t hand_rows = std::max<size_t>((size_t) strata::kernels::kVerifyMaxT, (size_t) c.rows);
    const size_t hand_bytes = hand_rows * (size_t) core::Verifier::handoff_floats(g) * sizeof(float);
    m.hand.clear();
    for (size_t b = 0; b + 1 < stages.size(); ++b) {
        // LANE OVERLAP: TWO buffers per boundary (parity A/B).  stage0[N+1] writes one parity while
        // stage1[N] still reads the other; the captured graphs bake the ACTIVE pointers, so every
        // verifier's captures are keyed by the parity too (verify.hpp set_stage_pingpong).
        m.hand.push_back(Impl::Boundary{});
        auto& boundary = m.hand.back();
        for (int p = 0; p < 3; ++p) {
            float* host = nullptr;
            if (cudaHostAlloc((void**) &host, hand_bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
                err = "concurrency: the layer-split hand-off allocation failed";
                return false;
            }
            std::memset(host, 0, hand_bytes);
            float* dev = nullptr;
            if (cudaHostGetDevicePointer((void**) &dev, host, 0) != cudaSuccess) {
                cudaFreeHost(host);
                err = "concurrency: the layer-split hand-off mapping failed";
                return false;
            }
            boundary.host[p] = host;
            boundary.dev[p] = dev;
        }
    }
    return true;
}

int ConcurrentServe::run(const std::vector<ServeStage>& stages, core::ExpertSource* source,
                         core::PoolMultiFn pool, void* user, core::GpuPlanSink** stage_plans, std::string& err) {
    auto& m = *impl_;
    const auto& c = m.config;
    const auto& g = *m.geometry;
    if (stages.empty() || !stages[0].wt || !stages[0].cache || !stages[0].dispatch) {
        err = "concurrency: no stage"; return 1;
    }
    // HiCache L2: RadixTree dynamic prefix-cache parking is native to the concurrent multi-device path.
    if (c.conversation_cache_mib > 0) {
        std::fprintf(stderr, "strata-agx concurrent: HiCache L2 host-RAM conversation cache enabled (budget: %lld MiB, slots: %d)\n", (long long) c.conversation_cache_mib, c.conversation_cache_slots);
    }
    // C1 boundary: at N=1 every binding below is the very object the old run() took as a separate
    // argument - same call sequence, same objects, same order (re-root, not a rewrite).
    const ServeStage& st0 = stages[0];
    const ServeStage& stl = stages.back();   // C4: the head, the drafter and the history live on the LAST stage
    const core::WeightTable& wt = *stl.wt;   // ... and output.weight (the vocab) is in ITS weight table
    const core::NativeHead* head = stl.head;
    core::ExpertCache& cache = *st0.cache;
    int32_t* host_res = st0.host_res;
    const core::VerifyHits& hits = st0.hits;
    core::ExpertDispatch& dispatch = *st0.dispatch;
    m.stages = stages;
    m.stage_plans = stage_plans;
    if (m.stage_rt.size() != stages.size() || m.hand.size() + 1 != stages.size()) {
        err = "concurrency: the stage prompt runtimes or hand-off buffers are missing (prepare)"; return 1;
    }
    if (!head || !head->loaded() || !wt.find("output.weight")) { err = "concurrency: native head required"; return 1; }
    if (!source || !host_res || !hits.d_res || cache.slots() < 1) {
        err = "concurrency: no profile-filled expert cache fits; reduce context/concurrency or increase available VRAM";
        return 1;
    }
    // Sprint 4: Cross-GPU PCIe Mode Configuration (0: Async DMA, 1: Direct P2P, 2: Mapped Kernel)
    const int configured_pcie_mode = []() {
        const char* mode_env = std::getenv("STRATA_PCIE_MODE");
        if (mode_env != nullptr) {
            if (std::strcmp(mode_env, "dma") == 0 || std::strcmp(mode_env, "0") == 0) return 0;
            if (std::strcmp(mode_env, "direct") == 0 || std::strcmp(mode_env, "1") == 0) return 1;
            if (std::strcmp(mode_env, "kernel") == 0 || std::strcmp(mode_env, "2") == 0) return 2;
        }
        return 0; // Default to high-throughput asynchronous DMA for Sprint 4 (v0.1.40-agx.3.0.0)
    }();
    std::fprintf(stderr, "strata-agx concurrent: cross-GPU PCIe pipeline mode set to %d (%s)\n",
                 configured_pcie_mode, configured_pcie_mode == 0 ? "Asynchronous DMA" : (configured_pcie_mode == 1 ? "Direct P2P" : "Mapped Kernel"));

    // Phase 4: Compute Kernels & MoE Router Acceleration
    {
        const char* disable_env = std::getenv("STRATA_DISABLE_NATIVE_ROUTER");
        const bool use_native = (disable_env == nullptr || std::strcmp(disable_env, "1") != 0);
        strata::kernels::native_router_set_enabled(use_native);
        std::fprintf(stderr, "strata-agx concurrent: %s router enabled (SM89 Ada Lovelace warp-cooperative Top-10)\n",
                     use_native ? "native fused" : "generic");
    }
    if (const char* shm_env = std::getenv("STRATA_IPC_SHM")) {
        if (ipc::ShmProducer::instance()->init(shm_env)) {
            std::fprintf(stderr, "strata-agx concurrent: zero-copy shared memory IPC enabled (%s)\n", shm_env);
        }
    }
    struct ProgressGuard { ~ProgressGuard() { core::progress().busy.store(false); } } progress_guard;
    std::jthread watchdog; // outlives the batch graph, including teardown after a failed GPU execution
    // C4: one batch coordinator per stage (at N=1 this is today's single `batch`).  Each carries its
    // stage's range, hit view and boundary buffers, chained like the slot verifiers; the chain drives them
    // from INSIDE run_batch (batch[0] replays, then next_->run_batch) - DEPENDS-C2: inert until C2's
    // run_batch accepts a chained stage (a multi-stage batch round is refused at the round dispatch below).
    std::vector<core::Verifier> batch(m.stages.size());
    for (size_t st = 0; st < m.stages.size(); ++st) {
        const auto& sg = m.stages[st];
        const core::OnDevice on(sg.device);
        const bool last = st + 1 == m.stages.size();
        auto& b = batch[st];
        b.set_batch_cache(c.graph_cache, 128);
        b.set_batch_parallel(c.parallel_batch);
        b.set_stage_ring(sg.lb, last ? -1 : sg.le,
                         st == 0 ? nullptr : m.hand[st - 1].dev[0], last ? nullptr : m.hand[st].dev[0],
                         st == 0 ? nullptr : m.hand[st - 1].dev[1], last ? nullptr : m.hand[st].dev[1],
                         st == 0 ? nullptr : m.hand[st - 1].dev[2], last ? nullptr : m.hand[st].dev[2]);
        if (!b.init(*sg.wt, g, *m.slots[0]->stages[st].state, sg.hits, sg.head, std::max(2, c.rows), err, true)) return 1;
        b.set_pcie_mode(configured_pcie_mode);
        if (!last) b.set_next(&batch[st + 1], user);
    }
    // C4: per-slot per-stage chains, wired like the single-request split path (generate.cpp 3641-3730):
    // each stage's verifier runs [lb, le) and hands its rows to the next one's boundary buffer, the prompt
    // paths chain the same way, and the drafter, the history and on_chunk belong to the LAST stage.
    for (auto& ptr : m.slots) {
        auto& s = *ptr;
        if (m.lazy_slots && !s.brought_up) continue;   // STRATA_SLOT_LAZY: its first admission initializes it
        for (size_t st = 0; st < m.stages.size(); ++st) {
            const auto& sg = m.stages[st];
            const core::OnDevice on(sg.device);
            const bool last = st + 1 == m.stages.size();
            s.stages[st].verify.set_stage_ring(sg.lb, last ? -1 : sg.le,   // set_stage BEFORE init
                                               st == 0 ? nullptr : m.hand[st - 1].dev[0],
                                               last ? nullptr : m.hand[st].dev[0],
                                               st == 0 ? nullptr : m.hand[st - 1].dev[1],
                                               last ? nullptr : m.hand[st].dev[1],
                                               st == 0 ? nullptr : m.hand[st - 1].dev[2],
                                               last ? nullptr : m.hand[st].dev[2]);
            if (!s.stages[st].verify.init(*sg.wt, g, *s.stages[st].state, sg.hits, sg.head, c.window, err)) return 1;
            s.stages[st].verify.set_pcie_mode(configured_pcie_mode);
            if (!last) s.stages[st].verify.set_next(&s.stages[st + 1].verify, user);   // user = &split_drive
            s.stages[st].prompt.set_stage(sg.lb, last ? -1 : sg.le, last ? nullptr : &s.stages[st + 1].prompt);
            if (!s.stages[st].prompt.init(*sg.wt, g, *s.stages[st].state, source, sg.cache, sg.host_res,
                                          c.prefill_chunk, (void*) m.stage_rt[st].prompt_stream, err,
                                          m.stage_rt[st].prompt_workspace, m.stage_rt[st].prompt_bytes)) return 1;
            // Lane pipeline-prefill: this pump may leave a chunk's stage-1 run in flight between chunks
            // (STRATA_PREFILL_CHAIN=1).  Only this path opts in, so a leaked env is inert elsewhere.
            s.stages[st].prompt.set_chain_defer(true);
            // Lane w3-chaingate: the FIRST stage's prompt is the one that spawns the deferred next-stage
            // runs; one gate across all slots keeps them from overlapping.  Stages >= 1 spawn their own
            // deferred runs from INSIDE a gated run (3+ stage shapes) and must not take this gate.
            if (st == 0) s.stages[st].prompt.set_chain_gate(&m.chain_gate);
        }
        // The drafter reads the chain's last residual and the LAST stage's weights/head (generate.cpp:3750).
        if (!s.draft->bind(*m.stages.back().wt, m.stages.back().head, s.stages[0].verify.final_R_all(), err)) return 1;
        s.history.resize((size_t) c.window * 4096, -1);
        // C4: the last stage's sampler reads it, so it is allocated on the last stage's device.
        if (const core::OnDevice on(m.stages.back().device);
            cudaMalloc(&s.history_device, s.history.size() * sizeof(int32_t)) != cudaSuccess) {
            err = "concurrency: penalty buffer allocation failed"; return 1;
        }
        auto* slot = &s;
        // C4: on_chunk belongs to the LAST stage (prefill.hpp:104; generate.cpp:3940 moves it there in the
        // single path) - it is called when the prompt chunk has crossed every stage.
        s.stages.back().prompt.on_chunk = [slot](const float* residual, int64_t n, int64_t position, std::string& e) {
            std::vector<int32_t> next((size_t) n);
            for (int64_t j = 0; j < n; ++j)
                next[(size_t) j] = (int32_t) (position + j + 1 < (int64_t) slot->request.tokens.size()
                                                  ? slot->request.tokens[(size_t) (position + j + 1)]
                                                  : 0);
            // Use the existing draft implementation; both slots share only immutable weights.
            return slot->draft->prefill(residual, next.data(), n, position, e);
        };
    }
    size_t free = 0, total = 0;
    for (const auto& sg : m.stages) {
        const core::OnDevice on(sg.device);   // every stage must keep the configured reserve on ITS device
        cudaMemGetInfo(&free, &total);
        if (free < 256ULL * 1048576) {
            err = "concurrency: remaining VRAM is below the requested reserve; lower expert cache/context"; return 1;
        }
    }
    const bool adaptive = c.adapt_every > 0 && c.adapt_swaps > 0;
    if (adaptive) dispatch.usage.assign((size_t) g.n_layers * g.n_expert, 0.0f);
    int64_t rounds = 0;
    int64_t batch_sizes[17]{};   // one slot per member count 1..16 (M2 request-lattice raise; 17 entries)
    const bool profiling = std::getenv("STRATA_CONCURRENT_PROFILE") != nullptr;
    double target_ms = 0, draft_ms = 0, commit_ms = 0, adapt_ms = 0;
    std::atomic<double> prefill_ms{0};   // written by the pump thread, read by the profile (lane prefill)
    int64_t produced = 0, target_rows = 0;

    // spec acceptance census (lane-spec instrumentation; rebased to d319d46 + per-source/per-position by
    // lane-spec-fuse).  windows served, drafts offered to verify, drafts accepted, and the histograms that
    // show WHERE the draft chain truncates (count) and how much commits (keep).  The DIRECT per-request
    // channel (draft_n/draft_n_accepted, serve/server.py:472) sums these same two fields per request, so
    // this census is the engine-side aggregate of that measure; the owner audit (2026-10-02) retires the
    // aggregate-profile-identity reconstruction of acceptance in favour of reading these counters.
    int64_t windows_served = 0, offered_total = 0, accepted_total = 0;
    int64_t count_hist[9] = {}, keep_hist[9] = {};
    // per-source split (F5): [0] the MTP chain, [1] a suffix-lookup window; per-position: the draft at
    // window row t (t = 1..) was offered, and accepted when the window's committed rows keep >= t.
    int64_t source_windows[2] = {}, source_offered[2] = {}, source_accepted[2] = {};
    int64_t pos_offered[9] = {}, pos_accepted[9] = {};
    // D1: admission-scan stall samples - the whole-queue match pass's added time at the safe point,
    // for the falsifier (p50/max printed in the profile; only ever written while retain is on).
    constexpr size_t kAdmissionStallSamples = 256;
    int64_t admit_scans = 0;
    double admit_ms_sum = 0.0, admit_ms_max = 0.0, admit_ms_ring[kAdmissionStallSamples]{};
    auto report_profile = [&]() {
        if (!profiling || !rounds) return;
        // every stage of the chain waits - a split's stage 1 included: sum them all, not stage 0's only.
        double wait = 0, pool_ms = 0, host = 0;
        for (const auto& b : batch) { wait += b.ms_wait; pool_ms += b.ms_pool; host += b.ms_host; }
        for (const auto& s : m.slots)
            for (const auto& gs : s->stages) {
                wait += gs.verify.ms_wait; pool_ms += gs.verify.ms_pool; host += gs.verify.ms_host;
            }
        std::fprintf(stderr, "strata concurrent profile: rounds=%lld tokens=%lld rows=%lld windows=%lld "
                     "offered=%lld accepted=%lld target_ms=%.1f "
                     "draft_ms=%.1f commit_ms=%.1f adapt_ms=%.1f prefill_ms=%.1f "
                     "target_wait_ms=%.1f target_pool_ms=%.1f target_host_ms=%.1f "
                     "dispatch_plan_ms=%.1f actq_ms=%.1f jobs_ms=%.1f run_ms=%.1f\n",
                     (long long) rounds, (long long) produced, (long long) target_rows, (long long) windows_served,
                     (long long) offered_total, (long long) accepted_total, target_ms,
                     draft_ms, commit_ms, adapt_ms, prefill_ms.load(), wait, pool_ms, host,
                     dispatch.ms_plan, dispatch.ms_actq, dispatch.ms_jobs, dispatch.ms_run);
        std::fflush(stderr);
        std::fprintf(stderr, "strata concurrent detail: captures=%lld capture_ms=%.1f gpu_pre_ms=%.1f "
                     "gpu_experts_ms=%.1f gpu_post_ms=%.1f gpu_head_ms=%.1f\n",
                     (long long) batch[0].batch_captures, batch[0].ms_batch_capture, batch[0].batch_gpu_ms[0],
                     batch[0].batch_gpu_ms[1], batch[0].batch_gpu_ms[2], batch[0].batch_gpu_ms[3]);
        if (admit_scans) {   // D1: only when the retain pre-pass ran (retain off => unchanged output)
            const size_t n = std::min<size_t>((size_t) admit_scans, kAdmissionStallSamples);
            std::vector<double> samples(admit_ms_ring, admit_ms_ring + n);
            std::sort(samples.begin(), samples.end());
            std::fprintf(stderr, "strata concurrent admission: passes=%lld scan_ms_mean=%.3f scan_ms_p50=%.3f "
                         "scan_ms_max=%.3f (D1 whole-queue match pass; window=%zu)\n",
                         (long long) admit_scans, admit_ms_sum / (double) admit_scans,
                         samples[samples.size() / 2], admit_ms_max, n);
        }
    };
    auto adapt = [&]() -> bool {
        // All target, commit, prefill and draft work has finished at this boundary. Updating both
        // residency tables here makes cached graph pointers safe without per-request invalidation.
        struct Swap { float gain; int64_t layer; int in, out; };
        std::vector<Swap> swaps;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            std::vector<std::pair<float, int>> candidates, victims;
            for (int e = 0; e < g.n_expert; ++e) {
                const size_t i = (size_t) l * g.n_expert + e;
                const float u = dispatch.usage[i];
                if (host_res[i] >= 0) victims.emplace_back(u, e);
                else if (u >= 2.0f) candidates.emplace_back(u, e);
            }
            std::sort(candidates.rbegin(), candidates.rend());
            std::sort(victims.begin(), victims.end());
            for (size_t i = 0; i < std::min(candidates.size(), victims.size()); ++i) {
                const float gain = candidates[i].first - victims[i].first;
                if (gain < 1.5f) break;
                swaps.push_back({gain, l, candidates[i].second, victims[i].second});
            }
        }
        std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
        if (swaps.size() > (size_t) c.adapt_swaps) swaps.resize((size_t) c.adapt_swaps);
        // C4: a swap stays in its layer's own stage: the fill goes through THAT stage's cache on ITS device,
        // and the full residency table is re-uploaded to every stage it touched (the host table is one shared
        // buffer; each stage has its own device copy in its hits.d_res).
        auto stage_of = [&](int64_t l) -> size_t {
            size_t st = 0;
            while (st + 1 < m.stages.size() && l >= (m.stages[st].le < 0 ? g.n_layers : m.stages[st].le)) ++st;
            return st;
        };
        std::vector<char> touched(m.stages.size(), 0);
        for (const auto& s : swaps) {
            const size_t st = stage_of(s.layer);
            const auto& sg = m.stages[st];
            const size_t incoming = (size_t) s.layer * g.n_expert + s.in, outgoing = (size_t) s.layer * g.n_expert + s.out;
            const int slot = sg.host_res[outgoing];   // the slot lives in that stage's cache
            const auto* blob = source->blob(s.layer, s.in);
            const core::OnDevice on(sg.device);
            if (!blob || !sg.cache->fill_slot_blocking(slot, blob, err, (int64_t) kernels::cpu::expert_layout().blob_bytes(s.layer))) return false;
            sg.host_res[outgoing] = core::kNotResident; sg.host_res[incoming] = slot; touched[st] = 1;
        }
        for (size_t st = 0; st < m.stages.size(); ++st) if (touched[st]) {
            const auto& sg = m.stages[st];
            const core::OnDevice on(sg.device);
            if (cudaMemcpy((void*) sg.hits.d_res, sg.host_res, (size_t) g.n_layers * g.n_expert * sizeof(int32_t),
                           cudaMemcpyHostToDevice) != cudaSuccess) { err = "concurrency: residency upload failed"; return false; }
        }
        for (float& usage : dispatch.usage) usage *= 0.7f;
        return true;
    };
    // P4 live retention (STRATA_CONCURRENT_RETAIN, default 0 = off): a request whose prompt starts with
    // the full token history a slot still holds resumes that slot's live sessions instead of zeroing
    // them - the sustained-traffic lever (a 128K agent turn re-reads only its new tokens). Default off,
    // so the loop below is byte-identical to the pre-P4 engine; the kill-switch is the same env.
    const bool retain = [] {
        const char* v = std::getenv("STRATA_CONCURRENT_RETAIN");
        return v != nullptr && std::atoi(v) != 0;
    }();
    // LANE spec-fuse: STRATA_DRAFT_WAVEFRONT - retire_unit defers every slot's draft past its prints and
    // position updates, then launches all slots' chains before waiting any (per-slot drafter state is
    // disjoint; each chain runs on its own drafter stream).  Default off: unset, retire_unit's per-slot
    // commit->draft order is byte-preserved.
    const bool wavefront = [] {
        const char* v = std::getenv("STRATA_DRAFT_WAVEFRONT");
        return v != nullptr && std::atoi(v) != 0;
    }();
    // P1-cache-revive: the concurrent path parks nothing yet (host-RAM snapshots); live retention is
    // separate from parking and the P1 refusal above stays in force for conversation_cache_mib > 0.
    // The DONE trailer field now carries the live-retention reuse count (was hardcoded 0).
    std::fprintf(stderr, "strata-agx concurrent: shared expert batching; independent MTP; adaptive cache %s; %s\n",
                 adaptive ? "on" : "off",
                 retain ? "prefix-cache live-retention on (parking not yet supported on this path)"
                        : "prefix-cache off (parking not yet supported on this path)");
    // D1: advertise the sess= wire hint only while retention is on - the off path stays byte-identical
    // (stdout and wire), and the server sends the hint only to an engine that advertises it.
    int64_t slots_all = 0, mib_all = 0;
    for (const auto& sg : m.stages) if (sg.cache) {
        slots_all += sg.cache->slots();
        mib_all += (sg.cache->bytes() >> 20);
    }
    std::printf("INFO engine=" STRATA_AGX_VERSION " concurrency=%d batch_rows=%d batch_policy=%s context=%lld kv=%s lookup=%d expert_policy=%s expert_slots=%lld expert_cache_mib=%lld%s\n",
                c.requests, c.rows, c.depth ? "depth" : "fair", (long long) c.context, c.kv.c_str(), c.suffix, adaptive ? "adaptive" : "static",
                (long long) slots_all, (long long) mib_all,
                retain ? " admission=d1" : "");
    std::printf("READY %lld stop multiplex\n", (long long) c.context);
    // LANE sched-impl P2 (host-loop O1/O2 surface): echo the effective spin posture so the primary's
    // A/B reads off this log line. Zero wait-posture change: the knobs are honored where they already
    // were (STRATA_REMOTE_SPIN in remote_experts.cpp, STRATA_POOL_SPIN_US in pool.cpp); unset = old path.
    {
        const char* remote_spin = std::getenv("STRATA_REMOTE_SPIN");
        const char* pool_spin = std::getenv("STRATA_POOL_SPIN_US");
        std::fprintf(stderr, "strata-agx concurrent: spin posture remote_spin=%s pool_spin_us=%s (defaults: spin|maphost, 20000us)\n",
                     (remote_spin && remote_spin[0]) ? remote_spin : "(unset:spin)",
                     (pool_spin && pool_spin[0]) ? pool_spin : "(unset:20ms)");
    }
    std::fflush(stdout);
    auto input = std::make_shared<Input>();
    std::thread([input] {
        strata::kernels::cpu::adopt_spawn_mask();   // STRATA_AUX_WIDE: not the host loop; do not inherit its core
        std::string line;
        while (std::getline(std::cin, line)) {
            std::unique_lock<std::mutex> lock(input->mutex);
            input->cv.wait(lock, [&] { return input->lines.size() < 32 || input->closed; });
            if (input->closed) return;
            const bool quit = line == "QUIT";
            input->lines.push_back(std::move(line));
            input->cv.notify_all();
            if (quit) break;
        }
        std::lock_guard<std::mutex> lock(input->mutex);
        input->eof = true;
        input->cv.notify_all();
    }).detach();
    struct InputGuard { std::shared_ptr<Input> input; ~InputGuard() { std::lock_guard<std::mutex> l(input->mutex); input->closed = true; input->cv.notify_all(); } } guard{input};
    // ---- lane prefill: the chunk pump -------------------------------------------------------------------
    // A prompt chunk is a bounded, host-synchronized unit of work (Prefill::run syncs the stage stream at
    // every MoE layer's routing group, waits on the expert staging, gathers the PLE rows).  Running it on
    // this loop thread between decode rounds serializes the two: prefill_ms was 26-31% of the mixed-load
    // wall at c=8.  The pump moves every chunk onto its own thread, so the chunk's host-bound phases
    // overlap the decode rounds' GPU work instead of alternating with them.  Invariants:
    //   * ONE chunk in flight (the per-stage prompt workspace/stream are shared by all slots) - the pump
    //     is serial by construction, in the same fair rotation the loop used;
    //   * a slot is decodable only when read == position, and only the pump advances `read` during prompt
    //     processing, so the loop can never observe a half-read prompt (release store in run_chunk);
    //   * fences (pump_fence/pump_resume) hold the pump at a chunk boundary around admission (session_zero
    //     runs on the same stage prompt streams), CSTOP (finish of a slot with a chunk in flight), adapt()
    //     (cache swaps) and exit.  Decode rounds need NO fence: they touch other slots' state only.
    // Continuous iteration-level scheduler: token-level slot recycling (uniform slots)
    core::ContinuousScheduler continuous_sched((int) m.slots.size());

    // STRATA_PREFILL_PUMP=0 keeps the pre-change inline path below for the A/B.
    auto run_chunk = [&](size_t i, std::string& chunk_err) -> bool {
        // One bounded prompt chunk for one slot: the same call and the same bookkeeping, whichever thread
        // runs it (the pump, or this loop when the pump is off).
        auto& s = *m.slots[i];
        const int64_t read0 = s.read.load(std::memory_order_acquire);
        // Task 1.2: Fine-grained dynamic chunk sizing & micro-prefill
        int active_decodes = 0;
        int active_prefills = 0;
        for (const auto& ptr : m.slots) {
            const auto& sl = *ptr;
            if (sl.active.load(std::memory_order_relaxed)) {
                if (sl.read.load(std::memory_order_relaxed) >= sl.position.load(std::memory_order_relaxed)) {
                    ++active_decodes;
                } else {
                    ++active_prefills;
                }
            }
        }
        int64_t dynamic_chunk = c.prefill_chunk;
        const int64_t rem = s.position.load(std::memory_order_relaxed) - read0;
        // Granular Workload-Aware Dynamic Chunking (Max 4096):
        // Dynamic ladder based on system load (active decodes and parallel prefills):
        // - Heavy decode pressure (active_decodes >= 2): clamp to 512 tokens to guarantee near-zero ITL jitter.
        // - Single active decode (active_decodes == 1): clamp to 1024 tokens to bound ITL < 15ms while keeping ~1,500 tok/s.
        // - Parallel prefill contest (active_decodes == 0 && active_prefills >= 2): balance bandwidth at 2048 tokens.
        // - Idle/solo prefill (active_decodes == 0 && active_prefills <= 1): expand up to 4096 for saturated Ada SM89 GEMM (>3,000 tok/s).
        if (active_decodes >= 2) {
            dynamic_chunk = std::min<int64_t>(dynamic_chunk, 512);
        } else if (active_decodes == 1) {
            dynamic_chunk = std::min<int64_t>(dynamic_chunk, 1024);
        } else if (active_prefills >= 2) {
            dynamic_chunk = std::min<int64_t>(dynamic_chunk, 2048);
        } else {
            // Idle solo prefill: allow full 4096 chunk if rem >= 4096, or exact rem / 2048
            if (rem >= 4096) {
                dynamic_chunk = std::min<int64_t>(c.prefill_chunk, 4096);
            } else if (rem >= 2048) {
                dynamic_chunk = 2048;
            } else {
                dynamic_chunk = rem;
            }
        }
        // Short prompt micro-prefill bypass: avoid rounding stalls when rem is smaller than dynamic_chunk
        if (rem <= dynamic_chunk) {
            dynamic_chunk = rem;
        }
        int64_t n = std::min<int64_t>(dynamic_chunk, rem);

        // Prefill runs uninterrupted micro-chunks to ensure continuous streaming throughput
        // Frontier 2: Enforce GPU stream dependency prior to prefill chunk launch
        for (size_t st = 0; st < s.stages.size(); ++st) {
            if (s.stages[st].fork_in_flight) {
                const core::OnDevice on(m.stages[st].device);
                cudaStreamWaitEvent(m.stage_rt[st].prompt_stream, s.stages[st].fork_event, 0);
                s.stages[st].fork_in_flight = false;
            }
        }
        const auto start = Clock::now();
        if (!s.stages[0].prompt.run(s.request.tokens.data() + read0, n, read0, chunk_err)) return false;
        // Lane pipeline-prefill: the chunk that completes this slot's prompt must drain the deferred
        // stage-1 run BEFORE read reaches position - the decode loop reads the slot only at read ==
        // position, and stage-1 is what writes this chunk's layers' KV/GDN state (and the drafter's
        // K/V).  A mid-prompt chunk leaves stage-1 in flight by design.
        if (read0 + n >= s.position.load(std::memory_order_relaxed) &&
            !s.stages[0].prompt.chain_wait(chunk_err)) return false;
        for (int64_t t = 0; t < n; ++t) s.consumed.push_back((int32_t) s.request.tokens[(size_t) (read0 + t)]);
        const double chunk_ms = elapsed(start);
        s.prompt_ms.fetch_add(chunk_ms);
        prefill_ms.fetch_add(chunk_ms);
        // Release: the consumed rows, prompt_ms and the session state are visible to this loop the moment
        // its ready check (acquire) sees read == position; the fences order it for the other readers.
        s.read.store(read0 + n, std::memory_order_release);
        if (ipc::ShmProducer::instance()->is_active()) {
            ipc::ShmProducer::instance()->write_progress(s.request.id, read0 + n, (int64_t) s.request.tokens.size());
        }
        const double chunk_rate = chunk_ms > 0 ? (n / (chunk_ms / 1000.0)) : 0.0;
        {
            std::lock_guard<std::mutex> lock(s_stdout_mutex);
            std::printf("R %llu PP %lld %zu %.1f %.1f %s\n", (unsigned long long) s.request.id, (long long) (read0 + n),
                        s.request.tokens.size(), chunk_ms, chunk_rate, s.cache_tier.c_str());
            std::fflush(stdout);
        }
        // R2 (STRATA_PLE_PREFETCH, default off; a no-op without the env): this slot still has prompt left, so
        // its next run() call will want the next chunk's PLE rows - gather them on a thread now, while the
        // rotation runs the other slots.  The token window stays inside the request (read0 + n + n_next <=
        // position <= tokens.size()) and is copied inside the call; stage 0 owns layer 1, the PLE block's layer.
        if (const int64_t pos = s.position.load(std::memory_order_relaxed); read0 + n < pos)
            s.stages[0].prompt.ple_prefetch_next(s.request.tokens.data() + read0 + n,
                                                 std::min<int64_t>(c.prefill_chunk, pos - (read0 + n)));
        return true;
    };
    std::jthread pump;
    {
        const bool pump_on = [] {
            const char* v = std::getenv("STRATA_PREFILL_PUMP");
            return v == nullptr || std::atoi(v) != 0;   // default ON; 0 = the pre-change A/B arm
        }();
        if (pump_on) pump = std::jthread([&](std::stop_token stop) {
            strata::kernels::cpu::adopt_spawn_mask();   // STRATA_AUX_WIDE: the chunk pump is not the host loop
            // jthread's destructor requests stop without a notify; this callback wakes the wait.
            std::stop_callback wake(stop, [&] { m.pump_cv.notify_all(); });
            for (;;) {
                std::unique_lock<std::mutex> lock(m.pump_mutex);
                m.pump_cv.wait(lock, [&] {
                    if (stop.stop_requested()) return true;
                    if (m.pump_paused) return false;
                    for (const auto& ptr : m.slots) {
                        const auto& s = *ptr;
                        if (s.active.load() && s.read.load(std::memory_order_acquire) < s.position.load(std::memory_order_relaxed))
                            return true;
                    }
                    return false;
                });
                if (stop.stop_requested()) return;
                size_t pick = m.slots.size();
                const size_t rot_snapshot = m.prompt_rotation.load(std::memory_order_relaxed);
                for (size_t j = 0; j < m.slots.size(); ++j) {
                    const size_t i = (rot_snapshot + j) % m.slots.size();
                    const auto& s = *m.slots[i];
                    if (s.active.load() && s.read.load(std::memory_order_acquire) < s.position.load(std::memory_order_relaxed)) {
                        pick = i;
                        break;
                    }
                }
                if (pick == m.slots.size()) continue;   // raced away (cancel/exit); the wait re-checks
                m.prompt_rotation.store((pick + 1) % m.slots.size(), std::memory_order_relaxed);
                m.pump_busy = true;
                lock.unlock();
                std::string chunk_err;
                const bool ok = run_chunk(pick, chunk_err);
                lock.lock();
                m.pump_busy = false;
                if (!ok) {
                    m.pump_err = chunk_err;
                    m.pump_failed.store(true, std::memory_order_release);
                }
                m.pump_cv.notify_all();
                if (m.pump_failed.load(std::memory_order_acquire)) return;
            }
        });
    }
    // Lane pipeline-prefill: with STRATA_PREFILL_CHAIN a chunk boundary can still have stage-1 runs in
    // flight (that is the overlap).  The fences below guard admission/CSTOP/adapt/exit, which need FULL
    // quiescence (session_zero, cache swaps), so they drain the deferred chains too.  A no-op when the
    // feature is off (no chain future is ever valid then).
    auto chain_drain_all = [&]() -> bool {
        for (auto& ptr : m.slots) {
            ptr->stages[0].prompt.ple_prefetch_drain();   // R2: no prefetch thread outlives a fence or exit
            if (!ptr->stages[0].prompt.chain_wait(err)) return false;
        }
        return true;
    };
    // Hold the pump at a chunk boundary: it may not START another chunk; one already in flight finishes.
    auto pump_fence = [&]() -> bool {
        if (!pump.joinable()) return chain_drain_all();
        std::unique_lock<std::mutex> lock(m.pump_mutex);
        m.pump_paused = true;
        m.pump_cv.notify_all();
        m.pump_cv.wait(lock, [&] { return !m.pump_busy || m.pump_failed.load(std::memory_order_acquire); });
        if (m.pump_failed.load(std::memory_order_acquire)) { err = m.pump_err; return false; }
        return chain_drain_all();
    };
    auto pump_resume = [&]() {
        if (!pump.joinable()) return;
        std::lock_guard<std::mutex> lock(m.pump_mutex);
        m.pump_paused = false;
        m.pump_cv.notify_all();
    };
    auto pump_check = [&]() -> bool {   // a chunk failure is reported exactly where the inline call failed
        if (!m.pump_failed.load(std::memory_order_acquire)) return true;
        std::lock_guard<std::mutex> lock(m.pump_mutex);
        err = m.pump_err;
        return false;
    };
    std::deque<Request> pending;
    std::unordered_set<uint64_t> live;
    size_t rotation = 0;   // prompt_rotation is pump-owned (Impl::prompt_rotation)
    const char* watchdog_env = std::getenv("STRATA_WATCHDOG_S");
    const int watchdog_seconds = watchdog_env ? std::max(0, std::atoi(watchdog_env)) : 60;
    watchdog = std::jthread([watchdog_seconds](std::stop_token stop) {
        auto& progress = core::progress();
        uint64_t last = progress.beats.load();
        auto since = Clock::now();
        while (!stop.stop_requested()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            const auto now = Clock::now();
            const auto beat = progress.beats.load();
            if (!progress.busy.load() || beat != last) { last = beat; since = now; continue; }
            if (watchdog_seconds > 0 && now - since >= std::chrono::seconds(watchdog_seconds)) {
                std::fprintf(stderr, "strata-agx concurrent: no progress for %d seconds (%s); stopping stalled engine\n",
                             watchdog_seconds, progress.where.load());
                if (auto diag = core::diag_verify_fn().load()) diag(stderr);
                if (auto diag = core::diag_pool_fn().load()) diag(stderr);
                core::release_gpu_waits(stderr);   // #267: no spin kernel outlives the process
                std::fflush(stderr);
                std::abort();
            }
        }
    });
    const size_t radix_slots = [] {
        const char* e = std::getenv("STRATA_RADIX_VRAM_SLOTS");
        return e ? (size_t) std::atoi(e) : (size_t) 4;
    }();
    const size_t radix_host_slots = [&] {
        const char* e = std::getenv("STRATA_RADIX_HOST_SLOTS");
        if (e && std::atoi(e) > 0) return (size_t) std::atoi(e);
        if (c.conversation_cache_slots > 0) return (size_t) c.conversation_cache_slots;
        return (size_t) 64;
    }();
    core::RadixTree radix_tree(radix_slots, radix_host_slots);
    const char* nvme_dir_env = std::getenv("STRATA_NVME_TIER_DIR");
    if (nvme_dir_env && *nvme_dir_env) {
        auto nvme_tier = std::make_shared<core::NVMeStorageTier>(std::string(nvme_dir_env));
        radix_tree.set_nvme_tier(nvme_tier);
        if (radix_tree.restore_manifest()) {
            std::fprintf(stderr, "strata-agx concurrent: restored persistent L3 RadixTree topology from NVMe (%zu cached NVMe snapshots, %zu total nodes)\n",
                         radix_tree.cached_nvme_snapshot_count(), radix_tree.total_nodes());
        }
        std::fprintf(stderr, "strata-agx concurrent: RadixTree DirectStorage L3 NVMe tier enabled (%s)\n", nvme_tier->base_path().c_str());
    }
    std::fprintf(stderr, "strata-agx concurrent: RadixTree HiCache L2 host-RAM parking enabled (VRAM slots: %zu, Host-RAM slots: %zu)\n",
                 radix_slots, radix_host_slots);
    core::RadixCompactor radix_compactor(&radix_tree);
    std::fprintf(stderr, "strata-agx concurrent: ContinuousScheduler enabled (slots: %zu)\n",
                 m.slots.size());
    // WEDGE FIX (2026-10-05): the radix snapshot/insert/evict path issues device-syncing frees
    // (RadixNode::park_to_host -> cudaFree).  Run while a stage-1 pass is in flight, that free parks
    // the host inside cuMemFree while the GPU spins on unserviced doorbells (the crossdev x retention
    // wedge; gdb: evict_lru_locked -> park_to_host -> cudaFree under retire_unit).  Snapshots taken
    // while the overlap loop runs are queued here and flushed at the safe points below, strictly
    // before every admission (service_input), where nothing is in flight.
    // Snapshot size ceiling: a capture copies the session's full QSA KV (multi-GB at 254K-class)
    // and evictions can park those copies to pinned host RAM; the 3x254K acceptance host-OOM'd
    // once with captures on (2026-10-07).  Prompts above the ceiling stay capture-free (the
    // pre-capture memory profile), which is the acceptance's own shape.
    constexpr int64_t kSnapshotMaxPrefixTokens = 262144;
    std::vector<Impl::Slot*> pending_snapshots;
    bool defer_snapshots = false;   // set from the overlap gate below (finish() is defined earlier)
    bool pipeline_in_flight = true; // S26f3: maintained by the round loop; gates admission-time flushes
    auto save_slot_snapshot = [&](Impl::Slot& s, int64_t prefix_len, const std::vector<int32_t>& tokens, bool safe_no_evict = false) {
        if (prefix_len < 64 || prefix_len > kSnapshotMaxPrefixTokens || tokens.size() < (size_t) prefix_len) return;
        s.saved_prefix = prefix_len;
        s.saved_consumed.assign(tokens.begin(), tokens.begin() + prefix_len);
        for (size_t st = 0; st < m.stages.size(); ++st) {
            auto& ss = s.stages[st];
            const core::OnDevice on(m.stages[st].device);
            if (!ss.gdn_saved && ss.state->gdn_alloc > 0) {
                cudaMalloc(&ss.gdn_saved, (size_t) ss.state->gdn_alloc * core::gdn_state_floats(g) * sizeof(float));
            }
            if (ss.gdn_saved && ss.state->gdn_state) {
                cudaMemcpyAsync(ss.gdn_saved, ss.state->gdn_state,
                                (size_t) ss.state->gdn_alloc * core::gdn_state_floats(g) * sizeof(float),
                                cudaMemcpyDeviceToDevice, m.stage_rt[st].prompt_stream);
            }
            if (!ss.ple_saved && ss.state->ple_hist) {
                cudaMalloc(&ss.ple_saved, (size_t) core::ple_hist_bytes());
            }
            if (ss.ple_saved && ss.state->ple_hist) {
                cudaMemcpyAsync(ss.ple_saved, ss.state->ple_hist, (size_t) core::ple_hist_bytes(),
                                cudaMemcpyDeviceToDevice, m.stage_rt[st].prompt_stream);
            }
            if (!ss.R_saved && ss.state->block.R) {
                cudaMalloc(&ss.R_saved, (size_t) g.hc * g.n_embd * sizeof(float));
            }
            if (ss.R_saved && ss.state->block.R) {
                cudaMemcpyAsync(ss.R_saved, ss.state->block.R, (size_t) g.hc * g.n_embd * sizeof(float),
                                cudaMemcpyDeviceToDevice, m.stage_rt[st].prompt_stream);
            }
            ss.ple_prev_saved[0] = ss.state->ple_prev[0];
            ss.ple_prev_saved[1] = ss.state->ple_prev[1];
            ss.ple_token_saved = ss.state->ple_token;
        }
        for (size_t st = 0; st < m.stage_rt.size(); ++st) {
            const core::OnDevice on(m.stages[st].device);
            cudaStreamSynchronize(m.stage_rt[st].prompt_stream);
        }
        // SPRINT 4: Dynamic RadixTree insert
        std::vector<int> r_devs;
        std::vector<const core::SessionState*> r_states;
        std::vector<const float*> r_R;
        std::vector<void*> r_streams;
        for (size_t st = 0; st < m.stages.size(); ++st) {
            r_devs.push_back(m.stages[st].device);
            r_states.push_back(s.stages[st].state);
            r_R.push_back(s.stages[st].R_saved ? s.stages[st].R_saved : s.stages[st].state->block.R);
            r_streams.push_back((void*) m.stage_rt[st].prompt_stream);
        }
        if (!safe_no_evict) radix_tree.evict_lru(radix_slots);   // S26f7
        auto r_node = (radix_slots == 0)
            ? radix_tree.insert_direct_device_to_host(tokens.data(), tokens.size(), prefix_len, r_devs, r_states, r_R, g, r_streams, safe_no_evict)
            : radix_tree.insert(tokens.data(), tokens.size(), prefix_len, r_devs, r_states, r_R, g, r_streams, safe_no_evict);
        if (r_node) {
            if (s.radix_node && s.radix_node != r_node) radix_tree.release(s.radix_node);
            s.radix_node = r_node;
            radix_tree.acquire(s.radix_node);
        }
    };
    auto flush_pending_snapshots = [&]() {
        for (auto* q : pending_snapshots) {
            save_slot_snapshot(*q, (int64_t) q->consumed.size(), q->consumed);
            if (q->radix_node) { radix_tree.release(q->radix_node); q->radix_node = nullptr; }
        }
        pending_snapshots.clear();
    };
    // P18 L1: capture the prompt-end state (state@position - exactly where the first verify window
    // opens) for every slot whose prefill just completed.  With the sound gate in place, an
    // identical or longer resend of the same prompt then forks at `position` and runs only the
    // window - the serial path's own first-window shape.  Runs in the same safe-point envelope as
    // flush_pending_snapshots; the one-per-admission flag keeps it single-shot.
    auto capture_prompt_end_snapshots = [&]() {
        for (auto& ptr : m.slots) {
            auto& s = *ptr;
            if (!s.active.load(std::memory_order_relaxed) || !s.first || s.prompt_snap_done) continue;
            const int64_t pos = s.position.load(std::memory_order_relaxed);
            const int64_t cur_read = s.read.load(std::memory_order_acquire);
            if (cur_read < pos) continue;

            const int64_t n = (int64_t) s.consumed.size();
            // S26f9: end-append prompts re-tokenize at the junction, so a snapshot at the very end
            // is unusable for the next (appended) turn.  Capture the deepest real state below the
            // end instead: either exactly at the end (small prompts) or the last pre-end state
            // (within the final prefill chunk).  The state is copied NOW, so its position is real.
            if (n < 64 || n > pos) continue;
            if (n != pos && (pos - n) > 6208) continue;
            s.prompt_snap_done = true;
            save_slot_snapshot(s, n, s.consumed, /*safe_no_evict=*/false);   // S26f7/S26f9
        }
    };
    auto finish = [&](Impl::Slot& s, const char* reason) {
        const bool is_cancelled = (reason && (std::strcmp(reason, "cancel") == 0 ||
                                              std::strcmp(reason, "abort") == 0 ||
                                              std::strcmp(reason, "cstop") == 0));
        const bool defer_snapshot = defer_snapshots;
        if (!is_cancelled) {
            if (!defer_snapshot) save_slot_snapshot(s, (int64_t) s.consumed.size(), s.consumed);
        }
        const double decode = s.first ? 0 : elapsed(s.decode_start);
        // P4: the trailing field is reused_prefix_tokens (server.py _parse_done maps it to
        // `reused`/cached_tokens). Live retention fills it with the tokens this request resumed;
        // 0 = the request read its whole prompt (fresh admission - all of today's traffic).
        // Task 1: Expose byte-accurate host_snapshot_bytes() from RadixTree in DONE telemetry
        const size_t live_l2_bytes = radix_tree.host_snapshot_bytes();
        {
            std::lock_guard<std::mutex> lock(s_stdout_mutex);
            std::printf("R %llu DONE %lld %zu %.1f %.1f %s %lld %lld %lld %s %zu\n", (unsigned long long) s.request.id,
                        (long long) s.generated, s.request.tokens.size(), s.prompt_ms.load(), decode, reason,
                        (long long) s.accepted, (long long) s.offered, (long long) s.reused_prefix, s.cache_tier.c_str(), live_l2_bytes);
            std::fflush(stdout);
        }
        if (defer_snapshot && !is_cancelled) {
            pending_snapshots.push_back(&s);   // flushed with its snapshot at a safe point
        } else if (s.radix_node) {
            radix_tree.release(s.radix_node);
            s.radix_node = nullptr;
        }
        for (size_t st = 0; st < m.stages.size(); ++st) {
            const core::OnDevice on(m.stages[st].device);
            cudaStreamSynchronize(m.stage_rt[st].prompt_stream);
        }
        s.stages[0].prompt.embd_rows = nullptr;
        live.erase(s.request.id);
        s.in_flight_prefix_hash.store(0, std::memory_order_release); s.active.store(false);
        int finish_slot_idx = -1;
        for (size_t si = 0; si < m.slots.size(); ++si) {
            if (m.slots[si].get() == &s) { finish_slot_idx = (int) si; break; }
        }
        if (finish_slot_idx >= 0) continuous_sched.recycle_slot(finish_slot_idx);
        // LANE sched-impl P1 (admission wake; T4 L7/L13 micro): a freed slot is re-admitted at the
        // next service_input, but the bottom idle park sleeps on input->cv (2 ms tick). Wake it when
        // queued work waits so a STOP re-admits without the tick. Wake-only: no scheduling decision
        // changes; admission still only runs in service_input (safe point); H1 region untouched.
        if (!pending.empty()) input->cv.notify_all();
    };
    // LANE failclean (K1b): a batch capture refused for want of the VRAM reserve fails THIS round:
    // every request that participated gets its ERR line and its slot released exactly as finish()
    // releases one, the refusal is named loudly, and the engine stays up for the next round.  The
    // Verifier destroys the refused graph and caches nothing (prepare_batch), so the next round
    // recaptures cleanly.  One semantics for the serial round and BOTH stage-overlap launch sites.
    auto refuse_round = [&](const std::vector<Impl::Slot*>& round, const std::string& why) {
        for (auto* ptr : round) {
            auto& rs = *ptr;
            if (!rs.count) continue;
            error(rs.request.id, why);
            live.erase(rs.request.id);
            rs.active.store(false);
            rs.count = 0;
        }
        std::fprintf(stderr, "strata-agx concurrent: batch round refused (%s); requests errored, engine up\n", why.c_str());
    };
    // ============================ STAGE-PIPELINE OVERLAP (lane overlap) ============================
    // stage0[N+1] || stage1[N]: with the hand-off DOUBLE-BUFFERED by parity, stage0 of the next
    // unit runs on CUDA0 while stage1 of the current unit still reads the other parity on CUDA1.
    // The two passes in flight own DISJOINT slots (a slot whose window is in flight is excluded
    // from the next unit until its epilogue drafted the following window), and every mapped-input
    // write / flag reset / capture happens only after the pass it belongs to has completed (the
    // driver waits its completion event).  STRATA_STAGE_OVERLAP=0 restores the serial loop; a
    // single-stage engine or a 3+-stage split never takes this path.
    // LANE split1-r2: the two-device stage-1 park is fixed - the stage-1 launch leaves no host
    // driver call on the in-flight cross-device stream (verify.cpp's launch_pass) - so the
    // two-device split takes the overlap path too.  The 2-stage shape is a real precondition
    // (the hand-off parity machinery below is exactly two-staged); STRATA_STAGE_OVERLAP=0
    // restores the serial loop.
    // LANE split1-r2fix-v3: the v2 build's record refusal is fixed at the object - a completion
    // event lives in the context it is recorded in, so ev0/ev1 are created (and destroyed) on
    // their own stage's device in the event setup below; the launch span stays as v2 left it.
    // 2026-10-01 measurement: the corrected two-device overlap is functionally sound but costs ~30%
    // decode at the 5x87.5K shape (23 vs 32-37 tok/s; prefill flat) - serial is the best-known default
    // for two-DEVICE splits; cross-device overlap stays available as an explicit experiment
    // (STRATA_STAGE_OVERLAP_CROSSDEV=1).  Same-device keeps its historic default-on behavior.
    const bool overlap = m.stages.size() == 2 && [&] {
        const char* v = std::getenv("STRATA_STAGE_OVERLAP");
        if (v != nullptr && std::atoi(v) == 0) return false;              // explicit serial
        if (m.stages[0].device != m.stages[1].device) {                   // two-device: opt-in
            const char* x = std::getenv("STRATA_STAGE_OVERLAP_CROSSDEV");
            return x != nullptr && std::atoi(x) != 0;
        }
        return true;                                                       // same-device: default on
    }();
    defer_snapshots = overlap;   // WEDGE FIX: finish() (defined earlier) defers radix snapshots under overlap
    // R2 (split1-v3): a completion event is a CUDA object of the device whose pass it times - ev0 is
    // recorded on stage 0's stream, ev1 on stage 1's - so each is created (and destroyed, on every
    // exit path) on its OWN stage's device: the same context semantics the same-device overlap always
    // ran under, where creation and record contexts coincide.  The v2 build created both under run()'s
    // entry context (CUDA0) and the stage-1 record failed on the CUDA1 stream (P1 2026-10-01 19:14Z,
    // "verify: the pass completion event failed").  Same-device: the two devices are equal and already
    // current, so the guards are no-ops and the call sequence is unchanged.
    const int overlap_dev0 = m.stages.empty() ? -1 : m.stages[0].device;
    const int overlap_dev1 = m.stages.size() > 1 ? m.stages[1].device : overlap_dev0;
    cudaEvent_t ev0[3] = {}, ev1[3] = {};
    bool ev1_valid[3] = {false, false, false};
    struct OverlapEventsGuard {   // every exit path of run() destroys them, each on its own device
        cudaEvent_t* ev0; cudaEvent_t* ev1; int dev0; int dev1;
        ~OverlapEventsGuard() {
            for (int q = 0; q < 3; ++q) {
                if (ev0[q]) { const core::OnDevice on(dev0); cudaEventDestroy(ev0[q]); }
                if (ev1[q]) { const core::OnDevice on(dev1); cudaEventDestroy(ev1[q]); }
            }
        }
    } overlap_events_guard{ev0, ev1, overlap_dev0, overlap_dev1};
    if (overlap) {
        for (int q = 0; q < 3; ++q) {
            bool made = false;
            { const core::OnDevice on(overlap_dev0); made = cudaEventCreateWithFlags(&ev0[q], cudaEventDisableTiming) == cudaSuccess; }
            if (made) { const core::OnDevice on(overlap_dev1); made = cudaEventCreateWithFlags(&ev1[q], cudaEventDisableTiming) == cudaSuccess; }
            if (!made) {
                err = "concurrency: the stage-overlap events failed";
                return 1;
            }
        }
        std::fprintf(stderr, "strata-agx concurrent: stage overlap on: stage0[N+1] runs with stage1[N] "
                             "(3-slot TripleBufferIPC ring hand-off, per-slot captures)\n");
    }

    // N1 (C2 acceptance kit, env-gated, default off): the stale-sink negative control.  Point a
    // stage's publish target at ANOTHER stage's sink so the executing verifier's empty-plan
    // fallback fires (verify.cpp service_one) and the round completes with silently wrong tokens,
    // no hang.  The decision is resolved ONCE here - the env cannot change under a serve loop, so
    // with it unset the round sites below keep only this cached branch and production behavior is
    // unchanged.  The poison is applied at EVERY site that points a stage's publish target,
    // because the pool reads split_drive.plan[st] at the LAYER DISPATCH of whichever pass it
    // services:
    //   * the top-of-round refresh (which skips stages >= 1 on the overlap path), and
    //   * the overlap path's launch-site refresh of stages >= 1: the launched stage-1 pass is
    //     serviced in a LATER round (the next drive, or the no-windows branch that closes the
    //     previous unit), so a poison applied at the top of a round alone is overwritten by that
    //     refresh before any stage-1 dispatch reads it - the stale arm went token-identical
    //     (0/12 runs differed) on the two-device overlap path while the same hook was VALID on
    //     the serial path (N1-hook forensics, 2026-10-01).
    const char* const n1_stale_env = std::getenv("STRATA_PLAN_SINK_STALE");
    auto n1_poison_plan_sinks = [&]() {
        if (n1_stale_env == nullptr || stage_plans == nullptr || m.stages.size() < 2) return;
        const int stale = std::abs(std::atoi(n1_stale_env)) % (int) m.stages.size();
        stage_plans[stale] = stage_plans[(stale + 1) % (int) m.stages.size()];
    };
    // One round unit: the windows of one iteration, their stage-0 and stage-1 pass owners, and the
    // slots whose epilogue (acceptance, commit, drafts) is still pending.
    struct Unit {
        bool valid = false;
        bool single = false;                                  // one window vs a batch round
        int parity = 0;
        Impl::Slot* slot = nullptr;                           // single: the window's slot
        int T = 0;
        const int32_t* tokens = nullptr;
        int64_t pos0 = 0;
        int32_t* out = nullptr;
        std::vector<core::Verifier::BatchWindow> windows;     // batch: the stage-0 members
        std::vector<core::Verifier::BatchWindow> chain;       // batch: the stage-1 members
        std::vector<Impl::Slot*> ready;                       // slots whose epilogue is pending
        core::Verifier* lane0 = nullptr;
        core::Verifier* lane1 = nullptr;
        Clock::time_point t_start{};
    } prev;
    int unit_parity = 0;
    auto unit_lane = [&](Unit& u, size_t st) -> core::Verifier* {
        return u.single ? &u.slot->stages[st].verify : &batch[st];
    };
    // Every verifier of a unit runs on the unit's parity: the coordinator (batch) or the slot's
    // stage chain (single), and the batch members (their phase-5/phase-0 copies bake the pointers).
    auto unit_parity_set = [&](Unit& u, int q) {
        if (u.single) {
            for (size_t st = 0; st < m.stages.size(); ++st) u.slot->stages[st].verify.set_hand_parity(q);
            return;
        }
        for (size_t st = 0; st < m.stages.size(); ++st) {
            batch[st].set_hand_parity(q);
            for (const auto& w : u.windows) w.verifier->set_hand_parity(q);
            for (const auto& w : u.chain) w.verifier->set_hand_parity(q);
        }
    };
    // The serial round loop's tail, once per retired unit: acceptance, commit, the printed tokens
    // and the next window's drafts.  Both paths run exactly this.
    auto retire_unit = [&](Unit& u) -> bool {
        // S1c-fix: the fused draft cap's budget term - the round's fair per-slot row allocation
        // (schedule_rows' round-robin floor over the served slots).  An INPUT of the window allocator,
        // never an output of any window the chain sized, so it cannot feed back into L and lock (the
        // first build's T-1 term did exactly that).  The step path ignores it; same value for all slots.
        const int alloc_share = (int) std::max<int64_t>(1, (int64_t) c.rows / (int64_t) std::max<size_t>(1, u.ready.size()));
        // LANE hostloop (commit batch): phase 1 decides acceptance and LAUNCHES every slot's commit
        // chain; phase 2 waits each chain and finishes that slot's epilogue (prints, drafts).  The
        // per-slot order (commit -> its own draft) and the print order are unchanged; the commit
        // kernels of different slots now overlap instead of running one-slot-at-a-time.
        struct Trail { Impl::Slot* s; int keep; bool eos; };
        std::vector<Trail> trail;
        trail.reserve(u.ready.size());
        {
            const auto commit_start = Clock::now();
            for (auto* ptr : u.ready) {
                auto& s = *ptr;
                if (!s.count) continue;
                // LANE hostloop: a CSTOP processed by an intervening service_input - or by the epilogue
                // reorder below - has already finished this slot; never finish (DONE) it twice.
                if (!s.active.load()) continue;
                // A CSTOP can arrive while this unit's window is in flight (the serial loop only ever
                // saw CSTOP with nothing in flight): the request was already erased from `live`, so
                // finish it as cancelled instead of committing its window.
                if (!live.count(s.request.id)) { finish(s, "cancel"); continue; }
                // Task 3.1: Speculative Draft Verification with Leviathan / vLLM Rejection Sampling (ALGO-01 & ALGO-02)
                int accepted = 0;
                const bool is_sampled = !s.request.sampling.greedy && s.request.sampling.temperature >= 1e-4f;
                if (!is_sampled) {
                    // Greedy mode: linear prefix match
                    while (accepted < s.count - 1 && s.window[accepted + 1] == s.output[accepted]) ++accepted;
                } else {
                    // Temperature > 0: Leviathan / vLLM speculative residual rejection sampling
                    for (int t = 0; t < s.count - 1; ++t) {
                        const int32_t draft_tok = s.window[t + 1];
                        const int32_t target_tok = s.output[t];
                        if (draft_tok == target_tok) {
                            accepted = t + 1;
                            continue;
                        }
                        const float q = (s.probability[t] > 0.0f) ? s.probability[t] : 1.0f;
                        const float p = (s.stages[0].verify.target_draft_prob(t) > 0.0f)
                                            ? s.stages[0].verify.target_draft_prob(t)
                                            : 0.0f;
                        const float alpha = std::min(1.0f, p / q);
                        uint64_t draw_state = s.request.sampling.seed ^ ((uint64_t)s.position.load() + t);
                        draw_state = draw_state * 6364136223846793005ULL + 1442695040888963407ULL;
                        const float u = (float)((draw_state >> 32) & 0xFFFFFF) / 16777216.0f;
                        bool accept_draft = false;
                        if (alpha > p && p < 1.0f) {
                            const float residual_thresh = (alpha - p) / (1.0f - p);
                            if (u < residual_thresh) {
                                accept_draft = true;
                            }
                        }
                        if (accept_draft) {
                            s.output[t] = draft_tok;
                            accepted = t + 1;
                        } else {
                            // Rejected: sample replacement token from true normalized residual distribution
                            // p_prime(y) = max(0, p(y) - q(y)) / sum_z max(0, p(z) - q(z))
                            const int64_t nv = s.stages[0].verify.vocab();
                            if (nv > 0) {
                                std::vector<float> logits((size_t) nv);
                                if (s.stages[0].verify.copy_logits(t, logits.data())) {
                                    float max_l = -1e30f;
                                    for (int64_t i = 0; i < nv; ++i) {
                                        if (logits[i] > max_l) max_l = logits[i];
                                    }
                                    const float inv_temp = 1.0f / s.request.sampling.temperature;
                                    float sum_exp = 0.0f;
                                    for (int64_t i = 0; i < nv; ++i) {
                                        logits[i] = std::exp((logits[i] - max_l) * inv_temp);
                                        sum_exp += logits[i];
                                    }
                                    const float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;
                                    float sum_res = 0.0f;
                                    for (int64_t i = 0; i < nv; ++i) {
                                        const float p_y = logits[i] * inv_sum;
                                        const float q_y = (i == (int64_t) draft_tok) ? q : 0.0f;
                                        const float diff = std::max(0.0f, p_y - q_y);
                                        logits[i] = diff;
                                        sum_res += diff;
                                    }
                                    if (sum_res > 0.0f) {
                                        draw_state = draw_state * 6364136223846793005ULL + 1442695040888963407ULL;
                                        const float r = (float)((draw_state >> 32) & 0xFFFFFF) / 16777216.0f;
                                        const float thresh = r * sum_res;
                                        float cdf = 0.0f;
                                        int32_t rep_tok = target_tok;
                                        for (int64_t i = 0; i < nv; ++i) {
                                            cdf += logits[i];
                                            if (cdf >= thresh) {
                                                rep_tok = (int32_t) i;
                                                break;
                                            }
                                        }
                                        s.output[t] = rep_tok;
                                    }
                                }
                            }
                            break;
                        }
                    }
                }
                int keep = accepted + 1;

                // ALGO-01: Multi-path tree speculation lacks intra-tree causal masking in verify attention.
                // Restrict verification strictly to the contiguous linear prefix above to ensure Branch A
                // rejected tokens are NEVER committed to the KV cache when alternate branches are evaluated.

                static const bool s_spec_tree_telemetry = [] {
                    const char* v = std::getenv("STRATA_SPEC_TREE_TELEMETRY");
                    return v != nullptr && std::atoi(v) != 0;
                }();
                if (s_spec_tree_telemetry && s.count > 1) {
                    std::fprintf(stderr, "[SPEC_TREE] slot=%llu offered=%d accepted=%d rate=%.1f%%\n",
                                 (unsigned long long)s.request.id, s.count - 1, accepted,
                                 (s.count > 1 ? (100.0 * accepted / (s.count - 1)) : 0.0));
                }

                bool eos = false;
                for (int t = 0; t < keep; ++t) if (std::find(c.eos.begin(), c.eos.end(), s.output[t]) != c.eos.end()) {
                    keep = t + 1; eos = true; break;
                }
                if (!s.stages[0].verify.commit_launch(keep, err)) return false;
                trail.push_back({&s, keep, eos});
            }
            commit_ms += elapsed(commit_start);
        }
        struct DraftJob { Impl::Slot* s; int keep; int64_t p; };
        std::vector<DraftJob> jobs;   // wavefront: the deferred drafts (launched after every slot's prints)
        for (auto& p : trail) {
            auto& s = *p.s;
            const int keep = p.keep;
            const auto commit_start = Clock::now();
            if (!s.stages[0].verify.commit_wait(err)) return false;
            commit_ms += elapsed(commit_start);
            produced += keep;
            s.offered += s.count - 1; s.accepted += keep - 1;
            offered_total += s.count - 1; accepted_total += keep - 1;
            ++count_hist[std::min(s.count, 8)]; ++keep_hist[std::min(keep, 8)];
            const int spec_src = s.lookup ? 1 : 0;
            ++source_windows[spec_src];
            source_offered[spec_src] += s.count - 1; source_accepted[spec_src] += keep - 1;
            for (int t = 1; t < s.count; ++t) ++pos_offered[std::min(t, 8)];
            for (int t = 1; t < keep; ++t) ++pos_accepted[std::min(t, 8)];
            int retire_slot_idx = -1;
            for (size_t si = 0; si < m.slots.size(); ++si) {
                if (m.slots[si].get() == &s) { retire_slot_idx = (int) si; break; }
            }
            for (int t = 0; t < keep; ++t) {
                if (retire_slot_idx >= 0) continuous_sched.step_token(retire_slot_idx, s.output[t], (t == keep - 1 && p.eos));
                s.consumed.push_back(s.output[t]); s.suffix.append(s.output[t]); ++s.generated;
            }
            {
                std::lock_guard<std::mutex> lock(s_stdout_mutex);
                for (int t = 0; t < keep; ++t) {
                    if (ipc::ShmProducer::instance()->is_active()) {
                        if (!ipc::ShmProducer::instance()->write_token(s.request.id, s.output[t])) {
                            std::printf("R %llu T %d\n", (unsigned long long) s.request.id, s.output[t]);
                        }
                    } else {
                        std::printf("R %llu T %d\n", (unsigned long long) s.request.id, s.output[t]);
                    }
                }
                std::fflush(stdout);
            }
            s.first = false;
            if (p.eos || s.generated >= s.request.max_new || s.position.load() + keep >= s.max_context) {
                finish(s, p.eos ? "stop" : "length"); continue;
            }
            // Catch-up consumes the verified window; limit the extra speculative chain near the context boundary.
            s.draft->set_max_drafts((int) std::min<int64_t>(c.mtp_window_rows - 1, s.max_context - (s.position.load() + keep)));
            s.draft->set_alloc_share(alloc_share);   // S1c-fix: the fused cap's budget term for this round
            if (wavefront) {
                // STRATA_DRAFT_WAVEFRONT: this slot's chain is launched below, after EVERY slot's commit,
                // prints and position update; its cell base is THIS window's start, and the per-slot
                // drafter state is disjoint, so the chains overlap on their own streams.
                jobs.push_back({&s, keep, s.position.load()});
                s.policy.observe(s.lookup, s.count, keep - 1, s.match, elapsed(u.t_start));
                s.current = s.output[keep - 1];
                s.position.store(s.position.load() + keep);
                s.read.store(s.position.load());
                continue;
            }
            const auto draft_start = Clock::now();
            if (!s.draft->draft(s.count, s.output, s.position.load(), keep - 1, s.drafts, err, s.probability, s.request.spec_min_p)) return false;
            draft_ms += elapsed(draft_start);
            s.policy.observe(s.lookup, s.count, keep - 1, s.match, elapsed(u.t_start));
            s.current = s.output[keep - 1];
            s.position.store(s.position.load() + keep);
            s.read.store(s.position.load());   // prompt done; read tracks position until the next turn
        }
        if (!jobs.empty()) {
            // all launches -> all waits (STRATA_DRAFT_WAVEFRONT): every slot's chain runs on its own
            // drafter stream, so the launches overlap instead of serializing per slot.
            const auto wave_start = Clock::now();
            for (auto& j : jobs)
                if (!j.s->draft->draft_begin(j.s->count, j.s->output, j.p, j.keep - 1, err)) return false;
            for (auto& j : jobs)
                if (!j.s->draft->draft_end(j.s->drafts, err, j.s->probability, j.s->request.spec_min_p, nullptr)) return false;
            draft_ms += elapsed(wave_start);
        }
        if (overlap)
            // The drafts are asynchronous on the drafter's stream.  The serial round had a whole
            // stage-0 pass between them and the next stage-1 pass that overwrites the residual R
            // they read; the pipeline does not, so the next pass is gated on the drafters' idle.
            for (auto* ptr : u.ready) if (!ptr->draft->idle(err)) return false;
        return true;
    };
    // LANE overlap-width (merged drain): a slot re-enters only after its unit's EPILOGUE ran
    // (acceptance -> commit -> the draft that writes the next window's tokens) - that is the data
    // dependency, not a policy.  What was policy is WHEN the drain ran: the old loop retired the
    // in-flight unit in a separate iteration (a second "round") and launched its successor only in
    // the one after.  With every slot inside the single in-flight unit (the locked-step shape: c
    // requests admitted together, c == slots), the exclusion left the ready set empty, so the loop
    // alternated launch-round / retire-only-round: rounds doubled (930 vs 492 at 5x87.5K, ~5.5
    // tok/round) and no stage-0/stage-1 pair ever overlapped (prev was always already retired when
    // the next unit launched).  The merged drain below retires the unit at the TOP of the iteration
    // and falls through to build + launch its successor in the SAME iteration: one round per unit,
    // full per-round batch width, like the serial loop.  Partial shapes (some slots mid-prompt)
    // keep the two-units-in-flight pipeline path exactly as it was.
    auto drain_prev_unit = [&]() -> int {
        if (!core::Verifier::drive_passes(nullptr, prev.lane1, err)) return 1;
        if (prev.single) { if (!prev.lane1->end_pass_window(prev.T, prev.out, err)) return 1; }
        else { if (!prev.lane1->end_pass_batch(prev.chain, err)) return 1; }
        target_ms += elapsed(prev.t_start);
        target_rows += prev.single ? prev.T : [&] { int64_t r = 0; for (const auto& w : prev.windows) r += w.count; return r; }();
        if (!retire_unit(prev)) return 1;
        prev.valid = false;
        return 0;
    };
    // The merged-drain precondition, read WITHOUT the ready-scan's side effects: could any slot
    // launch a window this instant?  Mirrors the scan's inclusion test exactly where it matters
    // (active, caught up, not owned by the in-flight unit, and the context / max_new headroom that
    // n >= 1 needs); the scan still owns finish()/allocation.  Over-inclusion is safe - the normal
    // path then runs and the scan decides; under-inclusion cannot happen, because every condition
    // here is necessary for the scan to produce a window for that slot.
    auto slot_launchable = [&](Impl::Slot& s) -> bool {
        if (!s.active.load() || s.read.load(std::memory_order_acquire) < s.position.load()) return false;
        if (overlap && prev.valid &&
            std::find(prev.ready.begin(), prev.ready.end(), &s) != prev.ready.end()) return false;
        if (s.position.load() >= s.max_context) return false;
        if (s.generated >= s.request.max_new) return false;
        return true;
    };
    // LANE alloc-empty: STRATA_SLOT_LAZY's deferred per-slot bring-up.  Same calls in the same order as
    // the boot-time creation (prepare()'s per-slot block, then run()'s), so an on-demand slot is
    // initialized exactly like a boot-time one; the difference is WHEN and WHERE the memory comes from -
    // each stage session is carved from that stage cache's tail (the released pairs fall back to
    // streaming; both residency tables are updated before the next plan, which is what keeps the captured
    // graphs safe - the adapt() doctrine), and the drafter/verifier allocations draw on the reserve the
    // auto cache sizing kept for them.  Runs at the safe point: the caller holds the pump fence and the
    // serial loop has no pass in flight.
    static constexpr int64_t kLazySlotFloorSlots = 128;   // the caches' shared floor (the lend plan's 128)
    auto bringup_slot = [&](Impl::Slot& s, std::string& err) -> bool {
        for (size_t st = 0; st < m.stages.size(); ++st) {
            const auto& sg = m.stages[st];
            const core::OnDevice on(sg.device);
            auto& gs = s.stages[st];
            if (!gs.fork_event) cudaEventCreateWithFlags(&gs.fork_event, cudaEventDisableTiming);
            uint8_t* base = nullptr;
            const int64_t slot_context = s.max_context > 0 ? s.max_context : c.context;
            const int64_t def_bytes = (int64_t) core::session_bytes(g, slot_context, m.session_k, sg.lb, sg.le);
            const int64_t first = sg.cache->release_tail_bytes(def_bytes, kLazySlotFloorSlots, &base);
            if (first < 0) {
                err = "concurrency: STRATA_SLOT_LAZY: the expert cache's tail cannot hold this slot's session (stage " +
                      std::to_string(st) + ")";
                return false;
            }
            if (sg.host_res != nullptr) {
                const int64_t cells = g.n_layers * g.n_expert;
                for (int64_t i = 0; i < cells; ++i) if (sg.host_res[i] >= first) sg.host_res[i] = core::kNotResident;
                if (sg.hits.d_res != nullptr &&
                    cudaMemcpy((void*) sg.hits.d_res, sg.host_res, (size_t) cells * sizeof(int32_t),
                               cudaMemcpyHostToDevice) != cudaSuccess) {
                    err = "concurrency: STRATA_SLOT_LAZY: residency upload failed";
                    return false;
                }
            }
            if (!core::session_init(g, slot_context, m.session_k, base, gs.owned, sg.lb, sg.le)) {
                err = "concurrency: STRATA_SLOT_LAZY: session initialization failed";
                return false;
            }
            gs.arena = base;
            gs.carved = true;
            if (st == 0) {
                // The PLE is a layer-1 module: only the stage that holds layer 1 wires it (prepare's note).
                gs.owned.ple = m.primary_state->ple;
                gs.owned.ple.hist = gs.owned.ple_hist;
                gs.owned.ple.prev = gs.owned.ple_prev;
                gs.owned.ple.token = &gs.owned.ple_token;
                // lane lazy-enabler: --batch-parallel composes with this mode; its per-slot scratch was
                // allocated in prepare - point the member stream at it (absent = the serial path, which
                // keeps the primary's scratch pointer exactly as before).
                if (gs.ple_scratch) gs.owned.ple.scratch = static_cast<float*>(gs.ple_scratch);
            }
        }
        // The drafter: prepare()'s block, including the shared head on the first owned slot that loads it.
        s.draft_owner = std::make_unique<core::MtpDrafter>();
        s.draft = s.draft_owner.get();
        static const core::ModelGeometry draft_geometry{};
        {
            const core::OnDevice on(m.stages.back().device);
            if (!s.draft->load(c.mtp_dir, draft_geometry, *s.stages.back().state, c.window, err, c.draft_context, m.shared_draft)) return false;
        }
        s.draft->set_max_drafts(c.mtp_window_rows - 1);
        // run()'s per-slot block: the verifiers, the prompt chain, the bind and the history.
        for (size_t st = 0; st < m.stages.size(); ++st) {
            const auto& sg = m.stages[st];
            const core::OnDevice on(sg.device);
            const bool last = st + 1 == m.stages.size();
            s.stages[st].verify.set_stage_ring(sg.lb, last ? -1 : sg.le,
                                               st == 0 ? nullptr : m.hand[st - 1].dev[0],
                                               last ? nullptr : m.hand[st].dev[0],
                                               st == 0 ? nullptr : m.hand[st - 1].dev[1],
                                               last ? nullptr : m.hand[st].dev[1],
                                               st == 0 ? nullptr : m.hand[st - 1].dev[2],
                                               last ? nullptr : m.hand[st].dev[2]);
            if (!s.stages[st].verify.init(*sg.wt, g, *s.stages[st].state, sg.hits, sg.head, c.window, err)) return false;
            s.stages[st].verify.set_pcie_mode(configured_pcie_mode);
            if (!last) s.stages[st].verify.set_next(&s.stages[st + 1].verify, user);
            s.stages[st].prompt.set_stage(sg.lb, last ? -1 : sg.le, last ? nullptr : &s.stages[st + 1].prompt);
            if (!s.stages[st].prompt.init(*sg.wt, g, *s.stages[st].state, source, sg.cache, sg.host_res,
                                          c.prefill_chunk, (void*) m.stage_rt[st].prompt_stream, err,
                                          m.stage_rt[st].prompt_workspace, m.stage_rt[st].prompt_bytes)) return false;
            s.stages[st].prompt.set_chain_defer(true);
            if (st == 0) s.stages[st].prompt.set_chain_gate(&m.chain_gate);
        }
        if (!s.draft->bind(*m.stages.back().wt, m.stages.back().head, s.stages[0].verify.final_R_all(), err)) return false;
        s.history.resize((size_t) c.window * 4096, -1);
        {
            const core::OnDevice on(m.stages.back().device);
            if (cudaMalloc(&s.history_device, s.history.size() * sizeof(int32_t)) != cudaSuccess) {
                err = "concurrency: STRATA_SLOT_LAZY: penalty buffer allocation failed";
                return false;
            }
        }
        auto* slot = &s;
        s.stages.back().prompt.on_chunk = [slot](const float* residual, int64_t n, int64_t position, std::string& e) {
            std::vector<int32_t> next((size_t) n);
            for (int64_t j = 0; j < n; ++j)
                next[(size_t) j] = (int32_t) (position + j + 1 < (int64_t) slot->request.tokens.size()
                                                  ? slot->request.tokens[(size_t) (position + j + 1)]
                                                  : 0);
            return slot->draft->prefill(residual, next.data(), n, position, e);
        };
        s.brought_up = true;
        size_t f0 = 0, t0 = 0, f1 = 0, t1 = 0;
        { const core::OnDevice on0(0); cudaMemGetInfo(&f0, &t0); }
        { const core::OnDevice on1(1); cudaMemGetInfo(&f1, &t1); }
        std::fprintf(stderr, "strata-agx concurrent: slot-lazy: slot sessions carved from the expert caches' tails (CUDA0 free %zu MiB, CUDA1 free %zu MiB)\n", f0 >> 20, f1 >> 20);
        return true;
    };
    // P4: one slot's admission - the exact sequence the loop always ran (bookkeeping, per-stage reset
    // on each stage's device, drafter re-arm), plus the live-retention arm. `reused` = tokens of this
    // request the slot's sessions already hold (0 = fresh: zero every stage, today's path). The caller
    // has moved the request in and holds the pump fence; the pump only sees the slot after pump_resume.
    auto admit_slot = [&](Impl::Slot& s, int64_t reused, Impl::Slot* parent = nullptr, std::shared_ptr<core::RadixNode> radix_parent = nullptr) -> bool {
        if ((int64_t) s.request.tokens.size() > s.max_context) { err = "concurrency: request prompt (" + std::to_string(s.request.tokens.size()) + ") exceeds slot context (" + std::to_string(s.max_context) + ")"; return false; }
        if (m.lazy_slots && !s.brought_up && !bringup_slot(s, err)) return false;
        s.read.store(reused); s.generated = s.offered = s.accepted = 0;
        s.prompt_ms.store(0); s.first = true; s.in_flight_prefix_hash.store(s.request.tokens.size() >= 64 ? core::compute_token_chunk_hash(s.request.tokens.data(), 64) : 0, std::memory_order_release); s.active.store(true);
        s.position.store((int64_t) s.request.tokens.size() - 1);
        s.current = (int32_t) s.request.tokens.back();
        s.reused_prefix = reused;
        s.prompt_snap_done = false;
        if (retain) s.session = s.request.session;   // D1: keep the slot's conversation key current
        if (reused == 0) s.consumed.clear();
        std::fill(std::begin(s.probability), std::end(s.probability), 0.0f);
        std::fill(std::begin(s.drafts), std::end(s.drafts), 0);
        std::fill(std::begin(s.lookup_tokens), std::end(s.lookup_tokens), 0);
        std::fill(std::begin(s.window), std::end(s.window), 0);
        std::fill(std::begin(s.output), std::end(s.output), 0);
        s.suffix.reset(); s.policy = spec::DraftPolicy{c.window};
        for (auto token : s.request.tokens) s.suffix.append((int32_t) token);

        if (reused == 0) {
            s.saved_prefix = 0;
            s.saved_consumed.clear();
            s.cache_tier = "Fresh";
            if (s.radix_node) {
                radix_tree.release(s.radix_node);
                s.radix_node = nullptr;
            }
            for (size_t st = 0; st < m.stages.size(); ++st) {   // C4: every stage's state resets on its device
                const core::OnDevice on(m.stages[st].device);
                core::session_zero(*s.stages[st].state, g, nullptr, m.stage_rt[st].prompt_stream);
                core::progress_at("concurrency: resetting a stage", (int64_t) st);
                if (cudaStreamSynchronize(m.stage_rt[st].prompt_stream) != cudaSuccess) { err = "concurrency: reset failed"; return false; }
            }
            s.draft->reset();
        } else if (radix_parent != nullptr) {
            // SPRINT 4: Dynamic RadixTree prefix fork
            s.consumed.clear();
            for (int64_t i = 0; i < reused; ++i) s.consumed.push_back((int32_t) s.request.tokens[(size_t) i]);
            std::vector<int> r_devs;
            std::vector<core::SessionState*> r_child_states;
            std::vector<void*> r_streams;
            for (size_t st = 0; st < m.stages.size(); ++st) {
                r_devs.push_back(m.stages[st].device);
                r_child_states.push_back(s.stages[st].state);
                r_streams.push_back((void*) m.stage_rt[st].prompt_stream);
            }
            std::string fork_err;
            if (!radix_tree.fork_to_session(radix_parent, reused, r_devs, r_child_states, g, r_streams, fork_err)) {
                err = "concurrency: radix tree fork failed: " + fork_err;
                return false;
            }
            for (size_t st = 0; st < m.stages.size(); ++st) {
                if (s.stages[st].fork_event) {
                    cudaEventRecord(s.stages[st].fork_event, m.stage_rt[st].prompt_stream);
                    s.stages[st].fork_in_flight = true;
                }
            }
            if (s.draft) s.draft->kv_restore(reused);
            if (s.radix_node && s.radix_node != radix_parent) radix_tree.release(s.radix_node);
            s.radix_node = radix_parent; // Already holds ref_count acquired in match_prefix()
            s.cache_tier = radix_parent->has_device_snapshot() ? "VRAM" : "Host";
            std::fprintf(stderr, "strata-agx concurrent: radix-tree fork: slot forks %lld tokens from RadixNode #%lld (%s)\n", (long long) reused, (long long) radix_parent->id, s.cache_tier.c_str());
        } else if (parent != nullptr) {
            // SPRINT 1: Cross-slot or self-slot prefix fork from snapshot
            s.consumed.assign(parent->saved_consumed.begin(), parent->saved_consumed.begin() + reused);
            // Frontier 2: Non-blocking multi-stage KV fork overlapping
            // Dispatch asynchronous snapshot fork concurrently across each stage's prompt stream
            // and record a completion event per stage instead of host-blocking synchronization.
            for (size_t st = 0; st < m.stages.size(); ++st) {
                const core::OnDevice on(m.stages[st].device);
                std::string fork_err;
                const auto& ps = parent->stages[st];
                if (!core::session_fork(*ps.state, *s.stages[st].state, g, reused,
                                        (void*) m.stage_rt[st].prompt_stream, fork_err,
                                        ps.R_saved, ps.gdn_saved, ps.ple_saved,
                                        ps.ple_prev_saved, ps.ple_token_saved)) {
                    err = "concurrency: snapshot fork failed on stage " + std::to_string(st) + ": " + fork_err;
                    return false;
                }
                if (s.stages[st].fork_event) {
                    cudaEventRecord(s.stages[st].fork_event, m.stage_rt[st].prompt_stream);
                    s.stages[st].fork_in_flight = true;
                }
            }
            if (s.draft && parent->draft) {
                std::string draft_err;
                if (!s.draft->fork_from(*parent->draft, reused, draft_err)) {
                    err = "concurrency: fork draft failed: " + draft_err;
                    return false;
                }
                s.draft->kv_restore(reused);
            }
            s.cache_tier = (parent == &s ? "VRAM-Warm" : "VRAM");
            std::fprintf(stderr, "strata-agx concurrent: snapshot fork: slot forks %lld tokens from %s (saved=%lld)\n",
                         (long long) reused, (parent == &s ? "self" : "parent"), (long long) parent->saved_prefix);
        } else {
            // Intra-slot retention: restore verified GDN/PLE/QSA snapshot state to prevent state corruption across turns
            if (s.saved_prefix >= reused && s.stages[0].gdn_saved) {
                for (size_t st = 0; st < m.stages.size(); ++st) {
                    const core::OnDevice on(m.stages[st].device);
                    std::string fork_err;
                    const auto& ss = s.stages[st];
                    core::session_fork(*ss.state, *ss.state, g, reused,
                                       (void*) m.stage_rt[st].prompt_stream, fork_err,
                                       ss.R_saved, ss.gdn_saved, ss.ple_saved,
                                       ss.ple_prev_saved, ss.ple_token_saved);
                    cudaStreamSynchronize(m.stage_rt[st].prompt_stream);
                }
            }
            if ((int64_t) s.consumed.size() > reused) {
                s.consumed.resize((size_t) reused);
            }
            s.draft->kv_restore(reused);
            s.cache_tier = reused > 0 ? "Intra" : "Fresh";
            std::fprintf(stderr, "strata-agx concurrent: live retention: slot resumes %lld tokens (truncated)\n", (long long) reused);
        }
        s.draft->set_prompt_len((int64_t) s.request.tokens.size());
        s.stages[0].verify.set_sampling(s.request.sampling);
        s.stages[0].prompt.embd_rows = !s.request.row_ptr.empty() ? s.request.row_ptr.data() : nullptr;
        int admit_slot_idx = -1;
        for (size_t si = 0; si < m.slots.size(); ++si) {
            if (m.slots[si].get() == &s) { admit_slot_idx = (int) si; break; }
        }
        if (admit_slot_idx >= 0) {
            continuous_sched.allocate_slot_at(admit_slot_idx, s.request.id, (int64_t) s.request.tokens.size(), s.request.max_new, s.max_context);
        }
        return true;
    };
    // The input side: commands, admission and ONE bounded prompt chunk.  In the overlapped loop it
    // runs at the SAFE POINT (no pass in flight) because the prompt path touches the expert caches
    // and the drafter state and must never race a decode pass.  Returns 1 on a hard error.
    auto service_input = [&](bool& quit) -> int {
        quit = false;
        core::progress().busy.store(!live.empty());
        std::deque<std::string> commands;
        {
            std::unique_lock<std::mutex> lock(input->mutex);
            if (live.empty() && pending.empty()) input->cv.wait(lock, [&] { return !input->lines.empty() || input->eof; });
            commands.swap(input->lines);
            if (input->eof && commands.empty()) quit = true;
            input->cv.notify_all();
        }
        if (!pump_check()) return 1;   // a chunk failed while this loop was in a round
        if (g_sigterm_received.load(std::memory_order_relaxed)) {
            std::fprintf(stderr, "strata-agx concurrent: caught SIGTERM/SIGINT -> initiating clean shutdown\n");
            quit = true;
        }
        for (const auto& line : commands) {
            if (line == "QUIT") { quit = true; break; }
            if (line.rfind("BYIELD ", 0) == 0) {
                // BYIELD <number>: cooperative preemption yield request from scheduler
                continue;
            }
            if (line.rfind("CSTOP ", 0) == 0) {
                // Task 1.2: Thread-Safe Cancellation with pump synchronization
                uint64_t id = 0; std::istringstream(line.substr(6)) >> id;
                if (!pump_fence()) return 1;
                for (auto& s : m.slots) {
                    if (s->active.load() && s->request.id == id) {
                        finish(*s, "cancel");
                    }
                }
                pump_resume();
                for (auto p = pending.begin(); p != pending.end();) {
                    if (p->id == id) { error(id, "cancelled before admission"); live.erase(id); p = pending.erase(p); }
                    else ++p;
                }
                continue;
            }
            Request request;
            std::string reason;
            if (!parse_request(line, c, g, wt.find("output.weight")->ne1, request, reason)) { error(request.id, reason); continue; }
            if (live.count(request.id)) { error(request.id, "duplicate request id"); continue; }
            if (pending.size() >= 32) { error(request.id, "request queue is full"); continue; }
            live.insert(request.id);
            pending.push_back(std::move(request));
        }

        if (quit) return 0;
        core::progress().busy.store(!live.empty());
        // Fence: session_zero and its stream sync run on the stage prompt streams the pump shares.  Only
        // when a request will actually be admitted - with every slot busy there is nothing to reset.
        const bool admitting = !pending.empty() &&
            std::any_of(m.slots.begin(), m.slots.end(), [](const auto& ptr) { return !ptr->active.load(); });
        if (admitting && !pump_fence()) return 1;
        // S26f3: flush deferred radix captures strictly before every admission (the design intent;
        // continuous traffic previously never reached the loop's input-side flush). Safe: the pump
        // fence just ran and the overlap pipeline is drained (pipeline_in_flight false).
        if (admitting && !pipeline_in_flight) {
            flush_pending_snapshots();
            capture_prompt_end_snapshots();
        }
        // P4 live retention (retain): the slot's `consumed` IS the conversation its sessions hold (the
        // invariant every writer keeps - see the Slot comment). A queued request whose prompt starts
        // with a slot's FULL retained history, and is longer than it, continues that slot with no
        // copy: admission skips session_zero and the pump reads only [reused, n-1). The bound and the
        // exact-token rule are the serial loop's starts_with (generate.cpp:4576-4581): at most n-1
        // retained tokens, the last token always opens the first verify window. A partial or divergent
        // prefix is a miss and the walk below zeroes the slot exactly as before.
        // D1 prefix-aware admission: the match pass scans the WHOLE pending deque (bounded by its own
        // cap of 32 entries) against every idle slot and admits the longest exact match FIRST; ties
        // break by arrival order, then by slot index.  The head-only pass this replaces could only
        // ever admit the queue head, so a longer-matching turn behind it was served fresh and the
        // retained history it matched was lost.  The CGEN sess= hint (pending[q].session) makes a
        // request prefer its own slot when that slot is idle and still holds a matching history; a
        // busy or diverged keyed slot falls back to the longest-match scan.  Cost is bounded by the
        // length prefilter (a candidate must beat the current best) and early-exit compares; the
        // added stall is sampled below for the falsifier.
        // Off (default): this pre-pass is dead code and the walk is byte-identical.
        constexpr size_t kAdmissionScanQueue = 32;   // the pending deque's own cap (see the overflow check above; M2 raise 16->32 for burst headroom)
        const auto admit_start = retain ? Clock::now() : Clock::time_point{};
        std::vector<Impl::Slot*> admitted_this_round;
        // Pass 0: Intra-Slot VRAM Warm Snapshot Retention (0ms TTFT, zero PCIe transfer, zero session_zero wipe)
        while (retain && !pending.empty()) {
            Impl::Slot* pick = nullptr;
            size_t pick_q = 0;
            int64_t best = 0;
            const size_t qn = std::min<size_t>(pending.size(), kAdmissionScanQueue);
            for (size_t q = 0; q < qn; ++q) {
                const auto& tokens = pending[q].tokens;
                const int64_t nmax = (int64_t) tokens.size() - 1;
                if (nmax < 64 || nmax <= best) continue;

                for (auto& ptr : m.slots) {
                    auto* p = ptr.get();
                    if (p->active.load() || (int64_t) tokens.size() >= p->max_context) continue;
                    if (std::find(admitted_this_round.begin(), admitted_this_round.end(), p) != admitted_this_round.end()) continue;
                    // Enforce session affinity: intra-slot retention requires matching session
                    if (pending[q].session != 0 && p->session != 0 && p->session != pending[q].session) continue;

                    const int64_t saved_len = p->saved_prefix;
                    if (saved_len < 64 || saved_len > nmax || saved_len <= best) continue;
                    if (!p->stages[0].gdn_saved) continue;

                    // Exact token prefix match against slot's verified VRAM snapshot
                    bool exact_match = true;
                    for (int64_t i = 0; i < saved_len && exact_match; ++i) {
                        exact_match = (p->saved_consumed[(size_t) i] == (int32_t) tokens[(size_t) i]);
                    }
                    if (exact_match && saved_len > best) {
                        best = saved_len;
                        pick = p;
                        pick_q = q;
                    }
                }
            }
            if (pick == nullptr || best == 0) break;

            auto& s = *pick;
            s.request = std::move(pending[pick_q]);
            pending.erase(pending.begin() + (std::ptrdiff_t) pick_q);
            // Admit slot with parent == &s to trigger instant VRAM snapshot restore!
            if (!admit_slot(s, best, &s)) return 1;
            admitted_this_round.push_back(&s);
        }

        // SPRINT 4: Dynamic RadixTree prefix fork pass
        while (retain && !pending.empty()) {
            size_t pick_q = 0;
            int64_t best = 0;
            std::shared_ptr<core::RadixNode> best_node = nullptr;
            const size_t qn = std::min<size_t>(pending.size(), kAdmissionScanQueue);
            for (size_t q = 0; q < qn; ++q) {
                const auto& tokens = pending[q].tokens;
                const int64_t nmax = (int64_t) tokens.size() - 1;
                if (nmax < 64 || nmax <= best) continue;
                auto match = radix_tree.match_prefix(tokens.data(), tokens.size());
                if (std::getenv("STRATA_S26F_LOG"))
                    std::fprintf(stderr, "S26F match: qlen=%zu matched=%lld node=%lld nodecov=%lld\n",
                                 tokens.size(), (long long) match.matched_tokens,
                                 match.node ? (long long) match.node->id : -1LL,
                                 match.node ? (long long) match.node->prefix_len : -1LL);
                // Mandate 4: PQP-TR Partial Prefix Retrieval
                // In match_prefix(), full hits have match.matched_tokens == match.node->prefix_len,
                // while partial edge hits are truncated to 64-token page boundaries (M_safe = (M / 64) * 64).
                // Both are safe to restore from Host RAM, with remainder tokens prefilled as catch-up.
                const bool pqp_match = match.node && (match.matched_tokens == match.node->prefix_len);
                if (pqp_match && match.matched_tokens >= 64 && match.matched_tokens > best && nmax >= match.matched_tokens) {
                    if (best_node && best_node != match.node) {
                        radix_tree.release(best_node);
                    }
                    best = std::min<int64_t>(match.matched_tokens, nmax);
                    best_node = match.node;
                    pick_q = q;
                } else if (match.node) {
                    radix_tree.release(match.node); // Not chosen, release ref_count
                }
            }
            if (best_node == nullptr || best == 0) break;

            const int64_t req_len = (int64_t) pending[pick_q].tokens.size();
            Impl::Slot* pick_child = nullptr;
            if (pick_child == nullptr) {
                for (auto& ptr : m.slots) {
                    if (!ptr->active.load() && req_len < ptr->max_context &&
                        std::find(admitted_this_round.begin(), admitted_this_round.end(), ptr.get()) == admitted_this_round.end()) {
                        pick_child = ptr.get(); break;
                    }
                }
            }
            if (pick_child == nullptr) {
                if (best_node) radix_tree.release(best_node);
                break;
            }

            auto& s = *pick_child;
            s.request = std::move(pending[pick_q]);
            pending.erase(pending.begin() + (std::ptrdiff_t) pick_q);
            if (!admit_slot(s, best, nullptr, best_node)) return 1;
            admitted_this_round.push_back(&s);
        }

        // SPRINT 1: Cross-slot prefix fork pass.
        while (retain && !pending.empty()) {
            Impl::Slot* pick_parent = nullptr;
            size_t pick_q = 0;
            int64_t best = 0;
            const size_t qn = std::min<size_t>(pending.size(), kAdmissionScanQueue);
            for (size_t q = 0; q < qn; ++q) {
                const auto& tokens = pending[q].tokens;
                const int64_t nmax = (int64_t) tokens.size() - 1;
                if (nmax < 1) continue;
                if (nmax <= best) continue;

                for (auto& p_ptr : m.slots) {
                    auto* p = p_ptr.get();
                    const int64_t valid_len = p->saved_prefix;
                    if (valid_len < 64 || valid_len > nmax || valid_len <= best) continue;
                    // Enforce session affinity: cross-slot snapshot reuse only permitted for same session
                    if (pending[q].session != 0 && p->session != 0 && p->session != pending[q].session) continue;
                    bool same = true;
                    for (int64_t i = 0; i < valid_len && same; ++i) {
                        same = (p->saved_consumed[(size_t) i] == (int32_t) tokens[(size_t) i]);
                    }
                    if (same) {
                        best = valid_len;
                        pick_parent = p;
                        pick_q = q;
                    }
                }
            }
            if (pick_parent == nullptr || best == 0) break;

            const int64_t req_len = (int64_t) pending[pick_q].tokens.size();
            Impl::Slot* pick_child = nullptr;
            // Prefer the parent slot itself if it is idle, otherwise pick any idle child slot
            if (pick_parent != nullptr && !pick_parent->active.load() && req_len < pick_parent->max_context &&
                std::find(admitted_this_round.begin(), admitted_this_round.end(), pick_parent) == admitted_this_round.end()) {
                pick_child = pick_parent;
            }
            if (pick_child == nullptr) {
                for (auto& ptr : m.slots) {
                    if (!ptr->active.load() && req_len < ptr->max_context &&
                        std::find(admitted_this_round.begin(), admitted_this_round.end(), ptr.get()) == admitted_this_round.end()) {
                        pick_child = ptr.get(); break;
                    }
                }
            }
            if (pick_child == nullptr) break;

            auto& s = *pick_child;
            s.request = std::move(pending[pick_q]);
            pending.erase(pending.begin() + (std::ptrdiff_t) pick_q);
            if (!admit_slot(s, best, pick_parent)) return 1;
            admitted_this_round.push_back(&s);
        }

        if (retain && admitting) {   // D1 falsifier: the whole-queue scan's added stall at the safe point
            const double ms = elapsed(admit_start);
            ++admit_scans; admit_ms_sum += ms;
            if (ms > admit_ms_max) admit_ms_max = ms;
            admit_ms_ring[(size_t) (admit_scans - 1) % kAdmissionStallSamples] = ms;
        }
        for (size_t q = 0; q < pending.size(); ) {
            const auto& req = pending[q];
            // Phase 2: In-batch simulated queue prefix matching & prefill deduplication (INQ2-01 Thread-Safe Atomic)
            bool prefix_in_flight = false;
            if (req.tokens.size() >= 64) {
                const uint64_t prompt_chunk_hash = core::compute_token_chunk_hash(req.tokens.data(), 64);
                for (const auto& ptr : m.slots) {
                    if (ptr->active.load() && ptr->read.load() < ptr->position.load()) {
                        const uint64_t slot_hash = ptr->in_flight_prefix_hash.load(std::memory_order_acquire);
                        if (slot_hash != 0 && slot_hash == prompt_chunk_hash) {
                            prefix_in_flight = true;
                            break;
                        }
                    }
                }
            }
            // INQ2-03: Only defer if no other slot is currently idle and ready
            bool has_other_idle_slot = false;
            for (const auto& ptr : m.slots) {
                if (!ptr->active.load() && std::find(admitted_this_round.begin(), admitted_this_round.end(), ptr.get()) == admitted_this_round.end()) {
                    has_other_idle_slot = true;
                    break;
                }
            }
            if (prefix_in_flight && !has_other_idle_slot && pending.size() > 1 && q + 1 < pending.size()) {
                // Defer duplicate prefill: another slot is prefilling this exact prefix right now!
                // Sibling request will achieve 100% warm hit once current prefill commits.
                ++q;
                continue;
            }
            const int64_t req_len = (int64_t) req.tokens.size();
            Impl::Slot* pick = nullptr;

            // 0. Prefer idle slot matching request session
            if (req.session != 0) {
                for (auto& ptr : m.slots) {
                    if (!ptr->active.load() && req_len < ptr->max_context && ptr->session == req.session &&
                        std::find(admitted_this_round.begin(), admitted_this_round.end(), ptr.get()) == admitted_this_round.end()) {
                        pick = ptr.get();
                        break;
                    }
                }
            }
            // 1. Prefer an empty slot (consumed.empty()) to preserve cached slots for other conversations
            if (pick == nullptr) {
                for (auto& ptr : m.slots) {
                    if (!ptr->active.load() && req_len < ptr->max_context && ptr->consumed.empty() &&
                        std::find(admitted_this_round.begin(), admitted_this_round.end(), ptr.get()) == admitted_this_round.end()) {
                        pick = ptr.get();
                        break;
                    }
                }
            }
            // 2. If no empty slots, pick round-robin / rotation to avoid constantly thrashing Slot 0
            if (pick == nullptr) {
                const size_t rot_snap = m.prompt_rotation.load(std::memory_order_relaxed);
                for (size_t si = 0; si < m.slots.size(); ++si) {
                    auto& ptr = m.slots[(rot_snap + si) % m.slots.size()];
                    if (!ptr->active.load() && req_len < ptr->max_context &&
                        std::find(admitted_this_round.begin(), admitted_this_round.end(), ptr.get()) == admitted_this_round.end()) {
                        pick = ptr.get();
                        m.prompt_rotation.store((rot_snap + si + 1) % m.slots.size(), std::memory_order_relaxed);
                        break;
                    }
                }
            }
            if (pick != nullptr) {
                auto& s = *pick;
                s.request = std::move(pending[q]);
                pending.erase(pending.begin() + (std::ptrdiff_t) q);
                if (!admit_slot(s, 0, nullptr)) return 1;
                admitted_this_round.push_back(&s);
            } else {
                ++q;
            }
        }

        pump_resume();   // new prefillable slots are announced here (a no-op when the pump is off)
        core::progress().busy.store(!live.empty());
        if (!pump.joinable()) {
            // A/B arm (STRATA_PREFILL_PUMP=0): the pre-change inline path - at most ONE bounded prompt chunk
            // before returning to ready decoders, on this thread.
            const size_t rot_fallback = m.prompt_rotation.load(std::memory_order_relaxed);
            for (size_t j = 0; j < m.slots.size(); ++j) {
                const size_t i = (rot_fallback + j) % m.slots.size();
                const auto& s = *m.slots[i];
                if (!s.active.load() || s.read.load() >= s.position.load()) continue;
                m.prompt_rotation.store((i + 1) % m.slots.size(), std::memory_order_relaxed);
                if (!run_chunk(i, err)) return 1;
                break;
            }
        }

        return 0;
    };
    bool quitting = false;
    while (!quitting) {
        bool quit = false, input_done = false;
        capture_prompt_end_snapshots();   // S26f7: per-round L1 scan (safe insert; evictions deferred to flush points)
        core::progress().busy.store(!live.empty());
        // LANE overlap-width (merged drain): when the in-flight unit owns every slot that could
        // launch, retire it HERE and fall through - the successor unit is built and launched in
        // this same iteration (one round per unit; see drain_prev_unit).  A partial ready set
        // still takes the pipelined path below, exactly as before.
        if (overlap && prev.valid) {
            bool launchable = false;
            for (size_t j = 0; j < m.slots.size() && !launchable; ++j)
                launchable = slot_launchable(*m.slots[(rotation + j) % m.slots.size()]);
            if (!launchable && drain_prev_unit()) return 1;
        }
        pipeline_in_flight = (overlap && prev.valid);   // S26f3
        if (!(overlap && prev.valid)) {   // nothing in flight: the input side runs now (the original position)
            flush_pending_snapshots();
            capture_prompt_end_snapshots();
            if (service_input(quit)) return 1;
            input_done = true;
            if (quit) break;
        }
        // Micro-batch partitioning for stage overlap:
        // When overlap is enabled and nothing is currently in flight (!prev.valid),
        // partition the eligible slots into two micro-batches so that Unit A and Unit B
        // can ping-pong across Stage 0 (GPU 1) and Stage 1 (GPU 0) concurrently.
        // Task 8.1: Overlapped Nanobatch Partitioning (K in [2..4])
        size_t max_unit_slots = m.slots.size();
        static bool alt_unit = false;
        const int nanobatch_k = (c.nanobatch >= 2 && c.nanobatch <= 4) ? c.nanobatch : 2;
        core::NanobatchScheduler nanobatch_sched(nanobatch_k);
        if (overlap && !prev.valid && !c.unit_all) {
            size_t eligible = 0;
            for (size_t j = 0; j < m.slots.size(); ++j) {
                const auto& s = *m.slots[j];
                if (s.active.load() && s.read.load(std::memory_order_acquire) >= s.position.load() &&
                    s.position.load() < s.max_context && s.generated < s.request.max_new)
                    ++eligible;
            }
            if (eligible >= (size_t) nanobatch_k) {
                max_unit_slots = (eligible + nanobatch_k - 1) / nanobatch_k;
            } else if (eligible >= 2) {
                if (eligible % 2 != 0) {
                    max_unit_slots = alt_unit ? (eligible / 2) : ((eligible + 1) / 2);
                    alt_unit = !alt_unit;
                } else {
                    max_unit_slots = eligible / 2;
                }
            }
        }

        // S27e: optionally hold the boundary briefly for a straggler slot, so the unit can pack two
        // arrivals (the phase-locked rotation staggers slots ~1 round apart).  Default 0 = off.
        if (c.unit_wait_ms > 0 && overlap && !prev.valid) {
            // Only worth waiting when another slot is genuinely mid-flight and non-terminal;
            // terminal slots (max_new reached, context full) or lone agents must never stall.
            size_t inflight = 0;
            for (size_t j = 0; j < m.slots.size(); ++j) {
                const auto& s = *m.slots[j];
                if (s.active.load() && s.read.load(std::memory_order_acquire) < s.position.load() &&
                    s.position.load() < s.max_context && s.generated < s.request.max_new) {
                    if (s.generated + 1 < s.request.max_new) ++inflight;
                }
            }
            if (inflight > 0) {
                const auto t_wait = Clock::now();
                for (;;) {
                    size_t launchable_now = 0;
                    for (size_t j = 0; j < m.slots.size(); ++j)
                        if (slot_launchable(*m.slots[j])) ++launchable_now;
                    if (launchable_now >= 2) break;
                    if (std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t_wait).count() >= c.unit_wait_ms) break;
                    std::this_thread::sleep_for(std::chrono::microseconds(500));
                }
            }
        }
        std::vector<Impl::Slot*> ready;
        std::vector<int> wanted;
        for (size_t j = 0; j < m.slots.size(); ++j) {
            if (ready.size() >= max_unit_slots) break;
            auto& s = *m.slots[(rotation + j) % m.slots.size()];
            // Acquire: the pump's release store of `read` publishes its consumed rows and session writes.
            if (!s.active.load() || s.read.load(std::memory_order_acquire) < s.position.load()) continue;
            // lane overlap: a slot whose window is still in flight in the previous unit is NOT ready -
            // its next window's tokens are that unit's epilogue's drafts.  Including it would replay
            // the in-flight window (a double-generated token).
            if (overlap && prev.valid &&
                std::find(prev.ready.begin(), prev.ready.end(), &s) != prev.ready.end()) continue;
            if (s.position.load() >= s.max_context) { finish(s, "length"); continue; }
            int n = s.first ? 1 : c.mtp_window_rows;
            if (!s.first && s.request.spec_min_p > 0) {
                const float p0 = s.probability[0];
                const int dyn_depth = s.policy.decide_depth(p0, c.mtp_window_rows, s.request.spec_min_p);
                n = 1;
                float thresh = s.request.spec_min_p;
                while (n < dyn_depth && s.probability[n - 1] >= thresh) {
                    ++n;
                    thresh *= 0.85f; // Compound autoregressive entropy decay
                }
            }
            s.lookup = false; s.match = 0;
            if (!s.first && c.suffix > 0) {
                const int k = s.suffix.propose(c.window - 1, s.lookup_tokens);
                s.match = s.suffix.last_match();
                if (k > 0 && s.lookup_tokens[0] == s.drafts[0]) {
                    const auto pick = s.policy.choose(n, k, s.match);
                    if (pick.lookup) { n = pick.t; s.lookup = true; }
                }
            }
            n = (int) std::min<int64_t>(n, std::min(s.max_context - s.position.load(), s.request.max_new - s.generated));
            if (n < 1) { finish(s, "length"); continue; }
            ready.push_back(&s); wanted.push_back(n);
        }
        const auto allocation = schedule_rows(wanted, c.rows, c.depth);
        std::vector<core::Verifier::BatchWindow> windows;
        std::vector<Impl::Slot*> win_slots;   // C4: parallel to `windows`; a window's chain head is its slot's stage 0
        for (size_t i = 0; i < ready.size(); ++i) {
            auto& s = *ready[i]; s.count = allocation[i];
            if (!s.count) continue;
            if (s.first) s.decode_start = Clock::now();
            s.window[0] = s.current;
            for (int t = 1; t < s.count; ++t) s.window[t] = s.lookup ? s.lookup_tokens[t - 1] : s.drafts[t - 1];
            // Outputs beyond the real window are never emitted, accepted or committed.
            // Fixed shapes can amortize graph construction without asking MTP for more drafts.
            for (int t = s.count; t < c.window; ++t) s.window[t] = s.current;
            const int history = s.request.sampling.penalty_last_n;
            if (history > 0) {
                kernels::penalty_rows(s.consumed.data(), (int64_t) s.consumed.size(), s.window, s.count, history, s.history.data());
                if (cudaMemcpy(s.history_device, s.history.data(), (size_t) s.count * history * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
                    err = "concurrency: history upload failed"; return 1;
                }
            }
            s.stages[0].verify.set_history(history ? s.history_device : nullptr, history);
            windows.push_back({&s.stages[0].verify, s.count, s.window, s.position.load(), s.output});
            win_slots.push_back(&s);
        }

        if (!windows.empty()) {
            ++batch_sizes[windows.size()];
            windows_served += (int64_t) windows.size();
            if (c.pad_batch && windows.size() > 1) {
                int padded_rows = 0;
                for (size_t wi = 0; wi < windows.size(); ++wi)
                    padded_rows += (int) std::min<int64_t>(std::max(windows[wi].count, c.mtp_window_rows), win_slots[wi]->max_context - windows[wi].position);
                if (padded_rows <= c.rows)
                    for (size_t wi = 0; wi < windows.size(); ++wi)
                        windows[wi].count = (int) std::min<int64_t>(std::max(windows[wi].count, c.mtp_window_rows), win_slots[wi]->max_context - windows[wi].position);
            }
            // Stable packing order avoids recapturing a graph merely because fairness rotated the request order.
            // Synchronize win_slots and ready with windows sorting order!
            struct WinEntry {
                core::Verifier::BatchWindow win;
                Impl::Slot* slot;
            };
            std::vector<WinEntry> entries(windows.size());
            for (size_t wi = 0; wi < windows.size(); ++wi) {
                entries[wi] = {windows[wi], win_slots[wi]};
            }
            std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
                return std::less<core::Verifier*>{}(a.win.verifier, b.win.verifier);
            });
            for (size_t wi = 0; wi < windows.size(); ++wi) {
                windows[wi] = entries[wi].win;
                win_slots[wi] = entries[wi].slot;
            }
            // Sync ready vector order to match sorted win_slots
            ready = win_slots;
            const auto start = Clock::now();
            dispatch.failed = false;
            // C3/C4 (design-wiring §5): refresh EVERY stage's publish target for THIS round before it runs.
            // A stale sink is not a hang: the empty-plan fallback silently drops the resident experts'
            // contribution (verify.cpp:1073-1081), and round parity is the only detector.  A single-window
            // round publishes through the window's slot chain, a batch round through its stage coordinators.
            for (size_t st = 0; st < m.stages.size(); ++st) {
                if (!stage_plans) continue;
                // lane overlap: stage >= 1 belongs to the stage-1 pass STILL IN FLIGHT from the
                // previous unit; it was refreshed when that pass was launched.  Refreshing it here
                // would point the in-flight pass's pool at the wrong sink (empty-plan fallback ->
                // silently wrong tokens).
                if (overlap && st > 0) continue;
                stage_plans[st] = batch[st].plan_sink();
            }
            dispatch.plan = batch[0].plan_sink();
            // N1 (C2 acceptance kit, env-gated, default off): the stale-sink negative control -
            // see n1_poison_plan_sinks.  The poison consumes exactly what the refresh above set,
            // so it hits single-window AND batch rounds; the launch-site call below re-applies it
            // where the overlap path points the stage>=1 targets.
            n1_poison_plan_sinks();
            bool ok = true;

            if (!overlap) {
                Unit u;   // the serial path: one unit, run to completion (the original loop's shape)
                u.single = false;
                u.windows = windows;
                u.ready = ready;
                u.t_start = start;
                // Unified: batch coordinator executes all rounds (single and multi-window) with unified
                // graph caching, device planning, and multi-stage chaining.
                ok = batch[0].run_batch(windows, pool, user, err);
            if (!ok || dispatch.failed) {
                if (dispatch.failed) err = dispatch.fail ? dispatch.fail : "expert dispatch failed";
                // K1b: a refused batch capture (free VRAM below the reserve) fails THIS round, not
                // the engine.  The Verifier destroys the refused graph and caches nothing, so the
                // next round recaptures cleanly; error the round's requests and stay up for it.
                // Any other failure (wiring, dispatch, pool) keeps fail-stop semantics.
                // LANE failclean: a refused capture keeps the engine up on the serial path AND on
                // both stage-overlap launch sites below - every path a refusal can surface on fails
                // THIS round only; any other failure keeps fail-stop semantics.
                if (!dispatch.failed && err.find("below VRAM reserve") != std::string::npos) {
                    refuse_round(ready, err);
                    rotation = (rotation + 1) % m.slots.size();
                    continue;
                }
                return 1;
            }

            target_ms += elapsed(start);
            for (const auto& w : windows) target_rows += w.count;

                if (!retire_unit(u)) return 1;
            ++rounds;
            if (adaptive && rounds % c.adapt_every == 0) {
                // Fence: a swap rewrites the residency table and the cache slots the pump reads for chunks.
                if (!pump_fence()) return 1;
                const auto adapt_start = Clock::now();
                const bool adapted = adapt();
                adapt_ms += elapsed(adapt_start);
                pump_resume();
                if (!adapted) return 1;
            }

                if (rounds % 64 == 0) {
                    report_profile();
                    auto cstats = radix_compactor.compact(&radix_tree);
                    if (cstats.dead_leaves_pruned > 0 || cstats.chains_merged > 0) {
                        std::fprintf(stderr, "strata-agx concurrent: RadixCompactor defragmented KV tree (pruned %zu leaves, merged %zu chains, reclaimed %zu bytes in %.1f us)\n",
                                     cstats.dead_leaves_pruned, cstats.chains_merged, cstats.bytes_reclaimed, cstats.compaction_time_us);
                    }
                }
            } else {
                // Unit U_i: launch its stage-0 pass on this unit's parity, drive it together with the
                // PREVIOUS unit's stage-1 pass (which still reads the other parity), then close both
                // and run the previous unit's epilogue.  U_i's stage-1 pass is launched at the safe
                // point below and runs during the NEXT iteration - that is the overlap.
                Unit u;
                u.single = false;
                u.parity = unit_parity;
                u.ready = ready;
                u.t_start = Clock::now();
                u.windows = windows;
                u.chain.reserve(windows.size());
                for (const auto& w : windows)
                    u.chain.push_back({w.verifier->next(), w.count, w.tokens, w.position, w.output});
                u.lane0 = unit_lane(u, 0);
                u.lane1 = unit_lane(u, 1);
                unit_parity_set(u, u.parity);
                // The stage-1 pass of the unit TWO back read this parity's buffers; its completion
                // event gates the overwrite (the ping-pong's WAR edge).
                if (ev1_valid[u.parity] &&
                    cudaStreamWaitEvent(u.lane0->stream(), ev1[u.parity], 0) != cudaSuccess) {
                    err = "concurrency: the stage-overlap buffer gate failed";
                    return 1;
                }
                ok = u.lane0->begin_pass_batch(u.windows, pool, user, ev0[u.parity], err);
                if (!ok) {
                    // LANE failclean: a capture refused for want of the VRAM reserve fails THIS round;
                    // any other failure keeps fail-stop.  The refused unit launched nothing, so the
                    // PREVIOUS unit (still in flight) is finished here first - its requests are
                    // healthy and must not be starved by a refusal streak - and u consumed no parity:
                    // unit_parity is not advanced, so the next round simply retries (and captures
                    // again once the reserve allows).
                    if (err.find("below VRAM reserve") == std::string::npos) return 1;
                    if (!core::Verifier::drive_passes(nullptr, prev.valid ? prev.lane1 : nullptr, err)) return 1;
                    if (prev.valid) {
                        if (prev.single) { if (!prev.lane1->end_pass_window(prev.T, prev.out, err)) return 1; }
                        else { if (!prev.lane1->end_pass_batch(prev.chain, err)) return 1; }
                        target_ms += elapsed(prev.t_start);
                        target_rows += prev.single ? prev.T : [&] { int64_t r = 0; for (const auto& w : prev.windows) r += w.count; return r; }();
                        if (!retire_unit(prev)) return 1;
                        prev.valid = false;
                    }
                    refuse_round(u.ready, err);
                    rotation = (rotation + 1) % m.slots.size();
                    continue;
                }
                if (!core::Verifier::drive_passes(u.lane0, prev.valid ? prev.lane1 : nullptr, err)) return 1;
                if (u.single) { if (!u.lane0->end_pass_window(u.T, nullptr, err)) return 1; }
                else { if (!u.lane0->end_pass_batch(u.windows, err)) return 1; }
                // Stream Synchronization / Epilogue Decoupling: retire the previous unit HERE
                // BEFORE launching the next stage-1 pass. Running retire_unit while stage-1 is in
                // flight causes cudaStreamSynchronize(cs_) on CUDA 1 to hang while lane0 on CUDA 0
                // waits on m_flagA_ (the #31 class: host in CUDA call while GPU spins on flag).
                if (prev.valid) {
                    if (prev.single) { if (!prev.lane1->end_pass_window(prev.T, prev.out, err)) return 1; }
                    else { if (!prev.lane1->end_pass_batch(prev.chain, err)) return 1; }
                    target_ms += elapsed(prev.t_start);
                    target_rows += prev.single ? prev.T : [&] { int64_t r = 0; for (const auto& w : prev.windows) r += w.count; return r; }();
                    if (!retire_unit(prev)) return 1;
                    prev.valid = false;
                    pipeline_in_flight = false;   // S26f4: a unit just retired; safe window until the next launch
                    // S26f6: the inter-unit moment is the only safe point that sees a slot whose prefill
                    // just completed (state@position, before its first window opens) - capture the
                    // prompt-end snapshot here; admissions alone never hit that one-round window.
                    capture_prompt_end_snapshots();
                }
                if (dispatch.failed) {
                    err = dispatch.fail ? dispatch.fail : "expert dispatch failed";
                    return 1;
                }
                if (!input_done) { flush_pending_snapshots(); capture_prompt_end_snapshots(); if (service_input(quit)) return 1; input_done = true; }
                if (!quit) {
                    // Launch U_i's stage-1 pass, gated on ITS stage-0 completion event ONLY.  The
                    // stage>=1 plan sinks are refreshed here: the pool routes by layer through
                    // split_drive.plan, so the two units in flight publish to their own stage's sink.
                    if (cudaStreamWaitEvent(u.lane1->stream(), ev0[u.parity], 0) != cudaSuccess) {
                        err = "concurrency: the stage-overlap gate failed";
                        return 1;
                    }
                    unit_parity_set(u, u.parity);
                    pipeline_in_flight = true;   // S26f4: a unit is in flight again
                    for (size_t st = 1; st < m.stages.size(); ++st)
                        if (stage_plans)
                            stage_plans[st] = u.single ? u.slot->stages[st].verify.plan_sink() : batch[st].plan_sink();
                    // N1: re-apply the stale-sink poison AFTER this refresh.  This is the site that
                    // owns the just-launched stage-1 pass's sink, and the pass is not serviced until a
                    // later round reads split_drive.plan[st] - without this second application the
                    // top-of-round poison is overwritten here before any stage-1 dispatch sees it.
                    n1_poison_plan_sinks();
                    ok = u.lane1->begin_pass_batch(u.chain, pool, user, ev1[u.parity], err);
                    if (!ok) {
                        // LANE failclean: the same refusal rule as the stage-0 launch above.  This
                        // unit's stage 0 completed (end_pass_batch above) without a commit, and the
                        // previous unit was already retired, so nothing is in flight; unit_parity is
                        // not advanced and the next round retries.
                        if (err.find("below VRAM reserve") == std::string::npos) return 1;
                        refuse_round(u.ready, err);
                        rotation = (rotation + 1) % m.slots.size();
                        continue;
                    }
                    ev1_valid[u.parity] = true;
                    prev = std::move(u);
                    prev.valid = true;
                } else {
                    prev.valid = false;
                }
                unit_parity = (unit_parity + 1) % 3;
                ++rounds;
                core::progress_beat();
                if (adaptive && rounds % c.adapt_every == 0) {
                    // Fence: a swap rewrites the residency table and the cache slots the pump reads for chunks.
                    if (!pump_fence()) return 1;
                    const auto adapt_start = Clock::now();
                    const bool adapted = adapt();
                    adapt_ms += elapsed(adapt_start);
                    pump_resume();
                    if (!adapted) return 1;
                }
                if (rounds % 64 == 0) report_profile();
            }
        } else if (overlap && prev.valid) {
            // Nothing to launch: finish the previous unit (its stage-1 pass is the only thing in flight).
            if (drain_prev_unit()) return 1;
            if (dispatch.failed) {
                err = dispatch.fail ? dispatch.fail : "expert dispatch failed";
                return 1;
            }
            if (!input_done) { flush_pending_snapshots(); capture_prompt_end_snapshots(); if (service_input(quit)) return 1; input_done = true; }
            ++rounds;
            // CRIT-11: Progress-bound heartbeat: only tick progress_beat when sequence positions advance
            // or tokens are actively generated, preventing watchdog blinding during infinite rejection loops.
            static int64_t last_total_toks = 0;
            int64_t curr_total_toks = 0;
            for (const auto& s_ptr : m.slots) {
                if (s_ptr->active.load()) curr_total_toks += s_ptr->generated + s_ptr->position.load();
            }
            if (curr_total_toks != last_total_toks || pending.empty()) {
                core::progress_beat();
                last_total_toks = curr_total_toks;
            }
            if (adaptive && rounds % c.adapt_every == 0) {
                // Fence: a swap rewrites the residency table and the cache slots the pump reads for chunks.
                if (!pump_fence()) return 1;
                const auto adapt_start = Clock::now();
                const bool adapted = adapt();
                adapt_ms += elapsed(adapt_start);
                pump_resume();
                if (!adapted) return 1;
            }
            if (rounds % 64 == 0) report_profile();
        }
        if (pump.joinable() && windows.empty() && !prev.valid && !quit) {
            // Nothing to parse and no ready window: the pump is the only producer.  Wait for input (or a
            // 2 ms tick) instead of spinning; the next round starts as soon as a chunk completes a prompt.
            std::unique_lock<std::mutex> lock(input->mutex);
            // LANE sched-impl P1: also wake when queued requests wait AND a slot is free, so the
            // finish()-notify above shortens the STOP-to-readmit park. The free-slot conjunct keeps
            // the old 2 ms tick whenever everything queued is unadmittable (no busy spin while
            // prefill-bound with a full house).
            input->cv.wait_for(lock, std::chrono::milliseconds(2), [&] {
                return !input->lines.empty() || input->eof ||
                       (!pending.empty() && std::any_of(m.slots.begin(), m.slots.end(),
                                                        [](const auto& ptr) { return !ptr->active.load(); }));
            });
        }
        if (quit) break;
        rotation = (rotation + 1) % m.slots.size();
    }
    if (pump.joinable()) { pump.request_stop(); m.pump_cv.notify_all(); pump.join(); }   // the stop_callback also wakes it
    if (!chain_drain_all()) return 1;   // lane pipeline-prefill: deferred stage-1 runs finish before teardown
    if (!pump_check()) return 1;
    for (auto& s : m.slots) if (s->active.load()) finish(*s, "cancel");
    for (const auto& r : pending) error(r.id, "server shutting down");
    if (nvme_dir_env && *nvme_dir_env) {
        radix_tree.drain_nvme_staging();
        radix_tree.persist_manifest();
        std::fprintf(stderr, "strata-agx concurrent: persistent L3 RadixTree manifest saved to NVMe (%zu nodes)\n",
                     radix_tree.total_nodes());
    }
    report_profile();
    std::fprintf(stderr, "strata-agx concurrent: target rounds by active batch size: 1=%lld 2=%lld 3=%lld 4=%lld 5=%lld 6=%lld 7=%lld 8=%lld 9=%lld 10=%lld 11=%lld 12=%lld 13=%lld 14=%lld 15=%lld 16=%lld\n",
                 (long long) batch_sizes[1], (long long) batch_sizes[2], (long long) batch_sizes[3], (long long) batch_sizes[4],
                 (long long) batch_sizes[5], (long long) batch_sizes[6], (long long) batch_sizes[7], (long long) batch_sizes[8],
                 (long long) batch_sizes[9], (long long) batch_sizes[10], (long long) batch_sizes[11], (long long) batch_sizes[12],
                 (long long) batch_sizes[13], (long long) batch_sizes[14], (long long) batch_sizes[15], (long long) batch_sizes[16]);
    // lane-spec: the acceptance census at exit - windows served, drafts offered vs accepted (offered = the
    // sum over windows of count-1; accepted likewise of keep-1 - the same fields the DONE line carries per
    // request), split by the window's source and read off by draft position.
    std::fprintf(stderr, "strata-agx concurrent: MoE dispatch census: single-CTA=%llu multi-CTA=%llu\n", (unsigned long long) prefill::gpu_group::get_moe_dispatch_stats().single_cta_launches, (unsigned long long) prefill::gpu_group::get_moe_dispatch_stats().multi_cta_launches);
    std::fprintf(stderr, "strata-agx concurrent: spec census: windows=%lld offered=%lld accepted=%lld "
                 "mtp[windows=%lld offered=%lld accepted=%lld] lookup[windows=%lld offered=%lld accepted=%lld]\n",
                 (long long) windows_served, (long long) offered_total, (long long) accepted_total,
                 (long long) source_windows[0], (long long) source_offered[0], (long long) source_accepted[0],
                 (long long) source_windows[1], (long long) source_offered[1], (long long) source_accepted[1]);
    std::fprintf(stderr, "strata-agx concurrent: spec windows by verified count: 1=%lld 2=%lld 3=%lld 4=%lld 5=%lld 6=%lld 7=%lld 8=%lld\n",
                 (long long) count_hist[1], (long long) count_hist[2], (long long) count_hist[3], (long long) count_hist[4],
                 (long long) count_hist[5], (long long) count_hist[6], (long long) count_hist[7], (long long) count_hist[8]);
    std::fprintf(stderr, "strata-agx concurrent: spec windows by committed tokens: 1=%lld 2=%lld 3=%lld 4=%lld 5=%lld 6=%lld 7=%lld 8=%lld\n",
                 (long long) keep_hist[1], (long long) keep_hist[2], (long long) keep_hist[3], (long long) keep_hist[4],
                 (long long) keep_hist[5], (long long) keep_hist[6], (long long) keep_hist[7], (long long) keep_hist[8]);
    std::fprintf(stderr, "strata-agx concurrent: spec offered by position: 1=%lld 2=%lld 3=%lld 4=%lld 5=%lld 6=%lld 7=%lld 8=%lld\n",
                 (long long) pos_offered[1], (long long) pos_offered[2], (long long) pos_offered[3], (long long) pos_offered[4],
                 (long long) pos_offered[5], (long long) pos_offered[6], (long long) pos_offered[7], (long long) pos_offered[8]);
    std::fprintf(stderr, "strata-agx concurrent: spec accepted by position: 1=%lld 2=%lld 3=%lld 4=%lld 5=%lld 6=%lld 7=%lld 8=%lld\n",
                 (long long) pos_accepted[1], (long long) pos_accepted[2], (long long) pos_accepted[3], (long long) pos_accepted[4],
                 (long long) pos_accepted[5], (long long) pos_accepted[6], (long long) pos_accepted[7], (long long) pos_accepted[8]);
    return 0;
}
}
