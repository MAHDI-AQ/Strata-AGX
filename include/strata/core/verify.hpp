// include/strata/core/verify.hpp - plan v0.3 P6: the speculative VERIFY window.
//
// T tokens at consecutive positions p0 .. p0+T-1 - the last accepted token and T-1 drafts - go through all 48
// layers in ONE captured graph, and the head's argmax is produced for every one of them.  Token t's argmax is
// what plain greedy decode would produce after token t, BIT FOR BIT: every kernel here is either the single-token
// kernel applied per token, or a multi-token kernel whose per-token arithmetic is the single-token kernel's
// (multi-column MMVQ in exact mode, the T-token GDN kernels, the per-token hit activation, the multi-token CPU
// expert rows).  So a draft is accepted exactly when greedy decode would have produced it.
//
// What the window costs is the dense weights read ONCE for T tokens and the union of the T tokens' missed
// experts on the CPU (measured on decode traces: 1.75x one token's misses for T=2, 2.4x for 3, 3.05x for 4).
//
// STATE.  The window appends K/V and indexer keys for all T positions and leaves the GDN state untouched.
// `commit(n_keep)` then makes the first `n_keep` tokens permanent: the GDN conv history and recurrent state are
// advanced by replaying those tokens from inputs the window stored, the indexer's key tail is restored from a
// snapshot and the accepted keys re-appended (a rejected key can land in a slot the current block still needs),
// and the PLE history is set to its snapshot after token n_keep-1.  K/V cells past the accepted prefix are simply
// overwritten when those positions are processed again, before any query can read them.
//
// Requires the default native decode configuration (native projections, fused GR, fused GDN, fast attention and
// selection, native indexer) and a profile-filled VRAM expert tier with its residency table on the device.
#pragma once

#include <atomic>
#include <cstdio>

#include "strata/core/expert_source.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/kernels/sampler.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

class NativeHead;

/// The CPU pool for a window: x_f (n_tok, n_embd), ids (n_tok, k) -> out (n_tok * k, n_embd), hit rows zeroed.
using PoolMultiFn = void (*)(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                             int64_t layer);

struct VerifyHits {
    const int32_t* d_res = nullptr;      ///< device [n_layers * n_expert] slot or -1
    const uint8_t* cache_base = nullptr; ///< slot 0 of the VRAM expert arena
    const uint64_t* slot_off = nullptr;   ///< E-6: host per-slot offsets when slots differ in size (null: slot * blob)
    int64_t n_slots = 0;                  ///< E-6: how many (for the device copy)
    int64_t blob = 0;
};

class Verifier {
public:
    Verifier() = default;
    ~Verifier();
    Verifier(const Verifier&) = delete;
    Verifier& operator=(const Verifier&) = delete;

    /// The watchdog's view of the window in flight (issue #31): the layer, the GPU's sequence, the flags.
    void diag(std::FILE* f) const;

    /// #267: raise every flag the window's spin kernels wait on past any ring (UINT32_MAX), so a window the GPU
    /// cannot finish drains instead of staying resident, then wait up to `timeout_ms` for its streams.  For the
    /// paths that give up on the engine (a timed-out window, the serve watchdog): the window then ran on whatever
    /// the flags guarded, so this verifier refuses every later window.  True when the streams finished.
    bool release_gpu_waits(int timeout_ms);

    /// `max_t` <= kVerifyMaxT; a dedicated expert-only batch workspace may reserve up to MAXT rows.
    /// `head` may be null (the canonical head is then run per token).
    bool init(const WeightTable& wt, const ModelGeometry& g, SessionState& ss, const VerifyHits& hits,
              const NativeHead* head, int max_t, std::string& err, bool batch_workspace = false);

    /// One window: `tokens[0..T)` at positions pos0.., the pool served per layer; `out[t]` = argmax after token t.
    /// The PLE rows are gathered here from `ss.ple_prev` and the tokens.  Captures the T-token graph on first use.
    bool run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, void* user, int32_t* out, std::string& err);
    struct BatchWindow {
        Verifier* verifier;
        int count;
        const int32_t* tokens;
        int64_t position;
        int32_t* output;
    };
    // This instance is a batch coordinator for ONE stage [lb_, le_) of the layer split: every member
    // runs this stage's layers on this device, with THIS stage's range and shared hand-off buffers.
    // The stage chain must be CONTIGUOUS (next_->lb_ == le_); a chained stage writes the packed rows
    // to the hand-off (phase 5) and recurses into next_->run_batch with the members' own next stages;
    // a full-range coordinator keeps today's path (no chain, head per member, phase 4).
    // Row packing: member i's rows start at row0 = sum of counts[j], j < i — identical rows land at
    // identical offsets in every stage by construction. Attention, recurrence and commit stay per member.
    bool run_batch(const std::vector<BatchWindow>& batch, PoolMultiFn pool, void* user, std::string& err);

    // A PASS is one stage's graph execution for one round unit: begin_pass_* stages the inputs,
    // captures/replays this stage's graph for (shape, parity), launches it on cs_ and records `done`
    // after the launch (it fires when the graph, hand-off copies included, has completed).  NO host
    // loop runs inside begin_pass_*: pass_step() services one doorbell at a time, so TWO passes on
    // two stages can be driven interleaved by drive_passes() - stage0[N+1] on CUDA0 while stage1[N]
    // on CUDA1 still reads the OTHER parity's hand-off.  end_pass_* closes a pass (syncs, counters,
    // prof; a last stage also samples and copies its outputs).  The serial run()/run_batch() paths
    // are unchanged and do not use this API.
    bool begin_pass_window(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, void* user,
                           cudaEvent_t done, std::string& err);
    bool begin_pass_batch(const std::vector<BatchWindow>& batch, PoolMultiFn pool, void* user,
                          cudaEvent_t done, std::string& err);
    /// Poll one doorbell of a begun pass: 1 = one layer/group serviced, 0 = not ready yet (the
    /// stall check and the 20 s bound run inside), -1 = the pass failed (err set).
    int pass_step(std::string& err);
    bool pass_finished() const { return !pass_live_ || pass_k_ >= pass_total_; }
    bool end_pass_window(int T, int32_t* out, std::string& err);
    bool end_pass_batch(const std::vector<BatchWindow>& batch, std::string& err);
    /// Service two begun passes' doorbells, interleaved, until both are finished.  Either may be null.
    static bool drive_passes(Verifier* a, Verifier* b, std::string& err);
    /// The stream a pass runs on (the overlap driver gates cross-stream event waits on it).
    cudaStream_t stream() const { return cs_; }
    // Configure before init; CLI validation supplies a positive bounded cache limit.
    void set_batch_cache(int limit, int reserve_mib) { batch_cache_limit_ = limit; batch_reserve_mib_ = reserve_mib; }
    void set_batch_parallel(bool enabled) { batch_parallel_ = enabled; }
    /// Diagnostics: row `t` of the last window's head logits (n_vocab floats) to the host. Valid after run().
    bool copy_logits(int t, float* host) const;
    /// Target draft probability for draft token t in the last window (Leviathan residual sampling). Valid after run().
    float target_draft_prob(int t) const;
    int64_t vocab() const { return next_ ? next_->vocab() : n_vocab_; }
    /// The sampling the verify window's head applies (temperature / top_p / top_k / seed).  Set per
    /// request; greedy by default.  The sampling itself runs OUTSIDE the captured graph - its
    /// parameters would otherwise be baked forever - so this can change between requests freely.
    void set_sampling(const strata::kernels::SamplerParams& sp) {
        sampling_ = sp;   // row t of a window at pos0 draws Philox(seed, pos0 + t): see run()
        if (next_) next_->set_sampling(sp);
    }

    /// The penalty histories for `sampling_.penalty_last_n`: ONE ROW PER WINDOW ROW, T rows of `history_len`
    /// int32 slots at that stride (`strata::kernels::penalty_rows` builds them), most recent token LAST, unused
    /// front slots -1 (the kernel reads only the tail window).  Row t follows the window's drafts 1..t - staging
    /// row 0 alone (before 0.1.19) left the drafted rows with unwritten histories.  Null disables the penalties
    /// entirely - the neutral run's sampling call is byte-for-byte what it was.  The engine re-uploads the rows
    /// before every window; the buffer must hold kVerifyMaxT rows and stay alive across the request.
    void set_history(const int32_t* history, int history_len) {
        hist_d_ = history;
        hist_len_ = history_len;
        if (next_) next_->set_history(history, history_len);
    }
    /// Off: `run` skips the request's head sampling and `out` is the recorded greedy pick.  For windows whose
    /// picks are discarded - a prompt read through windows commits every token - so they cost no sampler launch
    /// or sync and never read a history staged for another position.
    void set_head_sampling(bool on) { head_sampling_ = on; if (next_) next_->set_head_sampling(on); }

    /// LAYER SPLIT (multi-GPU): this verifier runs layers [layer_begin, layer_end) of every window.  A stage that
    /// does not start at layer 0 takes its residual from `handoff_in` instead of embedding the tokens; a stage that
    /// does not end at the last layer writes its residual to `handoff_out` and has no head.  The hand-off holds,
    /// per token, the residual R (hc x n_embd), the last layer's pending write bo (n_embd) and inject (hc): the next
    /// stage folds that write into its first read exactly as the unsplit window does, so the split is bit-exact.
    /// Both pointers must be device-visible (mapped pinned memory, portable when the stages are on two devices).
    /// Set before `init`.  Default: the whole model, no hand-off.
    void set_stage(int64_t layer_begin, int64_t layer_end, const float* handoff_in, float* handoff_out) {
        lb_ = layer_begin; le_ = layer_end;
        hand_in_pairs_[0] = hand_in_pairs_[1] = handoff_in;
        hand_out_pairs_[0] = hand_out_pairs_[1] = handoff_out;
        hand_in_ = handoff_in; hand_out_ = handoff_out;
    }
    /// STAGE-PIPELINE OVERLAP (lane overlap): a PING-PONG hand-off.  Two buffer pairs (parities A/B) are
    /// installed; `set_hand_parity` selects the ACTIVE pair before a pass begins.  stage0[N+1] writes parity
    /// (N+1)&1 while stage1[N] still reads parity N&1, so the two passes can be in flight at once.  The
    /// captured graphs bake the active pointers, so captures are keyed by the parity as well.  The serial
    /// engine keeps `set_stage` (both parities hold the same pointers).
    void set_stage_pingpong(int64_t layer_begin, int64_t layer_end,
                            const float* in_a, float* out_a, const float* in_b, float* out_b) {
        lb_ = layer_begin; le_ = layer_end;
        hand_in_pairs_[0] = in_a; hand_out_pairs_[0] = out_a;
        hand_in_pairs_[1] = in_b; hand_out_pairs_[1] = out_b;
        hand_in_pairs_[2] = in_a; hand_out_pairs_[2] = out_a;
        hand_in_ = in_a; hand_out_ = out_a; parity_ = 0;
    }
    /// Task 8.2: Triple-Buffered Circular Transfer Arena (TripleBufferIPC)
    void set_stage_ring(int64_t layer_begin, int64_t layer_end,
                        const float* in_0, float* out_0,
                        const float* in_1, float* out_1,
                        const float* in_2, float* out_2) {
        lb_ = layer_begin; le_ = layer_end;
        hand_in_pairs_[0] = in_0; hand_out_pairs_[0] = out_0;
        hand_in_pairs_[1] = in_1; hand_out_pairs_[1] = out_1;
        hand_in_pairs_[2] = in_2; hand_out_pairs_[2] = out_2;
        hand_in_ = in_0; hand_out_ = out_0; parity_ = 0;
    }
    void set_hand_parity(int p) {
        parity_ = (p % 3 + 3) % 3;
        hand_in_ = hand_in_pairs_[parity_];
        hand_out_ = hand_out_pairs_[parity_];
    }
    int hand_parity() const { return parity_; }
    /// The next stage: `run` and `commit` continue into it (its pool calls get `next_user`); sampling settings
    /// and `final_R` are the last stage's.
    void set_next(Verifier* next, void* next_user) { next_ = next; next_user_ = next_user; }
    /// The next stage of the chain (null on a last stage): the stage-pipeline driver derives a batch
    /// unit's stage-1 member list from it (the serial chain derives the same list in run_batch).
    Verifier* next() const { return next_; }
    /// floats per token in a hand-off buffer
    static int64_t handoff_floats(const ModelGeometry& g) { return (int64_t) g.hc * g.n_embd + g.n_embd + g.hc; }

    /// Keep the first `n_keep` (1..T) tokens of the last window; advances `ss.ple_prev` by them.
    bool commit(int n_keep, std::string& err);          ///< launch the chain and wait its tail (the pair below)
    bool commit_launch(int n_keep, std::string& err);   ///< fill + launch this stage's commit and chain on (no wait)
    bool commit_wait(std::string& err);                 ///< wait the CHAIN TAIL's stream (commit_launch's pair)
    static void set_commit_async(bool on);
    bool wait_commit(std::string& err);
    bool window_logprobs(const int32_t* targets, int T, int64_t pos0, int32_t extra_id, std::FILE* out,
                         std::string& err);

    /// Token t's residual after the last layer, (hc, n_embd) on the device, valid until the next `run`.
    const float* final_R(int t) const;
    const float* final_R_all() const { return next_ ? next_->final_R_all() : R_; }

    /// The GPU plan the pool writes each layer (VRAM hits + the PCIe share of the misses); give it to the
    /// dispatch (`ExpertDispatch::plan`) before the first `run`.
    GpuPlanSink* plan_sink() { return &sink_; }
    /// Plan v0.3 P6: split the window into two token groups and pipeline the CPU experts of one with the GPU work
    /// of the other (default on).  Set before the first `run`.
    void set_split(bool on) { split_ = on; }
    /// Plan v0.3 P6: how the PCIe share of the misses reaches the GPU: 0 = DMA into staging (the copy engine works
    /// beside the CPU; best when the CPU is compute-bound, the i-quants), 1 = the grouped kernel reads the mapped
    /// arena directly, 2 = a copy kernel stages it inside the graph (no API calls on the pool's thread; best when
    /// the CPU is RAM-bound, Q2_0).  Set before the first `run`.
    void set_pcie_mode(int mode) { sink_.pcie_mode = mode; }
    /// the pool never plans a PCIe share (--pcie-frac 0): the window skips that path.  Before the first run.

    double ms_wait = 0, ms_pool = 0, ms_host = 0, ms_commit = 0;
    double ms_batch_capture = 0;
    int64_t batch_captures = 0;
    double batch_gpu_ms[4] = {}; // member pre, shared expert dispatch, member post, head
    int64_t windows = 0;
    /// LANE sched-impl P3 (host-loop O3 counter-only): drive_passes no-progress accounting, written
    /// only when STRATA_DRIVE_HISTO is set. Zero wait-posture change; the O3 behavior change (if any)
    /// waits for the H1 repair.
    uint64_t drive_idle_iters = 0, drive_idle_maxburst = 0, drive_calls = 0;
    /// Copy-stream host functions queued by fetch_dma (n > 0) since construction.  A window syncs the
    /// copy stream only when IT queued one: a window that queued none cannot raise flag B in the next
    /// one, so the stream is already drained (run()).
    std::atomic<uint32_t> fetches_{0};
    /// STRATA_VERIFY_PROFILE=1 - GPU stage times of the windows since the last call (ms per
    /// window), as one line; empty when off.
    std::string profile_report();

private:
    bool capture(int T, std::string& err);
    strata::kernels::SamplerParams sampling_ = [] {
        strata::kernels::SamplerParams s;
        s.greedy = true;
        s.temperature = 0.0f;
        return s;
    }();   ///< greedy by default; per-request via set_sampling
    const int32_t* hist_d_ = nullptr;   ///< penalty-history row (set_history); null = no penalties apply
    int hist_len_ = 0;
    bool head_sampling_ = true;          ///< set_head_sampling
    int device_ = -1;                    ///< the device `init` ran on: run/commit switch to it (layer split)
    std::atomic<bool> released_{false};  ///< #267: release_gpu_waits ran (maybe on the watchdog thread): no more windows
    bool device_plan_ = false;            ///< E-6: resident-only layers planned on the device (STRATA_VERIFY_DEVICE_PLAN)
    uint32_t* skip_ = nullptr;            ///< E-6: per group, the ring whose plan the device built (0: the host's)
    unsigned long long* slot_off_d_ = nullptr;   ///< E-6: the slot offsets on the device
    int64_t lb_ = 0, le_ = -1;           ///< set_stage: the layers this verifier runs (-1: to the last)
    const float* hand_in_ = nullptr;     ///< the ACTIVE hand-off pair (set_hand_parity selects)
    float* hand_out_ = nullptr;
    const float* hand_in_pairs_[3] = {}; ///< the ring/ping-pong pairs
    float* hand_out_pairs_[3] = {};
    int parity_ = 0;                     ///< the active parity; the captures are keyed by it
    Verifier* next_ = nullptr;
    void* next_user_ = nullptr;
    bool ple_stage() const { return lb_ <= 1 && 1 < le_; }   ///< holds layer 1, where the PLE block runs
    bool capture_commit(std::string& err);
    /// phase < 0: the single-window record (unchanged, row0 = 0); 0: window inputs (lb_ > 0 reads
    /// the hand-off); 1: pre(l); 2: post(l) expert stage; 3: member combine; 4: head; 5: stage
    /// hand-off OUT (a split's earlier stage).  `row0` = this window's first row in a batch
    /// round's packed hand-off buffer; single-window callers leave it 0.
    bool record_window(int T, cudaStream_t cs, std::string& err, int phase = -1, int64_t layer = 0,
                       int64_t row0 = 0);
    bool stage_inputs(int T, const int32_t* tokens, int64_t pos0, std::string& err);
    /// C2: the shared batch-round preparation (validation, per-member staging, graph lookup and
    /// capture) - run_batch (serial) and begin_pass_batch (stage-pipeline overlap) both call it.
    bool prepare_batch(const std::vector<BatchWindow>& batch, int& total,
                       std::vector<std::pair<Verifier*, int>>& shape, cudaGraphExec_t& graph_exec,
                       std::string& err);
    struct BatchGraph { std::vector<std::pair<Verifier*, int>> shape; int parity = 0; cudaGraphExec_t graph = nullptr; };
    std::vector<BatchGraph> batch_graphs_;
    int batch_cache_limit_ = 8, batch_reserve_mib_ = 0;
    bool batch_parallel_ = false;
    cudaEvent_t batch_fork_ = nullptr, batch_join_[16] = {};   ///< batch-parallel member events, one per request (16 = the M2 request lattice)
    cudaGraphExec_t batch_replay_ = nullptr;
    static constexpr int kProfPer = 33;              // stamps per layer (33 avoids collision)
    bool prof_on_ = false;
    unsigned long long* prof_ = nullptr;              // device: n_layers * kProfPer + 4 stamps
    std::vector<unsigned long long> prof_h_;
    double prof_sum_[2][kProfPer] = {};   // [GDN / QSA layers][stage]
    int64_t prof_windows_ = 0;

    const WeightTable* wt_ = nullptr;
    const ModelGeometry* g_ = nullptr;
    SessionState* ss_ = nullptr;
    VerifyHits hits_;
    const NativeHead* head_ = nullptr;
    int max_t_ = 0;
    int last_t_ = 0;
    int64_t last_pos0_ = 0;
    int32_t last_tokens_[8] = {};
    int64_t n_vocab_ = 0;
    cudaStream_t cs_ = nullptr;
    cudaGraphExec_t exec_[3][9] = {};   ///< [parity][T]: the hand-off pointers are baked into the capture
    cudaGraphExec_t commit_exec_ = nullptr;

    // ---- stage-pipeline overlap: the pass begun by begin_pass_* and serviced by pass_step ----
    bool pass_live_ = false;            ///< a pass is begun and not yet finished
    int64_t pass_k_ = 0;                ///< doorbells serviced in this pass
    int64_t pass_total_ = 0;            ///< (le_-lb_) * G of this pass
    PoolMultiFn pass_pool_ = nullptr;   ///< the pool of this pass (its stage's dispatch)
    void* pass_user_ = nullptr;
    std::chrono::steady_clock::time_point pass_wait_start_{};   ///< when the CURRENT layer's wait began
    std::chrono::steady_clock::time_point pass_last_flush_{};   ///< the last cudaStreamQuery of the stall check
    bool pass_wait_reported_ = false;
    cudaEvent_t deferred_done_ = nullptr; ///< R2 (split1): a pass's completion event, recorded at
                                          ///< the pass-end sync (end_pass_window/end_pass_batch);
                                          ///< launch_pass documents why it is not recorded at launch.
    uint32_t pass_fetches_at_launch_ = 0; ///< sync invariant (rebase): fetches_ as seen by
                                       ///< launch_pass; end_pass_* syncs copy_ only when a
                                       ///< fetch_dma(n > 0) landed after it        ///< progress_at was published for this layer's wait
    bool launch_pass(int T, cudaEvent_t done, std::string& err);
    bool service_one(int64_t k, PoolMultiFn pool, void* user, std::string& err);
    bool pass_doorbell(uint32_t want) const { return *(volatile uint32_t*) h_seq_ >= want; }
    bool pass_stall(uint32_t want, int64_t l, std::chrono::steady_clock::time_point wait_start,
                    std::chrono::steady_clock::time_point& last_flush, std::string& err);
    void pass_prof_dump();
    bool pass_finish_outputs(int T, int32_t* out, std::string& err);

    // mapped staging (host pointer, device alias)
    int32_t* h_tok_ = nullptr;   int32_t* m_tok_ = nullptr;     // T
    int32_t* h_step_ = nullptr;  int32_t* m_step_ = nullptr;    // T * kStepCount
    int32_t* h_pos_ = nullptr;   int32_t* m_pos_ = nullptr;     // T * n_head
    int32_t* h_commit_ = nullptr; int32_t* m_commit_ = nullptr; // [n_keep, n_keep-1, pos_0 .. pos_{T-1}]
    float* h_ple_ = nullptr;     float* m_ple_ = nullptr;       // T * n_embd
    int32_t* h_out_ = nullptr;   int32_t* m_out_ = nullptr;     // T argmax ids
    float* h_probs_ = nullptr;   float* m_probs_ = nullptr;     // T target draft probabilities (ALGO-02)
    float* h_x_ = nullptr;       float* m_x_ = nullptr;         // doorbell payload: T * n_embd
    int32_t* h_ids_ = nullptr;   int32_t* m_ids_ = nullptr;     // T * k
    float* h_w_ = nullptr;       float* m_w_ = nullptr;         // T * k
    uint32_t* h_seq_ = nullptr;  uint32_t* m_seq_ = nullptr;
    uint32_t* h_flag_ = nullptr; uint32_t* m_flag_ = nullptr;
    uint32_t* h_flagA_ = nullptr; uint32_t* m_flagA_ = nullptr;  // the GPU plan is in place
    uint32_t* h_flagB_ = nullptr; uint32_t* m_flagB_ = nullptr;  // the PCIe share's DMA copies have landed
    cudaEvent_t commit_done_ = nullptr;   // recorded after an async commit (set_commit_async); see wait_commit
    bool commit_pending_ = false;
    cudaStream_t copy_ = nullptr;                                 // the copy engine's stream (DMA of missed experts)
    struct FlagSet { uint32_t* flag; uint32_t value; };
    FlagSet flag_sets_[2 * 64 * 2] = {};                          // host-function arguments, one per (layer, group)
    static void fetch_dma(void* ctx, const uint8_t* const* src, int n, size_t bytes);
    static void raise_flag(uint32_t* flag, uint32_t value);
    int32_t* h_plan_ = nullptr;  int32_t* m_plan_ = nullptr;     // counts | start | dst | tok | ptr (as int32 pairs)
    int64_t plan_i32_ = 0;                                        // int32 words in the plan block
    GpuPlanSink sink_;
    uint32_t cur_layer_ = 0;
    static void publish_plan(void* ctx);
    void set_plan_slot(int grp);
    bool split_ = false;   // opt-in (--spec-split): exact but slower, see the overlap study
    int groups_[strata::kernels::cpu::MAXT + 1] = {};   ///< group count by window size T (2 = the split window)
    float* h_ymiss_ = nullptr;   float* m_ymiss_ = nullptr;     // T * k * n_embd

    // device
    void* arena_ = nullptr;
    int32_t *tok_ = nullptr, *step_ = nullptr, *pos_ = nullptr, *commit_ = nullptr;
    float *ple_ = nullptr, *emb_ = nullptr, *R_ = nullptr, *mixed_ = nullptr, *bo_ = nullptr;
    float *inj_ = nullptr, *inj2_ = nullptr, *lo_ = nullptr, *rs_ = nullptr, *xn_ = nullptr;
    uint8_t* xq_ = nullptr;                                   // T columns of q8_1
    uint8_t* xil_ = nullptr;                                  // fork F4: the interleaved copy of xq_'s 2-4 columns
    float *qkv_L_ = nullptr, *h_L_ = nullptr, *gate_L_ = nullptr, *beta_L_ = nullptr;   // per GDN layer
    float *z_ = nullptr, *y_ = nullptr, *y_dummy_ = nullptr;
    float *qfull_ = nullptr, *qcur_ = nullptr, *kcur_ = nullptr, *vcur_ = nullptr, *idx_raw_L_ = nullptr;
    float *qidx_ = nullptr, *scores_ = nullptr, *attn_ = nullptr, *attn32_ = nullptr, *attn_scratch_ = nullptr;
    float* tail_snap_ = nullptr;                              // per QSA layer
    int32_t* block_pos_snap_ = nullptr;                       // per QSA layer (ALGO-03)
    float* pooled_snap_ = nullptr;                            // per QSA layer (ALGO-03)
    int32_t* sel_ = nullptr;
    float *logits_ = nullptr, *w_ = nullptr, *shared_ = nullptr, *parts_ = nullptr, *hit_out_ = nullptr;
    int32_t *ids_ = nullptr, *hit_slot_ = nullptr, *hit_dst_ = nullptr, *hit_count_ = nullptr;
    int32_t* plan_ = nullptr;                                     // device copy of the plan block
    uint8_t* staging_ = nullptr;                                  // VRAM slots for the PCIe share of the misses
    static constexpr int64_t kStagingBlobs = 16;
    static constexpr int64_t kPcieGroupRows = 4;                  // the PCIe call's groups side by side (of <= 16)
    uint8_t* hit_xq_ = nullptr;
    uint8_t* nat_xq_ = nullptr;   // plan v0.3 P6: q8_1 activations for a native pack's grouped experts
    float* hit_xs_ = nullptr;
    void* hit_scratch_ = nullptr;
    float *head_mixed_ = nullptr, *head_inj_ = nullptr, *head_logits_ = nullptr;
    uint16_t* sh_bf16_ = nullptr;
    uint8_t* arg_scratch_ = nullptr;   ///< argmax_rows' partials and counters
    float *sh_gate_ = nullptr, *sh_up_ = nullptr, *sh_g_ = nullptr;
    float* hist_snap_ = nullptr;                              // T * NG_HIST * NG_HC_DIM
    int64_t cap_ = 0, max_blocks_ = 0, attn_scratch_floats_ = 0;
};

}  // namespace strata::core
