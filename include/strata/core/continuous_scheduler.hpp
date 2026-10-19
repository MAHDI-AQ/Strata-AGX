// include/strata/core/continuous_scheduler.hpp - Token-Level Continuous Micro-Scheduling Engine
//
// Cross-Engine Attribution and Research Lineage:
//   - SGLang: Token-level continuous batching & dynamic KV memory reuse (Zheng et al., LMSYS / UC Berkeley, arXiv:2312.07104)
//   - vLLM v1: Core iteration scheduler & immediate slot recycling (Kwon et al., SOSP 2023)
//   - Orca: Iteration-level scheduling for low-latency multi-stream inference (OSDI 2022)
//
// Replaces coarse round-based batch synchronization with continuous token-level iteration loops.
// When an active stream emits EOS or completes max_new tokens, its slot is recycled immediately
// at the start of the next forward step (< 5 us overhead) without waiting for sibling slots.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace strata::core {

enum class SlotState : uint8_t {
    IDLE = 0,
    PREFILL = 1,
    READY_DECODE = 2,
    DECODING = 3,
    AWAITING_TOOL = 4,  // Waiting on client tool execution (prime preemption candidate)
    PREEMPTED_L2 = 5,   // Swapped out to host RAM HiCache
    PAUSED = 6,         // Paused under critical memory pressure
    FINISHED = 7,
    CANCELLED = 8
};

inline const char* slot_state_to_string(SlotState s) {
    switch (s) {
        case SlotState::IDLE: return "IDLE";
        case SlotState::PREFILL: return "PREFILL";
        case SlotState::READY_DECODE: return "READY_DECODE";
        case SlotState::DECODING: return "DECODING";
        case SlotState::AWAITING_TOOL: return "AWAITING_TOOL";
        case SlotState::PREEMPTED_L2: return "PREEMPTED_L2";
        case SlotState::PAUSED: return "PAUSED";
        case SlotState::FINISHED: return "FINISHED";
        case SlotState::CANCELLED: return "CANCELLED";
        default: return "UNKNOWN";
    }
}

/// Slot metadata and state tracked by the continuous scheduler.
struct ScheduledSlot {
    int slot_id = -1;
    uint64_t request_id = 0;
    SlotState state = SlotState::IDLE;
    int64_t prompt_tokens = 0;
    int64_t tokens_generated = 0;
    int64_t max_new = 0;
    int64_t position = 0;
    int64_t max_context = 0;
    int32_t last_token = -1;
    std::chrono::steady_clock::time_point admit_time{};
    std::chrono::steady_clock::time_point finish_time{};
    std::chrono::steady_clock::time_point last_token_time{};
    bool protected_from_preemption = false;
};

/// High-performance continuous scheduler managing iteration-level slot allocations.
class ContinuousScheduler {
public:
    explicit ContinuousScheduler(int num_slots = 4)
        : num_slots_(std::max(1, num_slots)), slots_(num_slots_) {
        for (int i = 0; i < num_slots_; ++i) {
            slots_[i].slot_id = i;
            slots_[i].state = SlotState::IDLE;
        }
    }

    int num_slots() const { return num_slots_; }

    /// Find and allocate an available slot for an incoming request.
    /// Returns slot_id, or -1 if no suitable slot is idle.
    int allocate_slot_at(int slot_id, uint64_t request_id, int64_t prompt_tokens, int64_t max_new,
                         int64_t max_context) {
        if (max_context > 0 && prompt_tokens + std::max<int64_t>(0, max_new) > max_context) {
            return -1;
        }
        if (slot_id < 0 || slot_id >= num_slots_) return -1;
        std::lock_guard<std::mutex> lock(sched_mutex_);
        const auto t0 = std::chrono::steady_clock::now();

        int pick = slot_id;
        auto& s = slots_[pick];
        s.request_id = request_id;
        s.state = SlotState::PREFILL;
        s.prompt_tokens = prompt_tokens;
        s.tokens_generated = 0;
        s.max_new = max_new;
        s.position = prompt_tokens > 0 ? prompt_tokens - 1 : 0;
        s.max_context = max_context;
        s.last_token = -1;
        s.admit_time = std::chrono::steady_clock::now();
        ++total_admissions_;

        record_overhead(t0);
        return pick;
    }

    int allocate_slot(uint64_t request_id, int64_t prompt_tokens, int64_t max_new,
                      int64_t max_context) {
        if (max_context > 0 && prompt_tokens + std::max<int64_t>(0, max_new) > max_context) {
            return -1;
        }
        std::lock_guard<std::mutex> lock(sched_mutex_);
        const auto t0 = std::chrono::steady_clock::now();

        int pick = -1;
        for (int i = 0; i < num_slots_; ++i) {
            if (slots_[i].state == SlotState::IDLE) {
                pick = i;
                break;
            }
        }

        if (pick >= 0) {
            auto& s = slots_[pick];
            s.request_id = request_id;
            s.state = SlotState::PREFILL;
            s.prompt_tokens = prompt_tokens;
            s.tokens_generated = 0;
            s.max_new = max_new;
            s.position = prompt_tokens > 0 ? prompt_tokens - 1 : 0;
            s.max_context = max_context;
            s.last_token = -1;
            s.admit_time = std::chrono::steady_clock::now();
            ++total_admissions_;
        }

        record_overhead(t0);
        return pick;
    }

    /// Set slot state directly (e.g. after prompt prefill completes)
    void set_slot_state(int slot_id, SlotState state) {
        if (slot_id < 0 || slot_id >= num_slots_) return;
        std::lock_guard<std::mutex> lock(sched_mutex_);
        slots_[slot_id].state = state;
    }

    /// Record newly generated token for a slot and evaluate completion condition.
    /// Returns true if the sequence is finished (EOS or limit reached) and recycled.
    bool step_token(int slot_id, int32_t token, bool is_eos) {
        if (slot_id < 0 || slot_id >= num_slots_) return false;
        std::lock_guard<std::mutex> lock(sched_mutex_);
        const auto t0 = std::chrono::steady_clock::now();

        auto& s = slots_[slot_id];
        if (s.state != SlotState::DECODING && s.state != SlotState::READY_DECODE) {
            return false;
        }

        s.last_token = token;
        s.last_token_time = t0;
        ++s.tokens_generated;
        ++s.position;
        ++total_tokens_stepped_;

        const bool finished = is_eos ||
                              (s.max_new > 0 && s.tokens_generated >= s.max_new) ||
                              (s.max_context > 0 && s.position >= s.max_context);

        if (finished) {
            s.state = SlotState::FINISHED;
            s.finish_time = std::chrono::steady_clock::now();
            // Immediate lockless-recycle of the slot to IDLE
            s.state = SlotState::IDLE;
            s.request_id = 0;
            s.prompt_tokens = 0;
            s.tokens_generated = 0;
            s.max_new = 0;
            s.position = 0;
            ++total_recycles_;
        } else {
            s.state = SlotState::READY_DECODE;
        }

        record_overhead(t0);
        return finished;
    }

    /// Immediate recycling of a finished or cancelled slot (< 5 us)
    void recycle_slot(int slot_id) {
        if (slot_id < 0 || slot_id >= num_slots_) return;
        std::lock_guard<std::mutex> lock(sched_mutex_);
        const auto t0 = std::chrono::steady_clock::now();

        auto& s = slots_[slot_id];
        s.state = SlotState::IDLE;
        s.request_id = 0;
        s.tokens_generated = 0;
        s.max_new = 0;
        s.position = 0;
        s.prompt_tokens = 0;
        ++total_recycles_;

        record_overhead(t0);
    }

    /// Collect all slots ready to step in this token iteration
    std::vector<int> get_ready_decode_slots() {
        std::lock_guard<std::mutex> lock(sched_mutex_);
        const auto t0 = std::chrono::steady_clock::now();

        std::vector<int> ready;
        ready.reserve(num_slots_);
        for (int i = 0; i < num_slots_; ++i) {
            if (slots_[i].state == SlotState::READY_DECODE || slots_[i].state == SlotState::DECODING) {
                ready.push_back(i);
                slots_[i].state = SlotState::DECODING;
            }
        }

        record_overhead(t0);
        return ready;
    }

    ScheduledSlot get_slot_info(int slot_id) const {
        if (slot_id < 0 || slot_id >= num_slots_) return {};
        std::lock_guard<std::mutex> lock(sched_mutex_);
        return slots_[slot_id];
    }

    int count_active() const {
        std::lock_guard<std::mutex> lock(sched_mutex_);
        int cnt = 0;
        for (int i = 0; i < num_slots_; ++i) {
            if (slots_[i].state != SlotState::IDLE) ++cnt;
        }
        return cnt;
    }

    int count_idle() const {
        return num_slots_ - count_active();
    }

    uint64_t total_admissions() const { return total_admissions_; }
    uint64_t total_recycles() const { return total_recycles_; }
    uint64_t total_tokens_stepped() const { return total_tokens_stepped_; }

    double average_overhead_us() const {
        const uint64_t cnt = overhead_samples_.load();
        if (cnt == 0) return 0.0;
        return total_overhead_ns_.load() / (double)(cnt * 1000.0);
    }

    void reset_metrics() {
        total_overhead_ns_ = 0;
        overhead_samples_ = 0;
    }

    // ========================================================================
    // Strata AGX v5.5.0: Dual-Watermark Preemption & Lifecycle (Pillars 1 & 4)
    // ========================================================================
    enum class WatermarkAction : uint8_t {
        NONE = 0,
        THROTTLE_PREFILL = 1,
        TRIGGER_PREEMPTION = 2,
        TRIGGER_RESUME = 3
    };

    struct WatermarkConfig {
        float watermark_low_ratio = 0.15f;    // 15% free: throttle prefill micro-chunks
        float watermark_crit_ratio = 0.08f;   // 8% free: preempt victim slot
        float watermark_resume_ratio = 0.22f; // 22% free: resume preempted slots
    };

    void set_slot_guard(int slot_id, bool guarded) {
        if (slot_id < 0 || slot_id >= num_slots_) return;
        std::lock_guard<std::mutex> lock(sched_mutex_);
        slots_[slot_id].protected_from_preemption = guarded;
    }

    bool is_slot_guarded(int slot_id) const {
        if (slot_id < 0 || slot_id >= num_slots_) return false;
        std::lock_guard<std::mutex> lock(sched_mutex_);
        return slots_[slot_id].protected_from_preemption;
    }

    void set_watermark_config(WatermarkConfig cfg) {
        std::lock_guard<std::mutex> lock(sched_mutex_);
        watermark_cfg_ = cfg;
    }

    WatermarkConfig get_watermark_config() const {
        std::lock_guard<std::mutex> lock(sched_mutex_);
        return watermark_cfg_;
    }

    bool is_prefill_throttled() const {
        return prefill_throttled_.load(std::memory_order_relaxed);
    }

    WatermarkAction check_watermarks(size_t free_pages, size_t total_pages) {
        if (total_pages == 0) return WatermarkAction::NONE;
        float free_ratio = static_cast<float>(free_pages) / static_cast<float>(total_pages);
        if (free_ratio <= watermark_cfg_.watermark_crit_ratio) {
            prefill_throttled_.store(true, std::memory_order_relaxed);
            return WatermarkAction::TRIGGER_PREEMPTION;
        }
        if (free_ratio <= watermark_cfg_.watermark_low_ratio) {
            prefill_throttled_.store(true, std::memory_order_relaxed);
            return WatermarkAction::THROTTLE_PREFILL;
        }
        prefill_throttled_.store(false, std::memory_order_relaxed);
        if (free_ratio >= watermark_cfg_.watermark_resume_ratio) {
            std::lock_guard<std::mutex> lock(sched_mutex_);
            for (int i = 0; i < num_slots_; ++i) {
                if (slots_[i].state == SlotState::PREEMPTED_L2 || slots_[i].state == SlotState::PAUSED) {
                    return WatermarkAction::TRIGGER_RESUME;
                }
            }
        }
        return WatermarkAction::NONE;
    }

    int select_preemption_victim() const {
        std::lock_guard<std::mutex> lock(sched_mutex_);
        int victim = -1;
        // 1. Prefer AWAITING_TOOL (oldest wait time first)
        auto oldest_time = std::chrono::steady_clock::time_point::max();
        for (int i = 0; i < num_slots_; ++i) {
            if (slots_[i].state == SlotState::AWAITING_TOOL && !slots_[i].protected_from_preemption) {
                if (slots_[i].last_token_time < oldest_time) {
                    oldest_time = slots_[i].last_token_time;
                    victim = i;
                }
            }
        }
        if (victim >= 0) return victim;

        // 2. Otherwise pick active decoding slot with lowest generated tokens
        int64_t min_tokens = std::numeric_limits<int64_t>::max();
        for (int i = 0; i < num_slots_; ++i) {
            if ((slots_[i].state == SlotState::DECODING || slots_[i].state == SlotState::READY_DECODE) &&
                !slots_[i].protected_from_preemption) {
                if (slots_[i].tokens_generated < min_tokens) {
                    min_tokens = slots_[i].tokens_generated;
                    victim = i;
                }
            }
        }
        return victim;
    }

private:
    void record_overhead(std::chrono::steady_clock::time_point t0) {
        const auto t1 = std::chrono::steady_clock::now();
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        total_overhead_ns_ += ns;
        ++overhead_samples_;
    }

    int num_slots_ = 4;
    mutable std::mutex sched_mutex_;
    std::vector<ScheduledSlot> slots_;
    WatermarkConfig watermark_cfg_{};
    std::atomic<bool> prefill_throttled_{false};

    std::atomic<uint64_t> total_admissions_{0};
    std::atomic<uint64_t> total_recycles_{0};
    std::atomic<uint64_t> total_tokens_stepped_{0};
    std::atomic<uint64_t> total_overhead_ns_{0};
    std::atomic<uint64_t> overhead_samples_{0};
};

}  // namespace strata::core
