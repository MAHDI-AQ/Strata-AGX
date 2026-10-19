#pragma once
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>
#include <cstdint>
#include <mutex>

#include "strata/kernels/cpu/expert.hpp"   // MAXT: the batch row envelope this budget is capped by

namespace strata::program {
// Requests are supplied in rotating fairness order. Zero means this request waits for the next iteration.
inline std::vector<int> schedule_rows(const std::vector<int>& wanted, int budget, bool depth) {
    // The budget is the batch ROW ENVELOPE (MAXT): the pool's window tables, the verify workspace and
    // `--batch-rows` all follow that one constant - a cap is a lattice, keep them in lockstep.
    // The request lattice is 16 concurrent requests (M2 raise from 8; 16-legal/17-reject is covered in
    // the schedule tests).
    if (budget < 1 || budget > strata::kernels::cpu::MAXT || wanted.size() > 16)
        throw std::invalid_argument("batch budget must be 1.." + std::to_string(strata::kernels::cpu::MAXT) +
                                    ", requests <= 16");
    for (int n : wanted) if (n < 1 || n > 8) throw std::invalid_argument("window must be 1..8");
    std::vector<int> rows(wanted.size(), 0);
    if (depth) {
        for (size_t i = 0; i < wanted.size() && budget; ++i) {
            rows[i] = std::min(wanted[i], budget);
            budget -= rows[i];
        }
    } else {
        bool changed = true;
        while (budget && changed) {
            changed = false;
            for (size_t i = 0; i < wanted.size() && budget; ++i)
                if (rows[i] < wanted[i]) { ++rows[i]; --budget; changed = true; }
        }
    }
    return rows;
}

// ============================================================================
// Strata AGX v5.4.0: 4 Uniform Dynamic Slots Continuous Scheduler
// ============================================================================
constexpr size_t MAX_UNIFORM_SLOTS = 4;
constexpr size_t MAX_PAGES_PER_SLOT = 65536; // 65536 * 4 tokens/page = 262,144 max context
constexpr size_t SLOT_PAGE_SHIFT = 16;       // 1 << 16 = 65536

enum class SlotState : uint8_t {
    IDLE = 0,
    PREFILL = 1,
    DECODE = 2,
    PAUSED = 3,
    DRAINING = 4
};

struct SlotScheduleMetrics {
    int slot_id{0};
    SlotState state{SlotState::IDLE};
    size_t allocated_pages{0};
    size_t pending_prefill_tokens{0};
    size_t generated_tokens{0};
    int64_t deficit_tokens{0};
    float priority_weight{1.0f};
};

struct ContinuousSchedulerConfig {
    size_t total_paged_pool_capacity{524288}; // Total tokens in PagedKVPool
    size_t page_size_tokens{16};
    size_t prefill_quantum_tokens{1024};
    size_t decode_quantum_tokens{1};
    float watermark_low_ratio{0.10f};  // 10% free pages: throttle
    float watermark_crit_ratio{0.05f}; // 5% free pages: preempt/pause
    float watermark_high_ratio{0.20f}; // 20% free pages: resume
};

class ContinuousScheduler {
public:
    explicit ContinuousScheduler(ContinuousSchedulerConfig cfg = {})
        : cfg_(cfg),
          total_pages_(cfg.total_paged_pool_capacity / cfg.page_size_tokens),
          watermark_low_pages_(static_cast<size_t>(total_pages_ * cfg.watermark_low_ratio)),
          watermark_crit_pages_(static_cast<size_t>(total_pages_ * cfg.watermark_crit_ratio)),
          watermark_high_pages_(static_cast<size_t>(total_pages_ * cfg.watermark_high_ratio)) {
        for (size_t i = 0; i < MAX_UNIFORM_SLOTS; ++i) {
            slots_[i].slot_id = static_cast<int>(i);
            slots_[i].state = SlotState::IDLE;
            slots_[i].deficit_tokens = static_cast<int64_t>(cfg_.prefill_quantum_tokens);
            slots_[i].priority_weight = 1.0f;
        }
    }

    struct BatchPlan {
        int active_slots[MAX_UNIFORM_SLOTS]{-1, -1, -1, -1};
        size_t tokens_to_process[MAX_UNIFORM_SLOTS]{0, 0, 0, 0};
        bool is_prefill[MAX_UNIFORM_SLOTS]{false, false, false, false};
        size_t scheduled_count{0};
        bool watermark_throttled{false};
    };

    BatchPlan schedule_batch(size_t currently_free_pages) {
        std::lock_guard<std::mutex> lock(sched_mutex_);
        BatchPlan plan;
        bool crit_mode = (currently_free_pages <= watermark_crit_pages_);
        bool low_mode = (currently_free_pages <= watermark_low_pages_);
        plan.watermark_throttled = low_mode;

        for (size_t i = 0; i < MAX_UNIFORM_SLOTS; ++i) {
            size_t idx = (rr_cursor_ + i) % MAX_UNIFORM_SLOTS;
            auto& slot = slots_[idx];

            if (slot.state == SlotState::IDLE || slot.state == SlotState::PAUSED) {
                continue;
            }

            if (crit_mode && slot.state == SlotState::PREFILL) {
                continue;
            }

            slot.deficit_tokens += static_cast<int64_t>(
                (slot.state == SlotState::PREFILL ? cfg_.prefill_quantum_tokens : cfg_.decode_quantum_tokens)
                * slot.priority_weight
            );

            if (slot.deficit_tokens > 0) {
                size_t step_tokens = 0;
                bool prefill = (slot.state == SlotState::PREFILL);
                if (prefill) {
                    step_tokens = std::min(static_cast<size_t>(slot.deficit_tokens), slot.pending_prefill_tokens);
                    if (low_mode) {
                        step_tokens = std::min(step_tokens, cfg_.prefill_quantum_tokens / 2);
                    }
                    slot.deficit_tokens -= step_tokens;
                } else {
                    step_tokens = 1;
                    slot.deficit_tokens -= cfg_.decode_quantum_tokens;
                }

                if (step_tokens > 0) {
                    plan.active_slots[plan.scheduled_count] = slot.slot_id;
                    plan.tokens_to_process[plan.scheduled_count] = step_tokens;
                    plan.is_prefill[plan.scheduled_count] = prefill;
                    plan.scheduled_count++;
                }
            }
        }

        rr_cursor_ = (rr_cursor_ + 1) % MAX_UNIFORM_SLOTS;
        return plan;
    }

    void set_slot_state(int slot_id, SlotState st, size_t pending_prefill = 0) {
        std::lock_guard<std::mutex> lock(sched_mutex_);
        if (slot_id >= 0 && slot_id < static_cast<int>(MAX_UNIFORM_SLOTS)) {
            slots_[slot_id].state = st;
            slots_[slot_id].pending_prefill_tokens = pending_prefill;
            if (st == SlotState::IDLE) {
                slots_[slot_id].allocated_pages = 0;
                slots_[slot_id].generated_tokens = 0;
                slots_[slot_id].deficit_tokens = static_cast<int64_t>(cfg_.prefill_quantum_tokens);
            }
        }
    }

private:
    ContinuousSchedulerConfig cfg_;
    size_t total_pages_;
    size_t watermark_low_pages_;
    size_t watermark_crit_pages_;
    size_t watermark_high_pages_;
    size_t rr_cursor_{0};
    std::mutex sched_mutex_;
    SlotScheduleMetrics slots_[MAX_UNIFORM_SLOTS];
};

}
