// include/strata/core/radix_tree.hpp - Lockless Chunk-Hashed RadixTree Prefix Cache & HiCache L2 DMA
//
// Cross-Engine Attribution and Research Lineage:
//   - SGLang: "Efficient Execution of Structured Language Model Programs" (Zheng et al., LMSYS / UC Berkeley, arXiv:2312.07104)
//     * RadixAttention tree-structured prefix caching & automatic prefix cache reuse
//     * Chunk-level 64-token hash fingerprinting via hardware SSE4.2 CRC32C (_mm_crc32_u64)
//     * Hierarchical cache memory (L1 VRAM <-> L2 Pinned Host RAM DMA parking & wire-speed restoration)
//   - vLLM: Automatic Prefix Caching (APC) and physical memory block table management (Kwon et al., SOSP 2023)
//
#pragma once

#include <cuda_runtime.h>
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/radix_compactor.hpp"
#include "strata/core/nvme_tier.hpp"

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <shared_mutex>
#include <mutex>
#include <atomic>
#include <queue>
#include <thread>
#include <condition_variable>

#if defined(__SSE4_2__)
#include <nmmintrin.h>
#endif

namespace strata::core {

static constexpr size_t kRadixChunkTokens = 64;

inline uint64_t compute_token_chunk_hash(const int32_t* tokens, size_t count) {
    uint64_t h1 = 0xFFFFFFFF53545241ULL;
    uint64_t h2 = 0xCBF29CE484222325ULL;
    for (size_t i = 0; i < count; ++i) {
#if defined(__SSE4_2__)
        h1 = _mm_crc32_u64(h1, static_cast<uint32_t>(tokens[i]));
#else
        h1 = (h1 ^ static_cast<uint32_t>(tokens[i])) * 1099511628211ULL;
#endif
        h2 = (h2 * 31) + static_cast<uint32_t>(tokens[i]);
    }
    return h1 ^ ((h2 << 32) | (h2 >> 32));
}

inline uint64_t compute_token_chunk_hash(const int64_t* tokens, size_t count) {
    uint64_t h1 = 0xFFFFFFFF53545241ULL;
    uint64_t h2 = 0xCBF29CE484222325ULL;
    for (size_t i = 0; i < count; ++i) {
#if defined(__SSE4_2__)
        h1 = _mm_crc32_u64(h1, static_cast<uint32_t>(tokens[i]));
#else
        h1 = (h1 ^ static_cast<uint32_t>(tokens[i])) * 1099511628211ULL;
#endif
        h2 = (h2 * 31) + static_cast<uint32_t>(tokens[i]);
    }
    return h1 ^ ((h2 << 32) | (h2 >> 32));
}

struct PinnedBuffer {
    void* ptr_ = nullptr;
    size_t size_ = 0;
    bool is_fallback_ = false;

    PinnedBuffer() = default;
    explicit PinnedBuffer(size_t bytes) { allocate(bytes); }
    ~PinnedBuffer() { free(); }

    void allocate(size_t bytes) {
        if (bytes == size_ && ptr_) return;
        free();
        if (bytes > 0) {
            cudaError_t err = cudaHostAlloc(&ptr_, bytes, cudaHostAllocPortable);
            if (err == cudaSuccess && ptr_) {
                size_ = bytes;
                is_fallback_ = false;
            } else {
                int rc = posix_memalign(&ptr_, 4096, bytes);
                if (rc == 0 && ptr_) {
                    size_ = bytes;
                    is_fallback_ = true;
                } else {
                    ptr_ = nullptr;
                    size_ = 0;
                    is_fallback_ = false;
                }
            }
        }
    }
    void resize(size_t bytes) { allocate(bytes); }
    void free() {
        if (ptr_) {
            if (is_fallback_) {
                std::free(ptr_);
            } else {
                cudaFreeHost(ptr_);
            }
            ptr_ = nullptr;
            size_ = 0;
            is_fallback_ = false;
        }
    }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&& o) noexcept
        : ptr_(o.ptr_), size_(o.size_), is_fallback_(o.is_fallback_) {
        o.ptr_ = nullptr; o.size_ = 0; o.is_fallback_ = false;
    }
    PinnedBuffer& operator=(PinnedBuffer&& o) noexcept {
        if (this != &o) {
            free();
            ptr_ = o.ptr_; size_ = o.size_; is_fallback_ = o.is_fallback_;
            o.ptr_ = nullptr; o.size_ = 0; o.is_fallback_ = false;
        }
        return *this;
    }
    uint8_t* data() noexcept { return static_cast<uint8_t*>(ptr_); }
    const uint8_t* data() const noexcept { return static_cast<const uint8_t*>(ptr_); }
    size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return ptr_ == nullptr || size_ == 0; }
};

struct RadixQsaSlice {
    void* k_q = nullptr;
    void* v_q4 = nullptr;
    void* k_scale = nullptr;
    void* v_scale = nullptr;
    void* k_pool = nullptr;
    void* v_pool = nullptr;
    size_t k_bytes = 0;
    size_t v_bytes = 0;
    size_t k_scale_bytes = 0;
    size_t v_scale_bytes = 0;

    void* idx_tail = nullptr;
    size_t idx_tail_bytes = 0;
    void* idx_dead = nullptr;
    size_t idx_dead_bytes = 0;
    void* idx_pooled = nullptr;
    size_t idx_pooled_bytes = 0;
    int32_t* idx_block_pos = nullptr;
};

struct RadixStageSnapshot {
    int device = -1;
    float* gdn_saved = nullptr;
    size_t gdn_bytes = 0;

    float* ple_saved = nullptr;
    size_t ple_bytes = 0;

    float* R_saved = nullptr;
    size_t R_bytes = 0;

    int32_t ple_prev_saved[2] = {-1, -1};
    int32_t ple_token_saved = -1;

    int64_t qsa_ord0 = 0;
    int64_t qsa_alloc = 0;
    std::vector<RadixQsaSlice> qsa_slices;

    void free_device();
};

// HiCache L2 Tier: Host-RAM parking structures with zero-copy pinned DMA memory
struct RadixQsaHostSlice {
    PinnedBuffer k_q;
    PinnedBuffer v_q4;
    PinnedBuffer k_scale;
    PinnedBuffer v_scale;
    PinnedBuffer idx_tail;
    PinnedBuffer idx_dead;
    PinnedBuffer idx_pooled;
    int32_t idx_block_pos_val = 0;
    bool has_idx_block_pos = false;
};

struct RadixStageHostSnapshot {
    int device = -1;
    PinnedBuffer gdn_data;
    PinnedBuffer ple_data;
    PinnedBuffer R_data;
    int32_t ple_prev_saved[2] = {-1, -1};
    int32_t ple_token_saved = -1;
    int64_t qsa_ord0 = 0;
    int64_t qsa_alloc = 0;
    std::vector<RadixQsaHostSlice> qsa_slices;
};

struct RadixNode : public std::enable_shared_from_this<RadixNode> {
    int64_t id = 0;
    int64_t prefix_len = 0;
    std::vector<int32_t> edge_tokens;
    std::vector<uint64_t> edge_chunk_hashes;
    std::atomic<int> ref_count{0};
    std::chrono::steady_clock::time_point last_accessed;

    std::weak_ptr<RadixNode> parent;
    std::unordered_map<int32_t, std::shared_ptr<RadixNode>> children;

    std::vector<RadixStageSnapshot> stage_snapshots;
    std::vector<RadixStageHostSnapshot> stage_host_snapshots;
    bool is_host_parked = false;
    bool is_l3_offloaded = false;
    std::atomic<bool> is_staging_to_nvme{false};
    size_t l3_bytes = 0;

    void update_chunk_hashes() {
        edge_chunk_hashes.clear();
        const size_t n_chunks = edge_tokens.size() / kRadixChunkTokens;
        edge_chunk_hashes.reserve(n_chunks);
        for (size_t c = 0; c < n_chunks; ++c) {
            edge_chunk_hashes.push_back(
                compute_token_chunk_hash(edge_tokens.data() + c * kRadixChunkTokens, kRadixChunkTokens));
        }
    }

    bool has_device_snapshot() const {
        return !stage_snapshots.empty() && stage_snapshots[0].gdn_saved != nullptr;
    }

    bool has_host_snapshot() const {
        return is_host_parked && !stage_host_snapshots.empty();
    }

    bool has_l3_snapshot() const {
        return is_l3_offloaded;
    }

    bool has_snapshot() const {
        return has_device_snapshot() || has_host_snapshot() || has_l3_snapshot();
    }

    size_t total_vram_bytes() const;
    size_t total_host_bytes() const;
    size_t total_l3_bytes() const { return l3_bytes; }
    void park_to_host();
    /// P18/P20: the park's inverse - restore the device snapshot from the host copy (the L2 hit path).
    bool unpark_to_device();
    bool offload_to_nvme(class NVMeStorageTier& nvme);
    bool hydrate_from_nvme(class NVMeStorageTier& nvme);
    void free_device();
    void free_host();
    void free_nvme(class NVMeStorageTier& nvme);
    ~RadixNode();
};

struct RadixMatch {
    std::shared_ptr<RadixNode> node;
    int64_t matched_tokens = 0;
};

class RadixTree {
public:
    explicit RadixTree(size_t max_cached_snapshots = 4, size_t max_host_snapshots = 64);
    ~RadixTree();

    RadixMatch match_prefix(const int32_t* tokens, size_t n) const;
    RadixMatch match_prefix_overlap(const int32_t* tokens, size_t n, float* out_overlap_ratio = nullptr) const;
    RadixMatch match_prefix(const int64_t* tokens, size_t n) const;
    RadixMatch match_prefix(const std::vector<int32_t>& tokens) const { return match_prefix(tokens.data(), tokens.size()); }
    RadixMatch match_prefix(const std::vector<int64_t>& tokens) const { return match_prefix(tokens.data(), tokens.size()); }

    std::shared_ptr<RadixNode> insert(
        const int32_t* tokens,
        size_t n,
        int64_t prefix_len,
        const std::vector<int>& stage_devices,
        const std::vector<const SessionState*>& states,
        const std::vector<const float*>& R_ptrs,
        const ModelGeometry& g,
        const std::vector<void*>& streams,
        bool safe_no_evict = false);

    std::shared_ptr<RadixNode> insert(
        const std::vector<int32_t>& tokens,
        int64_t prefix_len,
        const std::vector<int>& stage_devices,
        const std::vector<const SessionState*>& states,
        const std::vector<const float*>& R_ptrs,
        const ModelGeometry& g,
        const std::vector<void*>& streams,
        bool safe_no_evict = false) {
        return insert(tokens.data(), tokens.size(), prefix_len, stage_devices, states, R_ptrs, g, streams, safe_no_evict);
    }

    std::shared_ptr<RadixNode> insert_direct_device_to_host(
        const int32_t* tokens,
        size_t n,
        int64_t prefix_len,
        const std::vector<int>& stage_devices,
        const std::vector<const SessionState*>& states,
        const std::vector<const float*>& R_ptrs,
        const ModelGeometry& g,
        const std::vector<void*>& streams,
        bool safe_no_evict = false);

    std::shared_ptr<RadixNode> insert_direct_device_to_host(
        const std::vector<int32_t>& tokens,
        int64_t prefix_len,
        const std::vector<int>& stage_devices,
        const std::vector<const SessionState*>& states,
        const std::vector<const float*>& R_ptrs,
        const ModelGeometry& g,
        const std::vector<void*>& streams,
        bool safe_no_evict = false) {
        return insert_direct_device_to_host(tokens.data(), tokens.size(), prefix_len, stage_devices, states, R_ptrs, g, streams, safe_no_evict);
    }

    void acquire(const std::shared_ptr<RadixNode>& node);
    void release(const std::shared_ptr<RadixNode>& node);

    bool fork_to_session(
        const std::shared_ptr<RadixNode>& node,
        int64_t prefix_len,
        const std::vector<int>& stage_devices,
        std::vector<SessionState*>& child_states,
        const ModelGeometry& g,
        const std::vector<void*>& streams,
        std::string& err);

    size_t evict_lru(size_t max_snapshots, size_t min_free_vram_mib = 0);
    size_t evict_lru_locked(size_t max_snapshots, size_t min_free_vram_mib = 0);

    /// Dynamic KV Fragmentation Compaction (Task 9.3)
        bool persist_manifest(const std::string& model_id_hash = "");
    bool restore_manifest(const std::string& model_id_hash = "");
    CompactionStats compact_tree();

    void set_nvme_tier(std::shared_ptr<class NVMeStorageTier> nvme) { nvme_tier_ = nvme; }
    std::shared_ptr<class NVMeStorageTier> nvme_tier() const { return nvme_tier_; }

    /// S26f: byte budget for the device snapshot tier (deep captures coexist instead of failing silently).
    void set_vram_budget(size_t bytes) { vram_budget_ = bytes; }
    size_t vram_budget() const { return vram_budget_; }
    size_t device_snapshot_bytes() const;

    size_t cached_snapshot_count() const { return cached_snapshots_; }
    size_t cached_host_snapshot_count() const { return cached_host_snapshots_; }
    size_t cached_nvme_snapshot_count() const { return cached_nvme_snapshots_; }
    size_t total_nodes() const { return node_count_; }
    size_t host_snapshot_bytes() const;

private:
    mutable std::shared_mutex rw_lock_;
    std::shared_ptr<RadixNode> root_;
    std::shared_ptr<class NVMeStorageTier> nvme_tier_;
    size_t max_cached_snapshots_ = 4;
    size_t max_host_snapshots_ = 64;
    size_t host_ram_budget_ = (size_t) 32ULL << 30;   // 32 GiB calibrated L2 Host-RAM cache ceiling   // 16 GiB safety ceiling for Host RAM snapshots
    size_t host_snapshot_bytes_locked() const;
    size_t cached_snapshots_ = 0;
    size_t cached_host_snapshots_ = 0;
    size_t cached_nvme_snapshots_ = 0;
    size_t node_count_ = 0;
    int64_t next_node_id_ = 1;
    size_t vram_budget_ = (size_t) 768 << 20;   // S26f: device snapshot byte budget (STRATA_RADIX_VRAM_MIB)
    size_t deep_reserve_ = (size_t) 1280 << 20;   // S26g: tier bytes reserved for deep captures (STRATA_RADIX_DEEP_RESERVE_MIB)
    static constexpr int64_t kDeepPrefixTokens = 12288;   // S26g: prefix at/above which a capture is deep (value-per-byte)

    size_t device_snapshot_bytes_locked() const;
    size_t evict_to_budget(size_t need_bytes);

    void collect_unreferenced_leaves(
        const std::shared_ptr<RadixNode>& curr,
        std::vector<std::shared_ptr<RadixNode>>& leaves);

    void collect_unreferenced_host_nodes(
        const std::shared_ptr<RadixNode>& curr,
        std::vector<std::shared_ptr<RadixNode>>& nodes);

    struct NVMeStagingTask {
        int64_t node_id = 0;
        std::weak_ptr<RadixNode> node_weak;
        AlignedBuffer payload;
        size_t total_payload = 0;
    };

    std::queue<NVMeStagingTask> nvme_queue_;
    std::mutex nvme_queue_mutex_;
    std::condition_variable nvme_cv_;
    std::atomic<bool> nvme_stop_{false};
    std::thread nvme_worker_;

    void nvme_worker_loop();
    void drain_nvme_staging();
};

}  // namespace strata::core
