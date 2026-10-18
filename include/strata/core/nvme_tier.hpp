// include/strata/core/nvme_tier.hpp - DirectStorage L3 NVMe Storage Tier for RadixTree
#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <atomic>
#include <unordered_set>

namespace strata::core {

static constexpr size_t kDirectBlockAlignment = 4096;
static constexpr uint64_t kNVMeMagic = 0x5354524154414E56ULL; // "STRATANV"
static constexpr uint32_t kNVMeVersion = 1;

#if defined(__SSE4_2__)
#include <nmmintrin.h>
#endif

inline uint64_t compute_crc32c(const void* data, size_t bytes) {
    uint64_t crc = 0xFFFFFFFF53545241ULL;
    const uint8_t* p = static_cast<const uint8_t*>(data);
    size_t i = 0;
#if defined(__SSE4_2__)
    while (i + 8 <= bytes) {
        uint64_t val;
        std::memcpy(&val, p + i, sizeof(val));
        crc = _mm_crc32_u64(crc, val);
        i += 8;
    }
    while (i < bytes) {
        crc = _mm_crc32_u8(static_cast<uint32_t>(crc), p[i]);
        i += 1;
    }
#else
    for (; i < bytes; ++i) {
        crc = (crc ^ p[i]) * 1099511628211ULL;
    }
#endif
    return crc;
}


/// RAII page-aligned buffer for direct Linux kernel I/O (O_DIRECT requires 4096-byte alignment).
struct AlignedBuffer {
    void* ptr_ = nullptr;
    size_t size_ = 0;
    size_t capacity_ = 0;

    AlignedBuffer() = default;
    explicit AlignedBuffer(size_t bytes) { allocate(bytes); }
    ~AlignedBuffer() { free(); }

    static size_t align_up(size_t bytes, size_t alignment = kDirectBlockAlignment) {
        return (bytes + alignment - 1) & ~(alignment - 1);
    }

    void allocate(size_t bytes) {
        if (bytes == 0) {
            free();
            return;
        }
        size_t aligned_bytes = align_up(bytes);
        if (aligned_bytes <= capacity_ && ptr_) {
            size_ = bytes;
            return;
        }
        free();
        int rc = posix_memalign(&ptr_, kDirectBlockAlignment, aligned_bytes);
        if (rc != 0 || !ptr_) {
            ptr_ = nullptr;
            size_ = 0;
            capacity_ = 0;
            return;
        }
        capacity_ = aligned_bytes;
        size_ = bytes;
        std::memset(ptr_, 0, capacity_);
    }

    void free() {
        if (ptr_) {
            std::free(ptr_);
            ptr_ = nullptr;
            size_ = 0;
            capacity_ = 0;
        }
    }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& o) noexcept
        : ptr_(o.ptr_), size_(o.size_), capacity_(o.capacity_) {
        o.ptr_ = nullptr;
        o.size_ = 0;
        o.capacity_ = 0;
    }

    AlignedBuffer& operator=(AlignedBuffer&& o) noexcept {
        if (this != &o) {
            free();
            ptr_ = o.ptr_;
            size_ = o.size_;
            capacity_ = o.capacity_;
            o.ptr_ = nullptr;
            o.size_ = 0;
            o.capacity_ = 0;
        }
        return *this;
    }

    uint8_t* data() noexcept { return static_cast<uint8_t*>(ptr_); }
    const uint8_t* data() const noexcept { return static_cast<const uint8_t*>(ptr_); }
    size_t size() const noexcept { return size_; }
    size_t capacity() const noexcept { return capacity_; }
    bool empty() const noexcept { return ptr_ == nullptr || size_ == 0; }
};

/// Serialized RadixNode snapshot descriptor stored on NVMe L3.
struct NVMeManifestHeader {
    uint64_t magic = kNVMeMagic;
    uint32_t version = kNVMeVersion;
    uint32_t num_nodes = 0;
    uint64_t max_node_id = 0;
    char model_id_hash[64] = {0};
};

struct NVMeManifestNodeRecord {
    int64_t node_id = 0;
    int64_t parent_id = -1;
    int64_t prefix_len = 0;
    size_t l3_bytes = 0;
    uint32_t n_edge_tokens = 0;
    bool is_l3_offloaded = false;
};

struct NVMeNodeHeader {
    uint64_t magic = kNVMeMagic;
    uint32_t version = kNVMeVersion;
    int64_t node_id = 0;
    int64_t prefix_len = 0;
    uint32_t n_edge_tokens = 0;
    uint32_t n_stages = 0;
    uint64_t total_payload_bytes = 0;
    uint64_t checksum = 0;
};

struct NVMeQsaSliceHeader {
    size_t k_bytes = 0;
    size_t v_bytes = 0;
    size_t k_scale_bytes = 0;
    size_t v_scale_bytes = 0;
    size_t idx_tail_bytes = 0;
    size_t idx_dead_bytes = 0;
    size_t idx_pooled_bytes = 0;
    int32_t idx_block_pos_val = 0;
    uint32_t has_idx_block_pos = 0;
};

struct NVMeStageHeader {
    int32_t device = -1;
    int32_t ple_prev[2] = {-1, -1};
    int32_t ple_token = -1;
    size_t gdn_bytes = 0;
    size_t ple_bytes = 0;
    size_t R_bytes = 0;
    int64_t qsa_ord0 = 0;
    int64_t qsa_alloc = 0;
    uint32_t n_qsa_slices = 0;
};

/// High-performance L3 DirectStorage NVMe engine with Two-Phase Atomic Writes & Crash Recovery.
class NVMeStorageTier {
public:
    explicit NVMeStorageTier(const std::string& base_path = "");
    ~NVMeStorageTier();

    static constexpr uint32_t kMaxConsecutiveWriteErrors = 3;
    bool is_circuit_tripped() const { return circuit_tripped_.load(std::memory_order_relaxed); }
    void reset_circuit_breaker() {
        circuit_tripped_.store(false, std::memory_order_release);
        consecutive_write_errors_.store(0, std::memory_order_relaxed);
    }
    uint32_t consecutive_write_errors() const { return consecutive_write_errors_.load(std::memory_order_relaxed); }

    bool init();
    void shutdown();

    /// Offloads serialized data of a RadixNode to NVMe L3 with two-phase atomic write (.tmp -> fdatasync -> rename).
    bool write_node_direct(int64_t node_id, const void* data, size_t bytes);

    /// Reads serialized data of a RadixNode from NVMe L3 with Direct I/O.
    bool read_node_direct(int64_t node_id, void* out_data, size_t bytes);

    /// Checks if a node exists on NVMe L3.
    bool has_node(int64_t node_id) const;

    /// Removes an offloaded node from NVMe L3.
    bool remove_node(int64_t node_id);

    /// Clears all L3 files in storage directory.
    void purge();

    /// Sweeps leftover temporary files from aborted/crashed processes.
    size_t clean_crash_temporaries();

    /// Purges orphaned node_*.bin files not indexed in the active tree.
    size_t clean_orphans(const std::unordered_set<int64_t>& valid_node_ids);

    /// Binary manifest persistence.
    bool write_manifest(const std::string& model_sha256, const std::vector<NVMeManifestNodeRecord>& records);
    bool load_manifest(std::string& out_model_sha256, std::vector<NVMeManifestNodeRecord>& out_records);

    /// Benchmark tools for Gate G10.1 verification.
    double benchmark_write_throughput_mb_s(size_t total_bytes, size_t chunk_bytes = 1048576);
    double benchmark_read_throughput_mb_s(size_t total_bytes, size_t chunk_bytes = 1048576);

    const std::string& base_path() const { return base_path_; }
    size_t total_offloaded_nodes() const { return offloaded_nodes_.load(std::memory_order_relaxed); }
    uint64_t total_bytes_written() const { return bytes_written_.load(std::memory_order_relaxed); }
    uint64_t total_bytes_read() const { return bytes_read_.load(std::memory_order_relaxed); }

    static std::string detect_default_storage_path();

private:
    std::string base_path_;
    bool direct_io_supported_ = true;
    mutable std::shared_mutex mutex_;
    std::atomic<size_t> offloaded_nodes_{0};
    std::atomic<uint64_t> bytes_written_{0};
    std::atomic<uint64_t> bytes_read_{0};
    std::atomic<bool> circuit_tripped_{false};
    std::atomic<uint32_t> consecutive_write_errors_{0};

    std::string node_file_path(int64_t node_id) const;
    std::string tmp_node_file_path(int64_t node_id) const;
    std::string manifest_file_path() const;
    std::string tmp_manifest_file_path() const;
};

}  // namespace strata::core
