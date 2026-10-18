// include/strata/core/pinned_slab_pool.hpp - Uniform 64-Token Pinned Memory Host RAM Slab Allocator
#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <vector>

namespace strata::core {

class PinnedSlabPool {
public:
    static constexpr size_t kDefaultAlignment = 256;
    static constexpr size_t kDefaultChunkTokens = 64;

    static constexpr size_t calculate_slab_bytes(size_t bytes_per_token, size_t tokens = kDefaultChunkTokens) noexcept {
        return bytes_per_token * tokens;
    }

    PinnedSlabPool(size_t slab_bytes, int32_t total_slabs, size_t alignment = kDefaultAlignment)
        : slab_bytes_(slab_bytes), total_slabs_(total_slabs), alignment_(alignment) {
        if (slab_bytes_ == 0 || total_slabs_ <= 0) {
            throw std::invalid_argument("PinnedSlabPool: slab_bytes and total_slabs must be positive");
        }
        if (alignment_ < 64 || (alignment_ & (alignment_ - 1)) != 0) {
            alignment_ = kDefaultAlignment;
        }

        stride_bytes_ = (slab_bytes_ + alignment_ - 1) & ~(alignment_ - 1);
        total_bytes_ = stride_bytes_ * static_cast<size_t>(total_slabs_);

        cudaError_t err = cudaHostAlloc(&base_ptr_, total_bytes_, cudaHostAllocPortable);
        if (err == cudaSuccess && base_ptr_ != nullptr) {
            is_fallback_ = false;
        } else {
            int rc = posix_memalign(&base_ptr_, 4096, total_bytes_);
            if (rc == 0 && base_ptr_ != nullptr) {
                is_fallback_ = true;
            } else {
                base_ptr_ = nullptr;
                throw std::bad_alloc();
            }
        }
        if (reinterpret_cast<uintptr_t>(base_ptr_) % alignment_ != 0) {
            if (!is_fallback_) cudaFreeHost(base_ptr_);
            else std::free(base_ptr_);
            base_ptr_ = nullptr;
            throw std::runtime_error("PinnedSlabPool: Base pointer failed alignment constraint");
        }

        free_slabs_.reserve(total_slabs_);
        for (int32_t i = total_slabs_ - 1; i >= 0; --i) {
            free_slabs_.push_back(i);
        }
        in_use_.assign(total_slabs_, false);
    }

    ~PinnedSlabPool() {
        if (base_ptr_ != nullptr) {
            if (is_fallback_) {
                std::free(base_ptr_);
            } else {
                cudaFreeHost(base_ptr_);
            }
            base_ptr_ = nullptr;
        }
    }

    PinnedSlabPool(const PinnedSlabPool&) = delete;
    PinnedSlabPool& operator=(const PinnedSlabPool&) = delete;
    PinnedSlabPool(PinnedSlabPool&&) = delete;
    PinnedSlabPool& operator=(PinnedSlabPool&&) = delete;

    int32_t allocate_slab() noexcept {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        if (free_slabs_.empty()) {
            return -1;
        }

        int32_t slab_id = free_slabs_.back();
        free_slabs_.pop_back();
        in_use_[slab_id] = true;

        int32_t current_alloc = ++allocated_slabs_;
        int32_t peak = peak_slabs_.load(std::memory_order_relaxed);
        while (current_alloc > peak &&
               !peak_slabs_.compare_exchange_weak(peak, current_alloc, std::memory_order_relaxed)) {
        }

        return slab_id;
    }

    bool free_slab(int32_t slab_id) noexcept {
        if (slab_id < 0 || slab_id >= total_slabs_) {
            return false;
        }

        std::lock_guard<std::mutex> lock(pool_mutex_);
        if (!in_use_[slab_id]) {
            return false;
        }

        in_use_[slab_id] = false;
        free_slabs_.push_back(slab_id);
        allocated_slabs_.fetch_sub(1, std::memory_order_relaxed);
        return true;
    }

    void* get_slab_ptr(int32_t slab_id) noexcept {
        if (slab_id < 0 || slab_id >= total_slabs_ || base_ptr_ == nullptr) {
            return nullptr;
        }
        return static_cast<uint8_t*>(base_ptr_) + static_cast<size_t>(slab_id) * stride_bytes_;
    }

    const void* get_slab_ptr(int32_t slab_id) const noexcept {
        if (slab_id < 0 || slab_id >= total_slabs_ || base_ptr_ == nullptr) {
            return nullptr;
        }
        return static_cast<const uint8_t*>(base_ptr_) + static_cast<size_t>(slab_id) * stride_bytes_;
    }

    int32_t allocated_slabs() const noexcept { return allocated_slabs_.load(std::memory_order_relaxed); }
    int32_t peak_slabs() const noexcept { return peak_slabs_.load(std::memory_order_relaxed); }
    int32_t free_slabs_count() const noexcept {
        int32_t alloc = allocated_slabs();
        return total_slabs_ > alloc ? total_slabs_ - alloc : 0;
    }
    int32_t total_slabs() const noexcept { return total_slabs_; }
    size_t slab_bytes() const noexcept { return slab_bytes_; }
    size_t stride_bytes() const noexcept { return stride_bytes_; }
    size_t total_bytes() const noexcept { return total_bytes_; }
    size_t alignment() const noexcept { return alignment_; }
    bool is_fallback() const noexcept { return is_fallback_; }
    void* base_ptr() noexcept { return base_ptr_; }
    const void* base_ptr() const noexcept { return base_ptr_; }

    double utilization_ratio() const noexcept {
        if (total_slabs_ <= 0) return 0.0;
        return static_cast<double>(allocated_slabs()) / static_cast<double>(total_slabs_);
    }

private:
    void* base_ptr_ = nullptr;
    size_t slab_bytes_ = 0;
    size_t stride_bytes_ = 0;
    int32_t total_slabs_ = 0;
    size_t total_bytes_ = 0;
    size_t alignment_ = kDefaultAlignment;
    bool is_fallback_ = false;

    mutable std::mutex pool_mutex_;
    std::vector<int32_t> free_slabs_;
    std::vector<bool> in_use_;

    std::atomic<int32_t> allocated_slabs_{0};
    std::atomic<int32_t> peak_slabs_{0};
};

} // namespace strata::core
