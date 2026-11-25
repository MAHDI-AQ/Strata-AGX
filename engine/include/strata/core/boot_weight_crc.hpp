#pragma once
// Phase B0 (opt-in STRATA_BOOT_WEIGHT_CRC=1): deterministic 256-lane SplitMix64 VRAM weight integrity verification at boot.
// HIP-portable, Metal-safe (no-op stubs), zero CUDA headers required for pure host/CPU unit tests.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace strata::core {

inline constexpr uint64_t kBootCrcChunkBytes = 2ULL << 20;   // 2 MiB slice cap

struct BootCrcEntry {
    const void* dev_ptr;
    uint64_t bytes;
    uint64_t expected;
    int dev;
    char kind[16];
    char tag[48];
};

inline uint64_t boot_crc_rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

inline uint64_t boot_crc_splitmix64_mix(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Cache-conscious CPU equivalent of the GPU 256-lane SplitMix64 kernel: linear streaming in 256-word tiles
// updates the exact same per-lane accumulators in the exact same order, bitwise identical for any alignment.
inline uint64_t boot_crc64_host(const void* data, uint64_t bytes) {
    uint64_t s[256];
    for (int tid = 0; tid < 256; ++tid) {
        s[tid] = 0x9e3779b97f4a7c15ULL ^ (uint64_t) tid ^ (bytes * 0xbf58476d1ce4e5b9ULL);
    }
    const uint8_t* p = static_cast<const uint8_t*>(data);
    const uint64_t full_words = bytes / 8;
    const bool aligned = (reinterpret_cast<std::uintptr_t>(p) & 7u) == 0;
    if (aligned) {
        const uint64_t* pw = reinterpret_cast<const uint64_t*>(p);
        uint64_t base = 0;
        for (; base + 256 <= full_words; base += 256) {
            for (int tid = 0; tid < 256; ++tid) {
                s[tid] = boot_crc_splitmix64_mix(s[tid] ^ pw[base + (uint64_t) tid]);
            }
        }
        for (uint64_t rem = 0; base + rem < full_words; ++rem) {
            s[rem] = boot_crc_splitmix64_mix(s[rem] ^ pw[base + rem]);
        }
    } else {
        uint64_t base = 0;
        for (; base + 256 <= full_words; base += 256) {
            for (int tid = 0; tid < 256; ++tid) {
                const uint8_t* bp = p + (base + (uint64_t) tid) * 8;
                uint64_t w = 0;
                for (int b = 0; b < 8; ++b) w |= ((uint64_t) bp[b]) << (8 * b);
                s[tid] = boot_crc_splitmix64_mix(s[tid] ^ w);
            }
        }
        for (uint64_t rem = 0; base + rem < full_words; ++rem) {
            const uint8_t* bp = p + (base + rem) * 8;
            uint64_t w = 0;
            for (int b = 0; b < 8; ++b) w |= ((uint64_t) bp[b]) << (8 * b);
            s[rem] = boot_crc_splitmix64_mix(s[rem] ^ w);
        }
    }
    const unsigned n_tail = (unsigned) (bytes & 7u);
    if (n_tail != 0) {
        const uint8_t* tp = p + full_words * 8;
        uint64_t tail_word = 0;
        for (unsigned b = 0; b < n_tail; ++b) {
            tail_word |= ((uint64_t) tp[b]) << (8u * b);
        }
        s[0] = boot_crc_splitmix64_mix(s[0] ^ (tail_word | ((uint64_t) n_tail << 56)));
    }
    for (int stride = 128; stride > 0; stride >>= 1) {
        for (int tid = 0; tid < stride; ++tid) {
            s[tid] = boot_crc_splitmix64_mix(s[tid] ^ boot_crc_rotl64(s[tid + stride], 17));
        }
    }
    return s[0];
}

class BootCrcTable {
public:
    void record(int dev, const char* kind, const char* tag, const void* dev_ptr, uint64_t bytes, const void* host_ptr) {
        if (bytes == 0 || dev_ptr == nullptr || host_ptr == nullptr) return;
        const char* kstr = kind ? kind : "";
        const char* tstr = tag ? tag : "";
        const uint8_t* dp = static_cast<const uint8_t*>(dev_ptr);
        const uint8_t* hp = static_cast<const uint8_t*>(host_ptr);

        if (bytes <= kBootCrcChunkBytes) {
            BootCrcEntry e{};
            e.dev_ptr = dp;
            e.bytes = bytes;
            e.expected = boot_crc64_host(hp, bytes);
            e.dev = dev;
            std::snprintf(e.kind, sizeof(e.kind), "%s", kstr);
            std::snprintf(e.tag, sizeof(e.tag), "%s", tstr);
            std::lock_guard<std::mutex> lock(mu_);
            entries_.push_back(e);
            return;
        }

        const size_t n_slices = (size_t) ((bytes + kBootCrcChunkBytes - 1) / kBootCrcChunkBytes);
        std::vector<BootCrcEntry> batch;
        batch.reserve(n_slices);
        for (size_t sidx = 0; sidx < n_slices; ++sidx) {
            const uint64_t off = (uint64_t) sidx * kBootCrcChunkBytes;
            const uint64_t sub = std::min<uint64_t>(kBootCrcChunkBytes, bytes - off);
            BootCrcEntry e{};
            e.dev_ptr = dp + off;
            e.bytes = sub;
            e.expected = boot_crc64_host(hp + off, sub);
            e.dev = dev;
            std::snprintf(e.kind, sizeof(e.kind), "%s", kstr);
            std::snprintf(e.tag, sizeof(e.tag), "%.35s#%zu", tstr, sidx);
            batch.push_back(e);
        }
        std::lock_guard<std::mutex> lock(mu_);
        entries_.insert(entries_.end(), batch.begin(), batch.end());
    }

    void clear_holder(int dev, const char* kind) {
        const char* kstr = kind ? kind : "";
        std::lock_guard<std::mutex> lock(mu_);
        entries_.erase(
            std::remove_if(entries_.begin(), entries_.end(),
                           [&](const BootCrcEntry& e) {
                               return (dev < 0 || e.dev == dev) &&
                                      std::strncmp(e.kind, kstr, sizeof(e.kind)) == 0;
                           }),
            entries_.end());
    }

    void clear_all() {
        std::lock_guard<std::mutex> lock(mu_);
        entries_.clear();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mu_);
        return entries_.size();
    }

    std::vector<BootCrcEntry> snapshot() const {
        std::lock_guard<std::mutex> lock(mu_);
        return entries_;
    }

private:
    mutable std::mutex mu_;
    std::vector<BootCrcEntry> entries_;
};

#if defined(STRATA_GPU_METAL)
inline bool boot_weight_crc_enabled() { return false; }
inline void boot_crc_record(int, const char*, const char*, const void*, uint64_t, const void*) {}
inline void boot_crc_clear_holder(int, const char*) {}
inline void boot_crc_clear_all() {}
inline size_t boot_crc_region_count() { return 0; }
inline bool boot_weight_crc_verify_all(std::string&) { return true; }
#else
bool boot_weight_crc_enabled();
void boot_crc_record(int dev, const char* kind, const char* tag, const void* dev_ptr, uint64_t bytes, const void* host_ptr);
void boot_crc_clear_holder(int dev, const char* kind);
void boot_crc_clear_all();
size_t boot_crc_region_count();
bool boot_weight_crc_verify_all(std::string& err);
#endif

}  // namespace strata::core
