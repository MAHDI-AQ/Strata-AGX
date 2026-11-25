// src/core/boot_weight_crc_cpu_test.cpp - CPU unit test for BootWeightCrc host hash, 2 MiB slicing and clear_holder.
#include "strata/core/boot_weight_crc.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

namespace {

// Direct lane-by-lane reference matching the GPU thread structure word for word.
uint64_t reference_gpu_order_crc(const void* data, uint64_t bytes) {
    uint64_t s[256];
    const uint8_t* p = static_cast<const uint8_t*>(data);
    const uint64_t full_words = bytes / 8;
    const unsigned n_tail = (unsigned) (bytes & 7u);
    for (int tid = 0; tid < 256; ++tid) {
        uint64_t acc = 0x9e3779b97f4a7c15ULL ^ (uint64_t) tid ^ (bytes * 0xbf58476d1ce4e5b9ULL);
        for (uint64_t idx = (uint64_t) tid; idx < full_words; idx += 256) {
            uint64_t w = 0;
            for (int b = 0; b < 8; ++b) w |= ((uint64_t) p[idx * 8 + b]) << (8 * b);
            acc = strata::core::boot_crc_splitmix64_mix(acc ^ w);
        }
        if (tid == 0 && n_tail != 0) {
            uint64_t tw = 0;
            for (unsigned b = 0; b < n_tail; ++b) tw |= ((uint64_t) p[full_words * 8 + b]) << (8u * b);
            acc = strata::core::boot_crc_splitmix64_mix(acc ^ (tw | ((uint64_t) n_tail << 56)));
        }
        s[tid] = acc;
    }
    for (int stride = 128; stride > 0; stride >>= 1) {
        for (int tid = 0; tid < stride; ++tid) {
            s[tid] = strata::core::boot_crc_splitmix64_mix(
                s[tid] ^ strata::core::boot_crc_rotl64(s[tid + stride], 17));
        }
    }
    return s[0];
}

int check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        return 1;
    }
    return 0;
}

}  // namespace

int main() {
    static_assert(std::is_trivially_copyable<strata::core::BootCrcEntry>::value, "BootCrcEntry must remain POD");
    int fails = 0;

    const uint64_t sizes[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 63, 64,
        255, 256, 2040, 2047, 2048, 2049, 4096,
        (2ULL << 20) - 1, (2ULL << 20), (2ULL << 20) + 13
    };

    for (uint64_t sz : sizes) {
        std::vector<uint8_t> aligned_buf((size_t) sz + 64, 0);
        for (size_t i = 0; i < sz; ++i) {
            aligned_buf[i] = static_cast<uint8_t>((i * 131 + 17) ^ (i >> 8));
        }
        const uint8_t* ptr = aligned_buf.data();
        const uint64_t got_aligned = strata::core::boot_crc64_host(ptr, sz);
        const uint64_t ref = reference_gpu_order_crc(ptr, sz);
        fails += check(got_aligned == ref, "aligned host hash must equal GPU lane-order reference");

        // Unaligned pointer test (+1 .. +7)
        for (size_t off = 1; off <= 7; ++off) {
            std::vector<uint8_t> unaligned_buf((size_t) sz + 16, 0xaa);
            if (sz > 0) std::memcpy(unaligned_buf.data() + off, ptr, (size_t) sz);
            const uint64_t got_unaligned = strata::core::boot_crc64_host(unaligned_buf.data() + off, sz);
            fails += check(got_unaligned == got_aligned, "unaligned pointer hash must match aligned hash bitwise");
        }

        // Trailing-zero / length sensitivity
        if (sz + 1 < aligned_buf.size()) {
            const uint64_t plus_zero = strata::core::boot_crc64_host(ptr, sz + 1);
            fails += check(plus_zero != got_aligned, "appending a zero byte must change hash");
        }

        // Single-bit flip sensitivity (head, middle, tail)
        if (sz > 0) {
            const uint64_t probes[] = {0, sz / 2, sz - 1};
            for (uint64_t pos : probes) {
                aligned_buf[(size_t) pos] ^= 0x01;
                const uint64_t flipped = strata::core::boot_crc64_host(ptr, sz);
                aligned_buf[(size_t) pos] ^= 0x01;
                fails += check(flipped != got_aligned, "single-bit flip must change hash");
            }
        }
    }

    // Null / empty safety
    fails += check(strata::core::boot_crc64_host(nullptr, 0) == reference_gpu_order_crc(nullptr, 0),
                   "null+0 bytes hash mismatch");

    // Table slicing + clear_holder checks
    strata::core::BootCrcTable tbl;
    const uint64_t big = (2ULL << 20) + 13;
    std::vector<uint8_t> blob((size_t) big);
    for (size_t i = 0; i < blob.size(); ++i) blob[i] = static_cast<uint8_t>(i * 73 + 5);
    const auto* fake_dev0 = reinterpret_cast<const uint8_t*>(uintptr_t{0x10000000ULL});
    const auto* fake_dev1 = reinterpret_cast<const uint8_t*>(uintptr_t{0x20000000ULL});

    tbl.record(0, "wt", "empty", fake_dev0, 0, blob.data());
    fails += check(tbl.size() == 0, "0-byte record must be ignored");

    tbl.record(0, "wt", "exact_2m", fake_dev0, 2ULL << 20, blob.data());
    fails += check(tbl.size() == 1, "exact 2 MiB must produce 1 slice");

    tbl.record(0, "wt", "big_tensor", fake_dev0 + 0x1000000ULL, big, blob.data());
    fails += check(tbl.size() == 3, "2 MiB + 13 B must slice into 2 sub-regions (total 3)");

    tbl.record(0, "dense", "proj", fake_dev0 + 0x2000000ULL, 1024, blob.data());
    tbl.record(1, "wt", "other_dev", fake_dev1, 1024, blob.data());
    fails += check(tbl.size() == 5, "total entries before clear_holder must be 5");

    const auto snap = tbl.snapshot();
    fails += check(snap[1].bytes == (2ULL << 20) && snap[2].bytes == 13, "sliced byte counts exact");
    fails += check(snap[2].dev_ptr == fake_dev0 + 0x1000000ULL + (2ULL << 20), "slice 2 dev_ptr offset exact");
    fails += check(snap[2].expected == strata::core::boot_crc64_host(blob.data() + (2ULL << 20), 13),
                   "slice 2 hash exact");

    tbl.clear_holder(0, "wt");
    fails += check(tbl.size() == 2, "clear_holder(0, 'wt') must keep (0,'dense') and (1,'wt')");
    const auto after = tbl.snapshot();
    fails += check(after[0].dev == 0 && std::strcmp(after[0].kind, "dense") == 0 &&
                   after[1].dev == 1 && std::strcmp(after[1].kind, "wt") == 0,
                   "remaining entries after clear_holder must be exact");

    if (fails != 0) return 1;
    std::printf("boot_weight_crc_cpu_test PASS (sizes=%zu, slicing+clear_holder ok)\n",
                sizeof(sizes) / sizeof(sizes[0]));
    return 0;
}
