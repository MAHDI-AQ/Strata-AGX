#include "strata/core/pinned_slab_pool.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>
#include <set>

using strata::core::PinnedSlabPool;

void test_initialization_and_alignment() {
    std::cout << "[Test 1] Initialization, Memory Backing & Pointer Alignment..." << std::endl;
    const size_t slab_bytes = 64 * 128;
    const int32_t total_slabs = 16;
    PinnedSlabPool pool(slab_bytes, total_slabs, 256);

    assert(pool.total_slabs() == total_slabs);
    assert(pool.slab_bytes() == slab_bytes);
    assert(pool.stride_bytes() >= slab_bytes);
    assert(pool.stride_bytes() % 256 == 0);
    assert(pool.allocated_slabs() == 0);
    assert(pool.peak_slabs() == 0);
    assert(pool.utilization_ratio() == 0.0);
    assert(pool.base_ptr() != nullptr);

    uintptr_t base_addr = reinterpret_cast<uintptr_t>(pool.base_ptr());
    assert(base_addr % 256 == 0);

    for (int32_t i = 0; i < total_slabs; ++i) {
        void* ptr = pool.get_slab_ptr(i);
        assert(ptr != nullptr);
        uintptr_t slab_addr = reinterpret_cast<uintptr_t>(ptr);
        assert(slab_addr % 256 == 0);
        assert(slab_addr == base_addr + static_cast<size_t>(i) * pool.stride_bytes());
    }

    assert(pool.get_slab_ptr(-1) == nullptr);
    assert(pool.get_slab_ptr(total_slabs) == nullptr);
    std::cout << "  -> PASS (Allocated monolithic " << pool.total_bytes()
              << " bytes, fallback=" << (pool.is_fallback() ? "true" : "false") << ")" << std::endl;
}

void test_allocation_exhaustion_and_reuse() {
    std::cout << "[Test 2] Allocation, Exhaustion & LIFO Reuse..." << std::endl;
    const size_t slab_bytes = 1024;
    const int32_t total_slabs = 8;
    PinnedSlabPool pool(slab_bytes, total_slabs);

    std::vector<int32_t> ids;
    for (int32_t i = 0; i < total_slabs; ++i) {
        int32_t id = pool.allocate_slab();
        assert(id >= 0 && id < total_slabs);
        ids.push_back(id);
    }
    assert(pool.allocated_slabs() == total_slabs);
    assert(pool.peak_slabs() == total_slabs);
    assert(pool.utilization_ratio() == 1.0);

    int32_t fail_id = pool.allocate_slab();
    assert(fail_id == -1);
    assert(pool.allocated_slabs() == total_slabs);

    int32_t freed_id = ids.back();
    assert(pool.free_slab(freed_id));
    assert(pool.allocated_slabs() == total_slabs - 1);
    assert(pool.utilization_ratio() < 1.0);

    assert(!pool.free_slab(freed_id));
    assert(!pool.free_slab(-1));
    assert(!pool.free_slab(total_slabs));

    int32_t reallocated_id = pool.allocate_slab();
    assert(reallocated_id == freed_id);
    assert(pool.allocated_slabs() == total_slabs);

    for (int32_t id : ids) {
        assert(pool.free_slab(id));
    }
    assert(pool.allocated_slabs() == 0);
    assert(pool.utilization_ratio() == 0.0);
    assert(pool.peak_slabs() == total_slabs);
    std::cout << "  -> PASS" << std::endl;
}

void test_concurrent_allocation_and_memory_safety() {
    std::cout << "[Test 3] Multi-Threaded Concurrent Allocation & Memory Isolation..." << std::endl;
    const size_t slab_bytes = 4096;
    const int32_t total_slabs = 128;
    PinnedSlabPool pool(slab_bytes, total_slabs);

    const int num_threads = 8;
    const int ops_per_thread = 500;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&pool, t, ops_per_thread, slab_bytes]() {
            for (int i = 0; i < ops_per_thread; ++i) {
                int32_t id = pool.allocate_slab();
                if (id >= 0) {
                    uint8_t* ptr = static_cast<uint8_t*>(pool.get_slab_ptr(id));
                    assert(ptr != nullptr);
                    uint8_t pattern = static_cast<uint8_t>((t + i) & 0xFF);
                    std::memset(ptr, pattern, slab_bytes);

                    for (size_t b = 0; b < 64; ++b) {
                        assert(ptr[b] == pattern);
                    }

                    bool ok = pool.free_slab(id);
                    assert(ok);
                }
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    assert(pool.allocated_slabs() == 0);
    assert(pool.utilization_ratio() == 0.0);
    assert(pool.peak_slabs() > 0);
    assert(pool.peak_slabs() <= total_slabs);
    std::cout << "  -> PASS (All " << (num_threads * ops_per_thread)
              << " concurrent ops passed; peak_slabs=" << pool.peak_slabs() << ")" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Running Strata-AGX PinnedSlabPool Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        test_initialization_and_alignment();
        test_allocation_exhaustion_and_reuse();
        test_concurrent_allocation_and_memory_safety();
    } catch (const std::exception& ex) {
        std::cerr << "FAIL: Unhandled exception: " << ex.what() << std::endl;
        return 1;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "ALL PinnedSlabPool TESTS PASSED (100%)" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
