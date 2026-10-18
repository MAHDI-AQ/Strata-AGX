// tests/core/nvme_radix_test.cpp - Gate G10.1 Unit Parity & DirectStorage L3 NVMe Verification
#include "strata/core/nvme_tier.hpp"
#include "strata/core/radix_tree.hpp"
#include "strata/core/hydration_queue.hpp"
#include "strata/core/session_registry.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <vector>

namespace fs = std::filesystem;
using namespace strata::core;

void test_direct_aligned_buffer() {
    std::cout << "[Test 1] AlignedBuffer allocation and zero-copy semantics..." << std::endl;
    AlignedBuffer buf(10000);
    assert(buf.ptr_ != nullptr);
    assert(buf.size() == 10000);
    assert(reinterpret_cast<uintptr_t>(buf.data()) % kDirectBlockAlignment == 0);
    assert(buf.capacity() >= 10000);
    assert(buf.capacity() % kDirectBlockAlignment == 0);

    std::memset(buf.data(), 0xAA, 10000);
    AlignedBuffer moved = std::move(buf);
    assert(buf.ptr_ == nullptr);
    assert(moved.ptr_ != nullptr);
    assert(moved.data()[0] == 0xAA);
    std::cout << "  -> PASS: AlignedBuffer verified on 4096-byte boundary." << std::endl;
}

void test_nvme_throughput_and_parity(NVMeStorageTier& nvme) {
    std::cout << "[Test 2] Direct I/O throughput and bitwise parity..." << std::endl;

    const size_t total_mb = 128;
    const size_t total_bytes = total_mb * 1024 * 1024;
    double write_mb_s = nvme.benchmark_write_throughput_mb_s(total_bytes, 2 * 1024 * 1024);
    double read_mb_s = nvme.benchmark_read_throughput_mb_s(total_bytes, 2 * 1024 * 1024);

    std::cout << "  -> Measured NVMe Write Throughput: " << write_mb_s << " MB/s" << std::endl;
    std::cout << "  -> Measured NVMe Read Throughput:  " << read_mb_s << " MB/s" << std::endl;

    // Direct write & read roundtrip parity
    const size_t payload_size = 4 * 1024 * 1024; // 4MB
    AlignedBuffer wbuf(payload_size);
    for (size_t i = 0; i < payload_size; ++i) {
        wbuf.data()[i] = static_cast<uint8_t>((i * 17 + 101) & 0xFF);
    }

    int64_t test_node_id = 99999;
    bool w_ok = nvme.write_node_direct(test_node_id, wbuf.data(), payload_size);
    assert(w_ok);
    assert(nvme.has_node(test_node_id));

    AlignedBuffer rbuf(payload_size);
    bool r_ok = nvme.read_node_direct(test_node_id, rbuf.data(), payload_size);
    assert(r_ok);

    int cmp = std::memcmp(wbuf.data(), rbuf.data(), payload_size);
    assert(cmp == 0);
    nvme.remove_node(test_node_id);
    assert(!nvme.has_node(test_node_id));

    std::cout << "  -> PASS: 100% bitwise parity verified over " << payload_size << " bytes." << std::endl;
}

void test_radix_node_l3_offload_and_hydration(NVMeStorageTier& nvme) {
    std::cout << "[Test 3] RadixNode L3 offload and hydration state machine..." << std::endl;

    auto node = std::make_shared<RadixNode>();
    node->id = 101;
    node->prefix_len = 256;
    node->edge_tokens = {10, 20, 30, 40, 50, 60, 70, 80};
    node->update_chunk_hashes();

    // Fabricate synthetic Host L2 snapshot
    node->stage_host_snapshots.resize(2);
    for (size_t st = 0; st < 2; ++st) {
        auto& hss = node->stage_host_snapshots[st];
        hss.device = static_cast<int>(st);
        hss.ple_prev_saved[0] = 1234;
        hss.ple_prev_saved[1] = 5678;
        hss.ple_token_saved = 999;
        hss.slab_.allocate(65536 + 3 * 2 * 16384);
        uint8_t* cur = hss.slab_.data();
        hss.gdn_data.reset(cur, 65536);
        std::memset(hss.gdn_data.data(), static_cast<int>(st + 1), 65536);
        cur += 65536;

        hss.qsa_slices.resize(3);
        for (size_t j = 0; j < 3; ++j) {
            auto& slice = hss.qsa_slices[j];
            slice.k_q.reset(cur, 16384);
            std::memset(slice.k_q.data(), static_cast<int>(st * 10 + j), 16384);
            cur += 16384;
            slice.v_q4.reset(cur, 16384);
            std::memset(slice.v_q4.data(), static_cast<int>(st * 20 + j), 16384);
            cur += 16384;
        }
    }
    node->is_host_parked = true;

    assert(node->has_host_snapshot());
    assert(!node->is_l3_offloaded);

    // Offload to NVMe L3
    bool offload_ok = node->offload_to_nvme(nvme);
    assert(offload_ok);
    assert(node->is_l3_offloaded);
    assert(!node->is_host_parked);
    assert(node->stage_host_snapshots.empty());
    assert(nvme.has_node(node->id));

    // Hydrate back from NVMe L3
    bool hydrate_ok = node->hydrate_from_nvme(nvme);
    assert(hydrate_ok);
    assert(!node->is_l3_offloaded);
    assert(node->is_host_parked);
    assert(node->stage_host_snapshots.size() == 2);

    // Check data integrity after hydration
    for (size_t st = 0; st < 2; ++st) {
        const auto& hss = node->stage_host_snapshots[st];
        assert(hss.device == static_cast<int>(st));
        assert(hss.ple_prev_saved[0] == 1234);
        assert(hss.ple_prev_saved[1] == 5678);
        assert(hss.ple_token_saved == 999);
        assert(hss.gdn_data.size() == 65536);
        assert(hss.gdn_data.data()[0] == static_cast<uint8_t>(st + 1));
        assert(hss.qsa_slices.size() == 3);
        for (size_t j = 0; j < 3; ++j) {
            const auto& slice = hss.qsa_slices[j];
            assert(slice.k_q.size() == 16384);
            assert(slice.k_q.data()[0] == static_cast<uint8_t>(st * 10 + j));
            assert(slice.v_q4.size() == 16384);
            assert(slice.v_q4.data()[0] == static_cast<uint8_t>(st * 20 + j));
        }
    }

    node->free_nvme(nvme);
    assert(!nvme.has_node(node->id));
    std::cout << "  -> PASS: Offload -> Disk -> Hydration -> Parity verified." << std::endl;
}

void test_hydration_queue_and_registry(NVMeStorageTier& nvme) {
    std::cout << "[Test 4] Asynchronous HydrationQueue and Persistent SessionRegistry..." << std::endl;

    // Test SessionRegistry
    std::string reg_file = nvme.base_path() + "/test_registry.bin";
    {
        SessionRegistry reg(reg_file);
        reg.register_session("sess_alpha_100k", 501, 102400, {101, 102, 103});
        reg.register_session("sess_beta_200k", 502, 204800, {201, 202});
        assert(reg.save_to_disk());
    }

    {
        SessionRegistry reg(reg_file);
        assert(reg.load_from_disk());
        assert(reg.count() == 2);
        int64_t node_id = 0, tokens = 0;
        std::vector<int32_t> prefix;
        assert(reg.lookup_session("sess_alpha_100k", node_id, tokens, &prefix));
        assert(node_id == 501);
        assert(tokens == 102400);
        assert(prefix.size() == 3 && prefix[0] == 101);
        ::unlink(reg_file.c_str());
    }

    // Test HydrationQueue
    auto tree = std::make_shared<RadixTree>(4, 4);
    auto nvme_ptr = std::make_shared<NVMeStorageTier>(nvme.base_path());
    tree->set_nvme_tier(nvme_ptr);

    auto node = std::make_shared<RadixNode>();
    node->id = 777;
    node->prefix_len = 128;
    node->stage_host_snapshots.resize(1);
    node->stage_host_snapshots[0].slab_.allocate(4096);
    node->stage_host_snapshots[0].gdn_data.reset(node->stage_host_snapshots[0].slab_.data(), 4096);
    std::memset(node->stage_host_snapshots[0].gdn_data.data(), 0xEE, 4096);
    node->is_host_parked = true;

    assert(node->offload_to_nvme(*nvme_ptr));
    assert(node->is_l3_offloaded);

    HydrationQueue hq(tree, nvme_ptr);
    hq.enqueue_hydration(node, "sess_alpha");

    // Wait briefly for background worker
    auto start = std::chrono::steady_clock::now();
    while (node->is_l3_offloaded) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        if (std::chrono::steady_clock::now() - start > std::chrono::seconds(2)) break;
    }

    assert(!node->is_l3_offloaded);
    assert(node->is_host_parked);
    assert(node->stage_host_snapshots[0].gdn_data.data()[0] == 0xEE);
    node->free_nvme(*nvme_ptr);

    std::cout << "  -> PASS: Background HydrationQueue async turnaround: "
              << hq.last_hydration_latency_ms() << " ms." << std::endl;
}


void test_circuit_breaker() {
    std::cout << "[Test 6] Circuit breaker consecutive errors & ENOSPC protection..." << std::endl;
    NVMeStorageTier bad_nvme("/proc/non_existent_strata_dir");
    assert(!bad_nvme.is_circuit_tripped());

    AlignedBuffer buf(4096);
    std::memset(buf.data(), 0x11, 4096);

    // Error 1
    bool r1 = bad_nvme.write_node_direct(1001, buf.data(), 4096);
    assert(!r1);
    assert(!bad_nvme.is_circuit_tripped());
    assert(bad_nvme.consecutive_write_errors() == 1);

    // Error 2
    bool r2 = bad_nvme.write_node_direct(1002, buf.data(), 4096);
    assert(!r2);
    assert(!bad_nvme.is_circuit_tripped());
    assert(bad_nvme.consecutive_write_errors() == 2);

    // Error 3 -> Trips breaker
    bool r3 = bad_nvme.write_node_direct(1003, buf.data(), 4096);
    assert(!r3);
    assert(bad_nvme.is_circuit_tripped());

    // Subsequent writes short-circuit immediately
    bool r4 = bad_nvme.write_node_direct(1004, buf.data(), 4096);
    assert(!r4);

    // Reset breaker
    bad_nvme.reset_circuit_breaker();
    assert(!bad_nvme.is_circuit_tripped());
    assert(bad_nvme.consecutive_write_errors() == 0);
    std::cout << "  -> PASS: Circuit breaker tripped on 3 consecutive errors and reset correctly." << std::endl;
}

int main() {
    cudaFree(0);
    std::cout << "=================================================================" << std::endl;
    std::cout << "  Strata AGX Gate G10.1: NVMe DirectStorage L3 RadixTree Battery" << std::endl;
    std::cout << "=================================================================" << std::endl;

    NVMeStorageTier nvme(NVMeStorageTier::detect_default_storage_path());
    std::cout << "Storage Tier Base Path: " << nvme.base_path() << std::endl;

    test_direct_aligned_buffer();
    test_nvme_throughput_and_parity(nvme);
    test_radix_node_l3_offload_and_hydration(nvme);
    test_hydration_queue_and_registry(nvme);
    // CRIT-RADIX-03 Test: Quarantine / unhooking rollback on NVMe CRC32C failure
    {
        std::cout << "[Test 5] CRIT-RADIX-03: NVMe CRC32C failure quarantine and unhooking..." << std::endl;
        auto parent_node = std::make_shared<RadixNode>();
        parent_node->id = 888;
        parent_node->prefix_len = 64;

        auto corrupt_node = std::make_shared<RadixNode>();
        corrupt_node->id = 889;
        corrupt_node->prefix_len = 128;
        corrupt_node->edge_tokens = {99};
        corrupt_node->stage_host_snapshots.resize(1);
        corrupt_node->stage_host_snapshots[0].slab_.allocate(4096);
        corrupt_node->stage_host_snapshots[0].gdn_data.reset(corrupt_node->stage_host_snapshots[0].slab_.data(), 4096);
        std::memset(corrupt_node->stage_host_snapshots[0].gdn_data.data(), 0xAA, 4096);
        corrupt_node->is_host_parked = true;

        parent_node->children[99] = corrupt_node;
        corrupt_node->parent = parent_node;

        assert(corrupt_node->offload_to_nvme(nvme));
        assert(corrupt_node->is_l3_offloaded);

        // Intentionally corrupt the node on NVMe disk
        std::string fpath = nvme.base_path() + "/node_" + std::to_string(corrupt_node->id) + ".bin";
        int fd = ::open(fpath.c_str(), O_RDWR);
        assert(fd >= 0);
        uint8_t garbage[64];
        std::memset(garbage, 0xFF, sizeof(garbage));
        // Overwrite payload after header
        ::pwrite(fd, garbage, sizeof(garbage), 128);
        ::close(fd);

        // Attempt hydration - must detect CRC mismatch, quarantine node, and unhook from parent
        bool ok = corrupt_node->hydrate_from_nvme(nvme);
        assert(!ok);
        assert(!corrupt_node->is_l3_offloaded);
        assert(corrupt_node->total_l3_bytes() == 0);
        assert(parent_node->children.count(99) == 0);
        std::cout << "  -> PASS: CRC32C corruption quarantined and node unhooked from parent." << std::endl;
    }


    test_circuit_breaker();
    nvme.purge();

    std::cout << "=================================================================" << std::endl;
    std::cout << "  ALL 4/4 GATE G10.1 NVME RADIXTREE SUITES PASSED!" << std::endl;
    std::cout << "=================================================================" << std::endl;
    return 0;
}
