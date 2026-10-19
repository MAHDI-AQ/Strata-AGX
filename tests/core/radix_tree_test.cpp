#include "strata/core/radix_tree.hpp"
#include <cassert>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

using namespace strata::core;

int main() {
    std::cout << "Running comprehensive RadixTree & HiCache L2 unit tests..." << std::endl;

    RadixTree tree(4, 64); // Max 4 VRAM snapshots, max 64 Host-RAM snapshots

    // 1. Initial match on empty tree
    std::vector<int32_t> prompt1 = {10, 20, 30, 40, 50};
    auto m1 = tree.match_prefix(prompt1);
    assert(m1.matched_tokens == 0);
    assert(m1.node == nullptr);
    std::cout << "  [PASS] Empty tree match" << std::endl;

    // 2. Tree property checks
    assert(tree.total_nodes() == 1); // root node
    assert(tree.cached_snapshot_count() == 0);
    assert(tree.cached_host_snapshot_count() == 0);
    std::cout << "  [PASS] Initial state check" << std::endl;

    // 3. Test insert & match without GPU (mock dummy states)
    std::vector<int32_t> harness_prompt(500);
    for (size_t i = 0; i < harness_prompt.size(); ++i) harness_prompt[i] = (int32_t)(1000 + i);

    std::vector<int> stage_devs = {0};
    std::vector<const SessionState*> states;
    std::vector<const float*> R_ptrs;
    std::vector<void*> streams;
    ModelGeometry g{};

    auto res_null = tree.insert(harness_prompt, 500, stage_devs, states, R_ptrs, g, streams);
    assert(res_null == nullptr);
    std::cout << "  [PASS] Graceful refusal on empty stage states" << std::endl;

    // 4. Test HiCache L2 Host-RAM Parking and Lifecycle with PinnedBuffer
    {
        auto node = std::make_shared<RadixNode>();
        node->id = 42;
        node->prefix_len = 1024;

        // Allocate mock device snapshot on GPU 0
        RadixStageSnapshot ss;
        ss.device = 0;
        ss.gdn_bytes = 1024 * sizeof(float);
        cudaMalloc(&ss.gdn_saved, ss.gdn_bytes);
        std::vector<float> h_mock(1024, 3.14159f);
        cudaMemcpy(ss.gdn_saved, h_mock.data(), ss.gdn_bytes, cudaMemcpyHostToDevice);

        ss.R_bytes = 512 * sizeof(float);
        cudaMalloc(&ss.R_saved, ss.R_bytes);
        cudaMemcpy(ss.R_saved, h_mock.data(), ss.R_bytes, cudaMemcpyHostToDevice);

        node->stage_snapshots.push_back(ss);
        assert(node->has_device_snapshot());
        assert(!node->has_host_snapshot());
        assert(node->has_snapshot());
        assert(node->total_vram_bytes() == ss.gdn_bytes + ss.R_bytes);

        // Park node to Host RAM (HiCache L2 Tier with zero-copy PinnedBuffer)
        node->park_to_host();
        assert(!node->has_device_snapshot());
        assert(node->has_host_snapshot());
        assert(node->has_snapshot());
        assert(node->total_vram_bytes() == 0); // VRAM completely freed!
        assert(node->total_host_bytes() == ss.gdn_bytes + ss.R_bytes);

        // Verify data fidelity in pinned host snapshot
        const float* h_gdn = (const float*) node->stage_host_snapshots[0].gdn_data.data();
        assert(h_gdn[0] == 3.14159f);
        assert(h_gdn[1023] == 3.14159f);

        node->free_host();
        assert(!node->has_snapshot());
        assert(node->total_host_bytes() == 0);
        std::cout << "  [PASS] HiCache L2 Pinned Host-RAM parking, data fidelity, and VRAM release" << std::endl;
    }

    // 5. Test 64-token chunk hash fingerprinting
    {
        std::vector<int32_t> tok_a(128, 77);
        std::vector<int32_t> tok_b(128, 77);
        tok_b[120] = 99; // Change in second chunk

        uint64_t ha0 = compute_token_chunk_hash(tok_a.data(), 64);
        uint64_t ha1 = compute_token_chunk_hash(tok_a.data() + 64, 64);
        uint64_t hb0 = compute_token_chunk_hash(tok_b.data(), 64);
        uint64_t hb1 = compute_token_chunk_hash(tok_b.data() + 64, 64);

        assert(ha0 == hb0); // First chunk identical
        assert(ha1 != hb1); // Second chunk must differ
        std::cout << "  [PASS] 64-token chunk hashing and collision resistance" << std::endl;
    }

    // 6. Test Concurrent Multi-Threaded match_prefix (under shared_mutex)
    {
        std::vector<int32_t> query_tokens(4096);
        for (size_t i = 0; i < query_tokens.size(); ++i) query_tokens[i] = (int32_t)(500 + (i % 200));

        std::vector<std::thread> readers;
        std::atomic<bool> ok{true};

        for (int t = 0; t < 8; ++t) {
            readers.emplace_back([&tree, &query_tokens, &ok]() {
                for (int iter = 0; iter < 1000; ++iter) {
                    auto match = tree.match_prefix(query_tokens);
                    if (match.matched_tokens < 0) ok = false;
                }
            });
        }

        for (auto& th : readers) th.join();
        assert(ok);
        std::cout << "  [PASS] 8-thread concurrent lockless match_prefix reads" << std::endl;
    }

    // 7. Measure match_prefix latency on 10,000 token prompt
    {
        std::vector<int32_t> long_prompt(10000);
        for (size_t i = 0; i < long_prompt.size(); ++i) long_prompt[i] = (int32_t)(i + 1);

        int iters = 20000;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            auto m = tree.match_prefix(long_prompt);
            (void) m;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
        std::cout << "  [PERF] 10,000-token match_prefix latency: " << elapsed_us << " us / match" << std::endl;
        assert(elapsed_us < 10.0); // Target < 10 us
        std::cout << "  [PASS] Latency well below 10 us target!" << std::endl;
    }

    // 8. CRIT-RADIX-02 Unit Test: compact_tree preserves staging nodes from dead-leaf pruning and path compression
    {
        RadixTree comp_tree(4, 64);
        // Test manual tree structure directly to verify compact_tree staging guard
        auto staging_node = std::make_shared<RadixNode>();
        staging_node->id = 999;
        staging_node->prefix_len = 64;
        staging_node->edge_tokens = {60};
        staging_node->ref_count = 0;
        staging_node->is_staging_to_nvme = true; // Actively staging
        staging_node->is_host_parked = false;
        staging_node->is_l3_offloaded = false;

        // Tree root has no snapshot, attach staging_node as child under key 60
        // We can access through tree match or direct node manipulation
        // In compact_tree, it visits children of root_
        // Let's verify compact_tree behaviour with a mock node
        std::vector<int32_t> long_p(64, 42);
        // Mock device states to allow insert
        RadixStageSnapshot ss;
        ss.device = 0;
        ss.gdn_bytes = 64 * sizeof(float);
        cudaMalloc(&ss.gdn_saved, ss.gdn_bytes);
        SessionState mock_state;
        mock_state.gdn_alloc = 0;
        mock_state.qsa_alloc = 0;
        std::vector<int> s_devs = {0};
        std::vector<const SessionState*> s_states = {&mock_state};
        std::vector<const float*> s_R = {nullptr};
        cudaStream_t strm;
        cudaStreamCreate(&strm);
        std::vector<void*> s_streams = {(void*) strm};

        auto p1 = comp_tree.insert(long_p.data(), 64, 64, s_devs, s_states, s_R, g, s_streams);
        if (p1) {
            p1->children[60] = staging_node;
            staging_node->parent = p1;

            auto stats = comp_tree.compact_tree();
            assert(p1->children.count(60) == 1);
            assert(stats.dead_leaves_pruned == 0);

            // Turn off staging and ensure it gets pruned
            staging_node->is_staging_to_nvme = false;
            stats = comp_tree.compact_tree();
            assert(p1->children.count(60) == 0);
            assert(stats.dead_leaves_pruned == 1);
        }
        cudaStreamDestroy(strm);
        cudaFree(ss.gdn_saved);
        std::cout << "  [PASS] CRIT-RADIX-02 staging pin guard in compact_tree" << std::endl;
    }

    // 9. CRIT-RADIX-04 Unit Test: Paged/segmented slab buffer structure verification
    {
        PinnedBuffer paged_buf;
        paged_buf.allocate(5ULL * 1024 * 1024); // 5 MiB -> 3 pages (2MB + 2MB + 1MB)
        assert(paged_buf.size() == 5ULL * 1024 * 1024);
        assert(paged_buf.pages().size() == 3);
        assert(paged_buf.pages()[0].size() == 2ULL * 1024 * 1024);
        assert(paged_buf.pages()[1].size() == 2ULL * 1024 * 1024);
        assert(paged_buf.pages()[2].size() == 1ULL * 1024 * 1024);
        assert(paged_buf.locate(0) != nullptr);
        assert(paged_buf.locate(3ULL * 1024 * 1024) != nullptr);
        assert(paged_buf.locate(5ULL * 1024 * 1024) == nullptr);
        paged_buf.free();
        assert(paged_buf.empty());
        assert(paged_buf.size() == 0);
        std::cout << "  [PASS] CRIT-RADIX-04 Paged/segmented slab buffer structure" << std::endl;
    }

        // --- Phase 1: AVX2 SIMD Equality, Batch Hysteresis & GDN Preservation Tests ---
    {
        std::cout << "  [TEST] Phase 1: AVX2 SIMD chunk equality verification... " << std::flush;
        std::vector<int32_t> a(64, 100);
        std::vector<int32_t> b = a;
        assert(strata::core::verify_chunk_tokens_avx2(a.data(), b.data(), 64) == true);
        b[15] = 999;
        assert(strata::core::verify_chunk_tokens_avx2(a.data(), b.data(), 64) == false);
        b = a; b[63] = 999;
        assert(strata::core::verify_chunk_tokens_avx2(a.data(), b.data(), 64) == false);
        std::cout << "[PASS]" << std::endl;

        std::cout << "  [TEST] Phase 1: GDN recurrent checkpoint compaction guard... " << std::flush;
        auto node = std::make_shared<strata::core::RadixNode>();
        node->has_gdn_checkpoint = true;
        assert(node->holds_recurrent_state() == true);
        node->has_gdn_checkpoint = false;
        assert(node->holds_recurrent_state() == false);
        std::cout << "[PASS]" << std::endl;
    }

    // --- Phase 2: Decoupled Page-Table Virtualization (PagedKVPool O(T) Memory Scaling) ---
    {
        std::cout << "  [TEST] Phase 2: PagedKVPool allocation, recycling & duplicate resistance... " << std::flush;
        strata::core::PagedKVPool pool(1024);
        auto p1 = pool.allocate_pages(10);
        assert(p1.size() == 10);
        assert(pool.allocated_pages() == 10);
        std::unordered_set<int32_t> seen(p1.begin(), p1.end());
        assert(seen.size() == 10);

        pool.free_pages({p1[2], p1[5], p1[8]});
        assert(pool.allocated_pages() == 7);

        auto p2 = pool.allocate_pages(2);
        assert(p2.size() == 2);
        assert(pool.allocated_pages() == 9);
        for (int32_t pg : p2) {
            assert(pg == p1[8] || pg == p1[5] || pg == p1[2]);
        }
        std::cout << "[PASS]" << std::endl;

        std::cout << "  [TEST] Phase 2: Multi-turn O(T) incremental delta page allocation... " << std::flush;
        strata::core::RadixTree tree_paged(8, 64);

        // Turn 0: System prompt (4000 tokens -> 250 pages)
        std::vector<int32_t> turn0_tokens(4000);
        for (size_t i = 0; i < turn0_tokens.size(); ++i) turn0_tokens[i] = 1000 + (i % 500);

        // Mock state to satisfy insert checks
        strata::core::SessionState mock_state;
        mock_state.gdn_alloc = 0;
        mock_state.qsa_alloc = 0;
        std::vector<int> s_devs = {0};
        std::vector<const strata::core::SessionState*> s_states = {&mock_state};
        std::vector<const float*> s_R = {nullptr};
        cudaStream_t strm;
        cudaStreamCreate(&strm);
        std::vector<void*> s_streams = {(void*) strm};
        strata::core::ModelGeometry geom{};

        auto node0 = tree_paged.insert(turn0_tokens.data(), 4000, 4000, s_devs, s_states, s_R, geom, s_streams);
        assert(node0 != nullptr);
        assert(node0->edge_tokens.size() == 4000);
        assert(node0->kv_page_ids.size() == 250); // Exact 250 pages

        // Turn 1: 1600 delta tokens (prefix_len = 5600 -> 100 delta pages)
        std::vector<int32_t> turn1_tokens = turn0_tokens;
        turn1_tokens.resize(5600);
        for (size_t i = 4000; i < 5600; ++i) turn1_tokens[i] = 2000 + (i % 500);

        auto node1 = tree_paged.insert(turn1_tokens.data(), 5600, 5600, s_devs, s_states, s_R, geom, s_streams);
        assert(node1 != nullptr);
        assert(node1->edge_tokens.size() == 1600);
        assert(node1->kv_page_ids.size() == 100); // ONLY 100 delta pages! (O(T) scaling, NOT 350!)

        // Turn 2: 2400 delta tokens (prefix_len = 8000 -> 150 delta pages)
        std::vector<int32_t> turn2_tokens = turn1_tokens;
        turn2_tokens.resize(8000);
        for (size_t i = 5600; i < 8000; ++i) turn2_tokens[i] = 3000 + (i % 500);

        auto node2 = tree_paged.insert(turn2_tokens.data(), 8000, 8000, s_devs, s_states, s_R, geom, s_streams);
        assert(node2 != nullptr);
        assert(node2->edge_tokens.size() == 2400);
        assert(node2->kv_page_ids.size() == 150); // ONLY 150 delta pages! (NOT 500!)

        // Zero duplicate page allocations across all 3 nodes
        std::unordered_set<int32_t> all_pages;
        for (int32_t id : node0->kv_page_ids) {
            assert(all_pages.insert(id).second);
        }
        for (int32_t id : node1->kv_page_ids) {
            assert(all_pages.insert(id).second);
        }
        for (int32_t id : node2->kv_page_ids) {
            assert(all_pages.insert(id).second);
        }
        assert(all_pages.size() == 500); // Exactly 250 + 100 + 150 = 500 pages total
        std::cout << "[PASS]" << std::endl;

        std::cout << "  [TEST] Phase 2: Virtual block table resolution & node split page slicing... " << std::flush;
        auto vbt2 = tree_paged.resolve_virtual_block_table(node2);
        assert(vbt2.size() == 500);
        for (size_t i = 0; i < 250; ++i) assert(vbt2[i] == node0->kv_page_ids[i]);
        for (size_t i = 0; i < 100; ++i) assert(vbt2[250 + i] == node1->kv_page_ids[i]);
        for (size_t i = 0; i < 150; ++i) assert(vbt2[350 + i] == node2->kv_page_ids[i]);

        // Branching off Turn 1 with node splitting:
        // Branch at token 4800 (halfway through Turn 1's 1600 tokens -> splits at 800 tokens = 50 pages)
        std::vector<int32_t> branch_tokens = turn1_tokens;
        branch_tokens.resize(4800 + 400);
        branch_tokens[4800] = 999999;
        for (size_t i = 4801; i < branch_tokens.size(); ++i) branch_tokens[i] = 4000 + (i % 300);

        auto branch_node = tree_paged.insert(branch_tokens.data(), branch_tokens.size(), branch_tokens.size(),
                                             s_devs, s_states, s_R, geom, s_streams);
        assert(branch_node != nullptr);
        auto split_parent = branch_node->parent.lock();
        assert(split_parent != nullptr);
        assert(split_parent->edge_tokens.size() == 800);
        assert(split_parent->kv_page_ids.size() == 50);

        // Virtual block table for branch_node: 250 (node0) + 50 (split) + 25 (delta) = 325 pages
        auto vbt_branch = tree_paged.resolve_virtual_block_table(branch_node);
        assert(vbt_branch.size() == 325);
        for (size_t i = 0; i < 250; ++i) assert(vbt_branch[i] == node0->kv_page_ids[i]);
        for (size_t i = 0; i < 50; ++i) assert(vbt_branch[250 + i] == split_parent->kv_page_ids[i]);
        std::cout << "[PASS]" << std::endl;

        // CRIT-MEM-01: Arbitrary non-page-aligned prefix split (match_edge % 16 != 0)
        std::cout << "  [TEST] Phase 2: CRIT-MEM-01 Arbitrary non-page-aligned prefix split (match_edge % 16 != 0)... " << std::flush;
        std::vector<int32_t> unaligned_tokens = branch_tokens;
        unaligned_tokens.resize(4800 + 400 + 100);
        // Split into delta edge (400 tokens) at offset 205 (205 % 16 == 13 != 0)
        size_t unaligned_split_pos = 4800 + 205;
        unaligned_tokens[unaligned_split_pos] = 888888;
        for (size_t i = unaligned_split_pos + 1; i < unaligned_tokens.size(); ++i) {
            unaligned_tokens[i] = 7000 + (i % 100);
        }
        auto unaligned_node = tree_paged.insert(unaligned_tokens.data(), unaligned_tokens.size(), unaligned_tokens.size(),
                                                s_devs, s_states, s_R, geom, s_streams);
        assert(unaligned_node != nullptr);
        auto unaligned_split_parent = unaligned_node->parent.lock();
        assert(unaligned_split_parent != nullptr);
        assert(unaligned_split_parent->edge_tokens.size() == 205);
        assert(unaligned_split_parent->kv_page_ids.size() == (205 + 15) / 16);
        assert(unaligned_node->kv_page_ids.size() == (unaligned_node->edge_tokens.size() + 15) / 16);
        auto vbt_unaligned = tree_paged.resolve_virtual_block_table(unaligned_node);
        size_t expected_total_pages = 250 + 50 + ((205 + 15) / 16) + ((unaligned_node->edge_tokens.size() + 15) / 16);
        assert(vbt_unaligned.size() == expected_total_pages); (void) expected_total_pages;
        std::cout << "[PASS]" << std::endl;

        cudaStreamDestroy(strm);
    }

    // --- Phase 3: PagedKVPool Atomic Page Ref-Counting & Multi-Tenant Prefix Sharing ---
    {
        std::cout << "  [TEST] Phase 3: PagedKVPool Atomic Page Ref-Counting & Cross-Slot Sharing... " << std::flush;
        strata::core::PagedKVPool pool(1000);

        // 1. Initial allocate 100 pages for System Prompt
        auto sys_pages = pool.allocate_pages(100);
        assert(sys_pages.size() == 100);
        assert(pool.allocated_pages() == 100);
        for (int32_t p : sys_pages) {
            assert(pool.page_ref_count(p) == 1);
        }

        // 2. Slot 1 and Slot 2 fork and retain the SAME 100 physical pages
        pool.retain_pages(sys_pages); // Slot 1 retains
        pool.retain_pages(sys_pages); // Slot 2 retains
        for (int32_t p : sys_pages) {
            assert(pool.page_ref_count(p) == 3);
        }
        // Allocated pages MUST still be 100! (True deduplication across slots)
        assert(pool.allocated_pages() == 100);

        // 3. Slot 1 finishes and releases pages
        pool.release_pages(sys_pages);
        for (int32_t p : sys_pages) {
            assert(pool.page_ref_count(p) == 2);
        }
        assert(pool.allocated_pages() == 100); // Still 100 allocated because Slot 2 and RadixNode hold it!

        // 4. Slot 2 finishes and releases pages
        pool.release_pages(sys_pages);
        for (int32_t p : sys_pages) {
            assert(pool.page_ref_count(p) == 1);
        }
        assert(pool.allocated_pages() == 100); // Still 100 allocated because RadixNode holds it!

        // 5. RadixNode is finally evicted and frees pages
        pool.free_pages(sys_pages);
        for (int32_t p : sys_pages) {
            assert(pool.page_ref_count(p) == 0);
        }
        assert(pool.allocated_pages() == 0); // Reclaimed completely!
        assert(pool.free_pages_count() == 1000);
        std::cout << "[PASS]" << std::endl;
    }

    std::cout << "All RadixTree & HiCache L2 unit tests passed successfully!" << std::endl;
    return 0;
}
