// src/core/radix_tree.cpp - Lockless RadixTree Prefix Matching, Mutation & Pinned DMA Implementation
//
// Cross-Engine Attribution and Research Lineage:
//   - SGLang: RadixAttention tree prefix caching, hardware CRC32C chunking, and multi-tier memory management (arXiv:2312.07104)
//
#include "strata/core/radix_tree.hpp"
#include "strata/core/nvme_tier.hpp"
#include "strata/core/radix_compactor.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

namespace strata::core {

void RadixStageSnapshot::free_device() {
    if (device >= 0) {
        const core::OnDevice on(device);
        if (gdn_saved) { cudaFree(gdn_saved); gdn_saved = nullptr; }
        if (ple_saved) { cudaFree(ple_saved); ple_saved = nullptr; }
        if (R_saved) { cudaFree(R_saved); R_saved = nullptr; }
        for (auto& s : qsa_slices) {
            if (s.k_q) { cudaFree(s.k_q); s.k_q = nullptr; }
            if (s.v_q4) { cudaFree(s.v_q4); s.v_q4 = nullptr; }
            if (s.k_scale) { cudaFree(s.k_scale); s.k_scale = nullptr; }
            if (s.v_scale) { cudaFree(s.v_scale); s.v_scale = nullptr; }
            if (s.k_pool) { cudaFree(s.k_pool); s.k_pool = nullptr; }
            if (s.v_pool) { cudaFree(s.v_pool); s.v_pool = nullptr; }
            if (s.idx_tail) { cudaFree(s.idx_tail); s.idx_tail = nullptr; }
            if (s.idx_dead) { cudaFree(s.idx_dead); s.idx_dead = nullptr; }
            if (s.idx_pooled) { cudaFree(s.idx_pooled); s.idx_pooled = nullptr; }
            if (s.idx_block_pos) { cudaFree(s.idx_block_pos); s.idx_block_pos = nullptr; }
        }
        qsa_slices.clear();
    }
    gdn_bytes = ple_bytes = R_bytes = 0;
}

size_t RadixNode::total_vram_bytes() const {
    size_t total = 0;
    for (const auto& ss : stage_snapshots) {
        total += ss.gdn_bytes + ss.ple_bytes + ss.R_bytes;
        for (const auto& qs : ss.qsa_slices) {
            total += qs.k_bytes + qs.v_bytes + qs.k_scale_bytes + qs.v_scale_bytes;
            total += qs.idx_tail_bytes + qs.idx_dead_bytes + qs.idx_pooled_bytes;
            if (qs.idx_block_pos) total += sizeof(int32_t);
        }
    }
    return total;
}

size_t RadixNode::total_host_bytes() const {
    size_t total = 0;
    for (const auto& hss : stage_host_snapshots) {
        total += hss.gdn_data.size() + hss.ple_data.size() + hss.R_data.size();
        for (const auto& qs : hss.qsa_slices) {
            total += qs.k_q.size() + qs.v_q4.size() + qs.k_scale.size() + qs.v_scale.size();
            total += qs.idx_tail.size() + qs.idx_dead.size() + qs.idx_pooled.size();
            if (qs.has_idx_block_pos) total += sizeof(int32_t);
        }
    }
    return total;
}

void RadixNode::free_device() {
    for (auto& ss : stage_snapshots) {
        ss.free_device();
    }
    stage_snapshots.clear();
}

void RadixNode::free_host() {
    stage_host_snapshots.clear();
    is_host_parked = false;
}

bool RadixNode::unpark_to_device() {
    if (!is_host_parked || stage_host_snapshots.empty()) return false;
    stage_snapshots.resize(stage_host_snapshots.size());
    for (size_t st = 0; st < stage_host_snapshots.size(); ++st) {
        auto& hss = stage_host_snapshots[st];
        auto& ss = stage_snapshots[st];
        ss.device = hss.device;
        ss.ple_prev_saved[0] = hss.ple_prev_saved[0];
        ss.ple_prev_saved[1] = hss.ple_prev_saved[1];
        ss.ple_token_saved = hss.ple_token_saved;
        ss.qsa_ord0 = hss.qsa_ord0;
        ss.qsa_alloc = hss.qsa_alloc;
        if (ss.device < 0) continue;
        const core::OnDevice on(ss.device);
        auto up = [](PinnedBuffer& h, void** d, size_t& bytes) {
            if (h.empty()) return;
            bytes = h.size();
            if (cudaMalloc(d, bytes) != cudaSuccess) { *d = nullptr; bytes = 0; return; }
            cudaMemcpy(*d, h.data(), bytes, cudaMemcpyHostToDevice);
        };
        up(hss.gdn_data, (void**) &ss.gdn_saved, ss.gdn_bytes);
        up(hss.ple_data, (void**) &ss.ple_saved, ss.ple_bytes);
        up(hss.R_data, (void**) &ss.R_saved, ss.R_bytes);
        ss.qsa_slices.resize(hss.qsa_slices.size());
        for (size_t j = 0; j < hss.qsa_slices.size(); ++j) {
            auto& hs = hss.qsa_slices[j];
            auto& ds = ss.qsa_slices[j];
            up(hs.k_q, &ds.k_q, ds.k_bytes);
            up(hs.v_q4, &ds.v_q4, ds.v_bytes);
            up(hs.k_scale, &ds.k_scale, ds.k_scale_bytes);
            up(hs.v_scale, &ds.v_scale, ds.v_scale_bytes);
            up(hs.idx_tail, &ds.idx_tail, ds.idx_tail_bytes);
            up(hs.idx_dead, &ds.idx_dead, ds.idx_dead_bytes);
            up(hs.idx_pooled, &ds.idx_pooled, ds.idx_pooled_bytes);
            if (hs.has_idx_block_pos) {
                if (cudaMalloc((void**) &ds.idx_block_pos, sizeof(int32_t)) == cudaSuccess) {
                    cudaMemcpy(ds.idx_block_pos, &hs.idx_block_pos_val, sizeof(int32_t), cudaMemcpyHostToDevice);
                }
            }
        }
    }
    is_host_parked = false;
    return true;
}

void RadixNode::park_to_host() {
    if (stage_snapshots.empty()) return;
    stage_host_snapshots.resize(stage_snapshots.size());
    for (size_t st = 0; st < stage_snapshots.size(); ++st) {
        auto& ss = stage_snapshots[st];
        auto& hss = stage_host_snapshots[st];
        hss.device = ss.device;
        hss.ple_prev_saved[0] = ss.ple_prev_saved[0];
        hss.ple_prev_saved[1] = ss.ple_prev_saved[1];
        hss.ple_token_saved = ss.ple_token_saved;
        hss.qsa_ord0 = ss.qsa_ord0;
        hss.qsa_alloc = ss.qsa_alloc;

        if (ss.device >= 0) {
            const core::OnDevice on(ss.device);
            if (ss.gdn_saved && ss.gdn_bytes > 0) {
                hss.gdn_data.resize(ss.gdn_bytes);
                cudaMemcpy(hss.gdn_data.data(), ss.gdn_saved, ss.gdn_bytes, cudaMemcpyDeviceToHost);
            }
            if (ss.ple_saved && ss.ple_bytes > 0) {
                hss.ple_data.resize(ss.ple_bytes);
                cudaMemcpy(hss.ple_data.data(), ss.ple_saved, ss.ple_bytes, cudaMemcpyDeviceToHost);
            }
            if (ss.R_saved && ss.R_bytes > 0) {
                hss.R_data.resize(ss.R_bytes);
                cudaMemcpy(hss.R_data.data(), ss.R_saved, ss.R_bytes, cudaMemcpyDeviceToHost);
            }
            hss.qsa_slices.resize(ss.qsa_slices.size());
            for (size_t j = 0; j < ss.qsa_slices.size(); ++j) {
                const auto& src = ss.qsa_slices[j];
                auto& dst = hss.qsa_slices[j];
                if (src.k_q && src.k_bytes > 0) {
                    dst.k_q.resize(src.k_bytes);
                    cudaMemcpy(dst.k_q.data(), src.k_q, src.k_bytes, cudaMemcpyDeviceToHost);
                }
                if (src.v_q4 && src.v_bytes > 0) {
                    dst.v_q4.resize(src.v_bytes);
                    cudaMemcpy(dst.v_q4.data(), src.v_q4, src.v_bytes, cudaMemcpyDeviceToHost);
                }
                if (src.k_scale && src.k_scale_bytes > 0) {
                    dst.k_scale.resize(src.k_scale_bytes);
                    cudaMemcpy(dst.k_scale.data(), src.k_scale, src.k_scale_bytes, cudaMemcpyDeviceToHost);
                }
                if (src.v_scale && src.v_scale_bytes > 0) {
                    dst.v_scale.resize(src.v_scale_bytes);
                    cudaMemcpy(dst.v_scale.data(), src.v_scale, src.v_scale_bytes, cudaMemcpyDeviceToHost);
                }
                if (src.idx_tail && src.idx_tail_bytes > 0) {
                    dst.idx_tail.resize(src.idx_tail_bytes);
                    cudaMemcpy(dst.idx_tail.data(), src.idx_tail, src.idx_tail_bytes, cudaMemcpyDeviceToHost);
                }
                if (src.idx_dead && src.idx_dead_bytes > 0) {
                    dst.idx_dead.resize(src.idx_dead_bytes);
                    cudaMemcpy(dst.idx_dead.data(), src.idx_dead, src.idx_dead_bytes, cudaMemcpyDeviceToHost);
                }
                if (src.idx_pooled && src.idx_pooled_bytes > 0) {
                    dst.idx_pooled.resize(src.idx_pooled_bytes);
                    cudaMemcpy(dst.idx_pooled.data(), src.idx_pooled, src.idx_pooled_bytes, cudaMemcpyDeviceToHost);
                }
                if (src.idx_block_pos) {
                    dst.has_idx_block_pos = true;
                    cudaMemcpy(&dst.idx_block_pos_val, src.idx_block_pos, sizeof(int32_t), cudaMemcpyDeviceToHost);
                }
            }
        }
        ss.free_device();
    }
    stage_snapshots.clear();
    is_host_parked = true;
}

bool RadixNode::offload_to_nvme(NVMeStorageTier& nvme) {
    if (!has_host_snapshot() && has_device_snapshot()) {
        park_to_host();
    }
    if (!has_host_snapshot()) return false;

    // Calculate total serialization size
    size_t total_payload = sizeof(NVMeNodeHeader);
    total_payload += edge_tokens.size() * sizeof(int32_t);
    total_payload += stage_host_snapshots.size() * sizeof(NVMeStageHeader);

    for (const auto& hss : stage_host_snapshots) {
        total_payload += hss.gdn_data.size();
        total_payload += hss.ple_data.size();
        total_payload += hss.R_data.size();
        total_payload += hss.qsa_slices.size() * sizeof(NVMeQsaSliceHeader);
        for (const auto& slice : hss.qsa_slices) {
            total_payload += slice.k_q.size();
            total_payload += slice.v_q4.size();
            total_payload += slice.k_scale.size();
            total_payload += slice.v_scale.size();
            total_payload += slice.idx_tail.size();
            total_payload += slice.idx_dead.size();
            total_payload += slice.idx_pooled.size();
        }
    }

    AlignedBuffer buffer(total_payload);
    uint8_t* p = buffer.data();
    size_t offset = 0;

    NVMeNodeHeader hdr;
    hdr.magic = kNVMeMagic;
    hdr.version = kNVMeVersion;
    hdr.node_id = id;
    hdr.prefix_len = prefix_len;
    hdr.n_edge_tokens = static_cast<uint32_t>(edge_tokens.size());
    hdr.n_stages = static_cast<uint32_t>(stage_host_snapshots.size());
    hdr.total_payload_bytes = total_payload;
    hdr.checksum = 0xFFFFFFFF53545241ULL; // Task 2.3: Seeded CRC32C

    std::memcpy(p + offset, &hdr, sizeof(hdr));
    offset += sizeof(hdr);

    if (!edge_tokens.empty()) {
        size_t bytes = edge_tokens.size() * sizeof(int32_t);
        std::memcpy(p + offset, edge_tokens.data(), bytes);
        offset += bytes;
    }

    for (const auto& hss : stage_host_snapshots) {
        NVMeStageHeader shdr;
        shdr.device = hss.device;
        shdr.ple_prev[0] = hss.ple_prev_saved[0];
        shdr.ple_prev[1] = hss.ple_prev_saved[1];
        shdr.ple_token = hss.ple_token_saved;
        shdr.gdn_bytes = hss.gdn_data.size();
        shdr.ple_bytes = hss.ple_data.size();
        shdr.R_bytes = hss.R_data.size();
        shdr.qsa_ord0 = hss.qsa_ord0;
        shdr.qsa_alloc = hss.qsa_alloc;
        shdr.n_qsa_slices = static_cast<uint32_t>(hss.qsa_slices.size());

        std::memcpy(p + offset, &shdr, sizeof(shdr));
        offset += sizeof(shdr);

        if (shdr.gdn_bytes > 0) {
            std::memcpy(p + offset, hss.gdn_data.data(), shdr.gdn_bytes);
            offset += shdr.gdn_bytes;
        }
        if (shdr.ple_bytes > 0) {
            std::memcpy(p + offset, hss.ple_data.data(), shdr.ple_bytes);
            offset += shdr.ple_bytes;
        }
        if (shdr.R_bytes > 0) {
            std::memcpy(p + offset, hss.R_data.data(), shdr.R_bytes);
            offset += shdr.R_bytes;
        }

        for (const auto& slice : hss.qsa_slices) {
            NVMeQsaSliceHeader qhdr;
            qhdr.k_bytes = slice.k_q.size();
            qhdr.v_bytes = slice.v_q4.size();
            qhdr.k_scale_bytes = slice.k_scale.size();
            qhdr.v_scale_bytes = slice.v_scale.size();
            qhdr.idx_tail_bytes = slice.idx_tail.size();
            qhdr.idx_dead_bytes = slice.idx_dead.size();
            qhdr.idx_pooled_bytes = slice.idx_pooled.size();
            qhdr.idx_block_pos_val = slice.idx_block_pos_val;
            qhdr.has_idx_block_pos = slice.has_idx_block_pos ? 1 : 0;

            std::memcpy(p + offset, &qhdr, sizeof(qhdr));
            offset += sizeof(qhdr);

            if (qhdr.k_bytes > 0) {
                std::memcpy(p + offset, slice.k_q.data(), qhdr.k_bytes);
                offset += qhdr.k_bytes;
            }
            if (qhdr.v_bytes > 0) {
                std::memcpy(p + offset, slice.v_q4.data(), qhdr.v_bytes);
                offset += qhdr.v_bytes;
            }
            if (qhdr.k_scale_bytes > 0) {
                std::memcpy(p + offset, slice.k_scale.data(), qhdr.k_scale_bytes);
                offset += qhdr.k_scale_bytes;
            }
            if (qhdr.v_scale_bytes > 0) {
                std::memcpy(p + offset, slice.v_scale.data(), qhdr.v_scale_bytes);
                offset += qhdr.v_scale_bytes;
            }
            if (qhdr.idx_tail_bytes > 0) {
                std::memcpy(p + offset, slice.idx_tail.data(), qhdr.idx_tail_bytes);
                offset += qhdr.idx_tail_bytes;
            }
            if (qhdr.idx_dead_bytes > 0) {
                std::memcpy(p + offset, slice.idx_dead.data(), qhdr.idx_dead_bytes);
                offset += qhdr.idx_dead_bytes;
            }
            if (qhdr.idx_pooled_bytes > 0) {
                std::memcpy(p + offset, slice.idx_pooled.data(), qhdr.idx_pooled_bytes);
                offset += qhdr.idx_pooled_bytes;
            }
        }
    }

    // Task 2.3: Compute hardware SSE4.2 CRC32C over serialized payload
    uint64_t crc = 0xFFFFFFFF53545241ULL;
    const uint8_t* payload_bytes = buffer.data() + sizeof(NVMeNodeHeader);
    const size_t crc_len = total_payload - sizeof(NVMeNodeHeader);
    const size_t qwords = crc_len / 8;
    const uint64_t* qp = reinterpret_cast<const uint64_t*>(payload_bytes);
    for (size_t qi = 0; qi < qwords; ++qi) {
#if defined(__SSE4_2__)
        crc = _mm_crc32_u64(crc, qp[qi]);
#else
        crc = (crc ^ qp[qi]) * 1099511628211ULL;
#endif
    }
    // Update header checksum in buffer
    reinterpret_cast<NVMeNodeHeader*>(buffer.data())->checksum = crc;

    if (!nvme.write_node_direct(id, buffer.data(), total_payload)) {
        return false;
    }

    l3_bytes = total_payload;
    is_l3_offloaded = true;
    free_host();
    return true;
}

bool RadixNode::hydrate_from_nvme(NVMeStorageTier& nvme) {
    if (!is_l3_offloaded) return true;
    if (l3_bytes == 0) return false;

    // Task 4: RAII ScopedReadTracker to prioritize L3 reads over background writes
    struct ScopedReadTracker {
        ScopedReadTracker() { RadixTree::in_flight_l3_reads_.fetch_add(1, std::memory_order_relaxed); }
        ~ScopedReadTracker() { RadixTree::in_flight_l3_reads_.fetch_sub(1, std::memory_order_relaxed); }
    } read_guard;

    AlignedBuffer buffer(l3_bytes);
    if (!nvme.read_node_direct(id, buffer.data(), l3_bytes)) {
        return false;
    }

    const uint8_t* p = buffer.data();
    size_t offset = 0;

    if (!p || l3_bytes < sizeof(NVMeNodeHeader)) return false;
    NVMeNodeHeader hdr;
    std::memcpy(&hdr, p + offset, sizeof(hdr));
    offset += sizeof(hdr);

    if (hdr.magic != kNVMeMagic || hdr.version != kNVMeVersion || hdr.node_id != id) {
        return false;
    }
    // Task 2.3: Hardware SSE4.2 CRC32C Validation
    if (hdr.checksum != 0) {
        uint64_t crc = 0xFFFFFFFF53545241ULL;
        const uint8_t* payload_bytes = buffer.data() + sizeof(NVMeNodeHeader);
        const size_t crc_len = l3_bytes - sizeof(NVMeNodeHeader);
        const size_t qwords = crc_len / 8;
        const uint64_t* qp = reinterpret_cast<const uint64_t*>(payload_bytes);
        for (size_t qi = 0; qi < qwords; ++qi) {
#if defined(__SSE4_2__)
            crc = _mm_crc32_u64(crc, qp[qi]);
#else
            crc = (crc ^ qp[qi]) * 1099511628211ULL;
#endif
        }
        if (crc != hdr.checksum) {
            std::fprintf(stderr, "strata-agx radix: CRC32C mismatch on NVMe node #%lld (got %llx, expected %llx)\n",
                         (long long) id, (unsigned long long) crc, (unsigned long long) hdr.checksum);
            return false;
        }
    }

    if (hdr.n_edge_tokens > 0) {
        size_t bytes = hdr.n_edge_tokens * sizeof(int32_t);
        edge_tokens.resize(hdr.n_edge_tokens);
        std::memcpy(edge_tokens.data(), p + offset, bytes);
        offset += bytes;
        update_chunk_hashes();
    }

    stage_host_snapshots.resize(hdr.n_stages);
    for (uint32_t st = 0; st < hdr.n_stages; ++st) {
        NVMeStageHeader shdr;
        std::memcpy(&shdr, p + offset, sizeof(shdr));
        offset += sizeof(shdr);

        auto& hss = stage_host_snapshots[st];
        hss.device = shdr.device;
        hss.ple_prev_saved[0] = shdr.ple_prev[0];
        hss.ple_prev_saved[1] = shdr.ple_prev[1];
        hss.ple_token_saved = shdr.ple_token;
        hss.qsa_ord0 = shdr.qsa_ord0;
        hss.qsa_alloc = shdr.qsa_alloc;

        if (shdr.gdn_bytes > 0) {
            hss.gdn_data.resize(shdr.gdn_bytes);
            std::memcpy(hss.gdn_data.data(), p + offset, shdr.gdn_bytes);
            offset += shdr.gdn_bytes;
        }
        if (shdr.ple_bytes > 0) {
            hss.ple_data.resize(shdr.ple_bytes);
            std::memcpy(hss.ple_data.data(), p + offset, shdr.ple_bytes);
            offset += shdr.ple_bytes;
        }
        if (shdr.R_bytes > 0) {
            hss.R_data.resize(shdr.R_bytes);
            std::memcpy(hss.R_data.data(), p + offset, shdr.R_bytes);
            offset += shdr.R_bytes;
        }

        hss.qsa_slices.resize(shdr.n_qsa_slices);
        for (uint32_t j = 0; j < shdr.n_qsa_slices; ++j) {
            NVMeQsaSliceHeader qhdr;
            std::memcpy(&qhdr, p + offset, sizeof(qhdr));
            offset += sizeof(qhdr);

            auto& slice = hss.qsa_slices[j];
            slice.idx_block_pos_val = qhdr.idx_block_pos_val;
            slice.has_idx_block_pos = qhdr.has_idx_block_pos != 0;

            if (qhdr.k_bytes > 0) {
                slice.k_q.resize(qhdr.k_bytes);
                std::memcpy(slice.k_q.data(), p + offset, qhdr.k_bytes);
                offset += qhdr.k_bytes;
            }
            if (qhdr.v_bytes > 0) {
                slice.v_q4.resize(qhdr.v_bytes);
                std::memcpy(slice.v_q4.data(), p + offset, qhdr.v_bytes);
                offset += qhdr.v_bytes;
            }
            if (qhdr.k_scale_bytes > 0) {
                slice.k_scale.resize(qhdr.k_scale_bytes);
                std::memcpy(slice.k_scale.data(), p + offset, qhdr.k_scale_bytes);
                offset += qhdr.k_scale_bytes;
            }
            if (qhdr.v_scale_bytes > 0) {
                slice.v_scale.resize(qhdr.v_scale_bytes);
                std::memcpy(slice.v_scale.data(), p + offset, qhdr.v_scale_bytes);
                offset += qhdr.v_scale_bytes;
            }
            if (qhdr.idx_tail_bytes > 0) {
                slice.idx_tail.resize(qhdr.idx_tail_bytes);
                std::memcpy(slice.idx_tail.data(), p + offset, qhdr.idx_tail_bytes);
                offset += qhdr.idx_tail_bytes;
            }
            if (qhdr.idx_dead_bytes > 0) {
                slice.idx_dead.resize(qhdr.idx_dead_bytes);
                std::memcpy(slice.idx_dead.data(), p + offset, qhdr.idx_dead_bytes);
                offset += qhdr.idx_dead_bytes;
            }
            if (qhdr.idx_pooled_bytes > 0) {
                slice.idx_pooled.resize(qhdr.idx_pooled_bytes);
                std::memcpy(slice.idx_pooled.data(), p + offset, qhdr.idx_pooled_bytes);
                offset += qhdr.idx_pooled_bytes;
            }
        }
    }

    is_host_parked = true;
    is_l3_offloaded = false;
    return true;
}

void RadixNode::free_nvme(NVMeStorageTier& nvme) {
    if (is_l3_offloaded) {
        nvme.remove_node(id);
        is_l3_offloaded = false;
        l3_bytes = 0;
    }
}

RadixNode::~RadixNode() {
    free_device();
    free_host();
}

RadixTree::RadixTree(size_t max_cached_snapshots, size_t max_host_snapshots)
    : max_cached_snapshots_(max_cached_snapshots), max_host_snapshots_(max_host_snapshots) {
    const char* env_l2 = std::getenv("STRATA_HICACHE_L2_SLOTS");
    if (env_l2) {
        max_host_snapshots_ = (size_t) std::strtoul(env_l2, nullptr, 10);
    }
    const char* env_hg = std::getenv("STRATA_RADIX_HOST_GIB");
    if (env_hg && *env_hg) {
        host_ram_budget_ = (size_t) std::strtoul(env_hg, nullptr, 10) << 30;
    }
    const char* env_vb = std::getenv("STRATA_RADIX_VRAM_MIB");
    if (env_vb && *env_vb) {
        vram_budget_ = (size_t) std::strtoul(env_vb, nullptr, 10) << 20;
    }
    const char* env_dr = std::getenv("STRATA_RADIX_DEEP_RESERVE_MIB");
    if (env_dr && *env_dr) {
        deep_reserve_ = (size_t) std::strtoul(env_dr, nullptr, 10) << 20;
    }
    const char* env_nvme_gib = std::getenv("STRATA_RADIX_NVME_MAX_GIB");
    if (env_nvme_gib && *env_nvme_gib) {
        nvme_budget_bytes_ = (size_t) std::strtoul(env_nvme_gib, nullptr, 10) << 30;
    }
    const char* env_nvme_cd = std::getenv("STRATA_RADIX_NVME_COOLDOWN_SEC");
    if (env_nvme_cd && *env_nvme_cd) {
        nvme_cooldown_sec_ = std::atoi(env_nvme_cd);
    }
    root_ = std::make_shared<RadixNode>();
    root_->created_at = std::chrono::steady_clock::now();
    root_->id = 0;
    root_->prefix_len = 0;
    root_->last_accessed = std::chrono::steady_clock::now();
    node_count_ = 1;

    nvme_worker_ = std::thread(&RadixTree::nvme_worker_loop, this);
}

RadixTree::~RadixTree() {
    drain_nvme_staging();
}

void RadixTree::drain_nvme_staging() {
    {
        std::lock_guard<std::mutex> lk(nvme_queue_mutex_);
        nvme_stop_ = true;
    }
    nvme_cv_.notify_all();
    if (nvme_worker_.joinable()) {
        nvme_worker_.join();
    }
}

void RadixTree::nvme_worker_loop() {
    while (true) {
        NVMeStagingTask task;
        {
            std::unique_lock<std::mutex> lk(nvme_queue_mutex_);
            nvme_cv_.wait(lk, [&]() {
                return nvme_stop_.load() || !nvme_queue_.empty();
            });
            if (nvme_stop_.load() && nvme_queue_.empty()) {
                break;
            }
            if (nvme_queue_.empty()) continue;
            task = std::move(nvme_queue_.front());
            nvme_queue_.pop();
        }

        if (nvme_tier_ && task.payload.data() && task.total_payload > 0) {
            // Task 4: I/O priority gate - yield background writes while active L3 read is in flight
            while (in_flight_l3_reads_.load(std::memory_order_relaxed) > 0 && !nvme_stop_.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            bool ok = nvme_tier_->write_node_direct(task.node_id, task.payload.data(), task.total_payload);
            if (auto node = task.node_weak.lock()) {
                std::unique_lock<std::shared_mutex> lk(rw_lock_);
                if (ok) {
                    node->l3_bytes = task.total_payload;
                    node->is_l3_offloaded = true;
                    node->has_l3_persisted = true;
                    // Continuous write-through: do NOT call node->free_host()!
                    // Node remains hot in L2 Host RAM for sub-5ms restore while backed up to L3.
                } else {
                    if (cached_nvme_snapshots_ > 0) --cached_nvme_snapshots_;
                }
                node->is_staging_to_nvme = false;
            } else {
                std::unique_lock<std::shared_mutex> lk(rw_lock_);
                if (!ok && cached_nvme_snapshots_ > 0) --cached_nvme_snapshots_;
            }
        } else {
            if (auto node = task.node_weak.lock()) {
                std::unique_lock<std::shared_mutex> lk(rw_lock_);
                if (cached_nvme_snapshots_ > 0) --cached_nvme_snapshots_;
                node->is_staging_to_nvme = false;
            }
        }
    }
}

// S26f: total device-snapshot bytes across the tree (walk; trees are small).
size_t RadixTree::device_snapshot_bytes_locked() const {
    size_t total = 0;
    std::vector<std::shared_ptr<RadixNode>> stack;
    stack.push_back(root_);
    while (!stack.empty()) {
        auto node = stack.back();
        stack.pop_back();
        if (!node) continue;
        for (auto& ss : node->stage_snapshots) {
            total += ss.gdn_bytes + ss.ple_bytes + ss.R_bytes;
            for (auto& sl : ss.qsa_slices) {
                total += sl.k_bytes + sl.v_bytes + sl.k_scale_bytes + sl.v_scale_bytes
                       + sl.idx_tail_bytes + sl.idx_dead_bytes + sl.idx_pooled_bytes;
            }
        }
        for (auto& kv : node->children) stack.push_back(kv.second);
    }
    return total;
}

size_t RadixTree::device_snapshot_bytes() const {
    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    return device_snapshot_bytes_locked();
}

size_t RadixTree::host_snapshot_bytes_locked() const {
    size_t total = 0;
    std::vector<std::shared_ptr<RadixNode>> stack;
    stack.push_back(root_);
    while (!stack.empty()) {
        auto node = stack.back();
        stack.pop_back();
        if (!node) continue;
        if (node->has_host_snapshot()) {
            total += node->total_host_bytes();
        }
        for (auto& kv : node->children) stack.push_back(kv.second);
    }
    return total;
}


size_t RadixTree::l3_snapshot_bytes() const {
    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    size_t total = 0;
    std::vector<std::shared_ptr<RadixNode>> stack;
    stack.push_back(root_);
    while (!stack.empty()) {
        auto node = stack.back();
        stack.pop_back();
        if (!node) continue;
        if (node->is_l3_offloaded) {
            total += node->l3_bytes;
        }
        for (const auto& kv : node->children) stack.push_back(kv.second);
    }
    return total;
}

bool RadixTree::queue_nvme_write_through_locked(const std::shared_ptr<RadixNode>& node) {
    if (!nvme_tier_ || !node || !node->has_host_snapshot()) return false;
    if (node->has_l3_persisted || node->is_l3_offloaded || node->is_staging_to_nvme.load()) return false;

    // Guardrail: Minimum prefix length (must be SYSTEM_ROOT or >= 512 tokens)
    if (node->kind != RadixNodeKind::SYSTEM_ROOT && node->prefix_len < 512) {
        return false;
    }

    // Guardrail: TBW Cooldown protection for turn/leaf nodes (skip ephemeral micro-bursts)
    if (node->kind != RadixNodeKind::SYSTEM_ROOT && nvme_cooldown_sec_ > 0) {
        auto age_sec = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - node->created_at).count();
        if (age_sec < nvme_cooldown_sec_) {
            return false;
        }
    }

    // Guardrail: Enforce 500 GiB NVMe budget ceiling via background LRU prune
    if (nvme_budget_bytes_ > 0) {
        prune_nvme_lru_locked(nvme_budget_bytes_);
    }

    size_t total_payload = sizeof(NVMeNodeHeader);
    total_payload += node->edge_tokens.size() * sizeof(int32_t);
    total_payload += node->stage_host_snapshots.size() * sizeof(NVMeStageHeader);

    for (const auto& hss : node->stage_host_snapshots) {
        total_payload += hss.gdn_data.size();
        total_payload += hss.ple_data.size();
        total_payload += hss.R_data.size();
        total_payload += hss.qsa_slices.size() * sizeof(NVMeQsaSliceHeader);
        for (const auto& slice : hss.qsa_slices) {
            total_payload += slice.k_q.size();
            total_payload += slice.v_q4.size();
            total_payload += slice.k_scale.size();
            total_payload += slice.v_scale.size();
            total_payload += slice.idx_tail.size();
            total_payload += slice.idx_dead.size();
            total_payload += slice.idx_pooled.size();
        }
    }

    AlignedBuffer buffer(total_payload);
    uint8_t* p = buffer.data();
    size_t offset = 0;

    NVMeNodeHeader hdr;
    hdr.magic = kNVMeMagic;
    hdr.version = kNVMeVersion;
    hdr.node_id = node->id;
    hdr.prefix_len = node->prefix_len;
    hdr.n_edge_tokens = static_cast<uint32_t>(node->edge_tokens.size());
    hdr.n_stages = static_cast<uint32_t>(node->stage_host_snapshots.size());
    hdr.total_payload_bytes = total_payload;
    hdr.checksum = 0xFFFFFFFF53545241ULL;

    std::memcpy(p + offset, &hdr, sizeof(hdr));
    offset += sizeof(hdr);

    if (!node->edge_tokens.empty()) {
        size_t bytes = node->edge_tokens.size() * sizeof(int32_t);
        std::memcpy(p + offset, node->edge_tokens.data(), bytes);
        offset += bytes;
    }

    for (const auto& hss : node->stage_host_snapshots) {
        NVMeStageHeader shdr;
        shdr.device = hss.device;
        shdr.ple_prev[0] = hss.ple_prev_saved[0];
        shdr.ple_prev[1] = hss.ple_prev_saved[1];
        shdr.ple_token = hss.ple_token_saved;
        shdr.gdn_bytes = hss.gdn_data.size();
        shdr.ple_bytes = hss.ple_data.size();
        shdr.R_bytes = hss.R_data.size();
        shdr.qsa_ord0 = hss.qsa_ord0;
        shdr.qsa_alloc = hss.qsa_alloc;
        shdr.n_qsa_slices = static_cast<uint32_t>(hss.qsa_slices.size());

        std::memcpy(p + offset, &shdr, sizeof(shdr));
        offset += sizeof(shdr);

        if (shdr.gdn_bytes > 0) {
            std::memcpy(p + offset, hss.gdn_data.data(), shdr.gdn_bytes);
            offset += shdr.gdn_bytes;
        }
        if (shdr.ple_bytes > 0) {
            std::memcpy(p + offset, hss.ple_data.data(), shdr.ple_bytes);
            offset += shdr.ple_bytes;
        }
        if (shdr.R_bytes > 0) {
            std::memcpy(p + offset, hss.R_data.data(), shdr.R_bytes);
            offset += shdr.R_bytes;
        }

        for (const auto& slice : hss.qsa_slices) {
            NVMeQsaSliceHeader qhdr;
            qhdr.k_bytes = slice.k_q.size();
            qhdr.v_bytes = slice.v_q4.size();
            qhdr.k_scale_bytes = slice.k_scale.size();
            qhdr.v_scale_bytes = slice.v_scale.size();
            qhdr.idx_tail_bytes = slice.idx_tail.size();
            qhdr.idx_dead_bytes = slice.idx_dead.size();
            qhdr.idx_pooled_bytes = slice.idx_pooled.size();
            qhdr.idx_block_pos_val = slice.idx_block_pos_val;
            qhdr.has_idx_block_pos = slice.has_idx_block_pos ? 1 : 0;

            std::memcpy(p + offset, &qhdr, sizeof(qhdr));
            offset += sizeof(qhdr);

            if (qhdr.k_bytes > 0) {
                std::memcpy(p + offset, slice.k_q.data(), qhdr.k_bytes);
                offset += qhdr.k_bytes;
            }
            if (qhdr.v_bytes > 0) {
                std::memcpy(p + offset, slice.v_q4.data(), qhdr.v_bytes);
                offset += qhdr.v_bytes;
            }
            if (qhdr.k_scale_bytes > 0) {
                std::memcpy(p + offset, slice.k_scale.data(), qhdr.k_scale_bytes);
                offset += qhdr.k_scale_bytes;
            }
            if (qhdr.v_scale_bytes > 0) {
                std::memcpy(p + offset, slice.v_scale.data(), qhdr.v_scale_bytes);
                offset += qhdr.v_scale_bytes;
            }
            if (qhdr.idx_tail_bytes > 0) {
                std::memcpy(p + offset, slice.idx_tail.data(), qhdr.idx_tail_bytes);
                offset += qhdr.idx_tail_bytes;
            }
            if (qhdr.idx_dead_bytes > 0) {
                std::memcpy(p + offset, slice.idx_dead.data(), qhdr.idx_dead_bytes);
                offset += qhdr.idx_dead_bytes;
            }
            if (qhdr.idx_pooled_bytes > 0) {
                std::memcpy(p + offset, slice.idx_pooled.data(), qhdr.idx_pooled_bytes);
                offset += qhdr.idx_pooled_bytes;
            }
        }
    }

    uint64_t crc = 0xFFFFFFFF53545241ULL;
    const uint8_t* payload_bytes = buffer.data() + sizeof(NVMeNodeHeader);
    const size_t crc_len = total_payload - sizeof(NVMeNodeHeader);
    const size_t qwords = crc_len / 8;
    const uint64_t* qp = reinterpret_cast<const uint64_t*>(payload_bytes);
    for (size_t qi = 0; qi < qwords; ++qi) {
#if defined(__SSE4_2__)
        crc = _mm_crc32_u64(crc, qp[qi]);
#else
        crc = (crc ^ qp[qi]) * 1099511628211ULL;
#endif
    }
    reinterpret_cast<NVMeNodeHeader*>(buffer.data())->checksum = crc;

    node->is_staging_to_nvme = true;
    {
        std::lock_guard<std::mutex> lk(nvme_queue_mutex_);
        nvme_queue_.push(NVMeStagingTask{
            node->id,
            node,
            std::move(buffer),
            total_payload
        });
    }
    nvme_cv_.notify_one();
    ++cached_nvme_snapshots_;
    return true;
}

size_t RadixTree::prune_nvme_lru_locked(size_t max_bytes) {
    if (!nvme_tier_) return 0;
    size_t total_l3 = 0;
    std::vector<std::shared_ptr<RadixNode>> l3_nodes;
    std::vector<std::shared_ptr<RadixNode>> stack;
    stack.push_back(root_);
    while (!stack.empty()) {
        auto node = stack.back();
        stack.pop_back();
        if (!node) continue;
        if (node->is_l3_offloaded) {
            total_l3 += node->l3_bytes;
            l3_nodes.push_back(node);
        }
        for (const auto& kv : node->children) stack.push_back(kv.second);
    }

    if (total_l3 <= max_bytes) return 0;

    // Prune oldest non-permanent nodes first
    std::sort(l3_nodes.begin(), l3_nodes.end(), [](const auto& a, const auto& b) {
        if (a->is_permanent != b->is_permanent) return !a->is_permanent; // non-permanent first
        return a->last_accessed < b->last_accessed;
    });

    size_t pruned = 0;
    for (auto& node : l3_nodes) {
        if (total_l3 <= max_bytes) break;
        if (node->ref_count > 0 || node->is_permanent) continue;
        size_t b = node->l3_bytes;
        node->free_nvme(*nvme_tier_);
        total_l3 -= b;
        if (cached_nvme_snapshots_ > 0) --cached_nvme_snapshots_;
        ++pruned;
    }
    return pruned;
}

size_t RadixTree::host_snapshot_bytes() const {
    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    return host_snapshot_bytes_locked();
}

// S26f: evict (LRU, parking to host first) until the budget can hold `need_bytes`.
size_t RadixTree::evict_to_budget(size_t need_bytes) {
    size_t evicted = 0;
    if (vram_budget_ == 0) return 0;
    while (device_snapshot_bytes_locked() + need_bytes > vram_budget_) {
        const size_t before = cached_snapshots_;
        evict_lru_locked(before > 0 ? before - 1 : 0);
        if (cached_snapshots_ >= before) {
            // If leaves were empty or unreferenced leaves were not available, break to avoid infinite loop
            break;
        }
        ++evicted;
    }
    return evicted;
}

RadixMatch RadixTree::match_prefix(const int32_t* tokens, size_t n) const {
    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    RadixMatch best{nullptr, 0};
    std::shared_ptr<RadixNode> curr = root_;
    size_t matched_len = 0;

    while (curr && matched_len < n) {
        const int32_t next_tok = tokens[matched_len];
        auto it = curr->children.find(next_tok);
        if (it == curr->children.end()) break;

        auto child = it->second;
        const auto& edge = child->edge_tokens;
        const auto& chunk_hashes = child->edge_chunk_hashes;
        size_t match_edge = 0;
        const size_t max_comp = std::min(edge.size(), n - matched_len);
        const size_t num_chunks = max_comp / kRadixChunkTokens;

        // Task 2.1: 64-token chunk hash comparison
        size_t c = 0;
        for (; c < num_chunks && c < chunk_hashes.size(); ++c) {
            const uint64_t q_hash = compute_token_chunk_hash(tokens + matched_len + c * kRadixChunkTokens, kRadixChunkTokens);
            if (chunk_hashes[c] != q_hash) break;
            match_edge += kRadixChunkTokens;
        }

        // Remainder scalar comparison
        while (match_edge < max_comp && edge[match_edge] == tokens[matched_len + match_edge]) {
            ++match_edge;
        }

        if (match_edge == edge.size()) {
            matched_len += edge.size();
            curr = child;
            // P18 L1: a node is fork-usable only when the query extends at least one token past its
            // coverage (the reused prefix must not pass the query's last index); a shorter hit falls
            // back to the next-shallower snapshotted ancestor already held in `best`.
            if (curr->has_snapshot() && (int64_t) n - 1 >= (int64_t) matched_len) {
                best = RadixMatch{curr, (int64_t) matched_len};
            }
        } else {
            // Disallow mid-edge partial snapshot restore for hybrid recurrent models (GDN + QSA + PLE).
            // A snapshot saved at prefix_len holds recurrent states (gdn_saved, ple_saved, R_saved)
            // that cannot be unrolled backward. Reusing a partial edge hit causes recurrent state
            // pollution from ghost tokens of divergent requests. Exact node snapshot match is required.
            break;
        }
    }
    return best;
}

RadixMatch RadixTree::match_prefix_overlap(const int32_t* tokens, size_t n, float* out_overlap_ratio) const {
    auto match = match_prefix(tokens, n);
    if (out_overlap_ratio) {
        *out_overlap_ratio = (n > 0) ? (static_cast<float>(match.matched_tokens) / static_cast<float>(n)) : 0.0f;
    }
    return match;
}

RadixMatch RadixTree::match_prefix(const int64_t* tokens, size_t n) const {
    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    RadixMatch best{nullptr, 0};
    std::shared_ptr<RadixNode> curr = root_;
    size_t matched_len = 0;

    while (curr && matched_len < n) {
        const int32_t next_tok = (int32_t) tokens[matched_len];
        auto it = curr->children.find(next_tok);
        if (it == curr->children.end()) break;

        auto child = it->second;
        const auto& edge = child->edge_tokens;
        const auto& chunk_hashes = child->edge_chunk_hashes;
        size_t match_edge = 0;
        const size_t max_comp = std::min(edge.size(), n - matched_len);
        const size_t num_chunks = max_comp / kRadixChunkTokens;

        // Task 2.1: 64-token chunk hash comparison
        size_t c = 0;
        for (; c < num_chunks && c < chunk_hashes.size(); ++c) {
            const uint64_t q_hash = compute_token_chunk_hash(tokens + matched_len + c * kRadixChunkTokens, kRadixChunkTokens);
            if (chunk_hashes[c] != q_hash) break;
            match_edge += kRadixChunkTokens;
        }

        // Remainder scalar comparison
        while (match_edge < max_comp && (int64_t) edge[match_edge] == tokens[matched_len + match_edge]) {
            ++match_edge;
        }

        if (match_edge == edge.size()) {
            matched_len += edge.size();
            curr = child;
            // P18 L1: a node is fork-usable only when the query extends at least one token past its
            // coverage (the reused prefix must not pass the query's last index); a shorter hit falls
            // back to the next-shallower snapshotted ancestor already held in `best`.
            if (curr->has_snapshot() && (int64_t) n - 1 >= (int64_t) matched_len) {
                best = RadixMatch{curr, (int64_t) matched_len};
            }
        } else {
            // Disallow mid-edge partial snapshot restore for hybrid recurrent models (GDN + QSA + PLE).
            // A snapshot saved at prefix_len holds recurrent states (gdn_saved, ple_saved, R_saved)
            // that cannot be unrolled backward. Reusing a partial edge hit causes recurrent state
            // pollution from ghost tokens of divergent requests. Exact node snapshot match is required.
            break;
        }
    }
    return best;
}

std::shared_ptr<RadixNode> RadixTree::insert(
    const int32_t* tokens,
    size_t n,
    int64_t prefix_len,
    const std::vector<int>& stage_devices,
    const std::vector<const SessionState*>& states,
    const std::vector<const float*>& R_ptrs,
    const ModelGeometry& g,
    const std::vector<void*>& streams,
    bool safe_no_evict) {

    if (prefix_len < 64 || (size_t) prefix_len > n ||
        states.size() != stage_devices.size() || streams.size() != stage_devices.size()) {
        return nullptr;
    }

    std::unique_lock<std::shared_mutex> lock(rw_lock_);
    std::shared_ptr<RadixNode> curr = root_;
    size_t matched_len = 0;

    while (matched_len < (size_t) prefix_len) {
        const int32_t next_tok = tokens[matched_len];
        auto it = curr->children.find(next_tok);
        if (it == curr->children.end()) {
            auto new_node = std::make_shared<RadixNode>();
            new_node->id = next_node_id_++;
            new_node->prefix_len = (int64_t) prefix_len;
            new_node->edge_tokens.assign(tokens + matched_len, tokens + prefix_len);
            new_node->update_chunk_hashes();
            new_node->parent = curr;
            new_node->last_accessed = std::chrono::steady_clock::now();
            curr->children[next_tok] = new_node;
            ++node_count_;
            curr = new_node;
            matched_len = (size_t) prefix_len;
            break;
        }

        auto child = it->second;
        const auto& edge = child->edge_tokens;
        size_t match_edge = 0;
        const size_t max_comp = std::min(edge.size(), (size_t) prefix_len - matched_len);
        while (match_edge < max_comp && edge[match_edge] == tokens[matched_len + match_edge]) {
            ++match_edge;
        }

        if (match_edge == edge.size()) {
            matched_len += edge.size();
            curr = child;
        } else {
            auto split_node = std::make_shared<RadixNode>();
            split_node->id = next_node_id_++;
            split_node->prefix_len = curr->prefix_len + (int64_t) match_edge;
            split_node->edge_tokens.assign(edge.begin(), edge.begin() + match_edge);
            split_node->update_chunk_hashes();
            split_node->parent = curr;
            split_node->last_accessed = std::chrono::steady_clock::now();

            child->edge_tokens.erase(child->edge_tokens.begin(), child->edge_tokens.begin() + match_edge);
            child->update_chunk_hashes();
            child->parent = split_node;
            split_node->children[child->edge_tokens[0]] = child;

            curr->children[next_tok] = split_node;
            ++node_count_;

            matched_len += match_edge;

            if (matched_len == (size_t) prefix_len) {
                curr = split_node;
            } else {
                auto new_node = std::make_shared<RadixNode>();
                new_node->id = next_node_id_++;
                new_node->prefix_len = (int64_t) prefix_len;
                new_node->edge_tokens.assign(tokens + matched_len, tokens + prefix_len);
                new_node->update_chunk_hashes();
                new_node->parent = split_node;
                new_node->last_accessed = std::chrono::steady_clock::now();
                split_node->children[tokens[matched_len]] = new_node;
                ++node_count_;
                curr = new_node;
                matched_len = (size_t) prefix_len;
            }
            break;
        }
    }

    curr->last_accessed = std::chrono::steady_clock::now();

    // S26f: byte-budget the device snapshot tier. Estimate the capture size; evict LRU (park to
    // host) until it fits; if it still cannot fit, keep the path but refuse the capture (a clean
    // skip beats a silent partial allocation against the VRAM margin).
    if (!curr->has_snapshot() && vram_budget_ > 0) {
        size_t est = 0;
        strata::kernels::QsaShapes qs_e = strata::kernels::qsa_real_shapes();
        qs_e.n_head = g.n_head; qs_e.n_head_kv = g.n_head_kv; qs_e.head_dim = g.head_dim;
        const int64_t pages_e = ((int64_t) prefix_len + qs_e.page_size - 1) / qs_e.page_size;
        const size_t qsa_rows_e = (size_t) pages_e * qs_e.n_head_kv * qs_e.page_size;
        for (size_t st = 0; st < stage_devices.size(); ++st) {
            const auto* s_e = states[st];
            if (!s_e) continue;
            est += (size_t) s_e->qsa_alloc * 2 * qsa_rows_e * strata::kernels::kv_q4_bytes_per_head((int) qs_e.head_dim);
            est += (size_t) core::ple_hist_bytes() + (size_t) g.hc * g.n_embd * sizeof(float);
            if (s_e->gdn_alloc > 0) est += (size_t) s_e->gdn_alloc * core::gdn_state_floats(g) * sizeof(float);
        }
        if (est > vram_budget_) {
            if (std::getenv("STRATA_S26F_LOG"))
                std::fprintf(stderr, "S26F capture REFUSED(oversize): prefix=%lld est=%zu budget=%zu\n",
                            (long long) prefix_len, est, vram_budget_);
            return curr;   // S26f3: never evict working captures for one that cannot fit at all
        }
        // S26g capture-priority: shallow/mid captures leave a deep reserve inside the tier;
        // deep captures keep the full budget.  Per-byte value rises with prefix_len (a deep
        // snapshot serves every later turn of its session), and the leaf-only LRU can never
        // reclaim the interior nodes the early shallow/mid flood creates - so admission is
        // the only lever that keeps deep captures alive under the census arrival pattern
        // (the 33%-vs-54.6% class: P-1 proved the ceiling is the tier itself, not the gate).
        const size_t admit_cap = ((int64_t) prefix_len >= kDeepPrefixTokens)
                               ? vram_budget_
                               : (deep_reserve_ < vram_budget_ ? vram_budget_ - deep_reserve_ : 0);
        if (est > 0 && device_snapshot_bytes_locked() + est > admit_cap) {
            if (std::getenv("STRATA_S26F_LOG"))
                std::fprintf(stderr, "S26F capture REFUSED(reserve): prefix=%lld est=%zu bytes=%zu reserve=%zu budget=%zu\n",
                            (long long) prefix_len, est, device_snapshot_bytes_locked(), deep_reserve_, vram_budget_);
            return curr;   // S26g: shallow/mid capture refused to preserve deep headroom
        }
        if (est > 0 && device_snapshot_bytes_locked() + est > vram_budget_) {
            const size_t before_ev = device_snapshot_bytes_locked();
            if (!safe_no_evict) evict_to_budget(est);   // S26f7: safe inserts defer eviction to flush points
            if (device_snapshot_bytes_locked() + est > vram_budget_) {
                if (std::getenv("STRATA_S26F_LOG"))
                    std::fprintf(stderr, "S26F capture REFUSED: prefix=%lld est=%zu bytes=%zu(was %zu) budget=%zu\n",
                                (long long) prefix_len, est, device_snapshot_bytes_locked(),
                                before_ev, vram_budget_);
                return curr;   // S26f: budget refusal - path kept, capture skipped
            }
        } else if (std::getenv("STRATA_S26F_LOG")) {
            std::fprintf(stderr, "S26F capture ADMIT: prefix=%lld est=%zu bytes=%zu budget=%zu\n",
                        (long long) prefix_len, est, device_snapshot_bytes_locked(), vram_budget_);
        }
    }

    if (vram_budget_ == 0 || max_cached_snapshots_ == 0) {
        return curr;
    }

    if (!curr->has_snapshot()) {
        const size_t n_stages = stage_devices.size();
        curr->stage_snapshots.resize(n_stages);

        strata::kernels::QsaShapes qs = strata::kernels::qsa_real_shapes();
        qs.n_head = g.n_head;
        qs.n_head_kv = g.n_head_kv;
        qs.head_dim = g.head_dim;
        qs.idx_n_head = g.idx_q_heads;
        qs.idx_dim = g.idx_key_dim;

        const int64_t pages = (prefix_len + qs.page_size - 1) / qs.page_size;
        const size_t qsa_rows = (size_t) pages * qs.n_head_kv * qs.page_size;

        for (size_t st = 0; st < n_stages; ++st) {
            auto& ss = curr->stage_snapshots[st];
            ss.device = stage_devices[st];
            const core::OnDevice on(ss.device);
            cudaStream_t cs = (cudaStream_t) streams[st];
            const auto* s = states[st];

            if (s->gdn_alloc > 0 && s->gdn_state) {
                ss.gdn_bytes = (size_t) s->gdn_alloc * core::gdn_state_floats(g) * sizeof(float);
                cudaMalloc(&ss.gdn_saved, ss.gdn_bytes);
                cudaMemcpyAsync(ss.gdn_saved, s->gdn_state, ss.gdn_bytes, cudaMemcpyDeviceToDevice, cs);
            }

            if (s->ple_hist) {
                ss.ple_bytes = (size_t) core::ple_hist_bytes();
                cudaMalloc(&ss.ple_saved, ss.ple_bytes);
                cudaMemcpyAsync(ss.ple_saved, s->ple_hist, ss.ple_bytes, cudaMemcpyDeviceToDevice, cs);
                ss.ple_prev_saved[0] = s->ple_prev[0];
                ss.ple_prev_saved[1] = s->ple_prev[1];
                ss.ple_token_saved = s->ple_token;
            }

            const float* src_R = R_ptrs[st] ? R_ptrs[st] : s->block.R;
            if (src_R) {
                ss.R_bytes = (size_t) g.hc * g.n_embd * sizeof(float);
                cudaMalloc(&ss.R_saved, ss.R_bytes);
                cudaMemcpyAsync(ss.R_saved, src_R, ss.R_bytes, cudaMemcpyDeviceToDevice, cs);
            }

            ss.qsa_ord0 = s->qsa_ord0;
            ss.qsa_alloc = s->qsa_alloc;
            ss.qsa_slices.resize(s->qsa_alloc);

            for (int64_t j = 0; j < s->qsa_alloc; ++j) {
                const QsaState& qst = s->qsa_states[s->qsa_ord0 + j];
                auto& slice = ss.qsa_slices[j];

                if (qst.kv_q4) {
                    slice.k_bytes = qsa_rows * strata::kernels::kv_q4_bytes_per_head((int) qs.head_dim);
                    slice.v_bytes = slice.k_bytes;
                    if (qst.k_q4) {
                        cudaMalloc(&slice.k_q, slice.k_bytes);
                        cudaMemcpyAsync(slice.k_q, qst.k_q4, slice.k_bytes, cudaMemcpyDeviceToDevice, cs);
                    }
                    if (qst.v_q4) {
                        cudaMalloc(&slice.v_q4, slice.v_bytes);
                        cudaMemcpyAsync(slice.v_q4, qst.v_q4, slice.v_bytes, cudaMemcpyDeviceToDevice, cs);
                    }
                } else if (qst.kv_hybrid) {
                    slice.k_bytes = qsa_rows * qs.head_dim;
                    slice.k_scale_bytes = qsa_rows * (qs.head_dim / strata::kernels::KV_Q8_GROUP) * 2;
                    slice.v_bytes = qsa_rows * strata::kernels::kv_q4_bytes_per_head((int) qs.head_dim);
                    if (qst.k_q) {
                        cudaMalloc(&slice.k_q, slice.k_bytes);
                        cudaMemcpyAsync(slice.k_q, qst.k_q, slice.k_bytes, cudaMemcpyDeviceToDevice, cs);
                    }
                    if (qst.k_scale) {
                        cudaMalloc(&slice.k_scale, slice.k_scale_bytes);
                        cudaMemcpyAsync(slice.k_scale, qst.k_scale, slice.k_scale_bytes, cudaMemcpyDeviceToDevice, cs);
                    }
                    if (qst.v_q4) {
                        cudaMalloc(&slice.v_q4, slice.v_bytes);
                        cudaMemcpyAsync(slice.v_q4, qst.v_q4, slice.v_bytes, cudaMemcpyDeviceToDevice, cs);
                    }
                } else if (qst.kv_int8) {
                    slice.k_bytes = qsa_rows * qs.head_dim;
                    slice.k_scale_bytes = qsa_rows * (qs.head_dim / strata::kernels::KV_Q8_GROUP) * 2;
                    if (qst.k_q) {
                        cudaMalloc(&slice.k_q, slice.k_bytes);
                        cudaMemcpyAsync(slice.k_q, qst.k_q, slice.k_bytes, cudaMemcpyDeviceToDevice, cs);
                    }
                    if (qst.v_q) {
                        cudaMalloc(&slice.v_q4, slice.k_bytes);
                        cudaMemcpyAsync(slice.v_q4, qst.v_q, slice.k_bytes, cudaMemcpyDeviceToDevice, cs);
                    }
                    if (qst.k_scale) {
                        cudaMalloc(&slice.k_scale, slice.k_scale_bytes);
                        cudaMemcpyAsync(slice.k_scale, qst.k_scale, slice.k_scale_bytes, cudaMemcpyDeviceToDevice, cs);
                    }
                    if (qst.v_scale) {
                        cudaMalloc(&slice.v_scale, slice.k_scale_bytes);
                        cudaMemcpyAsync(slice.v_scale, qst.v_scale, slice.k_scale_bytes, cudaMemcpyDeviceToDevice, cs);
                    }
                }

                slice.idx_tail_bytes = (size_t) (qs.idx_block - 1) * qs.idx_dim * sizeof(float);
                slice.idx_dead_bytes = (size_t) qs.idx_dim * sizeof(float);
                if (qst.idx_tail) {
                    cudaMalloc(&slice.idx_tail, slice.idx_tail_bytes);
                    cudaMemcpyAsync(slice.idx_tail, qst.idx_tail, slice.idx_tail_bytes, cudaMemcpyDeviceToDevice, cs);
                }
                if (qst.idx_dead) {
                    cudaMalloc(&slice.idx_dead, slice.idx_dead_bytes);
                    cudaMemcpyAsync(slice.idx_dead, qst.idx_dead, slice.idx_dead_bytes, cudaMemcpyDeviceToDevice, cs);
                }
                if (qst.idx_block_pos) {
                    cudaMalloc(&slice.idx_block_pos, sizeof(int32_t));
                    cudaMemcpyAsync(slice.idx_block_pos, qst.idx_block_pos, sizeof(int32_t), cudaMemcpyDeviceToDevice, cs);
                }
                const int64_t pooled_rows = prefix_len / qs.idx_block;
                if (pooled_rows > 0 && qst.idx_pooled) {
                    slice.idx_pooled_bytes = (size_t) (pooled_rows + 1) * qs.idx_dim * sizeof(float);
                    cudaMalloc(&slice.idx_pooled, slice.idx_pooled_bytes);
                    cudaMemcpyAsync(slice.idx_pooled, qst.idx_pooled, (size_t) pooled_rows * qs.idx_dim * sizeof(float), cudaMemcpyDeviceToDevice, cs);
                    cudaMemcpyAsync((char*) slice.idx_pooled + pooled_rows * qs.idx_dim * sizeof(float),
                                    qst.idx_dead, slice.idx_dead_bytes, cudaMemcpyDeviceToDevice, cs);
                }
            }
        }
        for (size_t st = 0; st < n_stages; ++st) {
            const core::OnDevice on(stage_devices[st]);
            cudaStreamSynchronize((cudaStream_t) streams[st]);
        }
        // S26f: verify the capture actually landed (cudaMalloc OOMs are otherwise silent here).
        // A failed capture frees its partial allocations and leaves the path snapshot-less.
        bool snap_ok = true;
        for (size_t st = 0; st < n_stages && snap_ok; ++st) {
            auto& ss = curr->stage_snapshots[st];
            const auto* s = states[st];
            if (!s) continue;
            if (s->gdn_alloc > 0 && s->gdn_state && !ss.gdn_saved) snap_ok = false;
            if (s->ple_hist && !ss.ple_saved) snap_ok = false;
            for (int64_t j = 0; j < s->qsa_alloc && snap_ok; ++j) {
                const QsaState& qst = s->qsa_states[s->qsa_ord0 + j];
                auto& sl = ss.qsa_slices[j];
                if ((qst.k_q4 || qst.k_q) && !sl.k_q) snap_ok = false;
                if ((qst.v_q4 || qst.v_q) && !sl.v_q4) snap_ok = false;
            }
        }
        if (!snap_ok) {
            if (std::getenv("STRATA_S26F_LOG"))
                std::fprintf(stderr, "S26F capture FAILED: prefix=%lld (alloc failure; freed partial)\n",
                            (long long) prefix_len);
            for (auto& ss2 : curr->stage_snapshots) ss2.free_device();
            curr->stage_snapshots.clear();
            return curr;   // S26f: failed capture -> path only, never a broken snapshot
        }
        if (std::getenv("STRATA_S26F_LOG"))
            std::fprintf(stderr, "S26F capture OK: prefix=%lld node=%lld bytes_now=%zu\n",
                        (long long) prefix_len, (long long) curr->id, device_snapshot_bytes_locked());
        if (safe_no_evict) {
            curr->is_permanent = true;
            curr->kind = RadixNodeKind::SYSTEM_ROOT;
        }
        ++cached_snapshots_;
        if (!safe_no_evict) evict_lru_locked(max_cached_snapshots_);   // S26f7
    }
    return curr;
}

void RadixTree::acquire(const std::shared_ptr<RadixNode>& node) {
    if (node) ++node->ref_count;
}

void RadixTree::release(const std::shared_ptr<RadixNode>& node) {
    if (node && node->ref_count > 0) --node->ref_count;
}

bool RadixTree::fork_to_session(
    const std::shared_ptr<RadixNode>& node,
    int64_t prefix_len,
    const std::vector<int>& stage_devices,
    std::vector<SessionState*>& child_states,
    const ModelGeometry& g,
    const std::vector<void*>& streams,
    std::string& err) {

    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    if (!node || !node->has_snapshot()) {
        err = "radix_fork: node has no snapshot";
        return false;
    }

    if (node->is_l3_offloaded && nvme_tier_) {
        if (!node->hydrate_from_nvme(*nvme_tier_)) {
            err = "radix_fork: failed to hydrate node from NVMe L3";
            return false;
        }
    }

    // Mandate 3: Direct Host-to-Session DMA Fork (Zero-VRAM Restore)
    // Eliminate intermediate node->unpark_to_device() allocations!
    // Directly stream from Host PinnedBuffer to child_states over PCIe Gen 4 x16 streams.

    const size_t n_stages = stage_devices.size();
    strata::kernels::QsaShapes qs = strata::kernels::qsa_real_shapes();
    qs.n_head = g.n_head;
    qs.n_head_kv = g.n_head_kv;
    qs.head_dim = g.head_dim;
    qs.idx_n_head = g.idx_q_heads;
    qs.idx_dim = g.idx_key_dim;

    const int64_t pages = (prefix_len + qs.page_size - 1) / qs.page_size;
    const size_t qsa_rows = (size_t) pages * qs.n_head_kv * qs.page_size;

    if (node->has_device_snapshot()) {
        for (size_t st = 0; st < n_stages && st < node->stage_snapshots.size(); ++st) {
            const auto& ss = node->stage_snapshots[st];
            auto* child = child_states[st];
            const core::OnDevice on(ss.device);
            cudaStream_t cs = (cudaStream_t) streams[st];

            if (ss.R_saved && child->block.R) {
                cudaMemcpyAsync(child->block.R, ss.R_saved, ss.R_bytes, cudaMemcpyDeviceToDevice, cs);
            }

            if (ss.gdn_saved && child->gdn_state) {
                cudaMemcpyAsync(child->gdn_state, ss.gdn_saved, ss.gdn_bytes, cudaMemcpyDeviceToDevice, cs);
            }

            if (ss.ple_saved && child->ple_hist) {
                cudaMemcpyAsync(child->ple_hist, ss.ple_saved, ss.ple_bytes, cudaMemcpyDeviceToDevice, cs);
                child->ple_prev[0] = ss.ple_prev_saved[0];
                child->ple_prev[1] = ss.ple_prev_saved[1];
                child->ple_token = ss.ple_token_saved;
            }

            for (int64_t j = 0; j < child->qsa_alloc && j < (int64_t) ss.qsa_slices.size(); ++j) {
                const auto& slice = ss.qsa_slices[j];
                QsaState& cst = child->qsa_states[child->qsa_ord0 + j];

                if (cst.kv_q4) {
                    const size_t bytes = qsa_rows * strata::kernels::kv_q4_bytes_per_head((int) qs.head_dim);
                    if (slice.k_q && cst.k_q4) cudaMemcpyAsync(cst.k_q4, slice.k_q, bytes, cudaMemcpyDeviceToDevice, cs);
                    if (slice.v_q4 && cst.v_q4) cudaMemcpyAsync(cst.v_q4, slice.v_q4, bytes, cudaMemcpyDeviceToDevice, cs);
                } else if (cst.kv_hybrid) {
                    const size_t k_bytes = qsa_rows * qs.head_dim;
                    const size_t sc_bytes = qsa_rows * (qs.head_dim / strata::kernels::KV_Q8_GROUP) * 2;
                    const size_t v_bytes = qsa_rows * strata::kernels::kv_q4_bytes_per_head((int) qs.head_dim);
                    if (slice.k_q && cst.k_q) cudaMemcpyAsync(cst.k_q, slice.k_q, k_bytes, cudaMemcpyDeviceToDevice, cs);
                    if (slice.k_scale && cst.k_scale) cudaMemcpyAsync(cst.k_scale, slice.k_scale, sc_bytes, cudaMemcpyDeviceToDevice, cs);
                    if (slice.v_q4 && cst.v_q4) cudaMemcpyAsync(cst.v_q4, slice.v_q4, v_bytes, cudaMemcpyDeviceToDevice, cs);
                } else if (cst.kv_int8) {
                    const size_t bytes = qsa_rows * qs.head_dim;
                    const size_t sc_bytes = qsa_rows * (qs.head_dim / strata::kernels::KV_Q8_GROUP) * 2;
                    if (slice.k_q && cst.k_q) cudaMemcpyAsync(cst.k_q, slice.k_q, bytes, cudaMemcpyDeviceToDevice, cs);
                    if (slice.v_q4 && cst.v_q) cudaMemcpyAsync(cst.v_q, slice.v_q4, bytes, cudaMemcpyDeviceToDevice, cs);
                    if (slice.k_scale && cst.k_scale) cudaMemcpyAsync(cst.k_scale, slice.k_scale, sc_bytes, cudaMemcpyDeviceToDevice, cs);
                    if (slice.v_scale && cst.v_scale) cudaMemcpyAsync(cst.v_scale, slice.v_scale, sc_bytes, cudaMemcpyDeviceToDevice, cs);
                }

                if (slice.idx_tail && cst.idx_tail) {
                    cudaMemcpyAsync(cst.idx_tail, slice.idx_tail, slice.idx_tail_bytes, cudaMemcpyDeviceToDevice, cs);
                }
                if (slice.idx_dead && cst.idx_dead) {
                    cudaMemcpyAsync(cst.idx_dead, slice.idx_dead, slice.idx_dead_bytes, cudaMemcpyDeviceToDevice, cs);
                }
                if (slice.idx_block_pos && cst.idx_block_pos) {
                    cudaMemcpyAsync(cst.idx_block_pos, slice.idx_block_pos, sizeof(int32_t), cudaMemcpyDeviceToDevice, cs);
                }
                const int64_t pooled_rows = prefix_len / qs.idx_block;
                if (pooled_rows > 0 && slice.idx_pooled && cst.idx_pooled) {
                    cudaMemcpyAsync(cst.idx_pooled, slice.idx_pooled, slice.idx_pooled_bytes, cudaMemcpyDeviceToDevice, cs);
                }
            }
        }
    } else if (node->has_host_snapshot()) {
        for (size_t st = 0; st < n_stages && st < node->stage_host_snapshots.size(); ++st) {
            const auto& hss = node->stage_host_snapshots[st];
            auto* child = child_states[st];
            const core::OnDevice on(hss.device);
            cudaStream_t cs = (cudaStream_t) streams[st];

            if (!hss.R_data.empty() && child->block.R) {
                cudaMemcpyAsync(child->block.R, hss.R_data.data(), hss.R_data.size(), cudaMemcpyHostToDevice, cs);
            }

            if (!hss.gdn_data.empty() && child->gdn_state) {
                cudaMemcpyAsync(child->gdn_state, hss.gdn_data.data(), hss.gdn_data.size(), cudaMemcpyHostToDevice, cs);
            }

            if (!hss.ple_data.empty() && child->ple_hist) {
                cudaMemcpyAsync(child->ple_hist, hss.ple_data.data(), hss.ple_data.size(), cudaMemcpyHostToDevice, cs);
                child->ple_prev[0] = hss.ple_prev_saved[0];
                child->ple_prev[1] = hss.ple_prev_saved[1];
                child->ple_token = hss.ple_token_saved;
            }

            for (int64_t j = 0; j < child->qsa_alloc && j < (int64_t) hss.qsa_slices.size(); ++j) {
                const auto& slice = hss.qsa_slices[j];
                QsaState& cst = child->qsa_states[child->qsa_ord0 + j];

                if (cst.kv_q4) {
                    const size_t bytes = qsa_rows * strata::kernels::kv_q4_bytes_per_head((int) qs.head_dim);
                    if (!slice.k_q.empty() && cst.k_q4) cudaMemcpyAsync(cst.k_q4, slice.k_q.data(), bytes, cudaMemcpyHostToDevice, cs);
                    if (!slice.v_q4.empty() && cst.v_q4) cudaMemcpyAsync(cst.v_q4, slice.v_q4.data(), bytes, cudaMemcpyHostToDevice, cs);
                } else if (cst.kv_hybrid) {
                    const size_t k_bytes = qsa_rows * qs.head_dim;
                    const size_t sc_bytes = qsa_rows * (qs.head_dim / strata::kernels::KV_Q8_GROUP) * 2;
                    const size_t v_bytes = qsa_rows * strata::kernels::kv_q4_bytes_per_head((int) qs.head_dim);
                    if (!slice.k_q.empty() && cst.k_q) cudaMemcpyAsync(cst.k_q, slice.k_q.data(), k_bytes, cudaMemcpyHostToDevice, cs);
                    if (!slice.k_scale.empty() && cst.k_scale) cudaMemcpyAsync(cst.k_scale, slice.k_scale.data(), sc_bytes, cudaMemcpyHostToDevice, cs);
                    if (!slice.v_q4.empty() && cst.v_q4) cudaMemcpyAsync(cst.v_q4, slice.v_q4.data(), v_bytes, cudaMemcpyHostToDevice, cs);
                } else if (cst.kv_int8) {
                    const size_t bytes = qsa_rows * qs.head_dim;
                    const size_t sc_bytes = qsa_rows * (qs.head_dim / strata::kernels::KV_Q8_GROUP) * 2;
                    if (!slice.k_q.empty() && cst.k_q) cudaMemcpyAsync(cst.k_q, slice.k_q.data(), bytes, cudaMemcpyHostToDevice, cs);
                    if (!slice.v_q4.empty() && cst.v_q) cudaMemcpyAsync(cst.v_q, slice.v_q4.data(), bytes, cudaMemcpyHostToDevice, cs);
                    if (!slice.k_scale.empty() && cst.k_scale) cudaMemcpyAsync(cst.k_scale, slice.k_scale.data(), sc_bytes, cudaMemcpyHostToDevice, cs);
                    if (!slice.v_scale.empty() && cst.v_scale) cudaMemcpyAsync(cst.v_scale, slice.v_scale.data(), sc_bytes, cudaMemcpyHostToDevice, cs);
                }

                if (!slice.idx_tail.empty() && cst.idx_tail) {
                    cudaMemcpyAsync(cst.idx_tail, slice.idx_tail.data(), slice.idx_tail.size(), cudaMemcpyHostToDevice, cs);
                }
                if (!slice.idx_dead.empty() && cst.idx_dead) {
                    cudaMemcpyAsync(cst.idx_dead, slice.idx_dead.data(), slice.idx_dead.size(), cudaMemcpyHostToDevice, cs);
                }
                if (slice.has_idx_block_pos && cst.idx_block_pos) {
                    cudaMemcpyAsync(cst.idx_block_pos, &slice.idx_block_pos_val, sizeof(int32_t), cudaMemcpyHostToDevice, cs);
                }
                const int64_t pooled_rows = prefix_len / qs.idx_block;
                if (pooled_rows > 0 && !slice.idx_pooled.empty() && cst.idx_pooled) {
                    cudaMemcpyAsync(cst.idx_pooled, slice.idx_pooled.data(), slice.idx_pooled.size(), cudaMemcpyHostToDevice, cs);
                }
            }
        }
    }
    node->last_accessed = std::chrono::steady_clock::now();
    lock.unlock(); // Drop rw_lock_ before host-blocking cudaStreamSynchronize to eliminate scheduler deadlock

    // Task 2.4: Asynchronous Multi-Stage KV Forking across concurrent CUDA streams
    // Stage 0 and Stage 1 DMA memcpy operations overlap concurrently across streams[0] and streams[1].
    for (size_t st = 0; st < n_stages; ++st) {
        const core::OnDevice on(stage_devices[st]);
        if (cudaStreamSynchronize((cudaStream_t) streams[st]) != cudaSuccess) {
            err = "radix_fork: cudaStreamSynchronize failed on stage " + std::to_string(st);
            return false;
        }
    }
    return true;
}

void RadixTree::collect_unreferenced_leaves(
    const std::shared_ptr<RadixNode>& curr,
    std::vector<std::shared_ptr<RadixNode>>& leaves) {
    if (!curr) return;
    // Task 2.1: Decoupled Interior VRAM Eviction:
    // Any node with ref_count == 0 holding a device snapshot is evictable to L2 Host RAM,
    // even if it has active child branches!
    if (curr != root_ && curr->ref_count == 0 && curr->has_device_snapshot()) {
        leaves.push_back(curr);
    } else if (curr != root_ && curr->ref_count == 0 && curr->children.empty()) {
        leaves.push_back(curr);
        return;
    }
    for (const auto& kv : curr->children) {
        collect_unreferenced_leaves(kv.second, leaves);
    }
}

void RadixTree::collect_unreferenced_host_nodes(
    const std::shared_ptr<RadixNode>& curr,
    std::vector<std::shared_ptr<RadixNode>>& nodes) {
    if (!curr) return;
    // SGLang / vLLM Standard: Leaf-Pruning Tree-LRU.
    // An interior/parent node (system prompts, early turns) CANNOT be evicted while child nodes exist.
    // In addition, SYSTEM_ROOT nodes and permanent boundary nodes are strictly LRU-exempt.
    if (curr != root_ && curr->ref_count == 0 && curr->has_host_snapshot() && !curr->is_permanent && curr->kind != RadixNodeKind::SYSTEM_ROOT) {
        nodes.push_back(curr);
    }
    for (const auto& kv : curr->children) {
        collect_unreferenced_host_nodes(kv.second, nodes);
    }
}

size_t RadixTree::evict_lru(size_t max_snapshots, size_t min_free_vram_mib) {
    std::unique_lock<std::shared_mutex> lock(rw_lock_);
    return evict_lru_locked(max_snapshots, min_free_vram_mib);
}

size_t RadixTree::evict_lru_locked(size_t max_snapshots, size_t min_free_vram_mib) {
    size_t evicted = 0;
    auto should_evict = [&]() -> bool {
        if (cached_snapshots_ > max_snapshots) return true;
        if (min_free_vram_mib > 0) {
            size_t free0 = 0, tot0 = 0, free1 = 0, tot1 = 0;
            { const core::OnDevice on(0); cudaMemGetInfo(&free0, &tot0); }
            { const core::OnDevice on(1); cudaMemGetInfo(&free1, &tot1); }
            const size_t reserve_bytes = min_free_vram_mib * 1048576ULL;
            if (free0 < reserve_bytes || free1 < reserve_bytes) return true;
        }
        return false;
    };

    // Tier 1: Park excess VRAM snapshots to HiCache L2 Host-RAM
    while (cached_snapshots_ > 0 && should_evict()) {
        std::vector<std::shared_ptr<RadixNode>> leaves;
        collect_unreferenced_leaves(root_, leaves);
        if (leaves.empty()) break; // All leaves actively in use

        std::sort(leaves.begin(), leaves.end(), [](const auto& a, const auto& b) {
            return a->last_accessed < b->last_accessed;
        });

        std::shared_ptr<RadixNode> victim = nullptr;
        for (const auto& leaf : leaves) {
            if (leaf->has_device_snapshot()) {
                victim = leaf;
                break;
            }
        }
        if (!victim) break;

        victim->park_to_host();
        if (cached_snapshots_ > 0) --cached_snapshots_;
        ++cached_host_snapshots_;
        ++evicted;
    }

    // Tier 2: Prune host snapshots when exceeding max_host_snapshots_ or host_ram_budget_
    // Using Decoupled Payload Eviction: Can evict host snapshots from cold interior nodes
    // while keeping routing metadata (edge_tokens, child pointers) intact.
    auto should_evict_host = [&]() -> bool {
        if (cached_host_snapshots_ > max_host_snapshots_) return true;
        if (host_ram_budget_ > 0 && host_snapshot_bytes_locked() > host_ram_budget_) return true;
        return false;
    };
    while (cached_host_snapshots_ > 0 && should_evict_host()) {
        std::vector<std::shared_ptr<RadixNode>> host_nodes;
        collect_unreferenced_host_nodes(root_, host_nodes);
        if (host_nodes.empty()) break;

        // Leaf-Pruning Tree-LRU: Prioritize unreferenced leaf nodes (children.empty())
        // over interior/parent nodes, then by oldest last_accessed.
        std::sort(host_nodes.begin(), host_nodes.end(), [](const auto& a, const auto& b) {
            bool a_leaf = a->children.empty();
            bool b_leaf = b->children.empty();
            if (a_leaf != b_leaf) return a_leaf > b_leaf; // true (leaf) comes first
            return a->last_accessed < b->last_accessed;
        });

        std::shared_ptr<RadixNode> host_victim = nullptr;
        for (const auto& nd : host_nodes) {
            if (!nd->is_staging_to_nvme.load()) {
                host_victim = nd;
                break;
            }
        }
        if (!host_victim) break;

        if (nvme_tier_ && host_victim->has_host_snapshot()) {
            // Serialize host snapshot in-memory into AlignedBuffer under lock (<10 us),
            // then delegate the NVMe disk write to the background worker thread.
            size_t total_payload = sizeof(NVMeNodeHeader);
            total_payload += host_victim->edge_tokens.size() * sizeof(int32_t);
            total_payload += host_victim->stage_host_snapshots.size() * sizeof(NVMeStageHeader);

            for (const auto& hss : host_victim->stage_host_snapshots) {
                total_payload += hss.gdn_data.size();
                total_payload += hss.ple_data.size();
                total_payload += hss.R_data.size();
                total_payload += hss.qsa_slices.size() * sizeof(NVMeQsaSliceHeader);
                for (const auto& slice : hss.qsa_slices) {
                    total_payload += slice.k_q.size();
                    total_payload += slice.v_q4.size();
                    total_payload += slice.k_scale.size();
                    total_payload += slice.v_scale.size();
                    total_payload += slice.idx_tail.size();
                    total_payload += slice.idx_dead.size();
                    total_payload += slice.idx_pooled.size();
                }
            }

            AlignedBuffer buffer(total_payload);
            uint8_t* p = buffer.data();
            size_t offset = 0;

            NVMeNodeHeader hdr;
            hdr.magic = kNVMeMagic;
            hdr.version = kNVMeVersion;
            hdr.node_id = host_victim->id;
            hdr.prefix_len = host_victim->prefix_len;
            hdr.n_edge_tokens = static_cast<uint32_t>(host_victim->edge_tokens.size());
            hdr.n_stages = static_cast<uint32_t>(host_victim->stage_host_snapshots.size());
            hdr.total_payload_bytes = total_payload;
            hdr.checksum = 0xFFFFFFFF53545241ULL;

            std::memcpy(p + offset, &hdr, sizeof(hdr));
            offset += sizeof(hdr);

            if (!host_victim->edge_tokens.empty()) {
                size_t bytes = host_victim->edge_tokens.size() * sizeof(int32_t);
                std::memcpy(p + offset, host_victim->edge_tokens.data(), bytes);
                offset += bytes;
            }

            for (const auto& hss : host_victim->stage_host_snapshots) {
                NVMeStageHeader shdr;
                shdr.device = hss.device;
                shdr.ple_prev[0] = hss.ple_prev_saved[0];
                shdr.ple_prev[1] = hss.ple_prev_saved[1];
                shdr.ple_token = hss.ple_token_saved;
                shdr.gdn_bytes = hss.gdn_data.size();
                shdr.ple_bytes = hss.ple_data.size();
                shdr.R_bytes = hss.R_data.size();
                shdr.qsa_ord0 = hss.qsa_ord0;
                shdr.qsa_alloc = hss.qsa_alloc;
                shdr.n_qsa_slices = static_cast<uint32_t>(hss.qsa_slices.size());

                std::memcpy(p + offset, &shdr, sizeof(shdr));
                offset += sizeof(shdr);

                if (shdr.gdn_bytes > 0) {
                    std::memcpy(p + offset, hss.gdn_data.data(), shdr.gdn_bytes);
                    offset += shdr.gdn_bytes;
                }
                if (shdr.ple_bytes > 0) {
                    std::memcpy(p + offset, hss.ple_data.data(), shdr.ple_bytes);
                    offset += shdr.ple_bytes;
                }
                if (shdr.R_bytes > 0) {
                    std::memcpy(p + offset, hss.R_data.data(), shdr.R_bytes);
                    offset += shdr.R_bytes;
                }

                for (const auto& slice : hss.qsa_slices) {
                    NVMeQsaSliceHeader qhdr;
                    qhdr.k_bytes = slice.k_q.size();
                    qhdr.v_bytes = slice.v_q4.size();
                    qhdr.k_scale_bytes = slice.k_scale.size();
                    qhdr.v_scale_bytes = slice.v_scale.size();
                    qhdr.idx_tail_bytes = slice.idx_tail.size();
                    qhdr.idx_dead_bytes = slice.idx_dead.size();
                    qhdr.idx_pooled_bytes = slice.idx_pooled.size();
                    qhdr.idx_block_pos_val = slice.idx_block_pos_val;
                    qhdr.has_idx_block_pos = slice.has_idx_block_pos ? 1 : 0;

                    std::memcpy(p + offset, &qhdr, sizeof(qhdr));
                    offset += sizeof(qhdr);

                    if (qhdr.k_bytes > 0) {
                        std::memcpy(p + offset, slice.k_q.data(), qhdr.k_bytes);
                        offset += qhdr.k_bytes;
                    }
                    if (qhdr.v_bytes > 0) {
                        std::memcpy(p + offset, slice.v_q4.data(), qhdr.v_bytes);
                        offset += qhdr.v_bytes;
                    }
                    if (qhdr.k_scale_bytes > 0) {
                        std::memcpy(p + offset, slice.k_scale.data(), qhdr.k_scale_bytes);
                        offset += qhdr.k_scale_bytes;
                    }
                    if (qhdr.v_scale_bytes > 0) {
                        std::memcpy(p + offset, slice.v_scale.data(), qhdr.v_scale_bytes);
                        offset += qhdr.v_scale_bytes;
                    }
                    if (qhdr.idx_tail_bytes > 0) {
                        std::memcpy(p + offset, slice.idx_tail.data(), qhdr.idx_tail_bytes);
                        offset += qhdr.idx_tail_bytes;
                    }
                    if (qhdr.idx_dead_bytes > 0) {
                        std::memcpy(p + offset, slice.idx_dead.data(), qhdr.idx_dead_bytes);
                        offset += qhdr.idx_dead_bytes;
                    }
                    if (qhdr.idx_pooled_bytes > 0) {
                        std::memcpy(p + offset, slice.idx_pooled.data(), qhdr.idx_pooled_bytes);
                        offset += qhdr.idx_pooled_bytes;
                    }
                }
            }

            // Task 2.3: Hardware SSE4.2 CRC32C over serialized payload
            uint64_t crc = 0xFFFFFFFF53545241ULL;
            const uint8_t* payload_bytes = buffer.data() + sizeof(NVMeNodeHeader);
            const size_t crc_len = total_payload - sizeof(NVMeNodeHeader);
            const size_t qwords = crc_len / 8;
            const uint64_t* qp = reinterpret_cast<const uint64_t*>(payload_bytes);
            for (size_t qi = 0; qi < qwords; ++qi) {
#if defined(__SSE4_2__)
                crc = _mm_crc32_u64(crc, qp[qi]);
#else
                crc = (crc ^ qp[qi]) * 1099511628211ULL;
#endif
            }
            reinterpret_cast<NVMeNodeHeader*>(buffer.data())->checksum = crc;

            host_victim->is_staging_to_nvme = true;
            {
                std::lock_guard<std::mutex> lk(nvme_queue_mutex_);
                nvme_queue_.push(NVMeStagingTask{
                    host_victim->id,
                    host_victim,
                    std::move(buffer),
                    total_payload
                });
            }
            nvme_cv_.notify_one();
            ++cached_nvme_snapshots_;
        } else {
            host_victim->free_host();
        }
        if (cached_host_snapshots_ > 0) --cached_host_snapshots_;

        // CRIT-04: Free NVMe disk payload before removing topological edge
        if (host_victim->children.empty() && host_victim->ref_count == 0) {
            if (host_victim->is_l3_offloaded && nvme_tier_) {
                host_victim->free_nvme(*nvme_tier_);
                if (cached_nvme_snapshots_ > 0) --cached_nvme_snapshots_;
            }
            if (!host_victim->has_snapshot() && !host_victim->is_staging_to_nvme.load()) {
                if (auto p = host_victim->parent.lock()) {
                    if (!host_victim->edge_tokens.empty()) {
                        p->children.erase(host_victim->edge_tokens[0]);
                        --node_count_;
                    }
                }
            }
        }
    }
    return evicted;
}

CompactionStats RadixTree::compact_tree() {
    std::unique_lock<std::shared_mutex> lock(rw_lock_);
    CompactionStats stats;
    const auto start = std::chrono::steady_clock::now();

    size_t active_nodes = 0;
    size_t vram_bytes = 0;
    std::vector<std::shared_ptr<RadixNode>> stack = {root_};
    while (!stack.empty()) {
        auto node = stack.back();
        stack.pop_back();
        if (!node) continue;
        ++stats.nodes_scanned;
        if (node->ref_count > 0 || node->has_snapshot()) {
            ++active_nodes;
            vram_bytes += node->total_vram_bytes();
        }
        for (const auto& kv : node->children) {
            if (kv.second) stack.push_back(kv.second);
        }
    }
    stats.initial_fragmentation = RadixCompactor::compute_fragmentation(
        active_nodes, node_count_ > 0 ? node_count_ : 1,
        vram_bytes, (vram_bytes > 0 ? vram_bytes : 1) * 2);

    std::function<void(std::shared_ptr<RadixNode>)> prune_dead_leaves = [&](std::shared_ptr<RadixNode> curr) {
        if (!curr) return;
        std::vector<int32_t> dead_keys;
        for (auto& kv : curr->children) {
            prune_dead_leaves(kv.second);
            if (kv.second && kv.second->children.empty() && kv.second->ref_count == 0 && !kv.second->has_snapshot()) {
                dead_keys.push_back(kv.first);
            }
        }
        for (int32_t k : dead_keys) {
            auto victim = curr->children[k];
            if (victim) {
                stats.bytes_reclaimed += victim->total_vram_bytes() + victim->total_host_bytes();
            }
            curr->children.erase(k);
            if (node_count_ > 0) --node_count_;
            ++stats.dead_leaves_pruned;
        }
    };
    prune_dead_leaves(root_);

    std::function<void(std::shared_ptr<RadixNode>)> compress_paths = [&](std::shared_ptr<RadixNode> curr) {
        if (!curr) return;
        for (auto& kv : curr->children) {
            compress_paths(kv.second);
        }
        // Task 2.2: Assert child->ref_count == 0 before collapsing single-child path!
        if (curr != root_ && curr->ref_count == 0 && !curr->has_snapshot() && curr->children.size() == 1) {
            auto child = curr->children.begin()->second;
            if (child && child->ref_count == 0) {
                curr->edge_tokens.insert(curr->edge_tokens.end(), child->edge_tokens.begin(), child->edge_tokens.end());
                curr->update_chunk_hashes();
                curr->children = std::move(child->children);
                for (auto& c_kv : curr->children) {
                    if (c_kv.second) c_kv.second->parent = curr;
                }
                curr->stage_snapshots = std::move(child->stage_snapshots);
                curr->stage_host_snapshots = std::move(child->stage_host_snapshots);
                curr->is_host_parked = child->is_host_parked;
                curr->is_l3_offloaded = child->is_l3_offloaded;
                curr->l3_bytes = child->l3_bytes;
                curr->prefix_len = child->prefix_len;
                curr->last_accessed = std::max(curr->last_accessed, child->last_accessed);
                if (node_count_ > 0) --node_count_;
                ++stats.chains_merged;
            }
        }
    };
    compress_paths(root_);

    const auto end = std::chrono::steady_clock::now();
    stats.compaction_time_us = std::chrono::duration<double, std::micro>(end - start).count();

    active_nodes = 0;
    vram_bytes = 0;
    stack = {root_};
    while (!stack.empty()) {
        auto node = stack.back();
        stack.pop_back();
        if (!node) continue;
        if (node->ref_count > 0 || node->has_snapshot()) {
            ++active_nodes;
            vram_bytes += node->total_vram_bytes();
        }
        for (const auto& kv : node->children) {
            if (kv.second) stack.push_back(kv.second);
        }
    }
    stats.final_fragmentation = RadixCompactor::compute_fragmentation(
        active_nodes, node_count_ > 0 ? node_count_ : 1,
        vram_bytes, (vram_bytes > 0 ? vram_bytes : 1) * 2);

    return stats;
}

CompactionStats RadixCompactor::compact(RadixTree* tree) {
    std::lock_guard<std::mutex> lock(mutex_);
    RadixTree* target = tree ? tree : tree_;
    if (!target) return {};
    CompactionStats stats = target->compact_tree();
    ++total_compactions_;
    total_leaves_pruned_ += stats.dead_leaves_pruned;
    total_chains_merged_ += stats.chains_merged;
    total_bytes_reclaimed_ += stats.bytes_reclaimed;
    return stats;
}


std::shared_ptr<RadixNode> RadixTree::insert_direct_device_to_host(
    const int32_t* tokens,
    size_t n,
    int64_t prefix_len,
    const std::vector<int>& stage_devices,
    const std::vector<const SessionState*>& states,
    const std::vector<const float*>& R_ptrs,
    const ModelGeometry& g,
    const std::vector<void*>& streams,
    bool safe_no_evict) {

    if (prefix_len < 64 || (size_t) prefix_len > n ||
        states.size() != stage_devices.size() || streams.size() != stage_devices.size()) {
        return nullptr;
    }

    std::unique_lock<std::shared_mutex> lock(rw_lock_);
    std::shared_ptr<RadixNode> curr = root_;
    size_t matched_len = 0;

    while (matched_len < (size_t) prefix_len) {
        const int32_t next_tok = tokens[matched_len];
        auto it = curr->children.find(next_tok);
        if (it == curr->children.end()) {
            auto new_node = std::make_shared<RadixNode>();
            new_node->id = next_node_id_++;
            new_node->prefix_len = (int64_t) prefix_len;
            new_node->edge_tokens.assign(tokens + matched_len, tokens + prefix_len);
            new_node->update_chunk_hashes();
            new_node->parent = curr;
            new_node->last_accessed = std::chrono::steady_clock::now();
            curr->children[next_tok] = new_node;
            ++node_count_;
            curr = new_node;
            matched_len = (size_t) prefix_len;
            break;
        }

        auto child = it->second;
        const auto& edge = child->edge_tokens;
        size_t match_edge = 0;
        const size_t max_comp = std::min(edge.size(), (size_t) prefix_len - matched_len);
        while (match_edge < max_comp && edge[match_edge] == tokens[matched_len + match_edge]) {
            ++match_edge;
        }

        if (match_edge == edge.size()) {
            matched_len += edge.size();
            curr = child;
        } else {
            auto split_node = std::make_shared<RadixNode>();
            split_node->id = next_node_id_++;
            split_node->prefix_len = curr->prefix_len + (int64_t) match_edge;
            split_node->edge_tokens.assign(edge.begin(), edge.begin() + match_edge);
            split_node->update_chunk_hashes();
            split_node->parent = curr;
            split_node->last_accessed = std::chrono::steady_clock::now();

            child->edge_tokens.erase(child->edge_tokens.begin(), child->edge_tokens.begin() + match_edge);
            child->update_chunk_hashes();
            child->parent = split_node;
            split_node->children[child->edge_tokens[0]] = child;

            curr->children[next_tok] = split_node;
            ++node_count_;

            matched_len += match_edge;

            if (matched_len == (size_t) prefix_len) {
                curr = split_node;
            } else {
                auto new_node = std::make_shared<RadixNode>();
                new_node->id = next_node_id_++;
                new_node->prefix_len = (int64_t) prefix_len;
                new_node->edge_tokens.assign(tokens + matched_len, tokens + prefix_len);
                new_node->update_chunk_hashes();
                new_node->parent = split_node;
                new_node->last_accessed = std::chrono::steady_clock::now();
                split_node->children[tokens[matched_len]] = new_node;
                ++node_count_;
                curr = new_node;
                matched_len = (size_t) prefix_len;
            }
            break;
        }
    }

    curr->last_accessed = std::chrono::steady_clock::now();

    // Mandate 1 & 2: Direct-to-Host DMA Capture with Zero Persistent VRAM Allocation
    if (max_host_snapshots_ == 0) {
        return curr;
    }

    if (!curr->has_snapshot()) {
        const size_t n_stages = stage_devices.size();
        curr->stage_host_snapshots.resize(n_stages);

        strata::kernels::QsaShapes qs = strata::kernels::qsa_real_shapes();
        qs.n_head = g.n_head;
        qs.n_head_kv = g.n_head_kv;
        qs.head_dim = g.head_dim;
        qs.idx_n_head = g.idx_q_heads;
        qs.idx_dim = g.idx_key_dim;

        const int64_t pages = (prefix_len + qs.page_size - 1) / qs.page_size;
        const size_t qsa_rows = (size_t) pages * qs.n_head_kv * qs.page_size;

        for (size_t st = 0; st < n_stages; ++st) {
            auto& hss = curr->stage_host_snapshots[st];
            hss.device = stage_devices[st];
            const core::OnDevice on(hss.device);
            cudaStream_t cs = (cudaStream_t) streams[st];
            const auto* s = states[st];

            if (s->gdn_alloc > 0 && s->gdn_state) {
                const size_t gdn_bytes = (size_t) s->gdn_alloc * core::gdn_state_floats(g) * sizeof(float);
                hss.gdn_data.resize(gdn_bytes);
                cudaMemcpyAsync(hss.gdn_data.data(), s->gdn_state, gdn_bytes, cudaMemcpyDeviceToHost, cs);
            }

            if (s->ple_hist) {
                const size_t ple_bytes = (size_t) core::ple_hist_bytes();
                hss.ple_data.resize(ple_bytes);
                cudaMemcpyAsync(hss.ple_data.data(), s->ple_hist, ple_bytes, cudaMemcpyDeviceToHost, cs);
                hss.ple_prev_saved[0] = s->ple_prev[0];
                hss.ple_prev_saved[1] = s->ple_prev[1];
                hss.ple_token_saved = s->ple_token;
            }

            const float* src_R = R_ptrs[st] ? R_ptrs[st] : s->block.R;
            if (src_R) {
                const size_t R_bytes = (size_t) g.hc * g.n_embd * sizeof(float);
                hss.R_data.resize(R_bytes);
                cudaMemcpyAsync(hss.R_data.data(), src_R, R_bytes, cudaMemcpyDeviceToHost, cs);
            }

            hss.qsa_ord0 = s->qsa_ord0;
            hss.qsa_alloc = s->qsa_alloc;
            hss.qsa_slices.resize(s->qsa_alloc);

            for (int64_t j = 0; j < s->qsa_alloc; ++j) {
                const QsaState& qst = s->qsa_states[s->qsa_ord0 + j];
                auto& slice = hss.qsa_slices[j];

                if (qst.kv_q4) {
                    const size_t k_bytes = qsa_rows * strata::kernels::kv_q4_bytes_per_head((int) qs.head_dim);
                    const size_t v_bytes = k_bytes;
                    if (qst.k_q4) {
                        slice.k_q.resize(k_bytes);
                        cudaMemcpyAsync(slice.k_q.data(), qst.k_q4, k_bytes, cudaMemcpyDeviceToHost, cs);
                    }
                    if (qst.v_q4) {
                        slice.v_q4.resize(v_bytes);
                        cudaMemcpyAsync(slice.v_q4.data(), qst.v_q4, v_bytes, cudaMemcpyDeviceToHost, cs);
                    }
                } else if (qst.kv_hybrid) {
                    const size_t k_bytes = qsa_rows * qs.head_dim;
                    const size_t sc_bytes = qsa_rows * (qs.head_dim / strata::kernels::KV_Q8_GROUP) * 2;
                    const size_t v_bytes = qsa_rows * strata::kernels::kv_q4_bytes_per_head((int) qs.head_dim);
                    if (qst.k_q) {
                        slice.k_q.resize(k_bytes);
                        cudaMemcpyAsync(slice.k_q.data(), qst.k_q, k_bytes, cudaMemcpyDeviceToHost, cs);
                    }
                    if (qst.k_scale) {
                        slice.k_scale.resize(sc_bytes);
                        cudaMemcpyAsync(slice.k_scale.data(), qst.k_scale, sc_bytes, cudaMemcpyDeviceToHost, cs);
                    }
                    if (qst.v_q4) {
                        slice.v_q4.resize(v_bytes);
                        cudaMemcpyAsync(slice.v_q4.data(), qst.v_q4, v_bytes, cudaMemcpyDeviceToHost, cs);
                    }
                } else if (qst.kv_int8) {
                    const size_t k_bytes = qsa_rows * qs.head_dim;
                    const size_t sc_bytes = qsa_rows * (qs.head_dim / strata::kernels::KV_Q8_GROUP) * 2;
                    if (qst.k_q) {
                        slice.k_q.resize(k_bytes);
                        cudaMemcpyAsync(slice.k_q.data(), qst.k_q, k_bytes, cudaMemcpyDeviceToHost, cs);
                    }
                    if (qst.v_q) {
                        slice.v_q4.resize(k_bytes);
                        cudaMemcpyAsync(slice.v_q4.data(), qst.v_q, k_bytes, cudaMemcpyDeviceToHost, cs);
                    }
                    if (qst.k_scale) {
                        slice.k_scale.resize(sc_bytes);
                        cudaMemcpyAsync(slice.k_scale.data(), qst.k_scale, sc_bytes, cudaMemcpyDeviceToHost, cs);
                    }
                    if (qst.v_scale) {
                        slice.v_scale.resize(sc_bytes);
                        cudaMemcpyAsync(slice.v_scale.data(), qst.v_scale, sc_bytes, cudaMemcpyDeviceToHost, cs);
                    }
                }

                const size_t idx_tail_bytes = (size_t) (qs.idx_block - 1) * qs.idx_dim * sizeof(float);
                const size_t idx_dead_bytes = (size_t) qs.idx_dim * sizeof(float);
                if (qst.idx_tail) {
                    slice.idx_tail.resize(idx_tail_bytes);
                    cudaMemcpyAsync(slice.idx_tail.data(), qst.idx_tail, idx_tail_bytes, cudaMemcpyDeviceToHost, cs);
                }
                if (qst.idx_dead) {
                    slice.idx_dead.resize(idx_dead_bytes);
                    cudaMemcpyAsync(slice.idx_dead.data(), qst.idx_dead, idx_dead_bytes, cudaMemcpyDeviceToHost, cs);
                }
                if (qst.idx_block_pos) {
                    slice.has_idx_block_pos = true;
                    cudaMemcpyAsync(&slice.idx_block_pos_val, qst.idx_block_pos, sizeof(int32_t), cudaMemcpyDeviceToHost, cs);
                }
                const int64_t pooled_rows = prefix_len / qs.idx_block;
                if (pooled_rows > 0 && qst.idx_pooled) {
                    const size_t idx_pooled_bytes = (size_t) (pooled_rows + 1) * qs.idx_dim * sizeof(float);
                    slice.idx_pooled.resize(idx_pooled_bytes);
                    cudaMemcpyAsync(slice.idx_pooled.data(), qst.idx_pooled, (size_t) pooled_rows * qs.idx_dim * sizeof(float), cudaMemcpyDeviceToHost, cs);
                    cudaMemcpyAsync((char*) slice.idx_pooled.data() + pooled_rows * qs.idx_dim * sizeof(float),
                                    qst.idx_dead, idx_dead_bytes, cudaMemcpyDeviceToHost, cs);
                }
            }
        }
        for (size_t st = 0; st < n_stages; ++st) {
            const core::OnDevice on(stage_devices[st]);
            cudaStreamSynchronize((cudaStream_t) streams[st]);
        }

        bool snap_ok = true;
        for (size_t st = 0; st < n_stages && snap_ok; ++st) {
            auto& hss = curr->stage_host_snapshots[st];
            const auto* s = states[st];
            if (!s) continue;
            if (s->gdn_alloc > 0 && s->gdn_state && hss.gdn_data.empty()) snap_ok = false;
            if (s->ple_hist && hss.ple_data.empty()) snap_ok = false;
            for (int64_t j = 0; j < s->qsa_alloc && snap_ok; ++j) {
                const QsaState& qst = s->qsa_states[s->qsa_ord0 + j];
                auto& sl = hss.qsa_slices[j];
                if ((qst.k_q4 || qst.k_q) && sl.k_q.empty()) snap_ok = false;
                if ((qst.v_q4 || qst.v_q) && sl.v_q4.empty()) snap_ok = false;
            }
        }
        if (!snap_ok) {
            for (auto& hss2 : curr->stage_host_snapshots) {
                hss2.gdn_data.free();
                hss2.ple_data.free();
                hss2.R_data.free();
                hss2.qsa_slices.clear();
            }
            curr->stage_host_snapshots.clear();
            return curr;
        }

        curr->is_host_parked = true;
        curr->created_at = std::chrono::steady_clock::now();
        if (safe_no_evict) {
            curr->is_permanent = true;
            curr->kind = RadixNodeKind::SYSTEM_ROOT;
        }
        ++cached_host_snapshots_;
        // Continuous NVMe write-through staging with TBW endurance guardrails
        queue_nvme_write_through_locked(curr);

        if (!safe_no_evict) {
            evict_lru_locked(max_cached_snapshots_);
        }
    }
    return curr;
}

}  // namespace strata::core


namespace strata::core {

bool RadixTree::persist_manifest(const std::string& model_id_hash) {
    std::shared_lock<std::shared_mutex> lock(rw_lock_);
    if (!nvme_tier_) return false;

    std::string manifest_path = nvme_tier_->base_path() + "/radix_manifest.bin";
    std::string manifest_tmp = manifest_path + ".tmp";
    std::ofstream ofs(manifest_tmp, std::ios::binary);
    if (!ofs.is_open()) return false;

    std::vector<std::shared_ptr<RadixNode>> all_nodes;
    std::vector<std::shared_ptr<RadixNode>> stack;
    stack.push_back(root_);
    while (!stack.empty()) {
        auto node = stack.back();
        stack.pop_back();
        if (!node) continue;
        if (node != root_) all_nodes.push_back(node);
        for (const auto& kv : node->children) {
            stack.push_back(kv.second);
        }
    }

    NVMeManifestHeader hdr;
    hdr.magic = kNVMeMagic;
    hdr.version = kNVMeVersion;
    hdr.num_nodes = static_cast<uint32_t>(all_nodes.size());
    hdr.max_node_id = next_node_id_;
    if (!model_id_hash.empty()) {
        std::strncpy(hdr.model_id_hash, model_id_hash.c_str(), sizeof(hdr.model_id_hash) - 1);
    }
    ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

    for (const auto& node : all_nodes) {
        NVMeManifestNodeRecord rec;
        rec.node_id = node->id;
        if (auto p = node->parent.lock()) {
            rec.parent_id = p->id;
        } else {
            rec.parent_id = 0;
        }
        rec.prefix_len = node->prefix_len;
        rec.l3_bytes = node->l3_bytes;
        rec.n_edge_tokens = static_cast<uint32_t>(node->edge_tokens.size());
        rec.is_l3_offloaded = node->is_l3_offloaded;

        ofs.write(reinterpret_cast<const char*>(&rec), sizeof(rec));
        if (rec.n_edge_tokens > 0) {
            ofs.write(reinterpret_cast<const char*>(node->edge_tokens.data()),
                      rec.n_edge_tokens * sizeof(int32_t));
        }
    }
    ofs.flush();
    ofs.close();
    if (::rename(manifest_tmp.c_str(), manifest_path.c_str()) != 0) {
        return false;
    }
    return true;
}

bool RadixTree::restore_manifest(const std::string& model_id_hash) {
    std::unique_lock<std::shared_mutex> lock(rw_lock_);
    if (!nvme_tier_) return false;

    std::string manifest_path = nvme_tier_->base_path() + "/radix_manifest.bin";
    std::ifstream ifs(manifest_path, std::ios::binary);
    if (!ifs.is_open()) return false;

    NVMeManifestHeader hdr;
    ifs.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!ifs.good()) return false;
    if (hdr.magic != kNVMeMagic || hdr.version != kNVMeVersion) {
        return false;
    }
    if (!model_id_hash.empty() && std::strncmp(hdr.model_id_hash, model_id_hash.c_str(), sizeof(hdr.model_id_hash) - 1) != 0) {
        // Model hash mismatch: invalidate manifest
        return false;
    }

    std::unordered_map<int64_t, std::shared_ptr<RadixNode>> id_to_node;
    id_to_node[0] = root_;
    size_t restored_l3_count = 0;

    for (uint32_t i = 0; i < hdr.num_nodes; ++i) {
        NVMeManifestNodeRecord rec;
        ifs.read(reinterpret_cast<char*>(&rec), sizeof(rec));
        if (!ifs.good()) break;

        auto node = std::make_shared<RadixNode>();
        node->id = rec.node_id;
        node->prefix_len = rec.prefix_len;
        node->l3_bytes = rec.l3_bytes;
        node->is_l3_offloaded = rec.is_l3_offloaded;
        node->last_accessed = std::chrono::steady_clock::now();

        if (rec.n_edge_tokens > 0) {
            node->edge_tokens.resize(rec.n_edge_tokens);
            ifs.read(reinterpret_cast<char*>(node->edge_tokens.data()),
                     rec.n_edge_tokens * sizeof(int32_t));
            node->update_chunk_hashes();
        }

        id_to_node[rec.node_id] = node;
        auto parent_it = id_to_node.find(rec.parent_id);
        if (parent_it != id_to_node.end()) {
            node->parent = parent_it->second;
            if (!node->edge_tokens.empty()) {
                parent_it->second->children[node->edge_tokens[0]] = node;
            }
        }
        if (node->is_l3_offloaded) {
            ++restored_l3_count;
        }
    }

    next_node_id_ = std::max<int64_t>(next_node_id_, hdr.max_node_id);
    cached_nvme_snapshots_ = restored_l3_count;
    node_count_ = id_to_node.size();
    return true;
}

}  // namespace strata::core
