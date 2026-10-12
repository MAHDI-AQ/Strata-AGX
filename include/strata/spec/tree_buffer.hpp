// include/strata/spec/tree_buffer.hpp - Frontier 4: Canonical Speculative Tree Buffers
#pragma once

#include <cstdint>
#include <cstddef>

namespace strata::spec {

inline constexpr int kMaxTreeBufferNodes = 16;

/// Unified Device Tree Buffer: Device-side / Pinned Host-Mapped representation of candidate tree
struct alignas(64) DeviceTreeSpec {
    int32_t T = 0;                        // Number of active nodes in the tree (e.g. 4 or 6)
    int32_t max_depth = 0;                // Maximum depth in tree
    int32_t tokens[kMaxTreeBufferNodes] = {};       // Candidate token IDs per node
    int32_t parents[kMaxTreeBufferNodes] = {};      // Parent node index (-1 for root/anchor)
    int32_t depths[kMaxTreeBufferNodes] = {};       // Depth of each node (0 = root)
    int64_t positions[kMaxTreeBufferNodes] = {};    // Logical sequence position
    float   log_probs[kMaxTreeBufferNodes] = {};    // Cumulative or node log-prob
    uint64_t ancestor_mask[kMaxTreeBufferNodes] = {}; // 64-bit ancestor bitmask
    uint8_t  tree_mask[kMaxTreeBufferNodes * kMaxTreeBufferNodes] = {}; // 2D causal mask [16x16]
};

/// Result buffer produced by on-device DP winner selection kernel (spec_tree_verify_dp)
struct alignas(64) DeviceTreeResult {
    int32_t n_accepted = 0;                         // Number of accepted tokens along winning branch
    int32_t best_path_indices[kMaxTreeBufferNodes] = {}; // Node indices of winning branch in root->leaf order
    int32_t best_path_tokens[kMaxTreeBufferNodes] = {};  // Token IDs of winning branch
    float   best_path_score = 0.0f;                 // Cumulative score
    int32_t winning_leaf = 0;                       // Index of winning leaf node
};

/// Topologies helper to construct canonical 4-node and 6-node trees
struct CanonicalTrees {
    /// Canonical 4-node tree:
    /// Node 0: Anchor token x0 (Parent -1, Depth 0)
    /// Node 1: Top-1 candidate x1_0 (Parent 0, Depth 1)
    /// Node 2: Top-2 candidate x1_1 (Parent 0, Depth 1)
    /// Node 3: Top-1 continuation of Node 1 x2_0 (Parent 1, Depth 2)
    static inline void init_4node(DeviceTreeSpec& spec, int32_t root_token, int64_t pos0) {
        spec.T = 4;
        spec.max_depth = 2;
        spec.tokens[0] = root_token;
        spec.parents[0] = -1;
        spec.depths[0] = 0;
        spec.positions[0] = pos0;

        spec.parents[1] = 0;
        spec.depths[1] = 1;
        spec.positions[1] = pos0 + 1;

        spec.parents[2] = 0;
        spec.depths[2] = 1;
        spec.positions[2] = pos0 + 1;

        spec.parents[3] = 1;
        spec.depths[3] = 2;
        spec.positions[3] = pos0 + 2;

        compute_masks(spec);
    }

    /// Canonical 6-node tree:
    /// Node 0: Anchor (Parent -1, Depth 0)
    /// Node 1: Top-1 candidate (Parent 0, Depth 1)
    /// Node 2: Top-2 candidate (Parent 0, Depth 1)
    /// Node 3: Top-1 continuation of Node 1 (Parent 1, Depth 2)
    /// Node 4: Top-2 continuation of Node 1 (Parent 1, Depth 2)
    /// Node 5: Top-1 continuation of Node 2 (Parent 2, Depth 2)
    static inline void init_6node(DeviceTreeSpec& spec, int32_t root_token, int64_t pos0) {
        spec.T = 6;
        spec.max_depth = 2;
        spec.tokens[0] = root_token;
        spec.parents[0] = -1;
        spec.depths[0] = 0;
        spec.positions[0] = pos0;

        spec.parents[1] = 0;
        spec.depths[1] = 1;
        spec.positions[1] = pos0 + 1;

        spec.parents[2] = 0;
        spec.depths[2] = 1;
        spec.positions[2] = pos0 + 1;

        spec.parents[3] = 1;
        spec.depths[3] = 2;
        spec.positions[3] = pos0 + 2;

        spec.parents[4] = 1;
        spec.depths[4] = 2;
        spec.positions[4] = pos0 + 2;

        spec.parents[5] = 2;
        spec.depths[5] = 2;
        spec.positions[5] = pos0 + 2;

        compute_masks(spec);
    }

    static inline void compute_masks(DeviceTreeSpec& spec) {
        for (int i = 0; i < kMaxTreeBufferNodes; ++i) {
            spec.ancestor_mask[i] = 0ULL;
        }
        for (int i = 0; i < 256; ++i) {
            spec.tree_mask[i] = 0;
        }

        const int N = spec.T;
        for (int i = 0; i < N; ++i) {
            const int p = spec.parents[i];
            uint64_t a_mask = 0ULL;
            if (p >= 0 && p < N) {
                a_mask = spec.ancestor_mask[p] | (1ULL << p);
            }
            spec.ancestor_mask[i] = a_mask;
            const uint64_t full_mask = a_mask | (1ULL << i); // reflexive

            for (int j = 0; j < N; ++j) {
                if ((full_mask >> j) & 1ULL) {
                    spec.tree_mask[i * kMaxTreeBufferNodes + j] = 1;
                }
            }
        }
    }
};

} // namespace strata::spec
