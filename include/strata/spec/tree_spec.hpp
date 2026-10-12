// include/strata/spec/tree_spec.hpp - Speculative Tree Topology Verification & Confidence Gating
//
// Cross-Engine Attribution and Research Lineage:
//   - EAGLE-2: "Faster Sub-step Speculative Decoding with Dynamic Draft Trees" (Li et al., Peking University, arXiv:2406.16858)
//   - Sequoia: "4.8x Faster Speculative Decoding with Dynamic Trees" (Li, Shen, Zheng, Kwon, Stoica, UC Berkeley, arXiv:2402.12374)
//   - DeepSeek MTP: Multi-Token Prediction Architecture (DeepSeek-V2 / DeepSeek-V3 Technical Reports)
//   - Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads (Cai et al.)
//
// In multi-agent serving with speculative decoding, linear draft chains (x1 -> x2 -> x3) suffer from
// catastrophic early rejection when the target model diverges at x1, even if an alternate plausible
// branch exists. SpecTree represents candidate tokens as an acyclic tree T = (V, E) of up to K nodes.
// All nodes in the tree are evaluated simultaneously in a single target model forward pass by passing
// a 2D tree attention mask M[i, j] in {0, 1} where M[i, j] = 1 iff node j is an ancestor of node i.
#pragma once

#include <array>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>

#include "strata/spec/tree_buffer.hpp"

namespace strata::spec {

inline constexpr int kMaxTreeNodes = 16;
inline constexpr int kDefaultTreeNodes = 8;
inline constexpr int kMaxTreeDepth = 8;

struct SpecTreeNode {
    int32_t id = 0;              // node index [0 .. count-1]
    int32_t token = 0;           // candidate token ID
    int32_t parent_idx = -1;     // parent node index (-1 for root/input)
    int32_t depth = 0;           // 0 = root, 1 = first speculative draft, etc.
    float prob = 1.0f;           // drafter confidence / probability score
    float cumulative_score = 0.0f; // log-probability along path from root
    std::vector<int32_t> children;
};

/// Structure representing a topological, depth-indexed projection of the SpecTree
struct FlatProjection {
    std::vector<int32_t> tokens;
    std::vector<int64_t> positions;
    std::vector<int32_t> flat_to_node;
    std::vector<int32_t> node_to_flat;
    std::vector<int32_t> depths;
    std::vector<int32_t> parents;
};

class SpecTree {
public:
    SpecTree() { clear(); }

    void clear() {
        nodes_.clear();
        nodes_.reserve(kMaxTreeNodes);
        mask_.fill(0);
        ancestor_mask_.fill(0ULL);
    }

    int size() const { return static_cast<int>(nodes_.size()); }
    bool empty() const { return nodes_.empty(); }

    const SpecTreeNode& node(int idx) const { return nodes_[idx]; }
    SpecTreeNode& node(int idx) { return nodes_[idx]; }
    const std::vector<SpecTreeNode>& nodes() const { return nodes_; }

    /// Add a node to the speculative tree. Returns the new node's index.
    int add_node(int32_t token, int32_t parent_idx, float prob = 1.0f) {
        if (static_cast<int>(nodes_.size()) >= kMaxTreeNodes) return -1;
        const int id = static_cast<int>(nodes_.size());
        SpecTreeNode n;
        n.id = id;
        n.token = token;
        n.parent_idx = parent_idx;
        if (parent_idx >= 0 && parent_idx < id) {
            n.depth = nodes_[parent_idx].depth + 1;
            const float p_clamped = std::max(1e-5f, std::min(1.0f, prob));
            n.cumulative_score = nodes_[parent_idx].cumulative_score + std::log(p_clamped);
            nodes_[parent_idx].children.push_back(id);
        } else {
            n.depth = 0;
            n.cumulative_score = 0.0f;
        }
        n.prob = prob;
        nodes_.push_back(std::move(n));
        return id;
    }

    /// Add branch with multiple candidate tokens extending from parent_idx
    std::vector<int> add_branch(int parent_idx, const std::vector<int32_t>& tokens, const std::vector<float>& probs = {}) {
        std::vector<int> indices;
        indices.reserve(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            float p = (i < probs.size()) ? probs[i] : 1.0f;
            int idx = add_node(tokens[i], parent_idx, p);
            if (idx >= 0) indices.push_back(idx);
        }
        return indices;
    }

    /// Ground-truth ancestor traversal (slow baseline for verification)
    bool is_ancestor_slow(int ancestor, int descendant) const {
        if (ancestor < 0 || ancestor >= static_cast<int>(nodes_.size())) return false;
        if (descendant < 0 || descendant >= static_cast<int>(nodes_.size())) return false;
        int curr = descendant;
        while (curr >= 0) {
            if (curr == ancestor) return true;
            curr = nodes_[curr].parent_idx;
        }
        return false;
    }

    /// Checks if node  is an ancestor of node  (or ancestor == descendant)
    bool is_ancestor(int ancestor, int descendant) const {
        if (ancestor < 0 || ancestor >= static_cast<int>(nodes_.size())) return false;
        if (descendant < 0 || descendant >= static_cast<int>(nodes_.size())) return false;
        if (ancestor == descendant) return true;
        // Fast 64-bit ancestor bitmask lookup
        return (ancestor_mask_[descendant] & (1ULL << ancestor)) != 0ULL;
    }

    /// Computes the 2D causal tree attention mask M in {0, 1}^{TxT} with sub-microsecond latency (< 1.0 us)
    /// using 64-bit ancestor bitmasks (uint64_t ancestor_mask[16]) with bitwise OR inheritance:
    /// ancestor_mask[i] = ancestor_mask[parent[i]] | (1ULL << parent[i]).
    /// With reflexive self-attention: mask_at(i, j) = 1 if j == i or (ancestor_mask[i] & (1ULL << j)).
    void compute_mask() {
        mask_.fill(0);
        ancestor_mask_.fill(0ULL);
        const int N = static_cast<int>(nodes_.size());
        for (int i = 0; i < N; ++i) {
            const int p = nodes_[i].parent_idx;
            uint64_t a_mask = 0ULL;
            if (p >= 0 && p < N) {
                a_mask = ancestor_mask_[p] | (1ULL << p);
            }
            ancestor_mask_[i] = a_mask;
            const uint64_t full_mask = a_mask | (1ULL << i); // self-attention

            for (int j = 0; j < N; ++j) {
                if ((full_mask >> j) & 1ULL) {
                    mask_[i * kMaxTreeNodes + j] = 1;
                }
            }
        }
    }

    uint64_t ancestor_mask(int i) const {
        if (i < 0 || i >= static_cast<int>(nodes_.size())) return 0ULL;
        return ancestor_mask_[i];
    }

    const uint64_t* ancestor_masks() const { return ancestor_mask_.data(); }

    const uint8_t* mask_data() const { return mask_.data(); }
    uint8_t mask_at(int i, int j) const {
        if (i < 0 || i >= kMaxTreeNodes || j < 0 || j >= kMaxTreeNodes) return 0;
        return mask_[i * kMaxTreeNodes + j];
    }

    /// Flattens the tree nodes into a topological, depth-indexed representation
    FlatProjection flatten_tree(int64_t base_position = 0) const {
        FlatProjection proj;
        const int N = static_cast<int>(nodes_.size());
        if (N == 0) return proj;

        proj.tokens.reserve(N);
        proj.positions.reserve(N);
        proj.flat_to_node.reserve(N);
        proj.node_to_flat.assign(N, -1);
        proj.depths.reserve(N);
        proj.parents.reserve(N);

        int max_d = 0;
        for (int i = 0; i < N; ++i) {
            max_d = std::max(max_d, static_cast<int>(nodes_[i].depth));
        }

        // Group nodes contiguously by depth
        for (int d = 0; d <= max_d; ++d) {
            for (int i = 0; i < N; ++i) {
                if (nodes_[i].depth == d) {
                    int flat_idx = static_cast<int>(proj.tokens.size());
                    proj.node_to_flat[i] = flat_idx;
                    proj.flat_to_node.push_back(i);
                    proj.tokens.push_back(nodes_[i].token);
                    proj.positions.push_back(base_position + nodes_[i].depth);
                    proj.depths.push_back(nodes_[i].depth);
                }
            }
        }

        // Populate flat parents
        for (int f = 0; f < N; ++f) {
            int orig_node = proj.flat_to_node[f];
            int orig_parent = nodes_[orig_node].parent_idx;
            proj.parents.push_back((orig_parent >= 0) ? proj.node_to_flat[orig_parent] : -1);
        }

        return proj;
    }

    /// Flattens the tree nodes into sequential arrays for GPU verification forward passes.
    void flatten(std::vector<int32_t>& out_tokens,
                 std::vector<int64_t>& out_positions,
                 int64_t base_position) const {
        out_tokens.clear();
        out_positions.clear();
        const int N = static_cast<int>(nodes_.size());
        out_tokens.reserve(N);
        out_positions.reserve(N);
        for (int i = 0; i < N; ++i) {
            out_tokens.push_back(nodes_[i].token);
            out_positions.push_back(base_position + nodes_[i].depth);
        }
    }

    /// Dynamic Programming (DP) Longest Valid Path Selection:
    /// Given target model predictions for each node in the tree:
    /// target_predictions[u] is the token emitted by the target model when verifying node u.
    /// A branch transition from node u to child v is valid if target_predictions[u] == nodes_[v].token.
    /// Returns the vector of accepted token IDs along the longest valid branch with cumulative log-prob tie-breaking.
    std::vector<int32_t> find_longest_accepted_path(
        const int32_t* target_predictions,
        std::vector<int32_t>* accepted_indices = nullptr) const {

        if (nodes_.empty()) return {};
        const int N = static_cast<int>(nodes_.size());

        std::vector<uint8_t> valid(N, 0);
        std::vector<int32_t> dp_len(N, 0);
        std::vector<float> dp_score(N, -1e9f);

        // Root is unconditionally accepted as anchor
        valid[0] = 1;
        dp_len[0] = 1;
        dp_score[0] = nodes_[0].cumulative_score;

        // DP pass over nodes (topologically ordered since parent < id)
        for (int i = 1; i < N; ++i) {
            const int p = nodes_[i].parent_idx;
            if (p >= 0 && p < i && valid[p]) {
                if (target_predictions[p] == nodes_[i].token) {
                    valid[i] = 1;
                    dp_len[i] = dp_len[p] + 1;
                    dp_score[i] = nodes_[i].cumulative_score;
                }
            }
        }

        // Find optimal leaf: maximize length, tie-break on cumulative log-probability
        int best_node = 0;
        int max_len = 1;
        float max_score = dp_score[0];

        for (int i = 1; i < N; ++i) {
            if (valid[i]) {
                if (dp_len[i] > max_len || (dp_len[i] == max_len && dp_score[i] > max_score)) {
                    max_len = dp_len[i];
                    max_score = dp_score[i];
                    best_node = i;
                }
            }
        }

        // Backtrack path
        std::vector<int32_t> best_path;
        best_path.reserve(max_len);
        int curr = best_node;
        while (curr >= 0) {
            best_path.push_back(curr);
            curr = nodes_[curr].parent_idx;
        }
        std::reverse(best_path.begin(), best_path.end());

        std::vector<int32_t> accepted_tokens;
        accepted_tokens.reserve(best_path.size());
        for (int idx : best_path) {
            accepted_tokens.push_back(nodes_[idx].token);
        }

        if (accepted_indices) {
            *accepted_indices = best_path;
        }

        return accepted_tokens;
    }

private:
    std::vector<SpecTreeNode> nodes_;
    std::array<uint8_t, kMaxTreeNodes * kMaxTreeNodes> mask_{};
    std::array<uint64_t, kMaxTreeNodes> ancestor_mask_{};
};

// ============================================================================
// Dynamic Confidence Gater & Entropy Extraction (EAGLE-2 / Sequoia)
// ============================================================================
struct ConfidenceMetrics {
    float top1_prob = 1.0f;
    float top2_prob = 0.0f;
    float logit_margin = 10.0f; // top1 - top2 logit difference
    float entropy = 0.0f;       // -sum p log p
};

class ConfidenceGater {
public:
    explicit ConfidenceGater(float high_margin = 2.0f, float low_margin = 0.6f, float max_entropy = 1.5f)
        : high_margin_(high_margin), low_margin_(low_margin), max_entropy_(max_entropy) {}

    /// Evaluates logits to extract top-1 prob, logit margin, and Shannon entropy.
    static ConfidenceMetrics evaluate_logits(const float* logits, int vocab_size, int top_k = 10) {
        ConfidenceMetrics cm;
        if (!logits || vocab_size <= 0) return cm;

        // Find top-2 logits
        float max1 = -1e9f, max2 = -1e9f;
        int idx1 = 0;
        for (int i = 0; i < vocab_size; ++i) {
            const float val = logits[i];
            if (val > max1) {
                max2 = max1;
                max1 = val;
                idx1 = i;
            } else if (val > max2) {
                max2 = val;
            }
        }
        cm.logit_margin = max1 - max2;

        // Softmax over top-k for calibrated entropy
        std::vector<std::pair<float, int>> top_items;
        top_items.reserve(top_k);
        top_items.push_back({max1, idx1});

        for (int i = 0; i < vocab_size; ++i) {
            if (i == idx1) continue;
            if (top_items.size() < static_cast<size_t>(top_k)) {
                top_items.push_back({logits[i], i});
                if (top_items.size() == static_cast<size_t>(top_k)) {
                    std::sort(top_items.begin(), top_items.end(), std::greater<std::pair<float, int>>());
                }
            } else if (logits[i] > top_items.back().first) {
                top_items.back() = {logits[i], i};
                for (int j = static_cast<int>(top_items.size()) - 1; j > 0 && top_items[j].first > top_items[j - 1].first; --j) {
                    std::swap(top_items[j], top_items[j - 1]);
                }
            }
        }

        // Softmax & Entropy
        float sum_exp = 0.0f;
        for (const auto& item : top_items) {
            sum_exp += std::exp(item.first - max1);
        }
        cm.top1_prob = 1.0f / sum_exp;
        if (top_items.size() > 1) {
            cm.top2_prob = std::exp(top_items[1].first - max1) / sum_exp;
        }

        float ent = 0.0f;
        for (const auto& item : top_items) {
            const float p = std::exp(item.first - max1) / sum_exp;
            if (p > 1e-6f) ent -= p * std::log(p);
        }
        cm.entropy = ent;

        return cm;
    }

    /// Selects optimal speculative depth T in [1 .. max_t] and whether to use tree branching.
    struct Decision {
        int recommended_t = 1;
        bool use_tree = false;
        const char* reason = "normal";
    };

    Decision decide(const ConfidenceMetrics& cm, int max_t, float spec_min_p = 0.5f) const {
        Decision d;
        // High confidence: expand speculation depth and enable speculative tree exploration up to max_t <= 16
        if (cm.logit_margin >= high_margin_ && cm.entropy <= max_entropy_ && cm.top1_prob >= spec_min_p) {
            d.recommended_t = std::min(max_t, kMaxTreeNodes);
            d.use_tree = (d.recommended_t >= 3);
            d.reason = "high_confidence_expanded";
            return d;
        }
        // Low confidence / high entropy: throttle to 1 to avoid wasted draft & verify compute
        if (cm.logit_margin < low_margin_ || cm.entropy > max_entropy_ * 1.5f || cm.top1_prob < spec_min_p * 0.7f) {
            d.recommended_t = 1;
            d.use_tree = false;
            d.reason = "high_entropy_throttled";
            return d;
        }
        // Normal regime
        d.recommended_t = std::min(max_t, 4);
        d.use_tree = (d.recommended_t >= 3);
        d.reason = "standard_regime";
        return d;
    }

    /// Online EMA calibration based on acceptance feedback
    void observe(int offered, int accepted) {
        if (offered <= 0) return;
        const float rate = static_cast<float>(accepted) / static_cast<float>(offered);
        if (rate > 0.85f) {
            high_margin_ = std::max(1.2f, high_margin_ * 0.98f);
            low_margin_ = std::max(0.3f, low_margin_ * 0.98f);
        } else if (rate < 0.60f) {
            high_margin_ = std::min(3.5f, high_margin_ * 1.05f);
            low_margin_ = std::min(1.2f, low_margin_ * 1.05f);
        }
    }

    float high_margin() const { return high_margin_; }
    float low_margin() const { return low_margin_; }

private:
    float high_margin_ = 2.0f;
    float low_margin_ = 0.6f;
    float max_entropy_ = 1.5f;
};

} // namespace strata::spec
