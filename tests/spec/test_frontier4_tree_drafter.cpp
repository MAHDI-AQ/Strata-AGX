#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>

#include "strata/spec/tree_buffer.hpp"
#include "strata/spec/tree_spec.hpp"
#include "strata/kernels/verify_kernels.hpp"

using namespace strata::spec;
using namespace strata::kernels;

int main() {
    std::cout << "=== Running Frontier 4: Direct MTP Multi-Branch Tree Drafter Verification Battery ===" << std::endl;

    // 1. Verify Canonical 4-node and 6-node tree topologies
    DeviceTreeSpec spec4;
    CanonicalTrees::init_4node(spec4, 101, 1000);
    assert(spec4.T == 4);
    assert(spec4.tokens[0] == 101);
    assert(spec4.parents[0] == -1);
    assert(spec4.parents[1] == 0 && spec4.depths[1] == 1);
    assert(spec4.parents[2] == 0 && spec4.depths[2] == 1);
    assert(spec4.parents[3] == 1 && spec4.depths[3] == 2);
    
    // Verify 2D causal mask bits
    assert(spec4.tree_mask[0 * 16 + 0] == 1);
    assert(spec4.tree_mask[0 * 16 + 1] == 0);
    assert(spec4.tree_mask[0 * 16 + 2] == 0);
    assert(spec4.tree_mask[0 * 16 + 3] == 0);

    assert(spec4.tree_mask[1 * 16 + 0] == 1);
    assert(spec4.tree_mask[1 * 16 + 1] == 1);
    assert(spec4.tree_mask[1 * 16 + 2] == 0);
    assert(spec4.tree_mask[1 * 16 + 3] == 0);

    assert(spec4.tree_mask[2 * 16 + 0] == 1);
    assert(spec4.tree_mask[2 * 16 + 1] == 0);
    assert(spec4.tree_mask[2 * 16 + 2] == 1);
    assert(spec4.tree_mask[2 * 16 + 3] == 0);

    assert(spec4.tree_mask[3 * 16 + 0] == 1);
    assert(spec4.tree_mask[3 * 16 + 1] == 1);
    assert(spec4.tree_mask[3 * 16 + 2] == 0);
    assert(spec4.tree_mask[3 * 16 + 3] == 1);
    std::cout << "[PASS] Canonical 4-node tree topology & 2D causal attention mask validated." << std::endl;

    // 2. Test Device-Side Top-2 Branch Selection Kernel
    constexpr int kVocab = 1000;
    std::vector<float> h_logits(kVocab, -10.0f);
    h_logits[500] = 12.5f;
    h_logits[750] = 10.2f;

    float* d_logits = nullptr;
    int32_t* d_tree_tokens = nullptr;
    int32_t* d_parents = nullptr;
    int32_t* d_depths = nullptr;
    int64_t* d_positions = nullptr;
    float* d_log_probs = nullptr;
    int32_t* d_anchor = nullptr;

    cudaMalloc(&d_logits, kVocab * sizeof(float));
    cudaMalloc(&d_tree_tokens, 16 * sizeof(int32_t));
    cudaMalloc(&d_parents, 16 * sizeof(int32_t));
    cudaMalloc(&d_depths, 16 * sizeof(int32_t));
    cudaMalloc(&d_positions, 16 * sizeof(int64_t));
    cudaMalloc(&d_log_probs, 16 * sizeof(float));
    cudaMalloc(&d_anchor, sizeof(int32_t));

    int32_t anchor_val = 101;
    cudaMemcpy(d_anchor, &anchor_val, sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_logits, h_logits.data(), kVocab * sizeof(float), cudaMemcpyHostToDevice);

    mtp_tree_branch_select(d_logits, kVocab, d_anchor, 2000, d_tree_tokens, d_parents, d_depths, d_positions, d_log_probs, nullptr);
    cudaDeviceSynchronize();

    std::vector<int32_t> out_tokens(16);
    cudaMemcpy(out_tokens.data(), d_tree_tokens, 16 * sizeof(int32_t), cudaMemcpyDeviceToHost);

    assert(out_tokens[0] == 101);
    assert(out_tokens[1] == 500);
    assert(out_tokens[2] == 750);
    std::cout << "[PASS] mtp_tree_branch_select GPU warp kernel correctly extracted Top-1 (500) and Top-2 (750)." << std::endl;

    // 3. Test GPU DP Winner Selection
    std::vector<int32_t> h_tree_toks = {101, 500, 750, 600};
    std::vector<int32_t> h_target_toks = {750, 999, 888, 777};
    std::vector<int32_t> h_parents = {-1, 0, 0, 1};

    int32_t* d_ttoks = nullptr;
    int32_t* d_target = nullptr;
    int32_t* d_pars = nullptr;
    int32_t* d_best_indices = nullptr;
    int32_t* d_n_accepted = nullptr;

    cudaMalloc(&d_ttoks, 16 * sizeof(int32_t));
    cudaMalloc(&d_target, 16 * sizeof(int32_t));
    cudaMalloc(&d_pars, 16 * sizeof(int32_t));
    cudaMalloc(&d_best_indices, 16 * sizeof(int32_t));
    cudaMalloc(&d_n_accepted, sizeof(int32_t));

    cudaMemcpy(d_ttoks, h_tree_toks.data(), 4 * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, h_target_toks.data(), 4 * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pars, h_parents.data(), 4 * sizeof(int32_t), cudaMemcpyHostToDevice);

    spec_tree_verify_dp(d_ttoks, d_target, d_pars, nullptr, d_best_indices, d_n_accepted, 4, nullptr);
    cudaDeviceSynchronize();

    int32_t n_acc = 0;
    std::vector<int32_t> best_idx(4);
    cudaMemcpy(&n_acc, d_n_accepted, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(best_idx.data(), d_best_indices, 4 * sizeof(int32_t), cudaMemcpyDeviceToHost);

    assert(n_acc == 2);
    assert(best_idx[0] == 0);
    assert(best_idx[1] == 2);
    std::cout << "[PASS] spec_tree_verify_dp successfully rescued primary divergence: recovered Top-2 branch (Node 2)!" << std::endl;

    cudaFree(d_logits); cudaFree(d_tree_tokens); cudaFree(d_parents); cudaFree(d_depths);
    cudaFree(d_positions); cudaFree(d_log_probs); cudaFree(d_anchor);
    cudaFree(d_ttoks); cudaFree(d_target); cudaFree(d_pars); cudaFree(d_best_indices); cudaFree(d_n_accepted);

    std::cout << "=== ALL FRONTIER 4 MULTI-BRANCH TREE TESTS PASSED WITH 100% PARITY! ===" << std::endl;
    return 0;
}
