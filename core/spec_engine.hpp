// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU speculative-decoding helpers:
//  * SuffixTree: model-free SuffixDecoding drafts. Every window of the token history (prompt + output) is
//    inserted into a depth-bounded suffix trie; the longest suffix of the live context that matches is
//    expanded best-first into a draft tree scored by empirical continuation frequency.
//  * MoeKGovernor: online draft-length controller for MoE targets. Verifying K drafts touches the union of
//    the experts routed by K+1 tokens, so verification cost grows almost linearly in K; the governor picks
//    K* = argmax E[accepted+1] / cost(K) from the observed acceptance rate and returns 0 when speculation
//    cannot pay off.
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace vegajet::spec {

struct DraftNode {
    int32_t token = -1;
    int32_t parent = -1;   // index into the draft vector, -1 = root (continues the live context)
    double score = 0.0;    // product of empirical continuation probabilities along the path
};

class SuffixTree {
public:
    explicit SuffixTree(uint32_t max_depth = 24) : max_depth_(max_depth) { nodes_.emplace_back(); }
    // Appends tokens to the history and inserts all new windows.
    void extend(const std::vector<int32_t>& tokens);
    void push(int32_t token);
    // Longest context suffix found in the trie (0 if none).
    uint32_t match_length(const std::vector<int32_t>& context) const;
    // Draft tree with at most max_nodes nodes and branching <= max_branch; nodes are in BFS-compatible
    // order (parents before children). Empty if no suffix of the context occurs earlier with a continuation.
    std::vector<DraftNode> draft(const std::vector<int32_t>& context, uint32_t max_nodes, uint32_t max_branch = 2,
                                 double min_score = 0.05) const;
    size_t node_count() const { return nodes_.size(); }
    const std::vector<int32_t>& history() const { return history_; }

private:
    struct Node { std::unordered_map<int32_t, uint32_t> next; uint32_t count = 0; };
    int64_t walk(const int32_t* toks, size_t n) const;  // node id or -1
    uint32_t max_depth_;
    std::vector<Node> nodes_;
    std::vector<int32_t> history_;
};

struct MoeCostModel {
    uint32_t n_experts = 256;
    uint32_t experts_per_token = 8;
    double moe_byte_fraction = 0.7;   // share of per-token decode bytes that are routed-expert weights
    double draft_cost_per_token = 0.02;  // draft cost relative to one target decode step
    // Expected distinct experts touched by n tokens (uniform routing approximation).
    double unique_experts(uint32_t n_tokens) const;
    // Cost of verifying k drafts (k+1 tokens) relative to a single-token decode, plus drafting.
    double step_cost(uint32_t k) const;
};

class MoeKGovernor {
public:
    explicit MoeKGovernor(MoeCostModel m = {}, uint32_t k_max = 8, double ema = 0.1)
        : model_(m), k_max_(k_max), ema_(ema) {}
    // Expected tokens per step with k drafts at per-token acceptance alpha: (1 - a^(k+1)) / (1 - a).
    static double expected_tokens(double alpha, uint32_t k);
    // Best k in [0, k_max] for a given alpha; 0 when no k beats plain decoding.
    uint32_t best_k(double alpha) const;
    // Online update from one verification: `proposed` drafts, `accepted` of them taken.
    void observe(uint32_t proposed, uint32_t accepted);
    double alpha() const { return alpha_; }
    uint32_t k_star() const { return best_k(alpha_); }

private:
    MoeCostModel model_;
    uint32_t k_max_;
    double ema_;
    double alpha_ = 0.6;  // prior
};

} // namespace vegajet::spec
