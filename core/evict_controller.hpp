// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// EVICT: expert-union-aware online truncation of speculative drafts for MoE targets, plus an adaptive draft depth.
//
// Verifying K draft tokens in one pass costs (on a bandwidth-bound gfx906 MoE decode) roughly
//     time(S) = fixed + per_token * (1 + |S|) + per_expert * U(S)
// where U(S) is the number of unique (layer, expert) pairs routed by the verified token plus the kept draft nodes
// S: every additional draft token can pull in up to top_k new experts per layer. The expected output of the pass is
//     E(S) = 1 + sum_{i in S} P(path to i accepted)            (accepted drafts + the bonus token)
// The controller keeps the subset S (closed under ancestors) that maximizes E(S) / time(S):
//   * chains (each node's parent is the previous node): exact scan over all prefixes, plus a per-depth fan-out
//     table for K = k_min..k_max (unique experts, inflation, expected tokens, tokens/ms);
//   * trees: greedy leaf removal while the score improves.
// A hard cap prunes first: while inflation = U(S) / U(root alone) exceeds max_inflation, the leaf with the least
// path probability per newly loaded expert is removed. After verification, observe() feeds back the accepted length
// and the measured inflation; next_depth() is the proposal depth for the next round, clamped to [k_min, k_max].
// Drafts without routing predictions (e.g. suffix-tree drafts) use decide_chain_unrouted(), which models U(k) with
// the measured per-token inflation slope.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::evict {

using Routing = std::vector<std::vector<uint16_t>>;  // [layer] -> expert ids routed for one token

struct DraftNode {
    int32_t token = -1;
    int32_t parent = -1;     // index of an earlier node, or -1 for a child of the verified token
    double p_accept = 0.0;   // P(accepted | parent accepted), in [0, 1]
    Routing experts;         // predicted routing (may be empty for decide_chain_unrouted)
};

struct CostModel {
    double fixed_us = 0.0;      // per verify pass (launches, norms, attention setup, LM head)
    double per_token_us = 0.0;  // per verified token (attention over the context, activations)
    double per_expert_us = 0.0; // per unique (layer, expert): expert bytes / sustained HBM bandwidth
};

struct ControllerConfig {
    uint32_t n_layers = 0;       // MoE layers
    uint32_t n_experts = 0;      // routed experts per layer
    uint32_t k_min = 3, k_max = 12;
    double max_inflation = 3.0;  // hard cap on U(S) / U(root)
    double ewma = 0.25;          // smoothing of the depth and inflation-slope estimates
};

struct DepthScore {
    uint32_t k = 0;
    uint64_t unique_experts = 0;
    double inflation = 0.0;
    double expected_tokens = 0.0;
    double time_us = 0.0;
    double tokens_per_ms = 0.0;
};

struct Decision {
    bool ok = false;
    std::string error;
    std::vector<int32_t> keep;      // kept node indices (ancestor-closed), ascending
    std::vector<DepthScore> table;  // chains: k = k_min..min(k_max, nodes)
    uint64_t root_experts = 0, unique_experts = 0;
    double inflation = 0.0, expected_tokens = 0.0, time_us = 0.0, tokens_per_ms = 0.0;
    uint32_t pruned_by_inflation = 0, pruned_by_score = 0;
};

class EvictController {
public:
    EvictController(ControllerConfig cfg, CostModel cost);

    Decision decide(const Routing& root, const std::vector<DraftNode>& draft) const;
    // Chain without routing predictions: U(k) = U0 * (1 + slope * k), slope from observe() (or the given prior).
    Decision decide_chain_unrouted(uint64_t root_unique, const std::vector<double>& p_accept) const;

    // Feedback after a verify pass: kept draft length, accepted drafts, measured U(S)/U(root) of the pass.
    void observe(uint32_t kept, uint32_t accepted, double measured_inflation);
    uint32_t next_depth() const;
    double inflation_slope() const { return slope_; }
    void set_inflation_slope(double s) { slope_ = s; }

    // Exposed for tests: unique (layer, expert) pairs of the root plus the given nodes.
    static uint64_t count_unique(const Routing& root, const std::vector<DraftNode>& draft, const std::vector<int32_t>& nodes,
                                 uint32_t n_layers, uint32_t n_experts);
    double score(double expected_tokens, uint64_t unique, uint32_t n_tokens) const;  // tokens per ms

private:
    ControllerConfig cfg_;
    CostModel cost_;
    double depth_ = 0.0;   // EWMA of the useful depth
    double slope_ = 0.5;   // EWMA of (inflation - 1) / kept
};

} // namespace vegajet::evict
