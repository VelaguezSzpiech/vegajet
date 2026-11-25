// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/spec_engine.hpp"

#include <algorithm>
#include <cmath>
#include <queue>

namespace vegajet::spec {

void SuffixTree::push(int32_t token) {
    history_.push_back(token);
    // The new token extends every window that started within the last max_depth positions; re-inserting
    // only the windows ending here keeps counts consistent: each (start, length) prefix counted once.
    const size_t n = history_.size();
    const size_t first = n > max_depth_ ? n - max_depth_ : 0;
    for (size_t s = first; s < n; ++s) {
        // Walk the existing prefix [s, n-1) then add the final token.
        uint32_t node = 0;
        bool ok = true;
        for (size_t i = s; i + 1 < n; ++i) {
            auto it = nodes_[node].next.find(history_[i]);
            if (it == nodes_[node].next.end()) { ok = false; break; }
            node = it->second;
        }
        if (!ok) continue;
        auto it = nodes_[node].next.find(token);
        uint32_t child;
        if (it == nodes_[node].next.end()) {
            child = static_cast<uint32_t>(nodes_.size());
            nodes_[node].next.emplace(token, child);
            nodes_.emplace_back();
        } else {
            child = it->second;
        }
        nodes_[child].count++;
    }
}

void SuffixTree::extend(const std::vector<int32_t>& tokens) {
    for (int32_t t : tokens) push(t);
}

int64_t SuffixTree::walk(const int32_t* toks, size_t n) const {
    uint32_t node = 0;
    for (size_t i = 0; i < n; ++i) {
        auto it = nodes_[node].next.find(toks[i]);
        if (it == nodes_[node].next.end()) return -1;
        node = it->second;
    }
    return node;
}

uint32_t SuffixTree::match_length(const std::vector<int32_t>& ctx) const {
    const size_t max_m = std::min<size_t>(ctx.size(), max_depth_ - 1);
    for (size_t m = max_m; m > 0; --m) {
        const int64_t node = walk(ctx.data() + ctx.size() - m, m);
        if (node >= 0 && !nodes_[node].next.empty()) return static_cast<uint32_t>(m);
    }
    return 0;
}

std::vector<DraftNode> SuffixTree::draft(const std::vector<int32_t>& ctx, uint32_t max_nodes, uint32_t max_branch,
                                         double min_score) const {
    std::vector<DraftNode> out;
    const uint32_t m = match_length(ctx);
    if (m == 0 || max_nodes == 0) return out;
    const int64_t start = walk(ctx.data() + ctx.size() - m, m);
    struct Item { double score; uint32_t trie; int32_t token; int32_t parent; uint32_t depth; };
    auto cmp = [](const Item& a, const Item& b) { return a.score < b.score; };
    std::priority_queue<Item, std::vector<Item>, decltype(cmp)> pq(cmp);
    auto push_children = [&](uint32_t trie, double base, int32_t parent, uint32_t depth) {
        const auto& nd = nodes_[trie];
        uint32_t total = 0;
        for (const auto& kv : nd.next) total += nodes_[kv.second].count;
        std::vector<std::pair<uint32_t, std::pair<int32_t, uint32_t>>> kids;
        for (const auto& kv : nd.next) kids.push_back({nodes_[kv.second].count, {kv.first, kv.second}});
        std::sort(kids.begin(), kids.end(), [](const auto& a, const auto& b) {
            return a.first != b.first ? a.first > b.first : a.second.first < b.second.first;
        });
        for (size_t i = 0; i < kids.size() && i < max_branch; ++i)
            pq.push({base * kids[i].first / std::max(1u, total), kids[i].second.second, kids[i].second.first, parent, depth + 1});
    };
    push_children(static_cast<uint32_t>(start), 1.0, -1, m);
    while (!pq.empty() && out.size() < max_nodes) {
        Item it = pq.top();
        pq.pop();
        if (it.score < min_score) break;
        out.push_back({it.token, it.parent, it.score});
        if (it.depth < max_depth_ - 1)
            push_children(it.trie, it.score, static_cast<int32_t>(out.size() - 1), it.depth);
    }
    return out;
}

double MoeCostModel::unique_experts(uint32_t n) const {
    if (n_experts == 0) return 0;
    const double p = static_cast<double>(experts_per_token) / n_experts;
    return n_experts * (1.0 - std::pow(1.0 - p, static_cast<double>(n)));
}

double MoeCostModel::step_cost(uint32_t k) const {
    const double growth = unique_experts(k + 1) / std::max(1e-9, unique_experts(1));
    return (1.0 - moe_byte_fraction) + moe_byte_fraction * growth + draft_cost_per_token * k;
}

double MoeKGovernor::expected_tokens(double a, uint32_t k) {
    if (a >= 1.0) return k + 1.0;
    return (1.0 - std::pow(a, static_cast<double>(k + 1))) / (1.0 - a);
}

uint32_t MoeKGovernor::best_k(double a) const {
    uint32_t best = 0;
    double best_rate = 1.0;  // plain decoding: 1 token per unit cost
    for (uint32_t k = 1; k <= k_max_; ++k) {
        const double rate = expected_tokens(a, k) / model_.step_cost(k);
        if (rate > best_rate + 1e-9) { best_rate = rate; best = k; }
    }
    return best;
}

void MoeKGovernor::observe(uint32_t proposed, uint32_t accepted) {
    if (proposed == 0) return;
    accepted = std::min(accepted, proposed);
    // Geometric acceptance MLE for one step: accepted successes, one failure unless all were accepted.
    const double sample = accepted == proposed ? 1.0 : static_cast<double>(accepted) / (accepted + 1.0);
    alpha_ = (1.0 - ema_) * alpha_ + ema_ * sample;
}

} // namespace vegajet::spec
