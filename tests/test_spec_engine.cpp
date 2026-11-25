// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// SuffixDecoding tree insertion / lookup / draft trees and the adaptive MoE K* governor.
#include "core/spec_engine.hpp"
#include "tests/test_common.hpp"

#include <cmath>

using namespace vegajet::spec;

int main() {
    SuffixTree t(8);
    // "a b c d | a b c e | a b c d" with a=1 b=2 c=3 d=4 e=5
    t.extend({1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 3, 4});
    VJ_CHECK(t.history().size() == 12);
    VJ_CHECK(t.match_length({9, 1, 2, 3}) == 3);
    VJ_CHECK(t.match_length({7, 7}) == 0);
    VJ_CHECK(t.draft({7, 7}, 4).empty());

    auto d = t.draft({1, 2, 3}, 6, 2, 0.0);
    VJ_CHECK(!d.empty());
    VJ_CHECK(d[0].token == 4 && d[0].parent == -1 && std::fabs(d[0].score - 2.0 / 3.0) < 1e-9);
    bool saw_e = false;
    for (size_t i = 0; i < d.size(); ++i) {
        VJ_CHECK(d[i].parent < static_cast<int32_t>(i));  // parents precede children
        if (d[i].parent >= 0) VJ_CHECK(d[i].score <= d[d[i].parent].score + 1e-12);
        saw_e |= d[i].token == 5 && d[i].parent == -1;
    }
    VJ_CHECK(saw_e);  // second branch at the root
    // Continuation after "d": history shows "d a" once -> chain grows along the most frequent path.
    VJ_CHECK(d.size() >= 3 && d[1].parent == 0 && d[1].token == 1);
    // Incremental push keeps counts consistent with batch extend.
    SuffixTree t2(8);
    for (int32_t x : {1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 3, 4}) t2.push(x);
    VJ_CHECK(t2.node_count() == t.node_count());
    // Budget respected.
    VJ_CHECK(t.draft({1, 2, 3}, 2, 2, 0.0).size() == 2);
    std::printf("[PASS] SuffixDecoding: insertion, longest-suffix match, best-first draft tree (%zu nodes, top=%d p=%.2f)\n",
                d.size(), d[0].token, d[0].score);

    // MoE governor: 256 experts, 8 per token, 70% of decode bytes in routed experts.
    MoeKGovernor g(MoeCostModel{256, 8, 0.7, 0.02});
    for (double a : {0.0, 0.2, 0.4, 0.5}) VJ_CHECK(g.best_k(a) == 0);
    VJ_CHECK(g.best_k(0.9) > 0 && g.best_k(0.97) >= g.best_k(0.9));
    uint32_t prev = 0;
    for (double a = 0.0; a <= 0.99; a += 0.01) { const uint32_t k = g.best_k(a); VJ_CHECK(k >= prev); prev = k; }
    // Dense target (no expert-union growth): speculation already pays at alpha = 0.5.
    MoeKGovernor dense(MoeCostModel{256, 8, 0.0, 0.02});
    VJ_CHECK(dense.best_k(0.5) > 0);
    // Online: repeated rejections drive alpha below 0.5 and K* to 0; acceptance recovers it.
    for (int i = 0; i < 60; ++i) g.observe(4, 0);
    VJ_CHECK(g.alpha() < 0.5 && g.k_star() == 0);
    for (int i = 0; i < 120; ++i) g.observe(4, 4);
    VJ_CHECK(g.alpha() > 0.9 && g.k_star() > 0);
    VJ_CHECK(std::fabs(MoeKGovernor::expected_tokens(0.5, 3) - 1.875) < 1e-12);
    double a_star = 0;
    for (double a = 0.5; a < 1.0; a += 0.01) if (g.best_k(a) > 0) { a_star = a; break; }
    std::printf("[PASS] MoE K* governor: K*=0 for alpha<=0.50 (first K>0 at alpha=%.2f), K*(0.9)=%u, dense K*(0.5)=%u\n",
                a_star, g.best_k(0.9), dense.best_k(0.5));
    return 0;
}
