// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/evict_controller: unique-expert counting vs a naive set, the K = 3..12 fan-out table, exact prefix choice on
// chains (vs brute force), greedy pruning on trees (vs exhaustive search over ancestor-closed subsets), the hard
// inflation cap, unrouted chains, depth adaptation, and input validation.
#include "core/evict_controller.hpp"
#include "tests/test_common.hpp"

#include <algorithm>
#include <cstdio>
#include <random>
#include <set>

using namespace vegajet::evict;

namespace {

std::mt19937_64 rng(11);

std::vector<uint16_t> random_topk(uint32_t X, uint32_t k) {
    std::vector<uint16_t> v;
    while (v.size() < k) {
        const uint16_t e = static_cast<uint16_t>(rng() % X);
        if (std::find(v.begin(), v.end(), e) == v.end()) v.push_back(e);
    }
    return v;
}

// Routing that keeps each of `base`'s experts with probability `keep`, otherwise replaces it with a random one.
Routing correlated(const Routing& base, uint32_t X, double keep) {
    Routing r = base;
    std::uniform_real_distribution<double> U(0, 1);
    for (auto& l : r)
        for (auto& e : l)
            if (U(rng) > keep) {
                uint16_t c;
                do c = static_cast<uint16_t>(rng() % X); while (std::find(l.begin(), l.end(), c) != l.end());
                e = c;
            }
    return r;
}

Routing random_routing(uint32_t L, uint32_t X, uint32_t k) {
    Routing r(L);
    for (auto& l : r) l = random_topk(X, k);
    return r;
}

struct Brute { double score = -1; uint64_t mask = 0; };

// Exhaustive optimum over ancestor-closed subsets with the inflation cap.
Brute brute(const EvictController& c, const Routing& root, const std::vector<DraftNode>& d, uint32_t L, uint32_t X, double cap) {
    Brute b;
    const size_t n = d.size();
    const uint64_t u0 = EvictController::count_unique(root, d, {}, L, X);
    for (uint64_t m = 0; m < (1ull << n); ++m) {
        bool closed = true;
        std::vector<int32_t> nodes;
        double E = 1.0;
        for (size_t i = 0; i < n && closed; ++i) {
            if (!(m >> i & 1)) continue;
            if (d[i].parent >= 0 && !(m >> d[i].parent & 1)) closed = false;
            nodes.push_back(static_cast<int32_t>(i));
            double p = 1.0;
            for (int32_t j = static_cast<int32_t>(i); j >= 0; j = d[static_cast<size_t>(j)].parent) p *= d[static_cast<size_t>(j)].p_accept;
            E += p;
        }
        if (!closed) continue;
        const uint64_t u = EvictController::count_unique(root, d, nodes, L, X);
        if (double(u) / u0 > cap) continue;
        const double s = c.score(E, u, static_cast<uint32_t>(1 + nodes.size()));
        if (s > b.score) { b.score = s; b.mask = m; }
    }
    return b;
}

} // namespace

int main() {
    // Bandwidth-bound MoE verify pass: ~2 us per (layer, expert) streamed, 3 ms fixed, 50 us per extra token.
    const CostModel cost{3000.0, 50.0, 2.0};
    const uint32_t L = 48, X = 512, TOPK = 10;

    // ---- count_unique vs a naive set
    for (int t = 0; t < 50; ++t) {
        const Routing root = random_routing(6, 64, 4);
        std::vector<DraftNode> d(5);
        std::vector<int32_t> nodes;
        std::set<std::pair<uint32_t, uint16_t>> ref;
        for (uint32_t l = 0; l < 6; ++l) for (auto e : root[l]) ref.insert({l, e});
        for (size_t i = 0; i < d.size(); ++i) {
            d[i].experts = correlated(root, 64, 0.5);
            if (rng() % 2) {
                nodes.push_back(static_cast<int32_t>(i));
                for (uint32_t l = 0; l < 6; ++l) for (auto e : d[i].experts[l]) ref.insert({l, e});
            }
        }
        VJ_CHECK(EvictController::count_unique(root, d, nodes, 6, 64) == ref.size());
    }

    ControllerConfig cfg;
    cfg.n_layers = L;
    cfg.n_experts = X;
    cfg.max_inflation = 3.0;
    EvictController ctl(cfg, cost);

    // ---- A: routing of the draft tokens close to the verified token's, high acceptance -> keep the whole chain.
    {
        const Routing root = random_routing(L, X, TOPK);
        std::vector<DraftNode> d(12);
        for (size_t i = 0; i < d.size(); ++i) {
            d[i].parent = static_cast<int32_t>(i) - 1;
            d[i].p_accept = 0.9;
            d[i].experts = correlated(root, X, 0.97);
        }
        const Decision dec = ctl.decide(root, d);
        VJ_CHECK(dec.ok && dec.keep.size() == 12 && dec.pruned_by_inflation == 0);
        VJ_CHECK(dec.table.size() == 10 && dec.table.front().k == 3 && dec.table.back().k == 12);
        for (size_t i = 1; i < dec.table.size(); ++i) VJ_CHECK(dec.table[i].unique_experts >= dec.table[i - 1].unique_experts);
        std::printf("  correlated routing: K=3 inflation %.2f, K=12 inflation %.2f, kept %zu, %.3f tok/ms\n",
                    dec.table.front().inflation, dec.table.back().inflation, dec.keep.size(), dec.tokens_per_ms);
    }
    // ---- B: independent routing per draft token -> every draft adds ~top_k experts per layer; EVICT trims.
    {
        const Routing root = random_routing(L, X, TOPK);
        std::vector<DraftNode> d(12);
        for (size_t i = 0; i < d.size(); ++i) {
            d[i].parent = static_cast<int32_t>(i) - 1;
            d[i].p_accept = 0.6;
            d[i].experts = random_routing(L, X, TOPK);
        }
        const Decision dec = ctl.decide(root, d);
        VJ_CHECK(dec.ok && dec.keep.size() < 4 && dec.inflation <= cfg.max_inflation);
        const double all = ctl.score(1.0 + 0.6 * (1 - std::pow(0.6, 12)) / 0.4,
                                     EvictController::count_unique(root, d, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, L, X), 13);
        std::printf("  independent routing: K=3 inflation %.2f, K=12 inflation %.2f -> kept %zu (inflation %.2f), "
                    "%.3f tok/ms vs %.3f keeping all 12\n", dec.table.front().inflation, dec.table.back().inflation,
                    dec.keep.size(), dec.inflation, dec.tokens_per_ms, all);
        VJ_CHECK(dec.tokens_per_ms > 1.3 * all);
        const Brute b = brute(ctl, root, d, L, X, cfg.max_inflation);
        VJ_CHECK(std::fabs(b.score - dec.tokens_per_ms) < 1e-9 && b.mask == (1ull << dec.keep.size()) - 1);
    }
    // ---- C: chains, exact vs brute force (random acceptance, mixed correlation, random caps).
    for (int t = 0; t < 200; ++t) {
        ControllerConfig c2;
        c2.n_layers = 6;
        c2.n_experts = 64;
        c2.max_inflation = 1.5 + (rng() % 30) / 10.0;
        const CostModel cm{100.0 + rng() % 400, 5.0, 1.0 + (rng() % 40) / 10.0};
        EvictController e(c2, cm);
        const Routing root = random_routing(6, 64, 4);
        std::vector<DraftNode> d(1 + rng() % 12);
        for (size_t i = 0; i < d.size(); ++i) {
            d[i].parent = static_cast<int32_t>(i) - 1;
            d[i].p_accept = (rng() % 1000) / 1000.0;
            d[i].experts = correlated(root, 64, (rng() % 100) / 100.0);
        }
        const Decision dec = e.decide(root, d);
        const Brute b = brute(e, root, d, 6, 64, c2.max_inflation);
        VJ_CHECK(dec.ok && std::fabs(b.score - dec.tokens_per_ms) < 1e-9 * (1 + b.score));
        VJ_CHECK(dec.inflation <= c2.max_inflation + 1e-12);
    }
    // ---- D: trees, greedy vs exhaustive.
    {
        double worst = 1.0, mean = 0.0;
        int exact = 0;
        const int trials = 300;
        for (int t = 0; t < trials; ++t) {
            ControllerConfig c2;
            c2.n_layers = 4;
            c2.n_experts = 64;
            c2.max_inflation = 2.0 + (rng() % 20) / 10.0;
            EvictController e(c2, CostModel{150.0, 5.0, 2.0 + (rng() % 30) / 10.0});
            const Routing root = random_routing(4, 64, 4);
            std::vector<DraftNode> d(2 + rng() % 11);
            for (size_t i = 0; i < d.size(); ++i) {
                d[i].parent = i == 0 ? -1 : static_cast<int32_t>(rng() % (i + 1)) - 1;  // random tree
                d[i].p_accept = 0.2 + (rng() % 800) / 1000.0;
                d[i].experts = correlated(root, 64, (rng() % 100) / 100.0);
            }
            const Decision dec = e.decide(root, d);
            VJ_CHECK(dec.ok && dec.inflation <= c2.max_inflation + 1e-12);
            for (int32_t i : dec.keep) {
                const int32_t p = d[static_cast<size_t>(i)].parent;
                VJ_CHECK(p < 0 || std::find(dec.keep.begin(), dec.keep.end(), p) != dec.keep.end());
            }
            const Brute b = brute(e, root, d, 4, 64, c2.max_inflation);
            const double r = dec.tokens_per_ms / b.score;
            VJ_CHECK(r <= 1.0 + 1e-9);
            worst = std::min(worst, r);
            mean += r / trials;
            exact += r > 1.0 - 1e-9;
        }
        std::printf("  trees (2..12 nodes): greedy/optimal mean %.4f, worst %.4f, optimal in %d/%d\n", mean, worst, exact, trials);
        VJ_CHECK(mean > 0.99 && worst > 0.9);
    }
    // ---- E: unrouted chains (suffix-tree drafts): inflation modeled by the measured slope.
    {
        EvictController e(cfg, cost);
        e.set_inflation_slope(1.0);
        const Decision d1 = e.decide_chain_unrouted(L * TOPK, std::vector<double>(12, 0.8));
        VJ_CHECK(d1.ok && d1.keep.size() <= 2 && d1.inflation <= 3.0 && d1.table.size() == 10);
        for (int i = 0; i < 40; ++i) e.observe(4, 2, 1.4);  // measured: 4 drafts cost only 1.4x the experts
        VJ_CHECK(std::fabs(e.inflation_slope() - 0.1) < 1e-3);
        const Decision d2 = e.decide_chain_unrouted(L * TOPK, std::vector<double>(12, 0.8));
        VJ_CHECK(d2.ok && d2.keep.size() > d1.keep.size());
        std::printf("  unrouted chain p=0.8: slope 1.0 keeps %zu, measured slope %.3f keeps %zu\n", d1.keep.size(),
                    e.inflation_slope(), d2.keep.size());
        VJ_CHECK(!e.decide_chain_unrouted(0, {0.5}).ok && !e.decide_chain_unrouted(10, {1.5}).ok);
    }
    // ---- F: depth adaptation within [k_min, k_max].
    {
        EvictController e(cfg, cost);
        VJ_CHECK(e.next_depth() >= 3 && e.next_depth() <= 12);
        for (int i = 0; i < 60; ++i) { const uint32_t k = e.next_depth(); e.observe(k, k, 1.2); }
        VJ_CHECK(e.next_depth() == 12);
        for (int i = 0; i < 60; ++i) e.observe(e.next_depth(), 0, 2.0);
        VJ_CHECK(e.next_depth() == 3);
        for (int i = 0; i < 60; ++i) e.observe(8, 5, 1.5);
        VJ_CHECK(e.next_depth() == 6);
    }
    // ---- G: validation
    {
        const Routing root = random_routing(L, X, TOPK);
        std::vector<DraftNode> d(2);
        d[0].experts = d[1].experts = root;
        d[0].p_accept = d[1].p_accept = 0.5;
        d[1].parent = 1;  // forward / self reference
        VJ_CHECK(!ctl.decide(root, d).ok);
        d[1].parent = 0;
        d[1].p_accept = 1.5;
        VJ_CHECK(!ctl.decide(root, d).ok);
        d[1].p_accept = 0.5;
        d[1].experts[3][0] = 600;
        VJ_CHECK(ctl.decide(root, d).error.find("out of range") != std::string::npos);
        d[1].experts.pop_back();
        VJ_CHECK(!ctl.decide(root, d).ok);
        VJ_CHECK(ctl.decide(root, {}).ok && ctl.decide(root, {}).keep.empty());
        VJ_CHECK(!EvictController(ControllerConfig{}, cost).decide(root, {}).ok);
    }
    std::printf("test_evict_controller: counting, fan-out table, chains exact, trees near-optimal, cap, unrouted, "
                "adaptation, validation OK\nPASS\n");
    return 0;
}
