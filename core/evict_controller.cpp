// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/evict_controller.hpp"

#include <algorithm>
#include <cmath>

namespace vegajet::evict {

EvictController::EvictController(ControllerConfig cfg, CostModel cost) : cfg_(cfg), cost_(cost) {
    if (cfg_.k_min > cfg_.k_max) std::swap(cfg_.k_min, cfg_.k_max);
    depth_ = 0.5 * (cfg_.k_min + cfg_.k_max);
}

double EvictController::score(double expected_tokens, uint64_t unique, uint32_t n_tokens) const {
    const double t = cost_.fixed_us + cost_.per_token_us * n_tokens + cost_.per_expert_us * static_cast<double>(unique);
    return t > 0.0 ? 1000.0 * expected_tokens / t : 0.0;
}

uint64_t EvictController::count_unique(const Routing& root, const std::vector<DraftNode>& draft,
                                       const std::vector<int32_t>& nodes, uint32_t n_layers, uint32_t n_experts) {
    std::vector<uint8_t> seen(size_t(n_layers) * n_experts, 0);
    uint64_t u = 0;
    auto add = [&](const Routing& r) {
        for (uint32_t l = 0; l < n_layers && l < r.size(); ++l)
            for (uint16_t e : r[l])
                if (e < n_experts && !seen[size_t(l) * n_experts + e]) { seen[size_t(l) * n_experts + e] = 1; ++u; }
    };
    add(root);
    for (int32_t i : nodes) add(draft[static_cast<size_t>(i)].experts);
    return u;
}

namespace {

bool check_routing(const Routing& r, uint32_t layers, uint32_t experts, std::string& err, const char* who) {
    if (r.size() != layers) { err = std::string(who) + ": routing must have one entry per MoE layer"; return false; }
    for (const auto& l : r)
        for (uint16_t e : l)
            if (e >= experts) { err = std::string(who) + ": expert id out of range"; return false; }
    return true;
}

std::vector<uint16_t> dedup(std::vector<uint16_t> v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

} // namespace

Decision EvictController::decide(const Routing& root, const std::vector<DraftNode>& draft) const {
    Decision d;
    const uint32_t L = cfg_.n_layers, X = cfg_.n_experts;
    if (L == 0 || X == 0) { d.error = "n_layers and n_experts must be > 0"; return d; }
    if (!check_routing(root, L, X, d.error, "root")) return d;
    const size_t n = draft.size();
    std::vector<double> P(n);
    for (size_t i = 0; i < n; ++i) {
        const DraftNode& nd = draft[i];
        if (nd.parent < -1 || nd.parent >= static_cast<int32_t>(i)) { d.error = "draft nodes must follow their parent"; return d; }
        if (!(nd.p_accept >= 0.0 && nd.p_accept <= 1.0)) { d.error = "p_accept must be in [0, 1]"; return d; }
        if (!check_routing(nd.experts, L, X, d.error, "draft node")) return d;
        P[i] = nd.p_accept * (nd.parent < 0 ? 1.0 : P[static_cast<size_t>(nd.parent)]);
    }

    // Reference counts of every (layer, expert); the root's own experts carry one extra count and never free up.
    std::vector<std::vector<std::vector<uint16_t>>> ex(n);
    std::vector<uint16_t> cnt(size_t(L) * X, 0);
    uint64_t U = 0;
    for (uint32_t l = 0; l < L; ++l)
        for (uint16_t e : dedup(root[l]))
            if (cnt[size_t(l) * X + e]++ == 0) ++U;
    d.root_experts = U;
    for (size_t i = 0; i < n; ++i) {
        ex[i].resize(L);
        for (uint32_t l = 0; l < L; ++l) {
            ex[i][l] = dedup(draft[i].experts[l]);
            for (uint16_t e : ex[i][l])
                if (cnt[size_t(l) * X + e]++ == 0) ++U;
        }
    }
    const double U0 = static_cast<double>(std::max<uint64_t>(1, d.root_experts));

    // Chain fan-out table over the full draft (K = k_min..k_max).
    bool chain = true;
    for (size_t i = 0; i < n; ++i) chain = chain && draft[i].parent == static_cast<int32_t>(i) - 1;
    if (chain && n) {
        std::vector<uint8_t> seen(size_t(L) * X, 0);
        uint64_t u = 0;
        for (uint32_t l = 0; l < L; ++l)
            for (uint16_t e : root[l])
                if (!seen[size_t(l) * X + e]) { seen[size_t(l) * X + e] = 1; ++u; }
        double E = 1.0;
        for (size_t k = 1; k <= n && k <= cfg_.k_max; ++k) {
            for (uint32_t l = 0; l < L; ++l)
                for (uint16_t e : ex[k - 1][l])
                    if (!seen[size_t(l) * X + e]) { seen[size_t(l) * X + e] = 1; ++u; }
            E += P[k - 1];
            if (k < cfg_.k_min) continue;
            DepthScore s;
            s.k = static_cast<uint32_t>(k);
            s.unique_experts = u;
            s.inflation = u / U0;
            s.expected_tokens = E;
            s.time_us = cost_.fixed_us + cost_.per_token_us * (1 + k) + cost_.per_expert_us * u;
            s.tokens_per_ms = score(E, u, static_cast<uint32_t>(1 + k));
            d.table.push_back(s);
        }
    }

    std::vector<uint8_t> in(n, 1);
    std::vector<uint32_t> kids(n, 0);
    for (size_t i = 0; i < n; ++i)
        if (draft[i].parent >= 0) ++kids[static_cast<size_t>(draft[i].parent)];
    double E = 1.0;
    for (double p : P) E += p;
    uint32_t N = static_cast<uint32_t>(1 + n);
    auto marginal = [&](size_t i) {
        uint64_t m = 0;
        for (uint32_t l = 0; l < L; ++l)
            for (uint16_t e : ex[i][l]) m += cnt[size_t(l) * X + e] == 1;
        return m;
    };
    auto remove = [&](size_t i) {
        U -= marginal(i);
        for (uint32_t l = 0; l < L; ++l)
            for (uint16_t e : ex[i][l]) --cnt[size_t(l) * X + e];
        E -= P[i];
        --N;
        in[i] = 0;
        if (draft[i].parent >= 0) --kids[static_cast<size_t>(draft[i].parent)];
    };

    // Subtree helpers (trees): nodes follow their parents, so one forward scan collects a subtree.
    auto subtree = [&](size_t root_i) {
        std::vector<uint8_t> sub(n, 0);
        std::vector<size_t> out;
        for (size_t j = root_i; j < n; ++j) {
            if (!in[j]) continue;
            if (j == root_i || (draft[j].parent >= 0 && sub[static_cast<size_t>(draft[j].parent)])) { sub[j] = 1; out.push_back(j); }
        }
        return out;
    };
    auto freed_by = [&](const std::vector<size_t>& nodes, double& dE) {  // experts freed if `nodes` were removed
        uint64_t freed = 0;
        dE = 0.0;
        for (size_t i : nodes) {
            dE += P[i];
            for (uint32_t l = 0; l < L; ++l)
                for (uint16_t e : ex[i][l]) freed += --cnt[size_t(l) * X + e] == 0;
        }
        for (size_t i : nodes)
            for (uint32_t l = 0; l < L; ++l)
                for (uint16_t e : ex[i][l]) ++cnt[size_t(l) * X + e];
        return freed;
    };
    auto remove_all = [&](const std::vector<size_t>& nodes) {
        for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) remove(*it);  // children before parents
    };

    // 1. Hard inflation cap. Chains: trim the tail (U only grows with depth, so this is the longest feasible prefix).
    // Trees: drop the subtree with the least path probability per freed expert (the hill climb below may re-add).
    if (chain) {
        for (size_t k = n; k > 0 && U / U0 > cfg_.max_inflation; --k) {
            remove(k - 1);
            ++d.pruned_by_inflation;
        }
    }
    while (!chain && U / U0 > cfg_.max_inflation) {
        std::vector<size_t> best_nodes;
        double key = 0.0;
        for (size_t i = 0; i < n; ++i) {
            if (!in[i]) continue;
            const auto nodes = subtree(i);
            double dE = 0.0;
            const uint64_t fr = freed_by(nodes, dE);
            if (fr == 0) continue;
            const double k = dE / static_cast<double>(fr);
            if (best_nodes.empty() || k < key) { best_nodes = nodes; key = k; }
        }
        if (best_nodes.empty()) break;
        d.pruned_by_inflation += static_cast<uint32_t>(best_nodes.size());
        remove_all(best_nodes);
    }

    // 2. Throughput: chains take the best prefix exactly, trees drop leaves greedily while the score improves.
    if (chain) {
        size_t m = 0;
        while (m < n && in[m]) ++m;
        std::vector<uint64_t> Uk(m + 1);
        std::vector<double> Ek(m + 1);
        Uk[m] = U;
        Ek[m] = E;
        for (size_t k = m; k > 0; --k) {  // peel from the tail to get U(k), E(k) for every prefix
            remove(k - 1);
            Uk[k - 1] = U;
            Ek[k - 1] = E;
        }
        size_t best = 0;
        for (size_t k = 1; k <= m; ++k)
            if (score(Ek[k], Uk[k], static_cast<uint32_t>(1 + k)) > score(Ek[best], Uk[best], static_cast<uint32_t>(1 + best))) best = k;
        for (size_t k = 0; k < best; ++k) {  // re-add the kept prefix
            in[k] = 1;
            for (uint32_t l = 0; l < L; ++l)
                for (uint16_t e : ex[k][l]) ++cnt[size_t(l) * X + e];
        }
        U = Uk[best];
        E = Ek[best];
        N = static_cast<uint32_t>(1 + best);
        d.pruned_by_score = static_cast<uint32_t>(m - best);
    } else {
        // Hill climb: best of {remove any subtree, re-add any removed node whose parent is kept} while the score
        // improves and the cap holds. Every accepted move strictly increases the score, so it terminates.
        const uint32_t removed_by_cap = d.pruned_by_inflation;
        for (;;) {
            const double cur = score(E, U, N);
            double best_s = cur * (1.0 + 1e-12);
            std::vector<size_t> best_rm;
            size_t best_add = n;
            for (size_t i = 0; i < n; ++i) {
                if (in[i]) {
                    const auto nodes = subtree(i);
                    double dE = 0.0;
                    const uint64_t fr = freed_by(nodes, dE);
                    const double s2 = score(E - dE, U - fr, N - static_cast<uint32_t>(nodes.size()));
                    if (s2 > best_s) { best_s = s2; best_rm = nodes; best_add = n; }
                } else if (draft[i].parent < 0 || in[static_cast<size_t>(draft[i].parent)]) {
                    uint64_t add = 0;
                    for (uint32_t l = 0; l < L; ++l)
                        for (uint16_t e : ex[i][l]) add += cnt[size_t(l) * X + e] == 0;
                    if ((U + add) / U0 > cfg_.max_inflation) continue;
                    const double s2 = score(E + P[i], U + add, N + 1);
                    if (s2 > best_s) { best_s = s2; best_add = i; best_rm.clear(); }
                }
            }
            if (!best_rm.empty()) {
                remove_all(best_rm);
            } else if (best_add < n) {
                for (uint32_t l = 0; l < L; ++l)
                    for (uint16_t e : ex[best_add][l]) U += cnt[size_t(l) * X + e]++ == 0;
                E += P[best_add];
                ++N;
                in[best_add] = 1;
                if (draft[best_add].parent >= 0) ++kids[static_cast<size_t>(draft[best_add].parent)];
            } else {
                break;
            }
        }
        uint32_t kept = 0;
        for (size_t i = 0; i < n; ++i) kept += in[i];
        d.pruned_by_score = static_cast<uint32_t>(n) - kept - std::min<uint32_t>(removed_by_cap, static_cast<uint32_t>(n) - kept);
    }

    for (size_t i = 0; i < n; ++i)
        if (in[i]) d.keep.push_back(static_cast<int32_t>(i));
    d.unique_experts = U;
    d.inflation = U / U0;
    d.expected_tokens = E;
    d.time_us = cost_.fixed_us + cost_.per_token_us * N + cost_.per_expert_us * static_cast<double>(U);
    d.tokens_per_ms = score(E, U, N);
    d.ok = true;
    return d;
}

Decision EvictController::decide_chain_unrouted(uint64_t root_unique, const std::vector<double>& p_accept) const {
    Decision d;
    if (root_unique == 0) { d.error = "root_unique must be > 0"; return d; }
    for (double p : p_accept)
        if (!(p >= 0.0 && p <= 1.0)) { d.error = "p_accept must be in [0, 1]"; return d; }
    const size_t n = p_accept.size();
    std::vector<double> Ek(n + 1, 1.0);
    std::vector<uint64_t> Uk(n + 1);
    double P = 1.0;
    for (size_t k = 0; k <= n; ++k) {
        if (k) { P *= p_accept[k - 1]; Ek[k] = Ek[k - 1] + P; }
        Uk[k] = static_cast<uint64_t>(std::llround(root_unique * (1.0 + slope_ * k)));
        if (k >= cfg_.k_min && k <= cfg_.k_max) {
            DepthScore s;
            s.k = static_cast<uint32_t>(k);
            s.unique_experts = Uk[k];
            s.inflation = 1.0 + slope_ * k;
            s.expected_tokens = Ek[k];
            s.time_us = cost_.fixed_us + cost_.per_token_us * (1 + k) + cost_.per_expert_us * Uk[k];
            s.tokens_per_ms = score(Ek[k], Uk[k], static_cast<uint32_t>(1 + k));
            d.table.push_back(s);
        }
    }
    size_t best = 0;
    for (size_t k = 1; k <= n; ++k) {
        if (1.0 + slope_ * k > cfg_.max_inflation) { d.pruned_by_inflation = static_cast<uint32_t>(n - k + 1); break; }
        if (score(Ek[k], Uk[k], static_cast<uint32_t>(1 + k)) > score(Ek[best], Uk[best], static_cast<uint32_t>(1 + best))) best = k;
    }
    for (size_t k = 0; k < best; ++k) d.keep.push_back(static_cast<int32_t>(k));
    d.pruned_by_score = static_cast<uint32_t>(n - best - d.pruned_by_inflation);
    d.root_experts = root_unique;
    d.unique_experts = Uk[best];
    d.inflation = 1.0 + slope_ * best;
    d.expected_tokens = Ek[best];
    d.time_us = cost_.fixed_us + cost_.per_token_us * (1 + best) + cost_.per_expert_us * Uk[best];
    d.tokens_per_ms = score(Ek[best], Uk[best], static_cast<uint32_t>(1 + best));
    d.ok = true;
    return d;
}

void EvictController::observe(uint32_t kept, uint32_t accepted, double measured_inflation) {
    accepted = std::min(accepted, kept);
    // Everything accepted: the draft was the limit, probe deeper. Otherwise aim just past the accepted length.
    const double target = accepted == kept ? kept + 2.0 : accepted + 1.0;
    depth_ = (1.0 - cfg_.ewma) * depth_ + cfg_.ewma * target;
    if (kept > 0 && measured_inflation >= 1.0)
        slope_ = (1.0 - cfg_.ewma) * slope_ + cfg_.ewma * (measured_inflation - 1.0) / kept;
}

uint32_t EvictController::next_depth() const {
    const long r = std::lround(depth_);
    return static_cast<uint32_t>(std::clamp<long>(r, cfg_.k_min, cfg_.k_max));
}

} // namespace vegajet::evict
