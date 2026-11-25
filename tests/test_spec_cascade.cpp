// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/spec_cascade: suffix continuation degrees vs brute force, the confidence-gated SUFFIX_ONLY / NEURAL_DRAFTER
// decision and its env toggle, EVICT expert-union dedup + exact budget truncation (hand example and brute force at
// Ornith shapes: 48 MoE layers, 256 experts, top-8), a lossless greedy verify loop on repetitive code / prose /
// math streams (and a random control), and the CPU budget of < 100 us per step at a 16K-token history (-O3).
#include "core/spec_cascade.hpp"
#include "tests/test_common.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>

using namespace vegajet;
using namespace vegajet::spec_cascade;

namespace {

#if defined(__OPTIMIZE__) && !defined(__SANITIZE_ADDRESS__) && !defined(__SANITIZE_THREAD__)
constexpr bool kTimingBuild = true;
#else
constexpr bool kTimingBuild = false;  // sanitizers / -O0: latency is reported, not enforced
#endif

// Longest L such that the last L tokens also occur ending at an earlier index; returns (L, first such end).
std::pair<uint32_t, int64_t> brute_match(const std::vector<int32_t>& h) {
    const int64_t n = static_cast<int64_t>(h.size());
    for (int64_t L = n - 1; L >= 1; --L)
        for (int64_t e = L - 1; e < n - 1; ++e) {
            bool ok = true;
            for (int64_t k = 0; k < L && ok; ++k) ok = h[e - k] == h[n - 1 - k];
            if (ok) return {static_cast<uint32_t>(L), e};
        }
    return {0, -1};
}

// Distinct tokens that followed an occurrence of s (ending before the last history position); optionally the
// number of such occurrences and how many of them were followed by `tok`.
uint32_t brute_degree(const std::vector<int32_t>& h, const std::vector<int32_t>& s, int32_t tok = -1,
                      uint32_t* occ = nullptr, uint32_t* hits = nullptr) {
    std::set<int32_t> f;
    uint32_t o = 0, c = 0;
    const size_t L = s.size(), n = h.size();
    for (size_t e = L - 1; e + 1 < n; ++e)
        if (std::equal(s.begin(), s.end(), h.begin() + static_cast<std::ptrdiff_t>(e + 1 - L))) {
            f.insert(h[e + 1]);
            ++o;
            if (h[e + 1] == tok) ++c;
        }
    if (occ) *occ = o;
    if (hits) *hits = c;
    return static_cast<uint32_t>(f.size());
}

double pct(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(p * static_cast<double>(v.size())))];
}

// --- synthetic token streams -------------------------------------------------------------------------------
std::vector<int32_t> code_stream(size_t n, std::mt19937& rng) {
    // Templated statements over a keyword range with varying identifiers / literals.
    const std::vector<std::vector<int32_t>> tmpl = {
        {101, 102, 103, 900, 104, 105, 106, 900, 107, 901, 108, 900, 109, 110},  // for (int i = 0; i < N; ++i) {
        {111, 900, 112, 902, 113, 114, 903, 112, 902, 113, 115, 904, 116},        //   a[i] = b[i] + c;
        {117, 118, 119, 902, 120, 121, 122, 123},                                 //   if (x) return y;
        {124, 116},                                                               // }
    };
    std::vector<int32_t> s;
    while (s.size() < n) {
        const int32_t i_var = 1000 + static_cast<int32_t>(rng() % 6u), lit = 2000 + static_cast<int32_t>(rng() % 40u);
        const int32_t a = 1100 + static_cast<int32_t>(rng() % 30u), b = 1200 + static_cast<int32_t>(rng() % 30u);
        for (int32_t t : tmpl[0]) s.push_back(t == 900 ? i_var : t == 901 ? lit : t);
        for (uint32_t k = 0, nb = 1u + rng() % 3u; k < nb; ++k) {
            const auto& body = tmpl[1u + rng() % 2u];
            for (int32_t t : body) s.push_back(t == 900 ? a : t == 902 ? i_var : t == 903 ? b : t == 904 ? lit : t);
        }
        for (int32_t t : tmpl[3]) s.push_back(t);
    }
    s.resize(n);
    return s;
}

std::vector<int32_t> prose_stream(size_t n, std::mt19937& rng) {
    // Zipf words with recurring multi-word phrases (names, idioms, repeated clauses).
    std::vector<double> w(5000);
    for (size_t i = 0; i < w.size(); ++i) w[i] = 1.0 / static_cast<double>(i + 1);
    std::discrete_distribution<int32_t> zipf(w.begin(), w.end());
    std::vector<std::vector<int32_t>> phrases(40);
    for (auto& p : phrases) {
        p.resize(6u + rng() % 7u);
        for (auto& t : p) t = 3000 + zipf(rng);
    }
    std::vector<int32_t> s;
    while (s.size() < n) {
        if (rng() % 10u < 4u) {
            const auto& p = phrases[rng() % phrases.size()];
            s.insert(s.end(), p.begin(), p.end());
        } else {
            for (uint32_t k = 0, nw = 1u + rng() % 4u; k < nw; ++k) s.push_back(3000 + zipf(rng));
        }
    }
    s.resize(n);
    return s;
}

std::vector<int32_t> math_stream(size_t n, std::mt19937& rng) {
    // Step-by-step arithmetic: "x_k = x_{k-1} + d ;" with digit tokens, restating the same expression shape.
    std::vector<int32_t> s;
    auto digits = [&](uint32_t v) {
        std::vector<int32_t> d;
        do { d.push_back(10 + static_cast<int32_t>(v % 10u)); v /= 10u; } while (v);
        s.insert(s.end(), d.rbegin(), d.rend());
    };
    uint32_t x = rng() % 50u;
    while (s.size() < n) {
        const uint32_t d = 1u + rng() % 9u;
        s.push_back(40); s.push_back(41);  // "x" "="
        digits(x); s.push_back(42); digits(d); s.push_back(41); digits(x + d); s.push_back(43);
        x = (x + d) % 1000u;
        if (rng() % 16u == 0) { s.push_back(44); s.push_back(45); s.push_back(46); s.push_back(43); }  // "so we get ;"
    }
    s.resize(n);
    return s;
}

std::vector<int32_t> random_stream(size_t n, std::mt19937& rng) {
    std::vector<int32_t> s(n);
    for (auto& t : s) t = static_cast<int32_t>(rng() % 50000u);
    return s;
}

Routing routing_of(int32_t tok) {
    Routing r(48);
    uint64_t h = static_cast<uint64_t>(static_cast<uint32_t>(tok)) * 0x9E3779B97F4A7C15ull + 1;
    for (uint32_t l = 0; l < 48; ++l)
        for (uint32_t k = 0; k < 8; ++k) {
            h ^= h >> 31; h *= 0xBF58476D1CE4E5B9ull; h ^= h >> 27;
            r[l].push_back(static_cast<uint16_t>(h % 256u));
        }
    return r;
}

struct LoopStats {
    uint64_t cycles = 0, suffix_cycles = 0, suffix_proposed = 0, suffix_accepted = 0;
    uint64_t neural_proposed = 0, neural_accepted = 0, evicted = 0;
};

// Greedy speculative decoding of `target` (the target model's own greedy stream) from a prompt prefix.
LoopStats run_loop(const std::vector<int32_t>& target, size_t prompt, CascadeConfig cfg, std::mt19937& rng,
                   std::vector<int32_t>* out) {
    ZeroTaxCascade c(cfg);
    c.reserve(target.size() + 16);
    out->assign(target.begin(), target.begin() + static_cast<std::ptrdiff_t>(prompt));
    c.commit(*out);
    UnionBudget budget;
    budget.max_unique_per_layer = 40;  // ~5 tokens of top-8 fan-out per MoE layer
    LoopStats st;
    while (out->size() < target.size()) {
        const size_t pos = out->size();
        CascadeStep step = c.decide();
        std::vector<int32_t> draft;
        const bool sfx = step.source == DraftSource::SUFFIX_ONLY;
        if (sfx) {
            draft = step.draft;
        } else {
            for (size_t j = 0; j < 4 && pos + j < target.size(); ++j)  // mock neural drafter: ~65% per token
                draft.push_back(rng() % 100u < 65u ? target[pos + j] : -5);
        }
        std::vector<Routing> cand = {routing_of(out->back())};
        for (int32_t t : draft) cand.push_back(routing_of(t));
        ExpertUnion u;
        std::string err;
        VJ_CHECK(expert_union_truncate(cand, 48, 256, budget, &u, &err));
        const auto kept = truncate_draft(draft, u);
        VJ_CHECK(kept.size() + 1u == u.keep && u.max_layer_after[u.keep - 1u] <= 40u);
        st.evicted += draft.size() - kept.size();
        size_t acc = 0;
        while (acc < kept.size() && pos + acc < target.size() && kept[acc] == target[pos + acc]) ++acc;
        std::vector<int32_t> commit(target.begin() + static_cast<std::ptrdiff_t>(pos),
                                    target.begin() + static_cast<std::ptrdiff_t>(std::min(target.size(), pos + acc + 1u)));
        out->insert(out->end(), commit.begin(), commit.end());  // accepted drafts + the verifier's own token
        c.commit(commit);
        ++st.cycles;
        (sfx ? st.suffix_proposed : st.neural_proposed) += kept.size();
        (sfx ? st.suffix_accepted : st.neural_accepted) += acc;
        if (sfx) ++st.suffix_cycles;
    }
    VJ_CHECK(c.stats().steps == st.cycles && c.stats().suffix_only == st.suffix_cycles);
    return st;
}

} // namespace

int main() {
    std::mt19937 rng(20261125);
    uint64_t checks = 0;

    // --- continuation degrees and confidences vs brute force ---
    for (int alpha : {2, 3, 5}) {
        for (int rep = 0; rep < 3; ++rep) {
            CascadeConfig cfg;
            cfg.enabled = true;
            cfg.max_draft = 6;
            cfg.min_match = 1;
            cfg.min_confident = 1;
            ZeroTaxCascade c(cfg);
            std::vector<int32_t> h;
            for (int i = 0; i < 160; ++i) {
                const int32_t t = static_cast<int32_t>(rng() % static_cast<uint32_t>(alpha));
                c.commit(t);
                h.push_back(t);
                const auto ref = brute_match(h);
                const SuffixDraft d = c.suffix_draft();
                VJ_CHECK(d.match_length == ref.first);
                if (!ref.first) { VJ_CHECK(d.tokens.empty()); continue; }
                std::vector<int32_t> ctx(h.end() - ref.first, h.end());
                std::vector<int32_t> seq = h;  // history + chain, for the order-N support context
                VJ_CHECK(d.tokens.size() == std::min<size_t>(6, h.size() - 1 - static_cast<size_t>(ref.second)));
                double conf = 1.0;
                for (size_t j = 0; j < d.tokens.size(); ++j) {
                    VJ_CHECK(d.tokens[j] == h[static_cast<size_t>(ref.second) + 1 + j]);
                    const uint32_t deg = brute_degree(h, ctx);
                    VJ_CHECK(d.degree[j] == deg && deg >= 1);
                    uint32_t occ = 0, hits = 0;
                    if (seq.size() >= 4) brute_degree(h, std::vector<int32_t>(seq.end() - 4, seq.end()), d.tokens[j], &occ, &hits);
                    VJ_CHECK(d.support[j] == occ);
                    const double p = static_cast<double>(hits) / (occ + 1.0);
                    VJ_CHECK(std::fabs(d.prob[j] - p) < 1e-6);
                    conf *= std::min(1.0 / deg, p);
                    VJ_CHECK(std::fabs(d.confidence[j] - conf) < 1e-6);
                    VJ_CHECK(j == 0 || d.confidence[j] <= d.confidence[j - 1]);
                    ctx.push_back(d.tokens[j]);
                    seq.push_back(d.tokens[j]);
                }
                ++checks;
            }
        }
    }

    // --- decision gate ---
    {
        CascadeConfig cfg;
        cfg.enabled = true;
        const std::vector<int32_t> period = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        // Seen once: the long match is unambiguous (degree 1) but carries no evidence -> neural.
        ZeroTaxCascade once(cfg);
        once.commit(period);
        once.commit({10, 11, 12, 13});
        const CascadeStep s1 = once.decide();
        VJ_CHECK(s1.suffix.match_length == 4 && s1.suffix.degree[0] == 1 && s1.suffix.support[0] == 1);
        VJ_CHECK(s1.suffix.confidence[0] == 0.5f && s1.suffix.confidence[1] == 0.25f && s1.suffix.confident == 1);
        VJ_CHECK(s1.source == DraftSource::NEURAL_DRAFTER && s1.draft.empty());
        sfx::CascadeHybrid legacy;  // match + chain length only: takes the seen-once draft
        legacy.commit(period);
        legacy.commit({10, 11, 12, 13});
        VJ_CHECK(legacy.decide().source == sfx::DraftSource::kSuffix);
        // Seen three times: 3 / 4 per token -> 0.75, 0.56, 0.42, 0.32 >= 0.3 -> four confident tokens.
        ZeroTaxCascade rep(cfg);
        for (int k = 0; k < 3; ++k) rep.commit(period);
        rep.commit({10, 11, 12, 13});
        const CascadeStep s3 = rep.decide();
        VJ_CHECK(s3.source == DraftSource::SUFFIX_ONLY && (s3.draft == std::vector<int32_t>{14, 15, 16, 17}));
        VJ_CHECK(s3.suffix.support[0] == 3 && s3.suffix.prob[0] == 0.75f && s3.suffix.confident == 4);
        VJ_CHECK(rep.stats().suffix_only == 1 && rep.stats().suffix_tokens == 4);
        // Ambiguous long context: "1 2 3 4" (preceded by 23, unique) was followed by 5 twice and by 9 once.
        ZeroTaxCascade amb(cfg);
        amb.commit({20, 1, 2, 3, 4, 5, 6, 7, 8, 21, 1, 2, 3, 4, 5, 6, 7, 8, 22, 1, 2, 3, 4, 9, 6, 7, 8, 23, 1, 2, 3, 4});
        const CascadeStep sa = amb.decide();
        VJ_CHECK(sa.suffix.match_length == 4 && sa.suffix.tokens[0] == 5 && sa.suffix.degree[0] == 2);
        VJ_CHECK(sa.suffix.support[0] == 3 && sa.suffix.prob[0] == 0.5f && sa.suffix.confidence[0] == 0.5f);
        VJ_CHECK(sa.source == DraftSource::NEURAL_DRAFTER);  // 0.5 * 2/3 * 2/3 * 2/3 drops below 0.3 after 2 tokens
        // Chain of only 3 continuation tokens (earlier occurrence ends 3 tokens before the history) -> neural,
        // even with the confidence gate off.
        CascadeConfig loose = cfg;
        loose.min_confidence = 0.0;
        ZeroTaxCascade e(loose);
        e.commit({5, 1, 2, 3, 1, 2, 3, 1});
        const CascadeStep se = e.peek();
        VJ_CHECK(se.suffix.match_length == 4 && (se.suffix.tokens == std::vector<int32_t>{2, 3, 1}));
        VJ_CHECK(se.suffix.confident == 3 && se.source == DraftSource::NEURAL_DRAFTER);
        CascadeConfig three = loose;
        three.min_confident = 3;  // the threshold is configurable
        ZeroTaxCascade e3(three);
        e3.commit({5, 1, 2, 3, 1, 2, 3, 1});
        VJ_CHECK(e3.peek().source == DraftSource::SUFFIX_ONLY && e3.peek().draft.size() == 3);
        // Disabled: always neural, index not queried.
        ZeroTaxCascade off;
        for (int k = 0; k < 3; ++k) off.commit(period);
        off.commit({10, 11, 12, 13});
        const CascadeStep so = off.decide();
        VJ_CHECK(so.source == DraftSource::NEURAL_DRAFTER && so.suffix.tokens.empty() && off.stats().neural == 1);
        VJ_CHECK(std::string(draft_source_name(DraftSource::SUFFIX_ONLY)) == "SUFFIX_ONLY");
        VJ_CHECK(std::string(draft_source_name(DraftSource::NEURAL_DRAFTER)) == "NEURAL_DRAFTER");
        checks += 6;
    }

    // --- env toggle ---
    {
        std::string note;
        VJ_CHECK(unsetenv(kCascadeEnv) == 0 && unsetenv(kCascadeMinEnv) == 0);
        VJ_CHECK(!config_from_env({}, &note).enabled && note.empty());
        VJ_CHECK(setenv(kCascadeEnv, "1", 1) == 0 && config_from_env({}, &note).enabled && note.empty());
        VJ_CHECK(setenv(kCascadeMinEnv, "6", 1) == 0 && config_from_env().min_confident == 6);
        VJ_CHECK(setenv(kCascadeMinEnv, "0", 1) == 0 && config_from_env({}, &note).min_confident == 4 && !note.empty());
        VJ_CHECK(setenv(kCascadeEnv, "yes", 1) == 0 && !config_from_env({}, &note).enabled && !note.empty());
        CascadeConfig on;
        on.enabled = true;
        VJ_CHECK(setenv(kCascadeEnv, "0", 1) == 0 && !config_from_env(on).enabled);
        VJ_CHECK(unsetenv(kCascadeEnv) == 0 && unsetenv(kCascadeMinEnv) == 0);
        checks += 1;
    }

    // --- EVICT expert union: hand example ---
    {
        const std::vector<Routing> c = {
            {{0, 1}, {2, 3}},
            {{1, 4}, {3}},
            {{5, 6}, {2, 7, 7}},
            {{0}, {2}},
        };
        ExpertUnion u;
        std::string err;
        VJ_CHECK(expert_union_truncate(c, 2, 8, {}, &u, &err));
        VJ_CHECK((u.unique_after == std::vector<uint64_t>{4, 5, 8, 8}) && (u.max_layer_after == std::vector<uint32_t>{2, 3, 5, 5}));
        VJ_CHECK(u.keep == 4 && !u.truncated && u.unique_kept == 8 && u.fan_out == 2.0);
        VJ_CHECK((u.experts[0] == std::vector<uint16_t>{0, 1, 4, 5, 6}) && (u.experts[1] == std::vector<uint16_t>{2, 3, 7}));
        UnionBudget b;
        b.max_unique_total = 6;
        VJ_CHECK(expert_union_truncate(c, 2, 8, b, &u, &err) && u.keep == 2 && u.truncated && u.unique_kept == 5);
        VJ_CHECK((u.experts[0] == std::vector<uint16_t>{0, 1, 4}) && (u.experts[1] == std::vector<uint16_t>{2, 3}) && u.fan_out == 1.25);
        VJ_CHECK((truncate_draft({10, 11, 12}, u) == std::vector<int32_t>{10}));
        b.max_unique_total = 8;
        VJ_CHECK(expert_union_truncate(c, 2, 8, b, &u, &err) && u.keep == 4 && !u.truncated);
        b = {};
        b.max_unique_per_layer = 3;
        VJ_CHECK(expert_union_truncate(c, 2, 8, b, &u, &err) && u.keep == 2);
        b = {};
        b.max_unique_total = 3;  // even the root alone is over budget: keep min_keep = 1
        VJ_CHECK(expert_union_truncate(c, 2, 8, b, &u, &err) && u.keep == 1 && u.unique_kept == 4 && truncate_draft({10, 11, 12}, u).empty());
        b.min_keep = 3;
        VJ_CHECK(expert_union_truncate(c, 2, 8, b, &u, &err) && u.keep == 3);
        VJ_CHECK(expert_union_truncate({}, 2, 8, b, &u, &err) && u.keep == 0 && u.unique_kept == 0);
        VJ_CHECK(!expert_union_truncate({{{0}, {8}}}, 2, 8, {}, &u, &err) && !err.empty());
        VJ_CHECK(!expert_union_truncate({{{0}}}, 2, 8, {}, &u, &err));
        VJ_CHECK(!expert_union_truncate(c, 2, 0, {}, &u, &err));
        checks += 5;
    }

    // --- EVICT expert union vs brute force at Ornith shapes ---
    {
        for (int it = 0; it < 150; ++it) {
            const uint32_t M = 1u + rng() % 9u;
            std::vector<Routing> c(M, Routing(48));
            for (auto& r : c)
                for (auto& l : r)
                    for (int k = 0; k < 8; ++k) l.push_back(static_cast<uint16_t>(rng() % (it % 3 == 0 ? 24u : 256u)));
            UnionBudget b;
            b.max_unique_total = it % 2 ? 384u + rng() % 2000u : 0u;
            b.max_unique_per_layer = it % 5 == 0 ? 8u + rng() % 40u : 0u;
            ExpertUnion u;
            std::string err;
            VJ_CHECK(expert_union_truncate(c, 48, 256, b, &u, &err));
            std::vector<std::set<uint16_t>> sets(48);
            uint32_t want_keep = M;
            bool cut = false;
            for (uint32_t i = 0; i < M; ++i) {
                uint64_t tot = 0;
                uint32_t mx = 0;
                for (uint32_t l = 0; l < 48; ++l) {
                    sets[l].insert(c[i][l].begin(), c[i][l].end());
                    tot += sets[l].size();
                    mx = std::max<uint32_t>(mx, static_cast<uint32_t>(sets[l].size()));
                }
                VJ_CHECK(u.unique_after[i] == tot && u.max_layer_after[i] == mx);
                if (!cut && ((b.max_unique_total && tot > b.max_unique_total) || (b.max_unique_per_layer && mx > b.max_unique_per_layer))) {
                    cut = true;
                    want_keep = std::max(i, 1u);
                }
            }
            VJ_CHECK(u.keep == want_keep && u.truncated == (want_keep < M));
            std::vector<std::set<uint16_t>> kept(48);
            for (uint32_t i = 0; i < u.keep; ++i)
                for (uint32_t l = 0; l < 48; ++l) kept[l].insert(c[i][l].begin(), c[i][l].end());
            for (uint32_t l = 0; l < 48; ++l) VJ_CHECK(std::vector<uint16_t>(kept[l].begin(), kept[l].end()) == u.experts[l]);
            ++checks;
        }
    }

    // --- lossless greedy verify loop on code / prose / math streams + random control ---
    struct Named { const char* name; std::vector<int32_t> s; };
    std::vector<Named> streams = {
        {"code", code_stream(16384, rng)}, {"prose", prose_stream(16384, rng)},
        {"math", math_stream(16384, rng)}, {"random", random_stream(16384, rng)},
    };
    {
        CascadeConfig cfg;
        cfg.enabled = true;
        for (const auto& st : streams) {
            const std::vector<int32_t> target(st.s.begin(), st.s.begin() + 6000);
            std::vector<int32_t> out;
            const LoopStats ls = run_loop(target, 512, cfg, rng, &out);
            VJ_CHECK(out == target);
            const double share = static_cast<double>(ls.suffix_cycles) / static_cast<double>(ls.cycles);
            const double sacc = ls.suffix_proposed ? static_cast<double>(ls.suffix_accepted) / static_cast<double>(ls.suffix_proposed) : 0.0;
            const double tpc = static_cast<double>(target.size() - 512) / static_cast<double>(ls.cycles);
            std::printf("[spec_cascade] %-6s: %llu cycles, SUFFIX_ONLY %.1f%% (GPU drafter skipped), suffix acceptance %.1f%%, "
                        "neural acceptance %.1f%%, %.2f tokens/cycle, %llu drafts evicted by the expert budget\n",
                        st.name, static_cast<unsigned long long>(ls.cycles), 100.0 * share, 100.0 * sacc,
                        ls.neural_proposed ? 100.0 * static_cast<double>(ls.neural_accepted) / static_cast<double>(ls.neural_proposed) : 0.0,
                        tpc, static_cast<unsigned long long>(ls.evicted));
            if (std::string(st.name) == "random") VJ_CHECK(ls.suffix_cycles == 0);
            else VJ_CHECK(ls.suffix_cycles > 0 && sacc > 0.5);
            ++checks;
        }
    }

    // --- confidence threshold sweep (report only; every run must stay lossless) ---
    for (const auto& st : streams) {
        if (std::string(st.name) == "random") continue;
        const std::vector<int32_t> target(st.s.begin(), st.s.begin() + 6000);
        std::string line;
        for (double thr : {0.0, 0.15, 0.3, 0.5}) {
            CascadeConfig cfg;
            cfg.enabled = true;
            cfg.min_confidence = thr;
            std::vector<int32_t> out;
            const LoopStats ls = run_loop(target, 512, cfg, rng, &out);
            VJ_CHECK(out == target);
            char buf[128];
            std::snprintf(buf, sizeof(buf), "  thr %.2f: %4.1f%% suffix-only @ %5.1f%% acc", thr,
                          100.0 * static_cast<double>(ls.suffix_cycles) / static_cast<double>(ls.cycles),
                          ls.suffix_proposed ? 100.0 * static_cast<double>(ls.suffix_accepted) / static_cast<double>(ls.suffix_proposed) : 0.0);
            line += buf;
            ++checks;
        }
        std::printf("[spec_cascade] %-6s sweep:%s\n", st.name, line.c_str());
    }

    // --- CPU latency per step (commit + decision) at a 16K-token history ---
    {
        CascadeConfig cfg;
        cfg.enabled = true;
        for (const auto& st : streams) {
            ZeroTaxCascade c(cfg);
            c.reserve(st.s.size());
            std::vector<double> us;
            us.reserve(st.s.size());
            uint64_t sink = 0;
            for (int32_t t : st.s) {
                const auto t0 = std::chrono::steady_clock::now();
                c.commit(t);
                const CascadeStep s = c.decide();
                const auto t1 = std::chrono::steady_clock::now();
                sink += s.draft.size();
                us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            double mean = 0.0;
            for (double v : us) mean += v;
            mean /= static_cast<double>(us.size());
            const double p99 = pct(us, 0.99), mx = pct(us, 1.0);
            std::printf("[spec_cascade] %-6s 16K history: mean %.2f us, p99 %.2f us, max %.2f us per step (%llu draft tokens)%s\n",
                        st.name, mean, p99, mx, static_cast<unsigned long long>(sink),
                        kTimingBuild ? "" : " [not enforced: sanitizer / unoptimized build]");
            if (kTimingBuild) VJ_CHECK(mean < 100.0 && p99 < 100.0);
            ++checks;
        }
    }

    std::printf("test_spec_cascade: all %llu checks passed\n", static_cast<unsigned long long>(checks));
    return 0;
}
