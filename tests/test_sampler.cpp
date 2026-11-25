// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/sampler: greedy fast path and tie-breaking, determinism per seed, top-k / top-p / min-p / repeat penalty,
// empirical distribution at T=1, counter-based RNG formula, and full-vocab vs device candidate-list equivalence.
#include "core/sampler.hpp"
#include "tests/test_common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>

using namespace vegajet::sampling;

int main() {
    // Greedy: argmax, ties -> lowest id, NaN ignored.
    {
        Sampler s;
        const float l[] = {0.1f, 2.0f, -1.0f, 2.0f, NAN};
        VJ_CHECK(s.sample(l, 5) == 1);
        const float n[] = {NAN, NAN};
        VJ_CHECK(s.sample(n, 2) == -1);
        VJ_CHECK(s.sample(nullptr, 0) == -1);
        SamplerParams p; p.top_k = 1; p.temperature = 0.8f;
        Sampler k1(p);
        VJ_CHECK(k1.sample(l, 5) == 1);  // top-k = 1 is greedy at any temperature
        p = {}; p.temperature = 1.0f; p.top_p = 1e-6f;
        VJ_CHECK(Sampler(p).sample(l, 5) == 1);
        p = {}; p.temperature = 1.0f; p.min_p = 1.0f;
        const float u[] = {0.1f, 2.0f, -1.0f, 1.5f};
        VJ_CHECK(Sampler(p).sample(u, 4) == 1);  // min-p = 1 keeps only the max
    }
    // Counter-based RNG matches the documented formula.
    {
        const uint64_t seed = 42, step = 7;
        uint64_t x = seed ^ (step * 0x9E3779B97F4A7C15ull);
        x += 0x9E3779B97F4A7C15ull;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        x ^= x >> 31;
        VJ_CHECK(counter_uniform(seed, step) == static_cast<double>(x >> 11) * 0x1.0p-53);
        for (uint64_t t = 0; t < 1000; ++t) { const double u = counter_uniform(9, t); VJ_CHECK(u >= 0.0 && u < 1.0); }
    }
    // Repeat penalty flips a greedy choice; positive logits divided, negative multiplied.
    {
        SamplerParams p; p.repeat_penalty = 2.0f;
        Sampler s(p);
        const float l[] = {1.0f, 1.5f, -0.5f};
        VJ_CHECK(s.sample(l, 3) == 1);
        s.accept(1);
        VJ_CHECK(s.sample(l, 3) == 0);  // 1.5 / 2 = 0.75 < 1.0
        p.repeat_last_n = 1;
        Sampler w(p);
        w.accept(1); w.accept(0);       // window holds only token 0 now
        VJ_CHECK(w.sample(l, 3) == 1);
    }
    // Determinism: same seed -> same sequence; different seed -> different sequence.
    std::vector<float> logits(1000);
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    for (auto& v : logits) v = nd(rng);
    auto run = [&](uint64_t seed, SamplerParams p) {
        p.seed = seed;
        Sampler s(p);
        std::vector<int32_t> out;
        for (int i = 0; i < 64; ++i) { const int32_t t = s.sample(logits.data(), logits.size()); out.push_back(t); s.accept(t); }
        return out;
    };
    SamplerParams sp; sp.temperature = 0.9f; sp.top_k = 50; sp.top_p = 0.95f; sp.min_p = 0.02f; sp.repeat_penalty = 1.1f;
    VJ_CHECK(run(7, sp) == run(7, sp));
    VJ_CHECK(run(7, sp) != run(8, sp));
    // Empirical distribution at T=1 (no penalty): p = {0.5, 0.3, 0.2}.
    {
        SamplerParams p; p.temperature = 1.0f; p.seed = 123; p.repeat_last_n = 0;
        Sampler s(p);
        const float l[] = {std::log(0.5f), std::log(0.3f), std::log(0.2f)};
        int cnt[3] = {0, 0, 0};
        const int N = 40000;
        for (int i = 0; i < N; ++i) { const int32_t t = s.sample(l, 3); cnt[t]++; s.accept(t); }
        VJ_CHECK(std::fabs(cnt[0] / double(N) - 0.5) < 0.01 && std::fabs(cnt[1] / double(N) - 0.3) < 0.01 &&
                 std::fabs(cnt[2] / double(N) - 0.2) < 0.01);
        // top-p 0.75 keeps {0.5, 0.3}: token 2 never drawn.
        p.top_p = 0.75f;
        Sampler q(p);
        for (int i = 0; i < 5000; ++i) { const int32_t t = q.sample(l, 3); VJ_CHECK(t != 2); q.accept(t); }
        // min-p 0.5 keeps p >= 0.25: token 2 never drawn either.
        p.top_p = 1.0f; p.min_p = 0.5f;
        Sampler m(p);
        for (int i = 0; i < 5000; ++i) { const int32_t t = m.sample(l, 3); VJ_CHECK(t != 2); m.accept(t); }
    }
    // Device candidate list (top-K computed elsewhere) == full-vocab sampling with top_k = K, draw for draw.
    {
        SamplerParams p; p.temperature = 0.8f; p.top_k = 64; p.top_p = 0.9f; p.seed = 99; p.repeat_last_n = 0;
        Sampler full(p), cand(p);
        std::vector<std::pair<float, int32_t>> order;
        for (size_t i = 0; i < logits.size(); ++i) order.push_back({logits[i], static_cast<int32_t>(i)});
        std::sort(order.begin(), order.end(), [](auto& a, auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
        std::vector<Candidate> top;
        for (int i = 0; i < 64; ++i) top.push_back({order[i].second, order[i].first});
        for (int i = 0; i < 200; ++i) {
            const int32_t a = full.sample(logits.data(), logits.size());
            const int32_t b = cand.sample_candidates(top);
            VJ_CHECK(a == b);
            full.accept(a); cand.accept(b);
        }
    }
    // Large vocabulary timing (informational).
    {
        std::vector<float> big(248320);
        for (auto& v : big) v = nd(rng);
        SamplerParams p; p.temperature = 0.7f; p.top_k = 40; p.seed = 5;
        Sampler s(p);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 50; ++i) s.accept(s.sample(big.data(), big.size()));
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 50;
        Sampler g;
        const auto t1 = std::chrono::steady_clock::now();
        for (int i = 0; i < 50; ++i) (void)g.sample(big.data(), big.size());
        const double gus = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t1).count() / 50;
        std::printf("  vocab 248320: top-k 40 sample %.0f us, greedy fast path %.0f us\n", us, gus);
    }
    std::printf("test_sampler: greedy/ties/NaN, RNG formula, penalty, determinism, distribution, top-p/min-p, "
                "candidate-list equivalence OK\nPASS\n");
    return 0;
}
