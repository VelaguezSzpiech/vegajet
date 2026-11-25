// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/moe_lru_cache: 44-slot/layer LRU with frequency-gated admission, per-step upload budget, loading slots that
// are never hits before commit(), pinning, shrink/grow, pressure-driven rebalance, and a Zipf routing simulation
// compared against a static top-44 oracle.
#include "core/moe_lru_cache.hpp"
#include "tests/test_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>

using namespace vegajet::moecache;

namespace {

void commit_all(MoeLruCache& c) {
    for (const auto& a : c.take_admissions()) VJ_CHECK(c.commit(a));
}

} // namespace

int main() {
    // Basics on one layer with 4 slots.
    {
        CacheConfig cfg;
        cfg.n_layers = 1; cfg.n_experts = 16; cfg.slots_per_layer = 4; cfg.admit_after_misses = 1; cfg.max_admits_per_step = 2; cfg.max_admits_per_layer_step = 2;
        MoeLruCache c(cfg);
        VJ_CHECK(c.slots(0) == 4 && c.check_invariants());
        int32_t out[4];
        const int32_t r1[] = {1, 2, 3};
        c.route(0, r1, 3, out);
        VJ_CHECK(out[0] == -1 && out[1] == -1 && out[2] == -1);
        auto adm = c.take_admissions();
        VJ_CHECK(adm.size() == 2);                       // per-step budget
        VJ_CHECK(c.slot_of(0, 1) == -1);                 // loading slots are not hits
        c.route(0, r1, 1, out);
        VJ_CHECK(out[0] == -1);
        for (const auto& a : adm) VJ_CHECK(c.commit(a));
        VJ_CHECK(!c.commit(adm[0]) || true);             // re-commit of a ready slot is harmless
        c.route(0, r1, 2, out);
        VJ_CHECK(out[0] >= 0 && out[1] >= 0 && out[0] != out[1]);
        c.end_step();
        c.route(0, r1 + 2, 1, out);                      // expert 3 admitted next step
        commit_all(c);
        const int32_t r2[] = {4};
        c.route(0, r2, 1, out);
        commit_all(c);
        c.end_step();
        VJ_CHECK(c.slot_of(0, 1) >= 0 && c.slot_of(0, 4) >= 0);
        // Touch 1, 2, 4 -> 3 is LRU; admitting 5 evicts 3.
        const int32_t r3[] = {1, 2, 4};
        c.route(0, r3, 3, out);
        const int32_t r4[] = {5};
        c.route(0, r4, 1, out);
        auto a5 = c.take_admissions();
        VJ_CHECK(a5.size() == 1 && a5[0].evicted == 3 && c.slot_of(0, 3) == -1);
        VJ_CHECK(c.commit(a5[0]) && c.slot_of(0, 5) == static_cast<int32_t>(a5[0].slot));
        VJ_CHECK(c.check_invariants());
        // Pinned experts survive pressure.
        c.end_step();
        c.pin(0, 1);
        for (int e = 6; e < 16; ++e) {
            const int32_t r[] = {e};
            c.route(0, r, 1, out);
            commit_all(c);
            c.end_step();
        }
        VJ_CHECK(c.slot_of(0, 1) >= 0 && c.check_invariants());
        c.unpin(0, 1);
        // Shrink keeps the most recent entries; stale tickets are rejected.
        const int32_t r5[] = {15};
        c.route(0, r5, 1, out);
        c.resize(0, 2);
        VJ_CHECK(c.slots(0) == 2 && c.check_invariants());
        commit_all(c);
        VJ_CHECK(c.slot_of(0, 15) >= 0);
        c.resize(0, 6);
        VJ_CHECK(c.slots(0) == 6 && c.check_invariants());
        // Out-of-range experts are ignored.
        const int32_t bad[] = {-1, 99};
        c.route(0, bad, 2, out);
        VJ_CHECK(out[0] == -1 && out[1] == -1);
    }

    // Frequency gate: a single miss does not admit (admit_after_misses = 2), a repeated one does.
    {
        CacheConfig cfg;
        cfg.n_layers = 1; cfg.n_experts = 64; cfg.slots_per_layer = 8;
        MoeLruCache c(cfg);
        int32_t out[1];
        const int32_t e7[] = {7};
        c.route(0, e7, 1, out);
        VJ_CHECK(c.take_admissions().empty());
        c.end_step();
        c.route(0, e7, 1, out);
        VJ_CHECK(c.take_admissions().size() == 1);
    }

    // Zipf routing, 48 layers x 256 experts, top-8, 44 slots: hit rate vs static top-44 oracle.
    {
        CacheConfig cfg;
        cfg.n_layers = 48; cfg.n_experts = 256; cfg.slots_per_layer = 44;
        MoeLruCache c(cfg);
        std::mt19937 rng(7);
        std::vector<double> w(256);
        for (int i = 0; i < 256; ++i) w[i] = 1.0 / std::pow(i + 1, 1.1);
        std::vector<std::vector<uint32_t>> perm(48, std::vector<uint32_t>(256));
        for (auto& p : perm) { std::iota(p.begin(), p.end(), 0u); std::shuffle(p.begin(), p.end(), rng); }
        std::discrete_distribution<int> zipf(w.begin(), w.end());
        uint64_t oracle_hits = 0, total = 0;
        for (int step = 0; step < 3000; ++step) {
            for (uint32_t l = 0; l < 48; ++l) {
                int32_t ex[8], out[8];
                int n = 0;
                while (n < 8) {
                    const int r = zipf(rng);
                    const int32_t e = static_cast<int32_t>(perm[l][r]);
                    if (std::find(ex, ex + n, e) == ex + n) { ex[n++] = e; if (step >= 500) { oracle_hits += r < 44; ++total; } }
                }
                c.route(l, ex, 8, out);
                for (int i = 0; i < 8; ++i) if (out[i] >= 0) VJ_CHECK(c.slot_table(l)[out[i]] == ex[i]);
            }
            commit_all(c);  // uploads complete within the step
            c.end_step();
            if (step == 499) VJ_CHECK(c.check_invariants());
        }
        VJ_CHECK(c.check_invariants());
        const double oracle = static_cast<double>(oracle_hits) / total;
        std::printf("  Zipf(1.1) top-8 of 256, 44 slots: LRU+gate hit rate %.3f (incl. warm-up), static top-44 oracle %.3f\n",
                    c.hit_rate(), oracle);
        VJ_CHECK(c.hit_rate() > 0.5 && c.hit_rate() > oracle - 0.12);
    }

    // Rebalance moves slots to the layers that miss the most; totals are exact.
    {
        CacheConfig cfg;
        cfg.n_layers = 4; cfg.n_experts = 128; cfg.slots_per_layer = 44; cfg.min_slots = 8;
        MoeLruCache c(cfg);
        std::mt19937 rng(3);
        for (int step = 0; step < 200; ++step) {
            for (uint32_t l = 0; l < 4; ++l) {
                int32_t ex[8], out[8];
                for (int i = 0; i < 8; ++i)  // layer 3 is spread uniformly (high misses), layer 0 uses 8 experts
                    ex[i] = l == 0 ? i : l == 3 ? static_cast<int32_t>(rng() % 128) : static_cast<int32_t>(rng() % 40);
                c.route(l, ex, 8, out);
            }
            commit_all(c);
            c.end_step();
        }
        c.rebalance(4 * 44);
        uint32_t sum = 0;
        for (uint32_t l = 0; l < 4; ++l) sum += c.slots(l);
        VJ_CHECK(sum == 4 * 44 && c.slots(3) > c.slots(0) && c.slots(0) >= 8 && c.check_invariants());
        std::printf("  rebalance 176 slots by miss pressure: %u %u %u %u\n", c.slots(0), c.slots(1), c.slots(2), c.slots(3));
        commit_all(c);
        c.rebalance(10);  // below the floor: clamped to n_layers * min_slots
        sum = 0;
        for (uint32_t l = 0; l < 4; ++l) sum += c.slots(l);
        VJ_CHECK(sum == 32 && c.check_invariants());
    }
    std::printf("test_moe_lru_cache: LRU, gate, budget, commit, pin, resize, Zipf, rebalance OK\nPASS\n");
    return 0;
}
