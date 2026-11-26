// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// EVICT prefix-depth controller: range invariants, cold start, first-token rejection with target-only progress,
// full acceptance, unique-expert pressure, and the code/prose replay used by the Claude trace contract.
#include "core/evict_depth.hpp"
#include "tests/test_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace vegajet;

namespace {

struct Row {
    uint32_t proposed = 0;
    uint64_t unique = 0;
    uint32_t verified = 0;
    int32_t correction = -1;
    uint32_t emitted = 0;
    bool target_only = false;
    bool probing = false;
};

// One replay step.  The controller only chooses the prefix limit; the "target" (the `accept` argument) is the
// authority for the verified prefix, and the caller builds the emitted tokens from that decision alone.
Row replay_step(EvictDepth& depth, uint32_t capacity, uint32_t accept, uint64_t unique, int32_t& next_token) {
    Row r;
    r.unique = unique;
    const uint32_t k = depth.choose(capacity, unique);
    r.proposed = k;
    if (k == 0) {
        r.target_only = true;
        r.correction = next_token++;
        r.emitted = 1;  // ordinary target decode always makes progress
        return r;
    }
    r.verified = std::min(accept, k);
    r.correction = next_token++;
    depth.observe(k, r.verified);
    r.probing = depth.last_choice().probing;
    r.emitted = r.verified + 1;  // accepted drafts + the target-authorized correction/bonus token
    return r;
}

void check_rows(const std::vector<Row>& rows) {
    uint64_t emitted = 0, verified = 0;
    for (const Row& r : rows) {
        VJ_CHECK(r.emitted >= 1);  // no step before EOS may emit nothing
        VJ_CHECK(r.correction >= 0);
        if (r.proposed == 0) {
            VJ_CHECK(r.target_only && r.verified == 0 && r.emitted == 1);
        } else {
            VJ_CHECK(!r.target_only);
            VJ_CHECK(r.verified <= r.proposed);
            VJ_CHECK(r.emitted == r.verified + 1);
        }
        emitted += r.emitted;
        verified += r.verified;
    }
    VJ_CHECK(emitted == rows.size() + verified);
}

EvictDepth warm_full(uint32_t max_depth = 8) {
    EvictDepth d(max_depth);
    for (int i = 0; i < 10; ++i) d.observe(8, 8);
    return d;
}

} // namespace

int main() {
    // ---- range invariants across cold, warm and noisy histories
    {
        std::mt19937_64 rng(1234);
        for (uint32_t max_depth : {0u, 1u, 3u, 8u, 32u}) {
            EvictDepth d(max_depth);
            for (int i = 0; i < 2000; ++i) {
                const uint32_t available = static_cast<uint32_t>(rng() % 40);
                const uint64_t unique = rng() % 100000;
                const uint32_t k = d.choose(available, unique);
                VJ_CHECK(k <= std::min(available, max_depth));
                VJ_CHECK(k <= d.last_choice().cap);
                VJ_CHECK(d.last_choice().depth == k);
                if (rng() % 3 == 0) {
                    const uint32_t attempted = static_cast<uint32_t>(rng() % 17);
                    const uint32_t accepted = static_cast<uint32_t>(rng() % 19);  // sometimes > attempted
                    d.observe(attempted, accepted);
                }
            }
        }
        EvictDepth zero(0);
        VJ_CHECK(zero.choose(8, 0) == 0 && zero.last_choice().target_only);
    }
    // ---- cold start: no history, but the prior still permits a draft
    {
        EvictDepth d(8);
        VJ_CHECK(std::fabs(d.acceptance() - d.config().prior_accept) < 1e-12);
        const uint32_t k = d.choose(8, 0);
        VJ_CHECK(k > 0 && k <= 8);
        VJ_CHECK(!d.last_choice().target_only && !d.last_choice().probing);
        VJ_CHECK(d.choose(3, 0) <= 3);
    }
    // ---- first-token rejection: hard zero, target-only progress, probe, cooldown
    {
        EvictDepth d(8);
        d.observe(8, 0);
        d.observe(8, 0);
        VJ_CHECK(d.zero_streak() == 2);
        uint32_t emitted = 0;
        for (uint32_t i = 0; i < d.config().probe_after; ++i) {
            VJ_CHECK(d.choose(8, 64) == 0 && d.last_choice().target_only);
            ++emitted;  // the caller's ordinary target decode produces exactly one token per zero proposal
        }
        VJ_CHECK(emitted == d.config().probe_after);
        const uint32_t probe = d.choose(8, 64);
        VJ_CHECK(probe == 1 && d.last_choice().probing && !d.last_choice().target_only);
        d.observe(probe, 0);  // the target rejected the probe too
        VJ_CHECK(d.choose(8, 64) == 0);  // cooldown restarts after every verification
        std::printf("  first-token rejection: zero -> %u target-only steps -> probe %u -> zero again\n",
                    d.config().probe_after, probe);
    }
    // ---- full acceptance lifts the proposal to the configured maximum
    {
        EvictDepth d = warm_full(8);
        VJ_CHECK(d.choose(8, 0) == 8);
        VJ_CHECK(d.choose(5, 0) == 5);
        VJ_CHECK(d.choose(2, 0) == 2);
        VJ_CHECK(!d.last_choice().limited_by_experts);
    }
    // ---- partial acceptance shortens the draft without reaching zero
    {
        EvictDepth d(8);
        for (int i = 0; i < 12; ++i) d.observe(8, 1);  // per-position estimator 1/2
        const uint32_t k = d.choose(8, 0);
        VJ_CHECK(k > 0 && k < 8);
        VJ_CHECK(d.choose(2, 0) <= 2);
        std::printf("  partial acceptance: alpha %.3f -> depth %u of 8\n", d.acceptance(), k);
    }
    // ---- unique routed-expert pressure: same history, larger union -> shorter prefix, never longer
    {
        uint32_t previous = 8;
        for (uint64_t unique : {0ull, 64ull, 256ull, 512ull, 1024ull, 4096ull, 65536ull}) {
            EvictDepth d = warm_full(8);
            const uint32_t k = d.choose(8, unique);
            VJ_CHECK(k <= previous);
            previous = k;
            if (unique == 0) VJ_CHECK(k == 8);
            if (unique == 512) VJ_CHECK(k < 8 && d.last_choice().limited_by_experts);
            if (unique == 65536) VJ_CHECK(k == 0 && d.last_choice().target_only);
            std::printf("  unique=%llu pressure=%.2f -> depth %u\n",
                        static_cast<unsigned long long>(unique), d.last_choice().expert_pressure, k);
        }
    }
    // ---- observe edge cases: zero attempts are ignored, over-acceptance is clamped, reset returns to cold start
    {
        EvictDepth d(4);
        const double before = d.acceptance();
        d.observe(0, 0);
        d.observe(0, 7);
        VJ_CHECK(d.acceptance() == before && d.observations() == 0);
        d.observe(2, 9);  // clamped to a full accept
        VJ_CHECK(d.observations() == 1 && d.zero_streak() == 0);
        d.reset();
        VJ_CHECK(d.observations() == 0 && std::fabs(d.acceptance() - d.config().prior_accept) < 1e-12);
        VJ_CHECK(d.choose(4, 0) > 0);
    }
    // ---- code -> prose -> code replay: every step makes progress and probes recover the depth
    {
        EvictDepth d(8);
        int32_t next_token = 1000;
        std::vector<Row> rows;
        auto phase = [&](uint32_t accept, int steps) {
            for (int i = 0; i < steps; ++i) rows.push_back(replay_step(d, 8, accept, 64, next_token));
        };
        phase(8, 6);  // code: full acceptance
        const size_t before_prose = rows.size();
        VJ_CHECK(rows.back().proposed > 0);
        phase(0, 12);  // prose: every draft rejected at the first token
        bool saw_zero = false;
        for (size_t i = before_prose; i < rows.size(); ++i) saw_zero = saw_zero || rows[i].target_only;
        VJ_CHECK(saw_zero);
        const size_t before_recovery = rows.size();
        phase(8, 16);  // code returns: probes must find full acceptance again
        bool recovered = false;
        for (size_t i = before_recovery; i < rows.size(); ++i) recovered = recovered || (!rows[i].target_only && rows[i].proposed == 8);
        VJ_CHECK(recovered);
        check_rows(rows);
        std::printf("  code/prose/code: %zu steps, %u target-only, depth %u at end, alpha %.3f\n", rows.size(),
                    static_cast<uint32_t>(std::count_if(rows.begin(), rows.end(), [](const Row& r) { return r.target_only; })),
                    rows.back().proposed, d.acceptance());
    }
    // ---- hard zero streak is configurable and counts attempted==1 rejections too
    {
        EvictDepthConfig cfg;
        cfg.hard_zero_streak = 1;
        cfg.probe_after = 1;
        EvictDepth d(8, cfg);
        d.observe(1, 0);  // rejection at token one
        VJ_CHECK(d.choose(8, 0) == 0);
        VJ_CHECK(d.choose(8, 0) == 1 && d.last_choice().probing);
    }
    std::printf("PASS evict_depth\n");
    return 0;
}
