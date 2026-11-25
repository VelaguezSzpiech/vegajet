// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Suffix-first hybrid policy, acceptance governor, EVICT planning and drafter-KV resync bookkeeping on
// synthetic acceptance traces. A deterministic toy "target model" checks that every policy decision leaves
// the greedy output stream unchanged (lossless) while EVICT reduces verified-but-rejected work.
#include "core/spec_policy.hpp"
#include "tests/test_common.hpp"

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <random>

using namespace vegajet;
using namespace vegajet::spec_policy;

namespace {

// Deterministic toy target: next token = hash of the last 3 tokens, folded into a small vocab with
// repetitive structure so the suffix tree finds matches.
int32_t target_next(const std::vector<int32_t>& ctx) {
    const size_t n = ctx.size();
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = (n > 2 ? n - 2 : 0); i < n; ++i) h = (h ^ static_cast<uint64_t>(ctx[i] + 7)) * 1099511628211ULL;
    return static_cast<int32_t>((h >> 17) % 23u);
}

struct RunResult {
    std::vector<int32_t> out;
    uint64_t verified_drafts = 0, rejected_drafts = 0, cycles = 0, neural_cycles = 0, suffix_cycles = 0;
    uint64_t catchup_steps = 0, stash_copies = 0, holes = 0;
};

// Greedy speculative loop mirroring run_greedy_mtp: verification = target argmax at each position.
// `neural_err` gives the per-step probability that the neural draft is wrong (synthetic acceptance trace).
RunResult run_loop(const std::vector<int32_t>& prompt, int n_gen, bool use_policy, bool use_evict,
                   const std::function<double(int)>& neural_err, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    RunResult r;
    spec::SuffixTree sfx;
    sfx.extend(prompt);
    std::vector<int32_t> ctx = prompt;
    int32_t last = target_next(ctx);
    std::vector<int32_t> gen{last};
    sfx.push(last);
    PolicyConfig pc;
    SpecPolicy pol(pc);
    DrafterKvSync sync;
    uint32_t pos = static_cast<uint32_t>(prompt.size());
    while (static_cast<int>(gen.size()) < n_gen) {
        ++r.cycles;
        const uint32_t rem = static_cast<uint32_t>(n_gen - static_cast<int>(gen.size()));
        std::vector<int32_t> qctx = ctx;
        qctx.push_back(gen.back());
        CycleInputs in;
        in.remaining = rem;
        in.suffix = best_chain(sfx.draft(qctx, 4, 1, 0.05));
        in.suffix_match_len = sfx.match_length(qctx);
        CycleDecision d;
        if (use_policy) {
            d = pol.decide(in);
        } else {
            d.source = DraftSource::kNeural;
            d.neural_k = std::min<uint32_t>(4u, rem > 1 ? rem - 1 : 0);
            if (d.neural_k == 0) d.source = DraftSource::kPlain;
        }
        std::vector<int32_t> drafts;
        if (d.source == DraftSource::kNeural) {
            auto pre = sync.before_neural();
            r.catchup_steps += pre.catchup.size();
            std::vector<int32_t> tmp = qctx;
            for (uint32_t j = 0; j < d.neural_k; ++j) {
                int32_t t = target_next(tmp);
                if (U(rng) < neural_err(static_cast<int>(gen.size()))) t = (t + 1 + static_cast<int32_t>(rng() % 5u)) % 23;
                drafts.push_back(t);
                tmp.push_back(t);
            }
            if (use_evict) drafts.resize(pol.evict_neural(static_cast<uint32_t>(drafts.size())));
            ++r.neural_cycles;
        } else if (d.source == DraftSource::kSuffix) {
            drafts = d.suffix_tokens;
            ++r.suffix_cycles;
        }
        // Verify: target argmax at each of the dk+1 positions.
        std::vector<int32_t> preds;
        std::vector<int32_t> tmp = qctx;
        for (size_t j = 0; j <= drafts.size(); ++j) {
            preds.push_back(target_next(tmp));
            if (j < drafts.size()) tmp.push_back(drafts[j]);
        }
        uint32_t acc = 0;
        while (acc < drafts.size() && preds[acc] == drafts[acc]) ++acc;
        r.verified_drafts += drafts.size();
        r.rejected_drafts += drafts.size() - acc;
        if (use_policy && d.source != DraftSource::kPlain) pol.observe(d.source, static_cast<uint32_t>(drafts.size()), acc);
        std::vector<int32_t> committed{gen.back()};
        ctx.push_back(gen.back());
        for (uint32_t j = 0; j < acc; ++j) {
            gen.push_back(drafts[j]);
            ctx.push_back(drafts[j]);
            sfx.push(drafts[j]);
            committed.push_back(drafts[j]);
        }
        if (static_cast<int>(gen.size()) < n_gen) { gen.push_back(preds[acc]); sfx.push(preds[acc]); }
        auto plan = sync.after_verify(committed, pos, d.source == DraftSource::kNeural, false);
        r.catchup_steps += plan.catchup.size();
        r.stash_copies += plan.stash.size();
        r.holes += plan.holes_added;
        pos += acc + 1u;
    }
    r.out = gen;
    return r;
}

std::vector<int32_t> run_plain(const std::vector<int32_t>& prompt, int n_gen) {
    std::vector<int32_t> ctx = prompt, gen;
    gen.push_back(target_next(ctx));
    while (static_cast<int>(gen.size()) < n_gen) {
        ctx.push_back(gen.back());
        gen.push_back(target_next(ctx));
    }
    return gen;
}

} // namespace

int main() {
    int n = 0;

    // --- best_chain: linear path out of a branching tree ---
    {
        spec::SuffixTree t(8);
        t.extend({1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 3, 4, 9});
        auto tree = t.draft({1, 2, 3}, 6, 2, 0.0);
        bool sibling = false;
        for (size_t i = 1; i < tree.size(); ++i) sibling |= tree[i].parent < static_cast<int32_t>(i) - 1;
        VJ_CHECK(sibling);  // flattened tree is NOT a chain (T3)
        Chain c = best_chain(tree);
        VJ_CHECK(c.size() >= 2 && c.tokens[0] == 4 && c.tokens[1] == 1);  // tie 1 vs 9 -> lower id
        for (size_t j = 1; j < c.size(); ++j) VJ_CHECK(c.cum_score[j] <= c.cum_score[j - 1] + 1e-12);
        auto p = conditional_probs(c);
        VJ_CHECK(std::fabs(p[0] - 2.0 / 3.0) < 1e-9 && p[1] <= 1.0);
        VJ_CHECK(best_chain({}).size() == 0);
        n += 4;
    }

    // --- governor: rise, hysteresis band, fall, disable, probe, recover ---
    {
        GovernorConfig gc;
        gc.k_init = 2;
        AcceptanceGovernor g(gc);
        for (int i = 0; i < 40; ++i) { g.tick(); g.observe(g.k(), g.k()); }  // all accepted
        VJ_CHECK(g.k() == 4);
        const uint64_t ch = g.changes();
        // Band: alpha ~0.57 (4 drafts, accept 4 then reject: a/(a+1) with a = 1.33) -> no change.
        for (int i = 0; i < 60; ++i) { g.tick(); g.observe(4, (i % 3 == 0) ? 2u : 1u); }
        const double a_band = g.alpha();
        VJ_CHECK(a_band > gc.lower_below && a_band < gc.raise_above);
        VJ_CHECK(g.changes() == ch && g.k() == 4);
        for (int i = 0; i < 80; ++i) { g.tick(); g.observe(std::max(1u, g.k()), 0); }  // all rejected
        VJ_CHECK(g.k_level() == 0);
        uint32_t probes = 0;
        for (int i = 0; i < 100; ++i) {
            g.tick();
            if (g.k() > 0) { ++probes; g.observe(g.k(), 0); }
        }
        VJ_CHECK(probes >= 2 && probes <= 4 && g.k_level() == 0);
        // Recovery: probes succeed.
        for (int i = 0; i < 400 && g.k_level() == 0; ++i) {
            g.tick();
            if (g.k() > 0) g.observe(g.k(), g.k());
        }
        VJ_CHECK(g.k_level() >= 1);
        // Noisy trace straddling one threshold: dwell bounds the number of changes.
        AcceptanceGovernor g2(gc);
        std::mt19937 rng(7);
        for (int i = 0; i < 400; ++i) { g2.tick(); const uint32_t k = std::max(1u, g2.k()); g2.observe(k, (rng() % 2) ? k : 0u); }
        VJ_CHECK(g2.changes() <= 400 / gc.dwell);
        std::printf("[spec_policy] governor: band alpha %.2f, probes %u, noisy-trace changes %llu\n", a_band, probes,
                    static_cast<unsigned long long>(g2.changes()));
        n += 6;
    }

    // --- EVICT planning ---
    {
        VerifyCostModel cm;
        VJ_CHECK(cm.cost_ms(5) > cm.cost_ms(2) && cm.cost_ms(2) > cm.cost_ms(1));
        auto hi = plan_evict({0.95, 0.95, 0.95, 0.95}, cm);
        auto lo = plan_evict({0.05, 0.05, 0.05, 0.05}, cm);
        auto mid = plan_evict({0.9, 0.6, 0.2, 0.1}, cm);
        VJ_CHECK(hi.keep == 4 && lo.keep == 0);
        VJ_CHECK(mid.keep >= 1 && mid.keep <= 3);
        uint32_t prev = 0;
        for (double a = 0.0; a <= 1.0; a += 0.05) {  // monotone in acceptance
            const uint32_t k = plan_evict({a, a, a, a}, cm).keep;
            VJ_CHECK(k >= prev);
            prev = k;
        }
        VJ_CHECK(plan_evict({0.05, 0.05}, cm, 2).keep == 2);  // min_keep honoured
        // Measured table overrides the model; observed expert unions feed the model.
        VerifyCostModel m2;
        m2.measured_ms = {0.0, 10.4, 30.0, 40.0, 50.0, 60.0};
        VJ_CHECK(plan_evict({0.8, 0.8, 0.8, 0.8}, m2).keep == 0);
        ExpertUnionTracker ut;
        ut.observe(5, 8.0);  // perfectly shared experts: union does not grow
        ut.observe(1, 8.0);
        VerifyCostModel m3;
        ut.apply(&m3);
        VJ_CHECK(std::fabs(m3.cost_ms(5) - (m3.dense_ms + m3.expert_ms + 5 * m3.per_tok_ms)) < 1e-9);
        VJ_CHECK(plan_evict({0.3, 0.3, 0.3, 0.3}, m3).keep >= plan_evict({0.3, 0.3, 0.3, 0.3}, cm).keep);
        std::printf("[spec_policy] evict keep: a=.95 -> %u, a=.05 -> %u, decaying -> %u; cost(1..5) = %.2f %.2f %.2f %.2f %.2f ms\n",
                    hi.keep, lo.keep, mid.keep, cm.cost_ms(1), cm.cost_ms(2), cm.cost_ms(3), cm.cost_ms(4), cm.cost_ms(5));
        n += 7;
    }

    // --- decide(): suffix-first gate ---
    {
        SpecPolicy pol;
        CycleInputs in;
        in.remaining = 100;
        in.suffix_match_len = 6;
        in.suffix.tokens = {5, 6, 7, 8};
        in.suffix.cum_score = {1.0, 0.9, 0.8, 0.7};
        auto d = pol.decide(in);
        VJ_CHECK(d.source == DraftSource::kSuffix && d.suffix_tokens.size() == 4 && d.neural_k == 0);
        in.suffix.cum_score = {1.0, 0.9, 0.8, 0.1};  // only 3 confident -> neural drafter
        d = pol.decide(in);
        VJ_CHECK(d.source == DraftSource::kNeural && d.neural_k == 4 && d.confident == 3);
        in.suffix_match_len = 1;  // too short a match: not confident at all
        in.suffix.cum_score = {1.0, 1.0, 1.0, 1.0};
        VJ_CHECK(pol.decide(in).source == DraftSource::kNeural);
        in.suffix_match_len = 6;
        in.remaining = 3;  // budget 2: two confident tokens suffice
        in.suffix.cum_score = {1.0, 1.0, 0.0, 0.0};
        d = pol.decide(in);
        VJ_CHECK(d.source == DraftSource::kSuffix && d.suffix_tokens.size() == 2);
        in.remaining = 1;
        VJ_CHECK(pol.decide(in).source == DraftSource::kPlain);
        // Neural source off -> short suffix chains through EVICT, empty chain -> plain.
        in.remaining = 100;
        in.neural_available = false;
        in.suffix.tokens = {5, 6};
        in.suffix.cum_score = {0.9, 0.8};
        d = pol.decide(in);
        VJ_CHECK(d.source == DraftSource::kSuffix && d.suffix_tokens.size() >= 1);
        in.suffix = {};
        VJ_CHECK(pol.decide(in).source == DraftSource::kPlain);
        n += 7;
    }

    // --- drafter KV sync bookkeeping ---
    {
        SyncConfig sc;
        sc.catchup_window = 6;
        sc.stash_slots = 12;
        SyncConfig sc_lenient = sc;
        sc_lenient.max_hole_frac = 0.95;  // keep deferring through this short run
        DrafterKvSync s(sc_lenient);
        std::vector<int> kv(1000, -1);  // drafter KV: token id written at each position, -1 = hole
        auto apply = [&](const SyncPlan& p) { for (const auto& c : p.catchup) kv[c.pos] = c.token; };
        // Neural cycle: draft step 0 already ingested vtoks[0]; catch-up j = 1..acc from trunk rows.
        auto p = s.after_verify({10, 11, 12}, 100, true, false);
        VJ_CHECK(p.stash.empty() && p.catchup.size() == 2 && p.catchup[0].pos == 101 &&
                 p.catchup[0].h.kind == HSource::kTrunkRow && p.catchup[0].h.index == 0 && p.catchup[1].h.index == 1);
        kv[100] = 10;
        apply(p);
        // Suffix run: deferred -> only stash copies, nothing replayed.
        uint32_t pos = 103, holes = 0;
        for (int c = 0; c < 3; ++c) {
            auto q = s.after_verify({20 + c * 5, 21 + c * 5, 22 + c * 5, 23 + c * 5, 24 + c * 5}, pos, false, false);
            VJ_CHECK(q.catchup.empty() && q.stash.size() == 5);
            VJ_CHECK(q.stash[0].src.kind == HSource::kPend && q.stash[1].src.kind == HSource::kTrunkRow);
            holes += q.holes_added;
            pos += 5;
        }
        VJ_CHECK(s.pending() == 6 && holes == 9 && s.total_holes() == 9);
        auto rp = s.before_neural();
        VJ_CHECK(rp.catchup.size() == 6 && s.pending() == 0);
        for (size_t i = 1; i < rp.catchup.size(); ++i) VJ_CHECK(rp.catchup[i].pos == rp.catchup[i - 1].pos + 1);
        VJ_CHECK(rp.catchup.back().pos == pos - 1 && rp.catchup[0].h.kind == HSource::kStash);
        apply(rp);
        // Every committed position is either in the drafter KV or counted as a hole.
        uint32_t filled = 0;
        for (uint32_t q = 100; q < pos; ++q) filled += kv[q] >= 0;
        VJ_CHECK(filled + holes == pos - 100);
        // Many holes -> eager fallback replays immediately.
        DrafterKvSync s2(sc);
        for (int c = 0; c < 2; ++c) (void) s2.after_verify({1, 2, 3, 4, 5}, 200 + c * 5, false, false);
        VJ_CHECK(s2.hole_fraction() > sc.max_hole_frac && s2.effective_mode() == SyncMode::kEager);
        auto e = s2.after_verify({7, 8}, 210, false, false);
        VJ_CHECK(e.stash.empty() && e.catchup.size() == 8 && e.catchup.back().pos == 211);  // 6 pending + 2 new
        // Hysteresis: stays eager in the band, returns to deferred once holes are rare.
        uint32_t p2 = 212, eager_cycles = 0;
        while (s2.effective_mode() == SyncMode::kEager && eager_cycles < 200) {
            (void) s2.after_verify({1, 2}, p2, false, false);
            p2 += 2;
            ++eager_cycles;
        }
        VJ_CHECK(s2.effective_mode() == SyncMode::kDeferred && s2.hole_fraction() < sc.resume_hole_frac);
        VJ_CHECK(eager_cycles > 3);
        // Eager config == today's behaviour: suffix cycle replays vtoks[0] (h_pend) + accepted tokens.
        SyncConfig ec;
        ec.mode = SyncMode::kEager;
        DrafterKvSync s3(ec);
        auto f = s3.after_verify({1, 2, 3}, 50, false, false);
        VJ_CHECK(f.catchup.size() == 3 && f.catchup[0].h.kind == HSource::kPend && f.catchup[0].pos == 50);
        n += 9;
    }

    // --- end-to-end synthetic traces: lossless + EVICT saves rejected verify work ---
    {
        std::vector<int32_t> prompt;
        for (int i = 0; i < 64; ++i) prompt.push_back((i * 7) % 23);
        const int n_gen = 400;
        const auto ref = run_plain(prompt, n_gen);
        struct Trace { const char* name; std::function<double(int)> err; };
        const Trace traces[] = {
            {"high-accept", [](int) { return 0.05; }},
            {"low-accept", [](int) { return 0.80; }},
            {"phase-shift", [](int t) { return t < 200 ? 0.1 : 0.9; }},
        };
        for (const auto& tr : traces) {
            for (uint32_t seed : {1u, 2u, 3u}) {
                auto base = run_loop(prompt, n_gen, false, false, tr.err, seed);
                auto pol  = run_loop(prompt, n_gen, true, false, tr.err, seed);
                auto ev   = run_loop(prompt, n_gen, true, true, tr.err, seed);
                VJ_CHECK(base.out == ref && pol.out == ref && ev.out == ref);  // lossless in every mode
                if (seed == 1u)
                    std::printf("[spec_policy] %-11s cycles base/policy/evict %llu/%llu/%llu | rejected drafts %llu/%llu/%llu | "
                                "suffix cycles %llu, catch-up steps %llu, stash %llu, holes %llu\n",
                                tr.name, (unsigned long long) base.cycles, (unsigned long long) pol.cycles,
                                (unsigned long long) ev.cycles, (unsigned long long) base.rejected_drafts,
                                (unsigned long long) pol.rejected_drafts, (unsigned long long) ev.rejected_drafts,
                                (unsigned long long) ev.suffix_cycles, (unsigned long long) ev.catchup_steps,
                                (unsigned long long) ev.stash_copies, (unsigned long long) ev.holes);
                if (std::string(tr.name) == "low-accept") VJ_CHECK(ev.rejected_drafts < base.rejected_drafts);
                ++n;
            }
        }
    }

    std::printf("test_spec_policy: all %d checks passed\n", n);
    return 0;
}
