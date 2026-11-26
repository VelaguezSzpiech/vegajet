// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// C2 macro-batch prefill scheduler (core/prefill_sched): M = 256 / 512 macro batches with ordered GDN sub-chunks of
// C = 16 / 32. For every boundary (C - 1, C, C + 1, M - 1, M, M + 1, 2M +- 1) plus pp512 and pp2048, from position 0
// and from a cached prefix, the schedule passes verify_schedule() and an exact CPU model run in schedule order
// (dense per macro batch, chunk-parallel gated recurrence per sub-chunk, causal attention over cache + own prefix)
// matches the token-by-token model. Scratch sizing falls back 512 -> 256 -> current path by budget. Broken
// schedules (sub-chunk order swapped, a macro batch at the wrong position, a dropped final partial) are rejected and
// change the model output.
#include "core/prefill_sched.hpp"
#include "tests/test_common.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace vegajet::prefill_sched;

namespace {

std::vector<int32_t> toks_of(uint32_t n, uint32_t salt) {
    std::vector<int32_t> t(n);
    for (uint32_t i = 0; i < n; ++i) t[i] = static_cast<int32_t>((i * 2654435761u + salt * 97u) % 50000u);
    return t;
}

double max_rel(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < a[i].size(); ++j) m = std::max(m, std::fabs(a[i][j] - b[i][j]) / std::max(1.0, std::fabs(b[i][j])));
    return m;
}

// Runs a prefix of `pre` tokens token-by-token, then `n` tokens either sequentially or by schedule; compares.
double compare(uint32_t pre, uint32_t n, const Config& cfg, Schedule* out_s = nullptr) {
    const ToyModel tm{8, 11};
    const auto all = toks_of(pre + n, n);
    const std::vector<int32_t> prefix(all.begin(), all.begin() + pre), body(all.begin() + pre, all.end());
    ToyState a, b;
    (void) toy_sequential(tm, prefix, 0, &a);
    (void) toy_sequential(tm, prefix, 0, &b);
    const auto ref = toy_sequential(tm, body, pre, &a);
    const Schedule s = schedule(n, pre, cfg);
    std::string err;
    if (!verify_schedule(s, n, pre, &err)) { std::fprintf(stderr, "verify: %s\n", err.c_str()); VJ_CHECK(false); }
    const auto got = toy_scheduled(tm, body, s, &b);
    VJ_CHECK(b.absorbed == a.absorbed);
    if (out_s) *out_s = s;
    return max_rel(got, ref);
}

void test_boundaries() {
    uint32_t cases = 0;
    double worst = 0.0;
    for (uint32_t M : {256u, 512u}) {
        for (uint32_t C : {16u, 32u}) {
            Config cfg; cfg.macro_m = M; cfg.gdn_c = C;
            const uint32_t lens[] = {1u, C - 1u, C, C + 1u, M - 1u, M, M + 1u, 2u * M - 1u, 2u * M + 1u, 512u, 2048u};
            for (uint32_t L : lens) {
                for (uint32_t pre : {0u, 37u}) {
                    if (L == 2048u && pre != 0u) continue;
                    Schedule s;
                    const double e = compare(pre, L, cfg, &s);
                    worst = std::max(worst, e);
                    if (e > 1e-9) std::fprintf(stderr, "M=%u C=%u L=%u pre=%u: rel err %.3e\n", M, C, L, pre, e);
                    VJ_CHECK(e <= 1e-9);
                    VJ_CHECK(s.c2 && s.macro_m == M && s.gdn_c == C);
                    // shape checks
                    VJ_CHECK(s.macros.size() == (L + M - 1u) / M);
                    const auto& last = s.macros.back();
                    VJ_CHECK(last.n == L - (static_cast<uint32_t>(s.macros.size()) - 1u) * M && last.compute_logits);
                    VJ_CHECK(last.gdn.back().n == ((last.n % C) ? last.n % C : C));
                    ++cases;
                }
            }
        }
    }
    std::printf("  %u schedules (M 256/512 x C 16/32 x 11 lengths x {pos 0, cached prefix 37}): exact vs token-by-token, max rel err %.2e\n",
                cases, worst);
}

void test_budget_fallback() {
    Config cfg; cfg.macro_m = 512; cfg.gdn_c = 32;
    const uint64_t s512 = scratch_for(512, 32, cfg), s256 = scratch_for(256, 32, cfg);
    VJ_CHECK(s512 > s256 && s256 > 0u);
    cfg.budget_bytes = s512;
    Schedule s = schedule(2048, 0, cfg);
    VJ_CHECK(s.c2 && s.macro_m == 512 && s.reason.empty() && s.scratch_bytes == s512);
    cfg.budget_bytes = s512 - 1u;
    s = schedule(2048, 0, cfg);
    VJ_CHECK(s.c2 && s.macro_m == 256 && !s.reason.empty());
    cfg.budget_bytes = s256 - 1u;
    s = schedule(2048, 0, cfg);
    VJ_CHECK(!s.c2 && s.macro_m == cfg.legacy_chunk && s.macros.size() == 2u && s.macros[0].gdn.size() == 1u);
    std::string err;
    VJ_CHECK(verify_schedule(s, 2048, 0, &err));
    std::printf("  budget: M=512 needs %.1f MiB, M=256 %.1f MiB (FP32 legacy intermediates, no aliasing); fallback: %s\n",
                static_cast<double>(s512) / 1048576.0, static_cast<double>(s256) / 1048576.0, s.reason.c_str());
    // aliased FP16 plan is smaller
    Config f16 = cfg; f16.precision = vegajet::prefill::Precision::kFp16Intermediate; f16.aliased = true;
    VJ_CHECK(scratch_for(512, 32, f16) < s512);
    // invalid requests keep the current path
    Config bad; bad.macro_m = 384;
    VJ_CHECK(!schedule(1000, 0, bad).c2);
    bad = Config{}; bad.gdn_c = 64;
    VJ_CHECK(!schedule(1000, 0, bad).c2);
    bad = Config{}; bad.enabled = false;
    const Schedule off = schedule(3000, 0, bad);
    VJ_CHECK(!off.c2 && off.macros.size() == 3u && verify_schedule(off, 3000, 0, &err));
    // the current path is still exact in the model (GDN un-split = one chunk per macro batch)
    VJ_CHECK(compare(0, 1500, bad) <= 1e-9);
}

void test_broken_schedules() {
    const ToyModel tm{8, 11};
    Config cfg; cfg.macro_m = 256; cfg.gdn_c = 32;
    const uint32_t L = 600;
    const auto body = toks_of(L, 5);
    ToyState a;
    const auto ref = toy_sequential(tm, body, 0, &a);
    std::string err;
    {   // swapped sub-chunks
        Schedule s = schedule(L, 0, cfg);
        std::swap(s.macros[1].gdn[2], s.macros[1].gdn[3]);
        VJ_CHECK(!verify_schedule(s, L, 0, &err));
        ToyState b;
        VJ_CHECK(max_rel(toy_scheduled(tm, body, s, &b), ref) > 1e-6);
    }
    {   // macro batch at the wrong position
        Schedule s = schedule(L, 0, cfg);
        s.macros[1].pos0 += 1u;
        VJ_CHECK(!verify_schedule(s, L, 0, &err));
    }
    {   // dropped final partial
        Schedule s = schedule(L, 0, cfg);
        s.macros.pop_back();
        VJ_CHECK(!verify_schedule(s, L, 0, &err));
        VJ_CHECK(err.find("logits") != std::string::npos);  // the new last batch never computes logits
    }
    {   // partial sub-chunk in the middle of a batch
        Schedule s = schedule(L, 0, cfg);
        s.macros[0].gdn[1].n -= 1u;
        VJ_CHECK(!verify_schedule(s, L, 0, &err));
    }
    {   // kv_hi beyond the batch (non-causal boundary)
        Schedule s = schedule(L, 0, cfg);
        s.macros[0].kv_hi += 1u;
        VJ_CHECK(!verify_schedule(s, L, 0, &err));
    }
}

}  // namespace

void test_parse() {
    bool en = true; uint32_t M = 0, C = 0; std::string err;
    VJ_CHECK(parse_c2_spec("", &en, &M, &C, &err) && !en);
    VJ_CHECK(parse_c2_spec("off", &en, &M, &C, &err) && !en);
    VJ_CHECK(parse_c2_spec("512", &en, &M, &C, &err) && en && M == 512u && C == 32u);
    VJ_CHECK(parse_c2_spec("256:16", &en, &M, &C, &err) && en && M == 256u && C == 16u);
    VJ_CHECK(!parse_c2_spec("1024", &en, &M, &C, &err) && !en);
    VJ_CHECK(!parse_c2_spec("512:64", &en, &M, &C, &err));
}

int main() {
    test_parse();
    test_boundaries();
    test_budget_fallback();
    test_broken_schedules();
    std::printf("test_prefill_sched: PASS\n");
    return 0;
}
