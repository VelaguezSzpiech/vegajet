// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/fp_oracles: FP16 conversion exactness, error accounting, the GDN single-write verify oracle (M = 5..8,
// every accepted prefix) and the EVICT expert-union dedup oracle on synthetic weights. Tolerances are the
// measured worst case rounded up; the printed lines carry the actual numbers per precision.
#include "core/fp_oracles.hpp"
#include "tests/test_common.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>

using namespace vegajet::oracles;

int main() {
    // ---- FP16 conversion: exact normals, subnormals, signed zero, inf, nan, ties-to-even.
    VJ_CHECK(f32_to_f16(1.0f) == 0x3C00);
    VJ_CHECK(f32_to_f16(-2.0f) == 0xC000);
    VJ_CHECK(f32_to_f16(65504.0f) == 0x7BFF);
    VJ_CHECK(f32_to_f16(65520.0f) == 0x7C00);  // overflow to inf
    VJ_CHECK(f32_to_f16(std::ldexp(1.0f, -24)) == 0x0001);
    VJ_CHECK(f32_to_f16(std::ldexp(1.0f, -25)) == 0x0000);          // half of the smallest subnormal: even -> 0
    VJ_CHECK(f32_to_f16(std::ldexp(3.0f, -25)) == 0x0002);          // 1.5 * 2^-24 -> 2
    VJ_CHECK(f32_to_f16(0.0f) == 0x0000 && f32_to_f16(-0.0f) == 0x8000);
    VJ_CHECK(f32_to_f16(INFINITY) == 0x7C00 && (f32_to_f16(NAN) & 0x7C00) == 0x7C00);
    VJ_CHECK(f16_to_f32(0x0001) == std::ldexp(1.0f, -24));
    VJ_CHECK(f16_to_f32(0x7BFF) == 65504.0f);
    // Every finite half round-trips through the two conversions bit for bit.
    for (uint32_t h = 0; h < 0x7C00; ++h) VJ_CHECK(f32_to_f16(f16_to_f32(static_cast<uint16_t>(h))) == h);
    VJ_CHECK(std::signbit(f16_to_f32(0x8000)));
    std::printf("  f16 conversion: round trip over all 31744 finite patterns, subnormals, inf/nan OK\n");

    // ---- compare / ErrStats.
    const double ref[4] = {1.0, -2.0, 1e-3, 0.0}, got[4] = {1.0, -2.5, 1e-3 + 1e-6, 1e-9};
    const ErrStats e = compare(ref, got, 4);
    VJ_CHECK(e.n == 4 && std::fabs(e.max_abs - 0.5) < 1e-15 && e.max_rel >= 0.25);
    VJ_CHECK(e.at_abs == 1);
    ErrStats m;
    m.merge(e);
    m.merge(e);
    VJ_CHECK(m.n == 8 && std::fabs(m.max_abs - 0.5) < 1e-15);

    // ---- GDN: one case per M, both shapes, every prefix.
    for (uint32_t M = 5; M <= 8; ++M) {
        const GdnOracleReport r = gdn_verify_oracle(32, 48, M, 100 + M);
        VJ_CHECK(r.single_write);
        VJ_CHECK(r.algo_f64.n > 0);
        VJ_CHECK(r.algo_f64.max_abs < 1e-12);
        for (const auto& p : r.by_precision) {
            VJ_CHECK(p.out_err.n == M * 48);
            VJ_CHECK(p.state_err.n == (M + 1) * 48 * 32);
        }
        std::printf("  %s\n", r.summary.c_str());
        for (const auto& p : r.by_precision) {
            if (p.prec == Precision::F32) VJ_CHECK(p.out_err.rel_l2() < 1e-5 && p.state_err.rel_l2() < 1e-5);
            if (p.prec == Precision::F16) VJ_CHECK(p.out_err.rel_l2() < 5e-3 && p.state_err.rel_l2() < 5e-3);
        }
    }

    // The state after accepting k tokens is the sequential state after k tokens, not a re-simulated value:
    // accepting 0 must leave S0 untouched.
    {
        const GdnProblem p = gdn_make(16, 16, 6, 7);
        const GdnVerify v = gdn_verify(p, Precision::F64);
        VJ_CHECK(v.state.size() == 7);
        for (size_t i = 0; i < p.S0.size(); ++i) VJ_CHECK(v.state[0][i] == p.S0[i]);
        const auto seq = gdn_sequential(p, nullptr);
        for (uint32_t k = 0; k <= p.M; ++k) {
            const ErrStats s = compare(seq[k].data(), v.state[k].data(), seq[k].size());
            VJ_CHECK(s.rel_l2() < 1e-12);
        }
    }

    // ---- EVICT: dedup equals naive per-token evaluation bit for bit and loads fewer experts.
    {
        const EvictProblem p = evict_make(4, 16, 32, 6, 3, 99);
        const EvictOracleReport r = evict_oracle(p);
        VJ_CHECK(r.err.n == p.M * p.dim);
        VJ_CHECK(r.bitwise_equal);
        VJ_CHECK(r.unique < r.occurrences);
        VJ_CHECK(r.dedupe_loads == r.unique);
        VJ_CHECK(r.naive_loads == r.occurrences);
        std::printf("  %s\n", r.summary.c_str());
    }
    // Same routing for every token: the union collapses to one load set and still matches.
    {
        EvictProblem p = evict_make(2, 8, 16, 5, 2, 5);
        for (uint32_t t = 1; t < p.M; ++t) {
            p.route_layer[t] = p.route_layer[0];
            p.route_expert[t] = p.route_expert[0];
        }
        const EvictOracleReport r = evict_oracle(p);
        VJ_CHECK(r.bitwise_equal);
        VJ_CHECK(r.unique == p.route_expert[0].size());
        VJ_CHECK(r.dedupe_loads == r.unique && r.naive_loads == 5 * r.unique);
    }

    const GdnSweepReport sw = gdn_verify_sweep();
    std::printf("  %s\n", sw.summary.c_str());
    VJ_CHECK(sw.single_write);
    VJ_CHECK(sw.cases == 8);
    VJ_CHECK(sw.algo_f64.max_abs < 1e-10);
    VJ_CHECK(sw.worst[0].out_err.rel_l2() < 1e-5 && sw.worst[0].state_err.rel_l2() < 1e-5);
    VJ_CHECK(sw.worst[1].out_err.rel_l2() < 5e-3 && sw.worst[1].state_err.rel_l2() < 5e-3);
    std::printf("test_fp_oracles: GDN verify + EVICT dedup oracles OK\nPASS\n");
    return 0;
}
