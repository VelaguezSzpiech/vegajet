// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Measured-cost depth governor (v1.2 C7+S1): per-M verify time measured by the decode loop + live acceptance
// -> proposal depth. Pure host math, so the depth law is checked here without a GPU.
#include "core/spec_policy.hpp"
#include "tests/test_common.hpp"

#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

using namespace vegajet;
using namespace vegajet::spec_policy;

int main() {
    // 1. Cold: nothing measured -> no opinion (the loop keeps its configured depth).
    {
        MeasuredDepthGovernor gov;
        VJ_CHECK(gov.next_depth(15u) == 0u);
        VJ_CHECK(gov.time_inflation(7u) == 0.0);
        VJ_CHECK(gov.samples(1u) == 0u);
    }

    // 2. One sample per M is below min_samples -> still cold.
    {
        MeasuredDepthConfig cfg;
        cfg.min_samples = 2;
        MeasuredDepthGovernor gov(cfg);
        gov.observe_verify(1u, 5.0);
        gov.observe_verify(8u, 9.0);
        VJ_CHECK(gov.next_depth(15u) == 0u);
        gov.observe_verify(1u, 5.0);
        gov.observe_verify(8u, 9.0);
        VJ_CHECK(gov.next_depth(15u) != 0u);
    }

    // 3. Cheap M=1 with a nearly flat verify cost and high acceptance -> deep proposals.
    {
        MeasuredDepthGovernor gov;
        for (uint32_t i = 0; i < 4; ++i) {
            gov.observe_accept(8u, 7u);  // alpha 0.875
            for (uint32_t m = 1; m <= 16; ++m) gov.observe_verify(m, 4.0 + 0.15 * static_cast<double>(m));
        }
        VJ_CHECK(gov.alpha() > 0.8);
        const uint32_t k = gov.next_depth(15u);
        VJ_CHECK(k >= 5u);
        VJ_CHECK(k <= 15u);
    }

    // 4. Same cost, low acceptance -> the governor stops paying for a long prefix.
    {
        MeasuredDepthGovernor gov;
        for (uint32_t i = 0; i < 4; ++i) {
            gov.observe_accept(15u, 0u);  // no accepted conditional trials
            for (uint32_t m = 1; m <= 16; ++m) gov.observe_verify(m, 4.0 + 0.15 * static_cast<double>(m));
        }
        VJ_CHECK(gov.alpha() < 0.1);
        const uint32_t deep = 15u;
        const uint32_t k_low = gov.next_depth(deep);
        VJ_CHECK(k_low >= 1u);
        VJ_CHECK(k_low <= 2u);
    }

    // 5. Same acceptance, but verify cost explodes with M -> shallower than the cheap-cost case.
    {
        MeasuredDepthGovernor cheap, dear;
        for (uint32_t i = 0; i < 4; ++i) {
            cheap.observe_accept(8u, 6u);
            dear.observe_accept(8u, 6u);
            for (uint32_t m = 1; m <= 16; ++m) {
                cheap.observe_verify(m, 4.0 + 0.15 * static_cast<double>(m));
                dear.observe_verify(m, 4.0 + 3.0 * static_cast<double>(m) * static_cast<double>(m));
            }
        }
        const uint32_t k_cheap = cheap.next_depth(15u);
        const uint32_t k_dear  = dear.next_depth(15u);
        VJ_CHECK(k_dear < k_cheap);
        VJ_CHECK(k_dear >= 1u);
    }

    // 6. Cap and k_min are honoured.
    {
        MeasuredDepthGovernor gov;
        for (uint32_t i = 0; i < 4; ++i) {
            gov.observe_accept(16u, 16u);  // alpha 1.0
            for (uint32_t m = 1; m <= 16; ++m) gov.observe_verify(m, 3.0);
        }
        VJ_CHECK(gov.next_depth(3u) <= 3u);
        VJ_CHECK(gov.next_depth(15u) <= 15u);
        VJ_CHECK(gov.next_depth(0u) == 0u);
    }

    // 7. Measured time inflation replaces the fabricated constant: monotone, 1.0 at M=1, no downward
    //    extrapolation past the last measured M.
    {
        MeasuredDepthGovernor gov;
        for (uint32_t i = 0; i < 3; ++i) {
            for (uint32_t m = 2; m <= 8; m += 2) gov.observe_verify(m, 4.0 * static_cast<double>(m));
        }
        const double infl1 = gov.time_inflation(1u);  // M=1 unmeasured: derived from the parametric shape
        VJ_CHECK(infl1 > 0.99 && infl1 < 1.01);       // inflation is 1.0 at M=1 by definition
        VJ_CHECK(gov.time_inflation(2u) > infl1);
        VJ_CHECK(gov.time_inflation(8u) > gov.time_inflation(4u));
        VJ_CHECK(gov.time_inflation(15u) >= gov.time_inflation(8u));  // extrapolation stays monotone
        // Interpolated value sits between its neighbours.
        const double i6 = gov.time_inflation(6u);
        VJ_CHECK(i6 > gov.time_inflation(4u));
        VJ_CHECK(i6 < gov.time_inflation(8u));
    }

    // 8. Zero / absurd samples are ignored instead of poisoning the model.
    {
        MeasuredDepthGovernor gov;
        gov.observe_verify(4u, 0.0);
        gov.observe_verify(0u, 9.0);
        gov.observe_accept(0u, 0u);
        VJ_CHECK(gov.samples(4u) == 0u);
        VJ_CHECK(gov.next_depth(8u) == 0u);
        VJ_CHECK(gov.alpha() > 0.0);  // neutral prior, still a legal acceptance
    }

    // The same conditional acceptance must not fall just because K is deeper.
    for (uint32_t depth : {1u, 4u, 8u, 15u}) {
        MeasuredDepthConfig cfg;
        cfg.alpha_window = 20000u;
        MeasuredDepthGovernor gov(cfg);
        std::mt19937 rng(721);
        for (uint32_t i = 0; i < 20000u; ++i) {
            uint32_t accepted = 0;
            while (accepted < depth && rng() % 10u != 0u) ++accepted;
            gov.observe_accept(depth, accepted);
        }
        VJ_CHECK(std::abs(gov.alpha() - 0.9) < 0.01);
    }
    std::printf("test_spec_governor: ok\n");
    return 0;
}
