// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Residency-aware launch planning (core/launch_planner) on 60-CU (MI50) and 64-CU (MI60) gfx906:
//  * exhaustive split-K S in {1,2,4,8} x rows-per-WG x resident cap: the planner's choice equals an independent
//    brute-force optimum, every K chunk is a multiple of 32 and covers K exactly;
//  * skinny projections (N x K = 3072x4096, 512x3072): >= 85% wave fill, <= 25% tail-wave waste;
//  * sweep: every projection >= 2M weights reaches >= 85% fill (smaller ones are latency-bound, reported only);
//  * split-KV for 16:1 GQA (16 local q heads < n_cu): >= 85% fill / <= 25% tail once ctx >= 4096;
//  * the planner never does worse than HEAD plan_gemv_splitk measured on resident slots.
#include "core/grid_planner.hpp"
#include "core/launch_planner.hpp"
#include "tests/test_common.hpp"

#include <cmath>
#include <cstdio>

using namespace vegajet::planner;

namespace {

GemvLaunchPlan brute_force(uint32_t n, uint32_t k, uint32_t n_cu, const GemvLaunchLimits& lim) {
    const auto occ = compute_gfx906_occupancy(lim.vgpr, lim.sgpr, lim.lds_bytes, lim.wg_threads, n_cu);
    const uint32_t max_res = std::max<uint32_t>(1, occ.resident_wgs_per_cu);
    GemvLaunchPlan best;
    for (uint32_t rpw : lim.rows_per_wg)
        for (uint32_t s : lim.split_k) {
            const uint32_t chunk = s == 1 ? k : ((k + s - 1) / s + 31) / 32 * 32;
            if (s > 1 && (chunk < lim.min_k_chunk || (s - 1) * chunk >= k)) continue;
            if (rpw * chunk < lim.min_elems_per_wg && !(s == 1 && rpw == lim.rows_per_wg.back())) continue;
            const uint32_t wgs = (n + rpw - 1) / rpw * s;
            for (uint32_t res = std::min(lim.min_resident_per_cu, max_res); res <= max_res; ++res) {
                const auto f = wave_fill(wgs, n_cu * res);
                const double sc = gemv_plan_score(f, s, res, max_res, lim);
                if (!best.valid || sc > best.score + 1e-9) {
                    best.valid = true; best.score = sc; best.split_k = s; best.rows_per_wg = rpw;
                    best.resident_per_cu = res; best.fill = f; best.total_wgs = wgs; best.k_chunk = chunk;
                }
            }
        }
    return best;
}

void check_plan_shape(const GemvLaunchPlan& p) {
    VJ_CHECK(p.valid);
    VJ_CHECK(p.split_k == 1 || p.split_k == 2 || p.split_k == 4 || p.split_k == 8);
    VJ_CHECK(p.total_wgs == (p.n + p.rows_per_wg - 1) / p.rows_per_wg * p.split_k);
    if (p.split_k > 1) {
        VJ_CHECK(p.k_chunk % 32 == 0);
        VJ_CHECK(static_cast<uint64_t>(p.split_k - 1) * p.k_chunk < p.k);
        VJ_CHECK(static_cast<uint64_t>(p.split_k) * p.k_chunk >= p.k);
    } else {
        VJ_CHECK(p.k_chunk == p.k);
    }
    VJ_CHECK(p.resident_per_cu >= 1 && p.resident_per_cu <= p.max_resident_per_cu);
    VJ_CHECK(p.fill.slots == p.n_cu * p.resident_per_cu);
    VJ_CHECK(p.fill.fill > 0.0 && p.fill.fill <= 1.0);
}

} // namespace

int main() {
    // wave_fill arithmetic.
    {
        const auto f = wave_fill(384, 360);  // HEAD attn_o / ssm_out grid on 60 CUs x 6 resident
        VJ_CHECK(f.waves == 2 && std::fabs(f.fill - 384.0 / 720.0) < 1e-12);
        VJ_CHECK(std::fabs(f.tail_waste - (360.0 - 24.0) / 360.0) < 1e-12);
        const auto g = wave_fill(240, 240);
        VJ_CHECK(g.waves == 1 && g.fill == 1.0 && g.tail_waste == 0.0);
        VJ_CHECK(wave_fill(0, 240).waves == 0 && wave_fill(10, 0).waves == 0);
    }

    // The 38-VGPR / 256-thread GEMV keeps 6 WGs per CU resident (the default limits model that kernel).
    VJ_CHECK(compute_gfx906_occupancy(38, 24, 0, 256, 60).resident_wgs_per_cu == 6);

    const GemvLaunchLimits lim;
    uint32_t sweep = 0, sweep_big = 0, small_below = 0;
    for (uint32_t n_cu : {60u, 64u}) {
        // Skinny projections: hard targets.
        for (auto nk : {std::pair<uint32_t, uint32_t>{3072, 4096}, {512, 3072}}) {
            const auto p = plan_gemv_launch(nk.first, nk.second, n_cu, lim);
            check_plan_shape(p);
            VJ_CHECK(p.fill.fill >= 0.85);
            VJ_CHECK(p.fill.tail_waste <= 0.25);
            const auto h = plan_gemv_splitk(nk.first, nk.second, n_cu, 8);
            const auto hf = wave_fill(h.total_wgs, n_cu * p.max_resident_per_cu);
            VJ_CHECK(p.fill.fill + 1e-12 >= hf.fill);
            std::printf("  %2u CU  %4ux%-4u  rows/WG %2u  S=%u  chunk %4u  resident %u/%u  WGs %4u  fill %.3f  tail %.3f"
                        "   (HEAD splitk: S=%u WGs %u -> resident fill %.3f)\n",
                        n_cu, nk.first, nk.second, p.rows_per_wg, p.split_k, p.k_chunk, p.resident_per_cu,
                        p.max_resident_per_cu, p.total_wgs, p.fill.fill, p.fill.tail_waste, h.split_k, h.total_wgs,
                        hf.fill);
        }
        // Exhaustive sweep: optimality vs brute force, shape invariants, fill floor on bandwidth-relevant sizes.
        for (uint32_t n = 256; n <= 32768; n += 256) {
            for (uint32_t k : {512u, 1024u, 1536u, 2048u, 2560u, 3072u, 4096u, 6144u, 8192u}) {
                const auto p = plan_gemv_launch(n, k, n_cu, lim);
                check_plan_shape(p);
                const auto b = brute_force(n, k, n_cu, lim);
                VJ_CHECK(std::fabs(p.score - b.score) < 1e-9);
                VJ_CHECK(p.fill.fill + 1e-9 >= b.fill.fill - (lim.split_penalty * 7 + lim.residency_penalty * 6 +
                                                             lim.tail_penalty));
                const auto h = plan_gemv_splitk(n, k, n_cu, 8);
                VJ_CHECK(p.fill.fill + 0.02 >= wave_fill(h.total_wgs, n_cu * p.max_resident_per_cu).fill);
                ++sweep;
                if (static_cast<uint64_t>(n) * k >= (2ull << 20)) {
                    VJ_CHECK(p.fill.fill >= 0.85);
                    if (p.fill.waves <= 4) VJ_CHECK(p.fill.tail_waste <= 0.25);
                    ++sweep_big;
                } else if (p.fill.fill < 0.85) {
                    ++small_below;
                }
            }
        }
        // Split-KV for 16:1 GQA (16 local q heads, 1 local KV head) and a few other head counts below n_cu.
        for (uint32_t q : {4u, 8u, 16u, 32u}) {
            for (uint32_t ctx : {256u, 1024u, 4096u, 8192u, 32768u, 131072u}) {
                const auto s = plan_split_kv(q, ctx, n_cu);
                VJ_CHECK(s.chunks >= 1 && s.total_wgs == q * s.chunks);
                VJ_CHECK(s.chunks == 1 || (s.chunk_tokens % 64 == 0 && s.chunk_tokens >= 256));
                VJ_CHECK(static_cast<uint64_t>(s.chunks) * s.chunk_tokens >= ctx);
                VJ_CHECK(static_cast<uint64_t>(s.chunks - 1) * s.chunk_tokens < ctx);
                if (ctx >= 4096) {
                    VJ_CHECK(s.fill.fill >= 0.85);
                    VJ_CHECK(s.fill.tail_waste <= 0.25);
                }
                if (q == 16 && (ctx == 4096 || ctx == 131072))
                    std::printf("  %2u CU  GQA 16:1 q_local=16 ctx %6u -> %2u chunks x %5u tok, %3u WGs, fill %.3f tail %.3f\n",
                                n_cu, ctx, s.chunks, s.chunk_tokens, s.total_wgs, s.fill.fill, s.fill.tail_waste);
            }
        }
        // q heads >= slots: no split.
        VJ_CHECK(plan_split_kv(n_cu, 65536, n_cu).chunks == 1);
    }

    std::printf("test_grid_planner_splitk: %u shapes optimal vs brute force; %u bandwidth-relevant shapes >= 85%% fill; "
                "%u sub-2M-weight shapes below 85%% (latency-bound, not asserted)\n",
                sweep, sweep_big, small_below);
    std::printf("PASS\n");
    return 0;
}
