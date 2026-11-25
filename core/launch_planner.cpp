// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/launch_planner.hpp"

#include "core/grid_planner.hpp"

#include <algorithm>

namespace vegajet::planner {

namespace {
uint32_t ceil_div(uint32_t a, uint32_t b) { return b ? (a + b - 1u) / b : 0u; }
uint32_t align_up(uint32_t v, uint32_t a) { return ceil_div(v, a) * a; }
} // namespace

WaveFill wave_fill(uint32_t wgs, uint32_t slots) {
    WaveFill f;
    f.wgs = wgs;
    f.slots = slots;
    if (!wgs || !slots) return f;
    f.waves = ceil_div(wgs, slots);
    const uint64_t cap = static_cast<uint64_t>(f.waves) * slots;
    f.fill = static_cast<double>(wgs) / static_cast<double>(cap);
    const uint32_t last = wgs - (f.waves - 1u) * slots;
    f.tail_waste = static_cast<double>(slots - last) / static_cast<double>(slots);
    return f;
}

double gemv_plan_score(const WaveFill& f, uint32_t split_k, uint32_t resident, uint32_t max_resident,
                       const GemvLaunchLimits& lim) {
    return f.fill - lim.split_penalty * (split_k - 1u) - lim.tail_penalty * f.tail_waste -
           lim.residency_penalty * static_cast<double>(max_resident - std::min(resident, max_resident));
}

GemvLaunchPlan plan_gemv_launch(uint32_t n, uint32_t k, uint32_t n_cu, const GemvLaunchLimits& lim) {
    GemvLaunchPlan best;
    best.n = n;
    best.k = k;
    best.n_cu = n_cu;
    if (!n || !k || !n_cu) return best;
    const OccupancyInfo occ = compute_gfx906_occupancy(lim.vgpr, lim.sgpr, lim.lds_bytes, lim.wg_threads, n_cu);
    const uint32_t max_res = std::max<uint32_t>(1, occ.resident_wgs_per_cu);
    const uint32_t min_res = std::min(std::max<uint32_t>(1, lim.min_resident_per_cu), max_res);
    best.max_resident_per_cu = max_res;
    for (uint32_t rpw : lim.rows_per_wg) {
        if (!rpw) continue;
        const uint32_t row_wgs = ceil_div(n, rpw);
        for (uint32_t s : lim.split_k) {
            if (!s) continue;
            const uint32_t chunk = s == 1 ? k : align_up(ceil_div(k, s), 32);
            if (s > 1 && (chunk < lim.min_k_chunk || static_cast<uint64_t>(s - 1u) * chunk >= k)) continue;
            if (static_cast<uint64_t>(rpw) * chunk < lim.min_elems_per_wg && !(s == 1 && rpw == lim.rows_per_wg.back()))
                continue;
            const uint32_t wgs = row_wgs * s;
            for (uint32_t res = max_res; res >= min_res; --res) {
                const WaveFill f = wave_fill(wgs, n_cu * res);
                const double sc = gemv_plan_score(f, s, res, max_res, lim);
                if (!best.valid || sc > best.score + 1e-9) {
                    best.valid = true;
                    best.rows_per_wg = rpw;
                    best.split_k = s;
                    best.k_chunk = chunk;
                    best.row_wgs = row_wgs;
                    best.total_wgs = wgs;
                    best.resident_per_cu = res;
                    best.fill = f;
                    best.score = sc;
                }
                if (res == 1) break;
            }
        }
    }
    return best;
}

SplitKvPlan plan_split_kv(uint32_t n_q_local, uint32_t ctx, uint32_t n_cu, uint32_t resident, uint32_t min_chunk,
                          uint32_t max_chunks, double merge_penalty) {
    SplitKvPlan best;
    best.n_q_local = n_q_local;
    best.ctx_tokens = ctx;
    best.n_cu = n_cu;
    if (!n_q_local || !ctx || !n_cu) return best;
    const uint32_t slots = n_cu * std::max<uint32_t>(1, resident);
    bool have = false;
    uint32_t last_chunks = 0;
    for (uint32_t c = 1; c <= std::max<uint32_t>(1, max_chunks); ++c) {
        const uint32_t ct = c == 1 ? ctx : align_up(ceil_div(ctx, c), 64);
        if (c > 1 && ct < min_chunk) break;
        const uint32_t actual = ceil_div(ctx, ct);  // alignment may merge the tail into fewer chunks
        if (actual == last_chunks) continue;
        last_chunks = actual;
        const WaveFill f = wave_fill(n_q_local * actual, slots);
        const double sc = f.fill - merge_penalty * (actual - 1u);
        if (!have || sc > best.score + 1e-9) {
            have = true;
            best.chunks = actual;
            best.chunk_tokens = ct;
            best.total_wgs = n_q_local * actual;
            best.fill = f;
            best.score = sc;
        }
    }
    return best;
}

} // namespace vegajet::planner
