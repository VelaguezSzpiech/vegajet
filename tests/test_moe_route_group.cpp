// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Expert-grouped routed MoE Down (ops/kernels/vj_moe_down_grouped.hip[h], core/moe_route_group.*), pure CPU:
//   1. moe_route_group_ref invariants, and a lane-level emulation of vj_moe_route_group_kernel (4 waves,
//      8-ballot equal-peer masks, mbcnt ranks, packed 13/9/10-bit scan) must reproduce it word for word.
//   2. CPU model of grouped_dp + combine_dp must equal the CPU model of vj_moe_down_dot8_kernel bit for bit
//      for every min_group (incl. all-skinny and all-wide); the f32 (fast) scheme is deterministic and close.
//   3. Prints the DRAM byte model at M = 5, 12, 128, 512 on a skewed synthetic routing.
// Build note: compile without FP contraction (e.g. -ffp-contract=off); the models round every product.
#include "core/moe_route_group.hpp"
#include "core/q4_dot8_repack.hpp"
#include "tests/test_common.hpp"

#include <algorithm>
#include <cmath>
#include <random>

using namespace vegajet;
using namespace vegajet::moe_grp;

namespace {

std::mt19937_64 rng(0xD0A7C0DEULL);

// 8 distinct experts per token, popularity ~ 1/(rank+1)^s over a fixed random permutation.
std::vector<int32_t> skewed_routing(uint32_t M, double s, uint64_t seed) {
    std::mt19937_64 r(seed);
    std::vector<uint32_t> perm(kNumExperts);
    for (uint32_t i = 0; i < kNumExperts; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), r);
    std::vector<double> w(kNumExperts);
    for (uint32_t i = 0; i < kNumExperts; ++i) w[i] = 1.0 / std::pow(static_cast<double>(i + 1), s);
    std::discrete_distribution<uint32_t> dist(w.begin(), w.end());
    std::vector<int32_t> ids(static_cast<size_t>(M) * kTopK);
    for (uint32_t m = 0; m < M; ++m) {
        uint32_t n = 0;
        while (n < kTopK) {
            const uint32_t e = perm[dist(r)];
            bool dup = false;
            for (uint32_t k = 0; k < n; ++k) dup |= static_cast<uint32_t>(ids[m * kTopK + k]) == e;
            if (!dup) ids[m * kTopK + n++] = static_cast<int32_t>(e);
        }
    }
    return ids;
}

// ---- lane-level emulation of vj_moe_route_group_kernel -----------------------------------------
uint64_t peers8(const uint32_t e[64], const bool valid[64], uint32_t lane) {
    uint64_t m = 0;
    for (uint32_t l = 0; l < 64u; ++l) m |= static_cast<uint64_t>(valid[l]) << l;
    for (uint32_t b = 0; b < 8u; ++b) {
        uint64_t bb = 0;
        for (uint32_t l = 0; l < 64u; ++l) bb |= static_cast<uint64_t>((e[l] >> b) & 1u) << l;
        m &= ((e[lane] >> b) & 1u) ? bb : ~bb;
    }
    return m;
}
uint32_t mbcnt(uint64_t m, uint32_t lane) { return static_cast<uint32_t>(__builtin_popcountll(m & ((1ull << lane) - 1ull))); }

void emulate_route_group_kernel(const int32_t* top_ids, uint32_t M, uint32_t min_group, uint32_t cap,
                                MoeRouteGroup* g) {
    const uint32_t R = std::min(M, kMaxM) * kTopK;
    const uint32_t Q = ((R + 255u) / 256u) * 64u;
    std::vector<uint8_t> id(R);
    for (uint32_t i = 0; i < R; ++i) id[i] = static_cast<uint8_t>(top_ids[i] & 255);
    std::vector<uint32_t> wc(4u * 256u, 0u);
    g->exp_cnt.assign(256, 0); g->exp_off.assign(257, 0); g->uniq.assign(256, 0xFFFFFFFFu);
    g->items.assign(cap, 0xFFFFFFFFu); g->routes.assign(R, 0xFFFFFFFFu);

    auto pass = [&](bool scatter) {
        for (uint32_t w = 0; w < 4u; ++w) {
            const uint32_t lo = std::min(R, w * Q), hi = std::min(R, lo + Q);
            for (uint32_t base = lo; base < hi; base += 64u) {
                uint32_t e[64]; bool v[64];
                for (uint32_t l = 0; l < 64u; ++l) { v[l] = base + l < hi; e[l] = v[l] ? id[base + l] : 0u; }
                uint32_t cur[64]; uint64_t pm[64];
                for (uint32_t l = 0; l < 64u; ++l) { pm[l] = peers8(e, v, l); cur[l] = wc[w * 256u + e[l]]; }  // one ds_read
                for (uint32_t l = 0; l < 64u; ++l) {
                    if (!v[l]) continue;
                    const uint32_t rank = mbcnt(pm[l], l);
                    if (scatter) g->routes[cur[l] + rank] = base + l;
                    if (rank == 0u) wc[w * 256u + e[l]] = cur[l] + static_cast<uint32_t>(__builtin_popcountll(pm[l]));
                }
            }
        }
    };
    pass(false);
    std::vector<uint32_t> pk(256), inc(256);
    for (uint32_t t = 0; t < 256u; ++t) {
        const uint32_t c = wc[t] + wc[256u + t] + wc[512u + t] + wc[768u + t];
        pk[t] = c | ((c != 0u ? 1u : 0u) << 13) | (items_for_count(c, min_group) << 22);
    }
    uint32_t run = 0;
    for (uint32_t t = 0; t < 256u; ++t) { run += pk[t]; inc[t] = run; }
    uint32_t grouped = 0;
    for (uint32_t t = 0; t < 256u; ++t) {
        const uint32_t c = pk[t] & 0x1FFFu, nit = pk[t] >> 22, ex = inc[t] - pk[t];
        const uint32_t off = ex & 0x1FFFu, uoff = (ex >> 13) & 0x1FFu, ioff = ex >> 22;
        g->exp_cnt[t] = c; g->exp_off[t] = off;
        if (c) g->uniq[uoff] = t;
        for (uint32_t k = 0; k < nit; ++k)
            if (ioff + k < cap) g->items[ioff + k] = item_pack(t, off + k * kItemRoutes, std::min(kItemRoutes, c - k * kItemRoutes));
        const uint32_t c0 = wc[t], c1 = wc[256u + t], c2 = wc[512u + t];
        wc[t] = off; wc[256u + t] = off + c0; wc[512u + t] = off + c0 + c1; wc[768u + t] = off + c0 + c1 + c2;
        if (nit) grouped += c;
    }
    g->exp_off[256] = inc[255] & 0x1FFFu;
    g->hdr[kHdrUnique] = (inc[255] >> 13) & 0x1FFu;
    g->hdr[kHdrItems] = std::min(inc[255] >> 22, cap);
    g->hdr[kHdrGroupedRoutes] = grouped;
    g->hdr[kHdrRoutes] = R;
    pass(true);
}

void check_route_group(const std::vector<int32_t>& ids, uint32_t M, uint32_t min_group) {
    MoeRouteGroup ref, dev;
    moe_route_group_ref(ids.data(), M, min_group, &ref);
    const uint32_t R = M * kTopK;
    // invariants
    VJ_CHECK(ref.exp_off[256] == R);
    uint32_t prev_e = 0, n_items = 0;
    for (uint32_t e = 0; e < 256u; ++e) {
        VJ_CHECK(ref.exp_off[e + 1] - ref.exp_off[e] == ref.exp_cnt[e]);
        for (uint32_t i = ref.exp_off[e]; i < ref.exp_off[e + 1]; ++i) {
            VJ_CHECK((static_cast<uint32_t>(ids[ref.routes[i]]) & 255u) == e);
            if (i > ref.exp_off[e]) VJ_CHECK(ref.routes[i] > ref.routes[i - 1]);  // ascending token, then slot
        }
        n_items += items_for_count(ref.exp_cnt[e], min_group);
    }
    for (uint32_t u = 0; u < ref.hdr[kHdrUnique]; ++u) {
        VJ_CHECK(ref.exp_cnt[ref.uniq[u]] > 0u);
        if (u) VJ_CHECK(ref.uniq[u] > prev_e);
        prev_e = ref.uniq[u];
    }
    VJ_CHECK(n_items == ref.hdr[kHdrItems] && n_items <= item_cap(M));
    uint32_t covered = 0;
    for (uint32_t it : ref.items) {
        VJ_CHECK(item_len(it) >= 1u && item_len(it) <= kItemRoutes);
        const uint32_t e = item_expert(it);
        VJ_CHECK(ref.exp_cnt[e] >= min_group);
        VJ_CHECK(item_start(it) >= ref.exp_off[e] && item_start(it) + item_len(it) <= ref.exp_off[e + 1]);
        covered += item_len(it);
    }
    VJ_CHECK(covered == ref.hdr[kHdrGroupedRoutes]);
    // device algorithm emulation must match word for word
    const uint32_t cap = item_cap(M);
    emulate_route_group_kernel(ids.data(), M, min_group, cap, &dev);
    for (uint32_t k = 0; k < kHdrWords; ++k) VJ_CHECK(dev.hdr[k] == ref.hdr[k]);
    VJ_CHECK(dev.exp_cnt == ref.exp_cnt);
    VJ_CHECK(dev.exp_off == ref.exp_off);
    VJ_CHECK(dev.routes == ref.routes);
    for (uint32_t u = 0; u < ref.hdr[kHdrUnique]; ++u) VJ_CHECK(dev.uniq[u] == ref.uniq[u]);
    for (uint32_t i = 0; i < ref.hdr[kHdrItems]; ++i) VJ_CHECK(dev.items[i] == ref.items[i]);
}

// ---- arithmetic ---------------------------------------------------------------------------------
struct Weights {
    uint32_t D = 0, w_stride = 256, s_stride = 16;
    std::vector<uint8_t> qs, sc;
};
Weights random_weights(uint32_t D) {
    Weights w;
    w.D = D;
    const uint64_t rows = static_cast<uint64_t>(kNumExperts) * D;
    w.qs.resize(rows * w.w_stride);
    w.sc.resize(rows * w.s_stride);
    for (auto& b : w.qs) b = static_cast<uint8_t>(rng());
    for (uint64_t r = 0; r < rows; ++r)
        for (uint32_t sb = 0; sb < 2u; ++sb) {
            const uint32_t nib = static_cast<uint32_t>(rng());
            const float d = std::ldexp(static_cast<float>(1 + rng() % 2047), -18) * ((rng() & 1u) ? -1.0f : 1.0f);
            std::memcpy(&w.sc[r * w.s_stride + sb * 8u], &nib, 4);
            std::memcpy(&w.sc[r * w.s_stride + sb * 8u + 4u], &d, 4);
        }
    return w;
}

uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

void check_arith(const Weights& W, uint32_t M, double skew, uint64_t seed) {
    const auto ids = skewed_routing(M, skew, seed);
    const uint32_t R = M * kTopK, D = W.D;
    std::vector<int8_t> mq(static_cast<size_t>(R) * kCols);
    std::vector<float> md(static_cast<size_t>(R) * 16u), tw(R), sh(static_cast<size_t>(M) * D);
    for (auto& v : mq) v = static_cast<int8_t>(static_cast<int>(rng() % 255u) - 127);
    for (auto& v : md) v = static_cast<float>(1 + rng() % 1000) * 1e-4f;
    for (auto& v : tw) v = static_cast<float>(1 + rng() % 1000) * 1e-3f;
    for (auto& v : sh) v = (static_cast<float>(rng() % 2001) - 1000.0f) * 1e-3f;

    MoeDownProblem p;
    p.w_qs = W.qs.data(); p.w_stride = W.w_stride; p.w_sc = W.sc.data(); p.s_stride = W.s_stride;
    p.mid_q8 = mq.data(); p.mid_d = md.data(); p.top_ids = ids.data(); p.top_w = tw.data();
    p.sh_down_in = (M & 1u) ? sh.data() : nullptr;
    p.M = M; p.D = D;

    std::vector<float> ref(static_cast<size_t>(M) * D);
    moe_down_per_token_ref(p, ref.data());

    for (uint32_t min_group : {1u, 2u, 3u, 4u, 1000u}) {
        MoeRouteGroup g;
        moe_route_group_ref(ids.data(), M, min_group, &g);
        // exact scheme
        std::vector<int32_t> dp(static_cast<size_t>(R) * D * 16u + 4u, 0x7F7F7F7F);  // +16 B pad (x4 over-read)
        moe_down_grouped_dp_ref(p, g, dp.data());
        std::vector<float> out(ref.size());
        moe_down_combine_dp_ref(p, g.exp_cnt.data(), min_group, dp.data(), out.data());
        for (size_t i = 0; i < out.size(); ++i) VJ_CHECK(f2u(out[i]) == f2u(ref[i]));
        // fast scheme
        std::vector<float> part(static_cast<size_t>(R) * D, 0.0f), out2(ref.size()), out3(ref.size());
        moe_down_grouped_f32_ref(p, g, part.data());
        moe_down_combine_f32_ref(p, g.exp_cnt.data(), min_group, part.data(), out2.data());
        moe_down_combine_f32_ref(p, g.exp_cnt.data(), min_group, part.data(), out3.data());
        double max_rel = 0.0, max_abs = 0.0;
        size_t n_diff = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            VJ_CHECK(f2u(out2[i]) == f2u(out3[i]));
            const double a = std::fabs(static_cast<double>(out2[i]) - ref[i]);
            n_diff += f2u(out2[i]) != f2u(ref[i]);
            max_abs = std::max(max_abs, a);
            max_rel = std::max(max_rel, a / std::max(1e-6, std::fabs(static_cast<double>(ref[i]))));
            VJ_CHECK(a <= 1e-4 * (1.0 + std::fabs(static_cast<double>(ref[i]))));
        }
        if (min_group >= 1000u) VJ_CHECK(n_diff == 0);  // nothing grouped: identical to per token
        std::printf("  M=%3u min_group=%4u: wide routes %4u/%4u items %3u | exact: bit-identical | "
                    "fast: %zu/%zu outputs differ, max |d| %.3g, max rel %.3g\n",
                    M, min_group, g.hdr[kHdrGroupedRoutes], R, g.hdr[kHdrItems], n_diff, ref.size(), max_abs, max_rel);
    }
}

void print_bytes(uint32_t M, double skew, uint64_t seed) {
    const auto ids = skewed_routing(M, skew, seed);
    MoeRouteGroup g;
    moe_route_group_ref(ids.data(), M, 2u, &g);
    uint32_t mx = 0;
    for (uint32_t c : g.exp_cnt) mx = std::max(mx, c);
    const uint32_t D = 3072;
    auto mb = [](uint64_t b) { return static_cast<double>(b) / 1e6; };
    std::printf("M=%3u routes %4u unique %3u max/expert %3u\n", M, M * kTopK, g.hdr[kHdrUnique], mx);
    for (int exact = 1; exact >= 0; --exact)
        for (uint32_t mg : {2u, 3u}) {
            const MoeDownBytes b = moe_down_bytes(g.exp_cnt.data(), D, mg, exact != 0);
            std::printf("  %-5s min_group=%u: per-token %8.1f MB | grouped W %7.1f + buf wr %7.1f + combine rd %7.1f = %8.1f MB  (%.2fx)\n",
                        exact ? "exact" : "fast", mg, mb(b.per_token), mb(b.grouped_weights), mb(b.buf_write),
                        mb(b.combine_reads), mb(b.grouped_total()),
                        static_cast<double>(b.per_token) / static_cast<double>(b.grouped_total()));
        }
}

} // namespace

int main() {
    // 1. routing groups
    for (uint32_t M : {1u, 2u, 3u, 5u, 8u, 12u, 31u, 64u, 128u, 200u, 511u, 512u})
        for (double s : {0.0, 1.0, 1.5, 3.0})
            for (uint32_t mg : {1u, 2u, 3u, 64u})
                check_route_group(skewed_routing(M, s, 1000u + M), M, mg);
    {   // adversarial: one expert in every token (count = M, > 64 -> several items), tails of 65
        const uint32_t M = 512;
        std::vector<int32_t> ids(M * kTopK);
        for (uint32_t m = 0; m < M; ++m)
            for (uint32_t s = 0; s < kTopK; ++s) ids[m * kTopK + s] = static_cast<int32_t>(s == 0 ? 7u : 8u + ((m / 65u) * 7u + s) % 248u);
        check_route_group(ids, M, 2u);
        check_route_group(ids, M, 1u);
    }
    std::printf("route grouping: CPU reference == lane-level device emulation (all cases)\n");

    // 2. arithmetic (D = 128 rows per expert keeps the CPU model fast; the per-row math is D independent)
    const Weights W = random_weights(128);
    for (uint32_t M : {1u, 2u, 5u, 12u, 128u}) check_arith(W, M, 1.5, 77u + M);
    check_arith(W, 512u, 1.5, 4242u);

    // 3. byte model, skewed synthetic routing (s = 1.5: ~unique 23 @ M=5, ~137 @ M=128; measured 25 / 116)
    for (uint32_t M : {5u, 12u, 128u, 512u}) print_bytes(M, 1.5, 9u + M);
    std::printf("test_moe_route_group: PASS\n");
    return 0;
}
