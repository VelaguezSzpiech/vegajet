// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Cross-module invariants of the host-side planners and codecs, pure CPU (no device, no model file):
//   1. core/q4_dot8_repack.hpp   merged IQ2_S Gate+Up re-code is exact against an independent IQ2_S dequantizer
//                                (embedded and separate scale planes, TP column slices, partial row ranges), and
//                                plan_dot8_promotion respects the pool at 0 bytes, partial fits and all 48 layers.
//   2. core/spec_policy.hpp      DrafterKvSync: eager / deferred catch-up across reject, partial accept, full
//                                accept, the catch-up window limit and the hole-fraction limit (with hysteresis).
//   3. core/fr_spec.hpp          ShortlistManager: construction, hit / miss accounting and the rebuild fallback,
//                                argmax merge on an empty shortlist dispatch, determinism.
//   4. core/prefill_planner.hpp  macro batches of 128 / 256 / 512, GDN sub-chunk independence, scratch budget.
// Sources needed: core/q4_dot8_repack.cpp core/spec_policy.cpp core/spec_engine.cpp core/fr_spec.cpp
//                 core/prefill_planner.cpp
#include "core/fr_spec.hpp"
#include "core/ggml_iq_tables.hpp"
#include "core/prefill_planner.hpp"
#include "core/q4_dot8_repack.hpp"
#include "core/spec_policy.hpp"
#include "tests/test_common.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <set>

using namespace vegajet;
namespace TB = vegajet::ggml_tables;

namespace {

std::mt19937_64 rng(0xA0D17C0DEULL);
uint32_t rnd(uint32_t n) { return static_cast<uint32_t>(rng() % n); }

uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
uint32_t rd32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
uint16_t rd16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }

// IEEE binary16 -> binary32, written out here so the check does not lean on the converter under test.
float half_to_float(uint16_t h) {
    const uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 31u, man = h & 1023u;
    float v;
    if (exp == 0u)       v = std::ldexp(static_cast<float>(man), -24);
    else if (exp == 31u) v = man ? std::nanf("") : INFINITY;
    else                 v = std::ldexp(static_cast<float>(man | 1024u), static_cast<int>(exp) - 25);
    return sign ? -v : v;
}

// Finite fp16 scale, both signs, mostly normal and sometimes subnormal.
uint16_t random_fp16() {
    const uint32_t sign = rnd(2) << 15;
    const uint32_t exp = (rnd(16) == 0) ? 0u : 1u + rnd(29);
    return static_cast<uint16_t>(sign | (exp << 10) | rnd(1024));
}

void random_iq2s_block(uint8_t* b) {
    const uint16_t d = random_fp16();
    std::memcpy(b, &d, 2);
    for (uint32_t i = 2; i < kIq2sBlockBytes; ++i) b[i] = static_cast<uint8_t>(rnd(256));
}

// Independent IQ2_S dequantizer for weight idx (0..255) of one 82 B superblock, following the ggml reference:
//   db = d * (0.5 + scale4) * 0.25, y = db * grid * sign.
// Layout: d fp16 @0, qs[32] @2, signs[32] @34, qh[8] @66, scales[8] @74.
float iq2s_dequant(const uint8_t* blk, uint32_t idx, int* int_weight) {
    const uint32_t ib = idx / 32u, w = idx % 32u, l = w / 8u, j = w % 8u;
    const uint32_t grid_index = blk[2u + 4u * ib + l] | (((static_cast<uint32_t>(blk[66u + ib]) >> (2u * l)) & 3u) << 8);
    const int mag = static_cast<int>((TB::iq2s_grid[grid_index] >> (8u * j)) & 0xFFu);
    const bool neg = (blk[34u + 4u * ib + l] >> j) & 1u;
    const uint32_t scale4 = (w < 16u) ? (blk[74u + ib] & 0xFu) : (blk[74u + ib] >> 4);
    const float db = half_to_float(rd16(blk)) * (0.5f + static_cast<float>(scale4)) * 0.25f;
    *int_weight = neg ? -mag : mag;
    return db * static_cast<float>(mag) * (neg ? -1.0f : 1.0f);
}

// Repack rows x cols (taken from superblock sb_begin of sources that are src_sb superblocks wide) and compare
// every weight of both tensors with the independent dequantizer, bit for bit.
void check_gu_exact(uint32_t rows, uint32_t cols, uint32_t src_sb, uint32_t sb_begin) {
    const uint32_t nsb = cols / 256u, nblk = cols / 32u;
    const uint64_t src_stride = static_cast<uint64_t>(src_sb) * kIq2sBlockBytes;
    std::vector<uint8_t> sg(rows * src_stride), su(rows * src_stride);
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t s = 0; s < src_sb; ++s) {
            random_iq2s_block(sg.data() + r * src_stride + s * kIq2sBlockBytes);
            random_iq2s_block(su.data() + r * src_stride + s * kIq2sBlockBytes);
        }
    const Dot8RowLayout lay = dot8_gu_iq2s_layout(rows, cols);
    VJ_CHECK(lay.rows == rows && lay.cols == cols);
    VJ_CHECK(lay.w_raw_row_bytes == nblk * 32u && lay.s_raw_row_bytes == nblk * 2u + nsb * 4u);
    VJ_CHECK(lay.w_row_stride_bytes % 256u == 0u && ((lay.w_row_stride_bytes / 256u) & 1u) == 1u);
    VJ_CHECK(lay.w_row_stride_bytes >= lay.w_raw_row_bytes);
    VJ_CHECK(lay.read_bytes_per_row() == static_cast<uint64_t>(nblk) * 34u + nsb * 4u);

    const uint8_t fill = 0xCD;
    std::vector<uint8_t> wp(lay.w_plane_bytes(), fill), sp(lay.s_plane_bytes(), fill);
    uint8_t* s_plane = lay.scale_embedded ? nullptr : sp.data();
    std::string err;
    VJ_CHECK(dot8_repack_gu_iq2s_rows(sg.data(), su.data(), src_stride, sb_begin, lay, 0, rows, wp.data(), s_plane, &err));
    VJ_CHECK(err.empty());

    std::vector<int8_t> wg(cols), wu(cols);
    for (uint32_t r = 0; r < rows; ++r) {
        const uint8_t* row = wp.data() + r * lay.w_row_stride_bytes;
        const uint8_t* sc = lay.scale_embedded ? row + lay.s_offset_bytes : sp.data() + r * lay.s_row_stride_bytes;
        // Padding after the payload is zero (never the fill pattern, never stale data).
        const uint32_t payload = lay.w_raw_row_bytes + (lay.scale_embedded ? lay.s_raw_row_bytes : 0u);
        for (uint32_t i = payload; i < lay.w_row_stride_bytes; ++i) VJ_CHECK(row[i] == 0u);

        dot8_decode_row_iq2_int(row, cols, 32, wg.data());
        dot8_decode_row_iq2_int(row + 16, cols, 32, wu.data());
        for (uint32_t i = 0; i < cols; ++i) {
            const uint32_t sb = sb_begin + i / 256u;
            const uint8_t* gb = sg.data() + r * src_stride + sb * kIq2sBlockBytes;
            const uint8_t* ub = su.data() + r * src_stride + sb * kIq2sBlockBytes;
            int gi = 0, ui = 0;
            const float gref = iq2s_dequant(gb, i % 256u, &gi);
            const float uref = iq2s_dequant(ub, i % 256u, &ui);
            VJ_CHECK(wg[i] == gi && wu[i] == ui);

            const uint32_t b = i / 32u, half = (i % 32u) / 16u;
            const uint32_t sc16 = rd16(sc + 2u * b);
            const uint32_t d2 = rd32(sc + 2u * nblk + 4u * (b / 8u));
            const uint32_t sg4 = half ? ((sc16 >> 4) & 0xFu) : (sc16 & 0xFu);
            const uint32_t su4 = half ? (sc16 >> 12) : ((sc16 >> 8) & 0xFu);
            const float dg = half_to_float(static_cast<uint16_t>(d2 & 0xFFFFu)) * (0.5f + static_cast<float>(sg4)) * 0.25f;
            const float du = half_to_float(static_cast<uint16_t>(d2 >> 16)) * (0.5f + static_cast<float>(su4)) * 0.25f;
            VJ_CHECK(f2u(dg * static_cast<float>(wg[i])) == f2u(gref));
            VJ_CHECK(f2u(du * static_cast<float>(wu[i])) == f2u(uref));
        }
    }

    // A partial row range writes exactly those rows and produces the same bytes as the full pass.
    if (rows >= 6u) {
        std::vector<uint8_t> wp2(lay.w_plane_bytes(), fill), sp2(lay.s_plane_bytes(), fill);
        uint8_t* s2 = lay.scale_embedded ? nullptr : sp2.data();
        VJ_CHECK(dot8_repack_gu_iq2s_rows(sg.data(), su.data(), src_stride, sb_begin, lay, 2, 5, wp2.data(), s2, &err));
        for (uint32_t r = 0; r < rows; ++r) {
            const bool in = r >= 2u && r < 5u;
            const uint8_t* a = wp.data() + r * lay.w_row_stride_bytes;
            const uint8_t* b = wp2.data() + r * lay.w_row_stride_bytes;
            for (uint32_t i = 0; i < lay.w_row_stride_bytes; ++i) VJ_CHECK(b[i] == (in ? a[i] : fill));
            if (!lay.scale_embedded) {
                const uint8_t* sa = sp.data() + r * lay.s_row_stride_bytes;
                const uint8_t* sb2 = sp2.data() + r * lay.s_row_stride_bytes;
                for (uint32_t i = 0; i < lay.s_row_stride_bytes; ++i) VJ_CHECK(sb2[i] == (in ? sa[i] : fill));
            }
        }
        // Empty range: success, nothing written.
        std::vector<uint8_t> wp3(lay.w_plane_bytes(), fill), sp3(lay.s_plane_bytes(), fill);
        VJ_CHECK(dot8_repack_gu_iq2s_rows(sg.data(), su.data(), src_stride, sb_begin, lay, 3, 3, wp3.data(),
                                          lay.scale_embedded ? nullptr : sp3.data(), &err));
        VJ_CHECK(std::all_of(wp3.begin(), wp3.end(), [&](uint8_t v) { return v == fill; }));
    }

    // Bad arguments are rejected with a message and without touching the destination.
    std::vector<uint8_t> wp4(lay.w_plane_bytes(), fill);
    std::string e2;
    VJ_CHECK(!dot8_repack_gu_iq2s_rows(sg.data(), su.data(), src_stride, sb_begin, lay, 0, rows + 1u, wp4.data(), s_plane, &e2));
    VJ_CHECK(!e2.empty());
    e2.clear();
    VJ_CHECK(!dot8_repack_gu_iq2s_rows(nullptr, su.data(), src_stride, sb_begin, lay, 0, rows, wp4.data(), s_plane, &e2));
    VJ_CHECK(!e2.empty());
    e2.clear();
    const Dot8RowLayout wrong = dot8_iq3_layout(rows, cols);   // an IQ3_XXS layout is not a Gate+Up layout
    VJ_CHECK(!dot8_repack_gu_iq2s_rows(sg.data(), su.data(), src_stride, sb_begin, wrong, 0, rows, wp4.data(), s_plane, &e2));
    VJ_CHECK(!e2.empty());
    VJ_CHECK(std::all_of(wp4.begin(), wp4.end(), [&](uint8_t v) { return v == fill; }));
}

Dot8ClassSpec down_class(uint32_t n_layers, uint64_t old_b, uint64_t new_b) {
    Dot8ClassSpec c;
    c.name = "moe_down_iq3xxs";
    c.src_type = GgmlType::IQ3_XXS;
    c.role = Dot8Role::MoeDown;
    c.target = Dot8Target::VJ_Q4_DOT8;
    for (uint32_t l = 0; l < n_layers; ++l) c.layers.push_back(l);
    c.old_vram_bytes_per_layer = old_b;
    c.new_vram_bytes_per_layer = new_b;
    c.old_read_bytes_per_layer = 1000000;   // 3.06 bpw source streamed slowly ...
    c.new_read_bytes_per_layer = 1380000;   // ... 4.25 bpw target streamed fast enough to win
    c.bw_old_gbs = 100.0;
    c.bw_new_gbs = 200.0;
    return c;
}

void test_dot8() {
    // ---- exactness of the merged Gate+Up re-code
    check_gu_exact(8, 512, 2, 0);     // scale region embedded in the row padding
    check_gu_exact(8, 256, 1, 0);     // scale region in a separate plane
    check_gu_exact(8, 512, 5, 3);     // TP column slice: superblocks 3..4 of a 5-superblock source
    check_gu_exact(3, 3072, 12, 0);   // engine row width, fewer rows than the partial-range check needs
    VJ_CHECK(dot8_gu_iq2s_layout(8, 512).scale_embedded);
    VJ_CHECK(!dot8_gu_iq2s_layout(8, 256).scale_embedded);
    {   // cols must be a positive multiple of 256
        Dot8RowLayout bad = dot8_gu_iq2s_layout(4, 512);
        bad.cols = 500;
        uint8_t dummy[16] = {};
        std::string err;
        VJ_CHECK(!dot8_repack_gu_iq2s_rows(dummy, dummy, 82, 0, bad, 0, 0, dummy, dummy, &err));
        VJ_CHECK(!err.empty());
    }

    // ---- promotion planner: pool edge cases
    const uint64_t MiB = 1048576ull;
    const uint64_t old_b = 40ull * MiB, new_b = 56ull * MiB, delta = new_b - old_b;
    const std::vector<Dot8ClassSpec> one = {down_class(48, old_b, new_b)};
    const double us_layer = 1000000.0 / (100.0 * 1e3) - 1380000.0 / (200.0 * 1e3);
    VJ_CHECK(us_layer > 0.0);

    {   // nothing to plan
        const Dot8PromotionPlan p = plan_dot8_promotion({}, 123);
        VJ_CHECK(p.promotions.empty() && p.verdicts.empty() && p.used_bytes == 0 && p.pool_bytes == 123);
        VJ_CHECK(p.us_saved_total == 0.0);
    }
    {   // 0 bytes free: eligible but nothing promoted
        const Dot8PromotionPlan p = plan_dot8_promotion(one, 0);
        VJ_CHECK(p.promotions.empty() && p.used_bytes == 0 && p.us_saved_total == 0.0);
        VJ_CHECK(p.verdicts.size() == 1 && p.verdicts[0].eligible && p.verdicts[0].layers_promoted == 0);
        VJ_CHECK(p.verdicts[0].priority == 0 && dot8_class_priority(one[0]) == 0);
    }
    {   // 0 bytes free, but the new format is smaller and the old copy is released: every layer goes
        const std::vector<Dot8ClassSpec> shrink = {down_class(48, new_b, old_b)};
        const Dot8PromotionPlan p = plan_dot8_promotion(shrink, 0);
        VJ_CHECK(p.promotions.size() == 48 && p.verdicts[0].layers_promoted == 48 && p.used_bytes == 0);
        for (const Dot8Promotion& x : p.promotions) VJ_CHECK(x.delta_bytes < 0 && x.cum_bytes == 0);
    }
    {   // one byte short of a single layer
        const Dot8PromotionPlan p = plan_dot8_promotion(one, delta - 1);
        VJ_CHECK(p.promotions.empty() && p.used_bytes == 0);
    }
    {   // partial: 10.5 layers of pool -> exactly the first 10 candidate layers, in order
        const Dot8PromotionPlan p = plan_dot8_promotion(one, 10 * delta + delta / 2);
        VJ_CHECK(p.promotions.size() == 10 && p.verdicts[0].layers_promoted == 10);
        VJ_CHECK(p.used_bytes == 10 * delta && p.used_bytes <= p.pool_bytes && p.used_bytes + delta > p.pool_bytes);
        for (uint32_t i = 0; i < 10; ++i) {
            VJ_CHECK(p.promotions[i].layer == i && p.promotions[i].class_index == 0);
            VJ_CHECK(p.promotions[i].delta_bytes == static_cast<int64_t>(delta));
            VJ_CHECK(p.promotions[i].cum_bytes == (i + 1) * delta);
            VJ_CHECK(std::fabs(p.promotions[i].us_saved - us_layer) < 1e-9);
        }
        VJ_CHECK(std::fabs(p.us_saved_total - 10.0 * us_layer) < 1e-6);
    }
    {   // full 48 layers: an exact fit is accepted, one byte less drops the last layer
        const Dot8PromotionPlan full = plan_dot8_promotion(one, 48 * delta);
        VJ_CHECK(full.promotions.size() == 48 && full.used_bytes == 48 * delta);
        VJ_CHECK(full.verdicts[0].layers_promoted == 48);
        VJ_CHECK(full.promotions.back().layer == 47 && full.promotions.back().cum_bytes == 48 * delta);
        const Dot8PromotionPlan huge = plan_dot8_promotion(one, 1ull << 40);
        VJ_CHECK(huge.promotions.size() == 48 && huge.used_bytes == 48 * delta);
        const Dot8PromotionPlan short1 = plan_dot8_promotion(one, 48 * delta - 1);
        VJ_CHECK(short1.promotions.size() == 47 && short1.used_bytes == 47 * delta);
    }
    {   // the old copy stays resident: the full new size is charged
        std::vector<Dot8ClassSpec> keep = one;
        keep[0].frees_old = false;
        const Dot8PromotionPlan p = plan_dot8_promotion(keep, 10 * new_b + 1);
        VJ_CHECK(p.promotions.size() == 10 && p.used_bytes == 10 * new_b);
    }
    {   // two classes: a budget-limited higher class blocks the lower one only under strict priority
        Dot8ClassSpec low = down_class(4, 1 * MiB, 2 * MiB);
        low.name = "gate_up_iq2s";
        low.src_type = GgmlType::IQ2_S;
        low.role = Dot8Role::MoeGateUp;
        const std::vector<Dot8ClassSpec> two = {low, one[0]};   // input order is not priority order
        VJ_CHECK(dot8_class_priority(two[1]) < dot8_class_priority(two[0]));
        const uint64_t pool = 10 * delta + 3 * MiB;
        const Dot8PromotionPlan strict = plan_dot8_promotion(two, pool, true);
        VJ_CHECK(strict.promotions.size() == 10 && strict.verdicts[1].layers_promoted == 10);
        VJ_CHECK(strict.verdicts[0].eligible && strict.verdicts[0].layers_promoted == 0);
        for (const Dot8Promotion& x : strict.promotions) VJ_CHECK(x.class_index == 1);
        const Dot8PromotionPlan loose = plan_dot8_promotion(two, pool, false);
        VJ_CHECK(loose.verdicts[1].layers_promoted == 10 && loose.verdicts[0].layers_promoted == 3);
        VJ_CHECK(loose.used_bytes == 10 * delta + 3 * MiB && loose.used_bytes <= pool);
        VJ_CHECK(loose.promotions.size() == 13 && loose.promotions.front().class_index == 1);
        // With room for everything both are fully promoted, higher priority first.
        const Dot8PromotionPlan all = plan_dot8_promotion(two, 48 * delta + 4 * MiB, true);
        VJ_CHECK(all.promotions.size() == 52 && all.promotions[47].class_index == 1 && all.promotions[48].class_index == 0);
    }
    {   // ineligible classes never consume pool
        Dot8ClassSpec tie = one[0];            // bw ratio == bpw ratio: the crossover law is strict
        tie.bw_new_gbs = 138.0;
        Dot8ClassSpec unmeasured = one[0];
        unmeasured.bw_new_gbs = 0.0;
        Dot8ClassSpec banned = one[0];         // IQ2_S -> VJ_Q4_A8 at M=1 is a known regression
        banned.src_type = GgmlType::IQ2_S;
        banned.target = Dot8Target::VJ_Q4_A8;
        banned.batch_m = 1;
        banned.bw_new_gbs = 1000.0;
        const Dot8PromotionPlan p = plan_dot8_promotion({tie, unmeasured, banned}, 1ull << 40);
        VJ_CHECK(p.promotions.empty() && p.used_bytes == 0);
        for (const Dot8ClassVerdict& v : p.verdicts) VJ_CHECK(!v.eligible && v.layers_promoted == 0 && !v.reason.empty());
    }
}

// ---------------------------------------------------------------------------------------------------------------
void check_increasing_positions(const spec_policy::SyncPlan& p) {
    for (size_t i = 1; i < p.catchup.size(); ++i) VJ_CHECK(p.catchup[i].pos > p.catchup[i - 1].pos);
}

void test_drafter_kv_sync() {
    using namespace spec_policy;
    SyncConfig cfg;
    cfg.mode = SyncMode::kDeferred;
    cfg.catchup_window = 4;
    cfg.stash_slots = 6;
    cfg.hole_window = 16;
    cfg.max_hole_frac = 0.25;
    cfg.resume_hole_frac = 0.10;

    {   // neural cycles are always caught up at once
        DrafterKvSync s(cfg);
        // full accept: v0 was ingested by draft step 0, the 4 accepted drafts are replayed from the trunk rows
        SyncPlan p = s.after_verify({10, 11, 12, 13, 14}, 100, true, true);
        VJ_CHECK(p.stash.empty() && p.holes_added == 0 && p.catchup.size() == 4 && s.pending() == 0);
        for (uint32_t j = 0; j < 4; ++j) {
            VJ_CHECK(p.catchup[j].token == static_cast<int32_t>(11 + j) && p.catchup[j].pos == 101 + j);
            VJ_CHECK(p.catchup[j].h.kind == HSource::kTrunkRow && p.catchup[j].h.index == j);
        }
        // reject: only v0 was committed and the drafter already holds it
        p = s.after_verify({20}, 105, true, true);
        VJ_CHECK(p.stash.empty() && p.catchup.empty() && p.holes_added == 0 && s.pending() == 0);
        // partial accept (2 of 4), next cycle expected to be a suffix cycle: still eager, the drafter stays warm
        p = s.after_verify({30, 31, 32}, 106, true, false);
        VJ_CHECK(p.stash.empty() && p.catchup.size() == 2 && s.pending() == 0);
        VJ_CHECK(p.catchup[0].pos == 107 && p.catchup[1].pos == 108);
        VJ_CHECK(s.total_holes() == 0 && s.hole_fraction() == 0.0 && s.effective_mode() == SyncMode::kDeferred);
        // nothing committed at all (generation stopped at the context / token limit)
        p = s.after_verify({}, 109, false, false);
        VJ_CHECK(p.stash.empty() && p.catchup.empty() && p.holes_added == 0 && s.pending() == 0);
    }
    {   // suffix cycles defer: stash now, replay before the next neural cycle
        DrafterKvSync s(cfg);
        SyncPlan p = s.after_verify({40, 41, 42}, 200, false, false);      // partial accept, 3 committed
        VJ_CHECK(p.catchup.empty() && p.holes_added == 0 && p.stash.size() == 3 && s.pending() == 3);
        VJ_CHECK(p.stash[0].src.kind == HSource::kPend);
        VJ_CHECK(p.stash[1].src.kind == HSource::kTrunkRow && p.stash[1].src.index == 0);
        VJ_CHECK(p.stash[2].src.kind == HSource::kTrunkRow && p.stash[2].src.index == 1);
        VJ_CHECK(p.stash[0].dst_slot == 0 && p.stash[1].dst_slot == 1 && p.stash[2].dst_slot == 2);

        p = s.after_verify({43}, 203, false, false);                       // reject: plain token, window now full
        VJ_CHECK(p.stash.size() == 1 && p.stash[0].src.kind == HSource::kPend && p.stash[0].dst_slot == 3);
        VJ_CHECK(p.holes_added == 0 && s.pending() == 4 && s.total_holes() == 0);

        const SyncPlan replay = s.before_neural();
        VJ_CHECK(replay.stash.empty() && replay.catchup.size() == 4 && s.pending() == 0);
        check_increasing_positions(replay);
        for (uint32_t j = 0; j < 4; ++j) {
            VJ_CHECK(replay.catchup[j].token == static_cast<int32_t>(40 + j) && replay.catchup[j].pos == 200 + j);
            VJ_CHECK(replay.catchup[j].h.kind == HSource::kStash && replay.catchup[j].h.index == j);
        }
        VJ_CHECK(s.before_neural().catchup.empty());                       // idempotent once drained
    }
    {   // a deferred backlog is replayed first when the next cycle is likely neural
        DrafterKvSync s(cfg);
        (void) s.after_verify({50, 51}, 300, false, false);
        VJ_CHECK(s.pending() == 2);
        const SyncPlan p = s.after_verify({52, 53}, 302, false, true);
        VJ_CHECK(p.stash.empty() && p.catchup.size() == 4 && s.pending() == 0);
        check_increasing_positions(p);
        VJ_CHECK(p.catchup[0].h.kind == HSource::kStash && p.catchup[1].h.kind == HSource::kStash);
        VJ_CHECK(p.catchup[2].h.kind == HSource::kPend && p.catchup[3].h.kind == HSource::kTrunkRow);
        VJ_CHECK(p.catchup[3].token == 53 && p.catchup[3].pos == 303);
    }
    {   // window limit: the oldest pending positions become drafter KV holes; then the hole-fraction limit
        DrafterKvSync s(cfg);
        (void) s.after_verify({60, 61, 62}, 400, false, false);
        SyncPlan p = s.after_verify({63, 64, 65}, 403, false, false);      // 6 positions into a window of 4
        VJ_CHECK(p.holes_added == 2 && s.pending() == 4 && s.total_holes() == 2 && p.stash.size() == 3);
        VJ_CHECK(p.stash[0].dst_slot == 3 && p.stash[1].dst_slot == 4 && p.stash[2].dst_slot == 5);
        VJ_CHECK(std::fabs(s.hole_fraction() - 2.0 / 6.0) < 1e-12);
        VJ_CHECK(s.effective_mode() == SyncMode::kEager);                  // 0.33 > max_hole_frac

        // Latched eager: a suffix cycle now replays the surviving backlog and catches up directly.
        p = s.after_verify({66, 67}, 406, false, false);
        VJ_CHECK(p.stash.empty() && p.holes_added == 0 && s.pending() == 0 && p.catchup.size() == 6);
        check_increasing_positions(p);
        VJ_CHECK(p.catchup.front().pos == 402 && p.catchup.front().token == 62);   // 400 and 401 are the holes
        VJ_CHECK(p.catchup.back().pos == 407 && p.catchup.back().token == 67);
        std::set<uint32_t> slots;
        for (uint32_t j = 0; j < 4; ++j) {
            VJ_CHECK(p.catchup[j].h.kind == HSource::kStash);
            slots.insert(p.catchup[j].h.index);
        }
        VJ_CHECK(slots.size() == 4);                                       // live stash slots never collide

        // Hysteresis: the latch holds between the two thresholds and releases only below resume_hole_frac.
        uint32_t pos = 408, cycles = 0;
        bool saw_band = false;
        while (s.effective_mode() == SyncMode::kEager) {
            VJ_CHECK(++cycles < 64);
            if (s.hole_fraction() <= cfg.max_hole_frac && s.hole_fraction() >= cfg.resume_hole_frac) saw_band = true;
            const SyncPlan q = s.after_verify({70}, pos++, false, false);
            VJ_CHECK(q.stash.empty() && q.catchup.size() == 1 && q.holes_added == 0);
        }
        VJ_CHECK(saw_band && s.hole_fraction() < cfg.resume_hole_frac && s.total_holes() == 2);
        const SyncPlan q = s.after_verify({71}, pos, false, false);        // deferred again
        VJ_CHECK(q.catchup.empty() && q.stash.size() == 1 && s.pending() == 1);
    }
    {   // smallest catch-up window: one position survives, every older deferred position is a hole.
        // (catchup_window = 0 is not exercised: after_verify pops an empty deque for it, core/spec_policy.cpp
        //  "while (pending_.size() >= cfg_.catchup_window)"; no caller configures 0 today.)
        SyncConfig z = cfg;
        z.catchup_window = 1;
        z.max_hole_frac = 2.0;                                             // keep the latch out of this case
        DrafterKvSync s(z);
        const SyncPlan p = s.after_verify({80, 81, 82}, 500, false, false);
        VJ_CHECK(p.catchup.empty() && p.stash.size() == 3 && p.holes_added == 2 && s.pending() == 1);
        VJ_CHECK(s.total_holes() == 2);
        const SyncPlan r = s.before_neural();
        VJ_CHECK(r.catchup.size() == 1 && r.catchup[0].token == 82 && r.catchup[0].pos == 502);
        VJ_CHECK(r.catchup[0].h.kind == HSource::kStash && r.catchup[0].h.index == p.stash[2].dst_slot);
    }
    {   // too few stash slots are raised to window + 1, so a full window never aliases a slot
        SyncConfig t = cfg;
        t.catchup_window = 8;
        t.stash_slots = 1;
        t.max_hole_frac = 2.0;
        DrafterKvSync s(t);
        (void) s.after_verify({1, 2, 3, 4, 5}, 600, false, false);
        (void) s.after_verify({6, 7, 8, 9, 10}, 605, false, false);        // 10 positions, 2 holes
        const SyncPlan r = s.before_neural();
        VJ_CHECK(r.catchup.size() == 8 && r.catchup.front().pos == 602 && r.catchup.back().pos == 609);
        std::set<uint32_t> slots;
        for (const CatchupStep& c : r.catchup) slots.insert(c.h.index);
        VJ_CHECK(slots.size() == 8 && *slots.rbegin() <= 8u);
    }
    {   // configured eager mode never defers
        SyncConfig e = cfg;
        e.mode = SyncMode::kEager;
        DrafterKvSync s(e);
        const SyncPlan p = s.after_verify({90, 91, 92}, 700, false, false);
        VJ_CHECK(p.stash.empty() && p.catchup.size() == 3 && s.pending() == 0 && s.total_holes() == 0);
        VJ_CHECK(p.catchup[0].h.kind == HSource::kPend && s.effective_mode() == SyncMode::kEager);
    }
}

// ---------------------------------------------------------------------------------------------------------------
fr_spec::ManagerConfig manager_config() {
    fr_spec::ManagerConfig mc;
    mc.shortlist.rows = 512;
    mc.shortlist.balance = fr_spec::Balance::kEqualPerRank;
    mc.rebuild_min_tokens = 64;
    mc.rebuild_miss_rate = 0.02;
    mc.prefer_window = 128;
    return mc;
}

void check_shortlist_shape(const fr_spec::Shortlist& sl, const fr_spec::VocabSplit& split) {
    VJ_CHECK(sl.rows == 512 && sl.rank_rows.size() == 2 && sl.rank_base.size() == 2);
    VJ_CHECK(sl.rank_rows[0] == 256 && sl.rank_rows[1] == 256 && sl.rank_base[0] == 0 && sl.rank_base[1] == 256);
    VJ_CHECK(sl.row_to_token.size() == sl.rows && sl.token_to_row.size() >= split.vocab);
    uint32_t present = 0;
    for (uint32_t t = 0; t < split.vocab; ++t) present += sl.contains(static_cast<int32_t>(t)) ? 1u : 0u;
    VJ_CHECK(present == sl.rows);
    for (uint32_t r = 0; r < 2; ++r) {
        const auto& sorted = sl.sorted_local_rows[r];
        VJ_CHECK(sorted.size() == sl.rank_rows[r] && std::is_sorted(sorted.begin(), sorted.end()));
        VJ_CHECK(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end());
        std::vector<uint32_t> dev = sl.device_local_rows[r];
        for (uint32_t pos = 0; pos < dev.size(); ++pos) {   // device position <-> token map round trip
            const int32_t tok = sl.row_to_token[sl.rank_base[r] + pos];
            VJ_CHECK(tok == split.token_of(r, dev[pos]) && split.rank_of(tok) == r);
            VJ_CHECK(sl.token_to_row[static_cast<size_t>(tok)] == static_cast<int32_t>(sl.rank_base[r] + pos));
        }
        std::sort(dev.begin(), dev.end());
        VJ_CHECK(dev == sorted);                            // device order is a permutation of the sorted rows
    }
}

void test_shortlist_manager() {
    using namespace fr_spec;
    VocabSplit split;
    split.vocab = 4096;
    split.n_ranks = 2;
    const std::vector<int32_t> must = {4095, 2047, 2048};
    std::vector<int32_t> prompt;
    for (int i = 0; i < 40; ++i) prompt.push_back(i % 7);          // frequent low ids
    for (int32_t t : {4000, 3000, 2500, 1900}) prompt.push_back(t);   // rare ids, only reachable through `prefer`

    ShortlistManager m(split, manager_config());
    m.set_must(must);
    std::string err;
    VJ_CHECK(m.init(prompt, &err));
    VJ_CHECK(err.empty());
    const Shortlist first = m.current();
    check_shortlist_shape(first, split);
    VJ_CHECK(first.generation == 1);
    for (int32_t t : must) VJ_CHECK(first.contains(t));
    VJ_CHECK(first.n_must == must.size());
    for (int32_t t : prompt) VJ_CHECK(first.contains(t));          // prompt tokens fit the prefer budget
    VJ_CHECK(first.n_prefer_dropped == 0);
    for (int32_t t = 0; t < 32; ++t) VJ_CHECK(first.contains(t));  // head of the tokenizer-rank prior
    VJ_CHECK(!first.contains(3500) && !first.contains(-1) && !first.contains(4096));
    const double cov = coverage(first, m.model());
    VJ_CHECK(cov > 0.5 && cov <= 1.0 + 1e-9);

    // ---- hits: no misses, no rebuild however many tokens arrive
    VJ_CHECK(m.window_miss_rate() == 0.0 && !m.should_rebuild());
    std::vector<int32_t> hits(200);
    for (size_t i = 0; i < hits.size(); ++i) hits[i] = static_cast<int32_t>(i % 16);
    m.observe_verified(hits);
    VJ_CHECK(m.window_miss_rate() == 0.0 && !m.should_rebuild());

    // ---- misses: counted per token, rebuild only once the minimum token count is reached
    ShortlistManager m2(split, manager_config());
    m2.set_must(must);
    VJ_CHECK(m2.init(prompt, &err));
    m2.observe_verified({3500, 3501, 3500, 5});                   // 3 of 4 outside the shortlist
    VJ_CHECK(std::fabs(m2.window_miss_rate() - 0.75) < 1e-12);
    VJ_CHECK(!m2.should_rebuild());                               // 4 < rebuild_min_tokens
    std::vector<int32_t> context = prompt;
    std::vector<int32_t> more;
    for (int i = 0; i < 60; ++i) more.push_back(i % 2 ? 3500 : 3);
    m2.observe_verified(more);
    VJ_CHECK(std::fabs(m2.window_miss_rate() - 33.0 / 64.0) < 1e-12 && m2.should_rebuild());
    context.insert(context.end(), {3500, 3501, 3500, 5});
    context.insert(context.end(), more.begin(), more.end());

    // ---- fallback: the rebuild pulls the missed tokens in and resets the miss window
    VJ_CHECK(m2.rebuild(context, &err));
    const Shortlist& second = m2.current();
    check_shortlist_shape(second, split);
    VJ_CHECK(second.generation == 2 && second.contains(3500) && second.contains(3501));
    for (int32_t t : must) VJ_CHECK(second.contains(t));
    VJ_CHECK(m2.window_miss_rate() == 0.0 && !m2.should_rebuild());
    m2.observe_verified({3500, 3501});
    VJ_CHECK(m2.window_miss_rate() == 0.0);

    // ---- a failed rebuild keeps the previous shortlist usable
    {
        ManagerConfig bad = manager_config();
        bad.shortlist.rows = 300;                                 // not a multiple of 256
        ShortlistManager mb(split, bad);
        std::string e;
        VJ_CHECK(!mb.init(prompt, &e));
        VJ_CHECK(!e.empty() && mb.current().rows == 0 && !mb.current().contains(0));

        ShortlistManager mm(split, manager_config());
        std::vector<int32_t> too_many;                            // 600 must tokens cannot fit 512 rows
        for (int32_t t = 0; t < 600; ++t) too_many.push_back(t);
        VJ_CHECK(mm.init(prompt, &e));
        const std::vector<int32_t> before = mm.current().row_to_token;
        mm.set_must(too_many);
        e.clear();
        VJ_CHECK(!mm.rebuild(prompt, &e));
        VJ_CHECK(!e.empty() && mm.current().row_to_token == before && mm.current().generation == 1);
    }

    // ---- argmax merge: a drafted token outside the shortlist can never be produced; empty dispatch -> no pick
    {
        const uint32_t nb = 4;
        float v0[nb], v1[nb];
        int32_t i0[nb], i1[nb];
        const float* vals[2] = {v0, v1};
        const int32_t* ids[2] = {i0, i1};
        for (uint32_t b = 0; b < nb; ++b) { v0[b] = v1[b] = -1e30f; i0[b] = i1[b] = -1; }
        ArgmaxPick pick = merge_shortlist_argmax(first, vals, ids, nb);
        VJ_CHECK(pick.token == -1 && pick.shortlist_row == -1);    // caller falls back to the full-vocab head
        i0[1] = 3; v0[1] = -1e30f;                                 // a valid row at the init value still never wins
        VJ_CHECK(merge_shortlist_argmax(first, vals, ids, nb).token == -1);

        i0[1] = 3; v0[1] = 2.5f;
        i1[2] = 300; v1[2] = 7.0f;
        i1[3] = 100000; v1[3] = 99.0f;                             // out-of-range row id is ignored
        pick = merge_shortlist_argmax(first, vals, ids, nb);
        VJ_CHECK(pick.shortlist_row == 300 && pick.token == first.row_to_token[300] && pick.value == 7.0f);
        VJ_CHECK(first.contains(pick.token));

        v0[1] = 7.0f;                                              // exact tie: the full-vocab tie order decides
        pick = merge_shortlist_argmax(first, vals, ids, nb);
        const uint64_t ka = full_vocab_tie_key(first.row_to_token[3], split, first.full_layout);
        const uint64_t kb = full_vocab_tie_key(first.row_to_token[300], split, first.full_layout);
        VJ_CHECK(ka != kb && pick.shortlist_row == (ka < kb ? 3 : 300));
    }

    // ---- determinism: same inputs, same shortlist, byte for byte, through init and a rebuild
    {
        ShortlistManager a(split, manager_config()), b(split, manager_config());
        a.set_must(must);
        b.set_must(must);
        VJ_CHECK(a.init(prompt, &err) && b.init(prompt, &err));
        for (int round = 0; round < 2; ++round) {
            VJ_CHECK(a.current().row_to_token == b.current().row_to_token);
            VJ_CHECK(a.current().token_to_row == b.current().token_to_row);
            VJ_CHECK(a.current().device_local_rows == b.current().device_local_rows);
            VJ_CHECK(a.current().sorted_local_rows == b.current().sorted_local_rows);
            VJ_CHECK(a.current().generation == b.current().generation);
            a.observe_verified(more);
            b.observe_verified(more);
            VJ_CHECK(a.window_miss_rate() == b.window_miss_rate());
            VJ_CHECK(a.rebuild(context, &err) && b.rebuild(context, &err));
        }
        VJ_CHECK(a.current().row_to_token == b.current().row_to_token);
        VJ_CHECK(a.current().row_to_token == m2.current().row_to_token || a.current().generation != m2.current().generation);
    }
}

// ---------------------------------------------------------------------------------------------------------------
void test_prefill_planner() {
    using namespace prefill;
    const ModelDims dm = ornith_122b_tp2();
    const uint64_t MiB = 1048576ull;
    auto make = [&](uint32_t prompt, uint32_t m, uint32_t c) {
        Request rq;
        rq.prompt_tokens = prompt;
        rq.macro_m = m;
        rq.gdn_chunk = c;
        return plan(dm, rq);
    };

    // ---- macro batches 128 / 256 / 512 on a 2,048-token prompt
    std::vector<Plan> plans;
    for (uint32_t m : {128u, 256u, 512u}) {
        const Plan p = make(2048, m, 32);
        VJ_CHECK(p.ok && p.error.empty());
        VJ_CHECK(p.macro_m == m && p.n_macro == 2048u / m && p.macro_m * p.n_macro == 2048u);
        // AllReduce payload per direction: FP16 wire is the default, INT8 stays opt-in
        VJ_CHECK(p.ar.fp16_bytes == static_cast<uint64_t>(m) * dm.D * 2u);
        VJ_CHECK(p.ar.int8_bytes == static_cast<uint64_t>(m) * (dm.D + dm.D / 8u));
        VJ_CHECK(!p.ar.use_int8 && p.ar.per_layer == 2);
        VJ_CHECK(p.ar.fp16_chunks == (p.ar.fp16_bytes + MiB - 1) / MiB);
        VJ_CHECK(static_cast<uint64_t>(p.ar.fp16_chunks) * MiB >= p.ar.fp16_bytes);
        VJ_CHECK(p.ar.use_sdma);
        // working sets
        VJ_CHECK(p.bytes_aliased < p.bytes_no_alias);
        uint64_t sum = 0, c0 = 0, c1 = 0, c2 = 0;
        for (const Buffer& b : p.buffers) {
            VJ_CHECK(b.bytes > 0 && b.alias_class <= 2 && !b.name.empty());
            sum += b.bytes;
            (b.alias_class == 0 ? c0 : (b.alias_class == 1 ? c1 : c2)) += b.bytes;
        }
        VJ_CHECK(sum == p.bytes_no_alias && c0 + std::max(c1, c2) == p.bytes_aliased && c1 > 0 && c2 > 0);
        VJ_CHECK(!describe(p).empty());
        plans.push_back(p);
    }
    VJ_CHECK(plans[0].ar.fp16_chunks == 1 && plans[1].ar.fp16_chunks == 2 && plans[2].ar.fp16_chunks == 3);
    // Every buffer is either proportional to M or independent of it; nothing grows faster than M.
    for (size_t i = 0; i < plans[0].buffers.size(); ++i) {
        const uint64_t b128 = plans[0].buffers[i].bytes, b256 = plans[1].buffers[i].bytes, b512 = plans[2].buffers[i].bytes;
        VJ_CHECK(plans[0].buffers[i].name == plans[2].buffers[i].name);
        const bool fixed = b128 == b256 && b256 == b512;
        const bool linear = b256 == 2 * b128 && b512 == 4 * b128;
        const bool affine = b256 - b128 == (b512 - b256) / 2 && b512 > b256;   // constant + linear term
        VJ_CHECK(fixed || linear || affine);
    }
    VJ_CHECK(plans[2].bytes_aliased < 4 * plans[0].bytes_aliased + MiB);
    VJ_CHECK(plans[2].bytes_aliased > plans[1].bytes_aliased && plans[1].bytes_aliased > plans[0].bytes_aliased);
    // The same prompt streams the same number of causal KV tiles whatever the macro batch.
    VJ_CHECK(plans[0].attn.kv_tiles_total == plans[1].attn.kv_tiles_total);
    VJ_CHECK(plans[1].attn.kv_tiles_total == plans[2].attn.kv_tiles_total);

    // ---- GDN sub-chunk is sized independently of the macro batch
    for (uint32_t m : {128u, 256u, 512u}) {
        const Plan c16 = make(2048, m, 16), c32 = make(2048, m, 32);
        VJ_CHECK(c16.ok && c32.ok && c16.macro_m == m && c32.macro_m == m && c16.n_macro == c32.n_macro);
        VJ_CHECK(c16.gdn.chunk == 16 && c32.gdn.chunk == 32);
        VJ_CHECK(c16.gdn.chunks_per_macro == m / 16u && c32.gdn.chunks_per_macro == m / 32u);
        VJ_CHECK(c16.gdn.wgs == c32.gdn.wgs && c16.gdn.wgs == dm.gdn_nv * c16.gdn.wgs_per_head && c16.gdn.wgs >= 128u);
        VJ_CHECK(c16.bytes_no_alias == c32.bytes_no_alias && c16.bytes_aliased == c32.bytes_aliased);
        VJ_CHECK(c16.buffers.size() == c32.buffers.size());
        for (size_t i = 0; i < c16.buffers.size(); ++i) VJ_CHECK(c16.buffers[i].bytes == c32.buffers[i].bytes);
        VJ_CHECK(c16.ar.fp16_bytes == c32.ar.fp16_bytes && c16.ar.fp16_chunks == c32.ar.fp16_chunks);
        VJ_CHECK(c16.attn.kv_tiles_total == c32.attn.kv_tiles_total && c16.attn.wgs_per_macro == c32.attn.wgs_per_macro);
    }
    {   // ... and the macro batch does not change the GDN chunk
        const Plan a = make(2048, 128, 16), b = make(2048, 512, 16);
        VJ_CHECK(a.gdn.chunk == b.gdn.chunk && a.gdn.wgs == b.gdn.wgs && a.gdn.wgs_per_head == b.gdn.wgs_per_head);
        VJ_CHECK(!make(2048, 256, 24).ok && !make(2048, 256, 8).ok && !make(2048, 256, 64).ok);
    }

    // ---- partial final chunk and prompts shorter than one macro batch
    {
        const Plan p = make(2148, 512, 32);            // 4 full batches + 100 tokens
        VJ_CHECK(p.ok && p.macro_m == 512 && p.n_macro == 5);
        const Plan q = make(300, 512, 32);             // clamped to the prompt, rounded up to 16
        VJ_CHECK(q.ok && q.macro_m == 304 && q.n_macro == 1 && q.gdn.chunks_per_macro == 10);
        const Plan r = make(2049, 256, 16);
        VJ_CHECK(r.ok && r.n_macro == 9);
        VJ_CHECK(make(2048, 100, 32).macro_m == 96 && make(2048, 4, 32).macro_m == 16 && make(8192, 4096, 32).macro_m == 1024);
        VJ_CHECK(!make(0, 256, 32).ok && !make(0, 256, 32).error.empty());
        Request cont;                                  // a continuation streams more keys than a fresh prompt
        cont.prompt_tokens = 512; cont.macro_m = 256; cont.pos0 = 2048;
        Request fresh = cont;
        fresh.pos0 = 0;
        VJ_CHECK(plan(dm, cont).attn.kv_tiles_total > plan(dm, fresh).attn.kv_tiles_total);
        VJ_CHECK(plan(dm, cont).bytes_aliased == plan(dm, fresh).bytes_aliased);
    }

    // ---- scratch budget: pick the largest macro batch whose aliased working set fits
    {
        auto largest_fit = [&](uint64_t budget) {
            uint32_t best = 0;
            for (uint32_t m : {128u, 256u, 512u}) {
                const Plan p = make(2048, m, 32);
                if (p.ok && p.bytes_aliased <= budget) best = m;
            }
            return best;
        };
        const uint64_t b128 = plans[0].bytes_aliased, b256 = plans[1].bytes_aliased, b512 = plans[2].bytes_aliased;
        VJ_CHECK(largest_fit(0) == 0 && largest_fit(b128 - 1) == 0 && largest_fit(b128) == 128);
        VJ_CHECK(largest_fit(b256 - 1) == 128 && largest_fit(b256) == 256);
        VJ_CHECK(largest_fit(b512 - 1) == 256 && largest_fit(b512) == 512);
        // The largest setting stays well inside 1 GiB of scratch per GPU at FP16 intermediates.
        VJ_CHECK(b512 < 1024 * MiB);
        Request legacy;
        legacy.prompt_tokens = 2048; legacy.macro_m = 512; legacy.precision = Precision::kFp32Legacy;
        const Plan pl = plan(dm, legacy);
        VJ_CHECK(pl.ok && pl.bytes_aliased > b512 && pl.ar.fp16_bytes == plans[2].ar.fp16_bytes);
        Request wire;                                  // INT8 wire only when asked for; 512 rows still need 2 pieces
        wire.prompt_tokens = 2048; wire.macro_m = 512; wire.allow_int8_wire = true;
        const Plan pw = plan(dm, wire);
        VJ_CHECK(pw.ok && pw.ar.use_int8 && pw.ar.int8_chunks == 2 && pw.ar.int8_bytes < pw.ar.fp16_bytes);
    }
}

} // namespace

int main() {
    test_dot8();
    test_drafter_kv_sync();
    test_shortlist_manager();
    test_prefill_planner();
    std::printf("test_audit_invariants: OK\n");
    return 0;
}
