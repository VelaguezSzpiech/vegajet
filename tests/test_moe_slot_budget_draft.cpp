// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Combined allocator test: deployed DFlash2 drafter footprint + M3 hot-expert cache budget, CPU only.
//
// Two stacks, one per-card budget:
//   * the drafter stack: the exact VRAM an external Ornith-1.5-122B DFlash2 Q4_K_M drafter commits on the one
//     card it runs on (VJ_Q4_A8 weights with the engine's TCC strides, per-layer KV rings, df_alloc scratch);
//   * the cache stacks: the ACTIVE production VJ_Q4_DOT8 layout (merged Gate+Up, embedded scales, 3,328 B row)
//     and the SEALED layout (linear code plane + separate Scale plane, 2,560 B row) from the same M3 patch.
//
// Every drafter constant below is the header parse of the deployed file (read-only, offline; no GGUF, GPU or
// KFD access at run time):
//   Ornith-1.5-122B-DFlash2-Q4_K_M (deployed GGUF)
//   551,110,784 B, 96 tensors, Q4_K_M; block_count 6, embedding 3072, feed_forward 9216, head_count 32,
//   head_count_kv 8, key/value_length 128, sliding_window 2048, sliding_window_pattern [1,1,1,1,1,1],
//   block_size 16, selector_rank 256, selector_top_k 16, target_layers [2,8,15,21,27,33,40,46].
// Reproduce the numbers with results/budget-work/analysis.py; the pinned totals are in the same directory.
//
// What the test pins:
//   1. the drafter's real footprint is 707,303,424 B (~0.659 GiB) on ONE card, below the flat 1.1 GiB floor;
//   2. that footprint is dominated by weights + batch-scaled taps, not by KV (48 MiB at the deployed pattern);
//   3. the frozen policy still grants 21 active slots / 27 sealed slots per card, unchanged;
//   4. per-card measured reserves give 35 active / 46 sealed (marginal, drafter already in `used`) or
//      26 active / 34 sealed (defensive, footprint kept free on the host card only);
//   5. conditional bounds when the measured free arena is smaller, including the negative cases where the
//      drafter card is limiting, the reserve does not fit, or the arena is overcommitted;
//   6. controls: a full-attention drafter layer plan (218,103,808 B, 208 MiB) and the legacy all-ring plan
//      (100,663,296 B, 96 MiB) both stay below the floor at the default batch, and the prefill-chunk bound.
#include "core/moe_draft_budget.hpp"
#include "core/moe_lru_sealed.hpp"
#include "core/moe_slot_budget.hpp"
#include "tests/test_common.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#if defined(__has_include)
#if __has_include("core/spec_replay.hpp")
#include "core/spec_replay.hpp"
#define VJ_TEST_HAVE_SPEC_REPLAY 1
#endif
#endif

using namespace vegajet::moeslot;

namespace {

// kGiB comes from vegajet::moeslot (1 << 30); do not redeclare it next to the using-directive.
constexpr uint64_t kActiveSlot = 48ull * 1703936ull; // 48 trunk layers, one merged Gate+Up expert each

SlotBudgetInput baseline() {
    SlotBudgetInput in;
    in.n_layers = 48;
    in.n_experts = 256;
    in.local_ff = 512;
    in.hidden = 3072;
    in.slots_per_layer = 44;
    return in;
}

std::array<CardArenaMeasurement, kBudgetCards> measured(uint64_t used0, uint64_t used1, uint64_t pending0 = 0,
                                                        uint64_t pending1 = 0) {
    std::array<CardArenaMeasurement, kBudgetCards> m{};
    m[0].capacity_bytes = kArenaCeilingBytes;
    m[0].used_bytes = used0;
    m[0].pending_bytes = pending0;
    m[1].capacity_bytes = kArenaCeilingBytes;
    m[1].used_bytes = used1;
    m[1].pending_bytes = pending1;
    return m;
}

Dflash2Meta deployed_meta() {
    Dflash2Meta m; // defaults are the pinned deployed file
    m.swa_pattern.assign(m.n_layer, 1u);
    return m;
}

Dflash2Meta zlab_meta() { // control: 5 SWA layers + 1 full layer at W = 4096 (core/spec_replay kPattern)
    Dflash2Meta m;
    m.sliding_window = 4096;
    m.max_ctx = 32768; // the frozen "96 MiB -> 208 MiB" control runs at >= 32K context
    m.swa_pattern = {1, 1, 1, 1, 1, 0};
    return m;
}

} // namespace

int main() {
    // ---- 1. deployed meta is the pinned one, and the Q4A8 layout reproduces the engine's +256 swizzle ----
    {
        const Dflash2Meta m = deployed_meta();
        VJ_CHECK(dflash2_meta_valid(m));
        VJ_CHECK(m.n_layer == 6 && m.hidden == 3072 && m.n_ff == 9216);
        VJ_CHECK(m.n_head == 32 && m.n_head_kv == 8 && m.head_dim == 128);
        VJ_CHECK(m.sliding_window == 2048 && m.block_size == 16 && m.selector_rank == 256);
        VJ_CHECK(m.swa_pattern.size() == 6 && m.n_taps == 8);

        // fc.weight [n_taps*D, D] is the only swizzled plane: stride = raw + 256 = 12,544 B.
        const Q4a8Plane fc = q4a8_plane(m.hidden, m.n_taps * m.hidden);
        VJ_CHECK(fc.swizzled);
        VJ_CHECK(fc.row_stride == 12544u);
        VJ_CHECK(fc.qs_bytes == 3072ull * 12544ull);          // 38,535,168
        VJ_CHECK(fc.dm_bytes == 3072ull * 768ull * 4ull);     // 9,437,184
        VJ_CHECK(fc.total() == 47972352ull);
        // Control: a plain odd-256B stride (the value select_tcc_layout() alone returns) would be 786,432 B
        // smaller; that is exactly one 256 B tile per row. This is why the footprint is not a file-size copy.
        VJ_CHECK(fc.total() - 3072ull * 256ull == 47185920ull);

        const Q4a8Plane sel = q4a8_plane(m.selector_rank, m.hidden);
        VJ_CHECK(!sel.swizzled && sel.row_stride == 1792u && sel.total() == 557056ull);

        const DraftWeightBytes w = dflash2_weight_bytes(m);
        VJ_CHECK(w.top == 48553984ull);              // 2 norms + fc + selector_hidden
        VJ_CHECK(w.per_layer == 84075520ull);        // one 6-layer drafter block
        VJ_CHECK(w.total(m.n_layer) == 553007104ull);
    }

    // ---- 2. KV and scratch: the deployed footprint, plus the frozen-plan controls ----
    {
        const Dflash2Meta m = deployed_meta();
        const DraftKvBytes kv = dflash2_kv_bytes(m);
        // full_slots is unused with an all-SWA pattern; it documents the formula at the engine's
        // default --max-ctx = 8192 (last pow2 below the 32768 full_ctx_cap).
        VJ_CHECK(kv.full_slots == 8192u);
        VJ_CHECK(kv.n_swa == 6 && kv.n_full == 0);
        VJ_CHECK(kv.total == 50331648ull);            // 6 * 2 * 2048 * 8 * 128 * 2 = 48 MiB
        VJ_CHECK(kv.total == 6ull * 2ull * 2048ull * 8ull * 128ull * 2ull);

        // Controls reproduce the frozen host-track claim (96 MiB legacy -> 208 MiB per-layer plan) exactly.
        const DraftKvBytes zl = dflash2_kv_bytes(zlab_meta());
        VJ_CHECK(zl.n_swa == 5 && zl.n_full == 1);
        VJ_CHECK(zl.total == 218103808ull);           // 208 MiB
        VJ_CHECK(zl.total == 5ull * 2ull * 4096ull * 8ull * 128ull * 2ull +
                                2ull * 32768ull * 8ull * 128ull * 2ull);

        Dflash2Meta legacy = zlab_meta();
        legacy.swa_pattern.assign(legacy.n_layer, 1u); // every layer a W ring: the d51c19a layout
        VJ_CHECK(dflash2_kv_bytes(legacy).total == 100663296ull); // 96 MiB

        // KV-window control: sizing the rings from dflash.context_length (262,144) instead of the sliding
        // window would be 6,442,450,944 B and exceed the whole 1.1 GiB reserve. The reserve is adequate only
        // under the window reading the engine and the frozen plan use.
        const uint64_t full_ctx_kv = 6ull * 2ull * 262144ull * 8ull * 128ull * 2ull;
        VJ_CHECK(full_ctx_kv == 6442450944ull);
        VJ_CHECK(full_ctx_kv > kDraftReserveMinBytes);

#if defined(VJ_TEST_HAVE_SPEC_REPLAY)
        // Cross-check against the frozen implementation when the tree carries it.
        vegajet::spec_replay::DraftKvConfig kc;
        kc.n_layer = m.n_layer;
        kc.swa_pattern = m.swa_pattern;
        kc.swa_window = m.sliding_window;
        kc.block_len = m.block_size;
        kc.max_ctx = m.max_ctx;
        kc.full_ctx_cap = m.full_ctx_cap;
        kc.n_kv = m.n_head_kv;
        kc.head_dim = m.head_dim;
        const vegajet::spec_replay::DraftKvPlan p = vegajet::spec_replay::plan_draft_kv(kc);
        VJ_CHECK(p.ok && p.total_bytes == kv.total);
        VJ_CHECK(p.max_anchor_pos == std::numeric_limits<uint32_t>::max()); // no full layer: unbounded
        kc.swa_window = 4096;
        kc.max_ctx = zlab_meta().max_ctx; // the control's 208 MiB needs >= 32K full-layer context
        std::vector<uint8_t> zpat = {1, 1, 1, 1, 1, 0};
        kc.swa_pattern = zpat;
        const vegajet::spec_replay::DraftKvPlan pz = vegajet::spec_replay::plan_draft_kv(kc);
        VJ_CHECK(pz.ok && pz.total_bytes == zl.total);
        const vegajet::spec_replay::DraftKvPlan pl = vegajet::spec_replay::legacy_draft_kv(kc);
        VJ_CHECK(pl.ok && pl.total_bytes == 100663296ull);
        VJ_CHECK(pz.max_anchor_pos == 32768u - m.block_size); // B = block_size = 16 -> 32752
#endif

        // Scratch is the exact df_alloc list; only taps_va scales with max_batch.
        VJ_CHECK(dflash2_scratch_bytes(m, 1024) == 103952384ull); // 99.14 MiB at the default prefill chunk
        VJ_CHECK(dflash2_scratch_bytes(m, 128) == 15872000ull);
        VJ_CHECK(dflash2_scratch_bytes(m, 1024) - dflash2_scratch_bytes(m, 128) ==
                 (1024ull - 128ull) * 8ull * 3072ull * 4ull); // 88,080,384 B = taps_va delta
        VJ_CHECK(dflash2_scratch_bytes(m, 0) == 3289088ull);  // the fixed part

        // ---- 1 again: the combined footprint on the drafter's card ----
        const DraftFootprint f = dflash2_footprint(m, 1024, m.hidden * 4ull);
        VJ_CHECK(f.mask_embedding == 12288ull);
        VJ_CHECK(f.weights == 553007104ull && f.kv == 50331648ull && f.scratch == 103952384ull);
        VJ_CHECK(f.total() == 707303424ull);          // 674.54 MiB = 0.658728 GiB
        VJ_CHECK(f.total() < kDraftReserveMinBytes);  // the flat 1.1 GiB floor has 473,812,583 B of slack
        VJ_CHECK(kDraftReserveMinBytes - f.total() == 473812583ull);
        VJ_CHECK(f.total() > (600ull << 20));         // 674.5 MiB: the reserve is not trivially small either

        const DraftFootprint f128 = dflash2_footprint(m, 128, m.hidden * 4ull);
        VJ_CHECK(f128.total() == 619223040ull);       // 590.46 MiB
        const DraftFootprint fz = dflash2_footprint(zlab_meta(), 1024, zlab_meta().hidden * 4ull);
        VJ_CHECK(fz.total() == 875075584ull);         // 834.5 MiB: still below the floor
        VJ_CHECK(fz.total() < kDraftReserveMinBytes);
    }

    // ---- 3. the flat 1.1 GiB policy is unchanged: 21 active slots, 27 sealed slots per card ----
    {
        SlotBudgetInput in = baseline();
        const MultiCardGrant g = plan_measured_slots(in, measured(22ull * kGiB, 22ull * kGiB), 2);
        VJ_CHECK(g.ok && g.slots_granted == 21 && g.requested_slots == 44);
        std::printf("Active cache: %u hot-expert slots/layer (requested %u; capped-headroom fixture)\n",
                    unsigned(g.slots_granted), unsigned(g.requested_slots));
        VJ_CHECK(g.draft_reserve_bytes[0] == kDraftReserveMinBytes);
        VJ_CHECK(g.draft_reserve_bytes[1] == kDraftReserveMinBytes);
        VJ_CHECK(g.cache_bytes_per_card == 21ull * kActiveSlot);
        VJ_CHECK(g.slot_table_bytes_per_card == 49152ull);
        VJ_CHECK(g.why.find("grant 21 of 44") != std::string::npos);

        SlotBudgetPlan p = plan_slot_budget(in);
        VJ_CHECK(p.ok && p.max_slots_fit == 21 && p.slots_granted == 21);
        VJ_CHECK(p.headroom.cache_usable_bytes == 1717986918ull);
        VJ_CHECK(max_base_bytes_for_slots(in, 21) == 29850392985ull);

        // Same 2.70 GiB claim, sealed layout: 27 slots (62,914,560 B per slot, not 81,788,928).
        const sealed::CacheLayout sl = sealed::cache_layout(512, 3072, nullptr);
        VJ_CHECK(sl.qs_row_stride == 1792u && sl.scale_row_stride == 768u);
        VJ_CHECK(sl.slot_bytes() == 1310720ull);
        VJ_CHECK(48ull * sl.slot_bytes() == 62914560ull);
        VJ_CHECK((kCacheUsableBytes - 49152ull) / (48ull * sl.slot_bytes()) == 27ull);
    }

    // ---- 4. per-card measured reserves: the deployed order is marginal, the defensive mode is card-bounded ----
    {
        SlotBudgetInput in = baseline();
        const std::array<CardArenaMeasurement, kBudgetCards> free270 = measured(22ull * kGiB, 22ull * kGiB);
        const DraftFootprint f = dflash2_footprint(deployed_meta(), 1024, 3072ull * 4ull);

        // Default placement: host card 1, marginal (drafter already inside `used`).
        const std::array<uint64_t, kBudgetCards> marginal = draft_reserve_per_card(f, DraftPlacement{});
        VJ_CHECK(marginal[0] == 0 && marginal[1] == 0);

        const MultiCardGrant gm = plan_measured_slots_with_draft(in, free270, 2, f, DraftPlacement{});
        VJ_CHECK(gm.ok && gm.slots_granted == 35 && gm.requested_slots == 44);
        VJ_CHECK(gm.limiting_card == 0); // equal cards: the first fit is reported
        VJ_CHECK(gm.cache_bytes_per_card == 35ull * kActiveSlot);
        VJ_CHECK(gm.fits[0].ok && gm.fits[1].ok);
        VJ_CHECK(gm.why.find("grant 35 of 44") != std::string::npos);
        // The marginal plan must not claim the 1.1 GiB subtraction in its limiting reason.
        VJ_CHECK(gm.why.find("1.1 GiB draft reserve") == std::string::npos);
        VJ_CHECK(gm.draft_reserve_bytes[0] == 0 && gm.draft_reserve_bytes[1] == 0);
        VJ_CHECK(gm.effective_headroom_bytes[0] == kHeadroomAvailableBytes);
        // 35 slots is the true ceiling of the 2.70 GiB claim: 36 needs 2,944,401,408 B of cache.
        VJ_CHECK(35ull * kActiveSlot + 49152ull <= kHeadroomAvailableBytes);
        VJ_CHECK(36ull * kActiveSlot + 49152ull > kHeadroomAvailableBytes);

        // Defensive: keep the measured footprint free on the host card only. Card 1 is limiting at 26.
        const DraftPlacement defensive{1, DraftReserveMode::ActualFootprint};
        const std::array<uint64_t, kBudgetCards> dres = draft_reserve_per_card(f, defensive);
        VJ_CHECK(dres[0] == 0 && dres[1] == 707303424ull);
        const MultiCardGrant gd = plan_measured_slots_with_draft(in, free270, 2, f, defensive);
        VJ_CHECK(gd.ok && gd.slots_granted == 26 && gd.limiting_card == 1);
        VJ_CHECK(gd.draft_reserve_bytes[1] == 707303424ull);
        VJ_CHECK(gd.why.find("card 1 is limiting") != std::string::npos);
        VJ_CHECK(gd.fits[0].ok && gd.fits[1].ok);
        VJ_CHECK(gd.fits[1].cache_bytes == 26ull * kActiveSlot);
        VJ_CHECK(26ull * kActiveSlot + 49152ull <= kHeadroomAvailableBytes - 707303424ull);
        VJ_CHECK(27ull * kActiveSlot + 49152ull > kHeadroomAvailableBytes - 707303424ull);

        // The non-drafter card alone would take 35; only the both-card minimum is granted.
        VJ_CHECK((kHeadroomAvailableBytes - 49152ull) / kActiveSlot == 35ull);

        // VEGAJET_DFLASH2_GPU0 swaps the host card; the reserve mirrors, the grant does not change.
        const DraftPlacement on0{0, DraftReserveMode::ActualFootprint};
        const std::array<uint64_t, kBudgetCards> r0 = draft_reserve_per_card(f, on0);
        VJ_CHECK(r0[0] == 707303424ull && r0[1] == 0);
        const MultiCardGrant g0 = plan_measured_slots_with_draft(in, free270, 2, f, on0);
        VJ_CHECK(g0.ok && g0.slots_granted == 26 && g0.limiting_card == 0);

        // Requested counts are still capped, never pushed up.
        SlotBudgetInput ask8 = baseline();
        ask8.slots_per_layer = 8;
        const MultiCardGrant g8 = plan_measured_slots_with_draft(ask8, free270, 2, f, DraftPlacement{});
        VJ_CHECK(g8.ok && g8.slots_granted == 8 && g8.why.empty());
        SlotBudgetInput ask64 = baseline();
        ask64.slots_per_layer = 64;
        const MultiCardGrant g64 = plan_measured_slots_with_draft(ask64, free270, 2, f, DraftPlacement{});
        VJ_CHECK(g64.ok && g64.slots_granted == 35 && g64.requested_slots == 64);
    }

    // ---- 5. conditional bounds and negative/control cases when the measured free space is smaller ----
    {
        SlotBudgetInput in = baseline();
        const DraftFootprint f = dflash2_footprint(deployed_meta(), 1024, 3072ull * 4ull);
        const DraftPlacement defensive{1, DraftReserveMode::ActualFootprint};

        // Card 1 keeps only 1.5 GiB for everything draft+cache. Frozen policy: 5 slots. Marginal: 19.
        const auto half = measured(22ull * kGiB, kArenaCeilingBytes - (3ull * kGiB) / 2ull);
        const MultiCardGrant gf = plan_measured_slots(in, half, 2);
        VJ_CHECK(gf.ok && gf.slots_granted == 5 && gf.limiting_card == 1);
        const MultiCardGrant gm = plan_measured_slots_with_draft(in, half, 2, f, DraftPlacement{});
        VJ_CHECK(gm.ok && gm.slots_granted == 19 && gm.limiting_card == 1);
        VJ_CHECK(gm.why.find("grant 19 of 44") != std::string::npos);
        VJ_CHECK(gf.slots_granted * 3 < gm.slots_granted); // the flat floor loses > 2/3 of the cache here

        // Defensive on the same card: 1.5 GiB - 674.5 MiB footprint -> 11 slots.
        const MultiCardGrant gd = plan_measured_slots_with_draft(in, half, 2, f, defensive);
        VJ_CHECK(gd.ok && gd.slots_granted == 11 && gd.limiting_card == 1);

        // Below the footprint: a completed disable decision, not a planning failure.
        const auto tight = measured(22ull * kGiB, kArenaCeilingBytes - (500ull << 20));
        const MultiCardGrant gt = plan_measured_slots_with_draft(in, tight, 2, f, defensive);
        VJ_CHECK(gt.ok && gt.slots_granted == 0);
        VJ_CHECK(gt.why.find("draft reserve") != std::string::npos);
        VJ_CHECK(!gt.fits[1].ok && gt.fits[0].ok);
        // Marginal on the same card still grants (the arena really has 500 MiB free): negative control that
        // shows the reserve, not the arena, is what disabled the cache above.
        const MultiCardGrant gtm = plan_measured_slots_with_draft(in, tight, 2, f, DraftPlacement{});
        VJ_CHECK(gtm.ok && gtm.slots_granted == 6); // (524,288,000 - 49,152) / 81,788,928 = 6

        // Overcommitted card and invalid geometry fail closed.
        const MultiCardGrant gov = plan_measured_slots_with_draft(in, measured(22ull * kGiB, kArenaCeilingBytes + 1), 2,
                                                                  f, DraftPlacement{});
        VJ_CHECK(gov.ok && gov.slots_granted == 0);
        VJ_CHECK(!plan_measured_slots_with_draft(in, measured(0, 0), 0, f, DraftPlacement{}).ok);
        VJ_CHECK(!plan_measured_slots_with_draft(in, measured(0, 0), 3, f, DraftPlacement{}).ok);
        auto zero_cap = measured(0, 0);
        zero_cap[0].capacity_bytes = 0;
        VJ_CHECK(!plan_measured_slots_with_draft(in, zero_cap, 2, f, DraftPlacement{}).ok);

        // The floor is not silently removed: the low-level planner rejects a sub-floor reserve unless the
        // caller opted out, and the default validator still rejects it too.
        std::string why;
        SlotBudgetInput floored = baseline();
        VJ_CHECK(validate_budget_input(floored, &why)); // the default floor reserve passes
        floored.draft_reserve_bytes = kDraftReserveMinBytes - 1ull;
        VJ_CHECK(!validate_budget_input(floored, &why));
        VJ_CHECK(why.find("1.1 GiB/card") != std::string::npos);
        floored.enforce_draft_floor = false;
        VJ_CHECK(validate_budget_input(floored, &why));
        const std::array<uint64_t, kBudgetCards> low{0, 0};
        const MultiCardGrant gfloor = plan_measured_slots_per_card(baseline(), measured(22ull * kGiB, 22ull * kGiB), 2, low);
        VJ_CHECK(!gfloor.ok && gfloor.why.find("enforce_draft_floor = false") != std::string::npos);

        // Pending bytes count exactly like used bytes: 6.5 GiB pending leaves 2 GiB free on card 1,
        // so the marginal policy grants 26 slots (2,147,483,648 - 49,152) / 81,788,928.
        const MultiCardGrant gp = plan_measured_slots_with_draft(
            in, measured(22ull * kGiB, 22ull * kGiB, 0, kArenaCeilingBytes - 22ull * kGiB - 2ull * kGiB), 2, f,
            DraftPlacement{});
        VJ_CHECK(gp.ok && gp.slots_granted == 26 && gp.limiting_card == 1);
        VJ_CHECK(gp.available_bytes[1] == static_cast<int64_t>(2ull * kGiB));
    }

    // ---- 6. conditional slot-limit table: the measured free space is the only missing input ----
    {
        // frozen policy: limit(free) = floor((min(2.70 GiB, free) - 1.1 GiB - tables) / 81,788,928)
        struct Case { uint64_t free_bytes; uint32_t frozen; uint32_t marginal; };
        const Case cases[] = {
            {kGiB * 12 / 10, 1u, 15u},  // 1.20 GiB
            {kGiB * 3 / 2, 5u, 19u},    // 1.50 GiB
            {kGiB * 2, 11u, 26u},       // 2.00 GiB
            {kGiB * 5 / 2, 18u, 32u},   // 2.50 GiB
            {kGiB * 11 / 4, 21u, 35u},  // 2.75 GiB: the 2.70 GiB claim caps both
        };
        const SlotBudgetInput in0 = baseline();
        SlotBudgetInput in = in0;
        in.used_base_bytes = kArenaCeilingBytes - (kGiB * 12 / 10); // ignored by the measured planner
        for (const Case& c : cases) {
            const auto m = measured(kArenaCeilingBytes - c.free_bytes, 22ull * kGiB);
            const MultiCardGrant gf = plan_measured_slots(in, m, 2);
            const MultiCardGrant gm =
                plan_measured_slots_with_draft(in, m, 2, dflash2_footprint(deployed_meta(), 1024, 12288), DraftPlacement{});
            VJ_CHECK(gf.ok && gm.ok);
            if (c.free_bytes >= kGiB * 11 / 4) {
                VJ_CHECK(gf.slots_granted == 21 && gm.slots_granted == 35); // claim-capped
            } else {
                VJ_CHECK(gf.slots_granted == c.frozen);
                VJ_CHECK(gm.slots_granted == c.marginal);
            }
        }
        // The 1.20 GiB row by hand: usable = 1,288,490,188 - 1,181,116,007 = 107,374,181 B; minus 49,152 B
        // tables leaves 107,325,029 B, so exactly one slot fits. Marginal: 1,288,441,036 / 81,788,928 = 15.75.
        VJ_CHECK((107374181ull - 49152ull) / kActiveSlot == 1ull);
        VJ_CHECK((1288490188ull - 49152ull) / kActiveSlot == 15ull);
    }

    // ---- 7. both cache stacks under each policy, including the mixed and Down-bearing cases ----
    {
        const sealed::CacheLayout gu = sealed::cache_layout(512, 3072, nullptr);
        const sealed::CacheLayout dn = sealed::cache_layout(3072, 512, nullptr);
        VJ_CHECK(gu.slot_bytes() == 1310720ull && dn.slot_bytes() == 1572864ull);
        VJ_CHECK(48ull * gu.slot_bytes() == 62914560ull);
        VJ_CHECK(48ull * dn.slot_bytes() == 75497472ull);

        // The active stack's Down projection is cheaper than the sealed Down plane; a mixed budget must add
        // the exact plane it allocates, not "a Down slot".
        SlotBudgetInput down = baseline();
        down.cache_down = true;
        VJ_CHECK(cache_bytes_for_slots(down, 1) == kActiveSlot + 48ull * 835584ull);
        VJ_CHECK(48ull * 835584ull == 40108032ull);
        VJ_CHECK(48ull * dn.slot_bytes() > 48ull * 835584ull);

        // Sealed limits under the same three policies (tables charged the same way; the sealed ledger itself
        // is host RAM and the per-job slots array lives in the launch buffer).
        VJ_CHECK((kCacheUsableBytes - 49152ull) / (48ull * gu.slot_bytes()) == 27ull);   // frozen, claim-capped
        VJ_CHECK((kHeadroomAvailableBytes - 49152ull) / (48ull * gu.slot_bytes()) == 46ull); // marginal
        VJ_CHECK((kHeadroomAvailableBytes - 707303424ull - 49152ull) / (48ull * gu.slot_bytes()) == 34ull); // defensive
        // Sealed Gate+Up and Down together: 138,412,032 B per slot -> 20 slots at the claim.
        VJ_CHECK(48ull * (gu.slot_bytes() + dn.slot_bytes()) == 138412032ull);
        VJ_CHECK((kHeadroomAvailableBytes - 49152ull) / 138412032ull == 20ull);
        // Active Gate+Up + Down: 121,896,960 B per slot -> 23 slots.
        VJ_CHECK(kActiveSlot + 48ull * 835584ull == 121896960ull);
        VJ_CHECK((kHeadroomAvailableBytes - 49152ull) / 121896960ull == 23ull);

        // The two layouts are not interchangeable: a sealed-sized plan must not be used for the active kernel.
        VJ_CHECK(gu.slot_bytes() != 512ull * 3328ull);
        VJ_CHECK(512ull * 3328ull == 1703936ull);
    }

    // ---- 8. prefill-chunk bound: the floor covers the deployed drafter up to 5,843 tokens per batch ----
    {
        const Dflash2Meta m = deployed_meta();
        const uint64_t fixed = dflash2_weight_bytes(m).total(m.n_layer) + dflash2_kv_bytes(m).total + m.hidden * 4ull +
                               dflash2_scratch_bytes(m, 0);
        VJ_CHECK(fixed == 606640128ull);
        VJ_CHECK((kDraftReserveMinBytes - fixed) / (8ull * 3072ull * 4ull) == 5843ull);
        VJ_CHECK(dflash2_footprint(m, 5843, m.hidden * 4ull).total() <= kDraftReserveMinBytes);
        VJ_CHECK(dflash2_footprint(m, 5844, m.hidden * 4ull).total() > kDraftReserveMinBytes);
        // With a full-attention drafter layer (208 MiB KV) the same floor only covers 4,137 tokens per batch.
        const Dflash2Meta z = zlab_meta();
        const uint64_t zfixed = dflash2_weight_bytes(z).total(z.n_layer) + dflash2_kv_bytes(z).total + z.hidden * 4ull +
                                dflash2_scratch_bytes(z, 0);
        VJ_CHECK(zfixed == 774412288ull);
        VJ_CHECK((kDraftReserveMinBytes - zfixed) / (8ull * 3072ull * 4ull) == 4137ull);
        VJ_CHECK(dflash2_footprint(z, 4138, z.hidden * 4ull).total() > kDraftReserveMinBytes);
    }

    // ---- 9. boundary: the measured-footprint ceiling and the exact grant arithmetic ----
    {
        SlotBudgetInput in = baseline();
        const uint64_t per_slot = kActiveSlot;
        const uint64_t fixed = 35ull * per_slot + 49152ull; // 2,862,661,632 B
        VJ_CHECK(fixed == 2862661632ull);
        VJ_CHECK(kArenaCeilingBytes - fixed == 29886464000ull);
        in.slots_per_layer = 35;
        in.used_base_bytes = 29886464000ull;
        SlotBudgetPlan p = plan_slot_budget(in);
        VJ_CHECK(p.ok && !p.fits && p.slots_granted == 20); // frozen floor: 20 slots at this base
        // With the floor disabled (measured-reserve callers) the same base fits exactly.
        in.enforce_draft_floor = false;
        in.draft_reserve_bytes = 0;
        in.headroom_available_bytes = kHeadroomAvailableBytes;
        p = plan_slot_budget(in);
        VJ_CHECK(p.ok && p.fits && p.requested.remaining_bytes == 0);
        in.used_base_bytes += 1ull;
        p = plan_slot_budget(in);
        VJ_CHECK(p.ok && !p.fits && p.slots_granted == 34 && p.requested.remaining_bytes == -1);
    }

    std::printf("test_moe_slot_budget_draft: deployed DFlash2 footprint + active/sealed cache budgets OK\nPASS\n");
    return 0;
}
