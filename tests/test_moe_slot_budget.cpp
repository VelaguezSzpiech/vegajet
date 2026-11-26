// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/moe_slot_budget: real per-card DOT8 slot bytes (all layers, Gate+Up and optional Down), allocation
// overhead, the 30.50 GiB ceiling and the ONE reading of the 2.70 GiB figure: total available headroom,
// which includes the 1.1 GiB draft reserve, leaving ~1.60 GiB for the cache. Also pins the production-site
// measured-arena decision: both cards must fit, pending allocations count, and a zero grant disables.
#include "core/moe_slot_budget.hpp"
#include "core/moe_lru_cache.hpp"
#include "tests/test_common.hpp"

#include <algorithm>
#include <cstdio>

using namespace vegajet::moeslot;

namespace {

SlotBudgetInput baseline() {
    SlotBudgetInput in;
    in.n_layers = 48;
    in.n_experts = 256;
    in.local_ff = 512;
    in.hidden = 3072;
    in.slots_per_layer = 44;
    return in;
}

std::array<CardArenaMeasurement, kBudgetCards> measured(uint64_t used0, uint64_t used1,
                                                        uint64_t pending0 = 0, uint64_t pending1 = 0) {
    std::array<CardArenaMeasurement, kBudgetCards> m{};
    m[0].capacity_bytes = kArenaCeilingBytes;
    m[0].used_bytes = used0;
    m[0].pending_bytes = pending0;
    m[1].capacity_bytes = kArenaCeilingBytes;
    m[1].used_bytes = used1;
    m[1].pending_bytes = pending1;
    return m;
}

} // namespace

int main() {
    // ---- constants: 30.50 GiB ceiling, 1.1 GiB draft floor, 2.70 GiB TOTAL available, ~1.60 GiB usable ----
    {
        VJ_CHECK(kArenaCeilingBytes == 32749125632ull);
        VJ_CHECK(kDraftReserveMinBytes == 1181116007ull);    // ceil(1.1 GiB)
        VJ_CHECK(kHeadroomAvailableBytes == 2899102925ull);  // ceil(2.70 GiB), draft included
        VJ_CHECK(kCacheUsableBytes == 1717986918ull);        // 2.70 - 1.1 GiB
        VJ_CHECK(kHeadroomAvailableBytes == kDraftReserveMinBytes + kCacheUsableBytes);
        VJ_CHECK(kCacheUsableBytes < kHeadroomAvailableBytes); // not an additional reserve
    }

    // ---- real per-slot bytes: one merged Gate+Up expert = 1,703,936 B on this rank split ----
    {
        const SlotBudgetInput in = baseline();
        SlotBudgetPlan p = plan_slot_budget(in);
        VJ_CHECK(p.ok);
        VJ_CHECK(p.projections.gate_up_layer == 1703936ull);
        VJ_CHECK(p.projections.gate_up_per_slot == 48ull * 1703936ull);
        VJ_CHECK(p.projections.gate_up_per_slot == 81788928ull);   // 78 MiB / layer
        VJ_CHECK(p.projections.down_layer == 0 && p.projections.down_per_slot == 0);
        VJ_CHECK(p.projections.per_slot() == 81788928ull);

        // 44 slots x 48 layers: 3.351 GiB / card (3,598,712,832 B), the number the budget must justify.
        VJ_CHECK(p.requested.cache_bytes == 3598712832ull);
        VJ_CHECK(p.alignment_padding_bytes == 0ull); // rows*odd-256B strides are 4096-aligned at these shapes
        VJ_CHECK(p.slot_table_bytes == 48ull * 1024ull);
        VJ_CHECK(p.fixed_overhead_bytes == 49152ull);

        // Down is a separate projection: 3,072 rows x 512 cols exact-DOT8 per expert.
        SlotBudgetInput both = in;
        both.cache_down = true;
        SlotBudgetPlan pd = plan_slot_budget(both);
        VJ_CHECK(pd.ok);
        VJ_CHECK(pd.projections.down_layer == 835584ull);
        VJ_CHECK(pd.projections.down_per_slot == 48ull * 835584ull);
        VJ_CHECK(pd.projections.down_per_slot == 40108032ull);
        VJ_CHECK(pd.requested.cache_bytes == 3598712832ull + 48ull * 44ull * 835584ull);
        VJ_CHECK(pd.requested.cache_bytes == 5363466240ull);

        // 49 layers = 48 trunk + blk.48 MTP: "all layers" must not silently mean "48" if MTP experts are cached.
        SlotBudgetInput mtp = in;
        mtp.n_layers = 49;
        SlotBudgetPlan pm = plan_slot_budget(mtp);
        VJ_CHECK(pm.ok);
        VJ_CHECK(pm.requested.cache_bytes == 49ull * 44ull * 1703936ull);
        VJ_CHECK(pm.slot_table_bytes == 49ull * 1024ull);
    }

    // ---- exact cache cost is linear in slots and zero when disabled ----
    {
        const SlotBudgetInput in = baseline();
        VJ_CHECK(cache_bytes_for_slots(in, 0) == 0);
        VJ_CHECK(slot_table_bytes_for(in, 0) == 0);
        VJ_CHECK(cache_bytes_for_slots(in, 1) == 48ull * 1703936ull);
        VJ_CHECK(cache_bytes_for_slots(in, 44) == 44ull * cache_bytes_for_slots(in, 1));
        VJ_CHECK(cache_bytes_for_slots(in, 64) == 64ull * cache_bytes_for_slots(in, 1));
        uint64_t pad = 1;
        (void)cache_bytes_for_slots(in, 44, &pad);
        VJ_CHECK(pad == 0);
    }

    // ---- one policy: 2.70 GiB total available, 1.1 GiB draft, only ~1.60 GiB cache-usable ----
    {
        const SlotBudgetInput in = baseline();
        SlotBudgetPlan p = plan_slot_budget(in);
        VJ_CHECK(p.ok);
        VJ_CHECK(p.headroom.ok);
        VJ_CHECK(p.headroom.available_bytes == 2899102925ull);
        VJ_CHECK(p.headroom.draft_bytes == 1181116007ull);
        VJ_CHECK(p.headroom.cache_usable_bytes == 1717986918ull);

        // 44 slots cost 3.35 GiB of cache, which cannot come out of the 1.60 GiB usable headroom,
        // even with an empty measured base. The grant is capped to the largest fitting count.
        VJ_CHECK(max_base_bytes_for_slots(in, 44) == 0ull);
        VJ_CHECK(!p.fits);
        VJ_CHECK(p.max_slots_fit == 21);
        VJ_CHECK(p.slots_granted == 21);
        VJ_CHECK(p.why.find("grant 21 of 44") != std::string::npos);

        // 21 slots: 1,717,567,488 B cache + 49,152 B tables <= 1,717,986,918 B usable.
        VJ_CHECK(21ull * 81788928ull == 1717567488ull);
        VJ_CHECK(21ull * 81788928ull + 49152ull == 1717616640ull);
        VJ_CHECK(22ull * 81788928ull + 49152ull > 1717986918ull);

        // The 21-slot boundary: at the exact maximum base it fits; one byte more grants 20.
        SlotBudgetInput at = in;
        at.slots_per_layer = 21;
        at.used_base_bytes = 29850392985ull;
        SlotBudgetPlan pa = plan_slot_budget(at);
        VJ_CHECK(pa.fits);
        VJ_CHECK(pa.requested.remaining_bytes == 0);
        VJ_CHECK(pa.max_slots_fit == 21);
        VJ_CHECK(pa.slots_granted == 21);
        at.used_base_bytes = 29850392986ull;
        SlotBudgetPlan pb = plan_slot_budget(at);
        VJ_CHECK(!pb.fits);
        VJ_CHECK(pb.requested.remaining_bytes == -1);
        VJ_CHECK(pb.max_slots_fit == 20);
        VJ_CHECK(pb.slots_granted == 20);
        VJ_CHECK(pb.why.find("grant 20 of 21") != std::string::npos);

        // A lower measured headroom shrinks the usable cache budget directly.
        SlotBudgetInput low = in;
        low.headroom_available_bytes = 2ull * kGiB;
        VJ_CHECK(low.headroom_available_bytes - low.draft_reserve_bytes == 966367641ull);
        low.slots_per_layer = 44;
        SlotBudgetPlan pl = plan_slot_budget(low);
        VJ_CHECK(pl.ok && pl.max_slots_fit == 11 && pl.slots_granted == 11);
    }

    // ---- regression: the historical "384 MiB reserve + 8 slots" guard ignores the 2.70 GiB budget ----
    {
        SlotBudgetInput in = baseline();
        const uint64_t per_slot_all_layers = in.n_layers * 1703936ull;
        const uint64_t arena_free = 4ull * kGiB;
        const uint64_t old_reserve = 384ull * 1024ull * 1024ull;
        const uint32_t old_max_fit =
            arena_free > old_reserve ? static_cast<uint32_t>((arena_free - old_reserve) / per_slot_all_layers) : 0u;
        VJ_CHECK(old_max_fit >= 44); // the old guard would grant the full 44

        in.used_base_bytes = kArenaCeilingBytes - arena_free;
        SlotBudgetPlan p = plan_slot_budget(in);
        VJ_CHECK(!p.fits);
        VJ_CHECK(p.slots_granted == 21); // 1.60 GiB usable / 78 MiB per slot
        VJ_CHECK(p.requested.why.find("usable headroom") != std::string::npos);
        VJ_CHECK(p.why.find("grant 21 of 44") != std::string::npos);
    }

    // ---- measured production decision: measured free caps the 2.70 GiB claim; both cards; pending; zero ----
    {
        SlotBudgetInput in = baseline();
        in.used_base_bytes = 22ull * kGiB; // ignored by plan_measured_slots
        const uint64_t headroom = 2899102925ull;

        MultiCardGrant g = plan_measured_slots(in, measured(22ull * kGiB, 22ull * kGiB), 2);
        VJ_CHECK(g.ok);
        VJ_CHECK(g.slots_granted == 21);
        VJ_CHECK(g.requested_slots == 44);
        VJ_CHECK(g.effective_headroom_bytes[0] == headroom && g.effective_headroom_bytes[1] == headroom);
        VJ_CHECK(g.cache_bytes_per_card == 21ull * 81788928ull);
        VJ_CHECK(g.slot_table_bytes_per_card == 49152ull);
        VJ_CHECK(g.fits[0].ok && g.fits[1].ok);
        VJ_CHECK(g.why.find("grant 21 of 44") != std::string::npos);
        VJ_CHECK(measured_available_bytes({kArenaCeilingBytes, 22ull * kGiB, 0}) ==
                 static_cast<int64_t>(kArenaCeilingBytes - 22ull * kGiB));

        // The claim is a ceiling, not a floor: extra free arena is not spent on the cache.
        g = plan_measured_slots(in, measured(0, 0), 2);
        VJ_CHECK(g.ok && g.slots_granted == 21);
        VJ_CHECK(g.effective_headroom_bytes[0] == headroom);

        // A card with only 2 GiB free reduces the shared grant to 11 slots.
        g = plan_measured_slots(in, measured(22ull * kGiB, kArenaCeilingBytes - 2ull * kGiB), 2);
        VJ_CHECK(g.ok && g.slots_granted == 11 && g.limiting_card == 1);
        VJ_CHECK(g.effective_headroom_bytes[1] == 2ull * kGiB);
        VJ_CHECK(g.fits[0].ok && g.fits[1].ok);
        VJ_CHECK(g.why.find("card 1") != std::string::npos);

        // Pending bytes are subtracted exactly like used bytes: one byte below the 11-slot threshold
        // leaves only 10 slots on that card, and both cards are granted 10.
        VJ_CHECK(measured_available_bytes({1000, 600, 400}) == 0);
        VJ_CHECK(measured_available_bytes({1000, 600, 401}) == -1);
        const uint64_t thr11 = 2080843367ull;
        g = plan_measured_slots(in, measured(22ull * kGiB, kArenaCeilingBytes - thr11, 0, 1), 2);
        VJ_CHECK(g.ok && g.slots_granted == 10 && g.limiting_card == 1);

        // Pending can also make card 0 limiting.
        g = plan_measured_slots(in, measured(kArenaCeilingBytes - 2ull * kGiB, 22ull * kGiB, 0, 0), 2);
        VJ_CHECK(g.ok && g.slots_granted == 11 && g.limiting_card == 0);

        // Exactly the draft reserve left: zero slots fit, but the plan is still well formed.
        g = plan_measured_slots(in, measured(kArenaCeilingBytes - 1181116007ull, 22ull * kGiB), 2);
        VJ_CHECK(g.ok && g.slots_granted == 0);
        VJ_CHECK(g.fits[0].ok && g.fits[1].ok);
        VJ_CHECK(g.why.find("no slots fit") != std::string::npos);

        // Less than the draft reserve: still a completed disable decision.
        g = plan_measured_slots(in, measured(22ull * kGiB, kArenaCeilingBytes - 1181116006ull), 2);
        VJ_CHECK(g.ok && g.slots_granted == 0);
        VJ_CHECK(!g.fits[1].ok);
        VJ_CHECK(g.why.find("draft reserve") != std::string::npos);

        // Overcommitted card: zero grant, no crash.
        g = plan_measured_slots(in, measured(22ull * kGiB, kArenaCeilingBytes + 1ull), 2);
        VJ_CHECK(g.ok && g.slots_granted == 0);

        g = plan_measured_slots(in, measured(0, 0), 0);
        VJ_CHECK(!g.ok);
        g = plan_measured_slots(in, measured(0, 0), 3);
        VJ_CHECK(!g.ok);
        auto zero_cap = measured(0, 0);
        zero_cap[0].capacity_bytes = 0;
        g = plan_measured_slots(in, zero_cap, 2);
        VJ_CHECK(!g.ok && g.why.find("zero arena capacity") != std::string::npos);

        // Requested slots are capped, not forced: ask for 64 and get 21.
        SlotBudgetInput ask = in;
        ask.slots_per_layer = 64;
        g = plan_measured_slots(ask, measured(0, 0), 2);
        VJ_CHECK(g.ok && g.requested_slots == 64 && g.slots_granted == 21);
        VJ_CHECK(g.why.find("grant 21 of 64") != std::string::npos);
    }

    // ---- bad inputs fail closed with a reason ----
    {
        SlotBudgetInput in = baseline();
        std::string why;
        in.local_ff = 500;
        VJ_CHECK(!validate_budget_input(in, &why) && why.find("local_ff") != std::string::npos);

        in = baseline();
        in.slots_per_layer = 300;
        VJ_CHECK(!validate_budget_input(in, &why) && why.find("n_experts") != std::string::npos);

        in = baseline();
        in.cache_gate_up = false;
        VJ_CHECK(!validate_budget_input(in, &why) && why.find("no projection") != std::string::npos);

        in = baseline();
        in.draft_reserve_bytes = kDraftReserveMinBytes - 1;
        VJ_CHECK(!validate_budget_input(in, &why) && why.find("1.1 GiB/card") != std::string::npos);
        in.draft_reserve_bytes = kDraftReserveMinBytes;
        VJ_CHECK(validate_budget_input(in, &why));

        in = baseline();
        in.headroom_available_bytes = kDraftReserveMinBytes - 1;
        VJ_CHECK(!validate_budget_input(in, &why));
        VJ_CHECK(why.find("does not cover") != std::string::npos);

        in = baseline();
        in.headroom_available_bytes = kHeadroomAvailableBytes + 1;
        VJ_CHECK(max_base_bytes_for_slots(in, 21) == 29850392985ull); // draft, not headroom, sets base limit

        in = baseline();
        in.slot_table_bytes_per_layer = 16;
        VJ_CHECK(!validate_budget_input(in, &why) && why.find("slot table") != std::string::npos);

        SlotBudgetPlan bad = plan_slot_budget(in);
        VJ_CHECK(!bad.ok && !bad.why.empty());
    }

    // ---- the host LRU default (44 slots/layer) is compatible with the grant at a sane base use ----
    {
        SlotBudgetInput in = baseline();
        in.used_base_bytes = 22ull * kGiB;
        SlotBudgetPlan p = plan_slot_budget(in);
        VJ_CHECK(p.ok && !p.fits && p.slots_granted == 21);

        vegajet::moecache::CacheConfig cfg;
        cfg.n_layers = in.n_layers;
        cfg.n_experts = in.n_experts;
        cfg.slots_per_layer = p.slots_granted;
        vegajet::moecache::MoeLruCache cache(cfg);
        VJ_CHECK(cache.slots(0) == 21 && cache.check_invariants());
    }

    std::printf("test_moe_slot_budget: real slot bytes, one 2.70 GiB policy, measured both-card grant OK\nPASS\n");
    return 0;
}
