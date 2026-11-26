// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Production ledger regression: measured draft residency is in used(), not a second allocation.
// Offline fixtures only; no device, model file or environment access.
#include "core/moe_draft_budget.hpp"
#include "core/spec_replay.hpp"
#include "tests/test_common.hpp"

#include <algorithm>
#include <array>
#include <cstdio>

using namespace vegajet;
using namespace vegajet::moeslot;

namespace {
using Measurements = std::array<CardArenaMeasurement, kBudgetCards>;
using DraftBytes = std::array<uint64_t, kBudgetCards>;

Measurements with_free(uint64_t free0, uint64_t free1) {
    return {{{kArenaCeilingBytes, kArenaCeilingBytes - free0, 0u},
             {kArenaCeilingBytes, kArenaCeilingBytes - free1, 0u}}};
}

uint32_t pooled_fit(const Measurements& measured, const DraftBytes& draft, uint32_t requested) {
    // Independent fixed-shape floor arithmetic: active merged Gate+Up, 48 layers.
    uint32_t slots = std::min(requested, 21u);
    for (uint32_t c = 0; c < kBudgetCards; ++c) {
        const uint64_t occupied = measured[c].used_bytes + measured[c].pending_bytes;
        const uint64_t free = occupied < measured[c].capacity_bytes ? measured[c].capacity_bytes - occupied : 0u;
        const uint64_t available = std::min(free, kHeadroomAvailableBytes);
        const uint64_t remainder = draft[c] < kDraftReserveMinBytes ? kDraftReserveMinBytes - draft[c] : 0u;
        const uint64_t fixed = remainder + 49152u;
        const uint32_t fit = available > fixed ? static_cast<uint32_t>((available - fixed) / 81788928u) : 0u;
        slots = std::min(slots, fit);
    }
    return slots;
}
} // namespace

int main() {
    Dflash2Meta meta;
    meta.swa_pattern.assign(6u, 1u); // deployed GGUF: six SWA layers, W=2048, B=16
    const DraftFootprint footprint = dflash2_footprint(meta, 1024u, 3072u * 4u);
    VJ_CHECK(footprint.kv == 50331648u); // 48 MiB, not the separate 96 -> 208 MiB control
    VJ_CHECK(footprint.total() == 707303424u);
    spec_replay::DraftKvConfig kv;
    kv.n_layer = meta.n_layer;
    kv.swa_pattern = meta.swa_pattern;
    kv.swa_window = meta.sliding_window;
    kv.block_len = meta.block_size;
    kv.max_ctx = meta.max_ctx;
    kv.full_ctx_cap = meta.full_ctx_cap;
    kv.n_kv = meta.n_head_kv;
    kv.head_dim = meta.head_dim;
    const auto planned = spec_replay::plan_draft_kv(kv);
    VJ_CHECK(planned.ok && planned.total_bytes == footprint.kv);

    SlotBudgetInput in;
    auto measured = with_free(kHeadroomAvailableBytes, kHeadroomAvailableBytes);
    DraftBytes draft{800000000u, 800000000u + footprint.total()};
    auto grant = plan_production_cache(in, measured, draft, 2u);
    VJ_CHECK(grant.ok && grant.slots_granted == 21u && grant.requested_slots == 44u);
    VJ_CHECK(grant.slots_granted == pooled_fit(measured, draft, in.slots_per_layer));
    VJ_CHECK(grant.draft_reserve_bytes[0] == kDraftReserveMinBytes - draft[0]);
    VJ_CHECK(grant.draft_reserve_bytes[1] == 0u); // resident draft exceeds the floor; no second charge
    VJ_CHECK(grant.why.find("production cap 21") != std::string::npos);
    VJ_CHECK(grant.fits[0].ok && grant.fits[1].ok);

    // Empty draft means preserve the whole reserve on both cards. Flat callers are unchanged.
    draft = {};
    grant = plan_production_cache(in, measured, draft, 2u);
    VJ_CHECK(grant.ok && grant.slots_granted == plan_measured_slots(in, measured, 2u).slots_granted);
    VJ_CHECK(grant.draft_reserve_bytes[0] == kDraftReserveMinBytes);
    VJ_CHECK(grant.draft_reserve_bytes[1] == kDraftReserveMinBytes);

    // Only card 1 hosts the deployed DFlash2. The remaining floor, not the whole footprint or floor,
    // must fit its free arena; card 0 retains its entire policy reserve.
    measured = with_free(kHeadroomAvailableBytes, kGiB);
    draft = {0u, footprint.total()};
    grant = plan_production_cache(in, measured, draft, 2u);
    VJ_CHECK(grant.ok && grant.slots_granted == 7u && grant.limiting_card == 1u);
    VJ_CHECK(grant.slots_granted == pooled_fit(measured, draft, in.slots_per_layer));
    VJ_CHECK(plan_measured_slots(in, measured, 2u).slots_granted == 0u); // double-charge control
    VJ_CHECK(grant.draft_reserve_bytes[1] == 473812583u);
    VJ_CHECK(grant.fits[1].total_bytes == measured[1].used_bytes + 473812583u +
                                           grant.cache_bytes_per_card + grant.slot_table_bytes_per_card);
    std::swap(measured[0], measured[1]);
    std::swap(draft[0], draft[1]);
    const auto swapped = plan_production_cache(in, measured, draft, 2u);
    VJ_CHECK(swapped.ok && swapped.slots_granted == 7u && swapped.limiting_card == 0u);

    measured[0].pending_bytes = 500000000u;
    grant = plan_production_cache(in, measured, draft, 2u);
    VJ_CHECK(grant.ok && grant.slots_granted == 1u);
    VJ_CHECK(grant.slots_granted == pooled_fit(measured, draft, in.slots_per_layer));
    measured[0].used_bytes += measured[0].pending_bytes;
    measured[0].pending_bytes = 0u;
    VJ_CHECK(plan_production_cache(in, measured, draft, 2u).slots_granted == grant.slots_granted);

    // Larger free measurements, looser caller ceilings and over-reserve draft never raise the cap.
    measured = with_free(10u * kGiB, 10u * kGiB);
    draft = {2u * kGiB, 2u * kGiB};
    in.slots_per_layer = 64u;
    in.headroom_available_bytes = 20u * kGiB;
    grant = plan_production_cache(in, measured, draft, 2u);
    VJ_CHECK(grant.ok && grant.slots_granted == 21u);
    VJ_CHECK(grant.effective_headroom_bytes[0] == kHeadroomAvailableBytes);
    in.slots_per_layer = 8u;
    VJ_CHECK(plan_production_cache(in, measured, draft, 2u).slots_granted == 8u);
    in.slots_per_layer = 0u;
    VJ_CHECK(plan_production_cache(in, measured, draft, 2u).slots_granted == 0u);

    VJ_CHECK(!plan_production_cache(in, measured, draft, 0u).ok);
    VJ_CHECK(!plan_production_cache(in, measured, draft, 3u).ok);
    measured[1].capacity_bytes = 0u;
    VJ_CHECK(!plan_production_cache(in, measured, draft, 2u).ok);
    measured = with_free(kHeadroomAvailableBytes, kHeadroomAvailableBytes);
    draft[1] = measured[1].used_bytes + 1u;
    VJ_CHECK(!plan_production_cache(in, measured, draft, 2u).ok);
    draft = {};
    in = SlotBudgetInput{};
    measured[1].used_bytes = kArenaCeilingBytes + 1u;
    grant = plan_production_cache(in, measured, draft, 2u);
    VJ_CHECK(grant.ok && grant.slots_granted == 0u);

    std::puts("test_moe_slot_budget_production: PASS (48 MiB deployed KV, measured once, <=21 slots)");
    return 0;
}
