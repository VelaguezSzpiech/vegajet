// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Allocation budget for the per-layer hot-expert VJ_Q4_DOT8 cache (host side, no device types).
//
// The engine keeps `slots_per_layer` exact-DOT8 copies of the hottest routed experts of every MoE layer on
// each card (core/moe_lru_cache.hpp decides which experts). Sizing that cache needs more than the historical
// "free VRAM > 384 MiB + 8 slots" guard:
//   * the Ornith runner maps a hard 30.50 GiB arena per card (models/ornith/ornith_gpu.cpp);
//   * the policy assumes draft/MTP weights and state need at least 1.1 GiB per card; the measured
//     deployed drafter is smaller and single-card (next bullet);
//   * the planning figure "2.70 GiB headroom" is the TOTAL available space on top of the measured base,
//     and the 1.1 GiB draft reserve must be paid out of it: the cache itself may spend at most about
//     1.60 GiB (kHeadroomAvailableBytes - kDraftReserveMinBytes). It is NOT an additional reserve on top
//     of the draft. There is one policy; the old additive/outside readings are gone.
//   * the 1.1 GiB figure is a POLICY floor, not a measurement of one drafter: the deployed DFlash2 drafter
//     lives on one card only and is uploaded before this plan runs (core/moe_draft_budget.hpp measures
//     707,303,424 B total). `enforce_draft_floor = false` plus plan_measured_slots_per_card() lets a caller
//     use the measured per-card reserve; the default keeps the floor and the frozen behaviour.
//
// Slot bytes come from the real production layouts, never from per-parameter estimates: one Gate+Up slot is
// one merged IQ2_S expert as laid out by dot8_gu_iq2s_layout(local_ff, hidden); one Down slot is one
// IQ3_XXS expert as laid out by dot8_iq3_layout(hidden, local_ff). The cache planes are allocated once per
// layer for `slots * level_rows` rows (exactly what the runner does), so the byte cost is linear in slots.
#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace vegajet::moeslot {

inline constexpr uint64_t kGiB = 1ull << 30;

// 30.50 GiB arena ceiling that `rt.init(30.5, ...)` maps on each card.
inline constexpr uint64_t kArenaCeilingBytes = 30ull * kGiB + kGiB / 2ull; // 32,749,125,632 B

// Per-card reserve floor for the draft / MTP weights and state: 1.1 GiB, rounded up to whole bytes.
inline constexpr uint64_t kDraftReserveMinBytes = (11ull * kGiB + 9ull) / 10ull; // 1,181,116,007 B

// TOTAL per-card headroom on top of the measured base: 2.70 GiB, rounded up. It includes the draft reserve.
inline constexpr uint64_t kHeadroomAvailableBytes = (27ull * kGiB + 9ull) / 10ull; // 2,899,102,925 B

// What the cache may actually spend: headroom minus the draft reserve (~1.60 GiB).
inline constexpr uint64_t kCacheUsableBytes = kHeadroomAvailableBytes - kDraftReserveMinBytes; // 1,717,986,918 B

// Production allocation alignments (models/ornith/ornith_gpu.cpp: weight planes 4096, scale planes 256,
// per-layer expert->slot tables 256).
inline constexpr uint64_t kPlaneAlignmentBytes = 4096;
inline constexpr uint64_t kScaleAlignmentBytes = 256;
inline constexpr uint64_t kTableAlignmentBytes = 256;

inline constexpr uint32_t kBudgetCards = 2;

// Conservative production upper bound; a measured grant may be smaller, never larger.
inline constexpr uint32_t kProductionMaxSlots = 21u;

struct SlotBudgetInput {
    uint32_t n_layers = 48;   // trunk MoE layers; MTP blocks with routed experts are extra layers
    uint32_t n_experts = 256;
    uint32_t local_ff = 512;  // routed FFN rows per rank (Gate and Up each)
    uint32_t hidden = 3072;   // model width D
    uint32_t slots_per_layer = 44;
    bool cache_gate_up = true;  // merged IQ2_S Gate+Up -> exact DOT8 (production cache)
    bool cache_down = false;    // IQ3_XXS Down -> exact DOT8 (opt-in extra projection)
    uint64_t draft_reserve_bytes = kDraftReserveMinBytes;
    // true (default): `draft_reserve_bytes` is also a per-card floor and validate_budget_input() rejects a
    // smaller value. Set false only when the caller supplies measured per-card reserves through
    // plan_measured_slots_per_card() (core/moe_draft_budget.hpp).
    bool enforce_draft_floor = true;
    uint64_t headroom_available_bytes = kHeadroomAvailableBytes; // total, draft included
    uint64_t arena_ceiling_bytes = kArenaCeilingBytes;
    // Caller-measured arena use before this cache is planned (weights, shared experts, KV/SSM, workspace).
    uint64_t used_base_bytes = 0;
    // Further caller reserve that must stay free inside the ceiling (io ring, guard buffer, ...).
    uint64_t other_fixed_bytes = 0;
    uint32_t slot_table_bytes_per_layer = 256u * 4u; // expert -> slot table (int32 per expert)
};

// Exact byte cost of one slot (one expert copy of each enabled projection) on one card.
struct ProjectionBytes {
    uint64_t gate_up_layer = 0;  // one merged Gate+Up expert, one layer
    uint64_t down_layer = 0;     // one Down expert, one layer
    uint64_t gate_up_per_slot = 0;  // all layers
    uint64_t down_per_slot = 0;     // all layers
    uint64_t per_slot() const { return gate_up_per_slot + down_per_slot; }
};

// The single reserve reading: total available headroom (2.70 GiB unless measured free is smaller),
// the draft part and the cache part. Only cache_usable_bytes may be spent on the cache; available_bytes
// is not added on top of any other reserve.
struct HeadroomBreakdown {
    bool ok = false;
    std::string why;
    uint64_t available_bytes = 0;     // total available headroom (measured or claimed, whichever is smaller)
    uint64_t draft_bytes = 0;         // 1.1 GiB that must stay free inside available_bytes
    uint64_t cache_usable_bytes = 0;  // available_bytes - draft_bytes
};

struct SlotsFit {
    bool ok = false;
    uint32_t slots = 0;
    uint64_t cache_bytes = 0;
    uint64_t overhead_bytes = 0;   // alignment padding + slot tables; also paid from cache_usable_bytes
    uint64_t reserve_bytes = 0;    // draft reserve kept free (NOT headroom added on top)
    uint64_t total_bytes = 0;      // used_base + other_fixed + overhead + cache + draft
    int64_t remaining_bytes = 0;   // ceiling - total_bytes (negative when overcommitted)
    std::string why;
};

struct SlotBudgetPlan {
    bool ok = false;
    std::string why;
    ProjectionBytes projections;
    uint64_t alignment_padding_bytes = 0; // sum of per-layer align slacks
    uint64_t slot_table_bytes = 0;        // all layers (0 when the cache is disabled)
    uint64_t fixed_overhead_bytes = 0;    // alignment padding + tables
    uint64_t used_base_bytes = 0;
    uint64_t other_fixed_bytes = 0;
    uint64_t ceiling_bytes = 0;
    HeadroomBreakdown headroom;
    SlotsFit requested;          // the input slot count
    uint32_t max_slots_fit = 0;  // largest slot count that fits
    uint32_t slots_granted = 0;  // min(slots_per_layer, max_slots_fit), 0 when even 0 slots cannot fit
    bool fits = false;           // requested slot count fits
};

// One card's real arena numbers read at the production allocation site, before the cache is allocated.
struct CardArenaMeasurement {
    uint64_t capacity_bytes = 0;   // e.g. rt.vram_arena(g).capacity() (30.50 GiB)
    uint64_t used_bytes = 0;       // e.g. rt.vram_arena(g).used()
    uint64_t pending_bytes = 0;    // allocations already committed/queued but not yet reflected in used_bytes
};

// capacity - used - pending, or a negative value when already overcommitted (saturating at int64 min).
int64_t measured_available_bytes(const CardArenaMeasurement& m);

// Both-card decision at the production allocation site. Every card must fit; the grant is the minimum
// per-card fit, capped at the requested slots. `slots_granted == 0` means disable the cache.
struct MultiCardGrant {
    bool ok = false;              // planning succeeded (a zero grant is still ok, see slots_granted)
    std::string why;
    uint32_t requested_slots = 0;
    uint32_t slots_granted = 0;
    uint32_t limiting_card = 0;   // card whose fit is the minimum
    uint64_t cache_bytes_per_card = 0;
    uint64_t slot_table_bytes_per_card = 0;
    std::array<int64_t, kBudgetCards> available_bytes{};
    std::array<uint64_t, kBudgetCards> effective_headroom_bytes{}; // min(claim, measured available)
    std::array<uint64_t, kBudgetCards> draft_reserve_bytes{};      // reserve applied per card (0 = none)
    std::array<SlotsFit, kBudgetCards> fits{}; // each card at the final slots_granted
};

// Validates shapes and arithmetic range, and the 1.1 GiB floor while `enforce_draft_floor` is set. `why`
// gets the first failure.
bool validate_budget_input(const SlotBudgetInput& in, std::string* why);

// The one headroom reading. Fails when the available headroom cannot cover the draft reserve.
HeadroomBreakdown headroom_breakdown(const SlotBudgetInput& in);

// Exact cache byte cost at `slots` slots per layer, plus the allocation alignment slack in `*align_pad`.
uint64_t cache_bytes_for_slots(const SlotBudgetInput& in, uint32_t slots, uint64_t* align_pad = nullptr);
uint64_t slot_table_bytes_for(const SlotBudgetInput& in, uint32_t slots);

// Fit of `slots`; `max_base_bytes` (optional) gets the largest used_base that would still fit.
SlotsFit fit_slots(const SlotBudgetInput& in, uint32_t slots, uint64_t* max_base_bytes = nullptr);

// Full plan (requested slots, largest fit, grant).
SlotBudgetPlan plan_slot_budget(const SlotBudgetInput& in);

// Largest base usage that still fits `slots` (0 when even an empty base cannot fit).
uint64_t max_base_bytes_for_slots(const SlotBudgetInput& in, uint32_t slots, bool* fits = nullptr);

// Production site helper: consumes measured per-card capacity/used/pending, requires every card to fit,
// caps the requested slots and returns a zero grant when nothing can be allocated.
MultiCardGrant plan_measured_slots(const SlotBudgetInput& in,
                                   const std::array<CardArenaMeasurement, kBudgetCards>& measured,
                                   uint32_t cards);

// Same both-card decision, but with a measured reserve per card (0 = nothing to reserve on that card). The
// deployed load order uploads the drafter before this plan runs, so its bytes are already inside `used`;
// passing the measured footprint again would count them twice. Each reserve must fit that card's measured
// free space (otherwise the grant is 0); a value below kDraftReserveMinBytes requires
// `in.enforce_draft_floor == false`.
MultiCardGrant plan_measured_slots_per_card(const SlotBudgetInput& in,
                                            const std::array<CardArenaMeasurement, kBudgetCards>& measured,
                                            uint32_t cards,
                                            const std::array<uint64_t, kBudgetCards>& draft_reserve_per_card);

// Draft allocations are already included in measured.used_bytes. Charge them against the policy floor
// once, keep only the remainder free on each card, and retain the 2.70 GiB envelope and 21-slot cap.
// measured_draft_bytes must be arena deltas around draft-only allocations, excluding shared workspace.
MultiCardGrant plan_production_cache(const SlotBudgetInput& in,
                                     const std::array<CardArenaMeasurement, kBudgetCards>& measured,
                                     const std::array<uint64_t, kBudgetCards>& measured_draft_bytes,
                                     uint32_t cards);

} // namespace vegajet::moeslot
