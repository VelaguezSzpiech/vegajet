// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/moe_slot_budget.hpp"

#include "core/q4_dot8_repack.hpp"

#include <algorithm>
#include <limits>

namespace vegajet::moeslot {
namespace {

uint64_t mul_sat(uint64_t a, uint64_t b) {
    uint64_t r = 0;
    return __builtin_mul_overflow(a, b, &r) ? std::numeric_limits<uint64_t>::max() : r;
}

uint64_t align_up(uint64_t v, uint64_t a) { return a ? ((v + a - 1ull) / a) * a : v; }

uint64_t align_slack(uint64_t bytes, uint64_t align) {
    if (bytes == 0 || align == 0) return 0;
    return align_up(bytes, align) - bytes;
}

uint64_t sat_add(uint64_t a, uint64_t b) {
    const uint64_t r = a + b;
    return r < a ? std::numeric_limits<uint64_t>::max() : r;
}

std::string mib(uint64_t bytes) { return std::to_string(bytes >> 20) + " MiB"; }

// One layer's cache planes for `slots` experts, exactly as the engine allocates them:
// a single merged plane for [slots * local_ff] Gate+Up rows, and (optionally) single Down planes for
// [slots * hidden] rows. `pad` accumulates the per-plane alignment slack.
uint64_t layer_cache_bytes(const SlotBudgetInput& in, uint32_t slots, uint64_t* pad) {
    uint64_t total = 0;
    if (slots == 0) return 0;
    if (in.cache_gate_up) {
        const Dot8RowLayout gu =
            dot8_gu_iq2s_layout(static_cast<uint64_t>(slots) * in.local_ff, in.hidden);
        total += gu.vram_bytes();
        if (pad) {
            *pad += align_slack(gu.w_plane_bytes(), kPlaneAlignmentBytes);
            *pad += align_slack(gu.s_plane_bytes(), kScaleAlignmentBytes);
        }
    }
    if (in.cache_down) {
        const Dot8RowLayout dn = dot8_iq3_layout(static_cast<uint64_t>(slots) * in.hidden, in.local_ff);
        total += dn.vram_bytes();
        if (pad) {
            *pad += align_slack(dn.w_plane_bytes(), kPlaneAlignmentBytes);
            *pad += align_slack(dn.s_plane_bytes(), kScaleAlignmentBytes);
        }
    }
    return total;
}

void set_reason(std::string* why, const std::string& text) {
    if (why && why->empty()) *why = text;
}

} // namespace

bool validate_budget_input(const SlotBudgetInput& in, std::string* why) {
    if (why) why->clear();
    if (in.n_layers == 0) { set_reason(why, "n_layers must be > 0"); return false; }
    if (in.n_experts == 0) { set_reason(why, "n_experts must be > 0"); return false; }
    if (in.local_ff == 0 || in.local_ff % kDot8SuperElems != 0) {
        set_reason(why, "local_ff must be a positive multiple of 256");
        return false;
    }
    if (in.hidden == 0 || in.hidden % kDot8SuperElems != 0) {
        set_reason(why, "hidden must be a positive multiple of 256");
        return false;
    }
    if (in.slots_per_layer > in.n_experts) {
        set_reason(why, "slots_per_layer exceeds n_experts");
        return false;
    }
    if (!in.cache_gate_up && !in.cache_down) {
        set_reason(why, "no projection selected for the expert cache");
        return false;
    }
    if (in.arena_ceiling_bytes == 0) { set_reason(why, "arena ceiling must be > 0"); return false; }
    if (in.enforce_draft_floor && in.draft_reserve_bytes < kDraftReserveMinBytes) {
        set_reason(why, "draft reserve " + std::to_string(in.draft_reserve_bytes) + " B is below the " +
                            std::to_string(kDraftReserveMinBytes) + " B (1.1 GiB/card) floor");
        return false;
    }
    if (in.headroom_available_bytes < in.draft_reserve_bytes) {
        set_reason(why, "available headroom " + std::to_string(in.headroom_available_bytes) +
                            " B does not cover the " + std::to_string(in.draft_reserve_bytes) +
                            " B draft reserve; the 2.70 GiB figure is the total, not an extra reserve");
        return false;
    }
    if (in.slot_table_bytes_per_layer < in.n_experts * 4u) {
        set_reason(why, "slot table is smaller than n_experts * 4 bytes");
        return false;
    }
    return true;
}

HeadroomBreakdown headroom_breakdown(const SlotBudgetInput& in) {
    HeadroomBreakdown h;
    h.available_bytes = in.headroom_available_bytes;
    h.draft_bytes = in.draft_reserve_bytes;
    if (in.headroom_available_bytes < in.draft_reserve_bytes) {
        h.why = "available headroom " + mib(in.headroom_available_bytes) + " does not cover the " +
                mib(in.draft_reserve_bytes) + " draft reserve";
        return h;
    }
    h.cache_usable_bytes = in.headroom_available_bytes - in.draft_reserve_bytes;
    h.ok = true;
    return h;
}

uint64_t cache_bytes_for_slots(const SlotBudgetInput& in, uint32_t slots, uint64_t* align_pad) {
    if (align_pad) *align_pad = 0;
    uint64_t pad = 0;
    const uint64_t per_layer = layer_cache_bytes(in, slots, &pad);
    if (align_pad) *align_pad = mul_sat(pad, in.n_layers);
    return mul_sat(per_layer, in.n_layers);
}

uint64_t slot_table_bytes_for(const SlotBudgetInput& in, uint32_t slots) {
    if (slots == 0) return 0;
    return mul_sat(align_up(in.slot_table_bytes_per_layer, kTableAlignmentBytes), in.n_layers);
}

SlotsFit fit_slots(const SlotBudgetInput& in, uint32_t slots, uint64_t* max_base_bytes) {
    SlotsFit f;
    f.slots = slots;
    std::string why;
    if (max_base_bytes) *max_base_bytes = 0;
    if (!validate_budget_input(in, &why)) {
        f.why = why;
        return f;
    }
    const HeadroomBreakdown headroom = headroom_breakdown(in);
    f.cache_bytes = cache_bytes_for_slots(in, slots, nullptr);
    {
        uint64_t pad = 0;
        (void)cache_bytes_for_slots(in, slots, &pad);
        const uint64_t tables = slot_table_bytes_for(in, slots);
        f.overhead_bytes = sat_add(pad, tables);
    }
    f.reserve_bytes = headroom.draft_bytes; // kept free inside the ceiling; NOT added headroom
    f.total_bytes = sat_add(sat_add(sat_add(in.used_base_bytes, in.other_fixed_bytes),
                                    sat_add(f.overhead_bytes, f.cache_bytes)),
                            f.reserve_bytes);
    if (!headroom.ok) {
        f.why = headroom.why;
        return f;
    }
    if (f.overhead_bytes > headroom.cache_usable_bytes ||
        f.cache_bytes > headroom.cache_usable_bytes - f.overhead_bytes) {
        // The parenthetical must name the reserve that was actually applied: the frozen 1.1 GiB policy, a
        // measured per-card reserve, or none at all (plan_measured_slots_with_draft in Marginal mode).
        const std::string applied =
            in.enforce_draft_floor
                ? std::string(" (2.70 GiB total minus the 1.1 GiB draft reserve)")
                : (in.draft_reserve_bytes == 0
                       ? std::string(" (2.70 GiB total, no per-card draft reserve)")
                       : " (2.70 GiB total minus the " + mib(in.draft_reserve_bytes) + " measured draft reserve)");
        f.why = "cache + tables " + mib(sat_add(f.cache_bytes, f.overhead_bytes)) +
                " exceed the usable headroom " + mib(headroom.cache_usable_bytes) + applied;
        return f;
    }
    if (f.total_bytes <= in.arena_ceiling_bytes) {
        f.ok = true;
        f.remaining_bytes = static_cast<int64_t>(in.arena_ceiling_bytes - f.total_bytes);
    } else {
        const uint64_t over = f.total_bytes - in.arena_ceiling_bytes;
        f.remaining_bytes = over > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())
                                ? std::numeric_limits<int64_t>::min()
                                : -static_cast<int64_t>(over);
        f.why = "over the " + mib(in.arena_ceiling_bytes) + " ceiling by " + mib(over);
    }
    const uint64_t fixed = sat_add(sat_add(in.other_fixed_bytes, f.overhead_bytes),
                                   sat_add(f.cache_bytes, f.reserve_bytes));
    if (max_base_bytes)
        *max_base_bytes = fixed < in.arena_ceiling_bytes ? in.arena_ceiling_bytes - fixed : 0;
    return f;
}

uint64_t max_base_bytes_for_slots(const SlotBudgetInput& in, uint32_t slots, bool* fits) {
    uint64_t max_base = 0;
    const SlotsFit f = fit_slots(in, slots, &max_base);
    if (fits) *fits = f.ok && in.used_base_bytes <= max_base;
    return max_base;
}

SlotBudgetPlan plan_slot_budget(const SlotBudgetInput& in) {
    SlotBudgetPlan p;
    if (!validate_budget_input(in, &p.why)) return p;
    p.ok = true;
    p.used_base_bytes = in.used_base_bytes;
    p.other_fixed_bytes = in.other_fixed_bytes;
    p.ceiling_bytes = in.arena_ceiling_bytes;
    p.headroom = headroom_breakdown(in);

    const Dot8RowLayout gu = dot8_gu_iq2s_layout(in.local_ff, in.hidden);
    p.projections.gate_up_layer = in.cache_gate_up ? gu.vram_bytes() : 0;
    const Dot8RowLayout dn = dot8_iq3_layout(in.hidden, in.local_ff);
    p.projections.down_layer = in.cache_down ? dn.vram_bytes() : 0;
    p.projections.gate_up_per_slot = mul_sat(p.projections.gate_up_layer, in.n_layers);
    p.projections.down_per_slot = mul_sat(p.projections.down_layer, in.n_layers);

    p.slot_table_bytes = slot_table_bytes_for(in, in.slots_per_layer);
    (void)cache_bytes_for_slots(in, in.slots_per_layer, &p.alignment_padding_bytes);
    p.fixed_overhead_bytes = sat_add(p.alignment_padding_bytes, p.slot_table_bytes);

    p.requested = fit_slots(in, in.slots_per_layer, nullptr);

    // Cache and table bytes are non-decreasing in the slot count, so the fit is a prefix of [0, n_experts].
    uint32_t lo = 0, hi = in.n_experts, best = 0;
    while (lo <= hi) {
        const uint32_t mid = lo + (hi - lo) / 2u;
        if (fit_slots(in, mid, nullptr).ok) {
            best = mid;
            if (mid == std::numeric_limits<uint32_t>::max()) break;
            lo = mid + 1u;
        } else {
            if (mid == 0) break;
            hi = mid - 1u;
        }
    }
    p.max_slots_fit = best;
    p.slots_granted = in.slots_per_layer < best ? in.slots_per_layer : best;
    p.fits = p.requested.ok;
    if (!p.requested.ok)
        p.why = p.requested.why + "; grant " + std::to_string(p.slots_granted) + " of " +
                std::to_string(in.slots_per_layer) + " slots";
    return p;
}

int64_t measured_available_bytes(const CardArenaMeasurement& m) {
    const uint64_t used_and_pending = sat_add(m.used_bytes, m.pending_bytes);
    if (used_and_pending > m.capacity_bytes) {
        const uint64_t over = used_and_pending - m.capacity_bytes;
        return over > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())
                   ? std::numeric_limits<int64_t>::min()
                   : -static_cast<int64_t>(over);
    }
    const uint64_t available = m.capacity_bytes - used_and_pending;
    return available > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())
               ? std::numeric_limits<int64_t>::max()
               : static_cast<int64_t>(available);
}

MultiCardGrant plan_measured_slots_per_card(const SlotBudgetInput& in,
                                            const std::array<CardArenaMeasurement, kBudgetCards>& measured,
                                            uint32_t cards,
                                            const std::array<uint64_t, kBudgetCards>& draft_reserve_per_card) {
    MultiCardGrant g;
    std::string why;
    if (!validate_budget_input(in, &why)) {
        g.why = why;
        return g;
    }
    if (cards == 0 || cards > kBudgetCards) {
        g.why = "cards must be in [1, " + std::to_string(kBudgetCards) + "]";
        return g;
    }
    // The 1.1 GiB floor is a policy default. A measured per-card reserve may be smaller (or zero) only when
    // the caller says so explicitly; the default path below always honours the floor.
    if (in.enforce_draft_floor) {
        for (uint32_t c = 0; c < cards; ++c) {
            if (draft_reserve_per_card[c] < kDraftReserveMinBytes) {
                g.why = "card " + std::to_string(c) + " draft reserve " + std::to_string(draft_reserve_per_card[c]) +
                        " B is below the " + std::to_string(kDraftReserveMinBytes) +
                        " B (1.1 GiB/card) floor; set enforce_draft_floor = false for a measured reserve";
                return g;
            }
        }
    }
    g.requested_slots = in.slots_per_layer;
    uint32_t min_grant = in.slots_per_layer;
    std::string limiting_reason;
    std::array<SlotBudgetInput, kBudgetCards> card_in{};
    std::array<bool, kBudgetCards> card_usable{};
    for (uint32_t c = 0; c < cards; ++c) {
        const CardArenaMeasurement& m = measured[c];
        if (m.capacity_bytes == 0) {
            g.why = "card " + std::to_string(c) + " reports zero arena capacity";
            return g;
        }
        const int64_t available = measured_available_bytes(m);
        g.available_bytes[c] = available;
        g.draft_reserve_bytes[c] = draft_reserve_per_card[c];
        if (available <= 0) {
            g.effective_headroom_bytes[c] = 0;
            if (min_grant != 0) {
                min_grant = 0;
                g.limiting_card = c;
                limiting_reason = "card " + std::to_string(c) + " has no free arena (" +
                                  std::to_string(available) + " B)";
            }
            continue;
        }
        // The plan may never spend more than the 2.70 GiB claim, even when the measured arena has more.
        // When it has less, the measured free space is the cap.
        const uint64_t effective =
            std::min<uint64_t>(in.headroom_available_bytes, static_cast<uint64_t>(available));
        g.effective_headroom_bytes[c] = effective;
        if (effective < draft_reserve_per_card[c]) {
            if (min_grant != 0) {
                min_grant = 0;
                g.limiting_card = c;
                limiting_reason = "card " + std::to_string(c) + " has only " + std::to_string(effective) +
                                  " B free, below the " + std::to_string(draft_reserve_per_card[c]) +
                                  " B draft reserve";
            }
            continue;
        }
        SlotBudgetInput ci = in;
        ci.arena_ceiling_bytes = m.capacity_bytes; // the measured arena is the authority, not the constant
        ci.used_base_bytes = sat_add(m.used_bytes, m.pending_bytes);
        ci.headroom_available_bytes = effective;
        ci.draft_reserve_bytes = draft_reserve_per_card[c];
        card_in[c] = ci;
        card_usable[c] = true;
        const SlotBudgetPlan plan = plan_slot_budget(ci);
        if (!plan.ok) {
            g.why = "card " + std::to_string(c) + ": " + plan.why;
            return g;
        }
        if (plan.slots_granted < min_grant) {
            min_grant = plan.slots_granted;
            g.limiting_card = c;
            limiting_reason = plan.why;
        }
    }
    g.slots_granted = min_grant;
    for (uint32_t c = 0; c < cards; ++c) {
        if (!card_usable[c]) {
            g.fits[c].why = "card " + std::to_string(c) +
                            " cannot keep the draft reserve inside its measured free space";
            continue;
        }
        g.fits[c] = fit_slots(card_in[c], g.slots_granted, nullptr);
        if (!g.fits[c].ok && g.slots_granted > 0) {
            g.why = "card " + std::to_string(c) + " cannot fit the granted " +
                    std::to_string(g.slots_granted) + " slots: " + g.fits[c].why;
            g.slots_granted = 0;
            return g;
        }
    }
    g.cache_bytes_per_card = cache_bytes_for_slots(in, g.slots_granted, nullptr);
    g.slot_table_bytes_per_card = slot_table_bytes_for(in, g.slots_granted);
    g.ok = true;
    if (g.slots_granted == 0) {
        g.why = "no slots fit on both cards" + (limiting_reason.empty() ? "" : ": " + limiting_reason);
    } else if (g.slots_granted < g.requested_slots) {
        g.why = "grant " + std::to_string(g.slots_granted) + " of " + std::to_string(g.requested_slots) +
                " slots; card " + std::to_string(g.limiting_card) + " is limiting" +
                (limiting_reason.empty() ? "" : " (" + limiting_reason + ")");
    }
    return g;
}

MultiCardGrant plan_measured_slots(const SlotBudgetInput& in,
                                   const std::array<CardArenaMeasurement, kBudgetCards>& measured,
                                   uint32_t cards) {
    std::array<uint64_t, kBudgetCards> reserves{};
    for (uint32_t c = 0; c < kBudgetCards; ++c) reserves[c] = in.draft_reserve_bytes;
    return plan_measured_slots_per_card(in, measured, cards, reserves);
}

MultiCardGrant plan_production_cache(const SlotBudgetInput& in,
                                     const std::array<CardArenaMeasurement, kBudgetCards>& measured,
                                     const std::array<uint64_t, kBudgetCards>& measured_draft_bytes,
                                     uint32_t cards) {
    MultiCardGrant invalid;
    if (!validate_budget_input(in, &invalid.why)) return invalid;
    if (cards == 0u || cards > kBudgetCards) {
        invalid.why = "cards must be in [1, " + std::to_string(kBudgetCards) + "]";
        return invalid;
    }
    std::array<uint64_t, kBudgetCards> reserves{};
    auto bounded = measured;
    const uint64_t floor = std::max(in.draft_reserve_bytes, kDraftReserveMinBytes);
    for (uint32_t c = 0; c < cards; ++c) {
        if (measured_draft_bytes[c] > measured[c].used_bytes) {
            invalid.why = "card " + std::to_string(c) + " draft bytes are not included in measured used bytes";
            return invalid;
        }
        reserves[c] = floor > measured_draft_bytes[c] ? floor - measured_draft_bytes[c] : 0u;
        bounded[c].capacity_bytes = std::min(measured[c].capacity_bytes,
                                            std::min(in.arena_ceiling_bytes, kArenaCeilingBytes));
    }
    SlotBudgetInput production = in;
    production.enforce_draft_floor = false; // only the unconsumed part of the floor remains free
    production.slots_per_layer = std::min(in.slots_per_layer, kProductionMaxSlots);
    production.headroom_available_bytes = std::min(in.headroom_available_bytes, kHeadroomAvailableBytes);
    MultiCardGrant grant = plan_measured_slots_per_card(production, bounded, cards, reserves);
    grant.requested_slots = in.slots_per_layer;
    if (grant.ok && in.slots_per_layer > kProductionMaxSlots) {
        if (!grant.why.empty()) grant.why += "; ";
        grant.why += "conservative production cap " + std::to_string(kProductionMaxSlots) + " slots";
    }
    return grant;
}

} // namespace vegajet::moeslot
