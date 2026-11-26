// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/moe_lru_sealed.hpp"

#include <algorithm>
#include <limits>

namespace vegajet::moeslot::sealed {
namespace {

void set_reason(std::string* why, const std::string& text) {
    if (why && why->empty()) *why = text;
}

bool valid_source(Source s) { return s == Source::iq2_s || s == Source::iq3_xxs; }

uint32_t block_bytes(Source s) { return s == Source::iq2_s ? 82u : 98u; }

bool valid_tables(const Tables& t) { return t.iq2 != 0 && t.iq3 != 0 && t.signs != 0; }

std::string s32(uint32_t v) { return std::to_string(v); }

} // namespace

uint32_t sealed_stride(uint32_t bytes) {
    if (bytes > std::numeric_limits<uint32_t>::max() - 255u) return 0;
    return (((bytes + 255u) / 256u) | 1u) * 256u;
}

CacheLayout cache_layout(uint32_t rows, uint32_t cols, std::string* why) {
    CacheLayout l;
    if (why) why->clear();
    if (rows == 0) { set_reason(why, "rows must be > 0"); return l; }
    if (cols == 0 || cols % 256u != 0) {
        set_reason(why, "cols must be a positive multiple of 256");
        return l;
    }
    l.rows = rows;
    l.cols = cols;
    l.blocks32 = cols / 32u;
    l.qs_row_bytes = cols / 2u;
    l.qs_row_stride = sealed_stride(l.qs_row_bytes);
    l.scale_row_bytes = l.blocks32 * 4u;
    l.scale_row_stride = sealed_stride(l.scale_row_bytes);
    if (l.qs_row_stride == 0 || l.scale_row_stride == 0) {
        set_reason(why, "cache row stride overflows uint32");
        return CacheLayout{};
    }
    return l;
}

uint64_t qs_slot_offset(const CacheLayout& lay, uint32_t slot) {
    return static_cast<uint64_t>(slot) * lay.qs_slot_bytes();
}

uint64_t scale_slot_offset(const CacheLayout& lay, uint32_t slot) {
    return static_cast<uint64_t>(slot) * lay.scale_slot_bytes();
}

uint64_t qs_row_offset(const CacheLayout& lay, uint32_t slot, uint32_t row) {
    return qs_slot_offset(lay, slot) + static_cast<uint64_t>(row) * lay.qs_row_stride;
}

uint64_t scale_row_offset(const CacheLayout& lay, uint32_t slot, uint32_t row) {
    return scale_slot_offset(lay, slot) + static_cast<uint64_t>(row) * lay.scale_row_stride;
}

uint64_t raw_expert_bytes(uint32_t rows, uint32_t cols, Source type, std::string* why) {
    if (why) why->clear();
    if (!valid_source(type)) { set_reason(why, "unknown source type"); return 0; }
    if (rows == 0) { set_reason(why, "rows must be > 0"); return 0; }
    if (cols == 0 || cols % 256u != 0) {
        set_reason(why, "cols must be a positive multiple of 256");
        return 0;
    }
    return static_cast<uint64_t>(rows) * (cols / 256u) * block_bytes(type);
}

RefillPlan build_refill_plan(const RefillRequest& req, uint32_t expert, uint32_t slot, uint32_t slots) {
    RefillPlan p;
    if (req.cards == 0 || req.cards > kMaxCards) {
        p.why = "cards must be in [1, " + s32(kMaxCards) + "]";
        return p;
    }
    if (!valid_source(req.source)) { p.why = "unknown source type"; return p; }
    if (slots == 0) { p.why = "slots must be > 0"; return p; }
    if (slot >= slots) {
        p.why = "slot " + s32(slot) + " outside [0, " + s32(slots) + ")";
        return p;
    }
    std::string why;
    p.layout = cache_layout(req.rows, req.cols, &why);
    if (!why.empty()) { p.why = why; return p; }
    const uint64_t raw_bytes = raw_expert_bytes(req.rows, req.cols, req.source, &why);
    if (!why.empty()) { p.why = why; return p; }
    if (!valid_tables(req.tables)) { p.why = "all three GGML table addresses must be non-null"; return p; }
    if (raw_bytes > std::numeric_limits<uint64_t>::max() / (expert + 1ull)) {
        p.why = "expert source offset overflows uint64";
        return p;
    }
    const uint64_t src_offset = static_cast<uint64_t>(expert) * raw_bytes;
    p.qs_slot_offset = qs_slot_offset(p.layout, slot);
    p.scale_slot_offset = scale_slot_offset(p.layout, slot);

    const uint64_t tiles64 = static_cast<uint64_t>(req.rows) * (req.cols / 32u);
    const uint64_t grid64 = (tiles64 + kRefillBlockThreads - 1u) / kRefillBlockThreads;
    if (grid64 == 0 || grid64 > std::numeric_limits<uint32_t>::max()) {
        p.why = "refill grid size out of range";
        return p;
    }
    p.launch.symbol = kRefillSymbol;
    p.launch.grid_x = static_cast<uint32_t>(grid64);
    p.launch.grid_y = 1;
    p.launch.grid_z = 1;
    p.launch.block_x = kRefillBlockThreads;
    p.launch.block_y = 1;
    p.launch.block_z = 1;
    p.launch.kernarg_bytes = kRefillKernargBytes;
    p.launch.barrier = true;
    p.cards = req.cards;
    for (uint32_t c = 0; c < req.cards; ++c) {
        const uint64_t raw_base = req.raw_plane[c];
        const uint64_t qs_base = req.qs_cache[c];
        const uint64_t ds_base = req.ds_cache[c];
        if (raw_base == 0 || qs_base == 0 || ds_base == 0) {
            p.why = "card " + s32(c) + ": raw/qs/ds address is null";
            return p;
        }
        if (raw_base > std::numeric_limits<uint64_t>::max() - src_offset ||
            qs_base > std::numeric_limits<uint64_t>::max() - p.qs_slot_offset ||
            ds_base > std::numeric_limits<uint64_t>::max() - p.scale_slot_offset) {
            p.why = "card " + s32(c) + ": slot address overflows uint64";
            return p;
        }
        RefillKernarg& k = p.args[c];
        k.src = raw_base + src_offset;
        k.qs = qs_base + p.qs_slot_offset;
        k.ds = ds_base + p.scale_slot_offset;
        k.rows = req.rows;
        k.cols = req.cols;
        k.type = static_cast<uint32_t>(req.source);
        k.tables = req.tables;
    }
    p.launch.ok = true;
    p.ok = true;
    return p;
}

HitPlan build_hit_plan(const HitRequest& req, HitVariant variant, uint32_t stagger,
                       bool allow_noncanonical) {
    HitPlan p;
    if (req.cards == 0 || req.cards > kMaxCards) {
        p.why = "cards must be in [1, " + s32(kMaxCards) + "]";
        return p;
    }
    if (!valid_source(req.source)) { p.why = "unknown source type"; return p; }
    if (req.jobs == 0) { p.why = "jobs must be > 0"; return p; }
    if (req.slot_count == 0) { p.why = "slot_count must be > 0"; return p; }
    if (stagger != 0 && !allow_noncanonical) {
        p.why = "stagger must be 0 for canonical FP accumulation order (pass allow_noncanonical to override)";
        return p;
    }
    std::string why;
    p.layout = cache_layout(req.rows, req.cols, &why);
    if (!why.empty()) { p.why = why; return p; }
    if (req.host_slots) {
        if (req.host_slots->size() != req.jobs) {
            p.why = "host_slots size does not match jobs";
            return p;
        }
        for (uint32_t s : *req.host_slots)
            if (s >= req.slot_count) {
                p.why = "host slot " + s32(s) + " outside slot_count " + s32(req.slot_count);
                return p;
            }
    }
    if (req.host_tokens) {
        if (req.host_tokens->size() != req.jobs) {
            p.why = "host_tokens size does not match jobs";
            return p;
        }
        if (req.token_count == 0) { p.why = "token_count must be > 0 with host_tokens"; return p; }
        for (uint32_t t : *req.host_tokens)
            if (t >= req.token_count) {
                p.why = "host token " + s32(t) + " outside token_count " + s32(req.token_count);
                return p;
            }
    }
    const uint64_t grid_x64 = (static_cast<uint64_t>(req.rows) + kHitRowsPerBlock - 1u) / kHitRowsPerBlock;
    if (grid_x64 == 0 || grid_x64 > std::numeric_limits<uint32_t>::max()) {
        p.why = "hit grid.x out of range";
        return p;
    }
    p.launch.symbol = variant == HitVariant::dot4 ? kHitDot4Symbol : kHitDot8Symbol;
    p.launch.grid_x = static_cast<uint32_t>(grid_x64);
    p.launch.grid_y = req.jobs;
    p.launch.grid_z = 1;
    p.launch.block_x = kHitBlockThreads;
    p.launch.block_y = 1;
    p.launch.block_z = 1;
    p.launch.kernarg_bytes = kHitKernargBytes;
    p.launch.barrier = true;
    p.cards = req.cards;
    for (uint32_t c = 0; c < req.cards; ++c) {
        if (req.qs_cache[c] == 0 || req.ds_cache[c] == 0 || req.x[c] == 0 || req.xd[c] == 0 ||
            req.slots[c] == 0 || req.tokens[c] == 0 || req.y[c] == 0) {
            p.why = "card " + s32(c) + ": hit argument address is null";
            return p;
        }
        HitKernarg& k = p.args[c];
        k.qs = req.qs_cache[c];
        k.ds = req.ds_cache[c];
        k.x = req.x[c];
        k.xd = req.xd[c];
        k.slots = req.slots[c];
        k.tokens = req.tokens[c];
        k.y = req.y[c];
        k.rows = req.rows;
        k.cols = req.cols;
        k.type = static_cast<uint32_t>(req.source);
        k.stagger = stagger;
    }
    p.launch.ok = true;
    p.ok = true;
    return p;
}

SlotLedger::SlotLedger(uint32_t layers, uint32_t slots) : layers_(layers), slots_(slots) {
    for (auto& l : layers_) l.slot.assign(slots_, Slot{});
}

uint64_t SlotLedger::generation(uint32_t layer) const {
    return layer < layers_.size() ? layers_[layer].generation : 0;
}

bool SlotLedger::pending(uint32_t layer, uint32_t slot) const {
    return layer < layers_.size() && slot < slots_ && layers_[layer].slot[slot].pending;
}

bool SlotLedger::ready(uint32_t layer, uint32_t slot) const {
    return layer < layers_.size() && slot < slots_ && layers_[layer].slot[slot].ready;
}

int32_t SlotLedger::expert(uint32_t layer, uint32_t slot) const {
    return (layer < layers_.size() && slot < slots_) ? layers_[layer].slot[slot].expert : -1;
}

uint32_t SlotLedger::pending_count() const {
    uint32_t n = 0;
    for (const auto& l : layers_)
        for (const auto& s : l.slot)
            if (s.pending) ++n;
    return n;
}

uint64_t SlotLedger::begin_refill(uint32_t layer, uint32_t expert, uint32_t slot, std::string* why) {
    if (why) why->clear();
    if (layer >= layers_.size()) { set_reason(why, "layer out of range"); return 0; }
    if (slot >= slots_) { set_reason(why, "slot out of range"); return 0; }
    if (expert > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
        set_reason(why, "expert does not fit int32");
        return 0;
    }
    Layer& L = layers_[layer];
    Slot& s = L.slot[slot];
    if (s.ready && s.expert == static_cast<int32_t>(expert)) {
        set_reason(why, "expert is already ready in this slot");
        return 0;
    }
    // A new binding (or a restart of a pending one) invalidates any earlier ticket and any snapshot.
    ++L.generation;
    s.expert = static_cast<int32_t>(expert);
    s.ticket = next_ticket_++;
    s.pending = true;
    s.ready = false;
    return s.ticket;
}

bool SlotLedger::complete_refill(uint64_t ticket, std::string* why) {
    if (why) why->clear();
    if (ticket == 0) { set_reason(why, "zero ticket"); return false; }
    for (auto& L : layers_) {
        for (auto& s : L.slot) {
            if (s.pending && s.ticket == ticket) {
                s.pending = false;
                s.ready = true;
                return true;
            }
        }
    }
    set_reason(why, "stale refill ticket (slot re-bound or evicted)");
    return false;
}

bool SlotLedger::evict(uint32_t layer, uint32_t slot, std::string* why) {
    if (why) why->clear();
    if (layer >= layers_.size()) { set_reason(why, "layer out of range"); return false; }
    if (slot >= slots_) { set_reason(why, "slot out of range"); return false; }
    Layer& L = layers_[layer];
    Slot& s = L.slot[slot];
    if (!s.pending && !s.ready && s.expert < 0) {
        set_reason(why, "slot is already empty");
        return false;
    }
    ++L.generation;
    s = Slot{};
    return true;
}

std::vector<uint32_t> SlotLedger::ready_slots(uint32_t layer) const {
    std::vector<uint32_t> out;
    if (layer >= layers_.size()) return out;
    for (uint32_t s = 0; s < slots_; ++s)
        if (layers_[layer].slot[s].ready) out.push_back(s);
    return out;
}

bool SlotLedger::snapshot(uint32_t layer, const std::vector<uint32_t>& slots, Snapshot* out,
                          std::string* why) const {
    if (why) why->clear();
    if (!out) { set_reason(why, "null snapshot output"); return false; }
    if (layer >= layers_.size()) { set_reason(why, "layer out of range"); return false; }
    for (uint32_t s : slots) {
        if (s >= slots_) { set_reason(why, "snapshot slot out of range"); return false; }
        const Slot& st = layers_[layer].slot[s];
        if (!st.ready || st.pending) {
            set_reason(why, "slot " + s32(s) + " is not ready (pending or empty)");
            return false;
        }
    }
    out->layer = layer;
    out->generation = layers_[layer].generation;
    out->slots = slots;
    return true;
}

bool SlotLedger::validate_snapshot(const Snapshot& snap, std::string* why) const {
    if (why) why->clear();
    if (snap.layer >= layers_.size()) { set_reason(why, "snapshot layer out of range"); return false; }
    const Layer& L = layers_[snap.layer];
    if (snap.generation != L.generation) {
        set_reason(why, "stale slot-table generation " + std::to_string(snap.generation) + " (current " +
                            std::to_string(L.generation) + ")");
        return false;
    }
    for (uint32_t s : snap.slots) {
        if (s >= slots_) { set_reason(why, "snapshot slot out of range"); return false; }
        const Slot& st = L.slot[s];
        if (!st.ready || st.pending) {
            set_reason(why, "slot " + s32(s) + " is not ready (pending or empty)");
            return false;
        }
    }
    return true;
}

HitPlan build_hit_plan_checked(const HitRequest& req, HitVariant variant, const SlotLedger& ledger,
                               const SlotLedger::Snapshot& snapshot, uint32_t stagger,
                               bool allow_noncanonical) {
    HitPlan p;
    std::string why;
    if (!ledger.validate_snapshot(snapshot, &why)) {
        p.why = "refusing stale/pending slot snapshot: " + why;
        return p;
    }
    p = build_hit_plan(req, variant, stagger, allow_noncanonical);
    if (p.ok) {
        if (!req.host_slots || req.host_slots->size() != snapshot.slots.size() ||
            req.jobs != snapshot.slots.size()) {
            p.ok = false;
            p.why = "snapshot slot count does not match the host slot list/jobs";
            return p;
        }
        for (size_t i = 0; i < snapshot.slots.size(); ++i)
            if ((*req.host_slots)[i] != snapshot.slots[i]) {
                p.ok = false;
                p.why = "host slot list does not match the validated snapshot";
                return p;
            }
    }
    return p;
}

} // namespace vegajet::moeslot::sealed
