// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/moe_slot_launch.hpp"

#include "core/q4_dot8_repack.hpp"

#include <algorithm>
#include <limits>

namespace vegajet::moeslot {
namespace {

void set_reason(std::string* why, const std::string& text) {
    if (why && why->empty()) *why = text;
}

bool gu_layout_for(const SlotCacheView& view, Dot8RowLayout* lay, std::string* why) {
    if (view.local_ff == 0 || view.local_ff % kDot8SuperElems != 0) {
        set_reason(why, "local_ff must be a positive multiple of 256");
        return false;
    }
    if (view.hidden == 0 || view.hidden % kDot8SuperElems != 0) {
        set_reason(why, "hidden must be a positive multiple of 256");
        return false;
    }
    *lay = dot8_gu_iq2s_layout(view.local_ff, view.hidden);
    return true;
}

bool check_layer_addrs(const SlotCacheView& view, uint32_t layer, uint32_t card, std::string* why) {
    const CardCacheAddr& a = view.layer[layer][card];
    if (a.iq2s_plane == 0) {
        set_reason(why, "card " + std::to_string(card) + " layer " + std::to_string(layer) +
                            ": base IQ2_S plane address is null");
        return false;
    }
    if (a.w_plane == 0) {
        set_reason(why, "card " + std::to_string(card) + " layer " + std::to_string(layer) +
                            ": DOT8 weight plane address is null");
        return false;
    }
    if (a.s_plane == 0) {
        set_reason(why, "card " + std::to_string(card) + " layer " + std::to_string(layer) +
                            ": DOT8 scale region address is null");
        return false;
    }
    if (a.e2s_table == 0) {
        set_reason(why, "card " + std::to_string(card) + " layer " + std::to_string(layer) +
                            ": expert->slot table address is null");
        return false;
    }
    return true;
}

} // namespace

bool validate_slot_cache_view(const SlotCacheView& view, std::string* why) {
    if (why) why->clear();
    if (view.n_layers == 0) { set_reason(why, "n_layers must be > 0"); return false; }
    if (view.n_experts == 0) { set_reason(why, "n_experts must be > 0"); return false; }
    if (view.local_ff == 0 || view.local_ff % 32u != 0) {
        set_reason(why, "local_ff must be a positive multiple of 32");
        return false;
    }
    if (view.hidden == 0 || view.hidden % kDot8SuperElems != 0) {
        set_reason(why, "hidden must be a positive multiple of 256");
        return false;
    }
    if (view.slots_per_layer == 0 || view.slots_per_layer > view.n_experts) {
        set_reason(why, "slots_per_layer must be in [1, n_experts]");
        return false;
    }
    if (view.cards == 0 || view.cards > kMaxCards) {
        set_reason(why, "cards must be in [1, " + std::to_string(kMaxCards) + "]");
        return false;
    }
    if (view.layer.size() != view.n_layers) {
        set_reason(why, "layer address table has " + std::to_string(view.layer.size()) + " entries for " +
                            std::to_string(view.n_layers) + " layers");
        return false;
    }
    for (uint32_t l = 0; l < view.n_layers; ++l)
        for (uint32_t c = 0; c < view.cards; ++c)
            if (!check_layer_addrs(view, l, c, why)) return false;
    return true;
}

DispatchPlan build_dispatch_plan(const SlotCacheView& view, uint32_t layer, uint32_t tokens,
                                 const std::array<DispatchWorkspace, kMaxCards>& ws) {
    DispatchPlan p;
    std::string why;
    if (!validate_slot_cache_view(view, &why)) { p.why = why; return p; }
    if (layer >= view.n_layers) {
        p.why = "layer " + std::to_string(layer) + " out of range";
        return p;
    }
    if (tokens < kLruMinTokens || tokens > kLruMaxTokens) {
        // M=1 uses the fused super-kernel path; larger batches use the grouped kernels. Neither is this kernel.
        p.why = "tokens " + std::to_string(tokens) + " outside the LRU kernel range [" +
                std::to_string(kLruMinTokens) + ", " + std::to_string(kLruMaxTokens) + "]";
        return p;
    }
    Dot8RowLayout lay{};
    if (!gu_layout_for(view, &lay, &why)) { p.why = why; return p; }

    p.launch.symbol = kDispatchKernel;
    p.launch.grid_x = kRouterTopK * (view.local_ff / 32u); // one block per (routed slot, 32 rows)
    p.launch.grid_y = tokens;
    p.launch.grid_z = 1;
    p.launch.block_x = kLruBlockThreads;
    p.launch.block_y = 1;
    p.launch.block_z = 1;
    p.launch.kernarg_bytes = kDispatchKernargBytes;
    p.launch.barrier = true;
    p.cards = view.cards;
    for (uint32_t c = 0; c < view.cards; ++c) {
        const CardCacheAddr& a = view.layer[layer][c];
        const DispatchWorkspace& w = ws[c];
        if (w.x_q8 == 0 || w.x_ds == 0 || w.top_ids == 0 || w.mid_q8 == 0 || w.mid_d == 0) {
            p.why = "card " + std::to_string(c) + ": dispatch workspace address is null";
            return p;
        }
        DispatchKernarg& k = p.args[c];
        k.wgu_iq2s = a.iq2s_plane;
        k.cache_w = a.w_plane;
        k.cache_sc = a.s_plane;
        k.exp_to_slot = a.e2s_table;
        k.x_q8 = w.x_q8;
        k.x_ds = w.x_ds;
        k.top_ids = w.top_ids;
        k.mid_q8 = w.mid_q8;
        k.mid_d = w.mid_d;
        k.tokens = tokens;
        k.local_ff = view.local_ff;
        k.w_stride = lay.w_row_stride_bytes;
        k.s_stride = lay.s_row_stride_bytes;
    }
    p.launch.ok = true;
    p.ok = true;
    return p;
}

RefillPlan build_refill_plan(const SlotCacheView& view, uint32_t layer, uint32_t expert, uint32_t slot) {
    RefillPlan p;
    std::string why;
    if (!validate_slot_cache_view(view, &why)) { p.why = why; return p; }
    if (layer >= view.n_layers) {
        p.why = "layer " + std::to_string(layer) + " out of range";
        return p;
    }
    if (expert >= view.n_experts) {
        p.why = "expert " + std::to_string(expert) + " out of range";
        return p;
    }
    if (slot >= view.slots_per_layer) {
        p.why = "slot " + std::to_string(slot) + " outside [0, " + std::to_string(view.slots_per_layer) + ")";
        return p;
    }
    Dot8RowLayout lay{};
    if (!gu_layout_for(view, &lay, &why)) { p.why = why; return p; }

    p.launch.symbol = kRefillKernel;
    p.launch.grid_x = view.local_ff; // one workgroup per DOT8 row of the slot
    p.launch.grid_y = 1;
    p.launch.grid_z = 1;
    p.launch.block_x = kRepackBlockThreads;
    p.launch.block_y = 1;
    p.launch.block_z = 1;
    p.launch.kernarg_bytes = kRefillKernargBytes;
    p.launch.barrier = true;
    p.cards = view.cards;
    for (uint32_t c = 0; c < view.cards; ++c) {
        const CardCacheAddr& a = view.layer[layer][c];
        RefillKernarg& k = p.args[c];
        k.wgu_iq2s = a.iq2s_plane;
        k.cache_w = a.w_plane;
        k.cache_sc = a.s_plane;
        k.expert = expert;
        k.slot = slot;
        k.local_ff = view.local_ff;
        k.w_stride = lay.w_row_stride_bytes;
        k.s_stride = lay.s_row_stride_bytes;
    }
    p.launch.ok = true;
    p.ok = true;
    return p;
}

SlotTablePatch make_evict_patch(uint32_t layer, uint32_t expert, uint32_t n_experts) {
    SlotTablePatch p;
    p.layer = layer;
    p.expert = expert;
    p.new_slot = -1;
    p.table_offset = static_cast<uint64_t>(layer) * n_experts * 4u;
    return p;
}

SlotTablePatch make_admit_patch(uint32_t layer, uint32_t expert, uint32_t slot, uint32_t n_experts) {
    SlotTablePatch p;
    p.layer = layer;
    p.expert = expert;
    p.new_slot = static_cast<int32_t>(slot);
    p.table_offset = static_cast<uint64_t>(layer) * n_experts * 4u;
    return p;
}

EvictionPlan build_eviction_plan(const SlotCacheView& view, uint32_t layer, uint32_t expert) {
    EvictionPlan p;
    std::string why;
    if (!validate_slot_cache_view(view, &why)) { p.why = why; return p; }
    if (layer >= view.n_layers) { p.why = "layer out of range"; return p; }
    if (expert >= view.n_experts) { p.why = "expert out of range"; return p; }
    p.patch = make_evict_patch(layer, expert, view.n_experts);
    p.upload = make_table_upload(layer, view.n_experts);
    p.ok = true;
    return p;
}

SlotTableUpload make_table_upload(uint32_t layer, uint32_t n_experts) {
    SlotTableUpload u;
    u.layer = layer;
    u.table_offset = static_cast<uint64_t>(layer) * n_experts * 4u;
    u.bytes = static_cast<uint64_t>(n_experts) * 4u;
    u.after_refills = true;
    return u;
}

bool apply_slot_patch(std::vector<int32_t>& table, uint32_t n_layers, uint32_t n_experts,
                      const SlotTablePatch& patch, std::string* why) {
    if (why) why->clear();
    if (patch.layer >= n_layers) { set_reason(why, "patch layer out of range"); return false; }
    if (patch.expert >= n_experts) { set_reason(why, "patch expert out of range"); return false; }
    const uint64_t expected_off = static_cast<uint64_t>(patch.layer) * n_experts * 4u;
    if (patch.table_offset != expected_off) {
        set_reason(why, "patch table_offset does not match layer * n_experts * 4");
        return false;
    }
    if (table.size() != static_cast<size_t>(n_layers) * n_experts) {
        set_reason(why, "host table size does not match n_layers * n_experts");
        return false;
    }
    const size_t idx = static_cast<size_t>(patch.layer) * n_experts + patch.expert;
    if (patch.new_slot < 0) {
        if (table[idx] < 0) { set_reason(why, "evict of an expert that is not cached"); return false; }
        table[idx] = -1;
        return true;
    }
    if (static_cast<uint32_t>(patch.new_slot) >= n_experts) {
        set_reason(why, "admit slot out of range");
        return false;
    }
    if (table[idx] >= 0) { set_reason(why, "admit of an expert that already owns a slot"); return false; }
    const int32_t slot = patch.new_slot;
    const size_t base = static_cast<size_t>(patch.layer) * n_experts;
    for (uint32_t e = 0; e < n_experts; ++e)
        if (table[base + e] == slot) { set_reason(why, "admit slot is already mapped to another expert"); return false; }
    table[idx] = slot;
    return true;
}

AdmissionPlan build_admission_plan(const SlotCacheView& view, const std::vector<moecache::Admission>& admissions) {
    AdmissionPlan p;
    std::string why;
    if (!validate_slot_cache_view(view, &why)) { p.why = why; return p; }
    std::vector<uint32_t> touched;
    std::vector<AdmissionOp> ops;
    ops.reserve(admissions.size());
    for (const auto& a : admissions) {
        if (a.layer >= view.n_layers) { p.why = "admission layer out of range"; return p; }
        if (a.expert >= view.n_experts) { p.why = "admission expert out of range"; return p; }
        if (a.slot >= view.slots_per_layer) { p.why = "admission slot outside slots_per_layer"; return p; }
        AdmissionOp op;
        op.admission = a;
        if (a.evicted >= 0) {
            op.has_evict = true;
            op.evict = make_evict_patch(a.layer, static_cast<uint32_t>(a.evicted), view.n_experts);
        }
        op.install = make_admit_patch(a.layer, a.expert, a.slot, view.n_experts);
        op.refill = build_refill_plan(view, a.layer, a.expert, a.slot);
        if (!op.refill.ok) { p.why = "refill for layer " + std::to_string(a.layer) + " expert " +
                                     std::to_string(a.expert) + ": " + op.refill.why; return p; }
        if (std::find(touched.begin(), touched.end(), a.layer) == touched.end()) touched.push_back(a.layer);
        ops.push_back(std::move(op));
    }
    for (uint32_t layer : touched) p.uploads.push_back(make_table_upload(layer, view.n_experts));
    p.ops = std::move(ops);
    p.uploads_after_refills = true;
    p.ok = true;
    return p;
}

} // namespace vegajet::moeslot
