// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/moe_slot_launch: dispatch/refill/eviction descriptors for the sealed DOT8 hot-expert kernels, the
// expert->slot table patch + upload contract, and a CPU LRU simulation that keeps the host cache, the
// per-card slot tables and the launch kernargs mutually consistent under a budget-granted slot count.
#include "core/moe_slot_launch.hpp"
#include "tests/test_common.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <vector>

using namespace vegajet::moeslot;

namespace {

SlotCacheView make_view(uint32_t slots = 44, uint32_t cards = 2) {
    SlotCacheView v;
    v.n_layers = 48;
    v.n_experts = 256;
    v.local_ff = 512;
    v.hidden = 3072;
    v.slots_per_layer = slots;
    v.cards = cards;
    v.layer.resize(v.n_layers);
    for (uint32_t l = 0; l < v.n_layers; ++l) {
        for (uint32_t c = 0; c < kMaxCards; ++c) {
            v.layer[l][c].iq2s_plane = 0x100000000ull + l * 0x100000ull + c * 0x10000ull;
            v.layer[l][c].w_plane = 0x200000000ull + l * 0x1000000ull + c * 0x1000ull;
            v.layer[l][c].s_plane = v.layer[l][c].w_plane + 3072u;
            v.layer[l][c].e2s_table = 0x300000000ull + l * 0x1000ull + c * 0x100ull;
        }
    }
    return v;
}

std::array<DispatchWorkspace, kMaxCards> make_ws() {
    std::array<DispatchWorkspace, kMaxCards> ws{};
    for (uint32_t c = 0; c < kMaxCards; ++c) {
        ws[c].x_q8 = 0x400000000ull + c * 0x1000ull;
        ws[c].x_ds = 0x410000000ull + c * 0x1000ull;
        ws[c].top_ids = 0x420000000ull + c * 0x1000ull;
        ws[c].mid_q8 = 0x430000000ull + c * 0x1000ull;
        ws[c].mid_d = 0x440000000ull + c * 0x1000ull;
    }
    return ws;
}

} // namespace

int main() {
    // ---- kernarg ABI pinned to the sealed kernels ----
    {
        VJ_CHECK(kDispatchKernargBytes == 88);
        VJ_CHECK(kRefillKernargBytes == 44);
        VJ_CHECK(sizeof(DispatchKernarg) == 88);
        VJ_CHECK(sizeof(RefillKernarg) == 48);
        VJ_CHECK(std::string(kDispatchKernel) == "vj_moe_gu_dot8_lru_kernel");
        VJ_CHECK(std::string(kRefillKernel) == "vj_repack_iq2s_to_dot8_gu_kernel");
        VJ_CHECK(kRouterTopK == 8 && kLruBlockThreads == 256 && kRepackBlockThreads == 128);
        // The cache slot count does NOT scale the dispatch grid: blockIdx.x encodes the routed top-8 index.
        // The slot count does not scale the grid: the kernel still launches 8 * (local_ff/32) WGs per token.
        VJ_CHECK(8u * (512u / 32u) == 128u);
    }

    // ---- dispatch ----
    {
        SlotCacheView v = make_view();
        std::string why;
        VJ_CHECK(validate_slot_cache_view(v, &why));
        const auto ws = make_ws();
        for (uint32_t m : {2u, 4u, 8u, 16u}) {
            DispatchPlan p = build_dispatch_plan(v, 7, m, ws);
            VJ_CHECK(p.ok);
            VJ_CHECK(std::string(p.launch.symbol) == kDispatchKernel);
            VJ_CHECK(p.launch.grid_x == 128 && p.launch.grid_y == m && p.launch.grid_z == 1);
            VJ_CHECK(p.launch.block_x == 256 && p.launch.block_y == 1 && p.launch.block_z == 1);
            VJ_CHECK(p.launch.kernarg_bytes == 88 && p.launch.barrier);
            VJ_CHECK(p.cards == 2);
            for (uint32_t c = 0; c < 2; ++c) {
                const DispatchKernarg& k = p.args[c];
                VJ_CHECK(k.wgu_iq2s == v.layer[7][c].iq2s_plane);
                VJ_CHECK(k.cache_w == v.layer[7][c].w_plane);
                VJ_CHECK(k.cache_sc == v.layer[7][c].s_plane);
                VJ_CHECK(k.exp_to_slot == v.layer[7][c].e2s_table);
                VJ_CHECK(k.x_q8 == ws[c].x_q8 && k.mid_d == ws[c].mid_d);
                VJ_CHECK(k.tokens == m && k.local_ff == 512);
                VJ_CHECK(k.w_stride == 3328 && k.s_stride == 3328); // embedded scale region
            }
        }
        // M=1 is the fused super-kernel path; M>16 is the grouped path. Neither is this kernel.
        VJ_CHECK(!build_dispatch_plan(v, 7, 1, ws).ok);
        VJ_CHECK(!build_dispatch_plan(v, 7, 17, ws).ok);
        VJ_CHECK(!build_dispatch_plan(v, 48, 4, ws).ok);
        auto bad_ws = ws;
        bad_ws[1].top_ids = 0;
        VJ_CHECK(!build_dispatch_plan(v, 7, 4, bad_ws).ok);
        SlotCacheView bad = v;
        bad.layer[5][1].s_plane = 0;
        VJ_CHECK(!build_dispatch_plan(bad, 5, 4, ws).ok);
    }

    // ---- refill: one launch per card, same expert/slot, 512 rows x 128 threads ----
    {
        SlotCacheView v = make_view();
        RefillPlan p = build_refill_plan(v, 3, 200, 43);
        VJ_CHECK(p.ok);
        VJ_CHECK(std::string(p.launch.symbol) == kRefillKernel);
        VJ_CHECK(p.launch.grid_x == 512 && p.launch.grid_y == 1 && p.launch.grid_z == 1);
        VJ_CHECK(p.launch.block_x == 128 && p.launch.block_y == 1 && p.launch.block_z == 1);
        VJ_CHECK(p.launch.kernarg_bytes == 44 && p.launch.barrier);
        VJ_CHECK(p.cards == 2);
        for (uint32_t c = 0; c < 2; ++c) {
            VJ_CHECK(p.args[c].wgu_iq2s == v.layer[3][c].iq2s_plane);
            VJ_CHECK(p.args[c].cache_w == v.layer[3][c].w_plane);
            VJ_CHECK(p.args[c].cache_sc == v.layer[3][c].s_plane);
            VJ_CHECK(p.args[c].expert == 200 && p.args[c].slot == 43);
            VJ_CHECK(p.args[c].local_ff == 512 && p.args[c].w_stride == 3328 && p.args[c].s_stride == 3328);
        }
        VJ_CHECK(!build_refill_plan(v, 3, 200, 44).ok); // slot outside the configured cache
        VJ_CHECK(!build_refill_plan(v, 3, 256, 1).ok);
        VJ_CHECK(!build_refill_plan(v, 48, 1, 1).ok);
        VJ_CHECK(!build_refill_plan(make_view(8), 3, 200, 8).ok);
        VJ_CHECK(build_refill_plan(make_view(8), 3, 200, 7).ok);
    }

    // ---- eviction/admission table patches and the post-refill upload contract ----
    {
        const uint32_t n_layers = 48, n_experts = 256;
        std::vector<int32_t> table(static_cast<size_t>(n_layers) * n_experts, -1);
        std::string why;
        VJ_CHECK(apply_slot_patch(table, n_layers, n_experts, make_admit_patch(3, 200, 43, n_experts), &why));
        VJ_CHECK(table[3 * n_experts + 200] == 43);
        VJ_CHECK(!apply_slot_patch(table, n_layers, n_experts, make_admit_patch(3, 200, 42, n_experts), &why));
        VJ_CHECK(!apply_slot_patch(table, n_layers, n_experts, make_admit_patch(3, 7, 43, n_experts), &why));
        VJ_CHECK(table[3 * n_experts + 7] == -1); // rejected patch must not write
        VJ_CHECK(!apply_slot_patch(table, n_layers, n_experts, make_evict_patch(3, 7, n_experts), &why));
        VJ_CHECK(apply_slot_patch(table, n_layers, n_experts, make_evict_patch(3, 200, n_experts), &why));
        VJ_CHECK(table[3 * n_experts + 200] == -1);

        SlotTablePatch stale = make_admit_patch(3, 9, 42, n_experts);
        stale.table_offset += 4;
        VJ_CHECK(!apply_slot_patch(table, n_layers, n_experts, stale, &why));
        VJ_CHECK(!apply_slot_patch(table, n_layers, 128, make_admit_patch(3, 9, 42, 128), &why)); // table size mismatch

        const SlotTableUpload u = make_table_upload(3, n_experts);
        VJ_CHECK(u.layer == 3 && u.table_offset == 3ull * 1024ull && u.bytes == 1024ull && u.after_refills);

        const SlotCacheView ev = make_view();
        const EvictionPlan ep = build_eviction_plan(ev, 3, 200);
        VJ_CHECK(ep.ok && ep.host_only && ep.upload.after_refills);
        VJ_CHECK(ep.patch.layer == 3 && ep.patch.expert == 200 && ep.patch.new_slot == -1);
        VJ_CHECK(ep.upload.table_offset == 3ull * 1024ull && ep.upload.bytes == 1024ull);
        VJ_CHECK(!build_eviction_plan(ev, 48, 1).ok);
        VJ_CHECK(!build_eviction_plan(ev, 0, 256).ok);
    }

    // ---- CPU LRU + budget + launch simulation ----
    {
        SlotBudgetInput in;
        in.used_base_bytes = 22ull * kGiB; // below the additive break-even
        SlotBudgetPlan budget = plan_slot_budget(in);
        VJ_CHECK(budget.ok && !budget.fits && budget.slots_granted == 21);

        SlotCacheView v = make_view(budget.slots_granted);
        const auto ws = make_ws();
        std::vector<int32_t> table(static_cast<size_t>(v.n_layers) * v.n_experts, -1);

        vegajet::moecache::CacheConfig cfg;
        cfg.n_layers = v.n_layers;
        cfg.n_experts = v.n_experts;
        cfg.slots_per_layer = v.slots_per_layer;
        cfg.max_admits_per_step = 48;
        cfg.max_admits_per_layer_step = 1;
        cfg.admit_after_misses = 2;
        vegajet::moecache::MoeLruCache cache(cfg);

        const uint32_t m_tokens = 4;
        uint64_t hits = 0, misses = 0, admissions = 0, refills = 0, uploads = 0;
        int32_t out[kRouterTopK * 4];
        int32_t experts[kRouterTopK * 4];
        for (uint32_t step = 0; step < 80u; ++step) {
            std::vector<vegajet::moecache::Admission> pending;
            for (uint32_t l = 0; l < v.n_layers; ++l) {
                // Deterministic skewed routing (different experts every step, top-8 per token).
                for (uint32_t t = 0; t < m_tokens; ++t)
                    for (uint32_t k = 0; k < kRouterTopK; ++k)
                        experts[t * kRouterTopK + k] = static_cast<int32_t>((l * 37u + step * 13u + t * 5u + k * 29u) % v.n_experts);
                cache.route(l, experts, m_tokens * kRouterTopK, out);
                for (uint32_t i = 0; i < m_tokens * kRouterTopK; ++i) {
                    if (out[i] < 0) {
                        ++misses;
                    } else {
                        ++hits;
                        VJ_CHECK(static_cast<uint32_t>(out[i]) < v.slots_per_layer);
                        VJ_CHECK(cache.slot_of(l, static_cast<uint32_t>(experts[i])) == out[i]);
                        VJ_CHECK(table[static_cast<size_t>(l) * v.n_experts + static_cast<uint32_t>(experts[i])] == out[i]);
                    }
                }
            }
            cache.end_step();
            const auto admits = cache.take_admissions();
            pending.insert(pending.end(), admits.begin(), admits.end());

            AdmissionPlan plan = build_admission_plan(v, pending);
            VJ_CHECK(plan.ok && plan.uploads_after_refills);
            VJ_CHECK(plan.uploads.size() <= pending.size());
            // (1) host table patches, (2) refills, (3) wait, (4) uploads.
            for (auto& op : plan.ops) {
                if (op.has_evict) VJ_CHECK(apply_slot_patch(table, v.n_layers, v.n_experts, op.evict));
                VJ_CHECK(apply_slot_patch(table, v.n_layers, v.n_experts, op.install));
                VJ_CHECK(op.refill.ok);
                VJ_CHECK(op.refill.cards == 2 && op.refill.launch.grid_x == 512);
                VJ_CHECK(op.refill.args[0].expert == op.admission.expert);
                VJ_CHECK(op.refill.args[0].slot == op.admission.slot);
                VJ_CHECK(op.refill.args[1].slot == op.admission.slot);
                VJ_CHECK(static_cast<uint32_t>(op.admission.slot) < v.slots_per_layer);
                refills += op.refill.cards;
            }
            for (const auto& a : plan.uploads) {
                VJ_CHECK(a.bytes == v.n_experts * 4u && a.after_refills);
                ++uploads;
            }
            for (const auto& a : pending) {
                VJ_CHECK(cache.commit(a));
                ++admissions;
            }
            cache.end_step();
            VJ_CHECK(cache.check_invariants());

            // Every cached expert must still be inside the granted slot budget after this step.
            for (uint32_t l = 0; l < v.n_layers; ++l)
                for (uint32_t e = 0; e < v.n_experts; ++e)
                    VJ_CHECK(cache.slot_of(l, e) < static_cast<int32_t>(v.slots_per_layer));
        }
        VJ_CHECK(hits > 0 && misses > 0); // cache warms up then serves hits
        VJ_CHECK(admissions > 0 && refills >= admissions && uploads > 0);
        VJ_CHECK(refills == admissions * 2u); // one launch per card
        for (uint32_t l = 0; l < v.n_layers; ++l) {
            const DispatchPlan d = build_dispatch_plan(v, l, m_tokens, ws);
            VJ_CHECK(d.ok && d.launch.grid_x == 128);
        }
        std::printf("  CPU LRU: %llu hits / %llu misses, %llu admissions, %llu refill launches, %llu table uploads\n",
                    static_cast<unsigned long long>(hits), static_cast<unsigned long long>(misses),
                    static_cast<unsigned long long>(admissions), static_cast<unsigned long long>(refills),
                    static_cast<unsigned long long>(uploads));

        // A reduced grant shrinks the legal slot ids and refill launches still validate against it.
        SlotBudgetInput tight = in;
        tight.used_base_bytes = kArenaCeilingBytes - 4ull * kGiB;
        SlotBudgetPlan tight_budget = plan_slot_budget(tight);
        VJ_CHECK(tight_budget.ok && tight_budget.slots_granted > 0 && tight_budget.slots_granted < 44);
        SlotCacheView tv = make_view(tight_budget.slots_granted);
        VJ_CHECK(!build_refill_plan(tv, 0, 1, tight_budget.slots_granted).ok);
        VJ_CHECK(build_refill_plan(tv, 0, 1, tight_budget.slots_granted - 1).ok);
    }

    std::printf("test_moe_slot_launch: dispatch/refill/evict descriptors, table contract, CPU LRU simulation OK\nPASS\n");
    return 0;
}
