// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Sealed cache path tests: the exact symbols in ops/kernels/vj_moe_lru_refill_dot8.hiph, their cache
// layout (linear code rows + separate Scale rows), the dispatch/refill descriptors, and the host
// generation/pending ledger that keeps a hit dispatch away from a slot being refilled or evicted.
//
// This is deliberately separate from core/moe_slot_launch.* (the ACTIVE production vj_moe_dot8.hip
// path with Dot8RowLayout and the in-kernel expert->slot table). The two cache payloads differ.
#include "core/moe_lru_sealed.hpp"
#include "core/moe_lru_cache.hpp"
#include "core/moe_slot_budget.hpp"
#include "core/q4_dot8_repack.hpp"
#include "ops/kernels/vj_moe_lru_refill_dot8.hiph"
#include "tests/test_common.hpp"

#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

using namespace vegajet::moeslot::sealed;

namespace {

RefillRequest refill_request() {
    RefillRequest r;
    r.rows = 512;
    r.cols = 3072;
    r.source = Source::iq2_s;
    r.tables = {0x10000000ull, 0x20000000ull, 0x30000000ull};
    r.cards = 2;
    r.raw_plane = {0x1000000ull, 0x2000000ull};
    r.qs_cache = {0x3000000ull, 0x4000000ull};
    r.ds_cache = {0x5000000ull, 0x6000000ull};
    return r;
}

HitRequest hit_request() {
    HitRequest r;
    r.rows = 512;
    r.cols = 3072;
    r.jobs = 8;
    r.slot_count = 44;
    r.source = Source::iq2_s;
    r.cards = 2;
    r.qs_cache = {0x3000000ull, 0x4000000ull};
    r.ds_cache = {0x5000000ull, 0x6000000ull};
    r.x = {0x7000000ull, 0x7100000ull};
    r.xd = {0x7200000ull, 0x7300000ull};
    r.slots = {0x7400000ull, 0x7500000ull};
    r.tokens = {0x7600000ull, 0x7700000ull};
    r.y = {0x7800000ull, 0x7900000ull};
    static const std::vector<uint32_t> slots{0, 1, 2, 3, 4, 5, 6, 7};
    static const std::vector<uint32_t> tokens{0, 1, 2, 3, 4, 5, 6, 7};
    r.host_slots = &slots;
    r.host_tokens = &tokens;
    r.token_count = 8;
    return r;
}

} // namespace

int main() {
    // ---- ABI pinned to the sealed header and its compiled metadata (TheRock 7.12, gfx906) ----
    {
        VJ_CHECK(sizeof(RefillKernarg) == 64 && kRefillKernargBytes == 64);
        VJ_CHECK(sizeof(HitKernarg) == 72 && kHitKernargBytes == 72);
        VJ_CHECK(sizeof(vj_refill::Scale) == 4);
        VJ_CHECK(std::string(kRefillSymbol) == "vj_moe_lru_refill_dot8");
        VJ_CHECK(std::string(kHitDot4Symbol) == "vj_moe_lru_hit_dot4");
        VJ_CHECK(std::string(kHitDot8Symbol) == "vj_moe_lru_hit_dot8");
        VJ_CHECK(offsetof(RefillKernarg, tables) == 40);
        VJ_CHECK(offsetof(HitKernarg, rows) == 56 && offsetof(HitKernarg, stagger) == 68);
        VJ_CHECK(kRefillBlockThreads == 256 && kHitBlockThreads == 256 && kHitRowsPerBlock == 4);
        VJ_CHECK(static_cast<uint32_t>(Source::iq2_s) == 0 && static_cast<uint32_t>(Source::iq3_xxs) == 1);
    }

    // ---- exact sealed cache layout: linear code rows, separate Scale rows, odd-256B stride ----
    {
        std::string why;
        const CacheLayout gu = cache_layout(512, 3072, &why);
        VJ_CHECK(why.empty());
        VJ_CHECK(gu.rows == 512 && gu.cols == 3072 && gu.blocks32 == 96);
        VJ_CHECK(gu.qs_row_bytes == 1536 && gu.qs_row_stride == 1792);
        VJ_CHECK(gu.scale_row_bytes == 384 && gu.scale_row_stride == 768);
        VJ_CHECK(gu.qs_slot_bytes() == 512ull * 1792ull);      // 917,504 B
        VJ_CHECK(gu.scale_slot_bytes() == 512ull * 768ull);    // 393,216 B
        VJ_CHECK(gu.slot_bytes() == 1310720ull);
        // Match the device header's own stride helper exactly, not a reimplementation.
        VJ_CHECK(gu.qs_row_stride == vj_refill::stride(gu.qs_row_bytes));
        VJ_CHECK(gu.scale_row_stride == vj_refill::stride(gu.scale_row_bytes));

        // The refill kernel writes rows from the slot base it is given; the hit kernels index slot*rows.
        VJ_CHECK(qs_slot_offset(gu, 3) == 3ull * 917504ull);
        VJ_CHECK(scale_slot_offset(gu, 3) == 3ull * 393216ull);
        VJ_CHECK(qs_row_offset(gu, 3, 511) == 3ull * 917504ull + 511ull * 1792ull);
        VJ_CHECK(scale_row_offset(gu, 3, 511) == 3ull * 393216ull + 511ull * 768ull);

        // Down projection shape.
        const CacheLayout dn = cache_layout(3072, 512, &why);
        VJ_CHECK(why.empty());
        VJ_CHECK(dn.blocks32 == 16);
        VJ_CHECK(dn.qs_row_bytes == 256 && dn.qs_row_stride == 256);
        VJ_CHECK(dn.scale_row_bytes == 64 && dn.scale_row_stride == 256);
        VJ_CHECK(dn.slot_bytes() == 3072ull * (256ull + 256ull));

        // Distinguish the active production layout (embedded scales, merged GU) from this sealed payload.
        const vegajet::Dot8RowLayout active = vegajet::dot8_gu_iq2s_layout(512, 3072);
        VJ_CHECK(active.w_row_stride_bytes == 3328);
        VJ_CHECK(active.s_row_stride_bytes == 3328);
        VJ_CHECK(active.vram_bytes() == 1703936ull);
        VJ_CHECK(active.vram_bytes() != gu.slot_bytes());
        VJ_CHECK(active.w_row_stride_bytes != gu.qs_row_stride);

        VJ_CHECK(cache_layout(0, 3072, &why).rows == 0 && why.find("rows") != std::string::npos);
        why.clear();
        VJ_CHECK(cache_layout(512, 300, &why).rows == 0 && why.find("multiple of 256") != std::string::npos);
        why.clear();
        VJ_CHECK(raw_expert_bytes(512, 3072, Source::iq2_s) == 503808ull);
        VJ_CHECK(raw_expert_bytes(3072, 512, Source::iq3_xxs) == 602112ull);
        VJ_CHECK(raw_expert_bytes(512, 3072, static_cast<Source>(7), &why) == 0 && !why.empty());
    }

    // ---- refill descriptors: source-expert offset, slot offset, grid and symbol ----
    {
        const RefillRequest req = refill_request();
        RefillPlan p = build_refill_plan(req, 7, 3, 44);
        VJ_CHECK(p.ok);
        VJ_CHECK(std::string(p.launch.symbol) == kRefillSymbol);
        VJ_CHECK(p.launch.grid_x == 192 && p.launch.grid_y == 1 && p.launch.grid_z == 1);
        VJ_CHECK(p.launch.block_x == 256 && p.launch.block_y == 1 && p.launch.block_z == 1);
        VJ_CHECK(p.launch.kernarg_bytes == 64 && p.launch.barrier);
        VJ_CHECK(p.cards == 2);
        VJ_CHECK(p.qs_slot_offset == 3ull * 917504ull);
        VJ_CHECK(p.scale_slot_offset == 3ull * 393216ull);
        for (uint32_t c = 0; c < 2; ++c) {
            const uint64_t raw = req.raw_plane[c] + 7ull * 503808ull;
            VJ_CHECK(p.args[c].src == raw);
            VJ_CHECK(p.args[c].qs == req.qs_cache[c] + 3ull * 917504ull);
            VJ_CHECK(p.args[c].ds == req.ds_cache[c] + 3ull * 393216ull);
            VJ_CHECK(p.args[c].rows == 512 && p.args[c].cols == 3072);
            VJ_CHECK(p.args[c].type == static_cast<uint32_t>(Source::iq2_s));
            VJ_CHECK(p.args[c].tables.iq2 == req.tables.iq2);
            VJ_CHECK(p.args[c].tables.iq3 == req.tables.iq3);
            VJ_CHECK(p.args[c].tables.signs == req.tables.signs);
        }

        // IQ3_XXS Down shape: 3,072 rows x 512 cols, 98 B per 256 weights in the source.
        RefillRequest dn = req;
        dn.rows = 3072;
        dn.cols = 512;
        dn.source = Source::iq3_xxs;
        RefillPlan pd = build_refill_plan(dn, 255, 0, 1);
        VJ_CHECK(pd.ok);
        VJ_CHECK(pd.launch.grid_x == 192);
        VJ_CHECK(pd.args[0].src == dn.raw_plane[0] + 255ull * 602112ull);
        VJ_CHECK(pd.args[0].qs == dn.qs_cache[0]);
        VJ_CHECK(pd.args[0].type == static_cast<uint32_t>(Source::iq3_xxs));
        VJ_CHECK(pd.layout.slot_bytes() == 3072ull * 512ull);

        VJ_CHECK(!build_refill_plan(req, 0, 44, 44).ok); // slot outside the grant
        VJ_CHECK(!build_refill_plan(req, 0, 0, 0).ok);
        RefillRequest bad = req;
        bad.raw_plane[1] = 0;
        VJ_CHECK(!build_refill_plan(bad, 0, 0, 1).ok);
        bad = req;
        bad.tables.iq3 = 0;
        VJ_CHECK(!build_refill_plan(bad, 0, 0, 1).ok);
        bad = req;
        bad.cards = 3;
        VJ_CHECK(!build_refill_plan(bad, 0, 0, 1).ok);
    }

    // ---- hit descriptors: jobs, grid, canonical stagger, cache base (not slot base) ----
    {
        const HitRequest req = hit_request();
        for (HitVariant variant : {HitVariant::dot4, HitVariant::dot8}) {
            HitPlan p = build_hit_plan(req, variant);
            VJ_CHECK(p.ok);
            VJ_CHECK(std::string(p.launch.symbol) ==
                     (variant == HitVariant::dot4 ? kHitDot4Symbol : kHitDot8Symbol));
            VJ_CHECK(p.launch.grid_x == 128 && p.launch.grid_y == 8 && p.launch.grid_z == 1);
            VJ_CHECK(p.launch.block_x == 256 && p.launch.block_y == 1 && p.launch.block_z == 1);
            VJ_CHECK(p.launch.kernarg_bytes == 72 && p.launch.barrier);
            VJ_CHECK(p.cards == 2);
            VJ_CHECK(p.layout.slot_bytes() == 1310720ull);
            for (uint32_t c = 0; c < 2; ++c) {
                VJ_CHECK(p.args[c].qs == req.qs_cache[c]); // hint: cache base, slot chosen per job
                VJ_CHECK(p.args[c].ds == req.ds_cache[c]);
                VJ_CHECK(p.args[c].x == req.x[c] && p.args[c].xd == req.xd[c]);
                VJ_CHECK(p.args[c].slots == req.slots[c] && p.args[c].tokens == req.tokens[c]);
                VJ_CHECK(p.args[c].y == req.y[c]);
                VJ_CHECK(p.args[c].rows == 512 && p.args[c].cols == 3072);
                VJ_CHECK(p.args[c].type == static_cast<uint32_t>(Source::iq2_s));
                VJ_CHECK(p.args[c].stagger == 0);
            }
        }
        VJ_CHECK(!build_hit_plan(req, HitVariant::dot4, 1).ok); // noncanonical order rejected by default
        HitPlan staggered = build_hit_plan(req, HitVariant::dot4, 1, true);
        VJ_CHECK(staggered.ok && staggered.args[0].stagger == 1);

        HitRequest dn = req;
        dn.rows = 3072;
        dn.cols = 512;
        dn.source = Source::iq3_xxs;
        HitPlan pd = build_hit_plan(dn, HitVariant::dot8);
        VJ_CHECK(pd.ok && pd.launch.grid_x == 768 && pd.launch.grid_y == 8);

        HitRequest bad = req;
        bad.host_slots = nullptr;
        static const std::vector<uint32_t> out_of_range{0, 1, 2, 3, 4, 5, 6, 44};
        bad.host_slots = &out_of_range;
        VJ_CHECK(!build_hit_plan(bad, HitVariant::dot4).ok);
        bad = req;
        bad.jobs = 0;
        VJ_CHECK(!build_hit_plan(bad, HitVariant::dot4).ok);
        bad = req;
        bad.slot_count = 0;
        VJ_CHECK(!build_hit_plan(bad, HitVariant::dot4).ok);
        bad = req;
        bad.y[1] = 0;
        VJ_CHECK(!build_hit_plan(bad, HitVariant::dot4).ok);
        bad = req;
        bad.cards = 3;
        VJ_CHECK(!build_hit_plan(bad, HitVariant::dot4).ok);
        static const std::vector<uint32_t> short_slots{0, 1};
        bad = req;
        bad.host_slots = &short_slots;
        VJ_CHECK(!build_hit_plan(bad, HitVariant::dot4).ok);
    }

    // ---- generation/pending ledger: never dispatch a slot under refill or after eviction ----
    {
        SlotLedger led(2, 4);
        std::string why;
        VJ_CHECK(led.layers() == 2 && led.slots() == 4);
        VJ_CHECK(led.generation(0) == 0 && led.pending_count() == 0);

        const uint64_t t1 = led.begin_refill(0, 7, 2, &why);
        VJ_CHECK(t1 != 0);
        VJ_CHECK(led.generation(0) == 1);
        VJ_CHECK(led.pending(0, 2) && !led.ready(0, 2) && led.expert(0, 2) == 7);
        VJ_CHECK(led.pending_count() == 1);
        VJ_CHECK(!led.complete_refill(0, &why));

        SlotLedger::Snapshot snap;
        VJ_CHECK(!led.snapshot(0, {2}, &snap, &why));
        VJ_CHECK(why.find("not ready") != std::string::npos);

        VJ_CHECK(led.complete_refill(t1, &why));
        VJ_CHECK(led.generation(0) == 1); // completion does not change the table generation
        VJ_CHECK(led.ready(0, 2) && !led.pending(0, 2) && led.pending_count() == 0);
        VJ_CHECK(led.snapshot(0, {2}, &snap, &why));
        VJ_CHECK(led.validate_snapshot(snap, &why));

        // Re-binding the same slot (evict + admit) bumps the generation and invalidates the old ticket.
        const uint64_t t2 = led.begin_refill(0, 8, 2, &why);
        VJ_CHECK(t2 != 0 && t2 != t1);
        VJ_CHECK(led.generation(0) == 2);
        VJ_CHECK(!led.complete_refill(t1, &why));
        VJ_CHECK(why.find("stale") != std::string::npos);
        VJ_CHECK(!led.validate_snapshot(snap, &why)); // stale generation
        VJ_CHECK(!led.snapshot(0, {2}, &snap, &why)); // still pending
        VJ_CHECK(led.complete_refill(t2, &why));
        VJ_CHECK(led.snapshot(0, {2}, &snap, &why));
        VJ_CHECK(led.validate_snapshot(snap, &why));
        VJ_CHECK(led.expert(0, 2) == 8);

        // The checked dispatch builder refuses the same snapshot as soon as the slot is re-bound.
        HitRequest one = hit_request();
        one.jobs = 1;
        one.slot_count = 4;
        static const std::vector<uint32_t> one_slot{2};
        static const std::vector<uint32_t> one_token{0};
        one.host_slots = &one_slot;
        one.host_tokens = &one_token;
        one.token_count = 1;
        VJ_CHECK(build_hit_plan_checked(one, HitVariant::dot4, led, snap).ok);
        const uint64_t t_pending = led.begin_refill(0, 9, 2, &why);
        VJ_CHECK(t_pending != 0);
        VJ_CHECK(!build_hit_plan_checked(one, HitVariant::dot4, led, snap).ok);
        VJ_CHECK(led.complete_refill(t_pending, &why));
        SlotLedger::Snapshot fresh;
        VJ_CHECK(led.snapshot(0, {2}, &fresh, &why));
        VJ_CHECK(build_hit_plan_checked(one, HitVariant::dot4, led, fresh).ok);
        VJ_CHECK(!build_hit_plan_checked(one, HitVariant::dot4, led, snap).ok); // old generation

        // Explicit eviction bumps the generation, clears the slot and makes a captured snapshot stale.
        VJ_CHECK(led.evict(0, 2, &why));
        VJ_CHECK(led.generation(0) == 4);
        VJ_CHECK(!led.ready(0, 2) && led.expert(0, 2) == -1);
        VJ_CHECK(!led.validate_snapshot(snap, &why));
        VJ_CHECK(!led.evict(0, 2, &why)); // already empty

        // Another layer's refill does not perturb this generation.
        const uint64_t t3 = led.begin_refill(1, 9, 0, &why);
        VJ_CHECK(t3 != 0 && led.generation(1) == 1 && led.generation(0) == 4);
        VJ_CHECK(led.complete_refill(t3, &why));
        VJ_CHECK(led.ready_slots(1).size() == 1 && led.ready_slots(1)[0] == 0);
        VJ_CHECK(led.ready_slots(0).empty());

        // Re-admitting a ready expert is a no-op; restarting a pending one invalidates its ticket.
        VJ_CHECK(led.begin_refill(1, 9, 0, &why) == 0);
        const uint64_t t4 = led.begin_refill(1, 10, 1, &why);
        VJ_CHECK(t4 != 0 && led.generation(1) == 2);
        const uint64_t t5 = led.begin_refill(1, 11, 1, &why);
        VJ_CHECK(t5 != 0 && led.generation(1) == 3 && !led.complete_refill(t4, &why));
        VJ_CHECK(led.complete_refill(t5, &why));
        VJ_CHECK(led.expert(1, 1) == 11);
    }

    // ---- integration with the host LRU: admissions -> refill tickets -> ready snapshot -> hit plan ----
    {
        vegajet::moecache::CacheConfig cfg;
        cfg.n_layers = 1;
        cfg.n_experts = 32;
        cfg.slots_per_layer = 8;
        cfg.max_admits_per_step = 16;
        cfg.max_admits_per_layer_step = 16;
        cfg.admit_after_misses = 1;
        vegajet::moecache::MoeLruCache cache(cfg);
        SlotLedger led(1, 8);

        int32_t experts[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        int32_t slots_out[8] = {};
        cache.route(0, experts, 8, slots_out);
        const auto admits = cache.take_admissions();
        VJ_CHECK(admits.size() == 8);
        HitRequest hr = hit_request();
        hr.jobs = 8;
        hr.slot_count = 8;
        hr.slots = {0x7400000ull, 0x7500000ull};
        hr.tokens = {0x7600000ull, 0x7700000ull};
        std::vector<uint32_t> ctx_slots, ctx_tokens;
        for (const auto& a : admits) {
            std::string why;
            const uint64_t t = led.begin_refill(a.layer, a.expert, a.slot, &why);
            VJ_CHECK(t != 0);
            const RefillPlan rp = build_refill_plan(refill_request(), a.expert, a.slot, 8);
            // The sealed refill has no expert/slot parameters: src and qs are already offset by the builder.
            VJ_CHECK(rp.ok && rp.args[0].src == refill_request().raw_plane[0] + a.expert * 503808ull);
            VJ_CHECK(rp.args[0].qs == refill_request().qs_cache[0] + a.slot * 917504ull);
            VJ_CHECK(led.complete_refill(t, &why));
            VJ_CHECK(cache.commit(a));
            ctx_slots.push_back(a.slot);
        }
        for (uint32_t t = 0; t < 8; ++t) ctx_tokens.push_back(t);
        hr.host_slots = &ctx_slots;
        hr.host_tokens = &ctx_tokens;
        hr.token_count = 8;
        SlotLedger::Snapshot snap;
        std::string why;
        VJ_CHECK(led.snapshot(0, ctx_slots, &snap, &why));
        VJ_CHECK(led.validate_snapshot(snap, &why));
        const HitPlan hp = build_hit_plan_checked(hr, HitVariant::dot4, led, snap);
        VJ_CHECK(hp.ok && hp.launch.grid_y == 8);
        VJ_CHECK(led.pending_count() == 0);
        VJ_CHECK(led.evict(0, ctx_slots[0], &why));
        VJ_CHECK(!led.validate_snapshot(snap, &why)); // a dispatch after this eviction must be refused
        VJ_CHECK(!build_hit_plan_checked(hr, HitVariant::dot4, led, snap).ok);
    }

    std::printf("test_moe_lru_sealed: sealed layout, refill/hit descriptors, ABI, generation/pending ledger OK\nPASS\n");
    return 0;
}
