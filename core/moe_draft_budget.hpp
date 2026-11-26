// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Physical footprint of the DEPLOYED external DFlash2 drafter, and the per-card draft reserve that the
// M3 hot-expert cache planner should use instead of the flat 1.1 GiB/card policy floor.
//
// Why this file exists
//   core/moe_slot_budget.hpp reserves kDraftReserveMinBytes (1.1 GiB, ceil) on EVERY card for "draft/MTP
//   weights and state". Measured against the deployed drafter and the real load order that is wrong in two
//   directions:
//     * the drafter lives on ONE card (models/ornith/ornith_gpu.cpp: VEGAJET_DFLASH2_GPU0 unset -> card 1,
//       set -> card 0). The other card hosts no drafter bytes at all;
//     * the drafter is opened, weight-uploaded, KV-allocated and scratch-allocated BEFORE the cache plan
//       reads rt.vram_arena(g).used() (same function, earlier lines). Its bytes are therefore already inside
//       the measured `used` base, so reserving the footprint again counts them twice.
//   Both errors make the cache SMALLER than the arena allows; they are conservative, not unsafe. This header
//   makes the real number explicit and lets the caller choose between
//     - DraftReserveMode::Marginal       : 0 on both cards (deployed order: drafter already in `used`), and
//     - DraftReserveMode::ActualFootprint: the measured footprint on the host card only (defensive).
//   The 1.1 GiB floor stays the default in moe_slot_budget; nothing changes unless a caller opts in through
//   plan_measured_slots_with_draft().
//
// Deployed provenance (read-only GGUF header parse, offline, 2026-11-25)
//   Ornith-1.5-122B-DFlash2-Q4_K_M (deployed GGUF)
//   file 551,110,784 B, 96 tensors, general.file_type 15 (Q4_K_M)
//   dflash.block_count=6, embedding_length=3072, feed_forward_length=9216,
//   attention.head_count=32, attention.head_count_kv=8, key_length=value_length=128,
//   attention.sliding_window=2048, attention.sliding_window_pattern=[1,1,1,1,1,1],
//   block_size=16, selector_rank=256, selector_top_k=16, target_layers=[2,8,15,21,27,33,40,46].
//   Sidecar mask_embedding.safetensors carries one BF16 row (D*2 B); the engine uploads it as F32 (D*4 B).
//   Every byte below is recomputed from these values; tests/test_moe_slot_budget_draft.cpp pins the totals.
//
// Layouts are the engine's, not estimates:
//   * Q4_K -> VJ_Q4_A8 via upload_fused_q4a8_tcc(): qs plane rows*stride_b where stride_b is
//     select_tcc_layout(cols/2).stride_bytes, except that the swizzled branch uses raw+256 (one extra
//     256 B tile; that is why fc.weight is 47,972,352 B, not 47,185,920 B), plus a dm plane
//     rows*(cols/32)*4 B;
//   * norms / conv bases are uploaded as plain F32 (`upload_f32_rows`);
//   * drafter KV uses the frozen spec_replay::plan_draft_kv() sizing: an SWA layer keeps a
//     `sliding_window`-slot ring, a full layer a power-of-two ring of min(max_ctx, full_ctx_cap) slots,
//     both counted as FP16 K + V = 2 * slots * n_kv * head_dim * 2 B;
//   * scratch is the exact `df_alloc(...)` list in ornith_gpu.cpp, with max_batch = max(128, prefill_chunk).
#pragma once

#include "core/moe_slot_budget.hpp"
#include "core/quant_repacker.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace vegajet::moeslot {

// GGUF-derived geometry of one deployed DFlash2 drafter. Defaults are the pinned deployed file.
struct Dflash2Meta {
    uint32_t n_layer = 6;             // dflash.block_count
    uint32_t hidden = 3072;           // dflash.embedding_length (= target D)
    uint32_t n_ff = 9216;             // dflash.feed_forward_length
    uint32_t n_head = 32;             // dflash.attention.head_count
    uint32_t n_head_kv = 8;           // dflash.attention.head_count_kv
    uint32_t head_dim = 128;          // dflash.attention.key_length == value_length
    uint32_t sliding_window = 2048;   // dflash.attention.sliding_window (ring slots of an SWA layer)
    uint32_t block_size = 16;         // dflash.block_size (B)
    uint32_t n_taps = 8;              // len(dflash.target_layers)
    uint32_t selector_rank = 256;     // dflash.selector_rank
    uint32_t full_ctx_cap = 32768;    // ornith_gpu.cpp: kc.full_ctx_cap
    uint32_t max_ctx = 8192;          // engine --max-ctx default (RunOptions); only full-attention
                                      // layers use it. The frozen 208 MiB control needs max_ctx >= 32768.
    std::vector<uint8_t> swa_pattern; // 1 = SWA ring, 0 = full-attention ring; empty = every layer SWA
};

inline bool dflash2_meta_valid(const Dflash2Meta& m) {
    if (m.n_layer == 0u || m.hidden == 0u || m.hidden % 256u != 0u || m.n_ff == 0u || m.n_ff % 256u != 0u) return false;
    if (m.n_head == 0u || m.n_head_kv == 0u || m.head_dim == 0u) return false;
    if (m.sliding_window == 0u || m.n_taps == 0u || m.selector_rank == 0u) return false;
    return m.swa_pattern.empty() || m.swa_pattern.size() == m.n_layer;
}

// upload_fused_q4a8_tcc(): one tensor's qs + dm planes. Mirrors the engine's stride rule exactly.
struct Q4a8Plane {
    uint64_t qs_bytes = 0;
    uint64_t dm_bytes = 0;
    uint32_t row_stride = 0;
    bool swizzled = false;
    uint64_t total() const { return qs_bytes + dm_bytes; }
};

inline Q4a8Plane q4a8_plane(uint64_t rows, uint32_t cols) {
    Q4a8Plane p;
    if (cols == 0u || cols % 32u != 0u) return p;
    const uint32_t raw = cols / 2u;
    const quant::TccRowLayout lay = quant::select_tcc_layout(raw);
    p.swizzled = lay.mode == quant::TccLayoutMode::XorTileSwizzle;
    p.row_stride = p.swizzled ? raw + quant::kTccTileBytes : lay.stride_bytes;
    p.qs_bytes = rows * p.row_stride;
    p.dm_bytes = rows * (cols / 32u) * 4ull;
    return p;
}

struct DraftWeightBytes {
    uint64_t top = 0;        // norms + fc + selector_hidden
    uint64_t per_layer = 0;  // one drafter block
    uint64_t total(uint32_t n_layer) const { return top + static_cast<uint64_t>(n_layer) * per_layer; }
};

// All bytes upload_fused_q4a8_tcc()/upload_f32_rows() commit for the DFlash2 drafter's weights (no host arrays).
inline DraftWeightBytes dflash2_weight_bytes(const Dflash2Meta& m) {
    DraftWeightBytes w;
    const uint64_t D = m.hidden;
    w.top = 2ull * D * 4ull                                        // enc.output_norm, output_norm (F32)
          + q4a8_plane(D, m.n_taps * m.hidden).total()             // fc.weight [n_taps*D, D] (swizzled)
          + q4a8_plane(m.selector_rank, m.hidden).total();         // selector_hidden.weight [D, selector_rank]
    const uint64_t dyn_rows = 4ull * (D / 16u);                    // attn/ffn conv projection rows
    const uint64_t qkv_rows = (m.n_head + 2ull * m.n_head_kv) * m.head_dim;       // q | k | v fused rows
    w.per_layer = 2ull * D * 4ull                                  // attn_norm, ffn_norm
                + 2ull * D * 2ull * 2ull * 4ull                    // attn_conv_base, ffn_conv_base [D,2,2]
                + 2ull * m.head_dim * 4ull                         // attn_q_norm, attn_k_norm
                + q4a8_plane(dyn_rows, m.hidden).total()          // attn_conv_proj.weight
                + q4a8_plane(qkv_rows, m.hidden).total()          // attn_q/k/v weights
                + q4a8_plane(D, m.n_head * m.head_dim).total()    // attn_output.weight
                + q4a8_plane(dyn_rows, m.hidden).total()          // ffn_conv_proj.weight
                + q4a8_plane(2ull * m.n_ff, m.hidden).total()     // ffn_gate + ffn_up
                + q4a8_plane(D, m.n_ff).total();                  // ffn_down
    return w;
}

inline uint32_t draft_next_pow2_u32(uint32_t v) {
    uint32_t p = 1u;
    while (p < v && p <= (1u << 30)) p <<= 1u;
    return p;
}

struct DraftKvBytes {
    uint64_t total = 0;
    uint32_t full_slots = 0;
    uint32_t n_full = 0;
    uint32_t n_swa = 0;
};

// Same sizing as spec_replay::plan_draft_kv(): SWA ring = sliding_window, full ring = pow2(min(max_ctx, cap)).
inline DraftKvBytes dflash2_kv_bytes(const Dflash2Meta& m) {
    DraftKvBytes k;
    const uint32_t want = std::min(std::max(m.max_ctx, 1u), std::max(m.full_ctx_cap, 1u));
    k.full_slots = std::max(m.sliding_window, draft_next_pow2_u32(want));
    for (uint32_t l = 0; l < m.n_layer; ++l) {
        const bool swa = m.swa_pattern.empty() || m.swa_pattern[l] != 0u;
        const uint32_t slots = swa ? m.sliding_window : k.full_slots;
        k.total += 2ull * slots * m.n_head_kv * m.head_dim * 2ull; // FP16 K + V ring
        if (swa) ++k.n_swa; else ++k.n_full;
    }
    return k;
}

// Exact `df_alloc(...)` list (ornith_gpu.cpp), with max_batch = max(128, opt.prefill_chunk).
inline uint64_t dflash2_scratch_bytes(const Dflash2Meta& m, uint32_t max_batch) {
    const uint64_t D = m.hidden, FF = m.n_ff, NT = m.n_taps;
    const uint64_t dyn = 4ull * (D / 16u);
    return static_cast<uint64_t>(max_batch) * NT * D * 4ull                        // df.taps_va (scales with batch)
         + 16ull * NT * D                                                          // taps_q8
         + 16ull * ((NT * D) / 32ull) * 8ull                                       // taps_ds
         + 16ull * D * 4ull + 16ull * D + 16ull * (D / 32ull) * 8ull               // ctx_{f32,q8,ds}
         + 16ull * D * 4ull + 16ull * D * 4ull + 16ull * D + 16ull * (D / 32ull) * 8ull // h, xnorm, xq8, xds
         + 16ull * dyn * 4ull                                                      // dyn
         + 16ull * (m.n_head + 2ull * m.n_head_kv) * m.head_dim * 4ull             // qkv
         + 16ull * m.n_head * m.head_dim                                           // attn_q8
         + 16ull * ((m.n_head * m.head_dim) / 32ull) * 8ull                        // attn_ds
         + 16ull * D * 4ull                                                        // out
         + 16ull * 2ull * FF * 4ull                                                // gu
         + 16ull * FF + 16ull * (FF / 32ull) * 8ull;                               // mid_{q8,ds}
}

struct DraftFootprint {
    uint64_t weights = 0;
    uint64_t kv = 0;
    uint64_t mask_embedding = 0;
    uint64_t scratch = 0;
    uint64_t total() const { return weights + kv + mask_embedding + scratch; }
};

// mask_embedding_bytes: the engine allocates D*4 F32 after widening the BF16 sidecar row.
inline DraftFootprint dflash2_footprint(const Dflash2Meta& m, uint32_t max_batch, uint64_t mask_embedding_bytes) {
    DraftFootprint f;
    f.weights = dflash2_weight_bytes(m).total(m.n_layer);
    f.kv = dflash2_kv_bytes(m).total;
    f.mask_embedding = mask_embedding_bytes;
    f.scratch = dflash2_scratch_bytes(m, max_batch);
    return f;
}

enum class DraftReserveMode {
    Marginal,        // deployed order: the drafter is already inside the measured `used` bytes -> reserve 0
    ActualFootprint, // defensive: keep the measured footprint free on the host card as well
};

struct DraftPlacement {
    uint32_t host_card = 1;  // VEGAJET_DFLASH2_GPU0 unset -> 1; set -> 0
    DraftReserveMode mode = DraftReserveMode::Marginal;
};

// Reserve per card: only the host card can carry drafter bytes, and Marginal reserves nothing.
inline std::array<uint64_t, kBudgetCards> draft_reserve_per_card(const DraftFootprint& f, const DraftPlacement& p) {
    std::array<uint64_t, kBudgetCards> r{};
    if (p.host_card < kBudgetCards && p.mode == DraftReserveMode::ActualFootprint) r[p.host_card] = f.total();
    return r;
}

// Both-card cache plan with the drafter's real per-card reserve. The flat 1.1 GiB floor is disabled here on
// purpose: the measured footprint (or 0 in Marginal mode) replaces it, and every other M3 rule still applies
// (the 2.70 GiB claim is a ceiling, measured free caps it, both cards must fit, a zero grant disables).
inline MultiCardGrant plan_measured_slots_with_draft(
    const SlotBudgetInput& in, const std::array<CardArenaMeasurement, kBudgetCards>& measured, uint32_t cards,
    const DraftFootprint& footprint, const DraftPlacement& placement) {
    SlotBudgetInput ci = in;
    ci.enforce_draft_floor = false;
    ci.draft_reserve_bytes = 0; // the per-card array below is the reserve; this only satisfies shape checks
    return plan_measured_slots_per_card(ci, measured, cards, draft_reserve_per_card(footprint, placement));
}

} // namespace vegajet::moeslot
