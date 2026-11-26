// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Shared host/device scalar math for the 64x64x64 dequant-once-to-LDS routed MoE prefill kernels
// (ops/kernels/vj_prefill_moe_iq_lds.hip) and its CPU model (tests/test_prefill_moe_iq_lds.cpp).
//
// One tile = 32 expert rows x 64 routed tokens, advanced in K steps of 64 elements (two 32-element IQ
// groups); a 64-row expert block is two such column tiles. A tile decodes each weight of the step ONCE
// into INT8 LDS (gate/up together for IQ2_S), and two 32-token passes reuse that decode; the kernels
// accumulate in the production order (one int32 v_dot4 chain per production dot group, then one FP32 fma
// per group with (weight scale * activation scale)).
//
// Two production packed layouts are decoded here, both lossless re-layouts of ggml block bytes:
//   * IQ2_S Gate+Up merged plane ("wgu_merged", uint4 rows, 123 uint4 = 1968 B per row at D = 3072):
//       q/sign plane : nib*16 B, dword (ib*4 + part), part = {gate lo, gate hi, up lo, up hi};
//                      each dword = {idx0, sign0, idx1, sign1}
//       hs plane     : nib*4 B at nib*16, dword ib = {gate qh, gate scales, up qh, up scales}
//       d plane      : (nib/8)*4 B at nib*20, dword (ib/8) = {gate FP16 d, up FP16 d}
//     (identical to the packing in models/ornith/ornith_gpu.cpp pack_row)
//   * IQ3_XXS compact split plane ("w_dn_iq3xxs", 13 uint4 = 208 B per row at H = 512):
//       q plane : 16 lanes * 8 B  (lane l = weights 32l..32l+31; 8 grid index bytes)
//       aux     : 16 lanes * 4 B  (4 x 7-bit sign masks + 4-bit block scale in bits 28..31)
//       d       : 16 B at 192 (FP16 d of block 0 in the low half, block 1 in the high half)
//     (identical to the packing in models/ornith/ornith_gpu.cpp pack_dn)
//
// Decode is exact: 8-byte grid magnitudes from ggml iq2s_grid / 4-byte magnitudes from iq3xxs_grid,
// sign masks applied as int8 two's complement. Both grids contain no zero magnitude (IQ2_S values are
// {8, 25, 43}, IQ3_XXS max is 62), so the SWAR negation used by production is exact for every code.
#pragma once

#include <cstdint>
#ifndef __HIPCC__
#include <cstring>
#endif

#ifdef __HIPCC__
#define VJ_IL_HD __host__ __device__ __forceinline__
#else
#define VJ_IL_HD inline
#endif

namespace vegajet {
namespace il_lds {

enum : uint32_t {
    IL_WG        = 256,  // threads per work-group
    IL_ROWS      = 32,   // expert weight rows per tile (exactly one block-32 quant group)
    IL_TOKENS    = 64,   // routed tokens per work item (two 32-token passes)
    IL_TPASS     = 32,   // tokens per accumulator pass (8 FP32 accumulators/thread)
    IL_KSTEP     = 64,   // K elements staged per step (two 32-element IQ groups)
    IL_WS        = 20,   // LDS dword stride of one 64-weight row (80 B, 16 B aligned)
    IL_H         = 512,  // IQ3_XXS compact-plane row width (16 lanes x 32 weights)
    IL_IQ3_ROW   = 208,  // compact IQ3 row bytes (13 uint4)
    IL_IQ2_BLOCK = 82,   // ggml IQ2_S block bytes
    IL_IQ3_BLOCK = 98,   // ggml IQ3_XXS block bytes

    // ---- the requested single 64x64x64 work item -------------------------------------------
    // 64 expert rows x 64 routed tokens x K step 64 as ONE work item: a 512-thread work-group is
    // two 256-thread cohorts, cohort c owns weight rows c*32..c*32+31 (its own 32-row LDS slice),
    // and both cohorts consume the common activation tile staged by the whole work-group. All 64
    // tokens of the item are accumulated in a single pass, so the step's decode is consumed by 64
    // routed tokens and each weight is dequantised exactly once per work item.
    IL_WG64      = 512,  // threads per 64-row work-group
    IL_ROWS64    = 64,   // expert weight rows per work item (both block-32 quant groups)
    IL_ACC64     = 8,    // routed tokens per thread: 16 FP32 accumulators for {gate, up}
    IL_DEC64     = 1,    // 16-weight decode units per thread and step (64 rows x 8 units = 512)
};

// ---- layout offsets (D = hidden, H = 512) -----------------------------------------------
VJ_IL_HD uint32_t il_gu_row_bytes(uint32_t nib) { return nib * 16u + nib * 4u + (nib >> 3) * 4u; }
VJ_IL_HD uint32_t il_gu_q_word(uint32_t ib, uint32_t part) { return ib * 4u + part; }
VJ_IL_HD uint32_t il_gu_hs_word(uint32_t nib, uint32_t ib) { return nib * 4u + ib; }
VJ_IL_HD uint32_t il_gu_d_word(uint32_t nib, uint32_t ib) { return nib * 5u + (ib >> 3); }
// nib is D/32 and must be a multiple of 8 for the dword d plane (D multiple of 256).

VJ_IL_HD uint32_t il_iq3_q_off(uint32_t lane16) { return lane16 * 8u; }
VJ_IL_HD uint32_t il_iq3_aux_off(uint32_t lane16) { return 128u + lane16 * 4u; }
VJ_IL_HD uint32_t il_iq3_d_off() { return 192u; }

// ---- scalar helpers --------------------------------------------------------------------
VJ_IL_HD float il_h2f(uint32_t h16) {  // IEEE half -> float, no device intrinsics so the CPU model is identical
    const uint32_t sign = (h16 & 0x8000u) << 16;
    const uint32_t expo = (h16 >> 10) & 0x1Fu;
    const uint32_t mant = h16 & 0x3FFu;
    uint32_t bits;
    if (expo == 0u) {
        if (mant == 0u) { bits = sign; }
        else {  // subnormal: normalize
            uint32_t m = mant, e = 127u - 15u + 1u;
            while ((m & 0x400u) == 0u) { m <<= 1; --e; }
            bits = sign | (e << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (expo == 31u) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((expo + 127u - 15u) << 23) | (mant << 13);
    }
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}

VJ_IL_HD uint32_t il_sign4(uint32_t mag, uint32_t sign4) {
    // Per-byte two's complement of four int8 magnitudes. Exact for non-zero magnitudes (both IQ grids).
    const uint32_t add  = (sign4 * 0x00204081u) & 0x01010101u;
    const uint32_t xorm = (add << 8) - add;
    return (mag ^ xorm) + add;
}

VJ_IL_HD int32_t il_sdot4(uint32_t a, uint32_t b, int32_t c) {
    const int32_t a0 = int32_t(int8_t(a & 0xFFu)), a1 = int32_t(int8_t((a >> 8) & 0xFFu));
    const int32_t a2 = int32_t(int8_t((a >> 16) & 0xFFu)), a3 = int32_t(int8_t(a >> 24));
    const int32_t b0 = int32_t(int8_t(b & 0xFFu)), b1 = int32_t(int8_t((b >> 8) & 0xFFu));
    const int32_t b2 = int32_t(int8_t((b >> 16) & 0xFFu)), b3 = int32_t(int8_t(b >> 24));
    return c + a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
}

VJ_IL_HD int32_t il_dot4x4(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3,
                                  uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3) {
    int32_t d = il_sdot4(w0, x0, 0);
    d = il_sdot4(w1, x1, d);
    d = il_sdot4(w2, x2, d);
    d = il_sdot4(w3, x3, d);
    return d;
}

VJ_IL_HD int32_t il_dot8x8(const uint32_t* w, const uint32_t* x) {  // one 32-element IQ3 block
    int32_t d = il_sdot4(w[0], x[0], 0);
    for (uint32_t k = 1; k < 8u; ++k) d = il_sdot4(w[k], x[k], d);
    return d;
}

// 16 IQ2_S weights (one half of a 32-element group) -> four int8 dwords, matching the production decode:
//   sd = dword (gate FP16 | up FP16 << 16), shs = {qh, scales, qh, scales}, mat = 0 gate / 1 up,
//   half = 0 weights 0..15 / 1 weights 16..31. scale = d * (0.5 + 4-bit block scale) * 0.25.
struct IlIq2Unit { uint32_t w[4]; float scale; };
VJ_IL_HD IlIq2Unit il_iq2_unit(uint32_t sq, uint32_t shs, uint32_t sd, uint32_t mat, uint32_t half,
                                      const uint64_t* grid) {
    const uint32_t qh  = (shs >> (16u * mat)) & 0xFFu;
    const uint32_t scb = (shs >> (16u * mat + 8u)) & 0xFFu;
    const float d = il_h2f((sd >> (16u * mat)) & 0xFFFFu) * 0.25f;
    const uint32_t sh0 = half ? 4u : 8u;
    const uint32_t i0 = (sq & 0xFFu) | ((qh << sh0) & 0x300u);
    const uint32_t i1 = ((sq >> 16) & 0xFFu) | ((qh << (sh0 - 2u)) & 0x300u);
    const uint64_t g0 = grid[i0], g1 = grid[i1];
    IlIq2Unit u;
    u.w[0] = il_sign4(uint32_t(g0), (sq >> 8) & 0xFu);
    u.w[1] = il_sign4(uint32_t(g0 >> 32), (sq >> 12) & 0xFu);
    u.w[2] = il_sign4(uint32_t(g1), (sq >> 24) & 0xFu);
    u.w[3] = il_sign4(uint32_t(g1 >> 32), sq >> 28);
    u.scale = d * (0.5f + float(half ? (scb >> 4) : (scb & 0xFu)));
    return u;
}

// 32 IQ3_XXS weights (one compact-plane lane) -> eight int8 dwords. q2 = eight grid index bytes,
// aux = {4 x 7-bit sign masks, 4-bit scale}, d2 = {FP16 d block 0, FP16 d block 1} (lane 0..7 / 8..15).
struct IlIq3Unit { uint32_t w[8]; float scale; };
VJ_IL_HD IlIq3Unit il_iq3_unit(const uint32_t* q2, uint32_t aux, uint32_t d2, uint32_t lane16,
                                      const uint32_t* grid) {
    const uint8_t* qb = reinterpret_cast<const uint8_t*>(q2);
    IlIq3Unit u;
    for (uint32_t l = 0; l < 4u; ++l) {
        uint32_t s = (aux >> (7u * l)) & 127u;
        s |= uint32_t(__builtin_popcount(s) & 1u) << 7;      // ksigns_iq2xs implied 8th sign
        u.w[2u * l + 0u] = il_sign4(grid[qb[2u * l + 0u]], s & 0xFu);
        u.w[2u * l + 1u] = il_sign4(grid[qb[2u * l + 1u]], s >> 4);
    }
    const float d = il_h2f((lane16 < 8u) ? (d2 & 0xFFFFu) : (d2 >> 16));
    u.scale = d * (0.5f + float(aux >> 28)) * 0.5f;
    return u;
}

// ---- work-item enumeration (mirrored by the CPU model and the kernel queue) --------------
VJ_IL_HD uint32_t il_row_tiles(uint32_t lff) { return lff / IL_ROWS; }
VJ_IL_HD uint32_t il_items_per_expert(uint32_t cnt, uint32_t rt) {
    return ((cnt + IL_TOKENS - 1u) / IL_TOKENS) * rt;
}
// 64-row work items: one item is the whole 64-row block x all 64 tokens of a token tile.
VJ_IL_HD uint32_t il_row_tiles64(uint32_t lff) { return lff / IL_ROWS64; }
VJ_IL_HD uint32_t il_items_per_expert64(uint32_t cnt, uint32_t rt) { return il_items_per_expert(cnt, rt); }
// Thread roles of the 64-row work item (LAW: every device thread identity maps to exactly one
// decode unit and one (row, token-slot) accumulator pair; the CPU model replays them verbatim).
VJ_IL_HD uint32_t il_dec_row64(uint32_t tid) { return tid & (IL_ROWS64 - 1u); }        // decode row
VJ_IL_HD uint32_t il_dec_unit64(uint32_t tid) { return tid / IL_ROWS64; }              // 0..7 = (ib, mat, half)
VJ_IL_HD uint32_t il_acc_cohort64(uint32_t tid) { return tid / IL_WG; }                // 256-thread row cohort 0/1
VJ_IL_HD uint32_t il_acc_row64(uint32_t tid) {                                          // owned weight row 0..63
    return (tid / IL_WG) * IL_ROWS + (tid % IL_ROWS);
}
VJ_IL_HD uint32_t il_acc_slot64(uint32_t tid) { return (tid / IL_ROWS) % (IL_TOKENS / IL_ACC64); }  // 0..7
VJ_IL_HD uint32_t il_acc_tok64(uint32_t tid, uint32_t j) { return il_acc_slot64(tid) * IL_ACC64 + j; }

#ifndef __HIPCC__
// Host packing helpers: reproduce models/ornith/ornith_gpu.cpp pack_row / pack_dn exactly.
inline void il_pack_gu_row(const uint8_t* g, const uint8_t* u, uint32_t nib, uint8_t* dst) {
    uint32_t* qp = reinterpret_cast<uint32_t*>(dst);
    uint32_t* hs = qp + nib * 4u;
    uint32_t* dp = qp + nib * 5u;
    for (uint32_t b = 0; b < nib / 8u; ++b) {
        const uint8_t* gb = g + b * IL_IQ2_BLOCK;
        const uint8_t* ub = u + b * IL_IQ2_BLOCK;
        uint16_t gd = 0, ud = 0;
        std::memcpy(&gd, gb, 2);
        std::memcpy(&ud, ub, 2);
        dp[b] = uint32_t(gd) | (uint32_t(ud) << 16);
        for (uint32_t sb = 0; sb < 8u; ++sb) {
            const uint32_t ib = b * 8u + sb;
            const uint8_t* gq = gb + 2u + sb * 4u;
            const uint8_t* gs = gb + 34u + sb * 4u;
            const uint8_t* uq = ub + 2u + sb * 4u;
            const uint8_t* us = ub + 34u + sb * 4u;
            qp[ib * 4u + 0u] = uint32_t(gq[0]) | (uint32_t(gs[0]) << 8) | (uint32_t(gq[1]) << 16) | (uint32_t(gs[1]) << 24);
            qp[ib * 4u + 1u] = uint32_t(gq[2]) | (uint32_t(gs[2]) << 8) | (uint32_t(gq[3]) << 16) | (uint32_t(gs[3]) << 24);
            qp[ib * 4u + 2u] = uint32_t(uq[0]) | (uint32_t(us[0]) << 8) | (uint32_t(uq[1]) << 16) | (uint32_t(us[1]) << 24);
            qp[ib * 4u + 3u] = uint32_t(uq[2]) | (uint32_t(us[2]) << 8) | (uint32_t(uq[3]) << 16) | (uint32_t(us[3]) << 24);
            hs[ib] = uint32_t(gb[66u + sb]) | (uint32_t(gb[74u + sb]) << 8) |
                     (uint32_t(ub[66u + sb]) << 16) | (uint32_t(ub[74u + sb]) << 24);
        }
    }
}

// src = two ggml IQ3_XXS blocks (98 B each, H = 512) of one row.
inline void il_pack_iq3_row(const uint8_t* src, uint8_t* dst) {
    std::memset(dst, 0, IL_IQ3_ROW);
    std::memcpy(dst, src + 2u, 64u);
    std::memcpy(dst + 64u, src + IL_IQ3_BLOCK + 2u, 64u);
    std::memcpy(dst + 128u, src + 66u, 32u);
    std::memcpy(dst + 160u, src + IL_IQ3_BLOCK + 66u, 32u);
    uint16_t d0 = 0, d1 = 0;
    std::memcpy(&d0, src, 2);
    std::memcpy(&d1, src + IL_IQ3_BLOCK, 2);
    const uint32_t d2 = uint32_t(d0) | (uint32_t(d1) << 16);
    std::memcpy(dst + 192u, &d2, 4u);
}
#endif

} // namespace il_lds
} // namespace vegajet

#undef VJ_IL_HD
