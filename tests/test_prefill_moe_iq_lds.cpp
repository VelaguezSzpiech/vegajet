// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU model, production-decoder cross-check and edge/coverage oracle for the dequant-once-to-LDS
// routed MoE prefill kernels (ops/kernels/vj_prefill_moe_iq_lds.hip):
//   * the requested single 64x64x64 work item: 64 expert rows x 64 routed tokens x K step 64 in one
//     512-thread work-group (two 256-thread row cohorts, 16 FP32 accumulators/thread, ONE token
//     pass so each weight of a step is dequantised once for all 64 tokens), and
//   * the 32-row x 64-token diagnostic tile (one 256-thread work-group, two 32-token passes).
//
// What is verified, and how it stays independent of the kernel:
//  1. Layout fixtures are built from ggml IQ2_S / IQ3_XXS block bytes, packed with the production
//     merged / compact split-plane packing, and the packing is re-derived byte-for-byte by a second,
//     literal transcription of models/ornith/ornith_gpu.cpp pack_row / pack_dn in this file.
//  2. Every decoded weight of a tile is compared element-by-element with the bit-exact production
//     float decoder (quant::DecoderRegistry) on the original block bytes: int8 value and per-group
//     scale must reproduce the production dequantisation exactly.
//  3. The full thread-mapped kernel model (decode -> LDS -> 1x4 micro-tile -> block-32 quant for
//     Gate+Up, and -> 32-block dot chain -> 8-slot part_out for Down) is replayed for every thread,
//     every token pass and every K step, and compared with a double-precision oracle built from the
//     production decoder and the same quantised activations.
//  4. Edges: 64-token tile tails (cnt 1..129), row tiles 64/128/512, K = 256 / 512 / 3072
//     (nib 8 / 16 / 96), empty experts, per-output write coverage (exactly one writer), and the
//     quantisation contrasts (round-half-even int8, amax block scale, reconstruction error).
#include "ops/kernels/vj_prefill_iq_lds_math.hpp"
#include "core/ggml_iq_tables.hpp"
#include "core/quant_repacker.hpp"
#include "tests/test_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

using namespace vegajet;
using namespace vegajet::il_lds;

namespace {

uint32_t g_seed = 0x5EED1234u;
uint32_t rnd() { g_seed = g_seed * 1664525u + 1013904223u; return g_seed >> 8; }

uint16_t half_bits(float f) { return quant::f32_to_fp16(f); }

// ---------------------------------------------------------------------------------------------
// Fixtures: raw ggml blocks + production packing + production float decode
// ---------------------------------------------------------------------------------------------
struct GuFx {
    uint32_t E, lff, D, nib, row_bytes, rows;
    std::vector<uint8_t> raw_g, raw_u, merged;
    std::vector<float> ref_g, ref_u;
    GuFx(uint32_t e, uint32_t ff, uint32_t d)
        : E(e), lff(ff), D(d), nib(d / 32u), row_bytes(il_gu_row_bytes(d / 32u)), rows(e * ff) {
        const size_t raw_row = size_t(nib) * IL_IQ2_BLOCK;
        raw_g.resize(size_t(rows) * raw_row);
        raw_u.resize(raw_g.size());
        merged.resize(size_t(rows) * row_bytes);
        ref_g.resize(size_t(rows) * D);
        ref_u.resize(ref_g.size());
        auto& reg = quant::DecoderRegistry::instance();
        for (uint32_t r = 0; r < rows; ++r) {
            uint8_t* g = raw_g.data() + size_t(r) * raw_row;
            uint8_t* u = raw_u.data() + size_t(r) * raw_row;
            for (uint32_t b = 0; b < nib; ++b) {
                uint8_t* gb = g + size_t(b) * IL_IQ2_BLOCK;
                uint8_t* ub = u + size_t(b) * IL_IQ2_BLOCK;
                const uint16_t gd = half_bits(0.0002f * float(1u + (r * 7u + b) % 29u));
                const uint16_t ud = half_bits(0.0003f * float(1u + (r * 11u + b) % 23u));
                std::memcpy(gb, &gd, 2);
                std::memcpy(ub, &ud, 2);
                for (uint32_t i = 2; i < IL_IQ2_BLOCK; ++i) {
                    gb[i] = uint8_t(rnd());
                    ub[i] = uint8_t(rnd() ^ 0x5Au);
                }
            }
            VJ_CHECK(reg.decode_row_f32(GgmlType::IQ2_S, g, ref_g.data() + size_t(r) * D, D));
            VJ_CHECK(reg.decode_row_f32(GgmlType::IQ2_S, u, ref_u.data() + size_t(r) * D, D));
            il_pack_gu_row(g, u, nib, merged.data() + size_t(r) * row_bytes);
        }
        // Second, independent transcription of the production packing: every merged dword must match.
        for (uint32_t r = 0; r < rows; ++r) {
            const uint8_t* g = raw_g.data() + size_t(r) * raw_row;
            const uint8_t* u = raw_u.data() + size_t(r) * raw_row;
            const uint8_t* m = merged.data() + size_t(r) * row_bytes;
            for (uint32_t ib = 0; ib < nib; ++ib) {
                const uint32_t b = ib >> 3, sb = ib & 7u;
                const uint8_t* gb = g + size_t(b) * IL_IQ2_BLOCK;
                const uint8_t* ub = u + size_t(b) * IL_IQ2_BLOCK;
                for (uint32_t part = 0; part < 4u; ++part) {
                    const uint8_t* src_q = (part < 2u) ? gb + 2u + sb * 4u : ub + 2u + sb * 4u;
                    const uint8_t* src_s = (part < 2u) ? gb + 34u + sb * 4u : ub + 34u + sb * 4u;
                    const uint32_t h = part & 1u;
                    const uint32_t expect = uint32_t(src_q[h * 2u + 0u]) | (uint32_t(src_s[h * 2u + 0u]) << 8) |
                                            (uint32_t(src_q[h * 2u + 1u]) << 16) | (uint32_t(src_s[h * 2u + 1u]) << 24);
                    uint32_t got;
                    std::memcpy(&got, m + il_gu_q_word(ib, part) * 4u, 4);
                    VJ_CHECK(got == expect);
                }
                const uint32_t expect_hs = uint32_t(gb[66u + sb]) | (uint32_t(gb[74u + sb]) << 8) |
                                           (uint32_t(ub[66u + sb]) << 16) | (uint32_t(ub[74u + sb]) << 24);
                uint32_t got_hs;
                std::memcpy(&got_hs, m + il_gu_hs_word(nib, ib) * 4u, 4);
                VJ_CHECK(got_hs == expect_hs);
            }
            for (uint32_t b = 0; b < nib / 8u; ++b) {
                const uint8_t* gb = g + size_t(b) * IL_IQ2_BLOCK;
                const uint8_t* ub = u + size_t(b) * IL_IQ2_BLOCK;
                uint16_t gd = 0, ud = 0;
                std::memcpy(&gd, gb, 2);
                std::memcpy(&ud, ub, 2);
                const uint32_t expect = uint32_t(gd) | (uint32_t(ud) << 16);
                uint32_t got;
                std::memcpy(&got, m + il_gu_d_word(nib, b * 8u) * 4u, 4);
                VJ_CHECK(got == expect);
            }
        }
    }
    // Production scale of one 16-weight group of a row (mirrors ggml db[] for the given matrix).
    float group_scale(uint32_t row, uint32_t ib, uint32_t mat) const {
        const std::vector<uint8_t>& raw = mat ? raw_u : raw_g;
        const uint8_t* g = raw.data() + size_t(row) * nib * IL_IQ2_BLOCK + size_t(ib >> 3) * IL_IQ2_BLOCK;
        uint16_t dbits = 0;
        std::memcpy(&dbits, g, 2);
        return float(il_h2f(dbits)) * (0.5f + float(g[74u + (ib & 7u)] & 0xFu)) * 0.25f;
    }
    float group_scale_hi(uint32_t row, uint32_t ib, uint32_t mat) const {
        const std::vector<uint8_t>& raw = mat ? raw_u : raw_g;
        const uint8_t* g = raw.data() + size_t(row) * nib * IL_IQ2_BLOCK + size_t(ib >> 3) * IL_IQ2_BLOCK;
        uint16_t dbits = 0;
        std::memcpy(&dbits, g, 2);
        return float(il_h2f(dbits)) * (0.5f + float(g[74u + (ib & 7u)] >> 4)) * 0.25f;
    }
};

struct DnFx {
    uint32_t E, D, rows;
    std::vector<uint8_t> raw, merged;
    std::vector<float> ref;
    DnFx(uint32_t e, uint32_t d) : E(e), D(d), rows(e * d) {
        raw.resize(size_t(rows) * 2u * IL_IQ3_BLOCK);
        merged.resize(size_t(rows) * IL_IQ3_ROW);
        ref.resize(size_t(rows) * IL_H);
        auto& reg = quant::DecoderRegistry::instance();
        for (uint32_t r = 0; r < rows; ++r) {
            uint8_t* b = raw.data() + size_t(r) * 2u * IL_IQ3_BLOCK;
            for (uint32_t k = 0; k < 2u; ++k) {
                uint8_t* blk = b + size_t(k) * IL_IQ3_BLOCK;
                const uint16_t dd = half_bits(0.0004f * float(1u + (r * 5u + k) % 19u));
                std::memcpy(blk, &dd, 2);
                for (uint32_t i = 2; i < IL_IQ3_BLOCK; ++i) blk[i] = uint8_t(rnd());
            }
            VJ_CHECK(reg.decode_row_f32(GgmlType::IQ3_XXS, b, ref.data() + size_t(r) * IL_H, IL_H));
            il_pack_iq3_row(b, merged.data() + size_t(r) * IL_IQ3_ROW);
        }
        for (uint32_t r = 0; r < rows; ++r) {
            const uint8_t* b = raw.data() + size_t(r) * 2u * IL_IQ3_BLOCK;
            const uint8_t* m = merged.data() + size_t(r) * IL_IQ3_ROW;
            for (uint32_t lane = 0; lane < 16u; ++lane) {
                const uint32_t k = lane >> 3;
                const uint8_t* blk = b + size_t(k) * IL_IQ3_BLOCK;
                uint32_t expect_q[2];
                std::memcpy(expect_q, blk + 2u + size_t(lane & 7u) * 8u, 8);
                uint32_t got_q[2];
                std::memcpy(got_q, m + il_iq3_q_off(lane), 8);
                VJ_CHECK(got_q[0] == expect_q[0] && got_q[1] == expect_q[1]);
                uint32_t expect_aux = 0, got_aux = 0;
                std::memcpy(&expect_aux, blk + 66u + size_t(lane & 7u) * 4u, 4);
                std::memcpy(&got_aux, m + il_iq3_aux_off(lane), 4);
                VJ_CHECK(got_aux == expect_aux);
            }
            uint16_t d0 = 0, d1 = 0;
            std::memcpy(&d0, b, 2);
            std::memcpy(&d1, b + IL_IQ3_BLOCK, 2);
            uint32_t got_d = 0;
            std::memcpy(&got_d, m + il_iq3_d_off(), 4);
            VJ_CHECK(got_d == (uint32_t(d0) | (uint32_t(d1) << 16)));
        }
    }
    float block_scale(uint32_t row, uint32_t lane) const {
        const uint8_t* blk = raw.data() + size_t(row) * 2u * IL_IQ3_BLOCK + size_t(lane >> 3) * IL_IQ3_BLOCK;
        uint32_t aux = 0;
        uint16_t dbits = 0;
        std::memcpy(&aux, blk + 66u + size_t(lane & 7u) * 4u, 4);
        std::memcpy(&dbits, blk, 2);
        return float(il_h2f(dbits)) * (0.5f + float(aux >> 28)) * 0.5f;
    }
};

// Synthetic block-32 quantised activations.
struct Act {
    uint32_t tokens, D;
    std::vector<int8_t> q;
    std::vector<float> d;  // one scale per 32 elements (x_ds.x / mid_d)
    Act(uint32_t m, uint32_t dim) : tokens(m), D(dim), q(size_t(m) * dim), d(size_t(m) * (dim / 32u)) {
        for (uint32_t t = 0; t < tokens; ++t) {
            for (uint32_t i = 0; i < D; ++i) q[size_t(t) * D + i] = int8_t(int(rnd() % 255u) - 127);
            for (uint32_t b = 0; b < D / 32u; ++b)
                d[size_t(t) * (D / 32u) + b] = float(il_h2f(half_bits(0.001f * float(1u + (t + b) % 7u))));
        }
    }
};

// ---------------------------------------------------------------------------------------------
// Exhaustive grid/sign decode checks against the ggml tables themselves
// ---------------------------------------------------------------------------------------------
void check_decode_tables() {
    const uint64_t* g2 = ggml_tables::iq2s_grid;
    const uint32_t* g3 = ggml_tables::iq3xxs_grid;
    // Every 10-bit index and every 4-bit sign nibble of a 16-weight IQ2_S half-block.
    for (uint32_t low = 0; low < 256u; ++low) {
        for (uint32_t high = 0; high < 4u; ++high) {
            for (uint32_t half = 0; half < 2u; ++half) {
                const uint32_t sh0 = half ? 4u : 8u;
                const uint32_t qh = (half ? (high << 4) | (high << 6) : high);
                for (uint32_t s0 = 0; s0 < 16u; ++s0) {
                    const uint32_t s1 = 15u - s0;
                    const uint32_t low1 = low ^ 0x3Cu;
                    const uint32_t sq = low | (s0 << 8) | (low1 << 16) | (s1 << 24);
                    const IlIq2Unit u = il_iq2_unit(sq, qh, 0u, 0u, half, g2);
                    const uint32_t i0 = low | ((qh << sh0) & 0x300u);
                    const uint32_t i1 = low1 | ((qh << (sh0 - 2u)) & 0x300u);
                    for (uint32_t j = 0; j < 4u; ++j) {
                        const int32_t e0 = int32_t((g2[i0] >> (8u * j)) & 0xFFu) * int32_t((s0 >> j) & 1u ? -1 : 1);
                        const int32_t e1 = int32_t((g2[i1] >> (8u * j)) & 0xFFu) * int32_t((s1 >> j) & 1u ? -1 : 1);
                        VJ_CHECK(int32_t(int8_t(uint8_t(u.w[0] >> (8u * j)))) == e0);
                        VJ_CHECK(int32_t(int8_t(uint8_t(u.w[1] >> (8u * j)))) == int32_t((g2[i0] >> (8u * (j + 4))) & 0xFFu) *
                                                                                 int32_t((s0 >> (j + 4)) & 1u ? -1 : 1));
                        VJ_CHECK(int32_t(int8_t(uint8_t(u.w[2] >> (8u * j)))) == e1);
                        VJ_CHECK(int32_t(int8_t(uint8_t(u.w[3] >> (8u * j)))) == int32_t((g2[i1] >> (8u * (j + 4))) & 0xFFu) *
                                                                                 int32_t((s1 >> (j + 4)) & 1u ? -1 : 1));
                    }
                    VJ_CHECK(u.scale == 0.0f);  // zero d bits and a 0.5 block scale -> 0
                }
            }
        }
    }
    // IQ3_XXS: all 256 grid indices x all 128 sign patterns, both d halves and a spread of scale nibbles.
    for (uint32_t idx = 0; idx < 256u; ++idx) {
        for (uint32_t pat = 0; pat < 128u; ++pat) {
            const uint32_t qw[2] = {idx | ((idx ^ 0xA5u) << 8) | ((idx ^ 0x11u) << 16) | ((idx ^ 0x27u) << 24),
                                    (idx ^ 0x5Au) | ((idx ^ 0x33u) << 8) | ((idx ^ 0x0Fu) << 16) | ((idx ^ 0x71u) << 24)};
            const uint32_t aux = pat | (pat << 7) | (((pat ^ 0x55u) & 127u) << 14) | ((pat ^ 0x2Au) << 21);
            const uint8_t* qb = reinterpret_cast<const uint8_t*>(qw);
            for (uint32_t lane16 : {0u, 9u}) {
                const IlIq3Unit u = il_iq3_unit(qw, aux, 0u, lane16, g3);
                for (uint32_t l = 0; l < 4u; ++l) {
                    uint32_t sbits = (aux >> (7u * l)) & 127u;
                    sbits |= uint32_t(__builtin_popcount(sbits) & 1u) << 7;   // ksigns_iq2xs implied 8th sign
                    for (uint32_t j = 0; j < 8u; ++j) {
                        const uint32_t e = (g3[qb[(j >> 2) + 2u * l]] >> (8u * (j & 3u))) & 0xFFu;
                        const int32_t expect = int32_t(e) * ((((sbits >> j) & 1u) != 0u) ? -1 : 1);
                        const uint32_t word = u.w[2u * l + (j >> 2)];
                        VJ_CHECK(int32_t(int8_t(uint8_t(word >> (8u * (j & 3u))))) == expect);
                    }
                }
            }
        }
    }
    // Scale arithmetic of both formats: FP16 d pairs and the 4-bit block scales.
    for (uint32_t dsel = 0; dsel < 2u; ++dsel) {
        const uint16_t dd = half_bits(0.001171875f);
        const uint32_t sd = dsel ? (uint32_t(dd) << 16) : uint32_t(dd);
        for (uint32_t mat = 0; mat < 2u; ++mat) {
            for (uint32_t half = 0; half < 2u; ++half) {
                const uint32_t scb = 0x3F4Bu;
                const uint32_t shs = 0x11u | (scb << 8) | (0x22u << 16) | (0x33u << 24);
                const IlIq2Unit u = il_iq2_unit(0u, shs, sd, mat, half, g2);
                const uint32_t scb_m = (shs >> (16u * mat + 8u)) & 0xFFu;
                const float want = float(il_h2f((sd >> (16u * mat)) & 0xFFFFu)) * (0.5f + float(half ? (scb_m >> 4) : (scb_m & 0xFu))) * 0.25f;
                VJ_CHECK(std::fabs(u.scale - want) <= 1e-9f * (1.0f + want));
            }
        }
        const uint32_t zeros[2] = {0u, 0u};
        for (uint32_t lane16 = 0; lane16 < 16u; ++lane16) {
            const uint32_t aux = 9u << 28;
            const IlIq3Unit u = il_iq3_unit(zeros, aux, sd, lane16, g3);
            const uint32_t h16 = (lane16 < 8u) ? (dsel ? 0u : uint32_t(dd)) : (dsel ? uint32_t(dd) : 0u);
            const float want = float(il_h2f(h16)) * (0.5f + 9.0f) * 0.5f;
            VJ_CHECK(std::fabs(u.scale - want) <= 1e-9f * (1.0f + want));
        }
    }
    std::printf("[OK] decode: 32,768 IQ2_S halves + 65,536 IQ3_XXS units match the ggml grids/signs exactly\n");
}

// ---------------------------------------------------------------------------------------------
// Gate+Up kernel model (thread by thread, LDS verbatim) + oracle
// ---------------------------------------------------------------------------------------------
struct GuTables {
    std::vector<uint32_t> off = std::vector<uint32_t>(256, 0), cnt = std::vector<uint32_t>(256, 0), sms;
};

struct GuResult {
    std::vector<int8_t> q8;
    std::vector<float> d;
    std::vector<uint8_t> q8_writer, d_writer;
    std::vector<float> g_acc, u_acc;  // the single writer thread's FP32 accumulators per (route,row)
    uint64_t decodes = 0;             // 16-weight decode units staged into LDS
    uint64_t steps = 0;               // tile steps executed (passes included)
    uint64_t passes = 0;
};

uint32_t gu_model(const GuFx& fx, const GuTables& tb, const Act& act, GuResult& out, bool verbose) {
    const uint32_t lff = fx.lff, D = fx.D, nib = fx.nib, rt_cnt = il_row_tiles(lff), step_cnt = nib >> 1;
    out.q8.assign(size_t(act.tokens) * lff, int8_t(0x7F));
    out.d.assign(size_t(act.tokens) * (lff / 32u), -1.0f);
    out.q8_writer.assign(out.q8.size(), 0);
    out.d_writer.assign(out.d.size(), 0);
    out.g_acc.assign(out.q8.size(), 0.0f);
    out.u_acc.assign(out.q8.size(), 0.0f);

    std::vector<uint16_t> pref(257, 0), next(257, 0);
    for (uint32_t t = 0; t < 256u; ++t) pref[t + 1] = uint16_t(il_items_per_expert(tb.cnt[t], rt_cnt));
    for (uint32_t step = 1; step < 256u; step <<= 1) {   // inclusive scan, old values only (WG barrier semantics)
        for (uint32_t t = 0; t < 256u; ++t)
            next[t + 1] = uint16_t(pref[t + 1] + ((t >= step) ? pref[t + 1 - step] : 0u));
        pref.swap(next);
    }
    const uint32_t total = pref[256];

    std::vector<uint32_t> lds_w(size_t(2) * IL_ROWS * IL_WS, 0xCDCDCDCDu);
    std::vector<uint32_t> lds_x(size_t(IL_TPASS) * IL_WS, 0xCDCDCDCDu);
    std::vector<float> lds_wsc(size_t(2) * IL_ROWS * 4u, -1.0f), lds_xsc(size_t(IL_TPASS) * 2u, -1.0f);
    std::vector<uint32_t> ms(IL_TPASS, 0u);
    std::vector<float> acc(size_t(IL_WG) * 8u, 0.0f);

    for (uint32_t item = 0; item < total; ++item) {
        uint32_t lo = 0, hi = 256;
        while (hi - lo > 1u) {
            const uint32_t mid = (lo + hi) >> 1;
            if (pref[mid] <= item) lo = mid; else hi = mid;
        }
        const uint32_t e = lo, local = item - pref[e], cnt = tb.cnt[e], off = tb.off[e];
        const uint32_t tt = local / rt_cnt, rt = local - tt * rt_cnt;
        const uint32_t t0 = tt * IL_TOKENS, tcnt = std::min<uint32_t>(IL_TOKENS, cnt - t0);
        VJ_CHECK(tcnt >= 1u && tcnt <= IL_TOKENS);
        for (uint32_t pass = 0; pass < 2u; ++pass) {
            const uint32_t pw = (tcnt > IL_TPASS * pass) ? std::min<uint32_t>(IL_TPASS, tcnt - IL_TPASS * pass) : 0u;
            if (!pw) break;
            ++out.passes;
            for (uint32_t t = 0; t < IL_TPASS; ++t) ms[t] = tb.sms[off + t0 + IL_TPASS * pass + std::min(t, pw - 1u)];
            std::fill(acc.begin(), acc.end(), 0.0f);

            for (uint32_t s = 0; s < step_cnt; ++s) {
                for (uint32_t tid = 0; tid < IL_WG; ++tid) {   // decode: 32 rows x 8 (ib, matrix, half) units
                    const uint32_t drow = tid & 31u, unit = tid >> 5;
                    const uint32_t ibs = unit >> 2, mat = (unit >> 1) & 1u, half = unit & 1u;
                    const uint32_t ib = 2u * s + ibs;
                    const uint32_t row = e * lff + rt * IL_ROWS + drow;
                    const uint8_t* mrow = fx.merged.data() + size_t(row) * fx.row_bytes;
                    uint32_t sd = 0, sq = 0, shs = 0;
                    std::memcpy(&sd, mrow + size_t(il_gu_d_word(nib, 2u * s)) * 4u, 4);
                    std::memcpy(&sq, mrow + size_t(il_gu_q_word(ib, mat * 2u + half)) * 4u, 4);
                    std::memcpy(&shs, mrow + size_t(il_gu_hs_word(nib, ib)) * 4u, 4);
                    const IlIq2Unit u = il_iq2_unit(sq, shs, sd, mat, half, ggml_tables::iq2s_grid);
                    uint32_t* dst = &lds_w[size_t(mat) * IL_ROWS * IL_WS + drow * IL_WS + ibs * 8u + half * 4u];
                    dst[0] = u.w[0]; dst[1] = u.w[1]; dst[2] = u.w[2]; dst[3] = u.w[3];
                    lds_wsc[size_t(mat) * IL_ROWS * 4u + drow * 4u + ibs * 2u + half] = u.scale;
                    ++out.decodes;
                }
                for (uint32_t tid = 0; tid < IL_WG; ++tid) {   // activations of the pass
                    if (tid < IL_TPASS * 4u) {
                        const uint32_t tok = tid >> 2, quarter = tid & 3u;
                        if (tok < pw) std::memcpy(&lds_x[tok * IL_WS + quarter * 4u],
                                                  &act.q[size_t(ms[tok]) * D + s * IL_KSTEP + quarter * 16u], 16);
                    } else if (tid < IL_TPASS * 5u) {
                        const uint32_t tok = tid - IL_TPASS * 4u;
                        if (tok < pw) {
                            lds_xsc[tok * 2u + 0u] = act.d[size_t(ms[tok]) * (D / 32u) + s * 2u + 0u];
                            lds_xsc[tok * 2u + 1u] = act.d[size_t(ms[tok]) * (D / 32u) + s * 2u + 1u];
                        }
                    }
                }
                if (item == 0 && pass == 0 && verbose) {   // decode vs the production float decoder
                    for (uint32_t drow = 0; drow < IL_ROWS; ++drow) {
                        const uint32_t row = e * lff + rt * IL_ROWS + drow;
                        for (uint32_t mat = 0; mat < 2u; ++mat) {
                            const std::vector<float>& ref = mat ? fx.ref_u : fx.ref_g;
                            for (uint32_t g = 0; g < 4u; ++g) {
                                const uint32_t ib = 2u * s + (g >> 1), half = g & 1u;
                                const float sc = lds_wsc[size_t(mat) * IL_ROWS * 4u + drow * 4u + g];
                                const float psc = half ? fx.group_scale_hi(row, ib, mat) : fx.group_scale(row, ib, mat);
                                VJ_CHECK(std::fabs(sc - psc) <= 1e-7f * (1.0f + std::fabs(psc)));
                                for (uint32_t j = 0; j < 4u; ++j) {
                                    const uint32_t word = lds_w[size_t(mat) * IL_ROWS * IL_WS + drow * IL_WS + g * 4u + j];
                                    for (uint32_t b = 0; b < 4u; ++b) {
                                        const uint32_t k = ib * 32u + half * 16u + j * 4u + b;
                                        const int32_t v = int32_t(int8_t(uint8_t(word >> (8u * b))));
                                        const float want = ref[size_t(row) * D + k];
                                        VJ_CHECK(std::fabs(want - psc * float(v)) <= 1e-6f * (1.0f + std::fabs(want)));
                                    }
                                }
                            }
                        }
                    }
                }
                for (uint32_t tid = 0; tid < IL_WG; ++tid) {   // 1 row x 4 tokens x {gate, up}
                    const uint32_t rc = tid & 31u, tc = tid >> 5;
                    for (uint32_t g = 0; g < 4u; ++g) {
                        const uint32_t xoff = g * 4u, xsi = g >> 1;
                        for (uint32_t mat = 0; mat < 2u; ++mat) {
                            const float sc = lds_wsc[size_t(mat) * IL_ROWS * 4u + rc * 4u + g];
                            const uint32_t* ww = &lds_w[size_t(mat) * IL_ROWS * IL_WS + rc * IL_WS + xoff];
                            for (uint32_t j = 0; j < 4u; ++j) {
                                const uint32_t tok = tc * 4u + j;
                                const uint32_t* xw = &lds_x[tok * IL_WS + xoff];
                                const int32_t dp = il_dot4x4(ww[0], ww[1], ww[2], ww[3], xw[0], xw[1], xw[2], xw[3]);
                                float& a = acc[size_t(tid) * 8u + mat * 4u + j];
                                a = std::fma(sc * lds_xsc[tok * 2u + xsi], float(dp), a);
                            }
                        }
                    }
                }
                ++out.steps;
            }
            for (uint32_t tid = 0; tid < IL_WG; ++tid) {   // SwiGLU + block-32 quant (32-lane reduce = the tile rows)
                const uint32_t rc = tid & 31u, tc = tid >> 5;
                for (uint32_t j = 0; j < 4u; ++j) {
                    const uint32_t tok = tc * 4u + j;
                    const float gv = acc[size_t(tid) * 8u + 0u * 4u + j];
                    const float uv = acc[size_t(tid) * 8u + 1u * 4u + j];
                    const float y = (gv / (1.0f + std::exp(-gv))) * uv;
                    float amax = std::fabs(y);
                    for (uint32_t lane = 0; lane < 32u; ++lane) {   // same token, all 32 rows of the block
                        if (lane == rc) continue;
                        const uint32_t other = tc * 32u + lane;
                        const float ov = acc[size_t(other) * 8u + 0u * 4u + j];
                        const float ou = acc[size_t(other) * 8u + 1u * 4u + j];
                        amax = std::max(amax, std::fabs((ov / (1.0f + std::exp(-ov))) * ou));
                    }
                    const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                    const int32_t qv = std::max(-127, std::min(127, int32_t(std::nearbyint(y * inv))));
                    if (tok < pw) {
                        const size_t dst = size_t(ms[tok]) * lff + rt * IL_ROWS + rc;
                        out.q8[dst] = int8_t(qv);
                        out.g_acc[dst] = gv;
                        out.u_acc[dst] = uv;
                        ++out.q8_writer[dst];
                        if (rc == 0u) {
                            const size_t dd = size_t(ms[tok]) * (lff / 32u) + rt;
                            out.d[dd] = amax * (1.0f / 127.0f);
                            ++out.d_writer[dd];
                        }
                    }
                }
            }
        }
    }
    return total;
}

void gu_oracle(const GuFx& fx, const GuTables& tb, const Act& act, GuResult& res, bool verbose) {
    const uint32_t routes = act.tokens;
    std::vector<uint8_t> used(routes, 0);
    std::vector<uint32_t> rex(routes, 0u);
    for (uint32_t e = 0; e < 256u; ++e)
        for (uint32_t t = 0; t < tb.cnt[e]; ++t) {
            const uint32_t r = tb.sms[tb.off[e] + t];
            VJ_CHECK(r < routes && used[r] == 0u);   // one expert per route, as in the production sort
            used[r] = 1;
            rex[r] = e;
        }
    for (uint32_t t = 0; t < routes; ++t)
        for (uint32_t row = 0; row < fx.lff; ++row) {
            VJ_CHECK(res.q8_writer[size_t(t) * fx.lff + row] == used[t]);
            if (!used[t]) continue;
            VJ_CHECK(res.d_writer[size_t(t) * (fx.lff / 32u) + row / 32u] == 1u);
        }
    double worst_g = 0.0, worst_u = 0.0, worst_d = 0.0, worst_r = 0.0;
    uint32_t lsb_outliers = 0;
    for (uint32_t e = 0; e < 256u; ++e) {
        for (uint32_t t = 0; t < tb.cnt[e]; ++t) {
            const uint32_t r = tb.sms[tb.off[e] + t];
            const uint32_t grow = rex[r] * fx.lff;
            std::vector<double> yv(fx.lff), gv(fx.lff), uv(fx.lff);
            for (uint32_t row = 0; row < fx.lff; ++row) {
                double g = 0.0, u = 0.0;
                for (uint32_t k = 0; k < fx.D; ++k) {
                    const double xv = double(act.q[size_t(r) * fx.D + k]) * double(act.d[size_t(r) * (fx.D / 32u) + k / 32u]);
                    g += double(fx.ref_g[size_t(grow + row) * fx.D + k]) * xv;
                    u += double(fx.ref_u[size_t(grow + row) * fx.D + k]) * xv;
                }
                const size_t at = size_t(r) * fx.lff + row;
                const double ag = res.g_acc[at], au = res.u_acc[at];
                VJ_CHECK(std::fabs(ag - g) <= 2e-4 * (1.0 + std::fabs(g)));
                VJ_CHECK(std::fabs(au - u) <= 2e-4 * (1.0 + std::fabs(u)));
                worst_g = std::max(worst_g, std::fabs(ag - g) / (1.0 + std::fabs(g)));
                worst_u = std::max(worst_u, std::fabs(au - u) / (1.0 + std::fabs(u)));
                gv[row] = g;
                uv[row] = u;
                yv[row] = (g / (1.0 + std::exp(-g))) * u;
            }
            for (uint32_t blk = 0; blk < fx.lff / 32u; ++blk) {
                double amax = 0.0;
                for (uint32_t k = 0; k < 32u; ++k) amax = std::max(amax, std::fabs(yv[blk * 32u + k]));
                const double dy = double(res.d[size_t(r) * (fx.lff / 32u) + blk]);
                const double want_d = amax / 127.0;
                worst_d = std::max(worst_d, std::fabs(dy - want_d) / (1.0 + want_d));
                VJ_CHECK(std::fabs(dy - want_d) <= 1e-5 * (1.0 + want_d));
                for (uint32_t k = 0; k < 32u; ++k) {
                    const uint32_t row = blk * 32u + k;
                    const size_t at = size_t(r) * fx.lff + row;
                    if (dy > 0.0) {
                        const double recon = double(res.q8[at]) * dy;
                        worst_r = std::max(worst_r, std::fabs(recon - yv[row]) / dy);
                        if (std::fabs(double(res.q8[at]) - yv[row] / dy) > 1.0) ++lsb_outliers;
                        VJ_CHECK(std::fabs(double(res.q8[at]) - yv[row] / dy) <= 1.0);
                    } else {
                        VJ_CHECK(yv[row] == 0.0 && res.q8[at] == 0);
                    }
                }
            }
        }
    }
    if (verbose)
        std::printf("[OK] Gate+Up oracle: g %.2e, u %.2e, block-scale %.2e, recon %.2e, int8 LSB outliers %u\n",
                    worst_g, worst_u, worst_d, worst_r, lsb_outliers);
    VJ_CHECK(worst_g <= 2e-4 && worst_u <= 2e-4 && worst_r <= 0.5 + 2e-3 && lsb_outliers == 0u);
}

// ---------------------------------------------------------------------------------------------
// Down kernel model + oracle
// ---------------------------------------------------------------------------------------------
struct DnResult {
    std::vector<float> part;
    std::vector<uint32_t> writer;
};

uint32_t dn_model(const DnFx& fx, const GuTables& tb, const Act& mid, const std::vector<float>& mid_d,
                  const std::vector<float>& ew, DnResult& out) {
    const uint32_t D = fx.D, rt_cnt = D / IL_ROWS, step_cnt = IL_H / IL_KSTEP;
    out.part.assign(size_t(mid.tokens) * D, 0.0f);
    out.writer.assign(out.part.size(), 0u);
    std::vector<uint16_t> pref(257, 0), next(257, 0);
    for (uint32_t t = 0; t < 256u; ++t) pref[t + 1] = uint16_t(il_items_per_expert(tb.cnt[t], rt_cnt));
    for (uint32_t step = 1; step < 256u; step <<= 1) {   // old values only (WG barrier semantics)
        for (uint32_t t = 0; t < 256u; ++t)
            next[t + 1] = uint16_t(pref[t + 1] + ((t >= step) ? pref[t + 1 - step] : 0u));
        pref.swap(next);
    }
    const uint32_t total = pref[256];

    std::vector<uint32_t> lds_w(size_t(IL_ROWS) * IL_WS, 0xCDCDCDCDu);
    std::vector<uint32_t> lds_x(size_t(IL_TPASS) * IL_WS, 0xCDCDCDCDu);
    std::vector<float> lds_wsc(size_t(IL_ROWS) * 2u, -1.0f), lds_xsc(size_t(IL_TPASS) * 2u, -1.0f);
    std::vector<uint32_t> ms(IL_TPASS, 0u);
    std::vector<float> acc(size_t(IL_WG) * 4u, 0.0f);

    for (uint32_t item = 0; item < total; ++item) {
        uint32_t lo = 0, hi = 256;
        while (hi - lo > 1u) {
            const uint32_t mid_i = (lo + hi) >> 1;
            if (pref[mid_i] <= item) lo = mid_i; else hi = mid_i;
        }
        const uint32_t e = lo, local = item - pref[e], cnt = tb.cnt[e], off = tb.off[e];
        const uint32_t tt = local / rt_cnt, rt = local - tt * rt_cnt;
        const uint32_t t0 = tt * IL_TOKENS, tcnt = std::min<uint32_t>(IL_TOKENS, cnt - t0);
        for (uint32_t pass = 0; pass < 2u; ++pass) {
            const uint32_t pw = (tcnt > IL_TPASS * pass) ? std::min<uint32_t>(IL_TPASS, tcnt - IL_TPASS * pass) : 0u;
            if (!pw) break;
            for (uint32_t t = 0; t < IL_TPASS; ++t) ms[t] = tb.sms[off + t0 + IL_TPASS * pass + std::min(t, pw - 1u)];
            std::fill(acc.begin(), acc.end(), 0.0f);
            for (uint32_t s = 0; s < step_cnt; ++s) {
                for (uint32_t tid = 0; tid < 64u; ++tid) {   // 32 rows x 2 lanes
                    const uint32_t drow = tid & 31u, half = tid >> 5;
                    const uint32_t lane16 = 2u * s + half;
                    const uint32_t row = e * D + rt * IL_ROWS + drow;
                    const uint8_t* mrow = fx.merged.data() + size_t(row) * IL_IQ3_ROW;
                    uint32_t qw[2] = {0u, 0u}, aux = 0, d2 = 0;
                    std::memcpy(qw, mrow + il_iq3_q_off(lane16), 8);
                    std::memcpy(&aux, mrow + il_iq3_aux_off(lane16), 4);
                    std::memcpy(&d2, mrow + il_iq3_d_off(), 4);
                    const IlIq3Unit u = il_iq3_unit(qw, aux, d2, lane16, ggml_tables::iq3xxs_grid);
                    uint32_t* dst = &lds_w[drow * IL_WS + half * 8u];
                    for (uint32_t k = 0; k < 8u; ++k) dst[k] = u.w[k];
                    lds_wsc[drow * 2u + half] = u.scale;
                }
                for (uint32_t tid = 64u; tid < IL_WG; ++tid) {   // activations
                    const uint32_t u = tid - 64u;
                    if (u < IL_TPASS * 4u) {
                        const uint32_t tok = u >> 2, quarter = u & 3u;
                        if (tok < pw) std::memcpy(&lds_x[tok * IL_WS + quarter * 4u],
                                                  &mid.q[size_t(ms[tok]) * IL_H + s * IL_KSTEP + quarter * 16u], 16);
                    }
                    if (u >= IL_TPASS * 4u && u < IL_TPASS * 5u) {   // ew * xd for the two 32-blocks
                        const uint32_t tok = u - IL_TPASS * 4u;
                        if (tok < pw) {
                            const uint32_t r = ms[tok];
                            lds_xsc[tok * 2u + 0u] = ew[r] * mid_d[size_t(r) * (IL_H / 32u) + s * 2u + 0u];
                            lds_xsc[tok * 2u + 1u] = ew[r] * mid_d[size_t(r) * (IL_H / 32u) + s * 2u + 1u];
                        }
                    }
                }
                if (item == 0 && pass == 0) {   // decode vs the production IQ3_XXS float decoder
                    for (uint32_t drow = 0; drow < IL_ROWS; ++drow) {
                        const uint32_t row = e * D + rt * IL_ROWS + drow;
                        for (uint32_t b = 0; b < 2u; ++b) {
                            const uint32_t lane = 2u * s + b;
                            const float sc = lds_wsc[drow * 2u + b];
                            const float psc = fx.block_scale(row, lane);
                            VJ_CHECK(std::fabs(sc - psc) <= 1e-7f * (1.0f + std::fabs(psc)));
                            for (uint32_t k = 0; k < 32u; ++k) {
                                const uint32_t word = lds_w[drow * IL_WS + b * 8u + (k >> 2)];
                                const int32_t v = int32_t(int8_t(uint8_t(word >> (8u * (k & 3u)))));
                                const float want = fx.ref[size_t(row) * IL_H + s * IL_KSTEP + b * 32u + k];
                                VJ_CHECK(std::fabs(want - psc * float(v)) <= 1e-6f * (1.0f + std::fabs(want)));
                            }
                        }
                    }
                }
                for (uint32_t tid = 0; tid < IL_WG; ++tid) {   // 1 row x 4 tokens per 32-block
                    const uint32_t rc = tid & 31u, tc = tid >> 5;
                    for (uint32_t b = 0; b < 2u; ++b) {
                        const float sc = lds_wsc[rc * 2u + b];
                        const uint32_t* wr = &lds_w[rc * IL_WS + b * 8u];
                        for (uint32_t j = 0; j < 4u; ++j) {
                            const uint32_t tok = tc * 4u + j;
                            const int32_t dp = il_dot8x8(wr, &lds_x[tok * IL_WS + b * 8u]);
                            float& a = acc[size_t(tid) * 4u + j];
                            a = std::fma(sc * lds_xsc[tok * 2u + b], float(dp), a);
                        }
                    }
                }
            }
            for (uint32_t tid = 0; tid < IL_WG; ++tid) {
                const uint32_t rc = tid & 31u, tc = tid >> 5;
                for (uint32_t j = 0; j < 4u; ++j) {
                    const uint32_t tok = tc * 4u + j;
                    if (tok >= pw) continue;
                    const size_t dst = size_t(ms[tok]) * D + rt * IL_ROWS + rc;
                    out.part[dst] = acc[size_t(tid) * 4u + j];
                    ++out.writer[dst];
                }
            }
        }
    }
    return total;
}

// ---------------------------------------------------------------------------------------------
// The requested single 64x64x64 work item, replayed thread by thread: 512 threads = two 256-thread
// row cohorts; cohort c owns weight rows c*32..c*32+31 (its own LDS slice) and both cohorts share
// the staged activation tile of the item. ONE token pass covers all 64 tokens, so each of the 512
// decode units of a K step (64 rows x 8 units) is staged once and consumed by all 64 tokens.
// Gate+Up: 1 row x 8 tokens x {gate, up} = 16 FP32 accumulators/thread. Down: 1 row x 8 tokens.
// ---------------------------------------------------------------------------------------------
uint32_t gu_model64(const GuFx& fx, const GuTables& tb, const Act& act, GuResult& out, bool verbose) {
    const uint32_t lff = fx.lff, D = fx.D, nib = fx.nib, rt_cnt = il_row_tiles64(lff), step_cnt = nib >> 1;
    out.q8.assign(size_t(act.tokens) * lff, int8_t(0x7F));
    out.d.assign(size_t(act.tokens) * (lff / 32u), -1.0f);
    out.q8_writer.assign(out.q8.size(), 0);
    out.d_writer.assign(out.d.size(), 0);
    out.g_acc.assign(out.q8.size(), 0.0f);
    out.u_acc.assign(out.q8.size(), 0.0f);
    out.decodes = out.steps = out.passes = 0;

    std::vector<uint16_t> pref(257, 0), next(257, 0);
    for (uint32_t t = 0; t < 256u; ++t) pref[t + 1] = uint16_t(il_items_per_expert64(tb.cnt[t], rt_cnt));
    for (uint32_t step = 1; step < 256u; step <<= 1) {   // old values only (WG barrier semantics)
        for (uint32_t t = 0; t < 256u; ++t)
            next[t + 1] = uint16_t(pref[t + 1] + ((t >= step) ? pref[t + 1 - step] : 0u));
        pref.swap(next);
    }
    const uint32_t total = pref[256];

    std::vector<uint32_t> lds_w(size_t(2) * IL_ROWS64 * IL_WS, 0xCDCDCDCDu);
    std::vector<uint32_t> lds_x(size_t(IL_TOKENS) * IL_WS, 0xCDCDCDCDu);
    std::vector<float> lds_wsc(size_t(2) * IL_ROWS64 * 4u, -1.0f), lds_xsc(size_t(IL_TOKENS) * 2u, -1.0f);
    std::vector<uint32_t> ms(IL_TOKENS, 0u);
    std::vector<float> acc(size_t(IL_WG64) * 2u * IL_ACC64, 0.0f);
    std::vector<uint32_t> dec_cov(size_t(IL_ROWS64) * 8u, 0u);
    std::vector<uint32_t> acc_cov(size_t(IL_ROWS64) * IL_TOKENS * 2u, 0u);

    for (uint32_t item = 0; item < total; ++item) {
        uint32_t lo = 0, hi = 256;
        while (hi - lo > 1u) {
            const uint32_t mid = (lo + hi) >> 1;
            if (pref[mid] <= item) lo = mid; else hi = mid;
        }
        const uint32_t e = lo, local = item - pref[e], cnt = tb.cnt[e], off = tb.off[e];
        const uint32_t tt = local / rt_cnt, rt = local - tt * rt_cnt;
        const uint32_t t0 = tt * IL_TOKENS, tcnt = std::min<uint32_t>(IL_TOKENS, cnt - t0);
        VJ_CHECK(tcnt >= 1u && tcnt <= IL_TOKENS);
        VJ_CHECK(rt < rt_cnt);
        ++out.passes;                                   // exactly one token pass per work item
        for (uint32_t t = 0; t < IL_TOKENS; ++t) ms[t] = tb.sms[off + t0 + std::min(t, tcnt - 1u)];
        std::fill(acc.begin(), acc.end(), 0.0f);

        for (uint32_t s = 0; s < step_cnt; ++s) {
            std::fill(dec_cov.begin(), dec_cov.end(), 0u);
            std::fill(acc_cov.begin(), acc_cov.end(), 0u);
            for (uint32_t tid = 0; tid < IL_WG64; ++tid) {   // decode: one 16-weight unit per thread
                const uint32_t drow = il_dec_row64(tid), unit = il_dec_unit64(tid);
                const uint32_t ibs = unit >> 2, mat = (unit >> 1) & 1u, half = unit & 1u;
                const uint32_t ib = 2u * s + ibs;
                const uint32_t row = e * lff + rt * IL_ROWS64 + drow;
                const uint8_t* mrow = fx.merged.data() + size_t(row) * fx.row_bytes;
                uint32_t sd = 0, sq = 0, shs = 0;
                std::memcpy(&sd, mrow + size_t(il_gu_d_word(nib, 2u * s)) * 4u, 4);
                std::memcpy(&sq, mrow + size_t(il_gu_q_word(ib, mat * 2u + half)) * 4u, 4);
                std::memcpy(&shs, mrow + size_t(il_gu_hs_word(nib, ib)) * 4u, 4);
                const IlIq2Unit u = il_iq2_unit(sq, shs, sd, mat, half, ggml_tables::iq2s_grid);
                uint32_t* dst = &lds_w[size_t(mat) * IL_ROWS64 * IL_WS + drow * IL_WS + ibs * 8u + half * 4u];
                dst[0] = u.w[0]; dst[1] = u.w[1]; dst[2] = u.w[2]; dst[3] = u.w[3];
                lds_wsc[size_t(mat) * IL_ROWS64 * 4u + drow * 4u + ibs * 2u + half] = u.scale;
                VJ_CHECK(++dec_cov[size_t(drow) * 8u + unit] == 1u);   // each (row, unit) staged once
                ++out.decodes;
            }
            for (uint32_t tid = 0; tid < IL_TOKENS * 5u; ++tid) {   // activations of all 64 tokens
                if (tid < IL_TOKENS * 4u) {
                    const uint32_t tok = tid >> 2, quarter = tid & 3u;
                    if (tok < tcnt) std::memcpy(&lds_x[tok * IL_WS + quarter * 4u],
                                                &act.q[size_t(ms[tok]) * D + s * IL_KSTEP + quarter * 16u], 16);
                } else {
                    const uint32_t tok = tid - IL_TOKENS * 4u;
                    if (tok < tcnt) {
                        lds_xsc[tok * 2u + 0u] = act.d[size_t(ms[tok]) * (D / 32u) + s * 2u + 0u];
                        lds_xsc[tok * 2u + 1u] = act.d[size_t(ms[tok]) * (D / 32u) + s * 2u + 1u];
                    }
                }
            }
            if (item == 0 && verbose) {   // decode vs the bit-exact production float decoder
                for (uint32_t drow = 0; drow < IL_ROWS64; ++drow) {
                    const uint32_t row = e * lff + rt * IL_ROWS64 + drow;
                    for (uint32_t mat = 0; mat < 2u; ++mat) {
                        const std::vector<float>& ref = mat ? fx.ref_u : fx.ref_g;
                        for (uint32_t g = 0; g < 4u; ++g) {
                            const uint32_t ib = 2u * s + (g >> 1), half = g & 1u;
                            const float sc = lds_wsc[size_t(mat) * IL_ROWS64 * 4u + drow * 4u + g];
                            const float psc = half ? fx.group_scale_hi(row, ib, mat) : fx.group_scale(row, ib, mat);
                            VJ_CHECK(std::fabs(sc - psc) <= 1e-7f * (1.0f + std::fabs(psc)));
                            for (uint32_t j = 0; j < 4u; ++j) {
                                const uint32_t word =
                                    lds_w[size_t(mat) * IL_ROWS64 * IL_WS + drow * IL_WS + g * 4u + j];
                                for (uint32_t b = 0; b < 4u; ++b) {
                                    const uint32_t k = ib * 32u + half * 16u + j * 4u + b;
                                    const int32_t v = int32_t(int8_t(uint8_t(word >> (8u * b))));
                                    const float want = ref[size_t(row) * D + k];
                                    VJ_CHECK(std::fabs(want - psc * float(v)) <= 1e-6f * (1.0f + std::fabs(want)));
                                }
                            }
                        }
                    }
                }
            }
            for (uint32_t tid = 0; tid < IL_WG64; ++tid) {   // 1 row x 8 tokens x {gate, up}
                const uint32_t arow = il_acc_row64(tid);
                VJ_CHECK((tid & 31u) == (arow & 31u));       // cohort lanes reduce over their own 32 rows
                for (uint32_t g = 0; g < 4u; ++g) {
                    const uint32_t xoff = g * 4u, xsi = g >> 1;
                    for (uint32_t mat = 0; mat < 2u; ++mat) {
                        const float sc = lds_wsc[size_t(mat) * IL_ROWS64 * 4u + arow * 4u + g];
                        const uint32_t* ww = &lds_w[size_t(mat) * IL_ROWS64 * IL_WS + arow * IL_WS + xoff];
                        for (uint32_t j = 0; j < IL_ACC64; ++j) {
                            const uint32_t tok = il_acc_tok64(tid, j);
                            const uint32_t* xw = &lds_x[tok * IL_WS + xoff];
                            const int32_t dp = il_dot4x4(ww[0], ww[1], ww[2], ww[3], xw[0], xw[1], xw[2], xw[3]);
                            float& a = acc[size_t(tid) * 2u * IL_ACC64 + mat * IL_ACC64 + j];
                            a = std::fma(sc * lds_xsc[tok * 2u + xsi], float(dp), a);
                            if (g == 0u)   // one accumulating thread per (row, token, matrix)
                                VJ_CHECK(++acc_cov[(size_t(arow) * IL_TOKENS + tok) * 2u + mat] == 1u);
                        }
                    }
                }
            }
            // every (row, unit) decode slot and every (row, token, matrix) accumulator is written by
            // exactly ONE thread of the 512-thread work item (no cross-cohort overwrite).
            for (uint32_t i = 0; i < dec_cov.size(); ++i) VJ_CHECK(dec_cov[i] == 1u);
            for (uint32_t i = 0; i < acc_cov.size(); ++i) VJ_CHECK(acc_cov[i] == 1u);
            ++out.steps;
        }
        for (uint32_t tid = 0; tid < IL_WG64; ++tid) {   // SwiGLU + block-32 quant; a cohort IS one block
            const uint32_t arow = il_acc_row64(tid);
            for (uint32_t j = 0; j < IL_ACC64; ++j) {
                const uint32_t tok = il_acc_tok64(tid, j);
                const float gv = acc[size_t(tid) * 2u * IL_ACC64 + 0u * IL_ACC64 + j];
                const float uv = acc[size_t(tid) * 2u * IL_ACC64 + 1u * IL_ACC64 + j];
                const float y = (gv / (1.0f + std::exp(-gv))) * uv;
                float amax = std::fabs(y);
                for (uint32_t lane = 0; lane < 32u; ++lane) {   // same token, the cohort's 32 rows
                    if (lane == (tid & 31u)) continue;
                    const uint32_t other = (tid & ~31u) + lane;
                    const float ov = acc[size_t(other) * 2u * IL_ACC64 + 0u * IL_ACC64 + j];
                    const float ou = acc[size_t(other) * 2u * IL_ACC64 + 1u * IL_ACC64 + j];
                    amax = std::max(amax, std::fabs((ov / (1.0f + std::exp(-ov))) * ou));
                }
                const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                const int32_t qv = std::max(-127, std::min(127, int32_t(std::nearbyint(y * inv))));
                if (tok < tcnt) {
                    const size_t dst = size_t(ms[tok]) * lff + rt * IL_ROWS64 + arow;
                    out.q8[dst] = int8_t(qv);
                    out.g_acc[dst] = gv;
                    out.u_acc[dst] = uv;
                    ++out.q8_writer[dst];
                    if ((tid & 31u) == 0u) {
                        const size_t dd = size_t(ms[tok]) * (lff / 32u) + rt * 2u + (arow >> 5);
                        out.d[dd] = amax * (1.0f / 127.0f);
                        ++out.d_writer[dd];
                    }
                }
            }
        }
    }
    return total;
}

uint32_t dn_model64(const DnFx& fx, const GuTables& tb, const Act& mid, const std::vector<float>& mid_d,
                    const std::vector<float>& ew, DnResult& out) {
    const uint32_t D = fx.D, rt_cnt = D / IL_ROWS64, step_cnt = IL_H / IL_KSTEP;
    out.part.assign(size_t(mid.tokens) * D, 0.0f);
    out.writer.assign(out.part.size(), 0u);
    std::vector<uint16_t> pref(257, 0), next(257, 0);
    for (uint32_t t = 0; t < 256u; ++t) pref[t + 1] = uint16_t(il_items_per_expert64(tb.cnt[t], rt_cnt));
    for (uint32_t step = 1; step < 256u; step <<= 1) {
        for (uint32_t t = 0; t < 256u; ++t)
            next[t + 1] = uint16_t(pref[t + 1] + ((t >= step) ? pref[t + 1 - step] : 0u));
        pref.swap(next);
    }
    const uint32_t total = pref[256];

    std::vector<uint32_t> lds_w(size_t(IL_ROWS64) * IL_WS, 0xCDCDCDCDu);
    std::vector<uint32_t> lds_x(size_t(IL_TOKENS) * IL_WS, 0xCDCDCDCDu);
    std::vector<float> lds_wsc(size_t(IL_ROWS64) * 2u, -1.0f), lds_xsc(size_t(IL_TOKENS) * 2u, -1.0f);
    std::vector<uint32_t> ms(IL_TOKENS, 0u);
    std::vector<float> acc(size_t(IL_WG64) * IL_ACC64, 0.0f);
    std::vector<uint32_t> dec_cov(size_t(IL_ROWS64) * 2u, 0u);

    for (uint32_t item = 0; item < total; ++item) {
        uint32_t lo = 0, hi = 256;
        while (hi - lo > 1u) {
            const uint32_t m2 = (lo + hi) >> 1;
            if (pref[m2] <= item) lo = m2; else hi = m2;
        }
        const uint32_t e = lo, local = item - pref[e], cnt = tb.cnt[e], off = tb.off[e];
        const uint32_t tt = local / rt_cnt, rt = local - tt * rt_cnt;
        const uint32_t t0 = tt * IL_TOKENS, tcnt = std::min<uint32_t>(IL_TOKENS, cnt - t0);
        for (uint32_t t = 0; t < IL_TOKENS; ++t) ms[t] = tb.sms[off + t0 + std::min(t, tcnt - 1u)];
        std::fill(acc.begin(), acc.end(), 0.0f);
        for (uint32_t s = 0; s < step_cnt; ++s) {
            std::fill(dec_cov.begin(), dec_cov.end(), 0u);
            for (uint32_t tid = 0; tid < IL_ROWS64 * 2u; ++tid) {   // decode 64 rows x 2 lanes
                const uint32_t drow = tid & 63u, dhf = (tid >> 6) & 1u;
                const uint32_t lane16 = 2u * s + dhf;
                const uint32_t row = e * D + rt * IL_ROWS64 + drow;
                const uint8_t* mrow = fx.merged.data() + size_t(row) * IL_IQ3_ROW;
                uint32_t qw[2] = {0u, 0u}, aux = 0, d2 = 0;
                std::memcpy(qw, mrow + il_iq3_q_off(lane16), 8);
                std::memcpy(&aux, mrow + il_iq3_aux_off(lane16), 4);
                std::memcpy(&d2, mrow + il_iq3_d_off(), 4);
                const IlIq3Unit u = il_iq3_unit(qw, aux, d2, lane16, ggml_tables::iq3xxs_grid);
                uint32_t* dst = &lds_w[drow * IL_WS + dhf * 8u];
                for (uint32_t k = 0; k < 8u; ++k) dst[k] = u.w[k];
                lds_wsc[drow * 2u + dhf] = u.scale;
                VJ_CHECK(++dec_cov[size_t(drow) * 2u + dhf] == 1u);
            }
            for (uint32_t tid = IL_ROWS64 * 2u; tid < IL_WG64 - IL_ROWS64; ++tid) {   // 0..383 stage
                const uint32_t u = tid - IL_ROWS64 * 2u;
                if (u < IL_TOKENS * 4u) {
                    const uint32_t tok = u >> 2, quarter = u & 3u;
                    if (tok < tcnt) std::memcpy(&lds_x[tok * IL_WS + quarter * 4u],
                                                &mid.q[size_t(ms[tok]) * IL_H + s * IL_KSTEP + quarter * 16u], 16);
                } else if (u < IL_TOKENS * 5u) {
                    const uint32_t tok = u - IL_TOKENS * 4u;
                    if (tok < tcnt) {
                        const uint32_t r = ms[tok];
                        lds_xsc[tok * 2u + 0u] = ew[r] * mid_d[size_t(r) * (IL_H / 32u) + s * 2u + 0u];
                        lds_xsc[tok * 2u + 1u] = ew[r] * mid_d[size_t(r) * (IL_H / 32u) + s * 2u + 1u];
                    }
                }
            }
            for (uint32_t i = 0; i < dec_cov.size(); ++i) VJ_CHECK(dec_cov[i] == 1u);
            for (uint32_t tid = 0; tid < IL_WG64; ++tid) {   // 1 row x 8 tokens, one chain per 32-block
                const uint32_t arow = il_acc_row64(tid);
                for (uint32_t b = 0; b < 2u; ++b) {
                    const float sc = lds_wsc[arow * 2u + b];
                    const uint32_t* wr = &lds_w[arow * IL_WS + b * 8u];
                    for (uint32_t j = 0; j < IL_ACC64; ++j) {
                        const uint32_t tok = il_acc_tok64(tid, j);
                        const int32_t dp = il_dot8x8(wr, &lds_x[tok * IL_WS + b * 8u]);
                        float& a = acc[size_t(tid) * IL_ACC64 + j];
                        a = std::fma(sc * lds_xsc[tok * 2u + b], float(dp), a);
                    }
                }
            }
        }
        for (uint32_t tid = 0; tid < IL_WG64; ++tid) {
            const uint32_t arow = il_acc_row64(tid);
            for (uint32_t j = 0; j < IL_ACC64; ++j) {
                const uint32_t tok = il_acc_tok64(tid, j);
                if (tok >= tcnt) continue;
                const size_t dst = size_t(ms[tok]) * D + rt * IL_ROWS64 + arow;
                out.part[dst] = acc[size_t(tid) * IL_ACC64 + j];
                ++out.writer[dst];
            }
        }
    }
    return total;
}

} // namespace

// Independent Down check shared by both work-item shapes: exact per-output write coverage, the
// part dot chain against a double oracle, and the 8-slot combine.
static void dn_oracle(const DnFx& fx, const GuTables& tb, const Act& mid, const std::vector<float>& mid_d,
                      const std::vector<float>& ew, const DnResult& res, uint32_t tokens, uint32_t total,
                      const char* tag) {
    const uint32_t D = fx.D, routes = tokens * 8u;
    std::vector<uint8_t> seen(routes, 0);
    std::vector<uint32_t> rex(routes, 0u);
    for (uint32_t e = 0; e < 256u; ++e)
        for (uint32_t t = 0; t < tb.cnt[e]; ++t) {
            const uint32_t r = tb.sms[tb.off[e] + t];
            VJ_CHECK(seen[r] == 0u);
            seen[r] = 1;
            rex[r] = e;
        }
    for (uint32_t t = 0; t < tokens; ++t)
        for (uint32_t row = 0; row < D; ++row)
            VJ_CHECK(res.writer[size_t(t) * D + row] == (seen[t] ? 1u : 0u));
    double worst = 0.0;
    for (uint32_t e = 0; e < 256u; ++e) {
        for (uint32_t t = 0; t < tb.cnt[e]; ++t) {
            const uint32_t r = tb.sms[tb.off[e] + t];
            for (uint32_t row = 0; row < D; ++row) {
                const size_t grow = size_t(rex[r] * D + row) * IL_H;
                double expect = 0.0, absum = 0.0;
                for (uint32_t k = 0; k < IL_H; ++k) {
                    const double h = double(mid.q[size_t(r) * IL_H + k]) * double(mid_d[size_t(r) * (IL_H / 32u) + k / 32u]);
                    const double term = double(fx.ref[grow + k]) * h;
                    expect += term;
                    absum += std::fabs(term);
                }
                expect *= double(ew[r]);
                absum *= std::fabs(double(ew[r]));
                const double got = double(res.part[size_t(r) * D + row]);
                worst = std::max(worst, std::fabs(got - expect) / (1.0 + absum));
                VJ_CHECK(std::fabs(got - expect) <= 2e-5 * absum + 1e-5);
            }
        }
    }
    std::vector<float> sh_down(size_t(tokens) * D);
    for (size_t i = 0; i < sh_down.size(); ++i) sh_down[i] = float(int(rnd() % 2001u) - 1000) * 1e-4f;
    double worst_c = 0.0;
    for (uint32_t t = 0; t < tokens; ++t) {
        for (uint32_t row = 0; row < D; ++row) {
            float acc = 0.0f;
            for (uint32_t s = 0; s < 8u; ++s) acc += res.part[size_t(t * 8u + s) * D + row];
            acc += sh_down[size_t(t) * D + row];
            double expect = double(sh_down[size_t(t) * D + row]);
            double absum = std::fabs(expect);
            for (uint32_t s = 0; s < 8u; ++s) {
                const uint32_t r = t * 8u + s;
                const size_t grow = size_t(rex[r] * D + row) * IL_H;
                for (uint32_t k = 0; k < IL_H; ++k) {
                    const double h = double(mid.q[size_t(r) * IL_H + k]) * double(mid_d[size_t(r) * (IL_H / 32u) + k / 32u]);
                    const double term = double(fx.ref[grow + k]) * h * double(ew[r]);
                    expect += term;
                    absum += std::fabs(term);
                }
            }
            worst_c = std::max(worst_c, std::fabs(double(acc) - expect) / (1.0 + absum));
            VJ_CHECK(std::fabs(double(acc) - expect) <= 3e-5 * absum + 1e-5);
        }
    }
    std::printf("[OK] Down %s D=%u: %u items, part err %.2e, combine err %.2e, coverage exact\n",
                tag, D, total, worst, worst_c);
}

static uint32_t lff_of(const GuFx& fx) { return fx.lff; }

int main() {
    check_decode_tables();

    // ---- production-shaped Gate+Up case: D = 3072 (nib 96), 1 row tile, 65 + 63 routes ----
    {
        const GuFx fx(2, 64, 3072);
        GuTables tb;
        tb.cnt[0] = 65; tb.cnt[1] = 63;
        tb.off[0] = 0; tb.off[1] = 65;
        tb.sms.resize(128);
        for (uint32_t i = 0; i < 128u; ++i) tb.sms[i] = i;   // 16 tokens x 8 slots
        const Act act(128, 3072);
        GuResult res;
        const uint32_t total = gu_model(fx, tb, act, res, true);
        const uint32_t rt_cnt = lff_of(fx) / IL_ROWS;
        VJ_CHECK(total == (2u + 1u) * rt_cnt);               // ceil(65/64) + ceil(63/64) token tiles
        gu_oracle(fx, tb, act, res, true);
        // 256 decode units per step per pass (32 rows x 8 (ib, matrix, half) units) x row tiles.
        VJ_CHECK(res.steps == res.passes * (fx.nib / 2u));
        VJ_CHECK(res.decodes == res.passes * (fx.nib / 2u) * 256u);
        uint64_t passes_expect = 0, prod = 0;
        for (uint32_t e = 0; e < 2u; ++e) {
            for (uint32_t t0 = 0; t0 < tb.cnt[e]; t0 += IL_TOKENS) {
                const uint32_t tiles = std::min<uint32_t>(IL_TOKENS, tb.cnt[e] - t0);
                passes_expect += rt_cnt * ((tiles + IL_TPASS - 1u) / IL_TPASS);
            }
            prod += uint64_t((tb.cnt[e] + 15u) / 16u) * fx.nib * 128u * rt_cnt;   // 16-token tiles, 32 rows each
        }
        VJ_CHECK(res.passes == passes_expect);
        std::printf("[OK] decode-once: %llu units for 128 routes vs %llu in the 16-token tiled path (%.2fx)\n",
                    (unsigned long long)res.decodes, (unsigned long long)prod, double(prod) / double(res.decodes));
        VJ_CHECK(res.decodes <= prod);   // never more decode work than the 16-token tiled path
    }

    // ---- one full 64-token tile: exactly half the decode units of the 16-token tiled path ----
    {
        const GuFx fx(1, 32, 256);   // one 32-row column tile
        GuTables tb;
        tb.cnt[0] = 64;
        tb.sms.resize(64);
        for (uint32_t i = 0; i < 64u; ++i) tb.sms[i] = i;
        const Act act(64, 256);
        GuResult res;
        const uint32_t total = gu_model(fx, tb, act, res, false);
        VJ_CHECK(total == 1u && res.passes == 2u);
        gu_oracle(fx, tb, act, res, false);
        const uint64_t prod = uint64_t(4u) * fx.nib * 128u;   // four 16-token tiles x nib x 128 units (32 rows)
        std::printf("[OK] full tile decode-once: %llu units vs %llu tiled (exactly %.2fx)\n",
                    (unsigned long long)res.decodes, (unsigned long long)prod, double(prod) / double(res.decodes));
        VJ_CHECK(res.decodes * 2u == prod);
    }

    // ---- narrow row tiles, short K, partial tile tails ----
    for (auto shape : {std::pair<uint32_t, uint32_t>{256u, 64u}, {512u, 128u}, {256u, 512u}}) {
        const GuFx fx(3, shape.second, shape.first);
        GuTables tb;
        tb.cnt[0] = 129; tb.cnt[1] = 33; tb.cnt[2] = 7;
        tb.off[0] = 0; tb.off[1] = 129; tb.off[2] = 162;
        tb.sms.resize(169);
        for (uint32_t i = 0; i < 169u; ++i) tb.sms[i] = i;   // 22 tokens x 8 slots (169 <= 176)
        const Act act(176, shape.first);
        GuResult res;
        const uint32_t total = gu_model(fx, tb, act, res, shape.first == 256u && shape.second == 64u);
        gu_oracle(fx, tb, act, res, false);
        VJ_CHECK(res.decodes == res.passes * (fx.nib / 2u) * 256u);
        const uint32_t rt_cnt = fx.lff / IL_ROWS;
        const uint32_t want_items = 3u * rt_cnt + 1u * rt_cnt + 1u * rt_cnt;   // ceil(129/64), 33, 7
        VJ_CHECK(total == want_items);
        std::printf("[OK] Gate+Up D=%u lff=%u: %u items, %llu passes\n", fx.D, fx.lff, total,
                    (unsigned long long)res.passes);
    }

    // ---- THE requested single 64x64x64 work item: one item IS 64 rows x 64 tokens x K step 64 ----
    {
        const GuFx fx(1, 512, 3072);                        // production local_ff = 512, D = 3072
        GuTables tb;
        tb.cnt[0] = 64;
        tb.sms.resize(64);
        for (uint32_t i = 0; i < 64u; ++i) tb.sms[i] = i;
        const Act act(64, 3072);
        GuResult res;
        const uint32_t total = gu_model64(fx, tb, act, res, true);
        VJ_CHECK(total == il_row_tiles64(512u));            // 8 items = 8 x 64 rows, one token tile
        VJ_CHECK(res.passes == total);                      // ONE token pass per work item
        VJ_CHECK(res.steps == total * (fx.nib / 2u));
        VJ_CHECK(res.decodes == total * (fx.nib / 2u) * IL_WG64);   // one 16-weight unit per thread
        gu_oracle(fx, tb, act, res, true);                  // every (route, row) written exactly once
        // Same 64 rows x 64 tokens through 16-token production tiles: each weight is decoded four times.
        const uint64_t prod = uint64_t(64u / 16u) * fx.nib * 128u * 2u * il_row_tiles64(512u);
        std::printf("[OK] 64x64x64 single work item: %llu decode units vs %llu production (%.2fx fewer)\n",
                    (unsigned long long)res.decodes, (unsigned long long)prod,
                    double(prod) / double(res.decodes));
        VJ_CHECK(prod == res.decodes * 4u);
    }

    // ---- the 64-row item halves the decode units of the 32-row diagnostic on the same rows ----
    {
        const GuFx fx(1, 64, 256);
        GuTables tb;
        tb.cnt[0] = 64;
        tb.sms.resize(64);
        for (uint32_t i = 0; i < 64u; ++i) tb.sms[i] = i;
        const Act act(64, 256);
        GuResult r32;
        const uint32_t n32 = gu_model(fx, tb, act, r32, false);
        gu_oracle(fx, tb, act, r32, false);
        GuResult r64;
        const uint32_t n64 = gu_model64(fx, tb, act, r64, false);
        gu_oracle(fx, tb, act, r64, false);
        VJ_CHECK(n32 == 2u && n64 == 1u);                    // two 32-row tiles -> one 64-row item
        VJ_CHECK(r64.decodes * 2u == r32.decodes);
        std::printf("[OK] 64-row item vs 32-row tiles: %llu vs %llu decode units (exactly 2x fewer)\n",
                    (unsigned long long)r64.decodes, (unsigned long long)r32.decodes);
    }

    // ---- 64-row items: token tails, row tiles, short K, empty experts ----
    for (auto shape : {std::pair<uint32_t, uint32_t>{256u, 64u}, {512u, 128u}, {256u, 256u}}) {
        const GuFx fx(3, shape.second, shape.first);
        GuTables tb;
        tb.cnt[0] = 129; tb.cnt[1] = 33; tb.cnt[2] = 7;
        tb.off[0] = 0; tb.off[1] = 129; tb.off[2] = 162;
        tb.sms.resize(169);
        for (uint32_t i = 0; i < 169u; ++i) tb.sms[i] = i;
        const Act act(176, shape.first);
        GuResult res;
        const uint32_t total = gu_model64(fx, tb, act, res, shape.first == 256u && shape.second == 64u);
        gu_oracle(fx, tb, act, res, false);
        const uint32_t rt_cnt = il_row_tiles64(fx.lff);
        const uint32_t want_items = 3u * rt_cnt + 1u * rt_cnt + 1u * rt_cnt;   // ceil(129/64), 33, 7
        VJ_CHECK(total == want_items);
        VJ_CHECK(res.passes == want_items);
        VJ_CHECK(res.decodes == want_items * (fx.nib / 2u) * IL_WG64);
        std::printf("[OK] 64-row item D=%u lff=%u: %u items, %llu decode units, coverage exact\n",
                    fx.D, fx.lff, total, (unsigned long long)res.decodes);
    }

    // ---- Down: 32-row (diagnostic) and 64-row work items, D = 64 / 128 / 512, 64/32-token tails ----
    for (uint32_t D : {64u, 128u, 512u}) {
        const DnFx fx(2, D);
        GuTables tb;
        tb.cnt[0] = 65; tb.cnt[1] = 63;
        tb.off[0] = 0; tb.off[1] = 65;
        const uint32_t tokens = 16;                        // 128 routes = 16 tokens x 8 slots
        const uint32_t routes = tokens * 8u;
        tb.sms.resize(routes);
        for (uint32_t i = 0; i < routes; ++i) tb.sms[i] = i;
        Act mid(routes, IL_H);                             // one INT8 hidden row per route
        std::vector<float> mid_d(size_t(routes) * (IL_H / 32u)), ew(routes, 0.0f);
        for (uint32_t t = 0; t < routes; ++t) {
            for (uint32_t b = 0; b < IL_H / 32u; ++b) {
                float amax = 0.0f;
                for (uint32_t k = 0; k < 32u; ++k)
                    amax = std::max(amax, std::fabs(float(mid.q[size_t(t) * IL_H + b * 32u + k])));
                mid_d[size_t(t) * (IL_H / 32u) + b] = amax * (1.0f / 127.0f);
            }
            ew[t] = 0.0625f + 0.125f * float(t % 8u);
        }
        DnResult res32;
        const uint32_t total32 = dn_model(fx, tb, mid, mid_d, ew, res32);
        dn_oracle(fx, tb, mid, mid_d, ew, res32, tokens, total32, "32-row tile");
        DnResult res64;
        const uint32_t total64 = dn_model64(fx, tb, mid, mid_d, ew, res64);
        dn_oracle(fx, tb, mid, mid_d, ew, res64, tokens, total64, "64-row item");
        VJ_CHECK(total32 == (D / IL_ROWS) * 3u && total64 == (D / IL_ROWS64) * 3u);
        std::printf("[OK] Down work items: D=%u -> %u x 32-row vs %u x 64-row (one item per 64 rows)\n",
                    D, total32, total64);
    }

    // ---- work-item enumeration / layout offsets ----
    for (uint32_t rt = 1; rt <= 8u; ++rt) {
        for (uint32_t cnt : {0u, 1u, 31u, 32u, 33u, 63u, 64u, 65u, 127u, 128u, 129u}) {
            VJ_CHECK(il_items_per_expert(cnt, rt) == ((cnt + IL_TOKENS - 1u) / IL_TOKENS) * rt);
            VJ_CHECK(il_row_tiles(rt * 32u) == rt);
        }
    }
    VJ_CHECK(il_gu_row_bytes(96u) == 1968u && il_gu_row_bytes(8u) == 164u);
    VJ_CHECK(il_row_tiles(32u) == 1u && il_row_tiles(512u) == 16u);
    VJ_CHECK(il_gu_hs_word(96u, 0u) == 384u && il_gu_d_word(96u, 0u) == 480u);
    VJ_CHECK(il_gu_d_word(8u, 0u) == 40u && il_gu_d_word(8u, 8u) == 41u);
    VJ_CHECK(IL_IQ3_ROW == 13u * 16u && il_iq3_aux_off(15u) == 188u && il_iq3_d_off() == 192u);
    // The device kernels derive these roles with bit operations; the header helpers (replayed by the
    // models) must agree for every thread of the 512-thread 64-row work item.
    for (uint32_t tid = 0; tid < IL_WG64; ++tid) {
        VJ_CHECK(il_dec_row64(tid) == (tid & 63u));
        VJ_CHECK(il_dec_unit64(tid) == (tid >> 6));
        VJ_CHECK(il_acc_cohort64(tid) == (tid >> 8));
        VJ_CHECK(il_acc_row64(tid) == (tid >> 8) * 32u + (tid & 31u));
        VJ_CHECK(il_acc_slot64(tid) == ((tid >> 5) & 7u));
        for (uint32_t j = 0; j < IL_ACC64; ++j) VJ_CHECK(il_acc_tok64(tid, j) == (((tid >> 5) & 7u) * 8u + j));
    }
    VJ_CHECK(il_row_tiles64(512u) == 8u && il_row_tiles64(64u) == 1u);
    VJ_CHECK(il_items_per_expert64(65u, 8u) == 16u && il_items_per_expert64(0u, 8u) == 0u);
    VJ_CHECK(IL_WG64 == 512u && IL_ROWS64 == 64u && IL_WG64 * IL_ACC64 == IL_ROWS64 * IL_TOKENS);   // 4096 (row, token) pairs
    std::printf("[OK] layout offsets, work-item enumeration and 512-thread role law\n");
    std::puts("PASS: single 64x64x64 (and 32-row diagnostic) dequant-once-to-LDS MoE prefill model "
              "matches production IQ2_S/IQ3_XXS decoding and accumulation on merged/compact layouts");
    return 0;
}
