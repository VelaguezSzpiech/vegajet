// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// VJ_Q4_DOT8 lossless re-code (core/q4_dot8_repack.*): random IQ3_XXS / IQ2_S blocks built from the real ggml
// tables -> repack -> device-identical CPU decode (v_perm model) must equal the IQ integers exactly, the float
// weights must equal the ggml reference decoder bit for bit, and the emulated 9b / 9c row sums must equal the
// emulated DOT8 tile sums bit for bit. Plus layout and promotion-planner checks. Pure CPU.
#include "core/ggml_iq_tables.hpp"
#include "core/q4_dot8_repack.hpp"
#include "core/quant_repacker.hpp"
#include "ops/kernels/vj_moe_dot8.hiph"
#include "tests/test_common.hpp"

#include <cmath>
#include <random>

using namespace vegajet;
namespace TB = vegajet::ggml_tables;

// The kernel header's literal tables must be the solved ones.
static_assert(dot8_dev::kIq3ALo == kDot8Iq3Perm.a_lo && dot8_dev::kIq3AHi == kDot8Iq3Perm.a_hi, "IQ3 A table");
static_assert(dot8_dev::kIq3BLo == kDot8Iq3Perm.b_lo && dot8_dev::kIq3BHi == kDot8Iq3Perm.b_hi, "IQ3 B table");
static_assert(dot8_dev::kIq2Lo == kDot8Iq2PermLo && dot8_dev::kIq2Hi == kDot8Iq2PermHi, "IQ2 table");

namespace {

std::mt19937_64 rng(0x5EED1234ULL);
uint32_t rnd(uint32_t n) { return static_cast<uint32_t>(rng() % n); }

uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
uint32_t rd32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
uint16_t rd16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }

uint16_t random_fp16() {
    // finite, both signs; mostly normal, sometimes subnormal
    const uint32_t sign = rnd(2) << 15;
    const uint32_t exp = (rnd(16) == 0) ? 0u : 1u + rnd(29);
    return static_cast<uint16_t>(sign | (exp << 10) | rnd(1024));
}

void random_iq3xxs_block(uint8_t* b) {
    const uint16_t d = random_fp16();
    std::memcpy(b, &d, 2);
    for (int i = 0; i < 64; ++i) b[2 + i] = static_cast<uint8_t>(rnd(256));
    for (int ib = 0; ib < 8; ++ib) {
        const uint32_t aux = rnd(1u << 28) | (rnd(16) << 28);
        std::memcpy(b + 66 + 4 * ib, &aux, 4);
    }
}

void random_iq2s_block(uint8_t* b) {
    const uint16_t d = random_fp16();
    std::memcpy(b, &d, 2);
    for (int i = 2; i < 82; ++i) b[i] = static_cast<uint8_t>(rnd(256));
}

// Independent IQ3_XXS reference: weight value (int) and scale, sign from the 7-bit field + parity (as 9c does),
// not from ksigns_iq2xs, so the table path in the encoder is cross-checked.
int iq3_ref_weight(const uint8_t* blk, uint32_t idx /*0..255*/) {
    const uint32_t ib = idx / 32u, w = idx % 32u, l = w / 8u, j = w % 8u;
    const uint32_t aux = rd32(blk + 66 + 4 * ib);
    uint32_t s = (aux >> (7u * l)) & 127u;
    s |= static_cast<uint32_t>(__builtin_popcount(s) & 1) << 7;
    const uint8_t q = blk[2 + 8 * ib + 2 * l + (j / 4u)];
    const int g = static_cast<int>((TB::iq3xxs_grid[q] >> (8u * (j % 4u))) & 0xFFu);
    return ((s >> j) & 1u) ? -g : g;
}
uint32_t iq3_ref_scale(const uint8_t* blk, uint32_t ib) { return rd32(blk + 66 + 4 * ib) >> 28; }

int iq2_ref_weight(const uint8_t* blk, uint32_t idx) {
    const uint32_t ib = idx / 32u, w = idx % 32u, l = w / 8u, j = w % 8u;
    const uint8_t qs = blk[2 + 4 * ib + l];
    const uint8_t sg = blk[2 + 32 + 4 * ib + l];
    const uint8_t qh = blk[66 + ib];
    const uint32_t gi = qs | ((static_cast<uint32_t>(qh) >> (2u * l) & 3u) << 8);
    const int g = static_cast<int>((TB::iq2s_grid[gi] >> (8u * j)) & 0xFFu);
    return ((sg >> j) & 1u) ? -g : g;
}
uint32_t iq2_ref_scale(const uint8_t* blk, uint32_t ib, uint32_t half) {
    const uint8_t s = blk[74 + ib];
    return half ? (s >> 4) : (s & 0xFu);
}

// --- emulation of the device integer helpers ---
int32_t sdot4(uint32_t a, uint32_t b, int32_t c) {
    for (int i = 0; i < 4; ++i)
        c += static_cast<int32_t>(static_cast<int8_t>(a >> (8 * i))) * static_cast<int32_t>(static_cast<int8_t>(b >> (8 * i)));
    return c;
}
uint32_t apply_4signs(uint32_t mag, uint32_t sign4) {  // vj_ornith_ops.hip apply_4signs_i8x4
    const uint32_t add = (sign4 * 0x00204081u) & 0x01010101u;
    const uint32_t xorm = (add << 8) - add;
    return (mag ^ xorm) + add;
}
float sum16(const float* v) {  // butterfly xor 8,4,2,1 as in 9b/9c, lane 0 result
    float a[16];
    std::memcpy(a, v, sizeof(a));
    for (int m : {8, 4, 2, 1}) {
        float b[16];
        for (int l = 0; l < 16; ++l) b[l] = a[l] + a[l ^ m];
        std::memcpy(a, b, sizeof(a));
    }
    return a[0];
}

} // namespace

int main() {
    // ---------------------------------------------------------------- level sets from the real tables
    {
        bool seen3[256] = {}, seen2[256] = {};
        for (uint32_t i = 0; i < 256; ++i)
            for (int b = 0; b < 4; ++b) seen3[(TB::iq3xxs_grid[i] >> (8 * b)) & 0xFF] = true;
        for (uint32_t i = 0; i < 1024; ++i)
            for (int b = 0; b < 8; ++b) seen2[(TB::iq2s_grid[i] >> (8 * b)) & 0xFF] = true;
        int n3 = 0, n2 = 0;
        for (int v = 0; v < 256; ++v) { n3 += seen3[v]; n2 += seen2[v]; }
        VJ_CHECK(n3 == 8 && n2 == 3);
        for (int k = 0; k < 8; ++k) VJ_CHECK(seen3[static_cast<uint8_t>(kDot8Iq3Mag[k])]);
        for (int k = 0; k < 3; ++k) VJ_CHECK(seen2[static_cast<uint8_t>(kDot8Iq2Mag[k])]);
        VJ_CHECK(seen3[62] && !seen3[60]);  // the non-affine top level
    }

    // ---------------------------------------------------------------- v_perm model + exhaustive 4-code decode
    VJ_CHECK(dot8_perm_b32(0x88776655u, 0x44332211u, 0x07060100u) == 0x88772211u);  // values from clang constant folding
    VJ_CHECK(dot8_perm_b32(0x80706050u, 0x40302010u, 0x0B0A0908u) == 0xFF000000u);
    VJ_CHECK(dot8_perm_b32(0x0070FF50u, 0x80008000u, 0x0B0A0908u) == 0x00FFFFFFu);
    VJ_CHECK(dot8_perm_b32(0x11223344u, 0x55667788u, 0x0F0E0D0Cu) == 0xFFFFFF00u);
    for (uint32_t s = 0; s < 65536u; ++s) {
        const uint32_t c[4] = {s & 15u, (s >> 4) & 15u, (s >> 8) & 15u, s >> 12};
        uint32_t w = 0, w2 = 0;  // codes in the low nibbles (w) / high nibbles (w2), random junk in the other nibble
        for (int i = 0; i < 4; ++i) {
            w |= (c[i] | (rnd(16) << 4)) << (8 * i);
            w2 |= ((c[i] << 4) | rnd(16)) << (8 * i);
        }
        const uint32_t lo = dot8_decode4_iq3(w & 0x0F0F0F0Fu);
        const uint32_t hi = dot8_decode4_iq3((w2 >> 4) & 0x0F0F0F0Fu);  // the kernel's high-nibble path
        for (int i = 0; i < 4; ++i) {
            VJ_CHECK(static_cast<int8_t>(lo >> (8 * i)) == dot8_iq3_level(c[i]));
            VJ_CHECK(static_cast<int8_t>(hi >> (8 * i)) == dot8_iq3_level(c[i]));
        }
        if (c[0] < 6 && c[1] < 6 && c[2] < 6 && c[3] < 6) {
            const uint32_t v = dot8_decode4_iq2(w);  // masks 0x07 itself; other-nibble junk must not leak
            const uint32_t v2 = dot8_decode4_iq2(w2 >> 4);
            for (int i = 0; i < 4; ++i) VJ_CHECK(static_cast<int8_t>(v >> (8 * i)) == dot8_iq2_level(c[i]));
            for (int i = 0; i < 4; ++i) VJ_CHECK(static_cast<int8_t>(v2 >> (8 * i)) == dot8_iq2_level(c[i]));
        }
    }
    // pack/unpack round trip
    {
        uint8_t codes[32], packed[16], back[32];
        for (int i = 0; i < 32; ++i) codes[i] = static_cast<uint8_t>(rnd(16));
        dot8_pack_codes(codes, packed);
        dot8_unpack_codes(packed, back);
        VJ_CHECK(std::memcmp(codes, back, 32) == 0);
    }

    // ---------------------------------------------------------------- layouts / TCC
    {
        VJ_CHECK(dot8_odd256_stride(256) == 256 && dot8_odd256_stride(257) == 768 && dot8_odd256_stride(1536) == 1792);
        VJ_CHECK(dot8_odd256_stride(3072) == 3328 && dot8_odd256_stride(3312) == 3328 && dot8_odd256_stride(1) == 256);
        const Dot8RowLayout dn = dot8_iq3_layout(256ull * 3072ull, 512);
        VJ_CHECK(dn.w_raw_row_bytes == 256 && dn.w_row_stride_bytes == 256 && !dn.scale_embedded);
        VJ_CHECK(dn.s_row_stride_bytes == 16 && dn.read_bytes_per_row() == 272);
        VJ_CHECK(dn.vram_bytes() == 256ull * 3072ull * 272ull);
        VJ_CHECK(dn.read_bytes_per_row() * 8.0 / 512.0 == kDot8Iq3ReadBpw);
        const Dot8RowLayout gu = dot8_gu_iq2s_layout(256ull * 512ull, 3072);
        VJ_CHECK(gu.w_raw_row_bytes == 3072 && gu.s_raw_row_bytes == 240 && gu.w_row_stride_bytes == 3328);
        VJ_CHECK(gu.scale_embedded && gu.s_offset_bytes == 3072 && gu.s_row_stride_bytes == 3328);
        VJ_CHECK(gu.vram_bytes() == 256ull * 512ull * 3328ull && gu.s_plane_bytes() == 0);
        VJ_CHECK(gu.read_bytes_per_row() * 8.0 / (2.0 * 3072.0) == kDot8Iq2ReadBpw);
        for (uint32_t stride : {dn.w_row_stride_bytes, gu.w_row_stride_bytes}) {  // same tile, 16 rows -> 16 channels
            uint32_t seen = 0;
            for (uint32_t r = 0; r < 16; ++r) seen |= 1u << ((static_cast<uint64_t>(r) * stride / 256u) & 15u);
            VJ_CHECK(seen == 0xFFFFu);
            VJ_CHECK((stride / 256u) % 2u == 1u && stride % 16u == 0u);
        }
        std::string err;
        Dot8RowLayout bad = dot8_iq3_layout(4, 384);
        std::vector<uint8_t> tmp(4096);
        VJ_CHECK(!dot8_repack_iq3xxs_rows(tmp.data(), 98, 0, bad, 0, 4, tmp.data(), tmp.data(), &err) && !err.empty());
    }

    const auto& reg = quant::DecoderRegistry::instance();

    // ---------------------------------------------------------------- IQ3_XXS Down: repack -> exact
    {
        // 48 rows of a full 1024-col disk row (4 superblocks); this "GPU" takes the 2nd half (sb_begin = 2).
        const uint32_t rows = 48, full_sb = 4, sb_begin = 2, cols = 512;
        const uint64_t src_stride = full_sb * kIq3xxsBlockBytes;
        std::vector<uint8_t> src(rows * src_stride);
        for (uint32_t r = 0; r < rows; ++r)
            for (uint32_t s = 0; s < full_sb; ++s) random_iq3xxs_block(src.data() + r * src_stride + s * kIq3xxsBlockBytes);
        const Dot8RowLayout lay = dot8_iq3_layout(rows, cols);
        std::vector<uint8_t> wp(lay.w_plane_bytes(), 0xCD), sp(lay.s_plane_bytes(), 0xCD);
        std::string err;
        // two disjoint row ranges, as an OpenMP host loop would do
        VJ_CHECK(dot8_repack_iq3xxs_rows(src.data(), src_stride, sb_begin, lay, 0, 20, wp.data(), sp.data(), &err));
        VJ_CHECK(dot8_repack_iq3xxs_rows(src.data(), src_stride, sb_begin, lay, 20, rows, wp.data(), sp.data(), &err));

        std::vector<int8_t> wi(cols);
        std::vector<float> ref(cols);
        uint64_t n62 = 0;
        for (uint32_t r = 0; r < rows; ++r) {
            const uint8_t* srow = src.data() + r * src_stride + sb_begin * kIq3xxsBlockBytes;
            dot8_decode_row_iq3_int(wp.data() + r * lay.w_row_stride_bytes, cols, wi.data());
            VJ_CHECK(reg.decode_row_f32(GgmlType::IQ3_XXS, srow, ref.data(), cols));
            const uint8_t* sc = sp.data() + r * lay.s_row_stride_bytes;
            for (uint32_t i = 0; i < cols; ++i) {
                const uint8_t* blk = srow + (i / 256u) * kIq3xxsBlockBytes;
                const int w = iq3_ref_weight(blk, i % 256u);
                VJ_CHECK(wi[i] == w);  // integer domain: exact
                n62 += (w == 62 || w == -62);
                const uint32_t b = i / 32u;
                const uint32_t s4 = (rd32(sc + (b / 8u) * 8u) >> (4u * (b % 8u))) & 0xFu;
                const float d = u2f(rd32(sc + (b / 8u) * 8u + 4u));
                VJ_CHECK(s4 == iq3_ref_scale(blk, (i % 256u) / 32u));
                VJ_CHECK(f2u(d) == f2u(quant::fp16_to_f32(rd16(blk))));
                // float domain: the ggml reference decoder's value, bit for bit
                const float db = d * (0.5f + static_cast<float>(s4)) * 0.5f;
                VJ_CHECK(f2u(db * static_cast<float>(w)) == f2u(ref[i]) || (ref[i] == 0.0f && db * static_cast<float>(w) == 0.0f));
            }
        }
        VJ_CHECK(n62 > 0);  // the 62 level was exercised
        VJ_CHECK(sp.size() == rows * 16u);  // stride == raw here: no padding, separate dense scale plane

        // ---- emulated 9c row sum vs emulated DOT8 tile sum, bit for bit (rows = 48 -> D = 6 per "expert")
        const uint32_t D = 6, n_exp = rows / D;
        int32_t ids[8];
        float tw[8], md[8 * 16];
        int8_t mq[8 * 512];
        for (int s = 0; s < 8; ++s) { ids[s] = static_cast<int32_t>(rnd(n_exp)); tw[s] = 0.05f + static_cast<float>(rnd(1000)) * 1e-3f; }
        for (float& v : md) v = static_cast<float>(1 + rnd(5000)) * 1.37e-5f;
        for (int8_t& v : mq) v = static_cast<int8_t>(static_cast<int>(rnd(255)) - 127);
        for (uint32_t row = 0; row < D; ++row) {
            float lanes_ref[16], lanes_new[16];
            for (uint32_t lane = 0; lane < 16; ++lane) {
                float tot_ref = 0.0f, tot_new = 0.0f;
                for (uint32_t slot = 0; slot < 8; ++slot) {
                    const uint32_t er = static_cast<uint32_t>(ids[slot]) * D + row;
                    const uint8_t* srow = src.data() + er * src_stride + sb_begin * kIq3xxsBlockBytes;
                    const uint8_t* blk = srow + (lane / 8u) * kIq3xxsBlockBytes;
                    const uint32_t ib = lane % 8u;
                    const int8_t* x = mq + slot * 512 + lane * 32;
                    uint32_t xw[8];
                    std::memcpy(xw, x, 32);
                    // 9c: split-plane IQ3_XXS with LDS grid + SWAR signs
                    const uint32_t ga = rd32(blk + 66 + 4 * ib);
                    const float gdb = quant::fp16_to_f32(rd16(blk)) * (0.5f + static_cast<float>(ga >> 28)) * 0.5f;
                    int32_t dp = 0;
                    for (uint32_t l = 0; l < 4; ++l) {
                        uint32_t s = (ga >> (7u * l)) & 127u;
                        s |= static_cast<uint32_t>(__builtin_popcount(s) & 1) << 7;
                        dp = sdot4(apply_4signs(TB::iq3xxs_grid[blk[2 + 8 * ib + 2 * l]], s & 0xFu), xw[2 * l], dp);
                        dp = sdot4(apply_4signs(TB::iq3xxs_grid[blk[2 + 8 * ib + 2 * l + 1]], s >> 4), xw[2 * l + 1], dp);
                    }
                    tot_ref = std::fma(tw[slot] * gdb * md[slot * 16 + lane], static_cast<float>(dp), tot_ref);
                    // DOT8: v_perm decode of the packed block + stored f32 d and nibble scale
                    const uint8_t* wb = wp.data() + static_cast<uint64_t>(er) * lay.w_row_stride_bytes + lane * 16u;
                    const uint8_t* sb = sp.data() + static_cast<uint64_t>(er) * lay.s_row_stride_bytes + (lane >> 3) * 8u;
                    int32_t dq = 0;
                    for (uint32_t j = 0; j < 4; ++j) dq = sdot4(dot8_decode4_iq3(rd32(wb + 4 * j) & 0x0F0F0F0Fu), xw[j], dq);
                    for (uint32_t j = 0; j < 4; ++j) dq = sdot4(dot8_decode4_iq3((rd32(wb + 4 * j) >> 4) & 0x0F0F0F0Fu), xw[4 + j], dq);
                    VJ_CHECK(dq == dp);
                    const uint32_t s4 = (rd32(sb) >> ((lane & 7u) * 4u)) & 0xFu;
                    const float gdb2 = u2f(rd32(sb + 4)) * (0.5f + static_cast<float>(s4)) * 0.5f;
                    VJ_CHECK(f2u(gdb2) == f2u(gdb));
                    tot_new = std::fma(tw[slot] * gdb2 * md[slot * 16 + lane], static_cast<float>(dq), tot_new);
                }
                lanes_ref[lane] = tot_ref;
                lanes_new[lane] = tot_new;
            }
            VJ_CHECK(f2u(sum16(lanes_ref)) == f2u(sum16(lanes_new)));
        }
    }

    // ---------------------------------------------------------------- IQ2_S merged Gate+Up: repack -> exact
    {
        // 2 "experts" x 16 rows, full rows of 3072 (12 superblocks); this "GPU" uses sb_begin = 0 on row slices.
        const uint32_t rows = 32, cols = 3072, nsb = cols / 256, nblk = cols / 32;
        const uint64_t src_stride = nsb * kIq2sBlockBytes;
        std::vector<uint8_t> sg(rows * src_stride), su(rows * src_stride);
        for (uint32_t r = 0; r < rows; ++r)
            for (uint32_t s = 0; s < nsb; ++s) {
                random_iq2s_block(sg.data() + r * src_stride + s * kIq2sBlockBytes);
                random_iq2s_block(su.data() + r * src_stride + s * kIq2sBlockBytes);
            }
        const Dot8RowLayout lay = dot8_gu_iq2s_layout(rows, cols);
        VJ_CHECK(lay.scale_embedded);
        std::vector<uint8_t> wp(lay.w_plane_bytes(), 0xCD);
        std::string err;
        VJ_CHECK(dot8_repack_gu_iq2s_rows(sg.data(), su.data(), src_stride, 0, lay, 0, rows, wp.data(), nullptr, &err));

        std::vector<int8_t> wg(cols), wu(cols);
        std::vector<float> rg(cols), ru(cols);
        for (uint32_t r = 0; r < rows; ++r) {
            const uint8_t* row = wp.data() + r * lay.w_row_stride_bytes;
            const uint8_t* sc = row + lay.s_offset_bytes;
            for (uint32_t i = lay.s_offset_bytes + lay.s_raw_row_bytes; i < lay.w_row_stride_bytes; ++i) VJ_CHECK(row[i] == 0);
            dot8_decode_row_iq2_int(row, cols, 32, wg.data());
            dot8_decode_row_iq2_int(row + 16, cols, 32, wu.data());
            const uint8_t* g_src = sg.data() + r * src_stride;
            const uint8_t* u_src = su.data() + r * src_stride;
            VJ_CHECK(reg.decode_row_f32(GgmlType::IQ2_S, g_src, rg.data(), cols));
            VJ_CHECK(reg.decode_row_f32(GgmlType::IQ2_S, u_src, ru.data(), cols));
            for (uint32_t i = 0; i < cols; ++i) {
                const uint8_t* gb = g_src + (i / 256u) * kIq2sBlockBytes;
                const uint8_t* ub = u_src + (i / 256u) * kIq2sBlockBytes;
                VJ_CHECK(wg[i] == iq2_ref_weight(gb, i % 256u));
                VJ_CHECK(wu[i] == iq2_ref_weight(ub, i % 256u));
                const uint32_t b = i / 32u, half = (i % 32u) / 16u;
                const uint32_t sc16 = rd16(sc + 2 * b);
                const uint32_t d2 = rd32(sc + 2 * nblk + 4 * (b / 8u));
                const uint32_t sg4 = half ? ((sc16 >> 4) & 0xF) : (sc16 & 0xF);
                const uint32_t su4 = half ? (sc16 >> 12) : ((sc16 >> 8) & 0xF);
                VJ_CHECK(sg4 == iq2_ref_scale(gb, (i % 256u) / 32u, half));
                VJ_CHECK(su4 == iq2_ref_scale(ub, (i % 256u) / 32u, half));
                VJ_CHECK((d2 & 0xFFFF) == rd16(gb) && (d2 >> 16) == rd16(ub));
                const float dlg = quant::fp16_to_f32(static_cast<uint16_t>(d2 & 0xFFFF)) * (0.5f + static_cast<float>(sg4)) * 0.25f;
                const float dlu = quant::fp16_to_f32(static_cast<uint16_t>(d2 >> 16)) * (0.5f + static_cast<float>(su4)) * 0.25f;
                VJ_CHECK(f2u(dlg * static_cast<float>(wg[i])) == f2u(rg[i]) || (rg[i] == 0.0f && dlg * wg[i] == 0.0f));
                VJ_CHECK(f2u(dlu * static_cast<float>(wu[i])) == f2u(ru[i]) || (ru[i] == 0.0f && dlu * wu[i] == 0.0f));
            }
        }

        // ---- emulated 9b (gate, up) row sums vs emulated DOT8 tile, bit for bit
        int8_t xq[3072];
        float xd[96];
        for (int8_t& v : xq) v = static_cast<int8_t>(static_cast<int>(rnd(255)) - 127);
        for (float& v : xd) v = static_cast<float>(1 + rnd(5000)) * 2.11e-5f;
        for (uint32_t r = 0; r < rows; ++r) {
            float ag_ref[16], au_ref[16], ag_new[16], au_new[16];
            const uint8_t* row = wp.data() + r * lay.w_row_stride_bytes;
            const uint8_t* sc = row + lay.s_offset_bytes;
            for (uint32_t lane = 0; lane < 16; ++lane) {
                float ag = 0.f, au = 0.f, bg = 0.f, bu = 0.f;
                for (uint32_t it = 0; it < 6; ++it) {
                    const uint32_t ib = it * 16 + lane;
                    uint32_t xw[8];
                    std::memcpy(xw, xq + ib * 32, 32);
                    const float xda = xd[ib];
                    for (uint32_t part = 0; part < 2; ++part) {  // 9b: gate then up
                        const uint8_t* src = (part ? su.data() : sg.data()) + r * src_stride + (ib / 8u) * kIq2sBlockBytes;
                        const uint32_t sb = ib % 8u;
                        const float pd = quant::fp16_to_f32(rd16(src)) * 0.25f * xda;
                        const uint8_t scb = src[74 + sb], qh = src[66 + sb];
                        int32_t dpv[2] = {0, 0};
                        for (uint32_t l = 0; l < 4; ++l) {
                            const uint64_t g = TB::iq2s_grid[src[2 + 4 * sb + l] | ((static_cast<uint32_t>(qh) << (8 - 2 * l)) & 0x300u)];
                            const uint8_t s8 = src[34 + 4 * sb + l];
                            dpv[l / 2] = sdot4(apply_4signs(static_cast<uint32_t>(g), s8 & 0xFu), xw[2 * l], dpv[l / 2]);
                            dpv[l / 2] = sdot4(apply_4signs(static_cast<uint32_t>(g >> 32), s8 >> 4), xw[2 * l + 1], dpv[l / 2]);
                        }
                        float& acc = part ? au : ag;
                        acc = std::fma(pd * (0.5f + static_cast<float>(scb & 0xF)), static_cast<float>(dpv[0]), acc);
                        acc = std::fma(pd * (0.5f + static_cast<float>(scb >> 4)), static_cast<float>(dpv[1]), acc);
                    }
                    // DOT8 tile body order: (g lo, u lo, g hi, u hi) on two independent chains
                    const uint8_t* gw = row + ib * 32u;
                    const uint8_t* uw = gw + 16;
                    const uint32_t s16 = rd16(sc + ib * 2u);
                    const uint32_t d2 = rd32(sc + nblk * 2u + (ib >> 3) * 4u);
                    const float pg = quant::fp16_to_f32(static_cast<uint16_t>(d2 & 0xFFFF)) * 0.25f * xda;
                    const float pu = quant::fp16_to_f32(static_cast<uint16_t>(d2 >> 16)) * 0.25f * xda;
                    int32_t g0 = 0, u0 = 0, g1 = 0, u1 = 0;
                    for (uint32_t j = 0; j < 4; ++j) {
                        g0 = sdot4(dot8_decode4_iq2(rd32(gw + 4 * j)), xw[j], g0);
                        u0 = sdot4(dot8_decode4_iq2(rd32(uw + 4 * j)), xw[j], u0);
                        g1 = sdot4(dot8_decode4_iq2(rd32(gw + 4 * j) >> 4), xw[4 + j], g1);
                        u1 = sdot4(dot8_decode4_iq2(rd32(uw + 4 * j) >> 4), xw[4 + j], u1);
                    }
                    bg = std::fma(pg * (0.5f + static_cast<float>(s16 & 0xF)), static_cast<float>(g0), bg);
                    bu = std::fma(pu * (0.5f + static_cast<float>((s16 >> 8) & 0xF)), static_cast<float>(u0), bu);
                    bg = std::fma(pg * (0.5f + static_cast<float>((s16 >> 4) & 0xF)), static_cast<float>(g1), bg);
                    bu = std::fma(pu * (0.5f + static_cast<float>(s16 >> 12)), static_cast<float>(u1), bu);
                }
                ag_ref[lane] = ag; au_ref[lane] = au; ag_new[lane] = bg; au_new[lane] = bu;
                VJ_CHECK(f2u(ag) == f2u(bg) && f2u(au) == f2u(bu));
            }
            VJ_CHECK(f2u(sum16(ag_ref)) == f2u(sum16(ag_new)) && f2u(sum16(au_ref)) == f2u(sum16(au_new)));
        }
    }

    // ---------------------------------------------------------------- planner
    {
        const uint64_t MiB = 1ull << 20;
        const uint64_t pool = static_cast<uint64_t>(5.05 * 1073741824.0);
        std::vector<uint32_t> L48(48);
        for (uint32_t i = 0; i < 48; ++i) L48[i] = i;
        const Dot8RowLayout dn = dot8_iq3_layout(256ull * 3072ull, 512);
        const Dot8RowLayout gu = dot8_gu_iq2s_layout(256ull * 512ull, 3072);

        Dot8ClassSpec down;
        down.name = "moe_down_iq3xxs";
        down.src_type = GgmlType::IQ3_XXS;
        down.role = Dot8Role::MoeDown;
        down.layers = L48;
        down.old_vram_bytes_per_layer = 256ull * 3072ull * 208ull;   // 13 uint4 split rows
        down.new_vram_bytes_per_layer = dn.vram_bytes();
        down.old_read_bytes_per_layer = 8ull * 3072ull * 208ull;
        down.new_read_bytes_per_layer = 8ull * 3072ull * dn.read_bytes_per_row();
        down.bw_old_gbs = 290.0;
        down.bw_new_gbs = 600.0;

        Dot8ClassSpec gup;
        gup.name = "moe_gate_up_iq2s";
        gup.src_type = GgmlType::IQ2_S;
        gup.role = Dot8Role::MoeGateUp;
        gup.layers = L48;
        gup.old_vram_bytes_per_layer = 256ull * 512ull * 123ull * 16ull;
        gup.new_vram_bytes_per_layer = gu.vram_bytes();
        gup.old_read_bytes_per_layer = 8ull * 512ull * 123ull * 16ull;
        gup.new_read_bytes_per_layer = 8ull * 512ull * gu.read_bytes_per_row();
        gup.bw_old_gbs = 320.0;
        gup.bw_new_gbs = 500.0;  // 1.5625 < 1.683 -> crossover fails

        // priority: GU listed first, Down still promoted first; GU rejected by the crossover law
        Dot8PromotionPlan p = plan_dot8_promotion({gup, down}, pool);
        VJ_CHECK(p.verdicts[0].priority == 3 && p.verdicts[1].priority == 0);
        VJ_CHECK(!p.verdicts[0].eligible && p.verdicts[0].reason.find("crossover") != std::string::npos);
        VJ_CHECK(p.verdicts[1].eligible && p.verdicts[1].layers_promoted == 48);
        VJ_CHECK(p.promotions.size() == 48 && p.promotions.front().name == "moe_down_iq3xxs");
        VJ_CHECK(p.used_bytes == 48ull * 48ull * MiB);  // +48 MiB per layer per GPU
        VJ_CHECK(p.used_bytes <= pool && p.us_saved_total > 0.0);

        // crossover holds -> GU fills the remaining budget after all 48 Down layers, in layer order
        gup.bw_new_gbs = 600.0;
        p = plan_dot8_promotion({gup, down}, pool);
        VJ_CHECK(p.verdicts[0].eligible && p.verdicts[1].layers_promoted == 48);
        const uint64_t gu_delta = gu.vram_bytes() - gup.old_vram_bytes_per_layer;
        const uint32_t expect_gu = static_cast<uint32_t>((pool - 48ull * 48ull * MiB) / gu_delta);
        VJ_CHECK(p.verdicts[0].layers_promoted == expect_gu && expect_gu > 0 && expect_gu < 48);
        for (size_t i = 0; i < p.promotions.size(); ++i) {
            VJ_CHECK((i < 48) == (p.promotions[i].name == "moe_down_iq3xxs"));
            if (i >= 48) VJ_CHECK(p.promotions[i].layer == i - 48);
            VJ_CHECK(p.promotions[i].cum_bytes <= pool);
        }
        VJ_CHECK(p.used_bytes + gu_delta > pool);

        // exact equality on the crossover law is a rejection (strict >)
        Dot8ClassSpec eq = down;
        eq.old_read_bytes_per_layer = 1000; eq.new_read_bytes_per_layer = 1500;
        eq.bw_old_gbs = 400.0; eq.bw_new_gbs = 600.0;
        VJ_CHECK(!plan_dot8_promotion({eq}, pool).verdicts[0].eligible);
        eq.bw_new_gbs = 600.001;
        VJ_CHECK(plan_dot8_promotion({eq}, pool).verdicts[0].eligible);

        // banned regression: IQ2_S -> VJ_Q4_A8 at M=1, whatever the bandwidth
        Dot8ClassSpec ban = gup;
        ban.target = Dot8Target::VJ_Q4_A8;
        ban.bw_new_gbs = 5000.0;
        p = plan_dot8_promotion({ban}, pool);
        VJ_CHECK(!p.verdicts[0].eligible && p.verdicts[0].reason.find("banned") != std::string::npos && p.promotions.empty());

        // unmeasured bandwidth
        Dot8ClassSpec um = down;
        um.bw_new_gbs = 0.0;
        VJ_CHECK(!plan_dot8_promotion({um}, pool).verdicts[0].eligible);

        // budget: only 10 Down layers fit; strict priority blocks a smaller eligible IQ2_S class
        Dot8ClassSpec small = gup;
        small.name = "tiny_iq2s";
        small.new_vram_bytes_per_layer = small.old_vram_bytes_per_layer + 1 * MiB;
        const uint64_t pool10 = 10ull * 48ull * MiB + 20ull * MiB;
        p = plan_dot8_promotion({down, small}, pool10);
        VJ_CHECK(p.verdicts[0].layers_promoted == 10 && p.verdicts[1].layers_promoted == 0);
        VJ_CHECK(p.verdicts[1].reason.find("not reached") != std::string::npos);
        p = plan_dot8_promotion({down, small}, pool10, /*strict_priority=*/false);
        VJ_CHECK(p.verdicts[0].layers_promoted == 10 && p.verdicts[1].layers_promoted == 20);
        VJ_CHECK(p.used_bytes <= pool10);

        // without freeing the old copy the full new size is charged: all 48 Down layers no longer fit
        Dot8ClassSpec keep = down;
        keep.frees_old = false;
        p = plan_dot8_promotion({keep}, pool);
        VJ_CHECK(p.verdicts[0].layers_promoted == static_cast<uint32_t>(pool / dn.vram_bytes()));
        VJ_CHECK(p.verdicts[0].layers_promoted < 48);
    }

    std::printf("test_q4_dot8: OK\n");
    return 0;
}
