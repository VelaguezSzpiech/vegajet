// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// VJ_Q4_DOT8 lossless IQ3_XXS / IQ2_S re-code + VRAM promotion planner. See core/q4_dot8_repack.hpp.
// IQ block layouts follow the public ggml definitions (MIT, see THIRD_PARTY_NOTICES.md).
#include "core/q4_dot8_repack.hpp"
#include "core/ggml_iq_tables.hpp"

#include <algorithm>
#include <cstring>

namespace vegajet {

namespace {

namespace T = vegajet::ggml_tables;

static_assert(dot8_iq3_level(0) == 4 && dot8_iq3_level(7) == 62 && dot8_iq3_level(15) == -62, "levels");

// Compile-time proof that the register lookups reproduce every level.
constexpr bool tables_exact() {
    for (uint32_t c = 0; c < 16u; ++c) {
        const uint32_t sel = c * 0x01010101u;
        const uint32_t w = dot8_decode4_iq3(sel);
        for (uint32_t i = 0; i < 4u; ++i)
            if (static_cast<int8_t>((w >> (8u * i)) & 0xFFu) != dot8_iq3_level(c)) return false;
    }
    for (uint32_t c = 0; c < 6u; ++c) {
        const uint32_t w = dot8_decode4_iq2(c * 0x01010101u);
        for (uint32_t i = 0; i < 4u; ++i)
            if (static_cast<int8_t>((w >> (8u * i)) & 0xFFu) != dot8_iq2_level(c)) return false;
    }
    return true;
}
static_assert(tables_exact(), "v_perm lookup tables do not reproduce the IQ levels");

void set_err(std::string* err, const char* msg) {
    if (err) *err = msg;
}

// grid byte -> magnitude index, 0xFF if not a level
constexpr uint8_t iq3_mag_index(uint8_t g) {
    for (uint8_t k = 0; k < 8u; ++k)
        if (static_cast<uint8_t>(kDot8Iq3Mag[k]) == g) return k;
    return 0xFFu;
}
constexpr uint8_t iq2_mag_index(uint8_t g) {
    for (uint8_t k = 0; k < 3u; ++k)
        if (static_cast<uint8_t>(kDot8Iq2Mag[k]) == g) return k;
    return 0xFFu;
}

uint16_t rd16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
uint32_t rd32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
void wr16(uint8_t* p, uint16_t v) { std::memcpy(p, &v, 2); }
void wr32(uint8_t* p, uint32_t v) { std::memcpy(p, &v, 4); }

// Exact fp16 -> f32 (every fp16 value, incl. subnormals, is representable in f32).
float half_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            int e = -1;
            do { man <<= 1; ++e; } while ((man & 0x400u) == 0);
            bits = sign | static_cast<uint32_t>(127 - 15 - e) << 23 | ((man & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

Dot8RowLayout finish_layout(uint64_t rows, uint32_t cols, uint32_t w_raw, uint32_t s_raw) {
    Dot8RowLayout l;
    l.rows = rows;
    l.cols = cols;
    l.w_raw_row_bytes = w_raw;
    l.s_raw_row_bytes = s_raw;
    l.w_row_stride_bytes = dot8_odd256_stride(w_raw);
    if (dot8_odd256_stride(w_raw + s_raw) == l.w_row_stride_bytes) {
        l.scale_embedded = true;
        l.s_offset_bytes = w_raw;
        l.s_row_stride_bytes = l.w_row_stride_bytes;
    } else {
        l.scale_embedded = false;
        l.s_offset_bytes = 0;
        l.s_row_stride_bytes = s_raw;
    }
    return l;
}

bool layout_ok(const Dot8RowLayout& lay, uint64_t row_begin, uint64_t row_end, std::string* err) {
    if (lay.cols == 0 || lay.cols % kDot8SuperElems != 0) { set_err(err, "dot8: cols must be a positive multiple of 256"); return false; }
    if (row_begin > row_end || row_end > lay.rows) { set_err(err, "dot8: bad row range"); return false; }
    return true;
}

uint8_t* scale_row_ptr(const Dot8RowLayout& lay, uint8_t* w_plane, uint8_t* s_plane, uint64_t r) {
    return lay.scale_embedded ? w_plane + r * lay.w_row_stride_bytes + lay.s_offset_bytes
                              : s_plane + r * lay.s_row_stride_bytes;
}

} // namespace

// =================================================================================================
// Layout
// =================================================================================================
uint32_t dot8_odd256_stride(uint32_t raw_bytes) {
    uint32_t tiles = (raw_bytes + kDot8TileBytes - 1u) / kDot8TileBytes;
    if (tiles == 0) tiles = 1;
    if ((tiles & 1u) == 0) ++tiles;
    return tiles * kDot8TileBytes;
}

Dot8RowLayout dot8_iq3_layout(uint64_t rows, uint32_t cols) {
    const uint32_t nsb = cols / kDot8SuperElems;
    return finish_layout(rows, cols, cols / 2u, nsb * 8u);
}

Dot8RowLayout dot8_gu_iq2s_layout(uint64_t rows, uint32_t cols) {
    const uint32_t nblk = cols / kDot8BlockElems;
    const uint32_t nsb = cols / kDot8SuperElems;
    return finish_layout(rows, cols, nblk * 2u * kDot8BlockBytes, nblk * 2u + nsb * 4u);
}

// =================================================================================================
// Block codecs
// =================================================================================================
void dot8_pack_codes(const uint8_t codes[32], uint8_t out16[16]) {
    for (uint32_t i = 0; i < 16u; ++i) {
        // weight i -> word i/4, byte i%4, low nibble; weight 16+i -> same byte, high nibble
        out16[i] = static_cast<uint8_t>((codes[i] & 0xFu) | ((codes[16u + i] & 0xFu) << 4));
    }
}

void dot8_unpack_codes(const uint8_t in16[16], uint8_t codes[32]) {
    for (uint32_t i = 0; i < 16u; ++i) {
        codes[i] = in16[i] & 0xFu;
        codes[16u + i] = in16[i] >> 4;
    }
}

bool dot8_encode_iq3xxs_sub(const uint8_t* blk98, uint32_t ib32, uint8_t codes[32], uint8_t* scale4) {
    const uint8_t* qs = blk98 + 2u + 8u * ib32;
    const uint32_t aux = rd32(blk98 + 2u + 64u + 4u * ib32);
    *scale4 = static_cast<uint8_t>(aux >> 28);
    for (uint32_t l = 0; l < 4u; ++l) {
        const uint8_t signs = T::ksigns_iq2xs[(aux >> (7u * l)) & 127u];
        const uint32_t g1 = T::iq3xxs_grid[qs[2u * l + 0u]];
        const uint32_t g2 = T::iq3xxs_grid[qs[2u * l + 1u]];
        for (uint32_t j = 0; j < 8u; ++j) {
            const uint8_t g = static_cast<uint8_t>(((j < 4u ? g1 : g2) >> (8u * (j & 3u))) & 0xFFu);
            const uint8_t k = iq3_mag_index(g);
            if (k == 0xFFu) return false;
            const uint8_t neg = (signs & T::kmask_iq2xs[j]) ? 1u : 0u;
            codes[8u * l + j] = static_cast<uint8_t>(k | (neg << 3));
        }
    }
    return true;
}

bool dot8_encode_iq2s_sub(const uint8_t* blk82, uint32_t ib32, uint8_t codes[32], uint8_t* scales8) {
    const uint8_t* qs = blk82 + 2u + 4u * ib32;
    const uint8_t* signs = blk82 + 2u + 32u + 4u * ib32;
    const uint8_t qh = blk82[2u + 64u + ib32];
    *scales8 = blk82[2u + 64u + 8u + ib32];
    for (uint32_t l = 0; l < 4u; ++l) {
        const uint64_t grid = T::iq2s_grid[qs[l] | ((static_cast<uint32_t>(qh) << (8u - 2u * l)) & 0x300u)];
        for (uint32_t j = 0; j < 8u; ++j) {
            const uint8_t k = iq2_mag_index(static_cast<uint8_t>((grid >> (8u * j)) & 0xFFu));
            if (k == 0xFFu) return false;
            const uint8_t neg = (signs[l] & T::kmask_iq2xs[j]) ? 1u : 0u;
            codes[8u * l + j] = static_cast<uint8_t>(k + 3u * neg);
        }
    }
    return true;
}

// =================================================================================================
// Repackers
// =================================================================================================
bool dot8_repack_iq3xxs_rows(const uint8_t* src, uint64_t src_row_stride_bytes, uint32_t sb_begin,
                             const Dot8RowLayout& lay, uint64_t row_begin, uint64_t row_end,
                             uint8_t* w_plane, uint8_t* s_plane, std::string* err) {
    if (!layout_ok(lay, row_begin, row_end, err)) return false;
    if (lay.w_raw_row_bytes != lay.cols / 2u || lay.s_raw_row_bytes != lay.cols / kDot8SuperElems * 8u) {
        set_err(err, "dot8: layout is not an IQ3_XXS layout");
        return false;
    }
    if (!src || !w_plane || (!lay.scale_embedded && !s_plane)) { set_err(err, "dot8: null buffer"); return false; }
    const uint32_t nsb = lay.cols / kDot8SuperElems;
    uint8_t codes[32];
    for (uint64_t r = row_begin; r < row_end; ++r) {
        const uint8_t* srow = src + r * src_row_stride_bytes + static_cast<uint64_t>(sb_begin) * kIq3xxsBlockBytes;
        uint8_t* wrow = w_plane + r * lay.w_row_stride_bytes;
        uint8_t* srow_out = scale_row_ptr(lay, w_plane, s_plane, r);
        std::memset(wrow + lay.w_raw_row_bytes, 0, lay.w_row_stride_bytes - lay.w_raw_row_bytes);
        for (uint32_t sb = 0; sb < nsb; ++sb) {
            const uint8_t* blk = srow + static_cast<uint64_t>(sb) * kIq3xxsBlockBytes;
            uint32_t snib = 0;
            for (uint32_t ib = 0; ib < 8u; ++ib) {
                uint8_t s4 = 0;
                if (!dot8_encode_iq3xxs_sub(blk, ib, codes, &s4)) { set_err(err, "dot8: IQ3_XXS grid byte outside the level set"); return false; }
                dot8_pack_codes(codes, wrow + (static_cast<uint64_t>(sb) * 8u + ib) * kDot8BlockBytes);
                snib |= static_cast<uint32_t>(s4) << (4u * ib);
            }
            const float d = half_to_f32(rd16(blk));
            uint32_t dbits;
            std::memcpy(&dbits, &d, 4);
            wr32(srow_out + sb * 8u + 0u, snib);
            wr32(srow_out + sb * 8u + 4u, dbits);
        }
    }
    return true;
}

bool dot8_repack_gu_iq2s_rows(const uint8_t* src_gate, const uint8_t* src_up, uint64_t src_row_stride_bytes,
                              uint32_t sb_begin, const Dot8RowLayout& lay, uint64_t row_begin, uint64_t row_end,
                              uint8_t* w_plane, uint8_t* s_plane, std::string* err) {
    if (!layout_ok(lay, row_begin, row_end, err)) return false;
    const uint32_t nblk = lay.cols / kDot8BlockElems;
    const uint32_t nsb = lay.cols / kDot8SuperElems;
    if (lay.w_raw_row_bytes != nblk * 32u || lay.s_raw_row_bytes != nblk * 2u + nsb * 4u) {
        set_err(err, "dot8: layout is not a merged IQ2_S Gate+Up layout");
        return false;
    }
    if (!src_gate || !src_up || !w_plane || (!lay.scale_embedded && !s_plane)) { set_err(err, "dot8: null buffer"); return false; }
    uint8_t codes[32];
    for (uint64_t r = row_begin; r < row_end; ++r) {
        const uint64_t off = r * src_row_stride_bytes + static_cast<uint64_t>(sb_begin) * kIq2sBlockBytes;
        const uint8_t* grow = src_gate + off;
        const uint8_t* urow = src_up + off;
        uint8_t* wrow = w_plane + r * lay.w_row_stride_bytes;
        uint8_t* sc = scale_row_ptr(lay, w_plane, s_plane, r);
        // Zero the padding (and the embedded scale region, which is filled below).
        std::memset(wrow + lay.w_raw_row_bytes, 0, lay.w_row_stride_bytes - lay.w_raw_row_bytes);
        for (uint32_t sb = 0; sb < nsb; ++sb) {
            const uint8_t* gb = grow + static_cast<uint64_t>(sb) * kIq2sBlockBytes;
            const uint8_t* ub = urow + static_cast<uint64_t>(sb) * kIq2sBlockBytes;
            for (uint32_t ib = 0; ib < 8u; ++ib) {
                const uint32_t blk = sb * 8u + ib;
                uint8_t gs = 0, us = 0;
                if (!dot8_encode_iq2s_sub(gb, ib, codes, &gs)) { set_err(err, "dot8: IQ2_S grid byte outside the level set"); return false; }
                dot8_pack_codes(codes, wrow + static_cast<uint64_t>(blk) * 32u);
                if (!dot8_encode_iq2s_sub(ub, ib, codes, &us)) { set_err(err, "dot8: IQ2_S grid byte outside the level set"); return false; }
                dot8_pack_codes(codes, wrow + static_cast<uint64_t>(blk) * 32u + 16u);
                wr16(sc + blk * 2u, static_cast<uint16_t>(gs | (static_cast<uint32_t>(us) << 8)));
            }
            wr32(sc + nblk * 2u + sb * 4u, static_cast<uint32_t>(rd16(gb)) | (static_cast<uint32_t>(rd16(ub)) << 16));
        }
    }
    return true;
}

// =================================================================================================
// CPU reference of the device decode (same masks, shifts and perm tables as ops/kernels/vj_moe_dot8.hiph)
// =================================================================================================
void dot8_decode_row_iq3_int(const uint8_t* w_row, uint32_t cols, int8_t* out) {
    for (uint32_t b = 0; b < cols / kDot8BlockElems; ++b) {
        const uint8_t* blk = w_row + static_cast<uint64_t>(b) * kDot8BlockBytes;
        for (uint32_t j = 0; j < 4u; ++j) {
            const uint32_t w = rd32(blk + 4u * j);
            const uint32_t lo = dot8_decode4_iq3(w & 0x0F0F0F0Fu);
            const uint32_t hi = dot8_decode4_iq3((w >> 4) & 0x0F0F0F0Fu);
            for (uint32_t i = 0; i < 4u; ++i) {
                out[b * 32u + 4u * j + i] = static_cast<int8_t>((lo >> (8u * i)) & 0xFFu);
                out[b * 32u + 16u + 4u * j + i] = static_cast<int8_t>((hi >> (8u * i)) & 0xFFu);
            }
        }
    }
}

void dot8_decode_row_iq2_int(const uint8_t* w_codes_row, uint32_t cols, uint32_t block_stride_bytes, int8_t* out) {
    for (uint32_t b = 0; b < cols / kDot8BlockElems; ++b) {
        const uint8_t* blk = w_codes_row + static_cast<uint64_t>(b) * block_stride_bytes;
        for (uint32_t j = 0; j < 4u; ++j) {
            const uint32_t w = rd32(blk + 4u * j);
            const uint32_t lo = dot8_decode4_iq2(w);
            const uint32_t hi = dot8_decode4_iq2(w >> 4);
            for (uint32_t i = 0; i < 4u; ++i) {
                out[b * 32u + 4u * j + i] = static_cast<int8_t>((lo >> (8u * i)) & 0xFFu);
                out[b * 32u + 16u + 4u * j + i] = static_cast<int8_t>((hi >> (8u * i)) & 0xFFu);
            }
        }
    }
}

// =================================================================================================
// Lossless 3-bit bit-sliced merged Gate+Up IQ2_S (VJ_Q3_IQ2S): 2304 B w_plane + 256 B s_plane @ D=3072
// =================================================================================================
Dot8RowLayout q3_gu_iq2s_layout(uint64_t rows, uint32_t cols) {
    Dot8RowLayout lay;
    lay.rows = rows;
    lay.cols = cols;
    if (cols == 0u || (cols % kDot8SuperElems) != 0u) return lay;
    const uint32_t nblk = cols / kDot8BlockElems;
    const uint32_t nsb  = cols / kDot8SuperElems;
    lay.w_raw_row_bytes    = nblk * 24u;                    // nblk * (16 B mags + 8 B signs) = 2304 B for D=3072
    lay.s_raw_row_bytes    = nblk * 2u + nsb * 4u;          // 240 B for D=3072
    lay.w_row_stride_bytes = dot8_odd256_stride(lay.w_raw_row_bytes);
    lay.scale_embedded     = false;
    lay.s_offset_bytes     = 0u;
    lay.s_row_stride_bytes = dot8_odd256_stride(lay.s_raw_row_bytes); // 256 B for D=3072
    return lay;
}

bool q3_repack_gu_iq2s_rows(const uint8_t* src_gate, const uint8_t* src_up, uint64_t src_row_stride_bytes,
                            uint32_t sb_begin, const Dot8RowLayout& lay, uint64_t row_begin, uint64_t row_end,
                            uint8_t* w_plane, uint8_t* s_plane, std::string* err) {
    if (!layout_ok(lay, row_begin, row_end, err)) return false;
    const uint32_t nblk = lay.cols / kDot8BlockElems;
    const uint32_t nsb  = lay.cols / kDot8SuperElems;
    if (lay.w_raw_row_bytes != nblk * 24u || lay.s_raw_row_bytes != nblk * 2u + nsb * 4u) {
        set_err(err, "q3: layout is not a VJ_Q3_IQ2S Gate+Up layout");
        return false;
    }
    if (!src_gate || !src_up || !w_plane || !s_plane) { set_err(err, "q3: null buffer"); return false; }
    uint8_t gc[32], uc[32];
    auto pack_mags16 = [](const uint8_t* codes) -> uint32_t {
        uint32_t w = 0u;
        for (uint32_t k = 0u; k < 4u; ++k) {
            for (uint32_t b = 0u; b < 4u; ++b) {
                const uint32_t c = codes[4u * k + b];
                const uint32_t m = (c >= 3u) ? (c - 3u) : c;
                w |= (m & 3u) << (8u * b + 2u * k);
            }
        }
        return w;
    };
    auto pack_signs32 = [](const uint8_t* codes) -> uint32_t {
        uint32_t w = 0u;
        for (uint32_t k = 0u; k < 8u; ++k) {
            for (uint32_t b = 0u; b < 4u; ++b) {
                const uint32_t s = (codes[4u * k + b] >= 3u) ? 1u : 0u;
                w |= s << (8u * b + k);
            }
        }
        return w;
    };
    for (uint64_t r = row_begin; r < row_end; ++r) {
        const uint64_t off = r * src_row_stride_bytes + static_cast<uint64_t>(sb_begin) * kIq2sBlockBytes;
        const uint8_t* grow = src_gate + off;
        const uint8_t* urow = src_up + off;
        uint8_t* wrow = w_plane + r * lay.w_row_stride_bytes;
        uint8_t* srow = s_plane + r * lay.s_row_stride_bytes;
        if (lay.w_row_stride_bytes > lay.w_raw_row_bytes) {
            std::memset(wrow + lay.w_raw_row_bytes, 0, lay.w_row_stride_bytes - lay.w_raw_row_bytes);
        }
        if (lay.s_row_stride_bytes > lay.s_raw_row_bytes) {
            std::memset(srow + lay.s_raw_row_bytes, 0, lay.s_row_stride_bytes - lay.s_raw_row_bytes);
        }
        uint8_t* mag_base = wrow;
        uint8_t* sgn_base = wrow + static_cast<uint64_t>(nblk) * 16u;
        for (uint32_t sb = 0u; sb < nsb; ++sb) {
            const uint8_t* gb = grow + static_cast<uint64_t>(sb) * kIq2sBlockBytes;
            const uint8_t* ub = urow + static_cast<uint64_t>(sb) * kIq2sBlockBytes;
            for (uint32_t ib = 0u; ib < 8u; ++ib) {
                const uint32_t blk = sb * 8u + ib;
                uint8_t gs = 0, us = 0;
                if (!dot8_encode_iq2s_sub(gb, ib, gc, &gs) || !dot8_encode_iq2s_sub(ub, ib, uc, &us)) {
                    set_err(err, "q3: IQ2_S grid byte outside the level set");
                    return false;
                }
                uint8_t* mp = mag_base + static_cast<uint64_t>(blk) * 16u;
                uint8_t* sp = sgn_base + static_cast<uint64_t>(blk) * 8u;
                wr32(mp +  0u, pack_mags16(gc));
                wr32(mp +  4u, pack_mags16(gc + 16u));
                wr32(mp +  8u, pack_mags16(uc));
                wr32(mp + 12u, pack_mags16(uc + 16u));
                wr32(sp +  0u, pack_signs32(gc));
                wr32(sp +  4u, pack_signs32(uc));
                wr16(srow + blk * 2u, static_cast<uint16_t>(gs | (static_cast<uint32_t>(us) << 8)));
            }
            wr32(srow + nblk * 2u + sb * 4u, static_cast<uint32_t>(rd16(gb)) | (static_cast<uint32_t>(rd16(ub)) << 16));
        }
    }
    return true;
}

void q3_decode_gu_row_int(const uint8_t* w_row, uint32_t cols, int8_t* out_gate, int8_t* out_up) {
    const uint32_t nblk = cols / kDot8BlockElems;
    const uint8_t* mag_base = w_row;
    const uint8_t* sgn_base = w_row + static_cast<uint64_t>(nblk) * 16u;
    auto dec4 = [](uint32_t m_sh, uint32_t s_sh) -> uint32_t {
        const uint32_t sel = (m_sh & 0x03030303u) | (s_sh & 0x04040404u);
        return dot8_perm_b32(0x00D5E7F8u, 0x002B1908u, sel);
    };
    auto dec32 = [&](uint32_t m_lo, uint32_t m_hi, uint32_t sg, int8_t* dst) {
        const uint32_t w[8] = {
            dec4(m_lo,      sg << 2), dec4(m_lo >> 2, sg << 1), dec4(m_lo >> 4, sg),      dec4(m_lo >> 6, sg >> 1),
            dec4(m_hi,      sg >> 2), dec4(m_hi >> 2, sg >> 3), dec4(m_hi >> 4, sg >> 4), dec4(m_hi >> 6, sg >> 5),
        };
        for (uint32_t k = 0u; k < 8u; ++k) {
            for (uint32_t b = 0u; b < 4u; ++b) dst[4u * k + b] = static_cast<int8_t>((w[k] >> (8u * b)) & 0xFFu);
        }
    };
    for (uint32_t blk = 0u; blk < nblk; ++blk) {
        const uint8_t* mp = mag_base + static_cast<uint64_t>(blk) * 16u;
        const uint8_t* sp = sgn_base + static_cast<uint64_t>(blk) * 8u;
        dec32(rd32(mp + 0u), rd32(mp +  4u), rd32(sp + 0u), out_gate + blk * 32u);
        dec32(rd32(mp + 8u), rd32(mp + 12u), rd32(sp + 4u), out_up   + blk * 32u);
    }
}

// =================================================================================================
// Planner
// =================================================================================================
uint32_t dot8_class_priority(const Dot8ClassSpec& c) {
    if (c.src_type == GgmlType::IQ3_XXS) {
        if (c.role == Dot8Role::MoeDown) return 0u;
        if (c.role == Dot8Role::MoeGateUp) return 1u;
        return 2u;
    }
    return 3u;  // IQ2_S and anything else last
}

Dot8PromotionPlan plan_dot8_promotion(const std::vector<Dot8ClassSpec>& classes, uint64_t pool_bytes_per_gpu,
                                      bool strict_priority) {
    Dot8PromotionPlan plan;
    plan.pool_bytes = pool_bytes_per_gpu;
    plan.verdicts.resize(classes.size());

    std::vector<size_t> order;
    for (size_t i = 0; i < classes.size(); ++i) {
        const Dot8ClassSpec& c = classes[i];
        Dot8ClassVerdict& v = plan.verdicts[i];
        v.name = c.name;
        v.priority = dot8_class_priority(c);
        v.bw_ratio = (c.bw_old_gbs > 0.0) ? c.bw_new_gbs / c.bw_old_gbs : 0.0;
        v.bpw_ratio = (c.old_read_bytes_per_layer > 0)
                          ? static_cast<double>(c.new_read_bytes_per_layer) / static_cast<double>(c.old_read_bytes_per_layer)
                          : 0.0;
        if (c.src_type != GgmlType::IQ3_XXS && c.src_type != GgmlType::IQ2_S) {
            v.reason = "unsupported source type (IQ3_XXS / IQ2_S only)";
        } else if (c.src_type == GgmlType::IQ2_S && c.target == Dot8Target::VJ_Q4_A8 && c.batch_m == 1u) {
            v.reason = "banned: IQ2_S -> VJ_Q4_A8 at M=1";
        } else if (c.target == Dot8Target::VJ_Q4_A8) {
            v.reason = "VJ_Q4_A8 is a lossy requantization; only lossless VJ_Q4_DOT8 is planned here";
        } else if (c.bw_old_gbs <= 0.0 || c.bw_new_gbs <= 0.0) {
            v.reason = "bandwidth not measured";
        } else if (c.old_read_bytes_per_layer == 0 || c.new_read_bytes_per_layer == 0) {
            v.reason = "read bytes missing";
        } else if (!(c.bw_new_gbs * static_cast<double>(c.old_read_bytes_per_layer) >
                     c.bw_old_gbs * static_cast<double>(c.new_read_bytes_per_layer))) {
            v.reason = "crossover law fails: bw_new/bw_old <= bpw_new/bpw_old";
        } else {
            v.eligible = true;
            v.reason = "eligible";
            order.push_back(i);
        }
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return plan.verdicts[a].priority < plan.verdicts[b].priority;
    });

    int64_t used = 0;
    const int64_t pool = static_cast<int64_t>(pool_bytes_per_gpu);
    bool blocked = false;
    for (size_t idx : order) {
        const Dot8ClassSpec& c = classes[idx];
        Dot8ClassVerdict& v = plan.verdicts[idx];
        if (blocked) {
            v.reason = "eligible, not reached: a higher-priority class did not fit completely";
            continue;
        }
        const int64_t delta = static_cast<int64_t>(c.new_vram_bytes_per_layer) -
                              (c.frees_old ? static_cast<int64_t>(c.old_vram_bytes_per_layer) : 0);
        const double us = (static_cast<double>(c.old_read_bytes_per_layer) / (c.bw_old_gbs * 1e3)) -
                          (static_cast<double>(c.new_read_bytes_per_layer) / (c.bw_new_gbs * 1e3));
        for (uint32_t layer : c.layers) {
            if (used + delta > pool) break;
            used += delta;
            Dot8Promotion p;
            p.class_index = idx;
            p.name = c.name;
            p.layer = layer;
            p.delta_bytes = delta;
            p.cum_bytes = static_cast<uint64_t>(std::max<int64_t>(used, 0));
            p.us_saved = us;
            plan.us_saved_total += us;
            plan.promotions.push_back(p);
            ++v.layers_promoted;
        }
        if (v.layers_promoted < c.layers.size()) {
            v.reason = "eligible, budget-limited";
            if (strict_priority) blocked = true;
        } else {
            v.reason = "eligible, fully promoted";
        }
    }
    plan.used_bytes = static_cast<uint64_t>(std::max<int64_t>(used, 0));
    return plan;
}

} // namespace vegajet
