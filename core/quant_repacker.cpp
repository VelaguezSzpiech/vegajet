// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Block decoders follow the public ggml block layouts (MIT, see THIRD_PARTY_NOTICES.md) and keep the
// reference arithmetic order so results are bit-identical to ggml's dequantize_row_* functions.
#include "core/quant_repacker.hpp"
#include "core/ggml_iq_tables.hpp"
#include "core/vj_check.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace vegajet::quant {

namespace T = vegajet::ggml_tables;

const char* native_format_name(NativeFormat f) {
    switch (f) {
        case NativeFormat::VJ_Q2_DOT8: return "VJ_Q2_DOT8";
        case NativeFormat::VJ_Q3_DOT8: return "VJ_Q3_DOT8";
        case NativeFormat::VJ_Q8_DP4A: return "VJ_Q8_DP4A";
        case NativeFormat::VJ_Q4_DOT8: return "VJ_Q4_DOT8";
        case NativeFormat::VJ_IQ_PACKED: return "VJ_IQ_PACKED";
        case NativeFormat::VJ_Q4_A8: return "VJ_Q4_A8";
    }
    return "?";
}

// =================================================================================================
// Scalar conversions
// =================================================================================================
float fp16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t frac = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (frac == 0) {
            bits = sign;
        } else {
            exp = 127 - 14;
            while ((frac & 0x400u) == 0) { frac <<= 1; --exp; }
            frac &= 0x3FFu;
            bits = sign | (exp << 23) | (frac << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (frac << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (frac << 13);
    }
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

uint16_t f32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    if ((x & 0x7F800000u) == 0x7F800000u)  // inf / nan
        return static_cast<uint16_t>(sign | 0x7C00u | ((x & 0x7FFFFFu) ? 0x200u : 0u));
    const int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t r = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u), half = 1u << (shift - 1u);
        if (rem > half || (rem == half && (r & 1u))) ++r;  // round to nearest even
        return static_cast<uint16_t>(sign | r);
    }
    uint32_t r = (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) ++r;
    if (r >= 0x7C00u) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | r);
}

float bf16_to_f32(uint16_t b) {
    const uint32_t bits = static_cast<uint32_t>(b) << 16;
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

float fp8_e4m3_to_f32(uint8_t x) {
    const uint8_t ax = x & 0x7F;
    if (ax == 0x7F) return NAN;
    const int exp = (ax >> 3) & 0xF;
    const int man = ax & 0x7;
    const float v = exp == 0 ? std::ldexp(static_cast<float>(man), -9)
                             : std::ldexp(1.0f + static_cast<float>(man) / 8.0f, exp - 7);
    return (x & 0x80) ? -v : v;
}

void fwht64_inplace(float* x) {
    for (int len = 1; len < 64; len <<= 1)
        for (int i = 0; i < 64; i += len << 1)
            for (int j = 0; j < len; ++j) {
                const float u = x[i + j], v = x[i + j + len];
                x[i + j] = u + v;
                x[i + j + len] = u - v;
            }
    for (int i = 0; i < 64; ++i) x[i] *= 0.125f;
}

// =================================================================================================
// Block layouts (ggml)
// =================================================================================================
namespace {

constexpr int QK_K = 256;

#pragma pack(push, 1)
struct block_q4_0 { uint16_t d; uint8_t qs[16]; };
struct block_q4_1 { uint16_t d, m; uint8_t qs[16]; };
struct block_q5_0 { uint16_t d; uint8_t qh[4]; uint8_t qs[16]; };
struct block_q5_1 { uint16_t d, m; uint8_t qh[4]; uint8_t qs[16]; };
struct block_q8_0 { uint16_t d; int8_t qs[32]; };
struct block_q8_1 { uint16_t d, s; int8_t qs[32]; };
struct block_q2_0 { uint16_t d; uint8_t qs[16]; };
struct block_q2_K { uint8_t scales[16]; uint8_t qs[64]; uint16_t d, dmin; };
struct block_q3_K { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; uint16_t d; };
struct block_q4_K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; };
struct block_q5_K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
struct block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; };
struct block_iq2_xxs { uint16_t d; uint16_t qs[32]; };
struct block_iq2_xs { uint16_t d; uint16_t qs[32]; uint8_t scales[8]; };
struct block_iq2_s { uint16_t d; uint8_t qs[64]; uint8_t qh[8]; uint8_t scales[8]; };
struct block_iq3_xxs { uint16_t d; uint8_t qs[96]; };
struct block_iq3_s { uint16_t d; uint8_t qs[64]; uint8_t qh[8]; uint8_t signs[32]; uint8_t scales[4]; };
struct block_iq4_nl { uint16_t d; uint8_t qs[16]; };
struct block_iq4_xs { uint16_t d; uint16_t scales_h; uint8_t scales_l[4]; uint8_t qs[128]; };
#pragma pack(pop)

static_assert(sizeof(block_q4_0) == 18 && sizeof(block_q4_1) == 20 && sizeof(block_q5_0) == 22, "");
static_assert(sizeof(block_q5_1) == 24 && sizeof(block_q8_0) == 34 && sizeof(block_q8_1) == 36, "");
static_assert(sizeof(block_q2_0) == 18 && sizeof(block_q2_K) == 84 && sizeof(block_q3_K) == 110, "");
static_assert(sizeof(block_q4_K) == 144 && sizeof(block_q5_K) == 176 && sizeof(block_q6_K) == 210, "");
static_assert(sizeof(block_iq2_xxs) == 66 && sizeof(block_iq2_xs) == 74 && sizeof(block_iq2_s) == 82, "");
static_assert(sizeof(block_iq3_xxs) == 98 && sizeof(block_iq3_s) == 110, "");
static_assert(sizeof(block_iq4_nl) == 18 && sizeof(block_iq4_xs) == 136, "");

template <typename B> const B* blocks(const uint8_t* p) { return reinterpret_cast<const B*>(p); }
inline float H(uint16_t h) { return fp16_to_f32(h); }

void dec_f32(const uint8_t* s, float* y, uint64_t n) { std::memcpy(y, s, n * 4); }
void dec_f16(const uint8_t* s, float* y, uint64_t n) {
    for (uint64_t i = 0; i < n; ++i) { uint16_t h; std::memcpy(&h, s + 2 * i, 2); y[i] = H(h); }
}
void dec_bf16(const uint8_t* s, float* y, uint64_t n) {
    for (uint64_t i = 0; i < n; ++i) { uint16_t h; std::memcpy(&h, s + 2 * i, 2); y[i] = bf16_to_f32(h); }
}
void dec_f8(const uint8_t* s, float* y, uint64_t n) {
    for (uint64_t i = 0; i < n; ++i) y[i] = fp8_e4m3_to_f32(s[i]);
}

void dec_q4_0(const uint8_t* s, float* y, uint64_t n) {
    const auto* x = blocks<block_q4_0>(s);
    for (uint64_t i = 0; i < n / 32; ++i) {
        const float d = H(x[i].d);
        for (int j = 0; j < 16; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8, x1 = (x[i].qs[j] >> 4) - 8;
            y[i * 32 + j] = x0 * d;
            y[i * 32 + j + 16] = x1 * d;
        }
    }
}
void dec_q4_1(const uint8_t* s, float* y, uint64_t n) {
    const auto* x = blocks<block_q4_1>(s);
    for (uint64_t i = 0; i < n / 32; ++i) {
        const float d = H(x[i].d), m = H(x[i].m);
        for (int j = 0; j < 16; ++j) {
            const int x0 = x[i].qs[j] & 0x0F, x1 = x[i].qs[j] >> 4;
            y[i * 32 + j] = x0 * d + m;
            y[i * 32 + j + 16] = x1 * d + m;
        }
    }
}
void dec_q5_0(const uint8_t* s, float* y, uint64_t n) {
    const auto* x = blocks<block_q5_0>(s);
    for (uint64_t i = 0; i < n / 32; ++i) {
        const float d = H(x[i].d);
        uint32_t qh; std::memcpy(&qh, x[i].qh, 4);
        for (int j = 0; j < 16; ++j) {
            const uint8_t xh0 = ((qh >> (j + 0)) << 4) & 0x10, xh1 = ((qh >> (j + 12))) & 0x10;
            const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh0) - 16, x1 = ((x[i].qs[j] >> 4) | xh1) - 16;
            y[i * 32 + j] = x0 * d;
            y[i * 32 + j + 16] = x1 * d;
        }
    }
}
void dec_q5_1(const uint8_t* s, float* y, uint64_t n) {
    const auto* x = blocks<block_q5_1>(s);
    for (uint64_t i = 0; i < n / 32; ++i) {
        const float d = H(x[i].d), m = H(x[i].m);
        uint32_t qh; std::memcpy(&qh, x[i].qh, 4);
        for (int j = 0; j < 16; ++j) {
            const uint8_t xh0 = ((qh >> (j + 0)) << 4) & 0x10, xh1 = ((qh >> (j + 12))) & 0x10;
            const int x0 = (x[i].qs[j] & 0x0F) | xh0, x1 = (x[i].qs[j] >> 4) | xh1;
            y[i * 32 + j] = x0 * d + m;
            y[i * 32 + j + 16] = x1 * d + m;
        }
    }
}
void dec_q8_0(const uint8_t* s, float* y, uint64_t n) {
    const auto* x = blocks<block_q8_0>(s);
    for (uint64_t i = 0; i < n / 32; ++i) {
        const float d = H(x[i].d);
        for (int j = 0; j < 32; ++j) y[i * 32 + j] = x[i].qs[j] * d;
    }
}
void dec_q8_1(const uint8_t* s, float* y, uint64_t n) {  // `s` field (d * sum) is metadata only
    const auto* x = blocks<block_q8_1>(s);
    for (uint64_t i = 0; i < n / 32; ++i) {
        const float d = H(x[i].d);
        for (int j = 0; j < 32; ++j) y[i * 32 + j] = x[i].qs[j] * d;
    }
}
void dec_q2_0(const uint8_t* s, float* y, uint64_t n) {
    const auto* x = blocks<block_q2_0>(s);
    for (uint64_t i = 0; i < n / 64; ++i) {
        const float d = H(x[i].d);
        for (int j = 0; j < 64; ++j) {
            const uint8_t q = (x[i].qs[j / 4] >> ((j % 4) * 2)) & 0x03;  // 00=-1 01=0 10=+1 11=+2
            y[i * 64 + j] = ((int)q - 1) * d;
        }
    }
}

void dec_q2_K(const uint8_t* s, float* y, uint64_t n) {
    const auto* x = blocks<block_q2_K>(s);
    for (uint64_t i = 0; i < n / QK_K; ++i) {
        const float d = H(x[i].d), min = H(x[i].dmin);
        const uint8_t* q = x[i].qs;
        int is = 0;
        for (int k = 0; k < QK_K; k += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = x[i].scales[is++];
                float dl = d * (sc & 0xF), ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;
                sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3)) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
}

void dec_q3_K(const uint8_t* s, float* y, uint64_t n) {
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    const auto* x = blocks<block_q3_K>(s);
    uint32_t aux[4];
    const int8_t* scales = reinterpret_cast<const int8_t*>(aux);
    for (uint64_t i = 0; i < n / QK_K; ++i) {
        const float d_all = H(x[i].d);
        const uint8_t* q = x[i].qs;
        const uint8_t* hm = x[i].hmask;
        uint8_t m = 1;
        std::memcpy(aux, x[i].scales, 12);
        const uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        int is = 0;
        for (int k = 0; k < QK_K; k += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * ((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

void dec_q4_K(const uint8_t* s, float* y, uint64_t n) {
    const auto* x = blocks<block_q4_K>(s);
    for (uint64_t i = 0; i < n / QK_K; ++i) {
        const uint8_t* q = x[i].qs;
        const float d = H(x[i].d), min = H(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

void dec_q5_K(const uint8_t* s, float* y, uint64_t n) {
    const auto* x = blocks<block_q5_K>(s);
    for (uint64_t i = 0; i < n / QK_K; ++i) {
        const uint8_t* ql = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const float d = H(x[i].d), min = H(x[i].dmin);
        int is = 0;
        uint8_t sc, m, u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

void dec_q6_K(const uint8_t* s, float* y, uint64_t n) {
    const auto* x = blocks<block_q6_K>(s);
    for (uint64_t i = 0; i < n / QK_K; ++i) {
        const float d = H(x[i].d);
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* sc = x[i].scales;
        for (int k = 0; k < QK_K; k += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void dec_iq2_xxs(const uint8_t* s, float* y, uint64_t n) {
    const auto* x = blocks<block_iq2_xxs>(s);
    uint32_t aux32[2];
    const uint8_t* aux8 = reinterpret_cast<const uint8_t*>(aux32);
    for (uint64_t i = 0; i < n / QK_K; ++i) {
        const float d = H(x[i].d);
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            std::memcpy(aux32, x[i].qs + 4 * ib32, 2 * sizeof(uint32_t));
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid = reinterpret_cast<const uint8_t*>(T::iq2xxs_grid + aux8[l]);
                const uint8_t signs = T::ksigns_iq2xs[(aux32[1] >> 7 * l) & 127];
                for (int j = 0; j < 8; ++j) y[j] = db * grid[j] * (signs & T::kmask_iq2xs[j] ? -1.f : 1.f);
                y += 8;
            }
        }
    }
}

void dec_iq2_xs(const uint8_t* s, float* y, uint64_t n) {
    const auto* x = blocks<block_iq2_xs>(s);
    float db[2];
    for (uint64_t i = 0; i < n / QK_K; ++i) {
        const float d = H(x[i].d);
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid = reinterpret_cast<const uint8_t*>(T::iq2xs_grid + (x[i].qs[4 * ib32 + l] & 511));
                const uint8_t signs = T::ksigns_iq2xs[x[i].qs[4 * ib32 + l] >> 9];
                for (int j = 0; j < 8; ++j) y[j] = db[l / 2] * grid[j] * (signs & T::kmask_iq2xs[j] ? -1.f : 1.f);
                y += 8;
            }
        }
    }
}

void dec_iq2_s(const uint8_t* s, float* y, uint64_t n) {
    const auto* x = blocks<block_iq2_s>(s);
    float db[2];
    for (uint64_t i = 0; i < n / QK_K; ++i) {
        const float d = H(x[i].d);
        const uint8_t* qs = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const uint8_t* signs = qs + QK_K / 8;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const float dl = db[l / 2];
                const uint8_t* grid = reinterpret_cast<const uint8_t*>(
                    T::iq2s_grid + (qs[l] | (qh[ib32] << (8 - 2 * l) & 0x300)));
                for (int j = 0; j < 8; ++j) y[j] = dl * grid[j] * (signs[l] & T::kmask_iq2xs[j] ? -1.f : 1.f);
                y += 8;
            }
            qs += 4;
            signs += 4;
        }
    }
}

void dec_iq3_xxs(const uint8_t* s, float* y, uint64_t n) {
    const auto* x = blocks<block_iq3_xxs>(s);
    uint32_t aux32;
    for (uint64_t i = 0; i < n / QK_K; ++i) {
        const float d = H(x[i].d);
        const uint8_t* qs = x[i].qs;
        const uint8_t* scales_and_signs = qs + QK_K / 4;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            std::memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = T::ksigns_iq2xs[(aux32 >> 7 * l) & 127];
                const uint8_t* grid1 = reinterpret_cast<const uint8_t*>(T::iq3xxs_grid + qs[2 * l + 0]);
                const uint8_t* grid2 = reinterpret_cast<const uint8_t*>(T::iq3xxs_grid + qs[2 * l + 1]);
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db * grid1[j] * (signs & T::kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    y[j + 4] = db * grid2[j] * (signs & T::kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
        }
    }
}

void dec_iq3_s(const uint8_t* s, float* y, uint64_t n) {
    const auto* x = blocks<block_iq3_s>(s);
    for (uint64_t i = 0; i < n / QK_K; ++i) {
        const float d = H(x[i].d);
        const uint8_t* qs = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const uint8_t* signs = x[i].signs;
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const float db1 = d * (1 + 2 * (x[i].scales[ib32 / 2] & 0xf));
            const float db2 = d * (1 + 2 * (x[i].scales[ib32 / 2] >> 4));
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid1 = reinterpret_cast<const uint8_t*>(T::iq3s_grid + (qs[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 256)));
                const uint8_t* grid2 = reinterpret_cast<const uint8_t*>(T::iq3s_grid + (qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db1 * grid1[j] * (signs[l] & T::kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    y[j + 4] = db1 * grid2[j] * (signs[l] & T::kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid1 = reinterpret_cast<const uint8_t*>(T::iq3s_grid + (qs[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 256)));
                const uint8_t* grid2 = reinterpret_cast<const uint8_t*>(T::iq3s_grid + (qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db2 * grid1[j] * (signs[l] & T::kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    y[j + 4] = db2 * grid2[j] * (signs[l] & T::kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
}

void dec_iq4_nl(const uint8_t* s, float* y, uint64_t n) {
    const auto* x = blocks<block_iq4_nl>(s);
    for (uint64_t i = 0; i < n / 32; ++i) {
        const float d = H(x[i].d);
        for (int j = 0; j < 16; ++j) {
            y[j + 0] = d * T::kvalues_iq4nl[x[i].qs[j] & 0xf];
            y[j + 16] = d * T::kvalues_iq4nl[x[i].qs[j] >> 4];
        }
        y += 32;
    }
}

void dec_iq4_xs(const uint8_t* s, float* y, uint64_t n) {
    const auto* x = blocks<block_iq4_xs>(s);
    for (uint64_t i = 0; i < n / QK_K; ++i) {
        const uint8_t* qs = x[i].qs;
        const float d = H(x[i].d);
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const int ls = ((x[i].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x[i].scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j + 0] = dl * T::kvalues_iq4nl[qs[j] & 0xf];
                y[j + 16] = dl * T::kvalues_iq4nl[qs[j] >> 4];
            }
            y += 32;
            qs += 16;
        }
    }
}

} // namespace

// =================================================================================================
// Registry
// =================================================================================================
DecoderRegistry& DecoderRegistry::instance() {
    static DecoderRegistry reg;
    return reg;
}

DecoderRegistry::DecoderRegistry() {
    using D = DiskQuantType;
    const struct { D t; BlockDecodeFn fn; } table[] = {
        {D::F32, dec_f32}, {D::F16, dec_f16}, {D::BF16, dec_bf16}, {D::F8_E4M3, dec_f8},
        {D::Q8_0, dec_q8_0}, {D::Q8_1, dec_q8_1}, {D::Q6_K, dec_q6_K}, {D::Q5_0, dec_q5_0},
        {D::Q5_1, dec_q5_1}, {D::Q5_K, dec_q5_K}, {D::Q4_0, dec_q4_0}, {D::Q4_1, dec_q4_1},
        {D::Q4_K, dec_q4_K}, {D::IQ4_NL, dec_iq4_nl}, {D::IQ4_XS, dec_iq4_xs}, {D::Q3_K, dec_q3_K},
        {D::Q2_K, dec_q2_K}, {D::Q2_0, dec_q2_0}, {D::IQ3_XXS, dec_iq3_xxs}, {D::IQ3_S, dec_iq3_s},
        {D::IQ2_XXS, dec_iq2_xxs}, {D::IQ2_XS, dec_iq2_xs}, {D::IQ2_S, dec_iq2_s},
    };
    for (const auto& e : table) {
        const GgmlTypeTraits tr = ggml_type_traits(static_cast<uint32_t>(e.t));
        register_format(e.t, FormatInfo{tr.block_elems, tr.block_bytes, e.fn, tr.name});
    }
}

void DecoderRegistry::register_format(DiskQuantType type, const FormatInfo& info) {
    formats_[static_cast<uint32_t>(type)] = info;
}

const FormatInfo* DecoderRegistry::find(DiskQuantType type) const {
    auto it = formats_.find(static_cast<uint32_t>(type));
    return it == formats_.end() ? nullptr : &it->second;
}

std::vector<DiskQuantType> DecoderRegistry::types() const {
    std::vector<DiskQuantType> out;
    for (const auto& kv : formats_) out.push_back(static_cast<DiskQuantType>(kv.first));
    std::sort(out.begin(), out.end());
    return out;
}

uint64_t DecoderRegistry::row_bytes(DiskQuantType type, uint64_t cols) const {
    const FormatInfo* fi = find(type);
    if (!fi || cols % fi->block_elems != 0) return 0;
    return cols / fi->block_elems * fi->block_bytes;
}

bool DecoderRegistry::decode_row_f32(DiskQuantType type, const uint8_t* src, float* dst, uint64_t cols) const {
    const FormatInfo* fi = find(type);
    if (!fi || !fi->decode || cols % fi->block_elems != 0) return false;
    fi->decode(src, dst, cols);
    return true;
}

// =================================================================================================
// Native packers
// =================================================================================================
namespace {

inline void quant_block_q8(const float* blk, int8_t* q, uint16_t* d_out) {
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(blk[j]));
    const uint16_t dh = f32_to_fp16(amax / 127.0f);
    const float d = fp16_to_f32(dh);
    const float inv = d > 0.0f ? 1.0f / d : 0.0f;
    *d_out = dh;
    for (int j = 0; j < 32; ++j) q[j] = static_cast<int8_t>(std::max(-127L, std::min(127L, std::lrintf(blk[j] * inv))));
}

inline void quant_block_q4(const float* blk, uint32_t* words, uint16_t* d_out) {
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(blk[j]));
    const uint16_t dh = f32_to_fp16(amax / 7.0f);
    const float d = fp16_to_f32(dh);
    const float inv = d > 0.0f ? 1.0f / d : 0.0f;
    *d_out = dh;
    for (int w = 0; w < 4; ++w) {
        uint32_t packed = 0;
        for (int k = 0; k < 8; ++k) {
            const long q = std::max(-7L, std::min(7L, std::lrintf(blk[w * 8 + k] * inv)));
            packed |= (static_cast<uint32_t>(q) & 0xFu) << (k * 4);
        }
        words[w] = packed;
    }
}

} // namespace

PackedVjQ8 pack_f32_to_vj_q8(const float* src, uint32_t rows, uint32_t cols) {
    VJ_CHECK(cols % 32 == 0);
    PackedVjQ8 out;
    out.rows = rows; out.cols = cols;
    const uint32_t nb = cols / 32;
    out.qs.resize(static_cast<size_t>(rows) * cols);
    out.ds.resize(static_cast<size_t>(rows) * nb);
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t b = 0; b < nb; ++b)
            quant_block_q8(src + static_cast<size_t>(r) * cols + b * 32, out.qs.data() + static_cast<size_t>(r) * cols + b * 32,
                           out.ds.data() + static_cast<size_t>(r) * nb + b);
    return out;
}

PackedVjQ4Dot8 pack_f32_to_vj_q4_dot8(const float* src, uint32_t rows, uint32_t cols, bool fwht64) {
    VJ_CHECK(cols % 32 == 0 && (!fwht64 || cols % 64 == 0));
    PackedVjQ4Dot8 out;
    out.rows = rows; out.cols = cols; out.fwht64 = fwht64;
    const uint32_t nb = cols / 32, nw = cols / 8;
    out.qs.resize(static_cast<size_t>(rows) * nw);
    out.ds.resize(static_cast<size_t>(rows) * nb);
    std::vector<float> tmp(cols);
    for (uint32_t r = 0; r < rows; ++r) {
        const float* x = src + static_cast<size_t>(r) * cols;
        if (fwht64) {
            std::memcpy(tmp.data(), x, cols * sizeof(float));
            for (uint32_t c = 0; c < cols; c += 64) fwht64_inplace(tmp.data() + c);
            x = tmp.data();
        }
        for (uint32_t b = 0; b < nb; ++b)
            quant_block_q4(x + b * 32, out.qs.data() + static_cast<size_t>(r) * nw + b * 4,
                           out.ds.data() + static_cast<size_t>(r) * nb + b);
    }
    return out;
}

PackedVjQ8 repack_to_vj_q8(DiskQuantType t, const uint8_t* src, uint32_t rows, uint32_t cols) {
    const auto& reg = DecoderRegistry::instance();
    const uint64_t rb = reg.row_bytes(t, cols);
    VJ_CHECK(rb > 0 && cols % 32 == 0);
    if (t == DiskQuantType::Q8_0) {  // lossless: identical int8 codes and FP16 scale per 32
        PackedVjQ8 out;
        out.rows = rows; out.cols = cols;
        out.qs.resize(static_cast<size_t>(rows) * cols);
        out.ds.resize(static_cast<size_t>(rows) * cols / 32);
        const auto* b = reinterpret_cast<const block_q8_0*>(src);
        for (size_t i = 0; i < out.ds.size(); ++i) {
            out.ds[i] = b[i].d;
            std::memcpy(out.qs.data() + i * 32, b[i].qs, 32);
        }
        return out;
    }
    if (t == DiskQuantType::Q2_0) {  // lossless: {-1,0,1,2} codes, the 64-block scale repeats per 32
        PackedVjQ8 out;
        out.rows = rows; out.cols = cols;
        out.qs.resize(static_cast<size_t>(rows) * cols);
        out.ds.resize(static_cast<size_t>(rows) * cols / 32);
        const auto* b = reinterpret_cast<const block_q2_0*>(src);
        for (size_t i = 0; i < out.qs.size() / 64; ++i) {
            out.ds[2 * i] = out.ds[2 * i + 1] = b[i].d;
            for (int j = 0; j < 64; ++j)
                out.qs[i * 64 + j] = static_cast<int8_t>(((b[i].qs[j / 4] >> ((j % 4) * 2)) & 3) - 1);
        }
        return out;
    }
    std::vector<float> f(static_cast<size_t>(rows) * cols);
    for (uint32_t r = 0; r < rows; ++r)
        reg.decode_row_f32(t, src + r * rb, f.data() + static_cast<size_t>(r) * cols, cols);
    return pack_f32_to_vj_q8(f.data(), rows, cols);
}

PackedVjQ4Dot8 repack_to_vj_q4_dot8(DiskQuantType t, const uint8_t* src, uint32_t rows, uint32_t cols, bool fwht64) {
    const auto& reg = DecoderRegistry::instance();
    const uint64_t rb = reg.row_bytes(t, cols);
    VJ_CHECK(rb > 0 && cols % 32 == 0);
    if (t == DiskQuantType::Q2_0 && !fwht64) {  // lossless: codes fit signed 4-bit
        PackedVjQ4Dot8 out;
        out.rows = rows; out.cols = cols;
        out.qs.resize(static_cast<size_t>(rows) * cols / 8);
        out.ds.resize(static_cast<size_t>(rows) * cols / 32);
        const auto* b = reinterpret_cast<const block_q2_0*>(src);
        for (size_t i = 0; i < out.qs.size() / 8; ++i) {
            out.ds[2 * i] = out.ds[2 * i + 1] = b[i].d;
            for (int w = 0; w < 8; ++w) {
                uint32_t packed = 0;
                for (int k = 0; k < 8; ++k) {
                    const int j = w * 8 + k;
                    const int q = ((b[i].qs[j / 4] >> ((j % 4) * 2)) & 3) - 1;
                    packed |= (static_cast<uint32_t>(q) & 0xFu) << (k * 4);
                }
                out.qs[i * 8 + w] = packed;
            }
        }
        return out;
    }
    std::vector<float> f(static_cast<size_t>(rows) * cols);
    for (uint32_t r = 0; r < rows; ++r)
        reg.decode_row_f32(t, src + r * rb, f.data() + static_cast<size_t>(r) * cols, cols);
    return pack_f32_to_vj_q4_dot8(f.data(), rows, cols, fwht64);
}

PackedVjQ4A8 pack_f32_to_vj_q4_a8(const float* src, uint32_t rows, uint32_t cols, bool tcc_pad_dm) {
    VJ_CHECK(cols % 32 == 0);
    PackedVjQ4A8 out;
    out.rows = rows;
    out.cols = cols;
    const uint32_t nb = cols / 32u;
    const TccRowLayout qs_lay = select_tcc_odd256_layout(cols / 2u);
    const TccRowLayout dm_lay = tcc_pad_dm ? select_tcc_odd256_layout(nb * 4u)
                                           : TccRowLayout{TccLayoutMode::NaturalOdd256, nb * 4u, nb * 4u, 0u};
    out.qs_row_stride_bytes = qs_lay.stride_bytes;
    out.dm_row_stride_u32 = dm_lay.stride_bytes / 4u;
    out.qs.assign(static_cast<size_t>(rows) * out.qs_row_stride_bytes, 0);
    out.dm.assign(static_cast<size_t>(rows) * out.dm_row_stride_u32, 0);
    for (uint32_t r = 0; r < rows; ++r) {
        const float* srow = src + static_cast<size_t>(r) * cols;
        uint8_t* dst_row = out.qs.data() + static_cast<size_t>(r) * out.qs_row_stride_bytes;
        uint32_t* dmrow = out.dm.data() + static_cast<size_t>(r) * out.dm_row_stride_u32;
        for (uint32_t b = 0; b < nb; ++b) {
            const float* blk = srow + b * 32u;
            float mn = blk[0], mx = blk[0];
            for (uint32_t i = 1; i < 32u; ++i) { mn = std::min(mn, blk[i]); mx = std::max(mx, blk[i]); }
            const uint16_t dh = f32_to_fp16((mx - mn) / 15.0f);
            const uint16_t mh = f32_to_fp16(mn);
            const float d_eff = fp16_to_f32(dh);
            const float m_eff = fp16_to_f32(mh);
            const float inv_d = (d_eff > 0.0f) ? (1.0f / d_eff) : 0.0f;
            dmrow[b] = static_cast<uint32_t>(dh) | (static_cast<uint32_t>(mh) << 16);
            uint8_t* qblk = dst_row + b * 16u;
            for (uint32_t i = 0; i < 16u; ++i) {
                int q0 = std::max(0, std::min(15, static_cast<int>(std::lrintf((blk[i] - m_eff) * inv_d))));
                int q1 = std::max(0, std::min(15, static_cast<int>(std::lrintf((blk[i + 16u] - m_eff) * inv_d))));
                qblk[i] = static_cast<uint8_t>(q0 | (q1 << 4));
            }
        }
    }
    return out;
}

PackedVjQ4A8 repack_to_vj_q4_a8(DiskQuantType t, const uint8_t* src, uint32_t rows, uint32_t cols, bool tcc_pad_dm) {
    const auto& reg = DecoderRegistry::instance();
    const uint64_t rb = reg.row_bytes(t, cols);
    VJ_CHECK(rb > 0 && cols % 32 == 0);
    std::vector<float> f(static_cast<size_t>(rows) * cols);
    for (uint32_t r = 0; r < rows; ++r)
        reg.decode_row_f32(t, src + r * rb, f.data() + static_cast<size_t>(r) * cols, cols);
    return pack_f32_to_vj_q4_a8(f.data(), rows, cols, tcc_pad_dm);
}

void dequantize_vj_q4_a8(const PackedVjQ4A8& p, float* out) {
    const uint32_t nb = p.cols / 32u;
    for (uint32_t r = 0; r < p.rows; ++r) {
        const uint8_t* qrow = p.qs.data() + static_cast<size_t>(r) * p.qs_row_stride_bytes;
        const uint32_t* dmrow = p.dm.data() + static_cast<size_t>(r) * p.dm_row_stride_u32;
        float* y = out + static_cast<size_t>(r) * p.cols;
        for (uint32_t b = 0; b < nb; ++b) {
            const float d = fp16_to_f32(static_cast<uint16_t>(dmrow[b] & 0xFFFFu));
            const float m = fp16_to_f32(static_cast<uint16_t>(dmrow[b] >> 16));
            const uint8_t* qblk = qrow + b * 16u;
            for (uint32_t i = 0; i < 16u; ++i) {
                y[b * 32u + i]       = d * static_cast<float>(qblk[i] & 0xFu) + m;
                y[b * 32u + 16u + i] = d * static_cast<float>(qblk[i] >> 4) + m;
            }
        }
    }
}

void dequantize_vj_q8(const PackedVjQ8& p, float* out) {
    const uint32_t nb = p.cols / 32;
    for (uint32_t r = 0; r < p.rows; ++r)
        for (uint32_t b = 0; b < nb; ++b) {
            const float d = fp16_to_f32(p.ds[static_cast<size_t>(r) * nb + b]);
            for (int j = 0; j < 32; ++j) {
                const size_t i = static_cast<size_t>(r) * p.cols + b * 32 + j;
                out[i] = d * p.qs[i];
            }
        }
}

void dequantize_vj_q4_dot8(const PackedVjQ4Dot8& p, float* out) {
    const uint32_t nb = p.cols / 32, nw = p.cols / 8;
    for (uint32_t r = 0; r < p.rows; ++r) {
        float* y = out + static_cast<size_t>(r) * p.cols;
        for (uint32_t b = 0; b < nb; ++b) {
            const float d = fp16_to_f32(p.ds[static_cast<size_t>(r) * nb + b]);
            for (int w = 0; w < 4; ++w) {
                const uint32_t word = p.qs[static_cast<size_t>(r) * nw + b * 4 + w];
                for (int k = 0; k < 8; ++k) {
                    int nib = static_cast<int>((word >> (k * 4)) & 0xFu);
                    if (nib >= 8) nib -= 16;
                    y[b * 32 + w * 8 + k] = d * static_cast<float>(nib);
                }
            }
        }
        if (p.fwht64)
            for (uint32_t c = 0; c < p.cols; c += 64) fwht64_inplace(y + c);  // H is self-inverse
    }
}

// =================================================================================================
// VJ_IQ_PACKED: lossless split planes
// =================================================================================================
std::vector<IqSegment> iq_block_segments(DiskQuantType t) {
    using P = IqPlane;
    std::vector<IqSegment> s;
    switch (t) {
        case DiskQuantType::IQ2_XXS:  // per 32: 4 grid index bytes + 4 bytes (7-bit sign groups + 4-bit scale)
            s.push_back({0, 2, P::Scale});
            for (uint16_t g = 0; g < 8; ++g) {
                s.push_back({static_cast<uint16_t>(2 + 8 * g), 4, P::Index});
                s.push_back({static_cast<uint16_t>(6 + 8 * g), 4, P::Sign});
            }
            break;
        case DiskQuantType::IQ2_XS:  // uint16 = 9-bit index | 7-bit sign index (kept together)
            s = {{0, 2, P::Scale}, {2, 64, P::Index}, {66, 8, P::Scale}};
            break;
        case DiskQuantType::IQ2_S:
            s = {{0, 2, P::Scale}, {2, 32, P::Index}, {34, 32, P::Sign}, {66, 8, P::Index}, {74, 8, P::Scale}};
            break;
        case DiskQuantType::IQ3_XXS:
            s = {{0, 2, P::Scale}, {2, 64, P::Index}, {66, 32, P::Sign}};
            break;
        case DiskQuantType::IQ3_S:
            s = {{0, 2, P::Scale}, {2, 64, P::Index}, {66, 8, P::Index}, {74, 32, P::Sign}, {106, 4, P::Scale}};
            break;
        case DiskQuantType::IQ4_XS:
            s = {{0, 8, P::Scale}, {8, 128, P::Index}};
            break;
        case DiskQuantType::IQ4_NL: case DiskQuantType::Q4_0: case DiskQuantType::Q2_0:
            s = {{0, 2, P::Scale}, {2, 16, P::Index}};
            break;
        default: {
            const GgmlTypeTraits tr = ggml_type_traits(static_cast<uint32_t>(t));
            if (tr.block_bytes) s = {{0, static_cast<uint16_t>(tr.block_bytes), P::Index}};
        }
    }
    return s;
}

PackedVjIq repack_to_vj_iq(DiskQuantType t, const uint8_t* src, uint32_t rows, uint32_t cols) {
    const auto& reg = DecoderRegistry::instance();
    const FormatInfo* fi = reg.find(t);
    VJ_CHECK(fi && cols % fi->block_elems == 0);
    PackedVjIq out;
    out.orig_type = t; out.rows = rows; out.cols = cols;
    const auto segs = iq_block_segments(t);
    const uint64_t n_blocks = static_cast<uint64_t>(rows) * cols / fi->block_elems;
    for (uint64_t b = 0; b < n_blocks; ++b) {
        const uint8_t* blk = src + b * fi->block_bytes;
        for (const auto& sg : segs) {
            auto& plane = sg.plane == IqPlane::Scale ? out.scale_plane
                        : sg.plane == IqPlane::Sign  ? out.sign_plane : out.index_plane;
            plane.insert(plane.end(), blk + sg.offset, blk + sg.offset + sg.bytes);
        }
    }
    return out;
}

std::vector<uint8_t> unpack_vj_iq_to_disk(const PackedVjIq& p) {
    const FormatInfo* fi = DecoderRegistry::instance().find(p.orig_type);
    const auto segs = iq_block_segments(p.orig_type);
    const uint64_t n_blocks = static_cast<uint64_t>(p.rows) * p.cols / fi->block_elems;
    std::vector<uint8_t> out(n_blocks * fi->block_bytes);
    size_t pos[3] = {0, 0, 0};
    const std::vector<uint8_t>* planes[3] = {&p.scale_plane, &p.sign_plane, &p.index_plane};
    for (uint64_t b = 0; b < n_blocks; ++b)
        for (const auto& sg : segs) {
            const int k = static_cast<int>(sg.plane);
            std::memcpy(out.data() + b * fi->block_bytes + sg.offset, planes[k]->data() + pos[k], sg.bytes);
            pos[k] += sg.bytes;
        }
    return out;
}

void dequantize_vj_iq(const PackedVjIq& p, float* out) {
    const auto disk = unpack_vj_iq_to_disk(p);
    const auto& reg = DecoderRegistry::instance();
    const uint64_t rb = reg.row_bytes(p.orig_type, p.cols);
    for (uint32_t r = 0; r < p.rows; ++r)
        reg.decode_row_f32(p.orig_type, disk.data() + r * rb, out + static_cast<size_t>(r) * p.cols, p.cols);
}

// =================================================================================================
// Roofline profile
// =================================================================================================
double GpuRooflineInfo::i_star(double reuse_factor) const {
    if (bw_hbm_gbs <= 0.0 || reuse_factor <= 0.0) return 4.25;
    return (static_cast<double>(n_cu) * 64.0 * sclk_ghz * eta) / (bw_hbm_gbs * reuse_factor);
}

double HardwareRooflineProfile::min_i_star(double reuse_factor) const {
    if (gpus.empty()) return 4.25 / std::max(1.0, reuse_factor);
    double m = 1e9;
    for (const auto& g : gpus) m = std::min(m, g.i_star(reuse_factor));
    return m;
}

std::string HardwareRooflineProfile::default_cache_path() {
    const char* home = std::getenv("HOME");
    if (!home || !home[0]) return "/tmp/vegajet_hw_profile.json";
    return std::string(home) + "/.config/vegajet/hw_profile.json";
}

bool HardwareRooflineProfile::save_json(const std::string& path, const HardwareRooflineProfile& prof) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path p(path);
    if (p.has_parent_path()) fs::create_directories(p.parent_path(), ec);
    std::ofstream ofs(path);
    if (!ofs.is_open()) return false;
    ofs << "{\n  \"gpus\": [\n";
    for (size_t i = 0; i < prof.gpus.size(); ++i) {
        const auto& g = prof.gpus[i];
        ofs << "    {\"bdf\": \"" << g.id << "\"";
        if (!g.unique_id.empty()) ofs << ", \"unique_id\": \"" << g.unique_id << "\"";
        ofs << ", \"n_cu\": " << g.n_cu << ", \"sclk_ghz\": " << g.sclk_ghz
            << ", \"mclk_mhz\": " << g.mclk_mhz << ", \"bw_hbm_gbs\": " << g.bw_hbm_gbs
            << ", \"power_cap_w\": " << g.power_cap_w << ", \"eta\": " << g.eta << ", \"i_star\": " << g.i_star()
            << "}" << (i + 1 < prof.gpus.size() ? "," : "") << "\n";
    }
    ofs << "  ]\n}\n";
    return ofs.good();
}

bool HardwareRooflineProfile::load_json(const std::string& path, HardwareRooflineProfile& out) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return false;
    out.gpus.clear();
    auto num = [](const std::string& s, const char* key, double& v) {
        auto p = s.find(key);
        if (p == std::string::npos) return false;
        p = s.find(':', p);
        if (p == std::string::npos) return false;
        v = std::atof(s.c_str() + p + 1);
        return true;
    };
    auto str_field = [](const std::string& s, const char* key, std::string& v) {
        auto p = s.find(key);
        if (p == std::string::npos) return false;
        p = s.find(':', p);
        if (p == std::string::npos) return false;
        const auto q1 = s.find('"', p + 1);
        if (q1 == std::string::npos) return false;
        const auto q2 = s.find('"', q1 + 1);
        if (q2 == std::string::npos) return false;
        v = s.substr(q1 + 1, q2 - q1 - 1);
        return true;
    };
    std::string line;
    while (std::getline(ifs, line)) {
        const auto bp = line.find("\"bdf\"");
        const auto up = line.find("\"unique_id\"");
        if (bp == std::string::npos && up == std::string::npos) continue;
        GpuRooflineInfo g;
        (void)str_field(line, "\"bdf\"", g.id);
        if (str_field(line, "\"unique_id\"", g.unique_id)) {
            for (char& c : g.unique_id) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
        }
        double v = 0;
        if (num(line, "\"n_cu\"", v)) g.n_cu = static_cast<uint32_t>(v);
        if (num(line, "\"sclk_ghz\"", v)) g.sclk_ghz = v;
        if (num(line, "\"mclk_mhz\"", v)) g.mclk_mhz = v;
        if (num(line, "\"bw_hbm_gbs\"", v)) g.bw_hbm_gbs = v;
        if (num(line, "\"power_cap_w\"", v)) g.power_cap_w = v;
        if (num(line, "\"eta\"", v)) g.eta = v;
        out.gpus.push_back(g);
    }
    return !out.gpus.empty();
}

// =================================================================================================
// Knapsack
// =================================================================================================
uint64_t VramBudgetPlanner::estimate_bytes(NativeFormat fmt, DiskQuantType disk, uint64_t rows, uint64_t cols,
                                           uint64_t experts) {
    const uint64_t elems = rows * cols * experts;
    switch (fmt) {
        case NativeFormat::VJ_Q2_DOT8: return elems / 4 + elems / 64 * 4 + 16;
        case NativeFormat::VJ_Q3_DOT8: return elems * 3 / 8 + elems / 64 * 4 + 16;
        case NativeFormat::VJ_Q8_DP4A: return elems + elems / 32 * 2;
        case NativeFormat::VJ_Q4_DOT8: return elems / 2 + elems / 32 * 2;
        case NativeFormat::VJ_Q4_A8:   return elems / 2 + elems / 32 * 4;
        case NativeFormat::VJ_IQ_PACKED: {
            const uint64_t rb = ggml_row_bytes(static_cast<uint32_t>(disk), cols);
            return rb ? rb * rows * experts : elems / 2;
        }
    }
    return elems;
}

// Decode cost (VALU instructions per weight byte) per disk format: bit-unpacking, scale assembly and
// codebook lookups. Inherited calibration constants; refresh with `vegajet doctor --bench` + kernel timing.
double VramBudgetPlanner::format_inst_per_byte(DiskQuantType t) {
    switch (t) {
        case DiskQuantType::F32: case DiskQuantType::F16: case DiskQuantType::BF16: return 0.50;
        case DiskQuantType::F8_E4M3: return 1.10;
        case DiskQuantType::Q8_0: case DiskQuantType::Q8_1: return 0.78;
        case DiskQuantType::Q4_0: return 2.60;
        case DiskQuantType::Q4_1: return 2.85;
        case DiskQuantType::Q5_0: case DiskQuantType::Q5_1: return 3.40;
        case DiskQuantType::IQ4_NL: return 2.80;
        case DiskQuantType::Q4_K: return 3.10;
        case DiskQuantType::IQ4_XS: return 3.20;
        case DiskQuantType::Q5_K: return 4.60;
        case DiskQuantType::Q6_K: return 5.90;
        case DiskQuantType::Q3_K: return 5.40;
        case DiskQuantType::IQ3_S: return 5.60;
        case DiskQuantType::IQ3_XXS: return 5.80;
        case DiskQuantType::Q2_K: return 6.20;
        case DiskQuantType::Q2_0: return 14.50;
        case DiskQuantType::IQ2_XS: return 7.10;
        case DiskQuantType::IQ2_XXS: return 7.10;
        case DiskQuantType::IQ2_S: return 7.40;
        default: return 4.0;
    }
}

double VramBudgetPlanner::native_inst_per_byte(NativeFormat fmt, DiskQuantType disk, bool w4a4) {
    switch (fmt) {
        case NativeFormat::VJ_Q2_DOT8: return 1.0; // analytical, not measured
        case NativeFormat::VJ_Q3_DOT8: return 1.5; // analytical, not measured
        case NativeFormat::VJ_Q8_DP4A: return 0.78;
        case NativeFormat::VJ_Q4_DOT8: return w4a4 ? 0.53 : 1.41;
        case NativeFormat::VJ_Q4_A8:   return w4a4 ? 0.65 : 1.35;
        case NativeFormat::VJ_IQ_PACKED: return format_inst_per_byte(disk);
    }
    return 1.0;
}

double VramBudgetPlanner::estimate_read_ms(uint64_t bytes, double ipb, double bw_gbs, double i_star, double hit_rate) {
    if (bw_gbs <= 0.0 || i_star <= 0.0 || hit_rate <= 0.0) return 0.0;
    const double bw_ms = static_cast<double>(bytes) / (bw_gbs * 1e6);
    return bw_ms * std::max(1.0, ipb / i_star) * hit_rate;
}

namespace {
bool is_high_precision(DiskQuantType t) {
    return t == DiskQuantType::F32 || t == DiskQuantType::F16 || t == DiskQuantType::BF16 ||
           t == DiskQuantType::F8_E4M3 || t == DiskQuantType::Q8_0 || t == DiskQuantType::Q8_1 ||
           t == DiskQuantType::Q6_K || t == DiskQuantType::Q5_K || t == DiskQuantType::Q5_0 ||
           t == DiskQuantType::Q5_1;
}
} // namespace

std::vector<TensorPlanDecision> VramBudgetPlanner::plan_roofline_knapsack(const std::vector<TensorPlanSpec>& specs,
                                                                          uint64_t vram_budget_bytes,
                                                                          uint64_t reserve_bytes,
                                                                          const HardwareRooflineProfile& hw,
                                                                          double reuse) {
    const uint64_t avail = vram_budget_bytes > reserve_bytes ? vram_budget_bytes - reserve_bytes : 0;
    const double i_star = hw.min_i_star(reuse);
    const double bw = hw.gpus.empty() ? 800.0 : hw.gpus[0].bw_hbm_gbs;

    // Pass 1: minimal-footprint baseline.
    std::vector<TensorPlanDecision> out(specs.size());
    uint64_t used = 0;
    for (size_t i = 0; i < specs.size(); ++i) {
        const auto& s = specs[i];
        NativeFormat f = NativeFormat::VJ_IQ_PACKED;
        if (s.disk_type == DiskQuantType::Q8_0 || s.disk_type == DiskQuantType::Q8_1 ||
            s.disk_type == DiskQuantType::F16 || s.disk_type == DiskQuantType::BF16 ||
            s.disk_type == DiskQuantType::F32 || s.disk_type == DiskQuantType::F8_E4M3)
            f = NativeFormat::VJ_Q8_DP4A;
        else if (s.disk_type == DiskQuantType::Q4_0 || s.disk_type == DiskQuantType::Q4_1 ||
                 s.disk_type == DiskQuantType::Q4_K)
            f = NativeFormat::VJ_Q4_DOT8;  // same bytes; VJ_IQ_PACKED kernels cover IQ / sub-4-bit only
        const uint64_t b = estimate_bytes(f, s.disk_type, s.rows, s.cols, s.experts);
        // FWHT-64 on requantized 4-bit baselines: measured on real Q4_K / IQ3_XXS weights it lifts cosine
        // from 0.994 (0.969 for IQ3_XXS) to >= 0.995 by spreading outliers across each 64-block.
        const bool fwht = f == NativeFormat::VJ_Q4_DOT8 && s.cols % 64 == 0;
        out[i] = TensorPlanDecision{s.name, f, fwht, b, 0.0, 0.0, false};
        used += b;
    }

    // Pass 2: best positive-ROI candidate per tensor.
    struct Cand { size_t idx; NativeFormat target; bool fwht; uint64_t extra; double saved; double roi; };
    std::vector<Cand> cands;
    for (size_t i = 0; i < specs.size(); ++i) {
        const auto& s = specs[i];
        if (s.role == TensorRole::Other || out[i].target_format == NativeFormat::VJ_Q8_DP4A) continue;
        const double hr = s.hit_rate > 0.0 ? s.hit_rate : 1.0;
        const uint64_t base_b = out[i].estimated_vram_bytes;
        const double base_ms = estimate_read_ms(base_b, native_inst_per_byte(out[i].target_format, s.disk_type, s.prefer_w4a4),
                                                bw, i_star, hr);
        Cand best{};
        bool found = false, saw_neg = false;
        for (NativeFormat tf : {NativeFormat::VJ_Q4_DOT8, NativeFormat::VJ_Q8_DP4A}) {
            if (tf == out[i].target_format) continue;
            // Never quantize a >=5-bit disk tensor down to 4 bits.
            if (is_high_precision(s.disk_type) && tf == NativeFormat::VJ_Q4_DOT8) continue;
            const uint64_t nb = estimate_bytes(tf, s.disk_type, s.rows, s.cols, s.experts);
            const double saved = base_ms - estimate_read_ms(nb, native_inst_per_byte(tf, s.disk_type, s.prefer_w4a4), bw, i_star, hr);
            if (saved <= 0.0) { saw_neg = true; continue; }
            const uint64_t extra = nb > base_b ? nb - base_b : 0;
            const double roi = saved / std::max(1e-6, static_cast<double>(extra) / (1024.0 * 1024.0 * 1024.0));
            if (!found || roi > best.roi) {
                best = Cand{i, tf, tf == NativeFormat::VJ_Q4_DOT8 && s.disk_type != DiskQuantType::Q2_0 && s.cols % 64 == 0,
                            extra, saved, roi};
                found = true;
            }
        }
        if (found) cands.push_back(best);
        else if (saw_neg) out[i].rejected_negative_roi = true;
    }

    // Pass 3: 0/1 knapsack by ROI density.
    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
        return std::fabs(a.roi - b.roi) > 1e-9 ? a.roi > b.roi : a.saved > b.saved;
    });
    for (const auto& c : cands) {
        if (used + c.extra > avail) continue;
        used += c.extra;
        auto& d = out[c.idx];
        d.target_format = c.target;
        d.use_fwht64 = c.fwht;
        d.estimated_vram_bytes += c.extra;
        d.expected_ms_saved = c.saved;
        d.roi_ms_per_gib = c.roi;
    }
    return out;
}

// =================================================================================================
// TCC L2 anti-camping
// =================================================================================================
TccRowLayout select_tcc_layout(uint32_t raw_row_bytes) {
    TccRowLayout lay;
    lay.raw_row_bytes = raw_row_bytes;
    if (raw_row_bytes == 0) return lay;
    const uint32_t tiles = (raw_row_bytes + kTccTileBytes - 1) / kTccTileBytes;
    if (tiles >= kTccChannels && tiles % kTccChannels == 0 && raw_row_bytes % kTccTileBytes == 0) {
        lay.mode = TccLayoutMode::XorTileSwizzle;
        lay.stride_bytes = raw_row_bytes;
        lay.tiles_per_row = tiles;
    } else if (raw_row_bytes % kTccTileBytes == 0 && (tiles & 1u)) {
        lay.mode = TccLayoutMode::NaturalOdd256;
        lay.stride_bytes = raw_row_bytes;
        lay.tiles_per_row = tiles;
    } else {
        const uint32_t padded = (tiles & 1u) ? tiles : tiles + 1;
        lay.mode = TccLayoutMode::PadOdd256;
        lay.stride_bytes = padded * kTccTileBytes;
        lay.tiles_per_row = padded;
    }
    return lay;
}

TccRowLayout select_tcc_odd256_layout(uint32_t raw_row_bytes) {
    TccRowLayout lay;
    lay.raw_row_bytes = raw_row_bytes;
    if (raw_row_bytes == 0) return lay;
    const uint32_t tiles = (raw_row_bytes + kTccTileBytes - 1u) / kTccTileBytes;
    const uint32_t odd_tiles = (tiles & 1u) ? tiles : (tiles + 1u);
    lay.mode = (raw_row_bytes % kTccTileBytes == 0 && (tiles & 1u)) ? TccLayoutMode::NaturalOdd256 : TccLayoutMode::PadOdd256;
    lay.stride_bytes = odd_tiles * kTccTileBytes;
    lay.tiles_per_row = odd_tiles;
    return lay;
}

std::vector<uint8_t> pack_tcc_deswizzled(const uint8_t* src, uint32_t rows, const TccRowLayout& lay) {
    std::vector<uint8_t> out(static_cast<size_t>(rows) * lay.stride_bytes, 0);
    for (uint32_t r = 0; r < rows; ++r) {
        const uint8_t* s = src + static_cast<size_t>(r) * lay.raw_row_bytes;
        uint8_t* d = out.data() + static_cast<size_t>(r) * lay.stride_bytes;
        if (lay.mode == TccLayoutMode::XorTileSwizzle) {
            const uint32_t mask = r & 15u;
            for (uint32_t kt = 0; kt < lay.tiles_per_row; ++kt)
                std::memcpy(d + static_cast<size_t>(kt ^ mask) * kTccTileBytes, s + static_cast<size_t>(kt) * kTccTileBytes, kTccTileBytes);
        } else {
            std::memcpy(d, s, lay.raw_row_bytes);
        }
    }
    return out;
}

std::vector<uint8_t> unpack_tcc_deswizzled(const uint8_t* swz, uint32_t rows, const TccRowLayout& lay) {
    std::vector<uint8_t> out(static_cast<size_t>(rows) * lay.raw_row_bytes);
    for (uint32_t r = 0; r < rows; ++r) {
        const uint8_t* s = swz + static_cast<size_t>(r) * lay.stride_bytes;
        uint8_t* d = out.data() + static_cast<size_t>(r) * lay.raw_row_bytes;
        if (lay.mode == TccLayoutMode::XorTileSwizzle) {
            const uint32_t mask = r & 15u;
            for (uint32_t kt = 0; kt < lay.tiles_per_row; ++kt)
                std::memcpy(d + static_cast<size_t>(kt) * kTccTileBytes, s + static_cast<size_t>(kt ^ mask) * kTccTileBytes, kTccTileBytes);
        } else {
            std::memcpy(d, s, lay.raw_row_bytes);
        }
    }
    return out;
}

uint32_t tcc_channel_spread_count(uint32_t rows, const TccRowLayout& lay, uint32_t logical_tile) {
    uint32_t seen = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t phys = lay.mode == TccLayoutMode::XorTileSwizzle ? (logical_tile ^ (r & 15u)) : logical_tile;
        const uint64_t addr = static_cast<uint64_t>(r) * lay.stride_bytes + static_cast<uint64_t>(phys) * kTccTileBytes;
        seen |= 1u << ((addr / kTccTileBytes) & 15u);
    }
    return static_cast<uint32_t>(__builtin_popcount(seen));
}


namespace {
uint32_t sub4_bits(NativeFormat f) {
    if (f == NativeFormat::VJ_Q2_DOT8) return 2;
    if (f == NativeFormat::VJ_Q3_DOT8) return 3;
    throw std::invalid_argument("sub4 format must be Q2/Q3 DOT8");
}
}
PackedVjSub4 pack_f32_to_vj_sub4(const float* src, uint32_t rows, uint32_t cols,
                               NativeFormat format, uint32_t group) {
    const uint32_t bits = sub4_bits(format), top = (1u<<bits)-1;
    if (!rows || !cols || (group != 64 && group != 128) || cols%group)
        throw std::invalid_argument("sub4 requires nonempty rows and cols divisible by group64/128");
    PackedVjSub4 p; p.format=format; p.rows=rows; p.cols=cols; p.group=group;
    const size_t n=size_t(rows)*cols;
    p.qs.assign(n/32*bits+4,0); p.dm.resize(n/group);
    for (size_t g=0;g<n/group;++g) {
        float lo=src[g*group], hi=lo;
        for (uint32_t j=0;j<group;++j) {
            float v=src[g*group+j];
            if (!std::isfinite(v)) throw std::invalid_argument("nonfinite sub4 weight");
            lo=std::min(lo,v); hi=std::max(hi,v);
        }
        float d=hi==lo ? 1.f : (hi-lo)/top;
        uint16_t dh=f32_to_fp16(d);
        d=fp16_to_f32(dh);
        if (d==0) { dh=1; d=fp16_to_f32(dh); }
        uint16_t zh=f32_to_fp16(-lo/d);
        float z=fp16_to_f32(zh);
        if (!std::isfinite(d) || !std::isfinite(z))
            throw std::invalid_argument("sub4 affine metadata exceeds fp16 range");
        p.dm[g]=uint32_t(dh)|(uint32_t(zh)<<16);
        for(uint32_t j=0;j<group;++j) {
            size_t i=g*group+j, b=i/32*bits; uint32_t k=i%32;
            // Explicit nearest-even rounding, independent of the host rounding mode.
            float v=std::clamp(src[i]/d+z,0.f,float(top));
            uint32_t q=uint32_t(std::floor(v));
            if(v-q>0.5f || (v-q==0.5f && (q&1))) ++q;
            p.qs[b+k/16] |= (q&3)<<(4*((k%16)/2)+2*(k%2));
            if(bits==3) p.qs[b+2] |= (q>>2)<<(4*((k%16)/2)+2*(k/16)+k%2);
        }
    }
    return p;
}
PackedVjSub4 pack_f16_to_vj_sub4(const uint16_t* src,uint32_t r,uint32_t c,NativeFormat f,uint32_t g) {
    std::vector<float> v(size_t(r)*c);
    for(size_t i=0;i<v.size();++i) v[i]=fp16_to_f32(src[i]);
    return pack_f32_to_vj_sub4(v.data(),r,c,f,g);
}
uint32_t sub4_code(const PackedVjSub4& p,size_t i) {
    uint32_t bits=sub4_bits(p.format), k=i%32; size_t b=i/32*bits;
    if(p.tcc_xor) b=((i/p.cols)*(p.cols/32)+((i%p.cols/32)^((i/p.cols)&15)))*bits;
    uint32_t q=(p.qs[b+k/16]>>(4*((k%16)/2)+2*(k%2)))&3;
    if(bits==3) q|=(((p.qs[b+2]>>(4*((k%16)/2)+2*(k/16)+k%2))&1)<<2);
    return q;
}
void dequantize_vj_sub4(const PackedVjSub4& p,float* out) {
    for(size_t i=0;i<size_t(p.rows)*p.cols;++i) {
        uint32_t dm=p.tcc_xor ? p.dm[(i/p.cols)*(p.cols/32)+((i%p.cols/32)^((i/p.cols)&15))] : p.dm[i/p.group];
        out[i]=fp16_to_f32(dm&65535)*(float(sub4_code(p,i))-fp16_to_f32(dm>>16));
    }
}
void gemv_vj_sub4(const PackedVjSub4& p,const int8_t* x,const float* scales,float* y,uint32_t M) {
    for(uint32_t m=0;m<M;++m) for(uint32_t r=0;r<p.rows;++r) {
        float acc=0;
        for(uint32_t b=0;b<p.cols/32;++b) {
            int dot=0,sum=0;
            for(uint32_t j=0;j<32;++j) {
                int a=x[size_t(m)*p.cols+b*32+j];
                if(a < -8 || a > 7) throw std::invalid_argument("activation outside int4");
                dot+=int(sub4_code(p,size_t(r)*p.cols+b*32+j))*a; sum+=a;
            }
            uint32_t dm=p.tcc_xor ? p.dm[size_t(r)*(p.cols/32)+(b^(r&15))] : p.dm[size_t(r)*(p.cols/p.group)+b*32/p.group];
            acc+=fp16_to_f32(dm&65535)*scales[size_t(m)*(p.cols/32)+b]*(dot-fp16_to_f32(dm>>16)*sum);
        }
        y[size_t(m)*p.rows+r]=acc;
    }
}
uint32_t sub4_dot8_bits() {
    const char* e=std::getenv("VEGAJET_SUB4_DOT8");
    return e && std::strcmp(e,"q2")==0 ? 2 : e && std::strcmp(e,"q3")==0 ? 3 : 0;
}
bool tcc_xor_enabled() { const char* e=std::getenv("VEGAJET_TCC_XOR"); return e && std::strcmp(e,"1")==0; }
PackedVjQ4A8 repack_tcc_xor(const PackedVjQ4A8& p) {
    if(!p.cols || p.cols%512) throw std::invalid_argument("TCC XOR needs cols multiple of 512");
    auto q=p;
    for(uint32_t r=0;r<p.rows;++r) for(uint32_t b=0;b<p.cols/32;++b) {
        uint32_t phys=b^(r&15);
        std::memcpy(q.qs.data()+size_t(r)*p.qs_row_stride_bytes+phys*16,
                    p.qs.data()+size_t(r)*p.qs_row_stride_bytes+b*16,16);
        q.dm[size_t(r)*p.dm_row_stride_u32+phys]=p.dm[size_t(r)*p.dm_row_stride_u32+b];
    }
    return q;
}
PackedVjSub4 repack_tcc_xor(const PackedVjSub4& p) {
    if(p.tcc_xor || !p.cols || p.cols%512) throw std::invalid_argument("TCC XOR needs plain cols multiple of 512");
    auto q=p; q.tcc_xor=true; q.dm.resize(size_t(p.rows)*p.cols/32);
    const uint32_t bits=sub4_bits(p.format), nb=p.cols/32;
    for(uint32_t r=0;r<p.rows;++r) for(uint32_t b=0;b<nb;++b) {
        uint32_t phys=b^(r&15);
        for(uint32_t j=0;j<bits;++j) q.qs[(size_t(r)*nb+phys)*bits+j]=p.qs[(size_t(r)*nb+b)*bits+j];
        q.dm[size_t(r)*nb+phys]=p.dm[size_t(r)*(p.cols/p.group)+b*32/p.group];
    }
    return q;
}
bool milo_enabled() { const char* e=std::getenv("VEGAJET_MILO"); return e && std::strcmp(e,"1")==0; }
MiloResidual fit_milo(const float* original,const PackedVjSub4& p,uint32_t rank,uint32_t iterations,MiloStorage storage) {
    if(!iterations) throw std::invalid_argument("MiLo requires positive iteration count");
    if(rank!=16 && rank!=32) throw std::invalid_argument("MiLo rank must be 16/32");
    MiloResidual f; f.rows=p.rows; f.cols=p.cols; f.rank=rank;
    f.u.assign(size_t(p.rows)*rank,0); f.v.assign(size_t(rank)*p.cols,0);
    std::vector<float> residual(size_t(p.rows)*p.cols); dequantize_vj_sub4(p,residual.data());
    for(size_t i=0;i<residual.size();++i) residual[i]=original[i]-residual[i];
    std::vector<double> u(p.rows),v(p.cols);
    for(uint32_t k=0;k<rank;++k) {
        for(uint32_t c=0;c<p.cols;++c) v[c]=std::sin(double((c+1)*(k+1))*1.61803398875);
        for(uint32_t it=0;it<iterations;++it) {
            std::fill(u.begin(),u.end(),0);
            for(uint32_t r=0;r<p.rows;++r) for(uint32_t c=0;c<p.cols;++c) u[r]+=residual[size_t(r)*p.cols+c]*v[c];
            std::fill(v.begin(),v.end(),0);
            for(uint32_t r=0;r<p.rows;++r) for(uint32_t c=0;c<p.cols;++c) v[c]+=residual[size_t(r)*p.cols+c]*u[r];
            double norm=0; for(double a:v) norm+=a*a; norm=std::sqrt(norm);
            if(norm<1e-30) break;
            for(double& a:v) a/=norm;
        }
        for(uint32_t c=0;c<p.cols;++c) f.v[size_t(k)*p.cols+c]=f32_to_fp16(float(v[c]));
        for(uint32_t r=0;r<p.rows;++r) {
            double a=0; for(uint32_t c=0;c<p.cols;++c) a+=residual[size_t(r)*p.cols+c]*fp16_to_f32(f.v[size_t(k)*p.cols+c]);
            f.u[size_t(r)*rank+k]=f32_to_fp16(float(a));
            for(uint32_t c=0;c<p.cols;++c) residual[size_t(r)*p.cols+c]-=fp16_to_f32(f.u[size_t(r)*rank+k])*fp16_to_f32(f.v[size_t(k)*p.cols+c]);
        }
    }
    if(storage==MiloStorage::Int3) {
        auto pack=[](const std::vector<uint16_t>& a,uint32_t rows,uint32_t cols,
                     std::vector<uint32_t>& q,std::vector<float>& scales) {
            q.assign((a.size()*3+31)/32+1,0); scales.resize(rows);
            for(uint32_t r=0;r<rows;++r) {
                float mx=0; for(uint32_t c=0;c<cols;++c) mx=std::max(mx,std::abs(fp16_to_f32(a[size_t(r)*cols+c])));
                float d=mx>0?mx/3:1; scales[r]=d;
                for(uint32_t c=0;c<cols;++c) {
                    size_t i=size_t(r)*cols+c, bit=i*3;
                    uint64_t v=uint32_t(std::clamp(int(std::round(fp16_to_f32(a[i])/d)),-3,3))&7;
                    v<<=bit%32; q[bit/32]|=uint32_t(v); q[bit/32+1]|=uint32_t(v>>32);
                }
            }
        };
        pack(f.u,f.rows,f.rank,f.u3,f.us); pack(f.v,f.rank,f.cols,f.v3,f.vs);
        f.u.clear(); f.u.shrink_to_fit(); f.v.clear(); f.v.shrink_to_fit(); f.storage=storage;
    }
    return f;
}
float milo_factor(const MiloResidual& f,bool u,size_t i) {
    if(f.storage==MiloStorage::Fp16) return fp16_to_f32((u?f.u:f.v)[i]);
    const auto& q=u?f.u3:f.v3; size_t bit=i*3;
    uint32_t v=uint32_t((uint64_t(q[bit/32])|(uint64_t(q[bit/32+1])<<32))>>(bit%32))&7;
    return float(int(v^4)-4)*(u?f.us[i/f.rank]:f.vs[i/f.cols]);
}
void apply_milo(const MiloResidual& f,const float* x,float* y) {
    for(uint32_t k=0;k<f.rank;++k) {
        float a=0; for(uint32_t c=0;c<f.cols;++c) a+=milo_factor(f,false,size_t(k)*f.cols+c)*x[c];
        for(uint32_t r=0;r<f.rows;++r) y[r]+=milo_factor(f,true,size_t(r)*f.rank+k)*a;
    }
}

} // namespace vegajet::quant
