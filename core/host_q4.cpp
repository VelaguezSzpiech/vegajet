// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/host_q4.hpp"

#include <cmath>
#include <cstring>

namespace vegajet::hostq4 {

namespace {
float half_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; --exp; }
            man &= 0x3FFu;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

uint16_t float_to_half(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127;
    uint32_t man = bits & 0x7FFFFFu;
    if (((bits >> 23) & 0xFFu) == 0xFFu) return static_cast<uint16_t>(sign | 0x7C00u | (man ? 0x200u : 0u));
    if (exp > 15) return static_cast<uint16_t>(sign | 0x7C00u);
    if (exp >= -14) {
        uint32_t m = man >> 13, e = static_cast<uint32_t>(exp + 15) << 10;
        const uint32_t rem = man & 0x1FFFu, half = 0x1000u;
        if (rem > half || (rem == half && (m & 1u))) {
            ++m;
            if (m == 0x400u) { m = 0; e += 0x400u; if ((e >> 10) >= 31u) return static_cast<uint16_t>(sign | 0x7C00u); }
        }
        return static_cast<uint16_t>(sign | e | m);
    }
    if (exp < -25) return static_cast<uint16_t>(sign);
    const uint32_t sig = man | 0x800000u;
    const uint32_t sh = static_cast<uint32_t>(-exp - 1);
    uint32_t q = sig >> sh;
    const uint32_t rem = sig & ((1u << sh) - 1u), half = 1u << (sh - 1);
    if (rem > half || (rem == half && (q & 1u))) ++q;
    if (q >= 0x400u) return static_cast<uint16_t>(sign | 0x0400u);
    return static_cast<uint16_t>(sign | q);
}
}  // namespace

uint32_t row_bytes(uint32_t dim) {
    if (dim == 0 || dim % kBlock) return 0;
    return (dim / kBlock) * kBlockBytes;
}

bool pack_row(const float* x, uint32_t dim, uint8_t* out) {
    if (!x || !out || row_bytes(dim) == 0) return false;
    for (uint32_t b = 0; b < dim; b += kBlock) {
        const float* blk = x + b;
        float amax = 0.0f;
        for (uint32_t i = 0; i < kBlock; ++i) amax = std::fmax(amax, std::fabs(blk[i]));
        const float scale = amax > 0.0f ? amax / 7.0f : 0.0f;
        const uint16_t sh = float_to_half(scale);
        const float s = half_to_float(sh);
        uint8_t* dst = out + (b / kBlock) * kBlockBytes;
        std::memcpy(dst, &sh, 2);
        std::memset(dst + 2, 0, 16);
        for (uint32_t i = 0; i < kBlock; ++i) {
            int q = s > 0.0f ? static_cast<int>(std::lround(blk[i] / s)) : 0;
            q = q < -8 ? -8 : (q > 7 ? 7 : q);
            const uint8_t nib = static_cast<uint8_t>(q + 8);
            uint8_t& byte = dst[2 + (i & 15u)];
            if (i < 16) byte = static_cast<uint8_t>((byte & 0xF0u) | nib);
            else byte = static_cast<uint8_t>((byte & 0x0Fu) | (nib << 4));
        }
    }
    return true;
}

bool unpack_row(const uint8_t* in, uint32_t dim, float* out) {
    if (!in || !out || row_bytes(dim) == 0) return false;
    for (uint32_t b = 0; b < dim; b += kBlock) {
        const uint8_t* src = in + (b / kBlock) * kBlockBytes;
        uint16_t sh;
        std::memcpy(&sh, src, 2);
        const float s = half_to_float(sh);
        for (uint32_t i = 0; i < kBlock; ++i) {
            const uint8_t byte = src[2 + (i & 15u)];
            const int nib = i < 16 ? (byte & 0x0Fu) : (byte >> 4);
            out[b + i] = s * static_cast<float>(nib - 8);
        }
    }
    return true;
}

double max_abs_error_bound(const float* x, uint32_t dim) {
    if (!x || row_bytes(dim) == 0) return 0.0;
    double worst = 0.0;
    for (uint32_t b = 0; b < dim; b += kBlock) {
        float amax = 0.0f;
        for (uint32_t i = 0; i < kBlock; ++i) amax = std::fmax(amax, std::fabs(x[b + i]));
        const float s = half_to_float(float_to_half(amax > 0.0f ? amax / 7.0f : 0.0f));
        worst = std::fmax(worst, 0.5 * double(s));
    }
    return worst;
}

bool pack_table(const float* x, uint32_t rows, uint32_t dim, std::vector<uint8_t>* out) {
    const uint32_t rb = row_bytes(dim);
    if (!x || !out || rb == 0) return false;
    out->resize(size_t(rows) * rb);
    for (uint32_t r = 0; r < rows; ++r)
        if (!pack_row(x + size_t(r) * dim, dim, out->data() + size_t(r) * rb)) return false;
    return true;
}

bool unpack_table(const uint8_t* in, uint32_t rows, uint32_t dim, float* out) {
    const uint32_t rb = row_bytes(dim);
    if (!in || !out || rb == 0) return false;
    for (uint32_t r = 0; r < rows; ++r)
        if (!unpack_row(in + size_t(r) * rb, dim, out + size_t(r) * dim)) return false;
    return true;
}

}  // namespace vegajet::hostq4
