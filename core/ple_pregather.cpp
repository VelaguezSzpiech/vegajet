// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/ple_pregather.hpp"

#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vegajet::ple {

float half_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {  // subnormal: normalize
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

size_t row_bytes(RowFmt f, uint32_t dim) {
    switch (f) {
        case RowFmt::F32: return size_t(dim) * 4;
        case RowFmt::F16: return size_t(dim) * 2;
        case RowFmt::Q8_0: return dim % 32 ? 0 : size_t(dim / 32) * 34;
    }
    return 0;
}

int max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

namespace {

bool validate(const PleTables& t, const int32_t* tokens, uint32_t m, std::string* err) {
    auto fail = [&](const std::string& s) { if (err) *err = s; return false; };
    if (!t.n_layers || !t.dim || !t.vocab) return fail("empty table description");
    if (t.layer.size() != t.n_layers) return fail("layer pointer count != n_layers");
    if (row_bytes(t.fmt, t.dim) == 0) return fail("dim not valid for the row format");
    for (const uint8_t* p : t.layer) if (!p) return fail("null layer table");
    if (m && !tokens) return fail("null token array");
    for (uint32_t i = 0; i < m; ++i)
        if (tokens[i] < 0 || static_cast<uint32_t>(tokens[i]) >= t.vocab) return fail("token " + std::to_string(i) + " out of range");
    return true;
}

inline void row_to_f32(RowFmt f, const uint8_t* src, uint32_t dim, float* dst) {
    switch (f) {
        case RowFmt::F32: std::memcpy(dst, src, size_t(dim) * 4); break;
        case RowFmt::F16:
            for (uint32_t d = 0; d < dim; ++d) { uint16_t h; std::memcpy(&h, src + 2 * d, 2); dst[d] = half_to_float(h); }
            break;
        case RowFmt::Q8_0:
            for (uint32_t b = 0; b < dim / 32; ++b) {
                uint16_t h;
                std::memcpy(&h, src + b * 34, 2);
                const float s = half_to_float(h);
                const int8_t* q = reinterpret_cast<const int8_t*>(src + b * 34 + 2);
                for (uint32_t i = 0; i < 32; ++i) dst[b * 32 + i] = s * static_cast<float>(q[i]);
            }
            break;
    }
}

inline size_t out_row(Layout lay, uint32_t l, uint32_t mi, uint32_t L, uint32_t M) {
    return lay == Layout::LayerMajor ? size_t(l) * M + mi : size_t(mi) * L + l;
}

} // namespace

bool pregather(const PleTables& t, const int32_t* tokens, uint32_t m, float* out, Layout lay, std::string* err) {
    if (!validate(t, tokens, m, err)) return false;
    if (m && !out) { if (err) *err = "null output"; return false; }
    const size_t rb = row_bytes(t.fmt, t.dim);
    const int64_t total = int64_t(t.n_layers) * m;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < total; ++i) {
        const uint32_t l = static_cast<uint32_t>(i / m), mi = static_cast<uint32_t>(i % m);
        row_to_f32(t.fmt, t.layer[l] + size_t(tokens[mi]) * rb, t.dim, out + out_row(lay, l, mi, t.n_layers, m) * t.dim);
    }
    return true;
}

bool pregather_reference(const PleTables& t, const int32_t* tokens, uint32_t m, float* out, Layout lay) {
    if (!validate(t, tokens, m, nullptr)) return false;
    const size_t rb = row_bytes(t.fmt, t.dim);
    for (uint32_t l = 0; l < t.n_layers; ++l)
        for (uint32_t mi = 0; mi < m; ++mi)
            for (uint32_t d = 0; d < t.dim; ++d) {
                const uint8_t* row = t.layer[l] + size_t(tokens[mi]) * rb;
                float v = 0;
                if (t.fmt == RowFmt::F32) std::memcpy(&v, row + 4 * d, 4);
                else if (t.fmt == RowFmt::F16) { uint16_t h; std::memcpy(&h, row + 2 * d, 2); v = half_to_float(h); }
                else {
                    uint16_t h;
                    std::memcpy(&h, row + (d / 32) * 34, 2);
                    v = half_to_float(h) * static_cast<float>(static_cast<int8_t>(row[(d / 32) * 34 + 2 + d % 32]));
                }
                out[out_row(lay, l, mi, t.n_layers, m) * t.dim + d] = v;
            }
    return true;
}

} // namespace vegajet::ple
