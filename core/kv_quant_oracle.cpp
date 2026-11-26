// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/kv_quant_oracle.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

namespace vegajet::kvq {

uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t exp = (x >> 23) & 0xFFu;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp == 0xFFu) return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
    const int e = static_cast<int>(exp) - 127 + 15;
    if (e >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    if (e <= 0) {  // half subnormal (or zero)
        if (e < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        const int shift = 14 - e;
        uint32_t half = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u), mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1u))) ++half;
        return static_cast<uint16_t>(sign | half);
    }
    uint32_t half = (static_cast<uint32_t>(e) << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) ++half;  // a carry into the exponent is correct rounding
    return static_cast<uint16_t>(sign | half);
}

float f16_to_f32(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu, mant = h & 0x3FFu, bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {  // subnormal: normalize
            int e = -1;
            do { mant <<= 1; ++e; } while (!(mant & 0x400u));
            bits = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | ((mant & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// ------------------------------------------------------------------------------------------------ q4_0 / v_dot8
void q4_pack(const float* x, uint32_t dim, uint32_t* codes, uint16_t* scales) {
    for (uint32_t b = 0; b < dim / kBlock; ++b) {
        const float* xb = x + b * kBlock;
        float amax = 0.0f, maxv = 0.0f;
        for (uint32_t j = 0; j < kBlock; ++j)
            if (std::fabs(xb[j]) > amax) { amax = std::fabs(xb[j]); maxv = xb[j]; }
        const float d = maxv / -8.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        scales[b] = f32_to_f16(d);
        for (uint32_t w = 0; w < 4; ++w) {
            uint32_t word = 0;
            for (uint32_t n = 0; n < 8; ++n) {
                const int u = std::min(15, static_cast<int>(xb[8 * w + n] * id + 8.5f));  // ggml Q4_0 rounding
                word |= (static_cast<uint32_t>(u - 8) & 0xFu) << (4 * n);
            }
            codes[4 * b + w] = word;
        }
    }
}

namespace {
inline int sext4(uint32_t v) { return v >= 8 ? static_cast<int>(v) - 16 : static_cast<int>(v); }
}

void q4_unpack(const uint32_t* codes, const uint16_t* scales, uint32_t dim, float* out) {
    for (uint32_t b = 0; b < dim / kBlock; ++b) {
        const float d = f16_to_f32(scales[b]);
        for (uint32_t j = 0; j < kBlock; ++j)
            out[b * kBlock + j] = d * static_cast<float>(sext4((codes[4 * b + j / 8] >> (4 * (j % 8))) & 0xFu));
    }
}

void ggml_q4_0_to_dot8(const uint8_t* blk, uint32_t codes[4], uint16_t* scale) {
    std::memcpy(scale, blk, 2);
    for (uint32_t w = 0; w < 4; ++w) codes[w] = 0;
    for (uint32_t j = 0; j < 16; ++j) {
        const uint32_t lo = (blk[2 + j] & 0xFu) - 8u, hi = (blk[2 + j] >> 4) - 8u;  // unsigned -> two's complement
        codes[j / 8] |= (lo & 0xFu) << (4 * (j % 8));
        codes[(j + 16) / 8] |= (hi & 0xFu) << (4 * ((j + 16) % 8));
    }
}

void dot8_to_ggml_q4_0(const uint32_t codes[4], uint16_t scale, uint8_t* blk) {
    std::memcpy(blk, &scale, 2);
    for (uint32_t j = 0; j < 16; ++j) {
        const uint32_t lo = (((codes[j / 8] >> (4 * (j % 8))) & 0xFu) + 8u) & 0xFu;
        const uint32_t hi = (((codes[(j + 16) / 8] >> (4 * ((j + 16) % 8))) & 0xFu) + 8u) & 0xFu;
        blk[2 + j] = static_cast<uint8_t>(lo | (hi << 4));
    }
}

void query_pack_i4(const float* q, uint32_t dim, uint32_t* codes, float* scales) {
    for (uint32_t b = 0; b < dim / kBlock; ++b) {
        float amax = 0.0f;
        for (uint32_t j = 0; j < kBlock; ++j) amax = std::max(amax, std::fabs(q[b * kBlock + j]));
        const float s = amax / 7.0f, is = s > 0.0f ? 1.0f / s : 0.0f;
        scales[b] = s;
        for (uint32_t w = 0; w < 4; ++w) {
            uint32_t word = 0;
            for (uint32_t n = 0; n < 8; ++n) {
                const int c = std::clamp(static_cast<int>(std::lround(q[b * kBlock + 8 * w + n] * is)), -7, 7);
                word |= (static_cast<uint32_t>(c) & 0xFu) << (4 * n);
            }
            codes[4 * b + w] = word;
        }
    }
}

void query_pack_i8(const float* q, uint32_t dim, int8_t* codes, float* scales) {
    for (uint32_t b = 0; b < dim / kBlock; ++b) {
        float amax = 0.0f;
        for (uint32_t j = 0; j < kBlock; ++j) amax = std::max(amax, std::fabs(q[b * kBlock + j]));
        const float s = amax / 127.0f, is = s > 0.0f ? 1.0f / s : 0.0f;
        scales[b] = s;
        for (uint32_t j = 0; j < kBlock; ++j)
            codes[b * kBlock + j] = static_cast<int8_t>(std::clamp(static_cast<int>(std::lround(q[b * kBlock + j] * is)), -127, 127));
    }
}

float q4_dot_i4(const uint32_t* qc, const float* qs, const uint32_t* kc, const uint16_t* ks, uint32_t dim) {
    float sum = 0.0f;
    for (uint32_t b = 0; b < dim / kBlock; ++b) {
        int32_t acc = 0;
        for (uint32_t w = 0; w < 4; ++w) acc = dot8_i4(qc[4 * b + w], kc[4 * b + w], acc);
        sum += static_cast<float>(acc) * qs[b] * f16_to_f32(ks[b]);
    }
    return sum;
}

float q4_dot_i8(const int8_t* qc, const float* qs, const uint32_t* kc, const uint16_t* ks, uint32_t dim) {
    float sum = 0.0f;
    for (uint32_t b = 0; b < dim / kBlock; ++b) {
        int32_t acc = 0;
        for (uint32_t j = 0; j < kBlock; ++j)
            acc += qc[b * kBlock + j] * sext4((kc[4 * b + j / 8] >> (4 * (j % 8))) & 0xFu);
        sum += static_cast<float>(acc) * qs[b] * f16_to_f32(ks[b]);
    }
    return sum;
}

// ------------------------------------------------------------------------------------------------ turbo3
// Lloyd-Max quantizer for N(0, 1), 8 levels (Max 1960); mean squared error 0.03454 per unit variance.
const float kTurbo3Codebook[8] = {-2.1519f, -1.3439f, -0.7560f, -0.2451f, 0.2451f, 0.7560f, 1.3439f, 2.1519f};

namespace {

uint64_t splitmix64(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

inline uint32_t nearest_code(float t) {
    uint32_t i = 0;
    while (i < 7 && t > 0.5f * (kTurbo3Codebook[i] + kTurbo3Codebook[i + 1])) ++i;
    return i;
}

inline uint32_t t3_index(const uint32_t* blk3, uint32_t j) {
    return ((blk3[j / 16] >> (2 * (j % 16))) & 3u) | (((blk3[2] >> j) & 1u) << 2);
}

} // namespace

std::vector<int8_t> turbo3_signs(uint32_t dim, uint64_t seed) {
    std::vector<int8_t> s(dim);
    uint64_t st = seed;
    uint64_t bits = 0;
    for (uint32_t i = 0; i < dim; ++i) {
        if (i % 64 == 0) bits = splitmix64(st);
        s[i] = (bits >> (i % 64)) & 1u ? int8_t(-1) : int8_t(1);
    }
    return s;
}

void fwht_normalized(float* x, uint32_t n) {
    for (uint32_t h = 1; h < n; h <<= 1)
        for (uint32_t i = 0; i < n; i += 2 * h)
            for (uint32_t j = i; j < i + h; ++j) {
                const float a = x[j], b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
    const float inv = 1.0f / std::sqrt(static_cast<float>(n));
    for (uint32_t i = 0; i < n; ++i) x[i] *= inv;
}

void turbo3_rotate(const float* x, uint32_t dim, const int8_t* signs, float* out) {
    for (uint32_t i = 0; i < dim; ++i) out[i] = signs[i] * x[i];
    fwht_normalized(out, dim);
}

void turbo3_unrotate(const float* y, uint32_t dim, const int8_t* signs, float* out) {
    std::vector<float> t(y, y + dim);
    fwht_normalized(t.data(), dim);
    for (uint32_t i = 0; i < dim; ++i) out[i] = signs[i] * t[i];
}

void turbo3_pack_rotated(const float* y, uint32_t dim, uint32_t* codes, uint16_t* scales) {
    for (uint32_t b = 0; b < dim / kBlock; ++b) {
        const float* yb = y + b * kBlock;
        double ss = 0.0, amax = 0.0;
        for (uint32_t j = 0; j < kBlock; ++j) {
            ss += double(yb[j]) * yb[j];
            amax = std::max(amax, double(std::fabs(yb[j])));
        }
        // Two starting scales (block RMS: right for Gaussian blocks; absmax / outer level: right for blocks that are
        // far from Gaussian), each refined by alternating nearest-code and least-squares scale; keep the one with
        // the lower squared error after fp16 rounding of the scale.
        uint16_t best_h = 0;
        double best_err = -1.0;
        const float starts[2] = {static_cast<float>(std::sqrt(ss / kBlock)), static_cast<float>(amax / kTurbo3Codebook[7])};
        for (float s : starts) {
            for (int iter = 0; iter < 3 && s > 0.0f; ++iter) {
                double num = 0.0, den = 0.0;
                for (uint32_t j = 0; j < kBlock; ++j) {
                    const float c = kTurbo3Codebook[nearest_code(yb[j] / s)];
                    num += double(yb[j]) * c;
                    den += double(c) * c;
                }
                if (den > 0.0 && num > 0.0) s = static_cast<float>(num / den);
            }
            const uint16_t h = f32_to_f16(s);
            const float sh = f16_to_f32(h);
            double e = 0.0;
            for (uint32_t j = 0; j < kBlock; ++j) {
                const double r = sh > 0.0f ? yb[j] - sh * kTurbo3Codebook[nearest_code(yb[j] / sh)] : yb[j];
                e += r * r;
            }
            if (best_err < 0.0 || e < best_err) { best_err = e; best_h = h; }
        }
        scales[b] = best_h;
        const float sh = f16_to_f32(best_h);
        uint32_t* blk = codes + 3 * b;
        blk[0] = blk[1] = blk[2] = 0;
        for (uint32_t j = 0; j < kBlock; ++j) {
            const uint32_t i = sh > 0.0f ? nearest_code(yb[j] / sh) : 4u;  // final codes match the stored scale
            blk[j / 16] |= (i & 3u) << (2 * (j % 16));
            blk[2] |= (i >> 2) << j;
        }
    }
}

void turbo3_pack(const float* x, uint32_t dim, const int8_t* signs, uint32_t* codes, uint16_t* scales) {
    std::vector<float> y(dim);
    turbo3_rotate(x, dim, signs, y.data());
    turbo3_pack_rotated(y.data(), dim, codes, scales);
}

void turbo3_unpack_rotated(const uint32_t* codes, const uint16_t* scales, uint32_t dim, float* y) {
    for (uint32_t b = 0; b < dim / kBlock; ++b) {
        const float s = f16_to_f32(scales[b]);
        for (uint32_t j = 0; j < kBlock; ++j) y[b * kBlock + j] = s * kTurbo3Codebook[t3_index(codes + 3 * b, j)];
    }
}

void turbo3_unpack(const uint32_t* codes, const uint16_t* scales, uint32_t dim, const int8_t* signs, float* out) {
    std::vector<float> y(dim);
    turbo3_unpack_rotated(codes, scales, dim, y.data());
    turbo3_unrotate(y.data(), dim, signs, out);
}

float turbo3_dot_rotated(const float* q_rot, const uint32_t* codes, const uint16_t* scales, uint32_t dim) {
    float sum = 0.0f;
    for (uint32_t b = 0; b < dim / kBlock; ++b) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < kBlock; ++j) acc += q_rot[b * kBlock + j] * kTurbo3Codebook[t3_index(codes + 3 * b, j)];
        sum += acc * f16_to_f32(scales[b]);
    }
    return sum;
}

// CPU byte-array helpers use memcpy, so callers may pass unaligned storage.
void pack_device_vector(const float* x, uint32_t d, DeviceFormat f, uint8_t* out) {
    std::vector<uint16_t> scales(d/32);
    std::vector<uint32_t> codes(d/8);
    if (f == DeviceFormat::Fp16) {
        for (uint32_t i=0;i<d;++i) { auto h=f32_to_f16(x[i]); std::memcpy(out+2*i,&h,2); }
    } else if (f == DeviceFormat::Q8) {
        std::vector<int8_t> c(d); std::vector<float> sc(d/32);
        query_pack_i8(x,d,c.data(),sc.data());
        for (uint32_t b=0;b<d/32;++b) scales[b]=f32_to_f16(sc[b]);
        std::memcpy(out,scales.data(),d/16); std::memcpy(out+d/16,c.data(),d);
    } else {
        if (f == DeviceFormat::Q4) q4_pack(x,d,codes.data(),scales.data());
        else { const auto signs=turbo3_signs(d,0x5455524230u); turbo3_pack(x,d,signs.data(),codes.data(),scales.data()); }
        std::memcpy(out,scales.data(),d/16);
        std::memcpy(out+d/16,codes.data(),vector_bytes(f,d)-d/16);
    }
}
void unpack_device_vector(const uint8_t* x,uint32_t d,DeviceFormat f,float* out) {
    if (f == DeviceFormat::Fp16) {
        for(uint32_t i=0;i<d;++i) {uint16_t h;std::memcpy(&h,x+2*i,2);out[i]=f16_to_f32(h);}
        return;
    }
    std::vector<uint16_t> scales(d/32); std::memcpy(scales.data(),x,d/16);
    if (f == DeviceFormat::Q8) {
        for(uint32_t i=0;i<d;++i) out[i]=f16_to_f32(scales[i/32])*int8_t(x[d/16+i]);
    } else {
        std::vector<uint32_t> c(d/8); std::memcpy(c.data(),x+d/16,vector_bytes(f,d)-d/16);
        if(f==DeviceFormat::Q4) q4_unpack(c.data(),scales.data(),d,out);
        else { const auto signs=turbo3_signs(d,0x5455524230u);turbo3_unpack(c.data(),scales.data(),d,signs.data(),out); }
    }
}
std::vector<float> device_attention_reference(const float* q,const uint8_t* k,const uint8_t* v,
    uint32_t d,uint32_t tokens,DeviceFormat f) {
    std::vector<float> out(d,0),kd(d),vd(d),qd(q,q+d),qs(d/32);
    if(!tokens) return out;
    if(f==DeviceFormat::Q4) {
        std::vector<uint32_t> qc(d/8); query_pack_i4(q,d,qc.data(),qs.data());
        for(uint32_t i=0;i<d;++i) {unsigned n=(qc[i/8]>>(4*(i%8)))&15;qd[i]=float(n>=8?int(n)-16:int(n))*qs[i/32];}
    }
    std::vector<double> scores(tokens);double peak=-INFINITY,z=0;
    for(uint32_t t=0;t<tokens;++t) {
        unpack_device_vector(k+size_t(t)*vector_bytes(f,d),d,f,kd.data());
        double score=0;for(uint32_t i=0;i<d;++i) score+=double(qd[i])*kd[i];
        scores[t]=score/std::sqrt(double(d));peak=std::max(peak,scores[t]);
    }
    for(auto& s:scores) {s=std::exp(s-peak);z+=s;}
    std::vector<double> accum(d,0);
    for(uint32_t t=0;t<tokens;++t) {
        unpack_device_vector(v+size_t(t)*vector_bytes(f,d),d,f,vd.data());
        for(uint32_t i=0;i<d;++i) accum[i]+=scores[t]/z*vd[i];
    }
    for(uint32_t i=0;i<d;++i) out[i]=float(accum[i]);
    return out;
}

// ------------------------------------------------------------------------------------------------ oracles
const char* format_name(KvFormat f) {
    switch (f) {
        case KvFormat::F32: return "f32";
        case KvFormat::Q4Dot8: return "q4_0/dot8";
        case KvFormat::Turbo3: return "turbo3";
    }
    return "?";
}

const char* query_path_name(QueryPath p) {
    switch (p) {
        case QueryPath::F32: return "f32";
        case QueryPath::I8: return "i8";
        case QueryPath::I4: return "i4";
    }
    return "?";
}

double bits_per_value(KvFormat f) {
    switch (f) {
        case KvFormat::F32: return 32.0;
        case KvFormat::Q4Dot8: return 8.0 * kQ4BytesPerBlock / kBlock;
        case KvFormat::Turbo3: return 8.0 * kTurbo3BytesPerBlock / kBlock;
    }
    return 0.0;
}

namespace {

// Round-trips one vector through a format (dequantized values in the original, unrotated domain).
void roundtrip(KvFormat f, const float* x, uint32_t d, const int8_t* signs, float* out) {
    if (f == KvFormat::F32) {
        std::copy(x, x + d, out);
    } else if (f == KvFormat::Q4Dot8) {
        std::vector<uint32_t> c(d / 8);
        std::vector<uint16_t> s(d / kBlock);
        q4_pack(x, d, c.data(), s.data());
        q4_unpack(c.data(), s.data(), d, out);
    } else {
        std::vector<uint32_t> c(3 * d / kBlock);
        std::vector<uint16_t> s(d / kBlock);
        turbo3_pack(x, d, signs, c.data(), s.data());
        turbo3_unpack(c.data(), s.data(), d, signs, out);
    }
}

void log_softmax(const std::vector<double>& s, std::vector<double>& out) {
    const double m = *std::max_element(s.begin(), s.end());
    double z = 0.0;
    for (double v : s) z += std::exp(v - m);
    const double lz = m + std::log(z);
    out.resize(s.size());
    for (size_t i = 0; i < s.size(); ++i) out[i] = s[i] - lz;
}

} // namespace

OracleStats run_oracle(const OracleConfig& cfg, KvFormat k_fmt, KvFormat v_fmt, QueryPath qp, std::string* err) {
    OracleStats st;
    const uint32_t D = cfg.head_dim, T = cfg.n_keys;
    auto fail = [&](const char* m) { if (err) *err = m; return st; };
    if (D == 0 || D % kBlock || T == 0 || cfg.n_queries == 0) return fail("head_dim must be a multiple of 32, keys/queries > 0");
    if ((k_fmt == KvFormat::Turbo3 || v_fmt == KvFormat::Turbo3) && (D & (D - 1))) return fail("turbo3 needs a power-of-two head_dim");
    if (qp != QueryPath::F32 && k_fmt != KvFormat::Q4Dot8) return fail("integer query paths need q4_0/dot8 keys");
    if (cfg.outlier_channels > D) return fail("more outlier channels than head_dim");

    std::mt19937_64 rng(cfg.seed);
    std::normal_distribution<float> N(0.0f, 1.0f);
    std::vector<uint32_t> perm(D);
    for (uint32_t i = 0; i < D; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<float> chan(D, 1.0f);
    for (uint32_t i = 0; i < cfg.outlier_channels; ++i) chan[perm[i]] = cfg.outlier_scale;

    std::vector<float> K(size_t(T) * D), V(size_t(T) * D), Kq(K.size()), Vq(V.size());
    for (size_t i = 0; i < K.size(); ++i) K[i] = N(rng) * chan[i % D];
    for (auto& v : V) v = N(rng);
    const std::vector<int8_t> signs = turbo3_signs(D, cfg.turbo_seed);
    double kerr = 0.0, knorm = 0.0;
    for (uint32_t t = 0; t < T; ++t) {
        roundtrip(k_fmt, &K[size_t(t) * D], D, signs.data(), &Kq[size_t(t) * D]);
        roundtrip(v_fmt, &V[size_t(t) * D], D, signs.data(), &Vq[size_t(t) * D]);
        for (uint32_t i = 0; i < D; ++i) {
            const double e = double(Kq[size_t(t) * D + i]) - K[size_t(t) * D + i];
            kerr += e * e;
            knorm += double(K[size_t(t) * D + i]) * K[size_t(t) * D + i];
        }
    }
    st.k_rel_mse = kerr / knorm;

    // Packed keys for the integer query paths.
    std::vector<uint32_t> kc(size_t(T) * D / 8);
    std::vector<uint16_t> ks(size_t(T) * D / kBlock);
    if (k_fmt == KvFormat::Q4Dot8)
        for (uint32_t t = 0; t < T; ++t) q4_pack(&K[size_t(t) * D], D, &kc[size_t(t) * D / 8], &ks[size_t(t) * D / kBlock]);

    const double inv_sqrt_d = 1.0 / std::sqrt(double(D));
    double se = 0.0, sref2 = 0.0, kld_sum = 0.0, out_err = 0.0;
    uint32_t agree = 0;
    std::uniform_int_distribution<uint32_t> pick(0, T - 1);
    std::vector<float> q(D), qs(D / kBlock);
    std::vector<int8_t> q8(D);
    std::vector<uint32_t> q4(D / 8);
    std::vector<double> sref(T), sq(T), lref, lq;
    for (uint32_t qi = 0; qi < cfg.n_queries; ++qi) {
        // Query = noise + lambda * (one key), lambda sized so that key gets a boost of `focus` score std devs.
        for (auto& v : q) v = N(rng);
        const uint32_t tgt = pick(rng);
        double m = 0.0, m2 = 0.0, kk = 0.0;
        for (uint32_t t = 0; t < T; ++t) {
            double s = 0.0;
            for (uint32_t i = 0; i < D; ++i) s += double(q[i]) * K[size_t(t) * D + i];
            s *= inv_sqrt_d;
            m += s;
            m2 += s * s;
        }
        for (uint32_t i = 0; i < D; ++i) kk += double(K[size_t(tgt) * D + i]) * K[size_t(tgt) * D + i];
        const double sd = std::sqrt(std::max(1e-12, m2 / T - (m / T) * (m / T)));
        const double lambda = cfg.focus * sd / (kk * inv_sqrt_d);
        for (uint32_t i = 0; i < D; ++i) q[i] += static_cast<float>(lambda * K[size_t(tgt) * D + i]);

        if (qp == QueryPath::I8) query_pack_i8(q.data(), D, q8.data(), qs.data());
        if (qp == QueryPath::I4) query_pack_i4(q.data(), D, q4.data(), qs.data());
        for (uint32_t t = 0; t < T; ++t) {
            double r = 0.0, s = 0.0;
            for (uint32_t i = 0; i < D; ++i) r += double(q[i]) * K[size_t(t) * D + i];
            if (qp == QueryPath::F32) {
                for (uint32_t i = 0; i < D; ++i) s += double(q[i]) * Kq[size_t(t) * D + i];
            } else if (qp == QueryPath::I8) {
                s = q4_dot_i8(q8.data(), qs.data(), &kc[size_t(t) * D / 8], &ks[size_t(t) * D / kBlock], D);
            } else {
                s = q4_dot_i4(q4.data(), qs.data(), &kc[size_t(t) * D / 8], &ks[size_t(t) * D / kBlock], D);
            }
            sref[t] = r * inv_sqrt_d;
            sq[t] = s * inv_sqrt_d;
            const double e = sq[t] - sref[t];
            se += e * e;
            sref2 += sref[t] * sref[t];
            st.score_max_abs = std::max(st.score_max_abs, std::fabs(e));
        }
        log_softmax(sref, lref);
        log_softmax(sq, lq);
        double kl = 0.0;
        for (uint32_t t = 0; t < T; ++t) kl += std::exp(lref[t]) * (lref[t] - lq[t]);
        kld_sum += kl;
        st.kld_max = std::max(st.kld_max, kl);
        agree += std::max_element(sref.begin(), sref.end()) - sref.begin() == std::max_element(sq.begin(), sq.end()) - sq.begin();
        double on = 0.0, oe = 0.0;
        for (uint32_t i = 0; i < D; ++i) {
            double o_ref = 0.0, o = 0.0;
            for (uint32_t t = 0; t < T; ++t) {
                o_ref += std::exp(lref[t]) * V[size_t(t) * D + i];
                o += std::exp(lq[t]) * Vq[size_t(t) * D + i];
            }
            on += o_ref * o_ref;
            oe += (o - o_ref) * (o - o_ref);
            st.out_max_abs = std::max(st.out_max_abs, std::fabs(o-o_ref));
        }
        out_err += std::sqrt(oe / std::max(on, 1e-30));
    }
    st.score_rel_rms = std::sqrt(se / sref2);
    st.kld_mean = kld_sum / cfg.n_queries;
    st.top1_agree = double(agree) / cfg.n_queries;
    st.out_rel_err = out_err / cfg.n_queries;
    if (err) err->clear();
    return st;
}

} // namespace vegajet::kvq
