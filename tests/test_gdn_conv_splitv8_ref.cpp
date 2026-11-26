// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// A2 reference: CPU emulation of the selectable Split-V=8 GDN ConvRec
// (ops/kernels/vj_ornith_gdn_sv8.hip, two ordered dispatches) against
//   1. the current Split-V=4 fused kernel (ornith_gdn_conv_recur_splitv_quant_m1_kernel),
//      with which it must agree bit for bit (state, conv ring, fo, q8, d/bsum);
//   2. an independent deterministic FP32 oracle that uses a plain per-column order
//      (single ascending 128-row dot, left-to-right RMS / block sums) instead of the
//      kernel's 8x16 row groups and 64-lane trees, checked with a relative tolerance.
//
// The emulation follows the kernels' buffer layouts, thread/WG mapping and op order. It is a
// CPU model of the arithmetic, not a device execution or device-parity claim (the device
// q/k norm and gated norm use rsqrtf/__expf; this model uses 1/sqrtf/expf consistently in
// both variants).
//
// Two-dispatch structure: stage1() models ornith_gdn_conv_recur_splitv8_recur_m1_kernel and
// returns exactly what that kernel publishes (conv ring, S, fo[0..127], pair-squares[0..63]).
// stage2() models ornith_gdn_conv_recur_splitv8_quant_m1_kernel and may read ONLY that
// published scratch, the z gate and ssm_norm_w. The main() checks perturb a single published
// float to prove stage 2 really consumes the published values instead of recomputing them.
#include "tests/test_common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {

constexpr uint32_t NK = 8, NV = 32, HD = 128;
constexpr uint32_t CONV = (2 * NK + NV) * HD;                  // 6144
constexpr uint32_t STRIDE = 10304;
constexpr uint32_t Z_OFF = 6144, ALPHA_OFF = 10240, BETA_OFF = 10272;
constexpr uint32_t SLOTS = 17;
constexpr float EPS = 1e-6f;

float wave64_sum(const float* v) {
    float a[64];
    std::memcpy(a, v, sizeof(a));
    for (uint32_t mask = 1u; mask < 64u; mask <<= 1) {
        float n[64];
        for (uint32_t l = 0; l < 64u; ++l) n[l] = a[l] + a[l ^ mask];
        std::memcpy(a, n, sizeof(a));
    }
    return a[0];
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }

struct Inputs {
    std::vector<float> qkv;        // STRIDE
    std::vector<float> conv_w;     // CONV*4
    std::vector<float> conv_ring;  // SLOTS*3*CONV, slot 0 = in
    std::vector<float> dt_bias, ssm_a, ssm_norm_w;
    std::vector<float> S0;         // NV*HD*HD (slot 0)
};

struct Outputs {
    std::vector<float> conv_out;   // 3*CONV (slot 1)
    std::vector<float> S;          // NV*HD*HD
    std::vector<float> fo;         // NV*HD
    std::vector<int8_t> q8;        // NV*HD
    std::vector<float> ds;         // NV*4*2 (d, bsum)
};

// Everything stage 1 publishes for stage 2 (plus the state/conv outputs it also writes).
struct Scratch {
    std::vector<float> conv_out;   // 3*CONV
    std::vector<float> S;          // NV*HD*HD
    std::vector<float> fo;         // NV*HD   = sq_scratch[0..127] per head
    std::vector<float> sq;         // NV*64   = sq_scratch[128..191] per head
};

struct HeadGate { float g, beta; };

HeadGate head_gate(const Inputs& in, uint32_t hv) {
    const float a = in.qkv[ALPHA_OFF + hv] + in.dt_bias[hv];
    const float sp = (a > 20.0f) ? a : std::log1p(std::exp(a));
    HeadGate r;
    r.g = std::exp(sp * in.ssm_a[hv]);
    r.beta = 1.0f / (1.0f + std::exp(-in.qkv[BETA_OFF + hv]));
    return r;
}

// Conv1D + SiLU + L2 norm for one Q or K head, exactly the kernel's wave64-tree order:
// threads 0..127 (Q) or 128..255 (K), wave64 sums per 64 lanes, then sm_red[0]+sm_red[1].
void conv_qk(const Inputs& in, uint32_t hv, uint32_t is_k, float out[HD]) {
    const uint32_t hk = hv % NK;
    const uint32_t ch0 = (is_k ? (NK + hk) : hk) * 128u;
    float y[HD], lane_sq[128];
    for (uint32_t l = 0; l < HD; ++l) {
        const uint32_t ch = ch0 + l;
        const float* w = &in.conv_w[static_cast<size_t>(ch) * 4u];
        const float s0 = in.conv_ring[0ull * CONV + ch];
        const float s1 = in.conv_ring[1ull * CONV + ch];
        const float s2 = in.conv_ring[2ull * CONV + ch];
        const float x  = in.qkv[ch];
        const float acc = std::fma(w[0], s0, std::fma(w[1], s1, std::fma(w[2], s2, w[3] * x)));
        y[l] = silu(acc);
        lane_sq[l] = y[l] * y[l];
    }
    float sm_red[2];
    for (uint32_t w = 0; w < 2u; ++w) sm_red[w] = wave64_sum(&lane_sq[w * 64u]);
    float inv_n = 1.0f / std::sqrt(sm_red[0] + sm_red[1] + EPS);
    if (!is_k) inv_n *= 0.08838834764831845f;
    for (uint32_t l = 0; l < HD; ++l) out[l] = y[l] * inv_n;
}

// Stage 1: Conv1D + L2 norm + gate + Split-V recurrence + publication of fo/pair-squares.
Scratch stage1(const Inputs& in, uint32_t parts) {
    const uint32_t cols = HD / parts;   // 32 (Split-V=4) or 16 (Split-V=8)
    Scratch o;
    o.conv_out.assign(3ull * CONV, 0.0f);
    o.S = in.S0;
    o.fo.assign(static_cast<size_t>(NV) * HD, 0.0f);
    o.sq.assign(static_cast<size_t>(NV) * 64u, 0.0f);

    // Phase 0: Q/K normalization (recomputed per head, written to the ring only by hv<nk/part0)
    // and the disjoint V conv outputs.
    std::vector<float> qn(static_cast<size_t>(NV) * HD), kn(static_cast<size_t>(NV) * HD);
    std::vector<float> vconv(static_cast<size_t>(NV) * HD);
    for (uint32_t hv = 0; hv < NV; ++hv) {
        const uint32_t hk = hv % NK;
        conv_qk(in, hv, 0u, &qn[static_cast<size_t>(hv) * HD]);
        conv_qk(in, hv, 1u, &kn[static_cast<size_t>(hv) * HD]);
        if (hv < NK) {
            for (uint32_t is_k = 0; is_k < 2u; ++is_k) {
                const uint32_t ch0 = (is_k ? (NK + hk) : hk) * 128u;
                for (uint32_t l = 0; l < HD; ++l) {
                    const uint32_t ch = ch0 + l;
                    o.conv_out[0ull * CONV + ch] = in.conv_ring[1ull * CONV + ch];
                    o.conv_out[1ull * CONV + ch] = in.conv_ring[2ull * CONV + ch];
                    o.conv_out[2ull * CONV + ch] = in.qkv[ch];
                }
            }
        }
        for (uint32_t part = 0; part < parts; ++part) {
            for (uint32_t t = 0; t < cols; ++t) {
                const uint32_t ch_v = (2u * NK + hv) * 128u + part * cols + t;
                const float* w = &in.conv_w[static_cast<size_t>(ch_v) * 4u];
                const float s0 = in.conv_ring[0ull * CONV + ch_v];
                const float s1 = in.conv_ring[1ull * CONV + ch_v];
                const float s2 = in.conv_ring[2ull * CONV + ch_v];
                const float x  = in.qkv[ch_v];
                const float acc = std::fma(w[0], s0, std::fma(w[1], s1, std::fma(w[2], s2, w[3] * x)));
                vconv[static_cast<size_t>(hv) * HD + part * cols + t] = silu(acc);
                o.conv_out[0ull * CONV + ch_v] = s1;
                o.conv_out[1ull * CONV + ch_v] = s2;
                o.conv_out[2ull * CONV + ch_v] = x;
            }
        }
    }

    // Phase 1: per-column recurrence, 8 row groups x 16 rows ascending (both variants).
    for (uint32_t hv = 0; hv < NV; ++hv) {
        const HeadGate hg = head_gate(in, hv);
        float* S = &o.S[static_cast<size_t>(hv) * HD * HD];
        const float* q = &qn[static_cast<size_t>(hv) * HD];
        const float* k = &kn[static_cast<size_t>(hv) * HD];
        for (uint32_t col = 0; col < HD; ++col) {
            float pg[8];
            for (uint32_t rg = 0; rg < 8u; ++rg) {
                float p = 0.0f;
                for (uint32_t r = 0; r < 16u; ++r) {
                    const uint32_t i = rg * 16u + r;
                    p = std::fma(S[static_cast<size_t>(i) * HD + col], k[i], p);
                }
                pg[rg] = p;
            }
            const float pred = (((((((pg[0] + pg[1]) + pg[2]) + pg[3]) + pg[4]) + pg[5]) + pg[6]) + pg[7]);
            const float dv = hg.beta * (vconv[static_cast<size_t>(hv) * HD + col] - hg.g * pred);
            float og[8];
            for (uint32_t rg = 0; rg < 8u; ++rg) {
                float oo = 0.0f;
                for (uint32_t r = 0; r < 16u; ++r) {
                    const uint32_t i = rg * 16u + r;
                    float& s = S[static_cast<size_t>(i) * HD + col];
                    s = std::fma(k[i], dv, hg.g * s);
                    oo = std::fma(s, q[i], oo);
                }
                og[rg] = oo;
            }
            const float fo = (((((((og[0] + og[1]) + og[2]) + og[3]) + og[4]) + og[5]) + og[6]) + og[7]);
            o.fo[static_cast<size_t>(hv) * HD + col] = fo;
        }
        // 64 global column pairs -> sq[lane], identical lane assignment for both variants.
        for (uint32_t g = 0; g < 64u; ++g) {
            const float a = o.fo[static_cast<size_t>(hv) * HD + 2u * g];
            const float b = o.fo[static_cast<size_t>(hv) * HD + 2u * g + 1u];
            o.sq[static_cast<size_t>(hv) * 64u + g] = a * a + b * b;
        }
    }
    return o;
}

// Stage 2: gated RMSNorm + block-32 quant, derived only from the published scratch.
Outputs stage2(const Inputs& in, const Scratch& sc, uint32_t parts, bool fault_scale = false) {
    const uint32_t cols = HD / parts;
    Outputs o;
    o.conv_out = sc.conv_out;
    o.S = sc.S;
    o.fo = sc.fo;
    o.q8.assign(static_cast<size_t>(NV) * HD, 0);
    o.ds.assign(static_cast<size_t>(NV) * 4u * 2u, 0.0f);

    // Every part computes its 32-column block from the published fo; the part owning a column
    // writes q8, the even part writes d/bsum (disjoint bytes, identical values).
    for (uint32_t hv = 0; hv < NV; ++hv) {
        float lane_sq[64];
        std::memcpy(lane_sq, &sc.sq[static_cast<size_t>(hv) * 64u], sizeof(lane_sq));
        const float total_sq = wave64_sum(lane_sq);
        const float inv_r = 1.0f / std::sqrt(total_sq * (1.0f / 128.0f) + EPS);
        const float* zp = &in.qkv[Z_OFF + hv * 128u];
        for (uint32_t part = 0; part < parts; ++part) {
            const uint32_t blk = part * cols / 32u;
            const uint32_t col_b = blk * 32u;
            float y[32];
            for (uint32_t t = 0; t < 16u; ++t) {
                const uint32_t j0 = col_b + t * 2u;
                const float f0 = sc.fo[static_cast<size_t>(hv) * HD + j0];
                const float f1 = sc.fo[static_cast<size_t>(hv) * HD + j0 + 1u];
                const float z0 = zp[j0], z1 = zp[j0 + 1u];
                y[2u * t]      = (f0 * inv_r * in.ssm_norm_w[j0])      * silu(z0);
                y[2u * t + 1u] = (f1 * inv_r * in.ssm_norm_w[j0 + 1u]) * silu(z1);
            }
            // 16-lane XOR trees: lane t holds (y[2t], y[2t+1]).
            float av[16], bv[16];
            for (uint32_t t = 0; t < 16u; ++t) {
                av[t] = std::fmax(std::fabs(y[2u * t]), std::fabs(y[2u * t + 1u]));
                bv[t] = y[2u * t] + y[2u * t + 1u];
            }
            for (uint32_t mask : {8u, 4u, 2u, 1u}) {
                float an[16], bn[16];
                for (uint32_t t = 0; t < 16u; ++t) {
                    an[t] = std::fmax(av[t], av[t ^ mask]);
                    bn[t] = bv[t] + bv[t ^ mask];
                }
                std::memcpy(av, an, sizeof(av));
                std::memcpy(bv, bn, sizeof(bv));
            }
            const float amax = av[0];
            const float bsum = bv[0];
            float d = amax * (1.0f / 127.0f);
            float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            if (fault_scale) { d *= 0.5f; inv_d *= 2.0f; }   // negative control: wrong quant scale
            const uint32_t own0 = part * cols;
            for (uint32_t t = 0; t < 16u; ++t) {
                const uint32_t j0 = col_b + t * 2u;
                if (j0 >= own0 && j0 < own0 + cols) {
                    o.q8[static_cast<size_t>(hv) * HD + j0] =
                        static_cast<int8_t>(std::max(-127, std::min(127, static_cast<int>(std::nearbyint(y[2u * t] * inv_d)))));
                    o.q8[static_cast<size_t>(hv) * HD + j0 + 1u] =
                        static_cast<int8_t>(std::max(-127, std::min(127, static_cast<int>(std::nearbyint(y[2u * t + 1u] * inv_d)))));
                }
            }
            if ((part % (parts / 4u)) == 0u) {
                o.ds[static_cast<size_t>(hv) * 8u + blk * 2u]     = d;
                o.ds[static_cast<size_t>(hv) * 8u + blk * 2u + 1u] = bsum;
            }
        }
    }
    return o;
}

Outputs run_variant(const Inputs& in, uint32_t parts, bool fault_scale = false) {
    return stage2(in, stage1(in, parts), parts, fault_scale);
}

Outputs run_oracle(const Inputs& in) {
    Outputs o;
    o.conv_out.assign(3ull * CONV, 0.0f);
    o.S = in.S0;
    o.fo.assign(static_cast<size_t>(NV) * HD, 0.0f);
    o.q8.assign(static_cast<size_t>(NV) * HD, 0);
    o.ds.assign(static_cast<size_t>(NV) * 4u * 2u, 0.0f);

    std::vector<float> qn(static_cast<size_t>(NV) * HD), kn(static_cast<size_t>(NV) * HD);
    std::vector<float> vconv(static_cast<size_t>(NV) * HD);
    for (uint32_t hv = 0; hv < NV; ++hv) {
        const uint32_t hk = hv % NK;
        for (uint32_t is_k = 0; is_k < 2u; ++is_k) {
            const uint32_t ch0 = (is_k ? (NK + hk) : hk) * 128u;
            float* dst = is_k ? &kn[static_cast<size_t>(hv) * HD] : &qn[static_cast<size_t>(hv) * HD];
            float ss = 0.0f;
            for (uint32_t l = 0; l < HD; ++l) {
                const uint32_t ch = ch0 + l;
                const float* w = &in.conv_w[static_cast<size_t>(ch) * 4u];
                const float s0 = in.conv_ring[0ull * CONV + ch];
                const float s1 = in.conv_ring[1ull * CONV + ch];
                const float s2 = in.conv_ring[2ull * CONV + ch];
                const float x  = in.qkv[ch];
                const float acc = std::fma(w[0], s0, std::fma(w[1], s1, std::fma(w[2], s2, w[3] * x)));
                dst[l] = silu(acc);
                ss += dst[l] * dst[l];
            }
            float inv_n = 1.0f / std::sqrt(ss + EPS);
            if (!is_k) inv_n *= 0.08838834764831845f;
            for (uint32_t l = 0; l < HD; ++l) dst[l] *= inv_n;
            if (!is_k && hv < NK) {
                for (uint32_t l = 0; l < HD; ++l) {
                    const uint32_t ch = ch0 + l;
                    o.conv_out[0ull * CONV + ch] = in.conv_ring[1ull * CONV + ch];
                    o.conv_out[1ull * CONV + ch] = in.conv_ring[2ull * CONV + ch];
                    o.conv_out[2ull * CONV + ch] = in.qkv[ch];
                }
            } else if (is_k && hv < NK) {
                for (uint32_t l = 0; l < HD; ++l) {
                    const uint32_t ch = ch0 + l;
                    o.conv_out[0ull * CONV + ch] = in.conv_ring[1ull * CONV + ch];
                    o.conv_out[1ull * CONV + ch] = in.conv_ring[2ull * CONV + ch];
                    o.conv_out[2ull * CONV + ch] = in.qkv[ch];
                }
            }
        }
        for (uint32_t col = 0; col < HD; ++col) {
            const uint32_t ch_v = (2u * NK + hv) * 128u + col;
            const float* w = &in.conv_w[static_cast<size_t>(ch_v) * 4u];
            const float s0 = in.conv_ring[0ull * CONV + ch_v];
            const float s1 = in.conv_ring[1ull * CONV + ch_v];
            const float s2 = in.conv_ring[2ull * CONV + ch_v];
            const float x  = in.qkv[ch_v];
            const float acc = std::fma(w[0], s0, std::fma(w[1], s1, std::fma(w[2], s2, w[3] * x)));
            vconv[static_cast<size_t>(hv) * HD + col] = silu(acc);
            o.conv_out[0ull * CONV + ch_v] = s1;
            o.conv_out[1ull * CONV + ch_v] = s2;
            o.conv_out[2ull * CONV + ch_v] = x;
        }
    }

    for (uint32_t hv = 0; hv < NV; ++hv) {
        const HeadGate hg = head_gate(in, hv);
        float* S = &o.S[static_cast<size_t>(hv) * HD * HD];
        const float* q = &qn[static_cast<size_t>(hv) * HD];
        const float* k = &kn[static_cast<size_t>(hv) * HD];
        for (uint32_t col = 0; col < HD; ++col) {
            float pred = 0.0f;
            for (uint32_t i = 0; i < HD; ++i) pred = std::fma(S[static_cast<size_t>(i) * HD + col], k[i], pred);
            const float dv = hg.beta * (vconv[static_cast<size_t>(hv) * HD + col] - hg.g * pred);
            float fo = 0.0f;
            for (uint32_t i = 0; i < HD; ++i) {
                float& s = S[static_cast<size_t>(i) * HD + col];
                s = std::fma(k[i], dv, hg.g * s);
                fo = std::fma(s, q[i], fo);
            }
            o.fo[static_cast<size_t>(hv) * HD + col] = fo;
        }
        float total = 0.0f;
        for (uint32_t col = 0; col < HD; ++col) {
            const float f = o.fo[static_cast<size_t>(hv) * HD + col];
            total += f * f;
        }
        const float inv_r = 1.0f / std::sqrt(total * (1.0f / 128.0f) + EPS);
        const float* zp = &in.qkv[Z_OFF + hv * 128u];
        for (uint32_t blk = 0; blk < 4u; ++blk) {
            float* ds = &o.ds[static_cast<size_t>(hv) * 8u + blk * 2u];
            float amax = 0.0f, bsum = 0.0f;
            int8_t qv[32];
            float yv[32];
            for (uint32_t t = 0; t < 32u; ++t) {
                const uint32_t j = blk * 32u + t;
                const float f = o.fo[static_cast<size_t>(hv) * HD + j];
                float y = (f * inv_r * in.ssm_norm_w[j]) * silu(zp[j]);
                yv[t] = y;
                amax = std::fmax(amax, std::fabs(y));
                bsum += y;
            }
            const float d = amax * (1.0f / 127.0f);
            const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            for (uint32_t t = 0; t < 32u; ++t) {
                qv[t] = static_cast<int8_t>(std::max(-127, std::min(127, static_cast<int>(std::nearbyint(yv[t] * inv_d)))));
                o.q8[static_cast<size_t>(hv) * HD + blk * 32u + t] = qv[t];
            }
            ds[0] = d;
            ds[1] = bsum;
        }
    }
    return o;
}

double rel_err(float a, float b) {
    return std::fabs(static_cast<double>(a) - b) / std::max(1e-6, std::fabs(static_cast<double>(b)));
}

// Absolute + relative bound, so near-zero cancellation is judged by atol and normal magnitudes
// by rtol. The oracle uses a different reduction order, so this is a numerical-equivalence check.
void compare_close(const char* what, const std::vector<float>& a, const std::vector<float>& b,
                   double atol, double rtol) {
    VJ_CHECK(a.size() == b.size());
    double worst = 0.0, mrel = 0.0, mabs = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double err = std::fabs(static_cast<double>(a[i]) - b[i]);
        worst = std::max(worst, err / (atol + rtol * std::fabs(static_cast<double>(b[i]))));
        mrel  = std::max(mrel, rel_err(a[i], b[i]));
        mabs  = std::max(mabs, err);
    }
    std::printf("  %-14s err/limit=%.3g rel=%.3g abs=%.3g\n", what, worst, mrel, mabs);
    VJ_CHECK(worst <= 1.0);
}

void add_case(std::mt19937& rng, Inputs& in, bool zero_z) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    in.qkv.assign(STRIDE, 0.0f);
    in.conv_w.resize(static_cast<size_t>(CONV) * 4u);
    in.conv_ring.assign(static_cast<size_t>(SLOTS) * 3u * CONV, 0.0f);
    in.dt_bias.resize(NV);
    in.ssm_a.resize(NV);
    in.ssm_norm_w.resize(HD);
    in.S0.assign(static_cast<size_t>(NV) * HD * HD, 0.0f);

    for (auto& v : in.conv_w) v = nd(rng) * 0.25f;
    for (auto& v : in.conv_ring) v = nd(rng) * 0.20f;
    for (auto& v : in.S0) v = nd(rng) * 0.05f;
    for (auto& v : in.ssm_norm_w) v = 1.0f + 0.1f * nd(rng);
    for (uint32_t h = 0; h < NV; ++h) {
        in.dt_bias[h] = 0.5f * nd(rng);
        in.ssm_a[h] = -0.05f - 0.45f * ud(rng);   // decay in (0, 1)
    }
    for (uint32_t i = 0; i < STRIDE; ++i) in.qkv[i] = nd(rng);
    for (uint32_t hv = 0; hv < NV; ++hv) {
        in.qkv[ALPHA_OFF + hv] = 0.3f * nd(rng);
        in.qkv[BETA_OFF + hv]  = 0.5f * nd(rng);
    }
    // Softplus > 20 branch for a few heads, and a near-1 sigmoid for beta.
    in.qkv[ALPHA_OFF + 3u] = 24.0f; in.dt_bias[3] = 1.0f;
    in.qkv[BETA_OFF + 7u]  = 6.0f;
    if (zero_z) {
        for (uint32_t hv = 0; hv < NV; ++hv)
            for (uint32_t j = 0; j < HD; ++j) in.qkv[Z_OFF + hv * 128u + j] = 0.0f;
    }
}

uint32_t count_q8_diff(const std::vector<int8_t>& a, const std::vector<int8_t>& b) {
    uint32_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) n += (a[i] != b[i]);
    return n;
}

bool q8_or_ds_differ(const Outputs& a, const Outputs& b) {
    return std::memcmp(a.q8.data(), b.q8.data(), a.q8.size()) != 0 ||
           std::memcmp(a.ds.data(), b.ds.data(), a.ds.size() * 4u) != 0;
}

} // namespace

int main() {
    std::printf("[gdn_sv8_ref] Split-V=8 GDN ConvRec (two ordered dispatches): bit-exact vs Split-V=4, "
                "FP32 oracle tolerance\n");
    std::printf("[gdn_sv8_ref] stage1 publishes fo/pair-squares; stage2 consumes only that scratch\n");
    std::mt19937 rng(20261125u);
    for (int trial = 0; trial < 3; ++trial) {
        const bool zero_z = (trial == 2);
        Inputs in;
        add_case(rng, in, zero_z);

        const Outputs v4  = run_variant(in, 4u);
        const Outputs v4b = run_variant(in, 4u);
        const Outputs v8  = run_variant(in, 8u);
        const Outputs v8bad = run_variant(in, 8u, true);
        const Outputs ref = run_oracle(in);

        // Deterministic model (no scheduling dependence).
        VJ_CHECK(std::memcmp(v4.S.data(), v4b.S.data(), v4.S.size() * 4u) == 0);
        VJ_CHECK(std::memcmp(v4.q8.data(), v4b.q8.data(), v4.q8.size()) == 0);
        VJ_CHECK(std::memcmp(v4.ds.data(), v4b.ds.data(), v4.ds.size() * 4u) == 0);

        // Split-V=8 must reproduce Split-V=4 bit for bit.
        VJ_CHECK(std::memcmp(v8.conv_out.data(), v4.conv_out.data(), v4.conv_out.size() * 4u) == 0);
        VJ_CHECK(std::memcmp(v8.S.data(), v4.S.data(), v4.S.size() * 4u) == 0);
        VJ_CHECK(std::memcmp(v8.fo.data(), v4.fo.data(), v4.fo.size() * 4u) == 0);
        VJ_CHECK(std::memcmp(v8.q8.data(), v4.q8.data(), v4.q8.size()) == 0);
        VJ_CHECK(std::memcmp(v8.ds.data(), v4.ds.data(), v4.ds.size() * 4u) == 0);
        std::printf("[trial %d] split8 == split4: conv ring, S, fo, q8 (%zu), d/bsum bit-exact%s\n",
                    trial, v8.q8.size(), zero_z ? " [zero-z amax=0 case]" : "");

        // Two-dispatch model: stage 2 must be a pure function of the published scratch. Perturb one
        // published value (fo of column 0 / first pair-square) by 0.1% and require the quant outputs
        // to move; a stage 2 that re-derived fo from the inputs would ignore the perturbation. (A
        // 1-ulp bump can be absorbed by the downstream multiply/round, which would make the check
        // vacuous, so the perturbation is deliberately above float rounding. The zero-z case is
        // statistically immune: y = ... * silu(0) = 0 for every column, so any scratch bump is
        // multiplied by zero; that case is covered by the bit-exact and oracle checks instead.)
        if (!zero_z) {
            const Scratch s1 = stage1(in, 8u);
            const Outputs base = stage2(in, s1, 8u);
            VJ_CHECK(std::memcmp(base.q8.data(), v8.q8.data(), base.q8.size()) == 0);
            VJ_CHECK(std::memcmp(base.ds.data(), v8.ds.data(), base.ds.size() * 4u) == 0);

            Scratch pfo = s1;
            pfo.fo[0] *= 1.001f;
            VJ_CHECK(q8_or_ds_differ(stage2(in, pfo, 8u), base));

            Scratch psq = s1;
            psq.sq[0] *= 1.001f;
            VJ_CHECK(q8_or_ds_differ(stage2(in, psq, 8u), base));
            std::printf("  published-scratch dependency: fo and sq perturbations change q8/ds\n");
        }

        // Negative control: the same comparison must reject a wrong quant scale in the 8-way kernel.
        if (!zero_z) {
            VJ_CHECK(std::memcmp(v8bad.ds.data(), v4.ds.data(), v4.ds.size() * 4u) != 0);
            VJ_CHECK(std::memcmp(v8bad.q8.data(), v4.q8.data(), v4.q8.size()) != 0);
            std::printf("  negative control (halved d / doubled inv_d) detected: ds and q8 differ\n");
        }

        // Independent FP32 oracle (different reduction order) tolerance.
        compare_close("conv ring", v8.conv_out, ref.conv_out, 2e-6, 2e-5);
        compare_close("S state", v8.S, ref.S, 5e-7, 2e-5);
        compare_close("fo", v8.fo, ref.fo, 5e-6, 2e-5);
        compare_close("d/bsum", v8.ds, ref.ds, 2e-6, 2e-5);
        const uint32_t qdiff = count_q8_diff(v8.q8, ref.q8);
        std::printf("  q8 vs oracle   differing bytes=%u / %zu\n", qdiff, v8.q8.size());
        VJ_CHECK(qdiff * 1000u <= static_cast<uint32_t>(v8.q8.size()));  // <= 0.1% one-LSB boundary flips
    }
    std::printf("test_gdn_conv_splitv8_ref: all checks passed\n");
    return 0;
}
