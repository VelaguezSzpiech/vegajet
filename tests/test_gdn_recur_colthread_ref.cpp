// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU emulation of the two M=1 GDN ConvRec layouts in ops/kernels/vj_ornith_attn.hip:
//   5b3 ornith_gdn_conv_recur_splitv_quant_m1_kernel     (default, 8 row groups + LDS partials)
//   5b4 ornith_gdn_conv_recur_colthread_quant_m1_kernel  (VEGAJET_V12_GDN_SPLITV=1, 8 lanes per column)
// Both are emulated at thread level (grid NV*4 x 256) with the kernels' own buffer layouts, column
// ownership and arithmetic order, through Phase 0 (Conv1D + SiLU + L2 norm + gate), the recurrence and the
// gated RMSNorm/quant epilogue. What is checked:
//   * the column-thread kernel is bit-identical to the Split-V one for state, conv ring, int8 output and ds
//     (its ordered 8-lane __shfl chain adds the same eight 16-row partials left to right as 5b3's sm_red);
//   * both equal a plain row-sequential recurrence (one 128-term fma chain per column), so splitting the sum
//     into eight 16-row chains changes no bit;
//   * the ordered chain is actually exercised: the same eight partials through a butterfly tree do differ.
// GPU-only approximations (rsqrtf, __expf) are replaced by shared host equivalents: only agreement between
// the two layouts and the reference is asserted, never the exact GPU approximation.
#include "tests/test_common.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr uint32_t NK = 2, NV = 2, HD = 128;      // k heads, v heads, head dim
constexpr uint32_t CONV = (2 * NK + NV) * HD;     // Q,K,V channel count
constexpr uint32_t ZOFF = CONV, AOFF = CONV + NV * HD, BOFF = AOFF + NV;
constexpr uint32_t GBDIM = BOFF + NV;             // qkv_gab row: Q,K,V,Z,alpha,beta
constexpr uint32_t LPC = 8, ROWS = 16, COLS = 32; // column-thread layout: lanes/column, rows/lane
constexpr uint32_t SLOTS = 17;                    // VJ_GDN_RING_SLOTS

float vj_exp(float x) { return std::exp(x); }
float vj_rsqrt(float x) { return 1.0f / std::sqrt(x); }

// Kernel order for dv, with the multiply kept out of the subtraction (no host fma contraction).
inline float vj_dv(float beta, float vj, float g, float pred) {
    const float t = g * pred;
    const float u = vj - t;
    return beta * u;
}

// wave64_sum: xor butterfly 1,2,4,8,16 (ds_swizzle) then 32. Converges to one value in every lane.
float w64_sum(std::vector<float> v) {
    for (uint32_t off = 1; off < 64; off <<= 1) {
        std::vector<float> n(v.size());
        for (size_t l = 0; l < v.size(); ++l) n[l] = v[l] + v[l ^ off];
        v.swap(n);
    }
    return v[0];
}

// Ordered 8-lane sum: partial 0 + 1 + ... + 7, the order of 5b3's sm_red chain and of 5b4's
// vj_gdn_ct_group_sum8.
float ordered8(const float* p) {
    float s = p[0];
    s += p[1]; s += p[2]; s += p[3]; s += p[4]; s += p[5]; s += p[6]; s += p[7];
    return s;
}

// The same eight partials through a butterfly tree (what a cheaper shfl/xor reduction would give).
float butterfly8(float* p) {
    for (uint32_t off = 1; off < 8; off <<= 1)
        for (uint32_t l = 0; l < 8; ++l) p[l] = p[l] + p[l ^ off];
    return p[0];
}

// 16-lane xor tree in the epilogue's offset order 8,4,2,1 (__shfl_xor chain of the bsum reduction).
float bsum16(const float* b) {
    float v[16];
    for (uint32_t i = 0; i < 16; ++i) v[i] = b[i];
    for (uint32_t off = 8; off >= 1; off >>= 1)
        for (uint32_t l = 0; l < 16; ++l) v[l] = v[l] + v[l ^ off];
    return v[0];
}

struct Bufs {
    std::vector<float> qkv, conv_w, conv_ring, dt, sa, nw, S;
    uint32_t c_in = 0, c_out = 1, shead = 0, uring = 0;
    float eps = 1e-6f;
};

struct Res {
    std::vector<float> S, conv_ring;
    std::vector<int8_t> q8;
    std::vector<float> ds;          // [hv][part][2] = d, bsum
    std::vector<float> o_out;       // per-head o before the epilogue (for the row reference)
    std::vector<float> q0, k0, v0;  // phase-0 output per head (for the row reference)
    std::vector<float> g0, b0;
};

// One token through one of the two fused M=1 GDN kernels: grid NV*4 WGs x 256 threads.
Res run_stage(const Bufs& b, bool colthread) {
    Res r;
    r.S = b.S;
    r.conv_ring = b.conv_ring;
    r.q8.assign(NV * HD, 0);
    r.ds.assign(NV * 4 * 2, 0.0f);
    r.q0.assign(NV * HD, 0.0f);
    r.k0.assign(NV * HD, 0.0f);
    r.v0.assign(NV * HD, 0.0f);
    r.o_out.assign(NV * HD, 0.0f);
    r.g0.assign(NV, 0.0f);
    r.b0.assign(NV, 0.0f);
    std::vector<float> sq(NV * 64, 0.0f);
    const float* cr_in = b.conv_ring.data() + uint64_t(b.c_in) * 3 * CONV;
    float* cr_out = r.conv_ring.data() + uint64_t(b.c_out) * 3 * CONV;

    for (uint32_t hv = 0; hv < NV; ++hv) {
        std::vector<float> o_all(4 * COLS, 0.0f);  // sm_fo / sm_o of every part
        for (uint32_t part = 0; part < 4; ++part) {
            const uint32_t col0 = part * COLS, hk = hv % NK;
            float sm_q[HD], sm_k[HD], sm_red[8 * COLS], sm_dv[COLS], sm_gb[2];
            float y_raw[256];

            // ---- Phase 0: Conv1D + SiLU + L2 norm (Q/K) + V conv + alpha/beta gate ----
            for (uint32_t tid = 0; tid < 256; ++tid) {
                const uint32_t is_k = tid >> 7, l = tid & 127u;
                const uint32_t ch = (is_k ? (NK + hk) : hk) * HD + l;
                const float* w = &b.conv_w[uint64_t(ch) * 4];
                const float s0 = cr_in[0ULL * CONV + ch], s1 = cr_in[1ULL * CONV + ch], s2 = cr_in[2ULL * CONV + ch];
                const float x = b.qkv[ch];
                if (hv < NK && part == 0) {
                    cr_out[0ULL * CONV + ch] = s1;
                    cr_out[1ULL * CONV + ch] = s2;
                    cr_out[2ULL * CONV + ch] = x;
                }
                const float acc = std::fma(w[0], s0, std::fma(w[1], s1, std::fma(w[2], s2, w[3] * x)));
                y_raw[tid] = acc / (1.0f + vj_exp(-acc));
            }
            for (uint32_t wave = 0; wave < 4; ++wave) {
                std::vector<float> v(64);
                for (uint32_t l = 0; l < 64; ++l) v[l] = y_raw[wave * 64 + l] * y_raw[wave * 64 + l];
                sm_red[wave] = w64_sum(v);
            }
            for (uint32_t tid = 0; tid < 256; ++tid) {
                const uint32_t is_k = tid >> 7, l = tid & 127u;
                float inv_n = vj_rsqrt(sm_red[is_k ? 2u : 0u] + sm_red[is_k ? 3u : 1u] + b.eps);
                if (!is_k) inv_n *= 0.08838834764831845f;
                const float y = y_raw[tid] * inv_n;
                if (!is_k) sm_q[l] = y; else sm_k[l] = y;
            }
            for (uint32_t t = 0; t < COLS; ++t) {  // tid < 32: V conv of this part's 32 channels
                const uint32_t ch_v = (2u * NK + hv) * HD + col0 + t;
                const float* wv = &b.conv_w[uint64_t(ch_v) * 4];
                const float sv0 = cr_in[0ULL * CONV + ch_v], sv1 = cr_in[1ULL * CONV + ch_v], sv2 = cr_in[2ULL * CONV + ch_v];
                const float xv = b.qkv[ch_v];
                const float av = std::fma(wv[0], sv0, std::fma(wv[1], sv1, std::fma(wv[2], sv2, wv[3] * xv)));
                sm_dv[t] = av / (1.0f + vj_exp(-av));
                cr_out[0ULL * CONV + ch_v] = sv1;
                cr_out[1ULL * CONV + ch_v] = sv2;
                cr_out[2ULL * CONV + ch_v] = xv;
            }
            const float a  = b.qkv[AOFF + hv] + b.dt[hv];
            const float sp = (a > 20.0f) ? a : std::log1p(vj_exp(a));
            sm_gb[0] = vj_exp(sp * b.sa[hv]);
            sm_gb[1] = 1.0f / (1.0f + vj_exp(-b.qkv[BOFF + hv]));
            const float g_exp = sm_gb[0], beta = sm_gb[1];

            // ---- Phase 1: recurrence (only the ownership of the 128 rows differs) ----
            const uint64_t head = uint64_t(hv) * HD * HD;
            const uint64_t slot = uint64_t(b.shead) * uint64_t(NV) * HD * HD;
            const uint64_t out_slot = (b.uring != 0) ? uint64_t((b.shead + 1u) % SLOTS) * uint64_t(NV) * HD * HD : slot;
            float* S_in = r.S.data() + slot + head;
            float* S_out = r.S.data() + out_slot + head;
            float dv[COLS];
            if (!colthread) {  // 5b3: thread (rg, c) owns rows rg*16..+15 of column col0+c
                float s_reg[8][COLS][ROWS];
                for (uint32_t rg = 0; rg < 8; ++rg)
                    for (uint32_t c = 0; c < COLS; ++c)
                        for (uint32_t i = 0; i < ROWS; ++i)
                            s_reg[rg][c][i] = S_in[uint64_t(rg * ROWS + i) * HD + col0 + c];
                for (uint32_t c = 0; c < COLS; ++c)
                    for (uint32_t rg = 0; rg < 8; ++rg) {
                        float p = 0.0f;
                        for (uint32_t i = 0; i < ROWS; ++i) p = std::fma(s_reg[rg][c][i], sm_k[rg * ROWS + i], p);
                        sm_red[rg * COLS + c] = p;
                    }
                for (uint32_t c = 0; c < COLS; ++c) {
                    float pp[8];
                    for (uint32_t rg = 0; rg < 8; ++rg) pp[rg] = sm_red[rg * COLS + c];
                    dv[c] = vj_dv(beta, sm_dv[c], g_exp, ordered8(pp));
                }
                for (uint32_t c = 0; c < COLS; ++c) {
                    float oo[8];
                    for (uint32_t rg = 0; rg < 8; ++rg) {
                        float o = 0.0f;
                        for (uint32_t i = 0; i < ROWS; ++i) {
                            const uint32_t row = rg * ROWS + i;
                            const float sv = std::fma(sm_k[row], dv[c], g_exp * s_reg[rg][c][i]);
                            S_out[uint64_t(row) * HD + col0 + c] = sv;
                            o = std::fma(sv, sm_q[row], o);
                        }
                        oo[rg] = o;
                    }
                    o_all[part * COLS + c] = ordered8(oo);
                }
            } else {  // 5b4: lane group (c, sub) owns rows sub*16..+15 of column col0+c
                float s_reg[COLS][LPC][ROWS];
                for (uint32_t c = 0; c < COLS; ++c)
                    for (uint32_t sub = 0; sub < LPC; ++sub)
                        for (uint32_t i = 0; i < ROWS; ++i)
                            s_reg[c][sub][i] = S_in[uint64_t(sub * ROWS + i) * HD + col0 + c];
                for (uint32_t c = 0; c < COLS; ++c) {
                    float pp[LPC];
                    for (uint32_t sub = 0; sub < LPC; ++sub) {
                        float p = 0.0f;
                        for (uint32_t i = 0; i < ROWS; ++i) p = std::fma(s_reg[c][sub][i], sm_k[sub * ROWS + i], p);
                        pp[sub] = p;
                    }
                    dv[c] = vj_dv(beta, sm_dv[c], g_exp, ordered8(pp));  // == vj_gdn_ct_group_sum8
                }
                for (uint32_t c = 0; c < COLS; ++c) {
                    float oo[LPC];
                    for (uint32_t sub = 0; sub < LPC; ++sub) {
                        float o = 0.0f;
                        for (uint32_t i = 0; i < ROWS; ++i) {
                            const uint32_t row = sub * ROWS + i;
                            const float sv = std::fma(sm_k[row], dv[c], g_exp * s_reg[c][sub][i]);
                            S_out[uint64_t(row) * HD + col0 + c] = sv;
                            o = std::fma(sv, sm_q[row], o);
                        }
                        oo[sub] = o;
                    }
                    o_all[part * COLS + c] = ordered8(oo);  // == vj_gdn_ct_group_sum8
                }
            }
            // Sum-of-squares partials keep sq[hv*64 + part*16 + t] with t = column pair; 5b4 reads the odd
            // column with one wave broadcast (__shfl from the adjacent 8-lane group = column 2t+1).
            for (uint32_t t = 0; t < 16; ++t) {
                const float fo0 = o_all[part * COLS + 2 * t];
                const float fo1 = o_all[part * COLS + 2 * t + 1];
                sq[hv * 64 + part * 16 + t] = fo0 * fo0 + fo1 * fo1;
            }
            for (uint32_t c = 0; c < COLS; ++c) {
                r.o_out[hv * HD + col0 + c] = o_all[part * COLS + c];
                r.q0[hv * HD + col0 + c] = sm_q[col0 + c];
                r.k0[hv * HD + col0 + c] = sm_k[col0 + c];
                r.v0[hv * HD + col0 + c] = sm_dv[c];
            }
            if (part == 0) { r.g0[hv] = g_exp; r.b0[hv] = beta; }
        }
        // ---- Phase 2: gated RMSNorm + quant ----
        std::vector<float> v(64);
        for (uint32_t l = 0; l < 64; ++l) v[l] = sq[hv * 64 + l];
        const float total_sq = w64_sum(v);
        for (uint32_t part = 0; part < 4; ++part) {
            const float inv_r = vj_rsqrt(total_sq * (1.0f / 128.0f) + b.eps);
            float y0s[16], y1s[16], bs[16];
            for (uint32_t t = 0; t < 16; ++t) {
                const uint32_t j0 = part * COLS + t * 2u;
                const float fo0 = o_all[part * COLS + t * 2u], fo1 = o_all[part * COLS + t * 2u + 1u];
                const float z0 = b.qkv[ZOFF + hv * HD + j0], z1 = b.qkv[ZOFF + hv * HD + j0 + 1u];
                y0s[t] = (fo0 * inv_r * b.nw[j0]) * (z0 / (1.0f + vj_exp(-z0)));
                y1s[t] = (fo1 * inv_r * b.nw[j0 + 1u]) * (z1 / (1.0f + vj_exp(-z1)));
                bs[t] = y0s[t] + y1s[t];
            }
            float amax = 0.0f;
            for (uint32_t t = 0; t < 16; ++t) amax = std::fmax(amax, std::fmax(std::fabs(y0s[t]), std::fabs(y1s[t])));
            const float bsum = bsum16(bs);
            const float d = amax * (1.0f / 127.0f);
            const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            for (uint32_t t = 0; t < 16; ++t) {
                const uint32_t j0 = part * COLS + t * 2u;
                const float q0v = y0s[t] * inv_d, q1v = y1s[t] * inv_d;
                r.q8[hv * HD + j0] = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, std::nearbyint(q0v))));
                r.q8[hv * HD + j0 + 1u] = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, std::nearbyint(q1v))));
            }
            r.ds[(hv * 4 + part) * 2 + 0] = d;
            r.ds[(hv * 4 + part) * 2 + 1] = bsum;
        }
    }
    return r;
}

// Plain per-column reference on the same phase-0 q/k/v/g/beta: eight 16-row fma chains per column, summed
// left to right (the arithmetic both kernels implement, written without any thread/lane mapping).
void row_ref(const Bufs& b, const Res& p0, std::vector<float>& S, std::vector<float>& o) {
    S = b.S;
    o.assign(NV * HD, 0.0f);
    const uint64_t slot = uint64_t(b.shead) * uint64_t(NV) * HD * HD;
    const uint64_t out_slot = (b.uring != 0) ? uint64_t((b.shead + 1u) % SLOTS) * uint64_t(NV) * HD * HD : slot;
    for (uint32_t hv = 0; hv < NV; ++hv) {
        const float* q = &p0.q0[hv * HD];
        const float* k = &p0.k0[hv * HD];
        const float* v = &p0.v0[hv * HD];
        const float g = p0.g0[hv], bt = p0.b0[hv];
        float* Si = S.data() + slot + uint64_t(hv) * HD * HD;
        float* So = S.data() + out_slot + uint64_t(hv) * HD * HD;
        for (uint32_t j = 0; j < HD; ++j) {
            float pp[8], ow[8];
            for (uint32_t w = 0; w < 8; ++w) {
                float p = 0.0f;
                for (uint32_t i = 0; i < ROWS; ++i)
                    p = std::fma(Si[uint64_t(w * ROWS + i) * HD + j], k[w * ROWS + i], p);
                pp[w] = p;
            }
            const float dvj = vj_dv(bt, v[j], g, ordered8(pp));
            for (uint32_t w = 0; w < 8; ++w) {
                float oo = 0.0f;
                for (uint32_t i = 0; i < ROWS; ++i) {
                    const uint32_t row = w * ROWS + i;
                    const float sv = std::fma(k[row], dvj, g * Si[uint64_t(row) * HD + j]);
                    So[uint64_t(row) * HD + j] = sv;
                    oo = std::fma(sv, q[row], oo);
                }
                ow[w] = oo;
            }
            o[hv * HD + j] = ordered8(ow);
        }
    }
}

}  // namespace

int main() {
    std::mt19937 rng(20261122u);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    int checks = 0, order_mismatch = 0;

    for (uint32_t trial = 0; trial < 3; ++trial) {
        Bufs b;
        b.qkv.resize(GBDIM);
        for (auto& x : b.qkv) x = nd(rng) * 0.7f;
        b.conv_w.resize(uint64_t(CONV) * 4);
        for (auto& x : b.conv_w) x = nd(rng) * 0.5f;
        b.conv_ring.assign(uint64_t(SLOTS) * 3 * CONV, 0.0f);
        b.S.assign(uint64_t(SLOTS) * NV * HD * HD, 0.0f);
        b.dt.resize(NV);
        b.sa.resize(NV);
        b.nw.resize(NV * HD);
        for (uint32_t i = 0; i < NV; ++i) { b.dt[i] = 0.5f * ud(rng) + 0.1f; b.sa[i] = -(0.2f + 0.6f * ud(rng)); }
        for (auto& x : b.nw) x = 0.5f + ud(rng);
        b.c_in  = trial;            // conv ring input slot (first trial: all-zero history)
        b.c_out = (trial + 1) % SLOTS;
        b.shead = trial;
        b.uring = (trial == 2) ? 1u : 0u;  // last trial exercises the next-ring-slot state write
        const uint64_t slot = uint64_t(b.shead) * uint64_t(NV) * HD * HD;
        for (uint32_t i = 0; i < NV * HD * HD; ++i) b.S[slot + i] = nd(rng) * 0.05f;

        const Res legacy = run_stage(b, false);
        const Res ct     = run_stage(b, true);

        VJ_CHECK(std::memcmp(legacy.S.data(), ct.S.data(), legacy.S.size() * 4) == 0);
        VJ_CHECK(std::memcmp(legacy.conv_ring.data(), ct.conv_ring.data(), legacy.conv_ring.size() * 4) == 0);
        VJ_CHECK(legacy.q8 == ct.q8);
        VJ_CHECK(std::memcmp(legacy.ds.data(), ct.ds.data(), legacy.ds.size() * 4) == 0);

        std::vector<float> S_ref, o_ref;
        row_ref(b, ct, S_ref, o_ref);
        VJ_CHECK(std::memcmp(S_ref.data(), ct.S.data(), S_ref.size() * 4) == 0);   // state bit-exact
        VJ_CHECK(std::memcmp(o_ref.data(), ct.o_out.data(), o_ref.size() * 4) == 0);  // o bit-exact

        // The ordered chain must matter: a butterfly over the same partials is a different number.
        for (uint32_t s = 0; s < 8; ++s) {
            float p[8];
            for (uint32_t i = 0; i < 8; ++i) p[i] = nd(rng) * (1.0f + 0.3f * float(i));
            float pc[8];
            std::memcpy(pc, p, sizeof(p));
            const float a8 = ordered8(p), b8 = butterfly8(pc);
            if (std::memcmp(&a8, &b8, 4) != 0) ++order_mismatch;
        }
        ++checks;
        std::printf("[gdn_recur_colthread] trial %u (uring=%u): column-thread == split-v == row reference, "
                    "state %zu floats, q8 %zu bytes\n", trial, b.uring, ct.S.size(), ct.q8.size());
    }
    VJ_CHECK(order_mismatch > 0);  // the test can see a reduction-order change
    std::printf("test_gdn_recur_colthread_ref: %d bit-exact stage checks, %d/%d butterfly mismatches\n",
                checks, order_mismatch, 3 * 8);
    return 0;
}
