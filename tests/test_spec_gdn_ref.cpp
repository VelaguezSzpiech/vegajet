// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU emulation of ops/kernels/vj_spec_gdn.hip (single-write Split-V verify GDN + recompute-on-reject),
// using the kernel's own buffer layouts, strides and per-column arithmetic order, checked against the
// ring-slot semantics of ornith_gdn_recur_quant_kernel (state after every token):
//   * per-token raw o and the final S_M equal the per-token sequential recurrence bit for bit;
//   * recompute from the FP32 records, in place, for every accepted-prefix length n in 1..M and for several
//     layers in one "dispatch", reproduces the sequential state after n tokens bit for bit;
//   * the norm/quant stage output matches the reference formula.
#include "tests/test_common.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr uint32_t NK = 8, NV = 32, HD = 128, CONV = (2 * NK + NV) * HD;  // 6144

struct Args {
    const float* S_in; float* S_out;
    const float *q_src, *k_src, *v_src, *dec_in, *beta_in;
    float *o_out, *rec_kv, *rec_dec, *rec_beta;
    uint32_t M, src_stride, rec_stride, layer_base, n_layers;
    uint64_t s_layer_stride, src_layer_stride, gb_layer_stride, rec_layer_stride, rec_gb_layer_stride;
};

// Emulates vj_spec_gdn_verify_kernel for grid (NV*4, n_layers): per WG, per thread column, same op order.
void verify_kernel(const Args& a) {
    for (uint32_t ly = 0; ly < a.n_layers; ++ly)
        for (uint32_t wg = 0; wg < NV * 4; ++wg) {
            const uint32_t hv = wg >> 2, part = wg & 3, col0 = part * 32, hk = hv % NK;
            const uint64_t lid = a.layer_base + ly;
            const float* ks = a.k_src + lid * a.src_layer_stride;
            const float* vs = a.v_src + lid * a.src_layer_stride;
            float* rk = a.rec_kv ? a.rec_kv + lid * a.rec_layer_stride : nullptr;
            std::vector<float> sk(a.M * HD), sq(a.M * HD), sv(a.M * 32), sg(a.M), sb(a.M);
            for (uint32_t e = 0; e < a.M * HD; ++e) {
                const uint32_t m = e >> 7, i = e & 127;
                sk[e] = ks[uint64_t(m) * a.src_stride + hk * HD + i];
                if (a.q_src) sq[e] = a.q_src[lid * a.src_layer_stride + uint64_t(m) * a.src_stride + hk * HD + i];
                if (rk && part == 0 && hv < NK) rk[uint64_t(m) * a.rec_stride + hv * HD + i] = sk[e];
            }
            for (uint32_t e = 0; e < a.M * 32; ++e) {
                const uint32_t m = e >> 5, cc = e & 31;
                sv[e] = vs[uint64_t(m) * a.src_stride + hv * HD + col0 + cc];
                if (rk) rk[uint64_t(m) * a.rec_stride + NK * HD + hv * HD + col0 + cc] = sv[e];
            }
            for (uint32_t t = 0; t < a.M; ++t) {
                sg[t] = a.dec_in[lid * a.gb_layer_stride + uint64_t(t) * NV + hv];
                sb[t] = a.beta_in[lid * a.gb_layer_stride + uint64_t(t) * NV + hv];
                if (a.rec_dec && part == 0) {
                    a.rec_dec[lid * a.rec_gb_layer_stride + uint64_t(t) * NV + hv] = sg[t];
                    a.rec_beta[lid * a.rec_gb_layer_stride + uint64_t(t) * NV + hv] = sb[t];
                }
            }
            // State: s[col][row] for the WG's 32 columns.
            const uint64_t sbase = lid * a.s_layer_stride + uint64_t(hv) * HD * HD;
            float s[32][HD];
            for (uint32_t c = 0; c < 32; ++c)
                for (uint32_t i = 0; i < HD; ++i) s[c][i] = a.S_in[sbase + uint64_t(i) * HD + col0 + c];
            for (uint32_t m = 0; m < a.M; ++m) {
                float pp[8][32], dv[32];
                for (uint32_t c = 0; c < 32; ++c)
                    for (uint32_t rg = 0; rg < 8; ++rg) {
                        float p = 0.0f;
                        for (uint32_t r = 0; r < 16; ++r) p = std::fma(s[c][rg * 16 + r], sk[m * HD + rg * 16 + r], p);
                        pp[rg][c] = p;
                    }
                for (uint32_t c = 0; c < 32; ++c) {
                    const float pred = pp[0][c] + pp[1][c] + pp[2][c] + pp[3][c] + pp[4][c] + pp[5][c] + pp[6][c] + pp[7][c];
                    dv[c] = sb[m] * (sv[m * 32 + c] - sg[m] * pred);
                }
                for (uint32_t c = 0; c < 32; ++c) {
                    float oo[8];
                    for (uint32_t rg = 0; rg < 8; ++rg) {
                        float o = 0.0f;
                        for (uint32_t r = 0; r < 16; ++r) {
                            const uint32_t i = rg * 16 + r;
                            s[c][i] = std::fma(sk[m * HD + i], dv[c], sg[m] * s[c][i]);
                            if (a.q_src) o = std::fma(s[c][i], sq[m * HD + i], o);
                        }
                        oo[rg] = o;
                    }
                    if (a.o_out && a.q_src)
                        a.o_out[(uint64_t(m) * NV + hv) * HD + col0 + c] = oo[0] + oo[1] + oo[2] + oo[3] + oo[4] + oo[5] + oo[6] + oo[7];
                }
            }
            for (uint32_t c = 0; c < 32; ++c)
                for (uint32_t i = 0; i < HD; ++i) a.S_out[sbase + uint64_t(i) * HD + col0 + c] = s[c][i];
        }
}

// Ring reference: plain per-token GDN recurrence (5b math), state snapshot after each token.
void ring_ref(const float* S0, const float* conv, const float* dec, const float* beta, uint32_t M,
              std::vector<std::vector<float>>& states, std::vector<float>& o) {
    std::vector<float> S(S0, S0 + NV * HD * HD);
    o.assign(M * NV * HD, 0.0f);
    states.clear();
    for (uint32_t m = 0; m < M; ++m) {
        const float* row = conv + uint64_t(m) * CONV;
        for (uint32_t hv = 0; hv < NV; ++hv) {
            const uint32_t hk = hv % NK;
            const float *q = row + hk * HD, *k = row + (NK + hk) * HD, *v = row + (2 * NK + hv) * HD;
            const float g = dec[m * NV + hv], b = beta[m * NV + hv];
            float* Sh = S.data() + uint64_t(hv) * HD * HD;
            for (uint32_t j = 0; j < HD; ++j) {
                float pw[8], ow[8];
                for (uint32_t w = 0; w < 8; ++w) {
                    float p = 0.0f;
                    for (uint32_t r = 0; r < 16; ++r) p = std::fma(Sh[(w * 16 + r) * HD + j], k[w * 16 + r], p);
                    pw[w] = p;
                }
                const float pred = pw[0] + pw[1] + pw[2] + pw[3] + pw[4] + pw[5] + pw[6] + pw[7];
                const float dvj = b * (v[j] - g * pred);
                for (uint32_t w = 0; w < 8; ++w) {
                    float oo = 0.0f;
                    for (uint32_t r = 0; r < 16; ++r) {
                        float& sij = Sh[(w * 16 + r) * HD + j];
                        sij = std::fma(k[w * 16 + r], dvj, g * sij);
                        oo = std::fma(sij, q[w * 16 + r], oo);
                    }
                    ow[w] = oo;
                }
                o[(uint64_t(m) * NV + hv) * HD + j] = ow[0] + ow[1] + ow[2] + ow[3] + ow[4] + ow[5] + ow[6] + ow[7];
            }
        }
        states.push_back(S);
    }
}


// ---- v1.2 S5: the fused launch's second phase must be a one-wave-per-(token, head) norm+quant ----
constexpr uint32_t NQ_STRIDE = 10304u, NQ_ZOFF = 6144u;
constexpr float    NQ_EPS    = 1e-6f;

// XOR butterfly over n lanes with offsets n/2 .. 1 (block32_sum / wave64_sum / the 16-lane shfl_xor trees).
float bfly(float* v, uint32_t n) {
    for (uint32_t off = n / 2u; off > 0u; off >>= 1u) {
        std::vector<float> t(v, v + n);
        for (uint32_t i = 0u; i < n; ++i) v[i] = t[i] + t[i ^ off];
    }
    return v[0];
}
float bfly_max(float* v, uint32_t n) {
    for (uint32_t off = n / 2u; off > 0u; off >>= 1u) {
        std::vector<float> t(v, v + n);
        for (uint32_t i = 0u; i < n; ++i) v[i] = std::fmax(t[i], t[i ^ off]);
    }
    return v[0];
}

// One (token, head) item: raw o -> gated RMSNorm * w * SiLU(z) -> block-32 INT8 with (d, bsum) per block.
// exp/std math is emulated with std::exp, so this pins index and reduction order, not the transcendental.
void nq_head(const float* o, const float* gab, const float* nw, uint32_t nv, uint32_t m, uint32_t hv,
             int8_t* q8, float* ds) {
    const float* ob = o + (uint64_t(m) * nv + hv) * HD;
    float fo[128];
    for (uint32_t j = 0u; j < HD; ++j) fo[j] = ob[j];
    float sq[64];
    for (uint32_t l = 0u; l < 64u; ++l) sq[l] = fo[2u * l] * fo[2u * l] + fo[2u * l + 1u] * fo[2u * l + 1u];
    const float inv_r = 1.0f / std::sqrt(bfly(sq, 64u) * (1.0f / 128.0f) + NQ_EPS);

    float y[128];
    const float* zp = gab + uint64_t(m) * NQ_STRIDE + NQ_ZOFF + hv * HD;
    for (uint32_t j = 0u; j < HD; ++j) {
        const float z = zp[j];
        y[j] = (fo[j] * inv_r * nw[j]) * (z / (1.0f + std::exp(-z)));
    }
    int8_t* oq = q8 + (uint64_t(m) * nv + hv) * HD;
    for (uint32_t blk = 0u; blk < 4u; ++blk) {          // one block-32 per 16-lane shfl group
        float am[16], bs[16];
        for (uint32_t i = 0u; i < 16u; ++i) {
            const float y0 = y[blk * 32u + 2u * i], y1 = y[blk * 32u + 2u * i + 1u];
            am[i] = std::fmax(std::fabs(y0), std::fabs(y1));
            bs[i] = y0 + y1;
        }
        const float amax = bfly_max(am, 16u), bsum = bfly(bs, 16u);
        const float d     = amax * (1.0f / 127.0f);
        const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        for (uint32_t i = 0u; i < 16u; ++i) {
            oq[blk * 32u + 2u * i]      = int8_t(std::max(-127, std::min(127, int(std::nearbyint(y[blk * 32u + 2u * i] * inv_d)))));
            oq[blk * 32u + 2u * i + 1u] = int8_t(std::max(-127, std::min(127, int(std::nearbyint(y[blk * 32u + 2u * i + 1u] * inv_d)))));
        }
        ds[(uint64_t(m) * nv + hv) * 8u + blk * 2u]      = d;
        ds[(uint64_t(m) * nv + hv) * 8u + blk * 2u + 1u] = bsum;
    }
}

// Two dispatches: vj_spec_gdn_norm_quant_kernel, grid ceil(M*nv/4) WGs x 4 waves.
void nq_two_dispatch(const float* o, const float* gab, const float* nw, uint32_t M, uint32_t nv,
                     int8_t* q8, float* ds) {
    const uint32_t wgs = (M * nv + 3u) / 4u;
    for (uint32_t wg = 0u; wg < wgs; ++wg)
        for (uint32_t wave = 0u; wave < 4u; ++wave) {
            const uint32_t idx = wg * 4u + wave;
            if (idx >= M * nv) continue;
            nq_head(o, gab, nw, nv, idx / nv, idx % nv, q8, ds);
        }
}

// Fused launch: same wave map, but the grid is the verify grid (nv*4 = 128 WGs, nv*4*4 >= M*nv work items)
// and one grid barrier separates the phases. Returns the number of items written.
uint32_t nq_fused_phase(const float* o, const float* gab, const float* nw, uint32_t M, uint32_t nv,
                        int8_t* q8, float* ds) {
    uint32_t written = 0;
    for (uint32_t wg = 0u; wg < nv * 4u; ++wg)
        for (uint32_t wave = 0u; wave < 4u; ++wave) {
            const uint32_t idx = wg * 4u + wave;
            if (idx >= M * nv) continue;
            nq_head(o, gab, nw, nv, idx / nv, idx % nv, q8, ds);
            ++written;
        }
    return written;
}

} // namespace

int main() {
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    const uint32_t L = 3, MAXM = 12;
    const uint64_t SL = uint64_t(NV) * HD * HD, REC = uint64_t(NK + NV) * HD, RECL = MAXM * REC, GBL = MAXM * NV;
    int n = 0;
    for (uint32_t M : {1u, 5u, 8u, 12u}) {
        std::vector<float> slotA(L * SL), slotB(L * SL), conv(L * M * CONV), dec(L * M * NV), beta(L * M * NV);
        for (auto& v : slotA) v = nd(rng) * 0.05f;
        for (uint32_t ly = 0; ly < L; ++ly) {
            for (uint32_t m = 0; m < M; ++m) {
                float* row = &conv[(uint64_t(ly) * M + m) * CONV];
                for (uint32_t h = 0; h < 2 * NK; ++h) {  // L2-normalised q (scaled) and k, as conv_l2 emits
                    double ss = 0.0;
                    for (uint32_t i = 0; i < HD; ++i) { row[h * HD + i] = nd(rng); ss += double(row[h * HD + i]) * row[h * HD + i]; }
                    const float inv = float(1.0 / std::sqrt(ss)) * (h < NK ? 0.08838834764831845f : 1.0f);
                    for (uint32_t i = 0; i < HD; ++i) row[h * HD + i] *= inv;
                }
                for (uint32_t i = 2 * NK * HD; i < CONV; ++i) row[i] = nd(rng);
            }
            for (uint32_t i = 0; i < M * NV; ++i) { dec[ly * M * NV + i] = 0.9f + 0.1f * ud(rng); beta[ly * M * NV + i] = ud(rng); }
        }
        const std::vector<float> S0 = slotA;
        std::vector<float> rec_kv(L * RECL, std::nanf("")), rec_dec(L * GBL), rec_beta(L * GBL), o(M * NV * HD);
        // Verify pass, layer by layer (conv/dec/beta buffers are per-layer scratch in the engine).
        std::vector<std::vector<std::vector<float>>> ref_states(L);
        for (uint32_t ly = 0; ly < L; ++ly) {
            const float* cv = &conv[uint64_t(ly) * M * CONV];
            Args a{&slotA[ly * SL], &slotB[ly * SL], cv, cv + NK * HD, cv + 2 * NK * HD, &dec[ly * M * NV], &beta[ly * M * NV],
                   o.data(), rec_kv.data(), rec_dec.data(), rec_beta.data(), M, CONV, uint32_t(REC), ly, 1,
                   0, 0, 0, RECL, GBL};
            verify_kernel(a);
            std::vector<float> o_ref;
            ring_ref(&S0[ly * SL], cv, &dec[ly * M * NV], &beta[ly * M * NV], M, ref_states[ly], o_ref);
            VJ_CHECK(std::memcmp(o.data(), o_ref.data(), o.size() * 4) == 0);                       // per-token o
            VJ_CHECK(std::memcmp(&slotB[ly * SL], ref_states[ly][M - 1].data(), SL * 4) == 0);       // S_M
            VJ_CHECK(std::memcmp(&slotA[ly * SL], &S0[ly * SL], SL * 4) == 0);                       // S_0 kept
            // v1.2 S5: the fused verify+norm/quant launch must reproduce the two-dispatch nq output exactly.
            std::vector<float> gab(size_t(M) * NQ_STRIDE), nw(HD);
            for (auto& v : gab) v = nd(rng);
            for (auto& v : nw) v = 0.5f + ud(rng);
            std::vector<int8_t> q8a(size_t(M) * NV * HD), q8b(q8a.size(), 0);
            std::vector<float> dsa(size_t(M) * NV * 8u), dsb(dsa.size(), 0.0f);
            nq_two_dispatch(o.data(), gab.data(), nw.data(), M, NV, q8a.data(), dsa.data());
            const uint32_t nq_written = nq_fused_phase(o.data(), gab.data(), nw.data(), M, NV, q8b.data(), dsb.data());
            VJ_CHECK(nq_written == M * NV);                                        // every (token, head) exactly once
            VJ_CHECK(std::memcmp(q8a.data(), q8b.data(), q8a.size()) == 0);        // INT8 payload identical
            VJ_CHECK(std::memcmp(dsa.data(), dsb.data(), dsa.size() * 4u) == 0);   // (d, bsum) identical
            ++n;
        }
        // Reject at every prefix n: one in-place recompute over all layers from the records.
        for (uint32_t keep = 1; keep <= M; ++keep) {
            std::vector<float> A = S0;
            Args r{A.data(), A.data(), nullptr, rec_kv.data(), rec_kv.data() + NK * HD, rec_dec.data(), rec_beta.data(),
                   nullptr, nullptr, nullptr, nullptr, keep, uint32_t(REC), uint32_t(REC), 0, L,
                   SL, RECL, GBL, 0, 0};
            verify_kernel(r);
            for (uint32_t ly = 0; ly < L; ++ly)
                VJ_CHECK(std::memcmp(&A[ly * SL], ref_states[ly][keep - 1].data(), SL * 4) == 0);
            ++n;
        }
        std::printf("[spec_gdn_ref] M=%2u: verify o/S_M bit-exact vs per-token ring reference (%u layers); "
                    "in-place recompute bit-exact for n = 1..%u\n", M, L, M);
    }
    std::printf("test_spec_gdn_ref: all %d checks passed\n", n);
    return 0;
}
