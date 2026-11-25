// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Big Lever 3 CPU numerical references for the new prefill kernels (no GPU):
//  1. FA2 (vj_prefill_fa2.hip): the wave reduce-scatter lane mapping, the INT8 -> FP16 magic conversion, and the
//     Br=16 / Bc=32 tiled online softmax (FP16 Q, INT8-exact K/V, per-tile rescale) vs the per-position online
//     softmax of ornith_attn_flash_quant_kernel and a double-precision softmax.
//  2. Tiled W4A8 GEMM (vj_prefill_gemm.hip): a faithful emulation of the 64x64 / 64x128 tiles (thread mapping,
//     LDS unpack, K-tiles, tails) vs the sequential per-block order it claims, vs the 32-lane + butterfly order
//     of ornith_gemm_q4a8_tcc_kernel, vs double.
//  3. IQ tiles: SWAR sign application vs plain negation on the real ggml grids; the MoE tile decode (wave-part
//     mapping of vj_prefill_moe_gu_iq2s_tiled_kernel) vs ggml's dequantize_row_iq2_s on raw blocks packed with the
//     loader's merged layout; the dequant-once Gate+Up tile vs the grouped kernel's lane order; IQ3_XXS SWAR decode
//     vs ggml's dequantize_row_iq3_xxs.
//  4. INT8 AllReduce pack / unpack bound (reference in core/prefill_planner.cpp).
#include "core/ggml_iq_tables.hpp"
#include "core/prefill_planner.hpp"
#include "tests/test_common.hpp"

#include <algorithm>
#include <cmath>
#include <random>

using vegajet::prefill::f16_to_f32;
using vegajet::prefill::f32_to_f16_rne;
namespace gt = vegajet::ggml_tables;

namespace {

float h16(float x) { return f16_to_f32(f32_to_f16_rne(x)); }
int32_t rn_even(float x) { return static_cast<int32_t>(std::nearbyint(x)); }

// =============================================================================================================
// 1. FA2
// =============================================================================================================
void test_fa2_reduce_scatter() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    for (int trial = 0; trial < 50; ++trial) {
        float part[64][8];
        double want[8] = {};
        for (int l = 0; l < 64; ++l) for (int k = 0; k < 8; ++k) { part[l][k] = ud(rng); want[k] += part[l][k]; }
        float cur[64][8];
        std::memcpy(cur, part, sizeof(cur));
        auto step = [&](int mask, int n) {        // keep/send halving step exactly as the kernel
            float nxt[64][8];
            std::memcpy(nxt, cur, sizeof(nxt));
            for (int l = 0; l < 64; ++l) {
                const bool up = (l & mask) != 0;
                const int pl = l ^ mask;
                const bool pup = (pl & mask) != 0;
                for (int i = 0; i < n; ++i) {
                    const float keep = up ? cur[l][i + n] : cur[l][i];
                    const float psend = pup ? cur[pl][i] : cur[pl][i + n];   // what the partner sends
                    nxt[l][i] = keep + psend;
                }
            }
            std::memcpy(cur, nxt, sizeof(cur));
        };
        step(32, 4); step(16, 2); step(8, 1);
        for (int mask : {4, 2, 1}) {
            float nxt[64];
            for (int l = 0; l < 64; ++l) nxt[l] = cur[l][0] + cur[l ^ mask][0];
            for (int l = 0; l < 64; ++l) cur[l][0] = nxt[l];
        }
        for (int l = 0; l < 64; l += 8) {
            const int k = (l >> 3) & 7;       // k = r*2 + jj, r = l >> 4
            VJ_CHECK(std::fabs(cur[l][0] - want[k]) < 1e-4);
            VJ_CHECK((k >> 1) == (l >> 4));
        }
    }
    // INT8 -> FP16 magic: half bits 0x64XX with XX = b ^ 0x80, minus 1152.0 (exact).
    for (int b = -128; b < 128; ++b) {
        const uint16_t bits = static_cast<uint16_t>(0x6400u | ((static_cast<uint32_t>(b) & 0xFFu) ^ 0x80u));
        const float v = f16_to_f32(bits) - 1152.0f;
        VJ_CHECK(v == static_cast<float>(b));
        VJ_CHECK(h16(v) == v);
    }
    std::printf("[fa2] reduce-scatter lane map + int8->fp16 magic: OK\n");
}

struct KvRow { uint16_t sc[8]; int8_t q[256]; };

KvRow quant_kv(const float* x) {
    KvRow r;
    for (int b = 0; b < 8; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[b * 32 + i]));
        const float sc = amax * (1.0f / 127.0f);
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        r.sc[b] = f32_to_f16_rne(sc);
        for (int i = 0; i < 32; ++i) r.q[b * 32 + i] = static_cast<int8_t>(std::max(-127, std::min(127, rn_even(x[b * 32 + i] * inv))));
    }
    return r;
}

void quant_out(const float* y, int8_t* q, float* sc) {
    for (int b = 0; b < 8; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(y[b * 32 + i]));
        sc[b] = amax / 127.0f;
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        for (int i = 0; i < 32; ++i) q[b * 32 + i] = static_cast<int8_t>(std::max(-127, std::min(127, rn_even(y[b * 32 + i] * inv))));
    }
}

void test_fa2_softmax() {
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int L_max = 300;
    std::vector<KvRow> K(L_max), V(L_max);
    std::vector<float> tmp(256);
    double max_err_ref = 0.0, max_err_fa2 = 0.0, max_diff = 0.0, max_err_f32q = 0.0;
    uint64_t q_total = 0, q_mismatch = 0, q_off_by_more = 0, q_mismatch_f32q = 0;
    for (int t = 0; t < L_max; ++t) {
        for (auto& v : tmp) v = nd(rng) * 2.0f;
        K[t] = quant_kv(tmp.data());
        for (auto& v : tmp) v = nd(rng);
        V[t] = quant_kv(tmp.data());
    }
    for (int seqlen : {1, 31, 32, 33, 64, 100, 257, 300}) {
        for (int h = 0; h < 16; ++h) {
            float q[256], gate[256];
            for (int d = 0; d < 256; ++d) { q[d] = nd(rng) * 0.5f; gate[d] = nd(rng); }
            auto kval = [&](int t, int d) { return f16_to_f32(K[t].sc[d / 32]) * static_cast<float>(K[t].q[d]); };
            auto vval = [&](int t, int d) { return f16_to_f32(V[t].sc[d / 32]) * static_cast<float>(V[t].q[d]); };
            // (iii) double reference
            std::vector<double> s(seqlen);
            double mx = -1e300;
            for (int t = 0; t < seqlen; ++t) {
                double a = 0.0;
                for (int d = 0; d < 256; ++d) a += static_cast<double>(q[d]) * kval(t, d);
                s[t] = a * 0.0625;
                mx = std::max(mx, s[t]);
            }
            double l = 0.0;
            for (int t = 0; t < seqlen; ++t) { s[t] = std::exp(s[t] - mx); l += s[t]; }
            double o_ref[256];
            for (int d = 0; d < 256; ++d) {
                double a = 0.0;
                for (int t = 0; t < seqlen; ++t) a += s[t] * vval(t, d);
                o_ref[d] = a / l;
            }
            // (i) per-position online softmax (ornith_attn_flash_quant_kernel arithmetic, FP32)
            float o_i[256];
            {
                float m_run = -1e30f, l_run = 0.0f, o_d[256] = {};
                for (int t = 0; t < seqlen; ++t) {
                    float dot = 0.0f;
                    for (int d = 0; d < 256; ++d) dot += q[d] * kval(t, d);
                    const float score = dot * 0.0625f;
                    const float m_new = std::fmax(m_run, score);
                    const float alpha = std::exp(m_run - m_new), beta = std::exp(score - m_new);
                    l_run = l_run * alpha + beta;
                    m_run = m_new;
                    for (int d = 0; d < 256; ++d) o_d[d] = std::fma(beta, vval(t, d), o_d[d] * alpha);
                }
                for (int d = 0; d < 256; ++d) o_i[d] = o_d[d] / l_run;
            }
            // (ii) FA2 tile emulation: FP16 Q (vj_prefill_fa2_quant_kernel) or FP32 Q (..._f32q_kernel), exact INT8
            // K/V with FP32 block scales, per-lane partial * scale, 64-lane sum, tile-max online softmax, P in FP32,
            // O += (P * vscale) * v.
            auto fa2 = [&](bool f16q, float* o_out) {
                float qh[256];
                for (int d = 0; d < 256; ++d) qh[d] = f16q ? h16(q[d]) : q[d];
                float m_run = -1e30f, l_run = 0.0f, o[256] = {};
                for (int t0 = 0; t0 < seqlen; t0 += 32) {
                    float sc[32];
                    for (int j = 0; j < 32; ++j) {
                        const int t = std::min(t0 + j, seqlen - 1);
                        float acc = 0.0f;
                        for (int lane = 0; lane < 64; ++lane) {
                            const int d = lane * 4;
                            const float p01 = qh[d] * K[t].q[d] + qh[d + 1] * static_cast<float>(K[t].q[d + 1]);
                            const float p23 = qh[d + 2] * K[t].q[d + 2] + qh[d + 3] * static_cast<float>(K[t].q[d + 3]);
                            acc += (p01 + p23) * f16_to_f32(K[t].sc[d / 32]);
                        }
                        sc[j] = (t0 + j < seqlen) ? acc * 0.0625f : -INFINITY;
                    }
                    float mx2 = -INFINITY;
                    for (int j = 0; j < 32; ++j) mx2 = std::fmax(mx2, sc[j]);
                    const float m_new = std::fmax(m_run, mx2);
                    const float alpha = std::exp(m_run - m_new);
                    float ps = 0.0f, p[32];
                    for (int j = 0; j < 32; ++j) { p[j] = std::exp(sc[j] - m_new); ps += p[j]; }
                    l_run = std::fma(l_run, alpha, ps);
                    m_run = m_new;
                    for (int d = 0; d < 256; ++d) o[d] *= alpha;
                    for (int j = 0; j < 32; ++j) {
                        const int t = std::min(t0 + j, seqlen - 1);
                        for (int d = 0; d < 256; ++d)
                            o[d] = std::fma(p[j] * f16_to_f32(V[t].sc[d / 32]), static_cast<float>(V[t].q[d]), o[d]);
                    }
                }
                for (int d = 0; d < 256; ++d) o_out[d] = o[d] / l_run;
            };
            float o_ii[256], o_iv[256];
            fa2(true, o_ii);
            fa2(false, o_iv);
            double oscale = 1e-30;
            for (int d = 0; d < 256; ++d) oscale = std::max(oscale, std::fabs(o_ref[d]));
            float y_i[256], y_ii[256], y_iv[256];
            for (int d = 0; d < 256; ++d) {
                max_err_f32q = std::max(max_err_f32q, std::fabs(o_iv[d] - o_ref[d]) / oscale);
                y_iv[d] = o_iv[d] * (1.0f / (1.0f + std::exp(-gate[d])));
                max_err_ref = std::max(max_err_ref, std::fabs(o_i[d] - o_ref[d]) / oscale);
                max_err_fa2 = std::max(max_err_fa2, std::fabs(o_ii[d] - o_ref[d]) / oscale);
                max_diff = std::max(max_diff, std::fabs(static_cast<double>(o_ii[d]) - o_i[d]) / oscale);
                const float sg = 1.0f / (1.0f + std::exp(-gate[d]));
                y_i[d] = o_i[d] * sg;
                y_ii[d] = o_ii[d] * sg;
            }
            int8_t qa[256], qb[256], qc[256];
            float sa[8], sb[8], sc4[8];
            quant_out(y_i, qa, sa);
            quant_out(y_ii, qb, sb);
            quant_out(y_iv, qc, sc4);
            for (int d = 0; d < 256; ++d) {
                if (qa[d] != qc[d]) ++q_mismatch_f32q;
                ++q_total;
                if (qa[d] != qb[d]) ++q_mismatch;
                if (std::abs(qa[d] - qb[d]) > 1) ++q_off_by_more;
            }
        }
    }
    std::printf("[fa2] max |o - o_double| / max|o|: per-position(ref kernel) %.3e, FA2 tiles %.3e, FA2 vs ref %.3e\n",
                max_err_ref, max_err_fa2, max_diff);
    std::printf("[fa2] int8 outputs: %llu / %llu differ from the ref-kernel arithmetic (all by <= 1 LSB: %s)\n",
                static_cast<unsigned long long>(q_mismatch), static_cast<unsigned long long>(q_total),
                q_off_by_more == 0 ? "yes" : "no");
    std::printf("[fa2] FP32-Q variant: max rel err %.3e, int8 outputs differing from the ref-kernel arithmetic: %llu / %llu\n",
                max_err_f32q, static_cast<unsigned long long>(q_mismatch_f32q), static_cast<unsigned long long>(q_total));
    VJ_CHECK(max_err_fa2 < 4e-3);       // FP16 Q is the dominant term
    VJ_CHECK(max_err_f32q < 1e-5);
    VJ_CHECK(q_off_by_more == 0);
}

// =============================================================================================================
// 2. Tiled W4A8 GEMM
// =============================================================================================================
struct Q4Mat {
    uint32_t rows = 0, cols = 0, stride = 0;
    std::vector<uint8_t> qs;     // rows x stride bytes, 16 B per block (low nibble = elem i, high = elem 16+i)
    std::vector<uint32_t> dm;    // rows x nb
};
struct Act {
    uint32_t M = 0, cols = 0;
    std::vector<int8_t> q;
    std::vector<float> d, s;     // per (token, block)
};

int32_t block_dp(const Q4Mat& w, const Act& x, uint32_t row, uint32_t m, uint32_t b) {
    int32_t dp = 0;
    const uint8_t* wb = &w.qs[static_cast<size_t>(row) * w.stride + b * 16u];
    const int8_t* xb = &x.q[static_cast<size_t>(m) * x.cols + b * 32u];
    for (int i = 0; i < 16; ++i) dp += (wb[i] & 0xF) * xb[i] + (wb[i] >> 4) * xb[16 + i];
    return dp;
}
float wd_of(const Q4Mat& w, uint32_t row, uint32_t b) { return f16_to_f32(static_cast<uint16_t>(w.dm[row * (w.cols / 32) + b] & 0xFFFF)); }
float wm_of(const Q4Mat& w, uint32_t row, uint32_t b) { return f16_to_f32(static_cast<uint16_t>(w.dm[row * (w.cols / 32) + b] >> 16)); }

float tcc_order(const Q4Mat& w, const Act& x, uint32_t row, uint32_t m) {
    const uint32_t nb = w.cols / 32;
    float lane[32] = {};
    for (uint32_t b0 = 0; b0 < nb; b0 += 32) {
        for (uint32_t l = 0; l < 32 && b0 + l < nb; ++l) {
            const uint32_t b = b0 + l;
            const float dp = static_cast<float>(block_dp(w, x, row, m, b));
            const float wd = wd_of(w, row, b), wm = wm_of(w, row, b);
            const float xd = x.d[m * nb + b], xs = x.s[m * nb + b];
            lane[l] = std::fma(wd * xd, dp, std::fma(wm, xs, lane[l]));
        }
    }
    for (uint32_t sft : {16u, 8u, 4u, 2u, 1u}) {
        float nx[32];
        for (uint32_t l = 0; l < 32; ++l) nx[l] = lane[l] + lane[l ^ sft];
        std::memcpy(lane, nx, sizeof(nx));
    }
    return lane[0];
}

float seq_order(const Q4Mat& w, const Act& x, uint32_t row, uint32_t m) {
    const uint32_t nb = w.cols / 32;
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; ++b) {
        const float dp = static_cast<float>(block_dp(w, x, row, m, b));
        acc = std::fma(wd_of(w, row, b) * x.d[m * nb + b], dp, std::fma(wm_of(w, row, b), x.s[m * nb + b], acc));
    }
    return acc;
}

double dbl_ref(const Q4Mat& w, const Act& x, uint32_t row, uint32_t m) {
    const uint32_t nb = w.cols / 32;
    double acc = 0.0;
    for (uint32_t b = 0; b < nb; ++b)
        acc += static_cast<double>(wd_of(w, row, b)) * x.d[m * nb + b] * block_dp(w, x, row, m, b) +
               static_cast<double>(wm_of(w, row, b)) * x.s[m * nb + b];
    return acc;
}

int8_t sdot_byte(uint32_t v, int k) { return static_cast<int8_t>((v >> (8 * k)) & 0xFF); }
int32_t sdot4(uint32_t a, uint32_t b, int32_t c) {
    for (int k = 0; k < 4; ++k) c += sdot_byte(a, k) * sdot_byte(b, k);
    return c;
}

// Faithful emulation of gemm_q4a8_tile<BN, NT>: per WG LDS arrays, loaders, K-tiles, thread mapping, stores.
void emulate_tiled(const Q4Mat& w, const Act& x, uint32_t BN, uint32_t NT, std::vector<float>& y) {
    const uint32_t RSW = 17;   // dwords per LDS row
    const uint32_t NTY = NT / 16;
    const uint32_t nb = w.cols / 32, nkt = w.cols / 64;
    y.assign(static_cast<size_t>(x.M) * w.rows, std::nanf(""));
    std::vector<uint32_t> smw(64 * RSW), smx(BN * RSW);
    std::vector<float> wdm(64 * 2 * 2), xds(BN * 2 * 2);
    for (uint32_t bx = 0; bx * 64 < w.rows; ++bx) {
        for (uint32_t by = 0; by * BN < x.M; ++by) {
            const uint32_t row0 = bx * 64, m0 = by * BN;
            std::vector<float> acc(static_cast<size_t>(NT) * 16, 0.0f);
            for (uint32_t kt = 0; kt < nkt; ++kt) {
                // store_stage for every thread
                for (uint32_t tid = 0; tid < NT; ++tid) {
                    {
                        const uint32_t t = tid >> 2, q = tid & 3;
                        const uint32_t m = std::min(m0 + t, x.M - 1);
                        const int8_t* src = &x.q[static_cast<size_t>(m) * x.cols + kt * 64 + q * 16];
                        for (int k = 0; k < 4; ++k) {
                            uint32_t v;
                            std::memcpy(&v, src + 4 * k, 4);
                            smx[t * RSW + q * 4 + k] = v;
                        }
                    }
                    if (tid < 128) {
                        const uint32_t r = tid >> 1, b = tid & 1;
                        const uint32_t row = std::min(row0 + r, w.rows - 1);
                        uint32_t wq[4];
                        std::memcpy(wq, &w.qs[static_cast<size_t>(row) * w.stride + (kt * 2 + b) * 16], 16);
                        for (int k = 0; k < 4; ++k) {
                            smw[r * RSW + b * 8 + k] = wq[k] & 0x0F0F0F0Fu;
                            smw[r * RSW + b * 8 + 4 + k] = (wq[k] >> 4) & 0x0F0F0F0Fu;
                        }
                        wdm[(r * 2 + b) * 2 + 0] = wd_of(w, row, kt * 2 + b);
                        wdm[(r * 2 + b) * 2 + 1] = wm_of(w, row, kt * 2 + b);
                    } else if (tid < 128 + 2 * BN) {
                        const uint32_t u = tid - 128, t = u >> 1, b = u & 1;
                        const uint32_t m = std::min(m0 + t, x.M - 1);
                        xds[u * 2 + 0] = x.d[m * nb + kt * 2 + b];
                        xds[u * 2 + 1] = x.s[m * nb + kt * 2 + b];
                    }
                }
                // MACs
                for (uint32_t tid = 0; tid < NT; ++tid) {
                    const uint32_t tx = tid & 15, ty = tid >> 4;
                    for (uint32_t b = 0; b < 2; ++b) {
                        int32_t dp[4][4] = {};
                        for (uint32_t kk = 0; kk < 8; ++kk)
                            for (uint32_t i = 0; i < 4; ++i)
                                for (uint32_t j = 0; j < 4; ++j)
                                    dp[i][j] = sdot4(smw[(tx + 16 * i) * RSW + b * 8 + kk], smx[(ty + NTY * j) * RSW + b * 8 + kk], dp[i][j]);
                        for (uint32_t j = 0; j < 4; ++j) {
                            const float xd = xds[((ty + NTY * j) * 2 + b) * 2], xs = xds[((ty + NTY * j) * 2 + b) * 2 + 1];
                            for (uint32_t i = 0; i < 4; ++i) {
                                const float wd = wdm[((tx + 16 * i) * 2 + b) * 2], wm = wdm[((tx + 16 * i) * 2 + b) * 2 + 1];
                                float& a = acc[tid * 16 + i * 4 + j];
                                a = std::fma(wd * xd, static_cast<float>(dp[i][j]), std::fma(wm, xs, a));
                            }
                        }
                    }
                }
            }
            for (uint32_t tid = 0; tid < NT; ++tid) {
                const uint32_t tx = tid & 15, ty = tid >> 4;
                for (uint32_t j = 0; j < 4; ++j) {
                    const uint32_t m = m0 + ty + NTY * j;
                    if (m >= x.M) continue;
                    for (uint32_t i = 0; i < 4; ++i) {
                        const uint32_t row = row0 + tx + 16 * i;
                        if (row < w.rows) y[static_cast<size_t>(m) * w.rows + row] = acc[tid * 16 + i * 4 + j];
                    }
                }
            }
        }
    }
}

void make_q4(Q4Mat& w, uint32_t rows, uint32_t cols, std::mt19937& rng) {
    w.rows = rows; w.cols = cols; w.stride = cols / 2 + 16;     // padded stride like the TCC odd-256B layout
    w.qs.resize(static_cast<size_t>(rows) * w.stride);
    for (auto& b : w.qs) b = static_cast<uint8_t>(rng());
    std::uniform_real_distribution<float> ud(0.001f, 0.02f), um(-0.15f, 0.0f);
    w.dm.resize(static_cast<size_t>(rows) * (cols / 32));
    for (auto& v : w.dm) v = f32_to_f16_rne(ud(rng)) | (static_cast<uint32_t>(f32_to_f16_rne(um(rng))) << 16);
}

void make_act(Act& x, uint32_t M, uint32_t cols, std::mt19937& rng) {
    x.M = M; x.cols = cols;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> v(cols);
    x.q.resize(static_cast<size_t>(M) * cols);
    x.d.resize(static_cast<size_t>(M) * cols / 32);
    x.s.resize(static_cast<size_t>(M) * cols / 32);
    for (uint32_t m = 0; m < M; ++m) {
        for (auto& e : v) e = nd(rng) * ((rng() % 50 == 0) ? 8.0f : 1.0f);
        for (uint32_t b = 0; b < cols / 32; ++b) {
            float amax = 0.0f, sum = 0.0f;
            for (int i = 0; i < 32; ++i) { amax = std::fmax(amax, std::fabs(v[b * 32 + i])); sum += v[b * 32 + i]; }
            const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
            for (int i = 0; i < 32; ++i) x.q[static_cast<size_t>(m) * cols + b * 32 + i] =
                static_cast<int8_t>(std::max(-127, std::min(127, rn_even(v[b * 32 + i] * inv))));
            x.d[m * (cols / 32) + b] = amax / 127.0f;
            x.s[m * (cols / 32) + b] = sum;
        }
    }
}

void test_gemm() {
    std::mt19937 rng(21);
    struct Shape { uint32_t rows, cols, M, BN, NT; };
    for (const Shape sh : {Shape{200, 512, 70, 64, 256}, Shape{130, 3072, 50, 64, 256}, Shape{96, 4096, 140, 128, 512}}) {
        Q4Mat w; Act x;
        make_q4(w, sh.rows, sh.cols, rng);
        make_act(x, sh.M, sh.cols, rng);
        std::vector<float> y;
        emulate_tiled(w, x, sh.BN, sh.NT, y);
        double max_rel_tcc = 0.0, max_rel_dbl_tile = 0.0, max_rel_dbl_tcc = 0.0;
        uint64_t bit_equal_tcc = 0;
        for (uint32_t m = 0; m < sh.M; ++m) {
            for (uint32_t r = 0; r < sh.rows; ++r) {
                const float t = y[static_cast<size_t>(m) * sh.rows + r];
                const float s = seq_order(w, x, r, m);
                VJ_CHECK(std::memcmp(&t, &s, 4) == 0);      // emulation == the documented sequential order
                const float c = tcc_order(w, x, r, m);
                const double dref = dbl_ref(w, x, r, m);
                double mag = 0.0;                            // scale: sum of |block terms|
                for (uint32_t b = 0; b < sh.cols / 32; ++b)
                    mag += std::fabs(static_cast<double>(wd_of(w, r, b)) * x.d[m * (sh.cols / 32) + b] * block_dp(w, x, r, m, b)) +
                           std::fabs(static_cast<double>(wm_of(w, r, b)) * x.s[m * (sh.cols / 32) + b]);
                if (std::memcmp(&t, &c, 4) == 0) ++bit_equal_tcc;
                max_rel_tcc = std::max(max_rel_tcc, std::fabs(static_cast<double>(t) - c) / mag);
                max_rel_dbl_tile = std::max(max_rel_dbl_tile, std::fabs(t - dref) / mag);
                max_rel_dbl_tcc = std::max(max_rel_dbl_tcc, std::fabs(c - dref) / mag);
            }
        }
        std::printf("[gemm %ux%u M=%u tile %ux64] tile==seq order bitwise; vs tcc: %.1f%% bit-equal, max |diff|/sum|terms| %.2e; "
                    "vs double: tile %.2e tcc %.2e\n",
                    sh.rows, sh.cols, sh.M, sh.BN, 100.0 * bit_equal_tcc / (static_cast<double>(sh.M) * sh.rows),
                    max_rel_tcc, max_rel_dbl_tile, max_rel_dbl_tcc);
        VJ_CHECK(max_rel_tcc < 1e-5);
        VJ_CHECK(max_rel_dbl_tile < 1e-5);
    }
}

// =============================================================================================================
// 3. IQ tiles
// =============================================================================================================
uint32_t apply_4signs(uint32_t mag, uint32_t sign4) {
    const uint32_t add = (sign4 * 0x00204081u) & 0x01010101u;
    const uint32_t xorm = (add << 8) - add;
    return (mag ^ xorm) + add;
}
uint32_t plain_4signs(uint32_t mag, uint32_t sign4) {
    uint32_t out = 0;
    for (int k = 0; k < 4; ++k) {
        int v = static_cast<int>((mag >> (8 * k)) & 0xFF);
        if (sign4 & (1u << k)) v = -v;
        out |= (static_cast<uint32_t>(v) & 0xFFu) << (8 * k);
    }
    return out;
}

void test_swar_signs() {
    for (uint32_t i = 0; i < 1024; ++i)
        for (uint32_t half = 0; half < 2; ++half)
            for (uint32_t s = 0; s < 16; ++s) {
                const uint32_t mag = static_cast<uint32_t>(gt::iq2s_grid[i] >> (32 * half));
                VJ_CHECK(apply_4signs(mag, s) == plain_4signs(mag, s));
            }
    for (uint32_t i = 0; i < 256; ++i)
        for (uint32_t s = 0; s < 16; ++s) VJ_CHECK(apply_4signs(gt::iq3xxs_grid[i], s) == plain_4signs(gt::iq3xxs_grid[i], s));
    std::printf("[iq] SWAR signs == plain negation on all iq2s (1024x2x16) and iq3xxs (256x16) grid words\n");
}

// ggml dequantize_row_iq2_s for one 256-element block, returning the signed grid value and the scale per element.
void ggml_iq2s_block(const uint8_t* blk, int8_t* val, float* scale) {
    uint16_t dh;
    std::memcpy(&dh, blk, 2);
    const float d = f16_to_f32(dh);
    const uint8_t* qs = blk + 2;
    const uint8_t* signs = qs + 32;
    const uint8_t* qh = blk + 66;
    const uint8_t* scales = blk + 74;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const float db[2] = {d * (0.5f + (scales[ib32] & 0xf)) * 0.25f, d * (0.5f + (scales[ib32] >> 4)) * 0.25f};
        for (int l = 0; l < 4; ++l) {
            const uint64_t g = gt::iq2s_grid[qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)];
            for (int j = 0; j < 8; ++j) {
                const int v = static_cast<int>((g >> (8 * j)) & 0xFF) * ((signs[l] & gt::kmask_iq2xs[j]) ? -1 : 1);
                val[ib32 * 32 + l * 8 + j] = static_cast<int8_t>(v);
                scale[ib32 * 32 + l * 8 + j] = db[l / 2];
            }
        }
        qs += 4;
        signs += 4;
    }
}

// Loader's merged-row packing (models/ornith/ornith_gpu.cpp pack_row), 12 blocks per row (D = 3072).
void pack_iq2s_row(const uint8_t* g_row, const uint8_t* u_row, uint32_t* dst) {
    uint32_t* qp = dst;
    uint32_t* hs2 = dst + 96 * 4;
    uint32_t* d2 = dst + 120 * 4;
    for (uint32_t b = 0; b < 12; ++b) {
        const uint8_t* gb = g_row + b * 82;
        const uint8_t* ub = u_row + b * 82;
        uint16_t gd = 0, ud = 0;
        std::memcpy(&gd, gb, 2);
        std::memcpy(&ud, ub, 2);
        d2[b] = gd | (static_cast<uint32_t>(ud) << 16);
        for (uint32_t sb = 0; sb < 8; ++sb) {
            const uint32_t ib = b * 8 + sb;
            const uint8_t* gq = gb + 2 + sb * 4; const uint8_t* gs = gb + 34 + sb * 4;
            const uint8_t* uq = ub + 2 + sb * 4; const uint8_t* us = ub + 34 + sb * 4;
            qp[ib * 4 + 0] = gq[0] | (gs[0] << 8) | (gq[1] << 16) | (static_cast<uint32_t>(gs[1]) << 24);
            qp[ib * 4 + 1] = gq[2] | (gs[2] << 8) | (gq[3] << 16) | (static_cast<uint32_t>(gs[3]) << 24);
            qp[ib * 4 + 2] = uq[0] | (us[0] << 8) | (uq[1] << 16) | (static_cast<uint32_t>(us[1]) << 24);
            qp[ib * 4 + 3] = uq[2] | (us[2] << 8) | (uq[3] << 16) | (static_cast<uint32_t>(us[3]) << 24);
            hs2[ib] = gb[66 + sb] | (gb[74 + sb] << 8) | (ub[66 + sb] << 16) | (static_cast<uint32_t>(ub[74 + sb]) << 24);
        }
    }
}

// vj_prefill_moe_gu_iq2s_tiled_kernel decode of one (row, ib) for wave `part`: 4 INT8 dwords + scale.
void tile_decode(const uint32_t* row, uint32_t ib, uint32_t part, uint32_t out[4], float* sc) {
    const uint32_t s_q = row[ib * 4 + part], s_hs = row[96 * 4 + ib], s_d = row[120 * 4 + (ib >> 3)];
    const uint32_t mat = part >> 1, half = part & 1;
    const uint32_t qh = (s_hs >> (16 * mat)) & 0xFF;
    const uint32_t scb = (s_hs >> (16 * mat + 8)) & 0xFF;
    const float dd = f16_to_f32(static_cast<uint16_t>(s_d >> (16 * mat))) * 0.25f;
    const uint32_t sh0 = half ? 4 : 8;
    const uint64_t g0 = gt::iq2s_grid[(s_q & 0xFF) | ((qh << sh0) & 0x300)];
    const uint64_t g1 = gt::iq2s_grid[((s_q >> 16) & 0xFF) | ((qh << (sh0 - 2)) & 0x300)];
    out[0] = apply_4signs(static_cast<uint32_t>(g0), (s_q >> 8) & 0xF);
    out[1] = apply_4signs(static_cast<uint32_t>(g0 >> 32), (s_q >> 12) & 0xF);
    out[2] = apply_4signs(static_cast<uint32_t>(g1), (s_q >> 24) & 0xF);
    out[3] = apply_4signs(static_cast<uint32_t>(g1 >> 32), s_q >> 28);
    *sc = dd * (0.5f + static_cast<float>(half ? (scb >> 4) : (scb & 0xF)));
}

void random_iq2s_block(uint8_t* blk, std::mt19937& rng) {
    for (int i = 0; i < 82; ++i) blk[i] = static_cast<uint8_t>(rng());
    const uint16_t d = f32_to_f16_rne(0.001f + static_cast<float>(rng() % 1000) * 1e-5f);
    std::memcpy(blk, &d, 2);
}

void test_iq2s_tiles() {
    std::mt19937 rng(31);
    constexpr uint32_t D = 3072, ROWS = 64, T = 13;
    std::vector<uint8_t> graw(ROWS * 12 * 82), uraw(ROWS * 12 * 82);
    for (uint32_t i = 0; i < ROWS * 12; ++i) { random_iq2s_block(&graw[i * 82], rng); random_iq2s_block(&uraw[i * 82], rng); }
    std::vector<uint32_t> packed(ROWS * 123 * 4);
    for (uint32_t r = 0; r < ROWS; ++r) pack_iq2s_row(&graw[r * 12 * 82], &uraw[r * 12 * 82], &packed[r * 123 * 4]);
    // Decode equality vs ggml for every element.
    std::vector<int8_t> gval(ROWS * D), uval(ROWS * D);
    std::vector<float> gsc(ROWS * D), usc(ROWS * D);
    for (uint32_t r = 0; r < ROWS; ++r)
        for (uint32_t b = 0; b < 12; ++b) {
            ggml_iq2s_block(&graw[(r * 12 + b) * 82], &gval[r * D + b * 256], &gsc[r * D + b * 256]);
            ggml_iq2s_block(&uraw[(r * 12 + b) * 82], &uval[r * D + b * 256], &usc[r * D + b * 256]);
        }
    double max_sc_rel = 0.0;
    for (uint32_t r = 0; r < ROWS; ++r)
        for (uint32_t ib = 0; ib < 96; ++ib)
            for (uint32_t part = 0; part < 4; ++part) {
                uint32_t w[4];
                float sc;
                tile_decode(&packed[r * 123 * 4], ib, part, w, &sc);
                const bool up = part >= 2;
                const uint32_t e0 = ib * 32 + (part & 1) * 16;
                for (int k = 0; k < 16; ++k) {
                    const int8_t v = static_cast<int8_t>((w[k / 4] >> (8 * (k % 4))) & 0xFF);
                    VJ_CHECK(v == (up ? uval : gval)[r * D + e0 + k]);
                    const float ref = (up ? usc : gsc)[r * D + e0 + k];
                    max_sc_rel = std::max(max_sc_rel, std::fabs(static_cast<double>(sc) - ref) / ref);
                }
            }
    std::printf("[iq2s] tile decode == ggml dequantize_row_iq2_s values for 64 rows x 3072 (gate+up); scale max rel %.1e "
                "(ggml multiplies d*(0.5+s)*0.25, kernels (d*0.25)*(0.5+s))\n", max_sc_rel);
    VJ_CHECK(max_sc_rel < 2.5e-7);

    // Dequant-once tile (sequential ib order, half 1 then half 0) vs the grouped kernel's 16-lane order.
    Act x;
    make_act(x, T, D, rng);
    double max_rel = 0.0;
    uint64_t q_diff = 0, q_tot = 0;
    for (uint32_t t = 0; t < T; ++t) {
        float yt[64], yg[64];
        for (uint32_t r = 0; r < ROWS; ++r) {
            const uint32_t* row = &packed[r * 123 * 4];
            auto dot16 = [&](const uint32_t w[4], uint32_t ib, uint32_t half) {
                int32_t dp = 0;
                for (int k = 0; k < 4; ++k) {
                    uint32_t xv;
                    std::memcpy(&xv, &x.q[t * D + ib * 32 + half * 16 + 4 * k], 4);
                    dp = sdot4(w[k], xv, dp);
                }
                return dp;
            };
            // tile order
            float ag = 0.0f, au = 0.0f;
            for (uint32_t ib = 0; ib < 96; ++ib) {
                const float xd = x.d[t * 96 + ib];
                for (uint32_t hh = 0; hh < 2; ++hh) {
                    const uint32_t h = 1 - hh;
                    uint32_t wg[4], wu[4];
                    float sg, su;
                    tile_decode(row, ib, h, wg, &sg);
                    tile_decode(row, ib, 2 + h, wu, &su);
                    ag = std::fma(sg * xd, static_cast<float>(dot16(wg, ib, h)), ag);
                    au = std::fma(su * xd, static_cast<float>(dot16(wu, ib, h)), au);
                }
            }
            yt[r] = (ag / (1.0f + std::exp(-ag))) * au;
            // grouped kernel order: 16 lanes, ib = it*16 + lane, fmaf(gd0*xd, dp0, fmaf(gd1*xd, dp1, acc)), xor tree
            float lg[16] = {}, lu[16] = {};
            for (uint32_t it = 0; it < 6; ++it)
                for (uint32_t ln = 0; ln < 16; ++ln) {
                    const uint32_t ib = it * 16 + ln;
                    const float xd = x.d[t * 96 + ib];
                    uint32_t w0[4], w1[4], v0[4], v1[4];
                    float g0, g1, u0, u1;
                    tile_decode(row, ib, 0, w0, &g0); tile_decode(row, ib, 1, w1, &g1);
                    tile_decode(row, ib, 2, v0, &u0); tile_decode(row, ib, 3, v1, &u1);
                    lg[ln] = std::fma(g0 * xd, static_cast<float>(dot16(w0, ib, 0)), std::fma(g1 * xd, static_cast<float>(dot16(w1, ib, 1)), lg[ln]));
                    lu[ln] = std::fma(u0 * xd, static_cast<float>(dot16(v0, ib, 0)), std::fma(u1 * xd, static_cast<float>(dot16(v1, ib, 1)), lu[ln]));
                }
            for (uint32_t sft : {8u, 4u, 2u, 1u}) {
                float ng[16], nu[16];
                for (uint32_t l = 0; l < 16; ++l) { ng[l] = lg[l] + lg[l ^ sft]; nu[l] = lu[l] + lu[l ^ sft]; }
                std::memcpy(lg, ng, sizeof(ng));
                std::memcpy(lu, nu, sizeof(nu));
            }
            yg[r] = (lg[0] / (1.0f + std::exp(-lg[0]))) * lu[0];
            max_rel = std::max(max_rel, std::fabs(static_cast<double>(yt[r]) - yg[r]) / std::max(1e-6, std::fabs(static_cast<double>(yg[r]))));
        }
        for (uint32_t bq = 0; bq < 2; ++bq) {
            float at = 0.0f, ag2 = 0.0f;
            for (int i = 0; i < 32; ++i) { at = std::fmax(at, std::fabs(yt[bq * 32 + i])); ag2 = std::fmax(ag2, std::fabs(yg[bq * 32 + i])); }
            for (int i = 0; i < 32; ++i) {
                const int qa = std::max(-127, std::min(127, rn_even(yt[bq * 32 + i] * (127.0f / at))));
                const int qb = std::max(-127, std::min(127, rn_even(yg[bq * 32 + i] * (127.0f / ag2))));
                ++q_tot;
                if (qa != qb) ++q_diff;
                VJ_CHECK(std::abs(qa - qb) <= 1);
            }
        }
    }
    std::printf("[iq2s] dequant-once tile vs grouped-kernel order: max rel |dy| %.2e, INT8 mid differs in %llu / %llu (<= 1 LSB)\n",
                max_rel, static_cast<unsigned long long>(q_diff), static_cast<unsigned long long>(q_tot));
}

// ggml dequantize_row_iq3_xxs vs the split-layout SWAR decode used by the IQ3_XXS kernels.
void test_iq3xxs() {
    std::mt19937 rng(41);
    for (int trial = 0; trial < 2000; ++trial) {
        uint8_t blk[98];
        for (auto& b : blk) b = static_cast<uint8_t>(rng());
        const uint16_t d = f32_to_f16_rne(0.01f);
        std::memcpy(blk, &d, 2);
        const uint8_t* qs = blk + 2;
        const uint8_t* sas = blk + 2 + 64;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            uint32_t aux;
            std::memcpy(&aux, sas + 4 * ib32, 4);
            // ggml
            int8_t ref[32];
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = gt::ksigns_iq2xs[(aux >> (7 * l)) & 127];
                const uint32_t g1 = gt::iq3xxs_grid[qs[ib32 * 8 + 2 * l]], g2 = gt::iq3xxs_grid[qs[ib32 * 8 + 2 * l + 1]];
                for (int j = 0; j < 4; ++j) {
                    ref[l * 8 + j] = static_cast<int8_t>(static_cast<int>((g1 >> (8 * j)) & 0xFF) * ((signs & gt::kmask_iq2xs[j]) ? -1 : 1));
                    ref[l * 8 + 4 + j] = static_cast<int8_t>(static_cast<int>((g2 >> (8 * j)) & 0xFF) * ((signs & gt::kmask_iq2xs[j + 4]) ? -1 : 1));
                }
            }
            // kernel: q2 = 8 grid bytes (uint2), ga = aux; s_k = 7 bits | parity << 7
            uint32_t q2x, q2y;
            std::memcpy(&q2x, qs + ib32 * 8, 4);
            std::memcpy(&q2y, qs + ib32 * 8 + 4, 4);
            uint32_t s[4];
            for (int k = 0; k < 4; ++k) {
                s[k] = (aux >> (7 * k)) & 127u;
                s[k] |= (static_cast<uint32_t>(__builtin_popcount(s[k])) & 1u) << 7;
            }
            const uint32_t w[8] = {
                apply_4signs(gt::iq3xxs_grid[q2x & 0xFF], s[0] & 0xF), apply_4signs(gt::iq3xxs_grid[(q2x >> 8) & 0xFF], s[0] >> 4),
                apply_4signs(gt::iq3xxs_grid[(q2x >> 16) & 0xFF], s[1] & 0xF), apply_4signs(gt::iq3xxs_grid[q2x >> 24], s[1] >> 4),
                apply_4signs(gt::iq3xxs_grid[q2y & 0xFF], s[2] & 0xF), apply_4signs(gt::iq3xxs_grid[(q2y >> 8) & 0xFF], s[2] >> 4),
                apply_4signs(gt::iq3xxs_grid[(q2y >> 16) & 0xFF], s[3] & 0xF), apply_4signs(gt::iq3xxs_grid[q2y >> 24], s[3] >> 4)};
            for (int k = 0; k < 32; ++k) VJ_CHECK(static_cast<int8_t>((w[k / 4] >> (8 * (k % 4))) & 0xFF) == ref[k]);
        }
    }
    std::printf("[iq3xxs] split-layout SWAR decode == ggml dequantize_row_iq3_xxs on 2000 random blocks\n");
}

// =============================================================================================================
// 4. INT8 AllReduce wire bound (quick; the full test lives in test_prefill_planner)
// =============================================================================================================
void test_ar_bound() {
    std::mt19937 rng(51);
    std::normal_distribution<float> nd(0.0f, 3.0f);
    std::vector<float> a(3072), b(3072), sc(96);
    std::vector<int8_t> q(3072);
    for (int trial = 0; trial < 50; ++trial) {
        for (auto& v : a) v = nd(rng);
        vegajet::prefill::ar_q8_pack_row(a.data(), 3072, q.data(), sc.data());
        vegajet::prefill::ar_q8_unpack_row(q.data(), sc.data(), 3072, b.data());
        for (int blk = 0; blk < 96; ++blk) {
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(a[blk * 32 + i]));
            for (int i = 0; i < 32; ++i)
                VJ_CHECK(std::fabs(a[blk * 32 + i] - b[blk * 32 + i]) <= vegajet::prefill::ar_q8_err_bound(amax));
        }
    }
    std::printf("[ar] INT8 block-32 pack/unpack within bound\n");
}


// =============================================================================================================
// 5. GDN prefill: chunked WY / Neumann (vj_prefill_gdn_wy_split_kernel order, C = 16) and the sequential Split-V
//    order (vj_prefill_gdn_splitv_batch_kernel = kernel 5b arithmetic) vs a double-precision recurrence.
// =============================================================================================================
void test_gdn() {
    constexpr int DK = 128, DV = 128, CH = 16;
    std::mt19937 rng(61);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ug(0.90f, 0.999f), ub(0.05f, 0.95f);
    for (int Mtok : {1, 15, 16, 17, 50}) {
        std::vector<float> q(Mtok * DK), k(Mtok * DK), v(Mtok * DV), g(Mtok), beta(Mtok);
        for (int t = 0; t < Mtok; ++t) {
            float nq = 0.0f, nk2 = 0.0f;
            for (int i = 0; i < DK; ++i) { q[t * DK + i] = nd(rng); k[t * DK + i] = nd(rng); nq += q[t * DK + i] * q[t * DK + i]; nk2 += k[t * DK + i] * k[t * DK + i]; }
            for (int i = 0; i < DK; ++i) { q[t * DK + i] /= std::sqrt(nq) * std::sqrt(128.0f); k[t * DK + i] /= std::sqrt(nk2); }
            for (int j = 0; j < DV; ++j) v[t * DV + j] = nd(rng);
            g[t] = ug(rng); beta[t] = ub(rng);
        }
        std::vector<float> S0(DK * DV);
        for (auto& x : S0) x = nd(rng) * 0.1f;
        // double reference
        std::vector<double> Sd(S0.begin(), S0.end()), fod(Mtok * DV);
        for (int t = 0; t < Mtok; ++t) {
            for (int j = 0; j < DV; ++j) {
                double pred = 0.0;
                for (int i = 0; i < DK; ++i) pred += Sd[i * DV + j] * k[t * DK + i];
                const double dv = beta[t] * (v[t * DV + j] - g[t] * pred);
                double o = 0.0;
                for (int i = 0; i < DK; ++i) { Sd[i * DV + j] = g[t] * Sd[i * DV + j] + k[t * DK + i] * dv; o += Sd[i * DV + j] * q[t * DK + i]; }
                fod[t * DV + j] = o;
            }
        }
        // Split-V / 5b order (float)
        std::vector<float> Sb(S0), fob(Mtok * DV);
        for (int t = 0; t < Mtok; ++t) {
            for (int j = 0; j < DV; ++j) {
                float pg[8], og[8];
                for (int rg = 0; rg < 8; ++rg) { float p = 0.0f; for (int r = 0; r < 16; ++r) p = std::fma(Sb[(rg * 16 + r) * DV + j], k[t * DK + rg * 16 + r], p); pg[rg] = p; }
                const float pred = pg[0] + pg[1] + pg[2] + pg[3] + pg[4] + pg[5] + pg[6] + pg[7];
                const float dv = beta[t] * (v[t * DV + j] - g[t] * pred);
                for (int rg = 0; rg < 8; ++rg) {
                    float o = 0.0f;
                    for (int r = 0; r < 16; ++r) {
                        float& sr = Sb[(rg * 16 + r) * DV + j];
                        sr = std::fma(k[t * DK + rg * 16 + r], dv, g[t] * sr);
                        o = std::fma(sr, q[t * DK + rg * 16 + r], o);
                    }
                    og[rg] = o;
                }
                fob[t * DV + j] = og[0] + og[1] + og[2] + og[3] + og[4] + og[5] + og[6] + og[7];
            }
        }
        // Chunked WY (float), kernel order
        std::vector<float> Sa(S0), foa(Mtok * DV);
        for (int m0 = 0; m0 < Mtok; m0 += CH) {
            const int clen = std::min(CH, Mtok - m0);
            float cg[CH], P[CH * CH], T[CH * CH], QK[CH * CH], tmp[CH * CH];
            float acc = 1.0f;
            for (int t = 0; t < CH; ++t) { if (t < clen) acc *= g[m0 + t]; cg[t] = acc; }
            for (int r = 0; r < CH; ++r)
                for (int c = 0; c < CH; ++c) {
                    const bool valid = r < clen && c < clen;
                    const float ratio = (valid && r >= c) ? cg[r] / std::fmax(cg[c], 1e-30f) : 0.0f;
                    float kk = 0.0f, qk = 0.0f;
                    if (valid && r >= c)
                        for (int d = 0; d < DK; ++d) {
                            if (r > c) kk = std::fma(k[(m0 + r) * DK + d], k[(m0 + c) * DK + d], kk);
                            qk = std::fma(q[(m0 + r) * DK + d], k[(m0 + c) * DK + d], qk);
                        }
                    const float a = (valid && r > c) ? (-beta[m0 + r] * ratio * kk) : 0.0f;
                    P[r * CH + c] = a; T[r * CH + c] = (r == c) ? 1.0f : a; QK[r * CH + c] = (valid && r >= c) ? ratio * qk : 0.0f;
                }
            for (int st = 0; st < 3; ++st) {
                for (int r = 0; r < CH; ++r) for (int c = 0; c < CH; ++c) {
                    float p2 = 0.0f;
                    if (r > c) for (int kx = c + 1; kx < r; ++kx) p2 = std::fma(P[r * CH + kx], P[kx * CH + c], p2);
                    tmp[r * CH + c] = p2;
                }
                std::memcpy(P, tmp, sizeof(P));
                for (int r = 0; r < CH; ++r) for (int c = 0; c < CH; ++c) {
                    float tp = T[r * CH + c];
                    if (r > c) for (int kx = c + 1; kx <= r; ++kx) tp = std::fma(T[r * CH + kx], P[kx * CH + c], tp);
                    tmp[r * CH + c] = tp;
                }
                std::memcpy(T, tmp, sizeof(T));
            }
            std::vector<float> rhs(CH * DV, 0.0f), qs(CH * DV, 0.0f), dvv(CH * DV, 0.0f);
            for (int t = 0; t < clen; ++t)
                for (int j = 0; j < DV; ++j) {
                    float pk[8], pq[8];
                    for (int rg = 0; rg < 8; ++rg) {
                        float a = 0.0f, b = 0.0f;
                        for (int r = 0; r < 16; ++r) { a = std::fma(Sa[(rg * 16 + r) * DV + j], k[(m0 + t) * DK + rg * 16 + r], a); b = std::fma(Sa[(rg * 16 + r) * DV + j], q[(m0 + t) * DK + rg * 16 + r], b); }
                        pk[rg] = a; pq[rg] = b;
                    }
                    const float sk = pk[0] + pk[1] + pk[2] + pk[3] + pk[4] + pk[5] + pk[6] + pk[7];
                    const float sq = pq[0] + pq[1] + pq[2] + pq[3] + pq[4] + pq[5] + pq[6] + pq[7];
                    rhs[t * DV + j] = beta[m0 + t] * (v[(m0 + t) * DV + j] - sk * cg[t]);
                    qs[t * DV + j] = sq * cg[t];
                }
            for (int t = 0; t < clen; ++t) for (int j = 0; j < DV; ++j) {
                float d = 0.0f;
                for (int u = 0; u <= t; ++u) d = std::fma(T[t * CH + u], rhs[u * DV + j], d);
                dvv[t * DV + j] = d;
            }
            for (int t = 0; t < clen; ++t) for (int j = 0; j < DV; ++j) {
                float intra = 0.0f;
                for (int u = 0; u <= t; ++u) intra = std::fma(QK[t * CH + u], dvv[u * DV + j], intra);
                foa[(m0 + t) * DV + j] = qs[t * DV + j] + intra;
            }
            const float g_end = cg[clen - 1];
            for (int i = 0; i < DK; ++i) for (int j = 0; j < DV; ++j) {
                float sv = Sa[i * DV + j] * g_end;
                for (int u = 0; u < clen; ++u) sv = std::fma((g_end / std::fmax(cg[u], 1e-30f)) * k[(m0 + u) * DK + i], dvv[u * DV + j], sv);
                Sa[i * DV + j] = sv;
            }
        }
        double fo_scale = 1e-30, s_scale = 1e-30, ea = 0.0, eb = 0.0, sa_err = 0.0, sb_err = 0.0;
        for (int i = 0; i < Mtok * DV; ++i) fo_scale = std::max(fo_scale, std::fabs(fod[i]));
        for (int i = 0; i < DK * DV; ++i) s_scale = std::max(s_scale, std::fabs(Sd[i]));
        for (int i = 0; i < Mtok * DV; ++i) { ea = std::max(ea, std::fabs(foa[i] - fod[i]) / fo_scale); eb = std::max(eb, std::fabs(fob[i] - fod[i]) / fo_scale); }
        for (int i = 0; i < DK * DV; ++i) { sa_err = std::max(sa_err, std::fabs(Sa[i] - Sd[i]) / s_scale); sb_err = std::max(sb_err, std::fabs(Sb[i] - Sd[i]) / s_scale); }
        std::printf("[gdn M=%d] max rel err vs double: fo WY-split %.2e, fo split-V(5b) %.2e | S WY-split %.2e, S split-V %.2e\n",
                    Mtok, ea, eb, sa_err, sb_err);
        VJ_CHECK(ea < 1e-4 && eb < 1e-4 && sa_err < 1e-4 && sb_err < 1e-4);
    }
}

} // namespace

int main() {
    test_fa2_reduce_scatter();
    test_fa2_softmax();
    test_gemm();
    test_swar_signs();
    test_iq2s_tiles();
    test_iq3xxs();
    test_ar_bound();
    test_gdn();
    std::printf("test_prefill_ref: OK\n");
    return 0;
}
