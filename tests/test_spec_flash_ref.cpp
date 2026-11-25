// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU reference for ops/kernels/vj_spec_flash.hip (16:1 head-packed Split-KV FlashDecoding).
//   * seq   : the sequential online softmax of ornith_attn_flash_quant_kernel (position by position, FP32)
//   * split : a lane-level emulation of vj_spec_flash_partial_kernel + vj_spec_flash_merge_quant_kernel
//             (same byte addressing, same per-lane 8-dim partial dots with the block scale applied before the
//             32-lane butterfly, 4-position rescale batches, chunk states merged by log-sum-exp)
//   * exact : FP64 softmax attention
// Both FP32 orders are bounded against FP64, and the INT8 outputs of split vs seq may differ by at most one
// quantisation step on a small fraction of elements: token-level (not bit-level) equivalence.
#include "tests/test_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr uint32_t kHd = 256, kRow = 272;

uint16_t f2h(float f) { _Float16 h = static_cast<_Float16>(f); uint16_t u; std::memcpy(&u, &h, 2); return u; }
float h2f(uint16_t u) { _Float16 h; std::memcpy(&h, &u, 2); return static_cast<float>(h); }

struct Problem {
    uint32_t M, pos0, nq, nkv;
    std::vector<float> q, gate;           // [M, nq, 256]
    std::vector<uint8_t> kc, vc;          // [pos, nkv, 272]
};

Problem make_problem(uint32_t M, uint32_t pos0, uint32_t nq, uint32_t nkv, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    Problem p{M, pos0, nq, nkv, {}, {}, {}, {}};
    p.q.resize(static_cast<size_t>(M) * nq * kHd);
    p.gate.resize(p.q.size());
    for (auto& v : p.q) v = nd(rng) * 1.5f;   // RMS-normed + RoPE'd q has O(1) entries; 1.5 sharpens softmax
    for (auto& v : p.gate) v = nd(rng);
    const uint32_t npos = pos0 + M;
    p.kc.assign(static_cast<size_t>(npos) * nkv * kRow, 0);
    p.vc.assign(p.kc.size(), 0);
    for (auto* buf : {&p.kc, &p.vc}) {
        for (uint32_t t = 0; t < npos; ++t)
            for (uint32_t h = 0; h < nkv; ++h) {
                uint8_t* r = buf->data() + (static_cast<size_t>(t) * nkv + h) * kRow;
                for (uint32_t b = 0; b < 8; ++b) {
                    float x[32], amax = 0.0f;
                    for (float& v : x) { v = nd(rng) * (1.0f + 0.5f * static_cast<float>(b & 3)); amax = std::max(amax, std::fabs(v)); }
                    const uint16_t sh = f2h(amax / 127.0f);
                    std::memcpy(r + 2 * b, &sh, 2);
                    const float sc = h2f(sh);
                    for (uint32_t i = 0; i < 32; ++i)
                        r[16 + b * 32 + i] = static_cast<uint8_t>(static_cast<int8_t>(std::clamp(static_cast<int>(std::lrint(x[i] / sc)), -127, 127)));
                }
            }
    }
    return p;
}

float kv_elem(const std::vector<uint8_t>& buf, uint32_t nkv, uint32_t t, uint32_t h, uint32_t d) {
    const uint8_t* r = buf.data() + (static_cast<size_t>(t) * nkv + h) * kRow;
    uint16_t sh;
    std::memcpy(&sh, r + 2 * (d >> 5), 2);
    return h2f(sh) * static_cast<float>(static_cast<int8_t>(r[16 + d]));
}

void gate_quant(const float* attn, const float* gate, float* y, int8_t* q8, float* ds) {
    for (uint32_t d = 0; d < kHd; ++d) y[d] = attn[d] * (1.0f / (1.0f + std::exp(-gate[d])));
    for (uint32_t b = 0; b < 8; ++b) {
        float amax = 0.0f, sum = 0.0f;
        for (uint32_t i = 0; i < 32; ++i) { amax = std::max(amax, std::fabs(y[b * 32 + i])); sum += y[b * 32 + i]; }
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        for (uint32_t i = 0; i < 32; ++i)
            q8[b * 32 + i] = static_cast<int8_t>(std::clamp(static_cast<int>(std::nearbyint(y[b * 32 + i] * inv)), -127, 127));
        ds[2 * b] = amax / 127.0f;
        ds[2 * b + 1] = sum;
    }
}

struct Out { std::vector<float> y; std::vector<int8_t> q8; std::vector<float> ds; };

Out run_seq(const Problem& p) {
    Out o{std::vector<float>(p.q.size()), std::vector<int8_t>(p.q.size()), std::vector<float>(p.q.size() / 16)};
    for (uint32_t m = 0; m < p.M; ++m)
        for (uint32_t h = 0; h < p.nq; ++h) {
            const uint32_t kvh = h * p.nkv / p.nq, seqlen = p.pos0 + m + 1;
            const float* q = &p.q[(static_cast<size_t>(m) * p.nq + h) * kHd];
            float mr = -1e30f, lr = 0.0f, acc[kHd] = {};
            for (uint32_t t = 0; t < seqlen; ++t) {
                float w[4] = {};
                for (uint32_t d = 0; d < kHd; ++d) w[d >> 6] += q[d] * kv_elem(p.kc, p.nkv, t, kvh, d);
                const float s = (w[0] + w[1] + w[2] + w[3]) * 0.0625f;
                const float mn = std::max(mr, s), a = std::exp(mr - mn), b = std::exp(s - mn);
                lr = lr * a + b;
                mr = mn;
                for (uint32_t d = 0; d < kHd; ++d) acc[d] = std::fma(b, kv_elem(p.vc, p.nkv, t, kvh, d), acc[d] * a);
            }
            float attn[kHd];
            for (uint32_t d = 0; d < kHd; ++d) attn[d] = lr > 0.0f ? acc[d] / lr : 0.0f;
            const size_t oi = (static_cast<size_t>(m) * p.nq + h) * kHd;
            gate_quant(attn, &p.gate[oi], &o.y[oi], &o.q8[oi], &o.ds[oi / 16]);
        }
    return o;
}

std::vector<double> run_exact(const Problem& p) {
    std::vector<double> y(p.q.size());
    for (uint32_t m = 0; m < p.M; ++m)
        for (uint32_t h = 0; h < p.nq; ++h) {
            const uint32_t kvh = h * p.nkv / p.nq, seqlen = p.pos0 + m + 1;
            const float* q = &p.q[(static_cast<size_t>(m) * p.nq + h) * kHd];
            std::vector<double> s(seqlen);
            double mx = -1e300;
            for (uint32_t t = 0; t < seqlen; ++t) {
                double a = 0.0;
                for (uint32_t d = 0; d < kHd; ++d) a += static_cast<double>(q[d]) * kv_elem(p.kc, p.nkv, t, kvh, d);
                s[t] = a / 16.0;
                mx = std::max(mx, s[t]);
            }
            double l = 0.0;
            std::vector<double> acc(kHd, 0.0);
            for (uint32_t t = 0; t < seqlen; ++t) {
                const double w = std::exp(s[t] - mx);
                l += w;
                for (uint32_t d = 0; d < kHd; ++d) acc[d] += w * kv_elem(p.vc, p.nkv, t, kvh, d);
            }
            const size_t oi = (static_cast<size_t>(m) * p.nq + h) * kHd;
            for (uint32_t d = 0; d < kHd; ++d)
                y[oi + d] = acc[d] / l / (1.0 + std::exp(-static_cast<double>(p.gate[oi + d])));
        }
    return y;
}

// Host launch rule for the split kernel.
void choose_chunks(uint32_t pos0, uint32_t M, uint32_t max_chunks, uint32_t min_chunk, uint32_t* n_chunks, uint32_t* chunk_len) {
    const uint32_t span = pos0 + M;
    *n_chunks = std::clamp((span + min_chunk - 1) / min_chunk, 1u, max_chunks);
    *chunk_len = ((span + *n_chunks - 1) / *n_chunks + 3u) & ~3u;
}

float i8_at(uint32_t w, uint32_t k) { return static_cast<float>(static_cast<int8_t>((w >> (8u * k)) & 0xFFu)); }

// Lane-level emulation of the two kernels (512 threads: wave w -> heads 2w, 2w+1; 32 lanes x 8 dims).
Out run_split(const Problem& p, uint32_t n_chunks, uint32_t chunk_len) {
    const uint32_t nq = p.nq, M = p.M;
    std::vector<float> part_o(static_cast<size_t>(M) * nq * n_chunks * kHd, std::nanf("")), part_ml(static_cast<size_t>(M) * nq * n_chunks * 2);
    const uint32_t row_stride = p.nkv * kRow;
    for (uint32_t kvh = 0; kvh < p.nkv; ++kvh)
        for (uint32_t m = 0; m < M; ++m)
            for (uint32_t c = 0; c < n_chunks; ++c)
                for (uint32_t hl = 0; hl < 16; ++hl) {
                    const uint32_t h = kvh * 16 + hl, seqlen = p.pos0 + m + 1;
                    const uint32_t tb = c * chunk_len, te = std::min(tb + chunk_len, seqlen);
                    const size_t st = (static_cast<size_t>(m) * nq + h) * n_chunks + c;
                    if (tb >= te) { part_ml[st * 2] = -1e30f; part_ml[st * 2 + 1] = 0.0f; continue; }
                    float q[32][8], o[32][8] = {}, mr = -1e30f, lr = 0.0f;
                    for (uint32_t l = 0; l < 32; ++l)
                        for (uint32_t i = 0; i < 8; ++i) q[l][i] = p.q[(static_cast<size_t>(m) * nq + h) * kHd + l * 8 + i];
                    for (uint32_t t0 = tb; t0 < te; t0 += 4) {
                        float s[4];
                        uint32_t vwx[4][32], vwy[4][32];
                        float vsc[4][32];
                        for (uint32_t j = 0; j < 4; ++j) {
                            const size_t ro = static_cast<size_t>(std::min(t0 + j, te - 1)) * row_stride;
                            float lanev[32];
                            for (uint32_t l = 0; l < 32; ++l) {
                                const uint32_t qs_off = kvh * kRow + 16 + l * 8, sc_off = kvh * kRow + 2 * (l >> 2);
                                uint32_t kx, ky, vx, vy;
                                uint16_t ks, vs;
                                std::memcpy(&kx, &p.kc[ro + qs_off], 4); std::memcpy(&ky, &p.kc[ro + qs_off + 4], 4);
                                std::memcpy(&ks, &p.kc[ro + sc_off], 2);
                                std::memcpy(&vx, &p.vc[ro + qs_off], 4); std::memcpy(&vy, &p.vc[ro + qs_off + 4], 4);
                                std::memcpy(&vs, &p.vc[ro + sc_off], 2);
                                vwx[j][l] = vx; vwy[j][l] = vy; vsc[j][l] = h2f(vs);
                                float a0 = 0.0f, a1 = 0.0f;
                                for (uint32_t k = 0; k < 4; k += 2) {
                                    a0 = std::fma(q[l][k], i8_at(kx, k), a0);
                                    a1 = std::fma(q[l][k + 1], i8_at(kx, k + 1), a1);
                                    a0 = std::fma(q[l][4 + k], i8_at(ky, k), a0);
                                    a1 = std::fma(q[l][5 + k], i8_at(ky, k + 1), a1);
                                }
                                lanev[l] = (a0 + a1) * h2f(ks);
                            }
                            for (uint32_t sh : {16u, 8u, 4u, 2u, 1u}) {  // butterfly (identical on every lane)
                                float nv[32];
                                for (uint32_t l = 0; l < 32; ++l) nv[l] = lanev[l] + lanev[l ^ sh];
                                std::memcpy(lanev, nv, sizeof(nv));
                            }
                            s[j] = lanev[0] * 0.0625f;
                        }
                        float mn = mr;
                        for (uint32_t j = 0; j < 4; ++j) if (t0 + j < te) mn = std::max(mn, s[j]);
                        const float a = std::exp(mr - mn);
                        lr *= a;
                        for (auto& ol : o) for (float& v : ol) v *= a;
                        for (uint32_t j = 0; j < 4; ++j) {
                            const float b = (t0 + j < te) ? std::exp(s[j] - mn) : 0.0f;
                            lr += b;
                            for (uint32_t l = 0; l < 32; ++l) {
                                const float bv = b * vsc[j][l];
                                for (uint32_t k = 0; k < 4; ++k) {
                                    o[l][k] = std::fma(bv, i8_at(vwx[j][l], k), o[l][k]);
                                    o[l][4 + k] = std::fma(bv, i8_at(vwy[j][l], k), o[l][4 + k]);
                                }
                            }
                        }
                        mr = mn;
                    }
                    for (uint32_t l = 0; l < 32; ++l)
                        for (uint32_t i = 0; i < 8; ++i) part_o[st * kHd + l * 8 + i] = o[l][i];
                    part_ml[st * 2] = mr;
                    part_ml[st * 2 + 1] = lr;
                }
    Out out{std::vector<float>(p.q.size()), std::vector<int8_t>(p.q.size()), std::vector<float>(p.q.size() / 16)};
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t h = 0; h < nq; ++h) {
            const size_t base = (static_cast<size_t>(m) * nq + h) * n_chunks;
            float gmax = -1e30f;
            for (uint32_t c = 0; c < n_chunks; ++c) if (part_ml[(base + c) * 2 + 1] > 0.0f) gmax = std::max(gmax, part_ml[(base + c) * 2]);
            float attn[kHd];
            for (uint32_t d = 0; d < kHd; ++d) {
                float o = 0.0f, l = 0.0f;
                for (uint32_t c = 0; c < n_chunks; ++c) {
                    const float lc = part_ml[(base + c) * 2 + 1];
                    if (lc > 0.0f) {
                        const float w = std::exp(part_ml[(base + c) * 2] - gmax);
                        l = std::fma(w, lc, l);
                        o = std::fma(w, part_o[(base + c) * kHd + d], o);
                    }
                }
                attn[d] = l > 0.0f ? o / l : 0.0f;
            }
            const size_t oi = (static_cast<size_t>(m) * nq + h) * kHd;
            gate_quant(attn, &p.gate[oi], &out.y[oi], &out.q8[oi], &out.ds[oi / 16]);
        }
    return out;
}

} // namespace

int main() {
    struct Case { uint32_t M, pos0, nq, nkv, max_chunks; };
    const Case cases[] = {
        {1, 0, 16, 1, 60},   {1, 3, 16, 1, 60},   {1, 127, 16, 1, 60}, {1, 511, 16, 1, 60}, {1, 4095, 16, 1, 60},
        {5, 0, 16, 1, 60},   {5, 130, 16, 1, 60}, {8, 509, 16, 1, 60}, {12, 1000, 16, 1, 60},
        {3, 200, 32, 2, 60}, {1, 777, 16, 1, 1},  {4, 64, 16, 1, 128},
    };
    double worst_split = 0.0, worst_seq = 0.0;
    size_t q8_diff = 0, q8_total = 0;
    int max_lsb = 0;
    int n = 0;
    for (const Case& cs : cases) {
        const Problem p = make_problem(cs.M, cs.pos0, cs.nq, cs.nkv, 1000u + cs.pos0 * 7u + cs.M);
        uint32_t nc, cl;
        choose_chunks(cs.pos0, cs.M, cs.max_chunks, 32u, &nc, &cl);
        VJ_CHECK(cl % 4 == 0 && nc * cl >= cs.pos0 + cs.M && nc <= cs.max_chunks);
        const Out seq = run_seq(p), spl = run_split(p, nc, cl);
        const std::vector<double> ex = run_exact(p);
        double e_split = 0.0, e_seq = 0.0, ymax = 0.0;
        for (size_t i = 0; i < ex.size(); ++i) {
            ymax = std::max(ymax, std::fabs(ex[i]));
            e_split = std::max(e_split, std::fabs(spl.y[i] - ex[i]));
            e_seq = std::max(e_seq, std::fabs(seq.y[i] - ex[i]));
            VJ_CHECK(std::isfinite(spl.y[i]));
            const int dq = std::abs(static_cast<int>(spl.q8[i]) - static_cast<int>(seq.q8[i]));
            max_lsb = std::max(max_lsb, dq);
            q8_diff += dq != 0;
            ++q8_total;
        }
        for (size_t b = 0; b < spl.ds.size(); b += 2) {
            VJ_CHECK(std::fabs(spl.ds[b] - seq.ds[b]) <= 1e-5f * std::max(1.0f, std::fabs(seq.ds[b])));
            VJ_CHECK(std::fabs(spl.ds[b + 1] - seq.ds[b + 1]) <= 1e-4f * 32.0f * std::max(1.0f, seq.ds[b] * 127.0f));
        }
        const double rs = e_split / std::max(ymax, 1e-30), rq = e_seq / std::max(ymax, 1e-30);
        worst_split = std::max(worst_split, rs);
        worst_seq = std::max(worst_seq, rq);
        std::printf("[spec_flash_ref] M=%2u pos0=%4u nq=%u nkv=%u chunks=%2u x %3u | max|y-fp64|/max|y|: split %.2e seq %.2e\n",
                    cs.M, cs.pos0, cs.nq, cs.nkv, nc, cl, rs, rq);
        VJ_CHECK(rs < 2e-5 && rq < 2e-5);
        ++n;
    }
    // Chunk-count independence: 1 chunk vs many chunks agree to FP32 rounding.
    {
        const Problem p = make_problem(2, 300, 16, 1, 99);
        const Out a = run_split(p, 1, 304), b = run_split(p, 60, 8), c = run_split(p, 7, 44);
        double d1 = 0.0, d2 = 0.0;
        for (size_t i = 0; i < a.y.size(); ++i) { d1 = std::max(d1, (double) std::fabs(a.y[i] - b.y[i])); d2 = std::max(d2, (double) std::fabs(a.y[i] - c.y[i])); }
        VJ_CHECK(d1 < 1e-5 && d2 < 1e-5);
        ++n;
    }
    std::printf("[spec_flash_ref] worst rel err split %.2e seq %.2e | int8 split vs seq: %zu/%zu differ, max %d LSB\n",
                worst_split, worst_seq, q8_diff, q8_total, max_lsb);
    VJ_CHECK(max_lsb <= 1);
    VJ_CHECK(q8_diff * 100 <= q8_total);  // <= 1 % of elements move by one step
    std::printf("test_spec_flash_ref: all %d checks passed\n", n);
    return 0;
}
