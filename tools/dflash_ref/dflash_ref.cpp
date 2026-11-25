// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// dflash_ref: exact CPU reference of the z-lab DFlash block-diffusion drafter (arXiv 2602.06036,
// github.com/z-lab/dflash, HF "DFlashDraftModel", model_type qwen3) for measuring draft acceptance
// against hidden-state dumps from the VegaJet target. No GPU, no llama.cpp.
//
// Build (from the repo root):
//   g++ -std=c++20 -O3 -march=native -fopenmp -I . tools/dflash_ref/dflash_ref.cpp core/gguf_reader.cpp core/quant_repacker.cpp -o dflash_ref
//
// Usage:
//   dflash_ref selftest
//   dflash_ref info  --draft <dflash-model>
//   dflash_ref run   --draft <dflash-model> --target <target-model> --dump <file.vjhs>:<n_prompt> [--dump ...]
//                    [--block 8,16] [--sweep] [--check-dump] [semantic switches below]
//
// Semantics implemented (z-lab dflash/model.py + HF config; evidence block at the end of this comment):
//   ctx features   c_t = hidden_norm( fc( concat_k state[t][tap_k] ) ),  tap_k = dflash.target_layers[k] + tap_shift
//                  (dump state s = residual stream entering target layer s, i.e. HF hidden_states[s])
//   context K/V    per draft layer: K = rope(k_norm(Wk c_t), t), V = Wv c_t   (no attn_norm on c_t)
//   block          tokens [anchor=ids[p], MASK x (B-1)] at positions p..p+B-1, embeddings = target token_embd
//   layer          x = attn_norm(h); q = rope(q_norm(Wq x)); k,v from x likewise; attention of the B queries over
//                  [context 0..p-1 (window on SWA layers)] + [block tokens]; block mask per z-lab model.py:
//                  sliding_attention layers are CAUSAL (key index <= query index), full_attention layers are
//                  bidirectional (is_causal defaults to layer_type == "sliding_attention"); h += Wo o;
//                  h += Wd(silu(Wg n) * Wu n), n = ffn_norm(h)
//   output         target_lm_head( output_norm(h) ) at block slots 1..B-1 -> drafts for ids[p+1..p+B-1]
//   one denoising step; greedy argmax.
//
// Semantic switches (defaults = z-lab):
//   --tap-shift N        add N to every GGUF target_layers entry when indexing dump states (default 0)
//   --norm-order M       fc-norm (default) | norm-fc (enc norm per tapped chunk, then fc) | fc | rms-fc-norm
//   --ctx-attn-norm 0|1  apply each layer's attn_norm to c_t before Wk/Wv (default 0)
//   --anchor-ctx 0|1     context also includes the anchor position p itself (default 0)
//   --block-mask M       zlab (default: causal on SWA layers, bidirectional on full layers) | bidir (all layers
//                        bidirectional = what the llama.cpp fork does) | causal (all layers causal)
//   --rope neox|norm     rotate-half (default, HF Qwen3) or interleaved pairs
//   --window W           sliding window for SWA layers (default from GGUF; 0 = no SWA anywhere)
//   --mask-id ID         mask token (default from GGUF)
//   --no-ctx 0|1         diagnostic: drop all target context (drafter sees only the block) (default 0)
//
// Evidence (z-lab = github.com/z-lab/dflash dflash/model.py @95c8aec; fork = llama.cpp gfx906 fork):
//   taps     z-lab extract_context_feature uses hidden_states[layer_id + 1] (HF index 0 = embeddings), HF ids
//            [1,7,14,20,26,32,39,45] -> HF hidden_states [2,8,...,46]. Fork converter writes ids+1 into
//            dflash.target_layers (conversion/qwen.py:715-718) and taps the INPUT of target layer k
//            (models/qwen35moe.cpp:179 t_layer_inp[il] = inpL; common/speculative.cpp:1110), i.e. the same
//            residual stream. Dump state s == HF hidden_states[s], so tap-shift 0 is correct; all pre-final-norm.
//   fc/norm  z-lab: target_hidden = hidden_norm(fc(target_hidden)); fork dflash.cpp:603-604 identical.
//   ctx K/V  z-lab: k = k_norm(cat(k_proj(ctx), k_proj(noise))), v unnormed, no input_layernorm on ctx, RoPE after
//            norms; fork dflash.cpp:610-619 identical (KV injected into the draft cache at target positions).
//   mask     z-lab: is_causal = (layer_type == "sliding_attention") unless config.is_causal is set; HF config has
//            no is_causal -> 5 SWA layers causal over [ctx|block] (key index <= query index, window 4096), the
//            full layer bidirectional. Fork: one global flag from dflash.attention.causal (absent -> all layers
//            bidirectional; common/speculative.cpp:1038,1115). DEVIATION (fork is bidirectional on SWA layers).
//   block    z-lab: [last committed token, MASK...] at positions start..start+B-1, logits of slots 1..B-1,
//            single forward, draft cache cropped to start; fork speculative.cpp:1265-1268 same layout but with
//            n_block_tokens = n_max+1 (n_max=3 -> a 4-token block, not 16).
//   rope     HF qwen3 rotate-half, theta 1e7, full 128 dims; fork uses MROPE with sections [64,0,0,0] =
//            temporal-only = NeoX (llama-model.cpp:3081-3087).
#include "core/gguf_reader.hpp"
#include "core/quant_repacker.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <random>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace vegajet;
using namespace vegajet::quant;

namespace {

// ================================================================================================
// Config / weights
// ================================================================================================
struct Cfg {
    int D = 3072, H = 32, HKV = 8, HD = 128, FF = 9216, L = 6;
    float eps = 1e-6f, theta = 1e7f;
    int window = 4096;
    std::vector<int> swa;   // per layer 1 = sliding window
    std::vector<int> taps;  // GGUF target_layers
    int mask_id = 248077;
    int block_size = 16;
};

enum class NormOrder { FcNorm = 0, NormFc = 1, FcOnly = 2, RmsFcNorm = 3 };

struct Opt {
    int tap_shift = 0;
    NormOrder norm_order = NormOrder::FcNorm;
    bool ctx_attn_norm = false;
    bool anchor_ctx = false;
    int block_mask = 2;  // 0 = bidir (fork), 1 = causal on all layers, 2 = z-lab (causal on SWA layers)
    bool rope_neox = true;
    int window = -1;  // -1 = from cfg
    bool no_ctx = false;
};

bool layer_causal(const Opt& o, const Cfg& g, int l) { return o.block_mask == 1 || (o.block_mask == 2 && g.swa[l]); }

struct Layer {
    std::vector<float> wq, wk, wv, wo, qn, kn, an, fn, wg, wu, wd;
};

struct Drafter {
    Cfg cfg;
    std::vector<float> fc, enc_norm, out_norm;  // fc: [D][n_taps*D]
    std::vector<Layer> layers;
};

// ================================================================================================
// Math helpers
// ================================================================================================
// Y[n][M] = X[n][K] * W[M][K]^T
void matmul(const float* X, size_t n, size_t K, const float* W, size_t M, float* Y) {
    const size_t OB = 32, NB = 64;
    const size_t n_ob = (M + OB - 1) / OB, n_nb = (n + NB - 1) / NB;
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t t = 0; t < n_ob * n_nb; ++t) {
        const size_t ob = (t / n_nb) * OB, nb = (t % n_nb) * NB;
        const size_t oe = std::min(M, ob + OB), ne = std::min(n, nb + NB);
        for (size_t r = nb; r < ne; r += 4) {
            const size_t rn = std::min<size_t>(4, ne - r);
            for (size_t o = ob; o < oe; o += 4) {
                const size_t on = std::min<size_t>(4, oe - o);
                if (rn == 4 && on == 4) {
                    const float *x0 = X + r * K, *x1 = x0 + K, *x2 = x1 + K, *x3 = x2 + K;
                    const float *w0 = W + o * K, *w1 = w0 + K, *w2 = w1 + K, *w3 = w2 + K;
                    float a00 = 0, a01 = 0, a02 = 0, a03 = 0, a10 = 0, a11 = 0, a12 = 0, a13 = 0;
                    float a20 = 0, a21 = 0, a22 = 0, a23 = 0, a30 = 0, a31 = 0, a32 = 0, a33 = 0;
#pragma omp simd reduction(+ : a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33)
                    for (size_t k = 0; k < K; ++k) {
                        const float v0 = x0[k], v1 = x1[k], v2 = x2[k], v3 = x3[k];
                        const float u0 = w0[k], u1 = w1[k], u2 = w2[k], u3 = w3[k];
                        a00 += v0 * u0; a01 += v0 * u1; a02 += v0 * u2; a03 += v0 * u3;
                        a10 += v1 * u0; a11 += v1 * u1; a12 += v1 * u2; a13 += v1 * u3;
                        a20 += v2 * u0; a21 += v2 * u1; a22 += v2 * u2; a23 += v2 * u3;
                        a30 += v3 * u0; a31 += v3 * u1; a32 += v3 * u2; a33 += v3 * u3;
                    }
                    float* y = Y + r * M + o;
                    y[0] = a00; y[1] = a01; y[2] = a02; y[3] = a03; y += M;
                    y[0] = a10; y[1] = a11; y[2] = a12; y[3] = a13; y += M;
                    y[0] = a20; y[1] = a21; y[2] = a22; y[3] = a23; y += M;
                    y[0] = a30; y[1] = a31; y[2] = a32; y[3] = a33;
                } else {
                    for (size_t i = 0; i < rn; ++i)
                        for (size_t j = 0; j < on; ++j) {
                            const float* x = X + (r + i) * K;
                            const float* w = W + (o + j) * K;
                            float s = 0;
#pragma omp simd reduction(+ : s)
                            for (size_t k = 0; k < K; ++k) s += x[k] * w[k];
                            Y[(r + i) * M + o + j] = s;
                        }
                }
            }
        }
    }
}

// in-place RMSNorm of n rows of width d (w may be null = weightless)
void rmsnorm_rows(float* x, size_t n, size_t d, const float* w, float eps) {
#pragma omp parallel for if (n * d > 65536)
    for (size_t r = 0; r < n; ++r) {
        float* v = x + r * d;
        double ss = 0;
        for (size_t i = 0; i < d; ++i) ss += (double)v[i] * v[i];
        const float s = 1.0f / std::sqrt((float)(ss / d) + eps);
        for (size_t i = 0; i < d; ++i) v[i] = v[i] * s * (w ? w[i] : 1.0f);
    }
}

// in-place RoPE of one head vector (HD) at position pos
void rope_head(float* v, int HD, int64_t pos, float theta, bool neox) {
    const int half = HD / 2;
    for (int i = 0; i < half; ++i) {
        const double inv = std::pow((double)theta, -2.0 * i / HD);
        const double ang = (double)pos * inv;
        const float c = (float)std::cos(ang), s = (float)std::sin(ang);
        const int a = neox ? i : 2 * i, b = neox ? i + half : 2 * i + 1;
        const float x0 = v[a], x1 = v[b];
        v[a] = x0 * c - x1 * s;
        v[b] = x0 * s + x1 * c;
    }
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

// ================================================================================================
// Context feature path (shared by fast path and selftest)
// ================================================================================================
// taps: [n][n_taps*D] raw dump states. out c: [n][D]
void ctx_features(const Drafter& m, const Opt& o, const float* taps, size_t n, std::vector<float>& c) {
    const Cfg& g = m.cfg;
    const size_t nt = g.taps.size(), DI = nt * g.D;
    std::vector<float> in(taps, taps + n * DI);
    if (o.norm_order == NormOrder::NormFc) rmsnorm_rows(in.data(), n * nt, g.D, m.enc_norm.data(), g.eps);
    if (o.norm_order == NormOrder::RmsFcNorm) rmsnorm_rows(in.data(), n * nt, g.D, nullptr, g.eps);
    c.assign(n * g.D, 0.f);
    matmul(in.data(), n, DI, m.fc.data(), g.D, c.data());
    if (o.norm_order == NormOrder::FcNorm || o.norm_order == NormOrder::RmsFcNorm)
        rmsnorm_rows(c.data(), n, g.D, m.enc_norm.data(), g.eps);
}

// Per layer context K (normed + roped) and V for positions 0..n-1. kc/vc: [L][n][HKV*HD]
void ctx_kv(const Drafter& m, const Opt& o, const std::vector<float>& c, size_t n,
            std::vector<std::vector<float>>& kc, std::vector<std::vector<float>>& vc) {
    const Cfg& g = m.cfg;
    const size_t KVD = (size_t)g.HKV * g.HD;
    kc.assign(g.L, {});
    vc.assign(g.L, {});
    std::vector<float> x(c);
    for (int l = 0; l < g.L; ++l) {
        const Layer& ly = m.layers[l];
        const float* src = c.data();
        if (o.ctx_attn_norm) {
            x = c;
            rmsnorm_rows(x.data(), n, g.D, ly.an.data(), g.eps);
            src = x.data();
        }
        kc[l].assign(n * KVD, 0.f);
        vc[l].assign(n * KVD, 0.f);
        matmul(src, n, g.D, ly.wk.data(), KVD, kc[l].data());
        matmul(src, n, g.D, ly.wv.data(), KVD, vc[l].data());
        rmsnorm_rows(kc[l].data(), n * g.HKV, g.HD, ly.kn.data(), g.eps);
#pragma omp parallel for
        for (size_t t = 0; t < n; ++t)
            for (int h = 0; h < g.HKV; ++h) rope_head(kc[l].data() + t * KVD + h * g.HD, g.HD, (int64_t)t, g.theta, o.rope_neox);
    }
}

// ================================================================================================
// Fast batched block forward: all cycles at once
// ================================================================================================
// cyc_p[j]: anchor position of cycle j; emb: [ncyc*B][D] block input embeddings.
// Context for cycle j: positions [0, p) (or [0, p] with anchor_ctx), each within the SWA window on SWA layers.
// Returns final normed hidden [ncyc*B][D].
void block_forward(const Drafter& m, const Opt& o, const std::vector<std::vector<float>>& kc,
                   const std::vector<std::vector<float>>& vc, size_t n_ctx_avail, const std::vector<int>& cyc_p, int B,
                   std::vector<float> h, std::vector<float>& out) {
    const Cfg& g = m.cfg;
    const size_t ncyc = cyc_p.size(), R = ncyc * B;
    const size_t QD = (size_t)g.H * g.HD, KVD = (size_t)g.HKV * g.HD;
    const int rep = g.H / g.HKV;
    const float scale = 1.0f / std::sqrt((float)g.HD);
    const int window = o.window >= 0 ? o.window : g.window;
    std::vector<float> x(R * g.D), q(R * QD), k(R * KVD), v(R * KVD), att(R * QD), tmp(R * g.D);
    std::vector<float> gt(R * g.FF), up(R * g.FF);
    for (int l = 0; l < g.L; ++l) {
        const Layer& ly = m.layers[l];
        const bool swa = window > 0 && g.swa[l];
        const bool causal = layer_causal(o, g, l);
        x = h;
        rmsnorm_rows(x.data(), R, g.D, ly.an.data(), g.eps);
        matmul(x.data(), R, g.D, ly.wq.data(), QD, q.data());
        matmul(x.data(), R, g.D, ly.wk.data(), KVD, k.data());
        matmul(x.data(), R, g.D, ly.wv.data(), KVD, v.data());
        rmsnorm_rows(q.data(), R * g.H, g.HD, ly.qn.data(), g.eps);
        rmsnorm_rows(k.data(), R * g.HKV, g.HD, ly.kn.data(), g.eps);
#pragma omp parallel for
        for (size_t r = 0; r < R; ++r) {
            const int64_t pos = cyc_p[r / B] + (int64_t)(r % B);
            for (int hh = 0; hh < g.H; ++hh) rope_head(q.data() + r * QD + hh * g.HD, g.HD, pos, g.theta, o.rope_neox);
            for (int hh = 0; hh < g.HKV; ++hh) rope_head(k.data() + r * KVD + hh * g.HD, g.HD, pos, g.theta, o.rope_neox);
        }
        // attention
#pragma omp parallel for schedule(dynamic, 1) collapse(2)
        for (size_t j = 0; j < ncyc; ++j)
            for (int hh = 0; hh < g.H; ++hh) {
                const int p = cyc_p[j];
                const int kvh = hh / rep;
                const int ctx_end = o.no_ctx ? 0 : std::min<int>(o.anchor_ctx ? p + 1 : p, (int)n_ctx_avail);
                std::vector<float> sc;
                for (int i = 0; i < B; ++i) {
                    const size_t r = j * B + i;
                    const int qpos = p + i;
                    const float* qv = q.data() + r * QD + hh * g.HD;
                    const int ctx_beg = swa ? std::max(0, qpos - window + 1) : 0;
                    sc.clear();
                    for (int t = ctx_beg; t < ctx_end; ++t) {
                        const float* kv = kc[l].data() + (size_t)t * KVD + kvh * g.HD;
                        float s = 0;
                        for (int d = 0; d < g.HD; ++d) s += qv[d] * kv[d];
                        sc.push_back(s * scale);
                    }
                    const int nb = causal ? i + 1 : B;
                    for (int i2 = 0; i2 < nb; ++i2) {
                        const float* kv = k.data() + (j * B + i2) * KVD + kvh * g.HD;
                        float s = 0;
                        for (int d = 0; d < g.HD; ++d) s += qv[d] * kv[d];
                        sc.push_back(s * scale);
                    }
                    float mx = -INFINITY;
                    for (float s : sc) mx = std::max(mx, s);
                    double den = 0;
                    for (float& s : sc) { s = std::exp(s - mx); den += s; }
                    float* ov = att.data() + r * QD + hh * g.HD;
                    std::fill(ov, ov + g.HD, 0.f);
                    size_t idx = 0;
                    for (int t = ctx_beg; t < ctx_end; ++t, ++idx) {
                        const float w = (float)(sc[idx] / den);
                        const float* vv = vc[l].data() + (size_t)t * KVD + kvh * g.HD;
                        for (int d = 0; d < g.HD; ++d) ov[d] += w * vv[d];
                    }
                    for (int i2 = 0; i2 < nb; ++i2, ++idx) {
                        const float w = (float)(sc[idx] / den);
                        const float* vv = v.data() + (j * B + i2) * KVD + kvh * g.HD;
                        for (int d = 0; d < g.HD; ++d) ov[d] += w * vv[d];
                    }
                }
            }
        matmul(att.data(), R, QD, ly.wo.data(), g.D, tmp.data());
        for (size_t i = 0; i < R * g.D; ++i) h[i] += tmp[i];
        x = h;
        rmsnorm_rows(x.data(), R, g.D, ly.fn.data(), g.eps);
        matmul(x.data(), R, g.D, ly.wg.data(), g.FF, gt.data());
        matmul(x.data(), R, g.D, ly.wu.data(), g.FF, up.data());
#pragma omp parallel for
        for (size_t i = 0; i < R * (size_t)g.FF; ++i) gt[i] = silu(gt[i]) * up[i];
        matmul(gt.data(), R, g.FF, ly.wd.data(), g.D, tmp.data());
        for (size_t i = 0; i < R * g.D; ++i) h[i] += tmp[i];
    }
    rmsnorm_rows(h.data(), R, g.D, m.out_norm.data(), g.eps);
    out = std::move(h);
}

// ================================================================================================
// Slow, obviously-correct single-cycle forward (selftest only). Double precision, explicit sequence,
// explicit mask matrix, complex-number RoPE, recomputes everything from raw taps.
// ================================================================================================
std::vector<double> slow_cycle(const Drafter& m, const Opt& o, const std::vector<float>& taps /*[n][nt*D]*/,
                               size_t n_avail, int p, const std::vector<std::vector<float>>& blk_emb /*B x D*/) {
    const Cfg& g = m.cfg;
    const int B = (int)blk_emb.size(), D = g.D, nt = (int)g.taps.size(), HD = g.HD;
    const int window = o.window >= 0 ? o.window : g.window;
    auto norm = [&](std::vector<double> v, const float* w) {
        double ss = 0;
        for (double a : v) ss += a * a;
        const double s = 1.0 / std::sqrt(ss / v.size() + g.eps);
        for (size_t i = 0; i < v.size(); ++i) v[i] *= s * (w ? w[i] : 1.0);
        return v;
    };
    auto lin = [&](const std::vector<float>& W, const std::vector<double>& xv, int out) {
        std::vector<double> y(out, 0.0);
        const size_t in = xv.size();
        for (int r = 0; r < out; ++r)
            for (size_t i = 0; i < in; ++i) y[r] += (double)W[r * in + i] * xv[i];
        return y;
    };
    auto rope = [&](std::vector<double>& vh, int64_t pos) {  // one head, complex form
        for (int i = 0; i < HD / 2; ++i) {
            const int a = o.rope_neox ? i : 2 * i, b = o.rope_neox ? i + HD / 2 : 2 * i + 1;
            std::complex<double> z(vh[a], vh[b]);
            z *= std::polar(1.0, (double)pos * std::pow((double)g.theta, -(2.0 * i) / HD));
            vh[a] = z.real();
            vh[b] = z.imag();
        }
    };
    // context sequence: positions 0..ce-1
    const int ce = o.no_ctx ? 0 : std::min<int>(o.anchor_ctx ? p + 1 : p, (int)n_avail);
    std::vector<std::vector<double>> cfeat(ce);
    for (int t = 0; t < ce; ++t) {
        std::vector<double> in((size_t)nt * D);
        for (int k = 0; k < nt; ++k) {
            std::vector<double> chunk(D);
            for (int d = 0; d < D; ++d) chunk[d] = taps[(size_t)t * nt * D + (size_t)k * D + d];
            if (o.norm_order == NormOrder::NormFc) chunk = norm(chunk, m.enc_norm.data());
            if (o.norm_order == NormOrder::RmsFcNorm) chunk = norm(chunk, nullptr);
            for (int d = 0; d < D; ++d) in[(size_t)k * D + d] = chunk[d];
        }
        auto y = lin(m.fc, in, D);
        if (o.norm_order == NormOrder::FcNorm || o.norm_order == NormOrder::RmsFcNorm) y = norm(y, m.enc_norm.data());
        cfeat[t] = y;
    }
    std::vector<std::vector<double>> hb(B);
    for (int i = 0; i < B; ++i) hb[i].assign(blk_emb[i].begin(), blk_emb[i].end());
    // full key sequence = [ctx 0..ce-1 | block 0..B-1]; positions:
    const int S = ce + B;
    std::vector<int64_t> kpos(S);
    for (int t = 0; t < ce; ++t) kpos[t] = t;
    for (int i = 0; i < B; ++i) kpos[ce + i] = p + i;
    for (int l = 0; l < g.L; ++l) {
        const Layer& ly = m.layers[l];
        const bool swa = window > 0 && g.swa[l];
        // explicit mask M[i][s]
        std::vector<std::vector<char>> M(B, std::vector<char>(S, 0));
        for (int i = 0; i < B; ++i)
            for (int s = 0; s < S; ++s) {
                const int64_t qp = p + i, kp = kpos[s];
                bool ok = true;
                if (s >= ce && layer_causal(o, g, l) && (s - ce) > i) ok = false;
                if (s < ce && swa && qp - kp >= window) ok = false;
                M[i][s] = ok;
            }
        std::vector<std::vector<double>> K(S), V(S), Q(B);
        for (int s = 0; s < S; ++s) {
            std::vector<double> src = s < ce ? cfeat[s] : norm(hb[s - ce], ly.an.data());
            if (s < ce && o.ctx_attn_norm) src = norm(src, ly.an.data());
            K[s] = lin(ly.wk, src, g.HKV * HD);
            V[s] = lin(ly.wv, src, g.HKV * HD);
            for (int hh = 0; hh < g.HKV; ++hh) {
                std::vector<double> kh(K[s].begin() + hh * HD, K[s].begin() + (hh + 1) * HD);
                kh = norm(kh, ly.kn.data());
                rope(kh, kpos[s]);
                std::copy(kh.begin(), kh.end(), K[s].begin() + hh * HD);
            }
        }
        std::vector<std::vector<double>> newh(B);
        for (int i = 0; i < B; ++i) {
            Q[i] = lin(ly.wq, norm(hb[i], ly.an.data()), g.H * HD);
            std::vector<double> o_all((size_t)g.H * HD, 0.0);
            for (int hh = 0; hh < g.H; ++hh) {
                std::vector<double> qh(Q[i].begin() + hh * HD, Q[i].begin() + (hh + 1) * HD);
                qh = norm(qh, ly.qn.data());
                rope(qh, p + i);
                const int kvh = hh * g.HKV / g.H;
                std::vector<double> sc(S, -INFINITY);
                double mx = -INFINITY;
                for (int s = 0; s < S; ++s) {
                    if (!M[i][s]) continue;
                    double d = 0;
                    for (int e = 0; e < HD; ++e) d += qh[e] * K[s][kvh * HD + e];
                    sc[s] = d / std::sqrt((double)HD);
                    mx = std::max(mx, sc[s]);
                }
                double den = 0;
                for (int s = 0; s < S; ++s) if (M[i][s]) den += std::exp(sc[s] - mx);
                for (int s = 0; s < S; ++s) {
                    if (!M[i][s]) continue;
                    const double w = std::exp(sc[s] - mx) / den;
                    for (int e = 0; e < HD; ++e) o_all[hh * HD + e] += w * V[s][kvh * HD + e];
                }
            }
            auto ao = lin(ly.wo, o_all, D);
            std::vector<double> h1(D);
            for (int d = 0; d < D; ++d) h1[d] = hb[i][d] + ao[d];
            auto nn = norm(h1, ly.fn.data());
            auto ga = lin(ly.wg, nn, g.FF), uu = lin(ly.wu, nn, g.FF);
            for (int f = 0; f < g.FF; ++f) ga[f] = ga[f] / (1.0 + std::exp(-ga[f])) * uu[f];
            auto dd = lin(ly.wd, ga, D);
            for (int d = 0; d < D; ++d) h1[d] += dd[d];
            newh[i] = h1;
        }
        hb = newh;
    }
    std::vector<double> out;
    for (int i = 0; i < B; ++i) {
        auto y = norm(hb[i], m.out_norm.data());
        out.insert(out.end(), y.begin(), y.end());
    }
    return out;
}

// ================================================================================================
// GGUF loading
// ================================================================================================
bool load_tensor_f32(GgufFile& f, const std::string& name, std::vector<float>& out, uint64_t expect_elems) {
    const GgufTensorInfo* t = f.tensor(name);
    if (!t) { fprintf(stderr, "missing tensor %s\n", name.c_str()); return false; }
    if (expect_elems && t->n_elems() != expect_elems) {
        fprintf(stderr, "tensor %s: %lu elems, expected %lu\n", name.c_str(), (unsigned long)t->n_elems(), (unsigned long)expect_elems);
        return false;
    }
    const uint8_t* src = f.tensor_data(*t);
    if (!src) { fprintf(stderr, "map failed for %s\n", name.c_str()); return false; }
    const uint64_t cols = t->ne[0], rows = t->n_elems() / cols;
    const auto& reg = DecoderRegistry::instance();
    const uint64_t rb = reg.row_bytes((DiskQuantType)t->type, cols);
    if (!rb) { fprintf(stderr, "unsupported type %u for %s\n", t->type, name.c_str()); return false; }
    out.resize(t->n_elems());
    bool ok = true;
#pragma omp parallel for reduction(&& : ok)
    for (uint64_t r = 0; r < rows; ++r) ok = ok && reg.decode_row_f32((DiskQuantType)t->type, src + r * rb, out.data() + r * cols, cols);
    f.release(src, t->n_bytes());
    return ok;
}

bool load_drafter(const std::string& path, Drafter& m) {
    GgufFile f;
    if (!f.open(path)) { fprintf(stderr, "open %s: %s\n", path.c_str(), f.error().c_str()); return false; }
    if (f.architecture() != "dflash") { fprintf(stderr, "%s: architecture '%s' != dflash\n", path.c_str(), f.architecture().c_str()); return false; }
    Cfg& g = m.cfg;
    uint64_t u;
    double d;
    if (f.get_arch_u64("embedding_length", u)) g.D = (int)u;
    if (f.get_arch_u64("attention.head_count", u)) g.H = (int)u;
    if (f.get_arch_u64("attention.head_count_kv", u)) g.HKV = (int)u;
    if (f.get_arch_u64("attention.key_length", u)) g.HD = (int)u;
    if (f.get_arch_u64("feed_forward_length", u)) g.FF = (int)u;
    if (f.get_arch_u64("block_count", u)) g.L = (int)u;
    if (f.get_arch_f64("attention.layer_norm_rms_epsilon", d)) g.eps = (float)d;
    if (f.get_arch_f64("rope.freq_base", d)) g.theta = (float)d;
    if (f.get_arch_u64("attention.sliding_window", u)) g.window = (int)u; else g.window = 0;
    if (f.get_arch_u64("block_size", u)) g.block_size = (int)u;
    if (f.get_u64("tokenizer.ggml.mask_token_id", u)) g.mask_id = (int)u;
    if (f.get_arch_u64("rope.dimension_count", u) && (int)u != g.HD) {
        fprintf(stderr, "partial rotary (%lu of %d) not supported by this reference\n", (unsigned long)u, g.HD);
        return false;
    }
    g.swa.assign(g.L, 0);
    if (const GgufValue* v = f.find("dflash.attention.sliding_window_pattern"))
        for (int l = 0; l < g.L && l < (int)v->arr_num.size(); ++l) g.swa[l] = (int)v->arr_num[l];
    const GgufValue* tl = f.find("dflash.target_layers");
    if (!tl || tl->arr_num.empty()) { fprintf(stderr, "missing dflash.target_layers\n"); return false; }
    g.taps.clear();
    for (double x : tl->arr_num) g.taps.push_back((int)x);
    const uint64_t D = g.D, QD = (uint64_t)g.H * g.HD, KVD = (uint64_t)g.HKV * g.HD, FF = g.FF;
    bool ok = load_tensor_f32(f, "fc.weight", m.fc, D * D * g.taps.size()) &&
              load_tensor_f32(f, "enc.output_norm.weight", m.enc_norm, D) &&
              load_tensor_f32(f, "output_norm.weight", m.out_norm, D);
    m.layers.resize(g.L);
    for (int l = 0; l < g.L && ok; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        Layer& ly = m.layers[l];
        ok = load_tensor_f32(f, p + "attn_q.weight", ly.wq, QD * D) && load_tensor_f32(f, p + "attn_k.weight", ly.wk, KVD * D) &&
             load_tensor_f32(f, p + "attn_v.weight", ly.wv, KVD * D) && load_tensor_f32(f, p + "attn_output.weight", ly.wo, D * QD) &&
             load_tensor_f32(f, p + "attn_q_norm.weight", ly.qn, g.HD) && load_tensor_f32(f, p + "attn_k_norm.weight", ly.kn, g.HD) &&
             load_tensor_f32(f, p + "attn_norm.weight", ly.an, D) && load_tensor_f32(f, p + "ffn_norm.weight", ly.fn, D) &&
             load_tensor_f32(f, p + "ffn_gate.weight", ly.wg, FF * D) && load_tensor_f32(f, p + "ffn_up.weight", ly.wu, FF * D) &&
             load_tensor_f32(f, p + "ffn_down.weight", ly.wd, D * FF);
    }
    return ok;
}

// Target: token embedding rows on demand + full LM head (f32) + output_norm (for --check-dump).
struct Target {
    GgufFile f;
    const GgufTensorInfo* emb = nullptr;
    std::vector<float> head;  // [V][D]
    std::vector<float> out_norm;
    uint64_t V = 0, D = 0;
    bool open(const std::string& path, bool need_head) {
        if (!f.open(path)) { fprintf(stderr, "open %s: %s\n", path.c_str(), f.error().c_str()); return false; }
        emb = f.tensor("token_embd.weight");
        if (!emb) { fprintf(stderr, "target: no token_embd.weight\n"); return false; }
        D = emb->ne[0];
        if (!load_tensor_f32(f, "output_norm.weight", out_norm, D)) return false;
        if (need_head) {
            const GgufTensorInfo* o = f.tensor("output.weight");
            if (!o) { fprintf(stderr, "target: no output.weight\n"); return false; }
            V = o->ne[1];
            fprintf(stderr, "[load] target LM head %lu x %lu (type %u) -> f32 (%.2f GiB)\n", (unsigned long)V, (unsigned long)D, o->type,
                    V * D * 4.0 / (1 << 30));
            if (!load_tensor_f32(f, "output.weight", head, 0)) return false;
        }
        return true;
    }
    bool embed(int tok, float* dst) {
        if (tok < 0 || (uint64_t)tok >= emb->ne[1]) return false;
        const auto& reg = DecoderRegistry::instance();
        const uint64_t rb = reg.row_bytes((DiskQuantType)emb->type, D);
        const uint8_t* base = f.tensor_data(*emb);
        return base && reg.decode_row_f32((DiskQuantType)emb->type, base + (uint64_t)tok * rb, dst, D);
    }
};

// top-5 argmax of rows of X [n][D] against head [V][D]
struct Top5 { int id[5]; float v[5]; };
void lm_top5(const float* head, uint64_t V, uint64_t D, const float* X, size_t n, std::vector<Top5>& res) {
    res.assign(n, Top5{});
    for (auto& r : res) for (int k = 0; k < 5; ++k) { r.id[k] = -1; r.v[k] = -INFINITY; }
    const size_t CH = 8192;
    std::vector<float> lg(n * CH);
    for (uint64_t v0 = 0; v0 < V; v0 += CH) {
        const size_t cn = std::min<uint64_t>(CH, V - v0);
        // logits^T chunk: [cn][n] would need X as W; compute [n][cn]
        matmul(X, n, D, head + v0 * D, cn, lg.data());
#pragma omp parallel for
        for (size_t r = 0; r < n; ++r) {
            Top5& t = res[r];
            const float* L = lg.data() + r * cn;
            for (size_t c = 0; c < cn; ++c) {
                const float val = L[c];
                if (val <= t.v[4]) continue;
                int k = 4;
                while (k > 0 && t.v[k - 1] < val) { t.v[k] = t.v[k - 1]; t.id[k] = t.id[k - 1]; --k; }
                t.v[k] = val;
                t.id[k] = (int)(v0 + c);
            }
        }
    }
}

// ================================================================================================
// Dump reader
// ================================================================================================
struct Dump {
    std::string path;
    int n_prompt = 0;
    uint32_t n = 0, ns = 0, D = 0;
    const int32_t* ids = nullptr;
    const float* states = nullptr;
    void* map = nullptr;
    size_t map_len = 0;
    bool open(const std::string& p, int np) {
        path = p;
        n_prompt = np;
        const int fd = ::open(p.c_str(), O_RDONLY);
        if (fd < 0) { fprintf(stderr, "open %s failed\n", p.c_str()); return false; }
        struct stat st {};
        fstat(fd, &st);
        map_len = (size_t)st.st_size;
        map = mmap(nullptr, map_len, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (map == MAP_FAILED || map_len < 20) { fprintf(stderr, "mmap %s failed\n", p.c_str()); return false; }
        const uint32_t* h = (const uint32_t*)map;
        if (h[0] != 0x53484A56u) { fprintf(stderr, "%s: bad magic 0x%08x (want VJHS)\n", p.c_str(), h[0]); return false; }
        if (h[1] != 1) { fprintf(stderr, "%s: unsupported version %u\n", p.c_str(), h[1]); return false; }
        n = h[2]; ns = h[3]; D = h[4];
        const size_t need = 20 + 4ull * n + 4ull * n * ns * D;
        if (map_len != need) { fprintf(stderr, "%s: size %zu != expected %zu\n", p.c_str(), map_len, need); return false; }
        ids = (const int32_t*)((const uint8_t*)map + 20);
        states = (const float*)((const uint8_t*)map + 20 + 4ull * n);
        if (n_prompt < 1 || n_prompt >= (int)n) { fprintf(stderr, "%s: n_prompt %d out of range (n=%u)\n", p.c_str(), n_prompt, n); return false; }
        return true;
    }
    const float* state(uint32_t t, uint32_t s) const { return states + ((size_t)t * ns + s) * D; }
};

// ================================================================================================
// Evaluation
// ================================================================================================
struct BlockStats {
    int B = 0;
    // teacher-forced, cycles with a full comparable block
    int n_full = 0;
    double sum_acc = 0;
    std::vector<int> hist;
    std::vector<int> slot_hit, slot_hit5, slot_n;
    // greedy walk
    int walk_cycles = 0, walk_acc = 0, walk_tokens = 0;
};

struct EvalCtx {
    Drafter* m;
    Target* tgt;
};

// Runs one (dump, B, opt) evaluation. verbose_cycles prints per-cycle proposals.
BlockStats evaluate(EvalCtx& E, const Dump& dp, const Opt& o, int B, bool verbose, std::vector<float>* emb_cache_mask) {
    const Drafter& m = *E.m;
    const Cfg& g = m.cfg;
    const size_t nt = g.taps.size(), D = g.D;
    BlockStats st;
    st.B = B;
    st.hist.assign(B, 0);
    st.slot_hit.assign(B, 0); st.slot_hit5.assign(B, 0); st.slot_n.assign(B, 0);
    // taps
    std::vector<int> sidx(nt);
    for (size_t k = 0; k < nt; ++k) {
        sidx[k] = g.taps[k] + o.tap_shift;
        if (sidx[k] < 0 || sidx[k] >= (int)dp.ns) { fprintf(stderr, "tap %d out of range of %u states\n", sidx[k], dp.ns); exit(2); }
    }
    const size_t n = dp.n;
    std::vector<float> taps(n * nt * D);
    for (size_t t = 0; t < n; ++t)
        for (size_t k = 0; k < nt; ++k) memcpy(&taps[(t * nt + k) * D], dp.state(t, sidx[k]), D * 4);
    std::vector<float> c;
    ctx_features(m, o, taps.data(), n, c);
    std::vector<std::vector<float>> kc, vc;
    ctx_kv(m, o, c, n, kc, vc);
    // cycles: anchor p from n_prompt .. n-2 (anchor = target's own token; context = positions < p)
    std::vector<int> cyc;
    for (int p = dp.n_prompt; p <= (int)n - 2; ++p) cyc.push_back(p);
    const size_t R = cyc.size() * B;
    std::vector<float> h(R * D);
    std::vector<float> mask_emb(D);
    if (emb_cache_mask && !emb_cache_mask->empty()) mask_emb = *emb_cache_mask;
    else if (!E.tgt->embed(g.mask_id, mask_emb.data())) { fprintf(stderr, "mask embed failed\n"); exit(2); }
    for (size_t j = 0; j < cyc.size(); ++j) {
        if (!E.tgt->embed(dp.ids[cyc[j]], &h[j * B * D])) { fprintf(stderr, "embed failed\n"); exit(2); }
        for (int i = 1; i < B; ++i) memcpy(&h[(j * B + i) * D], mask_emb.data(), D * 4);
    }
    std::vector<float> out;
    block_forward(m, o, kc, vc, n, cyc, B, std::move(h), out);
    // LM head on slots 1..B-1 only
    std::vector<float> sel(cyc.size() * (B - 1) * D);
    for (size_t j = 0; j < cyc.size(); ++j)
        for (int i = 1; i < B; ++i) memcpy(&sel[(j * (B - 1) + i - 1) * D], &out[(j * B + i) * D], D * 4);
    std::vector<Top5> top;
    lm_top5(E.tgt->head.data(), E.tgt->V, D, sel.data(), cyc.size() * (B - 1), top);
    std::vector<int> acc_at(n, 0);
    for (size_t j = 0; j < cyc.size(); ++j) {
        const int p = cyc[j];
        int acc = 0;
        bool run = true;
        const int avail = std::min(B - 1, (int)n - 1 - p);
        for (int i = 1; i <= avail; ++i) {
            const Top5& t = top[j * (B - 1) + i - 1];
            const int want = dp.ids[p + i];
            const bool hit = t.id[0] == want;
            bool hit5 = false;
            for (int k = 0; k < 5; ++k) hit5 |= t.id[k] == want;
            st.slot_n[i]++; st.slot_hit[i] += hit; st.slot_hit5[i] += hit5;
            if (run && hit) ++acc; else run = false;
        }
        acc_at[p] = acc;
        if (avail == B - 1) { st.n_full++; st.sum_acc += acc; st.hist[acc]++; }
        if (verbose) {
            printf("  p=%4d anchor=%6d acc=%2d draft:", p, dp.ids[p], acc);
            for (int i = 1; i < B; ++i) printf(" %d%s", top[j * (B - 1) + i - 1].id[0], (i <= avail && top[j * (B - 1) + i - 1].id[0] == dp.ids[p + i]) ? "*" : "");
            printf(" | target:");
            for (int i = 1; i <= avail; ++i) printf(" %d", dp.ids[p + i]);
            printf("\n");
        }
    }
    for (int p = dp.n_prompt; p <= (int)n - 2;) {
        st.walk_cycles++;
        st.walk_acc += acc_at[p];
        st.walk_tokens += acc_at[p] + 1;
        p += acc_at[p] + 1;
    }
    return st;
}

void print_stats(const char* tag, const BlockStats& s) {
    const double mean = s.n_full ? s.sum_acc / s.n_full : 0.0;
    printf("%s B=%-2d teacher-forced: cycles=%d mean_accepted_drafts=%.3f mean_accept_len(+1)=%.3f | walk: cycles=%d mean_accepted_drafts=%.3f tokens/cycle=%.3f\n",
           tag, s.B, s.n_full, mean, mean + 1.0, s.walk_cycles, s.walk_cycles ? (double)s.walk_acc / s.walk_cycles : 0.0,
           s.walk_cycles ? (double)s.walk_tokens / s.walk_cycles : 0.0);
    printf("%s B=%-2d hist(accepted=0..%d):", tag, s.B, s.B - 1);
    for (int v : s.hist) printf(" %d", v);
    printf("\n%s B=%-2d slot top1 / top5 agreement:", tag, s.B);
    for (int i = 1; i < s.B; ++i) printf(" %d:%.2f/%.2f", i, s.slot_n[i] ? (double)s.slot_hit[i] / s.slot_n[i] : 0.0, s.slot_n[i] ? (double)s.slot_hit5[i] / s.slot_n[i] : 0.0);
    printf("\n");
}

const char* norm_name(NormOrder n) {
    switch (n) {
        case NormOrder::FcNorm: return "fc-norm";
        case NormOrder::NormFc: return "norm-fc";
        case NormOrder::FcOnly: return "fc";
        case NormOrder::RmsFcNorm: return "rms-fc-norm";
    }
    return "?";
}

std::string opt_desc(const Opt& o) {
    char b[256];
    snprintf(b, sizeof b, "tap-shift=%+d norm-order=%s ctx-attn-norm=%d anchor-ctx=%d block-mask=%s rope=%s window=%d no-ctx=%d", o.tap_shift,
             norm_name(o.norm_order), o.ctx_attn_norm, o.anchor_ctx, o.block_mask == 0 ? "bidir" : (o.block_mask == 1 ? "causal" : "zlab"), o.rope_neox ? "neox" : "norm", o.window, o.no_ctx);
    return b;
}

// ================================================================================================
// Self-test
// ================================================================================================
int selftest() {
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    Drafter m;
    Cfg& g = m.cfg;
    g.D = 64; g.H = 4; g.HKV = 2; g.HD = 16; g.FF = 96; g.L = 3; g.window = 5; g.swa = {1, 0, 1}; g.taps = {1, 3};
    g.theta = 10000.f;
    const size_t D = g.D, nt = g.taps.size();
    auto rnd = [&](size_t k, float s) { std::vector<float> v(k); for (auto& x : v) x = nd(rng) * s; return v; };
    auto rnd_norm = [&](size_t k) { std::vector<float> v(k); for (auto& x : v) x = 1.f + 0.2f * nd(rng); return v; };
    m.fc = rnd(D * D * nt, 0.1f); m.enc_norm = rnd_norm(D); m.out_norm = rnd_norm(D);
    m.layers.resize(g.L);
    for (auto& ly : m.layers) {
        ly.wq = rnd(g.H * g.HD * D, 0.15f); ly.wk = rnd(g.HKV * g.HD * D, 0.15f); ly.wv = rnd(g.HKV * g.HD * D, 0.15f);
        ly.wo = rnd(D * g.H * g.HD, 0.1f); ly.qn = rnd_norm(g.HD); ly.kn = rnd_norm(g.HD); ly.an = rnd_norm(D); ly.fn = rnd_norm(D);
        ly.wg = rnd(g.FF * D, 0.1f); ly.wu = rnd(g.FF * D, 0.1f); ly.wd = rnd(D * g.FF, 0.1f);
    }
    const int n = 14, B = 4;
    std::vector<float> taps = rnd(n * nt * D, 1.0f);
    std::vector<std::vector<float>> emb_tab(10);
    for (auto& e : emb_tab) e = rnd(D, 0.5f);
    int fails = 0;
    auto check = [&](bool ok, const char* what) { printf("[selftest] %-62s %s\n", what, ok ? "PASS" : "FAIL"); fails += !ok; };

    // 1. RoPE: relative-position property and agreement with complex form
    {
        std::vector<float> q = rnd(g.HD, 1.f), k = rnd(g.HD, 1.f);
        bool ok = true;
        for (int mode = 0; mode < 2; ++mode) {
            auto dotp = [&](int a, int b) {
                std::vector<float> qq = q, kk = k;
                rope_head(qq.data(), g.HD, a, g.theta, mode == 0);
                rope_head(kk.data(), g.HD, b, g.theta, mode == 0);
                double s = 0;
                for (int i = 0; i < g.HD; ++i) s += qq[i] * kk[i];
                return s;
            };
            ok &= std::fabs(dotp(7, 3) - dotp(107, 103)) < 1e-3;
            ok &= std::fabs(dotp(0, 0) - [&] { double s = 0; for (int i = 0; i < g.HD; ++i) s += q[i] * k[i]; return s; }()) < 1e-4;
        }
        // neox pairs (i, i+HD/2): position 1 rotates element 0 with element HD/2 by angle 1 rad
        std::vector<float> e(g.HD, 0.f); e[0] = 1.f;
        rope_head(e.data(), g.HD, 1, g.theta, true);
        ok &= std::fabs(e[0] - std::cos(1.0)) < 1e-6 && std::fabs(e[g.HD / 2] - std::sin(1.0)) < 1e-6;
        check(ok, "rope: relative-position invariance + neox pairing");
    }
    // 2. fast batched path == slow explicit path for all switch combinations
    {
        double worst = 0;
        int combos = 0;
        for (int ac = 0; ac < 2; ++ac)
            for (int cb = 0; cb < 3; ++cb)
                for (int rn = 0; rn < 2; ++rn)
                    for (int no = 0; no < 4; ++no)
                        for (int can = 0; can < 2; ++can)
                            for (int w : {-1, 0}) {
                                Opt o;
                                o.anchor_ctx = ac; o.block_mask = cb; o.rope_neox = rn; o.norm_order = (NormOrder)no;
                                o.ctx_attn_norm = can; o.window = w;
                                std::vector<float> c;
                                ctx_features(m, o, taps.data(), n, c);
                                std::vector<std::vector<float>> kc, vc;
                                ctx_kv(m, o, c, n, kc, vc);
                                std::vector<int> cyc = {1, 4, 9, 12};
                                std::vector<float> h(cyc.size() * B * D);
                                std::vector<std::vector<std::vector<float>>> be(cyc.size());
                                for (size_t j = 0; j < cyc.size(); ++j)
                                    for (int i = 0; i < B; ++i) {
                                        const auto& e = emb_tab[i == 0 ? (cyc[j] % 9) : 9];
                                        be[j].push_back(e);
                                        memcpy(&h[(j * B + i) * D], e.data(), D * 4);
                                    }
                                std::vector<float> out;
                                block_forward(m, o, kc, vc, n, cyc, B, h, out);
                                for (size_t j = 0; j < cyc.size(); ++j) {
                                    auto ref = slow_cycle(m, o, taps, n, cyc[j], be[j]);
                                    for (size_t i = 0; i < ref.size(); ++i) worst = std::max(worst, std::fabs(ref[i] - out[j * B * D + i]));
                                }
                                ++combos;
                            }
        char msg[128];
        snprintf(msg, sizeof msg, "fast == slow over %d switch combos (max abs err %.2e)", combos, worst);
        check(worst < 2e-4, msg);
    }
    // 3. mask semantics via the slow path
    {
        Opt o;
        std::vector<std::vector<float>> be;
        for (int i = 0; i < B; ++i) be.push_back(emb_tab[i == 0 ? 3 : 9]);
        const int p = 12;
        auto base = slow_cycle(m, o, taps, n, p, be);
        // (a) perturbing context position 0 (outside the window of 5 for queries at >= 12) on an all-SWA model: no effect
        Drafter m2 = m;
        m2.cfg.swa = {1, 1, 1};
        auto a0 = slow_cycle(m2, o, taps, n, p, be);
        std::vector<float> t2 = taps;
        for (size_t d = 0; d < nt * D; ++d) t2[d] += 1.0f;
        auto a1 = slow_cycle(m2, o, t2, n, p, be);
        double da = 0;
        for (size_t i = 0; i < a0.size(); ++i) da = std::max(da, std::fabs(a0[i] - a1[i]));
        auto b1 = slow_cycle(m, o, t2, n, p, be);  // layer 1 is full attention -> must change
        double db = 0;
        for (size_t i = 0; i < base.size(); ++i) db = std::max(db, std::fabs(base[i] - b1[i]));
        check(da < 1e-12 && db > 1e-6, "mask: SWA hides out-of-window context, full layer sees it");
        // (b) context position p (anchor's own state) must not be visible by default
        std::vector<float> t3 = taps;
        for (size_t d = 0; d < nt * D; ++d) t3[(size_t)p * nt * D + d] += 1.0f;
        auto c1 = slow_cycle(m, o, t3, n, p, be);
        double dc = 0;
        for (size_t i = 0; i < base.size(); ++i) dc = std::max(dc, std::fabs(base[i] - c1[i]));
        Opt oa = o; oa.anchor_ctx = true;
        auto c2 = slow_cycle(m, oa, taps, n, p, be), c3 = slow_cycle(m, oa, t3, n, p, be);
        double dc2 = 0;
        for (size_t i = 0; i < c2.size(); ++i) dc2 = std::max(dc2, std::fabs(c2[i] - c3[i]));
        check(dc < 1e-12 && dc2 > 1e-6, "mask: context ends before anchor (unless --anchor-ctx 1)");
        // (c) bidirectional block: changing the last block token changes slot 1, unless causal
        auto be2 = be;
        be2[B - 1] = emb_tab[5];
        auto d1 = slow_cycle(m, o, taps, n, p, be2);
        double dd = 0;
        for (size_t i = D; i < 2 * D; ++i) dd = std::max(dd, std::fabs(base[i] - d1[i]));
        Opt oc = o; oc.block_mask = 1;
        auto e0 = slow_cycle(m, oc, taps, n, p, be), e1 = slow_cycle(m, oc, taps, n, p, be2);
        double de = 0;
        for (size_t i = D; i < 2 * D; ++i) de = std::max(de, std::fabs(e0[i] - e1[i]));
        check(dd > 1e-6 && de < 1e-12, "mask: full layer bidirectional (causal mode isolates slot 1)");
        // (d) z-lab mode on an all-SWA model: every layer causal -> slot 1 isolated from later block tokens
        Drafter m3 = m;
        m3.cfg.swa = {1, 1, 1};
        auto f0 = slow_cycle(m3, o, taps, n, p, be), f1 = slow_cycle(m3, o, taps, n, p, be2);
        Opt ob = o; ob.block_mask = 0;
        auto f2 = slow_cycle(m3, ob, taps, n, p, be), f3 = slow_cycle(m3, ob, taps, n, p, be2);
        double df = 0, dg = 0;
        for (size_t i = D; i < 2 * D; ++i) { df = std::max(df, std::fabs(f0[i] - f1[i])); dg = std::max(dg, std::fabs(f2[i] - f3[i])); }
        check(df < 1e-12 && dg > 1e-6, "mask: zlab mode = causal on SWA layers, bidir mode is not");
    }
    // 4. shapes: matmul against naive
    {
        std::vector<float> X = rnd(7 * 37, 1.f), W = rnd(13 * 37, 1.f), Y(7 * 13);
        matmul(X.data(), 7, 37, W.data(), 13, Y.data());
        double err = 0;
        for (int r = 0; r < 7; ++r)
            for (int o = 0; o < 13; ++o) {
                double s = 0;
                for (int k = 0; k < 37; ++k) s += (double)X[r * 37 + k] * W[o * 37 + k];
                err = std::max(err, std::fabs(s - Y[r * 13 + o]));
            }
        check(err < 1e-4, "matmul tails (7x37 * 13x37^T) vs naive");
    }
    // 5. top-5 LM head
    {
        std::vector<float> Hm = rnd(20000 * 16, 1.f), X = rnd(3 * 16, 1.f);
        std::vector<Top5> t;
        lm_top5(Hm.data(), 20000, 16, X.data(), 3, t);
        bool ok = true;
        for (int r = 0; r < 3; ++r) {
            std::vector<std::pair<float, int>> all;
            for (int v = 0; v < 20000; ++v) {
                float s = 0;
                for (int k = 0; k < 16; ++k) s += Hm[v * 16 + k] * X[r * 16 + k];
                all.push_back({s, v});
            }
            std::partial_sort(all.begin(), all.begin() + 5, all.end(), [](auto& a, auto& b) { return a.first > b.first; });
            for (int k = 0; k < 5; ++k) ok &= all[k].second == t[r].id[k];
        }
        check(ok, "LM head top-5 (chunked) vs brute force");
    }
    printf("[selftest] %s (%d failure%s)\n", fails ? "FAILED" : "ALL PASS", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}

// ================================================================================================
// CLI
// ================================================================================================
void usage() {
    fprintf(stderr,
            "usage:\n"
            "  dflash_ref selftest\n"
            "  dflash_ref info --draft <dflash-model>\n"
            "  dflash_ref run --draft <dflash-model> --target <target-model> --dump <f.vjhs>:<n_prompt> [--dump ...]\n"
            "             [--block 8,16] [--sweep] [--check-dump] [--verbose]\n"
            "             [--tap-shift N] [--norm-order fc-norm|norm-fc|fc|rms-fc-norm] [--ctx-attn-norm 0|1]\n"
            "             [--anchor-ctx 0|1] [--block-mask zlab|bidir|causal] [--rope neox|norm] [--window W] [--mask-id ID] [--no-ctx 0|1]\n");
}

int info(const std::string& path) {
    Drafter m;
    if (!load_drafter(path, m)) return 1;
    const Cfg& g = m.cfg;
    printf("D=%d H=%d HKV=%d HD=%d FF=%d L=%d eps=%g theta=%g window=%d block_size=%d mask_id=%d\n", g.D, g.H, g.HKV, g.HD, g.FF, g.L,
           g.eps, g.theta, g.window, g.block_size, g.mask_id);
    printf("swa pattern:");
    for (int s : g.swa) printf(" %d", s);
    printf("\ntarget_layers (GGUF, = dump state index at tap-shift 0):");
    for (int t : g.taps) printf(" %d", t);
    auto mean = [](const std::vector<float>& v) { double s = 0; for (float x : v) s += x; return s / v.size(); };
    printf("\nnorm weight means (plain RMSNorm weights should be O(1), ~0 would mean a (1+w) convention):\n");
    printf("  enc.output_norm %.4f  output_norm %.4f\n", mean(m.enc_norm), mean(m.out_norm));
    for (int l = 0; l < g.L; ++l)
        printf("  blk.%d attn_norm %.4f ffn_norm %.4f q_norm %.4f k_norm %.4f\n", l, mean(m.layers[l].an), mean(m.layers[l].fn),
               mean(m.layers[l].qn), mean(m.layers[l].kn));
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 2; }
    const std::string cmd = argv[1];
    if (cmd == "selftest") return selftest();
    std::string draft, target;
    std::vector<std::pair<std::string, int>> dumps;
    std::vector<int> blocks = {8, 16};
    bool sweep = false, check_dump = false, verbose = false;
    Opt o;
    int mask_id = -1;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { usage(); exit(2); } return argv[++i]; };
        if (a == "--draft") draft = next();
        else if (a == "--target") target = next();
        else if (a == "--dump") {
            const std::string s = next();
            const size_t c = s.rfind(':');
            if (c == std::string::npos) { fprintf(stderr, "--dump needs <file>:<n_prompt>\n"); return 2; }
            dumps.push_back({s.substr(0, c), std::atoi(s.c_str() + c + 1)});
        } else if (a == "--block") {
            blocks.clear();
            std::string s = next();
            size_t pos = 0;
            while (pos < s.size()) { blocks.push_back(std::atoi(s.c_str() + pos)); pos = s.find(',', pos); if (pos == std::string::npos) break; ++pos; }
        } else if (a == "--sweep") sweep = true;
        else if (a == "--check-dump") check_dump = true;
        else if (a == "--verbose") verbose = true;
        else if (a == "--tap-shift") o.tap_shift = std::atoi(next().c_str());
        else if (a == "--norm-order") {
            const std::string s = next();
            if (s == "fc-norm") o.norm_order = NormOrder::FcNorm;
            else if (s == "norm-fc") o.norm_order = NormOrder::NormFc;
            else if (s == "fc") o.norm_order = NormOrder::FcOnly;
            else if (s == "rms-fc-norm") o.norm_order = NormOrder::RmsFcNorm;
            else { usage(); return 2; }
        } else if (a == "--ctx-attn-norm") o.ctx_attn_norm = std::atoi(next().c_str());
        else if (a == "--anchor-ctx") o.anchor_ctx = std::atoi(next().c_str());
        else if (a == "--block-mask") {
            const std::string s = next();
            if (s == "bidir") o.block_mask = 0; else if (s == "causal") o.block_mask = 1; else if (s == "zlab") o.block_mask = 2;
            else { usage(); return 2; }
        }
        else if (a == "--rope") o.rope_neox = next() != "norm";
        else if (a == "--window") o.window = std::atoi(next().c_str());
        else if (a == "--no-ctx") o.no_ctx = std::atoi(next().c_str());
        else if (a == "--mask-id") mask_id = std::atoi(next().c_str());
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); usage(); return 2; }
    }
    if (cmd == "info") { if (draft.empty()) { usage(); return 2; } return info(draft); }
    if (cmd != "run" || draft.empty() || target.empty() || dumps.empty()) { usage(); return 2; }

    Drafter m;
    fprintf(stderr, "[load] drafter %s\n", draft.c_str());
    if (!load_drafter(draft, m)) return 1;
    if (mask_id >= 0) m.cfg.mask_id = mask_id;
    Target tgt;
    fprintf(stderr, "[load] target %s\n", target.c_str());
    if (!tgt.open(target, true)) return 1;
    if (tgt.D != (uint64_t)m.cfg.D) { fprintf(stderr, "target D %lu != drafter D %d\n", (unsigned long)tgt.D, m.cfg.D); return 1; }
    std::vector<Dump> ds(dumps.size());
    for (size_t i = 0; i < dumps.size(); ++i) {
        if (!ds[i].open(dumps[i].first, dumps[i].second)) return 1;
        if (ds[i].D != (uint32_t)m.cfg.D) { fprintf(stderr, "dump D mismatch\n"); return 1; }
        printf("[dump] %s: n_tokens=%u n_states=%u D=%u n_prompt=%d cycles(anchor p=%d..%u)\n", ds[i].path.c_str(), ds[i].n, ds[i].ns, ds[i].D,
               ds[i].n_prompt, ds[i].n_prompt, ds[i].n - 2);
    }
    printf("[conv] accepted = leading draft tokens equal to the target's greedy tokens (0..B-1); accept_len = accepted+1 (bonus),\n"
           "       which is the convention of the z-lab/SGLang 'accept length' tables. Gate uses mean accepted drafts at B=8.\n");
    EvalCtx E{&m, &tgt};

    if (check_dump) {
        for (auto& d : ds) {
            int hit = 0, tot = 0;
            std::vector<float> X;
            std::vector<int> want;
            for (uint32_t t = d.n_prompt - 1; t + 1 < d.n; ++t) {
                X.insert(X.end(), d.state(t, d.ns - 1), d.state(t, d.ns - 1) + d.D);
                want.push_back(d.ids[t + 1]);
            }
            rmsnorm_rows(X.data(), want.size(), d.D, tgt.out_norm.data(), m.cfg.eps);
            std::vector<Top5> top;
            lm_top5(tgt.head.data(), tgt.V, tgt.D, X.data(), want.size(), top);
            for (size_t i = 0; i < want.size(); ++i) { hit += top[i].id[0] == want[i]; ++tot; }
            printf("[check-dump] %s: target_lm_head(output_norm(state[%u])) argmax == next id: %d/%d\n", d.path.c_str(), d.ns - 1, hit, tot);
            for (int s : {0, 1, m.cfg.taps[0], m.cfg.taps[0] + 1, m.cfg.taps.back(), (int)d.ns - 1}) {
                double ss = 0;
                for (uint32_t t = 0; t < d.n; ++t) for (uint32_t k = 0; k < d.D; ++k) { const double v = d.state(t, s)[k]; ss += v * v; }
                printf("[check-dump]   state %2d mean rms %.4f\n", s, std::sqrt(ss / ((double)d.n * d.D)));
            }
        }
    }

    std::vector<Opt> configs;
    if (sweep) {
        Opt b = o;
        configs.push_back(b);
        for (int s : {-1, 1}) { Opt x = b; x.tap_shift = s; configs.push_back(x); }
        for (int no : {1, 2, 3}) { Opt x = b; x.norm_order = (NormOrder)no; configs.push_back(x); }
        { Opt x = b; x.anchor_ctx = true; configs.push_back(x); }
        { Opt x = b; x.ctx_attn_norm = true; configs.push_back(x); }
        { Opt x = b; x.block_mask = 0; configs.push_back(x); }
        { Opt x = b; x.block_mask = 1; configs.push_back(x); }
        { Opt x = b; x.rope_neox = false; configs.push_back(x); }
        { Opt x = b; x.no_ctx = true; configs.push_back(x); }
    } else configs.push_back(o);

    std::vector<float> mask_emb(m.cfg.D);
    if (!tgt.embed(m.cfg.mask_id, mask_emb.data())) { fprintf(stderr, "mask embed failed\n"); return 1; }
    printf("\n");
    for (size_t ci = 0; ci < configs.size(); ++ci) {
        printf("=== config %zu: %s\n", ci, opt_desc(configs[ci]).c_str());
        for (int B : blocks) {
            if (B < 2) continue;
            BlockStats agg;
            agg.B = B; agg.hist.assign(B, 0); agg.slot_hit.assign(B, 0); agg.slot_hit5.assign(B, 0); agg.slot_n.assign(B, 0);
            for (auto& d : ds) {
                BlockStats s = evaluate(E, d, configs[ci], B, verbose, &mask_emb);
                std::string tag = "[" + d.path.substr(d.path.find_last_of('/') + 1) + "]";
                print_stats(tag.c_str(), s);
                agg.n_full += s.n_full; agg.sum_acc += s.sum_acc; agg.walk_cycles += s.walk_cycles; agg.walk_acc += s.walk_acc;
                agg.walk_tokens += s.walk_tokens;
                for (int i = 0; i < B; ++i) { agg.hist[i] += s.hist[i]; agg.slot_hit[i] += s.slot_hit[i]; agg.slot_hit5[i] += s.slot_hit5[i]; agg.slot_n[i] += s.slot_n[i]; }
            }
            if (ds.size() > 1) print_stats("[ALL]", agg);
            if (B == 8) {
                const double mean = agg.n_full ? agg.sum_acc / agg.n_full : 0.0;
                printf("GATE(Issue #152) config %zu B=8 mean_accepted_drafts=%.3f -> %s\n", ci, mean,
                       mean >= 4.0 ? "PASS" : (mean < 3.5 ? "ABORT" : "MARGINAL"));
            }
            fflush(stdout);
        }
    }
    return 0;
}
