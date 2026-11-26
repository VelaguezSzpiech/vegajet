// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/dflash_step_ref.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace vegajet::dflash_step {

bool validate(const Config& c, std::string* err) {
    auto fail = [&](const std::string& s) { if (err) *err = s; return false; };
    if (c.d <= 0 || c.hd <= 0 || c.block <= 0) return fail("dimensions must be positive");
    if (c.pos < 0) return fail("negative anchor position");
    if (c.window <= 0) return fail("window must be positive");
    if (c.layers.empty()) return fail("empty layer list");
    for (size_t i = 0; i < c.layers.size(); ++i) {
        if (c.layers[i] < 0 || c.layers[i] >= c.n_states) return fail("tap layer outside the dump");
        if (i && c.layers[i] <= c.layers[i - 1]) return fail("tap layers must be strictly ascending");
    }
    return true;
}

bool mask_allowed(const Config& c, bool swa_layer, int slot, int key_pos) {
    const int q = c.pos + slot;
    if (key_pos >= c.pos) {            // a key inside the draft block
        if (!swa_layer) return true;   // full-attention layers are bidirectional over the block
        return key_pos <= q && q - key_pos < c.window;
    }
    if (!swa_layer) return true;       // full attention sees the whole context
    return q - key_pos < c.window;     // z-lab SWA rule: q - k < W
}

std::vector<uint8_t> build_mask(const Config& c, bool swa_layer) {
    const int n_keys = c.pos + c.block;
    std::vector<uint8_t> m(size_t(c.block) * n_keys, 0);
    for (int i = 0; i < c.block; ++i)
        for (int k = 0; k < n_keys; ++k) m[size_t(i) * n_keys + k] = mask_allowed(c, swa_layer, i, k) ? 1 : 0;
    return m;
}

bool build_context(const Config& c, const std::vector<double>& states, const std::vector<double>& fc,
                   const std::vector<double>* norm_w, std::vector<double>* out, std::string* err) {
    auto fail = [&](const std::string& s) { if (err) *err = s; return false; };
    if (!validate(c, err)) return false;
    const size_t nt = c.layers.size(), d = size_t(c.d);
    if (states.size() < size_t(c.n_states) * d) return fail("state buffer too small");
    if (fc.size() != d * nt * d) return fail("fc must be d x (n_taps * d)");
    if (norm_w && norm_w->size() != d) return fail("norm weights must be d");
    std::vector<double> cat(nt * d);
    for (size_t k = 0; k < nt; ++k)
        for (size_t i = 0; i < d; ++i) cat[k * d + i] = states[size_t(c.layers[k]) * d + i];
    std::vector<double> proj(d, 0.0);
    for (size_t i = 0; i < d; ++i) {
        double s = 0;
        const double* row = &fc[i * nt * d];
        for (size_t j = 0; j < nt * d; ++j) s += row[j] * cat[j];
        proj[i] = s;
    }
    double ss = 0;
    for (double v : proj) ss += v * v;
    const double rms = std::sqrt(ss / double(d) + c.eps);
    out->resize(d);
    for (size_t i = 0; i < d; ++i) (*out)[i] = proj[i] / rms * (norm_w ? (*norm_w)[i] : 1.0);
    return true;
}

namespace {
// y [out_dim] = W [out_dim][in_dim] * x [in_dim]
void matvec(const std::vector<double>& W, const double* x, size_t in_dim, size_t out_dim, double* y) {
    for (size_t o = 0; o < out_dim; ++o) {
        double s = 0;
        const double* row = &W[o * in_dim];
        for (size_t i = 0; i < in_dim; ++i) s += row[i] * x[i];
        y[o] = s;
    }
}
}  // namespace

bool single_step(const Config& c, const std::vector<double>& states, const std::vector<double>& fc,
                 const std::vector<double>* norm_w, const std::vector<double>& block_x,
                 const std::vector<double>& ctx_k, const std::vector<double>& ctx_v,
                 const std::vector<double>& wq, const std::vector<double>& wk, const std::vector<double>& wv,
                 bool swa_layer, Step* out, std::string* err) {
    auto fail = [&](const std::string& s) { if (err) *err = s; return false; };
    const size_t d = size_t(c.d), hd = size_t(c.hd), B = size_t(c.block), P = size_t(c.pos);
    if (!build_context(c, states, fc, norm_w, &out->context, err)) return false;
    if (block_x.size() != B * d) return fail("block_x must be block x d");
    if (ctx_k.size() != P * hd || ctx_v.size() != P * hd) return fail("context keys/values must be pos x hd");
    if (wq.size() != hd * d || wk.size() != hd * d || wv.size() != hd * d) return fail("projections must be hd x d");

    const size_t n_keys = P + B;
    std::vector<double> q(B * hd), k(n_keys * hd), v(n_keys * hd);
    std::vector<double> xc(d);
    for (size_t b = 0; b < B; ++b) {
        for (size_t i = 0; i < d; ++i) xc[i] = block_x[b * d + i] + out->context[i];
        matvec(wq, xc.data(), d, hd, &q[b * hd]);
        matvec(wk, &block_x[b * d], d, hd, &k[(P + b) * hd]);
        matvec(wv, &block_x[b * d], d, hd, &v[(P + b) * hd]);
    }
    for (size_t j = 0; j < P * hd; ++j) { k[j] = ctx_k[j]; v[j] = ctx_v[j]; }

    out->mask = build_mask(c, swa_layer);
    const double scale = 1.0 / std::sqrt(double(hd));
    out->out.assign(B * hd, 0.0);
    std::vector<double> logits(n_keys);
    for (size_t b = 0; b < B; ++b) {
        const uint8_t* m = &out->mask[b * n_keys];
        double m_max = -INFINITY;
        for (size_t j = 0; j < n_keys; ++j) {
            if (!m[j]) { logits[j] = -INFINITY; continue; }
            double s = 0;
            for (size_t i = 0; i < hd; ++i) s += q[b * hd + i] * k[j * hd + i];
            logits[j] = s * scale;
            if (logits[j] > m_max) m_max = logits[j];
        }
        double den = 0;
        for (size_t j = 0; j < n_keys; ++j) {
            logits[j] = m[j] ? std::exp(logits[j] - m_max) : 0.0;
            den += logits[j];
        }
        if (den <= 0) return fail("no visible key for a block slot");
        for (size_t j = 0; j < n_keys; ++j) {
            const double p = logits[j] / den;
            if (p == 0.0) continue;
            for (size_t i = 0; i < hd; ++i) out->out[b * hd + i] += p * v[j * hd + i];
        }
    }
    return true;
}

namespace {
struct Err { double max_abs = 0, max_rel = 0; };
Err cmp(const double* a, const double* b, size_t n) {
    Err e;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs(a[i] - b[i]);
        e.max_abs = std::max(e.max_abs, d);
        e.max_rel = std::max(e.max_rel, d / std::max(std::fabs(a[i]), 1e-12));
    }
    return e;
}
}  // namespace

DiffReport diff(const Config& c, const std::vector<double>& ref_states, const std::vector<double>& got_states,
                const std::vector<double>& fc, const std::vector<double>* norm_w,
                const std::vector<double>* observed_ctx, const std::vector<uint8_t>* observed_mask, bool swa_layer,
                double tol) {
    DiffReport rep;
    rep.layers = c.layers;
    const size_t d = size_t(c.d);
    if (!validate(c, nullptr) || ref_states.size() != got_states.size() || ref_states.size() < size_t(c.n_states) * d) {
        rep.summary = "invalid input for diff";
        rep.first_bad_layer = c.layers.empty() ? -1 : c.layers.front();
        return rep;
    }
    for (int L : c.layers) {
        const Err e = cmp(&ref_states[size_t(L) * d], &got_states[size_t(L) * d], d);
        rep.layer_max_abs.push_back(e.max_abs);
        rep.layer_max_rel.push_back(e.max_rel);
        if (rep.first_bad_layer < 0 && e.max_rel > tol) rep.first_bad_layer = L;
    }
    if (!fc.empty()) {
        std::vector<double> ctx;
        if (build_context(c, ref_states, fc, norm_w, &ctx, nullptr) && observed_ctx && observed_ctx->size() == d) {
            const Err e = cmp(ctx.data(), observed_ctx->data(), d);
            rep.ctx_max_abs = e.max_abs;
            rep.ctx_max_rel = e.max_rel;
            rep.ctx_compared = true;
        }
    }
    if (observed_mask) {
        const std::vector<uint8_t> m = build_mask(c, swa_layer);
        if (observed_mask->size() != m.size()) {
            rep.mask_equal = false;
            rep.mask_mismatches = -1;
        } else {
            for (size_t i = 0; i < m.size(); ++i)
                if ((*observed_mask)[i] != m[i]) ++rep.mask_mismatches;
            rep.mask_equal = rep.mask_mismatches == 0;
        }
    }
    char buf[320];
    std::snprintf(buf, sizeof buf,
                  "dflash step: taps=%zu first_bad_layer=%d ctx_rel=%.3e masks=%s(%d)", rep.layers.size(),
                  rep.first_bad_layer, rep.ctx_max_rel, rep.mask_equal ? "equal" : "differ",
                  rep.mask_mismatches);
    rep.summary = buf;
    return rep;
}

}  // namespace vegajet::dflash_step
