// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/dflash_host.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>

namespace vegajet::dflash_host {

double rel_l2(const float* a, const float* b, int n) {
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        num += d * d;
        den += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    return std::sqrt(num) / std::max(std::sqrt(den), 1e-30);
}

// ---- 1. taps ---------------------------------------------------------------------------------------------------
bool gather_taps(const float* states, int n_states, int D, const std::vector<int>& taps, int shift, float* out,
                 std::string* err) {
    for (size_t k = 0; k < taps.size(); ++k) {
        const int s = taps[k] + shift;
        if (s < 0 || s >= n_states) {
            if (err) *err = "tap " + std::to_string(taps[k]) + " + shift " + std::to_string(shift) + " = state " +
                            std::to_string(s) + " is outside 0.." + std::to_string(n_states - 1);
            return false;
        }
        std::copy(states + static_cast<size_t>(s) * D, states + static_cast<size_t>(s + 1) * D, out + k * D);
    }
    return true;
}

ShiftScore score_tap_shift(const float* states, int n_tokens, int n_states, int D, const std::vector<int>& taps,
                           const float* observed, const std::vector<int>& candidates) {
    ShiftScore r;
    const size_t row = taps.size() * static_cast<size_t>(D);
    std::vector<float> buf(row);
    double best = std::numeric_limits<double>::infinity();
    for (int sh : candidates) {
        double num = 0.0, den = 0.0;
        bool ok = true;
        for (int t = 0; t < n_tokens && ok; ++t) {
            ok = gather_taps(states + static_cast<size_t>(t) * n_states * D, n_states, D, taps, sh, buf.data(), nullptr);
            const float* o = observed + static_cast<size_t>(t) * row;
            for (size_t i = 0; ok && i < row; ++i) {
                const double d = static_cast<double>(buf[i]) - o[i];
                num += d * d;
                den += static_cast<double>(o[i]) * o[i];
            }
        }
        const double e = ok ? std::sqrt(num) / std::max(std::sqrt(den), 1e-30) : std::numeric_limits<double>::infinity();
        r.shifts.push_back(sh);
        r.rel_err.push_back(e);
        if (e < best) { best = e; r.best_shift = sh; }
    }
    r.unambiguous = best < 1e-6;
    for (size_t i = 0; i < r.shifts.size(); ++i)
        if (r.shifts[i] != r.best_shift && r.rel_err[i] < 1e-3) r.unambiguous = false;
    return r;
}

// ---- 2. fc + hidden_norm ---------------------------------------------------------------------------------------
const char* norm_order_name(NormOrder o) {
    switch (o) {
        case NormOrder::FcNorm: return "hidden_norm(fc(cat))";
        case NormOrder::NormFc: return "fc(cat(norm(tap_k)))";
        case NormOrder::NormCatFc: return "fc(norm(cat))";
        case NormOrder::Fc: return "fc(cat) (no norm)";
    }
    return "?";
}

void rms_norm(const float* x, const float* w, int n, float eps, float* y) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
    const double inv = 1.0 / std::sqrt(ss / n + static_cast<double>(eps));
    for (int i = 0; i < n; ++i) y[i] = static_cast<float>(x[i] * inv * (w ? w[i] : 1.0f));
}

namespace {
void gemv(const float* m, const float* x, int rows, int cols, float* y) {
    for (int r = 0; r < rows; ++r) {
        double acc = 0.0;
        const float* row = m + static_cast<size_t>(r) * cols;
        for (int c = 0; c < cols; ++c) acc += static_cast<double>(row[c]) * x[c];
        y[r] = static_cast<float>(acc);
    }
}
}  // namespace

void project_context(const float* cat, const float* fc, const float* norm_w, int D, int n_taps, float eps,
                     NormOrder order, float* out) {
    const int K = n_taps * D;
    std::vector<float> tmp(static_cast<size_t>(K));
    switch (order) {
        case NormOrder::FcNorm: {
            std::vector<float> y(static_cast<size_t>(D));
            gemv(fc, cat, D, K, y.data());
            rms_norm(y.data(), norm_w, D, eps, out);
            return;
        }
        case NormOrder::NormFc:
            for (int k = 0; k < n_taps; ++k) rms_norm(cat + k * D, norm_w, D, eps, tmp.data() + k * D);
            gemv(fc, tmp.data(), D, K, out);
            return;
        case NormOrder::NormCatFc:
            rms_norm(cat, nullptr, K, eps, tmp.data());
            gemv(fc, tmp.data(), D, K, out);
            return;
        case NormOrder::Fc:
            gemv(fc, cat, D, K, out);
            return;
    }
}

OrderScore classify_norm_order(const float* cat, const float* fc, const float* norm_w, int D, int n_taps, float eps,
                               const float* observed) {
    OrderScore r;
    std::vector<float> y(static_cast<size_t>(D));
    double best = std::numeric_limits<double>::infinity();
    for (NormOrder o : {NormOrder::FcNorm, NormOrder::NormFc, NormOrder::NormCatFc, NormOrder::Fc}) {
        project_context(cat, fc, norm_w, D, n_taps, eps, o, y.data());
        const double e = rel_l2(y.data(), observed, D);
        r.rel_err.emplace_back(o, e);
        if (e < best) { best = e; r.best = o; }
    }
    return r;
}

// ---- 3. masks --------------------------------------------------------------------------------------------------
const char* block_mask_name(BlockMask m) {
    switch (m) {
        case BlockMask::ZLab: return "zlab (causal on SWA layers, bidirectional on full layers)";
        case BlockMask::Bidirectional: return "bidirectional on every layer";
        case BlockMask::Causal: return "causal on every layer";
    }
    return "?";
}
const char* window_rule_name(WindowRule r) { return r == WindowRule::Lt ? "q - k < W" : "q - k <= W"; }

bool mask_allowed(const MaskSpec& s, int slot, int key_pos) {
    const int q = s.block_start + slot;
    if (s.swa_layer && s.window > 0) {
        const int dist = q - key_pos;
        const bool in_window = s.rule == WindowRule::Lt ? dist < s.window : dist <= s.window;
        if (!in_window) return false;
    }
    if (key_pos < s.block_start) return true;  // context
    const bool causal = s.mode == BlockMask::Causal || (s.mode == BlockMask::ZLab && s.swa_layer);
    return !causal || key_pos <= q;
}

std::vector<uint8_t> build_mask(const MaskSpec& s) {
    const int n_keys = s.block_start + s.block_len;
    std::vector<uint8_t> m(static_cast<size_t>(s.block_len) * n_keys);
    for (int i = 0; i < s.block_len; ++i)
        for (int k = 0; k < n_keys; ++k) m[static_cast<size_t>(i) * n_keys + k] = mask_allowed(s, i, k) ? 1 : 0;
    return m;
}

MaskDiff diff_mask(const MaskSpec& expected, const std::vector<uint8_t>& observed) {
    MaskDiff d;
    const int n_keys = expected.block_start + expected.block_len;
    if (observed.size() != static_cast<size_t>(expected.block_len) * n_keys) {
        d.mismatches = -1;
        d.summary = "observed mask has " + std::to_string(observed.size()) + " entries, expected " +
                    std::to_string(expected.block_len) + " x " + std::to_string(n_keys) +
                    " (block x [context | block]); a different context length (e.g. the anchor counted as context) "
                    "or block size";
        return d;
    }
    const auto want = build_mask(expected);
    for (size_t i = 0; i < want.size(); ++i) {
        if (want[i] == observed[i]) continue;
        if (d.mismatches++ == 0) {
            d.first_slot = static_cast<int>(i / n_keys);
            d.first_key = static_cast<int>(i % n_keys);
        }
    }
    if (d.mismatches == 0) {
        d.matched_variant = true;
        d.variant_mode = expected.mode;
        d.variant_rule = expected.rule;
        d.summary = "mask matches the expected " + std::string(block_mask_name(expected.mode)) + ", " +
                    window_rule_name(expected.rule);
        return d;
    }
    for (BlockMask mode : {BlockMask::ZLab, BlockMask::Bidirectional, BlockMask::Causal}) {
        for (WindowRule rule : {WindowRule::Lt, WindowRule::Le}) {
            MaskSpec v = expected;
            v.mode = mode;
            v.rule = rule;
            if (build_mask(v) == observed) {
                d.matched_variant = true;
                d.variant_mode = mode;
                d.variant_rule = rule;
            }
        }
    }
    char b[256];
    std::snprintf(b, sizeof(b), "%d mismatches, first at block slot %d (query pos %d) / key pos %d (%s)", d.mismatches,
                  d.first_slot, expected.block_start + d.first_slot, d.first_key,
                  d.first_key < expected.block_start ? "context" : "block");
    d.summary = b;
    if (d.matched_variant)
        d.summary += "; observed equals " + std::string(block_mask_name(d.variant_mode)) + ", " +
                     window_rule_name(d.variant_rule) + (expected.swa_layer ? " on this SWA layer" : " on this full layer");
    else
        d.summary += "; matches no known variant";
    return d;
}

// ---- 3b. engine block-mask plan --------------------------------------------------------------------------------
const char* block_mask_mode_name(BlockMaskMode m) {
    switch (m) {
        case BlockMaskMode::Bidir: return "bidir";
        case BlockMaskMode::ZLab: return "zlab";
        case BlockMaskMode::Causal: return "causal";
    }
    return "?";
}

bool parse_block_mask_mode(const char* text, BlockMaskMode* out) {
    if (out == nullptr) return false;
    if (text == nullptr || text[0] == '\0') {
        *out = BlockMaskMode::Bidir;
        return true;
    }
    if (std::strcmp(text, "bidir") == 0) { *out = BlockMaskMode::Bidir; return true; }
    if (std::strcmp(text, "zlab") == 0) { *out = BlockMaskMode::ZLab; return true; }
    if (std::strcmp(text, "causal") == 0) { *out = BlockMaskMode::Causal; return true; }
    return false;
}

DraftBlockMaskPlan plan_draft_block_mask(const DraftBlockMaskConfig& c) {
    DraftBlockMaskPlan p;
    char buf[256];
    if (c.mode == BlockMaskMode::Bidir) {
        p.ok = true;
        p.summary = "mode=bidir (shipped: bidirectional block on every layer, no arm)";
        return p;
    }
    const std::string mode = block_mask_mode_name(c.mode);
    if (!c.pattern_known) {
        p.error = "mode=" + mode +
                  " needs dflash.attention.sliding_window_pattern (one entry per drafter layer) to tell the "
                  "sliding_attention layers from the full one; unknown pattern: fail closed";
        return p;
    }
    if (c.n_layer == 0u || c.n_layer > 32u) {
        p.error = "mode=" + mode + " needs 1..32 drafter layers, got " + std::to_string(c.n_layer);
        return p;
    }
    if (c.block_len == 0u || c.block_len > 32u) {
        p.error = "block_len (B) out of range 1..32: " + std::to_string(c.block_len);
        return p;
    }
    const uint32_t known = (c.n_layer == 32u) ? 0xFFFFFFFFu : ((1u << c.n_layer) - 1u);
    if ((c.swa_layer_mask & ~known) != 0u) {
        snprintf(buf, sizeof(buf), "swa_layer_mask=0x%x has bits outside n_layer=%u",
                 static_cast<unsigned>(c.swa_layer_mask), static_cast<unsigned>(c.n_layer));
        p.error = buf;
        return p;
    }
    if (c.swa_window == 0u) {
        p.error = "attention.sliding_window is 0: the short-context boundary would be empty (mode=" + mode + ")";
        return p;
    }
    const uint32_t swa = c.swa_layer_mask & known;
    const uint32_t masked = (c.mode == BlockMaskMode::Causal) ? known : swa;
    const uint64_t last_query = static_cast<uint64_t>(c.block_start) + static_cast<uint64_t>(c.block_len);
    p.arm = last_query <= static_cast<uint64_t>(c.swa_window);
    p.causal_mask = p.arm ? masked : 0u;
    p.ok = true;
    snprintf(buf, sizeof(buf),
             "mode=%s arm=%d s_pos=%u B=%u W=%u causal_layers=0x%x (pattern %u swa of 0x%x, %u full)",
             mode.c_str(), p.arm ? 1 : 0, static_cast<unsigned>(c.block_start), static_cast<unsigned>(c.block_len),
             static_cast<unsigned>(c.swa_window), static_cast<unsigned>(p.causal_mask),
             static_cast<unsigned>(__builtin_popcount(swa)), static_cast<unsigned>(swa),
             static_cast<unsigned>(__builtin_popcount(known & ~swa)));
    p.summary = buf;
    return p;
}

void masked_attention(const float* q, const float* k, const float* v, const std::vector<uint8_t>& mask, int B,
                      int n_keys, int hd, float* out) {
    const double scale = 1.0 / std::sqrt(static_cast<double>(hd));
    std::vector<double> s(static_cast<size_t>(n_keys));
    for (int i = 0; i < B; ++i) {
        double mx = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < n_keys; ++j) {
            if (!mask[static_cast<size_t>(i) * n_keys + j]) continue;
            double dot = 0.0;
            for (int c = 0; c < hd; ++c) dot += static_cast<double>(q[i * hd + c]) * k[static_cast<size_t>(j) * hd + c];
            s[j] = dot * scale;
            mx = std::max(mx, s[j]);
        }
        double sum = 0.0;
        std::vector<double> acc(static_cast<size_t>(hd), 0.0);
        for (int j = 0; j < n_keys; ++j) {
            if (!mask[static_cast<size_t>(i) * n_keys + j]) continue;
            const double w = std::exp(s[j] - mx);
            sum += w;
            for (int c = 0; c < hd; ++c) acc[c] += w * v[static_cast<size_t>(j) * hd + c];
        }
        for (int c = 0; c < hd; ++c) out[i * hd + c] = sum > 0.0 ? static_cast<float>(acc[c] / sum) : 0.0f;
    }
}

// ---- 4. acceptance ---------------------------------------------------------------------------------------------
int accepted_prefix(const Cycle& c) {
    const size_t n = std::min(c.draft.size(), c.target.size());
    size_t a = 0;
    while (a < n && c.draft[a] == c.target[a]) ++a;
    return static_cast<int>(a);
}

const char* gate_name(Gate g) {
    switch (g) {
        case Gate::Pass: return "PASS";
        case Gate::Marginal: return "MARGINAL";
        case Gate::Abort: return "ABORT";
        case Gate::NoData: return "NO_DATA";
    }
    return "?";
}

Gate gate_of(double mean_accepted, int full_cycles) {
    if (full_cycles <= 0) return Gate::NoData;
    if (mean_accepted >= kGatePass) return Gate::Pass;
    if (mean_accepted < kGateAbort) return Gate::Abort;
    return Gate::Marginal;
}

AcceptanceReport compare_cycles(const std::vector<Cycle>& engine, const std::vector<Cycle>& reference, int block_len) {
    AcceptanceReport r;
    r.block_len = block_len;
    r.engine_cycles = static_cast<int>(engine.size());
    r.reference_cycles = static_cast<int>(reference.size());
    r.engine_hist.assign(static_cast<size_t>(std::max(block_len, 1)), 0);
    const size_t need = static_cast<size_t>(std::max(block_len - 1, 0));
    auto full = [&](const Cycle& c) { return c.draft.size() >= need && c.target.size() >= need; };
    double se = 0.0, sr = 0.0;
    for (const auto& c : engine) {
        if (!full(c)) continue;
        const int a = std::min(accepted_prefix(c), block_len - 1);
        se += a;
        ++r.engine_full;
        ++r.engine_hist[static_cast<size_t>(a)];
    }
    for (const auto& c : reference) {
        if (!full(c)) continue;
        sr += std::min(accepted_prefix(c), block_len - 1);
        ++r.reference_full;
    }
    r.engine_mean = r.engine_full ? se / r.engine_full : 0.0;
    r.reference_mean = r.reference_full ? sr / r.reference_full : 0.0;
    r.engine_gate = gate_of(r.engine_mean, r.engine_full);
    r.reference_gate = gate_of(r.reference_mean, r.reference_full);
    r.flagged = r.engine_full > 0 && r.engine_mean < kGateAbort;

    std::map<int, const Cycle*> ref_at;
    for (const auto& c : reference) ref_at.emplace(c.pos, &c);
    std::vector<const Cycle*> eng(engine.size());
    for (size_t i = 0; i < engine.size(); ++i) eng[i] = &engine[i];
    std::stable_sort(eng.begin(), eng.end(), [](const Cycle* a, const Cycle* b) { return a->pos < b->pos; });
    for (const Cycle* e : eng) {
        const auto it = ref_at.find(e->pos);
        if (it == ref_at.end()) continue;
        const Cycle& f = *it->second;
        CycleDiff d;
        d.pos = e->pos;
        d.engine_accepted = accepted_prefix(*e);
        d.reference_accepted = accepted_prefix(f);
        const size_t n = std::min(e->draft.size(), f.draft.size());
        for (size_t i = 0; i < n; ++i)
            if (e->draft[i] != f.draft[i]) { d.first_slot = static_cast<int>(i); break; }
        if (r.first_divergent_cycle < 0 && (d.first_slot >= 0 || d.engine_accepted != d.reference_accepted))
            r.first_divergent_cycle = static_cast<int>(r.cycles.size());
        r.cycles.push_back(d);
    }
    r.aligned = static_cast<int>(r.cycles.size());

    char b[512];
    std::snprintf(b, sizeof(b), "B=%d engine mean_accepted_drafts=%.3f over %d full cycles (%s) vs reference %.3f over %d (%s)",
                  block_len, r.engine_mean, r.engine_full, gate_name(r.engine_gate), r.reference_mean, r.reference_full,
                  gate_name(r.reference_gate));
    r.summary = b;
    if (block_len != 8) r.summary += "; the Issue #152 gate is defined for B=8";
    if (r.first_divergent_cycle >= 0) {
        const CycleDiff& d = r.cycles[static_cast<size_t>(r.first_divergent_cycle)];
        std::snprintf(b, sizeof(b), "; first divergence at pos %d (aligned cycle %d), draft slot %d, accepted %d vs %d",
                      d.pos, r.first_divergent_cycle, d.first_slot, d.engine_accepted, d.reference_accepted);
        r.summary += b;
    } else if (r.aligned > 0) {
        r.summary += "; engine drafts equal the reference on all " + std::to_string(r.aligned) + " aligned cycles";
    }
    if (r.flagged) r.summary += "; FLAG: engine below " + std::to_string(kGateAbort).substr(0, 3);
    return r;
}

LayerDivergence first_layer_divergence(const float* engine, const float* reference, int n_layers, int B, int D,
                                       double tol) {
    LayerDivergence r;
    r.layer_max.assign(static_cast<size_t>(std::max(n_layers, 0)), 0.0);
    for (int l = 0; l < n_layers; ++l) {
        for (int s = 0; s < B; ++s) {
            const size_t off = (static_cast<size_t>(l) * B + s) * D;
            const double e = rel_l2(engine + off, reference + off, D);
            r.layer_max[static_cast<size_t>(l)] = std::max(r.layer_max[static_cast<size_t>(l)], e);
            if (r.layer < 0 && e > tol) {
                r.layer = l;
                r.slot = s;
                r.rel_err = e;
            }
        }
    }
    return r;
}

}  // namespace vegajet::dflash_host
