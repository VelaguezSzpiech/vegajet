// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/prefill_sched.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace vegajet::prefill_sched {

uint64_t scratch_for(uint32_t M, uint32_t C, const Config& cfg) {
    prefill::Request rq;
    rq.prompt_tokens = M;  // the planner sizes buffers by min(M, prompt); M tokens is the steady-state macro
    rq.macro_m = M;
    rq.gdn_chunk = C;
    rq.precision = cfg.precision;
    const prefill::Plan p = prefill::plan(cfg.dims, rq);
    if (!p.ok && !p.error.empty()) return 0;
    return cfg.aliased ? p.bytes_aliased : p.bytes_no_alias;
}

namespace {

void build(Schedule* s, uint32_t prompt_len, uint32_t pos0, uint32_t M, uint32_t C) {
    s->macros.clear();
    uint32_t idx = 0;
    for (uint32_t off = 0; off < prompt_len; off += M, ++idx) {
        MacroBatch mb;
        mb.index = idx;
        mb.pos0 = pos0 + off;
        mb.n = std::min(M, prompt_len - off);
        mb.kv_hi = mb.pos0 + mb.n;
        mb.final_partial = mb.n < M;
        mb.compute_logits = off + mb.n == prompt_len;
        for (uint32_t r = 0; r < mb.n; r += C) {
            SubChunk sc;
            sc.row0 = r;
            sc.pos0 = mb.pos0 + r;
            sc.n = std::min(C, mb.n - r);
            sc.state_in = sc.pos0;
            sc.state_out = sc.pos0 + sc.n;
            mb.gdn.push_back(sc);
        }
        s->macros.push_back(std::move(mb));
    }
}

}  // namespace

Schedule schedule(uint32_t prompt_len, uint32_t pos0, const Config& cfg) {
    Schedule s;
    auto legacy = [&](const std::string& why) {
        s.c2 = false;
        s.reason = why;
        const uint32_t L = std::max(cfg.legacy_chunk, 1u);
        s.macro_m = L;
        s.gdn_c = L;  // GDN un-split: one sub-chunk per macro batch
        s.scratch_bytes = 0;
        build(&s, prompt_len, pos0, L, L);
    };
    if (prompt_len == 0u) { legacy("empty prompt"); s.summary = "empty"; return s; }
    if (!cfg.enabled) legacy("C2 disabled");
    else if (cfg.macro_m != 256u && cfg.macro_m != 512u) legacy("macro M must be 256 or 512");
    else if (cfg.gdn_c != 16u && cfg.gdn_c != 32u) legacy("GDN sub-chunk C must be 16 or 32");
    else if (static_cast<uint64_t>(pos0) + prompt_len > 0xFFFFFFFFull) legacy("position overflow");
    else {
        // Largest requested shape that fits; 512 falls back to 256 before the current path.
        for (uint32_t M : {cfg.macro_m, 256u}) {
            if (M > cfg.macro_m) continue;
            const uint64_t need = scratch_for(M, cfg.gdn_c, cfg);
            if (need == 0u) continue;
            if (cfg.budget_bytes == 0u || need <= cfg.budget_bytes) {
                s.c2 = true;
                s.macro_m = M;
                s.gdn_c = cfg.gdn_c;
                s.scratch_bytes = need;
                if (M != cfg.macro_m) s.reason = "M=" + std::to_string(cfg.macro_m) + " scratch did not fit; using M=256";
                build(&s, prompt_len, pos0, M, cfg.gdn_c);
                break;
            }
        }
        if (!s.c2) {
            char b[160];
            std::snprintf(b, sizeof(b), "scratch %.1f MiB (M=256, C=%u) exceeds the %.1f MiB budget",
                          static_cast<double>(scratch_for(256u, cfg.gdn_c, cfg)) / 1048576.0, cfg.gdn_c,
                          static_cast<double>(cfg.budget_bytes) / 1048576.0);
            legacy(b);
        }
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s: %u tokens from pos %u -> %zu macro batches of M=%u, GDN C=%u, scratch %.1f MiB%s%s",
                  s.c2 ? "C2" : "current path", prompt_len, pos0, s.macros.size(), s.macro_m, s.gdn_c,
                  static_cast<double>(s.scratch_bytes) / 1048576.0, s.reason.empty() ? "" : " | ", s.reason.c_str());
    s.summary = buf;
    return s;
}

bool parse_c2_spec(const std::string& text, bool* enabled, uint32_t* macro_m, uint32_t* gdn_c, std::string* err) {
    *enabled = false;
    if (text.empty() || text == "0" || text == "off") return true;
    const size_t colon = text.find(':');
    const std::string m = text.substr(0, colon), c = colon == std::string::npos ? "32" : text.substr(colon + 1);
    if ((m != "256" && m != "512") || (c != "16" && c != "32")) {
        if (err) *err = "expected 256 or 512, optionally :16 or :32 (got '" + text + "')";
        return false;
    }
    *enabled = true;
    *macro_m = static_cast<uint32_t>(std::stoul(m));
    *gdn_c = static_cast<uint32_t>(std::stoul(c));
    return true;
}

bool verify_schedule(const Schedule& s, uint32_t prompt_len, uint32_t pos0, std::string* err) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    uint32_t next = pos0;
    for (size_t i = 0; i < s.macros.size(); ++i) {
        const MacroBatch& mb = s.macros[i];
        const std::string at = "macro " + std::to_string(i) + ": ";
        if (mb.index != i) return fail(at + "index out of order");
        if (mb.pos0 != next) return fail(at + "starts at " + std::to_string(mb.pos0) + ", expected " + std::to_string(next));
        if (mb.n == 0u || mb.n > s.macro_m) return fail(at + "size out of range");
        const bool last = i + 1u == s.macros.size();
        if (mb.n < s.macro_m && !last) return fail(at + "partial batch before the end");
        if (mb.final_partial != (mb.n < s.macro_m)) return fail(at + "final_partial flag");
        if (mb.compute_logits != last) return fail(at + "logits flag");
        if (mb.kv_hi != mb.pos0 + mb.n) return fail(at + "kv_hi must be pos0 + n (causal boundary)");
        uint32_t row = 0, st = mb.pos0;
        for (size_t j = 0; j < mb.gdn.size(); ++j) {
            const SubChunk& c = mb.gdn[j];
            const std::string sat = at + "sub-chunk " + std::to_string(j) + ": ";
            if (c.row0 != row || c.pos0 != mb.pos0 + row) return fail(sat + "not contiguous");
            if (c.n == 0u || c.n > s.gdn_c) return fail(sat + "size out of range");
            if (c.n < s.gdn_c && j + 1u != mb.gdn.size()) return fail(sat + "partial sub-chunk before the end of the batch");
            if (c.state_in != st || c.state_out != c.state_in + c.n) return fail(sat + "recurrent state hand-off broken");
            st = c.state_out;
            row += c.n;
        }
        if (row != mb.n) return fail(at + "sub-chunks cover " + std::to_string(row) + " of " + std::to_string(mb.n) + " rows");
        next += mb.n;
    }
    if (next != pos0 + prompt_len) return fail("schedule covers " + std::to_string(next - pos0) + " of " + std::to_string(prompt_len) + " tokens");
    return true;
}

// ---- toy reference ---------------------------------------------------------------------------------------------
namespace {

double hash01(uint64_t a, uint64_t b) {
    uint64_t x = a * 0x9E3779B97F4A7C15ULL ^ (b + 0x632BE59BD9B4E019ULL);
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return static_cast<double>(x >> 11) / 9007199254740992.0;
}

struct Tok {
    std::vector<double> x, q, k, v;
    double g = 1.0, beta = 0.0;
};

// Dense input projection: depends on token id and absolute position (so a wrong position changes everything).
Tok embed(const ToyModel& m, int32_t tok, uint32_t pos) {
    Tok t;
    t.x.resize(m.d); t.q.resize(m.d); t.k.resize(m.d); t.v.resize(m.d);
    for (uint32_t i = 0; i < m.d; ++i) {
        const double ph = std::sin(0.01 * pos * (i + 1));
        t.x[i] = hash01(m.seed ^ static_cast<uint64_t>(tok), i) - 0.5 + 0.1 * ph;
        t.q[i] = hash01(m.seed + 1u, static_cast<uint64_t>(tok) * 64u + i) - 0.5 + 0.1 * ph;
        t.k[i] = hash01(m.seed + 2u, static_cast<uint64_t>(tok) * 64u + i) - 0.5 - 0.1 * ph;
        t.v[i] = hash01(m.seed + 3u, static_cast<uint64_t>(tok) * 64u + i) - 0.5;
    }
    t.g = 0.90 + 0.09 * hash01(m.seed + 4u, static_cast<uint64_t>(tok) * 131u + pos);
    t.beta = 0.2 + 0.5 * hash01(m.seed + 5u, static_cast<uint64_t>(tok));
    return t;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

// Attention projections of h1 (key / value / query mixes of the residual).
void attn_proj(const ToyModel& m, const std::vector<double>& h1, std::vector<double>* q, std::vector<double>* k, std::vector<double>* v) {
    q->assign(m.d, 0.0); k->assign(m.d, 0.0); v->assign(m.d, 0.0);
    for (uint32_t i = 0; i < m.d; ++i) {
        const uint32_t j = (i + 1u) % m.d;
        (*q)[i] = h1[i] + 0.5 * h1[j];
        (*k)[i] = h1[i] - 0.25 * h1[j];
        (*v)[i] = 0.75 * h1[i] + 0.1 * h1[j];
    }
}

std::vector<double> attend(const std::vector<double>& q, const ToyState& st, uint32_t hi) {
    double mx = -1e300;
    std::vector<double> sc(hi);
    for (uint32_t j = 0; j < hi; ++j) { sc[j] = dot(q, st.keys[j]); mx = std::max(mx, sc[j]); }
    double z = 0.0;
    std::vector<double> o(q.size(), 0.0);
    for (uint32_t j = 0; j < hi; ++j) {
        const double w = std::exp(sc[j] - mx);
        z += w;
        for (size_t i = 0; i < o.size(); ++i) o[i] += w * st.vals[j][i];
    }
    for (auto& x : o) x /= z;
    return o;
}

std::vector<double> head(const std::vector<double>& h2) {
    std::vector<double> o(h2.size());
    for (size_t i = 0; i < h2.size(); ++i) o[i] = std::tanh(h2[i] + 0.3 * h2[(i + 2) % h2.size()]);
    return o;
}

}  // namespace

std::vector<std::vector<double>> toy_sequential(const ToyModel& m, const std::vector<int32_t>& toks, uint32_t pos0, ToyState* st) {
    if (st->S.empty()) st->S.assign(static_cast<size_t>(m.d) * m.d, 0.0);
    std::vector<std::vector<double>> out;
    for (size_t t = 0; t < toks.size(); ++t) {
        const uint32_t pos = pos0 + static_cast<uint32_t>(t);
        const Tok e = embed(m, toks[t], pos);
        // S = g S + beta k v^T ; o = S^T q
        for (uint32_t a = 0; a < m.d; ++a)
            for (uint32_t b = 0; b < m.d; ++b) st->S[a * m.d + b] = e.g * st->S[a * m.d + b] + e.beta * e.k[a] * e.v[b];
        std::vector<double> h1 = e.x;
        for (uint32_t b = 0; b < m.d; ++b) { double o = 0.0; for (uint32_t a = 0; a < m.d; ++a) o += st->S[a * m.d + b] * e.q[a]; h1[b] += o; }
        ++st->absorbed;
        std::vector<double> q, k, v;
        attn_proj(m, h1, &q, &k, &v);
        if (st->keys.size() <= pos) { st->keys.resize(pos + 1u); st->vals.resize(pos + 1u); }
        st->keys[pos] = k; st->vals[pos] = v;
        const std::vector<double> a = attend(q, *st, pos + 1u);
        for (uint32_t i = 0; i < m.d; ++i) h1[i] += a[i];
        out.push_back(head(h1));
    }
    return out;
}

std::vector<std::vector<double>> toy_scheduled(const ToyModel& m, const std::vector<int32_t>& toks, const Schedule& s, ToyState* st) {
    if (st->S.empty()) st->S.assign(static_cast<size_t>(m.d) * m.d, 0.0);
    std::vector<std::vector<double>> out(toks.size());
    const uint32_t base = s.macros.empty() ? 0u : s.macros.front().pos0;
    for (const MacroBatch& mb : s.macros) {
        // 1. dense input projection for the whole macro batch
        std::vector<Tok> e(mb.n);
        for (uint32_t r = 0; r < mb.n; ++r) e[r] = embed(m, toks[mb.pos0 - base + r], mb.pos0 + r);
        std::vector<std::vector<double>> h1(mb.n);
        for (uint32_t r = 0; r < mb.n; ++r) h1[r] = e[r].x;
        // 2. GDN: ordered sub-chunks in chunk-parallel form, each continuing the previous state
        for (const SubChunk& c : mb.gdn) {
            const uint32_t n = c.n, r0 = c.row0;
            // decay products G[t][s] = prod_{r = s+1 .. t} g_r (s < t), 1 at s = t; Gin[t] = prod_{r = 0 .. t} g_r
            std::vector<double> gin(n);
            double acc = 1.0;
            for (uint32_t t = 0; t < n; ++t) { acc *= e[r0 + t].g; gin[t] = acc; }
            for (uint32_t t = 0; t < n; ++t) {
                const Tok& qt = e[r0 + t];
                for (uint32_t b = 0; b < m.d; ++b) {
                    double o = 0.0;
                    for (uint32_t a = 0; a < m.d; ++a) o += st->S[a * m.d + b] * qt.q[a];
                    o *= gin[t];
                    for (uint32_t s2 = 0; s2 <= t; ++s2) {
                        double dec = 1.0;
                        for (uint32_t r = s2 + 1; r <= t; ++r) dec *= e[r0 + r].g;
                        o += dec * e[r0 + s2].beta * dot(e[r0 + s2].k, qt.q) * e[r0 + s2].v[b];
                    }
                    h1[r0 + t][b] += o;
                }
            }
            // state out = Gin[n-1] S + sum_s (prod_{r>s} g_r) beta_s k_s v_s^T
            std::vector<double> S2(st->S.size());
            for (uint32_t a = 0; a < m.d; ++a)
                for (uint32_t b = 0; b < m.d; ++b) {
                    double v = gin[n - 1u] * st->S[a * m.d + b];
                    for (uint32_t s2 = 0; s2 < n; ++s2) {
                        double dec = 1.0;
                        for (uint32_t r = s2 + 1; r < n; ++r) dec *= e[r0 + r].g;
                        v += dec * e[r0 + s2].beta * e[r0 + s2].k[a] * e[r0 + s2].v[b];
                    }
                    S2[a * m.d + b] = v;
                }
            st->S.swap(S2);
            st->absorbed = c.state_out;
        }
        // 3. attention: write this batch's KV, then each query sees keys [0, its own position]
        std::vector<std::vector<double>> qs(mb.n);
        for (uint32_t r = 0; r < mb.n; ++r) {
            std::vector<double> k, v;
            attn_proj(m, h1[r], &qs[r], &k, &v);
            const uint32_t pos = mb.pos0 + r;
            if (st->keys.size() <= pos) { st->keys.resize(pos + 1u); st->vals.resize(pos + 1u); }
            st->keys[pos] = k; st->vals[pos] = v;
        }
        for (uint32_t r = 0; r < mb.n; ++r) {
            const std::vector<double> a = attend(qs[r], *st, mb.pos0 + r + 1u);
            for (uint32_t i = 0; i < m.d; ++i) h1[r][i] += a[i];
            out[mb.pos0 - base + r] = head(h1[r]);
        }
    }
    return out;
}

}  // namespace vegajet::prefill_sched
