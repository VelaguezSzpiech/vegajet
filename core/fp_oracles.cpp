// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/fp_oracles.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>

namespace vegajet::oracles {

// ------------------------------------------------------------------------------------------------- FP16 helpers
uint16_t f32_to_f16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127;
    uint32_t man = bits & 0x7FFFFFu;
    if (((bits >> 23) & 0xFFu) == 0xFFu) {  // inf / nan
        return static_cast<uint16_t>(sign | 0x7C00u | (man ? 0x200u : 0u));
    }
    if (exp > 15) return static_cast<uint16_t>(sign | 0x7C00u);  // overflow -> inf
    if (exp >= -14) {                                            // normal
        uint32_t m = man >> 13;
        uint32_t e = static_cast<uint32_t>(exp + 15) << 10;
        const uint32_t rem = man & 0x1FFFu, half = 0x1000u;
        if (rem > half || (rem == half && (m & 1u))) {
            ++m;
            if (m == 0x400u) { m = 0; e += 0x400u; if ((e >> 10) >= 31u) return static_cast<uint16_t>(sign | 0x7C00u); }
        }
        return static_cast<uint16_t>(sign | e | m);
    }
    if (exp < -25) return static_cast<uint16_t>(sign);  // underflow to signed zero
    // Subnormal: value = sig * 2^(exp - 23) with sig = man | 0x800000; half units are 2^-24, so shift by -exp-1.
    const uint32_t sig = man | 0x800000u;
    const uint32_t sh = static_cast<uint32_t>(-exp - 1);  // exp = -15 -> 14, exp = -25 -> 24
    uint32_t q = sig >> sh;
    const uint32_t rem = sig & ((1u << sh) - 1u), half = 1u << (sh - 1);
    if (rem > half || (rem == half && (q & 1u))) ++q;
    if (q >= 0x400u) return static_cast<uint16_t>(sign | 0x0400u);  // rounded up to the smallest normal
    return static_cast<uint16_t>(sign | q);
}

float f16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; --exp; }
            man &= 0x3FFu;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

const char* precision_name(Precision p) {
    switch (p) {
        case Precision::F64: return "f64";
        case Precision::F32: return "f32";
        case Precision::F16: return "f16";
    }
    return "?";
}

// ------------------------------------------------------------------------------------------------- error stats
void ErrStats::add(double ref, double got, double floor) {
    const double d = std::fabs(ref - got);
    const double r = d / std::max(std::fabs(ref), floor);
    if (n == 0 || d > max_abs) { max_abs = d; at_abs = n; }
    if (n == 0 || r > max_rel) { max_rel = r; at_rel = n; }
    num_sq += d * d;
    den_sq += ref * ref;
    ++n;
}

void ErrStats::merge(const ErrStats& o) {
    if (o.n == 0) return;
    if (n == 0 || o.max_abs > max_abs) { max_abs = o.max_abs; at_abs = o.at_abs + n; }
    if (n == 0 || o.max_rel > max_rel) { max_rel = o.max_rel; at_rel = o.at_rel + n; }
    num_sq += o.num_sq;
    den_sq += o.den_sq;
    n += o.n;
}

ErrStats compare(const double* ref, const double* got, size_t n, double floor) {
    ErrStats e;
    for (size_t i = 0; i < n; ++i) e.add(ref[i], got[i], floor);
    return e;
}

std::string format_err(const char* label, const ErrStats& e) {
    char buf[224];
    std::snprintf(buf, sizeof buf, "%s: n=%zu max_abs=%.3e max_rel=%.3e l2_rel=%.3e", label, e.n, e.max_abs,
                  e.max_rel, e.rel_l2());
    return buf;
}

namespace {

// ---- precision arithmetic: every operation is rounded to T ------------------------------------------------
struct H16 { uint16_t bits; };
inline double to_d(double v) { return v; }
inline double to_d(float v) { return v; }
inline double to_d(H16 v) { return f16_to_f32(v.bits); }

template <class T> T rn(double v);
template <> double rn<double>(double v) { return v; }
template <> float rn<float>(double v) { return static_cast<float>(v); }
template <> H16 rn<H16>(double v) { return H16{f32_to_f16(static_cast<float>(v))}; }

template <class T> T add(T a, T b) { return rn<T>(to_d(a) + to_d(b)); }
template <class T> T sub(T a, T b) { return rn<T>(to_d(a) - to_d(b)); }
template <class T> T mul(T a, T b) { return rn<T>(to_d(a) * to_d(b)); }
template <class T> T divv(T a, T b) { return rn<T>(to_d(a) / to_d(b)); }

template <class T> T dot(const T* x, const T* y, uint32_t n) {
    T s = rn<T>(0.0);
    for (uint32_t i = 0; i < n; ++i) s = add(s, mul(x[i], y[i]));
    return s;
}

// ---- GDN verify, one pass over the chain, templated on the arithmetic precision ---------------------------
template <class T> struct GdnRun {
    uint32_t dk = 0, dv = 0, M = 0;
    std::vector<T> S0, q, k, v, a, b;
    std::vector<T> out;                       // M x dv
    std::vector<std::vector<T>> state;        // (M+1) x dv*dk
    std::vector<uint32_t> writes;             // dv*dk, final-state write counters
};

template <class T> GdnRun<T> to_prec(const GdnProblem& p) {
    GdnRun<T> r;
    r.dk = p.dk; r.dv = p.dv; r.M = p.M;
    auto cp = [](const std::vector<double>& s, std::vector<T>& d) {
        d.resize(s.size());
        for (size_t i = 0; i < s.size(); ++i) d[i] = rn<T>(s[i]);
    };
    cp(p.S0, r.S0); cp(p.q, r.q); cp(p.k, r.k); cp(p.v, r.v); cp(p.a, r.a); cp(p.b, r.b);
    return r;
}

// Chunked WY verify. g_t = prod_{i<=t} a_i; A_tj = (g_t/g_j) b_j (k_j . k_t); (I + A) E = V - diag(g) K S0^T.
// The state after accepting k tokens is S_k = g_k S0 + sum_{j<=k} (g_k/g_j) b_j E_j k_j^T, so every prefix state
// comes out of the same pass (rollback without re-running the recurrence).
template <class T> GdnRun<T> verify_chunk(GdnRun<T> r) {
    const uint32_t M = r.M, dk = r.dk, dv = r.dv;
    std::vector<T> g(M);
    for (uint32_t t = 0; t < M; ++t) g[t] = mul(r.a[t], t ? g[t - 1] : rn<T>(1.0));

    std::vector<T> E(size_t(M) * dv, rn<T>(0.0));
    std::vector<T> O(size_t(M) * dv, rn<T>(0.0));
    // kk[j*M+t] = k_j . k_t, kq[j*M+t] = k_j . q_t (lower-triangular use only).
    std::vector<T> kk(size_t(M) * M, rn<T>(0.0)), kq(size_t(M) * M, rn<T>(0.0));
    for (uint32_t j = 0; j < M; ++j) {
        for (uint32_t t = j; t < M; ++t) kk[size_t(j) * M + t] = dot(&r.k[size_t(j) * dk], &r.k[size_t(t) * dk], dk);
        for (uint32_t t = j; t < M; ++t) kq[size_t(j) * M + t] = dot(&r.k[size_t(j) * dk], &r.q[size_t(t) * dk], dk);
    }
    for (uint32_t t = 0; t < M; ++t) {
        const T gt = g[t];
        for (uint32_t row = 0; row < dv; ++row) {
            T e = sub(r.v[size_t(t) * dv + row],
                      mul(gt, dot(&r.S0[size_t(row) * dk], &r.k[size_t(t) * dk], dk)));
            for (uint32_t j = 0; j < t; ++j) {
                const T f = mul(mul(divv(gt, g[j]), r.b[j]), kk[size_t(j) * M + t]);
                e = sub(e, mul(f, E[size_t(j) * dv + row]));
            }
            E[size_t(t) * dv + row] = e;
            T o = mul(gt, dot(&r.S0[size_t(row) * dk], &r.q[size_t(t) * dk], dk));
            for (uint32_t j = 0; j <= t; ++j) {
                const T f = mul(mul(divv(gt, g[j]), r.b[j]), kq[size_t(j) * M + t]);
                o = add(o, mul(f, E[size_t(j) * dv + row]));
            }
            O[size_t(t) * dv + row] = o;
        }
    }
    r.out = O;
    r.state.assign(size_t(M) + 1, std::vector<T>(size_t(dv) * dk, rn<T>(0.0)));
    r.state[0] = r.S0;
    r.writes.assign(size_t(dv) * dk, 0u);
    for (uint32_t kk_ = 1; kk_ <= M; ++kk_) {  // prefix length a = 1..M
        const uint32_t a = kk_ - 1;
        for (uint32_t row = 0; row < dv; ++row) {
            for (uint32_t i = 0; i < dk; ++i) {
                T s = mul(g[a], r.S0[size_t(row) * dk + i]);
                for (uint32_t j = 0; j <= a; ++j) {
                    const T f = mul(mul(divv(g[a], g[j]), r.b[j]), E[size_t(j) * dv + row]);
                    s = add(s, mul(f, r.k[size_t(j) * dk + i]));
                }
                r.state[kk_][size_t(row) * dk + i] = s;  // the only write to this element
                if (kk_ == M) ++r.writes[size_t(row) * dk + i];
            }
        }
    }
    return r;
}

std::vector<double> to_d_vec(const std::vector<H16>& s) {
    std::vector<double> d(s.size());
    for (size_t i = 0; i < s.size(); ++i) d[i] = f16_to_f32(s[i].bits);
    return d;
}
std::vector<double> to_d_vec(const std::vector<float>& s) {
    std::vector<double> d(s.size());
    for (size_t i = 0; i < s.size(); ++i) d[i] = s[i];
    return d;
}

template <class T> GdnPrecisionReport run_precision(const GdnProblem& p, Precision prec,
                                                    const std::vector<std::vector<double>>& seq_state,
                                                    const std::vector<double>& seq_out) {
    GdnRun<T> r = verify_chunk(to_prec<T>(p));
    GdnPrecisionReport rep;
    rep.prec = prec;
    const std::vector<double> out = to_d_vec(r.out);
    rep.out_err = compare(seq_out.data(), out.data(), out.size());
    for (uint32_t k = 0; k <= p.M; ++k) {
        const std::vector<double> st = to_d_vec(r.state[k]);
        rep.state_err.merge(compare(seq_state[k].data(), st.data(), st.size()));
    }
    return rep;
}

}  // namespace

// ------------------------------------------------------------------------------------------------- GDN
GdnProblem gdn_make(uint32_t dk, uint32_t dv, uint32_t M, uint64_t seed) {
    std::mt19937_64 g(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> ua(0.85, 0.999), ub(0.05, 0.95);
    GdnProblem p;
    p.dk = dk; p.dv = dv; p.M = M;
    p.S0.resize(size_t(dv) * dk);
    for (auto& x : p.S0) x = 0.3 * nd(g);
    p.q.resize(size_t(M) * dk);
    p.k.resize(size_t(M) * dk);
    p.v.resize(size_t(M) * dv);
    for (auto& x : p.q) x = nd(g) / std::sqrt(double(dk));
    for (auto& x : p.v) x = nd(g);
    for (uint32_t t = 0; t < M; ++t) {
        double n = 0;
        for (uint32_t i = 0; i < dk; ++i) {
            const double x = nd(g);
            p.k[size_t(t) * dk + i] = x;
            n += x * x;
        }
        n = std::sqrt(n);
        for (uint32_t i = 0; i < dk; ++i) p.k[size_t(t) * dk + i] /= n;
        p.a.push_back(ua(g));
        p.b.push_back(ub(g));
    }
    return p;
}

std::vector<std::vector<double>> gdn_sequential(const GdnProblem& p, std::vector<double>* out) {
    std::vector<double> S = p.S0;
    std::vector<std::vector<double>> states;
    states.push_back(S);
    if (out) out->assign(size_t(p.M) * p.dv, 0.0);
    for (uint32_t t = 0; t < p.M; ++t) {
        const double* kt = &p.k[size_t(t) * p.dk];
        const double* vt = &p.v[size_t(t) * p.dv];
        for (uint32_t row = 0; row < p.dv; ++row) {
            double* Sr = &S[size_t(row) * p.dk];
            double sk = 0;
            for (uint32_t i = 0; i < p.dk; ++i) sk += Sr[i] * kt[i];
            const double e = vt[row] - p.a[t] * sk;  // v - a S k
            for (uint32_t i = 0; i < p.dk; ++i) Sr[i] = p.a[t] * Sr[i] + p.b[t] * e * kt[i];
        }
        states.push_back(S);
        if (out) {
            const double* qt = &p.q[size_t(t) * p.dk];
            for (uint32_t row = 0; row < p.dv; ++row) {
                const double* Sr = &S[size_t(row) * p.dk];
                double o = 0;
                for (uint32_t i = 0; i < p.dk; ++i) o += Sr[i] * qt[i];
                (*out)[size_t(t) * p.dv + row] = o;
            }
        }
    }
    return states;
}

GdnVerify gdn_verify(const GdnProblem& p, Precision prec) {
    GdnVerify v;
    if (prec == Precision::F32) {
        GdnRun<float> r = verify_chunk(to_prec<float>(p));
        v.out = to_d_vec(r.out);
        v.state.resize(r.state.size());
        for (size_t i = 0; i < r.state.size(); ++i) v.state[i] = to_d_vec(r.state[i]);
        v.writes = r.writes;
    } else if (prec == Precision::F16) {
        GdnRun<H16> r = verify_chunk(to_prec<H16>(p));
        v.out = to_d_vec(r.out);
        v.state.resize(r.state.size());
        for (size_t i = 0; i < r.state.size(); ++i) v.state[i] = to_d_vec(r.state[i]);
        v.writes = r.writes;
    } else {
        GdnRun<double> r = verify_chunk(to_prec<double>(p));
        v.out = r.out;
        v.state = r.state;
        v.writes = r.writes;
    }
    return v;
}

GdnOracleReport gdn_verify_oracle(uint32_t dk, uint32_t dv, uint32_t M, uint64_t seed) {
    const GdnProblem p = gdn_make(dk, dv, M, seed);
    std::vector<double> seq_out;
    const auto seq_state = gdn_sequential(p, &seq_out);
    const GdnVerify f64 = gdn_verify(p, Precision::F64);
    GdnOracleReport rep;
    rep.dk = dk; rep.dv = dv; rep.M = M;
    rep.algo_f64 = compare(seq_out.data(), f64.out.data(), f64.out.size());
    for (uint32_t k = 0; k <= M; ++k) rep.algo_f64.merge(compare(seq_state[k].data(), f64.state[k].data(), f64.state[k].size()));
    rep.by_precision.push_back(run_precision<float>(p, Precision::F32, seq_state, seq_out));
    rep.by_precision.push_back(run_precision<H16>(p, Precision::F16, seq_state, seq_out));
    rep.state_writes_min = *std::min_element(f64.writes.begin(), f64.writes.end());
    rep.state_writes_max = *std::max_element(f64.writes.begin(), f64.writes.end());
    rep.single_write = rep.state_writes_min == 1 && rep.state_writes_max == 1;
    std::string s = "GDN verify M=" + std::to_string(M) + " dk/dv=" + std::to_string(dk) + "/" + std::to_string(dv) +
                    ": f64 chunk vs sequential max_abs=" + std::to_string(rep.algo_f64.max_abs);
    for (const auto& e : rep.by_precision)
        s += std::string(" ") + precision_name(e.prec) + " out_abs=" + std::to_string(e.out_err.max_abs) +
             " out_l2=" + std::to_string(e.out_err.rel_l2()) + " state_abs=" +
             std::to_string(e.state_err.max_abs) + " state_l2=" + std::to_string(e.state_err.rel_l2());
    rep.summary = s;
    return rep;
}

GdnSweepReport gdn_verify_sweep(uint64_t seed) {
    GdnSweepReport sw;
    sw.worst.emplace_back();
    sw.worst[0].prec = Precision::F32;
    sw.worst.emplace_back();
    sw.worst[1].prec = Precision::F16;
    sw.single_write = true;
    for (const auto& dims : {std::pair<uint32_t, uint32_t>{32, 48}, std::pair<uint32_t, uint32_t>{64, 64}}) {
        for (uint32_t M = 5; M <= 8; ++M) {
            const GdnOracleReport r = gdn_verify_oracle(dims.first, dims.second, M, seed + M * 7 + dims.first);
            sw.algo_f64.merge(r.algo_f64);
            for (size_t i = 0; i < r.by_precision.size(); ++i) {
                sw.worst[i].out_err.merge(r.by_precision[i].out_err);
                sw.worst[i].state_err.merge(r.by_precision[i].state_err);
            }
            sw.single_write = sw.single_write && r.single_write;
            ++sw.cases;
        }
    }
    char buf[320];
    std::snprintf(buf, sizeof buf,
                  "GDN sweep: %zu cases (M=5..8, dk/dv 32/48 and 64/64), f64 abs=%.2e, f32 out/state l2=%.2e/%.2e, "
                  "f16 out/state l2=%.2e/%.2e, single_write=%s",
                  sw.cases, sw.algo_f64.max_abs, sw.worst[0].out_err.rel_l2(), sw.worst[0].state_err.rel_l2(),
                  sw.worst[1].out_err.rel_l2(), sw.worst[1].state_err.rel_l2(), sw.single_write ? "yes" : "no");
    sw.summary = buf;
    return sw;
}

// ------------------------------------------------------------------------------------------------- EVICT
namespace {

// Deterministic expert weights from (layer, expert): the "materialized" payload whose fetch dominates the pass.
struct ExpertW {
    std::vector<double> gate, up, down;
};

ExpertW materialize(uint32_t layer, uint32_t expert, uint32_t dim, uint64_t salt) {
    uint64_t s = (uint64_t(layer) << 32) ^ (uint64_t(expert) * 0x9E3779B97F4A7C15ull) ^ salt;
    auto next = [&s]() {
        s ^= s >> 33;
        s *= 0xFF51AFD7ED558CCDull;
        s ^= s >> 33;
        return double(s >> 11) / double(1ull << 53) * 2.0 - 1.0;
    };
    ExpertW w;
    w.gate.resize(dim);
    w.up.resize(dim);
    w.down.resize(dim);
    for (uint32_t d = 0; d < dim; ++d) w.gate[d] = next();
    for (uint32_t d = 0; d < dim; ++d) w.up[d] = next();
    for (uint32_t d = 0; d < dim; ++d) w.down[d] = next() * 0.5;
    return w;
}

// One routed expert applied to a token: out += weight * down * (tanh(gate.x) * (up.x)).
void apply_expert(const ExpertW& w, const std::vector<double>& x, double weight, std::vector<double>& acc) {
    double g = 0, u = 0;
    for (size_t d = 0; d < x.size(); ++d) {
        g += w.gate[d] * x[d];
        u += w.up[d] * x[d];
    }
    const double h = weight * std::tanh(g) * u;
    for (size_t d = 0; d < x.size(); ++d) acc[d] += w.down[d] * h;
}

}  // namespace

EvictProblem evict_make(uint32_t n_layers, uint32_t n_experts, uint32_t dim, uint32_t M, uint32_t top_k,
                        uint64_t seed) {
    std::mt19937_64 g(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    EvictProblem p;
    p.n_layers = n_layers; p.n_experts = n_experts; p.dim = dim; p.M = M;
    const uint32_t k = std::min(top_k, n_experts);
    for (uint32_t t = 0; t < M; ++t) {
        std::vector<uint32_t> rl, re;
        std::vector<double> rw;
        for (uint32_t l = 0; l < n_layers; ++l) {
            std::vector<uint32_t> ids(n_experts);
            for (uint32_t i = 0; i < n_experts; ++i) ids[i] = i;
            std::shuffle(ids.begin(), ids.end(), g);
            ids.resize(k);
            std::sort(ids.begin(), ids.end());
            for (uint32_t i = 0; i < k; ++i) {
                rl.push_back(l);
                re.push_back(ids[i]);
                rw.push_back(nd(g));
            }
        }
        p.route_layer.push_back(std::move(rl));
        p.route_expert.push_back(std::move(re));
        p.weight.push_back(std::move(rw));
        std::vector<double> x(dim);
        for (auto& v : x) v = nd(g);
        p.x.push_back(std::move(x));
    }
    return p;
}

EvictRun evict_naive(const EvictProblem& p) {
    EvictRun run;
    for (uint32_t t = 0; t < p.M; ++t) {
        std::vector<double> acc(p.dim, 0.0);
        for (size_t i = 0; i < p.route_layer[t].size(); ++i) {
            const ExpertW w = materialize(p.route_layer[t][i], p.route_expert[t][i], p.dim, 0x5EEDu);
            ++run.expert_loads;
            apply_expert(w, p.x[t], p.weight[t][i], acc);
        }
        run.out.push_back(std::move(acc));
    }
    return run;
}

EvictRun evict_dedupe(const EvictProblem& p) {
    EvictRun run;
    std::map<std::pair<uint32_t, uint32_t>, size_t> slot;
    std::vector<ExpertW> loaded;
    for (uint32_t t = 0; t < p.M; ++t) {
        std::vector<double> acc(p.dim, 0.0);
        for (size_t i = 0; i < p.route_layer[t].size(); ++i) {
            const std::pair<uint32_t, uint32_t> key{p.route_layer[t][i], p.route_expert[t][i]};
            auto it = slot.find(key);
            size_t idx;
            if (it == slot.end()) {
                idx = loaded.size();
                loaded.push_back(materialize(key.first, key.second, p.dim, 0x5EEDu));
                slot.emplace(key, idx);
                run.unique_order.push_back(key);
                ++run.expert_loads;
            } else {
                idx = it->second;
            }
            apply_expert(loaded[idx], p.x[t], p.weight[t][i], acc);
        }
        run.out.push_back(std::move(acc));
    }
    return run;
}

EvictOracleReport evict_oracle(const EvictProblem& p) {
    const EvictRun naive = evict_naive(p);
    const EvictRun dedupe = evict_dedupe(p);
    EvictOracleReport rep;
    rep.naive_loads = naive.expert_loads;
    rep.dedupe_loads = dedupe.expert_loads;
    rep.unique = dedupe.unique_order.size();
    for (uint32_t t = 0; t < p.M; ++t) {
        rep.occurrences += p.route_layer[t].size();
        rep.err.merge(compare(naive.out[t].data(), dedupe.out[t].data(), naive.out[t].size()));
    }
    rep.bitwise_equal = rep.err.max_abs == 0.0;
    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "EVICT dedup: occurrences=%llu unique=%llu naive_loads=%llu dedupe_loads=%llu max_abs=%.3e "
                  "bitwise=%s",
                  static_cast<unsigned long long>(rep.occurrences), static_cast<unsigned long long>(rep.unique),
                  static_cast<unsigned long long>(rep.naive_loads), static_cast<unsigned long long>(rep.dedupe_loads),
                  rep.err.max_abs, rep.bitwise_equal ? "yes" : "no");
    rep.summary = buf;
    return rep;
}

}  // namespace vegajet::oracles
