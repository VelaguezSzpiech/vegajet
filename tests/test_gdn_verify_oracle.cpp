// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU oracle for multi-token gated-delta-net verification (M = 5..12) against the sequential recurrence:
//   S_t = a_t S_{t-1} (I - b_t k_t k_t^T) + b_t v_t k_t^T,   o_t = S_t q_t        (S: dv x dk, keys L2-normalized)
// Chunked WY form (one pass over the M tokens, float):
//   g_t = prod_{i<=t} a_i,  A_tj = (g_t/g_j) b_j (k_j . k_t) for j < t,  (I + A) E = V - diag(g) K S_0^T
//   o_t = g_t S_0 q_t + sum_{j<=t} (g_t/g_j) b_j (k_j . q_t) e_j,   S_a = g_a S_0 + sum_{j<=a} (g_a/g_j) b_j e_j k_j^T
// Checked here:
//   * outputs and the state after every accepted prefix a = 1..M match the sequential recurrence (rollback without
//     re-running the chunk);
//   * Split-V: each value slice is solved independently (A depends only on k, b, g) and is bitwise identical to
//     the full solve;
//   * Single-Write: the verify pass writes every element of the final state exactly once;
//   * TreeWY: for a draft tree (parents precede children), "j < t" becomes "j is an ancestor of t"; every node's
//     output and state equal the sequential recurrence along its root path.
#include "tests/test_common.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

struct Problem {
    uint32_t dk = 0, dv = 0, M = 0;
    std::vector<float> S0;             // dv x dk
    std::vector<float> q, k, v;        // M x dk / M x dk / M x dv
    std::vector<float> a, b;           // M
    std::vector<int32_t> parent;       // M, -1 = root (continues S0); chain: parent[t] = t-1
};

Problem make(std::mt19937& g, uint32_t dk, uint32_t dv, uint32_t M, bool tree) {
    std::normal_distribution<float> nd(0, 1);
    std::uniform_real_distribution<float> ua(0.85f, 0.999f), ub(0.05f, 0.95f);
    Problem p{dk, dv, M, {}, {}, {}, {}, {}, {}, {}};
    p.S0.resize(size_t(dv) * dk);
    for (auto& x : p.S0) x = 0.3f * nd(g);
    p.q.resize(size_t(M) * dk); p.k.resize(size_t(M) * dk); p.v.resize(size_t(M) * dv);
    for (auto& x : p.q) x = nd(g) / std::sqrt(float(dk));
    for (auto& x : p.v) x = nd(g);
    for (uint32_t t = 0; t < M; ++t) {
        double n = 0;
        for (uint32_t i = 0; i < dk; ++i) { float x = nd(g); p.k[t * dk + i] = x; n += double(x) * x; }
        for (uint32_t i = 0; i < dk; ++i) p.k[t * dk + i] = static_cast<float>(p.k[t * dk + i] / std::sqrt(n));
        p.a.push_back(ua(g));
        p.b.push_back(ub(g));
        p.parent.push_back(t == 0 ? -1 : tree ? static_cast<int32_t>(g() % t) - (g() % 4 == 0 ? static_cast<int32_t>(t) + 1 : 0) : static_cast<int32_t>(t) - 1);
        if (p.parent.back() < -1) p.parent.back() = -1;  // some tree nodes branch from the root
    }
    return p;
}

// Sequential reference (double) along the root path of node t: returns o_t and S_t.
void sequential(const Problem& p, uint32_t t, std::vector<double>& o, std::vector<double>& S) {
    std::vector<uint32_t> path;
    for (int32_t x = static_cast<int32_t>(t); x >= 0; x = p.parent[x]) path.insert(path.begin(), static_cast<uint32_t>(x));
    S.assign(p.S0.begin(), p.S0.end());
    for (uint32_t s : path) {
        const float* k = &p.k[size_t(s) * p.dk];
        const float* v = &p.v[size_t(s) * p.dv];
        for (uint32_t r = 0; r < p.dv; ++r) {
            double sk = 0;
            for (uint32_t i = 0; i < p.dk; ++i) sk += S[r * p.dk + i] * k[i];
            const double e = v[r] - p.a[s] * sk;  // S_t = a S + b (v - a S k) k^T
            for (uint32_t i = 0; i < p.dk; ++i) S[r * p.dk + i] = p.a[s] * S[r * p.dk + i] + p.b[s] * e * k[i];
        }
    }
    o.assign(p.dv, 0.0);
    const float* q = &p.q[size_t(t) * p.dk];
    for (uint32_t r = 0; r < p.dv; ++r)
        for (uint32_t i = 0; i < p.dk; ++i) o[r] += S[r * p.dk + i] * q[i];
}

struct Chunk {
    std::vector<float> g;   // path decay products per node
    std::vector<float> E;   // M x dv (only the solved slice is meaningful)
    std::vector<float> O;   // M x dv
};

bool ancestor(const Problem& p, uint32_t j, uint32_t t) {  // j is a proper ancestor of t
    for (int32_t x = p.parent[t]; x >= 0; x = p.parent[x]) if (static_cast<uint32_t>(x) == j) return true;
    return false;
}

// WY solve over value columns [v0, v1). Writes the final state of node `final_node` (rows v0..v1) once per element.
Chunk wy_verify(const Problem& p, uint32_t v0, uint32_t v1, uint32_t final_node, float* s_out, std::vector<uint32_t>* writes) {
    const uint32_t M = p.M, dk = p.dk, dv = p.dv;
    Chunk c;
    c.g.resize(M);
    for (uint32_t t = 0; t < M; ++t) c.g[t] = p.a[t] * (p.parent[t] >= 0 ? c.g[p.parent[t]] : 1.0f);
    auto dot = [&](const float* x, const float* y) { float s = 0; for (uint32_t i = 0; i < dk; ++i) s += x[i] * y[i]; return s; };
    c.E.assign(size_t(M) * dv, 0.0f);
    c.O.assign(size_t(M) * dv, 0.0f);
    for (uint32_t t = 0; t < M; ++t) {  // forward substitution (parents precede children)
        const float* kt = &p.k[size_t(t) * dk];
        for (uint32_t r = v0; r < v1; ++r) {
            const float s0k = dot(&p.S0[size_t(r) * dk], kt);
            float e = p.v[size_t(t) * dv + r] - c.g[t] * s0k;
            for (uint32_t j = 0; j < t; ++j)
                if (ancestor(p, j, t)) e -= (c.g[t] / c.g[j]) * p.b[j] * dot(&p.k[size_t(j) * dk], kt) * c.E[size_t(j) * dv + r];
            c.E[size_t(t) * dv + r] = e;
        }
    }
    for (uint32_t t = 0; t < M; ++t) {
        const float* qt = &p.q[size_t(t) * dk];
        for (uint32_t r = v0; r < v1; ++r) {
            float o = c.g[t] * dot(&p.S0[size_t(r) * dk], qt);
            for (uint32_t j = 0; j <= t; ++j)
                if (j == t || ancestor(p, j, t)) o += (c.g[t] / c.g[j]) * p.b[j] * dot(&p.k[size_t(j) * dk], qt) * c.E[size_t(j) * dv + r];
            c.O[size_t(t) * dv + r] = o;
        }
    }
    // Single write of the final state for the committed node.
    if (s_out) {
        const uint32_t a = final_node;
        for (uint32_t r = v0; r < v1; ++r)
            for (uint32_t i = 0; i < dk; ++i) {
                float s = c.g[a] * p.S0[size_t(r) * dk + i];
                for (uint32_t j = 0; j <= a; ++j)
                    if (j == a || ancestor(p, j, a)) s += (c.g[a] / c.g[j]) * p.b[j] * c.E[size_t(j) * dv + r] * p.k[size_t(j) * dk + i];
                s_out[size_t(r) * dk + i] = s;
                if (writes) (*writes)[size_t(r) * dk + i]++;
            }
    }
    return c;
}

// Rollback: state after node a from the chunk's E (no re-run).
std::vector<double> state_at(const Problem& p, const Chunk& c, uint32_t a) {
    std::vector<double> S(size_t(p.dv) * p.dk);
    for (uint32_t r = 0; r < p.dv; ++r)
        for (uint32_t i = 0; i < p.dk; ++i) {
            double s = double(c.g[a]) * p.S0[size_t(r) * p.dk + i];
            for (uint32_t j = 0; j <= a; ++j)
                if (j == a || ancestor(p, j, a)) s += double(c.g[a] / c.g[j]) * p.b[j] * c.E[size_t(j) * p.dv + r] * p.k[size_t(j) * p.dk + i];
            S[size_t(r) * p.dk + i] = s;
        }
    return S;
}

double rel_err(const std::vector<double>& ref, const float* got, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) { num = std::max(num, std::fabs(ref[i] - got[i])); den = std::max(den, std::fabs(ref[i])); }
    return num / std::max(den, 1e-12);
}

} // namespace

int main() {
    std::mt19937 g(2026);
    double worst_o = 0, worst_s = 0;
    uint32_t cases = 0;
    for (bool tree : {false, true}) {
        for (uint32_t M = 5; M <= 12; ++M) {
            for (auto dims : {std::pair<uint32_t, uint32_t>{32, 48}, {128, 128}}) {
                const Problem p = make(g, dims.first, dims.second, M, tree);
                const uint32_t last = M - 1;
                std::vector<float> s_full(size_t(p.dv) * p.dk, NAN);
                std::vector<uint32_t> writes(s_full.size(), 0);
                const Chunk full = wy_verify(p, 0, p.dv, last, s_full.data(), &writes);
                for (uint32_t w : writes) VJ_CHECK(w == 1);  // Single-Write
                // Split-V: 4 slices solved independently, bitwise equal to the full solve.
                std::vector<float> s_split(s_full.size(), NAN);
                const uint32_t slices = 4, step = (p.dv + slices - 1) / slices;
                for (uint32_t v0 = 0; v0 < p.dv; v0 += step) {
                    const uint32_t v1 = std::min(p.dv, v0 + step);
                    const Chunk part = wy_verify(p, v0, v1, last, s_split.data(), nullptr);
                    for (uint32_t t = 0; t < M; ++t)
                        for (uint32_t r = v0; r < v1; ++r) {
                            VJ_CHECK(part.O[size_t(t) * p.dv + r] == full.O[size_t(t) * p.dv + r]);
                            VJ_CHECK(part.E[size_t(t) * p.dv + r] == full.E[size_t(t) * p.dv + r]);
                        }
                }
                for (size_t i = 0; i < s_full.size(); ++i) VJ_CHECK(s_split[i] == s_full[i]);
                // Outputs and per-node states (= every accepted prefix for a chain) vs the sequential recurrence.
                for (uint32_t t = 0; t < M; ++t) {
                    std::vector<double> o, S;
                    sequential(p, t, o, S);
                    const double eo = rel_err(o, &full.O[size_t(t) * p.dv], p.dv);
                    const auto St = state_at(p, full, t);
                    std::vector<float> Stf(St.begin(), St.end());
                    const double es = rel_err(S, Stf.data(), S.size());
                    worst_o = std::max(worst_o, eo);
                    worst_s = std::max(worst_s, es);
                    VJ_CHECK(eo < 2e-4 && es < 2e-4);
                    if (t == last) VJ_CHECK(rel_err(S, s_full.data(), S.size()) < 2e-4);
                }
                ++cases;
            }
        }
    }
    std::printf("  %u cases (chain + tree, M=5..12, dk/dv 32/48 and 128/128): worst rel err output %.2e, state %.2e\n", cases,
                worst_o, worst_s);
    std::printf("test_gdn_verify_oracle: WY chunk == sequential, Split-V bitwise, Single-Write, TreeWY, rollback OK\nPASS\n");
    return 0;
}
