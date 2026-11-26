// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU reference for ops/kernels/vj_tree_gdn.hip indexing (not device numerics evidence):
// branched conv (conv_inputs) and per-node GDN recurrence (parents + node_states, restore skipped when
// parent == m-1, as the kernel does) must equal a serial chain over each root-to-node path bit for bit;
// sibling order must not matter; commit-plan replay rows and conv tail must reproduce the node state/slot.
#include "core/spec_tree.hpp"
#include "tests/test_common.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using namespace vegajet::spec;
using Vec = std::vector<float>;

constexpr uint32_t kSlots = 17, kC = 8, kD = 128;  // conv channels, GDN head dim

static bool same(const float* a, const float* b, size_t n) { return std::memcmp(a, b, n * 4) == 0; }
static float mul(float a, float b) { volatile float r = a * b; return r; }
static float sub(float a, float b) { volatile float r = a - b; return r; }

// Per-token inputs, keyed by token so sibling permutations carry their data along.
struct Row { Vec x, k, q, v; std::vector<float> g, b; };
static Row make_row(int32_t token, uint32_t H) {
    std::mt19937 rng(static_cast<uint32_t>(token) * 7919u + 1u);
    std::uniform_real_distribution<float> u(-1.f, 1.f), p(0.5f, 0.99f);
    Row r;
    for (uint32_t i = 0; i < kC; ++i) r.x.push_back(u(rng));
    for (uint32_t i = 0; i < H * kD; ++i) { r.k.push_back(0.1f * u(rng)); r.q.push_back(0.1f * u(rng)); r.v.push_back(u(rng)); }
    for (uint32_t h = 0; h < H; ++h) { r.g.push_back(p(rng)); r.b.push_back(p(rng)); }
    return r;
}

// ornith_gdn_conv_l2_gate_kernel per-channel arithmetic (h0 oldest).
static float conv_y(float w0, float w1, float w2, float w3, float h0, float h1, float h2, float x) {
    const float acc = std::fma(w0, h0, std::fma(w1, h1, std::fma(w2, h2, mul(w3, x))));
    return acc / (1.0f + std::exp(-acc));
}

// One GDN token for all heads/columns, vj_spec_gdn_verify_kernel ordering (8 x 16-row partials, sum8 L->R).
static void gdn_step(Vec& S, const Row& r, uint32_t H, float* o) {
    for (uint32_t h = 0; h < H; ++h)
        for (uint32_t c = 0; c < kD; ++c) {
            float* s = &S[h * kD * kD];
            const float* k = &r.k[h * kD];
            const float* q = &r.q[h * kD];
            float part[8];
            for (uint32_t g = 0; g < 8; ++g) {
                float p = 0.f;
                for (uint32_t i = 16 * g; i < 16 * g + 16; ++i) p = std::fma(s[i * kD + c], k[i], p);
                part[g] = p;
            }
            float pred = part[0];
            for (uint32_t g = 1; g < 8; ++g) pred += part[g];
            const float dv = mul(r.b[h], sub(r.v[h * kD + c], mul(r.g[h], pred)));
            for (uint32_t g = 0; g < 8; ++g) {
                float oo = 0.f;
                for (uint32_t i = 16 * g; i < 16 * g + 16; ++i) {
                    s[i * kD + c] = std::fma(k[i], dv, mul(r.g[h], s[i * kD + c]));
                    oo = std::fma(s[i * kD + c], q[i], oo);
                }
                part[g] = oo;
            }
            float ov = part[0];
            for (uint32_t g = 1; g < 8; ++g) ov += part[g];
            o[h * kD + c] = ov;
        }
}

struct Fixture {
    uint32_t H;
    float w[kC][4];
    float hist[3][kC];   // immutable conv history at ring_head, slot 0 oldest
    Vec S0;              // immutable GDN state
    explicit Fixture(uint32_t heads) : H(heads), S0(heads * kD * kD) {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> u(-1.f, 1.f);
        for (auto& ch : w) for (float& v : ch) v = 0.5f * u(rng);
        for (auto& sl : hist) for (float& v : sl) v = u(rng);
        for (float& v : S0) v = 0.05f * u(rng);
    }
};

struct TreeOut {
    std::vector<Vec> y, o, node_state;   // per row
    std::vector<Vec> ring;               // kSlots x (3*kC)
};

// Mirror of the two tree kernels.
static TreeOut run_tree(const Fixture& f, const VerifyTree& t, const std::vector<Row>& rows, uint32_t head) {
    const uint32_t M = static_cast<uint32_t>(t.nodes.size());
    TreeOut out;
    out.ring.assign(kSlots, Vec(3 * kC, -7.f));
    for (uint32_t s = 0; s < 3; ++s) for (uint32_t ch = 0; ch < kC; ++ch) out.ring[head][s * kC + ch] = f.hist[s][ch];
    auto hval = [&](int32_t idx, uint32_t m, uint32_t ch) {
        VJ_CHECK(idx >= -3 && idx < static_cast<int32_t>(m));  // ancestors only, never a later sibling
        return idx < 0 ? f.hist[3 + idx][ch] : rows[idx].x[ch];
    };
    for (uint32_t m = 0; m < M; ++m) {
        const auto& ci = t.conv_inputs[m];
        Vec y(kC);
        const uint32_t slot = (head + m + 1) % kSlots;
        for (uint32_t ch = 0; ch < kC; ++ch) {
            const float h0 = hval(ci[0], m, ch), h1 = hval(ci[1], m, ch), h2 = hval(ci[2], m, ch);
            y[ch] = conv_y(f.w[ch][0], f.w[ch][1], f.w[ch][2], f.w[ch][3], h0, h1, h2, rows[m].x[ch]);
            out.ring[slot][0 * kC + ch] = h1;
            out.ring[slot][1 * kC + ch] = h2;
            out.ring[slot][2 * kC + ch] = rows[m].x[ch];
        }
        out.y.push_back(y);
    }
    Vec regs = f.S0;
    for (uint32_t m = 0; m < M; ++m) {
        const int32_t par = t.nodes[m].parent;
        VJ_CHECK((m == 0) == (par == -1) && par < static_cast<int32_t>(m));
        if (m > 0 && static_cast<uint32_t>(par) + 1 != m) regs = out.node_state[par];
        Vec o(f.H * kD);
        gdn_step(regs, rows[m], f.H, o.data());
        out.o.push_back(o);
        out.node_state.push_back(regs);
    }
    return out;
}

static std::vector<uint32_t> path_to(const VerifyTree& t, uint32_t n) {
    std::vector<uint32_t> p;
    for (int32_t r = static_cast<int32_t>(n); r >= 0; r = t.nodes[r].parent) p.insert(p.begin(), r);
    return p;
}

// Serial chain over path rows from the immutable snapshot (the chain conv kernel and spec-GDN verify).
struct ChainOut { Vec y, o, S; float last_hist[3][kC]; };
static ChainOut run_chain(const Fixture& f, const std::vector<Row>& rows, const std::vector<uint32_t>& path) {
    ChainOut c;
    std::memcpy(c.last_hist, f.hist, sizeof c.last_hist);
    c.S = f.S0;
    for (uint32_t row : path) {
        c.y.assign(kC, 0.f);
        for (uint32_t ch = 0; ch < kC; ++ch) {
            auto& hh = c.last_hist;
            c.y[ch] = conv_y(f.w[ch][0], f.w[ch][1], f.w[ch][2], f.w[ch][3], hh[0][ch], hh[1][ch], hh[2][ch], rows[row].x[ch]);
            hh[0][ch] = hh[1][ch]; hh[1][ch] = hh[2][ch]; hh[2][ch] = rows[row].x[ch];
        }
        c.o.assign(f.H * kD, 0.f);
        gdn_step(c.S, rows[row], f.H, c.o.data());
    }
    return c;
}

static std::vector<Row> rows_for(const VerifyTree& t, uint32_t H) {
    std::vector<Row> rows;
    for (const auto& n : t.nodes) rows.push_back(make_row(n.token, H));
    return rows;
}

// Every node equals the serial chain on its path; every terminal commits exactly via plan replay.
static void check_tree(const Fixture& f, const VerifyTree& t, uint32_t head) {
    const auto rows = rows_for(t, f.H);
    const TreeOut out = run_tree(f, t, rows, head);
    for (uint32_t n = 0; n < t.nodes.size(); ++n) {
        const auto path = path_to(t, n);
        const ChainOut c = run_chain(f, rows, path);
        VJ_CHECK(same(out.y[n].data(), c.y.data(), kC));
        VJ_CHECK(same(out.o[n].data(), c.o.data(), c.o.size()));
        VJ_CHECK(same(out.node_state[n].data(), c.S.data(), c.S.size()));
        const Vec& slot = out.ring[(head + n + 1) % kSlots];
        VJ_CHECK(same(slot.data(), &c.last_hist[0][0], 3 * kC));

        // Commit: replay gathered records from the immutable snapshot (existing spec-GDN replay).
        const TreeCommitPlan plan = plan_tree_commit(t, path, 100, 1000, 64, head, kSlots);
        VJ_CHECK(plan.replay_rows == path && plan.state_node == n);
        std::vector<Row> gathered;
        for (uint32_t r : plan.replay_rows) gathered.push_back(rows[r]);
        Vec S = f.S0, o(f.H * kD);
        for (const auto& r : gathered) gdn_step(S, r, f.H, o.data());
        VJ_CHECK(same(S.data(), out.node_state[n].data(), S.size()));
        // Conv tail resolves to the slot the tree kernel wrote for n. The committed chain slot is
        // next_ring_head = head + path length, NOT head + n + 1: the executor installs/copies it.
        VJ_CHECK(plan.next_ring_head == (head + path.size()) % kSlots);
        for (uint32_t s = 0; s < 3; ++s)
            for (uint32_t ch = 0; ch < kC; ++ch) {
                const int32_t idx = plan.conv_tail[s];
                const float v = idx < 0 ? f.hist[3 + idx][ch] : rows[idx].x[ch];
                VJ_CHECK(std::memcmp(&v, &slot[s * kC + ch], 4) == 0);
            }
    }
}

// Token path from root, used to match nodes across sibling orders.
static std::vector<int32_t> token_path(const VerifyTree& t, uint32_t n) {
    std::vector<int32_t> p;
    for (uint32_t r : path_to(t, n)) p.push_back(t.nodes[r].token);
    return p;
}

static void sibling_permutation(const Fixture& f) {
    using D = DraftNode;
    // Same logical tree, children/grandchildren listed in different orders.
    const VerifyTree a = make_verify_tree(5, {D{11, -1}, D{12, -1}, D{21, 0}, D{22, 0}, D{23, 1}, D{24, 1}});
    const VerifyTree b = make_verify_tree(5, {D{12, -1}, D{11, -1}, D{24, 0}, D{23, 0}, D{22, 1}, D{21, 1}});
    const TreeOut oa = run_tree(f, a, rows_for(a, f.H), 3), ob = run_tree(f, b, rows_for(b, f.H), 3);
    for (uint32_t i = 0; i < a.nodes.size(); ++i) {
        uint32_t j = 0;
        while (j < b.nodes.size() && token_path(b, j) != token_path(a, i)) ++j;
        VJ_CHECK(j < b.nodes.size());
        VJ_CHECK(oa.y[i] == ob.y[j] && same(oa.o[i].data(), ob.o[j].data(), oa.o[i].size()));
        VJ_CHECK(same(oa.node_state[i].data(), ob.node_state[j].data(), oa.node_state[i].size()));
    }
}

int main() {
    const Fixture f2(2), f1(1);
    // DFlash2 top-k shape (anchor, 2 children, 4 grandchildren) and ring wrap at head 15/16.
    const VerifyTree topk = make_topk_verify_tree(7, {{20, 21}, {30, 31}});
    VJ_CHECK(topk.nodes.size() == kTreeVerifyRows);
    for (uint32_t head : {0u, 15u, 16u}) check_tree(f2, topk, head);
    sibling_permutation(f2);

    // All 720 ordered six-draft parent layouts (deep chains, stars, mixed), single head for speed.
    uint32_t layouts = 0;
    std::array<int32_t, 6> par;
    par.fill(-1);
    for (;;) {
        std::vector<DraftNode> drafts;
        for (int32_t i = 0; i < 6; ++i) drafts.push_back({100 + i, par[i], 0.0});
        check_tree(f1, make_verify_tree(1, drafts), layouts % kSlots);
        ++layouts;
        int32_t i = 5;
        while (i >= 0 && par[i] == i - 1) par[i--] = -1;
        if (i < 0) break;
        ++par[i];
    }
    VJ_CHECK(layouts == 720);
    std::printf("test_tree_gdn_ref: PASS (%u layouts, topk 3 ring heads, sibling permutation)\n", layouts);
    return 0;
}
