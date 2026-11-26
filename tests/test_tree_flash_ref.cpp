// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU reference for the S3 tree-verify kernels (VEGAJET_V12_TREE_VERIFY, opt-in):
//   vj_tree_flash_partial_kernel (+ unchanged vj_spec_flash_merge_quant_kernel) and the logical RoPE position of
//   ornith_tree_attn_qkv_prep_kernel.
// Checks:
//   * logical -> physical row mapping (j-th set ancestor bit) equals the parent-link path of every node;
//   * per-row chunk_len/n_chunks come from the LOGICAL span pos0 + depth + 1 (crossing the 1920/3840/7680
//     thresholds where the physical span pos0 + M would pick a different chunk_len);
//   * a lane-level emulation of the tree kernel is bit-identical (partial states, y, INT8, (scale, sum)) to the
//     same emulation of the serial M = 1 spec-flash decode over the path's own contiguous KV rows;
//   * the tree result is close to an FP64 masked-attention reference built from parent links only;
//   * malformed rows (missing self bit, extra/missing ancestor bits, too few chunks) produce identity states.
#include "core/spec_tree.hpp"
#include "tests/test_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

namespace {

using vegajet::spec::VerifyTree;
constexpr uint32_t kHd = 256, kRow = 272, kNq = 16;

uint16_t f2h(float f) { _Float16 h = static_cast<_Float16>(f); uint16_t u; std::memcpy(&u, &h, 2); return u; }
float h2f(uint16_t u) { _Float16 h; std::memcpy(&h, &u, 2); return static_cast<float>(h); }
float i8_at(uint32_t w, uint32_t k) { return static_cast<float>(static_cast<int8_t>((w >> (8u * k)) & 0xFFu)); }

// Same rules as the kernel / host serial M = 1 launch.
uint32_t chunk_len_for(uint32_t span) { return span <= 1920u ? 32u : (span <= 3840u ? 64u : (span <= 7680u ? 128u : 256u)); }
uint32_t chunks_for(uint32_t span) { const uint32_t cl = chunk_len_for(span); return (span + cl - 1u) / cl; }
uint32_t tree_phys_row(uint32_t t, uint32_t pos0, uint32_t mask) {
    if (t < pos0) return t;
    for (uint32_t j = t - pos0; j > 0u; --j) mask &= mask - 1u;
    return pos0 + static_cast<uint32_t>(__builtin_ctz(mask));
}
bool row_ok(uint32_t M, uint32_t m, uint32_t mask, uint32_t depth, uint32_t pos0, uint32_t n_chunks) {
    const uint32_t L = pos0 + depth + 1u, cl = chunk_len_for(L);
    return M <= 32u && (mask >> m) == 1u && static_cast<uint32_t>(__builtin_popcount(mask)) == depth + 1u &&
           (L + cl - 1u) / cl <= n_chunks;
}

void fill_kv_row(uint8_t* r, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
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
float kv_elem(const std::vector<uint8_t>& buf, uint32_t t, uint32_t d) {
    const uint8_t* r = buf.data() + static_cast<size_t>(t) * kRow;
    uint16_t sh;
    std::memcpy(&sh, r + 2 * (d >> 5), 2);
    return h2f(sh) * static_cast<float>(static_cast<int8_t>(r[16 + d]));
}

struct RowOut {
    std::vector<float> ml;   // [kNq, n_chunks, 2]
    std::vector<float> po;   // [kNq, n_chunks, 256] (only non-empty chunks written; NaN otherwise)
    std::vector<float> y;    // [kNq, 256]
    std::vector<int8_t> q8;
    std::vector<float> ds;   // [kNq, 8, 2]
};

// Lane-level emulation of one row m of the partial kernel (n_kv_local = 1, 16 heads) + the merge kernel.
// phys(t) maps a logical index to a KV row; ok = false emulates the kernel's malformed-row guard.
RowOut run_row(const float* q, const float* gate, const std::vector<uint8_t>& kc, const std::vector<uint8_t>& vc,
               uint32_t seqlen, uint32_t chunk_len, uint32_t n_chunks, bool ok,
               const std::function<uint32_t(uint32_t)>& phys) {
    RowOut r{std::vector<float>(kNq * n_chunks * 2), std::vector<float>(kNq * n_chunks * kHd, std::nanf("")),
             std::vector<float>(kNq * kHd), std::vector<int8_t>(kNq * kHd), std::vector<float>(kNq * 16)};
    for (uint32_t c = 0; c < n_chunks; ++c)
        for (uint32_t h = 0; h < kNq; ++h) {
            const uint32_t tb = c * chunk_len, te = ok ? std::min(tb + chunk_len, seqlen) : 0u;
            const size_t st = static_cast<size_t>(h) * n_chunks + c;
            if (tb >= te) { r.ml[st * 2] = -1e30f; r.ml[st * 2 + 1] = 0.0f; continue; }
            float qq[32][8], o[32][8] = {}, mr = -1e30f, lr = 0.0f;
            for (uint32_t l = 0; l < 32; ++l)
                for (uint32_t i = 0; i < 8; ++i) qq[l][i] = q[h * kHd + l * 8 + i];
            for (uint32_t t0 = tb; t0 < te; t0 += 4) {
                float s[4], vsc[4][32];
                uint32_t vwx[4][32], vwy[4][32];
                for (uint32_t j = 0; j < 4; ++j) {
                    const size_t ro = static_cast<size_t>(phys(std::min(t0 + j, te - 1))) * kRow;
                    float lanev[32];
                    for (uint32_t l = 0; l < 32; ++l) {
                        const uint32_t qs = 16 + l * 8, sc = 2 * (l >> 2);
                        uint32_t kx, ky;
                        uint16_t ks, vs;
                        std::memcpy(&kx, &kc[ro + qs], 4); std::memcpy(&ky, &kc[ro + qs + 4], 4);
                        std::memcpy(&ks, &kc[ro + sc], 2);
                        std::memcpy(&vwx[j][l], &vc[ro + qs], 4); std::memcpy(&vwy[j][l], &vc[ro + qs + 4], 4);
                        std::memcpy(&vs, &vc[ro + sc], 2);
                        vsc[j][l] = h2f(vs);
                        float a0 = 0.0f, a1 = 0.0f;
                        for (uint32_t k = 0; k < 4; k += 2) {
                            a0 = std::fma(qq[l][k], i8_at(kx, k), a0);
                            a1 = std::fma(qq[l][k + 1], i8_at(kx, k + 1), a1);
                            a0 = std::fma(qq[l][4 + k], i8_at(ky, k), a0);
                            a1 = std::fma(qq[l][5 + k], i8_at(ky, k + 1), a1);
                        }
                        lanev[l] = (a0 + a1) * h2f(ks);
                    }
                    for (uint32_t sh : {16u, 8u, 4u, 2u, 1u}) {
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
                for (uint32_t i = 0; i < 8; ++i) r.po[st * kHd + l * 8 + i] = o[l][i];
            r.ml[st * 2] = mr;
            r.ml[st * 2 + 1] = lr;
        }
    for (uint32_t h = 0; h < kNq; ++h) {
        const size_t base = static_cast<size_t>(h) * n_chunks;
        float gmax = -1e30f;
        for (uint32_t c = 0; c < n_chunks; ++c) if (r.ml[(base + c) * 2 + 1] > 0.0f) gmax = std::max(gmax, r.ml[(base + c) * 2]);
        float y[kHd];
        for (uint32_t d = 0; d < kHd; ++d) {
            float o = 0.0f, l = 0.0f;
            for (uint32_t c = 0; c < n_chunks; ++c) {
                const float lc = r.ml[(base + c) * 2 + 1];
                if (lc > 0.0f) {
                    const float w = std::exp(r.ml[(base + c) * 2] - gmax);
                    l = std::fma(w, lc, l);
                    o = std::fma(w, r.po[(base + c) * kHd + d], o);
                }
            }
            y[d] = (l > 0.0f ? o / l : 0.0f) * (1.0f / (1.0f + std::exp(-gate[h * kHd + d])));
        }
        for (uint32_t b = 0; b < 8; ++b) {
            float amax = 0.0f, sum = 0.0f;
            for (uint32_t i = 0; i < 32; ++i) { amax = std::max(amax, std::fabs(y[b * 32 + i])); sum += y[b * 32 + i]; }
            const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
            for (uint32_t i = 0; i < 32; ++i)
                r.q8[h * kHd + b * 32 + i] = static_cast<int8_t>(std::clamp(static_cast<int>(std::nearbyint(y[b * 32 + i] * inv)), -127, 127));
            r.ds[(h * 8 + b) * 2] = amax / 127.0f;
            r.ds[(h * 8 + b) * 2 + 1] = sum;
        }
        std::memcpy(&r.y[h * kHd], y, sizeof(y));
    }
    return r;
}

bool same_bits(const void* a, const void* b, size_t n) { return std::memcmp(a, b, n) == 0; }

// Path of node m (root first) from parent links only (independent of the bit mapping).
std::vector<uint32_t> path_of(const VerifyTree& t, uint32_t m) {
    std::vector<uint32_t> p;
    for (int32_t i = static_cast<int32_t>(m); i >= 0; i = t.nodes[i].parent) p.push_back(static_cast<uint32_t>(i));
    std::reverse(p.begin(), p.end());
    return p;
}

int check_tree(const VerifyTree& t, uint32_t pos0, uint32_t seed) {
    const uint32_t M = static_cast<uint32_t>(t.nodes.size());
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint8_t> kc(static_cast<size_t>(pos0 + M) * kRow), vc(kc.size());
    for (uint32_t r = 0; r < pos0 + M; ++r) { fill_kv_row(&kc[r * kRow], rng); fill_kv_row(&vc[r * kRow], rng); }
    std::vector<float> q(static_cast<size_t>(M) * kNq * kHd), gate(q.size());
    for (auto& v : q) v = nd(rng) * 1.5f;
    for (auto& v : gate) v = nd(rng);

    // Host launch rule: common n_chunks = max over rows of the row's logical chunk count.
    uint32_t n_chunks = 1;
    for (uint32_t m = 0; m < M; ++m) n_chunks = std::max(n_chunks, chunks_for(pos0 + t.depth[m] + 1u));
    int checks = 0;
    for (uint32_t m = 0; m < M; ++m) {
        const uint32_t mask = t.ancestor_mask[m], depth = t.depth[m], L = pos0 + depth + 1u;
        const std::vector<uint32_t> path = path_of(t, m);
        VJ_CHECK(path.size() == depth + 1u && path.front() == 0u && path.back() == m);
        VJ_CHECK(row_ok(M, m, mask, depth, pos0, n_chunks));
        // Index mapping: prefix identity, suffix = ancestors in depth order.
        for (uint32_t i = 0; i < std::min(pos0, 3u); ++i) VJ_CHECK(tree_phys_row(i, pos0, mask) == i);
        if (pos0 > 0) VJ_CHECK(tree_phys_row(pos0 - 1u, pos0, mask) == pos0 - 1u);
        for (uint32_t j = 0; j <= depth; ++j) VJ_CHECK(tree_phys_row(pos0 + j, pos0, mask) == pos0 + path[j]);
        // Prep: RoPE position of row m equals the serial position of its path end; KV row stays pos0 + m.
        VJ_CHECK(pos0 + depth == pos0 + static_cast<uint32_t>(path.size()) - 1u);

        // Serial M = 1 decode of this path: contiguous KV [prefix | ancestors], pos = L - 1, span = pos + 1.
        std::vector<uint8_t> skc(static_cast<size_t>(L) * kRow), svc(skc.size());
        std::memcpy(skc.data(), kc.data(), static_cast<size_t>(pos0) * kRow);
        std::memcpy(svc.data(), vc.data(), static_cast<size_t>(pos0) * kRow);
        for (uint32_t j = 0; j <= depth; ++j) {
            std::memcpy(&skc[(pos0 + j) * kRow], &kc[(pos0 + path[j]) * kRow], kRow);
            std::memcpy(&svc[(pos0 + j) * kRow], &vc[(pos0 + path[j]) * kRow], kRow);
        }
        const float* qm = &q[static_cast<size_t>(m) * kNq * kHd];
        const float* gm = &gate[static_cast<size_t>(m) * kNq * kHd];
        const uint32_t s_nc = chunks_for(L);
        const RowOut ser = run_row(qm, gm, skc, svc, L, chunk_len_for(L), s_nc, true, [](uint32_t x) { return x; });
        const RowOut tre = run_row(qm, gm, kc, vc, L, chunk_len_for(L), n_chunks, true,
                                   [&](uint32_t x) { return tree_phys_row(x, pos0, mask); });
        // Partial states: equal for every serial chunk; extra tree chunks are identity.
        for (uint32_t h = 0; h < kNq; ++h) {
            for (uint32_t c = 0; c < s_nc; ++c) {
                VJ_CHECK(same_bits(&ser.ml[(h * s_nc + c) * 2], &tre.ml[(h * n_chunks + c) * 2], 2 * sizeof(float)));
                VJ_CHECK(same_bits(&ser.po[(h * s_nc + c) * kHd], &tre.po[(h * n_chunks + c) * kHd], kHd * sizeof(float)));
            }
            for (uint32_t c = s_nc; c < n_chunks; ++c) VJ_CHECK(tre.ml[(h * n_chunks + c) * 2 + 1] == 0.0f);
        }
        VJ_CHECK(same_bits(ser.y.data(), tre.y.data(), ser.y.size() * sizeof(float)));
        VJ_CHECK(same_bits(ser.q8.data(), tre.q8.data(), ser.q8.size()));
        VJ_CHECK(same_bits(ser.ds.data(), tre.ds.data(), ser.ds.size() * sizeof(float)));

        // FP64 masked attention from parent links: visible rows = prefix + path.
        std::vector<uint32_t> vis;
        for (uint32_t i = 0; i < pos0; ++i) vis.push_back(i);
        for (uint32_t a : path) vis.push_back(pos0 + a);
        double worst = 0.0, ymax = 0.0;
        for (uint32_t h = 0; h < kNq; ++h) {
            std::vector<double> s(vis.size());
            double mx = -1e300;
            for (size_t i = 0; i < vis.size(); ++i) {
                double a = 0.0;
                for (uint32_t d = 0; d < kHd; ++d) a += static_cast<double>(qm[h * kHd + d]) * kv_elem(kc, vis[i], d);
                s[i] = a / 16.0;
                mx = std::max(mx, s[i]);
            }
            double l = 0.0;
            std::vector<double> acc(kHd, 0.0);
            for (size_t i = 0; i < vis.size(); ++i) {
                const double w = std::exp(s[i] - mx);
                l += w;
                for (uint32_t d = 0; d < kHd; ++d) acc[d] += w * kv_elem(vc, vis[i], d);
            }
            for (uint32_t d = 0; d < kHd; ++d) {
                const double ex = acc[d] / l / (1.0 + std::exp(-static_cast<double>(gm[h * kHd + d])));
                ymax = std::max(ymax, std::fabs(ex));
                worst = std::max(worst, std::fabs(tre.y[h * kHd + d] - ex));
            }
        }
        VJ_CHECK(worst / std::max(ymax, 1e-30) < 2e-5);
        checks += 1;
    }
    std::printf("[tree_flash_ref] M=%u pos0=%5u n_chunks=%2u (physical-span chunk_len %3u, logical %3u..%3u): %d rows bit-identical to serial\n",
                M, pos0, n_chunks, chunk_len_for(pos0 + M), chunk_len_for(pos0 + 1u),
                chunk_len_for(pos0 + *std::max_element(t.depth.begin(), t.depth.end()) + 1u), checks);
    return checks;
}

} // namespace

int main() {
    using vegajet::spec::make_topk_verify_tree;
    using vegajet::spec::make_verify_tree;
    using vegajet::spec::DraftNode;
    // Seven-row top-k tree (depths 0,1,1,2,2,2,2) and a deep suffix tree (chain 1-2-3-4 + branches).
    const VerifyTree topk = make_topk_verify_tree(10, {{20, 21}, {30, 31}});
    VJ_CHECK(topk.nodes.size() == 7);
    std::vector<DraftNode> d(6);
    d[0] = {1, -1, 0.0}; d[1] = {2, 0, 0.0}; d[2] = {3, -1, 0.0}; d[3] = {4, 1, 0.0}; d[4] = {5, 3, 0.0}; d[5] = {6, 1, 0.0};
    const VerifyTree deep = make_verify_tree(10, d);
    VJ_CHECK(deep.nodes.size() == 7 && *std::max_element(deep.depth.begin(), deep.depth.end()) == 4);

    int n = 0;
    // pos0 chosen around the chunk_len thresholds: physical span pos0 + M crosses them, logical spans differ per row.
    for (uint32_t pos0 : {0u, 3u, 61u, 1914u, 1917u, 3836u, 7678u}) {
        n += check_tree(topk, pos0, 100u + pos0);
        n += check_tree(deep, pos0, 200u + pos0);
    }
    // At pos0 = 1914 the chain kernel's physical span (1921) picks chunk_len 64; every logical row needs 32.
    VJ_CHECK(chunk_len_for(1914u + 7u) == 64u && chunk_len_for(1914u + 4u + 1u) == 32u);

    // Malformed rows: guard emulation rejects them and the row output is identity (all l = 0, y = 0).
    {
        const uint32_t pos0 = 40, M = 7;
        const uint32_t nc = chunks_for(pos0 + 3u);
        VJ_CHECK(!row_ok(M, 3, 0b1001u, 2, pos0, nc));    // missing parent bit
        VJ_CHECK(!row_ok(M, 3, 0b0011u, 1, pos0, nc));    // missing self bit
        VJ_CHECK(!row_ok(M, 3, 0b11011u, 2, pos0, nc));   // bit above self
        VJ_CHECK(!row_ok(M, 3, 0b1011u, 3, pos0, nc));    // popcount != depth + 1
        VJ_CHECK(!row_ok(33, 3, 0b1011u, 2, pos0, nc));   // M above 32 mask bits
        VJ_CHECK(!row_ok(M, 3, 0b1011u, 2, 4000u, 1u));   // n_chunks too small for the logical span
        VJ_CHECK(row_ok(M, 3, 0b1011u, 2, pos0, nc));
        std::vector<uint8_t> kc(static_cast<size_t>(pos0 + M) * kRow, 0), vc(kc.size(), 0);
        std::vector<float> q(kNq * kHd, 1.0f), g(kNq * kHd, 0.0f);
        const RowOut r = run_row(q.data(), g.data(), kc, vc, pos0 + 3u, chunk_len_for(pos0 + 3u), nc, false,
                                 [](uint32_t x) { return x; });
        for (uint32_t i = 0; i < kNq * nc; ++i) VJ_CHECK(r.ml[i * 2 + 1] == 0.0f);
        for (float y : r.y) VJ_CHECK(y == 0.0f);
        ++n;
    }
    std::printf("test_tree_flash_ref: all %d checks passed\n", n);
    return 0;
}
