// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU qualification of the sealed GDN production-input producer (ops/gdn_wire_pack.hpp) and of
// the numerical contract it serves.
//
// Part 1 — descriptor and byte evidence:
//   * every invalid pointer, geometry, stride and alias is rejected before any write;
//   * the host mirror rewrites every element the sealed verifier reads (poisoned destinations
//     come out byte-identical to the conv-layout source), keeps a non-default conv stride, and
//     leaves rows outside the sealed width untouched (stale-scratch discipline);
//   * the parent chain is copied exactly, including a branching tree.
//
// Part 2 — numerical equivalence:
//   * a production-shaped interleaved conv fixture (q/k already L2-normalized, q scaled by
//     1/sqrt(128), decay/beta post-gate) is packed by the producer;
//   * the packed planes are fed to the independently transcribed sealed WY record build and to
//     the production sequential recurrence (the existing verifier's arithmetic);
//   * at widths 5/8/12 the sealed per-node output and the committed state after every accepted
//     prefix (root-only restore / partial / full) match the sequential verifier within 1e-5, and
//     commit(0) restores S0 bit-for-bit.
//
// This is CPU evidence only. It makes no device-execution, resource or throughput claim; the
// device kernel is qualified separately by the pinned GPU-mode compile and ISA/resource gates.
#include "ops/gdn_wire_pack.hpp"
#include "ops/kernels/vj_gdn_treewy_verify.hiph"
#include "tests/test_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

using vegajet::gdn::wire_pack::Args;
using vegajet::gdn::wire_pack::HostBuffers;
using vegajet::treewy::Record;

namespace {

constexpr int H = 32;
constexpr int D = 128;
constexpr int K = 8;
constexpr int kState = H * D * D;

uint32_t bits(float f) {
    uint32_t b = 0;
    std::memcpy(&b, &f, sizeof b);
    return b;
}

// Volatile helpers keep the CPU models from contracting a multiplication into a neighbouring FMA
// (same discipline as tests/test_gdn_treewy_verify.cpp).
float mul(float a, float b) {
    volatile float r = a * b;
    return r;
}
float sub(float a, float b) {
    volatile float r = a - b;
    return r;
}

struct Inputs {
    int m = 0;
    std::vector<float> s;       // [H][D][D]
    std::vector<float> q, k;    // [m][K][D]
    std::vector<float> v;       // [m][H][D]
    std::vector<float> g, b;    // [m][H]
    std::vector<int32_t> parent;
};

struct Oracle {
    std::vector<std::vector<float>> states;
    std::vector<float> output;  // [m][H][D]
};

// Production sequential verifier arithmetic (same rule as tests/test_gdn_treewy_verify.cpp):
// 8 groups of 16 FMAs, groups summed left-to-right, beta*(v - g*pred), then fmaf(k,dv,g*S).
Oracle chronological(const Inputs& in) {
    Oracle ref;
    ref.states.reserve(in.m);
    ref.output.assign(static_cast<size_t>(in.m) * H * D, 0.0f);
    for (int n = 0; n < in.m; ++n) {
        std::vector<float> s = in.parent[n] < 0 ? in.s : ref.states[in.parent[n]];
        for (int h = 0; h < H; ++h) {
            for (int c = 0; c < D; ++c) {
                float p[8] = {}, o[8] = {};
                for (int group = 0; group < 8; ++group) {
                    for (int r = 0; r < 16; ++r) {
                        const int row = group * 16 + r;
                        p[group] = std::fma(s[h * D * D + row * D + c], in.k[(n * K + h % K) * D + row],
                                            p[group]);
                    }
                }
                float pred = p[0];
                for (int group = 1; group < 8; ++group) pred = pred + p[group];
                const float dv = mul(in.b[n * H + h], sub(in.v[(n * H + h) * D + c],
                                                          mul(in.g[n * H + h], pred)));
                for (int group = 0; group < 8; ++group) {
                    for (int r = 0; r < 16; ++r) {
                        const int row = group * 16 + r;
                        const int idx = h * D * D + row * D + c;
                        s[idx] = std::fma(in.k[(n * K + h % K) * D + row], dv,
                                          mul(in.g[n * H + h], s[idx]));
                        o[group] = std::fma(s[idx], in.q[(n * K + h % K) * D + row], o[group]);
                    }
                }
                float out = o[0];
                for (int group = 1; group < 8; ++group) out = out + o[group];
                ref.output[(n * H + h) * D + c] = out;
            }
        }
        ref.states.push_back(std::move(s));
    }
    return ref;
}

float reconstruct(const Record& r, int accepted, int row, int c) {
    float z = r.initial[row * 32 + c];
    if (accepted != 0) {
        const int i = accepted - 1;
        z = mul(r.root[i], z);
        for (int j = 0; j <= i; ++j) {
            if (r.path[i * 12 + j] == 0.0f) continue;
            z = std::fma(r.key[j * 128 + row], mul(r.path[i * 12 + j], r.delta[j * 32 + c]), z);
        }
    }
    return z;
}

std::vector<float> commit(const std::vector<Record>& rec, int accepted) {
    std::vector<float> state(kState, 0.0f);
    for (int wg = 0; wg < 128; ++wg) {
        for (int row = 0; row < D; ++row) {
            for (int c = 0; c < 32; ++c) {
                state[(wg / 4) * D * D + row * D + (wg % 4) * 32 + c] =
                    reconstruct(rec[wg], accepted, row, c);
            }
        }
    }
    return state;
}

// CPU model of the sealed verify kernel (vj_gdn_treewy_verify_kernel). The statement order is the
// kernel's own grouped-FMA dot and nilpotent inverse-by-doubling solve.
std::vector<float> sealed_verify(const Inputs& in, std::vector<Record>& records,
                                 std::vector<float>& scratch) {
    const int m = in.m;
    std::vector<float> out(static_cast<size_t>(m) * H * D, 0.0f);
    scratch.resize(2 * 144 + 2 * 384);
    records.assign(128, Record{});
    for (int wg = 0; wg < 128; ++wg) {
        const int h = wg / 4, part = wg % 4;
        Record& r = records[wg];
        for (int row = 0; row < D; ++row) {
            for (int c = 0; c < 32; ++c)
                r.initial[row * 32 + c] = in.s[h * D * D + row * D + part * 32 + c];
        }
        for (int i = 0; i < m; ++i) {
            std::copy_n(in.k.data() + (i * K + h % K) * D, D, r.key + i * D);
            float d = 1.0f;
            for (int j = 0; j < m; ++j) r.path[i * 12 + j] = 0.0f;
            for (int j = i; j >= 0; j = in.parent[j]) {
                r.path[i * 12 + j] = d;
                d = mul(d, in.g[j * H + h]);
            }
            r.root[i] = d;
        }
        float* b = scratch.data();
        float* bn = b + 144;
        float* x = bn + 144;
        float* xn = x + 384;
        std::fill(scratch.begin(), scratch.end(), 0.0f);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < i; ++j) {
                if (r.path[i * 12 + j] != 0.0f) {
                    b[i * 12 + j] = -mul(mul(in.b[i * H + h], r.path[i * 12 + j]),
                                         vegajet::treewy::dot128(r.key + i * D, r.key + j * D));
                }
            }
            for (int c = 0; c < 32; ++c) {
                float column[D];
                for (int row = 0; row < D; ++row) column[row] = r.initial[row * 32 + c];
                const float pred = vegajet::treewy::dot128(column, r.key + i * D);
                x[i * 32 + c] = mul(in.b[i * H + h],
                                    sub(in.v[(i * H + h) * D + part * 32 + c],
                                        mul(r.root[i], pred)));
            }
        }
        for (int span = 1; span < m; span *= 2) {
            for (int i = 0; i < m; ++i) {
                for (int c = 0; c < 32; ++c) {
                    float z = x[i * 32 + c];
                    for (int j = 0; j < i; ++j) z = std::fma(b[i * 12 + j], x[j * 32 + c], z);
                    xn[i * 32 + c] = z;
                }
            }
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < m; ++j) {
                    float z = 0.0f;
                    for (int l = j + 1; l < i; ++l) z = std::fma(b[i * 12 + l], b[l * 12 + j], z);
                    bn[i * 12 + j] = z;
                }
            }
            std::copy_n(xn, m * 32, x);
            std::copy_n(bn, 144, b);
        }
        std::copy_n(x, m * 32, r.delta);
        for (int i = 0; i < m; ++i) {
            for (int c = 0; c < 32; ++c) {
                float column[D];
                for (int row = 0; row < D; ++row) column[row] = reconstruct(r, i + 1, row, c);
                out[(i * H + h) * D + part * 32 + c] =
                    vegajet::treewy::dot128(column, in.q.data() + (i * K + h % K) * D);
            }
        }
    }
    return out;
}

struct Diff {
    size_t bit_mismatches = 0;
    double max_abs = 0.0;
};

Diff compare(const std::vector<float>& a, const std::vector<float>& b) {
    VJ_CHECK(a.size() == b.size());
    Diff d;
    for (size_t i = 0; i < a.size(); ++i) {
        if (bits(a[i]) != bits(b[i])) ++d.bit_mismatches;
        const double e = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (e > d.max_abs) d.max_abs = e;
    }
    return d;
}

// ---- production-shaped fixture ----------------------------------------------------------
struct Fixture {
    uint32_t m = 0;
    uint32_t stride = vegajet::gdn::wire_pack::kConvStride;
    int seed = 0;
    std::vector<float> conv;   // [m][stride] interleaved q|k|v
    std::vector<float> decay;  // [m][H]
    std::vector<float> beta;   // [m][H]
    std::vector<float> s0;     // [H][D][D]
    std::vector<int32_t> parent;

    void build(uint32_t width, uint32_t conv_stride, int fixture_seed, bool branching = false) {
        using namespace vegajet::gdn::wire_pack;
        m = width;
        stride = conv_stride;
        seed = fixture_seed;
        std::mt19937 rng(static_cast<uint32_t>(fixture_seed));
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::uniform_real_distribution<float> ua(0.85f, 0.999f), ub(0.05f, 0.95f);
        s0.assign(kState, 0.0f);
        for (float& x : s0) x = 0.3f * nd(rng);
        conv.assign(static_cast<size_t>(m) * stride, 0.0f);
        decay.assign(static_cast<size_t>(m) * H, 0.0f);
        beta.assign(static_cast<size_t>(m) * H, 0.0f);
        parent.assign(m, -1);
        const float qscale = 0.08838834764831845f;  // 1/sqrt(128), the conv kernel's Q scale
        for (uint32_t t = 0; t < m; ++t) {
            float* row = conv.data() + static_cast<size_t>(t) * stride;
            for (int hk = 0; hk < K; ++hk) {
                double nq = 0.0, nk = 0.0;
                float* qh = row + hk * D;
                float* kh = row + (K + hk) * D;
                for (int i = 0; i < D; ++i) {
                    qh[i] = nd(rng);
                    kh[i] = nd(rng);
                    nq += static_cast<double>(qh[i]) * qh[i];
                    nk += static_cast<double>(kh[i]) * kh[i];
                }
                const float iq = static_cast<float>(1.0 / std::sqrt(nq));
                const float ik = static_cast<float>(1.0 / std::sqrt(nk));
                for (int i = 0; i < D; ++i) {
                    qh[i] = qh[i] * iq * qscale;
                    kh[i] = kh[i] * ik;
                }
            }
            for (int hv = 0; hv < H; ++hv) {
                float* vh = row + (2 * K + hv) * D;
                for (int i = 0; i < D; ++i) vh[i] = nd(rng);
                decay[static_cast<size_t>(t) * H + hv] = ua(rng);
                beta[static_cast<size_t>(t) * H + hv] = ub(rng);
            }
            parent[t] = (t == 0u) ? -1 : static_cast<int32_t>(t - 1u);
        }
        if (branching && m >= 5u) {
            parent[3] = 1;
            parent[4] = 1;
        }
    }

    Args args(float* q, float* k, float* v, float* dec, float* bet, int32_t* parent_plane,
              const int32_t* parent_staging) const {
        using namespace vegajet::gdn::wire_pack;
        Args a;
        a.conv_source = reinterpret_cast<uint64_t>(conv.data());
        a.decay_source = reinterpret_cast<uint64_t>(decay.data());
        a.beta_source = reinterpret_cast<uint64_t>(beta.data());
        a.query = reinterpret_cast<uint64_t>(q);
        a.key = reinterpret_cast<uint64_t>(k);
        a.value = reinterpret_cast<uint64_t>(v);
        a.decay = reinterpret_cast<uint64_t>(dec);
        a.beta = reinterpret_cast<uint64_t>(bet);
        a.parent_source = reinterpret_cast<uint64_t>(parent_staging);
        a.parent = reinterpret_cast<uint64_t>(parent_plane);
        a.m = m;
        a.nk = kQueryHeads;
        a.nv = kValueHeads;
        a.conv_stride = stride;
        return a;
    }
};

// ---- part 1: descriptors and producer bytes ---------------------------------------------
void test_validation() {
    using namespace vegajet::gdn::wire_pack;
    std::printf("[wire] descriptor validation\n");
    Fixture f;
    f.build(12u, kConvStride, 7);
    std::vector<float> q(query_bytes(12) / 4), k(key_bytes(12) / 4), v(value_bytes(12) / 4);
    std::vector<float> dec(gate_bytes(12) / 4), bet(gate_bytes(12) / 4);
    std::vector<int32_t> p(12), ps(12);
    for (uint32_t i = 0; i < 12u; ++i) { p[i] = static_cast<int32_t>(i) - 1; ps[i] = p[i]; }
    const Args good = f.args(q.data(), k.data(), v.data(), dec.data(), bet.data(), p.data(), ps.data());
    std::string err;
    VJ_CHECK(validate(good, err));
    VJ_CHECK(err.empty());

    Args a = good;
    a.m = 7u;
    VJ_CHECK(!validate(a, err) && err.find("width") != std::string::npos);
    a = good;
    a.nk = 16u;
    VJ_CHECK(!validate(a, err) && err.find("nk") != std::string::npos);
    a = good;
    a.nv = 16u;
    VJ_CHECK(!validate(a, err));
    a = good;
    a.conv_stride = kConvStride - 1u;
    VJ_CHECK(!validate(a, err));
    a = good;
    a.query = 0;
    VJ_CHECK(!validate(a, err));
    a = good;
    a.parent = 0;
    VJ_CHECK(!validate(a, err));
    a = good;
    a.query = a.conv_source;  // destination aliases a source
    VJ_CHECK(!validate(a, err) && err.find("overlap") != std::string::npos);
    a = good;
    a.parent_source = a.parent;  // staging aliases its destination
    VJ_CHECK(!validate(a, err));
    a = good;
    a.decay_source = a.beta_source;  // two sources alias
    VJ_CHECK(!validate(a, err) && err.find("overlap") != std::string::npos);
    a = good;
    a.key = a.query + 2u;  // 4-byte misaligned
    VJ_CHECK(!validate(a, err) && err.find("aligned") != std::string::npos);
    a = good;
    a.parent_source = std::numeric_limits<uint64_t>::max() - 4u;
    VJ_CHECK(!validate(a, err));
    a = good;
    a.conv_source = std::numeric_limits<uint64_t>::max() - 8u;
    VJ_CHECK(!validate(a, err));
    // A destination placed in the conv row padding still overlaps the source range.
    a = good;
    a.query = a.conv_source + static_cast<uint64_t>(kConvStride) * 4u;
    VJ_CHECK(!validate(a, err) && err.find("overlap") != std::string::npos);
    std::printf("  descriptor cases rejected before any write\n");
}

void test_pack_bytes() {
    using namespace vegajet::gdn::wire_pack;
    std::printf("[wire] producer bytes, stride and stale-row discipline\n");
    for (uint32_t width : {5u, 8u, 12u}) {
        for (uint32_t stride : {static_cast<uint32_t>(kConvStride), 6208u}) {
            for (bool branching : {false, true}) {
                Fixture f;
                f.build(width, stride, static_cast<int>(width * 10 + stride), branching);
                const size_t qk = static_cast<size_t>(width) * K * D;
                const size_t vdim = static_cast<size_t>(width) * H * D;
                const size_t gate = static_cast<size_t>(width) * H;
                std::vector<float> q(12u * K * D, std::numeric_limits<float>::quiet_NaN());
                std::vector<float> k(12u * K * D, std::numeric_limits<float>::quiet_NaN());
                std::vector<float> v(12u * H * D, std::numeric_limits<float>::quiet_NaN());
                std::vector<float> dec(12u * H, std::numeric_limits<float>::quiet_NaN());
                std::vector<float> bet(12u * H, std::numeric_limits<float>::quiet_NaN());
                std::vector<int32_t> p(12u, std::numeric_limits<int32_t>::min());
                std::vector<int32_t> ps(f.parent);
                const Args a = f.args(q.data(), k.data(), v.data(), dec.data(), bet.data(),
                                      p.data(), ps.data());
                std::string err;
                VJ_CHECK(validate(a, err));
                HostBuffers hb;
                hb.conv = f.conv.data();
                hb.decay = f.decay.data();
                hb.beta = f.beta.data();
                hb.query = q.data();
                hb.key = k.data();
                hb.value = v.data();
                hb.decay_out = dec.data();
                hb.beta_out = bet.data();
                hb.parent = ps.data();
                hb.parent_out = p.data();
                VJ_CHECK(pack_host(a, hb, err));
                for (uint32_t t = 0; t < width; ++t) {
                    const float* row = f.conv.data() + static_cast<size_t>(t) * stride;
                    for (size_t i = 0; i < static_cast<size_t>(K) * D; ++i) {
                        VJ_CHECK(bits(q[t * static_cast<size_t>(K) * D + i]) == bits(row[i]));
                        VJ_CHECK(bits(k[t * static_cast<size_t>(K) * D + i]) ==
                                 bits(row[static_cast<size_t>(K) * D + i]));
                    }
                    for (size_t i = 0; i < static_cast<size_t>(H) * D; ++i) {
                        VJ_CHECK(bits(v[t * static_cast<size_t>(H) * D + i]) ==
                                 bits(row[2u * static_cast<size_t>(K) * D + i]));
                    }
                    for (uint32_t i = 0; i < H; ++i) {
                        VJ_CHECK(bits(dec[t * H + i]) == bits(f.decay[t * H + i]));
                        VJ_CHECK(bits(bet[t * H + i]) == bits(f.beta[t * H + i]));
                    }
                    VJ_CHECK(p[t] == f.parent[t]);
                }
                // The verifier reads exactly [0, width); those rows are now source-exact. Rows
                // outside the width stay poisoned, so a later narrower/wider cycle can never be
                // served stale data it did not produce.
                if (width < 12u) {
                    VJ_CHECK(std::isnan(q[static_cast<size_t>(width) * K * D]));
                    VJ_CHECK(std::isnan(v[static_cast<size_t>(width) * H * D]));
                    VJ_CHECK(p[width] == std::numeric_limits<int32_t>::min());
                }
                VJ_CHECK(q.size() == 12u * K * D && qk == static_cast<size_t>(width) * K * D &&
                         vdim == qk * H / K && gate == static_cast<size_t>(width) * H);
            }
        }
    }
    std::printf("  3 widths x 2 strides x chain/branch: byte-exact, bounded writes\n");
}

// ---- part 2: packed planes -> sealed WY vs sequential verifier ---------------------------
void test_numerics() {
    using namespace vegajet::gdn::wire_pack;
    std::printf("[wire] packed planes -> sealed WY vs sequential verifier (state+output)\n");
    uint64_t selected = 0;
    for (uint32_t width : {5u, 8u, 12u}) {
        for (bool branching : {false, true}) {
            if (branching && width == 5u) continue;
            Fixture f;
            f.build(width, kConvStride, static_cast<int>(1000 + width * 3 + (branching ? 1 : 0)),
                    branching);
            const size_t qk = static_cast<size_t>(width) * K * D;
            const size_t vdim = static_cast<size_t>(width) * H * D;
            const size_t gate = static_cast<size_t>(width) * H;
            std::vector<float> q(qk), k(qk), v(vdim), dec(gate), bet(gate);
            std::vector<int32_t> p(width), ps(f.parent);
            const Args a = f.args(q.data(), k.data(), v.data(), dec.data(), bet.data(), p.data(),
                                  ps.data());
            std::string err;
            VJ_CHECK(validate(a, err));
            HostBuffers hb;
            hb.conv = f.conv.data();
            hb.decay = f.decay.data();
            hb.beta = f.beta.data();
            hb.query = q.data();
            hb.key = k.data();
            hb.value = v.data();
            hb.decay_out = dec.data();
            hb.beta_out = bet.data();
            hb.parent = ps.data();
            hb.parent_out = p.data();
            VJ_CHECK(pack_host(a, hb, err));

            // Producer transparency: the packed planes are the fixture's semantic arrays.
            Inputs in;
            in.m = static_cast<int>(width);
            in.s = f.s0;
            in.q = q;
            in.k = k;
            in.v = v;
            in.g = dec;
            in.b = bet;
            in.parent = p;
            for (uint32_t t = 0; t < width; ++t) {
                const float* row = f.conv.data() + static_cast<size_t>(t) * kConvStride;
                for (size_t i = 0; i < static_cast<size_t>(K) * D; ++i) {
                    VJ_CHECK(bits(in.q[t * static_cast<size_t>(K) * D + i]) == bits(row[i]));
                    VJ_CHECK(bits(in.k[t * static_cast<size_t>(K) * D + i]) ==
                             bits(row[static_cast<size_t>(K) * D + i]));
                }
                for (size_t i = 0; i < static_cast<size_t>(H) * D; ++i) {
                    VJ_CHECK(bits(in.v[t * static_cast<size_t>(H) * D + i]) ==
                             bits(row[2u * static_cast<size_t>(K) * D + i]));
                }
            }

            const Oracle seq = chronological(in);
            std::vector<Record> records;
            std::vector<float> scratch;
            const std::vector<float> sealed = sealed_verify(in, records, scratch);
            const Diff od = compare(sealed, seq.output);
            VJ_CHECK(od.max_abs <= 1e-5);

            // Every accepted prefix (node id accepted-1) must reproduce the sequential state.
            double worst = 0.0;
            for (int a = 1; a <= in.m; ++a) {
                const std::vector<float> committed = commit(records, a);
                const Diff sd = compare(committed, seq.states[static_cast<size_t>(a - 1u)]);
                VJ_CHECK(sd.max_abs <= 1e-5);
                worst = std::max(worst, sd.max_abs);
            }
            // Root-only rollback: commit(0) restores S0 bit-for-bit.
            const Diff restore = compare(commit(records, 0), in.s);
            VJ_CHECK(restore.bit_mismatches == 0 && restore.max_abs == 0.0);
            // A partition/full walk must have produced a real advance.
            VJ_CHECK(compare(commit(records, in.m), in.s).max_abs > 1e-3);
            ++selected;
            std::printf("  packed M=%u tree=%u output_max_abs=%.9g worst_state_max_abs=%.9g "
                        "root_restore=exact\n",
                        width, branching ? 1u : 0u, od.max_abs, worst);
        }
    }
    VJ_CHECK(selected == 5u);
    std::printf("TREEWY_WIRE_PACK selected=%llu nonzero=1 widths=5/8/12\n",
                static_cast<unsigned long long>(selected));
}

}  // namespace

int main() {
    test_validation();
    test_pack_bytes();
    test_numerics();
    std::printf("PASS sealed GDN production-input producer: byte-exact packing and CPU "
                "state/output equivalence at root-only/partial/full prefixes; device parity "
                "UNTESTED, no GPU used.\n");
    return 0;
}
