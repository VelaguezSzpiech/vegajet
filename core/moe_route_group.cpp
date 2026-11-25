// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/moe_route_group.hpp"

#include "core/q4_dot8_repack.hpp"

#include <cmath>
#include <cstring>

namespace vegajet {

using namespace moe_grp;

void moe_route_group_ref(const int32_t* top_ids, uint32_t M, uint32_t min_group, MoeRouteGroup* out) {
    const uint32_t R = M * kTopK;
    out->exp_cnt.assign(kNumExperts, 0u);
    out->exp_off.assign(kNumExperts + 1u, 0u);
    out->uniq.clear();
    out->items.clear();
    out->routes.assign(R, 0u);
    for (uint32_t i = 0; i < R; ++i) ++out->exp_cnt[static_cast<uint32_t>(top_ids[i]) & 255u];
    uint32_t grouped = 0;
    for (uint32_t e = 0; e < kNumExperts; ++e) {
        const uint32_t c = out->exp_cnt[e];
        out->exp_off[e + 1u] = out->exp_off[e] + c;
        if (c != 0u) out->uniq.push_back(e);
        const uint32_t nit = items_for_count(c, min_group);
        if (nit != 0u) grouped += c;
        for (uint32_t k = 0; k < nit; ++k) {
            const uint32_t len = (c - k * kItemRoutes) < kItemRoutes ? (c - k * kItemRoutes) : kItemRoutes;
            out->items.push_back(item_pack(e, out->exp_off[e] + k * kItemRoutes, len));
        }
    }
    std::vector<uint32_t> cur(out->exp_off.begin(), out->exp_off.end() - 1);
    for (uint32_t i = 0; i < R; ++i) out->routes[cur[static_cast<uint32_t>(top_ids[i]) & 255u]++] = i;  // stable
    out->hdr[kHdrUnique] = static_cast<uint32_t>(out->uniq.size());
    out->hdr[kHdrItems] = static_cast<uint32_t>(out->items.size());
    out->hdr[kHdrGroupedRoutes] = grouped;
    out->hdr[kHdrRoutes] = R;
}

namespace {

inline uint32_t rd32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
inline int32_t sdot4(uint32_t a, uint32_t b, int32_t c) {
    for (uint32_t i = 0; i < 4u; ++i)
        c += static_cast<int32_t>(static_cast<int8_t>(a >> (8u * i))) * static_cast<int32_t>(static_cast<int8_t>(b >> (8u * i)));
    return c;
}
inline uint32_t dec4(uint32_t ws) { return dot8_decode4_iq3(ws & 0x0F0F0F0Fu); }

} // namespace

int32_t moe_down_block_dp(const MoeDownProblem& p, uint32_t e, uint32_t row, uint32_t lane16, uint32_t r) {
    const uint8_t* w = p.w_qs + (static_cast<uint64_t>(e) * p.D + row) * p.w_stride + lane16 * 16u;
    const uint8_t* x = reinterpret_cast<const uint8_t*>(p.mid_q8) + static_cast<uint64_t>(r) * kCols + lane16 * 32u;
    int32_t dp = 0;
    for (uint32_t j = 0; j < 4u; ++j) dp = sdot4(dec4(rd32(w + 4u * j)), rd32(x + 4u * j), dp);
    for (uint32_t j = 0; j < 4u; ++j) dp = sdot4(dec4(rd32(w + 4u * j) >> 4), rd32(x + 16u + 4u * j), dp);
    return dp;
}

float moe_down_block_gdb(const MoeDownProblem& p, uint32_t e, uint32_t row, uint32_t lane16) {
    const uint8_t* s = p.w_sc + (static_cast<uint64_t>(e) * p.D + row) * p.s_stride + (lane16 >> 3) * 8u;
    const uint32_t sc = (rd32(s) >> ((lane16 & 7u) * 4u)) & 0xFu;
    const float d = u2f(rd32(s + 4));
    const float t = d * (0.5f + static_cast<float>(sc));
    return t * 0.5f;
}

float moe_sum16_ref(const float v_in[16]) {
    float v[16];
    std::memcpy(v, v_in, sizeof(v));
    for (uint32_t m = 8u; m != 0u; m >>= 1) {
        float n[16];
        for (uint32_t l = 0; l < 16u; ++l) n[l] = v[l] + v[l ^ m];
        std::memcpy(v, n, sizeof(v));
    }
    return v[0];
}

namespace {

inline float slot_scale(const MoeDownProblem& p, uint32_t r, uint32_t e, uint32_t row, uint32_t lane16) {
    const float ew = p.top_w[r];
    const float gdb = moe_down_block_gdb(p, e, row, lane16);
    const float xd = p.mid_d[static_cast<uint64_t>(r) * 16u + lane16];
    const float t = ew * gdb;
    return t * xd;
}
inline float finish(const MoeDownProblem& p, uint32_t m, uint32_t row, float total) {
    const float sh = p.sh_down_in ? p.sh_down_in[static_cast<uint64_t>(m) * p.D + row] : 0.0f;
    return total + sh;
}

} // namespace

void moe_down_per_token_ref(const MoeDownProblem& p, float* out) {
    for (uint32_t m = 0; m < p.M; ++m)
        for (uint32_t row = 0; row < p.D; ++row) {
            float lanes[16];
            for (uint32_t l = 0; l < 16u; ++l) {
                float total = 0.0f;
                for (uint32_t s = 0; s < kTopK; ++s) {
                    const uint32_t r = m * kTopK + s;
                    const uint32_t e = static_cast<uint32_t>(p.top_ids[r]);
                    const int32_t dp = moe_down_block_dp(p, e, row, l, r);
                    total = std::fmaf(slot_scale(p, r, e, row, l), static_cast<float>(dp), total);
                }
                lanes[l] = total;
            }
            out[static_cast<uint64_t>(m) * p.D + row] = finish(p, m, row, moe_sum16_ref(lanes));
        }
}

namespace {

// Device order of the grouped kernels: items -> row tiles -> lanes; weights decoded once per (item, row, lane)
// into 8 words (decode32_iq3), then one dot per route (dot32_dec).
template <typename F>
void for_each_grouped(const MoeDownProblem& p, const MoeRouteGroup& g, F&& f) {
    for (uint32_t ii = 0; ii < g.hdr[kHdrItems]; ++ii) {
        const uint32_t it = g.items[ii];
        const uint32_t e = item_expert(it), start = item_start(it), len = item_len(it);
        for (uint32_t row = 0; row < p.D; ++row)
            for (uint32_t l = 0; l < 16u; ++l) {
                const uint8_t* w = p.w_qs + (static_cast<uint64_t>(e) * p.D + row) * p.w_stride + l * 16u;
                uint32_t wd[8];
                for (uint32_t j = 0; j < 4u; ++j) { wd[j] = dec4(rd32(w + 4u * j)); wd[4u + j] = dec4(rd32(w + 4u * j) >> 4); }
                for (uint32_t t = 0; t < len; ++t) {
                    const uint32_t r = g.routes[start + t];
                    const uint8_t* x = reinterpret_cast<const uint8_t*>(p.mid_q8) + static_cast<uint64_t>(r) * kCols + l * 32u;
                    int32_t dp = 0;
                    for (uint32_t j = 0; j < 8u; ++j) dp = sdot4(wd[j], rd32(x + 4u * j), dp);
                    f(e, row, l, r, dp);
                }
            }
    }
}

} // namespace

void moe_down_grouped_dp_ref(const MoeDownProblem& p, const MoeRouteGroup& g, int32_t* dp) {
    for_each_grouped(p, g, [&](uint32_t, uint32_t row, uint32_t l, uint32_t r, int32_t v) {
        dp[(static_cast<uint64_t>(r) * p.D + row) * 16u + l] = v;
    });
}

void moe_down_combine_dp_ref(const MoeDownProblem& p, const uint32_t* exp_cnt, uint32_t min_group,
                             const int32_t* dp, float* out) {
    for (uint32_t m = 0; m < p.M; ++m)
        for (uint32_t row = 0; row < p.D; ++row) {
            float lanes[16];
            for (uint32_t l = 0; l < 16u; ++l) {
                float total = 0.0f;
                for (uint32_t s = 0; s < kTopK; ++s) {
                    const uint32_t r = m * kTopK + s;
                    const uint32_t e = static_cast<uint32_t>(p.top_ids[r]);
                    const bool wide = exp_cnt[e & 255u] >= min_group;
                    const int32_t d = wide ? dp[(static_cast<uint64_t>(r) * p.D + row) * 16u + l]
                                           : moe_down_block_dp(p, e, row, l, r);
                    total = std::fmaf(slot_scale(p, r, e, row, l), static_cast<float>(d), total);
                }
                lanes[l] = total;
            }
            out[static_cast<uint64_t>(m) * p.D + row] = finish(p, m, row, moe_sum16_ref(lanes));
        }
}

void moe_down_grouped_f32_ref(const MoeDownProblem& p, const MoeRouteGroup& g, float* part) {
    // Gather the 16 lane products of each (route, row), then reduce like the device.
    std::vector<float> lanes(static_cast<size_t>(g.hdr[kHdrRoutes]) * p.D * 16u, 0.0f);
    std::vector<uint8_t> seen(static_cast<size_t>(g.hdr[kHdrRoutes]) * p.D, 0u);
    for_each_grouped(p, g, [&](uint32_t e, uint32_t row, uint32_t l, uint32_t r, int32_t v) {
        const float prod = slot_scale(p, r, e, row, l) * static_cast<float>(v);
        lanes[(static_cast<size_t>(r) * p.D + row) * 16u + l] = prod;
        seen[static_cast<size_t>(r) * p.D + row] = 1u;
    });
    for (size_t i = 0; i < seen.size(); ++i)
        if (seen[i]) part[i] = moe_sum16_ref(&lanes[i * 16u]);
}

void moe_down_combine_f32_ref(const MoeDownProblem& p, const uint32_t* exp_cnt, uint32_t min_group,
                              const float* part, float* out) {
    for (uint32_t m = 0; m < p.M; ++m)
        for (uint32_t row = 0; row < p.D; ++row) {
            float lanes[16];
            for (uint32_t l = 0; l < 16u; ++l) {
                float total = 0.0f;
                for (uint32_t s = 0; s < kTopK; ++s) {
                    const uint32_t r = m * kTopK + s;
                    const uint32_t e = static_cast<uint32_t>(p.top_ids[r]);
                    if (exp_cnt[e & 255u] >= min_group) continue;
                    const int32_t d = moe_down_block_dp(p, e, row, l, r);
                    total = std::fmaf(slot_scale(p, r, e, row, l), static_cast<float>(d), total);
                }
                lanes[l] = total;
            }
            float total = moe_sum16_ref(lanes);
            for (uint32_t s = 0; s < kTopK; ++s) {
                const uint32_t r = m * kTopK + s;
                if (exp_cnt[static_cast<uint32_t>(p.top_ids[r]) & 255u] >= min_group)
                    total += part[static_cast<uint64_t>(r) * p.D + row];
            }
            out[static_cast<uint64_t>(m) * p.D + row] = finish(p, m, row, total);
        }
}

MoeDownBytes moe_down_bytes(const uint32_t* exp_cnt, uint32_t D, uint32_t min_group, bool exact) {
    constexpr uint64_t kCodes = 256u, kScales = 16u, kRow = kCodes + kScales;
    MoeDownBytes b;
    for (uint32_t e = 0; e < kNumExperts; ++e) {
        const uint64_t c = exp_cnt[e];
        b.per_token += c * D * kRow;
        const uint32_t nit = items_for_count(static_cast<uint32_t>(c), min_group);
        if (nit == 0u) {
            b.combine_reads += c * D * kRow;
            continue;
        }
        b.grouped_weights += static_cast<uint64_t>(nit) * D * (exact ? kCodes : kRow);
        b.buf_write += c * D * (exact ? 64u : 4u);
        b.combine_reads += c * D * (exact ? (64u + kScales) : 4u);
    }
    return b;
}

} // namespace vegajet
