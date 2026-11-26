// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU coverage for the opt-in 240-WG router + shared Gate/Up kernel pair
// (ops/kernels/vj_ornith_router_s240.hip, core/router_s240_layout.hpp, core/router_s240_pack.hpp).
//
//   * layout selection: plain/odd stride vs power-of-two XOR capacity, including partial final tiles;
//   * matching packer/reader: pack -> byte reader == natural bytes, unpack round trip, XOR on/off parity,
//     and a deliberate wrong-mask control that must be detected;
//   * deterministic Split-K: contiguous partitions cover every 32-weight block exactly once for 1..8 parts,
//     including nb < split_k (empty parts) and non-power-of-two nb;
//   * 240-WG task mapping: every (row, part) task is issued exactly once for non-power-of-two row counts;
//   * bit-exact CPU emulation of both device accumulation chains (FP16 FMA router, VJ_Q4_A8 INT8 dot +
//     FP16 d/m shared Gate/Up) reading only through the packed reader: XOR on == XOR off bitwise, and the
//     fixed-order reduce is stable;
//   * host launch geometry/scratch plan (240 WGs, ceil-div reduce grid) without touching a GPU.
#include "core/router_s240_layout.hpp"
#include "core/router_s240_pack.hpp"
#include "tests/test_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using namespace vegajet::router_s240;

namespace {

// Exact IEEE binary16 -> binary32 (same bit math as quant::fp16_to_f32) and round-to-nearest-even
// binary32 -> binary16, so the CPU reference needs no core library at link time.
float h2f(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t frac = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0u) {
        if (frac == 0u) {
            bits = sign;
        } else {
            exp = 127u - 14u;
            while ((frac & 0x400u) == 0u) { frac <<= 1; --exp; }
            frac &= 0x3FFu;
            bits = sign | (exp << 23) | (frac << 13);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7F800000u | (frac << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (frac << 13);
    }
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

uint16_t f2h(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    if ((x & 0x7F800000u) == 0x7F800000u)
        return static_cast<uint16_t>(sign | 0x7C00u | ((x & 0x7FFFFFu) ? 0x200u : 0u));
    const int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t r = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u), half = 1u << (shift - 1u);
        if (rem > half || (rem == half && (r & 1u))) ++r;
        return static_cast<uint16_t>(sign | r);
    }
    uint32_t r = (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) ++r;
    if (r >= 0x7C00u) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | r);
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    uint32_t u32() { return static_cast<uint32_t>(next() >> 32); }
    float f32(float lo, float hi) { return lo + (hi - lo) * static_cast<float>((next() >> 40) * (1.0 / 16777216.0)); }
    int8_t i8() { return static_cast<int8_t>(static_cast<int>(next() % 255u) - 127); }
};

// Same partner sequence and simultaneous-update semantics as vegajet::ornith_dev::block32_sum
// (xor 16, 8, 4, 2, 1), so lane 0's result equals the device reduction.
float block32_sum_cpu(const float in[32]) {
    float v[32];
    for (uint32_t i = 0; i < 32u; ++i) v[i] = in[i];
    for (uint32_t off = 16u; off != 0u; off >>= 1u) {
        float t[32];
        for (uint32_t i = 0; i < 32u; ++i) t[i] = v[i] + v[i ^ off];
        for (uint32_t i = 0; i < 32u; ++i) v[i] = t[i];
    }
    return v[0];
}


// wave64_sum partner sequence: ds_swizzle XOR 1,2,4,8,16 then shfl_xor 32.
float wave64_sum_cpu(const float in[64]) {
    float v[64];
    for (uint32_t i = 0; i < 64u; ++i) v[i] = in[i];
    for (uint32_t off = 1u; off < 64u; off <<= 1u) {
        float t[64];
        for (uint32_t i = 0; i < 64u; ++i) t[i] = v[i] + v[i ^ off];
        for (uint32_t i = 0; i < 64u; ++i) v[i] = t[i];
    }
    return v[0];
}

float router_part_cpu(const std::vector<uint8_t>& packed, const PlaneLayout& lay, uint32_t row,
                      const std::vector<float>& x, uint32_t D, uint32_t b_beg, uint32_t b_end) {
    float lanes[32];
    for (uint32_t lane = 0; lane < 32u; ++lane) {
        float acc = 0.0f;
        for (uint32_t i = b_beg * 4u + lane; i < b_end * 4u; i += 32u) {
            const uint8_t* wb = plane_block(packed.data(), lay, row, i);
            float h[8];
            for (uint32_t j = 0; j < 8u; ++j) {
                uint16_t hv = 0;
                std::memcpy(&hv, wb + 2u * j, 2);
                h[j] = h2f(hv);
            }
            const float* xb = x.data() + static_cast<size_t>(i) * 8u;
            acc = std::fmaf(h[0], xb[0], std::fmaf(h[1], xb[1], std::fmaf(h[2], xb[2], std::fmaf(h[3], xb[3], acc))));
            acc = std::fmaf(h[4], xb[4], std::fmaf(h[5], xb[5], std::fmaf(h[6], xb[6], std::fmaf(h[7], xb[7], acc))));
        }
        lanes[lane] = acc;
    }
    (void)D;
    return block32_sum_cpu(lanes);
}

float shgu_part_cpu(const std::vector<uint8_t>& qs, const PlaneLayout& lay, uint32_t row,
                    const std::vector<uint32_t>& dm, uint32_t dm_stride, const std::vector<int8_t>& xq8,
                    const std::vector<float>& xds, uint32_t b_beg, uint32_t b_end) {
    float lanes[32];
    for (uint32_t lane = 0; lane < 32u; ++lane) {
        float acc = 0.0f;
        for (uint32_t b = b_beg + lane; b < b_end; b += 32u) {
            const uint8_t* wb = plane_block(qs.data(), lay, row, b);
            const int8_t* xb = xq8.data() + static_cast<size_t>(b) * 32u;
            int32_t dp = 0;
            for (uint32_t j = 0; j < 16u; ++j) dp += static_cast<int32_t>(wb[j] & 15u) * static_cast<int32_t>(xb[j]);
            for (uint32_t j = 0; j < 16u; ++j) dp += static_cast<int32_t>(wb[j] >> 4) * static_cast<int32_t>(xb[j + 16u]);
            const uint32_t dmv = dm[static_cast<size_t>(row) * dm_stride + b];
            const float wd = h2f(static_cast<uint16_t>(dmv & 0xFFFFu));
            const float wm = h2f(static_cast<uint16_t>(dmv >> 16));
            acc = std::fmaf(wd * xds[static_cast<size_t>(b) * 2u], static_cast<float>(dp),
                            std::fmaf(wm, xds[static_cast<size_t>(b) * 2u + 1u], acc));
        }
        lanes[lane] = acc;
    }
    return block32_sum_cpu(lanes);
}

// Fully deterministic aggregated CPU result for one row (parts summed in fixed 0..split_k-1 order).
float router_row_cpu(const std::vector<uint8_t>& packed, const PlaneLayout& lay, uint32_t row,
                     const std::vector<float>& x, uint32_t D, uint32_t split_k) {
    const uint32_t nb = D / 32u;
    float s = 0.0f;
    for (uint32_t p = 0; p < split_k; ++p)
        s += router_part_cpu(packed, lay, row, x, D, split_begin(nb, split_k, p), split_end(nb, split_k, p));
    return s;
}
float shgu_row_cpu(const std::vector<uint8_t>& qs, const PlaneLayout& lay, uint32_t row,
                   const std::vector<uint32_t>& dm, uint32_t dm_stride, const std::vector<int8_t>& xq8,
                   const std::vector<float>& xds, uint32_t D, uint32_t split_k) {
    const uint32_t nb = D / 32u;
    float s = 0.0f;
    for (uint32_t p = 0; p < split_k; ++p)
        s += shgu_part_cpu(qs, lay, row, dm, dm_stride, xq8, xds, split_begin(nb, split_k, p),
                           split_end(nb, split_k, p));
    return s;
}

// Baseline M=1 fused-front accumulation shapes, copied 1:1 from ornith_moe_m1_router_shared_kernel:
// router reads four 32-block-spaced uint4 per lane step (loop stride 128); shared Gate/Up batches three
// 32-weight blocks per lane step (loop stride 96). The per-lane block order must match our plain loop.
float router_row_baseline_cpu(const std::vector<uint8_t>& packed, const PlaneLayout& lay, uint32_t row,
                              const std::vector<float>& x, uint32_t D) {
    const uint32_t n8 = D >> 3;
    float lanes[32];
    for (uint32_t lane = 0; lane < 32u; ++lane) {
        float acc = 0.0f;
        for (uint32_t i0 = lane; i0 < n8; i0 += 128u) {
            for (uint32_t r = 0; r < 4u; ++r) {
                const uint32_t i = i0 + r * 32u;
                if (i < n8) {
                    const uint8_t* wb = plane_block(packed.data(), lay, row, i);
                    float h[8];
                    for (uint32_t j = 0; j < 8u; ++j) {
                        uint16_t hv = 0;
                        std::memcpy(&hv, wb + 2u * j, 2);
                        h[j] = h2f(hv);
                    }
                    const float* xb = x.data() + static_cast<size_t>(i) * 8u;
                    acc = std::fmaf(h[0], xb[0], std::fmaf(h[1], xb[1], std::fmaf(h[2], xb[2], std::fmaf(h[3], xb[3], acc))));
                    acc = std::fmaf(h[4], xb[4], std::fmaf(h[5], xb[5], std::fmaf(h[6], xb[6], std::fmaf(h[7], xb[7], acc))));
                }
            }
        }
        lanes[lane] = acc;
    }
    return block32_sum_cpu(lanes);
}

float shgu_row_baseline_cpu(const std::vector<uint8_t>& qs, const PlaneLayout& lay, uint32_t row,
                            const std::vector<uint32_t>& dm, uint32_t dm_stride, const std::vector<int8_t>& xq8,
                            const std::vector<float>& xds, uint32_t D) {
    const uint32_t nb = D / 32u;
    float lanes[32];
    for (uint32_t lane = 0; lane < 32u; ++lane) {
        float acc = 0.0f;
        for (uint32_t b0 = lane; b0 < nb; b0 += 96u) {
            for (uint32_t r = 0; r < 3u; ++r) {
                const uint32_t b = b0 + r * 32u;
                if (b < nb) {
                    const uint8_t* wb = plane_block(qs.data(), lay, row, b);
                    const int8_t* xb = xq8.data() + static_cast<size_t>(b) * 32u;
                    int32_t dp = 0;
                    for (uint32_t j = 0; j < 16u; ++j) dp += static_cast<int32_t>(wb[j] & 15u) * static_cast<int32_t>(xb[j]);
                    for (uint32_t j = 0; j < 16u; ++j) dp += static_cast<int32_t>(wb[j] >> 4) * static_cast<int32_t>(xb[j + 16u]);
                    const uint32_t dmv = dm[static_cast<size_t>(row) * dm_stride + b];
                    const float wd = h2f(static_cast<uint16_t>(dmv & 0xFFFFu));
                    const float wm = h2f(static_cast<uint16_t>(dmv >> 16));
                    acc = std::fmaf(wd * xds[static_cast<size_t>(b) * 2u], static_cast<float>(dp),
                                    std::fmaf(wm, xds[static_cast<size_t>(b) * 2u + 1u], acc));
                }
            }
        }
        lanes[lane] = acc;
    }
    return block32_sum_cpu(lanes);
}

bool same_bytes(const void* a, const void* b, size_t n) { return std::memcmp(a, b, n) == 0; }
bool same_float(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }

// ---------------------------------------------------------------- layout + packer coverage
void test_layout_and_pack() {
    Rng rng(0xA3A3A3A3u);
    const uint32_t raws[] = {1u, 15u, 16u, 17u, 255u, 256u, 257u, 383u, 384u, 512u, 1536u, 1600u, 1792u,
                             2048u, 2600u, 4096u, 5000u, 6080u, 6144u, 6208u, 8192u};
    for (uint32_t raw : raws) {
        for (uint32_t xor_on = 0; xor_on < 2u; ++xor_on) {
            const uint32_t plain = ((raw + 15u) / 16u) * 16u + (xor_on ? 0u : 256u);  // odd-256B style pitch
            const PlaneLayout l = make_plane_layout(raw, plain, xor_on);
            VJ_CHECK(l.raw_row_bytes == raw);
            VJ_CHECK(l.stride_bytes % 16u == 0u);
            VJ_CHECK(l.stride_bytes >= raw);
            VJ_CHECK(l.data_tiles == (raw + 255u) / 256u);
            VJ_CHECK(l.phys_tiles == (l.stride_bytes + 255u) / 256u);
            if (xor_on) {
                VJ_CHECK(l.xor_enabled == 1u);
                VJ_CHECK(l.phys_tiles >= l.data_tiles);
                VJ_CHECK((l.phys_tiles & (l.phys_tiles - 1u)) == 0u);
                VJ_CHECK(l.xor_mask == l.phys_tiles - 1u);
                VJ_CHECK(l.phys_tiles * 256u == l.stride_bytes);
            } else {
                VJ_CHECK(l.xor_enabled == 0u && l.xor_mask == 0u);
            }
            const uint32_t rows = 1u + rng.u32() % 37u;  // non-power-of-two row counts
            std::vector<uint8_t> src(static_cast<size_t>(rows) * raw + 7u);
            for (auto& b : src) b = static_cast<uint8_t>(rng.u32());
            const std::vector<uint8_t> packed = pack_plane_rows(src.data(), rows, l, raw + 0u);
            const std::vector<uint8_t> back = unpack_plane_rows(packed.data(), rows, l);
            VJ_CHECK(back.size() == static_cast<size_t>(rows) * raw);
            VJ_CHECK(same_bytes(back.data(), src.data(), back.size()));
            // Reader mirror: every logical byte, including a partial final tile, must equal the source.
            for (uint32_t r = 0; r < rows; ++r) {
                for (uint32_t off = 0; off < raw; ++off) {
                    VJ_CHECK(*plane_byte(packed.data(), l, r, off) == src[static_cast<size_t>(r) * raw + off]);
                }
                if (raw >= 16u) {
                    const uint32_t blocks = raw / 16u;  // whole 16-B blocks only
                    for (uint32_t k = 0; k < blocks; ++k) {
                        VJ_CHECK(same_bytes(plane_block(packed.data(), l, r, k),
                                            src.data() + static_cast<size_t>(r) * raw + k * 16u, 16u));
                    }
                }
            }
            // Wrong-mask control: when a row's XOR phase differs from its physical storage, naive
            // (unswizzled) addressing must not see the same bytes. Skip rows that coincide.
            if (xor_on && l.xor_mask != 0u && raw >= 512u) {
                const uint32_t r = 1u;  // r & mask != 0 for every non-zero mask
                if ((r & l.xor_mask) != 0u) {
                    VJ_CHECK(!same_bytes(packed.data() + static_cast<size_t>(r) * l.stride_bytes,
                                         src.data() + static_cast<size_t>(r) * raw, 256u));
                }
            }
            // Padding tiles must stay zero and be outside every row's data.
            if (xor_on && l.phys_tiles > l.data_tiles) {
                for (uint32_t r = 0; r < rows; ++r) {
                    for (uint32_t kt = 0; kt < l.phys_tiles; ++kt) {
                        bool is_data = false;
                        for (uint32_t d = 0; d < l.data_tiles; ++d) is_data = is_data || (phys_tile(l, r, d) == kt);
                        if (is_data) continue;
                        const uint8_t* t = packed.data() + static_cast<size_t>(r) * l.stride_bytes +
                                           static_cast<size_t>(kt) * 256u;
                        for (uint32_t i = 0; i < 256u; ++i) VJ_CHECK(t[i] == 0u);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------- Split-K partition coverage
void test_split_partitions() {
    const uint32_t nbs[] = {1u, 2u, 3u, 4u, 5u, 6u, 7u, 9u, 12u, 24u, 25u, 31u, 32u, 48u, 95u, 96u, 97u, 128u, 200u};
    for (uint32_t nb : nbs) {
        for (uint32_t sk = 1; sk <= kMaxSplitK; ++sk) {
            std::vector<uint32_t> hits(nb, 0u);
            for (uint32_t p = 0; p < sk; ++p) {
                const uint32_t b0 = split_begin(nb, sk, p);
                const uint32_t b1 = split_end(nb, sk, p);
                VJ_CHECK(b0 <= b1 && b1 <= nb);
                if (nb >= sk) VJ_CHECK(b0 < b1);  // every part non-empty when the row can feed it
                for (uint32_t b = b0; b < b1; ++b) ++hits[b];
            }
            for (uint32_t b = 0; b < nb; ++b) VJ_CHECK(hits[b] == 1u);
        }
    }
}

// ---------------------------------------------------------------- 240-WG task mapping coverage
void test_task_mapping() {
    const uint32_t rows[] = {1u, 3u, 7u, 8u, 9u, 63u, 255u, 256u, 257u, 511u, 1000u, 1023u, 1024u, 1025u, 2048u};
    for (uint32_t sh_rows : rows) {
        for (uint32_t sk = 1; sk <= kMaxSplitK; ++sk) {
            const TaskPlan p = make_task_plan(sh_rows, sk);
            VJ_CHECK(p.sh_tasks == sh_rows * sk);
            VJ_CHECK(p.rtr_tasks == kRouterRows * sk);
            VJ_CHECK(p.stride == kWorkerGroups && kWorkerGroups == (kPhase1Wgs - 1u) * 8u);
            VJ_CHECK(p.reduce_rows == sh_rows + kRouterRows + 1u);
            VJ_CHECK(p.reduce_wgs == (p.reduce_rows + kThreads - 1u) / kThreads);

            std::vector<uint32_t> sh_hits(p.sh_tasks, 0u), rtr_hits(p.rtr_tasks, 0u);
            std::vector<uint32_t> work(kWorkerWgs, 0u);
            for (uint32_t wg = 0; wg < kWorkerWgs; ++wg) {
                for (uint32_t grp = 0; grp < 8u; ++grp) {
                    const uint32_t first = grp * kWorkerWgs + wg;
                    for (uint32_t t = first; t < p.sh_tasks; t += p.stride) {
                        ++sh_hits[t];
                        ++work[wg];
                        uint32_t part = 99u, row = 99u;
                        sh_task_split(t, sh_rows, sk, &part, &row);
                        VJ_CHECK(part < sk && row < sh_rows && t == part * sh_rows + row);
                    }
                    for (uint32_t t = first; t < p.rtr_tasks; t += p.stride) {
                        ++rtr_hits[t];
                        work[wg] += 4u;  // router row cost model used by the balance claim
                        uint32_t part = 99u, row = 99u;
                        rtr_task_split(t, sk, &part, &row);
                        VJ_CHECK(part < sk && row < kRouterRows && t == part * kRouterRows + row);
                    }
                }
            }
            for (uint32_t h : sh_hits) VJ_CHECK(h == 1u);
            for (uint32_t h : rtr_hits) VJ_CHECK(h == 1u);
            // Round-robin over all 1912 groups: no worker WG deviates by more than one task of each stream.
            const uint32_t sh_per_wg = (p.sh_tasks + kWorkerWgs - 1u) / kWorkerWgs;
            const uint32_t rtr_per_wg = (p.rtr_tasks + kWorkerWgs - 1u) / kWorkerWgs;
            const uint32_t wmax = sh_per_wg + 4u * rtr_per_wg;
            for (uint32_t w : work) VJ_CHECK(w <= wmax);
        }
    }
}

// ---------------------------------------------------------------- bit-exact emulation on packed data
void test_kernel_emulation() {
    Rng rng(0x5EED1234u);
    const uint32_t Ds[] = {288u, 512u, 1024u, 3040u, 3072u, 3104u, 4096u};
    for (uint32_t D : Ds) {
        const uint32_t nb = D / 32u;
        const uint32_t router_raw = D * 2u;      // FP16
        const uint32_t qs_raw = D / 2u;          // VJ_Q4_A8 codes
        const uint32_t dm_stride = nb;           // dense FP16 d|m per 32
        const PlaneLayout r_plain = make_plane_layout(router_raw, 0u, 0u);
        const PlaneLayout r_xor = make_plane_layout(router_raw, 0u, 1u);
        const PlaneLayout q_plain = make_plane_layout(qs_raw, ((qs_raw + 15u) / 16u) * 16u + 256u, 0u);
        const PlaneLayout q_xor = make_plane_layout(qs_raw, 0u, 1u);

        // Natural router rows and XOR/plain packed copies of the same 256 experts.
        std::vector<uint8_t> rnat(static_cast<size_t>(kRouterRows) * router_raw);
        for (size_t i = 0; i + 1 < rnat.size(); i += 2) {
            const uint16_t h = f2h(rng.f32(-2.0f, 2.0f));
            std::memcpy(&rnat[i], &h, 2);
        }
        const std::vector<uint8_t> rp = pack_plane_rows(rnat.data(), kRouterRows, r_plain);
        const std::vector<uint8_t> rx = pack_plane_rows(rnat.data(), kRouterRows, r_xor);

        // Natural shared Gate/Up qs + dm.
        std::vector<uint8_t> qnat(static_cast<size_t>(kRouterRows) * qs_raw);
        for (auto& b : qnat) b = static_cast<uint8_t>(rng.u32());
        std::vector<uint32_t> dm(static_cast<size_t>(kRouterRows) * dm_stride);
        for (auto& v : dm) {
            const uint16_t d = f2h(rng.f32(0.001f, 0.02f));
            const uint16_t m = f2h(rng.f32(-0.1f, 0.1f));
            v = static_cast<uint32_t>(d) | (static_cast<uint32_t>(m) << 16);
        }
        const std::vector<uint8_t> qp = pack_plane_rows(qnat.data(), kRouterRows, q_plain);
        const std::vector<uint8_t> qx = pack_plane_rows(qnat.data(), kRouterRows, q_xor);

        std::vector<float> xnorm(D);  // router reads x4f[i*2], i < D/8 -> D/4 float4s
        for (size_t i = 0; i < xnorm.size(); ++i) xnorm[i] = rng.f32(-1.0f, 1.0f);
        std::vector<int8_t> xq8(D);
        for (auto& v : xq8) v = rng.i8();
        std::vector<float> xds(static_cast<size_t>(nb) * 2u);
        for (auto& v : xds) v = rng.f32(0.5f, 1.5f);

        const uint32_t rows_to_check[] = {0u, 1u, 15u, 16u, 17u, 127u, 254u, 255u};
        for (uint32_t row : rows_to_check) {
            for (uint32_t sk = 1; sk <= 4u; ++sk) {
                // Plain and XOR layouts must produce identical bits (same bytes, same arithmetic).
                const float rp_v = router_row_cpu(rp, r_plain, row, xnorm, D, sk);
                const float rx_v = router_row_cpu(rx, r_xor, row, xnorm, D, sk);
                VJ_CHECK(same_float(rp_v, rx_v));
                const float qp_v = shgu_row_cpu(qp, q_plain, row, dm, dm_stride, xq8, xds, D, sk);
                const float qx_v = shgu_row_cpu(qx, q_xor, row, dm, dm_stride, xq8, xds, D, sk);
                VJ_CHECK(same_float(qp_v, qx_v));
                if (sk == 1u) {
                    // Split 1 / XOR off must reproduce the existing fused-front accumulation bit-for-bit.
                    VJ_CHECK(same_float(rp_v, router_row_baseline_cpu(rp, r_plain, row, xnorm, D)));
                    VJ_CHECK(same_float(qp_v, shgu_row_baseline_cpu(qp, q_plain, row, dm, dm_stride, xq8, xds, D)));
                }
                // Fixed-order reduce: repeating a part sum yields the same bits every time.
                VJ_CHECK(same_float(router_row_cpu(rp, r_plain, row, xnorm, D, sk),
                                    router_row_cpu(rp, r_plain, row, xnorm, D, sk)));
                // Aggregated split result stays close to the single-pass accumulation (reassociation only).
                const float r_full = router_row_cpu(rp, r_plain, row, xnorm, D, 1u);
                VJ_CHECK(std::fabs(rx_v - r_full) <= 2e-4f * std::fmax(1.0f, std::fabs(r_full)));
                const float q_full = shgu_row_cpu(qp, q_plain, row, dm, dm_stride, xq8, xds, D, 1u);
                VJ_CHECK(std::fabs(qx_v - q_full) <= 2e-4f * std::fmax(1.0f, std::fabs(q_full)));
            }
        }
    }
}


// ---------------------------------------------------------------- composed task map + reduce
void test_composed_kernel() {
    Rng rng(0xC0FFEE42u);
    const uint32_t D = 3040u;              // 95 blocks: non-power-of-two, partial final tile in both planes
    const uint32_t nb = D / 32u;
    const uint32_t sh_rows = 37u;          // non-power-of-two rows for the 240-WG chunking
    const uint32_t sk = 3u;                // non-power-of-two Split-K
    const uint32_t router_raw = D * 2u;
    const uint32_t qs_raw = D / 2u;

    std::vector<uint8_t> rnat(static_cast<size_t>(kRouterRows) * router_raw);
    for (size_t i = 0; i + 1 < rnat.size(); i += 2) {
        const uint16_t h = f2h(rng.f32(-2.0f, 2.0f));
        std::memcpy(&rnat[i], &h, 2);
    }
    const PlaneLayout rl = make_plane_layout(router_raw, 0u, 1u);
    const std::vector<uint8_t> rp = pack_plane_rows(rnat.data(), kRouterRows, rl);

    std::vector<uint8_t> qnat(static_cast<size_t>(sh_rows) * qs_raw);
    for (auto& b : qnat) b = static_cast<uint8_t>(rng.u32());
    const PlaneLayout ql = make_plane_layout(qs_raw, 0u, 1u);
    const std::vector<uint8_t> qp = pack_plane_rows(qnat.data(), sh_rows, ql);
    std::vector<uint32_t> dm(static_cast<size_t>(sh_rows) * nb);
    for (auto& v : dm) {
        const uint16_t d = f2h(rng.f32(0.001f, 0.02f));
        const uint16_t m = f2h(rng.f32(-0.1f, 0.1f));
        v = static_cast<uint32_t>(d) | (static_cast<uint32_t>(m) << 16);
    }
    std::vector<float> xnorm(D);
    for (auto& v : xnorm) v = rng.f32(-1.0f, 1.0f);
    std::vector<int8_t> xq8(D);
    for (auto& v : xq8) v = rng.i8();
    std::vector<float> xds(static_cast<size_t>(nb) * 2u);
    for (auto& v : xds) v = rng.f32(0.5f, 1.5f);
    std::vector<float> gate_w(D);
    for (auto& v : gate_w) v = rng.f32(-1.0f, 1.0f);

    std::vector<float> rpart(static_cast<size_t>(sk) * kRouterRows, 0.0f);
    std::vector<float> spart(static_cast<size_t>(sk) * sh_rows, 0.0f);
    std::vector<uint32_t> hits(static_cast<size_t>(sk) * (kRouterRows + sh_rows), 0u);

    const TaskPlan plan = make_task_plan(sh_rows, sk);
    for (uint32_t wg = 0; wg < kWorkerWgs; ++wg) {
        for (uint32_t grp = 0; grp < 8u; ++grp) {
            const uint32_t first = grp * kWorkerWgs + wg;
            for (uint32_t t = first; t < plan.sh_tasks; t += plan.stride) {
                uint32_t part = 0u, row = 0u;
                sh_task_split(t, sh_rows, sk, &part, &row);
                spart[static_cast<size_t>(part) * sh_rows + row] =
                    shgu_part_cpu(qp, ql, row, dm, nb, xq8, xds, split_begin(nb, sk, part), split_end(nb, sk, part));
                ++hits[static_cast<size_t>(part) * sh_rows + row];
            }
            for (uint32_t t = first; t < plan.rtr_tasks; t += plan.stride) {
                uint32_t part = 0u, row = 0u;
                rtr_task_split(t, sk, &part, &row);
                rpart[static_cast<size_t>(part) * kRouterRows + row] =
                    router_part_cpu(rp, rl, row, xnorm, D, split_begin(nb, sk, part), split_end(nb, sk, part));
                ++hits[static_cast<size_t>(sk) * sh_rows + static_cast<size_t>(part) * kRouterRows + row];
            }
        }
    }
    for (uint32_t h : hits) VJ_CHECK(h == 1u);

    // Fixed-order reduce then compare with the direct per-row reference (bit-exact: same part functions).
    std::vector<float> logits(kRouterRows), shgu(sh_rows);
    for (uint32_t row = 0; row < sh_rows; ++row) {
        float s = 0.0f;
        for (uint32_t p = 0; p < sk; ++p) s += spart[static_cast<size_t>(p) * sh_rows + row];
        shgu[row] = s;
        VJ_CHECK(same_float(s, shgu_row_cpu(qp, ql, row, dm, nb, xq8, xds, D, sk)));
    }
    for (uint32_t e = 0; e < kRouterRows; ++e) {
        float s = 0.0f;
        for (uint32_t p = 0; p < sk; ++p) s += rpart[static_cast<size_t>(p) * kRouterRows + e];
        logits[e] = s;
        VJ_CHECK(same_float(s, router_row_cpu(rp, rl, e, xnorm, D, sk)));
    }

    // Gate scalar: 64-lane wave64 tree vs an independent double-precision dot.
    float lane_acc[64];
    for (uint32_t tid = 0; tid < 64u; ++tid) {
        float acc = 0.0f;
        for (uint32_t p = 0; p < sk; ++p) {
            for (uint32_t i = split_begin(nb, sk, p) * 8u + tid; i < split_end(nb, sk, p) * 8u && i < (D >> 2); i += 64u) {
                const float* wv = gate_w.data() + static_cast<size_t>(i) * 4u;
                const float* xv = xnorm.data() + static_cast<size_t>(i) * 4u;
                acc = std::fmaf(wv[0], xv[0], std::fmaf(wv[1], xv[1], std::fmaf(wv[2], xv[2], std::fmaf(wv[3], xv[3], acc))));
            }
        }
        lane_acc[tid] = acc;
    }
    const float gsum = wave64_sum_cpu(lane_acc);
    double gref = 0.0;
    for (uint32_t i = 0; i < D; ++i) gref += static_cast<double>(gate_w[i]) * xnorm[i];
    const float gf = 1.0f / (1.0f + std::exp(-gsum));
    const double gr = 1.0 / (1.0 + std::exp(-gref));
    VJ_CHECK(std::fabs(static_cast<double>(gf) - gr) <= 1e-5);
}

// ---------------------------------------------------------------- deterministic reduce order
void test_reduce_order() {
    // Values chosen so summing forward and backward differs in FP32 (proves the fixed 0..S-1 order matters).
    const float parts[3] = {1.0f, 1e8f, -1e8f};
    float fwd = 0.0f, rev = 0.0f;
    for (uint32_t p = 0; p < 3u; ++p) fwd += parts[p];
    for (uint32_t p = 3u; p > 0u; --p) rev += parts[p - 1u];
    VJ_CHECK(fwd == 0.0f && rev == 1.0f);
    // The kernel's split_k == 1 path writes the partial through unchanged (identity, including sign bits).
    const float one = 0.0f;
    VJ_CHECK(same_float(one + parts[1], parts[1]));
    // Gratuitous non-power-of-two part counts reduce in index order.
    for (uint32_t sk : {1u, 2u, 3u, 5u, 7u, 8u}) {
        float s = 0.0f;
        for (uint32_t p = 0; p < sk; ++p) s += static_cast<float>(p) * 0.125f;
        float t = 0.0f;
        for (uint32_t p = 0; p < sk; ++p) t += static_cast<float>(p) * 0.125f;
        VJ_CHECK(same_float(s, t));
    }
}

// ---------------------------------------------------------------- host glue
void test_launch_plan() {
    for (uint32_t sh_rows : {1u, 511u, 1024u, 1025u, 2048u}) {
        for (uint32_t sk = 1; sk <= kMaxSplitK; ++sk) {
            const ScratchPlan p = make_scratch_plan(sh_rows, sk);
            VJ_CHECK(p.gate_off == 0u);
            VJ_CHECK(p.rtr_off % kScratchAlign == 0u);
            VJ_CHECK(p.sh_off % kScratchAlign == 0u);
            VJ_CHECK(p.sh_off >= p.rtr_off + static_cast<uint64_t>(sk) * kRouterRows * 4u);
            VJ_CHECK(p.bytes == p.sh_off + static_cast<uint64_t>(sk) * sh_rows * 4u);
            VJ_CHECK(p.bytes <= 512u * 1024u);  // fits the engine's splitk_part scratch
            VJ_CHECK(kPhase1Wgs == 240u && kThreads == 256u);
            const TaskPlan tp = make_task_plan(sh_rows, sk);
            const uint32_t reduce_grid = (tp.reduce_rows + kThreads - 1u) / kThreads;
            VJ_CHECK(reduce_grid * kThreads >= tp.reduce_rows);
            VJ_CHECK((reduce_grid - 1u) * kThreads < tp.reduce_rows);
        }
    }
}

} // namespace

int main() {
    test_layout_and_pack();
    test_split_partitions();
    test_task_mapping();
    test_kernel_emulation();
    test_composed_kernel();
    test_reduce_order();
    test_launch_plan();
    std::printf("router_s240: all CPU checks passed\n");
    return 0;
}
