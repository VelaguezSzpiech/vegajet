// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU emulation of the v1.2 S4 narrow multi-column verify GEMV (ornith_gemv_q4a8_mn_kernel) against the
// two M=1 decode paths it must match bit for bit:
//   * ornith_gemv_q4a8_m1_w32_kernel            (dense M=1 decode, 1 warp per row, two rounds of 2 blocks)
//   * gemv_q4a8_persistent inside the M=1 super  (same order: blocks lane, lane+32, lane+64, lane+96)
// and against the existing M<=8 verify GEMM (ornith_gemm_q4a8_m8_kernel, 32-block LDS tiles).
//
// The three kernels only agree if the per-row accumulation order is identical: one FMA per 32-wide weight
// block, blocks assigned to lane (b % 32) in ascending order, then one 32-lane butterfly. A split-K order
// (two half-K accumulators summed at the end) is used as a negative control: it must NOT match.
#include "tests/test_common.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr uint32_t kLanes = 32;

// XOR butterfly over the 32 lane partials, exactly the ornith_dev::block32_sum sequence.
float butterfly(float v[kLanes]) {
    for (uint32_t off = 16u; off > 0u; off >>= 1u) {
        float t[kLanes];
        for (uint32_t i = 0u; i < kLanes; ++i) t[i] = v[i] + v[i ^ off];
        for (uint32_t i = 0u; i < kLanes; ++i) v[i] = t[i];
    }
    return v[0];
}

struct Q4 {
    uint32_t rows, cols;
    std::vector<uint8_t> qs;   // 16 bytes per 32-wide block
    std::vector<float> wd, wm; // per block: d (scale) and min (offset) as the kernel dequantises them
    uint32_t nb() const { return cols >> 5; }
    const uint8_t* blk(uint32_t row, uint32_t b) const { return qs.data() + (uint64_t(row) * nb() + b) * 16u; }
};

struct X8 {
    uint32_t M, cols;
    std::vector<int8_t> q8;
    std::vector<float> d, bsum;  // per block
    uint32_t nb() const { return cols >> 5; }
    const int8_t* blk(uint32_t t, uint32_t b) const { return q8.data() + uint64_t(t) * cols + uint64_t(b) * 32u; }
};

// One weighted block: dp = 8-step sdot4 chain (low nibbles then high), then the two-term FMA.
float block_mac(const uint8_t* w, const int8_t* x, float wd, float wm, float xd, float xbs, float acc) {
    int32_t dp = 0;
    for (uint32_t k = 0u; k < 4u; ++k) {           // low nibbles, uint4 byte order (v.x, v.y, v.z, v.w)
        int32_t s = 0;
        for (uint32_t j = 0u; j < 4u; ++j) s += int32_t(w[k * 4u + j] & 0x0F) * int32_t(x[k * 4u + j]);
        dp += s;
    }
    for (uint32_t k = 0u; k < 4u; ++k) {           // high nibbles, same byte order
        int32_t s = 0;
        for (uint32_t j = 0u; j < 4u; ++j) s += int32_t((w[k * 4u + j] >> 4) & 0x0F) * int32_t(x[16u + k * 4u + j]);
        dp += s;
    }
    return std::fma(wd * xd, static_cast<float>(dp), std::fma(wm, xbs, acc));
}

// M=1 w32: lane l owns blocks l, l+32, ... ascending (the kernel runs two rounds of two blocks; same order).
float row_w32(const Q4& w, const X8& x, uint32_t row, uint32_t t) {
    float acc[kLanes] = {0.0f};
    for (uint32_t r = 0u; r * 64u < w.nb(); ++r) {
        for (uint32_t half = 0u; half < 2u; ++half) {
            for (uint32_t l = 0u; l < kLanes; ++l) {
                const uint32_t b = l + r * 64u + half * 32u;
                if (b < w.nb())
                    acc[l] = block_mac(w.blk(row, b), x.blk(t, b), w.wd[row * w.nb() + b], w.wm[row * w.nb() + b],
                                       x.d[t * x.nb() + b], x.bsum[t * x.nb() + b], acc[l]);
            }
        }
    }
    return butterfly(acc);
}

// M<=16 verify GEMM (m8): 32-block LDS tiles, lanes >= tile_nb sit out; accumulation stays ascending.
float row_m8(const Q4& w, const X8& x, uint32_t row, uint32_t t) {
    float acc[kLanes] = {0.0f};
    for (uint32_t b_base = 0u; b_base < w.nb(); b_base += 32u) {
        const uint32_t tile_nb = std::min<uint32_t>(32u, w.nb() - b_base);
        if (tile_nb == 0u) continue;
        for (uint32_t l = 0u; l < kLanes; ++l) {
            if (l >= tile_nb) continue;
            const uint32_t b = b_base + l;
            acc[l] = block_mac(w.blk(row, b), x.blk(t, b), w.wd[row * w.nb() + b], w.wm[row * w.nb() + b],
                               x.d[t * x.nb() + b], x.bsum[t * x.nb() + b], acc[l]);
        }
    }
    return butterfly(acc);
}

// S4 narrow kernel: same lane map, one weight pass, M accumulators (order per row unchanged).
std::vector<float> rows_mn(const Q4& w, const X8& x, uint32_t row) {
    std::vector<float> acc(size_t(x.M) * kLanes, 0.0f);
    for (uint32_t b = 0u; b < w.nb(); ++b) {
        const uint32_t l = b & 31u;
        for (uint32_t t = 0u; t < x.M; ++t)
            acc[size_t(t) * kLanes + l] =
                block_mac(w.blk(row, b), x.blk(t, b), w.wd[row * w.nb() + b], w.wm[row * w.nb() + b],
                          x.d[t * x.nb() + b], x.bsum[t * x.nb() + b], acc[size_t(t) * kLanes + l]);
    }
    std::vector<float> out(x.M);
    for (uint32_t t = 0u; t < x.M; ++t) out[t] = butterfly(&acc[size_t(t) * kLanes]);
    return out;
}

// Negative control: split-K order (two halves summed at the end) must not match the M=1 order.
float row_splitk(const Q4& w, const X8& x, uint32_t row, uint32_t t) {
    float acc[kLanes] = {0.0f};
    float half1[kLanes] = {0.0f};
    float half2[kLanes] = {0.0f};
    const uint32_t mid = w.nb() / 2u;
    for (uint32_t l = 0u; l < kLanes; ++l)
        for (uint32_t b = l; b < w.nb(); b += 32u) {
            float& a = (b < mid) ? half1[l] : half2[l];
            a = block_mac(w.blk(row, b), x.blk(t, b), w.wd[row * w.nb() + b], w.wm[row * w.nb() + b],
                          x.d[t * x.nb() + b], x.bsum[t * x.nb() + b], a);
        }
    for (uint32_t l = 0u; l < kLanes; ++l) acc[l] = half1[l] + half2[l];
    return butterfly(acc);
}

} // namespace

int main() {
    std::mt19937 rng(20261126u);
    std::uniform_int_distribution<int> byte(0, 255), i8(-127, 127);
    std::uniform_real_distribution<float> sc(0.001f, 0.02f), off(-0.05f, 0.05f);
    int checks = 0, differing = 0;

    for (uint32_t cols : {3072u, 4096u}) {
        for (uint32_t M = 2u; M <= 16u; ++M) {
            Q4 w;
            w.rows = 24u;
            w.cols = cols;
            w.qs.resize(size_t(w.rows) * w.nb() * 16u);
            w.wd.resize(size_t(w.rows) * w.nb());
            w.wm.resize(size_t(w.rows) * w.nb());
            X8 x;
            x.M = M;
            x.cols = cols;
            x.q8.resize(size_t(M) * cols);
            x.d.resize(size_t(M) * x.nb());
            x.bsum.resize(size_t(M) * x.nb());
            for (auto& v : w.qs) v = uint8_t(byte(rng));
            for (auto& v : w.wd) v = sc(rng);
            for (auto& v : w.wm) v = off(rng);
            for (auto& v : x.q8) v = int8_t(i8(rng));
            for (auto& v : x.d) v = sc(rng);
            for (auto& v : x.bsum) v = float(i8(rng)) * 0.5f;

            for (uint32_t row = 0u; row < w.rows; ++row) {
                const std::vector<float> mn = rows_mn(w, x, row);
                for (uint32_t t = 0u; t < M; ++t) {
                    const float a = row_w32(w, x, row, t);
                    const float b = row_m8(w, x, row, t);
                    const float c = mn[t];
                    VJ_CHECK(std::memcmp(&a, &b, sizeof(float)) == 0);  // verify GEMM == M=1 decode
                    VJ_CHECK(std::memcmp(&a, &c, sizeof(float)) == 0);  // S4 narrow == M=1 decode
                    const float sk = row_splitk(w, x, row, t);
                    if (std::memcmp(&a, &sk, sizeof(float)) != 0) ++differing;
                    ++checks;
                }
            }
        }
    }
    // The negative control proves the comparison is order-sensitive (a split-K sum changes the last bits).
    VJ_CHECK(differing > 0);
    std::printf("[gemv_q4a8_mn_ref] %d row outputs bit-identical across M=1 w32 / m8 / narrow-M paths "
                "(%d/%d split-K controls differ)\n", checks, differing, checks);
    std::printf("test_gemv_q4a8_mn_ref: all checks passed\n");
    return 0;
}
