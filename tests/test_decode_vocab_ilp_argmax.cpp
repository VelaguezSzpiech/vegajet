// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Host emulation of ornith_vocab_argmax_q8_ilp_kernel (V1.2 D3, VEGAJET_V12_VOCAB_ILP) against the
// sequential ornith_vocab_argmax_q8_kernel chain: same chunk/scale indexing, same wave64 butterfly,
// same per-wave and per-WG argmax reduction.
// The two variants only differ in fp32 summation order, so the contract checked here is:
//   * identical chunk coverage (each of the n16 16-byte chunks is used exactly once),
//   * identical argmax id and value (bit-identical when two rows tie exactly, because then every
//     row goes through the same sequence and the lowest row wins in both variants),
//   * logits within 4 ulp on random data.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int D     = 3072;      // cols
constexpr int N16   = D / 16;    // 16-byte chunks per row (192)
constexpr int LANES = 64;

float chunk_dot(const int8_t* w, const int8_t* x, int chunk) {
    int dp = 0;
    for (int j = 0; j < 16; ++j) dp += int(w[chunk * 16 + j]) * int(x[chunk * 16 + j]);
    return float(dp);
}

// ornith_vocab_argmax_q8_kernel: one dependent chain per lane.
float row_seq(const int8_t* w, const _Float16* d, const int8_t* x, const float* xd, int lane) {
    float acc = 0.0f;
    for (int i = lane; i < N16; i += 64)
        acc = std::fma(float(d[i >> 1]) * xd[i >> 1], chunk_dot(w, x, i), acc);
    return acc;
}

// ornith_vocab_argmax_q8_ilp_kernel: 2-chunk step (both chunks independent) + single-chunk tail.
float row_ilp(const int8_t* w, const _Float16* d, const int8_t* x, const float* xd, int lane) {
    float acc0 = 0.0f, acc1 = 0.0f;
    int i = lane;
    while (i + 64 < N16) {
        acc0 = std::fma(float(d[i >> 1]) * xd[i >> 1], chunk_dot(w, x, i), acc0);
        acc1 = std::fma(float(d[(i >> 1) + 32]) * xd[(i >> 1) + 32], chunk_dot(w, x, i + 64), acc1);
        i += 128;
    }
    if (i < N16)
        acc0 = std::fma(float(d[i >> 1]) * xd[i >> 1], chunk_dot(w, x, i), acc0);
    return acc0 + acc1;
}

// vegajet::ornith_dev::wave64_sum: 5 ds_swizzle butterflies (xor 1,2,4,8,16) then shfl_xor 32.
float wave64_sum(std::array<float, LANES> v) {
    for (int k = 1; k < LANES; k *= 2) {
        const std::array<float, LANES> p = v;
        for (int l = 0; l < LANES; ++l) v[l] = p[l] + p[l ^ k];
    }
    return v[0];
}

float row_logit(const int8_t* w, const _Float16* d, const int8_t* x, const float* xd, bool ilp) {
    std::array<float, LANES> v{};
    for (int l = 0; l < LANES; ++l) v[l] = ilp ? row_ilp(w, d, x, xd, l) : row_seq(w, d, x, xd, l);
    return wave64_sum(v);
}

// Per-WG argmax exactly as both kernels do it: waves 0..3 own rows 4*wg+wave, stride grid*4,
// strict ">" so the lowest row index keeps a tie.
struct Pick { float v; int id; };
Pick wg_pick(const std::vector<float>& logit, int rows, int grid, int wg, int row_off) {
    Pick best{-1e30f, -1};
    for (int wave = 0; wave < 4; ++wave)
        for (int r = 4 * wg + wave; r < rows; r += grid * 4)
            if (logit[size_t(r)] > best.v) { best.v = logit[size_t(r)]; best.id = r + row_off; }
    return best;
}

// Reordering three chunk terms inside a lane (c0+c2)+c1 instead of ((c0+c1)+c2) perturbs the
// logit by a few eps of the accumulated magnitude, not of the logit itself (which can sit near
// zero through cancellation), so the bound is taken against sum|terms| of that row.
float row_mag(const int8_t* w, const _Float16* d, const int8_t* x, const float* xd) {
    float mag = 0.0f;
    for (int lane = 0; lane < LANES; ++lane)
        for (int i = lane; i < N16; i += 64)
            mag += std::fabs(float(d[i >> 1]) * xd[i >> 1] * chunk_dot(w, x, i));
    return mag;
}

bool within_bound(float a, float b, float mag) {
    if (std::memcmp(&a, &b, 4) == 0) return true;
    return std::fabs(a - b) <= 16.0f * 1.1920929e-7f * mag;
}

}  // namespace

int main() {
    constexpr int ROWS = 400, GRID = 7, ROW_OFF = 71;

    std::mt19937 rng(1313);
    std::vector<int8_t>   w(size_t(ROWS) * D), x(D);
    std::vector<_Float16> d(size_t(ROWS) * (D / 32));
    std::array<float, D / 32> xd{};
    for (auto& z : w) z = int8_t(int(rng() % 255) - 127);
    for (auto& z : x) z = int8_t(int(rng() % 255) - 127);
    for (auto& z : d) z = _Float16((int(rng() % 2001) - 1000) / 999.3f);
    for (auto& z : xd)  z = (int(rng() % 2001) - 1000) / 997.1f;

    // 1) chunk coverage: every 16-byte chunk of a row is consumed exactly once, at the scale index
    //    the sequential chain uses (this is what pins ds[(i>>1)+32] to chunk i+64).
    {
        std::array<int, N16> used{};
        for (int lane = 0; lane < LANES; ++lane) {
            int i = lane;
            while (i + 64 < N16) {
                ++used[i]; ++used[i + 64];
                if ((i >> 1) + 32 != ((i + 64) >> 1)) return 1;  // scale index of chunk i+64
                i += 128;
            }
            if (i < N16) ++used[i];
        }
        int total = 0;
        for (int c = 0; c < N16; ++c) { if (used[c] != 1) return 2; total += used[c]; }
        if (total != N16) return 3;
    }

    for (int ties = 0; ties < 2; ++ties) {
        if (ties) {  // every row identical -> exact ties, lowest row must win in both variants
            for (int r = 1; r < ROWS; ++r) {
                std::memcpy(&w[size_t(r) * D], w.data(), size_t(D));
                std::memcpy(&d[size_t(r) * (D / 32)], d.data(), size_t(D / 32) * sizeof(d[0]));
            }
        }
        std::vector<float> seq(static_cast<std::size_t>(ROWS), 0.0f);
        std::vector<float> ilp(static_cast<std::size_t>(ROWS), 0.0f);
        std::vector<float> mag(static_cast<std::size_t>(ROWS), 0.0f);
        float worst_rel = 0.0f;
        for (int r = 0; r < ROWS; ++r) {
            const int8_t* wr = &w[size_t(r) * D];
            const _Float16* dr = &d[size_t(r) * (D / 32)];
            seq[size_t(r)] = row_logit(wr, dr, x.data(), xd.data(), false);
            ilp[size_t(r)] = row_logit(wr, dr, x.data(), xd.data(), true);
            mag[size_t(r)] = row_mag(wr, dr, x.data(), xd.data());
            if (ties) {
                if (std::memcmp(&seq[size_t(r)], &ilp[size_t(r)], 4) != 0) return 4;
            } else {
                if (!within_bound(seq[size_t(r)], ilp[size_t(r)], mag[size_t(r)])) return 5;
                const float rel = std::fabs(seq[size_t(r)] - ilp[size_t(r)]) / mag[size_t(r)];
                if (rel > worst_rel) worst_rel = rel;
            }
        }
        for (int wg = 0; wg < GRID; ++wg) {
            const Pick a = wg_pick(seq, ROWS, GRID, wg, ROW_OFF);
            const Pick b = wg_pick(ilp, ROWS, GRID, wg, ROW_OFF);
            if (a.id != b.id) return 6;
            if (ties ? std::memcmp(&a.v, &b.v, 4) != 0
                     : !within_bound(a.v, b.v, mag[size_t(a.id - ROW_OFF)])) return 7;
            if (ties && a.id != 4 * wg + ROW_OFF) return 8;  // lowest row of that wave keeps the tie
        }
        if (!ties) {
            std::printf("random rows: max |seq-ilp| / sum|terms| = %.3g (bound 16 eps = %.3g)\n",
                        double(worst_rel), double(16.0f * 1.1920929e-7f));
            if (worst_rel > 16.0f * 1.1920929e-7f) return 9;
        }
    }

    std::printf("vocab ILP argmax: %d rows x %d chunks, sequential vs 2-way ILP chain agree "
                "(same argmax id, logits within eps*sum|terms|, exact ties bit-identical, lowest row wins)\n",
                ROWS, N16);
    return 0;
}
