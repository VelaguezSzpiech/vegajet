// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/host_q4 + the Q4 pre-gather row format: block layout, exactness for representable values, the per-row error
// bound (amax / 14 plus the FP16 scale rounding), table round trip, and OpenMP pregather == serial reference for
// 4-bit PLE/Engram rows in both layouts.
#include "core/host_q4.hpp"
#include "core/ple_pregather.hpp"
#include "tests/test_common.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace vegajet::hostq4;

namespace {
// A row whose values are exact multiples of the stored block scale (digit * scale) must round-trip bitwise.
std::vector<float> exact_row(uint32_t dim, uint32_t block_index) {
    // digits -7..7, scale 2^-3 = 0.125, block amax 0.875 -> s = amax / 7 = 0.125 exactly.
    const float scale = std::ldexp(1.0f, -3);
    std::vector<float> v(dim);
    for (uint32_t i = 0; i < dim; ++i) v[i] = scale * static_cast<float>(static_cast<int>((i + block_index) % 15) - 7);
    return v;
}
}  // namespace

int main() {
    VJ_CHECK(row_bytes(0) == 0 && row_bytes(31) == 0 && row_bytes(32) == 18 && row_bytes(128) == 72);
    const uint32_t dim = 128;
    std::vector<uint8_t> packed(row_bytes(dim));

    // Exact rows round trip bitwise.
    for (uint32_t b = 0; b < 3; ++b) {
        const std::vector<float> v = exact_row(dim, b);
        VJ_CHECK(pack_row(v.data(), dim, packed.data()));
        std::vector<float> back(dim);
        VJ_CHECK(unpack_row(packed.data(), dim, back.data()));
        VJ_CHECK(std::memcmp(v.data(), back.data(), dim * 4) == 0);
        // A row of zeros packs to all-zero payload (scale 0) and unpacks to exact zeros.
        std::vector<float> z(dim, 0.0f);
        VJ_CHECK(pack_row(z.data(), dim, packed.data()));
        VJ_CHECK(unpack_row(packed.data(), dim, back.data()));
        for (float x : back) VJ_CHECK(x == 0.0f);
    }

    // Random rows: error stays inside the analytic bound, and the packing actually shrinks the row.
    std::mt19937 g(11);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    double worst_ratio = 0.0;
    for (int rep = 0; rep < 64; ++rep) {
        std::vector<float> v(dim);
        for (auto& x : v) x = u(g);
        v[rep % dim] = 0.75f;  // fix the block amax so the bound is tight
        VJ_CHECK(pack_row(v.data(), dim, packed.data()));
        std::vector<float> back(dim);
        VJ_CHECK(unpack_row(packed.data(), dim, back.data()));
        const double bound = max_abs_error_bound(v.data(), dim);
        double err = 0;
        for (uint32_t i = 0; i < dim; ++i) err = std::max(err, double(std::fabs(v[i] - back[i])));
        VJ_CHECK(bound > 0.0 && err <= bound * (1.0 + 1e-9));
        worst_ratio = std::max(worst_ratio, err / bound);
        VJ_CHECK(row_bytes(dim) == dim / 2 + dim / 16);  // 4.5 bits per value
    }

    // Table round trip [rows][dim].
    const uint32_t rows = 37;
    std::vector<float> table(size_t(rows) * dim);
    for (auto& x : table) x = u(g) * 3.0f;
    std::vector<uint8_t> pt;
    VJ_CHECK(pack_table(table.data(), rows, dim, &pt));
    VJ_CHECK(pt.size() == size_t(rows) * row_bytes(dim));
    std::vector<float> back(table.size());
    VJ_CHECK(unpack_table(pt.data(), rows, dim, back.data()));
    for (uint32_t r = 0; r < rows; ++r) {
        const double bound = max_abs_error_bound(&table[size_t(r) * dim], dim);
        for (uint32_t i = 0; i < dim; ++i)
            VJ_CHECK(std::fabs(table[size_t(r) * dim + i] - back[size_t(r) * dim + i]) <= bound * (1.0 + 1e-9) + 1e-12);
    }
    VJ_CHECK(!pack_table(table.data(), rows, 31, &pt));  // dim must be a multiple of 32

    // Q4 pre-gather (OpenMP) == serial reference, both layouts, several M.
    const uint32_t L = 12, vocab = 256;
    std::vector<std::vector<uint8_t>> tabs(L, std::vector<uint8_t>(vocab * row_bytes(dim)));
    for (auto& t : tabs) {
        for (size_t i = 0; i < t.size(); i += row_bytes(dim)) {
            std::vector<float> v(dim);
            for (auto& x : v) x = u(g);
            VJ_CHECK(pack_row(v.data(), dim, t.data() + i));
        }
    }
    vegajet::ple::PleTables tbl{L, vocab, dim, vegajet::ple::RowFmt::Q4, {}};
    for (auto& t : tabs) tbl.layer.push_back(t.data());
    for (uint32_t M : {1u, 7u, 33u}) {
        std::vector<int32_t> tok(M);
        for (auto& x : tok) x = static_cast<int32_t>(g() % vocab);
        for (auto lay : {vegajet::ple::Layout::LayerMajor, vegajet::ple::Layout::TokenMajor}) {
            std::vector<float> a(size_t(L) * M * dim), b(a.size());
            std::string err;
            VJ_CHECK(vegajet::ple::pregather(tbl, tok.data(), M, a.data(), lay, &err));
            VJ_CHECK(vegajet::ple::pregather_reference(tbl, tok.data(), M, b.data(), lay));
            VJ_CHECK(std::memcmp(a.data(), b.data(), a.size() * 4) == 0);
        }
    }
    // Tokens are validated before anything is written.
    {
        std::vector<int32_t> tok{0, static_cast<int32_t>(vocab)};
        std::vector<float> a(size_t(L) * 2 * dim, -1.0f);
        std::string err;
        VJ_CHECK(!vegajet::ple::pregather(tbl, tok.data(), 2, a.data(), vegajet::ple::Layout::LayerMajor, &err));
        VJ_CHECK(err.find("out of range") != std::string::npos);
        for (float x : a) VJ_CHECK(x == -1.0f);
    }
    std::printf("  worst q4 error / analytic bound = %.3f (dim=%u, %u rows, %u layers)\n", worst_ratio, dim, rows, L);
    std::printf("test_host_q4: 4-bit round trip + bound + Q4 pregather == reference OK\nPASS\n");
    return 0;
}
