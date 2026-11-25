// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/ple_pregather: the OpenMP one-shot gather (48 layers x M tokens into one contiguous buffer) is bitwise equal
// to the serial element-by-element reference for F32 / F16 / Q8_0 rows and both layouts; bad input is rejected
// without writing; FP16 conversion is exact (normals, subnormals, inf, signed zero); throughput is reported.
#include "core/ple_pregather.hpp"
#include "tests/test_common.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

using namespace vegajet::ple;

int main() {
    // FP16 conversion spot checks.
    VJ_CHECK(half_to_float(0x3C00) == 1.0f && half_to_float(0xC000) == -2.0f && half_to_float(0x7BFF) == 65504.0f);
    VJ_CHECK(half_to_float(0x0001) == std::ldexp(1.0f, -24) && std::isinf(half_to_float(0x7C00)));
    VJ_CHECK(std::signbit(half_to_float(0x8000)) && half_to_float(0x8000) == 0.0f && std::isnan(half_to_float(0x7E00)));

    const uint32_t L = 48, vocab = 4096, dim = 256;
    std::mt19937 rng(17);
    std::printf("  OpenMP threads: %d\n", max_threads());
    for (RowFmt f : {RowFmt::F32, RowFmt::F16, RowFmt::Q8_0}) {
        const size_t rb = row_bytes(f, dim);
        std::vector<std::vector<uint8_t>> tables(L, std::vector<uint8_t>(vocab * rb));
        for (auto& t : tables) {
            for (size_t i = 0; i < t.size(); ++i) t[i] = static_cast<uint8_t>(rng());
            if (f != RowFmt::F32) {  // keep FP16 fields finite (exponent < 31) so bitwise compare is meaningful
                const size_t stride = f == RowFmt::F16 ? 2 : 34;
                for (size_t i = 0; i + 1 < t.size(); i += stride) t[i + 1] &= 0x7B;
            } else {
                for (size_t i = 0; i + 3 < t.size(); i += 4) t[i + 3] &= 0x3F;  // finite F32
            }
        }
        PleTables pt{L, vocab, dim, f, {}};
        for (auto& t : tables) pt.layer.push_back(t.data());
        for (uint32_t M : {1u, 7u, 512u}) {
            std::vector<int32_t> tok(M);
            for (auto& x : tok) x = static_cast<int32_t>(rng() % vocab);
            for (Layout lay : {Layout::LayerMajor, Layout::TokenMajor}) {
                std::vector<float> a(size_t(L) * M * dim, -1.0f), b(a.size(), -2.0f);
                std::string err;
                VJ_CHECK(pregather(pt, tok.data(), M, a.data(), lay, &err));
                VJ_CHECK(pregather_reference(pt, tok.data(), M, b.data(), lay));
                VJ_CHECK(std::memcmp(a.data(), b.data(), a.size() * 4) == 0);
            }
        }
        // Timing at M = 512 (informational).
        std::vector<int32_t> tok(512);
        for (auto& x : tok) x = static_cast<int32_t>(rng() % vocab);
        std::vector<float> out(size_t(L) * 512 * dim);
        const auto t0 = std::chrono::steady_clock::now();
        const int reps = 5;
        for (int r = 0; r < reps; ++r) VJ_CHECK(pregather(pt, tok.data(), 512, out.data(), Layout::LayerMajor));
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
        std::printf("  %s: 48 x 512 x %u -> %.1f MiB FP32 in %.2f ms (%.1f GB/s written)\n",
                    f == RowFmt::F32 ? "F32 " : f == RowFmt::F16 ? "F16 " : "Q8_0", dim, out.size() * 4.0 / (1 << 20), ms,
                    out.size() * 4.0 / (ms * 1e6));
        // Bad input: out-of-range token -> false, output untouched.
        std::vector<int32_t> bad = {1, 2, static_cast<int32_t>(vocab)};
        std::vector<float> o(size_t(L) * 3 * dim, 7.0f);
        std::string err;
        VJ_CHECK(!pregather(pt, bad.data(), 3, o.data(), Layout::LayerMajor, &err) && err.find("out of range") != std::string::npos);
        for (float v : o) VJ_CHECK(v == 7.0f);
        bad[2] = -1;
        VJ_CHECK(!pregather(pt, bad.data(), 3, o.data(), Layout::LayerMajor, &err));
        PleTables broken = pt;
        broken.layer.pop_back();
        VJ_CHECK(!pregather(broken, tok.data(), 1, o.data(), Layout::LayerMajor, &err));
    }
    PleTables q8{1, 1, 48, RowFmt::Q8_0, {nullptr}};
    std::string err;
    VJ_CHECK(row_bytes(RowFmt::Q8_0, 48) == 0 && !pregather(q8, nullptr, 0, nullptr, Layout::LayerMajor, &err));
    std::printf("test_ple_pregather: OpenMP gather == scalar reference (3 formats x 2 layouts x 3 batch sizes) OK\nPASS\n");
    return 0;
}
