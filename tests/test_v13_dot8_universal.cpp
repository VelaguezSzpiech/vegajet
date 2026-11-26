// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Universal VJ_Q4_DOT8 repacker (core/q4_dot8_repack.*): one entry point for IQ3_XXS and IQ2_S, odd-256B
// rows and the opt-in TCC XOR tile swizzle phys_k_blk = k_blk ^ (row & 15). Checks the swizzle index math
// round-trips exactly, the FP32 dequant of every repacked row matches the ggml reference (cos_sim >= 0.999
// on random and structured blocks), and the CPU row dot product agrees. Pure CPU.
#include "core/ggml_iq_tables.hpp"
#include "core/q4_dot8_repack.hpp"
#include "core/quant_repacker.hpp"
#include "tests/test_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using namespace vegajet;
namespace TB = vegajet::ggml_tables;

namespace {

std::mt19937_64 rng(0xD07A8A13ULL);
uint32_t rnd(uint32_t n) { return static_cast<uint32_t>(rng() % n); }

uint16_t random_fp16() {
    const uint32_t sign = rnd(2) << 15;
    const uint32_t exp = (rnd(16) == 0) ? 0u : 1u + rnd(29);
    return static_cast<uint16_t>(sign | (exp << 10) | rnd(1024));
}

void random_iq3xxs_block(uint8_t* b) {
    const uint16_t d = random_fp16();
    std::memcpy(b, &d, 2);
    for (int i = 0; i < 64; ++i) b[2 + i] = static_cast<uint8_t>(rnd(256));
    for (int ib = 0; ib < 8; ++ib) {
        const uint32_t aux = rnd(1u << 28) | (rnd(16) << 28);
        std::memcpy(b + 66 + 4 * ib, &aux, 4);
    }
}

void random_iq2s_block(uint8_t* b) {
    const uint16_t d = random_fp16();
    std::memcpy(b, &d, 2);
    for (int i = 2; i < 82; ++i) b[i] = static_cast<uint8_t>(rnd(256));
}

// Structured blocks: degenerate inputs a random generator rarely produces.
enum class Shape { Zeros, Min, Max, Alternating, SingleScale };

void structured_iq3xxs_block(uint8_t* b, Shape s) {
    std::memset(b, 0, kIq3xxsBlockBytes);
    uint16_t d = 0x3C00;  // 1.0
    if (s == Shape::Zeros) d = 0x0000;
    std::memcpy(b, &d, 2);
    uint8_t q = 0;
    if (s == Shape::Max) q = 0xFF;
    if (s == Shape::Alternating) q = 0xAA;
    if (s == Shape::SingleScale) q = 0x00;
    std::memset(b + 2, q, 64);
    for (int ib = 0; ib < 8; ++ib) {
        uint32_t aux = 0;
        if (s == Shape::Max) aux = 0x0FFFFFFFu | (15u << 28);
        if (s == Shape::Alternating) aux = 0x0555AAA5u | (7u << 28);
        if (s == Shape::SingleScale && ib == 3) aux = 0x00000000u | (9u << 28);
        std::memcpy(b + 66 + 4 * ib, &aux, 4);
    }
}

void structured_iq2s_block(uint8_t* b, Shape s) {
    std::memset(b, 0, kIq2sBlockBytes);
    uint16_t d = 0x3C00;
    if (s == Shape::Zeros) d = 0x0000;
    std::memcpy(b, &d, 2);
    uint8_t q = 0;
    if (s == Shape::Max) q = 0xFF;
    if (s == Shape::Alternating) q = 0x33;
    std::memset(b + 2, q, 72);  // qs, signs and qh
    if (s == Shape::Max) std::memset(b + 74, 0xFF, 8);
    if (s == Shape::Alternating) {
        for (int i = 0; i < 8; ++i) b[74 + i] = static_cast<uint8_t>(i % 2 ? 0xF0 : 0x0F);
    }
    if (s == Shape::SingleScale) b[74 + 4] = 0x39;
}

// cos similarity with an explicit zero-norm rule (degenerate rows are legal: all-zero weights).
bool cos_ok(const std::vector<float>& a, const std::vector<float>& b, double min_cos) {
    double na = 0, nb = 0, dot = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        na += double(a[i]) * a[i];
        nb += double(b[i]) * b[i];
        dot += double(a[i]) * b[i];
    }
    if (na == 0.0 || nb == 0.0) return na == nb;   // both zero: trivially identical
    return dot / std::sqrt(na * nb) >= min_cos;
}

} // namespace

int main() {
    // ---------------------------------------------------------------- layout + swizzle index math
    {
        const uint32_t cols = 3072;
        const Dot8UniversalLayout odd3 = dot8_universal_layout(32, cols, GgmlType::IQ3_XXS);
        VJ_CHECK(odd3.ok() && odd3.mode == Dot8RowMode::Odd256);
        VJ_CHECK(odd3.w_raw_row_bytes == 1536 && odd3.s_raw_row_bytes == 96);
        VJ_CHECK(odd3.w_row_stride_bytes == 1792 && odd3.w_tiles == 7);  // odd number of 256 B tiles
        VJ_CHECK(odd3.w_tiles % 2u == 1u);
        for (uint64_t r = 0; r < 32; ++r)
            for (uint32_t k = 0; k < odd3.w_tiles; ++k) VJ_CHECK(dot8_swizzle_tile(k, r, odd3) == k);

        const Dot8UniversalLayout odd2 = dot8_universal_layout(32, cols, GgmlType::IQ2_S);
        VJ_CHECK(odd2.s_raw_row_bytes == 96 + 48 && odd2.w_row_stride_bytes == 1792 && odd2.scale_embedded);

        // Xor16: whole 16-tile groups, phys_k_blk = k_blk ^ (row & 15), a permutation of every row.
        const Dot8UniversalLayout xr = dot8_universal_layout(64, cols, GgmlType::IQ3_XXS, Dot8RowMode::Xor16);
        VJ_CHECK(xr.mode == Dot8RowMode::Xor16 && xr.w_tiles == 16 && xr.w_row_stride_bytes == 4096);
        VJ_CHECK(!xr.scale_embedded && xr.s_row_stride_bytes == 96);
        for (uint64_t r = 0; r < 64; ++r) {
            uint32_t seen = 0;
            for (uint32_t k = 0; k < xr.w_tiles; ++k) {
                const uint32_t phys = dot8_swizzle_tile(k, r, xr);
                VJ_CHECK(phys == (k ^ static_cast<uint32_t>(r & 15u)));
                VJ_CHECK(phys < xr.w_tiles);
                VJ_CHECK(dot8_swizzle_tile(phys, r, xr) == k);  // applying it twice is the identity
                VJ_CHECK((seen & (1u << phys)) == 0u);          // no aliasing inside the row
                seen |= 1u << phys;
            }
            VJ_CHECK(seen == 0xFFFFu);  // every tile of the row is written exactly once
        }
        // Code block -> byte offset: the sub-tile position is preserved, only the tile moves.
        for (uint32_t b = 0; b < cols / 32u; ++b)
            VJ_CHECK(dot8_block_offset(b, 7, xr) == dot8_swizzle_tile(b / 16u, 7, xr) * 256u + (b % 16u) * 16u);

        // Rows that already are a whole 16-tile group cost no extra VRAM.
        const Dot8UniversalLayout exact = dot8_universal_layout(8, 8192, GgmlType::IQ2_S, Dot8RowMode::Xor16);
        VJ_CHECK(exact.w_tiles == 16 && exact.w_row_stride_bytes == exact.w_raw_row_bytes);

        // Odd-256B rows already spread 16 consecutive rows over 16 channels, no swizzle needed.
        uint32_t spread = 0;
        for (uint32_t r = 0; r < 16; ++r) spread |= 1u << ((r * odd3.w_row_stride_bytes / 256u) & 15u);
        VJ_CHECK(spread == 0xFFFFu);

        // Plane swizzle / unswizzle is an exact inverse and really moves the data.
        std::vector<uint8_t> plane(static_cast<size_t>(xr.w_plane_bytes()));
        for (auto& v : plane) v = static_cast<uint8_t>(rnd(256));
        const auto swz = dot8_swizzle_plane(plane.data(), xr);
        const auto back = dot8_unswizzle_plane(swz.data(), xr);
        VJ_CHECK(back.size() == plane.size() && std::memcmp(back.data(), plane.data(), plane.size()) == 0);
        VJ_CHECK(std::memcmp(swz.data(), plane.data(), plane.size()) != 0);
        // Row-local: each swizzled row is a permutation of the same bytes.
        for (uint64_t r = 0; r < 8; ++r) {
            std::vector<uint8_t> a(plane.begin() + r * xr.w_row_stride_bytes,
                                   plane.begin() + (r + 1) * xr.w_row_stride_bytes);
            std::vector<uint8_t> b(swz.begin() + r * xr.w_row_stride_bytes, swz.begin() + (r + 1) * xr.w_row_stride_bytes);
            std::sort(a.begin(), a.end());
            std::sort(b.begin(), b.end());
            VJ_CHECK(a == b);
        }
        // The odd layout is not swizzled: plane copy is byte identical.
        const auto same = dot8_swizzle_plane(plane.data(), odd3.ok() ? odd3 : xr);
        if (odd3.ok()) VJ_CHECK(std::memcmp(same.data(), plane.data(),
                                            std::min<size_t>(same.size(), plane.size())) == 0);
    }

    // ---------------------------------------------------------------- universal repack: both sources
    const auto& reg = quant::DecoderRegistry::instance();
    const uint32_t rows = 40, cols = 3072, nsb = cols / 256;

    for (GgmlType src : {GgmlType::IQ3_XXS, GgmlType::IQ2_S}) {
        for (Dot8RowMode mode : {Dot8RowMode::Odd256, Dot8RowMode::Xor16}) {
            for (int structured = 0; structured < 5; ++structured) {
                const Shape shape = static_cast<Shape>(structured);
                const uint32_t sb_bytes = src == GgmlType::IQ3_XXS ? kIq3xxsBlockBytes : kIq2sBlockBytes;
                const uint64_t src_stride = static_cast<uint64_t>(nsb) * sb_bytes;
                std::vector<uint8_t> disk(rows * src_stride);
                for (uint32_t r = 0; r < rows; ++r)
                    for (uint32_t s = 0; s < nsb; ++s) {
                        uint8_t* blk = disk.data() + r * src_stride + s * sb_bytes;
                        if (structured == 0) {
                            if (src == GgmlType::IQ3_XXS) random_iq3xxs_block(blk);
                            else random_iq2s_block(blk);
                        } else if (src == GgmlType::IQ3_XXS) {
                            structured_iq3xxs_block(blk, shape);
                        } else {
                            structured_iq2s_block(blk, shape);
                        }
                    }

                const Dot8UniversalLayout lay = dot8_universal_layout(rows, cols, src, mode);
                VJ_CHECK(lay.ok());
                std::vector<uint8_t> wp(static_cast<size_t>(lay.w_plane_bytes()), 0xCD);
                std::vector<uint8_t> sp(static_cast<size_t>(lay.s_plane_bytes()), 0xCD);
                uint8_t* sp_or_null = sp.empty() ? nullptr : sp.data();
                std::string err;
                // two disjoint row ranges, as a host thread pool would split them
                VJ_CHECK(dot8_repack_universal_rows(disk.data(), src_stride, 0, lay, 0, 17, wp.data(), sp_or_null, &err));
                VJ_CHECK(dot8_repack_universal_rows(disk.data(), src_stride, 0, lay, 17, rows, wp.data(), sp_or_null, &err));

                std::vector<float> got(cols), ref(cols);
                for (uint32_t r = 0; r < rows; ++r) {
                    const uint8_t* srow = disk.data() + r * src_stride;
                    VJ_CHECK(reg.decode_row_f32(src, srow, ref.data(), cols));
                    VJ_CHECK(dot8_dequant_row_f32(wp.data(), sp_or_null, lay, r, src, got.data()));
                    VJ_CHECK(cos_ok(got, ref, 0.999));  // the requirement: cos_sim >= 0.999 vs FP32 dequant

                    double nref = 0, ndiff = 0;
                    for (uint32_t i = 0; i < cols; ++i) {
                        nref += double(ref[i]) * ref[i];
                        ndiff += (double(got[i]) - ref[i]) * (double(got[i]) - ref[i]);
                    }
                    VJ_CHECK(nref == 0.0 ? ndiff == 0.0 : ndiff <= 1e-12 * nref);  // lossless, up to float order

                    // CPU reference row dot vs the same dot on the FP32 reference dequant
                    std::vector<float> x(cols);
                    for (uint32_t i = 0; i < cols; ++i) x[i] = (static_cast<int>(rnd(2001)) - 1000) * 1e-3f;
                    const double d_new = dot8_dot_row_f32(wp.data(), sp_or_null, lay, r, src, x.data());
                    double d_ref = 0.0;
                    for (uint32_t i = 0; i < cols; ++i) d_ref += static_cast<double>(ref[i]) * x[i];
                    VJ_CHECK(std::fabs(d_new - d_ref) <= 1e-9 * std::max(1.0, std::fabs(d_ref)));

                    // Bytes past the code payload (and past the embedded scales) stay zero.
                    if (lay.mode == Dot8RowMode::Odd256) {
                        const uint32_t pad_from = lay.w_raw_row_bytes + (lay.scale_embedded ? lay.s_raw_row_bytes : 0u);
                        for (uint32_t i = pad_from; i < lay.w_row_stride_bytes; ++i)
                            VJ_CHECK(wp[static_cast<size_t>(r) * lay.w_row_stride_bytes + i] == 0);
                    }
                }
                if (lay.mode == Dot8RowMode::Xor16) {
                    // The tiles move, so the padding invariant is checked through the exact inverse.
                    const auto plain = dot8_unswizzle_plane(wp.data(), lay);
                    for (uint32_t r = 0; r < rows; ++r)
                        for (uint32_t i = lay.w_raw_row_bytes; i < lay.w_row_stride_bytes; ++i)
                            VJ_CHECK(plain[static_cast<size_t>(r) * lay.w_row_stride_bytes + i] == 0);
                }
            }
        }
    }

    // ---------------------------------------------------------------- Odd256 must match the existing path
    {
        const uint32_t rows_iq3 = 24, cols_iq3 = 1024, nsb_iq3 = cols_iq3 / 256;
        const uint64_t stride = static_cast<uint64_t>(nsb_iq3) * kIq3xxsBlockBytes;
        std::vector<uint8_t> disk(rows_iq3 * stride);
        for (uint32_t r = 0; r < rows_iq3; ++r)
            for (uint32_t s = 0; s < nsb_iq3; ++s) random_iq3xxs_block(disk.data() + r * stride + s * kIq3xxsBlockBytes);
        const Dot8RowLayout legacy = dot8_iq3_layout(rows_iq3, cols_iq3);
        const Dot8UniversalLayout uni = dot8_universal_layout(rows_iq3, cols_iq3, GgmlType::IQ3_XXS);
        VJ_CHECK(legacy.w_row_stride_bytes == uni.w_row_stride_bytes && legacy.s_row_stride_bytes == uni.s_row_stride_bytes);
        std::vector<uint8_t> w1(static_cast<size_t>(legacy.w_plane_bytes()), 0), s1(static_cast<size_t>(legacy.s_plane_bytes()), 0);
        std::vector<uint8_t> w2(static_cast<size_t>(uni.w_plane_bytes()), 0), s2(static_cast<size_t>(uni.s_plane_bytes()), 0);
        std::string err;
        VJ_CHECK(dot8_repack_iq3xxs_rows(disk.data(), stride, 0, legacy, 0, rows_iq3, w1.data(), s1.data(), &err));
        VJ_CHECK(dot8_repack_universal_rows(disk.data(), stride, 0, uni, 0, rows_iq3, w2.data(), s2.data(), &err));
        VJ_CHECK(w1 == w2 && s1 == s2);  // same bytes: the universal path is a superset, not a variant
    }

    // ---------------------------------------------------------------- error paths fail closed
    {
        std::string err;
        const Dot8UniversalLayout bad = dot8_universal_layout(4, 384, GgmlType::IQ3_XXS);  // not a 256 multiple
        VJ_CHECK(!bad.ok());
        std::vector<uint8_t> buf(1 << 16, 0);
        VJ_CHECK(!dot8_repack_universal_rows(buf.data(), 384, 0, bad, 0, 4, buf.data(), buf.data(), &err) && !err.empty());

        const Dot8UniversalLayout lay = dot8_universal_layout(4, 512, GgmlType::IQ3_XXS);
        err.clear();
        VJ_CHECK(!dot8_repack_universal_rows(buf.data(), kIq3xxsBlockBytes, 0, lay, 0, 4, buf.data(), nullptr, &err) &&
                 !err.empty());  // separate scale plane required here
        err.clear();
        VJ_CHECK(!dot8_repack_universal_rows(buf.data(), kIq3xxsBlockBytes, 1, lay, 0, 4, buf.data(), buf.data(), &err) &&
                 !err.empty());  // sb_begin 1 + 2 superblocks does not fit a one-superblock row
        err.clear();
        VJ_CHECK(!dot8_repack_universal_rows(buf.data(), kIq3xxsBlockBytes, 0, lay, 2, 1, buf.data(), buf.data(), &err) &&
                 !err.empty());  // inverted row range
    }

    std::printf("test_v13_dot8_universal: OK\n");
    return 0;
}
