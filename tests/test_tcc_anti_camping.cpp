// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// TCC L2 anti-camping (16 channels x 256 B): every quantized plane of every native format must spread 16 rows that
// read the same logical tile over all 16 channels, for K in {1536 .. 8192}.
//  * qs and scale planes laid out with quant::select_tcc_layout (scale planes included: a 4096-column VJ_Q4_A8 dm
//    plane is 512 B = 2 tiles and camps on 8 channels at its natural stride);
//  * XOR tile swizzle (phys_tile = k_tile ^ (row & 15)) round-trips exactly and lands bytes where the formula says;
//  * PadOdd256 round-trips exactly and zero-fills padding;
//  * per-WG k_start stagger is a bijection over the data tiles and spreads sibling WGs over min(16, tiles) channels.
#include "core/quant_repacker.hpp"
#include "core/tcc_stagger.hpp"
#include "tests/test_common.hpp"

#include <cstdio>
#include <random>
#include <set>

using namespace vegajet::quant;

namespace {

uint32_t spread_all_tiles_min(const TccRowLayout& lay, uint32_t rows) {
    uint32_t worst = kTccChannels;
    for (uint32_t t = 0; t < tcc_data_tiles(lay); ++t) {
        const uint32_t s = tcc_channel_spread_count(rows, lay, t);
        // Cross-check the library count against the independent helper in tcc_stagger.hpp.
        std::set<uint32_t> ch;
        for (uint32_t r = 0; r < rows; ++r) ch.insert(tcc_channel(r, t, lay));
        VJ_CHECK(ch.size() == s);
        worst = std::min(worst, s);
    }
    return worst;
}

TccRowLayout natural(uint32_t raw) {  // what an uploader that ignores the layout produces
    TccRowLayout l;
    l.mode = TccLayoutMode::NaturalOdd256;
    l.raw_row_bytes = raw;
    l.stride_bytes = raw;
    l.tiles_per_row = (raw + kTccTileBytes - 1) / kTccTileBytes;
    return l;
}

void check_layout_invariants(const TccRowLayout& l) {
    VJ_CHECK(l.stride_bytes % kTccTileBytes == 0);
    VJ_CHECK(l.stride_bytes >= l.raw_row_bytes);
    VJ_CHECK(l.tiles_per_row * kTccTileBytes == l.stride_bytes);
    if (l.mode == TccLayoutMode::XorTileSwizzle) {
        VJ_CHECK(l.stride_bytes == l.raw_row_bytes);  // zero extra VRAM
        VJ_CHECK(l.tiles_per_row % kTccChannels == 0);
    } else {
        VJ_CHECK(l.tiles_per_row & 1u);                // odd tile stride -> gcd(16, tiles) == 1
        VJ_CHECK(l.stride_bytes - l.raw_row_bytes < 2 * kTccTileBytes);  // at most one pad tile + rounding
    }
}

void round_trip(const TccRowLayout& lay, uint32_t rows, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<uint8_t> src(static_cast<size_t>(rows) * lay.raw_row_bytes);
    for (auto& b : src) b = static_cast<uint8_t>(rng());
    const auto packed = pack_tcc_deswizzled(src.data(), rows, lay);
    VJ_CHECK(packed.size() == static_cast<size_t>(rows) * lay.stride_bytes);
    const auto back = unpack_tcc_deswizzled(packed.data(), rows, lay);
    VJ_CHECK(back == src);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t kt = 0; kt < tcc_data_tiles(lay); ++kt) {
            const uint32_t pt = tcc_physical_tile(r, kt, lay);
            for (uint32_t b : {0u, 1u, 127u, 255u}) {
                const size_t s_off = static_cast<size_t>(r) * lay.raw_row_bytes + kt * kTccTileBytes + b;
                if (s_off >= static_cast<size_t>(r + 1) * lay.raw_row_bytes) continue;
                const size_t d_off = static_cast<size_t>(r) * lay.stride_bytes + pt * kTccTileBytes + b;
                VJ_CHECK(packed[d_off] == src[s_off]);
            }
        }
        if (lay.mode != TccLayoutMode::XorTileSwizzle)
            for (uint32_t b = lay.raw_row_bytes; b < lay.stride_bytes; ++b)
                VJ_CHECK(packed[static_cast<size_t>(r) * lay.stride_bytes + b] == 0);
    }
}

} // namespace

int main() {
    const uint32_t Ks[] = {1536, 2048, 2560, 3072, 4096, 6144, 8192};
    struct Plane { const char* fmt; const char* plane; uint32_t div_num, div_den; };
    // bytes per row = K * num / den
    const Plane planes[] = {
        {"VJ_Q4_A8", "qs", 1, 2}, {"VJ_Q4_A8", "dm", 1, 8},
        {"VJ_Q4_DOT8", "qs", 1, 2}, {"VJ_Q4_DOT8", "ds", 1, 16},
        {"VJ_Q8_DP4A", "qs", 1, 1}, {"VJ_Q8_DP4A", "ds", 1, 16},
    };

    uint32_t checked = 0, naive_camping = 0;
    for (uint32_t K : Ks) {
        for (const auto& p : planes) {
            const uint32_t raw = K * p.div_num / p.div_den;
            const TccRowLayout lay = select_tcc_layout(raw);
            check_layout_invariants(lay);
            for (uint32_t rows : {16u, 64u, 3072u}) VJ_CHECK(spread_all_tiles_min(lay, rows) == kTccChannels);
            const uint32_t naive = spread_all_tiles_min(natural(raw), 16);
            if (naive < kTccChannels) {
                ++naive_camping;
                std::printf("  natural-stride camping: %-10s %-2s K=%-5u raw=%4u B -> %2u/16 channels; layout %s stride %u B -> 16/16\n",
                            p.fmt, p.plane, K, raw, naive,
                            lay.mode == TccLayoutMode::XorTileSwizzle ? "XOR" :
                            lay.mode == TccLayoutMode::PadOdd256 ? "PadOdd256" : "NaturalOdd256",
                            lay.stride_bytes);
            }
            ++checked;
        }
    }

    // The specific bug: VJ_Q4_A8 dm plane at K=4096 is 512 B (2 tiles). Natural stride -> 8/16; laid out -> 16/16.
    {
        const auto l = select_q4a8_layouts(4096);
        VJ_CHECK(l.sc.raw_row_bytes == 512);
        VJ_CHECK(spread_all_tiles_min(natural(512), 16) == 8);
        VJ_CHECK(l.sc.mode == TccLayoutMode::PadOdd256 && l.sc.stride_bytes == 768);
        VJ_CHECK(spread_all_tiles_min(l.sc, 16) == 16);
        // qs plane at K=4096 is 2048 B = 8 tiles -> padded to 9 (the live uploader's 2304 B stride).
        VJ_CHECK(l.qs.mode == TccLayoutMode::PadOdd256 && l.qs.stride_bytes == 2304);
        // K=8192 qs plane: 16 tiles -> XOR swizzle with zero padding.
        VJ_CHECK(select_q4a8_layouts(8192).qs.mode == TccLayoutMode::XorTileSwizzle);
    }

    // XOR swizzle round trip (16- and 32-tile rows) and PadOdd256 round trip, odd row counts included.
    for (uint32_t raw : {4096u, 8192u}) {
        const auto lay = select_tcc_layout(raw);
        VJ_CHECK(lay.mode == TccLayoutMode::XorTileSwizzle);
        for (uint32_t rows : {1u, 15u, 16u, 17u, 33u, 64u}) round_trip(lay, rows, raw + rows);
        // phys = k ^ (row & 15) is a permutation of each row's tiles.
        for (uint32_t r = 0; r < 32; ++r) {
            std::set<uint32_t> s;
            for (uint32_t kt = 0; kt < lay.tiles_per_row; ++kt) s.insert(tcc_physical_tile(r, kt, lay));
            VJ_CHECK(s.size() == lay.tiles_per_row && *s.rbegin() == lay.tiles_per_row - 1);
        }
    }
    for (uint32_t raw : {192u, 512u, 1968u, 2048u, 3072u}) {
        const auto lay = select_tcc_layout(raw);
        VJ_CHECK(lay.mode != TccLayoutMode::XorTileSwizzle);
        for (uint32_t rows : {1u, 7u, 16u, 33u}) round_trip(lay, rows, raw * 7 + rows);
    }

    // k_start stagger: each WG reads every data tile exactly once; sibling WGs reading the same rows at the same
    // step start on min(16, tiles) distinct channels.
    uint32_t stagger_cases = 0;
    double min_stagger = 1.0, mean_stagger_min = 1.0;
    for (uint32_t K : Ks) {
        for (const auto& lay : {select_q4a8_layouts(K).qs, select_q4a8_layouts(K).sc, select_q8_layouts(K).qs}) {
            const uint32_t tiles = tcc_data_tiles(lay);
            for (uint32_t wg = 0; wg < 64; ++wg) {
                std::vector<uint8_t> seen(tiles, 0);
                for (uint32_t it = 0; it < tiles; ++it) seen[stagger_tile(wg, it, tiles)]++;
                for (uint8_t c : seen) VJ_CHECK(c == 1);
            }
            // Unstaggered siblings all hit one channel. Staggered: the first step is exact; later steps can lose a few
            // channels where the 16-WG window wraps past a tile count that is not a multiple of 16.
            const uint32_t ideal = std::min(kTccChannels, tiles);
            for (uint32_t row : {0u, 5u, 17u}) {
                uint32_t worst = kTccChannels, sum = 0;
                for (uint32_t it = 0; it < tiles; ++it) {
                    std::set<uint32_t> ch;
                    for (uint32_t wg = 0; wg < kTccChannels; ++wg) ch.insert(tcc_channel(row, stagger_tile(wg, it, tiles), lay));
                    if (it == 0) VJ_CHECK(ch.size() == ideal);
                    worst = std::min<uint32_t>(worst, static_cast<uint32_t>(ch.size()));
                    sum += static_cast<uint32_t>(ch.size());
                }
                VJ_CHECK(2 * worst >= ideal);                 // never worse than half the ideal spread
                VJ_CHECK(4 * sum >= 3 * ideal * tiles);       // >= 75% of ideal averaged over the K walk
                min_stagger = std::min(min_stagger, static_cast<double>(worst) / ideal);
                mean_stagger_min = std::min(mean_stagger_min, static_cast<double>(sum) / (ideal * tiles));
            }
            ++stagger_cases;
        }
    }

    std::printf("test_tcc_anti_camping: %u plane/K layouts at 16/16 channels, %u natural-stride camping cases caught, "
                "XOR+pad round trips exact, %u stagger layouts covered (worst step %.0f%%, worst mean %.0f%% of ideal spread)\n",
                checked, naive_camping, stagger_cases, 100.0 * min_stagger, 100.0 * mean_stagger_min);
    std::printf("PASS\n");
    return 0;
}
