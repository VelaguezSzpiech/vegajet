// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Exhaustive CPU tests for the universal repacker.
//   * Every supported GGUF type: deterministic synthetic blocks -> decode -> golden FNV-1a hash of the FP32
//     output (hashes were cross-checked bit-for-bit against the ggml reference dequantizers).
//   * decode -> VJ_Q8_DP4A / VJ_Q4_DOT8 (+FWHT-64) -> dequantize: cosine + RMSE bounds.
//   * VJ_IQ_PACKED split planes are lossless.
//   * TCC L2 16/16 channel spread across K in {1536..8192} for both native formats.
//   * Roofline knapsack: positive-ROI sub-4-bit upgrades accepted, IQ4_NL -> Q8 rejected.
// Set VEGAJET_DUMP_BLOCKS_DIR to write <type>.blocks / <type>.f32 for external reference checks.
#include "core/quant_repacker.hpp"
#include "tests/test_common.hpp"

#include <cmath>
#include <string>

using namespace vegajet;
using namespace vegajet::quant;

namespace {

struct Rng {
    uint64_t s;
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    float uniform(float lo, float hi) { return lo + (hi - lo) * static_cast<float>((next() >> 40) * (1.0 / 16777216.0)); }
};

void put_f16(std::vector<uint8_t>& b, size_t off, float v) {
    const uint16_t h = f32_to_fp16(v);
    std::memcpy(&b[off], &h, 2);
}

// Offsets of FP16 scale fields inside one block: {offset, is_min}.
std::vector<std::pair<int, bool>> scale_fields(DiskQuantType t) {
    using D = DiskQuantType;
    switch (t) {
        case D::Q4_1: case D::Q5_1: case D::Q4_K: case D::Q5_K: return {{0, false}, {2, true}};
        case D::Q8_1: return {{0, false}, {2, true}};
        case D::Q2_K: return {{80, false}, {82, true}};
        case D::Q3_K: return {{108, false}};
        case D::Q6_K: return {{208, false}};
        default: return {{0, false}};
    }
}

std::vector<uint8_t> synth(DiskQuantType t, uint64_t n_elems, uint64_t seed) {
    const auto* fi = DecoderRegistry::instance().find(t);
    const uint64_t n_blocks = n_elems / fi->block_elems;
    std::vector<uint8_t> b(n_blocks * fi->block_bytes);
    Rng r{seed};
    for (auto& x : b) x = static_cast<uint8_t>(r.next());
    switch (t) {
        case DiskQuantType::F32:
            for (uint64_t i = 0; i < n_elems; ++i) { float v = r.uniform(-1, 1); std::memcpy(&b[i * 4], &v, 4); }
            return b;
        case DiskQuantType::F16:
            for (uint64_t i = 0; i < n_elems; ++i) put_f16(b, i * 2, r.uniform(-1, 1));
            return b;
        case DiskQuantType::BF16:
            for (uint64_t i = 0; i < n_elems; ++i) {
                float v = r.uniform(-1, 1); uint32_t u; std::memcpy(&u, &v, 4);
                const uint16_t h = static_cast<uint16_t>(u >> 16); std::memcpy(&b[i * 2], &h, 2);
            }
            return b;
        case DiskQuantType::F8_E4M3:
            for (auto& x : b) if ((x & 0x7F) == 0x7F) x &= 0xFE;  // avoid NaN encodings
            return b;
        default: break;
    }
    for (uint64_t k = 0; k < n_blocks; ++k)
        for (auto [off, is_min] : scale_fields(t))
            put_f16(b, k * fi->block_bytes + off, is_min ? r.uniform(0.0f, 0.004f) : r.uniform(0.0005f, 0.002f));
    return b;
}

uint64_t fnv1a(const void* p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; ++i) { h ^= static_cast<const uint8_t*>(p)[i]; h *= 0x100000001b3ull; }
    return h;
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < a.size(); ++i) { ab += double(a[i]) * b[i]; aa += double(a[i]) * a[i]; bb += double(b[i]) * b[i]; }
    return (aa == 0 || bb == 0) ? (aa == bb ? 1.0 : 0.0) : ab / std::sqrt(aa * bb);
}

// RMSE normalized by the RMS of the reference.
double rel_rmse(const std::vector<float>& ref, const std::vector<float>& x) {
    double e = 0, s = 0;
    for (size_t i = 0; i < ref.size(); ++i) { e += (double(ref[i]) - x[i]) * (double(ref[i]) - x[i]); s += double(ref[i]) * ref[i]; }
    return s == 0 ? 0 : std::sqrt(e / s);
}

struct Golden { DiskQuantType t; uint64_t hash; };
// FNV-1a of decoded FP32 (8 x 1024, seed = 1000 + type id). Verified equal to ggml dequantize_row_*.
const Golden kGolden[] = {
#include "tests/golden_decode_hashes.inc"
};

} // namespace

int main() {
    constexpr uint32_t kRows = 8, kCols = 1024;
    const char* dump_dir = std::getenv("VEGAJET_DUMP_BLOCKS_DIR");
    const auto& reg = DecoderRegistry::instance();

    const DiskQuantType required[] = {
        DiskQuantType::F32, DiskQuantType::F16, DiskQuantType::BF16, DiskQuantType::F8_E4M3,
        DiskQuantType::Q8_0, DiskQuantType::Q8_1, DiskQuantType::Q6_K, DiskQuantType::Q5_0, DiskQuantType::Q5_1,
        DiskQuantType::Q5_K, DiskQuantType::Q4_0, DiskQuantType::Q4_1, DiskQuantType::Q4_K, DiskQuantType::IQ4_NL,
        DiskQuantType::IQ4_XS, DiskQuantType::Q3_K, DiskQuantType::Q2_K, DiskQuantType::Q2_0,
        DiskQuantType::IQ3_XXS, DiskQuantType::IQ3_S, DiskQuantType::IQ2_XXS, DiskQuantType::IQ2_XS,
        DiskQuantType::IQ2_S,
    };
    VJ_CHECK(sizeof(required) / sizeof(required[0]) == 23);

    std::printf("%-8s %6s %6s | %-18s | %-10s %-9s | %-10s %-9s | %-10s | %s\n", "type", "B/blk", "elems",
                "decode hash", "Q8 cos", "Q8 rRMSE", "Q4 cos", "Q4 rRMSE", "Q4+FWHT", "IQ lossless");
    int golden_hits = 0;
    for (DiskQuantType t : required) {
        const FormatInfo* fi = reg.find(t);
        VJ_CHECK(fi != nullptr);
        const GgmlTypeTraits tr = ggml_type_traits(static_cast<uint32_t>(t));
        VJ_CHECK(fi->block_bytes == tr.block_bytes && fi->block_elems == tr.block_elems);

        const uint64_t n = uint64_t(kRows) * kCols;
        const auto blocks = synth(t, n, 1000 + static_cast<uint32_t>(t));
        VJ_CHECK(blocks.size() == reg.row_bytes(t, kCols) * kRows);
        std::vector<float> ref(n);
        for (uint32_t r = 0; r < kRows; ++r)
            VJ_CHECK(reg.decode_row_f32(t, blocks.data() + r * reg.row_bytes(t, kCols), ref.data() + size_t(r) * kCols, kCols));
        for (float v : ref) VJ_CHECK(std::isfinite(v));
        const uint64_t h = fnv1a(ref.data(), ref.size() * 4);

        if (dump_dir) {
            const std::string base = std::string(dump_dir) + "/" + tr.name;
            FILE* f = std::fopen((base + ".blocks").c_str(), "wb"); std::fwrite(blocks.data(), 1, blocks.size(), f); std::fclose(f);
            f = std::fopen((base + ".f32").c_str(), "wb"); std::fwrite(ref.data(), 4, ref.size(), f); std::fclose(f);
        }
        bool have_golden = false;
        for (const auto& g : kGolden)
            if (g.t == t) { have_golden = true; VJ_CHECK(g.hash == h); ++golden_hits; }

        // Native round trips.
        auto q8 = repack_to_vj_q8(t, blocks.data(), kRows, kCols);
        std::vector<float> d8(n), d4(n), d4f(n), diq(n);
        dequantize_vj_q8(q8, d8.data());
        auto q4 = repack_to_vj_q4_dot8(t, blocks.data(), kRows, kCols, false);
        dequantize_vj_q4_dot8(q4, d4.data());
        auto q4f = repack_to_vj_q4_dot8(t, blocks.data(), kRows, kCols, true);
        dequantize_vj_q4_dot8(q4f, d4f.data());
        auto iq = repack_to_vj_iq(t, blocks.data(), kRows, kCols);
        VJ_CHECK(unpack_vj_iq_to_disk(iq) == blocks);
        dequantize_vj_iq(iq, diq.data());
        VJ_CHECK(std::memcmp(diq.data(), ref.data(), n * 4) == 0);
        VJ_CHECK(iq.total_bytes() == blocks.size());

        const double c8 = cosine(ref, d8), c4 = cosine(ref, d4), c4f = cosine(ref, d4f);
        const double e8 = rel_rmse(ref, d8), e4 = rel_rmse(ref, d4);
        std::printf("%-8s %6u %6u | %016llx%s | %.7f %.2e | %.6f %.2e | %.6f   | yes\n", tr.name, fi->block_bytes,
                    fi->block_elems, static_cast<unsigned long long>(h), have_golden ? "  " : " ?", c8, e8, c4, e4, c4f);
        VJ_CHECK(c8 >= 0.9999 && e8 <= 0.015);
        VJ_CHECK(c4 >= 0.995 && e4 <= 0.10);
        // FWHT-64 is meant for zero-mean (symmetric) weights; affine-min formats are reported only.
        const bool has_min = t == DiskQuantType::Q4_1 || t == DiskQuantType::Q5_1 || t == DiskQuantType::Q4_K ||
                             t == DiskQuantType::Q5_K || t == DiskQuantType::Q2_K;
        if (!has_min) VJ_CHECK(c4f >= 0.99);  // synthetic uniform data: rotation cannot help, only bound loss
        if (t == DiskQuantType::Q8_0 || t == DiskQuantType::Q2_0) VJ_CHECK(std::memcmp(d8.data(), ref.data(), n * 4) == 0);
        if (t == DiskQuantType::Q2_0) VJ_CHECK(std::memcmp(d4.data(), ref.data(), n * 4) == 0);
    }
    if (!dump_dir) VJ_CHECK(golden_hits == 23);
    std::printf("[PASS] 23/23 GGUF decoders: golden decode hashes, Q8 cos>=0.9999, Q4 cos>=0.995, IQ_PACKED lossless\n");

    // TCC L2 anti-camping.
    for (uint32_t K : {1536u, 2048u, 2560u, 4096u, 6144u, 8192u}) {
        for (int fmt = 0; fmt < 2; ++fmt) {
            const uint32_t row_bytes = fmt == 0 ? K : K / 2;  // Q8: 1 B/elem, Q4: 0.5 B/elem
            const TccRowLayout lay = select_tcc_layout(row_bytes);
            VJ_CHECK(lay.stride_bytes >= row_bytes && lay.stride_bytes % kTccTileBytes == 0);
            if (lay.mode == TccLayoutMode::XorTileSwizzle) VJ_CHECK(lay.stride_bytes == row_bytes && lay.tiles_per_row >= 16);
            else VJ_CHECK(lay.tiles_per_row % 2 == 1);
            for (uint32_t lt = 0; lt < std::min<uint32_t>(lay.tiles_per_row, 4); ++lt)
                VJ_CHECK(tcc_channel_spread_count(16, lay, lt) == 16);
            std::vector<uint8_t> raw(size_t(32) * row_bytes);
            for (size_t i = 0; i < raw.size(); ++i) raw[i] = static_cast<uint8_t>(i * 131u + 17u);
            const auto pk = pack_tcc_deswizzled(raw.data(), 32, lay);
            VJ_CHECK(unpack_tcc_deswizzled(pk.data(), 32, lay) == raw);
            std::printf("  TCC K=%-5u %s row=%5u B -> %-14s stride=%5u B tiles=%2u spread=16/16\n", K,
                        fmt == 0 ? "VJ_Q8_DP4A" : "VJ_Q4_DOT8", row_bytes,
                        lay.mode == TccLayoutMode::XorTileSwizzle ? "XorTileSwizzle"
                        : lay.mode == TccLayoutMode::PadOdd256 ? "PadOdd256" : "NaturalOdd256",
                        lay.stride_bytes, lay.tiles_per_row);
        }
    }
    VJ_CHECK(select_tcc_layout(4096).mode == TccLayoutMode::XorTileSwizzle);
    VJ_CHECK(select_tcc_layout(2048).mode == TccLayoutMode::PadOdd256 && select_tcc_layout(2048).stride_bytes == 2304);
    VJ_CHECK(select_tcc_layout(1280).mode == TccLayoutMode::NaturalOdd256);
    std::printf("[PASS] TCC L2 16/16 channel spread for K in {1536,2048,2560,4096,6144,8192} (Q8 + Q4), bit-exact pack/unpack\n");

    // Roofline profile + JSON round trip.
    HardwareRooflineProfile hw;
    hw.gpus = {GpuRooflineInfo{"gpu0", 60, 1.50, 1080.0, 732.0, 200.0, 0.55},
               GpuRooflineInfo{"gpu1", 60, 1.50, 1140.0, 761.0, 200.0, 0.55}};
    VJ_CHECK(std::fabs(hw.gpus[0].i_star() - 4.33) < 0.05 && std::fabs(hw.gpus[1].i_star() - 4.16) < 0.05);
    VJ_CHECK(std::fabs(hw.min_i_star(2.0) - 2.08) < 0.05);
    const std::string js = "/tmp/vegajet_test_hw_profile_" + std::to_string(getpid()) + ".json";
    VJ_CHECK(HardwareRooflineProfile::save_json(js, hw));
    HardwareRooflineProfile hw2;
    VJ_CHECK(HardwareRooflineProfile::load_json(js, hw2) && hw2.gpus.size() == 2 && std::fabs(hw2.gpus[1].bw_hbm_gbs - 761) < 0.1);
    std::remove(js.c_str());

    // Knapsack at decode (reuse 1.0, I* = 4.16).
    const uint64_t GiB = 1ull << 30;
    std::vector<TensorPlanSpec> specs = {
        {"vocab_iq4nl", DiskQuantType::IQ4_NL, TensorRole::VocabHead, 124160, 2560, 1, 1.0, false},
        {"vocab_q6k", DiskQuantType::Q6_K, TensorRole::VocabHead, 124160, 2560, 1, 1.0, false},
        {"moe_down_q2_0", DiskQuantType::Q2_0, TensorRole::RoutedMoE, 2560, 320, 512, 10.0 / 512.0, false},
        {"moe_up_iq2_xxs", DiskQuantType::IQ2_XXS, TensorRole::RoutedMoE, 1024, 2048, 256, 8.0 / 256.0, false},
    };
    auto dec = VramBudgetPlanner::plan_roofline_knapsack(specs, 30 * GiB, 2 * GiB, hw, 1.0);
    VJ_CHECK(dec[0].target_format == NativeFormat::VJ_IQ_PACKED && dec[0].rejected_negative_roi);
    VJ_CHECK(dec[1].target_format == NativeFormat::VJ_Q8_DP4A && dec[1].expected_ms_saved > 0);
    VJ_CHECK(dec[2].target_format == NativeFormat::VJ_Q4_DOT8 && dec[2].expected_ms_saved > 0);
    VJ_CHECK(dec[3].target_format == NativeFormat::VJ_IQ_PACKED);  // 7.1 inst/B vs 4.16: not worth 2.2x bytes
    // With 2x reuse (multi-token verify) I* halves: IQ2_XXS becomes VALU-bound enough to upgrade.
    auto dec2 = VramBudgetPlanner::plan_roofline_knapsack(specs, 30 * GiB, 2 * GiB, hw, 2.0);
    VJ_CHECK(dec2[0].target_format != NativeFormat::VJ_Q8_DP4A);  // Q8 still negative ROI; same-size Q4 may win
    VJ_CHECK(dec2[3].target_format == NativeFormat::VJ_Q4_DOT8 && dec2[3].roi_ms_per_gib > 0);
    // A tight budget keeps the baseline (no upgrade fits).
    auto tight = VramBudgetPlanner::plan_roofline_knapsack(specs, 2 * GiB + 300 * 1024 * 1024, 2 * GiB, hw, 1.0);
    VJ_CHECK(tight[1].target_format == NativeFormat::VJ_IQ_PACKED && tight[2].target_format == NativeFormat::VJ_IQ_PACKED);
    std::printf("[PASS] Knapsack: IQ4_NL->Q8 rejected (neg ROI); Q6_K->Q8 +%.3f ms, Q2_0->Q4 +%.4f ms accepted; "
                "IQ2_XXS->Q4 accepted at 2x reuse (%.2f ms/GiB); tight budget keeps baseline\n",
                dec[1].expected_ms_saved, dec[2].expected_ms_saved, dec2[3].roi_ms_per_gib);

    // First-class VJ_Q4_A8 round-trip + 16/16 TCC channel spread across ALL K in {1536,2048,2560,3072,4096,6144,8192}
    // for BOTH qs and dm planes!
    const uint32_t k_vals[] = {1536, 2048, 2560, 3072, 4096, 6144, 8192};
    for (uint32_t K : k_vals) {
        const auto qs_lay = select_tcc_odd256_layout(K / 2u);
        const auto dm_lay = select_tcc_odd256_layout((K / 32u) * 4u);
        const uint32_t sp_qs = tcc_channel_spread_count(16, qs_lay, 0);
        const uint32_t sp_dm = tcc_channel_spread_count(16, dm_lay, 0);
        VJ_CHECK(sp_qs == 16u && sp_dm == 16u);
        const auto qs_auto = select_tcc_layout(K / 2u);
        const auto dm_auto = select_tcc_layout((K / 32u) * 4u);
        VJ_CHECK(tcc_channel_spread_count(16, qs_auto, 0) == 16u);
        VJ_CHECK(tcc_channel_spread_count(16, dm_auto, 0) == 16u);
    }
    {
        std::vector<float> src(16 * 4096);
        for (size_t i = 0; i < src.size(); ++i) src[i] = std::sin(static_cast<float>(i) * 0.13f) * 2.5f - 0.3f;
        const auto p = pack_f32_to_vj_q4_a8(src.data(), 16, 4096, true);
        VJ_CHECK(p.qs_row_stride_bytes == 2304u && p.dm_row_stride_u32 == 192u); // 9*256 B & 3*256 B
        std::vector<float> dec_a8(16 * 4096);
        dequantize_vj_q4_a8(p, dec_a8.data());
        double dot = 0, na = 0, nb = 0;
        for (size_t i = 0; i < src.size(); ++i) { dot += src[i] * dec_a8[i]; na += src[i] * src[i]; nb += dec_a8[i] * dec_a8[i]; }
        const double cos_sim = dot / std::sqrt(na * nb);
        VJ_CHECK(cos_sim > 0.996);
        std::printf("[PASS] VJ_Q4_A8 native format + 16/16 TCC spread on qs & dm across 7 K sizes (K=4096 cos=%.5f)\n", cos_sim);
    }
    return 0;
}
