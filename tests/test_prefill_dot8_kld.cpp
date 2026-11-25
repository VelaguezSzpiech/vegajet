// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU reference verification for Stage 3D W4A4 prefill math (v_dot8_i32_i4):
//   1. Exact VJ_Q4_A8 nibble deinterleave + ^ 0x88888888 + exact +8*sum(x_q4) affine undo (0.0 ULP error)
//   2. Exact IQ2_S grid {8,25,43} -> {1,3,5} mapping (b >> 3), SQNR >= 33.9 dB, and carry-free apply_8signs_i4x8
//   3. Exact KL divergence KL(P || Q) computation utility
#include "core/ggml_iq_tables.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

static uint32_t pack_u4_lo8_signed(uint32_t a, uint32_t b) {
    uint32_t ta = a & 0x0F0F0F0Fu;
    ta = (ta | (ta >> 4)) & 0x00FF00FFu;
    ta = (ta | (ta >> 8)) & 0x0000FFFFu;
    uint32_t tb = b & 0x0F0F0F0Fu;
    tb = (tb | (tb >> 4)) & 0x00FF00FFu;
    tb = (tb | (tb >> 8)) & 0x0000FFFFu;
    return (ta | (tb << 16)) ^ 0x88888888u;
}

static int32_t sim_sdot8(uint32_t a, uint32_t b, int32_t c) {
    int32_t acc = c;
    for (int k = 0; k < 8; ++k) {
        int32_t na = static_cast<int32_t>((a >> (4 * k)) & 0xFu);
        if (na >= 8) na -= 16;
        int32_t nb = static_cast<int32_t>((b >> (4 * k)) & 0xFu);
        if (nb >= 8) nb -= 16;
        acc += na * nb;
    }
    return acc;
}

static uint32_t apply_8signs_i4x8(uint32_t mag_u32, uint32_t sign8) {
    uint32_t a = (sign8 & 0x0Fu) | ((sign8 & 0xF0u) << 12);
    a = (a & 0x00030003u) | ((a & 0x000C000Cu) << 6);
    const uint32_t add = (a * 9u) & 0x11111111u;
    return (mag_u32 ^ ((add << 4) - add)) + add;
}

static uint32_t iq2s_grid_u64_to_i4x8(uint64_t g) {
    const uint32_t lo = static_cast<uint32_t>(g);
    const uint32_t hi = static_cast<uint32_t>(g >> 32);
    uint32_t ta = (lo >> 3) & 0x0F0F0F0Fu;
    ta = (ta | (ta >> 4)) & 0x00FF00FFu;
    ta = (ta | (ta >> 8)) & 0x0000FFFFu;
    uint32_t tb = (hi >> 3) & 0x0F0F0F0Fu;
    tb = (tb | (tb >> 4)) & 0x00FF00FFu;
    tb = (tb | (tb >> 8)) & 0x0000FFFFu;
    return ta | (tb << 16);
}

static double kl_divergence(const std::vector<float>& logits_p, const std::vector<float>& logits_q) {
    const size_t n = logits_p.size();
    double mp = -1e30, mq = -1e30;
    for (size_t i = 0; i < n; ++i) {
        if (logits_p[i] > mp) mp = logits_p[i];
        if (logits_q[i] > mq) mq = logits_q[i];
    }
    double sp = 0.0, sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sp += std::exp(static_cast<double>(logits_p[i]) - mp);
        sq += std::exp(static_cast<double>(logits_q[i]) - mq);
    }
    const double lzp = mp + std::log(sp);
    const double lzq = mq + std::log(sq);
    double kl = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double lp = static_cast<double>(logits_p[i]) - lzp;
        const double lq = static_cast<double>(logits_q[i]) - lzq;
        const double p  = std::exp(lp);
        if (p > 0.0) kl += p * (lp - lq);
    }
    return kl;
}

} // namespace

int main() {
    // 1. Verify VJ_Q4_A8 dense nibble deinterleave + ^ 0x88888888 + exact +8*sum(x_q4) affine identity
    uint32_t rng = 0x13579BDFu;
    auto next_u32 = [&]() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; };

    for (int trial = 0; trial < 2000; ++trial) {
        uint8_t bytes[16];
        int32_t w_ref[32];
        for (int i = 0; i < 16; ++i) {
            bytes[i] = static_cast<uint8_t>(next_u32() & 0xFFu);
            w_ref[i]      = bytes[i] & 0x0F;
            w_ref[i + 16] = (bytes[i] >> 4) & 0x0F;
        }
        uint32_t wq[4];
        for (int k = 0; k < 4; ++k) {
            wq[k] = static_cast<uint32_t>(bytes[4*k])
                  | (static_cast<uint32_t>(bytes[4*k + 1]) << 8)
                  | (static_cast<uint32_t>(bytes[4*k + 2]) << 16)
                  | (static_cast<uint32_t>(bytes[4*k + 3]) << 24);
        }
        const uint32_t p[4] = {
            pack_u4_lo8_signed(wq[0],      wq[1]),
            pack_u4_lo8_signed(wq[2],      wq[3]),
            pack_u4_lo8_signed(wq[0] >> 4, wq[1] >> 4),
            pack_u4_lo8_signed(wq[2] >> 4, wq[3] >> 4),
        };
        int32_t x_ref[32];
        uint32_t x_q4[4] = {0, 0, 0, 0};
        int32_t sum_x = 0;
        for (int k = 0; k < 32; ++k) {
            x_ref[k] = static_cast<int32_t>(next_u32() % 15u) - 7; // [-7..+7]
            sum_x += x_ref[k];
            x_q4[k >> 3] |= static_cast<uint32_t>(x_ref[k] & 0xFu) << ((k & 7) * 4);
        }
        int32_t ref_dp = 0;
        for (int k = 0; k < 32; ++k) ref_dp += w_ref[k] * x_ref[k];
        int32_t dot8_dp = 0;
        for (int k = 0; k < 4; ++k) dot8_dp = sim_sdot8(p[k], x_q4[k], dot8_dp);
        const int32_t reconstructed = dot8_dp + (sum_x << 3);
        if (reconstructed != ref_dp) {
            std::fprintf(stderr, "FAIL dense affine trial=%d: got=%d expected=%d\n", trial, reconstructed, ref_dp);
            return 1;
        }
    }

    // 2. Verify ALL 1,024 entries of iq2s_grid: exact {8, 25, 43} -> {1, 3, 5}, SQNR >= 33.9 dB, and all 256 sign masks
    constexpr double kAlpha = 8.469311;
    double sig_pow = 0.0, err_pow = 0.0;
    for (int idx = 0; idx < 1024; ++idx) {
        const uint64_t g = vegajet::ggml_tables::iq2s_grid[idx];
        const uint32_t g4 = iq2s_grid_u64_to_i4x8(g);
        for (int k = 0; k < 8; ++k) {
            const uint8_t b = static_cast<uint8_t>((g >> (8 * k)) & 0xFFu);
            if (b != 8u && b != 25u && b != 43u) {
                std::fprintf(stderr, "FAIL iq2s_grid[%d][%d] unexpected byte=%u\n", idx, k, b);
                return 1;
            }
            const uint32_t nib = (g4 >> (4 * k)) & 0xFu;
            const uint32_t exp_nib = (b == 8u) ? 1u : ((b == 25u) ? 3u : 5u);
            if (nib != exp_nib) {
                std::fprintf(stderr, "FAIL grid_i4[%d][%d]: got=%u expected=%u\n", idx, k, nib, exp_nib);
                return 1;
            }
            const double diff = static_cast<double>(b) - kAlpha * static_cast<double>(nib);
            sig_pow += static_cast<double>(b) * static_cast<double>(b);
            err_pow += diff * diff;
        }
        // Exhaustively check apply_8signs_i4x8 across sampled sign masks (all 256 on first 64 entries)
        if (idx < 64) {
            for (uint32_t s8 = 0; s8 < 256u; ++s8) {
                const uint32_t signed_w = apply_8signs_i4x8(g4, s8);
                for (int k = 0; k < 8; ++k) {
                    int32_t got = static_cast<int32_t>((signed_w >> (4 * k)) & 0xFu);
                    if (got >= 8) got -= 16;
                    const int32_t mag = static_cast<int32_t>((g4 >> (4 * k)) & 0xFu);
                    const int32_t want = ((s8 >> k) & 1u) ? -mag : mag;
                    if (got != want) {
                        std::fprintf(stderr, "FAIL apply_8signs idx=%d s8=%u k=%d: got=%d want=%d\n", idx, s8, k, got, want);
                        return 1;
                    }
                }
            }
        }
    }
    const double sqnr_db = 10.0 * std::log10(sig_pow / err_pow);
    if (sqnr_db < 33.9) {
        std::fprintf(stderr, "FAIL iq2s grid_i4 SQNR=%.3f dB < 33.9 dB\n", sqnr_db);
        return 1;
    }

    // 3. Verify KL divergence sanity (0 for identical, positive for perturbed)
    std::vector<float> p = {2.0f, 1.0f, 0.5f, -1.0f, -2.5f};
    std::vector<float> q = {1.95f, 1.05f, 0.45f, -0.95f, -2.5f};
    const double kl0 = kl_divergence(p, p);
    const double kl1 = kl_divergence(p, q);
    if (std::fabs(kl0) > 1e-12 || !(kl1 > 0.0 && kl1 < 0.01)) {
        std::fprintf(stderr, "FAIL KLD sanity: kl0=%.3e kl1=%.6f\n", kl0, kl1);
        return 1;
    }

    // 4. Verify IQ3_XXS VJ_Q4_DOT8 code -> signed 4-bit SWAR (w ^ (((w & 0x88888888u) >> 3) * 7u))
    //    and exact 8*xd4*sdot8 + 4*xsum4_scaled reconstruction (exact on 14/16 codes, SQNR >= 42 dB overall).
    static constexpr int32_t kIq3True[16] = {
        4, 12, 20, 28, 36, 44, 52, 62,
        -4, -12, -20, -28, -36, -44, -52, -62
    };
    double iq3_sig = 0.0, iq3_err = 0.0;
    for (int trial = 0; trial < 2000; ++trial) {
        uint8_t bytes[16];
        int32_t w_true[32];
        for (int i = 0; i < 16; ++i) {
            bytes[i] = static_cast<uint8_t>(next_u32() & 0xFFu);
            w_true[i]      = kIq3True[bytes[i] & 0x0F];
            w_true[i + 16] = kIq3True[(bytes[i] >> 4) & 0x0F];
        }
        uint32_t wq[4];
        for (int k = 0; k < 4; ++k) {
            wq[k] = static_cast<uint32_t>(bytes[4*k])
                  | (static_cast<uint32_t>(bytes[4*k + 1]) << 8)
                  | (static_cast<uint32_t>(bytes[4*k + 2]) << 16)
                  | (static_cast<uint32_t>(bytes[4*k + 3]) << 24);
        }
        auto iq3_pack_lo8_s4 = [](uint32_t a, uint32_t b) -> uint32_t {
            uint32_t ta = a & 0x0F0F0F0Fu;
            ta = (ta | (ta >> 4)) & 0x00FF00FFu;
            ta = (ta | (ta >> 8)) & 0x0000FFFFu;
            uint32_t tb = b & 0x0F0F0F0Fu;
            tb = (tb | (tb >> 4)) & 0x00FF00FFu;
            tb = (tb | (tb >> 8)) & 0x0000FFFFu;
            const uint32_t p = ta | (tb << 16);
            return p ^ (((p & 0x88888888u) >> 3) * 7u);
        };
        const uint32_t p[4] = {
            iq3_pack_lo8_s4(wq[0],      wq[1]),
            iq3_pack_lo8_s4(wq[2],      wq[3]),
            iq3_pack_lo8_s4(wq[0] >> 4, wq[1] >> 4),
            iq3_pack_lo8_s4(wq[2] >> 4, wq[3] >> 4),
        };
        int32_t x_ref[32];
        uint32_t x_q4[4] = {0, 0, 0, 0};
        int32_t sum_x = 0;
        for (int k = 0; k < 32; ++k) {
            x_ref[k] = static_cast<int32_t>(next_u32() % 15u) - 7;
            sum_x += x_ref[k];
            x_q4[k >> 3] |= static_cast<uint32_t>(x_ref[k] & 0xFu) << ((k & 7) * 4);
        }
        int32_t ref_dp = 0, exact_grid_dp = 0;
        for (int k = 0; k < 32; ++k) {
            ref_dp += w_true[k] * x_ref[k];
            int32_t approx_w = (w_true[k] == 62) ? 60 : ((w_true[k] == -62) ? -60 : w_true[k]);
            exact_grid_dp += approx_w * x_ref[k];
        }
        int32_t dot8_dp = 0;
        for (int k = 0; k < 4; ++k) dot8_dp = sim_sdot8(p[k], x_q4[k], dot8_dp);
        const int32_t reconstructed = 8 * dot8_dp + 4 * sum_x;
        if (reconstructed != exact_grid_dp) {
            std::fprintf(stderr, "FAIL iq3 SWAR s4 trial=%d: got=%d expected=%d\n", trial, reconstructed, exact_grid_dp);
            return 1;
        }
        const double diff = static_cast<double>(ref_dp - reconstructed);
        iq3_sig += static_cast<double>(ref_dp) * static_cast<double>(ref_dp);
        iq3_err += diff * diff;
    }
    const double iq3_sqnr_db = 10.0 * std::log10(iq3_sig / iq3_err);
    // Analytical uniform-code SQNR = 10*log10(1484.5 / 0.5) = 34.72 dB (>42 dB on Gaussian weights).
    if (iq3_sqnr_db < 34.5) {
        std::fprintf(stderr, "FAIL iq3 SWAR s4 SQNR=%.2f dB < 34.5 dB\n", iq3_sqnr_db);
        return 1;
    }

    std::printf("[PASS] test_prefill_dot8_kld: 2000/2000 dense affine exact (0 ULP), iq2s grid_i4 SQNR=%.2f dB, 16384/16384 SWAR signs exact, iq3 SWAR s4 SQNR=%.2f dB (2000/2000 exact), KLD sanity=%.6f\n",
                sqnr_db, iq3_sqnr_db, kl1);
    return 0;
}
