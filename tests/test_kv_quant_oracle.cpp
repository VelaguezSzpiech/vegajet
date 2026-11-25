// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/kv_quant_oracle: fp16 conversion, the v_dot8_i32_i4 model, q4_0 / v_dot8 packing against an independent ggml
// Q4_0 reference (bit-exact both ways), the integer score paths, turbo3 rotation / bit planes / Lloyd-Max distortion,
// and the attention-score error + KLD oracles for head_dim 128 and 256 (with and without outlier channels).
#include "core/kv_quant_oracle.hpp"
#include "tests/test_common.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

using namespace vegajet::kvq;

namespace {

// ggml quantize_row_q4_0_ref / dequantize_row_q4_0 rewritten from the published block definition:
// struct { fp16 d; uint8 qs[16]; }, qs[j] = q(j) | q(j + 16) << 4, value = d * (q - 8).
void ggml_quantize(const float* x, uint8_t* blk) {
    float amax = 0.0f, max = 0.0f;
    for (int j = 0; j < 32; ++j)
        if (amax < std::fabs(x[j])) { amax = std::fabs(x[j]); max = x[j]; }
    const float d = max / -8, id = d ? 1.0f / d : 0.0f;
    const uint16_t dh = f32_to_f16(d);
    std::memcpy(blk, &dh, 2);
    for (int j = 0; j < 16; ++j) {
        const float x0 = x[j] * id, x1 = x[16 + j] * id;
        const uint8_t a = static_cast<uint8_t>(std::min(15, static_cast<int>(static_cast<int8_t>(x0 + 8.5f))));
        const uint8_t b = static_cast<uint8_t>(std::min(15, static_cast<int>(static_cast<int8_t>(x1 + 8.5f))));
        blk[2 + j] = static_cast<uint8_t>(a | (b << 4));
    }
}

void ggml_dequantize(const uint8_t* blk, float* y) {
    uint16_t dh;
    std::memcpy(&dh, blk, 2);
    const float d = f16_to_f32(dh);
    for (int j = 0; j < 16; ++j) {
        y[j] = static_cast<float>((blk[2 + j] & 0x0F) - 8) * d;
        y[j + 16] = static_cast<float>((blk[2 + j] >> 4) - 8) * d;
    }
}

double dot(const float* a, const float* b, uint32_t n) {
    double s = 0.0;
    for (uint32_t i = 0; i < n; ++i) s += double(a[i]) * b[i];
    return s;
}

} // namespace

int main() {
    std::mt19937_64 rng(7);
    std::normal_distribution<float> N(0.0f, 1.0f);

    // ---- fp16: every finite half round-trips; ties go to even; overflow -> inf; subnormals survive.
    for (uint32_t h = 0; h < 0x10000u; ++h) {
        if ((h & 0x7C00u) == 0x7C00u && (h & 0x3FFu)) continue;  // NaN payloads
        VJ_CHECK(f32_to_f16(f16_to_f32(static_cast<uint16_t>(h))) == h);
    }
    VJ_CHECK(f32_to_f16(1.0f + std::ldexp(1.0f, -11)) == 0x3C00u);      // tie -> even (1.0)
    VJ_CHECK(f32_to_f16(1.0f + 3 * std::ldexp(1.0f, -11)) == 0x3C02u);  // tie -> even (1 + 2^-9)
    VJ_CHECK(f32_to_f16(70000.0f) == 0x7C00u && f32_to_f16(-1e-9f) == 0x8000u && f32_to_f16(std::ldexp(1.0f, -24)) == 1u);

    // ---- v_dot8_i32_i4 model: every nibble pair in every lane, accumulator, random words vs scalar.
    static_assert(dot8_i4(0x88888888u, 0x88888888u, 0) == 8 * 64, "-8 * -8 in all lanes");
    static_assert(dot8_i4(0x77777777u, 0x88888888u, 5) == 5 - 8 * 56, "7 * -8 in all lanes");
    for (int n = 0; n < 8; ++n)
        for (uint32_t a = 0; a < 16; ++a)
            for (uint32_t b = 0; b < 16; ++b) {
                const int sa = a >= 8 ? int(a) - 16 : int(a), sb = b >= 8 ? int(b) - 16 : int(b);
                VJ_CHECK(dot8_i4(a << (4 * n), b << (4 * n), 3) == 3 + sa * sb);
            }
    for (int i = 0; i < 10000; ++i) {
        const uint32_t a = static_cast<uint32_t>(rng()), b = static_cast<uint32_t>(rng());
        int ref = 0;
        for (int n = 0; n < 8; ++n) {
            const int x = (a >> (4 * n)) & 15, y = (b >> (4 * n)) & 15;
            ref += (x >= 8 ? x - 16 : x) * (y >= 8 ? y - 16 : y);
        }
        VJ_CHECK(dot8_i4(a, b, 0) == ref);
    }

    // ---- q4_0 / v_dot8 vs the ggml reference (bit-exact codes, scales and dequantized values).
    for (uint32_t D : {128u, 256u}) {
        for (int trial = 0; trial < 200; ++trial) {
            std::vector<float> x(D);
            for (auto& v : x) v = N(rng) * (trial % 3 == 0 ? 0.01f : 1.0f);
            if (trial % 5 == 1) x[rng() % D] = 40.0f * (trial % 2 ? -1.0f : 1.0f);  // single outlier
            if (trial == 2) std::fill(x.begin(), x.end(), 0.0f);
            if (trial == 4) std::fill(x.begin(), x.end(), -3.0f);
            std::vector<uint32_t> c(D / 8);
            std::vector<uint16_t> s(D / 32);
            std::vector<float> y(D), yr(D);
            q4_pack(x.data(), D, c.data(), s.data());
            q4_unpack(c.data(), s.data(), D, y.data());
            for (uint32_t b = 0; b < D / 32; ++b) {
                uint8_t ref[18], got[18];
                ggml_quantize(&x[32 * b], ref);
                dot8_to_ggml_q4_0(&c[4 * b], s[b], got);
                VJ_CHECK(std::memcmp(ref, got, 18) == 0);
                ggml_dequantize(ref, &yr[32 * b]);
            }
            VJ_CHECK(std::memcmp(y.data(), yr.data(), D * sizeof(float)) == 0);
        }
    }
    for (int i = 0; i < 5000; ++i) {  // any 18-byte ggml block converts losslessly both ways
        uint8_t blk[18], back[18];
        for (auto& b : blk) b = static_cast<uint8_t>(rng());
        uint32_t c[4];
        uint16_t s;
        ggml_q4_0_to_dot8(blk, c, &s);
        dot8_to_ggml_q4_0(c, s, back);
        VJ_CHECK(std::memcmp(blk, back, 18) == 0);
        float a[32], b2[32];
        ggml_dequantize(blk, a);
        q4_unpack(c, &s, 32, b2);
        for (int j = 0; j < 32; ++j) VJ_CHECK((std::isnan(a[j]) && std::isnan(b2[j])) || a[j] == b2[j]);
    }

    // ---- integer score paths == float dot of the dequantized operands.
    for (uint32_t D : {128u, 256u}) {
        for (int trial = 0; trial < 100; ++trial) {
            std::vector<float> q(D), k(D), kd(D), qd(D), qs(D / 32);
            for (auto& v : q) v = N(rng);
            for (auto& v : k) v = N(rng);
            std::vector<uint32_t> kc(D / 8), qc(D / 8);
            std::vector<uint16_t> ks(D / 32);
            std::vector<int8_t> q8(D);
            q4_pack(k.data(), D, kc.data(), ks.data());
            q4_unpack(kc.data(), ks.data(), D, kd.data());
            query_pack_i4(q.data(), D, qc.data(), qs.data());
            for (uint32_t j = 0; j < D; ++j) {
                const uint32_t nib = (qc[j / 8] >> (4 * (j % 8))) & 15u;
                qd[j] = qs[j / 32] * static_cast<float>(nib >= 8 ? int(nib) - 16 : int(nib));
            }
            const double r4 = dot(qd.data(), kd.data(), D);
            VJ_CHECK(std::fabs(q4_dot_i4(qc.data(), qs.data(), kc.data(), ks.data(), D) - r4) <= 1e-4 * (1 + std::fabs(r4)));
            query_pack_i8(q.data(), D, q8.data(), qs.data());
            for (uint32_t j = 0; j < D; ++j) qd[j] = qs[j / 32] * q8[j];
            const double r8 = dot(qd.data(), kd.data(), D);
            VJ_CHECK(std::fabs(q4_dot_i8(q8.data(), qs.data(), kc.data(), ks.data(), D) - r8) <= 1e-4 * (1 + std::fabs(r8)));
        }
    }

    // ---- turbo3
    {
        const auto s1 = turbo3_signs(256, 99), s2 = turbo3_signs(256, 99), s3 = turbo3_signs(256, 100);
        VJ_CHECK(s1 == s2 && s1 != s3);
        int plus = 0;
        for (auto v : s1) { VJ_CHECK(v == 1 || v == -1); plus += v == 1; }
        VJ_CHECK(plus > 96 && plus < 160);
        VJ_CHECK(bits_per_value(KvFormat::Turbo3) == 3.5 && bits_per_value(KvFormat::Q4Dot8) == 4.5);
    }
    for (uint32_t D : {128u, 256u}) {
        const auto sg = turbo3_signs(D, 0x5455524230u);
        std::vector<float> x(D), y(D), z(D), q(D), qr(D);
        for (auto& v : x) v = N(rng);
        for (auto& v : q) v = N(rng);
        turbo3_rotate(x.data(), D, sg.data(), y.data());
        turbo3_unrotate(y.data(), D, sg.data(), z.data());
        for (uint32_t i = 0; i < D; ++i) VJ_CHECK(std::fabs(z[i] - x[i]) < 1e-5f);
        turbo3_rotate(q.data(), D, sg.data(), qr.data());
        VJ_CHECK(std::fabs(dot(x.data(), x.data(), D) - dot(y.data(), y.data(), D)) < 1e-3 * D);
        VJ_CHECK(std::fabs(dot(q.data(), x.data(), D) - dot(qr.data(), y.data(), D)) < 1e-3 * std::sqrt(double(D)));
        // Bit planes: a rotated vector made of exact codebook values decodes to the same indices (scale 1).
        std::vector<float> e(D);
        for (uint32_t j = 0; j < D; ++j) e[j] = kTurbo3Codebook[(j * 5 + j / 32) % 8];
        std::vector<uint32_t> c(3 * D / 32);
        std::vector<uint16_t> s(D / 32);
        turbo3_pack_rotated(e.data(), D, c.data(), s.data());
        std::vector<float> back(D);
        turbo3_unpack_rotated(c.data(), s.data(), D, back.data());
        for (uint32_t j = 0; j < D; ++j) VJ_CHECK(std::fabs(back[j] - e[j]) < 2e-3f);
        for (uint32_t b = 0; b < D / 32; ++b) VJ_CHECK(std::fabs(f16_to_f32(s[b]) - 1.0f) < 1e-3f);
        // Layout written independently from the header: low 2 bits of element 16w+j at bits 2j of dword w, high bit
        // of element j at bit j of dword 2.
        for (uint32_t b = 0; b < D / 32; ++b) {
            c[3 * b] = c[3 * b + 1] = c[3 * b + 2] = 0;
            s[b] = f32_to_f16(0.5f);
            for (uint32_t j = 0; j < 32; ++j) {
                const uint32_t idx = (j * 3 + b) % 8;
                c[3 * b + j / 16] |= (idx & 3u) << (2 * (j % 16));
                c[3 * b + 2] |= (idx >> 2) << j;
            }
        }
        turbo3_unpack_rotated(c.data(), s.data(), D, back.data());
        for (uint32_t j = 0; j < D; ++j) VJ_CHECK(back[j] == 0.5f * kTurbo3Codebook[((j % 32) * 3 + j / 32) % 8]);
        // Lloyd-Max distortion on Gaussian data (fixed-scale theory 0.0345; the per-block scale refit does better) and the rotated-domain score.
        double err = 0.0, nrm = 0.0, worst_score = 0.0;
        for (int trial = 0; trial < 200; ++trial) {
            for (auto& v : x) v = N(rng);
            turbo3_pack(x.data(), D, sg.data(), c.data(), s.data());
            turbo3_unpack(c.data(), s.data(), D, sg.data(), z.data());
            for (uint32_t i = 0; i < D; ++i) { err += double(z[i] - x[i]) * (z[i] - x[i]); nrm += double(x[i]) * x[i]; }
            const double a = turbo3_dot_rotated(qr.data(), c.data(), s.data(), D), b = dot(q.data(), z.data(), D);
            worst_score = std::max(worst_score, std::fabs(a - b) / (1 + std::fabs(b)));
        }
        std::printf("  turbo3 D=%u: relative MSE %.4f (fixed-scale Lloyd-Max theory 0.0345), rotated-vs-unrotated score "
                    "diff %.2e\n", D, err / nrm, worst_score);
        VJ_CHECK(err / nrm > 0.018 && err / nrm < 0.0345);  // per-block scale refit: at or below fixed-scale Lloyd-Max
        VJ_CHECK(worst_score < 1e-4);
    }

    // ---- oracles: argument checks, then the table.
    {
        std::string err;
        OracleConfig bad;
        bad.head_dim = 96;
        run_oracle(bad, KvFormat::Turbo3, KvFormat::F32, QueryPath::F32, &err);
        VJ_CHECK(err.find("power-of-two") != std::string::npos);
        run_oracle(OracleConfig{}, KvFormat::Turbo3, KvFormat::F32, QueryPath::I8, &err);
        VJ_CHECK(err.find("integer query") != std::string::npos);
        const OracleStats ex = run_oracle(OracleConfig{}, KvFormat::F32, KvFormat::F32, QueryPath::F32, &err);
        VJ_CHECK(err.empty() && ex.kld_max < 1e-12 && ex.score_max_abs < 1e-9 && ex.top1_agree == 1.0);
    }
    struct Row { KvFormat k; QueryPath qp; };
    const Row rows[] = {{KvFormat::Q4Dot8, QueryPath::F32}, {KvFormat::Q4Dot8, QueryPath::I8},
                        {KvFormat::Q4Dot8, QueryPath::I4}, {KvFormat::Turbo3, QueryPath::F32}};
    std::printf("  %-4s %-8s %-10s %-4s %5s %10s %10s %10s %10s %7s %9s\n", "D", "outliers", "K format", "Q", "bits",
                "K rel MSE", "score rRMS", "KLD mean", "KLD max", "top-1", "out err");
    for (uint32_t D : {128u, 256u}) {
        for (uint32_t outl : {0u, 4u}) {
            OracleStats st[4];
            for (int r = 0; r < 4; ++r) {
                OracleConfig cfg;
                cfg.head_dim = D;
                cfg.outlier_channels = outl;
                cfg.seed = 1000 + D + outl;
                std::string err;
                st[r] = run_oracle(cfg, rows[r].k, rows[r].k, rows[r].qp, &err);
                VJ_CHECK(err.empty());
                std::printf("  %-4u %-8u %-10s %-4s %5.2f %10.5f %10.5f %10.2e %10.2e %7.3f %9.5f\n", D, outl,
                            format_name(rows[r].k), query_path_name(rows[r].qp), bits_per_value(rows[r].k), st[r].k_rel_mse,
                            st[r].score_rel_rms, st[r].kld_mean, st[r].kld_max, st[r].top1_agree, st[r].out_rel_err);
            }
            const OracleStats &q4 = st[0], &q4i8 = st[1], &q4i4 = st[2], &t3 = st[3];
            // Regression bounds on this synthetic data (roughly 2x the values measured when the oracle was written).
            if (outl == 0) {
                VJ_CHECK(q4.kld_mean < 0.01 && q4i8.kld_mean < 0.01 && q4i4.kld_mean < 0.02 && t3.kld_mean < 0.03);
                VJ_CHECK(q4.top1_agree >= 0.9 && t3.top1_agree >= 0.75);
                // Without outliers one more bit wins: q4_0 (4.5 b) beats turbo3 (3.5 b) on every metric.
                VJ_CHECK(q4.k_rel_mse < t3.k_rel_mse && q4.kld_mean < t3.kld_mean && q4.out_rel_err < t3.out_rel_err);
            } else {
                VJ_CHECK(q4.kld_mean < 0.2 && t3.kld_mean < 0.15 && q4.top1_agree >= 0.65 && t3.top1_agree >= 0.65);
                // Persistent outlier channels: absmax blocks lose the other 31 values, the rotation spreads the
                // outlier, so turbo3 wins with one bit less per value.
                VJ_CHECK(t3.k_rel_mse < q4.k_rel_mse && t3.kld_mean < q4.kld_mean && t3.score_rel_rms < q4.score_rel_rms);
            }
            // An int8 query adds almost nothing over float; an int4 query (pure v_dot8) adds measurable error.
            VJ_CHECK(q4i8.score_rel_rms < 1.15 * q4.score_rel_rms);
            VJ_CHECK(q4i4.score_rel_rms > 1.1 * q4.score_rel_rms && q4i4.kld_mean > q4.kld_mean);
            // turbo3 keeps its distortion whatever the channel scales.
            VJ_CHECK(t3.k_rel_mse > 0.018 && t3.k_rel_mse < 0.0345);
        }
    }
    std::printf("test_kv_quant_oracle: fp16, v_dot8 model, q4_0/dot8 vs ggml, integer scores, turbo3, oracles OK\nPASS\n");
    return 0;
}
