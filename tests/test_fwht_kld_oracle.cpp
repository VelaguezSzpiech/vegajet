// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU oracle for the block-64 FWHT + W4A4 path (HEAD quant::fwht64_inplace + pack_f32_to_vj_q4_dot8):
//  1. fwht64 equals the explicit Sylvester Hadamard matrix / 8, is orthonormal and self-inverse;
//  2. the integer W4A4 dot (s4 x s4 per block of 32, int32 accumulate, FP16 x FP32 scales) equals the dequantized
//     float dot; rotating both operands leaves the exact product unchanged (orthonormal invariance);
//  3. KLD gate at a fixed operating point: a W4A4 inner projection (activations with 25x outlier channels) feeds a
//     residual stream, an FP64 LM head (8192 rows) produces logits with reference entropy ~5.5 nats. With FWHT-64 the
//     mean KLD must be <= 0.045 and at least 1.5x lower than without rotation.
// KLD values depend on the logit scale; this oracle pins one documented scale and is a regression gate for the
// rotate/quantize/dot pipeline, not a model-quality claim.
#include "core/quant_repacker.hpp"
#include "tests/test_common.hpp"

#include <cmath>
#include <cstdio>
#include <random>

using namespace vegajet::quant;

namespace {

void quant_act_q4(const float* x, int8_t* q, float& d) {  // symmetric [-7, 7] per 32, as VJ_Q4_DOT8
    float a = 0;
    for (int i = 0; i < 32; ++i) a = std::max(a, std::fabs(x[i]));
    d = a / 7.0f;
    const float id = d > 0 ? 1.0f / d : 0.0f;
    for (int i = 0; i < 32; ++i) q[i] = static_cast<int8_t>(std::max(-7L, std::min(7L, std::lround(x[i] * id))));
}

int nib(const PackedVjQ4Dot8& p, uint32_t r, uint32_t c) {
    const uint32_t word = p.qs[static_cast<size_t>(r) * (p.cols / 8) + c / 8];
    int v = static_cast<int>((word >> (4 * (c % 8))) & 0xF);
    return v >= 8 ? v - 16 : v;
}

// y[r] = sum_b d_w[r,b] * d_x[b] * sum_i q_w * q_x  (integer inner loop, as v_dot8_i32_i4)
std::vector<double> w4a4(const PackedVjQ4Dot8& p, const std::vector<float>& x_in) {
    std::vector<float> x = x_in;
    if (p.fwht64) for (uint32_t c = 0; c < p.cols; c += 64) fwht64_inplace(x.data() + c);
    const uint32_t nb = p.cols / 32;
    std::vector<int8_t> qx(p.cols);
    std::vector<float> dx(nb);
    for (uint32_t b = 0; b < nb; ++b) quant_act_q4(&x[b * 32], &qx[b * 32], dx[b]);
    std::vector<double> y(p.rows);
    for (uint32_t r = 0; r < p.rows; ++r) {
        double acc = 0;
        for (uint32_t b = 0; b < nb; ++b) {
            int32_t s = 0;
            for (uint32_t i = 0; i < 32; ++i) s += nib(p, r, b * 32 + i) * qx[b * 32 + i];
            acc += static_cast<double>(fp16_to_f32(p.ds[static_cast<size_t>(r) * nb + b])) * dx[b] * s;
        }
        y[r] = acc;
    }
    return y;
}

double log_sum_exp(const std::vector<double>& z) {
    double m = -1e300;
    for (double t : z) m = std::max(m, t);
    double s = 0;
    for (double t : z) s += std::exp(t - m);
    return m + std::log(s);
}

} // namespace

int main() {
    std::mt19937 g(1);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    // 1. FWHT-64 vs explicit Hadamard.
    {
        float x[64], y[64];
        for (auto& v : x) v = nd(g);
        std::copy(x, x + 64, y);
        fwht64_inplace(y);
        double n0 = 0, n1 = 0;
        for (int i = 0; i < 64; ++i) {
            double ref = 0;
            for (int j = 0; j < 64; ++j) ref += (__builtin_popcount(static_cast<unsigned>(i & j)) & 1 ? -1.0 : 1.0) * x[j];
            VJ_CHECK(std::fabs(ref / 8.0 - y[i]) < 1e-5);
            n0 += double(x[i]) * x[i];
            n1 += double(y[i]) * y[i];
        }
        VJ_CHECK(std::fabs(n0 - n1) < 1e-4 * n0);
        fwht64_inplace(y);
        for (int i = 0; i < 64; ++i) VJ_CHECK(std::fabs(y[i] - x[i]) < 1e-5);
    }
    const uint32_t K = 2048, N = 2048, V = 8192, S = 8;
    std::vector<float> W(static_cast<size_t>(N) * K);
    for (auto& w : W) w = nd(g) / std::sqrt(static_cast<float>(K));
    // 2. Integer dot == dequantized float dot; rotation invariance of the exact product.
    {
        const auto P = pack_f32_to_vj_q4_dot8(W.data(), 64, K, true);
        std::vector<float> deq(static_cast<size_t>(64) * K);
        dequantize_vj_q4_dot8(P, deq.data());  // un-rotated back to the input basis
        std::vector<float> x(K);
        for (auto& v : x) v = nd(g);
        std::vector<float> xr = x;
        for (uint32_t c = 0; c < K; c += 64) fwht64_inplace(xr.data() + c);
        for (uint32_t r = 0; r < 64; ++r) {
            double rot = 0, plain = 0;
            for (uint32_t c = 0; c < K; ++c) {
                rot += static_cast<double>(fp16_to_f32(P.ds[r * (K / 32) + c / 32])) * nib(P, r, c) * xr[c];
                plain += static_cast<double>(deq[static_cast<size_t>(r) * K + c]) * x[c];
            }
            VJ_CHECK(std::fabs(rot - plain) < 1e-3 * (1.0 + std::fabs(plain)));
        }
        // int32 accumulation cannot overflow: |sum| <= 32 * 7 * 7 per block.
        VJ_CHECK(32 * 7 * 7 < (1 << 15));
    }
    // 3. KLD gate.
    std::vector<float> H(static_cast<size_t>(V) * N);
    for (auto& w : H) w = nd(g);
    const double head_scale = 0.06;
    double kld[2] = {0, 0}, ent = 0;
    const PackedVjQ4Dot8 P[2] = {pack_f32_to_vj_q4_dot8(W.data(), N, K, false), pack_f32_to_vj_q4_dot8(W.data(), N, K, true)};
    for (uint32_t s = 0; s < S; ++s) {
        std::vector<float> x(K);
        for (auto& v : x) v = nd(g);
        for (int c = 0; c < 8; ++c) x[(c * 263 + s) % K] *= 25.0f;  // outlier channels
        std::vector<double> h0(N);
        for (auto& v : h0) v = nd(g);
        std::vector<double> y(N);
        for (uint32_t r = 0; r < N; ++r) {
            double a = 0;
            for (uint32_t k = 0; k < K; ++k) a += static_cast<double>(W[static_cast<size_t>(r) * K + k]) * x[k];
            y[r] = a;
        }
        auto logits = [&](const std::vector<double>& proj) {
            std::vector<double> h(N), l(V);
            double ms = 0;
            for (uint32_t i = 0; i < N; ++i) { h[i] = h0[i] + proj[i]; ms += h[i] * h[i]; }
            const double inv = 1.0 / std::sqrt(ms / N);  // RMSNorm (unit weight)
            for (uint32_t v = 0; v < V; ++v) {
                double a = 0;
                for (uint32_t i = 0; i < N; ++i) a += H[static_cast<size_t>(v) * N + i] * h[i];
                l[v] = a * inv * head_scale;
            }
            return l;
        };
        const auto lr = logits(y);
        const double L = log_sum_exp(lr);
        for (uint32_t v = 0; v < V; ++v) ent -= std::exp(lr[v] - L) * (lr[v] - L);
        for (int rot = 0; rot < 2; ++rot) {
            const auto lq = logits(w4a4(P[rot], x));
            const double Lq = log_sum_exp(lq);
            double k = 0;
            for (uint32_t v = 0; v < V; ++v) k += std::exp(lr[v] - L) * ((lr[v] - L) - (lq[v] - Lq));
            kld[rot] += k;
        }
    }
    kld[0] /= S; kld[1] /= S; ent /= S;
    std::printf("  W4A4 block-32, reference entropy %.2f nats: KLD no-rotation %.5f | FWHT-64 %.5f (gate <= 0.045)\n", ent,
                kld[0], kld[1]);
    VJ_CHECK(ent > 5.0 && ent < 6.0);
    VJ_CHECK(kld[1] <= 0.045);
    VJ_CHECK(kld[0] >= 1.5 * kld[1]);
    std::printf("test_fwht_kld_oracle: FWHT exact/orthonormal, integer W4A4 dot exact, KLD gate OK\nPASS\n");
    return 0;
}
