// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU oracle for the arithmetic whose scheduling changes in Wave 2. No runtime/GPU use.
#include "core/quant_repacker.hpp"
#include "tests/test_common.hpp"
#include <array>
#include <cmath>
#include <random>

namespace {
uint32_t bits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }
float f16(float x) { return vegajet::quant::fp16_to_f32(vegajet::quant::f32_to_fp16(x)); }

// Device block32_sum uses xor16,8,4,2,1; keeping that order is significant.
float reduce32(std::array<float, 32> a) {
    for (unsigned s : {16u, 8u, 4u, 2u, 1u}) {
        const auto old = a;
        for (unsigned l = 0; l < 32; ++l) a[l] += old[l ^ s];
    }
    return a[0];
}

void q4_order() {
    std::mt19937 r(0x4a80003u);
    // TP=3 QKV / GQA Out / GDN Out, and one non-specialized shape.
    for (unsigned nb : {80u, 96u, 128u, 64u}) {
        std::vector<std::array<int8_t, 32>> x(nb);
        std::vector<std::array<uint8_t, 32>> w(nb);
        std::vector<float> xd(nb), xs(nb), wd(nb), wm(nb);
        for (unsigned b = 0; b < nb; ++b) {
            xd[b] = f16(static_cast<float>((r() % 1023u) + 1u) / 2048.0f);
            wd[b] = f16(static_cast<float>((r() % 1023u) + 1u) / 1024.0f);
            wm[b] = f16(static_cast<float>(static_cast<int>(r() % 1023u) - 511) / 1024.0f);
            xs[b] = 0.0f;
            for (unsigned j = 0; j < 32; ++j) {
                x[b][j] = static_cast<int8_t>(static_cast<int>(r() % 255u) - 127);
                w[b][j] = static_cast<uint8_t>(r() & 15u);
                xs[b] += static_cast<float>(x[b][j]) * xd[b];
            }
        }
        std::array<float, 32> oracle{}, mega{};
        for (unsigned lane = 0; lane < 32; ++lane) {
            // Reference uses complete grouped K loads then accumulates ascending K.
            for (unsigned k = 0; k < (nb + 31u) / 32u; ++k) {
                const unsigned b = lane + k * 32u;
                if (b >= nb) continue;
                int32_t dp = 0;
                for (unsigned j = 0; j < 32; ++j) dp += x[b][j] * w[b][j];
                oracle[lane] = std::fma(wd[b] * xd[b], static_cast<float>(dp),
                                      std::fma(wm[b], xs[b], oracle[lane]));
            }
            // Mega streams one block, prefetching its successor, instead of grouped loads.
            for (unsigned b = lane; b < nb; b += 32u) {
                int32_t dp = 0;
                // Eight sdot4 instructions: low half first, then high half.
                for (unsigned i = 0; i < 8; ++i)
                    for (unsigned j = 0; j < 4; ++j) dp += x[b][i * 4u + j] * w[b][i * 4u + j];
                mega[lane] = std::fma(wd[b] * xd[b], static_cast<float>(dp),
                                    std::fma(wm[b], xs[b], mega[lane]));
            }
            VJ_CHECK(bits(oracle[lane]) == bits(mega[lane]));
        }
        VJ_CHECK(bits(reduce32(oracle)) == bits(reduce32(mega)));
    }
}

void residual_reread() {
    constexpr unsigned D = 3072u;
    std::mt19937 r(0xdead122bu);
    std::array<std::vector<float>, 3> partial;
    for (auto& p : partial) {
        p.resize(D);
        for (float& f : p) f = static_cast<float>(static_cast<int>(r() % 65535u) - 32767) / 37.0f;
    }
    std::vector<float> h(D), oracle(D), mega(D);
    for (float& f : h) f = static_cast<float>(static_cast<int>(r() % 65535u) - 32767) / 173.0f;
    for (unsigned rank = 0; rank < 3u; ++rank) {
        // Expected TP reduction always rounds *all* ranks through FP16, including local rank.
        for (unsigned i = 0; i < D; ++i) {
            float sum = f16(partial[0][i]);
            sum += f16(partial[1][i]); sum += f16(partial[2][i]);
            oracle[i] = h[i] + sum;
        }
        mega = h;
        std::array<float, 256> ss_mega{}, ss_oracle{};
        for (unsigned tid = 0; tid < 256u; ++tid) {
            std::array<std::array<float, 8>, 2> kept{};
            for (unsigned it = 0; it < 2u; ++it) {
                const unsigned i8 = tid + 256u * it;
                if (i8 >= D / 8u) continue;
                for (unsigned j = 0; j < 8u; ++j) {
                    const unsigned i = i8 * 8u + j;
                    float sum = 0.0f;
                    for (unsigned src = 0; src < 3u; ++src) {
                        // Local path does float->half->float, peer path reads prepacked half.
                        const float v = f16(partial[src][i]);
                        if (src == 0u) sum = v; else sum += v;
                    }
                    mega[i] += sum;
                    kept[it][j] = oracle[i];
                    VJ_CHECK(bits(mega[i]) == bits(kept[it][j]));
                }
                for (unsigned base : {0u, 4u}) {
                    float& a = ss_mega[tid]; const unsigned i = i8 * 8u + base;
                    a = std::fma(mega[i], mega[i], std::fma(mega[i+1], mega[i+1],
                        std::fma(mega[i+2], mega[i+2], std::fma(mega[i+3], mega[i+3], a))));
                    float& b = ss_oracle[tid];
                    b = std::fma(kept[it][base], kept[it][base], std::fma(kept[it][base+1], kept[it][base+1],
                        std::fma(kept[it][base+2], kept[it][base+2], std::fma(kept[it][base+3], kept[it][base+3], b))));
                }
            }
            VJ_CHECK(bits(ss_mega[tid]) == bits(ss_oracle[tid]));
        }
    }
}
}
int main() { q4_order(); residual_reread(); std::puts("mega arithmetic CPU oracle passed"); }
