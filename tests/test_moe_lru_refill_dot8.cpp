// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Standalone C++17: use pristine source root as the include directory.
// CPU-only model of the device re-code; no host production refill path.
#include "ops/kernels/vj_moe_lru_refill_dot8.hiph"
#include "core/ggml_iq_tables.hpp"
#include "core/q4_dot8_repack.hpp"
#include <array>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
#include <numeric>
#ifdef NDEBUG
#error This proof requires assertions enabled.
#endif

namespace T = vegajet::ggml_tables;
using vj_refill::Source;
static uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float half_reference(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16;
    uint32_t e = (h >> 10) & 31, m = h & 1023, u;
    if (!e) {
        if (!m) u = sign;
        else {
            int shift = 0;
            while (!(m & 1024)) { m <<= 1; ++shift; }
            u = sign | (uint32_t(113 - shift) << 23) | ((m & 1023) << 13);
        }
    } else u = sign | ((e == 31 ? 255 : e + 112) << 23) | (m << 13);
    float f; std::memcpy(&f, &u, 4); return f;
}
struct Pristine { int integer; float scale; };
// Independent decode of pristine GGUF bytes/tables, not the cache decoder.
static Pristine source_reference(const uint8_t* p, Source type, unsigned i) {
    uint16_t h; std::memcpy(&h, p, 2);
    const unsigned b = i / 32, l = (i % 32) / 8, j = i % 8;
    if (type == Source::iq2_s) {
        const unsigned index = p[2 + b * 4 + l] + (((p[66 + b] >> (2 * l)) & 3u) << 8);
        const unsigned scale = (p[74 + b] >> (4 * (l / 2))) & 15;
        const int mag = int((T::iq2s_grid[index] >> (j * 8)) & 255);
        return {((p[34 + b * 4 + l] >> j) & 1) ? -mag : mag,
            half_reference(h) * (0.5f + scale) * 0.25f};
    }
    uint32_t aux; std::memcpy(&aux, p + 66 + b * 4, 4);
    const unsigned signs = T::ksigns_iq2xs[(aux >> (l * 7)) & 127];
    const uint32_t grid = T::iq3xxs_grid[p[2 + b * 8 + l * 2 + j / 4]];
    const int mag = int((grid >> ((j % 4) * 8)) & 255);
    return {((signs >> j) & 1) ? -mag : mag,
        half_reference(h) * (0.5f + (aux >> 28)) * 0.5f};
}
static uint32_t random_word(uint32_t& s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
static int signed_byte(uint32_t w, unsigned j) {
    const int b = int((w >> (j * 8)) & 255); return b < 128 ? b : b - 256;
}
static int unpack(const uint32_t* w, Source type, unsigned i) {
    const unsigned h = i / 16, j = i % 16;
    return signed_byte(vj_refill::decode4((w[j / 4] >> (4 * h)) & 0x0f0f0f0fu, type), j % 4);
}
static int activation(unsigned token, unsigned k) {
    uint32_t value = (k + 1) * 0x9e3779b9u ^ (token + 1) * 0x85ebca6bu;
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    return int(value % 16) - 8;
}
static int signed_nibble(uint32_t word, unsigned n) {
    const int value = int((word >> (4 * n)) & 15u);
    return value < 8 ? value : value - 16;
}
// Emulate v_dot8_i32_i4, independently of the pristine scalar dot.
static int dot8(uint32_t weights, uint32_t activations, bool unsigned_x = false) {
    int sum = 0;
    for (unsigned n = 0; n < 8; ++n)
        sum += signed_nibble(weights, n) * (unsigned_x ?
            int((activations >> (4 * n)) & 15u) : signed_nibble(activations, n));
    return sum;
}
// Same packed-word extraction and two-dot decomposition as hit_dot8.
// The expected path below never calls this, decode4, or signed_nibble.
static int cache_dot8(const uint32_t* words, Source type, unsigned half,
                      unsigned group, uint32_t activations, bool unsigned_x = false) {
    uint32_t low = 0, high = 0;
    for (unsigned u = 0; u < 2; ++u) {
        const uint32_t decoded = vj_refill::decode4(
            (words[2 * group + u] >> (4 * half)) & 0x0f0f0f0fu, type);
        for (unsigned c = 0; c < 4; ++c) {
            const int weight = signed_byte(decoded, c);
            const int lo = int((unsigned(weight) + 8u) & 15u) - 8;
            const int hi = (weight - lo) / 16;
            assert(hi >= -8 && hi <= 7);
            const unsigned shift = 16 * u + 4 * c;
            low |= (unsigned(lo) & 15u) << shift;
            high |= (unsigned(hi) & 15u) << shift;
        }
    }
    return dot8(low, activations, unsigned_x) + 16 * dot8(high, activations, unsigned_x);
}
static float activation_scale(unsigned token, unsigned b) {
    return float(3 + ((token * 7 + b * 11) % 29)) / 8192.f;
}
static void run(Source type, unsigned rows, unsigned cols) {
    const unsigned nb = cols / 32, block_bytes = type == Source::iq2_s ? 82 : 98;
    const unsigned ws = vj_refill::stride(cols / 2), ss = vj_refill::stride(nb * sizeof(vj_refill::Scale));
    const unsigned wgcd = std::gcd(ws / 256, 32u), sgcd = std::gcd(ss / 256, 32u);
    assert(ws >= cols / 2 && ws % 256 == 0 && wgcd == 1);
    assert(ss >= nb * sizeof(vj_refill::Scale) && ss % 256 == 0 && sgcd == 1);
    assert(std::gcd(ws, 8192u) == 256 && std::gcd(ss, 8192u) == 256);
    constexpr unsigned slot_count = 2, token_count = 7;
    std::vector<uint32_t> weights(size_t(slot_count) * rows * ws / 4, 0xdeadbeef);
    std::vector<vj_refill::Scale> scales(size_t(slot_count) * rows * ss / 4, {0xdead, 0xad, 0xde});
    std::vector<uint8_t> src(size_t(slot_count) * rows * cols / 256 * block_bytes);
    uint32_t rng = 0x71837421;
    for (auto& b : src) b = uint8_t(random_word(rng));
    constexpr uint16_t edge[] = {0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3555, 0xb555, 0x3c00, 0x7bff};
    for (size_t b = 0; b < src.size() / block_bytes; ++b) {
        const uint16_t h = b % 17 < 10 ? edge[b % 17] : uint16_t((random_word(rng) & 0x83ff) | 0x2800);
        std::memcpy(src.data() + b * block_bytes, &h, 2);
    }
    const vj_refill::Tables tables{T::iq2s_grid, T::iq3xxs_grid, T::ksigns_iq2xs};
    uint64_t count62 = 0, elements = 0, element_mismatches = 0, scale_mismatches = 0;
    for (unsigned row = 0; row < slot_count * rows; ++row) {
        auto* w = weights.data() + size_t(row) * ws / 4;
        auto* s = scales.data() + size_t(row) * ss / 4;
        for (unsigned b = 0; b < nb; ++b) {
            const auto* p = src.data() + (size_t(row) * cols / 256 + b / 8) * block_bytes;
            vj_refill::repack32(p, type, b % 8, tables, w + b * 4, s + b);
            uint16_t d; std::memcpy(&d, p, 2);
            assert(s[b].d == d && s[b].reserved == 0);
            assert(s[b].sub == (type == Source::iq2_s ? p[74 + b % 8] : p[69 + (b % 8) * 4] >> 4));
            for (unsigned j = 0; j < 32; ++j) {
                const auto ref = source_reference(p, type, (b % 8) * 32 + j);
                const int q = unpack(w + b * 4, type, j);
                ++elements;
                element_mismatches += q != ref.integer;
                scale_mismatches += bits(vj_refill::scale_value(s[b], type, j / 16)) != bits(ref.scale);
                assert(q == ref.integer);
                assert(bits(vj_refill::scale_value(s[b], type, j / 16)) == bits(ref.scale));
                assert(bits(vj_refill::scale_value(s[b], type, j / 16) * float(q)) == bits(ref.scale * float(ref.integer)));
                count62 += q == 62 || q == -62;
            }
        }
        for (unsigned j = nb * 4; j < ws / 4; ++j) assert(w[j] == 0xdeadbeef);
        for (unsigned j = nb; j < ss / 4; ++j) assert(s[j].d == 0xdead && s[j].sub == 0xad && s[j].reserved == 0xde);
    }
    if (type == Source::iq3_xxs) assert(count62 > 0);
    // Packed activation storage is built from independent signed values; the
    // pristine path reads those values directly, not a cache unpack helper.
    std::vector<int> values(size_t(token_count) * cols);
    std::vector<uint32_t> packed(size_t(token_count) * cols / 8, 0);
    std::vector<float> xd(size_t(token_count) * nb);
    std::array<unsigned, 16> activation_counts{};
    for (unsigned token = 0; token < token_count; ++token) {
        for (unsigned k = 0; k < cols; ++k) {
            const int x = activation(token, k);
            values[size_t(token) * cols + k] = x;
            ++activation_counts[unsigned(x + 8)];
            packed[size_t(token) * cols / 8 + k / 8] |= (unsigned(x) & 15u) << (4 * (k % 8));
        }
        for (unsigned b = 0; b < nb; ++b) xd[size_t(token) * nb + b] = activation_scale(token, b);
    }
    for (auto n : activation_counts) assert(n > 0);
    // Non-identity job maps exercise repeated slots and distinct tokens.
    constexpr std::array<unsigned, 3> slots{1, 0, 1}, tokens{6, 2, 4};
    uint64_t outputs = 0, output_mismatches = 0, dots = 0, dot_mismatches = 0;
    uint64_t half_bad_outputs = 0, sign_bad_outputs = 0;
    uint64_t half_bad_dots = 0, sign_bad_dots = 0, rotated_outputs = 0;
    for (unsigned stagger : {0u, 16u, 31u}) {
        const uint64_t previous_half = half_bad_outputs, previous_sign = sign_bad_outputs;
        std::vector<float> actual(slots.size() * rows), expected(slots.size() * rows);
        for (unsigned job = 0; job < slots.size(); ++job) for (unsigned row = 0; row < rows; ++row) {
            const unsigned token = tokens[job];
            const size_t r = size_t(slots[job]) * rows + row;
            const unsigned start = ((row / 4) * stagger) % nb;
            rotated_outputs += start != 0;
            const auto* w = weights.data() + r * ws / 4;
            const auto* s = scales.data() + r * ss / 4;
            std::array<float, 64> cached{}, pristine{}, half_bad{}, sign_bad{};
            std::vector<unsigned> visits(nb);
            for (unsigned lane = 0; lane < 64; ++lane) for (unsigned i = lane; i < nb; i += 64) {
                const unsigned b = (start + i) % nb;
                ++visits[b];
                const auto* p = src.data() + (r * (cols / 256) + b / 8) * block_bytes;
                for (unsigned h = 0; h < 2; ++h) {
                    int dp = 0, reference = 0, wrong_half = 0, wrong_sign = 0;
                    for (unsigned j = 0; j < 2; ++j) {
                        const uint32_t x = packed[(size_t(token) * nb + b) * 4 + h * 2 + j];
                        const int got = cache_dot8(w + b * 4, type, h, j, x);
                        const int bad_half = cache_dot8(w + b * 4, type, h ^ 1u, j, x);
                        const int bad_sign = cache_dot8(w + b * 4, type, h, j, x, true);
                        int want = 0;
                        for (unsigned k = 0; k < 8; ++k) {
                            const unsigned n = h * 16 + j * 8 + k;
                            want += source_reference(p, type, (b % 8) * 32 + n).integer *
                                values[size_t(token) * cols + b * 32 + n];
                        }
                        ++dots;
                        dot_mismatches += got != want;
                        half_bad_dots += bad_half != want;
                        sign_bad_dots += bad_sign != want;
                        dp += got; reference += want;
                        wrong_half += bad_half; wrong_sign += bad_sign;
                    }
                    const float scale = vj_refill::scale_value(s[b], type, h) * xd[size_t(token) * nb + b];
                    const float ref_scale = source_reference(p, type, (b % 8) * 32 + h * 16).scale * xd[size_t(token) * nb + b];
                    cached[lane] = std::fma(scale, float(dp), cached[lane]);
                    pristine[lane] = std::fma(ref_scale, float(reference), pristine[lane]);
                    half_bad[lane] = std::fma(scale, float(wrong_half), half_bad[lane]);
                    sign_bad[lane] = std::fma(scale, float(wrong_sign), sign_bad[lane]);
                }
            }
            for (auto n : visits) assert(n == 1);
            // Only the lane-zero dependency tree is needed; every add uses
            // the same operands/order as the device's descending shfl_down.
            for (unsigned delta = 32; delta; delta /= 2)
                for (unsigned lane = 0; lane < delta; ++lane) {
                    cached[lane] += cached[lane + delta]; pristine[lane] += pristine[lane + delta];
                    half_bad[lane] += half_bad[lane + delta]; sign_bad[lane] += sign_bad[lane + delta];
                }
            const size_t out = size_t(job) * rows + row;
            actual[out] = cached[0]; expected[out] = pristine[0];
            half_bad_outputs += bits(half_bad[0]) != bits(pristine[0]);
            sign_bad_outputs += bits(sign_bad[0]) != bits(pristine[0]);
        }
        for (size_t out = 0; out < actual.size(); ++out) {
            ++outputs;
            output_mismatches += bits(actual[out]) != bits(expected[out]);
        }
        assert(half_bad_outputs > previous_half && sign_bad_outputs > previous_sign);
    }
    assert(elements == uint64_t(slot_count) * rows * cols);
    assert(outputs == uint64_t(rows) * slots.size() * 3);
    assert(dots == outputs * nb * 4 && rotated_outputs > 0);
    std::printf("%s %ux%u: elements=%llu element_mismatches=%llu scale_mismatches=%llu "
        "dot8_groups=%llu dot_mismatches=%llu outputs=%llu output_mismatches=%llu "
        "negative_half_dots=%llu negative_sign_dots=%llu negative_half_outputs=%llu negative_sign_outputs=%llu "
        "rotated_outputs=%llu weight_stride=%u scale_stride=%u gcd_weight_units_32=%u gcd_scale_units_32=%u "
        "gcd_weight_bytes_8192=%u gcd_scale_bytes_8192=%u |62|=%llu\n",
        type == Source::iq2_s ? "IQ2_S" : "IQ3_XXS", rows, cols,
        (unsigned long long)elements, (unsigned long long)element_mismatches, (unsigned long long)scale_mismatches,
        (unsigned long long)dots, (unsigned long long)dot_mismatches, (unsigned long long)outputs,
        (unsigned long long)output_mismatches, (unsigned long long)half_bad_dots, (unsigned long long)sign_bad_dots,
        (unsigned long long)half_bad_outputs, (unsigned long long)sign_bad_outputs, (unsigned long long)rotated_outputs,
        ws, ss, wgcd, sgcd, std::gcd(ws, 8192u), std::gcd(ss, 8192u), (unsigned long long)count62);
    assert(element_mismatches == 0 && scale_mismatches == 0 && dot_mismatches == 0 && output_mismatches == 0);
    assert(half_bad_dots > 0 && sign_bad_dots > 0 && half_bad_outputs > 0 && sign_bad_outputs > 0);
}
int main() {
    for (int w=-62;w<=62;++w) for(int x=-8;x<8;++x) {
        int lo=int((unsigned(w)+8)&15)-8,hi=(w-lo)/16;
        assert(hi>=-8 && hi<=7 && x*w==x*lo+16*x*hi);
    }
    // Compare every selector byte against the pristine constexpr perm model,
    // including the unusual IQ3 negative-code selectors 8..15.
    for (unsigned c = 0; c < 16; ++c) for (unsigned byte = 0; byte < 4; ++byte) {
        const uint32_t code = c << (byte * 8);
        assert(vj_refill::decode4(code, Source::iq3_xxs) == vegajet::dot8_decode4_iq3(code));
        assert(vj_refill::decode4(code, Source::iq2_s) == vegajet::dot8_decode4_iq2(code));
    }
    for (Source type : {Source::iq2_s, Source::iq3_xxs}) {
        run(type, 1024, 3072); run(type, 3072, 1024);
    }
}
