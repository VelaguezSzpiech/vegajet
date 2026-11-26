// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU contract and independent-reference test for the normal-engine sealed-wire Q/KV producer.
//
// It does not run device code. It proves, byte for byte or within the declared quantizer error:
//   * the host producer matches an independently written native-layout reference (Q8 and Q4),
//   * the Q8 KV adaptor is lossless with respect to the active 272 B Q8_0 rows,
//   * a consumer that reads only the produced wire bytes reproduces a direct causal softmax for
//     M = 1 and M = 5..8, including causal tails and empty splits,
//   * the epilogue reproduces the active sigmoid-gate + block-32 INT8 quantizer,
//   * cache updates refresh exactly the uncommitted rows after a speculative rollback,
//   * every unsupported path returns a typed fallback with zero dispatch count (never "exercise"),
//   * the priority rule disables common fast paths when the wire path is requested.
#include "ops/wire_normal_launch.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace wn = vegajet::wire_normal;
namespace fskv = vegajet::flash_decode_splitkv;

namespace {

int g_failures = 0;

#define CHECK(condition)                                                                       \
    do {                                                                                       \
        if (!(condition)) {                                                                    \
            std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__,           \
                         #condition);                                                          \
            ++g_failures;                                                                      \
        }                                                                                      \
    } while (0)

#define CHECK_NEAR(value, expected, tolerance)                                                 \
    do {                                                                                       \
        const double check_value = static_cast<double>(value);                                 \
        const double check_expected = static_cast<double>(expected);                           \
        if (!(std::fabs(check_value - check_expected) <= (tolerance))) {                       \
            std::fprintf(stderr,                                                               \
                         "CHECK_NEAR failed at %s:%d: %.9g vs %.9g (tol %.3g)\n", __FILE__,    \
                         __LINE__, check_value, check_expected,                                \
                         static_cast<double>(tolerance));                                      \
            ++g_failures;                                                                      \
        }                                                                                      \
    } while (0)

// ---- independent half conversion (different algorithm than the production header) ---------
uint16_t ref_half_bits(float value) {
    if (value == 0.0f) return std::signbit(value) ? 0x8000u : 0x0000u;
    const bool negative = std::signbit(value);
    const double magnitude = std::fabs(static_cast<double>(value));
    int exponent = 0;
    const double fraction = std::frexp(magnitude, &exponent); // magnitude = fraction * 2^exponent
    int half_exponent = exponent + 14; // 1.x * 2^(exponent-1) -> half exponent
    if (half_exponent >= 31) return static_cast<uint16_t>((negative ? 0x8000u : 0u) | 0x7C00u);
    if (half_exponent <= 0) {
        const double scaled = std::ldexp(magnitude, 24);
        const long quantized = std::lrint(scaled); // current rounding mode: nearest-even
        if (quantized >= 1024) { // rounded up into the smallest normal
            return static_cast<uint16_t>((negative ? 0x8000u : 0u) | (1u << 10));
        }
        return static_cast<uint16_t>((negative ? 0x8000u : 0u) | static_cast<uint16_t>(quantized));
    }
    const double significand = std::ldexp(magnitude, -(half_exponent - 15));
    long mantissa = std::lrint((significand - 1.0) * 1024.0);
    if (mantissa == 1024) {
        mantissa = 0;
        ++half_exponent;
        if (half_exponent >= 31)
            return static_cast<uint16_t>((negative ? 0x8000u : 0u) | 0x7C00u);
    }
    (void)fraction;
    return static_cast<uint16_t>((negative ? 0x8000u : 0u) |
                                 (static_cast<uint16_t>(half_exponent) << 10) |
                                 static_cast<uint16_t>(mantissa));
}

float ref_half_value(uint16_t bits) {
    const bool negative = (bits & 0x8000u) != 0u;
    const int exponent = (bits >> 10) & 0x1F;
    const int mantissa = bits & 0x3FF;
    double magnitude;
    if (exponent == 0) magnitude = std::ldexp(static_cast<double>(mantissa), -24);
    else if (exponent == 31) magnitude = mantissa == 0 ? std::numeric_limits<double>::infinity()
                                                       : std::numeric_limits<double>::quiet_NaN();
    else magnitude = std::ldexp(static_cast<double>(1024 + mantissa), exponent - 25);
    return static_cast<float>(negative ? -magnitude : magnitude);
}

// ---- deterministic LCG -------------------------------------------------------------------
struct Lcg {
    uint64_t state;
    explicit Lcg(uint64_t seed) : state(seed) {}
    uint32_t next() {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint32_t>(state >> 33);
    }
    float uniform(float low, float high) {
        return low + (high - low) * (static_cast<float>(next()) / 4294967296.0f);
    }
};

// ---- independent native-layout reference --------------------------------------------------
// Active engine: 8 FP16 scales at bytes 0..15, 256 int8 at bytes 16..271. Scale = amax/127.
void ref_native_row(const float* values, uint8_t* row) {
    std::memset(row, 0, wn::kNativeKvRowBytes);
    for (uint32_t block = 0; block < wn::kBlocksPerHead; ++block) {
        float amax = 0.0f;
        for (uint32_t i = 0; i < 32u; ++i) {
            const float a = std::fabs(values[block * 32u + i]);
            if (a > amax) amax = a;
        }
        const float scale = amax * (1.0f / 127.0f);
        const float inverse = amax > 0.0f ? (127.0f / amax) : 0.0f;
        const uint16_t bits = ref_half_bits(scale);
        row[2u * block] = static_cast<uint8_t>(bits & 0xFFu);
        row[2u * block + 1u] = static_cast<uint8_t>(bits >> 8);
        for (uint32_t i = 0; i < 32u; ++i) {
            int q = static_cast<int>(std::nearbyintf(values[block * 32u + i] * inverse));
            if (q < -127) q = -127;
            if (q > 127) q = 127;
            row[16u + block * 32u + i] = static_cast<uint8_t>(static_cast<int8_t>(q));
        }
    }
}

float ref_native_value(const uint8_t* row, uint32_t element) {
    const uint32_t block = element >> 5;
    const uint32_t sub = element & 31u;
    const uint16_t bits =
        static_cast<uint16_t>(row[2u * block] | (row[2u * block + 1u] << 8));
    return ref_half_value(bits) *
           static_cast<float>(static_cast<int8_t>(row[16u + block * 32u + sub]));
}

// Independently written Q8/Q4 block encoders. Not shared with ops/wire_normal_launch.hpp.
void ref_q8_block(const float* values, uint8_t* block) {
    float amax = 0.0f;
    for (uint32_t i = 0; i < 32u; ++i) amax = std::fmax(amax, std::fabs(values[i]));
    const float scale = amax * (1.0f / 127.0f);
    const float inverse = amax > 0.0f ? (127.0f / amax) : 0.0f;
    std::memset(block, 0, 48u);
    const uint16_t bits = ref_half_bits(scale);
    block[0] = static_cast<uint8_t>(bits & 0xFFu);
    block[1] = static_cast<uint8_t>(bits >> 8);
    for (uint32_t i = 0; i < 32u; ++i) {
        int q = static_cast<int>(std::nearbyintf(values[i] * inverse));
        if (q < -127) q = -127;
        if (q > 127) q = 127;
        block[16u + i] = static_cast<uint8_t>(static_cast<int8_t>(q));
    }
}

void ref_q4_block(const float* values, uint8_t* block) {
    float amax = 0.0f;
    float signed_max = 0.0f;
    for (uint32_t i = 0; i < 32u; ++i) {
        const float a = std::fabs(values[i]);
        // Same deterministic tie rule as the production header and device reduction.
        if (a > amax || (a == amax && values[i] > signed_max)) {
            amax = a;
            signed_max = values[i];
        }
    }
    const float d = signed_max == 0.0f ? 0.0f : (signed_max / -8.0f);
    const uint16_t bits = ref_half_bits(d);
    const float half_d = ref_half_value(bits);
    const float inverse = half_d != 0.0f ? (1.0f / half_d) : 0.0f;
    uint8_t nibbles[32];
    for (uint32_t i = 0; i < 32u; ++i) {
        int q = static_cast<int>(std::nearbyintf(values[i] * inverse));
        if (q < -8) q = -8;
        if (q > 7) q = 7;
        nibbles[i] = static_cast<uint8_t>(q + 8);
    }
    std::memset(block, 0, 32u);
    block[0] = static_cast<uint8_t>(bits & 0xFFu);
    block[1] = static_cast<uint8_t>(bits >> 8);
    for (uint32_t j = 0; j < 16u; ++j)
        block[16u + j] = static_cast<uint8_t>(nibbles[j] | (nibbles[j + 16u] << 4));
}

float wire_value(const uint8_t* wire, wn::WireFormat format, uint32_t head, uint32_t element) {
    const uint32_t block_bytes = wn::wire_block_bytes(format);
    const uint32_t head_bytes = wn::wire_head_bytes(format);
    const uint8_t* block = wire + head * head_bytes + (element >> 5) * block_bytes;
    const uint16_t bits = static_cast<uint16_t>(block[0] | (block[1] << 8));
    const float scale = ref_half_value(bits);
    if (format == wn::WireFormat::Q8) {
        const int8_t q = static_cast<int8_t>(block[16u + (element & 31u)]);
        return scale * static_cast<float>(q);
    }
    const uint8_t packed = block[16u + (element & 15u)];
    const uint32_t nibble = (element & 16u) != 0u ? (packed >> 4) : (packed & 15u);
    return scale * static_cast<float>(static_cast<int>(nibble) - 8);
}

// ---- producers under test ----------------------------------------------------------------
void produce_q4_from_native(const uint8_t* native, uint8_t* wire) {
    for (uint32_t block = 0; block < wn::kBlocksPerHead; ++block) {
        float values[32];
        for (uint32_t i = 0; i < 32u; ++i) values[i] = ref_native_value(native, block * 32u + i);
        ref_q4_block(values, wire + block * 32u);
    }
}

// ---- causal consumer (reads only the produced wire bytes) --------------------------------
double direct_window(const uint8_t* q_wire, const uint8_t* k_wire, const uint8_t* v_wire,
                     wn::WireFormat format, uint32_t row, uint32_t head, uint32_t past,
                     uint32_t m, uint32_t dim) {
    const uint32_t visible = past + m + 1u;
    double maximum = -std::numeric_limits<double>::infinity();
    std::vector<double> scores(visible, 0.0);
    for (uint32_t t = 0; t < visible; ++t) {
        double score = 0.0;
        for (uint32_t d = 0; d < dim; ++d) {
            const uint32_t q_block = (row * wn::kHeads + head) * wn::kBlocksPerHead + (d >> 5);
            const uint32_t kv_block = t * wn::kBlocksPerHead + (d >> 5);
            const uint8_t* qb = q_wire + q_block * wn::wire_block_bytes(format);
            const uint8_t* kb = k_wire + kv_block * wn::wire_block_bytes(format);
            const float qs = ref_half_value(static_cast<uint16_t>(qb[0] | (qb[1] << 8)));
            const float ks = ref_half_value(static_cast<uint16_t>(kb[0] | (kb[1] << 8)));
            if (format == wn::WireFormat::Q8) {
                const int8_t qv = static_cast<int8_t>(qb[16u + (d & 31u)]);
                const int8_t kv = static_cast<int8_t>(kb[16u + (d & 31u)]);
                score += static_cast<double>(qv) * static_cast<double>(kv) *
                         static_cast<double>(qs) * static_cast<double>(ks);
            } else {
                const uint8_t qp = qb[16u + (d & 15u)];
                const uint8_t kp = kb[16u + (d & 15u)];
                const int qv = static_cast<int>((d & 16u) ? (qp >> 4) : (qp & 15u)) - 8;
                const int kv = static_cast<int>((d & 16u) ? (kp >> 4) : (kp & 15u)) - 8;
                score += static_cast<double>(qv) * static_cast<double>(kv) *
                         static_cast<double>(qs) * static_cast<double>(ks);
            }
        }
        score *= static_cast<double>(fskv::kDefaultAttentionScale);
        scores[t] = score;
        maximum = std::fmax(maximum, score);
    }
    double denominator = 0.0;
    double output = 0.0;
    for (uint32_t t = 0; t < visible; ++t)
        denominator += std::exp(scores[t] - maximum);
    for (uint32_t d = 0; d < dim; ++d) {
        double numerator = 0.0;
        for (uint32_t t = 0; t < visible; ++t) {
            const uint32_t kv_block = t * wn::kBlocksPerHead + (d >> 5);
            const uint8_t* vb = v_wire + kv_block * wn::wire_block_bytes(format);
            const float vs = ref_half_value(static_cast<uint16_t>(vb[0] | (vb[1] << 8)));
            double value;
            if (format == wn::WireFormat::Q8) {
                value = static_cast<double>(static_cast<int8_t>(vb[16u + (d & 31u)])) * vs;
            } else {
                const uint8_t packed = vb[16u + (d & 15u)];
                value = static_cast<double>(static_cast<int>((d & 16u) ? (packed >> 4)
                                                                     : (packed & 15u)) - 8) * vs;
            }
            numerator += std::exp(scores[t] - maximum) * value;
        }
        output += numerator / denominator;
    }
    return output / dim;
}

// Partition + merge CPU model of the sealed kernel (only for the consumer test).
void sealed_cpu(const uint8_t* q_wire, const uint8_t* k_wire, const uint8_t* v_wire,
                wn::WireFormat format, uint32_t rows, uint32_t past, uint32_t splits,
                uint32_t dim, std::vector<float>& output) {
    const uint32_t cache_rows = past + rows;
    std::vector<float> partial(static_cast<size_t>(rows) * wn::kHeads * splits * dim, 0.0f);
    std::vector<float> lse(static_cast<size_t>(rows) * wn::kHeads * splits,
                           -std::numeric_limits<float>::infinity());
    for (uint32_t m = 0; m < rows; ++m) {
        for (uint32_t head = 0; head < wn::kHeads; ++head) {
            for (uint32_t split = 0; split < splits; ++split) {
                const uint32_t begin = static_cast<uint32_t>(
                    static_cast<uint64_t>(cache_rows) * split / splits);
                const uint32_t end = static_cast<uint32_t>(
                    static_cast<uint64_t>(cache_rows) * (split + 1u) / splits);
                const uint32_t causal_end = past + m + 1u;
                const uint32_t stop = end < causal_end ? end : causal_end;
                float maximum = -std::numeric_limits<float>::infinity();
                float denominator = 0.0f;
                std::vector<float> accumulator(dim, 0.0f);
                for (uint32_t t = begin; t < stop; ++t) {
                    float score = 0.0f;
                    for (uint32_t d = 0; d < dim; ++d) {
                        const uint32_t q_block =
                            (m * wn::kHeads + head) * wn::kBlocksPerHead + (d >> 5);
                        const uint32_t kv_block = t * wn::kBlocksPerHead + (d >> 5);
                        const uint8_t* qb = q_wire + q_block * wn::wire_block_bytes(format);
                        const uint8_t* kb = k_wire + kv_block * wn::wire_block_bytes(format);
                        const float qs = ref_half_value(
                            static_cast<uint16_t>(qb[0] | (qb[1] << 8)));
                        const float ks = ref_half_value(
                            static_cast<uint16_t>(kb[0] | (kb[1] << 8)));
                        if (format == wn::WireFormat::Q8) {
                            score += static_cast<float>(static_cast<int8_t>(qb[16u + (d & 31u)])) *
                                     static_cast<float>(static_cast<int8_t>(kb[16u + (d & 31u)])) *
                                     qs * ks;
                        } else {
                            const uint8_t qp = qb[16u + (d & 15u)];
                            const uint8_t kp = kb[16u + (d & 15u)];
                            const int qv = static_cast<int>((d & 16u) ? (qp >> 4) : (qp & 15u)) - 8;
                            const int kv = static_cast<int>((d & 16u) ? (kp >> 4) : (kp & 15u)) - 8;
                            score += static_cast<float>(qv) * static_cast<float>(kv) * qs * ks;
                        }
                    }
                    score *= fskv::kDefaultAttentionScale;
                    const float next = std::fmax(maximum, score);
                    const float alpha = std::exp(maximum - next);
                    const float beta = std::exp(score - next);
                    denominator = denominator * alpha + beta;
                    maximum = next;
                    for (uint32_t d = 0; d < dim; ++d) {
                        const uint32_t kv_block = t * wn::kBlocksPerHead + (d >> 5);
                        const uint8_t* vb = v_wire + kv_block * wn::wire_block_bytes(format);
                        const float vs = ref_half_value(
                            static_cast<uint16_t>(vb[0] | (vb[1] << 8)));
                        float value;
                        if (format == wn::WireFormat::Q8) {
                            value = static_cast<float>(static_cast<int8_t>(vb[16u + (d & 31u)])) * vs;
                        } else {
                            const uint8_t packed = vb[16u + (d & 15u)];
                            value = static_cast<float>(static_cast<int>(
                                        (d & 16u) ? (packed >> 4) : (packed & 15u)) - 8) * vs;
                        }
                        accumulator[d] = beta * value + accumulator[d] * alpha;
                    }
                }
                const float reciprocal = denominator > 0.0f ? 1.0f / denominator : 0.0f;
                const size_t row = (static_cast<size_t>(m) * wn::kHeads + head) * splits + split;
                for (uint32_t d = 0; d < dim; ++d)
                    partial[row * dim + d] = accumulator[d] * reciprocal;
                if (denominator > 0.0f)
                    lse[row] = maximum + std::log(denominator);
            }
        }
    }
    output.assign(static_cast<size_t>(rows) * wn::kHeads * dim, 0.0f);
    for (uint32_t m = 0; m < rows; ++m) {
        for (uint32_t head = 0; head < wn::kHeads; ++head) {
            float maximum = -std::numeric_limits<float>::infinity();
            for (uint32_t split = 0; split < splits; ++split)
                maximum = std::fmax(maximum, lse[(static_cast<size_t>(m) * wn::kHeads + head) *
                                                     splits + split]);
            float denominator = 0.0f;
            for (uint32_t split = 0; split < splits; ++split) {
                const float x = lse[(static_cast<size_t>(m) * wn::kHeads + head) * splits + split];
                if (x != -std::numeric_limits<float>::infinity())
                    denominator += std::exp(x - maximum);
            }
            for (uint32_t d = 0; d < dim; ++d) {
                float numerator = 0.0f;
                for (uint32_t split = 0; split < splits; ++split) {
                    const size_t row =
                        (static_cast<size_t>(m) * wn::kHeads + head) * splits + split;
                    const float x = lse[row];
                    if (x == -std::numeric_limits<float>::infinity()) continue;
                    numerator += std::exp(x - maximum) * partial[row * dim + d];
                }
                output[(static_cast<size_t>(m) * wn::kHeads + head) * dim + d] =
                    denominator > 0.0f ? numerator / denominator : 0.0f;
            }
        }
    }
}

// ---- test sections -----------------------------------------------------------------------
void test_half_conversion() {
    const float values[] = {0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 1.0f / 3.0f, 65504.0f, 6.1035156e-5f,
                            1.2e-6f, 123.456f, -789.01f};
    for (float value : values) {
        CHECK(wn::f32_to_f16_bits(value) == ref_half_bits(value));
        CHECK_NEAR(wn::f16_bits_to_f32(wn::f32_to_f16_bits(value)), ref_half_value(ref_half_bits(value)),
                   0.0);
    }
    Lcg rng(0x1234u);
    for (int i = 0; i < 2000; ++i) {
        const float value = rng.uniform(-65000.0f, 65000.0f);
        CHECK(wn::f32_to_f16_bits(value) == ref_half_bits(value));
    }
}

void test_q8_producer_matches_independent_reference() {
    constexpr uint32_t rows = 8u;
    std::vector<float> q(static_cast<size_t>(rows) * wn::kHeads * wn::kHeadDim);
    Lcg rng(0x88u);
    for (float& value : q) value = rng.uniform(-6.0f, 6.0f);
    std::vector<uint8_t> wire(static_cast<size_t>(rows) * wn::q_wire_row_bytes(wn::WireFormat::Q8));
    wn::produce_q_wire(q.data(), rows, wn::WireFormat::Q8, wire.data());
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t head = 0; head < wn::kHeads; ++head) {
            const float* source = q.data() + (static_cast<size_t>(row) * wn::kHeads + head) * wn::kHeadDim;
            const uint8_t* target =
                wire.data() + (static_cast<size_t>(row) * wn::kHeads + head) * wn::wire_head_bytes(
                                  wn::WireFormat::Q8);
            for (uint32_t block = 0; block < wn::kBlocksPerHead; ++block) {
                uint8_t expected[48];
                ref_q8_block(source + block * 32u, expected);
                CHECK(std::memcmp(expected, target + block * 48u, 48u) == 0);
                const float scale = ref_half_value(
                    static_cast<uint16_t>(expected[0] | (expected[1] << 8)));
                for (uint32_t i = 0; i < 32u; ++i) {
                    const int8_t code = static_cast<int8_t>(expected[16u + i]);
                    const double decoded = static_cast<double>(scale) * code;
                    const double original = source[block * 32u + i];
                    CHECK(std::fabs(decoded - original) <=
                          std::fabs(static_cast<double>(scale)) * 0.75 + 1e-6);
                }
            }
        }
    }
}

void test_q4_producer_matches_independent_reference() {
    constexpr uint32_t rows = 8u;
    std::vector<float> q(static_cast<size_t>(rows) * wn::kHeads * wn::kHeadDim);
    Lcg rng(0x44u);
    for (float& value : q) value = rng.uniform(-6.0f, 6.0f);
    std::vector<uint8_t> wire(static_cast<size_t>(rows) * wn::q_wire_row_bytes(wn::WireFormat::Q4));
    wn::produce_q_wire(q.data(), rows, wn::WireFormat::Q4, wire.data());
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t head = 0; head < wn::kHeads; ++head) {
            const float* source = q.data() + (static_cast<size_t>(row) * wn::kHeads + head) * wn::kHeadDim;
            const uint8_t* target =
                wire.data() + (static_cast<size_t>(row) * wn::kHeads + head) * wn::wire_head_bytes(
                                  wn::WireFormat::Q4);
            for (uint32_t block = 0; block < wn::kBlocksPerHead; ++block) {
                uint8_t expected[32];
                ref_q4_block(source + block * 32u, expected);
                CHECK(std::memcmp(expected, target + block * 32u, 32u) == 0);
                const float half_d = ref_half_value(
                    static_cast<uint16_t>(expected[0] | (expected[1] << 8)));
                for (uint32_t i = 0; i < 32u; ++i) {
                    const uint8_t packed = expected[16u + (i & 15u)];
                    const int qv = static_cast<int>((i & 16u) ? (packed >> 4) : (packed & 15u)) - 8;
                    const double decoded = static_cast<double>(half_d) * qv;
                    CHECK(std::fabs(decoded - source[block * 32u + i]) <=
                          std::fabs(static_cast<double>(half_d)) * 0.5 + 1e-6);
                }
            }
        }
    }
    // All-zero block: d = -0, every nibble is 8, every decoded value is zero.
    float zeros[32] = {};
    uint8_t block[32];
    wn::quantize_q4_block(zeros, block);
    CHECK((block[0] | (block[1] << 8)) == 0x0000u);
    for (uint32_t j = 0; j < 16u; ++j) CHECK(block[16u + j] == 0x88u);
}

void test_q8_kv_adaptor_is_lossless() {
    constexpr uint32_t rows = 37u;
    std::vector<uint8_t> native(static_cast<size_t>(rows) * wn::kNativeKvRowBytes);
    std::vector<float> values(256);
    Lcg rng(0x55u);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t i = 0; i < 256u; ++i) values[i] = rng.uniform(-3.0f, 3.0f);
        ref_native_row(values.data(), native.data() + static_cast<size_t>(row) * wn::kNativeKvRowBytes);
    }
    std::vector<uint8_t> wire(static_cast<size_t>(rows) * wn::wire_row_bytes(wn::WireFormat::Q8));
    std::vector<uint8_t> wire_v(wire.size());
    wn::produce_kv_wire(native.data(), native.data(), 0u, rows, wn::WireFormat::Q8,
                        wire.data(), wire_v.data());
    for (uint32_t row = 0; row < rows; ++row) {
        const uint8_t* source = native.data() + static_cast<size_t>(row) * wn::kNativeKvRowBytes;
        const uint8_t* target = wire.data() + static_cast<size_t>(row) * wn::wire_row_bytes(
                                                     wn::WireFormat::Q8);
        for (uint32_t block = 0; block < wn::kBlocksPerHead; ++block) {
            const uint8_t* block_source = source + 16u + block * 32u;
            const uint8_t* block_target = target + block * 48u;
            CHECK(block_target[0] == source[2u * block]);
            CHECK(block_target[1] == source[2u * block + 1u]);
            for (uint32_t i = 2u; i < 16u; ++i) CHECK(block_target[i] == 0u);
            CHECK(std::memcmp(block_source, block_target + 16u, 32u) == 0);
            for (uint32_t i = 0; i < 32u; ++i)
                CHECK(ref_native_value(source, block * 32u + i) ==
                      wire_value(target, wn::WireFormat::Q8, 0u, block * 32u + i));
        }
    }
}

void test_q4_kv_adaptor_matches_reference() {
    constexpr uint32_t rows = 29u;
    std::vector<uint8_t> native(static_cast<size_t>(rows) * wn::kNativeKvRowBytes);
    std::vector<float> values(256);
    Lcg rng(0x99u);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t i = 0; i < 256u; ++i) values[i] = rng.uniform(-2.0f, 2.0f);
        ref_native_row(values.data(), native.data() + static_cast<size_t>(row) * wn::kNativeKvRowBytes);
    }
    std::vector<uint8_t> wire(static_cast<size_t>(rows) * wn::wire_row_bytes(wn::WireFormat::Q4));
    std::vector<uint8_t> wire_v(wire.size());
    wn::produce_kv_wire(native.data(), native.data(), 0u, rows, wn::WireFormat::Q4,
                        wire.data(), wire_v.data());
    for (uint32_t row = 0; row < rows; ++row) {
        const uint8_t* source = native.data() + static_cast<size_t>(row) * wn::kNativeKvRowBytes;
        const uint8_t* target = wire.data() + static_cast<size_t>(row) * wn::wire_row_bytes(
                                                     wn::WireFormat::Q4);
        uint8_t expected[256];
        produce_q4_from_native(source, expected);
        CHECK(std::memcmp(expected, target, 256u) == 0);
        for (uint32_t block = 0; block < wn::kBlocksPerHead; ++block) {
            const float half_d = ref_half_value(
                static_cast<uint16_t>(target[block * 32u] | (target[block * 32u + 1u] << 8)));
            for (uint32_t i = 0; i < 32u; ++i) {
                const double decoded = wire_value(target, wn::WireFormat::Q4, 0u, block * 32u + i);
                const double original = ref_native_value(source, block * 32u + i);
                CHECK(std::fabs(decoded - original) <=
                      std::fabs(static_cast<double>(half_d)) * 1.5 + 1e-6);
            }
        }
    }
}

void test_cache_updates_and_rollback() {
    wn::CacheRange range = wn::plan_cache_update(0u, 30u, 1u, 128u);
    CHECK(range.valid && range.begin == 0u && range.end == 31u);
    range = wn::plan_cache_update(30u, 30u, 1u, 128u);
    CHECK(range.valid && range.begin == 30u && range.end == 31u);
    // Speculative draft rows 31..34 were produced, then the batch only committed to row 30.
    range = wn::plan_cache_update(35u, 30u, 5u, 128u);
    CHECK(range.valid && range.begin == 30u && range.end == 35u);
    range = wn::plan_cache_update(40u, 30u, 8u, 48u);
    CHECK(range.valid && range.begin == 30u && range.end == 38u);
    CHECK(!wn::plan_cache_update(0u, 46u, 5u, 48u).valid);
    CHECK(!wn::plan_cache_update(0u, 0u, 2u, 48u).valid);
    CHECK(!wn::plan_cache_update(0u, 0u, 9u, 48u).valid);
    CHECK(wn::plan_cache_update(31u, 30u, 5u, 128u).begin == 30u);

    // Producer plan: only the uncommitted or never-produced rows are refreshed.
    wn::ProducerPlan plan = wn::plan_producers(1u, 31u, 30u, 30u, 128u, wn::WireFormat::Q8);
    CHECK(plan.valid && plan.kv_begin == 30u && plan.kv_end == 31u);
    CHECK(plan.q_grid_x == wn::kHeads);
    CHECK(plan.q_bytes == wn::q_wire_row_bytes(wn::WireFormat::Q8));
    CHECK(plan.k_bytes == static_cast<uint64_t>(31u) * wn::wire_row_bytes(wn::WireFormat::Q8));
    CHECK(plan.q_entry == wn::kQ8QEntry && plan.kv_entry == wn::kQ8KvEntry);
    plan = wn::plan_producers(8u, 40u, 35u, 32u, 128u, wn::WireFormat::Q4);
    CHECK(plan.valid && plan.kv_begin == 32u && plan.kv_end == 40u);
    CHECK(plan.q_entry == wn::kQ4QEntry && plan.kv_entry == wn::kQ4KvEntry);
    CHECK(!wn::plan_producers(1u, 129u, 0u, 0u, 128u, wn::WireFormat::Q8).valid);
}

void test_abi_images() {
    CHECK(sizeof(wn::QProducerArgs) == 24u);
    CHECK(sizeof(wn::KvProducerArgs) == 48u);
    CHECK(sizeof(wn::EpilogueArgs) == 40u);
    CHECK(offsetof(wn::QProducerArgs, q_f32) == 0u);
    CHECK(offsetof(wn::QProducerArgs, q_wire) == 8u);
    CHECK(offsetof(wn::QProducerArgs, rows) == 16u);
    CHECK(offsetof(wn::QProducerArgs, heads) == 20u);
    CHECK(offsetof(wn::KvProducerArgs, src_k) == 0u);
    CHECK(offsetof(wn::KvProducerArgs, src_v) == 8u);
    CHECK(offsetof(wn::KvProducerArgs, dst_k) == 16u);
    CHECK(offsetof(wn::KvProducerArgs, dst_v) == 24u);
    CHECK(offsetof(wn::KvProducerArgs, begin) == 32u);
    CHECK(offsetof(wn::KvProducerArgs, end) == 36u);
    CHECK(offsetof(wn::KvProducerArgs, kv_heads) == 40u);
    CHECK(offsetof(wn::EpilogueArgs, attention) == 0u);
    CHECK(offsetof(wn::EpilogueArgs, gate) == 8u);
    CHECK(offsetof(wn::EpilogueArgs, out_q8) == 16u);
    CHECK(offsetof(wn::EpilogueArgs, out_ds) == 24u);
    CHECK(offsetof(wn::EpilogueArgs, rows) == 32u);
    CHECK(offsetof(wn::EpilogueArgs, heads) == 36u);
    CHECK(wn::kQ8BlockBytes == 48u && wn::kQ4BlockBytes == 32u);
    CHECK(wn::kNativeKvRowBytes == 272u);
    CHECK(wn::wire_row_bytes(wn::WireFormat::Q8) == 384u);
    CHECK(wn::wire_row_bytes(wn::WireFormat::Q4) == 256u);
}

void test_selection_and_fallbacks() {
    wn::Policy policy = wn::production_policy(true, false);
    wn::Capacities capacities;
    capacities.cache_rows = 8192u;
    capacities.q_bytes = 8192u;
    capacities.scratch_bytes = 16ull * 1024ull * 1024ull;

    wn::Attempt attempt =
        wn::decide(fskv::active_shape(1u, 8191u), wn::WireFormat::Q8, 8191u, capacities, true,
                   policy);
    CHECK(attempt.requested && attempt.engaged);
    CHECK(attempt.format == wn::WireFormat::Q8);
    CHECK(attempt.splits == 64u);
    CHECK(attempt.dispatch_total == 5u); // KV refresh + Q + partition + merge + epilogue
    CHECK(attempt.partition_grid_x == 64u && attempt.partition_grid_y == 1u);
    CHECK(attempt.merge_grid_x == 16u && attempt.merge_grid_y == 1u);
    CHECK(attempt.epilogue_grid_x == 16u);
    CHECK(attempt.scratch_bytes > 0u && attempt.scratch_bytes <= capacities.scratch_bytes);
    CHECK(attempt.common_fast_path_blocked);

    // The rows the pair will read are always refreshed from the active cache. After a speculative
    // rollback (produced > pos0) the stale draft rows must be re-adapted, so a high-water mark is
    // clamped to pos0 instead of being trusted.
    attempt = wn::decide(fskv::active_shape(1u, 8190u), wn::WireFormat::Q8, 8191u, capacities,
                         true, policy);
    CHECK(attempt.engaged && attempt.kv_dispatches == 1u && attempt.dispatch_total == 5u);
    CHECK(attempt.kv_begin == 8190u && attempt.kv_end == 8191u);

    // Q4 is opt-in.
    attempt = wn::decide(fskv::active_shape(1u, 8191u), wn::WireFormat::Q4, 8191u, capacities,
                         true, policy);
    CHECK(!attempt.engaged && attempt.fallback == wn::Fallback::Format);
    CHECK(attempt.dispatch_total == 0u);
    policy.allow_q4 = true;
    attempt = wn::decide(fskv::active_shape(1u, 8191u), wn::WireFormat::Q4, 8191u, capacities,
                         true, policy);
    CHECK(attempt.engaged && attempt.format == wn::WireFormat::Q4);
    CHECK(attempt.partition_grid_x == 64u);

    // Not requested is not a fallback "exercise".
    attempt = wn::decide(fskv::active_shape(1u, 8191u), wn::WireFormat::Q8, 8191u, capacities,
                         true, wn::production_policy(false, false));
    CHECK(!attempt.requested && !attempt.engaged && attempt.fallback == wn::Fallback::NotRequested);

    // Shape / storage rejections.
    attempt = wn::decide(fskv::active_shape(2u, 1000u), wn::WireFormat::Q8, 0u, capacities, true,
                         policy);
    CHECK(!attempt.engaged && attempt.fallback == wn::Fallback::BatchRows);
    attempt = wn::decide(fskv::Shape{1u, 16u, 1u, 256u, 100u, 102u}, wn::WireFormat::Q8, 0u,
                         capacities, true, policy);
    CHECK(!attempt.engaged && attempt.fallback == wn::Fallback::CacheRows);
    attempt = wn::decide(fskv::active_shape(1u, 300u), wn::WireFormat::Q8, 0u, capacities, true,
                         policy);
    CHECK(!attempt.engaged && attempt.fallback == wn::Fallback::SmallContext);
    policy.exercise_short_context = true;
    attempt = wn::decide(fskv::active_shape(1u, 300u), wn::WireFormat::Q8, 0u, capacities, true,
                         policy);
    CHECK(attempt.engaged && attempt.splits >= 2u);
    policy.exercise_short_context = false;

    // Missing producer module and a scratch budget below the split floor.
    attempt = wn::decide(fskv::active_shape(1u, 8191u), wn::WireFormat::Q8, 8191u, capacities,
                         false, policy);
    CHECK(!attempt.engaged && attempt.fallback == wn::Fallback::ProducerMissing);
    wn::Capacities tiny_scratch = capacities;
    tiny_scratch.scratch_bytes = 1024u;
    attempt = wn::decide(fskv::active_shape(1u, 8191u), wn::WireFormat::Q8, 8191u, tiny_scratch,
                         true, policy);
    CHECK(!attempt.engaged && attempt.fallback == wn::Fallback::BufferCapacity);
    wn::Capacities tiny_cache = capacities;
    tiny_cache.cache_rows = 1024u;
    attempt = wn::decide(fskv::active_shape(1u, 8191u), wn::WireFormat::Q8, 8191u, tiny_cache,
                         true, policy);
    CHECK(!attempt.engaged && attempt.fallback == wn::Fallback::ProducerRange);

    // Priority: a requested wire path disables every common fast path.
    wn::PathFlags flags;
    flags.wire_requested = true;
    flags.common_fused_m1 = true;
    flags.common_auto_splitkv = true;
    flags.common_spec_flash = true;
    const wn::PathChoice choice = wn::choose_path(flags);
    CHECK(choice.wire && !choice.common_fused_m1 && !choice.common_auto_splitkv &&
          !choice.common_spec_flash);
    flags.wire_requested = false;
    const wn::PathChoice fallback_choice = wn::choose_path(flags);
    CHECK(!fallback_choice.wire && fallback_choice.common_fused_m1 &&
          fallback_choice.common_auto_splitkv && fallback_choice.common_spec_flash);
}

void test_consumer_causal_tails_and_empty_splits() {
    for (wn::WireFormat format : {wn::WireFormat::Q8, wn::WireFormat::Q4}) {
        for (uint32_t rows : {1u, 5u, 8u}) {
            for (uint32_t cache_rows : {37u, 256u}) {
                const uint32_t past = cache_rows - rows;
                Lcg rng(0xABC000u + static_cast<uint32_t>(format) * 7u + rows * 13u + cache_rows);
                std::vector<float> q(static_cast<size_t>(rows) * wn::kHeads * wn::kHeadDim);
                for (float& value : q) value = rng.uniform(-2.0f, 2.0f);
                std::vector<float> qkv(256u);
                std::vector<uint8_t> native_k(static_cast<size_t>(cache_rows) * wn::kNativeKvRowBytes);
                std::vector<uint8_t> native_v(native_k.size());
                for (uint32_t t = 0; t < cache_rows; ++t) {
                    for (uint32_t i = 0; i < 256u; ++i) qkv[i] = rng.uniform(-3.0f, 3.0f);
                    ref_native_row(qkv.data(), native_k.data() + static_cast<size_t>(t) * 272u);
                    for (uint32_t i = 0; i < 256u; ++i) qkv[i] = rng.uniform(-3.0f, 3.0f);
                    ref_native_row(qkv.data(), native_v.data() + static_cast<size_t>(t) * 272u);
                }
                std::vector<uint8_t> q_wire(static_cast<size_t>(rows) *
                                            wn::q_wire_row_bytes(format));
                std::vector<uint8_t> k_wire(static_cast<size_t>(cache_rows) *
                                            wn::wire_row_bytes(format));
                std::vector<uint8_t> v_wire(k_wire.size());
                wn::produce_q_wire(q.data(), rows, format, q_wire.data());
                wn::produce_kv_wire(native_k.data(), native_v.data(), 0u, cache_rows, format,
                                    k_wire.data(), v_wire.data());
                for (uint32_t splits : {2u, 17u, 64u}) {
                    std::vector<float> sealed;
                    sealed_cpu(q_wire.data(), k_wire.data(), v_wire.data(), format, rows, past,
                               splits, 256u, sealed);
                    CHECK(sealed.size() == static_cast<size_t>(rows) * wn::kHeads * 256u);
                    for (uint32_t m = 0; m < rows; ++m) {
                        for (uint32_t head = 0; head < wn::kHeads; ++head) {
                            const double direct = direct_window(
                                q_wire.data(), k_wire.data(), v_wire.data(), format, m, head,
                                past, m, 256u);
                            double mean = 0.0;
                            for (uint32_t d = 0; d < 256u; ++d)
                                mean += sealed[(static_cast<size_t>(m) * wn::kHeads + head) * 256u + d];
                            mean /= 256.0;
                            CHECK_NEAR(mean, direct, 1e-3);
                        }
                    }
                }
            }
        }
    }
}

void test_epilogue_matches_active_formula() {
    constexpr uint32_t rows = 8u;
    Lcg rng(0xE01u);
    std::vector<float> attention(static_cast<size_t>(rows) * wn::kHeads * wn::kHeadDim);
    std::vector<float> gate(attention.size());
    for (float& value : attention) value = rng.uniform(-3.0f, 3.0f);
    for (float& value : gate) value = rng.uniform(-8.0f, 8.0f);
    std::vector<int8_t> out_q8(attention.size());
    std::vector<float> out_ds(static_cast<size_t>(rows) * wn::kHeads * wn::kBlocksPerHead * 2u);
    wn::epilogue_reference(attention.data(), gate.data(), rows, out_q8.data(), out_ds.data());
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t head = 0; head < wn::kHeads; ++head) {
            const size_t base = (static_cast<size_t>(row) * wn::kHeads + head) * 256u;
            for (uint32_t block = 0; block < wn::kBlocksPerHead; ++block) {
                const size_t pair =
                    ((static_cast<size_t>(row) * wn::kHeads + head) * wn::kBlocksPerHead + block) * 2u;
                float amax = 0.0f;
                float sum = 0.0f;
                for (uint32_t i = 0; i < 32u; ++i) {
                    const float y = attention[base + block * 32u + i] /
                                    (1.0f + std::exp(-gate[base + block * 32u + i]));
                    amax = std::fmax(amax, std::fabs(y));
                    sum += y;
                }
                const float scale = amax * (1.0f / 127.0f);
                CHECK_NEAR(out_ds[pair], scale, 1e-7);
                CHECK_NEAR(out_ds[pair + 1u], sum, 2e-6);
                const float inverse = amax > 0.0f ? (127.0f / amax) : 0.0f;
                for (uint32_t i = 0; i < 32u; ++i) {
                    const float y = attention[base + block * 32u + i] /
                                    (1.0f + std::exp(-gate[base + block * 32u + i]));
                    int expected = static_cast<int>(std::nearbyint(y * inverse));
                    if (expected < -127) expected = -127;
                    if (expected > 127) expected = 127;
                    CHECK(out_q8[base + block * 32u + i] == expected);
                }
            }
        }
    }
}

} // namespace

int main() {
    std::fprintf(stderr, "test_wire_normal: half, producers, cache, selection, consumer, epilogue\n");
    test_half_conversion();
    test_abi_images();
    test_q8_producer_matches_independent_reference();
    test_q4_producer_matches_independent_reference();
    test_q8_kv_adaptor_is_lossless();
    test_q4_kv_adaptor_matches_reference();
    test_cache_updates_and_rollback();
    test_selection_and_fallbacks();
    test_consumer_causal_tails_and_empty_splits();
    test_epilogue_matches_active_formula();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_wire_normal: FAIL (%d check failures)\n", g_failures);
        return 1;
    }
    std::printf("test_wire_normal: PASS\n");
    return 0;
}
