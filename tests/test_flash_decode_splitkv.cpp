// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
constexpr unsigned dim = 256;
using Vector = std::array<float, dim>;
struct Random {
    uint32_t state = 0x853c49e7u;
    uint32_t next() { state = state * 1664525u + 1013904223u; return state; }
};
[[noreturn]] void fail(const char* what, unsigned length, unsigned split, unsigned head, float error) {
    std::fprintf(stderr, "%s: length=%u splits=%u head=%u error=%.9g\n", what, length, split, head, error);
    std::exit(1);
}
// Standalone CPU contract model, not a device execution/parity claim.
// q8 blocks: 48 bytes; q4 blocks: 32 bytes; FP16 scale at 0, payload at 16.
// q4 low/high nibbles encode coordinates j/j+16 with an unsigned bias of 8.
struct alignas(16) Wire { std::array<uint8_t, 384> bytes{}; };
unsigned stride(bool q4) { return q4 ? 32 : 48; }
float half(uint16_t bits) {
    const unsigned e = (bits >> 10) & 31, m = bits & 1023;
    const float value = e ? std::ldexp(float(1024 + m), int(e) - 25)
                          : std::ldexp(float(m), -24);
    return bits & 32768 ? -value : value;
}
float scale(const uint8_t* block) { return half(uint16_t(block[0]) | uint16_t(block[1]) << 8); }
Wire random_wire(Random& rng, bool q4) {
    Wire wire;
    for (unsigned b = 0; b < 8; ++b) {
        uint8_t* p = wire.bytes.data() + b * stride(q4);
        // Random non-power-of-two FP16 scales, rather than exact integer-dot fixtures.
        const uint16_t bits = uint16_t(((q4 ? 10u : 7u) << 10) | (1 + (rng.next() >> 16) % 1023));
        p[0] = uint8_t(bits); p[1] = uint8_t(bits >> 8);
        for (unsigned j = 2; j < 16; ++j) p[j] = 0xa5; // ignored header padding
        for (unsigned j = 0; j < (q4 ? 16u : 32u); ++j) p[16 + j] = uint8_t(rng.next() >> 24);
    }
    return wire;
}
Vector dequantize(const Wire& wire, bool q4) {
    Vector result{};
    for (unsigned d = 0; d < dim; ++d) {
        const uint8_t* p = wire.bytes.data() + (d / 32) * stride(q4);
        const unsigned j = d % 32;
        const int value = q4 ? int((p[16 + j % 16] >> (j >= 16 ? 4 : 0)) & 15) - 8
                            : (p[16 + j] < 128 ? int(p[16 + j]) : int(p[16 + j]) - 256);
        result[d] = float(value) * scale(p);
    }
    return result;
}
uint32_t load32(const uint8_t* p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
int signed_field(uint32_t word, unsigned shift, unsigned bits) {
    const int field = int((word >> shift) & ((1u << bits) - 1));
    return field & (1 << (bits - 1)) ? field - (1 << bits) : field;
}
int packed_dot(uint32_t a, uint32_t b, bool q4) {
    int sum = 0;
    const unsigned bits = q4 ? 4 : 8;
    // XOR changes biased q4 nibbles into signed dot8 operands.
    if (q4) { a ^= 0x88888888u; b ^= 0x88888888u; }
    for (unsigned shift = 0; shift < 32; shift += bits)
        sum += signed_field(a, shift, bits) * signed_field(b, shift, bits);
    return sum;
}
float packed_score(const Wire& q, const Wire& k, bool q4) {
    float lanes[16]{};
    for (unsigned b = 0; b < 8; ++b) {
        const uint8_t* qp = q.bytes.data() + b * stride(q4);
        const uint8_t* kp = k.bytes.data() + b * stride(q4);
        int dot = 0;
        for (unsigned j = 0; j < (q4 ? 16u : 32u); j += 4)
            dot += packed_dot(load32(qp + 16 + j), load32(kp + 16 + j), q4);
        lanes[b] = (float(dot) * scale(qp)) * scale(kp);
    }
    for (unsigned offset = 8; offset; offset /= 2)
        for (unsigned lane = 0; lane < offset; ++lane) lanes[lane] += lanes[lane + offset];
    return lanes[0] * 0.0625f;
}
struct Result { Vector output{}; float lse = -INFINITY; };
// Independent two-pass softmax: no packed dot or online-update helpers.
Result reference(const Vector& query, const std::vector<Vector>& keys,
                 const std::vector<Vector>& values, unsigned visible) {
    Result result;
    if (!visible) return result;
    std::vector<float> scores(visible);
    float peak = -INFINITY;
    for (unsigned t = 0; t < visible; ++t) {
        float dot = 0;
        for (unsigned d = 0; d < dim; ++d) dot = std::fma(query[d], keys[t][d], dot);
        scores[t] = dot * 0.0625f;
        peak = std::max(peak, scores[t]);
    }
    float denominator = 0;
    for (unsigned t = 0; t < visible; ++t) {
        const float weight = std::exp(scores[t] - peak);
        denominator += weight;
        for (unsigned d = 0; d < dim; ++d) result.output[d] = std::fma(weight, values[t][d], result.output[d]);
    }
    for (float& x : result.output) x /= denominator;
    result.lse = peak + std::log(denominator);
    return result;
}
Result partition(const std::vector<float>& scores, const std::vector<Vector>& values,
                 unsigned length, unsigned visible, unsigned splits) {
    std::vector<Result> partial(splits);
    unsigned covered = 0;
    for (unsigned s = 0; s < splits; ++s) {
        const unsigned begin = uint64_t(length) * s / splits;
        const unsigned end = uint64_t(length) * (s + 1) / splits;
        if (begin != covered || end < begin || end > length) fail("split coverage", length, splits, 0, 0);
        covered = end;
        float peak = -INFINITY, sum = 0;
        for (unsigned t = begin; t < end && t < visible; ++t) {
            const float next = std::max(peak, scores[t]);
            const float old_weight = std::exp(peak - next), new_weight = std::exp(scores[t] - next);
            sum = std::fma(sum, old_weight, new_weight);
            for (unsigned d = 0; d < dim; ++d)
                partial[s].output[d] = std::fma(new_weight, values[t][d], partial[s].output[d] * old_weight);
            peak = next;
        }
        if (sum) {
            const float reciprocal = 1.0f / sum;
            for (float& x : partial[s].output) x *= reciprocal;
            partial[s].lse = peak + std::log(sum);
        }
    }
    if (covered != length) fail("split tail", length, splits, 0, 0);
    Result result;
    float peak = -INFINITY, sum = 0;
    for (const Result& p : partial) peak = std::max(peak, p.lse);
    for (const Result& p : partial) if (p.lse != -INFINITY) {
        const float weight = std::exp(p.lse - peak);
        sum += weight;
        for (unsigned d = 0; d < dim; ++d) result.output[d] = std::fma(weight, p.output[d], result.output[d]);
    }
    if (sum) {
        for (float& x : result.output) x /= sum;
        result.lse = peak + std::log(sum);
    }
    return result;
}
} // namespace

int main() {
    const std::array<unsigned, 24> lengths{{0, 1, 7, 15, 16, 17, 31, 32, 33, 65,
        127, 128, 129, 511, 512, 513, 2048, 4095, 4096, 8191, 8192, 16384, 16385, 32768}};
    const std::array<unsigned, 5> batches{{1, 5, 6, 7, 8}};
    Random rng;
    uint64_t cases = 0, coordinates = 0, differing = 0, empty = 0, masked = 0;
    float max_error = 0, max_lse_error = 0, max_score_error = 0;
    std::vector<unsigned> covered_splits;
    // Exhaustive signed endpoints as well as all intermediate packed values.
    for (bool q4 : {false, true}) {
        const unsigned bits = q4 ? 4 : 8, count = 1u << bits;
        for (unsigned a = 0; a < count; ++a) for (unsigned b = 0; b < count; ++b) {
            uint32_t aw = 0, bw = 0;
            for (unsigned shift = 0; shift < 32; shift += bits) { aw |= a << shift; bw |= b << shift; }
            const int av = q4 ? int(a) - 8 : (a < 128 ? int(a) : int(a) - 256);
            const int bv = q4 ? int(b) - 8 : (b < 128 ? int(b) : int(b) - 256);
            if (packed_dot(aw, bw, q4) != av * bv * int(32 / bits)) fail("packed integer dot", 0, 0, 0, 0);
        }
        for (unsigned li = 0; li < lengths.size(); ++li) {
            const unsigned length = lengths[li];
            std::vector<Wire> keys_wire(length);
            std::vector<Vector> keys(length), values(length);
            for (unsigned t = 0; t < length; ++t) {
                keys_wire[t] = random_wire(rng, q4);
                keys[t] = dequantize(keys_wire[t], q4);
                values[t] = dequantize(random_wire(rng, q4), q4);
            }
            // Full M/query/head cross-product for short tails; longer contexts
            // select M cyclically and boundary query/head positions. Reference
            // and packed scores are reused across splits, never recomputed per S.
            for (unsigned bi = 0; bi < batches.size(); ++bi) {
                const bool required = length == 128 || length == 512 || length == 4096 || length == 16384;
                if (!required && length > 65 && bi != li % batches.size()) continue;
                const unsigned batch = batches[bi];
                std::vector<unsigned> splits{1, 4, 15, 32, (60 + batch - 1) / batch};
                if (length <= 65) splits.push_back(length + 1);
                if (length == 17) splits.insert(splits.end(), {7, 16, 32, 64});
                std::sort(splits.begin(), splits.end());
                splits.erase(std::unique(splits.begin(), splits.end()), splits.end());
                for (unsigned s : splits) covered_splits.push_back(s);
                for (unsigned m = 0; m < batch; ++m) {
                    if (length > 65 && m != 0 && m != batch - 1) continue;
                    for (unsigned h = 0; h < 16; ++h) {
                        if (length > 65 && h != 0 && h != 15) continue;
                        const Wire query_wire = random_wire(rng, q4);
                        const Vector query = dequantize(query_wire, q4);
                        std::vector<float> scores(length);
                        for (unsigned t = 0; t < length; ++t) {
                            scores[t] = packed_score(query_wire, keys_wire[t], q4);
                            float scalar = 0;
                            for (unsigned d = 0; d < dim; ++d) scalar = std::fma(query[d], keys[t][d], scalar);
                            const float error = std::fabs(scores[t] - scalar * 0.0625f);
                            max_score_error = std::max(max_score_error, error);
                            if (!(error <= 2e-4f)) fail("packed score", length, 0, h, error);
                        }
                        std::vector<unsigned> pasts{0, length > batch ? length - batch : 0};
                        std::sort(pasts.begin(), pasts.end());
                        pasts.erase(std::unique(pasts.begin(), pasts.end()), pasts.end());
                        for (unsigned past : pasts) {
                            const unsigned visible = std::min(length, past + m + 1);
                            const Result want = reference(query, keys, values, visible);
                            for (unsigned s : splits) {
                                const Result got = partition(scores, values, length, visible, s);
                                ++cases;
                                empty += visible == 0;
                                masked += visible < length;
                                if (!visible) {
                                    if (got.lse != -INFINITY) fail("empty LSE", length, s, h, got.lse);
                                } else {
                                    const float error = std::fabs(got.lse - want.lse);
                                    max_lse_error = std::max(max_lse_error, error);
                                    if (!(error <= 2e-4f + 2e-4f * std::fabs(want.lse)))
                                        fail("LSE", length, s, h, error);
                                }
                                for (unsigned d = 0; d < dim; ++d) {
                                    const float error = std::fabs(got.output[d] - want.output[d]);
                                    max_error = std::max(max_error, error);
                                    ++coordinates;
                                    differing += got.output[d] != want.output[d];
                                    if (!(error <= 2e-4f + 2e-4f * std::fabs(want.output[d])))
                                        fail("attention output", length, s, h, error);
                                    if (!visible && got.output[d] != 0) fail("empty output", length, s, h, error);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if (!differing || !masked || !empty) fail("missing adversarial coverage", 0, 0, 0, 0);
    std::sort(covered_splits.begin(), covered_splits.end());
    covered_splits.erase(std::unique(covered_splits.begin(), covered_splits.end()), covered_splits.end());
    std::printf("CPU q4/q8 split-KV PASS: cases=%llu coordinates=%llu differing=%llu empty=%llu causal-masked=%llu\n",
        static_cast<unsigned long long>(cases), static_cast<unsigned long long>(coordinates),
        static_cast<unsigned long long>(differing), static_cast<unsigned long long>(empty), static_cast<unsigned long long>(masked));
    std::printf("measured max_abs_output=%.9g max_abs_lse=%.9g max_abs_score=%.9g; atol=2e-4 rtol=2e-4\n",
        max_error, max_lse_error, max_score_error);
    std::printf("M=1,5,6,7,8; all 16 head-specific queries for T<=65; boundary m/head selection above; T=");
    for (unsigned n : lengths) std::printf("%u,", n);
    std::printf(" S=");
    for (unsigned s : covered_splits) std::printf("%u,", s);
    std::puts("\nIndependent scalar FP32 dequantized reference vs packed-dot/online/LSE CPU model; no device parity claim.");
}
