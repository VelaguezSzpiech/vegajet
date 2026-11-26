// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU contract coverage for ops/flash_decode_splitkv_launch.hpp: the host launch interface of
// the sealed Q8_0/Q4_0 Split-KV FlashDecoding pair.
//
// What this test deliberately covers, beyond the sealed kernel math test
// (tests/test_flash_decode_splitkv.cpp):
//   * exact kernel argument images, declared kernarg byte counts and launch geometry;
//   * the scratch byte plan (sizes, 256 B alignment, ordering, uint64 overflow);
//   * safe selection / fallback invariants over an adversarial shape/policy matrix;
//   * the active-attention mapping T = pos0 + M, past = pos0, stop = min(end, past+m+1);
//   * partition tails driven by the device formula (floor(T*s/S)), empty splits, causal
//     masking with M = 1 and 5..8, Q8 and Q4 wire rows, and the exact merge arithmetic
//     (including an all-empty window) against an independent FP64 two-pass reference.
//
// It is a CPU contract model, not a device execution or parity claim.
#include "ops/flash_decode_splitkv_launch.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace ld = vegajet::flash_decode_splitkv;

namespace {

constexpr unsigned kDim = ld::kHeadDim;
using Vector = std::array<float, kDim>;
using DoubleVector = std::array<double, kDim>;

[[noreturn]] void fail(const char* what, uint64_t a = 0, uint64_t b = 0, uint64_t c = 0,
                       double error = 0.0) {
    std::fprintf(stderr, "FAIL: %s (a=%llu b=%llu c=%llu error=%.9g)\n", what,
                 static_cast<unsigned long long>(a), static_cast<unsigned long long>(b),
                 static_cast<unsigned long long>(c), error);
    std::exit(1);
}

void expect(bool condition, const char* what) {
    if (!condition) fail(what);
}

// splitmix64: independent of the LCG used by the sealed math test.
struct Rng {
    uint64_t state;
    explicit Rng(uint64_t seed) : state(seed) {}
    uint64_t next() {
        state += 0x9E3779B97F4A7C15ull;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    uint32_t below(uint32_t n) { return static_cast<uint32_t>(next() >> 32) % n; }
};

float half_scale(const uint8_t* block) {
    const uint16_t bits = static_cast<uint16_t>(block[0]) |
                          static_cast<uint16_t>(static_cast<uint16_t>(block[1]) << 8);
    const unsigned exponent = (bits >> 10) & 31u, mantissa = bits & 1023u;
    const float value = exponent ? std::ldexp(static_cast<float>(1024u + mantissa),
                                              static_cast<int>(exponent) - 25)
                                 : std::ldexp(static_cast<float>(mantissa), -24);
    return (bits & 0x8000u) ? -value : value;
}

struct Wire {
    std::array<uint8_t, ld::kBlocksPerHead * ld::kQ8BlockBytes> bytes{};
};

unsigned wire_stride(ld::WireFormat format) {
    const unsigned stride = ld::block_bytes(format);
    expect(stride == 32u || stride == 48u, "wire stride");
    return stride;
}

void fill_wire(Wire& wire, ld::WireFormat format, Rng& rng, unsigned pattern) {
    const unsigned stride = wire_stride(format);
    const unsigned payload = format == ld::WireFormat::Q8 ? 32u : 16u;
    std::fill(wire.bytes.begin(), wire.bytes.end(), static_cast<uint8_t>(0xCC));
    for (unsigned block = 0; block < ld::kBlocksPerHead; ++block) {
        uint8_t* p = wire.bytes.data() + block * stride;
        uint16_t scale = 0;
        if (pattern != 5u) { // pattern 5 exercises a zero FP16 scale on block 0
            // Keep the same magnitude class as the sealed math test so the FP64 reference
            // tolerance measures association, not float dequantisation noise.
            const unsigned base = format == ld::WireFormat::Q8 ? 6u : 8u;
            const unsigned exponent = base + rng.below(3u);
            const unsigned mantissa = 1u + rng.below(1023u);
            scale = static_cast<uint16_t>((exponent << 10) | mantissa);
        }
        p[0] = static_cast<uint8_t>(scale);
        p[1] = static_cast<uint8_t>(scale >> 8);
        for (unsigned j = 2; j < 16; ++j) p[j] = 0xA5; // ignored header padding
        for (unsigned j = 0; j < payload; ++j) {
            switch (pattern) {
                case 0: p[16 + j] = static_cast<uint8_t>(rng.next()); break;
                case 1: p[16 + j] = 0x00; break;
                case 2: p[16 + j] = 0x7F; break;
                case 3: p[16 + j] = 0x80; break;
                case 4: p[16 + j] = (j & 1u) ? 0x88 : 0x77; break;
                case 5: p[16 + j] = 0xFF; break;
                default: p[16 + j] = static_cast<uint8_t>((j * 37u + block * 11u) & 0xFFu); break;
            }
        }
    }
}

Vector dequantize(const Wire& wire, ld::WireFormat format) {
    Vector out{};
    const unsigned stride = wire_stride(format);
    for (unsigned d = 0; d < kDim; ++d) {
        const uint8_t* block = wire.bytes.data() + (d / 32u) * stride;
        const unsigned j = d % 32u;
        int value = 0;
        if (format == ld::WireFormat::Q8) {
            value = static_cast<int>(static_cast<int8_t>(block[16 + j]));
        } else {
            const unsigned byte = block[16 + (j & 15u)];
            value = static_cast<int>((byte >> (j >= 16u ? 4u : 0u)) & 15u) - 8;
        }
        out[d] = static_cast<float>(value) * half_scale(block);
    }
    return out;
}

uint32_t load32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | static_cast<uint32_t>(p[1]) << 8 |
           static_cast<uint32_t>(p[2]) << 16 | static_cast<uint32_t>(p[3]) << 24;
}

int signed_field(uint32_t word, unsigned shift, unsigned bits) {
    const unsigned mask = (1u << bits) - 1u;
    const int field = static_cast<int>((word >> shift) & mask);
    return (field & (1 << (bits - 1u))) ? field - (1 << bits) : field;
}

// One packed 32-lane dot for a 4-byte word: int8 for Q8, native biased nibbles XOR 8 for Q4.
int packed_dot(uint32_t a, uint32_t b, ld::WireFormat format) {
    int sum = 0;
    if (format == ld::WireFormat::Q4) {
        a ^= 0x88888888u;
        b ^= 0x88888888u;
        for (unsigned shift = 0; shift < 32; shift += 4)
            sum += signed_field(a, shift, 4) * signed_field(b, shift, 4);
    } else {
        for (unsigned shift = 0; shift < 32; shift += 8)
            sum += signed_field(a, shift, 8) * signed_field(b, shift, 8);
    }
    return sum;
}

// Device order: lanes 0..7 own one 32-element block, lanes 8..15 are zero, then the 16-lane
// xor tree (8,4,2,1) and the attention scale.
float packed_score(const Wire& q, const Wire& k, ld::WireFormat format, float attention_scale) {
    const unsigned stride = wire_stride(format);
    const unsigned words = format == ld::WireFormat::Q8 ? 8u : 4u;
    float lanes[16] = {};
    for (unsigned block = 0; block < ld::kBlocksPerHead; ++block) {
        const uint8_t* qp = q.bytes.data() + block * stride;
        const uint8_t* kp = k.bytes.data() + block * stride;
        int dot = 0;
        for (unsigned word = 0; word < words; ++word)
            dot += packed_dot(load32(qp + 16 + word * 4), load32(kp + 16 + word * 4), format);
        lanes[block] = (static_cast<float>(dot) * half_scale(qp)) * half_scale(kp);
    }
    for (unsigned delta = 8; delta; delta >>= 1)
        for (unsigned lane = 0; lane < delta; ++lane) lanes[lane] += lanes[lane + delta];
    return lanes[0] * attention_scale;
}

struct Ref {
    Vector out{};
    float lse = -INFINITY;
};

// Independent two-pass FP64 softmax over dequantized K/V; no packed dot or online helper.
Ref reference(const Vector& q, const std::vector<Vector>& keys, const std::vector<Vector>& values,
              uint32_t visible, float attention_scale) {
    Ref result;
    if (visible == 0) return result;
    std::vector<double> scores(visible);
    double peak = -std::numeric_limits<double>::infinity();
    for (uint32_t t = 0; t < visible; ++t) {
        double dot = 0.0;
        for (unsigned d = 0; d < kDim; ++d) dot += static_cast<double>(q[d]) * keys[t][d];
        scores[t] = dot * static_cast<double>(attention_scale);
        peak = std::max(peak, scores[t]);
    }
    double denominator = 0.0;
    DoubleVector acc{};
    for (uint32_t t = 0; t < visible; ++t) {
        const double weight = std::exp(scores[t] - peak);
        denominator += weight;
        for (unsigned d = 0; d < kDim; ++d) acc[d] += weight * static_cast<double>(values[t][d]);
    }
    for (unsigned d = 0; d < kDim; ++d) result.out[d] = static_cast<float>(acc[d] / denominator);
    result.lse = static_cast<float>(peak + std::log(denominator));
    return result;
}

struct Partial {
    Vector out{};
    float lse = -INFINITY;
};

// Exact device partition arithmetic: per-token online update in the device's lane/dimension
// mapping, then the per-split normalisation, then the LSE.
Partial model_partition(const std::vector<float>& scores, const std::vector<Vector>& values,
                        uint32_t cache_rows, uint32_t past, uint32_t row, uint32_t splits,
                        uint32_t split) {
    Partial part;
    float maximum = -INFINITY, denominator = 0.0f;
    float acc[16][16] = {};
    const uint32_t begin = ld::split_begin(cache_rows, split, splits);
    const uint32_t stop = ld::split_stop(cache_rows, split, splits,
                                         static_cast<uint64_t>(past) + row + 1u);
    for (uint32_t t = begin; t < stop; ++t) {
        const float next = std::fmax(maximum, scores[t]);
        const float alpha = std::exp(maximum - next);
        const float beta = std::exp(scores[t] - next);
        denominator = denominator * alpha + beta;
        maximum = next;
        for (unsigned lane = 0; lane < 16; ++lane) {
            for (unsigned j = 0; j < 16; ++j) {
                const unsigned d = lane + 16u * j;
                acc[lane][j] = std::fma(beta, values[t][d], acc[lane][j] * alpha);
            }
        }
    }
    const float reciprocal = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    for (unsigned lane = 0; lane < 16; ++lane)
        for (unsigned j = 0; j < 16; ++j) part.out[lane + 16u * j] = acc[lane][j] * reciprocal;
    part.lse = denominator > 0.0f ? maximum + std::log(denominator) : -INFINITY;
    return part;
}

struct Merged {
    Vector out{};
    float lse = -INFINITY;
};

// Exact device merge arithmetic, including the -inf weight of an empty split and the
// all-empty window (output 0, LSE -inf).
Merged model_merge(const std::vector<Partial>& parts) {
    Merged merged;
    float maximum = -INFINITY;
    for (const Partial& part : parts) maximum = std::fmax(maximum, part.lse);
    float numerator[kDim] = {};
    float denominator = 0.0f;
    for (const Partial& part : parts) {
        const float x = part.lse;
        const float weight = x == -INFINITY ? 0.0f : std::exp(x - maximum);
        for (unsigned d = 0; d < kDim; ++d)
            numerator[d] = std::fma(weight, part.out[d], numerator[d]);
        denominator += weight;
    }
    for (unsigned d = 0; d < kDim; ++d)
        merged.out[d] = denominator > 0.0f ? numerator[d] / denominator : 0.0f;
    merged.lse = denominator > 0.0f ? maximum + std::log(denominator) : -INFINITY;
    return merged;
}

bool close(float got, float want, double atol, double rtol) {
    return std::fabs(static_cast<double>(got) - static_cast<double>(want)) <=
           atol + rtol * std::fabs(static_cast<double>(want));
}

struct Stats {
    uint64_t cases = 0;
    uint64_t coordinates = 0;
    uint64_t differing = 0;
    uint64_t tail_cases = 0;
    uint64_t empty_partials = 0;
    uint64_t masked_tokens = 0;
    uint64_t all_empty_merges = 0;
    uint64_t q8_cases = 0;
    uint64_t q4_cases = 0;
    float max_out_error = 0.0f;
    float max_lse_error = 0.0f;
    float max_score_error = 0.0f;
};

constexpr float kAtol = 2e-4f;
constexpr float kRtol = 2e-4f;

void compare_partition_merge(const char* tag, const std::vector<Partial>& parts, const Ref& want,
                             uint32_t cache_rows, uint32_t splits, uint32_t row, uint32_t head,
                             Stats& stats) {
    const Merged got = model_merge(parts);
    ++stats.cases;
    for (const Partial& part : parts) {
        if (part.lse == -INFINITY) ++stats.empty_partials;
    }
    if (splits > 1 && (cache_rows % splits) != 0u) ++stats.tail_cases;
    for (unsigned d = 0; d < kDim; ++d) {
        const float error = std::fabs(got.out[d] - want.out[d]);
        stats.max_out_error = std::max(stats.max_out_error, error);
        ++stats.coordinates;
        stats.differing += got.out[d] != want.out[d];
        if (!close(got.out[d], want.out[d], kAtol, kRtol))
            fail(tag, cache_rows, splits, (static_cast<uint64_t>(head) << 32) | row, error);
    }
    const float lse_error = std::fabs(got.lse - want.lse);
    if (want.lse == -INFINITY) {
        if (got.lse != -INFINITY) fail("empty LSE", cache_rows, splits, head, got.lse);
    } else {
        stats.max_lse_error = std::max(stats.max_lse_error, lse_error);
        if (!close(got.lse, want.lse, kAtol, kRtol))
            fail("merge LSE", cache_rows, splits, head, lse_error);
    }
}

// ---- launch-contract checks ----------------------------------------------
void check_interface() {
    static_assert(sizeof(ld::PartitionArgs) == ld::kPartitionKernargBytes, "partition image");
    static_assert(sizeof(ld::MergeArgs) == 40u, "merge natural size");
    static_assert(ld::kMergeKernargBytes == 36u, "merge declared kernarg size");
    static_assert(ld::kPartitionThreads == 256u && ld::kMergeThreads == 256u, "block size");

    const ld::LaunchGeometry p = ld::partition_geometry(17u, 5u);
    expect(p.grid_x == 17u && p.grid_y == 5u && p.grid_z == 1u, "partition grid");
    expect(p.block_x == 256u && p.block_y == 1u && p.block_z == 1u, "partition block");
    const ld::LaunchGeometry m = ld::merge_geometry(5u);
    expect(m.grid_x == 16u && m.grid_y == 5u && m.grid_z == 1u, "merge grid");
    expect(m.block_x == 256u && m.block_y == 1u && m.block_z == 1u, "merge block");

    const ld::Shape decode = ld::active_shape(1u, 4095u);
    expect(decode.cache_rows == 4096u && decode.pos0 == 4095u, "active shape decode");
    const ld::Shape verify = ld::active_shape(8u, 32u);
    expect(verify.cache_rows == 40u && verify.pos0 == 32u, "active shape verify");
    expect(ld::head_bytes(ld::WireFormat::Q8) == 384u, "q8 head bytes");
    expect(ld::head_bytes(ld::WireFormat::Q4) == 256u, "q4 head bytes");
}

void check_split_boundaries() {
    const std::array<uint32_t, 11> splits{1, 2, 3, 5, 7, 13, 17, 31, 32, 60, 64};
    for (uint32_t cache_rows = 0; cache_rows <= 600; ++cache_rows) {
        for (uint32_t s : splits) {
            uint32_t covered = 0;
            uint32_t min_len = std::numeric_limits<uint32_t>::max(), max_len = 0;
            uint64_t sum = 0;
            for (uint32_t split = 0; split < s; ++split) {
                const uint32_t begin = ld::split_begin(cache_rows, split, s);
                const uint32_t end = ld::split_end(cache_rows, split, s);
                if (begin != covered || end < begin || end > cache_rows)
                    fail("split coverage", cache_rows, s, split);
                covered = end;
                const uint32_t len = end - begin;
                min_len = std::min(min_len, len);
                max_len = std::max(max_len, len);
                sum += len;
                // Empty splits may fall anywhere when S > T (the device writes zeros/-inf).
                const uint32_t stop = ld::split_stop(cache_rows, split, s, 3u);
                expect(stop == std::min(end, 3u), "split stop clamp");
            }
            expect(covered == cache_rows, "split coverage end");
            expect(sum == cache_rows, "split length sum");
            expect(min_len == cache_rows / s, "min split length");
            expect(max_len == (cache_rows + s - 1u) / s, "max split length");
        }
    }
}

void check_scratch_plan() {
    for (uint64_t rows = 1; rows <= 8; ++rows) {
        for (uint64_t splits = 1; splits <= 64; ++splits) {
            const ld::ScratchPlan p = ld::plan_scratch(rows, splits);
            expect(!p.overflow, "scratch overflow flag");
            expect(p.partial_offset == 0u, "partial offset");
            expect(p.partial_bytes == rows * splits * 16384u, "partial bytes");
            expect(p.lse_bytes == rows * splits * 64u, "lse bytes");
            expect(p.output_bytes == rows * 16384u, "output bytes");
            expect(p.output_lse_bytes == rows * 64u, "output_lse bytes");
            expect(p.lse_offset % ld::kScratchAlign == 0u, "lse alignment");
            expect(p.output_offset % ld::kScratchAlign == 0u, "output alignment");
            expect(p.output_lse_offset % ld::kScratchAlign == 0u, "output_lse alignment");
            expect(p.lse_offset >= p.partial_offset + p.partial_bytes, "lse after partial");
            expect(p.output_offset >= p.lse_offset + p.lse_bytes, "output after lse");
            expect(p.output_lse_offset >= p.output_offset + p.output_bytes,
                   "output_lse after output");
            expect(p.total_bytes >= p.output_lse_offset + p.output_lse_bytes,
                   "total covers output_lse");
            expect(p.total_bytes % ld::kScratchAlign == 0u, "total alignment");
        }
    }
    // Overflow must be reported, never wrapped.
    expect(ld::plan_scratch(1ull << 40, 1ull << 40).overflow, "scratch overflow product");
    expect(ld::plan_scratch(std::numeric_limits<uint64_t>::max(), 2u).overflow,
           "scratch overflow rows");
    expect(ld::plan_scratch(1u, std::numeric_limits<uint64_t>::max()).overflow,
           "scratch overflow splits");
    expect(ld::plan_scratch(0u, 4u).overflow, "scratch rejects M=0");
    expect(ld::plan_scratch(1u, 0u).overflow, "scratch rejects S=0");

    for (ld::WireFormat format : {ld::WireFormat::Q8, ld::WireFormat::Q4}) {
        const uint64_t head = ld::head_bytes(format);
        for (uint64_t rows = 1; rows <= 8; ++rows) {
            for (uint64_t cache_rows : {0ull, 1ull, 7ull, 64ull, 4096ull}) {
                const ld::Shape shape = {static_cast<uint32_t>(rows), ld::kHeads, 1u, kDim,
                                         0u, cache_rows};
                const ld::WirePlan plan = ld::plan_wire(shape, format);
                expect(!plan.overflow, "wire overflow flag");
                expect(plan.q_bytes == rows * ld::kHeads * head, "wire q bytes");
                expect(plan.k_bytes == cache_rows * head, "wire k bytes");
                expect(plan.v_bytes == cache_rows * head, "wire v bytes");
                expect(plan.total_bytes == plan.q_bytes + 2u * plan.k_bytes, "wire total");
            }
        }
    }
    expect(ld::plan_wire(ld::Shape{1u, 16u, 1u, 256u, 0u,
                                   std::numeric_limits<uint64_t>::max()},
                         ld::WireFormat::Q8).overflow,
           "wire overflow");
}

void check_selection() {
    const ld::Shape decode = ld::active_shape(1u, 4095u);   // T = 4096
    ld::Selection sel = ld::select(decode, ld::WireFormat::Q8, ld::WireFormat::Q8,
                                   ld::WireFormat::Q8, ld::kDefaultAttentionScale);
    expect(sel.use_split_kv, "default decode selection engages");
    expect(sel.splits == 64u, "default decode splits");
    expect(sel.min_split_tokens == 64u && sel.max_split_tokens == 64u, "default decode chunks");
    expect(sel.scratch_bytes == ld::plan_scratch(1u, 64u).total_bytes, "default decode scratch");
    expect(sel.fallback == ld::Fallback::None && sel.reason && sel.reason[0], "default reason");

    const ld::Shape verify = ld::active_shape(5u, 2048u);   // T = 2053, rows 5
    sel = ld::select(verify, ld::WireFormat::Q8, ld::WireFormat::Q8, ld::WireFormat::Q8,
                     ld::kDefaultAttentionScale);
    expect(sel.use_split_kv && sel.splits == 64u, "verify selection");
    expect(sel.min_split_tokens == 32u && sel.max_split_tokens == 33u, "verify tail chunks");

    struct Case {
        ld::Shape shape;
        ld::WireFormat q, k, v;
        float scale;
        ld::Policy policy;
        ld::Fallback want;
        uint32_t want_splits;
    };
    const ld::Shape base = ld::active_shape(1u, 4095u);
    ld::Policy tight = ld::Policy{};
    tight.scratch_budget_bytes = ld::plan_scratch(1u, 2u).total_bytes - 1u;
    ld::Policy exact = ld::Policy{};
    exact.scratch_budget_bytes = ld::plan_scratch(1u, 64u).total_bytes;
    ld::Policy half = ld::Policy{};
    half.scratch_budget_bytes = ld::plan_scratch(1u, 32u).total_bytes;
    ld::Policy q4 = ld::Policy{};
    q4.allow_q4 = true;
    ld::Policy one_split = ld::Policy{};
    one_split.max_splits = 1u;
    ld::Policy huge_cap = ld::Policy{};
    huge_cap.max_splits = 1000u;
    ld::Policy zero_min = ld::Policy{};
    zero_min.min_tokens_per_split = 0u;

    ld::Shape bad_heads = base;
    bad_heads.heads = 8u;
    ld::Shape bad_kv = base;
    bad_kv.kv_heads = 2u;
    ld::Shape bad_dim = base;
    bad_dim.head_dim = 128u;
    ld::Shape bad_rows2 = ld::active_shape(2u, 4094u);
    ld::Shape bad_rows9 = ld::active_shape(9u, 4087u);
    ld::Shape bad_cache = base;
    bad_cache.cache_rows += 1u;
    ld::Shape big_index = base;
    big_index.pos0 = 0x7fffffffull;
    big_index.cache_rows = 0x7fffffffull + 1u;
    ld::Shape small = ld::active_shape(1u, 63u);           // T = 64
    ld::Shape tiny = ld::active_shape(1u, 0u);             // T = 1

    const std::array<Case, 18> cases{{
        {base, ld::WireFormat::Q8, ld::WireFormat::Q8, ld::WireFormat::Q8,
         ld::kDefaultAttentionScale, ld::Policy{}, ld::Fallback::None, 64u},
        {base, ld::WireFormat::Q8, ld::WireFormat::Q8, ld::WireFormat::Q8,
         ld::kDefaultAttentionScale, exact, ld::Fallback::None, 64u},
        {base, ld::WireFormat::Q8, ld::WireFormat::Q8, ld::WireFormat::Q8,
         ld::kDefaultAttentionScale, half, ld::Fallback::None, 32u},
        {base, ld::WireFormat::Q8, ld::WireFormat::Q8, ld::WireFormat::Q8,
         ld::kDefaultAttentionScale, tight, ld::Fallback::ScratchBudget, 1u},
        {base, ld::WireFormat::Q8, ld::WireFormat::Q8, ld::WireFormat::Q8,
         ld::kDefaultAttentionScale, huge_cap, ld::Fallback::None, 64u},
        {base, ld::WireFormat::Q8, ld::WireFormat::Q8, ld::WireFormat::Q8,
         ld::kDefaultAttentionScale, zero_min, ld::Fallback::None, 64u},
        {base, ld::WireFormat::Q8, ld::WireFormat::Q8, ld::WireFormat::Q8,
         ld::kDefaultAttentionScale, one_split, ld::Fallback::Policy, 1u},
        {base, ld::WireFormat::Q4, ld::WireFormat::Q4, ld::WireFormat::Q4,
         ld::kDefaultAttentionScale, ld::Policy{}, ld::Fallback::Format, 1u},
        {base, ld::WireFormat::Q4, ld::WireFormat::Q4, ld::WireFormat::Q4,
         ld::kDefaultAttentionScale, q4, ld::Fallback::None, 64u},
        {base, ld::WireFormat::Q8, ld::WireFormat::Q4, ld::WireFormat::Q8,
         ld::kDefaultAttentionScale, q4, ld::Fallback::Format, 1u},
        {base, ld::WireFormat::None, ld::WireFormat::None, ld::WireFormat::None,
         ld::kDefaultAttentionScale, ld::Policy{}, ld::Fallback::Format, 1u},
        {base, ld::WireFormat::Q8, ld::WireFormat::Q8, ld::WireFormat::Q8, 0.0f,
         ld::Policy{}, ld::Fallback::AttentionScale, 1u},
        {base, ld::WireFormat::Q8, ld::WireFormat::Q8, ld::WireFormat::Q8,
         std::numeric_limits<float>::quiet_NaN(), ld::Policy{},
         ld::Fallback::AttentionScale, 1u},
        {bad_heads, ld::WireFormat::Q8, ld::WireFormat::Q8, ld::WireFormat::Q8,
         ld::kDefaultAttentionScale, ld::Policy{}, ld::Fallback::Dimensions, 1u},
        {bad_kv, ld::WireFormat::Q8, ld::WireFormat::Q8, ld::WireFormat::Q8,
         ld::kDefaultAttentionScale, ld::Policy{}, ld::Fallback::Dimensions, 1u},
        {bad_dim, ld::WireFormat::Q8, ld::WireFormat::Q8, ld::WireFormat::Q8,
         ld::kDefaultAttentionScale, ld::Policy{}, ld::Fallback::Dimensions, 1u},
        {bad_rows2, ld::WireFormat::Q8, ld::WireFormat::Q8, ld::WireFormat::Q8,
         ld::kDefaultAttentionScale, ld::Policy{}, ld::Fallback::BatchRows, 1u},
        {bad_rows9, ld::WireFormat::Q8, ld::WireFormat::Q8, ld::WireFormat::Q8,
         ld::kDefaultAttentionScale, ld::Policy{}, ld::Fallback::BatchRows, 1u},
    }};

    for (const Case& test : cases) {
        const ld::Selection got = ld::select(test.shape, test.q, test.k, test.v, test.scale,
                                             test.policy);
        if (got.fallback != test.want)
            fail("selection fallback", static_cast<uint64_t>(got.fallback),
                 static_cast<uint64_t>(test.want));
        if (got.use_split_kv != (test.want == ld::Fallback::None))
            fail("selection engagement", got.use_split_kv,
                 static_cast<uint64_t>(test.want));
        if (got.splits != test.want_splits) fail("selection splits", got.splits, test.want_splits);
        if (!got.reason || !got.reason[0]) fail("selection reason text");
        // Determinism.
        const ld::Selection again = ld::select(test.shape, test.q, test.k, test.v, test.scale,
                                               test.policy);
        expect(again.use_split_kv == got.use_split_kv && again.splits == got.splits &&
               again.fallback == got.fallback && again.scratch_bytes == got.scratch_bytes,
               "selection determinism");
    }

    for (const ld::Shape& shape : {bad_cache, big_index, small, tiny}) {
        const ld::Selection got = ld::select(shape, ld::WireFormat::Q8, ld::WireFormat::Q8,
                                             ld::WireFormat::Q8, ld::kDefaultAttentionScale);
        expect(!got.use_split_kv, "invalid/small shape must fall back");
        expect(got.splits == 1u && got.format == ld::WireFormat::None, "fallback shape");
    }
    // Explicit reasons for the non-obvious rows.
    expect(ld::select(small, ld::WireFormat::Q8, ld::WireFormat::Q8, ld::WireFormat::Q8,
                      ld::kDefaultAttentionScale).fallback == ld::Fallback::SmallContext,
           "small context reason");
    expect(ld::select(tiny, ld::WireFormat::Q8, ld::WireFormat::Q8, ld::WireFormat::Q8,
                      ld::kDefaultAttentionScale).fallback == ld::Fallback::SmallContext,
           "tiny context reason");
    expect(ld::select(bad_cache, ld::WireFormat::Q8, ld::WireFormat::Q8, ld::WireFormat::Q8,
                      ld::kDefaultAttentionScale).fallback == ld::Fallback::CacheRows,
           "cache rows reason");
    expect(ld::select(big_index, ld::WireFormat::Q8, ld::WireFormat::Q8, ld::WireFormat::Q8,
                      ld::kDefaultAttentionScale).fallback == ld::Fallback::CacheRows,
           "signed int range reason");

    // Parallelism floor: S must exceed the 16-head sequential launch to buy anything.
    const ld::Shape floor_shape = ld::active_shape(1u, 511u); // T = 512 -> S = 16
    expect(ld::select(floor_shape, ld::WireFormat::Q8, ld::WireFormat::Q8,
                      ld::WireFormat::Q8, ld::kDefaultAttentionScale).fallback ==
               ld::Fallback::SmallContext,
           "parallelism floor fallback");
    ld::Policy floor_ok = ld::Policy{};
    floor_ok.min_splits = 2u;
    const ld::Selection floor_engages = ld::select(floor_shape, ld::WireFormat::Q8,
                                                   ld::WireFormat::Q8, ld::WireFormat::Q8,
                                                   ld::kDefaultAttentionScale, floor_ok);
    expect(floor_engages.use_split_kv && floor_engages.splits == 16u,
           "floor override engages S=16");
    ld::Policy inconsistent = ld::Policy{};
    inconsistent.min_splits = 32u;
    inconsistent.max_splits = 16u;
    expect(ld::select(base, ld::WireFormat::Q8, ld::WireFormat::Q8, ld::WireFormat::Q8,
                      ld::kDefaultAttentionScale, inconsistent).fallback ==
               ld::Fallback::Policy,
           "min_splits above max_splits");
    ld::Policy zero_floor = ld::Policy{};
    zero_floor.min_splits = 0u; // sanitized to 2
    const ld::Selection zero_floor_sel = ld::select(
        floor_shape, ld::WireFormat::Q8, ld::WireFormat::Q8, ld::WireFormat::Q8,
        ld::kDefaultAttentionScale, zero_floor);
    expect(zero_floor_sel.use_split_kv && zero_floor_sel.splits == 16u,
           "min_splits zero sanitized to 2");
    ld::Policy high_floor = ld::Policy{};
    high_floor.min_splits = 1000u; // sanitized to kMaxSplits
    const ld::Selection high_floor_sel = ld::select(
        base, ld::WireFormat::Q8, ld::WireFormat::Q8, ld::WireFormat::Q8,
        ld::kDefaultAttentionScale, high_floor);
    expect(high_floor_sel.use_split_kv && high_floor_sel.splits == 64u,
           "min_splits above kMaxSplits sanitized");

    // Every policy-driven selection must satisfy the engagement invariants.
    const std::array<uint32_t, 9> totals{0, 1, 2, 63, 64, 65, 255, 256, 4096};
    const std::array<uint32_t, 6> min_tokens{1, 2, 8, 32, 64, 128};
    const std::array<uint32_t, 5> caps{2, 3, 8, 60, 64};
    for (uint32_t rows : {1u, 5u, 8u}) {
        for (uint32_t total : totals) {
            if (total < rows) continue;
            const ld::Shape shape = ld::active_shape(rows, total - rows);
            for (uint32_t min_token : min_tokens) {
                for (uint32_t cap : caps) {
                    ld::Policy policy = ld::Policy{};
                    policy.min_tokens_per_split = min_token;
                    policy.max_splits = cap;
                    policy.min_splits = 2u;
                    policy.min_total_tokens = 0u;
                    const ld::Selection got = ld::select(
                        shape, ld::WireFormat::Q8, ld::WireFormat::Q8, ld::WireFormat::Q8,
                        ld::kDefaultAttentionScale, policy);
                    if (!got.use_split_kv) {
                        expect(got.splits == 1u && got.format == ld::WireFormat::None,
                               "fallback shape");
                        continue;
                    }
                    expect(got.splits >= 2u && got.splits <= std::min(cap, ld::kMaxSplits),
                           "engaged splits range");
                    expect(total / got.splits >= std::max(1u, min_token),
                           "engaged min tokens per split");
                    expect(got.min_split_tokens == total / got.splits, "engaged min chunk");
                    expect(got.max_split_tokens == (total + got.splits - 1u) / got.splits,
                           "engaged max chunk");
                    expect(got.splits >= policy.min_splits, "engaged split floor");
                    expect(got.scratch_bytes == ld::plan_scratch(rows, got.splits).total_bytes,
                           "engaged scratch bytes");
                    expect(got.format == ld::WireFormat::Q8, "engaged format");
                }
            }
        }
    }

    // The launch record must only carry geometry/scratch when selection engaged.
    const ld::Launch fallback_launch = ld::plan_launch(
        base, ld::WireFormat::None, ld::WireFormat::None, ld::WireFormat::None,
        ld::kDefaultAttentionScale);
    expect(!fallback_launch.selection.use_split_kv, "fallback launch");
    expect(fallback_launch.partition.grid_x == 0u && fallback_launch.merge.grid_x == 0u,
           "fallback geometry");
    expect(fallback_launch.scratch.total_bytes == 0u &&
               fallback_launch.wire.total_bytes == 0u,
           "fallback plans");
    const ld::Launch engaged = ld::plan_launch(base, ld::WireFormat::Q8, ld::WireFormat::Q8,
                                               ld::WireFormat::Q8, ld::kDefaultAttentionScale);
    expect(engaged.partition.grid_x == 64u && engaged.partition.grid_y == 1u,
           "engaged partition geometry");
    expect(engaged.merge.grid_x == 16u && engaged.merge.grid_y == 1u, "engaged merge geometry");
    expect(engaged.scratch.total_bytes == engaged.selection.scratch_bytes, "engaged scratch");
    expect(engaged.partition_kernarg_bytes == 56u && engaged.merge_kernarg_bytes == 36u,
           "engaged kernarg sizes");
    expect(engaged.partition_entry && std::strcmp(engaged.partition_entry,
                                                  "vj_flash_decode_q8_partition") == 0,
           "engaged q8 entry point");
    expect(engaged.merge_entry && std::strcmp(engaged.merge_entry,
                                              "vj_flash_decode_merge") == 0,
           "engaged merge entry point");
    expect(fallback_launch.partition_entry == nullptr &&
               fallback_launch.merge_entry == nullptr,
           "fallback entry points");
    const ld::Launch q4_launch = ld::plan_launch(base, ld::WireFormat::Q4, ld::WireFormat::Q4,
                                                 ld::WireFormat::Q4, ld::kDefaultAttentionScale,
                                                 q4);
    expect(q4_launch.partition_entry &&
               std::strcmp(q4_launch.partition_entry,
                           "vj_flash_decode_q4_partition") == 0,
           "engaged q4 entry point");

    // Argument builders: active mapping and int fields.
    const ld::Shape verify2 = ld::active_shape(8u, 32u);
    ld::Policy verify_policy = ld::Policy{};
    verify_policy.min_total_tokens = 1u;
    verify_policy.min_tokens_per_split = 8u;
    verify_policy.min_splits = 2u;
    const ld::Selection verify_sel = ld::select(
        verify2, ld::WireFormat::Q8, ld::WireFormat::Q8, ld::WireFormat::Q8,
        ld::kDefaultAttentionScale, verify_policy);
    expect(verify_sel.use_split_kv && verify_sel.splits == 5u, "verify T=40 selection");
    const ld::PartitionArgs args = ld::partition_args(0x1000u, 0x2000u, 0x3000u, 0x4000u,
                                                      0x5000u, verify2, verify_sel.splits,
                                                      ld::kDefaultAttentionScale);
    expect(args.q == 0x1000u && args.k == 0x2000u && args.v == 0x3000u, "partition arg VAs");
    expect(args.partial == 0x4000u && args.lse == 0x5000u, "partition scratch VAs");
    expect(args.T == 40 && args.past == 32 && args.splits == 5, "partition args mapping");
    expect(args.attention_scale == ld::kDefaultAttentionScale, "partition args scale");
    const ld::MergeArgs margs = ld::merge_args(0x4000u, 0x5000u, 0x6000u, 0x7000u,
                                               verify_sel.splits);
    expect(margs.partial == 0x4000u && margs.lse == 0x5000u, "merge scratch VAs");
    expect(margs.output == 0x6000u && margs.output_lse == 0x7000u, "merge output VAs");
    expect(margs.splits == 5, "merge args splits");
}

// ---- CPU reference over tails and merge ------------------------------------
void run_cpu_reference() {
    Rng rng(0x5EED1234ABCDEF01ull);
    Stats stats;

    const std::array<uint32_t, 5> batches{1, 5, 6, 7, 8};
    const std::array<uint32_t, 18> small_lengths{0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32,
                                                 33, 63, 64, 65, 96};
    const std::array<uint32_t, 12> large_lengths{127, 128, 129, 255, 256, 257, 511, 512, 513,
                                                 1024, 2048, 2049};

    const auto run_one = [&](ld::WireFormat format, uint32_t length, uint32_t batch) {
        // One sharing K/V cache for all 16 Q heads (16:1 GQA), as in the active model.
        std::vector<Wire> key_wire(length), value_wire(length);
        std::vector<Vector> keys(length), values(length);
        for (uint32_t t = 0; t < length; ++t) {
            const unsigned pattern = (t % 8u == 0u) ? 1u + (t / 8u) % 6u : 0u;
            fill_wire(key_wire[t], format, rng, pattern);
            fill_wire(value_wire[t], format, rng, (pattern + 2u) % 7u);
            keys[t] = dequantize(key_wire[t], format);
            values[t] = dequantize(value_wire[t], format);
        }
        const uint32_t past = length - batch; // active verify window: past = T - M
        std::vector<uint32_t> heads;
        if (length <= 96u) {
            for (uint32_t head = 0; head < 16u; ++head) heads.push_back(head);
        } else {
            heads = {0u, 7u, 15u};
        }

        for (uint32_t head : heads) {
            Wire query_wire;
            fill_wire(query_wire, format, rng, head % 7u);
            const Vector query = dequantize(query_wire, format);

            std::vector<float> scores(length);
            for (uint32_t t = 0; t < length; ++t) {
                scores[t] = packed_score(query_wire, key_wire[t], format,
                                         ld::kDefaultAttentionScale);
                if (length <= 96u || t % 97u == 0u) {
                    double scalar = 0.0;
                    for (unsigned d = 0; d < kDim; ++d)
                        scalar += static_cast<double>(query[d]) * keys[t][d];
                    const float error = std::fabs(
                        scores[t] - static_cast<float>(scalar * ld::kDefaultAttentionScale));
                    stats.max_score_error = std::max(stats.max_score_error, error);
                    if (!(error <= kAtol))
                        fail("packed score", length, head, t, error);
                }
            }

            std::vector<uint32_t> splits{1u, 2u, 3u, 5u};
            if (length >= 7u) splits.push_back(7u);
            if (length >= 13u) splits.push_back(13u);
            if (length >= 17u) splits.push_back(17u);
            if (length >= 31u) splits.push_back(31u);
            if (length >= 32u) splits.push_back(32u);
            if (length >= 60u) splits.push_back(60u);
            if (length >= 64u) splits.push_back(64u);
            if (length <= 65u) splits.push_back(length + 1u);
            // Include the exact split count the launch policy would pick for this shape.
            const ld::Shape active = ld::active_shape(batch, past);
            ld::Policy chosen_policy = ld::Policy{};
            chosen_policy.allow_q4 = format == ld::WireFormat::Q4;
            const ld::Selection chosen = ld::select(
                active, format, format, format, ld::kDefaultAttentionScale, chosen_policy);
            if (chosen.use_split_kv) splits.push_back(chosen.splits);
            std::sort(splits.begin(), splits.end());
            splits.erase(std::unique(splits.begin(), splits.end()), splits.end());

            for (uint32_t row = 0; row < batch; ++row) {
                const uint32_t visible =
                    static_cast<uint32_t>(std::min<uint64_t>(length, uint64_t(past) + row + 1u));
                const Ref want =
                    reference(query, keys, values, visible, ld::kDefaultAttentionScale);
                stats.masked_tokens += length - visible;
                for (uint32_t count : splits) {
                    std::vector<Partial> parts(count);
                    uint64_t covered = 0;
                    for (uint32_t split = 0; split < count; ++split) {
                        parts[split] = model_partition(scores, values, length, past, row, count,
                                                       split);
                        const uint32_t begin = ld::split_begin(length, split, count);
                        const uint32_t stop = ld::split_stop(length, split, count,
                                                             uint64_t(past) + row + 1u);
                        // A whole split can sit beyond the causal stop (begin > stop).
                        covered += stop > begin ? stop - begin : 0u;
                    }
                    if (covered != visible) fail("visible coverage", length, count, row);
                    if (format == ld::WireFormat::Q8) ++stats.q8_cases;
                    else ++stats.q4_cases;
                    compare_partition_merge("partition/merge", parts, want, length, count, row,
                                            head, stats);
                }
            }

            // Raw contract: past independent of T (cache longer than the window), M = 1.
            if (length <= 65u) {
                for (uint32_t raw_past : {0u, length / 2u, length > 0u ? length - 1u : 0u}) {
                    const uint32_t visible = static_cast<uint32_t>(
                        std::min<uint64_t>(length, uint64_t(raw_past) + 1u));
                    const Ref want =
                        reference(query, keys, values, visible, ld::kDefaultAttentionScale);
                    stats.masked_tokens += length - visible;
                    for (uint32_t count : {1u, 2u, 3u, 5u, 7u}) {
                        std::vector<Partial> parts(count);
                        for (uint32_t split = 0; split < count; ++split)
                            parts[split] = model_partition(scores, values, length, raw_past, 0u,
                                                           count, split);
                        compare_partition_merge("raw past partition/merge", parts, want, length,
                                                count, 0u, head, stats);
                    }
                }
            }
        }
    };

    for (ld::WireFormat format : {ld::WireFormat::Q8, ld::WireFormat::Q4}) {
        for (uint32_t length : small_lengths)
            for (uint32_t batch : batches)
                if (batch <= length) run_one(format, length, batch);
        for (uint32_t length : large_lengths) {
            if (format == ld::WireFormat::Q4 && length > 513u) continue;
            for (uint32_t batch : {1u, 8u})
                if (batch <= length) run_one(format, length, batch);
        }
    }

    // Explicit all-empty merge (T = 0): every split writes zeros/-inf; output is 0, LSE -inf.
    for (uint32_t count : {1u, 2u, 4u, 64u}) {
        std::vector<Partial> parts(count);
        const Merged merged = model_merge(parts);
        for (unsigned d = 0; d < kDim; ++d)
            if (merged.out[d] != 0.0f) fail("all-empty output", count, d);
        if (merged.lse != -INFINITY) fail("all-empty LSE", count);
        ++stats.all_empty_merges;
    }

    expect(stats.cases > 0u && stats.coordinates > 0u, "coverage non-empty");
    expect(stats.q8_cases > 0u && stats.q4_cases > 0u, "q8/q4 coverage");
    expect(stats.tail_cases > 0u, "tail coverage");
    expect(stats.empty_partials > 0u, "empty partial coverage");
    expect(stats.masked_tokens > 0u, "causal masking coverage");
    expect(stats.all_empty_merges > 0u, "all-empty merge coverage");
    expect(stats.differing > 0u, "device model must differ from the FP64 reference somewhere");

    std::printf("CPU q8/q4 launch contract PASS: cases=%llu coordinates=%llu differing=%llu "
                "q8=%llu q4=%llu tails=%llu empty-partials=%llu masked-tokens=%llu "
                "all-empty-merges=%llu\n",
                static_cast<unsigned long long>(stats.cases),
                static_cast<unsigned long long>(stats.coordinates),
                static_cast<unsigned long long>(stats.differing),
                static_cast<unsigned long long>(stats.q8_cases),
                static_cast<unsigned long long>(stats.q4_cases),
                static_cast<unsigned long long>(stats.tail_cases),
                static_cast<unsigned long long>(stats.empty_partials),
                static_cast<unsigned long long>(stats.masked_tokens),
                static_cast<unsigned long long>(stats.all_empty_merges));
    std::printf("measured max_abs_output=%.9g max_abs_lse=%.9g max_abs_score=%.9g; "
                "atol=%g rtol=%g\n",
                static_cast<double>(stats.max_out_error),
                static_cast<double>(stats.max_lse_error),
                static_cast<double>(stats.max_score_error),
                static_cast<double>(kAtol), static_cast<double>(kRtol));
}

} // namespace

int main() {
    check_interface();
    check_split_boundaries();
    check_scratch_plan();
    check_selection();
    std::printf("launch interface PASS: partition kernarg=%u merge kernarg=%u "
                "partition block=%u merge block=%u max_splits=%u\n",
                ld::kPartitionKernargBytes, ld::kMergeKernargBytes, ld::kPartitionThreads,
                ld::kMergeThreads, ld::kMaxSplits);
    run_cpu_reference();
    std::puts("selection/scratch/geometry invariants and CPU partition/merge tails PASS; "
              "no device parity claim.");
    return 0;
}
