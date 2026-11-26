// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// One-layer parity for the SEALED hot-expert cache path:
//   vj_moe_lru_refill_dot8  (IQ2_S / IQ3_XXS -> linear code plane + separate Scale plane)
//   vj_moe_lru_hit_dot4     (signed int8 activations, v_dot4_i32_i8)
//   vj_moe_lru_hit_dot8     (packed signed int4 activations, two exact v_dot8_i32_i4)
// against an INDEPENDENT decode of the pristine GGUF superblocks (core/ggml_iq_tables.hpp, not the
// cache decoder). One layer of the production Gate+Up / Down shapes plus their transposes.
//
// The three kernels are emulated on the host from the descriptors produced by core/moe_lru_sealed.cpp:
// the refill uses the plan's slot offsets and the device grid, the hit uses the plan's
// (ceil(rows/4), jobs) / block 256 geometry with stagger = 0 (canonical K order), and each thread/lane
// walks exactly the elements that launch assigns to it. The expected value is the same FP chain
// (per-lane FMAs in K order, then the descending 64-lane shfl tree) over the pristine IQ decode, so
// the comparison is bitwise, not approximate. CPU only: no HIP device, runtime, model or A/B run.
//
// Both hit kinds must reproduce the same integer dots and the same float result; the wrong-half and
// unsigned-activation probes are negative controls that must be detected.
#include "core/ggml_iq_tables.hpp"
#include "core/moe_lru_sealed.hpp"
#include "ops/kernels/vj_moe_lru_refill_dot8.hiph"
#include "tests/test_common.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace sealed = vegajet::moeslot::sealed;
namespace T = vegajet::ggml_tables;
using Src = vj_refill::Source;
using Scale = vj_refill::Scale;

namespace {

uint32_t bits_of(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

// IEEE-754 binary16 -> binary32, written independently of the cache header's converter.
float half_reference(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 31u, m = h & 1023u, u = 0;
    if (!e) {
        if (!m) {
            u = sign;
        } else {
            int shift = 0;
            while (!(m & 1024u)) { m <<= 1; ++shift; }
            u = sign | (static_cast<uint32_t>(113 - shift) << 23) | ((m & 1023u) << 13);
        }
    } else {
        u = sign | (static_cast<uint32_t>(e == 31u ? 255u : e + 112u) << 23) | (m << 13);
    }
    float f = 0.f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

struct Pristine {
    int integer = 0;
    float scale = 0.f;
};

// Independent decode of one weight of a pristine GGML IQ superblock (256 weights, `p` at its base).
Pristine source_reference(const uint8_t* p, Src type, uint32_t i) {
    uint16_t d = 0;
    std::memcpy(&d, p, sizeof(d));
    const uint32_t b = i / 32u, l = (i % 32u) / 8u, j = i % 8u;
    if (type == Src::iq2_s) {
        const uint32_t index = static_cast<uint32_t>(p[2 + b * 4 + l]) +
            (((static_cast<uint32_t>(p[66 + b]) >> (2u * l)) & 3u) << 8);
        const uint32_t scale = (static_cast<uint32_t>(p[74 + b]) >> (4u * (l / 2u))) & 15u;
        const int mag = static_cast<int>((T::iq2s_grid[index] >> (j * 8u)) & 255u);
        return {((static_cast<uint32_t>(p[34 + b * 4 + l]) >> j) & 1u) ? -mag : mag,
                half_reference(d) * (0.5f + static_cast<float>(scale)) * 0.25f};
    }
    uint32_t aux = 0;
    std::memcpy(&aux, p + 66 + b * 4, sizeof(aux));
    const uint32_t signs = T::ksigns_iq2xs[(aux >> (l * 7u)) & 127u];
    const uint32_t grid = T::iq3xxs_grid[p[2 + b * 8 + l * 2 + j / 4u]];
    const int mag = static_cast<int>((grid >> ((j % 4u) * 8u)) & 255u);
    return {((signs >> j) & 1u) ? -mag : mag,
            half_reference(d) * (0.5f + static_cast<float>(aux >> 28)) * 0.5f};
}

uint32_t rng_state = 0x243f6a88u;

uint32_t next_random() {
    uint32_t s = rng_state;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    rng_state = s;
    return s;
}

int signed_byte(uint32_t w, uint32_t j) {
    const int b = static_cast<int>((w >> (8u * j)) & 255u);
    return b < 128 ? b : b - 256;
}

int signed_nibble(uint32_t w, uint32_t n) {
    const int v = static_cast<int>((w >> (4u * n)) & 15u);
    return v < 8 ? v : v - 16;
}

// v_dot4_i32_i8 over four signed int8 lanes.
int dot4_lanes(uint32_t w, uint32_t x, int acc) {
    for (uint32_t j = 0; j < 4u; ++j) acc += signed_byte(w, j) * signed_byte(x, j);
    return acc;
}

// v_dot8_i32_i4 over eight 4-bit lanes: signed weights, signed or unsigned activations.
int dot8_lanes(uint32_t w, uint32_t x, bool unsigned_x) {
    int acc = 0;
    for (uint32_t n = 0; n < 8u; ++n) {
        const int a = unsigned_x ? static_cast<int>((x >> (4u * n)) & 15u) : signed_nibble(x, n);
        acc += signed_nibble(w, n) * a;
    }
    return acc;
}

int activation_value(uint32_t token, uint32_t k) {
    uint32_t v = (k + 1u) * 0x9e3779b9u ^ (token + 1u) * 0x85ebca6bu;
    v ^= v >> 16;
    v *= 0x7feb352du;
    v ^= v >> 15;
    return static_cast<int>(v % 16u) - 8;
}

float activation_scale(uint32_t token, uint32_t b) {
    return static_cast<float>(3u + ((token * 7u + b * 11u) % 29u)) / 8192.f;
}

Src to_refill(sealed::Source s) {
    return s == sealed::Source::iq2_s ? Src::iq2_s : Src::iq3_xxs;
}

void run_case(uint32_t rows, uint32_t cols, sealed::Source source, const char* label) {
    const Src type = to_refill(source);
    const uint32_t slots = 2u, jobs = 3u;
    const uint32_t nb = cols / 32u;                                  // 32-weight tiles per row
    const uint32_t block_bytes = (type == Src::iq2_s) ? 82u : 98u;   // one GGML IQ superblock
    const uint64_t raw_bytes = sealed::raw_expert_bytes(rows, cols, source);
    VJ_CHECK(raw_bytes != 0);

    // ---- sealed cache layout: linear code rows + separate Scale rows, both from the device helper ----
    const sealed::CacheLayout lay = sealed::cache_layout(rows, cols);
    VJ_CHECK(lay.rows == rows && lay.cols == cols && lay.blocks32 == nb);
    VJ_CHECK(lay.qs_row_stride == vj_refill::stride(cols / 2u));
    VJ_CHECK(lay.scale_row_stride == vj_refill::stride(nb * 4u));
    VJ_CHECK(sealed::qs_slot_offset(lay, 1) == lay.qs_slot_bytes());
    VJ_CHECK(sealed::scale_slot_offset(lay, 1) == lay.scale_slot_bytes());
    VJ_CHECK(sealed::qs_row_offset(lay, 1, 1) == lay.qs_slot_bytes() + lay.qs_row_stride);
    VJ_CHECK(sealed::scale_row_offset(lay, 1, 1) == lay.scale_slot_bytes() + lay.scale_row_stride);

    std::vector<uint32_t> qs(static_cast<size_t>(lay.qs_slot_bytes() * slots / 4u), 0xdeadbeefu);
    std::vector<Scale> ds(static_cast<size_t>(lay.scale_slot_bytes() * slots / 4u),
                          Scale{0xdead, 0xad, 0xde});
    uint8_t* const qs_bytes = reinterpret_cast<uint8_t*>(qs.data());
    uint8_t* const ds_bytes = reinterpret_cast<uint8_t*>(ds.data());

    // ---- pristine GGUF-shaped source for `slots` experts (fp16 d edge cases on every 17th block) ----
    std::vector<uint8_t> raw(static_cast<size_t>(raw_bytes * slots));
    rng_state = 0x71837421u ^ (static_cast<uint32_t>(type) * 0x9e3779b9u) ^ (rows * 131u + cols);
    for (auto& byte : raw) byte = static_cast<uint8_t>(next_random());
    const uint16_t edge[] = {0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3555, 0xb555, 0x3c00, 0x7bff};
    for (size_t blk = 0; blk < raw.size() / block_bytes; ++blk) {
        const uint16_t d = (blk % 17u < 10u)
            ? edge[blk % 17u]
            : static_cast<uint16_t>((next_random() & 0x83ffu) | 0x2800u);
        std::memcpy(raw.data() + blk * block_bytes, &d, sizeof(d));
    }

    // ---- refill: descriptors from core/moe_lru_sealed.cpp, loop shape from the device kernel ----
    sealed::RefillRequest rr;
    rr.rows = rows;
    rr.cols = cols;
    rr.source = source;
    rr.cards = 1;
    rr.tables = {0x10000000ull, 0x20000000ull, 0x30000000ull};
    rr.raw_plane = {0x1000000ull, 0};
    rr.qs_cache = {0x2000000ull, 0};
    rr.ds_cache = {0x3000000ull, 0};
    const vj_refill::Tables tables{T::iq2s_grid, T::iq3xxs_grid, T::ksigns_iq2xs};
    // Non-identity binding: slot 0 holds expert 1, slot 1 holds expert 0.
    const std::array<uint32_t, 2> expert_of_slot{{1u, 0u}};

    const uint64_t tiles = static_cast<uint64_t>(rows) * nb;
    const uint32_t refill_grid = static_cast<uint32_t>((tiles + 255u) / 256u);
    uint64_t elements = 0, element_mismatches = 0, scale_mismatches = 0, padding_writes = 0;
    for (uint32_t slot = 0; slot < slots; ++slot) {
        const sealed::RefillPlan rp = sealed::build_refill_plan(rr, expert_of_slot[slot], slot, slots);
        VJ_CHECK(rp.ok);
        VJ_CHECK(std::string(rp.launch.symbol) == sealed::kRefillSymbol);
        VJ_CHECK(rp.launch.grid_x == refill_grid && rp.launch.grid_y == 1 && rp.launch.grid_z == 1);
        VJ_CHECK(rp.launch.block_x == 256 && rp.launch.kernarg_bytes == sealed::kRefillKernargBytes);
        VJ_CHECK(rp.qs_slot_offset == static_cast<uint64_t>(slot) * lay.qs_slot_bytes());
        VJ_CHECK(rp.scale_slot_offset == static_cast<uint64_t>(slot) * lay.scale_slot_bytes());
        VJ_CHECK(rp.args[0].src ==
                 rr.raw_plane[0] + static_cast<uint64_t>(expert_of_slot[slot]) * raw_bytes);
        VJ_CHECK(rp.args[0].qs == rr.qs_cache[0] + rp.qs_slot_offset);
        VJ_CHECK(rp.args[0].ds == rr.ds_cache[0] + rp.scale_slot_offset);
        VJ_CHECK(rp.args[0].rows == rows && rp.args[0].cols == cols);
        VJ_CHECK(rp.args[0].type == static_cast<uint32_t>(source));

        const uint8_t* const expert_src =
            raw.data() + static_cast<uint64_t>(expert_of_slot[slot]) * raw_bytes;
        for (uint32_t gx = 0; gx < refill_grid; ++gx) {
            for (uint32_t tid = 0; tid < 256u; ++tid) {
                const uint64_t tile = static_cast<uint64_t>(gx) * 256u + tid;
                if (tile >= tiles) continue;
                const uint32_t row = static_cast<uint32_t>(tile / nb);
                const uint32_t b = static_cast<uint32_t>(tile % nb);
                const uint8_t* in =
                    expert_src + (static_cast<uint64_t>(row) * (cols / 256u) + b / 8u) * block_bytes;
                // Device addressing: qs + row * stride(cols/2), ds + row * stride(nb*4), slot base
                // already added by the caller through the plan offsets.
                uint32_t* w = reinterpret_cast<uint32_t*>(qs_bytes + rp.qs_slot_offset +
                                                          static_cast<uint64_t>(row) * lay.qs_row_stride);
                Scale* s = reinterpret_cast<Scale*>(ds_bytes + rp.scale_slot_offset +
                                                    static_cast<uint64_t>(row) * lay.scale_row_stride);
                vj_refill::repack32(in, type, b % 8u, tables, w + b * 4u, s + b);
            }
        }

        // Lossless against the pristine decode, and no write outside the used part of a row.
        for (uint32_t row = 0; row < rows; ++row) {
            const uint8_t* row_src =
                expert_src + static_cast<uint64_t>(row) * (cols / 256u) * block_bytes;
            const uint32_t* w =
                reinterpret_cast<const uint32_t*>(qs_bytes + rp.qs_slot_offset +
                                                  static_cast<uint64_t>(row) * lay.qs_row_stride);
            const Scale* s =
                reinterpret_cast<const Scale*>(ds_bytes + rp.scale_slot_offset +
                                               static_cast<uint64_t>(row) * lay.scale_row_stride);
            for (uint32_t b = 0; b < nb; ++b) {
                const uint8_t* p = row_src + static_cast<uint64_t>(b / 8u) * block_bytes;
                for (uint32_t half = 0; half < 2u; ++half) {
                    const Pristine ref = source_reference(p, type, (b % 8u) * 32u + half * 16u);
                    scale_mismatches +=
                        bits_of(vj_refill::scale_value(s[b], type, half)) != bits_of(ref.scale);
                }
                for (uint32_t e = 0; e < 32u; ++e) {
                    const uint32_t word = w[b * 4u + (e % 16u) / 4u];
                    const uint32_t dec = vj_refill::decode4(
                        (word >> (e < 16u ? 0u : 4u)) & 0x0f0f0f0fu, type);
                    const Pristine ref = source_reference(p, type, (b % 8u) * 32u + e);
                    ++elements;
                    element_mismatches += signed_byte(dec, e % 4u) != ref.integer;
                }
            }
            for (uint32_t j = nb * 4u; j < lay.qs_row_stride / 4u; ++j)
                padding_writes += w[j] != 0xdeadbeefu;
            for (uint32_t j = nb; j < lay.scale_row_stride / 4u; ++j)
                padding_writes += s[j].d != 0xdead || s[j].sub != 0xad || s[j].reserved != 0xde;
        }
    }

    // ---- activations: one array of signed values, packed both ways for the two hit kinds ----
    const uint32_t token_count = 4u;
    std::vector<int> values(static_cast<size_t>(token_count) * cols);
    std::vector<uint32_t> x4(static_cast<size_t>(token_count) * nb * 8u, 0u); // int8 bytes
    std::vector<uint32_t> x8(static_cast<size_t>(token_count) * nb * 4u, 0u); // int4 nibbles
    std::vector<float> xd(static_cast<size_t>(token_count) * nb, 0.f);
    std::array<uint32_t, 16> value_counts{};
    for (uint32_t token = 0; token < token_count; ++token) {
        for (uint32_t k = 0; k < cols; ++k) {
            const int v = activation_value(token, k);
            values[static_cast<size_t>(token) * cols + k] = v;
            ++value_counts[static_cast<size_t>(v + 8)];
        }
        for (uint32_t w = 0; w < nb * 8u; ++w) {
            uint32_t word = 0;
            for (uint32_t c = 0; c < 4u; ++c)
                word |= (static_cast<uint32_t>(values[static_cast<size_t>(token) * cols + w * 4u + c]) &
                         0xffu) << (8u * c);
            x4[static_cast<size_t>(token) * nb * 8u + w] = word;
        }
        for (uint32_t w = 0; w < nb * 4u; ++w) {
            uint32_t word = 0;
            for (uint32_t c = 0; c < 8u; ++c)
                word |= (static_cast<uint32_t>(values[static_cast<size_t>(token) * cols + w * 8u + c]) &
                         0x0fu) << (4u * c);
            x8[static_cast<size_t>(token) * nb * 4u + w] = word;
        }
        for (uint32_t b = 0; b < nb; ++b)
            xd[static_cast<size_t>(token) * nb + b] = activation_scale(token, b);
    }
    for (uint32_t c = 0; c < 16u; ++c) VJ_CHECK(value_counts[c] > 0);

    // ---- hit descriptors: both kinds, same request, production grid ----
    std::vector<uint32_t> slot_map{1u, 0u, 1u};
    std::vector<uint32_t> token_map{3u, 0u, 2u};
    sealed::HitRequest hr;
    hr.rows = rows;
    hr.cols = cols;
    hr.jobs = jobs;
    hr.slot_count = slots;
    hr.source = source;
    hr.cards = 1;
    hr.qs_cache = {0x2000000ull, 0};
    hr.ds_cache = {0x3000000ull, 0};
    hr.x = {0x4000000ull, 0};
    hr.xd = {0x5000000ull, 0};
    hr.slots = {0x6000000ull, 0};
    hr.tokens = {0x7000000ull, 0};
    hr.y = {0x8000000ull, 0};
    hr.host_slots = &slot_map;
    hr.host_tokens = &token_map;
    hr.token_count = token_count;

    uint32_t hit_grid_x = 0;
    for (const sealed::HitVariant variant : {sealed::HitVariant::dot4, sealed::HitVariant::dot8}) {
        const sealed::HitPlan hp = sealed::build_hit_plan(hr, variant);
        VJ_CHECK(hp.ok);
        VJ_CHECK(std::string(hp.launch.symbol) == (variant == sealed::HitVariant::dot4
                                                       ? sealed::kHitDot4Symbol
                                                       : sealed::kHitDot8Symbol));
        VJ_CHECK(hp.launch.grid_y == jobs && hp.launch.grid_z == 1);
        VJ_CHECK(hp.launch.block_x == 256 && hp.launch.block_y == 1);
        VJ_CHECK(hp.launch.kernarg_bytes == sealed::kHitKernargBytes);
        VJ_CHECK(hp.args[0].stagger == 0 && hp.args[0].qs == hr.qs_cache[0]);
        VJ_CHECK(hp.args[0].slots == hr.slots[0] && hp.args[0].tokens == hr.tokens[0]);
        VJ_CHECK(hp.layout.qs_row_stride == lay.qs_row_stride &&
                 hp.layout.scale_row_stride == lay.scale_row_stride);
        hit_grid_x = hp.launch.grid_x;
    }
    VJ_CHECK(hit_grid_x == (rows + 3u) / 4u);

    // The launch maps every row of every job to exactly one (blockIdx.x, threadIdx.x/64) thread.
    std::vector<uint32_t> visits(rows, 0u);
    for (uint32_t gx = 0; gx < hit_grid_x; ++gx) {
        for (uint32_t tid = 0; tid < 256u; ++tid) {
            const uint32_t row = gx * 4u + tid / 64u;
            if (row < rows && (tid % 64u) == 0u) ++visits[row]; // lane zero stores the wave reduction
        }
    }
    for (uint32_t row = 0; row < rows; ++row) VJ_CHECK(visits[row] == 1u);

    // One hit thread's 64-lane row dot, exactly as the sealed kernels spell it (stagger = 0).
    auto probe = [&](sealed::HitVariant variant, bool wrong_half, bool unsigned_act, uint32_t token,
                     uint32_t slot, uint32_t row) -> float {
        const uint64_t r = static_cast<uint64_t>(slot) * rows + row;
        const uint32_t* w = reinterpret_cast<const uint32_t*>(qs_bytes + r * lay.qs_row_stride);
        const Scale* s = reinterpret_cast<const Scale*>(ds_bytes + r * lay.scale_row_stride);
        std::array<float, 64> lane_acc{};
        for (uint32_t lane = 0; lane < 64u; ++lane) {
            float acc = 0.f;
            for (uint32_t b = lane; b < nb; b += 64u) {
                for (uint32_t h = 0; h < 2u; ++h) {
                    const uint32_t hh = wrong_half ? (h ^ 1u) : h;
                    int dp = 0;
                    if (variant == sealed::HitVariant::dot4) {
                        for (uint32_t j = 0; j < 4u; ++j) {
                            const uint32_t dec = vj_refill::decode4(
                                (w[b * 4u + j] >> (4u * hh)) & 0x0f0f0f0fu, type);
                            dp = dot4_lanes(
                                dec,
                                x4[(static_cast<size_t>(token) * nb + b) * 8u + h * 4u + j], dp);
                        }
                    } else {
                        for (uint32_t j = 0; j < 2u; ++j) {
                            uint32_t low = 0, high = 0;
                            for (uint32_t u = 0; u < 2u; ++u) {
                                const uint32_t dec = vj_refill::decode4(
                                    (w[b * 4u + 2u * j + u] >> (4u * hh)) & 0x0f0f0f0fu, type);
                                for (uint32_t c = 0; c < 4u; ++c) {
                                    const int weight = signed_byte(dec, c);
                                    const int lo = static_cast<int>(
                                        (static_cast<unsigned>(weight) + 8u) & 15u) - 8;
                                    const int hi = (weight - lo) / 16;
                                    const uint32_t shift = 16u * u + 4u * c;
                                    low |= (static_cast<uint32_t>(lo) & 15u) << shift;
                                    high |= (static_cast<uint32_t>(hi) & 15u) << shift;
                                }
                            }
                            const uint32_t activ =
                                x8[(static_cast<size_t>(token) * nb + b) * 4u + h * 2u + j];
                            dp += dot8_lanes(low, activ, unsigned_act) +
                                  16 * dot8_lanes(high, activ, unsigned_act);
                        }
                    }
                    const float scale = vj_refill::scale_value(s[b], type, h) *
                        xd[static_cast<size_t>(token) * nb + b];
                    acc = std::fmaf(scale, static_cast<float>(dp), acc);
                }
            }
            lane_acc[lane] = acc;
        }
        for (uint32_t delta = 32; delta; delta /= 2u)
            for (uint32_t lane = 0; lane < delta; ++lane) lane_acc[lane] += lane_acc[lane + delta];
        return lane_acc[0];
    };

    // Same FP chain, but every weight and every scale comes from the pristine decode.
    auto reference = [&](uint32_t token, uint32_t slot, uint32_t row) -> float {
        const uint8_t* row_src = raw.data() +
            static_cast<uint64_t>(expert_of_slot[slot]) * raw_bytes +
            static_cast<uint64_t>(row) * (cols / 256u) * block_bytes;
        std::array<float, 64> lane_acc{};
        for (uint32_t lane = 0; lane < 64u; ++lane) {
            float acc = 0.f;
            for (uint32_t b = lane; b < nb; b += 64u) {
                const uint8_t* p = row_src + static_cast<uint64_t>(b / 8u) * block_bytes;
                for (uint32_t h = 0; h < 2u; ++h) {
                    int dp = 0;
                    for (uint32_t n = 0; n < 16u; ++n) {
                        const uint32_t k = b * 32u + h * 16u + n;
                        dp += source_reference(p, type, (b % 8u) * 32u + h * 16u + n).integer *
                            values[static_cast<size_t>(token) * cols + k];
                    }
                    const float scale =
                        source_reference(p, type, (b % 8u) * 32u + h * 16u).scale *
                        xd[static_cast<size_t>(token) * nb + b];
                    acc = std::fmaf(scale, static_cast<float>(dp), acc);
                }
            }
            lane_acc[lane] = acc;
        }
        for (uint32_t delta = 32; delta; delta /= 2u)
            for (uint32_t lane = 0; lane < delta; ++lane) lane_acc[lane] += lane_acc[lane + delta];
        return lane_acc[0];
    };

    uint64_t outputs = 0, dot4_mismatch = 0, dot8_mismatch = 0, cross_kind_mismatch = 0;
    uint64_t neg_half_dot4 = 0, neg_half_dot8 = 0, neg_unsigned_dot8 = 0;
    for (uint32_t job = 0; job < jobs; ++job) {
        const uint32_t token = token_map[job], slot = slot_map[job];
        VJ_CHECK(slot < slots && token < token_count);
        for (uint32_t row = 0; row < rows; ++row) {
            const float want = reference(token, slot, row);
            const float got4 = probe(sealed::HitVariant::dot4, false, false, token, slot, row);
            const float got8 = probe(sealed::HitVariant::dot8, false, false, token, slot, row);
            ++outputs;
            dot4_mismatch += bits_of(got4) != bits_of(want);
            dot8_mismatch += bits_of(got8) != bits_of(want);
            cross_kind_mismatch += bits_of(got4) != bits_of(got8);
            neg_half_dot4 += bits_of(probe(sealed::HitVariant::dot4, true, false, token, slot, row)) !=
                bits_of(want);
            neg_half_dot8 += bits_of(probe(sealed::HitVariant::dot8, true, false, token, slot, row)) !=
                bits_of(want);
            neg_unsigned_dot8 +=
                bits_of(probe(sealed::HitVariant::dot8, false, true, token, slot, row)) != bits_of(want);
        }
    }

    VJ_CHECK(element_mismatches == 0 && scale_mismatches == 0 && padding_writes == 0);
    VJ_CHECK(dot4_mismatch == 0 && dot8_mismatch == 0 && cross_kind_mismatch == 0);
    VJ_CHECK(neg_half_dot4 > 0 && neg_half_dot8 > 0 && neg_unsigned_dot8 > 0);

    std::printf(
        "sealed refill+hit parity %-18s %s %ux%u: elements=%llu element_mismatches=%llu "
        "scale_mismatches=%llu padding_writes=%llu outputs=%llu dot4_mismatch=%llu dot8_mismatch=%llu "
        "dot4_vs_dot8_mismatch=%llu neg_half_dot4=%llu neg_half_dot8=%llu neg_unsigned_dot8=%llu "
        "qs_stride=%u scale_stride=%u\n",
        label, type == Src::iq2_s ? "IQ2_S" : "IQ3_XXS", rows, cols,
        static_cast<unsigned long long>(elements), static_cast<unsigned long long>(element_mismatches),
        static_cast<unsigned long long>(scale_mismatches),
        static_cast<unsigned long long>(padding_writes), static_cast<unsigned long long>(outputs),
        static_cast<unsigned long long>(dot4_mismatch),
        static_cast<unsigned long long>(dot8_mismatch),
        static_cast<unsigned long long>(cross_kind_mismatch),
        static_cast<unsigned long long>(neg_half_dot4),
        static_cast<unsigned long long>(neg_half_dot8),
        static_cast<unsigned long long>(neg_unsigned_dot8), lay.qs_row_stride, lay.scale_row_stride);
}

} // namespace

int main() {
    // One layer each in the production geometry, plus the transposed shapes so both source types
    // cover 512x3072 and 3072x512.
    run_case(512u, 3072u, sealed::Source::iq2_s, "Gate+Up");
    run_case(3072u, 512u, sealed::Source::iq3_xxs, "Down");
    run_case(512u, 3072u, sealed::Source::iq3_xxs, "Gate+Up-mixed");
    run_case(3072u, 512u, sealed::Source::iq2_s, "Down-mixed");
    std::printf("test_moe_lru_sealed_hit_parity: sealed refill + hit_dot4/hit_dot8 bitwise-equal to the "
                "pristine IQ decode on both hit kinds\nPASS\n");
    return 0;
}
