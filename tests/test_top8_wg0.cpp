// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU numerical oracle for persistent MoE Top8 and its 64-byte WG0 publication contract.
// This does not validate HIP cache coherence, rendezvous progress, __expf accuracy, or GPU parity.
#include "tests/test_common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>

namespace {
using Logits = std::array<float, 256>;
struct Payload {
    std::array<int32_t, 8> ids;
    std::array<float, 8> weights;
};
static_assert(sizeof(Payload) == 64 && offsetof(Payload, weights) == 32,
              "Top8 publication is exactly 32 bytes of ids plus 32 bytes of weights");

// Emulate the existing four-logits/lane winner selection, including lane-first/element-first ties.
// Both default local evaluation and WG0 publication retain this exact arithmetic/order on the GPU.
Payload wave_top8(const Logits& input) {
    float values[64][4];
    float local_max[64];
    for (size_t lane = 0; lane < 64; ++lane) {
        for (size_t q = 0; q < 4; ++q) values[lane][q] = input[lane * 4 + q];
        local_max[lane] = std::fmax(std::fmax(values[lane][0], values[lane][1]),
                                    std::fmax(values[lane][2], values[lane][3]));
    }
    Payload out{};
    float maximum = 0.0f;
    for (size_t rank = 0; rank < 8; ++rank) {
        float best = local_max[0];
        for (size_t lane = 1; lane < 64; ++lane) best = std::fmax(best, local_max[lane]);
        if (rank == 0) maximum = best;
        uint64_t ballot = 0;
        for (size_t lane = 0; lane < 64; ++lane)
            for (size_t q = 0; q < 4; ++q)
                if (values[lane][q] == best) ballot |= uint64_t{1} << lane;
        VJ_CHECK(ballot != 0);
        const size_t winner_lane = static_cast<size_t>(__builtin_ctzll(ballot));
        size_t winner_q = 0;
        while (values[winner_lane][winner_q] != best) ++winner_q;
        out.ids[rank] = static_cast<int32_t>(winner_lane * 4 + winner_q);
        out.weights[rank] = best;
        values[winner_lane][winner_q] = -1e30f;
        local_max[winner_lane] = std::fmax(std::fmax(values[winner_lane][0], values[winner_lane][1]),
                                           std::fmax(values[winner_lane][2], values[winner_lane][3]));
    }
    for (float& weight : out.weights) weight = std::exp(weight - maximum);
    float sum = 0.0f;
    for (float weight : out.weights) sum += weight;
    const float inverse = 1.0f / std::fmax(sum, 1e-30f);
    for (float& weight : out.weights) weight *= inverse;
    return out;
}

void check(const Logits& logits, bool replay_all_wgs) {
    // Independent stable-sort reference: unlike wave_top8, it never masks/updates lane maxima.
    std::array<int32_t, 256> sorted;
    std::iota(sorted.begin(), sorted.end(), int32_t{0});
    std::stable_sort(sorted.begin(), sorted.end(), [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
    std::array<double, 8> reference_weights;
    double sum = 0.0;
    for (size_t i = 0; i < 8; ++i) {
        reference_weights[i] = std::exp(static_cast<double>(logits[sorted[i]]) - logits[sorted[0]]);
        sum += reference_weights[i];
    }
    const Payload published = wave_top8(logits);
    double normalized_sum = 0.0;
    for (size_t i = 0; i < 8; ++i) {
        VJ_CHECK(published.ids[i] == sorted[i]);
        VJ_CHECK(std::isfinite(published.weights[i]) && published.weights[i] >= 0.0f);
        VJ_CHECK(std::fabs(published.weights[i] - reference_weights[i] / sum) < 2e-7);
        normalized_sum += published.weights[i];
    }
    VJ_CHECK(std::fabs(normalized_sum - 1.0) < 3e-7);
    // Model the two distinct global arrays and eight publishing/reading lanes, not struct aliasing.
    int32_t global_ids[8];
    float global_weights[8];
    for (size_t lane = 0; lane < 8; ++lane) {
        global_ids[lane] = published.ids[lane];
        global_weights[lane] = published.weights[lane];
    }
    for (size_t wg = 0; wg < 240; ++wg) {
        Payload received{};
        for (size_t lane = 0; lane < 8; ++lane) {
            received.ids[lane] = global_ids[lane];
            received.weights[lane] = global_weights[lane];
        }
        VJ_CHECK(std::memcmp(&received, &published, sizeof(Payload)) == 0);
        if (replay_all_wgs) {
            const Payload local = wave_top8(logits);
            VJ_CHECK(std::memcmp(&received, &local, sizeof(Payload)) == 0);
        }
    }
}
} // namespace

int main() {
    Logits x{};
    // All ties, including ties that cross lane/quad boundaries: ids must be ascending.
    x.fill(1.0f);
    check(x, true);
    x.fill(-9.0f);
    for (int id : {0, 3, 4, 63, 64, 127, 128, 251, 252, 255}) x[static_cast<size_t>(id)] = 7.0f;
    check(x, true);
    // Equal signed zeros compare as ties; large gaps exercise exponent underflow without NaNs.
    for (size_t i = 0; i < x.size(); ++i) x[i] = (i & 1u) ? -0.0f : 0.0f;
    check(x, true);
    for (size_t i = 0; i < x.size(); ++i) x[i] = 1000.0f - static_cast<float>(i) * 100.0f;
    check(x, true);
    // A fixed integer generator makes both dense ties and untied cases reproducible across standard libraries.
    uint32_t state = 0x6a09e667u;
    for (size_t trial = 0; trial < 512; ++trial) {
        for (float& value : x) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            const int32_t sample = static_cast<int32_t>(state & 0xffffu) - 32768;
            value = (trial & 1u) ? static_cast<float>(sample) / 1024.0f
                                : static_cast<float>(sample % 13);
        }
        check(x, trial < 16);
    }
    std::puts("PASS top8_wg0: 516 cases, stable-sort ids/weights, 240-WG bit-exact payload broadcast (CPU only)");
}
