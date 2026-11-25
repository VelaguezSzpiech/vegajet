// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Deterministic token sampler (host reference, GPU-compatible).
//
// Pipeline (llama.cpp order): repeat penalty -> greedy fast path (temperature <= 0) -> top-k -> temperature softmax
// -> top-p -> min-p -> draw. Ties always break toward the lower token id, so results never depend on sort stability.
//
// GPU compatibility: the random draw is counter-based. Step t of a request with seed s uses
//     u = (splitmix64(s ^ (t * 0x9E3779B97F4A7C15)) >> 11) * 2^-53        in [0, 1)
// so a device kernel (or a top-k candidate list produced on the device) reproduces the host draw bit-for-bit with
// no shared RNG state. sample_candidates() takes such a device-side candidate list directly.
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

namespace vegajet::sampling {

struct SamplerParams {
    float temperature = 0.0f;      // <= 0: greedy argmax
    int32_t top_k = 0;             // <= 0: disabled
    float top_p = 1.0f;            // >= 1: disabled
    float min_p = 0.0f;            // <= 0: disabled; keeps p >= min_p * p_max
    float repeat_penalty = 1.0f;   // 1: disabled; logit > 0 is divided, logit < 0 multiplied
    int32_t repeat_last_n = 64;    // window of recent tokens the penalty sees (0: disabled, -1: whole history)
    uint64_t seed = 0;
};

struct Candidate {
    int32_t id = 0;
    float logit = 0.0f;
};

uint64_t splitmix64(uint64_t x);
// Uniform [0, 1) draw for step `step` of a request seeded with `seed` (see header comment).
double counter_uniform(uint64_t seed, uint64_t step);

class Sampler {
public:
    explicit Sampler(SamplerParams p = {});

    // Samples from full-vocabulary logits. Does not record the token: call accept() with what was committed.
    int32_t sample(const float* logits, size_t n_vocab);
    // Samples from a candidate subset (for example a device top-k list). Ids outside the subset are never chosen.
    int32_t sample_candidates(std::vector<Candidate> cands);
    // Records a committed token (penalty window + draw counter). Prompt tokens may be fed here too.
    void accept(int32_t id);
    void reset();

    const SamplerParams& params() const { return p_; }
    uint64_t step() const { return step_; }

private:
    void apply_penalty(std::vector<Candidate>& c) const;
    int32_t pick(std::vector<Candidate>& c);

    SamplerParams p_;
    std::deque<int32_t> recent_;
    uint64_t step_ = 0;
};

} // namespace vegajet::sampling
