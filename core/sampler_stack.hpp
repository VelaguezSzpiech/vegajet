// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Full deterministic sampler chain (host side), in llama.cpp's default order:
//
//   logit bias -> repetition / presence / frequency penalties -> DRY -> top-k -> typical-p -> top-p -> min-p -> XTC
//   -> temperature -> draw                    (mirostat = 2: bias -> penalties -> DRY -> temperature -> Mirostat v2)
//
// Truncation stages see temperature-1 probabilities; temperature only shapes the final draw (as in llama.cpp).
// Draws are counter-based (core/sampler: splitmix64 of seed, step and a stream id), so a request replays bit for
// bit from its seed regardless of threading or batch position.
//
// Cost model, 248K-token vocabularies: bias / penalties / DRY touch only the affected ids; candidates come from one
// streaming pass that skips 64-logit blocks whose maximum is below the running top-k threshold; the full-vocabulary
// normalizer (logprobs, Mirostat, or no top-k) is one SIMD polynomial-exp pass, split over `threads` OpenMP threads
// with a fixed chunk order (bit-identical results for any thread count). When top-k is off and a
// truncation stage would need tokens beyond the candidate set (`candidate_cap`), the set grows 8x and the step is
// redone (exactly, up to the whole vocabulary), so results never depend on the cap.
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace vegajet::sampling {

struct StackParams {
    float temperature = 0.8f;      // <= 0: greedy (after the truncation stages)
    int32_t top_k = 40;            // <= 0: off
    float top_p = 0.95f;           // >= 1: off
    float min_p = 0.05f;           // <= 0: off
    float typical_p = 1.0f;        // >= 1: off
    float xtc_probability = 0.0f;  // <= 0: off
    float xtc_threshold = 0.1f;    // > 0.5: off
    float repeat_penalty = 1.0f;   // 1: off
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    int32_t penalty_last_n = 64;   // 0: penalties off, -1: whole history
    float dry_multiplier = 0.0f;   // 0: off
    float dry_base = 1.75f;
    int32_t dry_allowed_length = 2;
    int32_t dry_last_n = -1;       // -1: whole history (at most 8192 tokens)
    std::vector<int32_t> dry_breakers;  // token ids that end a repeated sequence (newline, ':', '"', '*')
    int32_t mirostat = 0;          // 0 off, 2 Mirostat v2
    float mirostat_tau = 5.0f;
    float mirostat_eta = 0.1f;
    std::vector<std::pair<int32_t, float>> logit_bias;  // added to the logit; -INFINITY bans the token
    uint64_t seed = 0;
    int32_t n_logprobs = 0;        // top-N logprobs (0..20) of the full temperature-1 distribution after bias / penalties
    uint32_t candidate_cap = 256;  // candidates kept when top_k is off (exact fallback beyond it)
    int32_t threads = 4;           // full-vocabulary normalizer threads (OpenMP builds, vocab >= 64K); 1 = serial.
                                   // Fixed chunking: results are bit-identical for any value.
};

struct TokenLogprob {
    int32_t id = -1;
    float logprob = 0.0f;
};

struct SampleResult {
    int32_t id = -1;
    float logprob = 0.0f;              // of the chosen token (only when n_logprobs > 0 or Mirostat is on)
    std::vector<TokenLogprob> top;     // n_logprobs entries, most likely first
    uint32_t n_candidates = 0;         // candidates left for the final draw
    bool full_vocab_fallback = false;  // a stage needed more than the candidate set: grown 8x per retry (exact)
};

class SamplerStack {
public:
    explicit SamplerStack(StackParams p = {});
    SampleResult sample(const float* logits, size_t n_vocab);
    // Records a committed token (penalty and DRY history, draw counter). Prompt tokens may be fed too.
    void accept(int32_t id);
    void reset();
    const StackParams& params() const { return p_; }
    float mirostat_mu() const { return mu_; }
    uint64_t step() const { return step_; }

    struct Cand {
        float logit;
        int32_t id;
        float p;
    };

private:
    bool run_chain(std::vector<Cand>& c, double tail, double m, int32_t& chosen, bool& need_full);
    void collect(const float* logits, size_t n_vocab, size_t k, std::vector<Cand>& out, float& raw_max);
    StackParams p_;
    std::vector<int32_t> hist_;
    std::vector<std::pair<int32_t, float>> overrides_;  // (id, adjusted logit)
    std::vector<uint32_t> stamp_;  // stamp_[id] == epoch_: id has an override, at overrides_[slot_[id]]
    std::vector<uint32_t> slot_;
    uint32_t epoch_ = 0;
    float mu_ = 10.0f;
    uint64_t step_ = 0;
};

// Sum of exp(inv_t * (x[i] - m)) over n values (x <= m): SIMD polynomial exp (relative error ~1e-7 per term), fixed
// 4096-term chunks summed in index order, so the result does not depend on `threads`.
double sum_exp_shifted(const float* x, size_t n, float m, float inv_t = 1.0f, int threads = 1);

} // namespace vegajet::sampling
