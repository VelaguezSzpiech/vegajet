// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Zero-Tax Cascade Hybrid Router (host only, 0 B VRAM).
//
// Per speculative cycle the runner asks one question: is the CPU suffix draft good enough to skip the GPU
// neural drafter (blk.48 MTP / DFlash) entirely? Every neural cycle pays the drafter's launches and LM-head
// pass before verification even starts; a SUFFIX_ONLY cycle pays nothing on the GPU before verification.
//
// What this adds on top of core/suffix_trie.hpp (sfx::SuffixIndex, the online suffix automaton):
//   * a per-token CONFIDENCE for the suffix chain. sfx::CascadeHybrid gates on match length and chain length
//     only, so a long match whose context was continued five different ways earlier, or that occurred exactly
//     once, still counts as a sure thing. Here
//         confidence[j] = prod_{i<=j} min(1 / degree_i, prob_i)
//     degree_i: distinct tokens that followed (match + chain[0..i-1]) earlier in the history
//               (SuffixIndex::continuation_degrees, O(K) automaton walk) -> ambiguity of the long context;
//     prob_i:   count(ctx_i -> chain[i]) / (count(ctx_i) + 1) over the last `support_order` tokens ctx_i
//               (online n-gram follower counts, O(1) per commit) -> evidence: a context seen once scores 0.5;
//   * the explicit decision: >= min_confident tokens with confidence >= min_confidence (and a match of
//     >= min_match tokens) -> DraftSource::SUFFIX_ONLY with that confident prefix as the draft; otherwise
//     DraftSource::NEURAL_DRAFTER. Off by default (VEGAJET_SPEC_CASCADE=1 enables it); when off, every cycle
//     is NEURAL_DRAFTER and the index is not queried. Verification is unchanged, so routing is lossless;
//   * EVICT expert-union truncation with a HARD budget: deduplicates the (layer, expert) sets of the M tokens
//     of a verify batch and returns the exact prefix that keeps the unique fan-out within budget (the
//     probabilistic score-based truncation lives in core/evict_controller.hpp; this is the cap that must hold
//     regardless of acceptance estimates, e.g. the routed experts that fit the per-pass VRAM/LRU window).
//     Truncation is lossless: greedy verification of a shorter prefix commits a prefix of the same stream.
// Cost per step: push O(1) amortized + O(K * N) lookups; the test holds it under 100 us at a 16K history.
#pragma once

#include "core/suffix_trie.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace vegajet::spec_cascade {

constexpr const char* kCascadeEnv    = "VEGAJET_SPEC_CASCADE";      // "1" = on (default off)
constexpr const char* kCascadeMinEnv = "VEGAJET_SPEC_CASCADE_MIN";  // optional min_confident override (1..64)

enum class DraftSource : uint8_t { SUFFIX_ONLY = 0, NEURAL_DRAFTER = 1 };
const char* draft_source_name(DraftSource s);

struct CascadeConfig {
    bool     enabled        = false;
    uint32_t max_draft      = 8;    // K: suffix tokens looked at per step
    uint32_t min_match      = 4;    // matched suffix tokens before the chain is considered at all
    uint32_t min_confident  = 4;    // >= this many confident tokens -> SUFFIX_ONLY
    double   min_confidence = 0.3;  // cumulative confidence a token needs to count as confident
    uint32_t support_order  = 4;    // n-gram context length of the follower counts (1..16)
};
// `base` with VEGAJET_SPEC_CASCADE ("1"/"0") and VEGAJET_SPEC_CASCADE_MIN applied; invalid values are ignored
// and reported in *note.
CascadeConfig config_from_env(CascadeConfig base = {}, std::string* note = nullptr);

struct SuffixDraft {
    std::vector<int32_t>  tokens;      // linear continuation chain, up to max_draft tokens
    std::vector<uint32_t> degree;      // distinct earlier continuations per position (long context)
    std::vector<uint32_t> support;     // earlier occurrences of the order-N context per position
    std::vector<float>    prob;        // count(context -> token) / (support + 1) per position
    std::vector<float>    confidence;  // cumulative, non-increasing, in [0, 1]
    uint32_t match_length = 0;
    uint32_t confident = 0;            // leading tokens with confidence >= min_confidence (0 if match < min_match)
};

struct CascadeStep {
    DraftSource source = DraftSource::NEURAL_DRAFTER;
    std::vector<int32_t> draft;  // SUFFIX_ONLY: the confident prefix to verify after the last committed token
    SuffixDraft suffix;          // the suffix view (empty when the cascade is disabled)
};

struct CascadeStats {
    uint64_t steps = 0, suffix_only = 0, neural = 0;
    uint64_t suffix_tokens = 0;  // draft tokens handed out by SUFFIX_ONLY steps
};

class ZeroTaxCascade {
public:
    explicit ZeroTaxCascade(CascadeConfig cfg = {});
    void reserve(size_t tokens);
    void commit(int32_t token);
    void commit(const std::vector<int32_t>& tokens) { for (int32_t t : tokens) commit(t); }
    SuffixDraft suffix_draft() const;
    CascadeStep peek() const;  // the decision, without touching the stats
    CascadeStep decide();      // the decision, counted in stats()
    const CascadeConfig& config() const { return cfg_; }
    const CascadeStats& stats() const { return st_; }
    const sfx::SuffixIndex& index() const { return idx_; }
    void clear();

private:
    uint64_t ctx_hash(const std::vector<int32_t>& seq, size_t end) const;  // hash of seq[end-N, end)
    CascadeConfig cfg_;
    sfx::SuffixIndex idx_;
    CascadeStats st_;
    std::unordered_map<uint64_t, uint32_t> ctx_total_;   // order-N context -> occurrences with a follower
    std::unordered_map<uint64_t, uint32_t> ctx_follow_;  // (context, follower) -> count
};

// ---------------------------------------------------------------------------------------------------------
// EVICT expert union + prefix truncation.
// ---------------------------------------------------------------------------------------------------------
using Routing = std::vector<std::vector<uint16_t>>;  // [layer] -> routed expert ids of one token

struct UnionBudget {
    uint64_t max_unique_total     = 0;  // unique (layer, expert) pairs over the kept prefix; 0 = unlimited
    uint32_t max_unique_per_layer = 0;  // unique experts in any single layer; 0 = unlimited
    uint32_t min_keep             = 1;  // never keep fewer tokens than this (token 0 = the verified root)
};

struct ExpertUnion {
    uint32_t keep = 0;                           // candidates [0, keep) are verified
    bool truncated = false;                      // keep < M because of the budget
    std::vector<uint64_t> unique_after;          // [M] unique pairs over candidates [0, i]
    std::vector<uint32_t> max_layer_after;       // [M] largest per-layer unique count over candidates [0, i]
    std::vector<std::vector<uint16_t>> experts;  // [layer] ascending unique experts of the kept prefix
    uint64_t unique_kept = 0;
    double fan_out = 0.0;                        // unique_kept / unique pairs of candidate 0 alone
};

// candidates[i] = routing of the i-th token of the verify batch (0 = the root token, then the drafts in
// order; predicted routings are fine). Each routing must have n_layers entries with ids < n_experts
// (duplicates inside one token are counted once). keep = the largest prefix within budget, but at least
// min(min_keep, M).
bool expert_union_truncate(const std::vector<Routing>& candidates, uint32_t n_layers, uint32_t n_experts,
                           const UnionBudget& budget, ExpertUnion* out, std::string* err);
// The drafts that survive: candidates 1..keep-1 correspond to draft[0..keep-2].
std::vector<int32_t> truncate_draft(const std::vector<int32_t>& draft, const ExpertUnion& u);

} // namespace vegajet::spec_cascade
