// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Zero-Tax Suffix-First Hybrid speculation policy + online acceptance governor + lossless EVICT planning.
// Pure host logic: no GPU types, no device addresses. The engine loop asks this module what to draft, how
// much of it to verify and how to resynchronise the neural drafter's KV cache, then reports the result.
//
// Why the current Hybrid (SuffixTree + blk.48 MTP, run_greedy_mtp(use_hybrid_suffix=true)) can lose to pure
// suffix decoding, from reading models/ornith/ornith_gpu.cpp:
//   T1  Any suffix match pre-empts the neural drafter, even a single token at score 0.05: sfx.draft(qctx, k)
//       returns >= 1 node and `used_suffix` wins, so a 4-token MTP draft (~60% per-token acceptance) is often
//       replaced by a 1-2 token low-confidence suffix draft.
//   T2  Every suffix cycle still pays the drafter-KV catch-up: step_mtp() for vtoks[0] plus every accepted
//       token (acc+1 M=1 passes of the MTP layer, ~15 dispatches each) in an extra submit+wait. Pure suffix
//       decoding has no such tax, so long suffix runs are slower in Hybrid than in pure suffix mode.
//   T3  SuffixTree::draft() returns a TREE (max_branch=2 by default) in best-first order; the loop flattens
//       it into one linear verify sequence, so a sibling branch can sit at position j as if it continued
//       position j-1. Such positions can never be accepted, but they widen verification (MoE expert union).
//   T4  Neural drafting runs K serial submit+wait round trips, each with a full-vocab LM head
//       (~0.48 ms/GPU); FR-Spec (core/fr_spec.hpp) addresses that part.
// This module fixes T1-T3 on the host side:
//   * suffix-first with a confidence gate: a cycle uses the suffix chain only when it offers >= 4 (= the
//     ring-limited max draft) high-confidence LINEAR tokens; otherwise the neural drafter runs;
//   * best_chain() extracts a linear chain from the draft tree (or call draft(..., max_branch = 1));
//   * DrafterKvSync defers the drafter catch-up during suffix runs (only an h-row stash copy is paid per
//     suffix cycle) and replays at most `catchup_window` pending positions right before the next neural
//     cycle; older positions become KV holes in the drafter's own cache (never in the target model);
//   * AcceptanceGovernor adapts K per source from a windowed acceptance rate with hysteresis;
//   * plan_evict() truncates the verify prefix when the expected gain does not pay for the extra MoE expert
//     fan-out. Truncation is lossless: greedy verification of a shorter prefix commits a prefix of the same
//     token stream (the bonus token at the first unverified position is the target's own argmax).
#pragma once

#include "core/spec_engine.hpp"

#include <cstdint>
#include <deque>
#include <vector>

namespace vegajet::spec_policy {

enum class DraftSource : uint8_t { kPlain = 0, kSuffix = 1, kNeural = 2 };

// Linear draft chain: tokens[j] continues tokens[j-1]; cum_score[j] = product of empirical continuation
// probabilities along the path (non-increasing).
struct Chain {
    std::vector<int32_t> tokens;
    std::vector<double>  cum_score;
    size_t size() const { return tokens.size(); }
};
// Greedy best path through a best-first draft tree (parents precede children): from the root follow the
// highest-score child at each level.
Chain best_chain(const std::vector<spec::DraftNode>& nodes);
// Conditional per-position acceptance estimates from a chain: c_j = cum[j] / cum[j-1] (c_0 = cum[0]).
std::vector<double> conditional_probs(const Chain& c);

// ---------------------------------------------------------------------------------------------------------
// Acceptance governor: per source, windowed per-position acceptance alpha (geometric MLE over the last
// `window` cycles), K raised / lowered by one with hysteresis and a dwell time; K = 0 disables a source,
// with a periodic probe so a disabled source can recover.
// ---------------------------------------------------------------------------------------------------------
struct GovernorConfig {
    uint32_t k_min = 1, k_max = 4, k_init = 4;
    uint32_t window = 16;          // cycles in the sliding window
    uint32_t min_samples = 4;      // cycles before the first adjustment
    double   raise_above = 0.70;   // alpha above -> K+1
    double   lower_below = 0.45;   // alpha below -> K-1 (gap = hysteresis band)
    double   disable_below = 0.12; // alpha below at k_min -> K = 0 (source off)
    uint32_t dwell = 4;            // cycles between two K changes
    uint32_t probe_every = 32;     // a disabled source gets one k_min probe every N cycles
};

class AcceptanceGovernor {
public:
    explicit AcceptanceGovernor(GovernorConfig cfg = {});
    void observe(uint32_t proposed, uint32_t accepted);  // one verification of this source
    void tick();                                         // one engine cycle elapsed (any source)
    uint32_t k() const;                                  // K to use now (0 = off unless probing)
    double alpha() const;                                // windowed per-position acceptance
    uint32_t k_level() const { return k_; }
    uint64_t changes() const { return changes_; }

private:
    GovernorConfig cfg_;
    std::deque<std::pair<uint32_t, uint32_t>> win_;  // (proposed, accepted)
    uint32_t k_;
    uint32_t since_change_ = 0, since_probe_ = 0;
    uint64_t changes_ = 0;
};

// ---------------------------------------------------------------------------------------------------------
// Verification cost model + expert-union bookkeeping for EVICT.
// ---------------------------------------------------------------------------------------------------------
struct VerifyCostModel {
    // Measured verify time for M tokens (index M, entry 0 unused). If set for an M it wins.
    std::vector<double> measured_ms;
    double dense_ms   = 6.0;    // M-independent part of a target forward (dense trunk, attention, LM head, sync)
    double expert_ms  = 4.4;    // routed-expert weight streaming at M=1 (scales with the expert union)
    double per_tok_ms = 0.05;   // residual linear cost per verified token
    uint32_t n_experts = 256, experts_per_token = 8;
    // Optional observed expert-union sizes: union_obs[M] = mean distinct experts per layer for an M-token batch.
    std::vector<double> union_obs;
    double expected_union(uint32_t m) const;  // observed if available, else uniform-routing approximation
    double cost_ms(uint32_t m) const;
};

class ExpertUnionTracker {
public:
    // Record the measured mean distinct-expert count per MoE layer for one verify batch of m tokens
    // (e.g. from top_ids read back, or derived from the moe grouped-sort counters).
    void observe(uint32_t m, double distinct_per_layer, double ema = 0.2);
    void apply(VerifyCostModel* cm) const;  // writes union_obs
private:
    std::vector<double> mean_;
    std::vector<uint32_t> n_;
};

struct EvictDecision {
    uint32_t keep = 0;          // drafts to verify (prefix length); 0 = plain decode
    double   expected_tokens = 1.0;  // committed tokens expected (accepted + bonus)
    double   tokens_per_ms = 0.0;
};
// Choose m in [min_keep, cond.size()] maximising (1 + sum_{j<m} prod_{i<=j} cond[i]) / cost_ms(m + 1).
EvictDecision plan_evict(const std::vector<double>& cond, const VerifyCostModel& cm, uint32_t min_keep = 0,
                         double draft_ms = 0.0);

// ---------------------------------------------------------------------------------------------------------
// Measured-cost depth governor (v1.2 C7+S1). Chases the same objective as plan_evict, but the per-M verify
// cost comes from the wall time the decode loop measures for each verify batch (t_ver[M]) and the acceptance
// from the live accepted length, so no fabricated inflation constant is needed. Unseen M values are
// interpolated between the measured ones (parametric ratio below the smallest measured M) and the result is
// only used once the caller has fed it at least one sample.
// ---------------------------------------------------------------------------------------------------------
struct MeasuredDepthConfig {
    uint32_t k_min = 1;          // shortest proposal the governor may return (0 would mean plain decode)
    uint32_t min_samples = 2;    // verify samples of one M before that point is trusted on its own
    double   ema = 0.3;          // smoothing of the measured per-M time
    double   alpha_lo = 0.02;    // clamp for the live acceptance
    double   alpha_hi = 0.99;
    uint32_t alpha_window = 32;  // cycles of (proposed, accepted) behind the live acceptance
};

class MeasuredDepthGovernor {
public:
    explicit MeasuredDepthGovernor(MeasuredDepthConfig cfg = MeasuredDepthConfig());

    void observe_verify(uint32_t m, double ms);          // one verify pass over m tokens (m >= 1)
    void observe_accept(uint32_t proposed, uint32_t accepted);

    // Proposal depth for a batch capped at `cap`; 0 = not warmed up, the caller keeps its own depth.
    uint32_t next_depth(uint32_t cap) const;
    double alpha() const;
    double measured_ms(uint32_t m) const;   // smoothed measured verify time, 0 when unseen
    double est_ms(uint32_t m) const;        // measured / interpolated verify time, 0 when no sample at all
    double time_inflation(uint32_t m) const; // est_ms(m) / est_ms(1), 0 when unknown
    uint32_t samples(uint32_t m) const;

private:
    MeasuredDepthConfig cfg_;
    std::vector<double>   ms_;      // EWMA of the measured verify time per M
    std::vector<uint32_t> n_;       // samples per M
    std::deque<std::pair<uint32_t, uint32_t>> win_;  // (proposed, accepted) tail
};

// ---------------------------------------------------------------------------------------------------------
// Drafter (MTP head) KV resynchronisation bookkeeping.
// h sources map to engine buffers: kPend -> S[g].mtp_h_pend_va, kTrunkRow j -> S[g].h_va + j*D*4,
// kStash s -> (new) stash buffer + s*D*4.
// ---------------------------------------------------------------------------------------------------------
struct HSource {
    enum Kind : uint8_t { kPend = 0, kTrunkRow = 1, kStash = 2 } kind = kPend;
    uint32_t index = 0;
};
struct StashCopy { HSource src; uint32_t dst_slot = 0; };
struct CatchupStep { int32_t token = -1; uint32_t pos = 0; HSource h; };
struct SyncPlan {
    std::vector<StashCopy>   stash;     // enqueue first (before h rows are overwritten)
    std::vector<CatchupStep> catchup;   // then one step_mtp(token, pos, h, logits=false) each, in order
    uint32_t holes_added = 0;
};

enum class SyncMode : uint8_t { kEager = 0, kDeferred = 1 };

struct SyncConfig {
    SyncMode mode = SyncMode::kDeferred;
    uint32_t catchup_window = 8;     // pending positions replayed before the next neural cycle
    uint32_t stash_slots    = 16;    // >= catchup_window + max commit per cycle
    uint32_t hole_window    = 256;   // positions over which the hole fraction is measured
    double   max_hole_frac  = 0.25;  // above this, suffix cycles fall back to eager catch-up ...
    double   resume_hole_frac = 0.10; // ... until the windowed hole fraction drops below this (hysteresis)
};

class DrafterKvSync {
public:
    explicit DrafterKvSync(SyncConfig cfg = {});
    // After a verification. committed = vtoks[0..acc] (the verified input tokens whose KV the drafter must
    // hold), pos0 = position of vtoks[0]. drafter_ingested_first: true for a neural cycle (draft step 0 ran on
    // vtoks[0] with the correct h_pend), false for suffix / plain cycles.
    SyncPlan after_verify(const std::vector<int32_t>& committed, uint32_t pos0, bool drafter_ingested_first,
                          bool next_cycle_neural_likely);
    // Before a neural drafting cycle: replay the pending positions (bounded by catchup_window).
    SyncPlan before_neural();
    uint32_t pending() const { return static_cast<uint32_t>(pending_.size()); }
    double hole_fraction() const;
    uint64_t total_holes() const { return holes_total_; }
    SyncMode effective_mode() const;

private:
    struct Pending { int32_t token; uint32_t pos; uint32_t slot; };
    void note_positions(uint32_t n, uint32_t holes);
    SyncConfig cfg_;
    std::deque<Pending> pending_;
    uint32_t next_slot_ = 0;
    std::deque<uint8_t> hole_hist_;
    uint32_t hole_count_ = 0;
    uint64_t holes_total_ = 0;
    bool eager_latched_ = false;
};

// ---------------------------------------------------------------------------------------------------------
// Per-cycle decision.
// ---------------------------------------------------------------------------------------------------------
struct PolicyConfig {
    uint32_t max_draft = 4;             // ring-limited: verify M = K+1 <= kGdnRingSlots - 1
    uint32_t suffix_first_min = 4;      // >= this many confident suffix tokens -> skip the neural drafter
    double   suffix_conf_score = 0.30;  // cum_score threshold of a "confident" suffix token
    uint32_t suffix_min_match = 2;      // minimal context match length for a confident suffix draft
    bool     short_suffix_fallback = true;  // when the neural source is off, use shorter suffix chains
    GovernorConfig neural_gov;
    GovernorConfig suffix_gov;
    VerifyCostModel cost;
    double   neural_draft_ms_per_tok = 0.7;  // drafting cost per neural step (with FR-Spec lower)
};

struct CycleInputs {
    uint32_t remaining = 0;       // tokens still to generate
    Chain    suffix;              // linear suffix chain (best_chain or draft(..., max_branch=1))
    uint32_t suffix_match_len = 0;
    bool     neural_available = true;
};

struct CycleDecision {
    DraftSource source = DraftSource::kPlain;
    std::vector<int32_t> suffix_tokens;  // kSuffix: tokens to verify after gen.back()
    uint32_t neural_k = 0;               // kNeural: drafter steps to run
    uint32_t confident = 0;              // confident suffix tokens seen (diagnostic)
};

struct PolicyStats {
    uint64_t cycles = 0, plain = 0, suffix = 0, neural = 0;
    uint64_t proposed = 0, accepted = 0, evicted = 0;
};

class SpecPolicy {
public:
    explicit SpecPolicy(PolicyConfig cfg = {});
    CycleDecision decide(const CycleInputs& in);
    // EVICT for neural drafts (after drafting, before verification): returns the prefix length to verify.
    // cond: per-step conditional acceptance estimates (empty -> governor alpha for every step).
    uint32_t evict_neural(uint32_t drafted, const std::vector<double>& cond = {});
    // Report the verification outcome (proposed = drafts actually verified).
    void observe(DraftSource src, uint32_t proposed, uint32_t accepted);
    const PolicyStats& stats() const { return st_; }
    const AcceptanceGovernor& neural_governor() const { return gov_neural_; }
    const AcceptanceGovernor& suffix_governor() const { return gov_suffix_; }
    VerifyCostModel& cost_model() { return cfg_.cost; }

private:
    PolicyConfig cfg_;
    AcceptanceGovernor gov_neural_, gov_suffix_;
    PolicyStats st_;
};

} // namespace vegajet::spec_policy
