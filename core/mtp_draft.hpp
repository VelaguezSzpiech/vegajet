// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Host side of the blk.48 MTP draft path (v0.1.1): launch descriptors for K chained draft steps with no host wait
// between them, the drafter-cache bookkeeping around verify, FR-Spec draft ids, the >= 4-token suffix bypass and the
// per-cycle statistics the bench prints.
//
// Today (models/ornith/ornith_gpu.cpp run_greedy_mtp) every draft step ends in submit + wait_idle + a host argmax over
// 2 x 240 block maxima, and the next step's token is written by the host. plan_chain() describes the same K steps as
// one device-chained sequence: step 0 reads the committed token from the host I/O slot and h_pend; step j > 0 reads
// the token the device argmax of step j - 1 produced (resolved through the FR-Spec row map on the device when the
// shortlist is on) and the hidden state step j - 1 wrote into its own h slot (no in-place reuse of one h buffer).
// Only the last step is read by the host. Device dispatch and synchronisation are the runtime owner's; this header
// fixes the layout and the invariants, and resolve_draft_id() is the exact host reference of the id mapping.
//
// Pure host code: no GPU, no file I/O, no environment reads.
#pragma once

#include "core/spec_replay.hpp"
#include "core/suffix_trie.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::mtp_draft {

constexpr uint32_t kMaxChain = 16;       // kMaxSpecBatch: h_cur / io slots available per cycle
constexpr uint32_t kSuffixBypassMin = 4; // suffix match >= 4 tokens and chain >= 4 tokens skips the neural drafter

// Alpha's threshold unless the stricter continuation rule is explicitly selected. Pure host policy:
// the engine alone reads VEGAJET_MTP_SUFFIX_MIN4, so tests need no environment mutation.
constexpr uint32_t suffix_min_chain(uint32_t depth, bool zero_tax_catchup, bool min4_enabled) {
    return (zero_tax_catchup || min4_enabled) && depth > 0u
               ? (depth < kSuffixBypassMin ? depth : kSuffixBypassMin) : 1u;
}

enum class TokSrc : uint32_t { kHostSlot = 0, kDeviceArgmax = 1 };
enum class HSrc : uint32_t { kPend = 0, kMtpOut = 1 };
enum class LmMode : uint32_t { kFullVocab = 0, kShortlist = 1 };

// 64-byte POD consumed by the device chain (one per step, uploaded once per cycle).
struct StepDesc {
    uint32_t step = 0;
    uint32_t pos = 0;          // position of the input token; the MTP KV row written by this step
    uint32_t tok_src = 0;      // TokSrc
    uint32_t tok_slot = 0;     // kHostSlot: io_toks slot; kDeviceArgmax: argmax slot of the producing step
    uint32_t h_src = 0;        // HSrc
    uint32_t h_slot = 0;       // kMtpOut: h_cur slot written by step - 1
    uint32_t h_out_slot = 0;   // h_cur slot this step writes (== step)
    uint32_t lm_mode = 0;      // LmMode
    uint32_t lm_rows = 0;      // LM-head rows per rank this step scans
    uint32_t argmax_slot = 0;  // where this step's (value, id) block maxima land (== step)
    uint32_t host_read = 0;    // 1 on the last step only: the host waits once, after it
    uint32_t attend_hi = 0;    // keys 0 .. attend_hi - 1 of the MTP cache are read (== pos + 1)
    uint32_t reserved[4] = {0, 0, 0, 0};
};
static_assert(sizeof(StepDesc) == 64, "StepDesc is a 64-byte device record");

struct ChainConfig {
    uint32_t k = 4;            // draft steps
    uint32_t max_ctx = 32768;  // MTP KV rows allocated
    uint32_t vocab = 0;
    uint32_t n_ranks = 2;
    const spec_replay::Shortlist* shortlist = nullptr;  // FR-Spec rows (nullptr: full vocabulary)
    std::vector<uint32_t> voc_rows;  // full-vocabulary rows per rank (TP split), used when shortlist == nullptr
};

struct ChainPlan {
    bool ok = false;
    std::string error;
    std::vector<StepDesc> steps;
    uint32_t host_waits = 0;   // 1 for a valid chain
};
ChainPlan plan_chain(const ChainConfig& cfg, uint32_t anchor_pos);
// Structural check of a plan (what the device chain relies on).
bool check_chain(const ChainPlan& p, const ChainConfig& cfg, uint32_t anchor_pos, std::string* err);

// Argmax id -> vocabulary id. Full vocabulary: the kernel already returns the global id. Shortlist: the kernel
// returns rank * rows_per_rank + local row (ro = rank * lr), mapped through tok_map (engine step_mtp, lines after
// the host argmax). -1 for an id outside the table.
int32_t resolve_draft_id(int32_t kernel_id, const ChainConfig& cfg);

// ---- MTP cache bookkeeping -----------------------------------------------------------------------------------------
// Row states of the blk.48 KV cache: exact (written from the target's hidden state of the committed prefix),
// provisional (written during drafting from the MTP head's own hidden state), empty.
enum class Row : uint8_t { kEmpty = 0, kExact, kProvisional };

struct CatchupStep { int32_t token = -1; uint32_t pos = 0; uint32_t trunk_row = 0; };
struct VerifyPlan {
    std::vector<CatchupStep> catchup;  // step_mtp(token, pos, h = trunk row, logits off), in order
    uint32_t h_pend_trunk_row = 0;     // h_pend <- trunk hidden of vtoks[acc]
    uint32_t next_anchor = 0;
};

class MtpLedger {
public:
    explicit MtpLedger(uint32_t max_ctx) : rows_(max_ctx, Row::kEmpty) {}
    // Prefill fed positions [0, n) exactly (h = previous trunk row, first = zero vector).
    bool prefill(uint32_t n, std::string* err);
    // A neural chain from `anchor` with k steps: every row < anchor must be exact; writes anchor exact (h_pend is the
    // target's) and anchor + 1 .. anchor + k - 1 provisional.
    bool draft(uint32_t anchor, uint32_t k, std::string* err);
    // After verify of [vtoks[0] at anchor, drafts...] with `acc` accepted. neural: the chain ran from anchor.
    // Returns the catch-up so that rows < anchor + acc + 1 are exact again.
    VerifyPlan after_verify(const std::vector<int32_t>& vtoks, uint32_t anchor, uint32_t acc, bool neural);
    void apply(const VerifyPlan& v);       // marks the catch-up rows exact
    uint32_t exact_prefix() const;         // rows [0, n) all exact
    Row row(uint32_t p) const { return p < rows_.size() ? rows_[p] : Row::kEmpty; }

private:
    std::vector<Row> rows_;
};

// ---- suffix bypass ---------------------------------------------------------------------------------------------------
struct BypassDecision {
    bool bypass = false;
    std::vector<int32_t> draft;
    uint32_t match_len = 0;
};
// history: committed tokens (prompt + output). A suffix match of >= kSuffixBypassMin tokens offering a chain of
// >= kSuffixBypassMin tokens replaces the neural draft (chain cut to max_draft).
class SuffixBypass {
public:
    explicit SuffixBypass(uint32_t max_draft);
    void commit(int32_t tok);
    void commit(const std::vector<int32_t>& toks);
    BypassDecision decide();  // counts hits / misses

    uint64_t hits = 0, misses = 0;

private:
    uint32_t max_draft_;
    sfx::CascadeHybrid cascade_;  // the engine's suffix lookup (min_match = min_chain = kSuffixBypassMin)
};

// ---- per-cycle statistics ------------------------------------------------------------------------------------------
struct CycleSample {
    bool suffix = false;
    uint32_t depth = 0;        // drafts proposed
    uint32_t accepted = 0;
    double draft_ms = 0.0, verify_ms = 0.0;
    // routed expert ids touched by the verify batch, [layer][token * top_k + j] (empty: not captured)
    std::vector<std::vector<int32_t>> experts;
};

struct SpecStats {
    uint64_t cycles = 0, neural = 0, suffix = 0, drafted = 0, accepted = 0, committed = 0;
    double draft_ms = 0.0, verify_ms = 0.0, wall_s = 0.0;
    uint64_t expert_cycles = 0;
    double unique_experts_sum = 0.0;   // mean over layers of distinct experts per verify batch, summed over cycles
    void add(const CycleSample& c);
    void add_single() { ++committed; ++single_steps; }  // a plain M = 1 target step (no draft)
    uint64_t single_steps = 0;
    double accepted_per_cycle() const { return cycles ? static_cast<double>(accepted) / cycles : 0.0; }
    double tokens_per_cycle() const { return cycles ? static_cast<double>(committed - single_steps) / cycles : 0.0; }
    double mean_depth() const { return cycles ? static_cast<double>(drafted) / cycles : 0.0; }
    double suffix_hit_rate() const { return cycles ? static_cast<double>(suffix) / cycles : 0.0; }
    double unique_experts() const { return expert_cycles ? unique_experts_sum / expert_cycles : -1.0; }
    // committed (accepted + bonus + single-step) output tokens per wall-clock second of the decode loop
    double accepted_tok_s() const { return wall_s > 0.0 ? static_cast<double>(committed) / wall_s : 0.0; }
};
// One parseable bench line:
// [SPEC-STATS <label>] cycles=N acc/cyc=A tok/cyc=T depth=D sfx_hit=P% uniq_exp=U draft_ms=X verify_ms=Y acc_tok_s=Z
std::string format_stats(const std::string& label, const SpecStats& s);

}  // namespace vegajet::mtp_draft
