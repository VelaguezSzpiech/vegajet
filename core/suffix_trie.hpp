// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// O(1)-amortized suffix index for the Zero-Tax Cascade Hybrid drafter (host, per committed token).
//
// HEAD's spec::SuffixTree inserts every window up to max_depth per token (depth-bounded trie, map per node).
// SuffixIndex is an online suffix automaton over the committed token history instead:
//   * push(): amortized O(1) per token, unbounded match length (no depth cap);
//   * longest_match(): the longest suffix of the history that also occurred earlier, plus where its first earlier
//     occurrence ended. That is link(last) in the automaton, so the query is O(1);
//   * draft(): the tokens that followed that earlier occurrence (a linear chain, never a tree).
// CascadeHybrid applies the zero-tax rule: when the suffix match is >= min_match tokens and it offers >= min_chain
// continuation tokens, the cycle uses the suffix draft and skips the GPU drafter entirely; otherwise the neural
// drafter runs. Verification is unchanged, so the cascade is lossless by construction.
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace vegajet::sfx {

struct Match {
    uint32_t length = 0;     // tokens of the current suffix that occurred earlier
    int64_t end_pos = -1;    // index in history() of the last token of the first earlier occurrence
};

class SuffixIndex {
public:
    SuffixIndex();
    void push(int32_t token);
    void extend(const std::vector<int32_t>& tokens) { for (int32_t t : tokens) push(t); }
    // Pre-sizes for `tokens` history tokens so push() never rehashes (removes the rehash latency spikes).
    void reserve(size_t tokens);
    Match longest_match() const;
    // Up to max_tokens tokens that followed the earlier occurrence (stops at the end of the history).
    std::vector<int32_t> draft(uint32_t max_tokens) const;
    // For a chain returned by draft(): degree[j] = number of distinct tokens that followed (the longest match +
    // chain[0..j-1]) anywhere earlier in the history (1 = that context was always continued the same way).
    // Walks the automaton along the chain: O(chain length), no history scan. Stops early on a token the
    // automaton has no transition for (i.e. a chain not produced by draft()).
    std::vector<uint32_t> continuation_degrees(const std::vector<int32_t>& chain) const;
    const std::vector<int32_t>& history() const { return hist_; }
    size_t states() const { return st_.size(); }
    void clear();

private:
    struct State { int32_t link = -1; uint32_t len = 0; int64_t first_end = -1; };
    int32_t next(int32_t s, int32_t tok) const;
    void set_next(int32_t s, int32_t tok, int32_t to);
    static uint64_t key(int32_t s, int32_t tok) { return (static_cast<uint64_t>(static_cast<uint32_t>(s)) << 32) | static_cast<uint32_t>(tok); }

    std::vector<State> st_;
    std::unordered_map<uint64_t, int32_t> tr_;   // (state, token) -> state
    std::vector<std::vector<int32_t>> out_;      // outgoing tokens per state (for cloning)
    int32_t last_ = 0;
    std::vector<int32_t> hist_;
};

enum class DraftSource : uint8_t { kSuffix = 0, kNeural = 1 };

struct CascadeConfig {
    uint32_t min_match = 4;   // suffix tokens that must match before the GPU drafter is skipped
    uint32_t min_chain = 4;   // continuation tokens the suffix must offer
    uint32_t max_draft = 8;
};

struct CascadeDecision {
    DraftSource source = DraftSource::kNeural;
    std::vector<int32_t> draft;  // filled for kSuffix
    Match match;
};

class CascadeHybrid {
public:
    explicit CascadeHybrid(CascadeConfig cfg = {}) : cfg_(cfg) {}
    void reserve(size_t tokens) { idx_.reserve(tokens); }
    void commit(int32_t token) { idx_.push(token); }
    void commit(const std::vector<int32_t>& tokens) { idx_.extend(tokens); }
    CascadeDecision decide() const;
    const SuffixIndex& index() const { return idx_; }
    uint64_t suffix_cycles = 0, neural_cycles = 0;
    CascadeDecision decide_and_count() { auto d = decide(); (d.source == DraftSource::kSuffix ? suffix_cycles : neural_cycles)++; return d; }

private:
    CascadeConfig cfg_;
    SuffixIndex idx_;
};

} // namespace vegajet::sfx
