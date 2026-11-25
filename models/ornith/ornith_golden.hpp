// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Tier-1 llama.cpp greedy references for Ornith-122B-AD-D4 (temperature 0, top_k 1, no BOS, 32 new tokens),
// recorded from llama-server /completion with the prompt given as token ids on the same GGUF.
#pragma once

#include <cstdint>
#include <vector>

namespace vegajet::ornith::golden {

// "def fibonacci(n):"  -> gating reference: the bench requires 32/32 exact token match.
inline const std::vector<int32_t> kCodePrompt = {727, 73111, 1393, 1590};
inline const std::vector<int32_t> kCodeRef32 = {
    198, 262, 413, 307, 2564, 220, 15, 25, 198, 285, 460, 2958, 198, 262, 4265, 307,
    606, 220, 16, 25, 198, 285, 460, 498, 15, 60, 198, 262, 4265, 307, 606, 220};

// "The capital of France is" -> informational only (reported as matched-prefix length, not gated).
inline const std::vector<int32_t> kProsePrompt = {760, 6511, 314, 9338, 369};
inline const std::vector<int32_t> kProseRef32 = {
    11751, 13, 3437, 369, 279, 6511, 314, 9564, 30, 271, 248068, 198, 760, 1156, 369, 9859,
    883, 279, 6511, 314, 9564, 13, 1061, 369, 264, 4145, 57879, 3296, 13, 561, 6511, 314};

// Number of leading tokens of `out` equal to `ref`.
inline size_t matched_prefix(const std::vector<int32_t>& out, const std::vector<int32_t>& ref) {
    size_t n = 0;
    while (n < out.size() && n < ref.size() && out[n] == ref[n]) ++n;
    return n;
}

} // namespace vegajet::ornith::golden
