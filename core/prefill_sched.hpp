// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// C2 macro-batch prefill scheduler (v0.1.1, host side).
//
// Dense and MoE prefill run on macro batches of M = 256 or 512 tokens; the GDN mixer advances through each macro
// batch in ordered sub-chunks of C = 16 or 32 tokens, each continuing the recurrent (and conv) state the previous
// sub-chunk left. Attention layers are causal by absolute position: a macro batch's queries see the KV of every
// earlier macro batch plus the causal prefix of their own. schedule() turns (prompt length, start position,
// requested M / C, available memory) into that ordered plan:
//   * macro batches cover [pos0, pos0 + n) contiguously, in order; the last one may be partial;
//   * GDN sub-chunks cover each macro batch contiguously, in order; the last sub-chunk of a macro batch may be
//     partial, and the recurrent state hand-off chain is explicit (state_in of sub-chunk j == state_out of j - 1,
//     across macro boundaries too);
//   * scratch comes from core/prefill_planner for the selected shapes; if it does not fit the budget, M = 512 falls
//     back to 256, and if that does not fit either (or the request is invalid) the schedule keeps the current path
//     (one dense macro of `legacy_chunk` tokens, GDN un-split), with the reason recorded.
// verify_schedule() checks every invariant above; reference_prefill() is a small exact CPU model (gated linear
// recurrence + causal softmax attention + per-token dense) used by the tests to prove the order is lossless.
//
// Pure host code: no GPU, no file I/O, no environment reads.
#pragma once

#include "core/prefill_planner.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::prefill_sched {

struct SubChunk {
    uint32_t pos0 = 0;      // absolute position of the first token
    uint32_t n = 0;         // tokens (<= C; the last of a macro batch may be shorter)
    uint32_t row0 = 0;      // row offset inside the macro batch's activation buffers
    uint32_t state_in = 0;  // recurrent state consumed: tokens absorbed so far (absolute, == pos0)
    uint32_t state_out = 0; // state produced (state_in + n); the next sub-chunk's state_in
};

struct MacroBatch {
    uint32_t index = 0;
    uint32_t pos0 = 0, n = 0;
    uint32_t kv_hi = 0;          // attention keys [0, kv_hi) exist after this batch's KV write (= pos0 + n)
    bool final_partial = false;  // n < M (only the last batch)
    bool compute_logits = false; // last batch only
    std::vector<SubChunk> gdn;
};

struct Config {
    uint32_t macro_m = 512;          // requested M: 256 or 512 (C2); anything else is rejected
    uint32_t gdn_c = 32;             // 16 or 32
    uint32_t legacy_chunk = 1024;    // --prefill-chunk of the current path
    uint64_t budget_bytes = 0;       // per-GPU bytes available for prefill scratch (0 = do not check)
    prefill::ModelDims dims = prefill::ornith_122b_tp2();
    prefill::Precision precision = prefill::Precision::kFp32Legacy;  // today's W4A8 kernels: FP32 intermediates
    bool aliased = false;            // engine aliases the GDN / attention working sets
    bool enabled = true;             // false: always the current path
};

struct Schedule {
    bool c2 = false;                 // false: current path
    std::string reason;              // why the current path was kept ("" when c2)
    uint32_t macro_m = 0, gdn_c = 0;
    uint64_t scratch_bytes = 0;      // planner bytes for the selected M (aliased or not, per Config)
    std::vector<MacroBatch> macros;
    std::string summary;
};

Schedule schedule(uint32_t prompt_len, uint32_t pos0, const Config& cfg);
// "256" / "512" (C = 32) or "256:16" / "512:32" ...; "", "0", "off" -> disabled. false + err on anything else.
bool parse_c2_spec(const std::string& text, bool* enabled, uint32_t* macro_m, uint32_t* gdn_c, std::string* err);
// Scratch bytes per GPU for macro size M and chunk C under cfg (0 on planner error).
uint64_t scratch_for(uint32_t M, uint32_t C, const Config& cfg);
bool verify_schedule(const Schedule& s, uint32_t prompt_len, uint32_t pos0, std::string* err);

// ---- exact CPU reference model (tests) -----------------------------------------------------------------------
struct ToyModel {
    uint32_t d = 8;      // width of every toy vector
    uint64_t seed = 1;
};
// Per-token outputs [n][d] of a toy block (dense -> GDN-like gated recurrence -> causal attention -> dense) for
// tokens `toks` at positions pos0.., starting from the state/KV a previous call left (state, keys, vals).
struct ToyState {
    std::vector<double> S;           // d x d recurrent state
    std::vector<std::vector<double>> keys, vals;  // attention KV by absolute position
    uint32_t absorbed = 0;           // tokens absorbed by S
};
// Token-by-token reference.
std::vector<std::vector<double>> toy_sequential(const ToyModel& m, const std::vector<int32_t>& toks, uint32_t pos0, ToyState* st);
// Schedule-driven execution: dense over each macro batch at once, GDN in chunk-parallel form per sub-chunk, causal
// attention per macro batch over the KV cache + its own causal prefix.
std::vector<std::vector<double>> toy_scheduled(const ToyModel& m, const std::vector<int32_t>& toks, const Schedule& s, ToyState* st);

}  // namespace vegajet::prefill_sched
