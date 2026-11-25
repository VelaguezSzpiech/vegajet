// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// DFlash2 host-side parity diagnostics (Issue #152: engine acceptance far below the reference drafter's).
//
// An independent CPU reference for the four places where an engine-side DFlash drafter silently diverges from the
// z-lab reference (tools/dflash_ref is the full-model reference; this module is the linkable, unit-tested core
// that the engine, tests and tools/dflash_parity_diag.py share the conventions of):
//
//  1. hidden-state extraction: the drafter conditions on target layers [2, 8, 15, 21, 27, 33, 40, 46]. A VJHS dump
//     state s is the residual stream ENTERING target layer s (state 0 = embeddings, HF hidden_states[s]), so tap L
//     is state L + shift with shift 0. gather_taps() applies a shift; score_tap_shift() scores candidate shifts
//     against an observed drafter input and names the off-by-one;
//  2. hidden_norm + fc: c = hidden_norm(fc(concat_k tap_k)) (z-lab order). project_context() implements the four
//     orders seen in ports, classify_norm_order() says which one an observed context vector used;
//  3. masks: block of B tokens [anchor, MASK x (B-1)] at positions p..p+B-1 over context 0..p-1. Sliding-window
//     layers (window W = 4096) are causal over [context | block] in z-lab (key <= query), full-attention layers
//     are block-bidirectional; ports commonly make every layer bidirectional or use q - k <= W instead of
//     q - k < W. build_mask() / diff_mask() name the variant an observed mask matches;
//  4. acceptance: per verify cycle, accepted = leading drafts equal to the target's greedy tokens. compare_cycles()
//     aligns engine and reference cycles by anchor position, computes both means, applies the Issue #152 gate
//     (>= 4.0 PASS, < 3.5 ABORT, else MARGINAL on B = 8) and locates the first cycle / slot where the engine's
//     drafts leave the reference's; first_layer_divergence() then names the drafter layer whose hidden state
//     first exceeds a tolerance at that cycle.
// Pure functions on caller-owned float32 buffers; no GPU, no file I/O.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::dflash_host {

constexpr int kDefaultTaps[8] = {2, 8, 15, 21, 27, 33, 40, 46};
constexpr int kDefaultWindow = 4096;
constexpr double kGatePass = 4.0;   // Issue #152 gate on B = 8: mean accepted drafts per full cycle
constexpr double kGateAbort = 3.5;

// ---- 1. taps ---------------------------------------------------------------------------------------------------
// states: [n_states][D] for one token. Writes concat_k states[taps[k] + shift] into out ([taps.size() * D]).
bool gather_taps(const float* states, int n_states, int D, const std::vector<int>& taps, int shift, float* out,
                 std::string* err);

struct ShiftScore {
    int best_shift = 0;
    std::vector<int> shifts;        // candidates scored
    std::vector<double> rel_err;    // relative L2 error per candidate (inf if out of range)
    bool unambiguous = false;       // best is exact (rel_err < 1e-6) and every other candidate is not
};
// states: [n_tokens][n_states][D]; observed: [n_tokens][taps.size() * D] (what the engine fed its fc).
ShiftScore score_tap_shift(const float* states, int n_tokens, int n_states, int D, const std::vector<int>& taps,
                           const float* observed, const std::vector<int>& candidates = {-1, 0, 1});

// ---- 2. fc + hidden_norm ---------------------------------------------------------------------------------------
enum class NormOrder {
    FcNorm,     // hidden_norm(fc(cat))            z-lab / reference
    NormFc,     // fc(cat_k(rms(tap_k)))           per-tap norm before fc (norm weights per chunk, unit if absent)
    NormCatFc,  // fc(hidden_norm_8D(cat))         one norm over the concatenation (unit weights)
    Fc,         // fc(cat)                         norm forgotten
};
const char* norm_order_name(NormOrder o);
void rms_norm(const float* x, const float* w, int n, float eps, float* y);  // w == nullptr: unit weights
// cat: [n_taps * D]; fc: [D][n_taps * D] row-major (out = fc * cat); norm_w: [D] (hidden_norm weights).
void project_context(const float* cat, const float* fc, const float* norm_w, int D, int n_taps, float eps,
                     NormOrder order, float* out);

struct OrderScore {
    NormOrder best = NormOrder::FcNorm;
    std::vector<std::pair<NormOrder, double>> rel_err;  // every order, in enum order
};
OrderScore classify_norm_order(const float* cat, const float* fc, const float* norm_w, int D, int n_taps, float eps,
                               const float* observed);

// ---- 3. masks --------------------------------------------------------------------------------------------------
enum class BlockMask { ZLab, Bidirectional, Causal };
enum class WindowRule { Lt, Le };  // context key k visible on an SWA layer iff q - k < W (Lt, z-lab) or <= W (Le)
const char* block_mask_name(BlockMask m);
const char* window_rule_name(WindowRule r);

struct MaskSpec {
    int block_start = 0;  // p: position of the anchor (block slot 0); context is positions 0..p-1
    int block_len = 8;    // B
    bool swa_layer = false;
    int window = kDefaultWindow;  // <= 0: no window
    BlockMask mode = BlockMask::ZLab;
    WindowRule rule = WindowRule::Lt;
};
// Visibility of key position k for the query at block slot i (absolute position p + i).
bool mask_allowed(const MaskSpec& s, int slot, int key_pos);
// [B][n_keys] (n_keys = p + B: context then block), 1 = visible.
std::vector<uint8_t> build_mask(const MaskSpec& s);

struct MaskDiff {
    int mismatches = 0;
    int first_slot = -1, first_key = -1;         // first mismatch in row-major order, -1 if none
    bool matched_variant = false;                 // observed equals one of the scored variants
    BlockMask variant_mode = BlockMask::ZLab;
    WindowRule variant_rule = WindowRule::Lt;
    std::string summary;
};
// Compares observed ([B][p + B]) against the spec, and if they differ, names the variant (mode x window rule) the
// observed mask equals.
MaskDiff diff_mask(const MaskSpec& expected, const std::vector<uint8_t>& observed);

// Reference masked attention for one head: q [B][hd], k/v [n_keys][hd], mask [B][n_keys] -> out [B][hd].
void masked_attention(const float* q, const float* k, const float* v, const std::vector<uint8_t>& mask, int B,
                      int n_keys, int hd, float* out);

// ---- 4. acceptance ---------------------------------------------------------------------------------------------
struct Cycle {
    int pos = 0;                  // anchor position (the last committed token)
    std::vector<int32_t> draft;   // draft tokens for positions pos+1 ..
    std::vector<int32_t> target;  // the target's greedy tokens for the same positions (may be shorter)
};
int accepted_prefix(const Cycle& c);  // leading draft tokens equal to target

enum class Gate { Pass, Marginal, Abort, NoData };
const char* gate_name(Gate g);
Gate gate_of(double mean_accepted, int full_cycles);

struct CycleDiff {
    int pos = 0;
    int engine_accepted = 0, reference_accepted = 0;
    int first_slot = -1;  // first slot where engine draft != reference draft (-1: identical over the shorter)
};
struct AcceptanceReport {
    int block_len = 0;          // B (draft length + 1)
    int engine_cycles = 0, reference_cycles = 0, aligned = 0;
    int engine_full = 0, reference_full = 0;  // cycles with a full target window (only those enter the means)
    double engine_mean = 0.0, reference_mean = 0.0;
    Gate engine_gate = Gate::NoData, reference_gate = Gate::NoData;
    bool flagged = false;       // engine mean < kGateAbort
    int first_divergent_cycle = -1;  // index into `cycles` below
    std::vector<CycleDiff> cycles;   // aligned cycles, by position
    std::vector<int> engine_hist;    // accepted = 0..B-1 over full engine cycles
    std::string summary;
};
// block_len = B. Cycles are aligned by pos (positions missing on either side are counted, not compared).
AcceptanceReport compare_cycles(const std::vector<Cycle>& engine, const std::vector<Cycle>& reference, int block_len);

struct LayerDivergence {
    int layer = -1, slot = -1;       // first (layer, slot) with rel_err > tol, -1 if none
    double rel_err = 0.0;            // at that point
    std::vector<double> layer_max;   // max rel_err per layer over slots
};
// engine / reference: [n_layers][B][D] hidden states of one cycle (e.g. the block after each drafter layer).
LayerDivergence first_layer_divergence(const float* engine, const float* reference, int n_layers, int B, int D,
                                       double tol);

double rel_l2(const float* a, const float* b, int n);  // ||a - b|| / max(||b||, 1e-30)

}  // namespace vegajet::dflash_host
