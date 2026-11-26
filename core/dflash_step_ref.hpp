// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// FP64 reference for one DFlash drafter step, and a per-layer hidden-state diff over a configurable tap list
// (default [2, 8, 15, 21, 27, 33, 40, 46]).
//
// A single drafter step conditions on the target model's residual stream at the tap layers:
//   cat  = concat_k states[layer_k]                       (FP64)
//   c    = hidden_norm(fc(cat))                           (z-lab order: fc, then RMSNorm)
//   q_b  = Wq (x_b + c),  k_b = Wk x_b,  v_b = Wv x_b     (one head, block slot b = 0..B-1)
//   mask = full-attention layer: every [context | block] key visible, block is bidirectional;
//          sliding-window layer: causal and windowed over absolute positions, key k visible for query q iff
//          k <= q and q - k < W (strict: z-lab uses q - k < W, not <=).
//   o_b  = softmax(q_b . k / sqrt(hd)) v      over the visible keys
//
// diff() compares two dumps of per-tap residual states ([n_states][d] per token), the contexts built from them
// (when fc / hidden_norm weights are given) and the mask of every layer against the reference, and names the
// first tap layer whose relative error exceeds the tolerance.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::dflash_step {

constexpr int kDefaultTaps[8] = {2, 8, 15, 21, 27, 33, 40, 46};
constexpr int kDefaultWindow = 4096;

struct Config {
    std::vector<int> layers{kDefaultTaps[0], kDefaultTaps[1], kDefaultTaps[2], kDefaultTaps[3],
                            kDefaultTaps[4], kDefaultTaps[5], kDefaultTaps[6], kDefaultTaps[7]};
    int n_states = 48;   // residual states in the dump (state 0 = embeddings)
    int d = 0;           // hidden size
    int hd = 0;          // attention head size
    int block = 8;       // draft block length B
    int pos = 0;         // absolute position of block slot 0 (context is 0..pos-1)
    int window = kDefaultWindow;
    double eps = 1e-6;   // RMSNorm epsilon
};

// Validates: ascending unique taps inside [0, n_states), d > 0, hd > 0, block > 0, pos >= 0, window > 0.
bool validate(const Config& c, std::string* err);

// Visibility of key position k for block slot i (query position pos + i).
bool mask_allowed(const Config& c, bool swa_layer, int slot, int key_pos);
// [block][pos + block], 1 = visible.
std::vector<uint8_t> build_mask(const Config& c, bool swa_layer);

// c = hidden_norm(fc(concat taps)). states: [n_states][d]; fc: [d][n_taps * d]; norm_w: [d] (nullptr = unit).
bool build_context(const Config& c, const std::vector<double>& states, const std::vector<double>& fc,
                   const std::vector<double>* norm_w, std::vector<double>* out, std::string* err);

struct Step {
    std::vector<double> context;  // d
    std::vector<double> out;      // block * hd
    std::vector<uint8_t> mask;    // block * (pos + block)
};
// ctx_k / ctx_v: [pos][hd] context keys / values (may be empty when pos == 0).
bool single_step(const Config& c, const std::vector<double>& states, const std::vector<double>& fc,
                 const std::vector<double>* norm_w, const std::vector<double>& block_x /* block x d */,
                 const std::vector<double>& ctx_k, const std::vector<double>& ctx_v,
                 const std::vector<double>& wq, const std::vector<double>& wk, const std::vector<double>& wv,
                 bool swa_layer, Step* out, std::string* err);

struct DiffReport {
    std::vector<int> layers;
    std::vector<double> layer_max_abs, layer_max_rel;  // per tap layer, residual states
    double ctx_max_abs = 0.0, ctx_max_rel = 0.0;       // built context vs observed (0 when not compared)
    bool ctx_compared = false;
    int first_bad_layer = -1;                          // first tap whose rel error > tol
    bool mask_equal = true;                            // observed mask (if given) == reference for every layer
    int mask_mismatches = 0;
    std::string summary;
};

// ref_states / got_states: [n_states][d]. observed_ctx optional ([d]); observed_mask optional
// ([block][pos+block]); swa_layer selects the reference mask. tol is a relative error.
DiffReport diff(const Config& c, const std::vector<double>& ref_states, const std::vector<double>& got_states,
                const std::vector<double>& fc, const std::vector<double>* norm_w,
                const std::vector<double>* observed_ctx, const std::vector<uint8_t>* observed_mask, bool swa_layer,
                double tol);

}  // namespace vegajet::dflash_step
