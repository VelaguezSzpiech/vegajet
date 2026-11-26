// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Deterministic CPU reference oracles in FP64, with FP32/FP16 comparison helpers.
//
// Purpose: a small, dependency-free, reproducible yardstick for the two non-obvious numerical devices in the
// engine, so a GPU result can be blamed on a device path instead of on the reference:
//
//  1. GDN multi-token verify (ops/kernels/vj_spec_gdn.hip). Given M draft tokens (5..8 here), one verify pass
//     must produce the per-token outputs AND the recurrent state after accepting any prefix k = 0..M without
//     re-running the recurrence, and every element of the state it returns is written exactly once
//     ("single write"). The oracle:
//       * computes the sequential M = 1 recurrence in FP64 (the definition);
//       * computes the chunked WY verify form in one pass in FP64 (chain: only j <= t contributes);
//       * checks the chunk result equals the sequential state after every k, for k = 0..M, and that the final
//         state is written once per element;
//       * re-runs the same verify arithmetic with every input and every intermediate rounded to FP32 and to
//         FP16 and reports the max abs / rel error against the FP64 sequential reference, per precision.
//
//  2. MoE EVICT expert-union deduplication (core/evict_controller.h: U(S) counts unique (layer, expert) pairs).
//     A deduplicated pass must return bitwise-equal token outputs and load every unique expert once, where
//     "load" is the host-side fetch/repack that dominates the pass. The oracle compares the deduplicated
//     evaluation with the naive per-token evaluation and reports unique vs. occurrence counts.
//
// Everything is a pure function of its arguments and a seed; no I/O, no threads, no floating-point environment
// changes. Errors are max abs and max rel (|ref - got| / max(|ref|, floor)) over the compared values.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace vegajet::oracles {

enum class Precision : uint8_t { F64 = 0, F32 = 1, F16 = 2 };
const char* precision_name(Precision p);

// ---- FP16 helpers (round-to-nearest-even, subnormals, inf/nan) ----------------------------------------------
uint16_t f32_to_f16(float f);
float f16_to_f32(uint16_t h);

// ---- error accounting ---------------------------------------------------------------------------------------
struct ErrStats {
    size_t n = 0;            // compared elements
    double max_abs = 0.0;
    double max_rel = 0.0;    // elementwise, floored at 1e-12 (near-zero elements dominate this)
    size_t at_abs = 0;       // index of the max abs error
    size_t at_rel = 0;       // index of the max rel error
    double num_sq = 0.0, den_sq = 0.0;  // for the L2 relative error (stable for near-zero elements)
    void add(double ref, double got, double floor = 1e-12);
    void merge(const ErrStats& o);
    double rel_l2() const { return den_sq > 0.0 ? std::sqrt(num_sq / den_sq) : (num_sq > 0.0 ? INFINITY : 0.0); }
};
// Elementwise |ref-got| and |ref-got|/max(|ref|,floor).
ErrStats compare(const double* ref, const double* got, size_t n, double floor = 1e-12);
std::string format_err(const char* label, const ErrStats& e);

// ---- 1. GDN single-write multi-token verify -----------------------------------------------------------------
struct GdnProblem {
    uint32_t dk = 0, dv = 0, M = 0;
    std::vector<double> S0;   // dv x dk, row-major
    std::vector<double> q, k, v;  // M x dk, M x dk, M x dv (k rows L2-normalized)
    std::vector<double> a, b;     // M
};

// Deterministic problem: k rows L2-normalized, a in [0.85, 0.999], b in [0.05, 0.95].
GdnProblem gdn_make(uint32_t dk, uint32_t dv, uint32_t M, uint64_t seed);

struct GdnVerify {
    std::vector<double> out;                    // M x dv
    std::vector<std::vector<double>> state;     // (M+1) x dv*dk: state[0] = S0, state[k] = after accepting k
    std::vector<uint32_t> writes;               // dv*dk counters: how often the final state was written
};
// Sequential M = 1 recurrence in FP64 ("out" optional; state[k] = state after k tokens).
std::vector<std::vector<double>> gdn_sequential(const GdnProblem& p, std::vector<double>* out = nullptr);

// Chunked WY verify in one pass (FP64 unless a precision is given): outputs for all M tokens plus the state
// after every accepted prefix, in a single pass over the chunk.
GdnVerify gdn_verify(const GdnProblem& p, Precision prec = Precision::F64);

struct GdnPrecisionReport {
    Precision prec = Precision::F32;
    ErrStats out_err;    // outputs vs FP64 sequential
    ErrStats state_err;  // states after k = 0..M vs FP64 sequential
};
struct GdnOracleReport {
    uint32_t dk = 0, dv = 0, M = 0;
    ErrStats algo_f64;              // FP64 verify vs FP64 sequential (algorithm equivalence)
    std::vector<GdnPrecisionReport> by_precision;  // F32, F16
    size_t state_writes_min = 0, state_writes_max = 0;  // final-state write counters
    bool single_write = false;
    std::string summary;
};
GdnOracleReport gdn_verify_oracle(uint32_t dk, uint32_t dv, uint32_t M, uint64_t seed);

// Worst error per precision over M = 5..8 with two (dk, dv) shapes; prints one line per precision.
struct GdnSweepReport {
    std::vector<GdnPrecisionReport> worst;  // F32, F16
    ErrStats algo_f64;
    size_t cases = 0;
    bool single_write = false;
    std::string summary;
};
GdnSweepReport gdn_verify_sweep(uint64_t seed = 2026);

// ---- 2. MoE EVICT expert-union dedup ------------------------------------------------------------------------
// One token's routing: for every routed expert, (layer, expert) and its gate weight. `x` is the token's input.
struct EvictProblem {
    uint32_t n_layers = 0, n_experts = 0, dim = 0, M = 0;
    std::vector<std::vector<uint32_t>> route_layer, route_expert;  // M x top_k (may differ per token)
    std::vector<std::vector<double>> weight;                       // M x top_k
    std::vector<std::vector<double>> x;                            // M x dim
};
// Deterministic routing: per token, top_k distinct experts per layer drawn without replacement.
EvictProblem evict_make(uint32_t n_layers, uint32_t n_experts, uint32_t dim, uint32_t M, uint32_t top_k,
                        uint64_t seed);

struct EvictRun {
    std::vector<std::vector<double>> out;                     // M x dim
    std::vector<std::pair<uint32_t, uint32_t>> unique_order;  // first-seen order of (layer, expert)
    uint64_t expert_loads = 0;                                // weight materializations (fetch/repack count)
};
// Per-token expert evaluation: every occurrence materializes its expert weights.
EvictRun evict_naive(const EvictProblem& p);
// Expert-union pass: each unique (layer, expert) is materialized once and reused; same accumulation order per
// token, so the outputs are bitwise equal.
EvictRun evict_dedupe(const EvictProblem& p);

struct EvictOracleReport {
    ErrStats err;
    bool bitwise_equal = false;
    uint64_t occurrences = 0, unique = 0, naive_loads = 0, dedupe_loads = 0;
    std::string summary;
};
EvictOracleReport evict_oracle(const EvictProblem& p);

}  // namespace vegajet::oracles
