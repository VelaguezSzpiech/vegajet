// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Pure TP=3 planner: every layer split across three gfx906 GPUs (no pipeline stages), asymmetric shares
// (default 31.7 / 34.1 / 34.1 %: the GPU behind the slower link gets less work), power-cap checks, and a P2P
// all-reduce schedule (direct all-to-all or ring) with a cost model and an exact CPU simulation.
//
// Sharding reuses core/sharder (HEAD): attention by query heads with GQA KV replication (sequence sharding for a
// single latent KV), gated-delta-net by K groups, FFN / split-FF experts by 32-row units, whole experts by count,
// LM head by vocab rows. The plan reports per-GPU weight bytes and how far the realized split is from the target
// (head granularity makes attention coarse; FFN rows are fine-grained).
//
// Power: with three cards the per-GPU cap ceiling is 150 W (the rule release/contrib/hardware/set_power_cap.sh
// enforces: 200 W for <= 2 GPUs, 150 W for 3+), floor 50 W, and optionally caps + host draw <= PSU budget.
//
// All-reduce of a row-parallel output (B bytes, partial sums on every GPU):
//   Direct: phase 0 reduce-scatter (each GPU sends chunk_j of its partial to owner j; the owner sums the three
//           partials in rank order 0,1,2), phase 1 all-gather (each owner sends its reduced chunk to both peers).
//           2 latency steps; every chunk is summed in the same order -> bit-identical on all ranks and equal to a
//           canonical ((x0 + x1) + x2).
//   Ring:   2 (N - 1) = 4 steps, each GPU sends one chunk to its successor per step. Bit-identical on all ranks
//           (each chunk is reduced once, then forwarded), but the summation order depends on the chunk.
// Balance: with split-FF experts the row unit decides the granularity. expert_ff = 512 at 32-row units is only
// 16 units -> 5 / 6 / 5, so one GPU streams 37.5 % of the expert bytes (target 34.1 %) and sets the layer time;
// 16-row units or whole-expert sharding bring it within 1 %. The balance check reports this instead of hiding it.
//
// Chunks are proportional to the GPU shares (in 64-element units). For N = 3 both algorithms move 4/3 B per GPU,
// so Direct wins whenever latency matters; the cost model also covers asymmetric links, shared egress and pairs
// without peer-to-peer access (bounced through host memory).
#pragma once

#include "core/sharder.hpp"
#include "ops/layer_plan.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::tp3 {

constexpr uint32_t kGpus = 3;

struct Link {
    double gbs = 12.0;        // sustained P2P write bandwidth, GB/s
    double latency_us = 8.0;  // per message (doorbell + DMA setup + completion)
    bool p2p = true;          // false: bounced through host memory (two transfers, host_gbs)
};

struct Tp3Config {
    std::array<double, kGpus> share = {0.317, 0.341, 0.341};
    std::array<double, kGpus> power_cap_w = {150.0, 150.0, 150.0};
    double cap_ceiling_w = 150.0;   // per-GPU ceiling with three cards
    double cap_floor_w = 50.0;
    double psu_budget_w = 0.0;      // 0: not checked
    double host_w = 0.0;            // non-GPU draw counted against the PSU budget
    std::array<double, kGpus> vram_bytes = {32e9, 32e9, 32e9};
    Link link[kGpus][kGpus];        // [src][dst]
    double host_gbs = 10.0;         // host-bounce bandwidth for non-P2P pairs
    bool shared_egress = true;      // a GPU's concurrent sends share its uplink
    double bits_dense = 4.5, bits_expert = 3.0, bits_head = 8.5;
    uint32_t row_unit = shard::kRowUnit;  // FFN / expert / vocab row granularity (HEAD's split_rows uses 32)
    double balance_tolerance = 0.03;      // max |realized - target| share before the balance check fails
};

struct Check {
    std::string name;
    bool ok = true;
    std::string detail;
};

struct GpuPlan {
    shard::Range q_heads, kv_heads, gdn_groups, ffn_rows, expert_rows, experts, vocab_rows;
    double seq_fraction = 0.0;
    double weight_bytes = 0.0;
    double realized_share = 0.0;
};

struct Plan {
    bool ok = false;
    std::vector<Check> checks;
    std::array<GpuPlan, kGpus> gpu;
    double total_weight_bytes = 0.0;
    double max_share_error = 0.0;   // max |realized - target| over GPUs (absolute fraction)
    std::string summary() const;
};

Plan plan_tp3(const ops::ModelDesc& m, const Tp3Config& cfg);
std::vector<Check> check_power(const Tp3Config& cfg);

// ---- all-reduce
enum class ArAlgo : uint8_t { Direct = 0, Ring = 1 };
const char* algo_name(ArAlgo a);

struct ArStep {
    uint32_t phase = 0;
    uint32_t src = 0, dst = 0;
    uint64_t begin = 0, end = 0;   // element range [begin, end)
    bool reduce = false;           // true: dst accumulates (Direct: staged, summed in rank order); false: copy
};

struct ArSchedule {
    ArAlgo algo = ArAlgo::Direct;
    uint64_t elems = 0;
    uint32_t elem_bytes = 2;
    uint32_t phases = 0;
    std::array<shard::Range, kGpus> chunk;  // owner ranges
    std::vector<ArStep> steps;
    double modeled_us = 0.0;
    bool canonical_order = false;           // every element summed as ((x0 + x1) + x2)
};

ArSchedule plan_allreduce(uint64_t elems, uint32_t elem_bytes, ArAlgo algo, const Tp3Config& cfg);
ArSchedule choose_allreduce(uint64_t elems, uint32_t elem_bytes, const Tp3Config& cfg);  // lower modeled time
// Executes the schedule on three float buffers (partials in, reduced result out). Returns false on a malformed
// schedule (overlapping / out-of-range steps).
bool simulate_allreduce(const ArSchedule& s, std::array<std::vector<float>, kGpus>& bufs, std::string* err = nullptr);

} // namespace vegajet::tp3
