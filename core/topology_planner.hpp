// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Universal topology planner for 1..8 GPUs on one node (16/32 GiB cards, 60/64 CU).
//
// The planner answers one question: given the hardware profile and a model footprint, how should the model be
// served? It scores every allowed TP degree (1..4) over every disjoint grouping of the visible GPUs and keeps
// the grouping with the highest aggregate decode throughput.
//
//   * TP=1 on one GPU: the all-reduce is elided. There is no phantom collective: ar_bytes_per_token and ar_ms
//     are exactly 0, which is the N=1 zero-overhead path.
//   * TP=2/3/4: the model is split proportionally to measured streaming bandwidth. Each member's share is
//     its calibrated bandwidth fraction, so every member streams for the same wall time; the all-reduce cost
//     is added per token from the P2P matrix (min link bandwidth in the group, worst per-message latency).
//   * Capacity is a hard constraint: a member that cannot hold its proportional share is pinned at
//     vram - reserve and the remainder is redistributed over the other members. When a group cannot hold the
//     footprint, it is rejected rather than overcommitted.
//   * Multi-instance Swarm: several independent replicas, each owning a disjoint set of GPUs. When the model
//     fits on fewer cards than are visible, the leftover cards raise aggregate throughput through more
//     replicas instead of an ever-wider TP group. A swarm of TP groups is allowed when capacity forces TP.
//
// The cost model is decode-oriented (memory bound): per-token work is `token_bytes` read once per replica,
// split by the member shares; all-reduce payload defaults to layers * hidden * 2 B (FP16) for reduce-scatter
// plus all-gather, moved as a ring ((tp-1)/tp of the payload per GPU). It is a planning model, not a kernel
// measurement: it decides placement, and never claims achieved tokens/s.
#pragma once

#include "core/hw_profile.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::topo {

constexpr uint32_t kTpCap = 4;  // TP=2/3/4 sharding supported; 1 is the single-GPU path

enum class Mode : uint8_t {
    Insufficient = 0,  // the model does not fit any allowed topology
    SingleGpu,         // one replica on one GPU (N=1, or one idle-card replica)
    TensorParallel,    // one replica spread over 2..4 GPUs
    Swarm,             // more than one replica
};
const char* mode_name(Mode m);

struct ModelFit {
    std::string name;
    uint64_t resident_bytes = 0;  // weights + persistent state held by one replica
    uint64_t kv_bytes = 0;        // KV / activation cache at the target context
    uint64_t token_bytes = 0;     // streamed per decode token; 0 uses resident_bytes
    uint32_t layers = 0;          // all-reduce sizing when ar_bytes_per_token is 0
    uint32_t hidden = 0;
    uint64_t ar_bytes_per_token = 0;  // explicit FP16 all-reduce payload per token (0: derive)
};

struct LinkPolicy {
    double host_gbs = 10.0;          // effective bandwidth for a pair without direct P2P
    double host_latency_us = 30.0;   // per message through host memory
    double min_link_gbs = 1.0;       // groups whose slowest link is below this are rejected
    bool require_direct_p2p = false; // reject a TP group if any pair is host-bounced
};

struct PlanConfig {
    uint64_t reserve_bytes = 2ull << 30;  // per GPU, excluded from usable capacity
    uint32_t max_tp = kTpCap;             // highest TP degree considered
    uint32_t force_tp = 0;                // 0: search all degrees; 1..4: only this degree
    bool allow_tp = true;
    bool allow_swarm = true;
    uint32_t max_instances = 0;           // 0: unlimited
    LinkPolicy links;
    double ar_latency_us = 6.0;           // per collective phase when a link has no measured latency
};

struct Check {
    std::string name;
    bool ok = true;
    std::string detail;
};

struct InstancePlan {
    std::vector<uint32_t> gpus;      // positions in hw::Profile::gpus, ascending
    uint32_t tp = 1;
    std::vector<double> share;       // fraction of the replica per member, sums to 1
    std::vector<uint64_t> resident_bytes;
    std::vector<uint64_t> kv_bytes;
    bool host_bounced = false;       // at least one ordered pair used host memory
    bool p2p_complete = true;        // every pair in direct()
    double effective_link_gbs = 0.0; // slowest link used by the all-reduce (0 for TP=1)
    uint64_t ar_bytes = 0;           // model-level payload per token (not the per-GPU volume)
    double compute_ms = 0.0;         // per replica per token, slowest member
    double ar_ms = 0.0;
    double per_token_ms = 0.0;
    double tps = 0.0;                // 1000 / per_token_ms
};

struct Plan {
    bool ok = false;
    Mode mode = Mode::Insufficient;
    uint32_t tp = 1;
    uint32_t instances = 0;
    bool allreduce = false;
    uint64_t ar_bytes_per_token = 0;
    double compute_ms = 0.0;   // worst instance
    double ar_ms = 0.0;        // worst instance
    double per_token_ms = 0.0; // worst instance
    double aggregate_tps = 0.0;
    double single_gpu_ms = 0.0;  // fastest single GPU that fits (0 when none fits)
    double speedup = 0.0;        // single_gpu_ms / per_token_ms (0 when no single-GPU reference)
    bool p2p_complete = true;
    std::vector<InstancePlan> groups;
    std::vector<Check> checks;
    std::string detail;

    bool allreduce_elided() const { return !allreduce; }
    std::string summary() const;  // deterministic multi-line report
};

Plan plan(const hw::Profile& hw, const ModelFit& model, const PlanConfig& cfg = {});
// Same search restricted to one TP degree (a plan with several groups is still a swarm).
Plan plan_tp(const hw::Profile& hw, const ModelFit& model, uint32_t tp, const PlanConfig& cfg = {});

} // namespace vegajet::topo
