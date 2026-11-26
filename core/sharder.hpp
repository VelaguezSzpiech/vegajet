// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Bandwidth-proportional asymmetric tensor parallelism. Work is apportioned in b=32-row units by the
// largest-remainder (Hamilton) method so faster GPUs get proportionally more rows, experts or heads.
#pragma once

#include <cstdint>
#include <vector>

namespace vegajet::shard {

constexpr uint32_t kRowUnit = 32;  // b = 32 quant-block aligned row granularity

// Largest-remainder apportionment of `units` across weights (ties -> lower rank). Sum is exact.
std::vector<uint64_t> apportion(uint64_t units, const std::vector<double>& weights);

struct Range { uint64_t begin = 0, end = 0; uint64_t size() const { return end - begin; } };

// Splits `rows` into contiguous per-GPU ranges, multiples of `row_unit` (default kRowUnit=32; pass 16 for
// TP=3 expert_ff=512 balance) except the last non-empty one.
std::vector<Range> split_rows(uint64_t rows, const std::vector<double>& bw, uint32_t row_unit = kRowUnit);
// Source Gate/Up tensors are [expert][full_ff][D], never [rank][expert][local_ff][D].
constexpr uint64_t expert_row_index(uint64_t expert, uint32_t full_ff, uint32_t begin, uint32_t row) {
    return expert * full_ff + begin + row;
}
// Two nonempty, 32-row-aligned intermediate shards; default retains TP=2 halves.
std::vector<Range> split_expert_intermediate(uint32_t rows, const std::vector<double>& bw, bool enabled);

// Splits whole experts (MoE expert parallel) across GPUs.
std::vector<Range> split_experts(uint64_t n_experts, const std::vector<double>& bw);

enum class AttnShardMode : uint32_t {
    HeadParallel = 0,     // q heads split, each GPU holds the KV heads its q heads need
    SequenceSharded = 1,  // MLA / single latent KV: KV sequence split across GPUs (split-KV decoding)
};

struct AttnGpuShard {
    Range q_heads;           // [begin, end) query heads
    Range kv_heads;          // [begin, end) KV heads needed by those query heads
    uint32_t replicated_kv = 0;  // KV heads also held by another GPU (GQA group straddles a boundary)
    double seq_fraction = 0.0;   // SequenceSharded: fraction of the KV sequence owned
};

struct AttnShardPlan {
    AttnShardMode mode = AttnShardMode::HeadParallel;
    uint32_t n_q_heads = 0, n_kv_heads = 0, group = 0;
    std::vector<AttnGpuShard> gpus;
    uint32_t total_kv_heads_stored() const;
};

// GQA: q heads are apportioned by bandwidth; a KV head whose query group straddles a GPU boundary is
// replicated on both GPUs. is_mla (or n_kv_heads == 1 with more than one GPU) selects sequence sharding.
AttnShardPlan plan_attention(uint32_t n_q_heads, uint32_t n_kv_heads, const std::vector<double>& bw,
                             bool is_mla = false);

// Gated-delta-net heads: K groups are the unit (each owns n_v_heads / n_k_groups value heads).
struct GdnShard { Range k_groups; Range v_heads; };
std::vector<GdnShard> plan_gdn(uint32_t n_k_groups, uint32_t n_v_heads, const std::vector<double>& bw);

} // namespace vegajet::shard
