#pragma once
#include "core/universal_ir.hpp"
#include "core/onboard_shard.hpp"
#include <string>
#include <vector>

namespace universal {
// Offline, single-token FP32 execution of a two-layer compiler graph. Synthetic
// non-dyadic parameters are deterministic; this is not a model-weight loader.
struct TinyExecutionResult {
    std::vector<float> output, reference, state, reference_state;
    std::vector<uint32_t> routing_ids, reference_routing_ids;
    size_t emitted_ops = 0, checked_values = 0;
    float max_abs_error = 0;
};
bool run_tiny_execution(const Model& model, TinyExecutionResult& result,
                        std::string* error = nullptr);
// Synthetic DenseGQA graph execution only; other family/device matrix cells are
// capacity-only. QKV is replicated, GQA heads and FFN intermediate rows are split.
// The Q4/KV2 fixture leaves one attention rank idle at TP3, but splits FFN on all ranks.
struct ShardedTinyResult {
    TinyExecutionResult numeric;
    std::vector<size_t> rank_ops, ffn_begin, ffn_end, q_heads;
    size_t host_reductions = 0, attention_gathers = 0;
};
bool run_sharded_tiny_execution(const Model&, const onboard::Plan&,
                                ShardedTinyResult&, std::string* error = nullptr,
                                bool wrong_slice_control = false);
} // namespace universal
