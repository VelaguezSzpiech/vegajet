// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
// Ornith adapter for the generic N-rank executor (v1.3 M4).
//
// The model adapter is the only place that knows Ornith's geometry: it turns the plugin's
// own repacked tensor plan into a model-agnostic capacity spec, derives the executor layout
// for an owner-ordered device list, and names every feature that only the TP=2 fast path
// implements so admission can fail closed instead of silently running a 2-rank schedule.

#include "core/tp_executor.hpp"
#include "models/ornith/ornith_gpu.hpp"
#include "models/ornith/ornith_model.hpp"

#include <string>
#include <vector>

namespace vegajet::ornith {

enum class ExecutorMode : uint32_t { LegacyFast = 0, Generic = 1 };
// TP=2 keeps the fused super-kernel fast path; every other rank count uses generic per-op.
ExecutorMode ornith_executor_mode(uint32_t tp);

// Features the requested run selects, from its options and the environment.
tp::FeatureFlags ornith_feature_flags(const OrnithGpuOptions& options);

// Empty when `tp` can run this feature set; otherwise one clause per feature that cannot.
std::string ornith_admit(uint32_t tp, const tp::FeatureFlags& features);
// Extra runner switches have no generic implementation; never silently ignore them.
std::string ornith_raw_admit(const OrnithGpuOptions& options);
// Live raw path: one row range is reused at EVERY layer/expert, never a
// flattened pool of unrelated tensors. Head-index lists are not contiguous.
struct RawScratchBytes {
    uint64_t hidden = 3072 * 4, xq = 3072, xd = 96 * 8;
    uint64_t proj = 32768 * 4, conv_out = 12288 * 4, gates = 64 * 4;
    uint64_t aq = 8192, ad = 256 * 8, attn = 8192 * 4;
    uint64_t logits = 0, shared_gu = 0, shared_q = 0, shared_ds = 0;
    uint64_t moe_q = 0, moe_ds = 0, head_map = 0;
};
struct RawRank {
    RawScratchBytes scratch;
    shard::Range ff, shared_ff, vocab;
    tp::HeadMap attn;
    std::vector<uint32_t> groups, v_heads, conv_rows, qkv_rows, q_rows, kv_rows;
    uint64_t state_bytes = 0, conv_bytes = 0, kv_bytes = 0;
};
struct RawLayout {
    tp::ShardPlan shards;
    std::vector<RawRank> ranks;
    std::vector<size_t> rank_order;
};
RawLayout raw_layout(const Hparams& hp, uint32_t context,
                     const std::vector<tp::DeviceProfile>& devices,
                     const tp::ShardOptions& options = {});
std::vector<uint32_t> expand_heads(const std::vector<uint32_t>& heads, uint32_t width);

} // namespace vegajet::ornith
