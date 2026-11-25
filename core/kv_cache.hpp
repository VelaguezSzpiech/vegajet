// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Minimal KV footprint calculator and mixed-tier paged KV planner (Q8_0 / Q4_0 / TQ3).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::kv {

enum class KvTier : uint32_t { F16 = 0, Q8_0 = 1, Q4_0 = 2, TQ3 = 3 };
const char* tier_name(KvTier t);
// Storage bytes per element including per-32 FP16 scales (F16 2, Q8_0 34/32, Q4_0 18/32, TQ3 14/32).
double tier_bytes_per_elem(KvTier t);

constexpr uint32_t kPageTokens = 256;

struct KvGeometry {
    uint32_t n_layers = 0;    // attention layers holding KV on this GPU
    uint32_t n_kv_heads = 0;  // KV heads stored on this GPU (after GQA replication)
    uint32_t head_dim_k = 0, head_dim_v = 0;
    uint64_t elems_per_token() const { return uint64_t(n_layers) * n_kv_heads * (head_dim_k + head_dim_v); }
};

uint64_t kv_bytes(const KvGeometry& g, uint64_t tokens, KvTier tier);  // page-rounded

struct KvTierSpan { KvTier tier; uint64_t tokens; uint64_t pages; uint64_t bytes; };

struct KvPlan {
    bool fits = false;
    std::vector<KvTierSpan> spans;  // newest first: recent window, then older history
    uint64_t total_bytes = 0;
    std::string summary() const;
};

// Keeps the most recent `recent_tokens` in Q8_0; older history uses the best tier that fits the budget
// (Q8_0 -> Q4_0 -> TQ3). fits=false if even TQ3 history does not fit.
KvPlan plan_kv(const KvGeometry& g, uint64_t context_tokens, uint64_t budget_bytes, uint64_t recent_tokens = 4096);

} // namespace vegajet::kv
