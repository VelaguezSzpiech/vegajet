// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once

#include "core/sharder.hpp"
#include <map>
#include <string>
#include <vector>

namespace vegajet::tp {

// Rank order is supplied by the owner, never inferred from persistent ordinals.
struct DeviceProfile {
    std::string unique_id, bdf;
    double bw_hbm_gbs = 0, pcie_link_gbs = 0, mclk = 0;
    uint64_t usable_vram_bytes = 0;
    uint32_t n_cu = 0;
    int numa_node = -1;
    std::string key() const;
};
// Read-only sysfs identity; missing/zero unique_id falls back to full canonical BDF.
std::string device_key(const std::string& bdf, const std::string& sysfs_root = "/");
// Accepts hw_profile.json's schema-1 gpus array and aliases; no eight-card cap.
// usable_vram_bytes is authoritative; otherwise reserve is deducted from vram_bytes.
std::vector<DeviceProfile> profiles_from_json(const std::string& text, uint64_t reserve_bytes = 0);

enum class RowKind { Dense, MoeIntermediate, Vocabulary };
struct RowSpec {
    std::string name;
    RowKind kind = RowKind::Dense;
    uint64_t rows = 0, bytes_per_row = 0;
};
struct AttentionSpec {
    std::string name;
    uint32_t q_heads = 0, kv_heads = 0;
    // Projection, cache and state bytes for the requested context, after packing.
    uint64_t q_bytes_per_head = 0, kv_bytes_per_head = 0;
};
struct ModelSpec {
    uint64_t replicated_bytes = 0; // all other persistent/scratch/reserve costs per rank
    std::vector<RowSpec> rows;
    std::vector<AttentionSpec> attention;
};
struct AllocationOverride {
    std::vector<double> fractions;
    std::vector<uint64_t> rows; // attention: Q-head counts; whole GQA groups unless replication enabled
};
struct ShardOptions {
    uint32_t row_tile = 32;
    uint64_t split_k_threshold = 240;
    bool replicate_kv = false;
    std::map<std::string, AllocationOverride> tensors;
};
// Strict, transactional JSON parsing: unknown keys, duplicates and conflicting overrides fail.
ShardOptions shard_options_from_json(const std::string& text);
struct RowShard {
    shard::Range rows;
    uint64_t padded_rows = 0, bytes = 0;
    bool recommend_split_k = false; // zero-work ranks never launch a kernel
};
struct HeadMap {
    std::vector<uint32_t> q_heads, kv_heads, q_to_local_kv;
    uint64_t bytes = 0;
};
struct RankShard {
    std::string device_key;
    uint64_t used_bytes = 0;
    std::vector<RowShard> rows;
    std::vector<HeadMap> attention;
};
struct ShardPlan {
    uint32_t schema = 1;
    std::vector<RankShard> ranks;
};
// Throws invalid_argument / overflow_error / runtime_error with capacity diagnostics.
// All byte costs are caller-supplied model semantics, not guessed GGUF storage costs.
ShardPlan plan_shards(const ModelSpec& model, const std::vector<DeviceProfile>& devices,
                      const ShardOptions& options = {});

} // namespace vegajet::tp
