// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include "core/sharder.hpp"
#include <string>

namespace vegajet::shard {
// Empty/"0": unchanged Hamilton planner. "auto": discrete minimax whole-K-group split.
// Otherwise strict comma-separated V-head counts, one positive count per rank.
// V heads in GdnShard are cardinality ranges; Ornith maps group-major physical heads separately.
bool plan_gdn_vshard(uint32_t nk, uint32_t nv, const std::vector<double>& bw,
                     const std::string& mode, std::vector<GdnShard>& out, std::string* error);
}
