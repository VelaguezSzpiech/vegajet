// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include "core/tp_collective.hpp"
#include <string>
#include <vector>

namespace vegajet::tp {
struct RuntimeDevice {
    uint32_t node_id = 0, render_minor = 0;
    std::string bdf;
};
struct RuntimeTopology {
    std::vector<std::string> identities; // unique_id:<hex>, otherwise bdf:<canonical>
    PeerMatrix peers; // directed KFD capabilities; bandwidth is reported, never estimated
};
// Read-only; selected order is rank order. Unknown links remain non-P2P.
RuntimeTopology read_runtime_topology(const std::string& root, const std::vector<RuntimeDevice>& selected);
} // namespace vegajet::tp
