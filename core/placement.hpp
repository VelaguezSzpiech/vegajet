// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include "core/tp_collective.hpp"
#include "core/tp_sharder.hpp"
#include <functional>
#include <string>
#include <vector>

namespace vegajet::placement {
struct Identity {
    std::string unique_id, bdf; // exactly one selector, never an enumeration ordinal
};
struct TpGroup { std::string id; std::vector<Identity> devices; };
struct Stage { std::string tp_group; uint32_t layer_begin = 0, layer_end = 0; }; // half-open
struct Replica { std::string id, tp_group; }; // independent serving worker, TP=1 only
struct Drafter { std::string kind; Identity device; }; // DFlash2 or MTP
struct Config {
    std::vector<TpGroup> tp_groups;
    std::vector<Stage> pipeline_stages;
    std::vector<Replica> replicas;
    Drafter drafter;
};
Config from_json(const std::string& text); // strict, including duplicate/unknown keys
std::string to_json(const Config& config);
// Resolves both aliases against the selected inventory, preserving owner rank order.
// Missing, ambiguous and overlapping physical devices are errors, including alias overlap.
std::vector<std::vector<tp::DeviceProfile>> resolve(const Config& config,
                                                  const std::vector<tp::DeviceProfile>& selected);
// Caller owns model semantics (packed bytes, state, KV and scratch at this context).
// A stage adapter must charge only its layer range plus its required replicated state.
using CapacityCheck = std::function<void(const std::vector<tp::DeviceProfile>&, uint32_t, uint32_t)>;
void validate(const Config& config, const std::vector<tp::DeviceProfile>& selected,
              uint32_t layers, const CapacityCheck& capacity);
// Proposes the first full-model-fitting mutual-P2P island in selected order, using
// M2/M3 topology rules; never invents a peer edge or silently changes an owner config.
// If no island fits, requests an explicit owner strategy rather than guessing stages.
Config propose(const std::vector<tp::DeviceProfile>& selected, const tp::PeerMatrix& peers,
               uint32_t layers, const CapacityCheck& capacity,
               const tp::CollectiveOptions& options = {});
// Empty means the current executor can consume this config; unsupported pieces named.
std::string executor_error(const Config& config);
std::vector<tp::DeviceProfile> executor_devices(const Config& config,
                                               const std::vector<tp::DeviceProfile>& selected);
// Read-only selected sysfs inventory and directed topology, never device-node I/O.
std::vector<tp::DeviceProfile> selected_profiles();
tp::PeerMatrix selected_peers(const std::vector<tp::DeviceProfile>& selected);
// Resolve placement and export canonical BDF rank order for Runtime::init. This may
// reorder/subselect the existing selection, never add an unselected device.
void bind_executor(const Config& config, uint32_t ranks);
} // namespace vegajet::placement
