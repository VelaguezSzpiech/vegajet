// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/tp_runtime_topology.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>

namespace vegajet::tp {
namespace {
std::map<std::string, uint64_t> properties(const std::filesystem::path& path) {
    std::ifstream in(path);
    std::map<std::string, uint64_t> result;
    std::string key; uint64_t value;
    while (in >> key >> value) result[key] = value;
    return result;
}
std::string unique_id(const std::filesystem::path& path) {
    std::ifstream in(path); std::string text;
    if (!(in >> text)) return {};
    if (text.compare(0, 2, "0x") == 0) text.erase(0, 2);
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return c >= 'A' && c <= 'F' ? static_cast<char>(c - 'A' + 'a') : static_cast<char>(c);
    });
    if (text.empty() || text.find_first_not_of("0123456789abcdef") != std::string::npos) return {};
    const auto first = text.find_first_not_of('0');
    return first == std::string::npos ? std::string{} : text.substr(first);
}
}
RuntimeTopology read_runtime_topology(const std::string& root, const std::vector<RuntimeDevice>& selected) {
    if (selected.empty()) throw std::invalid_argument("runtime topology needs selected devices");
    namespace fs = std::filesystem;
    RuntimeTopology result;
    result.peers.assign(selected.size(), std::vector<PeerLink>(selected.size()));
    std::set<uint32_t> nodes;
    std::set<std::string> ids;
    for (const auto& d : selected) {
        if (!nodes.insert(d.node_id).second || d.bdf.empty())
            throw std::invalid_argument("duplicate topology node or missing BDF");
        auto uid = unique_id(fs::path(root) / "sys/class/drm" / ("renderD" + std::to_string(d.render_minor)) / "device/unique_id");
        if (uid.empty()) uid = unique_id(fs::path(root) / "sys/bus/pci/devices" / d.bdf / "unique_id");
        std::string id = uid.empty() ? "bdf:" + d.bdf : "unique_id:" + uid;
        if (!ids.insert(id).second) throw std::invalid_argument("selected devices have duplicate stable identity");
        result.identities.push_back(std::move(id));
    }
    const fs::path base = fs::path(root) / "sys/class/kfd/kfd/topology/nodes";
    for (size_t s = 0; s < selected.size(); ++s) {
        // io_links may describe direct or KFD-composed links; p2p_links supplies
        // explicit indirect GPU reachability on newer drivers. Never infer it
        // from a path through the CPU node or p2p_links_count alone.
        for (const char* directory : {"io_links", "p2p_links"}) {
            std::error_code ec;
            fs::directory_iterator it(base / std::to_string(selected[s].node_id) / directory, ec);
            if (ec) continue;
            for (const auto& entry : it) {
                const auto p = properties(entry.path() / "properties");
                auto get = [&](const char* key) { auto i = p.find(key); return i == p.end() ? uint64_t{0} : i->second; };
                if (!p.count("node_from") || !p.count("node_to") || get("node_from") != selected[s].node_id) continue;
                // KFD: bit 4 = NO_PEER_TO_PEER_DMA; non-coherent links are
                // supported by explicit glc/slc push and per-writer drains.
                // A positive PCIe/XGMI edge must also have a reported bandwidth.
                if ((get("flags") & 16u) || (get("type") != 2 && get("type") != 11)) continue;
                const uint64_t bandwidth = get("min_bandwidth") ? get("min_bandwidth") : get("max_bandwidth");
                if (!bandwidth) continue;
                for (size_t d = 0; d < selected.size(); ++d) if (s != d && get("node_to") == selected[d].node_id) {
                    auto& edge = result.peers[s][d];
                    const double gbs = static_cast<double>(bandwidth) / 1000.0; // KFD reports MB/s
                    edge = {true, edge.p2p ? std::min(edge.gbs, gbs) : gbs};
                }
            }
        }
    }
    return result;
}
} // namespace vegajet::tp
