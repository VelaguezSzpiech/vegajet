// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/placement.hpp"
#include "core/gpu_select.hpp"
#include "core/tp_json.hpp"
#include "core/tp_runtime_topology.hpp"
#include <cstdlib>
#include <limits>
#include <set>
#include <sstream>

namespace vegajet::placement {
namespace {
using namespace tp::detail;
uint32_t u32(const json::Value& v) {
    const auto n = integer(v);
    require(n <= std::numeric_limits<uint32_t>::max(), "placement integer exceeds uint32");
    return static_cast<uint32_t>(n);
}
void identity_valid(const Identity& id) {
    require(id.unique_id.empty() != id.bdf.empty(), "placement device needs exactly one unique_id or bdf");
    if (!id.bdf.empty()) {
        std::string canonical;
        require(gpusel::parse_bdf(id.bdf, canonical) && canonical == id.bdf, "placement bdf must be full canonical dddd:bb:dd.f");
    }
}
Identity identity(const json::Value& v) {
    keys(v, {"unique_id", "bdf"});
    Identity id;
    if (auto p = v.get("unique_id")) id.unique_id = string(*p);
    if (auto p = v.get("bdf")) id.bdf = string(*p);
    require(v.obj.size() == 1, "placement device needs exactly one unique_id or bdf");
    identity_valid(id);
    return id;
}
std::string quote(const std::string& s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += static_cast<char>(c); }
        else if (c < 32) {
            const char* hex = "0123456789abcdef";
            out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15];
        } else out += static_cast<char>(c);
    }
    return out + "\"";
}
std::string id_json(const Identity& id) {
    identity_valid(id);
    return id.unique_id.empty() ? "{\"bdf\":" + quote(id.bdf) + "}" : "{\"unique_id\":" + quote(id.unique_id) + "}";
}
size_t find_device(const Identity& id, const std::vector<tp::DeviceProfile>& devices) {
    identity_valid(id);
    size_t found = devices.size();
    for (size_t i = 0; i < devices.size(); ++i) {
        if ((!id.unique_id.empty() && id.unique_id == devices[i].unique_id) ||
            (!id.bdf.empty() && id.bdf == devices[i].bdf)) {
            require(found == devices.size(), "placement device identity is ambiguous");
            found = i;
        }
    }
    require(found != devices.size(), "placement device is missing from selected inventory");
    return found;
}
size_t group_index(const Config& c, const std::string& id) {
    for (size_t i = 0; i < c.tp_groups.size(); ++i) if (c.tp_groups[i].id == id) return i;
    throw std::invalid_argument("placement references missing TP group: " + id);
}
void array(const json::Value& v, const char* name) {
    require(v.is_array(), std::string("placement ") + name + " must be an array");
}
} // namespace

Config from_json(const std::string& text) {
    const auto v = parse(text);
    keys(v, {"schema", "tp_groups", "pipeline_stages", "replicas", "drafter"});
    require(integer(field(v, "schema")) == 1, "unsupported placement schema");
    Config c;
    const auto& groups = field(v, "tp_groups"); array(groups, "tp_groups");
    for (const auto& g : groups.arr) {
        keys(g, {"id", "devices"});
        TpGroup group; group.id = string(field(g, "id"));
        const auto& devices = field(g, "devices"); array(devices, "devices");
        for (const auto& d : devices.arr) group.devices.push_back(identity(d));
        c.tp_groups.push_back(std::move(group));
    }
    if (const auto* stages = v.get("pipeline_stages")) {
        array(*stages, "pipeline_stages");
        for (const auto& s : stages->arr) {
            keys(s, {"tp_group", "layer_begin", "layer_end"});
            c.pipeline_stages.push_back({string(field(s, "tp_group")), u32(field(s, "layer_begin")), u32(field(s, "layer_end"))});
        }
    }
    if (const auto* replicas = v.get("replicas")) {
        array(*replicas, "replicas");
        for (const auto& r : replicas->arr) {
            keys(r, {"id", "tp_group"});
            c.replicas.push_back({string(field(r, "id")), string(field(r, "tp_group"))});
        }
    }
    if (const auto* d = v.get("drafter")) {
        keys(*d, {"kind", "device"});
        c.drafter = {string(field(*d, "kind")), identity(field(*d, "device"))};
        require(c.drafter.kind == "DFlash2" || c.drafter.kind == "MTP", "placement drafter kind must be DFlash2 or MTP");
    }
    return c;
}

std::string to_json(const Config& c) {
    std::ostringstream o;
    o << "{\"schema\":1,\"tp_groups\":[";
    for (size_t g = 0; g < c.tp_groups.size(); ++g) {
        if (g) o << ',';
        o << "{\"id\":" << quote(c.tp_groups[g].id) << ",\"devices\":[";
        for (size_t r = 0; r < c.tp_groups[g].devices.size(); ++r) {
            if (r) o << ',';
            o << id_json(c.tp_groups[g].devices[r]);
        }
        o << "]}";
    }
    o << "],\"pipeline_stages\":[";
    for (size_t s = 0; s < c.pipeline_stages.size(); ++s) {
        if (s) o << ',';
        const auto& stage = c.pipeline_stages[s];
        o << "{\"tp_group\":" << quote(stage.tp_group) << ",\"layer_begin\":" << stage.layer_begin << ",\"layer_end\":" << stage.layer_end << '}';
    }
    o << "],\"replicas\":[";
    for (size_t r = 0; r < c.replicas.size(); ++r) {
        if (r) o << ',';
        o << "{\"id\":" << quote(c.replicas[r].id) << ",\"tp_group\":" << quote(c.replicas[r].tp_group) << '}';
    }
    o << ']';
    if (!c.drafter.kind.empty()) o << ",\"drafter\":{\"kind\":" << quote(c.drafter.kind) << ",\"device\":" << id_json(c.drafter.device) << '}';
    return o.str() + "}";
}

std::vector<std::vector<tp::DeviceProfile>> resolve(const Config& c, const std::vector<tp::DeviceProfile>& selected) {
    require(!c.tp_groups.empty(), "placement needs at least one TP group");
    std::set<std::string> names, physical;
    std::set<size_t> used;
    // Reject inventories with aliased physical devices even when selectors differ.
    for (const auto& d : selected) {
        require(!d.unique_id.empty() || !d.bdf.empty(), "selected device lacks persistent identity");
        if (!d.bdf.empty()) require(physical.insert("bdf:" + d.bdf).second, "selected inventory contains duplicate BDF");
        if (!d.unique_id.empty()) require(physical.insert("uid:" + d.unique_id).second, "selected inventory contains duplicate unique_id");
    }
    std::vector<std::vector<tp::DeviceProfile>> out;
    for (const auto& g : c.tp_groups) {
        require(!g.id.empty() && names.insert(g.id).second, "placement TP group id is empty or duplicated");
        require(!g.devices.empty(), "placement TP group has no devices: " + g.id);
        out.emplace_back();
        for (const auto& d : g.devices) {
            const size_t i = find_device(d, selected);
            require(used.insert(i).second, "placement device overlap in TP group: " + g.id);
            out.back().push_back(selected[i]);
        }
    }
    if (!c.drafter.kind.empty()) {
        require(c.drafter.kind == "DFlash2" || c.drafter.kind == "MTP", "placement drafter kind must be DFlash2 or MTP");
        require(used.insert(find_device(c.drafter.device, selected)).second, "placement drafter device overlaps target TP group");
    }
    return out;
}

void validate(const Config& c, const std::vector<tp::DeviceProfile>& selected, uint32_t layers, const CapacityCheck& capacity) {
    const auto groups = resolve(c, selected);
    require(layers > 0, "placement model has no layers");
    require(bool(capacity), "placement requires a model capacity adapter");
    require(c.pipeline_stages.empty() || c.replicas.empty(), "placement pipeline stages and replicas cannot be combined");
    std::set<size_t> referenced;
    if (!c.pipeline_stages.empty()) {
        uint32_t next = 0;
        for (const auto& s : c.pipeline_stages) {
            require(s.layer_begin == next, s.layer_begin < next ? "placement pipeline layer overlap/order" : "placement pipeline layer gap");
            require(s.layer_end > s.layer_begin && s.layer_end <= layers, "placement invalid pipeline layer range");
            const auto g = group_index(c, s.tp_group);
            require(referenced.insert(g).second, "placement TP group reused by pipeline stages");
            capacity(groups[g], s.layer_begin, s.layer_end);
            next = s.layer_end;
        }
        require(next == layers, "placement pipeline layer gap at end");
    } else if (!c.replicas.empty()) {
        std::set<std::string> ids;
        for (const auto& r : c.replicas) {
            require(!r.id.empty() && ids.insert(r.id).second, "placement replica id is empty or duplicated");
            const auto g = group_index(c, r.tp_group);
            require(groups[g].size() == 1, "placement serving replicas require N=1 TP groups");
            require(referenced.insert(g).second, "placement replicas overlap one TP group");
            capacity(groups[g], 0, layers);
        }
    } else {
        require(groups.size() == 1, "placement multiple TP groups need pipeline stages or N=1 replicas");
        capacity(groups[0], 0, layers);
        referenced.insert(0);
    }
    require(referenced.size() == groups.size(), "placement has unreferenced TP groups");
}

Config propose(const std::vector<tp::DeviceProfile>& selected, const tp::PeerMatrix& peers,
               uint32_t layers, const CapacityCheck& capacity, const tp::CollectiveOptions& options) {
    require(!selected.empty() && peers.size() == selected.size(), "placement topology/inventory size mismatch");
    require(layers > 0 && bool(capacity), "placement proposal needs layers and capacity adapter");
    const auto topology = tp::plan_topology(peers, options);
    // Owner topology stages override island defaults, but placement layer policy stays explicit.
    const auto& candidates = topology.stages;
    std::string last_error;
    for (const auto& island : candidates) {
        Config c; TpGroup group; group.id = "target";
        for (size_t i : island) {
            const auto& d = selected.at(i);
            group.devices.push_back(d.unique_id.empty() ? Identity{"", d.bdf} : Identity{d.unique_id, ""});
        }
        c.tp_groups.push_back(std::move(group));
        try { validate(c, selected, layers, capacity); return c; }
        catch (const std::runtime_error& e) { last_error = e.what(); }
    }
    throw std::runtime_error("placement: no P2P island fits the capacity adapter; owner must supply a placement/strategy" +
                             (last_error.empty() ? std::string{} : ": " + last_error));
}

std::string executor_error(const Config& c) {
    if (!c.pipeline_stages.empty()) return "placement: unsupported executor feature pipeline stages";
    if (!c.replicas.empty()) return "placement: unsupported executor feature N=1 serving replicas";
    if (!c.drafter.kind.empty()) return "placement: unsupported executor feature dedicated drafter device (" + c.drafter.kind + ")";
    if (c.tp_groups.size() != 1) return "placement: executor requires exactly one TP group";
    return {};
}
std::vector<tp::DeviceProfile> executor_devices(const Config& c, const std::vector<tp::DeviceProfile>& selected) {
    const auto error = executor_error(c);
    require(error.empty(), error);
    return resolve(c, selected).at(0);
}
std::vector<tp::DeviceProfile> selected_profiles() {
    auto devices = gpusel::scan_topology(gpusel::sysfs_root());
    std::string error;
    if (!gpusel::filter_from_env(devices, &error)) throw std::invalid_argument(error);
    std::vector<tp::DeviceProfile> out;
    for (const auto& d : devices) {
        tp::DeviceProfile p;
        p.bdf = gpusel::bdf(d.pci_domain, d.location_id);
        const auto key = tp::device_key(p.bdf, gpusel::sysfs_root());
        if (key.rfind("uid:", 0) == 0) p.unique_id = key.substr(4);
        p.usable_vram_bytes = d.vram_bytes > (1ULL << 30) ? d.vram_bytes - (1ULL << 30) : 0;
        p.n_cu = d.simd_count / 4; p.bw_hbm_gbs = 1; p.mclk = 1;
        out.push_back(p);
    }
    return out;
}
tp::PeerMatrix selected_peers(const std::vector<tp::DeviceProfile>& selected) {
    const auto devices = gpusel::scan_topology(gpusel::sysfs_root());
    std::vector<tp::RuntimeDevice> runtime;
    for (const auto& p : selected) {
        bool found = false;
        for (const auto& d : devices) if (p.bdf == gpusel::bdf(d.pci_domain, d.location_id)) {
            runtime.push_back({d.node_id, d.drm_render_minor, p.bdf}); found = true; break;
        }
        require(found, "placement topology missing selected device");
    }
    return tp::read_runtime_topology(gpusel::sysfs_root(), runtime).peers;
}
void bind_executor(const Config& config, uint32_t ranks) {
    const auto devices = executor_devices(config, selected_profiles());
    require(devices.size() == ranks, "placement TP-group size conflicts with --tp");
    std::string selection;
    for (const auto& p : devices) { if (!selection.empty()) selection += ','; selection += p.bdf; }
    require(!selection.empty(), "placement has no executable devices");
    if (setenv(gpusel::kEnvVar, selection.c_str(), 1) != 0) throw std::runtime_error("cannot bind placement device order");
}
} // namespace vegajet::placement
