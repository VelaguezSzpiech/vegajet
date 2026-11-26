// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/tp_sharder.hpp"
#include "core/tp_json.hpp"
#include "core/gpu_select.hpp"
#include <fstream>
#include <numeric>
#include <set>

namespace vegajet::tp {
using namespace detail;
namespace {
bool good_id(const std::string& id) {
    return !id.empty() && id != "0" && id != "0x0" &&
           id.find_first_not_of("0xX") != std::string::npos;
}
void validate_devices(const std::vector<DeviceProfile>& devs) {
    require(!devs.empty(), "TP needs at least one rank");
    std::set<std::string> ids;
    for (const auto& d : devs) {
        require(ids.insert(d.key()).second, "duplicate device identity: " + d.key());
        require(std::isfinite(d.bw_hbm_gbs) && d.bw_hbm_gbs > 0 &&
                std::isfinite(d.pcie_link_gbs) && d.pcie_link_gbs >= 0 &&
                std::isfinite(d.mclk) && d.mclk > 0 && d.n_cu && d.usable_vram_bytes && d.numa_node >= -1,
                "invalid device profile: " + d.key());
    }
}
std::vector<uint64_t> allocate(uint64_t units, const std::vector<double>& weights,
                               const std::vector<uint64_t>& caps) {
    // Reuse Hamilton apportionment; freeze saturated ranks and re-apportion the remainder.
    require(units <= 9007199254740991ULL, "row/head unit count exceeds exact apportionment range");
    std::vector<uint64_t> out(weights.size(), 0);
    uint64_t remaining = units;
    while (remaining) {
        std::vector<double> active(weights.size());
        bool available = false;
        for (size_t i = 0; i < weights.size(); ++i) {
            if (out[i] < caps[i] && weights[i] > 0) { active[i] = weights[i]; available = true; }
        }
        if (!available) throw std::runtime_error("model does not fit usable VRAM (including row-tile padding)");
        const auto shares = shard::apportion(remaining, active);
        uint64_t placed = 0;
        for (size_t i = 0; i < out.size(); ++i) {
            const uint64_t take = std::min(shares[i], caps[i] - out[i]);
            out[i] += take;
            placed = add(placed, take);
        }
        require(placed && placed <= remaining, "apportionment lost integer precision");
        remaining -= placed;
    }
    return out;
}
std::vector<double> weights_for(const AllocationOverride* o, const std::vector<double>& bw) {
    if (!o || o->fractions.empty()) return bw;
    require(o->rows.empty() && o->fractions.size() == bw.size(), "override needs one fraction per rank, not row counts too");
    long double total = 0;
    for (double f : o->fractions) { require(std::isfinite(f) && f >= 0, "invalid fraction"); total += f; }
    require(std::abs(total - 1) <= 1e-9L, "override fractions must sum to one");
    return o->fractions;
}
std::vector<uint64_t> counts_for(uint64_t rows, uint64_t tile, const AllocationOverride* o,
                                  const std::vector<double>& bw, const std::vector<uint64_t>& caps) {
    const uint64_t units = rows / tile + (rows % tile != 0);
    const auto weights = weights_for(o, bw);
    if (!o || o->rows.empty()) {
        if (o) {
            // Explicit fractions are binding after Hamilton rounding, never silently clamped.
            const auto counts = shard::apportion(units, weights);
            for (size_t i = 0; i < caps.size(); ++i)
                if (counts[i] > caps[i]) throw std::runtime_error("owner fraction override exceeds usable VRAM");
            return counts;
        }
        return allocate(units, weights, caps);
    }
    require(o->fractions.empty() && o->rows.size() == bw.size(), "override needs one row count per rank");
    uint64_t sum = 0;
    std::vector<uint64_t> result(bw.size());
    for (size_t i = 0; i < bw.size(); ++i) {
        const uint64_t count = o->rows[i];
        sum = add(sum, count);
        require(sum <= rows && (count % tile == 0 || sum == rows), "row override violates tile/head boundary");
        result[i] = count / tile + (count % tile != 0);
        if (result[i] > caps[i]) throw std::runtime_error("owner row override exceeds usable VRAM");
    }
    require(sum == rows, "owner row counts do not cover tensor");
    return result;
}
} // namespace

std::string DeviceProfile::key() const {
    if (good_id(unique_id)) return "uid:" + unique_id;
    std::string canonical;
    require(gpusel::parse_bdf(bdf, canonical), "device without unique_id requires full PCI BDF");
    return "bdf:" + canonical;
}
std::string device_key(const std::string& bdf, const std::string& root) {
    std::string canonical;
    require(gpusel::parse_bdf(bdf, canonical), "invalid full PCI BDF");
    std::ifstream f(root + "/sys/bus/pci/devices/" + canonical + "/unique_id");
    DeviceProfile d;
    d.bdf = canonical;
    f >> d.unique_id;
    return d.key();
}
std::vector<DeviceProfile> profiles_from_json(const std::string& text, uint64_t reserve) {
    const auto root = detail::parse(text);
    keys(root, {}, false);
    require(integer(field(root, "schema")) == 1, "unsupported hardware profile schema");
    const auto& gpus = field(root, "gpus");
    require(gpus.is_array(), "gpus must be an array");
    std::vector<DeviceProfile> result;
    for (const auto& v : gpus.arr) {
        keys(v, {}, false); // Preserve compatibility with hw_profile's other calibrated fields.
        DeviceProfile d;
        if (auto p = v.get("unique_id")) d.unique_id = string(*p);
        if (auto p = v.get("bdf")) d.bdf = string(*p);
        auto alias = [&](const std::string& a, const std::string& b) -> const json::Value& {
            auto p = v.get(a); return p ? *p : field(v, b);
        };
        d.bw_hbm_gbs = number(alias("bw_hbm_gbs", "bw_gbs"));
        const uint64_t cu = integer(alias("n_cu", "cu"));
        require(cu <= UINT32_MAX, "CU count out of range");
        d.n_cu = static_cast<uint32_t>(cu);
        if (auto p = v.get("mclk")) d.mclk = number(*p);
        else d.mclk = number(alias("mem_clock_mhz", "mclk_mhz"));
        if (auto p = v.get("usable_vram_bytes")) {
            d.usable_vram_bytes = integer(*p);
            if (auto total = v.get("vram_bytes"))
                require(d.usable_vram_bytes <= integer(*total), "usable VRAM exceeds physical VRAM");
        } else {
            const uint64_t total = integer(field(v, "vram_bytes"));
            require(reserve < total, "reserve exhausts device VRAM");
            d.usable_vram_bytes = total - reserve;
        }
        if (auto p = v.get("pcie_link_gbs")) d.pcie_link_gbs = number(*p);
        if (auto p = v.get("numa_node")) {
            const double numa = number(*p);
            require(numa >= -1 && numa <= INT32_MAX && std::floor(numa) == numa, "invalid NUMA node");
            d.numa_node = static_cast<int>(numa);
        }
        result.push_back(d);
    }
    validate_devices(result);
    return result;
}
ShardOptions shard_options_from_json(const std::string& text) {
    const auto root = detail::parse(text);
    keys(root, {"schema", "row_tile", "split_k_threshold", "replicate_kv", "tensors"});
    require(integer(field(root, "schema")) == 1, "unsupported shard options schema");
    ShardOptions o;
    if (auto p = root.get("row_tile")) {
        const auto tile = integer(*p);
        require(tile == 16 || tile == 32, "row_tile must be 16 or 32");
        o.row_tile = static_cast<uint32_t>(tile);
    }
    if (auto p = root.get("split_k_threshold")) o.split_k_threshold = integer(*p);
    if (auto p = root.get("replicate_kv")) o.replicate_kv = boolean(*p);
    if (auto p = root.get("tensors")) {
        keys(*p, {}, false);
        for (const auto& entry : p->obj) {
            keys(entry.second, {"fractions", "rows"});
            AllocationOverride a;
            const auto f = entry.second.get("fractions"), r = entry.second.get("rows");
            require((f != nullptr) != (r != nullptr), "choose exactly one of fractions or rows");
            const auto& values = f ? *f : *r;
            require(values.is_array() && !values.arr.empty(), "override array must be nonempty");
            for (const auto& v : values.arr) {
                if (f) a.fractions.push_back(number(v));
                else a.rows.push_back(integer(v));
            }
            o.tensors.emplace(entry.first, std::move(a));
        }
    }
    return o;
}
ShardPlan plan_shards(const ModelSpec& model, const std::vector<DeviceProfile>& devices, const ShardOptions& options) {
    validate_devices(devices);
    require(options.row_tile == 16 || options.row_tile == 32, "row_tile must be 16 or 32");
    ShardPlan plan;
    std::vector<double> bw;
    double max_bw = 0;
    for (const auto& d : devices) max_bw = std::max(max_bw, d.bw_hbm_gbs);
    for (const auto& d : devices) {
        if (model.replicated_bytes > d.usable_vram_bytes) throw std::runtime_error("model replicated bytes do not fit usable VRAM: " + d.key());
        RankShard r;
        r.device_key = d.key(); r.used_bytes = model.replicated_bytes;
        plan.ranks.push_back(r);
        bw.push_back(d.bw_hbm_gbs / max_bw);
    }
    std::set<std::string> names;
    auto override_for = [&](const std::string& name) -> const AllocationOverride* {
        require(!name.empty() && names.insert(name).second, "empty or duplicate tensor name");
        auto it = options.tensors.find(name);
        return it == options.tensors.end() ? nullptr : &it->second;
    };
    // Allocate indivisible GQA groups first, then tile-padded rows in caller order.
    for (const auto& a : model.attention) {
        const auto o = override_for(a.name);
        require(a.q_heads && a.kv_heads && a.q_heads % a.kv_heads == 0, "attention requires integral Q/KV groups");
        const uint64_t group = a.q_heads / a.kv_heads;
        const uint64_t unit = options.replicate_kv ? 1 : group;
        const uint64_t cost = add(mul(unit, a.q_bytes_per_head), a.kv_bytes_per_head);
        require(cost > 0, "attention bytes must be specified");
        // Start from desired bandwidth shares. Assign Q/group units in global head order;
        // replicated KV is charged once per rank, exactly when its first Q head is placed.
        const auto weights = weights_for(o, bw);
        std::vector<uint64_t> unlimited(devices.size(), a.q_heads);
        const auto desired = counts_for(a.q_heads, unit, o, weights, unlimited);
        std::vector<uint64_t> assigned(devices.size(), 0);
        std::vector<HeadMap> maps(devices.size());
        for (uint64_t first = 0; first < a.q_heads; first += unit) {
            size_t selected = devices.size();
            long double best = -std::numeric_limits<long double>::infinity();
            uint64_t chosen_cost = 0;
            for (size_t r = 0; r < devices.size(); ++r) {
                if (o && assigned[r] >= desired[r]) continue;
                const auto& kv = maps[r].kv_heads;
                const uint32_t head = first / static_cast<uint32_t>(group);
                const bool present = std::find(kv.begin(), kv.end(), head) != kv.end();
                const uint64_t need = add(mul(unit, a.q_bytes_per_head), present ? 0 : a.kv_bytes_per_head);
                if (need > devices[r].usable_vram_bytes - plan.ranks[r].used_bytes) continue;
                const long double deficit = static_cast<long double>(desired[r]) - assigned[r];
                if (deficit > best) { best = deficit; selected = r; chosen_cost = need; }
            }
            if (selected == devices.size()) throw std::runtime_error("attention/owner override does not fit usable VRAM: " + a.name);
            auto& m = maps[selected];
            const uint32_t kv = first / static_cast<uint32_t>(group);
            auto it = std::find(m.kv_heads.begin(), m.kv_heads.end(), kv);
            if (it == m.kv_heads.end()) { m.kv_heads.push_back(kv); it = m.kv_heads.end() - 1; }
            for (uint32_t q = first; q < first + unit; ++q) {
                m.q_heads.push_back(q);
                m.q_to_local_kv.push_back(static_cast<uint32_t>(it - m.kv_heads.begin()));
            }
            assigned[selected]++;
            m.bytes = add(m.bytes, chosen_cost);
            plan.ranks[selected].used_bytes = add(plan.ranks[selected].used_bytes, chosen_cost);
        }
        for (size_t r = 0; r < devices.size(); ++r) plan.ranks[r].attention.push_back(std::move(maps[r]));
    }
    for (const auto& s : model.rows) {
        const auto o = override_for(s.name);
        require(s.bytes_per_row > 0, "row bytes must be specified");
        const uint64_t tile = options.row_tile, cost = mul(tile, s.bytes_per_row);
        std::vector<uint64_t> caps;
        for (size_t r = 0; r < devices.size(); ++r)
            caps.push_back((devices[r].usable_vram_bytes - plan.ranks[r].used_bytes) / cost);
        const auto counts = counts_for(s.rows, tile, o, bw, caps);
        uint64_t pos = 0;
        for (size_t r = 0; r < devices.size(); ++r) {
            const auto padded = mul(counts[r], tile);
            const auto count = std::min(padded, s.rows - pos);
            RowShard rs;
            rs.rows = {pos, pos + count}; rs.padded_rows = padded; rs.bytes = mul(padded, s.bytes_per_row);
            rs.recommend_split_k = count > 0 && count < options.split_k_threshold;
            plan.ranks[r].rows.push_back(rs);
            plan.ranks[r].used_bytes = add(plan.ranks[r].used_bytes, rs.bytes);
            pos += count;
        }
        require(pos == s.rows, "row partition does not cover tensor");
    }
    for (const auto& entry : options.tensors) require(names.count(entry.first), "unknown tensor override: " + entry.first);
    return plan;
}
} // namespace vegajet::tp
