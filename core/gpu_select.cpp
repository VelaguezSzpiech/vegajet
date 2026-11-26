// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/gpu_select.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>

namespace vegajet::gpusel {

namespace {

std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t')) ++b;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t')) --e;
    return s.substr(b, e - b);
}

bool parse_index(const std::string& t, uint32_t& v) {
    if (t.empty() || t.size() > 3 || t.find_first_not_of("0123456789") != std::string::npos) return false;
    const unsigned long x = std::strtoul(t.c_str(), nullptr, 10);
    if (x > kMaxIndex) return false;
    v = static_cast<uint32_t>(x);
    return true;
}

}  // namespace

std::string bdf(uint32_t domain, uint32_t location) {
    char text[32];
    std::snprintf(text, sizeof(text), "%04x:%02x:%02x.%u", domain,
                  (location >> 8) & 255u, (location >> 3) & 31u, location & 7u);
    return text;
}

bool parse_bdf(const std::string& text, std::string& out) {
    if (text.size() != 12 || text[4] != ':' || text[7] != ':' || text[10] != '.') return false;
    for (size_t i : {0u,1u,2u,3u,5u,6u,8u,9u,11u})
        if (std::string("0123456789abcdefABCDEF").find(text[i]) == std::string::npos) return false;
    const auto domain = std::stoul(text.substr(0,4), nullptr, 16);
    const auto bus = std::stoul(text.substr(5,2), nullptr, 16);
    const auto dev = std::stoul(text.substr(8,2), nullptr, 16);
    const auto fn = std::stoul(text.substr(11,1), nullptr, 16);
    if (dev > 31 || fn > 7) return false;
    out = bdf(static_cast<uint32_t>(domain), static_cast<uint32_t>((bus << 8) | (dev << 3) | fn));
    return true;
}

bool parse(const std::string& spec_in, Selection& out, std::string* err) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    Selection s;
    const std::string spec = trim(spec_in);
    if (spec.empty() || spec == "all") { out = s; return true; }
    std::set<uint32_t> seen;
    size_t pos = 0;
    while (pos <= spec.size()) {
        const size_t comma = spec.find(',', pos);
        const std::string tok = trim(spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos));
        if (tok.empty()) return fail("empty entry in GPU list '" + spec + "'");
        if (tok.find(':') != std::string::npos || tok.find('.') != std::string::npos) {
            std::string identity;
            if (!parse_bdf(tok, identity)) return fail("invalid full PCI BDF '" + tok + "' (expected dddd:bb:dd.f)");
            if (!seen.empty()) return fail("cannot mix GPU ordinals and PCI BDFs");
            if (std::find(s.bdfs.begin(), s.bdfs.end(), identity) != s.bdfs.end()) return fail("duplicate PCI BDF '" + identity + "'");
            s.bdfs.push_back(identity);
            if (comma == std::string::npos) break;
            pos = comma + 1;
            continue;
        }
        if (!s.bdfs.empty()) return fail("cannot mix GPU ordinals and PCI BDFs");
        const size_t dash = tok.find('-');
        uint32_t a = 0, b = 0;
        if (dash == std::string::npos) {
            if (!parse_index(tok, a)) return fail("'" + tok + "' is not a GPU index 0.." + std::to_string(kMaxIndex));
            b = a;
        } else {
            if (!parse_index(trim(tok.substr(0, dash)), a) || !parse_index(trim(tok.substr(dash + 1)), b))
                return fail("'" + tok + "' is not a GPU index range A-B");
            if (b < a) return fail("range '" + tok + "' is reversed");
        }
        for (uint32_t i = a; i <= b; ++i)
            if (!seen.insert(i).second) return fail("GPU " + std::to_string(i) + " is listed twice in '" + spec + "'");
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    s.all = false;
    s.indices.assign(seen.begin(), seen.end());
    out = s;
    return true;
}

bool resolve(const std::string* flag, const char* env_value, Selection& out, std::string* err) {
    std::string why;
    if (flag) {
        if (!parse(*flag, out, &why)) { if (err) *err = "--gpu " + *flag + ": " + why; return false; }
        out.source = "--gpu";
        return true;
    }
    if (env_value) {
        if (!parse(env_value, out, &why)) { if (err) *err = std::string(kEnvVar) + "=" + env_value + ": " + why; return false; }
        out.source = kEnvVar;
        return true;
    }
    out = Selection{};
    return true;
}

std::string canonical(const Selection& s) {
    if (s.all) return "all";
    std::string o;
    if (!s.bdfs.empty()) {
        for (const auto& id : s.bdfs) { if (!o.empty()) o += ","; o += id; }
        return o;
    }
    for (size_t i = 0; i < s.indices.size(); ++i) o += (i ? "," : "") + std::to_string(s.indices[i]);
    return o;
}

bool pick(const Selection& s, size_t n, std::vector<size_t>& out, std::string* err) {
    out.clear();
    if (!s.bdfs.empty()) { if (err) *err = "PCI BDF selection requires device identities, not a count"; return false; }
    if (s.all) {
        for (size_t i = 0; i < n; ++i) out.push_back(i);
        return true;
    }
    for (uint32_t i : s.indices) {
        if (i >= n) {
            if (err)
                *err = "GPU " + std::to_string(i) + " selected (" + s.source + ") but only " + std::to_string(n) +
                       " gfx906 GPU(s) found" + (n ? " (valid: 0.." + std::to_string(n - 1) + ")" : "");
            return false;
        }
        out.push_back(i);
    }
    return true;
}

bool pick_bdfs(const Selection& s, const std::vector<std::string>& ids, std::vector<size_t>& out, std::string* err) {
    out.clear();
    std::vector<size_t> picked;
    for (const auto& id : s.bdfs) {
        size_t match = ids.size(), count = 0;
        for (size_t i = 0; i < ids.size(); ++i) if (ids[i] == id) { match = i; ++count; }
        if (count != 1) {
            if (err) *err = "PCI BDF " + id + " selected (" + s.source + ") " +
                            (count ? "is ambiguous in KFD topology" : "was not found in gfx906 topology");
            return false;
        }
        picked.push_back(match);
    }
    out = std::move(picked);
    return true;
}

std::vector<Device> scan_topology(const std::string& root) {
    namespace fs = std::filesystem;
    std::vector<Device> out;
    std::error_code ec;
    const fs::path base = fs::path(root) / "sys/class/kfd/kfd/topology/nodes";
    if (!fs::is_directory(base, ec)) return out;
    std::vector<uint32_t> ids;
    for (const auto& ent : fs::directory_iterator(base, ec)) {
        const std::string nm = ent.path().filename().string();
        if (nm.empty() || nm.find_first_not_of("0123456789") != std::string::npos || nm.size() > 6) continue;
        ids.push_back(static_cast<uint32_t>(std::stoul(nm)));
    }
    std::sort(ids.begin(), ids.end());
    for (uint32_t nid : ids) {
        const fs::path nd = base / std::to_string(nid);
        std::ifstream pf(nd / "properties");
        if (!pf) continue;
        Device d;
        d.node_id = nid;
        uint32_t gfx = 0;
        std::string key;
        uint64_t val = 0;
        while (pf >> key >> val) {
            if (key == "gfx_target_version") gfx = static_cast<uint32_t>(val);
            else if (key == "simd_count") d.simd_count = static_cast<uint32_t>(val);
            else if (key == "drm_render_minor") d.drm_render_minor = static_cast<uint32_t>(val);
            else if (key == "location_id") d.location_id = static_cast<uint32_t>(val);
            else if (key == "domain") d.pci_domain = static_cast<uint32_t>(val);
        }
        if (gfx != 90006u) continue;
        std::ifstream gf(nd / "gpu_id");
        uint64_t gid = 0;
        if (gf >> gid) d.gpu_id = static_cast<uint32_t>(gid);
        std::ifstream mf(nd / "mem_banks/0/properties");
        while (mf >> key >> val)
            if (key == "size_in_bytes") { d.vram_bytes = val; break; }
        out.push_back(d);
    }
    std::stable_sort(out.begin(), out.end(), [](const Device& a, const Device& b) { return a.pci_domain != b.pci_domain ? a.pci_domain < b.pci_domain : a.location_id < b.location_id; });
    for (size_t i = 0; i < out.size(); ++i) out[i].index = static_cast<uint32_t>(i);
    return out;
}

std::string sysfs_root() {
    const char* r = std::getenv("VJ_SYSFS_ROOT");
    return r && *r ? std::string(r) : std::string("/");
}

std::string format_list(const std::vector<Device>& devs, const Selection& s) {
    std::string o;
    char line[192];
    if (devs.empty()) return "no gfx906 GPUs found in the KFD topology\n";
    std::snprintf(line, sizeof(line), "  %-3s %-5s %-6s %-8s %-7s %-4s %-9s\n", "idx", "sel", "node", "gpu_id", "render", "CUs", "VRAM GiB");
    o += line;
    std::vector<size_t> picked;
    std::string why;
    if (!pick_devices(s, devs, picked, &why)) return why + "\n";
    for (const auto& d : devs) {
        const bool sel = std::find(picked.begin(), picked.end(), d.index) != picked.end();
        std::snprintf(line, sizeof(line), "  %-3u %-5s %-6u %-8u %-7u %-4u %-9.1f\n", d.index, sel ? "*" : "", d.node_id, d.gpu_id,
                      d.drm_render_minor, d.simd_count / 4u, d.vram_bytes / double(1ull << 30));
        o += line;
        o += "    PCI " + bdf(d.pci_domain, d.location_id) + "\n";
    }
    o += "selection: " + canonical(s) + " (" + s.source + ")\n";
    return o;
}

}  // namespace vegajet::gpusel
