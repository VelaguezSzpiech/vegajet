// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Zero-root, read-only host fabric preflight (VEGAJET_DECODE_FABRIC=1 / `vegajet doctor`).
// Never writes sysfs or PCI config. PCIe config space beyond the first 64 bytes is root-only on Linux, so the
// Relaxed Ordering / MaxPayload / ACS checks report "unknown" for unprivileged runs instead of guessing.
#include "core/hw_preflight.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <set>
#include <sstream>
#include <sched.h>

namespace fs = std::filesystem;
namespace vegajet::hw_preflight {
namespace {
std::string read_text(const fs::path& p) { std::ifstream f(p); std::ostringstream s; s << f.rdbuf(); return f ? s.str() : ""; }
std::vector<uint8_t> read_bytes(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return f ? std::vector<uint8_t>(std::istreambuf_iterator<char>(f), {}) : std::vector<uint8_t>{};
}
double number(const std::string& s) { try { return std::stod(s); } catch (...) { return 0.0; } }
uint16_t rd16(const std::vector<uint8_t>& c, size_t o) { return static_cast<uint16_t>(c[o] | (c[o + 1] << 8)); }
uint32_t rd32(const std::vector<uint8_t>& c, size_t o) { return rd16(c, o) | (static_cast<uint32_t>(rd16(c, o + 2)) << 16); }

// Offset of the PCI Express capability (ID 0x10) or 0.
size_t find_pcie_cap(const std::vector<uint8_t>& c) {
    if (c.size() < 0x40 || !(rd16(c, 0x06) & 0x10)) return 0;
    size_t p = c[0x34] & 0xFC;
    for (int guard = 0; p >= 0x40 && p + 2 <= c.size() && guard < 48; ++guard) {
        if (c[p] == 0x10) return p;
        p = c[p + 1] & 0xFC;
    }
    return 0;
}
// Offset of the extended ACS capability (ID 0x000D) or 0.
size_t find_acs_cap(const std::vector<uint8_t>& c) {
    size_t p = 0x100;
    for (int guard = 0; p >= 0x100 && p + 8 <= c.size() && guard < 64; ++guard) {
        const uint32_t h = rd32(c, p);
        if (h == 0 || h == 0xFFFFFFFFu) return 0;
        if ((h & 0xFFFFu) == 0x000Du) return p;
        p = (h >> 20) & 0xFFCu;
    }
    return 0;
}
} // namespace

std::vector<Warning> run(const std::vector<std::string>& bdfs, const std::string& root) {
    std::vector<Warning> out;
    auto warn = [&](const std::string& b, const std::string& c, const std::string& d) { out.push_back({b, c, d}); };
    std::set<std::string> seen_links;
    for (const auto& b : bdfs) {
        const fs::path dev = fs::path(root) / "bus/pci/devices" / b;
        std::error_code ec;
        fs::path real = fs::canonical(dev, ec);
        if (ec) real = dev;
        // 1. PCIe link speed/width on the device and every upstream port.
        for (fs::path p = real; !p.empty() && p != p.root_path(); p = p.parent_path()) {
            const auto speed = read_text(p / "current_link_speed"), max = read_text(p / "max_link_speed");
            const auto width = read_text(p / "current_link_width"), mw = read_text(p / "max_link_width");
            if (speed.empty() || max.empty() || !seen_links.insert(p.string()).second) continue;
            const bool slow = number(speed) < number(max) || number(speed) <= 2.5;
            const bool narrow = !width.empty() && !mw.empty() && number(width) < number(mw);
            if (slow || narrow)
                warn(b, "pcie_link", p.filename().string() + " at " + speed.substr(0, speed.find_first_of(" \n")) +
                                     " GT/s (max " + max.substr(0, max.find_first_of(" \n")) + "), width x" +
                                     std::to_string(static_cast<int>(number(width))) + "/x" + std::to_string(static_cast<int>(number(mw))));
        }
        // 2. Relaxed Ordering + MaxPayload from the device's PCIe capability (DevCtl).
        const auto cfg = read_bytes(real / "config");
        const size_t pc = find_pcie_cap(cfg);
        if (pc != 0 && pc + 10 <= cfg.size()) {
            const uint16_t devctl = rd16(cfg, pc + 8);
            const uint32_t mps = 128u << ((devctl >> 5) & 7u);
            if (!(devctl & 0x10)) warn(b, "relaxed_ordering", "RlxdOrd- (DevCtl bit 4 clear)");
            if (mps < 256u) warn(b, "max_payload", "MaxPayload " + std::to_string(mps) + " B < 256 B");
        } else {
            warn(b, "relaxed_ordering", "unknown (PCIe capability not readable: config space beyond 64 B needs root)");
            warn(b, "max_payload", "unknown (PCIe capability not readable: config space beyond 64 B needs root)");
        }
        // 3. ACS Request Redirect on upstream bridges (forces P2P through the root complex).
        bool acs_known = false;
        for (fs::path p = real.parent_path(); !p.empty() && p != p.root_path(); p = p.parent_path()) {
            if (!fs::exists(p / "config")) continue;
            const auto bc = read_bytes(p / "config");
            if (bc.size() < 0x104) continue;
            acs_known = true;
            const size_t ac = find_acs_cap(bc);
            if (ac != 0 && ac + 8 <= bc.size() && (rd16(bc, ac + 6) & 0x4))
                warn(b, "acs", p.filename().string() + " ACSCtl ReqRedir+ (peer-to-peer traffic redirected upstream)");
        }
        if (!acs_known) warn(b, "acs", "unknown (bridge extended config space needs root)");
        // 4. DPM clock latched low while 'high' is forced.
        const std::string force = read_text(real / "power_dpm_force_performance_level");
        const std::string dpm = read_text(real / "pp_dpm_sclk") + read_text(real / "pp_dpm_mclk");
        if (force.find("high") != std::string::npos && !dpm.empty()) {
            const std::regex active(R"(([0-9]+(?:\.[0-9]+)?)\s*MHz[^\n]*\*)");
            for (std::sregex_iterator i(dpm.begin(), dpm.end(), active), e; i != e; ++i)
                if (number((*i)[1].str()) <= 859.0) {
                    warn(b, "dpm_clock", "performance level 'high' but an active DPM clock is " + (*i)[1].str() + " MHz");
                    break;
                }
        }
    }
    // 5. Single NUMA node on a multi-die CPU.
    const std::string poss = read_text(fs::path(root) / "devices/system/node/possible");
    const bool one_node = !poss.empty() && poss.find_first_of(",-") == std::string::npos;
    if (one_node) {
        std::set<std::string> dies;
        std::error_code cec;
        for (const auto& e : fs::directory_iterator(fs::path(root) / "devices/system/cpu", cec)) {
            const auto nm = e.path().filename().string();
            if (nm.rfind("cpu", 0) != 0 || nm.size() < 4 || !std::isdigit(static_cast<unsigned char>(nm[3]))) continue;
            auto die = read_text(e.path() / "topology/die_id");
            auto l3  = read_text(e.path() / "cache/index3/shared_cpu_list");
            dies.insert(die.empty() ? l3 : die + "|" + l3);
        }
        if (dies.size() > 1)
            warn("host", "numa", "single NUMA node exposed but " + std::to_string(dies.size()) + " CPU dies / L3 domains (UMA interleave)");
    }
    return out;
}

std::vector<int> parse_cpulist(const std::string& in) {
    std::vector<int> out;
    std::string s;
    for (char c : in) if (!std::isspace(static_cast<unsigned char>(c))) s += c;
    if (s.empty()) return out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        const auto dash = tok.find('-');
        try {
            size_t used = 0;
            const int a = std::stoi(tok.substr(0, dash), &used);
            if (used != tok.substr(0, dash).size() || a < 0) return {};
            int b = a;
            if (dash != std::string::npos) {
                const std::string t2 = tok.substr(dash + 1);
                b = std::stoi(t2, &used);
                if (used != t2.size() || b < a) return {};
            }
            if (b - a > 4096) return {};
            for (int c = a; c <= b; ++c) out.push_back(c);
        } catch (...) { return {}; }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

std::vector<int> dispatch_cpus(const std::vector<std::string>& bdfs, const std::string& override_list,
                               std::string* why, const std::string& root) {
    auto say = [&](const std::string& m) { if (why) *why = m; };
    if (!override_list.empty()) {
        auto v = parse_cpulist(override_list);
        say(v.empty() ? "VEGAJET_DISPATCH_CPUS is malformed; affinity unchanged" : "VEGAJET_DISPATCH_CPUS=" + override_list);
        return v;
    }
    const std::string poss = read_text(fs::path(root) / "devices/system/node/possible");
    if (poss.empty() || poss.find_first_of(",-") == std::string::npos) {
        say("single NUMA node (UMA); affinity unchanged, set VEGAJET_DISPATCH_CPUS to pin");
        return {};
    }
    int node = -2;
    for (const auto& b : bdfs) {
        const std::string t = read_text(fs::path(root) / "bus/pci/devices" / b / "numa_node");
        const int n = t.empty() ? -1 : static_cast<int>(number(t));
        if (n < 0) { say("GPU " + b + " has no NUMA node; affinity unchanged"); return {}; }
        if (node == -2) node = n;
        else if (node != n) { say("GPUs span NUMA nodes; affinity unchanged"); return {}; }
    }
    if (node < 0) { say("no GPUs; affinity unchanged"); return {}; }
    auto v = parse_cpulist(read_text(fs::path(root) / ("devices/system/node/node" + std::to_string(node)) / "cpulist"));
    say(v.empty() ? "node cpulist unreadable; affinity unchanged" : "NUMA node " + std::to_string(node) + " of all GPUs");
    return v;
}

std::vector<int> pin_dispatch_thread(const std::vector<std::string>& bdfs) {
    std::string why;
    const char* ov = std::getenv("VEGAJET_DISPATCH_CPUS");
    auto cpus = dispatch_cpus(bdfs, ov ? ov : "", &why);
    if (!cpus.empty()) {
        cpu_set_t set;
        CPU_ZERO(&set);
        for (int c : cpus) if (c < CPU_SETSIZE) CPU_SET(c, &set);
        if (sched_setaffinity(0, sizeof(set), &set) != 0) { why += " (sched_setaffinity failed; affinity unchanged)"; cpus.clear(); }
    }
    std::fprintf(stderr, "[fabric] dispatch thread: %s%s\n", why.c_str(), cpus.empty() ? "" : (" -> " + std::to_string(cpus.size()) + " CPUs").c_str());
    return cpus;
}

void run_and_log(const std::vector<std::string>& bdfs) {
    for (const auto& w : run(bdfs))
        std::fprintf(stderr, "[WARN] fabric %s %s: %s\n", w.bdf.c_str(), w.check.c_str(), w.detail.c_str());
}
} // namespace vegajet::hw_preflight
