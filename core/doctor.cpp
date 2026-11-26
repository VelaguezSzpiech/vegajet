#include "core/doctor.hpp"
#include "core/portability.hpp"
#ifndef VEGAJET_CPU_ONLY
#include "core/kfd_runtime.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <thread>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <unordered_map>

#include "core/host_preflight.hpp"
#include "core/tp_executor.hpp"

namespace vegajet {
namespace doctor {

namespace {

constexpr uint64_t k32GiB         = 32ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr uint64_t k44BitCeiling  = 0x100000000000ULL;
constexpr double   kTmemWarnC     = 70.0;

std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

bool read_first_line(const std::string& path, std::string& out) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::getline(f, out);
    out = trim(out);
    return !out.empty();
}

bool read_hex_u32(const std::string& path, uint32_t& out) {
    std::string s;
    if (!read_first_line(path, s)) return false;
    char* end = nullptr;
    unsigned long v = std::strtoul(s.c_str(), &end, 0);
    if (end == s.c_str()) return false;
    out = static_cast<uint32_t>(v);
    return true;
}

bool read_u64_dec(const std::string& path, uint64_t& out) {
    std::string s;
    if (!read_first_line(path, s)) return false;
    char* end = nullptr;
    unsigned long long v = std::strtoull(s.c_str(), &end, 10);
    if (end == s.c_str()) return false;
    out = static_cast<uint64_t>(v);
    return true;
}

struct KfdNodeEntry {
    uint32_t node_id = 0;
    uint32_t domain = 0;
    uint32_t location_id = 0;
    uint32_t p2p_links_count = 0;
    uint32_t gfx_target_version = 0;
};

// KFD location_id encodes PCI BDF as (bus << 8) | (dev << 3) | func
std::string format_bdf(uint32_t domain, uint32_t loc_id) {
    uint32_t bus  = (loc_id >> 8) & 0xFFu;
    uint32_t dev  = (loc_id >> 3) & 0x1Fu;
    uint32_t func = loc_id & 0x7u;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04x:%02x:%02x.%u", domain, bus, dev, func);
    return std::string(buf);
}

std::unordered_map<std::string, KfdNodeEntry> scan_kfd_sysfs_gfx906() {
    std::unordered_map<std::string, KfdNodeEntry> out;
    const std::string base = "/sys/class/kfd/kfd/topology/nodes";
    DIR* dir = opendir(base.c_str());
    if (!dir) return out;

    while (auto* ent = readdir(dir)) {
        if (ent->d_name[0] < '0' || ent->d_name[0] > '9') continue;
        KfdNodeEntry e;
        e.node_id = static_cast<uint32_t>(std::strtoul(ent->d_name, nullptr, 10));
        std::ifstream pf(base + "/" + ent->d_name + "/properties");
        if (!pf.is_open()) continue;
        std::string key;
        uint64_t val = 0;
        while (pf >> key >> val) {
            if (key == "gfx_target_version") e.gfx_target_version = static_cast<uint32_t>(val);
            else if (key == "location_id") e.location_id = static_cast<uint32_t>(val);
            else if (key == "domain") e.domain = static_cast<uint32_t>(val);
            else if (key == "p2p_links_count") e.p2p_links_count = static_cast<uint32_t>(val);
        }
        if (e.gfx_target_version == 90006u) {
            out[format_bdf(e.domain, e.location_id)] = e;
        }
    }
    closedir(dir);
    return out;
}

bool is_vega20_mi50_did(uint32_t did) {
    // 0x66a0 (MI50 / Radeon VII, alternate VBIOS), 0x66a1 (MI50 stock), 0x66a3 (MI60/Vega20)
    return did == 0x66a0u || did == 0x66a1u || did == 0x66a3u;
}

std::string find_hwmon_dir(const std::string& pci_dir) {
    const std::string hbase = pci_dir + "/hwmon";
    DIR* dir = opendir(hbase.c_str());
    if (!dir) return "";
    std::string found;
    while (auto* ent = readdir(dir)) {
        if (std::string(ent->d_name).rfind("hwmon", 0) == 0) {
            found = hbase + "/" + ent->d_name;
            break;
        }
    }
    closedir(dir);
    return found;
}

RasBlockCount read_ras_block(const std::string& pci_dir, const std::string& blk) {
    RasBlockCount rc;
    rc.block = blk;
    std::ifstream f(pci_dir + "/ras/" + blk + "_err_count");
    if (!f.is_open()) return rc;
    rc.available = true;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream iss(line);
        std::string k;
        uint64_t v = 0;
        if (iss >> k >> v) {
            if (k == "ue:") rc.ue = v;
            else if (k == "ce:") rc.ce = v;
            else if (k == "de:") rc.de = v;
        }
    }
    return rc;
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '\\') out += "\\\\";
        else if (c == '"') out += "\\\"";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

const char* status_str(CheckStatus st) {
    switch (st) {
        case CheckStatus::Pass: return "PASS";
        case CheckStatus::Warn: return "WARN";
        case CheckStatus::Fail: return "FAIL";
    }
    return "UNKNOWN";
}

bool is_bdf_component(const std::string& c) {
    // dddd:bb:dd.f
    return c.size() == 12 && c[4] == ':' && c[7] == ':' && c[10] == '.';
}

PcieHop read_hop(const std::string& bdf) {
    PcieHop h;
    h.bdf = bdf;
    const std::string d = "/sys/bus/pci/devices/" + bdf;
    std::string s;
    if (read_first_line(d + "/current_link_speed", s)) h.cur_gts = parse_link_speed_gts(s);
    if (read_first_line(d + "/max_link_speed", s))     h.max_gts = parse_link_speed_gts(s);
    uint64_t w = 0;
    if (read_u64_dec(d + "/current_link_width", w)) h.cur_width = static_cast<uint32_t>(w);
    if (read_u64_dec(d + "/max_link_width", w))     h.max_width = static_cast<uint32_t>(w);
    return h;
}

void read_dpm_pcie(const std::string& bdf, HostLink& l) {
    l.dpm_levels.clear();
    l.dpm_active = -1;
    std::ifstream f("/sys/bus/pci/devices/" + bdf + "/pp_dpm_pcie");
    std::string line;
    while (f.is_open() && std::getline(f, line)) {
        PcieDpmLevel lv;
        if (!parse_pp_dpm_pcie_line(line, lv)) continue;
        if (lv.active) l.dpm_active = static_cast<int>(lv.index);
        l.dpm_levels.push_back(lv);
    }
}

// Walk the sysfs device path of the GPU up to its root port (no hardcoded BDFs).
HostLink read_host_link(const std::string& bdf) {
    HostLink l;
    std::error_code ec;
    const auto real = std::filesystem::canonical("/sys/bus/pci/devices/" + bdf, ec);
    if (ec) return l;
    std::vector<std::string> chain;
    for (const auto& part : real) {
        const std::string c = part.string();
        if (is_bdf_component(c)) chain.push_back(c);
    }
    read_dpm_pcie(bdf, l);
    if (chain.size() < 2) return l;
    l.root_port = read_hop(chain[0]);
    l.upstream  = read_hop(chain[1]);
    l.cap_gts   = std::min(l.root_port.max_gts, l.upstream.max_gts);
    l.cap_width = std::min(l.root_port.max_width, l.upstream.max_width);
    l.found     = l.root_port.max_gts > 0.0 && l.upstream.max_gts > 0.0;
    return l;
}

int read_busy_pct(const std::string& bdf) {
    uint64_t v = 0;
    return read_u64_dec("/sys/bus/pci/devices/" + bdf + "/gpu_busy_percent", v) ? static_cast<int>(v) : 0;
}

std::vector<std::string> scan_gfx906_bdfs() {
    auto kfd_map = scan_kfd_sysfs_gfx906();
    const std::string pci_root = "/sys/bus/pci/devices";
    std::vector<std::string> bdfs;
    if (DIR* dir = opendir(pci_root.c_str())) {
        while (auto* ent = readdir(dir)) {
            if (ent->d_name[0] == '.') continue;
            const std::string bdf = ent->d_name;
            uint32_t vid = 0, did = 0;
            if (!read_hex_u32(pci_root + "/" + bdf + "/vendor", vid) || vid != 0x1002u) continue;
            if (!read_hex_u32(pci_root + "/" + bdf + "/device", did)) continue;
            if (is_vega20_mi50_did(did) || kfd_map.count(bdf)) bdfs.push_back(bdf);
        }
        closedir(dir);
    }
    std::sort(bdfs.begin(), bdfs.end());
    return bdfs;
}

} // namespace

double parse_link_speed_gts(const std::string& s) {
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || v <= 0.0) return 0.0;
    return (std::string(end).find("GT/s") != std::string::npos) ? v : 0.0;
}

bool parse_pp_dpm_pcie_line(const std::string& line, PcieDpmLevel& out) {
    // "<idx>: <speed>GT/s, x<width> <lclk>Mhz [*]"
    unsigned idx = 0, width = 0;
    double gts = 0.0;
    if (std::sscanf(line.c_str(), " %u: %lfGT/s, x%u", &idx, &gts, &width) != 3) return false;
    out.index  = idx;
    out.gts    = gts;
    out.width  = width;
    out.active = line.find('*') != std::string::npos;
    return true;
}

LinkClassification classify_live_link(const std::vector<LinkSample>& samples) {
    LinkClassification c;
    c.samples = static_cast<uint32_t>(samples.size());
    int first_level = -2;
    for (const auto& s : samples) {
        if (first_level == -2) first_level = s.dpm_level;
        else if (s.dpm_level != first_level) c.dpm_level_changed = true;
        if (s.busy_pct < 50) continue;
        ++c.busy_samples;
        if (s.gts >= 7.9) ++c.busy_fast_samples;
    }
    if (c.busy_samples == 0) c.verdict = LinkVerdict::NoLoadSeen;
    else if (c.busy_fast_samples * 10u >= c.busy_samples * 9u) c.verdict = LinkVerdict::UpshiftsUnderLoad;
    else if (c.busy_fast_samples > 0) c.verdict = LinkVerdict::Partial;
    else c.verdict = LinkVerdict::StuckGen1;
    return c;
}

const char* link_verdict_name(LinkVerdict v) {
    switch (v) {
        case LinkVerdict::NoLoadSeen:        return "NO_LOAD_SEEN";
        case LinkVerdict::UpshiftsUnderLoad: return "UPSHIFTS_UNDER_LOAD";
        case LinkVerdict::Partial:           return "PARTIAL";
        case LinkVerdict::StuckGen1:         return "STUCK_GEN1_UNDER_LOAD";
    }
    return "UNKNOWN";
}

const char* link_verdict_action(LinkVerdict v) {
    switch (v) {
        case LinkVerdict::NoLoadSeen:
            return "Inconclusive: no sample with gpu_busy >= 50%. Re-run while a GPU kernel is running.";
        case LinkVerdict::UpshiftsUnderLoad:
            return "OK: the host link runs at >= 8 GT/s under load (any 2.5 GT/s reading is idle downclocking).";
        case LinkVerdict::Partial:
            return "Host link reaches >= 8 GT/s only part of the time under load; check link DPM hysteresis and AER counters.";
        case LinkVerdict::StuckGen1:
            return "Host link stays below 8 GT/s under load. Check `journalctl -k -b | grep 'available PCIe bandwidth'`: "
                   "if the link already enumerated at 2.5 GT/s before amdgpu loaded, the firmware trained it at Gen1 "
                   "(a root-port retrain and PowerPlay PcieGenSpeed edits do not raise it): set the slot to fixed Gen3 "
                   "in BIOS, add pcie_aspm=off, reseat the card or move an x8 card to a Gen3 x16 slot, then reboot.";
    }
    return "";
}

LiveLinkReport run_live_link(double seconds) {
    LiveLinkReport rep;
    rep.seconds = seconds;
    const auto bdfs = scan_gfx906_bdfs();
    std::vector<std::vector<LinkSample>> samples(bdfs.size());
    for (const auto& b : bdfs) {
        LiveLinkGpu g;
        g.bdf  = b;
        g.link = read_host_link(b);
        rep.gpus.push_back(std::move(g));
    }
    const auto t_end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    do {
        for (size_t i = 0; i < bdfs.size(); ++i) {
            if (!rep.gpus[i].link.found) continue;
            LinkSample s;
            s.busy_pct = read_busy_pct(bdfs[i]);
            std::string sp;
            if (read_first_line("/sys/bus/pci/devices/" + rep.gpus[i].link.root_port.bdf + "/current_link_speed", sp))
                s.gts = parse_link_speed_gts(sp);
            HostLink tmp;
            read_dpm_pcie(bdfs[i], tmp);
            s.dpm_level = tmp.dpm_active;
            if (s.gts > 0.0) samples[i].push_back(s);   // a failed sysfs read is not evidence of a slow link
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    } while (std::chrono::steady_clock::now() < t_end);

    for (size_t i = 0; i < bdfs.size(); ++i) {
        auto& g = rep.gpus[i];
        g.cls = classify_live_link(samples[i]);
        for (const auto& s : samples[i]) {
            if (g.min_gts == 0.0 || s.gts < g.min_gts) g.min_gts = s.gts;
            g.max_gts = std::max(g.max_gts, s.gts);
        }
        if (g.cls.verdict == LinkVerdict::StuckGen1 || g.cls.verdict == LinkVerdict::Partial) rep.any_warn = true;   // WARN, never FAIL: needs a firmware change
    }
    return rep;
}

void print_live_link_text(const LiveLinkReport& rep) {
    std::printf("=== VegaJet Doctor: live PCIe host-link check (%.1f s, 100 ms sampling, read-only sysfs) ===\n", rep.seconds);
    for (const auto& g : rep.gpus) {
        if (!g.link.found) { std::printf("  [%s] host link not found in sysfs\n", g.bdf.c_str()); continue; }
        std::printf("  [%s] host link %s <-> %s cap=%.1f GT/s x%u | samples=%u busy=%u busy@>=8GT/s=%u | speed min/max=%.1f/%.1f GT/s | "
                    "pp_dpm_pcie level changed=%s\n",
                    g.bdf.c_str(), g.link.root_port.bdf.c_str(), g.link.upstream.bdf.c_str(), g.link.cap_gts, g.link.cap_width,
                    g.cls.samples, g.cls.busy_samples, g.cls.busy_fast_samples, g.min_gts, g.max_gts,
                    g.cls.dpm_level_changed ? "yes" : "no");
        std::printf("      verdict: %s\n      action:  %s\n", link_verdict_name(g.cls.verdict), link_verdict_action(g.cls.verdict));
    }
    std::printf("\nSummary: %s\n", rep.any_fail ? "FAIL" : (rep.any_warn ? "WARN" : "OK"));
}

void print_live_link_json(const LiveLinkReport& rep) {
    std::printf("{\n  \"seconds\": %.1f,\n  \"result\": \"%s\",\n  \"gpus\": [\n", rep.seconds,
                rep.any_fail ? "FAIL" : (rep.any_warn ? "WARN" : "OK"));
    for (size_t i = 0; i < rep.gpus.size(); ++i) {
        const auto& g = rep.gpus[i];
        std::printf("    {\"bdf\": \"%s\", \"root_port\": \"%s\", \"upstream\": \"%s\", \"cap_gts\": %.1f, \"cap_width\": %u, "
                    "\"samples\": %u, \"busy_samples\": %u, \"busy_fast_samples\": %u, \"min_gts\": %.1f, \"max_gts\": %.1f, "
                    "\"dpm_level_changed\": %s, \"verdict\": \"%s\", \"action\": \"%s\"}%s\n",
                    json_escape(g.bdf).c_str(), json_escape(g.link.root_port.bdf).c_str(), json_escape(g.link.upstream.bdf).c_str(),
                    g.link.cap_gts, g.link.cap_width, g.cls.samples, g.cls.busy_samples, g.cls.busy_fast_samples,
                    g.min_gts, g.max_gts, g.cls.dpm_level_changed ? "true" : "false",
                    link_verdict_name(g.cls.verdict), json_escape(link_verdict_action(g.cls.verdict)).c_str(),
                    (i + 1 < rep.gpus.size()) ? "," : "");
    }
    std::printf("  ]\n}\n");
}

DoctorReport run_doctor() {
    DoctorReport rep;

    // 1. Read /proc/cmdline
    read_first_line("/proc/cmdline", rep.cmdline);

    auto add_check = [&](const std::string& name, CheckStatus st, const std::string& detail) {
        rep.checks.push_back(CheckResult{name, st, detail});
        if (st == CheckStatus::Pass) ++rep.pass_count;
        else if (st == CheckStatus::Warn) ++rep.warn_count;
        else ++rep.fail_count;
    };

    // Host compatibility (kernel, amdgpu/KFD, ROCm toolchain) and the supported-configuration summary: WARNs that
    // tell a first-time user what this host can run, never FAILs (the CPU tools work everywhere).
#ifdef VEGAJET_BUILD_ROCM_PATH
    const char* rocm_prefix = VEGAJET_BUILD_ROCM_PATH;  // -DVEGAJET_ROCM_PATH at configure time
#else
    const char* rocm_prefix = nullptr;                  // <root>/opt/rocm
#endif
    for (const auto& h : preflight::host_checks("/", rocm_prefix)) {
        const CheckStatus st = h.level == preflight::Level::Pass ? CheckStatus::Pass
                             : h.level == preflight::Level::Warn ? CheckStatus::Warn : CheckStatus::Fail;
        add_check(h.name, st, h.detail);
    }

    // Check unsafe Linux 7.0 amdgpu VM parameters. The engine never sets the VM geometry: any explicit
    // vm_block_size= / vm_size= override is unsafe, and noretry=1 changes fault handling the runtime depends on.
    {
        std::vector<std::string> tok;
        { std::istringstream is(rep.cmdline); std::string t; while (is >> t) tok.push_back(t); }
        auto has = [&](const char* t) { return std::find(tok.begin(), tok.end(), t) != tok.end(); };
        std::string bad_vm, noretry;
        for (const auto& t : tok) {
            if (t.rfind("amdgpu.vm_block_size=", 0) == 0 || t.rfind("amdgpu.vm_size=", 0) == 0) bad_vm += " " + t;
            if ((t.rfind("amdgpu.noretry=", 0) == 0 && t != "amdgpu.noretry=0") || t == "amdgpu.noretry") noretry = t;
        }
        const bool has_frag9    = has("amdgpu.vm_fragment_size=9");
        const bool has_iommu_pt = has("iommu=pt");

        if (!bad_vm.empty()) {
            add_check("kernel.cmdline.unsafe_vm", CheckStatus::Fail,
                      "Unsafe Linux 7.0 VM flag(s) active (" + trim(bad_vm) +
                      "): triggers 4 GiB Huge-PDE aliasing, CWSR BACO freezes, and ROCm SVM GPF. "
                      "Remove them and keep only amdgpu.vm_fragment_size=9.");
        } else {
            add_check("kernel.cmdline.unsafe_vm", CheckStatus::Pass,
                      "No explicit amdgpu.vm_block_size / amdgpu.vm_size override.");
        }

        if (!noretry.empty()) {
            add_check("kernel.cmdline.noretry", CheckStatus::Warn,
                      noretry + " is set; recommend dropping it for clean Linux 7.0 stability.");
        }

        if (has_frag9) {
            add_check("kernel.cmdline.vm_fragment_size", CheckStatus::Pass,
                      "amdgpu.vm_fragment_size=9 (2 MiB contiguous PTE fragments) is active.");
        } else {
            add_check("kernel.cmdline.vm_fragment_size", CheckStatus::Warn,
                      "amdgpu.vm_fragment_size=9 not found; the reference rig runs with it (2 MiB contiguous PTE "
                      "fragments) and the engine is unvalidated without it.");
        }

        if (!has_iommu_pt) {
            add_check("kernel.cmdline.iommu_pt", CheckStatus::Warn,
                      "iommu=pt not found; the reference rig uses it for direct PCIe BAR0 P2P (informational).");
        } else {
            add_check("kernel.cmdline.iommu_pt", CheckStatus::Pass, "iommu=pt is active.");
        }
    }

    // Device access (stat only; never opens a device node): a missing /dev/kfd or a missing render / video group
    // membership is the most common first-run failure on a new host, so say exactly which and how to fix it.
    for (const auto& a : portability::device_access("/", portability::current_identity()))
        add_check(a.name, a.verdict.status, a.verdict.detail);

    // 2. Scan KFD sysfs topology (read-only; never touches /dev/kfd)
    auto kfd_map = scan_kfd_sysfs_gfx906();

    // 3. Scan /sys/bus/pci/devices for AMD Vega 20 / gfx906 GPUs
    const std::string pci_root = "/sys/bus/pci/devices";
    std::vector<std::string> bdfs;
    if (DIR* dir = opendir(pci_root.c_str())) {
        while (auto* ent = readdir(dir)) {
            if (ent->d_name[0] == '.') continue;
            std::string bdf = ent->d_name;
            std::string pdir = pci_root + "/" + bdf;
            uint32_t vid = 0, did = 0;
            if (!read_hex_u32(pdir + "/vendor", vid) || vid != 0x1002u) continue;
            if (!read_hex_u32(pdir + "/device", did)) continue;
            if (is_vega20_mi50_did(did) || kfd_map.count(bdf)) {
                bdfs.push_back(bdf);
            }
        }
        closedir(dir);
    }
    std::sort(bdfs.begin(), bdfs.end());

    const std::vector<std::string> ras_blocks = {"umc", "gfx", "hdp", "mmhub", "pcie_bif", "sdma"};

    for (const auto& bdf : bdfs) {
        std::string pdir = pci_root + "/" + bdf;
        GpuHealth g;
        g.bdf = bdf;
        read_hex_u32(pdir + "/vendor", g.vendor_id);
        read_hex_u32(pdir + "/device", g.device_id);
        read_first_line(pdir + "/vbios_version", g.vbios_version);

        // BAR0 from line 0 of /sys/bus/pci/devices/<bdf>/resource
        std::string res0;
        if (read_first_line(pdir + "/resource", res0)) {
            std::istringstream iss(res0);
            uint64_t st = 0, en = 0, fl = 0;
            if (iss >> std::hex >> st >> en >> fl) {
                g.bar0_start = st;
                g.bar0_end   = en;
                if (en >= st) {
                    g.bar0_bytes = (en - st) + 1ULL;
                    g.bar0_32gib = (g.bar0_bytes >= k32GiB);
                    g.bar0_under_44bit = (en < k44BitCeiling);
                }
            }
        }

        auto kit = kfd_map.find(bdf);
        if (kit != kfd_map.end()) {
            g.kfd_matched     = true;
            g.kfd_node_id     = kit->second.node_id;
            g.p2p_links_count = kit->second.p2p_links_count;
        }

        // hwmon power1_cap (uW) and temp3_input (mC, HBM2 TMem)
        std::string hwdir = find_hwmon_dir(pdir);
        if (!hwdir.empty()) {
            uint64_t uw = 0, mc = 0;
            if (read_u64_dec(hwdir + "/power1_cap", uw)) g.power_cap_w = static_cast<double>(uw) * 1e-6;
            if (read_u64_dec(hwdir + "/temp3_input", mc)) g.tmem_c = static_cast<double>(mc) * 1e-3;
        }

        g.link = read_host_link(bdf);

        // 6-block RAS ECC health
        for (const auto& blk : ras_blocks) {
            auto rb = read_ras_block(pdir, blk);
            g.total_ue += rb.ue;
            g.total_de += rb.de;
            g.total_ce += rb.ce;
            g.ras.push_back(std::move(rb));
        }

        rep.gpus.push_back(std::move(g));
    }

    if (rep.gpus.empty()) {
        add_check("gpu.discovery", CheckStatus::Fail, "No AMD Vega 20 / gfx906 GPU (MI50 / MI60) detected in sysfs.");
        return rep;
    }

    add_check("gpu.discovery", CheckStatus::Pass,
              "Detected " + std::to_string(rep.gpus.size()) + " gfx906 Vega 20 GPU(s) via read-only sysfs.");
    {
        const auto v = portability::gpu_count(rep.gpus.size());
        add_check("gpu.count", v.status, v.detail);
    }

    // 4. VBIOS parity check across all detected MI50s
    {
        std::set<std::string> vbios_set;
        for (const auto& g : rep.gpus) vbios_set.insert(g.vbios_version);
        if (vbios_set.size() == 1 && !vbios_set.begin()->empty()) {
            add_check("gpu.vbios_parity", CheckStatus::Pass,
                      "All " + std::to_string(rep.gpus.size()) + " GPUs share identical VBIOS: " + *vbios_set.begin());
        } else {
            std::string msg = "VBIOS mismatch across GPUs: ";
            for (size_t i = 0; i < rep.gpus.size(); ++i) {
                if (i) msg += ", ";
                msg += rep.gpus[i].bdf + "=" + (rep.gpus[i].vbios_version.empty() ? "unknown" : rep.gpus[i].vbios_version);
            }
            add_check("gpu.vbios_parity", CheckStatus::Warn,
                      msg + " (informational: only the reference 2x-BAR0 P2P setup needs matching VBIOS)");
        }
    }

    // 5. Per-GPU BAR0, P2P, RAS ECC, TMem, and N-aware PSU power cap checks
    const size_t N = rep.gpus.size();

    for (const auto& g : rep.gpus) {
        // BAR0 32 GiB & < 0x100000000000
        {
            char buf[256];
            double gib = static_cast<double>(g.bar0_bytes) / (1024.0 * 1024.0 * 1024.0);
            std::snprintf(buf, sizeof(buf),
                          "%s BAR0 [0x%" PRIx64 "..0x%" PRIx64 "] = %.2f GiB (32GiB=%s, <0x100000000000=%s)",
                          g.bdf.c_str(), g.bar0_start, g.bar0_end, gib,
                          g.bar0_32gib ? "yes" : "NO", g.bar0_under_44bit ? "yes" : "NO");
            // Rig-specific (large-BAR VBIOS): WARN, not FAIL, so other cards get an informative report.
            const auto v = portability::bar0(g.bar0_bytes, g.bar0_under_44bit);
            add_check("gpu." + g.bdf + ".bar0", v.status, std::string(buf) + (v.status == CheckStatus::Pass ? "" : "; " + v.detail));
        }

        // KFD P2P links count
        if (N >= 2) {
            const auto v = portability::p2p_links(N, g.kfd_matched, g.p2p_links_count);
            add_check("gpu." + g.bdf + ".p2p_links", v.status, g.bdf + " KFD node " + std::to_string(g.kfd_node_id) + " " + v.detail);
        }

        // 6-block RAS ECC health
        {
            bool all_present = true;
            for (const auto& rb : g.ras) {
                if (!rb.available) all_present = false;
            }
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "%s 6-block RAS ECC: UE=%" PRIu64 " DE=%" PRIu64 " CE=%" PRIu64 " (blocks_readable=%s)",
                          g.bdf.c_str(), g.total_ue, g.total_de, g.total_ce, all_present ? "6/6" : "partial");
            if (g.total_ue > 0 || g.total_de > 0) {
                add_check("gpu." + g.bdf + ".ras_ecc", CheckStatus::Fail, buf);
            } else if (!all_present) {
                add_check("gpu." + g.bdf + ".ras_ecc", CheckStatus::Warn, buf);
            } else {
                add_check("gpu." + g.bdf + ".ras_ecc", CheckStatus::Pass, buf);
            }
        }

        // PCIe host link (root port <-> on-card switch upstream port). Speed below max at idle is normal;
        // only `--live-link` under load can tell idle downclocking from a link stuck at 2.5 GT/s.
        if (g.link.found) {
            char buf[320];
            const bool below = g.link.root_port.cur_gts + 0.05 < g.link.cap_gts;
            std::snprintf(buf, sizeof(buf), "%s host link %s<->%s: %.1f GT/s x%u now, capable %.1f GT/s x%u, pp_dpm_pcie level %d%s%s",
                          g.bdf.c_str(), g.link.root_port.bdf.c_str(), g.link.upstream.bdf.c_str(),
                          g.link.root_port.cur_gts, g.link.root_port.cur_width, g.link.cap_gts, g.link.cap_width, g.link.dpm_active,
                          g.link.cap_width < 16u ? " (x8 slot)" : "",
                          below ? "; below max, run `vegajet doctor --live-link` during GPU load" : "");
            // Width x8 (GPU in an x8 slot) is never a failure. A link capable of 8 GT/s but running slower is a
            // WARN: on the reference rig the firmware, not the driver, decides the trained speed.
            const CheckStatus st = g.link.cap_gts < 7.9 ? CheckStatus::Fail : (below ? CheckStatus::Warn : CheckStatus::Pass);
            add_check("gpu." + g.bdf + ".pcie_host_link", st, buf);
        }

        // HBM2 TMem thermal sanity
        if (g.tmem_c >= 0.0) {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "%s HBM2 TMem=%.1f C (limit < %.0f C)", g.bdf.c_str(), g.tmem_c, kTmemWarnC);
            add_check("gpu." + g.bdf + ".tmem", g.tmem_c < kTmemWarnC ? CheckStatus::Pass : CheckStatus::Fail, buf);
        }

        // N-aware PSU power cap sanity (200 W max for N<=2, 150 W max for N>=3)
        if (g.power_cap_w >= 0.0) {
            // The reference rig's PSU budget; on other hosts a stock cap above it is information, not a failure.
            const auto v = portability::power_cap(g.power_cap_w, N);
            add_check("gpu." + g.bdf + ".psu_power_cap", v.status, g.bdf + " " + v.detail);
        } else {
            add_check("gpu." + g.bdf + ".psu_power_cap", CheckStatus::Warn,
                      g.bdf + " hwmon power1_cap unreadable.");
        }
    }

    // TP executor: which rank counts this host can host, and which features stay on the TP=2 fast path.
    {
        const auto detected = static_cast<uint32_t>(rep.gpus.size());
        tp::FeatureFlags fast;
        fast.super_kernels = fast.asym_experts = fast.mtp_draft = fast.speculative = true;
        if (detected == 0) {
            add_check("tp.executor", CheckStatus::Warn,
                      "no gfx906 GPU detected: TP plans are host-side only (--tp N --dry-run)");
        } else {
            const std::string why = tp::admit_features(detected, fast);
            std::string d = "TP=2 uses the fused fast path; TP=1.." + std::to_string(detected) +
                            " use generic raw per-op execution (hardware parity unqualified)";
            if (why.empty()) d += " (fast-path features admitted).";
            else d += ". " + why;
            add_check("tp.executor", why.empty() ? CheckStatus::Pass : CheckStatus::Warn, d);
        }
    }

    return rep;
}

void print_report_text(const DoctorReport& rep) {
    std::printf("=== VegaJet Doctor (Read-Only CPU Pre-Flight | Zero /dev/kfd Access) ===\n");
    std::printf("Cmdline: %s\n\n", rep.cmdline.c_str());

    std::printf("Detected MI50 (gfx906) GPUs: %zu\n", rep.gpus.size());
    for (const auto& g : rep.gpus) {
        double gib = static_cast<double>(g.bar0_bytes) / (1024.0 * 1024.0 * 1024.0);
        std::printf("  [%s] did=0x%04x vbios=%s BAR0=%.1f GiB (0x%" PRIx64 ") p2p_links=%u cap=%.0fW TMem=%.1fC RAS(UE=%" PRIu64 ",DE=%" PRIu64 ",CE=%" PRIu64 ")\n",
                    g.bdf.c_str(), g.device_id, g.vbios_version.c_str(), gib, g.bar0_start,
                    g.p2p_links_count, g.power_cap_w, g.tmem_c, g.total_ue, g.total_de, g.total_ce);
    }

    std::printf("\nPre-Flight Checks:\n");
    for (const auto& c : rep.checks) {
        std::printf("  [%-4s] %-34s %s\n", status_str(c.status), c.name.c_str(), c.detail.c_str());
    }
    std::printf("\nSummary: %d PASS | %d WARN | %d FAIL\n",
                rep.pass_count, rep.warn_count, rep.fail_count);
}

void print_report_json(const DoctorReport& rep) {
    std::printf("{\n");
    std::printf("  \"cmdline\": \"%s\",\n", json_escape(rep.cmdline).c_str());
    std::printf("  \"summary\": {\"pass\": %d, \"warn\": %d, \"fail\": %d},\n",
                rep.pass_count, rep.warn_count, rep.fail_count);
    std::printf("  \"gpus\": [\n");
    for (size_t i = 0; i < rep.gpus.size(); ++i) {
        const auto& g = rep.gpus[i];
        std::printf("    {\"bdf\": \"%s\", \"device_id\": \"0x%04x\", \"vbios\": \"%s\", "
                    "\"bar0_start\": \"0x%" PRIx64 "\", \"bar0_bytes\": %" PRIu64 ", "
                    "\"p2p_links_count\": %u, \"power_cap_w\": %.1f, \"tmem_c\": %.1f, "
                    "\"ras_ue\": %" PRIu64 ", \"ras_de\": %" PRIu64 ", \"ras_ce\": %" PRIu64 ", "
                    "\"host_link_root_port\": \"%s\", \"host_link_gts\": %.1f, \"host_link_width\": %u, "
                    "\"host_link_cap_gts\": %.1f, \"host_link_cap_width\": %u, \"pp_dpm_pcie_level\": %d}%s\n",
                    json_escape(g.bdf).c_str(), g.device_id, json_escape(g.vbios_version).c_str(),
                    g.bar0_start, g.bar0_bytes, g.p2p_links_count, g.power_cap_w, g.tmem_c,
                    g.total_ue, g.total_de, g.total_ce,
                    json_escape(g.link.root_port.bdf).c_str(), g.link.root_port.cur_gts, g.link.root_port.cur_width,
                    g.link.cap_gts, g.link.cap_width, g.link.dpm_active, (i + 1 < rep.gpus.size()) ? "," : "");
    }
    std::printf("  ],\n");
    std::printf("  \"checks\": [\n");
    for (size_t i = 0; i < rep.checks.size(); ++i) {
        const auto& c = rep.checks[i];
        std::printf("    {\"name\": \"%s\", \"status\": \"%s\", \"detail\": \"%s\"}%s\n",
                    json_escape(c.name).c_str(), status_str(c.status),
                    json_escape(c.detail).c_str(), (i + 1 < rep.checks.size()) ? "," : "");
    }
    std::printf("  ]\n}\n");
}

bool calibrate_and_save_hw_profile(const std::string& out_path) {
    namespace fs = std::filesystem;
    std::string path = out_path;
    if (path.empty()) {
        const char* home = std::getenv("HOME");
        path = (home && home[0]) ? (std::string(home) + "/.config/vegajet/hw_profile.json")
                                 : "/tmp/vegajet_hw_profile.json";
    }
    auto rep = run_doctor();
    // Map PCI location_id -> CU count from KFD topology (simd_count / 4)
    std::unordered_map<uint32_t, uint32_t> loc_cu;
    std::error_code ec;
    const fs::path kfd_dir("/sys/class/kfd/kfd/topology/nodes");
    if (fs::exists(kfd_dir, ec)) {
        for (const auto& ent : fs::directory_iterator(kfd_dir, ec)) {
            std::ifstream pf(ent.path() / "properties");
            if (!pf.is_open()) continue;
            uint32_t simd = 0, loc = 0, gfx = 0;
            std::string k; uint64_t v = 0;
            while (pf >> k >> v) {
                if (k == "simd_count") simd = static_cast<uint32_t>(v);
                else if (k == "location_id") loc = static_cast<uint32_t>(v);
                else if (k == "gfx_target_version") gfx = static_cast<uint32_t>(v);
            }
            if (gfx == 90006u && simd > 0u) loc_cu[loc] = simd / 4u;
        }
    }
    auto read_dpm_mhz = [](const fs::path& p, double fb) -> double {
        std::ifstream f(p);
        if (!f.is_open()) return fb;
        std::string line;
        double mx = 0.0, act = 0.0;
        while (std::getline(f, line)) {
            auto c = line.find(':');
            if (c == std::string::npos) continue;
            double v = std::atof(line.c_str() + c + 1);
            if (v > mx) mx = v;
            if (line.find('*') != std::string::npos) act = v;
        }
        return (act > 0.0) ? act : ((mx > 0.0) ? mx : fb);
    };

    // On GPU builds, measure true on-GPU HBM2 streaming bandwidth via s_memrealtime() (512 MiB chunk, 40 passes).
    std::unordered_map<std::string, double> measured_bw_gbs;
#ifndef VEGAJET_CPU_ONLY
    {
        kfd::Runtime rt;
        std::string kerr;
        if (rt.init(1.0, 16, &kerr) && rt.load_hsaco_dir(kfd::resolve_hsaco_dir(), &kerr) && rt.num_gpus() > 0) {
            constexpr uint64_t kChunkBytes = 512ULL * 1024ULL * 1024ULL;
            constexpr uint64_t kNVec = kChunkBytes / 16ULL;
            constexpr uint32_t kWgs = 240u;
            #pragma pack(push, 1)
            struct FillArgs   { uint64_t dst; uint64_t n_vec; uint32_t seed; uint32_t pad; };
            struct VerifyArgs { uint64_t src; uint64_t n_vec; uint32_t seed; uint32_t pad; uint64_t err_count; uint64_t ticks; };
            #pragma pack(pop)
            const size_t G = rt.num_gpus();
            struct CalGpu {
                const kfd::LoadedKernel* fill = nullptr;
                const kfd::LoadedKernel* verify = nullptr;
                uint64_t chunk_va = 0, err_va = 0, ticks_va = 0;
                std::string bdf;
                uint64_t tick_sum = 0, tick_first = 0, tick_last = 0;
                uint32_t passes = 0;
                double wall_first = 0.0, wall_last = 0.0;
            };
            std::vector<CalGpu> cg(G);
            bool ok = true;
            for (size_t g = 0; g < G; ++g) {
                cg[g].fill = rt.get_kernel(g, "kfd_soak_fill_u32x4");
                cg[g].verify = rt.get_kernel(g, "kfd_soak_verify_u32x4");
                cg[g].chunk_va = rt.vram_arena(g).alloc(kChunkBytes, 2ULL << 20);
                cg[g].err_va = rt.vram_arena(g).alloc(4096, 4096);
                cg[g].ticks_va = rt.vram_arena(g).alloc(2u * kWgs * sizeof(uint64_t), 4096);
                const uint32_t loc = rt.devices()[g].location_id;
                char bdf_buf[32];
                std::snprintf(bdf_buf, sizeof(bdf_buf), "0000:%02x:%02x.%u", (loc >> 8) & 0xFFu, (loc >> 3) & 0x1Fu, loc & 7u);
                cg[g].bdf = bdf_buf;
                if (!cg[g].fill || !cg[g].verify || !cg[g].chunk_va || !cg[g].err_va || !cg[g].ticks_va) { ok = false; break; }
            }
            if (ok) {
                for (size_t g = 0; g < G; ++g) {
                    const FillArgs fa{cg[g].chunk_va, kNVec, 0x5eed906u, 0u};
                    ok = ok && rt.dispatch_async(g, *cg[g].fill, kWgs * 256u, 1, 1, 256, 1, 1, &fa, sizeof(fa));
                }
                for (size_t g = 0; g < G; ++g) ok = ok && rt.wait_idle(g);
            }
            std::vector<uint64_t> ticks(2u * kWgs);
            auto now_s = []() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
            // 0.4 s warmup to ramp DPM clocks, then 1.2 s measured streaming for accurate s_memrealtime() calibration.
            const double t_warm_end = now_s() + 0.4;
            const double t_meas_end = t_warm_end + 1.2;
            const uint32_t zero = 0;
            for (size_t g = 0; ok && g < G; ++g) ok = ok && rt.upload_to_vram(g, cg[g].err_va, &zero, sizeof(zero));
            while (ok && now_s() < t_meas_end) {
                const double t0 = now_s();
                for (size_t g = 0; g < G; ++g) {
                    const VerifyArgs va{cg[g].chunk_va, kNVec, 0x5eed906u, 0u, cg[g].err_va, cg[g].ticks_va};
                    ok = ok && rt.dispatch_async(g, *cg[g].verify, kWgs * 256u, 1, 1, 256, 1, 1, &va, sizeof(va));
                }
                for (size_t g = 0; g < G; ++g) ok = ok && rt.wait_idle(g);
                const double t1 = now_s();
                if (!ok || t0 < t_warm_end) continue;
                for (size_t g = 0; g < G; ++g) {
                    if (!rt.download_from_vram(g, ticks.data(), cg[g].ticks_va, ticks.size() * sizeof(uint64_t))) { ok = false; break; }
                    uint64_t tmin = ~0ULL, tmax = 0ULL;
                    for (uint32_t w = 0; w < kWgs; ++w) { tmin = std::min(tmin, ticks[2u * w]); tmax = std::max(tmax, ticks[2u * w + 1u]); }
                    if (cg[g].tick_first == 0ULL) { cg[g].tick_first = tmin; cg[g].wall_first = t0; }
                    cg[g].tick_last = tmax; cg[g].wall_last = t1;
                    if (tmax > tmin) { cg[g].tick_sum += (tmax - tmin); ++cg[g].passes; }
                }
            }
            if (ok) {
                for (size_t g = 0; g < G; ++g) {
                    if (cg[g].tick_last > cg[g].tick_first && cg[g].tick_sum > 0ULL && cg[g].passes > 0u) {
                        const double tick_s = (cg[g].wall_last - cg[g].wall_first) / static_cast<double>(cg[g].tick_last - cg[g].tick_first);
                        const double total_gb = static_cast<double>(cg[g].passes) * static_cast<double>(kChunkBytes) / 1e9;
                        const double bw = total_gb / (static_cast<double>(cg[g].tick_sum) * tick_s);
                        if (bw > 100.0 && bw < 2000.0) measured_bw_gbs[cg[g].bdf] = bw;
                    }
                }
            }
        }
    }
#endif

    fs::path p(path);
    if (p.has_parent_path()) fs::create_directories(p.parent_path(), ec);
    std::ofstream ofs(path);
    if (!ofs.is_open()) return false;
    ofs << "{\n  \"gpus\": [\n";
    std::printf("\n=== VegaJet Roofline Calibration (--bench -> %s) ===\n", path.c_str());
    for (size_t i = 0; i < rep.gpus.size(); ++i) {
        const auto& g = rep.gpus[i];
        uint32_t n_cu = 60u;
        unsigned dom = 0, bus = 0, slot = 0, fn = 0;
        if (std::sscanf(g.bdf.c_str(), "%x:%x:%x.%x", &dom, &bus, &slot, &fn) == 4) {
            uint32_t loc = (bus << 8) | (slot << 3) | fn;
            if (loc_cu.count(loc)) n_cu = loc_cu[loc];
        }
        fs::path dev = fs::path("/sys/bus/pci/devices") / g.bdf;
        double sclk_ghz = read_dpm_mhz(dev / "pp_dpm_sclk", 1725.0) / 1000.0;
        double mclk_mhz = read_dpm_mhz(dev / "pp_dpm_mclk", 1080.0);
        const bool is_measured = measured_bw_gbs.count(g.bdf) > 0;
        double bw_gbs   = is_measured ? measured_bw_gbs[g.bdf] : (mclk_mhz * 0.665);
        double eta      = 0.55;
        double i_star   = (static_cast<double>(n_cu) * 64.0 * sclk_ghz * eta) / bw_gbs;
        double pcap     = (g.power_cap_w > 0.0) ? g.power_cap_w : 200.0;
        std::printf("  [%s] n_cu=%u sclk=%.3f GHz mclk=%.0f MHz bw=%.1f GB/s (%s) cap=%.0f W | I*=%.2f inst/B (M=5 2x reuse: %.2f)\n",
                    g.bdf.c_str(), n_cu, sclk_ghz, mclk_mhz, bw_gbs, is_measured ? "s_memrealtime" : "sysfs est",
                    pcap, i_star, i_star * 0.5);
        ofs << "    {\"bdf\": \"" << g.bdf
            << "\", \"n_cu\": " << n_cu
            << ", \"sclk_ghz\": " << sclk_ghz
            << ", \"mclk_mhz\": " << mclk_mhz
            << ", \"bw_hbm_gbs\": " << bw_gbs
            << ", \"power_cap_w\": " << pcap
            << ", \"eta\": " << eta
            << ", \"i_star\": " << i_star
            << "}" << (i + 1 < rep.gpus.size() ? "," : "") << "\n";
    }
    ofs << "  ]\n}\n";
    return ofs.good();
}

} // namespace doctor
} // namespace vegajet
