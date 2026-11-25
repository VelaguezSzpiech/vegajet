// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/doctor_rules.hpp"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <thread>

#include <sys/stat.h>
#include <unistd.h>

#include <grp.h>

namespace fs = std::filesystem;

namespace vegajet::doctor_rules {

const char* status_name(Status s) {
    switch (s) {
        case Status::Pass: return "PASS";
        case Status::Warn: return "WARN";
        case Status::Fail: return "FAIL";
    }
    return "?";
}

const Check* Report::find(const std::string& name) const {
    for (const auto& c : checks) if (c.name == name) return &c;
    return nullptr;
}

namespace {

std::string trim(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    return s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}

// Reads at most 64 KiB of a small text file; never follows into device nodes.
bool read_text(const std::string& path, std::string& out) {
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) return false;
    std::ifstream f(path);
    if (!f) return false;
    out.assign(65536, '\0');
    f.read(&out[0], static_cast<std::streamsize>(out.size()));
    out.resize(static_cast<size_t>(f.gcount()));
    return true;
}

bool read_line(const std::string& path, std::string& out) {
    std::string t;
    if (!read_text(path, t)) return false;
    out = trim(t.substr(0, t.find('\n')));
    return !out.empty();
}

bool read_u64(const std::string& path, uint64_t& v, int base = 10) {
    std::string s;
    if (!read_line(path, s)) return false;
    char* end = nullptr;
    const unsigned long long x = std::strtoull(s.c_str(), &end, base);
    if (end == s.c_str()) return false;
    v = x;
    return true;
}

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return buf;
}

std::string join(const std::string& a, const std::string& b) {
    if (a.empty() || a == "/") return "/" + (b[0] == '/' ? b.substr(1) : b);
    return a + (b[0] == '/' ? b : "/" + b);
}

bool is_device_id_vega20(uint32_t did) { return did == 0x66a0u || did == 0x66a1u || did == 0x66a3u; }

Hop read_hop(const fs::path& dir) {
    Hop h;
    h.name = dir.filename().string();
    std::string s;
    uint64_t w = 0;
    if (read_line((dir / "current_link_speed").string(), s)) h.cur_gts = parse_link_speed_gts(s);
    if (read_line((dir / "max_link_speed").string(), s)) h.max_gts = parse_link_speed_gts(s);
    if (read_u64((dir / "current_link_width").string(), w)) h.cur_width = static_cast<uint32_t>(w);
    if (read_u64((dir / "max_link_width").string(), w)) h.max_width = static_cast<uint32_t>(w);
    return h;
}

// Walk the canonical device path from the top: every ancestor directory that exposes current_link_speed is a PCIe
// hop. The first is the root port, the second the device below it (the on-card switch upstream port on Vega 20).
void read_host_link(const std::string& dev_link, GpuFacts& g) {
    std::error_code ec;
    const fs::path real = fs::canonical(dev_link, ec);
    if (ec) return;
    std::vector<fs::path> chain;
    fs::path cur;
    for (const auto& part : real) {
        cur /= part;
        if (fs::exists(cur / "current_link_speed", ec) && fs::exists(cur / "max_link_speed", ec)) chain.push_back(cur);
    }
    if (chain.size() < 2) return;
    g.root_port = read_hop(chain[0]);
    g.upstream = read_hop(chain[1]);
    g.link_found = g.root_port.max_gts > 0 && g.upstream.max_gts > 0;
}

double parse_mclk_level0(const std::string& text) {
    std::istringstream is(text);
    std::string line;
    while (std::getline(is, line)) {
        unsigned idx = 0;
        double mhz = 0;
        if (std::sscanf(line.c_str(), " %u: %lfMhz", &idx, &mhz) == 2 && idx == 0) return mhz;
    }
    return -1;
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else if (static_cast<unsigned char>(c) < 0x20) o += fmt("\\u%04x", c);
        else o += c;
    }
    return o;
}

} // namespace

bool parse_osrelease(const std::string& s, int& major, int& minor) {
    return std::sscanf(s.c_str(), "%d.%d", &major, &minor) == 2;
}

double parse_link_speed_gts(const std::string& s) {
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    return end == s.c_str() ? 0.0 : v;
}

std::vector<std::string> split_cmdline(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream is(s);
    std::string t;
    while (is >> t) out.push_back(t);
    return out;
}

AccessIdentity current_access_identity() {
    AccessIdentity id;
    id.uid = ::geteuid();
    const int n = ::getgroups(0, nullptr);
    if (n > 0) {
        id.groups.resize(static_cast<size_t>(n));
        const int m = ::getgroups(n, id.groups.data());
        id.groups.resize(m > 0 ? static_cast<size_t>(m) : 0u);
    }
    id.groups.push_back(::getegid());
    id.group_name = [](gid_t g) {
        if (const struct group* gr = ::getgrgid(g)) return std::string(gr->gr_name);
        return "gid " + std::to_string(g);
    };
    return id;
}

bool device_rw_allowed(unsigned mode, uid_t owner, gid_t group, const AccessIdentity& who) {
    if (who.uid == 0) return true;
    mode_t r, w;
    if (owner == who.uid) { r = S_IRUSR; w = S_IWUSR; }
    else if (std::find(who.groups.begin(), who.groups.end(), group) != who.groups.end()) { r = S_IRGRP; w = S_IWGRP; }
    else { r = S_IROTH; w = S_IWOTH; }
    return (mode & static_cast<unsigned>(r)) && (mode & static_cast<unsigned>(w));
}

Report run(const Options& opt) {
    Report rep;
    const std::string R = opt.root;
    const Limits& L = opt.limits;
    // Reference-rig invariants (32 GiB BAR0, N-1 P2P links, the PSU power budget, the guard/fan units) are HARD
    // requirements only for the G-SF-12 pre-flight of the reference box. On any other host they are WARN: a stock
    // single card or a 16 GB card is a valid alpha setup, and `vegajet doctor --os` must not fail it.
    // (Named `preflight`, not `pre`: the per-GPU loop below already has a `pre` check-name prefix string.)
    const bool preflight = opt.mode == Mode::Gsf12Preflight;
    auto add = [&](const std::string& name, Status st, const std::string& detail) {
        rep.checks.push_back({name, st, detail});
        (st == Status::Pass ? rep.pass : st == Status::Warn ? rep.warn : rep.fail)++;
    };
    auto service_active = opt.service_active ? opt.service_active : [&R](const std::string& unit) {
        std::error_code ec;
        return fs::is_directory(join(R, "sys/fs/cgroup/system.slice/" + unit), ec);
    };
    auto sleep_fn = opt.sleep_fn ? opt.sleep_fn : [](uint32_t ms, uint32_t) {
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    };
    const AccessIdentity who = opt.identity_explicit ? opt.identity : current_access_identity();

    // ---- kernel
    {
        std::string rel;
        int ma = 0, mi = 0;
        if (!read_line(join(R, "proc/sys/kernel/osrelease"), rel) || !parse_osrelease(rel, ma, mi))
            add("kernel.osrelease", Status::Warn, "kernel release unreadable");
        else if (ma >= 7)
            add("kernel.osrelease", Status::Pass, "Linux " + rel + " (>= 7.0 KFD baseline)");
        else
            add("kernel.osrelease", Status::Fail, "Linux " + rel + " is older than the 7.0 KFD baseline this engine targets");
    }
    {
        std::string cl;
        read_line(join(R, "proc/cmdline"), cl);
        const auto tok = split_cmdline(cl);
        auto has = [&](const char* t) { return std::find(tok.begin(), tok.end(), t) != tok.end(); };
        std::string bad;
        bool noretry_set = false;
        std::string noretry_tok;
        for (const auto& t : tok) {
            if (t.rfind("amdgpu.vm_block_size=", 0) == 0 || t.rfind("amdgpu.vm_size=", 0) == 0) bad += t + " ";
            if ((t.rfind("amdgpu.noretry=", 0) == 0 && t != "amdgpu.noretry=0") || t == "amdgpu.noretry") {
                noretry_set = true;
                noretry_tok = t;
            }
        }
        if (!bad.empty())
            add("kernel.cmdline.unsafe_vm", Status::Fail,
                "unsafe VM flag(s) active: " + trim(bad) + "; remove them and keep only amdgpu.vm_fragment_size=9");
        else
            add("kernel.cmdline.unsafe_vm", Status::Pass, "no amdgpu.vm_block_size / amdgpu.vm_size override");
        if (noretry_set) add("kernel.cmdline.noretry", Status::Warn, noretry_tok + " is set; recommend dropping it");
        if (has("amdgpu.vm_fragment_size=9")) add("kernel.cmdline.vm_fragment_size", Status::Pass, "amdgpu.vm_fragment_size=9 active");
        else add("kernel.cmdline.vm_fragment_size", Status::Warn, "amdgpu.vm_fragment_size=9 missing (2 MiB PTE fragments)");
        if (has("iommu=pt")) add("kernel.cmdline.iommu_pt", Status::Pass, "iommu=pt active");
        else add("kernel.cmdline.iommu_pt", Status::Warn, "iommu=pt missing (recommended for BAR0 peer-to-peer)");
    }

    // ---- KFD (sysfs topology + device node presence by stat only)
    uint32_t kfd_gfx906 = 0;
    std::vector<uint32_t> kfd_p2p;
    {
        std::error_code ec;
        const fs::path nodes = join(R, "sys/class/kfd/kfd/topology/nodes");
        if (fs::is_directory(nodes, ec)) {
            std::vector<fs::path> ds;
            for (const auto& e : fs::directory_iterator(nodes, ec)) ds.push_back(e.path());
            std::sort(ds.begin(), ds.end());
            for (const auto& d : ds) {
                std::string t;
                if (!read_text((d / "properties").string(), t)) continue;
                std::istringstream is(t);
                std::string k;
                uint64_t v = 0, gfx = 0, p2p = 0;
                while (is >> k >> v) {
                    if (k == "gfx_target_version") gfx = v;
                    else if (k == "p2p_links_count") p2p = v;
                }
                if (gfx == 90006) { ++kfd_gfx906; kfd_p2p.push_back(static_cast<uint32_t>(p2p)); }
            }
        }
        struct stat st{};
        const std::string kfd_path = join(R, "dev/kfd");
        if (::stat(kfd_path.c_str(), &st) == 0) {
            add("kfd.device_node", Status::Pass, "/dev/kfd present (stat only, not opened)");
            const std::string mode = fmt("%04o", static_cast<unsigned>(st.st_mode & 07777));
            const std::string grp = who.group_name ? who.group_name(st.st_gid) : "gid " + std::to_string(st.st_gid);
            if (device_rw_allowed(static_cast<unsigned>(st.st_mode), st.st_uid, st.st_gid, who))
                add("kfd.access", Status::Pass,
                    "/dev/kfd is readable+writable by this user (mode " + mode + ", group " + grp + ")");
            else
                add("kfd.access", Status::Fail,
                    "/dev/kfd is not readable+writable by this user (mode " + mode + ", group " + grp +
                        "): run `sudo usermod -aG " + grp + " $USER`, then log out and back in");
        } else {
            add("kfd.device_node", Status::Fail, "/dev/kfd missing: amdgpu loaded without KFD, or the driver is not bound");
            add("kfd.access", Status::Fail,
                "/dev/kfd missing: load amdgpu with KFD (ROCm-capable kernel) and check `dmesg | grep kfd`");
        }
    }

    // ---- GPU discovery
    {
        std::error_code ec;
        const fs::path pci = join(R, "sys/bus/pci/devices");
        std::vector<std::string> devs;
        if (fs::is_directory(pci, ec))
            for (const auto& e : fs::directory_iterator(pci, ec)) devs.push_back(e.path().filename().string());
        std::sort(devs.begin(), devs.end());
        for (const auto& d : devs) {
            const std::string p = (pci / d).string();
            uint64_t vid = 0, did = 0;
            if (!read_u64(p + "/vendor", vid, 16) || vid != 0x1002 || !read_u64(p + "/device", did, 16) || !is_device_id_vega20(static_cast<uint32_t>(did)))
                continue;
            GpuFacts g;
            g.dev = d;
            g.device_id = static_cast<uint32_t>(did);
            read_line(p + "/vbios_version", g.vbios);
            std::string res;
            if (read_line(p + "/resource", res)) {
                unsigned long long s = 0, e = 0;
                if (std::sscanf(res.c_str(), "%llx %llx", &s, &e) == 2 && e >= s) { g.bar0_bytes = e - s + 1; g.bar0_end = e; }
            }
            fs::path hw;
            if (fs::is_directory(p + "/hwmon", ec)) {
                std::vector<fs::path> hs;
                for (const auto& e : fs::directory_iterator(p + "/hwmon", ec))
                    if (e.path().filename().string().rfind("hwmon", 0) == 0) hs.push_back(e.path());
                std::sort(hs.begin(), hs.end());
                if (!hs.empty()) hw = hs[0];
            }
            if (!hw.empty()) {
                uint64_t v = 0;
                if (read_u64((hw / "power1_cap").string(), v)) g.power_cap_w = v * 1e-6;
                if (read_u64((hw / "temp3_input").string(), v)) g.tmem_c = v * 1e-3;
                for (uint32_t i = 0; i < std::max(1u, opt.power_samples); ++i) {
                    if (i) sleep_fn(opt.sample_interval_ms, i);
                    if (read_u64((hw / "power1_input").string(), v) || read_u64((hw / "power1_average").string(), v))
                        g.power_samples_w.push_back(v * 1e-6);
                }
            }
            std::string mclk;
            if (read_text(p + "/pp_dpm_mclk", mclk)) g.uclk0_mhz = parse_mclk_level0(mclk);
            for (const char* blk : {"umc", "gfx", "hdp", "mmhub", "pcie_bif", "sdma"}) {
                RasCount rc;
                rc.block = blk;
                std::string t;
                if (read_text(p + "/ras/" + blk + "_err_count", t)) {
                    rc.available = true;
                    std::istringstream is(t);
                    std::string k;
                    uint64_t v = 0;
                    while (is >> k >> v) {
                        if (k == "ue:") rc.ue = v;
                        else if (k == "ce:") rc.ce = v;
                        else if (k == "de:") rc.de = v;
                    }
                }
                g.ras.push_back(rc);
            }
            read_host_link(p, g);
            rep.gpus.push_back(std::move(g));
        }
    }
    const size_t N = rep.gpus.size();
    if (N == 0) add("gpu.discovery", Status::Fail, "no Vega 20 / gfx906 GPU found under sys/bus/pci/devices");
    else add("gpu.discovery", Status::Pass, fmt("%zu gfx906 GPU(s)", N));

    if (kfd_gfx906 == 0) add("kfd.topology", N ? Status::Fail : Status::Warn, "no gfx906 node (gfx_target_version 90006) in the KFD topology");
    else if (kfd_gfx906 != N) add("kfd.topology", Status::Warn, fmt("%u gfx906 KFD node(s) for %zu PCI GPU(s)", kfd_gfx906, N));
    else add("kfd.topology", Status::Pass, fmt("%u gfx906 KFD node(s)", kfd_gfx906));

    if (N) {
        std::set<std::string> vb;
        for (const auto& g : rep.gpus) vb.insert(g.vbios);
        if (vb.size() == 1 && !vb.begin()->empty()) add("gpu.vbios_parity", Status::Pass, "identical VBIOS " + *vb.begin());
        else add("gpu.vbios_parity", preflight ? Status::Fail : Status::Warn,
                 "VBIOS differs across GPUs (or unreadable); only the reference 2x-BAR0 P2P setup needs matching VBIOS");
    }

    const double cap_lim = N >= 3 ? L.cap_w_large : L.cap_w_small;
    const double avg_lim = N >= 3 ? L.avg_w_large : L.avg_w_small;
    for (size_t gi = 0; gi < N; ++gi) {
        const GpuFacts& g = rep.gpus[gi];
        const std::string pre = "gpu." + g.dev + ".";
        // BAR0
        {
            const bool big = g.bar0_bytes >= (32ull << 30), low = g.bar0_end < 0x100000000000ull;
            add(pre + "bar0", big && low ? Status::Pass : (preflight ? Status::Fail : Status::Warn),
                fmt("BAR0 %.2f GiB (32 GiB=%s, below 44-bit=%s)%s", g.bar0_bytes / double(1ull << 30),
                    big ? "yes" : "NO", low ? "yes" : "NO",
                    big && low ? "" : "; a small BAR0 only disables the 2x 32 GiB P2P path (TP>1), single-GPU runs are fine"));
        }
        if (N >= 2 && gi < kfd_p2p.size()) {
            add(pre + "p2p_links", kfd_p2p[gi] >= N - 1 ? Status::Pass : (preflight ? Status::Fail : Status::Warn),
                fmt("KFD p2p_links_count=%u (expected >= %zu for TP>1)%s", kfd_p2p[gi], N - 1,
                    kfd_p2p[gi] >= N - 1 ? "" : "; use --gpu to run N=1 workers instead"));
        }
        // RAS
        {
            uint64_t ue = 0, de = 0, ce = 0;
            int avail = 0;
            std::string bad;
            for (const auto& r : g.ras) {
                ue += r.ue; de += r.de; ce += r.ce; avail += r.available;
                if (r.ue || r.de) bad += " " + r.block;
            }
            const std::string d = fmt("6-block RAS: UE=%" PRIu64 " DE=%" PRIu64 " CE=%" PRIu64 " (blocks readable %d/6)", ue, de, ce, avail);
            if (ue || de) add(pre + "ras_ecc", Status::Fail, d + "; uncorrectable/deferred in:" + bad);
            else if (avail < 6) add(pre + "ras_ecc", Status::Warn, d);
            else if (ce) add(pre + "ras_ecc", Status::Warn, d + "; correctable errors present, watch the trend");
            else add(pre + "ras_ecc", Status::Pass, d);
        }
        // PCIe host link
        if (g.link_found) {
            const double cap = std::min(g.root_port.max_gts, g.upstream.max_gts);
            const uint32_t capw = std::min(g.root_port.max_width, g.upstream.max_width);
            const double cur = g.root_port.cur_gts;
            const bool below = cur + 0.05 < cap;
            std::string d = fmt("host link %s<->%s: %.1f GT/s x%u now, capable %.1f GT/s x%u", g.root_port.name.c_str(),
                                g.upstream.name.c_str(), cur, g.root_port.cur_width, cap, capw);
            if (capw < 16) d += " (x8 slot: informational, never a failure)";
            if (below) d += cur < 2.6 ? "; trained at Gen1 2.5 GT/s" : "; below capability";
            if (below) d += ", confirm under load with `vegajet doctor --live-link`";
            add(pre + "pcie_host_link", cap < L.min_link_gts ? Status::Fail : below ? Status::Warn : Status::Pass, d);
        } else {
            add(pre + "pcie_host_link", Status::Warn, "host link not found (no root port above the device)");
        }
        // HBM2 temperature
        if (g.tmem_c >= 0) {
            const Status st = g.tmem_c >= L.tmem_fail_c ? Status::Fail : g.tmem_c >= L.tmem_warn_c ? Status::Warn : Status::Pass;
            add(pre + "tmem", st, fmt("HBM2 TMem %.1f C (warn >= %.0f, fail >= %.0f)", g.tmem_c, L.tmem_warn_c, L.tmem_fail_c));
        }
        // Power cap
        if (g.power_cap_w >= 0)
            add(pre + "psu_power_cap", g.power_cap_w <= cap_lim + 0.5 ? Status::Pass : (preflight ? Status::Fail : Status::Warn),
                fmt("power1_cap %.1f W (reference-rig PSU budget <= %.0f W for N=%zu)", g.power_cap_w, cap_lim, N));
        else
            add(pre + "psu_power_cap", Status::Warn, "power1_cap unreadable");
        // Power draw
        if (!g.power_samples_w.empty()) {
            double sum = 0, peak = 0;
            for (double w : g.power_samples_w) { sum += w; peak = std::max(peak, w); }
            const double mean = sum / g.power_samples_w.size();
            const std::string d = fmt("power1_input mean %.1f W over %zu sample(s) (<= %.0f W), peak %.1f W (< %.0f W)", mean,
                                      g.power_samples_w.size(), avg_lim, peak, L.peak_w);
            add(pre + "power_draw", (mean > avg_lim + 0.5 || peak >= L.peak_w) ? Status::Fail : Status::Pass, d);
        } else {
            add(pre + "power_draw", Status::Warn, "power1_input / power1_average unreadable");
        }
        // G-SF-12 invariant: memory DPM level 0 stays at 350 MHz.
        if (g.uclk0_mhz >= 0)
            add(pre + "uclk0", std::abs(g.uclk0_mhz - L.uclk0_mhz) < 0.5 ? Status::Pass : Status::Fail,
                fmt("pp_dpm_mclk level 0 = %.0f MHz (G-SF-12 keeps FreqTableUclk/0 = %.0f)", g.uclk0_mhz, L.uclk0_mhz));
    }

    // ---- G-SF-12: GPU job and sensor-reader exclusion
    {
        std::error_code ec;
        uint32_t holders = 0, unreadable = 0;
        std::string who;
        const fs::path proc = join(R, "proc");
        if (fs::is_directory(proc, ec)) {
            for (const auto& e : fs::directory_iterator(proc, ec)) {
                const std::string pid = e.path().filename().string();
                if (pid.empty() || pid.find_first_not_of("0123456789") != std::string::npos) continue;
                std::error_code ec2;
                fs::directory_iterator it(e.path() / "fd", ec2);
                if (ec2) { ++unreadable; continue; }
                for (const auto& fd : it) {
                    char buf[256];
                    const ssize_t n = ::readlink(fd.path().c_str(), buf, sizeof(buf) - 1);
                    if (n > 0 && std::string(buf, static_cast<size_t>(n)) == "/dev/kfd") {
                        ++holders;
                        if (who.size() < 64) who += " " + pid;
                        break;
                    }
                }
            }
        }
        // `preflight` comes from the top of run(): the reference-rig guard/fan units are only mandatory there.
        std::string d = fmt("%u process(es) hold /dev/kfd", holders);
        if (holders) d += " (pid" + who + ")";
        if (unreadable) d += fmt("; %u process fd table(s) unreadable (run as root to prove zero)", unreadable);
        if (holders) add("gsf12.kfd_holders", preflight ? Status::Fail : Status::Warn, d + (preflight ? "" : "; GPU runs must not overlap"));
        else if (unreadable && preflight) add("gsf12.kfd_holders", Status::Warn, d);
        else add("gsf12.kfd_holders", Status::Pass, d);

        for (const auto& u : opt.guard_units) {
            const bool on = service_active(u);
            if (preflight) add("gsf12.service." + u, on ? Status::Fail : Status::Pass, u + (on ? " active: stop it before a pp_table write" : " stopped"));
            else add("gsf12.service." + u, on ? Status::Pass : Status::Warn, u + (on ? " active" : " inactive: no thermal/power trip logging"));
        }
        for (const auto& u : opt.fan_units) {
            const bool on = service_active(u);
            if (preflight) add("gsf12.service." + u, on ? Status::Fail : Status::Pass, u + (on ? " active: it reads amdgpu sensors, stop it first" : " stopped"));
            else add("gsf12.service." + u, on ? Status::Pass : Status::Warn,
                     u + (on ? " active" : " inactive: reference-rig blower owner not running (VegaJet never manages fans; ensure forced airflow)"));
        }
        for (const auto& u : opt.sensor_reader_units) {
            const bool on = service_active(u);
            add("gsf12.service." + u, on ? (preflight ? Status::Fail : Status::Warn) : Status::Pass,
                u + (on ? " active: competing amdgpu sensor reader" : " inactive"));
        }
    }
    return rep;
}

std::string format_text(const Report& r) {
    std::string o;
    for (const auto& c : r.checks) o += fmt("[%s] %-44s %s\n", status_name(c.status), c.name.c_str(), c.detail.c_str());
    o += fmt("summary: %d pass, %d warn, %d fail -> %s\n", r.pass, r.warn, r.fail, r.ok() ? "OK" : "FAIL");
    return o;
}

std::string to_json(const Report& r) {
    std::string o = "{\"checks\":[";
    for (size_t i = 0; i < r.checks.size(); ++i) {
        const auto& c = r.checks[i];
        o += std::string(i ? "," : "") + "{\"name\":\"" + json_escape(c.name) + "\",\"status\":\"" + status_name(c.status) +
             "\",\"detail\":\"" + json_escape(c.detail) + "\"}";
    }
    o += fmt("],\"pass\":%d,\"warn\":%d,\"fail\":%d,\"ok\":%s}", r.pass, r.warn, r.fail, r.ok() ? "true" : "false");
    return o;
}

} // namespace vegajet::doctor_rules
