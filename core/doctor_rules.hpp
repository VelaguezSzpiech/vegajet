// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Root-relative doctor rule engine: the same read-only checks as `vegajet doctor`, evaluated against any filesystem
// root, so every rule can be unit-tested on a synthetic /proc + /sys tree. It only stats and reads small text files
// and readlink()s /proc/<pid>/fd entries; it never opens /dev/kfd, /dev/dri, pp_table or any device node, and never
// talks to systemd (a unit counts as active when its cgroup directory exists).
//
// Check names follow core/doctor.cpp (kernel.cmdline.*, gpu.discovery, gpu.<dev>.ras_ecc / pcie_host_link / tmem /
// psu_power_cap / bar0 / p2p_links, gpu.vbios_parity) and add kernel.osrelease, kfd.topology, kfd.device_node,
// kfd.access (stat-only file mode / identity verdict, nothing is opened), gpu.<dev>.power_draw, gpu.<dev>.uclk0,
// gsf12.kfd_holders and gsf12.service.<unit>.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <sys/types.h>
#include <vector>

namespace vegajet::doctor_rules {

enum class Status : uint8_t { Pass, Warn, Fail };
const char* status_name(Status s);

struct Check {
    std::string name;
    Status status = Status::Pass;
    std::string detail;
};

enum class Mode : uint8_t {
    Normal,           // pre-run: guard + fan daemon expected running, no foreign /dev/kfd holder
    Gsf12Preflight,   // before a pp_table / upp write: zero GPU jobs, guard + fan daemon + coolercontrold stopped
};

struct Limits {
    double cap_w_small = 200.0, cap_w_large = 150.0;   // power1_cap ceiling for N <= 2 / N >= 3 GPUs
    double avg_w_small = 205.0, avg_w_large = 155.0;   // power1_input mean ceiling (rolling-average gate)
    double peak_w = 270.0;                             // instantaneous ceiling
    double tmem_warn_c = 65.0, tmem_fail_c = 70.0;     // HBM2 temp3_input bands
    double uclk0_mhz = 350.0;                          // G-SF-12 invariant FreqTableUclk/0
    double min_link_gts = 7.9;                         // host link must be Gen3-capable
};

// Who the /dev/kfd permission verdict is for. Tests inject one; a real run leaves `identity_explicit` false and the
// check uses the current effective uid / supplementary groups (stat() only, the node itself is never opened).
struct AccessIdentity {
    uid_t uid = 0;
    std::vector<gid_t> groups;
    std::function<std::string(gid_t)> group_name;   // gid -> name for the message; empty: "gid <N>"
};

AccessIdentity current_access_identity();
// Classic owner / group / other read+write bits; uid 0 passes as the kernel allows. Pure, mode-only.
bool device_rw_allowed(unsigned mode, uid_t owner, gid_t group, const AccessIdentity& who);

struct Options {
    std::string root = "/";
    Mode mode = Mode::Normal;
    Limits limits;
    uint32_t power_samples = 1;          // power1_input reads per GPU (rolling mean over the samples)
    uint32_t sample_interval_ms = 100;
    AccessIdentity identity;             // used only when identity_explicit is true
    bool identity_explicit = false;
    std::vector<std::string> guard_units = {"mi50-guard.service"};
    std::vector<std::string> fan_units = {"vifted.service"};
    std::vector<std::string> sensor_reader_units = {"coolercontrold.service"};
    // Overridable for tests; defaults: cgroup directory under root / std::this_thread::sleep_for.
    std::function<bool(const std::string& unit)> service_active;
    std::function<void(uint32_t ms, uint32_t sample_index)> sleep_fn;
};

struct RasCount {
    std::string block;
    uint64_t ue = 0, ce = 0, de = 0;
    bool available = false;
};

struct Hop {
    std::string name;
    double cur_gts = 0, max_gts = 0;
    uint32_t cur_width = 0, max_width = 0;
};

struct GpuFacts {
    std::string dev;                 // directory name under sys/bus/pci/devices
    uint32_t device_id = 0;
    std::string vbios;
    uint64_t bar0_bytes = 0, bar0_end = 0;
    double power_cap_w = -1, tmem_c = -1, uclk0_mhz = -1;
    std::vector<double> power_samples_w;
    std::vector<RasCount> ras;
    bool link_found = false;
    Hop root_port, upstream;
};

struct Report {
    std::vector<Check> checks;
    std::vector<GpuFacts> gpus;
    int pass = 0, warn = 0, fail = 0;
    bool ok() const { return fail == 0; }
    const Check* find(const std::string& name) const;
};

Report run(const Options& opt);
std::string format_text(const Report& r);
std::string to_json(const Report& r);

// Pure helpers.
bool parse_osrelease(const std::string& s, int& major, int& minor);   // "7.0.0-43-generic" -> 7, 0
double parse_link_speed_gts(const std::string& s);                    // "8.0 GT/s PCIe" -> 8.0, 0 on error
std::vector<std::string> split_cmdline(const std::string& s);

} // namespace vegajet::doctor_rules
