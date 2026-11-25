// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vegajet {
namespace doctor {

enum class CheckStatus {
    Pass,
    Warn,
    Fail,
};

struct RasBlockCount {
    std::string block;
    uint64_t ue = 0;
    uint64_t ce = 0;
    uint64_t de = 0;
    bool available = false;
};

// One PCIe hop (sysfs current_/max_link_speed, current_/max_link_width).
struct PcieHop {
    std::string bdf;
    double   cur_gts = 0.0;
    double   max_gts = 0.0;
    uint32_t cur_width = 0;
    uint32_t max_width = 0;
};

// One line of amdgpu pp_dpm_pcie, e.g. "1: 8.0GT/s, x16 773Mhz *".
struct PcieDpmLevel {
    uint32_t index = 0;
    double   gts = 0.0;
    uint32_t width = 0;
    bool     active = false;
};

// The real host link of a GPU: root port <-> first device below it (the on-card switch upstream port on
// Vega 20; the endpoint itself only reports its internal 16 GT/s link).
struct HostLink {
    bool found = false;
    PcieHop root_port;
    PcieHop upstream;
    std::vector<PcieDpmLevel> dpm_levels;
    int    dpm_active = -1;      // active pp_dpm_pcie index, -1 if unreadable
    double cap_gts = 0.0;        // min(max speed) of both link ends
    uint32_t cap_width = 0;      // min(max width) of both link ends
};

struct GpuHealth {
    std::string bdf;
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    std::string vbios_version;
    uint64_t bar0_start = 0;
    uint64_t bar0_end = 0;
    uint64_t bar0_bytes = 0;
    bool bar0_32gib = false;
    bool bar0_under_44bit = false;
    uint32_t kfd_node_id = 0;
    uint32_t p2p_links_count = 0;
    bool kfd_matched = false;
    double power_cap_w = -1.0;
    double tmem_c = -1.0;
    std::vector<RasBlockCount> ras;
    uint64_t total_ue = 0;
    uint64_t total_de = 0;
    uint64_t total_ce = 0;
    HostLink link;
};

struct CheckResult {
    std::string name;
    CheckStatus status = CheckStatus::Pass;
    std::string detail;
};

struct DoctorReport {
    std::string cmdline;
    std::vector<GpuHealth> gpus;
    std::vector<CheckResult> checks;
    int pass_count = 0;
    int warn_count = 0;
    int fail_count = 0;
};

// Strictly read-only sysfs/procfs hardware & kernel pre-flight check.
// NEVER opens /dev/kfd, /dev/dri/*, pp_table, upp, or amdmemtweak.
DoctorReport run_doctor();

void print_report_text(const DoctorReport& rep);
void print_report_json(const DoctorReport& rep);

// ---- Live PCIe host-link check (`vegajet doctor --live-link`) -------------------------------------
// Pure helpers (unit-testable, no I/O).
double parse_link_speed_gts(const std::string& s);                       // "8.0 GT/s PCIe" -> 8.0, 0 on error
bool   parse_pp_dpm_pcie_line(const std::string& line, PcieDpmLevel& out);

struct LinkSample {
    int    busy_pct = 0;     // gpu_busy_percent
    double gts = 0.0;        // host-link current speed at the root port
    int    dpm_level = -1;   // active pp_dpm_pcie index
};

enum class LinkVerdict {
    NoLoadSeen,           // no sample with busy >= 50 %: inconclusive
    UpshiftsUnderLoad,    // >= 90 % of busy samples at >= 8 GT/s
    Partial,              // some but < 90 % of busy samples at >= 8 GT/s
    StuckGen1,            // busy, never >= 8 GT/s: the link is held below Gen3 (firmware/slot side)
};

struct LinkClassification {
    LinkVerdict verdict = LinkVerdict::NoLoadSeen;
    uint32_t samples = 0, busy_samples = 0, busy_fast_samples = 0;
    bool dpm_level_changed = false;
};

LinkClassification classify_live_link(const std::vector<LinkSample>& samples);
const char* link_verdict_name(LinkVerdict v);
const char* link_verdict_action(LinkVerdict v);

struct LiveLinkGpu {
    std::string bdf;
    HostLink link;
    LinkClassification cls;
    double min_gts = 0.0, max_gts = 0.0;
};

struct LiveLinkReport {
    double seconds = 0.0;
    std::vector<LiveLinkGpu> gpus;
    bool any_fail = false;
    bool any_warn = false;
};

// Samples every 100 ms for `seconds`; read-only sysfs, run it while a GPU kernel is executing.
LiveLinkReport run_live_link(double seconds);
void print_live_link_text(const LiveLinkReport& rep);
void print_live_link_json(const LiveLinkReport& rep);

// Priority 1: Calibrate hardware roofline (n_cu, sclk_ghz, mclk_mhz, bw_hbm_gbs, I_star)
// from sysfs and cache into ~/.config/vegajet/hw_profile.json (read-only sysfs, zero /dev/kfd).
bool calibrate_and_save_hw_profile(const std::string& out_path = "");

} // namespace doctor
} // namespace vegajet
