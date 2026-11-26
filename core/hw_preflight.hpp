// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include <string>
#include <vector>

namespace vegajet::hw_preflight {
struct Warning { std::string bdf; std::string check; std::string detail; };
// sysfs_root is injectable for offline tests; defaults to /sys.
std::vector<Warning> run(const std::vector<std::string>& bdfs, const std::string& sysfs_root = "/sys");
void run_and_log(const std::vector<std::string>& bdfs);
// Host dispatch placement (VEGAJET_DECODE_FABRIC=1). CPU list for the dispatch thread: `override_list`
// (e.g. VEGAJET_DISPATCH_CPUS="0-7,16-23") when non-empty, otherwise the CPUs of the NUMA node shared by all
// `bdfs` (sysfs numa_node). Empty when there is no single, known node (UMA / one node / mixed nodes):
// callers must then leave affinity unchanged. `why` explains the decision.
std::vector<int> dispatch_cpus(const std::vector<std::string>& bdfs, const std::string& override_list,
                               std::string* why, const std::string& sysfs_root = "/sys");
// Parse a Linux cpulist ("0-3,8,10-11"); empty on malformed input.
std::vector<int> parse_cpulist(const std::string& s);
// Clock values (MHz) of the lines marked active ('*') in pp_dpm_sclk / pp_dpm_mclk text ("1: 1725Mhz *").
std::vector<double> parse_active_dpm_mhz(const std::string& text);
// Applies dispatch_cpus() to the calling thread (sched_setaffinity, unprivileged). Returns the pinned set.
std::vector<int> pin_dispatch_thread(const std::vector<std::string>& bdfs);
}
