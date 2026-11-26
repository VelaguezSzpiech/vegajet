// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Per-process GPU selection: `--gpu <spec>` on the command line or VJ_GPU_DEVICES in the environment.
//
// A spec is "all", ordinals/ranges ("0,2-3"), or full PCI BDFs ("dddd:bb:dd.f,dddd:bb:dd.f"). Indices are positions
// in the gfx906 enumeration: KFD topology nodes with gfx_target_version 90006, ordered by PCI domain then location_id. That is
// exactly the order of kfd::Runtime::scan_gfx906_topology(), so index 0 is the same card in `vegajet --gpu list`,
// `vegajet doctor` and the runtime. --gpu wins over VJ_GPU_DEVICES, which wins over "all".
//
// The CLI resolves the selection once and exports it as VJ_GPU_DEVICES (canonical form), and kfd::Runtime::init()
// applies filter_from_env() to its device list, so every GPU path in the process (bench, run, soak, serve, doctor's
// live-link probe, the Ornith engine) sees only the selected cards. A worker started with `--gpu 1` therefore owns
// exactly one MI50, which is what tools/swarm_router.py expects of an N=1 worker.
// scan_topology() only reads sysfs text files under `root`; it never opens /dev/kfd or a render node.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace vegajet::gpusel {

constexpr const char* kEnvVar = "VJ_GPU_DEVICES";
constexpr uint32_t kMaxIndex = 63;

struct Selection {
    bool all = true;
    std::vector<uint32_t> indices;  // ascending, unique; empty for BDFs/all
    std::vector<std::string> bdfs;  // full canonical identities, in requested TP-rank order
    std::string source = "default"; // "default", "--gpu" or "VJ_GPU_DEVICES"
};

std::string bdf(uint32_t pci_domain, uint32_t location_id);
bool parse_bdf(const std::string& text, std::string& canonical_bdf);
bool parse(const std::string& spec, Selection& out, std::string* err);
// flag: the --gpu value (nullptr when the flag was not given); env_value: getenv(kEnvVar) (may be nullptr).
bool resolve(const std::string* flag, const char* env_value, Selection& out, std::string* err);
std::string canonical(const Selection& s);  // "all" or "0,2,3"
// Positions picked out of n devices. False (err set) when an index is >= n.
bool pick(const Selection& s, size_t n, std::vector<size_t>& out, std::string* err);

bool pick_bdfs(const Selection& s, const std::vector<std::string>& identities, std::vector<size_t>& out, std::string* err);
template <class T>
bool pick_devices(const Selection& s, const std::vector<T>& devs, std::vector<size_t>& out, std::string* err) {
    if (s.bdfs.empty()) return pick(s, devs.size(), out, err);
    std::vector<std::string> ids;
    for (const auto& d : devs) ids.push_back(bdf(d.pci_domain, d.location_id));
    return pick_bdfs(s, ids, out, err);
}
template <class T>
bool filter(std::vector<T>& devs, const Selection& s, std::string* err) {
    std::vector<size_t> idx;
    if (!pick_devices(s, devs, idx, err)) return false;
    std::vector<T> out;
    out.reserve(idx.size());
    for (size_t i : idx) out.push_back(devs[i]);
    devs.swap(out);
    return true;
}

// Applies VJ_GPU_DEVICES, if set, to a device list (kfd::Runtime::init calls this on its topology scan).
template <class T>
bool filter_from_env(std::vector<T>& devs, std::string* err) {
    const char* e = std::getenv(kEnvVar);
    if (!e) return true;
    Selection s;
    std::string why;
    if (!parse(e, s, &why) || !filter(devs, s, &why)) {
        if (err) *err = std::string(kEnvVar) + "=" + e + ": " + why;
        return false;
    }
    return true;
}

struct Device {
    uint32_t index = 0;  // position in the enumeration
    uint32_t node_id = 0, gpu_id = 0, location_id = 0, drm_render_minor = 0, simd_count = 0;
    uint64_t vram_bytes = 0;
    uint32_t pci_domain = 0;
};

// Read-only scan of <root>/sys/class/kfd/kfd/topology/nodes, same filter and order as the KFD runtime.
std::vector<Device> scan_topology(const std::string& root);
// Root for scan_topology(): VJ_SYSFS_ROOT (tests, containers with a bind-mounted sysfs) or "/".
std::string sysfs_root();
// Table for `--gpu list`; selected devices are marked with '*'.
std::string format_list(const std::vector<Device>& devs, const Selection& s);

}  // namespace vegajet::gpusel
