// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Alpha portability: verdicts that keep `vegajet doctor` informative on rigs other than the reference 2x MI50 / 32 GiB
// BAR0 / capped-PSU box, a device-access probe that says exactly why /dev/kfd or a render node is unusable, and a
// fit check that refuses a model before any VRAM arena is allocated when the selected GPUs cannot hold it.
//
// Everything here is pure or root-parameterized: the probe only stat()s <root>/dev/kfd and <root>/dev/dri/renderD*
// (it never opens a device node), and the fit check works on a device list (core/gpu_select scan_topology).
#pragma once

#include "core/doctor.hpp"
#include "core/gpu_select.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <sys/types.h>
#include <vector>

namespace vegajet::portability {

struct Verdict {
    doctor::CheckStatus status = doctor::CheckStatus::Pass;
    std::string detail;
};

constexpr uint64_t kGiB = 1ULL << 30;
constexpr uint64_t kLargeBar = 32ULL * kGiB;

// BAR0 that does not expose all 32 GiB below the 44-bit ceiling: WARN (TP>1 BAR0 P2P is unavailable, TP=1 and
// N=1 swarm workers are fine). Never FAIL: it is a VBIOS / firmware property of the card, not a broken host.
Verdict bar0(uint64_t bar0_bytes, bool under_44bit);
// Fewer KFD P2P links than N-1 on a multi-GPU host: WARN (run N=1 workers instead of TP). Single GPU: PASS, n/a.
Verdict p2p_links(size_t n_gpus, bool kfd_matched, uint32_t links);
// Power cap above the reference rig's PSU budget (200 W for N<=2, 150 W for N>=3): WARN with the budget to check.
Verdict power_cap(double cap_w, size_t n_gpus);
// GPU count: 1 is a valid alpha setup (TP=1, swarm worker); WARN only to say which runners need more.
Verdict gpu_count(size_t n_gpus);

// Who is asking: effective uid, supplementary groups (incl. the primary group), gid -> name for messages.
struct Identity {
    uid_t uid = 0;
    std::vector<gid_t> groups;
    std::function<std::string(gid_t)> group_name;
};
Identity current_identity();
struct AccessCheck {
    std::string name;  // "kfd.access", "dri.render_access"
    Verdict verdict;
};
// stat()-only probe of <root>/dev/kfd and <root>/dev/dri/renderD*: missing node, or no read+write permission for
// `who` (with the owning group and the usermod line that fixes it), or PASS naming the group that grants access.
std::vector<AccessCheck> device_access(const std::string& root, const Identity& who);

// Fit check before a runner allocates its per-GPU arenas.
struct FitRequest {
    std::string model;                                // file name, for the message
    uint32_t gpus_required = 1;                       // the runner's GPU count (Ornith-122B: 2) or --tp
    std::vector<uint64_t> bytes_per_gpu;              // planned weights + KV + state per GPU
    uint64_t arena_bytes_per_gpu = 0;                 // what the runner reserves per GPU (0: not fixed)
};
struct FitResult {
    bool ok = true;
    std::string message;  // actionable: what is needed vs what was found
};
FitResult check_fit(const FitRequest& req, const std::vector<gpusel::Device>& selected);
// scan_topology(root) + the VJ_GPU_DEVICES selection the CLI exported + check_fit (what a GPU runner sees).
FitResult preflight(const FitRequest& req, const std::string& root);

}  // namespace vegajet::portability
