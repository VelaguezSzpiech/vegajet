// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/portability.hpp"

#include <algorithm>
#include <cstdio>
#include <dirent.h>
#include <grp.h>
#include <sys/stat.h>
#include <unistd.h>

namespace vegajet::portability {

using doctor::CheckStatus;

namespace {

std::string gib(uint64_t b) {
    char s[32];
    std::snprintf(s, sizeof(s), "%.1f GiB", static_cast<double>(b) / static_cast<double>(kGiB));
    return s;
}

}  // namespace

Verdict bar0(uint64_t bar0_bytes, bool under_44bit) {
    if (bar0_bytes >= kLargeBar && under_44bit) return {CheckStatus::Pass, "BAR0 exposes 32 GiB below the 44-bit ceiling"};
    std::string why = bar0_bytes >= kLargeBar ? "BAR0 is mapped above the 44-bit ceiling"
                                              : "BAR0 is " + gib(bar0_bytes) + " (not the 32 GiB large BAR)";
    return {CheckStatus::Warn, why + ": BAR0 peer-to-peer for TP>1 is unavailable; TP=1 and N=1 swarm workers are "
                                     "unaffected. A large-BAR VBIOS and Above-4G decoding enable it."};
}

Verdict p2p_links(size_t n_gpus, bool kfd_matched, uint32_t links) {
    if (n_gpus < 2) return {CheckStatus::Pass, "single GPU: P2P links not needed"};
    const uint32_t need = static_cast<uint32_t>(n_gpus - 1);
    if (kfd_matched && links >= need)
        return {CheckStatus::Pass, "p2p_links_count=" + std::to_string(links) + " (need >=" + std::to_string(need) + ")"};
    return {CheckStatus::Warn, "p2p_links_count=" + std::to_string(links) + " (need >=" + std::to_string(need) +
                                   " for TP across all GPUs): run one N=1 worker per GPU (`vegajet serve --gpu <i>`), "
                                   "or check large BAR / P2P-capable VBIOS"};
}

Verdict power_cap(double cap_w, size_t n_gpus) {
    const double ref = n_gpus >= 3 ? 150.0 : 200.0;
    char d[200];
    std::snprintf(d, sizeof(d), "power1_cap=%.1f W (reference rig budget <= %.0f W per GPU at N=%zu)", cap_w, ref, n_gpus);
    if (cap_w <= ref + 0.5) return {CheckStatus::Pass, d};
    return {CheckStatus::Warn, std::string(d) + ": above the reference budget; make sure your PSU covers N x cap plus "
                                                "the host, or lower it (e.g. hardware/set_power_cap.sh)"};
}

Verdict gpu_count(size_t n_gpus) {
    if (n_gpus >= 2) return {CheckStatus::Pass, std::to_string(n_gpus) + " GPUs: TP and N=1 swarm layouts available"};
    return {CheckStatus::Warn, "1 GPU: TP=1 models and N=1 serve workers only; the Ornith-122B runner needs 2 GPUs"};
}

Identity current_identity() {
    Identity id;
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

namespace {

// read+write for `who` by the classic owner / group / other bits (root always passes, as the kernel does).
bool rw_ok(const struct stat& st, const Identity& who) {
    if (who.uid == 0) return true;
    mode_t r, w;
    if (st.st_uid == who.uid) { r = S_IRUSR; w = S_IWUSR; }
    else if (std::find(who.groups.begin(), who.groups.end(), st.st_gid) != who.groups.end()) { r = S_IRGRP; w = S_IWGRP; }
    else { r = S_IROTH; w = S_IWOTH; }
    return (st.st_mode & r) && (st.st_mode & w);
}

std::string gname(const Identity& who, gid_t g) { return who.group_name ? who.group_name(g) : "gid " + std::to_string(g); }

Verdict node_verdict(const std::string& shown, const struct stat& st, const Identity& who) {
    char mode[8];
    std::snprintf(mode, sizeof(mode), "%04o", static_cast<unsigned>(st.st_mode & 07777));
    const std::string grp = gname(who, st.st_gid);
    if (rw_ok(st, who)) return {CheckStatus::Pass, shown + " is accessible (group " + grp + ", mode " + mode + ")"};
    return {CheckStatus::Fail, shown + " is not readable+writable by this user (group " + grp + ", mode " + mode +
                                   "): run `sudo usermod -aG " + grp + " $USER`, then log out and back in"};
}

}  // namespace

std::vector<AccessCheck> device_access(const std::string& root, const Identity& who) {
    std::vector<AccessCheck> out;
    const std::string base = root.empty() || root == "/" ? std::string() : (root.back() == '/' ? root.substr(0, root.size() - 1) : root);
    struct stat st {};
    if (::stat((base + "/dev/kfd").c_str(), &st) != 0) {
        out.push_back({"kfd.access", {CheckStatus::Fail, "/dev/kfd is missing: load amdgpu with KFD (ROCm-capable kernel, "
                                                         "`modprobe amdgpu`) and check `dmesg | grep kfd`"}});
    } else {
        out.push_back({"kfd.access", node_verdict("/dev/kfd", st, who)});
    }
    std::vector<std::string> nodes;
    if (DIR* d = ::opendir((base + "/dev/dri").c_str())) {
        while (const dirent* e = ::readdir(d)) {
            const std::string n = e->d_name;
            if (n.rfind("renderD", 0) == 0) nodes.push_back(n);
        }
        ::closedir(d);
    }
    std::sort(nodes.begin(), nodes.end());
    if (nodes.empty()) {
        out.push_back({"dri.render_access", {CheckStatus::Fail, "no /dev/dri/renderD* node: the amdgpu DRM driver is not bound"}});
        return out;
    }
    Verdict worst{CheckStatus::Pass, ""};
    size_t ok = 0;
    for (const auto& n : nodes) {
        if (::stat((base + "/dev/dri/" + n).c_str(), &st) != 0) continue;
        Verdict v = node_verdict("/dev/dri/" + n, st, who);
        if (v.status == CheckStatus::Pass) { ++ok; if (worst.detail.empty()) worst.detail = v.detail; }
        else if (worst.status == CheckStatus::Pass) worst = v;
    }
    if (worst.status == CheckStatus::Pass)
        worst.detail = std::to_string(ok) + " of " + std::to_string(nodes.size()) + " render node(s) accessible; " + worst.detail;
    out.push_back({"dri.render_access", worst});
    return out;
}

FitResult check_fit(const FitRequest& req, const std::vector<gpusel::Device>& selected) {
    FitResult r;
    const uint32_t need = std::max<uint32_t>(1u, req.gpus_required);
    std::string found = std::to_string(selected.size()) + " gfx906 GPU(s)";
    if (!selected.empty()) {
        found += " (";
        for (size_t i = 0; i < selected.size(); ++i) found += (i ? ", " : "") + gib(selected[i].vram_bytes);
        found += ")";
    }
    if (selected.size() < need) {
        r.ok = false;
        r.message = req.model + " needs " + std::to_string(need) + " GPUs; this process has " + found +
                    (selected.empty() ? ". Check `vegajet doctor` (driver, /dev/kfd access) and `vegajet plan --gpu list`."
                                      : ". Select more GPUs (--gpu / VJ_GPU_DEVICES), or use a model that fits on fewer.");
        return r;
    }
    for (uint32_t g = 0; g < need; ++g) {
        const uint64_t planned = g < req.bytes_per_gpu.size() ? req.bytes_per_gpu[g] : 0;
        const uint64_t want = std::max(planned, req.arena_bytes_per_gpu);
        if (want > selected[g].vram_bytes) {
            r.ok = false;
            r.message = req.model + " needs " + gib(want) + " per GPU (planned weights + KV + state " + gib(planned) +
                        (req.arena_bytes_per_gpu ? ", runner arena " + gib(req.arena_bytes_per_gpu) : std::string()) +
                        ") on " + std::to_string(need) + " GPUs; GPU " + std::to_string(selected[g].index) + " has " +
                        gib(selected[g].vram_bytes) + ". Use GPUs with more VRAM, or a smaller / more quantized model.";
            return r;
        }
    }
    r.message = "fits: " + std::to_string(need) + " GPU(s), " + found;
    return r;
}

FitResult preflight(const FitRequest& req, const std::string& root) {
    auto devs = gpusel::scan_topology(root);
    std::string err;
    if (!gpusel::filter_from_env(devs, &err)) return {false, err};
    return check_fit(req, devs);
}

}  // namespace vegajet::portability
