// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/host_preflight.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace vegajet::preflight {

namespace fs = std::filesystem;

namespace {

std::string gib_str(double gib) {
    char b[32];
    std::snprintf(b, sizeof(b), "%.1f", gib);
    return b;
}

double to_gib(uint64_t bytes) { return static_cast<double>(bytes) / static_cast<double>(kGiB); }

std::string join_path(const std::string& root, const std::string& abs) {
    if (root.empty() || root == "/") return abs;
    return (root.back() == '/' ? root.substr(0, root.size() - 1) : root) + abs;
}

bool is_dir(const std::string& p) {
    std::error_code ec;
    return fs::is_directory(p, ec);
}

bool is_file(const std::string& p) {
    std::error_code ec;
    return fs::is_regular_file(p, ec) || fs::is_symlink(p, ec);
}

std::string first_line(const std::string& path) {
    std::ifstream f(path);
    std::string s;
    if (!f.is_open() || !std::getline(f, s)) return "";
    while (!s.empty() && (s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) s.pop_back();
    return s;
}

// What a host that fails the Ornith budget can still do, sized by its smallest known GPU.
std::string what_runs_here(const std::vector<uint64_t>& vram_bytes) {
    uint64_t smallest = 0;
    for (uint64_t v : vram_bytes)
        if (v > 0 && (smallest == 0 || v < smallest)) smallest = v;
    std::string s = "This alpha runs the Ornith-122B GPU path only on 2x 32 GB gfx906 (MI50 32 GB / MI60). Other "
                    "gfx906 hosts (one card, 16 GB MI50, Radeon VII) can use the CPU tools and `vegajet doctor`";
    const double soak_max = max_arena_gib(smallest) - 2.0;
    if (smallest > 0 && soak_max >= 1.0) s += ", and `vegajet soak --gib " + gib_str(soak_max) + "` or less";
    return s + ".";
}

}  // namespace

double max_arena_gib(uint64_t vram_bytes) {
    if (vram_bytes == 0) return 0.0;
    const double avail = to_gib(vram_bytes) - kArenaReserveGiB;
    if (avail <= 0.0) return 0.0;
    return std::floor(avail * 2.0) / 2.0;
}

double soak_arena_gib(double gib) { return std::min(30.5, gib + 2.0); }

ArenaCheck check_arena(const std::vector<uint64_t>& vram_bytes, double arena_gib, size_t min_gpus, const std::string& who) {
    ArenaCheck r;
    const double need = arena_gib + kArenaReserveGiB;
    std::string found;
    for (size_t i = 0; i < vram_bytes.size(); ++i) {
        found += (i ? ", " : "") + std::string("GPU ") + std::to_string(i) + ": " +
                 (vram_bytes[i] ? gib_str(to_gib(vram_bytes[i])) + " GiB" : std::string("VRAM size unknown"));
    }
    const std::string need_s = who + " needs " + std::to_string(min_gpus) + " gfx906 GPU" + (min_gpus == 1 ? "" : "s") +
                               " with at least " + gib_str(need) + " GiB VRAM" + (min_gpus == 1 ? "" : " each") + " (" +
                               gib_str(arena_gib) + " GiB arena + " + gib_str(kArenaReserveGiB) + " GiB reserve)";
    if (vram_bytes.size() < min_gpus) {
        r.ok = false;
        r.message = "VRAM budget: " + need_s + "; found " + std::to_string(vram_bytes.size()) + " GPU" +
                    (vram_bytes.size() == 1 ? "" : "s") + (found.empty() ? "" : " (" + found + ")") + ". " +
                    what_runs_here(vram_bytes);
        return r;
    }
    std::string small;
    for (size_t i = 0; i < vram_bytes.size(); ++i) {
        if (vram_bytes[i] == 0 || to_gib(vram_bytes[i]) + 1e-9 >= need) continue;
        small += (small.empty() ? "" : ", ") + std::string("GPU ") + std::to_string(i) + " has " +
                 gib_str(to_gib(vram_bytes[i])) + " GiB";
    }
    if (!small.empty()) {
        r.ok = false;
        r.message = "VRAM budget: " + need_s + "; " + small + ". " + what_runs_here(vram_bytes);
    }
    return r;
}

std::vector<uint64_t> gfx906_vram_bytes(const std::string& root) {
    std::vector<uint64_t> out;
    const std::string base = join_path(root, "/sys/class/kfd/kfd/topology/nodes");
    std::error_code ec;
    if (!fs::is_directory(base, ec)) return out;
    std::vector<unsigned long> ids;
    for (const auto& e : fs::directory_iterator(base, ec)) {
        const std::string n = e.path().filename().string();
        if (n.empty() || n.find_first_not_of("0123456789") != std::string::npos || n.size() > 6) continue;
        ids.push_back(std::stoul(n));
    }
    std::sort(ids.begin(), ids.end());
    for (unsigned long id : ids) {
        const std::string nd = base + "/" + std::to_string(id);
        std::ifstream p(nd + "/properties");
        std::string k;
        uint64_t v = 0, gfx = 0;
        while (p >> k >> v)
            if (k == "gfx_target_version") gfx = v;
        if (gfx != 90006u) continue;
        uint64_t size = 0;
        std::ifstream m(nd + "/mem_banks/0/properties");
        while (m >> k >> v)
            if (k == "size_in_bytes") { size = v; break; }
        out.push_back(size);
    }
    return out;
}

KernelRelease parse_kernel_release(const std::string& osrelease) {
    KernelRelease r;
    int maj = 0, min = 0;
    char dot = 0;
    std::istringstream is(osrelease);
    if (is >> maj >> dot >> min && dot == '.' && maj > 0 && min >= 0) {
        r.ok = true;
        r.major = maj;
        r.minor = min;
    }
    return r;
}

HostCheck kernel_check(const std::string& osrelease) {
    HostCheck c{"host.kernel", Level::Pass, ""};
    const KernelRelease k = parse_kernel_release(osrelease);
    if (!k.ok) {
        c.level = Level::Warn;
        c.detail = "Kernel release unknown ('" + osrelease + "'); VegaJet is tested on Linux 7.0.";
        return c;
    }
    const std::string rel = "Linux " + std::to_string(k.major) + "." + std::to_string(k.minor) + " (" + osrelease + ")";
    if (k.major == 7 && k.minor == 0) {
        c.detail = rel + ": the tested kernel series.";
    } else if (k.major < 4 || (k.major == 4 && k.minor < 19)) {
        c.level = Level::Warn;
        c.detail = rel + ": older than 4.19, the KFD GPU-memory ioctls the runtime uses are missing; tested on Linux 7.0.";
    } else {
        c.level = Level::Warn;
        c.detail = rel + ": untested; VegaJet is tested on Linux 7.0 (the KFD ioctls it uses exist since 4.19).";
    }
    return c;
}

HostCheck driver_check(const std::string& root) {
    HostCheck c{"host.amdgpu_kfd", Level::Pass, ""};
    const bool amdgpu = is_dir(join_path(root, "/sys/module/amdgpu"));
    const bool kfd = is_dir(join_path(root, "/sys/class/kfd/kfd/topology"));
    if (!amdgpu) {
        c.level = Level::Warn;
        c.detail = "amdgpu driver not loaded (no /sys/module/amdgpu): GPU runs need it, CPU tools do not.";
    } else if (!kfd) {
        c.level = Level::Warn;
        c.detail = "amdgpu is loaded but KFD is not exposed (no /sys/class/kfd/kfd/topology): GPU runs need a kernel "
                   "built with CONFIG_HSA_AMD.";
    } else {
        c.detail = "amdgpu loaded, KFD topology present.";
    }
    return c;
}

RocmInfo probe_rocm(const std::string& root, const char* prefix_override) {
    RocmInfo r;
    r.prefix = (prefix_override && prefix_override[0]) ? std::string(prefix_override) : join_path(root, "/opt/rocm");
    r.present = is_dir(r.prefix);
    if (!r.present) return r;
    r.version = first_line(r.prefix + "/.info/version");
    r.major = std::atoi(r.version.c_str());
    // Same search order as the kernel build's find_program() hints, plus the lib/llvm layout of newer packages.
    const std::vector<std::string> llvm_dirs = {"/llvm/bin", "/lib/llvm/bin", "/bin"};
    auto has = [&](const std::string& tool, const std::vector<std::string>& dirs) {
        for (const auto& d : dirs)
            if (is_file(r.prefix + d + "/" + tool)) return true;
        return false;
    };
    if (!has("hipcc", {"/bin"})) r.missing_tools.push_back("hipcc");
    if (!has("clang-offload-bundler", llvm_dirs)) r.missing_tools.push_back("clang-offload-bundler");
    if (!has("llvm-objdump", llvm_dirs)) r.missing_tools.push_back("llvm-objdump");
    return r;
}

HostCheck rocm_check(const RocmInfo& info) {
    HostCheck c{"host.rocm_toolchain", Level::Warn, ""};
    const std::string why = " Only GPU builds (-DVEGAJET_CPU_ONLY=OFF) need it; the runtime talks to /dev/kfd directly.";
    if (!info.present) {
        c.detail = "ROCm not found at " + info.prefix + " (configure with -DVEGAJET_ROCM_PATH=<prefix> if it is "
                   "elsewhere)." + why;
        return c;
    }
    const std::string ver = info.version.empty() ? std::string("version unknown") : "ROCm " + info.version;
    if (!info.missing_tools.empty()) {
        std::string m;
        for (const auto& t : info.missing_tools) m += (m.empty() ? "" : ", ") + t;
        c.detail = ver + " at " + info.prefix + ": missing " + m + "." + why;
        return c;
    }
    if (info.major >= 7) {
        c.level = Level::Pass;
        c.detail = ver + " at " + info.prefix + ": hipcc, clang-offload-bundler and llvm-objdump found.";
    } else if (info.major > 0) {
        c.detail = ver + " at " + info.prefix + ": the kernel build's VGPR gate (ops/kernels/hsaco_gates.txt) is exact "
                   "for the 7.x LLVM and has failed with ROCm 6.4; expect gate failures, or use prebuilt kernels.";
    } else {
        c.detail = ver + " at " + info.prefix + ": tools found, version not readable; the kernel build is gated "
                   "for the ROCm 7.x LLVM.";
    }
    return c;
}

HostCheck supported_config_check(const std::vector<uint64_t>& vram_bytes) {
    HostCheck c{"host.supported_config", Level::Pass, ""};
    const ArenaCheck a = check_arena(vram_bytes, kOrnithArenaGiB, kOrnithGpus, "the Ornith-122B GPU path");
    if (a.ok) {
        std::string list;
        for (uint64_t v : vram_bytes) list += (list.empty() ? "" : " + ") + gib_str(to_gib(v)) + " GiB";
        c.detail = std::to_string(vram_bytes.size()) + " gfx906 GPU(s) (" + list + "): the Ornith-122B GPU path fits.";
    } else {
        c.level = Level::Warn;
        c.detail = a.message;
    }
    return c;
}

std::vector<HostCheck> host_checks(const std::string& root, const char* rocm_prefix_override) {
    std::vector<HostCheck> out;
    out.push_back(kernel_check(first_line(join_path(root, "/proc/sys/kernel/osrelease"))));
    out.push_back(driver_check(root));
    out.push_back(rocm_check(probe_rocm(root, rocm_prefix_override)));
    out.push_back(supported_config_check(gfx906_vram_bytes(root)));
    return out;
}

std::string device_open_hint(int err_no) {
    if (err_no == EACCES || err_no == EPERM)
        return " (add your user to the 'render' and 'video' groups: sudo usermod -aG render,video \"$USER\", then log "
               "in again)";
    if (err_no == ENOENT)
        return " (device node missing: is the amdgpu driver loaded with KFD? see `vegajet doctor`)";
    return "";
}

}  // namespace vegajet::preflight
