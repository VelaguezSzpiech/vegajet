// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Alpha host preflight: the checks a first-time user needs before anything opens /dev/kfd.
//
//  * VRAM budget: every GPU runner reserves a fixed per-GPU arena (Ornith-122B: 30.0-30.5 GiB on each of 2 GPUs).
//    On a 16 GB MI50, a Radeon VII or a single card that used to fail late (after the model was planned, inside the
//    512 MiB BO allocation loop, with an errno and no word about the requirement). check_arena() turns that into
//    one sentence before the runtime opens a device: what is needed, what was found, what does run on this host.
//  * Host compatibility for `vegajet doctor`: kernel release, amdgpu / KFD driver presence, and the ROCm toolchain
//    that builds the gated .hsaco kernels (only needed for GPU builds; the runtime itself does not use ROCm).
//  * A hint appended to open(/dev/kfd) / open(renderD*) errors (EACCES: render/video groups; ENOENT: no driver).
//
// Everything is pure or root-parameterized (tests use a synthetic tree); nothing here opens a device node.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::preflight {

constexpr uint64_t kGiB = 1ULL << 30;
// VRAM kept free beside a runner's arena on every GPU (queues, CWSR save area, kernargs, code objects, firmware).
// A 32 GB MI50 reports 31.98 GiB, so the 30.5 GiB Ornith arena leaves 1.48 GiB: the reserve must stay below that.
constexpr double kArenaReserveGiB = 1.0;
constexpr double kOrnithArenaGiB  = 30.0;  // the runner tries 30.5 first and falls back to 30.0 GiB
constexpr size_t kOrnithGpus      = 2;

struct ArenaCheck {
    bool ok = true;
    std::string message;  // empty when ok; otherwise one actionable paragraph
};
// vram_bytes: one entry per GPU the runner would use (0 = size unknown: not counted against the budget).
// who: the runner, for the message ("Ornith-122B GPU runner", "vegajet soak", "KFD runtime").
ArenaCheck check_arena(const std::vector<uint64_t>& vram_bytes, double arena_gib, size_t min_gpus, const std::string& who);
// Largest arena (in 0.5 GiB steps) that fits beside the reserve on a GPU with `vram_bytes`; 0 if none / unknown.
double max_arena_gib(uint64_t vram_bytes);
// Arena `vegajet soak` reserves for a run over `gib` GiB of chunks: gib + 2 GiB of side allocations, at most the
// historical 30.5 GiB (the default --gib 28.5 keeps exactly the old 30.5 GiB arena).
double soak_arena_gib(double gib);

// VRAM of every gfx906 node in <root>/sys/class/kfd/kfd/topology/nodes (mem_banks/0 size_in_bytes), in node order.
std::vector<uint64_t> gfx906_vram_bytes(const std::string& root);

enum class Level { Pass, Warn, Fail };
struct HostCheck {
    std::string name;
    Level level = Level::Pass;
    std::string detail;
};

struct KernelRelease {
    bool ok = false;
    int major = 0, minor = 0;
};
KernelRelease parse_kernel_release(const std::string& osrelease);  // "7.0.0-43-generic" -> 7.0
// Linux 7.0 is the tested series: PASS. Any other parseable release: WARN (untested); below 4.19 the KFD GPUVM
// ioctls the runtime uses (ALLOC/MAP_MEMORY_OF_GPU) do not exist, which the WARN says.
HostCheck kernel_check(const std::string& osrelease);
// <root>/sys/module/amdgpu and <root>/sys/class/kfd/kfd/topology: driver loaded and KFD exposed.
HostCheck driver_check(const std::string& root);

struct RocmInfo {
    std::string prefix;     // the directory that was examined
    bool present = false;   // prefix exists
    std::string version;    // first line of <prefix>/.info/version, "" if absent
    int major = 0;
    std::vector<std::string> missing_tools;  // of hipcc, clang-offload-bundler, llvm-objdump
};
// prefix_override: the build's -DVEGAJET_ROCM_PATH when known, else <root>/opt/rocm.
RocmInfo probe_rocm(const std::string& root, const char* prefix_override);
// Only GPU builds need ROCm. Missing: WARN (CPU tools need nothing). 7.x with all tools: PASS. 6.x: WARN, the
// .hsaco VGPR gate (ops/kernels/hsaco_gates.txt) is register-count exact and has failed with ROCm 6.4.
HostCheck rocm_check(const RocmInfo& info);
// Supported-configuration summary: can this host run the Ornith-122B GPU path, and if not, what it can run.
HostCheck supported_config_check(const std::vector<uint64_t>& vram_bytes);

// Every host check above, in the order `vegajet doctor` prints them. root is "/" outside tests.
std::vector<HostCheck> host_checks(const std::string& root, const char* rocm_prefix_override);

// Suffix for an open() error on /dev/kfd or /dev/dri/renderD*: "" unless errno is EACCES/EPERM (groups) or
// ENOENT (driver).
std::string device_open_hint(int err_no);

}  // namespace vegajet::preflight
