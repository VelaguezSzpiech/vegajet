// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Alpha host preflight (core/host_preflight): VRAM budget messages for the hosts a first-time user actually has
// (2x 32 GB MI50, 1x 32 GB, 2x 16 GB MI50 / Radeon VII, mixed), soak arena sizing, the KFD topology VRAM scan, and
// the doctor host checks (kernel, amdgpu/KFD, ROCm toolchain, supported configuration) on synthetic root trees.
// No device node is opened and nothing outside a temporary directory is read.
#include "core/host_preflight.hpp"
#include "tests/test_common.hpp"

#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace vegajet::preflight;
namespace fs = std::filesystem;

namespace {

constexpr uint64_t k32GB = 34342961152ULL;  // what KFD reports for a 32 GB MI50 / MI60 (31.98 GiB)
constexpr uint64_t k16GB = 17163091968ULL;  // 16 GB MI50 / Radeon VII (15.98 GiB)

bool has(const std::string& s, const std::string& needle) { return s.find(needle) != std::string::npos; }

void put(const fs::path& p, const std::string& text) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << text;
}

struct TempRoot {
    fs::path dir;
    TempRoot() {
        std::string t = (fs::temp_directory_path() / "vj_preflight_XXXXXX").string();
        VJ_CHECK(mkdtemp(t.data()) != nullptr);
        dir = t;
    }
    ~TempRoot() { std::error_code ec; fs::remove_all(dir, ec); }
    std::string str() const { return dir.string(); }
};

void add_node(const fs::path& root, int id, uint64_t gfx, uint64_t vram) {
    const fs::path n = root / "sys/class/kfd/kfd/topology/nodes" / std::to_string(id);
    put(n / "properties", "cpu_cores_count 0\nsimd_count 240\ngfx_target_version " + std::to_string(gfx) + "\n");
    if (vram) put(n / "mem_banks/0/properties", "heap_type 1\nsize_in_bytes " + std::to_string(vram) + "\nflags 0\n");
}

void test_arena_budget() {
    // Reference rig: both runner arenas (30.5 first, 30.0 fallback) fit.
    VJ_CHECK(check_arena({k32GB, k32GB}, 30.0, 2, "R").ok);
    VJ_CHECK(check_arena({k32GB, k32GB}, 30.5, 2, "R").ok);
    VJ_CHECK(check_arena({k32GB, k32GB}, 30.5, 2, "R").message.empty());

    const auto one = check_arena({k32GB}, kOrnithArenaGiB, kOrnithGpus, "the Ornith-122B GPU runner");
    VJ_CHECK(!one.ok);
    VJ_CHECK(has(one.message, "the Ornith-122B GPU runner needs 2 gfx906 GPUs with at least 31.0 GiB VRAM each"));
    VJ_CHECK(has(one.message, "found 1 GPU (GPU 0: 32.0 GiB)"));
    VJ_CHECK(has(one.message, "2x 32 GB gfx906 (MI50 32 GB / MI60)"));
    VJ_CHECK(has(one.message, "`vegajet soak --gib 28.5` or less"));

    const auto small = check_arena({k16GB, k16GB}, kOrnithArenaGiB, kOrnithGpus, "R");
    VJ_CHECK(!small.ok);
    VJ_CHECK(has(small.message, "GPU 0 has 16.0 GiB, GPU 1 has 16.0 GiB"));
    VJ_CHECK(has(small.message, "`vegajet soak --gib 12.5` or less"));

    const auto mixed = check_arena({k32GB, k16GB}, kOrnithArenaGiB, kOrnithGpus, "R");
    VJ_CHECK(!mixed.ok && has(mixed.message, "; GPU 1 has 16.0 GiB.") && !has(mixed.message, "GPU 0 has"));

    const auto none = check_arena({}, kOrnithArenaGiB, kOrnithGpus, "R");
    VJ_CHECK(!none.ok && has(none.message, "found 0 GPUs.") && !has(none.message, "soak --gib"));

    // Unknown sizes are not held against the budget (the runtime reports its own allocation error then).
    VJ_CHECK(check_arena({0, 0}, kOrnithArenaGiB, kOrnithGpus, "R").ok);

    // The generic runtime guard (min 1 GPU) on a 16 GB card: the old 30.5 GiB soak arena is refused up front,
    // the --gib-sized one fits.
    const auto rt16 = check_arena({k16GB}, 30.5, 1, "the KFD runtime");
    VJ_CHECK(!rt16.ok && has(rt16.message, "the KFD runtime needs 1 gfx906 GPU with at least 31.5 GiB VRAM ("));
    VJ_CHECK(check_arena({k16GB}, soak_arena_gib(12.5), 1, "R").ok);
    VJ_CHECK(!check_arena({k16GB}, soak_arena_gib(13.5), 1, "R").ok);

    VJ_CHECK(max_arena_gib(k32GB) == 30.5 && max_arena_gib(k16GB) == 14.5 && max_arena_gib(0) == 0.0);
    VJ_CHECK(max_arena_gib(kGiB / 2) == 0.0);
    VJ_CHECK(soak_arena_gib(28.5) == 30.5 && soak_arena_gib(30.0) == 30.5 && soak_arena_gib(4.0) == 6.0);
    std::printf("  budget: 2x32 GB ok; 1x32 GB, 2x16 GB, 32+16 GB, none refused with what runs instead\n");
    std::printf("  e.g. %s\n", small.message.c_str());
}

void test_topology_scan() {
    TempRoot r;
    add_node(r.dir, 0, 0, 0);           // CPU node
    add_node(r.dir, 2, 90006, k16GB);
    add_node(r.dir, 1, 90006, k32GB);
    add_node(r.dir, 10, 90008, k32GB);  // another GPU family: not counted
    add_node(r.dir, 11, 90006, 0);      // gfx906 without mem_banks: size unknown
    const auto v = gfx906_vram_bytes(r.str());
    VJ_CHECK(v.size() == 3 && v[0] == k32GB && v[1] == k16GB && v[2] == 0);
    VJ_CHECK(gfx906_vram_bytes(r.str() + "/missing").empty());
}

void test_kernel() {
    VJ_CHECK(parse_kernel_release("7.0.0-43-generic").ok);
    const auto k = parse_kernel_release("6.8.12-arch1-1");
    VJ_CHECK(k.ok && k.major == 6 && k.minor == 8);
    VJ_CHECK(!parse_kernel_release("").ok && !parse_kernel_release("linux").ok);

    const auto tested = kernel_check("7.0.0-43-generic");
    VJ_CHECK(tested.level == Level::Pass && has(tested.detail, "tested kernel series"));
    const auto newer = kernel_check("6.8.0-50-generic");
    VJ_CHECK(newer.level == Level::Warn && has(newer.detail, "untested") && has(newer.detail, "Linux 6.8"));
    const auto old = kernel_check("4.15.0-20-generic");
    VJ_CHECK(old.level == Level::Warn && has(old.detail, "older than 4.19"));
    VJ_CHECK(kernel_check("4.19.0").level == Level::Warn && !has(kernel_check("4.19.0").detail, "older than"));
    VJ_CHECK(kernel_check("?").level == Level::Warn && has(kernel_check("?").detail, "unknown"));
}

void test_driver() {
    TempRoot r;
    auto c = driver_check(r.str());
    VJ_CHECK(c.level == Level::Warn && has(c.detail, "amdgpu driver not loaded"));
    fs::create_directories(r.dir / "sys/module/amdgpu");
    c = driver_check(r.str());
    VJ_CHECK(c.level == Level::Warn && has(c.detail, "KFD is not exposed"));
    fs::create_directories(r.dir / "sys/class/kfd/kfd/topology");
    c = driver_check(r.str());
    VJ_CHECK(c.level == Level::Pass);
}

void tools(const fs::path& prefix, const std::string& llvm_dir) {
    put(prefix / "bin/hipcc", "#!/bin/sh\n");
    put(prefix / llvm_dir / "clang-offload-bundler", "");
    put(prefix / llvm_dir / "llvm-objdump", "");
}

void test_rocm() {
    TempRoot r;
    auto info = probe_rocm(r.str(), nullptr);
    VJ_CHECK(!info.present && info.prefix == r.str() + "/opt/rocm");
    auto c = rocm_check(info);
    VJ_CHECK(c.level == Level::Warn && has(c.detail, "ROCm not found") && has(c.detail, "VEGAJET_ROCM_PATH"));
    VJ_CHECK(has(c.detail, "Only GPU builds"));

    const fs::path r7 = r.dir / "opt/rocm";
    put(r7 / ".info/version", "7.1.0-25\n");
    put(r7 / "bin/hipcc", "");
    info = probe_rocm(r.str(), nullptr);
    VJ_CHECK(info.present && info.major == 7 && info.missing_tools.size() == 2);
    c = rocm_check(info);
    VJ_CHECK(c.level == Level::Warn && has(c.detail, "missing clang-offload-bundler, llvm-objdump"));
    tools(r7, "llvm/bin");
    c = rocm_check(probe_rocm(r.str(), nullptr));
    VJ_CHECK(c.level == Level::Pass && has(c.detail, "ROCm 7.1.0-25"));

    // An override prefix with the lib/llvm layout and a 6.x version.
    const fs::path r6 = r.dir / "elsewhere/rocm-6.4.1";
    put(r6 / ".info/version", "6.4.1-120\n");
    tools(r6, "lib/llvm/bin");
    const std::string r6s = r6.string();
    info = probe_rocm(r.str(), r6s.c_str());
    VJ_CHECK(info.prefix == r6s && info.missing_tools.empty() && info.major == 6);
    c = rocm_check(info);
    VJ_CHECK(c.level == Level::Warn && has(c.detail, "ROCm 6.4.1-120") && has(c.detail, "failed with ROCm 6.4"));

    // Tools but no version file.
    const fs::path rx = r.dir / "rocm-x";
    tools(rx, "bin");
    const std::string rxs = rx.string();
    c = rocm_check(probe_rocm(r.str(), rxs.c_str()));
    VJ_CHECK(c.level == Level::Warn && has(c.detail, "version unknown") && has(c.detail, "version not readable"));
    VJ_CHECK(probe_rocm(r.str(), "").prefix == r.str() + "/opt/rocm");  // empty override = unset
}

void test_supported_config_and_host_checks() {
    VJ_CHECK(supported_config_check({k32GB, k32GB}).level == Level::Pass);
    VJ_CHECK(has(supported_config_check({k32GB, k32GB}).detail, "2 gfx906 GPU(s) (32.0 GiB + 32.0 GiB)"));
    const auto one16 = supported_config_check({k16GB});
    VJ_CHECK(one16.level == Level::Warn && has(one16.detail, "the Ornith-122B GPU path needs 2 gfx906 GPUs"));

    TempRoot r;
    put(r.dir / "proc/sys/kernel/osrelease", "7.0.0-43-generic\n");
    fs::create_directories(r.dir / "sys/module/amdgpu");
    add_node(r.dir, 1, 90006, k32GB);
    add_node(r.dir, 2, 90006, k32GB);
    tools(r.dir / "opt/rocm", "llvm/bin");
    put(r.dir / "opt/rocm/.info/version", "7.12.0\n");
    const auto all = host_checks(r.str(), nullptr);
    VJ_CHECK(all.size() == 4);
    VJ_CHECK(all[0].name == "host.kernel" && all[1].name == "host.amdgpu_kfd" && all[2].name == "host.rocm_toolchain" &&
             all[3].name == "host.supported_config");
    for (const auto& c : all) {
        VJ_CHECK(c.level != Level::Fail);  // doctor never fails a host on these
        VJ_CHECK(c.level == Level::Pass);
        std::printf("  %-22s PASS  %s\n", c.name.c_str(), c.detail.c_str());
    }
    // Nothing at all under the root: every check is a WARN with a reason, none is a FAIL.
    TempRoot empty;
    for (const auto& c : host_checks(empty.str(), nullptr)) VJ_CHECK(c.level == Level::Warn && !c.detail.empty());
}

void test_open_hint() {
    VJ_CHECK(has(device_open_hint(EACCES), "'render' and 'video' groups"));
    VJ_CHECK(has(device_open_hint(EPERM), "usermod -aG render,video"));
    VJ_CHECK(has(device_open_hint(ENOENT), "amdgpu driver"));
    VJ_CHECK(device_open_hint(EIO).empty() && device_open_hint(0).empty());
}

}  // namespace

int main() {
    test_arena_budget();
    test_topology_scan();
    test_kernel();
    test_driver();
    test_rocm();
    test_supported_config_and_host_checks();
    test_open_hint();
    std::printf("test_host_preflight: PASS\n");
    return 0;
}
