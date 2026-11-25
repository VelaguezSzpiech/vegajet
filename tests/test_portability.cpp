// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/portability: the relaxed doctor verdicts (small / high BAR0, missing P2P links, single GPU and stock power
// caps are WARN, the reference rig stays PASS), the stat()-only device-access probe against a synthetic /dev tree
// (missing /dev/kfd, wrong group with the exact usermod fix, owner / group / other / root access, missing and
// partially accessible render nodes), and the fit check + preflight against a synthetic KFD topology (too few GPUs,
// a 16 GiB card, the runner arena, VJ_GPU_DEVICES narrowing the selection, the reference 2 x 32 GiB host fitting).
#include "core/portability.hpp"
#include "tests/test_common.hpp"

#include <filesystem>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

using namespace vegajet;
using doctor::CheckStatus;
namespace fs = std::filesystem;

namespace {

constexpr uint64_t G = portability::kGiB;
constexpr uint64_t kMi50Vram = 34342961152ULL;  // KFD mem_banks size_in_bytes of a 32 GiB MI50

void put(const fs::path& p, const std::string& text, mode_t mode = 0644) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << text;
    ::chmod(p.c_str(), mode);
}

bool has(const std::string& s, const std::string& sub) { return s.find(sub) != std::string::npos; }

portability::Identity who(uid_t uid, std::vector<gid_t> groups) {
    portability::Identity id;
    id.uid = uid;
    id.groups = std::move(groups);
    id.group_name = [](gid_t g) { return "grp" + std::to_string(g); };
    return id;
}

gpusel::Device dev(uint32_t i, uint64_t vram) {
    gpusel::Device d;
    d.index = i;
    d.vram_bytes = vram;
    return d;
}

}  // namespace

int main() {
    // ---- relaxed doctor verdicts
    VJ_CHECK(portability::bar0(32 * G, true).status == CheckStatus::Pass);
    {
        const auto v = portability::bar0(256ULL << 20, true);
        VJ_CHECK(v.status == CheckStatus::Warn && has(v.detail, "0.2 GiB") && has(v.detail, "TP=1"));
        VJ_CHECK(portability::bar0(32 * G, false).status == CheckStatus::Warn);
        VJ_CHECK(has(portability::bar0(32 * G, false).detail, "44-bit"));
    }
    VJ_CHECK(portability::p2p_links(1, false, 0).status == CheckStatus::Pass);
    VJ_CHECK(portability::p2p_links(2, true, 1).status == CheckStatus::Pass);
    VJ_CHECK(portability::p2p_links(3, true, 1).status == CheckStatus::Warn);
    VJ_CHECK(has(portability::p2p_links(2, false, 0).detail, "--gpu"));
    VJ_CHECK(portability::power_cap(150.0, 2).status == CheckStatus::Pass);
    VJ_CHECK(portability::power_cap(200.0, 2).status == CheckStatus::Pass);
    VJ_CHECK(portability::power_cap(225.0, 1).status == CheckStatus::Warn);       // stock MI50 cap on a 1-GPU host
    VJ_CHECK(portability::power_cap(178.0, 3).status == CheckStatus::Warn);       // N>=3 budget is 150 W
    VJ_CHECK(has(portability::power_cap(225.0, 1).detail, "PSU"));
    VJ_CHECK(portability::gpu_count(1).status == CheckStatus::Warn && has(portability::gpu_count(1).detail, "N=1"));
    VJ_CHECK(portability::gpu_count(2).status == CheckStatus::Pass);
    // None of the rig-specific verdicts can FAIL.
    for (uint64_t bar : std::initializer_list<uint64_t>{0, 256ULL << 20, 16 * G, 32 * G})
        for (bool lo : {false, true}) VJ_CHECK(portability::bar0(bar, lo).status != CheckStatus::Fail);
    for (size_t n : std::initializer_list<size_t>{1, 2, 3, 8})
        for (double w : {0.0, 150.0, 300.0}) VJ_CHECK(portability::power_cap(w, n).status != CheckStatus::Fail);

    // ---- device access against a synthetic /dev tree (stat only)
    const char* tmp = std::getenv("TMPDIR");
    const fs::path root = fs::path(tmp && *tmp ? tmp : "/tmp") / ("vj-port-" + std::to_string(::getpid()));
    fs::remove_all(root);
    const uid_t me = ::getuid();
    const gid_t mygid = ::getgid();
    {
        fs::create_directories(root / "dev");
        auto a = portability::device_access(root.string(), who(me + 1, {}));
        VJ_CHECK(a.size() == 2 && a[0].name == "kfd.access" && a[0].verdict.status == CheckStatus::Fail);
        VJ_CHECK(has(a[0].verdict.detail, "missing") && has(a[0].verdict.detail, "modprobe amdgpu"));
        VJ_CHECK(a[1].name == "dri.render_access" && a[1].verdict.status == CheckStatus::Fail);
        VJ_CHECK(has(a[1].verdict.detail, "renderD"));

        put(root / "dev/kfd", "", 0660);
        put(root / "dev/dri/renderD128", "", 0660);
        put(root / "dev/dri/card0", "", 0600);  // not a render node: ignored
        // another user, not in the owning group: FAIL with the group and the fix
        a = portability::device_access(root.string(), who(me + 1, {mygid + 1}));
        VJ_CHECK(a[0].verdict.status == CheckStatus::Fail);
        VJ_CHECK(has(a[0].verdict.detail, "sudo usermod -aG grp" + std::to_string(mygid) + " $USER"));
        VJ_CHECK(has(a[0].verdict.detail, "mode 0660"));
        VJ_CHECK(a[1].verdict.status == CheckStatus::Fail && has(a[1].verdict.detail, "renderD128"));
        // member of the owning group: PASS naming it
        a = portability::device_access(root.string(), who(me + 1, {mygid}));
        VJ_CHECK(a[0].verdict.status == CheckStatus::Pass && has(a[0].verdict.detail, "grp" + std::to_string(mygid)));
        VJ_CHECK(a[1].verdict.status == CheckStatus::Pass && has(a[1].verdict.detail, "1 of 1 render node"));
        // owner bits; root always passes; world-rw nodes pass for anyone
        VJ_CHECK(portability::device_access(root.string(), who(me, {}))[0].verdict.status == CheckStatus::Pass);
        VJ_CHECK(portability::device_access(root.string(), who(0, {}))[0].verdict.status == CheckStatus::Pass);
        ::chmod((root / "dev/kfd").c_str(), 0666);
        VJ_CHECK(portability::device_access(root.string(), who(me + 1, {}))[0].verdict.status == CheckStatus::Pass);
        // read-only for the group is not enough (the runtime opens O_RDWR)
        ::chmod((root / "dev/kfd").c_str(), 0640);
        VJ_CHECK(portability::device_access(root.string(), who(me + 1, {mygid}))[0].verdict.status == CheckStatus::Fail);
        // two render nodes, one inaccessible: the inaccessible one is reported
        put(root / "dev/dri/renderD129", "", 0600);
        a = portability::device_access(root.string(), who(me + 1, {mygid}));
        VJ_CHECK(a[1].verdict.status == CheckStatus::Fail && has(a[1].verdict.detail, "renderD129"));
        // the current process identity is well-formed (its primary group is included)
        const auto id = portability::current_identity();
        VJ_CHECK(id.uid == ::geteuid() && !id.groups.empty() && id.group_name && !id.group_name(id.groups[0]).empty());
    }

    // ---- fit check
    {
        portability::FitRequest r;
        r.model = "big-model";
        r.gpus_required = 2;
        r.arena_bytes_per_gpu = 30 * G;
        auto f = portability::check_fit(r, {dev(0, kMi50Vram)});
        VJ_CHECK(!f.ok && has(f.message, "needs 2 GPUs") && has(f.message, "1 gfx906 GPU(s) (32.0 GiB)"));
        f = portability::check_fit(r, {});
        VJ_CHECK(!f.ok && has(f.message, "vegajet doctor"));
        f = portability::check_fit(r, {dev(0, kMi50Vram), dev(1, 16 * G)});
        VJ_CHECK(!f.ok && has(f.message, "GPU 1 has 16.0 GiB") && has(f.message, "runner arena 30.0 GiB"));
        VJ_CHECK(portability::check_fit(r, {dev(0, kMi50Vram), dev(1, kMi50Vram)}).ok);
        // planned bytes beyond the arena are what count
        r.bytes_per_gpu = {31 * G, 33 * G};
        f = portability::check_fit(r, {dev(0, kMi50Vram), dev(1, kMi50Vram)});
        VJ_CHECK(!f.ok && has(f.message, "33.0 GiB per GPU"));
        r.bytes_per_gpu = {20 * G, 20 * G};
        r.arena_bytes_per_gpu = 0;
        VJ_CHECK(portability::check_fit(r, {dev(0, 24 * G), dev(1, 24 * G)}).ok);
        // a 122B-class model on one 16 GiB card
        portability::FitRequest one;
        one.model = "big-model";
        one.gpus_required = 1;
        one.bytes_per_gpu = {45 * G};
        f = portability::check_fit(one, {dev(0, 16 * G)});
        VJ_CHECK(!f.ok && has(f.message, "45.0 GiB per GPU") && has(f.message, "smaller"));
    }

    // ---- preflight: synthetic KFD topology + VJ_GPU_DEVICES (what the runner would see)
    {
        const fs::path n = root / "sys/class/kfd/kfd/topology/nodes";
        put(n / "0/properties", "simd_count 0\ngfx_target_version 0\nlocation_id 0\n");
        put(n / "1/properties", "simd_count 240\ngfx_target_version 90006\ndrm_render_minor 128\nlocation_id 512\n");
        put(n / "1/gpu_id", "1001\n");
        put(n / "1/mem_banks/0/properties", "size_in_bytes 34342961152\n");
        put(n / "2/properties", "simd_count 240\ngfx_target_version 90006\ndrm_render_minor 129\nlocation_id 1024\n");
        put(n / "2/gpu_id", "1002\n");
        put(n / "2/mem_banks/0/properties", "size_in_bytes 17179869184\n");  // a 16 GiB card
        portability::FitRequest r;
        r.model = "big-model";
        r.gpus_required = 2;
        r.arena_bytes_per_gpu = 30 * G;
        unsetenv("VJ_GPU_DEVICES");
        auto f = portability::preflight(r, root.string());
        VJ_CHECK(!f.ok && has(f.message, "GPU 1 has 16.0 GiB"));
        setenv("VJ_GPU_DEVICES", "0", 1);
        f = portability::preflight(r, root.string());
        VJ_CHECK(!f.ok && has(f.message, "needs 2 GPUs") && has(f.message, "--gpu"));
        r.gpus_required = 1;
        VJ_CHECK(portability::preflight(r, root.string()).ok);
        setenv("VJ_GPU_DEVICES", "5", 1);
        f = portability::preflight(r, root.string());
        VJ_CHECK(!f.ok && has(f.message, "VJ_GPU_DEVICES=5"));
        unsetenv("VJ_GPU_DEVICES");
        put(n / "2/mem_banks/0/properties", "size_in_bytes 34342961152\n");
        r.gpus_required = 2;
        VJ_CHECK(portability::preflight(r, root.string()).ok);  // the reference 2 x 32 GiB host fits
    }
    fs::remove_all(root);
    std::printf("test_portability: relaxed doctor verdicts, device access (stat-only, synthetic /dev), fit check, "
                "preflight on a synthetic topology OK\nPASS\n");
    return 0;
}
