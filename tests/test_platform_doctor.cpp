// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// M9 platform/OS rules on a synthetic CPU fixture: the Linux 7.0 gate, /proc/cmdline amdgpu vm_block_size / vm_size
// / noretry, and the stat-only /dev/kfd presence + read+write permission verdict with an injected identity.
// Every path lives under a fresh temp directory; nothing reads or opens the real /proc, /sys or /dev, and no device
// node is ever opened (permissions are evaluated from stat() mode bits only).
#include "core/doctor_rules.hpp"
#include "tests/test_common.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace vegajet::doctor_rules;

namespace {

struct Fixture {
    fs::path root;
    Fixture() {
        const char* t = std::getenv("TMPDIR");
        std::string tmpl = std::string(t && *t ? t : "/tmp") + "/vj-platform-doctor-XXXXXX";
        VJ_CHECK(::mkdtemp(&tmpl[0]) != nullptr);
        root = tmpl;
    }
    ~Fixture() { std::error_code ec; fs::remove_all(root, ec); }
    void put(const std::string& rel, const std::string& text) {
        const fs::path p = root / rel;
        fs::create_directories(p.parent_path());
        std::ofstream(p) << text;
    }
    void rm(const std::string& rel) { std::error_code ec; fs::remove_all(root / rel, ec); }
    void chmod(const std::string& rel, unsigned mode) { VJ_CHECK(::chmod((root / rel).c_str(), mode) == 0); }
    struct stat stat_of(const std::string& rel) const {
        struct stat st {};
        VJ_CHECK(::stat((root / rel).c_str(), &st) == 0);
        return st;
    }
    std::string s() const { return root.string(); }

    // Minimal read-only root: no GPUs (GPU discovery is not what this test exercises).
    void base() {
        put("proc/sys/kernel/osrelease", "7.0.0-43-generic\n");
        put("proc/cmdline", "BOOT_IMAGE=/vmlinuz root=UUID=0 ro amdgpu.vm_fragment_size=9 iommu=pt quiet\n");
        put("dev/kfd", "");
        chmod("dev/kfd", 0666);
    }
};

AccessIdentity identity(uid_t uid, std::vector<gid_t> groups) {
    AccessIdentity who;
    who.uid = uid;
    who.groups = std::move(groups);
    who.group_name = [](gid_t g) { return "vg" + std::to_string(g); };
    return who;
}

Report run_as(const Fixture& f, const AccessIdentity& who, Mode m = Mode::Normal) {
    Options o;
    o.root = f.s();
    o.mode = m;
    o.identity = who;
    o.identity_explicit = true;
    return run(o);
}

Status st(const Report& r, const std::string& name) {
    const Check* c = r.find(name);
    if (!c) { std::fprintf(stderr, "missing check %s\n", name.c_str()); std::abort(); }
    return c->status;
}
bool has(const Report& r, const std::string& name) { return r.find(name) != nullptr; }
bool detail_has(const Report& r, const std::string& name, const std::string& s) {
    const Check* c = r.find(name);
    return c && c->detail.find(s) != std::string::npos;
}

} // namespace

int main() {
    // ---- pure permission helper (mode bits only, uid 0 bypasses as the kernel does)
    VJ_CHECK(device_rw_allowed(0600, 1000, 1000, identity(1000, {})));
    VJ_CHECK(!device_rw_allowed(0444, 1000, 1000, identity(1000, {})));       // owner: no write bit
    VJ_CHECK(device_rw_allowed(0660, 1000, 50, identity(1001, {50})));        // group: rw
    VJ_CHECK(!device_rw_allowed(0640, 1000, 50, identity(1001, {50})));       // group: read only
    VJ_CHECK(device_rw_allowed(0666, 1000, 50, identity(1001, {})));          // other: rw
    VJ_CHECK(!device_rw_allowed(0664, 1000, 50, identity(1001, {})));          // other: read only
    VJ_CHECK(device_rw_allowed(0400, 1000, 50, identity(0, {})));             // root
    VJ_CHECK(!device_rw_allowed(0000, 1000, 50, identity(1001, {})));

    // ---- Linux 7.0 gate
    {
        Fixture f;
        f.base();
        VJ_CHECK(st(run_as(f, identity(1000, {})), "kernel.osrelease") == Status::Pass);
        f.put("proc/sys/kernel/osrelease", "6.8.0-45-generic\n");
        VJ_CHECK(st(run_as(f, identity(1000, {})), "kernel.osrelease") == Status::Fail);
        f.put("proc/sys/kernel/osrelease", "6.12.3\n");
        VJ_CHECK(st(run_as(f, identity(1000, {})), "kernel.osrelease") == Status::Fail);
        f.put("proc/sys/kernel/osrelease", "garbage\n");
        VJ_CHECK(st(run_as(f, identity(1000, {})), "kernel.osrelease") == Status::Warn);
        f.put("proc/sys/kernel/osrelease", "7.2.1\n");
        VJ_CHECK(st(run_as(f, identity(1000, {})), "kernel.osrelease") == Status::Pass);
    }

    // ---- /proc/cmdline: any explicit vm_block_size= / vm_size= is an unsafe override; noretry != 0 warns
    {
        Fixture f;
        f.base();
        Report r = run_as(f, identity(1000, {}));
        VJ_CHECK(st(r, "kernel.cmdline.unsafe_vm") == Status::Pass);
        VJ_CHECK(!has(r, "kernel.cmdline.noretry"));
        VJ_CHECK(st(r, "kernel.cmdline.vm_fragment_size") == Status::Pass);
        VJ_CHECK(st(r, "kernel.cmdline.iommu_pt") == Status::Pass);

        f.put("proc/cmdline", "ro amdgpu.vm_block_size=11 amdgpu.vm_fragment_size=9 iommu=pt\n");
        r = run_as(f, identity(1000, {}));
        VJ_CHECK(st(r, "kernel.cmdline.unsafe_vm") == Status::Fail);
        VJ_CHECK(detail_has(r, "kernel.cmdline.unsafe_vm", "vm_block_size=11"));
        VJ_CHECK(detail_has(r, "kernel.cmdline.unsafe_vm", "keep only amdgpu.vm_fragment_size=9"));

        // The engine never sets vm geometry on the command line: any explicit value, even a plausible one, is a FAIL.
        f.put("proc/cmdline", "ro amdgpu.vm_block_size=9\n");
        r = run_as(f, identity(1000, {}));
        VJ_CHECK(st(r, "kernel.cmdline.unsafe_vm") == Status::Fail);
        VJ_CHECK(detail_has(r, "kernel.cmdline.unsafe_vm", "vm_block_size=9"));

        f.put("proc/cmdline", "ro amdgpu.vm_size=2048\n");
        VJ_CHECK(st(run_as(f, identity(1000, {})), "kernel.cmdline.unsafe_vm") == Status::Fail);
        f.put("proc/cmdline", "ro amdgpu.vm_size=4096\n");
        r = run_as(f, identity(1000, {}));
        VJ_CHECK(st(r, "kernel.cmdline.unsafe_vm") == Status::Fail);
        VJ_CHECK(detail_has(r, "kernel.cmdline.unsafe_vm", "vm_size=4096"));

        f.put("proc/cmdline", "ro amdgpu.noretry=1 amdgpu.vm_fragment_size=90\n");
        r = run_as(f, identity(1000, {}));
        VJ_CHECK(st(r, "kernel.cmdline.noretry") == Status::Warn);
        VJ_CHECK(detail_has(r, "kernel.cmdline.noretry", "amdgpu.noretry=1"));
        VJ_CHECK(st(r, "kernel.cmdline.vm_fragment_size") == Status::Warn);  // "=90" is not "=9"
        VJ_CHECK(st(r, "kernel.cmdline.iommu_pt") == Status::Warn);

        f.put("proc/cmdline", "ro amdgpu.noretry=0\n");
        r = run_as(f, identity(1000, {}));
        VJ_CHECK(!has(r, "kernel.cmdline.noretry"));  // explicitly disabled: no warning

        f.put("proc/cmdline", "ro amdgpu.noretry\n");
        VJ_CHECK(st(run_as(f, identity(1000, {})), "kernel.cmdline.noretry") == Status::Warn);
    }

    // ---- /dev/kfd: stat-only presence and read+write verdict for an injected identity
    {
        Fixture f;
        f.base();
        const AccessIdentity nobody = identity(424242, {});
        Report r = run_as(f, nobody);  // world rw
        VJ_CHECK(st(r, "kfd.device_node") == Status::Pass);
        VJ_CHECK(st(r, "kfd.access") == Status::Pass);
        VJ_CHECK(detail_has(r, "kfd.access", "readable+writable"));
        VJ_CHECK(detail_has(r, "kfd.access", "mode 0666"));

        const struct stat stt = f.stat_of("dev/kfd");
        f.chmod("dev/kfd", 0640);
        r = run_as(f, nobody);
        VJ_CHECK(st(r, "kfd.access") == Status::Fail);
        VJ_CHECK(detail_has(r, "kfd.access", "usermod -aG vg" + std::to_string(stt.st_gid)));
        VJ_CHECK(detail_has(r, "kfd.access", "mode 0640"));

        f.chmod("dev/kfd", 0660);
        VJ_CHECK(st(run_as(f, identity(424242, {stt.st_gid})), "kfd.access") == Status::Pass);
        VJ_CHECK(st(run_as(f, nobody), "kfd.access") == Status::Fail);

        f.chmod("dev/kfd", 0600);
        VJ_CHECK(st(run_as(f, identity(stt.st_uid, {})), "kfd.access") == Status::Pass);
        VJ_CHECK(st(run_as(f, identity(stt.st_uid + 1, {})), "kfd.access") == Status::Fail);

        f.chmod("dev/kfd", 0400);
        VJ_CHECK(st(run_as(f, identity(0, {})), "kfd.access") == Status::Pass);  // root bypasses mode

        f.rm("dev/kfd");
        r = run_as(f, nobody);
        VJ_CHECK(st(r, "kfd.device_node") == Status::Fail);
        VJ_CHECK(st(r, "kfd.access") == Status::Fail);
        VJ_CHECK(detail_has(r, "kfd.access", "missing"));
        VJ_CHECK(to_json(r).find("\"ok\":false") != std::string::npos);
        VJ_CHECK(format_text(r).find("[FAIL] kfd.access") != std::string::npos);
    }

    // ---- the default identity path (no injection) still produces a verdict, never opening the node
    {
        Fixture f;
        f.base();
        Options o;
        o.root = f.s();
        const Report r = run(o);
        VJ_CHECK(st(r, "kfd.device_node") == Status::Pass);
        VJ_CHECK(has(r, "kfd.access"));
    }

    std::printf("test_platform_doctor: Linux 7.0 / proc/cmdline vm flags / stat-only /dev/kfd access rules OK\nPASS\n");
    return 0;
}
