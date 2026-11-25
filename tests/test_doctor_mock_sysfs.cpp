// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/doctor_rules against synthetic /proc + /sys trees built in a temp directory: every rule is driven to PASS,
// WARN and FAIL (kernel release, cmdline flags, KFD topology and device node, PCIe Gen1 / x8 / Gen2-cap host links,
// power1_cap N-aware limits, power1_input rolling mean and peak, HBM2 TMem bands, 6-block RAS, pp_dpm_mclk level 0,
// G-SF-12 /dev/kfd holders and guard / fan / sensor-reader services in normal and preflight mode).
// Device directories use neutral names (gpu0, rp0, ...); nothing here touches the real /sys, /proc or /dev.
#include "core/doctor_rules.hpp"
#include "tests/test_common.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <unistd.h>

namespace fs = std::filesystem;
using namespace vegajet::doctor_rules;

namespace {

struct Fixture {
    fs::path root;
    Fixture() {
        const char* t = std::getenv("TMPDIR");
        std::string tmpl = std::string(t && *t ? t : "/tmp") + "/vj-doctor-XXXXXX";
        VJ_CHECK(::mkdtemp(&tmpl[0]) != nullptr);
        root = tmpl;
    }
    ~Fixture() { std::error_code ec; fs::remove_all(root, ec); }
    void put(const std::string& rel, const std::string& text) {
        const fs::path p = root / rel;
        fs::create_directories(p.parent_path());
        std::ofstream(p) << text;
    }
    void rm(const std::string& rel) { fs::remove_all(root / rel); }
    void mkdir(const std::string& rel) { fs::create_directories(root / rel); }
    void symlink(const std::string& target, const std::string& rel) {
        fs::create_directories((root / rel).parent_path());
        fs::create_symlink(target, root / rel);
    }
    std::string s() const { return root.string(); }

    static void link_files(Fixture& f, const std::string& dir, const char* cur, const char* max, int cw, int mw) {
        f.put(dir + "/current_link_speed", std::string(cur) + " PCIe\n");
        f.put(dir + "/max_link_speed", std::string(max) + " PCIe\n");
        f.put(dir + "/current_link_width", std::to_string(cw) + "\n");
        f.put(dir + "/max_link_width", std::to_string(mw) + "\n");
    }
    // A healthy gfx906 behind root port rp<i> -> switch up<i> -> down<i> -> gpu<i>.
    void add_gpu(int i) {
        const std::string n = std::to_string(i);
        const std::string rp = "sys/devices/pcihost0/rp" + n, up = rp + "/up" + n, dn = up + "/dn" + n, g = dn + "/gpu" + n;
        link_files(*this, rp, "8.0 GT/s", "8.0 GT/s", 16, 16);
        link_files(*this, up, "8.0 GT/s", "8.0 GT/s", 16, 16);
        link_files(*this, dn, "16.0 GT/s", "16.0 GT/s", 16, 16);
        link_files(*this, g, "16.0 GT/s", "16.0 GT/s", 16, 16);
        put(g + "/vendor", "0x1002\n");
        put(g + "/device", "0x66a1\n");
        put(g + "/vbios_version", "113-TEST-000\n");
        const unsigned long long base = 0x38000000000ull + (unsigned long long)i * 0x1000000000ull;
        char res[128];
        std::snprintf(res, sizeof(res), "0x%016llx 0x%016llx 0x000000000014220c\n", base, base + (32ull << 30) - 1);
        put(g + "/resource", res);
        put(g + "/hwmon/hwmon" + n + "/power1_cap", "150000000\n");
        put(g + "/hwmon/hwmon" + n + "/power1_input", "21000000\n");
        put(g + "/hwmon/hwmon" + n + "/temp3_input", "41000\n");
        put(g + "/pp_dpm_mclk", "0: 350Mhz\n1: 800Mhz *\n");
        for (const char* b : {"umc", "gfx", "hdp", "mmhub", "pcie_bif", "sdma"}) put(g + "/ras/" + b + "_err_count", "ue: 0\nce: 0\n");
        symlink("../../../devices/pcihost0/rp" + n + "/up" + n + "/dn" + n + "/gpu" + n, "sys/bus/pci/devices/gpu" + n);
        put("sys/class/kfd/kfd/topology/nodes/" + std::to_string(i + 1) + "/properties",
            "cpu_cores_count 0\ngfx_target_version 90006\np2p_links_count 1\n");
    }
    void healthy(int n_gpus) {
        put("proc/sys/kernel/osrelease", "7.0.0-43-generic\n");
        put("proc/cmdline", "BOOT_IMAGE=/vmlinuz root=UUID=0 ro amdgpu.vm_fragment_size=9 iommu=pt quiet\n");
        put("dev/kfd", "");
        // kfd.access evaluates classic owner bits against the invoking identity; the fixture file is owned by the
        // test user, so grant owner read+write (the real node grants rw through its group / mode instead).
        std::error_code perm_ec;
        fs::permissions(root / "dev/kfd", fs::perms::owner_all, perm_ec);
        put("sys/class/kfd/kfd/topology/nodes/0/properties", "cpu_cores_count 16\ngfx_target_version 0\n");
        for (int i = 0; i < n_gpus; ++i) add_gpu(i);
        put("sys/devices/pcihost0/nic0/vendor", "0x8086\n");  // non-AMD device: ignored
        put("sys/devices/pcihost0/nic0/device", "0x1533\n");
        symlink("../../../devices/pcihost0/nic0", "sys/bus/pci/devices/nic0");
        mkdir("proc/1/fd");
        symlink("/dev/null", "proc/1/fd/0");
        mkdir("proc/self");  // non-numeric: skipped
        mkdir("sys/fs/cgroup/system.slice/mi50-guard.service");
        mkdir("sys/fs/cgroup/system.slice/vifted.service");
    }
};

Status st(const Report& r, const std::string& name) {
    const Check* c = r.find(name);
    if (!c) { std::fprintf(stderr, "missing check %s\n", name.c_str()); std::abort(); }
    return c->status;
}
bool detail_has(const Report& r, const std::string& name, const std::string& s) {
    const Check* c = r.find(name);
    return c && c->detail.find(s) != std::string::npos;
}

Report run_on(const Fixture& f, Mode m = Mode::Normal) {
    Options o;
    o.root = f.s();
    o.mode = m;
    return run(o);
}

} // namespace

int main() {
    // ---- pure helpers
    int ma = 0, mi = 0;
    VJ_CHECK(parse_osrelease("7.0.0-43-generic", ma, mi) && ma == 7 && mi == 0 && !parse_osrelease("x", ma, mi));
    VJ_CHECK(parse_link_speed_gts("2.5 GT/s PCIe") == 2.5 && parse_link_speed_gts("Unknown") == 0.0);
    VJ_CHECK(split_cmdline("  a b\tc ").size() == 3);

    // ---- healthy 2-GPU host: zero WARN, zero FAIL
    {
        Fixture f;
        f.healthy(2);
        const Report r = run_on(f);
        if (r.warn || r.fail) std::fputs(format_text(r).c_str(), stderr);
        VJ_CHECK(r.ok() && r.warn == 0 && r.gpus.size() == 2);
        VJ_CHECK(st(r, "kernel.osrelease") == Status::Pass && st(r, "kfd.topology") == Status::Pass);
        VJ_CHECK(st(r, "gpu.gpu0.pcie_host_link") == Status::Pass && detail_has(r, "gpu.gpu0.pcie_host_link", "rp0<->up0"));
        VJ_CHECK(st(r, "gpu.gpu1.p2p_links") == Status::Pass && st(r, "gpu.gpu0.bar0") == Status::Pass);
        VJ_CHECK(st(r, "gsf12.kfd_holders") == Status::Pass && st(r, "gpu.gpu0.uclk0") == Status::Pass);
        VJ_CHECK(to_json(r).find("\"ok\":true") != std::string::npos);
        std::printf("  healthy fixture: %d checks pass\n", r.pass);
    }
    // ---- kernel release and cmdline
    {
        Fixture f;
        f.healthy(2);
        f.put("proc/sys/kernel/osrelease", "6.8.0-generic\n");
        f.put("proc/cmdline", "ro amdgpu.vm_block_size=11 amdgpu.noretry=1 amdgpu.vm_fragment_size=90\n");
        const Report r = run_on(f);
        VJ_CHECK(st(r, "kernel.osrelease") == Status::Fail);
        VJ_CHECK(st(r, "kernel.cmdline.unsafe_vm") == Status::Fail && detail_has(r, "kernel.cmdline.unsafe_vm", "vm_block_size=11"));
        VJ_CHECK(st(r, "kernel.cmdline.noretry") == Status::Warn);
        VJ_CHECK(st(r, "kernel.cmdline.vm_fragment_size") == Status::Warn);  // "=90" is not "=9"
        VJ_CHECK(st(r, "kernel.cmdline.iommu_pt") == Status::Warn);
        f.put("proc/cmdline", "ro amdgpu.vm_size=2048 amdgpu.vm_fragment_size=9 iommu=pt\n");
        VJ_CHECK(st(run_on(f), "kernel.cmdline.unsafe_vm") == Status::Fail);
        f.put("proc/sys/kernel/osrelease", "7.2.1\n");
        VJ_CHECK(st(run_on(f), "kernel.osrelease") == Status::Pass);
    }
    // ---- KFD
    {
        Fixture f;
        f.healthy(2);
        f.rm("dev/kfd");
        f.rm("sys/class/kfd/kfd/topology/nodes/2");
        Report r = run_on(f);
        VJ_CHECK(st(r, "kfd.device_node") == Status::Fail && st(r, "kfd.topology") == Status::Warn);
        f.rm("sys/class/kfd");
        r = run_on(f);
        VJ_CHECK(st(r, "kfd.topology") == Status::Fail);
    }
    // ---- PCIe: Gen1 WARN, x8 never FAIL, Gen2-capable FAIL
    {
        Fixture f;
        f.healthy(2);
        Fixture::link_files(f, "sys/devices/pcihost0/rp0", "2.5 GT/s", "8.0 GT/s", 16, 16);
        Fixture::link_files(f, "sys/devices/pcihost0/rp1", "8.0 GT/s", "8.0 GT/s", 8, 8);
        Report r = run_on(f);
        VJ_CHECK(st(r, "gpu.gpu0.pcie_host_link") == Status::Warn && detail_has(r, "gpu.gpu0.pcie_host_link", "Gen1 2.5 GT/s"));
        VJ_CHECK(st(r, "gpu.gpu1.pcie_host_link") == Status::Pass && detail_has(r, "gpu.gpu1.pcie_host_link", "x8 slot"));
        Fixture::link_files(f, "sys/devices/pcihost0/rp1", "2.5 GT/s", "8.0 GT/s", 8, 8);  // Gen1 x8: still only WARN
        VJ_CHECK(st(run_on(f), "gpu.gpu1.pcie_host_link") == Status::Warn);
        Fixture::link_files(f, "sys/devices/pcihost0/rp1", "5.0 GT/s", "5.0 GT/s", 16, 16);
        r = run_on(f);
        VJ_CHECK(st(r, "gpu.gpu1.pcie_host_link") == Status::Fail);
        f.rm("sys/devices/pcihost0/rp0/current_link_speed");  // root port no longer a hop -> up0/dn0 chain still found
        r = run_on(f);
        VJ_CHECK(detail_has(r, "gpu.gpu0.pcie_host_link", "up0<->dn0"));
    }
    // ---- power cap (N-aware) and power1_input rolling mean / peak
    {
        Fixture f;
        f.healthy(2);
        f.put("sys/devices/pcihost0/rp0/up0/dn0/gpu0/hwmon/hwmon0/power1_cap", "225000000\n");
        Report r = run_on(f);
        // Reference-rig PSU budget: advisory on a normal host, hard only in the G-SF-12 preflight.
        VJ_CHECK(st(r, "gpu.gpu0.psu_power_cap") == Status::Warn && st(r, "gpu.gpu1.psu_power_cap") == Status::Pass);
        VJ_CHECK(st(run_on(f, Mode::Gsf12Preflight), "gpu.gpu0.psu_power_cap") == Status::Fail);
        f.put("sys/devices/pcihost0/rp0/up0/dn0/gpu0/hwmon/hwmon0/power1_cap", "175000000\n");
        VJ_CHECK(st(run_on(f), "gpu.gpu0.psu_power_cap") == Status::Pass);
        f.add_gpu(2);  // N = 3: the limit drops to 150 W
        r = run_on(f);
        VJ_CHECK(st(r, "gpu.gpu0.psu_power_cap") == Status::Warn && st(r, "gpu.gpu2.p2p_links") == Status::Warn);  // 1 < N-1
        r = run_on(f, Mode::Gsf12Preflight);
        VJ_CHECK(st(r, "gpu.gpu0.psu_power_cap") == Status::Fail && st(r, "gpu.gpu2.p2p_links") == Status::Fail);
        // Rolling mean over samples: the sleep hook writes the next reading. GPUs are sampled in directory order, so
        // the sequence is driven on gpu0 (read first); later GPUs just see its last value.
        const std::string pin = "sys/devices/pcihost0/rp0/up0/dn0/gpu0/hwmon/hwmon0/power1_input";
        auto sampled = [&](std::vector<const char*> seq) {
            f.put(pin, seq[0]);
            Options o;
            o.root = f.s();
            o.power_samples = static_cast<uint32_t>(seq.size());
            o.sleep_fn = [&](uint32_t, uint32_t i) { f.put(pin, seq[i]); };
            return run(o);
        };
        r = sampled({"140000000", "160000000", "165000000"});  // N = 3: mean 155 W -> pass
        VJ_CHECK(st(r, "gpu.gpu0.power_draw") == Status::Pass && r.gpus[0].power_samples_w.size() == 3);
        r = sampled({"150000000", "170000000", "160000000"});  // mean 160 W > 155 W
        VJ_CHECK(st(r, "gpu.gpu0.power_draw") == Status::Fail);
        r = sampled({"20000000", "275000000"});                // peak >= 270 W
        VJ_CHECK(st(r, "gpu.gpu0.power_draw") == Status::Fail && detail_has(r, "gpu.gpu0.power_draw", "peak 275.0"));
        f.rm("sys/devices/pcihost0/rp1/up1/dn1/gpu1/hwmon/hwmon1/power1_input");
        f.put("sys/devices/pcihost0/rp1/up1/dn1/gpu1/hwmon/hwmon1/power1_average", "30000000\n");  // older kernels
        VJ_CHECK(st(run_on(f), "gpu.gpu1.power_draw") == Status::Pass);
        f.rm("sys/devices/pcihost0/rp1/up1/dn1/gpu1/hwmon/hwmon1/power1_average");
        f.rm("sys/devices/pcihost0/rp1/up1/dn1/gpu1/hwmon/hwmon1/power1_cap");
        r = run_on(f);
        VJ_CHECK(st(r, "gpu.gpu1.power_draw") == Status::Warn && st(r, "gpu.gpu1.psu_power_cap") == Status::Warn);
    }
    // ---- HBM2 TMem bands, RAS, uclk0, BAR0, VBIOS parity
    {
        Fixture f;
        f.healthy(2);
        const std::string g0 = "sys/devices/pcihost0/rp0/up0/dn0/gpu0", g1 = "sys/devices/pcihost0/rp1/up1/dn1/gpu1";
        for (auto [mc, want] : {std::pair<const char*, Status>{"64900", Status::Pass}, {"65000", Status::Warn},
                                {"69900", Status::Warn}, {"70000", Status::Fail}}) {
            f.put(g0 + "/hwmon/hwmon0/temp3_input", mc);
            VJ_CHECK(st(run_on(f), "gpu.gpu0.tmem") == want);
        }
        f.put(g0 + "/ras/umc_err_count", "ue: 1\nce: 0\n");
        f.put(g1 + "/ras/gfx_err_count", "ue: 0\nce: 3\n");
        Report r = run_on(f);
        VJ_CHECK(st(r, "gpu.gpu0.ras_ecc") == Status::Fail && detail_has(r, "gpu.gpu0.ras_ecc", "umc"));
        VJ_CHECK(st(r, "gpu.gpu1.ras_ecc") == Status::Warn && detail_has(r, "gpu.gpu1.ras_ecc", "CE=3"));
        f.put(g1 + "/ras/gfx_err_count", "ue: 0\nce: 0\nde: 2\n");
        VJ_CHECK(st(run_on(f), "gpu.gpu1.ras_ecc") == Status::Fail);
        f.put(g1 + "/ras/gfx_err_count", "ue: 0\nce: 0\n");
        f.rm(g1 + "/ras/sdma_err_count");
        r = run_on(f);
        VJ_CHECK(st(r, "gpu.gpu1.ras_ecc") == Status::Warn && detail_has(r, "gpu.gpu1.ras_ecc", "5/6"));
        f.put(g1 + "/pp_dpm_mclk", "0: 400Mhz\n1: 800Mhz *\n");
        VJ_CHECK(st(run_on(f), "gpu.gpu1.uclk0") == Status::Fail);
        f.put(g1 + "/resource", "0x00000000e0000000 0x00000000efffffff 0x0000000000140204\n");  // 256 MiB BAR
        f.put(g1 + "/vbios_version", "113-TEST-001\n");
        r = run_on(f);
        // Reference-rig reference (32 GiB BAR0, identical VBIOS) is advisory outside the G-SF-12 preflight.
        VJ_CHECK(st(r, "gpu.gpu1.bar0") == Status::Warn && st(r, "gpu.vbios_parity") == Status::Warn);
        r = run_on(f, Mode::Gsf12Preflight);
        VJ_CHECK(st(r, "gpu.gpu1.bar0") == Status::Fail && st(r, "gpu.vbios_parity") == Status::Fail);
    }
    // ---- G-SF-12: /dev/kfd holders and services, normal vs preflight
    {
        Fixture f;
        f.healthy(2);
        Report r = run_on(f, Mode::Gsf12Preflight);  // guard + fan daemon still running -> preflight fails
        VJ_CHECK(st(r, "gsf12.service.mi50-guard.service") == Status::Fail && st(r, "gsf12.service.vifted.service") == Status::Fail);
        f.rm("sys/fs/cgroup/system.slice/mi50-guard.service");
        f.rm("sys/fs/cgroup/system.slice/vifted.service");
        r = run_on(f, Mode::Gsf12Preflight);
        VJ_CHECK(st(r, "gsf12.service.mi50-guard.service") == Status::Pass && st(r, "gsf12.service.vifted.service") == Status::Pass);
        VJ_CHECK(st(r, "gsf12.service.coolercontrold.service") == Status::Pass && st(r, "gsf12.kfd_holders") == Status::Pass);
        r = run_on(f);  // the same stopped state is unsafe for a normal run
        VJ_CHECK(st(r, "gsf12.service.mi50-guard.service") == Status::Warn && st(r, "gsf12.service.vifted.service") == Status::Warn);
        f.mkdir("sys/fs/cgroup/system.slice/coolercontrold.service");
        VJ_CHECK(st(run_on(f, Mode::Gsf12Preflight), "gsf12.service.coolercontrold.service") == Status::Fail);
        VJ_CHECK(st(run_on(f), "gsf12.service.coolercontrold.service") == Status::Warn);
        f.symlink("/dev/kfd", "proc/4242/fd/7");  // readlink only: the target is never opened
        f.symlink("/dev/null", "proc/4242/fd/0");
        r = run_on(f, Mode::Gsf12Preflight);
        VJ_CHECK(st(r, "gsf12.kfd_holders") == Status::Fail && detail_has(r, "gsf12.kfd_holders", "4242"));
        VJ_CHECK(st(run_on(f), "gsf12.kfd_holders") == Status::Warn);
        // Injected service provider (e.g. a caller that already queried the init system).
        Options o;
        o.root = f.s();
        o.mode = Mode::Gsf12Preflight;
        o.service_active = [](const std::string& u) { return u == "vifted.service"; };
        r = run(o);
        VJ_CHECK(st(r, "gsf12.service.vifted.service") == Status::Fail && st(r, "gsf12.service.mi50-guard.service") == Status::Pass);
    }
    // ---- no GPU at all
    {
        Fixture f;
        f.put("proc/sys/kernel/osrelease", "7.0.0\n");
        const Report r = run_on(f);
        VJ_CHECK(st(r, "gpu.discovery") == Status::Fail && st(r, "kfd.topology") == Status::Warn && !r.ok());
        VJ_CHECK(to_json(r).find("\"ok\":false") != std::string::npos && format_text(r).find("-> FAIL") != std::string::npos);
    }
    std::printf("test_doctor_mock_sysfs: kernel/cmdline/KFD/PCIe/power/TMem/RAS/uclk0/G-SF-12 rules on synthetic sysfs OK\nPASS\n");
    return 0;
}
