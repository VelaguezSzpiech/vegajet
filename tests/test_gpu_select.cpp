// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/gpu_select: --gpu / VJ_GPU_DEVICES parsing (lists, ranges, all, whitespace, every malformed form), --gpu over
// environment over default, canonical round trip, picking against the device count (out of range names the source),
// the header template filter applied to a kfd::DeviceInfo-like struct exactly as kfd::Runtime::init does, the
// read-only topology scan of a synthetic sysfs tree (non-gfx906 nodes skipped, PCI location order), and the CLI
// helpers (extract_gpu_option, apply_gpu_option exporting VJ_GPU_DEVICES, `--gpu list`).
#include "cli/serve_cmd.hpp"
#include "core/gpu_select.hpp"
#include "tests/test_common.hpp"

#include <filesystem>
#include <fstream>

using namespace vegajet;
namespace fs = std::filesystem;

namespace {

gpusel::Selection P(const std::string& s) {
    gpusel::Selection x;
    std::string err;
    if (!gpusel::parse(s, x, &err)) { std::fprintf(stderr, "parse('%s') failed: %s\n", s.c_str(), err.c_str()); VJ_CHECK(false); }
    return x;
}

struct FakeInfo {  // the fields kfd::DeviceInfo has that matter here
    uint32_t node_id = 0, location_id = 0;
};

void put(const fs::path& p, const std::string& text) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << text;
}

}  // namespace

int main() {
    // ---- parsing
    VJ_CHECK(P("").all && P("all").all && P("  all ").all);
    VJ_CHECK(P("1").indices == std::vector<uint32_t>({1}));
    VJ_CHECK(P("2,0").indices == std::vector<uint32_t>({0, 2}));
    VJ_CHECK(P("0, 2-3").indices == std::vector<uint32_t>({0, 2, 3}));
    VJ_CHECK(P("1-1").indices == std::vector<uint32_t>({1}) && P("63").indices == std::vector<uint32_t>({63}));
    for (const char* bad : {"-1", "a", "1,", ",1", "1,,2", "3-1", "1-", "-", "0,0", "0-2,1", "64", "1.5", "0x1", "1 2", "1;2"}) {
        gpusel::Selection s;
        std::string err;
        VJ_CHECK(!gpusel::parse(bad, s, &err) && !err.empty());
    }
    VJ_CHECK(gpusel::canonical(P("3,1-2")) == "1,2,3" && gpusel::canonical(P("all")) == "all");
    VJ_CHECK(gpusel::canonical(P(gpusel::canonical(P("2,0-1")))) == "0,1,2");
    // ---- precedence: --gpu > VJ_GPU_DEVICES > all
    {
        gpusel::Selection s;
        std::string err, flag = "1";
        VJ_CHECK(gpusel::resolve(&flag, "0,2", s, &err) && s.source == "--gpu" && gpusel::canonical(s) == "1");
        VJ_CHECK(gpusel::resolve(nullptr, "0,2", s, &err) && s.source == "VJ_GPU_DEVICES" && gpusel::canonical(s) == "0,2");
        VJ_CHECK(gpusel::resolve(nullptr, nullptr, s, &err) && s.source == "default" && s.all);
        std::string badflag = "x";
        VJ_CHECK(!gpusel::resolve(&badflag, "0", s, &err) && err.find("--gpu x") == 0);
        VJ_CHECK(!gpusel::resolve(nullptr, "9-1", s, &err) && err.find("VJ_GPU_DEVICES=9-1") == 0);
    }
    // ---- picking / filtering
    {
        std::vector<size_t> idx;
        std::string err;
        VJ_CHECK(gpusel::pick(P("all"), 3, idx, &err) && idx == std::vector<size_t>({0, 1, 2}));
        VJ_CHECK(gpusel::pick(P("2"), 3, idx, &err) && idx == std::vector<size_t>({2}));
        gpusel::Selection s = P("1,3");
        s.source = "VJ_GPU_DEVICES";
        VJ_CHECK(!gpusel::pick(s, 3, idx, &err) && err.find("GPU 3") == 0 && err.find("VJ_GPU_DEVICES") != std::string::npos &&
                 err.find("valid: 0..2") != std::string::npos);
        VJ_CHECK(!gpusel::pick(P("0"), 0, idx, &err) && err.find("only 0") != std::string::npos);
        std::vector<FakeInfo> devs = {{4, 10}, {5, 20}, {6, 30}};
        VJ_CHECK(gpusel::filter(devs, P("0,2"), &err) && devs.size() == 2 && devs[0].node_id == 4 && devs[1].node_id == 6);
        // filter_from_env, as kfd::Runtime::init applies it
        std::vector<FakeInfo> d2 = {{4, 10}, {5, 20}, {6, 30}};
        unsetenv("VJ_GPU_DEVICES");
        VJ_CHECK(gpusel::filter_from_env(d2, &err) && d2.size() == 3);
        setenv("VJ_GPU_DEVICES", "1", 1);
        VJ_CHECK(gpusel::filter_from_env(d2, &err) && d2.size() == 1 && d2[0].node_id == 5);
        setenv("VJ_GPU_DEVICES", "4", 1);
        std::vector<FakeInfo> d3 = {{4, 10}, {5, 20}};
        VJ_CHECK(!gpusel::filter_from_env(d3, &err) && err.find("VJ_GPU_DEVICES=4") == 0 && d3.size() == 2);
        setenv("VJ_GPU_DEVICES", "zz", 1);
        VJ_CHECK(!gpusel::filter_from_env(d3, &err));
        unsetenv("VJ_GPU_DEVICES");
    }
    // ---- synthetic KFD topology: CPU node, two gfx906 in reverse PCI order, one non-gfx906 GPU
    const char* tmp = std::getenv("TMPDIR");
    const fs::path root = fs::path(tmp && *tmp ? tmp : "/tmp") / ("vj-gpusel-" + std::to_string(::getpid()));
    {
        const fs::path n = root / "sys/class/kfd/kfd/topology/nodes";
        put(n / "0/properties", "cpu_cores_count 16\nsimd_count 0\ngfx_target_version 0\nlocation_id 0\n");
        put(n / "1/properties", "simd_count 240\ngfx_target_version 90006\ndrm_render_minor 129\nlocation_id 1024\n");
        put(n / "1/gpu_id", "5555\n");
        put(n / "1/mem_banks/0/properties", "heap_type 1\nsize_in_bytes 34342961152\n");
        put(n / "2/properties", "simd_count 240\ngfx_target_version 90006\ndrm_render_minor 128\nlocation_id 512\n");
        put(n / "2/gpu_id", "4444\n");
        put(n / "2/mem_banks/0/properties", "size_in_bytes 34342961152\n");
        put(n / "3/properties", "simd_count 120\ngfx_target_version 110000\ndrm_render_minor 130\nlocation_id 256\n");
        put(n / "notanode/properties", "gfx_target_version 90006\n");
        const auto devs = gpusel::scan_topology(root.string());
        VJ_CHECK(devs.size() == 2);
        VJ_CHECK(devs[0].node_id == 2 && devs[0].index == 0 && devs[0].gpu_id == 4444 && devs[0].drm_render_minor == 128);
        VJ_CHECK(devs[1].node_id == 1 && devs[1].index == 1 && devs[1].simd_count == 240 && devs[1].vram_bytes == 34342961152ull);
        const std::string t = gpusel::format_list(devs, P("1"));
        VJ_CHECK(t.find("selection: 1") != std::string::npos && t.find("4444") != std::string::npos && t.find("32.0") != std::string::npos);
        VJ_CHECK(gpusel::scan_topology((root / "missing").string()).empty());
        VJ_CHECK(gpusel::format_list({}, P("all")).find("no gfx906") == 0);
    }
    // ---- CLI helpers
    {
        std::vector<std::string> args = {"model-file", "--gpu", "1", "--port", "0"};
        std::string spec, err;
        bool given = false;
        VJ_CHECK(cli::extract_gpu_option(args, spec, given, &err) == 0 && given && spec == "1");
        VJ_CHECK(args == std::vector<std::string>({"model-file", "--port", "0"}));
        args = {"--gpu=0,2", "x"};
        VJ_CHECK(cli::extract_gpu_option(args, spec, given, &err) == 0 && spec == "0,2" && args.size() == 1);
        args = {"--gpu", "1", "--gpu", "2"};
        VJ_CHECK(cli::extract_gpu_option(args, spec, given, &err) == 2 && err.find("more than once") != std::string::npos);
        args = {"--gpu"};
        VJ_CHECK(cli::extract_gpu_option(args, spec, given, &err) == 2);
        args = {"--port", "0"};
        VJ_CHECK(cli::extract_gpu_option(args, spec, given, &err) == 0 && !given);

        gpusel::Selection sel;
        bool listed = true;
        unsetenv("VJ_GPU_DEVICES");
        std::string f1 = "2,0";
        VJ_CHECK(cli::apply_gpu_option(&f1, sel, &listed, &err) == 0 && !listed && gpusel::canonical(sel) == "0,2");
        VJ_CHECK(std::getenv("VJ_GPU_DEVICES") && std::string(std::getenv("VJ_GPU_DEVICES")) == "0,2");  // exported
        VJ_CHECK(cli::apply_gpu_option(nullptr, sel, &listed, &err) == 0 && sel.source == "VJ_GPU_DEVICES");
        std::string bad = "1-0";
        VJ_CHECK(cli::apply_gpu_option(&bad, sel, &listed, &err) == 2);
        unsetenv("VJ_GPU_DEVICES");
        VJ_CHECK(cli::apply_gpu_option(nullptr, sel, &listed, &err) == 0 && sel.all && !std::getenv("VJ_GPU_DEVICES"));
        setenv("VJ_SYSFS_ROOT", root.c_str(), 1);
        std::string lst = "list";
        VJ_CHECK(cli::apply_gpu_option(&lst, sel, &listed, &err) == 0 && listed && !std::getenv("VJ_GPU_DEVICES"));
        unsetenv("VJ_SYSFS_ROOT");
    }
    fs::remove_all(root);
    std::printf("test_gpu_select: parsing, precedence, canonical, pick/filter, env filter (kfd hook), sysfs scan + order, "
                "CLI extract/apply/list OK\nPASS\n");
    return 0;
}
