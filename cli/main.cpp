#include "cli/inspect.hpp"
#include "cli/plan_rig.hpp"
#include "cli/serve_cmd.hpp"
#include "core/doctor.hpp"
#include "core/doctor_rules.hpp"
#include "core/hw_preflight.hpp"
#include "core/gpu_select.hpp"
#include "cli/chat_repl.hpp"
#include "cli/tui_top.hpp"
#include "models/ornith/ornith_gpu.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

void print_usage(const char* prog) {
    std::printf(
        "VegaJet CLI v0.1.2 — Bare-Metal gfx906 (Vega 20: MI50 / MI60) LLM Engine\n\n"
        "Usage:\n"
        "  %s inspect <model-file> [--all-keys]                Header-only GGUF metadata / tensor / footprint report\n"
        "  %s plan <model-file> [--tp N] [--ctx T] ...         Plan + graph summary only (= bench --dry-run; any build)\n"
        "  %s bench <model-file> [--tp N] [--json-out F] ...   Plan, validate code objects, run the GPU bench (GPU builds)\n"
        "  %s run <model-file> --prompt TEXT [--tp N] ...       Tokenize, plan and execute (GPU builds)\n"
        "  %s doctor [--json] [--dry-run] [--gsf12|--os] [--live-link]  Read-only CPU hardware/kernel pre-flight\n"
        "  %s soak [--seconds 60] [--gib 28.5]                 HBM2 arena pattern soak + bandwidth audit (GPU builds)\n"
        "  %s serve <model-file> | --mock [--slots N] ...      OpenAI-compatible HTTP server (tools, reasoning, batching)\n"
        "  %s top [--url URL] [--sysfs] [--once]               Live dashboard of a server's /metrics and GPU telemetry\n"
        "  %s chat [--url URL] [--system TEXT] ...             Interactive chat REPL (/stats /clear /save /load)\n"
        "  %s --help | -h                                      Show this help message\n"
        "  %s --version | -v                                   Show version\n\n"
        "GPU selection (plan, bench, run, doctor, soak, serve):\n"
        "  --gpu IDX[,IDX|A-B]...|all|list   GPUs this process may use; `--gpu list` prints the enumeration\n"
        "Environment: VJ_GPU_DEVICES (same syntax as --gpu), VJ_IO_URING=0 (table prefetch without io_uring),\n"
        "  VEGAJET_ENABLE_SUPER=0|1 (Ornith super-kernels off / on), VEGAJET_HSACO_DIR; see release/docs/CLI_REFERENCE.md.\n"
        "Exit codes: 0 ok, 1 check/plan/load failure, 2 usage error, 3 GPU backend or engine unavailable / execution failure.\n",
        prog, prog, prog, prog, prog, prog, prog, prog, prog, prog, prog);
}

void print_plan_usage() {
    std::printf("Usage: vegajet plan <model-file> [--tp N] [--vram-gib G] [--bw-gbs B[,B...]] [--ctx T] [--reuse R]\n"
                "                    [--draft <draft-file>] [--hsaco <code-object>]... [--prompt TEXT] [--gpu SPEC]\n"
                "  Dry-run load through the model plugin, per-GPU footprint, graph summary and code-object validation.\n"
                "  Never opens /dev/kfd. Same options as `vegajet bench` (bench --dry-run).\n"
                "Rig estimator: vegajet plan <model-file> --gpus N --vram 16|32 [--ctx T] [--kv-bits 16|8] [--cu 60|64] [--matrix]\n"
                "  Header-only per-GPU VRAM breakdown, repacking decisions, context headroom and MoE offload need.\n");
}

int run_doctor(const std::vector<std::string>& args, const char* prog) {
    bool as_json = false, do_bench = false, live_link = false, gsf12 = false, os_rules = false, root_given = false;
    std::string root = "/";
    double live_seconds = 10.0;
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == "--json") as_json = true;
        else if (arg == "--bench") do_bench = true;
        else if (arg == "--live-link") live_link = true;
        else if (arg == "--gsf12") gsf12 = true;
        else if (arg == "--os") os_rules = true;
        else if (arg == "--root" && i + 1 < args.size()) { root = args[++i]; root_given = true; }
        else if (arg == "--seconds" && i + 1 < args.size()) live_seconds = std::atof(args[++i].c_str());
        else if (arg == "--dry-run") {
            // doctor is strictly read-only
        } else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: %s doctor [--json] [--bench] [--dry-run] [--gsf12] [--os [--root DIR]]\n"
                        "                    [--live-link [--seconds N]] [--gpu SPEC]\n"
                        "  --gsf12      run read-only G-SF-12 preflight (zero GPU jobs, guard/fan services stopped)\n"
                        "  --os         rules-only OS report (kernel >= 7.0, amdgpu cmdline VM flags, /dev/kfd access);\n"
                        "               read-only, never opens a device node; --root runs it against a fixture tree\n"
                        "  --live-link  sample the PCIe host link while a GPU kernel runs (idle downclock vs stuck 2.5 GT/s)\n", prog);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown option for doctor: %s\n", arg.c_str());
            return 2;
        }
    }
    if (os_rules && (gsf12 || live_link)) {
        std::fprintf(stderr, "--os cannot be combined with --gsf12 / --live-link\n");
        return 2;
    }
    if (root_given && !os_rules) {
        std::fprintf(stderr, "--root is only valid with --os\n");
        return 2;
    }
    if (os_rules) {
        vegajet::doctor_rules::Options opt;
        opt.root = root;
        opt.mode = vegajet::doctor_rules::Mode::Normal;
        opt.power_samples = 1;  // no rolling-mean sleeps in a rules-only report
        const auto rep = vegajet::doctor_rules::run(opt);
        const std::string out = as_json ? vegajet::doctor_rules::to_json(rep) : vegajet::doctor_rules::format_text(rep);
        std::fwrite(out.data(), 1, out.size(), stdout);
        return rep.ok() ? 0 : 1;
    }
    if (gsf12) {
        vegajet::doctor_rules::Options opt;
        opt.root = "/";
        opt.mode = vegajet::doctor_rules::Mode::Gsf12Preflight;
        const auto rep = vegajet::doctor_rules::run(opt);
        const std::string out = as_json ? vegajet::doctor_rules::to_json(rep) : vegajet::doctor_rules::format_text(rep);
        std::fwrite(out.data(), 1, out.size(), stdout);
        return rep.ok() ? 0 : 1;
    }
    if (live_link) {
        if (!(live_seconds > 0.0) || live_seconds > 3600.0) live_seconds = 10.0;
        const auto lrep = vegajet::doctor::run_live_link(live_seconds);
        if (as_json) vegajet::doctor::print_live_link_json(lrep);
        else vegajet::doctor::print_live_link_text(lrep);
        return lrep.any_fail ? 1 : 0;
    }
    auto rep = vegajet::doctor::run_doctor();
    if (as_json) vegajet::doctor::print_report_json(rep);
    else vegajet::doctor::print_report_text(rep);
    for (const auto& w : vegajet::hw_preflight::run([&] { std::vector<std::string> b; for (const auto& g : rep.gpus) b.push_back(g.bdf); return b; }()))
        std::printf("[WARN] fabric %s %s: %s\n", w.bdf.c_str(), w.check.c_str(), w.detail.c_str());
    if (do_bench) {
        if (!vegajet::doctor::calibrate_and_save_hw_profile()) {
            std::fprintf(stderr, "[vegajet] Failed to write hw_profile.json\n");
            return 1;
        }
    }
    return (rep.fail_count == 0) ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string cmd = argv[1];
    if (cmd == "--help" || cmd == "-h") {
        print_usage(argv[0]);
        return 0;
    }
    if (cmd == "--version" || cmd == "-v") {
        std::printf("vegajet 0.1.2\n");
        return 0;
    }
    std::vector<std::string> args(argv + 2, argv + argc);

    if (cmd == "inspect") {
        return vegajet::cli::run_inspect(args);
    }
    if (cmd == "top") {  // network / sysfs reads only: no GPU selection
        return vegajet::cli::run_top(args);
    }
    if (cmd == "chat") {
        return vegajet::cli::run_chat(args);
    }

    const bool known = cmd == "plan" || cmd == "bench" || cmd == "run" || cmd == "soak" || cmd == "doctor" || cmd == "serve";
    if (!known) {
        std::fprintf(stderr, "Unknown subcommand: %s\n", cmd.c_str());
        print_usage(argv[0]);
        return 2;
    }

    // CLI-wide GPU selection: --gpu beats VJ_GPU_DEVICES; exported for kfd::Runtime::init.
    std::string gpu_spec, gerr;
    bool gpu_given = false, listed = false;
    vegajet::gpusel::Selection gpus;
    if (vegajet::cli::extract_gpu_option(args, gpu_spec, gpu_given, &gerr) != 0 ||
        vegajet::cli::apply_gpu_option(gpu_given ? &gpu_spec : nullptr, gpus, &listed, &gerr) != 0) {
        std::fprintf(stderr, "[vegajet] %s\n", gerr.c_str());
        return 2;
    }
    if (listed) return 0;

    if (cmd == "plan") {
        if (vegajet::cli::wants_rig_plan(args)) return vegajet::cli::run_rig_plan(args);
        for (const auto& a : args)
            if (a == "--help" || a == "-h") { print_plan_usage(); return 0; }
        args.insert(args.begin(), "--dry-run");  // first, so a trailing value-less option cannot swallow it
        return vegajet::cli::run_bench("bench", args);
    }

    if (cmd == "bench" || cmd == "run") {
        return vegajet::cli::run_bench(cmd, args);
    }

    if (cmd == "soak") {
        return vegajet::cli::run_soak(args);
    }

    if (cmd == "doctor") {
        return run_doctor(args, argv[0]);
    }

    vegajet::ornith::register_ornith_serve_engine();
    return vegajet::cli::run_serve(args, gpus);
}
