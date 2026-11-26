// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// `vegajet bench` / `vegajet run`: plan a model (dry-run load through its plugin), build the step graphs,
// validate GPU code objects against them, then hand the graphs to the KFD runtime. CPU-only builds stop
// after planning and validation.
#include "cli/inspect.hpp"
#include "cli/tp_ar_bench.hpp"

#include "core/bench_json.hpp"
#include "core/grid_planner.hpp"
#include "core/kfd_runtime.hpp"
#include "core/portability.hpp"
#include "core/tokenizer.hpp"
#include "core/gpu_select.hpp"
#include "core/tp_collective.hpp"
#include "core/tp_executor.hpp"
#include "models/ornith/ornith_gpu.hpp"
#include "models/ornith/ornith_tp_exec.hpp"
#include "core/placement.hpp"
#include "models/registry.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>

#include "core/fr_spec.hpp"  // VEGAJET_FR_SPEC_ROWS fallback for --fr-spec

namespace vegajet::cli {

namespace {

constexpr uint64_t GiB = 1ull << 30;

// Device profiles for an executor layout, read-only: --tp-profile wins, then the gfx906 sysfs scan,
// then the owner-declared --vram-gib/--bw-gbs defaults. Never opens a device node.
bool executor_profiles(uint32_t tp, const std::string& profile_path, double vram_gib,
                       const std::vector<double>& bws, std::vector<tp::DeviceProfile>& out, std::string* err) {
    if (!profile_path.empty()) {
        std::ifstream in(profile_path);
        if (!in.is_open()) { if (err) *err = "cannot read " + profile_path; return false; }
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        try {
            out = tp::profiles_from_json(text);
        } catch (const std::exception& e) {
            if (err) *err = e.what();
            return false;
        }
        if (out.size() != tp) {
            if (err) *err = "--tp-profile has " + std::to_string(out.size()) + " ranks, --tp is " + std::to_string(tp);
            return false;
        }
        return true;
    }
    auto found = gpusel::scan_topology(gpusel::sysfs_root());
    if (!found.empty() && !gpusel::filter_from_env(found, err)) return false;
    if (!found.empty() && found.size() != tp) {
        if (err) *err = "selected-device count must equal --tp";
        return false;
    }
    if (found.size() == tp) {
        for (uint32_t r = 0; r < tp; ++r) {
            tp::DeviceProfile p;
            p.bdf = gpusel::bdf(found[r].pci_domain, found[r].location_id);
            p.unique_id = tp::device_key(p.bdf, gpusel::sysfs_root());
            if (p.unique_id.rfind("uid:", 0) == 0) p.unique_id = p.unique_id.substr(4);
            else p.unique_id.clear();
            p.bw_hbm_gbs = r < bws.size() ? bws[r] : 1.0;
            p.n_cu = found[r].simd_count ? found[r].simd_count / 4u : 60u;
            p.mclk = 1000;
            p.usable_vram_bytes = found[r].vram_bytes > GiB ? found[r].vram_bytes - GiB : 0;
            out.push_back(std::move(p));
        }
        return true;
    }
    for (uint32_t r = 0; r < tp; ++r) {
        tp::DeviceProfile p;
        p.unique_id = "plan-" + std::to_string(r);  // synthetic: no device was found
        p.bw_hbm_gbs = r < bws.size() ? bws[r] : 1.0;
        p.n_cu = 60;
        p.mclk = 1000;
        p.usable_vram_bytes = static_cast<uint64_t>(vram_gib) * GiB;
        out.push_back(std::move(p));
    }
    return true;
}

#ifndef VEGAJET_GIT_COMMIT
#define VEGAJET_GIT_COMMIT "unknown"
#endif

void usage(const char* cmd) {
    std::printf("Usage: vegajet %s <model-file> [--tp N] [--vram-gib G] [--bw-gbs B[,B...]] [--ctx T]\n"
                "                    [--reuse R] [--draft <draft-file>] [--hsaco <code-object>]... [--hsaco-dir DIR] %s\n"
                "                    [--dry-run] [--n-predict|-n N] [--ar-wire fp16|int8] [--json-out FILE]\n"
                "  Decode flags:     [--[no-]fuse-router] [--[no-]gdn-splitv] [--[no-]fuse-back] [--back-wgs 1..240]\n"
                "                    [--gu-dot8-layers 0..48] [--spec-flash]\n"
                "  Prefill flags:    [--prefill-chunk 128|256|512|1024] [--down-grp-prefill 0|1|2|3] [--down-grp-verify 0|1|2]\n"
                "                    [--pf-fa2] [--pf-fa2-fp16] [--pf-gemm 0|64|128] [--pf-gdn 0|1|2] [--pf-gu-tiled] [--prefill-ar-int8]\n"
                "                    [--prefill-dot8] [--dot8-clip A] [--dot8-scope 0|1|2] [--no-fast-prefill]\n"
                "  Spec flags:       [--suffix-decode] [--suffix-max-draft 1..15] [--spec-gdn] [--zero-tax-catchup]\n"
                "                    [--fr-spec 0|32768|65536] [--mtp-k 1..6] [--gu-grp-verify] [--spec-force-reject]\n"
                "                    [--draft-model F] [--mask-embedding F] [--dflash-k 1..15] [--dflash-p-min P]\n"
                "  Diagnostics:      [--profile-stages] [--pp-stages] [--ar-timeline] [--dot8-ab] [--ab-no-prepush]\n"
                "                    [--dump-hidden FILE] [--dump-logits FILE] [--dump-tokens FILE] [--gdn-state-dump FILE] [--prefill-seq]\n"
                "                    [--interactive] [--interactive-fifo PATH] [--sweep \"VAR_A ; VAR_B\"]\n"
                "                    [--dump-golden-layers DIR]\n"
                "                    | --bench-ar [--tp N] [--dry-run] [--tp-profile FILE] [--tp-plan FILE]\n"
                "                    | --bench-gemv [--hsaco-dir DIR]\n"
                "  N-rank raw:       [--tp-shards FILE] [--tp-plan FILE]; FP16 wire, sequential per-op prefill/decode.\n"
                "  Placement:        [--placement FILE] [--placement-print]; print validates/proposes without execution.\n"
                "  GPU selection:    [--gpu IDX[,IDX|A-B]...|all|list] (CLI-wide; overrides VJ_GPU_DEVICES)\n"
                "  Dry-run load through the model plugin, per-GPU footprint, decode/prefill graph summary and\n"
                "  code-object validation (private_segment_fixed_size == 0, every graph kernel symbol exported).\n"
                "  --json-out FILE writes the vegajet.bench.v1 scorecard (tools/regression_gate.py, tools/scorecard.py);\n"
                "  an existing file for the same model + commit is merged (e.g. a W4A8 run and a --prefill-dot8 run).\n",
                cmd, std::string(cmd) == "run" ? "--prompt TEXT" : "[--prompt TEXT]");
}

} // namespace

int run_bench(const std::string& cmd, const std::vector<std::string>& args) {
    std::string model_path, draft, mask_embd, prompt, hsaco_dir, dump_hidden, dump_logits, json_out, dump_tokens;
    std::string interactive_fifo, sweep_spec, dump_golden_layers_dir, tp_profile, tp_plan, tp_shards, shard_text, collective_text;
    std::string placement_path, placement_text;
    bool placement_print = false;
    bool tp_given = false;
    std::vector<std::string> explicit_flags;
    bool interactive = false;
    std::vector<std::string> hsaco;
    uint32_t tp = 1, ctx = 32768;
    double vram_gib = 32.0, reuse = 1.0;
    bool suffix_decode = false;
    bool fr_spec_cli = false;  // --fr-spec given; otherwise VEGAJET_FR_SPEC_ROWS decides
    bool dry_only = false;
    bool bench_ar = false, bench_gemv = false, ar_int8 = false, fuse_router = true, gdn_splitv = true, fuse_back = true, profile_stages = false, force_reject = false, dot8_ab = false, ar_timeline = false, pp_stages = false, ab_no_prepush = false;
    uint32_t back_wgs = 240, gu_dot8_layers = 0, prefill_chunk = 1024, sfx_max = 11, fr_spec = 0, mtp_k = 4, dflash_k = 7, dgp = 3, dgv = 0, pf_gemm = 128, pf_gdn = 1, dot8_scope = 0;
    float dflash_p_min = 0.55f, dot8_clip = 1.0f;
    bool pf_fa2 = true, pf_fa2_fp16 = true, pf_gu = true, prefill_ar_int8 = true, prefill_dot8 = false, prefill_seq = false, spec_gdn = false, spec_flash = false, zero_tax_catchup = false, gu_grp_verify = false;
    std::string gdn_dump;
    int decode_n = 32;
    std::vector<double> bws;
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (a.rfind("--", 0) == 0) explicit_flags.push_back(a);
        auto next = [&]() -> std::string { return i + 1 < args.size() ? args[++i] : std::string(); };
        if (a == "--help" || a == "-h") { usage(cmd.c_str()); return 0; }
        else if (a == "--tp") {
            const auto value = next();
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), tp);
            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || tp == 0) {
                std::fprintf(stderr, "--tp requires a positive unsigned rank count\n"); return 2;
            }
            tp_given = true;
        }
        else if (a == "--tp-profile" || a == "--tp-plan" || a == "--tp-shards") {
            auto& path = a == "--tp-profile" ? tp_profile : (a == "--tp-plan" ? tp_plan : tp_shards);
            path = next();
            if (path.empty() || path[0] == '-') { std::fprintf(stderr, "%s requires a file\n", a.c_str()); return 2; }
        }
        else if (a == "--vram-gib") vram_gib = std::atof(next().c_str());
        else if (a == "--ctx") ctx = static_cast<uint32_t>(std::atoi(next().c_str()));
        else if (a == "--reuse") reuse = std::atof(next().c_str());
        else if (a == "--draft" || a == "--draft-model") draft = next();
        else if (a == "--mask-embedding") mask_embd = next();
        else if (a == "--dflash-k") { const int v = std::atoi(next().c_str()); if (v < 1 || v > 15) { std::fprintf(stderr, "--dflash-k takes 1..15\n"); return 2; } dflash_k = static_cast<uint32_t>(v); }
        else if (a == "--dflash-p-min") dflash_p_min = static_cast<float>(std::atof(next().c_str()));
        else if (a == "--hsaco") hsaco.push_back(next());
        else if (a == "--hsaco-dir") hsaco_dir = next();
        else if (a == "--suffix-decode") suffix_decode = true;
        else if (a == "--dry-run") dry_only = true;
        else if (a == "--bench-ar") bench_ar = true;
        else if (a == "--bench-gemv") bench_gemv = true;
        else if (a == "--fuse-router") fuse_router = true;
        else if (a == "--gdn-splitv") gdn_splitv = true;
        else if (a == "--no-fuse-router") fuse_router = false;
        else if (a == "--fuse-back") fuse_back = true;
        else if (a == "--no-fuse-back") fuse_back = false;
        else if (a == "--placement") { placement_path = next(); if (placement_path.empty() || placement_path[0] == '-') { std::fprintf(stderr, "--placement needs a file\n"); return 2; } }
        else if (a == "--placement-print") { placement_print = true; dry_only = true; }
        else if (a == "--profile-stages") profile_stages = true;
        else if (a == "--spec-force-reject") force_reject = true;
        else if (a == "--dot8-ab") dot8_ab = true;
        else if (a == "--ar-timeline") ar_timeline = true;
        else if (a == "--pp-stages") pp_stages = true;
        else if (a == "--gdn-state-dump") gdn_dump = next();
        else if (a == "--prefill-seq") prefill_seq = true;
        else if (a == "--pf-fa2") pf_fa2 = true;
        else if (a == "--pf-fa2-fp16") { pf_fa2 = true; pf_fa2_fp16 = true; }
        else if (a == "--pf-gu-tiled") pf_gu = true;
        else if (a == "--prefill-ar-int8") prefill_ar_int8 = true;
        else if (a == "--no-fast-prefill") { prefill_chunk = 128; pf_fa2 = false; pf_fa2_fp16 = false; pf_gemm = 0; pf_gdn = 0; pf_gu = false; dgp = 1; prefill_ar_int8 = false; }
        else if (a == "--ab-no-prepush") ab_no_prepush = true;
        else if (a == "--prefill-dot8") prefill_dot8 = true;
        else if (a == "--dot8-scope") dot8_scope = static_cast<uint32_t>(std::atoi(next().c_str()));
        else if (a == "--dot8-clip") dot8_clip = static_cast<float>(std::atof(next().c_str()));
        else if (a == "--spec-gdn") spec_gdn = true;
        else if (a == "--spec-flash") spec_flash = true;
        else if (a == "--zero-tax-catchup") zero_tax_catchup = true;
        else if (a == "--gu-grp-verify") gu_grp_verify = true;
        else if (a == "--fr-spec") { const int v = std::atoi(next().c_str()); if (v < 0 || (v != 0 && (v % 512 != 0 || v > 131072))) { std::fprintf(stderr, "--fr-spec takes 0 or a multiple of 512 up to 131072 (e.g. 32768, 65536)\n"); return 2; } fr_spec = static_cast<uint32_t>(v); fr_spec_cli = true; }
        else if (a == "--mtp-k") { const int v = std::atoi(next().c_str()); if (v < 1 || v > 6) { std::fprintf(stderr, "--mtp-k takes 1..6\n"); return 2; } mtp_k = static_cast<uint32_t>(v); }
        else if (a == "--prefill-chunk") { const int v = std::atoi(next().c_str()); if (v != 128 && v != 256 && v != 512 && v != 1024) { std::fprintf(stderr, "--prefill-chunk takes 128, 256, 512 or 1024\n"); return 2; } prefill_chunk = static_cast<uint32_t>(v); }
        else if (a == "--gu-dot8-layers") { const int v = std::atoi(next().c_str()); if (v < 0 || v > 48) { std::fprintf(stderr, "--gu-dot8-layers takes 0..48\n"); return 2; } gu_dot8_layers = static_cast<uint32_t>(v); }
        else if (a == "--pf-gemm") { const int v = std::atoi(next().c_str()); if (v != 0 && v != 64 && v != 128) { std::fprintf(stderr, "--pf-gemm takes 0, 64 or 128\n"); return 2; } pf_gemm = static_cast<uint32_t>(v); }
        else if (a == "--pf-gdn") { const int v = std::atoi(next().c_str()); if (v < 0 || v > 2) { std::fprintf(stderr, "--pf-gdn takes 0, 1 or 2\n"); return 2; } pf_gdn = static_cast<uint32_t>(v); }
        else if (a == "--down-grp-prefill" || a == "--down-grp-verify") {
            const int v = std::atoi(next().c_str());
            const int vmax = (a == "--down-grp-prefill") ? 3 : 2;
            if (v < 0 || v > vmax) { std::fprintf(stderr, "%s takes 0 (per-token), 1 (grouped exact), 2 (grouped fast)%s\n", a.c_str(), vmax == 3 ? " or 3 (LDS-tiled)" : ""); return 2; }
            (a == "--down-grp-prefill" ? dgp : dgv) = static_cast<uint32_t>(v);
        }
        else if (a == "--suffix-max-draft") { const int v = std::atoi(next().c_str()); if (v < 1 || v > 15) { std::fprintf(stderr, "--suffix-max-draft takes 1..15\n"); return 2; } sfx_max = static_cast<uint32_t>(v); }
        else if (a == "--dump-hidden") dump_hidden = next();
        else if (a == "--dump-logits") dump_logits = next();
        else if (a == "--dump-tokens") { dump_tokens = next(); if (dump_tokens.empty() || dump_tokens[0] == '-') { std::fprintf(stderr, "--dump-tokens needs a file\n"); return 2; } }
        else if (a == "--interactive") interactive = true;
        else if (a == "--interactive-fifo") { interactive = true; interactive_fifo = next(); }
        else if (a == "--sweep") sweep_spec = next();
        else if (a == "--dump-golden-layers") dump_golden_layers_dir = next();
        else if (a == "--back-wgs") { const int v = std::atoi(next().c_str()); if (v < 1 || v > 240) { std::fprintf(stderr, "--back-wgs takes 1..240\n"); return 2; } back_wgs = static_cast<uint32_t>(v); }
        else if (a == "--no-gdn-splitv") gdn_splitv = false;
        else if (a == "--ar-wire") { const std::string w = next(); if (w != "int8" && w != "fp16") { std::fprintf(stderr, "--ar-wire takes fp16|int8\n"); return 2; } ar_int8 = (w == "int8"); }
        else if (a == "--n-predict" || a == "-n") decode_n = std::atoi(next().c_str());
        else if (a == "--prompt") prompt = next();
        else if (a == "--json-out") { json_out = next(); if (json_out.empty() || json_out[0] == '-') { std::fprintf(stderr, "--json-out needs a file\n"); return 2; } }
        else if (a == "--bw-gbs") {
            std::string s = next();
            for (size_t p = 0; p < s.size();) {
                size_t q = s.find(',', p);
                bws.push_back(std::atof(s.substr(p, q - p).c_str()));
                p = q == std::string::npos ? s.size() : q + 1;
            }
        } else if (!a.empty() && a[0] == '-') { std::fprintf(stderr, "%s: unknown option %s\n", cmd.c_str(), a.c_str()); return 2; }
        else model_path = a;
    }
    if (!json_out.empty() && (interactive || !sweep_spec.empty())) {
        std::fprintf(stderr, "--json-out records one configuration: not with --interactive / --interactive-fifo / --sweep "
                             "(run one bench --json-out per variant)\n");
        return 2;
    }
    if (!dump_tokens.empty() && (interactive || !sweep_spec.empty())) {
        std::fprintf(stderr, "--dump-tokens records one token sequence: not with --interactive / --interactive-fifo / --sweep "
                             "(run one bench --dump-tokens per variant)\n");
        return 2;
    }
    if (tp > 256) { std::fprintf(stderr, "--tp exceeds the 256-rank runtime bound\n"); return 2; }
    if (!bench_ar && !placement_print && !tp_profile.empty()) {
        std::fprintf(stderr, "--tp-profile requires --bench-ar; decode uses selected-device identities and --tp-shards\n"); return 2;
    }
    if (bench_ar && !tp_shards.empty()) { std::fprintf(stderr, "--tp-shards is for decode, not --bench-ar\n"); return 2; }
    try {
        auto read_json = [](const std::string& path) {
            std::ifstream input(path);
            if (!input) throw std::runtime_error("cannot read TP JSON: " + path);
            return std::string(std::istreambuf_iterator<char>(input), {});
        };
        if (bench_ar && (!placement_path.empty() || placement_print)) throw std::runtime_error("placement is not supported by --bench-ar");
        if (!placement_path.empty()) {
            placement_text = read_json(placement_path);
            const auto config = placement::from_json(placement_text);
            if (!placement_print) {
                const auto why = placement::executor_error(config);
                if (!why.empty()) throw std::runtime_error(why);
                const auto ranks = config.tp_groups.at(0).devices.size();
                if (tp_given && tp != ranks) throw std::runtime_error("placement TP-group size conflicts with --tp");
                tp = static_cast<uint32_t>(ranks);
            }
        }
        if (!tp_shards.empty()) { shard_text = read_json(tp_shards); (void)tp::shard_options_from_json(shard_text); }
        if (!tp_plan.empty()) { collective_text = read_json(tp_plan); (void)tp::collective_options_from_json(collective_text); }
        if (!bench_ar && placement_text.empty() && tp == 2 && (!tp_shards.empty() || !tp_plan.empty()))
            throw std::runtime_error("TP=2 legacy fast path does not support --tp-shards/--tp-plan overrides");
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 2; }
    if (bench_ar) {
        if (!tp_given) tp = 2; // preserve the legacy standalone benchmark default
        if (tp > 256) { std::fprintf(stderr, "--bench-ar plan CLI admits at most 256 ranks (matrix resource bound)\n"); return 2; }
        if (ar_int8) { std::fprintf(stderr, "--bench-ar M2 planning supports FP16 wire only\n"); return 2; }
        tp::CollectiveOptions runtime_options;
        try {
            auto read = [](const std::string& path) {
                std::ifstream input(path);
                if (!input) throw std::invalid_argument("cannot read plan/profile: " + path);
                std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
                if (input.bad()) throw std::invalid_argument("failed reading plan/profile: " + path);
                return text;
            };
            tp::PeerMatrix peers(tp, std::vector<tp::PeerLink>(tp));
            if (!tp_profile.empty()) {
                peers = tp::peer_matrix_from_json(read(tp_profile));
                if (peers.size() != tp) throw std::invalid_argument("--tp does not match hardware profile rank count");
            } else std::printf("No peer profile: P2P capability unknown; all remote plan edges use host-bounce.\n");
            const auto options = tp_plan.empty() ? tp::CollectiveOptions{} : tp::collective_options_from_json(read(tp_plan));
            runtime_options = options;
            const auto topology = tp::plan_topology(peers, options);
            const auto collective = tp::plan_collective(3072, peers, options);
            std::printf("%s", tp::format_collective_plan(collective, topology).c_str());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "--bench-ar plan: %s\n", e.what()); return 2;
        }
        if (dry_only) return 0;
        if (!kfd::Runtime::compiled_with_gpu()) { std::fprintf(stderr, "--bench-ar: this build has no GPU backend\n"); return 3; }
        if (!tp_profile.empty()) std::printf("Profile is advisory; execution re-reads actual sysfs capabilities, never trusts claimed peer edges.\n");
        return run_tp_ar_bench(tp, hsaco_dir, runtime_options);
    }
    if (bench_gemv) {
        if (!kfd::Runtime::compiled_with_gpu()) { std::fprintf(stderr, "--bench-gemv: this build has no GPU backend\n"); return 3; }
        return ornith::run_ornith_gemv_ab_bench(hsaco_dir);
    }
    if (model_path.empty() || tp == 0) { usage(cmd.c_str()); return 2; }

    GgufFile f;
    if (!f.open(model_path)) { std::fprintf(stderr, "%s\n", f.error().c_str()); return 1; }
    std::string err;
    auto model = ModelRegistry::instance().create_for(f, &err);
    if (!model) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    // The Ornith GPU runner's live default is its two-rank fast path; --tp overrides it. Other plugins
    // keep the single-device planning default.
    if (!tp_given && placement_text.empty() && !placement_print && model->name() == "ornith") tp = 2;
    LoadOptions opt;
    opt.gpus.assign(tp, GpuTarget{60, 800.0, 1.725, static_cast<uint64_t>(vram_gib * GiB)});
    for (size_t i = 0; i < bws.size() && i < tp; ++i) opt.gpus[i].bw_hbm_gbs = bws[i];
    opt.context_length = ctx;
    opt.batch_reuse_factor = reuse;
    opt.draft_gguf = draft;
    const LoadReport rep = model->load(f, opt);
    if (!placement_text.empty() || placement_print) {
        try {
            if (!rep.ok) throw std::runtime_error(rep.error);
            const auto* om = dynamic_cast<const ornith::OrnithModel*>(model.get());
            if (!om) throw std::runtime_error("placement capacity adapter unsupported for this model plugin");
            auto devices = placement::selected_profiles();
            tp::PeerMatrix peers;
            if (!tp_profile.empty()) {
                std::ifstream input(tp_profile);
                if (!input) throw std::runtime_error("cannot read placement profile");
                const std::string text(std::istreambuf_iterator<char>(input), {});
                devices = tp::profiles_from_json(text);
                peers = tp::peer_matrix_from_json(text);
            } else if (placement_text.empty()) peers = placement::selected_peers(devices);
            const auto shard_options = shard_text.empty() ? tp::ShardOptions{} : tp::shard_options_from_json(shard_text);
            const auto capacity = [&](const auto& ranks, uint32_t begin, uint32_t end) {
                auto h = om->hparams();
                h.recurrent = std::vector<bool>(h.recurrent.begin() + begin, h.recurrent.begin() + end);
                h.n_trunk = end - begin;
                h.n_gdn = static_cast<uint32_t>(std::count(h.recurrent.begin(), h.recurrent.end(), true));
                h.n_attn = h.n_trunk - h.n_gdn;
                (void)ornith::raw_layout(h, ctx, ranks, shard_options);
            };
            const auto config = placement_text.empty()
                ? placement::propose(devices, peers, om->hparams().n_trunk, capacity,
                    collective_text.empty() ? tp::CollectiveOptions{} : tp::collective_options_from_json(collective_text))
                : placement::from_json(placement_text);
            placement::validate(config, devices, om->hparams().n_trunk, capacity);
            if (placement_print) { std::printf("%s\n", placement::to_json(config).c_str()); return 0; }
            placement::bind_executor(config, tp);
        } catch (const std::exception& e) { std::fprintf(stderr, "--placement: %s\n", e.what()); return 2; }
    }
    std::printf("== %s plan: plugin=%s arch=%s TP=%u ctx=%u reuse=%.1f -> %s\n", cmd.c_str(), model->name().c_str(),
                rep.architecture.c_str(), tp, ctx, reuse, rep.ok ? "fits" : rep.error.c_str());
    for (size_t i = 0; i < rep.weight_bytes_per_gpu.size(); ++i)
        std::printf("  gpu%zu weights %.2f GiB | KV %.2f GiB | state %.3f GiB\n", i, rep.weight_bytes_per_gpu[i] / double(GiB),
                    i < rep.kv_bytes_per_gpu.size() ? rep.kv_bytes_per_gpu[i] / double(GiB) : 0.0,
                    i < rep.state_bytes_per_gpu.size() ? rep.state_bytes_per_gpu[i] / double(GiB) : 0.0);
    for (const auto& n : rep.notes) std::printf("  note: %s\n", n.c_str());

    // --json-out: vegajet.bench.v1 scorecard (core/bench_json). Written on every exit path below.
    benchjson::Record rec;
    rec.command = dry_only ? "plan" : cmd;
    rec.commit = VEGAJET_GIT_COMMIT;
    rec.model = model_path.substr(model_path.find_last_of('/') == std::string::npos ? 0 : model_path.find_last_of('/') + 1);
    rec.architecture = rep.architecture;
    rec.tp = tp;
    rec.flags = benchjson::sanitize_flags(args);
    if (const char* sel = std::getenv("VJ_GPU_DEVICES")) rec.gpu_selection = sel;
    rec.prefill_path = prefill_dot8 ? "dot8" : "w4a8";
    rec.plan_status = rep.ok ? "fits" : rep.error;
    rec.timestamp_utc = benchjson::utc_now();
    auto finish = [&](int code) {
        if (json_out.empty()) return code;
        std::string jerr;
        bool merged = false;
        if (!benchjson::write_file(json_out, rec, &jerr, &merged)) {
            std::fprintf(stderr, "--json-out: %s\n", jerr.c_str());
            return code ? code : 1;
        }
        std::printf("  bench json: %s (%s)\n", json_out.c_str(), merged ? "merged" : "written");
        return code;
    };

    std::vector<int32_t> prompt_ids;
    if (!prompt.empty()) {
        BpeTokenizer tok;
        if (!tok.load_from_gguf(model_path, &err)) { std::fprintf(stderr, "tokenizer: %s\n", err.c_str()); return finish(1); }
        prompt_ids = tok.encode(prompt, false);
        std::printf("  prompt: %zu tokens\n", prompt_ids.size());
    }
    if (prompt_ids.empty()) prompt_ids.assign(512, 0);
    const auto pf = model->prefill(prompt_ids);
    const auto dec = model->decode_step(prompt_ids.back(), 1);
    std::map<std::string, uint32_t> hist;
    std::set<std::string> symbols;
    for (const auto* g : {&pf.graph, &dec.graph})
        for (const auto& d : *g) { hist[ops::op_name(d.kind)]++; symbols.insert(ops::kernel_symbol(d.kind)); }
    std::printf("  graphs: prefill %zu ops, decode %zu ops, %zu distinct kernels\n", pf.graph.size(), dec.graph.size(), symbols.size());
    for (const auto& [k, v] : hist) std::printf("    %-20s %u\n", k.c_str(), v);

    int rc = rep.ok ? 0 : 1;

    // v1.3 M4: N-rank executor admission and layout. TP=2 is the fused fast path; every other rank
    // count plans the generic per-op path and refuses, by name, any feature that path cannot own.
    if (const auto* om = dynamic_cast<const ornith::OrnithModel*>(model.get())) {
        ornith::OrnithGpuOptions feat;
        feat.draft_gguf      = draft;
        feat.use_suffix_spec = suffix_decode;
        feat.ar_wire_int8    = ar_int8;
        feat.prefill_dot8    = prefill_dot8;
        feat.spec_gdn        = spec_gdn;
        const std::string why = ornith::ornith_admit(tp, ornith::ornith_feature_flags(feat));
        if (!why.empty()) { std::fprintf(stderr, "[vegajet] %s\n", why.c_str()); return finish(2); }
        std::printf("  executor: %s\n", ornith::ornith_executor_mode(tp) == ornith::ExecutorMode::LegacyFast
                                             ? "TP=2 fused fast path" : "generic N-rank per-op path");
        std::vector<tp::DeviceProfile> devs;
        std::string derr;
        if (!executor_profiles(tp, tp_profile, vram_gib, bws, devs, &derr)) {
            std::printf("  executor layout: %s\n", derr.c_str());
            rc = 1;
        } else {
            try {
                if (tp != 2 || !placement_text.empty()) {
                    const auto layout = ornith::raw_layout(om->hparams(), ctx, devs,
                        shard_text.empty() ? tp::ShardOptions{} : tp::shard_options_from_json(shard_text));
                    for (size_t r = 0; r < layout.ranks.size(); ++r) {
                        const auto& m = layout.ranks[r];
                        std::printf("  rank %zu: FF [%llu,%llu), shared [%llu,%llu), vocab [%llu,%llu), Q=%zu KV=%zu GDN=%zu; capacity bound=%llu B\n",
                            r, (unsigned long long)m.ff.begin, (unsigned long long)m.ff.end,
                            (unsigned long long)m.shared_ff.begin, (unsigned long long)m.shared_ff.end,
                            (unsigned long long)m.vocab.begin, (unsigned long long)m.vocab.end,
                            m.attn.q_heads.size(), m.attn.kv_heads.size(), m.groups.size(),
                            (unsigned long long)layout.shards.ranks[r].used_bytes);
                    }
                }
            } catch (const std::exception& e) {
                std::printf("  executor layout: %s\n", e.what());
                rc = 1;
            }
        }
    }

    if (!hsaco.empty()) {
        std::set<std::string> exported;
        for (const auto& path : hsaco) {
            const auto co = planner::parse_code_object_file(path);
            std::string why;
            const bool ok = planner::validate_no_scratch(co, &why);
            std::printf("  code object %s: %s, %zu kernels, %s\n", path.c_str(), co.ok ? co.target.c_str() : co.error.c_str(),
                        co.kernels.size(), ok ? "no scratch" : why.c_str());
            if (!ok) rc = 1;
            for (const auto& k : co.kernels) {
                exported.insert(k.name);
                rec.spill_bytes_max = std::max<uint64_t>(rec.spill_bytes_max.value_or(0), k.private_segment_fixed_size);
            }
        }
        for (const auto& s : symbols)
            if (!exported.count(s)) { std::printf("  MISSING kernel symbol: %s\n", s.c_str()); rc = 1; }
    }

    if (kfd::Runtime::compiled_with_gpu() && !dry_only && model->name() == "ornith") {
        if (ornith::ornith_executor_mode(tp) == ornith::ExecutorMode::Generic)
            std::fprintf(stderr, "[vegajet] TP=%u: generic N-rank executor (admitted above)\n", tp);
        // Alpha portability: refuse before the runner reserves its per-GPU arenas, with what is needed vs found,
        // instead of failing late inside the per-GPU ~30 GiB allocation on a smaller host.
        {
            portability::FitRequest fr;
            const size_t slash = model_path.find_last_of('/');
            fr.model = slash == std::string::npos ? model_path : model_path.substr(slash + 1);
            fr.gpus_required = tp;
            fr.arena_bytes_per_gpu = 30ULL * portability::kGiB;          // rt.init(30.5), falling back to 30.0 GiB
            if (rep.weight_bytes_per_gpu.size() == fr.gpus_required)    // the plan describes the runner's layout
                for (size_t i = 0; i < rep.weight_bytes_per_gpu.size(); ++i)
                    fr.bytes_per_gpu.push_back(rep.weight_bytes_per_gpu[i] +
                                               (i < rep.kv_bytes_per_gpu.size() ? rep.kv_bytes_per_gpu[i] : 0) +
                                               (i < rep.state_bytes_per_gpu.size() ? rep.state_bytes_per_gpu[i] : 0));
            const auto fit = portability::preflight(fr, gpusel::sysfs_root());
            if (!fit.ok) {
                std::fprintf(stderr, "[vegajet] %s\n", fit.message.c_str());
                return 1;
            }
        }
        ornith::OrnithGpuOptions gopt;
        gopt.hsaco_dir       = hsaco_dir;
        gopt.draft_gguf      = draft;
        gopt.mask_embedding_path = mask_embd;
        gopt.dflash_k        = dflash_k;
        gopt.dflash_p_min    = dflash_p_min;
        gopt.max_ctx        = tp == 2 ? std::min<uint32_t>(ctx, 8192u) : ctx;
        gopt.prefill_chunk   = prefill_chunk;
        gopt.use_suffix_spec = suffix_decode;
        gopt.run_bench       = (cmd == "bench");
        gopt.run_parity      = (cmd == "bench");
        gopt.decode_tokens   = decode_n;
        gopt.ar_wire_int8    = ar_int8;
        gopt.fuse_moe_router = fuse_router;
        gopt.gdn_splitv      = gdn_splitv;
        gopt.fuse_moe_back   = fuse_back;
        gopt.gu_dot8_layers  = gu_dot8_layers;
        gopt.profile_stages  = profile_stages;
        gopt.spec_force_reject = force_reject;
        gopt.spec_gdn        = spec_gdn;
        gopt.spec_flash      = spec_flash;
        gopt.zero_tax_catchup = zero_tax_catchup;
        if (!fr_spec_cli) {
            // 0 / unset = off; 32k / 64k presets; other sizes are rounded up to whole 256-row blocks per GPU.
            std::string note;
            fr_spec = fr_spec::rows_from_env(fr_spec::VocabSplit{131072u, 2u}, fr_spec::Balance::kEqualPerRank, &note);
            if (!note.empty()) std::fprintf(stderr, "[fr_spec] %s\n", note.c_str());
        }
        gopt.fr_spec_rows    = fr_spec;
        gopt.tp              = tp;
        gopt.tp_shard_json = shard_text; gopt.tp_collective_json = collective_text;
        gopt.placement_json = placement_text;
        gopt.tp_bandwidths = bws;
        gopt.mtp_k           = mtp_k;
        gopt.gu_grp_verify   = gu_grp_verify;
        gopt.dot8_ab = dot8_ab;
        gopt.ar_timeline = ar_timeline;
        gopt.pp_stages = pp_stages;
        gopt.gdn_state_dump = gdn_dump;
        gopt.prefill_seq = prefill_seq;
        gopt.pf_fa2 = pf_fa2;
        gopt.pf_fa2_fp16 = pf_fa2_fp16;
        gopt.pf_gemm = pf_gemm;
        gopt.pf_gdn = pf_gdn;
        gopt.pf_gu_tiled = pf_gu;
        gopt.prefill_ar_int8 = prefill_ar_int8;
        gopt.ab_no_prepush   = ab_no_prepush;
        gopt.prefill_dot8 = prefill_dot8;
        gopt.dot8_scope = dot8_scope;
        gopt.dot8_clip_alpha = dot8_clip;
        gopt.dump_logits_path = dump_logits;
        gopt.dump_tokens_path = dump_tokens;
        gopt.down_grp_prefill = dgp;
        gopt.down_grp_verify = dgv;
        gopt.suffix_max_draft = sfx_max;
        gopt.dump_hidden_path = dump_hidden;
        gopt.interactive = interactive;
        gopt.interactive_fifo = interactive_fifo;
        gopt.sweep_spec = sweep_spec;
        gopt.dump_golden_layers_dir = dump_golden_layers_dir;
        if (!prompt.empty()) gopt.dump_prompt_ids = prompt_ids;
        gopt.moe_back_wgs    = back_wgs;
        if (tp != 2 || !placement_text.empty()) {
            const std::set<std::string> per_op = {"--placement", "--placement-print", "--tp", "--tp-shards", "--tp-plan", "--tp-profile", "--ctx", "--vram-gib", "--bw-gbs",
                "--prompt", "--n-predict", "--hsaco-dir", "--hsaco", "--json-out", "--dry-run", "--ar-wire", "--reuse",
                "--prefill-seq", "--no-fast-prefill", "--no-fuse-router", "--no-fuse-back", "--no-gdn-splitv", "--ab-no-prepush", "--profile-stages"};
            for (const auto& flag : explicit_flags) if (!per_op.count(flag)) {
                std::fprintf(stderr, "generic Ornith: unsupported feature %s (TP=%u)\n", flag.c_str(), tp); return finish(2);
            }
        }
        const std::string raw_error = ornith::ornith_raw_admit(gopt);
        if (!raw_error.empty()) { std::fprintf(stderr, "%s\n", raw_error.c_str()); return finish(2); }
        benchjson::StdoutTee tee;
        if (!json_out.empty()) {
            rec.gpus_before = benchjson::snapshot("/");  // read-only sysfs: TMem, power cap, 6-block RAS
            tee.begin();
        }
        const auto m = ornith::run_ornith_gpu(model_path, prompt, gopt);
        if (!json_out.empty()) {
            benchjson::parse_log(tee.end(), rec);
            rec.gpus_after = benchjson::snapshot("/");
            rec.executed = m.ok;
            rec.error = m.error;
            if (!rec.raw_tok_s && m.tg128_raw_tps > 0) rec.raw_tok_s = m.tg128_raw_tps;
            const std::string path = prefill_dot8 ? "dot8" : "w4a8";
            if (!rec.prefill.count("pp512_" + path) && m.pp512_tps > 0) rec.prefill["pp512_" + path] = m.pp512_tps;
            if (!rec.prefill.count("pp2048_" + path) && m.pp2048_tps > 0) rec.prefill["pp2048_" + path] = m.pp2048_tps;
        }
        if (!m.ok) {
            std::fprintf(stderr, "[ornith::gpu] ERROR: %s\n", m.error.c_str());
            return finish(3);
        }
        return finish(0);
    }

    // --dry-run (and `vegajet plan`) stops after planning and validation: never open /dev/kfd.
    if (dry_only) return finish(rc);

    // Only the Ornith runner executes on gfx906 in this alpha. Without this guard a GPU build opened the runtime for
    // any other plugin, "executed" the graph through symbol-name checks that dispatch nothing, and exited 0.
    if (kfd::Runtime::compiled_with_gpu()) {
        std::fprintf(stderr,
                     "[vegajet] no GPU executor for plugin \"%s\": this alpha runs only the Ornith-122B GPU runner "
                     "(use --dry-run or `vegajet plan` to inspect other models)\n",
                     model->name().c_str());
        return finish(3);
    }

    kfd::Runtime rt;
    if (!rt.open(&err)) {
        std::printf("  execution: %s\n", err.c_str());
        return finish(kfd::Runtime::compiled_with_gpu() ? 3 : rc);
    }
    for (const auto& path : hsaco)
        if (!rt.load_code_object(path, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return finish(3); }
    if (!rt.execute(pf.graph, &err) || !rt.execute(dec.graph, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return finish(3); }
    rec.executed = true;
    return finish(rc);
}

} // namespace vegajet::cli
