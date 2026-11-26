// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Unit tests for warm bench interactive protocol, sweep spec parsing, env scoping, and hot-reload lifecycle.
#include "core/kfd_runtime.hpp"
#include "core/warm_bench.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define VJ_ASSERT(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        std::exit(1); \
    } \
} while (0)

using namespace vegajet::warm;

static void test_command_parsing() {
    auto c0 = parse_warm_command("   # comment only");
    VJ_ASSERT(c0.kind == CommandKind::Empty);

    auto c1 = parse_warm_command("reload");
    VJ_ASSERT(c1.kind == CommandKind::Reload);
    VJ_ASSERT(c1.module_name.empty());

    auto c2 = parse_warm_command("reload vj_ornith_moe_m1");
    VJ_ASSERT(c2.kind == CommandKind::Reload);
    VJ_ASSERT(c2.module_name == "vj_ornith_moe_m1");

    auto c3 = parse_warm_command("reload all");
    VJ_ASSERT(c3.kind == CommandKind::Reload);
    VJ_ASSERT(c3.module_name.empty());

    auto c4 = parse_warm_command("run VEGAJET_MOE_LRU=32 VEGAJET_EVICT_ADAPT=1 --gu-grp-verify --down-grp-verify 1");
    VJ_ASSERT(c4.kind == CommandKind::Run);
    std::vector<std::string> cli;
    std::vector<EnvAssignment> envs;
    extract_env_overrides(c4.args, cli, envs);
    VJ_ASSERT(envs.size() == 2);
    VJ_ASSERT(envs[0].key == "VEGAJET_MOE_LRU" && envs[0].value == "32");
    VJ_ASSERT(envs[1].key == "VEGAJET_EVICT_ADAPT" && envs[1].value == "1");
    VJ_ASSERT(cli.size() == 3);
    VJ_ASSERT(cli[0] == "--gu-grp-verify" && cli[1] == "--down-grp-verify" && cli[2] == "1");

    auto c5 = parse_warm_command("parity --ar-wire fp16");
    VJ_ASSERT(c5.kind == CommandKind::Parity);
    VJ_ASSERT(c5.args.size() == 2);

    auto c6 = parse_warm_command("sweep --gu-grp-verify --down-grp-verify 1 ; VEGAJET_MOE_LRU=32 VEGAJET_EVICT_ADAPT=1");
    VJ_ASSERT(c6.kind == CommandKind::Sweep);
    VJ_ASSERT(c6.variants.size() == 2);
    VJ_ASSERT(c6.variants[0].size() == 3);
    VJ_ASSERT(c6.variants[1].size() == 2);

    auto c7 = parse_warm_command("quit");
    VJ_ASSERT(c7.kind == CommandKind::Quit);

    auto c8 = parse_warm_command("run \"unterminated");
    VJ_ASSERT(c8.kind == CommandKind::Invalid);

    auto c9 = parse_warm_command("bogus_op");
    VJ_ASSERT(c9.kind == CommandKind::Invalid);
}

static void test_scoped_env_overrides() {
    ::setenv("VJ_TEST_WARM_EXISTING", "orig", 1);
    ::unsetenv("VJ_TEST_WARM_NEW");
    {
        ScopedEnvOverrides guard({{"VJ_TEST_WARM_EXISTING", "override"}, {"VJ_TEST_WARM_NEW", "42"}});
        const char* v1 = std::getenv("VJ_TEST_WARM_EXISTING");
        const char* v2 = std::getenv("VJ_TEST_WARM_NEW");
        VJ_ASSERT(v1 != nullptr && std::string(v1) == "override");
        VJ_ASSERT(v2 != nullptr && std::string(v2) == "42");
    }
    const char* v1 = std::getenv("VJ_TEST_WARM_EXISTING");
    const char* v2 = std::getenv("VJ_TEST_WARM_NEW");
    VJ_ASSERT(v1 != nullptr && std::string(v1) == "orig");
    VJ_ASSERT(v2 == nullptr);
    ::unsetenv("VJ_TEST_WARM_EXISTING");
}

static void test_sweep_formatting() {
    std::vector<SweepVariantResult> rs(2);
    rs[0].label = "--gu-grp-verify --down-grp-verify 1";
    rs[0].ok = true;
    rs[0].parity_ok = true;
    rs[0].primary_tps = 200.0;
    rs[0].raw_tg128_tps = 135.0;
    rs[0].pp512_tps = 1250.0;
    rs[0].tmem_max_c = 45.0;
    rs[0].lossless_matched = 129;
    rs[0].lossless_total = 129;

    rs[1] = rs[0];
    rs[1].label = "VEGAJET_MOE_LRU=32 VEGAJET_EVICT_ADAPT=1";
    rs[1].primary_tps = 212.0;

    const std::string tbl = format_sweep_table(rs);
    VJ_ASSERT(tbl.find("+6.00%") != std::string::npos);
    VJ_ASSERT(tbl.find("129/129") != std::string::npos);

    const std::string js = format_sweep_json(rs);
    VJ_ASSERT(js.find("\"delta_pct\":6.00") != std::string::npos);
    VJ_ASSERT(js.find("\"ecc_ue\":0") != std::string::npos);
}

static void test_hot_reload_lifecycle_invariants() {
    vegajet::kfd::Runtime rt;
    VJ_ASSERT(rt.reload_generation() == 0);
    VJ_ASSERT(rt.loaded_module_names().empty());
    std::string err;
    // Reloading before open() must fail cleanly without crashing.
    VJ_ASSERT(!rt.reload_hsaco("vj_ornith_moe_m1", &err));
    VJ_ASSERT(!err.empty());
    err.clear();
    VJ_ASSERT(!rt.reload_all_hsacos(&err));
    VJ_ASSERT(!err.empty());
}

static void test_long_context_decode_prompt() {
    const std::vector<int32_t> seed{7, 8, 9};
    std::vector<int32_t> prompt;
    std::string err;
    VJ_ASSERT(make_decode_prompt("", seed, 32768, 128, prompt, &err));
    VJ_ASSERT(prompt == seed);
    VJ_ASSERT(make_decode_prompt("0", seed, 32768, 128, prompt, &err));
    VJ_ASSERT(prompt == seed);
    VJ_ASSERT(make_decode_prompt("16384", seed, 32768, 128, prompt, &err));
    VJ_ASSERT(prompt.size() == 16384 && prompt.back() == 7);
    for (size_t i = 0; i < prompt.size(); ++i) VJ_ASSERT(prompt[i] == seed[i % seed.size()]);
    VJ_ASSERT(make_decode_prompt("32640", seed, 32768, 128, prompt, &err));
    VJ_ASSERT(prompt.size() == 32640);
    VJ_ASSERT(!make_decode_prompt("32641", seed, 32768, 128, prompt, &err));
    VJ_ASSERT(!err.empty() && prompt.empty());
    VJ_ASSERT(!make_decode_prompt("32768", seed, 32768, 128, prompt, &err));
    for (const char* bad : {"-1", "1x", " 16", "1.0", "4294967296", "999999999999999999999"})
        VJ_ASSERT(!make_decode_prompt(bad, seed, 32768, 128, prompt, &err));
    VJ_ASSERT(!make_decode_prompt("16", {}, 32768, 128, prompt, &err));
    VJ_ASSERT(!make_decode_prompt("16", seed, 32768, 0, prompt, &err));
    VJ_ASSERT(make_decode_prompt("16", seed, 32768, 128, prompt, &err));
    VJ_ASSERT(err.empty() && prompt.size() == 16);
}

int main() {
    test_long_context_decode_prompt();
    test_command_parsing();
    test_scoped_env_overrides();
    test_sweep_formatting();
    test_hot_reload_lifecycle_invariants();
    std::printf("test_warm_bench_protocol: PASS\n");
    return 0;
}
