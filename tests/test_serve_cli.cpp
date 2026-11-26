// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// cli/serve_cmd + core/engine_factory without starting a server: `vegajet serve` argument parsing (every option,
// ranges, conflicts), the engine registry (CPU-only message, unknown architecture, a registered provider, a failing
// provider), MockBatchEngine reproducing MockEngine's reply token for token through the real request scheduler
// (with and without preemption), and run_serve's early exits: usage error -> 2, unreadable model -> 1, a valid model
// file whose architecture has no engine on a CPU-only build -> 3 (never touching /dev/kfd), bad --tp -> 2.
#include "cli/serve_cmd.hpp"
#include "core/engine_factory.hpp"
#include "core/request_scheduler.hpp"
#include "tests/test_common.hpp"

#include <cstdio>

using namespace vegajet;

namespace {

int parse(std::vector<std::string> a, cli::ServeConfig& c) {
    std::string err;
    c = cli::ServeConfig{};
    return cli::parse_serve_args(a, c, &err);
}

std::string mini_gguf() {  // tokenizer-only GGUF with an architecture that has no engine
    vjtest::GgufWriter w;
    w.kv_str("general.architecture", "qwen35moe");
    std::vector<std::string> toks;
    for (int c = 33; c < 127; ++c) toks.push_back(std::string(1, static_cast<char>(c)));
    toks.push_back("<|im_end|>");
    w.kv_str("tokenizer.ggml.model", "gpt2");
    w.kv_arr_str("tokenizer.ggml.tokens", toks);
    w.kv_arr_str("tokenizer.ggml.merges", {});
    w.kv_u32("tokenizer.ggml.eos_token_id", static_cast<uint32_t>(toks.size() - 1));
    return vjtest::write_temp(w.build(), "serve_cli");
}

}  // namespace

int main() {
    // ---- argument parsing
    cli::ServeConfig c;
    VJ_CHECK(parse({"--mock"}, c) == 0 && c.mock && c.slots == 1 && c.port == 8080 && c.host == "127.0.0.1");
    VJ_CHECK(parse({"m-model", "--port", "0", "--host", "0.0.0.0", "--slots", "4", "--max-queue", "8", "--tp", "2", "--ctx", "4096",
                    "--chat-format", "qwen3", "--enable-thinking", "--reasoning-format", "none", "--hsaco-dir", "h", "--model-id", "x"},
                   c) == 0);
    VJ_CHECK(c.model_path == "m-model" && c.port == 0 && c.host == "0.0.0.0" && c.slots == 4 && c.max_queue == 8 && c.tp == 2 &&
             c.ctx == 4096 && c.chat_format == "qwen3" && c.enable_thinking && !c.parse_reasoning && c.hsaco_dir == "h" && c.model_id == "x");
    VJ_CHECK(parse({"--model", "m-model"}, c) == 0 && c.model_path == "m-model");
    VJ_CHECK(parse({"--mock", "--gguf", "t-tokenizer"}, c) == 0 && c.tokenizer_gguf == "t-tokenizer");
    VJ_CHECK(parse({"--help"}, c) == 0 && c.help);
    // S-M4: the max_tokens cap is a real flag (0 or a bad value is a usage error).
    VJ_CHECK(parse({"--mock", "--max-tokens-cap", "512"}, c) == 0 && c.max_tokens_cap == 512);
    VJ_CHECK(parse({"--mock", "--max-tokens-cap", "1048576"}, c) == 0 && c.max_tokens_cap == 1048576);
    // S-M5: an optional API key and one explicit CORS origin (a wildcard is refused).
    VJ_CHECK(parse({"--mock", "--api-key", "k-123", "--cors-origin", "https://app.example"}, c) == 0 &&
             c.api_key == "k-123" && c.cors_origin == "https://app.example");
    const std::vector<std::vector<std::string>> bad = {
        {},                              // neither a model nor --mock
        {"--mock", "m-model"},            // both
        {"a-model", "b-model"},
        {"--mock", "--port", "70000"}, {"--mock", "--port"}, {"--mock", "--port", "x"},
        {"--mock", "--slots", "0"}, {"--mock", "--slots", "65"}, {"--mock", "--tp", "2"},
        {"m", "--tp", "0"}, {"m", "--ctx", "10"}, {"m", "--chat-format", "llama"}, {"m", "--reasoning-format", "raw"},
        {"m", "--max-queue", "0"}, {"m", "--bogus"}, {"m", "--host"},
        {"m", "--max-tokens-cap", "0"}, {"m", "--max-tokens-cap", "x"}, {"m", "--max-tokens-cap"},
        {"m", "--api-key"}, {"m", "--cors-origin"}, {"m", "--cors-origin", "*"},
    };
    for (const auto& b : bad) VJ_CHECK(parse(b, c) == 2);
    // ---- engine registry
    {
        auto& reg = serve::EngineRegistry::instance();
        serve::EngineSpec spec;
        spec.architecture = "nosuch";
        serve::EngineHandle h;
        std::string err;
        VJ_CHECK(!reg.create(spec, h, &err) && !h.engine);
#ifdef VEGAJET_CPU_ONLY
        VJ_CHECK(err.find("VEGAJET_CPU_ONLY=ON") != std::string::npos && err.find("'nosuch'") != std::string::npos);  // CPU-only build
#else
        VJ_CHECK(err.find("no streaming engine is registered") != std::string::npos && err.find("'nosuch'") != std::string::npos);  // GPU build
#endif
        reg.add("fixture-arch", [](const serve::EngineSpec& s, serve::EngineHandle& out, std::string*) {
            auto m = std::make_unique<MockEngine>("fixture");
            out.batch = std::make_unique<serve::MockBatchEngine>(*m);
            out.engine = std::move(m);
            out.description = "fixture tp=" + std::to_string(s.tp);
            return true;
        });
        reg.add("broken-arch", [](const serve::EngineSpec&, serve::EngineHandle&, std::string* e) { *e = "weights missing"; return false; });
        VJ_CHECK(reg.has("fixture-arch") && reg.architectures().size() == 2);
        spec.architecture = "fixture-arch";
        spec.tp = 2;
        VJ_CHECK(reg.create(spec, h, &err) && h.engine && h.batch && h.description == "fixture tp=2");
        spec.architecture = "broken-arch";
        serve::EngineHandle h2;
        VJ_CHECK(!reg.create(spec, h2, &err) && err.find("weights missing") != std::string::npos && !h2.engine);
    }
    // ---- MockBatchEngine == MockEngine, through the scheduler (roomy pool and a pool small enough to preempt)
    {
        MockEngine ref;
        std::vector<int32_t> want;
        GenerationParams gp;
        gp.max_tokens = 64;
        ref.generate({1, 2, 3}, gp, [&](const TokenEvent& ev) {
            if (ev.token_id >= 0 && !ev.is_eos) want.push_back(ev.token_id);
            return !ev.is_eos;
        });
        VJ_CHECK(want.size() == 8);
        for (uint32_t blocks : {256u, 6u}) {
            MockEngine m;
            serve::MockBatchEngine be(m);
            sched::RequestScheduler s(sched::SchedulerConfig{4, 64, 8, blocks, 4, false, 0});
            std::map<uint64_t, std::vector<int32_t>> got;
            std::map<uint64_t, sched::EndReason> end;
            for (int r = 0; r < 6; ++r) s.submit(std::vector<int32_t>(5 + r, 7), 64, be.eos());
            for (int step = 0; !s.idle() && step < 10000; ++step) {
                std::vector<sched::Delivery> d;
                sched::StepPlan p = s.schedule(nullptr, d);
                s.commit(p, be.step(p), d);
                for (const auto& x : d) {
                    got[x.req].insert(got[x.req].end(), x.tokens.begin(), x.tokens.end());
                    if (x.end != sched::EndReason::None) end[x.req] = x.end;
                }
            }
            VJ_CHECK(s.idle() && got.size() == 6);
            for (auto& kv : got) {
                VJ_CHECK(end[kv.first] == sched::EndReason::Eos && kv.second.back() == be.eos());
                kv.second.pop_back();
                VJ_CHECK(kv.second == want);
            }
            if (blocks == 6u) std::printf("  mock batch with a 24-token KV pool: %llu preemptions, replies identical\n",
                                          (unsigned long long)s.stats().preemptions);
        }
    }
    // ---- run_serve early exits (no server is started on these paths)
    {
        gpusel::Selection all;
        VJ_CHECK(cli::run_serve({"--port", "x", "--mock"}, all) == 2);
        VJ_CHECK(cli::run_serve({"/nonexistent/model-file"}, all) == 1);
        const std::string m = mini_gguf();
        VJ_CHECK(cli::run_serve({m, "--port", "0"}, all) == 3);  // CPU-only build: no engine, exit before any device
        gpusel::Selection two;
        std::string err;
        VJ_CHECK(gpusel::parse("0,1", two, &err));
        setenv("VJ_SYSFS_ROOT", "/nonexistent-root", 1);  // no topology: selection size decides
        VJ_CHECK(cli::run_serve({m, "--tp", "1"}, two) == 2);
        unsetenv("VJ_SYSFS_ROOT");
        std::remove(m.c_str());
    }
    std::printf("test_serve_cli: argument parsing, engine registry, MockBatchEngine == MockEngine via the scheduler, "
                "run_serve exit codes OK\nPASS\n");
    return 0;
}
