// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "cli/serve_cmd.hpp"

#include "core/bpe_tokenizer.hpp"
#include "core/engine_factory.hpp"
#include "core/metrics.hpp"
#include "core/request_scheduler.hpp"
#include "core/server.hpp"
#include "core/server_engine_adapter.hpp"
#include "core/tool_call_parser.hpp"

#include <algorithm>
#include <csignal>
#include <cstdlib>
#include <memory>

namespace vegajet::cli {

namespace {

serve::OpenAiServer* g_server = nullptr;
void on_signal(int) {
    if (g_server) g_server->request_stop();
}

bool parse_uint(const std::string& s, uint64_t lo, uint64_t hi, uint64_t& out) {
    if (s.empty() || s.size() > 10 || s.find_first_not_of("0123456789") != std::string::npos) return false;
    const uint64_t v = std::strtoull(s.c_str(), nullptr, 10);
    if (v < lo || v > hi) return false;
    out = v;
    return true;
}

bool chat_format_from(const std::string& name, bpe::ChatFormat& f) {
    if (name == "chatml") f = bpe::ChatFormat::ChatML;
    else if (name == "qwen25") f = bpe::ChatFormat::Qwen25;
    else if (name == "qwen3") f = bpe::ChatFormat::Qwen3;
    else if (name == "ornith") f = bpe::ChatFormat::Ornith;
    else return false;
    return true;
}

std::string base_name(const std::string& p) {
    const size_t s = p.find_last_of('/');
    std::string b = s == std::string::npos ? p : p.substr(s + 1);
    const size_t d = b.rfind('.');
    return d == std::string::npos || d == 0 ? b : b.substr(0, d);
}

}  // namespace

void print_serve_usage(std::FILE* f) {
    std::fprintf(f,
                 "Usage: vegajet serve <model-file>|--model FILE [--gpu IDX[,IDX|A-B]...|all|list] [--tp N] [--ctx T]\n"
                 "                     [--slots N] [--hsaco-dir DIR] [common options]\n"
                 "       vegajet serve --mock [--gguf TOKENIZER-FILE] [--slots N] [common options]\n"
                 "  common: [--host 127.0.0.1] [--port 8080] [--model-id NAME] [--max-queue 16]\n"
                 "          [--chat-format auto|chatml|qwen25|qwen3|ornith] [--enable-thinking] [--reasoning-format auto|none]\n"
                 "          [--request-log FILE] [--metrics-hw] [--max-tokens-cap N]\n"
                 "  --gpu       GPUs this process may use (positions in `vegajet serve --gpu list`); overrides VJ_GPU_DEVICES\n"
                 "  --slots     concurrent sequences (continuous batching; needs an engine with a batch path, e.g. --mock)\n"
                 "  --max-tokens-cap  cap on a request's max_tokens (default: the --ctx context length)\n"
                 "  --request-log  append one JSON line per request (timings, token counts; never prompt text)\n"
                 "  --metrics-hw   add read-only GPU sysfs telemetry (temperatures, power, RAS ECC) to GET /metrics\n"
                 "  Exit codes: 0 ok, 1 model/tokenizer/request-log unreadable, 2 usage error, 3 engine unavailable or bind failure.\n");
}

int parse_serve_args(const std::vector<std::string>& args, ServeConfig& c, std::string* err) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return 2; };
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        auto value = [&](std::string& v) -> bool {
            if (i + 1 >= args.size()) return false;
            v = args[++i];
            return true;
        };
        std::string v;
        uint64_t n = 0;
        if (a == "--help" || a == "-h") { c.help = true; return 0; }
        else if (a == "--mock") c.mock = true;
        else if (a == "--enable-thinking") c.enable_thinking = true;
        else if (a == "--host") { if (!value(c.host) || c.host.empty()) return fail("--host needs an address"); }
        else if (a == "--port") { if (!value(v) || !parse_uint(v, 0, 65535, n)) return fail("--port takes 0..65535"); c.port = static_cast<uint16_t>(n); }
        else if (a == "--model-id") { if (!value(c.model_id) || c.model_id.empty()) return fail("--model-id needs a name"); }
        else if (a == "--gguf") { if (!value(c.tokenizer_gguf)) return fail("--gguf needs a file"); }
        else if (a == "--model") { if (!value(c.model_path)) return fail("--model needs a file"); }
        else if (a == "--slots") { if (!value(v) || !parse_uint(v, 1, 64, n)) return fail("--slots takes 1..64"); c.slots = static_cast<uint32_t>(n); }
        else if (a == "--max-queue") { if (!value(v) || !parse_uint(v, 1, 4096, n)) return fail("--max-queue takes 1..4096"); c.max_queue = static_cast<uint32_t>(n); }
        else if (a == "--tp") { if (!value(v) || !parse_uint(v, 1, 8, n)) return fail("--tp takes 1..8"); c.tp = static_cast<uint32_t>(n); }
        else if (a == "--ctx") { if (!value(v) || !parse_uint(v, 256, 1u << 20, n)) return fail("--ctx takes 256..1048576"); c.ctx = static_cast<uint32_t>(n); }
        else if (a == "--hsaco-dir") { if (!value(c.hsaco_dir)) return fail("--hsaco-dir needs a directory"); }
        else if (a == "--request-log") { if (!value(c.request_log) || c.request_log.empty()) return fail("--request-log needs a file"); }
        else if (a == "--metrics-hw") c.metrics_hw = true;
        else if (a == "--max-tokens-cap") {
            if (!value(v) || !parse_uint(v, 1, 1u << 20, n)) return fail("--max-tokens-cap takes 1..1048576");
            c.max_tokens_cap = static_cast<int32_t>(n);
        }
        else if (a == "--chat-format") {
            bpe::ChatFormat f;
            if (!value(v) || (v != "auto" && !chat_format_from(v, f))) return fail("--chat-format takes auto|chatml|qwen25|qwen3|ornith");
            c.chat_format = v;
        } else if (a == "--reasoning-format") {
            if (!value(v) || (v != "auto" && v != "none")) return fail("--reasoning-format takes auto|none");
            c.parse_reasoning = v == "auto";
        } else if (!a.empty() && a[0] == '-') {
            return fail("unknown option for serve: " + a);
        } else if (c.model_path.empty()) {
            c.model_path = a;
        } else {
            return fail("unexpected argument for serve: " + a);
        }
    }
    if (c.mock && !c.model_path.empty()) return fail("--mock and a model file are mutually exclusive");
    if (!c.mock && c.model_path.empty()) return fail("serve needs a model file (or --mock)");
    if (c.mock && c.tp > 1) return fail("--tp applies to a model file, not --mock");
    return 0;
}

int extract_gpu_option(std::vector<std::string>& args, std::string& spec, bool& given, std::string* err) {
    given = false;
    std::vector<std::string> rest;
    for (size_t i = 0; i < args.size(); ++i) {
        std::string v;
        if (args[i] == "--gpu") {
            if (i + 1 >= args.size()) { if (err) *err = "--gpu needs a value (index list, range, all or list)"; return 2; }
            v = args[++i];
        } else if (args[i].rfind("--gpu=", 0) == 0) {
            v = args[i].substr(6);
        } else {
            rest.push_back(args[i]);
            continue;
        }
        if (given) { if (err) *err = "--gpu given more than once"; return 2; }
        given = true;
        spec = v;
    }
    args.swap(rest);
    return 0;
}

int apply_gpu_option(const std::string* spec, gpusel::Selection& out, bool* listed, std::string* err) {
    if (listed) *listed = false;
    if (spec && *spec == "list") {
        gpusel::Selection cur;
        std::string why;
        if (!gpusel::resolve(nullptr, std::getenv(gpusel::kEnvVar), cur, &why)) { if (err) *err = why; return 2; }
        const std::string t = gpusel::format_list(gpusel::scan_topology(gpusel::sysfs_root()), cur);
        std::fwrite(t.data(), 1, t.size(), stdout);
        if (listed) *listed = true;
        out = cur;
        return 0;
    }
    if (!gpusel::resolve(spec, std::getenv(gpusel::kEnvVar), out, err)) return 2;
    if (out.source != "default") setenv(gpusel::kEnvVar, gpusel::canonical(out).c_str(), 1);  // seen by kfd::Runtime::init
    return 0;
}

int run_serve(const std::vector<std::string>& args, const gpusel::Selection& gpus) {
    ServeConfig c;
    std::string err;
    if (parse_serve_args(args, c, &err) != 0) {
        std::fprintf(stderr, "[vegajet] %s\n", err.c_str());
        print_serve_usage(stderr);
        return 2;
    }
    if (c.help) { print_serve_usage(stdout); return 0; }

    // ---- tokenizer + chat template
    bpe::Tokenizer tok;
    bpe::ChatFormat fmt = bpe::ChatFormat::ChatML;
    bool have_tok = false;
    std::string architecture;
    const std::string tok_path = c.mock ? c.tokenizer_gguf : c.model_path;
    if (!tok_path.empty()) {
        bpe::Vocab vocab;
        if (!bpe::read_gguf_vocab(tok_path, vocab, &err) || !tok.load(vocab, &err)) {
            std::fprintf(stderr, "[vegajet] cannot read the tokenizer of %s: %s\n", tok_path.c_str(), err.c_str());
            return 1;
        }
        architecture = vocab.architecture;
        fmt = bpe::detect_chat_format(vocab.chat_template, vocab.architecture);
        have_tok = true;
    }
    if (c.chat_format != "auto") chat_format_from(c.chat_format, fmt);
    bpe::ChatOptions chat_opt;
    chat_opt.enable_thinking = c.enable_thinking;

    // ---- engine
    std::unique_ptr<MockEngine> mock;
    serve::EngineHandle handle;
    IEngine* engine = nullptr;
    sched::BatchEngine* batch = nullptr;
    int32_t eos = -1;
    if (c.mock) {
        mock = std::make_unique<MockEngine>(c.model_id.empty() ? "vegajet-mock-gfx906" : c.model_id);
        engine = mock.get();
        if (c.slots > 1) handle.batch = std::make_unique<serve::MockBatchEngine>(*mock);
        batch = handle.batch.get();
        eos = mock->tokenizer().eos_id();
    } else {
        const std::vector<gpusel::Device> devs = gpusel::scan_topology(gpusel::sysfs_root());
        std::vector<size_t> picked;
        if (!devs.empty() && !gpusel::pick_devices(gpus, devs, picked, &err)) {
            std::fprintf(stderr, "[vegajet] %s\n", err.c_str());
            return 2;
        }
        const uint32_t n_sel = devs.empty() ? static_cast<uint32_t>(gpus.all ? 0 : gpus.indices.size() + gpus.bdfs.size())
                                            : static_cast<uint32_t>(picked.size());
        if (c.tp == 0) c.tp = std::max<uint32_t>(1, n_sel);
        if (n_sel && c.tp != n_sel) {
            std::fprintf(stderr, "[vegajet] --tp %u does not match the %u GPU(s) selected (%s)\n", c.tp, n_sel, gpusel::canonical(gpus).c_str());
            return 2;
        }
        serve::EngineSpec spec;
        spec.model_path = c.model_path;
        spec.architecture = architecture;
        for (size_t i : picked) spec.gpu_indices.push_back(static_cast<uint32_t>(i));
        spec.tp = c.tp;
        spec.context = c.ctx;
        spec.hsaco_dir = c.hsaco_dir;
        if (!serve::EngineRegistry::instance().create(spec, handle, &err)) {
            std::fprintf(stderr, "[vegajet] %s\n", err.c_str());
            return 3;
        }
        engine = handle.engine.get();
        batch = handle.batch.get();
        eos = tok.eos();
        if (c.model_id.empty()) c.model_id = base_name(c.model_path);
    }
    if (c.slots > 1 && !batch) {
        std::fprintf(stderr, "[vegajet] note: this engine has no batch path; --slots %u -> 1\n", c.slots);
        c.slots = 1;
    }

    // ---- server
    serve::ServerOptions so;
    so.host = c.host;
    so.port = c.port;
    so.model_id = c.model_id.empty() ? "vegajet-mock-gfx906" : c.model_id;
    so.context_length = c.ctx;
    so.max_queue = std::max<uint32_t>(c.max_queue, c.slots);
    // S-M4: bound a single request's max_tokens. Default: the context length, so one stream cannot hold a queue slot
    // far beyond what the model can produce; --max-tokens-cap lowers it further.
    so.max_tokens_cap = c.max_tokens_cap > 0 ? c.max_tokens_cap : static_cast<int32_t>(c.ctx);
    so.parse_reasoning = c.parse_reasoning;
    so.workers = c.slots;
    std::unique_ptr<sched::ServeLoop> loop;
    serve::GenerateFn gen;
    if (batch && c.slots > 1) {
        sched::SchedulerConfig sc;
        sc.max_slots = c.slots;
        sc.kv_blocks = std::max<uint32_t>(64, c.ctx * c.slots / 16 + c.slots);
        sc.block_tokens = 16;
        sched::ServeHooks h;
        h.eos = eos;
        if (have_tok) {
            h.encode = [&tok, fmt, chat_opt](const serve::GenerateRequest& r) {
                return tok.encode(serve::format_request(r, fmt, chat_opt), tok.add_bos());
            };
            h.decode = [&tok](int32_t id) { return tok.is_special(id) ? std::string() : tok.token_bytes(id); };
        } else {  // mock vocabulary
            const BpeTokenizer& mt = engine->tokenizer();
            h.encode = [&mt](const serve::GenerateRequest& r) {
                return mt.encode(serve::format_request(r, bpe::ChatFormat::ChatML), mt.add_bos_default());
            };
            h.decode = [&mt](int32_t id) { return mt.decode_token(id, true); };
        }
        if (c.mock && have_tok) {  // the mock batch replies in the mock vocabulary
            const BpeTokenizer& mt = engine->tokenizer();
            h.decode = [&mt](int32_t id) { return mt.decode_token(id, true); };
        }
        loop = std::make_unique<sched::ServeLoop>(sc, *batch, h);
        gen = loop->generate_fn();
    } else {
        so.workers = 1;
        gen = have_tok ? serve::make_engine_generate_fn(*engine, tok, fmt, chat_opt) : serve::make_engine_generate_fn(*engine);
    }
    // v1.2: a greedy engine (the Ornith GPU runner) cannot honor sampling knobs, so serve answers HTTP 400 for them
    // instead of returning greedy text for a request that asked to sample.
    if (!loop) so.greedy_only = !engine->supports_sampling();
    // ---- observability: GET /metrics (+ read-only GPU sysfs telemetry) and the JSONL request log
    metrics::Registry registry;
    metrics::JsonlLogger request_log;
    if (!c.request_log.empty() && !request_log.open(c.request_log, &err)) {
        std::fprintf(stderr, "[vegajet] %s\n", err.c_str());
        return 1;
    }
    gen = metrics::instrument(std::move(gen), registry, request_log.is_open() ? &request_log : nullptr);
    serve::OpenAiServer* srv_ptr = nullptr;
    so.metrics_text = [&registry, &srv_ptr, hw = c.metrics_hw] {
        std::vector<metrics::GpuHw> gpus;
        if (hw) gpus = metrics::read_gpu_sysfs("/sys");
        std::string t = registry.exposition(hw ? &gpus : nullptr);
        if (srv_ptr) t += metrics::server_exposition(srv_ptr->stats());
        return t;
    };
    serve::OpenAiServer srv(so, gen);
    srv_ptr = &srv;
    if (!srv.start(&err)) {
        std::fprintf(stderr, "[vegajet] server start failed: %s\n", err.c_str());
        return 3;
    }
    g_server = &srv;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    std::printf("[vegajet] OpenAI server listening on http://%s:%u (model=%s, engine=%s, slots=%u, gpus=%s)\n", so.host.c_str(),
                static_cast<unsigned>(srv.port()), so.model_id.c_str(),
                c.mock ? (loop ? "mock+scheduler" : "mock") : handle.description.c_str(), static_cast<unsigned>(so.workers),
                gpusel::canonical(gpus).c_str());
    std::printf("[vegajet]   OpenAI API:    BASE_URL=http://%s:%u/v1  MODEL=%s  API_KEY=vegajet-local\n"
                "[vegajet]   Anthropic API: ANTHROPIC_BASE_URL=http://%s:%u  MODEL=%s\n"
                "[vegajet]   Endpoints:     GET /health  GET /v1/models  POST /v1/chat/completions  POST /v1/messages  GET /metrics\n",
                so.host.c_str(), static_cast<unsigned>(srv.port()), so.model_id.c_str(),
                so.host.c_str(), static_cast<unsigned>(srv.port()), so.model_id.c_str());
    std::fflush(stdout);
    srv.wait();
    g_server = nullptr;
    return 0;
}

}  // namespace vegajet::cli
