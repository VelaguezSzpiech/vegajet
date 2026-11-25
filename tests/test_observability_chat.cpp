// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU tests for cli_observability_chat.patch:
//   exposition   Prometheus text format: every family has HELP + TYPE before its samples, histograms cumulative with
//                le="+Inf" == _count, counters >= 0; the validator rejects each kind of violation; quantiles.
//   lock-free    8 writer threads + a concurrent scraper: exact totals, every scrape valid.
//   JSONL        instrument() records: schema (keys + types), TTFT / TPOT plausibility, one line per request under
//                concurrency (no interleaving), never the prompt or the reply text.
//   sysfs        a read-only mock tree: amdgpu cards only (by vendor), hwmon labels / fallbacks, power average vs
//                input, the six RAS blocks; the tree is byte-identical afterwards.
//   server       GET / HEAD /metrics on a live loopback server, 405 / 404, counters match the traffic.
//   top          frames rendered from live samples (rates from two scrapes) and from sysfs; run_top exit codes.
//   chat REPL    streaming and non-streaming turns, /stats /clear /save /load /system /help /quit, --system,
//                errors leave the history untouched, the next request carries the whole conversation.
#include "cli/chat_repl.hpp"
#include "cli/http_mini.hpp"
#include "cli/tui_top.hpp"
#include "core/json_lite.hpp"
#include "core/metrics.hpp"
#include "core/server.hpp"
#include "tests/test_common.hpp"

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>

using namespace vegajet;

namespace {

json::Value parse(const std::string& s) {
    json::Value v;
    if (!json::parse(s, v)) {
        std::fprintf(stderr, "bad JSON: %s\n", s.c_str());
        VJ_CHECK(false);
    }
    return v;
}

std::string read_file(const std::string& p) {
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void write_file(const std::string& p, const std::string& s) {
    std::ofstream f(p);
    f << s;
}

std::string tmpdir() {
    char t[] = "/tmp/vj_obs_XXXXXX";
    VJ_CHECK(::mkdtemp(t));
    return t;
}

void test_exposition() {
    metrics::Registry r;
    for (int i = 0; i < 10; ++i) {
        r.request_started();
        metrics::RequestRecord rec;
        rec.prompt_tokens = 100;
        rec.completion_tokens = 20;
        rec.ttft_ms = 40.0 + i;     // 40..49 ms
        rec.tpot_ms = 8.0;
        rec.e2e_ms = 200.0;
        rec.finish = "stop";
        if (i == 9) {
            rec.status = 500;
            rec.finish = "error";
        }
        r.request_finished(rec);
    }
    r.add_prefix_cache(750, 1000);
    r.add_moe_lru(90, 10);
    r.add_spec(400, 300, 100);
    r.set_queue_depth(3);
    std::vector<metrics::GpuHw> hw(1);
    hw[0].card = 1;
    hw[0].temp_edge_c = 45;
    hw[0].temp_junction_c = 61.5;
    hw[0].temp_mem_c = 70;
    hw[0].power_w = 182.25;
    for (const char* b : metrics::kRasBlocks) {
        metrics::GpuHw::Ras x;
        x.block = b;
        x.present = std::string(b) != "hdp";
        x.ce = std::string(b) == "umc" ? 3 : 0;
        hw[0].ras.push_back(x);
    }
    const std::string text = r.exposition(&hw);
    std::string err;
    if (!metrics::validate_exposition(text, &err)) std::fprintf(stderr, "%s\n%s\n", err.c_str(), text.c_str());
    VJ_CHECK(metrics::validate_exposition(text, &err));
    std::vector<metrics::Sample> s;
    VJ_CHECK(metrics::parse_exposition(text, s, &err));
    auto v = [&](const char* n, std::map<std::string, std::string> l = {}) { return metrics::find_sample(s, n, l); };
    VJ_CHECK(v("vegajet_requests_total") == 10 && v("vegajet_requests_failed_total") == 1);
    VJ_CHECK(v("vegajet_prompt_tokens_total") == 1000 && v("vegajet_generated_tokens_total") == 200);
    VJ_CHECK(v("vegajet_ttft_seconds_count") == 10 && v("vegajet_ttft_seconds_bucket", {{"le", "0.05"}}) == 10 &&
             v("vegajet_ttft_seconds_bucket", {{"le", "0.025"}}) == 0);
    VJ_CHECK(std::fabs(v("vegajet_ttft_seconds_sum") - 0.445) < 1e-6);
    VJ_CHECK(v("vegajet_tpot_seconds_bucket", {{"le", "0.01"}}) == 10 && v("vegajet_tpot_seconds_bucket", {{"le", "+Inf"}}) == 10);
    VJ_CHECK(v("vegajet_prefix_cache_hit_ratio") == 0.75 && v("vegajet_moe_lru_hit_ratio") == 0.9);
    VJ_CHECK(v("vegajet_spec_mean_accept") == 3 && v("vegajet_spec_acceptance_ratio") == 0.75);
    VJ_CHECK(v("vegajet_queue_depth") == 3 && v("vegajet_batch_size") == 0 && v("vegajet_active_requests") == 0);
    VJ_CHECK(v("vegajet_gpu_temperature_celsius", {{"card", "1"}, {"sensor", "mem"}}) == 70);
    VJ_CHECK(v("vegajet_gpu_power_watts", {{"card", "1"}}) == 182.25);
    VJ_CHECK(v("vegajet_gpu_ras_errors_total", {{"card", "1"}, {"block", "umc"}, {"type", "ce"}}) == 3);
    VJ_CHECK(std::isnan(v("vegajet_gpu_ras_errors_total", {{"block", "hdp"}})));  // absent block: no sample
    r.set_batch_size(6);
    VJ_CHECK(metrics::parse_exposition(r.exposition(), s, &err) && v("vegajet_batch_size") == 6);
    // quantiles from the snapshot: 10 samples in (0.025, 0.05]
    const auto snap = r.ttft().snapshot();
    VJ_CHECK(snap.quantile(0.5) > 0.025 && snap.quantile(0.5) <= 0.05 && snap.quantile(1.0) <= 0.05);
    // the validator catches each kind of violation
    const std::vector<std::pair<std::string, const char*>> bad = {
        {"x_total 1\n# TYPE x_total counter\n", "before its TYPE"},
        {"# TYPE x counter\n# TYPE x counter\nx 1\n", "duplicate TYPE"},
        {"# TYPE x counter\nx -1\n", "negative counter"},
        {"# TYPE 9x gauge\n9x 1\n", "bad metric name"},
        {"# TYPE h histogram\nh_bucket{le=\"1\"} 5\nh_bucket{le=\"2\"} 4\nh_bucket{le=\"+Inf\"} 5\nh_sum 1\nh_count 5\n", "cumulative"},
        {"# TYPE h histogram\nh_bucket{le=\"1\"} 5\nh_bucket{le=\"+Inf\"} 6\nh_sum 1\nh_count 5\n", "+Inf bucket != _count"},
        {"# TYPE h histogram\nh_bucket{le=\"1\"} 5\nh_sum 1\nh_count 5\n", "le=\"+Inf\""},
        {"# TYPE g gauge\ng{a=\"x\" 1\n", "label"},
        {"# TYPE g gauge\ng one\n", "bad value"},
        {"# TYPE g weird\ng 1\n", "unknown type"},
    };
    for (const auto& [t, want] : bad) {
        err.clear();
        VJ_CHECK(!metrics::validate_exposition(t, &err));
        if (err.find(want) == std::string::npos) std::fprintf(stderr, "validator on %s: %s\n", t.c_str(), err.c_str());
        VJ_CHECK(err.find(want) != std::string::npos);
    }
    std::printf("[metrics] exposition valid (%zu samples), values / ratios / quantiles exact, %zu validator refusals\n",
                s.size(), bad.size());
}

void test_lock_free() {
    metrics::Registry r;
    std::atomic<bool> done{false};
    std::atomic<int> scrapes{0}, invalid{0};
    std::thread scraper([&] {
        while (!done) {
            std::string err;
            if (!metrics::validate_exposition(r.exposition(), &err)) ++invalid;
            ++scrapes;
        }
    });
    std::vector<std::thread> w;
    for (int t = 0; t < 8; ++t)
        w.emplace_back([&, t] {
            for (int i = 0; i < 20000; ++i) {
                r.request_started();
                metrics::RequestRecord rec;
                rec.prompt_tokens = 3;
                rec.completion_tokens = 2;
                rec.ttft_ms = (i % 100) * 0.5;
                rec.tpot_ms = 1.0 + t;
                rec.finish = "stop";
                r.request_finished(rec);
                r.add_moe_lru(1, i % 2);
            }
        });
    for (auto& x : w) x.join();
    done = true;
    scraper.join();
    std::vector<metrics::Sample> s;
    VJ_CHECK(metrics::parse_exposition(r.exposition(), s));
    VJ_CHECK(metrics::find_sample(s, "vegajet_requests_total") == 160000);
    VJ_CHECK(metrics::find_sample(s, "vegajet_generated_tokens_total") == 320000);
    VJ_CHECK(metrics::find_sample(s, "vegajet_ttft_seconds_count") == 160000);
    VJ_CHECK(metrics::find_sample(s, "vegajet_moe_lru_misses_total") == 80000);
    VJ_CHECK(metrics::find_sample(s, "vegajet_active_requests") == 0);
    VJ_CHECK(invalid == 0 && scrapes > 0);
    std::printf("[metrics] lock-free: 8 x 20000 updates, exact totals, %d concurrent scrapes all valid\n", scrapes.load());
}

// Scripted generator: emits pieces with a small delay; remembers the last request.
struct Script {
    std::mutex mu;
    std::string text = "Hello there, friend.";
    int delay_ms = 2;
    serve::GenerateRequest last;
    serve::GenerateResult operator()(const serve::GenerateRequest& r, const serve::TokenSink& sink) {
        std::string t;
        int d;
        {
            std::lock_guard<std::mutex> lk(mu);
            last = r;
            t = text;
            d = delay_ms;
        }
        for (size_t i = 0; i < t.size(); i += 4) {
            if (d) std::this_thread::sleep_for(std::chrono::milliseconds(d));
            if (!sink(std::string_view(t).substr(i, 4))) break;
        }
        serve::GenerateResult res;
        res.prompt_tokens = 7;
        return res;
    }
};

void test_jsonl() {
    const std::string dir = tmpdir();
    const std::string path = dir + "/requests.jsonl";
    metrics::Registry reg;
    metrics::JsonlLogger log;
    std::string err;
    VJ_CHECK(log.open(path, &err));
    Script gen;
    gen.text = "reply text SECRET_REPLY_MARKER done";
    serve::GenerateFn fn = metrics::instrument(
        [&](const serve::GenerateRequest& r, const serve::TokenSink& s) { return gen(r, s); }, reg, &log);
    serve::GenerateRequest req;
    req.messages.push_back(serve::ChatTurn{"user", "my SECRET_PROMPT_MARKER question"});
    std::vector<std::thread> th;
    for (int t = 0; t < 6; ++t)
        th.emplace_back([&] {
            for (int i = 0; i < 5; ++i) fn(req, [](std::string_view) { return true; });
        });
    for (auto& x : th) x.join();
    // a cancelled-by-engine request and a failing one
    auto cancel_fn = metrics::instrument(
        [](const serve::GenerateRequest&, const serve::TokenSink& s) {
            s("x");
            serve::GenerateResult r;
            r.finish = serve::FinishReason::Cancelled;
            return r;
        },
        reg, &log);
    cancel_fn(req, [](std::string_view) { return true; });
    auto fail_fn = metrics::instrument(
        [](const serve::GenerateRequest&, const serve::TokenSink&) {
            serve::GenerateResult r;
            r.error = "boom";
            return r;
        },
        reg, &log);
    fail_fn(req, [](std::string_view) { return true; });
    VJ_CHECK(log.lines() == 32);
    const std::string all = read_file(path);
    VJ_CHECK(all.find("SECRET_PROMPT_MARKER") == std::string::npos && all.find("SECRET_REPLY_MARKER") == std::string::npos);
    std::istringstream ls(all);
    int n = 0, cancelled = 0, errors = 0;
    std::set<double> ids;
    for (std::string line; std::getline(ls, line); ++n) {
        const json::Value v = parse(line);
        VJ_CHECK(v.is_object() && v.obj.size() == 10);
        for (const char* k : {"ts", "id", "status", "prompt_tokens", "completion_tokens", "ttft_ms", "tpot_ms", "e2e_ms"})
            VJ_CHECK(v.get(k) && v.get(k)->is_number() && v.get(k)->num >= 0);
        VJ_CHECK(v.get("finish")->is_string() && v.get("cancelled")->is_bool());
        ids.insert(v.get("id")->num);
        if (v.get("cancelled")->b) {
            ++cancelled;
            VJ_CHECK(v.get("status")->num == 499 && v.get("finish")->str == "cancelled");
            continue;
        }
        if (v.get("status")->num == 500) {
            ++errors;
            VJ_CHECK(v.get("finish")->str == "error");
            continue;
        }
        VJ_CHECK(v.get("status")->num == 200 && v.get("finish")->str == "stop" && v.get("prompt_tokens")->num == 7);
        VJ_CHECK(v.get("completion_tokens")->num == 9);                         // 35 bytes in 4-byte pieces
        VJ_CHECK(v.get("ttft_ms")->num >= 1.5 && v.get("tpot_ms")->num >= 1.5);  // 2 ms per piece
        VJ_CHECK(v.get("e2e_ms")->num >= v.get("ttft_ms")->num);
    }
    VJ_CHECK(n == 32 && cancelled == 1 && errors == 1);
    VJ_CHECK(ids.size() == 30);  // ids count per instrumented generator: 1..30 (the two single-shot wrappers reuse 1)
    std::vector<metrics::Sample> s;
    VJ_CHECK(metrics::parse_exposition(reg.exposition(), s));
    VJ_CHECK(metrics::find_sample(s, "vegajet_requests_total") == 32 && metrics::find_sample(s, "vegajet_requests_cancelled_total") == 1 &&
             metrics::find_sample(s, "vegajet_requests_failed_total") == 1);
    struct stat st{};
    VJ_CHECK(::stat(path.c_str(), &st) == 0 && (st.st_mode & 0777) == 0600);
    std::remove(path.c_str());
    ::rmdir(dir.c_str());
    std::printf("[metrics] JSONL: 32 records from 6 concurrent clients, schema + timings OK, no prompt / reply text, "
                "file mode 0600\n");
}

// Mock sysfs: card0 (amdgpu, labelled hwmon, power1_average, full RAS), card1 (NVIDIA: skipped), card2 (amdgpu,
// unlabelled hwmon, power1_input only, two RAS blocks), card0-DP-1 (connector: skipped).
std::string make_sysfs() {
    const std::string root = tmpdir();
    auto mk = [&](const std::string& rel) { VJ_CHECK(::mkdir((root + rel).c_str(), 0755) == 0); };
    mk("/class");
    mk("/class/drm");
    for (const char* c : {"/class/drm/card0", "/class/drm/card1", "/class/drm/card2", "/class/drm/card0-DP-1"}) {
        mk(c);
        mk(std::string(c) + "/device");
    }
    write_file(root + "/class/drm/card0/device/vendor", "0x1002\n");
    write_file(root + "/class/drm/card1/device/vendor", "0x10de\n");
    write_file(root + "/class/drm/card2/device/vendor", "0x1002\n");
    write_file(root + "/class/drm/card0-DP-1/device/vendor", "0x1002\n");
    mk("/class/drm/card0/device/hwmon");
    mk("/class/drm/card0/device/hwmon/hwmon3");
    const std::string h0 = root + "/class/drm/card0/device/hwmon/hwmon3/";
    write_file(h0 + "temp1_input", "41000\n");
    write_file(h0 + "temp1_label", "edge\n");
    write_file(h0 + "temp2_input", "55000\n");
    write_file(h0 + "temp2_label", "junction\n");
    write_file(h0 + "temp3_input", "63000\n");
    write_file(h0 + "temp3_label", "mem\n");
    write_file(h0 + "power1_average", "123500000\n");
    write_file(h0 + "power1_input", "999000000\n");  // ignored: average preferred
    mk("/class/drm/card0/device/ras");
    for (const char* b : metrics::kRasBlocks)
        write_file(root + "/class/drm/card0/device/ras/" + b + "_err_count",
                   std::string(b) == "umc" ? "ue: 1\nce: 42\n" : "ue: 0\nce: 0\n");
    mk("/class/drm/card2/device/hwmon");
    mk("/class/drm/card2/device/hwmon/hwmon5");
    const std::string h2 = root + "/class/drm/card2/device/hwmon/hwmon5/";
    write_file(h2 + "temp1_input", "38000\n");
    write_file(h2 + "temp3_input", "51000\n");
    write_file(h2 + "power1_input", "20000000\n");
    mk("/class/drm/card2/device/ras");
    write_file(root + "/class/drm/card2/device/ras/gfx_err_count", "ue: 0\nce: 7\n");
    write_file(root + "/class/drm/card2/device/ras/sdma_err_count", "ue: 2\nce: 0\n");
    return root;
}

std::string tree_digest(const std::string& root) {  // paths + contents + modes, sorted
    std::string out;
    std::vector<std::string> stack{root};
    std::vector<std::string> all;
    while (!stack.empty()) {
        const std::string d = stack.back();
        stack.pop_back();
        DIR* dir = ::opendir(d.c_str());
        if (!dir) continue;
        while (dirent* e = ::readdir(dir)) {
            const std::string n = e->d_name;
            if (n == "." || n == "..") continue;
            const std::string p = d + "/" + n;
            struct stat st{};
            ::stat(p.c_str(), &st);
            if (S_ISDIR(st.st_mode)) stack.push_back(p);
            all.push_back(p + ":" + std::to_string(st.st_mode) + ":" + std::to_string(st.st_mtime) + ":" +
                          (S_ISREG(st.st_mode) ? read_file(p) : std::string()));
        }
        ::closedir(dir);
    }
    std::sort(all.begin(), all.end());
    for (const auto& a : all) out += a + "\n";
    return out;
}

void rm_tree(const std::string& root) {
    std::vector<std::string> stack{root}, dirs;
    while (!stack.empty()) {
        const std::string d = stack.back();
        stack.pop_back();
        dirs.push_back(d);
        DIR* dir = ::opendir(d.c_str());
        if (!dir) continue;
        while (dirent* e = ::readdir(dir)) {
            const std::string n = e->d_name;
            if (n == "." || n == "..") continue;
            const std::string p = d + "/" + n;
            struct stat st{};
            ::lstat(p.c_str(), &st);
            if (S_ISDIR(st.st_mode)) stack.push_back(p);
            else std::remove(p.c_str());
        }
        ::closedir(dir);
    }
    for (auto it = dirs.rbegin(); it != dirs.rend(); ++it) ::rmdir(it->c_str());
}

void test_sysfs(const std::string& root) {
    const std::string before = tree_digest(root);
    const auto g = metrics::read_gpu_sysfs(root);
    VJ_CHECK(tree_digest(root) == before);  // read-only
    VJ_CHECK(g.size() == 2 && g[0].card == 0 && g[1].card == 2);
    VJ_CHECK(g[0].temp_edge_c == 41 && g[0].temp_junction_c == 55 && g[0].temp_mem_c == 63 && g[0].power_w == 123.5);
    VJ_CHECK(g[0].ras.size() == 6 && g[0].ras[0].block == "umc" && g[0].ras[0].present && g[0].ras[0].ue == 1 && g[0].ras[0].ce == 42);
    for (const auto& r : g[0].ras) VJ_CHECK(r.present);
    VJ_CHECK(g[1].temp_edge_c == 38 && std::isnan(g[1].temp_junction_c) && g[1].temp_mem_c == 51 && g[1].power_w == 20);
    int present = 0;
    for (const auto& r : g[1].ras) present += r.present;
    VJ_CHECK(present == 2 && g[1].ras[1].block == "sdma" && g[1].ras[1].ue == 2 && g[1].ras[2].ce == 7);
    VJ_CHECK(metrics::read_gpu_sysfs(root + "/nonexistent").empty());
    // the rendered telemetry round-trips through the exposition text
    metrics::Registry reg;
    std::vector<metrics::Sample> s;
    VJ_CHECK(metrics::parse_exposition(reg.exposition(&g), s));
    const auto back = cli::gpus_from_samples(s);
    VJ_CHECK(back.size() == 2 && back[0].temp_mem_c == 63 && back[0].power_w == 123.5 && back[0].ras[0].ce == 42 &&
             back[1].ras[1].ue == 2 && !back[1].ras[0].present);
    std::printf("[sysfs] mock tree: 2 amdgpu cards of 4 entries, temps / power / 6 RAS blocks, tree unchanged (read-only)\n");
}

void test_server_and_top(const std::string& sysroot) {
    Script gen;
    metrics::Registry reg;
    serve::ServerOptions o;
    o.port = 0;
    serve::OpenAiServer* sp = nullptr;
    const auto hw = metrics::read_gpu_sysfs(sysroot);
    o.metrics_text = [&] {
        std::string t = reg.exposition(&hw);
        if (sp) t += metrics::server_exposition(sp->stats());
        return t;
    };
    serve::OpenAiServer srv(o, metrics::instrument([&](const serve::GenerateRequest& r, const serve::TokenSink& s) { return gen(r, s); }, reg));
    sp = &srv;
    std::string err;
    VJ_CHECK(srv.start(&err));
    cli::http::Url u;
    VJ_CHECK(cli::http::parse_url("http://127.0.0.1:" + std::to_string(srv.port()), u, &err));
    std::string body;
    VJ_CHECK(cli::http::get(u, "/metrics", body, &err) == 200);
    cli::TopView v0;
    v0.server = true;
    v0.source = "test";
    VJ_CHECK(metrics::parse_exposition(body, v0.samples, &err));
    v0.gpus = cli::gpus_from_samples(v0.samples);
    for (int i = 0; i < 5; ++i) {
        std::string resp;
        const int st = cli::http::request(u, "POST", "/v1/chat/completions", R"({"messages":[{"role":"user","content":"hi"}]})",
                                          [&](std::string_view p) {
                                              resp.append(p.data(), p.size());
                                              return true;
                                          },
                                          &err);
        VJ_CHECK(st == 200 && parse(resp).get("choices"));
    }
    VJ_CHECK(cli::http::get(u, "/metrics", body, &err) == 200);
    VJ_CHECK(metrics::validate_exposition(body, &err));
    cli::TopView v1;
    v1.server = true;
    v1.source = "test";
    VJ_CHECK(metrics::parse_exposition(body, v1.samples, &err));
    v1.gpus = cli::gpus_from_samples(v1.samples);
    VJ_CHECK(metrics::find_sample(v1.samples, "vegajet_requests_total") == 5);
    VJ_CHECK(metrics::find_sample(v1.samples, "vegajet_http_completions_total") == 5);
    VJ_CHECK(metrics::find_sample(v1.samples, "vegajet_generated_tokens_total") == 5 * 5);  // 20 bytes / 4
    VJ_CHECK(metrics::find_sample(v1.samples, "vegajet_ttft_seconds_count") == 5);
    VJ_CHECK(v1.gpus.size() == 2);
    // HEAD, wrong method, and a server without metrics
    std::string head;
    VJ_CHECK(cli::http::request(u, "HEAD", "/metrics", {}, [&](std::string_view p) { head.append(p.data(), p.size()); return true; }, &err) == 200 && head.empty());
    VJ_CHECK(cli::http::request(u, "POST", "/metrics", "{}", [](std::string_view) { return true; }, &err) == 405);
    {
        serve::ServerOptions o2;
        o2.port = 0;
        serve::OpenAiServer plain(o2, [&](const serve::GenerateRequest& r, const serve::TokenSink& s) { return gen(r, s); });
        VJ_CHECK(plain.start(&err));
        cli::http::Url u2;
        cli::http::parse_url("http://127.0.0.1:" + std::to_string(plain.port()), u2);
        VJ_CHECK(cli::http::get(u2, "/metrics", body, &err) == 404);
        plain.stop();
    }
    // top frames: rates from two scrapes, latency, GPU table, UE flag
    const std::string frame = cli::render_top(v1, &v0, 2.0, 100, false);
    for (const char* want : {"requests  5 total, 2.5 req/s", "generated 25", "12.5 tok/s", "TTFT p50", "TPOT p50", "prefix $",
                             "MoE LRU", "mean_accept", "mem(TMem)", " 124 W", "1/42", "UE!", "   2  "})
        if (frame.find(want) == std::string::npos) {
            std::fprintf(stderr, "frame lacks '%s':\n%s\n", want, frame.c_str());
            VJ_CHECK(false);
        }
    VJ_CHECK(frame.find("\x1b[") == std::string::npos);  // no ANSI without color
    VJ_CHECK(cli::render_top(v1, nullptr, 0, 100, true).find("\x1b[1m") != std::string::npos);
    cli::TopView sysv;
    sysv.source = "sysfs";
    sysv.gpus = metrics::read_gpu_sysfs(sysroot);
    const std::string sframe = cli::render_top(sysv, nullptr, 0, 80, false);
    VJ_CHECK(sframe.find("no server metrics") != std::string::npos && sframe.find("  41 ") != std::string::npos);
    // run_top exit codes (frames go to stdout, i.e. the test log)
    const std::string url = "http://127.0.0.1:" + std::to_string(srv.port());
    VJ_CHECK(cli::run_top({"--url", url, "--once"}) == 0);
    VJ_CHECK(cli::run_top({"--url", url, "--frames", "2", "--interval", "0.1", "--sysfs-root", sysroot}) == 0);
    VJ_CHECK(cli::run_top({"--url", "http://127.0.0.1:1", "--once"}) == 1);  // nothing readable
    VJ_CHECK(cli::run_top({"--url", "http://127.0.0.1:1", "--once", "--sysfs-root", sysroot}) == 0);
    VJ_CHECK(cli::run_top({"--interval", "0"}) == 2 && cli::run_top({"--bogus"}) == 2 && cli::run_top({"--url", "https://x"}) == 2);
    srv.stop();
    std::printf("[server] GET/HEAD /metrics valid after traffic (5 requests, 25 tokens), 405 / 404; top frames and "
                "exit codes OK\n");
}

void test_chat() {
    Script gen;
    gen.delay_ms = 0;
    serve::ServerOptions o;
    o.port = 0;
    serve::OpenAiServer srv(o, [&](const serve::GenerateRequest& r, const serve::TokenSink& s) { return gen(r, s); });
    std::string err;
    VJ_CHECK(srv.start(&err));
    cli::ChatConfig cfg;
    cfg.url = "http://127.0.0.1:" + std::to_string(srv.port());
    cfg.system = "You are terse.";
    cfg.max_tokens = 64;
    cli::ChatSession chat(cfg);
    std::ostringstream out;
    {
        std::lock_guard<std::mutex> lk(gen.mu);
        gen.text = "<think>pondering</think>The answer is 42.";
    }
    VJ_CHECK(chat.handle("What is it?", out));
    VJ_CHECK(out.str().find("pondering") != std::string::npos && out.str().find("The answer is 42.") != std::string::npos);
    VJ_CHECK(chat.history().size() == 2 && chat.history()[1].content == "The answer is 42.");
    VJ_CHECK(gen.last.messages.size() == 2 && gen.last.messages[0].role == "system" &&
             gen.last.messages[0].content == "You are terse." && gen.last.max_tokens == 64);
    {
        std::lock_guard<std::mutex> lk(gen.mu);
        gen.text = "Second reply.";
    }
    out.str("");
    VJ_CHECK(chat.handle("And then?", out) && out.str() == "Second reply.\n");
    VJ_CHECK(gen.last.messages.size() == 4 && gen.last.messages[2].content == "The answer is 42." &&
             gen.last.messages[3].content == "And then?");  // the whole conversation goes with each turn
    out.str("");
    VJ_CHECK(chat.handle("/stats", out));
    VJ_CHECK(out.str().find("turns 2") != std::string::npos && out.str().find("prompt tokens 14") != std::string::npos &&
             out.str().find("errors 0") != std::string::npos);
    VJ_CHECK(chat.stats().completion_tokens > 0);
    const std::string dir = tmpdir(), file = dir + "/conv.json";
    out.str("");
    VJ_CHECK(chat.handle("/save " + file, out) && out.str().find("saved 4 messages") != std::string::npos);
    const json::Value saved = parse(read_file(file));
    VJ_CHECK(saved.get("version")->num == 1 && saved.get("system")->str == "You are terse." && saved.get("messages")->arr.size() == 4);
    out.str("");
    VJ_CHECK(chat.handle("/clear", out) && chat.history().empty());
    VJ_CHECK(chat.handle("/load " + file, out) && chat.history().size() == 4 && chat.history()[3].content == "Second reply.");
    out.str("");
    VJ_CHECK(chat.handle("/load " + dir + "/missing.json", out) && out.str().find("error: cannot read") != std::string::npos);
    write_file(dir + "/bad.json", R"({"messages":[{"role":"system","content":"x"}]})");
    out.str("");
    VJ_CHECK(chat.handle("/load " + dir + "/bad.json", out) && out.str().find("role user / assistant") != std::string::npos &&
             chat.history().size() == 4);
    out.str("");
    VJ_CHECK(chat.handle("/system Be verbose.", out) && chat.config().system == "Be verbose.");
    VJ_CHECK(chat.handle("/help", out) && out.str().find("/save FILE") != std::string::npos);
    out.str("");
    VJ_CHECK(chat.handle("/frobnicate", out) && out.str().find("unknown command") != std::string::npos);
    VJ_CHECK(!chat.handle("/quit", out) && !chat.handle("/exit", out));
    // non-streaming mode
    cli::ChatConfig c2 = cfg;
    c2.stream = false;
    cli::ChatSession nostream(c2);
    out.str("");
    VJ_CHECK(nostream.handle("hi", out) && out.str() == "Second reply.\n" && nostream.history().size() == 2);
    // a server-side error leaves the history alone
    cli::ChatConfig c3 = cfg;
    c3.max_tokens = 0;
    cli::ChatSession bad(c3);
    out.str("");
    VJ_CHECK(bad.handle("hi", out) && out.str().find("error: 'max_tokens' must be >= 1") != std::string::npos &&
             bad.history().empty() && bad.stats().errors == 1);
    cli::ChatConfig c4 = cfg;
    c4.url = "http://127.0.0.1:1";
    cli::ChatSession down(c4);
    out.str("");
    VJ_CHECK(down.handle("hi", out) && out.str().find("error: cannot connect") != std::string::npos);
    VJ_CHECK(cli::run_chat({"--url", "http://127.0.0.1:1"}) == 1 && cli::run_chat({"--max-tokens", "0"}) == 2 &&
             cli::run_chat({"--bogus"}) == 2);
    std::remove(file.c_str());
    std::remove((dir + "/bad.json").c_str());
    ::rmdir(dir.c_str());
    srv.stop();
    std::printf("[chat] streaming (reasoning + content) and non-streaming turns, history carried, /stats /save /load "
                "/clear /system /help /quit, errors keep history, exit codes OK\n");
}

void test_url() {
    cli::http::Url u;
    VJ_CHECK(cli::http::parse_url("http://localhost:9000/base/", u) && u.host == "localhost" && u.port == 9000 && u.base == "/base");
    VJ_CHECK(cli::http::parse_url("http://[::1]:8080", u) && u.host == "::1" && u.port == 8080 && u.base.empty());
    VJ_CHECK(cli::http::parse_url("http://example", u) && u.port == 80);
    std::string err;
    VJ_CHECK(!cli::http::parse_url("https://x", u, &err) && !cli::http::parse_url("http://:80", u, &err) &&
             !cli::http::parse_url("http://h:99999", u, &err));
}

}  // namespace

int main() {
    test_url();
    test_exposition();
    test_lock_free();
    test_jsonl();
    const std::string sysroot = make_sysfs();
    test_sysfs(sysroot);
    test_server_and_top(sysroot);
    rm_tree(sysroot);
    test_chat();
    std::printf("test_observability_chat: all passed\n");
    return 0;
}
