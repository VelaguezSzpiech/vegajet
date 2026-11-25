// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "cli/tui_top.hpp"

#include "cli/http_mini.hpp"

#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <thread>

namespace vegajet::cli {
namespace {

std::atomic<bool> g_top_stop{false};
void on_top_signal(int) { g_top_stop = true; }

double val(const std::vector<metrics::Sample>& s, const char* name) {
    const double v = metrics::find_sample(s, name);
    return std::isnan(v) ? 0.0 : v;
}

// Histogram quantile from cumulative bucket samples (name_bucket{le=..}).
double quantile(const std::vector<metrics::Sample>& s, const std::string& name, double q) {
    std::vector<std::pair<double, double>> b;  // (le, cumulative)
    for (const auto& x : s) {
        if (x.name != name + "_bucket") continue;
        auto it = x.labels.find("le");
        if (it == x.labels.end()) continue;
        const double le = it->second == "+Inf" ? INFINITY : std::atof(it->second.c_str());
        b.emplace_back(le, x.value);
    }
    std::sort(b.begin(), b.end());
    if (b.empty() || b.back().second <= 0) return NAN;
    metrics::Histogram::Snapshot snap;
    double prev = 0;
    for (const auto& [le, cum] : b) {
        if (!std::isinf(le)) snap.bounds.push_back(le);
        snap.counts.push_back(static_cast<uint64_t>(std::max(0.0, cum - prev)));
        prev = cum;
    }
    if (snap.counts.size() == snap.bounds.size()) snap.counts.push_back(0);
    snap.count = static_cast<uint64_t>(b.back().second);
    return snap.quantile(q);
}

std::string ms(double seconds) {
    if (std::isnan(seconds)) return "-";
    char b[32];
    std::snprintf(b, sizeof b, seconds < 0.1 ? "%.1f ms" : "%.0f ms", seconds * 1000.0);
    return b;
}

std::string pct(double r) {
    char b[16];
    std::snprintf(b, sizeof b, "%5.1f %%", r * 100.0);
    return b;
}

std::string bar(double r, int w) {
    r = std::clamp(r, 0.0, 1.0);
    const int n = static_cast<int>(std::lround(r * w));
    return "[" + std::string(static_cast<size_t>(n), '#') + std::string(static_cast<size_t>(w - n), '.') + "]";
}

std::string num(double v, const char* f = "%.0f") {
    char b[32];
    std::snprintf(b, sizeof b, f, v);
    return b;
}

std::string temp(double c) { return std::isnan(c) ? "   -" : num(c, "%4.0f"); }

void usage(std::FILE* f) {
    std::fprintf(f,
                 "Usage: vegajet top [--url http://127.0.0.1:8080] [--sysfs] [--sysfs-root DIR] [--interval S]\n"
                 "                   [--frames N] [--once]\n"
                 "  --url         server whose GET /metrics is shown (default http://127.0.0.1:8080)\n"
                 "  --sysfs       read local GPU telemetry from sysfs directly (read-only; works without a server)\n"
                 "  --sysfs-root  sysfs mount point for --sysfs (default /sys)\n"
                 "  --interval    seconds between frames (default 1, 0.1..60)\n"
                 "  --frames      stop after N frames (default: until Ctrl-C)\n"
                 "  --once        print one plain frame (no screen clearing, no colors) and exit\n"
                 "  Exit codes: 0 ok, 1 nothing could be read (with --once), 2 usage error.\n");
}

}  // namespace

std::vector<metrics::GpuHw> gpus_from_samples(const std::vector<metrics::Sample>& s) {
    std::map<int, metrics::GpuHw> by;
    auto card = [&](const metrics::Sample& x) -> metrics::GpuHw& {
        const int c = std::atoi(x.labels.count("card") ? x.labels.at("card").c_str() : "-1");
        auto& g = by[c];
        if (g.card < 0) {
            g.card = c;
            for (const char* b : metrics::kRasBlocks) {
                metrics::GpuHw::Ras r;
                r.block = b;
                g.ras.push_back(r);
            }
        }
        return g;
    };
    for (const auto& x : s) {
        if (x.name == "vegajet_gpu_temperature_celsius") {
            auto& g = card(x);
            const std::string sensor = x.labels.count("sensor") ? x.labels.at("sensor") : "";
            if (sensor == "edge") g.temp_edge_c = x.value;
            else if (sensor == "junction") g.temp_junction_c = x.value;
            else if (sensor == "mem") g.temp_mem_c = x.value;
        } else if (x.name == "vegajet_gpu_power_watts") {
            card(x).power_w = x.value;
        } else if (x.name == "vegajet_gpu_ras_errors_total") {
            auto& g = card(x);
            const std::string block = x.labels.count("block") ? x.labels.at("block") : "";
            for (auto& r : g.ras)
                if (r.block == block) {
                    r.present = true;
                    if (x.labels.count("type") && x.labels.at("type") == "ue") r.ue = static_cast<uint64_t>(x.value);
                    else r.ce = static_cast<uint64_t>(x.value);
                }
        }
    }
    std::vector<metrics::GpuHw> out;
    for (auto& [c, g] : by) out.push_back(std::move(g));
    return out;
}

std::string render_top(const TopView& cur, const TopView* prev, double dt_s, int width, bool color) {
    const char* B = color ? "\x1b[1m" : "";
    const char* D = color ? "\x1b[2m" : "";
    const char* R = color ? "\x1b[31;1m" : "";
    const char* G = color ? "\x1b[32m" : "";
    const char* N = color ? "\x1b[0m" : "";
    width = std::max(60, width);
    std::string o;
    auto line = [&](const std::string& s) { o += s + "\n"; };
    line(std::string(B) + "vegajet top" + N + D + "  " + cur.source + N);
    line(std::string(static_cast<size_t>(std::min(width, 100)), '-'));
    const auto& s = cur.samples;
    if (cur.server) {
        auto rate = [&](const char* name) {
            if (!prev || !prev->server || dt_s <= 0) return std::string("-");
            return num(std::max(0.0, val(s, name) - val(prev->samples, name)) / dt_s, "%.1f");
        };
        line("requests  " + num(val(s, "vegajet_requests_total")) + " total, " + rate("vegajet_requests_total") +
             " req/s   failed " + num(val(s, "vegajet_requests_failed_total")) + "   cancelled " +
             num(val(s, "vegajet_http_cancelled_total")) + "   rejected (503) " + num(val(s, "vegajet_http_rejected_total")));
        line("tokens    prompt " + num(val(s, "vegajet_prompt_tokens_total")) + "   generated " +
             num(val(s, "vegajet_generated_tokens_total")) + "   " + G + rate("vegajet_generated_tokens_total") + " tok/s" + N);
        line("latency   TTFT p50 " + ms(quantile(s, "vegajet_ttft_seconds", 0.5)) + "  p95 " +
             ms(quantile(s, "vegajet_ttft_seconds", 0.95)) + "   TPOT p50 " + ms(quantile(s, "vegajet_tpot_seconds", 0.5)) +
             "  p95 " + ms(quantile(s, "vegajet_tpot_seconds", 0.95)));
        line("batch     active " + num(val(s, "vegajet_batch_size")) + "   queue " + num(val(s, "vegajet_queue_depth")));
        const double pc = val(s, "vegajet_prefix_cache_hit_ratio"), moe = val(s, "vegajet_moe_lru_hit_ratio");
        line("prefix $  " + bar(pc, 20) + " " + pct(pc) + "   (" + num(val(s, "vegajet_prefix_cache_hit_tokens_total")) +
             " / " + num(val(s, "vegajet_prefix_cache_lookup_tokens_total")) + " tokens)");
        line("MoE LRU   " + bar(moe, 20) + " " + pct(moe) + "   spec mean_accept " +
             num(val(s, "vegajet_spec_mean_accept"), "%.2f") + "  acceptance " + pct(val(s, "vegajet_spec_acceptance_ratio")));
    } else {
        line(std::string(D) + "no server metrics (" + cur.source + ")" + N);
    }
    line("");
    if (cur.gpus.empty()) {
        line(std::string(D) + "no GPU telemetry (start serve with --metrics-hw, or use --sysfs)" + N);
    } else {
        line(std::string(B) + "card  edge  junct  mem(TMem)  power   RAS ue/ce: umc sdma gfx mmhub pcie_bif hdp" + N);
        for (const auto& g : cur.gpus) {
            std::string ras;
            uint64_t ue = 0;
            for (const auto& r : g.ras) {
                if (!r.present) {
                    ras += " -";
                    continue;
                }
                ue += r.ue;
                ras += " " + std::to_string(r.ue) + "/" + std::to_string(r.ce);
            }
            const std::string pw = std::isnan(g.power_w) ? "    -" : num(g.power_w, "%5.0f");
            line(num(g.card, "%4.0f") + "  " + temp(g.temp_edge_c) + "   " + temp(g.temp_junction_c) + "     " +
                 temp(g.temp_mem_c) + "   " + pw + " W " + (ue ? R : "") + ras + (ue ? "  UE!" : "") + N);
        }
    }
    return o;
}

int run_top(const std::vector<std::string>& args) {
    std::string url = "http://127.0.0.1:8080", root = "/sys";
    bool sysfs = false, once = false;
    double interval = 1.0;
    long frames = 0;
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        auto value = [&](std::string& v) {
            if (i + 1 >= args.size()) return false;
            v = args[++i];
            return true;
        };
        std::string v;
        if (a == "--help" || a == "-h") {
            usage(stdout);
            return 0;
        } else if (a == "--url") {
            if (!value(url)) { usage(stderr); return 2; }
        } else if (a == "--sysfs") {
            sysfs = true;
        } else if (a == "--sysfs-root") {
            if (!value(root)) { usage(stderr); return 2; }
            sysfs = true;
        } else if (a == "--interval") {
            if (!value(v)) { usage(stderr); return 2; }
            interval = std::atof(v.c_str());
            if (!(interval >= 0.1 && interval <= 60)) { std::fprintf(stderr, "--interval takes 0.1..60\n"); return 2; }
        } else if (a == "--frames") {
            if (!value(v)) { usage(stderr); return 2; }
            frames = std::atol(v.c_str());
            if (frames < 1) { std::fprintf(stderr, "--frames takes N >= 1\n"); return 2; }
        } else if (a == "--once") {
            once = true;
        } else {
            std::fprintf(stderr, "Unknown option for top: %s\n", a.c_str());
            usage(stderr);
            return 2;
        }
    }
    http::Url u;
    std::string err;
    if (!http::parse_url(url, u, &err)) {
        std::fprintf(stderr, "[vegajet] %s\n", err.c_str());
        return 2;
    }
    if (once) frames = 1;
    const bool tty = !once && ::isatty(STDOUT_FILENO);
    int width = 100;
    if (winsize ws{}; tty && ::ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) width = ws.ws_col;
    std::signal(SIGINT, on_top_signal);
    std::signal(SIGTERM, on_top_signal);
    TopView prev;
    bool have_prev = false;
    auto t_prev = std::chrono::steady_clock::now();
    for (long f = 0; (frames == 0 || f < frames) && !g_top_stop; ++f) {
        if (f > 0) {
            const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(static_cast<int>(interval * 1000));
            while (!g_top_stop && std::chrono::steady_clock::now() < until)
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            if (g_top_stop) break;
        }
        TopView cur;
        cur.source = url;
        std::string body;
        if (http::get(u, "/metrics", body, &err, 3000) == 200 && metrics::parse_exposition(body, cur.samples, &err)) {
            cur.server = true;
            cur.gpus = gpus_from_samples(cur.samples);
        } else {
            cur.source = url + ": " + err;
        }
        if (sysfs) {
            cur.gpus = metrics::read_gpu_sysfs(root);
            if (!cur.server) cur.source = "sysfs " + root;
        }
        const auto now = std::chrono::steady_clock::now();
        const double dt = std::chrono::duration<double>(now - t_prev).count();
        const std::string frame = render_top(cur, have_prev ? &prev : nullptr, dt, width, tty);
        if (tty) std::fputs("\x1b[H\x1b[2J", stdout);
        std::fputs(frame.c_str(), stdout);
        std::fflush(stdout);
        if (once && !cur.server && cur.gpus.empty()) return 1;
        prev = std::move(cur);
        have_prev = true;
        t_prev = now;
    }
    return 0;
}

}  // namespace vegajet::cli
