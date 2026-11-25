// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/metrics.hpp"

#include "core/json_lite.hpp"

#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>

namespace vegajet::metrics {
namespace {

using Clock = std::chrono::steady_clock;

std::string fmt(double v) {
    if (std::isnan(v)) return "NaN";
    if (std::isinf(v)) return v > 0 ? "+Inf" : "-Inf";
    char b[48];
    if (v == std::floor(v) && std::fabs(v) < 1e15) std::snprintf(b, sizeof b, "%.0f", v);
    else std::snprintf(b, sizeof b, "%.9g", v);
    return b;
}

void family(std::string& o, const char* name, const char* type, const char* help) {
    o += "# HELP ";
    o += name;
    o += ' ';
    o += help;
    o += "\n# TYPE ";
    o += name;
    o += ' ';
    o += type;
    o += '\n';
}

void sample(std::string& o, const std::string& name, const std::string& labels, double v) {
    o += name;
    if (!labels.empty()) o += "{" + labels + "}";
    o += ' ';
    o += fmt(v);
    o += '\n';
}

void scalar(std::string& o, const char* name, const char* type, const char* help, double v) {
    family(o, name, type, help);
    sample(o, name, "", v);
}

void histogram(std::string& o, const char* name, const char* help, const Histogram& h) {
    family(o, name, "histogram", help);
    const auto s = h.snapshot();
    uint64_t cum = 0;
    for (size_t i = 0; i < s.bounds.size(); ++i) {
        cum += s.counts[i];
        sample(o, std::string(name) + "_bucket", "le=\"" + fmt(s.bounds[i]) + "\"", static_cast<double>(cum));
    }
    cum += s.counts.back();
    sample(o, std::string(name) + "_bucket", "le=\"+Inf\"", static_cast<double>(cum));
    sample(o, std::string(name) + "_sum", "", s.sum);
    sample(o, std::string(name) + "_count", "", static_cast<double>(cum));
}

double ratio(uint64_t a, uint64_t b) { return b ? static_cast<double>(a) / static_cast<double>(b) : 0.0; }

bool read_text(const std::string& path, std::string& out) {
    std::ifstream f(path);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

bool read_number(const std::string& path, double& v) {
    std::string t;
    if (!read_text(path, t)) return false;
    char* end = nullptr;
    v = std::strtod(t.c_str(), &end);
    return end != t.c_str();
}

std::string trim(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ' || s.back() == '\r' || s.back() == '\t')) s.pop_back();
    size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
    return s.substr(i);
}

std::vector<std::string> list_dir(const std::string& path) {
    std::vector<std::string> out;
    DIR* d = ::opendir(path.c_str());
    if (!d) return out;
    while (dirent* e = ::readdir(d)) {
        const std::string n = e->d_name;
        if (n != "." && n != "..") out.push_back(n);
    }
    ::closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

bool valid_metric_name(std::string_view n) {
    if (n.empty()) return false;
    for (size_t i = 0; i < n.size(); ++i) {
        const char c = n[i];
        const bool alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || c == ':';
        if (!(alpha || (i > 0 && c >= '0' && c <= '9'))) return false;
    }
    return true;
}

bool valid_label_name(std::string_view n) {
    if (n.empty()) return false;
    for (size_t i = 0; i < n.size(); ++i) {
        const char c = n[i];
        const bool alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
        if (!(alpha || (i > 0 && c >= '0' && c <= '9'))) return false;
    }
    return true;
}

bool parse_value(std::string_view t, double& v) {
    if (t == "+Inf" || t == "Inf") v = INFINITY;
    else if (t == "-Inf") v = -INFINITY;
    else if (t == "NaN") v = NAN;
    else {
        const std::string s(t);
        char* end = nullptr;
        v = std::strtod(s.c_str(), &end);
        if (end == s.c_str() || *end != '\0') return false;
    }
    return true;
}

} // namespace

const char* const kRasBlocks[6] = {"umc", "sdma", "gfx", "mmhub", "pcie_bif", "hdp"};

// ------------------------------------------------------------------------------------------------- Histogram

Histogram::Histogram(std::vector<double> upper_bounds) : bounds_(std::move(upper_bounds)) {
    std::sort(bounds_.begin(), bounds_.end());
    if (bounds_.size() > kMaxBuckets) bounds_.resize(kMaxBuckets);
}

void Histogram::observe(double seconds) {
    if (!(seconds >= 0)) seconds = 0;
    size_t i = 0;
    while (i < bounds_.size() && seconds > bounds_[i]) ++i;
    counts_[i].fetch_add(1, std::memory_order_relaxed);
    sum_ns_.fetch_add(static_cast<uint64_t>(seconds * 1e9 + 0.5), std::memory_order_relaxed);
    count_.fetch_add(1, std::memory_order_relaxed);
}

Histogram::Snapshot Histogram::snapshot() const {
    Snapshot s;
    s.bounds = bounds_;
    s.counts.resize(bounds_.size() + 1);
    for (size_t i = 0; i <= bounds_.size(); ++i) s.counts[i] = counts_[i].load(std::memory_order_relaxed);
    s.sum = static_cast<double>(sum_ns_.load(std::memory_order_relaxed)) * 1e-9;
    s.count = 0;
    for (uint64_t c : s.counts) s.count += c;
    return s;
}

double Histogram::Snapshot::quantile(double q) const {
    if (count == 0) return NAN;
    const double target = std::clamp(q, 0.0, 1.0) * static_cast<double>(count);
    uint64_t cum = 0;
    for (size_t i = 0; i < counts.size(); ++i) {
        const uint64_t prev = cum;
        cum += counts[i];
        if (static_cast<double>(cum) >= target && counts[i] > 0) {
            if (i >= bounds.size()) return bounds.empty() ? NAN : bounds.back();
            const double lo = i == 0 ? 0.0 : bounds[i - 1], hi = bounds[i];
            const double frac = (target - static_cast<double>(prev)) / static_cast<double>(counts[i]);
            return lo + (hi - lo) * std::clamp(frac, 0.0, 1.0);
        }
    }
    return bounds.empty() ? NAN : bounds.back();
}

// ------------------------------------------------------------------------------------------------- JSONL

std::string to_jsonl(const RequestRecord& r, double unix_ts) {
    char b[512];
    std::snprintf(b, sizeof b,
                  "{\"ts\":%.3f,\"id\":%llu,\"status\":%d,\"finish\":%s,\"prompt_tokens\":%u,\"completion_tokens\":%u,"
                  "\"ttft_ms\":%.3f,\"tpot_ms\":%.3f,\"e2e_ms\":%.3f,\"cancelled\":%s}\n",
                  unix_ts, static_cast<unsigned long long>(r.id), r.status, json::quote(r.finish).c_str(),
                  r.prompt_tokens, r.completion_tokens, r.ttft_ms, r.tpot_ms, r.e2e_ms, r.cancelled ? "true" : "false");
    return b;
}

JsonlLogger::~JsonlLogger() {
    if (fd_ >= 0) ::close(fd_);
}

bool JsonlLogger::open(const std::string& path, std::string* err) {
    if (fd_ >= 0) ::close(fd_);
    fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (fd_ < 0 && err) *err = "cannot open request log " + path + ": " + std::strerror(errno);
    return fd_ >= 0;
}

void JsonlLogger::log(const RequestRecord& r) {
    if (fd_ < 0) return;
    const double ts =
        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    const std::string line = to_jsonl(r, ts);
    // O_APPEND: each write lands at the end atomically; a line is one write.
    const ssize_t n = ::write(fd_, line.data(), line.size());
    if (n == static_cast<ssize_t>(line.size())) lines_.fetch_add(1, std::memory_order_relaxed);
}

// ------------------------------------------------------------------------------------------------- sysfs

std::vector<GpuHw> read_gpu_sysfs(const std::string& root) {
    std::vector<GpuHw> out;
    const std::string drm = root + "/class/drm";
    for (const std::string& name : list_dir(drm)) {
        if (name.size() < 5 || name.compare(0, 4, "card") != 0) continue;
        if (!std::all_of(name.begin() + 4, name.end(), [](char c) { return c >= '0' && c <= '9'; })) continue;
        const std::string dev = drm + "/" + name + "/device";
        std::string vendor;
        if (!read_text(dev + "/vendor", vendor) || trim(vendor) != "0x1002") continue;
        GpuHw g;
        g.card = std::atoi(name.c_str() + 4);
        for (const std::string& hw : list_dir(dev + "/hwmon")) {
            if (hw.compare(0, 5, "hwmon") != 0) continue;
            const std::string h = dev + "/hwmon/" + hw;
            for (int k = 1; k <= 8; ++k) {
                double v;
                if (!read_number(h + "/temp" + std::to_string(k) + "_input", v)) continue;
                std::string label;
                if (read_text(h + "/temp" + std::to_string(k) + "_label", label)) label = trim(label);
                else label = k == 1 ? "edge" : k == 2 ? "junction" : k == 3 ? "mem" : "";
                if (label == "edge") g.temp_edge_c = v / 1000.0;
                else if (label == "junction") g.temp_junction_c = v / 1000.0;
                else if (label == "mem") g.temp_mem_c = v / 1000.0;
            }
            double p;
            if (read_number(h + "/power1_average", p) || read_number(h + "/power1_input", p)) g.power_w = p / 1e6;
            break;  // one hwmon per card
        }
        for (const char* block : kRasBlocks) {
            GpuHw::Ras r;
            r.block = block;
            std::string t;
            if (read_text(dev + "/ras/" + std::string(block) + "_err_count", t)) {
                r.present = true;
                std::istringstream ls(t);
                std::string line;
                while (std::getline(ls, line)) {
                    const size_t c = line.find(':');
                    if (c == std::string::npos) continue;
                    const std::string key = trim(line.substr(0, c));
                    const uint64_t val = std::strtoull(line.c_str() + c + 1, nullptr, 10);
                    if (key == "ue") r.ue = val;
                    else if (key == "ce") r.ce = val;
                }
            }
            g.ras.push_back(r);
        }
        out.push_back(std::move(g));
    }
    std::sort(out.begin(), out.end(), [](const GpuHw& a, const GpuHw& b) { return a.card < b.card; });
    return out;
}

// ------------------------------------------------------------------------------------------------- Registry

Registry::Registry()
    : ttft_({0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30}),
      tpot_({0.001, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28}),
      e2e_({0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300}) {}

void Registry::request_started() { active_.fetch_add(1, std::memory_order_relaxed); }

void Registry::request_finished(const RequestRecord& r) {
    active_.fetch_sub(1, std::memory_order_relaxed);
    requests_.fetch_add(1, std::memory_order_relaxed);
    if (r.status >= 500) failed_.fetch_add(1, std::memory_order_relaxed);
    if (r.cancelled) cancelled_.fetch_add(1, std::memory_order_relaxed);
    prompt_tokens_.fetch_add(r.prompt_tokens, std::memory_order_relaxed);
    gen_tokens_.fetch_add(r.completion_tokens, std::memory_order_relaxed);
    if (r.completion_tokens > 0) ttft_.observe(r.ttft_ms / 1000.0);
    if (r.completion_tokens > 1) tpot_.observe(r.tpot_ms / 1000.0);
    e2e_.observe(r.e2e_ms / 1000.0);
}

void Registry::add_prefix_cache(uint64_t hit, uint64_t lookup) {
    pc_hit_.fetch_add(hit, std::memory_order_relaxed);
    pc_lookup_.fetch_add(lookup, std::memory_order_relaxed);
}

void Registry::add_moe_lru(uint64_t hits, uint64_t misses) {
    moe_hit_.fetch_add(hits, std::memory_order_relaxed);
    moe_miss_.fetch_add(misses, std::memory_order_relaxed);
}

void Registry::add_spec(uint64_t drafted, uint64_t accepted, uint64_t steps) {
    spec_drafted_.fetch_add(drafted, std::memory_order_relaxed);
    spec_accepted_.fetch_add(accepted, std::memory_order_relaxed);
    spec_steps_.fetch_add(steps, std::memory_order_relaxed);
}

void Registry::set_batch_size(uint32_t n) { batch_override_.store(n, std::memory_order_relaxed); }
void Registry::set_queue_depth(uint32_t n) { queue_depth_.store(n, std::memory_order_relaxed); }

std::string Registry::exposition(const std::vector<GpuHw>* hw) const {
    auto ld = [](const std::atomic<uint64_t>& a) { return a.load(std::memory_order_relaxed); };
    std::string o;
    o.reserve(8192);
    scalar(o, "vegajet_requests_total", "counter", "Generation requests finished (any outcome).", static_cast<double>(ld(requests_)));
    scalar(o, "vegajet_requests_failed_total", "counter", "Requests whose generator failed.", static_cast<double>(ld(failed_)));
    scalar(o, "vegajet_requests_cancelled_total", "counter", "Requests the engine reported as cancelled.", static_cast<double>(ld(cancelled_)));
    scalar(o, "vegajet_prompt_tokens_total", "counter", "Prompt tokens processed.", static_cast<double>(ld(prompt_tokens_)));
    scalar(o, "vegajet_generated_tokens_total", "counter", "Tokens generated.", static_cast<double>(ld(gen_tokens_)));
    const int64_t act = std::max<int64_t>(0, active_.load(std::memory_order_relaxed));
    const int64_t ovr = batch_override_.load(std::memory_order_relaxed);
    scalar(o, "vegajet_active_requests", "gauge", "Requests generating right now.", static_cast<double>(act));
    scalar(o, "vegajet_batch_size", "gauge", "Sequences in the active decode batch.", static_cast<double>(ovr >= 0 ? ovr : act));
    scalar(o, "vegajet_queue_depth", "gauge", "Requests waiting for a slot.", static_cast<double>(queue_depth_.load(std::memory_order_relaxed)));
    histogram(o, "vegajet_ttft_seconds", "Time to first token.", ttft_);
    histogram(o, "vegajet_tpot_seconds", "Time per output token after the first.", tpot_);
    histogram(o, "vegajet_e2e_seconds", "Request latency, first byte in to last token out.", e2e_);
    const uint64_t ph = ld(pc_hit_), pl = ld(pc_lookup_), mh = ld(moe_hit_), mm = ld(moe_miss_);
    const uint64_t sd = ld(spec_drafted_), sa = ld(spec_accepted_), ss = ld(spec_steps_);
    scalar(o, "vegajet_prefix_cache_hit_tokens_total", "counter", "Prompt tokens served from the prefix cache.", static_cast<double>(ph));
    scalar(o, "vegajet_prefix_cache_lookup_tokens_total", "counter", "Prompt tokens looked up in the prefix cache.", static_cast<double>(pl));
    scalar(o, "vegajet_prefix_cache_hit_ratio", "gauge", "Prefix-cache hit rate over all lookups.", ratio(ph, pl));
    scalar(o, "vegajet_moe_lru_hits_total", "counter", "MoE expert LRU cache hits.", static_cast<double>(mh));
    scalar(o, "vegajet_moe_lru_misses_total", "counter", "MoE expert LRU cache misses.", static_cast<double>(mm));
    scalar(o, "vegajet_moe_lru_hit_ratio", "gauge", "MoE expert LRU hit rate.", ratio(mh, mh + mm));
    scalar(o, "vegajet_spec_drafted_tokens_total", "counter", "Speculative draft tokens proposed.", static_cast<double>(sd));
    scalar(o, "vegajet_spec_accepted_tokens_total", "counter", "Speculative draft tokens accepted.", static_cast<double>(sa));
    scalar(o, "vegajet_spec_verify_steps_total", "counter", "Speculative verify steps.", static_cast<double>(ss));
    scalar(o, "vegajet_spec_mean_accept", "gauge", "Accepted draft tokens per verify step (mean_accept).", ratio(sa, ss));
    scalar(o, "vegajet_spec_acceptance_ratio", "gauge", "Accepted / drafted tokens.", ratio(sa, sd));
    if (hw) {
        family(o, "vegajet_gpu_temperature_celsius", "gauge", "GPU temperature (sysfs hwmon, read-only): edge, junction, mem (TMem).");
        for (const auto& g : *hw) {
            const std::string c = "card=\"" + std::to_string(g.card) + "\"";
            if (!std::isnan(g.temp_edge_c)) sample(o, "vegajet_gpu_temperature_celsius", c + ",sensor=\"edge\"", g.temp_edge_c);
            if (!std::isnan(g.temp_junction_c)) sample(o, "vegajet_gpu_temperature_celsius", c + ",sensor=\"junction\"", g.temp_junction_c);
            if (!std::isnan(g.temp_mem_c)) sample(o, "vegajet_gpu_temperature_celsius", c + ",sensor=\"mem\"", g.temp_mem_c);
        }
        family(o, "vegajet_gpu_power_watts", "gauge", "GPU package power (sysfs hwmon, read-only).");
        for (const auto& g : *hw)
            if (!std::isnan(g.power_w)) sample(o, "vegajet_gpu_power_watts", "card=\"" + std::to_string(g.card) + "\"", g.power_w);
        family(o, "vegajet_gpu_ras_errors_total", "counter", "RAS ECC error counts per block (sysfs, read-only): ue uncorrectable, ce correctable.");
        for (const auto& g : *hw)
            for (const auto& r : g.ras) {
                if (!r.present) continue;
                const std::string l = "card=\"" + std::to_string(g.card) + "\",block=\"" + r.block + "\"";
                sample(o, "vegajet_gpu_ras_errors_total", l + ",type=\"ue\"", static_cast<double>(r.ue));
                sample(o, "vegajet_gpu_ras_errors_total", l + ",type=\"ce\"", static_cast<double>(r.ce));
            }
    }
    return o;
}

std::string server_exposition(const serve::OpenAiServer::Stats& s) {
    std::string o;
    scalar(o, "vegajet_http_requests_total", "counter", "HTTP requests parsed.", static_cast<double>(s.requests));
    scalar(o, "vegajet_http_completions_total", "counter", "Chat / message completions answered.", static_cast<double>(s.completions));
    scalar(o, "vegajet_http_streamed_total", "counter", "Completions answered as SSE streams.", static_cast<double>(s.streamed));
    scalar(o, "vegajet_http_cancelled_total", "counter", "Generations stopped because the client went away.", static_cast<double>(s.cancelled));
    scalar(o, "vegajet_http_rejected_total", "counter", "Requests refused with 503 (queue full).", static_cast<double>(s.rejected));
    scalar(o, "vegajet_http_errors_total", "counter", "Generations that failed.", static_cast<double>(s.errors));
    return o;
}

serve::GenerateFn instrument(serve::GenerateFn inner, Registry& reg, JsonlLogger* log) {
    auto next_id = std::make_shared<std::atomic<uint64_t>>(1);
    return [inner = std::move(inner), &reg, log, next_id](const serve::GenerateRequest& req,
                                                          const serve::TokenSink& sink) -> serve::GenerateResult {
        RequestRecord rec;
        rec.id = next_id->fetch_add(1, std::memory_order_relaxed);
        reg.request_started();
        const auto t0 = Clock::now();
        Clock::time_point first{}, last{};
        uint32_t n = 0;
        const serve::TokenSink counted = [&](std::string_view piece) {
            last = Clock::now();
            if (n++ == 0) first = last;
            return sink(piece);
        };
        serve::GenerateResult res;
        auto finish = [&](bool threw) {
            const auto t1 = Clock::now();
            auto ms = [](Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };
            rec.prompt_tokens = res.prompt_tokens;
            rec.completion_tokens = res.completion_tokens ? res.completion_tokens : n;
            rec.ttft_ms = n ? ms(first - t0) : 0.0;
            rec.tpot_ms = n > 1 ? ms(last - first) / static_cast<double>(n - 1) : 0.0;
            rec.e2e_ms = ms(t1 - t0);
            if (threw || !res.error.empty()) {
                rec.status = 500;
                rec.finish = "error";
            } else {
                rec.finish = serve::finish_reason_name(res.finish);
                rec.cancelled = res.finish == serve::FinishReason::Cancelled;
                if (rec.cancelled) rec.status = 499;
            }
            reg.request_finished(rec);
            if (log) log->log(rec);
        };
        try {
            res = inner(req, counted);
        } catch (...) {
            finish(true);
            throw;
        }
        finish(false);
        return res;
    };
}

// ------------------------------------------------------------------------------------------------- parsing

bool parse_exposition(std::string_view text, std::vector<Sample>& out, std::string* err) {
    out.clear();
    size_t lineno = 0;
    auto fail = [&](const std::string& m) {
        if (err) *err = "line " + std::to_string(lineno) + ": " + m;
        return false;
    };
    for (size_t pos = 0; pos < text.size();) {
        size_t nl = text.find('\n', pos);
        if (nl == std::string_view::npos) nl = text.size();
        std::string_view line = text.substr(pos, nl - pos);
        pos = nl + 1;
        ++lineno;
        if (line.empty() || line[0] == '#') continue;
        Sample s;
        size_t i = 0;
        while (i < line.size() && line[i] != '{' && line[i] != ' ') ++i;
        s.name = std::string(line.substr(0, i));
        if (!valid_metric_name(s.name)) return fail("bad metric name '" + s.name + "'");
        if (i < line.size() && line[i] == '{') {
            ++i;
            while (true) {
                if (i >= line.size()) return fail("unterminated labels");
                if (line[i] == '}') {
                    ++i;
                    break;
                }
                size_t eq = line.find('=', i);
                if (eq == std::string_view::npos) return fail("label without =");
                const std::string key(line.substr(i, eq - i));
                if (!valid_label_name(key)) return fail("bad label name '" + key + "'");
                i = eq + 1;
                if (i >= line.size() || line[i] != '"') return fail("label value must be quoted");
                ++i;
                std::string val;
                while (i < line.size() && line[i] != '"') {
                    if (line[i] == '\\' && i + 1 < line.size()) {
                        const char e = line[i + 1];
                        val += e == 'n' ? '\n' : e;
                        i += 2;
                    } else {
                        val += line[i++];
                    }
                }
                if (i >= line.size()) return fail("unterminated label value");
                ++i;
                s.labels[key] = val;
                if (i < line.size() && line[i] == ',') ++i;
            }
        }
        if (i >= line.size() || line[i] != ' ') return fail("missing value");
        while (i < line.size() && line[i] == ' ') ++i;
        size_t e = line.find(' ', i);
        const std::string_view v = line.substr(i, e == std::string_view::npos ? std::string_view::npos : e - i);
        if (!parse_value(v, s.value)) return fail("bad value '" + std::string(v) + "'");
        out.push_back(std::move(s));
    }
    return true;
}

bool validate_exposition(std::string_view text, std::string* err) {
    std::vector<Sample> samples;
    if (!parse_exposition(text, samples, err)) return false;
    auto fail = [&](const std::string& m) {
        if (err) *err = m;
        return false;
    };
    std::map<std::string, std::string> types;
    std::set<std::string> helped, seen_family;
    size_t sample_idx = 0;
    // walk lines again for HELP / TYPE order
    for (size_t pos = 0; pos < text.size();) {
        size_t nl = text.find('\n', pos);
        if (nl == std::string_view::npos) nl = text.size();
        const std::string line(text.substr(pos, nl - pos));
        pos = nl + 1;
        if (line.rfind("# HELP ", 0) == 0) {
            const std::string name = line.substr(7, line.find(' ', 7) - 7);
            if (!helped.insert(name).second) return fail("duplicate HELP for " + name);
        } else if (line.rfind("# TYPE ", 0) == 0) {
            const size_t sp = line.find(' ', 7);
            if (sp == std::string::npos) return fail("TYPE without a type");
            const std::string name = line.substr(7, sp - 7), type = line.substr(sp + 1);
            if (type != "counter" && type != "gauge" && type != "histogram" && type != "summary" && type != "untyped")
                return fail("unknown type " + type);
            if (types.count(name)) return fail("duplicate TYPE for " + name);
            if (seen_family.count(name)) return fail("TYPE for " + name + " after its samples");
            types[name] = type;
        } else if (!line.empty() && line[0] != '#') {
            const Sample& s = samples[sample_idx++];
            std::string fam = s.name;
            for (const char* suf : {"_bucket", "_sum", "_count"}) {
                const size_t n = std::strlen(suf);
                if (fam.size() > n && fam.compare(fam.size() - n, n, suf) == 0 && types.count(fam.substr(0, fam.size() - n)) &&
                    types[fam.substr(0, fam.size() - n)] == "histogram") {
                    fam = fam.substr(0, fam.size() - n);
                    break;
                }
            }
            if (!types.count(fam)) return fail("sample " + s.name + " before its TYPE");
            seen_family.insert(fam);
            if (types[fam] == "counter" && !(s.value >= 0)) return fail("negative counter " + s.name);
        }
    }
    // histogram consistency per label set
    for (const auto& [fam, type] : types) {
        if (type != "histogram") continue;
        std::map<std::map<std::string, std::string>, std::vector<std::pair<double, double>>> buckets;
        std::map<std::map<std::string, std::string>, double> counts;
        for (const Sample& s : samples) {
            if (s.name == fam + "_bucket") {
                auto labels = s.labels;
                auto it = labels.find("le");
                if (it == labels.end()) return fail(fam + "_bucket without le");
                double le;
                if (!parse_value(it->second, le)) return fail("bad le " + it->second);
                labels.erase(it);
                buckets[labels].emplace_back(le, s.value);
            } else if (s.name == fam + "_count") {
                counts[s.labels] = s.value;
            }
        }
        for (const auto& [labels, b] : buckets) {
            for (size_t i = 1; i < b.size(); ++i)
                if (!(b[i].first > b[i - 1].first) || b[i].second < b[i - 1].second)
                    return fail(fam + ": buckets not ascending / cumulative");
            if (b.empty() || !std::isinf(b.back().first)) return fail(fam + ": last bucket is not le=\"+Inf\"");
            auto c = counts.find(labels);
            if (c == counts.end() || c->second != b.back().second) return fail(fam + ": +Inf bucket != _count");
        }
    }
    return true;
}

double find_sample(const std::vector<Sample>& s, std::string_view name, const std::map<std::string, std::string>& match) {
    for (const Sample& x : s) {
        if (x.name != name) continue;
        bool ok = true;
        for (const auto& [k, v] : match) {
            auto it = x.labels.find(k);
            if (it == x.labels.end() || it->second != v) {
                ok = false;
                break;
            }
        }
        if (ok) return x.value;
    }
    return NAN;
}

} // namespace vegajet::metrics
