// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/bench_json.hpp"

#include "core/doctor_rules.hpp"
#include "core/json_lite.hpp"

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>

namespace vegajet::benchjson {

namespace {

const char* kRasBlocks[6] = {"umc", "gfx", "hdp", "mmhub", "pcie_bif", "sdma"};


std::string spec_key(uint32_t tokens, const std::string& label) {
    std::string k;
    for (char c : label) {
        if (c == '=') continue;
        const char l = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (std::isalnum(static_cast<unsigned char>(l))) k += l;
        else if (!k.empty() && k.back() != '_') k += '_';
    }
    while (!k.empty() && k.back() == '_') k.pop_back();
    return tokens == 128 ? k : "tg" + std::to_string(tokens) + "_" + k;
}

Spec* find_spec(Record& r, const std::string& key) {
    for (auto& s : r.spec)
        if (s.key == key) return &s;
    return nullptr;
}

// ------------------------------------------------------------------ JSON writing
std::string num(double v) {
    if (!std::isfinite(v)) return "null";
    char b[40];
    std::snprintf(b, sizeof(b), "%.10g", v);
    return b;
}
template <class T> std::string opt(const std::optional<T>& v) { return v ? num(static_cast<double>(*v)) : "null"; }
std::string q(const std::string& s) { return json::quote(s); }

struct Obj {  // ordered JSON object builder
    std::vector<std::pair<std::string, std::string>> kv;
    Obj& add(const std::string& k, const std::string& raw) { kv.emplace_back(k, raw); return *this; }
    std::string str(int ind) const {
        if (kv.empty()) return "{}";
        const std::string pad(ind + 2, ' ');
        std::string o = "{\n";
        for (size_t i = 0; i < kv.size(); ++i) o += pad + q(kv[i].first) + ": " + kv[i].second + (i + 1 < kv.size() ? ",\n" : "\n");
        return o + std::string(ind, ' ') + "}";
    }
};

// ------------------------------------------------------------------ JSON reading helpers
const json::Value* at(const json::Value& v, const char* k) { return v.is_object() ? v.get(k) : nullptr; }
std::optional<double> gnum(const json::Value& v, const char* k) {
    const auto* x = at(v, k);
    return x && x->is_number() ? std::optional<double>(x->num) : std::nullopt;
}
std::string gstr(const json::Value& v, const char* k, const std::string& d = {}) {
    const auto* x = at(v, k);
    return x && x->is_string() ? x->str : d;
}

}  // namespace

namespace {

// Minimal scanner over one log line (no std::regex: GCC 13 warns inside it under sanitizer builds).
struct Scan {
    const std::string& s;
    size_t i = 0;
    bool find(const char* key) {  // moves past `key`
        const size_t p = s.find(key, i);
        if (p == std::string::npos) return false;
        i = p + std::strlen(key);
        return true;
    }
    bool lit(const char* t) {  // `t` must follow (after optional spaces)
        while (i < s.size() && s[i] == ' ') ++i;
        const size_t n = std::strlen(t);
        if (s.compare(i, n, t) != 0) return false;
        i += n;
        return true;
    }
    bool number(double& v) {
        while (i < s.size() && s[i] == ' ') ++i;
        const char* b = s.c_str() + i;
        char* e = nullptr;
        v = std::strtod(b, &e);
        if (e == b) return false;
        i += static_cast<size_t>(e - b);
        return true;
    }
    bool integer(long& v) {
        double d;
        const size_t at = i;
        if (!number(d) || d != std::floor(d)) { i = at; return false; }
        v = static_cast<long>(d);
        return true;
    }
    std::string word() {
        while (i < s.size() && s[i] == ' ') ++i;
        const size_t b = i;
        while (i < s.size() && (std::isalnum(static_cast<unsigned char>(s[i])) || s[i] == '_')) ++i;
        return s.substr(b, i - b);
    }
};

bool after_ratio(const std::string& line, const char* key, long& a, long& b) {  // key + "a/b"
    Scan c{line};
    return c.find(key) && c.integer(a) && c.lit("/") && c.integer(b);
}

}  // namespace

bool parse_line(const std::string& line, Record& r) {
    double x = 0;
    long n = 0, m = 0;
    if (line.find("[BENCH tg128 raw M=1]") != std::string::npos) {
        Scan c{line};
        c.find("[BENCH tg128 raw M=1]");
        double ms = 0;
        if (!c.number(x) || !c.lit("tok/s") || !c.lit("(") || !c.number(ms) || !c.lit("ms/tok")) return false;
        r.raw_tok_s = x;
        r.raw_ms_per_tok = ms;
        Scan e{line};
        double pct = 0, ceil = 0;
        if (e.find("eff_HBM2=") && e.number(x) && e.lit("GB/s") && e.lit("(") && e.number(pct) && e.lit("%") && e.lit("of") && e.number(ceil)) {
            r.eff_hbm2_gbs = x;
            r.hbm2_pct = pct;
            r.hbm2_ceiling_gbs = ceil;
        }
        Scan rr{line};
        if (rr.find("real_read=") && rr.number(x)) r.real_read_gb_per_tok = x;
        Scan d{line};
        if (d.find("disp/tok=") && d.integer(n) && n >= 0) r.dispatches_per_tok = static_cast<uint32_t>(n);
        return true;
    }
    if (line.find("[BENCH pp") != std::string::npos) {
        Scan c{line};
        c.find("[BENCH pp");
        if (!c.integer(n) || !c.find("M=") || !c.integer(m) || !c.lit("]") || !c.number(x) || !c.lit("tok/s")) return false;
        r.prefill["pp" + std::to_string(n) + "_" + r.prefill_path] = x;
        r.prefill_m["pp" + std::to_string(n)] = static_cast<uint32_t>(m);
        return true;
    }
    if (line.find("[BENCH tg") != std::string::npos) {
        Scan c{line};
        c.find("[BENCH tg");
        if (!c.integer(n) || n <= 0) return false;
        const size_t close = line.find(']', c.i);
        if (close == std::string::npos) return false;
        std::string label = line.substr(c.i, close - c.i);
        while (!label.empty() && label.front() == ' ') label.erase(0, 1);
        while (!label.empty() && label.back() == ' ') label.pop_back();
        c.i = close + 1;
        if (label.empty() || !c.number(x) || !c.lit("tok/s")) return false;
        Spec s;
        s.tokens = static_cast<uint32_t>(n);
        s.label = label;
        s.key = spec_key(s.tokens, label);
        s.tok_s = x;
        double sp = 0;
        const size_t at = c.i;
        if (c.lit("(") && c.number(sp) && c.lit("x vs raw")) s.speedup = sp;
        else c.i = at;
        long a = 0, b = 0;
        if (after_ratio(line, "lossless=", a, b)) {
            s.match = static_cast<uint32_t>(a);
            s.total = static_cast<uint32_t>(b);
            Scan f{line};
            long fd = 0;
            if (f.find("(first_diff=") && f.integer(fd)) s.first_diff = static_cast<int32_t>(fd);
        }
        Scan acc{line};
        if (acc.find("acc=") && acc.integer(a) && acc.lit("/") && acc.integer(b) && acc.lit("(") && acc.number(x) && acc.lit("%")) s.acceptance_pct = x;
        if (Spec* old = find_spec(r, s.key)) *old = s;
        else r.spec.push_back(s);
        return true;
    }
    if (line.find("[PARITY code") != std::string::npos && line.find("vs Tier-1 llama.cpp]") != std::string::npos) {
        Scan c{line};
        c.find("vs Tier-1 llama.cpp]");
        if (!c.integer(n) || !c.lit("/") || !c.integer(m) || !c.lit("exact greedy tokens")) return false;
        r.parity_match = static_cast<uint32_t>(n);
        r.parity_total = static_cast<uint32_t>(m);
        r.parity_verdict = c.lit("->") ? c.word() : std::string();
        return true;
    }
    if (line.find("[TELEM]") != std::string::npos && line.find("rolling avg=") != std::string::npos) {
        Scan c{line};
        double avg = 0, peak = 0, tmem = 0, lim = 0;
        if (!c.find("rolling avg=") || !c.number(avg)) return false;
        const size_t at = c.i;
        const bool has_lim = c.lit("W") && c.lit("(<=") && c.number(lim);
        if (!has_lim) c.i = at;
        if (!c.find("peak=") || !c.number(peak) || !c.find("max_TMem=") || !c.number(tmem) || !c.find("->")) return false;
        r.power_avg_w = avg;
        if (has_lim) r.power_avg_limit_w = lim;
        r.power_peak_w = peak;
        r.tmem_max_c = tmem;
        r.telem_verdict = c.word();
        return true;
    }
    if (line.find("[KLD dot8 prefill]") != std::string::npos) {
        Scan c{line};
        c.find("[KLD dot8 prefill]");
        if (!c.number(x)) return false;
        r.dot8_kld = x;
        return true;
    }
    return false;
}

void parse_log(const std::string& text, Record& r) {
    std::istringstream is(text);
    std::string line;
    while (std::getline(is, line)) parse_line(line, r);
}

std::string to_json(const Record& r) {
    Obj meta;
    meta.add("vegajet_version", q(r.vegajet_version)).add("commit", q(r.commit)).add("command", q(r.command))
        .add("model", q(r.model)).add("architecture", q(r.architecture)).add("tp", num(r.tp))
        .add("gpu_selection", q(r.gpu_selection)).add("prefill_path", q(r.prefill_path)).add("flags", q(r.flags))
        .add("executed", r.executed ? "true" : "false").add("plan_status", q(r.plan_status)).add("error", q(r.error))
        .add("timestamp_utc", q(r.timestamp_utc));
    Obj spec_tp, spec_detail, parity, first_diff;
    for (const auto& s : r.spec) {
        spec_tp.add(s.key + "_tok_s", num(s.tok_s));
        Obj d;
        d.add("label", q(s.label)).add("tokens", num(s.tokens)).add("speedup_vs_raw", opt(s.speedup)).add("acceptance_pct", opt(s.acceptance_pct));
        spec_detail.add(s.key, d.str(4));
        if (s.match && s.total) parity.add(s.key + "_lossless", Obj().add("match", num(*s.match)).add("total", num(*s.total)).str(4));
        if (s.first_diff) first_diff.add(s.key, num(*s.first_diff));
    }
    Obj tp;
    tp.add("raw_m1_tok_s", opt(r.raw_tok_s));
    for (const char* k : {"pp512_w4a8", "pp2048_w4a8", "pp512_dot8", "pp2048_dot8"}) {
        auto it = r.prefill.find(k);
        tp.add(std::string(k) + "_tok_s", it == r.prefill.end() ? "null" : num(it->second));
    }
    tp.add("spec", spec_tp.str(4));
    Obj dec;
    dec.add("raw_m1_ms_per_tok", opt(r.raw_ms_per_tok)).add("eff_hbm2_gbs", opt(r.eff_hbm2_gbs)).add("hbm2_pct_of_ceiling", opt(r.hbm2_pct))
        .add("hbm2_ceiling_gbs", opt(r.hbm2_ceiling_gbs)).add("real_read_gb_per_tok", opt(r.real_read_gb_per_tok))
        .add("dispatches_per_tok", opt(r.dispatches_per_tok));
    Obj pm;
    for (const auto& kv : r.prefill_m) pm.add(kv.first, num(kv.second));
    Obj par;
    if (r.parity_match && r.parity_total)
        par.add("code_vs_tier1", Obj().add("match", num(*r.parity_match)).add("total", num(*r.parity_total)).add("verdict", q(r.parity_verdict)).str(4));
    for (const auto& kv : parity.kv) par.add(kv.first, kv.second);
    Obj tel;
    tel.add("power_rolling_avg_w", opt(r.power_avg_w)).add("power_avg_limit_w", opt(r.power_avg_limit_w)).add("power_peak_w", opt(r.power_peak_w))
        .add("tmem_max_c", opt(r.tmem_max_c)).add("verdict", q(r.telem_verdict));
    std::string gpus = "[";
    const std::vector<GpuState>& last = r.gpus_after.empty() ? r.gpus_before : r.gpus_after;
    for (size_t i = 0; i < last.size(); ++i) {
        Obj g;
        g.add("id", q(last[i].id)).add("tmem_c", last[i].tmem_c >= 0 ? num(last[i].tmem_c) : "null")
            .add("power_cap_w", last[i].power_cap_w >= 0 ? num(last[i].power_cap_w) : "null");
        gpus += (i ? ", " : "") + g.str(4);
    }
    gpus += "]";
    Obj ras;  // deltas after - before; with one snapshot only, the absolute counters
    for (const auto& a : r.gpus_after.empty() ? r.gpus_before : r.gpus_after) {
        const GpuState* b = nullptr;
        if (!r.gpus_after.empty())
            for (const auto& x : r.gpus_before) if (x.id == a.id) b = &x;
        Obj g;
        for (const auto& blk : a.ras) {
            if (!blk.available) continue;
            uint64_t ue = blk.ue, de = blk.de, ce = blk.ce;
            if (b)
                for (const auto& o : b->ras)
                    if (o.block == blk.block && o.available) {
                        ue = blk.ue >= o.ue ? blk.ue - o.ue : 0;
                        de = blk.de >= o.de ? blk.de - o.de : 0;
                        ce = blk.ce >= o.ce ? blk.ce - o.ce : 0;
                    }
            g.add(blk.block, Obj().add("ue", num(static_cast<double>(ue))).add("de", num(static_cast<double>(de))).add("ce", num(static_cast<double>(ce))).str(6));
        }
        ras.add(a.id, g.str(4));
    }
    Obj root;
    root.add("schema", q(kSchema)).add("meta", meta.str(2)).add("throughput", tp.str(2)).add("decode", dec.str(2))
        .add("prefill_batch_m", pm.str(2)).add("speculative", spec_detail.str(2))
        .add("kld", Obj().add("dot8_prefill_kld", opt(r.dot8_kld)).str(2)).add("parity", par.str(2)).add("first_diff", first_diff.str(2))
        .add("spill", Obj().add("private_segment_bytes_max", opt(r.spill_bytes_max)).str(2)).add("telemetry", tel.str(2))
        .add("gpus", gpus).add("ras", ras.str(2))
        .add("ras_window", q(r.gpus_after.empty() ? "absolute" : "delta"));
    return root.str(0) + "\n";
}

bool from_json(const std::string& text, Record& r, std::string* err) {
    json::Value v;
    if (!json::parse(text, v) || !v.is_object()) { if (err) *err = "not a JSON object"; return false; }
    if (gstr(v, "schema") != kSchema) { if (err) *err = "schema is not " + std::string(kSchema); return false; }
    r = Record{};
    if (const auto* m = at(v, "meta")) {
        r.vegajet_version = gstr(*m, "vegajet_version", r.vegajet_version);
        r.commit = gstr(*m, "commit", r.commit);
        r.command = gstr(*m, "command", r.command);
        r.model = gstr(*m, "model");
        r.architecture = gstr(*m, "architecture");
        r.gpu_selection = gstr(*m, "gpu_selection", "all");
        r.prefill_path = gstr(*m, "prefill_path", "w4a8");
        r.flags = gstr(*m, "flags");
        r.plan_status = gstr(*m, "plan_status");
        r.error = gstr(*m, "error");
        r.timestamp_utc = gstr(*m, "timestamp_utc");
        if (auto t = gnum(*m, "tp")) r.tp = static_cast<uint32_t>(*t);
        if (const auto* e = at(*m, "executed"); e && e->is_bool()) r.executed = e->b;
    }
    const json::Value* spec_detail = at(v, "speculative");
    const json::Value* parity = at(v, "parity");
    const json::Value* fd = at(v, "first_diff");
    if (const auto* t = at(v, "throughput")) {
        r.raw_tok_s = gnum(*t, "raw_m1_tok_s");
        for (const char* k : {"pp512_w4a8", "pp2048_w4a8", "pp512_dot8", "pp2048_dot8"})
            if (auto x = gnum(*t, (std::string(k) + "_tok_s").c_str())) r.prefill[k] = *x;
        if (const auto* sp = at(*t, "spec"); sp && sp->is_object())
            for (const auto& kv : sp->obj) {
                if (!kv.second.is_number() || kv.first.size() < 7) continue;
                Spec s;
                s.key = kv.first.substr(0, kv.first.size() - 6);  // strip "_tok_s"
                s.tok_s = kv.second.num;
                if (const auto* d = spec_detail ? at(*spec_detail, s.key.c_str()) : nullptr) {
                    s.label = gstr(*d, "label");
                    if (auto n = gnum(*d, "tokens")) s.tokens = static_cast<uint32_t>(*n);
                    s.speedup = gnum(*d, "speedup_vs_raw");
                    s.acceptance_pct = gnum(*d, "acceptance_pct");
                }
                if (const auto* p = parity ? at(*parity, (s.key + "_lossless").c_str()) : nullptr) {
                    if (auto a = gnum(*p, "match")) s.match = static_cast<uint32_t>(*a);
                    if (auto b = gnum(*p, "total")) s.total = static_cast<uint32_t>(*b);
                }
                if (fd)
                    if (auto f = gnum(*fd, s.key.c_str())) s.first_diff = static_cast<int32_t>(*f);
                r.spec.push_back(s);
            }
    }
    if (const auto* d = at(v, "decode")) {
        r.raw_ms_per_tok = gnum(*d, "raw_m1_ms_per_tok");
        r.eff_hbm2_gbs = gnum(*d, "eff_hbm2_gbs");
        r.hbm2_pct = gnum(*d, "hbm2_pct_of_ceiling");
        r.hbm2_ceiling_gbs = gnum(*d, "hbm2_ceiling_gbs");
        r.real_read_gb_per_tok = gnum(*d, "real_read_gb_per_tok");
        if (auto x = gnum(*d, "dispatches_per_tok")) r.dispatches_per_tok = static_cast<uint32_t>(*x);
    }
    if (const auto* pm = at(v, "prefill_batch_m"); pm && pm->is_object())
        for (const auto& kv : pm->obj)
            if (kv.second.is_number()) r.prefill_m[kv.first] = static_cast<uint32_t>(kv.second.num);
    if (const auto* k = at(v, "kld")) r.dot8_kld = gnum(*k, "dot8_prefill_kld");
    if (const auto* c = parity ? at(*parity, "code_vs_tier1") : nullptr) {
        if (auto a = gnum(*c, "match")) r.parity_match = static_cast<uint32_t>(*a);
        if (auto b = gnum(*c, "total")) r.parity_total = static_cast<uint32_t>(*b);
        r.parity_verdict = gstr(*c, "verdict");
    }
    if (const auto* s = at(v, "spill"))
        if (auto x = gnum(*s, "private_segment_bytes_max")) r.spill_bytes_max = static_cast<uint64_t>(*x);
    if (const auto* t = at(v, "telemetry")) {
        r.power_avg_w = gnum(*t, "power_rolling_avg_w");
        r.power_avg_limit_w = gnum(*t, "power_avg_limit_w");
        r.power_peak_w = gnum(*t, "power_peak_w");
        r.tmem_max_c = gnum(*t, "tmem_max_c");
        r.telem_verdict = gstr(*t, "verdict");
    }
    // Stored RAS deltas come back as an "after" snapshot over a zero "before" (absolute counters as "before" only),
    // so to_json() reproduces them and merge() can add them.
    const bool delta = gstr(v, "ras_window") == "delta";
    std::map<std::string, std::pair<double, double>> gpu_info;
    if (const auto* g = at(v, "gpus"); g && g->is_array())
        for (const auto& e : g->arr) {
            auto t = gnum(e, "tmem_c"), p = gnum(e, "power_cap_w");
            gpu_info[gstr(e, "id")] = {t ? *t : -1.0, p ? *p : -1.0};
        }
    if (const auto* ras = at(v, "ras"); ras && ras->is_object())
        for (const auto& kv : ras->obj) {
            GpuState after, before;
            after.id = before.id = kv.first;
            if (gpu_info.count(kv.first)) { after.tmem_c = gpu_info[kv.first].first; after.power_cap_w = gpu_info[kv.first].second; }
            for (const auto& b : kv.second.obj) {
                RasBlock rb;
                rb.block = b.first;
                rb.available = true;
                if (auto x = gnum(b.second, "ue")) rb.ue = static_cast<uint64_t>(*x);
                if (auto x = gnum(b.second, "de")) rb.de = static_cast<uint64_t>(*x);
                if (auto x = gnum(b.second, "ce")) rb.ce = static_cast<uint64_t>(*x);
                after.ras.push_back(rb);
                RasBlock zero = rb;
                zero.ue = zero.de = zero.ce = 0;
                before.ras.push_back(zero);
            }
            r.gpus_after.push_back(after);
            if (delta) r.gpus_before.push_back(before);
        }
    if (!delta) { r.gpus_before = r.gpus_after; r.gpus_after.clear(); }
    return true;
}

void merge(Record& into, const Record& run) {
    auto take = [](auto& dst, const auto& src) { if (src) dst = src; };
    auto worst = [](std::optional<double>& dst, const std::optional<double>& src) { if (src && (!dst || *src > *dst)) dst = src; };
    into.executed = into.executed || run.executed;
    if (!run.flags.empty()) into.flags = into.flags.empty() ? run.flags : into.flags + " ; " + run.flags;
    if (run.prefill_path != into.prefill_path && into.prefill_path.find(run.prefill_path) == std::string::npos)
        into.prefill_path += "+" + run.prefill_path;
    if (!run.plan_status.empty()) into.plan_status = run.plan_status;
    if (!run.error.empty()) into.error = run.error;
    into.timestamp_utc = run.timestamp_utc;
    into.gpu_selection = run.gpu_selection;
    into.tp = run.tp;
    take(into.raw_tok_s, run.raw_tok_s); take(into.raw_ms_per_tok, run.raw_ms_per_tok); take(into.eff_hbm2_gbs, run.eff_hbm2_gbs);
    take(into.hbm2_pct, run.hbm2_pct); take(into.hbm2_ceiling_gbs, run.hbm2_ceiling_gbs); take(into.real_read_gb_per_tok, run.real_read_gb_per_tok);
    take(into.dispatches_per_tok, run.dispatches_per_tok); take(into.dot8_kld, run.dot8_kld); take(into.spill_bytes_max, run.spill_bytes_max);
    for (const auto& kv : run.prefill) into.prefill[kv.first] = kv.second;
    for (const auto& kv : run.prefill_m) into.prefill_m[kv.first] = kv.second;
    for (const auto& s : run.spec) {
        if (Spec* o = find_spec(into, s.key)) *o = s;
        else into.spec.push_back(s);
    }
    if (run.parity_match) { into.parity_match = run.parity_match; into.parity_total = run.parity_total; into.parity_verdict = run.parity_verdict; }
    worst(into.power_avg_w, run.power_avg_w); worst(into.power_peak_w, run.power_peak_w); worst(into.tmem_max_c, run.tmem_max_c);
    take(into.power_avg_limit_w, run.power_avg_limit_w);
    if (!run.telem_verdict.empty() && (into.telem_verdict.empty() || run.telem_verdict == "FAIL")) into.telem_verdict = run.telem_verdict;
    // RAS: fold this run's deltas (after - before) into the stored deltas. A stored absolute snapshot (no "after")
    // is replaced by the first delta; a run without an "after" snapshot adds nothing.
    if (run.gpus_after.empty()) return;
    auto delta_of = [&run](const GpuState& a) {
        GpuState d = a;
        for (auto& blk : d.ras)
            for (const auto& b : run.gpus_before)
                if (b.id == a.id)
                    for (const auto& o : b.ras)
                        if (o.block == blk.block) {
                            blk.ue -= std::min(blk.ue, o.ue);
                            blk.de -= std::min(blk.de, o.de);
                            blk.ce -= std::min(blk.ce, o.ce);
                        }
        return d;
    };
    auto zero_of = [](GpuState g) {
        for (auto& blk : g.ras) blk.ue = blk.de = blk.ce = 0;
        return g;
    };
    if (into.gpus_after.empty()) into.gpus_before.clear();
    for (const auto& a : run.gpus_after) {
        const GpuState d = delta_of(a);
        GpuState* dst = nullptr;
        for (auto& x : into.gpus_after) if (x.id == d.id) dst = &x;
        GpuState* zb = nullptr;
        for (auto& x : into.gpus_before) if (x.id == d.id) zb = &x;
        if (!dst) {
            into.gpus_after.push_back(d);
            if (!zb) into.gpus_before.push_back(zero_of(d));
            continue;
        }
        dst->tmem_c = std::max(dst->tmem_c, d.tmem_c);
        dst->power_cap_w = d.power_cap_w;
        for (const auto& blk : d.ras) {
            RasBlock* t = nullptr;
            for (auto& x : dst->ras) if (x.block == blk.block) t = &x;
            if (!t) {
                dst->ras.push_back(blk);
                if (zb) zb->ras.push_back(RasBlock{blk.block, 0, 0, 0, blk.available});
                continue;
            }
            t->ue += blk.ue;
            t->de += blk.de;
            t->ce += blk.ce;
            t->available = t->available || blk.available;
        }
    }
}

bool write_file(const std::string& path, const Record& r, std::string* err, bool* merged) {
    if (merged) *merged = false;
    Record out = r;
    {
        std::ifstream in(path);
        if (in) {
            std::stringstream ss;
            ss << in.rdbuf();
            Record old;
            if (from_json(ss.str(), old, nullptr) && old.model == r.model && old.commit == r.commit && old.architecture == r.architecture) {
                // Normalize this run to delta form first, then fold it into the stored scorecard.
                merge(old, r);
                out = old;
                if (merged) *merged = true;
            }
        }
    }
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::trunc);
        f << to_json(out);
        if (!f) { if (err) *err = "cannot write " + tmp; return false; }
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) { if (err) *err = "cannot rename " + tmp + " to " + path; return false; }
    return true;
}

std::vector<GpuState> snapshot(const std::string& root) {
    doctor_rules::Options o;
    o.root = root;
    o.power_samples = 1;
    o.service_active = [](const std::string&) { return false; };
    o.sleep_fn = [](uint32_t, uint32_t) {};
    const doctor_rules::Report rep = doctor_rules::run(o);
    std::vector<GpuState> out;
    for (size_t i = 0; i < rep.gpus.size(); ++i) {
        const auto& g = rep.gpus[i];
        GpuState s;
        s.id = "gpu" + std::to_string(i);
        s.tmem_c = g.tmem_c;
        s.power_cap_w = g.power_cap_w;
        for (const char* blk : kRasBlocks) {
            RasBlock b;
            b.block = blk;
            for (const auto& rc : g.ras)
                if (rc.block == blk) { b.ue = rc.ue; b.de = rc.de; b.ce = rc.ce; b.available = rc.available; }
            s.ras.push_back(b);
        }
        out.push_back(s);
    }
    return out;
}

std::string sanitize_flags(const std::vector<std::string>& args) {
    std::string o;
    for (const auto& a : args) {
        const size_t s = a.find_last_of('/');
        o += (o.empty() ? "" : " ") + (s == std::string::npos ? a : a.substr(s + 1));
    }
    return o;
}

std::string utc_now() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char b[32];
    std::strftime(b, sizeof(b), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return b;
}

StdoutTee::~StdoutTee() {
    if (saved_ >= 0) end();
}

bool StdoutTee::begin() {
    if (saved_ >= 0) return false;
    std::fflush(stdout);
    int p[2];
    if (::pipe(p) != 0) return false;
    saved_ = ::dup(1);
    if (saved_ < 0 || ::dup2(p[1], 1) < 0) {
        ::close(p[0]);
        ::close(p[1]);
        if (saved_ >= 0) ::close(saved_);
        saved_ = -1;
        return false;
    }
    ::close(p[1]);
    rd_ = p[0];
    buf_.clear();
    const int out = saved_;
    th_ = std::thread([this, out] {
        char b[8192];
        for (ssize_t n; (n = ::read(rd_, b, sizeof(b))) > 0;) {
            buf_.append(b, static_cast<size_t>(n));
            for (ssize_t off = 0; off < n;) {
                const ssize_t w = ::write(out, b + off, static_cast<size_t>(n - off));
                if (w <= 0) break;
                off += w;
            }
        }
    });
    return true;
}

std::string StdoutTee::end() {
    if (saved_ < 0) return buf_;
    std::fflush(stdout);
    ::dup2(saved_, 1);  // closes the pipe's last write end: the reader sees EOF
    th_.join();
    ::close(rd_);
    ::close(saved_);
    saved_ = rd_ = -1;
    return buf_;
}

}  // namespace vegajet::benchjson
