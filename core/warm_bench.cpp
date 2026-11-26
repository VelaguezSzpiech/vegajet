// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/warm_bench.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace vegajet::warm {

namespace {

bool is_env_key(const std::string& k) {
    if (k.empty()) return false;
    if (!(std::isalpha(static_cast<unsigned char>(k[0])) || k[0] == '_')) return false;
    for (char c : k) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) return false;
    }
    return true;
}

std::string trim(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

} // namespace

bool make_decode_prompt(std::string_view context, const std::vector<int32_t>& seed,
                        uint32_t max_ctx, uint32_t timed_steps,
                        std::vector<int32_t>& prompt, std::string* err) {
    prompt.clear();
    if (err) err->clear();
    uint64_t n = 0;
    for (char c : context) {
        if (c < '0' || c > '9' || n > (UINT32_MAX - uint32_t(c - '0')) / 10u) {
            if (err) *err = "VEGAJET_BENCH_CONTEXT must be a decimal token count";
            return false;
        }
        n = n * 10u + uint32_t(c - '0');
    }
    if (n == 0) n = seed.size();
    if (seed.empty() || n + timed_steps > max_ctx || timed_steps == 0) {
        if (err) *err = "VEGAJET_BENCH_CONTEXT needs a nonempty prompt and prompt + timed decode <= max_ctx";
        return false;
    }
    prompt.resize(static_cast<size_t>(n));
    for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = seed[i % seed.size()];
    return true;
}

bool tokenize_line(const std::string& line, std::vector<std::string>& out, std::string* err) {
    out.clear();
    std::string cur;
    bool in_tok = false;
    char quote = 0;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (quote != 0) {
            if (c == quote) {
                quote = 0;
            } else if (c == '\\' && quote == '"' && i + 1 < line.size()) {
                cur.push_back(line[++i]);
            } else {
                cur.push_back(c);
            }
            continue;
        }
        if (c == '#') {
            if (!in_tok) break;
            cur.push_back(c);
            continue;
        }
        if (c == '\'' || c == '"') {
            quote = c;
            in_tok = true;
            continue;
        }
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (in_tok) {
                out.push_back(cur);
                cur.clear();
                in_tok = false;
            }
            continue;
        }
        if (c == ';') {
            if (in_tok) {
                out.push_back(cur);
                cur.clear();
                in_tok = false;
            }
            out.emplace_back(";");
            continue;
        }
        cur.push_back(c);
        in_tok = true;
    }
    if (quote != 0) {
        if (err) *err = "unterminated quote in warm command";
        return false;
    }
    if (in_tok) out.push_back(cur);
    return true;
}

bool parse_sweep_spec(const std::string& spec,
                      std::vector<std::vector<std::string>>& out_variants,
                      std::string* err) {
    out_variants.clear();
    std::vector<std::string> toks;
    if (!tokenize_line(spec, toks, err)) return false;
    std::vector<std::string> cur;
    for (const auto& t : toks) {
        if (t == ";") {
            if (!cur.empty()) {
                out_variants.push_back(std::move(cur));
                cur.clear();
            }
        } else {
            cur.push_back(t);
        }
    }
    if (!cur.empty()) out_variants.push_back(std::move(cur));
    if (out_variants.empty()) {
        if (err) *err = "sweep spec contains no variants";
        return false;
    }
    return true;
}

WarmCommand parse_warm_command(const std::string& line) {
    WarmCommand cmd;
    cmd.raw_line = trim(line);
    if (cmd.raw_line.empty() || cmd.raw_line[0] == '#') {
        cmd.kind = CommandKind::Empty;
        return cmd;
    }
    std::vector<std::string> toks;
    std::string err;
    if (!tokenize_line(cmd.raw_line, toks, &err)) {
        cmd.kind = CommandKind::Invalid;
        cmd.error = err;
        return cmd;
    }
    if (toks.empty()) {
        cmd.kind = CommandKind::Empty;
        return cmd;
    }
    const std::string& op = toks[0];
    if (op == "quit" || op == "exit") {
        cmd.kind = CommandKind::Quit;
    } else if (op == "status") {
        cmd.kind = CommandKind::Status;
    } else if (op == "help" || op == "?") {
        cmd.kind = CommandKind::Help;
    } else if (op == "reload") {
        cmd.kind = CommandKind::Reload;
        if (toks.size() > 2) {
            cmd.kind = CommandKind::Invalid;
            cmd.error = "usage: reload [module_name|all]";
        } else if (toks.size() == 2 && toks[1] != "all") {
            cmd.module_name = toks[1];
        }
    } else if (op == "run") {
        cmd.kind = CommandKind::Run;
        cmd.args.assign(toks.begin() + 1, toks.end());
    } else if (op == "parity") {
        cmd.kind = CommandKind::Parity;
        cmd.args.assign(toks.begin() + 1, toks.end());
    } else if (op == "sweep") {
        const size_t pos = cmd.raw_line.find("sweep");
        const std::string rest = (pos != std::string::npos) ? trim(cmd.raw_line.substr(pos + 5)) : "";
        if (!parse_sweep_spec(rest, cmd.variants, &err)) {
            cmd.kind = CommandKind::Invalid;
            cmd.error = err;
        } else {
            cmd.kind = CommandKind::Sweep;
        }
    } else {
        cmd.kind = CommandKind::Invalid;
        cmd.error = "unknown warm command: " + op;
    }
    return cmd;
}

void extract_env_overrides(const std::vector<std::string>& in_args,
                           std::vector<std::string>& out_cli_args,
                           std::vector<EnvAssignment>& out_env) {
    out_cli_args.clear();
    out_env.clear();
    for (const auto& a : in_args) {
        const size_t eq = a.find('=');
        if (eq != std::string::npos && !a.empty() && a[0] != '-') {
            const std::string k = a.substr(0, eq);
            if (is_env_key(k)) {
                out_env.push_back(EnvAssignment{k, a.substr(eq + 1)});
                continue;
            }
        }
        out_cli_args.push_back(a);
    }
}

ScopedEnvOverrides::ScopedEnvOverrides(const std::vector<EnvAssignment>& overrides) {
    saved_.reserve(overrides.size());
    for (const auto& ov : overrides) {
        const char* cur = std::getenv(ov.key.c_str());
        SavedVar sv;
        sv.key = ov.key;
        sv.had_value = (cur != nullptr);
        if (cur) sv.old_value = cur;
        saved_.push_back(std::move(sv));
        if (ov.value.empty()) {
            ::unsetenv(ov.key.c_str());
        } else {
            ::setenv(ov.key.c_str(), ov.value.c_str(), 1);
        }
    }
}

ScopedEnvOverrides::~ScopedEnvOverrides() {
    for (auto it = saved_.rbegin(); it != saved_.rend(); ++it) {
        if (it->had_value) {
            ::setenv(it->key.c_str(), it->old_value.c_str(), 1);
        } else {
            ::unsetenv(it->key.c_str());
        }
    }
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    return out;
}

std::string format_sweep_table(const std::vector<SweepVariantResult>& results) {
    std::ostringstream oss;
    oss << "\n=== VEGAJET WARM SWEEP A/B DELTA TABLE ===\n";
    char hdr[256];
    std::snprintf(hdr, sizeof(hdr), "%-3s %-34s %9s %9s %9s %8s %9s %6s %5s\n",
                  "#", "Variant", "Primary", "Delta%", "Raw128", "PP512", "Parity", "TMem", "ECC");
    oss << hdr;
    const double base_tps = (!results.empty() && results[0].primary_tps > 0.0) ? results[0].primary_tps : 0.0;
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        std::string lbl = r.label;
        if (lbl.size() > 34) lbl = lbl.substr(0, 31) + "...";
        char delta_s[24] = "   base";
        if (i > 0 && base_tps > 0.0 && r.primary_tps > 0.0) {
            const double pct = 100.0 * (r.primary_tps - base_tps) / base_tps;
            std::snprintf(delta_s, sizeof(delta_s), "%+7.2f%%", pct);
        } else if (i > 0) {
            std::snprintf(delta_s, sizeof(delta_s), "     n/a");
        }
        char par_s[24];
        if (r.lossless_total > 0) {
            std::snprintf(par_s, sizeof(par_s), "%u/%u", r.lossless_matched, r.lossless_total);
        } else {
            std::snprintf(par_s, sizeof(par_s), "%s", r.parity_ok ? "PASS" : "FAIL");
        }
        char ecc_s[24];
        std::snprintf(ecc_s, sizeof(ecc_s), "%llu/%llu",
                      static_cast<unsigned long long>(r.ecc_ue_delta),
                      static_cast<unsigned long long>(r.ecc_de_delta));
        char row[256];
        std::snprintf(row, sizeof(row), "%-3zu %-34s %9.2f %9s %9.2f %8.1f %9s %5.1fC %5s\n",
                      i, lbl.c_str(), r.primary_tps, delta_s, r.raw_tg128_tps, r.pp512_tps, par_s, r.tmem_max_c, ecc_s);
        oss << row;
    }
    return oss.str();
}

std::string format_sweep_json(const std::vector<SweepVariantResult>& results) {
    std::ostringstream oss;
    const double base_tps = (!results.empty() && results[0].primary_tps > 0.0) ? results[0].primary_tps : 0.0;
    oss << "{\"event\":\"sweep_summary\",\"variants\":[";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        const double delta_pct = (i > 0 && base_tps > 0.0 && r.primary_tps > 0.0)
                                     ? 100.0 * (r.primary_tps - base_tps) / base_tps
                                     : 0.0;
        if (i > 0) oss << ",";
        char buf[512];
        std::snprintf(buf, sizeof(buf),
                      "{\"idx\":%zu,\"label\":\"%s\",\"ok\":%s,\"parity_ok\":%s,"
                      "\"primary_tps\":%.2f,\"delta_pct\":%.2f,\"raw_tg128_tps\":%.2f,"
                      "\"spec_tg128_tps\":%.2f,\"pp512_tps\":%.2f,\"pp2048_tps\":%.2f,"
                      "\"disp_per_tok\":%u,\"lossless\":\"%u/%u\",\"first_diff\":%d,"
                      "\"tmem_c\":%.1f,\"ecc_ue\":%llu,\"ecc_de\":%llu}",
                      i, json_escape(r.label).c_str(), r.ok ? "true" : "false", r.parity_ok ? "true" : "false",
                      r.primary_tps, delta_pct, r.raw_tg128_tps, r.spec_tg128_tps, r.pp512_tps, r.pp2048_tps,
                      r.dispatches_per_tok, r.lossless_matched, r.lossless_total, r.first_diff,
                      r.tmem_max_c,
                      static_cast<unsigned long long>(r.ecc_ue_delta),
                      static_cast<unsigned long long>(r.ecc_de_delta));
        oss << buf;
    }
    oss << "]}";
    return oss.str();
}

} // namespace vegajet::warm
