// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/hw_profile.hpp"

#include "core/json_lite.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>

namespace vegajet::hw {

namespace {

std::string lower(std::string s) {
    for (char& c : s) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    return s;
}

// JSON numbers arrive as double. Accept only finite, non-negative, integral values that fit uint64.
bool as_u64(const json::Value& v, uint64_t& out) {
    if (!v.is_number()) return false;
    const double d = v.num;
    if (!std::isfinite(d) || d < 0.0 || d > 9.0e18) return false;
    if (std::floor(d) != d) return false;
    out = static_cast<uint64_t>(d);
    return true;
}

bool want_num(const json::Value& o, const char* key, double& v) {
    const json::Value* f = o.get(key);
    if (!f || !f->is_number() || !std::isfinite(f->num)) return false;
    v = f->num;
    return true;
}

bool want_u64(const json::Value& o, const char* key, uint64_t& v) {
    const json::Value* f = o.get(key);
    return f && as_u64(*f, v);
}

bool want_str(const json::Value& o, const char* key, std::string& v) {
    const json::Value* f = o.get(key);
    if (!f || !f->is_string()) return false;
    v = f->str;
    return true;
}

bool set_err(std::string* err, const std::string& msg) {
    if (err) *err = msg;
    return false;
}

std::string num(double v) {
    char b[40];
    std::snprintf(b, sizeof(b), "%.10g", v);
    return b;
}

} // namespace

const char* mem_kind_name(MemKind m) {
    switch (m) {
        case MemKind::HBM2: return "hbm2";
        case MemKind::HBM2e: return "hbm2e";
        case MemKind::GDDR6: return "gddr6";
        case MemKind::DDR4: return "ddr4";
        case MemKind::Unknown: break;
    }
    return "unknown";
}

bool mem_kind_from_name(const std::string& s, MemKind& out) {
    const std::string v = lower(s);
    if (v == "hbm2") out = MemKind::HBM2;
    else if (v == "hbm2e") out = MemKind::HBM2e;
    else if (v == "gddr6") out = MemKind::GDDR6;
    else if (v == "ddr4") out = MemKind::DDR4;
    else if (v == "unknown" || v.empty()) out = MemKind::Unknown;
    else return false;
    return true;
}

const char* link_kind_name(LinkKind k) {
    switch (k) {
        case LinkKind::None: return "none";
        case LinkKind::Xgmi: return "xgmi";
        case LinkKind::Pcie: return "pcie";
        case LinkKind::Unknown: break;
    }
    return "unknown";
}

bool link_kind_from_name(const std::string& s, LinkKind& out) {
    const std::string v = lower(s);
    if (v == "none" || v == "self") out = LinkKind::None;
    else if (v == "xgmi") out = LinkKind::Xgmi;
    else if (v == "pcie") out = LinkKind::Pcie;
    else if (v == "unknown" || v.empty()) out = LinkKind::Unknown;
    else return false;
    return true;
}

const Gpu* Profile::find(uint32_t index) const {
    for (const auto& g : gpus) if (g.index == index) return &g;
    return nullptr;
}

P2pLink Profile::link(size_t from, size_t to) const {
    if (from >= gpus.size() || to >= gpus.size() || from == to) return P2pLink{};
    const auto& row = gpus[from].p2p;
    if (to >= row.size()) return P2pLink{};
    return row[to];
}

double Profile::aggregate_bw_gbs() const {
    double s = 0.0;
    for (const auto& g : gpus) s += g.bw_gbs;
    return s;
}

double Profile::min_bw_gbs() const {
    if (gpus.empty()) return 0.0;
    double m = std::numeric_limits<double>::infinity();
    for (const auto& g : gpus) m = std::min(m, g.bw_gbs);
    return m;
}

double Profile::max_bw_gbs() const {
    double m = 0.0;
    for (const auto& g : gpus) m = std::max(m, g.bw_gbs);
    return m;
}

uint64_t Profile::total_vram_bytes() const {
    uint64_t s = 0;
    for (const auto& g : gpus) {
        if (g.vram_bytes > std::numeric_limits<uint64_t>::max() - s) return std::numeric_limits<uint64_t>::max();
        s += g.vram_bytes;
    }
    return s;
}

uint64_t Profile::min_vram_bytes() const {
    if (gpus.empty()) return 0;
    uint64_t m = std::numeric_limits<uint64_t>::max();
    for (const auto& g : gpus) m = std::min(m, g.vram_bytes);
    return m;
}

void Profile::normalize() {
    const size_t n = gpus.size();
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [this](size_t a, size_t b) { return gpus[a].index < gpus[b].index; });
    if (n > 0) {
        std::vector<std::vector<P2pLink>> rows(n);
        for (size_t i = 0; i < n; ++i) rows[i] = gpus[i].p2p;
        std::vector<Gpu> sorted(n);
        for (size_t j = 0; j < n; ++j) sorted[j] = gpus[order[j]];
        gpus.swap(sorted);
        // Rows are positional, so they move with the GPU they describe and are re-indexed by the new order.
        for (size_t j = 0; j < n; ++j) {
            Gpu& g = gpus[j];
            const size_t src = order[j];
            g.p2p.assign(n, P2pLink{});
            if (rows[src].size() == n) {
                for (size_t k = 0; k < n; ++k) g.p2p[k] = rows[src][order[k]];
            }
            if (g.id.empty()) g.id = "gpu" + std::to_string(g.index);
            for (auto& l : g.p2p) {
                if (!std::isfinite(l.gbs) || l.gbs < 0.0) l.gbs = 0.0;
                if (!std::isfinite(l.latency_us) || l.latency_us < 0.0) l.latency_us = 0.0;
            }
        }
    }
    for (size_t i = 0; i < n; ++i) gpus[i].p2p[i] = P2pLink{};
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            P2pLink& a = gpus[i].p2p[j];
            P2pLink& b = gpus[j].p2p[i];
            // An explicit "no direct peer" is authoritative for the pair.
            if (a.kind == LinkKind::None || b.kind == LinkKind::None) {
                a = P2pLink{};
                a.kind = LinkKind::None;
                b = a;
                continue;
            }
            if (a.kind == LinkKind::Unknown && b.kind == LinkKind::Unknown) continue;
            if (a.kind == LinkKind::Unknown) { a = b; continue; }
            if (b.kind == LinkKind::Unknown) { b = a; continue; }
            const LinkKind kind = (a.kind == LinkKind::Xgmi || b.kind == LinkKind::Xgmi) ? LinkKind::Xgmi : LinkKind::Pcie;
            double gbs = 0.0;
            if (a.gbs > 0.0 && b.gbs > 0.0) gbs = std::min(a.gbs, b.gbs);
            else gbs = std::max(a.gbs, b.gbs);
            const double lat = std::max(a.latency_us, b.latency_us);
            a.kind = b.kind = kind;
            a.gbs = b.gbs = gbs;
            a.latency_us = b.latency_us = lat;
        }
    }
}

bool Profile::valid(std::string* why) const {
    if (why) why->clear();
    if (schema != kSchemaVersion) {
        return set_err(why, "unsupported schema " + std::to_string(schema) + " (expected " +
                                std::to_string(kSchemaVersion) + ")");
    }
    if (gpus.empty()) return set_err(why, "profile contains no GPUs");
    if (gpus.size() > kMaxGpus) {
        return set_err(why, "profile has " + std::to_string(gpus.size()) + " GPUs, at most " +
                                std::to_string(kMaxGpus) + " are supported");
    }
    for (size_t i = 0; i < gpus.size(); ++i) {
        const Gpu& g = gpus[i];
        const std::string tag = "gpu " + std::to_string(i) + ": ";
        if (g.cu == 0 || g.cu > 512) return set_err(why, tag + "invalid CU count " + std::to_string(g.cu));
        if (g.vram_bytes == 0) return set_err(why, tag + "zero VRAM capacity");
        if (g.bus_bits == 0 || g.bus_bits > 65536) return set_err(why, tag + "invalid memory bus width");
        if (!std::isfinite(g.mem_clock_mhz) || g.mem_clock_mhz <= 0.0)
            return set_err(why, tag + "invalid sustained memory clock");
        if (!std::isfinite(g.sclk_ghz) || g.sclk_ghz <= 0.0) return set_err(why, tag + "invalid sustained SCLK");
        if (!std::isfinite(g.bw_gbs) || g.bw_gbs <= 0.0) return set_err(why, tag + "invalid streaming bandwidth");
        if (!std::isfinite(g.power_cap_w) || g.power_cap_w < 0.0) return set_err(why, tag + "invalid power cap");
        if (!std::isfinite(g.eta) || g.eta <= 0.0 || g.eta > 1.0) return set_err(why, tag + "invalid eta");
        if (!g.p2p.empty() && g.p2p.size() != gpus.size())
            return set_err(why, tag + "P2P row has " + std::to_string(g.p2p.size()) + " entries for " +
                                     std::to_string(gpus.size()) + " GPUs");
        for (size_t j = 0; j < g.p2p.size(); ++j) {
            const P2pLink& l = g.p2p[j];
            if (!std::isfinite(l.gbs) || l.gbs < 0.0 || !std::isfinite(l.latency_us) || l.latency_us < 0.0)
                return set_err(why, tag + "invalid P2P link to gpu " + std::to_string(j));
            if (i != j && l.direct() && l.gbs <= 0.0)
                return set_err(why, tag + "direct P2P link to gpu " + std::to_string(j) + " without bandwidth");
        }
    }
    for (size_t i = 0; i < gpus.size(); ++i) {
        for (size_t j = i + 1; j < gpus.size(); ++j) {
            if (gpus[i].index == gpus[j].index)
                return set_err(why, "duplicate GPU index " + std::to_string(gpus[i].index));
        }
    }
    return true;
}

std::string to_json(const Profile& in) {
    Profile p = in;
    p.normalize();
    std::ostringstream o;
    o << "{\n";
    o << "  \"schema\": " << p.schema << ",\n";
    o << "  \"arch\": " << json::quote(p.arch) << ",\n";
    o << "  \"source\": " << json::quote(p.source) << ",\n";
    o << "  \"generated_at\": " << json::quote(p.generated_at) << ",\n";
    o << "  \"measured\": " << (p.measured ? "true" : "false") << ",\n";
    o << "  \"gpus\": [";
    for (size_t i = 0; i < p.gpus.size(); ++i) {
        const Gpu& g = p.gpus[i];
        o << (i == 0 ? "\n" : ",\n");
        o << "    {\"index\": " << g.index;
        o << ", \"id\": " << json::quote(g.id);
        o << ", \"bdf\": " << json::quote(g.bdf);
        o << ", \"cu\": " << g.cu << ", \"n_cu\": " << g.cu;
        o << ", \"vram_bytes\": " << g.vram_bytes;
        o << ", \"mem\": " << json::quote(mem_kind_name(g.mem));
        o << ", \"bus_bits\": " << g.bus_bits;
        o << ", \"mem_clock_mhz\": " << num(g.mem_clock_mhz) << ", \"mclk_mhz\": " << num(g.mem_clock_mhz);
        o << ", \"sclk_ghz\": " << num(g.sclk_ghz);
        o << ", \"bw_gbs\": " << num(g.bw_gbs) << ", \"bw_hbm_gbs\": " << num(g.bw_gbs);
        o << ", \"power_cap_w\": " << num(g.power_cap_w);
        o << ", \"eta\": " << num(g.eta);
        o << ", \"p2p\": [";
        for (size_t j = 0; j < g.p2p.size(); ++j) {
            const P2pLink& l = g.p2p[j];
            o << (j == 0 ? "" : ", ");
            o << "{\"kind\": " << json::quote(link_kind_name(l.kind)) << ", \"gbs\": " << num(l.gbs)
              << ", \"latency_us\": " << num(l.latency_us) << "}";
        }
        o << "]}";
    }
    if (!p.gpus.empty()) o << "\n";
    o << "  ]\n}\n";
    return o.str();
}

bool from_json(const std::string& text, Profile& out, std::string* err) {
    json::Value root;
    if (!json::parse(text, root)) return set_err(err, "invalid JSON text");
    if (!root.is_object()) return set_err(err, "top-level JSON value is not an object");

    const json::Value* sv = root.get("schema");
    uint64_t schema = 0;
    if (!sv || !as_u64(*sv, schema)) return set_err(err, "missing or invalid numeric \"schema\"");
    if (schema != kSchemaVersion) {
        return set_err(err, "unsupported schema " + std::to_string(schema) + " (expected " +
                                std::to_string(kSchemaVersion) + ")");
    }

    Profile p;
    p.schema = static_cast<uint32_t>(schema);
    const json::Value* f = root.get("arch");
    if (f && (!f->is_string())) return set_err(err, "invalid \"arch\"");
    if (f) p.arch = f->str;
    f = root.get("source");
    if (f && (!f->is_string())) return set_err(err, "invalid \"source\"");
    if (f) p.source = f->str;
    f = root.get("generated_at");
    if (f && (!f->is_string())) return set_err(err, "invalid \"generated_at\"");
    if (f) p.generated_at = f->str;
    f = root.get("measured");
    if (f && (!f->is_bool())) return set_err(err, "invalid \"measured\"");
    if (f) p.measured = f->b;

    const json::Value* ga = root.get("gpus");
    if (!ga || !ga->is_array() || ga->arr.empty()) return set_err(err, "missing non-empty \"gpus\" array");
    if (ga->arr.size() > kMaxGpus) {
        return set_err(err, "profile has " + std::to_string(ga->arr.size()) + " GPUs, at most " +
                                std::to_string(kMaxGpus) + " are supported");
    }

    for (size_t i = 0; i < ga->arr.size(); ++i) {
        const json::Value& e = ga->arr[i];
        const std::string tag = "gpu " + std::to_string(i) + ": ";
        if (!e.is_object()) return set_err(err, tag + "entry is not an object");
        Gpu g;
        g.index = static_cast<uint32_t>(i);
        const json::Value* v = e.get("index");
        if (v) {
            uint64_t x = 0;
            if (!as_u64(*v, x) || x > std::numeric_limits<uint32_t>::max())
                return set_err(err, tag + "invalid \"index\"");
            g.index = static_cast<uint32_t>(x);
        }
        if (e.get("id")) {
            if (!want_str(e, "id", g.id)) return set_err(err, tag + "invalid \"id\"");
        }
        if (e.get("bdf")) {
            if (!want_str(e, "bdf", g.bdf)) return set_err(err, tag + "invalid \"bdf\"");
        }
        uint64_t u = 0;
        const bool has_cu = e.get("cu") != nullptr;
        if (!want_u64(e, has_cu ? "cu" : "n_cu", u) || u > std::numeric_limits<uint32_t>::max())
            return set_err(err, tag + "missing or invalid CU count");
        g.cu = static_cast<uint32_t>(u);
        if (!want_u64(e, "vram_bytes", g.vram_bytes)) return set_err(err, tag + "missing or invalid \"vram_bytes\"");
        if (e.get("mem")) {
            std::string s;
            if (!want_str(e, "mem", s) || !mem_kind_from_name(s, g.mem))
                return set_err(err, tag + "invalid \"mem\"");
        }
        if (e.get("bus_bits")) {
            if (!want_u64(e, "bus_bits", u) || u > std::numeric_limits<uint32_t>::max())
                return set_err(err, tag + "invalid \"bus_bits\"");
            g.bus_bits = static_cast<uint32_t>(u);
        }
        const bool has_mclk = e.get("mem_clock_mhz") != nullptr;
        if (!want_num(e, has_mclk ? "mem_clock_mhz" : "mclk_mhz", g.mem_clock_mhz))
            return set_err(err, tag + "missing or invalid sustained memory clock");
        if (!want_num(e, "sclk_ghz", g.sclk_ghz)) return set_err(err, tag + "missing or invalid \"sclk_ghz\"");
        const bool has_bw = e.get("bw_gbs") != nullptr;
        if (!want_num(e, has_bw ? "bw_gbs" : "bw_hbm_gbs", g.bw_gbs))
            return set_err(err, tag + "missing or invalid streaming bandwidth");
        if (e.get("power_cap_w") && !want_num(e, "power_cap_w", g.power_cap_w))
            return set_err(err, tag + "invalid \"power_cap_w\"");
        if (e.get("eta") && !want_num(e, "eta", g.eta)) return set_err(err, tag + "invalid \"eta\"");
        const json::Value* pa = e.get("p2p");
        if (pa) {
            if (!pa->is_array() || pa->arr.size() != ga->arr.size())
                return set_err(err, tag + "\"p2p\" must be an array with one entry per GPU");
            g.p2p.resize(pa->arr.size());
            for (size_t j = 0; j < pa->arr.size(); ++j) {
                const json::Value& le = pa->arr[j];
                if (!le.is_object()) return set_err(err, tag + "P2P entry " + std::to_string(j) + " is not an object");
                P2pLink l;
                if (le.get("kind")) {
                    std::string s;
                    if (!want_str(le, "kind", s) || !link_kind_from_name(s, l.kind))
                        return set_err(err, tag + "invalid P2P kind at " + std::to_string(j));
                }
                if (le.get("gbs") && !want_num(le, "gbs", l.gbs))
                    return set_err(err, tag + "invalid P2P bandwidth at " + std::to_string(j));
                if (le.get("latency_us") && !want_num(le, "latency_us", l.latency_us))
                    return set_err(err, tag + "invalid P2P latency at " + std::to_string(j));
                g.p2p[j] = l;
            }
        }
        p.gpus.push_back(std::move(g));
    }

    p.normalize();
    std::string why;
    if (!p.valid(&why)) return set_err(err, "invalid profile: " + why);
    out = std::move(p);
    return true;
}

bool load(const std::string& path, Profile& out, std::string* err) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return set_err(err, "cannot open hardware profile cache");
    const std::string text((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    return from_json(text, out, err);
}

bool save(const std::string& path, const Profile& p, std::string* err) {
    namespace fs = std::filesystem;
    std::string why;
    if (!p.valid(&why)) return set_err(err, "refusing to write an invalid profile: " + why);
    std::error_code ec;
    const fs::path fp(path);
    if (fp.has_parent_path()) {
        fs::create_directories(fp.parent_path(), ec);
        if (ec) return set_err(err, "cannot create the cache directory");
    }
    const std::string tmp = path + ".tmp";
    {
        std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
        if (!ofs.is_open()) return set_err(err, "cannot open the cache file for writing");
        const std::string text = to_json(p);
        ofs.write(text.data(), static_cast<std::streamsize>(text.size()));
        ofs.flush();
        if (!ofs.good()) {
            ofs.close();
            std::error_code rec;
            fs::remove(tmp, rec);
            return set_err(err, "short write to the cache file");
        }
    }
    fs::rename(tmp, fp, ec);
    if (ec) {
        std::error_code rec;
        fs::remove(tmp, rec);
        return set_err(err, "cannot replace the cache file");
    }
    return true;
}

std::string default_path() {
    const char* home = std::getenv("HOME");
    if (!home || !home[0]) return "/tmp/vegajet_hw_profile.json";
    return std::string(home) + "/.config/vegajet/hw_profile.json";
}

double estimate_stream_gbs(double mem_clock_mhz, uint32_t bus_bits) {
    if (!std::isfinite(mem_clock_mhz) || mem_clock_mhz <= 0.0 || bus_bits == 0) return 0.0;
    // DDR transfer: two beats per clock. Peak bytes/s = mclk * 2 * bus/8; sustained fraction applies.
    return mem_clock_mhz * 2.0 * (static_cast<double>(bus_bits) / 8.0) / 1000.0 * kEstimatedPeakFraction;
}

const char* calibration_status_name(CalibrationStatus s) {
    switch (s) {
        case CalibrationStatus::Cached: return "cached";
        case CalibrationStatus::Measured: return "measured";
        case CalibrationStatus::Estimated: return "estimated";
        case CalibrationStatus::Failed: break;
    }
    return "failed";
}

CalibrationResult calibrate_and_cache(MeasurementBackend& backend, const CalibrationConfig& cfg) {
    CalibrationResult r;
    r.cache_path = cfg.cache_path.empty() ? default_path() : cfg.cache_path;

    std::vector<Gpu> gpus;
    std::string why;
    if (!backend.topology(gpus, &why)) {
        r.detail = "topology probe failed";
        if (!why.empty()) r.detail += ": " + why;
        return r;
    }
    if (gpus.empty()) {
        r.detail = "topology probe returned no GPUs";
        return r;
    }
    const uint32_t max_gpus = cfg.max_gpus == 0 ? kMaxGpus : std::min(cfg.max_gpus, kMaxGpus);
    if (gpus.size() > max_gpus) gpus.resize(max_gpus);
    for (size_t i = 0; i < gpus.size(); ++i) {
        const std::string tag = "gpu " + std::to_string(i) + ": ";
        if (gpus[i].cu == 0) {
            r.detail = tag + "topology reported no CU count";
            return r;
        }
        if (gpus[i].vram_bytes == 0) {
            r.detail = tag + "topology reported zero VRAM";
            return r;
        }
        if (gpus[i].id.empty()) gpus[i].id = "gpu" + std::to_string(gpus[i].index);
    }

    std::vector<std::string> notes;
    std::string link_err;
    if (!backend.links(gpus, &link_err)) {
        notes.push_back("link probe failed");
        if (!link_err.empty()) notes.back() += ": " + link_err;
    }

    bool all_measured = true;
    for (size_t i = 0; i < gpus.size(); ++i) {
        double gbs = 0.0;
        std::string merr;
        const bool sane = backend.measure_stream(gpus[i], gbs, &merr) && std::isfinite(gbs) &&
                          gbs >= cfg.expected_min_bw_gbs && gbs <= cfg.expected_max_bw_gbs;
        if (sane) {
            gpus[i].bw_gbs = gbs;
            continue;
        }
        all_measured = false;
        std::string reason = merr.empty() ? "streaming measurement out of range" : merr;
        if (!cfg.allow_estimate) {
            r.detail = "gpu " + std::to_string(i) + ": " + reason;
            return r;
        }
        if (!(gpus[i].mem_clock_mhz > 0.0)) {
            r.detail = "gpu " + std::to_string(i) + ": " + reason + "; no memory clock for an estimate";
            return r;
        }
        gpus[i].bw_gbs = estimate_stream_gbs(gpus[i].mem_clock_mhz, gpus[i].bus_bits);
        notes.push_back("gpu " + std::to_string(i) + ": " + reason + "; estimated " + num(gpus[i].bw_gbs) +
                        " GB/s");
    }

    Profile p;
    p.schema = kSchemaVersion;
    p.arch = backend.arch();
    p.source = all_measured ? "measured" : "estimated";
    p.generated_at = cfg.generated_at;
    p.measured = all_measured;
    p.gpus = std::move(gpus);
    p.normalize();

    std::string vwhy;
    if (!p.valid(&vwhy)) {
        r.profile = std::move(p);
        r.detail = "calibrated profile is invalid: " + vwhy;
        return r;
    }

    r.status = all_measured ? CalibrationStatus::Measured : CalibrationStatus::Estimated;
    r.profile = p;
    std::string serr;
    if (!save(r.cache_path, p, &serr)) notes.push_back(serr);
    else r.wrote_cache = true;

    r.detail = std::to_string(p.gpus.size()) + " GPUs, " +
               (all_measured ? std::string("all streaming measurements accepted")
                             : std::string("some streaming values estimated"));
    for (const auto& n : notes) r.detail += "; " + n;
    return r;
}

CalibrationResult load_or_calibrate(MeasurementBackend& backend, const CalibrationConfig& cfg) {
    const std::string path = cfg.cache_path.empty() ? default_path() : cfg.cache_path;
    std::string cache_why;
    if (!cfg.force) {
        Profile cached;
        if (load(path, cached, &cache_why)) {
            CalibrationResult r;
            r.status = CalibrationStatus::Cached;
            r.profile = std::move(cached);
            r.cache_path = path;
            r.detail = "cached schema=" + std::to_string(r.profile.schema) + " gpus=" +
                       std::to_string(r.profile.size()) + " measured=" + (r.profile.measured ? "yes" : "no");
            return r;
        }
    }
    CalibrationResult r = calibrate_and_cache(backend, cfg);
    if (!cache_why.empty() && r.status != CalibrationStatus::Failed) r.detail += " (cache: " + cache_why + ")";
    return r;
}

std::string format_profile(const Profile& p) {
    std::ostringstream o;
    o << "hw_profile schema=" << p.schema << " arch=" << p.arch << " gpus=" << p.gpus.size()
      << " source=" << p.source << " measured=" << (p.measured ? "yes" : "no") << "\n";
    char b[320];
    for (const auto& g : p.gpus) {
        std::snprintf(b, sizeof(b),
                      "  %s cu=%u vram=%.2f GiB mem=%s bus=%u mclk=%.0f MHz sclk=%.3f GHz bw=%.2f GB/s "
                      "cap=%.0f W eta=%.2f\n",
                      g.id.empty() ? "gpu" : g.id.c_str(), g.cu,
                      static_cast<double>(g.vram_bytes) / (1024.0 * 1024.0 * 1024.0), mem_kind_name(g.mem),
                      g.bus_bits, g.mem_clock_mhz, g.sclk_ghz, g.bw_gbs, g.power_cap_w, g.eta);
        o << b;
    }
    o << "  p2p:";
    for (size_t i = 0; i < p.gpus.size(); ++i) {
        for (size_t j = 0; j < p.gpus.size(); ++j) {
            if (i == j) continue;
            const P2pLink l = p.link(i, j);
            std::snprintf(b, sizeof(b), " %zu->%zu=%s/%.1f GB/s", i, j, link_kind_name(l.kind), l.gbs);
            o << b;
        }
    }
    o << "\n";
    return o.str();
}

} // namespace vegajet::hw
