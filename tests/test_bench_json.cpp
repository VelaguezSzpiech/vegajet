// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/bench_json: every engine bench line (printed here with the engine's own printf format strings, copied from
// models/ornith/ornith_gpu.cpp) parsed into the record; the vegajet.bench.v1 JSON (key names, nulls for metrics a
// run did not produce, no bus addresses, file names only); JSON -> record -> JSON idempotence; merging a W4A8 run
// and a --prefill-dot8 run into one scorecard (worst-case telemetry, RAS deltas adding up); write_file merge rules;
// read-only RAS / TMem snapshots of a synthetic sysfs tree as before / after deltas; the stdout tee.
//   test_bench_json [OUT_DIR]   with OUT_DIR: also writes bench.log, base.json, cand_ok.json, cand_bad.json and
//                               w4a8_then_dot8.json for tests/test_bench_json_roundtrip.py.
#include "core/bench_json.hpp"
#include "core/json_lite.hpp"
#include "tests/test_common.hpp"

#include <unistd.h>

#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace vegajet;
using namespace vegajet::benchjson;
namespace fs = std::filesystem;

namespace {

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
    char b[1024];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(b, sizeof(b), f, ap);
    va_end(ap);
    return b;
}

// One bench log as the engine prints it (format strings verbatim from ornith_gpu.cpp).
std::string engine_log(double raw, double pp512, double pp2048, unsigned code_match, double tmem, double avg_w, bool dot8_kld_line,
                       bool hyb_lossless = true) {
    std::string s;
    s += "[ornith::gpu] loading weights ...\n";
    s += fmt("[PARITY code  vs Tier-1 llama.cpp] %zu/%zu exact greedy tokens -> %s\n", size_t(code_match), size_t(32),
             code_match == 32 ? "PASS" : "FAIL");
    s += fmt("[BENCH tg128 raw M=1] %.2f tok/s (%.2f ms/tok) | eff_HBM2=%.1f GB/s (%.1f%% of %.1f GB/s calibrated) | real_read=%.3f GB/tok | disp/tok=%u\n",
             raw, 1000.0 / raw, 612.0, 71.2, 860.0, 18.512, 1203u);
    s += fmt("[BENCH tg128 Suffix ] %.2f tok/s (%.2fx vs raw) | lossless=%u/%u (first_diff=%d) | hits=%llu acc=%llu/%llu | 0 B VRAM\n",
             raw * 1.4, 1.4, 128u, 128u, -1, 30ull, 70ull, 90ull);
    s += fmt("[BENCH tg512 Suffix ] %.2f tok/s | hits=%llu acc=%llu/%llu | 0 B VRAM\n", raw * 1.5, 100ull, 300ull, 400ull);
    s += fmt("[BENCH tg128 MTP K=%u] %.2f tok/s (%.2fx vs raw %.2f) | prose lossless=%u/%zu (first_diff=%d) | acc=%llu/%llu (%.1f%%)\n",
             4u, raw * 1.9, 1.9, raw, 129u, size_t(129), -1, 300ull, 420ull, 71.4);
    s += fmt("[BENCH tg128 Hybrid ] %.2f tok/s (%.2fx vs raw) | code lossless=%u/%zu (first_diff=%d) | sfx_hits=%llu acc=%llu/%llu (%.1f%%)\n",
             raw * 2.0, 2.0, 129u, size_t(129), -1, 12ull, 340ull, 450ull, 75.6);
    s += fmt("[BENCH tg128 DFlash2 K=%u prose] %.2f tok/s (%.2fx vs raw %.2f) | prose lossless=%u/%zu (first_diff=%d) | acc=%llu/%llu (%.1f%%)\n",
             7u, raw * 2.1, 2.1, raw, 129u, size_t(129), -1, 500ull, 700ull, 71.4);
    s += fmt("[BENCH tg128 DFlash2 K=%u code ] %.2f tok/s (%.2fx vs raw %.2f) | code lossless=%u/%zu (first_diff=%d) | acc=%llu/%llu (%.1f%%)\n",
             7u, raw * 2.4, 2.4, raw, 129u, size_t(129), -1, 560ull, 700ull, 80.0);
    s += fmt("[BENCH tg128 DFlash2+Hyb code ] %.2f tok/s (%.2fx vs raw %.2f) | code lossless=%u/%zu (first_diff=%d) | acc=%llu/%llu (%.1f%%)\n",
             raw * 2.3, 2.3, raw, hyb_lossless ? 129u : 128u, size_t(129), hyb_lossless ? -1 : 77, 550ull, 700ull, 78.6);
    s += fmt("[BENCH pp512  M=%-2u  ] %.2f tok/s (WY_GDN=%s, 0 SOLVE_TRI, first_tok=%d)\n", 512u, pp512, "ON", 1234);
    s += fmt("[BENCH pp2048 M=%-2u  ] %.2f tok/s (WY_GDN=%s, 0 SOLVE_TRI, first_tok=%d)\n", 512u, pp2048, "ON", 1234);
    if (dot8_kld_line) s += "[KLD dot8 prefill] 0.00089 (held-out multi-domain prompts)\n";
    s += fmt("[TELEM] power1_input: worst 10 s rolling avg=%.1f W (<=%.0f W) | worst 1 s avg=%.1f W (info) | "
             "instantaneous peak=%.1f W (<270 W) | max_TMem=%.1f C (<65 C) | %llu samples -> %s\n",
             avg_w, 205.0, avg_w + 9, 241.0, tmem, 400ull, (avg_w <= 205 && tmem < 65) ? "PASS" : "FAIL");
    s += "unrelated line 99 tok/s\n";
    return s;
}

void put(const fs::path& p, const std::string& text) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << text;
}

void fake_gpu(const fs::path& root, const std::string& dev, uint64_t umc_ce, uint64_t umc_ue, double tmem) {
    const fs::path d = root / "sys/bus/pci/devices" / dev;
    put(d / "vendor", "0x1002\n");
    put(d / "device", "0x66a1\n");
    put(d / "hwmon/hwmon0/power1_cap", "200000000\n");
    put(d / "hwmon/hwmon0/power1_input", "150000000\n");
    put(d / "hwmon/hwmon0/temp3_input", std::to_string(static_cast<long>(tmem * 1000)) + "\n");
    put(d / "ras/umc_err_count", "ue: " + std::to_string(umc_ue) + "\nce: " + std::to_string(umc_ce) + "\n");
    for (const char* b : {"gfx", "hdp", "mmhub", "pcie_bif", "sdma"}) put(d / "ras" / (std::string(b) + "_err_count"), "ue: 0\nce: 0\n");
}

json::Value J(const std::string& s) {
    json::Value v;
    VJ_CHECK(json::parse(s, v));
    return v;
}
const json::Value& G(const json::Value& v, std::initializer_list<const char*> path) {
    const json::Value* c = &v;
    for (const char* k : path) {
        c = c->get(k);
        if (!c) { std::fprintf(stderr, "missing key %s\n", k); VJ_CHECK(false); }
    }
    return *c;
}

Record base_record(const std::string& path) {
    Record r;
    r.commit = "abc1234";
    r.model = "Ornith-Q2-model";
    r.architecture = "qwen35moe";
    r.tp = 2;
    r.prefill_path = path;
    r.flags = sanitize_flags({"Ornith-Q2-model", "--tp", "2", "--hsaco-dir", "/some/where/hsaco"});
    r.timestamp_utc = "2026-11-24T00:00:00Z";
    r.executed = true;
    r.plan_status = "fits";
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    // ---- parsing
    Record r = base_record("w4a8");
    const std::string log = engine_log(33.10, 1400.5, 1433.6, 32, 54.5, 180.0, false, false);
    parse_log(log, r);
    VJ_CHECK(r.raw_tok_s && *r.raw_tok_s == 33.10 && r.raw_ms_per_tok && std::abs(*r.raw_ms_per_tok - 30.21) < 1e-9);
    VJ_CHECK(*r.eff_hbm2_gbs == 612.0 && *r.hbm2_pct == 71.2 && *r.hbm2_ceiling_gbs == 860.0 && *r.real_read_gb_per_tok == 18.512 &&
             *r.dispatches_per_tok == 1203u);
    VJ_CHECK(r.prefill.at("pp512_w4a8") == 1400.5 && r.prefill.at("pp2048_w4a8") == 1433.6 && r.prefill_m.at("pp2048") == 512u);
    VJ_CHECK(!r.prefill.count("pp512_dot8") && !r.dot8_kld);
    VJ_CHECK(r.spec.size() == 7);
    std::map<std::string, Spec> by;
    for (const auto& s : r.spec) by[s.key] = s;
    VJ_CHECK(by.count("suffix") && by.count("tg512_suffix") && by.count("mtp_k4") && by.count("hybrid") &&
             by.count("dflash2_k7_prose") && by.count("dflash2_k7_code") && by.count("dflash2_hyb_code"));
    VJ_CHECK(by["mtp_k4"].label == "MTP K=4" && *by["mtp_k4"].speedup == 1.9 && *by["mtp_k4"].match == 129u &&
             *by["mtp_k4"].first_diff == -1 && *by["mtp_k4"].acceptance_pct == 71.4);
    VJ_CHECK(by["tg512_suffix"].tokens == 512 && !by["tg512_suffix"].match && !by["tg512_suffix"].speedup);
    VJ_CHECK(*by["dflash2_hyb_code"].match == 128u && *by["dflash2_hyb_code"].first_diff == 77);
    VJ_CHECK(*r.parity_match == 32u && *r.parity_total == 32u && r.parity_verdict == "PASS");
    VJ_CHECK(*r.power_avg_w == 180.0 && *r.power_avg_limit_w == 205.0 && *r.power_peak_w == 241.0 && *r.tmem_max_c == 54.5 &&
             r.telem_verdict == "PASS");
    VJ_CHECK(r.flags == "Ornith-Q2-model --tp 2 --hsaco-dir hsaco");  // directories never reach the scorecard
    // ---- JSON
    const std::string js = to_json(r);
    const json::Value v = J(js);
    VJ_CHECK(G(v, {"schema"}).str == "vegajet.bench.v1" && G(v, {"meta", "model"}).str == "Ornith-Q2-model");
    VJ_CHECK(G(v, {"throughput", "raw_m1_tok_s"}).num == 33.10 && G(v, {"throughput", "pp512_dot8_tok_s"}).is_null());
    VJ_CHECK(G(v, {"throughput", "spec", "dflash2_k7_code_tok_s"}).num > 79 && G(v, {"kld", "dot8_prefill_kld"}).is_null());
    VJ_CHECK(G(v, {"parity", "code_vs_tier1", "match"}).num == 32 && G(v, {"parity", "mtp_k4_lossless", "total"}).num == 129);
    VJ_CHECK(G(v, {"first_diff", "dflash2_hyb_code"}).num == 77 && G(v, {"telemetry", "tmem_max_c"}).num == 54.5);
    VJ_CHECK(G(v, {"spill", "private_segment_bytes_max"}).is_null() && G(v, {"prefill_batch_m", "pp512"}).num == 512);
    VJ_CHECK(js.find("/some/where") == std::string::npos);
    // JSON -> record -> JSON is the identity
    Record back;
    std::string err;
    VJ_CHECK(from_json(js, back, &err) && to_json(back) == js);
    VJ_CHECK(!from_json("{\"schema\":\"other\"}", back, &err) && !from_json("nope", back, &err));
    // ---- sysfs snapshots: RAS deltas and TMem
    const char* tmp = std::getenv("TMPDIR");
    const fs::path root = fs::path(tmp && *tmp ? tmp : "/tmp") / ("vj-benchjson-" + std::to_string(::getpid()));
    fake_gpu(root, "fake-gpu-a", 3, 0, 51.0);
    fake_gpu(root, "fake-gpu-b", 0, 0, 49.0);
    r.gpus_before = snapshot(root.string());
    VJ_CHECK(r.gpus_before.size() == 2 && r.gpus_before[0].id == "gpu0" && r.gpus_before[0].ras.size() == 6 &&
             r.gpus_before[0].ras[0].block == "umc" && r.gpus_before[0].ras[0].ce == 3 && r.gpus_before[0].tmem_c == 51.0);
    fake_gpu(root, "fake-gpu-a", 5, 0, 54.0);  // +2 CE during the run
    r.gpus_after = snapshot(root.string());
    const json::Value v2 = J(to_json(r));
    VJ_CHECK(G(v2, {"ras_window"}).str == "delta" && G(v2, {"ras", "gpu0", "umc", "ce"}).num == 2 && G(v2, {"ras", "gpu0", "umc", "ue"}).num == 0);
    VJ_CHECK(G(v2, {"ras", "gpu1", "sdma", "de"}).num == 0 && G(v2, {"gpus"}).arr.size() == 2 && G(v2, {"gpus"}).arr[0].get("tmem_c")->num == 54.0);
    VJ_CHECK(to_json(r).find("fake-gpu") == std::string::npos);  // ids, never device directory names
    VJ_CHECK(from_json(to_json(r), back, &err) && to_json(back) == to_json(r));
    // ---- merge: W4A8 run + dot8 run -> one scorecard
    Record dot8 = base_record("dot8");
    parse_log(engine_log(33.0, 1650.0, 1701.0, 32, 57.0, 170.0, true), dot8);
    dot8.gpus_before = r.gpus_after;
    fake_gpu(root, "fake-gpu-a", 6, 0, 57.0);  // +1 CE in the second run
    dot8.gpus_after = snapshot(root.string());
    Record merged = back;
    merge(merged, dot8);
    VJ_CHECK(merged.prefill.at("pp512_w4a8") == 1400.5 && merged.prefill.at("pp2048_dot8") == 1701.0 && *merged.dot8_kld == 0.00089);
    VJ_CHECK(merged.prefill_path == "w4a8+dot8" && *merged.tmem_max_c == 57.0 && *merged.power_avg_w == 180.0);  // worst case kept
    VJ_CHECK(*merged.raw_tok_s == 33.0);  // the newer run's measurement
    const json::Value v3 = J(to_json(merged));
    VJ_CHECK(G(v3, {"ras", "gpu0", "umc", "ce"}).num == 3);  // 2 + 1
    // ---- write_file: merge only for the same model + commit + architecture
    const std::string out = (root / "score.json").string();
    bool was_merged = true;
    VJ_CHECK(write_file(out, r, &err, &was_merged) && !was_merged);
    VJ_CHECK(write_file(out, dot8, &err, &was_merged) && was_merged);
    {
        std::ifstream f(out);
        std::stringstream ss;
        ss << f.rdbuf();
        Record got;
        VJ_CHECK(from_json(ss.str(), got, &err) && got.prefill.size() == 4 && got.dot8_kld);
    }
    Record other = dot8;
    other.commit = "fffffff";
    VJ_CHECK(write_file(out, other, &err, &was_merged) && !was_merged);
    // ---- stdout tee
    {
        StdoutTee tee;
        VJ_CHECK(tee.begin());
        std::printf("[BENCH tg128 raw M=1] 40.00 tok/s (25.00 ms/tok)\n");
        std::printf("tee check line\n");
        const std::string cap = tee.end();
        VJ_CHECK(cap.find("tee check line") != std::string::npos);
        Record t;
        parse_log(cap, t);
        VJ_CHECK(t.raw_tok_s && *t.raw_tok_s == 40.0 && !t.eff_hbm2_gbs);
    }
    // ---- files for the Python round trip (regression_gate.py + scorecard.py)
    if (argc > 1) {
        const fs::path d = argv[1];
        fs::create_directories(d);
        const std::string good_log = engine_log(33.10, 1400.5, 1433.6, 32, 54.5, 180.0, false);
        std::ofstream(d / "bench.log") << good_log;
        Record base = base_record("w4a8");
        parse_log(good_log, base);
        base.gpus_before = base.gpus_after = snapshot(root.string());
        std::ofstream(d / "base.json") << to_json(base);
        Record ok = base_record("w4a8");
        parse_log(engine_log(33.0, 1395.0, 1430.0, 32, 55.0, 182.0, false), ok);  // -0.3 % raw: within the budget
        ok.gpus_before = ok.gpus_after = base.gpus_after;
        std::ofstream(d / "cand_ok.json") << to_json(ok);
        Record bad = base_record("dot8");
        bad.prefill_path = "w4a8";
        parse_log(engine_log(31.0, 1400.0, 1433.0, 31, 66.0, 180.0, true, false), bad);  // -6.3 % raw, parity 31/32, TMem 66
        bad.dot8_kld = 0.051;
        bad.spill_bytes_max = 16;
        bad.gpus_before = base.gpus_after;
        fake_gpu(root, "fake-gpu-b", 0, 1, 49.0);  // one uncorrectable error during the run
        bad.gpus_after = snapshot(root.string());
        std::ofstream(d / "cand_bad.json") << to_json(bad);
        std::ofstream(d / "w4a8_then_dot8.json") << to_json(merged);
    }
    fs::remove_all(root);
    std::printf("test_bench_json: engine lines (engine format strings), vegajet.bench.v1 JSON, idempotent round trip, "
                "W4A8+dot8 merge, write_file rules, sysfs RAS/TMem deltas, stdout tee OK\nPASS\n");
    return 0;
}
