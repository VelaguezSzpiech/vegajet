// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/hw_profile + core/topology_planner on CPU mocks only (no device node is opened):
//   * schema-1 JSON round trip, alias keys for the legacy roofline reader, version/malformed/invalid rejection;
//   * first-boot calibration through an injectable backend, cache hit, forced recalibration, estimate fallback;
//   * N=1 all-reduce elision, N=2/3/4 bandwidth-proportional TP, N=8 TP=4 x 2 swarm, multi-instance swarm;
//   * asymmetric capacity clamping, hard capacity failures, 16/32 GiB boundaries, P2P matrix selection.
#include "core/hw_profile.hpp"
#include "core/topology_planner.hpp"
#include "tests/test_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace vegajet;

namespace {

constexpr uint64_t kGiB = 1ull << 30;
constexpr double kPiBw = 787.05;   // synthetic per-card streaming ceilings (GB/s)
constexpr double kPiBw2 = 839.18;

hw::Profile make_profile(const std::vector<double>& bw, const std::vector<uint64_t>& vram, uint32_t cu,
                         double direct_gbs = 50.0) {
    VJ_CHECK(bw.size() == vram.size());
    hw::Profile p;
    p.arch = "gfx906";
    p.source = "mock";
    p.measured = false;
    for (size_t i = 0; i < bw.size(); ++i) {
        hw::Gpu g;
        g.index = static_cast<uint32_t>(i);
        g.id = "mock" + std::to_string(i);
        g.cu = cu;
        g.vram_bytes = vram[i];
        g.mem = hw::MemKind::HBM2;
        g.bus_bits = 4096;
        g.mem_clock_mhz = 1000.0;
        g.sclk_ghz = 1.725;
        g.bw_gbs = bw[i];
        g.power_cap_w = 225.0;
        g.eta = 0.55;
        p.gpus.push_back(g);
    }
    p.normalize();
    if (direct_gbs > 0.0) {
        for (size_t i = 0; i < p.gpus.size(); ++i) {
            for (size_t j = 0; j < p.gpus.size(); ++j) {
                if (i == j) continue;
                p.gpus[i].p2p[j] = hw::P2pLink{hw::LinkKind::Xgmi, direct_gbs, 4.0};
            }
        }
    }
    return p;
}

topo::ModelFit make_model(uint64_t resident_gib, uint64_t kv_gib) {
    topo::ModelFit m;
    m.name = "mock-model";
    m.resident_bytes = resident_gib * kGiB;
    m.kv_bytes = kv_gib * kGiB;
    m.layers = 48;
    m.hidden = 5120;
    return m;
}

void set_link(hw::Profile& p, size_t a, size_t b, hw::LinkKind kind, double gbs, double latency_us) {
    const hw::P2pLink l{kind, gbs, latency_us};
    p.gpus[a].p2p[b] = l;
    p.gpus[b].p2p[a] = l;
}

const topo::Check* find_check(const topo::Plan& p, const std::string& name) {
    for (const auto& c : p.checks) {
        if (c.name == name) return &c;
    }
    return nullptr;
}

void write_file(const std::string& path, const std::string& text) {
    FILE* f = std::fopen(path.c_str(), "wb");
    VJ_CHECK(f != nullptr);
    VJ_CHECK(std::fwrite(text.data(), 1, text.size(), f) == text.size());
    std::fclose(f);
}

class MockBackend : public hw::MeasurementBackend {
public:
    std::vector<double> clocks = {1000.0, 1140.0};
    std::vector<double> streams = {kPiBw, kPiBw2};
    int topology_calls = 0;
    int stream_calls = 0;
    int link_calls = 0;
    bool fail_topology = false;
    bool fail_stream = false;

    bool topology(std::vector<hw::Gpu>& out, std::string*) override {
        ++topology_calls;
        if (fail_topology) return false;
        out.clear();
        for (size_t i = 0; i < streams.size(); ++i) {
            hw::Gpu g;
            g.index = static_cast<uint32_t>(i);
            g.cu = 60;
            g.vram_bytes = 32ull * kGiB;
            g.mem = hw::MemKind::HBM2;
            g.bus_bits = 4096;
            g.mem_clock_mhz = clocks[i];
            g.sclk_ghz = 1.725;
            g.power_cap_w = 225.0;
            g.eta = 0.55;
            out.push_back(g);
        }
        return true;
    }

    bool measure_stream(const hw::Gpu& gpu, double& gbs, std::string*) override {
        ++stream_calls;
        if (fail_stream) {
            gbs = 0.0;
            return false;
        }
        gbs = streams[gpu.index];
        return true;
    }

    bool links(std::vector<hw::Gpu>& gpus, std::string*) override {
        ++link_calls;
        for (auto& g : gpus) g.p2p.assign(gpus.size(), hw::P2pLink{});
        for (size_t i = 0; i < gpus.size(); ++i) {
            for (size_t j = 0; j < gpus.size(); ++j) {
                if (i == j) continue;
                gpus[i].p2p[j] = hw::P2pLink{hw::LinkKind::Xgmi, 50.0, 4.0};
            }
        }
        return true;
    }
};

void test_hw_profile_json() {
    std::string why;
    const hw::Profile p = make_profile({kPiBw, kPiBw2}, {32 * kGiB, 32 * kGiB}, 60);
    VJ_CHECK(p.valid(&why));
    const std::string js = hw::to_json(p);
    // Canonical keys and the aliases the line-oriented roofline reader expects.
    VJ_CHECK(js.find("\"schema\": 1") != std::string::npos);
    VJ_CHECK(js.find("\"n_cu\": 60") != std::string::npos);
    VJ_CHECK(js.find("\"bw_hbm_gbs\"") != std::string::npos);
    VJ_CHECK(js.find("\"mclk_mhz\"") != std::string::npos);
    VJ_CHECK(js.find("\"p2p\"") != std::string::npos);

    hw::Profile q;
    VJ_CHECK(hw::from_json(js, q, &why));
    VJ_CHECK(q.gpus.size() == 2 && q.schema == 1 && q.arch == "gfx906");
    VJ_CHECK(std::fabs(q.gpus[0].bw_gbs - kPiBw) < 1e-12);
    VJ_CHECK(std::fabs(q.gpus[1].bw_gbs - kPiBw2) < 1e-12);
    VJ_CHECK(q.gpus[1].cu == 60 && q.gpus[1].vram_bytes == 32 * kGiB);
    VJ_CHECK(std::fabs(q.gpus[0].sclk_ghz - 1.725) < 1e-12);
    VJ_CHECK(q.link(0, 1).kind == hw::LinkKind::Xgmi && std::fabs(q.link(0, 1).gbs - 50.0) < 1e-12);
    VJ_CHECK(q.link(0, 0).kind == hw::LinkKind::Unknown);

    // Version gate.
    std::string v2 = js;
    const std::string schema1 = "\"schema\": 1";
    const size_t pos = v2.find(schema1);
    VJ_CHECK(pos != std::string::npos);
    v2.replace(pos, schema1.size(), "\"schema\": 2");
    hw::Profile bad;
    VJ_CHECK(!hw::from_json(v2, bad, &why));
    VJ_CHECK(why.find("schema") != std::string::npos);

    // Malformed / missing sections.
    VJ_CHECK(!hw::from_json("{", bad, &why) && !why.empty());
    VJ_CHECK(!hw::from_json("{\"schema\": 1}", bad, &why));
    VJ_CHECK(!hw::from_json("{\"schema\": 1, \"gpus\": []}", bad, &why));
    VJ_CHECK(!hw::from_json("{\"schema\": 1, \"gpus\": [{\"cu\": 60}]}", bad, &why));

    // Zero CU / zero VRAM / zero bandwidth are rejected after parsing.
    for (int kind = 0; kind < 3; ++kind) {
        hw::Profile broken = p;
        if (kind == 0) broken.gpus[0].cu = 0;
        if (kind == 1) broken.gpus[0].vram_bytes = 0;
        if (kind == 2) broken.gpus[0].bw_gbs = 0.0;
        hw::Profile parsed;
        VJ_CHECK(!hw::from_json(hw::to_json(broken), parsed, &why));
    }

    // Duplicate indices are invalid.
    {
        hw::Profile dup = p;
        dup.gpus[1].index = 0;
        VJ_CHECK(!dup.valid(&why));
    }

    // Nine GPUs exceed the supported topology range.
    {
        hw::Profile nine = make_profile(std::vector<double>(9, 800.0), std::vector<uint64_t>(9, 32 * kGiB), 64);
        VJ_CHECK(!nine.valid(&why));
    }

    // The estimate follows mclk * 2 (DDR) * bus/8 * sustained fraction.
    VJ_CHECK(std::fabs(hw::estimate_stream_gbs(1000.0, 4096) - 819.2) < 1e-9);
    VJ_CHECK(hw::estimate_stream_gbs(0.0, 4096) == 0.0);

    // P2P normalization: mirror a one-sided link, and let an explicit None win over a conflicting direct link.
    {
        hw::Profile m = make_profile({800.0, 800.0}, {32 * kGiB, 32 * kGiB}, 60, 0.0);
        m.gpus[0].p2p[1] = hw::P2pLink{hw::LinkKind::Xgmi, 50.0, 4.0};
        m.normalize();
        VJ_CHECK(m.link(0, 1).kind == hw::LinkKind::Xgmi && m.link(1, 0).kind == hw::LinkKind::Xgmi);
        m.gpus[0].p2p[1] = hw::P2pLink{hw::LinkKind::None, 0.0, 0.0};
        m.gpus[1].p2p[0] = hw::P2pLink{hw::LinkKind::Pcie, 20.0, 8.0};
        m.normalize();
        VJ_CHECK(m.link(0, 1).kind == hw::LinkKind::None && m.link(1, 0).kind == hw::LinkKind::None);
    }

    VJ_CHECK(!hw::default_path().empty() && hw::default_path().find("hw_profile.json") != std::string::npos);
    const std::string rep = hw::format_profile(p);
    VJ_CHECK(rep.find("gpus=2") != std::string::npos && rep.find("bw=") != std::string::npos);
    VJ_CHECK(rep.find("0->1=xgmi/50.0 GB/s") != std::string::npos);
}

void test_calibration() {
    const std::string path = "/tmp/vegajet_test_hwprof_" + std::to_string(getpid()) + ".json";
    const std::string path_est = path + ".est";
    const std::string path_v2 = path + ".v2";
    std::remove(path.c_str());
    std::remove(path_est.c_str());
    std::remove(path_v2.c_str());

    hw::CalibrationConfig cfg;
    cfg.cache_path = path;
    MockBackend backend;
    hw::CalibrationResult r = hw::load_or_calibrate(backend, cfg);
    VJ_CHECK(r.status == hw::CalibrationStatus::Measured);
    VJ_CHECK(r.wrote_cache && r.profile.gpus.size() == 2 && r.profile.measured);
    VJ_CHECK(backend.topology_calls == 1 && backend.stream_calls == 2 && backend.link_calls == 1);
    VJ_CHECK(std::fabs(r.profile.gpus[1].bw_gbs - kPiBw2) < 1e-9);
    VJ_CHECK(r.profile.link(0, 1).kind == hw::LinkKind::Xgmi);

    // Second boot: cache hit, no probe.
    hw::Profile cached;
    std::string why;
    VJ_CHECK(hw::load(path, cached, &why));
    VJ_CHECK(cached.measured && cached.gpus.size() == 2 && cached.link(0, 1).direct());
    r = hw::load_or_calibrate(backend, cfg);
    VJ_CHECK(r.status == hw::CalibrationStatus::Cached && backend.topology_calls == 1);

    // Forced recalibration.
    cfg.force = true;
    r = hw::load_or_calibrate(backend, cfg);
    VJ_CHECK(r.status == hw::CalibrationStatus::Measured && backend.topology_calls == 2);

    // Failed measurement falls back to a datasheet estimate and marks the profile unmeasured.
    MockBackend failing;
    failing.fail_stream = true;
    hw::CalibrationConfig ce;
    ce.cache_path = path_est;
    ce.force = true;
    hw::CalibrationResult re = hw::load_or_calibrate(failing, ce);
    VJ_CHECK(re.status == hw::CalibrationStatus::Estimated && !re.profile.measured && re.wrote_cache);
    VJ_CHECK(std::fabs(re.profile.gpus[0].bw_gbs - hw::estimate_stream_gbs(1000.0, 4096)) < 1e-9);
    VJ_CHECK(std::fabs(re.profile.gpus[1].bw_gbs - hw::estimate_stream_gbs(1140.0, 4096)) < 1e-9);

    // Estimation refused: the calibration fails cleanly.
    ce.allow_estimate = false;
    hw::CalibrationResult rf = hw::load_or_calibrate(failing, ce);
    VJ_CHECK(rf.status == hw::CalibrationStatus::Failed && !rf.detail.empty());

    // Topology failure is fatal.
    MockBackend no_dev;
    no_dev.fail_topology = true;
    ce.allow_estimate = true;
    hw::CalibrationResult rt = hw::load_or_calibrate(no_dev, ce);
    VJ_CHECK(rt.status == hw::CalibrationStatus::Failed && rt.detail.find("topology") != std::string::npos);

    // A future schema in the cache is rejected and replaced by a fresh calibration.
    const std::string good = hw::to_json(cached);
    std::string v2 = good;
    const std::string schema1 = "\"schema\": 1";
    const size_t pos = v2.find(schema1);
    VJ_CHECK(pos != std::string::npos);
    v2.replace(pos, schema1.size(), "\"schema\": 2");
    write_file(path_v2, v2);
    hw::Profile stale;
    VJ_CHECK(!hw::load(path_v2, stale, &why) && why.find("schema") != std::string::npos);
    hw::CalibrationConfig cv;
    cv.cache_path = path_v2;
    MockBackend fresh;
    hw::CalibrationResult rv = hw::load_or_calibrate(fresh, cv);
    VJ_CHECK(rv.status == hw::CalibrationStatus::Measured && fresh.topology_calls == 1);
    VJ_CHECK(hw::load(path_v2, stale, &why) && stale.schema == 1);

    std::remove(path.c_str());
    std::remove(path_est.c_str());
    std::remove(path_v2.c_str());
}

void test_single_gpu_and_swarm() {
    // N=1: all-reduce is elided exactly, no phantom bytes or time.
    const hw::Profile p1 = make_profile({kPiBw}, {32 * kGiB}, 60);
    const topo::ModelFit fits = make_model(12, 1);
    topo::Plan r = topo::plan(p1, fits);
    VJ_CHECK(r.ok && r.mode == topo::Mode::SingleGpu && r.tp == 1 && r.instances == 1);
    VJ_CHECK(!r.allreduce && r.allreduce_elided() && r.ar_bytes_per_token == 0 && r.ar_ms == 0.0);
    VJ_CHECK(r.groups.size() == 1 && r.groups[0].share.size() == 1);
    VJ_CHECK(std::fabs(r.groups[0].share[0] - 1.0) < 1e-12);
    VJ_CHECK(std::fabs(r.per_token_ms - static_cast<double>(12 * kGiB) / (kPiBw * 1e9) * 1e3) < 1e-9);
    VJ_CHECK(std::fabs(r.speedup - 1.0) < 1e-12);
    VJ_CHECK(r.summary().find("allreduce=elided") != std::string::npos);
    VJ_CHECK(r.summary().find("mode=single_gpu") != std::string::npos);

    // N=2 with a model that fits one card: two replicas, still no all-reduce.
    const hw::Profile p2 = make_profile({kPiBw, kPiBw2}, {32 * kGiB, 32 * kGiB}, 60);
    r = topo::plan(p2, fits);
    VJ_CHECK(r.ok && r.mode == topo::Mode::Swarm && r.tp == 1 && r.instances == 2);
    VJ_CHECK(!r.allreduce && r.ar_bytes_per_token == 0 && r.ar_ms == 0.0);
    // The latency reference is the fastest single card; the worst replica sets per_token_ms.
    VJ_CHECK(std::fabs(r.speedup - kPiBw / kPiBw2) < 1e-9);
    const double expected = 1000.0 / (static_cast<double>(12 * kGiB) / (kPiBw * 1e9) * 1e3) +
                            1000.0 / (static_cast<double>(12 * kGiB) / (kPiBw2 * 1e9) * 1e3);
    VJ_CHECK(std::fabs(r.aggregate_tps - expected) / expected < 1e-9);

    // N=8 with a model that fits one card: eight replicas (swarm fallback), TP not used.
    const hw::Profile p8 = make_profile(std::vector<double>(8, 800.0), std::vector<uint64_t>(8, 32 * kGiB), 64);
    r = topo::plan(p8, fits);
    VJ_CHECK(r.ok && r.mode == topo::Mode::Swarm && r.tp == 1 && r.instances == 8);
    VJ_CHECK(!r.allreduce && r.groups.size() == 8);
    for (const auto& g : r.groups) {
        VJ_CHECK(g.tp == 1 && g.gpus.size() == 1 && !g.host_bounced);
        VJ_CHECK(std::fabs(g.share[0] - 1.0) < 1e-12);
    }
    VJ_CHECK(std::fabs(r.aggregate_tps - 8.0 * 800.0e9 / static_cast<double>(12 * kGiB)) < 1e-6);

    // Swarm disabled: a single replica is kept, but TP is still allowed (one wide group).
    topo::PlanConfig noswarm;
    noswarm.allow_swarm = false;
    r = topo::plan(p8, fits, noswarm);
    VJ_CHECK(r.ok && r.instances == 1 && r.tp == 4 && r.mode == topo::Mode::TensorParallel);
    // Both multi-instance and TP disabled: exactly one GPU, TP=1.
    topo::PlanConfig one;
    one.allow_swarm = false;
    one.allow_tp = false;
    r = topo::plan(p8, fits, one);
    VJ_CHECK(r.ok && r.instances == 1 && r.tp == 1 && r.mode == topo::Mode::SingleGpu);
    VJ_CHECK(std::fabs(r.aggregate_tps - r.groups[0].tps) < 1e-12);

    // Forcing TP=1 on eight GPUs is the swarm path again.
    r = topo::plan_tp(p8, fits, 1);
    VJ_CHECK(r.ok && r.tp == 1 && r.instances == 8 && r.mode == topo::Mode::Swarm);

    // Deterministic summary.
    const topo::Plan det1 = topo::plan(p8, fits);
    const topo::Plan det2 = topo::plan(p8, fits);
    VJ_CHECK(det1.summary() == det2.summary());
    VJ_CHECK(det1.summary().find("instances=8") != std::string::npos);
}

void test_tensor_parallel() {
    // N=3, 44 GiB resident footprint: only TP fits. Shares follow calibrated bandwidth.
    const hw::Profile p3 = make_profile({700.0, 800.0, 900.0}, std::vector<uint64_t>(3, 32 * kGiB), 60);
    topo::ModelFit big = make_model(40, 4);
    topo::Plan r = topo::plan(p3, big);
    VJ_CHECK(r.ok && r.mode == topo::Mode::TensorParallel && r.tp == 3 && r.instances == 1);
    VJ_CHECK(r.allreduce && !r.allreduce_elided());
    VJ_CHECK(r.ar_bytes_per_token == 48ull * 5120ull * 2ull * 2ull && r.ar_ms > 0.0);
    VJ_CHECK(r.groups.size() == 1 && r.groups[0].gpus.size() == 3);
    const double total_bw = 2400.0;
    double c0 = r.groups[0].compute_ms;
    double cmin = c0, cmax = c0;
    for (size_t i = 0; i < 3; ++i) {
        const double expected_share = r.groups[0].gpus[i] == 0 ? 700.0 / total_bw
                                        : r.groups[0].gpus[i] == 1 ? 800.0 / total_bw
                                                                   : 900.0 / total_bw;
        VJ_CHECK(std::fabs(r.groups[0].share[i] - expected_share) < 1e-6);
        const double bw = r.groups[0].gpus[i] == 0 ? 700.0 : r.groups[0].gpus[i] == 1 ? 800.0 : 900.0;
        const double ti = r.groups[0].share[i] * static_cast<double>(40 * kGiB) / (bw * 1e9) * 1e3;
        cmin = std::min(cmin, ti);
        cmax = std::max(cmax, ti);
    }
    VJ_CHECK(cmax - cmin < 1e-6);  // bandwidth-proportional split equalizes stream time
    VJ_CHECK(r.groups[0].effective_link_gbs == 50.0 && !r.groups[0].host_bounced);

    // Forced TP=3 on a model that also fits one card: the single-GPU reference exists and TP is faster.
    topo::Plan tpr = topo::plan_tp(p3, make_model(20, 2), 3);
    VJ_CHECK(tpr.ok && tpr.tp == 3 && tpr.instances == 1 && tpr.single_gpu_ms > 0.0 && tpr.speedup > 2.4);

    // Explicit all-reduce payload overrides the derived one.
    topo::ModelFit explicit_ar = big;
    explicit_ar.ar_bytes_per_token = 123456;
    VJ_CHECK(topo::plan(p3, explicit_ar).ar_bytes_per_token == 123456);

    // N=2 asymmetric bandwidth.
    const hw::Profile p2 = make_profile({kPiBw, kPiBw2}, {32 * kGiB, 32 * kGiB}, 60);
    r = topo::plan(p2, big);
    VJ_CHECK(r.ok && r.tp == 2 && r.instances == 1 && r.mode == topo::Mode::TensorParallel);
    const double sum2 = kPiBw + kPiBw2;
    VJ_CHECK(std::fabs(r.groups[0].share[0] - kPiBw / sum2) < 1e-6);
    VJ_CHECK(std::fabs(r.groups[0].share[1] - kPiBw2 / sum2) < 1e-6);

    // N=8 on 16 GiB cards: nothing below TP=4 can hold 44 GiB (usable 14 GiB/card), so TP=4 x 2 instances.
    const hw::Profile p816 = make_profile(std::vector<double>(8, 800.0), std::vector<uint64_t>(8, 16 * kGiB), 64);
    r = topo::plan(p816, big);
    VJ_CHECK(r.ok && r.tp == 4 && r.instances == 2 && r.mode == topo::Mode::Swarm);
    VJ_CHECK(r.ar_bytes_per_token == 48ull * 5120ull * 2ull * 2ull);
    VJ_CHECK(r.groups.size() == 2);
    for (const auto& g : r.groups) {
        VJ_CHECK(g.tp == 4 && g.gpus.size() == 4 && g.ar_ms > 0.0);
        for (double s : g.share) VJ_CHECK(std::fabs(s - 0.25) < 1e-6);
        for (size_t i = 0; i < 4; ++i) VJ_CHECK(g.resident_bytes[i] + g.kv_bytes[i] == 11 * kGiB);
    }

    // Forcing a TP degree that cannot hold the footprint fails with a capacity/forced explanation.
    VJ_CHECK(!topo::plan_tp(p816, big, 2).ok);
}

void test_capacity_and_asymmetry() {
    // 16 + 32 GiB: the small card is pinned at its usable capacity, the remainder moves to the large one.
    const hw::Profile p = make_profile({800.0, 800.0}, {16 * kGiB, 32 * kGiB}, 60);
    topo::Plan r = topo::plan(p, make_model(41, 1));  // 42 GiB total, 14 + 30 usable
    VJ_CHECK(r.ok && r.tp == 2 && r.instances == 1 && r.mode == topo::Mode::TensorParallel);
    VJ_CHECK(r.groups.size() == 1);
    VJ_CHECK(std::fabs(r.groups[0].share[0] - 1.0 / 3.0) < 1e-6);
    VJ_CHECK(std::fabs(r.groups[0].share[1] - 2.0 / 3.0) < 1e-6);
    VJ_CHECK(r.groups[0].resident_bytes[0] + r.groups[0].kv_bytes[0] == 14 * kGiB);
    VJ_CHECK(r.groups[0].resident_bytes[1] + r.groups[0].kv_bytes[1] == 28 * kGiB);
    VJ_CHECK(std::fabs(r.groups[0].compute_ms -
                       r.groups[0].share[1] * static_cast<double>(41 * kGiB) / (800.0 * 1e9) * 1e3) < 1e-9);
    for (const auto& g : r.groups) {
        for (size_t i = 0; i < g.gpus.size(); ++i) {
            VJ_CHECK(g.resident_bytes[i] + g.kv_bytes[i] <= p.gpus[g.gpus[i]].vram_bytes - (2 * kGiB));
        }
    }
    VJ_CHECK(find_check(r, "capacity") != nullptr && find_check(r, "capacity")->ok);
    VJ_CHECK(find_check(r, "balance") != nullptr && find_check(r, "balance")->detail.find("deviation") != std::string::npos);

    // Hard capacity failure: 45 GiB does not fit 14 + 30 usable with the reserve.
    r = topo::plan(p, make_model(44, 1));
    VJ_CHECK(!r.ok && r.mode == topo::Mode::Insufficient && r.groups.empty());
    const topo::Check* cap = find_check(r, "capacity");
    VJ_CHECK(cap != nullptr && !cap->ok);
    VJ_CHECK(r.detail.find("no feasible topology") != std::string::npos);

    // 16 GiB exact boundary: 14 GiB fits above a 2 GiB reserve, one more byte does not.
    const hw::Profile p16 = make_profile({800.0}, {16 * kGiB}, 60);
    VJ_CHECK(topo::plan(p16, make_model(14, 0)).ok);
    topo::ModelFit over = make_model(14, 0);
    over.resident_bytes += 1;
    VJ_CHECK(!topo::plan(p16, over).ok);

    // Invalid profiles and empty models are rejected before any planning.
    hw::Profile empty;
    r = topo::plan(empty, make_model(1, 0));
    VJ_CHECK(!r.ok && r.groups.empty() && r.detail.find("invalid hardware profile") != std::string::npos);
    const topo::Check* prof = find_check(r, "profile");
    VJ_CHECK(prof != nullptr && !prof->ok);

    hw::Profile zerocu = make_profile({800.0}, {32 * kGiB}, 0);
    VJ_CHECK(!topo::plan(zerocu, make_model(1, 0)).ok);

    topo::ModelFit zero;
    r = topo::plan(p16, zero);
    VJ_CHECK(!r.ok && r.detail.find("empty") != std::string::npos);

    // Forced TP outside the supported/n visible range.
    r = topo::plan_tp(p16, make_model(1, 0), 5);
    VJ_CHECK(!r.ok && r.detail.find("supported range") != std::string::npos);
    const hw::Profile p2 = make_profile({800.0, 800.0}, {32 * kGiB, 32 * kGiB}, 60);
    r = topo::plan_tp(p2, make_model(44, 0), 3);
    VJ_CHECK(!r.ok && r.detail.find("more GPUs") != std::string::npos);
}

void test_p2p_topology() {
    // No direct peer link: TP still plans, but the all-reduce is host-bounced at the host fallback bandwidth.
    const hw::Profile host_only = make_profile({800.0, 800.0}, {16 * kGiB, 16 * kGiB}, 60, 0.0);
    topo::Plan r = topo::plan_tp(host_only, make_model(20, 0), 2);
    VJ_CHECK(r.ok && r.tp == 2 && r.instances == 1);
    VJ_CHECK(r.groups[0].host_bounced && !r.p2p_complete && !r.groups[0].p2p_complete);
    VJ_CHECK(std::fabs(r.groups[0].effective_link_gbs - 10.0) < 1e-12);
    VJ_CHECK(r.allreduce && r.ar_ms > 0.0);

    // Requiring direct P2P rejects that group with a link-specific reason.
    topo::PlanConfig strict;
    strict.links.require_direct_p2p = true;
    r = topo::plan_tp(host_only, make_model(20, 0), 2, strict);
    VJ_CHECK(!r.ok && r.detail.find("P2P") != std::string::npos);

    // Four cards: the pair without a direct link is avoided; both instances stay on direct XGMI.
    hw::Profile mesh = make_profile({800.0, 800.0, 800.0, 800.0}, std::vector<uint64_t>(4, 16 * kGiB), 60);
    set_link(mesh, 0, 1, hw::LinkKind::None, 0.0, 0.0);
    topo::PlanConfig forced;
    forced.force_tp = 2;
    r = topo::plan(mesh, make_model(20, 0), forced);
    VJ_CHECK(r.ok && r.tp == 2 && r.instances == 2 && r.mode == topo::Mode::Swarm);
    for (const auto& g : r.groups) VJ_CHECK(!g.host_bounced && g.p2p_complete);
    VJ_CHECK((r.groups[0].gpus == std::vector<uint32_t>{0, 2}));
    VJ_CHECK((r.groups[1].gpus == std::vector<uint32_t>{1, 3}));
    VJ_CHECK(r.p2p_complete);
    VJ_CHECK(r.groups[0].effective_link_gbs == 50.0);
}

} // namespace

int main() {
    test_hw_profile_json();
    test_calibration();
    test_single_gpu_and_swarm();
    test_tensor_parallel();
    test_capacity_and_asymmetry();
    test_p2p_topology();
    std::printf("test_topology_planner: profile schema/calibration, N=1 elision, TP=2/3/4, swarm, capacity, P2P OK\n");
    std::printf("PASS\n");
    return 0;
}
