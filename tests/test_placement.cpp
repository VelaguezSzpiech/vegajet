// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/placement.hpp"
#include "core/gpu_select.hpp"
#include "tests/test_common.hpp"
#include <iostream>
using namespace vegajet;
namespace {
template<class F> void fails(F f, const std::string& needle) {
    bool hit = false;
    try { f(); } catch (const std::exception& e) { hit = std::string(e.what()).find(needle) != std::string::npos; if (!hit) std::cerr << e.what() << '\n'; }
    VJ_CHECK(hit);
}
std::vector<tp::DeviceProfile> inventory(uint32_t n) {
    std::vector<tp::DeviceProfile> out(n);
    for (uint32_t i = 0; i < n; ++i) {
        out[i].unique_id = "card-" + std::to_string(i);
        out[i].bdf = gpusel::bdf(0, (i + 1) << 8);
        out[i].usable_vram_bytes = 1000000; out[i].bw_hbm_gbs = 1; out[i].mclk = 1; out[i].n_cu = 1;
    }
    return out;
}
placement::Config single(uint32_t n) {
    placement::Config c; placement::TpGroup g; g.id = "target";
    for (uint32_t i = 0; i < n; ++i) g.devices.push_back({"card-" + std::to_string(n-i-1), ""});
    c.tp_groups.push_back(g); return c;
}
}
int main() {
    const auto capacity = [](const std::vector<tp::DeviceProfile>& d, uint32_t b, uint32_t e) {
        tp::ModelSpec m; m.replicated_bytes = 100;
        m.rows = {{"weights", tp::RowKind::Dense, 256, uint64_t(e-b)*16}};
        (void)tp::plan_shards(m, d);
    };
    for (uint32_t n : {1u, 2u, 3u, 8u}) {
        auto c = single(n); const auto d = inventory(n);
        placement::validate(c, d, 8, capacity);
        const auto text = placement::to_json(c);
        VJ_CHECK(placement::to_json(placement::from_json(text)) == text);
        const auto ordered = placement::executor_devices(c, d);
        VJ_CHECK(ordered.front().unique_id == d.back().unique_id);
        VJ_CHECK(ordered.back().unique_id == d.front().unique_id);
        tp::PeerMatrix peers(n, std::vector<tp::PeerLink>(n, {true, 1}));
        auto proposal = placement::propose(d, peers, 8, capacity);
        VJ_CHECK(proposal.tp_groups[0].devices.size() == n);
        placement::validate(proposal, d, 8, capacity);
        auto small = d; for (auto& p : small) p.usable_vram_bytes = 10;
        fails([&]{ placement::validate(c, small, 8, capacity); }, "VRAM");
        fails([&]{ (void)placement::propose(small, peers, 8, capacity); }, "no P2P island fits");
    }
    auto c = single(2); auto d = inventory(3);
    c.tp_groups[0].devices[1] = {"", d[1].bdf}; // same physical card as rank 0
    fails([&]{ placement::validate(c, d, 8, capacity); }, "overlap");
    c = single(2); c.tp_groups[0].devices[0].unique_id = "missing";
    fails([&]{ placement::resolve(c, d); }, "missing");
    c = single(2); auto duplicate = d; duplicate[2] = duplicate[0];
    fails([&]{ placement::resolve(c, duplicate); }, "duplicate BDF");
    c = single(1); c.tp_groups.push_back({"next", {{"card-1", ""}}});
    c.pipeline_stages = {{"target", 0, 4}, {"next", 4, 8}};
    placement::validate(c, d, 8, capacity);
    VJ_CHECK(placement::executor_error(c).find("pipeline stages") != std::string::npos);
    c.pipeline_stages[1].layer_begin = 5;
    fails([&]{ placement::validate(c, d, 8, capacity); }, "gap");
    c.pipeline_stages[1].layer_begin = 3;
    fails([&]{ placement::validate(c, d, 8, capacity); }, "overlap");
    c.pipeline_stages[1] = {"next", 4, 7};
    fails([&]{ placement::validate(c, d, 8, capacity); }, "gap at end");
    c.pipeline_stages = {{"missing", 0, 8}};
    fails([&]{ placement::validate(c, d, 8, capacity); }, "missing TP group");
    c.pipeline_stages.clear(); c.replicas = {{"a", "target"}, {"b", "next"}};
    placement::validate(c, d, 8, capacity);
    VJ_CHECK(placement::executor_error(c).find("replicas") != std::string::npos);
    c.tp_groups[1].devices.push_back({"card-2", ""});
    fails([&]{ placement::validate(c, d, 8, capacity); }, "N=1");
    c = single(1); c.drafter = {"DFlash2", {"card-1", ""}};
    placement::validate(c, d, 8, capacity);
    VJ_CHECK(placement::executor_error(c).find("dedicated drafter") != std::string::npos);
    c.drafter.device.unique_id = "card-0";
    fails([&]{ placement::validate(c, d, 8, capacity); }, "drafter device overlaps");
    for (const std::string bad : {
            R"({"schema":1,"schema":1,"tp_groups":[]})",
            R"({"schema":2,"tp_groups":[]})",
            R"({"schema":1,"tp_groups":[],"unknown":0})",
            R"({"schema":1,"tp_groups":[{"id":"x","devices":[{"unique_id":"a","bdf":"bad"}]}]})",
            R"({"schema":1,"tp_groups":[{"id":"x","devices":[{"bdf":"bad"}]}]})"})
        fails([&]{ (void)placement::from_json(bad); }, "");
    tp::PeerMatrix peers(3, std::vector<tp::PeerLink>(3));
    peers[0][1] = {true, 1}; // asymmetric link is not an island
    auto proposal = placement::propose(d, peers, 8, capacity);
    VJ_CHECK(proposal.tp_groups[0].devices.size() == 1);
    // First island does not fit, next does. There is no speed/bandwidth ordering.
    d[0].usable_vram_bytes = 10;
    proposal = placement::propose(d, peers, 8, capacity);
    VJ_CHECK(proposal.tp_groups[0].devices[0].unique_id == "card-1");
    std::cout << "placement: N=1,2,3,8 ordering, topology, capacity and fail-closed checks PASS\n";
}
