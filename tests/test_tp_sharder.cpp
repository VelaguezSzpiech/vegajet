// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/tp_sharder.hpp"
#include "core/gpu_select.hpp"
#include "core/hw_profile.hpp"
#include "tests/test_common.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>

using namespace vegajet;
using namespace vegajet::tp;
namespace {
template<class F> void fails(F f) {
    bool threw = false;
    try { f(); } catch (const std::exception&) { threw = true; }
    VJ_CHECK(threw);
}
std::vector<DeviceProfile> devices(size_t n) {
    std::vector<DeviceProfile> d(n);
    for (size_t r = 0; r < n; ++r) {
        d[r].unique_id = "test-" + std::to_string(r);
        d[r].bdf = gpusel::bdf(0, static_cast<uint32_t>((r + 1) << 8));
        d[r].bw_hbm_gbs = r % 3 == 0 ? 840 : (r % 3 == 1 ? 972 : 700);
        d[r].mclk = r % 3 == 0 ? 900 : (r % 3 == 1 ? 1000 : 800);
        d[r].n_cu = r % 2 ? 64 : 60;
        d[r].pcie_link_gbs = r % 3 == 0 ? 16 : (r % 3 == 1 ? 8 : 4);
        d[r].numa_node = static_cast<int>(r % 2);
        d[r].usable_vram_bytes = (r % 2 ? 32ULL : 16ULL) << 30;
    }
    return d;
}
void matrix() {
    for (size_t n : {1, 2, 3, 4, 6, 8, 12, 16, 20, 23}) {
        auto d = devices(n);
        for (uint32_t tile : {16, 32}) for (uint32_t kv : {1, 4, 32}) for (bool repl : {false, true}) {
            ModelSpec m;
            m.replicated_bytes = 1ULL << 30;
            m.rows = {{"dense", RowKind::Dense, 1024, 1ULL << 20},
                      {"moe", RowKind::MoeIntermediate, 512, 2ULL << 20},
                      {"vocab", RowKind::Vocabulary, 32003, 4096}};
            m.attention = {{"attn", kv * 4, kv, 4096, 1ULL << 20}};
            ShardOptions o; o.row_tile = tile; o.replicate_kv = repl;
            const auto p = plan_shards(m, d, o), repeat = plan_shards(m, d, o);
            VJ_CHECK(p.ranks.size() == n);
            for (size_t t = 0; t < m.rows.size(); ++t) {
                uint64_t pos = 0;
                double sum_bw = 0;
                for (const auto& dev : d) sum_bw += dev.bw_hbm_gbs;
                for (size_t r = 0; r < n; ++r) {
                    const auto& s = p.ranks[r].rows[t];
                    VJ_CHECK(s.rows.begin == pos);
                    VJ_CHECK(s.rows.end >= pos && s.rows.end <= m.rows[t].rows);
                    VJ_CHECK(s.rows.begin % tile == 0 || s.rows.begin == m.rows[t].rows);
                    VJ_CHECK(s.rows.end % tile == 0 || s.rows.end == m.rows[t].rows);
                    VJ_CHECK(s.padded_rows % tile == 0 && s.padded_rows >= s.rows.size());
                    VJ_CHECK(s.recommend_split_k == (s.rows.size() > 0 && s.rows.size() < 240));
                    const double units = static_cast<double>((m.rows[t].rows + tile - 1) / tile);
                    VJ_CHECK(std::abs(static_cast<double>(s.padded_rows / tile) - units * d[r].bw_hbm_gbs / sum_bw) < 1.000001);
                    VJ_CHECK(s.rows.begin == repeat.ranks[r].rows[t].rows.begin && s.rows.end == repeat.ranks[r].rows[t].rows.end);
                    pos = s.rows.end;
                }
                VJ_CHECK(pos == m.rows[t].rows);
            }
            std::set<uint32_t> all_q;
            std::vector<std::set<size_t>> owners(kv);
            for (size_t r = 0; r < n; ++r) {
                const auto& a = p.ranks[r].attention[0];
                VJ_CHECK(a.q_heads == repeat.ranks[r].attention[0].q_heads);
                VJ_CHECK(a.kv_heads == repeat.ranks[r].attention[0].kv_heads);
                VJ_CHECK(a.q_heads.size() == a.q_to_local_kv.size());
                for (size_t i = 0; i < a.q_heads.size(); ++i) {
                    VJ_CHECK(all_q.insert(a.q_heads[i]).second);
                    VJ_CHECK(a.q_to_local_kv[i] < a.kv_heads.size());
                    VJ_CHECK(a.kv_heads[a.q_to_local_kv[i]] == a.q_heads[i] / 4);
                    owners[a.q_heads[i] / 4].insert(r);
                }
                uint64_t expected = m.replicated_bytes + a.bytes;
                for (const auto& rs : p.ranks[r].rows) expected += rs.bytes;
                VJ_CHECK(p.ranks[r].used_bytes == expected);
                VJ_CHECK(expected <= d[r].usable_vram_bytes);
            }
            VJ_CHECK(all_q.size() == kv * 4);
            if (!repl) for (const auto& s : owners) VJ_CHECK(s.size() == 1);
        }
    }
}
void capacity_and_overrides() {
    auto d = devices(3);
    ModelSpec m;
    m.rows = {{"dense", RowKind::Dense, 1024, 1024}};
    d[0].usable_vram_bytes = 64 * 1024;
    auto p = plan_shards(m, d);
    VJ_CHECK(p.ranks[0].rows[0].rows.size() == 64);
    d[1].usable_vram_bytes = 32 * 1024;
    d[2].usable_vram_bytes = 32 * 1024;
    fails([&] { plan_shards(m, d); });
    d = devices(3);
    auto o = shard_options_from_json(R"({"schema":1,"row_tile":16,"split_k_threshold":400,"tensors":{"dense":{"rows":[320,352,352]}}})");
    p = plan_shards(m, d, o);
    VJ_CHECK(p.ranks[0].rows[0].rows.size() == 320 && p.ranks[1].rows[0].recommend_split_k);
    o = shard_options_from_json(R"({"schema":1,"tensors":{"dense":{"fractions":[0.25,0.5,0.25]}}})");
    p = plan_shards(m, d, o);
    VJ_CHECK(p.ranks[1].rows[0].rows.size() == 512);
    d[1].usable_vram_bytes = 32 * 1024;
    fails([&] { plan_shards(m, d, o); });
    d = devices(3);
    o.tensors["dense"].fractions = {0, 1, 0};
    p = plan_shards(m, d, o);
    VJ_CHECK(p.ranks[0].rows[0].rows.size() == 0 && !p.ranks[0].rows[0].recommend_split_k);
    m.attention = {{"a", 8, 2, 1024, 4096}};
    o = shard_options_from_json(R"({"schema":1,"tensors":{"a":{"rows":[3,3,2]}}})");
    fails([&] { plan_shards(m, d, o); });
    o.replicate_kv = true;
    p = plan_shards(m, d, o);
    VJ_CHECK(p.ranks[0].attention[0].q_heads.size() == 3);
    o.tensors["unknown"].rows = {1, 1, 1};
    fails([&] { plan_shards(m, d, o); });
    for (const auto& text : {R"({"schema":1,"row_tile":17})", R"({"schema":1,"typo":2})",
            R"({"schema":1,"schema":1})", R"({"schema":1,"tensors":{"x":{"rows":[1],"fractions":[1]}}})"})
        fails([&] { shard_options_from_json(text); });
    o = {}; o.tensors["dense"].fractions = {0.2, 0.2, 0.2};
    fails([&] { plan_shards(m, d, o); });
    d[1].unique_id = d[0].unique_id;
    fails([&] { plan_shards(m, d); });
    d = devices(1); d[0].bw_hbm_gbs = std::numeric_limits<double>::infinity();
    fails([&] { plan_shards(m, d); });
    d = devices(1); m = {}; m.replicated_bytes = d[0].usable_vram_bytes + 1;
    fails([&] { plan_shards(m, d); });
    m = {}; m.rows = {{"overflow", RowKind::Dense, 32, UINT64_MAX}};
    fails([&] { plan_shards(m, d); });
    m = {}; m.rows = {{"empty", RowKind::Dense, 0, 8}};
    VJ_CHECK(plan_shards(m, d).ranks[0].used_bytes == 0);
}
void profiles() {
    std::ostringstream s;
    s << R"({"schema":1,"gpus":[)";
    for (size_t r = 0; r < 20; ++r) {
        if (r) s << ',';
        s << "{\"unique_id\":\"test-" << r << R"(","bw_hbm_gbs":900,"usable_vram_bytes":17179869184,"n_cu":60,"pcie_link_gbs":8,"numa_node":-1,"mclk":900})";
    }
    s << "]}";
    auto physical_json = s.str();
    const auto capacity = physical_json.find("\"usable_vram_bytes\":17179869184");
    VJ_CHECK(capacity != std::string::npos);
    physical_json.insert(capacity, "\"vram_bytes\":8589934592,");
    fails([&] { profiles_from_json(physical_json); });
    VJ_CHECK(profiles_from_json(s.str()).size() == 20);
    hw::Profile legacy;
    hw::Gpu g; g.bdf = devices(1)[0].bdf; g.cu = 60; g.vram_bytes = 16ULL << 30;
    g.mem_clock_mhz = 900; g.sclk_ghz = 1.4; g.bw_gbs = 840;
    legacy.gpus.push_back(g);
    auto imported = profiles_from_json(hw::to_json(legacy), 1ULL << 30);
    VJ_CHECK(imported[0].key() == "bdf:" + g.bdf);
    VJ_CHECK(imported[0].usable_vram_bytes == 15ULL << 30);
    auto base = std::filesystem::temp_directory_path() / "vj-tp-profile-test";
    std::filesystem::remove_all(base);
    auto dir = base / "sys/bus/pci/devices" / g.bdf;
    std::filesystem::create_directories(dir);
    VJ_CHECK(device_key(g.bdf, base.string()) == "bdf:" + g.bdf);
    { std::ofstream f(dir / "unique_id"); f << "abcd1234\n"; }
    VJ_CHECK(device_key(g.bdf, base.string()) == "uid:abcd1234");
    { std::ofstream f(dir / "unique_id"); f << "0000000000000000\n"; }
    VJ_CHECK(device_key(g.bdf, base.string()) == "bdf:" + g.bdf);
    std::filesystem::remove_all(base);
    fails([] { profiles_from_json(R"({"schema":1,"gpus":[]})"); });
    fails([] { device_key("03:00.0"); });
}
} // namespace
int main() {
    matrix(); capacity_and_overrides(); profiles();
    std::cout << "tp_sharder: N=1,2,3,4,6,8,12,16,20,23 coverage/alignment/heads/VRAM/determinism/overrides PASS\n";
}
