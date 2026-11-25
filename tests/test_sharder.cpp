// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// b=32 largest-remainder asymmetric TP sharding (N = 1..4), GQA boundary KV replication, MLA fallback,
// GDN group split, plus the KV-tier and ReplaySSM planners that consume the shard geometry.
#include "core/kv_cache.hpp"
#include "core/sharder.hpp"
#include "core/ssm_state.hpp"
#include "tests/test_common.hpp"

#include <cmath>
#include <numeric>

using namespace vegajet;
using namespace vegajet::shard;

int main() {
    // Largest remainder: exact totals, ties to lower ranks, proportional to weights.
    VJ_CHECK((apportion(10, {1, 1, 1}) == std::vector<uint64_t>{4, 3, 3}));
    VJ_CHECK((apportion(7, {2, 1}) == std::vector<uint64_t>{5, 2}));
    VJ_CHECK((apportion(100, {1000, 800, 800, 800}) == std::vector<uint64_t>{29, 24, 24, 23}));
    VJ_CHECK((apportion(5, {0, 1}) == std::vector<uint64_t>{0, 5}));

    for (uint32_t n = 1; n <= 4; ++n) {
        for (int asym = 0; asym < 2; ++asym) {
            std::vector<double> bw(n, 800.0);
            if (asym) bw[0] = 1024.0;
            for (uint64_t rows : {32ull, 1000ull, 3072ull, 248320ull}) {
                const auto r = split_rows(rows, bw);
                VJ_CHECK(r.size() == n && r.front().begin == 0 && r.back().end == rows);
                uint64_t prev = 0;
                for (size_t i = 0; i < n; ++i) {
                    VJ_CHECK(r[i].begin == prev);
                    prev = r[i].end;
                    if (r[i].end != rows) VJ_CHECK(r[i].size() % kRowUnit == 0);
                }
                if (asym && n > 1 && rows >= 3072) VJ_CHECK(r[0].size() > r[1].size());
                if (!asym && rows % (32 * n) == 0) for (size_t i = 1; i < n; ++i) VJ_CHECK(r[i].size() == r[0].size());
            }
            const auto e = split_experts(256, bw);
            uint64_t tot = 0;
            for (const auto& x : e) tot += x.size();
            VJ_CHECK(tot == 256);
        }
    }
    std::printf("[PASS] b=32 largest-remainder row/expert sharding for N=1..4 (symmetric + asymmetric bandwidth)\n");

    // GQA: 32 q heads, 2 KV heads (group 16).
    const uint32_t expect_stored[5] = {0, 2, 2, 4, 4};
    for (uint32_t n = 1; n <= 4; ++n) {
        const auto p = plan_attention(32, 2, std::vector<double>(n, 800.0));
        VJ_CHECK(p.mode == AttnShardMode::HeadParallel && p.group == 16);
        uint64_t q = 0;
        for (const auto& g : p.gpus) {
            q += g.q_heads.size();
            // every q head's KV head is present locally
            for (uint64_t h = g.q_heads.begin; h < g.q_heads.end; ++h)
                VJ_CHECK(h / 16 >= g.kv_heads.begin && h / 16 < g.kv_heads.end);
        }
        VJ_CHECK(q == 32 && p.total_kv_heads_stored() == expect_stored[n]);
        std::printf("  GQA 32q/2kv N=%u:", n);
        for (const auto& g : p.gpus)
            std::printf(" [q %llu-%llu kv %llu-%llu rep %u]", (unsigned long long)g.q_heads.begin, (unsigned long long)g.q_heads.end - 1,
                        (unsigned long long)g.kv_heads.begin, (unsigned long long)g.kv_heads.end - 1, g.replicated_kv);
        std::printf("\n");
    }
    const auto p3 = plan_attention(32, 2, {800, 800, 800});
    VJ_CHECK(p3.gpus[0].replicated_kv == 1 && p3.gpus[1].replicated_kv == 2 && p3.gpus[2].replicated_kv == 1);
    // KV heads divisible by N: pure head parallel, no replication.
    const auto p8 = plan_attention(32, 8, {1, 1, 1, 1});
    VJ_CHECK(p8.total_kv_heads_stored() == 8);
    for (const auto& g : p8.gpus) VJ_CHECK(g.kv_heads.size() == 2 && g.replicated_kv == 0);
    std::printf("[PASS] GQA boundary KV head replication (straddling KV heads held by both neighbours)\n");

    // MLA (single latent KV) -> sequence-sharded split-KV, proportional to bandwidth.
    for (bool mla : {true, false}) {
        const auto pm = plan_attention(128, 1, {1024, 800}, mla);
        VJ_CHECK(pm.mode == AttnShardMode::SequenceSharded);
        VJ_CHECK(std::fabs(pm.gpus[0].seq_fraction + pm.gpus[1].seq_fraction - 1.0) < 1e-12);
        VJ_CHECK(std::fabs(pm.gpus[0].seq_fraction - 1024.0 / 1824.0) < 1e-12);
    }
    VJ_CHECK(plan_attention(128, 1, {1.0}).mode == AttnShardMode::HeadParallel);  // single GPU needs no split
    std::printf("[PASS] MLA / single-KV sequence-sharded fallback\n");

    // GDN: 16 key groups x 4 value heads.
    const auto gd = plan_gdn(16, 64, {800, 800, 800});
    VJ_CHECK(gd[0].k_groups.size() == 6 && gd[1].k_groups.size() == 5 && gd[2].k_groups.size() == 5);
    VJ_CHECK(gd[0].v_heads.size() == 24 && gd[2].v_heads.end == 64);
    std::printf("[PASS] GDN key-group sharding 16 groups -> 6/5/5\n");

    // KV tiers.
    kv::KvGeometry g{12, 1, 256, 256};
    const uint64_t q8 = kv::kv_bytes(g, 32768, kv::KvTier::Q8_0);
    VJ_CHECK(q8 == uint64_t(12) * 512 * 32768 * 34 / 32);
    auto all_q8 = kv::plan_kv(g, 32768, 1ull << 30);
    VJ_CHECK(all_q8.fits && all_q8.spans.size() == 2 && all_q8.spans[1].tier == kv::KvTier::Q8_0);
    auto mixed = kv::plan_kv(g, 262144, 1200ull << 20);
    VJ_CHECK(mixed.fits && mixed.spans[0].tier == kv::KvTier::Q8_0 && mixed.spans[1].tier == kv::KvTier::Q4_0);
    auto tq3 = kv::plan_kv(g, 262144, 800ull << 20);
    VJ_CHECK(tq3.fits && tq3.spans[1].tier == kv::KvTier::TQ3);
    VJ_CHECK(!kv::plan_kv(g, 262144, 100ull << 20).fits);
    std::printf("[PASS] KV tiers: %s | %s\n", mixed.summary().c_str(), tq3.summary().c_str());

    // ReplaySSM: bounded L=16 replay.
    ssm::GdnGeometry sg{36, 64, 128, 128, 4, 12288, 12288 + 128};
    const auto d = ssm::describe(sg);
    VJ_CHECK(d.state_bytes_per_layer == 64ull * 128 * 128 * 4 && d.checkpoint_bytes == d.live_bytes);
    ssm::ReplayRing ring;
    for (int i = 0; i < 10; ++i) VJ_CHECK(!ring.push());
    auto rb = ring.rollback(3);
    VJ_CHECK(rb.ok && rb.restore_checkpoint && rb.replay_tokens == 7);
    for (int i = 0; i < 8; ++i) ring.push();
    VJ_CHECK(ring.push());  // 16 tokens since checkpoint -> due
    ring.take_checkpoint();
    VJ_CHECK(!ring.rollback(1).ok);  // nothing after the checkpoint to roll back into
    std::printf("[PASS] ReplaySSM: %.1f MiB live state, %.1f MiB replay ring (36 layers), rollback replays <= %u tokens\n",
                d.live_bytes / 1048576.0, d.replay_ring_bytes / 1048576.0, ssm::kReplayWindow - 1);
    return 0;
}
