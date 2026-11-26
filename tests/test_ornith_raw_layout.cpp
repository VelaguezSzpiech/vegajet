// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "models/ornith/ornith_tp_exec.hpp"
#include "core/q4_dot8_repack.hpp"
#include "tests/test_common.hpp"
#include <algorithm>
#include <set>
#include <stdexcept>

using namespace vegajet;
using namespace vegajet::ornith;

static Hparams geometry() {
    Hparams h;
    h.n_embd = 3072; h.n_head = 32; h.n_head_kv = 2;
    h.head_dim_k = h.head_dim_v = 256; h.rope_dims = 64;
    h.ssm_groups = 16; h.ssm_v_heads = 64; h.ssm_state = 128; h.ssm_conv = 4;
    h.n_ff_exp = h.n_ff_shexp = 1024; h.n_expert = 256; h.n_expert_used = 8;
    h.vocab = 131072; h.n_trunk = 48; h.n_gdn = 36; h.n_attn = 12;
    h.rms_eps = 1e-6; h.rope_base = 1e7;
    for (uint32_t l = 0; l < h.n_trunk; ++l) h.recurrent.push_back((l + 1) % 4 != 0);
    return h;
}
static std::vector<tp::DeviceProfile> devices(uint32_t n, uint64_t capacity = 128ULL << 30) {
    std::vector<tp::DeviceProfile> d(n);
    for (uint32_t r = 0; r < n; ++r) {
        d[r].unique_id = "rank" + std::to_string(r);
        // Valid hex IDs, independent of host ordinals/BDFs.
        d[r].unique_id = std::to_string(r + 1);
        d[r].bw_hbm_gbs = 1; d[r].usable_vram_bytes = capacity; d[r].n_cu = 60; d[r].mclk = 1;
    }
    return d;
}
static void verify(uint32_t n, bool replication) {
    const auto h = geometry();
    tp::ShardOptions options; options.replicate_kv = replication;
    const auto plan = raw_layout(h, 8192, devices(n), options);
    VJ_CHECK(plan.ranks.size() == n && plan.rank_order.size() == n);
    std::vector<unsigned> q_seen(32), g_seen(16), v_seen(64);
    uint64_t ff = 0, shared = 0, vocab = 0;
    std::vector<unsigned> launches(n);
    for (size_t r : plan.rank_order) {
        VJ_CHECK(r < n); launches[r]++;
        const auto& m = plan.ranks[r];
        VJ_CHECK(m.ff.begin == ff && m.shared_ff.begin == shared && m.vocab.begin == vocab);
        ff = m.ff.end; shared = m.shared_ff.end; vocab = m.vocab.end;
        VJ_CHECK(m.ff.size() % 32 == 0 && m.shared_ff.size() % 32 == 0);
        VJ_CHECK(m.attn.q_heads.size() == m.attn.q_to_local_kv.size());
        for (size_t i = 0; i < m.attn.q_heads.size(); ++i) {
            const auto q = m.attn.q_heads[i], local = m.attn.q_to_local_kv[i];
            VJ_CHECK(q < 32 && local < m.attn.kv_heads.size());
            VJ_CHECK(m.attn.kv_heads[local] == q / 16); q_seen[q]++;
            for (uint32_t c = 0; c < 512; ++c) VJ_CHECK(m.q_rows[i * 512 + c] == q * 512 + c);
        }
        for (auto g : m.groups) { VJ_CHECK(g < 16); g_seen[g]++; }
        for (size_t i = 0; i < m.v_heads.size(); ++i) {
            const auto v = m.v_heads[i]; VJ_CHECK(v < 64); v_seen[v]++;
            VJ_CHECK(v % 16 == m.groups[i % m.groups.size()]);
            VJ_CHECK(v / 16 == i / m.groups.size());
        }
        const auto nk = m.groups.size(), nv = m.v_heads.size();
        VJ_CHECK(nv == nk * 4 && m.conv_rows.size() == (2 * nk + nv) * 128);
        for (size_t k = 0; k < nk; ++k) for (unsigned c = 0; c < 128; ++c) {
            VJ_CHECK(m.conv_rows[k * 128 + c] == m.groups[k] * 128 + c);
            VJ_CHECK(m.conv_rows[(nk + k) * 128 + c] == 2048 + m.groups[k] * 128 + c);
        }
        for (size_t v = 0; v < nv; ++v) for (unsigned c = 0; c < 128; ++c)
            VJ_CHECK(m.conv_rows[(2 * nk + v) * 128 + c] == 4096 + m.v_heads[v] * 128 + c);
        VJ_CHECK(m.state_bytes == nv * 128 * 128 * 4);
        VJ_CHECK(m.conv_bytes == m.conv_rows.size() * 3 * 4);
        VJ_CHECK(m.kv_bytes == 8192ULL * m.attn.kv_heads.size() * 544);
        const auto& b = m.scratch;
        VJ_CHECK(b.hidden == 3072 * 4 && b.xq == 3072 && b.xd == 96 * 8);
        VJ_CHECK(b.proj >= (m.conv_rows.size() + nv * 130) * 4);
        VJ_CHECK(b.conv_out >= m.conv_rows.size() * 4 && b.gates >= nv * 4);
        VJ_CHECK(b.aq >= nv * 128 && b.ad >= nv * 4 * 8);
        VJ_CHECK(b.aq >= m.attn.q_heads.size() * 256 && b.attn >= m.attn.q_heads.size() * 256 * 4);
        VJ_CHECK(b.logits == m.vocab.size() * 4 && b.head_map == m.attn.q_heads.size() * 4);
        VJ_CHECK(b.shared_gu == m.shared_ff.size() * 8 && b.shared_q == m.shared_ff.size());
        VJ_CHECK(b.shared_ds == m.shared_ff.size() / 32 * 8);
        VJ_CHECK(b.moe_q == m.ff.size() * 8 && b.moe_ds == m.ff.size());
        if (m.ff.size()) {
            const auto packed = dot8_iq3_asym_layout(256ULL * 3072, m.ff.size());
            VJ_CHECK(plan.shards.ranks[r].rows[0].bytes == uint64_t(h.n_trunk) *
                (256ULL * 1968 * m.ff.size() + packed.w_plane_bytes() + packed.s_plane_bytes()));
        }
    }
    VJ_CHECK(ff == 1024 && shared == 1024 && vocab == h.vocab);
    for (auto c : launches) VJ_CHECK(c == 1);
    for (auto c : q_seen) VJ_CHECK(c == 1);
    for (auto c : g_seen) VJ_CHECK(c == 1);
    for (auto c : v_seen) VJ_CHECK(c == 1);
    const auto ar = tp::plan_collective(3072, tp::PeerMatrix(n, std::vector<tp::PeerLink>(n)), {});
    VJ_CHECK(ar.ranks == n && ar.bypass == (n == 1));
    // All ranks, including zero-work attention shards, participate at both sites/layer.
    std::vector<unsigned> reductions(n);
    for (unsigned l = 0; l < h.n_trunk; ++l) for (unsigned site = 0; site < 2; ++site)
        for (auto r : plan.rank_order) reductions[r]++;
    for (auto c : reductions) VJ_CHECK(c == h.n_trunk * 2);
}
int main() {
    for (auto n : {1u, 2u, 3u, 4u, 8u}) for (bool replicate : {false, true}) verify(n, replicate);
    const auto h = geometry();
    // Real three-card arena ceiling; catches accidentally charging replicated/flattened expert pools.
    const auto p3 = raw_layout(h, 8192, devices(3, (61ULL << 29)));
    VJ_CHECK(p3.ranks.size() == 3);
    auto caps = devices(3);
    for (size_t r = 0; r < 3; ++r) caps[r].usable_vram_bytes = p3.shards.ranks[r].used_bytes;
    VJ_CHECK(raw_layout(h, 8192, caps).ranks.size() == 3);
    --caps[0].usable_vram_bytes;
    bool insufficient = false;
    try { (void)raw_layout(h, 8192, caps); } catch (const std::runtime_error&) { insufficient = true; }
    VJ_CHECK(insufficient);
    // Asymmetric IQ3 uses either inline scales or a separate plane, depending on the owner shard.
    for (uint32_t ff : {32u, 320u, 352u, 512u, 1024u}) {
        const auto down = dot8_iq3_asym_layout(256ULL * 3072, ff);
        if (down.scale_embedded) {
            VJ_CHECK(down.s_plane_bytes() == 0 && down.s_offset_bytes == ff / 2);
            VJ_CHECK(down.s_offset_bytes + ff / 32 * 8 <= down.w_row_stride_bytes);
        } else VJ_CHECK(down.s_plane_bytes() == 256ULL * 3072 * ff / 32 * 8);
    }
    tp::ShardOptions owner;
    owner.tensors["expert_ff"].rows = {0, 1024, 0};
    owner.tensors["shared_ff"].rows = {1024, 0, 0};
    owner.tensors["vocab"].rows = {0, 0, h.vocab};
    const auto p = raw_layout(h, 16, devices(3), owner);
    VJ_CHECK(p.ranks[0].ff.size() == 0 && p.ranks[1].ff.size() == 1024 && p.ranks[2].ff.size() == 0);
    VJ_CHECK(p.ranks[1].scratch.shared_q == 0 && p.ranks[2].vocab.size() == h.vocab);
    bool failed = false;
    try { (void)raw_layout(h, 8193, devices(3)); } catch (const std::invalid_argument&) { failed = true; }
    VJ_CHECK(failed);
    failed = false; owner.row_tile = 16;
    try { (void)raw_layout(h, 16, devices(3), owner); } catch (const std::invalid_argument&) { failed = true; }
    VJ_CHECK(failed);
    for (auto n : {1u, 2u, 3u, 4u, 8u}) {
        OrnithGpuOptions options; options.tp = n;
        VJ_CHECK(ornith_raw_admit(options).empty());
        options.use_suffix_spec = true;
        VJ_CHECK(ornith_raw_admit(options).empty() == (n == 2));
        options.use_suffix_spec = false; options.dump_logits_path = "output";
        VJ_CHECK(ornith_raw_admit(options).empty() == (n == 2));
        options.dump_logits_path.clear(); options.max_ctx = 8193;
        VJ_CHECK(ornith_raw_admit(options).empty() == (n == 2));
        options.max_ctx = 8192; options.ar_wire_int8 = true;
        VJ_CHECK(ornith_raw_admit(options).empty() == (n == 2));
    }
    VJ_CHECK(!ornith_admit(257, {}).empty());
    std::puts("Ornith raw mappings, explicit head maps, buffers and all rank loops N=1,2,3,4,8: PASS");
    return 0;
}
