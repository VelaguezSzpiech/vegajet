// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/tp_executor.hpp"
#include "core/tp_json.hpp"

#include <algorithm>
#include <limits>
#include <sstream>

namespace vegajet::tp {
using namespace detail;
namespace {

uint64_t align64(uint64_t n) { return mul((add(n, 63)) / 64, 64); }

// Row index of a named tensor in the shard plan, which preserves the caller's order.
size_t row_index(const ModelSpec& model, const std::string& name) {
    for (size_t i = 0; i < model.rows.size(); ++i)
        if (model.rows[i].name == name) return i;
    return model.rows.size();
}

std::string bytes_text(uint64_t bytes) {
    std::ostringstream o;
    o.setf(std::ios::fixed);
    o.precision(3);
    o << (static_cast<double>(bytes) / 1073741824.0) << " GiB";
    return o.str();
}

} // namespace

std::vector<FeatureUse> feature_uses(const FeatureFlags& f) {
    return {
        {"super-kernels", "VEGAJET_ENABLE_SUPER", f.super_kernels, 2, 2},
        {"D1 asymmetric expert layout", "VEGAJET_V12_ASYM_EXPERTS", f.asym_experts, 2, 2},
        {"MTP draft head", "draft/MTP placement", f.mtp_draft, 2, 2},
        {"speculative draft-verify", "suffix/DFlash2 speculation", f.speculative, 2, 2},
        {"INT8 all-reduce wire", "VEGAJET_V12_AR_INT8", f.int8_wire, 2, 2},
        {"fused verify+norm", "VEGAJET_V12_VERIFY_FUSED_NQ", f.fused_verify, 2, 2},
        {"W4A4 prefill", "--prefill-dot8", f.prefill_dot8, 2, 2},
        {"multi-slot ReplaySSM ring", "spec GDN state ring", f.ring_state, 2, 2},
    };
}

std::string admit_features(uint32_t ranks, const FeatureFlags& features) {
    std::string why;
    for (const auto& u : feature_uses(features)) {
        if (!u.enabled || (ranks >= u.min_ranks && ranks <= u.max_ranks)) continue;
        if (!why.empty()) why += "; ";
        why += std::string(u.feature) + " (" + u.flag + ") supports TP=";
        if (u.min_ranks == u.max_ranks) {
            why += std::to_string(u.min_ranks);
        } else {
            why += std::to_string(u.min_ranks) + ".." + std::to_string(u.max_ranks);
        }
        why += ", not TP=" + std::to_string(ranks);
    }
    if (!why.empty())
        why = "generic N-rank executor: " + why +
              "; disable the feature or select the supported rank count (owner decision)";
    return why;
}

uint64_t RankLayout::total_bytes() const {
    uint64_t t = weight_bytes;
    for (uint64_t b : {hidden_bytes, partial_bytes, logits_bytes, vocab_gather_bytes,
                       rx_payload_bytes, rx_flag_bytes})
        t = add(t, b);
    return t;
}

uint64_t RankLayout::headroom_bytes() const {
    const uint64_t used = total_bytes();
    return used >= usable_vram_bytes ? 0 : usable_vram_bytes - used;
}

std::vector<ArSlice> plan_ar_slices(uint32_t m, uint32_t elements, uint32_t max_frame_bytes,
                                    uint32_t first_epoch, uint32_t* next_epoch) {
    require(m > 0, "all-reduce needs at least one row");
    require(elements > 0, "all-reduce needs a positive row length");
    require(max_frame_bytes > 0, "all-reduce frame bound must be positive");
    const uint64_t wire = align64(mul(2, elements));  // FP16 wire, ABI 1
    if (wire > max_frame_bytes)
        throw std::invalid_argument("all-reduce row length " + std::to_string(elements) +
                                    " exceeds the " + std::to_string(max_frame_bytes) + " B frame bound");
    const uint64_t rows_per = std::max<uint64_t>(1, max_frame_bytes / wire);
    const uint64_t slices = (m + rows_per - 1) / rows_per;
    // Epochs must stay representable: the wire never carries zero and never wraps.
    if (static_cast<uint64_t>(first_epoch) + slices - 1 >= std::numeric_limits<uint32_t>::max())
        throw std::overflow_error("all-reduce epoch budget exhausted for this step");
    std::vector<ArSlice> out;
    out.reserve(static_cast<size_t>(slices));
    for (uint64_t s = 0; s < slices; ++s) {
        const uint64_t offset = s * rows_per;
        const uint64_t rows = std::min<uint64_t>(rows_per, m - offset);
        ArSlice sl;
        sl.offset = static_cast<uint32_t>(offset);
        sl.rows = static_cast<uint32_t>(rows);
        sl.epoch = first_epoch + static_cast<uint32_t>(s);
        sl.bank = sl.epoch & 1u;
        sl.drain = offset + rows < m;  // a further slice reuses the frame: quiesce first
        out.push_back(sl);
    }
    if (next_epoch) *next_epoch = first_epoch + static_cast<uint32_t>(slices);
    return out;
}

ExecutorLayout plan_executor(const ModelSpec& model, const std::vector<DeviceProfile>& devices,
                             const PeerMatrix& peers, const ShardOptions& options,
                             const ExecutorRequest& request) {
    require(request.m > 0, "executor needs at least one row of tokens");
    require(request.d_model > 0, "executor needs d_model");
    require(request.layers > 0, "executor needs at least one layer");
    require(request.ar_sites_per_layer > 0, "executor needs at least one reduction per layer");
    require(!model.rows.empty(), "executor needs at least one row-parallel tensor");
    if (!peers.empty() && peers.size() != devices.size())
        throw std::invalid_argument("peer matrix rank count does not match the device list");
    std::string vocab_name;
    for (const auto& r : model.rows)
        if (r.kind == RowKind::Vocabulary) { vocab_name = r.name; break; }

    const ShardPlan shards = plan_shards(model, devices, options);

    ExecutorLayout layout;
    layout.ranks = static_cast<uint32_t>(devices.size());
    layout.m = request.m;
    layout.d_model = request.d_model;
    layout.layers = request.layers;
    layout.ar_sites_per_layer = request.ar_sites_per_layer;
    layout.vocab_merge = request.vocab_merge;
    layout.max_frame_bytes = request.max_frame_bytes;

    if (peers.empty()) {
        // Unknown links: every directed edge is non-P2P, exactly as an absent profile plans.
        const PeerMatrix unknown(layout.ranks, std::vector<PeerLink>(layout.ranks));
        layout.collective = plan_collective(request.d_model, unknown, CollectiveOptions{});
    } else {
        layout.collective = plan_collective(request.d_model, peers, CollectiveOptions{});
    }
    layout.ar_bypass = layout.collective.bypass;
    layout.frames_per_rank = layout.ar_bypass ? 0u : layout.collective.ranks;

    const size_t vocab_slot = vocab_name.empty() ? model.rows.size() : row_index(model, vocab_name);
    uint64_t vocab_total = 0;
    if (vocab_slot < model.rows.size()) vocab_total = model.rows[vocab_slot].rows;

    // One reduction schedule per step, epochs in canonical order, banks alternating.
    uint32_t epoch = 1;
    uint32_t site_index = 0;
    auto add_site = [&](int32_t layer) {
        ArSite site;
        site.index = site_index++;
        site.layer = layer;
        site.elements = request.d_model;
        site.wire_bytes = align64(mul(2, request.d_model));
        if (!layout.ar_bypass)
            site.slices = plan_ar_slices(request.m, request.d_model, request.max_frame_bytes, epoch, &epoch);
        layout.sites.push_back(std::move(site));
    };
    for (uint32_t l = 0; l < request.layers; ++l)
        for (uint32_t s = 0; s < request.ar_sites_per_layer; ++s) add_site(static_cast<int32_t>(l));
    if (request.vocab_merge) add_site(-1);
    layout.epochs_per_step = layout.ar_bypass ? 0u : epoch - 1u;

    const uint32_t rows_per_slice = layout.ar_bypass || layout.sites.empty() || layout.sites[0].slices.empty()
                                        ? request.m
                                        : layout.sites[0].slices[0].rows;
    const uint64_t hidden = mul(mul(request.m, request.d_model), 4);
    for (uint32_t r = 0; r < layout.ranks; ++r) {
        const auto& shard = shards.ranks[r];
        RankLayout rank;
        rank.rank = r;
        rank.device_key = shard.device_key;
        rank.usable_vram_bytes = devices[r].usable_vram_bytes;
        rank.weight_bytes = shard.used_bytes;
        rank.hidden_bytes = hidden;
        rank.partial_bytes = layout.ar_bypass ? 0 : mul(mul(rows_per_slice, request.d_model), 4);
        rank.rx_payload_bytes = layout.collective.rx_payload_bytes_per_rank;
        rank.rx_flag_bytes = layout.collective.rx_flag_bytes_per_rank;
        if (vocab_slot < model.rows.size()) {
            rank.vocab = shard.rows[vocab_slot].rows;
            rank.logits_bytes = mul(rank.vocab.size(), 4);
            if (request.vocab_merge && r == layout.ranks - 1)
                rank.vocab_gather_bytes = mul(vocab_total, 4);
        }
        if (!shard.attention.empty()) {
            const HeadMap& attn = shard.attention[0];
            rank.q_heads = static_cast<uint32_t>(attn.q_heads.size());
            rank.kv_heads = static_cast<uint32_t>(attn.kv_heads.size());
        }
        if (shard.attention.size() > 1) {
            const HeadMap& gdn = shard.attention[1];
            rank.k_groups = static_cast<uint32_t>(gdn.kv_heads.size());
            rank.v_heads = static_cast<uint32_t>(gdn.q_heads.size());
        }
        if (!layout.ar_bypass)
            rank.pushes_per_epoch = layout.ranks - 1u;
        const uint64_t used = rank.total_bytes();
        if (used > rank.usable_vram_bytes)
            throw std::runtime_error("executor rank " + std::to_string(r) + " (" + rank.device_key +
                                     ") needs " + bytes_text(used) + " of " +
                                     bytes_text(rank.usable_vram_bytes) + " usable VRAM");
        layout.rank.push_back(std::move(rank));
    }
    return layout;
}

bool check_executor_layout(const ExecutorLayout& layout, const ModelSpec& model, std::string* err) {
    auto bad = [&](const std::string& why) {
        if (err) *err = "executor layout: " + why;
        return false;
    };
    if (layout.ranks == 0 || layout.rank.size() != layout.ranks)
        return bad("rank table does not cover the planned rank count");
    if (layout.m == 0 || layout.d_model == 0 || layout.layers == 0)
        return bad("empty step geometry");
    if (layout.ar_bypass != (layout.ranks == 1))
        return bad("all-reduce bypass must hold exactly when one rank is planned");
    if (!layout.ar_bypass && layout.collective.routes.size() != layout.ranks)
        return bad("collective routes do not cover the planned ranks");

    // Rows: the vocabulary partition must cover the tensor exactly once.
    std::string vocab_name;
    for (const auto& r : model.rows)
        if (r.kind == RowKind::Vocabulary) { vocab_name = r.name; break; }
    if (!vocab_name.empty()) {
        const size_t slot = row_index(model, vocab_name);
        uint64_t pos = 0;
        for (const auto& r : layout.rank) {
            if (r.rank >= layout.ranks) return bad("rank index out of range");
            if (r.vocab.begin != pos) return bad("vocabulary rows are not contiguous on rank " + std::to_string(r.rank));
            if (r.vocab.end < r.vocab.begin) return bad("inverted vocabulary range on rank " + std::to_string(r.rank));
            if (slot < model.rows.size() && r.vocab.end > model.rows[slot].rows)
                return bad("vocabulary range exceeds the tensor on rank " + std::to_string(r.rank));
            pos = r.vocab.end;
        }
        if (slot < model.rows.size() && pos != model.rows[slot].rows)
            return bad("vocabulary partition covers " + std::to_string(pos) + " of " +
                       std::to_string(model.rows[slot].rows) + " rows");
    }

    // Sizes implied by the maps.
    const uint64_t hidden = mul(mul(layout.m, layout.d_model), 4);
    for (const auto& r : layout.rank) {
        if (r.hidden_bytes != hidden)
            return bad("hidden buffer does not match m*d_model*4 on rank " + std::to_string(r.rank));
        if (r.logits_bytes != mul(r.vocab.size(), 4))
            return bad("logit buffer does not match the vocabulary rows on rank " + std::to_string(r.rank));
        if (layout.ar_bypass && (r.rx_payload_bytes || r.rx_flag_bytes || r.partial_bytes))
            return bad("bypass plan allocates collective storage on rank " + std::to_string(r.rank));
        if (!layout.ar_bypass && r.rx_payload_bytes != layout.collective.rx_payload_bytes_per_rank)
            return bad("RX payload bytes disagree with the collective plan on rank " + std::to_string(r.rank));
        if (!layout.ar_bypass) {
            uint32_t pushes = 0;
            for (const auto& t : layout.collective.routes[r.rank])
                if (t != Transport::Local) ++pushes;
            if (pushes != r.pushes_per_epoch)
                return bad("push count disagrees with the collective routes on rank " + std::to_string(r.rank));
        }
        if (r.q_heads && r.kv_heads && r.kv_heads > r.q_heads)
            return bad("more KV heads than Q heads on rank " + std::to_string(r.rank));
        if (r.total_bytes() > r.usable_vram_bytes)
            return bad("rank " + std::to_string(r.rank) + " over capacity");
    }

    // Schedule: every site reduced, epochs strictly increasing, banks alternating, frames
    // drained before reuse.
    const uint64_t expected_sites = static_cast<uint64_t>(layout.layers) * layout.ar_sites_per_layer +
                                    (layout.vocab_merge ? 1u : 0u);
    if (layout.sites.size() != expected_sites)
        return bad("site count " + std::to_string(layout.sites.size()) + " does not match the layer schedule");
    uint32_t last_epoch = 0;
    for (const auto& s : layout.sites) {
        if (s.elements != layout.d_model) return bad("reduction width does not match d_model at site " + std::to_string(s.index));
        if (s.wire_bytes != align64(mul(2, layout.d_model)))
            return bad("wire bytes do not match a FP16 row at site " + std::to_string(s.index));
        if (layout.ar_bypass) {
            if (!s.slices.empty()) return bad("bypass plan carries slices at site " + std::to_string(s.index));
            continue;
        }
        uint64_t covered = 0;
        for (size_t k = 0; k < s.slices.size(); ++k) {
            const auto& sl = s.slices[k];
            if (sl.rows == 0) return bad("empty slice at site " + std::to_string(s.index));
            if (sl.offset != covered) return bad("slice offset gap at site " + std::to_string(s.index));
            if (sl.epoch == 0) return bad("epoch zero on the wire at site " + std::to_string(s.index));
            if (sl.epoch <= last_epoch) return bad("epochs are not strictly increasing at site " + std::to_string(s.index));
            if (sl.bank != (sl.epoch & 1u)) return bad("bank does not follow the epoch parity at site " + std::to_string(s.index));
            if (align64(mul(2, s.elements)) > layout.max_frame_bytes)
                return bad("slice exceeds the frame bound at site " + std::to_string(s.index));
            if (sl.rows > layout.m || sl.offset > layout.m - sl.rows)
                return bad("slice exceeds the step at site " + std::to_string(s.index));
            if (sl.drain != (k + 1 < s.slices.size())) return bad("drain flag does not match slice reuse at site " + std::to_string(s.index));
            covered += sl.rows;
            last_epoch = sl.epoch;
        }
        if (covered != layout.m) return bad("site " + std::to_string(s.index) + " covers " + std::to_string(covered) + " of " + std::to_string(layout.m) + " rows");
    }
    if (!layout.ar_bypass && layout.epochs_per_step != last_epoch)
        return bad("step epoch count does not match the schedule");
    return true;
}

std::string format_executor_layout(const ExecutorLayout& l) {
    std::ostringstream o;
    o << "generic executor layout: TP=" << l.ranks << " m=" << l.m << " d_model=" << l.d_model
      << " layers=" << l.layers << " sites=" << l.sites.size()
      << " epochs/step=" << l.epochs_per_step << " bypass=" << (l.ar_bypass ? 1 : 0)
      << " banks=" << l.collective.banks << " frames/rank=" << l.frames_per_rank << "\n";
    for (const auto& r : l.rank) {
        o << "  rank " << r.rank << " " << r.device_key << " weights=" << bytes_text(r.weight_bytes)
          << " hidden=" << bytes_text(r.hidden_bytes) << " partial=" << bytes_text(r.partial_bytes)
          << " logits=" << bytes_text(r.logits_bytes) << " rx=" << bytes_text(r.rx_payload_bytes)
          << " flags=" << bytes_text(r.rx_flag_bytes) << " total=" << bytes_text(r.total_bytes())
          << " of " << bytes_text(r.usable_vram_bytes)
          << " | vocab=[" << r.vocab.begin << ',' << r.vocab.end << ')';
        if (r.q_heads || r.kv_heads) o << " q=" << r.q_heads << " kv=" << r.kv_heads;
        if (r.k_groups || r.v_heads) o << " gdn_groups=" << r.k_groups << " gdn_v=" << r.v_heads;
        o << " pushes=" << r.pushes_per_epoch << "\n";
    }
    uint64_t slices = 0;
    for (const auto& s : l.sites) slices += s.slices.size();
    o << "  reductions: " << l.sites.size() << " sites, " << slices << " slices, "
      << l.epochs_per_step << " epochs/step";
    if (!l.sites.empty() && !l.sites.front().slices.empty())
        o << ", first slice rows=" << l.sites.front().slices.front().rows;
    o << "\n";
    return o.str();
}

} // namespace vegajet::tp
