// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/ssm_state.hpp"

namespace vegajet::ssm {

SsmDescriptor describe(const GdnGeometry& g) {
    SsmDescriptor d;
    d.state_bytes_per_layer = uint64_t(g.n_v_heads) * g.head_k_dim * g.head_v_dim * 4;
    d.conv_bytes_per_layer = uint64_t(g.conv_kernel > 0 ? g.conv_kernel - 1 : 0) * g.conv_channels * 4;
    d.live_bytes = (d.state_bytes_per_layer + d.conv_bytes_per_layer) * g.n_layers;
    d.checkpoint_bytes = d.live_bytes;
    d.replay_ring_bytes = uint64_t(kReplayWindow) * g.replay_input_elems * 4 * g.n_layers;
    return d;
}

bool ReplayRing::push() {
    ++pos_;
    return pos_ - ckpt_pos_ >= kReplayWindow;
}

void ReplayRing::take_checkpoint() { ckpt_pos_ = pos_; }

ReplayRing::Rollback ReplayRing::rollback(uint32_t r) {
    Rollback out;
    if (r == 0) { out.ok = true; return out; }
    if (r > pos_ - ckpt_pos_) return out;  // older than the checkpoint: inputs are gone
    pos_ -= r;
    out.restore_checkpoint = true;
    out.replay_tokens = static_cast<uint32_t>(pos_ - ckpt_pos_);
    out.ok = out.replay_tokens < kReplayWindow;
    return out;
}

} // namespace vegajet::ssm
