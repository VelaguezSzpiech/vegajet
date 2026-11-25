// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// ReplaySSM: recurrent (gated-delta-net) state rollback for speculative decoding without keeping one
// state copy per draft token. A checkpoint of the full state is taken every kReplayWindow tokens, and the
// last kReplayWindow token inputs are kept in a ring; rolling back r tokens restores the checkpoint and
// replays at most kReplayWindow - 1 inputs.
#pragma once

#include <cstdint>
#include <vector>

namespace vegajet::ssm {

constexpr uint32_t kReplayWindow = 16;  // L

struct GdnGeometry {
    uint32_t n_layers = 0;        // recurrent layers on this GPU
    uint32_t n_v_heads = 0;       // value heads on this GPU
    uint32_t head_k_dim = 0;      // state rows (d_state)
    uint32_t head_v_dim = 0;      // state cols
    uint32_t conv_kernel = 4;
    uint32_t conv_channels = 0;   // q+k+v channels through the short conv on this GPU
    uint32_t replay_input_elems = 0;  // per-token per-layer inputs kept for replay (conv input + beta/alpha)
};

struct SsmDescriptor {
    uint64_t state_bytes_per_layer = 0;       // FP32 S matrices
    uint64_t conv_bytes_per_layer = 0;        // FP32 conv tail (kernel-1 tokens)
    uint64_t live_bytes = 0;                  // all layers: state + conv
    uint64_t checkpoint_bytes = 0;            // one full checkpoint copy
    uint64_t replay_ring_bytes = 0;           // L token inputs, all layers
    uint64_t total_bytes() const { return live_bytes + checkpoint_bytes + replay_ring_bytes; }
};

SsmDescriptor describe(const GdnGeometry& g);

// Host-side bookkeeping of the checkpoint/replay ring (the device executes the matching ops).
class ReplayRing {
public:
    // Records that a token was committed to the live state. Returns true when a checkpoint is due.
    bool push();
    struct Rollback { bool restore_checkpoint = false; uint32_t replay_tokens = 0; bool ok = false; };
    // Undo the newest `r` tokens. ok=false if r exceeds what the window can reconstruct.
    Rollback rollback(uint32_t r);
    void take_checkpoint();
    uint64_t position() const { return pos_; }
    uint64_t checkpoint_position() const { return ckpt_pos_; }

private:
    uint64_t pos_ = 0;       // tokens in live state
    uint64_t ckpt_pos_ = 0;  // tokens covered by the checkpoint
};

} // namespace vegajet::ssm
