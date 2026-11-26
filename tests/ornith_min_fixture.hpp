// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Minimal qwen35moe GGUF that reaches OrnithModel::parse_hparams() with caller-chosen numbers.
// Only the metadata `parse_hparams()` reads plus the two tensors `can_load()` looks for.
#pragma once

#include "core/gguf_reader.hpp"
#include "tests/test_common.hpp"

namespace vjtest {

struct OrnithHparams {
    uint32_t block_count = 1;
    bool nextn_present = true;
    uint64_t nextn = 0;              // written as u32 when it fits, u64 otherwise
    uint32_t embedding_length = 256;
    uint32_t head_count = 4;
    uint32_t head_count_kv = 2;
    uint32_t full_attn_interval = 4;
};

inline std::string write_min_ornith(const OrnithHparams& p, const std::string& tag) {
    RawGguf g(2, 0);
    const std::string a = "qwen35moe";
    g.kv_str("general.architecture", a);
    g.kv_u32(a + ".block_count", p.block_count);
    g.kv_u32(a + ".embedding_length", p.embedding_length);
    g.kv_u32(a + ".attention.head_count", p.head_count);
    g.kv_u32(a + ".attention.head_count_kv", p.head_count_kv);
    g.kv_u32(a + ".expert_count", 8);
    g.kv_u32(a + ".expert_used_count", 2);
    g.kv_u32(a + ".ssm.conv_kernel", 4);
    g.kv_u32(a + ".ssm.state_size", 32);
    g.kv_u32(a + ".ssm.group_count", 2);
    g.kv_u32(a + ".ssm.time_step_rank", 4);
    g.kv_u32(a + ".ssm.inner_size", 128);
    g.kv_u32(a + ".full_attention_interval", p.full_attn_interval);
    if (p.nextn_present) {
        if (p.nextn > 0xFFFFFFFFull) g.kv_u64(a + ".nextn_predict_layers", p.nextn);
        else g.kv_u32(a + ".nextn_predict_layers", static_cast<uint32_t>(p.nextn));
    }
    // can_load() signature tensors: 256 F32 at offset 0, then 4 F32 at offset 1024.
    g.tensor("blk.0.ffn_gate_exps.weight", {32, 8}, 0, 0);
    g.tensor("blk.0.ssm_a", {4}, 0, 1024);
    return write_temp(g.build(1056), tag);
}

} // namespace vjtest
