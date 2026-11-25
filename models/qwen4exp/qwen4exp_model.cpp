// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "models/qwen4exp/qwen4exp_model.hpp"

namespace vegajet::qwen4exp {

bool Qwen4ExpModel::can_load(const GgufFile& f) const {
    return f.architecture() == "qwen4exp";
}

LoadReport Qwen4ExpModel::load(GgufFile& f, const LoadOptions&) {
    LoadReport rep;
    rep.architecture = f.architecture();
    if (!can_load(f)) { rep.error = "not a qwen4exp GGUF"; return rep; }
    f.get_arch_u64("block_count", hp_.n_layer);
    f.get_arch_u64("embedding_length", hp_.n_embd);
    f.get_arch_u64("attention.head_count", hp_.n_head);
    f.get_arch_u64("attention.head_count_kv", hp_.n_head_kv);
    f.get_arch_u64("expert_count", hp_.n_expert);
    f.get_arch_u64("expert_used_count", hp_.n_expert_used);
    for (const auto& t : f.tensors()) rep.weight_bytes_disk += t.n_bytes();
    rep.notes.push_back("placeholder plugin: metadata only, no op graph");
    rep.ok = hp_.n_layer > 0;
    if (!rep.ok) rep.error = "missing qwen4exp.block_count";
    return rep;
}

StepResult Qwen4ExpModel::prefill(const std::vector<int32_t>& tokens) {
    StepResult r;
    r.tokens_in = static_cast<uint32_t>(tokens.size());
    return r;
}

StepResult Qwen4ExpModel::decode_step(int32_t, uint32_t verify_len) {
    StepResult r;
    r.tokens_in = verify_len;
    return r;
}

StepResult Qwen4ExpModel::draft_step(const std::vector<int32_t>&, uint32_t) { return {}; }

} // namespace vegajet::qwen4exp
