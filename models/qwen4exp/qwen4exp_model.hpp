// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Qwen4Exp placeholder plugin: reads hyper-parameters through the shared GGUF reader so the registry can
// identify the architecture. The full port (hyper-connections, PLE, GDN) lives outside this repository
// until it is rebuilt on the core/ chassis; step functions therefore emit empty graphs.
#pragma once

#include "core/model_interface.hpp"

namespace vegajet::qwen4exp {

struct Hparams {
    uint64_t n_layer = 0, n_embd = 0, n_head = 0, n_head_kv = 0, n_expert = 0, n_expert_used = 0;
};

class Qwen4ExpModel final : public VegaJetModel {
public:
    bool can_load(const GgufFile& f) const override;
    LoadReport load(GgufFile& f, const LoadOptions& opt) override;
    StepResult prefill(const std::vector<int32_t>& tokens) override;
    StepResult decode_step(int32_t token, uint32_t verify_len) override;
    StepResult draft_step(const std::vector<int32_t>& context, uint32_t max_draft) override;
    std::string name() const override { return "qwen4exp"; }
    const Hparams& hparams() const { return hp_; }

private:
    Hparams hp_;
};

} // namespace vegajet::qwen4exp
