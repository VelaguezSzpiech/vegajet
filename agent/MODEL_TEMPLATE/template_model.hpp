// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// MODEL_TEMPLATE: compiling skeleton for a new plugin. Copy to models/<name>/, rename the namespace, class and
// architecture string, then follow agent/ONBOARD_MODEL.md steps 3-5. This copy is built in CI so it stays in
// sync with the VegaJetModel contract.
#pragma once

#include "core/model_interface.hpp"
#include "core/quant_repacker.hpp"
#include "core/sharder.hpp"

#include <string>
#include <vector>

namespace vegajet::model_template {

constexpr const char* kArchitecture = "template";  // TODO: the GGUF general.architecture value

struct Hparams {
    uint32_t n_layer = 0, n_embd = 0, n_head = 0, n_head_kv = 0, head_dim = 0, n_ff = 0;
    uint64_t vocab = 0;
};

struct PlannedTensor {
    std::string name;
    quant::NativeFormat format = quant::NativeFormat::VJ_Q8_DP4A;
    uint64_t in = 0, out = 0;
    std::vector<shard::Range> rows_per_gpu;
};

class TemplateModel final : public VegaJetModel {
public:
    bool can_load(const GgufFile& f) const override;
    LoadReport load(GgufFile& f, const LoadOptions& opt) override;
    StepResult prefill(const std::vector<int32_t>& tokens) override;
    StepResult decode_step(int32_t token, uint32_t verify_len) override;
    StepResult draft_step(const std::vector<int32_t>& context, uint32_t max_draft) override;
    std::string name() const override { return kArchitecture; }
    const Hparams& hparams() const { return hp_; }

private:
    bool parse_hparams(const GgufFile& f, std::string* err);
    void emit_layer(ops::OpGraph& g, uint32_t layer, uint32_t m, bool prefill) const;
    Hparams hp_;
    std::vector<double> bw_;
    std::vector<PlannedTensor> plan_;
    bool loaded_ = false;
};

} // namespace vegajet::model_template
