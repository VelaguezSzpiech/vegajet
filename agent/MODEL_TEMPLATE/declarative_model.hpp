// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// MODEL_TEMPLATE (declarative): a plugin that only translates GGUF metadata into an ops::ModelDesc and lets
// ops::build_step_plan() emit every op graph. Copy to models/<name>/, rename the namespace and kArchitecture, then
// follow agent/ONBOARD_MODEL.md. Nothing here may hardcode a model dimension: every number comes from GGUF keys.
#pragma once

#include "core/model_interface.hpp"
#include "ops/layer_plan.hpp"

#include <string>
#include <vector>

namespace vegajet::declarative_template {

constexpr const char* kArchitecture = "template";  // TODO: the GGUF general.architecture value

class DeclarativeModel final : public VegaJetModel {
public:
    bool can_load(const GgufFile& f) const override;
    LoadReport load(GgufFile& f, const LoadOptions& opt) override;
    StepResult prefill(const std::vector<int32_t>& tokens) override;
    StepResult decode_step(int32_t token, uint32_t verify_len) override;
    StepResult draft_step(const std::vector<int32_t>& context, uint32_t max_draft) override;
    std::string name() const override { return kArchitecture; }

    const ops::ModelDesc& desc() const { return desc_; }
    // Metadata -> descriptor. Public so a unit test can check it on a synthetic GGUF.
    static bool describe(const GgufFile& f, ops::ModelDesc& out, std::string* err);

private:
    StepResult plan(uint32_t m, bool prefill) const;

    ops::ModelDesc desc_;
    ops::TpDesc tp_;
    uint32_t ctx_tokens_ = 0;   // tokens in the KV cache (advanced by prefill / decode_step)
    uint32_t ctx_capacity_ = 0;
    bool loaded_ = false;
};

} // namespace vegajet::declarative_template
