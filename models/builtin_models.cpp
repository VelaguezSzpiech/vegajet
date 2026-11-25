// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Built-in plugin table. Adding a model = one ModelRegistration here (see agent/ONBOARD_MODEL.md).
#include "models/qwen4exp/qwen4exp_model.hpp"
#include "models/registry.hpp"

namespace vegajet {

void register_ornith(ModelRegistry& r);

void register_builtin_models(ModelRegistry& r) {
    register_ornith(r);
    r.add({"qwen4exp", {"qwen4exp"}, [] { return std::make_unique<qwen4exp::Qwen4ExpModel>(); }});
}

} // namespace vegajet
