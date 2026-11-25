// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "models/registry.hpp"

#include <algorithm>

namespace vegajet {

// Each built-in plugin provides a registration function (defined in its own directory).
void register_builtin_models(ModelRegistry& r);

ModelRegistry& ModelRegistry::instance() {
    static ModelRegistry reg = [] {
        ModelRegistry r;
        register_builtin_models(r);
        return r;
    }();
    return reg;
}

void ModelRegistry::add(ModelRegistration reg) { regs_.push_back(std::move(reg)); }

std::unique_ptr<VegaJetModel> ModelRegistry::create_for(const GgufFile& f, std::string* err) const {
    const std::string arch = f.architecture();
    for (const auto& r : regs_) {
        if (std::find(r.architectures.begin(), r.architectures.end(), arch) == r.architectures.end()) continue;
        auto m = r.factory();
        if (m && m->can_load(f)) return m;
    }
    if (err) *err = "no registered plugin accepts general.architecture=\"" + arch + "\"";
    return nullptr;
}

} // namespace vegajet
