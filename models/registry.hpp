// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Maps a GGUF file to the plugin that can load it (general.architecture + tensor layout via can_load).
#pragma once

#include "core/model_interface.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vegajet {

using ModelFactory = std::function<std::unique_ptr<VegaJetModel>()>;

struct ModelRegistration {
    std::string plugin;                       // plugin name
    std::vector<std::string> architectures;   // general.architecture values it claims
    ModelFactory factory;
};

class ModelRegistry {
public:
    static ModelRegistry& instance();  // pre-populated with the built-in plugins
    void add(ModelRegistration reg);
    const std::vector<ModelRegistration>& entries() const { return regs_; }
    // First plugin whose architecture list matches and whose can_load() accepts the file.
    std::unique_ptr<VegaJetModel> create_for(const GgufFile& f, std::string* err) const;

private:
    std::vector<ModelRegistration> regs_;
};

} // namespace vegajet
