// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once

#include <string>
#include <vector>

namespace vegajet::cli {
int run_inspect(const std::vector<std::string>& args);
// `bench` and `run` share the planning + code-object validation path.
int run_soak(const std::vector<std::string>& args);
int run_bench(const std::string& cmd, const std::vector<std::string>& args);
} // namespace vegajet::cli
