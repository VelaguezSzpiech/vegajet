// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include "core/tp_collective.hpp"
#include <string>
namespace vegajet::cli {
int run_tp_ar_bench(uint32_t ranks, const std::string& hsaco_dir, const tp::CollectiveOptions& options);
}
