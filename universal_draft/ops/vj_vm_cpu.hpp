#pragma once

#include "ops/vj_vm_desc.hpp"
#include <cstddef>
#include <string>
#include <vector>

namespace vegajet::vm {

// Executes implemented descriptors over host buffers named by ptr fields.
// Returns false for malformed or unimplemented ops; unsupported work is never skipped.
bool run_cpu(const VjVmOpDesc* program, size_t count, std::string* error = nullptr);
inline bool run_cpu(const std::vector<VjVmOpDesc>& program, std::string* error = nullptr) {
    return run_cpu(program.data(), program.size(), error);
}

// Independent plain-float reference for tiny dense projections: y = W*x + bias.
void naive_linear(const float* weights, const float* x, const float* bias,
                  float* y, size_t rows, size_t cols);

} // namespace vegajet::vm
