// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include "core/kfd_runtime.hpp"

namespace vegajet::kfd {
struct CodeSegment { uint64_t file_offset, image_offset, file_bytes; };
struct CodeLoadPlan {
    uint64_t image_bytes = 0;
    std::vector<CodeSegment> segments;
    // kd_gpu_va is an image-relative offset until the runtime allocates its BO.
    std::unordered_map<std::string, LoadedKernel> kernels;
};
// Validate all live-loader input before reserving VA, allocating a BO or copying bytes.
bool plan_hsaco_load(const uint8_t* data, size_t bytes, CodeLoadPlan& out, std::string* err);
constexpr uint64_t kMaxCodeBytes = 64ULL << 20;
}
