// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// v1.3 dot8-wire: maps the planner's TensorPlan::format == VJ_Q4_DOT8 decisions onto the GPU upload
// routes that have a kernel. Active only with VEGAJET_V13_DOT8_PROMOTE=1; then the plan is the single
// source of truth for the routed Gate+Up format and every promoted tensor without a DOT8 upload +
// kernel route fails the load (fail closed). Pure host code, CPU-testable.
#pragma once

#include "models/ornith/ornith_model.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::ornith {

// DOT8 upload + kernel routes the caller can serve.
struct Dot8KernelRoutes {
    bool gate_up_dot8 = false;  // merged IQ2_S Gate+Up -> VJ_Q4_DOT8 repack + DOT8 Gate+Up kernel
    bool down_dot8 = false;     // routed IQ3_XXS Down -> exact DOT8 repack + DOT8 Down kernel
    bool gate_up_geometry = false;  // the caller's Gate+Up DOT8 layout supports this model's shape
};

struct Dot8UploadPlan {
    std::vector<uint8_t> gate_up_dot8;  // per trunk layer: 1 = upload routed Gate+Up as VJ_Q4_DOT8
    uint32_t gate_up_layers = 0;        // count of 1s in gate_up_dot8
    uint32_t down_layers = 0;           // promoted routed Down tensors (served by the DOT8 Down route)
};

// VEGAJET_V13_DOT8_PROMOTE=1 (default off).
bool dot8_promote_enabled();

// Resolves every non-float, non-draft TensorPlan entry with format VJ_Q4_DOT8:
//  - blk.N.ffn_down_exps (IQ3_XXS, routed)      -> DOT8 Down route (needs routes.down_dot8)
//  - blk.N.ffn_gate_exps + ffn_up_exps (IQ2_S)  -> DOT8 Gate+Up route for layer N; both halves of the
//    merged pair must be promoted (needs routes.gate_up_dot8 and routes.gate_up_geometry)
//  - layer >= n_trunk (nextn/MTP): an error when mtp_uploaded (the MTP head has no DOT8 route), else
//    ignored because the tensor is not uploaded
//  - anything else: an error naming the tensor.
// The routed Down route is an exact (lossless) re-layout of the IQ3_XXS blocks, so a Down tensor the
// plan keeps as VJ_IQ_PACKED is served by the same upload. Returns false with *err set on refusal.
bool resolve_dot8_upload_plan(const std::vector<TensorPlan>& plan, uint32_t n_trunk, bool mtp_uploaded,
                              const Dot8KernelRoutes& routes, Dot8UploadPlan& out, std::string* err);

} // namespace vegajet::ornith
