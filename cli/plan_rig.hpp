// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// `vegajet plan <model-file> --gpus N --vram 16|32 [--ctx T] [--kv-bits 16|8] [--cu 60|64] [--matrix]`:
// header-only rig estimator (core/rig_planner). Never opens a device node; works on every build.
#pragma once

#include <string>
#include <vector>

namespace vegajet::cli {
// True when `plan` should run the rig estimator instead of `bench --dry-run` (--gpus, --vram or --matrix given).
bool wants_rig_plan(const std::vector<std::string>& args);
// Exit codes: 0 fits (possibly with host-RAM MoE offload; with --matrix: the model was readable), 1 model unreadable
// or does not fit, 2 usage error.
int run_rig_plan(const std::vector<std::string>& args);
} // namespace vegajet::cli
