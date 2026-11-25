#pragma once
#include "core/universal_ir.hpp"
#include <cstddef>
#include <string>
#include <vector>
namespace universal {
struct FusionStats { size_t reductions=0, qkv_cache=0, moe=0, elided_allreduce=0; };
struct PassOptions { uint32_t tp_size=1, n_cu=60, m_hit=1; bool prefill=false; };
class PassManager { public: FusionStats run(Model& model,const PassOptions& options) const; };
} // namespace universal
