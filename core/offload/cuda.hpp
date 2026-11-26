// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include "core/offload/host.hpp"
#include <string>
namespace vegajet::offload {
// Optional CUDA device index is explicit; no device enumeration or KFD calls.
// One matrix per session, copied to the secondary device. Optional device-clock budget.
class CudaTier {
    struct Impl; std::unique_ptr<Impl> p_;
public:
    CudaTier(Matrix matrix,int device=0,uint32_t budget_ms=0); // 0: persist until teardown
    ~CudaTier();
    bool ok() const;
    const std::string& error() const;
    bool submit(Activation x); // SPSC; copies activation into ring
    bool poll(float* y);       // returns oldest completion, copies result out
};
}
