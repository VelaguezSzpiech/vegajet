// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include <stdint.h>
namespace vegajet::offload {
constexpr unsigned cuda_slots=8, cuda_cols=8192, cuda_rows=1024;
struct alignas(64) CudaCommand {
    unsigned state; // 0 free, 1 published doorbell, 2 completed; system acquire/release
    int8_t x[cuda_cols]; float scales[cuda_cols/32]; float y[cuda_rows];
};
struct alignas(64) CudaRing {
    unsigned stop, alive;
    CudaCommand commands[cuda_slots];
};
static_assert(sizeof(CudaRing)<2*1024*1024,"ring must fit one isolated 2 MiB VA region");
}
