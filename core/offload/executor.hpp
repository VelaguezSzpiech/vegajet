// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include "core/offload/host.hpp"
#include "core/offload/cuda.hpp"
#include "core/offload/engram.hpp"
#include <functional>
#include <map>
#include <memory>
namespace vegajet::offload {
// Synchronous decode-thread facade. Inputs and host matrices remain caller-owned.
// Backends return false without modifying output when unavailable. Errors propagate.
struct MissBackends {
    std::function<bool(uint64_t, Matrix, Activation, float*)> hot, cuda, cpu;
    std::function<bool(uint64_t)> fetch;
};
struct MissStats { uint64_t misses=0, hot=0, cuda=0, cpu=0, failed=0; };
class ExpertExecutor {
public:
    explicit ExpertExecutor(Switches switches, uint32_t hot_slots=8, MissBackends backends={});
    ~ExpertExecutor();
    bool miss(uint64_t key, Matrix matrix, Activation x, float* y);
    void end_step(); // frequency decay and fixed-pool refresh every eight steps
    const MissStats& stats() const { return stats_; }
    // Bind table reads without coupling a MoE implementation to io_uring APIs.
    static std::function<bool(uint64_t)> table_fetch(Engram& table);
private:
    struct Entry { Matrix matrix; double frequency=0; bool hot=false; };
    Switches switches_;
    uint32_t capacity_;
    uint64_t steps_=0;
    MissBackends backends_;
    std::map<uint64_t,Entry> entries_;
    // Fixed NO_VMM pool: at most capacity_ secondary-device matrix sessions.
    std::map<uint64_t,std::unique_ptr<CudaTier>> cuda_;
    MissStats stats_;
};
// Stable lowest-frequency selection; ties use expert id, so startup is reproducible.
std::vector<uint32_t> cold_experts(const std::vector<double>& frequency, double fraction);
} // namespace vegajet::offload

namespace vegajet::offload {
// Fixed-row Engram instantiation: each row contains one complete expert matrix.
// Encoded matrix offsets/strides are model-owned; row reads and tier dispatch are generic.
struct TableMatrix {
    size_t row_bytes=0, weights_offset=0, scales_offset=0;
    uint32_t rows=0, cols=0;
    size_t weight_stride=0, scale_stride=0;
    Format format=Format::Q4_0;
};
class TableExecutor {
    TableMatrix layout_;
    Engram table_;
    ExpertExecutor executor_;
public:
    TableExecutor(const std::string& path, uint64_t experts, TableMatrix layout, Switches switches);
    bool miss(uint64_t expert, Activation x, float* y);
    void end_step() { executor_.end_step(); }
    const char* backend() const { return table_.backend(); }
    bool direct() const { return table_.direct(); }
};
}
