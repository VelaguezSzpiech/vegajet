#pragma once

#include "core/universal_ir.hpp"
#include "ops/vj_vm_desc.hpp"
#include <cstddef>
#include <string>
#include <vector>

namespace universal {

// Fills the real device descriptor for one graph node (including buffer VAs).
using DescriptorBinder = bool (*)(const Op&, size_t index, vegajet::vm::VjVmOpDesc&, void* context);

// Preserves graph order; rejects graph nodes without a device opcode mapping and
// rejects descriptors whose opcode cannot implement the corresponding node.
bool emit_bytecode(const Model& model, DescriptorBinder bind, void* context,
                   std::vector<vegajet::vm::VjVmOpDesc>& out, std::string* error = nullptr);

enum class DeviceKernel { Attention, Ffn, Aux };
struct DeviceSegment {
    DeviceKernel kernel;
    size_t begin;
    size_t count;
};

// Ranges index the unchanged descriptor stream. Submit in order on one ordered
// queue (or wait for completion between launches). GridBarrier is host-only and
// is omitted; BarrierAfter always ends a launch. Each computational descriptor
// has its own launch: no unproven cross-workgroup dependency is fused.
// Shared opcodes use the next specialized owner in their boundary-delimited
// region, otherwise the preceding owner, otherwise Attention (both support them).
// Unknown opcodes fail without changing output. Empty/barrier-only streams pass.
bool plan_device_segments(const std::vector<vegajet::vm::VjVmOpDesc>& program,
                          std::vector<DeviceSegment>& segments, std::string* error = nullptr);
bool emit_device_plan(const Model& model, DescriptorBinder bind, void* context,
                      std::vector<vegajet::vm::VjVmOpDesc>& program,
                      std::vector<DeviceSegment>& segments, std::string* error = nullptr);

} // namespace universal
