// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// gfx906 code-object gates shared by the build (tools/hsaco_gate_main.cpp) and the KFD loader:
//   private_segment_fixed_size == 0, vgpr_count <= 48 (stream) / 64 (tiled), kernel code < 24576 B.
// The per-kernel table is generated at configure time from ops/kernels/hsaco_gates.txt.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace vegajet::gate {

enum class KernelClass { Stream, Tiled };

constexpr uint32_t kMaxVgprStream = 48;      // 5 waves/SIMD
constexpr uint32_t kMaxVgprTiled  = 64;      // 4 waves/SIMD
constexpr uint32_t kMaxCodeBytes  = 24576;   // strictly below: stays inside the 32 KB SQC L1I

struct GateEntry {
    const char* name;
    KernelClass klass;
    uint32_t    waiver_vgpr;   // 0 = no waiver
};

enum class GateResult { Ok, OkWaived, Fail };

inline uint32_t class_limit(KernelClass k) { return k == KernelClass::Stream ? kMaxVgprStream : kMaxVgprTiled; }

// Pure decision. `entry` == nullptr means the kernel is not in the manifest. `why` gets a one-line reason
// for Fail and for OkWaived.
GateResult evaluate(const GateEntry* entry, const std::string& kernel, uint32_t vgpr_count,
                    uint32_t private_segment_fixed_size, uint64_t code_bytes, std::string* why);

// Compiled-in manifest.
const std::vector<GateEntry>& manifest();
const GateEntry* find_entry(const std::string& kernel);

// ELF64 helper: kernel entry function name -> st_size, from .symtab/.dynsym.
std::unordered_map<std::string, uint64_t> elf_function_sizes(const uint8_t* data, size_t size);

struct FileGateReport {
    bool ok = false;
    uint32_t kernels = 0, waived = 0, max_vgpr = 0;
    uint64_t max_code_bytes = 0;
    std::vector<std::string> failures;
    std::vector<std::string> waivers;
};
// Runs every gate on one unbundled .hsaco file.
FileGateReport check_code_object_file(const std::string& path);

} // namespace vegajet::gate
