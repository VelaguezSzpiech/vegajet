// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// gfx906 machine-code scanner for code objects (.hsaco). It complements core/hsaco_gate (HEAD), which already
// enforces the metadata gates (0 B scratch, VGPR class limits, < 24576 B per kernel) from amdhsa.kernels and
// ELF symbol sizes, but cannot see instructions.
//
//  * decode_gfx9(): instruction-boundary decoder for GCN5 / gfx9 encodings (32/64-bit words + 32-bit literals,
//    SDWA/DPP extension dwords). It does not disassemble; it only needs to know where each instruction starts.
//  * scan_code_object(): walks every kernel function symbol in .text and flags MUBUF loads wider than one dword
//    with LDS=1. On gfx906, LDS-direct loads exist only for buffer_load_dword; buffer_load_dwordx4 ... lds is
//    gfx950-only and is banned (CLAUDE.md ISA gate).
//  * scan_disassembly_text(): the same rule over `llvm-objdump -d` output, for builds that keep listings.
// Pure host code: reads bytes, never loads or runs a code object.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::isa {

enum class Enc : uint8_t {
    SOP2, SOPK, SOP1, SOPC, SOPP, VOP2, VOP1, VOPC, VINTRP, VOP3, VOP3P, SMEM, DS, FLAT, MUBUF, MTBUF, MIMG, EXP,
    Unknown,
};
const char* enc_name(Enc e);

struct Inst {
    uint32_t offset = 0;  // byte offset inside the decoded range
    uint8_t size = 0;     // 4, 8 or 12 bytes (incl. literal / SDWA / DPP dword)
    Enc enc = Enc::Unknown;
    uint32_t w0 = 0, w1 = 0;
};

// Decodes [code, code+bytes). Stops with false (and err) at an unknown encoding or a truncated instruction.
bool decode_gfx9(const uint8_t* code, size_t bytes, std::vector<Inst>& out, std::string* err = nullptr);

// gfx9 MUBUF opcodes used by the scanner.
constexpr uint32_t kBufferLoadDword = 20, kBufferLoadDwordx2 = 21, kBufferLoadDwordx3 = 22, kBufferLoadDwordx4 = 23;

struct MubufFields {
    uint32_t op = 0;
    bool offen = false, idxen = false, glc = false, lds = false, slc = false;
};
MubufFields mubuf_fields(const Inst& i);  // only meaningful for Enc::MUBUF

struct IsaFinding {
    std::string kernel;
    uint32_t offset = 0;  // byte offset from the kernel entry
    std::string what;
};

struct IsaScanReport {
    bool ok = false;               // decoded every kernel and found nothing banned
    uint32_t kernels = 0;
    uint64_t instructions = 0;
    uint64_t text_bytes = 0;       // size of the .text section
    uint64_t mubuf_lds_dword = 0;  // legal LDS-direct dword loads (informational)
    std::vector<IsaFinding> findings;
    std::vector<std::string> errors;
};

IsaScanReport scan_code_object(const uint8_t* data, size_t size);
IsaScanReport scan_code_object_file(const std::string& path);

// `llvm-objdump -d --no-show-raw-insn` style text (raw bytes are tolerated). Kernel names come from the
// "<name>:" labels.
std::vector<IsaFinding> scan_disassembly_text(const std::string& text);

} // namespace vegajet::isa
