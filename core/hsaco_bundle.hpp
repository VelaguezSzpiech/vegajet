// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU-only gfx906 code-object bundle qualification (no ROCm, no <elf.h>, no GPU access):
//  * text manifest `<sha256> <bytes> <relative path>`: fail-closed parser, path sanitization, standalone
//    FIPS 180-4 SHA-256 verification against a bundle root;
//  * strict AMDGPU ELF64 reader for the NT_AMDGPU_METADATA MessagePack note (`amdhsa.target`,
//    `amdhsa.kernels`), rejecting truncated notes, trailing metadata bytes, duplicate notes/kernels and
//    .name/.symbol disagreement;
//  * gfx906 gates: private_segment_fixed_size == 0 (no scratch/spill), VGPR and LDS budgets, wavefront 64,
//    plus the ELF machine / e_flags target check.
// The parser is deliberately stricter than core/grid_planner.cpp: this is a release gate, so any ambiguity is
// a rejection rather than a skipped field.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace vegajet::bundle {

constexpr size_t kSha256Bytes = 32;
// Manifest entries above this size are rejected before hashing (code objects are well below it).
constexpr uint64_t kMaxBundleFileBytes = 64ull << 20;

using Digest = std::array<uint8_t, kSha256Bytes>;

Digest sha256(const uint8_t* data, size_t size);
std::string to_hex(const Digest& digest);
bool digest_from_hex(std::string_view hex, Digest& out);

struct ManifestEntry {
    std::string path;      // sanitized, '/' separated, relative to the bundle root
    uint64_t size = 0;     // exact file size in bytes
    Digest digest{};       // raw SHA-256
    std::string digest_hex;  // lowercase, mirrors digest
};

struct Manifest {
    std::vector<ManifestEntry> entries;
};

// Parses `<sha256hex> <bytes> <relative path>` lines ('#' comments, blank lines, CRLF tolerated). Exactly
// three whitespace-free fields per entry; duplicate paths, non-hex digests, bad sizes, absolute paths and
// '.'/'..' components are errors. An empty manifest is an error.
bool parse_manifest(std::string_view text, Manifest& out, std::string* err);
bool load_manifest(const std::string& path, Manifest& out, std::string* err);

// Rejects empty, absolute, '\\'-containing, control-byte or non-ASCII paths and empty/'.'/'..' components.
bool sanitize_relative_path(std::string_view in, std::string& out, std::string* why);

struct Gfx906Budget {
    uint32_t max_vgpr = 64;          // 4 resident wave64/SIMD (hardware max is 256)
    uint32_t max_lds_bytes = 65536;  // 64 KiB workgroup segment per CU
    uint32_t max_private_bytes = 0;  // any scratch/spill is a gate failure
    uint32_t wavefront_size = 64;    // gfx906 is wave64
};

struct KernelInfo {
    std::string name;
    uint32_t vgpr_count = 0;
    uint32_t sgpr_count = 0;
    uint32_t group_segment_fixed_size = 0;
    uint32_t private_segment_fixed_size = 0;
    uint32_t kernarg_segment_size = 0;
    uint32_t wavefront_size = 0;
    uint32_t max_flat_workgroup_size = 0;
    bool within_budget = false;
    std::vector<std::string> problems;
};

struct CodeObjectReport {
    bool ok = false;
    std::string error;                 // structural failure: fail closed, kernels may be partial
    std::string target;                // amdhsa.target
    std::vector<KernelInfo> kernels;
    std::vector<std::string> problems;  // budget/metadata conflicts
};

CodeObjectReport inspect_code_object(const uint8_t* data, size_t size, const Gfx906Budget& budget = {});
CodeObjectReport inspect_code_object_file(const std::string& path, const Gfx906Budget& budget = {});

struct FileReport {
    std::string path;
    bool present = false;
    uint64_t size = 0;
    std::string digest_hex;  // digest actually computed from the file
    bool size_ok = false;
    bool digest_ok = false;
    CodeObjectReport code;
    bool ok = false;
    std::vector<std::string> problems;
};

struct BundleReport {
    bool ok = false;
    std::vector<FileReport> files;
    std::vector<std::string> problems;
};

// Verifies every manifest entry below `root` (empty root means paths are relative to the CWD): exact size,
// SHA-256, then the gfx906 code-object gates. Returns out.ok; `err` is set only for a top-level failure.
bool verify_bundle(const Manifest& manifest, const std::string& root, const Gfx906Budget& budget,
                   BundleReport& out, std::string* err);
bool verify_bundle(const Manifest& manifest, const std::string& root, BundleReport& out, std::string* err);

} // namespace vegajet::bundle
