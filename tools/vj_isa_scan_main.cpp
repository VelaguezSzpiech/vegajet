// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// vj_isa_scan <file.hsaco>...        : fails (exit 1) if any kernel contains buffer_load_dwordx{2,3,4} ... lds or cannot
//                                      be decoded. Companion of vj_hsaco_gate (metadata gates).
// vj_isa_scan --text <listing>...    : the same rule over `llvm-objdump -d` text listings.
// Reads files only.
#include "core/hsaco_isa_scan.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

int main(int argc, char** argv) {
    const bool text = argc > 1 && std::strcmp(argv[1], "--text") == 0;
    const int first = text ? 2 : 1;
    if (argc <= first) {
        std::fprintf(stderr, "usage: %s <file.hsaco>...\n       %s --text <objdump listing>...\n", argv[0], argv[0]);
        return 2;
    }
    int bad = 0;
    for (int i = first; i < argc; ++i) {
        if (text) {
            std::ifstream f(argv[i]);
            if (!f) { std::printf("%s: FAIL cannot read\n", argv[i]); ++bad; continue; }
            std::stringstream ss;
            ss << f.rdbuf();
            const auto fs = vegajet::isa::scan_disassembly_text(ss.str());
            std::printf("%s: %s findings=%zu\n", argv[i], fs.empty() ? "OK" : "FAIL", fs.size());
            for (const auto& x : fs) std::printf("  %s +0x%x: %s\n", x.kernel.c_str(), x.offset, x.what.c_str());
            bad += !fs.empty();
            continue;
        }
        const auto r = vegajet::isa::scan_code_object_file(argv[i]);
        std::printf("%s: %s kernels=%u instructions=%llu text=%llu B lds_dword=%llu\n", argv[i], r.ok ? "OK" : "FAIL", r.kernels,
                    static_cast<unsigned long long>(r.instructions), static_cast<unsigned long long>(r.text_bytes),
                    static_cast<unsigned long long>(r.mubuf_lds_dword));
        for (const auto& f : r.findings) std::printf("  %s +0x%x: %s\n", f.kernel.c_str(), f.offset, f.what.c_str());
        for (const auto& e : r.errors) std::printf("  error: %s\n", e.c_str());
        bad += !r.ok;
    }
    return bad ? 1 : 0;
}
