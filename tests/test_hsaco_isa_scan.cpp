// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Code-object gates end to end on synthetic gfx906 ELF files (no GPU, no toolchain needed):
//  * HEAD core/hsaco_gate + core/grid_planner: amdhsa.kernels MessagePack note -> 0 B scratch, VGPR class limit,
//    < 24576 B per kernel;
//  * core/hsaco_isa_scan: gfx9 instruction boundaries (literals, SDWA, VOP2 madmk, 64-bit encodings) and the
//    ban on buffer_load_dwordx{2,3,4} ... lds (buffer_load_dword ... lds stays legal);
//  * the same rule over llvm-objdump text.
// Optional: VJ_OBJDUMP=<llvm-objdump> cross-checks the synthetic boundaries; VJ_HSACO_DIR=<dir> scans built files.
#include "core/grid_planner.hpp"
#include "core/hsaco_gate.hpp"
#include "core/hsaco_isa_scan.hpp"
#include "tests/test_common.hpp"

#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <set>
#include <string>

using namespace vegajet;

namespace {

// ---- tiny MessagePack writer ----
struct Mp {
    std::vector<uint8_t> b;
    void str(const std::string& s) {
        if (s.size() < 32) b.push_back(static_cast<uint8_t>(0xA0 | s.size()));
        else { b.push_back(0xD9); b.push_back(static_cast<uint8_t>(s.size())); }
        b.insert(b.end(), s.begin(), s.end());
    }
    void u(uint32_t v) {
        if (v < 128) { b.push_back(static_cast<uint8_t>(v)); return; }
        b.push_back(0xCE);
        for (int i = 3; i >= 0; --i) b.push_back(static_cast<uint8_t>(v >> (8 * i)));
    }
    void map(uint8_t n) { b.push_back(static_cast<uint8_t>(0x80 | n)); }
    void arr(uint8_t n) { b.push_back(static_cast<uint8_t>(0x90 | n)); }
};

struct KernelSpec {
    std::string name;
    uint32_t vgpr, priv;
    std::vector<uint32_t> words;  // machine code dwords
};

// ELF64: null, .note, .text, .symtab, .strtab, .shstrtab
std::vector<uint8_t> make_elf(const std::vector<KernelSpec>& ks) {
    Mp m;
    m.map(2);
    m.str("amdhsa.target");
    m.str("amdgcn-amd-amdhsa--gfx906");
    m.str("amdhsa.kernels");
    m.arr(static_cast<uint8_t>(ks.size()));
    for (const auto& k : ks) {
        m.map(8);
        m.str(".name"); m.str(k.name);
        m.str(".symbol"); m.str(k.name + ".kd");
        m.str(".vgpr_count"); m.u(k.vgpr);
        m.str(".sgpr_count"); m.u(24);
        m.str(".group_segment_fixed_size"); m.u(0);
        m.str(".private_segment_fixed_size"); m.u(k.priv);
        m.str(".wavefront_size"); m.u(64);
        m.str(".max_flat_workgroup_size"); m.u(256);
    }
    std::vector<uint8_t> note;
    auto put32 = [](std::vector<uint8_t>& v, uint32_t x) { for (int i = 0; i < 4; ++i) v.push_back(static_cast<uint8_t>(x >> (8 * i))); };
    auto put64 = [](std::vector<uint8_t>& v, uint64_t x) { for (int i = 0; i < 8; ++i) v.push_back(static_cast<uint8_t>(x >> (8 * i))); };
    put32(note, 7); put32(note, static_cast<uint32_t>(m.b.size())); put32(note, 32);
    note.insert(note.end(), {'A', 'M', 'D', 'G', 'P', 'U', 0, 0});
    note.insert(note.end(), m.b.begin(), m.b.end());
    while (note.size() % 4) note.push_back(0);

    const uint64_t text_addr = 0x1000;
    std::vector<uint8_t> text, symtab(24, 0), strtab(1, 0);
    for (const auto& k : ks) {
        const uint64_t off = text.size();
        for (uint32_t w : k.words) put32(text, w);
        const uint32_t name_off = static_cast<uint32_t>(strtab.size());
        strtab.insert(strtab.end(), k.name.begin(), k.name.end());
        strtab.push_back(0);
        put32(symtab, name_off);
        symtab.push_back(0x12);  // GLOBAL FUNC
        symtab.push_back(0);
        symtab.push_back(2); symtab.push_back(0);  // shndx = 2 (.text)
        put64(symtab, text_addr + off);
        put64(symtab, k.words.size() * 4);
        while (text.size() % 256) put32(text, 0xBF800000u);  // s_nop padding between kernels
    }
    const std::string shstr = std::string("\0.note\0.text\0.symtab\0.strtab\0.shstrtab\0", 39);
    std::vector<uint8_t> out(64, 0);
    auto align = [&](size_t a) { while (out.size() % a) out.push_back(0); };
    align(16); const uint64_t note_off = out.size(); out.insert(out.end(), note.begin(), note.end());
    out.resize(text_addr, 0); const uint64_t text_off = out.size(); out.insert(out.end(), text.begin(), text.end());
    align(8); const uint64_t sym_off = out.size(); out.insert(out.end(), symtab.begin(), symtab.end());
    const uint64_t str_off = out.size(); out.insert(out.end(), strtab.begin(), strtab.end());
    const uint64_t shs_off = out.size(); out.insert(out.end(), shstr.begin(), shstr.end());
    align(8);
    const uint64_t shoff = out.size();
    auto sh = [&](uint32_t name, uint32_t type, uint64_t flags, uint64_t addr, uint64_t off, uint64_t size, uint32_t link,
                  uint32_t info, uint64_t al, uint64_t ent) {
        put32(out, name); put32(out, type); put64(out, flags); put64(out, addr); put64(out, off); put64(out, size);
        put32(out, link); put32(out, info); put64(out, al); put64(out, ent);
    };
    sh(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    sh(1, 7, 2, 0, note_off, note.size(), 0, 0, 4, 0);
    sh(7, 1, 6, text_addr, text_off, text.size(), 0, 0, 256, 0);
    sh(13, 2, 0, 0, sym_off, symtab.size(), 4, 1, 8, 24);
    sh(21, 3, 0, 0, str_off, strtab.size(), 0, 0, 1, 0);
    sh(29, 3, 0, 0, shs_off, shstr.size(), 0, 0, 1, 0);
    // ELF header
    const uint8_t ident[16] = {0x7f, 'E', 'L', 'F', 2, 1, 1, 64, 2};
    std::memcpy(out.data(), ident, 16);
    auto w16 = [&](size_t at, uint16_t v) { std::memcpy(out.data() + at, &v, 2); };
    auto w32 = [&](size_t at, uint32_t v) { std::memcpy(out.data() + at, &v, 4); };
    auto w64 = [&](size_t at, uint64_t v) { std::memcpy(out.data() + at, &v, 8); };
    w16(16, 3); w16(18, 224); w32(20, 1); w64(40, shoff); w32(48, 0x2F);  // ET_DYN, EM_AMDGPU, gfx906
    w16(52, 64); w16(58, 64); w16(60, 6); w16(62, 5);
    return out;
}

// Instruction words (gfx9 encodings) and their expected sizes.
struct W { std::vector<uint32_t> words; uint32_t size; isa::Enc enc; };
const std::vector<W> kProgram = {
    {{0xBE8000FFu, 0x12345678u}, 8, isa::Enc::SOP1},   // s_mov_b32 s0, 0x12345678
    {{0x8000FF01u, 0x00000040u}, 8, isa::Enc::SOP2},   // s_add_u32 s0, s1, 0x40
    {{0x80000201u}, 4, isa::Enc::SOP2},                // s_add_u32 s0, s1, s2
    {{0xC0060002u, 0x00000000u}, 8, isa::Enc::SMEM},   // s_load_dwordx2
    {{0x7E0002FFu, 0x3F800000u}, 8, isa::Enc::VOP1},   // v_mov_b32 v0, 1.0
    {{0x02000501u}, 4, isa::Enc::VOP2},                // v_add_f32 v0, v1, v2
    {{0x2E000501u, 0x40000000u}, 8, isa::Enc::VOP2},   // v_madmk_f32 v0, v1, 2.0, v2 (literal always)
    {{0xD1CB0000u, 0x04020300u}, 8, isa::Enc::VOP3},   // v_fma_f32
    {{0xD9FE0000u, 0x00000000u}, 8, isa::Enc::DS},     // ds_read_b128
    {{0xDC5C8000u, 0x007F0000u}, 8, isa::Enc::FLAT},   // global_load_dwordx4
    {{0xE05C0000u, 0x80000000u}, 8, isa::Enc::MUBUF},  // buffer_load_dwordx4 (VGPR target: legal)
    {{0xE0510000u, 0x80000000u}, 8, isa::Enc::MUBUF},  // buffer_load_dword ... lds (legal on gfx906)
    {{0xBF8C0F70u}, 4, isa::Enc::SOPP},                // s_waitcnt
};
const std::vector<uint32_t> kBannedX4 = {0xE05D0000u, 0x80000000u};  // buffer_load_dwordx4 ... lds
const std::vector<uint32_t> kBannedX2 = {0xE0550000u, 0x80000000u};  // buffer_load_dwordx2 ... lds
const uint32_t kEndpgm = 0xBF810000u;

std::vector<uint32_t> program(const std::vector<uint32_t>& extra) {
    std::vector<uint32_t> w;
    for (const auto& p : kProgram) w.insert(w.end(), p.words.begin(), p.words.end());
    w.insert(w.end(), extra.begin(), extra.end());
    w.push_back(kEndpgm);
    return w;
}

std::string write_file(const std::vector<uint8_t>& b, const std::string& tag) { return vjtest::write_temp(b, tag); }

} // namespace

int main() {
    // ---- decoder on the reference program
    {
        const auto words = program({});
        std::vector<isa::Inst> in;
        std::string err;
        VJ_CHECK(isa::decode_gfx9(reinterpret_cast<const uint8_t*>(words.data()), words.size() * 4, in, &err));
        VJ_CHECK(in.size() == kProgram.size() + 1);
        uint32_t off = 0;
        for (size_t i = 0; i < kProgram.size(); ++i) {
            VJ_CHECK(in[i].offset == off && in[i].size == kProgram[i].size && in[i].enc == kProgram[i].enc);
            off += kProgram[i].size;
        }
        VJ_CHECK(in.back().enc == isa::Enc::SOPP);
        const auto f = isa::mubuf_fields(in[11]);
        VJ_CHECK(f.op == isa::kBufferLoadDword && f.lds && !f.glc);
        const uint32_t sdwa[] = {0x020000F9u, 0x06060600u};  // VOP2 with SDWA dword
        VJ_CHECK(isa::decode_gfx9(reinterpret_cast<const uint8_t*>(sdwa), 8, in, &err) && in.back().size == 8);
        const uint32_t trunc[] = {0xBE8000FFu};               // literal missing
        in.clear();
        VJ_CHECK(!isa::decode_gfx9(reinterpret_cast<const uint8_t*>(trunc), 4, in, &err) && err.find("truncated") != std::string::npos);
        const uint32_t bad[] = {0xF8000000u};                 // no gfx9 encoding
        VJ_CHECK(!isa::decode_gfx9(reinterpret_cast<const uint8_t*>(bad), 4, in, &err));
    }

    // ---- clean code object: both gates pass
    const auto clean = make_elf({{"ornith_embed_q8_kernel", 20, 0, program({})},
                                 {"vegajet_megakernel_vm", 53, 0, program({})}});
    const std::string clean_path = write_file(clean, "isa_clean");
    {
        const auto co = planner::parse_code_object(clean.data(), clean.size());
        VJ_CHECK(co.ok && co.kernels.size() == 2 && co.target == "amdgcn-amd-amdhsa--gfx906");
        const auto g = gate::check_code_object_file(clean_path);
        if (!g.ok) for (const auto& f : g.failures) std::fprintf(stderr, "  gate: %s\n", f.c_str());
        VJ_CHECK(g.ok && g.kernels == 2 && g.max_vgpr == 53);
        const auto s = isa::scan_code_object(clean.data(), clean.size());
        for (const auto& e : s.errors) std::fprintf(stderr, "  isa: %s\n", e.c_str());
        VJ_CHECK(s.ok && s.kernels == 2 && s.instructions == 2 * (kProgram.size() + 1) && s.mubuf_lds_dword == 2);
    }

    // ---- violations: spill + VGPR over class limit (metadata gate) and banned LDS loads (ISA gate)
    const auto dirty = make_elf({{"ornith_embed_q8_kernel", 70, 16, program(kBannedX4)},
                                 {"vegajet_megakernel_vm", 40, 0, program(kBannedX2)}});
    const std::string dirty_path = write_file(dirty, "isa_dirty");
    {
        const auto g = gate::check_code_object_file(dirty_path);
        VJ_CHECK(!g.ok && g.failures.size() == 1 && g.failures[0].find("private_segment_fixed_size=16") != std::string::npos);
        const auto s = isa::scan_code_object(dirty.data(), dirty.size());
        VJ_CHECK(!s.ok && s.errors.empty() && s.findings.size() == 2);
        std::set<std::string> where;
        for (const auto& f : s.findings) where.insert(f.kernel + ":" + f.what.substr(0, f.what.find(' ')));
        VJ_CHECK(where.count("ornith_embed_q8_kernel:buffer_load_dwordx4") && where.count("vegajet_megakernel_vm:buffer_load_dwordx2"));
        uint32_t expect_off = 0;
        for (const auto& p : kProgram) expect_off += p.size;
        for (const auto& f : s.findings) VJ_CHECK(f.offset == expect_off);
        // Unlisted kernel name -> metadata gate fails (manifest is the classification source).
        const auto unk = make_elf({{"brand_new_kernel", 20, 0, program({})}});
        const std::string unk_path = write_file(unk, "isa_unk");
        const auto gu = gate::check_code_object_file(unk_path);
        std::remove(unk_path.c_str());
        VJ_CHECK(!gu.ok && gu.failures[0].find("not listed") != std::string::npos);
        // Garbage input never crashes the scanner.
        std::vector<uint8_t> junk(dirty.begin(), dirty.begin() + 200);
        VJ_CHECK(!isa::scan_code_object(junk.data(), junk.size()).ok);
        VJ_CHECK(!isa::scan_code_object(nullptr, 0).ok);
    }

    // ---- disassembly text scanner
    {
        const std::string dis =
            "0000000000001000 <kern_a>:\n"
            "\tbuffer_load_dword v1, v0, s[4:7], 0 offen lds                 // 000000001000: E0511000 80010100\n"
            "\tbuffer_load_dwordx4 v[0:3], v4, s[4:7], 0 offen               // 000000001008: E05C1000 80010004\n"
            "0000000000001100 <kern_b>:\n"
            "\ts_nop 0\n"
            "\tbuffer_load_dwordx4 v[0:3], off, s[4:7], 0 lds\n"
            "\t; buffer_load_dwordx2 in a comment with lds is ignored // lds\n";
        const auto f = isa::scan_disassembly_text(dis);
        VJ_CHECK(f.size() == 1 && f[0].kernel == "kern_b" && f[0].offset == 1);
    }

    // ---- optional: llvm-objdump agrees with the decoder on the synthetic program
    if (const char* od = std::getenv("VJ_OBJDUMP")) {
        const std::string cmd = std::string(od) + " -d --mcpu=gfx906 " + clean_path + " 2>/dev/null";
        FILE* p = popen(cmd.c_str(), "r");
        std::set<uint32_t> ref;
        char line[512];
        while (p && std::fgets(line, sizeof(line), p)) {
            const char* c = std::strstr(line, "// ");
            unsigned addr = 0;
            if (c && std::sscanf(c, "// %x:", &addr) == 1) ref.insert(addr);
        }
        if (p) pclose(p);
        std::set<uint32_t> mine;
        uint32_t a = 0x1000;
        for (const auto& w : kProgram) { mine.insert(a); a += w.size; }
        mine.insert(a);
        uint32_t hit = 0;
        for (uint32_t x : mine) hit += ref.count(x);
        VJ_CHECK(hit == mine.size());
        std::printf("  llvm-objdump cross-check: %u/%zu synthetic instruction boundaries agree\n", hit, mine.size());
    }
    // ---- optional: scan every built code object in a directory
    if (const char* dir = std::getenv("VJ_HSACO_DIR")) {
        if (DIR* d = opendir(dir)) {
            uint32_t files = 0, findings = 0, errors = 0;
            uint64_t insts = 0;
            while (auto* e = readdir(d)) {
                const std::string n = e->d_name;
                if (n.size() < 6 || n.substr(n.size() - 6) != ".hsaco") continue;
                const auto r = isa::scan_code_object_file(std::string(dir) + "/" + n);
                ++files; insts += r.instructions; findings += static_cast<uint32_t>(r.findings.size());
                errors += static_cast<uint32_t>(r.errors.size());
                for (const auto& f : r.findings) std::printf("  FINDING %s %s +%u %s\n", n.c_str(), f.kernel.c_str(), f.offset, f.what.c_str());
            }
            closedir(d);
            std::printf("  VJ_HSACO_DIR: %u files, %llu instructions, %u findings, %u decode errors\n", files,
                        static_cast<unsigned long long>(insts), findings, errors);
            VJ_CHECK(findings == 0 && errors == 0);
        }
    }
    std::remove(clean_path.c_str());
    std::remove(dirty_path.c_str());
    std::printf("test_hsaco_isa_scan: metadata gates (HEAD) + gfx9 ISA scan + text scan OK\nPASS\n");
    return 0;
}
