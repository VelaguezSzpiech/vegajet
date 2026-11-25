// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/hsaco_isa_scan.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>

namespace vegajet::isa {

const char* enc_name(Enc e) {
    static const char* const k[] = {"SOP2", "SOPK", "SOP1", "SOPC", "SOPP", "VOP2", "VOP1", "VOPC", "VINTRP", "VOP3",
                                    "VOP3P", "SMEM", "DS", "FLAT", "MUBUF", "MTBUF", "MIMG", "EXP", "Unknown"};
    return k[static_cast<size_t>(e)];
}

namespace {

constexpr uint32_t kLiteral = 255, kSdwa = 0xF9, kDpp = 0xFA;
// gfx9 VOP2 opcodes that always carry a 32-bit literal (v_madmk/v_madak f32/f16).
bool vop2_has_literal(uint32_t op) { return op == 0x17 || op == 0x18 || op == 0x24 || op == 0x25; }

uint32_t rd32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

} // namespace

bool decode_gfx9(const uint8_t* code, size_t bytes, std::vector<Inst>& out, std::string* err) {
    size_t pc = 0;
    while (pc + 4 <= bytes) {
        Inst in;
        in.offset = static_cast<uint32_t>(pc);
        in.w0 = rd32(code + pc);
        const uint32_t w = in.w0;
        uint32_t size = 4;
        if ((w >> 23) == 0x17D) {                       // SOP1
            in.enc = Enc::SOP1;
            if ((w & 0xFF) == kLiteral) size = 8;
        } else if ((w >> 23) == 0x17E) {                // SOPC
            in.enc = Enc::SOPC;
            if ((w & 0xFF) == kLiteral || ((w >> 8) & 0xFF) == kLiteral) size = 8;
        } else if ((w >> 23) == 0x17F) {                // SOPP
            in.enc = Enc::SOPP;
        } else if ((w >> 28) == 0xB) {                  // SOPK
            in.enc = Enc::SOPK;
            if (((w >> 23) & 0x1F) == 20) size = 8;     // s_setreg_imm32_b32
        } else if ((w >> 30) == 0x2) {                  // SOP2
            in.enc = Enc::SOP2;
            if ((w & 0xFF) == kLiteral || ((w >> 8) & 0xFF) == kLiteral) size = 8;
        } else if ((w >> 31) == 0) {                    // VOP1 / VOPC / VOP2
            const uint32_t src0 = w & 0x1FF;
            if ((w >> 25) == 0x3F) in.enc = Enc::VOP1;
            else if ((w >> 25) == 0x3E) in.enc = Enc::VOPC;
            else in.enc = Enc::VOP2;
            if (src0 == kLiteral || src0 == kSdwa || src0 == kDpp) size = 8;
            else if (in.enc == Enc::VOP2 && vop2_has_literal((w >> 25) & 0x3F)) size = 8;
        } else {
            switch (w >> 26) {
                case 0x34: in.enc = ((w >> 23) == 0x1A7) ? Enc::VOP3P : Enc::VOP3; size = 8; break;
                case 0x35: in.enc = Enc::VINTRP; break;
                case 0x30: in.enc = Enc::SMEM; size = 8; break;
                case 0x31: in.enc = Enc::EXP; size = 8; break;
                case 0x36: in.enc = Enc::DS; size = 8; break;
                case 0x37: in.enc = Enc::FLAT; size = 8; break;
                case 0x38: in.enc = Enc::MUBUF; size = 8; break;
                case 0x3A: in.enc = Enc::MTBUF; size = 8; break;
                case 0x3C: in.enc = Enc::MIMG; size = 8; break;
                default:
                    if (err) *err = "unknown gfx9 encoding 0x" + [&] { std::ostringstream o; o << std::hex << w; return o.str(); }() +
                                    " at +" + std::to_string(pc);
                    return false;
            }
        }
        if (pc + size > bytes) {
            if (err) *err = std::string("truncated ") + enc_name(in.enc) + " at +" + std::to_string(pc);
            return false;
        }
        if (size >= 8) in.w1 = rd32(code + pc + 4);
        in.size = static_cast<uint8_t>(size);
        out.push_back(in);
        pc += size;
    }
    if (pc != bytes) {
        if (err) *err = "trailing " + std::to_string(bytes - pc) + " byte(s) (not dword aligned)";
        return false;
    }
    return true;
}

MubufFields mubuf_fields(const Inst& i) {
    MubufFields f;
    f.op = (i.w0 >> 18) & 0x7F;
    f.offen = (i.w0 >> 12) & 1u;
    f.idxen = (i.w0 >> 13) & 1u;
    f.glc = (i.w0 >> 14) & 1u;
    f.lds = (i.w0 >> 16) & 1u;
    f.slc = (i.w0 >> 17) & 1u;
    return f;
}

namespace {

struct Ehdr { uint8_t ident[16]; uint16_t type, machine; uint32_t version; uint64_t entry, phoff, shoff; uint32_t flags;
              uint16_t ehsize, phentsize, phnum, shentsize, shnum, shstrndx; };
struct Shdr { uint32_t name, type; uint64_t flags, addr, offset, size; uint32_t link, info; uint64_t addralign, entsize; };
struct Sym { uint32_t name; uint8_t info, other; uint16_t shndx; uint64_t value, size; };
static_assert(sizeof(Ehdr) == 64 && sizeof(Shdr) == 64 && sizeof(Sym) == 24, "ELF64 layout");

const char* wide_name(uint32_t op) {
    return op == kBufferLoadDwordx4 ? "buffer_load_dwordx4" : op == kBufferLoadDwordx3 ? "buffer_load_dwordx3"
                                                                                        : "buffer_load_dwordx2";
}

} // namespace

IsaScanReport scan_code_object(const uint8_t* d, size_t n) {
    IsaScanReport rep;
    auto fail = [&](const std::string& m) { rep.errors.push_back(m); rep.ok = false; return rep; };
    if (!d || n < sizeof(Ehdr) || std::memcmp(d, "\x7f" "ELF", 4) != 0 || d[4] != 2 || d[5] != 1)
        return fail("not a little-endian ELF64 file");
    Ehdr eh;
    std::memcpy(&eh, d, sizeof(eh));
    if (eh.machine != 224) return fail("ELF machine is not EM_AMDGPU (224)");
    if (eh.shentsize != sizeof(Shdr) || eh.shoff > n || uint64_t(eh.shnum) * sizeof(Shdr) > n - eh.shoff)
        return fail("bad section header table");
    std::vector<Shdr> sh(eh.shnum);
    for (uint32_t i = 0; i < eh.shnum; ++i) std::memcpy(&sh[i], d + eh.shoff + uint64_t(i) * sizeof(Shdr), sizeof(Shdr));
    auto in_file = [&](const Shdr& s) { return s.type == 8 /*NOBITS*/ || (s.offset <= n && s.size <= n - s.offset); };
    for (const auto& s : sh) if (!in_file(s)) return fail("section out of bounds");
    if (eh.shstrndx >= eh.shnum) return fail("bad shstrndx");
    const Shdr& shstr = sh[eh.shstrndx];
    auto sec_name = [&](const Shdr& s) -> std::string {
        if (s.name >= shstr.size) return {};
        const char* p = reinterpret_cast<const char*>(d + shstr.offset + s.name);
        return std::string(p, strnlen(p, shstr.size - s.name));
    };
    int text = -1;
    for (uint32_t i = 0; i < eh.shnum; ++i) if (sec_name(sh[i]) == ".text") text = static_cast<int>(i);
    if (text < 0) return fail("no .text section");
    const Shdr& tx = sh[static_cast<size_t>(text)];
    rep.text_bytes = tx.size;
    for (uint32_t i = 0; i < eh.shnum; ++i) {
        if (sh[i].type != 2 /*SYMTAB*/ || sh[i].link >= eh.shnum) continue;
        const Shdr& strs = sh[sh[i].link];
        for (uint64_t s = 0; s < sh[i].size / sizeof(Sym); ++s) {
            Sym sym;
            std::memcpy(&sym, d + sh[i].offset + s * sizeof(Sym), sizeof(Sym));
            if ((sym.info & 0xF) != 2 /*FUNC*/ || sym.shndx != static_cast<uint16_t>(text) || sym.name >= strs.size) continue;
            const char* nm = reinterpret_cast<const char*>(d + strs.offset + sym.name);
            const std::string name(nm, strnlen(nm, strs.size - sym.name));
            if (sym.value < tx.addr || sym.size > tx.size || sym.value - tx.addr > tx.size - sym.size) {
                rep.errors.push_back("kernel '" + name + "': symbol range outside .text");
                continue;
            }
            ++rep.kernels;
            std::vector<Inst> insts;
            std::string err;
            if (!decode_gfx9(d + tx.offset + (sym.value - tx.addr), sym.size, insts, &err))
                rep.errors.push_back("kernel '" + name + "': " + err);
            rep.instructions += insts.size();
            for (const Inst& in : insts) {
                if (in.enc != Enc::MUBUF) continue;
                const MubufFields f = mubuf_fields(in);
                if (!f.lds) continue;
                if (f.op == kBufferLoadDword) { ++rep.mubuf_lds_dword; continue; }
                if (f.op == kBufferLoadDwordx2 || f.op == kBufferLoadDwordx3 || f.op == kBufferLoadDwordx4)
                    rep.findings.push_back({name, in.offset, std::string(wide_name(f.op)) +
                                                              " ... lds (LDS-direct is dword-only on gfx906; x4 is gfx950-only)"});
            }
        }
    }
    if (rep.kernels == 0) rep.errors.push_back("no kernel function symbols in .text");
    rep.ok = rep.errors.empty() && rep.findings.empty();
    return rep;
}

IsaScanReport scan_code_object_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { IsaScanReport r; r.errors.push_back("cannot open " + path); return r; }
    const std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return scan_code_object(b.data(), b.size());
}

std::vector<IsaFinding> scan_disassembly_text(const std::string& text) {
    std::vector<IsaFinding> out;
    std::istringstream is(text);
    std::string line, kernel;
    uint32_t idx = 0;
    while (std::getline(is, line)) {
        // "0000000000001000 <kernel_name>:" or "kernel_name:" labels.
        const size_t lt = line.find('<'), gt = line.find(">:");
        if (lt != std::string::npos && gt != std::string::npos && gt > lt) { kernel = line.substr(lt + 1, gt - lt - 1); idx = 0; continue; }
        if (!line.empty() && line.back() == ':' && line.find(' ') == std::string::npos && line.find('\t') == std::string::npos) {
            kernel = line.substr(0, line.size() - 1);
            idx = 0;
            continue;
        }
        const size_t c = std::min(line.find("//"), line.find(';'));  // both are assembler comment markers
        const std::string code = c == std::string::npos ? line : line.substr(0, c);
        for (const char* op : {"buffer_load_dwordx4", "buffer_load_dwordx3", "buffer_load_dwordx2"}) {
            const size_t p = code.find(op);
            if (p == std::string::npos) continue;
            std::istringstream toks(code.substr(p + std::strlen(op)));
            std::string t;
            bool lds = false;
            while (toks >> t) {
                while (!t.empty() && t.back() == ',') t.pop_back();
                if (t == "lds") lds = true;
            }
            if (lds) out.push_back({kernel, idx, std::string(op) + " ... lds"});
            break;
        }
        ++idx;
    }
    return out;
}

} // namespace vegajet::isa
