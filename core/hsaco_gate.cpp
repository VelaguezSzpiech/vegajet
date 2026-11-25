// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/hsaco_gate.hpp"

#include "core/grid_planner.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>

namespace vegajet::gate {

namespace {

// Minimal ELF64 little-endian layouts (kept local so CPU-only builds need no <elf.h>).
struct Ehdr {
    uint8_t  ident[16];
    uint16_t type, machine;
    uint32_t version;
    uint64_t entry, phoff, shoff;
    uint32_t flags;
    uint16_t ehsize, phentsize, phnum, shentsize, shnum, shstrndx;
};
struct Shdr {
    uint32_t name, type;
    uint64_t flags, addr, offset, size;
    uint32_t link, info;
    uint64_t addralign, entsize;
};
struct Sym {
    uint32_t name;
    uint8_t  info, other;
    uint16_t shndx;
    uint64_t value, size;
};
constexpr uint32_t kShtSymtab = 2, kShtDynsym = 11;
constexpr uint8_t  kSttFunc = 2;

const std::vector<GateEntry> kManifest = {
#include "hsaco_gates.inc"
};

} // namespace

GateResult evaluate(const GateEntry* entry, const std::string& kernel, uint32_t vgpr_count,
                    uint32_t private_segment_fixed_size, uint64_t code_bytes, std::string* why) {
    auto fail = [&](const std::string& msg) {
        if (why) *why = "kernel '" + kernel + "': " + msg;
        return GateResult::Fail;
    };
    if (!entry) return fail("not listed in ops/kernels/hsaco_gates.txt (classify it as stream or tiled)");
    if (private_segment_fixed_size != 0u)
        return fail("private_segment_fixed_size=" + std::to_string(private_segment_fixed_size) + " B (must be 0 B)");
    if (code_bytes >= kMaxCodeBytes)
        return fail("code size " + std::to_string(code_bytes) + " B (must be < " + std::to_string(kMaxCodeBytes) + " B)");
    const uint32_t limit = class_limit(entry->klass);
    const char* cname = entry->klass == KernelClass::Stream ? "stream" : "tiled";
    if (entry->waiver_vgpr != 0u) {
        if (vgpr_count <= limit)
            return fail("stale waiver: vgpr_count=" + std::to_string(vgpr_count) + " already fits the " + cname +
                        " limit " + std::to_string(limit) + "; remove the waiver from hsaco_gates.txt");
        if (vgpr_count > entry->waiver_vgpr)
            return fail("vgpr_count=" + std::to_string(vgpr_count) + " exceeds its waiver ceiling " +
                        std::to_string(entry->waiver_vgpr) + " (" + cname + " limit " + std::to_string(limit) + ")");
        if (why) *why = "kernel '" + kernel + "': vgpr_count=" + std::to_string(vgpr_count) + " > " + cname + " limit " +
                        std::to_string(limit) + " (waived up to " + std::to_string(entry->waiver_vgpr) + ")";
        return GateResult::OkWaived;
    }
    if (vgpr_count > limit)
        return fail("vgpr_count=" + std::to_string(vgpr_count) + " exceeds the " + cname + " limit " + std::to_string(limit));
    return GateResult::Ok;
}

const std::vector<GateEntry>& manifest() { return kManifest; }

const GateEntry* find_entry(const std::string& kernel) {
    for (const auto& e : kManifest)
        if (kernel == e.name) return &e;
    return nullptr;
}

std::unordered_map<std::string, uint64_t> elf_function_sizes(const uint8_t* data, size_t size) {
    std::unordered_map<std::string, uint64_t> out;
    if (!data || size < sizeof(Ehdr) || std::memcmp(data, "\x7f" "ELF", 4) != 0 || data[4] != 2) return out;
    Ehdr eh;
    std::memcpy(&eh, data, sizeof(eh));
    if (eh.shentsize != sizeof(Shdr) || eh.shoff > size || static_cast<uint64_t>(eh.shnum) * sizeof(Shdr) > size - eh.shoff) return out;
    auto shdr = [&](uint32_t i) { Shdr s; std::memcpy(&s, data + eh.shoff + static_cast<uint64_t>(i) * sizeof(Shdr), sizeof(s)); return s; };
    for (uint32_t i = 0; i < eh.shnum; ++i) {
        const Shdr sh = shdr(i);
        if (sh.type != kShtSymtab && sh.type != kShtDynsym) continue;
        if (sh.link >= eh.shnum || sh.offset > size || sh.size > size - sh.offset) continue;
        const Shdr str = shdr(sh.link);
        if (str.offset > size || str.size > size - str.offset) continue;
        const size_t n = sh.size / sizeof(Sym);
        for (size_t s = 0; s < n; ++s) {
            Sym sym;
            std::memcpy(&sym, data + sh.offset + s * sizeof(Sym), sizeof(sym));
            if ((sym.info & 0xFu) != kSttFunc || sym.name >= str.size) continue;
            const char* nm = reinterpret_cast<const char*>(data + str.offset + sym.name);
            const size_t maxlen = str.size - sym.name;
            const size_t len = strnlen(nm, maxlen);
            if (len == maxlen) continue;
            auto& slot = out[std::string(nm, len)];
            if (sym.size > slot) slot = sym.size;
        }
    }
    return out;
}

FileGateReport check_code_object_file(const std::string& path) {
    FileGateReport rep;
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) { rep.failures.push_back("cannot open " + path); return rep; }
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const auto co = planner::parse_code_object(bytes.data(), bytes.size());
    if (!co.ok) { rep.failures.push_back(path + ": " + co.error); return rep; }
    if (co.kernels.empty()) { rep.failures.push_back(path + ": no kernels in amdhsa.kernels metadata"); return rep; }
    const auto sizes = elf_function_sizes(bytes.data(), bytes.size());
    for (const auto& k : co.kernels) {
        ++rep.kernels;
        const auto it = sizes.find(k.name);
        if (it == sizes.end()) { rep.failures.push_back("kernel '" + k.name + "': no function symbol (cannot check code size)"); continue; }
        rep.max_vgpr = std::max(rep.max_vgpr, k.vgpr_count);
        rep.max_code_bytes = std::max<uint64_t>(rep.max_code_bytes, it->second);
        std::string why;
        const GateResult r = evaluate(find_entry(k.name), k.name, k.vgpr_count, k.private_segment_fixed_size, it->second, &why);
        if (r == GateResult::Fail) rep.failures.push_back(why);
        else if (r == GateResult::OkWaived) { ++rep.waived; rep.waivers.push_back(why); }
    }
    rep.ok = rep.failures.empty();
    return rep;
}

} // namespace vegajet::gate
