// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/hsaco_loader.hpp"
#include "core/hsaco_bundle.hpp"
#include "core/hsaco_gate.hpp"
#include <cstring>
#include <elf.h>
#include <map>

namespace vegajet::kfd {
bool plan_hsaco_load(const uint8_t* data, size_t bytes, CodeLoadPlan& out, std::string* err) {
    out = {};
    auto fail = [&](const std::string& why) { if (err) *err = "Invalid live HSACO: " + why; return false; };
    auto range = [](uint64_t off, uint64_t n, uint64_t end) { return off <= end && n <= end - off; };
    if (!data || bytes < sizeof(Elf64_Ehdr) || bytes > kMaxCodeBytes) return fail("file size outside bounds");
    Elf64_Ehdr eh{};
    std::memcpy(&eh, data, sizeof(eh));
    if (std::memcmp(eh.e_ident, ELFMAG, SELFMAG) || eh.e_ident[EI_CLASS] != ELFCLASS64 ||
        eh.e_ident[EI_DATA] != ELFDATA2LSB || eh.e_ident[EI_VERSION] != EV_CURRENT ||
        eh.e_type != ET_DYN || eh.e_machine != EM_AMDGPU || eh.e_version != EV_CURRENT ||
        (eh.e_flags & 0xff) != 0x2f || (eh.e_flags & ~0xfffu) || eh.e_ident[EI_OSABI] != 64 ||
        (eh.e_ident[EI_ABIVERSION] != 3 && eh.e_ident[EI_ABIVERSION] != 4))
        return fail("expected gfx906 AMDHSA ELF64 code object ABI v5/v6");
    if (eh.e_ehsize != sizeof(eh) || eh.e_phentsize != sizeof(Elf64_Phdr) ||
        eh.e_shentsize != sizeof(Elf64_Shdr) || !eh.e_phnum || eh.e_phnum == PN_XNUM || !eh.e_shnum ||
        !range(eh.e_phoff, uint64_t(eh.e_phnum) * sizeof(Elf64_Phdr), bytes) ||
        !range(eh.e_shoff, uint64_t(eh.e_shnum) * sizeof(Elf64_Shdr), bytes))
        return fail("ELF header tables out of bounds or unsupported extended numbering");
    std::vector<Elf64_Phdr> ph(eh.e_phnum);
    std::vector<Elf64_Shdr> sh(eh.e_shnum);
    std::memcpy(ph.data(), data + eh.e_phoff, ph.size() * sizeof(ph[0]));
    std::memcpy(sh.data(), data + eh.e_shoff, sh.size() * sizeof(sh[0]));
    auto dynamic_ok = [&](uint64_t off, uint64_t size) {
        if (size % sizeof(Elf64_Dyn)) return false;
        for (uint64_t pos = 0; pos < size; pos += sizeof(Elf64_Dyn)) {
            Elf64_Dyn d{}; std::memcpy(&d, data + off + pos, sizeof(d));
            if (d.d_tag == DT_NEEDED || d.d_tag == DT_TEXTREL || d.d_tag == DT_JMPREL ||
                d.d_tag == DT_REL || d.d_tag == DT_RELA || d.d_tag == DT_RELR ||
                d.d_tag == DT_RELCOUNT || d.d_tag == DT_RELACOUNT) return false;
        }
        return true;
    };
    uint64_t lo = UINT64_MAX, hi = 0;
    for (size_t i = 0; i < ph.size(); ++i) {
        const auto& p = ph[i];
        if (!range(p.p_offset, p.p_filesz, bytes) || p.p_memsz > UINT64_MAX - p.p_vaddr)
            return fail("program segment source/VA overflow");
        if (p.p_type == PT_DYNAMIC && !dynamic_ok(p.p_offset, p.p_filesz))
            return fail("dynamic segment dependencies/relocations unsupported");
        if (p.p_type == PT_INTERP || p.p_type == PT_TLS) return fail("unsupported interpreter/TLS segment");
        if (p.p_type != PT_LOAD || !p.p_memsz) continue;
        if (p.p_filesz > p.p_memsz || (p.p_align > 1 &&
            ((p.p_align & (p.p_align - 1)) || p.p_vaddr % p.p_align != p.p_offset % p.p_align)))
            return fail("invalid PT_LOAD size/alignment");
        for (size_t j = 0; j < i; ++j)
            if (ph[j].p_type == PT_LOAD && ph[j].p_memsz && p.p_vaddr < ph[j].p_vaddr + ph[j].p_memsz &&
                ph[j].p_vaddr < p.p_vaddr + p.p_memsz) return fail("overlapping PT_LOAD segments");
        lo = std::min(lo, p.p_vaddr); hi = std::max(hi, p.p_vaddr + p.p_memsz);
    }
    if (lo >= hi || hi - lo > kMaxCodeBytes - 4095) return fail("missing/oversized load image");
    // Every descriptor/function must lie wholly in a file-backed load segment, not a gap or BSS.
    auto file_at = [&](uint64_t va, uint64_t n, bool exec, uint64_t& off) {
        for (const auto& p : ph) {
            if (p.p_type == PT_LOAD && (!exec || (p.p_flags & PF_X)) && va >= p.p_vaddr &&
                range(va - p.p_vaddr, n, p.p_filesz)) { off = p.p_offset + va - p.p_vaddr; return true; }
        }
        return false;
    };
    for (const auto& s : sh) {
        if (s.sh_type != SHT_NOBITS && !range(s.sh_offset, s.sh_size, bytes)) return fail("section outside file");
        if (s.sh_size && (s.sh_type == SHT_REL || s.sh_type == SHT_RELA || s.sh_type == SHT_RELR))
            return fail("relocations unsupported by direct loader");
        if (s.sh_flags & SHF_TLS) return fail("TLS unsupported by direct loader");
        if (s.sh_type == SHT_DYNAMIC) {
            if (s.sh_entsize != sizeof(Elf64_Dyn) || s.sh_size % sizeof(Elf64_Dyn)) return fail("invalid dynamic table");
            if (!dynamic_ok(s.sh_offset, s.sh_size))
                return fail("dynamic dependencies/relocations unsupported by direct loader");
        }
    }
    if (eh.e_shstrndx != SHN_UNDEF && (eh.e_shstrndx >= sh.size() || sh[eh.e_shstrndx].sh_type != SHT_STRTAB))
        return fail("invalid section-name table");
    std::map<std::string, Elf64_Sym> descriptors, functions;
    for (const auto& s : sh) {
        if (s.sh_type != SHT_SYMTAB && s.sh_type != SHT_DYNSYM) continue;
        if (s.sh_entsize != sizeof(Elf64_Sym) || s.sh_size % sizeof(Elf64_Sym) ||
            s.sh_link >= sh.size() || sh[s.sh_link].sh_type != SHT_STRTAB)
            return fail("invalid symbol table entry size/link");
        const auto& strings = sh[s.sh_link];
        for (uint64_t pos = 0; pos < s.sh_size; pos += sizeof(Elf64_Sym)) {
            Elf64_Sym sym{}; std::memcpy(&sym, data + s.sh_offset + pos, sizeof(sym));
            if (sym.st_name >= strings.sh_size) return fail("symbol name outside string table");
            const char* name = reinterpret_cast<const char*>(data + strings.sh_offset + sym.st_name);
            const void* end = std::memchr(name, 0, strings.sh_size - sym.st_name);
            if (!end) return fail("unterminated symbol name");
            const std::string n(name, static_cast<const char*>(end) - name);
            if (sym.st_shndx == SHN_UNDEF) {
                if (!n.empty()) return fail("unresolved symbol '" + n + "'");
                continue;
            }
            const bool kd = n.size() > 3 && n.compare(n.size() - 3, 3, ".kd") == 0;
            const bool fn = ELF64_ST_TYPE(sym.st_info) == STT_FUNC;
            if (!kd && !fn) continue;
            if (sym.st_shndx >= sh.size()) return fail("invalid symbol section index");
            const auto& section = sh[sym.st_shndx];
            const uint64_t need = kd ? sizeof(KernelDescriptor) : sym.st_size;
            uint64_t file = 0;
            if (!(section.sh_flags & SHF_ALLOC) || sym.st_value < section.sh_addr ||
                !range(sym.st_value - section.sh_addr, need, section.sh_size) ||
                !file_at(sym.st_value, need, fn, file) || !need || (kd && sym.st_value % 64))
                return fail("descriptor/function outside backed load segment: " + n);
            auto& table = kd ? descriptors : functions;
            auto [it, inserted] = table.emplace(kd ? n.substr(0, n.size() - 3) : n, sym);
            if (!inserted && (it->second.st_value != sym.st_value || it->second.st_size != sym.st_size))
                return fail("conflicting symbol: " + n);
        }
    }
    bundle::Gfx906Budget budget;
    budget.max_vgpr = 256;  // per-kernel gate table below owns occupancy limits and waivers
    const auto metadata = bundle::inspect_code_object(data, bytes, budget);
    if (!metadata.ok) return fail(!metadata.error.empty() ? metadata.error :
                                  metadata.problems.empty() ? "invalid metadata" : metadata.problems.front());
    if (metadata.kernels.size() != descriptors.size()) return fail("descriptor/metadata kernel set differs");
    CodeLoadPlan plan;
    plan.image_bytes = (hi - lo + 4095) & ~4095ULL;
    for (const auto& p : ph) if (p.p_type == PT_LOAD && p.p_memsz)
        plan.segments.push_back({p.p_offset, p.p_vaddr - lo, p.p_filesz});
    for (const auto& k : metadata.kernels) {
        const auto ds = descriptors.find(k.name), fn = functions.find(k.name);
        if (ds == descriptors.end() || fn == functions.end()) return fail("missing descriptor/function: " + k.name);
        uint64_t file = 0;
        if (!file_at(ds->second.st_value, sizeof(KernelDescriptor), false, file)) return fail("descriptor not backed");
        KernelDescriptor kd{};
        std::memcpy(&kd, data + file, sizeof(kd));
        const int64_t delta = kd.kernel_code_entry_byte_offset;
        uint64_t entry = ds->second.st_value;
        if (delta < 0) {
            const uint64_t neg = uint64_t(-(delta + 1)) + 1;
            if (neg > entry) return fail("kernel entry underflow");
            entry -= neg;
        } else {
            if (uint64_t(delta) > UINT64_MAX - entry) return fail("kernel entry overflow");
            entry += uint64_t(delta);
        }
        if (entry != fn->second.st_value || !file_at(entry, fn->second.st_size, true, file))
            return fail("kernel entry differs from executable function: " + k.name);
        if (kd.private_segment_fixed_size != k.private_segment_fixed_size || kd.group_segment_fixed_size != k.group_segment_fixed_size ||
            kd.kernarg_size != k.kernarg_segment_size || kd.kernarg_size > 512 || k.max_flat_workgroup_size > 1024 ||
            !k.max_flat_workgroup_size || !k.vgpr_count || k.sgpr_count > 102 || (kd.kernel_code_properties & 0x400))
            return fail("descriptor/metadata ABI mismatch or unsupported resources: " + k.name);
        std::string why;
        if (gate::evaluate(gate::find_entry(k.name), k.name, k.vgpr_count, k.private_segment_fixed_size, fn->second.st_size, &why) == gate::GateResult::Fail)
            return fail("code-object gate: " + why);
        LoadedKernel lk;
        lk.name = k.name; lk.kd_gpu_va = ds->second.st_value - lo;
        lk.group_segment_fixed_size = k.group_segment_fixed_size; lk.private_segment_fixed_size = k.private_segment_fixed_size;
        lk.kernarg_size = k.kernarg_segment_size; lk.vgpr_count = k.vgpr_count; lk.sgpr_count = k.sgpr_count;
        lk.wavefront_size = k.wavefront_size; lk.max_flat_workgroup_size = k.max_flat_workgroup_size; lk.from_msgpack_note = true;
        const auto vg = (std::max(4u, k.vgpr_count) + 3) & ~3u;
        const auto sg = (std::max(16u, k.sgpr_count + 6) + 15) & ~15u;
        lk.resident_waves_per_simd = std::min(10u, std::min(256u / vg, 800u / sg));
        plan.kernels.emplace(k.name, std::move(lk));
    }
    out = std::move(plan);
    return true;
}
}
