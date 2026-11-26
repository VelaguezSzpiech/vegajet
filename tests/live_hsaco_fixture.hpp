// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include "core/hsaco_loader.hpp"
#include <elf.h>
#include <cstring>
#include <vector>

namespace vjtest {
template<class T> void put(std::vector<uint8_t>& b, size_t off, const T& v) { std::memcpy(b.data() + off, &v, sizeof(v)); }
inline std::vector<uint8_t> live_hsaco_fixture() {
    std::vector<uint8_t> b(1536), mp;
    auto str = [&](const std::string& s) {
        if (s.size() < 32) mp.push_back(uint8_t(0xa0 | s.size()));
        else { mp.push_back(0xd9); mp.push_back(uint8_t(s.size())); }
        mp.insert(mp.end(), s.begin(), s.end());
    };
    mp.push_back(0x82); str("amdhsa.target"); str("amdgcn-amd-amdhsa--gfx906");
    str("amdhsa.kernels"); mp.push_back(0x91); mp.push_back(0x89);
    str(".name"); str("kfd_memset_u32"); str(".symbol"); str("kfd_memset_u32.kd");
    for (const auto& p : std::vector<std::pair<std::string, uint32_t>>{
        {".vgpr_count",32},{".sgpr_count",24},{".group_segment_fixed_size",0},{".private_segment_fixed_size",0},
        {".kernarg_segment_size",272},{".wavefront_size",64},{".max_flat_workgroup_size",256}}) {
        str(p.first); mp.push_back(0xce);
        for (int shift = 24; shift >= 0; shift -= 8) mp.push_back(uint8_t(p.second >> shift));
    }
    Elf64_Ehdr eh{};
    std::memcpy(eh.e_ident, ELFMAG, SELFMAG);
    eh.e_ident[EI_CLASS]=2; eh.e_ident[EI_DATA]=1; eh.e_ident[EI_VERSION]=1;
    eh.e_ident[EI_OSABI]=64; eh.e_ident[EI_ABIVERSION]=4;
    eh.e_type=ET_DYN; eh.e_machine=EM_AMDGPU; eh.e_version=1; eh.e_flags=0x52f;
    eh.e_ehsize=64; eh.e_phoff=64; eh.e_phnum=1; eh.e_phentsize=56;
    eh.e_shoff=128; eh.e_shnum=5; eh.e_shentsize=64;
    put(b,0,eh);
    Elf64_Phdr ph{}; ph.p_type=PT_LOAD; ph.p_filesz=b.size(); ph.p_memsz=b.size(); ph.p_flags=PF_R|PF_X; ph.p_align=4096;
    put(b,64,ph);
    Elf64_Shdr note{}; note.sh_type=SHT_NOTE; note.sh_offset=512; note.sh_size=(20+mp.size()+3)&~3ULL;
    put(b,128+64,note);
    Elf64_Nhdr nh{}; nh.n_namesz=7; nh.n_descsz=uint32_t(mp.size()); nh.n_type=32;
    put(b,512,nh); std::memcpy(b.data()+524,"AMDGPU",7); std::memcpy(b.data()+532,mp.data(),mp.size());
    const char names[]="\0kfd_memset_u32\0kfd_memset_u32.kd\0";
    std::memcpy(b.data()+1024,names,sizeof(names));
    Elf64_Shdr strings{}; strings.sh_type=SHT_STRTAB; strings.sh_offset=1024; strings.sh_size=sizeof(names);
    put(b,128+128,strings);
    Elf64_Shdr symbols{}; symbols.sh_type=SHT_SYMTAB; symbols.sh_offset=1152; symbols.sh_size=72; symbols.sh_entsize=24; symbols.sh_link=2;
    put(b,128+192,symbols);
    Elf64_Shdr code{}; code.sh_type=SHT_PROGBITS; code.sh_flags=SHF_ALLOC|SHF_EXECINSTR; code.sh_addr=1280; code.sh_offset=1280; code.sh_size=256;
    put(b,128+256,code);
    Elf64_Sym fn{}; fn.st_name=1; fn.st_info=ELF64_ST_INFO(STB_GLOBAL,STT_FUNC); fn.st_shndx=4; fn.st_value=1408; fn.st_size=4;
    put(b,1152+24,fn);
    Elf64_Sym ds{}; ds.st_name=16; ds.st_info=ELF64_ST_INFO(STB_GLOBAL,STT_OBJECT); ds.st_shndx=4; ds.st_value=1280; ds.st_size=64;
    put(b,1152+48,ds);
    vegajet::kfd::KernelDescriptor kd{}; kd.kernarg_size=272; kd.kernel_code_entry_byte_offset=128;
    put(b,1280,kd);
    return b;
}
}
