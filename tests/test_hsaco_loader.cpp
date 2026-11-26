// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "tests/test_common.hpp"
#include "tests/live_hsaco_fixture.hpp"
#include <filesystem>
#include <fstream>
using namespace vegajet::kfd;
int main(int argc, char** argv) {
    CodeLoadPlan plan;
    std::string err;
    const auto valid = vjtest::live_hsaco_fixture();
    if (!plan_hsaco_load(valid.data(),valid.size(),plan,&err)) {
        std::fprintf(stderr,"fixture: %s\n",err.c_str()); return 1;
    }
    VJ_CHECK(plan.image_bytes==4096 && plan.kernels.size()==1);
    auto bad64 = [&](size_t off, uint64_t v) {
        auto b=valid; vjtest::put(b,off,v);
        VJ_CHECK(!plan_hsaco_load(b.data(),b.size(),plan,&err));
        VJ_CHECK(!err.empty() && plan.kernels.empty());
    };
    bad64(32, UINT64_MAX); // program table
    bad64(40, UINT64_MAX); // section table
    bad64(64+8, UINT64_MAX); // segment source
    bad64(64+16, UINT64_MAX-16); // VA wrap
    bad64(64+32, 1537); // filesz > memsz
    bad64(64+40, kMaxCodeBytes+1); // oversized image
    bad64(128+192+24, UINT64_MAX); // symbol table source
    bad64(128+192+56, 8); // wrong symbol entry size
    auto bad32 = [&](size_t off,uint32_t v) {
        auto b=valid; vjtest::put(b,off,v);
        VJ_CHECK(!plan_hsaco_load(b.data(),b.size(),plan,&err));
    };
    bad32(128+192+40,99); // sh_link
    bad32(1152+48,99); // st_name
    bad32(128+64+4,SHT_RELA); // unsupported relocations
    bad32(1280+4,4); // descriptor private vs metadata
    bad32(1280+8,1024); // unsupported kernargs
    bad64(1152+48+8,0); // descriptor below load section
    bad64(1152+48+8,4096); // descriptor beyond image
    bad64(1280+16,UINT64_MAX); // entry offset underflow/mismatch
    auto b=valid; b[7]=0; VJ_CHECK(!plan_hsaco_load(b.data(),b.size(),plan,&err));
    b=valid; b[8]=1; VJ_CHECK(!plan_hsaco_load(b.data(),b.size(),plan,&err));
    b=valid; b[1024+sizeof("\0kfd_memset_u32\0kfd_memset_u32.kd\0")-1]='X';
    b[1024+sizeof("\0kfd_memset_u32\0kfd_memset_u32.kd\0")-2]='X';
    VJ_CHECK(!plan_hsaco_load(b.data(),b.size(),plan,&err));
    for (size_t n=0;n<valid.size();++n) VJ_CHECK(!plan_hsaco_load(valid.data(),n,plan,&err));
    // Optional qualification of every real ROCm artifact, still without a GPU.
    unsigned count=0;
    if (argc==2) for (const auto& p:std::filesystem::directory_iterator(argv[1])) {
        if (p.path().extension()!=".hsaco") continue;
        std::ifstream f(p.path(),std::ios::binary|std::ios::ate);
        const auto size=f.tellg(); VJ_CHECK(size>0);
        std::vector<uint8_t> bytes(static_cast<size_t>(size)); f.seekg(0);
        VJ_CHECK(bool(f.read(reinterpret_cast<char*>(bytes.data()),size)));
        if (!plan_hsaco_load(bytes.data(),bytes.size(),plan,&err)) {
            std::fprintf(stderr,"%s: %s\n",p.path().c_str(),err.c_str()); return 1;
        }
        ++count;
    }
    std::printf("live HSACO loader malformed-input tests PASS; real artifacts=%u\n",count);
}
