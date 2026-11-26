// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/offload/engram.hpp"
#include <cassert>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <unistd.h>
using namespace vegajet::offload;
int main() {
    char path[]="/tmp/vj-engram-test-XXXXXX"; int fd=mkstemp(path); assert(fd>=0);
    std::vector<uint8_t> rows(4096*16); for(size_t i=0;i<rows.size();++i) rows[i]=uint8_t(i*17+i/4096);
    assert(write(fd,rows.data(),rows.size())==ssize_t(rows.size())); assert(fsync(fd)==0);
    for(bool use_uring:{false,true}) {
        Engram io(path,4096,16,8,use_uring);
        std::cout<<io.backend()<<" direct="<<io.direct()<<" pinned="<<io.pinned()<<" registered="<<io.registered()<<'\n';
        for(int k=0;k<12;++k) {
            std::vector<uint64_t> ids={15,0,7,7,uint64_t(k)}; assert(io.submit(ids)); assert(!io.submit({0}));
            while(!io.poll()) std::this_thread::yield();
            assert(io.success());
            for(size_t j=0;j<ids.size();++j) { uint8_t reference[4096]; assert(pread(fd,reference,sizeof(reference),ids[j]*4096)==4096); assert(std::memcmp(io.data(j),reference,4096)==0); }
        }
        assert(!io.submit({16})); assert(!io.submit(std::vector<uint64_t>(9,0))); assert(io.submit({})); assert(io.poll());
    }
    close(fd); unlink(path);
}
