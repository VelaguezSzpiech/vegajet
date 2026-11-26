// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/offload/cuda_wire.hpp"
#include "core/offload/host.hpp"
#include <cassert>
#include <cstring>
using namespace vegajet::offload;
int main() {
    HostArena arena(huge_bytes); std::memset(arena.data(),0,arena.size());
    auto r=static_cast<CudaRing*>(arena.data());
    std::thread mock([&]{ for(unsigned i=0;i<10000;++i) { auto& c=r->commands[i%cuda_slots];
        while(__atomic_load_n(&c.state,__ATOMIC_ACQUIRE)!=1) std::this_thread::yield();
        c.y[0]=c.x[0]*c.scales[0]; __atomic_store_n(&c.state,2u,__ATOMIC_RELEASE);
    }});
    for(unsigned i=0;i<10000;++i) { auto& c=r->commands[i%cuda_slots]; assert(c.state==0); c.x[0]=int8_t(i); c.scales[0]=0.5f;
        __atomic_store_n(&c.state,1u,__ATOMIC_RELEASE);
        while(__atomic_load_n(&c.state,__ATOMIC_ACQUIRE)!=2) std::this_thread::yield();
        assert(c.y[0]==int8_t(i)*0.5f); __atomic_store_n(&c.state,0u,__ATOMIC_RELEASE);
    } mock.join();
}
