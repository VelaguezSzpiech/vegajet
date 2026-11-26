// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/offload/policies.hpp"
#include <cassert>
#include <limits>
#include <stdexcept>
using namespace vegajet::offload;
int main() {
    BorrowedKv pool(2,2*1024*1024,10,0.5);
    pool.route(3,10); pool.route(4,3);
    auto leases=pool.decode_step(250,0,4); assert(leases.size()==2); assert(leases[0].expert==3); assert(leases[0].first_page==250);
    assert(leases[1].first_page==256); assert(pool.counts()[3]==5);
    pool.route(5,1000);
    leases=pool.decode_step(255,0,4); assert(leases.size()==1 && leases[0].expert==4);
    for(int i=2;i<8;++i) { leases=pool.decode_step(255,0,4); assert(leases.size()==1 && leases[0].expert==4); }
    leases=pool.decode_step(255,0,4); assert(leases[0].expert==5);
    for(uint64_t kv=0;kv<=512;++kv) {
        auto l=pool.decode_step(kv,0,4); uint64_t used=0;
        for(auto a:l) { assert(a.first_page>=kv); assert(a.first_page+a.pages<=512); assert(a.first_page/256==(a.first_page+a.pages-1)/256); used+=a.pages; }
        assert(used<=512-kv);
        for(size_t a=0;a<l.size();++a) for(size_t b=a+1;b<l.size();++b) {
            assert(l[a].expert!=l[b].expert); assert(l[a].first_page+l[a].pages<=l[b].first_page);
        }
    }
    assert(pool.decode_step(UINT64_MAX,1,4).empty()); assert(pool.decode_step(500,UINT64_MAX,4).empty());
    assert(hobbit_plan(9,0,0,0.9).precision==FetchPrecision::Full);
    assert(hobbit_plan(9,1,0.1,0.1).precision==FetchPrecision::Low);
    assert(hobbit_plan(9,2,0.09,0.1).precision==FetchPrecision::Skip);
    assert(hobbit_plan(9,5,0,0).precision==FetchPrecision::Low);
    bool caught=false; try { hobbit_plan(0,1,std::numeric_limits<double>::quiet_NaN(),0); } catch(const std::invalid_argument&) { caught=true; } assert(caught);
}
