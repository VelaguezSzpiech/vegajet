// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/offload/policies.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
namespace vegajet::offload {
BorrowedKv::BorrowedKv(uint32_t chunks,uint64_t page,uint32_t experts,double decay):decay_(decay),counts_(experts,0) {
    constexpr uint64_t chunk=512ull*1024*1024;
    if(!chunks || !page || chunk%page || !experts || !std::isfinite(decay) || decay<0 || decay>1)
        throw std::invalid_argument("borrowed KV config");
    chunk_pages_=chunk/page; pages_=uint64_t(chunks)*chunk_pages_;
}
void BorrowedKv::route(uint32_t e,double w) {
    if(e>=counts_.size() || !std::isfinite(w) || w<0) throw std::invalid_argument("routing count");
    counts_[e]+=w;
}
const std::vector<Lease>& BorrowedKv::decode_step(uint64_t kv,uint64_t reserve,uint64_t size) {
    if(kv>pages_ || reserve>pages_-kv || !size) { leases_.clear(); ++step_; for(auto& c:counts_) c*=decay_; return leases_; }
    uint64_t start=kv+reserve;
    if(step_%8==0) {
        std::vector<uint32_t> order(counts_.size()); std::iota(order.begin(),order.end(),0);
        std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){ return counts_[a]>counts_[b]; });
        leases_.clear();
        for(auto e:order) {
            // Keep each lease within its arena chunk.
            if(size>chunk_pages_) break;
            if(start%chunk_pages_+size>chunk_pages_) start+=chunk_pages_-start%chunk_pages_;
            if(start>pages_ || size>pages_-start) break;
            if(counts_[e]>0) { leases_.push_back({e,start,size}); start+=size; }
        }
    } else {
        leases_.erase(std::remove_if(leases_.begin(),leases_.end(),[&](auto l){ return l.first_page<start || l.pages!=size; }),leases_.end());
    }
    for(auto& c:counts_) c*=decay_;
    ++step_; return leases_;
}
FetchPlan hobbit_plan(uint32_t e,uint32_t rank,double w,double t) {
    if(!std::isfinite(w)||!std::isfinite(t)||w<0||w>1||t<0||t>1) throw std::invalid_argument("HOBBIT weight/threshold");
    return {e,rank,rank==0?FetchPrecision::Full:(w<t?FetchPrecision::Skip:FetchPrecision::Low)};
}
}
