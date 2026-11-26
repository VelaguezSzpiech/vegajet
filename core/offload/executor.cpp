// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/offload/executor.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <thread>
namespace vegajet::offload {
ExpertExecutor::ExpertExecutor(Switches s,uint32_t slots,MissBackends b)
    : switches_(s),capacity_(slots),backends_(std::move(b)) {
    if (!slots) throw std::invalid_argument("expert pool capacity");
}
ExpertExecutor::~ExpertExecutor()=default;
bool ExpertExecutor::miss(uint64_t key,Matrix m,Activation x,float* y) {
    if (!m.valid() || !x.values || !x.scales || !y) throw std::invalid_argument("expert miss shape");
    ++stats_.misses;
    auto& e=entries_[key]; e.matrix=m; ++e.frequency;
    if (switches_.borrowed_kv && e.hot && backends_.hot && backends_.hot(key,m,x,y)) { ++stats_.hot; return true; }
    if (backends_.fetch && !backends_.fetch(key)) { ++stats_.failed; return false; }
    if (switches_.l2) {
        bool ok=false;
        if (backends_.cuda) ok=backends_.cuda(key,m,x,y);
        else {
            auto it=cuda_.find(key);
            if (it==cuda_.end()) {
                if (cuda_.size()>=capacity_) {
                    auto victim=std::min_element(cuda_.begin(),cuda_.end(),[&](const auto& a,const auto& b) {
                        return entries_.at(a.first).frequency<entries_.at(b.first).frequency;
                    });
                    cuda_.erase(victim);
                }
                it=cuda_.emplace(key,std::make_unique<CudaTier>(m)).first;
            }
            if (it!=cuda_.end() && it->second->ok() && it->second->submit(x)) {
                while (it->second->ok() && !(ok=it->second->poll(y))) std::this_thread::yield();
            }
        }
        if (ok) { ++stats_.cuda; return true; }
    }
    if (switches_.l3) {
        bool ok=true;
        if (backends_.cpu) ok=backends_.cpu(key,m,x,y);
        else gemv(m,x,y);
        if (ok) { ++stats_.cpu; return true; }
    }
    ++stats_.failed; return false;
}
void ExpertExecutor::end_step() {
    if (steps_++%8u==0u && switches_.borrowed_kv) {
        std::vector<uint64_t> order;
        for (auto& e:entries_) { e.second.hot=false; order.push_back(e.first); }
        std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return entries_.at(a).frequency>entries_.at(b).frequency;});
        if (order.size()>capacity_) order.resize(capacity_);
        for (auto key:order) entries_.at(key).hot=true;
        for (auto it=cuda_.begin();it!=cuda_.end();) {
            if (!entries_.at(it->first).hot) it=cuda_.erase(it); else ++it;
        }
    }
    for (auto& e:entries_) e.second.frequency*=0.95;
}
std::function<bool(uint64_t)> ExpertExecutor::table_fetch(Engram& table) {
    return [&table](uint64_t key) {
        if (!table.submit({key})) return false;
        while (!table.poll()) std::this_thread::yield();
        return table.success();
    };
}
std::vector<uint32_t> cold_experts(const std::vector<double>& freq,double fraction) {
    if (!std::isfinite(fraction)||fraction<0||fraction>1) throw std::invalid_argument("expert fraction must be 0..1");
    for (double f:freq) if (!std::isfinite(f)||f<0) throw std::invalid_argument("expert frequency");
    std::vector<uint32_t> ids(freq.size()); std::iota(ids.begin(),ids.end(),0u);
    std::stable_sort(ids.begin(),ids.end(),[&](auto a,auto b){return freq[a]<freq[b];});
    ids.resize(static_cast<size_t>(std::floor(fraction*freq.size()))); return ids;
}
} // namespace vegajet::offload

namespace vegajet::offload {
namespace {
TableMatrix valid_table(TableMatrix l) {
    auto fits=[&](size_t offset,size_t stride) {
        return offset<=l.row_bytes && stride && l.rows<=(l.row_bytes-offset)/stride;
    };
    if (!l.rows || !l.cols || l.cols%32 || !fits(l.weights_offset,l.weight_stride) ||
        l.weight_stride<size_t(l.cols/32)*(l.format==Format::Q4_0?18:16) ||
        (l.format!=Format::Q4_0 && (!fits(l.scales_offset,l.scale_stride)||l.scale_stride<size_t(l.cols/32)*4)))
        throw std::invalid_argument("expert table matrix bounds");
    return l;
}
}
TableExecutor::TableExecutor(const std::string& path,uint64_t experts,TableMatrix layout,Switches switches)
    :layout_(valid_table(layout)),table_(path,layout_.row_bytes,experts,1,switches.engram),executor_(switches) {}
bool TableExecutor::miss(uint64_t expert,Activation x,float* y) {
    if(!table_.submit({expert})) return false;
    while(!table_.poll()) std::this_thread::yield();
    if(!table_.success()) return false;
    auto* row=static_cast<const uint8_t*>(table_.data(0));
    Matrix m{row+layout_.weights_offset,layout_.format==Format::Q4_0?nullptr:row+layout_.scales_offset,
             layout_.rows,layout_.cols,layout_.weight_stride,layout_.scale_stride,layout_.format};
    // CPU consumes the row immediately; CUDA copies each expert at session creation.
    return executor_.miss(expert,m,x,y);
}
}
