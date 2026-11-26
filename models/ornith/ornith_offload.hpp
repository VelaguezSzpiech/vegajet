// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include "core/offload/executor.hpp"
#include "core/q4_dot8_repack.hpp"
#include <array>
#include <cstring>
#include <stdexcept>
namespace vegajet::ornith {
// Experimental routed Down offload. The resident prefix plus one zero expert is
// compact; cold matrices remain on host and are evaluated by the generic tier.
class DownOffload {
    struct Layer {
        std::array<int32_t,256> map{};
        std::array<int32_t,256> cold{};
        std::vector<uint8_t> weights, scales;
        uint32_t rows=0,cols=0,wstride=0,sstride=0,offset=0;
        bool embedded=false;
    };
    double fraction_;
    std::map<uint64_t,Layer> layers_;
    offload::ExpertExecutor executor_;
public:
    explicit DownOffload(double fraction):fraction_(fraction),executor_(offload::Switches::environment()) {}
    void compact(uint32_t rank,uint32_t layer,std::vector<uint8_t>& w,std::vector<uint8_t>& scales,
                 Dot8RowLayout& layout,uint32_t dim) {
        Layer l;
        auto cold=offload::cold_experts(std::vector<double>(256,0),fraction_);
        l.cold.fill(-1);
        for(size_t i=0;i<cold.size();++i) l.cold[cold[i]]=static_cast<int32_t>(i);
        uint32_t hot=0;
        for(uint32_t i=0;i<256;++i) if(l.cold[i]<0) l.map[i]=hot++;
        for(uint32_t i:cold) l.map[i]=hot; // shared all-zero expert
        l.rows=dim; l.cols=layout.cols; l.wstride=layout.w_row_stride_bytes;
        l.sstride=layout.s_row_stride_bytes; l.embedded=layout.scale_embedded; l.offset=layout.s_offset_bytes;
        l.weights.resize(cold.size()*dim*l.wstride);
        if(!l.embedded) l.scales.resize(cold.size()*dim*l.sstride);
        std::vector<uint8_t> compact_w(uint64_t(hot+1)*dim*l.wstride,0);
        std::vector<uint8_t> compact_s(l.embedded?0:uint64_t(hot+1)*dim*l.sstride,0);
        for(uint32_t e=0;e<256;++e) {
            auto* dst=l.cold[e]>=0?l.weights.data()+uint64_t(l.cold[e])*dim*l.wstride:compact_w.data()+uint64_t(l.map[e])*dim*l.wstride;
            std::memcpy(dst,w.data()+uint64_t(e)*dim*l.wstride,uint64_t(dim)*l.wstride);
            if(!l.embedded) {
                auto* sd=l.cold[e]>=0?l.scales.data()+uint64_t(l.cold[e])*dim*l.sstride:compact_s.data()+uint64_t(l.map[e])*dim*l.sstride;
                std::memcpy(sd,scales.data()+uint64_t(e)*dim*l.sstride,uint64_t(dim)*l.sstride);
            }
        }
        layout.rows=uint64_t(hot+1)*dim;
        w.swap(compact_w); scales.swap(compact_s);
        layers_[uint64_t(rank)*48+layer]=std::move(l);
    }
    // read/write copy device buffers; called only after the GPU queue is drained.
    template<class Read,class Write>
    void prepare(uint32_t rank,uint32_t layer,uint32_t rows,uint64_t ids_va,uint64_t q_va,uint64_t ds_va,uint64_t tw_va,
                 Read read,Write write,std::vector<float>& correction,std::vector<int32_t>& original) {
        auto& l=layers_.at(uint64_t(rank)*48+layer);
        original.resize(rows*8u); read(original.data(),ids_va,original.size()*4u);
        std::vector<int32_t> mapped(original);
        bool miss=false;
        for(size_t i=0;i<mapped.size();++i) { miss |= l.cold.at(original[i])>=0; mapped[i]=l.map.at(original[i]); }
        correction.assign(uint64_t(rows)*l.rows,0);
        if(miss) {
            std::vector<int8_t> q(uint64_t(rows)*8u*l.cols);
            std::vector<float> ds(uint64_t(rows)*8u*(l.cols/32u)), weights(rows*8u), y(l.rows), xscale(l.cols/32u);
            read(q.data(),q_va,q.size()); read(ds.data(),ds_va,ds.size()*4u); read(weights.data(),tw_va,weights.size()*4u);
            for(size_t i=0;i<original.size();++i) if(l.cold[original[i]]>=0) {
                auto cold=uint64_t(l.cold[original[i]]);
                auto* w=l.weights.data()+cold*l.rows*l.wstride;
                auto* s=l.embedded?w+l.offset:l.scales.data()+cold*l.rows*l.sstride;
                offload::Matrix m{w,s,l.rows,l.cols,l.wstride,l.sstride,offload::Format::Dot8Iq3Sub32};
                for(uint32_t b=0;b<l.cols/32u;++b) xscale[b]=ds[i*(l.cols/32u)+b];
                if(!executor_.miss((uint64_t(rank)*48+layer)*256+original[i],m,{q.data()+i*l.cols,xscale.data()},y.data()))
                    throw std::runtime_error("expert miss: enable VEGAJET_OFFLOAD_L3_CPU=1 or VEGAJET_OFFLOAD_L2_CUDA=1");
                for(uint32_t d=0;d<l.rows;++d) correction[(i/8u)*l.rows+d]+=weights[i]*y[d];
            }
        }
        write(ids_va,mapped.data(),mapped.size()*4u);
    }
    void end_step() { executor_.end_step(); }
};
} // namespace vegajet::ornith
