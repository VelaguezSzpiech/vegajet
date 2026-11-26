// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/offload/cuda.hpp"
#include "core/offload/cuda_wire.hpp"
#include <dlfcn.h>
#include <cstring>
#include <stdexcept>
namespace vegajet::offload {
struct CudaTier::Impl {
    std::string why;
    HostArena arena{huge_bytes};
    void* rt=nullptr; void* drv=nullptr; void* weights=nullptr; void* scales=nullptr;
    void* module=nullptr; void* stream=nullptr; CudaRing* mapped=nullptr;
    Matrix matrix; unsigned producer=0,consumer=0,inflight=0;
    bool registered=false, launched=false;
    int (*unregister_host)(void*)=nullptr;
    int (*free_device)(void*)=nullptr;
    int (*sync_stream)(void*)=nullptr;
    int (*query_stream)(void*)=nullptr;
    int (*destroy_stream)(void*)=nullptr;
    int (*unload)(void*)=nullptr;
    template<class F> F symbol(void* lib,const char* name) {
        auto p=dlsym(lib,name); if(!p) throw std::runtime_error(name); return reinterpret_cast<F>(p);
    }
    void check(int result,const char* name) { if(result) throw std::runtime_error(std::string(name)+": "+std::to_string(result)); }
    explicit Impl(Matrix m,int device,uint32_t budget_ms):matrix(m) {
#ifndef VJ_OFFLOAD_PTX
        (void)device; (void)budget_ms; why="CUDA offload not built";
#else
        try {
            if(!m.valid() || m.rows>cuda_rows || m.cols>cuda_cols) throw std::runtime_error("CUDA matrix shape");
            rt=dlopen("libcudart.so",RTLD_NOW|RTLD_LOCAL);
            if(!rt) rt=dlopen("/usr/local/cuda/lib64/libcudart.so",RTLD_NOW|RTLD_LOCAL);
            drv=dlopen("libcuda.so.1",RTLD_NOW|RTLD_LOCAL);
            if(!rt || !drv) throw std::runtime_error("CUDA runtime/driver unavailable");
            unregister_host=symbol<decltype(unregister_host)>(rt,"cudaHostUnregister");
            free_device=symbol<decltype(free_device)>(rt,"cudaFree");
            sync_stream=symbol<decltype(sync_stream)>(rt,"cudaStreamSynchronize");
            query_stream=symbol<decltype(query_stream)>(rt,"cudaStreamQuery");
            destroy_stream=symbol<decltype(destroy_stream)>(rt,"cudaStreamDestroy");
            unload=symbol<decltype(unload)>(drv,"cuModuleUnload");
            check(symbol<int(*)(int)>(rt,"cudaSetDevice")(device),"cudaSetDevice");
            check(symbol<int(*)(void*)>(rt,"cudaFree")(nullptr),"CUDA context");
            auto alloc=symbol<int(*)(void**,size_t)>(rt,"cudaMalloc");
            auto copy=symbol<int(*)(void*,const void*,size_t,int)>(rt,"cudaMemcpy");
            check(alloc(&weights,m.rows*m.weight_stride),"cudaMalloc weights");
            // Embedded scale pointers can start inside a row. Copy only the live
            // portion of the final row, not padding beyond the caller's matrix.
            const size_t weight_bytes=(m.rows-1u)*m.weight_stride+size_t(m.cols/32u)*(m.format==Format::Q4_0?18u:16u);
            check(copy(weights,m.weights,weight_bytes,1),"cudaMemcpy weights");
            if(m.format!=Format::Q4_0) {
                check(alloc(&scales,m.rows*m.scale_stride),"cudaMalloc scales");
                const size_t scale_bytes=(m.rows-1u)*m.scale_stride+size_t(m.cols/32u)*4u;
                check(copy(scales,m.scales,scale_bytes,1),"cudaMemcpy scales");
            }
            std::memset(arena.data(),0,arena.size());
            check(symbol<int(*)(void*,size_t,unsigned)>(rt,"cudaHostRegister")(arena.data(),arena.size(),2),"cudaHostRegisterMapped"); registered=true;
            check(symbol<int(*)(void**,void*,unsigned)>(rt,"cudaHostGetDevicePointer")(reinterpret_cast<void**>(&mapped),arena.data(),0),"cudaHostGetDevicePointer");
            check(symbol<int(*)(void**,const char*)>(drv,"cuModuleLoad")(&module,VJ_OFFLOAD_PTX),"cuModuleLoad");
            void* fn=nullptr;
            check(symbol<int(*)(void**,void*,const char*)>(drv,"cuModuleGetFunction")(&fn,module,"vj_offload_worker"),"cuModuleGetFunction");
            check(symbol<int(*)(void**,unsigned)>(rt,"cudaStreamCreateWithFlags")(&stream,1),"cudaStreamCreate");
            int clock=0; check(symbol<int(*)(int*,int,int)>(rt,"cudaDeviceGetAttribute")(&clock,13,device),"clock rate");
            uint64_t ticks=uint64_t(clock)*budget_ms; // kHz * milliseconds; zero means no expiry
            uint64_t ws=m.weight_stride,ss=m.scale_stride; unsigned format=m.format==Format::Q4_0?0:1;
            void* args[]={&mapped,&weights,&scales,&m.rows,&m.cols,&ws,&ss,&format,&ticks};
            check(symbol<int(*)(void*,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,void*,void**,void**)>(drv,"cuLaunchKernel")
                (fn,1,1,1,256,1,1,0,stream,args,nullptr),"cuLaunchKernel"); launched=true;
        } catch(const std::exception& e) { why=e.what(); }
#endif
    }
    ~Impl() {
        if(launched) { __atomic_store_n(&static_cast<CudaRing*>(arena.data())->stop,1u,__ATOMIC_RELEASE); sync_stream(stream); }
        if(stream && destroy_stream) destroy_stream(stream);
        if(module && unload) unload(module);
        if(registered && unregister_host) unregister_host(arena.data());
        if(scales && free_device) free_device(scales);
        if(weights && free_device) free_device(weights);
        if(drv) dlclose(drv);
        if(rt) dlclose(rt);
    }
};
CudaTier::CudaTier(Matrix m,int device,uint32_t budget_ms):p_(std::make_unique<Impl>(m,device,budget_ms)) {}
CudaTier::~CudaTier()=default;
bool CudaTier::ok() const { return p_->why.empty(); }
const std::string& CudaTier::error() const { return p_->why; }
bool CudaTier::submit(Activation x) {
    if(!ok() || !x.values || !x.scales || p_->inflight==cuda_slots) return false;
    const int status=p_->query_stream(p_->stream);
    if(status!=600) { p_->why=status==0?"CUDA worker session expired":"CUDA stream error: "+std::to_string(status); return false; }
    auto& c=static_cast<CudaRing*>(p_->arena.data())->commands[p_->producer%cuda_slots];
    if(__atomic_load_n(&c.state,__ATOMIC_ACQUIRE)!=0) return false;
    std::memcpy(c.x,x.values,p_->matrix.cols); std::memcpy(c.scales,x.scales,p_->matrix.cols/32*sizeof(float));
    __atomic_store_n(&c.state,1u,__ATOMIC_RELEASE); ++p_->producer; ++p_->inflight; return true;
}
bool CudaTier::poll(float* y) {
    if(!ok() || !p_->inflight || !y) return false;
    auto& c=static_cast<CudaRing*>(p_->arena.data())->commands[p_->consumer%cuda_slots];
    if(__atomic_load_n(&c.state,__ATOMIC_ACQUIRE)!=2) {
        const int status=p_->query_stream(p_->stream);
        if(status!=600) p_->why=status==0?"CUDA worker session expired":"CUDA stream error: "+std::to_string(status);
        return false;
    }
    std::memcpy(y,c.y,p_->matrix.rows*sizeof(float)); __atomic_store_n(&c.state,0u,__ATOMIC_RELEASE);
    ++p_->consumer; --p_->inflight; return true;
}
}
