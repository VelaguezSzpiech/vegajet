// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/offload/engram.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <limits>
namespace vegajet::offload {
namespace {
size_t staging_size(size_t bytes,size_t cap,uint64_t rows) {
    if(!bytes || bytes%4096 || !cap || cap>65535 || bytes>UINT32_MAX ||
       bytes>std::numeric_limits<size_t>::max()/cap || !rows || rows>uint64_t(INT64_MAX)/bytes)
        throw std::invalid_argument("engram needs 4096-byte aligned fixed rows and bounded capacity");
    return bytes*cap;
}
}
Engram::Engram(const std::string& path,size_t bytes,uint64_t rows,size_t cap,bool uring)
    :staging_(staging_size(bytes,cap,rows),-1,true),rows_(rows),row_bytes_(bytes),capacity_(cap) {
    fd_=open(path.c_str(),O_RDONLY|O_CLOEXEC|O_DIRECT);
    direct_=fd_>=0;
    if(fd_<0) fd_=open(path.c_str(),O_RDONLY|O_CLOEXEC);
    if(fd_<0) throw std::runtime_error("engram open");
    buffered_fd_=open(path.c_str(),O_RDONLY|O_CLOEXEC);
    if(buffered_fd_<0) { close(fd_); throw std::runtime_error("engram buffered open"); }
    try {
        if(uring) {
            prefetch::IoUringOptions opt; opt.entries=static_cast<uint32_t>(cap);
            for(size_t i=0;i<cap;++i) opt.registered_buffers.push_back({static_cast<uint8_t*>(staging_.data())+i*bytes,bytes});
            auto u=std::make_unique<prefetch::IoUringBackend>(opt);
            if(u->ok()) io_=std::move(u);
        }
        if(!io_) io_=std::make_unique<prefetch::PreadvBackend>(1);
    } catch(...) { close(buffered_fd_); close(fd_); throw; }
}
Engram::~Engram() { io_.reset(); if(fd_>=0) close(fd_); if(buffered_fd_>=0) close(buffered_fd_); }
bool Engram::registered() const { auto u=dynamic_cast<prefetch::IoUringBackend*>(io_.get()); return u && u->buffers_registered(); }
bool Engram::submit(const std::vector<uint64_t>& ids) {
    if(pending_ || ids.size()>capacity_) return false;
    for(auto id:ids) if(id>=rows_) return false;
    std::vector<prefetch::ReadOp> ops;
    for(size_t i=0;i<ids.size();++i) ops.push_back({fd_,ids[i]*row_bytes_,{{static_cast<uint8_t*>(staging_.data())+i*row_bytes_,row_bytes_}},i});
    ids_=ids; success_=true; pending_=ids.size(); io_->submit(std::move(ops)); return true;
}
bool Engram::poll() {
    std::vector<prefetch::Completion> done; io_->reap(done,0);
    for(auto c:done) {
        // Some filesystems accept O_DIRECT at open but reject reads. Retry via
        // the buffered descriptor; this also completes short direct-read tails.
        if(c.result != static_cast<int64_t>(row_bytes_)) {
            prefetch::ReadOp op{buffered_fd_,ids_.at(c.user_data)*row_bytes_,
                {{static_cast<uint8_t*>(staging_.data())+c.user_data*row_bytes_,row_bytes_}},0};
            c.result=prefetch::preadv_full(op);
        }
        success_ &= c.result==static_cast<int64_t>(row_bytes_); --pending_;
    }
    return pending_==0;
}
const void* Engram::data(size_t i) const {
    if(pending_ || i>=ids_.size()) throw std::out_of_range("engram slot");
    return static_cast<const uint8_t*>(staging_.data())+i*row_bytes_;
}
}
