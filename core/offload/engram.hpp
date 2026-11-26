// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include "core/offload/host.hpp"
#include "core/async_table_prefetch.hpp"
#include <string>
namespace vegajet::offload {
// Single decode-thread owner. One outstanding batch. The backend borrows its buffers.
class Engram {
    HostArena staging_;
    int fd_=-1, buffered_fd_=-1;
    std::vector<uint64_t> ids_;
    uint64_t rows_; size_t row_bytes_, capacity_, pending_=0;
    bool direct_=false, success_=true;
    std::unique_ptr<prefetch::IoBackend> io_;
public:
    Engram(const std::string& path, size_t row_bytes, uint64_t rows, size_t capacity, bool uring);
    ~Engram();
    bool submit(const std::vector<uint64_t>& ids);
    bool poll(); // true when batch complete; data(i) is then valid until next submit
    bool success() const { return success_; }
    const void* data(size_t i) const;
    bool pinned() const { return staging_.pinned(); }
    bool direct() const { return direct_; }
    bool registered() const;
    const char* backend() const { return io_->name(); }
};
}
