// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/async_table_prefetch.hpp"

#if defined(__linux__)
#include <linux/io_uring.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

#ifdef VJ_OFFLOAD_LIBURING
#include <liburing.h>
#endif

namespace vegajet::prefetch {

uint32_t row_bytes_for(RowFormat f, uint32_t elems) {
    if (elems == 0) return 0;
    switch (f) {
        case RowFormat::F16: return 2 * elems;
        case RowFormat::FP8: return 2 + elems;
        case RowFormat::Q4: return elems % 32 ? 0 : 18 * (elems / 32);
    }
    return 0;
}

float e4m3_to_f32(uint8_t b) {
    const int e = (b >> 3) & 0xF, m = b & 7;
    if (e == 15 && m == 7) return std::numeric_limits<float>::quiet_NaN();
    const float v = e == 0 ? std::ldexp(static_cast<float>(m), -9) : std::ldexp(static_cast<float>(8 + m), e - 10);
    return (b & 0x80) ? -v : v;
}

float f16_to_f32(uint16_t h) {
    const int e = (h >> 10) & 31, m = h & 1023;
    float v;
    if (e == 0) v = std::ldexp(static_cast<float>(m), -24);
    else if (e == 31) v = m ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
    else v = std::ldexp(static_cast<float>(m + 1024), e - 25);
    return (h & 0x8000) ? -v : v;
}

namespace {
inline uint16_t rd16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }
}

bool decode_row(RowFormat f, const uint8_t* row, uint32_t elems, float* out) {
    if (row_bytes_for(f, elems) == 0) return false;
    switch (f) {
        case RowFormat::F16:
            for (uint32_t i = 0; i < elems; ++i) out[i] = f16_to_f32(rd16(row + 2 * i));
            return true;
        case RowFormat::FP8: {
            const float s = f16_to_f32(rd16(row));
            for (uint32_t i = 0; i < elems; ++i) out[i] = s * e4m3_to_f32(row[2 + i]);
            return true;
        }
        case RowFormat::Q4:
            for (uint32_t b = 0; b < elems / 32; ++b) {
                const uint8_t* blk = row + 18 * b;
                const float s = f16_to_f32(rd16(blk));
                for (uint32_t j = 0; j < 16; ++j) {  // ggml Q4_0
                    out[32 * b + j] = s * static_cast<float>((blk[2 + j] & 0xF) - 8);
                    out[32 * b + 16 + j] = s * static_cast<float>((blk[2 + j] >> 4) - 8);
                }
            }
            return true;
    }
    return false;
}

// ------------------------------------------------------------------------------------------------ backends
int64_t preadv_full(const ReadOp& op) {
    std::vector<iovec> iov = op.iov;
    uint64_t total = 0;
    for (const auto& v : iov) total += v.iov_len;
    uint64_t done = 0;
    size_t first = 0;
    while (done < total) {
        const int cnt = static_cast<int>(std::min<size_t>(iov.size() - first, IOV_MAX));
        const ssize_t n = ::preadv(op.fd, iov.data() + first, cnt, static_cast<off_t>(op.offset + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            return -errno;
        }
        if (n == 0) break;  // end of file: short read
        done += static_cast<uint64_t>(n);
        size_t left = static_cast<size_t>(n);
        while (left && first < iov.size()) {  // advance the scatter list past what was filled
            if (left >= iov[first].iov_len) { left -= iov[first].iov_len; ++first; }
            else {
                iov[first].iov_base = static_cast<uint8_t*>(iov[first].iov_base) + left;
                iov[first].iov_len -= left;
                left = 0;
            }
        }
    }
    return static_cast<int64_t>(done);
}

void SyncBackend::submit(std::vector<ReadOp> ops) {
    for (const auto& op : ops) done_.push_back({op.user_data, preadv_full(op)});
}

size_t SyncBackend::reap(std::vector<Completion>& out, int) {
    const size_t n = done_.size();
    out.insert(out.end(), done_.begin(), done_.end());
    done_.clear();
    return n;
}

PreadvBackend::PreadvBackend(uint32_t threads) {
    for (uint32_t i = 0; i < std::max<uint32_t>(1, threads); ++i) threads_.emplace_back([this] { loop(); });
}

PreadvBackend::~PreadvBackend() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    work_cv_.notify_all();
    for (auto& t : threads_) t.join();
}

void PreadvBackend::loop() {
    for (;;) {
        ReadOp op;
        {
            std::unique_lock<std::mutex> lk(mu_);
            work_cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) return;  // stop requested and nothing left to read
            op = std::move(queue_.front());
            queue_.pop_front();
        }
        const int64_t r = preadv_full(op);
        {
            std::lock_guard<std::mutex> lk(mu_);
            done_.push_back({op.user_data, r});
        }
        done_cv_.notify_all();
    }
}

void PreadvBackend::submit(std::vector<ReadOp> ops) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto& op : ops) queue_.push_back(std::move(op));
    }
    work_cv_.notify_all();
}

size_t PreadvBackend::reap(std::vector<Completion>& out, int wait_ms) {
    std::unique_lock<std::mutex> lk(mu_);
    if (done_.empty() && wait_ms > 0) done_cv_.wait_for(lk, std::chrono::milliseconds(wait_ms), [&] { return !done_.empty(); });
    const size_t n = done_.size();
    out.insert(out.end(), done_.begin(), done_.end());
    done_.clear();
    return n;
}

// ------------------------------------------------------------------------------------------------ io_uring
struct IoUringBackend::Pending {
    uint64_t user_data = 0;
    int fd = -1;
    uint64_t offset = 0;
    std::vector<iovec> iov;      // remaining scatter list (advanced past filled bytes)
    size_t first = 0;
    uint64_t total = 0, done = 0;
    std::vector<iovec> sqe_iov;  // the slice of the outstanding SQE; kept alive until its CQE
};

#if defined(__linux__) && defined(__NR_io_uring_setup) && defined(__NR_io_uring_enter)
#define VJ_HAVE_IO_URING 1
struct IoUringBackend::Ring {
    int fd = -1;
    bool registered = false;
    void* sq_ptr = MAP_FAILED;
    size_t sq_size = 0;
    void* cq_ptr = MAP_FAILED;
    size_t cq_size = 0;
    io_uring_sqe* sqes = nullptr;
    size_t sqes_size = 0;
    uint32_t *sq_head = nullptr, *sq_tail = nullptr, *sq_array = nullptr;
    uint32_t sq_mask = 0, sq_entries = 0;
    uint32_t *cq_head = nullptr, *cq_tail = nullptr;
    uint32_t cq_mask = 0, cq_entries = 0;
    io_uring_cqe* cqes = nullptr;
    ~Ring() {
        if (sqes) munmap(sqes, sqes_size);
        if (cq_ptr != MAP_FAILED && cq_ptr != sq_ptr) munmap(cq_ptr, cq_size);
        if (sq_ptr != MAP_FAILED) munmap(sq_ptr, sq_size);
        if (fd >= 0) close(fd);
    }
};
#else
struct IoUringBackend::Ring {
    uint32_t sq_entries = 0;
};
#endif

namespace {
#ifdef VJ_OFFLOAD_LIBURING
int syscall_result(int r) { if(r<0) { errno=-r; return -1; } return r; }
#endif
long setup_ring(unsigned entries, io_uring_params* p) {
#ifdef VJ_OFFLOAD_LIBURING
    return syscall_result(io_uring_setup(entries,p));
#else
    return syscall(__NR_io_uring_setup,entries,p);
#endif
}
long register_ring(int fd, unsigned op, const void* arg, unsigned n) {
#ifdef VJ_OFFLOAD_LIBURING
    return syscall_result(io_uring_register(fd,op,arg,n));
#else
    return syscall(__NR_io_uring_register,fd,op,arg,n);
#endif
}
long enter_ring(int fd,unsigned count) {
#ifdef VJ_OFFLOAD_LIBURING
    return syscall_result(io_uring_enter(fd,count,0,0,nullptr));
#else
    return syscall(__NR_io_uring_enter,fd,count,0u,0u,nullptr,0);
#endif
}
template <class T> inline T load_acquire(const T* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
template <class T> inline void store_release(T* p, T v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
}  // namespace

IoUringBackend::IoUringBackend(IoUringOptions opt) : opt_(opt) {
#ifdef VJ_HAVE_IO_URING
    if (opt_.simulate_unavailable) { error_ = "io_uring_setup: " + std::string(std::strerror(ENOSYS)) + " (simulated)"; return; }
    io_uring_params p{};
    p.flags = IORING_SETUP_CLAMP | IORING_SETUP_CQSIZE;
    p.cq_entries = 2 * std::max<uint32_t>(1, opt_.entries);
    auto r = std::make_unique<Ring>();
    r->fd = static_cast<int>(setup_ring(std::max<uint32_t>(1, opt_.entries), &p));
    if (r->fd < 0) { error_ = std::string("io_uring_setup: ") + std::strerror(errno); return; }
    r->sq_size = p.sq_off.array + p.sq_entries * sizeof(uint32_t);
    r->cq_size = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
    const bool single = (p.features & IORING_FEAT_SINGLE_MMAP) != 0;
    if (single) r->sq_size = r->cq_size = std::max(r->sq_size, r->cq_size);
    r->sq_ptr = mmap(nullptr, r->sq_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, r->fd, IORING_OFF_SQ_RING);
    if (r->sq_ptr == MAP_FAILED) { error_ = std::string("mmap SQ ring: ") + std::strerror(errno); return; }
    r->cq_ptr = single ? r->sq_ptr
                       : mmap(nullptr, r->cq_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, r->fd, IORING_OFF_CQ_RING);
    if (r->cq_ptr == MAP_FAILED) { error_ = std::string("mmap CQ ring: ") + std::strerror(errno); return; }
    r->sqes_size = p.sq_entries * sizeof(io_uring_sqe);
    void* sq = mmap(nullptr, r->sqes_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, r->fd, IORING_OFF_SQES);
    if (sq == MAP_FAILED) { error_ = std::string("mmap SQEs: ") + std::strerror(errno); return; }
    r->sqes = static_cast<io_uring_sqe*>(sq);
    auto* sb = static_cast<uint8_t*>(r->sq_ptr);
    auto* cb = static_cast<uint8_t*>(r->cq_ptr);
    r->sq_head = reinterpret_cast<uint32_t*>(sb + p.sq_off.head);
    r->sq_tail = reinterpret_cast<uint32_t*>(sb + p.sq_off.tail);
    r->sq_mask = *reinterpret_cast<uint32_t*>(sb + p.sq_off.ring_mask);
    r->sq_entries = *reinterpret_cast<uint32_t*>(sb + p.sq_off.ring_entries);
    r->sq_array = reinterpret_cast<uint32_t*>(sb + p.sq_off.array);
    r->cq_head = reinterpret_cast<uint32_t*>(cb + p.cq_off.head);
    r->cq_tail = reinterpret_cast<uint32_t*>(cb + p.cq_off.tail);
    r->cq_mask = *reinterpret_cast<uint32_t*>(cb + p.cq_off.ring_mask);
    r->cq_entries = *reinterpret_cast<uint32_t*>(cb + p.cq_off.ring_entries);
    r->cqes = reinterpret_cast<io_uring_cqe*>(cb + p.cq_off.cqes);
    for (uint32_t i = 0; i < r->sq_entries; ++i) r->sq_array[i] = i;  // identity: SQE slot = ring index
    if (!opt_.registered_buffers.empty()) {
        r->registered = register_ring(r->fd, IORING_REGISTER_BUFFERS,
                                opt_.registered_buffers.data(), opt_.registered_buffers.size()) == 0;
    }
    ring_ = std::move(r);
#else
    error_ = "io_uring is not available on this platform";
#endif
}

IoUringBackend::~IoUringBackend() {
#ifdef VJ_HAVE_IO_URING
    if (!ring_) return;
    std::unique_lock<std::mutex> lk(mu_);
    backlog_.clear();  // never submitted: nobody can be waiting on their buffers
    while (!in_flight_.empty()) {  // the kernel may still write into these buffers: wait for every CQE
        pump();
        harvest();
        if (in_flight_.empty()) break;
        lk.unlock();
        pollfd pf{ring_->fd, POLLIN, 0};
        ::poll(&pf, 1, 50);
        lk.lock();
    }
#endif
}

bool IoUringBackend::ok(std::string* why) const {
    if (why) *why = error_;
    return error_.empty();
}

const char* IoUringBackend::name() const {
    std::lock_guard<std::mutex> lk(mu_);
    return st_.degraded || !ring_ ? "io_uring-degraded-preadv" : "io_uring";
}

IoUringStats IoUringBackend::stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    return st_;
}

bool IoUringBackend::buffers_registered() const {
#ifdef VJ_HAVE_IO_URING
    return ring_ && ring_->registered;
#else
    return false;
#endif
}
uint32_t IoUringBackend::sq_entries() const { return ring_ ? ring_->sq_entries : 0; }

void IoUringBackend::complete_inline(Pending& p) {
    ReadOp op;
    op.fd = p.fd;
    op.offset = p.offset + p.done;
    op.iov.assign(p.iov.begin() + static_cast<std::ptrdiff_t>(p.first), p.iov.end());
    const int64_t r = op.iov.empty() ? 0 : preadv_full(op);
    ++st_.inline_ops;
    done_.push_back({p.user_data, r < 0 ? r : static_cast<int64_t>(p.done) + r});
}

void IoUringBackend::submit(std::vector<ReadOp> ops) {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& op : ops) {
        auto p = std::make_unique<Pending>();
        p->user_data = op.user_data;
        p->fd = op.fd;
        p->offset = op.offset;
        p->iov = std::move(op.iov);
        for (const auto& v : p->iov) p->total += v.iov_len;
        ++st_.ops;
        if (!ring_ || st_.degraded) complete_inline(*p);
        else backlog_.push_back(std::move(p));
    }
    pump();
}

void IoUringBackend::degrade(int err) {
    st_.degraded = true;
    error_ = std::string("io_uring_enter: ") + std::strerror(err) + " (degraded to inline preadv)";
    // SQEs the kernel never took: run them inline. SQEs already taken still complete through the CQ.
    for (Pending* p : unsubmitted_) {
        for (auto it = in_flight_.begin(); it != in_flight_.end(); ++it) {
            if (it->second.get() != p) continue;
            complete_inline(*p);
            in_flight_.erase(it);
            break;
        }
    }
    unsubmitted_.clear();
    while (!backlog_.empty()) {
        complete_inline(*backlog_.front());
        backlog_.pop_front();
    }
}

void IoUringBackend::pump() {
#ifdef VJ_HAVE_IO_URING
    if (!ring_ || st_.degraded) return;
    Ring& r = *ring_;
    const size_t cap = std::min<size_t>(r.sq_entries, r.cq_entries);
    uint32_t tail = *r.sq_tail;  // only this thread (under mu_) writes the tail
    while (!backlog_.empty() && in_flight_.size() < cap && tail - load_acquire(r.sq_head) < r.sq_entries) {
        std::unique_ptr<Pending> p = std::move(backlog_.front());
        backlog_.pop_front();
        if (p->first >= p->iov.size()) {  // nothing to read (empty op)
            done_.push_back({p->user_data, static_cast<int64_t>(p->done)});
            continue;
        }
        // Slice: at most IOV_MAX iovecs and (test hook) at most max_bytes_per_sqe bytes.
        p->sqe_iov.clear();
        uint64_t budget = opt_.max_bytes_per_sqe ? opt_.max_bytes_per_sqe : UINT64_MAX;
        for (size_t i = p->first; i < p->iov.size() && p->sqe_iov.size() < IOV_MAX && budget > 0; ++i) {
            iovec v = p->iov[i];
            if (v.iov_len > budget) v.iov_len = static_cast<size_t>(budget);
            budget -= v.iov_len;
            p->sqe_iov.push_back(v);
        }
        const uint64_t tag = next_tag_++;
        io_uring_sqe* s = &r.sqes[tail & r.sq_mask];
        std::memset(s, 0, sizeof(*s));
        s->opcode = IORING_OP_READV;
        s->fd = p->fd;
        s->off = p->offset + p->done;
        s->addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p->sqe_iov.data()));
        s->len = static_cast<uint32_t>(p->sqe_iov.size());
        if (r.registered && p->sqe_iov.size() == 1) {
            const auto& v = p->sqe_iov[0];
            const auto addr = reinterpret_cast<uintptr_t>(v.iov_base);
            for (size_t i=0; i<opt_.registered_buffers.size() && i<=UINT16_MAX; ++i) {
                const auto& reg = opt_.registered_buffers[i];
                const auto base = reinterpret_cast<uintptr_t>(reg.iov_base);
                if (addr >= base && addr-base <= reg.iov_len && v.iov_len <= reg.iov_len-(addr-base)) {
                    s->opcode = IORING_OP_READ_FIXED;
                    s->addr = addr; s->len = static_cast<uint32_t>(v.iov_len);
                    s->buf_index = static_cast<uint16_t>(i); break;
                }
            }
        }
        s->user_data = tag;
        ++tail;
        ++st_.sqes;
        unsubmitted_.push_back(p.get());
        in_flight_.emplace(tag, std::move(p));
    }
    st_.max_in_flight = std::max<uint32_t>(st_.max_in_flight, static_cast<uint32_t>(in_flight_.size()));
    if (unsubmitted_.empty()) return;
    store_release(r.sq_tail, tail);
    for (int attempt = 0; attempt < 4 && !unsubmitted_.empty(); ++attempt) {
        ++st_.enter_calls;
        long ret;
        if (opt_.simulate_enter_errno) {
            errno = opt_.simulate_enter_errno;
            opt_.simulate_enter_errno = 0;
            ret = -1;
        } else {
            ret = enter_ring(r.fd, static_cast<unsigned>(unsubmitted_.size()));
        }
        if (ret >= 0) {
            const size_t took = std::min<size_t>(static_cast<size_t>(ret), unsubmitted_.size());
            unsubmitted_.erase(unsubmitted_.begin(), unsubmitted_.begin() + static_cast<std::ptrdiff_t>(took));
            if (took == 0) break;  // SQ busy (CQ backpressure): retried on the next pump
            continue;
        }
        const int e = errno;
        if (e == EINTR) continue;
        if (e == EAGAIN || e == EBUSY) break;  // transient: SQEs stay in the ring, the next pump resubmits
        degrade(e);
        return;
    }
#endif
}

void IoUringBackend::harvest() {
#ifdef VJ_HAVE_IO_URING
    if (!ring_) return;
    Ring& r = *ring_;
    uint32_t head = *r.cq_head;
    const uint32_t tail = load_acquire(r.cq_tail);
    std::vector<std::unique_ptr<Pending>> again;
    for (; head != tail; ++head) {
        const io_uring_cqe& c = r.cqes[head & r.cq_mask];
        ++st_.cqes;
        auto it = in_flight_.find(c.user_data);
        if (it == in_flight_.end()) continue;
        std::unique_ptr<Pending> p = std::move(it->second);
        in_flight_.erase(it);
        const int32_t res = c.res;
        if (res == -EINTR || res == -EAGAIN) { ++st_.retries; again.push_back(std::move(p)); continue; }
        if ((res == -EINVAL || res == -EOPNOTSUPP) && p->done == 0) { complete_inline(*p); continue; }
        if (res < 0) { done_.push_back({p->user_data, res}); continue; }
        if (res == 0) { done_.push_back({p->user_data, static_cast<int64_t>(p->done)}); continue; }  // end of file
        p->done += static_cast<uint64_t>(res);
        size_t left = static_cast<size_t>(res);
        while (left && p->first < p->iov.size()) {
            iovec& v = p->iov[p->first];
            if (left >= v.iov_len) { left -= v.iov_len; ++p->first; }
            else { v.iov_base = static_cast<uint8_t*>(v.iov_base) + left; v.iov_len -= left; left = 0; }
        }
        if (p->done < p->total) { ++st_.resubmits; again.push_back(std::move(p)); continue; }  // short read
        done_.push_back({p->user_data, static_cast<int64_t>(p->done)});
    }
    store_release(r.cq_head, head);
    for (auto it = again.rbegin(); it != again.rend(); ++it) backlog_.push_front(std::move(*it));
    if (!again.empty() && st_.degraded) {
        while (!backlog_.empty()) { complete_inline(*backlog_.front()); backlog_.pop_front(); }
    }
#endif
}

size_t IoUringBackend::reap(std::vector<Completion>& out, int wait_ms) {
    std::unique_lock<std::mutex> lk(mu_);
    harvest();
    pump();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(std::max(0, wait_ms));
    while (done_.empty() && wait_ms > 0 && ring_ && (!in_flight_.empty() || !backlog_.empty())) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) break;
        const int left = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count()) + 1;
#ifdef VJ_HAVE_IO_URING
        const int fd = ring_->fd;
        const bool stalled = !unsubmitted_.empty() || in_flight_.empty();  // nothing the kernel will signal for
        lk.unlock();
        pollfd pf{fd, POLLIN, 0};
        ::poll(&pf, 1, stalled ? 1 : std::min(left, 100));
        lk.lock();
#else
        (void)left;
#endif
        harvest();
        pump();
    }
    const size_t n = done_.size();
    out.insert(out.end(), done_.begin(), done_.end());
    done_.clear();
    return n;
}

std::unique_ptr<IoBackend> make_io_backend(BackendKind kind, uint32_t threads, std::string* why, IoUringOptions opt) {
    auto say = [&](const std::string& s) { if (why) *why = s; };
    if (kind == BackendKind::Sync) { say("sync-preadv"); return std::make_unique<SyncBackend>(); }
    if (kind == BackendKind::ThreadPool) { say("threadpool-preadv"); return std::make_unique<PreadvBackend>(threads); }
    const char* env = std::getenv("VJ_IO_URING");
    if (kind == BackendKind::Auto && env && std::strcmp(env, "0") == 0) {
        say("threadpool-preadv (io_uring disabled by VJ_IO_URING=0)");
        return std::make_unique<PreadvBackend>(threads);
    }
    auto u = std::make_unique<IoUringBackend>(opt);
    std::string err;
    if (u->ok(&err)) { say("io_uring"); return u; }
    say("threadpool-preadv (io_uring unavailable: " + err + ")");
    return std::make_unique<PreadvBackend>(threads);
}

// ------------------------------------------------------------------------------------------------ prefetcher
RowPrefetcher::RowPrefetcher(TableDesc table, PrefetchConfig cfg, std::unique_ptr<IoBackend> io)
    : t_(table), cfg_(cfg), io_(std::move(io)) {
    rb_ = t_.row_bytes();
    cfg_.max_run_rows = std::clamp<uint32_t>(cfg_.max_run_rows, 1, IOV_MAX);
    if (!io_) init_error_ = "no I/O backend";
    else if (t_.fd < 0) init_error_ = "table fd is not open";
    else if (rb_ == 0) init_error_ = "row_elems invalid for the row format";
    else if (t_.n_rows == 0 || t_.n_rows > UINT32_MAX) init_error_ = "n_rows must be in [1, 2^32)";
    else if (cfg_.slots == 0) init_error_ = "slots must be > 0";
    if (!init_error_.empty()) return;
    arena_.assign(size_t(cfg_.slots) * rb_, 0);
    slots_.resize(cfg_.slots);
    for (uint32_t s = cfg_.slots; s > 0; --s) free_.push_back(s - 1);
}

RowPrefetcher::~RowPrefetcher() {
    while (in_flight_rows_ > 0) drain(100);  // never free the arena under an outstanding read
}

bool RowPrefetcher::ok(std::string* err) const {
    if (err) *err = init_error_;
    return init_error_.empty();
}

int32_t RowPrefetcher::take_slot() {
    if (!free_.empty()) {
        const uint32_t s = free_.back();
        free_.pop_back();
        return static_cast<int32_t>(s);
    }
    if (lru_.empty()) return -1;
    const uint32_t s = lru_.front();
    lru_.pop_front();
    where_.erase(slots_[s].row);
    slots_[s].state = SlotState::Free;
    ++st_.evictions;
    return static_cast<int32_t>(s);
}

void RowPrefetcher::apply(const Completion& c) {
    auto it = runs_.find(c.user_data);
    if (it == runs_.end()) return;
    const std::vector<uint32_t> run = std::move(it->second);
    runs_.erase(it);
    for (size_t i = 0; i < run.size(); ++i) {
        Slot& sl = slots_[run[i]];
        const bool full = c.result >= 0 && static_cast<uint64_t>(c.result) >= (i + 1) * uint64_t(rb_);
        sl.state = full ? SlotState::Ready : SlotState::Error;
        sl.error = full ? 0 : (c.result < 0 ? c.result : -ENODATA);
        st_.errors += !full;
        lru_.push_back(run[i]);
        sl.lru = std::prev(lru_.end());
    }
    in_flight_rows_ -= run.size();
}

void RowPrefetcher::drain(int wait_ms) {
    std::vector<Completion> v;
    io_->reap(v, wait_ms);
    for (const auto& c : v) apply(c);
}

void RowPrefetcher::poll() {
    if (init_error_.empty()) drain(0);
}

size_t RowPrefetcher::prefetch(const uint32_t* rows, size_t n) {
    if (!init_error_.empty()) return 0;
    st_.requested += n;
    drain(0);
    std::vector<uint32_t> want;
    want.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const uint32_t r = rows[i];
        if (r >= t_.n_rows) continue;  // fetch() reports it
        auto it = where_.find(r);
        if (it != where_.end()) {
            ++st_.deduped;
            Slot& sl = slots_[it->second];
            if (sl.state != SlotState::Loading) lru_.splice(lru_.end(), lru_, sl.lru);  // about to be used
            continue;
        }
        want.push_back(r);
    }
    std::sort(want.begin(), want.end());
    const size_t before = want.size();
    want.erase(std::unique(want.begin(), want.end()), want.end());
    st_.deduped += before - want.size();

    std::vector<ReadOp> ops;
    size_t issued = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        const int32_t s = take_slot();
        if (s < 0) { st_.deferred += want.size() - i; break; }
        Slot& sl = slots_[static_cast<size_t>(s)];
        sl.state = SlotState::Loading;
        sl.row = want[i];
        where_[want[i]] = static_cast<uint32_t>(s);
        const bool extend = !ops.empty() && want[i] == want[i - 1] + 1 && ops.back().iov.size() < cfg_.max_run_rows;
        if (!extend) {
            ReadOp op;
            op.fd = t_.fd;
            op.offset = t_.file_offset + uint64_t(want[i]) * rb_;
            op.user_data = next_op_++;
            ops.push_back(std::move(op));
        }
        ops.back().iov.push_back({arena_.data() + size_t(s) * rb_, rb_});
        runs_[ops.back().user_data].push_back(static_cast<uint32_t>(s));
        ++issued;
    }
    if (!ops.empty()) {
        st_.read_ops += ops.size();
        st_.issued += issued;
        st_.bytes += uint64_t(issued) * rb_;
        in_flight_rows_ += issued;
        io_->submit(std::move(ops));
    }
    return issued;
}

bool RowPrefetcher::fetch(uint32_t row, uint8_t* dst, std::string* err) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    if (!init_error_.empty()) return fail(init_error_);
    if (row >= t_.n_rows) { ++st_.errors; return fail("row " + std::to_string(row) + " out of range"); }
    drain(0);
    auto it = where_.find(row);
    if (it != where_.end() && slots_[it->second].state == SlotState::Ready) {
        ++st_.hits;
    } else {
        ++st_.misses;
        if (it == where_.end()) {
            prefetch(&row, 1);
            it = where_.find(row);
            if (it == where_.end()) {  // every slot is being read: read this row directly into dst
                ReadOp op;
                op.fd = t_.fd;
                op.offset = t_.file_offset + uint64_t(row) * rb_;
                op.iov.push_back({dst, rb_});
                const int64_t r = preadv_full(op);
                ++st_.read_ops;
                if (r != static_cast<int64_t>(rb_)) { ++st_.errors; return fail(r < 0 ? std::strerror(static_cast<int>(-r)) : "short read"); }
                st_.bytes += rb_;
                return true;
            }
        }
        while (slots_[it->second].state == SlotState::Loading) drain(50);
    }
    const uint32_t s = it->second;
    Slot& sl = slots_[s];
    if (sl.state == SlotState::Error) {  // report once, then free the slot so a later fetch retries
        const int64_t e = sl.error;
        lru_.erase(sl.lru);
        where_.erase(it);
        sl.state = SlotState::Free;
        free_.push_back(s);
        return fail(e == -ENODATA ? "short read (row beyond end of file)" : std::strerror(static_cast<int>(-e)));
    }
    std::memcpy(dst, arena_.data() + size_t(s) * rb_, rb_);
    lru_.splice(lru_.end(), lru_, sl.lru);
    return true;
}

bool RowPrefetcher::fetch_f32(uint32_t row, float* dst, std::string* err) {
    std::vector<uint8_t> buf(rb_);
    return fetch(row, buf.data(), err) && decode_row(t_.fmt, buf.data(), t_.row_elems, dst);
}

uint64_t ngram_row(const int32_t* tokens, uint32_t n, uint64_t seed, uint64_t n_rows) {
    if (n_rows == 0) return 0;
    uint64_t h = seed ^ 0x9E3779B97F4A7C15ull ^ (uint64_t(n) << 56);
    for (uint32_t i = 0; i < n; ++i) {
        h ^= static_cast<uint32_t>(tokens[i]);
        h *= 0x100000001B3ull;
        h ^= h >> 29;
    }
    h ^= h >> 30;
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 27;
    h *= 0x94D049BB133111EBull;
    h ^= h >> 31;
    return h % n_rows;
}

} // namespace vegajet::prefetch
