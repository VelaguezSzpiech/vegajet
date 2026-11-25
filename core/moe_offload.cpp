// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/moe_offload.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>

#include <unistd.h>

#include <poll.h>
#include <sys/mman.h>

#if defined(__linux__)
#include <sys/syscall.h>
#endif

#if defined(__linux__) && defined(__has_include)
#if __has_include(<linux/io_uring.h>)
#include <linux/io_uring.h>
#if defined(__NR_io_uring_setup) && defined(__NR_io_uring_enter)
#define VJ_MO_IO_URING 1
#endif
#endif
#endif

namespace vegajet::moeoffload {

namespace {

// Full read of a single buffer: retries EINTR and short reads. Returns bytes read, or -errno.
int64_t pread_full(int fd, uint64_t offset, void* buf, size_t len) {
    size_t done = 0;
    while (done < len) {
        const ssize_t r = ::pread(fd, static_cast<uint8_t*>(buf) + done, len - done,
                                  static_cast<off_t>(offset + done));
        if (r < 0) {
            if (errno == EINTR) continue;
            return -errno;
        }
        if (r == 0) break; // end of file
        done += static_cast<size_t>(r);
    }
    return static_cast<int64_t>(done);
}

// Full vectored read that advances the caller's scatter list in place (the list is the engine's private copy).
int64_t readv_full(int fd, uint64_t offset, iovec* iov, uint32_t n) {
    uint64_t total = 0;
    for (uint32_t i = 0; i < n; ++i) total += iov[i].iov_len;
    uint64_t done = 0;
    uint32_t first = 0;
    while (done < total && first < n) {
        const int cnt = static_cast<int>(std::min<uint32_t>(n - first, static_cast<uint32_t>(IOV_MAX)));
        const ssize_t r = ::preadv(fd, iov + first, cnt, static_cast<off_t>(offset + done));
        if (r < 0) {
            if (errno == EINTR) continue;
            return -errno;
        }
        if (r == 0) break;
        uint64_t left = static_cast<uint64_t>(r);
        done += left;
        while (left && first < n) {
            if (left >= iov[first].iov_len) {
                left -= iov[first].iov_len;
                ++first;
            } else {
                iov[first].iov_base = static_cast<uint8_t*>(iov[first].iov_base) + left;
                iov[first].iov_len -= static_cast<size_t>(left);
                left = 0;
            }
        }
    }
    return static_cast<int64_t>(done);
}

template <class T> inline T load_acquire(const T* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
template <class T> inline void store_release(T* p, T v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }

} // namespace

// ------------------------------------------------------------------------------------------------ I/O engine
//
// Fixed-capacity asynchronous read engine. Every request slot, iovec and completion ring entry is allocated in the
// constructor; submit()/reap() never allocate. io_uring SQEs carry the request index as user_data, and an index is
// only recycled after its completion was reaped, so a stale CQE can never match a reused slot.
class MoeOffload::Engine {
public:
    Engine(IoKind kind, uint32_t entries, uint32_t max_inflight, uint32_t max_iov, uint32_t threads, std::string* why);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    bool usable() const { return error_.empty(); }
    std::string why() const { return error_; }
    const char* name() const { return name_.c_str(); }
    uint32_t capacity() const { return cap_; }
    size_t outstanding();

    // Copies fd/offset/iovecs into a free request slot. False if the batch does not fit (no partial submit).
    bool submit(const ReadRequest* ops, size_t n);
    // Moves finished completions into out; waits up to wait_ms for at least one when none is ready.
    size_t reap(IoCompletion* out, size_t cap, int wait_ms);

private:
    enum : uint8_t { StFree = 0, StQueued = 1, StInFlight = 2, StDone = 3 };
    struct Req {
        int fd = -1;
        uint64_t offset = 0, total = 0, done = 0;
        uint32_t first = 0, n = 0;
        uint32_t sqpos = 0; // ring position of the SQE that carries this request
        uint64_t user = 0;
        int64_t result = 0;
        uint8_t state = StFree;
    };
    struct Ring {
        std::vector<uint32_t> v;
        uint64_t h = 0, t = 0;
    };

    void init_common();
    bool setup_uring(uint32_t entries);
    void start_preadv(uint32_t threads);
    void worker();

    bool push(Ring& r, uint32_t idx);
    bool pop(Ring& r, uint32_t& idx);
    static bool empty(const Ring& r) { return r.t == r.h; }

    bool alloc_req(uint32_t& idx);
    void release_req(uint32_t idx);
    void fill_req(uint32_t idx, const ReadRequest& op);
    void run_inline(uint32_t idx);
    void pump_locked();
    void harvest_locked();
    void degrade_locked(int err);
    size_t drain_done(IoCompletion* out, size_t cap);
    size_t pending_locked() const { return held_ - (done_.t - done_.h); }

#if defined(VJ_MO_IO_URING)
    struct Uring {
        int fd = -1;
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
        bool ok = false;

        Uring() = default;
        Uring(const Uring&) = delete;
        Uring& operator=(const Uring&) = delete;
        ~Uring() {
            if (sqes) ::munmap(sqes, sqes_size);
            if (cq_ptr != MAP_FAILED && cq_ptr != sq_ptr) ::munmap(cq_ptr, cq_size);
            if (sq_ptr != MAP_FAILED) ::munmap(sq_ptr, sq_size);
            if (fd >= 0) ::close(fd);
        }
    };
    Uring ur_;
#else
    struct Uring {
        bool ok = false;
    };
    Uring ur_;
#endif

    bool uring_mode_ = false;
    bool degraded_ = false;
    IoKind kind_ = IoKind::Preadv;
    uint32_t cap_ = 1;
    uint32_t max_iov_ = 1;
    std::string error_;
    std::string name_ = "none";

    std::vector<Req> reqs_;
    std::vector<iovec> iovs_; // cap_ * max_iov_
    Ring free_, work_, done_;
    uint32_t held_ = 0;

    std::mutex mu_;
    std::condition_variable work_cv_, done_cv_;
    std::vector<std::thread> threads_;
    bool stop_ = false;
};

MoeOffload::Engine::Engine(IoKind kind, uint32_t entries, uint32_t max_inflight, uint32_t max_iov,
                           uint32_t threads, std::string* why) {
    kind_ = kind;
    max_iov_ = std::max<uint32_t>(1, std::min<uint32_t>(max_iov, static_cast<uint32_t>(IOV_MAX)));
    const uint32_t want_cap = std::max<uint32_t>(1, max_inflight);
    cap_ = want_cap;
#if defined(VJ_MO_IO_URING)
    if (kind_ != IoKind::Preadv) {
        if (setup_uring(std::max(entries, want_cap))) {
            uring_mode_ = true;
            cap_ = std::max<uint32_t>(1, std::min(want_cap, std::min(ur_.sq_entries, ur_.cq_entries)));
            init_common();
            name_ = "io_uring";
            if (why) *why = name_;
            return;
        }
        if (why) *why = "io_uring unavailable (" + error_ + "): threadpool-preadv";
    }
#endif
    (void)entries;
    error_.clear();
    uring_mode_ = false;
    cap_ = want_cap;
    init_common();
    name_ = "threadpool-preadv";
    start_preadv(threads);
    if (why) *why = name_;
}

void MoeOffload::Engine::init_common() {
    reqs_.assign(cap_, Req{});
    iovs_.assign(static_cast<size_t>(cap_) * max_iov_, iovec{nullptr, 0});
    free_.v.assign(cap_, 0);
    for (uint32_t i = 0; i < cap_; ++i) free_.v[i] = i;
    free_.h = 0;
    free_.t = cap_;
    work_.v.assign(cap_, 0);
    work_.h = work_.t = 0;
    done_.v.assign(cap_, 0);
    done_.h = done_.t = 0;
    held_ = 0;
}

#if defined(VJ_MO_IO_URING)
bool MoeOffload::Engine::setup_uring(uint32_t entries) {
    io_uring_params p{};
    p.flags = IORING_SETUP_CLAMP | IORING_SETUP_CQSIZE;
    p.cq_entries = 2 * std::max<uint32_t>(1, entries);
    const long fd = ::syscall(__NR_io_uring_setup, static_cast<unsigned>(std::max<uint32_t>(1, entries)), &p);
    if (fd < 0) {
        error_ = std::string("io_uring_setup: ") + std::strerror(errno);
        return false;
    }
    ur_.fd = static_cast<int>(fd);
    ur_.sq_size = p.sq_off.array + p.sq_entries * sizeof(uint32_t);
    ur_.cq_size = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
    const bool single = (p.features & IORING_FEAT_SINGLE_MMAP) != 0;
    if (single) {
        const size_t m = std::max(ur_.sq_size, ur_.cq_size);
        ur_.sq_size = ur_.cq_size = m;
    }
    ur_.sq_ptr = ::mmap(nullptr, ur_.sq_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ur_.fd,
                        IORING_OFF_SQ_RING);
    if (ur_.sq_ptr == MAP_FAILED) {
        error_ = std::string("mmap SQ ring: ") + std::strerror(errno);
        return false;
    }
    ur_.cq_ptr = single ? ur_.sq_ptr
                        : ::mmap(nullptr, ur_.cq_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ur_.fd,
                                 IORING_OFF_CQ_RING);
    if (ur_.cq_ptr == MAP_FAILED) {
        error_ = std::string("mmap CQ ring: ") + std::strerror(errno);
        return false;
    }
    ur_.sqes_size = static_cast<size_t>(p.sq_entries) * sizeof(io_uring_sqe);
    void* sq = ::mmap(nullptr, ur_.sqes_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ur_.fd,
                      IORING_OFF_SQES);
    if (sq == MAP_FAILED) {
        error_ = std::string("mmap SQEs: ") + std::strerror(errno);
        return false;
    }
    ur_.sqes = static_cast<io_uring_sqe*>(sq);
    uint8_t* sb = static_cast<uint8_t*>(ur_.sq_ptr);
    uint8_t* cb = static_cast<uint8_t*>(ur_.cq_ptr);
    ur_.sq_head = reinterpret_cast<uint32_t*>(sb + p.sq_off.head);
    ur_.sq_tail = reinterpret_cast<uint32_t*>(sb + p.sq_off.tail);
    ur_.sq_mask = *reinterpret_cast<uint32_t*>(sb + p.sq_off.ring_mask);
    ur_.sq_entries = *reinterpret_cast<uint32_t*>(sb + p.sq_off.ring_entries);
    ur_.sq_array = reinterpret_cast<uint32_t*>(sb + p.sq_off.array);
    ur_.cq_head = reinterpret_cast<uint32_t*>(cb + p.cq_off.head);
    ur_.cq_tail = reinterpret_cast<uint32_t*>(cb + p.cq_off.tail);
    ur_.cq_mask = *reinterpret_cast<uint32_t*>(cb + p.cq_off.ring_mask);
    ur_.cq_entries = *reinterpret_cast<uint32_t*>(cb + p.cq_off.ring_entries);
    ur_.cqes = reinterpret_cast<io_uring_cqe*>(cb + p.cq_off.cqes);
    for (uint32_t i = 0; i < ur_.sq_entries; ++i) ur_.sq_array[i] = i; // identity: SQE slot = ring index
    ur_.ok = true;
    return true;
}
#else
bool MoeOffload::Engine::setup_uring(uint32_t) {
    error_ = "io_uring is not available on this platform";
    return false;
}
#endif

void MoeOffload::Engine::start_preadv(uint32_t threads) {
    const uint32_t n = std::max<uint32_t>(1, threads);
    threads_.reserve(n);
    for (uint32_t i = 0; i < n; ++i) threads_.emplace_back([this] { worker(); });
}

MoeOffload::Engine::~Engine() {
    std::unique_lock<std::mutex> lk(mu_);
    if (!uring_mode_) {
        stop_ = true;
        work_cv_.notify_all();
        lk.unlock();
        for (auto& t : threads_) {
            if (t.joinable()) t.join();
        }
        return;
    }
#if defined(VJ_MO_IO_URING)
    // The kernel may still write into the arena of every SQE already submitted: reap them before unmapping.
    while (ur_.ok && pending_locked() > 0 && held_ > 0) {
        harvest_locked();
        pump_locked();
        if (pending_locked() == 0) break;
        lk.unlock();
        pollfd pf{ur_.fd, POLLIN, 0};
        ::poll(&pf, 1, 20);
        lk.lock();
    }
#else
    lk.unlock();
#endif
}

bool MoeOffload::Engine::push(Ring& r, uint32_t idx) {
    if (r.t - r.h >= cap_) return false;
    r.v[static_cast<size_t>(r.t % cap_)] = idx;
    ++r.t;
    return true;
}

bool MoeOffload::Engine::pop(Ring& r, uint32_t& idx) {
    if (r.t == r.h) return false;
    idx = r.v[static_cast<size_t>(r.h % cap_)];
    ++r.h;
    return true;
}

bool MoeOffload::Engine::alloc_req(uint32_t& idx) {
    if (free_.t == free_.h) return false;
    idx = free_.v[static_cast<size_t>(free_.h % cap_)];
    ++free_.h;
    ++held_;
    return true;
}

void MoeOffload::Engine::release_req(uint32_t idx) {
    free_.v[static_cast<size_t>(free_.t % cap_)] = idx;
    ++free_.t;
    --held_;
}

void MoeOffload::Engine::fill_req(uint32_t idx, const ReadRequest& op) {
    Req& r = reqs_[idx];
    r.fd = op.fd;
    r.offset = op.offset;
    r.total = 0;
    r.done = 0;
    r.first = 0;
    r.result = 0;
    r.user = op.user;
    r.n = std::min<uint32_t>(op.iov_count, max_iov_);
    iovec* dst = &iovs_[static_cast<size_t>(idx) * max_iov_];
    for (uint32_t i = 0; i < r.n; ++i) {
        dst[i] = op.iov[i];
        r.total += op.iov[i].iov_len;
    }
}

void MoeOffload::Engine::run_inline(uint32_t idx) {
    Req& r = reqs_[idx];
    const int64_t res = readv_full(r.fd, r.offset + r.done, &iovs_[static_cast<size_t>(idx) * max_iov_ + r.first],
                                   r.n - r.first);
    r.result = res < 0 ? res : static_cast<int64_t>(r.done + static_cast<uint64_t>(res));
    r.state = StDone;
    (void)push(done_, idx); // cannot fail: done_ has cap_ entries and at most cap_ live requests
}

void MoeOffload::Engine::degrade_locked(int err) {
#if defined(VJ_MO_IO_URING)
    degraded_ = true;
    error_ = std::string("io_uring_enter: ") + std::strerror(err) + " (degraded to inline preadv)";
    name_ = "io_uring-degraded-preadv";
    uint32_t idx = 0;
    while (pop(work_, idx)) run_inline(idx); // never written to the ring
    // SQEs beyond the kernel's head were published but never taken: no completion will ever arrive for them.
    const uint32_t consumed = load_acquire(ur_.sq_head);
    for (uint32_t i = 0; i < cap_; ++i) {
        Req& r = reqs_[i];
        if (r.state != StInFlight) continue;
        if (static_cast<int32_t>(r.sqpos - consumed) >= 0) run_inline(i);
    }
#else
    (void)err;
#endif
}

bool MoeOffload::Engine::submit(const ReadRequest* ops, size_t n) {
    if (n == 0) return true;
    std::lock_guard<std::mutex> lk(mu_);
    if (held_ + n > cap_) return false; // check before mutating: the caller keeps its slots on refusal
    if (!uring_mode_ || degraded_) {
        for (size_t i = 0; i < n; ++i) {
            uint32_t idx = 0;
            if (!alloc_req(idx)) return false;
            fill_req(idx, ops[i]);
            if (degraded_) {
                run_inline(idx);
            } else {
                reqs_[idx].state = StQueued;
                if (!push(work_, idx)) {
                    run_inline(idx);
                }
            }
        }
        if (!degraded_) work_cv_.notify_all();
        return true;
    }
    for (size_t i = 0; i < n; ++i) {
        uint32_t idx = 0;
        if (!alloc_req(idx)) return false;
        fill_req(idx, ops[i]);
        reqs_[idx].state = StQueued;
        if (!push(work_, idx)) run_inline(idx);
    }
    pump_locked();
    return true;
}

void MoeOffload::Engine::pump_locked() {
#if defined(VJ_MO_IO_URING)
    if (!uring_mode_ || degraded_ || !ur_.ok) return;
    uint32_t tail = *ur_.sq_tail;
    bool wrote = false;
    uint32_t idx = 0;
    while (pop(work_, idx)) {
        const uint32_t space = ur_.sq_entries - static_cast<uint32_t>(tail - load_acquire(ur_.sq_head));
        if (space == 0) {
            if (!push(work_, idx)) run_inline(idx);
            break;
        }
        Req& r = reqs_[idx];
        r.state = StInFlight;
        r.sqpos = tail;
        io_uring_sqe* s = &ur_.sqes[tail & ur_.sq_mask];
        std::memset(s, 0, sizeof(*s));
        s->opcode = IORING_OP_READV;
        s->fd = r.fd;
        s->off = r.offset + r.done;
        s->addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&iovs_[static_cast<size_t>(idx) * max_iov_ + r.first]));
        s->len = r.n - r.first;
        s->user_data = idx;
        ++tail;
        wrote = true;
    }
    if (wrote) store_release(ur_.sq_tail, tail);
    for (int attempt = 0; attempt < 16; ++attempt) {
        const uint32_t outstanding =
            static_cast<uint32_t>(tail - load_acquire(ur_.sq_head));
        if (outstanding == 0) return;
        const long ret = ::syscall(__NR_io_uring_enter, ur_.fd, outstanding, 0u, 0u, nullptr, 0);
        if (ret >= 0) return;
        const int e = errno;
        if (e == EINTR) continue;
        if (e == EAGAIN || e == EBUSY) return; // transient backpressure: retried on the next pump/reap
        degrade_locked(e);
        return;
    }
#endif
}

void MoeOffload::Engine::harvest_locked() {
#if defined(VJ_MO_IO_URING)
    if (!uring_mode_ || !ur_.ok) return;
    // Requeue a request for another SQE; when the ring degraded, finish it inline instead (pump is a no-op then).
    auto again = [this](uint32_t i) {
        if (degraded_) {
            run_inline(i);
            return;
        }
        reqs_[i].state = StQueued;
        if (!push(work_, i)) run_inline(i);
    };
    uint32_t head = *ur_.cq_head;
    const uint32_t tail = load_acquire(ur_.cq_tail);
    for (; head != tail; ++head) {
        const io_uring_cqe& c = ur_.cqes[head & ur_.cq_mask];
        const uint32_t idx = static_cast<uint32_t>(c.user_data);
        if (idx >= cap_ || reqs_[idx].state != StInFlight) continue;
        Req& r = reqs_[idx];
        const int32_t res = c.res;
        if (res == -EINTR || res == -EAGAIN) {
            again(idx);
            continue;
        }
        if ((res == -EINVAL || res == -EOPNOTSUPP) && r.done == 0) { // fd type without READV: read inline
            run_inline(idx);
            continue;
        }
        if (res < 0) {
            r.result = res;
            r.state = StDone;
            (void)push(done_, idx); // cannot fail: done_ has cap_ entries and there are at most cap_ requests
            continue;
        }
        if (res == 0) { // end of file
            r.result = static_cast<int64_t>(r.done);
            r.state = StDone;
            (void)push(done_, idx);
            continue;
        }
        r.done += static_cast<uint64_t>(res);
        uint64_t left = static_cast<uint64_t>(res);
        iovec* iv = &iovs_[static_cast<size_t>(idx) * max_iov_];
        while (left && r.first < r.n) {
            iovec& v = iv[r.first];
            if (left >= v.iov_len) {
                left -= v.iov_len;
                ++r.first;
            } else {
                v.iov_base = static_cast<uint8_t*>(v.iov_base) + left;
                v.iov_len -= static_cast<size_t>(left);
                left = 0;
            }
        }
        if (r.done < r.total) { // short read: resubmit the remainder under the same request slot
            again(idx);
            continue;
        }
        r.result = static_cast<int64_t>(r.done);
        r.state = StDone;
        (void)push(done_, idx);
    }
    store_release(ur_.cq_head, head);
#endif
}

size_t MoeOffload::Engine::drain_done(IoCompletion* out, size_t cap) {
    size_t n = 0;
    uint32_t idx = 0;
    while (n < cap && pop(done_, idx)) {
        out[n].user = reqs_[idx].user;
        out[n].result = reqs_[idx].result;
        reqs_[idx].state = StFree;
        release_req(idx);
        ++n;
    }
    return n;
}

size_t MoeOffload::Engine::outstanding() {
    std::lock_guard<std::mutex> lk(mu_);
    return pending_locked();
}

size_t MoeOffload::Engine::reap(IoCompletion* out, size_t cap, int wait_ms) {
    std::unique_lock<std::mutex> lk(mu_);
    size_t got = drain_done(out, cap);
    if (got > 0 || cap == 0) return got;

    if (uring_mode_ && !degraded_) pump_locked();
    if (uring_mode_) {
        harvest_locked();
        got = drain_done(out, cap);
        if (got > 0 || wait_ms <= 0) return got;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(wait_ms);
        while (pending_locked() > 0) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) break;
            const int left = static_cast<int>(
                                 std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count()) + 1;
#if defined(VJ_MO_IO_URING)
            if (ur_.ok) {
                const int fd = ur_.fd;
                lk.unlock();
                pollfd pf{fd, POLLIN, 0};
                ::poll(&pf, 1, std::min(left, 100));
                lk.lock();
            }
#else
            (void)left;
#endif
            harvest_locked();
            pump_locked();
            got = drain_done(out, cap);
            if (got > 0) return got;
        }
        return got;
    }

    if (wait_ms > 0 && empty(done_)) {
        done_cv_.wait_for(lk, std::chrono::milliseconds(wait_ms), [this] { return !empty(done_); });
    }
    return drain_done(out, cap);
}

void MoeOffload::Engine::worker() {
    for (;;) {
        uint32_t idx = 0;
        {
            std::unique_lock<std::mutex> lk(mu_);
            work_cv_.wait(lk, [this] { return stop_ || !empty(work_); });
            if (empty(work_) && stop_) return;
            if (!pop(work_, idx)) continue;
            reqs_[idx].state = StInFlight;
        }
        Req& r = reqs_[idx];
        const int64_t res =
            readv_full(r.fd, r.offset + r.done, &iovs_[static_cast<size_t>(idx) * max_iov_ + r.first], r.n - r.first);
        {
            std::lock_guard<std::mutex> lk(mu_);
            r.result = res;
            r.state = StDone;
            (void)push(done_, idx); // cannot fail: done_ has cap_ entries and at most cap_ live requests
            done_cv_.notify_one();
        }
    }
}

// ------------------------------------------------------------------------------------------------ MoeOffload
MoeOffload::MoeOffload(ExpertTable table, OffloadConfig cfg, std::string* why) : t_(table), cfg_(std::move(cfg)) {
    if (t_.fd < 0) init_error_ = "expert table fd is not open";
    else if (t_.n_layers == 0) init_error_ = "n_layers must be > 0";
    else if (t_.n_experts == 0) init_error_ = "n_experts must be > 0";
    else if (t_.expert_bytes == 0) init_error_ = "expert_bytes must be > 0";
    else if (cfg_.staging_slots == 0) init_error_ = "staging_slots must be > 0";
    else if (static_cast<uint64_t>(t_.n_layers) * t_.n_experts > 0xFFFFFFFFull)
        init_error_ = "n_layers * n_experts must fit in 2^32";
    else if (cfg_.cards.size() > 2) init_error_ = "at most two VRAM cards are supported";
    else if (!cfg_.hot_mask.empty() && cfg_.hot_mask.size() != static_cast<size_t>(t_.n_layers) * t_.n_experts)
        init_error_ = "hot_mask must have n_layers * n_experts entries";
    if (!init_error_.empty()) {
        if (why) *why = init_error_;
        return;
    }

    nkeys_ = static_cast<uint64_t>(t_.n_layers) * t_.n_experts;
    cfg_.max_run_rows = std::clamp<uint32_t>(cfg_.max_run_rows, 1, static_cast<uint32_t>(IOV_MAX));
    sanitize_cards();

    key_slot_.assign(static_cast<size_t>(nkeys_), -1);
    res_card_.assign(static_cast<size_t>(nkeys_), -1);
    slots_.resize(cfg_.staging_slots);
    free_.reserve(cfg_.staging_slots);
    for (uint32_t s = cfg_.staging_slots; s > 0; --s) free_.push_back(s - 1);
    want_.reserve(cfg_.staging_slots);
    batch_iov_.reserve(cfg_.staging_slots);
    batch_ops_.reserve(cfg_.staging_slots);

    if (!init_arena(why)) return;

    uint32_t req_cap = cfg_.max_in_flight ? std::min(cfg_.max_in_flight, cfg_.staging_slots) : cfg_.staging_slots;
    req_cap = std::max<uint32_t>(1, req_cap);
    const uint32_t entries = std::max(cfg_.io_entries, req_cap);
    eng_ = std::make_unique<Engine>(cfg_.io, entries, req_cap, cfg_.max_run_rows, cfg_.threads, why);
    if (!eng_->usable()) {
        init_error_ = eng_->why();
        if (why) *why = init_error_;
        return;
    }
    completions_.resize(eng_->capacity());
    if (n_cards_ > 0) replan();
    if (why) *why = eng_->name();
}

bool MoeOffload::init_arena(std::string* why) {
    const long pg = ::sysconf(_SC_PAGESIZE);
    if (pg <= 0) {
        init_error_ = "sysconf(_SC_PAGESIZE) failed";
        if (why) *why = init_error_;
        return false;
    }
    const size_t page = static_cast<size_t>(pg);
    const size_t bytes = t_.expert_bytes;
    slot_stride_ = (bytes + page - 1) / page * page;
    if (slot_stride_ / page > SIZE_MAX / cfg_.staging_slots) {
        init_error_ = "staging arena is too large";
        if (why) *why = init_error_;
        return false;
    }
    arena_bytes_ = slot_stride_ * cfg_.staging_slots;
    void* p = ::mmap(nullptr, arena_bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        init_error_ = std::string("mmap staging arena: ") + std::strerror(errno);
        if (why) *why = init_error_;
        return false;
    }
    arena_ = static_cast<uint8_t*>(p);
    pin_requested_ = true;
    pinned_ = ::syscall(SYS_mlock, arena_, arena_bytes_) == 0;
    if (!pinned_) {
        init_error_ = std::string("mlock staging arena: ") + std::strerror(errno);
        if (why) *why = init_error_;
        return false;
    }
    return true;
}

MoeOffload::~MoeOffload() {
    size_t spins = 0;
    while (in_flight_slots_ > 0 && spins < 600) {
        drain(100);
        ++spins;
    }
    eng_.reset(); // joins the preadv workers / reaps every submitted SQE
    if (arena_) {
        if (pinned_) ::syscall(SYS_munlock, arena_, arena_bytes_);
        ::madvise(arena_, arena_bytes_, MADV_DONTNEED);
        ::munmap(arena_, arena_bytes_);
        arena_ = nullptr;
    }
}

bool MoeOffload::ok(std::string* err) const {
    if (err) *err = init_error_;
    return init_error_.empty();
}

const char* MoeOffload::backend_name() const { return eng_ ? eng_->name() : "none"; }

bool MoeOffload::staging_pinned() const { return pinned_; }

size_t MoeOffload::staging_bytes() const { return arena_bytes_; }

size_t MoeOffload::in_flight() const { return in_flight_slots_; }

bool MoeOffload::is_hot(uint32_t layer, uint32_t expert) const {
    if (layer >= t_.n_layers || expert >= t_.n_experts) return false;
    if (!cfg_.hot_mask.empty()) return cfg_.hot_mask[static_cast<size_t>(key(layer, expert))] != 0;
    return expert < cfg_.hot_per_layer;
}

uint64_t MoeOffload::hot_total() const {
    if (!cfg_.hot_mask.empty()) {
        uint64_t n = 0;
        for (uint8_t b : cfg_.hot_mask) n += b != 0 ? 1 : 0;
        return n;
    }
    return static_cast<uint64_t>(std::min(cfg_.hot_per_layer, t_.n_experts)) * t_.n_layers;
}

void MoeOffload::sanitize_cards() {
    wm_ = cfg_.cards;
    n_cards_ = static_cast<uint32_t>(wm_.size());
    for (uint32_t c = 0; c < n_cards_; ++c) {
        CardWatermark& w = wm_[c];
        w.high = w.high ? std::min(w.high, w.capacity) : w.capacity;
        w.low = std::min(w.low, w.high);
        if (card_keys_[c].capacity() < w.capacity) card_keys_[c].reserve(w.capacity);
    }
}

bool MoeOffload::admit(uint32_t card, uint64_t k) {
    if (card >= n_cards_ || k >= nkeys_ || res_card_[static_cast<size_t>(k)] >= 0) return false;
    std::vector<uint32_t>& keys = card_keys_[card];
    if (keys.size() >= wm_[card].capacity || keys.size() >= keys.capacity()) return false;
    keys.push_back(static_cast<uint32_t>(k));
    res_card_[static_cast<size_t>(k)] = static_cast<int8_t>(card);
    return true;
}

std::vector<Placement> MoeOffload::plan() const {
    std::vector<Placement> out;
    if (n_cards_ == 0) return out;
    uint32_t counts[2] = {0, 0};
    for (uint32_t c = 0; c < n_cards_; ++c) counts[c] = 0;
    for (uint32_t l = 0; l < t_.n_layers; ++l) {
        for (uint32_t e = 0; e < t_.n_experts; ++e) {
            if (!is_hot(l, e)) continue;
            bool placed = false;
            for (uint32_t c = 0; c < n_cards_; ++c) {
                if (counts[c] >= wm_[c].high) continue;
                out.push_back(Placement{l, e, static_cast<uint8_t>(c)});
                ++counts[c];
                placed = true;
                break;
            }
            if (!placed && counts[0] >= wm_[0].high && (n_cards_ < 2 || counts[1] >= wm_[1].high)) return out;
        }
    }
    return out;
}

void MoeOffload::replan() {
    for (uint32_t c = 0; c < n_cards_; ++c) {
        for (uint32_t k : card_keys_[c]) res_card_[k] = -1;
        card_keys_[c].clear();
    }
    if (n_cards_ == 0) return;
    const std::vector<Placement> p = plan();
    for (const Placement& x : p) admit(x.card, key(x.layer, x.expert));
}

uint32_t MoeOffload::enforce_watermarks(uint32_t card) {
    if (card >= n_cards_) return 0;
    std::vector<uint32_t>& keys = card_keys_[card];
    const uint32_t high = wm_[card].high, low = wm_[card].low;
    uint32_t demoted = 0;
    if (keys.size() > high) {
        while (keys.size() > low && !keys.empty()) {
            const uint32_t k = keys.back();
            keys.pop_back();
            res_card_[k] = -1;
            ++demoted;
        }
    }
    while (keys.size() > wm_[card].capacity && !keys.empty()) {
        const uint32_t k = keys.back();
        keys.pop_back();
        res_card_[k] = -1;
        ++demoted;
    }
    return demoted;
}

void MoeOffload::set_watermarks(const std::vector<CardWatermark>& cards) {
    std::vector<CardWatermark> next = cards;
    if (next.size() > 2) next.resize(2);
    if (next.size() != n_cards_) {
        for (uint32_t c = 0; c < n_cards_; ++c) {
            for (uint32_t k : card_keys_[c]) res_card_[k] = -1;
            card_keys_[c].clear();
        }
        n_cards_ = 0;
    }
    cfg_.cards = std::move(next);
    sanitize_cards();
    for (uint32_t c = 0; c < n_cards_; ++c) enforce_watermarks(c);
    // Deterministic promotion back up to the high marks without disturbing existing residents.
    for (uint32_t l = 0; l < t_.n_layers; ++l) {
        for (uint32_t e = 0; e < t_.n_experts; ++e) {
            if (!is_hot(l, e)) continue;
            const uint64_t k = key(l, e);
            if (res_card_[static_cast<size_t>(k)] >= 0) continue;
            for (uint32_t c = 0; c < n_cards_; ++c) {
                if (card_keys_[c].size() >= wm_[c].high) continue;
                if (admit(c, k)) break;
            }
        }
    }
}

uint32_t MoeOffload::resize_card(uint32_t card, uint32_t capacity) {
    if (card >= n_cards_) return 0;
    wm_[card].capacity = capacity;
    wm_[card].high = wm_[card].high ? std::min(wm_[card].high, capacity) : capacity;
    wm_[card].low = std::min(wm_[card].low, wm_[card].high);
    if (card_keys_[card].capacity() < capacity) card_keys_[card].reserve(capacity);
    return enforce_watermarks(card);
}

int32_t MoeOffload::resident_card(uint32_t layer, uint32_t expert) const {
    if (layer >= t_.n_layers || expert >= t_.n_experts) return -1;
    return res_card_[static_cast<size_t>(key(layer, expert))];
}

uint32_t MoeOffload::resident_on(uint32_t card) const {
    return card < n_cards_ ? static_cast<uint32_t>(card_keys_[card].size()) : 0;
}

void MoeOffload::lru_unlink(uint32_t s) {
    Slot& sl = slots_[s];
    if (sl.lru_prev >= 0) slots_[static_cast<size_t>(sl.lru_prev)].lru_next = sl.lru_next;
    else lru_head_ = sl.lru_next;
    if (sl.lru_next >= 0) slots_[static_cast<size_t>(sl.lru_next)].lru_prev = sl.lru_prev;
    else lru_tail_ = sl.lru_prev;
    sl.lru_prev = -1;
    sl.lru_next = -1;
}

void MoeOffload::lru_push_back(uint32_t s) {
    Slot& sl = slots_[s];
    sl.lru_prev = lru_tail_;
    sl.lru_next = -1;
    if (lru_tail_ >= 0) slots_[static_cast<size_t>(lru_tail_)].lru_next = static_cast<int32_t>(s);
    else lru_head_ = static_cast<int32_t>(s);
    lru_tail_ = static_cast<int32_t>(s);
}

void MoeOffload::lru_touch(uint32_t s) {
    if (slots_[s].state != SlotState::Ready) return;
    if (lru_tail_ == static_cast<int32_t>(s)) return;
    lru_unlink(s);
    lru_push_back(s);
}

void MoeOffload::drop_slot(uint32_t s) {
    Slot& sl = slots_[s];
    if (sl.state == SlotState::Free) return;
    if (sl.state == SlotState::Ready) lru_unlink(s);
    const uint64_t k = key(sl.layer, sl.expert);
    if (key_slot_[static_cast<size_t>(k)] == static_cast<int32_t>(s)) key_slot_[static_cast<size_t>(k)] = -1;
    sl = Slot{};
    free_.push_back(s);
}

int32_t MoeOffload::take_slot() {
    if (!free_.empty()) {
        const uint32_t s = free_.back();
        free_.pop_back();
        return static_cast<int32_t>(s);
    }
    if (lru_head_ < 0) {
        // A failed read leaves an Error slot out of the LRU; it is dead weight until its expert is fetched again.
        for (uint32_t s = 0; s < cfg_.staging_slots; ++s) {
            Slot& dead = slots_[s];
            if (dead.state != SlotState::Error) continue;
            const uint64_t dk = key(dead.layer, dead.expert);
            if (key_slot_[static_cast<size_t>(dk)] == static_cast<int32_t>(s)) key_slot_[static_cast<size_t>(dk)] = -1;
            dead = Slot{};
            ++st_.evictions;
            return static_cast<int32_t>(s);
        }
        return -1;
    }
    const uint32_t s = static_cast<uint32_t>(lru_head_);
    Slot& sl = slots_[s];
    lru_unlink(s);
    const uint64_t k = key(sl.layer, sl.expert);
    if (key_slot_[static_cast<size_t>(k)] == static_cast<int32_t>(s)) key_slot_[static_cast<size_t>(k)] = -1;
    sl = Slot{};
    ++st_.evictions;
    return static_cast<int32_t>(s);
}

void MoeOffload::apply(const IoCompletion& c) {
    ++st_.completions;
    if (outstanding_ops_ > 0) --outstanding_ops_;
    if (c.result < 0) ++st_.errors;
    bool short_seen = false;
    for (uint32_t s = 0; s < cfg_.staging_slots; ++s) {
        Slot& sl = slots_[s];
        if (sl.state != SlotState::Loading || sl.run != c.user) continue;
        const uint64_t need = static_cast<uint64_t>(sl.run_pos + 1) * t_.expert_bytes;
        const bool full = c.result >= 0 && static_cast<uint64_t>(c.result) >= need;
        if (full) {
            sl.state = SlotState::Ready;
            lru_push_back(s);
        } else {
            sl.state = SlotState::Error;
            sl.error = c.result < 0 ? c.result : -ENODATA;
            if (c.result >= 0) short_seen = true;
        }
        if (in_flight_slots_ > 0) --in_flight_slots_;
    }
    if (short_seen) ++st_.short_reads;
}

void MoeOffload::drain(int wait_ms) {
    if (!eng_ || completions_.empty()) return;
    const size_t n = eng_->reap(completions_.data(), completions_.size(), wait_ms);
    for (size_t i = 0; i < n; ++i) apply(completions_[i]);
}

size_t MoeOffload::stage(uint32_t layer, const uint32_t* experts, size_t n, bool count_requested) {
    if (!init_error_.empty()) return 0;
    if (count_requested) st_.requested += n;
    if (layer >= t_.n_layers || experts == nullptr) {
        st_.deferred += n;
        return 0;
    }
    drain(0);
    want_.clear();
    for (size_t i = 0; i < n; ++i) {
        const uint32_t e = experts[i];
        if (e >= t_.n_experts) {
            ++st_.errors;
            ++st_.deferred;
            continue;
        }
        const int32_t s = key_slot_[static_cast<size_t>(key(layer, e))];
        if (s >= 0) {
            ++st_.deduped;
            lru_touch(static_cast<uint32_t>(s));
            continue;
        }
        if (want_.size() >= cfg_.staging_slots) {
            ++st_.deferred;
            continue;
        }
        want_.push_back(e);
    }
    std::sort(want_.begin(), want_.end());
    const size_t before = want_.size();
    want_.erase(std::unique(want_.begin(), want_.end()), want_.end());
    st_.deduped += before - want_.size();

    const size_t op_budget = outstanding_ops_ < eng_->capacity()
                                 ? static_cast<size_t>(eng_->capacity() - static_cast<uint32_t>(outstanding_ops_))
                                 : 0;
    batch_ops_.clear();
    batch_iov_.clear();
    size_t issued = 0;
    size_t ops_made = 0;
    size_t run_len = 0;
    for (size_t i = 0; i < want_.size(); ++i) {
        const uint32_t e = want_[i];
        const bool extend = run_len > 0 && e == want_[i - 1] + 1 && run_len < cfg_.max_run_rows;
        if (!extend && (ops_made >= op_budget || batch_ops_.size() >= batch_ops_.capacity())) {
            st_.deferred += want_.size() - i;
            break;
        }
        if (batch_iov_.size() >= batch_iov_.capacity()) {
            st_.deferred += want_.size() - i;
            break;
        }
        const int32_t s = take_slot();
        if (s < 0) {
            st_.deferred += want_.size() - i;
            break;
        }
        if (!extend) {
            ReadRequest rq;
            rq.fd = t_.fd;
            rq.offset = t_.expert_offset(layer, e);
            rq.iov = batch_iov_.data() + batch_iov_.size();
            rq.iov_count = 0;
            rq.user = next_run_++;
            batch_ops_.push_back(rq);
            ++ops_made;
            run_len = 0;
        }
        Slot& sl = slots_[static_cast<size_t>(s)];
        sl.state = SlotState::Loading;
        sl.layer = layer;
        sl.expert = e;
        sl.run_pos = static_cast<uint32_t>(run_len);
        sl.run = batch_ops_.back().user;
        sl.error = 0;
        key_slot_[static_cast<size_t>(key(layer, e))] = s;
        ++run_len;
        batch_iov_.push_back(iovec{arena_ + static_cast<size_t>(s) * slot_stride_, t_.expert_bytes});
        batch_ops_.back().iov_count = static_cast<uint32_t>(run_len);
        ++issued;
    }
    if (issued == 0) return 0;

    st_.read_ops += batch_ops_.size();
    st_.issued += issued;
    st_.bytes += static_cast<uint64_t>(issued) * t_.expert_bytes;
    in_flight_slots_ += issued;
    outstanding_ops_ += batch_ops_.size();
    if (!eng_->submit(batch_ops_.data(), batch_ops_.size())) {
        // The engine refused the batch (would only happen if its capacity were exceeded): roll every slot back.
        for (const ReadRequest& rq : batch_ops_) {
            for (uint32_t s = 0; s < cfg_.staging_slots; ++s) {
                if (slots_[s].state == SlotState::Loading && slots_[s].run == rq.user) drop_slot(s);
            }
        }
        ++st_.errors;
        st_.deferred += issued;
        st_.issued -= issued;
        st_.read_ops -= batch_ops_.size();
        st_.bytes -= static_cast<uint64_t>(issued) * t_.expert_bytes;
        outstanding_ops_ -= batch_ops_.size();
        in_flight_slots_ -= issued;
        return 0;
    }
    return issued;
}

size_t MoeOffload::prefetch(uint32_t layer, const uint32_t* experts, size_t n) {
    return stage(layer, experts, n, true);
}

void MoeOffload::poll() { drain(0); }

bool MoeOffload::staged(uint32_t layer, uint32_t expert) const {
    if (!init_error_.empty() || layer >= t_.n_layers || expert >= t_.n_experts) return false;
    const int32_t s = key_slot_[static_cast<size_t>(key(layer, expert))];
    return s >= 0 && slots_[static_cast<size_t>(s)].state == SlotState::Ready;
}

bool MoeOffload::read_direct(uint32_t layer, uint32_t expert, uint8_t* dst, std::string* err) {
    ++st_.read_ops;
    const int64_t r = pread_full(t_.fd, t_.expert_offset(layer, expert), dst, t_.expert_bytes);
    if (r < 0) {
        ++st_.errors;
        if (err) *err = std::strerror(static_cast<int>(-r));
        return false;
    }
    st_.bytes += static_cast<uint64_t>(r);
    if (r != static_cast<int64_t>(t_.expert_bytes)) {
        ++st_.errors;
        ++st_.short_reads;
        if (err) *err = "short read (expert beyond end of file)";
        return false;
    }
    return true;
}

bool MoeOffload::fetch(uint32_t layer, uint32_t expert, uint8_t* dst, std::string* err) {
    auto fail = [&](const std::string& m) {
        if (err) *err = m;
        return false;
    };
    if (!init_error_.empty()) return fail(init_error_);
    if (layer >= t_.n_layers || expert >= t_.n_experts) {
        ++st_.errors;
        return fail("expert out of range");
    }
    if (dst == nullptr) {
        ++st_.errors;
        return fail("null destination");
    }
    drain(0);
    int32_t s = key_slot_[static_cast<size_t>(key(layer, expert))];
    if (s < 0) {
        ++st_.misses;
        const uint32_t e = expert;
        stage(layer, &e, 1, false);
        s = key_slot_[static_cast<size_t>(key(layer, expert))];
        if (s < 0) return read_direct(layer, expert, dst, err); // every slot is loading: read straight to dst
    } else if (slots_[static_cast<size_t>(s)].state == SlotState::Ready) {
        ++st_.hits;
    } else {
        ++st_.misses; // queued or failed
    }
    size_t spins = 0;
    while (slots_[static_cast<size_t>(s)].state == SlotState::Loading) {
        drain(100);
        if (++spins > 600) {
            ++st_.errors;
            return fail("read did not complete");
        }
    }
    if (slots_[static_cast<size_t>(s)].state == SlotState::Error) {
        const int64_t e = slots_[static_cast<size_t>(s)].error;
        ++st_.errors;
        drop_slot(static_cast<uint32_t>(s));
        return fail(e == -ENODATA ? "short read (expert beyond end of file)"
                                  : std::strerror(static_cast<int>(-e)));
    }
    std::memcpy(dst, arena_ + static_cast<size_t>(s) * slot_stride_, t_.expert_bytes);
    lru_touch(static_cast<uint32_t>(s));
    return true;
}

bool MoeOffload::release(uint32_t layer, uint32_t expert) {
    if (!init_error_.empty() || layer >= t_.n_layers || expert >= t_.n_experts) return false;
    const int32_t s = key_slot_[static_cast<size_t>(key(layer, expert))];
    if (s < 0) return false;
    size_t spins = 0;
    while (slots_[static_cast<size_t>(s)].state == SlotState::Loading) {
        drain(100);
        if (++spins > 600) return false;
    }
    drop_slot(static_cast<uint32_t>(s));
    // mlock'ed VMAs reject MADV_DONTNEED, so unlock the slot, drop its physical pages, then re-lock it.
    uint8_t* p = arena_ + static_cast<size_t>(s) * slot_stride_;
    if (pin_requested_ && pinned_) ::syscall(SYS_munlock, p, slot_stride_);
    const bool dropped = ::madvise(p, slot_stride_, MADV_DONTNEED) == 0;
    if (pin_requested_ && pinned_ && ::mlock2(p, slot_stride_, MLOCK_ONFAULT) != 0) {
        pinned_ = false;
        init_error_ = "cannot repin released staging slot";
        return false;
    }
    ++st_.releases;
    if (dropped) st_.released_bytes += slot_stride_;
    return dropped;
}

void MoeOffload::release_all() {
    if (!init_error_.empty()) return;
    size_t spins = 0;
    while (in_flight_slots_ > 0 && spins < 600) {
        drain(100);
        ++spins;
    }
    if (in_flight_slots_ != 0) { init_error_ = "cannot retire staging reads"; return; }
    for (uint32_t s = 0; s < cfg_.staging_slots; ++s) slots_[s] = Slot{};
    for (size_t i = 0; i < key_slot_.size(); ++i) key_slot_[i] = -1;
    free_.clear();
    for (uint32_t s = cfg_.staging_slots; s > 0; --s) free_.push_back(s - 1);
    lru_head_ = lru_tail_ = -1;
    in_flight_slots_ = 0;
    outstanding_ops_ = 0;
    if (!arena_) return;
    if (pin_requested_ && pinned_) ::syscall(SYS_munlock, arena_, arena_bytes_);
    const bool dropped = ::madvise(arena_, arena_bytes_, MADV_DONTNEED) == 0;
    if (pin_requested_ && pinned_ && ::mlock2(arena_, arena_bytes_, MLOCK_ONFAULT) != 0) {
        pinned_ = false;
        init_error_ = "cannot repin staging arena";
    }
    ++st_.releases;
    if (dropped) st_.released_bytes += arena_bytes_;
}

} // namespace vegajet::moeoffload
