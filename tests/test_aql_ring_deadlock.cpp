// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Ring-full flush (aql_wait_space) on a CPU mock AQL ring; no device access. A consumer thread plays the CP: it
// only sees packets up to the published write_ptr, requires a valid header on every packet it consumes, records
// the order, invalidates the slot and advances read_ptr. The writer mirrors Runtime::enqueue_kernel (wait for 2
// free slots, header stored last) and Runtime::submit_batch (wait for 1, closing BARRIER_AND, publish), with the
// early doorbell after the first dispatch that the decode loop issues after layer 0. Checks:
//  * a token longer than the ring completes in exact order (the pre-fix wait spins until its timeout: shown);
//  * with room in the ring the guard never publishes, so the 2-doorbell early-start overlap is unchanged;
//  * an undrained ring publishes what it holds once, then times out within the bound;
//  * a full ring that is already published is not re-published.
#include "core/kfd_runtime.hpp"
#include "tests/test_common.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

using namespace vegajet::kfd;

namespace {

constexpr uint64_t kSignalVa = 0x7000'0000'0200ULL;
constexpr uint64_t kTimeoutUs = 5000000ULL;

struct MockQueue {
    std::vector<AqlDispatchPacket> ring;
    volatile uint64_t write_ptr = 0, read_ptr = 0, doorbell = 0;
    KfdAqlQueue q;
    uint64_t doorbells = 0;  // explicit (non-forced) doorbells issued by the writer
    explicit MockQueue(uint32_t packets) : ring(packets) {
        for (auto& p : ring) p.header = kPacketTypeInvalid;
        q.ring_packets = packets;
        q.ring_buf.host_ptr = ring.data();
        q.write_ptr = &write_ptr;
        q.read_ptr = &read_ptr;
        q.doorbell = &doorbell;
    }
};

struct Seen { uint16_t type; uint64_t kernel_object; uint64_t completion; };

class MockCp {
public:
    explicit MockCp(MockQueue& m) : m_(m), th_([this] { run(); }) {}
    ~MockCp() { finish(); }
    void finish() { stop_ = true; if (th_.joinable()) th_.join(); }
    const std::vector<Seen>& seen() const { return seen_; }
    bool bad() const { return bad_; }
private:
    void run() {
        uint64_t r = 0;
        for (;;) {
            const uint64_t w = __atomic_load_n(&m_.write_ptr, __ATOMIC_ACQUIRE);
            if (r == w) {
                if (stop_) return;
                std::this_thread::yield();
                continue;
            }
            auto* p = &m_.ring[r % m_.ring.size()];
            const uint16_t type = __atomic_load_n(&p->header, __ATOMIC_ACQUIRE) & 0xFFu;
            if (type != kPacketTypeKernelDispatch && type != kPacketTypeBarrierAnd) { bad_ = true; return; }
            seen_.push_back({type, p->kernel_object, p->completion_signal});
            __atomic_store_n(&p->header, kPacketTypeInvalid, __ATOMIC_RELEASE);
            ++r;
            __atomic_store_n(&m_.read_ptr, r, __ATOMIC_RELEASE);
        }
    }
    MockQueue& m_;
    std::atomic<bool> stop_{false};
    bool bad_ = false;
    std::vector<Seen> seen_;
    std::thread th_;
};

// The wait loop of enqueue_kernel / submit_batch before the fix, with a short timeout.
bool legacy_wait_space(KfdAqlQueue& q, uint32_t keep_free, uint64_t timeout_us) {
    const auto t0 = std::chrono::steady_clock::now();
    while (q.next_write_idx - __atomic_load_n(q.read_ptr, __ATOMIC_ACQUIRE) >= static_cast<uint64_t>(q.ring_packets - keep_free)) {
        if (static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t0).count()) >= timeout_us) return false;
        std::this_thread::yield();
    }
    return true;
}

using WaitFn = bool (*)(KfdAqlQueue&, uint32_t, uint64_t);

void write_header(AqlDispatchPacket* p, uint16_t hdr) { __atomic_store_n(&p->header, hdr, __ATOMIC_RELEASE); }

bool write_dispatch(KfdAqlQueue& q, uint64_t id, WaitFn wait, uint64_t timeout_us) {
    if (!wait(q, 2u, timeout_us)) return false;
    auto* p = reinterpret_cast<AqlDispatchPacket*>(q.ring_buf.host_ptr) + (q.next_write_idx++ % q.ring_packets);
    p->setup = 3;
    p->workgroup_size_x = 256; p->workgroup_size_y = 1; p->workgroup_size_z = 1;
    p->grid_size_x = 256; p->grid_size_y = 1; p->grid_size_z = 1;
    p->kernel_object = 0x1000 + id;
    p->kernarg_address = 0x9000'0000ULL + id * 512u;
    p->completion_signal = 0;
    __atomic_thread_fence(__ATOMIC_RELEASE);
    write_header(p, make_aql_header(kPacketTypeKernelDispatch, true, kFenceScopeAgent, kFenceScopeAgent));
    return true;
}

void publish(MockQueue& m) {  // Runtime::ring_doorbell / the tail of submit_batch
    __atomic_store_n(m.q.write_ptr, m.q.next_write_idx, __ATOMIC_RELEASE);
    __atomic_store_n(m.q.doorbell, m.q.next_write_idx - 1u, __ATOMIC_RELEASE);
    ++m.doorbells;
}

bool run_token(MockQueue& m, uint64_t first_id, uint32_t n, WaitFn wait, uint64_t timeout_us) {
    for (uint32_t i = 0; i < n; ++i) {
        if (!write_dispatch(m.q, first_id + i, wait, timeout_us)) return false;
        if (i == 0) publish(m);  // early-start doorbell after layer 0
    }
    if (!wait(m.q, 1u, timeout_us)) return false;
    auto* p = reinterpret_cast<AqlDispatchPacket*>(m.q.ring_buf.host_ptr) + (m.q.next_write_idx++ % m.q.ring_packets);
    auto* raw = reinterpret_cast<uint64_t*>(p);
    auto* raw16 = reinterpret_cast<uint16_t*>(raw);
    raw16[1] = 0; raw16[2] = 0; raw16[3] = 0;
    for (int i = 1; i < 7; ++i) raw[i] = 0;
    raw[7] = kSignalVa;
    __atomic_thread_fence(__ATOMIC_RELEASE);
    write_header(p, make_aql_header(kPacketTypeBarrierAnd, true, kFenceScopeSystem, kFenceScopeSystem));
    publish(m);
    return true;
}

void drain(MockQueue& m, MockCp& cp, uint64_t total) {
    while (__atomic_load_n(&m.read_ptr, __ATOMIC_ACQUIRE) < total) std::this_thread::yield();
    cp.finish();
}

void check_order(const std::vector<Seen>& seen, uint32_t tokens, uint32_t per_token) {
    VJ_CHECK(seen.size() == static_cast<size_t>(tokens) * (per_token + 1u));
    size_t k = 0;
    for (uint32_t t = 0; t < tokens; ++t) {
        for (uint32_t i = 0; i < per_token; ++i, ++k) {
            VJ_CHECK(seen[k].type == kPacketTypeKernelDispatch);
            VJ_CHECK(seen[k].kernel_object == 0x1000 + static_cast<uint64_t>(t) * per_token + i);
        }
        VJ_CHECK(seen[k].type == kPacketTypeBarrierAnd && seen[k].completion == kSignalVa);
        ++k;
    }
}

void test_legacy_wait_deadlocks() {
    MockQueue m(8);
    MockCp cp(m);
    // 20 dispatches on an 8-slot ring: after the early doorbell (1 packet) the CP drains it, then the writer fills
    // the ring with 6 unpublished packets and the old wait spins for the whole timeout.
    const auto t0 = std::chrono::steady_clock::now();
    VJ_CHECK(!run_token(m, 0, 20, legacy_wait_space, 200000ULL));
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    cp.finish();
    VJ_CHECK(!cp.bad() && cp.seen().size() == 1 && m.read_ptr == 1 && m.write_ptr == 1);
    VJ_CHECK(ms >= 190.0);
    std::printf("  pre-fix wait: token of 20 on an 8-slot ring stalls (%.0f ms timeout, CP saw 1 packet)\n", ms);
}

void test_long_token_completes() {
    MockQueue m(8);
    MockCp cp(m);
    const uint32_t tokens = 25, per_token = 37;  // 950 packets through an 8-slot ring
    for (uint32_t t = 0; t < tokens; ++t)
        VJ_CHECK(run_token(m, static_cast<uint64_t>(t) * per_token, per_token, aql_wait_space, kTimeoutUs));
    const uint64_t total = static_cast<uint64_t>(tokens) * (per_token + 1u);
    drain(m, cp, total);
    VJ_CHECK(!cp.bad());
    check_order(cp.seen(), tokens, per_token);
    VJ_CHECK(m.q.next_write_idx == total && m.read_ptr == total && m.write_ptr == total && m.doorbell == total - 1);
    VJ_CHECK(m.q.forced_flushes > 0);
    std::printf("  8-slot ring: %llu packets, %llu explicit doorbells + %llu ring-full publishes, order exact\n",
                static_cast<unsigned long long>(total), static_cast<unsigned long long>(m.doorbells),
                static_cast<unsigned long long>(m.q.forced_flushes));
}

void test_roomy_ring_keeps_two_doorbells() {
    MockQueue m(2048);  // the runtime's ring size; a 100-dispatch decode token fits
    MockCp cp(m);
    for (uint32_t t = 0; t < 30; ++t) {  // decode: each token is waited for (wait_idle) before the next one
        VJ_CHECK(run_token(m, static_cast<uint64_t>(t) * 100u, 100, aql_wait_space, kTimeoutUs));
        while (__atomic_load_n(&m.read_ptr, __ATOMIC_ACQUIRE) < (t + 1u) * 101u) std::this_thread::yield();
    }
    drain(m, cp, 30u * 101u);
    VJ_CHECK(!cp.bad());
    check_order(cp.seen(), 30, 100);
    VJ_CHECK(m.q.forced_flushes == 0 && m.doorbells == 60);  // exactly the early + closing doorbell per token
}

void test_undrained_ring_times_out() {
    MockQueue m(8);  // no consumer
    int ok = 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (write_dispatch(m.q, static_cast<uint64_t>(ok), aql_wait_space, 20000ULL)) ++ok;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    VJ_CHECK(ok == 6);  // 8 slots - 2 kept free
    VJ_CHECK(m.q.forced_flushes == 1 && m.write_ptr == 6 && m.doorbell == 5);
    VJ_CHECK(ms < 2000.0);
    for (int i = 0; i < 6; ++i) {
        VJ_CHECK((m.ring[static_cast<size_t>(i)].header & 0xFFu) == kPacketTypeKernelDispatch);
        VJ_CHECK(m.ring[static_cast<size_t>(i)].kernel_object == 0x1000u + static_cast<uint64_t>(i));
    }
    // Already published and still full: a second wait must not publish again.
    VJ_CHECK(!aql_wait_space(m.q, 2u, 1000ULL));
    VJ_CHECK(m.q.forced_flushes == 1 && m.write_ptr == 6);
}

} // namespace

int main() {
    test_legacy_wait_deadlocks();
    test_long_token_completes();
    test_roomy_ring_keeps_two_doorbells();
    test_undrained_ring_times_out();
    std::printf("test_aql_ring_deadlock: PASS\n");
    return 0;
}
