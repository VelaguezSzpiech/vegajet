// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Exercise the live host runtime with fake ioctls: no GPU or device files required.
#include "core/kfd_runtime.hpp"
#include <sys/ioctl.h>
#include <linux/kfd_ioctl.h>
#include "tests/test_common.hpp"
#include "tests/live_hsaco_fixture.hpp"
#include <fcntl.h>
#include <cerrno>
#include <cstdarg>
#include <cstdio>

namespace {
bool io_ok = false;
uint32_t map_done = 0, unmap_done = 0;
bool fail_unmap = false, fail_free = false;
unsigned free_calls = 0;
std::vector<uint32_t> unmapped;
}
extern "C" int __wrap_ioctl(int, unsigned long op, ...) {
    va_list ap;
    va_start(ap, op);
    void* arg = va_arg(ap, void*);
    va_end(ap);
    if (!io_ok) { errno = EIO; return -1; }
    if (op == AMDKFD_IOC_ALLOC_MEMORY_OF_GPU) {
        static uint64_t handle = 1000;
        auto* a = static_cast<kfd_ioctl_alloc_memory_of_gpu_args*>(arg);
        a->handle = ++handle;
    } else if (op == AMDKFD_IOC_MAP_MEMORY_TO_GPU) {
        auto* a = static_cast<kfd_ioctl_map_memory_to_gpu_args*>(arg);
        a->n_success = std::min(map_done, a->n_devices);
        if (a->n_success != a->n_devices) { errno = EIO; return -1; }
    } else if (op == AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU) {
        auto* a = static_cast<kfd_ioctl_unmap_memory_from_gpu_args*>(arg);
        auto* ids = reinterpret_cast<uint32_t*>(a->device_ids_array_ptr);
        unmapped.assign(ids, ids + a->n_devices);
        a->n_success = std::min(unmap_done, a->n_devices);
        if (fail_unmap) { errno = EIO; return -1; }
    } else if (op == AMDKFD_IOC_FREE_MEMORY_OF_GPU) {
        ++free_calls;
        if (fail_free) { errno = EIO; return -1; }
    }
    return 0;
}

namespace vegajet::kfd {
struct RuntimeTestAccess {
    static void mapping_ownership() {
        Runtime rt;
        rt.kfd_fd_ = 12345;
        rt.all_gpu_ids_ = {11, 22, 33};
        io_ok = true;
        map_done = 1;
        KfdBuffer b{};
        b.handle = 99;
        b.host_ptr = std::malloc(64);
        b.is_userptr = true;
        b.size = 64;
        VJ_CHECK(!rt.map_to_gpus(b, {22, 33}));
        VJ_CHECK(b.mapped_gpu_ids == std::vector<uint32_t>{22});
        const void* ptr = b.host_ptr;
        fail_unmap = true; unmap_done = 0;
        VJ_CHECK(!rt.free_buffer(b));
        VJ_CHECK(b.host_ptr == ptr && b.handle == 99 && free_calls == 0);
        VJ_CHECK(unmapped == std::vector<uint32_t>{22});
        VJ_CHECK(rt.retained_buffers_.count(99) == 1);
        fail_unmap = false; unmap_done = 1; fail_free = true;
        VJ_CHECK(!rt.free_buffer(b));
        VJ_CHECK(b.mapped_gpu_ids.empty() && b.host_ptr == ptr);
        fail_free = false;
        VJ_CHECK(rt.free_buffer(b));
        VJ_CHECK(!b.host_ptr && !b.handle && rt.retained_buffers_.empty());
        rt.kfd_fd_ = -1;
        io_ok = false;
    }
    static void completion_lifecycle() {
        Runtime rt;
        rt.open_ = true;
        rt.queues_.resize(1);
        auto& q = rt.queues_[0];
        AqlDispatchPacket ring[16]{};
        uint64_t rptr = 0, wptr = 0, doorbell = 0;
        AmdSignal signal{};
        q.ring_buf.host_ptr = ring;
        q.ring_packets = 16;
        q.read_ptr = &rptr; q.write_ptr = &wptr; q.doorbell = &doorbell;
        q.completion_sig = &signal;
        q.next_write_idx = 1;
        VJ_CHECK(rt.submit_batch(0));
        const auto wp = q.next_write_idx;
        VJ_CHECK(!rt.submit_batch(0));
        VJ_CHECK(q.next_write_idx == wp && signal.value == 1);
        LoadedKernel kernel{};
        VJ_CHECK(!rt.enqueue_kernel(0, kernel, 1, 1, 1, 1, 1, 1, nullptr, 0));
        signal.value = 0;
        VJ_CHECK(rt.wait_idle(0, 0));
        VJ_CHECK(!q.submission_in_flight && q.completed_write_idx == wp);
        // A stale zero cannot hide pending, already-rung work: wait adds a fresh fence and times out.
        ++q.next_write_idx;
        rt.ring_doorbell(0);
        VJ_CHECK(!rt.wait_idle(0, 0));
        VJ_CHECK(signal.value == 1 && !rt.open_);
        rt.open_ = true;
        signal.value = -1;
        VJ_CHECK(!rt.wait_idle(0, 0) && !rt.open_);
        q = KfdAqlQueue{};
    }
    static void kernarg_ownership() {
        Runtime rt;
        rt.open_ = true; rt.queues_.resize(1);
        auto& q = rt.queues_[0];
        std::vector<AqlDispatchPacket> ring(4096);
        std::vector<uint8_t> args(2048 * 512, 0xa5);
        uint64_t rptr = 0, wptr = 0, doorbell = 0;
        AmdSignal signal{};
        q.ring_buf.host_ptr = ring.data(); q.ring_packets = 4096;
        q.read_ptr = &rptr; q.write_ptr = &wptr; q.doorbell = &doorbell;
        q.completion_sig = &signal;
        q.kernarg_host = args.data(); q.kernarg_capacity = args.size();
        LoadedKernel kernel{};
        uint64_t arg = 0x1234;
        for (unsigned i = 0; i < 2048; ++i)
            VJ_CHECK(rt.enqueue_kernel(0, kernel, 1, 1, 1, 1, 1, 1, &arg, sizeof(arg)));
        const auto before = args;
        arg = 0x5678;
        VJ_CHECK(!rt.enqueue_kernel(0, kernel, 1, 1, 1, 1, 1, 1, &arg, sizeof(arg)));
        VJ_CHECK(q.kernarg_slot == 2048 && q.next_write_idx == 2048 && args == before);
        VJ_CHECK(rt.submit_batch(0));
        signal.value = 0; rptr = q.next_write_idx;
        VJ_CHECK(rt.wait_idle(0, 0) && q.kernarg_slot == 0);
        VJ_CHECK(rt.enqueue_kernel(0, kernel, 1, 1, 1, 1, 1, 1, &arg, sizeof(arg)));
        // AQL reservation failure must not mutate even the first byte or increment a cursor.
        q.ring_packets = 1; q.kernarg_slot = 0;
        const auto before_fail = args;
        VJ_CHECK(!rt.enqueue_kernel(0, kernel, 1, 1, 1, 1, 1, 1, &arg, sizeof(arg)));
        VJ_CHECK(q.kernarg_slot == 0 && args == before_fail && !rt.open_);
        q = KfdAqlQueue{};
    }
    static void reload_failure() {
        Runtime rt;
        rt.open_ = true;
        rt.devices_.resize(1);
        rt.queues_.resize(1);
        AmdSignal signal{};
        rt.queues_[0].completion_sig = &signal;
        Runtime::LoadedModule first;
        first.name = "missing";
        first.path = "/nonexistent/vegajet-runtime-test.hsaco";
        rt.loaded_modules_.push_back(first);
        Runtime::LoadedModule valid;
        valid.name = "valid";
        valid.path = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : ".") +
                     "/runtime-reload-" + std::to_string(getpid()) + ".hsaco";
        const auto bytes = vjtest::live_hsaco_fixture();
        FILE* file = std::fopen(valid.path.c_str(), "wb");
        VJ_CHECK(file && std::fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size());
        std::fclose(file);
        rt.loaded_modules_.insert(rt.loaded_modules_.begin(), valid);
        rt.kernels_.resize(1);
        rt.devices_[0].drm_fd = ::open("/dev/zero", O_RDWR);
        VJ_CHECK(rt.devices_[0].drm_fd >= 0);
        rt.kfd_fd_ = dup(rt.devices_[0].drm_fd);
        rt.va_cursor_ = 0x10000000000ULL; rt.va_limit_ = 0x20000000000ULL;
        io_ok = true; map_done = unmap_done = 32;
        std::string err;
        VJ_CHECK(!rt.reload_all_hsacos(&err));
        VJ_CHECK(rt.reload_generation_ == 1); // first module committed, second failed
        VJ_CHECK(err.find("runtime stopped") != std::string::npos);
        VJ_CHECK(!rt.open_ && rt.queues_.empty());
        VJ_CHECK(!rt.execute({}, &err) && err == "Runtime is not open");
        rt.shutdown();
        VJ_CHECK(rt.retained_buffers_.empty());
        std::remove(valid.path.c_str());
        io_ok = false;
        free_calls = 0;
    }
};
}
int main() {
    vegajet::kfd::RuntimeTestAccess::reload_failure();
    vegajet::kfd::RuntimeTestAccess::mapping_ownership();
    vegajet::kfd::RuntimeTestAccess::completion_lifecycle();
    vegajet::kfd::RuntimeTestAccess::kernarg_ownership();
    std::puts("runtime fail-closed tests PASS");
}
