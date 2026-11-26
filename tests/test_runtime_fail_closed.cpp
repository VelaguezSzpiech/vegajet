// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Exercise the live host runtime with fake ioctls: no GPU or device files required.
#include "core/kfd_runtime.hpp"
#include <sys/ioctl.h>
#include <linux/kfd_ioctl.h>
#include "tests/test_common.hpp"
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
    if (op == AMDKFD_IOC_MAP_MEMORY_TO_GPU) {
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
        std::string err;
        VJ_CHECK(!rt.reload_all_hsacos(&err));
        VJ_CHECK(err.find("runtime stopped") != std::string::npos);
        VJ_CHECK(!rt.open_ && rt.queues_.empty());
        VJ_CHECK(!rt.execute({}, &err) && err == "Runtime is not open");
    }
};
}
int main() {
    vegajet::kfd::RuntimeTestAccess::reload_failure();
    vegajet::kfd::RuntimeTestAccess::mapping_ownership();
    std::puts("runtime fail-closed tests PASS");
}
