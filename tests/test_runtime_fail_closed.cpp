// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Exercise the live host runtime with fake ioctls: no GPU or device files required.
#include "core/kfd_runtime.hpp"
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/kfd_ioctl.h>
#include "tests/test_common.hpp"
#include "tests/live_hsaco_fixture.hpp"
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <thread>
#include <chrono>
#include <cerrno>
#include <cstdarg>
#include <cstdio>

namespace {
bool io_ok = false, fake_devices = false;
std::vector<uint32_t> allocation_owners, mapping_targets;
uint32_t map_done = 0, unmap_done = 0;
bool fail_unmap = false, fail_free = false;
unsigned free_calls = 0, ioctl_calls = 0;
int forced_errno = 0, destroy_error = 0;
std::vector<uint32_t> unmapped;
}
extern "C" int __real_open(const char*, int, ...);
extern "C" int __wrap_open(const char* path, int flags, ...) {
    mode_t mode=0;
    if (flags & O_CREAT) { va_list ap; va_start(ap,flags); mode=va_arg(ap,mode_t); va_end(ap); }
    if (fake_devices && (std::string(path)=="/dev/kfd" || std::string(path).find("/dev/dri/renderD")==0))
        return __real_open("/dev/zero",flags,mode);
    return __real_open(path,flags,mode);
}
extern "C" int __wrap_ioctl(int, unsigned long op, ...) {
    va_list ap;
    va_start(ap, op);
    void* arg = va_arg(ap, void*);
    va_end(ap);
    ++ioctl_calls;
    if (forced_errno) { errno = forced_errno; return -1; }
    if (op == AMDKFD_IOC_DESTROY_QUEUE && destroy_error) { errno = destroy_error; return -1; }
    if (!io_ok) { errno = EIO; return -1; }
    if (op == AMDKFD_IOC_ALLOC_MEMORY_OF_GPU) {
        static uint64_t handle = 1000;
        auto* a = static_cast<kfd_ioctl_alloc_memory_of_gpu_args*>(arg);
        a->handle = ++handle;
        allocation_owners.push_back(a->gpu_id);
    } else if (op == AMDKFD_IOC_MAP_MEMORY_TO_GPU) {
        auto* a = static_cast<kfd_ioctl_map_memory_to_gpu_args*>(arg);
        const auto* ids = reinterpret_cast<const uint32_t*>(a->device_ids_array_ptr);
        mapping_targets.insert(mapping_targets.end(), ids, ids + a->n_devices);
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
    static void mailbox_reset() {
        Runtime rt;
        rt.open_ = true; rt.devices_.resize(2); rt.queues_.resize(2);
        AmdSignal signals[2]{};
        for (size_t g=0;g<2;++g) rt.queues_[g].completion_sig = &signals[g];
        rt.transport_ = AllReduceTransport::PeerBarP2P;
        rt.p2p_payload_slices_.resize(2); rt.p2p_flag_slices_.resize(2);
        std::vector<uint8_t> payload(2*kArGpuBytes,0xa5), flags(8192,0xa5);
        for (size_t g=0;g<2;++g) {
            rt.p2p_payload_slices_[g].size=kArGpuBytes;
            rt.p2p_payload_slices_[g].host_ptr=payload.data()+g*kArGpuBytes;
            rt.p2p_flag_slices_[g].size=4096;
            rt.p2p_flag_slices_[g].host_ptr=flags.data()+g*4096;
        }
        rt.p2p_payload_slices_[1].host_ptr=nullptr;
        std::string err;
        VJ_CHECK(!rt.reset_allreduce_mailboxes(&err));
        VJ_CHECK(rt.failed_ && !rt.open_ && flags[0]==0xa5 && payload[0]==0xa5);
        rt.p2p_payload_slices_[1].host_ptr=payload.data()+kArGpuBytes;
        rt.failed_=false; rt.open_=true;
        signals[1].value=-1;
        VJ_CHECK(!rt.reset_allreduce_mailboxes(&err) && flags[0]==0xa5 && payload[0]==0xa5);
        rt.failed_=false; rt.open_=true;
        signals[1].value=0; signals[0].value=1;
        rt.queues_[0].submission_in_flight=true;
        std::thread gpu([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            __atomic_store_n(&signals[0].value,0LL,__ATOMIC_RELEASE);
        });
        VJ_CHECK(rt.reset_allreduce_mailboxes(&err));
        gpu.join();
        VJ_CHECK(std::all_of(payload.begin(),payload.end(),[](auto v){return v==0;}));
        VJ_CHECK(std::all_of(flags.begin(),flags.end(),[](auto v){return v==0;}));
        rt.p2p_payload_slices_.clear(); rt.p2p_flag_slices_.clear();
        rt.open_=false;
        for (auto& q : rt.queues_) q=KfdAqlQueue{}; // normal init resets mailboxes before queue creation
        // A missing CPU mmap must not leave peer mode enabled; the fully mapped host fallback is explicit.
        rt.devices_[0].p2p_links_count=1;
        rt.all_gpu_ids_={1,2};
        rt.kfd_fd_=::open("/dev/zero",O_RDWR);
        rt.va_cursor_=0x10000000000ULL; rt.va_limit_=0x20000000000ULL;
        io_ok=true; map_done=unmap_done=32;
        VJ_CHECK(rt.setup_allreduce_mailbox());
        VJ_CHECK(rt.transport_==AllReduceTransport::HostPinnedUserptr && rt.mailbox_flag_buf_.host_ptr);
        rt.shutdown(); io_ok=false; free_calls=0;
    }
    static void teardown_failures() {
        io_ok=true;
        for (int error : {EIO,EPERM,EBUSY}) {
            Runtime rt;
            rt.open_=true; rt.queues_.resize(1);
            rt.kfd_fd_=::open("/dev/zero",O_RDWR);
            auto& q=rt.queues_[0];
            q.queue_active=true; q.queue_id=0; // zero is a valid live queue ID
            q.doorbell_map=mmap(nullptr,4096,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
            VJ_CHECK(q.doorbell_map!=MAP_FAILED);
            q.doorbell_map_size=4096; q.doorbell=static_cast<volatile uint64_t*>(q.doorbell_map);
            q.ring_buf.handle=123; q.ring_buf.size=64; q.ring_buf.is_userptr=true;
            q.ring_buf.host_ptr=std::malloc(64);
            KfdBuffer code{}; code.handle=124; code.host_ptr=std::malloc(64); code.size=64; code.is_userptr=true;
            rt.code_buffers_.push_back(code);
            const auto frees=free_calls;
            const void* db=q.doorbell_map;
            const void* ring=q.ring_buf.host_ptr;
            destroy_error=error;
            VJ_CHECK(!rt.abort_dispatches());
            rt.shutdown();
            VJ_CHECK(rt.failed_ && !rt.open_ && rt.queues_.size()==1 && rt.queues_[0].queue_active);
            VJ_CHECK(q.doorbell_map==db && q.ring_buf.host_ptr==ring && free_calls==frees);
            VJ_CHECK(rt.code_buffers_.size()==1 && rt.kfd_fd_>=0);
            std::string err;
            VJ_CHECK(!rt.init(1,1,&err) && err.find("cleanup")!=std::string::npos);
            destroy_error=0;
            rt.shutdown();
            VJ_CHECK(rt.queues_.empty() && rt.code_buffers_.empty() && rt.kfd_fd_==-1);
            VJ_CHECK(rt.retained_buffers_.empty());
        }
        for (int error : {EINTR,EAGAIN}) {
            Runtime rt;
            rt.kfd_fd_=::open("/dev/zero",O_RDWR);
            KfdBuffer b{}; b.handle=77;
            forced_errno=error; ioctl_calls=0;
            const auto start=std::chrono::steady_clock::now();
            VJ_CHECK(!rt.free_buffer(b));
            VJ_CHECK(ioctl_calls==64 && b.handle==77 && rt.retained_buffers_.count(77)==1 && errno==ETIMEDOUT);
            VJ_CHECK(std::chrono::steady_clock::now()-start < std::chrono::seconds(2));
            forced_errno=0;
            VJ_CHECK(rt.free_buffer(b) && rt.retained_buffers_.empty());
            rt.shutdown();
        }
        io_ok=false; free_calls=0;
    }
    static void exact_tp_selection() {
        namespace fs = std::filesystem;
        const auto root = fs::path(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : ".") /
                          ("runtime-topology-" + std::to_string(getpid()));
        for (unsigned i=0; i<3; ++i) {
            const auto dir = root / "sys/class/kfd/kfd/topology/nodes" / std::to_string(i);
            fs::create_directories(dir);
            std::ofstream(dir / "properties") << "gfx_target_version 90006\ndomain 0\nlocation_id " << ((3+64*i)<<8) << "\n";
            std::ofstream(dir / "gpu_id") << i+1;
            fs::create_directories(dir / "mem_banks/0");
            std::ofstream(dir / "mem_banks/0/properties") << "size_in_bytes " << (uint64_t{32} << 30) << "\n";
        }
        setenv("VJ_SYSFS_ROOT",root.c_str(),1);
        unsetenv("VJ_GPU_DEVICES");
        Runtime rt;
        std::string err;
        VJ_CHECK(!rt.init(0,0,&err,2) && err.find("exactly 2") != std::string::npos);
        VJ_CHECK(rt.kfd_fd_ == -1 && rt.queues_.empty() && rt.vram_arenas_.empty() && rt.all_gpu_ids_.empty());
        VJ_CHECK(rt.devices_.size() == 3);
        setenv("VJ_GPU_DEVICES","0000:03:00.0",1);
        VJ_CHECK(!rt.init(0,0,&err,2) && err.find("found 1") != std::string::npos);
        VJ_CHECK(rt.kfd_fd_ == -1 && rt.devices_.size() == 1 && rt.devices_[0].gpu_id == 1);
        // Exercise the complete healthy init order with fake device fds and ioctls, not just helper logic.
        setenv("VJ_GPU_DEVICES","0000:03:00.0,0000:43:00.0",1);
        io_ok=true; fake_devices=true; map_done=unmap_done=32;
        allocation_owners.clear(); mapping_targets.clear();
        if (!rt.init(0,0,&err,2)) { std::fprintf(stderr,"healthy init: %s\n",err.c_str()); std::exit(1); }
        VJ_CHECK(rt.open_ && rt.devices_.size()==2 && rt.all_gpu_ids_==std::vector<uint32_t>({1,2}));
        VJ_CHECK(!allocation_owners.empty() && !mapping_targets.empty());
        for (auto id : allocation_owners) VJ_CHECK(id==1 || id==2);
        for (auto id : mapping_targets) VJ_CHECK(id==1 || id==2);
        for (const auto& q : rt.queues_) VJ_CHECK(q.queue_active && q.kernarg_capacity==2048*512);
        rt.shutdown();
        VJ_CHECK(rt.queues_.empty() && rt.retained_buffers_.empty() && rt.kfd_fd_==-1);
        io_ok=false; fake_devices=false; free_calls=0;
        unsetenv("VJ_SYSFS_ROOT"); unsetenv("VJ_GPU_DEVICES");
        fs::remove_all(root);
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
    vegajet::kfd::RuntimeTestAccess::exact_tp_selection();
    vegajet::kfd::RuntimeTestAccess::mailbox_reset();
    vegajet::kfd::RuntimeTestAccess::teardown_failures();
    std::puts("runtime fail-closed tests PASS");
}
