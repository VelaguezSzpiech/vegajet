// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Exercise the live host runtime with fake ioctls: no GPU or device files required.
#include "core/kfd_runtime.hpp"
#include <sys/ioctl.h>
#include <linux/kfd_ioctl.h>
#include "tests/test_common.hpp"
#include <cerrno>
#include <cstdarg>
#include <cstdio>

extern "C" int __wrap_ioctl(int, unsigned long, ...) { errno = EIO; return -1; }

namespace vegajet::kfd {
struct RuntimeTestAccess {
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
    std::puts("runtime fail-closed tests PASS");
}
