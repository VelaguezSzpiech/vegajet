// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Always-on runtime check macro (never compiles out under -DNDEBUG).
#pragma once

#include <cstdio>
#include <cstdlib>

#ifndef VJ_CHECK
#define VJ_CHECK(expr)                                                                                  \
    do {                                                                                                \
        if (!(expr)) {                                                                                  \
            std::fprintf(stderr, "[VEGAJET FATAL] Check failed: %s (%s:%d)\n", #expr, __FILE__,         \
                         __LINE__);                                                                     \
            std::fflush(stderr);                                                                        \
            std::abort();                                                                               \
        }                                                                                               \
    } while (0)
#endif
