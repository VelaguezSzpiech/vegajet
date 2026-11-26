// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/raw_token_reference.hpp"
#include "tests/test_common.hpp"
#include <algorithm>

int main() {
    char name[] = "/tmp/vj_raw_reference_XXXXXX";
    const int fd = mkstemp(name);
    VJ_CHECK(fd >= 0);
    close(fd);
    const std::string path(name);
    const std::vector<int32_t> baseline{17, 12, 400, 42};
    std::vector<int32_t> ref;
    std::string error;
    VJ_CHECK(vegajet::warm::raw_token_reference("save:" + path, baseline, ref, error));
    VJ_CHECK(ref == baseline);
    // Simulated new engine has different output: saved baseline must survive, never self-seed.
    const std::vector<int32_t> variant{17, 13, 400, 42};
    VJ_CHECK(vegajet::warm::raw_token_reference("check:" + path, variant, ref, error));
    VJ_CHECK(ref == baseline && ref != variant);
    VJ_CHECK(!vegajet::warm::raw_token_reference("check:" + path, {17}, ref, error));
    VJ_CHECK(!vegajet::warm::raw_token_reference(path, baseline, ref, error));
    for (const char* data : {"bad 4\n17\n12\n400\n42\n", "VJ_RAW_CODE_TOKENS_V1 4\n17\n",
                              "VJ_RAW_CODE_TOKENS_V1 4\n17\n-1\n400\n42\n",
                              "VJ_RAW_CODE_TOKENS_V1 4\n17\n12\n400\n42\nextra"}) {
        std::ofstream(path) << data;
        VJ_CHECK(!vegajet::warm::raw_token_reference("check:" + path, baseline, ref, error));
    }
    unlink(name);
    VJ_CHECK(!vegajet::warm::raw_token_reference("check:" + path, baseline, ref, error));
    VJ_CHECK(access(name, F_OK) != 0);
    std::puts("PASS persisted raw-token baseline across engine restarts");
}
