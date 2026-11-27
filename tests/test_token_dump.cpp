// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// `vegajet bench --dump-tokens` writer (core/token_dump.hpp): the format tools/token_compare.py parses, and the
// fail-closed behaviour when the path is not writable. Host-only, so it is checked without a GPU.
#include "core/token_dump.hpp"
#include "tests/test_common.hpp"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace vegajet;

int main() {
    const std::string path = "test_token_dump.tokens";
    // 1. Round trip: one id per line after the '#' header, negatives kept (a special token id can be negative).
    {
        const std::vector<int32_t> ids{760, 6511, 314, -1, 0};
        std::string err;
        VJ_CHECK(bench_io::write_token_ids(path, ids, &err));
        VJ_CHECK(err.empty());
        std::ifstream in(path);
        VJ_CHECK(in.good());
        std::string line;
        VJ_CHECK(static_cast<bool>(std::getline(in, line)));
        VJ_CHECK(line.size() > 1 && line[0] == '#');
        size_t n = 0;
        while (std::getline(in, line)) {
            VJ_CHECK(n < ids.size());
            VJ_CHECK(std::stoi(line) == ids[n]);
            ++n;
        }
        VJ_CHECK(n == ids.size());
        std::remove(path.c_str());
    }
    // 2. Empty sequence: header only, still a legal dump (token_compare reports 0/0 match).
    {
        std::string err;
        VJ_CHECK(bench_io::write_token_ids(path, {}, &err));
        std::ifstream in(path);
        VJ_CHECK(in.good());
        std::string line;
        VJ_CHECK(static_cast<bool>(std::getline(in, line)));
        VJ_CHECK(line.size() > 1 && line[0] == '#');
        VJ_CHECK(!std::getline(in, line));
        std::remove(path.c_str());
    }
    // 3. Unwritable path fails closed with an error instead of a silent no-op.
    {
        std::string err;
        VJ_CHECK(!bench_io::write_token_ids("no-such-dir/tokens.tokens", {1}, &err));
        VJ_CHECK(!err.empty());
    }
    std::printf("test_token_dump: ok\n");
    return 0;
}
