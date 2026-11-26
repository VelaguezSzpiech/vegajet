// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Generated token-id dumps: `vegajet bench --dump-tokens FILE` writes the ids the greedy driver emitted, and
// tools/token_compare.py diffs two dumps (reference vs candidate) for the all-off / per-toggle parity check.
// Format: '#' comment lines, then one decimal token id per line. Host-only, header-only so the writer and the
// tool share one documented format.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace vegajet::bench_io {

// Write `ids` (one per line) to `path`. Returns false and fills `err` on any I/O failure so the caller can fail
// the run instead of silently reporting throughput without a token dump.
inline bool write_token_ids(const std::string& path, const std::vector<int32_t>& ids, std::string* err) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        if (err) *err = "cannot open token dump file " + path;
        return false;
    }
    bool ok = std::fprintf(f, "# vegajet token ids: %zu tokens, one id per line (tools/token_compare.py)\n",
                           ids.size()) >= 0;
    for (size_t i = 0; ok && i < ids.size(); ++i) ok = std::fprintf(f, "%d\n", static_cast<int>(ids[i])) >= 0;
    if (std::fclose(f) != 0) ok = false;
    if (!ok && err) *err = "short write to token dump file " + path;
    return ok;
}

} // namespace vegajet::bench_io
