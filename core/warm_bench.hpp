// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Warm interactive / multi-variant sweep protocol for VegaJet (`vegajet bench --interactive` & `--sweep`).
// Pure host logic (zero GPU/KFD dependency) so it is unit-tested in both CPU-only and GPU builds.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace vegajet::warm {

enum class CommandKind {
    Empty,
    Reload,
    Run,
    Parity,
    Sweep,
    Status,
    Help,
    Quit,
    Invalid,
};

struct WarmCommand {
    CommandKind kind = CommandKind::Empty;
    std::string module_name;                          // for Reload (empty or "all" => reload all)
    std::vector<std::string> args;                    // for Run / Parity
    std::vector<std::vector<std::string>> variants;   // for Sweep
    std::string raw_line;
    std::string error;
};

struct EnvAssignment {
    std::string key;
    std::string value;
};

// Tokenizes a command line respecting single/double quotes and '#' comments.
bool tokenize_line(const std::string& line, std::vector<std::string>& out, std::string* err = nullptr);

// Parses a single warm interactive protocol line:
//   reload [module|all]
//   run [KEY=VAL ...] [bench_flags...]
//   parity [KEY=VAL ...] [bench_flags...]
//   sweep <variantA> ; <variantB> [; <variantC> ...]
//   status | help | quit | exit
WarmCommand parse_warm_command(const std::string& line);

// Parses a semicolon-delimited `--sweep` specification into tokenized variants.
bool parse_sweep_spec(const std::string& spec,
                      std::vector<std::vector<std::string>>& out_variants,
                      std::string* err = nullptr);

// Splits inline `KEY=VAL` environment overrides out of an argument vector.
void extract_env_overrides(const std::vector<std::string>& in_args,
                           std::vector<std::string>& out_cli_args,
                           std::vector<EnvAssignment>& out_env);

// RAII guard that applies environment overrides for one run/variant and restores exact prior state on destruction.
class ScopedEnvOverrides {
public:
    explicit ScopedEnvOverrides(const std::vector<EnvAssignment>& overrides);
    ~ScopedEnvOverrides();
    ScopedEnvOverrides(const ScopedEnvOverrides&) = delete;
    ScopedEnvOverrides& operator=(const ScopedEnvOverrides&) = delete;

private:
    struct SavedVar {
        std::string key;
        bool had_value = false;
        std::string old_value;
    };
    std::vector<SavedVar> saved_;
};

struct SweepVariantResult {
    std::string label;
    bool        ok = false;
    bool        parity_ok = true;
    double      raw_tg128_tps = 0.0;
    double      spec_tg128_tps = 0.0;
    double      pp512_tps = 0.0;
    double      pp2048_tps = 0.0;
    double      primary_tps = 0.0;   // primary metric used for A/B delta comparison
    double      tmem_max_c = 0.0;
    uint64_t    ecc_ue_delta = 0;
    uint64_t    ecc_de_delta = 0;
    uint32_t    dispatches_per_tok = 0;
    uint32_t    lossless_matched = 0;
    uint32_t    lossless_total = 0;
    int32_t     first_diff = -1;
    std::string error;
};

// Formats a side-by-side A/B delta table against variant 0 as baseline.
std::string format_sweep_table(const std::vector<SweepVariantResult>& results);

// Formats a compact single-line JSON array/summary for machine parsing.
std::string format_sweep_json(const std::vector<SweepVariantResult>& results);

// Escapes a string for JSON emission.
std::string json_escape(const std::string& s);

} // namespace vegajet::warm
