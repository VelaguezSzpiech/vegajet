#pragma once
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>
namespace vegajet::universal {
enum class Type : uint32_t { IQ1_S=19, IQ1_M=29, TQ1_0=34, TQ2_0=35, MXFP4=39 };
struct Format { uint32_t block_elems,block_bytes;const char* name; };
const Format* format(Type) noexcept;const Format* format(std::string_view) noexcept;
// IQ1 decoders require the caller-validated canonical signed table [2048][8].
bool decode(Type,const uint8_t*,size_t,float*,size_t,size_t,const float* iq1_codebook=nullptr,size_t codebook_values=0) noexcept;
enum class NativeId : uint32_t { VJ_Q4_A8, VJ_Q8_DP4A, VJ_Q4_DOT8, VJ_IQ_PACKED };
struct NativeFormat { uint32_t id;const char* name; };
const std::vector<NativeFormat>& native_formats();
struct Upgrade { uint64_t extra_bytes;double saved_ms; };
std::vector<size_t> select_upgrades(const std::vector<Upgrade>&,uint64_t capacity);
bool crossover_wins(double,double,double,double) noexcept;
double roi_ms_per_gib(double,uint64_t) noexcept;
double min_i_star(uint32_t,double,double,double) noexcept;
uint64_t profile_budget_bytes(uint32_t gib_tier) noexcept;
struct Strides { size_t qs_bytes,dm_bytes; };
Strides odd_plane_strides(size_t,size_t) noexcept;
void pack_odd_planes(const uint8_t*,size_t,const uint8_t*,size_t,size_t,uint8_t*,uint8_t*,Strides) noexcept;
void unpack_odd_planes(uint8_t*,size_t,uint8_t*,size_t,size_t,const uint8_t*,const uint8_t*,Strides) noexcept;
uint32_t plane_channel_spread(size_t,size_t,size_t,size_t=256,size_t=16) noexcept;
enum class KvTier { Q8_0,Q4_0,Tq3 };
struct Candidate { uint64_t extra; double old_bw,new_bw,old_bpw,new_bpw,saved_ms; };
struct BoardSample { uint32_t cu,cap_w; uint64_t usable_bytes; double clock_ghz,eta,hbm_gbs; };
// Inputs are per-rank measurements at the stated power cap, not SKU defaults.
double fleet_crossover(const std::vector<BoardSample>& ranks) noexcept;
Candidate bandwidth_candidate(uint64_t old_bytes,uint64_t new_bytes,double old_gbs,double new_gbs) noexcept;
struct Residency { bool fits; KvTier tier; uint64_t kv_bytes,upgrade_bytes; std::vector<size_t> chosen; };
// Exact KV block bytes are supplied by the layout provider (including metadata).
Residency co_budget(uint64_t free_after_fixed, uint64_t blocks, const uint32_t (&block_bytes)[3], const std::vector<Candidate>&);
struct KvPlan { uint64_t bytes;uint32_t tokens; };
KvPlan plan_profile_kv(uint64_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,double=0.9) noexcept;
double cosine_similarity(const float*,const float*,size_t) noexcept;
}
