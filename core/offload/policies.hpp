// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include <cstdint>
#include <vector>
namespace vegajet::offload {
// Metadata only; the caller synchronously evicts before giving pages to KV.
struct Lease { uint32_t expert; uint64_t first_page, pages; };
class BorrowedKv {
    uint64_t pages_, chunk_pages_, step_=0;
    double decay_;
    std::vector<double> counts_;
    std::vector<Lease> leases_;
public:
    // Fixed 512 MiB chunks; page_bytes must divide 512 MiB.
    BorrowedKv(uint32_t chunks,uint64_t page_bytes,uint32_t experts,double decay=0.95);
    void route(uint32_t expert,double weight=1.0);
    // Returns current leases. KV occupies [0,kv_pages); reserve is also unavailable.
    // Shrinks immediately every step; reranks every eighth decode step.
    const std::vector<Lease>& decode_step(uint64_t kv_pages,uint64_t reserve_pages,uint64_t expert_pages);
    const std::vector<double>& counts() const { return counts_; }
};
enum class FetchPrecision { Full, Low, Skip };
struct FetchPlan { uint32_t expert, rank; FetchPrecision precision; };
// Misses only, rank zero = top-1. threshold is in [0,1]; no I/O performed here.
FetchPlan hobbit_plan(uint32_t expert,uint32_t rank,double weight,double tail_threshold);
}
