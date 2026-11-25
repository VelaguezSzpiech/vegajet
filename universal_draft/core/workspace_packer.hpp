#pragma once
#include "core/universal_ir.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace universal {
struct SymbolicShape { uint64_t bytes_per_token=0; uint32_t element_bytes=1; bool scales_with_seq=false, scales_with_m=false; };
struct BufferRequest { std::string name; size_t birth=0, death=0; SymbolicShape shape; };
struct BufferPlacement { std::string name; uint64_t offset=0, bytes=0; size_t birth=0, death=0; };
struct WorkspacePlan { uint64_t arena_bytes=0, naive_bytes=0; std::vector<BufferPlacement> buffers; std::string error; bool valid() const { return error.empty(); } };
struct WorkspaceShape { uint32_t m=1; uint64_t seq_len=512; };
WorkspacePlan pack_workspace(const std::vector<BufferRequest>& requests,const WorkspaceShape& shape,uint64_t alignment=2ull*1024*1024);
WorkspacePlan pack_workspace(const Model& model,const WorkspaceShape& shape,uint64_t alignment=2ull*1024*1024);
bool workspace_overlap_free(const WorkspacePlan& plan,size_t step);
} // namespace universal
