#pragma once
#include "core/gguf_reader.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace universal {
enum class OpKind { Linear, RMSNorm, ResidualAdd, AllReduce, RoPE, QKNorm, PagedKVWrite, AttentionGQA, AttentionMLA, SSMRecurrence, MoERouter, MoEFFN, ActMul, Quantize, GridBarrier };
enum class Family { DenseGQA, PureMoE, HybridGDNMoE, SparseMLAMoE };
enum class CachePolicy { Default, Temporal, Streaming };
struct Shape { std::vector<uint64_t> dims; uint64_t elements() const; };
struct Tensor { std::string name; Shape shape; uint32_t type=0; };
struct Op { OpKind kind; std::vector<size_t> inputs; std::vector<size_t> outputs; std::string tag; CachePolicy cache=CachePolicy::Default; bool glc=false, slc=false; uint32_t split_k=1, grid_x=0, grid_y=0; };
struct Model { Family family=Family::DenseGQA; uint64_t input_value=0, hidden=0, layers=0, heads=0, kv_heads=0, experts=0, head_dim=0; std::vector<Tensor> tensors; std::vector<Op> ops; std::string error; bool valid() const { return error.empty() && !ops.empty(); } };
struct Options { uint32_t tp_size=1, n_cu=60; uint32_t m=1; bool prefill=false; };
Model lower_gguf(const vegajet::GgufFile& gguf, const Options& options={});
} // namespace universal
