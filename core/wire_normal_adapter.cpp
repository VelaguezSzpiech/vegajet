// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/wire_normal_adapter.hpp"

#include <cstdio>
#include <cstring>

namespace vegajet::wire_normal {
namespace {

const char* fallback_name(Fallback fallback) {
    switch (fallback) {
    case Fallback::None: return "none";
    case Fallback::NotRequested: return "not_requested";
    case Fallback::Dimensions: return "dimensions";
    case Fallback::BatchRows: return "batch_rows";
    case Fallback::CacheRows: return "cache_rows";
    case Fallback::Format: return "format";
    case Fallback::AttentionScale: return "attention_scale";
    case Fallback::Policy: return "policy";
    case Fallback::SmallContext: return "small_context";
    case Fallback::ScratchBudget: return "scratch_budget";
    case Fallback::ProducerRange: return "producer_range";
    case Fallback::BufferCapacity: return "buffer_capacity";
    case Fallback::ProducerMissing: return "producer_missing";
    case Fallback::LaunchFailed: return "launch_failed";
    }
    return "unknown";
}

const kfd::LoadedKernel* q_kernel(const Kernels& kernels, WireFormat format) {
    return format == WireFormat::Q4 ? kernels.q4_q : kernels.q8_q;
}
const kfd::LoadedKernel* kv_kernel(const Kernels& kernels, WireFormat format) {
    return format == WireFormat::Q4 ? kernels.q4_kv : kernels.q8_kv;
}
const kfd::LoadedKernel* partition_kernel(const Kernels& kernels, WireFormat format) {
    return format == WireFormat::Q4 ? kernels.sealed_q4_partition : kernels.sealed_q8_partition;
}

bool loaded_for(const Kernels& kernels, WireFormat format) {
    return q_kernel(kernels, format) && kv_kernel(kernels, format) &&
           partition_kernel(kernels, format) && kernels.sealed_merge && kernels.epilogue;
}

void record_fallback(Counters& counters, const Attempt& attempt, uint32_t rows, uint32_t pos0,
                     WireFormat format) {
    const size_t index = static_cast<size_t>(attempt.fallback) < kFallbackCount
                             ? static_cast<size_t>(attempt.fallback)
                             : static_cast<size_t>(Fallback::Policy);
    ++counters.fallbacks[index];
    if (!counters.printed[index]) {
        counters.printed[index] = true;
        std::fprintf(stderr,
                     "[wire-attn] fallback=%s requested=%s M=%u pos0=%u reason=%s\n",
                     fallback_name(attempt.fallback),
                     format == WireFormat::Q4 ? "Q4" : (format == WireFormat::Q8 ? "Q8" : "none"),
                     rows, pos0, attempt.reason ? attempt.reason : "");
    }
}

} // namespace

Attempt run(kfd::Runtime& rt, uint32_t gpu, const Kernels& kernels, const Buffers& buffers,
            const Policy& policy, uint32_t layer, uint32_t pos0, uint32_t rows,
            uint64_t q_f32, uint64_t gate_f32, uint64_t out_q8, uint64_t out_ds,
            CacheState& cache, Counters& counters, std::string* error) {
    const WireFormat format = policy.allow_q4 ? WireFormat::Q4 : WireFormat::Q8;
    const flash_decode_splitkv::Shape shape = flash_decode_splitkv::active_shape(rows, pos0);
    Capacities capacities;
    capacities.cache_rows = buffers.cache_rows;
    capacities.q_bytes = buffers.q_bytes;
    capacities.scratch_bytes = buffers.scratch_bytes;
    const bool loaded = loaded_for(kernels, format);
    const Attempt attempt = decide(shape, format, cache.get(layer), capacities, loaded, policy);
    ++counters.attempts;

    if (!attempt.engaged) {
        record_fallback(counters, attempt, rows, pos0, format);
        if (error) *error = std::string("wire attention fallback: ") +
                            fallback_name(attempt.fallback) + " (" +
                            (attempt.reason ? attempt.reason : "") + ")";
        return attempt;
    }

    if (buffers.q_wire == 0u || buffers.k_wire == 0u || buffers.v_wire == 0u ||
        buffers.scratch == 0u || buffers.out == 0u || buffers.out_lse == 0u || q_f32 == 0u ||
        gate_f32 == 0u || out_q8 == 0u || out_ds == 0u) {
        Attempt failed = attempt;
        failed.engaged = false;
        failed.fallback = Fallback::BufferCapacity;
        failed.reason = "a required device buffer pointer is null";
        record_fallback(counters, failed, rows, pos0, format);
        if (error) *error = "wire attention fallback: null device buffer";
        return failed;
    }

    const auto enqueue = [&](const kfd::LoadedKernel* kernel, uint32_t grid_x, uint32_t grid_y,
                             uint32_t grid_z, const void* args, size_t bytes) -> bool {
        if (!kernel || grid_x == 0u || grid_y == 0u || grid_z == 0u) return false;
        return rt.enqueue_kernel(gpu, *kernel, grid_x, grid_y, grid_z,
                                 kProducerThreads, 1u, 1u, args, bytes, true);
    };

    if (attempt.kv_dispatches != 0u) {
        const KvProducerArgs args{buffers.native_k, buffers.native_v, buffers.k_wire,
                                  buffers.v_wire, attempt.kv_begin, attempt.kv_end, kKvHeads};
        if (!enqueue(kv_kernel(kernels, format), attempt.kv_end - attempt.kv_begin, 1u, 1u,
                     &args, sizeof(args))) {
            Attempt failed = attempt;
            failed.engaged = false;
            failed.fallback = Fallback::LaunchFailed;
            failed.reason = "KV producer enqueue failed";
            record_fallback(counters, failed, rows, pos0, format);
            if (error) *error = "wire attention fallback: KV producer enqueue failed";
            return failed;
        }
        // Rows [begin,end) are now guaranteed to be refreshed by the producer in queue order.
        cache.set(layer, attempt.kv_end);
    }

    QProducerArgs q_args{};
    q_args.q_f32 = q_f32;
    q_args.q_wire = buffers.q_wire;
    q_args.rows = rows;
    q_args.heads = kHeads;
    if (!enqueue(q_kernel(kernels, format), attempt.q_grid_x, 1u, 1u, &q_args, sizeof(q_args))) {
        Attempt failed = attempt;
        failed.engaged = false;
        failed.fallback = Fallback::LaunchFailed;
        failed.reason = "Q producer enqueue failed";
        record_fallback(counters, failed, rows, pos0, format);
        if (error) *error = "wire attention fallback: Q producer enqueue failed";
        return failed;
    }

    const auto& scratch = attempt.scratch;
    const flash_decode_splitkv::PartitionArgs part = flash_decode_splitkv::partition_args(
        buffers.q_wire, buffers.k_wire, buffers.v_wire,
        buffers.scratch + scratch.partial_offset, buffers.scratch + scratch.lse_offset,
        shape, attempt.splits, policy.attention_scale);
    const flash_decode_splitkv::MergeArgs merge = flash_decode_splitkv::merge_args(
        buffers.scratch + scratch.partial_offset, buffers.scratch + scratch.lse_offset,
        buffers.out, buffers.out_lse, attempt.splits);
    const flash_decode_splitkv::LaunchGeometry partition_geometry =
        flash_decode_splitkv::partition_geometry(attempt.splits, rows);
    const flash_decode_splitkv::LaunchGeometry merge_geometry =
        flash_decode_splitkv::merge_geometry(rows);
    if (!enqueue(partition_kernel(kernels, format), partition_geometry.grid_x,
                 partition_geometry.grid_y, partition_geometry.grid_z, &part, sizeof(part))) {
        Attempt failed = attempt;
        failed.engaged = false;
        failed.fallback = Fallback::LaunchFailed;
        failed.reason = "sealed partition enqueue failed";
        record_fallback(counters, failed, rows, pos0, format);
        if (error) *error = "wire attention fallback: sealed partition enqueue failed";
        return failed;
    }
    if (!enqueue(kernels.sealed_merge, merge_geometry.grid_x, merge_geometry.grid_y,
                 merge_geometry.grid_z, &merge, sizeof(merge))) {
        Attempt failed = attempt;
        failed.engaged = false;
        failed.fallback = Fallback::LaunchFailed;
        failed.reason = "sealed merge enqueue failed";
        record_fallback(counters, failed, rows, pos0, format);
        if (error) *error = "wire attention fallback: sealed merge enqueue failed";
        return failed;
    }

    EpilogueArgs epilogue{};
    epilogue.attention = buffers.out;
    epilogue.gate = gate_f32;
    epilogue.out_q8 = out_q8;
    epilogue.out_ds = out_ds;
    epilogue.rows = rows;
    epilogue.heads = kHeads;
    if (!enqueue(kernels.epilogue, attempt.epilogue_grid_x, 1u, 1u, &epilogue,
                 sizeof(epilogue))) {
        Attempt failed = attempt;
        failed.engaged = false;
        failed.fallback = Fallback::LaunchFailed;
        failed.reason = "epilogue enqueue failed";
        record_fallback(counters, failed, rows, pos0, format);
        if (error) *error = "wire attention fallback: epilogue enqueue failed";
        return failed;
    }

    ++counters.engaged;
    counters.q_dispatches += attempt.q_dispatches;
    counters.kv_dispatches += attempt.kv_dispatches;
    counters.partition_dispatches += attempt.partition_dispatches;
    counters.merge_dispatches += attempt.merge_dispatches;
    counters.epilogue_dispatches += attempt.epilogue_dispatches;
    counters.rows_produced += attempt.kv_dispatches != 0u ? (attempt.kv_end - attempt.kv_begin) : 0u;
    counters.last_splits = attempt.splits;
    counters.last_rows = rows;
    counters.last_format = format;
    counters.last_kv_begin = attempt.kv_begin;
    counters.last_kv_end = attempt.kv_end;
    if (error) error->clear();
    return attempt;
}

std::string summary(const Counters& counters) {
    uint64_t fallbacks = 0;
    for (size_t i = 0; i < kFallbackCount; ++i) fallbacks += counters.fallbacks[i];
    char line[512];
    std::snprintf(line, sizeof(line),
                  "attempts=%llu engaged=%llu q=%llu kv=%llu partition=%llu merge=%llu "
                  "epilogue=%llu rows_produced=%llu last=%s M=%u S=%u kv_rows=[%u,%u) "
                  "fallbacks=%llu",
                  static_cast<unsigned long long>(counters.attempts),
                  static_cast<unsigned long long>(counters.engaged),
                  static_cast<unsigned long long>(counters.q_dispatches),
                  static_cast<unsigned long long>(counters.kv_dispatches),
                  static_cast<unsigned long long>(counters.partition_dispatches),
                  static_cast<unsigned long long>(counters.merge_dispatches),
                  static_cast<unsigned long long>(counters.epilogue_dispatches),
                  static_cast<unsigned long long>(counters.rows_produced),
                  counters.last_format == WireFormat::Q4
                      ? "Q4"
                      : (counters.last_format == WireFormat::Q8 ? "Q8" : "none"),
                  counters.last_rows, counters.last_splits, counters.last_kv_begin,
                  counters.last_kv_end, static_cast<unsigned long long>(fallbacks));
    return std::string(line);
}

} // namespace vegajet::wire_normal
