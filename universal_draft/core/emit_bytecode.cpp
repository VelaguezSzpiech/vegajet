#include "core/emit_bytecode.hpp"

namespace universal {
namespace {
bool fail(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
}
const char* kind_name(OpKind kind) {
    switch (kind) {
    case OpKind::Linear: return "Linear";
    case OpKind::RMSNorm: return "RMSNorm";
    case OpKind::ResidualAdd: return "ResidualAdd";
    case OpKind::AllReduce: return "AllReduce";
    case OpKind::RoPE: return "RoPE";
    case OpKind::QKNorm: return "QKNorm";
    case OpKind::PagedKVWrite: return "PagedKVWrite";
    case OpKind::AttentionGQA: return "AttentionGQA";
    case OpKind::AttentionMLA: return "AttentionMLA";
    case OpKind::SSMRecurrence: return "SSMRecurrence";
    case OpKind::MoERouter: return "MoERouter";
    case OpKind::MoEFFN: return "MoEFFN";
    case OpKind::ActMul: return "ActMul";
    case OpKind::Quantize: return "Quantize";
    case OpKind::GridBarrier: return "GridBarrier";
    }
    return "Unknown";
}
bool compatible(OpKind kind, uint32_t opcode) {
    using namespace vegajet::vm;
    switch (kind) {
    case OpKind::AttentionMLA: return opcode == LatentAttention;
    case OpKind::SSMRecurrence: return opcode == GatedDelta;
    case OpKind::PagedKVWrite: return opcode == PagedKvStore;
    case OpKind::RMSNorm: return opcode == RmsF32;
    case OpKind::Linear: return opcode == GemvQ4A8 || opcode == GemvQ8Dp4a || opcode == GemvQ4Dot8 || opcode == GemvF32 || opcode == MoeGateUpF32;
    case OpKind::ResidualAdd: return opcode == ArResRmsQ8 || opcode == ResidualAddF32;
    case OpKind::AllReduce: return opcode == ArResRmsQ8;
    case OpKind::QKNorm: return opcode == RmsF32 || opcode == QkNormRopeKv;
    case OpKind::RoPE: return opcode == QkNormRopeKv;
    case OpKind::MoEFFN: return opcode == MoeDownF32;
    case OpKind::Quantize: return opcode == QuantizeF32;
    case OpKind::AttentionGQA: return opcode == FlashDecodeGqa;
    case OpKind::MoERouter: return opcode == MoeRouterTopK;
    case OpKind::ActMul: return opcode == ActMulQ8 || opcode == ActMulF32;
    case OpKind::GridBarrier: return opcode == GridBarrier;
    default: return false;
    }
}
bool bind_one(const Model& model, size_t index, DescriptorBinder binder, void* context,
              std::vector<vegajet::vm::VjVmOpDesc>& output, std::string* error) {
    using namespace vegajet::vm;
    VjVmOpDesc desc{};
    if (!binder(model.ops[index], index, desc, context))
        return fail(error, "descriptor binding failed at graph node " + std::to_string(index));
    if (!compatible(model.ops[index].kind, desc.opcode))
        return fail(error, "descriptor opcode does not implement " + std::string(kind_name(model.ops[index].kind)));
    if (desc.opcode == ArResRmsQ8 && model.ops[index].kind == OpKind::AllReduce &&
        model.ops[index].tag != "fused_ar_res_rms_quant")
        return fail(error, "AllReduce must be host-decomposed before rank-local opcode 1");
    if(desc.opcode==ArResRmsQ8 && model.ops[index].kind==OpKind::ResidualAdd &&
       model.ops[index].tag!="fused_local_res_rms_quant")
        return fail(error,"combined residual/RMS/quant opcode requires a fused group");
    if(model.ops[index].kind==OpKind::RoPE && !(desc.flags&SkipRms))
        return fail(error,"standalone RoPE descriptor must set SkipRms");
    if(model.ops[index].kind==OpKind::QKNorm && desc.opcode==QkNormRopeKv &&
       model.ops[index].tag!="fused_qkv_rope_kv")
        return fail(error,"combined QKNorm/RoPE opcode requires a fused group");
    desc.flags &= ~(NonTemporalWeights | DynamicCacheBypass);
    if (model.ops[index].glc && model.ops[index].slc) desc.flags |= NonTemporalWeights;
    output.push_back(desc);
    return true;
}
bool shared_opcode(uint32_t opcode) {
    return opcode == vegajet::vm::ArResRmsQ8 || opcode == vegajet::vm::GemvQ4A8;
}
bool kernel_owner(uint32_t opcode, DeviceKernel& owner) {
    using namespace vegajet::vm;
    switch (opcode) {
    case ArResRmsQ8: case GemvQ4A8:
    case QkNormRopeKv: case FlashDecodeGqa: case GemvQ8Dp4a:
        owner = DeviceKernel::Attention; return true;
    case MoeRouterTopK: case ActMulQ8: case GemvQ4Dot8:
        owner = DeviceKernel::Ffn; return true;
    case GridBarrier: // Host-only; never submitted to a device kernel.
    case PagedKvStore: case MoeExpertCombine: case RmsF32: case GatedDelta:
    case LatentAttention: case ResidualAddF32: case QuantizeF32: case GemvF32:
    case ActMulF32: case MoeGateUpF32: case MoeDownF32: case ConvertF32ToF16:
        owner = DeviceKernel::Aux; return true;
    default: return false;
    }
}
}
bool emit_bytecode(const Model& model, DescriptorBinder binder, void* context,
                   std::vector<vegajet::vm::VjVmOpDesc>& output, std::string* error) {
    if (error) error->clear();
    if (!model.valid()) return fail(error, "cannot emit invalid graph");
    if (!binder) return fail(error, "missing descriptor binder");
    std::vector<vegajet::vm::VjVmOpDesc> result;
    for (size_t i = 0; i < model.ops.size(); ++i) {
        const auto& op = model.ops[i];
        if(op.tag=="fused_local_res_rms_quant") {
            if(i+2>=model.ops.size() || op.kind!=OpKind::ResidualAdd || model.ops[i+1].kind!=OpKind::RMSNorm || model.ops[i+2].kind!=OpKind::Quantize || model.ops[i+1].tag!=op.tag || model.ops[i+2].tag!=op.tag)return fail(error,"malformed local fused group");
            if(!bind_one(model,i,binder,context,result,error))return false;
            if(result.back().opcode==vegajet::vm::ResidualAddF32 &&
               (!bind_one(model,i+1,binder,context,result,error)||!bind_one(model,i+2,binder,context,result,error)))return false;
            i+=2;continue;
        }
        if (op.tag == "fused_ar_res_rms_quant") {
            if (op.kind != OpKind::AllReduce || i + 3 >= model.ops.size() ||
                model.ops[i + 1].tag != op.tag || model.ops[i + 2].tag != op.tag || model.ops[i + 3].tag != op.tag ||
                model.ops[i + 1].kind != OpKind::ResidualAdd || model.ops[i + 2].kind != OpKind::RMSNorm ||
                model.ops[i + 3].kind != OpKind::Quantize)
                return fail(error, "malformed AllReduce/ResidualAdd/RMSNorm/Quantize fusion");
            if (!bind_one(model, i, binder, context, result, error)) return false;
            i += 3; // Host graph collective precedes this rank-local descriptor.
            continue;
        }
        if (op.tag == "fused_qkv_rope_kv" && (op.kind == OpKind::QKNorm || op.kind == OpKind::RoPE)) {
            size_t rope = i;
            if (model.ops[rope].kind == OpKind::QKNorm) ++rope;
            if (rope >= model.ops.size() || model.ops[rope].tag != op.tag || model.ops[rope].kind != OpKind::RoPE)
                return fail(error, "malformed QKNorm/RoPE/PagedKVWrite fusion");
            const size_t kv = rope + 1;
            if (kv >= model.ops.size() || model.ops[kv].tag != op.tag || model.ops[kv].kind != OpKind::PagedKVWrite)
                return fail(error, "malformed QKNorm/RoPE/PagedKVWrite fusion");
            if(!bind_one(model,i,binder,context,result,error))return false;
            if(i!=rope && result.back().opcode==vegajet::vm::RmsF32 && !bind_one(model,rope,binder,context,result,error))return false;
            if(!bind_one(model,kv,binder,context,result,error))return false;
            i=kv;continue;
        }
        if (!bind_one(model, i, binder, context, result, error)) return false;
    }
    output.swap(result);
    return true;
}
bool plan_device_segments(const std::vector<vegajet::vm::VjVmOpDesc>& program,
                          std::vector<DeviceSegment>& segments, std::string* error) {
    using namespace vegajet::vm;
    if (error) error->clear();
    std::vector<DeviceKernel> owners(program.size());
    DeviceKernel previous = DeviceKernel::Attention;
    for (size_t i = 0; i < program.size(); ++i) {
        const auto& op = program[i];
        if (!kernel_owner(op.opcode, owners[i]))
            return fail(error, "unknown device opcode at descriptor " + std::to_string(i));
        if (shared_opcode(op.opcode)) owners[i] = previous;
        else previous = owners[i] == DeviceKernel::Aux ? DeviceKernel::Attention : owners[i];
        if (op.flags & BarrierAfter) previous = DeviceKernel::Attention;
    }
    DeviceKernel next = DeviceKernel::Attention;
    bool have_next = false;
    for (size_t i = program.size(); i > 0;) {
        --i;
        const auto& op = program[i];
        if (op.flags & BarrierAfter) have_next = false;
        if (shared_opcode(op.opcode)) {
            if (have_next) owners[i] = next;
        } else if (owners[i] == DeviceKernel::Aux) {
            have_next = false;
        } else {
            next = owners[i];
            have_next = true;
        }
    }
    std::vector<DeviceSegment> result;
    result.reserve(program.size());
    for (size_t i = 0; i < program.size(); ++i) {
        if (program[i].opcode == GridBarrier) continue;
        // ponytail: one op per ordered launch until a dependency proof permits fusion.
        result.push_back({owners[i], i, 1});
    }
    segments.swap(result);
    return true;
}

bool emit_device_plan(const Model& model, DescriptorBinder binder, void* context,
                      std::vector<vegajet::vm::VjVmOpDesc>& program,
                      std::vector<DeviceSegment>& segments, std::string* error) {
    std::vector<vegajet::vm::VjVmOpDesc> emitted;
    std::vector<DeviceSegment> planned;
    if (!emit_bytecode(model, binder, context, emitted, error) ||
        !plan_device_segments(emitted, planned, error)) return false;
    program.swap(emitted);
    segments.swap(planned);
    return true;
}
} // namespace universal
