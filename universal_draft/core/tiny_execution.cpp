#include "core/tiny_execution.hpp"
#include "core/emit_bytecode.hpp"
#include "ops/vj_vm_cpu.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace universal {
namespace {
using namespace vegajet::vm;
constexpr size_t width = 64, top = 2, experts = 4;
constexpr float epsilon = .000013f;
template<class T> uint64_t address(T* p) { return reinterpret_cast<uintptr_t>(p); }
float sample(size_t i, size_t seed) {
    return std::sin(float(i + 1) * .173f + float(seed + 1) * .319f);
}
// Independent round-to-nearest-even conversion for the tiny fixture's finite
// FP16 cache values (including subnormals); no VM conversion helper is reused.
uint16_t to_half(float x) {
    const uint16_t sign = std::signbit(x) ? 0x8000 : 0;
    const float a = std::fabs(x);
    if (a == 0) return sign;
    int exponent = 0;
    std::frexp(a, &exponent);
    int e = exponent - 1;
    if (e < -14) return uint16_t(sign | uint16_t(std::nearbyint(std::ldexp(a, 24))));
    if (e > 15) return uint16_t(sign | 0x7c00);
    int m = int(std::nearbyint(std::ldexp(a, 10 - e)));
    if (m == 2048) { m = 1024; ++e; }
    return uint16_t(sign | uint16_t((e + 15) << 10) | uint16_t(m - 1024));
}
float from_half(uint16_t x) {
    const int e = (x >> 10) & 31, m = x & 1023;
    const float a = e ? std::ldexp(float(1024 + m), e - 25) : std::ldexp(float(m), -24);
    return x & 0x8000 ? -a : a;
}
struct Value {
    std::vector<float> data;
    std::vector<uint32_t> ids;
    size_t route = 0;
};
struct Node {
    std::vector<float> weights, up, norm, state, scales;
    std::vector<int8_t> quant;
    std::vector<uint16_t> keys, values;
    uint32_t pages[2] = {1, 0};
    float gates[2] = {.973f, .031f};
    KvStoreDesc store{};
    VjVmFlashDesc attention{};
    LatentDesc latent{};
    DeltaDesc delta{};
    MoeF32Desc moe{};
    size_t route = 0;
    bool gate = false;
};
struct Execution {
    const Model& model;
    std::vector<Value> values;
    std::vector<Node> nodes;
    std::vector<size_t> producer;
    size_t last = 0;
    explicit Execution(const Model& m) : model(m), nodes(m.ops.size()) {
        size_t max_id = m.input_value;
        for (const auto& op : m.ops) for (size_t id : op.outputs) max_id = std::max(max_id, id);
        values.resize(max_id + 1);
        producer.assign(max_id + 1, m.ops.size());
        values[m.input_value].data.resize(width);
        for (size_t j = 0; j < width; ++j) values[m.input_value].data[j] = .37f + .71f * sample(j, 1);
        for (size_t i = 0; i < m.ops.size(); ++i) {
            const auto& op = m.ops[i];
            Node& n = nodes[i];
            const size_t out = op.outputs[0];
            producer[out] = i;
            n.norm.resize(width);
            for (size_t j = 0; j < width; ++j) n.norm[j] = .91f + .13f * sample(j, i);
            n.gate = op.kind == OpKind::Linear && i + 1 < m.ops.size() && m.ops[i + 1].kind == OpKind::ActMul;
            size_t count = width;
            n.route = values[op.inputs[0]].route;
            if (op.kind == OpKind::MoERouter) { n.route = out; count = top; values[out].ids.resize(top); }
            if (n.gate && m.experts > 1) n.route = op.inputs[1];
            values[out].route = n.route;
            if (n.gate) count = 2 * width * (m.experts > 1 ? top : 1);
            if (op.kind == OpKind::ActMul && m.experts > 1) count = top * width;
            values[out].data.assign(count, 0);
            n.quant.resize(width);
            n.scales.resize(width / 32);
            const size_t matrices = (n.gate || op.kind == OpKind::MoEFFN) && m.experts > 1 ? experts : 1;
            n.weights.resize(matrices * width * width);
            n.up.resize(matrices * width * width);
            for (size_t j = 0; j < n.weights.size(); ++j) {
                n.weights[j] = .029f * sample(j, i + 7);
                n.up[j] = .033f * sample(j, i + 19);
            }
            n.state.resize(width * width);
            for (size_t j = 0; j < n.state.size(); ++j) n.state[j] = .017f * sample(j, i + 5);
            n.keys.resize(4 * width / 2); n.values.resize(4 * width / 2);
            for (size_t j = 0; j < n.keys.size(); ++j) {
                n.keys[j] = to_half(.29f * sample(j, i + 11));
                n.values[j] = to_half(.43f * sample(j, i + 31));
            }
            if (op.kind != OpKind::GridBarrier) last = out;
        }
    }
};
bool bind(const Op& op, size_t index, VjVmOpDesc& d, void* opaque) {
    auto& e = *static_cast<Execution*>(opaque);
    auto& n = e.nodes[index];
    auto& y = e.values[op.outputs[0]].data;
    const auto& x = e.values[op.inputs[0]].data;
    if (x.size() < width) return false;
    if (op.kind == OpKind::ResidualAdd && e.values[op.inputs[1]].data.size() != width) return false;
    if (op.kind == OpKind::ActMul && x.size() != 2 * y.size()) return false;
    if ((n.gate && e.model.experts > 1) || op.kind == OpKind::MoEFFN) {
        if (e.values[n.route].ids.size() != top) return false;
        if (op.kind == OpKind::MoEFFN && x.size() != top * width) return false;
        if (n.gate && op.inputs.size() > 2 && e.values[op.inputs[2]].data.size() != width) return false;
    }
    if (op.kind == OpKind::AttentionGQA || op.kind == OpKind::AttentionMLA) {
        const size_t kv = e.producer[op.inputs[1]];
        if (kv >= e.model.ops.size() || e.model.ops[kv].kind != OpKind::PagedKVWrite) return false;
    }
    d.rows = 1; d.cols = width; d.eps = epsilon;
    switch (op.kind) {
    case OpKind::RMSNorm: case OpKind::QKNorm:
        d.opcode = RmsF32;
        if (op.kind == OpKind::QKNorm) { d.rows = 4; d.cols = 16; }
        d.ptr0 = address(x.data()); d.ptr1 = address(n.norm.data()); d.ptr2 = address(y.data()); break;
    case OpKind::ResidualAdd:
        d.opcode = ResidualAddF32; d.ptr0 = address(x.data());
        d.ptr1 = address(e.values[op.inputs[1]].data.data()); d.ptr2 = address(y.data()); break;
    case OpKind::Quantize:
        d.opcode = QuantizeF32; d.ptr0 = address(x.data()); d.ptr1 = address(n.quant.data());
        d.ptr2 = address(n.scales.data()); d.ptr3 = address(y.data()); break;
    case OpKind::Linear:
        if (n.gate && e.model.experts > 1) {
            n.moe.weights = address(n.weights.data()); n.moe.up_weights = address(n.up.data());
            n.moe.input = address(x.data()); n.moe.ids = address(e.values[n.route].ids.data());
            n.moe.output = address(y.data()); n.moe.experts = experts; n.moe.hidden = width;
            n.moe.intermediate = width; n.moe.top_k = top;
            if (op.inputs.size() > 2) n.moe.shared = address(e.values[op.inputs[2]].data.data());
            d.opcode = MoeGateUpF32; d.ptr0 = address(&n.moe);
        } else {
            // Dense gate/up is a single 128-row matrix.
            if (n.gate) { n.weights.insert(n.weights.end(), n.up.begin(), n.up.end()); d.rows = 2 * width; }
            else d.rows = width;
            d.opcode = GemvF32; d.ptr0 = address(n.weights.data()); d.ptr1 = address(x.data()); d.ptr2 = address(y.data());
        }
        break;
    case OpKind::ActMul:
        d.opcode = ActMulF32; d.rows = uint32_t(y.size() / width);
        d.ptr0 = address(x.data()); d.ptr1 = address(x.data() + y.size()); d.ptr2 = address(y.data()); break;
    case OpKind::MoERouter:
        d.opcode = MoeRouterTopK; d.cols = experts; d.aux0 = top;
        d.ptr0 = address(x.data()); d.ptr1 = address(e.values[op.outputs[0]].ids.data()); d.ptr2 = address(y.data()); break;
    case OpKind::MoEFFN:
        n.moe.weights = address(n.weights.data()); n.moe.input = address(x.data());
        n.moe.ids = address(e.values[n.route].ids.data()); n.moe.mixture = address(e.values[n.route].data.data());
        n.moe.output = address(y.data()); n.moe.experts = experts; n.moe.hidden = width;
        n.moe.intermediate = width; n.moe.top_k = top;
        d.opcode = MoeDownF32; d.ptr0 = address(&n.moe); break;
    case OpKind::RoPE:
        d.opcode = QkNormRopeKv; d.flags = SkipRms; d.rows = 4; d.cols = 16; d.aux0 = 2;
        d.ptr0 = address(x.data()); d.ptr1 = address(n.norm.data()); d.ptr2 = address(y.data()); break;
    case OpKind::PagedKVWrite:
        n.store = {address(x.data()), address(x.data() + width / 2), address(n.keys.data()),
            address(n.values.data()), address(n.pages), 2, 2, 2, 2, 16};
        d.opcode = PagedKvStore; d.flags = F32KvInput; d.ptr0 = address(&n.store); break;
    case OpKind::AttentionGQA: {
        auto& kv = e.nodes[e.producer[op.inputs[1]]];
        n.attention = {address(x.data()), address(kv.keys.data()), address(kv.values.data()), address(y.data()),
            0, 4, 2, 16, 2, 3, 2, 2, 0, address(kv.pages), .25f};
        d.opcode = FlashDecodeGqa; d.ptr0 = address(&n.attention); break;
    }
    case OpKind::AttentionMLA: {
        auto& kv = e.nodes[e.producer[op.inputs[1]]];
        // All four physical slots are valid: three historical latents and the
        // just-written token. Attention is invariant to their page ordering.
        n.latent = {address(x.data()), 0, address(kv.keys.data()), 0,
            address(n.weights.data()), address(y.data()), 1, 4, 32, 0, 64, .125f};
        d.opcode = LatentAttention; d.flags = F16LatentCache; d.ptr0 = address(&n.latent); break;
    }
    case OpKind::SSMRecurrence:
        n.delta = {address(x.data()), address(x.data()), address(x.data()), address(n.gates),
            address(n.state.data()), address(y.data()), 64, 64};
        d.opcode = GatedDelta; d.ptr0 = address(&n.delta); break;
    case OpKind::GridBarrier: d.opcode = GridBarrier; break;
    case OpKind::AllReduce: return false; // TP1 graphs must elide collectives before binding.
    }
    return true;
}

void linear(const std::vector<float>& w, const float* x, float* y, size_t rows, size_t offset = 0) {
    for (size_t r = 0; r < rows; ++r) {
        float sum = 0;
        for (size_t c = 0; c < width; ++c) sum += w[offset + r * width + c] * x[c];
        y[r] = sum;
    }
}
void reference(Execution& e) {
    for (size_t index = 0; index < e.model.ops.size(); ++index) {
        const auto& op = e.model.ops[index]; auto& n = e.nodes[index];
        const auto& x = e.values[op.inputs[0]].data;
        auto& y = e.values[op.outputs[0]].data;
        switch (op.kind) {
        case OpKind::RMSNorm: case OpKind::QKNorm: {
            const size_t dim = op.kind == OpKind::QKNorm ? 16 : width;
            for (size_t h = 0; h < width / dim; ++h) {
                float sum = 0;
                for (size_t j = 0; j < dim; ++j) sum += x[h * dim + j] * x[h * dim + j];
                const float inv = 1 / std::sqrt(sum / float(dim) + epsilon);
                for (size_t j = 0; j < dim; ++j) y[h * dim + j] = x[h * dim + j] * inv * n.norm[j];
            }
            break;
        }
        case OpKind::ResidualAdd:
            for (size_t j = 0; j < width; ++j) y[j] = x[j] + e.values[op.inputs[1]].data[j];
            break;
        case OpKind::Quantize:
            for (size_t b = 0; b < width / 32; ++b) {
                float maximum = 0;
                for (size_t j = 0; j < 32; ++j) maximum = std::max(maximum, std::fabs(x[b * 32 + j]));
                const float scale = maximum / 127, inv = maximum > 0 ? 127 / maximum : 0;
                for (size_t j = 0; j < 32; ++j) y[b * 32 + j] = std::nearbyint(x[b * 32 + j] * inv) * scale;
            }
            break;
        case OpKind::Linear:
            if (n.gate && e.model.experts > 1) {
                for (size_t s = 0; s < top; ++s) {
                    const size_t offset = e.values[n.route].ids[s] * width * width;
                    linear(n.weights, x.data(), y.data() + s * width, width, offset);
                    linear(n.up, x.data(), y.data() + (top + s) * width, width, offset);
                    if (op.inputs.size() > 2) for (size_t j = 0; j < width; ++j)
                        y[s * width + j] += e.values[op.inputs[2]].data[j];
                }
            } else {
                linear(n.weights, x.data(), y.data(), width);
                if (n.gate) linear(n.up, x.data(), y.data() + width, width);
            }
            break;
        case OpKind::ActMul:
            for (size_t j = 0; j < y.size(); ++j) y[j] = (x[j] / (1 + std::exp(-x[j]))) * x[j + y.size()];
            break;
        case OpKind::MoERouter: {
            std::vector<uint32_t> order(experts); std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return x[a] == x[b] ? a < b : x[a] > x[b]; });
            float sum = 0;
            for (size_t j = 0; j < top; ++j) { y[j] = std::exp(x[order[j]] - x[order[0]]); sum += y[j]; }
            for (size_t j = 0; j < top; ++j) { y[j] /= sum; e.values[op.outputs[0]].ids[j] = order[j]; }
            break;
        }
        case OpKind::MoEFFN: {
            float selected[width];
            std::fill(y.begin(), y.end(), 0);
            for (size_t s = 0; s < top; ++s) {
                linear(n.weights, x.data() + s * width, selected, width, e.values[n.route].ids[s] * width * width);
                for (size_t j = 0; j < width; ++j) y[j] += selected[j] * e.values[n.route].data[s];
            }
            break;
        }
        case OpKind::RoPE:
            for (size_t h = 0; h < 4; ++h) for (size_t j = 0; j < 8; ++j) {
                const float theta = 2 * std::pow(10000.f, -float(j) / 8);
                const float a = x[h * 16 + j], b = x[h * 16 + j + 8];
                y[h * 16 + j] = a * std::cos(theta) - b * std::sin(theta);
                y[h * 16 + j + 8] = a * std::sin(theta) + b * std::cos(theta);
            }
            break;
        case OpKind::PagedKVWrite:
            for (size_t j = 0; j < width / 2; ++j) { n.keys[j] = to_half(x[j]); n.values[j] = to_half(x[j + width / 2]); }
            break;
        case OpKind::AttentionGQA: {
            const auto& kv = e.nodes[e.producer[op.inputs[1]]];
            for (size_t h = 0; h < 4; ++h) {
                float scores[3], maximum = -std::numeric_limits<float>::infinity(), denominator = 0;
                for (size_t t = 0; t < 3; ++t) {
                    const size_t base = ((kv.pages[t / 2] * 2 + t % 2) * 2 + h / 2) * 16;
                    float sum = 0;
                    for (size_t j = 0; j < 16; ++j) sum += x[h * 16 + j] * from_half(kv.keys[base + j]);
                    scores[t] = sum * .25f; maximum = std::max(maximum, scores[t]);
                }
                for (float& score : scores) { score = std::exp(score - maximum); denominator += score; }
                for (size_t j = 0; j < 16; ++j) {
                    float sum = 0;
                    for (size_t t = 0; t < 3; ++t) {
                        const size_t base = ((kv.pages[t / 2] * 2 + t % 2) * 2 + h / 2) * 16;
                        sum += scores[t] * from_half(kv.values[base + j]);
                    }
                    y[h * 16 + j] = sum / denominator;
                }
            }
            break;
        }
        case OpKind::AttentionMLA: {
            const auto& kv = e.nodes[e.producer[op.inputs[1]]];
            float probability[4], maximum = -std::numeric_limits<float>::infinity(), denominator = 0;
            for (size_t t = 0; t < 4; ++t) {
                float score = 0;
                for (size_t j = 0; j < 32; ++j) score += x[j] * from_half(kv.keys[t * 32 + j]);
                probability[t] = score * .125f; maximum = std::max(maximum, probability[t]);
            }
            for (float& p : probability) { p = std::exp(p - maximum); denominator += p; }
            // Project each token first, then average: independent of the VM's
            // latent-average-then-project accumulation order.
            for (size_t r = 0; r < width; ++r) {
                float sum = 0;
                for (size_t t = 0; t < 4; ++t) {
                    float value = 0;
                    for (size_t j = 0; j < 32; ++j) value += n.weights[r * 32 + j] * from_half(kv.keys[t * 32 + j]);
                    sum += probability[t] * value;
                }
                y[r] = sum / denominator;
            }
            break;
        }
        case OpKind::SSMRecurrence:
            for (size_t c = 0; c < width; ++c) {
                float prediction = 0, sum = 0;
                for (size_t j = 0; j < width; ++j) prediction += n.state[j * width + c] * x[j];
                const float delta = n.gates[1] * (x[c] - n.gates[0] * prediction);
                for (size_t j = 0; j < width; ++j) {
                    auto& state = n.state[j * width + c]; state = n.gates[0] * state + x[j] * delta;
                    sum += state * x[j];
                }
                y[c] = sum;
            }
            break;
        case OpKind::GridBarrier: break;
        case OpKind::AllReduce: y = x; break; // unsharded reference: identity
        }
    }
}
bool fail(std::string* error, const std::string& message) { if (error) *error = message; return false; }
} // namespace

bool run_tiny_execution(const Model& model, TinyExecutionResult& result, std::string* error) {
    result = {}; if (error) error->clear();
    if (!model.valid() || model.ops.size() > 256 || model.input_value > 1024 ||
        model.hidden != width || model.layers != 2 || model.heads != 4 ||
        model.kv_heads != 2 || (model.experts != 0 && model.experts != experts))
        return fail(error, "tiny runner requires two layers, hidden64, Q4/KV2, zero or four experts");
    // Validate producer IDs before binding raw addresses. TP1 AllReduce is an
    // identity; retain all other nodes, tags, joins and value IDs unchanged.
    Model local = model;
    for (size_t i = 0; i < local.ops.size();) {
        const auto& op = local.ops[i];
        if (op.kind != OpKind::AllReduce) { ++i; continue; }
        if (op.inputs.size() != 1 || op.outputs.size() != 1) return fail(error, "malformed TP1 collective");
        const size_t from = op.inputs[0], to = op.outputs[0];
        for (auto& consumer : local.ops) for (auto& id : consumer.inputs) if (id == to) id = from;
        if(op.tag=="fused_ar_res_rms_quant"){if(i+3>=local.ops.size())return fail(error,"short collective fusion");for(size_t j=i+1;j<=i+3;++j)local.ops[j].tag="fused_local_res_rms_quant";}
        local.ops.erase(local.ops.begin() + static_cast<std::ptrdiff_t>(i));
    }
    std::vector<size_t> defined{local.input_value};
    for (const auto& op : local.ops) {
        if (op.inputs.empty() || op.outputs.size() != 1) return fail(error, "tiny graph requires inputs and one output per node");
        if (op.outputs[0] > 1024) return fail(error, "tiny graph value ID exceeds fixture bounds");
        const bool join = op.kind == OpKind::ResidualAdd || op.kind == OpKind::AttentionGQA || op.kind == OpKind::AttentionMLA;
        if (join && op.inputs.size() != 2) return fail(error, "tiny graph join requires two inputs");
        if (op.kind == OpKind::Linear && local.experts > 1) {
            const auto index = static_cast<size_t>(&op - local.ops.data());
            if (index + 1 < local.ops.size() && local.ops[index + 1].kind == OpKind::ActMul && op.inputs.size() < 2)
                return fail(error, "routed gate is missing router producer");
        }
        for (size_t id : op.inputs) if (std::find(defined.begin(), defined.end(), id) == defined.end()) return fail(error, "undefined producer ID");
        if (std::find(defined.begin(), defined.end(), op.outputs[0]) != defined.end()) return fail(error, "duplicate producer ID");
        defined.push_back(op.outputs[0]);
    }
    Execution vm(local), naive(local);
    std::vector<VjVmOpDesc> program;
    if (!emit_bytecode(local, bind, &vm, program, error)) return false;
    if (!run_cpu(program, error)) return false;
    reference(naive);
    result.emitted_ops = program.size();
    result.output = vm.values[vm.last].data; result.reference = naive.values[naive.last].data;
    for (size_t i = 0; i < local.ops.size(); ++i) {
        const auto& op = local.ops[i]; const size_t id = op.outputs[0];
        if (op.kind == OpKind::PagedKVWrite &&
            (vm.nodes[i].keys != naive.nodes[i].keys || vm.nodes[i].values != naive.nodes[i].values))
            return fail(error, "paged cache mismatch at graph node " + std::to_string(i));
        if (op.kind == OpKind::SSMRecurrence) {
            result.state.insert(result.state.end(), vm.nodes[i].state.begin(), vm.nodes[i].state.end());
            result.reference_state.insert(result.reference_state.end(), naive.nodes[i].state.begin(), naive.nodes[i].state.end());
        }
        if (op.kind == OpKind::MoERouter) {
            result.routing_ids.insert(result.routing_ids.end(), vm.values[id].ids.begin(), vm.values[id].ids.end());
            result.reference_routing_ids.insert(result.reference_routing_ids.end(), naive.values[id].ids.begin(), naive.values[id].ids.end());
        }
        for (size_t j = 0; j < vm.values[id].data.size(); ++j) {
            const float actual = vm.values[id].data[j], expected = naive.values[id].data[j];
            const float difference = std::fabs(actual - expected);
            if (!std::isfinite(actual) || !std::isfinite(expected) || difference > 3e-5f * (1 + std::fabs(expected)))
                return fail(error, "numeric mismatch at graph node " + std::to_string(i) + " element " + std::to_string(j));
            result.max_abs_error = std::max(result.max_abs_error, difference); ++result.checked_values;
        }
    }
    if (result.routing_ids != result.reference_routing_ids) return fail(error, "routing ID mismatch");
    for (size_t i = 0; i < result.state.size(); ++i)
        if (!std::isfinite(result.state[i]) || std::fabs(result.state[i] - result.reference_state[i]) > 3e-5f * (1 + std::fabs(result.reference_state[i])))
            return fail(error, "recurrent state mismatch");
    return true;
}
bool run_sharded_tiny_execution(const Model& model, const onboard::Plan& plan,
                                ShardedTinyResult& result, std::string* error,
                                bool wrong_slice_control) {
    result = {}; if (error) error->clear();
    if (!model.valid() || model.family != Family::DenseGQA || model.experts != 0 ||
        model.hidden != width || model.layers != 2 || model.heads != 4 || model.kv_heads != 2 ||
        model.ops.size() > 256 || model.input_value > 1024 || !plan.feasible ||
        (plan.gpus.size() != 2 && plan.gpus.size() != 3) || !plan.model_weight_bytes)
        return fail(error, "sharded tiny runner requires feasible TP2/TP3 DenseGQA hidden64 Q4/KV2");
    uint32_t qend = 0, kend = 0;
    long double total = 0;
    for (const auto& s : plan.gpus) {
        if (s.q_head_begin != qend || s.kv_head_begin != kend ||
            s.q_head_end < qend || s.kv_head_end < kend || s.q_head_end > 4 || s.kv_head_end > 2 ||
            s.q_head_end - qend != 2 * (s.kv_head_end - kend) || !s.weight_bytes)
            return fail(error, "invalid tiny rank head slices");
        qend = s.q_head_end; kend = s.kv_head_end; total += s.weight_bytes;
    }
    if (qend != 4 || kend != 2 || total != static_cast<long double>(plan.model_weight_bytes))
        return fail(error, "incomplete tiny rank slices");
    std::vector<size_t> defined{model.input_value};
    size_t reductions = 0, attentions = 0;
    for (const auto& op : model.ops) {
        if (op.inputs.empty() || op.outputs.size() != 1 || op.outputs[0] > 1024 ||
            std::find(defined.begin(), defined.end(), op.outputs[0]) != defined.end())
            return fail(error, "malformed sharded graph values");
        for (size_t id : op.inputs)
            if (std::find(defined.begin(), defined.end(), id) == defined.end())
                return fail(error, "undefined sharded graph input");
        const bool join = op.kind == OpKind::AttentionGQA || op.kind == OpKind::ResidualAdd;
        if (op.inputs.size() != (join ? 2u : 1u)) return fail(error, "malformed sharded graph arity");
        if (op.kind == OpKind::AllReduce) ++reductions;
        if (op.kind == OpKind::AttentionGQA) ++attentions;
        if (op.kind == OpKind::AttentionMLA || op.kind == OpKind::SSMRecurrence ||
            op.kind == OpKind::MoERouter || op.kind == OpKind::MoEFFN)
            return fail(error, "non-dense sharded graph is CAPACITY-ONLY");
        defined.push_back(op.outputs[0]);
    }
    if (reductions != 2 || attentions != 2)
        return fail(error, "sharded graph requires two attention and two FFN boundaries");
    // Retain graph indices: synthetic weight seeds must match the independent
    // unsharded reference, including nodes on either side of host collectives.
    Execution naive(model);
    std::vector<Execution> ranks;
    ranks.reserve(plan.gpus.size());
    for (size_t r = 0; r < plan.gpus.size(); ++r) ranks.emplace_back(model);
    std::vector<std::vector<VjVmOpDesc>> programs(plan.gpus.size());
    long double cumulative = 0;
    size_t begin = 0;
    for (size_t r = 0; r < ranks.size(); ++r) {
        auto& e = ranks[r]; const auto& s = plan.gpus[r];
        cumulative += s.weight_bytes;
        const size_t end = r + 1 == ranks.size() ? width :
            static_cast<size_t>(std::floor(width * cumulative / total));
        if (end <= begin) return fail(error, "empty FFN rank slice");
        const size_t count = end - begin;
        result.ffn_begin.push_back(begin); result.ffn_end.push_back(end);
        result.q_heads.push_back(s.q_head_end - s.q_head_begin);
        for (size_t i = 0; i < model.ops.size(); ++i) {
            const auto& op = model.ops[i]; auto& n = e.nodes[i];
            auto& y = e.values[op.outputs[0]].data;
            const auto& x = e.values[op.inputs[0]].data;
            VjVmOpDesc d{};
            const size_t nk = s.kv_head_end - s.kv_head_begin;
            if (op.kind == OpKind::AllReduce ||
                (!nk && (op.kind == OpKind::PagedKVWrite || op.kind == OpKind::AttentionGQA))) {
                d.opcode = GridBarrier;
            } else {
                if (!bind(op, i, d, &e)) return fail(error, "rank descriptor binding failed");
                if (op.kind == OpKind::Linear && n.gate) {
                    std::vector<float> packed(2 * count * width);
                    for (size_t branch = 0; branch < 2; ++branch)
                        std::copy_n(n.weights.data() + (branch * width + begin) * width,
                                    count * width, packed.data() + branch * count * width);
                    n.weights.swap(packed); d.rows = static_cast<uint32_t>(2 * count);
                    d.ptr0 = address(n.weights.data());
                } else if (op.kind == OpKind::ActMul) {
                    d.rows = 1; d.cols = static_cast<uint32_t>(count);
                    d.ptr1 = address(x.data() + count);
                } else if (op.kind == OpKind::Linear &&
                           e.producer[op.inputs[0]] < model.ops.size() &&
                           model.ops[e.producer[op.inputs[0]]].kind == OpKind::ActMul) {
                    if (i + 1 >= model.ops.size() || model.ops[i + 1].kind != OpKind::AllReduce)
                        return fail(error, "split FFN down lacks collective");
                    std::vector<float> packed(width * count);
                    for (size_t row = 0; row < width; ++row) for (size_t c = 0; c < count; ++c) {
                        const size_t col = (begin + c + (wrong_slice_control && r == 0 ? width / 2 : 0)) % width;
                        packed[row * count + c] = n.weights[row * width + col];
                    }
                    n.weights.swap(packed); d.cols = static_cast<uint32_t>(count);
                    d.ptr0 = address(n.weights.data());
                } else if (op.kind == OpKind::PagedKVWrite) {
                    // Physically rank-local paged caches, including historical slots.
                    std::vector<uint16_t> keys(4 * nk * 16), values(keys.size());
                    for (size_t t = 0; t < 4; ++t) {
                        const size_t source = (t * 2 + s.kv_head_begin) * 16;
                        std::copy_n(n.keys.data() + source, nk * 16, keys.data() + t * nk * 16);
                        std::copy_n(n.values.data() + source, nk * 16, values.data() + t * nk * 16);
                    }
                    n.keys.swap(keys); n.values.swap(values);
                    n.store.k = address(x.data() + s.kv_head_begin * 16);
                    n.store.v = address(x.data() + width / 2 + s.kv_head_begin * 16);
                    n.store.k_cache = address(n.keys.data()); n.store.v_cache = address(n.values.data());
                    n.store.heads = static_cast<uint32_t>(nk);
                } else if (op.kind == OpKind::AttentionGQA) {
                    n.attention.q = address(x.data() + s.q_head_begin * 16);
                    n.attention.out = address(y.data() + s.q_head_begin * 16);
                    n.attention.q_heads = s.q_head_end - s.q_head_begin;
                    n.attention.kv_heads = static_cast<uint32_t>(nk);
                }
            }
            programs[r].push_back(d);
        }
        result.rank_ops.push_back(programs[r].size());
        result.numeric.emitted_ops += programs[r].size(); begin = end;
    }
    reference(naive);
    size_t start = 0;
    for (size_t i = 0; i < model.ops.size(); ++i) {
        const auto& op = model.ops[i];
        const bool collective = op.kind == OpKind::AllReduce;
        const bool gather = op.kind == OpKind::AttentionGQA;
        if (!collective && !gather && i + 1 != model.ops.size()) continue;
        for (size_t r = 0; r < ranks.size(); ++r)
            if (!run_cpu(programs[r].data() + start, i + 1 - start, error)) return false;
        start = i + 1;
        if (!collective && !gather) continue;
        const size_t source = collective ? op.inputs[0] : op.outputs[0];
        std::vector<float> sum(width, 0);
        for (const auto& rank : ranks) for (size_t j = 0; j < width; ++j)
            sum[j] += rank.values[source].data[j];
        // Copy, never reallocate: already-bound descriptors hold these addresses.
        for (auto& rank : ranks) std::copy(sum.begin(), sum.end(), rank.values[op.outputs[0]].data.begin());
        if (collective) ++result.host_reductions; else ++result.attention_gathers;
    }
    result.numeric.output = ranks[0].values[ranks[0].last].data;
    result.numeric.reference = naive.values[naive.last].data;
    // Check every replicated collective output and final result, not local partials.
    for (const auto& op : model.ops) {
        const size_t id = op.outputs[0];
        if (op.kind != OpKind::AllReduce && op.kind != OpKind::AttentionGQA && id != naive.last) continue;
        for (const auto& rank : ranks) for (size_t j = 0; j < width; ++j) {
            const float actual = rank.values[id].data[j], expected = naive.values[id].data[j];
            const float delta = std::fabs(actual - expected);
            result.numeric.max_abs_error = std::max(result.numeric.max_abs_error, delta);
            ++result.numeric.checked_values;
            if (!std::isfinite(actual) || !std::isfinite(expected) || delta > 3e-5f * (1 + std::fabs(expected)))
                return fail(error, "sharded numeric mismatch at value " + std::to_string(id));
        }
    }
    return true;
}
} // namespace universal
