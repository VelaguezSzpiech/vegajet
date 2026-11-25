#include "core/tiny_execution.hpp"
#include "core/fusion_pass.hpp"
#include "core/emit_bytecode.hpp"
#include "tests/universal_fixture.hpp"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstring>
#include "ops/vj_vm_cpu.hpp"
#include <string>
#include <utility>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "check failed %s:%d: %s\n", __FILE__, __LINE__, #x); return 1; } } while (0)


static int test_device_segments() {
    using namespace vegajet::vm;
    using universal::DeviceKernel;
    using universal::DeviceSegment;
    const DeviceKernel attention = DeviceKernel::Attention;
    const DeviceKernel ffn = DeviceKernel::Ffn;
    const DeviceKernel aux = DeviceKernel::Aux;
    const std::pair<uint32_t, DeviceKernel> supported[] = {
        {ArResRmsQ8, attention}, {GemvQ4A8, attention}, {GemvQ8Dp4a, attention},
        {QkNormRopeKv, attention}, {FlashDecodeGqa, attention},
        {GemvQ4Dot8, ffn}, {MoeRouterTopK, ffn}, {ActMulQ8, ffn},
        {MoeExpertCombine, aux}, {RmsF32, aux}, {PagedKvStore, aux},
        {GatedDelta, aux}, {LatentAttention, aux}, {ResidualAddF32, aux},
        {QuantizeF32, aux}, {GemvF32, aux}, {ActMulF32, aux},
        {MoeGateUpF32, aux}, {MoeDownF32, aux}, {ConvertF32ToF16, aux}
    };
    std::vector<DeviceSegment> segments;
    std::string error;
    for (const auto& entry : supported) {
        VjVmOpDesc op{};
        op.opcode = entry.first;
        CHECK(universal::plan_device_segments({op}, segments, &error));
        CHECK(error.empty() && segments.size() == 1);
        CHECK(segments[0].kernel == entry.second && segments[0].begin == 0 && segments[0].count == 1);
    }
    const uint32_t codes[] = {GridBarrier, ArResRmsQ8, GemvQ4A8, QkNormRopeKv,
        PagedKvStore, FlashDecodeGqa, GemvQ8Dp4a, ArResRmsQ8, GemvQ4A8,
        MoeRouterTopK, ActMulQ8, GemvQ4Dot8, ArResRmsQ8, GridBarrier,
        GemvQ4A8, MoeExpertCombine, GemvQ4A8, MoeRouterTopK, GridBarrier, GridBarrier};
    std::vector<VjVmOpDesc> program;
    for (uint32_t code : codes) { VjVmOpDesc op{}; op.opcode = code; program.push_back(op); }
    program[16].flags = BarrierAfter; // Cannot borrow the following FFN context.
    const DeviceKernel expected[] = {attention, attention, attention, aux, attention,
        attention, ffn, ffn, ffn, ffn, ffn, ffn, attention, aux, attention, ffn};
    const auto original = program;
    CHECK(universal::plan_device_segments(program, segments, &error));
    CHECK(segments.size() == sizeof(expected) / sizeof(expected[0]));
    size_t covered = 0;
    for (size_t i = 0; i < program.size(); ++i) {
        CHECK(std::memcmp(&program[i], &original[i], sizeof(VjVmOpDesc)) == 0);
        if (program[i].opcode == GridBarrier) continue;
        CHECK(covered < segments.size());
        CHECK(segments[covered].begin == i && segments[covered].count == 1);
        CHECK(segments[covered].kernel == expected[covered]);
        ++covered;
    }
    CHECK(covered == segments.size());
    for (uint32_t unknown : {0u, 22u, UINT32_MAX}) {
        const auto saved = segments;
        program.back().opcode = unknown;
        CHECK(!universal::plan_device_segments(program, segments, &error));
        CHECK(!error.empty() && segments.size() == saved.size());
        for (size_t i = 0; i < saved.size(); ++i)
            CHECK(segments[i].kernel == saved[i].kernel && segments[i].begin == saved[i].begin && segments[i].count == saved[i].count);
        CHECK(!run_cpu(&program.back(), 1));
    }
    CHECK(universal::plan_device_segments({}, segments, &error) && segments.empty() && error.empty());
    VjVmOpDesc barrier{}; barrier.opcode = GridBarrier;
    CHECK(universal::plan_device_segments({barrier, barrier}, segments, &error) && segments.empty());

    // Verify the combined emitter preserves exactly the flat emitter's stream.
    universal::Model model;
    const universal::OpKind kinds[] = {universal::OpKind::Linear, universal::OpKind::RoPE,
        universal::OpKind::GridBarrier, universal::OpKind::PagedKVWrite,
        universal::OpKind::AttentionGQA, universal::OpKind::MoERouter,
        universal::OpKind::ActMul, universal::OpKind::Linear};
    const uint32_t emitted_codes[] = {GemvQ4A8, QkNormRopeKv, GridBarrier,
        PagedKvStore, FlashDecodeGqa, MoeRouterTopK, ActMulQ8, GemvQ4Dot8};
    for (auto kind : kinds) { universal::Op op{}; op.kind = kind; model.ops.push_back(op); }
    const auto binder = [](const universal::Op&, size_t index, VjVmOpDesc& desc, void* context) {
        desc.opcode = static_cast<const uint32_t*>(context)[index];
        desc.flags = SkipRms | BarrierAfter;
        desc.rows = static_cast<uint32_t>(index + 1);
        return true;
    };
    // Binder's context API is mutable; keep the fixture mutable rather than cast away const.
    std::vector<uint32_t> binding(emitted_codes, emitted_codes + 8);
    std::vector<VjVmOpDesc> flat;
    CHECK(universal::emit_bytecode(model, binder, binding.data(), flat, &error));
    CHECK(universal::emit_device_plan(model, binder, binding.data(), program, segments, &error));
    CHECK(flat.size() == program.size() && segments.size() == flat.size() - 1);
    CHECK(std::memcmp(flat.data(), program.data(), flat.size() * sizeof(VjVmOpDesc)) == 0);
    const auto saved_program = program;
    const auto saved_segments = segments;
    binding[0] = UINT32_MAX;
    CHECK(!universal::emit_device_plan(model, binder, binding.data(), program, segments, &error));
    CHECK(program.size() == saved_program.size() && segments.size() == saved_segments.size());
    CHECK(std::memcmp(program.data(), saved_program.data(), program.size() * sizeof(VjVmOpDesc)) == 0);
    return 0;
}

int main() {
    using namespace vegajet::vm;
    CHECK(test_device_segments() == 0);
    { uint16_t k[2]={0x3800,0xb400}, v[2]={0x3a00,0x3000}, kc[8]{},vc[8]{};uint32_t pages[]={1,0};
      KvStoreDesc s{(uintptr_t)k,(uintptr_t)v,(uintptr_t)kc,(uintptr_t)vc,(uintptr_t)pages,0,2,2,1,2};
      float q[2]={.37f,-.23f},y[2]{};VjVmFlashDesc f{};f.q=(uintptr_t)q;f.k_cache=(uintptr_t)kc;f.v_cache=(uintptr_t)vc;f.out=(uintptr_t)y;f.q_heads=f.kv_heads=f.gqa_group=1;f.head_dim=2;f.page_tokens=f.page_count=2;f.context_tokens=1;f.page_table=(uintptr_t)pages;
      VjVmOpDesc p[2]{};p[0].opcode=PagedKvStore;p[0].ptr0=(uintptr_t)&s;p[1].opcode=FlashDecodeGqa;p[1].ptr0=(uintptr_t)&f;
      CHECK(run_cpu(p,2));CHECK(kc[0]==0&&kc[4]==k[0]&&vc[5]==v[1]&&y[0]==.75f&&y[1]==.125f);
      const float expected_y[] = {y[0], y[1]};
      std::fill(kc, kc + 8, uint16_t{0}); std::fill(vc, vc + 8, uint16_t{0});
      y[0] = y[1] = 0;
      VjVmOpDesc barrier{}; barrier.opcode = GridBarrier;
      p[0].flags |= BarrierAfter;
      const std::vector<VjVmOpDesc> stream = {barrier, p[0], barrier, p[1], barrier};
      std::vector<universal::DeviceSegment> segments;
      CHECK(universal::plan_device_segments(stream, segments));
      CHECK(segments.size() == 2 && segments[0].kernel == universal::DeviceKernel::Aux &&
            segments[1].kernel == universal::DeviceKernel::Attention);
      for (const auto& segment : segments) CHECK(run_cpu(stream.data() + segment.begin, segment.count));
      CHECK(kc[0]==0&&kc[4]==k[0]&&vc[5]==v[1]&&y[0]==expected_y[0]&&y[1]==expected_y[1]);
      pages[0]=2;CHECK(!run_cpu(p,2));}
    {float q[]={.13f,-.21f},k[]={.39f,.17f},v[]={.61f,-.53f,.27f},g[]={.97f,.31f},state[]={.11f,.21f,.31f,-.41f,.51f,-.61f},out[3]{};double ref[6];for(int i=0;i<6;++i)ref[i]=state[i];
     DeltaDesc a{(uintptr_t)q,(uintptr_t)k,(uintptr_t)v,(uintptr_t)g,(uintptr_t)state,(uintptr_t)out,2,3};VjVmOpDesc op{};op.ptr0=(uintptr_t)&a;op.opcode=GatedDelta;
     for(int t=0;t<5;++t){q[1]+=.013f;k[0]-=.021f;CHECK(run_cpu(&op,1));for(int c=0;c<3;++c){double pred=ref[c]*k[0]+ref[3+c]*k[1],delta=g[1]*(v[c]-g[0]*pred);
         for(int i=0;i<2;++i){ref[i*3+c]=g[0]*ref[i*3+c]+k[i]*delta;CHECK(std::abs(ref[i*3+c]-state[i*3+c])<2e-6);}CHECK(std::abs(ref[c]*q[0]+ref[3+c]*q[1]-out[c])<2e-6);}}}
    {float g[32],up[32],d=0;int8_t y[32];double ref[32],mx=0;for(int i=0;i<32;++i){g[i]=(i-19)*.163f;up[i]=.713f+.029f*i;double x=g[i];ref[i]=up[i]*.5*x*(1+std::tanh(std::sqrt(2/std::acos(-1.0))*(x+.044715*x*x*x)));mx=std::max(mx,std::abs(ref[i]));}
     VjVmOpDesc op{};op.opcode=ActMulQ8;op.flags=GeGluActivation;op.cols=32;op.ptr0=(uintptr_t)g;op.ptr1=(uintptr_t)up;op.ptr2=(uintptr_t)y;op.ptr3=(uintptr_t)&d;CHECK(run_cpu(&op,1));CHECK(std::abs(d-mx/127)<1e-7);for(int i=0;i<32;++i)CHECK(y[i]==std::nearbyint(ref[i]*127/mx));}
    {float q[]={.37f,-.12f},qr[]={-.71f},z[]={.18f,.63f,-.47f,.83f,.22f,-.16f},kr[]={.14f,-.32f,.59f},w[]={.23f,-.41f,-.63f,.17f},out[2]{};
     LatentDesc a{(uintptr_t)q,(uintptr_t)qr,(uintptr_t)z,(uintptr_t)kr,(uintptr_t)w,(uintptr_t)out,1,3,2,1,2,.61f};VjVmOpDesc op{};op.opcode=LatentAttention;op.ptr0=(uintptr_t)&a;CHECK(run_cpu(&op,1));for(int d=0;d<2;++d){double total=0,sum=0;for(int t=0;t<3;++t){double e=std::exp(a.scale*(double(q[0])*z[2*t]+double(q[1])*z[2*t+1]+double(qr[0])*kr[t]));total+=e;sum+=e*(double(w[2*d])*z[2*t]+double(w[2*d+1])*z[2*t+1]);}CHECK(std::abs(out[d]-sum/total)<2e-6);}}
    using universal_test::FixtureFamily;
    const FixtureFamily fixture[] = {FixtureFamily::Dense, FixtureFamily::MoE,
                                     FixtureFamily::Hybrid, FixtureFamily::MLA};
    const universal::Family expected[] = {universal::Family::DenseGQA, universal::Family::PureMoE,
        universal::Family::HybridGDNMoE, universal::Family::SparseMLAMoE};
    for (size_t f = 0; f < 4; ++f) {
        const std::string path = universal_test::write_tiny_gguf(fixture[f]);
        CHECK(!path.empty());
        vegajet::GgufFile gguf;
        CHECK(gguf.open(path));
        universal::Options options;
        options.tp_size = 1;
        options.n_cu = 60;
        universal::Model graph = universal::lower_gguf(gguf, options);
        CHECK(graph.valid() && graph.family == expected[f] && graph.ops.size() > 1);
        universal::TinyExecutionResult baseline, fused;
        std::string error;
        if (!universal::run_tiny_execution(graph, baseline, &error)) {
            std::fprintf(stderr, "unfused family %zu: %s\n", f, error.c_str()); return 1;
        }
        universal::PassManager passes;
        passes.run(graph, {1, 60, 1, false});
        if (!universal::run_tiny_execution(graph, fused, &error)) {
            std::fprintf(stderr, "fused family %zu: %s\n", f, error.c_str()); return 1;
        }
        CHECK(baseline.output.size() == 64 && fused.output.size() == 64);
        CHECK(baseline.emitted_ops > 20 && fused.emitted_ops > 20);
        CHECK(baseline.checked_values > 1000 && fused.checked_values > 1000);
        bool nonzero = false;
        for (size_t j = 0; j < 64; ++j) {
            CHECK(std::isfinite(baseline.output[j]));
            CHECK(std::abs(baseline.output[j] - baseline.reference[j]) < 1e-4f);
            CHECK(std::abs(fused.output[j] - fused.reference[j]) < 1e-4f);
            CHECK(std::abs(fused.output[j] - baseline.output[j]) < 1e-4f);
            nonzero = nonzero || std::abs(baseline.output[j]) > .01f;
        }
        CHECK(nonzero);
        CHECK(baseline.routing_ids == baseline.reference_routing_ids);
        CHECK(fused.routing_ids == baseline.routing_ids);
        CHECK(baseline.routing_ids.size() == (f ? 4u : 0u));
        for (size_t j = 0; j < baseline.routing_ids.size(); j += 2) {
            CHECK(baseline.routing_ids[j] < 4 && baseline.routing_ids[j + 1] < 4);
            CHECK(baseline.routing_ids[j] != baseline.routing_ids[j + 1]);
        }
        CHECK(baseline.state.size() == (fixture[f] == FixtureFamily::Hybrid ? 4096u : 0u));
        for (size_t j = 0; j < baseline.state.size(); ++j)
            CHECK(std::abs(baseline.state[j] - baseline.reference_state[j]) < 1e-4f);
        CHECK(::unlink(path.c_str()) == 0);
    }
    std::puts("[PASS] four connected two-layer emitted graphs match independent FP32 outputs, routes and state (unfused/fused)");
    return 0;
}
