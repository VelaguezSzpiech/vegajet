#include "ops/vj_vm_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace vegajet::vm {
namespace {
template<class T> T* ptr(uint64_t p) { return reinterpret_cast<T*>(static_cast<uintptr_t>(p)); }
bool fail(std::string* e, const char* s) { if (e) *e = s; return false; }
bool need(bool ok, std::string* e) { return ok || fail(e, "invalid VM descriptor dimensions or null pointer"); }
float half(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 31u, frac = h & 1023u, bits;
    if (!exp) {
        if (!frac) bits = sign;
        else {
            int shift = 0;
            while ((frac & 1024u) == 0u) { frac <<= 1; ++shift; }
            frac &= 1023u;
            bits = sign | (uint32_t(127 - 14 - shift) << 23) | (frac << 13);
        }
    } else if (exp == 31u) bits = sign | 0x7f800000u | (frac << 13);
    else bits = sign | ((exp + 112u) << 23) | (frac << 13);
    float result;
    std::memcpy(&result, &bits, sizeof result);
    return result;
}
int8_t quant(float x, float inverse) {
    return static_cast<int8_t>(std::max(-127, std::min(127, static_cast<int>(std::nearbyint(x * inverse)))));
}
int signed_nibble(const uint8_t* p, size_t i) {
    const int n = (p[i / 2] >> ((i & 1u) * 4u)) & 15;
    return (n ^ 8) - 8;
}
float activate(float g, uint32_t flags) {
    return (flags & GeGluActivation) ? 0.5f*g*(1.0f+std::tanh(0.7978845608f*(g+0.044715f*g*g*g))) : g/(1.0f+std::exp(-g));
}
uint16_t to_half(float x) {
    uint32_t bits; std::memcpy(&bits,&x,sizeof bits);
    const uint32_t sign=(bits>>16)&0x8000u, exp=(bits>>23)&255u, frac=bits&0x7fffffu;
    if(exp==255u)return static_cast<uint16_t>(sign|0x7c00u|(frac?0x200u:0u));
    if(exp>142u)return static_cast<uint16_t>(sign|0x7c00u);
    if(exp<102u)return static_cast<uint16_t>(sign);
    const uint32_t shift=exp<113u?126u-exp:13u;
    const uint32_t mant=exp<113u?frac|0x800000u:frac;
    uint32_t rounded=mant>>shift;
    const uint32_t tail=mant&((1u<<shift)-1u),mid=1u<<(shift-1u);
    rounded+=(tail>mid||(tail==mid&&(rounded&1u)))?1u:0u;
    return static_cast<uint16_t>(sign+((exp<113u?0u:exp-112u)<<10)+rounded);
}
}

bool run_cpu(const VjVmOpDesc* program, size_t count, std::string* error) {
    if (error) error->clear();
    if (count && !program) return fail(error, "null VM program");
    for (size_t pc = 0; pc < count; ++pc) {
        const auto& op = program[pc];
        switch (op.opcode) {
        case ResidualAddF32: case ActMulF32: {
            if(!need(op.rows&&op.cols&&op.ptr0&&op.ptr1&&op.ptr2,error))return false;
            for(size_t i=0;i<size_t(op.rows)*op.cols;++i){const float a=ptr<const float>(op.ptr0)[i],b=ptr<const float>(op.ptr1)[i];
                ptr<float>(op.ptr2)[i]=op.opcode==ResidualAddF32?a+b:activate(a,op.flags)*b;}
            break;
        }
        case ConvertF32ToF16: {
            if(!need(op.rows&&op.cols&&op.ptr0&&op.ptr1,error))return false;
            for(size_t i=0;i<size_t(op.rows)*op.cols;++i)ptr<uint16_t>(op.ptr1)[i]=to_half(ptr<const float>(op.ptr0)[i]);
            break;
        }
        case QuantizeF32: {
            if(!need(op.rows&&op.cols&&op.ptr0&&op.ptr1&&op.ptr2,error))return false;
            const size_t blocks=(size_t(op.cols)+31)/32;
            for(size_t r=0;r<op.rows;++r)for(size_t b=0;b<blocks;++b){
                const size_t begin=r*op.cols+b*32,end=r*op.cols+std::min(size_t(op.cols),b*32+32);float mx=0;
                for(size_t i=begin;i<end;++i){float v=ptr<const float>(op.ptr0)[i];if(!std::isfinite(v))return fail(error,"nonfinite quantization input");mx=std::max(mx,std::fabs(v));}
                const float scale=mx/127.0f,inverse=mx>0?127.0f/mx:0;
                ptr<float>(op.ptr2)[r*blocks+b]=scale;
                for(size_t i=begin;i<end;++i){const float v=ptr<const float>(op.ptr0)[i];const int8_t q=std::isfinite(inverse)?quant(v,inverse):quant(v/mx,127.0f);ptr<int8_t>(op.ptr1)[i]=q;if(op.ptr3)ptr<float>(op.ptr3)[i]=q*scale;}
            }
            break;
        }
        case GemvF32: {
            if(!need(op.rows&&op.cols&&op.ptr0&&op.ptr1&&op.ptr2,error))return false;
            for(size_t r=0;r<op.rows;++r){float sum=op.ptr3?ptr<const float>(op.ptr3)[r]:0;
                for(size_t c=0;c<op.cols;++c)sum=std::fma(ptr<const float>(op.ptr0)[r*op.cols+c],ptr<const float>(op.ptr1)[c],sum);
                ptr<float>(op.ptr2)[r]=sum;}
            break;
        }
        case MoeGateUpF32: case MoeDownF32: {
            if(!need(op.ptr0,error))return false;
            const auto& m=*ptr<const MoeF32Desc>(op.ptr0);const bool down=op.opcode==MoeDownF32;
            if(!need(m.weights&&m.input&&m.ids&&m.output&&m.experts&&m.hidden&&m.intermediate&&m.top_k&&m.top_k<=m.experts&&(down?m.mixture:m.up_weights),error))return false;
            const size_t matrix=size_t(m.hidden)*m.intermediate;
            if(!need(matrix<=SIZE_MAX/m.experts/sizeof(float),error))return false;
            for(size_t j=0;j<m.top_k;++j)if(!need(ptr<const uint32_t>(m.ids)[j]<m.experts,error))return false;
            if(down){for(size_t h=0;h<m.hidden;++h){float sum=0;
                for(size_t j=0;j<m.top_k;++j){const size_t e=ptr<const uint32_t>(m.ids)[j];float v=0;
                    for(size_t c=0;c<m.intermediate;++c)v=std::fma(ptr<const float>(m.weights)[e*matrix+h*m.intermediate+c],ptr<const float>(m.input)[j*m.intermediate+c],v);
                    sum=std::fma(v,ptr<const float>(m.mixture)[j],sum);}
                ptr<float>(m.output)[h]=sum;}}
            else {for(size_t j=0;j<m.top_k;++j)for(size_t c=0;c<m.intermediate;++c){const size_t e=ptr<const uint32_t>(m.ids)[j];float g=0,u=0;
                for(size_t h=0;h<m.hidden;++h){const size_t wi=e*matrix+c*m.hidden+h;const float x=ptr<const float>(m.input)[h];g=std::fma(ptr<const float>(m.weights)[wi],x,g);u=std::fma(ptr<const float>(m.up_weights)[wi],x,u);}
                ptr<float>(m.output)[j*m.intermediate+c]=g+(m.shared?ptr<const float>(m.shared)[c]:0);
                ptr<float>(m.output)[(size_t(m.top_k)+j)*m.intermediate+c]=u;}}
            break;
        }
        case RmsF32: {
            if(!need(op.rows&&op.cols&&op.ptr0&&op.ptr1&&op.ptr2&&op.eps>0,error))return false;
            for(size_t r=0;r<op.rows;++r){const float* x=ptr<const float>(op.ptr0)+r*op.cols;
                float ss=0;for(size_t c=0;c<op.cols;++c)ss=std::fma(x[c],x[c],ss);
                float inv=1/std::sqrt(ss/op.cols+op.eps);
                for(size_t c=0;c<op.cols;++c)ptr<float>(op.ptr2)[r*op.cols+c]=x[c]*inv*ptr<const float>(op.ptr1)[c];}
            break;
        }
        case ArResRmsQ8: {
            const size_t d = op.cols;
            if (!need(d && d % 32 == 0 && op.ptr0 && op.ptr1 && op.ptr2 && op.ptr3 && op.ptr4 && op.eps >= 0, error)) return false;
            auto x = ptr<const float>(op.ptr0); auto residual = ptr<float>(op.ptr1);
            auto norm = ptr<const float>(op.ptr2); auto q = ptr<int8_t>(op.ptr3); auto scales = ptr<float>(op.ptr4);
            float ss = 0;
            for (size_t i = 0; i < d; ++i) { float h = x[i] + residual[i]; residual[i] = h; ss = std::fma(h, h, ss); }
            const float inverse = 1.0f / std::sqrt(ss / static_cast<float>(d) + op.eps);
            for (size_t b = 0; b < d / 32; ++b) {
                float values[32], maximum = 0;
                for (size_t j = 0; j < 32; ++j) { size_t i = b * 32 + j; values[j] = residual[i] * inverse * norm[i]; maximum = std::max(maximum, std::fabs(values[j])); }
                scales[b] = maximum / 127.0f;
                const float qi = maximum > 0 ? 127.0f / maximum : 0;
                for (size_t j = 0; j < 32; ++j) q[b * 32 + j] = quant(values[j], qi);
            }
            break;
        }
        case GemvQ8Dp4a: case GemvQ4A8: case GemvQ4Dot8: {
            const size_t rows = op.rows, cols = op.cols, blocks = (cols + 31) / 32;
            if (!need(rows && cols && cols % 32 == 0 && op.ptr0 && op.ptr1 && op.ptr2 && op.ptr3 && op.ptr4, error)) return false;
            auto weights = ptr<const uint8_t>(op.ptr0); auto wd = ptr<const uint16_t>(op.ptr1);
            auto x = ptr<const uint8_t>(op.ptr2); auto xd = ptr<const float>(op.ptr3); auto y = ptr<float>(op.ptr4);
            const bool q8w = op.opcode == GemvQ8Dp4a, q4x = op.opcode == GemvQ4Dot8;
            for (size_t r = 0; r < rows; ++r) {
                float acc = 0;
                for (size_t b = 0; b < blocks; ++b) {
                    int32_t dot = 0;
                    for (size_t j = 0; j < 32; ++j) {
                        const size_t c = b * 32 + j;
                        const int w = q8w ? static_cast<int8_t>(weights[r * cols + c]) : signed_nibble(weights + r * ((cols + 1) / 2), c);
                        const int a = q4x ? signed_nibble(x, c) : static_cast<int8_t>(x[c]);
                        dot += w * a;
                    }
                    const float scale = half(wd[r * blocks + b]) * xd[b];
                    acc = std::fma(static_cast<float>(dot), scale, acc);
                }
                y[r] = acc;
            }
            break;
        }
        case QkNormRopeKv: {
            const size_t heads = op.rows, hd = op.cols;
            const bool skip=(op.flags&SkipRms)!=0;
            if (!need(heads && hd && hd <= 256 && (hd & (hd - 1)) == 0 && op.ptr0 && (skip || (op.ptr1 && op.eps >= 0)), error)) return false;
            auto vectors = ptr<const float>(op.ptr0); auto output=ptr<float>(op.ptr2?op.ptr2:op.ptr0); auto norm = ptr<const float>(op.ptr1);
            for (size_t h = 0; h < heads; ++h) {
                float ss = 0;
                for (size_t i = 0; i < hd; ++i) ss = std::fma(vectors[h * hd + i], vectors[h * hd + i], ss);
                const float inverse = skip?1.0f:1.0f / std::sqrt(ss / static_cast<float>(hd) + op.eps);
                const size_t halfdim = hd / 2;
                for (size_t i = 0; i < halfdim; ++i) {
                    const float a = vectors[h * hd + i] * inverse * (skip?1.0f:norm[i]);
                    const float b = vectors[h * hd + i + halfdim] * inverse * (skip?1.0f:norm[i + halfdim]);
                    const float theta = static_cast<float>(op.aux0) * std::pow(10000.0f, -2.0f * static_cast<float>(i) / static_cast<float>(hd));
                    const float s = std::sin(theta), c = std::cos(theta);
                    output[h * hd + i] = a * c - b * s;
                    output[h * hd + i + halfdim] = a * s + b * c;
                }
            }
            break;
        }
        case LatentAttention:{
            if(!need(op.ptr0,error))return false;
            const auto& a=*ptr<const LatentDesc>(op.ptr0);
            if(!need(a.query&&a.latent&&a.value_proj&&a.out&&a.heads&&a.tokens&&a.rank&&a.output_dim&&(!a.rot_dim||(a.rot_query&&a.rot_key))&&a.scale>0,error))return false;
            const auto cache=[&](uint64_t p,size_t i){return (op.flags&F16LatentCache)?half(ptr<const uint16_t>(p)[i]):ptr<const float>(p)[i];};
            for(size_t h=0;h<a.heads;++h){std::vector<float> weights(a.tokens),avg(a.rank,0);float mx=-INFINITY,den=0;
                for(size_t t=0;t<a.tokens;++t){float s=0;for(size_t k=0;k<a.rank;++k)s=std::fma(ptr<const float>(a.query)[h*a.rank+k],cache(a.latent,t*a.rank+k),s);
                    for(size_t k=0;k<a.rot_dim;++k)s=std::fma(ptr<const float>(a.rot_query)[h*a.rot_dim+k],cache(a.rot_key,t*a.rot_dim+k),s);
                    weights[t]=a.scale*s;mx=std::max(mx,weights[t]);}
                for(size_t t=0;t<a.tokens;++t){float w=std::exp(weights[t]-mx);den+=w;for(size_t k=0;k<a.rank;++k)avg[k]=std::fma(w,cache(a.latent,t*a.rank+k),avg[k]);}
                for(size_t d=0;d<a.output_dim;++d){float s=0;for(size_t k=0;k<a.rank;++k)s=std::fma(avg[k]/den,ptr<const float>(a.value_proj)[(h*a.output_dim+d)*a.rank+k],s);ptr<float>(a.out)[h*a.output_dim+d]=s;}}
            break;}
        case GatedDelta: {
            if(!need(op.ptr0,error))return false;
            const auto& s=*ptr<const DeltaDesc>(op.ptr0);
            if(!need(s.q&&s.k&&s.v&&s.gates&&s.state&&s.out&&s.key_dim&&s.value_dim,error))return false;
            auto state=ptr<float>(s.state);auto gates=ptr<const float>(s.gates);
            for(size_t c=0;c<s.value_dim;++c){float pred=0,sum=0;
                for(size_t i=0;i<s.key_dim;++i)pred=std::fma(state[i*s.value_dim+c],ptr<const float>(s.k)[i],pred);
                float delta=gates[1]*(ptr<const float>(s.v)[c]-gates[0]*pred);
                for(size_t i=0;i<s.key_dim;++i){auto& x=state[i*s.value_dim+c];x=std::fma(ptr<const float>(s.k)[i],delta,gates[0]*x);sum=std::fma(x,ptr<const float>(s.q)[i],sum);}ptr<float>(s.out)[c]=sum;}
            break;
        }
        case PagedKvStore: {
            if(!need(op.ptr0,error))return false;
            const auto& s=*ptr<const KvStoreDesc>(op.ptr0);
            if(!need(s.k&&s.v&&s.k_cache&&s.v_cache&&s.page_table&&s.heads&&s.dim&&s.page_tokens&&s.position/s.page_tokens<s.pages,error))return false;
            auto page=ptr<const uint32_t>(s.page_table)[s.position/s.page_tokens];
            if(!need(page<s.pages,error))return false;
            size_t width=size_t(s.heads)*s.dim, slot=size_t(page)*s.page_tokens+s.position%s.page_tokens;
            if(!need(slot<=SIZE_MAX/width && width<=SIZE_MAX/2,error))return false;
            for(size_t i=0;i<width;++i){ptr<uint16_t>(s.k_cache)[slot*width+i]=(op.flags&F32KvInput)?to_half(ptr<const float>(s.k)[i]):ptr<const uint16_t>(s.k)[i];ptr<uint16_t>(s.v_cache)[slot*width+i]=(op.flags&F32KvInput)?to_half(ptr<const float>(s.v)[i]):ptr<const uint16_t>(s.v)[i];}
            break;
        }
        case FlashDecodeGqa: {
            if (!need(op.ptr0, error)) return false;
            const auto& a = *ptr<const VjVmFlashDesc>(op.ptr0);
            const size_t qheads = a.q_heads, kvheads = a.kv_heads, hd = a.head_dim, page_tokens = a.page_tokens, context = a.context_tokens;
            if (!need(a.q && a.k_cache && a.v_cache && a.out && a.page_table && qheads && kvheads && hd && page_tokens && context && qheads % kvheads == 0 && a.gqa_group == qheads / kvheads && a.page_count >= (context + page_tokens - 1) / page_tokens, error)) return false;
            auto q = ptr<const float>(a.q); auto k = ptr<const uint16_t>(a.k_cache); auto v = ptr<const uint16_t>(a.v_cache);
            auto out = ptr<float>(a.out); auto lse = a.lse ? ptr<float>(a.lse) : nullptr; auto pages = ptr<const uint32_t>(a.page_table);
            const float scale = a.scale > 0 ? a.scale : 1.0f / std::sqrt(static_cast<float>(hd));
            for (size_t h = 0; h < qheads; ++h) {
                const size_t kvh = h / a.gqa_group;
                float maximum = -std::numeric_limits<float>::infinity();
                for (size_t t = 0; t < context; ++t) {
                    const size_t page = pages[t / page_tokens];
                    if (page >= a.page_count) return fail(error, "GQA page table index out of range");
                    const size_t base = ((page * page_tokens + t % page_tokens) * kvheads + kvh) * hd;
                    float score = 0;
                    for (size_t d = 0; d < hd; ++d) score = std::fma(q[h * hd + d], half(k[base + d]), score);
                    maximum = std::max(maximum, score * scale);
                }
                float denom = 0;
                for (size_t t = 0; t < context; ++t) {
                    const size_t page = pages[t / page_tokens], base = ((page * page_tokens + t % page_tokens) * kvheads + kvh) * hd;
                    float score = 0;
                    for (size_t d = 0; d < hd; ++d) score = std::fma(q[h * hd + d], half(k[base + d]), score);
                    denom += std::exp(score * scale - maximum);
                }
                for (size_t d = 0; d < hd; ++d) {
                    float sum = 0;
                    for (size_t t = 0; t < context; ++t) {
                        const size_t page = pages[t / page_tokens], base = ((page * page_tokens + t % page_tokens) * kvheads + kvh) * hd;
                        float score = 0;
                        for (size_t j = 0; j < hd; ++j) score = std::fma(q[h * hd + j], half(k[base + j]), score);
                        sum += std::exp(score * scale - maximum) * half(v[base + d]);
                    }
                    out[h * hd + d] = sum / denom;
                }
                if (lse) lse[h] = maximum + std::log(denom);
            }
            break;
        }
        case GridBarrier:
            break; // CPU execution is sequential; the device dispatcher splits at this marker.
        case MoeRouterTopK: {
            const size_t tokens = op.rows, experts = op.cols, top = op.aux0;
            if (!need(tokens && experts && top && top <= experts && op.ptr0 && op.ptr1 && op.ptr2, error)) return false;
            auto logits = ptr<const float>(op.ptr0); auto ids = ptr<uint32_t>(op.ptr1); auto weights = ptr<float>(op.ptr2);
            for (size_t t = 0; t < tokens; ++t) {
                std::vector<size_t> order(experts);
                for (size_t e = 0; e < experts; ++e) order[e] = e;
                std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(top), order.end(), [&](size_t a, size_t b) {
                    const float x = logits[t * experts + a], y = logits[t * experts + b];
                    return x == y ? a < b : x > y;
                });
                const float maximum = logits[t * experts + order[0]];
                float denom = 0;
                for (size_t j = 0; j < top; ++j) denom += std::exp(logits[t * experts + order[j]] - maximum);
                for (size_t j = 0; j < top; ++j) {
                    ids[t * top + j] = static_cast<uint32_t>(order[j]);
                    weights[t * top + j] = std::exp(logits[t * experts + order[j]] - maximum) / denom;
                }
            }
            break;
        }
        case ActMulQ8: {
            const size_t d = op.cols;
            if (!need(d && d % 32 == 0 && op.ptr0 && op.ptr1 && op.ptr2 && op.ptr3, error)) return false;
            auto gate = ptr<const float>(op.ptr0); auto up = ptr<const float>(op.ptr1); auto q = ptr<int8_t>(op.ptr2); auto scales = ptr<float>(op.ptr3);
            for (size_t b = 0; b < d / 32; ++b) {
                float values[32], maximum = 0;
                for (size_t j = 0; j < 32; ++j) {
                    const float g = gate[b * 32 + j];
                    const float act = (op.flags & GeGluActivation) ? (0.5f*g*(1.0f+std::tanh(0.7978845608f*(g+0.044715f*g*g*g)))) : g / (1.0f + std::exp(-g));
                    values[j] = act * up[b * 32 + j];
                    maximum = std::max(maximum, std::fabs(values[j]));
                }
                scales[b] = maximum / 127.0f;
                const float inverse = maximum > 0 ? 127.0f / maximum : 0;
                for (size_t j = 0; j < 32; ++j) q[b * 32 + j] = quant(values[j], inverse);
            }
            break;
        }
        case MoeExpertCombine: {
            const size_t rows = op.rows, hidden = op.cols, top = op.aux0;
            if (!need(rows && hidden && top && op.ptr0 && op.ptr2 && op.ptr3, error)) return false;
            auto experts = ptr<const float>(op.ptr0); auto weights = ptr<const float>(op.ptr2); auto out = ptr<float>(op.ptr3);
            for (size_t r = 0; r < rows; ++r) for (size_t d = 0; d < hidden; ++d) {
                float sum = 0;
                for (size_t j = 0; j < top; ++j) sum = std::fma(experts[(r * top + j) * hidden + d], weights[r * top + j], sum);
                out[r * hidden + d] = sum;
            }
            break;
        }
        default:
            return fail(error, "unsupported VM opcode");
        }
    }
    return true;
}

void naive_linear(const float* weights, const float* x, const float* bias, float* y, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        float sum = bias ? bias[r] : 0.0f;
        for (size_t c = 0; c < cols; ++c) sum += weights[r * cols + c] * x[c];
        y[r] = sum;
    }
}
} // namespace vegajet::vm
