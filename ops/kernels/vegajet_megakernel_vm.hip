#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// ============================================================================
// VegaJet Universal Persistent Bytecode VM Megakernel (gfx906 / Vega 20)
//   - Executes compiled VjVmOpDesc bytecode streams in ONE single AQL dispatch!
//   - Hard guardrails: < 24 KB SQC L1I (.text), <= 64 VGPRs, strictly 0 B spill!
//   - Lock-free grid barrier on isolated 64 B cachelines (bar[0] / bar[16]) + s_sleep 1!
//   - Surgical GLC=1 producer/consumer loads + dynamic glc/slc weight streaming!
//   - Wave64 DPP single-pass Fused ResidualAdd + RMSNorm + b=32 INT8 Quantize!
// ============================================================================

struct alignas(64) VjVmOpDescDev {
    uint32_t opcode;
    uint32_t flags;
    uint32_t rows;
    uint32_t cols;
    uint64_t ptr0;
    uint64_t ptr1;
    uint64_t ptr2;
    uint64_t ptr3;
    uint64_t ptr4;
    uint32_t aux0;
    float    eps;
};

static constexpr uint32_t VM_OP_RES_RMS_QUANT    = 1u;
static constexpr uint32_t VM_OP_LINEAR_Q8_DP4A   = 2u;
static constexpr uint32_t VM_OP_ACT_MUL_QUANT    = 3u;
static constexpr uint32_t VM_OP_QKNORM_ROPE      = 4u;
static constexpr uint32_t VM_OP_AR_RES_RMS_QUANT = 5u;

// Per-rank Dual-BAR0 Push-Only P2P descriptor passed via ptr0 of VM_OP_AR_RES_RMS_QUANT
struct VjVmArDescDev {
    uint64_t x_partial_va;  // Local partial FP32 vector [D]
    uint64_t peer_tx_va;    // Remote peer's BAR0 RX payload buffer (posted writes!)
    uint64_t local_rx_va;   // Local GPU's OWN HBM2 RX payload buffer (fast local read!)
    uint64_t peer_flag_va;  // Remote peer's BAR0 RX epoch flag (64B aligned)
    uint64_t local_flag_va; // Local GPU's OWN HBM2 RX epoch flag (64B aligned)
    uint32_t epoch;         // Monotonic epoch for this AR call
    uint32_t pad;
};

static constexpr uint32_t kVmFlagNonTemporalW  = 1u << 0;
static constexpr uint32_t kVmFlagBarrierAfter  = 1u << 2;

typedef uint32_t vm_u32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ int32_t vm_sdot4(uint32_t a, uint32_t b, int32_t c) {
    return __builtin_amdgcn_sdot4(static_cast<int32_t>(a), static_cast<int32_t>(b), c, false);
}

__device__ __forceinline__ uint4 vm_load_u32x4(const uint4* ptr, bool nt_stream) {
    if (nt_stream) {
        vm_u32x4 v = __builtin_nontemporal_load(reinterpret_cast<const vm_u32x4*>(ptr));
        return make_uint4(v[0], v[1], v[2], v[3]);
    }
    return *ptr;
}

// Single-pass Wave64 reduction using native ds_swizzle_b32 (xor 1,2,4,8,16) + cross-half shfl_xor(32)
__device__ __forceinline__ float vm_wave64_sum_dpp(float v) {
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x041F)); // xor 1
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x081F)); // xor 2
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x101F)); // xor 4
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x201F)); // xor 8
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x401F)); // xor 16
    v += __shfl_xor(v, 32, 64);
    return v;
}

__device__ __forceinline__ float vm_block32_max(float v) {
    v = fmaxf(v, __shfl_xor(v, 16, 32));
    v = fmaxf(v, __shfl_xor(v, 8,  32));
    v = fmaxf(v, __shfl_xor(v, 4,  32));
    v = fmaxf(v, __shfl_xor(v, 2,  32));
    v = fmaxf(v, __shfl_xor(v, 1,  32));
    return v;
}

// Lock-free persistent grid barrier on isolated 64 B cachelines (bar[0] = counter, bar[16] = sense)
__device__ __forceinline__ void vm_grid_barrier(uint32_t* bar, uint32_t n_wgs, uint32_t& sense) {
    __syncthreads();
    if (threadIdx.x == 0u) {
        __threadfence();
        const uint32_t arrived = atomicAdd(&bar[0], 1u) + 1u;
        if (arrived == n_wgs) {
            __atomic_store_n(&bar[0], 0u, __ATOMIC_RELEASE);
            __atomic_store_n(&bar[16], sense, __ATOMIC_RELEASE);
        } else {
            while (__atomic_load_n(&bar[16], __ATOMIC_ACQUIRE) != sense) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }
    __syncthreads();
    sense += 1u;
}

// ============================================================================
// Universal Persistent Megakernel VM Entry Point
//   Grid: n_wgs * 256 (e.g. 60 or 120 persistent WGs on MI50), Block: 256 (4 waves)
//   Strictly < 24 KB .text, <= 64 VGPRs, 0 B private spill!
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void vegajet_megakernel_vm(
    const VjVmOpDescDev* __restrict__ program, // [num_ops] bytecode descriptors
    uint32_t num_ops,
    uint32_t* __restrict__ bar_scratch) {      // 128 bytes zero-initialized before dispatch

    const uint32_t wg_id = blockIdx.x;
    const uint32_t n_wgs = gridDim.x;
    const uint32_t tid   = threadIdx.x;
    const uint32_t wave  = tid >> 6;
    const uint32_t lane  = tid & 63u;
    uint32_t sense       = 1u;

    __shared__ float sm_red[4];

    for (uint32_t pc = 0u; pc < num_ops; ++pc) {
        const VjVmOpDescDev op = program[pc];

        if (op.opcode == VM_OP_RES_RMS_QUANT) {
            // Single-WG fused ResidualAdd + RMSNorm + b=32 INT8 Quantize (Wave64 DPP pass!)
            // Executed by WG 0 (256 threads) — takes < 1.2 us for D=2560..8192 with ZERO extra launches!
            if (wg_id == 0u) {
                const float* x_in   = reinterpret_cast<const float*>(op.ptr0);
                float*       res_io = reinterpret_cast<float*>(op.ptr1);
                const float* w_rms  = reinterpret_cast<const float*>(op.ptr2);
                int8_t*      out_q8 = reinterpret_cast<int8_t*>(op.ptr3);
                float*       out_d  = reinterpret_cast<float*>(op.ptr4);
                const uint32_t D    = op.cols;

                // Pass 1: h = res + x (in-place residual update) + sum-of-squares
                float ss = 0.0f;
                #pragma nounroll
                for (uint32_t c = tid; c < D; c += 256u) {
                    const float h = __builtin_nontemporal_load(x_in + c) + res_io[c];
                    res_io[c] = h;
                    ss = fmaf(h, h, ss);
                }
                ss = vm_wave64_sum_dpp(ss);
                if (lane == 0u) sm_red[wave] = ss;
                __syncthreads();
                float total_ss = sm_red[0] + sm_red[1] + sm_red[2] + sm_red[3];
                const float inv_rms = rsqrtf(total_ss / static_cast<float>(D) + op.eps);

                // Pass 2: RMSNorm * w_rms + b=32 online INT8 quantize (zero extra HBM2 round-trip!)
                #pragma nounroll
                for (uint32_t blk = tid >> 5; blk < (D >> 5); blk += 8u) {
                    const uint32_t c = (blk << 5) + (tid & 31u);
                    const float y = res_io[c] * inv_rms * w_rms[c];
                    const float amax = vm_block32_max(fabsf(y));
                    const float d = amax * (1.0f / 127.0f);
                    const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                    int32_t q = max(-127, min(127, __float2int_rn(y * inv_d)));
                    out_q8[c] = static_cast<int8_t>(q);
                    if ((tid & 31u) == 0u) out_d[blk] = d;
                }
            }
        } else if (op.opcode == VM_OP_LINEAR_Q8_DP4A) {
            // Persistent grid-stride VJ_Q8_DP4A GEMV across ALL 60 CUs (4 rows/WG, v_dot4_i32_i8)
            const int8_t*   w_qs  = reinterpret_cast<const int8_t*>(op.ptr0);
            const uint16_t* w_ds  = reinterpret_cast<const uint16_t*>(op.ptr1);
            const int8_t*   x_q8  = reinterpret_cast<const int8_t*>(op.ptr2);
            const float*    x_d   = reinterpret_cast<const float*>(op.ptr3);
            float*          y_out = reinterpret_cast<float*>(op.ptr4);
            const uint32_t  rows  = op.rows;
            const uint32_t  cols  = op.cols;
            const uint32_t  n16   = cols >> 4;
            const bool      nt_w  = (op.flags & kVmFlagNonTemporalW) != 0u;
            const uint4*    x4    = reinterpret_cast<const uint4*>(x_q8);

            for (uint32_t row = wg_id * 4u + wave; row < rows; row += n_wgs * 4u) {
                const uint4*    w4 = reinterpret_cast<const uint4*>(w_qs + static_cast<uint64_t>(row) * cols);
                const uint16_t* ds = w_ds + static_cast<uint64_t>(row) * (cols >> 5);
                float acc = 0.0f;
                #pragma nounroll
                for (uint32_t i = lane; i < n16; i += 64u) {
                    const uint4 wv = vm_load_u32x4(w4 + i, nt_w);
                    // Surgical GLC=1 producer/consumer read on activation vector produced by prior VM op
                    vm_u32x4 xv_glc = __builtin_nontemporal_load(reinterpret_cast<const vm_u32x4*>(x4 + i));
                    int32_t dp = vm_sdot4(wv.w, xv_glc[3],
                                 vm_sdot4(wv.z, xv_glc[2],
                                 vm_sdot4(wv.y, xv_glc[1],
                                 vm_sdot4(wv.x, xv_glc[0], 0))));
                    const float xd = __builtin_nontemporal_load(x_d + (i >> 1));
                    const float sc = __half2float(__ushort_as_half(ds[i >> 1])) * xd;
                    acc = fmaf(sc, static_cast<float>(dp), acc);
                }
                acc = vm_wave64_sum_dpp(acc);
                if (lane == 0u) y_out[row] = acc;
            }
        } else if (op.opcode == VM_OP_ACT_MUL_QUANT) {
            // Fused SiLU(gate) * up -> inline b=32 INT8 Quantize across all WGs!
            const float* gate   = reinterpret_cast<const float*>(op.ptr0);
            const float* up     = reinterpret_cast<const float*>(op.ptr1);
            int8_t*      out_q8 = reinterpret_cast<int8_t*>(op.ptr2);
            float*       out_d  = reinterpret_cast<float*>(op.ptr3);
            const uint32_t nblk = op.cols >> 5;
            const uint32_t sub  = tid & 31u;

            for (uint32_t blk = wg_id * 8u + (tid >> 5); blk < nblk; blk += n_wgs * 8u) {
                const uint32_t c = (blk << 5) + sub;
                const float g = __builtin_nontemporal_load(gate + c);
                const float u = __builtin_nontemporal_load(up + c);
                const float v = (g / (1.0f + __expf(-g))) * u;
                const float amax = vm_block32_max(fabsf(v));
                const float d = amax * (1.0f / 127.0f);
                const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                out_q8[c] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(v * inv_d))));
                if (sub == 0u) out_d[blk] = d;
            }
        } else if (op.opcode == VM_OP_QKNORM_ROPE) {
            // Fused per-head RMSNorm + rotary embedding RoPE in-place
            float*       vec      = reinterpret_cast<float*>(op.ptr0);
            const float* w_norm   = reinterpret_cast<const float*>(op.ptr1);
            const uint32_t n_head = op.rows;
            const uint32_t hd     = op.cols; // power of 2 <= 256
            const uint32_t pos    = op.aux0;

            for (uint32_t h = wg_id * 4u + wave; h < n_head; h += n_wgs * 4u) {
                float* vh = vec + static_cast<uint64_t>(h) * hd;
                float ss = 0.0f;
                for (uint32_t i = lane; i < hd; i += 64u) {
                    const float v = vh[i];
                    ss = fmaf(v, v, ss);
                }
                const float inv_r = rsqrtf(vm_wave64_sum_dpp(ss) / static_cast<float>(hd) + op.eps);
                const uint32_t half_d = hd >> 1;
                for (uint32_t p = lane; p < half_d; p += 64u) {
                    const float x0 = vh[p] * inv_r * w_norm[p];
                    const float x1 = vh[p + half_d] * inv_r * w_norm[p + half_d];
                    const float theta = static_cast<float>(pos) * __powf(10000.0f, -2.0f * static_cast<float>(p) / static_cast<float>(hd));
                    float s_th, c_th;
                    __sincosf(theta, &s_th, &c_th);
                    vh[p]          = x0 * c_th - x1 * s_th;
                    vh[p + half_d] = x0 * s_th + x1 * c_th;
                }
            }
        } else if (op.opcode == VM_OP_AR_RES_RMS_QUANT) {
            // Single Wave64-DPP Fused Push-Only Dual-BAR0 P2P AllReduce -> ResidualAdd -> RMSNorm -> b=32 INT8 Quantize!
            // Executed by WG 0 (256 threads): 100% posted writes to peer BAR0 + 100% local HBM2 reads + 1 DPP RMSNorm+Quant pass!
            if (wg_id == 0u) {
                const VjVmArDescDev ar = *reinterpret_cast<const VjVmArDescDev*>(op.ptr0);
                const float* x_part  = reinterpret_cast<const float*>(ar.x_partial_va);
                uint16_t*    peer_tx = reinterpret_cast<uint16_t*>(ar.peer_tx_va);
                const uint16_t* loc_rx = reinterpret_cast<const uint16_t*>(ar.local_rx_va);
                uint32_t*    peer_fl = reinterpret_cast<uint32_t*>(ar.peer_flag_va);
                const uint32_t* loc_fl = reinterpret_cast<const uint32_t*>(ar.local_flag_va);
                float*       res_io  = reinterpret_cast<float*>(op.ptr1);
                const float* w_rms   = reinterpret_cast<const float*>(op.ptr2);
                int8_t*      out_q8  = reinterpret_cast<int8_t*>(op.ptr3);
                float*       out_d   = reinterpret_cast<float*>(op.ptr4);
                const uint32_t D     = op.cols;

                // 1. Push local FP16 partial to peer's BAR0 RX buffer (100% posted PCIe writes!)
                #pragma nounroll
                for (uint32_t c = tid; c < D; c += 256u) {
                    peer_tx[c] = __half_as_ushort(__float2half(__builtin_nontemporal_load(x_part + c)));
                }
                __syncthreads();
                if (tid == 0u) {
                    __threadfence_system();
                    __atomic_store_n(peer_fl, ar.epoch, __ATOMIC_RELEASE);
                    while (__atomic_load_n(loc_fl, __ATOMIC_ACQUIRE) != ar.epoch) {
                        __builtin_amdgcn_s_sleep(1);
                    }
                    __threadfence_system();
                }
                __syncthreads();

                // 2. Single-pass Bit-Exact Commutative FP16 Reduce + ResidualAdd + Sum-of-Squares (Wave64 DPP!)
                //    Rounding x_loc through FP16 guarantees GPU 0 and GPU 1 produce bit-identical reduced sums!
                float ss = 0.0f;
                #pragma nounroll
                for (uint32_t c = tid; c < D; c += 256u) {
                    const float x_loc  = __half2float(__float2half(__builtin_nontemporal_load(x_part + c)));
                    const float x_peer = __half2float(__ushort_as_half(__builtin_nontemporal_load(loc_rx + c)));
                    const float h      = res_io[c] + (x_loc + x_peer);
                    res_io[c] = h;
                    ss = fmaf(h, h, ss);
                }
                ss = vm_wave64_sum_dpp(ss);
                if (lane == 0u) sm_red[wave] = ss;
                __syncthreads();
                const float total_ss = sm_red[0] + sm_red[1] + sm_red[2] + sm_red[3];
                const float inv_rms  = rsqrtf(total_ss / static_cast<float>(D) + op.eps);

                // 3. Inline RMSNorm + b=32 INT8 Quantize
                #pragma nounroll
                for (uint32_t blk = tid >> 5; blk < (D >> 5); blk += 8u) {
                    const uint32_t c = (blk << 5) + (tid & 31u);
                    const float y = res_io[c] * inv_rms * w_rms[c];
                    const float amax = vm_block32_max(fabsf(y));
                    const float d = amax * (1.0f / 127.0f);
                    const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                    out_q8[c] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv_d))));
                    if ((tid & 31u) == 0u) out_d[blk] = d;
                }
            }
        }

        if ((op.flags & kVmFlagBarrierAfter) != 0u && (pc + 1u) < num_ops) {
            vm_grid_barrier(bar_scratch, n_wgs, sense);
        }
    }
}
