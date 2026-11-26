# v1.3 runtime collective (M3)

The selected device list is rank order. Runtime topology reads KFD `io_links` and
`p2p_links` for those nodes only. An explicit PCIe/XGMI directed GPU edge with
reported bandwidth and no `NO_PEER_TO_PEER_DMA` flag is a peer candidate. Actual
KFD mapping is required; a map failure stops the runtime, not a silent fallback.
Missing/disabled/unknown edges use a coherent pinned-host bounce frame. Reported
bandwidth is not a measurement or an estimate. Device identity prefers DRM/PCI
`unique_id` (canonical hex); a missing/zero/malformed ID falls back to PCI BDF.
Duplicate nodes/identities fail closed.

`Runtime::configure_collective` feeds this matrix to the M2 topology/collective
planners. Owner JSON options can force directed host-bounce routes and constrain
islands/stages. A saved hardware profile is advisory for CPU planning; execution
always re-reads sysfs, never treats a profile as permission to access a peer.

For N=1 the plan allocates no collective RX, routes, flags or status. Bypass
preserves FP32 values without a FP16 conversion. For N>=2 each destination owns
disjoint payload/epoch VRAM BOs with two parity banks and one source frame per
bank. Payload is coherent/NC and epoch flags UC, cacheline padded. Direct peers
push to these dual-BAR0 BOs. Non-peer routes push to separate owner RX frames in
coherent pinned host memory. No GPU reads peer VRAM, and no remote atomics occur.

One wave per rank pushes full FP16 RNE partials, drains **every writer lane**
(`vmcnt(0)` plus system fence), synchronizes, then lane 0 publishes epochs. Each
consumer polls only its owner-local or host-bounce RX, with a bounded timeout,
and sums in canonical source-rank order into FP32 output. All ranks are submitted
before any host wait. The initial generic per-op API is synchronous: every
consumer must complete before the next epoch, a stronger reuse condition than
RT-B1's two-bank asynchronous rule. Timeout, bad FP16 input, failed mapping,
epoch exhaustion, or queue failure stops all ranks; callers discard all tentative
outputs. Reconfiguration drains actual queues before freeing/resetting storage.

The legacy two-rank mailbox ABI and fused kernels are unchanged. Legacy mailbox
setup is limited to N=2; N!=2 uses the dynamic collective API.

## Owner hardware commands

Build on the MI50 host with its ROCm toolchain and the usual kernel gates:

```sh
cmake -S . -B build-v13-gpu -DVEGAJET_CPU_ONLY=OFF
cmake --build build-v13-gpu -j2
build-v13-gpu/vegajet --gpu 0,1,2 bench --bench-ar --tp 3 --hsaco-dir build-v13-gpu/hsaco
```

The benchmark checks lengths 1, 33, and 3072 for 64 changing-input epochs each,
including every rank's full output. Acceptance requires exit 0 and `verify_errs=0`.
The printed collective duration is observed host-wall time, not a prediction.

No GPU was opened in the development VM. CPU tests and code-object/ISA gates are
not evidence of real peer ordering, bandwidth, or decode correctness on hardware.
