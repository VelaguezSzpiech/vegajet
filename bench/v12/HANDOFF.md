# VegaJet v1.2 — handoff for the inference-machine agent

Branch: `release/v1.2` (draft PR #1). Base: `b741ae4` (main). Built and checked on a VM **without a GPU**.

## What was verified offline (and only this)
- CPU build + `ctest`: 100 % passed (final count in `OFFLINE_LOG.md`). `vegajet_layer_test` is skipped: it needs a GPU.
- GPU build with **TheRock ROCm 7.12.0a20260311** (gfx906) + `vj_hsaco_gate` + `vj_isa_scan`: 25/25 code objects PASS, 0 B spill.
- ROCm 7.14 fails 2 **unmodified** kernels (VGPR 75 > 73 waiver; 12 B private segment). Build with 7.12.
- Nothing ran on a GPU. **No speed number and no token parity is proven.** That is your job.

## Pin the cards (the machine now has 3 MI50s)
GPU selection accepts full PCI addresses. Ordinals can change when a card is added.
```bash
./build-gpu/vegajet serve --gpu list                    # find the BDFs
export GPUS=0000:03:00.0,0000:43:00.0                     # example values: the two cards for v1.2 numbers
./build-gpu/vegajet bench "$MODEL" --gpu "$GPUS"          # or VJ_GPU_DEVICES="$GPUS"
```
TP=2 requires exactly two selected cards and allocates only on them. Startup prints the BDF of each rank. Use the same pair for every v1.2 number.

## Rules for testing
1. Baseline first: build `main` (b741ae4) and `release/v1.2` with all toggles off. Same tokens; speed within ±1 %.
2. A lever counts only if the median of 5 runs beats its baseline by more than the run-to-run range, **and** tokens equal raw greedy.
3. Any token mismatch = leave that toggle off and write why in `bench/v12/LOG.md`.

## Always on (must not change tokens or speed in a healthy run)
| ID | What | GPU test |
|---|---|---|
| C1 | Super-kernel barriers + peer-mailbox waits fail closed (device fault record → `DEVICE FAULT …`, exit 3). Kernarg +8 B (`fault_va`). | Healthy bench: same tokens/speed as main. Fault injection: set `spin < 1u` in `vj_ornith_super.hip`, rebuild, decode → `DEVICE FAULT`, exit 3; revert. |
| C3 | Token-id bounds check before embedding; NaN-argmax guard. | Healthy run unchanged. |
| C5 | Spec/MTP/DFlash bench exits 3 on any token mismatch vs raw (`[BENCH-VALIDATE] FAIL`). | `--spec-force-reject` or a wrong draft → exit 3. |
| B4 fix | A C1/C3 failure stops generation, drafter work and serving (no `-1` tokens). | Same fault injection: no tokens after the fault. |
| RT-B1 | All-reduce slices: both ranks drain before the readiness word is reused (deadlock fix). Single-slice decode unchanged; multi-slice prefill may be slightly slower. | `bench/v12/FIXES2.md`: long prompts, `--prefill-chunk 128/256/512/1024`, FP16 and INT8 wire; no hang, no corruption. |
| RT-B2, RT-M1…M7 | Runtime: reload failure stops the run; exact unmap sets; completion-signal ownership; kernarg slot retirement; ELF validation; BAR mapping checked; queue teardown errors kept. | Normal runs unchanged; startup prints BDFs. |
| P-* | Model-file validation: bad GGUF/tokenizer/template data → clear error, no crash. Expert tensor shapes checked before load (P-M3). | Real model loads as before. |
| PD2 | Serve: bounded async SSE writer (`VEGAJET_V12_ASYNC_OUT=0` = old inline writer). | Slow reader (`curl --limit-rate 8k`): same bytes; client kill stops generation; `/metrics in_flight` → 0. |
| PD3 | Serve: HTTP 400 for sampling params the greedy engine cannot honor. | `"temperature":0.7` → 400; `"temperature":0` → 200. |
| S-* | Server hardening (`AUDIT-server.md`, `fix S-*` commits). | Normal API use unchanged. |

## Toggles, default OFF (`=1` turns on)
| ID | Env | Path | What | Risk / note |
|---|---|---|---|---|
| D3 | `VEGAJET_V12_VOCAB_ILP` | raw | LM-head argmax: 2 accumulator chains, 2 loads in flight | FP sum order differs → check parity |
| D8 | `VEGAJET_V12_IQ2S_REG` | raw | MoE super-kernel without the IQ2_S grid in LDS (8,896 → 4,800 B) | Bit-exact by test; speed unknown |
| D1 | `VEGAJET_V12_ASYM_EXPERTS` | raw | Bandwidth-proportional expert split per rank | 1080/1140 MHz rounds to 512/512 (no change): check the printed split. Rejected together with BATCH_CATCHUP. |
| D13 | `VEGAJET_V12_DENSE_PERSIST` | raw, with `VEGAJET_ENABLE_SUPER=0` | Persistent work-stealing dense GEMV | Only active outside the super-kernel; compare with the `ENABLE_SUPER=0` baseline |
| D6 | `VEGAJET_V12_GDN_SPLITV` | raw, with `VEGAJET_ENABLE_SUPER=0` | Column-thread GDN ConvRec, no LDS partials, 64 VGPR | Bit-exact by CPU emulation; 0 VGPR headroom |
| D5 | `VEGAJET_V12_BARRIER_VOL` | raw | `buffer_wbinvl1_vol` at stage barriers | **Highest risk.** Exact parity on ≥ 5 prompts × 512 tok **and** 10-min soak before keeping. Test last. |
| C7+S1 | `VEGAJET_V12_SPEC_GOV` | MTP / DFlash2 | Measured verify cost + live acceptance set draft depth | Lossless by construction |
| S2 | `VEGAJET_V12_BATCH_CATCHUP` | MTP | Drafter catch-up in one M-row forward | Never executed; test ≤ 8k ctx first |
| S4 | `VEGAJET_V12_NARROW_VERIFY` | MTP | Narrow multi-column verify GEMV, M=2..16 | May not fix the 5/129 MTP drift (likely cause: different activation quant at M=1 vs M>1) |
| S5 | `VEGAJET_V12_VERIFY_FUSED_NQ` | MTP | Verify norm+quant in one launch per layer (72 → 36 launches/step) | Fails closed on barrier timeout; falls back if WGs cannot be resident |
| PD1 | `VEGAJET_V12_SERVE_SPEC` | serve | `serve` uses the suffix-spec driver | Diff served text on vs off before trusting speed |
| S3 | `VEGAJET_V12_TREE_VERIFY` | — | Host-side tree plans only; **enabling it fails closed** | GPU path not done: `docs/v12-tree-verify.md` |

## How to run
```bash
git fetch && git switch release/v1.2
cmake -S . -B build-gpu -G Ninja -DVEGAJET_CPU_ONLY=OFF -DVEGAJET_ROCM_PATH=<rocm-7.12-prefix>
cmake --build build-gpu -j
GPUS=<bdf0>,<bdf1> MODEL=/path/ornith-122b.gguf DRAFT=/path/dflash2.gguf MTP=/path/blk48-mtp.gguf RUNS=5 N=128 bench/v12/run.sh
GPUS=... MODEL=... N=512 bench/v12/run.sh                                                               # 512-token parity/speed
GPUS=... MODEL=... EXTRA="--prompt 'Explain how a CPU executes instructions out of order.'" bench/v12/run.sh   # normal text
ONLY="VOCAB_ILP IQ2S_REG" GPUS=... MODEL=... bench/v12/run.sh                                           # subset
```
`run.sh` clears inherited `VEGAJET_V12_*` variables per run, records the env of each run, and exits non-zero if any run fails. It does **not** prove token parity against `main`: compare token IDs separately.

Log one line per lever in `bench/v12/LOG.md`: `ID | before → after tok/s (median, range) | parity | kept/reverted | reason`.

## Order
1. Baseline parity: main vs v1.2-all-off.
2. Fault injection (C1/B4) and C5 forced reject.
3. Raw levers one at a time: D3, D8, D1; with `ENABLE_SUPER=0`: D13, D6; D5 last.
4. Spec levers: SPEC_GOV, BATCH_CATCHUP, NARROW_VERIFY, VERIFY_FUSED_NQ.
5. Combine all kept levers: parity + 30-min soak (0 UE/DE, +0 CE).
6. Serve: PD1 on/off text diff; PD2 slow reader; PD3 and server limits.

## Reviews and audits in this branch
- `REVIEW.md`, `REVIEW2.md`: change reviews. All BLOCKERs fixed (`FIXES.md`, `FIXES2.md`).
- `AUDIT-runtime.md`, `AUDIT-parsers.md`, `AUDIT-server.md`: whole-engine audits; BLOCKERs and MAJORs fixed in `fix RT-*`, `fix P-*`, `fix S-*` commits.
- Still open: REVIEW M2 (cycle-cost model), M6 partial (no automated token-ID comparison against main), S3 GPU tree verifier.

## Next: TP=1..N (v1.3)
`docs/V13_TP3_DESIGN.md` lists every 2-rank assumption (file:line) and the modular mechanics for any TP size. The current CLI cannot run TP=3 decode yet. Section 9.1 lists the link and pairwise tests you can run on the 3 cards today.
