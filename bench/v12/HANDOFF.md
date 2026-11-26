# VegaJet v1.2 — handoff for the inference-machine agent

Branch: `release/v1.2`. Base: `b741ae4` (main). This work was built on a VM **without a GPU**.

## What was verified offline (and only this)
- CPU build + `ctest`: 100 % passed, 0 failed out of 96 (`vegajet_layer_test` skipped: needs a GPU).
- GPU build with **TheRock ROCm 7.12.0a20260311** (gfx906) + `vj_hsaco_gate` + `vj_isa_scan`: 24/24 code objects PASS, 0 B spill.
- ROCm 7.14 fails 2 **unmodified** kernels (VGPR 75 > 73 waiver; 12 B private segment). Build with 7.12.
- Nothing was run on a GPU. **No speed number and no token parity is proven.** That is your job.

See `bench/v12/OFFLINE_LOG.md` for the gate log.

## Rules for testing
1. Use the **same two MI50s** for every v1.2 number. Do not use the third card here. Record the GPU IDs, clocks, power cap.
2. Baseline first: build `main` (b741ae4) and `release/v1.2` with all toggles off; both must give the same tokens and the same speed (±1 %).
3. A lever counts only if the median of 5 runs beats the baseline by more than the run-to-run range, **and** the tokens equal raw greedy.
4. Any token mismatch = revert that lever (or leave its toggle off) and write why.

## Levers in this branch

### Always on (must not change tokens or speed in a healthy run)
| ID | What | GPU test |
|---|---|---|
| C1 | Super-kernel barriers + peer-mailbox waits fail closed: timeout writes a device fault record; host prints `DEVICE FAULT …`, exit 3, no tokens. Kernarg +8 B (`fault_va`). | (a) healthy bench: same tokens/speed as main, exit 0. (b) fault injection: set `spin < 1u` in `vj_ornith_super.hip`, rebuild, decode → expect `DEVICE FAULT`, exit 3; then revert. |
| C3 | Token-id bounds check before the embedding copy; NaN-argmax guard. | Healthy run unchanged. |
| C5 | Spec/MTP/DFlash bench returns exit 3 on any token mismatch vs raw (`[BENCH-VALIDATE] FAIL`). | Run with `--spec-force-reject` or a wrong draft: must exit 3. |
| PD2 | Serve: bounded async SSE writer (`VEGAJET_V12_ASYNC_OUT`, **default on**; `=0` = old inline writer). | Slow reader (`curl --limit-rate 8k`): same bytes; client kill stops generation; `/metrics in_flight` → 0. |
| PD3 | Serve: HTTP 400 for sampling params the greedy engine cannot honor (temperature ≠ 0, top_p, …). | `"temperature":0.7` → 400; `"temperature":0` → 200. |

### Toggles, default OFF (`=1` turns on)
| ID | Env | Path | What | Expect | Risk |
|---|---|---|---|---|---|
| D3 | `VEGAJET_V12_VOCAB_ILP` | raw | LM-head argmax: 2 accumulator chains, 2 loads in flight | lm_head stage faster (hipfire +7 % e2e) | FP sum order differs → near-tie could flip a token; check parity |
| D13 | `VEGAJET_V12_DENSE_PERSIST` | raw | Dense GEMV persistent work-stealing grid (atomic 8-row tiles) | est. +4–6 % on dense GEMV | low |
| D8 | `VEGAJET_V12_IQ2S_REG` | raw | MoE super-kernel without the 4 KiB IQ2_S grid in LDS (LDS 8,896 → 4,800 B) | higher occupancy on GU | bit-exact by test; speed unknown (L1 reads replace LDS) |
| D5 | `VEGAJET_V12_BARRIER_VOL` | raw | `buffer_wbinvl1_vol` in place of full `buffer_wbinvl1` at stage barriers | MoE sync time down | **Highest risk.** Cross-stage reads are not all glc. Needs exact parity on ≥ 5 prompts × 512 tok **and** a 10-min soak before keeping. |
| C7+S1 | `VEGAJET_V12_SPEC_GOV` | MTP / DFlash2 | Depth governor uses measured `t_ver[M]` + live acceptance (removes fabricated inflation) | higher tok/s, esp. ≥ 8k ctx; depth below fixed K | lossless by construction |
| S2 | `VEGAJET_V12_BATCH_CATCHUP` | MTP | Drafter catch-up/ingest in one M-row MTP forward (≤ 16 rows) | catch-up ms/cycle 3–5× lower for acc ≥ 3 | never executed; drafts may drift slightly; test ≤ 8k ctx first (`spec_fl_part_o` has 60 chunk columns) |
| PD1 | `VEGAJET_V12_SERVE_SPEC` | serve | `serve` uses the bench suffix-spec driver; startup line says `greedy+suffix-spec` | API tok/s up on repetitive text | diff served text on vs off before trusting speed; more VRAM |

Group-2 levers (D6 GDN ≤ 64 VGPR, S4 narrow verify GEMV, S5 fused verify norm+quant, S3 tree verification, D1 asymmetric experts) are added by later commits on this branch. Each adds its row here and an entry in `bench/v12/levers_extra.sh`.

## How to run
```bash
git fetch && git switch release/v1.2
cmake -S . -B build-gpu -G Ninja -DVEGAJET_CPU_ONLY=OFF -DVEGAJET_ROCM_PATH=<rocm-7.12-prefix>
cmake --build build-gpu -j
# pin the two MI50s, then:
MODEL=<model-file> DRAFT=<dflash2-file> MTP=<mtp-file> RUNS=5 N=128 bench/v12/run.sh
MODEL=... N=512 RUNS=5 bench/v12/run.sh        # 512-token parity/speed
EXTRA="--prompt 'Explain how a CPU executes instructions out of order.'" MODEL=... bench/v12/run.sh   # normal text
```
Then for each lever write one line in `bench/v12/LOG.md`: `ID | before → after tok/s (median, range) | parity | kept/reverted | reason`.

## Order
1. Baseline parity main vs v1.2-all-off. 2. C1 fault injection. 3. Raw levers one at a time (D3, D13, D8, then D5 last). 4. Spec levers (SPEC_GOV, BATCH_CATCHUP). 5. Combine all kept levers; re-run parity + 30-min soak. 6. Serve (PD1 on/off API diff).
