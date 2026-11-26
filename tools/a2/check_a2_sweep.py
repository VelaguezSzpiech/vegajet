#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Fail-closed verifier for the A2 Split-V=8 matched sweep.

The pair runs one engine load with two `--sweep` variants on the same model:

  1. baseline: `--no-gdn-splitv8 --bench-mode raw`  (VEGAJET_ENABLE_SUPER=0)
  2. candidate: `--gdn-splitv8 --bench-mode raw`    (VEGAJET_ENABLE_SUPER=0)

Both variants run the same `run_greedy_raw` path used by `tg128`/`tg512`; the
second variant's `vs_base` line is a real in-process token comparison against
the first.  This script proves, from the engine's own `VEGAJET_DISPATCH_LOG`
host journal, that the selector reached *both* A2 dispatches for every M=1 GDN
ConvRec in the candidate variant, and that the rest of the enqueue sequence is
identical after substituting the fused Split-V=4 kernel.

It never touches a GPU.  `--self-test` exercises the checks with synthetic
logs/stdout (positive and negative cases).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ST1 = "ornith_gdn_conv_recur_splitv8_recur_m1_kernel"
ST2 = "ornith_gdn_conv_recur_splitv8_quant_m1_kernel"
FUSED = "ornith_gdn_conv_recur_splitv_quant_m1_kernel"

# Host enqueue shapes from models/ornith/ornith_gpu.cpp (A2 branch / fused branch).
ST_GRID = (65536, 1, 1)     # 256u * 256u workgroups
FU_GRID = (32768, 1, 1)     # 128u * 256u workgroups
BLOCK = (256, 1, 1)

E_RE = re.compile(
    r"^E rank=(\d+) seq=(\d+) kernel=(\S+) "
    r"grid=(\d+)x(\d+)x(\d+) block=(\d+)x(\d+)x(\d+) "
    r"kernarg=(\d+) barrier=([01])$"
)
SC_RE = re.compile(r"^([SC]) rank=(\d+) upto_seq=(\d+)$")
SEC_RE = re.compile(r"^--- \[WARM SWEEP (\d+)/(\d+)\] (.*) ---$", re.M)
T128_RE = re.compile(
    r"\[BENCH tg128 raw M=1\]\s+([0-9.]+) tok/s.*?"
    r"disp/tok=(\d+).*?vs_base=(\d+)/(\d+) \(first_diff=(-?\d+)\)",
    re.S,
)
T512_RE = re.compile(r"\[BENCH tg512 raw M=1\]\s+([0-9.]+) tok/s")


@dataclass
class Gate:
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def req(self, cond: bool, msg: str) -> bool:
        if not cond:
            self.errors.append(msg)
        return bool(cond)

    def ok(self) -> bool:
        return not self.errors


@dataclass
class Rec:
    kind: str                 # "E" | "S" | "C"
    rank: int
    seq: int
    kernel: str = ""
    gx: int = 0
    gy: int = 0
    gz: int = 0
    bx: int = 0
    by: int = 0
    bz: int = 0
    kernarg: int = 0
    barrier: int = 0


# --------------------------------------------------------------------------- log
def parse_dispatch_log(path: Path, gate: Gate) -> list[Rec]:
    recs: list[Rec] = []
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError as exc:
        gate.req(False, f"cannot read dispatch log {path}: {exc}")
        return recs
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        m = E_RE.match(line)
        if m:
            recs.append(
                Rec(
                    "E", int(m.group(1)), int(m.group(2)), m.group(3),
                    int(m.group(4)), int(m.group(5)), int(m.group(6)),
                    int(m.group(7)), int(m.group(8)), int(m.group(9)),
                    int(m.group(10)), int(m.group(11)),
                )
            )
            continue
        m = SC_RE.match(line)
        if m:
            recs.append(Rec(m.group(1), int(m.group(2)), int(m.group(3))))
            continue
        gate.req(False, f"unrecognised dispatch-log line {n}: {line[:160]!r}")
    return recs


def _e_key(r: Rec) -> tuple[int, int, int, int, int, int, int]:
    return (r.gx, r.gy, r.gz, r.bx, r.by, r.bz, r.barrier)


def check_dispatch(
    recs: list[Rec],
    gdn_layers: int,
    steps: int | None,
    gate: Gate,
) -> dict:
    fused_all = [i for i, r in enumerate(recs) if r.kind == "E" and r.kernel == FUSED]
    st1_all = [i for i, r in enumerate(recs) if r.kind == "E" and r.kernel == ST1]
    st2_all = [i for i, r in enumerate(recs) if r.kind == "E" and r.kernel == ST2]

    counts = {
        "total_records": len(recs),
        "fused_total": len(fused_all),
        "stage1_total": len(st1_all),
        "stage2_total": len(st2_all),
    }

    if not gate.req(bool(st1_all), "selector did not reach stage 1: no "
                    f"{ST1} enqueue in the whole run"):
        return counts

    first_st1 = st1_all[0]
    fused_before = [i for i in fused_all if i < first_st1]
    fused_after = [i for i in fused_all if i >= first_st1]
    counts.update(
        fused_before=len(fused_before),
        fused_after=len(fused_after),
        first_stage1_record=first_st1,
        first_stage1_seq=recs[first_st1].seq,
    )

    per_step = 2 * gdn_layers  # two ranks per GDN layer
    if steps is not None:
        expected = per_step * steps
        counts["expected_per_variant"] = expected
        gate.req(
            len(fused_before) == expected,
            f"baseline phase has {len(fused_before)} fused GDN enqueues; expected {expected} "
            f"({gdn_layers} layers x 2 ranks x {steps} M=1 steps)",
        )
        gate.req(
            len(st1_all) == expected,
            f"candidate stage1 has {len(st1_all)} enqueues; expected {expected}",
        )
        gate.req(
            len(st2_all) == expected,
            f"candidate stage2 has {len(st2_all)} enqueues; expected {expected}",
        )
    else:
        gate.req(
            len(fused_before) == len(st1_all) == len(st2_all) and len(st1_all) > 0,
            "fused-before/stage1/stage2 counts differ: "
            f"{len(fused_before)}/{len(st1_all)}/{len(st2_all)}",
        )
        gate.req(
            len(st1_all) % per_step == 0,
            f"stage1 count {len(st1_all)} is not a multiple of {per_step}",
        )

    gate.req(len(st1_all) == len(st2_all),
             f"stage1={len(st1_all)} and stage2={len(st2_all)} counts differ")
    gate.req(not fused_after,
             f"candidate phase still enqueued the fused Split-V=4 kernel {len(fused_after)} time(s) "
             "after the first stage1 (silent fallback / partial selection)")

    # Per-rank counts must be symmetric (36 GDN layers on each of two ranks).
    for name, idxs, exp in (
        ("fused_before", fused_before, steps),
        ("stage1", st1_all, steps),
        ("stage2", st2_all, steps),
    ):
        c = Counter(recs[i].rank for i in idxs)
        if exp is not None:
            want = per_step * exp // 2
            gate.req(c.get(0, 0) == want and c.get(1, 0) == want,
                     f"{name} per-rank counts are {dict(c)}; expected rank0=rank1={want}")
        else:
            gate.req(c.get(0, 0) == c.get(1, 0),
                     f"{name} per-rank counts are asymmetric: {dict(c)}")
        counts[f"{name}_per_rank"] = dict(c)

    # Every stage1 must be immediately followed by its same-rank stage2.
    bad_pairs = []
    for i in st1_all:
        if i + 1 >= len(recs):
            bad_pairs.append((i, "end of log"))
            continue
        nxt = recs[i + 1]
        if nxt.kind != "E" or nxt.kernel != ST2 or nxt.rank != recs[i].rank:
            bad_pairs.append((i, f"{nxt.kind}/{nxt.kernel}/rank={nxt.rank}"))
    gate.req(not bad_pairs,
             "stage1 is not immediately followed by same-rank stage2 at record(s): "
             + ", ".join(f"{i}->{what}" for i, what in bad_pairs[:4]))

    # Shapes / barriers: exactly the two 256x256 dispatches, all AQL barrier packets.
    for name, idxs, grid in (("stage1", st1_all, ST_GRID),
                             ("stage2", st2_all, ST_GRID),
                             ("fused", fused_before, FU_GRID)):
        bad = []
        for i in idxs:
            r = recs[i]
            if (r.gx, r.gy, r.gz) != grid or (r.bx, r.by, r.bz) != BLOCK or r.barrier != 1:
                bad.append((i, f"grid={r.gx}x{r.gy}x{r.gz} block={r.bx}x{r.by}x{r.bz} barrier={r.barrier}"))
        gate.req(not bad,
                 f"{name} has unexpected launch shape/barrier at: "
                 + ", ".join(f"{i}({w})" for i, w in bad[:4]))

    # Non-zero host kernarg buffers (exact sizes are recorded, not gated: they are
    # the host struct sizes, 104/48 for A2 and 144 for the fused kernel).
    bad_args = [(i, recs[i].kernarg) for i in st1_all + st2_all + fused_before if recs[i].kernarg == 0]
    gate.req(not bad_args, f"zero-length kernarg at record(s) {bad_args[:4]}")

    counts["kernarg_sizes"] = {
        "stage1": sorted({recs[i].kernarg for i in st1_all}),
        "stage2": sorted({recs[i].kernarg for i in st2_all}),
        "fused": sorted({recs[i].kernarg for i in fused_before}),
    }

    # Canonical enqueue sequence: one baseline fused record == one candidate
    # stage1+stage2 pair.  Everything else (kernel, grid, block, barrier, S/C
    # order) must match exactly.  seq/kernarg are dropped because the extra
    # packet changes them by construction.
    def canon_e(r: Rec, gdn: bool) -> str:
        if gdn:
            return (f"E rank={r.rank} kernel=GDN_CONVREC grid={FU_GRID[0]}x1x1 "
                    f"block={BLOCK[0]}x1x1 barrier=1")
        return (f"E rank={r.rank} kernel={r.kernel} grid={r.gx}x{r.gy}x{r.gz} "
                f"block={r.bx}x{r.by}x{r.bz} barrier={r.barrier}")

    def canon_sc(r: Rec) -> str:
        return f"{r.kind} rank={r.rank}"

    base_seq: list[str] = []
    if fused_before:
        for r in recs[fused_before[0]: fused_before[-1] + 1]:
            if r.kind == "E":
                base_seq.append(canon_e(r, r.kernel == FUSED))
            else:
                base_seq.append(canon_sc(r))

    cand_seq: list[str] = []
    if st1_all:
        end = st1_all[-1] + 1  # include the last stage2
        i = st1_all[0]
        while i <= end:
            r = recs[i]
            if r.kind == "E" and r.kernel == ST1:
                cand_seq.append(canon_e(r, True))
                i += 2  # validated: the next record is its stage2
                continue
            if r.kind == "E" and r.kernel == ST2:
                gate.req(False, f"unpaired stage2 at record {i} in candidate slice")
                i += 1
                continue
            if r.kind == "E":
                cand_seq.append(canon_e(r, False))
            else:
                cand_seq.append(canon_sc(r))
            i += 1

    aligned = gate.req(len(base_seq) == len(cand_seq),
                       f"canonical sequence length differs: baseline={len(base_seq)} "
                       f"candidate={len(cand_seq)}")
    if aligned:
        for k, (a, b) in enumerate(zip(base_seq, cand_seq)):
            if a != b:
                gate.req(False,
                         f"canonical enqueue sequence differs at index {k}:\n"
                         f"  baseline : {a}\n  candidate: {b}")
                break
    counts["canonical_records"] = len(base_seq)
    return counts


# ------------------------------------------------------------------------ stdout
def _load_json_lines(text: str, prefix: str, gate: Gate) -> list[dict]:
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        if line.startswith(prefix):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                gate.req(False, f"bad JSON at stdout line {n}: {exc}")
    return out


def _norm_label(label: str) -> str:
    toks = [t for t in label.split() if t not in ("--gdn-splitv8", "--no-gdn-splitv8")]
    return " ".join(toks)


def check_stdout(text: str, exit_code: int, require_tg512: bool, gate: Gate) -> dict:
    data: dict = {}
    gate.req(exit_code == 0, f"engine exit code {exit_code} (expected 0)")

    sections = list(SEC_RE.finditer(text))
    gate.req(len(sections) == 2, f"expected 2 warm-sweep sections, found {len(sections)}")
    if len(sections) != 2:
        return data
    if not (sections[0].group(1) == "1" and sections[1].group(1) == "2"):
        gate.req(False, "warm-sweep section numbering is not 1/2, 2/2")
    labels = [sections[0].group(3).strip(), sections[1].group(3).strip()]
    data["labels"] = labels
    gate.req("--no-gdn-splitv8" in labels[0],
             f"section 1 is not the baseline (label={labels[0]!r})")
    gate.req("--gdn-splitv8" in labels[1] and "--no-gdn-splitv8" not in labels[1],
             f"section 2 is not the candidate (label={labels[1]!r})")
    gate.req(_norm_label(labels[0]) == _norm_label(labels[1]),
             f"variants are not matched: baseline={labels[0]!r} candidate={labels[1]!r}")

    for si, sec in enumerate(sections):
        body = text[sec.start(): sections[si + 1].start() if si + 1 < len(sections) else len(text)]
        m128 = T128_RE.search(body)
        if not gate.req(m128 is not None, f"section {si + 1}: no tg128 line"):
            continue
        tps, disp, eq, total, first_diff = m128.groups()
        rec = {
            "tg128_tps": float(tps),
            "disp_per_tok": int(disp),
            "vs_base": f"{eq}/{total}",
            "first_diff": int(first_diff),
        }
        gate.req(rec["tg128_tps"] > 0.0, f"section {si + 1}: tg128 tok/s is not positive")
        gate.req(int(eq) == 129 and int(total) == 129,
                 f"section {si + 1}: vs_base={eq}/{total}, expected 129/129")
        gate.req(int(first_diff) == -1,
                 f"section {si + 1}: first_diff={first_diff}, expected -1")
        m512 = T512_RE.search(body)
        if require_tg512:
            if gate.req(m512 is not None, f"section {si + 1}: VEGAJET_RAW_TG512=1 but no tg512 line"):
                rec["tg512_tps"] = float(m512.group(1))
                gate.req(rec["tg512_tps"] > 0.0,
                         f"section {si + 1}: tg512 tok/s is not positive")
        elif m512:
            rec["tg512_tps"] = float(m512.group(1))

        variants = _load_json_lines(body, '{"event":"variant"', gate)
        if not gate.req(len(variants) == 1,
                        f"section {si + 1}: expected one variant JSON, found {len(variants)}"):
            continue
        v = variants[0]
        rec["variant"] = v
        gate.req(v.get("label") == labels[si],
                 f"section {si + 1}: variant label mismatch: {v.get('label')!r} != {labels[si]!r}")
        gate.req(v.get("ok") is True, f"section {si + 1}: variant ok != true")
        gate.req(v.get("parity") == "32/32",
                 f"section {si + 1}: golden parity {v.get('parity')!r} != '32/32'")
        gate.req(int(v.get("ecc_ue", 0)) == 0 and int(v.get("ecc_de", 0)) == 0,
                 f"section {si + 1}: ECC deltas ue={v.get('ecc_ue')} de={v.get('ecc_de')}")
        gate.req(float(v.get("tmem_c", 999.0)) < 65.0,
                 f"section {si + 1}: TMem {v.get('tmem_c')} C >= 65 C")
        data[f"section{si + 1}"] = rec

    if len(data.get("section1", {})) and len(data.get("section2", {})):
        gate.req(data["section1"]["disp_per_tok"] == data["section2"]["disp_per_tok"],
                 "disp/tok differs between baseline and candidate "
                 f"({data['section1']['disp_per_tok']} vs {data['section2']['disp_per_tok']}); "
                 "the matched non-super path was not used")

    summaries = _load_json_lines(text, '{"event":"sweep_summary"', gate)
    if gate.req(len(summaries) == 1, f"expected one sweep_summary JSON, found {len(summaries)}"):
        sv = summaries[0].get("variants", [])
        if gate.req(len(sv) == 2, f"sweep_summary has {len(sv)} variants"):
            data["summary"] = sv
            gate.req(sv[0].get("label") == labels[0] and sv[1].get("label") == labels[1],
                     "sweep_summary labels do not match the printed sections")
            for i, v in enumerate(sv):
                gate.req(v.get("ok") is True, f"summary variant {i + 1}: ok != true")
                gate.req(v.get("parity_ok") is True, f"summary variant {i + 1}: parity_ok != true")
                gate.req(v.get("lossless") == "32/32",
                         f"summary variant {i + 1}: lossless {v.get('lossless')!r} != '32/32'")
                gate.req(int(v.get("ecc_ue", 0)) == 0 and int(v.get("ecc_de", 0)) == 0,
                         f"summary variant {i + 1}: ECC deltas non-zero")
                gate.req(float(v.get("tmem_c", 999.0)) < 65.0,
                         f"summary variant {i + 1}: TMem {v.get('tmem_c')} C >= 65 C")
            gate.req(sv[0].get("disp_per_tok") == sv[1].get("disp_per_tok"),
                     "summary disp_per_tok differs between variants")

    telem = [ln for ln in text.splitlines() if ln.startswith("[TELEM]")]
    if gate.req(bool(telem), "no final [TELEM] line"):
        last = telem[-1]
        gate.req("-> FAIL" not in last, f"final telemetry FAIL: {last}")
        gate.req(("-> PASS" in last) or ("-> SKIPPED" in last),
                 f"final telemetry line is neither PASS nor SKIPPED: {last}")
        data["telem"] = last
    return data


# ------------------------------------------------------------------------- verdict
def analyze(args) -> tuple[Gate, dict]:
    gate = Gate()
    summary: dict = {
        "tool": "check_a2_sweep",
        "gdn_layers": args.gdn_layers,
        "steps": args.steps,
        "require_tg512": args.require_tg512,
    }
    if args.stdout:
        try:
            text = Path(args.stdout).read_text(errors="replace")
        except OSError as exc:
            gate.req(False, f"cannot read stdout log {args.stdout}: {exc}")
            text = ""
        summary["stdout"] = check_stdout(text, args.exit_code, args.require_tg512, gate)
    else:
        gate.req(False, "no --stdout supplied")
    if args.dispatch_log:
        recs = parse_dispatch_log(Path(args.dispatch_log), gate)
        summary["dispatch"] = check_dispatch(recs, args.gdn_layers, args.steps, gate)
    else:
        gate.req(False, "no --dispatch-log supplied")
    summary["status"] = "PASS" if gate.ok() else "FAIL"
    summary["errors"] = gate.errors
    summary["notes"] = gate.notes
    if len(summary.get("stdout", {})) and len(summary.get("dispatch", {})):
        s1 = summary["stdout"].get("section1", {})
        s2 = summary["stdout"].get("section2", {})
        if s1 and s2:
            summary["timing"] = {
                "baseline_tg128_tps": s1["tg128_tps"],
                "candidate_tg128_tps": s2["tg128_tps"],
                "tg128_ratio": (s2["tg128_tps"] / s1["tg128_tps"]) if s1["tg128_tps"] else None,
                "baseline_tg512_tps": s1.get("tg512_tps"),
                "candidate_tg512_tps": s2.get("tg512_tps"),
                "tg512_ratio": ((s2["tg512_tps"] / s1["tg512_tps"])
                                if s1.get("tg512_tps") and s2.get("tg512_tps") else None),
            }
    return gate, summary


def print_verdict(gate: Gate, summary: dict) -> None:
    d = summary.get("dispatch", {})
    print(f"A2 matched sweep: {summary['status']}")
    if d:
        print(f"  fused(baseline)={d.get('fused_before')} "
              f"stage1={d.get('stage1_total')} stage2={d.get('stage2_total')} "
              f"fused_after_first_stage1={d.get('fused_after')} "
              f"per_rank={d.get('stage1_per_rank')}")
        print(f"  launch: stage grid={ST_GRID[0]}x1x1 block={BLOCK[0]} barrier=1; "
              f"fused grid={FU_GRID[0]}x1x1; canonical_records={d.get('canonical_records')}")
        if d.get("kernarg_sizes"):
            print(f"  host kernarg bytes: {d['kernarg_sizes']}")
    t = summary.get("timing", {})
    if t:
        print(f"  tg128: base {t['baseline_tg128_tps']:.2f} -> cand {t['candidate_tg128_tps']:.2f} tok/s "
              f"(ratio {t['tg128_ratio']:.3f})")
        if t.get("baseline_tg512_tps"):
            print(f"  tg512: base {t['baseline_tg512_tps']:.2f} -> cand {t['candidate_tg512_tps']:.2f} tok/s "
                  f"(ratio {t['tg512_ratio']:.3f})")
    if gate.errors:
        print("  FAILURES:")
        for e in gate.errors:
            print(f"   - {e}")


# ------------------------------------------------------------------------ self-test
def _synth_stdout(pass_case: bool, mutate: str | None) -> str:
    lines: list[str] = []
    for idx, label in enumerate(("--no-gdn-splitv8 --bench-mode raw",
                                 "--gdn-splitv8 --bench-mode raw"), 1):
        lines.append(f"--- [WARM SWEEP {idx}/2] {label} ---")
        tps = 100.0 if idx == 1 else 91.0
        vs = "129/129" if idx == 1 else ("129/129" if pass_case or mutate != "vs_base" else "128/129")
        fd = "-1" if (idx == 1 or pass_case or mutate != "vs_base") else "7"
        lines.append(f"[BENCH tg128 raw M=1] {tps:.2f} tok/s (10.00 ms/tok) | disp/tok=104 | "
                     f"vs_base={vs} (first_diff={fd})")
        if mutate != "no_tg512":
            lines.append(f"[BENCH tg512 raw M=1] {tps - 5:.2f} tok/s (11.00 ms/tok) | disp/tok=104")
        ok = "true" if (pass_case or mutate not in ("variant_ok", "ecc")) else "false"
        parity = "32/32" if (pass_case or mutate != "parity") else "31/32"
        ue = 0 if (pass_case or mutate != "ecc") else 1
        lines.append(
            '{"event":"variant","label":"%s","ok":%s,"primary_tps":%.2f,'
            '"raw_tg128_tps":%.2f,"spec_tg128_tps":0.00,"pp512_tps":0.00,'
            '"pp2048_tps":0.00,"parity":"%s","tmem_c":50.0,"ecc_ue":%d,"ecc_de":0}'
            % (label, ok, tps, tps, parity, ue)
        )
    sv = []
    for idx, label in enumerate(("--no-gdn-splitv8 --bench-mode raw",
                                 "--gdn-splitv8 --bench-mode raw")):
        sv.append(
            '{"idx":%d,"label":"%s","ok":true,"parity_ok":%s,"primary_tps":%.2f,'
            '"delta_pct":0.0,"raw_tg128_tps":%.2f,"spec_tg128_tps":0.00,'
            '"pp512_tps":0.00,"pp2048_tps":0.00,"disp_per_tok":104,"lossless":"%s",'
            '"first_diff":-1,"tmem_c":50.0,"ecc_ue":0,"ecc_de":0}'
            % (idx, label, "true" if (pass_case or mutate != "summary_parity") else "false",
               100.0 if idx == 0 else 91.0, 100.0 if idx == 0 else 91.0,
               "32/32" if (pass_case or mutate != "summary_parity") else "31/32")
        )
    lines.append('{"event":"sweep_summary","variants":[%s]}' % ",".join(sv))
    lines.append("[TELEM] power1_input: worst 10 s rolling avg=100.0 W (mean=90.0 W, <=205 W) | "
                 "worst 1 s avg=110.0 W (info) | instantaneous peak=120.0 W (<270 W) | "
                 "max_TMem=50.0 C (<65 C) | 100 samples -> PASS")
    return "\n".join(lines) + "\n"


def _synth_log(pass_case: bool, mutate: str | None) -> str:
    lines: list[str] = []
    seq = 0

    def e(rank: int, kernel: str, grid: int, kernarg: int, barrier: int = 1) -> None:
        nonlocal seq
        seq += 1
        lines.append(f"E rank={rank} seq={seq} kernel={kernel} grid={grid}x1x1 "
                     f"block=256x1x1 kernarg={kernarg} barrier={barrier}")

    def sc(kind: str, rank: int) -> None:
        lines.append(f"{kind} rank={rank} upto_seq={seq}")

    # load-time prologue (outside both compared slices)
    e(0, "kfd_memcpy_u32", 256, 24)
    e(1, "kfd_memcpy_u32", 256, 24)

    for phase in ("base", "cand"):
        for step in range(3):
            for layer in range(2):
                for rank in (0, 1):
                    e(rank, "qkv_proj", 256, 40)
                    if phase == "base":
                        e(rank, FUSED, FU_GRID[0], 144)
                    else:
                        g1 = ST_GRID[0]
                        if mutate == "bad_grid" and step == 1 and layer == 0 and rank == 0:
                            g1 = 32768
                        e(rank, ST1, g1, 104)
                        if mutate == "drop_stage2" and step == 1 and layer == 0 and rank == 0:
                            pass
                        elif mutate == "bad_order" and step == 1 and layer == 0 and rank == 0:
                            e(rank, "out_proj", 256, 40)
                            e(rank, ST2, ST_GRID[0], 48)
                        else:
                            e(rank, ST2, ST_GRID[0], 48)
                        if mutate == "fallback" and step == 2 and layer == 1 and rank == 1:
                            # replace the pair with the fused kernel: stage1/2 remain
                            # above, and the fused name appears after the first stage1.
                            e(rank, FUSED, FU_GRID[0], 144)
                    if not (mutate == "bad_order" and phase == "cand" and step == 1 and layer == 0 and rank == 0):
                        e(rank, "out_proj", 256, 40)
                    elif phase == "base":
                        e(rank, "out_proj", 256, 40)
                    if mutate == "kernel_diff" and phase == "cand" and step == 1 and layer == 0 and rank == 1:
                        # replace a normal kernel name inside the compared candidate slice
                        lines[-1] = lines[-1].replace("out_proj", "out_proj_ALT")
            e(0, "vocab", 256, 40)
            e(1, "vocab", 256, 40)
            sc("S", 0)
            sc("S", 1)
            sc("C", 0)
            sc("C", 1)
    # epilogue outside both slices
    e(0, "kfd_memcpy_u32", 256, 24)
    return "\n".join(lines) + "\n"


def self_test() -> int:
    cases = [
        (True, None, True),
        (False, "drop_stage2", False),
        (False, "fallback", False),
        (False, "bad_grid", False),
        (False, "bad_order", False),
        (False, "kernel_diff", False),
        (False, "vs_base", False),
        (False, "parity", False),
        (False, "no_tg512", False),
        (False, "variant_ok", False),
        (False, "ecc", False),
        (False, "summary_parity", False),
    ]
    failures = 0
    with tempfile.TemporaryDirectory(prefix="a2-selftest-") as td:
        for i, (pass_case, mutate, expect) in enumerate(cases, 1):
            stdout = Path(td) / f"case{i}.stdout"
            dispatch = Path(td) / f"case{i}.dispatch"
            stdout.write_text(_synth_stdout(pass_case, mutate))
            dispatch.write_text(_synth_log(pass_case, mutate))
            ns = argparse.Namespace(
                stdout=str(stdout),
                dispatch_log=str(dispatch),
                exit_code=0,
                gdn_layers=2,
                steps=3,
                require_tg512=True,
            )
            gate, _ = analyze(ns)
            got = gate.ok()
            label = "pass" if pass_case else f"mutate={mutate}"
            if got != expect:
                failures += 1
                print(f"  self-test {i} [{label}] FAIL: expected {'PASS' if expect else 'FAIL'}, "
                      f"got {'PASS' if got else 'FAIL'}")
                for err in gate.errors:
                    print(f"    - {err}")
            else:
                print(f"  self-test {i} [{label}] ok ({'PASS' if got else 'FAIL'})")
    if failures:
        print(f"self-test: {failures} case(s) failed")
        return 1
    print(f"self-test: {len(cases)} cases passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stdout", help="sweep stdout log")
    ap.add_argument("--dispatch-log", help="VEGAJET_DISPATCH_LOG journal")
    ap.add_argument("--exit-code", type=int, default=0, help="engine exit code")
    ap.add_argument("--gdn-layers", type=int, default=36)
    ap.add_argument("--steps", type=int, default=None,
                    help="M=1 steps per variant (671 with tg512, 159 without)")
    ap.add_argument("--require-tg512", action="store_true")
    ap.add_argument("--json-out", help="write the machine-readable verdict here")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.stdout or not args.dispatch_log:
        ap.error("--stdout and --dispatch-log are required unless --self-test")
    gate, summary = analyze(args)
    print_verdict(gate, summary)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if gate.ok() else 1


if __name__ == "__main__":
    sys.exit(main())
