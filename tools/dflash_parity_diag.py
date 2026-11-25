#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""DFlash2 acceptance parity diagnosis for Issue #152 (standard library only).

Compares the engine's speculative verify cycles with the CPU reference drafter's on the same prompt and says where
they part:

  * acceptance: mean accepted drafts per full cycle for both sides and the Issue #152 gate on B = 8
    (>= 4.0 PASS, < 3.5 ABORT -> flagged, else MARGINAL), plus the accepted-length histogram;
  * first divergence: the first anchor position whose engine drafts differ from the reference drafts, and the
    draft slot where they first differ (cycles are aligned by anchor position);
  * layer: with per-layer hidden dumps of that cycle (.npy, float32/float16, shape [layers][B][D], or
    [cycles][layers][B][D] indexed by aligned cycle), the first drafter layer and block slot whose relative L2
    error exceeds --tol.

Inputs (paths from the command line only):
  engine log     lines printed with VEGAJET_DFLASH2_DEBUG=1:
                   [DFVER pos=<p> acc=<a>/<k>] tgt=[t0,...] dft=[d0,...]
                 (the engine prints the first 5 entries; the logged acc/k is used for its acceptance)
  reference log  `dflash_ref run ... --verbose` cycle lines:  p=<p> anchor=<id> acc=<a> draft: d1* d2 ... | target: t1 ...
  either side may instead be JSON lines: {"pos": p, "draft": [...], "target": [...], "accepted": a (optional)}

Usage:
  tools/dflash_parity_diag.py --engine <log> --reference <log> [--block 8] [--engine-layers <npy>
                              --reference-layers <npy>] [--tol 1e-3] [--json]
  tools/dflash_parity_diag.py --self-test
Exit: 0 = engine not flagged, 1 = engine mean accepted < 3.5 (flagged), 2 = usage / input error.
Conventions match core/dflash_host.hpp (accepted = leading drafts equal to the target's greedy tokens).
"""
import argparse
import array
import ast
import json
import math
import os
import re
import struct
import sys
import tempfile

GATE_PASS, GATE_ABORT = 4.0, 3.5

ENGINE_RE = re.compile(r"\[DFVER pos=(\d+) acc=(\d+)/(\d+)\]\s*tgt=\[([-\d,\s]*)\]\s*dft=\[([-\d,\s]*)\]")
REF_RE = re.compile(r"^\s*p=\s*(\d+)\s+anchor=\s*(-?\d+)\s+acc=\s*(\d+)\s+draft:([^|]*)\|\s*target:(.*)$")


class Cycle:
    __slots__ = ("pos", "draft", "target", "accepted", "n_draft")

    def __init__(self, pos, draft, target, accepted=None, n_draft=None):
        self.pos, self.draft, self.target = pos, list(draft), list(target)
        self.n_draft = len(self.draft) if n_draft is None else n_draft
        self.accepted = accepted_prefix(self.draft, self.target) if accepted is None else accepted


def accepted_prefix(draft, target):
    a = 0
    while a < min(len(draft), len(target)) and draft[a] == target[a]:
        a += 1
    return a


def _ints(text):
    return [int(x) for x in re.findall(r"-?\d+", text)]


def parse_cycles(path):
    """Engine DFVER lines, reference verbose lines or JSON lines; returns (kind, [Cycle])."""
    cycles, kinds = [], set()
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            m = ENGINE_RE.search(s)
            if m:
                pos, acc, k = int(m.group(1)), int(m.group(2)), int(m.group(3))
                tgt = [t for t in _ints(m.group(4)) if t >= 0]
                dft = [d for d in _ints(m.group(5)) if d >= 0][:k]
                cycles.append(Cycle(pos, dft, tgt[:k], accepted=acc, n_draft=k))
                kinds.add("engine")
                continue
            m = REF_RE.match(s)
            if m:
                draft = _ints(m.group(4).replace("*", " "))
                cycles.append(Cycle(int(m.group(1)), draft, _ints(m.group(5)), accepted=int(m.group(3)),
                                    n_draft=len(draft)))
                kinds.add("reference")
                continue
            if s.startswith("{"):
                o = json.loads(s)
                cycles.append(Cycle(int(o["pos"]), o.get("draft", []), o.get("target", []), o.get("accepted"),
                                    o.get("n_draft")))
                kinds.add("jsonl")
    return ("/".join(sorted(kinds)) or "empty"), cycles


def gate(mean, n_full):
    if n_full <= 0:
        return "NO_DATA"
    return "PASS" if mean >= GATE_PASS else ("ABORT" if mean < GATE_ABORT else "MARGINAL")


def acceptance(cycles, block):
    full = [c for c in cycles if c.n_draft >= block - 1]
    hist = [0] * block
    for c in full:
        hist[min(c.accepted, block - 1)] += 1
    mean = sum(min(c.accepted, block - 1) for c in full) / len(full) if full else 0.0
    return {"cycles": len(cycles), "full_cycles": len(full), "mean_accepted": mean, "gate": gate(mean, len(full)),
            "hist": hist}


def first_divergence(engine, reference):
    ref_at = {c.pos: c for c in reference}
    aligned = []
    for e in sorted(engine, key=lambda c: c.pos):
        r = ref_at.get(e.pos)
        if r is None:
            continue
        n = min(len(e.draft), len(r.draft))
        slot = next((i for i in range(n) if e.draft[i] != r.draft[i]), -1)
        aligned.append({"pos": e.pos, "slot": slot, "engine_accepted": e.accepted, "reference_accepted": r.accepted})
    first = next((i for i, a in enumerate(aligned) if a["slot"] >= 0 or a["engine_accepted"] != a["reference_accepted"]),
                 -1)
    return aligned, first


# ----------------------------------------------------------------------------------------------- .npy layers
def read_npy(path):
    with open(path, "rb") as fh:
        if fh.read(6) != b"\x93NUMPY":
            raise ValueError("%s: not a .npy file" % path)
        major = fh.read(2)[0]
        hlen = struct.unpack("<H" if major == 1 else "<I", fh.read(2 if major == 1 else 4))[0]
        hdr = ast.literal_eval(fh.read(hlen).decode("latin1"))
        if hdr.get("fortran_order"):
            raise ValueError("%s: Fortran order is not supported" % path)
        shape, descr = tuple(hdr["shape"]), hdr["descr"]
        n = 1
        for s in shape:
            n *= s
        raw = fh.read()
    if descr == "<f4":
        a = array.array("f")
        a.frombytes(raw[:4 * n])
        return shape, list(a)
    if descr == "<f2":
        return shape, [struct.unpack_from("<e", raw, 2 * i)[0] for i in range(n)]
    raise ValueError("%s: unsupported dtype %s (float32 / float16 only)" % (path, descr))


def write_npy(path, shape, flat):
    hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (%s), }" % (", ".join(map(str, shape)) +
                                                                          ("," if len(shape) == 1 else ""))
    hdr += " " * ((64 - (10 + len(hdr) + 1) % 64) % 64) + "\n"
    with open(path, "wb") as fh:
        fh.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(hdr)) + hdr.encode("latin1"))
        fh.write(array.array("f", flat).tobytes())


def rel_l2(a, b):
    num = sum((x - y) * (x - y) for x, y in zip(a, b))
    den = sum(y * y for y in b)
    return math.sqrt(num) / max(math.sqrt(den), 1e-30)


def layer_divergence(eng_path, ref_path, cycle_index, tol):
    es, e = read_npy(eng_path)
    rs, r = read_npy(ref_path)
    if es != rs:
        raise ValueError("layer dumps differ in shape: %s vs %s" % (es, rs))
    if len(es) == 4:
        if not 0 <= cycle_index < es[0]:
            raise ValueError("cycle %d outside the %d dumped cycles" % (cycle_index, es[0]))
        per = es[1] * es[2] * es[3]
        e, r = e[cycle_index * per:(cycle_index + 1) * per], r[cycle_index * per:(cycle_index + 1) * per]
        es = es[1:]
    if len(es) != 3:
        raise ValueError("layer dumps must be [layers][B][D] or [cycles][layers][B][D], got %s" % (es,))
    L, B, D = es
    out = {"layer": -1, "slot": -1, "rel_err": 0.0, "layer_max": []}
    for layer in range(L):
        worst = 0.0
        for s in range(B):
            off = (layer * B + s) * D
            err = rel_l2(e[off:off + D], r[off:off + D])
            worst = max(worst, err)
            if out["layer"] < 0 and err > tol:
                out.update(layer=layer, slot=s, rel_err=err)
        out["layer_max"].append(worst)
    return out


# ----------------------------------------------------------------------------------------------- report
def diagnose(engine, reference, block, layers=None, tol=1e-3, cycle_index=None):
    ea, ra = acceptance(engine, block), acceptance(reference, block)
    aligned, first = first_divergence(engine, reference)
    rep = {"block": block, "engine": ea, "reference": ra, "aligned_cycles": len(aligned),
           "flagged": ea["full_cycles"] > 0 and ea["mean_accepted"] < GATE_ABORT,
           "first_divergence": aligned[first] if first >= 0 else None, "first_divergence_index": first}
    if layers:
        idx = first if cycle_index is None else cycle_index
        rep["layer_divergence"] = layer_divergence(layers[0], layers[1], max(idx, 0), tol)
        rep["layer_cycle_index"] = max(idx, 0)
    return rep


def render(rep):
    e, r = rep["engine"], rep["reference"]
    lines = ["B=%d engine   : mean_accepted_drafts=%.3f over %d full cycles -> %s  hist=%s" % (
                 rep["block"], e["mean_accepted"], e["full_cycles"], e["gate"], e["hist"]),
             "B=%d reference: mean_accepted_drafts=%.3f over %d full cycles -> %s  hist=%s" % (
                 rep["block"], r["mean_accepted"], r["full_cycles"], r["gate"], r["hist"])]
    if rep["block"] != 8:
        lines.append("note: the Issue #152 gate is defined for B=8")
    fd = rep["first_divergence"]
    if fd:
        lines.append("first divergence: pos %d (aligned cycle %d of %d), draft slot %d, accepted engine %d vs reference %d"
                     % (fd["pos"], rep["first_divergence_index"], rep["aligned_cycles"], fd["slot"],
                        fd["engine_accepted"], fd["reference_accepted"]))
    elif rep["aligned_cycles"]:
        lines.append("engine drafts equal the reference on all %d aligned cycles" % rep["aligned_cycles"])
    else:
        lines.append("no cycles at common anchor positions")
    ld = rep.get("layer_divergence")
    if ld:
        if ld["layer"] >= 0:
            lines.append("first diverging drafter layer: %d (block slot %d, rel L2 %.3g) at aligned cycle %d" % (
                ld["layer"], ld["slot"], ld["rel_err"], rep["layer_cycle_index"]))
        else:
            lines.append("drafter layers agree within tolerance at aligned cycle %d" % rep["layer_cycle_index"])
        lines.append("per-layer max rel L2: " + " ".join("%d:%.2g" % (i, v) for i, v in enumerate(ld["layer_max"])))
    lines.append("FLAG: engine mean accepted < %.1f (Issue #152)" % GATE_ABORT if rep["flagged"] else "not flagged")
    return "\n".join(lines)


def self_test():
    with tempfile.TemporaryDirectory(prefix="vj_dfdiag_") as tmp:
        eng = os.path.join(tmp, "engine.log")
        ref = os.path.join(tmp, "ref.log")
        with open(eng, "w") as fe, open(ref, "w") as fr:
            fe.write("[load] noise\n")
            for c in range(10):
                pos = 40 + 6 * c
                tgt = [pos * 10 + i for i in range(8)]
                rd = tgt[:7]
                rd[5] += 1
                ed = list(rd)
                if c >= 4:
                    ed[1] += 3
                acc_e = accepted_prefix(ed, tgt)
                fe.write("  [DFVER pos=%d acc=%d/7] tgt=[%s] dft=[%s]\n" % (pos, acc_e, ",".join(map(str, tgt[:5])),
                                                                          ",".join(map(str, ed[:5]))))
                fr.write("  p=%4d anchor=%6d acc=%2d draft: %s | target: %s\n" % (
                    pos, 7, accepted_prefix(rd, tgt),
                    " ".join("%d%s" % (d, "*" if i < 5 else "") for i, d in enumerate(rd)), " ".join(map(str, tgt[:7]))))
        _, e = parse_cycles(eng)
        _, r = parse_cycles(ref)
        assert len(e) == 10 and len(r) == 10, (len(e), len(r))
        rep = diagnose(e, r, 8)
        assert abs(rep["reference"]["mean_accepted"] - 5.0) < 1e-12, rep["reference"]
        assert abs(rep["engine"]["mean_accepted"] - (4 * 5 + 6 * 1) / 10.0) < 1e-12, rep["engine"]
        assert rep["flagged"] and rep["engine"]["gate"] == "ABORT" and rep["reference"]["gate"] == "PASS"
        assert rep["first_divergence"]["pos"] == 64 and rep["first_divergence"]["slot"] == 1, rep["first_divergence"]
        L, B, D = 4, 8, 6
        base = [math.sin(i * 0.37) for i in range(L * B * D)]
        bad = list(base)
        for d in range(D):
            bad[(2 * B + 3) * D + d] *= 1.1
        write_npy(os.path.join(tmp, "e.npy"), (L, B, D), bad)
        write_npy(os.path.join(tmp, "r.npy"), (L, B, D), base)
        rep = diagnose(e, r, 8, (os.path.join(tmp, "e.npy"), os.path.join(tmp, "r.npy")), 1e-3)
        ld = rep["layer_divergence"]
        assert ld["layer"] == 2 and ld["slot"] == 3 and abs(ld["rel_err"] - 0.1) < 1e-5, ld
        assert "first diverging drafter layer: 2" in render(rep)
    print("dflash_parity_diag self-test: PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--engine")
    ap.add_argument("--reference")
    ap.add_argument("--block", type=int, default=8, help="B = drafts per cycle + 1 (default 8)")
    ap.add_argument("--engine-layers")
    ap.add_argument("--reference-layers")
    ap.add_argument("--cycle-index", type=int, help="aligned cycle of a 4-D layer dump (default: first divergence)")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.engine or not a.reference:
        ap.print_usage(sys.stderr)
        return 2
    if bool(a.engine_layers) != bool(a.reference_layers):
        print("dflash_parity_diag: --engine-layers and --reference-layers go together", file=sys.stderr)
        return 2
    if a.block < 2:
        print("dflash_parity_diag: --block must be >= 2", file=sys.stderr)
        return 2
    try:
        _, e = parse_cycles(a.engine)
        _, r = parse_cycles(a.reference)
        layers = (a.engine_layers, a.reference_layers) if a.engine_layers else None
        rep = diagnose(e, r, a.block, layers, a.tol, a.cycle_index)
    except (OSError, ValueError, KeyError) as exc:
        print("dflash_parity_diag: %s" % exc, file=sys.stderr)
        return 2
    print(json.dumps(rep, indent=1) if a.json else render(rep))
    return 1 if rep["flagged"] else 0


if __name__ == "__main__":
    sys.exit(main())
