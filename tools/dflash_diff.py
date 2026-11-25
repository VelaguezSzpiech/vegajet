#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""DFlash drafter conditioning diff (standard library only; numpy is not required).

Diagnoses the three places where an engine-side DFlash drafter usually diverges from its reference:

  taps   Compare target-model residual taps between two dumps. The drafter reads the outputs of target layers
         [2, 8, 15, 21, 27, 33, 40, 46] by default. In a VJHS dump, state 0 is the embedding output and state i is
         the output of layer i-1, so tap layer L is state L + offset with offset = 1. `--offset auto` scores offsets
         0 and 1 and reports which one matches (the classic off-by-one).
  fc     Recompute the drafter input from the taps with the fc projection (D x 8D) and the hidden RMSNorm, in each
         candidate order (norm(fc(cat)), fc(cat(norm(tap_i))), fc(norm(cat))), and report which order matches a
         dumped drafter input. Uses a row sample, so it stays fast in pure Python.
  mask   Rebuild the draft-block attention mask (block-bidirectional, sliding window W on SWA layers) and diff it
         against a dumped mask; both window conventions (q - k < W, the default, and q - k <= W) are scored so an
         off-by-one window is named as such.

Inputs: VJHS dumps (`vegajet bench --dump-hidden`: "VJHS", u32 version=1, n_tokens, n_states, D; i32 ids[n_tokens];
float32 states[n_tokens][n_states][D]) and .npy arrays (float32/float16/bool/int32/uint8, C order).
File paths are taken from the command line only.
"""
import argparse
import array
import ast
import math
import random
import struct
import sys

DEFAULT_TAPS = [2, 8, 15, 21, 27, 33, 40, 46]
VJHS_MAGIC = 0x53484A56

# ----------------------------------------------------------------------------------------------- file formats


def read_npy(path):
    """Returns (shape, flat list). Supports <f4, <f2, |b1, <i4, |u1, C order."""
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
        a = array.array("f"); a.frombytes(raw[:4 * n]); data = list(a)
    elif descr == "<f2":
        data = [struct.unpack_from("<e", raw, 2 * i)[0] for i in range(n)]
    elif descr in ("|b1", "|u1"):
        data = list(raw[:n])
    elif descr == "<i4":
        a = array.array("i"); a.frombytes(raw[:4 * n]); data = list(a)
    else:
        raise ValueError("%s: unsupported dtype %s" % (path, descr))
    if len(data) != n:
        raise ValueError("%s: truncated payload" % path)
    return shape, data


def write_npy(path, shape, data, descr="<f4"):
    hdr = "{'descr': '%s', 'fortran_order': False, 'shape': %s, }" % (descr, repr(tuple(shape)))
    pad = 64 - (10 + len(hdr) + 1) % 64
    hdr = hdr + " " * pad + "\n"
    with open(path, "wb") as fh:
        fh.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(hdr)) + hdr.encode("latin1"))
        if descr == "<f4":
            fh.write(array.array("f", data).tobytes())
        elif descr in ("|b1", "|u1"):
            fh.write(bytes(int(x) for x in data))
        elif descr == "<i4":
            fh.write(array.array("i", data).tobytes())
        else:
            raise ValueError("unsupported dtype " + descr)


class Vjhs:
    """Lazy reader: states are read per (token, state) slice."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as fh:
            hdr = fh.read(20)
            if len(hdr) != 20:
                raise ValueError("%s: truncated header" % path)
            magic, ver, self.n_tok, self.n_states, self.dim = struct.unpack("<5I", hdr)
            if magic != VJHS_MAGIC or ver != 1:
                raise ValueError("%s: not a VJHS v1 dump" % path)
            ids = array.array("i"); ids.frombytes(fh.read(4 * self.n_tok)); self.ids = list(ids)
        self.base = 20 + 4 * self.n_tok
        need = self.base + 4 * self.n_tok * self.n_states * self.dim
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            if fh.tell() < need:
                raise ValueError("%s: truncated payload (%d < %d bytes)" % (path, fh.tell(), need))

    def state(self, tok, s):
        with open(self.path, "rb") as fh:
            fh.seek(self.base + 4 * ((tok * self.n_states + s) * self.dim))
            a = array.array("f"); a.frombytes(fh.read(4 * self.dim))
            return list(a)


def write_vjhs(path, ids, states):
    """states[t][s] = list of D floats."""
    n_tok, n_states, dim = len(states), len(states[0]), len(states[0][0])
    with open(path, "wb") as fh:
        fh.write(struct.pack("<5I", VJHS_MAGIC, 1, n_tok, n_states, dim))
        fh.write(array.array("i", ids).tobytes())
        for t in range(n_tok):
            for s in range(n_states):
                fh.write(array.array("f", states[t][s]).tobytes())


class TapSource:
    """Uniform access to taps: a VJHS dump (49 states) or a .npy [n_tok, n_taps_or_states, D]."""

    def __init__(self, path):
        if path.endswith(".npy"):
            shape, data = read_npy(path)
            if len(shape) != 3:
                raise ValueError("%s: expected [tokens, states, D]" % path)
            self.n_tok, self.n_states, self.dim = shape
            self._data, self._v = data, None
        else:
            self._v = Vjhs(path)
            self.n_tok, self.n_states, self.dim = self._v.n_tok, self._v.n_states, self._v.dim

    def state(self, tok, s):
        if self._v:
            return self._v.state(tok, s)
        o = (tok * self.n_states + s) * self.dim
        return self._data[o:o + self.dim]

# ----------------------------------------------------------------------------------------------- math


def stats(a, b):
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
    dot = sum(x * y for x, y in zip(a, b))
    cos = dot / (na * nb) if na and nb else float(na == nb)
    max_abs = max(abs(x - y) for x, y in zip(a, b)) if a else 0.0
    rel = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))) / nb if nb else 0.0
    return cos, max_abs, rel


def rmsnorm(x, w, eps):
    inv = 1.0 / math.sqrt(sum(v * v for v in x) / len(x) + eps)
    return [v * inv * (w[i] if w else 1.0) for i, v in enumerate(x)]

# ----------------------------------------------------------------------------------------------- commands


def cmd_taps(args):
    a, b = TapSource(args.a), TapSource(args.b)
    layers = [int(x) for x in args.layers.split(",")]
    if a.dim != b.dim:
        raise SystemExit("dimension mismatch: %d vs %d" % (a.dim, b.dim))
    toks = min(a.n_tok, b.n_tok, args.tokens) if args.tokens else min(a.n_tok, b.n_tok)
    # B may hold only the taps (n_states == len(layers)) or all states.
    b_taps_only = b.n_states == len(layers)

    def b_state(t, i, layer, off):
        return b.state(t, i) if b_taps_only else b.state(t, layer + off)

    offsets = [0, 1] if args.offset == "auto" else [int(args.offset)]
    results = {}
    for off in offsets:
        rows = []
        for i, layer in enumerate(layers):
            if layer + off >= a.n_states or (not b_taps_only and layer + off >= b.n_states):
                rows.append((layer, None)); continue
            worst = (1.0, 0.0, 0.0)
            for t in range(toks):
                c, m, r = stats(a.state(t, layer + off), b_state(t, i, layer, 1 if b_taps_only else off))
                worst = (min(worst[0], c), max(worst[1], m), max(worst[2], r))
            rows.append((layer, worst))
        results[off] = rows
    best = max(results, key=lambda o: min((w[0] for _, w in results[o] if w), default=-1.0))
    for off in offsets:
        print("offset %d (tap layer L -> A state L+%d)%s" % (off, off, "  <== best match" if off == best and len(offsets) > 1 else ""))
        for layer, w in results[off]:
            if w is None:
                print("  layer %2d: out of range" % layer)
            else:
                print("  layer %2d: worst cos %.6f  max_abs %.4g  rel %.4g" % (layer, w[0], w[1], w[2]))
    worst_cos = min((w[0] for _, w in results[best] if w), default=0.0)
    ok = worst_cos >= args.min_cos
    print("TAPS: %s (offset %d, worst cos %.6f, gate %.4f)" % ("MATCH" if ok else "MISMATCH", best, worst_cos, args.min_cos))
    return 0 if ok else 1


def cmd_fc(args):
    src = TapSource(args.taps)
    layers = [int(x) for x in args.layers.split(",")]
    wshape, w = read_npy(args.fc_weight)
    dim = src.dim
    if wshape != (dim, dim * len(layers)):
        raise SystemExit("fc weight shape %s, expected (%d, %d)" % (wshape, dim, dim * len(layers)))
    norm_w = read_npy(args.norm_weight)[1] if args.norm_weight else None
    xshape, x = read_npy(args.drafter_input)
    if len(xshape) != 2 or xshape[1] != dim:
        raise SystemExit("drafter input must be [tokens, D]")
    rng = random.Random(0)
    rows = sorted(rng.sample(range(dim), min(args.rows, dim)))
    taps_only = src.n_states == len(layers)
    k_in = dim * len(layers)

    def fc_rows(v):
        return [sum(w[r * k_in + k] * v[k] for k in range(k_in)) for r in rows]

    scores = {"norm(fc(cat))": [], "fc(cat(norm(tap)))": [], "fc(norm(cat))": []}
    toks = min(src.n_tok, xshape[0], args.tokens)
    for t in range(toks):
        taps = [src.state(t, i if taps_only else layer + args.offset) for i, layer in enumerate(layers)]
        cat = [v for tp in taps for v in tp]
        target = [x[t * dim + r] for r in rows]
        full = [sum(w[r * k_in + k] * cat[k] for k in range(k_in)) for r in range(dim)] if args.exact else None
        if full is not None:
            n = rmsnorm(full, norm_w, args.eps)
            cand1 = [n[r] for r in rows]
        else:
            # Without the full fc output the RMS of norm(fc(cat)) is unknown: compare up to scale (cosine only).
            cand1 = [v * (norm_w[r] if norm_w else 1.0) for v, r in zip(fc_rows(cat), rows)]
        cand2 = fc_rows([v for tp in taps for v in rmsnorm(tp, norm_w, args.eps)])
        nw_cat = (norm_w * len(layers)) if norm_w else None
        cand3 = fc_rows(rmsnorm(cat, nw_cat, args.eps))
        for name, c in zip(scores, (cand1, cand2, cand3)):
            scores[name].append(stats(c, target)[0])
    for name, cs in scores.items():
        print("  %-20s worst cos %.6f over %d tokens x %d rows" % (name, min(cs), len(cs), len(rows)))
    best = max(scores, key=lambda k: min(scores[k]))
    print("FC: best order %s (worst cos %.6f)" % (best, min(scores[best])))
    return 0 if min(scores[best]) >= args.min_cos else 1


def expected_mask(n_ctx, block, window, kind, convention):
    """[block, n_ctx + block] booleans. Block queries sit at positions n_ctx .. n_ctx+block-1."""
    m = []
    for i in range(block):
        q = n_ctx + i
        row = []
        for j in range(n_ctx + block):
            if j >= n_ctx:
                row.append(True)  # bidirectional inside the draft block
            elif kind == "full":
                row.append(True)
            else:
                row.append((q - j) < window if convention == "lt" else (q - j) <= window)
        m.append(row)
    return m


def cmd_mask(args):
    shape, data = read_npy(args.dump)
    if len(shape) != 2:
        raise SystemExit("mask must be 2-D [block, ctx+block]")
    block, cols = shape
    n_ctx = cols - block
    if args.float_mask:
        got = [v > -1e30 and not math.isinf(v) for v in data]
    else:
        got = [bool(v) for v in data]
    results = {}
    for conv in ("lt", "le"):
        exp = expected_mask(n_ctx, block, args.window, args.kind, conv)
        mism, cats = 0, {"window": 0, "block": 0, "other": 0}
        first = None
        for i in range(block):
            for j in range(cols):
                e, g = exp[i][j], got[i * cols + j]
                if e != g:
                    mism += 1
                    q = n_ctx + i
                    cat = "block" if j >= n_ctx else ("window" if abs((q - j) - args.window) <= 1 else "other")
                    cats[cat] += 1
                    if first is None:
                        first = (i, j, e, g)
        print("  convention q-k %s W: %d mismatches %s%s" % ("<" if conv == "lt" else "<=", mism, cats,
              "" if first is None else "  first at query %d key %d (expected %s, got %s)" % first))
        results[conv] = mism
    sym = {"lt": "<", "le": "<="}
    ok = results[args.convention] == 0
    other = "le" if args.convention == "lt" else "lt"
    hint = ""
    if not ok and results[other] == 0:
        hint = "; the dump uses q-k %s W (window off by one)" % sym[other]
    print("MASK: %s (kind=%s, W=%d, block=%d, ctx=%d, expected q-k %s W%s)" % (
        "MATCH" if ok else "MISMATCH", args.kind, args.window, block, n_ctx, sym[args.convention], hint))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("taps", help="compare residual taps of two dumps")
    t.add_argument("a"); t.add_argument("b")
    t.add_argument("--layers", default=",".join(map(str, DEFAULT_TAPS)))
    t.add_argument("--offset", default="auto", help="state = layer + offset in A (auto tries 0 and 1)")
    t.add_argument("--tokens", type=int, default=0, help="compare at most this many tokens (0 = all)")
    t.add_argument("--min-cos", type=float, default=0.999)
    f = sub.add_parser("fc", help="find the hidden_norm / fc order that reproduces a drafter input")
    f.add_argument("taps"); f.add_argument("--fc-weight", required=True); f.add_argument("--norm-weight")
    f.add_argument("--drafter-input", required=True)
    f.add_argument("--layers", default=",".join(map(str, DEFAULT_TAPS)))
    f.add_argument("--offset", type=int, default=1)
    f.add_argument("--rows", type=int, default=64); f.add_argument("--tokens", type=int, default=2)
    f.add_argument("--eps", type=float, default=1e-6)
    f.add_argument("--exact", action="store_true", help="compute every fc row (slow at D=3072 in pure Python)")
    f.add_argument("--min-cos", type=float, default=0.999)
    m = sub.add_parser("mask", help="diff a draft-block attention mask")
    m.add_argument("dump"); m.add_argument("--window", type=int, default=4096); m.add_argument("--kind", choices=("swa", "full"), default="swa")
    m.add_argument("--float-mask", action="store_true", help="mask stores 0 / -inf instead of booleans")
    m.add_argument("--convention", choices=("lt", "le"), default="lt",
                   help="expected window test: lt = key visible iff q-k < W (llama.cpp standard SWA), le = q-k <= W")
    args = ap.parse_args(argv)
    return {"taps": cmd_taps, "fc": cmd_fc, "mask": cmd_mask}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
