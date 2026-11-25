#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Offline bring-up planner for a third MI50 (N = 3, 3 x 32 GB, 150 W per card). CPU only, standard library only.
It never opens a device node, never reads /sys and never writes outside the paths it is given.

Subcommands
  profile <model-file> [<shard> ...]       Read the GGUF header(s) (metadata + tensor table, never the weights) and
                                           print a model profile (JSON): dims and exact bytes per tensor role.
  plan --profile <profile.json|model-file> TP=3 plan: exact balanced and bandwidth-asymmetric splits of the routed
       [--shares 0.317,0.341,0.341]      experts (split-FF in row units, whole-expert EP), the expected M=1 MoE
       [--vram-gib 32] [--ctx 32768]     imbalance, per-GPU VRAM (resident + KV + workspace) and the headroom for
       [--reserve-gib 2.0] [--json]      promoting all layers' routed experts to VJ_Q4_DOT8 (additive: next to the
                                           source planes, as the static promotion does; replace: instead of them).
  swarm-manifest [--gpus 3] [--base-port 8081] [--router-port 8090]
                                           Write the 3 x N=1 swarm launch manifest (JSON) with the exact commands of
                                           OPERATIONS.md section 4.3.
  validate-swarm <manifest.json> [--router tools/swarm_router.py] [--operations release/docs/OPERATIONS.md]
                 [--profile P]             Validate a manifest: one worker per GPU 0..N-1, unique loopback ports,
                                           --max-queue 1, worker URLs and router arguments accepted by
                                           swarm_router.py's own parser, command forms documented in
                                           OPERATIONS.md, and (with a profile) that the model fits ONE card.

VRAM model (stated, not measured; see the 'model' block of every plan): the usable arena is 30.5 GiB of a 32 GB
card (what the runtime reserves); routed Down planes in IQ3_XXS are resident as VJ_Q4_DOT8 (the engine repacks
them at load), every other tensor at its GGUF size; KV is fp16 for the full-attention layers over --ctx; the
workspace is 1.5 GiB per GPU (the runner's own headroom constant). DOT8 plane sizes use the engine's exact layout
rules (odd multiple of 256 B per row, scales embedded when they fit). Exit: 0 = all checks pass, 1 = a check
failed, 2 = usage / input error.
"""
import argparse
import importlib.util
import itertools
import json
import math
import os
import re
import struct
import sys
from fractions import Fraction

GIB = 1 << 30
ARENA_GIB = 30.5            # what Runtime::init reserves on a 32 GB card
WORKSPACE_GIB = 1.5         # runner headroom constant (ornith_gpu.cpp DOT8 auto plan)
CAP_W_N3 = 150.0            # per-card cap with 3+ cards (OPERATIONS.md 4.1, core/tp3_plan)
REF_SHARES = (0.317, 0.341, 0.341)
ROW_UNIT = 32               # split-FF row unit (core/sharder kRowUnit)
DOT8_TILE = 256

# ggml type id -> (name, block elems, block bytes); same table as core/ggml_types.hpp
GGML = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
        7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
        12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
        16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
        20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
        24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
        29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66),
        39: ("MXFP4", 32, 17), 40: ("NVFP4", 64, 36), 41: ("Q1_0", 128, 18), 42: ("Q2_0", 64, 18),
        43: ("F8_E4M3", 1, 1)}


class InputError(Exception):
    pass


# ============================================================================================ GGUF header
def _rd(fh, fmt):
    n = struct.calcsize(fmt)
    b = fh.read(n)
    if len(b) != n:
        raise InputError("truncated GGUF header")
    return struct.unpack("<" + fmt, b)


def _str(fh):
    (n,) = _rd(fh, "Q")
    if n > (1 << 24):
        raise InputError("GGUF string too long")
    b = fh.read(n)
    if len(b) != n:
        raise InputError("truncated GGUF header")
    return b.decode("utf-8", "replace")


_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def _value(fh, t, keep=True):
    if t in _SCALAR:
        return _rd(fh, _SCALAR[t])[0]
    if t == 8:
        return _str(fh)
    if t == 9:
        (et,) = _rd(fh, "I")
        (n,) = _rd(fh, "Q")
        if et in _SCALAR and not keep:  # skip big arrays (tokenizer tables) without decoding them
            fh.seek(n * struct.calcsize(_SCALAR[et]), 1)
            return n
        vals = [_value(fh, et, keep) for _ in range(n)]
        return vals if keep else n
    raise InputError("unknown GGUF value type %d" % t)


def read_gguf_header(path):
    """Returns (metadata dict, [(name, [ne...], type)]). Large arrays are counted, not decoded."""
    with open(path, "rb") as fh:
        if fh.read(4) != b"GGUF":
            raise InputError("%s: not a GGUF file" % os.path.basename(path))
        (ver,) = _rd(fh, "I")
        if ver not in (2, 3):
            raise InputError("unsupported GGUF version %d" % ver)
        n_t, n_kv = _rd(fh, "QQ")
        meta = {}
        for _ in range(n_kv):
            k = _str(fh)
            (t,) = _rd(fh, "I")
            keep = not k.startswith("tokenizer.ggml.") or k == "tokenizer.ggml.model"
            meta[k] = _value(fh, t, keep)
        tensors = []
        for _ in range(n_t):
            name = _str(fh)
            (nd,) = _rd(fh, "I")
            ne = list(_rd(fh, "%dQ" % nd)) if nd else []
            (tt,) = _rd(fh, "I")
            _rd(fh, "Q")  # offset
            tensors.append((name, ne, tt))
    return meta, tensors


def tensor_bytes(ne, tt):
    if tt not in GGML:
        raise InputError("unknown ggml type id %d" % tt)
    _, be, bb = GGML[tt]
    n = 1
    for x in ne:
        n *= x
    if n % be:
        raise InputError("tensor size %d not a multiple of block %d" % (n, be))
    return n // be * bb


# ============================================================================================ profile
ROLE_PATTERNS = [
    ("routed_gu", re.compile(r"\.ffn_(gate|up|gate_up)_exps\.")),
    ("routed_down", re.compile(r"\.ffn_down_exps\.")),
    ("router", re.compile(r"\.ffn_gate_inp(_shexp)?\.|\.exp_probs_b")),
    ("shared_expert", re.compile(r"_shexp\.")),
    ("norm", re.compile(r"norm")),
    ("ple_table", re.compile(r"^per_layer_token_embd\.")),   # row-gathered from host memory, not VRAM
    ("embedding", re.compile(r"^token_embd\.")),
    ("lm_head", re.compile(r"^output\.weight")),
]


def classify(name, n_layer):
    m = re.match(r"blk\.(\d+)\.", name)
    if m and n_layer and int(m.group(1)) >= n_layer:
        return "mtp"
    for role, rx in ROLE_PATTERNS:
        if rx.search(name):
            return role
    return "mixer_dense" if name.startswith("blk.") else "other"


def arch_key(meta, arch, *keys):
    for k in keys:
        v = meta.get("%s.%s" % (arch, k))
        if v is not None:
            return v
    return None


def build_profile(paths):
    meta, tensors = {}, []
    for p in paths:
        m, t = read_gguf_header(p)
        for k, v in m.items():
            meta.setdefault(k, v)
        tensors += t
    arch = meta.get("general.architecture", "")
    n_layer = int(arch_key(meta, arch, "block_count") or 0)
    nextn = int(arch_key(meta, arch, "nextn_predict_layers") or 0)
    main_layers = n_layer - nextn if nextn and n_layer > nextn else n_layer
    dims = {
        "arch": arch,
        "n_layer": main_layers,
        "nextn_layers": nextn,
        "d_model": arch_key(meta, arch, "embedding_length"),
        "n_expert": arch_key(meta, arch, "expert_count"),
        "n_expert_used": arch_key(meta, arch, "expert_used_count"),
        "expert_ff": arch_key(meta, arch, "expert_feed_forward_length"),
        "n_head": arch_key(meta, arch, "attention.head_count"),
        "n_head_kv": arch_key(meta, arch, "attention.head_count_kv"),
        "head_dim": arch_key(meta, arch, "attention.key_length"),
        "full_attention_interval": arch_key(meta, arch, "full_attention_interval"),
        "vocab": None,
    }
    for k in ("n_head", "n_head_kv"):  # per-layer arrays: keep the max
        if isinstance(dims[k], list):
            dims[k] = max(dims[k]) if dims[k] else None
    roles, gu_types, down_types, attn_layers = {}, {}, {}, set()
    for name, ne, tt in tensors:
        role = classify(name, main_layers)
        b = tensor_bytes(ne, tt)
        r = roles.setdefault(role, {"bytes": 0, "tensors": 0})
        r["bytes"] += b
        r["tensors"] += 1
        tname = GGML[tt][0]
        if role == "routed_gu":
            gu_types[tname] = gu_types.get(tname, 0) + 1
            if dims["expert_ff"] is None and len(ne) == 3:
                dims["expert_ff"] = ne[1]
        elif role == "routed_down":
            down_types[tname] = down_types.get(tname, 0) + 1
        if name == "token_embd.weight" and len(ne) == 2:
            dims["vocab"] = ne[1]
        m = re.match(r"blk\.(\d+)\.attn_k\.weight", name)
        if m and int(m.group(1)) < main_layers:
            attn_layers.add(int(m.group(1)))
    dims["full_attention_layers"] = len(attn_layers)
    return {"profile": "vj_tp3_preflight.v1", "arch": arch, "dims": dims,
            "file_bytes": sum(r["bytes"] for r in roles.values()), "roles": roles,
            "routed_gu_types": gu_types, "routed_down_types": down_types,
            "n_tensors": len(tensors)}


def load_profile(arg_paths):
    if len(arg_paths) == 1 and arg_paths[0].endswith(".json"):
        with open(arg_paths[0]) as fh:
            p = json.load(fh)
        if p.get("profile") != "vj_tp3_preflight.v1":
            raise InputError("not a vj_tp3_preflight.v1 profile")
        return p
    return build_profile(arg_paths)


# ============================================================================================ splits
def apportion(total, weights):
    """Largest-remainder apportionment of `total` integer units by `weights` (ties -> lower index)."""
    s = sum(Fraction(w).limit_denominator(10 ** 6) for w in weights)
    quotas = [Fraction(w).limit_denominator(10 ** 6) * total / s for w in weights]
    base = [int(math.floor(q)) for q in quotas]
    rest = total - sum(base)
    order = sorted(range(len(weights)), key=lambda i: (-(quotas[i] - base[i]), i))
    for i in order[:rest]:
        base[i] += 1
    return base


def odd256(raw):
    tiles = max(1, -(-raw // DOT8_TILE))
    if tiles % 2 == 0:
        tiles += 1
    return tiles * DOT8_TILE


def dot8_layout(rows, cols, kind):
    """Exact VJ_Q4_DOT8 plane bytes (core/q4_dot8_repack.cpp finish_layout)."""
    if cols % 256:
        return None
    nsb, nblk = cols // 256, cols // 32
    if kind == "q3_gu_iq2s":  # VJ_Q3_IQ2S (affebd8): 3-bit bit-sliced Gate+Up pair, separate odd-256 scale plane
        return rows * (odd256(nblk * 24) + odd256(nblk * 2 + nsb * 4))
    if kind == "gu_iq2s":   # merged gate+up pair
        w_raw, s_raw = nblk * 2 * 16, nblk * 2 + nsb * 4
    else:                   # one 4-bit plane (IQ3_XXS-sourced layout)
        w_raw, s_raw = cols // 2, nsb * 8
    w_stride = odd256(w_raw)
    embedded = odd256(w_raw + s_raw) == w_stride
    return rows * w_stride + (0 if embedded else rows * s_raw)


def multivariate_max_expectation(sizes, k):
    """E[max_i count_i] when k distinct experts are drawn uniformly from groups of the given sizes (exact)."""
    total = sum(sizes)
    denom = math.comb(total, k)
    e = Fraction(0)
    for counts in itertools.product(*[range(min(s, k) + 1) for s in sizes]):
        if sum(counts) != k:
            continue
        ways = 1
        for s, c in zip(sizes, counts):
            ways *= math.comb(s, c)
        e += Fraction(ways * max(counts), denom)
    return e


def split_ff(expert_ff, unit, shares):
    if expert_ff % unit:
        return None
    return [u * unit for u in apportion(expert_ff // unit, shares)]


def plan(profile, shares, vram_gib, ctx, reserve_gib, row_unit=ROW_UNIT, overhead_gib=None):
    d = profile["dims"]
    N = len(shares)
    E, D, ff, topk = d.get("n_expert"), d.get("d_model"), d.get("expert_ff"), d.get("n_expert_used")
    if not all(isinstance(x, int) and x > 0 for x in (E, D, ff, topk)):
        raise InputError("profile lacks MoE dims (n_expert, d_model, expert_ff, n_expert_used)")
    roles = profile["roles"]
    rb = lambda r: roles.get(r, {}).get("bytes", 0)
    down_types = profile.get("routed_down_types", {})
    gu_types = profile.get("routed_gu_types", {})
    down_iq3 = set(down_types) == {"IQ3_XXS"}
    down_block = max([GGML_BLOCK.get(t, 1) for t in down_types] or [1])  # column granularity of the Down split
    L = d["n_layer"]

    out = {"model": {"arena_gib": ARENA_GIB, "workspace_gib": WORKSPACE_GIB, "reserve_gib": reserve_gib,
                     "vram_gib": vram_gib, "ctx": ctx, "row_unit": row_unit,
                     "down_resident": "VJ_Q4_DOT8 (repacked from IQ3_XXS)" if down_iq3 else "GGUF size",
                     "kv": "fp16 K+V, full-attention layers, sharded by KV heads when n_head_kv >= N",
                     "promotion_modes": "additive = VJ_Q4_DOT8 next to the IQ planes (static promotion); replace = "
                                        "DOT8 instead of them; q3_inplace = VJ_Q3_IQ2S instead of IQ2_S (affebd8)"},
           "dims": d, "gpus": N, "shares_target": list(shares), "checks": [], "variants": {}}

    def check(name, ok, detail):
        out["checks"].append({"name": name, "ok": bool(ok), "detail": detail})

    # Reference point: TP=2 split-FF (what runs today): every GPU streams top_k x expert_ff/2 rows per token.
    tp2_rows = topk * ff // 2
    variants = {}
    # 1. split-FF with the Down superblock constraint (GU rows and Down cols share local_ff).
    for label, w in (("balanced", [1.0] * N), ("asymmetric", list(shares))):
        unit = max(row_unit, down_block)
        lf = split_ff(ff, unit, w)
        if lf:
            variants["split_ff_%s" % label] = {"kind": "split-FF", "unit": unit, "local_ff": lf, "target": label,
                                              "realized_share": [x / ff for x in lf],
                                              "m1_rows_max_gpu": topk * max(lf), "tp2_rows_per_gpu": tp2_rows}
        # Same split with Down requantized to 32-column blocks (row unit only).
        lf32 = split_ff(ff, row_unit, w)
        if lf32 and unit != row_unit:
            variants["split_ff_%s_down_requant" % label] = {
                "kind": "split-FF, Down requantized to 32-col blocks", "unit": row_unit, "local_ff": lf32,
                "target": label,
                "realized_share": [x / ff for x in lf32], "m1_rows_max_gpu": topk * max(lf32),
                "tp2_rows_per_gpu": tp2_rows}
    # 2. whole-expert sharding (EP): each GPU holds full expert_ff for its experts.
    for label, w in (("balanced", [1.0] * N), ("asymmetric", list(shares))):
        ex = apportion(E, w)
        emax = multivariate_max_expectation(ex, topk)
        variants["ep_%s" % label] = {"kind": "whole-expert (EP)", "experts": ex, "target": label,
                                    "realized_share": [x / E for x in ex],
                                    "expected_max_experts_m1": float(emax),
                                    "m1_rows_max_gpu": float(emax) * ff, "tp2_rows_per_gpu": tp2_rows}
    for v in variants.values():
        v["m1_moe_vs_tp2"] = v["m1_rows_max_gpu"] / tp2_rows  # < 1: faster than TP=2 (row-bound decode)
        tgt = [x / sum(shares) for x in shares] if v["target"] == "asymmetric" else [1.0 / N] * N
        v["max_share_error"] = max(abs(a - b) for a, b in zip(v["realized_share"], tgt))
    out["variants"] = variants

    # Per-GPU VRAM for each variant.
    fixed = rb("router") + rb("norm") + rb("embedding") + rb("other")        # replicated
    out["model"]["host_tables_gib"] = rb("ple_table") / GIB                  # PLE: host RAM, not counted
    sharded_dense = rb("mixer_dense") + rb("shared_expert") + rb("mtp")       # by share
    lm = rb("lm_head")
    gu_total = rb("routed_gu")
    down_src = rb("routed_down")
    attn_layers = d.get("full_attention_layers") or 0
    nkv, hd = d.get("n_head_kv") or 0, d.get("head_dim") or 0
    kv_total = attn_layers * 2 * nkv * hd * 2 * ctx
    kv_heads = apportion(nkv, [1.0] * N) if nkv >= N else None
    usable = min(ARENA_GIB, vram_gib - 1.5) * GIB
    # Everything beyond the weights. For qwen35moe (the Ornith runner) it is calibrated on a measured point:
    # affebd8 reports 27.77 GiB of the 30.50 GiB arena used per GPU at TP=2 with all 48 Gate+Up layers in VJ_Q3_IQ2S.
    # That measured figure already holds KV (at the bench context), scratch and workspace, so the calibrated model
    # adds no separate KV / workspace terms. A conservative second figure uses the runner's own TP=2 estimate
    # (ornith_gpu.cpp: 26400 MiB base) plus KV(--ctx) and the 1.5 GiB workspace on top.
    lf2 = ff // 2
    down2 = L * dot8_layout(E * D, lf2, "iq3") if (down_iq3 and lf2 % 256 == 0) else down_src / 2
    tp2 = fixed + (sharded_dense + lm) / 2 + gu_total / 2 + down2
    gu_iq2s = "IQ2_S" in gu_types and len(gu_types) == 1
    runner = profile["arch"] == "qwen35moe" and gu_iq2s and overhead_gib is None
    measured_used = 27.77 * GIB
    q3_tp2_extra = (L * dot8_layout(E * lf2, D, "q3_gu_iq2s") - gu_total / 2) if gu_iq2s else 0.0
    runner_const = 26400 * (1 << 20)
    if runner:
        overhead = measured_used - (tp2 + q3_tp2_extra)
        overhead_cons = max(0.0, runner_const - tp2)
    else:
        overhead = (overhead_gib or 0.0) * GIB
        overhead_cons = None
    count_kv_ws = not runner
    out["model"]["runner_overhead_gib"] = overhead / GIB
    out["model"]["kv_and_workspace_counted"] = count_kv_ws
    out["calibration_tp2"] = {"model_weights_gib_per_gpu": tp2 / GIB, "overhead_applied_gib": overhead / GIB,
                              "calibrated": runner,
                              "measured_used_gib": measured_used / GIB if runner else None,
                              "model_at_measured_point_gib": (tp2 + q3_tp2_extra) / GIB if runner else None,
                              "runner_constant_gib": runner_const / GIB if runner else None,
                              "note": "measured: affebd8, TP=2, 48/48 VJ_Q3_IQ2S, 27.77 GiB used per GPU (includes "
                                      "KV, scratch, workspace); conservative: ornith_gpu.cpp's 26400 MiB TP=2 "
                                      "estimate + KV(--ctx) + 1.5 GiB workspace. Other architectures have no GPU "
                                      "runner yet: pass --overhead-gib (KV and workspace are then added)."}
    for name, v in variants.items():
        share = v["realized_share"]
        gpus = []
        for g in range(N):
            if v["kind"].startswith("whole-expert"):
                frac = v["experts"][g] / E
                lf = ff
                rows = v["experts"][g] * ff
            else:
                frac = v["local_ff"][g] / ff
                lf = v["local_ff"][g]
                rows = E * lf
            gu = gu_total * frac
            if "requant" in name:
                down = L * E * D * lf * 18 // 32  # 4.5 bpw, 32-column blocks (IQ4_NL / Q4_0 sizing)
                down_note = "requantized to 32-col IQ4_NL/Q4_0 (re-encode: the parity reference changes)"
            elif down_iq3 and (lf % 256 == 0):
                down = L * dot8_layout((v["experts"][g] if "experts" in v else E) * D, lf, "iq3")
                down_note = "DOT8"
            else:
                down = down_src * frac
                down_note = "source (DOT8 needs local_ff % 256 == 0)" if down_iq3 else "source"
            kvg = kv_total * (kv_heads[g] / nkv if kv_heads else 1.0)
            weights = fixed + sharded_dense * share[g] + lm * share[g] + gu + down
            resident = weights + overhead
            ws = WORKSPACE_GIB * GIB
            # DOT8 promotion of the routed GU planes on every layer.
            if "IQ2_S" in gu_types and len(gu_types) == 1:
                gu_dot8 = L * dot8_layout(rows, D, "gu_iq2s")       # merged gate+up pair, every layer
            else:
                one = dot8_layout(rows, D, "iq3")
                gu_dot8 = None if one is None else L * 2 * one
            free_before = usable - resident - ((kvg + ws) if count_kv_ws else 0.0)
            free_cons = None if overhead_cons is None else usable - weights - overhead_cons - kvg - ws
            # VJ_Q3_IQ2S in place (affebd8, VEGAJET_GU_Q3=1): replaces the IQ2_S Gate+Up planes on every layer.
            q3 = L * dot8_layout(rows, D, "q3_gu_iq2s") if ("IQ2_S" in gu_types and len(gu_types) == 1) else None
            q3_head = None if q3 is None else free_before - (q3 - gu)
            add = None if gu_dot8 is None else free_before - gu_dot8
            rep = None if gu_dot8 is None else free_before - gu_dot8 + gu
            gpus.append({"gpu": g, "resident_gib": resident / GIB, "weights_gib": weights / GIB, "down_plane": down_note,
                         "kv_gib": kvg / GIB, "kv_and_workspace_counted": count_kv_ws,
                         "free_conservative_gib": None if free_cons is None else free_cons / GIB,
                         "headroom_q3_inplace_conservative_gib": None if (free_cons is None or q3 is None)
                         else (free_cons - (q3 - gu)) / GIB,
                         "workspace_gib": WORKSPACE_GIB, "free_before_promotion_gib": free_before / GIB,
                         "gu_dot8_all_layers_gib": None if gu_dot8 is None else gu_dot8 / GIB,
                         "gu_dot8_per_layer_mib": None if gu_dot8 is None else gu_dot8 / L / (1 << 20),
                         "headroom_additive_gib": None if add is None else add / GIB,
                         "headroom_replace_gib": None if rep is None else rep / GIB,
                         "fits": free_before >= reserve_gib * GIB,
                         "promote_all_additive": add is not None and add >= reserve_gib * GIB,
                         "promote_all_replace": rep is not None and rep >= reserve_gib * GIB,
                         "gu_q3_inplace_all_layers_gib": None if q3 is None else q3 / GIB,
                         "headroom_q3_inplace_gib": None if q3_head is None else q3_head / GIB,
                         "q3_inplace_all": q3_head is not None and q3_head >= reserve_gib * GIB})
        v["per_gpu"] = gpus
        v["fits_all"] = all(x["fits"] for x in gpus)
        v["promote_%d_of_%d_additive" % (L, L)] = all(x["promote_all_additive"] for x in gpus)
        v["promote_%d_of_%d_replace" % (L, L)] = all(x["promote_all_replace"] for x in gpus)
        v["q3_inplace_%d_of_%d" % (L, L)] = all(x["q3_inplace_all"] for x in gpus)
        # how many layers fit additively on the tightest GPU
        per = [x["gu_dot8_per_layer_mib"] for x in gpus]
        if None not in per:
            room = min((x["free_before_promotion_gib"] - reserve_gib) * 1024 / p for x, p in zip(gpus, per))
            v["additive_layers_promotable"] = max(0, min(L, int(math.floor(room))))
    # Checks.
    check("power_cap", True, "N=%d: cap every card at %.0f W (OPERATIONS.md 4.1); the plan assumes it" % (N, CAP_W_N3))
    sf = variants.get("split_ff_balanced")
    if sf:
        bad = sf["unit"] > row_unit
        check("split_ff_granularity", not (bad and max(sf["realized_share"]) - 1.0 / N > 0.03),
              "routed Down is %s with %d-column blocks: GU rows and Down columns share local_ff, so the split-FF "
              "unit is %d and expert_ff %d splits %s (max share %.1f %%)" % (
                  "/".join(sorted(down_types)) or "?", down_block, sf["unit"], ff,
                  "/".join(map(str, sf["local_ff"])), 100 * max(sf["realized_share"])))
    best = min(variants.items(), key=lambda kv: (not kv[1]["fits_all"], kv[1]["m1_moe_vs_tp2"]))
    check("fits_tp3", best[1]["fits_all"], "best fitting variant: %s (M=1 routed rows on the slowest GPU = %.3f x TP=2)"
          % (best[0], best[1]["m1_moe_vs_tp2"]))
    out["recommended"] = best[0]
    out["ok"] = all(c["ok"] for c in out["checks"])
    return out


GGML_BLOCK = {name: be for (name, be, _) in GGML.values()}


def render_plan(p):
    d = p["dims"]
    lines = ["model: %s  layers=%s  d_model=%s  experts=%s top-%s  expert_ff=%s  vocab=%s" % (
        p["dims"]["arch"], d["n_layer"], d["d_model"], d["n_expert"], d["n_expert_used"], d["expert_ff"], d["vocab"]),
        "target shares: %s   arena %.1f GiB/GPU, workspace %.1f, reserve %.1f GiB, ctx %d" % (
            "/".join("%.3f" % s for s in p["shares_target"]), ARENA_GIB, WORKSPACE_GIB, p["model"]["reserve_gib"],
            p["model"]["ctx"])]
    for name, v in p["variants"].items():
        split = ("local_ff " + "/".join(map(str, v["local_ff"]))) if "local_ff" in v else (
            "experts " + "/".join(map(str, v["experts"])) + " (E[max active] %.3f)" % v["expected_max_experts_m1"])
        L = d["n_layer"]
        q3 = v["per_gpu"][0]["headroom_q3_inplace_gib"] is not None
        lines.append("%-34s %-34s M=1 MoE rows vs TP=2: %.3f  fits: %s  DOT8 %d/%d additive: %s replace: %s  Q3 "
                     "in-place: %s" % (
            name, split, v["m1_moe_vs_tp2"], "yes" if v["fits_all"] else "NO", L, L,
            "yes" if v.get("promote_%d_of_%d_additive" % (L, L)) else "no (%s layers)" % v.get(
                "additive_layers_promotable", "-"),
            "yes" if v.get("promote_%d_of_%d_replace" % (L, L)) else "no",
            ("yes" if v.get("q3_inplace_%d_of_%d" % (L, L)) else "no") if q3 else "n/a"))
        for g in v["per_gpu"]:
            lines.append("    gpu%d resident %.2f GiB (Down %s)" % (g["gpu"], g["resident_gib"], g["down_plane"]) +
                         (" + KV %.2f + ws %.1f" % (g["kv_gib"], g["workspace_gib"]) if g["kv_and_workspace_counted"]
                          else " (KV/workspace inside the measured overhead)") +
                         " -> free %.2f GiB%s; GU DOT8 %s GiB (%s MiB/layer); headroom additive %s / replace %s GiB" % (
                             g["free_before_promotion_gib"],
                             "" if g["free_conservative_gib"] is None else " (conservative %.2f)" % g["free_conservative_gib"],
                             _f(g["gu_dot8_all_layers_gib"]), _f(g["gu_dot8_per_layer_mib"], 1),
                             _f(g["headroom_additive_gib"]), _f(g["headroom_replace_gib"])) + (
                             "; Q3 in-place %s GiB, headroom %s GiB (conservative %s)" % (
                                 _f(g["gu_q3_inplace_all_layers_gib"]), _f(g["headroom_q3_inplace_gib"]),
                                 _f(g["headroom_q3_inplace_conservative_gib"]))
                             if g["headroom_q3_inplace_gib"] is not None else ""))
    for c in p["checks"]:
        lines.append("%-5s %s: %s" % ("PASS" if c["ok"] else "FAIL", c["name"], c["detail"]))
    c = p.get("calibration_tp2")
    if c:
        if c["calibrated"]:
            lines.append("calibration: affebd8 measured %.2f GiB used/GPU at TP=2 with 48/48 Q3; this model gives %.2f "
                         "GiB weights there -> %.2f GiB non-weight overhead per GPU (conservative figures use the "
                         "runner's %.2f GiB TP=2 constant + KV + workspace)" % (
                             c["measured_used_gib"], c["model_at_measured_point_gib"], c["overhead_applied_gib"],
                             c["runner_constant_gib"]))
        else:
            lines.append("calibration: none for %s (no GPU runner yet); runner overhead %.2f GiB/GPU (--overhead-gib)"
                         % (p["dims"]["arch"], c["overhead_applied_gib"]))
    if p["model"].get("host_tables_gib"):
        lines.append("host-resident tables (PLE, not in VRAM): %.2f GiB" % p["model"]["host_tables_gib"])
    lines.append("recommended: %s" % p["recommended"])
    return "\n".join(lines)


def _f(x, nd=2):
    return "-" if x is None else ("%." + str(nd) + "f") % x


# ============================================================================================ swarm
def swarm_manifest(gpus=3, base_port=8081, router_port=8090, model="<model-file>"):
    workers = [{"gpu": g, "port": base_port + g, "max_queue": 1} for g in range(gpus)]
    m = {"manifest": "vj_swarm.v1", "model": model, "power_cap_w": CAP_W_N3 if gpus >= 3 else 200.0,
         "router": {"listen": "127.0.0.1:%d" % router_port, "max_queue": 64}, "workers": workers}
    m["commands"] = manifest_commands(m)
    return m


def manifest_commands(m):
    cmds = ["vegajet serve %s --gpu %d --port %d --max-queue %d" % (m["model"], w["gpu"], w["port"], w["max_queue"])
            for w in m["workers"]]
    router = ["python3", "tools/swarm_router.py", "--listen", m["router"]["listen"]]
    for w in m["workers"]:
        router += ["--worker", "http://127.0.0.1:%d" % w["port"]]
    if m["router"].get("max_queue") is not None:
        router += ["--max-queue", str(m["router"]["max_queue"])]
    return {"workers": cmds, "router": " ".join(router), "router_argv": router[2:]}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise InputError("cannot load %s" % os.path.basename(path))
    mod = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(mod)
    return mod


def validate_swarm(m, router_path=None, operations_path=None, profile=None, vram_gib=32.0, reserve_gib=2.0):
    checks = []

    def check(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    ws = m.get("workers", [])
    n = len(ws)
    gpus = [w.get("gpu") for w in ws]
    check("worker_count", n >= 1, "%d workers" % n)
    check("one_worker_per_gpu", sorted(gpus) == list(range(n)),
          "GPU indices %s (expected each of 0..%d exactly once: positions in `vegajet serve --gpu list`)" % (gpus, n - 1))
    ports = [w.get("port") for w in ws]
    listen = m.get("router", {}).get("listen", "")
    rport = int(listen.rsplit(":", 1)[1]) if ":" in listen and listen.rsplit(":", 1)[1].isdigit() else None
    check("ports_unique", len(set(ports)) == n and rport not in ports and
          all(isinstance(p, int) and 1024 <= p <= 65535 for p in ports),
          "worker ports %s, router port %s" % (ports, rport))
    check("router_loopback", listen.startswith("127.0.0.1:"), "router listens on %s (keep it on loopback)" % listen)
    check("max_queue_1", all(w.get("max_queue") == 1 for w in ws), "every worker --max-queue 1 (one sequence per GPU)")
    if n >= 3:
        check("power_cap", float(m.get("power_cap_w", 0)) <= CAP_W_N3,
              "power cap %.0f W per card (<= %.0f W with 3 cards)" % (float(m.get("power_cap_w", 0)), CAP_W_N3))
    cmds = manifest_commands(m)
    if m.get("commands") and m["commands"] != cmds:
        check("commands_consistent", False, "the manifest's commands do not match its workers/router fields")
    if router_path:
        try:
            sr = load_module(router_path, "swarm_router_under_test")
            args = sr.parse_args(cmds["router_argv"])
            urls = list(args.worker)
            pool = sr.Pool(urls, args.max_queue, args.affinity_wait)
            check("router_accepts_manifest", len(pool.workers) == n and
                  [w.port for w in pool.workers] == ports and all(w.host == "127.0.0.1" for w in pool.workers),
                  "swarm_router.parse_args + Pool accepted %d workers (%s)" % (len(pool.workers), args.listen))
        except SystemExit:
            check("router_accepts_manifest", False, "swarm_router.parse_args rejected the router arguments")
        except (ValueError, AttributeError, InputError) as exc:
            check("router_accepts_manifest", False, "swarm_router rejected the manifest: %s" % exc)
    if operations_path:
        with open(operations_path, encoding="utf-8") as fh:
            doc = fh.read()
        doc_flat = re.sub(r"\s*\\\n\s*", " ", doc)
        worker_form = re.search(r"vegajet serve <model-file> --gpu 0 --port <p1> --max-queue 1", doc_flat)
        router_form = re.search(r"python3 tools/swarm_router\.py --listen 127\.0\.0\.1:8090 --worker "
                                r"http://127\.0\.0\.1:<p1>", doc_flat)
        ok = bool(worker_form and router_form)
        same = all(re.fullmatch(r"vegajet serve \S+ --gpu \d+ --port \d+ --max-queue 1", c) for c in cmds["workers"])
        check("operations_md_forms", ok and same,
              "worker and router command forms %s OPERATIONS.md section 4.3" % ("match" if ok and same else
                                                                                "do NOT match"))
    if profile is not None:
        total = profile["file_bytes"] - profile["roles"].get("ple_table", {}).get("bytes", 0)
        usable = min(ARENA_GIB, vram_gib - 1.5) * GIB - WORKSPACE_GIB * GIB - reserve_gib * GIB
        check("model_fits_one_card", total <= usable,
              "N=1 needs the whole model on one card: %.2f GiB of weights vs %.2f GiB usable (arena %.1f - "
              "workspace %.1f - reserve %.1f)%s" % (total / GIB, usable / GIB, ARENA_GIB, WORKSPACE_GIB, reserve_gib,
                                                    "" if total <= usable else "; use TP=3 for this model"))
    return {"ok": all(c["ok"] for c in checks), "checks": checks, "commands": cmds}


# ============================================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    a_prof = sub.add_parser("profile")
    a_prof.add_argument("files", nargs="+")
    a_plan = sub.add_parser("plan")
    a_plan.add_argument("--profile", nargs="+", required=True)
    a_plan.add_argument("--shares", default=",".join(map(str, REF_SHARES)))
    a_plan.add_argument("--vram-gib", type=float, default=32.0)
    a_plan.add_argument("--ctx", type=int, default=32768)
    a_plan.add_argument("--reserve-gib", type=float, default=2.0)
    a_plan.add_argument("--row-unit", type=int, default=ROW_UNIT)
    a_plan.add_argument("--overhead-gib", type=float, help="runner buffers per GPU (default: calibrated, qwen35moe)")
    a_plan.add_argument("--json", action="store_true")
    a_man = sub.add_parser("swarm-manifest")
    a_man.add_argument("--gpus", type=int, default=3)
    a_man.add_argument("--base-port", type=int, default=8081)
    a_man.add_argument("--router-port", type=int, default=8090)
    a_val = sub.add_parser("validate-swarm")
    a_val.add_argument("manifest")
    a_val.add_argument("--router")
    a_val.add_argument("--operations")
    a_val.add_argument("--profile", nargs="+")
    a_val.add_argument("--vram-gib", type=float, default=32.0)
    a_val.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "profile":
            print(json.dumps(build_profile(a.files), indent=1))
            return 0
        if a.cmd == "plan":
            shares = [float(x) for x in a.shares.split(",")]
            if len(shares) < 2 or any(s <= 0 for s in shares):
                raise InputError("--shares needs >= 2 positive values")
            p = plan(load_profile(a.profile), shares, a.vram_gib, a.ctx, a.reserve_gib, a.row_unit, a.overhead_gib)
            print(json.dumps(p, indent=1) if a.json else render_plan(p))
            return 0 if p["ok"] else 1
        if a.cmd == "swarm-manifest":
            if a.gpus < 1:
                raise InputError("--gpus must be >= 1")
            print(json.dumps(swarm_manifest(a.gpus, a.base_port, a.router_port), indent=1))
            return 0
        if a.cmd == "validate-swarm":
            with open(a.manifest) as fh:
                m = json.load(fh)
            prof = load_profile(a.profile) if a.profile else None
            r = validate_swarm(m, a.router, a.operations, prof, a.vram_gib)
            if a.json:
                print(json.dumps(r, indent=1))
            else:
                for c in r["checks"]:
                    print("%-5s %s: %s" % ("PASS" if c["ok"] else "FAIL", c["name"], c["detail"]))
                print("SWARM MANIFEST: %s" % ("PASS" if r["ok"] else "FAIL"))
            return 0 if r["ok"] else 1
    except (OSError, ValueError, KeyError, InputError, json.JSONDecodeError) as exc:
        print("vj_tp3_preflight: %s" % exc, file=sys.stderr)
        return 2
    ap.print_usage(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
