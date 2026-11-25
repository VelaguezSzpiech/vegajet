# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Unit tests for tools/vj_tp3_preflight.py: synthetic GGUF headers (no weights), the exact split and DOT8 layout
arithmetic (sizes cross-checked against core/q4_dot8_repack.cpp), the TP=3 plan, and swarm manifests validated with
the repository's own tools/swarm_router.py. Standard library only; nothing outside a temporary directory is
written and no device or /sys path is touched."""
import contextlib
import io
import itertools
import json
import math
import os
import struct
import sys
import tempfile
import unittest
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.dont_write_bytecode = True

import vj_tp3_preflight as tp  # noqa: E402

ROUTER = os.path.join(ROOT, "tools", "swarm_router.py")
OPS_DOC = os.path.join(ROOT, "release", "docs", "OPERATIONS.md")
OPS_SNIPPET = """### 4.3 3x swarm
   - start three workers: `vegajet serve <model-file> --gpu 0 --port <p1> --max-queue 1`, the same with `--gpu 1` and
     `--gpu 2`;
     ```sh
     python3 tools/swarm_router.py --listen 127.0.0.1:8090 --worker http://127.0.0.1:<p1> \\
         --worker http://127.0.0.1:<p2> --worker http://127.0.0.1:<p3>
     ```
"""

# GGUF writer (header only: metadata + tensor table, no tensor data).
T_U32, T_STR, T_ARR = 4, 8, 9


def _s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, kv, tensors):
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for k, v in kv:
        out += _s(k)
        if isinstance(v, str):
            out += struct.pack("<I", T_STR) + _s(v)
        elif isinstance(v, list):
            out += struct.pack("<IIQ", T_ARR, T_U32, len(v)) + b"".join(struct.pack("<I", x) for x in v)
        else:
            out += struct.pack("<II", T_U32, v)
    for name, ne, tt in tensors:
        out += _s(name) + struct.pack("<I", len(ne)) + b"".join(struct.pack("<Q", x) for x in ne)
        out += struct.pack("<IQ", tt, 0)
    with open(path, "wb") as fh:
        fh.write(out)


Q8_0, F32, IQ2_S, IQ3_XXS, IQ4_NL = 8, 0, 22, 18, 20
D, FF, E, TOPK, VOCAB = 512, 1024, 12, 4, 1024


def mini_ornith(path, down_type=IQ3_XXS):
    """2 main layers + 1 nextn layer, qwen35moe naming; layer 1 is a full-attention layer."""
    kv = [("general.architecture", "qwen35moe"), ("qwen35moe.block_count", 3), ("qwen35moe.nextn_predict_layers", 1),
          ("qwen35moe.embedding_length", D), ("qwen35moe.expert_count", E), ("qwen35moe.expert_used_count", TOPK),
          ("qwen35moe.expert_feed_forward_length", FF), ("qwen35moe.attention.head_count", 4),
          ("qwen35moe.attention.head_count_kv", 2), ("qwen35moe.attention.key_length", 64),
          ("tokenizer.ggml.tokens_count_dummy", [1, 2, 3])]
    t = [("token_embd.weight", [D, VOCAB], Q8_0), ("output.weight", [D, VOCAB], Q8_0),
         ("output_norm.weight", [D], F32)]
    for il in range(2):
        b = "blk.%d." % il
        t += [(b + "attn_norm.weight", [D], F32), (b + "ffn_gate_inp.weight", [D, E], F32),
              (b + "ffn_gate_exps.weight", [D, FF, E], IQ2_S), (b + "ffn_up_exps.weight", [D, FF, E], IQ2_S),
              (b + "ffn_down_exps.weight", [FF, D, E], down_type), (b + "ffn_up_shexp.weight", [D, 256], Q8_0)]
        t += [(b + "attn_q.weight", [D, 256], Q8_0), (b + "attn_k.weight", [D, 128], Q8_0)] if il == 1 else \
             [(b + "ssm_out.weight", [256, D], Q8_0)]
    t += [("blk.2.nextn.eh_proj.weight", [2 * D, D], Q8_0), ("per_layer_token_embd.weight", [256, VOCAB], Q8_0)]
    write_gguf(path, kv, t)
    return t


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = tp.main(argv)
    return rc, out.getvalue(), err.getvalue()


class Arithmetic(unittest.TestCase):
    def test_apportion(self):
        self.assertEqual(tp.apportion(32, [1, 1, 1]), [11, 11, 10])
        self.assertEqual(tp.apportion(32, tp.REF_SHARES), [10, 11, 11])
        self.assertEqual(tp.apportion(4, [1, 1, 1]), [2, 1, 1])
        self.assertEqual(tp.apportion(256, [1, 1, 1]), [86, 85, 85])
        self.assertEqual(tp.apportion(256, tp.REF_SHARES), [81, 88, 87])
        self.assertEqual(sum(tp.apportion(97, [0.2, 0.5, 0.3])), 97)

    def test_dot8_layout_matches_engine(self):
        # (rows, cols, kind) -> bytes, printed by core/q4_dot8_repack.cpp (dot8_*_layout(...).vram_bytes()).
        cases = [(131072, 3072, "gu_iq2s", 436207616), (1, 512, "gu_iq2s", 768), (90112, 3072, "gu_iq2s", 299892736),
                 (1, 512, "iq3", 272), (1, 1024, "iq3", 768), (786432, 512, "iq3", 213909504), (1, 256, "iq3", 256),
                 (7, 2560, "iq3", 9520), (5, 2560, "gu_iq2s", 14080)]
        for rows, cols, kind, want in cases:
            self.assertEqual(tp.dot8_layout(rows, cols, kind), want, (rows, cols, kind))
        # VJ_Q3_IQ2S (affebd8 q3_gu_iq2s_layout(...).vram_bytes()).
        for rows, cols, want in [(1, 3072, 2560), (131072, 3072, 335544320), (90112, 3072, 230686720),
                                 (327680, 2560, 838860800)]:
            self.assertEqual(tp.dot8_layout(rows, cols, "q3_gu_iq2s"), want, (rows, cols))
        self.assertIsNone(tp.dot8_layout(1, 352, "iq3"))  # columns must be whole 256-wide superblocks
        self.assertEqual([tp.odd256(x) for x in (0, 1, 256, 257, 512, 768, 769)], [256, 256, 256, 768, 768, 768, 1280])

    def test_expected_max_is_exact(self):
        sizes, k = (3, 2, 2), 3
        groups = [g for g, s in enumerate(sizes) for _ in range(s)]
        combos = list(itertools.combinations(range(sum(sizes)), k))
        brute = Fraction(sum(max(sum(1 for i in c if groups[i] == g) for g in range(3)) for c in combos), len(combos))
        self.assertEqual(tp.multivariate_max_expectation(sizes, k), brute)
        self.assertEqual(tp.multivariate_max_expectation((86, 85, 85), 1), 1)
        self.assertAlmostEqual(float(tp.multivariate_max_expectation((86, 85, 85), 8)), 4.0521, places=3)

    def test_ggml_table_matches_header(self):
        hdr = os.path.join(ROOT, "core", "ggml_types.hpp")
        if not os.path.isfile(hdr):
            self.skipTest("core/ggml_types.hpp not in this tree")
        import re
        rows = re.findall(r'case (\d+):\s*return \{"(\w+)", (\d+), (\d+)\};', read(hdr))
        self.assertGreater(len(rows), 30)
        for tid, name, be, bb in rows:
            self.assertEqual(tp.GGML[int(tid)], (name, int(be), int(bb)), name)


class ProfileAndPlan(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="vj_tp3_test_")
        self.path = os.path.join(self.tmp.name, "mini")
        self.tensors = mini_ornith(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_profile(self):
        p = tp.build_profile([self.path])
        d = p["dims"]
        self.assertEqual((d["arch"], d["n_layer"], d["nextn_layers"], d["d_model"], d["n_expert"], d["n_expert_used"],
                          d["expert_ff"], d["vocab"], d["full_attention_layers"]),
                         ("qwen35moe", 2, 1, D, E, TOPK, FF, VOCAB, 1))
        self.assertEqual(p["routed_gu_types"], {"IQ2_S": 4})
        self.assertEqual(p["routed_down_types"], {"IQ3_XXS": 2})
        gu = 4 * (D * FF * E // 256 * 82)
        self.assertEqual(p["roles"]["routed_gu"]["bytes"], gu)
        self.assertEqual(p["roles"]["routed_down"]["bytes"], 2 * (D * FF * E // 256 * 98))
        self.assertEqual(p["roles"]["mtp"]["tensors"], 1)
        self.assertEqual(p["roles"]["ple_table"]["bytes"], 256 * VOCAB // 32 * 34)
        self.assertEqual(p["file_bytes"], sum(tp.tensor_bytes(ne, tt) for _, ne, tt in self.tensors))

    def test_plan_split_ff_and_ep(self):
        p = tp.plan(tp.build_profile([self.path]), list(tp.REF_SHARES), 32.0, 4096, 2.0)
        v = p["variants"]
        self.assertEqual(v["split_ff_balanced"]["local_ff"], [512, 256, 256])  # IQ3_XXS Down: 256-column unit
        self.assertEqual(v["split_ff_balanced"]["m1_moe_vs_tp2"], 1.0)        # the 512 GPU streams what TP=2 does
        self.assertEqual(v["split_ff_balanced_down_requant"]["local_ff"], [352, 352, 320])
        self.assertEqual(v["split_ff_asymmetric_down_requant"]["local_ff"], [320, 352, 352])
        self.assertAlmostEqual(v["split_ff_balanced_down_requant"]["m1_moe_vs_tp2"], 352 / 512)
        self.assertEqual(v["ep_balanced"]["experts"], [4, 4, 4])
        gran = [c for c in p["checks"] if c["name"] == "split_ff_granularity"][0]
        self.assertFalse(gran["ok"])
        self.assertIn("splits 512/256/256", gran["detail"])
        self.assertFalse(p["ok"])
        # Per-GPU bytes add up: the GU source planes split exactly across the three GPUs.
        prof = tp.build_profile([self.path])
        L = 2
        for name in ("split_ff_balanced", "ep_asymmetric"):
            dot8 = sum(g["gu_dot8_all_layers_gib"] for g in v[name]["per_gpu"]) * tp.GIB
            self.assertAlmostEqual(dot8, L * tp.dot8_layout(E * FF, D, "gu_iq2s"), delta=1.0)
        self.assertEqual(p["model"]["host_tables_gib"], prof["roles"]["ple_table"]["bytes"] / tp.GIB)
        self.assertTrue(p["calibration_tp2"]["calibrated"])
        # A small model fits and can promote every layer, next to the source planes.
        for name, var in v.items():
            self.assertTrue(var["fits_all"], name)
            self.assertTrue(var["promote_2_of_2_additive"], name)
            self.assertTrue(var["q3_inplace_2_of_2"], name)
            g0 = var["per_gpu"][0]
            rows = (var["experts"][0] * FF) if "experts" in var else E * var["local_ff"][0]
            self.assertAlmostEqual(g0["gu_q3_inplace_all_layers_gib"] * tp.GIB, 2 * tp.dot8_layout(rows, D, "q3_gu_iq2s"),
                                   delta=1.0)
            self.assertFalse(g0["kv_and_workspace_counted"])  # inside the measured qwen35moe overhead

    def test_plan_32col_down_has_no_superblock_constraint(self):
        mini_ornith(self.path, IQ4_NL)
        p = tp.plan(tp.build_profile([self.path]), [1.0, 1.0, 1.0], 32.0, 4096, 2.0)
        self.assertEqual(p["variants"]["split_ff_balanced"]["local_ff"], [352, 352, 320])
        self.assertNotIn("split_ff_balanced_down_requant", p["variants"])
        self.assertTrue([c for c in p["checks"] if c["name"] == "split_ff_granularity"][0]["ok"])
        self.assertTrue(all(g["down_plane"] == "source" for g in p["variants"]["split_ff_balanced"]["per_gpu"]))

    def test_cli(self):
        rc, out, _ = run(["profile", self.path])
        self.assertEqual(rc, 0)
        prof = os.path.join(self.tmp.name, "p.json")
        with open(prof, "w") as fh:
            fh.write(out)
        rc, out, _ = run(["plan", "--profile", prof])
        self.assertEqual(rc, 1)  # the granularity check fails for an IQ3_XXS Down
        self.assertIn("recommended: split_ff_balanced_down_requant", out)
        self.assertIn("calibration: affebd8 measured 27.77 GiB", out)
        rc, out, _ = run(["plan", "--profile", prof, "--json", "--overhead-gib", "0.5"])
        rep = json.loads(out)
        self.assertEqual(rep["model"]["runner_overhead_gib"], 0.5)
        self.assertTrue(rep["model"]["kv_and_workspace_counted"])  # an explicit overhead: KV + workspace on top
        self.assertFalse(rep["calibration_tp2"]["calibrated"])
        bad = os.path.join(self.tmp.name, "bad")
        with open(bad, "wb") as fh:
            fh.write(b"NOPE")
        self.assertEqual(run(["profile", bad])[0], 2)
        self.assertEqual(run(["plan", "--profile", prof, "--shares", "1"])[0], 2)
        self.assertEqual(run(["plan", "--profile", prof, "--shares", "1,-1,1"])[0], 2)
        self.assertEqual(run([])[0], 2)
        dense = os.path.join(self.tmp.name, "dense")
        write_gguf(dense, [("general.architecture", "qwen35"), ("qwen35.block_count", 1),
                           ("qwen35.embedding_length", D)], [("token_embd.weight", [D, VOCAB], Q8_0)])
        rc, _, err = run(["plan", "--profile", dense])
        self.assertEqual(rc, 2)
        self.assertIn("lacks MoE dims", err)

    def test_truncated_header(self):
        with open(self.path, "rb") as fh:
            data = fh.read()
        with open(self.path, "wb") as fh:
            fh.write(data[:40])
        with self.assertRaises(tp.InputError):
            tp.read_gguf_header(self.path)


class Swarm(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="vj_swarm_test_")
        self.ops = os.path.join(self.tmp.name, "OPERATIONS.md")
        with open(self.ops, "w") as fh:
            fh.write(OPS_SNIPPET)

    def tearDown(self):
        self.tmp.cleanup()

    def validate(self, m, **kw):
        return tp.validate_swarm(m, ROUTER, self.ops, **kw)

    def failed(self, r):
        return sorted(c["name"] for c in r["checks"] if not c["ok"])

    def test_generated_manifest_passes(self):
        m = tp.swarm_manifest()
        self.assertEqual(m["commands"]["workers"][1], "vegajet serve <model-file> --gpu 1 --port 8082 --max-queue 1")
        self.assertEqual(m["commands"]["router"],
                         "python3 tools/swarm_router.py --listen 127.0.0.1:8090 --worker http://127.0.0.1:8081 "
                         "--worker http://127.0.0.1:8082 --worker http://127.0.0.1:8083 --max-queue 64")
        r = self.validate(m)
        self.assertTrue(r["ok"], r["checks"])
        self.assertIn("router_accepts_manifest", [c["name"] for c in r["checks"]])

    @unittest.skipUnless(os.path.isfile(OPS_DOC), "release/docs/OPERATIONS.md comes with the Tracks 7-8 hand-off")
    def test_against_the_real_operations_doc(self):
        r = tp.validate_swarm(tp.swarm_manifest(), ROUTER, OPS_DOC)
        self.assertTrue(r["ok"], r["checks"])

    def test_bad_manifests(self):
        def mut(fn):
            m = tp.swarm_manifest()
            m.pop("commands")
            fn(m)
            return self.failed(self.validate(m))
        self.assertEqual(mut(lambda m: m["workers"][2].update(gpu=1)), ["one_worker_per_gpu"])
        self.assertEqual(mut(lambda m: m["workers"][0].update(port=8090)), ["ports_unique"])
        self.assertEqual(mut(lambda m: m["workers"][0].update(port=80)), ["ports_unique"])
        self.assertEqual(mut(lambda m: m["workers"][1].update(max_queue=2)), ["max_queue_1", "operations_md_forms"])
        self.assertEqual(mut(lambda m: m["router"].update(listen="0.0.0.0:8090")), ["router_loopback"])
        self.assertEqual(mut(lambda m: m.update(power_cap_w=200)), ["power_cap"])
        m = tp.swarm_manifest()
        m["workers"][0]["port"] = 9999  # stale commands block
        self.assertIn("commands_consistent", self.failed(self.validate(m)))

    def test_router_rejects(self):
        bogus = os.path.join(self.tmp.name, "router.py")
        with open(bogus, "w") as fh:
            fh.write("def parse_args(argv):\n    raise SystemExit(2)\n")
        r = tp.validate_swarm(tp.swarm_manifest(), bogus, None)
        self.assertEqual(self.failed(r), ["router_accepts_manifest"])

    def test_operations_doc_mismatch(self):
        with open(self.ops, "w") as fh:
            fh.write("# nothing about the swarm here\n")
        self.assertEqual(self.failed(self.validate(tp.swarm_manifest())), ["operations_md_forms"])

    def test_model_must_fit_one_card(self):
        small = {"file_bytes": 10 * tp.GIB, "roles": {}}
        big = {"file_bytes": 60 * tp.GIB, "roles": {"ple_table": {"bytes": 20 * tp.GIB}}}  # 40 GiB in VRAM
        host_only = {"file_bytes": 40 * tp.GIB, "roles": {"ple_table": {"bytes": 20 * tp.GIB}}}  # 20 GiB in VRAM
        self.assertTrue(self.validate(tp.swarm_manifest(), profile=small)["ok"])
        self.assertEqual(self.failed(self.validate(tp.swarm_manifest(), profile=big)), ["model_fits_one_card"])
        self.assertTrue(self.validate(tp.swarm_manifest(), profile=host_only)["ok"])

    def test_cli(self):
        rc, out, _ = run(["swarm-manifest", "--gpus", "3", "--base-port", "9001"])
        self.assertEqual(rc, 0)
        man = os.path.join(self.tmp.name, "m.json")
        with open(man, "w") as fh:
            fh.write(out)
        rc, out, _ = run(["validate-swarm", man, "--router", ROUTER, "--operations", self.ops])
        self.assertEqual(rc, 0, out)
        self.assertIn("SWARM MANIFEST: PASS", out)
        m = json.loads(read(man))
        m["workers"][0]["gpu"] = 2
        with open(man, "w") as fh:
            json.dump(m, fh)
        self.assertEqual(run(["validate-swarm", man])[0], 1)
        self.assertEqual(run(["validate-swarm", os.path.join(self.tmp.name, "missing.json")])[0], 2)


class NoDeviceAccess(unittest.TestCase):
    def test_source_never_names_device_or_sysfs_paths(self):
        src = read(os.path.join(ROOT, "tools", "vj_tp3_preflight.py"))
        self.assertNotIn("/dev/", src)
        self.assertNotIn("/sys/", src.replace("never reads /sys and", ""))


if __name__ == "__main__":
    unittest.main()
