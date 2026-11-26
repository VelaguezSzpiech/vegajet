#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Placement CLI contracts using metadata-only model and advisory inventory, no GPU I/O."""
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

BINARY = os.environ.get("VJ_TP_BINARY", "")

def metadata_model(path):
    def string(s):
        b = s.encode()
        return struct.pack("<Q", len(b)) + b
    a = "qwen35moe."
    values = {"block_count": 1, "context_length": 8192, "embedding_length": 3072,
              "attention.head_count": 32, "attention.head_count_kv": 2,
              "attention.key_length": 256, "attention.value_length": 256,
              "rope.dimension_count": 64, "expert_count": 256, "expert_used_count": 8,
              "expert_feed_forward_length": 1024, "expert_shared_feed_forward_length": 1024,
              "ssm.conv_kernel": 4, "ssm.state_size": 128, "ssm.group_count": 16,
              "ssm.time_step_rank": 64, "ssm.inner_size": 8192,
              "full_attention_interval": 1}
    kv = [string("general.architecture") + struct.pack("<I", 8) + string("qwen35moe")]
    kv.extend(string(a + k) + struct.pack("<II", 4, v) for k, v in values.items())
    # This fixture exercises metadata planning only; it is not an executable model.
    shapes = [("token_embd.weight", [3072, 512]), ("output.weight", [3072, 512]),
              ("output_norm.weight", [3072]), ("blk.0.attn_norm.weight", [3072]),
              ("blk.0.post_attention_norm.weight", [3072]),
              ("blk.0.ffn_gate_inp.weight", [3072, 256]), ("blk.0.ffn_gate_inp_shexp.weight", [3072]),
              ("blk.0.ffn_gate_exps.weight", [3072, 1024, 256]),
              ("blk.0.ffn_up_exps.weight", [3072, 1024, 256]),
              ("blk.0.ffn_down_exps.weight", [1024, 3072, 256]),
              ("blk.0.ffn_gate_shexp.weight", [3072, 1024]),
              ("blk.0.ffn_up_shexp.weight", [3072, 1024]),
              ("blk.0.ffn_down_shexp.weight", [1024, 3072]),
              ("blk.0.attn_q.weight", [3072, 16384]), ("blk.0.attn_k.weight", [3072, 512]),
              ("blk.0.attn_v.weight", [3072, 512]), ("blk.0.attn_output.weight", [8192, 3072]),
              ("blk.0.attn_q_norm.weight", [256]), ("blk.0.attn_k_norm.weight", [256]),
              ("blk.0.ssm_a", [1])]
    tensors = []
    for name, dims in shapes:
        elements = 1
        for dim in dims:
            elements *= dim
        tensors.append((name, dims, 1, elements * 2))
    offset = 0
    infos = []
    for name, dims, dtype, size in tensors:
        infos.append(string(name) + struct.pack("<I", len(dims)) +
                     struct.pack("<" + "Q" * len(dims), *dims) + struct.pack("<IQ", dtype, offset))
        offset += (size + 31) // 32 * 32
    header = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(kv)) + b"".join(kv) + b"".join(infos)
    header += bytes((-len(header)) % 32)
    with path.open("wb") as f:
        f.write(header)
        f.truncate(len(header) + offset)

@unittest.skipUnless(BINARY, "requires CMake VJ_TP_BINARY")
class PlacementCLI(unittest.TestCase):
    def test_cli(self):
        with tempfile.TemporaryDirectory(prefix="vj-placement-") as tmp:
            root = Path(tmp)
            model, profile, owner = (root / s for s in ["model.bin", "profile.json", "placement.json"])
            metadata_model(model)
            env = dict(os.environ)
            env.pop("VJ_GPU_DEVICES", None)
            def run(*args, command="bench"):
                return subprocess.run([BINARY, command, str(model), *args], env=env,
                                      text=True, capture_output=True, timeout=20)
            for n in [1, 2, 3, 8]:
                gpus = [{"unique_id": str(r + 1), "usable_vram_bytes": 16 << 30,
                         "bw_hbm_gbs": 1, "mclk": 1, "n_cu": 1,
                         "p2p": [{"kind": "pcie" if r != d else "unknown", "gbs": 1 if r != d else 0}
                                 for d in range(n)]} for r in range(n)]
                profile.write_text(json.dumps({"schema": 1, "gpus": gpus}))
                flags = ["--placement-print", "--ctx", "8192", "--tp-profile", str(profile)]
                p = run(*flags)
                self.assertEqual(p.returncode, 0, p.stderr)
                proposed = json.loads(p.stdout.strip().splitlines()[-1])
                self.assertEqual(len(proposed["tp_groups"][0]["devices"]), n)
                proposed["tp_groups"][0]["devices"].reverse()
                owner.write_text(json.dumps(proposed))
                p = run(*flags, "--placement", str(owner))
                self.assertEqual(p.returncode, 0, p.stderr)
                self.assertEqual(json.loads(p.stdout.strip().splitlines()[-1]), proposed)
                proposed["tp_groups"][0]["devices"][0] = {"unique_id": "missing"}
                owner.write_text(json.dumps(proposed))
                p = run(*flags, "--placement", str(owner))
                self.assertEqual(p.returncode, 2, p.stderr)
                self.assertIn("missing", p.stderr)
            config = {"schema": 1, "tp_groups": [{"id": "target", "devices": [{"unique_id": "1"}]}]}
            for feature, value, needle in [
                    ("pipeline_stages", [{"tp_group": "target", "layer_begin": 0, "layer_end": 1}], "pipeline stages"),
                    ("replicas", [{"id": "one", "tp_group": "target"}], "serving replicas"),
                    ("drafter", {"kind": "MTP", "device": {"unique_id": "2"}}, "dedicated drafter")]:
                owner.write_text(json.dumps(dict(config, **{feature: value})))
                for command in ["bench", "serve"]:
                    p = run("--placement", str(owner), command=command)
                    self.assertEqual(p.returncode, 2, p.stderr)
                    self.assertIn(needle, p.stderr)
            self.assertEqual(run("--placement").returncode, 2)
            owner.write_text('{"schema":1,"tp_groups":[],"unknown":1}')
            self.assertEqual(run("--placement", str(owner)).returncode, 2)

if __name__ == "__main__":
    unittest.main()
