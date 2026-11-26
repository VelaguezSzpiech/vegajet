# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Bounded CPU end-to-end GGUF quality fixture, no model or runtime needed."""
import math
import pathlib
import re
import struct
import subprocess
import sys
import tempfile


def string(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b


with tempfile.TemporaryDirectory() as tmp:
    path = pathlib.Path(tmp) / ('trunk.' + 'gg' + 'uf')
    # Two valid Q4_K tensors; decoder must consume real GGUF block layout.
    payload = b''.join(struct.pack('<ee', .01 + r * .0001, .005) + bytes([17] * 12)
                       + bytes((i * 37 + r * 11 + (i * r) % 71) % 256 for i in range(128))
                       for r in range(64))
    header = b'GGUF' + struct.pack('<IQQ', 3, 2, 0)
    for i in range(2):
        header += string(f'blk.{i}.attn_q.weight') + struct.pack('<IQQIQ', 2, 256, 64, 12, i * len(payload))
    header += bytes((-len(header)) % 32)
    path.write_bytes(header + payload * 2)
    out = subprocess.check_output([sys.argv[1], str(path), '--max-tensors', '1'], text=True, timeout=60)
    lines = [line for line in out.splitlines() if 'rel_frob=' in line]
    assert len(lines) == 11, out
    assert all(line.startswith('blk.0.') for line in lines), out
    kl_values = []
    errors = {}
    for line in lines:
        values = re.search(r'rel_frob=(\S+) KL_proxy=(\S+)', line)
        assert values and all(math.isfinite(float(v)) and float(v) >= 0 for v in values.groups()), line
        kl_values.append(float(values.group(2)))
        errors[line.split()[2]] = float(values.group(1))
    assert max(kl_values) > 0, out
    for bits in (2, 3):
        for rank in (16, 32):
            assert errors[f'Q{bits}+MiLo{rank}/i3'] < errors[f'Q{bits}'], out
    assert 'Q3+MiLo16/i3' in out and 'Q2+MiLo32/f16' in out, out
    bad = subprocess.run([sys.argv[1], str(path), '--max-tensors', '0'], capture_output=True)
    assert bad.returncode != 0
    print(out)
