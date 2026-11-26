# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import re
import subprocess
import sys
text = subprocess.check_output([sys.argv[1], '-d', sys.argv[2]], text=True)
for bits in (2, 3):
    for kind in ('gemv', 'gemm', 'gemv_xor', 'gemm_xor'):
        name = f'vj_sub4_q{bits}_{kind}'
        match = re.search(r'<' + name + r'>:\n(.*?)(?=\n[0-9a-f]+ <|\Z)', text, re.S)
        assert match, f'missing kernel {name}'
        body = match[1]
        assert 'v_dot8_i32_i4' in body, name
        if bits == 3:
            assert 'v_and_or_b32' in body, name
            # LLVM also uses AND_OR for activation nibble compaction.
            assert body.count('v_and_or_b32') >= body.count('v_dot8_i32_i4'), name
        assert not re.search(r'buffer_load[^\n]*\blds\b', body), name
        assert 'v_readfirstlane_b32' in body, name
        assert re.search(r'(?:buffer_load_dwordx4[^\n]*\n\s*){4}s_waitcnt vmcnt\(0\)', body), name
        assert 'flat_load' not in body, name
        print(f'{name}: MUBUF DOT8, Q3-only AND_OR, ILP4, uniform bases, no buffer_load LDS: PASS')
