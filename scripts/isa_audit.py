#!/usr/bin/env python3
"""Static gfx906 ISA audit of built .hsaco modules (CPU-only).

For every kernel: VGPR/SGPR (kernel descriptor), LDS and scratch bytes, occupancy (waves/SIMD from VGPR, SGPR, LDS and
the workgroup size), code bytes, instruction count, wide (dwordx4) vs narrow global/buffer loads, LDS ops, and
s_waitcnt / s_barrier counts and densities (per 100 instructions).

Usage: isa_audit.py --objdump LLVM_OBJDUMP [--json OUT.json] HSACO...
"""
import argparse, json, re, struct, subprocess, sys
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument('--objdump', required=True)
ap.add_argument('--json')
ap.add_argument('--wg-size', type=int, default=256, help='workgroup size assumed for occupancy (VegaJet launches 256)')
ap.add_argument('hsaco', nargs='+')
a = ap.parse_args()


def elf_kernels(path):
    data = open(path, 'rb').read()
    shoff = struct.unpack_from('<Q', data, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from('<HHH', data, 0x3A)
    secs = [struct.unpack_from('<IIQQQQIIQQ', data, shoff + i * shentsize) for i in range(shnum)]
    strtab = secs[shstrndx]
    nm = lambda o: data[strtab[4] + o:data.index(b'\0', strtab[4] + o)].decode()
    S = {nm(s[0]): s for s in secs}
    sym = S['.symtab']; ss = secs[sym[6]]
    syms = {}
    for i in range(sym[5] // 24):
        st_name, info, other, shndx, value, size = struct.unpack_from('<IBBHQQ', data, sym[4] + i * 24)
        n = data[ss[4] + st_name:data.index(b'\0', ss[4] + st_name)].decode()
        syms[n] = (shndx, value, size, info & 0xf)
    out = {}
    for n, (shndx, value, size, typ) in syms.items():
        if not n.endswith('.kd'):
            continue
        k = n[:-3]
        sec = secs[shndx]; off = sec[4] + (value - sec[3])
        lds, scratch = struct.unpack_from('<II', data, off)
        r1, r2 = struct.unpack_from('<II', data, off + 48)
        vgpr = ((r1 & 0x3f) + 1) * 4
        sgpr = (((r1 >> 6) & 0xf) + 1) * 8
        code = syms.get(k, (0, 0, 0, 0))[2]
        out[k] = dict(vgpr=vgpr, sgpr=sgpr, lds=lds, scratch=scratch, code=code)
    return out


def disasm(path):
    txt = subprocess.run([a.objdump, '-d', '--no-show-raw-insn', path], capture_output=True, text=True).stdout
    cur, insts = None, {}
    for line in txt.splitlines():
        m = re.match(r'^[0-9a-f]+ <([^>]+)>:', line)
        if m:
            cur = m.group(1); insts[cur] = []; continue
        if cur is None:
            continue
        s = line.strip()
        if not s or s.startswith(';'):
            continue
        op = s.split()[0]
        if re.match(r'^[a-z_0-9]+$', op):
            insts[cur].append(op)
    return insts


rows = []
for path in a.hsaco:
    kd = elf_kernels(path)
    ins = disasm(path)
    for k, r in sorted(kd.items()):
        ops = ins.get(k, [])
        c = Counter(ops)
        n = len(ops)
        wide = sum(v for o, v in c.items() if re.match(r'^(global|buffer|flat)_load_dwordx4', o))
        narrow = sum(v for o, v in c.items() if re.match(r'^(global|buffer|flat)_load_', o)) - wide
        stores = sum(v for o, v in c.items() if re.match(r'^(global|buffer|flat)_store_', o))
        lds_ops = sum(v for o, v in c.items() if o.startswith('ds_'))
        wait = c.get('s_waitcnt', 0); bar = c.get('s_barrier', 0)
        w_v = min(10, 256 // r['vgpr']) if r['vgpr'] else 10
        w_s = min(10, 800 // max(r['sgpr'] + 16, 1))  # gfx9: 800 SGPRs/SIMD incl. VCC/FLAT/XNACK reserve
        waves_per_wg = (a.wg_size + 63) // 64
        wg_per_cu_lds = (65536 // r['lds']) if r['lds'] else 40
        w_l = min(10, (wg_per_cu_lds * waves_per_wg) // 4)  # 4 SIMDs per CU
        occ = min(w_v, w_s, w_l)
        lim = 'VGPR' if occ == w_v else ('SGPR' if occ == w_s else 'LDS')
        rows.append(dict(module=path.split('/')[-1], kernel=k, **r, insts=n, load_x4=wide, load_narrow=narrow, stores=stores,
                         lds_ops=lds_ops, s_waitcnt=wait, s_barrier=bar,
                         waitcnt_per100=round(100.0 * wait / n, 2) if n else 0.0,
                         barrier_per100=round(100.0 * bar / n, 2) if n else 0.0,
                         x4_share=round(wide / (wide + narrow), 3) if (wide + narrow) else 0.0,
                         occ_waves_simd=occ, occ_limiter=lim))

hdr = ['module', 'kernel', 'vgpr', 'sgpr', 'lds', 'scratch', 'code', 'insts', 'load_x4', 'load_narrow', 'x4_share', 'stores',
       'lds_ops', 's_waitcnt', 'waitcnt_per100', 's_barrier', 'barrier_per100', 'occ_waves_simd', 'occ_limiter']
print('| ' + ' | '.join(hdr) + ' |')
print('|' + '---|' * len(hdr))
for r in rows:
    print('| ' + ' | '.join(str(r[h]) for h in hdr) + ' |')
if a.json:
    json.dump(rows, open(a.json, 'w'), indent=1)
