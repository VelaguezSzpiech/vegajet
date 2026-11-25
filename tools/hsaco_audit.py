#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Offline gfx906 metadata/ISA audit. No GPU runtime, third-party modules or waivers."""
import argparse
import collections
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


class AuditError(ValueError):
    pass


def metadata(text):
    """Read LLVM's rendered AMDHSA YAML, not arbitrary YAML or guessed defaults."""
    match = re.search(r'^amdhsa\.kernels:\s*\n(.*?)(?=^amdhsa\.|^\.\.\.)', text, re.M | re.S)
    if not match:
        raise AuditError('missing AMDHSA kernel metadata')
    blocks = re.split(r'^  - ', match[1], flags=re.M)[1:]
    result = []
    required = ('vgpr_count', 'sgpr_count', 'group_segment_fixed_size',
                'private_segment_fixed_size', 'wavefront_size')
    for block in blocks:
        fields = dict(re.findall(r'^\s{4}\.(\w+):\s*([^\n]+)', '    ' + block, re.M))
        if 'name' not in fields or any(k not in fields for k in required):
            raise AuditError('incomplete kernel metadata')
        row = {'name': fields['name'].strip().strip('\"\'')}
        for key in required:
            try:
                row[key] = int(fields[key], 0)
            except ValueError as exc:
                raise AuditError('invalid numeric metadata') from exc
            if row[key] < 0:
                raise AuditError('negative metadata')
        if row['wavefront_size'] != 64:
            raise AuditError('not Wave64')
        if fields.get('uses_dynamic_stack', 'false').strip() != 'false':
            raise AuditError('dynamic private stack')
        result.append(row)
    if not result or len({r['name'] for r in result}) != len(result):
        raise AuditError('empty or duplicate kernel metadata')
    return result


def symbol_sizes(text):
    sizes = {}
    for line in text.splitlines():
        match = re.match(r'^([0-9a-fA-F]+)\s+.*?\bF\s+\.text\s+([0-9a-fA-F]+)\s+(?:\.\w+\s+)?(\S+)$', line)
        if match:
            name = match[3]
            sizes[name] = (int(match[1], 16), int(match[2], 16))
    return sizes


def instructions(text, address, size):
    # AMD LLVM prints addresses after //; generic LLVM may put them before the opcode.
    result = []
    parsed = []
    targets = set()
    for line in text.splitlines():
        match = re.match(r'^\s*(.*?)\s*//\s*([0-9a-fA-F]+):\s*[0-9a-fA-F ]+(?:\s*<([^>]+)>)?$', line)
        if match:
            code, addr = match[1], int(match[2], 16)
            if address <= addr < address + size and match[3] and re.match(r's_(?:c?branch)', code.strip()):
                target = re.search(r'\+0x([0-9a-fA-F]+)$', match[3])
                targets.add(address + int(target[1], 16) if target else address)
        else:
            match = re.match(r'^\s*([0-9a-fA-F]+):\s+(?:[0-9a-fA-F]{2}\s+)+(.+)$', line)
            if not match:
                continue
            addr, code = int(match[1], 16), match[2]
        if address <= addr < address + size:
            parsed.append((addr, code.strip()))
    for addr, code in parsed:
        if addr in targets:
            result.append('label:')
        result.append(code)
    if not result or any('<unknown>' in line for line in result):
        raise AuditError('missing or unknown disassembly')
    return result


def analyze(row, code, size, klass):
    row = dict(row, text_bytes=size, klass=klass)
    row['waves'] = min(10, 256 // max(4, ((row['vgpr_count'] + 3) // 4) * 4),
                       800 // max(16, ((row['sgpr_count'] + 15) // 16) * 16))
    # Register-limited upper bound, not achieved occupancy: WG size/dynamic LDS are unknown.
    row['dots'] = [sum(bool(re.match(pattern, c)) for c in code)
                   for pattern in (r'v_dot4', r'v_dot8', r'v_(?:pk_)?f?dot2')]
    loads = [c for c in code if re.match(r'(?:buffer|global|flat)_load', c)]
    row['flags'] = dict(collections.Counter((int(bool(re.search(r'\bglc\b', c))),
                                           int(bool(re.search(r'\bslc\b', c)))) for c in loads))
    row['lds_bad'] = sum(bool(re.match(r'buffer_load_dwordx4\b.*\blds\b', c)) for c in code)
    row['waits'] = sorted({int(n) for c in code for n in re.findall(r'\bvmcnt\((\d+)\)', c)})
    row['cliff'] = '48 cliff' if 48 < row['vgpr_count'] <= 52 else ('64 cliff' if 64 < row['vgpr_count'] <= 68 else '')
    failures = []
    if row['private_segment_fixed_size']:
        failures.append('private segment/spill')
    if klass not in ('stream', 'tiled', 'super'):
        failures.append('unclassified')
    elif row['vgpr_count'] > (48 if klass == 'stream' else 64):
        failures.append('VGPR ceiling')
    if size >= 24576:
        failures.append('text ceiling')
    if row['lds_bad']:
        failures.append('illegal vector LDS load')
    for index, instruction in enumerate(code):
        if instruction != 's_barrier':
            continue
        fence = confirmed = lds_wait = False
        for previous in reversed(code[:index]):
            if re.match(r's_waitcnt\b.*\blgkmcnt\(0\)', previous):
                lds_wait = True
            if previous == 'buffer_wbinvl1_vol':
                fence = True
            if re.match(r's_waitcnt\b.*\bvmcnt\(0\)', previous):
                confirmed = fence
            if confirmed and lds_wait:
                break
            if re.match(r'(?:buffer|global|flat)_(?:load|store|atomic)|ds_(?:write|read|atomic)|s_(?:c?branch|barrier)|label:', previous):
                break
        if not confirmed or not lds_wait:
            failures.append('barrier lacks local vmcnt(0)/lgkmcnt(0)/system-fence sequence')
            break
    row['failures'] = failures
    return row


def manifest(path):
    result = {}
    if path:
        for line in Path(path).read_text().splitlines():
            fields = line.split('#', 1)[0].split()
            if not fields:
                continue
            if len(fields) not in (2, 3) or fields[1] not in ('stream', 'tiled', 'super') or fields[0] in result:
                raise AuditError('invalid gate manifest')
            # Legacy waiver column is intentionally ignored, never honored.
            result[fields[0]] = fields[1]
    return result


def run(tool, *args):
    proc = subprocess.run([tool, *args], capture_output=True, text=True, timeout=180)
    if proc.returncode:
        raise AuditError(Path(tool).name + ' failed (diagnostics suppressed for privacy)')
    return proc.stdout


def audit(path, objdump, readobj, classes, default_class=None):
    notes = run(readobj, '--notes', str(path))
    if not re.search(r'amdhsa\.target:.*gfx906\b', notes):
        raise AuditError('code object is not gfx906')
    dump = run(objdump, '-h', '-t', '-d', '--mcpu=gfx906', str(path))
    sizes = symbol_sizes(dump)
    rows = []
    for row in metadata(notes):
        if row['name'] not in sizes or not sizes[row['name']][1]:
            raise AuditError('missing sized kernel symbol')
        addr, size = sizes[row['name']]
        rows.append(analyze(row, instructions(dump, addr, size), size, classes.get(row['name'], default_class)))
    return rows


def table(rows):
    lines = ['| Kernel | VGPR | waves/SIMD upper bound | SGPR | LDS B | .text B | private B | dot4/dot8/fdot2 | load glc,slc:count | vmcnt thresholds | illegal LDS | gate |',
             '|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|---|']
    for r in rows:
        flags = ', '.join(f'{a},{b}:{n}' for (a, b), n in sorted(r['flags'].items())) or 'none'
        status = ', '.join(r['failures']) or 'PASS'
        if r['cliff']:
            status += ' **' + r['cliff'] + '**'
        name = r['name'].replace('|', '\\|')
        lines.append(f"| {name} | {r['vgpr_count']} | {r['waves']} | {r['sgpr_count']} | {r['group_segment_fixed_size']} | {r['text_bytes']} | {r['private_segment_fixed_size']} | {'/'.join(map(str, r['dots']))} | {flags} | {','.join(map(str, r['waits'])) or 'none'} | {r['lds_bad']} | {status} |")
    lines += ['', 'Occupancy is a register-limited upper bound, not a measurement. LDS/workgroup residency can lower it.',
              'vmcnt values are static outstanding-operation thresholds, NOT measured prefetch distance.',
              'Instruction counts are static; no bandwidth/roofline throughput is inferred without bytes, clocks and runtime.',
              'Cache flags are reported, not inferred as correct from absent workload contracts. Fence dominance and inter-GEMM storage require source/control-flow review.']
    return '\n'.join(lines)


def self_test():
    note = 'amdhsa.kernels:\n  - .args: []\n    .name: test\n    .vgpr_count: 49\n    .sgpr_count: 32\n    .group_segment_fixed_size: 0\n    .private_segment_fixed_size: 0\n    .wavefront_size: 64\namdhsa.version: [1, 2]\n'
    r = metadata(note)[0]
    d = '0000000000001000 g F .text 0000000000000010 .protected test\n buffer_load_dwordx4 v[0:3], v0, s[0:3], 0 glc slc lds // 000000001000: E0000000 00000000\n s_waitcnt vmcnt(3) // 000000001008: BF8C0F73\n v_dot4_i32_i8 v0, v1, v2, v0 // 00000000100C: D0000000\n'
    assert symbol_sizes(d)['test'] == (4096, 16)
    result = analyze(r, instructions(d, 4096, 16), 16, 'stream')
    assert result['waves'] == 4 and result['lds_bad'] == 1
    assert result['flags'] == {(1, 1): 1} and result['waits'] == [3]
    assert result['dots'] == [1, 0, 0] and result['cliff'] == '48 cliff'
    assert len(result['failures']) == 2
    assert analyze(dict(r, vgpr_count=64), ['s_endpgm'], 24576, 'super')['failures'] == ['text ceiling']
    assert analyze(dict(r, vgpr_count=48), ['s_endpgm'], 16, 'stream')['waves'] == 5
    assert analyze(dict(r, private_segment_fixed_size=4), ['s_endpgm'], 16, 'tiled')['failures']
    assert analyze(r, ['s_barrier'], 4, 'tiled')['failures']
    assert not analyze(r, ['s_waitcnt vmcnt(0) lgkmcnt(0)', 'buffer_wbinvl1_vol', 's_barrier'], 16, 'tiled')['failures']
    scheduled = ['s_waitcnt vmcnt(0) lgkmcnt(0)'] + ['v_add_u32 v0, v0, 1'] * 32 + ['buffer_wbinvl1_vol', 's_barrier']
    assert not analyze(r, scheduled, 160, 'tiled')['failures']
    for hazard in ('global_load_dword v0, v1, off', 's_branch 4'):
        assert analyze(r, ['s_waitcnt vmcnt(0)', hazard, 'buffer_wbinvl1_vol', 's_barrier'], 24, 'tiled')['failures']
    branch_dump = ' s_branch 0 // 000000001000: BF820000 <test+0x4>\n s_barrier // 000000001004: BF8A0000\n'
    assert instructions(branch_dump, 4096, 8) == ['s_branch 0', 'label:', 's_barrier']
    assert analyze(r, ['ds_write_b128 v0, v[1:4]', 's_waitcnt vmcnt(0)', 'buffer_wbinvl1_vol', 's_barrier'], 24, 'tiled')['failures']
    assert analyze(r, ['s_waitcnt vmcnt(0) lgkmcnt(0)', 'label:', 'buffer_wbinvl1_vol', 's_barrier'], 24, 'tiled')['failures']
    for broken in ('', note.replace('    .sgpr_count: 32\n', ''), note.replace('64\n', '32\n')):
        try:
            metadata(broken)
        except AuditError:
            pass
        else:
            raise AssertionError('malformed metadata accepted')
    print('hsaco_audit self-test: PASS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='*', type=Path)
    parser.add_argument('--objdump', default=os.environ.get('LLVM_OBJDUMP', 'llvm-objdump'))
    parser.add_argument('--readobj', default=os.environ.get('LLVM_READOBJ', 'llvm-readobj'))
    parser.add_argument('--manifest')
    parser.add_argument('--default-class', choices=('stream', 'tiled', 'super'))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    files = sorted({f for p in args.paths for f in (p.rglob('*.hsaco') if p.is_dir() else [p])})
    if not files:
        raise AuditError('no code objects: audit is not a pass')
    for name in ('objdump', 'readobj'):
        if not shutil.which(getattr(args, name)):
            raise AuditError('LLVM tool unavailable')
    classes = manifest(args.manifest)
    failed = False
    for index, path in enumerate(files, 1):
        print(f'\n## Code object {index}\n')
        try:
            rows = audit(path, args.objdump, args.readobj, classes, args.default_class)
            print(table(rows))
            failed |= any(r['failures'] for r in rows)
        except (AuditError, OSError, subprocess.TimeoutExpired) as exc:
            print('FAIL: ' + (str(exc) if isinstance(exc, AuditError) else 'tool/file failure'))
            failed = True
    return int(failed)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (AuditError, OSError) as error:
        print('FAIL: ' + (str(error) if isinstance(error, AuditError) else 'file access failure'), file=sys.stderr)
        sys.exit(1)
