"""CPU source contracts for GPU-only orchestration (not GPU runtime proof)."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FixContracts(unittest.TestCase):
    def test_layer_super_abi(self):
        src = (ROOT / 'tests/vegajet_layer_test.cpp').read_text()
        for kind in ('Gdn', 'Gqa', 'Moe'):
            self.assertIn('using KA = vegajet::ornith_super::' + kind + 'Args;', src)
        for name in ('gdn_desc_va', 'gqa_desc_va', 'moe_desc_va'):
            init = next(line for line in src.splitlines() if 'const KA ka{' + name in line)
            self.assertTrue(init.endswith('0ULL};'))

    def test_fault_storage_lives_through_last_read(self):
        src = (ROOT / 'models/ornith/ornith_gpu.cpp').read_text()
        tail = src[src.index('    } else {\n        if (opt.run_parity)'):src.rindex('\nint run_ornith_ar_bench')]
        self.assertLess(tail.index('device_fault_check("run end")'), tail.index('rt.free_buffer(io_buf)'))
        self.assertNotIn('device_fault_check(', tail[tail.index('rt.free_buffer(io_buf)'):])


if __name__ == '__main__':
    unittest.main()
