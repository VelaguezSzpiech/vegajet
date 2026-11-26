"""CPU source contracts for GPU-only orchestration (not GPU runtime proof)."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FixContracts(unittest.TestCase):
    def test_fault_storage_lives_through_last_read(self):
        src = (ROOT / 'models/ornith/ornith_gpu.cpp').read_text()
        tail = src[src.index('    } else {\n        if (opt.run_parity)'):src.rindex('\nint run_ornith_ar_bench')]
        self.assertLess(tail.index('device_fault_check("run end")'), tail.index('rt.free_buffer(io_buf)'))
        self.assertNotIn('device_fault_check(', tail[tail.index('rt.free_buffer(io_buf)'):])


if __name__ == '__main__':
    unittest.main()
