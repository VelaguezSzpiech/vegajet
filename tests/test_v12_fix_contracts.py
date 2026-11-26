"""CPU source contracts for GPU-only orchestration (not GPU runtime proof)."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FixContracts(unittest.TestCase):
    def test_failures_unwind_all_drivers(self):
        src = (ROOT / 'models/ornith/ornith_gpu.cpp').read_text()
        self.assertNotIn('return std::vector<int32_t>(M, -1)', src)
        self.assertLess(src.index('BAD ARGMAX'), src.index('// Update Dynamic LRU MoE'))
        for where in ('previous dispatch', 'decode step', 'MTP step', 'super profile'):
            self.assertIn('if (device_fault_check("' + where + '")) fail_run(res.error);', src)
        self.assertIn('catch (const guard::RunFailure& failure)', src)
        for where in ('MTP embedding', 'MTP argmax', 'DFlash anchor embedding', 'DFlash draft argmax'):
            self.assertIn('"' + where + '"', src)
        self.assertIn('if (!run_state.error().empty() || !res.error.empty() || device_faulted)', src)
        body = src[src.index('    auto run_full_bench_body'):src.index('    if (opt.serve_hook)')]
        self.assertNotIn('rt.free_buffer(io_buf)', body)
        self.assertIn('if (failed_) throw guard::RunFailure(start_err_);', src)
        self.assertIn('return req_done_ || stop_ || failed_;', src)
        self.assertIn('Ornith GPU engine stopped unexpectedly', src)

    def test_batched_collective_epochs(self):
        src = (ROOT / 'models/ornith/ornith_gpu.cpp').read_text()
        batch = src[src.index('    auto step_mtp_batch ='):src.index('    auto step_mtp_batch =') + 17000]
        for stage in ('attn', 'moe'):
            self.assertIn('const auto ' + stage + '_collective = dispatch::next_collective(ar_epoch);\n        for (size_t g', batch)
            self.assertIn('const uint32_t ep = ' + stage + '_collective.epoch;', batch)
            self.assertIn('const uint32_t pp = ' + stage + '_collective.parity;', batch)

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
