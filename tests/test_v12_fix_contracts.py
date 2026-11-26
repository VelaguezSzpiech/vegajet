"""CPU source contracts for GPU-only orchestration (not GPU runtime proof)."""
from pathlib import Path
import unittest
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


class FixContracts(unittest.TestCase):
    def test_runner_clears_inherited_flags_and_returns_failure(self):
        # A deliberate mock CLI validates the runner, never serves as inference evidence.
        with tempfile.TemporaryDirectory(prefix='v12-runner-', dir=ROOT) as td:
            td = Path(td)
            mock = td / 'mock'
            mock.write_text('#!/usr/bin/env python3\nimport os, sys\n'
                            'assert "VEGAJET_V12_FUTURE_FLAG" not in os.environ\n'
                            'assert "VEGAJET_V12_IQ2S_REG" not in os.environ\n'
                            'if "--json-out" in sys.argv: sys.exit(int(os.environ.get("MOCK_EXIT", "0")))\n')
            mock.chmod(0o755)
            env = dict(os.environ, MODEL='test-only', BIN=str(mock), RUNS='1', ONLY='VOCAB_ILP',
                       VEGAJET_V12_FUTURE_FLAG='1', VEGAJET_V12_IQ2S_REG='1')
            for status in (0, 17):
                out = td / str(status)
                env.update(OUT=str(out), MOCK_EXIT=str(status))
                result = subprocess.run(['bash', str(ROOT / 'bench/v12/run.sh')], cwd=ROOT,
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0 if status == 0 else 1, result.stdout + result.stderr)
                self.assertEqual((out / 'baseline_raw.env').read_text(), '')
                self.assertEqual((out / 'VOCAB_ILP.env').read_text(), 'VEGAJET_V12_VOCAB_ILP=1\n')
                if status:
                    self.assertIn('FAILED (exit 17)', result.stdout)

    def test_batched_eh_rows_map_to_x(self):
        src = (ROOT / 'models/ornith/ornith_gpu.cpp').read_text()
        self.assertIn('*K[g].mtp_eh_cat, dispatch::eh_grid_x(M), 1, 1, 256, 1, 1', src)
        self.assertNotIn('*K[g].mtp_eh_cat, 256u, M, 1', src)

    def test_batched_staging_consumed_before_reuse(self):
        src = (ROOT / 'models/ornith/ornith_gpu.cpp').read_text()
        batch = src[src.index('    auto step_mtp_batch ='):src.index('    auto run_greedy_mtp =')]
        self.assertNotIn('do_submit', batch)
        self.assertIn('step_mtp(toks[j], pos0 + j, hs0[j], hs1[j], false, true, 0u)', batch)
        self.assertIn('VJ_CHECK(rt.wait_idle(0)    && rt.wait_idle(1));\n        return true;', batch)

    def test_unused_super_objects_are_isolated(self):
        src = (ROOT / 'core/kfd_runtime.cpp').read_text()
        optional = src[src.index('const char* optional[]'):src.index('for (const char* f : optional)')]
        self.assertNotIn('vj_ornith_super_vol.hsaco', optional)
        self.assertNotIn('vj_ornith_super_iq2sreg.hsaco', optional)
        self.assertIn('if (variant.object && !load_code_object(', src)

    def test_dense_persist_cannot_be_silent_noop(self):
        src = (ROOT / 'models/ornith/ornith_gpu.cpp').read_text()
        self.assertIn('if (use_m1_super && v12_dense_persist())', src)
        self.assertIn('++dense_persist_launches;', src)
        self.assertIn('D13 requested but no persistent GEMV launched', src)

    def test_split_attention_checked_before_launch(self):
        src = (ROOT / 'models/ornith/ornith_gpu.cpp').read_text()
        for dims in ('M, n_q_local', '1u, 16u', 'M, 16u'):
            self.assertIn('require_split_scratch(' + dims + ', n_chunks);', src)
        self.assertIn('require_kv_span(pos0, n);', src)
        self.assertIn('require_kv_span(pos, 1u);', src)
        self.assertIn('require_kv_span(pos0, M);', src)
        self.assertNotIn('(span + chunk_len - 1u)', src)

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

    def test_slice_protocol_all_schedules(self):
        # Abstract publication/observation model, not GPU execution. A host drain
        # gates the next publish on BOTH observations, including delayed rank B.
        def explore(slices, fence):
            seen, pending, deadlocks, complete = set(), [((0, 0), (0, 0))], 0, 0
            while pending:
                state = pending.pop()
                if state in seen:
                    continue
                seen.add(state)
                pc, flags = state
                if pc == (2 * slices, 2 * slices):
                    complete += 1
                    continue
                successors = []
                for rank in range(2):
                    step = pc[rank]
                    if step == 2 * slices:
                        continue
                    epoch = step // 2 + 1
                    new_flags = list(flags)
                    if step % 2 == 0:
                        if fence and min(pc) < 2 * (epoch - 1):
                            continue
                        new_flags[1 - rank] = epoch
                    elif flags[rank] != epoch:
                        continue
                    new_pc = list(pc)
                    new_pc[rank] += 1
                    successors.append((tuple(new_pc), tuple(new_flags)))
                deadlocks += not successors
                pending.extend(successors)
            return deadlocks, complete
        for slices in (2, 3):
            self.assertGreater(explore(slices, False)[0], 0)  # regression counterexample
            deadlocks, complete = explore(slices, True)
            self.assertEqual(deadlocks, 0)
            self.assertEqual(complete, 1)

    def test_sliced_collectives_drain_both_ranks_before_reuse(self):
        src = (ROOT / 'models/ornith/ornith_gpu.cpp').read_text()
        ar = src[src.index('        auto enq_p2p_ar_rms ='):src.index('        const bool use_m1_super')]
        self.assertIn('dispatch::collective_slice_needs_drain(m0, mc, M)', ar)
        self.assertIn('!rt.submit_batch(0) || !rt.submit_batch(1) ||', ar)
        self.assertIn('!rt.wait_idle(0) || !rt.wait_idle(1)', ar)
        self.assertIn('fail_run("all-reduce slice completion failed or timed out before mailbox reuse")', ar)

    def test_verify_barrier_is_fail_closed(self):
        src = (ROOT / 'ops/kernels/vj_spec_gdn.hip').read_text()
        self.assertIn('if (!grid_barrier_wbinvl1(nq_bar, gridDim.x, sense, {fault_va, sense})) return;', src)
        self.assertIn('device_fault::fault_record(fc, vegajet::device_fault::kVjFaultBarrier, 5u, 1u, sense)', src)
        self.assertIn('return sm_ok != 0u;', src)
        host = (ROOT / 'models/ornith/ornith_gpu.cpp').read_text()
        self.assertIn('dispatch::fused_verify_resident(fused->from_msgpack_note, rt.devices()[g].n_cu,', host)
        self.assertIn('spec_nq_sense, eps, io_fault_va[g]', host)
        self.assertIn('v12_fused_nq && fused_resident', host)

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
