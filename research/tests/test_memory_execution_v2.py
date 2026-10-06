"""Exercise resource failures without running a financial experiment."""
from contextlib import ExitStack, redirect_stdout
import copy
import hashlib
import json
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from extension4.runtime import prior_seconds
from extension4.storage import commit_stage, write_json
from extension4_execution.memory_v2 import (PressureRuntime, amended_config,
                                          load_authority, verify_bindings,
                                          SOURCES, PROPOSAL, APPROVAL, EVIDENCE, main)

ROOT = Path(__file__).resolve().parents[1]


class MemoryAmendmentTests(unittest.TestCase):
    def setUp(self):
        self.original = json.loads((ROOT / 'config/extension4_study_v1_proposed.json').read_text())
        self.proposal = json.loads((ROOT / 'config/extension4_memory_limit_amendment_v2_proposed.json').read_text())
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'bound.txt').write_text('verified source')
        self.bindings = {'bound.txt': hashlib.sha256(b'verified source').hexdigest()}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.memory = SimpleNamespace(available=5 * 2**30)
        self.swap = SimpleNamespace(used=0)
        self.charge = dict(physical_footprint_bytes=7 * 2**30, resident_bytes=7 * 2**30)
        process = Mock()
        process.create_time.return_value = 10.
        process.children.return_value = []
        process.cpu_times.return_value = SimpleNamespace(user=1., system=1.)
        self.stack.enter_context(patch('extension4.runtime.psutil.Process', return_value=process))
        self.stack.enter_context(patch('extension4.runtime.usage', side_effect=lambda pid: self.charge))
        self.stack.enter_context(patch('extension4.runtime.psutil.virtual_memory', side_effect=lambda: self.memory))
        self.stack.enter_context(patch('extension4.runtime.psutil.swap_memory', side_effect=lambda: self.swap))
        self.stack.enter_context(patch('extension4.runtime.shutil.disk_usage', return_value=SimpleNamespace(free=100 * 2**30)))
        self.stack.enter_context(patch('extension4.runtime.threading.Thread'))
        self.pressure = self.stack.enter_context(patch('extension4_execution.memory_v2.pressure_level', return_value=1))

    def runtime(self):
        runtime = PressureRuntime(self.root / 'study', amended_config(self.original, self.proposal),
                                  'full', project_root=self.root, bindings=self.bindings)
        self.addCleanup(runtime.close, 'TEST_COMPLETE')
        return runtime

    def test_only_reserve_changes_and_original_is_not_mutated(self):
        before = copy.deepcopy(self.original)
        result = amended_config(self.original, self.proposal)
        self.assertEqual(self.original, before)
        result['compute_budget']['min_system_headroom_gib'] = 8.
        self.assertEqual(result, before)

    def test_read_only_status_cannot_start_a_study(self):
        with patch('extension4_execution.memory_v2.load_authority', side_effect=AssertionError('No execution on status')), redirect_stdout(io.StringIO()) as out:
            main(['status'])
        self.assertEqual(json.loads(out.getvalue())['native_pressure_dispatch_value'], 1)

    def test_rejects_unapproved_weaker_memory_or_swap_controls(self):
        for section, key, value in [('unchanged', 'max_physical_footprint_gib', 20.),
                                    ('unchanged', 'stop_on_swap_growth', False),
                                    ('pressure_guard', 'required_normal_dispatch_value', 2)]:
            with self.subTest(key=key):
                bad = copy.deepcopy(self.proposal)
                bad[section][key] = value
                with self.assertRaises(ValueError):
                    amended_config(self.original, bad)
        self.proposal['proposed_min_system_headroom_gib'] = 2.
        with self.assertRaises(ValueError):
            amended_config(self.original, self.proposal)

    def test_five_gib_and_normal_pressure_pass_with_amendment_record(self):
        runtime = self.runtime()
        self.assertEqual(runtime.last_sample['system_available_gib'], 5.)
        self.assertEqual(runtime.last_sample['swap_growth_bytes'], 0)
        record = json.loads(Path(str(runtime.stem) + '_execution_amendment.json').read_text())
        self.assertEqual(record['source_and_authority_sha256'], self.bindings)
        self.assertEqual(record['minimum_available_gib'], 4.)

    def test_below_four_gib_stops(self):
        runtime = self.runtime()
        self.memory.available = int(3.99 * 2**30)
        with self.assertRaisesRegex(MemoryError, 'headroom'):
            runtime.sample()

    def test_sixteen_gib_process_limit_is_retained(self):
        runtime = self.runtime()
        self.charge['physical_footprint_bytes'] = 17 * 2**30
        with self.assertRaisesRegex(MemoryError, 'Process tree'):
            runtime.sample()

    def test_any_swap_growth_stops(self):
        runtime = self.runtime()
        self.swap.used = 16384
        with self.assertRaisesRegex(MemoryError, 'swap grew'):
            runtime.sample()

    def test_warning_critical_and_unknown_pressure_stop(self):
        runtime = self.runtime()
        for level in [2, 4, 0, 3, 9]:
            with self.subTest(level=level):
                self.pressure.return_value = level
                with self.assertRaisesRegex(MemoryError, 'not NORMAL'):
                    runtime.sample()

    def test_missing_pressure_fails_closed_and_logs_error(self):
        runtime = self.runtime()
        self.pressure.side_effect = OSError('unavailable')
        with self.assertRaisesRegex(MemoryError, 'unavailable'):
            runtime.sample()
        last = json.loads(Path(str(runtime.stem) + '_pressure.jsonl').read_text().splitlines()[-1])
        self.assertFalse(last['normal'])
        self.assertIn('unavailable', last['error'])

    def test_existing_budget_and_incomplete_attempt_are_carried_forward(self):
        folder = self.root / 'study/attempts'
        folder.mkdir(parents=True)
        write_json(folder / '0001_start.json', dict(phase='full', sampling_interval_seconds=2.))
        (folder / '0001_resources.jsonl').write_text('{"active_seconds": 10}\n')
        runtime = self.runtime()
        self.assertEqual(runtime.previous, 12.)
        self.assertEqual(runtime.budget['total_active_seconds'], 259200)
        runtime.previous = 259201
        with self.assertRaises(TimeoutError):
            runtime.guard()

    def test_disk_reserve_remains_enforced(self):
        runtime = self.runtime()
        with patch('extension4.runtime.shutil.disk_usage', return_value=SimpleNamespace(free=49 * 2**30)):
            with self.assertRaises(OSError):
                runtime.sample()

    def test_changed_execution_binding_stops_before_more_work(self):
        runtime = self.runtime()
        (self.root / 'bound.txt').write_text('modified')
        with self.assertRaises(PermissionError):
            runtime.guard()

    def test_first_completed_fit_requires_commit_and_current_guard_pass(self):
        runtime = self.runtime()
        folder = runtime.folder / 'fits/1997/seed2'
        commit_stage(folder, {'artificial_test': True}, lambda p: write_json(p / 'fit.json', {'artificial': True}))
        runtime.event('fit_complete', year=1997, seed=2)
        receipt = json.loads(Path(str(runtime.stem) + '_first_completed_fit.json').read_text())
        self.assertEqual(receipt['status'], 'PASS')
        self.assertTrue(receipt['continue_approved_full_workflow'])

    def test_no_new_fit_gate_on_reused_checkpoints(self):
        runtime = self.runtime()
        runtime.event('fit_reused', year=1981, seed=1)
        self.assertFalse(Path(str(runtime.stem) + '_first_completed_fit.json').exists())

    def test_original_approval_does_not_authorize_new_resource_policy(self):
        config = self.original
        with patch('extension4_execution.memory_v2.require_approval', return_value=(config, {})), \
             patch('extension4_execution.memory_v2.require_engineering', return_value='old-engineering'):
            with self.assertRaisesRegex(PermissionError, 'Explicit 4-GiB'):
                load_authority(self.root)

    def test_bound_approval_accepts_exact_sources_and_rejects_drift(self):
        def save(rel, content):
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(content))
            return hashlib.sha256(p.read_bytes()).hexdigest()
        def digest(rel):
            return hashlib.sha256((self.root / rel).read_bytes()).hexdigest()
        config_hash = save('config/extension4_study_v1_proposed.json', self.original)
        original_approval_hash = save('config/extension4_end_to_end_approval.json', {'fixture': True})
        proposal_hash = save(PROPOSAL, self.proposal)
        for rel in SOURCES:
            save(rel, {'artificial_source_fixture': True})
        receipt_hash = save('evidence/test.json', {'artificial': True})
        evidence_hash = save(EVIDENCE, dict(status='VERIFIED_EXECUTION_ONLY_MEMORY_AMENDMENT',
            source_sha256={rel: digest(rel) for rel in SOURCES},
            artifact_sha256={'evidence/test.json': receipt_hash},
            checks={k: True for k in ['guard_tests', 'original_engineering',
                'checkpoint_compatibility', 'financial_preflight', 'native_pressure_read']}))
        save(APPROVAL, dict(status='APPROVED_BY_USER', user_message='Approve 4 GB reserve and resume',
            recorded_at_utc='artificial-test', proposal_sha256=proposal_hash,
            original_config_sha256=config_hash, original_approval_sha256=original_approval_hash,
            original_engineering_sha256='original-engineering', scientific_changes=False,
            continue_full_workflow_after_checks=True, engineering_manifest_sha256=evidence_hash))
        with patch('extension4_execution.memory_v2.require_approval', return_value=(self.original, {})), \
             patch('extension4_execution.memory_v2.require_engineering', return_value='original-engineering'):
            original, runtime, bindings = load_authority(self.root)
            self.assertEqual(original['compute_budget']['min_system_headroom_gib'], 8.)
            self.assertEqual(runtime['compute_budget']['min_system_headroom_gib'], 4.)
            verify_bindings(self.root, bindings)
            save(PROPOSAL, {'modified': True})
            with self.assertRaises(PermissionError):
                load_authority(self.root)


if __name__ == '__main__':
    unittest.main()
