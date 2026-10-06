"""Explicitly approved memory amendment; all financial code remains unchanged."""
import argparse
import copy
import ctypes
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path

from extension4.authority import require_approval
from extension4.run import Study, require_engineering, validate_protocol
from extension4.runtime import Runtime
from extension4.storage import verify_stage, write_json
from feature_geometry.history import sha
from temporal_fast.backend import single_thread_blas
import pyarrow as pa


PROPOSAL = 'config/extension4_memory_limit_amendment_v2_proposed.json'
APPROVAL = 'config/extension4_memory_limit_amendment_v2_approval.json'
EVIDENCE = 'output/extension4_memory_v2_20260926/engineering_manifest.json'
SOURCES = ('extension4_execution/__init__.py', 'extension4_execution/memory_v2.py',
           'scripts/extension4_resume_memory_v2.py', 'tests/test_memory_execution_v2.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def pressure_level():
    """Read the native dispatch value: NORMAL=1, WARN=2, CRITICAL=4."""
    libc = ctypes.CDLL(None, use_errno=True)
    call = libc.sysctlbyname
    call.argtypes = [ctypes.c_char_p, ctypes.c_void_p,
                     ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t]
    call.restype = ctypes.c_int
    value = ctypes.c_int()
    size = ctypes.c_size_t(ctypes.sizeof(value))
    if call(b'kern.memorystatus_vm_pressure_level', ctypes.byref(value),
            ctypes.byref(size), None, 0) != 0:
        raise OSError(ctypes.get_errno(), 'Cannot read native macOS memory pressure')
    if size.value != ctypes.sizeof(value):
        raise ValueError('Unexpected native memory-pressure representation')
    return value.value


def amended_config(config, proposal):
    """Copy only the execution budget; never alter the saved science protocol."""
    expected = dict(max_physical_footprint_gib=16.0, max_memory_cache_gib=4.0,
                    total_active_seconds=259200, maximum_workers_in_use=1,
                    blas_threads=1, max_additional_disk_gib=20.0,
                    min_free_disk_gib=50.0, stop_on_swap_growth=True,
                    local_only=True, paid=False)
    pressure = dict(source='kern.memorystatus_vm_pressure_level',
                    required_normal_dispatch_value=1,
                    stop_on_warning_critical_unknown_or_unavailable=True,
                    sampling_seconds=2.0)
    if (proposal.get('original_min_system_headroom_gib') != 8.0
            or proposal.get('proposed_min_system_headroom_gib') != 4.0
            or proposal.get('unchanged') != expected
            or proposal.get('pressure_guard') != pressure
            or proposal.get('scientific_changes') is not False
            or proposal.get('full_run_scope_unchanged') is not True):
        raise ValueError('Unsupported resource amendment')
    budget = config['compute_budget']
    if budget.get('min_system_headroom_gib') != 8.0:
        raise ValueError('The original approved reserve must remain intact')
    for key, value in expected.items():
        if key not in ('maximum_workers_in_use', 'stop_on_swap_growth') and budget.get(key) != value:
            raise ValueError('Unexpected original resource limit: ' + key)
    result = copy.deepcopy(config)
    result['compute_budget']['min_system_headroom_gib'] = 4.0
    return result


def verify_bindings(root, bindings):
    root = Path(root).resolve()
    for relative, expected in bindings.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path) != expected:
            raise PermissionError('Execution authority or implementation changed: ' + relative)


def load_authority(root):
    root = Path(root).resolve()
    original_config = root / 'config/extension4_study_v1_proposed.json'
    original_approval = root / 'config/extension4_end_to_end_approval.json'
    config, _ = require_approval(root, original_config, original_approval)
    validate_protocol(config)
    original_engineering_sha = require_engineering(root)
    approval_path = root / APPROVAL
    if not approval_path.is_file():
        raise PermissionError('Explicit 4-GiB resource amendment approval is required')
    approval = json.loads(approval_path.read_text())
    proposal = json.loads((root / PROPOSAL).read_text())
    if (approval.get('status') != 'APPROVED_BY_USER'
            or approval.get('user_message') != 'Approve 4 GB reserve and resume'
            or not approval.get('recorded_at_utc')
            or approval.get('proposal_sha256') != sha(root / PROPOSAL)
            or approval.get('original_config_sha256') != sha(original_config)
            or approval.get('original_approval_sha256') != sha(original_approval)
            or approval.get('original_engineering_sha256') != original_engineering_sha
            or approval.get('scientific_changes') is not False
            or approval.get('continue_full_workflow_after_checks') is not True):
        raise PermissionError('Incomplete or mismatched resource amendment approval')
    evidence_path = root / EVIDENCE
    if approval.get('engineering_manifest_sha256') != sha(evidence_path):
        raise PermissionError('Resource-engineering evidence does not match approval')
    evidence = json.loads(evidence_path.read_text())
    wanted = {relative: sha(root / relative) for relative in SOURCES}
    if (evidence.get('status') != 'VERIFIED_EXECUTION_ONLY_MEMORY_AMENDMENT'
            or evidence.get('source_sha256') != wanted
            or not evidence.get('artifact_sha256')
            or not all(evidence.get('checks', {}).get(k) is True for k in
                       ('guard_tests', 'original_engineering', 'checkpoint_compatibility',
                        'financial_preflight', 'native_pressure_read'))):
        raise PermissionError('Execution amendment is not verified')
    verify_bindings(root, evidence['artifact_sha256'])
    runtime_config = amended_config(config, proposal)
    bindings = {**wanted, PROPOSAL: sha(root / PROPOSAL), APPROVAL: sha(approval_path),
                EVIDENCE: sha(evidence_path), str(original_config.relative_to(root)): sha(original_config),
                str(original_approval.relative_to(root)): sha(original_approval)}
    return config, runtime_config, bindings


class PressureRuntime(Runtime):
    """Retain original accounting/guards and add native pressure telemetry."""

    def __init__(self, folder, config, phase, *, project_root, bindings):
        self.project_root = Path(project_root)
        self.execution_bindings = dict(bindings)
        verify_bindings(self.project_root, self.execution_bindings)
        super().__init__(folder, config, phase)

    def sample(self):
        # Runtime creates the unique attempt stem before its first sample.
        identity = Path(str(self.stem) + '_execution_amendment.json')
        if not identity.exists():
            write_json(identity, dict(version='memory_v2', recorded_at_utc=now(),
                original_scientific_lock_preserved=True,
                minimum_available_gib=4.0, required_native_pressure=1,
                source_and_authority_sha256=self.execution_bindings))
        error = None
        level = None
        try:
            try:
                level = pressure_level()
            except Exception as exc:
                raise MemoryError('Native memory pressure is unavailable; stop for review') from exc
            row = super().sample()
            if level != 1:
                raise MemoryError('Native memory pressure is not NORMAL: ' + str(level))
            return row
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            record = dict(at_utc=now(), native_pressure_dispatch_value=level,
                          normal=level == 1, error=error)
            with Path(str(self.stem) + '_pressure.jsonl').open('a') as handle:
                handle.write(json.dumps(record) + '\n')
                handle.flush()
                os.fsync(handle.fileno())

    def guard(self):
        super().guard()
        verify_bindings(self.project_root, self.execution_bindings)

    def event(self, event, **values):
        super().event(event, **values)
        if event == 'fit_complete':
            receipt = Path(str(self.stem) + '_first_completed_fit.json')
            if not receipt.exists():
                # Original Study.fit already verifies its saved model roundtrip
                # and commits its inventory before emitting this event.
                folder = self.folder / 'fits' / str(values['year']) / ('seed' + str(values['seed']))
                verify_stage(folder)
                sample = self.sample()
                write_json(receipt, dict(status='PASS', at_utc=now(), year=values['year'],
                    seed=values['seed'], fit_commit_sha256=sha(folder / 'commit.json'),
                    current_resources=sample, native_pressure_normal=True,
                    scientific_pipeline_unchanged=True,
                    continue_approved_full_workflow=True))


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['status', 'run'])
    args = parser.parse_args(argv)
    if args.command == 'status':
        print(json.dumps(dict(resource_amendment_approval_exists=(root / APPROVAL).is_file(),
                              engineering_evidence_exists=(root / EVIDENCE).is_file(),
                              native_pressure_dispatch_value=pressure_level()), indent=2))
        return
    config, runtime_config, bindings = load_authority(root)
    destination = (root / config['paths']['proposed_full_root']).resolve()
    if not destination.is_relative_to(root / 'output'):
        raise ValueError('Execution must remain in the authorized output directory')
    # This entry point resumes the already-passed historical pilot only.
    verify_stage(destination / 'pilot_gate')
    if json.loads((destination / 'pilot_gate/result.json').read_text())['status'] != 'PASS':
        raise ValueError('A passed original pilot is required')
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    runtime = None
    with (root / 'output/.extension3_heavy_job.lock').open('a+') as heavy, single_thread_blas():
        fcntl.flock(heavy, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            runtime = PressureRuntime(destination, runtime_config, 'full',
                                      project_root=root, bindings=bindings)
            # Study sees the ORIGINAL configuration and source identity. Only
            # Runtime receives the separately approved execution-budget copy.
            study = Study(root, root / 'config/extension4_study_v1_proposed.json',
                          root / 'config/extension4_end_to_end_approval.json')
            study.runtime = runtime
            verify_stage(destination / 'pilot_gate', dict(lock=study.lock))
            runtime.event('attempt_started', execution_amendment='memory_v2')
            study.full()
            runtime.close('COMPLETE')
        except BaseException as error:
            if runtime is not None:
                runtime.close('STOPPED', error)
            raise
