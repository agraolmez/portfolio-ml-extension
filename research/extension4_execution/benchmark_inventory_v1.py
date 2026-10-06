"""Admit only the approved F1024 comparison models through their accepted E2 chain.

The original E3 catalog includes E2's acceptance manifest and benchmark paths,
but not F1024 annual models (E3 reused its archived paths). E4 replays those
models. This separately versioned adapter changes no fit, aim, or accounting.
"""
import argparse
import fcntl
import json
from pathlib import Path

import pyarrow as pa

from extension4.run import Study
from extension4.storage import commit_stage, verify_stage, write_json
from extension4_execution.memory_v2 import PressureRuntime, load_authority, verify_bindings
from feature_geometry.history import sha
from temporal_fast.backend import single_thread_blas

GEOMETRY = 'output/feature_geometry_study_20260914'
EVIDENCE = 'output/extension4_benchmark_fix_20260928/engineering_manifest.json'
SOURCES = ('extension4_execution/benchmark_inventory_v1.py',
           'scripts/extension4_resume_benchmark_v1.py',
           'tests/test_extension4_benchmark_inventory.py',
           'scripts/verify_extension4_benchmark_inventory.py')


def include_accepted_benchmarks(catalog):
    """Verify the nested acceptance and all 135 pairs before changing the catalog."""
    manifest_rel = GEOMETRY + '/acceptance_manifest.json'
    # path/read uses the E3-accepted hash, never a newly calculated trust anchor.
    accepted = catalog.read(manifest_rel)
    if (accepted.get('status') != 'ACCEPTED_COMPLETED_HISTORICAL_GEOMETRY_STUDY'
            or accepted.get('accepted') is not True
            or accepted.get('study_complete') is not True
            or accepted.get('years') != [1981, 2025]
            or accepted.get('frequency_seeds') != [1, 2, 3]
            or not accepted.get('gates')
            or not all(g.get('passed') is True for g in accepted['gates'])):
        raise ValueError('Completed accepted geometry comparison required')
    inventory = accepted['artifact_sha256']
    admitted = {}
    for year in range(1981, 2026):
        for seed in (1, 2, 3):
            folder = f'{GEOMETRY}/fits/{year}/seed{seed}/fixed1024'
            for name in ('model.npz', 'fit.json'):
                relative = folder + '/' + name
                if relative not in inventory:
                    raise ValueError('Accepted F1024 artifact missing: ' + relative)
                expected = inventory[relative]
                if relative in catalog.inventory and catalog.inventory[relative] != expected:
                    raise ValueError('Conflicting accepted inventories: ' + relative)
                path = (catalog.root / relative).resolve()
                if not path.is_relative_to(catalog.root) or sha(path) != expected:
                    raise ValueError('Accepted F1024 file changed: ' + relative)
                admitted[relative] = expected
            receipt = json.loads((catalog.root / folder / 'fit.json').read_text())
            if (receipt.get('status') != 'EXACT_FIXED_FEATURE_SOLVE'
                    or receipt.get('method') != 'fixed1024'
                    or receipt.get('year') != year or receipt.get('seed') != seed
                    or receipt.get('cutoff') != f'{year-1}-12-31'
                    or receipt.get('files_sha256', {}).get('model.npz') != inventory[folder+'/model.npz']):
                raise ValueError('Accepted F1024 annual receipt mismatch: ' + folder)
    # Only the exact approved comparison files are added. Parent selection,
    # accepted-parent identity, and all existing inventory hashes remain intact.
    catalog.inventory.update(admitted)
    for relative in admitted:
        catalog.path(relative)
    return dict(parent_acceptance_sha256=catalog.acceptance_sha256,
                geometry_acceptance_sha256=catalog.inventory[manifest_rel],
                artifact_sha256=admitted, annual_comparisons=135,
                financial_code_changed=False, scientific_settings_changed=False)


def load_repair_bindings(root):
    evidence_path = root / EVIDENCE
    evidence = json.loads(evidence_path.read_text())
    wanted = {relative: sha(root / relative) for relative in SOURCES}
    if (evidence.get('status') != 'VERIFIED_ACCEPTED_BENCHMARK_INVENTORY_REPAIR'
            or evidence.get('source_sha256') != wanted
            or not all(evidence.get('checks', {}).get(name) is True for name in
                       ('authority_unchanged', 'nested_acceptance_chain', 'all_135_fits_preserved',
                        'unit_tests', 'financial_preflight', 'all_controls_actual_replay', 'saved_restart'))):
        raise ValueError('Verified benchmark repair evidence required')
    verify_bindings(root, evidence['artifact_sha256'])
    return {**wanted, EVIDENCE: sha(evidence_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run'])
    parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    config, runtime_config, bindings = load_authority(root)
    bindings.update(load_repair_bindings(root))
    destination = (root / config['paths']['proposed_full_root']).resolve()
    if not destination.is_relative_to(root / 'output'):
        raise ValueError('Execution must remain in project output')
    verify_stage(destination / 'pilot_gate')
    if json.loads((destination / 'pilot_gate/result.json').read_text())['status'] != 'PASS':
        raise ValueError('Passed original pilot required')
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    runtime = None
    with (root / 'output/.extension3_heavy_job.lock').open('a+') as heavy, single_thread_blas():
        fcntl.flock(heavy, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            runtime = PressureRuntime(destination, runtime_config, 'full', project_root=root, bindings=bindings)
            study = Study(root, root / 'config/extension4_study_v1_proposed.json',
                          root / 'config/extension4_end_to_end_approval.json')
            study.runtime = runtime
            receipt = include_accepted_benchmarks(study.catalog)
            commit_stage(destination / 'benchmark_inventory_v1',
                         dict(lock=study.lock, engineering_sha256=bindings[EVIDENCE]),
                         lambda p: write_json(p / 'receipt.json', receipt))
            verify_stage(destination / 'pilot_gate', dict(lock=study.lock))
            runtime.event('attempt_started', execution_amendment='memory_v2',
                          dependency_repair='benchmark_inventory_v1')
            study.full()
            runtime.close('COMPLETE')
        except BaseException as error:
            if runtime is not None:
                runtime.close('STOPPED', error)
            raise


if __name__ == '__main__':
    main()
