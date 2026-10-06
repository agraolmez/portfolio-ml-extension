"""Read-only numerical/source review of the single-seed engineering evidence.

Writes a new review receipt once; never fits, replays, selects or overwrites.
The failed first two attempts remain failed and are charged to the ledger.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/annual_ind_seed1_review_20260929'
PREFIX = 'output/annual_ind_seed1_'


def read(path):
    return json.loads((ROOT/path).read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    started = time.monotonic()
    destination = OUT/'review.json'
    if destination.exists():
        raise FileExistsError(destination)
    checks = []
    first = PREFIX+'speed_20260929'
    final = PREFIX+'speed_v3_20260929'
    late = PREFIX+'late_profile_20260929'
    geometry_fields = ('theta','beta','geometry_beta','trace','profile_evaluations',
                       'solve_residual','final_training_value','status')
    temporal_fields = ('gains','beta','gradient','objective_trace','projected_gradient_trace',
                       'accepted_steps','evaluation_count','status','converged')
    def identical(a,b,fields,label):
        for key in fields:
            if a[key] != b[key]:
                raise ValueError(label+' changed '+key)
        checks.append(dict(name=label,passed=True))
    for ridge in (0,50,100):
        identical(read(f'{first}/geometry_r{ridge}_0.json'),
                  read(f'{first}/geometry_r{ridge}_1.json'),geometry_fields,f'geometry_ridge_{ridge}')
    for rho in ('0p1','0p01','0p001'):
        reference = read(f'{final}/temporal_reference_{rho}.json')
        assert reference['converged'] is True
        for ordinal in range(4):
            result = read(f'{final}/temporal_{ordinal}_{rho}.json')
            assert result['converged'] is True
            identical(reference,result,temporal_fields,f'temporal_{rho}_order_{ordinal}')
    identical(read(f'{final}/late_geometry_0.json'),read(f'{final}/late_geometry_1.json'),
              geometry_fields,'three_late_month_geometry')
    locks = [read(f'{p}/lock.json') for p in (first,PREFIX+'speed_v2_20260929',final,late)]
    source_checks = 0
    checked = {}
    def verify(rel,expected,snapshot=None):
        nonlocal source_checks
        path = ROOT/rel
        if checked.get(str(path)) == expected:
            return
        actual = digest(path)
        if actual != expected:
            if snapshot is None or not snapshot.is_file() or digest(snapshot) != expected:
                raise ValueError('Source hash mismatch: '+rel)
            path = snapshot
        checked[str(path)] = expected
        source_checks += 1
    for index, lock in enumerate(locks):
        for section in ('code_sha256','protected_sha256'):
            for rel,expected in lock[section].items():
                verify(rel,expected,ROOT/first/'source_snapshot'/rel if index == 0 else None)
    later = read(late+'/result.json')
    assert later['status'] == 'PASS_COMPLETE_2020_NEUTRAL_PROFILE_ONLY'
    assert later['training_months'] == 649
    assert later['newly_fitted_learned_models'] == later['new_portfolio_paths'] == 0
    assert later['complete_historical_runtime_established'] is False
    for rel,expected in later['source_sha256'].items():
        verify(rel,expected)
    rows = read(late+'/source_row_checks.json')
    assert rows['status'] == 'PASS' and len(rows['checks']) == 6
    assert all(row['identical'] for row in rows['checks'])
    cache = later['source_cache']
    assert cache['peak_retained_bytes'] <= cache['maximum_bytes'] == 256*2**20
    # Separate algebra check, using the saved full-history moments and oracle.
    import numpy as np
    with np.load(ROOT/late/'neutral_moments.npz',allow_pickle=False) as actual, np.load(
        ROOT/'data/prepared/step9_v1/baseline/coefficients/g0/2020_p512.npz',allow_pickle=False) as oracle:
        for index in (0,50,100):
            np.testing.assert_allclose(actual[f'beta_{index}'],oracle['coefficients'][:,index],atol=1e-9,rtol=1e-8)
    checks.extend([dict(name='649_month_neutral_coefficients_and_six_source_row_checks',passed=True),
                   dict(name='source_protected_and_execution_hashes',passed=True)])
    attempts = []
    for p in (first,PREFIX+'speed_v2_20260929',final,late):
        end = read(p+'/attempts/0001_end.json')
        samples = [json.loads(line) for line in (ROOT/p/'attempts/0001_resources.jsonl').read_text().splitlines()]
        record = dict(directory=p,status=end['status'],active_seconds=end['active_seconds'],
            maximum_process_gib=max(s['physical_footprint_gib'] for s in samples),
            minimum_available_gib=min(s['system_available_gib'] for s in samples),
            maximum_swap_growth_bytes=max(s['swap_growth_bytes'] for s in samples),
            maximum_new_disk_gib=max(s['study_disk_gib'] for s in samples))
        if p in (final,late):
            assert record['status'] == 'COMPLETE'
            assert record['maximum_process_gib'] <= 16 and record['minimum_available_gib'] >= 4
            assert record['maximum_swap_growth_bytes'] == 0 and record['maximum_new_disk_gib'] <= 2
        else:
            assert record['status'] == 'STOPPED'
        attempts.append(record)
    tests = (OUT/'unit_tests.log').read_text()
    assert 'Ran 43 tests' in tests and tests.rstrip().endswith('OK')
    checks.append(dict(name='43_unit_tests',passed=True))
    prior = 'output/annual_ind_parallel_review_20260929/acceptance_manifest.json'
    for rel,expected in read(prior)['artifact_sha256'].items():
        verify(rel,expected)
    checks.append(dict(name='preceding_accepted_checkpoint_intact',passed=True))
    events = [json.loads(line) for line in (ROOT/first/'progress.jsonl').read_text().splitlines()]
    geometry = [r for r in events if r['event'] == 'geometry_complete']
    ref = sum(r['seconds'] for r in geometry if not r['reuse'])
    reused = sum(r['seconds'] for r in geometry if r['reuse'])+read(first+'/neutral_build.json')['seconds']
    temporal = read(final+'/result.json')['temporal']
    reference = sum(r['seconds'] for r in temporal if not r['reuse'])/2
    cached = sum(r['seconds'] for r in temporal if r['reuse'])/2
    result = dict(status='PASS_SAVED_ENGINEERING_REVIEW_NOT_FULL_STUDY',
        reviewed_at_utc=datetime.now(timezone.utc).isoformat(),checks=checks,
        source_hashes_checked=source_checks,attempts=attempts,
        geometry_three_ridges=dict(reference_seconds=ref,reuse_plus_common_build_seconds=reused,
                                  seconds_reduction_percent=100*(1-reused/ref),paired_repeats=False),
        temporal_three_restraints=dict(reference_mean_seconds=reference,reuse_mean_seconds=cached,
                                      seconds_reduction_percent=100*(1-cached/reference),paired_repeats=True),
        full_2020_neutral_profile=later,complete_historical_runtime_established=False,
        full_study_started=False,new_performance_result=False,review_seconds=time.monotonic()-started)
    with destination.open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('full_2020_neutral_profile','checks')}))


if __name__ == '__main__':
    main()
