"""Seal existing bounded benchmark evidence and a conditional capacity estimate.

No gain fitting, portfolio evaluation, or full-run authorization occurs here.
"""
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from feature_geometry.history import sha
from temporal_fast_v3.settings import EXECUTION, environment_identity


def main():
    output = ROOT / 'output/extension3_speed3_20260919'
    paths = dict(
        benchmark=output/'pilot_profiles01/benchmark_result.json',
        paired=output/'paired01/benchmark_result.json',
        progress=output/'pilot_profiles01/progress.log',
        tests=output/'tests_final.json',
        cache=ROOT/'output/extension3_temporal_workflow_20260918/cache_feasibility/cache_feasibility.json',
        previous_projection=ROOT/'output/extension3_speed_20260919/runtime_projection.json',
        integration=output/'integration_review.json',
    )
    for name in ('runtime_projection.json', 'speed_verification.json'):
        if (output/name).exists():
            raise FileExistsError('Preserve existing evidence: '+name)
    b = json.loads(paths['benchmark'].read_text())
    paired = json.loads(paths['paired'].read_text())
    cache = json.loads(paths['cache'].read_text())
    old = json.loads(paths['previous_projection'].read_text())
    environment = b['execution_environment']
    tests = json.loads(paths['tests'].read_text())
    integration = json.loads(paths['integration'].read_text())
    assert b['status'] == paired['status'] == tests['status'] == integration['status'] == 'PASS'
    assert b['scope'] == 'pilot_profiles' and b['profile_calls'] == 16
    assert b['gain_optimization_run'] is False and b['portfolio_evaluation_run'] is False
    assert b['numeric_tolerances'] == dict(atol=1e-11, rtol=1e-10)
    assert environment == environment_identity()
    for name, expected in b['code_sha256'].items():
        assert sha(ROOT/name) == expected, name
    # The eight-month run has its own earlier benchmark script snapshot.
    for name, expected in paired['code_sha256'].items():
        assert sha(output/'paired01/source_snapshot'/name) == expected, name
    for name, expected in integration['source_sha256'].items():
        assert sha(ROOT/name) == expected, name
    rows = [json.loads(line) for line in paths['progress'].read_text().splitlines()]
    calls = [r for r in rows if r['event'] == 'profile_complete']
    assert len(calls) == 16
    optimized = [r for r in calls if r['method'] == 'both']
    converged = [r for r in optimized if r['case'] == 'converged_pilot']
    assert {r['model_kind'] for r in converged} == {'temporal', 'current_only'}
    assert all(r['projected_gradient'] <= 1e-5 for r in converged)
    neutral = [r for r in b['equivalence'] if r['case'] == 'neutral'
               and r['method'] == 'both']
    assert {r['kind'] for r in neutral} == {'temporal', 'current_only'}
    assert all(r['maximum_absolute_errors']['archived_beta'] <= 1e-11 for r in neutral)
    timing = {(r['kind'], r['case']): r['methods']['both']['median_seconds']
              for r in b['measurements']}
    scenarios = []
    # Counts and history shapes are unchanged from the concrete full-study proposal.
    for power in (1, 2):
        denominator = sum(r['stocks']**power for r in cache['monthly_header_inventory'][:181])
        for disk_multiplier in (1, 4):
            annual = []
            for row in cache['annual_accounting']:
                n, year = row['training_months'], row['return_year']
                learned_per_family = 9 if year <= 2010 else 3
                growth = sum(r['stocks']**power for r in cache['monthly_header_inventory'][:n])/denominator
                learned_seconds = learned_per_family * sum(
                    timing[(kind, 'neutral')] + (calls-1)*timing[(kind, 'converged_pilot')]
                    for kind, calls in (('temporal', 20), ('current_only', 34))) * growth
                # Fixed smoothing does TWO single-pass solves: neutral and fixed.
                fixed_seconds = 3*.5*(timing[('temporal', 'neutral')]
                                      + timing[('temporal', 'converged_pilot')])*growth
                overflow = max(0, n-520)
                disk_seconds = (learned_per_family*(20+34)*2 + 3*2)*overflow*old[
                    'measured_disk_read_seconds']*disk_multiplier
                annual.append(dict(year=year, training_months=n,
                    learned_fits=learned_per_family*2, fixed_fits=3,
                    cpu_seconds=learned_seconds+fixed_seconds, disk_seconds=disk_seconds,
                    projected_hours=(learned_seconds+fixed_seconds+disk_seconds)/3600))
            assert sum(r['learned_fits'] for r in annual) == 630
            assert sum(r['fixed_fits'] for r in annual) == 135
            scenarios.append(dict(stock_count_power=power, disk_read_multiplier=disk_multiplier,
                fit_days=sum(r['projected_hours'] for r in annual)/24, annual=annual))
    projection = dict(status='CONDITIONAL_CAPACITY_PROJECTION_NOT_EXECUTION',
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        full_historical_run_authorized=False, historical_performance_read=False,
        annual_fit_count=765, learned_fit_count=630, fixed_fit_count=135,
        measured_profiles=b['measurements'], assumed_profile_calls=dict(temporal=20, current_only=34),
        measured_disk_read_seconds=old['measured_disk_read_seconds'], scenarios=scenarios,
        planning_days=[math.floor(min(r['fit_days'] for r in scenarios)+1),math.ceil(max(r['fit_days'] for r in scenarios)+2)],
        planning_basis='Measured fit scenarios plus an unmeasured 1–2-day allowance, rounded outward; not a confidence interval or deadline.',
        assumptions=[
            'One neutral evaluation plus 19 temporal or 33 current-only nonneutral evaluations, as in the saved pilot; later years/seeds/penalties may need more.',
            'Stock-count workload scales between N and N-squared across the complete expanding history.',
            'CPU speedup does not accelerate overflow disk reads; charge two reads per spilled month per learned profile and two single passes per fixed control.',
            'Retain the corrected two single-pass coefficient solves for every fixed-smoothing control.',
            'The 181-month measurement is warm RAM; it does not measure a 709-month end-to-end fit or maximum-history peak memory.',
            'The 1–2-day planning allowance covers preparation, paths, validation, inference and review; these stages are not timed by this benchmark.',
            'Continuous availability is assumed; sleeping or competing workloads extend calendar time.',
        ], evidence_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths.values()})
    (output/'runtime_projection.json').write_text(json.dumps(projection, indent=2)+'\n')
    code = sorted((ROOT/'temporal_fast_v3').glob('*.py')) + sorted((ROOT/'temporal_fast_v2').glob('*.py')) + sorted((ROOT/'temporal_fast').glob('*.py')) + [
        ROOT/'temporal_fast_v3/kernels.c', *[ROOT/name for name in environment['native_build']['artifact_sha256']],
        ROOT/'scripts/benchmark_extension3_fast_v3.py', ROOT/'scripts/profile_extension3_v2_remaining.py',
        ROOT/'scripts/extension3_fast_v3_study.py', ROOT/'scripts/summarize_extension3_speed_v3.py',
        ROOT/'config/extension3_speed_v3.json',
        *sorted((ROOT/'tests').glob('test_temporal_fast_v3*.py'))]
    receipt = dict(status='VERIFIED_OPTIMIZATION_ONLY',
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        execution_settings=dict(EXECUTION), execution_environment=environment,
        full_pilot_profiles_passed=True, both_neutral_coefficients_match_parent=True,
        both_saved_pilots_still_converged=True, portfolio_evaluation_run=False,
        gain_optimization_run=False, gain_optimization_scope='No historical optimization; synthetic optimizer equivalence tests did run.',
        full_historical_run_authorized=False, numeric_tolerances=b['numeric_tolerances'],
        optimized_profile_results=optimized,
        neutral_maximum_absolute_beta_error=max(r['maximum_absolute_errors']['archived_beta'] for r in neutral),
        parent_neutral_features_exact=True, neutral_operators_identical_to_previous=True,
        reference_engine='temporal_fast_v2', additional_persistent_cache_bytes=0, resource_limits_changed=False,
        neutral_qualification='Feature transformation is exact; archived coefficient agreement is to the unchanged tolerance, not bitwise identity under explicit native BLAS policy.',
        tests={k:v for k,v in tests.items() if k != 'log'}, peak_benchmark_rss_gib=b['peak_rss_gib'],
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in code},
        evidence_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths.values()})
    (output/'speed_verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(status=receipt['status'], planning_days=projection['planning_days'],
                         fit_days=[r['fit_days'] for r in scenarios]), indent=2))


if __name__ == '__main__':
    main()
