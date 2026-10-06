"""Verify completed optimization evidence; no historical fitting or portfolio run."""
import json,math
from pathlib import Path
from datetime import datetime,timezone
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from feature_geometry.history import sha
from temporal_fast_v4.settings import EXECUTION,environment_identity


def main():
    out=ROOT/'output/extension3_limit_20260919'
    names={'pilot':'pilot_selected01/benchmark_result.json','maximum':'max_history03/benchmark_result.json',
        'storage':'storage01/benchmark_result.json','candidates':'candidates01/benchmark_result.json',
        'sweep':'pilot_sweep01/benchmark_result.json','tests':'tests_final.json','integration':'integration_review_final.json','resources':'resource_checks/verification.json','resource_review':'resource_change_review.json'}
    values={k:json.loads((out/v).read_text()) for k,v in names.items()}
    for k,v in values.items():assert v['status']=='PASS',k
    for key in ('pilot','resources'):
        for name,expected in values[key]['code_sha256'].items():assert sha(ROOT/name)==expected,name
    allowed=values['resource_review']['permitted_post_benchmark_changes']
    for name,expected in values['maximum']['code_sha256'].items():
        assert sha(out/'max_history03/source_snapshot'/name)==expected,name
        if name in allowed:
            assert allowed[name]['before']==expected and allowed[name]['after']==sha(ROOT/name),name
        else:assert sha(ROOT/name)==expected,name
    assert values['resource_review']['numerical_equations_unchanged'] is True
    assert values['resources']['fixed_profile_calls']==12
    for key in ('storage','candidates','sweep'):
        for name,expected in values[key]['code_sha256'].items():
            assert sha(out/Path(names[key]).parent/'source_snapshot'/name)==expected,name
    for name,expected in values['integration']['source_sha256'].items():assert sha(ROOT/name)==expected,name
    pilot=values['pilot'];maximum=values['maximum'];storage=values['storage']
    assert pilot['profile_calls']==16 and maximum['profile_calls']==8 and storage['profile_calls']==30
    assert not pilot['gain_optimization_run'] and not maximum['historical_gain_optimization_run'] and not storage['historical_gain_optimization_run']
    assert all(not values[k]['portfolio_evaluation_run'] for k in ('pilot','maximum','storage'))
    progress=[json.loads(x) for x in (out/'pilot_selected01/progress.log').read_text().splitlines()]
    selected=[r for r in progress if r['event']=='profile_complete' and r['method']=='selected']
    learned=[r for r in selected if r['case']=='converged_pilot']
    assert {r['model_kind'] for r in learned}=={'temporal','current_only'}
    assert all(r['projected_gradient']<=1e-5 for r in learned)
    neutral=[r for r in pilot['equivalence'] if r['case']=='neutral' and r['method']=='selected']
    assert {r['kind'] for r in neutral}=={'temporal','current_only'}
    assert all(r['maximum_absolute_errors']['archived_beta']<=1e-11 for r in neutral)
    assert maximum['cache']['retained_months']==520 and maximum['cache']['disk_writes']==189
    assert maximum['panels']['retained_bytes']==0
    assert maximum['peak_rss_gib']<=24
    assert all(r['exact'] for r in maximum['reference_checks'])
    timings={(r['kind'],r['case']):r['methods']['selected']['median_seconds'] for r in pilot['measurements']}
    max_times={r['case']:r['methods']['selected']['median_seconds'] for r in maximum['measurements']}
    cache=json.loads((ROOT/'output/extension3_temporal_workflow_20260918/cache_feasibility/cache_feasibility.json').read_text())
    scenarios=[]
    for power in (1,2):
        denom=sum(r['stocks']**power for r in cache['monthly_header_inventory'][:181])
        end_growth=sum(r['stocks']**power for r in cache['monthly_header_inventory'])/denom
        # Residual at the measured longest window includes overflow and cache/headroom effects.
        # Do not apply the small-history speedup blindly to all 709 months.
        excess={case:max(0.,max_times['neutral' if case=='neutral' else 'nonneutral']-timings[('temporal',case)]*end_growth)
                for case in ('neutral','converged_pilot')}
        for excess_multiplier in (1,4):
            annual=[]
            for row in cache['annual_accounting']:
                n,year=row['training_months'],row['return_year']
                growth=sum(r['stocks']**power for r in cache['monthly_header_inventory'][:n])/denom
                overflow_fraction=max(0,n-520)/189
                def profile(kind,case):return timings[(kind,case)]*growth+excess[case]*overflow_fraction*excess_multiplier
                per_family=9 if year<=2010 else 3
                learned=per_family*sum(profile(kind,'neutral')+(calls-1)*profile(kind,'converged_pilot')
                    for kind,calls in (('temporal',20),('current_only',34)))
                fixed=3*.5*(profile('temporal','neutral')+profile('temporal','converged_pilot'))
                annual.append(dict(year=year,training_months=n,learned_fits=per_family*2,fixed_fits=3,
                    projected_seconds=learned+fixed))
            assert sum(r['learned_fits'] for r in annual)==630 and sum(r['fixed_fits'] for r in annual)==135
            scenarios.append(dict(stock_count_power=power,maximum_history_excess_multiplier=excess_multiplier,
                longest_window_excess_seconds=excess,fit_days=sum(r['projected_seconds'] for r in annual)/86400,annual=annual))
    projection=dict(status='CONDITIONAL_CAPACITY_PROJECTION_NOT_EXECUTION',created_at_utc=datetime.now(timezone.utc).isoformat(),
        annual_fit_count=765,learned_fit_count=630,fixed_fit_count=135,full_historical_run_authorized=False,
        assumed_profile_calls=dict(temporal=20,current_only=34),measured_pilot=pilot['measurements'],measured_maximum_history=maximum['measurements'],
        scenarios=scenarios,planning_days=[math.floor(min(r['fit_days'] for r in scenarios)+1),math.ceil(max(r['fit_days'] for r in scenarios)+2)],
        assumptions=['Fixed-profile timings are not later-year convergence measurements. Preserve pilot-based 20/34 call assumptions; actual fits can require more.',
        'Scale CPU work using observed stock counts to powers 1 and 2. Add the nonnegative measured maximum-history excess in proportion to spilled months.',
        'The two excess multipliers are sensitivity assumptions, not measured probabilities or confidence bounds. They include disk and smaller-reuse effects together.',
        'Only the temporal model has a measured 709-month timing. Use its excess for the current-only control, whose pilot timing is measured separately.',
        'Fixed smoothing retains two single-pass solves (neutral plus fixed); half the measured objective-plus-gradient time is a planning approximation.',
        'Keep the explicitly unmeasured 1–2-day allowance for complete paths, inference, validation, preparation and review; outward rounding is not a deadline.',
        'The 709-month timings precede stricter compressed-memory admission; the final policy can retain fewer intermediates. Source review, final actual-data cache/no-cache checks and full pilot tests verify this resource-only change. The timing remains a capacity estimate, not a final production benchmark.',
        'Continuous local availability is assumed. Competing work, memory pressure and machine sleep affect calendar time.'])
    environment=environment_identity()
    code=[*sorted((ROOT/'temporal_fast_v4').glob('*.py')),*sorted((ROOT/'temporal_fast_v4').glob('*.c')),
        *sorted((ROOT/'tests').glob('test_temporal_fast_v4*.py')),ROOT/'config/extension3_speed_v4.json',
        ROOT/'scripts/extension3_fast_v4_study.py',ROOT/'scripts/benchmark_extension3_limit.py',
        ROOT/'scripts/benchmark_extension3_storage_limit.py',ROOT/'scripts/benchmark_extension3_max_history.py',ROOT/'scripts/verify_extension3_resource_guards.py',Path(__file__).resolve()]
    for key in ('native_build','compact_build','source_checks_build'):code.extend(ROOT/name for name in environment[key]['artifact_sha256'])
    memory_rows=[json.loads(line) for line in (out/'max_history03/physical_memory_watch.jsonl').read_text().splitlines()]
    physical_peak=max(row['peak_physical_footprint_bytes'] for row in memory_rows)/2**30
    assert physical_peak<=24
    receipt=dict(status='VERIFIED_OPTIMIZATION_ONLY',created_at_utc=datetime.now(timezone.utc).isoformat(),execution_settings=dict(EXECUTION),
        execution_environment=environment,full_pilot_profiles_passed=True,both_neutral_coefficients_match_parent=True,
        both_saved_pilots_still_converged=True,maximum_history_profiles_passed=True,disk_and_reference_equivalence_passed=True,
        gain_optimization_run=False,portfolio_evaluation_run=False,full_historical_run_authorized=False,
        gain_optimization_scope='Only synthetic optimizer equivalence tests; historical benchmark gain vectors are fixed.',
        numeric_tolerances=pilot['numeric_tolerances'],resource_limits_changed=False,additional_persistent_data_cache_bytes=0,
        tests={k:v for k,v in values['tests'].items() if k!='log'},
        maximum_history_peak_physical_footprint_gib=physical_peak,
        maximum_history_timing_precedes_stricter_memory_admission=True,
        peak_benchmark_rss_gib=max(pilot['peak_rss_gib'],maximum['peak_rss_gib'],storage['peak_rss_gib']),
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in code},
        evidence_sha256={str((out/name).relative_to(ROOT)):sha(out/name) for name in names.values()})
    for name,value in [('runtime_projection.json',projection),('speed_verification.json',receipt)]:
        with (out/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=receipt['status'],planning_days=projection['planning_days'],fit_days=[r['fit_days'] for r in scenarios]),indent=2))

if __name__=='__main__':main()
