"""Read-only final verification of the completed Extension-3 historical study."""
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from temporal_study.report import verify_report_inputs


def main():
    base = ROOT / 'output/extension3_full_study_20260919'
    target = Path(sys.argv[1])
    assert not target.exists(), 'Receipts are append-only'
    cache = {}

    def read(path):
        return json.loads(path.read_text())

    def sha(path):
        path = path.resolve()
        if path not in cache:
            h = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(2**20), b''):
                    h.update(block)
            cache[path] = h.hexdigest()
        return cache[path]

    def inventory(values, root=ROOT):
        for rel, expected in values.items():
            assert sha(root / rel) == expected, f'Hash differs: {rel}'

    completion = read(base / 'completion_manifest.json')
    assert completion['numerical_historical_study_complete'] is True
    for key in ('artifact_sha256', 'consumed_inputs_sha256', 'parent_artifacts_sha256'):
        inventory(completion[key])
        print(f'Verified {key}: {len(completion[key])}', flush=True)
    lock = read(base / 'study_lock.json')
    inventory(lock['code_sha256'])
    inventory(lock['authority_sha256'])
    assert sha(base / 'study_lock.json') == completion['study_lock_sha256']
    assert sha(ROOT / 'config/extension3_full_run_approval_20260919.json') == completion['approval_sha256']
    assert sha(ROOT / completion['repair_path']) == completion['repair_sha256']
    layout = read(ROOT / 'config/project_layout.json')
    custom_count = 0
    for key in ('accepted_engineering', 'accepted_execution_milestones'):
        for entry in layout.get(key, []):
            evidence = read(ROOT / entry['manifest'])
            assert evidence['status'] == entry['status']
            assert evidence[entry['completion_key']] is True
            inventory(evidence[entry['checksum_key']])
            custom_count += len(evidence[entry['checksum_key']])
    fits = sorted((base / 'fits').glob('**/fit.json'))
    assert len(fits) == 765
    statuses = Counter()
    fit_keys = set()
    for path in fits:
        fit = read(path)
        spec = fit['fit_spec']
        fit_keys.add((spec['year'], spec['seed'], spec['kind'], spec['rho']))
        statuses[fit['status']] += 1
        assert fit['eligible_for_portfolio_evaluation'] is True
        assert fit['neutral_gate']['passed'] is True
        assert fit['solve_residual'] <= 1e-9
        assert fit['study_lock_sha256'] == completion['study_lock_sha256']
        assert fit['training_first'] == '1965-11-30'
        assert fit['training_last'] == f"{spec['year']-1}-11-30"
        assert fit['training_months'] == (spec['year']-1966)*12 + 1
        assert sha(path.parent / fit['model_file']) == fit['model_sha256']
        if spec['kind'] != 'fixed_temporal':
            assert fit['converged'] is True and fit['projected_gradient_norm'] <= 1e-5
        assert np.isfinite(fit['gains']).all()
        assert min(fit['gains']) >= 0 and max(fit['gains']) <= 2
    expected = {(y,s,'fixed_temporal',0.) for y in range(1981,2026) for s in (1,2,3)}
    expected |= {(y,s,k,r) for y in range(1981,2011) for s in (1,2,3)
                 for k in ('temporal','current_only') for r in (.1,.01,.001)}
    expected |= {(y,s,k,r) for y in range(2011,2026) for s in (1,2,3)
                 for k,r in [('temporal',.1),('current_only',.001)]}
    assert fit_keys == expected
    assert statuses == {'converged':630,'fixed_configuration_solved':135}
    monthly = pd.read_parquet(base / 'combined_monthly.parquet')
    assert len(monthly) == 9720 and not monthly.duplicated(['seed','method','decision_eom']).any()
    assert not monthly.holdings_reset.any()
    # Imported fixed1024 rows retain the accepted parent path's evidence rather
    # than duplicating its fit timestamps into this combined monthly table.
    own = monthly[~monthly.method.eq('fixed1024')]
    assert own.training_outcomes_available_at.notna().all()
    assert (own.training_outcomes_available_at <= own.decision_eom).all()
    assert monthly[monthly.method.eq('fixed1024')].training_outcomes_available_at.isna().all()
    dates = pd.date_range('1981-01-31', periods=540, freq='ME')
    continuity = 0
    for _, group in monthly.groupby(['seed','method']):
        group = group.sort_values('return_eom')
        assert np.array_equal(group.return_eom.to_numpy(), dates.to_numpy())
        assert np.array_equal(group.next_pretrade_sha256.to_numpy()[:-1], group.pretrade_sha256.to_numpy()[1:])
        continuity += len(group)-1
    parent = monthly[monthly.method.eq('parent')].sort_values(['seed','return_eom'])
    neutral = monthly[monthly.method.eq('neutral')].sort_values(['seed','return_eom'])
    numeric = ['gross_excess','net_excess','impact_cost','turnover','gross_exposure']
    assert np.array_equal(parent[numeric].to_numpy(),neutral[numeric].to_numpy())
    paths = read(base / 'selected_paths/paths_manifest.json')
    assert paths['neutral_maximum_position_error'] == 0
    assert paths['neutral_monthly_pairs_checked'] == 1620
    assert paths['archived_replay_checked'] and paths['archived_replay_maximum_position_error'] == 0
    summary = pd.read_csv(base / 'analysis/strategy_summary.csv')
    maximum = 0.
    periods = {'development':(1981,2000),'selection':(2001,2010),
               'evaluation':(2011,2020),'exposed_robustness':(2021,2025)}
    for row in summary.itertuples():
        lo,hi = periods[row.period]
        group = monthly[(monthly.seed==row.seed)&(monthly.method==row.method)
                        & monthly.return_eom.dt.year.between(lo,hi)]
        values = group.net_excess.to_numpy()
        assert len(group) == row.months
        calculated = {'annual_net_utility':12*(values.mean()-5*values.var(ddof=0)),
                      'annual_net_excess_mean':12*values.mean(),
                      'annual_net_volatility':np.sqrt(12*values.var(ddof=0)),
                      'annual_impact_bps':120000*group.impact_cost.mean()}
        for key,value in calculated.items():
            error = abs(value-getattr(row,key)); maximum=max(maximum,error)
            assert error < 1e-10, (row.period,row.method,key,error)
    source = read(base / 'report/report_source.json')
    mapping = {'result':source['analysis_result'],'completion_gates':source['completion_gates'],
               **{k:pd.read_csv(base/'analysis'/f'{k}.csv',float_precision='round_trip') for k in source['full_analysis_tables']}}
    diagnostics = pd.read_csv(base/'fit_diagnostics.csv',float_precision='round_trip')
    gains = pd.read_csv(base/'theme_gains.csv',float_precision='round_trip')
    verify_report_inputs(base/'report',mapping,fit_diagnostics=diagnostics,theme_gains=gains)
    assert all(source['completion_gates'].values())
    assert source['analysis_result']['all_conditions_met'] is False
    reference = read(base/'reference_verification/verification_manifest.json')
    assert reference['status']=='PASS' and reference['receipt']['bootstrap_utilities_checked']==720000
    result = {'status':'PASS','checked_at_utc':datetime.now(timezone.utc).isoformat(),
              'scope':'Independent completed-file integrity, coverage, convergence, continuity, metrics and report-source verification; no fitting or new selection',
              'unique_files_hashed':len(cache),'custom_inventory_entries':custom_count,
              'annual_fits':len(fits),'fit_statuses':dict(statuses),'paths':18,'months_per_path':540,
              'continuous_transitions_checked':continuity,'metric_groups_checked':len(summary),
              'chronology_rows_checked':len(own),
              'imported_fixed1024_chronology':'Retained accepted Extension-2 evidence; combined table has no duplicated fit timestamps',
              'maximum_recomputed_metric_error':maximum,'neutral_maximum_position_error':0.,
              'report_source_verified':True,'independent_R_verification':reference['receipt'],
              'completion_manifest_sha256':sha(base/'completion_manifest.json'),
              'review_script_sha256':sha(Path(__file__)),
              'visual_review_complete':False,'scientific_study_acceptance_complete':False}
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x') as handle:
        json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
