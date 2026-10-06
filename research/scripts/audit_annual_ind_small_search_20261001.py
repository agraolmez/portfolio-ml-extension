"""Read-only fit inventory and runtime scenarios; never trains or selects models."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from temporal_extension.model import TemporalModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = {}

    def digest(rel):
        if rel not in sources:
            h = hashlib.sha256()
            with (ROOT / rel).open('rb') as f:
                for chunk in iter(lambda: f.read(2**20), b''):
                    h.update(chunk)
            sources[rel] = h.hexdigest()
        return sources[rel]

    def read(rel):
        digest(rel)
        return json.loads((ROOT / rel).read_text())

    roots = {
        'E2': 'output/feature_geometry_study_20260914',
        'E3': 'output/extension3_full_study_20260919',
        'C4': 'output/extension4_full_study_20260924',
    }
    inventories = {k: read(v + '/acceptance_manifest.json')['artifact_sha256']
                   for k, v in roots.items()}

    def verify(kind, rel):
        if digest(rel) != inventories[kind][rel]:
            raise ValueError('Accepted artifact changed: ' + rel)

    selection_rel = 'data/prepared/step9_v1/baseline/annual_selection.parquet'
    digest(selection_rel)
    selection = pd.read_parquet(ROOT / selection_rel)
    rows, times, conditional = [], [], []
    for year in range(1971, 2021):
        cutoff = f'{year-1}-12-31'
        selected = selection.loc[selection.return_eom.eq(cutoff)]
        if year == 1971:
            assert selected.empty
            row = dict(year=year, width=None, gi=None, ridge_index=None,
                       startup='UNRESOLVED; pilot coordinate not assumed')
        else:
            assert len(selected) == 1
            s = selected.iloc[0]
            row = dict(year=year, width=int(s.p), gi=int(s.g_index),
                       ridge_index=int(s.lambda_index), ridge=float(s['lambda']))
        row.update(scale_reusable=False, temporal_reusable=[],
                   conditional_moments_reusable=[], existing_C4_coefficients=0)
        e2 = f'{roots["E2"]}/fits/{year}/seed1/full'
        if (ROOT / e2 / 'fit.json').is_file():
            for fn in ['fit.json', 'model.npz']:
                verify('E2', e2 + '/' + fn)
            receipt = read(e2 + '/fit.json')
            assert receipt['bandwidth_index'] == row['gi']
            assert receipt['coefficient_ridge'] == row['ridge']
            assert receipt['cutoff'] == cutoff
            assert receipt['training_last'] == f'{year-1}-11-30'
            with np.load(ROOT / e2 / 'model.npz', allow_pickle=False) as z:
                frequencies, theta = z['frequencies'], z['theta']
                assert frequencies.shape == (115, row['width']//2)
            row['scale_reusable'] = True
            times.append(dict(layer='E2', year=year, seconds=receipt['seconds'],
                              evaluations=receipt['profile_evaluations']))
            temporal_models = {}
            for rho, name in [(.1, '0p1'), (.01, '0p01'), (.001, '0p001')]:
                folder = f'{roots["E3"]}/fits/{year}/seed1/temporal/rho_{name}'
                if not (ROOT / folder / 'fit.json').is_file():
                    continue
                for fn in ['fit.json', 'model.npz', 'fit_commit.json']:
                    verify('E3', folder + '/' + fn)
                fit = read(folder + '/fit.json')
                assert fit['parent_artifacts_sha256'][e2+'/model.npz'] == digest(e2+'/model.npz')
                assert fit['parent_settings']['ridge'] == row['ridge']
                assert fit['parent_settings']['cutoff'] == cutoff
                assert fit['training_last'] == f'{year-1}-11-30'
                assert fit['converged'] and fit['eligible_for_portfolio_evaluation']
                assert fit['neutral_gate']['passed'] and fit['rho'] == rho
                model = TemporalModel.load(ROOT / folder / 'model.npz',
                                           expected_parent_fingerprint=fit['parent_fingerprint'])
                np.testing.assert_array_equal(model.feature_map.parent.frequencies, frequencies)
                np.testing.assert_array_equal(model.feature_map.theta, theta)
                assert model.cutoff == cutoff
                temporal_models[rho] = model
                row['temporal_reusable'].append(rho)
                times.append(dict(layer='E3', year=year, rho=rho,
                                  seconds=fit['elapsed_seconds'], evaluations=fit['evaluation_count']))
            c4 = f'{roots["C4"]}/fits/{year}/seed1'
            for fn in ['fit.json', 'moments.npz']:
                verify('C4', c4 + '/' + fn)
            fit = read(c4 + '/fit.json')
            assert fit['context']['parent'] == temporal_models[.1]._content_hash()
            assert fit['cutoff'] == cutoff and fit['parent_ridge'] == row['ridge']
            for item in fit['models']:
                if item['method'] == 'C4':
                    verify('C4', c4 + '/' + item['model_file'])
                    row['existing_C4_coefficients'] += 1
            row['conditional_moments_reusable'].append(.1)
            conditional.append(dict(year=year, seconds=fit['seconds'],
                                    month_seconds=sum(fit['month_seconds'])))
        rows.append(row)

    pilot = read('output/annual_ind_runner_pilot_v5_20261001/result.json')
    checkpoints = {c['key']: c['context'] for c in pilot['checkpoints']}
    late = checkpoints['2020/p512/g0/r50/rho0.1/E3']['seconds']
    early = checkpoints['1971/p512/g0/r50/rho0.1/E3']['seconds']
    # Conservative scenarios, NOT confidence intervals: long-window pilot rate,
    # observed weak-penalty evaluation counts, and deliberately no CPU speedup multiplier.
    # Thirty early temporal fits each use the maximum observed 1981-90 seed-1 time.
    # Twenty late missing fits each use 20 evaluations at the pilot's 14-evaluation rate.
    early_max = max(t['seconds'] for t in times if t['layer']=='E3' and t['year']<=1990)
    early_hours = 30 * early_max / 3600
    late_hours = 20 * 20 * late / 14 / 3600
    c4_hours = (2 * sum(t['seconds'] for t in conditional) +
                30 * max(t['seconds'] for t in conditional if t['year']<=1983)) / 3600
    counts = dict(scale_total=50, scale_existing=sum(r['scale_reusable'] for r in rows),
                  temporal_total=150, temporal_existing=sum(len(r['temporal_reusable']) for r in rows),
                  C4_groups_total=150,
                  C4_moment_groups_existing=sum(len(r['conditional_moments_reusable']) for r in rows),
                  C4_coefficients_total=450,
                  C4_coefficients_existing=sum(r['existing_C4_coefficients'] for r in rows))
    estimate = dict(
        status='ASSESSMENT_COMPLETE_NOT_RUN_AUTHORIZATION', at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Seed 1; one past-selected baseline coordinate/year; 3 temporal x 3 conditional settings',
        counts=counts, temporal_new_early=30, temporal_new_late=20,
        layers_verified='Archived file hashes, annual base selection, cutoffs, exact frozen parent arrays and C4 parent identity',
        reuse_execution_verified=False,
        reuse_limits='New candidate adapter, source/state equivalence, annual scoring and continuous replay still require integrated checks; no old selected-setting dates may be imported.',
        narrow_parent_years=[r for r in rows if r['width']==64],
        startup_1971_resolved=False,
        timing_basis=dict(early_pilot_temporal_seconds=early, late_pilot_temporal_seconds=late,
                          late_pilot_evaluations=14, assumed_missing_late_evaluations=20,
                          early_historical_max_seconds=early_max,
                          temporal_early_hours=early_hours, temporal_late_hours=late_hours,
                          conditional_groups_hours=c4_hours, scale_and_preparation_allowance_hours=1.,
                          validation_replay_review_allowance_hours=4.,
                          serial_scenario_hours=early_hours+late_hours+c4_hours+5.,
                          rounded_planning_range_active_hours=[24,48],
                          unmeasured_allowances=True, probability_bound=False,
                          assumptions='Reuse passes; cache retained within 28 GiB cap; convergence similar; no host interruptions. Timing coordinates differ from annual winners; no compounded speedups.'),
        historical_experiment_started=False, new_financial_results=False,
        pending=['1971 initial baseline coordinate or later validation start',
                 'Freeze annual candidate scoring rule and candidate-only reuse amendment',
                 'Approve full-run active-time and storage envelope after integration'])
    for name, data in [('assessment.json', estimate), ('inventory.json', rows),
                       ('timings.json', dict(layers=times, conditional_groups=conditional)),
                       ('source_sha256.json', sources)]:
        (out / name).write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(dict(counts=counts, timing=estimate['timing_basis'],
                          verified_source_files=len(sources)), indent=2))


if __name__ == '__main__':
    main()
