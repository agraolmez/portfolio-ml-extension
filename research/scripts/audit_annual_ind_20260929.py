"""Read-only source audit and historical-selector regression; no new fitting."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import psutil
from annual_ind.selection import author_objective, select_prior_history
from baseline.full import preflight
from feature_geometry.history import sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT/'output') or out.exists():
        raise ValueError('A fresh project output directory is required')
    out.mkdir(parents=True)
    started = time.monotonic()
    bindings = {}
    inventory = {}
    for rel in ['output/step3_author_conventions_20260911/bundle_checksums.json',
                'output/step4_20260911/acceptance_manifest.json',
                'output/step9_20260913/acceptance_manifest.json']:
        bindings[rel] = sha(ROOT/rel)
        receipt = json.loads((ROOT/rel).read_text())
        if 'gates' in receipt and not all(g['passed'] is True for g in receipt['gates']):
            raise ValueError('Source acceptance gate failed')
        inventory.update(receipt.get('artifact_sha256', receipt))

    def accepted(rel):
        path = ROOT/rel
        canonical = str(path.resolve().relative_to(ROOT))
        actual = sha(path)
        if inventory.get(rel, inventory.get(canonical)) != actual:
            raise ValueError('Missing/changed accepted source '+rel)
        bindings[rel] = actual
        return path

    print('Checking accepted financial inputs and protected decisions.', flush=True)
    fresh, _ = preflight()
    write(out/'financial_preflight.json', fresh)
    preparation = ROOT/'output/step6_20260911/preparation_manifest.json'
    if sha(preparation) != fresh['preparation_manifest_sha256']:
        raise ValueError('Financial preflight preparation identity differs')
    inventory.update(json.loads(preparation.read_text())['artifact_sha256'])
    bindings[str(preparation.relative_to(ROOT))] = sha(preparation)
    selection_path = accepted('data/prepared/step9_v1/baseline/annual_selection.parquet')
    scores_path = accepted('data/prepared/step9_v1/baseline/validation_scores.parquet')
    selections = pd.read_parquet(selection_path)
    scores = pd.read_parquet(scores_path)
    scores['candidate'] = (scores.g_index.astype(str)+'/'+scores.p.astype(str)+'/'+
                           scores.lambda_index.astype(str))
    scores['model_available_at'] = (scores.fit_year-1).astype(str)+'-12-31'
    order = list(scores[['g_index','p','lambda_index','candidate']].drop_duplicates().sort_values(
        ['g_index','p','lambda_index']).candidate)
    exact_matches = []
    for year in range(1981, 2021):
        choice = select_prior_history(scores, year=year, candidate_order=order,
                                      first_return='1971-01-31', tie_tolerance=0.)
        source = selections[selections.return_eom.eq(pd.Timestamp(year-1,12,31))].iloc[0]
        expected = f'{int(source.g_index)}/{int(source.p)}/{int(source.lambda_index)}'
        if choice['candidate'] != expected:
            raise ValueError('New selector differs from accepted original annual selection')
        exact_matches.append(dict(year=year, candidate=expected, months=choice['validation_months']))
    write(out/'baseline_selection_regression.json', dict(status='PASS',
        candidate_count=len(order), matched_years=exact_matches,
        validation_score='released author r_tilde beta - beta denom beta/2',
        actual_portfolio_utility_used_for_selection=False))
    print('All 40 baseline annual selections reproduced from the archived score table.', flush=True)
    months = pd.date_range('1965-11-30','2020-11-30',freq='ME')
    matched_ids = 0
    for date in months:
        key = str(date.date())
        matrix = accepted(f'data/prepared/step9_v1/baseline/matrices/{key}.npz')
        risk = accepted(f'data/prepared/step9_v1/baseline/risk/{key}.npz')
        with np.load(matrix, allow_pickle=False) as m, np.load(risk, allow_pickle=False) as r:
            if not np.array_equal(m['ids'], r['ids']) or not np.isfinite(m['m']).all():
                raise ValueError('Risk/matrix ID or finite-matrix check failed')
            if r['loadings'].shape[1] != 25 or not np.isfinite(r['factor_cov']).all():
                raise ValueError('Missing original 25-factor risk representation')
            matched_ids += len(m['ids'])
    print('All required monthly matrix/risk files are present and hash-verified.', flush=True)
    raw_path = accepted('data/prepared/step3_author_v3/source_characteristics.parquet')
    ranked_path = accepted('data/prepared/step3_author_v3/monthly_inputs.parquet')
    early_filters = [('eom','>=',pd.Timestamp('1964-05-31')),('eom','<=',pd.Timestamp('1980-11-30'))]
    names = json.loads(accepted('output/step2_20260910/source_settings.json').read_text())['features']
    raw = pd.read_parquet(raw_path, columns=['id','eom',*names], filters=early_filters)
    ranked = pd.read_parquet(ranked_path, columns=['id','eom','valid',*names], filters=early_filters)
    for frame in (raw, ranked):
        if frame.duplicated(['id','eom']).any():
            raise ValueError('Duplicate early source key')
    joined = ranked[['id','eom']].merge(raw[['id','eom']], how='left',
                                      on=['id','eom'], indicator=True, validate='one_to_one')
    if not joined['_merge'].eq('both').all() or not np.isfinite(ranked[names].to_numpy()).all():
        raise ValueError('Missing raw observation mask or nonfinite prepared input')
    early = dict(first_month=str(ranked.eom.min().date()), last_month=str(ranked.eom.max().date()),
        rows=len(ranked), calendar_months=ranked.eom.nunique(), characteristics=len(names),
        source_mask_join_complete=True, ranked_characteristics_finite=True,
        raw_missing_fraction=float(raw[names].isna().to_numpy().mean()))
    # Check executable formula against one actual archived monthly quadratic score.
    coefficient = accepted('data/prepared/step9_v1/baseline/coefficients/g0/1981_p512.npz')
    moment = accepted('data/prepared/step9_v1/baseline/moments/g0/1980-12-31.npz')
    with np.load(coefficient) as c, np.load(moment) as m:
        idx = c['indices']; beta = c['coefficients']
        computed = author_objective(m['r_tilde'][idx], m['denom'][np.ix_(idx,idx)], beta)
    stored = scores[(scores.g_index == 0) & (scores.p == 512) &
                    scores.return_eom.eq(pd.Timestamp('1981-01-31'))].sort_values('lambda_index')
    np.testing.assert_allclose(computed, stored.objective, atol=1e-12, rtol=1e-11)
    formula_error = float(np.max(np.abs(computed-stored.objective.to_numpy())))
    for rel in ['references/author_code/0 - Portfolio choice functions.R',
                'references/author_code/Main.R', 'baseline/engine.py',
                'annual_ind/selection.py', 'scripts/audit_annual_ind_20260929.py']:
        bindings[rel] = sha(ROOT/rel)
    result = dict(status='AUDIT_AND_SELECTOR_REGRESSION_PASS_NOT_A_HISTORICAL_RUN',
        at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
        monthly_matrix_risk_pairs=len(months), summed_security_month_ids=matched_ids,
        early_characteristic_sources=early,
        baseline_selection_first_return=str(selections.return_eom.min().date()),
        baseline_validation_first_return=str(scores.return_eom.min().date()),
        original_1981_2020_annual_choices_reproduced=len(exact_matches),
        actual_month_author_formula_max_error=formula_error,
        learned_parent_first_return_year=1981,
        missing_learned_parent_years=list(range(1971,1981)),
        unresolved_first_parent='No 1970 selection exists for an annual-selected 1971 E2 parent',
        new_fits=0, new_portfolio_paths=0, new_significance_tests=0,
        available_gib=psutil.virtual_memory().available/2**30,
        swap_used_gib=psutil.swap_memory().used/2**30,
        free_disk_gib=shutil.disk_usage(ROOT).free/2**30,
        input_and_code_sha256=bindings)
    write(out/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='input_and_code_sha256'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
