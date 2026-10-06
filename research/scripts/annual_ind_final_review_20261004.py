"""Read-only independent review of the completed staged study; never fits models.

By default print the findings. --write saves a separate review, never the study.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sqlite3
import statistics as st

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'output/annual_ind_staged_20261001'
OUT = ROOT / 'output/annual_ind_staged_final_review_20261004'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def review():
    checks = []
    def passed(name, **details):
        checks.append(dict(check=name, passed=True, **details))
    result = read(STUDY/'result.json')
    manifest = read(STUDY/'numerical_completion_manifest.json')
    for rel, digest in manifest['artifact_sha256'].items():
        require(sha(ROOT/rel) == digest, 'Changed numerical artifact: '+rel)
    passed('Original numerical artifact hashes', files=len(manifest['artifact_sha256']))
    lock = read(STUDY/'lock.json')
    identity = sha(STUDY/'lock.json')
    require(identity == result['lock_sha256'], 'Different source lock')
    for group in ('code_sha256', 'source_manifest_sha256', 'required_checks_sha256'):
        for rel, digest in lock[group].items():
            require(sha(ROOT/rel) == digest, 'Changed locked source: '+rel)
    require(sha(ROOT/'config/annual_ind_staged_20261001.json') == lock['configuration_sha256'], 'Changed configuration')
    passed('Locked code, configuration, required checks and source manifests')

    counts = Counter()
    with sqlite3.connect('file:'+str(STUDY/'models.sqlite')+'?mode=ro', uri=True) as c:
        require(c.execute('SELECT identity FROM run').fetchall() == [(identity,)], 'Checkpoint identity')
        keys = set()
        for key, context, payload, digest in c.execute('SELECT key,context,payload,digest FROM bundles'):
            require(hashlib.sha256(identity.encode()+b'\0'+context.encode()+b'\0'+payload).hexdigest() == digest, 'Bundle hash: '+key)
            meta = json.loads(context)
            year = int(key[:4]); layer = key.split('/')[-1]; keys.add(key); counts[layer] += 1
            require(meta['geometry_settings']['cutoff'] == f'{year-1}-12-31', 'Fit chronology: '+key)
            require(meta['geometry_settings']['max_steps'] == 8, 'Changed scale update budget')
            with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
                require(all(np.isfinite(arrays[n]).all() for n in arrays.files), 'Nonfinite fit: '+key)
                if layer == 'E3':
                    require(meta['converged'] and meta['neutral_gate']['passed'], 'Temporal gate: '+key)
                    require(np.all((arrays['gains'] >= 0) & (arrays['gains'] <= 2)), 'Gain bounds: '+key)
                if layer == 'C4':
                    require(meta['independent']['passed'], 'Independent coefficient solve: '+key)
        expected = {f'{y}/E2' for y in range(1971,2026)} | {f'{y}/rho{rho}/{layer}' for y in range(1971,2026) for rho in ('0.1','0.01','0.001') for layer in ('E3','C4')}
        require(keys == expected, 'Incomplete annual candidate banks')
    passed('All model bundle hashes, numerical finiteness and saved fit gates', bundles=sum(counts.values()), layers=dict(counts), scale_note='Fixed eight-update budget, not a convergence claim')

    scores = pd.read_parquet(STUDY/'validation_scores.parquet')
    require(len(scores) == 5940 and not scores.duplicated(['candidate','return_eom']).any(), 'Validation score calendar')
    require(np.isfinite(scores.objective).all(), 'Nonfinite validation objective')
    for y in range(1971,2026):
        receipt = read(STUDY/f'validation/{y}/verification.json')
        require(receipt == dict(candidates=9, direct_economic_accounting=True, months=12, passed=True), 'Validation accounting gate')
        v = scores[scores.fit_year.eq(y)]
        require(len(v) == 108 and set(v.model_available_at) == {f'{y-1}-12-31'}, 'Validation fit cutoff')
        expected_dates = list(pd.date_range(f'{y}-01-31', f'{y}-12-31', freq='ME').strftime('%Y-%m-%d'))
        for _, candidate_scores in v.groupby('candidate'):
            require(sorted(candidate_scores.return_eom) == expected_dates, 'Missing validation month')
    selections = read(STUDY/'selections.json')
    require(set(selections) == {str(y) for y in range(1981,2026)}, 'Selection years')
    winners = Counter()
    for ys, saved in selections.items():
        y = int(ys); cutoff = f'{y-1}-12-31'
        v = scores[(scores.return_eom >= '1971-01-31') & (scores.return_eom <= cutoff)]
        recomputed = {k: st.mean(g.objective.tolist()) for k,g in v.groupby('candidate')}
        order = [f'rho{r}_m{m}' for r in ('0.1','0.01','0.001') for m in (100,10,1)]
        require(saved['tie_order'] == order and saved['tie_tolerance'] == 0, 'Tie rule')
        require(saved['selection_available_at'] == cutoff and saved['last_validation_return'] == cutoff, 'Selection cutoff')
        require(saved['validation_months'] == (y-1971)*12, 'Selection window')
        for k,value in recomputed.items():
            require(math.isclose(value,saved['candidate_scores'][k],rel_tol=1e-11,abs_tol=1e-13), 'Selection score')
        winner = max(order, key=lambda k: recomputed[k])
        require(winner == saved['candidate'], 'Different selected model')
        winners[winner] += 1
    passed('Independent reconstruction of past-only annual selections', years=45, validation_scores=5940, selected_counts=dict(winners))

    frame = pd.read_parquet(STUDY/'monthly.parquet')
    require(len(frame) == 1620 and set(frame.method) == {'B1','C4','IND'} and set(frame.seed) == {1}, 'Portfolio scope')
    require(not frame.holdings_reset.any(), 'Unexpected holdings reset')
    expected = list(pd.date_range('1981-01-31','2025-12-31',freq='ME').strftime('%Y-%m-%d'))
    for method, v in frame.groupby('method'):
        v = v.sort_values('return_eom')
        require(list(v.return_eom) == expected, 'Missing/duplicate portfolio month')
        np.testing.assert_allclose(v.net_excess, v.gross_excess-v.impact_cost, atol=1e-14, rtol=0)
        require(list(v.next_pretrade_sha256)[:-1] == list(v.pretrade_sha256)[1:], 'Holdings fingerprint chain')
    previous_hash = None; previous_arrays = None; monthly_rows = []
    days = list(pd.date_range('1980-12-31','2025-11-30',freq='ME').strftime('%Y-%m-%d'))
    for day in days:
        folder = STUDY/'paths'/day; commit = read(folder/'commit.json')
        require(commit['status'] == 'COMPLETE' and commit['context']['day'] == day and commit['context']['lock'] == identity, 'Path context')
        if previous_hash:
            require(commit['context']['previous'] == previous_hash, 'Path commit chain')
        for name,digest in commit['artifact_sha256'].items():
            require(sha(folder/name) == digest, 'Path content hash')
        with np.load(folder/'positions.npz',allow_pickle=False) as a:
            require(list(a['methods']) == ['B1','C4','IND'] and list(a['seeds']) == [1,1,1], 'Path method ordering')
            if previous_arrays:
                require(np.array_equal(a['ids'],previous_arrays[0]) and np.array_equal(a['pretrade'],previous_arrays[1]), 'Portfolio continuity')
            np.testing.assert_allclose(a['trade'], a['weight']-a['pretrade'],atol=1e-14,rtol=1e-12)
            previous_arrays = (a['next_ids'].copy(), a['next_pretrade'].copy())
        monthly_rows.extend(read(folder/'monthly.json'))
        previous_hash = sha(folder/'commit.json')
    order = ['return_eom','method']
    pd.testing.assert_frame_equal(frame.sort_values(order).reset_index(drop=True),pd.DataFrame(monthly_rows)[frame.columns].sort_values(order).reset_index(drop=True),check_dtype=False,check_exact=True)
    passed('All saved portfolio paths: hashes, continuity, trade identities and return aggregation', months=540, portfolios=3, first_month_initialization='Original runtime verification retained; all 539 subsequent transitions rechecked')

    comparisons = []
    for saved in result['comparisons']:
        views = {}
        for method in ('B1','IND'):
            v = frame[(frame.method == method) & (frame.return_eom >= f"{saved['first']}-01-31") & (frame.return_eom <= f"{saved['last']}-12-31")].sort_values('return_eom')
            g, n = v.gross_excess.tolist(), v.net_excess.tolist()
            mean_g = st.mean(g)
            views[method] = (12*(st.mean(n)-5*st.variance(g)), [x-5*(r-mean_g)**2 for r,x in zip(g,n)])
        delta = [a-b for a,b in zip(views['IND'][1],views['B1'][1])]
        z = st.mean(delta)/(st.stdev(delta)/math.sqrt(len(delta)))
        p = .5*math.erfc(z/math.sqrt(2))
        row = dict(first=saved['first'],last=saved['last'],months=len(delta),baseline=views['B1'][0],IND=views['IND'][0],z=z,p=p,author_normal_score=1-p)
        for k in ('baseline','IND','z','p'):
            require(math.isclose(row[k],saved[k],rel_tol=1e-11,abs_tol=1e-13), 'Statistical comparison: '+k)
        comparisons.append(row)
    require(read(STUDY/'statistical_verification.json')['passed'], 'Original independent R statistics')
    passed('Independent standard-library recomputation of utility and author normal test', comparisons=3, original_base_R_check=True)
    return dict(status='PASS',checked_at_utc=datetime.now(timezone.utc).isoformat(),study_complete_numerically=True,
        original_result_sha256=sha(STUDY/'result.json'),numerical_manifest_sha256=sha(STUDY/'numerical_completion_manifest.json'),
        lock_sha256=identity,checks=checks,comparisons=comparisons,
        scope='Read-only final audit of every saved model bundle and portfolio month; no new fitting or selection search. Original direct financial accounting and baseline reproduction receipts remain part of acceptance.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true')
    args = parser.parse_args()
    result = review()
    if args.write:
        OUT.mkdir(parents=True,exist_ok=True)
        path = OUT/'verification.json'
        require(not path.exists(), 'Refusing to overwrite final review; run without --write to verify again')
        path.write_text(json.dumps(result,indent=2)+'\n')
        pd.DataFrame(result['comparisons']).to_csv(OUT/'comparison.csv',index=False)
    print(json.dumps(result,indent=2))
