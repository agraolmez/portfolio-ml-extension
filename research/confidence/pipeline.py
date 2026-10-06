"""Versioned, resumable Step 7 build; accepted baseline paths are read-only."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from baseline.io import ROOT, DATA, PROTOCOL, sha, save_json, datekey
from baseline.full import PREP as BASE, OUT as BASE_OUT, RUN, DATES, PROTECTED, preflight as baseline_preflight
from .core import replica_seed, moving_blocks, fit_coefficients, state_features

OUT = ROOT / 'output/step7_20260912'
PREP = ROOT / 'data/prepared/step7_v1'
BLOCKS = [PROTOCOL['aim_resampling']['block_months'], PROTOCOL['aim_resampling']['sensitivity_block_months']]
REPLICAS = PROTOCOL['aim_resampling']['replicas']
CODE = ['confidence/__init__.py', 'confidence/core.py', 'confidence/pipeline.py', 'docs/STEP_7_PLAN.md',
        'scripts/step7_measure.py', 'scripts/step7_synthetic.py', 'scripts/step7_reference.R']


def immutable():
    if (OUT / 'acceptance_manifest.json').exists():
        raise ValueError('Step 7 is accepted; create a separately versioned study')


def annual_models():
    selections = pd.read_parquet(BASE / 'annual_selection.parquet')
    records = []
    for year in range(1981, 2021):
        rows = selections[selections.return_eom.dt.year.eq(year - 1)]
        if len(rows) != 1:
            raise ValueError('Prior December selection must be unique')
        row = rows.iloc[0]
        g, p, li = int(row.g_index), int(row.p), int(row.lambda_index)
        path = BASE / f'coefficients/g{g}/{year}_p{p}.npz'
        with np.load(path, allow_pickle=False) as z:
            if str(z['training_cutoff']) != f'{year-1}-11-30' or z['lambdas'][li] != row['lambda']:
                raise ValueError('Incorrect coefficient cutoff or changed selected penalty')
            n = int(z['training_months'])
        records.append(dict(fit_year=year, g_index=g, p=p, lambda_index=li, penalty=float(row['lambda']),
            training_months=n, training_start='1965-11-30', training_cutoff=f'{year-1}-11-30',
            latest_training_return=f'{year-1}-12-31', selection_available_at=f'{year-1}-12-31',
            first_decision=f'{year-1}-12-31', last_decision=f'{year}-11-30',
            central_coefficients=str(path.relative_to(ROOT)), central_coefficients_sha256=sha(path)))
    return records


def initialize():
    immutable()
    if (OUT / 'start.json').exists():
        raise ValueError('Step 7 already initialized')
    receipt, _ = baseline_preflight()
    accepted_path = BASE_OUT / 'acceptance_manifest.json'
    accepted = json.loads(accepted_path.read_text())
    if accepted.get('status') != 'ACCEPTED_FULL_HISTORICAL_BASELINE' or not all(g['passed'] for g in accepted['gates']):
        raise ValueError('Accepted full baseline required')
    for rel, digest in accepted['artifact_sha256'].items():
        if sha(ROOT / rel) != digest:
            raise ValueError(f'Changed accepted Step 6 file: {rel}')
    models = annual_models()
    sources = {accepted_path, DATA / 'monthly_inputs.parquet', RUN / 'PolicyDecision.parquet', BASE / 'baseline_aims.parquet', BASE / 'annual_selection.parquet'}
    for gi in [0, 1]:
        sources.update(BASE_OUT / f'reference/fit/g{gi}/{name}' for name in ['denom.bin', 'rtilde.bin', 'dates.json'])
        sources.add(BASE_OUT / f'reference/rff_g{gi}.bin')
    for row in models:
        sources.add(ROOT / row['central_coefficients'])
        for d in pd.date_range(row['first_decision'], row['last_decision'], freq='ME'):
            sources.update([BASE / f'moments/g{row["g_index"]}/{datekey(d)}.npz',
                            BASE / f'matrices/{datekey(d)}.npz', BASE / f'risk/{datekey(d)}.npz'])
    OUT.mkdir(parents=True, exist_ok=True)
    PREP.mkdir(parents=True, exist_ok=True)
    save_json(OUT / 'annual_models.json', models)
    save_json(OUT / 'start.json', dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
        authorization="User: Let's start and finish Step 7", baseline_preflight=receipt,
        protected={p: sha(ROOT / p) for p in PROTECTED}, code_sha256={p: sha(ROOT / p) for p in CODE},
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted(sources)},
        annual_models_sha256=sha(OUT / 'annual_models.json'), protocol_id=PROTOCOL['protocol_id'],
        expected_coefficient_fits=40 * len(BLOCKS) * REPLICAS, expected_monthly_records=len(DATES) * len(BLOCKS),
        baseline_changed=False, speed_learner_run=False,
        scope='Conditional sensitivity on the accepted R1 reconstruction; original-vintage and causal-execution limits retained'))
    print('Step 7 source, specification and machinery lock created after full-baseline preflight.', flush=True)


def preflight():
    start = json.loads((OUT / 'start.json').read_text())
    for section in ['protected', 'code_sha256', 'source_sha256']:
        for rel, digest in start[section].items():
            if sha(ROOT / rel) != digest:
                raise ValueError(f'Changed Step 7 dependency: {rel}')
    if sha(OUT / 'annual_models.json') != start['annual_models_sha256']:
        raise ValueError('Changed annual model specification')
    return start


def input_moments(g):
    folder = BASE_OUT / f'reference/fit/g{g}'
    dates = pd.DatetimeIndex(json.loads((folder / 'dates.json').read_text()))
    if not dates.equals(pd.date_range('1965-11-30', '2020-11-30', freq='ME')):
        raise ValueError('Accepted binary moment calendar changed')
    a = np.memmap(folder / 'denom.bin', dtype='<f8', mode='r', shape=(len(dates), 513 * 513))
    r = np.memmap(folder / 'rtilde.bin', dtype='<f8', mode='r', shape=(len(dates), 513))
    return dates, a, r


def save_unit(path, arrays, metadata, lock_hash):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_suffix('.json').exists():
        raise ValueError('Refuse to overwrite a saved unit')
    np.savez_compressed(path, **arrays)
    save_json(path.with_suffix('.json'), dict(**metadata, start_sha256=lock_hash, array_sha256=sha(path)))


def reuse_unit(path, lock_hash):
    if not path.exists() and not path.with_suffix('.json').exists():
        return False
    if not path.exists() or not path.with_suffix('.json').exists():
        raise ValueError('Incomplete checkpoint requires explicit recovery; no silent overwrite')
    meta = json.loads(path.with_suffix('.json').read_text())
    if meta['start_sha256'] != lock_hash or meta['array_sha256'] != sha(path):
        raise ValueError('Stale unit checkpoint')
    return True


def fit():
    immutable(); preflight()
    if (OUT / 'fit_manifest.json').exists():
        raise ValueError('Coefficient stage already complete')
    models = json.loads((OUT / 'annual_models.json').read_text())
    lock_hash = sha(OUT / 'start.json')
    maps = {g: input_moments(g) for g in [0, 1]}
    all_meta, seed_rows = [], []
    for row in models:
        year, g, n = row['fit_year'], row['g_index'], row['training_months']
        dates, flat, scores = maps[g]
        if not dates[n - 1] == pd.Timestamp(row['training_cutoff']) or dates[n] != pd.Timestamp(row['first_decision']):
            raise ValueError('Resampling calendar extends beyond the annual estimation cutoff')
        with np.load(ROOT / row['central_coefficients']) as z:
            idx, central = z['indices'], z['coefficients'][:, row['lambda_index']]
        # The accepted selections all have p=512. Do not silently discard a changed design.
        if row['p'] != 512 or not np.array_equal(idx, np.arange(513)):
            raise ValueError('Selected feature representation differs from the locked Step 6 baseline')
        unweighted, _ = fit_coefficients(np.asarray(flat[:n]).mean(axis=0).reshape(513, 513),
                                          np.asarray(scores[:n]).mean(axis=0), row['penalty'])
        central_error = float(np.max(np.abs(unweighted - central)))
        if not np.allclose(unweighted, central, atol=1e-12, rtol=1e-9):
            raise ValueError('Uniform-weight refit does not match original central coefficients')
        for length in BLOCKS:
            path = PREP / f'coefficients/block{length}/{year}.npz'
            for b in range(1, REPLICAS + 1):
                seed, digest = replica_seed(PROTOCOL, length, row['first_decision'], b)
                seed_rows.append(dict(fit_year=year, block_months=length, replica=b, seed_hex=f'{seed:032x}', sha256=digest))
            if reuse_unit(path, lock_hash):
                # Resume after an interruption between the checkpoint and its reference export.
                ref = OUT / f'reference/block{length}'; ref.mkdir(parents=True, exist_ok=True)
                with np.load(path) as saved:
                    for name, values, fmt in [('counts', saved['counts'][[0,49,99]], '%d'),
                                             ('coefficients', saved['coefficients'][:,[0,49,99]], '%.17g')]:
                        dest = ref / f'{year}_{name}.csv'
                        if not dest.exists(): np.savetxt(dest, values, delimiter=',', fmt=fmt)
                all_meta.append(json.loads(path.with_suffix('.json').read_text())); continue
            starts, samples, counts = [], [], []
            for b in range(1, REPLICAS + 1):
                seed, _ = replica_seed(PROTOCOL, length, row['first_decision'], b)
                st, ix, ct = moving_blocks(n, length, seed)
                starts.append(st); samples.append(ix); counts.append(ct)
            counts = np.asarray(counts, dtype=np.int16)
            if not np.all(counts.sum(axis=1) == n):
                raise ValueError('Moving-block counts change estimation length')
            coefficients = np.empty((513, REPLICAS)); residuals = []
            for lo in range(0, REPLICAS, 10):
                weighted_a = (counts[lo:lo+10].astype(float) @ flat[:n]) / n
                weighted_r = (counts[lo:lo+10].astype(float) @ scores[:n]) / n
                for j in range(len(weighted_r)):
                    coef, residual = fit_coefficients(weighted_a[j].reshape(513, 513), weighted_r[j], row['penalty'])
                    coefficients[:, lo+j] = coef; residuals.append(residual)
            metadata = dict(**row, block_months=length, replicas=REPLICAS,
                uniform_central_max_abs_error=central_error, maximum_relative_solve_residual=max(residuals),
                minimum_unique_months=int((counts > 0).sum(axis=1).min()),
                maximum_unique_months=int((counts > 0).sum(axis=1).max()),
                moment_denom_sha256=sha(BASE_OUT / f'reference/fit/g{g}/denom.bin'))
            save_unit(path, dict(coefficients=coefficients, indices=idx, starts=np.asarray(starts, dtype=np.int16),
                sampled_indices=np.asarray(samples, dtype=np.int16), counts=counts,
                training_dates=dates[:n].strftime('%Y-%m-%d').to_numpy(dtype='U10'), residuals=np.asarray(residuals)), metadata, lock_hash)
            all_meta.append(json.loads(path.with_suffix('.json').read_text()))
            # Small, deliberately sampled independent R refit inputs; source moments are reused in place.
            ref = OUT / f'reference/block{length}'; ref.mkdir(parents=True, exist_ok=True)
            np.savetxt(ref / f'{year}_counts.csv', counts[[0, 49, 99]], delimiter=',', fmt='%d')
            np.savetxt(ref / f'{year}_coefficients.csv', coefficients[:, [0, 49, 99]], delimiter=',', fmt='%.17g')
            print(f'Conditional coefficients: {year}, {length}-month blocks, 100/100 fits.', flush=True)
    pd.DataFrame(seed_rows).to_parquet(PREP / 'replica_seeds.parquet', index=False)
    pd.DataFrame(all_meta).to_parquet(PREP / 'annual_refit_records.parquet', index=False)
    files = [p for p in (PREP / 'coefficients').rglob('*') if p.is_file()]
    files += [PREP / 'replica_seeds.parquet', PREP / 'annual_refit_records.parquet']
    save_json(OUT / 'fit_manifest.json', dict(status='ALL_CONDITIONAL_FITS_COMPLETE', annual_variants=len(all_meta),
        coefficient_fits=len(all_meta)*REPLICAS, maximum_relative_solve_residual=max(x['maximum_relative_solve_residual'] for x in all_meta),
        uniform_central_max_abs_error=max(x['uniform_central_max_abs_error'] for x in all_meta),
        artifact_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}))


def covariance(d):
    with np.load(BASE / f'risk/{datekey(d)}.npz') as z:
        return z['ids'], z['loadings'] @ z['factor_cov'] @ z['loadings'].T + np.diag(z['specific_variance'])


def features():
    immutable(); preflight()
    if (OUT / 'feature_manifest.json').exists():
        raise ValueError('Monthly feature stage already complete')
    fits = json.loads((OUT / 'fit_manifest.json').read_text())
    for rel, digest in fits['artifact_sha256'].items():
        if sha(ROOT / rel) != digest: raise ValueError('Changed coefficient artifact')
    positions = pd.read_parquet(RUN / 'PolicyDecision.parquet')
    inputs = pd.read_parquet(DATA / 'monthly_inputs.parquet', columns=['id','eom','lambda_diagonal','wealth_usd'],
        filters=[('eom','>=',DATES[0]), ('eom','<=',DATES[-1])])
    grouped = {d: f.set_index('id') for d, f in inputs.groupby('eom')}
    models = {r['fit_year']: r for r in json.loads((OUT / 'annual_models.json').read_text())}
    lock_hash, rows, sources = sha(OUT / 'start.json'), [], []
    for d, state in positions.groupby('eom', sort=True):
        row = models[(d + pd.offsets.MonthEnd(1)).year]; year, g = row['fit_year'], row['g_index']
        with np.load(BASE / f'moments/g{g}/{datekey(d)}.npz') as z:
            ids, signals = z['ids'], z['signal_t']
        state = state.set_index('id').reindex(ids)
        if state.pretrade.isna().any(): raise ValueError('Missing R1 state')
        risk_ids, sigma = covariance(d)
        with np.load(BASE / f'matrices/{datekey(d)}.npz') as z:
            if not np.array_equal(ids, z['ids']): raise ValueError('Matrix IDs differ')
            matrix = z['m']
        if not np.array_equal(ids, risk_ids): raise ValueError('Covariance IDs differ')
        frame = grouped[d].reindex(ids)
        if frame.isna().any().any() or frame.wealth_usd.nunique() != 1: raise ValueError('Missing state inputs')
        with np.load(ROOT / row['central_coefficients']) as z:
            central = signals[:, z['indices']] @ z['coefficients'][:, row['lambda_index']]
        if not np.allclose(central, state.aim, atol=1e-12, rtol=1e-9): raise ValueError('Central aim changed')
        for length in BLOCKS:
            path = PREP / f'aims/block{length}/{datekey(d)}.npz'
            coef_path = PREP / f'coefficients/block{length}/{year}.npz'
            if reuse_unit(path, lock_hash):
                meta = json.loads(path.with_suffix('.json').read_text())
                rows.append(meta['features']); sources.append(meta['sources']); continue
            with np.load(coef_path) as z:
                replicas = signals[:, z['indices']] @ z['coefficients']
            vals, q0, qbar = state_features(central, replicas, state.pretrade.to_numpy(), matrix, sigma,
                frame.lambda_diagonal.to_numpy(), frame.wealth_usd.iloc[0], gamma=PROTOCOL['economic']['risk_aversion'])
            if not np.allclose(q0, state.trade, atol=1e-12, rtol=1e-9): raise ValueError('Central baseline trade changed')
            vals.update(decision_eom=datekey(d), return_eom=datekey(d + pd.offsets.MonthEnd(1)),
                        fit_year=year, block_months=length, state_id='R1_continuous', securities=len(ids), replicas=REPLICAS,
                        latest_training_return=row['latest_training_return'], selection_available_at=row['selection_available_at'])
            lineage = dict(decision_eom=datekey(d), block_months=length, coefficients=str(coef_path.relative_to(ROOT)),
                coefficient_sha256=sha(coef_path), current_signals=f'data/prepared/step6_v1/moments/g{g}/{datekey(d)}.npz',
                current_signals_sha256=sha(BASE / f'moments/g{g}/{datekey(d)}.npz'),
                risk_sha256=sha(BASE / f'risk/{datekey(d)}.npz'), matrix_sha256=sha(BASE / f'matrices/{datekey(d)}.npz'),
                feature_draw_sha256=sha(BASE_OUT / f'reference/rff_g{g}.bin'),
                baseline_state_sha256=sha(RUN / 'PolicyDecision.parquet'),
                central_coefficients_sha256=row['central_coefficients_sha256'])
            save_unit(path, dict(ids=ids, replica_aims=replicas, central_aim=central,
                                mean_aim=replicas.mean(axis=1), baseline_mean_trade=qbar),
                      dict(features=vals, sources=lineage), lock_hash)
            rows.append(vals); sources.append(lineage)
        if d.month == 11 or d == DATES[0]:
            print(f'Monthly confidence: through {datekey(d)}, both fixed block lengths.', flush=True)
    table = pd.DataFrame(rows)
    for col in ['decision_eom','return_eom','latest_training_return','selection_available_at']:
        table[col] = pd.to_datetime(table[col])
    table.to_parquet(PREP / 'monthly_confidence.parquet', index=False)
    pd.DataFrame(sources).to_parquet(PREP / 'AimArtifact.parquet', index=False)
    table.to_csv(OUT / 'monthly_confidence.csv', index=False)
    files = [p for p in (PREP / 'aims').rglob('*') if p.is_file()]
    files += [PREP / 'monthly_confidence.parquet', PREP / 'AimArtifact.parquet', OUT / 'monthly_confidence.csv']
    save_json(OUT / 'feature_manifest.json', dict(status='MONTHLY_CONFIDENCE_COMPUTED_PENDING_REVIEW',
        rows=len(table), months=len(DATES), blocks=BLOCKS, monthly_replica_aim_values=sum(x['securities']*REPLICAS for x in rows),
        baseline_aim_unchanged=True, state_dependent=True, speed_learner_run=False,
        artifact_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['initialize','preflight','fit','features'])
    args = parser.parse_args()
    result = globals()[args.stage]()
    if args.stage == 'preflight': print('Step 7 source, code and protected-file preflight PASS.')
