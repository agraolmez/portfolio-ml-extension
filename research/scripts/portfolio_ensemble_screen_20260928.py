"""Bounded saved-target screen; no optimization or model fitting.

All paths start from the same actual baseline holding. Immutable monthly
checkpoints retain the full economic packet for independent-language replay.
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid

for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from baseline.core import transition
from extension4.runtime import Runtime, now
from inference.core import stationary, basic_interval

CONFIG = ROOT / 'config/portfolio_ensemble_screen_20260928.json'
GEOMETRY = 'output/feature_geometry_study_20260914'
E4 = 'output/extension4_full_study_20260924'
RAW = 'data/prepared/step3_author_v3/monthly_inputs.parquet'
METHODS = ('B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'BENS', 'CENS', 'HALF')
MIX = np.array([[1/3, 1/3, 1/3, 0, 0, 0],
                [0, 0, 0, 1/3, 1/3, 1/3], [1/6]*6], dtype=float).T
CONTRASTS = (('CENS', 'BENS'), ('CENS', 'B1'), ('HALF', 'BENS'), ('HALF', 'B1'))
ATOL, RTOL = 1e-11, 1e-10


def sha(path, guard=lambda: None):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        if hasattr(fcntl, 'F_NOCACHE'):
            fcntl.fcntl(handle.fileno(), fcntl.F_NOCACHE, 1)
        for chunk in iter(lambda: handle.read(2**20), b''):
            value.update(chunk)
            guard()
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + '.partial_' + uuid.uuid4().hex)
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.rename(path)


def native_pressure():
    level = ctypes.c_int()
    size = ctypes.c_size_t(ctypes.sizeof(level))
    error = ctypes.CDLL(None, use_errno=True).sysctlbyname(
        b'kern.memorystatus_vm_pressure_level', ctypes.byref(level), ctypes.byref(size), None, 0)
    if error or level.value != 1:
        raise MemoryError(f'Native memory pressure is not normal: level={level.value}, error={error}')


class GuardedRuntime(Runtime):
    def sample(self):
        row = super().sample()
        native_pressure()
        return row


def validate_config(config):
    if (config['decision_dates'] != ['2015-12-31', '2025-11-30']
            or config['periods'] != {'primary': ['2016-01-31', '2020-12-31'],
                                     'robustness': ['2021-01-31', '2025-12-31']}
            or tuple(config['paths']) != METHODS or config['gamma'] != 10
            or config['training_authorized'] is not False
            or config['coefficient_search'] is not False):
        raise ValueError('The frozen no-training experiment has changed')
    np.testing.assert_array_equal(np.array([config['mixtures'][x] for x in METHODS[6:]]).T, MIX)
    inference = config['inference']
    if any(inference[k] != v for k, v in dict(draws=10000, expected_block_lengths=[12, 24],
                    master_seed=20260928, alpha=.05, family=8).items()):
        raise ValueError('Frozen inference settings differ')
    if (config['gate']['both_pass_priority'] != 'HALF'
            or config['gate']['automatic_exhaustive_training'] is not False):
        raise ValueError('Frozen advancement rule differs')


def verify_inventory(inventory, guard=lambda: None):
    for relative, expected in inventory.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or sha(path, guard) != expected:
            raise ValueError('Changed or unaccepted input: ' + relative)


def prepare(config, out, guard=lambda: None):
    """Bind audited inputs and code before seeing any new financial outcome."""
    validate_config(config)
    frozen = read(out / 'protocol_freeze.json')
    if (frozen['config_sha256'] != sha(CONFIG)
            or frozen['new_financial_performance_calculated'] is not False
            or frozen['new_models_fitted'] != 0):
        raise ValueError('Pre-outcome protocol freeze differs from current configuration')
    preflight = out / 'financial_preflight.json'
    if read(preflight) != read(ROOT / GEOMETRY / 'financial_preflight.json'):
        raise ValueError('Fresh financial preflight differs from accepted geometry parent')
    chronology = read(out / 'chronology.json')
    if chronology.get('passed') is not True:
        raise ValueError('Independent parent chronology audit must pass before the screen')
    if (out / 'lock.json').exists():
        lock = read(out / 'lock.json')
        if lock['config'] != config:
            raise ValueError('Configuration differs from frozen lock')
        verify_inventory(lock['artifact_sha256'], guard)
        return lock
    manifests = [GEOMETRY + '/acceptance_manifest.json', E4 + '/acceptance_manifest.json',
                 'output/step9_20260913/acceptance_manifest.json',
                 'output/step3_author_conventions_20260911/bundle_checksums.json']
    accepted = {}
    consumed = {}
    for relative in manifests:
        value = read(ROOT / relative)
        if 'acceptance_manifest' in relative and not all(g['passed'] for g in value['gates']):
            raise ValueError('Parent acceptance has failed gates')
        for relative_input, digest in value.get('artifact_sha256', value).items():
            if relative_input in accepted and accepted[relative_input] != digest:
                raise ValueError('Conflicting accepted input hashes: ' + relative_input)
            accepted[relative_input] = digest
        consumed[relative] = sha(ROOT / relative, guard)
    dates = pd.date_range(*config['decision_dates'], freq='ME')
    required = [RAW, GEOMETRY + '/financial_preflight.json', GEOMETRY + '/plan_lock.json',
                E4 + '/selection/selected.json',
                'output/extension3_full_study_20260919/selection_canonical_v1/temporal.json',
                'baseline/core.py', 'baseline/io.py', 'extension4/runtime.py',
                'extension4/storage.py', 'temporal_fast_v4/memory.py', 'inference/core.py',
                'GOAL.md', 'docs/STEP_1_THEORY.md', 'docs/RECONSTRUCTION_AMENDMENT.md',
                'config/protocol_v1.json', 'config/reconstruction_amendment.json']
    for date in dates:
        key = str(date.date())
        required += [f'{GEOMETRY}/paths/{key}.npz', f'{E4}/final_paths/{key}/positions.npz',
                     f'data/prepared/step9_v1/baseline/matrices/{key}.npz']
    for relative in required:
        canonical = str((ROOT / relative).resolve().relative_to(ROOT))
        expected = accepted.get(relative, accepted.get(canonical))
        if expected is None and relative.endswith('.py'):
            expected = accepted.get(E4 + '/lock/source_snapshot/' + relative)
        if expected is None:
            raise ValueError('No accepted hash for required input: ' + relative)
        consumed[relative] = expected
    for relative, digest in chronology.get('artifact_sha256', {}).items():
        if relative in consumed and consumed[relative] != digest:
            raise ValueError('Chronology and accounting input hashes conflict: ' + relative)
        consumed[relative] = digest
    for path in [CONFIG, Path(__file__).resolve(), ROOT / 'tests/test_portfolio_ensemble_screen.py',
                 preflight, out / 'chronology.json', out / 'protocol_freeze.json']:
        consumed[str(path.relative_to(ROOT))] = sha(path, guard)
    verify_inventory(consumed, guard)
    selected = read(ROOT / E4 / 'selection/selected.json')
    for seed in (1, 2, 3):
        rows = [x for x in selected if x['method'] == 'C4' and x['seed'] == seed]
        if len(rows) != 1 or rows[0]['available_at'] > config['decision_dates'][0]:
            raise ValueError('C4 selection was unavailable before first decision')
    parent_selection = read(ROOT / 'output/extension3_full_study_20260919/selection_canonical_v1/temporal.json')
    if (parent_selection['selection_available_at'] != parent_selection['plan']['selection_available_at']
            or parent_selection['selection_available_at'] > config['decision_dates'][0]):
        raise ValueError('Temporal parent selection unavailable before first decision')
    lock = dict(status='FROZEN_BEFORE_NEW_OUTCOMES', created_at_utc=now(), config=config,
                artifact_sha256=consumed, path_order=list(METHODS),
                arithmetic='Mixtures of targets with own holdings and charged first switch',
                targets_reestimated=False)
    save_json(out / 'lock.json', lock)
    return lock


def combine_targets(components):
    components = np.asarray(components, dtype=float)
    if components.ndim != 2 or components.shape[1] != 6 or not np.isfinite(components).all():
        raise ValueError('Six finite component targets required')
    return np.column_stack((components, components @ MIX))


def replay_packet(packet):
    """Pure monthly economic operation; targets do not depend on outcomes."""
    ids, next_ids = packet['ids'], packet['next_ids']
    targets, holdings = packet['targets'], packet['pretrade']
    if targets.shape != holdings.shape or targets.shape != (len(ids), len(METHODS)):
        raise ValueError('Nine aligned own-holdings paths required')
    arrays = {k: np.array(v, copy=True) for k, v in packet.items()}
    arrays.update(weights=np.empty_like(holdings), trades=np.empty_like(holdings),
                  drifted_weights=np.empty_like(holdings),
                  next_pretrade=np.empty((len(next_ids), len(METHODS))))
    rows = []
    for column, name in enumerate(METHODS):
        result = transition(ids, holdings[:, column], targets[:, column], packet['m'],
                            packet['total_returns'], float(packet['market_growth']),
                            packet['impact'], float(packet['wealth']),
                            packet['excess_returns'], next_ids)
        for key in ('weights', 'trades', 'drifted_weights', 'next_pretrade'):
            arrays[key][:, column] = result[key]
        rows.append(dict(method=name, gross_excess=result['gross_excess'],
                         impact_cost=result['impact_cost'],
                         net_excess=result['gross_excess'] - result['impact_cost'],
                         turnover=result['turnover'], gross_exposure=result['gross_exposure']))
    for key in ('pretrade', 'targets', 'weights', 'trades', 'drifted_weights', 'next_pretrade'):
        np.testing.assert_allclose(arrays[key][:, 6:], arrays[key][:, :6] @ MIX,
                                   atol=ATOL, rtol=RTOL, err_msg=key)
    return arrays, rows


def source_utility(gross, net, axis=0):
    return 12 * (np.mean(net, axis=axis) - 5*np.var(gross, axis=axis, ddof=1))


def convexity_identity(gross, costs):
    gross, costs = np.asarray(gross), np.asarray(costs)
    if gross.shape != costs.shape or gross.ndim != 2 or gross.shape[1] != 9 or len(gross) < 2:
        raise ValueError('At least two monthly observations of nine paths required')
    np.testing.assert_allclose(gross[:, 6:], gross[:, :6] @ MIX, atol=ATOL, rtol=RTOL)
    cost_gain = np.mean(costs[:, :6] @ MIX - costs[:, 6:], axis=0)
    risk_gain = np.var(gross[:, :6], axis=0, ddof=1) @ MIX - np.var(gross[:, 6:], axis=0, ddof=1)
    actual = source_utility(gross, gross-costs)[6:] - source_utility(gross, gross-costs)[:6] @ MIX
    expected = 12*(cost_gain + 5*risk_gain)
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)
    if min(cost_gain.min(), risk_gain.min(), actual.min()) < -ATOL:
        raise ValueError('Convexity guarantee failed')
    return [dict(method=METHODS[6+i], utility_gain=float(actual[i]),
                 annual_cost_saving=float(12*cost_gain[i]),
                 annual_risk_penalty_saving=float(60*risk_gain[i])) for i in range(3)]


def gate_decision(metrics, intervals):
    lookup = {(r['period'], r['method']): r['source_utility'] for r in metrics}
    decisions = {}
    for candidate in ('CENS', 'HALF'):
        point = all(lookup[(period, candidate)] > lookup[(period, control)]
                    for period in ('primary', 'robustness') for control in ('BENS', 'B1'))
        relevant = [r for r in intervals if r['period'] == 'primary' and r['candidate'] == candidate]
        expected = {(control, length) for control in ('BENS', 'B1') for length in (12, 24)}
        found = {(r['control'], r['block_months']) for r in relevant}
        if found != expected or len(relevant) != 4:
            raise ValueError('Missing or duplicate prespecified primary intervals')
        inference = all(r['lower'] > 0 for r in relevant)
        decisions[candidate] = dict(positive_vs_both_controls_both_periods=bool(point),
                                    adjusted_primary_lower_bounds_positive=bool(inference),
                                    passed=bool(point and inference))
    selected = next((x for x in ('HALF', 'CENS') if decisions[x]['passed']), None)
    return dict(candidates=decisions, candidate_for_larger_study_consideration=selected,
                exploratory_gate_passed=selected is not None, automatic_training_authorized=False,
                fresh_confirmation=False, published_paper_outperformance_established=False)


def verify_checkpoint(folder, lock_hash, previous, date, guard=lambda: None):
    commit = read(folder / 'commit.json')
    if (commit['lock_sha256'] != lock_hash or commit['previous_commit_sha256'] != previous
            or commit['decision_eom'] != date):
        raise ValueError('Checkpoint chain or protocol differs')
    for name, expected in commit['files_sha256'].items():
        if '/' in name or sha(folder / name, guard) != expected:
            raise ValueError('Checkpoint content differs: ' + name)
    return commit


def month_packet(date, ids, holdings):
    """Load frozen targets before accessing the month's realized outcome rows."""
    baseline_path = ROOT / GEOMETRY / f'paths/{date}.npz'
    c4_path = ROOT / E4 / f'final_paths/{date}/positions.npz'
    with np.load(baseline_path, allow_pickle=False) as base, np.load(c4_path, allow_pickle=False) as c4:
        np.testing.assert_array_equal(base['ids'], c4['ids'])
        np.testing.assert_array_equal(base['next_ids'], c4['next_ids'])
        columns = []
        base_indices = []
        for archive, method in ((base, 'fixed512'), (c4, 'C4')):
            for seed in (1, 2, 3):
                index = np.flatnonzero((archive['methods'] == method) & (archive['seeds'] == seed))
                if len(index) != 1:
                    raise ValueError('Missing/duplicate archived target')
                columns.append(archive['target'][:, index[0]])
                if method == 'fixed512':
                    base_indices.append(int(index[0]))
        targets = combine_targets(np.column_stack(columns))
        current_ids, next_ids = base['ids'].copy(), base['next_ids'].copy()
        if ids is None:
            ids = current_ids
            holdings = np.repeat(base['pretrade'][:, base_indices[0], None], 9, axis=1)
        np.testing.assert_array_equal(ids, current_ids)
        archive_neutral = {key: base[key][:, base_indices[0]].copy()
                           for key in ('pretrade', 'weight', 'trade', 'drifted_weight', 'next_pretrade')}
    with np.load(ROOT / f'data/prepared/step9_v1/baseline/matrices/{date}.npz', allow_pickle=False) as matrix:
        np.testing.assert_array_equal(ids, matrix['ids'])
        m = matrix['m'].copy()
    raw = pd.read_parquet(ROOT / RAW, columns=['id', 'eom', 'valid', 'lambda_diagonal', 'wealth_usd',
                         'tr_ld1', 'mu_ld1', 'ret_ld1'], filters=[('eom', '==', pd.Timestamp(date))])
    active = raw.loc[raw.valid].sort_values('id')
    np.testing.assert_array_equal(ids, active.id.to_numpy())
    if active[['wealth_usd', 'mu_ld1']].nunique().tolist() != [1, 1]:
        raise ValueError('Inconsistent wealth/market return within month')
    packet = dict(ids=ids, next_ids=next_ids, m=m, targets=targets, pretrade=holdings,
                  impact=active.lambda_diagonal.to_numpy(), wealth=np.array(active.wealth_usd.iloc[0]),
                  total_returns=active.tr_ld1.to_numpy(), market_growth=np.array(active.mu_ld1.iloc[0]),
                  excess_returns=active.ret_ld1.to_numpy(), methods=np.array(METHODS))
    return packet, archive_neutral


def run_paths(config, out, lock, runtime):
    dates = [str(d.date()) for d in pd.date_range(*config['decision_dates'], freq='ME')]
    directory = out / 'checkpoints'; directory.mkdir(exist_ok=True)
    previous, ids, holdings = None, None, None
    lock_hash = sha(out / 'lock.json')
    all_rows = []
    for count, date in enumerate(dates, 1):
        runtime.guard(); native_pressure()
        folder = directory / date
        if folder.exists():
            verify_checkpoint(folder, lock_hash, previous, date, runtime.guard)
            with np.load(folder / 'arrays.npz', allow_pickle=False) as z:
                if ids is not None:
                    np.testing.assert_array_equal(z['ids'], ids)
                    np.testing.assert_array_equal(z['pretrade'], holdings)
                else:
                    np.testing.assert_array_equal(z['pretrade'],
                        np.repeat(z['baseline_archive_pretrade'][:, None], 9, axis=1))
                ids, holdings = z['next_ids'].copy(), z['next_pretrade'].copy()
            rows = read(folder / 'monthly.json')
        else:
            packet, neutral = month_packet(date, ids, holdings)
            arrays, rows = replay_packet(packet)
            for actual, expected in [('pretrade', 'pretrade'), ('weights', 'weight'),
                                     ('trades', 'trade'), ('drifted_weights', 'drifted_weight'),
                                     ('next_pretrade', 'next_pretrade')]:
                np.testing.assert_allclose(arrays[actual][:, 0], neutral[expected], atol=ATOL, rtol=RTOL)
            arrays.update(baseline_archive_pretrade=neutral['pretrade'], baseline_archive_weights=neutral['weight'])
            for row in rows:
                row.update(decision_eom=date,
                           return_eom=str((pd.Timestamp(date)+pd.offsets.MonthEnd(1)).date()))
            temporary = Path(tempfile.mkdtemp(prefix='.' + date + '.partial_', dir=directory))
            with (temporary / 'arrays.npz').open('xb') as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush(); os.fsync(stream.fileno())
            save_json(temporary / 'monthly.json', rows)
            sources = [f'{GEOMETRY}/paths/{date}.npz', f'{E4}/final_paths/{date}/positions.npz',
                       f'data/prepared/step9_v1/baseline/matrices/{date}.npz', RAW]
            save_json(temporary / 'commit.json', dict(decision_eom=date, lock_sha256=lock_hash,
                previous_commit_sha256=previous,
                sources_sha256={p: lock['artifact_sha256'][p] for p in sources},
                files_sha256={p.name: sha(p) for p in temporary.iterdir() if p.is_file()}))
            runtime.guard(); native_pressure()
            temporary.rename(folder)
            ids, holdings = arrays['next_ids'], arrays['next_pretrade']
        previous = sha(folder / 'commit.json')
        all_rows.extend(rows)
        runtime.event('month_complete', months=count, total_months=len(dates), decision_eom=date)
    return pd.DataFrame(all_rows), previous


def analyze(config, frame, destination, guard=lambda: None):
    """Fixed comparisons only; never select blend strengths or bootstrap seeds."""
    destination.mkdir(exist_ok=False)
    frame.to_csv(destination / 'monthly.csv', index=False)
    metrics, intervals, convex = [], [], []
    boot = destination / 'bootstrap'; boot.mkdir()
    for period, endpoints in config['periods'].items():
        guard()
        expected_dates = pd.date_range(*endpoints, freq='ME')
        gross, costs = [], []
        for method in METHODS:
            sub = frame[frame.method.eq(method)].copy()
            sub['return_eom'] = pd.to_datetime(sub.return_eom)
            sub = sub[sub.return_eom.between(*endpoints)].sort_values('return_eom')
            if list(sub.return_eom) != list(expected_dates):
                raise ValueError('Missing or duplicate evaluation months')
            np.testing.assert_allclose(sub.net_excess, sub.gross_excess-sub.impact_cost, atol=ATOL, rtol=RTOL)
            gross.append(sub.gross_excess.to_numpy()); costs.append(sub.impact_cost.to_numpy())
        gross, costs = np.column_stack(gross), np.column_stack(costs)
        net = gross-costs; utility = source_utility(gross, net)
        for column, method in enumerate(METHODS):
            metrics.append(dict(period=period, method=method, months=len(gross),
                source_utility=float(utility[column]),
                net_population_utility=float(12*(net[:, column].mean()-5*net[:, column].var(ddof=0))),
                annual_gross_volatility=float(np.sqrt(12*gross[:, column].var(ddof=1))),
                annual_net_mean=float(12*net[:, column].mean()), annual_cost=float(12*costs[:, column].mean())))
        convex.extend(dict(period=period, **row) for row in convexity_identity(gross, costs))
        for length in config['inference']['expected_block_lengths']:
            guard()
            indices, _, _, _ = stationary('portfolio_ensemble_screen_20260928_' + period,
                config['inference']['master_seed'], length, endpoints[-1], len(gross),
                replicas=config['inference']['draws'])
            prefix = f'{period}_L{length}'
            np.savetxt(boot / (prefix + '_indices.csv'), indices+1, delimiter=',', fmt='%d')
            draws = np.empty((len(indices), len(CONTRASTS)))
            for start in range(0, len(indices), 250):
                guard()
                ix = indices[start:start+250]
                sample = source_utility(gross[ix], net[ix], axis=1)
                for column, (candidate, control) in enumerate(CONTRASTS):
                    draws[start:start+250, column] = sample[:, METHODS.index(candidate)]-sample[:, METHODS.index(control)]
            pd.DataFrame(draws, columns=[a+'_vs_'+b for a, b in CONTRASTS]).to_csv(
                boot / (prefix + '_draws.csv'), index=False)
            for column, (candidate, control) in enumerate(CONTRASTS):
                point = float(utility[METHODS.index(candidate)]-utility[METHODS.index(control)])
                lower, upper = basic_interval(point, draws[:, column],
                    alpha=config['inference']['alpha'], family=config['inference']['family'])
                intervals.append(dict(period=period, candidate=candidate, control=control,
                    block_months=length, observed=point, lower=lower, upper=upper,
                    family=config['inference']['family'], draws=len(indices)))
    pd.DataFrame(metrics).to_csv(destination / 'metrics.csv', index=False)
    save_json(destination / 'intervals.json', intervals)
    save_json(destination / 'convexity.json', convex)
    result = dict(status='COMPUTED_PENDING_INDEPENDENT_REVIEW',
                  gate=gate_decision(metrics, intervals), periods=config['periods'],
                  months=120, paths=list(METHODS), no_models_refitted=True,
                  independent_review_required=True)
    save_json(destination / 'result.json', result)
    return result


def run(config, out):
    if config.get('execution_authorized') is not True:
        raise ValueError('Bounded screen execution is not authorized')
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'execution.lock').open('a') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / 'execution_complete.json').exists():
            complete = read(out / 'execution_complete.json')
            verify_inventory(complete['artifact_sha256'])
            print('Completed screen verified without rewriting.'); return
        native_pressure()
        runtime = GuardedRuntime(out, config, 'saved_target_ensemble_screen')
        try:
            lock = prepare(config, out, runtime.guard)
            runtime.event('lock_verified', config_sha256=sha(CONFIG), no_fitting=True)
            frame, final_checkpoint = run_paths(config, out, lock, runtime)
            analysis = out / 'analysis'
            if analysis.exists():
                commit = read(analysis / 'commit.json')
                if commit['final_checkpoint_sha256'] != final_checkpoint or commit['lock_sha256'] != sha(out / 'lock.json'):
                    raise ValueError('Analysis inputs differ from committed inputs')
                verify_inventory(commit['artifact_sha256'], runtime.guard)
            else:
                temporary = out / ('.analysis.partial_' + uuid.uuid4().hex)
                analyze(config, frame, temporary, runtime.guard)
                hashes = {str((analysis / p.relative_to(temporary)).relative_to(ROOT)): sha(p)
                          for p in temporary.rglob('*') if p.is_file()}
                save_json(temporary / 'commit.json', dict(lock_sha256=sha(out / 'lock.json'),
                    final_checkpoint_sha256=final_checkpoint, artifact_sha256=hashes))
                runtime.guard(); native_pressure(); temporary.rename(analysis)
            for name in ('monthly.csv', 'metrics.csv', 'intervals.json', 'convexity.json', 'result.json'):
                target = out / name
                if target.exists():
                    if sha(target) != sha(analysis / name):
                        raise ValueError('Existing public summary differs')
                else:
                    with target.open('xb') as handle:
                        handle.write((analysis / name).read_bytes())
                        handle.flush(); os.fsync(handle.fileno())
            verify_inventory(lock['artifact_sha256'], runtime.guard)
            runtime.guard(); native_pressure()
            artifacts = [out / 'lock.json', *list((out / 'checkpoints').glob('*/commit.json')),
                         *list((out / 'checkpoints').glob('*/arrays.npz')),
                         *list((out / 'checkpoints').glob('*/monthly.json')),
                         *[p for p in analysis.rglob('*') if p.is_file()],
                         *[out/name for name in ('monthly.csv','metrics.csv','intervals.json','convexity.json','result.json')]]
            save_json(out / 'execution_complete.json', dict(status='COMPUTED_PENDING_INDEPENDENT_REVIEW',
                artifact_sha256={str(p.relative_to(ROOT)): sha(p, runtime.guard) for p in artifacts}))
            runtime.event('screen_computed_pending_review', **read(out / 'result.json')['gate'])
            runtime.close('COMPUTED_PENDING_INDEPENDENT_REVIEW')
        except BaseException as error:
            runtime.close('STOPPED', error)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', help='Bind inputs/code without financial calculations')
    parser.add_argument('--run', action='store_true', help='Execute only the frozen bounded screen')
    args = parser.parse_args()
    if args.prepare == args.run:
        parser.error('Choose exactly one of --prepare or --run')
    config = read(CONFIG); validate_config(config)
    out = ROOT / config['output_root']
    if out.resolve() != ROOT / 'output/portfolio_ensemble_screen_20260928':
        raise ValueError('Unexpected screen output directory')
    if args.prepare:
        out.mkdir(parents=True, exist_ok=True)
        prepare(config, out)
        print('Input/code lock prepared; no new financial path calculated.')
    else:
        run(config, out)


if __name__ == '__main__':
    main()
